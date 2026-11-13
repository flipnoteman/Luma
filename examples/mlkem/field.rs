//! Arithmetic modulo q = 3329 on `u32` arrays, the number-theoretic transform and the products of
//! the NTT domain, all as lazy array operations on polynomials stored as `[rows, 256]` arrays
//! (coefficient `k` of polynomial `r` is element `r + rows * k`: coefficients of different
//! polynomials are neighbours in memory, so a thread per element reads coalesced).

use luma::*;

pub const Q: u32 = 3329;
/// `-q^-1 mod 2^16`: Montgomery reduction with R = 2^16.
const QINV_NEG: u32 = 3327;
/// R mod q and R^2 mod q for R = 2^16.
const R1: u32 = 65536 % Q;
const R2: u32 = ((R1 as u64 * R1 as u64) % Q as u64) as u32;
/// 128^-1 mod q.
const INV128: u32 = 3303;
/// The primitive 256-th root of unity of FIPS 203.
const ZETA: u32 = 17;

fn pow_mod(mut base: u64, mut exp: u32) -> u32 {
    let (q, mut acc) = (Q as u64, 1u64);
    base %= q;
    while exp > 0 {
        if exp & 1 == 1 {
            acc = acc * base % q;
        }
        base = base * base % q;
        exp >>= 1;
    }
    acc as u32
}

/// `i` with its 7 bits reversed.
pub fn brv7(i: u32) -> u32 {
    (0..7).fold(0, |acc, b| acc | (((i >> b) & 1) << (6 - b)))
}

/// `x * 2^16 mod q`.
fn to_mont(x: u32) -> u32 {
    ((x as u64 * 65536) % Q as u64) as u32
}

/// The constants of the transforms on the device, and how many layers one kernel computes.
pub struct Tables {
    /// `zetas[i] = 17^brv7(i)` in Montgomery form, for `i` in `0..128`.
    zetas: Array<u32>,
    /// `17^(2 brv7(i) + 1)` in Montgomery form, for `i` in `0..128`: the `gamma` of the base cases.
    gammas: Array<u32>,
    /// Layers of a transform computed between two evaluations of the array (the layers in between
    /// are one kernel that reads each element through the layers before it).
    pub fuse: usize,
}

impl Tables {
    pub fn new(ctx: &Context, fuse: usize) -> Result<Tables> {
        let zetas: Vec<u32> = (0..128).map(|i| to_mont(pow_mod(ZETA as u64, brv7(i)))).collect();
        let gammas: Vec<u32> = (0..128).map(|i| to_mont(pow_mod(ZETA as u64, 2 * brv7(i) + 1))).collect();
        Ok(Tables { zetas: Array::from_slice_on(ctx, &zetas, 128)?, gammas: Array::from_slice_on(ctx, &gammas, 128)?, fuse })
    }
}

/// `p / 2^16 mod q` for `p < 2^27`, in `0..q`.
pub fn mont(p: &Array<u32>) -> Array<u32> {
    let m = (p * QINV_NEG) & 0xFFFFu32;
    let t = (p + m * Q) >> 16u32;
    t.minimum(&(&t - Q))
}

/// `a + b mod q` for `a, b < q`.
pub fn add_mod(a: &Array<u32>, b: &Array<u32>) -> Array<u32> {
    let s = a + b;
    s.minimum(&(&s - Q))
}

/// `a - b mod q` for `a, b < q`.
pub fn sub_mod(a: &Array<u32>, b: &Array<u32>) -> Array<u32> {
    let d = (a + Q) - b;
    d.minimum(&(&d - Q))
}

/// The row and coefficient index of every element of a `[rows, 256]` array.
fn grid(ctx: &Context, rows: usize) -> Result<(Array<u32>, Array<u32>)> {
    Ok((Array::range_on(ctx, [rows, 256], 0)?, Array::range_on(ctx, [rows, 256], 1)?))
}

/// The element `kappa ^ flip` of the same row, for every element.
fn partner(x: &Array<u32>, row: &Array<u32>, kappa: &Array<u32>, rows: usize, flip: u32) -> Array<u32> {
    x.take(&(row + (kappa ^ flip) * rows as u32))
}

fn rows_of(x: &Array<u32>) -> usize {
    let dims = x.shape().dims();
    assert!(dims[1] == 256 && dims[2] == 1, "a polynomial array is [rows, 256], got {}", x.shape());
    dims[0]
}

/// FIPS 203 Algorithm 9 on every row: the coefficients in, the NTT out.
pub fn ntt(t: &Tables, x: &Array<u32>) -> Array<u32> {
    let rows = rows_of(x);
    let (row, kappa) = grid(x.context(), rows).expect("index arrays");
    let mut x = x.clone();
    for (layer, j) in (1..=7u32).rev().enumerate() {
        let len = 1u32 << j;
        let other = partner(&x, &row, &kappa, rows, len);
        let is_hi = (&kappa >> j) & 1u32;
        let hi = select(&is_hi, &x, &other);
        let lo = select(&is_hi, &other, &x);
        let zeta = t.zetas.take(&((&kappa >> (j + 1)) + (128 / len)));
        let product = mont(&(&hi * &zeta));
        // lo + t for the low element, lo - t for the high one: lo + (q - t) reduced.
        let signed = select(&is_hi, &(Q - &product), &product);
        let sum = &lo + &signed;
        x = sum.minimum(&(&sum - Q));
        if (layer + 1) % t.fuse == 0 && layer + 1 < 7 {
            x = x.cache();
        }
    }
    x
}

/// FIPS 203 Algorithm 10 on every row (including the final multiplication by 128^-1).
pub fn inv_ntt(t: &Tables, x: &Array<u32>) -> Array<u32> {
    let rows = rows_of(x);
    let (row, kappa) = grid(x.context(), rows).expect("index arrays");
    let mut x = x.clone();
    for (layer, j) in (1..=7u32).enumerate() {
        let len = 1u32 << j;
        let other = partner(&x, &row, &kappa, rows, len);
        let is_hi = (&kappa >> j) & 1u32;
        let hi = select(&is_hi, &x, &other);
        let lo = select(&is_hi, &other, &x);
        // The zeta of a block: k counts down from 127.
        let zeta = t.zetas.take(&((256 / len - 1) - (&kappa >> (j + 1))));
        let sum = add_mod(&lo, &hi);
        let difference = mont(&(&sub_mod(&hi, &lo) * &zeta));
        x = select(&is_hi, &difference, &sum);
        if (layer + 1) % t.fuse == 0 {
            x = x.cache();
        }
    }
    mont(&(&x * to_mont(INV128)))
}

/// A term of an inner product in the NTT domain: the two polynomials' elements, and the element
/// next to each (index `kappa ^ 1`).
pub struct Term {
    pub f: Array<u32>,
    pub f_other: Array<u32>,
    pub g: Array<u32>,
    pub g_other: Array<u32>,
}

/// `sum over terms of f * g` in the ring of the NTT domain (FIPS 203 Algorithms 11 and 12): the
/// element `kappa` of a product of two coefficient pairs `(f0, f1)`, `(g0, g1)` is `f0 g0 + f1 g1 gamma`
/// for an even `kappa` and `f0 g1 + f1 g0` for an odd one. `kappa` is the coefficient index of every
/// element of the terms' (equal) shape.
pub fn base_multiply_sum(t: &Tables, kappa: &Array<u32>, terms: &[Term]) -> Array<u32> {
    let odd = kappa & 1u32;
    let gamma = t.gammas.take(&(kappa >> 1u32));
    let mut acc: Option<Array<u32>> = None;
    for term in terms {
        // The factor of `f1` in the product: g1 gamma for an even element, g0 (the other one) for an odd one.
        let scaled = select(&odd, &term.g_other, &mont(&(&term.g_other * &gamma)));
        let first = select(&odd, &term.f_other, &term.f);
        let second = select(&odd, &term.f, &term.f_other);
        let product = &first * &term.g + &second * &scaled;
        acc = Some(match acc {
            Some(sum) => sum + product,
            None => product,
        });
    }
    // The sum is below 2^27; two Montgomery reductions leave R^-2 * R^2 = 1.
    mont(&(&mont(&acc.expect("at least one term")) * R2))
}
