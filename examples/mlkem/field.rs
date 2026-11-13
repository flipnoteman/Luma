//! Arithmetic modulo q = 3329 on `u32` arrays, the number-theoretic transform and the products of
//! the NTT domain, all as lazy array operations on polynomials stored as `[256, rows]` arrays
//! (coefficient `k` of polynomial `r` is element `k + 256 r`: the 256 coefficients of a polynomial
//! are contiguous, like a byte string).

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

/// The constants of the transforms, on the host (the layers that act on a whole coordinate of a
/// polynomial take their zetas as scalars) and on the device (the ones that need one per element).
pub struct Tables {
    /// `zetas[i] = 17^brv7(i)` in Montgomery form, for `i` in `0..128`.
    zeta_mont: [u32; 128],
    zetas: Array<u32>,
    /// `17^(2 brv7(i) + 1)` in Montgomery form, for `i` in `0..128`: the `gamma` of the base cases.
    gammas: Array<u32>,
}

impl Tables {
    pub fn new(ctx: &Context) -> Result<Tables> {
        let zeta_mont: [u32; 128] = std::array::from_fn(|i| to_mont(pow_mod(ZETA as u64, brv7(i as u32))));
        let gammas: Vec<u32> = (0..128).map(|i| to_mont(pow_mod(ZETA as u64, 2 * brv7(i) + 1))).collect();
        Ok(Tables { zeta_mont, zetas: Array::from_slice_on(ctx, &zeta_mont, 128)?, gammas: Array::from_slice_on(ctx, &gammas, 128)? })
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

fn rows_of(x: &Array<u32>) -> usize {
    let dims = x.shape().dims();
    assert!(dims[0] == 256 && dims[2] == 1, "a polynomial array is [256, rows], got {}", x.shape());
    dims[1]
}

/// The coordinate `axis` of every element of an array of `shape`.
fn coordinate(ctx: &Context, shape: [usize; 4], axis: usize) -> Array<u32> {
    Array::range_on(ctx, shape, axis).expect("index array")
}

/// The sides of a butterfly `(lo, hi)` -> `(lo + t, lo - t)`, `t = zeta * hi`.
fn cooley_tukey(lo: &Array<u32>, hi: &Array<u32>, zeta: impl Fn(&Array<u32>) -> Array<u32>) -> (Array<u32>, Array<u32>) {
    let t = mont(&zeta(hi));
    (add_mod(lo, &t), sub_mod(lo, &t))
}

/// The sides of a butterfly `(lo, hi)` -> `(lo + hi, zeta * (hi - lo))`.
fn gentleman_sande(lo: &Array<u32>, hi: &Array<u32>, zeta: impl Fn(&Array<u32>) -> Array<u32>) -> (Array<u32>, Array<u32>) {
    (add_mod(lo, hi), mont(&zeta(&sub_mod(hi, lo))))
}

/// Joins `parts` (arrays of one shape) along `axis` into one evaluated array (a balanced tree of two-part joins).
fn join(axis: usize, parts: &[Array<u32>]) -> Array<u32> {
    match parts {
        [one] => one.clone(),
        [a, b] => concat(axis, [a, b]),
        _ => {
            let (a, b) = parts.split_at(parts.len() / 2);
            concat(axis, [&join(axis, a), &join(axis, b)])
        }
    }
}

fn evaluate(arrays: &[Array<u32>]) {
    eval_all(&arrays.iter().collect::<Vec<_>>()).expect("evaluation");
}

/// The 7 layers of the NTT split in the two groups that a thread can do on registers: the four
/// layers whose butterflies pair coefficients `16` or more apart (sixteen slices of the
/// polynomial, one per `kappa / 16`, with the zetas as constants), and the three that pair
/// coefficients 8, 4 and 2 apart (eight slices of the 16 coefficients of a block by `kappa / 2 mod
/// 8`, the zetas looked up by block). A group is one kernel without recomputation: its results are
/// separate arrays, joined into one afterwards.
const BLOCKS: usize = 16;

/// FIPS 203 Algorithm 9 on every row: the coefficients in, the NTT out.
pub fn ntt(t: &Tables, x: &Array<u32>) -> Array<u32> {
    let ctx = x.context();
    let rows = rows_of(x);
    // Layers 1-4: 16 slices `v[c]` of the coefficients `16 c ..= 16 c + 15`.
    let blocks = x.reshape([16, BLOCKS, rows]);
    let mut v: Vec<Array<u32>> = (0..BLOCKS).map(|c| blocks.slice(1, c..c + 1)).collect();
    for step in [8, 4, 2, 1] {
        let len = 16 * step;
        for i in (0..BLOCKS).filter(|i| i & step == 0) {
            let zeta = t.zeta_mont[128 / len + i / (2 * step)];
            let (lo, hi) = cooley_tukey(&v[i], &v[i + step], |hi| hi * zeta);
            (v[i], v[i + step]) = (lo, hi);
        }
    }
    evaluate(&v);
    // Layers 5-7: 8 slices `s[m]` of the elements `2 m + j` of every block.
    let y = join(1, &v).cache().reshape([2, 8, BLOCKS, rows]);
    let s: Vec<Array<u32>> = (0..8).map(|m| y.slice(1, m..m + 1)).collect();
    let shape = [2, 1, BLOCKS, rows];
    let block = coordinate(ctx, shape, 2);
    let zeta = |base: u32, per_block: u32, extra: u32| t.zetas.take(&(&block * per_block + (base + extra)));
    let mut s = s;
    // Layer 5 (8 apart): block `c`'s zeta is `zetas[16 + c]`.
    let z8 = zeta(16, 1, 0);
    for m in 0..4 {
        let (lo, hi) = cooley_tukey(&s[m], &s[m + 4], |hi| hi * &z8);
        (s[m], s[m + 4]) = (lo, hi);
    }
    // Layer 6 (4 apart): the zeta of the first or the second half of a block.
    let z4 = [zeta(32, 2, 0), zeta(32, 2, 1)];
    for m in [0, 1, 4, 5] {
        let (lo, hi) = cooley_tukey(&s[m], &s[m + 2], |hi| hi * &z4[m / 4]);
        (s[m], s[m + 2]) = (lo, hi);
    }
    // Layer 7 (2 apart): one zeta per quarter of a block.
    let z2: Vec<Array<u32>> = (0..4).map(|q| zeta(64, 4, q)).collect();
    for m in [0, 2, 4, 6] {
        let (lo, hi) = cooley_tukey(&s[m], &s[m + 1], |hi| hi * &z2[m / 2]);
        (s[m], s[m + 1]) = (lo, hi);
    }
    evaluate(&s);
    join(1, &s).reshape([256, rows])
}

/// FIPS 203 Algorithm 10 on every row (including the final multiplication by 128^-1).
pub fn inv_ntt(t: &Tables, x: &Array<u32>) -> Array<u32> {
    let ctx = x.context();
    let rows = rows_of(x);
    // Layers 1-3 (2, 4 and 8 apart), in the grouping of the last three layers of `ntt`.
    let y = x.reshape([2, 8, BLOCKS, rows]);
    let mut s: Vec<Array<u32>> = (0..8).map(|m| y.slice(1, m..m + 1)).collect();
    let shape = [2, 1, BLOCKS, rows];
    let block = coordinate(ctx, shape, 2);
    // The zeta of a layer is `zetas[2^(8 - j) - 1 - b]` for the block `b` of the 2^j-apart pairs: reversed counts.
    let zeta = |top: u32, per_block: u32, extra: u32| t.zetas.take(&(top - (&block * per_block + extra)));
    let z2: Vec<Array<u32>> = (0..4).map(|q| zeta(127, 4, q)).collect();
    for m in [0, 2, 4, 6] {
        let (lo, hi) = gentleman_sande(&s[m], &s[m + 1], |d| d * &z2[m / 2]);
        (s[m], s[m + 1]) = (lo, hi);
    }
    let z4 = [zeta(63, 2, 0), zeta(63, 2, 1)];
    for m in [0, 1, 4, 5] {
        let (lo, hi) = gentleman_sande(&s[m], &s[m + 2], |d| d * &z4[m / 4]);
        (s[m], s[m + 2]) = (lo, hi);
    }
    let z8 = zeta(31, 1, 0);
    for m in 0..4 {
        let (lo, hi) = gentleman_sande(&s[m], &s[m + 4], |d| d * &z8);
        (s[m], s[m + 4]) = (lo, hi);
    }
    evaluate(&s);
    // Layers 4-7 (16, 32, 64 and 128 apart).
    let blocks = join(1, &s).cache().reshape([16, BLOCKS, rows]);
    let mut v: Vec<Array<u32>> = (0..BLOCKS).map(|c| blocks.slice(1, c..c + 1)).collect();
    for step in [1, 2, 4, 8] {
        let len = 16 * step;
        for i in (0..BLOCKS).filter(|i| i & step == 0) {
            let zeta = t.zeta_mont[256 / len - 1 - i / (2 * step)];
            let (lo, hi) = gentleman_sande(&v[i], &v[i + step], |d| d * zeta);
            (v[i], v[i + step]) = (lo, hi);
        }
    }
    let scale = to_mont(INV128);
    let v: Vec<Array<u32>> = v.iter().map(|p| mont(&(p * scale))).collect();
    evaluate(&v);
    join(1, &v).reshape([256, rows])
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
        let scaled = select(&odd, &term.g_other, mont(&(&term.g_other * &gamma)));
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
