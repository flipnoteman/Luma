//! ML-KEM (FIPS 203) KeyGen, Encaps and Decaps over a batch of independent instances, in lazy
//! array operations only.
//!
//! Keys, ciphertexts and seeds stay on the device as `[words, n]` arrays of little-endian `u32`
//! words (see `codec`): the bytes of an instance are contiguous, as in memory. Polynomials are
//! `[256, rows]` arrays with one row per (instance, polynomial) pair: the polynomial `i` of
//! instance `j` is row `j + n i`. A hash is a sponge over one array per state word (`[1, rows]`
//! each), so its inputs are columns of the `[words, n]` arrays (views) and its outputs are one
//! array per word.

use luma::*;

use super::codec::{cbd, compress, decompress, pack, unpack};
use super::field::{add_mod, base_multiply_sum, inv_ntt, ntt, sub_mod, Tables, Term};
use super::hash::{sha3_256, sha3_512, shake128, shake256};
use super::params::Params;
use super::sample::{sample_ntt, BLOCK_WORDS};

/// A batch of byte strings of equal length: `[words, n]`.
pub type Words = Array<u32>;

/// Words `from..to` of a `[words, n]` array as arrays of shape `[1, n]` (views).
fn cols(words: &Words, from: usize, to: usize) -> Vec<Array<u32>> {
    (from..to).map(|w| words.slice(0, w..w + 1)).collect()
}

/// Joins `parts` along `axis` (a balanced tree of two-part joins).
fn cat(axis: usize, parts: &[Array<u32>]) -> Array<u32> {
    match parts {
        [] => unreachable!("nothing to join"),
        [one] => one.clone(),
        [a, b] => concat(axis, [a, b]),
        _ => {
            let (a, b) = parts.split_at(parts.len() / 2);
            concat(axis, [&cat(axis, a), &cat(axis, b)])
        }
    }
}

/// The `[1, n]` arrays of `parts` as a `[words, n]` array.
fn stack(parts: &[Array<u32>]) -> Array<u32> {
    cat(0, parts)
}

/// `a` (`[x, n]`) repeated `copies` times along its second axis.
fn repeat(a: &Array<u32>, copies: usize) -> Array<u32> {
    let n = a.shape().dims()[1];
    if copies == 1 {
        a.clone()
    } else {
        a.pad([(0, 0), (0, n * (copies - 1))], PadMode::Wrap)
    }
}

/// The result of one operation and whether every polynomial sampled by SampleNTT got its 256
/// coefficients from the blocks squeezed (otherwise the operation is repeated with more blocks).
pub struct Checked<T> {
    pub value: T,
    pub complete: Array<u32>,
}

/// A batch size, a parameter set and the constants on the device.
pub struct Kem<'a> {
    pub ctx: &'a Context,
    pub p: Params,
    pub n: usize,
    pub tables: Tables,
    /// SHAKE128 blocks squeezed per matrix entry.
    pub blocks: usize,
}

impl<'a> Kem<'a> {
    pub fn new(ctx: &'a Context, p: Params, n: usize) -> Result<Kem<'a>> {
        Ok(Kem { ctx, p, n, tables: Tables::new(ctx)?, blocks: 4 })
    }

    /// `[1, rows]` filled with `value`.
    fn constant(&self, rows: usize, value: u32) -> Array<u32> {
        Array::full_on(self.ctx, [1, rows], value).expect("constant array")
    }

    /// The row of every element of a `[1, rows]` array.
    fn row_index(&self, rows: usize) -> Array<u32> {
        Array::<u32>::range_on(self.ctx, [1, rows], 1).expect("index array")
    }

    /// `ByteEncode_d` of the polynomials of a vector (`[256, n k]`) one after the other: `[8 d k, n]`.
    fn pack_vector(&self, v: &Array<u32>, d: u32) -> Array<u32> {
        let parts: Vec<Array<u32>> = (0..self.p.k).map(|i| pack(&self.rows(v, i, i + 1), d)).collect();
        cat(0, &parts)
    }

    /// `ByteDecode_d` of the `k` polynomials that start at word `first` of `words`: `[256, n k]`.
    fn unpack_vector(&self, words: &Words, first: usize, d: u32) -> Array<u32> {
        let parts: Vec<Array<u32>> = (0..self.p.k).map(|i| unpack(words, first + 8 * d as usize * i, d)).collect();
        cat(1, &parts)
    }

    /// The matrix `A` of the seed `rho` (8 arrays): entry `(i, j)` is polynomial `i + k j` (row
    /// `n (i + k j) + instance`), or the transposed matrix.
    fn matrix(&self, rho: &[Array<u32>], transposed: bool) -> (Array<u32>, Array<u32>) {
        let (n, k) = (self.n, self.p.k);
        let rows = n * k * k;
        let at = self.row_index(rows) / n as u32;
        let (i, j) = (&at % k as u32, &at / k as u32);
        let (first, second) = if transposed { (i, j) } else { (j, i) };
        let mut inputs: Vec<Array<u32>> = rho.iter().map(|w| repeat(w, k * k)).collect();
        inputs.push(first | (second << 8u32));
        let stream = shake128(self.ctx, [1, rows], &inputs, 34, 4 * BLOCK_WORDS * self.blocks).expect("sponge");
        sample_ntt(&self.rows_by_word(&stream))
    }

    /// The words of a squeezed stream (`[1, rows]` each) as `[rows, words]`: a row's words are `rows` apart.
    fn rows_by_word(&self, words: &[Array<u32>]) -> Array<u32> {
        let rows = words[0].len();
        let columns: Vec<Array<u32>> = words.iter().map(|w| w.reshape([rows, 1])).collect();
        cat(1, &columns)
    }

    /// `slots` polynomials per instance with noise from `PRF_eta(seed, nonce_base + slot)`: `[256, n slots]`.
    fn noise(&self, seed: &[Array<u32>], slots: usize, nonce_base: u32, eta: u32) -> Array<u32> {
        let rows = self.n * slots;
        let nonce = self.row_index(rows) / self.n as u32 + nonce_base;
        let mut inputs: Vec<Array<u32>> = seed.iter().map(|w| repeat(w, slots)).collect();
        inputs.push(nonce);
        let stream = shake256(self.ctx, [1, rows], &inputs, 33, 64 * eta as usize).expect("sponge");
        cbd(&self.rows_by_word(&stream), eta)
    }

    /// `sum over terms of f_t * g_t` in the NTT domain, for polynomial arrays `fs[t]`, `gs[t]` (`[256, rows]` each).
    fn inner(&self, fs: &[Array<u32>], gs: &[Array<u32>]) -> Array<u32> {
        let rows = fs[0].shape().dims()[1];
        let kappa = Array::<u32>::range_on(self.ctx, [256, rows], 0).expect("index array");
        let row = Array::<u32>::range_on(self.ctx, [256, rows], 1).expect("index array");
        // The other element of the pair (the neighbour with the lowest bit flipped).
        let other = (&kappa ^ 1u32) + &row * 256u32;
        let terms: Vec<Term> = fs
            .iter()
            .zip(gs)
            .map(|(f, g)| Term { f: f.clone(), f_other: f.take(&other), g: g.clone(), g_other: g.take(&other) })
            .collect();
        base_multiply_sum(&self.tables, &kappa, &terms)
    }

    /// The rows of polynomials `from..to` (of `n` rows each) of a polynomial array.
    fn rows(&self, v: &Array<u32>, from: usize, to: usize) -> Array<u32> {
        v.slice(1, self.n * from..self.n * to)
    }

    /// The matrix (`n k` rows per column: polynomial `i + k j`) times a vector (`[256, n k]`).
    fn matrix_times(&self, a: &Array<u32>, x: &Array<u32>) -> Array<u32> {
        let (n, k) = (self.n, self.p.k);
        let fs: Vec<Array<u32>> = (0..k).map(|j| a.slice(1, n * k * j..n * k * (j + 1))).collect();
        let gs: Vec<Array<u32>> = (0..k).map(|j| repeat(&self.rows(x, j, j + 1), k)).collect();
        self.inner(&fs, &gs)
    }

    /// The inner product of two vectors (`[256, n k]` each) for every instance: `[256, n]`.
    fn dot(&self, x: &Array<u32>, y: &Array<u32>) -> Array<u32> {
        let k = self.p.k;
        let fs: Vec<Array<u32>> = (0..k).map(|j| self.rows(x, j, j + 1)).collect();
        let gs: Vec<Array<u32>> = (0..k).map(|j| self.rows(y, j, j + 1)).collect();
        self.inner(&fs, &gs)
    }

    /// `ML-KEM.KeyGen_internal(d, z)`: the encapsulation keys and decapsulation keys of every instance.
    pub fn keygen(&self, d: &Words, z: &Words) -> Checked<(Words, Words)> {
        let (n, k) = (self.n, self.p.k);
        let mut input = cols(d, 0, 8);
        input.push(self.constant(n, k as u32));
        let g = sha3_512(self.ctx, [1, n], &input, 33).expect("sponge");
        let (rho, sigma) = g.split_at(8);
        let (a, complete) = self.matrix(rho, false);
        let a = a.cache();
        let hat = ntt(&self.tables, &self.noise(sigma, 2 * k, 0, self.p.eta1)).cache();
        let t_hat = add_mod(&self.matrix_times(&a, &self.rows(&hat, 0, k)), &self.rows(&hat, k, 2 * k));
        let ek = cat(0, &[self.pack_vector(&t_hat, 12), stack(rho)]).cache();
        let h = sha3_256(self.ctx, [1, n], &cols(&ek, 0, 96 * k + 8), 384 * k + 32).expect("sponge");
        let dk = cat(0, &[self.pack_vector(&self.rows(&hat, 0, k), 12), ek.clone(), stack(&h), z.clone()]);
        Checked { value: (ek, dk), complete }
    }

    /// `K-PKE.Encrypt(ek, m, r)` for the key `ek` (`[96 k + 8, n]`), messages `m` (`[8, n]`) and the
    /// randomness `r` (8 arrays).
    fn encrypt(&self, ek: &Words, m: &Words, r: &[Array<u32>]) -> Checked<Words> {
        let (k, p) = (self.p.k, &self.p);
        let (a_t, complete) = self.matrix(&cols(ek, 96 * k, 96 * k + 8), true);
        let y = self.noise(r, k, 0, p.eta1);
        let e = self.noise(r, k + 1, k as u32, p.eta2).cache();
        let y_hat = ntt(&self.tables, &y).cache();
        let t_hat = self.unpack_vector(ek, 0, 12);
        let u = add_mod(&inv_ntt(&self.tables, &self.matrix_times(&a_t.cache(), &y_hat).cache()), &self.rows(&e, 0, k));
        let mu = decompress(&unpack(m, 0, 1), 1);
        let v = add_mod(&add_mod(&inv_ntt(&self.tables, &self.dot(&t_hat, &y_hat).cache()), &self.rows(&e, k, k + 1)), &mu);
        let c1 = self.pack_vector(&compress(&u, p.du), p.du);
        let c2 = pack(&compress(&v, p.dv), p.dv);
        Checked { value: cat(0, &[c1, c2]), complete }
    }

    /// `ML-KEM.Encaps_internal(ek, m)`: the ciphertexts and shared keys.
    pub fn encaps(&self, ek: &Words, m: &Words) -> Checked<(Words, Words)> {
        let (n, k) = (self.n, self.p.k);
        let h = sha3_256(self.ctx, [1, n], &cols(ek, 0, 96 * k + 8), 384 * k + 32).expect("sponge");
        let mut input = cols(m, 0, 8);
        input.extend(h);
        let g = sha3_512(self.ctx, [1, n], &input, 64).expect("sponge");
        let (key, r) = g.split_at(8);
        let ct = self.encrypt(ek, m, r);
        Checked { value: (ct.value, stack(key)), complete: ct.complete }
    }

    /// `ML-KEM.Decaps(dk, c)` with implicit rejection: the shared keys.
    pub fn decaps(&self, dk: &Words, ct: &Words) -> Checked<Words> {
        let (n, k, p) = (self.n, self.p.k, &self.p);
        let ct_words = 8 * (p.du as usize * k + p.dv as usize);
        // K-PKE.Decrypt
        let u = decompress(&self.unpack_vector(ct, 0, p.du), p.du);
        let v = decompress(&unpack(ct, 8 * p.du as usize * k, p.dv), p.dv);
        let s_hat = self.unpack_vector(dk, 0, 12);
        let u_hat = ntt(&self.tables, &u).cache();
        let w = sub_mod(&v, &inv_ntt(&self.tables, &self.dot(&s_hat, &u_hat).cache()));
        let m = pack(&compress(&w, 1), 1).cache();
        // (K', r') = G(m' || h)
        let mut input = cols(&m, 0, 8);
        input.extend(cols(dk, 192 * k + 8, 192 * k + 16));
        let g = sha3_512(self.ctx, [1, n], &input, 64).expect("sponge");
        let (key, r) = g.split_at(8);
        // K-bar = J(z || c)
        let mut input = cols(dk, 192 * k + 16, 192 * k + 24);
        input.extend(cols(ct, 0, ct_words));
        let rejected = shake256(self.ctx, [1, n], &input, 32 + 4 * ct_words, 32).expect("sponge");
        // c' = K-PKE.Encrypt(ek, m', r')
        let ek = dk.slice(0, 96 * k..192 * k + 8);
        let again = self.encrypt(&ek, &m, r);
        let same = again.value.eq(ct).min_axis(0);
        let chosen: Vec<Array<u32>> = key.iter().zip(&rejected).map(|(a, b)| select(&same, a, b)).collect();
        Checked { value: stack(&chosen), complete: again.complete }
    }
}
