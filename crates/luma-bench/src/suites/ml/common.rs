//! Helpers shared by the groups of the `ml` suite: sizes, device arrays, host references.
//!
//! Matrices are row-major on the host; luma's axis 0 is the fastest, so a host `rows x cols`
//! matrix is a luma array of shape `[cols, rows]`, and `matmul(x[k, m], w[n, k])` is the
//! row-major `X (m x k) * W (k x n)`.

use std::thread;

use crate::harness::*;
use luma::*;

/// `quick` or `full` elements, and `cpu` (at most `full`) on the CPU backend.
pub fn size(h: &Harness, quick: usize, full: usize, cpu: usize) -> usize {
    if h.ctx.is_cpu() {
        if h.quick {
            quick.min(cpu)
        } else {
            cpu.min(full)
        }
    } else if h.quick {
        quick
    } else {
        full
    }
}

/// Whether a case of `flops` floating-point operations is in the budget of this run: the whole
/// suite must take under about 3 minutes (CUDA) / 5 minutes (CPU) full and 30 s quick.
pub fn fits(h: &Harness, flops: f64) -> bool {
    let limit = match (h.ctx.is_cpu(), h.quick) {
        (false, false) => f64::INFINITY,
        (false, true) => 4e9,
        (true, false) => 1.5e11,
        (true, true) => 2e9,
    };
    flops <= limit
}

/// An `f32` array of `shape` on the harness' context.
pub fn arr(h: &Harness, data: &[f32], shape: impl Into<Shape>) -> Array<f32> {
    Array::from_slice_on(&h.ctx, data, shape).unwrap()
}

/// A `u32` array of `shape` on the harness' context.
pub fn arr_u32(h: &Harness, data: &[u32], shape: impl Into<Shape>) -> Array<u32> {
    Array::from_slice_on(&h.ctx, data, shape).unwrap()
}

/// The `u32` index of every element along `axis` of an array of `shape`.
pub fn range(h: &Harness, shape: impl Into<Shape>, axis: usize) -> Array<u32> {
    Array::<u32>::range_on(&h.ctx, shape, axis).unwrap()
}

/// Up to `count` evenly spread distinct indices of `0..n`, always including the first and last.
pub fn sample(n: usize, count: usize) -> Vec<usize> {
    let mut v: Vec<usize> = (0..count).map(|i| i * (n - 1) / (count - 1).max(1)).collect();
    v.dedup();
    v
}

/// Rows `rows` of the row-major `cols`-wide matrix `data`, concatenated.
pub fn gather_rows(data: &[f32], cols: usize, rows: &[usize]) -> Vec<f32> {
    rows.iter().flat_map(|&r| data[r * cols..][..cols].iter().copied()).collect()
}

/// `check` against `f64` references.
pub fn check64(name: &str, got: &[f32], want: &[f64], rel: f32, abs: f32) {
    let want: Vec<f32> = want.iter().map(|&v| v as f32).collect();
    check(name, got, &want, rel, abs);
}

/// Like `check`, but up to `frac` of the elements may be off (for outputs that are
/// discontinuous in the inputs, such as an Adam step on a gradient that is zero).
pub fn check_frac(name: &str, got: &[f32], want: &[f32], rel: f32, abs: f32, frac: f64) {
    assert_eq!(got.len(), want.len(), "{name}: length");
    let bad = got.iter().zip(want).filter(|(g, w)| (*g - *w).abs() > abs.max(rel * w.abs()) || g.is_nan()).count();
    assert!(bad as f64 <= frac * got.len() as f64, "{name}: {bad} of {} elements are off", got.len());
}

/// Largest absolute value.
pub fn max_abs(v: &[f32]) -> f32 {
    v.iter().fold(0.0f32, |m, x| m.max(x.abs()))
}

// -------------------------------------------------------------------------------------------------
// Host linear algebra
// -------------------------------------------------------------------------------------------------

pub fn threads() -> usize {
    thread::available_parallelism().map_or(4, |n| n.get()).min(16)
}

/// Runs `f(first_row, block)` over blocks of whole rows of `out` on several threads.
pub fn par_rows<T: Send>(out: &mut [T], row_len: usize, f: impl Fn(usize, &mut [T]) + Sync) {
    let rows = out.len() / row_len.max(1);
    let per = rows.div_ceil(threads()).max(1);
    thread::scope(|s| {
        for (i, chunk) in out.chunks_mut(per * row_len).enumerate() {
            let f = &f;
            s.spawn(move || f(i * per, chunk));
        }
    });
}

/// Row-major `C (m x n) = A (m x k) * B (k x n)`, accumulated in `f64`.
pub fn gemm(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    assert_eq!((a.len(), b.len()), (m * k, k * n));
    let mut c = vec![0.0f32; m * n];
    par_rows(&mut c, n, |r0, block| {
        let mut acc = vec![0.0f64; n];
        for (i, row) in block.chunks_mut(n).enumerate() {
            acc.fill(0.0);
            for (kk, &av) in a[(r0 + i) * k..][..k].iter().enumerate() {
                let av = f64::from(av);
                for (c, &bv) in acc.iter_mut().zip(&b[kk * n..][..n]) {
                    *c += av * f64::from(bv);
                }
            }
            for (o, &c) in row.iter_mut().zip(&acc) {
                *o = c as f32;
            }
        }
    });
    c
}

/// Rows `rows` of `A * B` (`A` is `? x k`, `B` is `k x n`) in `f64`, concatenated.
pub fn gemm_rows(a: &[f32], b: &[f32], k: usize, n: usize, rows: &[usize]) -> Vec<f64> {
    let mut out = vec![0.0f64; rows.len() * n];
    par_rows(&mut out, n, |r0, block| {
        for (i, row) in block.chunks_mut(n).enumerate() {
            let r = rows[r0 + i];
            for (kk, &av) in a[r * k..][..k].iter().enumerate() {
                let av = f64::from(av);
                for (c, &bv) in row.iter_mut().zip(&b[kk * n..][..n]) {
                    *c += av * f64::from(bv);
                }
            }
        }
    });
    out
}

/// The transpose of the row-major `rows x cols` matrix `a`.
pub fn transpose(a: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut t = vec![0.0f32; a.len()];
    for r in 0..rows {
        for c in 0..cols {
            t[c * rows + r] = a[r * cols + c];
        }
    }
    t
}

/// Adds `bias` (one value per column) to every row of `m`, in place.
pub fn add_bias(m: &mut [f32], bias: &[f32]) {
    for row in m.chunks_mut(bias.len()) {
        for (v, b) in row.iter_mut().zip(bias) {
            *v += b;
        }
    }
}

/// The GELU (tanh form) of `x`.
pub fn gelu(x: f64) -> f64 {
    0.5 * x * (1.0 + (0.797_884_560_802_865_4 * (x + 0.044715 * x * x * x)).tanh())
}

/// Log-softmax of one row, in `f64`: `(row - max) - ln(sum(exp(row - max)))`, and the logsumexp.
pub fn softmax_row(row: &[f32]) -> (Vec<f64>, f64) {
    let m = row.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(f64::from(b)));
    let e: Vec<f64> = row.iter().map(|&v| (f64::from(v) - m).exp()).collect();
    let s: f64 = e.iter().sum();
    (e.iter().map(|v| v / s).collect(), m + s.ln())
}

/// The shape of `dims` (one to four extents, axis 0 first).
pub fn shape_of(dims: &[usize]) -> Shape {
    match *dims {
        [a] => Shape::from(a),
        [a, b] => Shape::from([a, b]),
        [a, b, c] => Shape::from([a, b, c]),
        [a, b, c, d] => Shape::from([a, b, c, d]),
        _ => panic!("shape_of: {} dims", dims.len()),
    }
}
