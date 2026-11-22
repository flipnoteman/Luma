//! The numerical rank of a matrix too big for the workgroup SVD, from QR factorisations with a
//! certificate, and a singular value decomposition only when the certificate fails.
//!
//! The decision is the one of the singular values: the rank is the number of singular values
//! above `tol * s_max`. Singular values of `n x n` matrices cost a Jacobi iteration, seconds at
//! 4096, while a rank that is not close to the tolerance needs only to be *bounded* from both
//! sides, and two QR factorisations do that.
//!
//! `A = Q R1` (Householder, no pivoting; `A` is taken tall, a wide matrix as its transpose). Let `k`
//! be such that the last `n - k` rows of `R1` have a spectral norm `u` of at most `tol s_lo`,
//! `s_lo` a lower estimate of the largest singular value (power iteration on `R1`); the rows are
//! those of the trailing `(n - k) x (n - k)` triangle (the rows are cut off from the bottom, the
//! largest `k` for which that holds is found by bisection):
//!
//! * `R1` with these rows set to zero has rank at most `k` and differs from `R1` by `u` in the
//!   spectral norm, so `s_{k+1}(A) <= u`: at most `k` singular values are above the cut;
//! * the rows `W = R1[:k, :]` have `s_k(A) >= s_k(W) = s_min(R_W)` (Cauchy interlacing, `R_W` the
//!   triangular factor of the QR factorisation of `Wᵀ`, or `R1` itself when `k = n`), and
//!   `s_min(R_W) >= 1 / ||R_W⁻¹||_F`: if that exceeds `tol s_hi` (`s_hi` an upper estimate of the
//!   largest singular value, 1.3 times the power iteration), at least `k` are above it.
//!
//! When both hold the rank is `k`, what the singular values give, unless a singular value lies
//! within the uncertainty of `s_max` (about 30%) of the cut, or within the rounding errors of the
//! factorisations (a few `eps ||A||_F`, the size of the backward error of the Jacobi iteration
//! too) of the other side of it. Otherwise (matrices whose rank is decided by singular values
//! close to the cut, as random square `f32` matrices of 4096 columns at the default tolerance, or
//! rows that are not cut off from the bottom) the caller computes the singular values. The
//! decision is also left to the singular values when the tolerance is so small that the rounding
//! errors of the factorisations would matter: the first row below the rank of a rank-deficient
//! matrix, which the rounding errors fill, has a norm of about `3 sqrt(n) eps ||A||_F` after an
//! unpivoted Householder QR factorisation (4096 and 1024 columns), as big as the default cut
//! `max(m, n) eps s_max`; at the default tolerance only matrices of full rank are decided here.

use super::super::Triangle;
use crate::array::Array;
use crate::error::LumaError;
use crate::ops::select;
use crate::shape::Shape;
/// Steps of the power iteration for the largest singular value of `R1`.
const POWER_STEPS: usize = 10;
/// Steps of the power iterations on the trailing triangles of the bisection.
const TAIL_STEPS: usize = 8;
/// How far the power iteration may be below the largest singular value.
const POWER_MARGIN: f32 = 1.3;

/// A lower estimate of the spectral norm of `m` (`[n, t]`: `t` rows of `n` columns): power
/// iteration on `mᵀ m` from a fixed start vector, `steps` of them.
fn spectral(m: &Array<f32>, steps: usize) -> Result<f32, LumaError> {
    let [n, t, ..] = m.shape().dims();
    let start: Vec<f32> = (0..n).map(|i| ((i as f32 + 1.0) * 0.618_034).fract() * 2.0 - 1.0).collect();
    // A zero vector stays zero (the 1e-30 keeps the norm off zero).
    let unit = |z: &Array<f32>| z / &(&(z * z).sum().sqrt() + 1e-30f32);
    let mut v = unit(&Array::from_slice_on(m.context(), &start, [1, n])?);
    for _ in 0..steps {
        let y = m.matmul_f32(&v);
        v = unit(&y.reshape([t, 1]).matmul_f32(m).reshape([1, n]));
    }
    let y = m.matmul_f32(&v);
    (&y * &y).sum().sqrt().item()
}

/// The rank of one matrix (`[cols, rows]`) if the certificate holds, `None` if the singular values
/// have to decide.
pub(super) fn certified(a: &Array<f32>, tol: f32) -> Result<Option<u32>, LumaError> {
    let ctx = a.context();
    let [cols, rows, ..] = a.shape().dims();
    let x = if rows >= cols { a.clone() } else { a.transpose() };
    let n = cols.min(rows);
    let mx = x.abs().max().item()?;
    if mx == 0.0 || !mx.is_finite() {
        // A zero matrix has rank 0; the singular values of a matrix with NaN or Inf are NaN.
        return Ok(Some(0));
    }
    let r1 = (&x / mx).cache().qr().r().cache();
    let row2 = (&r1 * &r1).sum_axis(0).to_vec()?;
    let frob = row2.iter().map(|&v| f64::from(v)).sum::<f64>().sqrt() as f32;
    if !frob.is_finite() {
        return Ok(None);
    }
    let est = spectral(&r1, POWER_STEPS)?;
    let s_lo = est.max(1.0);
    let s_hi = (est * POWER_MARGIN).min(frob).max(s_lo);
    let (cut_lo, cut_hi) = (tol * s_lo, tol * s_hi);
    // Rounding errors of the factorisations of a matrix of full rank: a few eps ||A||_F.
    if cut_lo.is_nan() || cut_lo <= 8.0 * f32::EPSILON * frob {
        return Ok(None);
    }
    // The rows cut off from the bottom: only rows of a norm within the cut can go, and the
    // trailing triangle of those is a matrix of the rounding errors to bound by `cut_lo`.
    let small = row2.iter().rev().take_while(|&&v| v <= cut_lo * cut_lo).count();
    // The first row below the rank of a rank-deficient matrix has a norm of about 3 sqrt(n) eps
    // ||A||_F (measured, 4096 and 1024 columns): a cut below twice that cannot be certified.
    if small > 0 && cut_lo <= 6.0 * (n as f32).sqrt() * f32::EPSILON * frob {
        return Ok(None);
    }
    let tail = |t: usize| -> Result<f32, LumaError> { spectral(&r1.slice(1, n - t..n).slice(0, n - t..n).cache(), TAIL_STEPS) };
    let mut cut_rows = 0;
    if small > 0 {
        if tail(small)? <= cut_lo {
            cut_rows = small;
        } else {
            let (mut lo, mut hi) = (0, small);
            while hi - lo > 1 {
                let mid = (lo + hi) / 2;
                if tail(mid)? <= cut_lo {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            cut_rows = lo;
        }
    }
    let k = n - cut_rows;
    if k == 0 {
        return Ok(Some(0));
    }
    let rw = if k == n { r1 } else { r1.slice(1, 0..k).transpose().cache().qr().r().cache() };
    let eye = {
        let shape = Shape::from([k, k]);
        select(&Array::<u32>::range_on(ctx, shape, 0)?.eq(&Array::<u32>::range_on(ctx, shape, 1)?), 1.0f32, 0.0f32)
    };
    let inv = rw.solve_triangular(&eye, Triangle::Upper, false, false);
    let finv = (&inv * &inv).sum().sqrt().item()?;
    let sigma_k = 1.0 / finv;
    // `!(a > b)` also rejects a NaN: a singular or failed factor.
    Ok(if sigma_k > cut_hi { Some(k as u32) } else { None })
}
