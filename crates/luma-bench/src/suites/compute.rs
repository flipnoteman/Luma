//! The `compute` suite: GPU compute workloads written with the array API.
//!
//! * `ew`, `wide`, `transpose`, `red`: memory-bound elementwise work (copy, saxpy, fused chains of
//!   2..64 operations, `f64` and double-float variants), transposes and reductions at many sizes;
//! * `matmul`: square products 128..4096, a tall-skinny one, matvec and a product with a lazily
//!   transposed operand;
//! * `scan`: prefix scans and axis reductions over contiguous, short and strided lines;
//! * `nbody`, `particles`: all-pairs forces by broadcasting and particle update loops;
//! * `mc`, `int`: random numbers, Monte Carlo pricing, integer hashing and scatter;
//! * `fractal`: Mandelbrot and Julia sets as `iterate_all` register loops and unrolled chains;
//! * `fft`, `image`: FFT convolution against direct convolution, and image filters built from
//!   shifts, slices, scans and gathers;
//! * `imageops`: the image operations of `luma` (convolutions, rank and morphological filters,
//!   Sobel, bilateral, summed-area tables, colour conversions, `unwrap`) at 1080p and 4K, 3-D filters;
//! * `visionops`: the feature detectors, matchers and image analysis of `luma::vision` at 1080p and 4K;
//! * `approx`, `warp`, `filters`: `approx1`/`approx2` interpolation, `resize`/`rotate`/`transform`
//!   image warps, and `fir` / `iir` filters (see `resample`);
//! * `keccak`, `mlkem`: batched SHA3 and ML-KEM in `u32` array operations (see `crypto`);
//! * `structure`: `tile`, `diag`, `identity`, `lower`/`upper`, `iota`, `replace` and `diff1`/`diff2` as
//!   kernels of their own (the structural half of ArrayFire's array API).
//!
//! Every case is checked against a host reference (`f64` where precision matters) before it is
//! timed. Sizes shrink under `--quick` and on the CPU backend.

mod bandwidth;
mod boundary;
mod boundary_conv;
mod imageops;
mod visionops;
mod crypto;
mod dense;
mod dense_svd;
mod linalg;
mod physics;
mod resample;
mod scans;
mod signal;
mod special;
mod structure;
mod spectral;

use std::time::{Duration, Instant};

use luma::*;

use crate::harness::Harness;

/// `quick` in `--quick` runs, else `full`.
pub fn pick<T>(h: &Harness, quick: T, full: T) -> T {
    if h.quick {
        quick
    } else {
        full
    }
}

/// Keeps the GPU busy for a moment so that the first cases do not time its clock ramp-up. The
/// kernel shares nothing with the cases (so their first-call costs stay theirs).
fn warm_up(h: &Harness) {
    if h.ctx.is_cpu() {
        return;
    }
    let x = Array::<f32>::linspace_on(&h.ctx, 0.0, 1.0, 1 << 24).unwrap();
    let start = Instant::now();
    while start.elapsed() < Duration::from_millis(400) {
        let mut z = x.clone();
        for _ in 0..6 {
            z = (z * 1.0001 + 0.1).tan().cos();
        }
        z.eval().unwrap();
        h.ctx.sync().unwrap();
    }
}

/// `data` as an evaluated array of `shape` on the harness's context.
pub fn upload(h: &Harness, data: &[f32], shape: impl Into<Shape>) -> Array<f32> {
    let a = Array::from_slice_on(&h.ctx, data, shape).unwrap();
    a.eval().unwrap();
    a
}

/// The array read back.
pub fn host(a: &Array<f32>) -> Vec<f32> {
    a.to_vec().unwrap()
}

/// Panics unless every `got[i]` is within `max(abs, rel * |want[i]|)` of the `f64` reference.
pub fn check64(name: &str, got: &[f32], want: &[f64], rel: f64, abs: f64) {
    assert_eq!(got.len(), want.len(), "{name}: length");
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        let tol = abs.max(rel * w.abs());
        assert!((f64::from(g) - w).abs() <= tol, "{name}: index {i}: got {g}, want {w} (tol {tol})");
    }
}

/// [`check64`] at every `stride`-th element only, with the reference computed per element by
/// `want(i)`: verifies large results without a host copy of the whole reference.
pub fn check_sampled(name: &str, got: &[f32], want: impl Fn(usize) -> f64, stride: usize, rel: f64, abs: f64) {
    for i in (0..got.len()).step_by(stride.max(1)) {
        let w = want(i);
        let tol = abs.max(rel * w.abs());
        assert!((f64::from(got[i]) - w).abs() <= tol, "{name}: index {i}: got {}, want {w} (tol {tol})", got[i]);
    }
}

pub fn run(h: &mut Harness) {
    warm_up(h);
    bandwidth::run(h);
    linalg::run(h);
    scans::run(h);
    boundary::run(h);
    boundary_conv::run(h);
    physics::run(h);
    signal::run(h);
    imageops::run(h);
    visionops::run(h);
    spectral::run(h);
    resample::run(h);
    crypto::run(h);
    special::run(h);
    structure::run(h);
    dense::run(h);
    dense_svd::run(h);
}
