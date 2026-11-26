//! Products of windows of stored matrices (blocks, transposed blocks, blocks of blocks): the backend's own kernels (CUDA's, the
//! CPU's) read the buffer where it is, through a leading dimension and an offset, instead of copying the block first; the other
//! backends fuse the view. Every case equals the host product, and where the kernels are native the product is one launch.

use crate::common;

use common::values;
use luma::{Array, BackendKind, Context, ContextOptions, DecisionSite};

/// A context whose products are `f32` on every backend. On CUDA the ragged kernel is forced: the cost model may rather copy a small
/// unaligned block into an aligned buffer (the padded kernel), which is not what these tests look at (where the forced strategy does
/// not fit, the default selection runs).
fn exact() -> Context {
    Context::with_options(ContextOptions::from_env().numerics(luma::Numerics::strict()).strategy("contraction:gemm-f32-ragged")).unwrap()
}

/// `C[n + N m] = Σ_k A[k + K m] B[n + N k]` in `f64`.
fn host(a: &[f32], b: &[f32], k: usize, m: usize, n: usize) -> Vec<f64> {
    let mut c = vec![0.0f64; n * m];
    for row in 0..m {
        for kk in 0..k {
            let x = f64::from(a[kk + k * row]);
            for col in 0..n {
                c[n * row + col] += x * f64::from(b[n * kk + col]);
            }
        }
    }
    c
}

fn close(got: &[f32], want: &[f64], k: usize, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        // Entries are sums of `k` products of values in `[-1, 1]`.
        assert!((f64::from(g) - w).abs() <= 1e-4 * (k as f64).sqrt() + 1e-5, "{what}: entry {i}: got {g}, want {w}");
    }
}

/// The `cols x rows` window of the matrix `data` of `width` columns that starts at column `c0` of row `r0`, in the order `[col + cols * row]`.
fn window(data: &[f32], width: usize, (c0, cols): (usize, usize), (r0, rows): (usize, usize)) -> Vec<f32> {
    (0..rows).flat_map(|r| (0..cols).map(move |c| data[c0 + c + width * (r0 + r)])).collect()
}

/// The transpose of `values` (`[col + cols * row]`, `rows` rows).
fn transpose(data: &[f32], cols: usize, rows: usize) -> Vec<f32> {
    (0..cols).flat_map(|c| (0..rows).map(move |r| data[c + cols * r])).collect()
}

/// A stored matrix of `width x height` and the arrays of a `k x m` left window at `(c0, r0)` and a `n x k` right window.
struct Stored {
    a_data: Vec<f32>,
    b_data: Vec<f32>,
    a: Array<f32>,
    b: Array<f32>,
}

fn stored(ctx: &Context, (k, m, n): (usize, usize, usize), a_at: (usize, usize), b_at: (usize, usize), seed: u64) -> (Stored, [usize; 2], [usize; 2]) {
    let a_dims = [k + a_at.0 + 5, m + a_at.1 + 3];
    let b_dims = [n + b_at.0 + 2, k + b_at.1 + 7];
    let a_data = values(a_dims[0] * a_dims[1], seed, -1.0, 1.0);
    let b_data = values(b_dims[0] * b_dims[1], seed + 1, -1.0, 1.0);
    let a = Array::from_slice_on(ctx, &a_data, a_dims).unwrap();
    let b = Array::from_slice_on(ctx, &b_data, b_dims).unwrap();
    a.eval().unwrap();
    b.eval().unwrap();
    ctx.flush().unwrap();
    (Stored { a_data, b_data, a, b }, a_dims, b_dims)
}

#[test]
fn a_block_of_a_stored_matrix_is_read_in_place() {
    let ctx = exact();
    let (k, m, n) = (130, 150, 120);
    // Offsets and leading dimensions that are not multiples of 4, and ones that are.
    for (a_at, b_at) in [((3, 5), (6, 2)), ((4, 8), (8, 4))] {
        let (s, a_dims, b_dims) = stored(&ctx, (k, m, n), a_at, b_at, 7);
        let product = || {
            let lhs = s.a.slice(1, a_at.1..a_at.1 + m).slice(0, a_at.0..a_at.0 + k);
            let rhs = s.b.slice(1, b_at.1..b_at.1 + k).slice(0, b_at.0..b_at.0 + n);
            lhs.matmul(&rhs)
        };
        let a = window(&s.a_data, a_dims[0], (a_at.0, k), (a_at.1, m));
        let b = window(&s.b_data, b_dims[0], (b_at.0, n), (b_at.1, k));
        let want = host(&a, &b, k, m, n);
        if ctx.profiling_supported() {
            // (The arrays the closure builds are dropped before the profile ends: it computes the lazy arrays still alive.)
            let (c, profile) = ctx.profile(product).unwrap();
            assert_eq!(profile.launches(), 1, "{a_at:?} {b_at:?}: {profile}");
            close(&c.to_vec().unwrap(), &want, k, "block times block");
        } else {
            close(&product().to_vec().unwrap(), &want, k, "block times block");
        }
    }
}

#[test]
fn a_transposed_block_is_read_in_place() {
    let ctx = exact();
    let (k, m, n) = (130, 150, 120);
    // The left window is the transpose of a stored `m x k` block, the right one the transpose of a stored `k x n` block.
    let (a_at, b_at) = ((3, 5), (6, 2));
    let a_dims = [m + a_at.0 + 5, k + a_at.1 + 3];
    let b_dims = [k + b_at.0 + 2, n + b_at.1 + 7];
    let a_data = values(a_dims[0] * a_dims[1], 11, -1.0, 1.0);
    let b_data = values(b_dims[0] * b_dims[1], 12, -1.0, 1.0);
    let (a_stored, b_stored) = (Array::from_slice_on(&ctx, &a_data, a_dims).unwrap(), Array::from_slice_on(&ctx, &b_data, b_dims).unwrap());
    a_stored.eval().unwrap();
    b_stored.eval().unwrap();
    ctx.flush().unwrap();
    let product = || {
        let lhs = a_stored.slice(1, a_at.1..a_at.1 + k).slice(0, a_at.0..a_at.0 + m).transpose();
        let rhs = b_stored.slice(1, b_at.1..b_at.1 + n).slice(0, b_at.0..b_at.0 + k).transpose();
        lhs.matmul(&rhs)
    };
    let a = transpose(&window(&a_data, a_dims[0], (a_at.0, m), (a_at.1, k)), m, k);
    let b = transpose(&window(&b_data, b_dims[0], (b_at.0, k), (b_at.1, n)), k, n);
    let want = host(&a, &b, k, m, n);
    if ctx.profiling_supported() {
        let (c, profile) = ctx.profile(product).unwrap();
        assert_eq!(profile.launches(), 1, "{profile}");
        close(&c.to_vec().unwrap(), &want, k, "transposed blocks");
    } else {
        close(&product().to_vec().unwrap(), &want, k, "transposed blocks");
    }
}

#[test]
fn a_block_of_a_view_of_a_block() {
    let ctx = exact();
    let (k, m, n) = (130, 150, 120);
    let (s, a_dims, b_dims) = stored(&ctx, (k + 8, m + 6, n + 4), (2, 1), (1, 3), 21);
    // The maps of the views compose: one window of the stored matrix.
    let product = || {
        let lhs = s.a.slice(1, 1..1 + m + 6).slice(0, 2..2 + k + 8).slice(1, 3..3 + m).slice(0, 1..1 + k);
        let rhs = s.b.slice(1, 3..3 + k + 8).slice(0, 1..1 + n + 4).slice(1, 2..2 + k).slice(0, 3..3 + n);
        lhs.matmul(&rhs)
    };
    let a = window(&s.a_data, a_dims[0], (2 + 1, k), (1 + 3, m));
    let b = window(&s.b_data, b_dims[0], (1 + 3, n), (3 + 2, k));
    let want = host(&a, &b, k, m, n);
    if ctx.profiling_supported() {
        let (c, profile) = ctx.profile(product).unwrap();
        assert_eq!(profile.launches(), 1, "{profile}");
        close(&c.to_vec().unwrap(), &want, k, "block of a block");
    } else {
        close(&product().to_vec().unwrap(), &want, k, "block of a block");
    }
}

#[test]
fn an_unaligned_block_goes_to_the_ragged_strategy() {
    let ctx = exact();
    if ctx.backend() != BackendKind::Cuda || std::env::var("LUMA_OWN_GEMM").as_deref().map(str::trim) == Ok("off") {
        return;
    }
    let (k, m, n) = (256, 512, 256);
    let (s, a_dims, b_dims) = stored(&ctx, (k, m, n), (3, 5), (6, 2), 31);
    let lhs = s.a.slice(1, 5..5 + m).slice(0, 3..3 + k);
    let rhs = s.b.slice(1, 2..2 + k).slice(0, 6..6 + n);
    ctx.record_decisions(true);
    let got = lhs.matmul(&rhs).to_vec().unwrap();
    let decisions = ctx.take_decisions();
    ctx.record_decisions(false);
    let chosen: Vec<String> = decisions.iter().filter(|d| d.site == DecisionSite::MatmulPlan).map(|d| d.chosen.to_string()).collect();
    // Unaligned leading dimensions and offsets: the padded kernel copies them, the ragged one reads them in place; never the aligned one.
    assert!(chosen.iter().any(|label| label.contains("ragged") || label.contains("padded")), "{chosen:?}");
    let a = window(&s.a_data, a_dims[0], (3, k), (5, m));
    let b = window(&s.b_data, b_dims[0], (6, n), (2, k));
    close(&got, &host(&a, &b, k, m, n), k, "unaligned block");
}

#[test]
fn nothing_outside_a_window_reaches_the_product() {
    let ctx = exact();
    let (k, m, n) = (130, 150, 120);
    let (a_at, b_at) = ((3, 5), (6, 2));
    let a_dims = [k + a_at.0 + 5, m + a_at.1 + 3];
    let b_dims = [n + b_at.0 + 2, k + b_at.1 + 7];
    // NaN everywhere but the window: the kernels must not read it (a masked position is an exact zero, not a NaN times zero).
    let mut a_data = vec![f32::NAN; a_dims[0] * a_dims[1]];
    let mut b_data = vec![f32::NAN; b_dims[0] * b_dims[1]];
    let inner = values(k * m + n * k, 41, -1.0, 1.0);
    for r in 0..m {
        for c in 0..k {
            a_data[a_at.0 + c + a_dims[0] * (a_at.1 + r)] = inner[c + k * r];
        }
    }
    for r in 0..k {
        for c in 0..n {
            b_data[b_at.0 + c + b_dims[0] * (b_at.1 + r)] = inner[k * m + c + n * r];
        }
    }
    let (a_stored, b_stored) = (Array::from_slice_on(&ctx, &a_data, a_dims).unwrap(), Array::from_slice_on(&ctx, &b_data, b_dims).unwrap());
    a_stored.eval().unwrap();
    b_stored.eval().unwrap();
    ctx.flush().unwrap();
    let lhs = a_stored.slice(1, a_at.1..a_at.1 + m).slice(0, a_at.0..a_at.0 + k);
    let rhs = b_stored.slice(1, b_at.1..b_at.1 + k).slice(0, b_at.0..b_at.0 + n);
    let a = window(&a_data, a_dims[0], (a_at.0, k), (a_at.1, m));
    let b = window(&b_data, b_dims[0], (b_at.0, n), (b_at.1, k));
    let got = lhs.matmul(&rhs).to_vec().unwrap();
    assert!(got.iter().all(|x| x.is_finite()), "a NaN of the padding reached the product");
    close(&got, &host(&a, &b, k, m, n), k, "window in a sea of NaN");
}

#[test]
fn a_view_with_a_negative_step_is_not_a_window() {
    let ctx = exact();
    let (k, m, n) = (130, 150, 120);
    let (s, a_dims, b_dims) = stored(&ctx, (k, m, n), (3, 5), (6, 2), 51);
    // A flipped window is not a start and a leading dimension: it takes the copy path, and the product is the same.
    let lhs = s.a.slice(1, 5..5 + m).slice(0, 3..3 + k).flip(0);
    let rhs = s.b.slice(1, 2..2 + k).slice(0, 6..6 + n);
    let a: Vec<f32> = window(&s.a_data, a_dims[0], (3, k), (5, m)).chunks(k).flat_map(|row| row.iter().rev().copied().collect::<Vec<_>>()).collect();
    let b = window(&s.b_data, b_dims[0], (6, n), (2, k));
    close(&lhs.matmul(&rhs).to_vec().unwrap(), &host(&a, &b, k, m, n), k, "flipped window");
}
