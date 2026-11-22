//! `matmul` against a host product in `f64`: shapes of both tile sizes, aligned and not, split
//! over K, with evaluated, transposed and lazy operands.

mod common;

use common::values;
use luma::{matmul, Array, Context, ContextOptions, LumaError};

/// `C[n + N m] = Σ_k A[k + K m] B[n + N k]` in `f64`. Every element is summed in order of `k`, as
/// ever; the rows of a large product are spread over threads (they are independent), which keeps
/// the host reference of the big shapes off the critical path of the test.
fn host(a: &[f32], b: &[f32], k: usize, m: usize, n: usize) -> Vec<f64> {
    let mut c = vec![0.0f64; n * m];
    let threads = if n * m * k < 1 << 22 { 1 } else { std::thread::available_parallelism().map_or(1, |t| t.get()).min(8) };
    let rows_each = m.div_ceil(threads).max(1);
    let product = |first_row: usize, rows: &mut [f64]| {
        for (r, out_row) in rows.chunks_mut(n).enumerate() {
            let row = first_row + r;
            for kk in 0..k {
                let x = f64::from(a[kk + k * row]);
                let line = &b[n * kk..n * (kk + 1)];
                for (out, &y) in out_row.iter_mut().zip(line) {
                    *out += x * f64::from(y);
                }
            }
        }
    };
    if threads == 1 {
        product(0, &mut c);
    } else {
        std::thread::scope(|scope| {
            for (part, rows) in c.chunks_mut(rows_each * n).enumerate() {
                let product = &product;
                scope.spawn(move || product(part * rows_each, rows));
            }
        });
    }
    c
}

/// Compares with relative tolerance 1e-4, and an absolute one scaled by the size of the sums for
/// entries near zero.
fn check(got: &[f32], want: &[f64], k: usize, a: &[f32], b: &[f32], what: &str) {
    check_scaled(got, want, k, max_abs(a), max_abs(b), what);
}

/// The largest magnitude in `v`.
fn max_abs(v: &[f32]) -> f64 {
    v.iter().fold(0.0f64, |m, &x| m.max(f64::from(x).abs()))
}

/// [`check`] with the largest magnitudes of the operands given (the tolerance of a sampled check
/// of a huge product must not scan the operands for every sample).
fn check_scaled(got: &[f32], want: &[f64], k: usize, max_a: f64, max_b: f64, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    let abs = (k as f64).sqrt() * max_a * max_b * 1e-6;
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        let tolerance = abs.max(1e-4 * w.abs());
        assert!((f64::from(g) - w).abs() <= tolerance, "{what}: index {i}: got {g}, want {w} (tolerance {tolerance})");
    }
}

/// `lhs` of `[K, M]` and `rhs` of `[N, K]`, as host data and as arrays.
struct Case {
    ctx: Context,
    k: usize,
    m: usize,
    n: usize,
    a: Vec<f32>,
    b: Vec<f32>,
}

impl Case {
    fn new(k: usize, m: usize, n: usize) -> Self {
        let a = values(k * m, 1 + (k * 31 + m) as u64, -1.0, 1.0);
        let b = values(n * k, 2 + (n * 17 + k) as u64, -1.0, 1.0);
        Case { ctx: Context::new().unwrap(), k, m, n, a, b }
    }

    fn lhs(&self) -> Array<f32> {
        Array::from_slice_on(&self.ctx, &self.a, [self.k, self.m]).unwrap()
    }

    fn rhs(&self) -> Array<f32> {
        Array::from_slice_on(&self.ctx, &self.b, [self.n, self.k]).unwrap()
    }
}

const SHAPES: [(usize, usize, usize); 10] = [
    (16, 128, 128),
    (1000, 1000, 1000),
    (64, 1 << 16, 64),
    (7, 5, 3),
    (1, 1, 1),
    (1 << 16, 64, 64),
    (5000, 33, 70),
    // Edges that are not whole tiles of a native product, without and with a K split of a slice that is not a multiple of 16.
    (300, 200, 150),
    (4099, 100, 100),
    (3000, 1, 1),
];

#[test]
fn matches_the_host_product_for_every_kernel_shape() {
    for (k, m, n) in SHAPES {
        let case = Case::new(k, m, n);
        let c = case.lhs().matmul(&case.rhs());
        assert_eq!(c.shape().dims(), [n, m, 1, 1], "k {k} m {m} n {n}");
        check(&c.to_vec().unwrap(), &host(&case.a, &case.b, k, m, n), k, &case.a, &case.b, &format!("k {k} m {m} n {n}"));
    }
}

/// Rows past what a grid dimension can number are folded over the grid: `[16, 1 << 22]` by `[8, 16]`
/// has 131072 row tiles of one column tile. Sampled rows (the first, the last, and every 9973rd).
/// The large shape is skipped on a device whose buffers cannot hold it (the small one still folds).
#[test]
fn a_tall_product_folds_its_tiles_over_the_grid() {
    let m = 1 << 22;
    let rows: Vec<usize> = (0..m).step_by(9973).chain([m - 1]).collect();
    let mut ran = 0;
    for (k, n) in [(16, 8), (4, 2)] {
        let case = Case::new(k, m, n);
        let sample = |a: &[f32], b: &[f32], what: &str, c: &Array<f32>| -> bool {
            if matches!(c.error().map(LumaError::root), Some(LumaError::Chunked(_))) {
                return false;
            }
            let got = c.to_vec().unwrap();
            assert_eq!(got.len(), n * m, "{what}");
            let (max_a, max_b) = (max_abs(a), max_abs(b));
            for &row in &rows {
                let want = host(&a[row * k..(row + 1) * k], b, k, 1, n);
                check_scaled(&got[row * n..(row + 1) * n], &want, k, max_a, max_b, &format!("{what} k {k} row {row}"));
            }
            true
        };
        let a2: Vec<f32> = case.a.iter().map(|&x| x * 2.0 + 1.0).collect();
        let b2: Vec<f32> = case.b.iter().map(|&x| x * 0.5).collect();
        let plain = sample(&case.a, &case.b, "buffers", &case.lhs().matmul(&case.rhs()));
        // Fused operands go through the other source of the same kernel.
        let fused = sample(&a2, &b2, "fused", &(&case.lhs() * 2.0 + 1.0).matmul(&(&case.rhs() * 0.5)));
        ran += usize::from(plain && fused);
    }
    assert!(ran >= 1, "no shape fit the device's buffers");
}

#[test]
fn the_free_function_is_the_method() {
    let case = Case::new(40, 33, 129);
    let c = matmul(&case.lhs(), &case.rhs());
    check(&c.to_vec().unwrap(), &host(&case.a, &case.b, 40, 33, 129), 40, &case.a, &case.b, "free function");
}

#[test]
fn small_edges_of_both_tiles_and_both_axes_are_guarded() {
    // Rows or columns just past a tile, K just past a slice, each side of the 128 threshold.
    for (k, m, n) in [(17, 129, 127), (33, 127, 129), (15, 31, 33), (16, 32, 32), (48, 160, 130), (4097, 40, 36)] {
        let case = Case::new(k, m, n);
        let c = case.lhs().matmul(&case.rhs());
        check(&c.to_vec().unwrap(), &host(&case.a, &case.b, k, m, n), k, &case.a, &case.b, &format!("k {k} m {m} n {n}"));
    }
}

#[test]
fn vectors_are_matrices_with_an_extent_of_one() {
    let k = 300;
    let case = Case::new(k, 1, 1);
    // `[K]` times `[1, K]`: a dot product.
    let dot = case.lhs().reshape(k).matmul(&case.rhs().reshape([1, k]));
    check(&dot.to_vec().unwrap(), &host(&case.a, &case.b, k, 1, 1), k, &case.a, &case.b, "dot");
}

#[test]
fn transposed_operands_are_read_through_their_views() {
    for (k, m, n) in [(64, 96, 80), (200, 150, 140), (1 << 12, 33, 48), (64, 128, 256)] {
        let case = Case::new(k, m, n);
        // The same operands, stored the other way round.
        let a_t: Vec<f32> = (0..k * m).map(|i| case.a[(i % m) * k + i / m]).collect();
        let b_t: Vec<f32> = (0..n * k).map(|i| case.b[(i % k) * n + i / k]).collect();
        let lhs = Array::from_slice_on(&case.ctx, &a_t, [m, k]).unwrap().transpose();
        let rhs = Array::from_slice_on(&case.ctx, &b_t, [k, n]).unwrap().transpose();
        let want = host(&case.a, &case.b, k, m, n);
        let what = format!("k {k} m {m} n {n}");
        check(&lhs.matmul(&case.rhs()).to_vec().unwrap(), &want, k, &case.a, &case.b, &format!("{what} lhs"));
        check(&case.lhs().matmul(&rhs).to_vec().unwrap(), &want, k, &case.a, &case.b, &format!("{what} rhs"));
        check(&lhs.matmul(&rhs).to_vec().unwrap(), &want, k, &case.a, &case.b, &format!("{what} both"));
    }
}

#[test]
fn lazy_operands_are_computed_inside_the_product() {
    for (k, m, n) in [(64, 128, 128), (1000, 300, 200), (7, 5, 3), (1 << 16, 64, 64), (5000, 33, 70)] {
        let case = Case::new(k, m, n);
        let scaled = |v: &[f32], scale: f32, shift: f32| v.iter().map(|&x| x * scale + shift).collect::<Vec<f32>>();
        let (a2, b2) = (scaled(&case.a, 2.0, 1.0), scaled(&case.b, 0.5, -0.25));
        let what = format!("k {k} m {m} n {n}");
        let lazy_a = &case.lhs() * 2.0 + 1.0;
        let lazy_b = &case.rhs() * 0.5 - 0.25;
        let lhs_only = lazy_a.matmul(&case.rhs());
        check(&lhs_only.to_vec().unwrap(), &host(&a2, &case.b, k, m, n), k, &a2, &case.b, &format!("{what} lhs lazy"));
        let rhs_only = case.lhs().matmul(&lazy_b);
        check(&rhs_only.to_vec().unwrap(), &host(&case.a, &b2, k, m, n), k, &case.a, &b2, &format!("{what} rhs lazy"));
        let both = lazy_a.matmul(&lazy_b);
        check(&both.to_vec().unwrap(), &host(&a2, &b2, k, m, n), k, &a2, &b2, &format!("{what} both lazy"));
        // The lazy operands are still usable: nothing was consumed.
        check(&lazy_a.to_vec().unwrap().iter().map(|&x| (x - 1.0) / 2.0).collect::<Vec<_>>(), &case.a.iter().map(|&x| f64::from(x)).collect::<Vec<_>>(), 1, &case.a, &case.a, &format!("{what} operand"));
    }
}

#[test]
fn a_lazy_operand_and_a_lazy_transposed_operand_mix() {
    let (k, m, n) = (96, 70, 130);
    let case = Case::new(k, m, n);
    let b_t: Vec<f32> = (0..n * k).map(|i| case.b[(i % k) * n + i / k]).collect();
    let lazy_a = &case.lhs() * 1.5;
    let rhs = (&Array::from_slice_on(&case.ctx, &b_t, [k, n]).unwrap() * 3.0).transpose();
    let a3: Vec<f32> = case.a.iter().map(|&x| x * 1.5).collect();
    let b3: Vec<f32> = case.b.iter().map(|&x| x * 3.0).collect();
    check(&lazy_a.matmul(&rhs).to_vec().unwrap(), &host(&a3, &b3, k, m, n), k, &a3, &b3, "lazy times lazy transposed");
}

#[test]
fn a_lazy_operand_is_not_evaluated_separately() {
    let ctx = Context::new().unwrap();
    if !ctx.profiling_supported() {
        return;
    }
    let (k, m, n) = (128, 128, 128);
    let a = Array::from_slice_on(&ctx, &values(k * m, 3, -1.0, 1.0), [k, m]).unwrap();
    let b = Array::from_slice_on(&ctx, &values(n * k, 4, -1.0, 1.0), [n, k]).unwrap();
    a.eval().unwrap();
    b.eval().unwrap();
    ctx.flush().unwrap();
    // A backend with native products (the CPU one) computes lazy operands first, together.
    let native = ctx.backend() == luma::BackendKind::Cpu && std::env::var("LUMA_NATIVE_GEMM").as_deref().map(str::trim) != Ok("off");
    let lazy = &a * 2.0;
    let (c, profile) = ctx.profile(|| lazy.matmul(&b)).unwrap();
    assert_eq!(profile.launches(), if native { 2 } else { 1 }, "{profile}");
    c.check().unwrap();
    // Both operands lazy: one launch for the two of them, the product being a second one if native.
    let (lazy_a, lazy_b) = (&a * 2.0, &b + 1.0);
    let (c, profile) = ctx.profile(|| lazy_a.matmul(&lazy_b)).unwrap();
    assert_eq!(profile.launches(), if native { 2 } else { 1 }, "{profile}");
    c.check().unwrap();
}

#[test]
fn shape_errors_poison_the_result() {
    let ctx = Context::new().unwrap();
    let a = Array::<f32>::zeros_on(&ctx, [4, 3]).unwrap();
    let b = Array::<f32>::zeros_on(&ctx, [5, 6]).unwrap();
    let c = a.matmul(&b);
    assert!(matches!(c.error().map(LumaError::root), Some(LumaError::ShapeMismatch { lhs: [4, 3, 1, 1], rhs: [5, 6, 1, 1] })), "{:?}", c.error());
    let cube = Array::<f32>::zeros_on(&ctx, [4, 4, 2]).unwrap();
    let ok = Array::<f32>::zeros_on(&ctx, [4, 4]).unwrap();
    for c in [cube.matmul(&ok), ok.matmul(&cube)] {
        assert!(matches!(c.error().map(LumaError::root), Some(LumaError::ShapeMismatch { .. })), "{:?}", c.error());
    }
}

#[test]
fn a_poisoned_operand_passes_its_error_on() {
    let ctx = Context::new().unwrap();
    let a = Array::<f32>::zeros_on(&ctx, [4, 3]).unwrap();
    let bad = Array::<f32>::zeros_on(&ctx, [5, 6]).unwrap().matmul(&Array::zeros_on(&ctx, [9, 9]).unwrap());
    let c = a.matmul(&bad);
    assert!(c.check().is_err());
    assert_eq!(format!("{}", c.error().unwrap()), format!("{}", bad.error().unwrap()));
}

#[test]
fn operands_of_two_contexts_poison_the_result() {
    let first = Context::new().unwrap();
    let second = Context::new().unwrap();
    let a = Array::<f32>::zeros_on(&first, [4, 3]).unwrap();
    let b = Array::<f32>::zeros_on(&second, [5, 4]).unwrap();
    let c = a.matmul(&b);
    assert!(matches!(c.error().map(LumaError::root), Some(LumaError::ContextMismatch)), "{:?}", c.error());
}

#[test]
fn a_gram_matrix_of_a_lazy_expression_reads_it_through_a_transposed_view() {
    for (features, samples) in [(64, 1 << 14), (33, 1000), (130, 70), (8, 5000)] {
        let ctx = Context::new().unwrap();
        let x = values(features * samples, 9 + samples as u64, -1.0, 1.0);
        let standard = |v: f32| (v - 0.25) * 1.5;
        let xs: Vec<f32> = x.iter().map(|&v| standard(v)).collect();
        let lazy = (&Array::from_slice_on(&ctx, &x, [features, samples]).unwrap() - 0.25) * 1.5;
        let gram = lazy.transpose().matmul(&lazy);
        assert_eq!(gram.shape().dims(), [features, features, 1, 1]);
        // `A[k + K m] = xs[m + F k]`, `B[n + F k] = xs[n + F k]`.
        let a: Vec<f32> = (0..features * samples).map(|i| xs[(i / samples) + features * (i % samples)]).collect();
        check(&gram.to_vec().unwrap(), &host(&a, &xs, samples, features, features), samples, &a, &xs, &format!("{features} x {samples}"));
    }
}

#[test]
fn operands_or_a_result_split_over_several_buffers_poison_the_result() {
    // Buffers of 4096 bytes: 1024 `f32`.
    let ctx = Context::with_options(ContextOptions::default().chunk_bytes(4096)).unwrap();
    let array = |e: usize, f: usize| Array::<f32>::zeros_on(&ctx, [e, f]).unwrap();
    let chunked = |c: &Array<f32>| match c.error().map(LumaError::root) {
        Some(LumaError::Chunked(_)) => Some(true),
        Some(_) => Some(false),
        None => None,
    };
    // `[K, M]` times `[N, K]`: an evaluated operand, a lazy one, and a result that is too large.
    assert_eq!(chunked(&array(100, 100).matmul(&array(4, 100))), Some(true));
    assert_eq!(chunked(&array(100, 4).matmul(&array(100, 100))), Some(true));
    assert_eq!(chunked(&(&array(100, 100) * 2.0).matmul(&array(4, 100))), Some(true));
    assert_eq!(chunked(&array(4, 40).matmul(&array(40, 4))), Some(true));
    assert_eq!(chunked(&array(8, 16).matmul(&array(16, 8))), None);
    assert_eq!(chunked(&(&array(8, 16) * 2.0).matmul(&array(16, 8))), None);
    assert_eq!(chunked(&array(8, 16).matmul(&array(16, 9))), Some(false));
}

/// Small outputs over a long `K` run as skinny kernels (register tiles loaded from global memory,
/// the `K` shared out between the invocations that own the same outputs): every shape of
/// invocation tile, tile counts that are and are not a power of two, outputs past a tile, `K`
/// that does not divide into the slices, and both operands as buffers, views and lazy expressions.
#[test]
fn a_small_output_over_a_long_k_matches_the_host_product() {
    for (k, m, n) in [
        (1_050_003, 1, 1),
        (40_000, 1, 40),
        (30_001, 40, 1),
        (100_003, 10, 32),
        (70_001, 64, 64),
        (20_000, 11, 9),
        (9_999, 2, 200),
        (66_000, 130, 70),
        (65_600, 90, 100),
        (2_049, 20, 25),
        (70_000, 10, 256),
        (1 << 18, 7, 5),
        (20_000, 100, 3),
        (6_000, 3_000, 2),
        (8_195, 1_025, 1),
        (3_000, 4_097, 8),
        (2_100, 65, 8),
    ] {
        let case = Case::new(k, m, n);
        let what = format!("k {k} m {m} n {n}");
        let want = host(&case.a, &case.b, k, m, n);
        let (max_a, max_b) = (max_abs(&case.a), max_abs(&case.b));
        let c = case.lhs().matmul(&case.rhs());
        assert_eq!(c.shape().dims(), [n, m, 1, 1], "{what}");
        let got = c.to_vec().unwrap();
        check_scaled(&got, &want, k, max_a, max_b, &what);
        // The sum order is fixed: the same product gives the same bits.
        assert_eq!(got, case.lhs().matmul(&case.rhs()).to_vec().unwrap(), "{what}: not reproducible");
        // Operands stored the other way round and read through views.
        let a_t: Vec<f32> = (0..k * m).map(|i| case.a[(i % m) * k + i / m]).collect();
        let b_t: Vec<f32> = (0..n * k).map(|i| case.b[(i % k) * n + i / k]).collect();
        let lhs = Array::from_slice_on(&case.ctx, &a_t, [m, k]).unwrap().transpose();
        let rhs = Array::from_slice_on(&case.ctx, &b_t, [k, n]).unwrap().transpose();
        check_scaled(&lhs.matmul(&case.rhs()).to_vec().unwrap(), &want, k, max_a, max_b, &format!("{what} lhs view"));
        check_scaled(&case.lhs().matmul(&rhs).to_vec().unwrap(), &want, k, max_a, max_b, &format!("{what} rhs view"));
        check_scaled(&lhs.matmul(&rhs).to_vec().unwrap(), &want, k, max_a, max_b, &format!("{what} both views"));
        // Lazy operands.
        let (a2, b2): (Vec<f32>, Vec<f32>) = (case.a.iter().map(|&x| x * 2.0 + 1.0).collect(), case.b.iter().map(|&x| x * 0.5 - 0.25).collect());
        let both = (&case.lhs() * 2.0 + 1.0).matmul(&(&case.rhs() * 0.5 - 0.25));
        check(&both.to_vec().unwrap(), &host(&a2, &b2, k, m, n), k, &a2, &b2, &format!("{what} lazy"));
    }
}

/// `onehot(codes)ᵀ X`: the class sums of a Gaussian naive Bayes fit, with the one-hot matrix a lazy
/// transposed expression that the kernel computes in its loads.
#[test]
fn class_sums_through_a_lazy_one_hot_matrix_match_the_host() {
    for (n, d, classes) in [(100_000usize, 32usize, 10usize), (20_001, 256, 10), (50_000, 7, 3), (30_000, 33, 11)] {
        let ctx = Context::new().unwrap();
        let x = values(n * d, 5 + d as u64, -1.0, 1.0);
        let codes: Vec<u32> = values(n, 6, 0.0, classes as f32).iter().map(|&v| (v as u32).min(classes as u32 - 1)).collect();
        let xs = Array::from_slice_on(&ctx, &x, [d, n]).unwrap();
        let labels = Array::<u32>::from_slice_on(&ctx, &codes, [n]).unwrap();
        let onehot = labels.reshape([1, n]).broadcast_to([classes, n]).eq(Array::<u32>::range_on(&ctx, [classes, n], 0).unwrap()).to_f32();
        let sums = onehot.transpose().matmul(&xs);
        assert_eq!(sums.shape().dims(), [d, classes, 1, 1]);
        let mut want = vec![0.0f64; classes * d];
        for (i, &c) in codes.iter().enumerate() {
            for f in 0..d {
                want[c as usize * d + f] += f64::from(x[i * d + f]);
            }
        }
        let got = sums.to_vec().unwrap();
        for (i, (&g, &w)) in got.iter().zip(&want).enumerate() {
            assert!((f64::from(g) - w).abs() <= 1e-3 + 1e-4 * w.abs(), "n {n} d {d}: index {i}: got {g}, want {w}");
        }
    }
}
