//! `matmul_batched` against a host product in `f64`: tile edges, aligned and not, odd sizes, a
//! batch of one, shared operands, long K, big products (computed one by one), lazy and
//! transposed operands, a grid folded over the dispatch limit.

use crate::common;

use common::values;
use luma::{matmul_batched, Array, Context, LumaError};

/// `C[b][n + N m] = Σ_k A[b][k + K m] B[b][n + N k]` in `f64` for matrix `b` of a batch whose
/// operands are `a` (`[K, M]` per matrix) and `b` (`[N, K]` per matrix).
fn host(a: &[f32], b: &[f32], k: usize, m: usize, n: usize) -> Vec<f64> {
    let mut c = vec![0.0f64; n * m];
    for row in 0..m {
        for kk in 0..k {
            let x = f64::from(a[kk + k * row]);
            for (out, &y) in c[n * row..n * (row + 1)].iter_mut().zip(&b[n * kk..n * (kk + 1)]) {
                *out += x * f64::from(y);
            }
        }
    }
    c
}

/// Compares matrix `i` of `got` with `want`: relative tolerance 1e-4, and an absolute one scaled
/// by the size of the sums for entries near zero.
fn close(got: &[f32], want: &[f64], k: usize, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    let abs = (k as f64).sqrt() * 1e-6 * 4.0;
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        let tolerance = abs.max(1e-4 * w.abs());
        assert!((f64::from(g) - w).abs() <= tolerance, "{what}: index {i}: got {g}, want {w} (tolerance {tolerance})");
    }
}

/// Operands of `batch_a` and `batch_b` matrices (1 for a shared one) as host data and arrays.
struct Case {
    ctx: Context,
    k: usize,
    m: usize,
    n: usize,
    batch: usize,
    a: Vec<f32>,
    b: Vec<f32>,
    batched: [bool; 2],
}

impl Case {
    fn new(k: usize, m: usize, n: usize, batch: usize, batched: [bool; 2]) -> Self {
        let (ba, bb) = (if batched[0] { batch } else { 1 }, if batched[1] { batch } else { 1 });
        let a = values(k * m * ba, 1 + (k * 31 + m * 7 + batch) as u64, -1.0, 1.0);
        let b = values(n * k * bb, 2 + (n * 17 + k + batch * 3) as u64, -1.0, 1.0);
        Case { ctx: Context::new().unwrap(), k, m, n, batch, a, b, batched }
    }

    fn lhs(&self) -> Array<f32> {
        let batch = if self.batched[0] { self.batch } else { 1 };
        Array::from_slice_on(&self.ctx, &self.a, [self.k, self.m, batch]).unwrap()
    }

    fn rhs(&self) -> Array<f32> {
        let batch = if self.batched[1] { self.batch } else { 1 };
        Array::from_slice_on(&self.ctx, &self.b, [self.n, self.k, batch]).unwrap()
    }

    /// The host product of matrix `i`.
    fn want(&self, i: usize, a: &[f32], b: &[f32]) -> Vec<f64> {
        let (ma, mb) = (self.k * self.m, self.n * self.k);
        let at = if self.batched[0] { i * ma } else { 0 };
        let bt = if self.batched[1] { i * mb } else { 0 };
        host(&a[at..at + ma], &b[bt..bt + mb], self.k, self.m, self.n)
    }

    /// Checks the result against the host for the matrices `which` (all if `None`).
    fn check(&self, got: &Array<f32>, a: &[f32], b: &[f32], which: Option<&[usize]>, what: &str) {
        assert_eq!(got.shape().dims(), [self.n, self.m, self.batch, 1], "{what}: shape");
        let got = got.to_vec().unwrap();
        let all: Vec<usize> = (0..self.batch).collect();
        for &i in which.unwrap_or(&all) {
            let size = self.n * self.m;
            close(&got[i * size..(i + 1) * size], &self.want(i, a, b), self.k, &format!("{what}: matrix {i}"));
        }
    }
}

/// `(k, m, n, batch)`.
const SHAPES: [(usize, usize, usize, usize); 22] = [
    // Whole tiles of each edge, K of whole slices.
    (16, 16, 16, 100),
    (32, 32, 32, 64),
    (64, 64, 64, 9),
    (128, 128, 128, 5),
    (48, 48, 48, 11),
    (16, 32, 64, 13),
    // Odd sizes: partial tiles, a K past a slice.
    (5, 7, 3, 33),
    (1, 1, 1, 17),
    (33, 17, 19, 40),
    (31, 50, 100, 12),
    (70, 33, 41, 7),
    (127, 100, 128, 6),
    (20, 8, 8, 1000),
    (17, 16, 16, 5),
    (16, 17, 49, 5),
    // Long K, up to the longest the tiles take and past it (one product each).
    (1024, 20, 20, 8),
    (1000, 24, 33, 5),
    (5000, 16, 16, 4),
    // Big products: one `matmul` each.
    (40, 200, 150, 3),
    (300, 129, 17, 3),
    // A single product.
    (40, 33, 20, 1),
    (1, 130, 1, 2),
];

#[test]
fn matches_the_host_product_for_every_kernel_shape() {
    for (k, m, n, batch) in SHAPES {
        let case = Case::new(k, m, n, batch, [true, true]);
        let c = case.lhs().matmul_batched(&case.rhs());
        case.check(&c, &case.a, &case.b, None, &format!("k {k} m {m} n {n} batch {batch}"));
    }
}

#[test]
fn a_shared_operand_is_applied_to_every_matrix() {
    for (k, m, n, batch) in [(32, 32, 32, 20), (7, 9, 5, 31), (16, 16, 16, 64), (24, 100, 40, 4), (40, 200, 150, 3), (30, 20, 10, 1)] {
        for batched in [[true, false], [false, true]] {
            let case = Case::new(k, m, n, batch, batched);
            let c = case.lhs().matmul_batched(&case.rhs());
            case.check(&c, &case.a, &case.b, None, &format!("k {k} m {m} n {n} batch {batch} {batched:?}"));
        }
    }
}

#[test]
fn both_operands_without_a_batch_are_one_product() {
    let case = Case::new(33, 20, 17, 1, [false, false]);
    let c = case.lhs().matmul_batched(&case.rhs());
    assert_eq!(c.shape().dims(), [17, 20, 1, 1]);
    close(&c.to_vec().unwrap(), &host(&case.a, &case.b, 33, 20, 17), 33, "single product");
    // 2-D operands are the same as `matmul_f32`.
    let (a, b) = (Array::from_slice_on(&case.ctx, &case.a, [33, 20]).unwrap(), Array::from_slice_on(&case.ctx, &case.b, [17, 33]).unwrap());
    assert_eq!(a.matmul_batched(&b).to_vec().unwrap(), a.matmul_f32(&b).to_vec().unwrap());
}

#[test]
fn a_batch_over_two_axes_is_flattened() {
    let (k, m, n, b2, b3) = (21, 18, 25, 4, 3);
    let ctx = Context::new().unwrap();
    let a = values(k * m * b2 * b3, 5, -1.0, 1.0);
    let b = values(n * k * b2 * b3, 6, -1.0, 1.0);
    let lhs = Array::from_slice_on(&ctx, &a, [k, m, b2, b3]).unwrap();
    let rhs = Array::from_slice_on(&ctx, &b, [n, k, b2, b3]).unwrap();
    let c = matmul_batched(&lhs, &rhs);
    assert_eq!(c.shape().dims(), [n, m, b2, b3]);
    let got = c.to_vec().unwrap();
    for i in 0..b2 * b3 {
        let want = host(&a[i * k * m..][..k * m], &b[i * n * k..][..n * k], k, m, n);
        close(&got[i * n * m..][..n * m], &want, k, &format!("matrix {i}"));
    }
    // One operand without the batch, shared over both axes.
    let shared = Array::from_slice_on(&ctx, &b[..n * k], [n, k]).unwrap();
    let c = lhs.matmul_batched(&shared);
    assert_eq!(c.shape().dims(), [n, m, b2, b3]);
    let got = c.to_vec().unwrap();
    for i in 0..b2 * b3 {
        let want = host(&a[i * k * m..][..k * m], &b[..n * k], k, m, n);
        close(&got[i * n * m..][..n * m], &want, k, &format!("shared matrix {i}"));
    }
}

#[test]
fn lazy_and_transposed_operands_are_evaluated_first() {
    let (k, m, n, batch) = (40, 30, 20, 17);
    let case = Case::new(k, m, n, batch, [true, true]);
    let scaled = |v: &[f32], scale: f32, shift: f32| v.iter().map(|&x| x * scale + shift).collect::<Vec<f32>>();
    let (a2, b2) = (scaled(&case.a, 2.0, 1.0), scaled(&case.b, 0.5, -0.25));
    let lazy_a = &case.lhs() * 2.0 + 1.0;
    let lazy_b = &case.rhs() * 0.5 - 0.25;
    case.check(&lazy_a.matmul_batched(&lazy_b), &a2, &b2, None, "lazy");
    // The lazy operands are still usable afterwards.
    let back: Vec<f32> = lazy_a.to_vec().unwrap().iter().map(|&x| (x - 1.0) / 2.0).collect();
    close(&back, &case.a.iter().map(|&x| f64::from(x)).collect::<Vec<_>>(), 1, "operand");
    // Matrices stored transposed: `[M, K, B]` transposed per matrix is `[K, M, B]`.
    let a_t: Vec<f32> = (0..batch).flat_map(|i| (0..k * m).map(move |j| (i, j))).map(|(i, j)| case.a[i * k * m + (j % m) * k + j / m]).collect();
    let lhs = Array::from_slice_on(&case.ctx, &a_t, [m, k, batch]).unwrap().transpose();
    case.check(&lhs.matmul_batched(&case.rhs()), &case.a, &case.b, Some(&[0, batch - 1]), "transposed");
}

#[test]
fn a_huge_batch_folds_its_workgroups_over_the_grid() {
    // 8x8 products, four tiles to a workgroup on a GPU: 400000 products are 100000 workgroups.
    let (k, m, n, batch) = (8, 8, 8, 400_000);
    let case = Case::new(k, m, n, batch, [true, true]);
    let c = case.lhs().matmul_batched(&case.rhs());
    if matches!(c.error().map(LumaError::root), Some(LumaError::Chunked(_))) {
        return;
    }
    case.check(&c, &case.a, &case.b, Some(&[0, 1, 65535, 65536, 99_999, 100_000, 399_998, 399_999]), "folded");
}

#[test]
fn products_of_one_batch_are_independent_of_its_other_products() {
    // The padded tiles of the last workgroup repeat the last product and store nothing: the
    // product after the last one must be untouched, so a batch of `n` agrees with a batch of `n - 1`.
    let case = Case::new(24, 16, 16, 7, [true, true]);
    let full = case.lhs().matmul_batched(&case.rhs()).to_vec().unwrap();
    let head = case.lhs().slice(2, 0..6).matmul_batched(&case.rhs().slice(2, 0..6)).to_vec().unwrap();
    assert_eq!(&full[..head.len()], &head[..]);
}

#[test]
fn shape_errors_poison_the_result() {
    let ctx = Context::new().unwrap();
    let z = |shape: [usize; 3]| Array::<f32>::zeros_on(&ctx, shape).unwrap();
    // K differs.
    let c = z([4, 3, 5]).matmul_batched(&z([5, 6, 5]));
    assert!(matches!(c.error().map(LumaError::root), Some(LumaError::ShapeMismatch { lhs: [4, 3, 5, 1], rhs: [5, 6, 5, 1] })), "{:?}", c.error());
    // Two batches that are not equal and neither is 1.
    let c = z([4, 3, 5]).matmul_batched(&z([4, 4, 6]));
    assert!(matches!(c.error().map(LumaError::root), Some(LumaError::ShapeMismatch { .. })), "{:?}", c.error());
}

#[test]
fn a_poisoned_operand_passes_its_error_on() {
    let ctx = Context::new().unwrap();
    let a = Array::<f32>::zeros_on(&ctx, [4, 3, 2]).unwrap();
    let bad = Array::<f32>::zeros_on(&ctx, [5, 6]).unwrap().matmul(&Array::zeros_on(&ctx, [9, 9]).unwrap());
    let c = a.matmul_batched(&Array::<f32>::zeros_on(&ctx, [4, 4, 2]).unwrap().matmul_batched(&bad));
    assert!(c.check().is_err());
}

#[test]
fn operands_of_two_contexts_poison_the_result() {
    let first = Context::new().unwrap();
    let second = Context::new().unwrap();
    let a = Array::<f32>::zeros_on(&first, [4, 3, 8]).unwrap();
    let b = Array::<f32>::zeros_on(&second, [5, 4, 8]).unwrap();
    let c = a.matmul_batched(&b);
    assert!(matches!(c.error().map(LumaError::root), Some(LumaError::ContextMismatch)), "{:?}", c.error());
}
