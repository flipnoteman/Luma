//! Products whose extents are not multiples of anything: on CUDA the backend's own kernels take them (masked tails, the ragged and
//! padded strategies); on every other backend the product still equals the host product.

use crate::common;

use common::values;
use luma::{matmul_tf32, Array, BackendKind, Context, ContextOptions, DecisionSite, Numerics};

fn exact() -> Context {
    Context::with_options(ContextOptions::from_env().numerics(Numerics::strict())).unwrap()
}

/// A context that forces the strategy `name` of the contraction family.
fn forced(name: &'static str) -> Context {
    Context::with_options(ContextOptions::from_env().numerics(Numerics::strict()).strategy(name)).unwrap()
}

/// Whether the backend runs the products on the backend's own CUDA kernels.
fn cuda_kernels(ctx: &Context) -> bool {
    ctx.backend() == BackendKind::Cuda && std::env::var("LUMA_OWN_GEMM").as_deref().map(str::trim) != Ok("off")
}

/// `C[n + N m] = Σ_k A[k + K m] B[n + N k]` and `Σ_k |A B|` in `f64`.
fn host(a: &[f32], b: &[f32], k: usize, m: usize, n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut c = vec![0.0f64; n * m];
    let mut abs = vec![0.0f64; n * m];
    for row in 0..m {
        for kk in 0..k {
            let x = f64::from(a[kk + k * row]);
            for col in 0..n {
                let product = x * f64::from(b[n * kk + col]);
                c[n * row + col] += product;
                abs[n * row + col] += product.abs();
            }
        }
    }
    (c, abs)
}

/// Every entry within `bound * Σ|ab|` of the `f64` product.
fn within(got: &[f32], want: &(Vec<f64>, Vec<f64>), bound: f64, what: &str) {
    assert_eq!(got.len(), want.0.len(), "{what}: length");
    for (i, ((&g, &w), &abs)) in got.iter().zip(&want.0).zip(&want.1).enumerate() {
        assert!((f64::from(g) - w).abs() <= bound * abs + 1e-6, "{what}: entry {i}: got {g}, want {w} (|ab| sum {abs})");
    }
}

/// The four layouts of a product: the left matrix `[K, M]` stored as such or as a view of its transpose, the right one `[N, K]` likewise.
fn operands(ctx: &Context, (m, n, k): (usize, usize, usize), transposed: [bool; 2], seed: u64) -> (Vec<f32>, Vec<f32>, Array<f32>, Array<f32>) {
    let a = values(k * m, seed, -1.0, 1.0);
    let b = values(n * k, seed + 1, -1.0, 1.0);
    let lhs = if transposed[0] {
        // A stored `m x k` array (`[M, K]`) read transposed.
        let mut t = vec![0.0f32; k * m];
        for r in 0..m {
            for c in 0..k {
                t[r + m * c] = a[c + k * r];
            }
        }
        Array::from_slice_on(ctx, &t, [m, k]).unwrap().transpose()
    } else {
        Array::from_slice_on(ctx, &a, [k, m]).unwrap()
    };
    let rhs = if transposed[1] {
        let mut t = vec![0.0f32; n * k];
        for r in 0..k {
            for c in 0..n {
                t[r + k * c] = b[c + n * r];
            }
        }
        Array::from_slice_on(ctx, &t, [k, n]).unwrap().transpose()
    } else {
        Array::from_slice_on(ctx, &b, [n, k]).unwrap()
    };
    (a, b, lhs, rhs)
}

#[test]
fn ragged_shapes_match_the_host_for_every_tile_and_layout() {
    let exact_ctx = exact();
    let tf32_ctx = Context::new().unwrap();
    let extents = [1usize, 3, 17, 31, 33, 100, 1001];
    for &m in &extents {
        for &n in &extents {
            for &k in &extents {
                // The big corners are covered by the sweeps below; the host reference of 1001^3 would dominate the test.
                if m * n * k > 4_000_000 {
                    continue;
                }
                for transposed in [[false; 2], [true, false], [false, true], [true; 2]] {
                    let (a, b, lhs, rhs) = operands(&exact_ctx, (m, n, k), transposed, 1 + (m * 7 + n * 13 + k) as u64);
                    let want = host(&a, &b, k, m, n);
                    within(&lhs.matmul(&rhs).to_vec().unwrap(), &want, 1e-5, &format!("f32 {m}x{n}x{k} {transposed:?}"));
                    let (_, _, lhs, rhs) = operands(&tf32_ctx, (m, n, k), transposed, 1 + (m * 7 + n * 13 + k) as u64);
                    within(&matmul_tf32(&lhs, &rhs).to_vec().unwrap(), &want, 1.5e-3, &format!("tf32 {m}x{n}x{k} {transposed:?}"));
                }
            }
        }
    }
}

/// Shapes that reach the ragged and padded kernels (above 32 on a side and a million multiply-adds), none a multiple of 4.
const ODD: [(usize, usize, usize); 4] = [(101, 103, 107), (257, 130, 129), (131, 4001, 67), (1001, 129, 255)];

#[test]
fn ragged_and_padded_are_bit_identical() {
    let probe = exact();
    if !cuda_kernels(&probe) {
        return;
    }
    let ragged = forced("contraction:gemm-f32-ragged");
    let padded = forced("contraction:gemm-f32-padded");
    for shape in ODD {
        for transposed in [[false; 2], [true, false], [false, true], [true; 2]] {
            let (_, _, lhs, rhs) = operands(&ragged, shape, transposed, 5);
            let (_, _, plhs, prhs) = operands(&padded, shape, transposed, 5);
            let a = lhs.matmul(&rhs).to_vec().unwrap();
            let b = plhs.matmul(&prhs).to_vec().unwrap();
            assert!(a.iter().zip(&b).all(|(x, y)| x.to_bits() == y.to_bits()), "{shape:?} {transposed:?}: ragged and padded differ");
        }
    }
    // The same for TF32.
    let ragged = Context::with_options(ContextOptions::from_env().strategy("contraction:gemm-tf32-ragged")).unwrap();
    let padded = Context::with_options(ContextOptions::from_env().strategy("contraction:gemm-tf32-padded")).unwrap();
    for shape in ODD {
        let (_, _, lhs, rhs) = operands(&ragged, shape, [false; 2], 6);
        let (_, _, plhs, prhs) = operands(&padded, shape, [false; 2], 6);
        let a = matmul_tf32(&lhs, &rhs).to_vec().unwrap();
        let b = matmul_tf32(&plhs, &prhs).to_vec().unwrap();
        assert!(a.iter().zip(&b).all(|(x, y)| x.to_bits() == y.to_bits()), "{shape:?}: tf32 ragged and padded differ");
    }
}

#[test]
fn a_ragged_product_is_bit_identical_to_the_dense_one_of_the_same_values() {
    let ctx = exact();
    if !cuda_kernels(&ctx) {
        return;
    }
    // 1001 x 997 x 1003 against the same values in the corner of a product that is a multiple of 4 on every side: every output is the
    // same chain of fused multiply-adds over `k` (the extra products are exact zeros).
    let (m, n, k) = (1001usize, 997usize, 1003usize);
    let (m4, n4, k4) = (m.next_multiple_of(4), n.next_multiple_of(4), k.next_multiple_of(4));
    let a = values(k * m, 61, -1.0, 1.0);
    let b = values(n * k, 62, -1.0, 1.0);
    let mut a4 = vec![0.0f32; k4 * m4];
    let mut b4 = vec![0.0f32; n4 * k4];
    for r in 0..m {
        a4[k4 * r..k4 * r + k].copy_from_slice(&a[k * r..k * r + k]);
    }
    for r in 0..k {
        b4[n4 * r..n4 * r + n].copy_from_slice(&b[n * r..n * r + n]);
    }
    let ragged = Array::from_slice_on(&ctx, &a, [k, m]).unwrap().matmul(&Array::from_slice_on(&ctx, &b, [n, k]).unwrap()).to_vec().unwrap();
    let dense = Array::from_slice_on(&ctx, &a4, [k4, m4]).unwrap().matmul(&Array::from_slice_on(&ctx, &b4, [n4, k4]).unwrap()).to_vec().unwrap();
    for r in 0..m {
        for c in 0..n {
            assert_eq!(ragged[c + n * r].to_bits(), dense[c + n4 * r].to_bits(), "entry ({c}, {r})");
        }
    }
}

#[test]
fn a_ragged_shape_returns_the_same_bits_cold_and_warm() {
    let ctx = exact();
    if !cuda_kernels(&ctx) {
        return;
    }
    // The first call may find the kernel compiling: it waits for it (nothing of its bit class stands in) rather than run another
    // kernel, so the second call, with the kernel warm, gives the same bits.
    let (m, n, k) = (131, 4001, 67);
    let (_, _, lhs, rhs) = operands(&ctx, (m, n, k), [false; 2], 71);
    let cold = lhs.matmul(&rhs).to_vec().unwrap();
    let warm = lhs.matmul(&rhs).to_vec().unwrap();
    assert!(cold.iter().zip(&warm).all(|(x, y)| x.to_bits() == y.to_bits()), "cold and warm calls differ");
}

#[test]
fn the_odd_squares_run_on_the_ragged_or_padded_kernels() {
    let ctx = exact();
    if !cuda_kernels(&ctx) {
        return;
    }
    let (_, _, lhs, rhs) = operands(&ctx, (257, 259, 261), [false; 2], 81);
    ctx.record_decisions(true);
    lhs.matmul(&rhs).eval().unwrap();
    ctx.sync().unwrap();
    let decisions = ctx.take_decisions();
    ctx.record_decisions(false);
    let chosen: Vec<String> = decisions.iter().filter(|d| d.site == DecisionSite::MatmulPlan).map(|d| d.chosen.to_string()).collect();
    assert!(chosen.iter().any(|label| label.starts_with("gemm") && (label.contains("ragged") || label.contains("padded"))), "{chosen:?}");
}

#[test]
fn small_and_skinny_products_keep_the_shader_kernels() {
    let ctx = exact();
    if !cuda_kernels(&ctx) {
        return;
    }
    // Below a million multiply-adds, or an extent below 32, or a long K over a small output: the shader kernels, as before.
    for (m, n, k) in [(17, 19, 23), (100, 100, 100), (4096, 10, 31), (10, 32, 100_003)] {
        let (_, _, lhs, rhs) = operands(&ctx, (m, n, k), [false; 2], 91);
        ctx.record_decisions(true);
        lhs.matmul(&rhs).eval().unwrap();
        ctx.sync().unwrap();
        let decisions = ctx.take_decisions();
        ctx.record_decisions(false);
        let chosen: Vec<String> = decisions.iter().filter(|d| d.site == DecisionSite::MatmulPlan).map(|d| d.chosen.to_string()).collect();
        assert!(chosen.iter().all(|label| !label.contains("ragged") && !label.contains("padded")), "{m}x{n}x{k}: {chosen:?}");
    }
}
