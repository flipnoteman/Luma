//! `matmul_tf32` against a host product in `f64`: on tensor cores (CUDA) the operands are rounded to
//! TF32, so every entry is within `2^-10` of the sum of the absolute products of its `f32`
//! operands; elsewhere (and for shapes the tensor-core kernels do not take) it is the `f32` product.

mod common;

use common::values;
use luma::{matmul_tf32, Array, Context};

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

/// Every entry within `bound * Σ|ab|` (+ the `f32` rounding of the sum) of the `f64` product, and
/// the RMS error of the result under `rms_bound` of the RMS of the product.
fn check(got: &[f32], want: &(Vec<f64>, Vec<f64>), bound: f64, rms_bound: f64, what: &str) {
    assert_eq!(got.len(), want.0.len(), "{what}: length");
    let (mut squares, mut errors) = (0.0f64, 0.0f64);
    for (i, ((&g, &w), &abs)) in got.iter().zip(&want.0).zip(&want.1).enumerate() {
        let error = (f64::from(g) - w).abs();
        assert!(error <= bound * abs + 1e-6, "{what}: index {i}: got {g}, want {w} (|ab| sum {abs})");
        squares += w * w;
        errors += error * error;
    }
    let (rms, rms_error) = ((squares / got.len() as f64).sqrt(), (errors / got.len() as f64).sqrt());
    assert!(rms_error <= rms_bound * rms.max(1e-9), "{what}: rms error {rms_error:e} of {rms:e}");
}

fn operands(ctx: &Context, k: usize, m: usize, n: usize) -> (Vec<f32>, Vec<f32>, Array<f32>, Array<f32>) {
    let a = values(k * m, 1 + (k * 31 + m) as u64, -1.0, 1.0);
    let b = values(n * k, 2 + (n * 17 + k) as u64, -1.0, 1.0);
    let (lhs, rhs) = (Array::from_slice_on(ctx, &a, [k, m]).unwrap(), Array::from_slice_on(ctx, &b, [n, k]).unwrap());
    (a, b, lhs, rhs)
}

/// Shapes: whole tiles, edges that are not, a K that is not a multiple of the staged slice, a long
/// K that is split, and shapes the tensor-core kernels do not take (odd, small).
const SHAPES: [(usize, usize, usize); 12] = [
    (128, 128, 128),
    (256, 512, 384),
    (100, 36, 68),
    (36, 132, 196),
    (4100, 64, 64),
    (1 << 14, 40, 56),
    (1000, 300, 200),
    (64, 32, 32),
    (7, 5, 3),
    (33, 100, 100),
    (16, 128, 128),
    (3000, 1, 1),
];

#[test]
fn tf32_products_are_within_the_rounding_of_their_operands() {
    let ctx = Context::new().unwrap();
    for (k, m, n) in SHAPES {
        let (a, b, lhs, rhs) = operands(&ctx, k, m, n);
        let want = host(&a, &b, k, m, n);
        check(&matmul_tf32(&lhs, &rhs).to_vec().unwrap(), &want, 1.1e-3, 5e-4, &format!("k {k} m {m} n {n}"));
        check(&lhs.matmul_tf32(&rhs).to_vec().unwrap(), &want, 1.1e-3, 5e-4, &format!("method k {k} m {m} n {n}"));
    }
}

#[test]
fn transposed_operands_are_read_in_place() {
    let ctx = Context::new().unwrap();
    for (k, m, n) in [(64, 96, 80), (200, 148, 140), (4096, 36, 48), (64, 128, 256), (36, 132, 64)] {
        let (a, b, _, _) = operands(&ctx, k, m, n);
        let a_t: Vec<f32> = (0..k * m).map(|i| a[(i % m) * k + i / m]).collect();
        let b_t: Vec<f32> = (0..n * k).map(|i| b[(i % k) * n + i / k]).collect();
        let lhs_t = Array::from_slice_on(&ctx, &a_t, [m, k]).unwrap().transpose();
        let rhs_t = Array::from_slice_on(&ctx, &b_t, [k, n]).unwrap().transpose();
        let lhs = Array::from_slice_on(&ctx, &a, [k, m]).unwrap();
        let rhs = Array::from_slice_on(&ctx, &b, [n, k]).unwrap();
        let want = host(&a, &b, k, m, n);
        let what = format!("k {k} m {m} n {n}");
        check(&lhs_t.matmul_tf32(&rhs).to_vec().unwrap(), &want, 1.1e-3, 5e-4, &format!("{what} lhs"));
        check(&lhs.matmul_tf32(&rhs_t).to_vec().unwrap(), &want, 1.1e-3, 5e-4, &format!("{what} rhs"));
        check(&lhs_t.matmul_tf32(&rhs_t).to_vec().unwrap(), &want, 1.1e-3, 5e-4, &format!("{what} both"));
    }
}

#[test]
fn lazy_operands_are_evaluated_first() {
    let ctx = Context::new().unwrap();
    let (k, m, n) = (96, 64, 128);
    let (a, b, lhs, rhs) = operands(&ctx, k, m, n);
    let scaled_a: Vec<f32> = a.iter().map(|&x| x * 0.5 + 0.25).collect();
    let scaled_b: Vec<f32> = b.iter().map(|&x| x * 2.0 - 0.5).collect();
    let want = host(&scaled_a, &scaled_b, k, m, n);
    check(&(&lhs * 0.5 + 0.25).matmul_tf32(&(&rhs * 2.0 - 0.5)).to_vec().unwrap(), &want, 1.1e-3, 5e-4, "lazy");
}

#[test]
fn the_default_product_does_not_round_its_operands() {
    let ctx = Context::new().unwrap();
    let (k, m, n) = (256, 128, 128);
    let (a, b, lhs, rhs) = operands(&ctx, k, m, n);
    let want = host(&a, &b, k, m, n);
    // `f32` accuracy: far below the TF32 bound.
    check(&lhs.matmul(&rhs).to_vec().unwrap(), &want, 5e-5, 1e-6, "f32");
}

#[test]
fn results_are_reproducible() {
    let ctx = Context::new().unwrap();
    let (_, _, lhs, rhs) = operands(&ctx, 4100, 64, 64);
    let first = lhs.matmul_tf32(&rhs).to_vec().unwrap();
    assert_eq!(first, lhs.matmul_tf32(&rhs).to_vec().unwrap());
}

#[test]
fn operands_are_rounded_to_tf32_where_tensor_cores_run() {
    let ctx = Context::new().unwrap();
    let (k, m, n) = (64, 64, 64);
    // 1 + 2^-12 is an `f32` but not a TF32 (10 bits of mantissa): it rounds to 1.
    let a = Array::from_slice_on(&ctx, &vec![1.0f32 + 1.0 / 4096.0; k * m], [k, m]).unwrap();
    let b = Array::from_slice_on(&ctx, &vec![1.0f32; n * k], [n, k]).unwrap();
    let exact = k as f32 * (1.0 + 1.0 / 4096.0);
    let rounded = k as f32;
    let tensor = a.matmul_tf32(&b).to_vec().unwrap();
    let plain = a.matmul(&b).to_vec().unwrap();
    assert!(plain.iter().all(|&x| x == exact), "the default product rounds its operands");
    assert!(tensor.iter().all(|&x| x == exact) || tensor.iter().all(|&x| x == rounded), "neither the f32 nor the TF32 product: {:?}", &tensor[..4]);
    eprintln!("matmul_tf32 on {}: {}", ctx.device_name(), if tensor[0] == rounded { "tensor cores" } else { "f32 product" });
}
