//! Fusion around structured values (W3-U4): `relu(a b + bias)` at the sizes where the epilogue's elementwise pass is 23 % (256^2), 8 %
//! (1024^2) and 4 % (4096^2) of the product, the bound of what absorbing it can save.

use super::common::*;
use crate::harness::*;
use luma::*;

/// `relu(a (n x n) * b (n x n) + bias)`.
fn bias_relu(h: &mut Harness, n: usize) {
    let flops = 2.0 * (n as f64).powi(3);
    let limit = match (h.ctx.is_cpu(), h.quick) {
        (false, false) => f64::INFINITY,
        (false, true) => 4e9,
        (true, false) => 1.5e11,
        (true, true) => 2e9,
    };
    if flops > limit {
        return;
    }
    let case = format!("fuse/bias_relu_{n}");
    h.guarded(&case, |h| {
        let mut rng = Rng(n as u64 * 7 + 1);
        let scale = 1.0 / (n as f32).sqrt();
        let a: Vec<f32> = (0..n * n).map(|_| rng.range(-1.0, 1.0)).collect();
        let b: Vec<f32> = (0..n * n).map(|_| rng.range(-scale, scale)).collect();
        let bias: Vec<f32> = (0..n).map(|_| rng.range(-0.5, 0.5)).collect();
        let (a, b, bias) = (
            Array::from_slice_on(&h.ctx, &a, [n, n]).unwrap(),
            Array::from_slice_on(&h.ctx, &b, [n, n]).unwrap(),
            Array::from_slice_on(&h.ctx, &bias, n).unwrap(),
        );
        let run = || (&matmul(&a, &b) + &bias).maximum(0.0);
        let first = run().to_vec().unwrap();
        assert!(first.iter().all(|v| v.is_finite() && *v >= 0.0), "{case}: relu output");
        h.case(&case, &format!("{n}x{n}x{n}"), Work::Flops(flops), &mut || run().eval().unwrap());
    });
}

pub fn run(h: &mut Harness) {
    for n in [256, 1024, 4096] {
        bias_relu(h, n);
    }
}
