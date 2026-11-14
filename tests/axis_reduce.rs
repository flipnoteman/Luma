//! Reductions along an axis: operands fused into the first pass, several reductions per pass,
//! mean, variance and standard deviation.

mod common;

use common::{assert_close, values};
use luma::{Array, Context, LumaError, Reduction};

/// Host reduction of `data` (luma layout, `dims` with the first axis fastest) along `axis`, with
/// the f64 values reduced by `fold` starting from the first element.
fn host(data: &[f64], dims: [usize; 3], axis: usize, fold: impl Fn(f64, f64) -> f64) -> Vec<f64> {
    let stride: usize = dims[..axis].iter().product();
    let n = dims[axis];
    let lines = data.len() / n;
    (0..lines)
        .map(|l| {
            let base = l % stride + (l / stride) * stride * n;
            (1..n).fold(data[base], |acc, s| fold(acc, data[base + s * stride]))
        })
        .collect()
}

fn as_f32(v: &[f64]) -> Vec<f32> {
    v.iter().map(|&x| x as f32).collect()
}

#[test]
fn fused_sum_axis_runs_without_a_store_kernel() {
    let ctx = Context::new().unwrap();
    // Arrays split over several buffers (this size on the CPU device) are not reduced along an axis.
    let (cols, rows) = (64usize, if ctx.is_cpu() { 1usize << 16 } else { 1 << 20 });
    let (a, b) = (values(cols * rows, 1, -1.0, 1.0), values(cols * rows, 2, -1.0, 1.0));
    let x = Array::from_slice_on(&ctx, &a, [cols, rows]).unwrap();
    let y = Array::from_slice_on(&ctx, &b, [cols, rows]).unwrap();
    x.eval().unwrap();
    y.eval().unwrap();
    ctx.flush().unwrap();
    let (sum, profile) = ctx.profile(|| (&x * &y).sum_axis(1)).unwrap();
    assert!(profile.launches() <= 2 || ctx.is_cpu(), "{profile}");
    for kernel in &profile.kernels {
        assert!(!kernel.label.starts_with("Fused(") && !kernel.label.starts_with("Fused["), "a store kernel ran: {profile}");
    }
    let want: Vec<f64> = (0..cols).map(|c| (0..rows).map(|r| f64::from(a[c + cols * r]) * f64::from(b[c + cols * r])).sum()).collect();
    assert_close(&sum.to_vec().unwrap(), &as_f32(&want), 2e-3, 1e-2);
}

#[test]
fn several_reductions_share_one_first_pass() {
    let ctx = Context::new().unwrap();
    let (cols, rows) = (64usize, 1usize << 18);
    let (a, b) = (values(cols * rows, 3, -1.0, 1.0), values(cols * rows, 4, 0.5, 1.5));
    let x = Array::from_slice_on(&ctx, &a, [cols, rows]).unwrap();
    let w = Array::from_slice_on(&ctx, &b[..cols], [cols, 1]).unwrap();
    x.eval().unwrap();
    w.eval().unwrap();
    ctx.flush().unwrap();
    let ((sum, squares), profile) = ctx
        .profile(|| {
            let [s, q] = (&x * &w).reduce_axis(1, [Reduction::Sum, Reduction::SumOfSquares]);
            (s, q)
        })
        .unwrap();
    let first: u32 = profile.kernels.iter().filter(|k| k.label.starts_with("FusedAxisReduce")).map(|k| k.launches).sum();
    assert_eq!(first, 1, "{profile}");
    assert!(profile.kernels.iter().all(|k| !k.label.starts_with("Fused(")), "{profile}");
    let y = |c: usize, r: usize| f64::from(a[c + cols * r]) * f64::from(b[c]);
    let want_sum: Vec<f64> = (0..cols).map(|c| (0..rows).map(|r| y(c, r)).sum()).collect();
    let want_squares: Vec<f64> = (0..cols).map(|c| (0..rows).map(|r| y(c, r) * y(c, r)).sum()).collect();
    assert_close(&sum.to_vec().unwrap(), &as_f32(&want_sum), 2e-3, 1e-2);
    assert_close(&squares.to_vec().unwrap(), &as_f32(&want_squares), 2e-3, 1e-2);
}

const ALL: [Reduction; 4] = [Reduction::Min, Reduction::Max, Reduction::Sum, Reduction::SumOfSquares];

#[test]
fn every_reduction_on_every_axis_f32() {
    let ctx = Context::new().unwrap();
    let dims = [37usize, 45, 3];
    let len: usize = dims.iter().product();
    let a = values(len, 5, -2.0, 2.0);
    let wide: Vec<f64> = a.iter().map(|&v| f64::from(v)).collect();
    let x = Array::from_slice_on(&ctx, &a, dims).unwrap();
    // The same data, evaluated and as a pending expression (`x * 1 + 0` is not simplified away
    // because of the scale below).
    let evaluated = x.clone();
    evaluated.eval().unwrap();
    let pending = &x * 2.0;
    let scaled: Vec<f64> = wide.iter().map(|v| v * 2.0).collect();
    for axis in 0..3 {
        for (input, data) in [(&evaluated, &wide), (&pending, &scaled)] {
            let [min, max, sum, squares] = input.reduce_axis(axis, ALL);
            let mut shape = dims;
            shape[axis] = 1;
            assert_eq!(min.shape().dims()[..3], shape, "axis {axis}");
            assert_close(&min.to_vec().unwrap(), &as_f32(&host(data, dims, axis, f64::min)), 1e-6, 1e-6);
            assert_close(&max.to_vec().unwrap(), &as_f32(&host(data, dims, axis, f64::max)), 1e-6, 1e-6);
            assert_close(&sum.to_vec().unwrap(), &as_f32(&host(data, dims, axis, |a, b| a + b)), 1e-4, 1e-4);
            let squared: Vec<f64> = data.iter().map(|v| v * v).collect();
            let want = host(&squared, dims, axis, |a, b| a + b);
            assert_close(&squares.to_vec().unwrap(), &as_f32(&want), 1e-4, 1e-4);
        }
    }
}

#[test]
fn every_reduction_on_every_axis_u32() {
    let ctx = Context::new().unwrap();
    let dims = [37usize, 45, 3];
    let len: usize = dims.iter().product();
    let a: Vec<u32> = (0..len as u32).map(|i| i.wrapping_mul(2654435761) >> 8).collect();
    let x = Array::from_slice_on(&ctx, &a, dims).unwrap();
    let evaluated = x.clone();
    evaluated.eval().unwrap();
    let pending = &x + 7u32;
    let shifted: Vec<u32> = a.iter().map(|v| v.wrapping_add(7)).collect();
    for axis in 0..3 {
        for (input, data) in [(&evaluated, &a), (&pending, &shifted)] {
            let [min, max, sum, squares] = input.reduce_axis(axis, ALL);
            let stride: usize = dims[..axis].iter().product();
            let n = dims[axis];
            let lines = len / n;
            let line = |l: usize| {
                let base = l % stride + (l / stride) * stride * n;
                (0..n).map(move |s| data[base + s * stride])
            };
            let want = |f: &dyn Fn(u32, u32) -> u32, g: &dyn Fn(u32) -> u32| -> Vec<u32> {
                (0..lines).map(|l| line(l).map(g).reduce(f).unwrap()).collect()
            };
            assert_eq!(min.to_vec().unwrap(), want(&|a, b| a.min(b), &|v| v), "min axis {axis}");
            assert_eq!(max.to_vec().unwrap(), want(&|a, b| a.max(b), &|v| v), "max axis {axis}");
            assert_eq!(sum.to_vec().unwrap(), want(&|a, b| a.wrapping_add(b), &|v| v), "sum axis {axis}");
            assert_eq!(squares.to_vec().unwrap(), want(&|a, b| a.wrapping_add(b), &|v| v.wrapping_mul(v)), "squares axis {axis}");
        }
    }
}

#[test]
fn every_reduction_on_every_axis_i32() {
    let ctx = Context::new().unwrap();
    let dims = [37usize, 45, 3];
    let len: usize = dims.iter().product();
    let a: Vec<i32> = (0..len as u32).map(|i| (i.wrapping_mul(2654435761) >> 8) as i32 - (1 << 23)).collect();
    let x = Array::from_slice_on(&ctx, &a, dims).unwrap();
    let evaluated = x.clone();
    evaluated.eval().unwrap();
    let pending = &x + 7i32;
    let shifted: Vec<i32> = a.iter().map(|v| v.wrapping_add(7)).collect();
    for axis in 0..3 {
        for (input, data) in [(&evaluated, &a), (&pending, &shifted)] {
            let [min, max, sum, squares] = input.reduce_axis(axis, ALL);
            let stride: usize = dims[..axis].iter().product();
            let n = dims[axis];
            let lines = len / n;
            let line = |l: usize| {
                let base = l % stride + (l / stride) * stride * n;
                (0..n).map(move |s| data[base + s * stride])
            };
            let want = |f: &dyn Fn(i32, i32) -> i32, g: &dyn Fn(i32) -> i32| -> Vec<i32> {
                (0..lines).map(|l| line(l).map(g).reduce(f).unwrap()).collect()
            };
            assert_eq!(min.to_vec().unwrap(), want(&|a, b| a.min(b), &|v| v), "min axis {axis}");
            assert_eq!(max.to_vec().unwrap(), want(&|a, b| a.max(b), &|v| v), "max axis {axis}");
            assert_eq!(sum.to_vec().unwrap(), want(&|a, b| a.wrapping_add(b), &|v| v), "sum axis {axis}");
            assert_eq!(squares.to_vec().unwrap(), want(&|a, b| a.wrapping_add(b), &|v| v.wrapping_mul(v)), "squares axis {axis}");
        }
    }
}

#[test]
fn line_shapes_short_long_few_and_many() {
    let ctx = Context::new().unwrap();
    // (dims, axis): contiguous short and long lines, strided with few and many lines.
    for (dims, axis) in [([33usize, 4000, 1], 0usize), ([1 << 20, 3, 1], 0), ([2, 1 << 20, 1], 1), ([1 << 17, 40, 1], 1), ([64, 1 << 12, 1], 1)] {
        let len: usize = dims.iter().product();
        let a = values(len, 6, -1.0, 1.0);
        let wide: Vec<f64> = a.iter().map(|&v| f64::from(v)).collect();
        let x = Array::from_slice_on(&ctx, &a, [dims[0], dims[1]]).unwrap();
        let pending = &x * 3.0;
        let tripled: Vec<f64> = wide.iter().map(|v| v * 3.0).collect();
        let [sum, max] = pending.reduce_axis(axis, [Reduction::Sum, Reduction::Max]);
        let tolerance = 1e-5 * (dims[axis] as f32).sqrt();
        assert_close(&sum.to_vec().unwrap(), &as_f32(&host(&tripled, dims, axis, |a, b| a + b)), 1e-3, tolerance.max(1e-3));
        assert_close(&max.to_vec().unwrap(), &as_f32(&host(&tripled, dims, axis, f64::max)), 1e-6, 1e-6);
        let sum = x.sum_axis(axis);
        assert_close(&sum.to_vec().unwrap(), &as_f32(&host(&wide, dims, axis, |a, b| a + b)), 1e-3, tolerance.max(1e-3));
    }
}

#[test]
fn min_and_max_see_nan_like_the_flat_reductions() {
    let ctx = Context::new().unwrap();
    let a = [1.0f32, f32::NAN, 3.0, 4.0, 5.0, 6.0, 0.5, 8.0];
    let x = Array::from_slice_on(&ctx, &a, [4, 2]).unwrap();
    let pending = &x * 1.0;
    let (axis_min, axis_max) = (pending.min_axis(1).to_vec().unwrap(), pending.max_axis(1).to_vec().unwrap());
    for c in 0..4 {
        let column = Array::from_slice_on(&ctx, &[a[c], a[c + 4]], 2).unwrap();
        let (min, max) = (column.min().item().unwrap(), column.max().item().unwrap());
        assert_eq!(axis_min[c].to_bits(), min.to_bits(), "min column {c}");
        assert_eq!(axis_max[c].to_bits(), max.to_bits(), "max column {c}");
    }
}

#[test]
fn mean_variance_and_standard_deviation() {
    let ctx = Context::new().unwrap();
    for (dims, axis) in [([64usize, 4096], 1usize), ([4096, 64], 0)] {
        let len = dims[0] * dims[1];
        // Far from zero, so that the naive sum of squares minus the squared mean would fail.
        let a: Vec<f32> = values(len, 7, -1.0, 1.0).iter().map(|v| v + 1000.0).collect();
        let x = Array::from_slice_on(&ctx, &a, dims).unwrap();
        let wide: Vec<f64> = a.iter().map(|&v| f64::from(v)).collect();
        let n = dims[axis] as f64;
        let mean = host(&wide, [dims[0], dims[1], 1], axis, |a, b| a + b).iter().map(|s| s / n).collect::<Vec<_>>();
        let stride: usize = dims[..axis].iter().product();
        let var: Vec<f64> = (0..mean.len())
            .map(|l| {
                let base = l % stride + (l / stride) * stride * dims[axis];
                (0..dims[axis]).map(|s| (wide[base + s * stride] - mean[l]).powi(2)).sum::<f64>() / n
            })
            .collect();
        let std: Vec<f64> = var.iter().map(|v| v.sqrt()).collect();
        assert_close(&x.mean_axis(axis).to_vec().unwrap(), &as_f32(&mean), 1e-5, 0.0);
        assert_close(&x.var_axis(axis).to_vec().unwrap(), &as_f32(&var), 1e-5, 0.0);
        assert_close(&x.std_axis(axis).to_vec().unwrap(), &as_f32(&std), 1e-5, 0.0);
    }
}

#[test]
fn var_axis_computes_the_deviation_inside_the_reduction() {
    let ctx = Context::new().unwrap();
    let (cols, rows) = (64usize, 1usize << 16);
    let a = values(cols * rows, 8, -1.0, 1.0);
    let x = Array::from_slice_on(&ctx, &a, [cols, rows]).unwrap();
    x.eval().unwrap();
    ctx.flush().unwrap();
    let (var, profile) = ctx.profile(|| var_forced(&x)).unwrap();
    // The one store kernel is the division of the mean, of 64 elements.
    let stores: u32 = profile.kernels.iter().filter(|k| k.label.starts_with("Fused(")).map(|k| k.launches).sum();
    assert_eq!(stores, 1, "{profile}");
    assert!(profile.kernels.iter().any(|k| k.label.starts_with("FusedAxisReduce")), "{profile}");
    var.check().unwrap();
}

fn var_forced(x: &Array<f32>) -> Array<f32> {
    let var = x.var_axis(1);
    var.eval().unwrap();
    var
}

#[test]
fn bad_arguments_poison_the_results() {
    let ctx = Context::new().unwrap();
    let x = Array::<f32>::zeros_on(&ctx, [4, 4]).unwrap();
    let five = x.reduce_axis(0, [Reduction::Sum, Reduction::Min, Reduction::Max, Reduction::Sum, Reduction::Min]);
    for result in &five {
        assert!(matches!(result.error().map(LumaError::root), Some(LumaError::InvalidShape(_))), "{:?}", result.error());
    }
    let none = x.reduce_axis(0, []);
    assert!(none.is_empty());
    let [bad] = x.reduce_axis(4, [Reduction::Sum]);
    assert!(matches!(bad.error().map(LumaError::root), Some(LumaError::InvalidShape(_))), "{:?}", bad.error());
}

#[test]
fn mask_arrays_reduce_by_min_and_max_only() {
    let ctx = Context::new().unwrap();
    let x = Array::from_slice_on(&ctx, &[1.0f32, -1.0, 2.0, 3.0, -4.0, -5.0], [3, 2]).unwrap();
    let m = x.gt(0.0);
    for result in m.reduce_axis(0, [Reduction::Sum]).iter().chain(m.reduce_axis(0, [Reduction::Min, Reduction::SumOfSquares]).iter()) {
        assert!(matches!(result.error().map(LumaError::root), Some(LumaError::Unsupported(_))), "{:?}", result.error());
    }
    assert!(matches!(m.sum_axis(0).error().map(LumaError::root), Some(LumaError::Unsupported(_))));
    assert!(matches!(m.cumsum(0).error().map(LumaError::root), Some(LumaError::Unsupported(_))));
    let max = m.max_axis(0).to_vec().unwrap();
    assert_eq!(max, vec![luma::Mask::TRUE, luma::Mask::TRUE]);
    let min = m.min_axis(0).to_vec().unwrap();
    assert_eq!(min, vec![luma::Mask::FALSE, luma::Mask::FALSE]);
}
