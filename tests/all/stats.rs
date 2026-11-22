//! Products, variances, weighted means, covariances, norms and the boolean axis reductions
//! against `f64` host computations: edge cases (NaN, infinities, constant and offset data, empty
//! and one-element arrays), every element type a statistic exists for, whole arrays and axes.

use crate::common;

use common::values;
use luma::{Array, Bias, Context, Df32, LumaError, Norm, Reduction};

fn has_f64(ctx: &Context) -> bool {
    ctx.features().contains(luma::Features::SHADER_F64)
}

/// `a` against `b`: relative `rel` with an absolute `abs` floor, NaN equal to NaN.
fn near(got: f64, want: f64, rel: f64, abs: f64, what: &str) {
    if want.is_nan() {
        assert!(got.is_nan(), "{what}: got {got}, want NaN");
    } else if want.is_infinite() {
        assert_eq!(got, want, "{what}");
    } else {
        assert!((got - want).abs() <= abs + rel * want.abs(), "{what}: got {got}, want {want}");
    }
}

fn f64s(v: &[f32]) -> Vec<f64> {
    v.iter().map(|&x| f64::from(x)).collect()
}

/// A tiny deterministic integer generator.
fn ints(n: usize, seed: u64) -> Vec<u32> {
    let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (state >> 33) as u32
        })
        .collect()
}

/// Lengths around the structure of the reduction kernels: one element, odd sizes, a workgroup's
/// block, a few blocks, enough for a second pass.
const LENGTHS: [usize; 9] = [1, 2, 3, 7, 1000, 2047, 2049, 100_003, (1 << 20) + 17];

// --- products ---------------------------------------------------------------------------------

#[test]
fn f32_product_of_every_length() {
    let ctx = Context::new().unwrap();
    for n in LENGTHS {
        let data = values(n, 3, 0.995, 1.005);
        let x = Array::from_slice_on(&ctx, &data, n).unwrap();
        let want: f64 = f64s(&data).iter().product();
        near(f64::from(x.prod().item().unwrap()), want, 3e-5 * (n as f64).sqrt().max(1.0), 0.0, &format!("prod n={n}"));
        assert_eq!(x.product().item().unwrap().to_bits(), x.prod().item().unwrap().to_bits());
    }
}

#[test]
fn f32_product_special_values() {
    let ctx = Context::new().unwrap();
    let case = |data: &[f32]| Array::from_slice_on(&ctx, data, data.len()).unwrap().prod().item().unwrap();
    assert_eq!(case(&[2.0, 3.0, -4.0]), -24.0);
    assert_eq!(case(&[2.0, 0.0, 5.0]), 0.0);
    assert!(case(&[2.0, f32::NAN, 5.0]).is_nan());
    assert!(case(&[0.0, f32::INFINITY]).is_nan());
    assert_eq!(case(&[2.0, f32::INFINITY]), f32::INFINITY);
    assert_eq!(case(&[-2.0, f32::INFINITY]), f32::NEG_INFINITY);
    assert_eq!(case(&[1.0e30, 1.0e30]), f32::INFINITY);
    assert_eq!(case(&[1.0e-30, 1.0e-30]), 0.0);
    assert_eq!(case(&[7.5]), 7.5);
    // A NaN or a zero anywhere in a long array, past the first block and in the last element.
    for n in [3000usize, 70_001] {
        for at in [0, 1, 2047, 2048, n - 1] {
            let mut data = vec![1.0f32; n];
            data[at] = f32::NAN;
            assert!(case(&data).is_nan(), "NaN at {at} of {n}");
            data[at] = 0.0;
            assert_eq!(case(&data), 0.0, "zero at {at} of {n}");
            data[at] = -3.0;
            assert_eq!(case(&data), -3.0, "factor at {at} of {n}");
        }
    }
}

#[test]
fn integer_products_wrap() {
    let ctx = Context::new().unwrap();
    for n in LENGTHS {
        let words: Vec<u32> = ints(n, 5).into_iter().map(|w| w | 1).collect();
        let x = Array::from_slice_on(&ctx, &words, n).unwrap();
        assert_eq!(x.prod().item().unwrap(), words.iter().fold(1u32, |acc, &w| acc.wrapping_mul(w)), "u32 n={n}");
        let signed: Vec<i32> = words.iter().map(|&w| w as i32).collect();
        let y = Array::from_slice_on(&ctx, &signed, n).unwrap();
        assert_eq!(y.prod().item().unwrap(), signed.iter().fold(1i32, |acc, &w| acc.wrapping_mul(w)), "i32 n={n}");
    }
    let x = Array::from_slice_on(&ctx, &[2u32, 3, 4, 0, 9], 5).unwrap();
    assert_eq!(x.prod().item().unwrap(), 0);
    let y = Array::from_slice_on(&ctx, &[-2i32, 3, -4], 3).unwrap();
    assert_eq!(y.prod().item().unwrap(), 24);
}

#[test]
fn product_of_an_expression_and_with_a_scalar_step() {
    let ctx = Context::new().unwrap();
    let n = 50_000;
    let data = values(n, 9, -0.01, 0.01);
    let x = Array::from_slice_on(&ctx, &data, n).unwrap();
    let want: f64 = f64s(&data).iter().map(|v| v + 1.0).product();
    near(f64::from((&x + 1.0).prod().item().unwrap()), want, 1e-4, 0.0, "(x + 1).prod()");
    near(f64::from((&(&x + 1.0).prod() * 2.0).item().unwrap()), 2.0 * want, 1e-4, 0.0, "prod * 2");
    // Together with other reductions of the same expression.
    let [sum, prod] = [(&x + 1.0).sum(), (&x + 1.0).prod()];
    let s: f64 = f64s(&data).iter().map(|v| v + 1.0).sum();
    near(f64::from(sum.item().unwrap()), s, 1e-5, 0.0, "sum next to prod");
    near(f64::from(prod.item().unwrap()), want, 1e-4, 0.0, "prod next to sum");
    let both = x.reduce([Reduction::Sum, Reduction::Product]).to_array::<2>().unwrap();
    let raw_sum: f64 = f64s(&data).iter().sum();
    let raw_prod: f64 = f64s(&data).iter().product();
    near(f64::from(both[0]), raw_sum, 1e-4, 1e-6, "reduce sum");
    near(f64::from(both[1]), raw_prod, 1e-3, 1e-12, "reduce prod");
}

#[test]
fn f64_and_df32_products() {
    let ctx = Context::new().unwrap();
    let data = values(10_001, 4, 0.999, 1.001);
    if has_f64(&ctx) {
        let wide: Vec<f64> = data.iter().map(|&v| f64::from(v) * 1.0000001).collect();
        let x = Array::from_slice_on(&ctx, &wide, wide.len()).unwrap();
        near(x.prod().item().unwrap(), wide.iter().product(), 1e-12, 0.0, "f64 prod");
        let y = Array::from_slice_on(&ctx, &[2.0f64, -3.0, 0.5], 3).unwrap();
        assert_eq!(y.prod().item().unwrap(), -3.0);
    }
    let dd: Vec<Df32> = data.iter().map(|&v| Df32::from(f64::from(v) * 1.0000001)).collect();
    let x = Array::from_slice_on(&ctx, &dd, dd.len()).unwrap();
    let want: f64 = dd.iter().map(|d| d.to_f64()).product();
    near(x.prod().item().unwrap().to_f64(), want, 1e-11, 0.0, "Df32 prod");
}

#[test]
fn product_along_axes() {
    let ctx = Context::new().unwrap();
    for dims in [[37usize, 5, 3], [1, 100, 1], [3000, 2, 1], [5, 3000, 1], [4, 3, 17], [2049, 1, 3]] {
        let n: usize = dims.iter().product();
        let data = values(n, 11, 0.9, 1.1);
        let x = Array::from_slice_on(&ctx, &data, dims).unwrap();
        for axis in 0..3 {
            let got = x.prod_axis(axis);
            let mut out_dims = dims;
            out_dims[axis] = 1;
            assert_eq!(got.shape().dims()[..3], out_dims, "shape of axis {axis}");
            let stride: usize = dims[..axis].iter().product();
            let (extent, lines) = (dims[axis], n / dims[axis]);
            let want: Vec<f64> = (0..lines)
                .map(|l| {
                    let base = l % stride + (l / stride) * stride * extent;
                    (0..extent).map(|s| f64::from(data[base + s * stride])).product()
                })
                .collect();
            let got = got.to_vec().unwrap();
            for (l, (g, w)) in got.iter().zip(&want).enumerate() {
                near(f64::from(*g), *w, 2e-5 * (extent as f64).sqrt().max(1.0), 0.0, &format!("{dims:?} axis {axis} line {l}"));
            }
        }
    }
    // Integers, an expression operand and several reductions of one pass.
    let words: Vec<u32> = ints(60, 2).into_iter().map(|w| w % 7 + 1).collect();
    let x = Array::from_slice_on(&ctx, &words, [6, 10]).unwrap();
    let got = x.prod_axis(0).to_vec().unwrap();
    for (l, g) in got.iter().enumerate() {
        assert_eq!(*g, words[l * 6..(l + 1) * 6].iter().fold(1u32, |a, &w| a.wrapping_mul(w)), "u32 line {l}");
    }
    let data = values(6 * 10, 3, 0.5, 1.5);
    let y = Array::from_slice_on(&ctx, &data, [6, 10]).unwrap();
    let [sum, prod] = (&y * 2.0).reduce_axis(1, [Reduction::Sum, Reduction::Product]);
    let (sum, prod) = (sum.to_vec().unwrap(), prod.to_vec().unwrap());
    for i in 0..6 {
        let column: Vec<f64> = (0..10).map(|j| 2.0 * f64::from(data[i + 6 * j])).collect();
        near(f64::from(sum[i]), column.iter().sum(), 1e-5, 0.0, "sum column");
        near(f64::from(prod[i]), column.iter().product(), 1e-4, 0.0, "product column");
    }
}

#[test]
fn empty_product_is_one() {
    let ctx = Context::new().unwrap();
    // An empty array is an error or the neutral value.
    if let Ok(empty) = Array::<f32>::zeros_on(&ctx, 0)
        && let Ok(value) = empty.prod().item()
    {
        assert_eq!(value, 1.0);
    }
    if let Ok(empty) = Array::<u32>::zeros_on(&ctx, 0)
        && let Ok(value) = empty.prod().item()
    {
        assert_eq!(value, 1);
    }
}

#[test]
fn poisoned_operands_poison_the_product() {
    let ctx = Context::new().unwrap();
    let a = Array::from_slice_on(&ctx, &[1.0f32, 2.0], 2).unwrap();
    let b = Array::from_slice_on(&ctx, &[1.0f32, 2.0, 3.0], 3).unwrap();
    let bad = &a + &b;
    assert!(matches!(bad.prod().item(), Err(LumaError::Op { .. })));
}

// --- variance, mean, weights --------------------------------------------------------------------

fn host_var(data: &[f64], ddof: f64) -> f64 {
    let n = data.len() as f64;
    let mean = data.iter().sum::<f64>() / n;
    data.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / (n - ddof)
}

#[test]
fn whole_array_variance_and_std() {
    let ctx = Context::new().unwrap();
    for n in LENGTHS {
        for (offset, scale) in [(0.0f32, 1.0f32), (1.0e4, 1.0), (-3.0, 100.0)] {
            let data: Vec<f32> = values(n, 21, -1.0, 1.0).iter().map(|v| offset + scale * v).collect();
            let x = Array::from_slice_on(&ctx, &data, n).unwrap();
            let wide = f64s(&data);
            for (bias, ddof) in [(Bias::Population, 0.0), (Bias::Sample, 1.0)] {
                let want = host_var(&wide, ddof);
                let tag = format!("n={n} offset={offset} scale={scale} {bias:?}");
                let got_var = x.var(bias).item().unwrap();
                let got_std = x.std(bias).item().unwrap();
                if n == 1 && ddof == 1.0 {
                    assert!(got_var.is_nan() && got_std.is_nan(), "{tag}: single element, sample");
                    continue;
                }
                near(f64::from(got_var), want, 2e-3, 1e-7 * f64::from(scale * scale), &format!("var {tag}"));
                near(f64::from(got_std), want.sqrt(), 1e-3, 1e-4 * f64::from(scale), &format!("std {tag}"));
            }
        }
    }
}

#[test]
fn variance_edge_cases() {
    let ctx = Context::new().unwrap();
    let var = |data: &[f32], bias| Array::from_slice_on(&ctx, data, data.len()).unwrap().var(bias).item().unwrap();
    // Constant data has variance exactly 0, also far from zero.
    assert_eq!(var(&[5.0; 3000], Bias::Population), 0.0);
    assert_eq!(var(&[-1.0e6; 777], Bias::Sample), 0.0);
    assert_eq!(var(&[3.0], Bias::Population), 0.0);
    assert!(var(&[3.0, f32::NAN, 1.0], Bias::Population).is_nan());
    assert!(var(&[3.0, f32::INFINITY, 1.0], Bias::Sample).is_nan());
    assert!(var(&[f32::INFINITY, 1.0], Bias::Population).is_nan());
    near(f64::from(var(&[1.0, 2.0, 3.0, 4.0], Bias::Population)), 1.25, 1e-6, 0.0, "population");
    near(f64::from(var(&[1.0, 2.0, 3.0, 4.0], Bias::Sample)), 5.0 / 3.0, 1e-6, 0.0, "sample");
    assert_eq!(Bias::default(), Bias::Population);
    // The population variance along an axis is the existing `var_axis`.
    let data = values(6 * 40, 13, 0.0, 10.0);
    let x = Array::from_slice_on(&ctx, &data, [6, 40]).unwrap();
    for axis in 0..2 {
        let a = x.var_axis(axis).to_vec().unwrap();
        let b = x.var_axis_bias(axis, Bias::Population).to_vec().unwrap();
        assert_eq!(a, b, "axis {axis}");
    }
}

#[test]
fn variance_of_an_expression_and_with_other_reductions() {
    let ctx = Context::new().unwrap();
    let n = 70_000;
    let data = values(n, 31, -2.0, 2.0);
    let x = Array::from_slice_on(&ctx, &data, n).unwrap();
    let shifted: Vec<f64> = f64s(&data).iter().map(|v| 3.0 * v + 7.0).collect();
    let expression = &(&x * 3.0) + 7.0;
    near(f64::from(expression.var(Bias::Sample).item().unwrap()), host_var(&shifted, 1.0), 2e-3, 0.0, "var of an expression");
    let (mean, var) = expression.meanvar(Bias::Population);
    near(f64::from(mean.item().unwrap()), shifted.iter().sum::<f64>() / n as f64, 1e-5, 0.0, "meanvar mean");
    near(f64::from(var.item().unwrap()), host_var(&shifted, 0.0), 2e-3, 0.0, "meanvar var");
}

#[test]
fn axis_variance_with_bias_and_meanvar() {
    let ctx = Context::new().unwrap();
    for dims in [[40usize, 7, 3], [1, 50, 1], [5, 3000, 1], [3000, 2, 2]] {
        let n: usize = dims.iter().product();
        let data = values(n, 41, -5.0, 9.0);
        let x = Array::from_slice_on(&ctx, &data, dims).unwrap();
        for axis in 0..3 {
            let stride: usize = dims[..axis].iter().product();
            let (extent, lines) = (dims[axis], n / dims[axis]);
            let line = |l: usize| -> Vec<f64> {
                let base = l % stride + (l / stride) * stride * extent;
                (0..extent).map(|s| f64::from(data[base + s * stride])).collect()
            };
            for (bias, ddof) in [(Bias::Population, 0.0), (Bias::Sample, 1.0)] {
                let var = x.var_axis_bias(axis, bias).to_vec().unwrap();
                let std = x.std_axis_bias(axis, bias).to_vec().unwrap();
                let (m, v) = x.meanvar_axis(axis, bias);
                let (m, v) = (m.to_vec().unwrap(), v.to_vec().unwrap());
                assert_eq!(var.len(), lines);
                for l in 0..lines {
                    let values = line(l);
                    let tag = format!("{dims:?} axis {axis} line {l} {bias:?}");
                    if extent == 1 && ddof == 1.0 {
                        assert!(var[l].is_nan(), "{tag}");
                        continue;
                    }
                    near(f64::from(var[l]), host_var(&values, ddof), 2e-3, 1e-5, &format!("var {tag}"));
                    near(f64::from(std[l]), host_var(&values, ddof).sqrt(), 1e-3, 1e-4, &format!("std {tag}"));
                    near(f64::from(m[l]), values.iter().sum::<f64>() / extent as f64, 1e-4, 1e-4, &format!("mean {tag}"));
                    near(f64::from(v[l]), host_var(&values, ddof), 2e-3, 1e-5, &format!("meanvar {tag}"));
                }
            }
        }
    }
}

#[test]
fn weighted_means_and_variances() {
    let ctx = Context::new().unwrap();
    for n in [1usize, 2, 5, 1000, 2049, 100_003] {
        let data = values(n, 51, -4.0, 6.0);
        let weights = values(n, 52, 0.1, 3.0);
        let (x, w) = (Array::from_slice_on(&ctx, &data, n).unwrap(), Array::from_slice_on(&ctx, &weights, n).unwrap());
        let (d, wd) = (f64s(&data), f64s(&weights));
        let s0: f64 = wd.iter().sum();
        let mean: f64 = d.iter().zip(&wd).map(|(x, w)| x * w).sum::<f64>() / s0;
        near(f64::from(x.mean_weighted(&w).item().unwrap()), mean, 2e-4, 1e-5, &format!("mean_weighted n={n}"));
        let var_sum: f64 = d.iter().zip(&wd).map(|(x, w)| w * (x - mean) * (x - mean)).sum();
        for (bias, ddof) in [(Bias::Population, 0.0), (Bias::Sample, 1.0)] {
            if n == 1 {
                continue;
            }
            let (m, v) = x.meanvar_weighted(&w, bias);
            near(f64::from(m.item().unwrap()), mean, 2e-4, 1e-5, &format!("meanvar_weighted mean n={n}"));
            near(f64::from(v.item().unwrap()), var_sum / (s0 - ddof), 5e-3, 1e-4, &format!("meanvar_weighted var n={n} {bias:?}"));
        }
        // Weights of all ones are the unweighted statistics.
        let ones = Array::<f32>::full_on(&ctx, n, 1.0).unwrap();
        near(f64::from(x.mean_weighted(&ones).item().unwrap()), d.iter().sum::<f64>() / n as f64, 1e-4, 1e-5, "unit weights");
    }
}

#[test]
fn weighted_means_along_axes_and_broadcast_weights() {
    let ctx = Context::new().unwrap();
    let dims = [6usize, 9, 2];
    let n: usize = dims.iter().product();
    let data = values(n, 61, 0.0, 10.0);
    let weights = values(n, 62, 0.5, 2.0);
    let (x, w) = (Array::from_slice_on(&ctx, &data, dims).unwrap(), Array::from_slice_on(&ctx, &weights, dims).unwrap());
    for axis in 0..3 {
        let stride: usize = dims[..axis].iter().product();
        let (extent, lines) = (dims[axis], n / dims[axis]);
        let got = x.mean_axis_weighted(axis, &w).to_vec().unwrap();
        let (m, v) = x.meanvar_axis_weighted(axis, &w, Bias::Population);
        let (m, v) = (m.to_vec().unwrap(), v.to_vec().unwrap());
        for l in 0..lines {
            let base = l % stride + (l / stride) * stride * extent;
            let (mut s0, mut s1) = (0.0f64, 0.0f64);
            for s in 0..extent {
                let (xv, wv) = (f64::from(data[base + s * stride]), f64::from(weights[base + s * stride]));
                s0 += wv;
                s1 += wv * xv;
            }
            let mean = s1 / s0;
            let var: f64 = (0..extent).map(|s| f64::from(weights[base + s * stride]) * (f64::from(data[base + s * stride]) - mean).powi(2)).sum::<f64>() / s0;
            near(f64::from(got[l]), mean, 1e-4, 1e-5, &format!("axis {axis} line {l}"));
            near(f64::from(m[l]), mean, 1e-4, 1e-5, &format!("axis {axis} line {l} meanvar mean"));
            near(f64::from(v[l]), var, 2e-3, 1e-4, &format!("axis {axis} line {l} meanvar var"));
        }
    }
    // Weights of extent 1 along an axis are broadcast.
    let row = Array::from_slice_on(&ctx, &values(6, 63, 1.0, 2.0), [6, 1, 1]).unwrap();
    let got = x.mean_axis_weighted(1, &row).to_vec().unwrap();
    assert_eq!(got.len(), 6 * 2);
    // Weights that do not fit poison the result.
    let bad = Array::from_slice_on(&ctx, &values(5, 64, 1.0, 2.0), 5).unwrap();
    assert!(matches!(x.mean_weighted(&bad).item(), Err(LumaError::Op { .. })));
    assert!(matches!(x.meanvar_weighted(&bad, Bias::Sample).1.item(), Err(LumaError::Op { .. })));
}

#[test]
fn f64_statistics() {
    let ctx = Context::new().unwrap();
    if !has_f64(&ctx) {
        return;
    }
    let n = 50_001;
    let data: Vec<f64> = values(n, 71, 0.0, 1.0).iter().map(|&v| 1.0e6 + f64::from(v)).collect();
    let other: Vec<f64> = values(n, 72, 0.0, 1.0).iter().zip(&data).map(|(&v, d)| 0.5 * (d - 1.0e6) + f64::from(v)).collect();
    let (x, y) = (Array::from_slice_on(&ctx, &data, n).unwrap(), Array::from_slice_on(&ctx, &other, n).unwrap());
    let mean = data.iter().sum::<f64>() / n as f64;
    near(x.mean().item().unwrap(), mean, 1e-13, 0.0, "f64 mean");
    near(x.var(Bias::Sample).item().unwrap(), host_var(&data, 1.0), 1e-8, 0.0, "f64 var");
    near(x.std(Bias::Population).item().unwrap(), host_var(&data, 0.0).sqrt(), 1e-9, 0.0, "f64 std");
    let (m, v) = x.meanvar(Bias::Population);
    near(m.item().unwrap(), mean, 1e-13, 0.0, "f64 meanvar mean");
    near(v.item().unwrap(), host_var(&data, 0.0), 1e-8, 0.0, "f64 meanvar var");
    let my = other.iter().sum::<f64>() / n as f64;
    let cov: f64 = data.iter().zip(&other).map(|(a, b)| (a - mean) * (b - my)).sum::<f64>() / (n - 1) as f64;
    near(x.cov(&y, Bias::Sample).item().unwrap(), cov, 1e-8, 0.0, "f64 cov");
    let corr = cov * (n - 1) as f64 / ((host_var(&data, 0.0) * n as f64) * (host_var(&other, 0.0) * n as f64)).sqrt();
    near(x.corrcoef(&y).item().unwrap(), corr, 1e-8, 0.0, "f64 corrcoef");
    near(x.norm_p(Norm::One).item().unwrap(), data.iter().sum::<f64>(), 1e-12, 0.0, "f64 L1");
    near(x.norm_p(Norm::Two).item().unwrap(), data.iter().map(|v| v * v).sum::<f64>().sqrt(), 1e-12, 0.0, "f64 L2");
    near(x.norm_p(Norm::Inf).item().unwrap(), data.iter().copied().fold(0.0, f64::max), 0.0, 0.0, "f64 Linf");
    assert!(matches!(x.norm_p(Norm::P(3.0)).item(), Err(LumaError::Op { .. })));
    near(x.norm_p(Norm::P(2.0)).item().unwrap(), data.iter().map(|v| v * v).sum::<f64>().sqrt(), 1e-12, 0.0, "f64 P(2)");
    let ws = Array::from_slice_on(&ctx, &vec![2.0f64; n], n).unwrap();
    near(x.mean_weighted(&ws).item().unwrap(), mean, 1e-13, 0.0, "f64 weighted mean");
    // Reductions along an axis are not available for f64: the result says so.
    assert!(matches!(x.prod_axis(0).item(), Err(LumaError::Op { .. })));
}

// --- covariance, correlation --------------------------------------------------------------------

fn host_cov(x: &[f64], y: &[f64], ddof: f64) -> f64 {
    let n = x.len() as f64;
    let (mx, my) = (x.iter().sum::<f64>() / n, y.iter().sum::<f64>() / n);
    x.iter().zip(y).map(|(a, b)| (a - mx) * (b - my)).sum::<f64>() / (n - ddof)
}

#[test]
fn covariance_and_correlation() {
    let ctx = Context::new().unwrap();
    for n in [2usize, 3, 1000, 2049, 100_003, (1 << 20) + 17] {
        let a = values(n, 81, -3.0, 3.0);
        let noise = values(n, 82, -1.0, 1.0);
        let b: Vec<f32> = a.iter().zip(&noise).map(|(a, e)| 2.0 * a + 100.0 + e).collect();
        let (x, y) = (Array::from_slice_on(&ctx, &a, n).unwrap(), Array::from_slice_on(&ctx, &b, n).unwrap());
        let (wa, wb) = (f64s(&a), f64s(&b));
        for (bias, ddof) in [(Bias::Population, 0.0), (Bias::Sample, 1.0)] {
            near(f64::from(x.cov(&y, bias).item().unwrap()), host_cov(&wa, &wb, ddof), 2e-3, 1e-4, &format!("cov n={n} {bias:?}"));
        }
        let corr = host_cov(&wa, &wb, 0.0) / (host_var(&wa, 0.0) * host_var(&wb, 0.0)).sqrt();
        near(f64::from(x.corrcoef(&y).item().unwrap()), corr, 2e-3, 1e-4, &format!("corrcoef n={n}"));
        near(f64::from(x.corrcoef(&x).item().unwrap()), 1.0, 0.0, 1e-4, "self correlation");
        let neg = &x * -1.0;
        near(f64::from(x.corrcoef(&neg).item().unwrap()), -1.0, 0.0, 1e-4, "anti correlation");
        near(f64::from(x.cov(&x, Bias::Population).item().unwrap()), host_var(&wa, 0.0), 2e-3, 1e-4, "cov with itself is the variance");
    }
    // A constant array has no correlation.
    let c = Array::<f32>::full_on(&ctx, 100, 4.0).unwrap();
    let v = Array::from_slice_on(&ctx, &values(100, 83, 0.0, 1.0), 100).unwrap();
    assert!(c.corrcoef(&v).item().unwrap().is_nan());
    assert_eq!(c.cov(&v, Bias::Population).item().unwrap(), 0.0);
    let nan = Array::from_slice_on(&ctx, &[1.0f32, f32::NAN, 3.0], 3).unwrap();
    let ok = Array::from_slice_on(&ctx, &[1.0f32, 2.0, 3.0], 3).unwrap();
    assert!(nan.cov(&ok, Bias::Sample).item().unwrap().is_nan());
    // Different shapes poison the result.
    let short = Array::from_slice_on(&ctx, &[1.0f32, 2.0], 2).unwrap();
    assert!(matches!(ok.cov(&short, Bias::Sample).item(), Err(LumaError::Op { .. })));
    assert!(matches!(ok.corrcoef(&short).item(), Err(LumaError::Op { .. })));
}

#[test]
fn covariance_along_axes() {
    let ctx = Context::new().unwrap();
    let dims = [7usize, 11, 3];
    let n: usize = dims.iter().product();
    let a = values(n, 91, 0.0, 5.0);
    let b = values(n, 92, -2.0, 2.0);
    let (x, y) = (Array::from_slice_on(&ctx, &a, dims).unwrap(), Array::from_slice_on(&ctx, &b, dims).unwrap());
    for axis in 0..3 {
        let stride: usize = dims[..axis].iter().product();
        let (extent, lines) = (dims[axis], n / dims[axis]);
        let cov = x.cov_axis(axis, &y, Bias::Sample).to_vec().unwrap();
        let corr = x.corrcoef_axis(axis, &y).to_vec().unwrap();
        assert_eq!(cov.len(), lines);
        for l in 0..lines {
            let base = l % stride + (l / stride) * stride * extent;
            let xs: Vec<f64> = (0..extent).map(|s| f64::from(a[base + s * stride])).collect();
            let ys: Vec<f64> = (0..extent).map(|s| f64::from(b[base + s * stride])).collect();
            near(f64::from(cov[l]), host_cov(&xs, &ys, 1.0), 2e-3, 1e-4, &format!("axis {axis} line {l}"));
            let want = host_cov(&xs, &ys, 0.0) / (host_var(&xs, 0.0) * host_var(&ys, 0.0)).sqrt();
            near(f64::from(corr[l]), want, 2e-3, 1e-4, &format!("corr axis {axis} line {l}"));
        }
    }
}

// --- norms --------------------------------------------------------------------------------------

#[test]
fn vector_norms() {
    let ctx = Context::new().unwrap();
    for n in LENGTHS {
        let data = values(n, 101, -2.0, 1.5);
        let x = Array::from_slice_on(&ctx, &data, n).unwrap();
        let w = f64s(&data);
        let l1: f64 = w.iter().map(|v| v.abs()).sum();
        let l2: f64 = w.iter().map(|v| v * v).sum::<f64>().sqrt();
        let linf = w.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        near(f64::from(x.norm_p(Norm::One).item().unwrap()), l1, 3e-5, 0.0, &format!("L1 n={n}"));
        near(f64::from(x.norm_p(Norm::Two).item().unwrap()), l2, 3e-5, 0.0, &format!("L2 n={n}"));
        near(f64::from(x.norm_p(Norm::Frobenius).item().unwrap()), l2, 3e-5, 0.0, &format!("Frobenius n={n}"));
        assert_eq!(f64::from(x.norm_p(Norm::Inf).item().unwrap()), linf, "Linf n={n}");
        assert_eq!(x.norm_p(Norm::Two).item().unwrap().to_bits(), x.norm().item().unwrap().to_bits());
        for p in [1.5f32, 3.0] {
            let want = w.iter().map(|v| v.abs().powf(f64::from(p))).sum::<f64>().powf(1.0 / f64::from(p));
            near(f64::from(x.norm_p(Norm::P(p)).item().unwrap()), want, 2e-4, 0.0, &format!("P({p}) n={n}"));
        }
        near(f64::from(x.norm_p(Norm::P(1.0)).item().unwrap()), l1, 3e-5, 0.0, "P(1)");
        near(f64::from(x.norm_p(Norm::P(2.0)).item().unwrap()), l2, 3e-5, 0.0, "P(2)");
        assert_eq!(f64::from(x.norm_p(Norm::P(f32::INFINITY)).item().unwrap()), linf, "P(inf)");
    }
}

#[test]
fn norm_special_values() {
    let ctx = Context::new().unwrap();
    let norm = |data: &[f32], norm| Array::from_slice_on(&ctx, data, data.len()).unwrap().norm_p(norm).item().unwrap();
    for kind in [Norm::One, Norm::Two, Norm::Inf, Norm::Frobenius, Norm::P(3.0)] {
        assert_eq!(norm(&[0.0; 5], kind), 0.0, "{kind:?} of zeros");
        assert!(norm(&[1.0, f32::NAN, -2.0], kind).is_nan(), "{kind:?} with a NaN");
        near(f64::from(norm(&[-3.5], kind)), 3.5, 1e-6, 0.0, &format!("{kind:?} of one element"));
    }
    assert_eq!(norm(&[3.0, f32::NEG_INFINITY], Norm::One), f32::INFINITY);
    assert_eq!(norm(&[3.0, f32::NEG_INFINITY], Norm::Two), f32::INFINITY);
    assert_eq!(norm(&[3.0, f32::NEG_INFINITY], Norm::Inf), f32::INFINITY);
    assert_eq!(norm(&[3.0, -4.0], Norm::Two), 5.0);
    assert_eq!(norm(&[3.0, -4.0], Norm::One), 7.0);
    assert_eq!(norm(&[3.0, -4.0], Norm::Inf), 4.0);
    // A norm of an expression and the arithmetic on a norm.
    let x = Array::from_slice_on(&ctx, &[1.0f32, -2.0, 3.0], 3).unwrap();
    assert_eq!((&x * 2.0).norm_p(Norm::One).item().unwrap(), 12.0);
    assert_eq!((&x.norm_p(Norm::One) / 2.0).item().unwrap(), 3.0);
}

#[test]
fn matrix_norms() {
    let ctx = Context::new().unwrap();
    // Axis 0 indexes the rows: the array of shape [rows, cols].
    for (rows, cols) in [(3usize, 4usize), (50, 7), (1, 9), (9, 1), (1000, 33)] {
        let data = values(rows * cols, 111, -3.0, 2.0);
        let x = Array::from_slice_on(&ctx, &data, [rows, cols]).unwrap();
        let at = |r: usize, c: usize| f64::from(data[r + rows * c]).abs();
        let one = (0..cols).map(|c| (0..rows).map(|r| at(r, c)).sum::<f64>()).fold(0.0, f64::max);
        let inf = (0..rows).map(|r| (0..cols).map(|c| at(r, c)).sum::<f64>()).fold(0.0, f64::max);
        let fro = data.iter().map(|&v| f64::from(v) * f64::from(v)).sum::<f64>().sqrt();
        near(f64::from(x.norm_p(Norm::MatrixOne).item().unwrap()), one, 3e-5, 0.0, &format!("matrix 1 {rows}x{cols}"));
        near(f64::from(x.norm_p(Norm::MatrixInf).item().unwrap()), inf, 3e-5, 0.0, &format!("matrix inf {rows}x{cols}"));
        near(f64::from(x.norm_p(Norm::Frobenius).item().unwrap()), fro, 3e-5, 0.0, &format!("frobenius {rows}x{cols}"));
    }
    // More than two dimensions have no matrix norm.
    let cube = Array::<f32>::full_on(&ctx, [2, 2, 2], 1.0).unwrap();
    assert!(matches!(cube.norm_p(Norm::MatrixOne).item(), Err(LumaError::Op { .. })));
    assert!(matches!(cube.norm_p(Norm::MatrixInf).item(), Err(LumaError::Op { .. })));
    assert_eq!(cube.norm_p(Norm::Frobenius).item().unwrap(), 8.0f32.sqrt());
    let nan = Array::from_slice_on(&ctx, &[1.0f32, f32::NAN, 2.0, 3.0], [2, 2]).unwrap();
    assert!(nan.norm_p(Norm::MatrixOne).item().unwrap().is_nan());
}

// --- boolean axis reductions -------------------------------------------------------------------

#[test]
fn boolean_axis_reductions() {
    let ctx = Context::new().unwrap();
    for dims in [[9usize, 5, 3], [1, 40, 1], [3000, 4, 1], [33, 1, 7], [2, 3, 5]] {
        let n: usize = dims.iter().product();
        for density in [0u32, 1, 3, 100] {
            // `density` 0: all false, 100: all true, in between: pseudo-random.
            let bits: Vec<bool> = ints(n, 7 + u64::from(density)).iter().map(|&w| density == 100 || (density != 0 && w % (density + 1) == 0)).collect();
            let words: Vec<u32> = bits.iter().map(|&b| u32::from(b)).collect();
            let from_words = Array::from_slice_on(&ctx, &words, dims).unwrap();
            let evaluated = from_words.ne(0u32);
            evaluated.eval().unwrap();
            // The same bits as a lazy expression.
            let lazy = from_words.gt(0u32);
            for x in [&evaluated, &lazy] {
                for axis in 0..3 {
                    let stride: usize = dims[..axis].iter().product();
                    let (extent, lines) = (dims[axis], n / dims[axis]);
                    let (count, all, any) = (x.count_axis(axis).to_vec().unwrap(), x.all_axis(axis).to_vec().unwrap(), x.any_axis(axis).to_vec().unwrap());
                    assert_eq!((count.len(), all.len(), any.len()), (lines, lines, lines));
                    for l in 0..lines {
                        let base = l % stride + (l / stride) * stride * extent;
                        let line: Vec<bool> = (0..extent).map(|s| bits[base + s * stride]).collect();
                        let tag = format!("{dims:?} axis {axis} line {l} density {density}");
                        assert_eq!(count[l], line.iter().filter(|&&b| b).count() as u32, "count {tag}");
                        assert_eq!(all[l], line.iter().all(|&b| b), "all {tag}");
                        assert_eq!(any[l], line.iter().any(|&b| b), "any {tag}");
                    }
                    let mut out = dims;
                    out[axis] = 1;
                    assert_eq!(x.all_axis(axis).shape().dims()[..3], out);
                }
            }
        }
    }
}
