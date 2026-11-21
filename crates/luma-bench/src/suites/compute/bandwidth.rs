//! Memory-bound work: elementwise copies and chains at several sizes and precisions, transposes
//! and full reductions.

use luma::*;

use super::{check_sampled, host, pick, upload};
use crate::harness::*;

pub fn run(h: &mut Harness) {
    elementwise(h);
    wide_types(h);
    transpose(h);
    reductions(h);
}

/// A chain of `m` cheap operations (multiply, add, minimum, subtract) with varying constants,
/// all fused into one kernel by the JIT.
fn chain(x: &Array<f32>, m: usize) -> Array<f32> {
    let mut z = x.clone();
    for i in 0..m {
        z = match i % 4 {
            0 => &z * (0.999 - 0.0001 * (i % 7) as f32),
            1 => &z + 0.01 * (1 + i % 5) as f32,
            2 => z.minimum(3.0),
            _ => &z - 0.005 * (1 + i % 3) as f32,
        };
    }
    z
}

fn chain_host(mut v: f32, m: usize) -> f32 {
    for i in 0..m {
        v = match i % 4 {
            0 => v * (0.999 - 0.0001 * (i % 7) as f32),
            1 => v + 0.01 * (1 + i % 5) as f32,
            2 => v.min(3.0),
            _ => v - 0.005 * (1 + i % 3) as f32,
        };
    }
    v
}

fn elementwise(h: &mut Harness) {
    h.guarded("ew", |h| {
        let cpu = h.ctx.is_cpu();
        let max_n: usize = if cpu { pick(h, 1 << 22, 1 << 24) } else { pick(h, 1 << 24, 1 << 26) };
        let xs = values(max_n, 1, 0.5, 1.5);
        let ys = values(max_n, 2, -1.0, 1.0);

        // Copy and saxpy from launch-bound to bandwidth-bound sizes.
        let sizes: Vec<usize> = [1usize << 12, 1 << 16, 1 << 20, 1 << 24, 1 << 26].into_iter().filter(|&n| n <= max_n).collect();
        for &n in &sizes {
            let x = upload(h, &xs[..n], n);
            let y = upload(h, &ys[..n], n);
            check_sampled("ew/saxpy", &host(&(&x * 2.5 + &y)), |i| f64::from(xs[i]) * 2.5 + f64::from(ys[i]), (n / 65536).max(1) | 1, 1e-6, 1e-6);
            h.case("ew/saxpy", &format!("{n}"), Work::Bytes(12.0 * n as f64), &mut || (&x * 2.5 + &y).eval().unwrap());
            if n >= 1 << 20 {
                h.case("ew/copy", &format!("{n}"), Work::Bytes(8.0 * n as f64), &mut || x.copy().eval().unwrap());
            }
        }

        // Everything below runs at the largest size, where it should be bandwidth-bound; results
        // are checked at a strided sample of the whole array.
        let n = max_n;
        let x = upload(h, &xs[..n], n);
        let y = upload(h, &ys[..n], n);
        let sample = 61;

        // In-place update of an existing buffer.
        let mut z0 = upload(h, &ys[..n], n);
        z0 += &(&x * 0.5);
        check_sampled("ew/axpy_inplace", &host(&z0), |i| f64::from(ys[i]) + f64::from(xs[i]) * 0.5, sample, 1e-6, 1e-6);
        drop(z0);
        let mut z = upload(h, &ys[..n], n);
        h.case("ew/axpy_inplace", &format!("{n}"), Work::Bytes(12.0 * n as f64), &mut || {
            z += &(&x * 0.5);
            z.eval().unwrap();
        });
        drop(z);

        // Fused chains of 2..64 operations: the cost of a fused kernel should stay one pass.
        for m in [2usize, 4, 8, 16, 32, 64] {
            check_sampled("ew/chain", &host(&chain(&x, m)), |i| f64::from(chain_host(xs[i], m)), sample, 1e-4, 1e-4);
            h.case(&format!("ew/chain{m}"), &format!("{n}"), Work::Bytes(8.0 * n as f64), &mut || chain(&x, m).eval().unwrap());
        }

        // Transcendentals: compute-bound.
        let sins = |x: &Array<f32>| {
            let mut z = x.clone();
            for _ in 0..16 {
                z = z.sin() * 0.99 + 0.01;
            }
            z
        };
        let sin_host = |i: usize| {
            let mut z = f64::from(xs[i]);
            for _ in 0..16 {
                z = z.sin() * 0.99 + 0.01;
            }
            z
        };
        check_sampled("ew/sin16", &host(&sins(&x)), sin_host, sample, 1e-4, 1e-4);
        h.case("ew/sin16", &format!("{n}"), Work::Items(16.0 * n as f64, "Msin/s"), &mut || sins(&x).eval().unwrap());

        let mix = |x: &Array<f32>, y: &Array<f32>| (x.exp() + y.abs().sqrt() + 1.0).ln().tanh();
        let mix_host = |i: usize| (f64::from(xs[i]).exp() + f64::from(ys[i]).abs().sqrt() + 1.0).ln().tanh();
        check_sampled("ew/exp_ln_tanh", &host(&mix(&x, &y)), mix_host, sample, 1e-4, 1e-5);
        h.case("ew/exp_ln_tanh", &format!("{n}"), Work::Bytes(12.0 * n as f64), &mut || mix(&x, &y).eval().unwrap());

        // Comparison and select (a leaky ReLU), the bool path.
        let leaky = |y: &Array<f32>| select(&y.gt(0.0), y, &(y * 0.01));
        let leaky_host = |i: usize| if ys[i] > 0.0 { f64::from(ys[i]) } else { f64::from(ys[i]) * 0.01 };
        check_sampled("ew/select", &host(&leaky(&y)), leaky_host, sample, 1e-6, 1e-7);
        h.case("ew/select", &format!("{n}"), Work::Bytes(8.0 * n as f64), &mut || leaky(&y).eval().unwrap());
    });
}

/// `f64` and double-float (`Df32`) arrays: twice the bytes, and for `Df32` several times the
/// arithmetic.
fn wide_types(h: &mut Harness) {
    let has_f64 = h.ctx.features().contains(Features::SHADER_F64);
    h.guarded("wide", |h| {
        let ctx = h.ctx.clone();
        let n: usize = if h.ctx.is_cpu() { pick(h, 1 << 20, 1 << 22) } else { pick(h, 1 << 22, 1 << 24) };
        let xs: Vec<f64> = values(n, 11, 0.5, 1.5).iter().map(|&v| f64::from(v) + f64::from(v) * 1e-9).collect();
        let ys: Vec<f64> = values(n, 12, -1.0, 1.0).iter().map(|&v| f64::from(v) * 1.0000001).collect();
        let chain16_host = |mut v: f64| {
            for i in 0..16usize {
                v = match i % 4 {
                    0 => v * (0.999 - 0.0001 * (i % 7) as f64),
                    1 => v + 0.01 * (1 + i % 5) as f64,
                    2 => v.abs() * 0.5,
                    _ => v - 0.005 * (1 + i % 3) as f64,
                };
            }
            v
        };
        let saxpy_want: Vec<f64> = (0..n).map(|i| xs[i] * 2.5 + ys[i]).collect();
        let chain_want: Vec<f64> = xs.iter().map(|&v| chain16_host(v)).collect();

        if has_f64 {
            let x = Array::<f64>::from_slice_on(&ctx, &xs, n).unwrap();
            let y = Array::<f64>::from_slice_on(&ctx, &ys, n).unwrap();
            x.eval().unwrap();
            y.eval().unwrap();
            let chain16 = |x: &Array<f64>| {
                let mut z = x.clone();
                for i in 0..16usize {
                    z = match i % 4 {
                        0 => &z * (0.999 - 0.0001 * (i % 7) as f64),
                        1 => &z + 0.01 * (1 + i % 5) as f64,
                        2 => z.abs() * 0.5,
                        _ => &z - 0.005 * (1 + i % 3) as f64,
                    };
                }
                z
            };
            let got = (&x * 2.5 + &y).to_vec().unwrap();
            assert!(got.iter().zip(&saxpy_want).all(|(g, w)| (g - w).abs() <= 1e-13 * w.abs().max(1.0)), "wide/saxpy_f64");
            let got = chain16(&x).to_vec().unwrap();
            assert!(got.iter().zip(&chain_want).all(|(g, w)| (g - w).abs() <= 1e-12 * w.abs().max(1.0)), "wide/chain16_f64");
            h.case("wide/saxpy_f64", &format!("{n}"), Work::Bytes(24.0 * n as f64), &mut || (&x * 2.5 + &y).eval().unwrap());
            h.case("wide/chain16_f64", &format!("{n}"), Work::Bytes(16.0 * n as f64), &mut || chain16(&x).eval().unwrap());
            let got = x.sum().item().unwrap();
            let want: f64 = xs.iter().sum();
            check_scalar("wide/sum_f64", got, want, 1e-10);
            h.case("wide/sum_f64", &format!("{n}"), Work::Bytes(8.0 * n as f64), &mut || {
                std::hint::black_box(x.sum().item().unwrap());
            });
        } else {
            h.skip("wide/saxpy_f64", "the device has no 64-bit shader floats");
            h.skip("wide/chain16_f64", "the device has no 64-bit shader floats");
            h.skip("wide/sum_f64", "the device has no 64-bit shader floats");
        }

        let dx: Vec<Df32> = xs.iter().map(|&v| Df32::from(v)).collect();
        let dy: Vec<Df32> = ys.iter().map(|&v| Df32::from(v)).collect();
        let x = Array::<Df32>::from_slice_on(&ctx, &dx, n).unwrap();
        let y = Array::<Df32>::from_slice_on(&ctx, &dy, n).unwrap();
        x.eval().unwrap();
        y.eval().unwrap();
        let chain16 = |x: &Array<Df32>| {
            let mut z = x.clone();
            for i in 0..16usize {
                z = match i % 4 {
                    0 => &z * Df32::from(0.999 - 0.0001 * (i % 7) as f64),
                    1 => &z + Df32::from(0.01 * (1 + i % 5) as f64),
                    2 => z.abs() * Df32::from(0.5),
                    _ => &z - Df32::from(0.005 * (1 + i % 3) as f64),
                };
            }
            z
        };
        let close = |name: &str, got: Vec<Df32>, want: &[f64], rel: f64| {
            for (i, (g, w)) in got.iter().zip(want).enumerate() {
                let g = g.to_f64();
                assert!((g - w).abs() <= rel * w.abs().max(1e-3), "{name}: index {i}: got {g}, want {w}");
            }
        };
        close("wide/saxpy_df32", (&x * Df32::from(2.5) + &y).to_vec().unwrap(), &saxpy_want, 1e-12);
        close("wide/chain16_df32", chain16(&x).to_vec().unwrap(), &chain_want, 1e-11);
        h.case("wide/saxpy_df32", &format!("{n}"), Work::Bytes(24.0 * n as f64), &mut || (&x * Df32::from(2.5) + &y).eval().unwrap());
        h.case("wide/chain16_df32", &format!("{n}"), Work::Bytes(16.0 * n as f64), &mut || chain16(&x).eval().unwrap());
        let exp_want: Vec<f64> = xs.iter().map(|&v| v.exp()).collect();
        close("wide/exp_df32", x.exp().to_vec().unwrap(), &exp_want, 1e-11);
        h.case("wide/exp_df32", &format!("{n}"), Work::Bytes(16.0 * n as f64), &mut || x.exp().eval().unwrap());
        let got = x.sum().item().unwrap().to_f64();
        let want: f64 = xs.iter().sum();
        check_scalar("wide/sum_df32", got, want, 1e-10);
        h.case("wide/sum_df32", &format!("{n}"), Work::Bytes(8.0 * n as f64), &mut || {
            std::hint::black_box(x.sum().item().unwrap());
        });
    });
}

fn transpose(h: &mut Harness) {
    h.guarded("transpose", |h| {
        let sizes: Vec<usize> = match (h.ctx.is_cpu(), h.quick) {
            (false, false) => vec![1024, 4096, 8192],
            (false, true) => vec![1024, 2048],
            (true, false) => vec![1024, 2048],
            (true, true) => vec![1024],
        };
        for n in sizes {
            let data = values(n * n, 5, -1.0, 1.0);
            let b_data = values(n * n, 6, -1.0, 1.0);
            let a = upload(h, &data, [n, n]);
            let b = upload(h, &b_data, [n, n]);
            let stride = (n * n / 65536).max(1) | 1;
            let at = |i: usize| f64::from(data[(i % n) * n + i / n]);
            check_sampled("transpose/copy", &host(&a.transpose().copy()), at, stride, 0.0, 0.0);
            check_sampled("transpose/add", &host(&(&a.transpose() + &b)), |i| at(i) + f64::from(b_data[i]), stride, 1e-6, 1e-6);
            let bytes = 8.0 * (n * n) as f64;
            h.case("transpose/copy", &format!("{n}x{n}"), Work::Bytes(bytes), &mut || a.transpose().copy().eval().unwrap());
            h.case("transpose/add", &format!("{n}x{n}"), Work::Bytes(1.5 * bytes), &mut || (&a.transpose() + &b).eval().unwrap());
        }
    });
}

fn reductions(h: &mut Harness) {
    h.guarded("red", |h| {
        let max_n: usize = if h.ctx.is_cpu() { pick(h, 1 << 22, 1 << 24) } else { pick(h, 1 << 24, 1 << 26) };
        let xs = values(max_n, 21, 0.0, 1.0);
        let ys = values(max_n, 22, 0.0, 1.0);
        let sum_to = |n: usize| xs[..n].iter().map(|&v| f64::from(v)).sum::<f64>();
        let sizes: Vec<usize> = [1usize << 10, 1 << 14, 1 << 18, 1 << 22, 1 << 26].into_iter().filter(|&n| n <= max_n).collect();
        for &n in &sizes {
            let x = upload(h, &xs[..n], n);
            check_scalar("red/sum", f64::from(x.sum().item().unwrap()), sum_to(n), 1e-4);
            h.case("red/sum", &format!("{n}"), Work::Bytes(4.0 * n as f64), &mut || {
                std::hint::black_box(x.sum().item().unwrap());
            });
        }
        for &n in sizes.iter().filter(|&&n| n == 1 << 10 || n == 1 << 22 || n == max_n) {
            let (x, y) = (upload(h, &xs[..n], n), upload(h, &ys[..n], n));
            let want: f64 = (0..n).map(|i| f64::from(xs[i]) * f64::from(ys[i])).sum();
            check_scalar("red/dot", f64::from((&x * &y).sum().item().unwrap()), want, 1e-4);
            h.case("red/dot", &format!("{n}"), Work::Bytes(8.0 * n as f64), &mut || {
                std::hint::black_box((&x * &y).sum().item().unwrap());
            });
        }

        let n = max_n.min(1 << 24);
        let x = upload(h, &xs[..n], n);
        let sq: f64 = xs[..n].iter().map(|&v| f64::from(v) * f64::from(v)).sum();
        check_scalar("red/norm", f64::from(x.norm().item().unwrap()), sq.sqrt(), 1e-4);
        h.case("red/norm", &format!("{n}"), Work::Bytes(4.0 * n as f64), &mut || {
            std::hint::black_box(x.norm().item().unwrap());
        });
        let max = xs[..n].iter().fold(0.0f32, |m, &v| m.max(v));
        assert_eq!(x.max().item().unwrap(), max, "red/max");
        h.case("red/max", &format!("{n}"), Work::Bytes(4.0 * n as f64), &mut || {
            std::hint::black_box(x.max().item().unwrap());
        });
        let min = xs[..n].iter().fold(1.0f32, |m, &v| m.min(v));
        let four = x.reduce([Reduction::Sum, Reduction::Min, Reduction::Max, Reduction::SumOfSquares]).to_vec().unwrap();
        check_scalar("red/multi4 sum", f64::from(four[0]), sum_to(n), 1e-4);
        assert_eq!((four[1], four[2]), (min, max), "red/multi4 min max");
        check_scalar("red/multi4 squares", f64::from(four[3]), sq, 1e-4);
        h.case("red/multi4", &format!("{n}"), Work::Bytes(4.0 * n as f64), &mut || {
            std::hint::black_box(x.reduce([Reduction::Sum, Reduction::Min, Reduction::Max, Reduction::SumOfSquares]).to_vec().unwrap());
        });
        let at = xs[..n].iter().position(|&v| v == max).unwrap() as u32;
        assert_eq!(x.argmax().item().unwrap(), at, "red/argmax");
        h.case("red/argmax", &format!("{n}"), Work::Bytes(4.0 * n as f64), &mut || {
            std::hint::black_box(x.argmax().item().unwrap());
        });
        check_scalar("red/sum_compensated", f64::from(x.sum_compensated().item().unwrap()), sum_to(n), 1e-6);
        h.case("red/sum_compensated", &format!("{n}"), Work::Bytes(4.0 * n as f64), &mut || {
            std::hint::black_box(x.sum_compensated().item().unwrap());
        });
        let above = xs[..n].iter().filter(|&&v| v > 0.75).count() as u32;
        assert_eq!(x.gt(0.75).count().item().unwrap(), above, "red/count");
        h.case("red/count_gt", &format!("{n}"), Work::Bytes(4.0 * n as f64), &mut || {
            std::hint::black_box(x.gt(0.75).count().item().unwrap());
        });
        let masked: f64 = xs[..n].iter().filter(|&&v| v > 0.75).map(|&v| f64::from(v)).sum();
        check_scalar("red/sum_where", f64::from(x.sum_where(&x.gt(0.75)).item().unwrap()), masked, 1e-4);
        h.case("red/sum_where", &format!("{n}"), Work::Bytes(4.0 * n as f64), &mut || {
            std::hint::black_box(x.sum_where(&x.gt(0.75)).item().unwrap());
        });
    });
}
