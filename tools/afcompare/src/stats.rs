//! Reduction and statistics cases (group `reduce`, after the cases of `cases::reductions`):
//! products, variance / standard deviation / mean and variance together, weighted means,
//! covariance and correlation, norms, and the counts and truth tests of booleans, by dimension.
//!
//! ArrayFire's `norm`, `corrcoef` (and the `*_all` functions) return the value on the host, so
//! their rows time a result read back to the host on both sides; the others leave a one-element
//! (or one value per line) result on the device.

use arrayfire as af;
use luma::{Array as LA, Bias, Norm};

use crate::afx;
use crate::bench::Bench;
use crate::lumax;
use crate::util::*;

use crate::cases::human;

fn host_var(data: &[f32], ddof: f64) -> f64 {
    let n = data.len() as f64;
    let mean = data.iter().map(|&v| f64::from(v)).sum::<f64>() / n;
    data.iter().map(|&v| (f64::from(v) - mean).powi(2)).sum::<f64>() / (n - ddof)
}

fn bias_af(bias: Bias) -> af::VarianceBias {
    match bias {
        Bias::Population => af::VarianceBias::POPULATION,
        Bias::Sample => af::VarianceBias::SAMPLE,
    }
}

fn name(bias: Bias) -> &'static str {
    match bias {
        Bias::Population => "population",
        Bias::Sample => "sample",
    }
}

pub fn reductions_extra(b: &mut Bench) {
    b.guarded("reduce", |b| {
        whole_array(b);
        by_axis(b);
        booleans(b);
        norms(b);
    });
}

/// Whole-array statistics of 1M and 100M elements.
fn whole_array(b: &mut Bench) {
    let ctx = b.ctx.clone();
    for n in [1_000_000usize, 100_000_000] {
        let size = human(n);
        // Values near 1, so that the product of all of them stays in range.
        let near_one = uniform(n, 31, 0.9999, 1.0001);
        let (xl, xa) = (lumax::arr(&ctx, &near_one, n), afx::arr(&near_one, &[n as u64]));
        let want: f64 = near_one.iter().map(|&v| f64::from(v)).product();
        drop(near_one);
        let (lp, ap) = (lumax::host(&xl.prod()), afx::host(&af::product(&xa, 0)));
        let v = compare(&lp, &ap, 2e-3, 0.0).and(compare_ref("luma", &lp, &[want], 2e-3, 0.0)).and(compare_ref("AF", &ap, &[want], 2e-3, 0.0));
        b.run(
            "3 reduce prod",
            &size,
            v,
            "device-side 1-element result",
            &mut || {
                xl.prod().eval().unwrap();
                ctx.sync().unwrap();
            },
            &mut || {
                af::product(&xa, 0).eval();
                afx::sync();
            },
        );
        drop((xl, xa));

        let data = uniform(n, 32, 0.0, 1.0);
        let other = uniform(n, 33, 0.0, 1.0);
        let weights = uniform(n, 34, 0.5, 1.5);
        let (xl, xa) = (lumax::arr(&ctx, &data, n), afx::arr(&data, &[n as u64]));
        let (yl, ya) = (lumax::arr(&ctx, &other, n), afx::arr(&other, &[n as u64]));
        let (wl, wa) = (lumax::arr(&ctx, &weights, n), afx::arr(&weights, &[n as u64]));
        let mean_ref = data.iter().map(|&v| f64::from(v)).sum::<f64>() / n as f64;

        let (lm, am) = (lumax::host(&xl.mean()), afx::host(&af::mean(&xa, 0)));
        let v = compare(&lm, &am, 1e-4, 0.0).and(compare_ref("luma", &lm, &[mean_ref], 1e-4, 0.0));
        b.run(
            "3 reduce mean",
            &size,
            v,
            "device-side 1-element result",
            &mut || {
                xl.mean().eval().unwrap();
                ctx.sync().unwrap();
            },
            &mut || {
                af::mean(&xa, 0).eval();
                afx::sync();
            },
        );

        let wsum: f64 = weights.iter().map(|&v| f64::from(v)).sum();
        let wmean_ref = data.iter().zip(&weights).map(|(&v, &w)| f64::from(v) * f64::from(w)).sum::<f64>() / wsum;
        let (lm, am) = (lumax::host(&xl.mean_weighted(&wl)), afx::host(&af::mean_weighted(&xa, &wa, 0)));
        let v = compare(&lm, &am, 1e-4, 0.0).and(compare_ref("luma", &lm, &[wmean_ref], 1e-4, 0.0));
        b.run(
            "3 reduce weighted mean",
            &size,
            v,
            "device-side 1-element result",
            &mut || {
                xl.mean_weighted(&wl).eval().unwrap();
                ctx.sync().unwrap();
            },
            &mut || {
                af::mean_weighted(&xa, &wa, 0).eval();
                afx::sync();
            },
        );

        for bias in [Bias::Population, Bias::Sample] {
            let ddof = if bias == Bias::Sample { 1.0 } else { 0.0 };
            let var_ref = host_var(&data, ddof);
            let (lv, av) = (lumax::host(&xl.var(bias)), afx::host(&af::var_v2(&xa, bias_af(bias), 0)));
            let v = compare(&lv, &av, 1e-3, 0.0).and(compare_ref("luma", &lv, &[var_ref], 1e-3, 0.0));
            b.run(
                &format!("3 reduce var ({})", name(bias)),
                &size,
                v,
                "device-side 1-element result",
                &mut || {
                    xl.var(bias).eval().unwrap();
                    ctx.sync().unwrap();
                },
                &mut || {
                    af::var_v2(&xa, bias_af(bias), 0).eval();
                    afx::sync();
                },
            );
            let (ls, asd) = (lumax::host(&xl.std(bias)), afx::host(&af::stdev_v2(&xa, bias_af(bias), 0)));
            let v = compare(&ls, &asd, 1e-3, 0.0).and(compare_ref("luma", &ls, &[var_ref.sqrt()], 1e-3, 0.0));
            b.run(
                &format!("3 reduce stdev ({})", name(bias)),
                &size,
                v,
                "device-side 1-element result",
                &mut || {
                    xl.std(bias).eval().unwrap();
                    ctx.sync().unwrap();
                },
                &mut || {
                    af::stdev_v2(&xa, bias_af(bias), 0).eval();
                    afx::sync();
                },
            );
        }

        let empty = af::Array::<f32>::new_empty(afx::dims(&[0]));
        let (lm, lv) = xl.meanvar(Bias::Population);
        let (am, av) = af::meanvar(&xa, &empty, af::VarianceBias::POPULATION, 0);
        let v = compare(&lumax::host(&lm), &afx::host(&am), 1e-4, 0.0).and(compare(&lumax::host(&lv), &afx::host(&av), 1e-3, 0.0));
        b.run(
            "3 reduce meanvar",
            &size,
            v,
            "population variance; both results evaluated together",
            &mut || {
                let (m, v) = xl.meanvar(Bias::Population);
                luma::eval_all(&[&m, &v]).unwrap();
                ctx.sync().unwrap();
            },
            &mut || {
                let (m, v) = af::meanvar(&xa, &empty, af::VarianceBias::POPULATION, 0);
                af::eval_multiple(vec![&m, &v]);
                afx::sync();
            },
        );

        let (mx, my) = (mean_ref, other.iter().map(|&v| f64::from(v)).sum::<f64>() / n as f64);
        let cov_ref = data.iter().zip(&other).map(|(&a, &b)| (f64::from(a) - mx) * (f64::from(b) - my)).sum::<f64>() / (n - 1) as f64;
        let (lc, ac) = (lumax::host(&xl.cov(&yl, Bias::Sample)), afx::host(&af::cov_v2(&xa, &ya, af::VarianceBias::SAMPLE)));
        let v = compare(&lc, &ac, 2e-2, 1e-6).and(compare_ref("luma", &lc, &[cov_ref], 2e-2, 1e-6));
        b.run(
            "3 reduce cov (sample)",
            &size,
            v,
            "device-side 1-element result; of independent data (value near 0)",
            &mut || {
                xl.cov(&yl, Bias::Sample).eval().unwrap();
                ctx.sync().unwrap();
            },
            &mut || {
                af::cov_v2(&xa, &ya, af::VarianceBias::SAMPLE).eval();
                afx::sync();
            },
        );

        // Correlated data, so that the coefficient is a real number.
        let mixed: Vec<f32> = data.iter().zip(&other).map(|(&a, &e)| 0.8 * a + 0.2 * e).collect();
        let (zl, za) = (lumax::arr(&ctx, &mixed, n), afx::arr(&mixed, &[n as u64]));
        let (lr, ar) = (lumax::host(&xl.corrcoef(&zl)), [af::corrcoef(&xa, &za).0 as f32]);
        let mz = mixed.iter().map(|&v| f64::from(v)).sum::<f64>() / n as f64;
        let cxz = data.iter().zip(&mixed).map(|(&a, &b)| (f64::from(a) - mx) * (f64::from(b) - mz)).sum::<f64>();
        let corr_ref = cxz / (host_var(&data, 0.0) * n as f64 * host_var(&mixed, 0.0) * n as f64).sqrt();
        let v = compare(&lr, &ar, 1e-3, 0.0).and(compare_ref("luma", &lr, &[corr_ref], 1e-3, 0.0));
        b.run(
            "3 reduce corrcoef",
            &size,
            v,
            "result read back to the host on both sides",
            &mut || {
                std::hint::black_box(xl.corrcoef(&zl).item().unwrap());
            },
            &mut || {
                std::hint::black_box(af::corrcoef(&xa, &za));
            },
        );
    }
}

/// Reductions along a dimension of a 4096 x 4096 array, and of the tall table of the describe case.
fn by_axis(b: &mut Bench) {
    let ctx = b.ctx.clone();
    let n = 4096usize;
    let data = uniform(n * n, 41, 0.9, 1.1);
    let weights = uniform(n * n, 42, 0.5, 1.5);
    let dims = [n as u64, n as u64];
    let (xl, xa) = (lumax::arr(&ctx, &data, [n, n]), afx::arr(&data, &dims));
    let (wl, wa) = (lumax::arr(&ctx, &weights, [n, n]), afx::arr(&weights, &dims));
    for axis in 0..2usize {
        let what = if axis == 0 { "axis 0 (contiguous lines)" } else { "axis 1 (strided)" };
        let line = |l: usize| -> Vec<f32> { (0..n).map(|s| if axis == 0 { data[s + n * l] } else { data[l + n * s] }).collect() };
        let weight_line = |l: usize| -> Vec<f32> { (0..n).map(|s| if axis == 0 { weights[s + n * l] } else { weights[l + n * s] }).collect() };
        let probe = [0usize, 17, n - 1];

        let (lp, ap) = (lumax::host(&xl.prod_axis(axis)), afx::host(&af::product(&xa, axis as i32)));
        let want: Vec<f64> = probe.iter().map(|&l| line(l).iter().map(|&v| f64::from(v)).product()).collect();
        let v = compare(&lp, &ap, 2e-3, 0.0).and(compare_ref("luma", &probe.map(|l| lp[l]), &want, 2e-3, 0.0));
        b.run(
            &format!("3 reduce prod along {what}"),
            "4096 x 4096",
            v,
            "",
            &mut || {
                xl.prod_axis(axis).eval().unwrap();
                ctx.sync().unwrap();
            },
            &mut || {
                af::product(&xa, axis as i32).eval();
                afx::sync();
            },
        );

        let (lm, am) = (lumax::host(&xl.mean_axis(axis)), afx::host(&af::mean(&xa, axis as i64)));
        let v = compare(&lm, &am, 1e-4, 0.0);
        b.run(
            &format!("3 reduce mean along {what}"),
            "4096 x 4096",
            v,
            "",
            &mut || {
                xl.mean_axis(axis).eval().unwrap();
                ctx.sync().unwrap();
            },
            &mut || {
                af::mean(&xa, axis as i64).eval();
                afx::sync();
            },
        );

        let (lm, am) = (lumax::host(&xl.mean_axis_weighted(axis, &wl)), afx::host(&af::mean_weighted(&xa, &wa, axis as i64)));
        let wmean: Vec<f64> = probe
            .iter()
            .map(|&l| line(l).iter().zip(weight_line(l)).map(|(&x, w)| f64::from(x) * f64::from(w)).sum::<f64>() / weight_line(l).iter().map(|&w| f64::from(w)).sum::<f64>())
            .collect();
        let v = compare(&lm, &am, 1e-4, 0.0).and(compare_ref("luma", &probe.map(|l| lm[l]), &wmean, 1e-4, 0.0));
        b.run(
            &format!("3 reduce weighted mean along {what}"),
            "4096 x 4096",
            v,
            "",
            &mut || {
                xl.mean_axis_weighted(axis, &wl).eval().unwrap();
                ctx.sync().unwrap();
            },
            &mut || {
                af::mean_weighted(&xa, &wa, axis as i64).eval();
                afx::sync();
            },
        );

        for bias in [Bias::Population, Bias::Sample] {
            let ddof = if bias == Bias::Sample { 1.0 } else { 0.0 };
            let (lv, av) = (lumax::host(&xl.var_axis_bias(axis, bias)), afx::host(&af::var_v2(&xa, bias_af(bias), axis as i64)));
            let want: Vec<f64> = probe.iter().map(|&l| host_var(&line(l), ddof)).collect();
            let v = compare(&lv, &av, 1e-3, 0.0).and(compare_ref("luma", &probe.map(|l| lv[l]), &want, 1e-3, 0.0));
            b.run(
                &format!("3 reduce var ({}) along {what}", name(bias)),
                "4096 x 4096",
                v,
                "",
                &mut || {
                    xl.var_axis_bias(axis, bias).eval().unwrap();
                    ctx.sync().unwrap();
                },
                &mut || {
                    af::var_v2(&xa, bias_af(bias), axis as i64).eval();
                    afx::sync();
                },
            );
            let (ls, asd) = (lumax::host(&xl.std_axis_bias(axis, bias)), afx::host(&af::stdev_v2(&xa, bias_af(bias), axis as i64)));
            let v = compare(&ls, &asd, 1e-3, 0.0);
            b.run(
                &format!("3 reduce stdev ({}) along {what}", name(bias)),
                "4096 x 4096",
                v,
                "",
                &mut || {
                    xl.std_axis_bias(axis, bias).eval().unwrap();
                    ctx.sync().unwrap();
                },
                &mut || {
                    af::stdev_v2(&xa, bias_af(bias), axis as i64).eval();
                    afx::sync();
                },
            );
        }

        let empty = af::Array::<f32>::new_empty(afx::dims(&[0]));
        let (lm, lv) = xl.meanvar_axis(axis, Bias::Population);
        let (am, av) = af::meanvar(&xa, &empty, af::VarianceBias::POPULATION, axis as i64);
        let v = compare(&lumax::host(&lm), &afx::host(&am), 1e-4, 0.0).and(compare(&lumax::host(&lv), &afx::host(&av), 1e-3, 0.0));
        b.run(
            &format!("3 reduce meanvar along {what}"),
            "4096 x 4096",
            v,
            "population variance; both results evaluated together",
            &mut || {
                let (m, v) = xl.meanvar_axis(axis, Bias::Population);
                luma::eval_all(&[&m, &v]).unwrap();
                ctx.sync().unwrap();
            },
            &mut || {
                let (m, v) = af::meanvar(&xa, &empty, af::VarianceBias::POPULATION, axis as i64);
                af::eval_multiple(vec![&m, &v]);
                afx::sync();
            },
        );
    }
}

/// The count and the truth tests, of an evaluated boolean array and of the comparison that makes it.
fn booleans(b: &mut Bench) {
    let ctx = b.ctx.clone();
    let n = 4096usize;
    let data = uniform(n * n, 51, 0.0, 1.0);
    let dims = [n as u64, n as u64];
    let (xl, xa) = (lumax::arr(&ctx, &data, [n, n]), afx::arr(&data, &dims));
    // Mostly true, so that `all` has lines that are not decided by their first element.
    let threshold = 0.0001f32;
    let (bl, ba) = (xl.gt(threshold), af::gt(&xa, &threshold, false));
    bl.eval().unwrap();
    ba.eval();
    ctx.sync().unwrap();
    afx::sync();
    for axis in 0..2usize {
        let what = if axis == 0 { "dim 0 (contiguous lines)" } else { "dim 1 (strided)" };
        let lines: Vec<Vec<bool>> = (0..n).map(|l| (0..n).map(|s| (if axis == 0 { data[s + n * l] } else { data[l + n * s] }) > threshold).collect()).collect();
        let counts: Vec<f32> = lines.iter().map(|l| l.iter().filter(|&&b| b).count() as f32).collect();
        let alls: Vec<f32> = lines.iter().map(|l| f32::from(u8::from(l.iter().all(|&b| b)))).collect();
        let anys: Vec<f32> = lines.iter().map(|l| f32::from(u8::from(l.iter().any(|&b| b)))).collect();
        let as_f32 = |v: Vec<u32>| v.into_iter().map(|x| x as f32).collect::<Vec<f32>>();
        let b2f = |v: Vec<bool>| v.into_iter().map(|x| f32::from(u8::from(x))).collect::<Vec<f32>>();
        let ab2f = |a: &af::Array<bool>| b2f(afx::host(a));
        let ac2f = |a: &af::Array<u32>| as_f32(afx::host(a));

        // Evaluated boolean inputs.
        let v = exact(&as_f32(bl.count_axis(axis).to_vec().unwrap()), &counts).and(exact(&ac2f(&af::count(&ba, axis as i32)), &counts));
        b.run(
            &format!("3 reduce count along {what}"),
            "4096 x 4096 bool",
            v,
            "evaluated bool input; both give u32",
            &mut || {
                bl.count_axis(axis).eval().unwrap();
                ctx.sync().unwrap();
            },
            &mut || {
                af::count(&ba, axis as i32).eval();
                afx::sync();
            },
        );
        let v = exact(&b2f(bl.all_axis(axis).to_vec().unwrap()), &alls).and(exact(&ab2f(&af::all_true(&ba, axis as i32)), &alls));
        b.run(
            &format!("3 reduce allTrue along {what}"),
            "4096 x 4096 bool",
            v,
            "evaluated bool input",
            &mut || {
                bl.all_axis(axis).eval().unwrap();
                ctx.sync().unwrap();
            },
            &mut || {
                af::all_true(&ba, axis as i32).eval();
                afx::sync();
            },
        );
        let v = exact(&b2f(bl.any_axis(axis).to_vec().unwrap()), &anys).and(exact(&ab2f(&af::any_true(&ba, axis as i32)), &anys));
        b.run(
            &format!("3 reduce anyTrue along {what}"),
            "4096 x 4096 bool",
            v,
            "evaluated bool input",
            &mut || {
                bl.any_axis(axis).eval().unwrap();
                ctx.sync().unwrap();
            },
            &mut || {
                af::any_true(&ba, axis as i32).eval();
                afx::sync();
            },
        );

        // The comparison that makes the booleans, inside the reduction (luma fuses it; ArrayFire evaluates it first).
        let v = exact(&as_f32(xl.gt(threshold).count_axis(axis).to_vec().unwrap()), &counts).and(exact(&ac2f(&af::count(&af::gt(&xa, &threshold, false), axis as i32)), &counts));
        b.run(
            &format!("3 reduce count of (x > c) along {what}"),
            "4096 x 4096 f32",
            v,
            "the comparison is fused into luma's reduction; ArrayFire evaluates it first",
            &mut || {
                xl.gt(threshold).count_axis(axis).eval().unwrap();
                ctx.sync().unwrap();
            },
            &mut || {
                af::count(&af::gt(&xa, &threshold, false), axis as i32).eval();
                afx::sync();
            },
        );
    }
    // The whole array as one line (what `count_all` etc. do, kept on the device here).
    let total = n * n;
    let flat_b: LA<bool> = bl.reshape(total);
    let flat_a = af::flat(&ba);
    let want = data.iter().filter(|&&v| v > threshold).count() as f32;
    let v = exact(&[flat_b.count().item().unwrap() as f32], &[want]).and(exact(&[afx::host(&af::count(&flat_a, 0))[0] as f32], &[want]));
    b.run(
        "3 reduce count (whole array)",
        "16M bool",
        v,
        "evaluated bool input",
        &mut || {
            flat_b.count().eval().unwrap();
            ctx.sync().unwrap();
        },
        &mut || {
            af::count(&flat_a, 0).eval();
            afx::sync();
        },
    );
    let v = Verdict::yes(format!("luma {} / AF {}", flat_b.all().item().unwrap(), afx::host(&af::all_true(&flat_a, 0))[0]));
    b.run(
        "3 reduce allTrue (whole array)",
        "16M bool",
        v,
        "evaluated bool input",
        &mut || {
            flat_b.all().eval().unwrap();
            ctx.sync().unwrap();
        },
        &mut || {
            af::all_true(&flat_a, 0).eval();
            afx::sync();
        },
    );
    let v = Verdict::yes(format!("luma {} / AF {}", flat_b.any().item().unwrap(), afx::host(&af::any_true(&flat_a, 0))[0]));
    b.run(
        "3 reduce anyTrue (whole array)",
        "16M bool",
        v,
        "evaluated bool input",
        &mut || {
            flat_b.any().eval().unwrap();
            ctx.sync().unwrap();
        },
        &mut || {
            af::any_true(&flat_a, 0).eval();
            afx::sync();
        },
    );
}

/// Norms of a vector of 100M elements and of a 4096 x 4096 matrix; the result is read back on both sides.
fn norms(b: &mut Bench) {
    let ctx = b.ctx.clone();
    let n = 100_000_000usize;
    let data = uniform(n, 61, -1.0, 1.0);
    let (xl, xa) = (lumax::arr(&ctx, &data, n), afx::arr(&data, &[n as u64]));
    let abs_sum: f64 = data.iter().map(|&v| f64::from(v).abs()).sum();
    let sq_sum: f64 = data.iter().map(|&v| f64::from(v) * f64::from(v)).sum();
    let max_abs = data.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
    let cube_sum: f64 = data.iter().map(|&v| f64::from(v).abs().powi(3)).sum();
    let vector: [(&str, Norm, af::NormType, f64, f64); 4] = [
        ("L1", Norm::One, af::NormType::VECTOR_1, 1.0, abs_sum),
        ("L2", Norm::Two, af::NormType::VECTOR_2, 2.0, sq_sum.sqrt()),
        ("Linf", Norm::Inf, af::NormType::VECTOR_INF, 1.0, f64::from(max_abs)),
        ("Lp (p = 3)", Norm::P(3.0), af::NormType::VECTOR_P, 3.0, cube_sum.cbrt()),
    ];
    for (label, ln, an, p, want) in vector {
        let (lv, av) = (xl.norm_p(ln).item().unwrap(), af::norm(&xa, an, p, 1.0));
        let v = compare(&[lv], &[av as f32], 1e-3, 0.0).and(compare_ref("luma", &[lv], &[want], 1e-3, 0.0));
        b.run(
            &format!("3 reduce norm {label}"),
            "100M",
            v,
            "result read back to the host on both sides",
            &mut || {
                std::hint::black_box(xl.norm_p(ln).item().unwrap());
            },
            &mut || {
                std::hint::black_box(af::norm(&xa, an, p, 1.0));
            },
        );
    }
    drop((xl, xa));

    let n = 4096usize;
    let data = uniform(n * n, 62, -1.0, 1.0);
    let (xl, xa) = (lumax::arr(&ctx, &data, [n, n]), afx::arr(&data, &[n as u64, n as u64]));
    let at = |r: usize, c: usize| f64::from(data[r + n * c]).abs();
    let one = (0..n).map(|c| (0..n).map(|r| at(r, c)).sum::<f64>()).fold(0.0, f64::max);
    let inf = (0..n).map(|r| (0..n).map(|c| at(r, c)).sum::<f64>()).fold(0.0, f64::max);
    let fro = data.iter().map(|&v| f64::from(v) * f64::from(v)).sum::<f64>().sqrt();
    // ArrayFire's Frobenius norm is the L(p, q) matrix norm with p = q = 2.
    let matrix: [(&str, Norm, af::NormType, f64); 3] =
        [("matrix 1-norm", Norm::MatrixOne, af::NormType::MATRIX_1, one), ("matrix inf-norm", Norm::MatrixInf, af::NormType::MATRIX_INF, inf), ("Frobenius", Norm::Frobenius, af::NormType::MATRIX_L_PQ, fro)];
    for (label, ln, an, want) in matrix {
        let pq = if an == af::NormType::MATRIX_L_PQ { 2.0 } else { 1.0 };
        let (lv, av) = (xl.norm_p(ln).item().unwrap(), af::norm(&xa, an, pq, pq));
        let v = compare(&[lv], &[av as f32], 1e-3, 0.0).and(compare_ref("luma", &[lv], &[want], 1e-3, 0.0));
        b.run(
            &format!("3 reduce norm {label}"),
            "4096 x 4096",
            v,
            "result read back to the host on both sides",
            &mut || {
                std::hint::black_box(xl.norm_p(ln).item().unwrap());
            },
            &mut || {
                std::hint::black_box(af::norm(&xa, an, pq, pq));
            },
        );
    }
}
