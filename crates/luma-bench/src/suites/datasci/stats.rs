//! `reduce/` cases of statistics: products, variances, weighted means, covariance, norms and the
//! counts and truth tests of booleans along an axis, each verified against an `f64` host
//! computation. The `afcompare` tool times the same operations against ArrayFire.

use crate::harness::*;
use luma::*;

use super::{big, case_trim, up};

fn host_var(x: &[f32], ddof: f64) -> f64 {
    let n = x.len() as f64;
    let mean = x.iter().map(|&v| f64::from(v)).sum::<f64>() / n;
    x.iter().map(|&v| (f64::from(v) - mean).powi(2)).sum::<f64>() / (n - ddof)
}

pub(super) fn run(h: &mut Harness) {
    whole_cases(h);
    axis_cases(h);
    bool_cases(h);
}

fn whole_cases(h: &mut Harness) {
    let n = big(h);
    let size = n.to_string();
    let bytes = Work::Bytes(4.0 * n as f64);
    if h.want("reduce/prod_f32") {
        // Ones with a factor near 1 in every 5000th place (a tree of products of values very close to 1 is biased by f32 rounding).
        let mut hx = vec![1.0f32; n];
        for (slot, v) in hx.iter_mut().step_by(5000).zip(values(n / 5000 + 1, 201, 0.9, 1.1)) {
            *slot = v;
        }
        let x = up(h, &hx);
        let want: f64 = hx.iter().map(|&v| f64::from(v)).product();
        check_scalar("reduce/prod_f32", f64::from(x.prod().item().unwrap()), want, 1e-3);
        case_trim(h, "reduce/prod_f32", &size, bytes, &mut || x.prod().eval().unwrap());
    }
    if h.want("reduce/prod_u32") {
        let hx = ints(n, 202, 1 << 20);
        let x = up(h, &hx);
        let want = hx.iter().fold(1u32, |acc, &v| acc.wrapping_mul(v));
        assert_eq!(x.prod().item().unwrap(), want, "reduce/prod_u32");
        case_trim(h, "reduce/prod_u32", &size, bytes, &mut || x.prod().eval().unwrap());
    }
    if h.want("reduce/mean_f32") || h.want("reduce/var_") || h.want("reduce/std_") || h.want("reduce/meanvar_") {
        let hx = values(n, 203, 0.0, 1.0);
        let x = up(h, &hx);
        let mean = hx.iter().map(|&v| f64::from(v)).sum::<f64>() / n as f64;
        if h.want("reduce/mean_f32") {
            check_scalar("reduce/mean_f32", f64::from(x.mean().item().unwrap()), mean, 1e-4);
            case_trim(h, "reduce/mean_f32", &size, bytes, &mut || x.mean().eval().unwrap());
        }
        for (name, bias, ddof) in [("reduce/var_population_f32", Bias::Population, 0.0), ("reduce/var_sample_f32", Bias::Sample, 1.0)] {
            if h.want(name) {
                check_scalar(name, f64::from(x.var(bias).item().unwrap()), host_var(&hx, ddof), 1e-3);
                case_trim(h, name, &size, bytes, &mut || x.var(bias).eval().unwrap());
            }
        }
        if h.want("reduce/std_population_f32") {
            check_scalar("reduce/std_population_f32", f64::from(x.std(Bias::Population).item().unwrap()), host_var(&hx, 0.0).sqrt(), 1e-3);
            case_trim(h, "reduce/std_population_f32", &size, bytes, &mut || x.std(Bias::Population).eval().unwrap());
        }
        if h.want("reduce/meanvar_f32") {
            let (m, v) = x.meanvar(Bias::Population);
            check_scalar("reduce/meanvar_f32 mean", f64::from(m.item().unwrap()), mean, 1e-4);
            check_scalar("reduce/meanvar_f32 var", f64::from(v.item().unwrap()), host_var(&hx, 0.0), 1e-3);
            case_trim(h, "reduce/meanvar_f32", &size, bytes, &mut || {
                let (m, v) = x.meanvar(Bias::Population);
                eval_all(&[&m, &v]).unwrap();
            });
        }
    }
    if h.want("reduce/mean_weighted_f32") {
        let (hx, hw) = (values(n, 204, 0.0, 1.0), values(n, 205, 0.5, 1.5));
        let (x, w) = (up(h, &hx), up(h, &hw));
        let want = hx.iter().zip(&hw).map(|(&a, &b)| f64::from(a) * f64::from(b)).sum::<f64>() / hw.iter().map(|&b| f64::from(b)).sum::<f64>();
        check_scalar("reduce/mean_weighted_f32", f64::from(x.mean_weighted(&w).item().unwrap()), want, 1e-4);
        case_trim(h, "reduce/mean_weighted_f32", &size, Work::Bytes(8.0 * n as f64), &mut || x.mean_weighted(&w).eval().unwrap());
    }
    if h.want("reduce/cov_") || h.want("reduce/corrcoef_") {
        let hx = values(n, 206, 0.0, 1.0);
        let hz: Vec<f32> = hx.iter().zip(values(n, 207, 0.0, 1.0)).map(|(&a, e)| 0.8 * a + 0.2 * e).collect();
        let (x, z) = (up(h, &hx), up(h, &hz));
        let (mx, mz) = (hx.iter().map(|&v| f64::from(v)).sum::<f64>() / n as f64, hz.iter().map(|&v| f64::from(v)).sum::<f64>() / n as f64);
        let cxz = hx.iter().zip(&hz).map(|(&a, &b)| (f64::from(a) - mx) * (f64::from(b) - mz)).sum::<f64>();
        let bytes = Work::Bytes(8.0 * n as f64);
        if h.want("reduce/cov_sample_f32") {
            check_scalar("reduce/cov_sample_f32", f64::from(x.cov(&z, Bias::Sample).item().unwrap()), cxz / (n - 1) as f64, 1e-3);
            case_trim(h, "reduce/cov_sample_f32", &size, bytes, &mut || x.cov(&z, Bias::Sample).eval().unwrap());
        }
        if h.want("reduce/corrcoef_f32") {
            let want = cxz / (host_var(&hx, 0.0) * n as f64 * host_var(&hz, 0.0) * n as f64).sqrt();
            check_scalar("reduce/corrcoef_f32", f64::from(x.corrcoef(&z).item().unwrap()), want, 1e-3);
            case_trim(h, "reduce/corrcoef_f32", &size, bytes, &mut || x.corrcoef(&z).eval().unwrap());
        }
    }
    if h.want("reduce/norm_") {
        let hx = values(n, 208, -1.0, 1.0);
        let x = up(h, &hx);
        let abs_sum: f64 = hx.iter().map(|&v| f64::from(v).abs()).sum();
        let sq_sum: f64 = hx.iter().map(|&v| f64::from(v) * f64::from(v)).sum();
        let max_abs = hx.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        let cubes: f64 = hx.iter().map(|&v| f64::from(v).abs().powi(3)).sum();
        for (name, norm, want) in [
            ("reduce/norm_l1_f32", Norm::One, abs_sum),
            ("reduce/norm_l2_f32", Norm::Two, sq_sum.sqrt()),
            ("reduce/norm_inf_f32", Norm::Inf, f64::from(max_abs)),
            ("reduce/norm_p3_f32", Norm::P(3.0), cubes.cbrt()),
        ] {
            if h.want(name) {
                check_scalar(name, f64::from(x.norm_p(norm).item().unwrap()), want, 1e-3);
                case_trim(h, name, &size, bytes, &mut || x.norm_p(norm).eval().unwrap());
            }
        }
    }
}

fn axis_cases(h: &mut Harness) {
    // A square table: every line is `side` long.
    let side = if h.quick || h.ctx.is_cpu() { 1024 } else { 4096 };
    let n = side * side;
    let size = format!("{side}x{side}");
    let bytes = Work::Bytes(4.0 * n as f64);
    let hx = values(n, 211, 0.9, 1.1);
    let hw = values(n, 212, 0.5, 1.5);
    let x = Array::from_slice_on(&h.ctx, &hx, [side, side]).unwrap();
    let w = Array::from_slice_on(&h.ctx, &hw, [side, side]).unwrap();
    let line = |data: &[f32], axis: usize, l: usize| -> Vec<f32> { (0..side).map(|s| if axis == 0 { data[s + side * l] } else { data[l + side * s] }).collect() };
    for axis in 0..2usize {
        let probe = [0usize, 17, side - 1];
        let at = |name: &str| format!("reduce/{name}_axis{axis}_f32");
        if h.want(&at("prod")) {
            let got = x.prod_axis(axis).to_vec().unwrap();
            for l in probe {
                let want: f64 = line(&hx, axis, l).iter().map(|&v| f64::from(v)).product();
                check_scalar(&at("prod"), f64::from(got[l]), want, 2e-3);
            }
            case_trim(h, &at("prod"), &size, bytes, &mut || x.prod_axis(axis).eval().unwrap());
        }
        for (name, bias, ddof) in [("var_population", Bias::Population, 0.0), ("var_sample", Bias::Sample, 1.0)] {
            if h.want(&at(name)) {
                let got = x.var_axis_bias(axis, bias).to_vec().unwrap();
                for l in probe {
                    check_scalar(&at(name), f64::from(got[l]), host_var(&line(&hx, axis, l), ddof), 1e-3);
                }
                case_trim(h, &at(name), &size, bytes, &mut || x.var_axis_bias(axis, bias).eval().unwrap());
            }
        }
        if h.want(&at("meanvar")) {
            let (m, v) = x.meanvar_axis(axis, Bias::Population);
            let (gm, gv) = (m.to_vec().unwrap(), v.to_vec().unwrap());
            for l in probe {
                let data = line(&hx, axis, l);
                check_scalar(&at("meanvar"), f64::from(gm[l]), data.iter().map(|&v| f64::from(v)).sum::<f64>() / side as f64, 1e-4);
                check_scalar(&at("meanvar"), f64::from(gv[l]), host_var(&data, 0.0), 1e-3);
            }
            case_trim(h, &at("meanvar"), &size, bytes, &mut || {
                let (m, v) = x.meanvar_axis(axis, Bias::Population);
                eval_all(&[&m, &v]).unwrap();
            });
        }
        if h.want(&at("mean_weighted")) {
            let got = x.mean_axis_weighted(axis, &w).to_vec().unwrap();
            for l in probe {
                let (d, wl) = (line(&hx, axis, l), line(&hw, axis, l));
                let want = d.iter().zip(&wl).map(|(&a, &b)| f64::from(a) * f64::from(b)).sum::<f64>() / wl.iter().map(|&b| f64::from(b)).sum::<f64>();
                check_scalar(&at("mean_weighted"), f64::from(got[l]), want, 1e-4);
            }
            case_trim(h, &at("mean_weighted"), &size, Work::Bytes(8.0 * n as f64), &mut || x.mean_axis_weighted(axis, &w).eval().unwrap());
        }
    }
}

fn bool_cases(h: &mut Harness) {
    let side = if h.quick || h.ctx.is_cpu() { 1024 } else { 4096 };
    let n = side * side;
    let size = format!("{side}x{side}");
    let hx = values(n, 221, 0.0, 1.0);
    let threshold = 0.0001f32;
    let x = Array::from_slice_on(&h.ctx, &hx, [side, side]).unwrap();
    let flags = x.gt(threshold);
    flags.eval().unwrap();
    h.ctx.sync().unwrap();
    for axis in 0..2usize {
        let lines: Vec<Vec<bool>> = (0..side).map(|l| (0..side).map(|s| (if axis == 0 { hx[s + side * l] } else { hx[l + side * s] }) > threshold).collect()).collect();
        let at = |name: &str| format!("reduce/{name}_axis{axis}_bool");
        if h.want(&at("count")) {
            let want: Vec<u32> = lines.iter().map(|l| l.iter().filter(|&&b| b).count() as u32).collect();
            assert_eq!(flags.count_axis(axis).to_vec().unwrap(), want, "{}", at("count"));
            case_trim(h, &at("count"), &size, Work::Bytes(n as f64 / 8.0), &mut || flags.count_axis(axis).eval().unwrap());
        }
        if h.want(&at("all")) {
            let want: Vec<bool> = lines.iter().map(|l| l.iter().all(|&b| b)).collect();
            assert_eq!(flags.all_axis(axis).to_vec().unwrap(), want, "{}", at("all"));
            case_trim(h, &at("all"), &size, Work::Bytes(n as f64 / 8.0), &mut || flags.all_axis(axis).eval().unwrap());
        }
        if h.want(&at("any")) {
            let want: Vec<bool> = lines.iter().map(|l| l.iter().any(|&b| b)).collect();
            assert_eq!(flags.any_axis(axis).to_vec().unwrap(), want, "{}", at("any"));
            case_trim(h, &at("any"), &size, Work::Bytes(n as f64 / 8.0), &mut || flags.any_axis(axis).eval().unwrap());
        }
        // The comparison fused into the reduction.
        if h.want(&at("count_gt")) {
            let want: Vec<u32> = lines.iter().map(|l| l.iter().filter(|&&b| b).count() as u32).collect();
            assert_eq!(x.gt(threshold).count_axis(axis).to_vec().unwrap(), want, "{}", at("count_gt"));
            case_trim(h, &at("count_gt"), &size, Work::Bytes(4.0 * n as f64), &mut || x.gt(threshold).count_axis(axis).eval().unwrap());
        }
    }
}
