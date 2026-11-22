//! The elementwise functions of ArrayFire's `arith.h` that luma added late: inverse trigonometric
//! and hyperbolic functions, `erf`/`erfc`, `lgamma`/`tgamma`, `cbrt`, `log2`/`log10`, `rsqrt`,
//! `pow2`, `sigmoid`, `pow` with array operands, `modulo`, `clamp` with array bounds, `is_zero`,
//! float `%`, and the `f64` forms (CUDA and CPU backends), against `f64` host references.

use luma::{Array, BackendKind, Context, ContextOptions};

fn context() -> Context {
    Context::with_options(ContextOptions::from_env()).unwrap()
}

fn has_wide_math(ctx: &Context) -> bool {
    ctx.features().contains(luma::Features::SHADER_F64) && matches!(ctx.backend(), BackendKind::Cuda | BackendKind::Cpu)
}

/// Deterministic values in `lo..hi`.
fn sweep(n: usize, seed: u64, lo: f64, hi: f64) -> Vec<f64> {
    let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            lo + ((state >> 11) as f64 / (1u64 << 53) as f64) * (hi - lo)
        })
        .collect()
}

/// Whether `got` matches `want` within `rel` (relative to `want`) or `abs`; NaN matches NaN and an
/// infinity matches the same infinity.
fn matches(got: f64, want: f64, rel: f64, abs: f64) -> bool {
    if want.is_nan() {
        return got.is_nan();
    }
    if want.is_infinite() {
        return got == want;
    }
    (got - want).abs() <= abs + rel * want.abs()
}

struct Case {
    name: &'static str,
    inputs: Vec<f64>,
    f32_op: fn(&Array<f32>) -> Array<f32>,
    f64_op: fn(&Array<f64>) -> Array<f64>,
    host: fn(f64) -> f64,
    rel: f64,
    abs: f64,
}

fn cases() -> Vec<Case> {
    let finite = |lo, hi, seed| sweep(997, seed, lo, hi);
    let mut v = Vec::new();
    let mut add = |name, inputs: Vec<f64>, f32_op, f64_op, host, rel, abs| v.push(Case { name, inputs, f32_op, f64_op, host, rel, abs });
    let unit = || {
        let mut x = finite(-1.0, 1.0, 1);
        x.extend([-1.0, 1.0, 0.0, -0.0, 1.5, -2.0, f64::NAN]);
        x
    };
    add("asin", unit(), |a| a.asin(), |a| a.asin(), f64::asin, 2e-6, 1e-7);
    add("acos", unit(), |a| a.acos(), |a| a.acos(), f64::acos, 2e-6, 1e-6);
    let mut wide = finite(-50.0, 50.0, 2);
    wide.extend([0.0, -0.0, 1e-8, f64::INFINITY, f64::NEG_INFINITY, f64::NAN]);
    add("atan", wide.clone(), |a| a.atan(), |a| a.atan(), f64::atan, 2e-6, 1e-7);
    let mut hyper = finite(-20.0, 20.0, 3);
    hyper.extend([0.0, 1e-4, -1e-4, 88.0, -88.0, 100.0, f64::INFINITY, f64::NEG_INFINITY, f64::NAN]);
    add("sinh", hyper.clone(), |a| a.sinh(), |a| a.sinh(), f64::sinh, 5e-6, 1e-6);
    add("cosh", hyper.clone(), |a| a.cosh(), |a| a.cosh(), f64::cosh, 5e-6, 0.0);
    add("asinh", wide.clone(), |a| a.asinh(), |a| a.asinh(), f64::asinh, 5e-6, 1e-6);
    let mut above = finite(1.0, 100.0, 4);
    above.extend([1.0, 0.5, 0.0, -3.0, 1e6, f64::INFINITY, f64::NAN]);
    add("acosh", above, |a| a.acosh(), |a| a.acosh(), f64::acosh, 5e-6, 1e-6);
    let mut inside = finite(-0.999, 0.999, 5);
    inside.extend([0.0, -0.0, 1.0, -1.0, 2.0, f64::NAN]);
    add("atanh", inside, |a| a.atanh(), |a| a.atanh(), f64::atanh, 5e-6, 1e-6);
    let mut pos = finite(0.001, 1000.0, 6);
    pos.extend([1.0, 10.0, 100.0, 1000.0, 0.0, -1.0, f64::INFINITY, f64::NAN, 1e-40]);
    add("log2", pos.clone(), |a| a.log2(), |a| a.log2(), f64::log2, 2e-6, 2e-6);
    add("log10", pos.clone(), |a| a.log10(), |a| a.log10(), f64::log10, 2e-6, 2e-6);
    add("rsqrt", pos.clone(), |a| a.rsqrt(), |a| a.rsqrt(), |x| 1.0 / x.sqrt(), 4e-6, 0.0);
    let mut cube = finite(-1000.0, 1000.0, 7);
    cube.extend([0.0, -0.0, 8.0, -8.0, 27.0, 1e-30, -1e30, 3e38, f64::INFINITY, f64::NEG_INFINITY, f64::NAN]);
    add("cbrt", cube, |a| a.cbrt(), |a| a.cbrt(), f64::cbrt, 2e-6, 0.0);
    let mut errf = finite(-6.0, 6.0, 8);
    errf.extend([0.0, -0.0, 0.25, 0.5, 0.75, 1.0, -1.0, 1.5, 3.0, 5.0, 9.0, -9.0, 12.0, -12.0, f64::INFINITY, f64::NEG_INFINITY, f64::NAN]);
    add("erf", errf.clone(), |a| a.erf(), |a| a.erf(), libm::erf, 3e-6, 1e-7);
    add("erfc", errf.clone(), |a| a.erfc(), |a| a.erfc(), libm::erfc, 1e-5, 0.0);
    let mut gam = finite(0.01, 30.0, 9);
    gam.extend(finite(-10.0, 0.0, 10));
    gam.extend([1.0, 2.0, 3.0, 0.5, 5.5, 10.0, 34.0, 100.0, 1e6, 0.0, -0.0, -1.0, -2.0, -7.0, -0.5, -2.5, f64::INFINITY, f64::NEG_INFINITY, f64::NAN, 1e-20]);
    add("lgamma", gam.clone(), |a| a.lgamma(), |a| a.lgamma(), libm::lgamma, 4e-6, 4e-6);
    let mut gam2 = finite(0.01, 33.0, 11);
    gam2.extend(finite(-10.0, 0.0, 12));
    gam2.extend([1.0, 2.0, 3.0, 4.0, 5.0, 0.5, 5.5, 10.0, 20.0, 34.0, 35.0, 36.0, 171.0, 1e6, 0.0, -0.0, -1.0, -2.0, -7.0, -0.5, -2.5, -30.5, -40.5, f64::INFINITY, f64::NEG_INFINITY, f64::NAN, 1e-20]);
    add("tgamma", gam2, |a| a.tgamma(), |a| a.tgamma(), libm::tgamma, 2e-5, 0.0);
    let mut sig = finite(-30.0, 30.0, 13);
    sig.extend([0.0, 100.0, -100.0, f64::INFINITY, f64::NEG_INFINITY, f64::NAN]);
    add("sigmoid", sig, |a| a.sigmoid(), |a| a.sigmoid(), |x| 1.0 / (1.0 + (-x).exp()), 2e-6, 0.0);
    let mut pow2 = finite(-20.0, 20.0, 14);
    pow2.extend([0.0, 1.0, -1.0, 10.0, 127.0, 128.0, -126.0, -150.0, f64::INFINITY, f64::NEG_INFINITY, f64::NAN]);
    add("pow2", pow2, |a| a.pow2(), |a| a.pow2(), f64::exp2, 4e-6, 0.0);
    v
}

/// `got` of `f32` results against the `f64` reference of the same (f32-rounded) input.
fn check_f32(case: &Case, ctx: &Context) {
    let xs: Vec<f32> = case.inputs.iter().map(|&v| v as f32).collect();
    let x = Array::from_slice_on(ctx, &xs, xs.len()).unwrap();
    let got = (case.f32_op)(&x).to_vec().unwrap();
    for (i, &v) in xs.iter().enumerate() {
        // The GPUs flush denormal operands.
        if v != 0.0 && v.abs() < 1.2e-38 {
            continue;
        }
        let want = (case.host)(f64::from(v));
        let want32 = want as f32;
        let g = f64::from(got[i]);
        // Results that overflow or underflow `f32` must do so; denormal results may flush.
        if want32.is_infinite() && !want.is_infinite() {
            assert!(g.is_infinite() && g.signum() == want.signum(), "{}({v}): got {g}, want overflow {want}", case.name);
        } else if want32 != 0.0 && f64::from(want32).abs() < 1.2e-38 {
            assert!(g.abs() < 2.4e-38, "{}({v}): got {g}, want ~{want}", case.name);
        } else {
            assert!(matches(g, f64::from(want32), case.rel, case.abs), "{}({v}): got {g}, want {want}", case.name);
        }
    }
}

#[test]
fn f32_functions_match_f64_references() {
    let ctx = context();
    for case in cases() {
        check_f32(&case, &ctx);
    }
}

#[test]
fn f64_functions_match_references() {
    let ctx = context();
    if !has_wide_math(&ctx) {
        return;
    }
    for case in cases() {
        let x = Array::from_slice_on(&ctx, &case.inputs, case.inputs.len()).unwrap();
        let got = (case.f64_op)(&x).to_vec().unwrap();
        for (i, &v) in case.inputs.iter().enumerate() {
            let want = (case.host)(v);
            // log10 of an exact power of ten is exact in libm only.
            assert!(matches(got[i], want, 1e-13, 1e-14), "f64 {}({v}): got {}, want {want}", case.name, got[i]);
        }
    }
}

#[test]
fn f64_math_is_unsupported_on_wgpu() {
    let ctx = context();
    if ctx.backend() != BackendKind::Wgpu || !ctx.features().contains(luma::Features::SHADER_F64) {
        return;
    }
    let x = Array::from_slice_on(&ctx, &[0.5f64, 1.0], 2).unwrap();
    assert!(x.erf().to_vec().is_err());
    assert!(x.exp().to_vec().is_err());
    assert_eq!(x.sqrt().to_vec().unwrap().len(), 2);
}

#[test]
fn new_functions_fuse_with_their_neighbours() {
    let ctx = context();
    let xs: Vec<f32> = sweep(5000, 21, -3.0, 3.0).iter().map(|&v| v as f32).collect();
    let x = Array::from_slice_on(&ctx, &xs, xs.len()).unwrap();
    // exp -> erf -> asinh -> times 2 -> sum: one expression feeding one reduction.
    let ((), profile) = ctx
        .profile(|| {
            let total = (x.exp().erf().asinh() * 2.0).sum().item().unwrap();
            let want: f64 = xs.iter().map(|&v| 2.0 * libm::erf(f64::from(v).exp()).asinh()).sum();
            assert!((f64::from(total) - want).abs() <= 1e-4 * want.abs(), "{total} vs {want}");
        })
        .unwrap();
    if ctx.profiling_supported() {
        assert!(profile.launches() <= 2, "{profile:?}");
    }
    // The same chain, elementwise, is a single kernel.
    let ((), profile) = ctx
        .profile(|| {
            let y = (x.sigmoid().cbrt() + x.lgamma().abs().log10().atan()).to_vec().unwrap();
            assert_eq!(y.len(), xs.len());
        })
        .unwrap();
    if ctx.profiling_supported() {
        assert_eq!(profile.launches(), 1, "{profile:?}");
    }
}

#[test]
fn pow_takes_arrays_and_scalars() {
    let ctx = context();
    let base = [0.0f32, 1.0, 2.0, 2.0, 4.0, -2.0, -2.0, 0.5, 9.0, f32::NAN, 3.0, 0.0, 1e10, 1.5, -8.0, f32::NAN, 1.0, -3.0, 0.0, f32::INFINITY];
    let exponent = [0.0f32, 5.0, 10.0, -1.0, 0.5, 3.0, 2.0, 3.0, 0.5, 2.0, f32::NAN, 3.0, 4.0, -0.5, 0.33333334, 0.0, f32::NAN, -3.0, -1.0, 2.0];
    let b = Array::from_slice_on(&ctx, &base, base.len()).unwrap();
    let e = Array::from_slice_on(&ctx, &exponent, exponent.len()).unwrap();
    let got = b.pow(&e).to_vec().unwrap();
    for i in 0..base.len() {
        let want = f64::from(base[i]).powf(f64::from(exponent[i]));
        let want32 = want as f32;
        assert!(
            matches(f64::from(got[i]), f64::from(want32), 1e-5, 0.0) || (want32.is_infinite() && got[i].is_infinite()),
            "{}^{} = {} (want {want})",
            base[i],
            exponent[i],
            got[i]
        );
    }
    // Scalar exponent and scalar base.
    let squares = b.pow(2.0).to_vec().unwrap();
    for i in 0..base.len() {
        assert!(matches(f64::from(squares[i]), f64::from(base[i]) * f64::from(base[i]), 1e-6, 0.0) || squares[i].is_infinite());
    }
    let pow2 = e.rpow(2.0).to_vec().unwrap();
    for i in 0..base.len() {
        if exponent[i].is_nan() {
            assert!(pow2[i].is_nan());
        } else {
            assert!(matches(f64::from(pow2[i]), 2f64.powf(f64::from(exponent[i])), 1e-5, 0.0), "2^{} = {}", exponent[i], pow2[i]);
        }
    }
    // Broadcasting: a column of bases and a row of exponents.
    let col = Array::from_slice_on(&ctx, &[1.0f32, 2.0, 3.0], [3, 1]).unwrap();
    let row = Array::from_slice_on(&ctx, &[0.0f32, 1.0, 2.0, 3.0], [1, 4]).unwrap();
    let table = col.pow(&row).to_vec().unwrap();
    assert_eq!(table.len(), 12);
    for j in 0..4usize {
        for i in 0..3usize {
            let want = f64::from(i as u32 + 1).powi(j as i32);
            assert!(matches(f64::from(table[i + 3 * j]), want, 1e-5, 0.0), "{i} {j}");
        }
    }
    // root
    let x = Array::from_slice_on(&ctx, &[8.0f32, 27.0, 16.0, 2.0, 0.0], 5).unwrap();
    let cube = x.root(3.0).to_vec().unwrap();
    for (got, want) in cube.iter().zip([2.0, 3.0, 2.519842, 1.259921, 0.0]) {
        assert!((got - want).abs() < 1e-5, "{got} vs {want}");
    }
    let n = Array::from_slice_on(&ctx, &[3.0f32, 3.0, 4.0, 2.0, 5.0], 5).unwrap();
    let by_array = x.root(&n).to_vec().unwrap();
    for (got, want) in by_array.iter().zip([2.0, 3.0, 2.0, std::f32::consts::SQRT_2, 0.0]) {
        assert!((got - want).abs() < 1e-5, "{got} vs {want}");
    }
}

#[test]
fn pow_on_f64() {
    let ctx = context();
    if !has_wide_math(&ctx) {
        return;
    }
    let base = [2.0f64, 9.0, 1e100, 0.5, 7.0];
    let exponent = [10.0f64, 0.5, 2.0, -3.0, 0.0];
    let b = Array::from_slice_on(&ctx, &base, 5).unwrap();
    let e = Array::from_slice_on(&ctx, &exponent, 5).unwrap();
    let got = b.pow(&e).to_vec().unwrap();
    for i in 0..5 {
        assert!(matches(got[i], base[i].powf(exponent[i]), 1e-12, 0.0), "{} ^ {} = {}", base[i], exponent[i], got[i]);
    }
    assert_eq!(b.powf(2.0).to_vec().unwrap()[1], 81.0);
    assert!(matches(b.root(2.0).to_vec().unwrap()[1], 3.0, 1e-14, 0.0));
    assert!(matches(e.rpow(2.0).to_vec().unwrap()[0], 1024.0, 1e-14, 0.0));
}

fn floor_mod(x: f64, y: f64) -> f64 {
    let r = x % y;
    if r != 0.0 && ((r < 0.0) != (y < 0.0)) { r + y } else { r }
}

#[test]
fn modulo_has_the_sign_of_the_divisor() {
    let ctx = context();
    let xs = [5.5f32, -5.5, 5.5, -5.5, 0.0, -0.0, 7.0, -7.0, 1e-3, -1e-3, 10.0, f32::NAN, 3.0, 3.0, f32::INFINITY];
    let ys = [2.0f32, 2.0, -2.0, -2.0, 3.0, 3.0, 7.0, 7.0, 1.0, 1.0, 0.0, 2.0, f32::NAN, f32::INFINITY, 2.0];
    let x = Array::from_slice_on(&ctx, &xs, xs.len()).unwrap();
    let y = Array::from_slice_on(&ctx, &ys, ys.len()).unwrap();
    let got = x.modulo(&y).to_vec().unwrap();
    let rem = (&x % &y).to_vec().unwrap();
    for i in 0..xs.len() {
        let want = floor_mod(f64::from(xs[i]), f64::from(ys[i]));
        let c_rem = f64::from(xs[i]) % f64::from(ys[i]);
        assert!(matches(f64::from(got[i]), want, 1e-6, 1e-6), "{} mod {} = {} (want {want})", xs[i], ys[i], got[i]);
        // The remainder with an infinite divisor is each backend's own (`NaN`, or the dividend).
        if ys[i].is_finite() {
            assert!(matches(f64::from(rem[i]), c_rem, 1e-6, 1e-6), "{} % {} = {} (want {c_rem})", xs[i], ys[i], rem[i]);
        }
    }
    let by_scalar = x.modulo(3.0).to_vec().unwrap();
    for i in 0..xs.len() {
        assert!(matches(f64::from(by_scalar[i]), floor_mod(f64::from(xs[i]), 3.0), 1e-6, 1e-6), "{} mod 3 = {}", xs[i], by_scalar[i]);
    }

    let ints = [7i32, -7, 7, -7, 0, 5, -5, i32::MIN, i32::MAX, 3, -1];
    let divs = [3i32, 3, -3, -3, 5, 0, 0, 7, -2, i32::MIN, i32::MAX];
    let a = Array::from_slice_on(&ctx, &ints, ints.len()).unwrap();
    let d = Array::from_slice_on(&ctx, &divs, divs.len()).unwrap();
    let got = a.modulo(&d).to_vec().unwrap();
    for i in 0..ints.len() {
        let want = if divs[i] == 0 {
            0
        } else {
            let r = ints[i].wrapping_rem(divs[i]);
            if r != 0 && ((r < 0) != (divs[i] < 0)) { r + divs[i] } else { r }
        };
        assert_eq!(got[i], want, "{} mod {}", ints[i], divs[i]);
    }

    let us = [7u32, 0, 9, u32::MAX, 5];
    let vs = [3u32, 4, 9, 10, 0];
    let a = Array::from_slice_on(&ctx, &us, 5).unwrap();
    let d = Array::from_slice_on(&ctx, &vs, 5).unwrap();
    let got = a.modulo(&d).to_vec().unwrap();
    assert_eq!(&got[..4], &[1, 0, 0, 5]);
    assert_eq!((&a % &d).to_vec().unwrap()[..4], got[..4]);

    if has_wide_math(&ctx) {
        let xs: Vec<f64> = xs.iter().map(|&v| f64::from(v)).collect();
        let ys: Vec<f64> = ys.iter().map(|&v| f64::from(v)).collect();
        let x = Array::from_slice_on(&ctx, &xs, xs.len()).unwrap();
        let y = Array::from_slice_on(&ctx, &ys, ys.len()).unwrap();
        let got = x.modulo(&y).to_vec().unwrap();
        for i in 0..xs.len() {
            assert!(matches(got[i], floor_mod(xs[i], ys[i]), 1e-12, 1e-12), "f64 {} mod {} = {}", xs[i], ys[i], got[i]);
        }
    }
}

#[test]
fn clamp_takes_array_bounds() {
    let ctx = context();
    let xs: Vec<f32> = (0..20).map(|i| i as f32 - 10.0).collect();
    let x = Array::from_slice_on(&ctx, &xs, 20).unwrap();
    let lo = Array::from_slice_on(&ctx, &(0..20).map(|i| -(i as f32) * 0.25).collect::<Vec<_>>(), 20).unwrap();
    let hi = Array::from_slice_on(&ctx, &(0..20).map(|i| 3.0 + (i % 4) as f32).collect::<Vec<_>>(), 20).unwrap();
    let got = x.clamp(&lo, &hi).to_vec().unwrap();
    for i in 0..20 {
        let (l, h) = (-(i as f32) * 0.25, 3.0 + (i % 4) as f32);
        assert_eq!(got[i], xs[i].max(l).min(h), "{i}");
    }
    // Mixed: scalar low bound, array high bound; broadcast bounds.
    let mixed = x.clamp(-2.0, &hi).to_vec().unwrap();
    for i in 0..20 {
        assert_eq!(mixed[i], xs[i].max(-2.0).min(3.0 + (i % 4) as f32));
    }
    let m = Array::from_slice_on(&ctx, &(0..12).map(|i| i as f32).collect::<Vec<_>>(), [3, 4]).unwrap();
    let col_lo = Array::from_slice_on(&ctx, &[2.0f32, 3.0, 4.0], [3, 1]).unwrap();
    let row_hi = Array::from_slice_on(&ctx, &[5.0f32, 6.0, 7.0, 8.0], [1, 4]).unwrap();
    let got = m.clamp(&col_lo, &row_hi).to_vec().unwrap();
    for j in 0..4usize {
        for i in 0..3usize {
            let want = ((i + 3 * j) as f32).max([2.0, 3.0, 4.0][i]).min([5.0, 6.0, 7.0, 8.0][j]);
            assert_eq!(got[i + 3 * j], want);
        }
    }
    // Scalars as before; NaN stays NaN.
    let n = Array::from_slice_on(&ctx, &[f32::NAN, 5.0, -5.0, f32::INFINITY], 4).unwrap();
    let got = n.clamp(-1.0, 1.0).to_vec().unwrap();
    assert!(got[0].is_nan() || got[0].abs() <= 1.0);
    assert_eq!(&got[1..], &[1.0, -1.0, 1.0]);
    // A shape that does not broadcast poisons the result.
    assert!(x.clamp(&m, 1.0).to_vec().is_err());

    let ints = Array::from_slice_on(&ctx, &[-5i32, 0, 5, 10], 4).unwrap();
    let lo = Array::from_slice_on(&ctx, &[-1i32, -1, 6, 0], 4).unwrap();
    assert_eq!(ints.clamp(&lo, 8).to_vec().unwrap(), vec![-1, 0, 6, 8]);
    let us = Array::from_slice_on(&ctx, &[1u32, 50, 100], 3).unwrap();
    assert_eq!(us.clamp(10, 60).to_vec().unwrap(), vec![10, 50, 60]);

    if has_wide_math(&ctx) {
        let w = Array::from_slice_on(&ctx, &[-1.5f64, 0.5, 9.0], 3).unwrap();
        assert_eq!(w.clamp(0.0, 1.0).to_vec().unwrap(), vec![0.0, 0.5, 1.0]);
        let lo = Array::from_slice_on(&ctx, &[-2.0f64, 1.0, 0.0], 3).unwrap();
        let hi = Array::from_slice_on(&ctx, &[-1.0f64, 2.0, 5.0], 3).unwrap();
        assert_eq!(w.clamp_arrays(&lo, &hi).to_vec().unwrap(), vec![-1.5, 1.0, 5.0]);
    }
}

#[test]
fn is_zero_tests_for_zero() {
    let ctx = context();
    let x = Array::from_slice_on(&ctx, &[0.0f32, -0.0, 1e-30, f32::NAN, -3.0], 5).unwrap();
    assert_eq!(x.is_zero().to_vec().unwrap(), vec![true, true, false, false, false]);
    let i = Array::from_slice_on(&ctx, &[0i32, 1, -1], 3).unwrap();
    assert_eq!(i.is_zero().to_vec().unwrap(), vec![true, false, false]);
    let u = Array::from_slice_on(&ctx, &[0u32, 7], 2).unwrap();
    assert_eq!(u.is_zero().to_vec().unwrap(), vec![true, false]);
}

#[test]
fn float_remainder_operator_truncates() {
    let ctx = context();
    let x = Array::from_slice_on(&ctx, &[5.5f32, -5.5, 7.0], 3).unwrap();
    assert_eq!((&x % 2.0).to_vec().unwrap(), vec![1.5, -1.5, 1.0]);
    let mut y = x.clone();
    y %= 4.0;
    assert_eq!(y.to_vec().unwrap(), vec![1.5, -1.5, 3.0]);
}

#[test]
fn poisoned_and_odd_sizes() {
    let ctx = context();
    for n in [1usize, 2, 3, 31, 33, 257] {
        let xs: Vec<f32> = sweep(n, 99, 0.1, 5.0).iter().map(|&v| v as f32).collect();
        let x = Array::from_slice_on(&ctx, &xs, n).unwrap();
        let got = x.erf().to_vec().unwrap();
        for i in 0..n {
            assert!(matches(f64::from(got[i]), libm::erf(f64::from(xs[i])), 3e-6, 1e-7));
        }
    }
    // Shape mismatches poison, they do not panic.
    let a = Array::from_slice_on(&ctx, &[1.0f32, 2.0, 3.0], 3).unwrap();
    let b = Array::from_slice_on(&ctx, &[1.0f32, 2.0], 2).unwrap();
    assert!(a.pow(&b).to_vec().is_err());
    assert!(a.modulo(&b).to_vec().is_err());
    // Integer arrays have no float math.
    let i = Array::from_slice_on(&ctx, &[1i32, 2], 2).unwrap();
    assert!(i.pow(&i).to_vec().is_err());
}

#[test]
fn f64_functions_fuse_into_reductions_and_binary_math() {
    let ctx = context();
    if !has_wide_math(&ctx) {
        return;
    }
    let xs = sweep(1000, 5, -4.0, 4.0);
    let x = Array::from_slice_on(&ctx, &xs, xs.len()).unwrap();
    let got = (x.erf() + x.lgamma().abs().log10().cosh()).sum().item().unwrap();
    let want: f64 = xs.iter().map(|&v| libm::erf(v) + libm::lgamma(v).abs().log10().cosh()).sum();
    assert!((got - want).abs() <= 1e-9 * want.abs(), "{got} vs {want}");
    let ys = sweep(1000, 6, -3.0, 3.0);
    let y = Array::from_slice_on(&ctx, &ys, ys.len()).unwrap();
    let (atan2, hypot, sign) = (x.atan2(&y).to_vec().unwrap(), x.hypot(&y).to_vec().unwrap(), x.copysign(&y).to_vec().unwrap());
    for i in 0..xs.len() {
        assert!(matches(atan2[i], xs[i].atan2(ys[i]), 1e-13, 1e-14));
        assert!(matches(hypot[i], xs[i].hypot(ys[i]), 1e-13, 0.0));
        assert_eq!(sign[i], xs[i].copysign(ys[i]));
    }
    let special = Array::from_slice_on(&ctx, &[0.0f64, -0.0, 1e300, f64::INFINITY], 4).unwrap();
    let other = Array::from_slice_on(&ctx, &[0.0f64, -0.0, 1e300, 2.0], 4).unwrap();
    let h = special.hypot(&other).to_vec().unwrap();
    assert_eq!(h[0], 0.0);
    assert!(matches(h[2], 1e300 * 2f64.sqrt(), 1e-13, 0.0));
    assert_eq!(h[3], f64::INFINITY);
    assert_eq!(special.copysign(&other).to_vec().unwrap()[1].to_bits(), (-0.0f64).to_bits());
}
