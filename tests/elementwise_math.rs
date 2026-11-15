//! `trunc`, `round`, `log1p` and `expm1` against `f64` host functions: rounding modes, accuracy
//! near 0 where the naive formulas lose their digits, and the special values.

use luma::{Array, Context, ContextOptions, Df32};

fn has_f64(ctx: &Context) -> bool {
    ctx.features().contains(luma::wgpu::Features::SHADER_F64)
}

/// A tiny deterministic generator (xorshift) of values in `[0, 1)`.
struct Random(u64);

impl Random {
    fn next(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Log-uniform magnitude in `[lo, hi)` with a random sign if `signed`.
    fn log_uniform(&mut self, lo: f64, hi: f64, signed: bool) -> f64 {
        let v = (lo.ln() + (hi.ln() - lo.ln()) * self.next()).exp();
        if signed && self.next() < 0.5 { -v } else { v }
    }
}

#[test]
fn trunc_and_round_follow_numpy_on_f32_and_f64() {
    let ctx = Context::new().unwrap();
    let data = [-2.5f32, -1.5, -0.5, -0.4, -0.0, 0.0, 0.4, 0.5, 1.5, 2.5, 3.5, 2.7, -2.7, 8388609.0, -1e20, f32::INFINITY, f32::NEG_INFINITY];
    let x = Array::from_slice_on(&ctx, &data, data.len()).unwrap();
    let round = x.round().to_vec().unwrap();
    let trunc = x.trunc().to_vec().unwrap();
    for (i, &v) in data.iter().enumerate() {
        assert_eq!(round[i].to_bits(), v.round_ties_even().to_bits(), "round({v}) = {}", round[i]);
        assert_eq!(trunc[i].to_bits(), v.trunc().to_bits(), "trunc({v}) = {}", trunc[i]);
    }
    let nan = Array::from_slice_on(&ctx, &[f32::NAN], 1).unwrap();
    assert!(nan.round().item().unwrap().is_nan() && nan.trunc().item().unwrap().is_nan());

    if has_f64(&ctx) {
        let wide: Vec<f64> = data.iter().map(|&v| f64::from(v) * 1.0000000001).chain([0.5, 1.5, 2.5, -2.5, 4503599627370497.5]).collect();
        let x = Array::from_slice_on(&ctx, &wide, wide.len()).unwrap();
        let (round, trunc) = (x.round().to_vec().unwrap(), x.trunc().to_vec().unwrap());
        let (floor, ceil) = (x.floor().to_vec().unwrap(), x.ceil().to_vec().unwrap());
        for (i, &v) in wide.iter().enumerate() {
            assert_eq!(round[i], v.round_ties_even(), "f64 round({v})");
            assert_eq!(trunc[i], v.trunc(), "f64 trunc({v})");
            assert_eq!(floor[i], v.floor(), "f64 floor({v})");
            assert_eq!(ceil[i], v.ceil(), "f64 ceil({v})");
        }
    }
}

#[test]
fn rounding_fuses_into_its_consumers() {
    let ctx = Context::new().unwrap();
    let x = Array::from_slice_on(&ctx, &[0.5f32, 1.5, 2.5, -0.5, 7.25], 5).unwrap();
    // round(x) -> to_u32 -> sum: one expression, no stored intermediate.
    let ((), profile) = ctx
        .profile(|| {
            assert_eq!(x.abs().round().sum().item().unwrap(), 0.0 + 2.0 + 2.0 + 0.0 + 7.0);
        })
        .unwrap();
    if ctx.profiling_supported() {
        assert!(profile.launches() <= 2, "{profile:?}");
    }
}

/// The worst relative error of `op` against the `f64` function `host` over `xs`.
fn worst(ctx: &Context, xs: &[f32], op: impl Fn(&Array<f32>) -> Array<f32>, host: impl Fn(f64) -> f64) -> (f64, f32) {
    let x = Array::from_slice_on(ctx, xs, xs.len()).unwrap();
    let got = op(&x).to_vec().unwrap();
    let mut worst = (0.0f64, 0.0f32);
    for (g, &v) in got.iter().zip(xs) {
        let want = host(f64::from(v));
        // The reference of an exact input; where it rounds to zero or infinity in f32 the result must too.
        let want32 = want as f32;
        if want32 == 0.0 || want32.is_infinite() {
            assert_eq!(*g, want32, "at {v}");
            continue;
        }
        let error = ((f64::from(*g) - want) / want).abs();
        if error > worst.0 {
            worst = (error, v);
        }
    }
    worst
}

#[test]
fn log1p_and_expm1_are_accurate_near_zero_and_everywhere() {
    let ctx = Context::new().unwrap();
    let mut random = Random(5);
    let tiny: Vec<f32> = (0..4096).map(|_| random.log_uniform(1e-12, 0.5, true) as f32).collect();
    let mid: Vec<f32> = (0..4096).map(|_| random.log_uniform(0.4, 90.0, true) as f32).collect();
    for (name, xs) in [("tiny", &tiny), ("wide", &mid)] {
        // log1p is defined above -1; stay clear of the pole in the "wide" set.
        let logs: Vec<f32> = xs.iter().map(|&v| if v <= -1.0 { v / -(v.abs() + 1.0) * 0.99 } else { v }).collect();
        let (error, at) = worst(&ctx, &logs, |x| x.log1p(), f64::ln_1p);
        eprintln!("log1p {name}: max relative error {error:.2e} at {at}");
        assert!(error < 1e-5, "log1p {name}: {error:e} at {at}");
        let (error, at) = worst(&ctx, xs, |x| x.expm1(), f64::exp_m1);
        eprintln!("expm1 {name}: max relative error {error:.2e} at {at}");
        assert!(error < 1e-5, "expm1 {name}: {error:e} at {at}");
    }
    // The point of the functions: where the naive formulas have no digits left.
    let x = Array::from_slice_on(&ctx, &[1e-8f32, -1e-8, 1e-5], 3).unwrap();
    let naive = (&x + 1.0).ln().to_vec().unwrap();
    let good = x.log1p().to_vec().unwrap();
    assert!(good.iter().zip([1e-8f32, -1e-8, 1e-5]).all(|(g, v)| ((g - v) / v).abs() < 1e-5), "{good:?}");
    assert!(((naive[2] - 1e-5) / 1e-5).abs() > 1e-4 || naive[0] == 0.0, "{naive:?}");
    let good = x.expm1().to_vec().unwrap();
    assert!(good.iter().zip([1e-8f32, -1e-8, 1e-5]).all(|(g, v)| ((g - v) / v).abs() < 1e-5), "{good:?}");
}

#[test]
fn log1p_and_expm1_handle_zeros_infinities_and_nans() {
    let ctx = Context::new().unwrap();
    let (inf, nan) = (f32::INFINITY, f32::NAN);
    let data = [0.0f32, -0.0, -1.0, -2.0, inf, -inf, nan, 3.0e38, -0.999999, 1e-30];
    let x = Array::from_slice_on(&ctx, &data, data.len()).unwrap();
    let l = x.log1p().to_vec().unwrap();
    assert!(l[0] == 0.0 && l[0].is_sign_positive() && l[1] == 0.0 && l[1].is_sign_negative(), "{l:?}");
    assert!(l[2] == f32::NEG_INFINITY && l[3].is_nan() && l[4] == inf && l[5].is_nan() && l[6].is_nan(), "{l:?}");
    assert!(((l[7] - 3.0e38f32.ln()) / 3.0e38f32.ln()).abs() < 1e-6 && ((f64::from(l[8]) / f64::from(data[8]).ln_1p()) - 1.0).abs() < 1e-5 && l[9] == 1e-30, "{l:?}");
    let data = [0.0f32, -0.0, inf, -inf, nan, 100.0, -100.0, 88.0, 1e-30];
    let x = Array::from_slice_on(&ctx, &data, data.len()).unwrap();
    let e = x.expm1().to_vec().unwrap();
    assert!(e[0] == 0.0 && e[1] == 0.0 && e[1].is_sign_negative(), "{e:?}");
    assert!(e[2] == inf && e[3] == -1.0 && e[4].is_nan() && e[5] == inf && e[6] == -1.0 && e[8] == 1e-30, "{e:?}");
    assert!(((f64::from(e[7]) / 88f64.exp_m1()) - 1.0).abs() < 1e-5, "{e:?}");
}

fn context(options: ContextOptions) -> Context {
    Context::with_options(options.calibrate(false)).unwrap()
}

/// 43 bits, as the other `Df32` math functions are held to.
const BITS_43: f64 = 2.0 / (1u64 << 44) as f64;

fn worst_df(ctx: &Context, xs: &[Df32], op: impl Fn(&Array<Df32>) -> Array<Df32>, host: impl Fn(f64) -> f64) -> (f64, f64) {
    let x = Array::from_slice_on(ctx, xs, xs.len()).unwrap();
    let got = op(&x).to_vec().unwrap();
    let mut worst = (0.0f64, 0.0f64);
    for (g, x) in got.iter().zip(xs) {
        let (a, want) = (x.to_f64(), host(x.to_f64()));
        let error = ((g.to_f64() - want) / want).abs();
        assert!(error.is_finite(), "{a}: got {}, want {want}", g.to_f64());
        if error > worst.0 {
            worst = (error, a);
        }
    }
    worst
}

#[test]
fn df32_log1p_and_expm1_are_accurate_to_43_bits_in_every_form() {
    for (form, options) in [
        ("default", ContextOptions::from_env()),
        ("dekker", ContextOptions::from_env().df_fma(false)),
        ("fast_math", ContextOptions::from_env().fast_math(true)),
    ] {
        let ctx = context(options);
        let mut random = Random(9);
        // Inputs of a double-float: a high word and a low word below its half ulp.
        let mut df = |lo: f64, hi: f64, signed: bool| -> Vec<Df32> { (0..4096).map(|_| Df32::from(random.log_uniform(lo, hi, signed))).collect() };
        let sets = [
            ("tiny", df(1e-20, 1e-3, true)),
            ("small", df(1e-3, 0.5, true)),
            ("around 1", df(0.3, 3.0, false)),
            ("large", df(3.0, 80.0, false)),
        ];
        for (name, xs) in &sets {
            for (op, f, label) in [(0, f64::exp_m1 as fn(f64) -> f64, "expm1"), (1, f64::ln_1p, "log1p")] {
                // `log1p` needs the inputs above -1; flip the large negative ones.
                let xs: Vec<Df32> = xs.iter().map(|x| if op == 1 && x.to_f64() <= -0.9 { Df32::from(-0.9 * x.to_f64().signum() * 0.5) } else { *x }).collect();
                let (error, at) = if op == 0 { worst_df(&ctx, &xs, |x| x.expm1(), f) } else { worst_df(&ctx, &xs, |x| x.log1p(), f) };
                eprintln!("{form:>9} {label} {name}: max relative error {error:.2e} at {at}");
                assert!(error < BITS_43, "{form}: {label} {name} at {at}: {error:e}");
            }
        }
        // Near the pole of log1p, and negative expm1 down to the flush of the low word.
        let near: Vec<Df32> = (0..512).map(|_| Df32::from(-1.0 + random.log_uniform(1e-6, 0.3, false))).collect();
        let (error, at) = worst_df(&ctx, &near, |x| x.log1p(), f64::ln_1p);
        assert!(error < 4.0 * BITS_43, "{form}: log1p near -1 at {at}: {error:e}");
        let negative: Vec<Df32> = (0..512).map(|_| Df32::from(-random.log_uniform(1e-3, 30.0, false))).collect();
        let (error, at) = worst_df(&ctx, &negative, |x| x.expm1(), f64::exp_m1);
        assert!(error < 4.0 * BITS_43, "{form}: expm1 of negatives at {at}: {error:e}");
    }
}

#[test]
fn df32_log1p_and_expm1_handle_the_special_values() {
    let ctx = context(ContextOptions::from_env());
    let (inf, nan) = (f64::INFINITY, f64::NAN);
    let run = |xs: &[f64], op: fn(&Array<Df32>) -> Array<Df32>| -> Vec<f64> {
        let data: Vec<Df32> = xs.iter().map(|&v| Df32::from(v)).collect();
        let x = Array::from_slice_on(&ctx, &data, data.len()).unwrap();
        op(&x).to_vec().unwrap().iter().map(|d| d.to_f64()).collect()
    };
    let l = run(&[0.0, -1.0, -2.0, inf, -inf, nan, 3.4e38], |x| x.log1p());
    assert!(l[0] == 0.0 && l[1] == -inf && l[2].is_nan() && l[3] == inf && l[4].is_nan() && l[5].is_nan(), "{l:?}");
    assert!((l[6] / 3.4e38f64.ln() - 1.0).abs() < 1e-13, "{l:?}");
    let e = run(&[0.0, inf, -inf, nan, 100.0, -100.0, 88.0], |x| x.expm1());
    assert!(e[0] == 0.0 && e[1] == inf && e[2] == -1.0 && e[3].is_nan() && e[4] == inf && e[5] == -1.0, "{e:?}");
    assert!((e[6] / 88f64.exp_m1() - 1.0).abs() < 1e-13, "{e:?}");
}
