//! The `Df32` arithmetic library in each of its forms: products from a fused `fma` or from
//! Dekker's splitting, and the cheaper additions `fast_math` allows. Each form is checked against
//! `f64` arithmetic on the exact values of its inputs.

use luma::{Array, Context, ContextOptions, Df32};

/// A tiny deterministic generator (xorshift) of values in `[lo, hi)`.
struct Random(u64);

impl Random {
    fn next(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }

    fn df32(&mut self, lo: f64, hi: f64) -> Df32 {
        Df32::from(lo + (hi - lo) * self.next())
    }
}

fn context(options: ContextOptions) -> Context {
    Context::with_options(options.calibrate(false)).unwrap()
}

/// The forms the library takes: (name, options).
fn forms() -> Vec<(&'static str, ContextOptions)> {
    vec![
        ("default", ContextOptions::from_env()),
        ("dekker", ContextOptions::from_env().df_fma(false)),
        ("fast_math", ContextOptions::from_env().fast_math(true)),
    ]
}

const N: usize = 4096;

fn inputs(seed: u64, lo: f64, hi: f64) -> Vec<Df32> {
    let mut random = Random(seed);
    (0..N).map(|_| random.df32(lo, hi)).collect()
}

/// Relative error of `got` against the exact `want`.
fn relative(got: Df32, want: f64) -> f64 {
    ((got.to_f64() - want) / want).abs()
}

/// 44 bits: what the documentation of `Df32` promises of its arithmetic.
const BITS_44: f64 = 1.0 / (1u64 << 44) as f64;

#[test]
fn products_quotients_and_roots_are_accurate_to_44_bits_in_every_form() {
    let (xs, ys) = (inputs(1, 0.25, 4.0), inputs(2, 0.25, 4.0));
    for (name, options) in forms() {
        let ctx = context(options);
        let x = Array::from_slice_on(&ctx, &xs, N).unwrap();
        let y = Array::from_slice_on(&ctx, &ys, N).unwrap();
        let product = (&x * &y).to_vec().unwrap();
        let quotient = (&x / &y).to_vec().unwrap();
        let root = x.sqrt().to_vec().unwrap();
        for i in 0..N {
            let (a, b) = (xs[i].to_f64(), ys[i].to_f64());
            assert!(relative(product[i], a * b) < BITS_44, "{name}: {a} * {b} = {}", product[i].to_f64());
            assert!(relative(quotient[i], a / b) < BITS_44, "{name}: {a} / {b} = {}", quotient[i].to_f64());
            assert!(relative(root[i], a.sqrt()) < BITS_44, "{name}: sqrt({a}) = {}", root[i].to_f64());
        }
    }
}

#[test]
fn sums_are_accurate_to_44_bits_of_the_result_by_default_and_of_the_operands_with_fast_math() {
    // Mixed signs, so that many sums cancel by a few bits and some by many.
    let (xs, ys) = (inputs(3, -4.0, 4.0), inputs(4, -4.0, 4.0));
    // Nearly equal operands of opposite sign: a cancellation of 20 bits and more.
    let near: Vec<Df32> = xs.iter().zip(inputs(5, 0.0, 1.0)).map(|(x, r)| Df32::from(-x.to_f64() * (1.0 + 1e-6 * r.to_f64()))).collect();
    for (name, options) in forms() {
        let ctx = context(options);
        let x = Array::from_slice_on(&ctx, &xs, N).unwrap();
        let y = Array::from_slice_on(&ctx, &ys, N).unwrap();
        let z = Array::from_slice_on(&ctx, &near, N).unwrap();
        let sums = [(&x + &y).to_vec().unwrap(), (&x - &y).to_vec().unwrap()];
        let cancelled = (&x + &z).to_vec().unwrap();
        let accurate = name != "fast_math";
        for i in 0..N {
            let (a, b, c) = (xs[i].to_f64(), ys[i].to_f64(), near[i].to_f64());
            for (got, want) in [(sums[0][i], a + b), (sums[1][i], a - b)] {
                let bound = if accurate { BITS_44 * want.abs() } else { BITS_44 * a.abs().max(b.abs()) };
                assert!((got.to_f64() - want).abs() <= bound, "{name}: {a} +- {b} = {}", got.to_f64());
            }
            let want = a + c;
            let bound = if accurate { BITS_44 * want.abs() } else { BITS_44 * a.abs().max(c.abs()) };
            assert!((cancelled[i].to_f64() - want).abs() <= bound, "{name}: {a} + {c} = {}, want {want}", cancelled[i].to_f64());
        }
    }
}

#[test]
fn the_fused_and_the_split_product_agree_to_the_bit() {
    // Both give the exact error of a product, so everything built on it is the same; a device
    // that fuses `fma` is needed for the comparison (the others take the split form in both).
    let fused = context(ContextOptions::from_env());
    let split = context(ContextOptions::from_env().df_fma(false));
    let (xs, ys) = (inputs(6, 0.25, 4.0), inputs(7, -4.0, 4.0));
    let run = |ctx: &Context| {
        let x = Array::from_slice_on(ctx, &xs, N).unwrap();
        let y = Array::from_slice_on(ctx, &ys, N).unwrap();
        ((&x * &y).to_vec().unwrap(), (&y / &x).to_vec().unwrap(), x.sqrt().to_vec().unwrap())
    };
    let (a, b) = (run(&fused), run(&split));
    let bits = |v: &[Df32]| v.iter().map(|d| (d.hi.to_bits(), d.lo.to_bits())).collect::<Vec<_>>();
    assert_eq!(bits(&a.0), bits(&b.0), "products");
    assert_eq!(bits(&a.1), bits(&b.1), "quotients");
    assert_eq!(bits(&a.2), bits(&b.2), "roots");
}

#[test]
fn ordering_and_absolute_values_follow_the_value_not_the_words() {
    let ctx = context(ContextOptions::from_env());
    // Equal high words, ordered by the low words; zeros of both signs; negative values.
    let data = vec![
        Df32 { hi: 1.0, lo: -1e-9 },
        Df32 { hi: 1.0, lo: 1e-9 },
        Df32 { hi: 1.0, lo: 1e-9 },
        Df32 { hi: -1.0, lo: 1e-9 },
        Df32 { hi: 0.0, lo: 0.0 },
        Df32 { hi: -0.0, lo: 0.0 },
        Df32 { hi: -2.0, lo: -1e-9 },
        Df32 { hi: 2.0, lo: 0.0 },
    ];
    let x = Array::from_slice_on(&ctx, &data, data.len()).unwrap();
    let y = Array::from_slice_on(&ctx, &[data[1], data[0], data[2], data[3], data[5], data[4], data[7], data[6]], data.len()).unwrap();
    let value = |d: Df32| d.to_f64();
    let want = |f: fn(f64, f64) -> bool| -> Vec<u32> { (0..data.len()).map(|i| u32::from(f(value(data[i]), value([data[1], data[0], data[2], data[3], data[5], data[4], data[7], data[6]][i])))).collect() };
    assert_eq!(x.lt(&y).to_u32().to_vec().unwrap(), want(|a, b| a < b));
    assert_eq!(x.le(&y).to_u32().to_vec().unwrap(), want(|a, b| a <= b));
    assert_eq!(x.gt(&y).to_u32().to_vec().unwrap(), want(|a, b| a > b));
    assert_eq!(x.ge(&y).to_u32().to_vec().unwrap(), want(|a, b| a >= b));
    assert_eq!(x.eq(&y).to_u32().to_vec().unwrap(), want(|a, b| a == b));
    let low = x.minimum(&y).to_vec().unwrap();
    let high = x.maximum(&y).to_vec().unwrap();
    let abs = x.abs().to_vec().unwrap();
    for i in 0..data.len() {
        let (a, b) = (value(data[i]), value([data[1], data[0], data[2], data[3], data[5], data[4], data[7], data[6]][i]));
        assert_eq!((value(low[i]), value(high[i])), (a.min(b), a.max(b)), "element {i}");
        assert_eq!(value(abs[i]), a.abs(), "element {i}");
    }
}

/// Log-uniform values in `[lo, hi)` (`0 < lo`), with the sign of `sign`.
fn log_inputs(seed: u64, lo: f64, hi: f64) -> Vec<Df32> {
    let mut random = Random(seed);
    (0..N).map(|_| Df32::from((lo.ln() + (hi.ln() - lo.ln()) * random.next()).exp())).collect()
}

/// The largest relative error, and its input, of `op` over `xs` against the `f64` function `host`
/// of the exact value of each input.
fn worst(ctx: &Context, xs: &[Df32], op: impl Fn(&Array<Df32>) -> Array<Df32>, host: impl Fn(f64) -> f64) -> (f64, f64) {
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

const BITS_43: f64 = 2.0 / (1u64 << 44) as f64;

/// The accuracy the math functions promise: a relative error below `1e-13` (`2^-43`).
fn assert_accurate(name: &str, form: &str, (error, at): (f64, f64)) {
    eprintln!("{form:>9} {name:<28} max relative error {error:.2e} at {at}");
    assert!(error < BITS_43, "{form}: {name} at {at}: relative error {error:e}");
}

#[test]
fn exp_ln_and_tanh_are_accurate_to_43_bits_in_every_form() {
    for (form, options) in forms() {
        let ctx = context(options);
        let mut random = Random(11);
        // Below about -55 the result is under 1e-24, where a low word lost to underflow (under
        // 1.2e-38) is more than 1e-13 of it.
        let wide: Vec<Df32> = (0..N).map(|_| random.df32(-55.0, 80.0)).collect();
        assert_accurate("exp [-55, 80]", form, worst(&ctx, &wide, |x| x.exp(), f64::exp));
        let small = inputs(12, -0.5, 0.5);
        assert_accurate("exp [-0.5, 0.5]", form, worst(&ctx, &small, |x| x.exp(), f64::exp));
        // Inputs under 1e-24 have lost bits of their low word (under 1.2e-38) before `ln` sees them.
        let logs = log_inputs(13, 1e-24, 1e30);
        assert_accurate("ln [1e-24, 1e30]", form, worst(&ctx, &logs, |x| x.ln(), f64::ln));
        let near_one = inputs(14, 0.5, 2.0);
        assert_accurate("ln [0.5, 2]", form, worst(&ctx, &near_one, |x| x.ln(), f64::ln));
        let tiny: Vec<Df32> = (0..N).map(|_| Df32::from(1.0 + 1e-7 * (random.next() - 0.5))).collect();
        assert_accurate("ln near 1", form, worst(&ctx, &tiny, |x| x.ln(), f64::ln));
        let tanh = inputs(15, -30.0, 30.0);
        assert_accurate("tanh [-30, 30]", form, worst(&ctx, &tanh, |x| x.tanh(), f64::tanh));
        let tanh_small = log_inputs(16, 1e-6, 1.0);
        assert_accurate("tanh [1e-6, 1]", form, worst(&ctx, &tanh_small, |x| x.tanh(), f64::tanh));
    }
}

#[test]
fn powf_is_accurate_to_43_bits_for_the_typical_exponents_in_every_form() {
    for (form, options) in forms() {
        let ctx = context(options);
        let temperatures = inputs(21, 1.0, 3000.0);
        for exponent in [1.5, 0.7, -0.5, 2.5, 3.0] {
            let name = format!("T^{exponent}, T in [1, 3000]");
            assert_accurate(&name, form, worst(&ctx, &temperatures, |x| x.powf(Df32::from(exponent)), |t| t.powf(exponent)));
        }
        let wide = log_inputs(22, 1e-6, 1e6);
        assert_accurate("x^0.3, x in [1e-6, 1e6]", form, worst(&ctx, &wide, |x| x.powf(Df32::from(0.3)), |t| t.powf(0.3)));
        let precise = Df32::from(std::f64::consts::PI / 3.0);
        assert_accurate("x^(pi/3), x in [1e-3, 1e3]", form, worst(&ctx, &log_inputs(23, 1e-3, 1e3), |x| x.powf(precise), |t| t.powf(precise.to_f64())));
    }
}

fn special(ctx: &Context, xs: &[f64], op: impl Fn(&Array<Df32>) -> Array<Df32>) -> Vec<f64> {
    let data: Vec<Df32> = xs.iter().map(|&v| Df32::from(v)).collect();
    let x = Array::from_slice_on(ctx, &data, data.len()).unwrap();
    op(&x).to_vec().unwrap().iter().map(|d| d.to_f64()).collect()
}

#[test]
fn the_math_functions_handle_zeros_infinities_and_nans_like_f32() {
    let ctx = context(ContextOptions::from_env());
    let (inf, nan) = (f64::INFINITY, f64::NAN);

    let got = special(&ctx, &[nan, inf, -inf, 100.0, -100.0, 0.0, 88.0, -87.0], |x| x.exp());
    assert!(got[0].is_nan() && got[1] == inf && got[2] == 0.0 && got[3] == inf && got[4] == 0.0 && got[5] == 1.0, "{got:?}");
    assert!((got[6] / 88f64.exp() - 1.0).abs() < 1e-13 && (got[7] / (-87f64).exp() - 1.0).abs() < 1e-6, "{got:?}");

    let got = special(&ctx, &[0.0, -1.0, inf, nan, 1.0, -inf, 3.4e38], |x| x.ln());
    assert!(got[0] == -inf && got[1].is_nan() && got[2] == inf && got[3].is_nan() && got[4] == 0.0 && got[5].is_nan(), "{got:?}");
    assert!((got[6] / 3.4e38f64.ln() - 1.0).abs() < 1e-13, "{got:?}");

    let got = special(&ctx, &[inf, -inf, 25.0, -25.0, 0.0, nan, 19.9], |x| x.tanh());
    assert!(got[0] == 1.0 && got[1] == -1.0 && got[2] == 1.0 && got[3] == -1.0 && got[4] == 0.0 && got[5].is_nan(), "{got:?}");
    assert!((got[6] / 19.9f64.tanh() - 1.0).abs() < 1e-13, "{got:?}");
    let signed = special(&ctx, &[-0.0], |x| x.tanh());
    assert!(signed[0] == 0.0 && signed[0].is_sign_negative(), "tanh(-0) = {signed:?}");

    let bases = [0.0, 0.0, -2.0, inf, inf, 3.0, nan, 1.0, 2.0, 2.0, 0.5, 1.0, 4.0];
    let powers = [2.0, -1.0, 2.0, 2.0, -1.0, 0.0, 0.0, nan, 1e31, -1e31, inf, inf, 0.5];
    let got: Vec<f64> = bases
        .iter()
        .zip(&powers)
        .map(|(&b, &p)| special(&ctx, &[b], |x| x.powf(Df32::from(p)))[0])
        .collect();
    // pow(0, 2) = 0, pow(0, -1) = inf, a negative base is NaN, inf^2 = inf, inf^-1 = 0, x^0 = 1 (also NaN^0),
    // 1^NaN is NaN here, 2^1e31 = inf, 2^-1e31 = 0, 0.5^inf = 0, 1^inf = 1, 4^0.5 = 2.
    assert!(got[0] == 0.0 && got[1] == inf && got[2].is_nan() && got[3] == inf && got[4] == 0.0 && got[5] == 1.0 && got[6] == 1.0, "{got:?}");
    assert!(got[7].is_nan() && got[8] == inf && got[9] == 0.0 && got[10] == 0.0 && got[11] == 1.0, "{got:?}");
    assert!((got[12] - 2.0).abs() < 1e-13, "{got:?}");
}

#[test]
fn the_math_functions_fuse_with_the_arithmetic_around_them() {
    let ctx = context(ContextOptions::from_env());
    let xs = inputs(31, 0.5, 4.0);
    let x = Array::from_slice_on(&ctx, &xs, N).unwrap();
    // One expression of all four functions with arithmetic between them.
    let got = (((&x * Df32::from(0.5)).exp() + Df32::from(1.0)).ln() * x.tanh() + x.powf(Df32::from(1.5)).sqrt()).to_vec().unwrap();
    for (g, v) in got.iter().zip(&xs) {
        let a = v.to_f64();
        let want = ((a * 0.5).exp() + 1.0).ln() * a.tanh() + a.powf(1.5).sqrt();
        assert!(relative(*g, want) < BITS_43, "{a}: {} vs {want}", g.to_f64());
    }
}
