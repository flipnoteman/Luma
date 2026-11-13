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
    assert_eq!(x.lt(&y).to_vec().unwrap(), want(|a, b| a < b));
    assert_eq!(x.le(&y).to_vec().unwrap(), want(|a, b| a <= b));
    assert_eq!(x.gt(&y).to_vec().unwrap(), want(|a, b| a > b));
    assert_eq!(x.ge(&y).to_vec().unwrap(), want(|a, b| a >= b));
    assert_eq!(x.eq(&y).to_vec().unwrap(), want(|a, b| a == b));
    let low = x.minimum(&y).to_vec().unwrap();
    let high = x.maximum(&y).to_vec().unwrap();
    let abs = x.abs().to_vec().unwrap();
    for i in 0..data.len() {
        let (a, b) = (value(data[i]), value([data[1], data[0], data[2], data[3], data[5], data[4], data[7], data[6]][i]));
        assert_eq!((value(low[i]), value(high[i])), (a.min(b), a.max(b)), "element {i}");
        assert_eq!(value(abs[i]), a.abs(), "element {i}");
    }
}
