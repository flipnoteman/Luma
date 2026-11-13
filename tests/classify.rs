//! `is_finite`, `is_nan` and `is_inf`: tests of the bits of the exponent, on stored values and
//! on values the kernel computes (`0 / 0`), for `f32`, `Df32` and `f64`.

use luma::{Array, Context, Df32};

fn has_f64(ctx: &Context) -> bool {
    ctx.features().contains(luma::wgpu::Features::SHADER_F64)
}

/// The special values of `f32`: both infinities, `NaN` (also with a sign and a payload), the
/// largest finite value, the smallest normal and subnormals, both zeros, ordinary numbers.
fn specials() -> Vec<f32> {
    vec![
        0.0,
        -0.0,
        1.5,
        -2.0e-3,
        f32::NAN,
        -f32::NAN,
        f32::from_bits(0x7fc0_1234),
        f32::from_bits(0xffa0_0001),
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::MAX,
        f32::MIN,
        f32::MIN_POSITIVE,
        1.0e-40,
        -1.0e-40,
        f32::from_bits(1),
        3.0e38,
        1.0e30,
    ]
}

fn masks(values: &[f32]) -> [Vec<u32>; 3] {
    [
        values.iter().map(|v| u32::from(v.is_finite())).collect(),
        values.iter().map(|v| u32::from(v.is_nan())).collect(),
        values.iter().map(|v| u32::from(v.is_infinite())).collect(),
    ]
}

#[test]
fn f32_classes_follow_the_bits() {
    let ctx = Context::new().unwrap();
    let data = specials();
    let x = Array::from_slice_on(&ctx, &data, data.len()).unwrap();
    let [finite, nan, inf] = masks(&data);
    assert_eq!(x.is_finite().to_vec().unwrap(), finite);
    assert_eq!(x.is_nan().to_vec().unwrap(), nan);
    assert_eq!(x.is_inf().to_vec().unwrap(), inf);
    // The three classes partition the values.
    let total = x.is_finite() + x.is_nan() + x.is_inf();
    assert_eq!(total.to_vec().unwrap(), vec![1; data.len()]);
}

#[test]
fn classes_of_computed_values_and_fusion() {
    let ctx = Context::new().unwrap();
    // 0 / 0 = NaN, 1 / 0 = inf, -1 / 0 = -inf, 4 / 2 = 2: produced by the kernel itself.
    let a = Array::from_slice_on(&ctx, &[0.0f32, 1.0, -1.0, 4.0], 4).unwrap();
    let b = Array::from_slice_on(&ctx, &[0.0f32, 0.0, 0.0, 2.0], 4).unwrap();
    let q = &a / &b;
    assert_eq!(q.is_nan().to_vec().unwrap(), vec![1, 0, 0, 0]);
    assert_eq!(q.is_inf().to_vec().unwrap(), vec![0, 1, 1, 0]);
    assert_eq!(q.is_finite().to_vec().unwrap(), vec![0, 0, 0, 1]);

    // The positivity damping of a solver: the new value where it is finite, else the old.
    let new = Array::from_slice_on(&ctx, &[1.0f32, f32::NAN, 3.0, f32::INFINITY], 4).unwrap();
    let old = Array::from_slice_on(&ctx, &[10.0f32, 20.0, 30.0, 40.0], 4).unwrap();
    let ok = (&new * 2.0).is_finite();
    let damped = luma::select(&ok, &(&new * 2.0), &old);
    assert_eq!(damped.to_vec().unwrap(), vec![2.0, 20.0, 6.0, 40.0]);
    // Both components of a pair finite.
    let two = Array::from_slice_on(&ctx, &[1.0f32, 2.0, f32::NAN, 4.0, 5.0, f32::NEG_INFINITY], [2, 3]).unwrap();
    let both = two.slice(0, ..1).is_finite() * two.slice(0, 1..).is_finite();
    assert_eq!(both.to_vec().unwrap(), vec![1, 0, 0]);
}

#[test]
fn df32_classes_follow_the_high_word() {
    let ctx = Context::new().unwrap();
    let data: Vec<Df32> = specials().into_iter().map(|hi| Df32 { hi, lo: 0.0 }).collect();
    let x = Array::from_slice_on(&ctx, &data, data.len()).unwrap();
    let [finite, nan, inf] = masks(&specials());
    assert_eq!(x.is_finite().to_vec().unwrap(), finite);
    assert_eq!(x.is_nan().to_vec().unwrap(), nan);
    assert_eq!(x.is_inf().to_vec().unwrap(), inf);

    // Computed in the library: a double-float quotient.
    let one = Array::<Df32>::from_slice_on(&ctx, &[Df32::from(1.0f32), Df32::from(1.0f32)], 2).unwrap();
    let den = Array::<Df32>::from_slice_on(&ctx, &[Df32::from(4.0f32), Df32::from(3.0f32)], 2).unwrap();
    assert_eq!((&one / &den).is_finite().to_vec().unwrap(), vec![1, 1]);
}

#[test]
fn f64_classes_cover_the_range_beyond_f32() {
    let ctx = Context::new().unwrap();
    if !has_f64(&ctx) {
        return;
    }
    let data: Vec<f64> = vec![
        0.0,
        -0.0,
        1.5,
        f64::NAN,
        -f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
        // Finite, though not as `f32`.
        f64::MAX,
        f64::MIN,
        1.0e300,
        3.5e38,
        f64::MIN_POSITIVE,
        5e-324,
        1.0e-310,
    ];
    let x = Array::from_slice_on(&ctx, &data, data.len()).unwrap();
    let want = |f: fn(&f64) -> bool| data.iter().map(|v| u32::from(f(v))).collect::<Vec<_>>();
    assert_eq!(x.is_finite().to_vec().unwrap(), want(|v| v.is_finite()));
    assert_eq!(x.is_nan().to_vec().unwrap(), want(|v| v.is_nan()));
    assert_eq!(x.is_inf().to_vec().unwrap(), want(|v| v.is_infinite()));
}

#[test]
fn classes_of_values_the_kernel_computes_in_every_type() {
    let ctx = Context::new().unwrap();
    // x * x overflows to infinity at 1e30, stays NaN and infinite, and is finite at 2.
    let data = [f32::NAN, f32::INFINITY, 1.0e30, 2.0, -1.0e30, 0.0];
    let want_nan = vec![1, 0, 0, 0, 0, 0];
    let want_inf = vec![0, 1, 1, 0, 1, 0];
    let x = Array::from_slice_on(&ctx, &data, data.len()).unwrap();
    let squared = &x * &x;
    assert_eq!(squared.is_nan().to_vec().unwrap(), want_nan);
    assert_eq!(squared.is_inf().to_vec().unwrap(), want_inf);
    assert_eq!(squared.is_finite().to_vec().unwrap(), vec![0, 0, 0, 1, 0, 1]);

    if has_f64(&ctx) {
        let wide = x.to_f64();
        // 1e30 squared is finite as `f64`.
        let squared = &wide * &wide;
        assert_eq!(squared.is_nan().to_vec().unwrap(), want_nan);
        assert_eq!(squared.is_inf().to_vec().unwrap(), vec![0, 1, 0, 0, 0, 0]);
        assert_eq!(squared.is_finite().to_vec().unwrap(), vec![0, 0, 1, 1, 1, 1]);
    }

    let df = x.to_df32();
    let squared = &df * &df;
    assert_eq!(squared.is_nan().to_vec().unwrap()[0], 1);
    assert_eq!(squared.is_finite().to_vec().unwrap()[3], 1);
    assert_eq!(squared.is_finite().to_vec().unwrap()[5], 1);
}
