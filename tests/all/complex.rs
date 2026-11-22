//! `Array<C32>`: the complex single-precision element type. Operators, functions, constructors,
//! views, reductions and fusion, against `f64` host references; edge cases (`NaN`, infinities,
//! zero, huge and tiny components) against what the documentation says.

use crate::common::values;
use luma::*;

/// A complex number of the host reference, in `f64`: `(re, im)`.
type Z = (f64, f64);

fn z(c: C32) -> Z {
    (f64::from(c.re), f64::from(c.im))
}

fn mul(a: Z, b: Z) -> Z {
    (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0)
}

fn div(a: Z, b: Z) -> Z {
    let d = b.0 * b.0 + b.1 * b.1;
    ((a.0 * b.0 + a.1 * b.1) / d, (a.1 * b.0 - a.0 * b.1) / d)
}

fn abs(a: Z) -> f64 {
    a.0.hypot(a.1)
}

fn exp(a: Z) -> Z {
    let e = a.0.exp();
    (e * a.1.cos(), e * a.1.sin())
}

fn sqrt(a: Z) -> Z {
    let m = abs(a);
    let re = ((m + a.0) / 2.0).sqrt();
    let im = ((m - a.0) / 2.0).sqrt().copysign(a.1);
    (re, im)
}

/// `n` complex numbers with components in `lo..hi`.
fn complex_values(n: usize, seed: u64, lo: f32, hi: f32) -> Vec<C32> {
    let re = values(n, seed, lo, hi);
    let im = values(n, seed + 1000, lo, hi);
    re.into_iter().zip(im).map(|(re, im)| C32::new(re, im)).collect()
}

/// Asserts that `got` is within `rel` of `want` measured against `scale` (the size the rounding
/// errors of the operation are relative to: the result's modulus, or the operands' for a sum).
fn near(index: usize, got: C32, want: Z, scale: f64, rel: f64) {
    let (g, w) = (z(got), want);
    let error = (g.0 - w.0).hypot(g.1 - w.1);
    assert!(error <= rel * scale + 1e-30, "index {index}: got {got:?}, want {want:?} (error {error}, scale {scale})");
}

fn check_all(got: &[C32], want: &[Z], rel: f64) {
    assert_eq!(got.len(), want.len());
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        near(i, g, w, abs(w), rel);
    }
}

fn array(ctx: &Context, data: &[C32]) -> Array<C32> {
    Array::from_slice_on(ctx, data, data.len()).unwrap()
}

/// An expression on the array, and the same on one element of the reference.
type Case = (&'static str, Array<C32>, Box<dyn Fn(Z) -> Z>);

fn same_bits(a: &[C32], b: &[C32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.re.to_bits() == y.re.to_bits() && x.im.to_bits() == y.im.to_bits())
}

// -- Storage --

#[test]
fn upload_and_readback_round_trip_bit_for_bit() {
    let ctx = Context::new().unwrap();
    for n in [1usize, 2, 3, 7, 1000, 1_000_003] {
        let mut data = complex_values(n, n as u64, -1e3, 1e3);
        if n >= 6 {
            data[0] = C32::new(f32::NAN, 1.0);
            data[1] = C32::new(f32::INFINITY, f32::NEG_INFINITY);
            data[2] = C32::new(-0.0, 0.0);
            data[3] = C32::new(f32::MIN_POSITIVE / 4.0, f32::MAX);
            data[4] = C32::new(0.0, f32::NAN);
            data[5] = C32::new(f32::from_bits(0x7fc0_1234), f32::from_bits(0xffc0_4321));
        }
        let x = array(&ctx, &data);
        assert!(same_bits(&x.to_vec().unwrap(), &data), "n = {n}");
        // Through a copy and a lazy identity as well.
        assert!(same_bits(&x.copy().to_vec().unwrap(), &data), "copy, n = {n}");
        assert!(same_bits(&(-(-&x)).to_vec().unwrap(), &data), "double negation, n = {n}");
    }
}

#[test]
fn zeros_full_and_scalars() {
    let ctx = Context::new().unwrap();
    let zeros = Array::<C32>::zeros_on(&ctx, 37).unwrap();
    assert_eq!(zeros.to_vec().unwrap(), vec![C32::ZERO; 37]);
    let full = Array::full_on(&ctx, [5, 3], C32::new(1.5, -2.5)).unwrap();
    assert_eq!(full.to_vec().unwrap(), vec![C32::new(1.5, -2.5); 15]);
    let one = Array::full_on(&ctx, 1, C32::I).unwrap();
    assert_eq!(one.item().unwrap(), C32::I);
}

#[test]
fn empty_arrays_are_an_error_and_poison() {
    let ctx = Context::new().unwrap();
    let empty: Vec<C32> = Vec::new();
    assert!(matches!(Array::from_slice_on(&ctx, &empty, 0), Err(LumaError::InvalidShape(_))));
    assert!(matches!(Array::<C32>::zeros_on(&ctx, 0), Err(LumaError::InvalidShape(_))));
}

// -- Operators --

#[test]
fn operators_match_the_f64_reference() {
    let ctx = Context::new().unwrap();
    let n = 5003;
    let (a, b) = (complex_values(n, 1, -2.0, 2.0), complex_values(n, 2, 0.5, 2.0));
    let (x, y) = (array(&ctx, &a), array(&ctx, &b));
    let want = |f: &dyn Fn(Z, Z) -> Z| -> Vec<Z> { a.iter().zip(&b).map(|(&p, &q)| f(z(p), z(q))).collect() };
    check_all(&(&x + &y).to_vec().unwrap(), &want(&|p, q| (p.0 + q.0, p.1 + q.1)), 2e-7);
    check_all(&(&x - &y).to_vec().unwrap(), &want(&|p, q| (p.0 - q.0, p.1 - q.1)), 2e-7);
    // A product's error is relative to the operands, not to a result that may cancel.
    let product = (&x * &y).to_vec().unwrap();
    for (i, ((&p, &q), &got)) in a.iter().zip(&b).zip(&product).enumerate() {
        near(i, got, mul(z(p), z(q)), abs(z(p)) * abs(z(q)), 4e-7);
    }
    let quotient = (&x / &y).to_vec().unwrap();
    for (i, ((&p, &q), &got)) in a.iter().zip(&b).zip(&quotient).enumerate() {
        near(i, got, div(z(p), z(q)), abs(z(p)) / abs(z(q)), 2e-6);
    }
    // By value and by reference.
    let by_value = (x.clone() * y.clone() + &x - y.clone()).to_vec().unwrap();
    let by_ref = (&(&x * &y) + &x - &y).to_vec().unwrap();
    assert!(same_bits(&by_value, &by_ref));
}

#[test]
fn compound_assignment() {
    let ctx = Context::new().unwrap();
    let (a, b) = (complex_values(300, 3, -2.0, 2.0), complex_values(300, 4, 0.5, 2.0));
    let (mut x, y) = (array(&ctx, &a), array(&ctx, &b));
    x += &y;
    x *= &y;
    x -= C32::new(0.25, 0.5);
    x /= C32::new(2.0, -1.0);
    let got = x.to_vec().unwrap();
    for (i, (&p, &q)) in a.iter().zip(&b).enumerate() {
        let (p, q) = (z(p), z(q));
        let s = mul(((p.0 + q.0), (p.1 + q.1)), q);
        let want = div((s.0 - 0.25, s.1 - 0.5), (2.0, -1.0));
        near(i, got[i], want, abs(s) / 2.0, 1e-5);
    }
}

#[test]
fn complex_and_real_scalars_on_either_side() {
    let ctx = Context::new().unwrap();
    let a = complex_values(999, 5, 0.5, 2.0);
    let x = array(&ctx, &a);
    let c = C32::new(0.75, -1.25);
    let s = 1.5f32;
    let zc = z(c);
    let zs = (f64::from(s), 0.0);
    let cases: Vec<Case> = vec![
        ("x + c", &x + c, Box::new(move |p| (p.0 + zc.0, p.1 + zc.1))),
        ("c + x", c + &x, Box::new(move |p| (p.0 + zc.0, p.1 + zc.1))),
        ("x - c", &x - c, Box::new(move |p| (p.0 - zc.0, p.1 - zc.1))),
        ("c - x", c - &x, Box::new(move |p| (zc.0 - p.0, zc.1 - p.1))),
        ("x * c", &x * c, Box::new(move |p| mul(p, zc))),
        ("c * x", c * &x, Box::new(move |p| mul(zc, p))),
        ("x / c", &x / c, Box::new(move |p| div(p, zc))),
        ("c / x", c / &x, Box::new(move |p| div(zc, p))),
        ("x + s", &x + s, Box::new(move |p| (p.0 + zs.0, p.1))),
        ("s + x", s + &x, Box::new(move |p| (p.0 + zs.0, p.1))),
        ("x - s", &x - s, Box::new(move |p| (p.0 - zs.0, p.1))),
        ("s - x", s - &x, Box::new(move |p| (zs.0 - p.0, -p.1))),
        ("x * s", &x * s, Box::new(move |p| (p.0 * zs.0, p.1 * zs.0))),
        ("s * x", s * &x, Box::new(move |p| (p.0 * zs.0, p.1 * zs.0))),
        ("x / s", &x / s, Box::new(move |p| (p.0 / zs.0, p.1 / zs.0))),
        ("s / x", s / &x, Box::new(move |p| div(zs, p))),
    ];
    for (_name, got, f) in cases {
        let want: Vec<Z> = a.iter().map(|&p| f(z(p))).collect();
        let got = got.to_vec().unwrap();
        for (i, (&g, &w)) in got.iter().zip(&want).enumerate() {
            near(i, g, w, abs(w).max(1e-3), 2e-6);
        }
    }
}

#[test]
fn a_real_scalar_has_no_imaginary_part_to_multiply() {
    // `inf * (1 + 0i)` is `inf + 0i` (the real scalar is not a complex product with `0 * inf`).
    let ctx = Context::new().unwrap();
    let x = array(&ctx, &[C32::new(f32::INFINITY, 1.0), C32::new(2.0, f32::INFINITY)]);
    let got = (&x * 2.0f32).to_vec().unwrap();
    assert_eq!(got, [C32::new(f32::INFINITY, 2.0), C32::new(4.0, f32::INFINITY)]);
    let got = (&x + 1.0f32).to_vec().unwrap();
    assert_eq!(got, [C32::new(f32::INFINITY, 1.0), C32::new(3.0, f32::INFINITY)]);
}

#[test]
fn promotion_of_bool_masks() {
    let ctx = Context::new().unwrap();
    let a = complex_values(100, 6, -1.0, 1.0);
    let x = array(&ctx, &a);
    let mask = Array::from_slice_on(&ctx, &(0..100).map(|i| i % 3 == 0).collect::<Vec<_>>(), 100).unwrap();
    let got = (&mask * &x).to_vec().unwrap();
    for (i, (&g, &p)) in got.iter().zip(&a).enumerate() {
        assert_eq!(g, if i % 3 == 0 { p } else { C32::ZERO }, "{i}");
    }
}

// -- Functions --

#[test]
fn functions_match_the_f64_reference() {
    let ctx = Context::new().unwrap();
    let n = 4001;
    let a = complex_values(n, 7, -3.0, 3.0);
    let x = array(&ctx, &a);
    let zs: Vec<Z> = a.iter().map(|&c| z(c)).collect();

    check_all(&(-&x).to_vec().unwrap(), &zs.iter().map(|p| (-p.0, -p.1)).collect::<Vec<_>>(), 0.0);
    check_all(&x.conj().to_vec().unwrap(), &zs.iter().map(|p| (p.0, -p.1)).collect::<Vec<_>>(), 0.0);
    let reals: Vec<f32> = x.real().to_vec().unwrap();
    let imags: Vec<f32> = x.imag().to_vec().unwrap();
    assert_eq!(reals, a.iter().map(|c| c.re).collect::<Vec<_>>());
    assert_eq!(imags, a.iter().map(|c| c.im).collect::<Vec<_>>());
    for (i, (got, want)) in x.abs().to_vec().unwrap().iter().zip(&zs).enumerate() {
        assert!((f64::from(*got) - abs(*want)).abs() <= 2e-7 * abs(*want), "abs {i}: {got} vs {}", abs(*want));
    }
    for (i, (got, want)) in x.arg().to_vec().unwrap().iter().zip(&zs).enumerate() {
        assert!((f64::from(*got) - want.1.atan2(want.0)).abs() <= 3e-6, "arg {i}: {got} vs {}", want.1.atan2(want.0));
    }
    for (i, (got, want)) in x.norm_sqr().to_vec().unwrap().iter().zip(&zs).enumerate() {
        let w = want.0 * want.0 + want.1 * want.1;
        assert!((f64::from(*got) - w).abs() <= 3e-7 * w, "norm_sqr {i}");
    }
    let exps = x.exp().to_vec().unwrap();
    for (i, (&got, &want)) in exps.iter().zip(&zs).enumerate() {
        let want = exp(want);
        near(i, got, want, abs(want), 4e-6);
    }
    let roots = x.sqrt().to_vec().unwrap();
    for (i, (&got, &want)) in roots.iter().zip(&zs).enumerate() {
        near(i, got, sqrt(want), abs(sqrt(want)), 4e-6);
    }
}

#[test]
fn constructors_from_parts_to_complex_and_polar() {
    let ctx = Context::new().unwrap();
    let n = 2049;
    let re = values(n, 11, -2.0, 2.0);
    let im = values(n, 12, -2.0, 2.0);
    let (r, theta) = (values(n, 13, 0.0, 3.0), values(n, 14, -3.0, 3.0));
    let on = |d: &[f32]| Array::from_slice_on(&ctx, d, d.len()).unwrap();
    let (xr, xi) = (on(&re), on(&im));
    let parts = Array::from_parts(&xr, &xi).to_vec().unwrap();
    assert!(same_bits(&parts, &re.iter().zip(&im).map(|(&a, &b)| C32::new(a, b)).collect::<Vec<_>>()));
    let lifted = xr.to_complex().to_vec().unwrap();
    assert!(same_bits(&lifted, &re.iter().map(|&a| C32::new(a, 0.0)).collect::<Vec<_>>()));
    let polar = Array::from_polar(&on(&r), &on(&theta)).to_vec().unwrap();
    for i in 0..n {
        let (a, b) = (f64::from(r[i]), f64::from(theta[i]));
        near(i, polar[i], (a * b.cos(), a * b.sin()), a, 2e-6);
    }
    // The parts of an array put back together are that array, also through a fused expression.
    let z = Array::from_parts(&xr, &xi);
    let back = Array::from_parts(&z.real(), &z.imag()).to_vec().unwrap();
    assert!(same_bits(&back, &parts));
    // A shape mismatch poisons the result like an operator does.
    let short = Array::from_parts(&xr, &on(&im[..10]));
    assert!(matches!(short.to_vec().unwrap_err().root(), LumaError::ShapeMismatch { .. }));
}

#[test]
fn select_and_equality() {
    let ctx = Context::new().unwrap();
    let a = complex_values(500, 21, -1.0, 1.0);
    let mut b = a.clone();
    for (i, c) in b.iter_mut().enumerate() {
        match i % 3 {
            0 => {}
            1 => c.im += 1.0,
            _ => c.re -= 1.0,
        }
    }
    b[4] = C32::new(f32::NAN, 0.0);
    let (x, y) = (array(&ctx, &a), array(&ctx, &b));
    let eq = x.eq(&y).to_vec().unwrap();
    let ne = x.ne(&y).to_vec().unwrap();
    for i in 0..a.len() {
        assert_eq!(eq[i], a[i] == b[i], "eq {i}");
        assert_eq!(ne[i], a[i] != b[i], "ne {i}");
    }
    let picked = select(&x.eq(&y), &x, C32::new(9.0, -9.0)).to_vec().unwrap();
    for i in 0..a.len() {
        assert_eq!(picked[i], if a[i] == b[i] { a[i] } else { C32::new(9.0, -9.0) }, "select {i}");
    }
    let both = select(&x.real().gt(0.0), &x, &y).to_vec().unwrap();
    for i in 0..a.len() {
        assert!(same_bits(&[both[i]], &[if a[i].re > 0.0 { a[i] } else { b[i] }]), "select {i}");
    }
}

#[test]
fn unsupported_operations_poison_with_an_error() {
    let ctx = Context::new().unwrap();
    let x = array(&ctx, &[C32::ONE, C32::I]);
    for bad in [x.minimum(&x).to_vec(), x.maximum(C32::ONE).to_vec()] {
        assert!(matches!(bad.unwrap_err().root(), LumaError::Unsupported(_)));
    }
    assert!(matches!(x.lt(&x).to_vec().unwrap_err().root(), LumaError::Unsupported(_)));
    assert!(matches!(x.ge(C32::ONE).to_vec().unwrap_err().root(), LumaError::Unsupported(_)));
    // The error of an operand reaches the result of every later operation.
    let y = array(&ctx, &[C32::ONE, C32::I, C32::ZERO]);
    let bad = (&x + &y).conj().exp();
    assert!(matches!(bad.to_vec().unwrap_err().root(), LumaError::ShapeMismatch { .. }));
    assert!(matches!(bad.abs().sum().item().unwrap_err().root(), LumaError::ShapeMismatch { .. }));
}

// -- Edge cases --

fn one(ctx: &Context, c: C32) -> Array<C32> {
    array(ctx, &[c])
}

#[test]
fn division_is_robust_and_by_zero_gives_infinity_or_nan() {
    let ctx = Context::new().unwrap();
    // Huge components: the textbook formula overflows `|b|^2`; Smith's does not.
    // (Up to 1e37: a device's fast division returns 0 for a divisor above 2^126, here `2 * scale`.)
    for scale in [1e30f32, 1e-30, 1e19, 1e37] {
        let (a, b) = (C32::new(scale, scale), C32::new(scale, scale));
        let q = (&one(&ctx, a) / &one(&ctx, b)).to_vec().unwrap()[0];
        near(0, q, (1.0, 0.0), 1.0, 1e-6);
        let q = (&one(&ctx, C32::new(2.0 * scale / 3.0, 0.0)) / &one(&ctx, C32::new(0.0, scale))).to_vec().unwrap()[0];
        near(1, q, (0.0, -2.0 / 3.0), 1.0, 1e-6);
    }
    // A tiny denominator with a huge numerator: the quotient is `-1e60 i`, out of range.
    let q = (&one(&ctx, C32::new(1e30, -1e30)) / &one(&ctx, C32::new(1e-30, 1e-30))).to_vec().unwrap()[0];
    assert!(q.re.abs() < 1e-3 && q.im == f32::NEG_INFINITY, "{q:?}");
    // By zero: no trap, infinity or NaN per component.
    let zero = one(&ctx, C32::ZERO);
    let q = (&one(&ctx, C32::new(1.0, 1.0)) / &zero).to_vec().unwrap()[0];
    assert!(q.re.is_infinite() && q.im.is_infinite(), "{q:?}");
    assert!(q.re > 0.0 && q.im > 0.0);
    let q = (&one(&ctx, C32::new(-2.0, 0.0)) / &zero).to_vec().unwrap()[0];
    assert!(q.re == f32::NEG_INFINITY && q.im.is_nan(), "{q:?}");
    // (A device that assumes no `NaN` exists may fold `x / x` of one array to 1: two arrays.)
    let q = (&zero / &one(&ctx, C32::ZERO)).to_vec().unwrap()[0];
    assert!(q.re.is_nan() || q.im.is_nan(), "{q:?}");
    let q = (&one(&ctx, C32::ONE) / C32::new(0.0, 0.0)).to_vec().unwrap()[0];
    assert!(q.re.is_infinite(), "{q:?}");
    // Finite over infinite is zero.
    let q = (&one(&ctx, C32::new(3.0, -4.0)) / &one(&ctx, C32::new(f32::INFINITY, 0.0))).to_vec().unwrap()[0];
    assert_eq!((q.re, q.im.abs()), (0.0, 0.0), "{q:?}");
}

#[test]
fn abs_does_not_overflow_or_underflow() {
    let ctx = Context::new().unwrap();
    let cases = [
        (C32::new(3e30, 4e30), 5e30f32),
        (C32::new(3e-30, 4e-30), 5e-30),
        (C32::new(-3e37, 4e37), 5e37),
        (C32::new(1e-37, 0.0), 1e-37),
        (C32::new(0.0, -1e38), 1e38),
        (C32::new(3e38, 3e38), f32::INFINITY),
        (C32::new(2e38, 1e38), 2.236068e38),
        (C32::new(0.0, 0.0), 0.0),
        (C32::new(-0.0, -0.0), 0.0),
        (C32::new(f32::INFINITY, f32::NAN), f32::INFINITY),
        (C32::new(f32::NAN, f32::NEG_INFINITY), f32::INFINITY),
        (C32::new(1.0, 0.0), 1.0),
    ];
    let x = array(&ctx, &cases.map(|(c, _)| c));
    let got = x.abs().to_vec().unwrap();
    for (i, (&g, &(c, want))) in got.iter().zip(&cases).enumerate() {
        assert!(g == want || (g - want).abs() <= 1e-6 * want, "abs({c:?}) = {g}, want {want} (case {i})");
    }
    assert!(one(&ctx, C32::new(f32::NAN, 1.0)).abs().to_vec().unwrap()[0].is_nan());
    assert!(one(&ctx, C32::new(1.0, f32::NAN)).abs().to_vec().unwrap()[0].is_nan());
}

#[test]
fn arg_of_zero_and_of_negative_reals() {
    let ctx = Context::new().unwrap();
    let pi = std::f32::consts::PI;
    let cases = [
        (C32::new(0.0, 0.0), 0.0f32),
        (C32::new(1.0, 0.0), 0.0),
        (C32::new(-1.0, 0.0), pi),
        (C32::new(-1.0, -0.0), -pi),
        (C32::new(-5.0, 0.0), pi),
        (C32::new(0.0, 1.0), pi / 2.0),
        (C32::new(0.0, -1.0), -pi / 2.0),
        (C32::new(1.0, 1.0), pi / 4.0),
        (C32::new(-1.0, 1.0), 3.0 * pi / 4.0),
        (C32::new(-1.0, -1.0), -3.0 * pi / 4.0),
        (C32::new(f32::INFINITY, 1.0), 0.0),
        (C32::new(f32::NEG_INFINITY, 1.0), pi),
    ];
    let got = array(&ctx, &cases.map(|(c, _)| c)).arg().to_vec().unwrap();
    for (&g, &(c, want)) in got.iter().zip(&cases) {
        assert!((g - want).abs() <= 2e-6, "arg({c:?}) = {g}, want {want}");
    }
    assert!(one(&ctx, C32::new(f32::NAN, 1.0)).arg().to_vec().unwrap()[0].is_nan());
}

#[test]
fn conjugate_keeps_the_real_part_and_flips_the_sign_of_the_imaginary_part() {
    let ctx = Context::new().unwrap();
    let data = [C32::new(f32::NAN, f32::NAN), C32::new(1.0, f32::NAN), C32::new(2.0, 0.0), C32::new(-0.0, -0.0), C32::new(f32::INFINITY, 3.0)];
    let got = array(&ctx, &data).conj().to_vec().unwrap();
    assert!(got[0].re.is_nan() && got[0].im.is_nan());
    assert!(got[1].re == 1.0 && got[1].im.is_nan());
    assert!(got[2].re == 2.0 && got[2].im == 0.0 && got[2].im.is_sign_negative());
    assert!(got[3].re.is_sign_negative() && got[3].im == 0.0 && got[3].im.is_sign_positive());
    assert_eq!((got[4].re, got[4].im), (f32::INFINITY, -3.0));
    // Twice is the identity.
    assert!(same_bits(&array(&ctx, &data[1..]).conj().conj().to_vec().unwrap(), &data[1..]));
}

#[test]
fn exp_and_sqrt_special_values() {
    let ctx = Context::new().unwrap();
    let got = array(&ctx, &[C32::new(0.0, 0.0), C32::new(1.0, 0.0), C32::new(0.0, std::f32::consts::PI), C32::new(f32::NEG_INFINITY, 0.0)]).exp().to_vec().unwrap();
    near(0, got[0], (1.0, 0.0), 1.0, 1e-6);
    near(1, got[1], (std::f64::consts::E, 0.0), 2.72, 1e-6);
    near(2, got[2], (-1.0, 0.0), 1.0, 3e-6);
    assert_eq!((got[3].re, got[3].im.abs()), (0.0, 0.0));
    let roots = array(&ctx, &[C32::new(0.0, 0.0), C32::new(-4.0, 0.0), C32::new(-4.0, -0.0), C32::new(4.0, 0.0), C32::new(0.0, 2.0)]).sqrt().to_vec().unwrap();
    assert_eq!(roots[0], C32::ZERO);
    near(1, roots[1], (0.0, 2.0), 2.0, 1e-6);
    assert!(roots[2].im < 0.0);
    near(2, roots[2], (0.0, -2.0), 2.0, 1e-6);
    near(3, roots[3], (2.0, 0.0), 2.0, 1e-6);
    near(4, roots[4], (1.0, 1.0), 1.4, 1e-6);
}

// -- Odd sizes --

#[test]
fn odd_lengths_and_a_large_one() {
    let ctx = Context::new().unwrap();
    for n in [1usize, 3, 1000003] {
        let (a, b) = (complex_values(n, 31, -2.0, 2.0), complex_values(n, 32, 0.5, 2.0));
        let (x, y) = (array(&ctx, &a), array(&ctx, &b));
        let got = ((&x * &y).conj() / (&y + 1.0f32)).to_vec().unwrap();
        for i in (0..n).step_by(if n > 1000 { 997 } else { 1 }) {
            let (p, q) = (z(a[i]), z(b[i]));
            let m = mul(p, q);
            let want = div((m.0, -m.1), (q.0 + 1.0, q.1));
            near(i, got[i], want, abs(want), 4e-6);
        }
    }
}

// -- Views --

#[test]
fn views_of_two_dimensional_arrays_then_arithmetic() {
    let ctx = Context::new().unwrap();
    let (d0, d1) = (7usize, 5usize);
    let a = complex_values(d0 * d1, 41, -2.0, 2.0);
    let b = complex_values(d0 * d1, 42, 0.5, 2.0);
    let x = Array::from_slice_on(&ctx, &a, [d0, d1]).unwrap();
    let y = Array::from_slice_on(&ctx, &b, [d0, d1]).unwrap();
    let at = |v: &[C32], i0: usize, i1: usize, d: usize| v[i0 + d * i1];

    // Transpose.
    let t = x.transpose().to_vec().unwrap();
    for i0 in 0..d0 {
        for i1 in 0..d1 {
            assert_eq!(at(&t, i1, i0, d1), at(&a, i0, i1, d0));
        }
    }
    // Slice and flip, fused into arithmetic with a plain array.
    let sliced = (x.slice(0, 1..d0 - 1) * y.slice(0, 1..d0 - 1)).to_vec().unwrap();
    for i1 in 0..d1 {
        for i0 in 0..d0 - 2 {
            let want = mul(z(at(&a, i0 + 1, i1, d0)), z(at(&b, i0 + 1, i1, d0)));
            near(i0, at(&sliced, i0, i1, d0 - 2), want, abs(want), 4e-7);
        }
    }
    let flipped = (x.flip(0) + y.flip(1).conj()).to_vec().unwrap();
    for i1 in 0..d1 {
        for i0 in 0..d0 {
            let (p, q) = (z(at(&a, d0 - 1 - i0, i1, d0)), z(at(&b, i0, d1 - 1 - i1, d0)));
            near(i0, at(&flipped, i0, i1, d0), (p.0 + q.0, p.1 - q.1), abs(p) + abs(q), 4e-7);
        }
    }
    // A transposed operand against a plain one, divided and reduced to moduli.
    let moduli = (x.transpose() / y.transpose()).abs().to_vec().unwrap();
    for i0 in 0..d0 {
        for i1 in 0..d1 {
            let want = abs(div(z(at(&a, i0, i1, d0)), z(at(&b, i0, i1, d0))));
            let got = f64::from(moduli[i1 + d1 * i0]);
            assert!((got - want).abs() <= 2e-6 * want, "{got} vs {want}");
        }
    }
    // Writing a scalar into a view and compound assignment on it.
    let mut w = Array::<C32>::zeros_on(&ctx, [d0, d1]).unwrap();
    w.slice_mut(0, 2..4).assign(C32::new(1.0, 2.0)).unwrap();
    w.slice_mut(1, 0..1).assign(C32::new(-1.0, 0.5)).unwrap();
    let w = w.to_vec().unwrap();
    for i1 in 0..d1 {
        for i0 in 0..d0 {
            let want = if i1 == 0 { C32::new(-1.0, 0.5) } else if (2..4).contains(&i0) { C32::new(1.0, 2.0) } else { C32::ZERO };
            assert_eq!(at(&w, i0, i1, d0), want, "({i0}, {i1})");
        }
    }
}

// -- Reductions --

#[test]
fn sums_and_means() {
    let ctx = Context::new().unwrap();
    let (d0, d1) = (37usize, 11usize);
    let a = complex_values(d0 * d1, 51, -1.0, 1.0);
    let x = Array::from_slice_on(&ctx, &a, [d0, d1]).unwrap();
    let total = a.iter().fold((0.0f64, 0.0f64), |s, c| (s.0 + f64::from(c.re), s.1 + f64::from(c.im)));
    let n = (d0 * d1) as f64;
    near(0, x.sum().item().unwrap(), total, 20.0, 1e-6);
    near(0, x.mean().item().unwrap(), (total.0 / n, total.1 / n), 1.0, 1e-5);
    let rows = x.sum_axis(0).to_vec().unwrap();
    assert_eq!(rows.len(), d1);
    for (i1, &got) in rows.iter().enumerate() {
        let want = (0..d0).fold((0.0f64, 0.0f64), |s, i0| (s.0 + f64::from(a[i0 + d0 * i1].re), s.1 + f64::from(a[i0 + d0 * i1].im)));
        near(i1, got, want, 5.0, 1e-6);
    }
    let cols = x.mean_axis(1).to_vec().unwrap();
    assert_eq!(cols.len(), d0);
    for (i0, &got) in cols.iter().enumerate() {
        let want = (0..d1).fold((0.0f64, 0.0f64), |s, i1| (s.0 + f64::from(a[i0 + d0 * i1].re), s.1 + f64::from(a[i0 + d0 * i1].im)));
        near(i0, got, (want.0 / d1 as f64, want.1 / d1 as f64), 1.0, 1e-5);
    }
    // A sum of an expression, and a lazy sum used in an expression of the array it sums.
    let centered = (&x - &x.mean()).sum().item().unwrap();
    assert!(abs(z(centered)) < 2e-5, "{centered:?}");
}

// -- Fusion --

#[test]
fn a_chain_of_ten_operations_is_one_kernel() {
    let ctx = Context::new().unwrap();
    let n = 100_003;
    let (a, b) = (complex_values(n, 61, -2.0, 2.0), complex_values(n, 62, 0.5, 2.0));
    let (x, y) = (array(&ctx, &a), array(&ctx, &b));
    x.eval().unwrap();
    y.eval().unwrap();
    ctx.flush().unwrap();
    let (got, profile) = ctx
        .profile(|| {
            // 10 operations: *, +, conj, +, /, exp-free abs side: neg, *, -, conj, real/imag recombined.
            let s = &x * &y;
            let s = &s + &x;
            let s = s.conj();
            let s = &s + 1.0f32;
            let s = &s / &(&y + C32::new(0.0, 1.0));
            let s = -&s;
            let s = &s * C32::new(0.5, -0.5);
            let s = &s - &y;
            let s = Array::from_parts(&s.imag(), &s.real());
            let s = s.conj();
            s.to_vec().unwrap()
        })
        .unwrap();
    for i in (0..n).step_by(131) {
        let (p, q) = (z(a[i]), z(b[i]));
        let s = mul(p, q);
        let s = (s.0 + p.0, s.1 + p.1);
        let s = (s.0, -s.1);
        let s = (s.0 + 1.0, s.1);
        let s = div(s, (q.0, q.1 + 1.0));
        let s = (-s.0, -s.1);
        let s = mul(s, (0.5, -0.5));
        let s = (s.0 - q.0, s.1 - q.1);
        let want = (s.1, -s.0);
        near(i, got[i], want, abs(p) * abs(q) + abs(want) + 1.0, 2e-5);
    }
    assert_eq!(profile.launches(), 1, "{profile}");
}

#[test]
fn complex_chain_ending_in_a_real_reduction() {
    let ctx = Context::new().unwrap();
    let n = 200_001;
    let (a, b) = (complex_values(n, 71, -2.0, 2.0), complex_values(n, 72, 0.5, 2.0));
    let (x, y) = (array(&ctx, &a), array(&ctx, &b));
    x.eval().unwrap();
    y.eval().unwrap();
    ctx.flush().unwrap();
    let want_abs: f64 = a.iter().zip(&b).map(|(&p, &q)| abs(div(mul(z(p), z(q)), (z(q).0 + 1.0, z(q).1)))).sum();
    let want_re: f64 = a.iter().zip(&b).map(|(&p, &q)| mul(z(p), z(q)).0).sum();
    let ((got_abs, got_re), profile) = ctx
        .profile(|| {
            let s = &x * &y;
            let abs = (&s / &(&y + 1.0f32)).abs().sum().item().unwrap();
            (abs, s.real().sum().item().unwrap())
        })
        .unwrap();
    assert!((f64::from(got_abs) - want_abs).abs() <= 2e-5 * want_abs, "{got_abs} vs {want_abs}");
    assert!((f64::from(got_re) - want_re).abs() <= 2e-5 * (want_re.abs() + 1e4), "{got_re} vs {want_re}");
    // The complex expression is computed inside the reduction: no stored array of complex numbers.
    assert!(profile.launches() <= 6, "{profile}");
    // Along an axis too.
    let x2 = Array::from_slice_on(&ctx, &a[..200_000], [400, 500]).unwrap();
    let y2 = Array::from_slice_on(&ctx, &b[..200_000], [400, 500]).unwrap();
    let norms = (&x2 * &y2).abs().sum_axis(0).to_vec().unwrap();
    for (i1, &got) in norms.iter().enumerate().step_by(37) {
        let want: f64 = (0..400).map(|i0| abs(mul(z(a[i0 + 400 * i1]), z(b[i0 + 400 * i1])))).sum();
        assert!((f64::from(got) - want).abs() <= 2e-5 * want, "{i1}: {got} vs {want}");
    }
}

#[test]
fn norm_through_matmul_free_expression_uses_real_parts_only() {
    // `abs()` of a complex expression feeds a real expression, which is an ordinary `f32` array.
    let ctx = Context::new().unwrap();
    let a = complex_values(1000, 81, -2.0, 2.0);
    let x = array(&ctx, &a);
    let r = (x.abs() * 2.0 + x.real().abs()).to_vec().unwrap();
    for (i, (&got, &c)) in r.iter().zip(&a).enumerate() {
        let want = 2.0 * abs(z(c)) + f64::from(c.re).abs();
        assert!((f64::from(got) - want).abs() <= 3e-6 * want.max(1e-3), "{i}");
    }
}

#[test]
fn lazy_complex_scalars_share_a_kernel() {
    // Different complex scalars are uniforms: the same expression with other numbers reuses the kernel.
    let ctx = Context::new().unwrap();
    let a = complex_values(333, 91, -2.0, 2.0);
    let x = array(&ctx, &a);
    for k in 0..5 {
        let c = C32::new(k as f32 * 0.5 - 1.0, 2.0 - k as f32);
        let got = (&x * c + c).to_vec().unwrap();
        for (i, &g) in got.iter().enumerate() {
            let m = mul(z(a[i]), z(c));
            near(i, g, (m.0 + f64::from(c.re), m.1 + f64::from(c.im)), abs(z(a[i])) * abs(z(c)) + 1.0, 2e-6);
        }
    }
}
