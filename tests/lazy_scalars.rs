//! Reductions are lazy one-element arrays: `item` / `to_array` read them, and they broadcast.

mod common;

use common::{assert_close, values};
use luma::{Array, Context, LumaError, Reduction, Scalar};

fn ctx() -> Context {
    Context::new().unwrap()
}

fn f32s(ctx: &Context, data: &[f32]) -> Array<f32> {
    Array::from_slice_on(ctx, data, data.len()).unwrap()
}

#[test]
fn item_matches_the_host_values() {
    let ctx = ctx();
    let xs = values(1000, 1, -1.0, 1.0);
    let x = f32s(&ctx, &xs);
    let want: f64 = xs.iter().map(|&v| f64::from(v)).sum();
    let got = f64::from(x.sum().item().unwrap());
    assert!((got - want).abs() < 1e-3, "{got} vs {want}");
    assert_eq!(x.min().item().unwrap(), xs.iter().copied().fold(f32::INFINITY, f32::min));
    assert_eq!(x.max().item().unwrap(), xs.iter().copied().fold(f32::NEG_INFINITY, f32::max));

    let [lo, hi] = x.reduce([Reduction::Min, Reduction::Max]).to_array().unwrap();
    assert_eq!(lo, x.min().item().unwrap());
    assert_eq!(hi, x.max().item().unwrap());

    let want_norm = xs.iter().map(|&v| f64::from(v) * f64::from(v)).sum::<f64>().sqrt();
    let norm = f64::from(x.norm().item().unwrap());
    assert!((norm - want_norm).abs() < 1e-4 * want_norm, "{norm} vs {want_norm}");

    // Lazy operands and materialized ones agree.
    let lazy = &x * 2.0;
    let got = f64::from(lazy.sum().item().unwrap());
    assert!((got - 2.0 * want).abs() < 2e-3);
    lazy.eval().unwrap();
    let again = f64::from(lazy.sum().item().unwrap());
    assert!((again - got).abs() < 1e-3);
}

#[test]
fn a_reduction_is_an_array_of_one_element() {
    let ctx = ctx();
    let x = f32s(&ctx, &values(64, 2, 0.0, 1.0));
    let s = (&x * 2.0).sum();
    assert_eq!(s.len(), 1);
    assert_eq!(s.to_vec().unwrap().len(), 1);
    assert_eq!(x.reduce([Reduction::Sum, Reduction::Max, Reduction::Min]).len(), 3);
    // Reading it twice, and using it afterwards, gives the same value.
    let first = s.item().unwrap();
    assert_eq!(first, s.item().unwrap());
    assert_eq!((&s * 1.0).item().unwrap(), first);
}

#[test]
fn broadcasting_a_reduction_stays_on_the_gpu() {
    let ctx = ctx();
    let xs = values(1000, 3, 0.5, 1.5);
    let x = f32s(&ctx, &xs);
    let sum: f32 = xs.iter().sum();
    let got = (&x / &x.sum()).to_vec().unwrap();
    let want: Vec<f32> = xs.iter().map(|v| v / sum).collect();
    assert_close(&got, &want, 1e-4, 1e-7);

    // The one-element array on the left takes the shape of the other operand.
    let m = &x.max() * &x;
    assert_eq!(m.shape(), x.shape());
    let max = xs.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let want: Vec<f32> = xs.iter().map(|v| max * v).collect();
    assert_close(&m.to_vec().unwrap(), &want, 1e-6, 1e-7);

    // A lazy expression works the same as a reduction, also in 2-D and in place.
    let m = Array::from_slice_on(&ctx, &xs[..12], [3, 4]).unwrap();
    let one = Array::from_slice_on(&ctx, &[2.0f32], 1).unwrap() * 1.5;
    let got = (&m - &one).to_vec().unwrap();
    let want: Vec<f32> = xs[..12].iter().map(|v| v - 3.0).collect();
    assert_close(&got, &want, 1e-6, 1e-6);
    let mut acc = m.reshape([3, 4]);
    acc += &one;
    let want: Vec<f32> = xs[..12].iter().map(|v| v + 3.0).collect();
    assert_close(&acc.to_vec().unwrap(), &want, 1e-6, 1e-6);

    // Mismatched shapes (neither a single element) still fail.
    let a = f32s(&ctx, &[1.0, 2.0, 3.0]);
    let b = f32s(&ctx, &[1.0, 2.0]);
    assert!(matches!((&a + &b).to_vec().unwrap_err().root(), LumaError::ShapeMismatch { .. }));
}

#[test]
fn select_takes_a_single_element_branch() {
    let ctx = ctx();
    let xs = values(100, 4, -1.0, 1.0);
    let x = f32s(&ctx, &xs);
    let clipped = luma::select(&x.gt(0.0), &x, x.min());
    let min = xs.iter().copied().fold(f32::INFINITY, f32::min);
    let want: Vec<f32> = xs.iter().map(|&v| if v > 0.0 { v } else { min }).collect();
    assert_eq!(clipped.to_vec().unwrap(), want);
}

#[test]
fn sum_where_adds_the_masked_elements() {
    let ctx = ctx();
    let xs = values(1000, 5, -1.0, 1.0);
    let x = f32s(&ctx, &xs);
    let got = f64::from(x.sum_where(&x.gt(0.0)).item().unwrap());
    let want: f64 = xs.iter().filter(|&&v| v > 0.0).map(|&v| f64::from(v)).sum();
    assert!((got - want).abs() < 1e-3, "{got} vs {want}");
    let counted = Array::from_slice_on(&ctx, &[1u32; 10], 10).unwrap();
    let mask = Array::from_slice_on(&ctx, &[1u32, 0, 1, 0, 1, 0, 1, 0, 1, 0], 10).unwrap();
    assert_eq!(counted.sum_where(&mask).item().unwrap(), 5);
}

#[test]
fn item_and_to_array_check_the_length() {
    let ctx = ctx();
    let x = f32s(&ctx, &[1.0, 2.0, 3.0]);
    assert!(matches!(x.item().unwrap_err().root(), LumaError::InvalidShape(_)));
    assert!(matches!(x.to_array::<2>().unwrap_err().root(), LumaError::LengthMismatch { expected: 2, actual: 3 }));
    assert_eq!(x.to_array::<3>().unwrap(), [1.0, 2.0, 3.0]);
    assert_eq!(x.sum().item().unwrap(), 6.0);
}

#[test]
fn poisoned_operands_poison_the_reduction() {
    let ctx = ctx();
    let a = f32s(&ctx, &[1.0, 2.0, 3.0]);
    let b = f32s(&ctx, &[1.0, 2.0]);
    let bad = &a + &b;
    for r in [bad.sum(), bad.max(), bad.mean(), bad.norm()] {
        assert!(matches!(r.item().unwrap_err().root(), LumaError::ShapeMismatch { .. }));
    }
}

#[test]
fn scalar_changes_between_operations() {
    let ctx = ctx();
    let xs = values(100, 6, 0.0, 1.0);
    let x = f32s(&ctx, &xs);
    let s = Scalar::new(2.0f32);
    let a = &x * &s;
    s.set(3.0);
    let b = &x * &s;
    assert_eq!(s.get(), 3.0);
    let double: Vec<f32> = xs.iter().map(|v| v * 2.0).collect();
    let triple: Vec<f32> = xs.iter().map(|v| v * 3.0).collect();
    assert_eq!(a.to_vec().unwrap(), double);
    assert_eq!(b.to_vec().unwrap(), triple);

    let c = &s * &x;
    let d = &x + &s;
    let mut e = x.reshape(100) * 1.0;
    e -= &s;
    assert_eq!(c.to_vec().unwrap(), triple);
    let want: Vec<f32> = xs.iter().map(|v| v + 3.0).collect();
    assert_eq!(d.to_vec().unwrap(), want);
    let want: Vec<f32> = xs.iter().map(|v| v - 3.0).collect();
    assert_eq!(e.to_vec().unwrap(), want);

    let clone = s.clone();
    clone.set(0.5);
    assert_eq!(s.get(), 0.5);
    assert_eq!(x.minimum(&s).min().item().unwrap(), xs.iter().copied().fold(f32::INFINITY, f32::min).min(0.5));
}
