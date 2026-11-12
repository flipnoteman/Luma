//! The finite check: with `ContextOptions::check_finite` an operation that produces a NaN or an
//! infinity fails at the line that wrote it.

mod common;

use common::values;
use luma::{Array, Context, ContextOptions, LumaError};

fn checked() -> Context {
    Context::with_options(ContextOptions::default().check_finite(true)).unwrap()
}

#[test]
fn a_division_by_zero_is_reported_at_its_line() {
    let ctx = checked();
    let x = Array::from_slice_on(&ctx, &values(32, 1, 1.0, 2.0), 32).unwrap();

    let line = line!() + 1;
    let y = &x / 0.0;
    let err = y.to_vec().unwrap_err();
    assert!(matches!(err.root(), LumaError::NonFinite), "{err}");
    let location = err.location().expect("the error names the operation");
    assert!(location.file().ends_with("check_finite.rs"), "{location}");
    assert_eq!(location.line(), line);
    assert!(err.to_string().contains("NaN or Inf"), "{err}");

    // A NaN is non-finite, too (0 / 0), and a finite result passes.
    let zero = Array::<f32>::zeros_on(&ctx, 8).unwrap();
    let err = (&zero / &zero).to_vec().unwrap_err();
    assert!(matches!(err.root(), LumaError::NonFinite), "{err}");
    let fine = (&x * 2.0 + 1.0).to_vec().unwrap();
    assert_eq!(fine.len(), 32);
}

#[test]
fn results_computed_together_are_checked_one_by_one() {
    let ctx = checked();
    let x = Array::from_slice_on(&ctx, &values(16, 2, 1.0, 2.0), 16).unwrap();
    let (good, bad) = (&x + 1.0, &x / 0.0);
    luma::eval_all(&[&good, &bad]).ok();
    assert_eq!(good.to_vec().unwrap().len(), 16);
    assert!(matches!(bad.to_vec().unwrap_err().root(), LumaError::NonFinite));
}

#[test]
fn writes_into_arrays_are_checked() {
    let ctx = checked();
    let w = Array::<f32>::zeros_on(&ctx, [4, 2]).unwrap();
    let bad = Array::from_slice_on(&ctx, &[1.0, f32::INFINITY, 1.0, 1.0], [4, 1]).unwrap();
    let err = w.slice(1, 0..1).assign(&bad).unwrap_err();
    assert!(matches!(err.root(), LumaError::NonFinite), "{err}");
    assert_eq!(err.location().map(|l| l.file().ends_with("check_finite.rs")), Some(true));
}

#[test]
fn a_default_context_computes_infinities() {
    let ctx = Context::with_options(ContextOptions::default()).unwrap();
    let x = Array::from_slice_on(&ctx, &values(8, 3, 1.0, 2.0), 8).unwrap();
    let y = (&x / 0.0).to_vec().unwrap();
    assert!(y.iter().all(|v| v.is_infinite()));
}
