//! `Array: Clone` is a handle on the same data, like `reshape`: writes through one clone are seen
//! by the others, and a lazy array is still one expression.

mod common;

use common::{assert_close, values};
use luma::{Array, Context, Fields, LumaError};

fn f32s(ctx: &Context, data: &[f32]) -> Array<f32> {
    Array::from_slice_on(ctx, data, data.len()).unwrap()
}

#[test]
fn clones_of_an_evaluated_array_see_each_others_writes() {
    let ctx = Context::new().unwrap();
    let mut a = f32s(&ctx, &[1.0, 2.0, 3.0, 4.0]);
    let mut b = a.clone();

    a.write(&[5.0, 6.0, 7.0, 8.0]).unwrap();
    assert_eq!(b.to_vec().unwrap(), [5.0, 6.0, 7.0, 8.0]);

    a += 1.0;
    assert_eq!(b.to_vec().unwrap(), [6.0, 7.0, 8.0, 9.0]);
    assert_eq!(a.to_vec().unwrap(), [6.0, 7.0, 8.0, 9.0]);

    b.slice_mut(0, ..2).assign(0.0).unwrap();
    assert_eq!(a.to_vec().unwrap(), [0.0, 0.0, 8.0, 9.0]);

    // A slice taken through `&a` is read-only (a clone of it too); `slice_mut` writes through.
    let mut s = a.slice(0, 2..);
    assert!(matches!(s.assign(1.0).unwrap_err().root(), LumaError::InvalidShape(_)));
    assert!(matches!(s.clone().write(&[1.0, 1.0]).unwrap_err().root(), LumaError::InvalidShape(_)));
    s += 1.0;
    assert!(matches!(s.error().unwrap().root(), LumaError::InvalidShape(_)));
    assert_eq!(b.to_vec().unwrap(), [0.0, 0.0, 8.0, 9.0]);
    b.slice_mut(0, 2..).assign(1.0).unwrap();
    assert_eq!(a.to_vec().unwrap(), [0.0, 0.0, 1.0, 1.0]);
    assert_eq!(b.to_vec().unwrap(), [0.0, 0.0, 1.0, 1.0]);

    // Expressions recorded before a write keep the values they were recorded against.
    let old = &b * 1.0;
    a += 10.0;
    assert_eq!(old.to_vec().unwrap(), [0.0, 0.0, 1.0, 1.0]);
    assert_eq!(b.to_vec().unwrap(), [10.0, 10.0, 11.0, 11.0]);
}

#[test]
fn copy_is_independent_of_the_original() {
    let ctx = Context::new().unwrap();
    let a = f32s(&ctx, &[1.0, 2.0, 3.0, 4.0]);

    let mut c = a.copy();
    c += 1.0;
    assert_eq!(c.to_vec().unwrap(), [2.0, 3.0, 4.0, 5.0]);
    assert_eq!(a.to_vec().unwrap(), [1.0, 2.0, 3.0, 4.0]);

    // The copy of a slice is an ordinary array that writes only itself.
    let mut sv = a.slice(0, 1..3).copy();
    sv += 1.0;
    sv.write(&[7.0, 8.0]).unwrap();
    assert_eq!(sv.to_vec().unwrap(), [7.0, 8.0]);
    assert_eq!(a.to_vec().unwrap(), [1.0, 2.0, 3.0, 4.0]);
    let mut sm = a.copy();
    sm.slice_mut(0, ..2).assign(0.0).unwrap();
    assert_eq!(a.to_vec().unwrap(), [1.0, 2.0, 3.0, 4.0]);

    // The copy of a poisoned array is poisoned with the same error.
    let bad = &a + &f32s(&ctx, &[1.0, 2.0]);
    assert!(matches!(bad.copy().to_vec().unwrap_err().root(), LumaError::ShapeMismatch { .. }));

    // Element types without built-in arithmetic are copied on the device.
    let mut v = Array::<[f32; 2]>::from_slice_on(&ctx, &[[1.0, 2.0], [3.0, 4.0]], 2).unwrap();
    let w = v.copy();
    v.write(&[[0.0, 0.0], [0.0, 0.0]]).unwrap();
    assert_eq!(w.to_vec().unwrap(), [[1.0, 2.0], [3.0, 4.0]]);
}

#[test]
fn clones_of_a_lazy_array_share_the_expression_and_its_writes() {
    let ctx = Context::new().unwrap();
    let xs = values(64, 1, -1.0, 1.0);
    let x = f32s(&ctx, &xs);
    let twice: Vec<f32> = xs.iter().map(|&v| v * 2.0).collect();
    let plus: Vec<f32> = twice.iter().map(|&v| v + 1.0).collect();

    // `+=` writes the data both handles are on, like on two reshapes of one array.
    let mut a = &x * 2.0;
    let b = a.clone();
    a += 1.0;
    assert_close(&b.to_vec().unwrap(), &plus, 1e-6, 1e-6);
    assert_close(&a.to_vec().unwrap(), &plus, 1e-6, 1e-6);

    // So do `write` and `assign`.
    let mut a = &x * 2.0;
    let b = a.clone();
    let data = values(64, 2, 3.0, 4.0);
    a.write(&data).unwrap();
    assert_eq!(b.to_vec().unwrap(), data);

    let a = &x * 2.0;
    let mut b = a.clone();
    b.slice_mut(0, ..10).assign(0.0).unwrap();
    let got = a.to_vec().unwrap();
    assert!(got[..10].iter().all(|&v| v == 0.0));
    assert_close(&got[10..], &twice[10..], 1e-6, 1e-6);

    // Without a clone `+=` still just extends the expression, and an expression recorded before
    // it is unchanged.
    let mut a = &x * 2.0;
    let before = a.clone();
    let copy = &a * 1.0;
    a += 1.0;
    drop(before);
    assert_close(&copy.to_vec().unwrap(), &twice, 1e-6, 1e-6);
    assert_close(&a.to_vec().unwrap(), &plus, 1e-6, 1e-6);
}

#[test]
fn clones_of_failed_and_reduced_arrays_keep_what_they_are() {
    let ctx = Context::new().unwrap();
    let x = f32s(&ctx, &[1.0, 2.0, 3.0]);
    let bad = &x + &f32s(&ctx, &[1.0, 2.0]);
    let again = bad.clone();
    assert!(matches!(again.to_vec().unwrap_err().root(), LumaError::ShapeMismatch { .. }));
    assert!(again.error().is_some());

    let total = (&x * 2.0).sum();
    let copy = total.clone();
    assert_eq!(copy.item().unwrap(), 12.0);
    assert_eq!(total.item().unwrap(), 12.0);
    assert_eq!((&x / &copy).to_vec().unwrap(), [1.0 / 12.0, 2.0 / 12.0, 3.0 / 12.0]);
}

#[derive(Clone, Fields)]
struct Pair {
    a: Array<f32>,
    b: Array<f32>,
}

#[test]
fn a_record_of_arrays_can_derive_clone() {
    let ctx = Context::new().unwrap();
    let mut p = Pair { a: f32s(&ctx, &[1.0, 2.0]), b: f32s(&ctx, &[3.0, 4.0]) };
    let q = p.clone();
    p.a.slice_mut(0, ..1).assign(9.0).unwrap();
    assert_eq!(q.a.to_vec().unwrap(), [9.0, 2.0]);
    assert_eq!(q.b.to_vec().unwrap(), [3.0, 4.0]);
}
