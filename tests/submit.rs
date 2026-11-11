// wgpu's handle types nest deeply; see tests/batching.rs.
#![recursion_limit = "256"]

//! `Array::submit` and `submit_all` record and hand the batch to the GPU without waiting for it.

mod common;

use common::{assert_close, values};
use luma::*;

fn f32s(data: &[f32]) -> Array<f32> {
    Array::from_slice(data, data.len()).unwrap()
}

#[test]
fn submit_then_to_vec_gives_what_eval_gives() {
    let x = f32s(&[1.0, 2.0, 3.0, 4.0]);

    let evaluated = &x * 2.0 + 1.0;
    evaluated.eval().unwrap();
    let submitted = &x * 2.0 + 1.0;
    submitted.submit().unwrap();

    assert_eq!(submitted.to_vec().unwrap(), evaluated.to_vec().unwrap());
    assert_eq!(submitted.to_vec().unwrap(), vec![3.0, 5.0, 7.0, 9.0]);

    // An array that is already computed, and submitting twice, are fine.
    submitted.submit().unwrap();
    x.submit().unwrap();
    assert_eq!(x.to_vec().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn submitted_work_runs_in_order_with_what_is_recorded_after_it() {
    let mut x = Array::<f32>::zeros(8).unwrap();
    // A view sharing the buffer makes every compound assignment a dispatch of its own.
    let view = x.reshape(8);
    x += 2.0;
    x.submit().unwrap();
    x *= 3.0;
    x += 1.0;
    assert_eq!(view.to_vec().unwrap(), vec![7.0; 8]);
}

#[test]
fn a_loop_of_submit_all_steps_needs_one_read_back() {
    const N: usize = 1000;
    let c = values(N, 7, -1.0, 1.0);
    let carr = f32s(&c);
    let mut z = Array::<f32>::zeros(N).unwrap();
    let mut sum = Array::<f32>::zeros(N).unwrap();
    // More steps than a batch holds dispatches, so some submissions happen on their own too.
    let steps = 600;
    for _ in 0..steps {
        z = &z * 0.5 + &carr;
        sum = &sum + &z;
        submit_all(&[&z, &sum]).unwrap();
    }

    let (mut want_z, mut want_sum) = (vec![0.0f32; N], vec![0.0f32; N]);
    for _ in 0..steps {
        for i in 0..N {
            want_z[i] = want_z[i] * 0.5 + c[i];
            want_sum[i] += want_z[i];
        }
    }
    assert_close(&z.to_vec().unwrap(), &want_z, 1e-5, 1e-5);
    assert_close(&sum.to_vec().unwrap(), &want_sum, 1e-4, 1e-4);
}

#[test]
fn submit_all_takes_arrays_of_different_lengths_and_no_arrays() {
    submit_all::<f32>(&[]).unwrap();

    let a = &f32s(&[1.0, 2.0, 3.0]) * 2.0;
    let b = &f32s(&[10.0, 20.0, 30.0, 40.0, 50.0]) + 1.0;
    let c = &f32s(&[1.0; 3]) + &a; // the same length as `a`, and uses it
    submit_all(&[&a, &b, &c]).unwrap();
    assert_eq!(a.to_vec().unwrap(), vec![2.0, 4.0, 6.0]);
    assert_eq!(b.to_vec().unwrap(), vec![11.0, 21.0, 31.0, 41.0, 51.0]);
    assert_eq!(c.to_vec().unwrap(), vec![3.0, 5.0, 7.0]);
}

#[test]
fn poisoned_arrays_return_their_error_from_submit() {
    let a = f32s(&[1.0, 2.0, 3.0]);
    let bad = &a + &f32s(&[1.0]);

    let err = bad.submit().unwrap_err();
    assert!(matches!(err.root(), LumaError::ShapeMismatch { .. }), "{err}");

    let err = submit_all(&[&a, &bad]).unwrap_err();
    assert!(matches!(err.root(), LumaError::ShapeMismatch { .. }), "{err}");

    // The healthy array is unharmed.
    a.submit().unwrap();
    assert_eq!(a.to_vec().unwrap(), vec![1.0, 2.0, 3.0]);
}

#[test]
fn submit_all_rejects_arrays_of_different_contexts() {
    let one = Context::new().unwrap();
    let two = Context::new().unwrap();
    let a = Array::from_slice_on(&one, &[1.0f32, 2.0], 2).unwrap();
    let b = Array::from_slice_on(&two, &[1.0f32, 2.0], 2).unwrap();
    let err = submit_all(&[&a, &b]).unwrap_err();
    assert!(matches!(err, LumaError::ContextMismatch), "{err}");
}
