// wgpu's handle types nest deeply; see tests/batching.rs.
#![recursion_limit = "256"]

//! `Array::submit` and `submit_all` record and hand the batch to the GPU without waiting for it.
//!
//! One test measures wall-clock time, so the tests of this binary run one at a time: other
//! tests sharing the GPU at the same time can stall submissions for milliseconds.

mod common;

use std::sync::{Mutex, MutexGuard, PoisonError};
use std::time::{Duration, Instant};

use common::{assert_close, values};
use luma::*;

static SERIAL: Mutex<()> = Mutex::new(());

fn serial() -> MutexGuard<'static, ()> {
    SERIAL.lock().unwrap_or_else(PoisonError::into_inner)
}

fn f32s(data: &[f32]) -> Array<f32> {
    Array::from_slice(data, data.len()).unwrap()
}

#[test]
fn submit_then_to_vec_gives_what_eval_gives() {
    let _serial = serial();
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
    let _serial = serial();
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
    let _serial = serial();
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
    let _serial = serial();
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
    let _serial = serial();
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
    let _serial = serial();
    let one = Context::new().unwrap();
    let two = Context::new().unwrap();
    let a = Array::from_slice_on(&one, &[1.0f32, 2.0], 2).unwrap();
    let b = Array::from_slice_on(&two, &[1.0f32, 2.0], 2).unwrap();
    let err = submit_all(&[&a, &b]).unwrap_err();
    assert!(matches!(err, LumaError::ContextMismatch), "{err}");
}

#[test]
fn submit_returns_while_the_gpu_is_still_running() {
    let _serial = serial();
    const SPIN: &str = r#"
    struct Params { rounds: u32 }
    @group(1) @binding(0) var<storage, read_write> a: array<f32>;
    @group(1) @binding(1) var<uniform> p: Params;
    @compute @workgroup_size(64)
    fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
        if (gid.x >= luma.size.x) { return; }
        var v = a[gid.x];
        for (var i = 0u; i < p.rounds; i = i + 1u) { v = v * 1.0000001 + 0.5; }
        a[gid.x] = v;
    }"#;
    #[derive(encase::ShaderType)]
    struct Params {
        rounds: u32,
    }

    let ctx = Context::new().unwrap();
    let spin = ctx.kernel(SPIN).build().unwrap();
    let n = 1u32 << 16;
    let mut a = Array::<f32>::zeros_on(&ctx, n as usize).unwrap();

    // Find a round count that takes the GPU a few tens of milliseconds, however fast it is.
    let mut rounds = 1u32 << 10;
    loop {
        let start = Instant::now();
        spin.launch([n], args![&mut a, uniform(&Params { rounds })]).unwrap();
        ctx.sync().unwrap();
        if start.elapsed() > Duration::from_millis(30) || rounds >= 1 << 28 {
            break;
        }
        rounds *= 2;
    }

    for all in [false, true] {
        spin.launch([n], args![&mut a, uniform(&Params { rounds })]).unwrap();
        let start = Instant::now();
        if all {
            submit_all(&[&a]).unwrap();
        } else {
            a.submit().unwrap();
        }
        let submit_time = start.elapsed();
        let start = Instant::now();
        ctx.sync().unwrap();
        let sync_time = start.elapsed();
        // The GPU ran while submit returned: the wait that is left is far longer than the submit.
        assert!(
            submit_time * 4 < sync_time,
            "submit (all: {all}) took {submit_time:?}, the sync after it {sync_time:?}"
        );
    }
}
