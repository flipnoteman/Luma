//! Array data beyond the memory budget moves to host memory and comes back on demand.

use luma::{Array, Context, ContextOptions};

/// A context with 64 KiB segments and a 1 MiB budget, so small arrays already spill.
fn small_budget() -> Context {
    Context::with_options(ContextOptions::default().chunk_bytes(1 << 16).memory_budget(1 << 20))
    .unwrap()
}

/// Distinct, exactly representable values.
fn values(len: usize, salt: usize) -> Vec<f32> {
    (0..len).map(|i| ((i * 7 + salt * 131) % 1021) as f32).collect()
}

#[test]
fn arrays_larger_than_the_budget_compute_correctly() {
    let ctx = small_budget();
    let len = 1 << 20;
    let (x, y) = (values(len, 1), values(len, 2));
    let a = Array::from_slice_on(&ctx, &x, len).unwrap();
    let b = Array::from_slice_on(&ctx, &y, len).unwrap();

    let c = &a * 2.0 + &b;
    let expected: Vec<f32> = x.iter().zip(&y).map(|(x, y)| x * 2.0 + y).collect();
    assert_eq!(c.to_vec().unwrap(), expected);

    let sum = c.sum().item().unwrap();
    let reference: f64 = expected.iter().map(|&v| f64::from(v)).sum();
    assert!((f64::from(sum) - reference).abs() <= reference * 1e-4, "{sum} vs {reference}");
    assert!(ctx.spilled_bytes() > 0);
}

#[test]
fn many_single_segment_arrays_survive_spilling() {
    let ctx = small_budget();
    let len = 16_384;
    let arrays: Vec<Array<f32>> = (0..40)
        .map(|k| Array::from_slice_on(&ctx, &values(len, k), len).unwrap())
        .collect();
    for (k, array) in arrays.iter().enumerate() {
        assert_eq!(array.to_vec().unwrap(), values(len, k), "array {k}");
    }
    assert!(ctx.spilled_bytes() > 0);
}

#[test]
fn spilled_arrays_keep_working_as_operands() {
    let ctx = small_budget();
    let len = 16_384;
    let mut arrays: Vec<Array<f32>> = (0..40)
        .map(|k| Array::from_slice_on(&ctx, &values(len, k), len).unwrap())
        .collect();
    assert!(ctx.spilled_bytes() > 0);

    // The first arrays were created long ago and have been moved out.
    let fresh = values(len, 99);
    arrays[0].write(&fresh).unwrap();
    assert_eq!(arrays[0].to_vec().unwrap(), fresh);

    arrays[1] += 1.0;
    let expected: Vec<f32> = values(len, 1).iter().map(|v| v + 1.0).collect();
    assert_eq!(arrays[1].to_vec().unwrap(), expected);

    let sum = (&arrays[2] + &arrays[3]).to_vec().unwrap();
    let expected: Vec<f32> = values(len, 2).iter().zip(values(len, 3)).map(|(a, b)| a + b).collect();
    assert_eq!(sum, expected);
}

#[test]
fn a_default_context_does_not_spill_small_workloads() {
    let ctx = Context::new().unwrap();
    let len = 1 << 17;
    let (x, y) = (values(len, 1), values(len, 2));
    let a = Array::from_slice_on(&ctx, &x, len).unwrap();
    let b = Array::from_slice_on(&ctx, &y, len).unwrap();
    let c = &a * 2.0 + &b;
    let expected: Vec<f32> = x.iter().zip(&y).map(|(x, y)| x * 2.0 + y).collect();
    assert_eq!(c.to_vec().unwrap(), expected);
    assert_eq!(ctx.spilled_bytes(), 0);
}

#[test]
fn eviction_while_tracing_an_iterate_step_completes() {
    let ctx = small_budget();
    let len = 16_384;
    // The state is the oldest resident segment; the fillers bring the context to its budget.
    let x = Array::from_slice_on(&ctx, &values(len, 0), len).unwrap();
    let fillers: Vec<Array<f32>> = (0..14).map(|k| Array::from_slice_on(&ctx, &values(len, k + 1), len).unwrap()).collect();
    let step = |[u]: &[Array<f32>; 1]| {
        let a = (u * 0.5 + 1.0).cache();
        let b = (&a * 0.5 + u).cache();
        let c = (&b * 0.25 + &a).cache();
        [c * 0.5]
    };
    let [got] = luma::iterate_all([&x], 5, step);
    let mut want = values(len, 0);
    for _ in 0..5 {
        for v in &mut want {
            let a = *v * 0.5 + 1.0;
            let b = a * 0.5 + *v;
            *v = (b * 0.25 + a) * 0.5;
        }
    }
    let got = got.to_vec().unwrap();
    assert!(got.iter().zip(&want).all(|(g, w)| (g - w).abs() <= w.abs() * 1e-5 + 1e-5));
    for (k, f) in fillers.iter().enumerate() {
        assert_eq!(f.to_vec().unwrap(), values(len, k + 1));
    }
}

#[test]
fn readbacks_in_a_loop_under_a_budget_do_not_deadlock() {
    let (done, finished) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let ctx = small_budget();
        let len = 1 << 17;
        // Arrays of half the budget each: every step has to evict while it speculates.
        let held: Vec<Array<f32>> = (0..6).map(|k| Array::from_slice_on(&ctx, &values(len, k), len).unwrap()).collect();
        let mut x = Array::from_slice_on(&ctx, &values(len, 9), len).unwrap();
        let mut total = 0.0f64;
        // Enough steps for the loop to be recognised and its readbacks speculated, which is when
        // the eviction path re-entered the speculation lock.
        for step in 0..24 {
            x = &x * 0.5 + &held[step % held.len()];
            total += f64::from(x.sum().item().unwrap());
        }
        let _ = done.send(total);
    });
    // Generous: a debug build on a small CI runner's software device is slow; a deadlock never ends.
    let total = finished.recv_timeout(std::time::Duration::from_secs(240)).expect("deadlock in a readback under a memory budget");
    assert!(total > 0.0);
}
