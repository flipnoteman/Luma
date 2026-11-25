//! The replay engine behind compiled callables (`crate::trace`): a function traced once and
//! replayed over new arrays, with no graph building per call.

use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

use crate::trace::{trace, trace_with, Failure, Outputs, ShapeClass, TraceError, TraceOptions};
use crate::*;

/// Deterministic pseudo-random values in `lo..hi` (a small LCG).
fn values(n: usize, seed: u64, lo: f32, hi: f32) -> Vec<f32> {
    let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            lo + ((state >> 40) as f32) / (1u64 << 24) as f32 * (hi - lo)
        })
        .collect()
}

fn assert_close(got: &[f32], want: &[f32], rel: f32, abs: f32) {
    assert_eq!(got.len(), want.len(), "length");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!((g - w).abs() <= abs.max(rel * w.abs()), "index {i}: got {g}, want {w}");
    }
}

fn ctx() -> Context {
    Context::new().unwrap()
}

fn f32s(ctx: &Context, data: &[f32]) -> Array<f32> {
    Array::from_slice_on(ctx, data, data.len()).unwrap()
}

fn on(ctx: &Context, shape: impl Into<Shape>, seed: u64) -> Array<f32> {
    let shape = shape.into();
    Array::from_slice_on(ctx, &values(shape.len(), seed, -1.0, 1.0), shape).unwrap()
}

fn bits(array: &Array<f32>) -> Vec<u32> {
    array.to_vec().unwrap().iter().map(|v| v.to_bits()).collect()
}

/// The `TraceError` of a refused call.
fn trace_error(error: &Failure) -> &TraceError {
    error.trace_error().unwrap_or_else(|| panic!("not a trace error: {error}"))
}

/// The error of the library underneath a failure.
fn root_of(error: &Failure) -> LumaError {
    match error {
        Failure::New(e) => e.root().clone(),
        Failure::Inherited(e) => e.root().clone(),
        Failure::Trace(e) => LumaError::Unsupported(e.to_string()),
    }
}

/// The step of the threads test: a padded copy with a ghost written by `assign`, a staged `cache()`.
fn stencil(u: &Array<f32>) -> Array<f32> {
    let n = u.len();
    let mut w = u.pad([(1, 1)], PadMode::Edge);
    w.slice_mut(0, ..1).assign(0.5).unwrap();
    let c = (w.slice(0, 1..n + 1) * 0.5 + w.slice(0, ..n) * 0.25 + w.slice(0, 2..n + 2) * 0.25).cache();
    c * 0.99 + 0.001
}

/// One step of a one-layer network: the loss and the activations.
fn mlp_step(x: &Array<f32>, w: &Array<f32>, b: &Array<f32>) -> (Array<f32>, Array<f32>) {
    let y = (x.matmul(w) + b).maximum(0.0);
    ((&y * &y).sum(), y)
}

// -- Correctness ---------------------------------------------------------------------------------

#[test]
fn a_compiled_step_matches_the_eager_function_bit_for_bit() {
    let ctx = ctx();
    // A stencil with a ghost assignment and a cache.
    let u = on(&ctx, 1000, 1);
    let compiled = trace(|inp| stencil(&inp.get::<f32>(0)).into());
    let got = compiled.call1::<f32>(&[&u]).unwrap();
    assert_eq!(bits(&got), bits(&stencil(&u)));
    // The same variant on another array.
    let v = on(&ctx, 1000, 2);
    assert_eq!(bits(&compiled.call1::<f32>(&[&v]).unwrap()), bits(&stencil(&v)));
    assert_eq!(compiled.stats().traces, 1);

    // A network step: two outputs of different shapes.
    let (x, w, b) = (on(&ctx, [16, 32], 3), on(&ctx, [8, 16], 4), on(&ctx, [8, 1], 5));
    let step = trace(|inp| {
        let (loss, y) = mlp_step(&inp.get::<f32>(0), &inp.get::<f32>(1), &inp.get::<f32>(2));
        (loss, y).into()
    });
    let mut out = step.call(&[&x, &w, &b]).unwrap();
    let (loss, y): (Array<f32>, Array<f32>) = (out.take(0).unwrap(), out.take(1).unwrap());
    let (want_loss, want_y) = mlp_step(&x, &w, &b);
    assert_eq!(bits(&loss), bits(&want_loss));
    assert_eq!(bits(&y), bits(&want_y));

    // A pair of reductions.
    let r = on(&ctx, 5000, 6);
    let pair = trace(|inp| {
        let x = inp.get::<f32>(0);
        (x.sum(), x.max(), &x - x.mean()).into()
    });
    let mut out = pair.call(&[&r]).unwrap();
    assert_eq!(bits(&out.take::<f32>(0).unwrap()), bits(&r.sum()));
    assert_eq!(bits(&out.take::<f32>(1).unwrap()), bits(&r.max()));
    assert_eq!(bits(&out.take::<f32>(2).unwrap()), bits(&(&r - r.mean())));
}

#[test]
fn the_second_call_runs_on_new_buffers() {
    let ctx = ctx();
    let compiled = trace(|inp| (&inp.get::<f32>(0) * 2.0 + 1.0).into());
    let (a, b) = (on(&ctx, 4096, 7), on(&ctx, 4096, 8));
    let first = compiled.call1::<f32>(&[&a]).unwrap();
    let second = compiled.call1::<f32>(&[&b]).unwrap();
    let want = |x: &Array<f32>| x.to_vec().unwrap().iter().map(|v| v * 2.0 + 1.0).collect::<Vec<_>>();
    assert_eq!(first.to_vec().unwrap(), want(&a));
    assert_eq!(second.to_vec().unwrap(), want(&b));
    let stats = compiled.stats();
    assert_eq!((stats.calls, stats.traces, stats.retraces), (2, 1, 0));
}

#[test]
fn a_matmul_in_a_recording_runs_the_tile_family_the_eager_step_does() {
    // The product of an MLP layer (native on the CPU and Metal backends, tiled or tensor-core GEMM on the GPU ones):
    // what a recording captures is the kernel the planner chose without one, not a shader fallback.
    let ctx = ctx();
    let (x, w, b) = (on(&ctx, [784, 256], 3), on(&ctx, [512, 784], 4), on(&ctx, [512, 1], 5));
    ctx.record_decisions(true);
    let (loss, _) = mlp_step(&x, &w, &b);
    loss.eval().unwrap();
    let eager: Vec<String> = ctx.take_decisions().into_iter().filter(|d| d.op == "matmul").map(|d| d.chosen.into_owned()).collect();
    ctx.record_decisions(false);
    assert_eq!(eager.len(), 1, "{eager:?}");
    if eager[0] == "host" {
        // A small product on a unified-memory device runs on the host outside a recording; nothing to compare.
        return;
    }
    let family = |chosen: &str| match chosen.split(' ').next() {
        Some("native") => "Native",
        Some("gemm") => "Gemm",
        Some("skinny") => "Skinny",
        _ => "T",
    };
    assert!(ctx.begin_recording(Vec::new()));
    let (loss, _) = mlp_step(&x, &w, &b);
    loss.eval().unwrap();
    let recording = ctx.end_recording().expect("the step records");
    let tiles: Vec<&str> = recording.cmds.iter().map(|cmd| &*cmd.kernel.label).filter(|label| label.starts_with("Matmul")).collect();
    assert_eq!(tiles.len(), 1, "{tiles:?}");
    assert!(tiles[0].contains(&format!("tile: {}", family(&eager[0]))), "eager chose {:?}, the recording captured {:?}", eager[0], tiles[0]);
    if matches!(ctx.backend(), luma_core::BackendKind::Cpu) {
        assert!(tiles[0].contains("tile: Native"), "{:?}", tiles[0]);
    }
}

#[test]
fn outputs_are_independent_arrays() {
    let ctx = ctx();
    let compiled = trace(|inp| (&inp.get::<f32>(0) + 1.0).into());
    let (a, b) = (f32s(&ctx, &[1.0; 64]), f32s(&ctx, &[5.0; 64]));
    let mut first = compiled.call1::<f32>(&[&a]).unwrap();
    let second = compiled.call1::<f32>(&[&b]).unwrap();
    first.write(&[100.0; 64]).unwrap();
    assert_eq!(second.to_vec().unwrap(), [6.0; 64]);
    let third = compiled.call1::<f32>(&[&a]).unwrap();
    assert_eq!(third.to_vec().unwrap(), [2.0; 64]);
    assert_eq!(first.to_vec().unwrap(), [100.0; 64]);
    assert_eq!(a.to_vec().unwrap(), [1.0; 64], "the argument is not touched");
}

#[test]
fn pass_through_and_captured_outputs_return_the_same_data() {
    let ctx = ctx();
    let w = f32s(&ctx, &[3.0; 32]);
    let captured = w.clone();
    let compiled = trace(move |inp| {
        let x = inp.get::<f32>(0);
        (x.clone(), captured.clone(), &x * 2.0).into()
    });
    let x = f32s(&ctx, &values(32, 9, 0.0, 1.0));
    for _ in 0..2 {
        let mut out = compiled.call(&[&x]).unwrap();
        assert_eq!(out.take::<f32>(0).unwrap().to_vec().unwrap(), x.to_vec().unwrap());
        assert_eq!(out.take::<f32>(1).unwrap().to_vec().unwrap(), [3.0; 32]);
        assert_eq!(out.take::<f32>(2).unwrap().to_vec().unwrap(), x.to_vec().unwrap().iter().map(|v| v * 2.0).collect::<Vec<_>>());
    }
}

#[test]
fn multi_output_reductions_come_back_as_independent_arrays() {
    let ctx = ctx();
    let compiled = trace(|inp| {
        let x = inp.get::<f32>(0);
        (x.sum(), x.max(), x.min(), (&x * &x).sum()).into()
    });
    let xs = values(10_000, 10, -1.0, 1.0);
    let x = f32s(&ctx, &xs);
    let mut out = compiled.call(&[&x]).unwrap();
    let mut got: Vec<Array<f32>> = (0..4).map(|k| out.take(k).unwrap()).collect();
    let want = [x.sum(), x.max(), x.min(), (&x * &x).sum()];
    for (g, w) in got.iter().zip(&want) {
        assert_eq!(bits(g), bits(w));
    }
    // Writing one leaves the others (and the next call's) alone.
    got[0].write(&[42.0]).unwrap();
    assert_eq!(bits(&got[1]), bits(&want[1]));
    let mut again = compiled.call(&[&x]).unwrap();
    assert_eq!(bits(&again.take::<f32>(0).unwrap()), bits(&want[0]));
}

#[test]
fn a_captured_weight_updated_in_place_is_seen_by_the_next_call() {
    let ctx = ctx();
    let mut w = f32s(&ctx, &[1.0, 2.0, 3.0, 4.0]);
    let captured = w.clone();
    let compiled = trace(move |inp| (&inp.get::<f32>(0) * &captured).into());
    let x = f32s(&ctx, &[1.0; 4]);
    assert_eq!(compiled.call1::<f32>(&[&x]).unwrap().to_vec().unwrap(), [1.0, 2.0, 3.0, 4.0]);
    w.write(&[10.0, 20.0, 30.0, 40.0]).unwrap();
    assert_eq!(compiled.call1::<f32>(&[&x]).unwrap().to_vec().unwrap(), [10.0, 20.0, 30.0, 40.0]);
    // One that is replaced is not seen: the trace read the buffer it was traced with.
    w = f32s(&ctx, &[7.0; 4]);
    assert_eq!(compiled.call1::<f32>(&[&x]).unwrap().to_vec().unwrap(), [10.0, 20.0, 30.0, 40.0]);
    drop(w);
}

#[test]
fn a_scalar_read_inside_the_function_is_a_constant() {
    let ctx = ctx();
    let scale = Scalar::new(2.0f32);
    let inside = scale.clone();
    let compiled = trace(move |inp| (&inp.get::<f32>(0) * &inside).into());
    let x = f32s(&ctx, &[1.0, 2.0, 3.0]);
    assert_eq!(compiled.call1::<f32>(&[&x]).unwrap().to_vec().unwrap(), [2.0, 4.0, 6.0]);
    scale.set(3.0);
    assert_eq!(compiled.call1::<f32>(&[&x]).unwrap().to_vec().unwrap(), [2.0, 4.0, 6.0], "the trace froze the value it read");
}

#[test]
fn a_scalar_parameter_is_a_one_element_array() {
    let ctx = ctx();
    let sgd = trace(|inp| {
        let (w, g, lr) = (inp.get::<f32>(0), inp.get::<f32>(1), inp.get::<f32>(2));
        (&w - &g * &lr).into()
    });
    let (w, g) = (f32s(&ctx, &[1.0, 2.0, 3.0]), f32s(&ctx, &[1.0, 1.0, 1.0]));
    for lr in [0.1f32, 0.5, 2.0] {
        let step = Array::from_slice_on(&ctx, &[lr], 1).unwrap();
        let got = sgd.call1::<f32>(&[&w, &g, &step]).unwrap().to_vec().unwrap();
        assert_eq!(got, [1.0 - lr, 2.0 - lr, 3.0 - lr]);
    }
    assert_eq!(sgd.stats().traces, 1, "a new learning rate is a new array, not a new trace");
}

#[test]
fn results_and_arguments_of_other_element_types_trace() {
    let ctx = ctx();
    let compiled = trace(|inp| {
        let (x, k) = (inp.get::<f32>(0), inp.get::<u32>(1));
        (x.gt(0.0), &k + 1u32, x.sum()).into()
    });
    let x = f32s(&ctx, &[-1.0, 2.0, -3.0, 4.0]);
    let k = Array::<u32>::from_slice_on(&ctx, &[1, 2, 3, 4], 4).unwrap();
    let mut out = compiled.call(&[&x, &k]).unwrap();
    assert_eq!(out.take::<bool>(0).unwrap().to_vec().unwrap(), [false, true, false, true]);
    assert_eq!(out.take::<u32>(1).unwrap().to_vec().unwrap(), [2, 3, 4, 5]);
    assert_eq!(out.take::<f32>(2).unwrap().to_vec().unwrap(), [2.0]);
    // A wrong type says what is there, and the output stays.
    let mut out = compiled.call(&[&x, &k]).unwrap();
    let message = out.take::<f32>(1).unwrap_err().to_string();
    assert!(message.contains("holds u32, not f32"), "{message}");
    assert_eq!(out.take::<u32>(1).unwrap().to_vec().unwrap(), [2, 3, 4, 5]);
    assert!(out.take::<u32>(1).unwrap_err().to_string().contains("already taken"));
    assert_eq!(out.into_vec().len(), 2);
}

// -- Library operations --------------------------------------------------------------------------

/// A function of one `f32` array of shape `[64, 64]` returning `f32` arrays.
type Case = (&'static str, fn(&Array<f32>) -> Vec<Array<f32>>);

fn library_cases() -> Vec<Case> {
    vec![
        ("elementwise with shifts and cache", |x| vec![(x.shift([1, 0]) + x.shift([-1, 0]) - x * 2.0).cache() * 0.5]),
        ("whole reduction", |x| vec![(x * x).sum()]),
        ("axis reductions", |x| vec![x.sum_axis(0), x.sum_axis(1)]),
        ("matmul", |x| vec![x.matmul(&x.transpose())]),
        ("matmul_f32", |x| vec![x.matmul_f32(x)]),
        ("slice_mut assignment into an array made inside", |x| {
            let mut w = x * 1.0;
            w.slice_mut(0, ..4).assign(0.5).unwrap();
            vec![w + 1.0]
        }),
        ("sort", |x| vec![x.sort()]),
        ("cumsum", |x| vec![x.cumsum(1)]),
        ("conv1d", |x| {
            let kernel = x.slice(0, ..1).reshape([64]).slice(0, ..5);
            vec![x.reshape([4096]).conv1d(&kernel.copy(), 1, 2)]
        }),
    ]
}

#[test]
fn these_library_ops_trace() {
    let ctx = ctx();
    for (name, f) in library_cases() {
        let compiled = trace(move |inp| f(&inp.get::<f32>(0)).into());
        for seed in [21, 22] {
            let x = on(&ctx, [64, 64], seed);
            let mut got = match compiled.call(&[&x]) {
                Ok(got) => got,
                Err(error) => panic!("{name}: {error}"),
            };
            for (k, want) in f(&x).iter().enumerate() {
                assert_eq!(bits(&got.take::<f32>(k).unwrap()), bits(want), "{name} output {k}");
            }
        }
        assert_eq!(compiled.stats().traces, 1, "{name}");
    }
    // The radix argsort has `u32` results.
    let compiled = trace(|inp| inp.get::<f32>(0).argsort().into());
    for seed in [23, 24] {
        let x = on(&ctx, 4096, seed);
        assert_eq!(compiled.call1::<u32>(&[&x]).unwrap().to_vec().unwrap(), x.argsort().to_vec().unwrap());
    }
}

type Body = Box<dyn Fn(&Array<f32>) -> Outputs + Send + Sync>;

/// The error the first call of a function traced over `x` gives.
fn first_error(body: Body, x: &Array<f32>) -> Failure {
    trace(move |inp| body(&inp.get::<f32>(0))).call(&[x]).unwrap_err()
}

#[test]
fn these_ops_say_why_they_cannot() {
    let ctx = ctx();
    let x = on(&ctx, [64, 64], 31);
    let cases: Vec<(&str, Body)> = vec![
        ("to_vec", Box::new(|x| {
            let _ = x.to_vec();
            (x * 2.0).into()
        })),
        ("item", Box::new(|x| {
            let _ = x.sum().item();
            (x * 2.0).into()
        })),
        ("unique", Box::new(|x| x.unique().into())),
        ("nonzero", Box::new(|x| match x.gt(0.0).nonzero() {
            Ok(Some(found)) => found.into(),
            _ => (x * 2.0).into(),
        })),
    ];
    for (name, body) in cases {
        let error = first_error(body, &x);
        match trace_error(&error) {
            TraceError::Aborted { reason } => println!("{name}: {reason}"),
            other => panic!("{name}: {other}"),
        }
    }
}

#[test]
fn readback_in_the_function_is_an_aborted_trace_naming_the_reason() {
    let ctx = ctx();
    let x = on(&ctx, 4096, 32);
    let inside = ctx.clone();
    let cases: Vec<(&str, &str, Body)> = vec![
        ("to_vec", "readback", Box::new(|x| {
            let _ = x.to_vec();
            (x * 2.0).into()
        })),
        ("item", "readback", Box::new(|x| {
            let _ = x.sum().item();
            (x * 2.0).into()
        })),
        ("sync", "sync", {
            let ctx = ctx.clone();
            Box::new(move |x| {
                ctx.sync().unwrap();
                (x * 2.0).into()
            })
        }),
        ("zeros", "clear", {
            let ctx = ctx.clone();
            Box::new(move |x| (x + Array::<f32>::zeros_on(&ctx, 4096).unwrap()).into())
        }),
        ("write", "upload", Box::new(|x| {
            let mut y = x * 2.0;
            y.write(&[1.0; 4096]).unwrap();
            y.into()
        })),
    ];
    for (name, reason, body) in cases {
        let error = first_error(body, &x);
        let TraceError::Aborted { reason: got } = trace_error(&error) else { panic!("{name}: {error}") };
        assert_eq!(*got, reason, "{name}");
    }
    // Host data goes to a device through a transfer, which a recording cannot hold; where the
    // device shares memory with the host (the CPU backend) the array is a constant of the trace.
    let compiled = trace(move |inp| (&inp.get::<f32>(0) + Array::from_slice_on(&inside, &[1.0f32; 4096], 4096).unwrap()).into());
    match compiled.call1::<f32>(&[&x]) {
        Err(error) => assert!(matches!(trace_error(&error), TraceError::Aborted { reason: "upload" }), "{error}"),
        Ok(got) => assert_eq!(bits(&got), bits(&(&x + 1.0))),
    }
}

#[test]
fn writing_an_input_is_refused() {
    let ctx = ctx();
    let x = on(&ctx, 256, 33);
    let error = first_error(
        Box::new(|x| {
            let mut y = x.clone();
            y.slice_mut(0, ..2).assign(0.0).unwrap();
            y.into()
        }),
        &x,
    );
    assert!(matches!(trace_error(&error), TraceError::WritesInput { index: 0 }), "{error}");
    let error = first_error(
        Box::new(|x| {
            let mut y = x.clone();
            y += 1.0;
            y.into()
        }),
        &x,
    );
    assert!(matches!(trace_error(&error), TraceError::WritesInput { index: 0 }), "{error}");
    assert_eq!(x.to_vec().unwrap(), values(256, 33, -1.0, 1.0), "the argument is untouched");
}

#[test]
fn writing_a_captured_array_is_refused() {
    let ctx = ctx();
    let x = on(&ctx, 256, 34);
    let captured = f32s(&ctx, &[1.0; 256]);
    let error = first_error(
        Box::new(move |x| {
            let mut w = captured.clone();
            w.slice_mut(0, ..2).assign(5.0).unwrap();
            (x + &w).into()
        }),
        &x,
    );
    assert!(matches!(trace_error(&error), TraceError::WritesCaptured), "{error}");
}

#[test]
fn a_function_that_launches_nothing_and_returns_nothing_of_its_inputs_is_empty() {
    let ctx = ctx();
    let x = on(&ctx, 16, 35);
    let captured = f32s(&ctx, &[1.0; 16]);
    let error = first_error(Box::new(move |_| captured.clone().into()), &x);
    assert!(matches!(trace_error(&error), TraceError::Empty), "{error}");
    // Handing an input back is a function.
    let same = trace(|inp| inp.get::<f32>(0).into()).call1::<f32>(&[&x]).unwrap();
    assert_eq!(same.to_vec().unwrap(), x.to_vec().unwrap());
}

#[test]
fn a_chunked_argument_gives_segmented() {
    let ctx = Context::with_options(ContextOptions::from_env().chunk_bytes(4096)).unwrap();
    let x = on(&ctx, 10_000, 36);
    let compiled = trace(|inp| (&inp.get::<f32>(0) * 2.0).into());
    let error = compiled.call(&[&x]).unwrap_err();
    assert!(matches!(trace_error(&error), TraceError::Segmented { what: "an argument" }), "{error}");
}

#[test]
fn a_poisoned_argument_returns_its_error() {
    let ctx = ctx();
    let compiled = trace(|inp| (&inp.get::<f32>(0) * 2.0).into());
    let good = on(&ctx, 3, 37);
    let bad = &good + &on(&ctx, 2, 38);
    let error = compiled.call(&[&bad]).unwrap_err();
    assert!(matches!(root_of(&error), LumaError::ShapeMismatch { .. }), "{error}");
    // With a variant for the shape, too.
    compiled.call1::<f32>(&[&good]).unwrap();
    let error = compiled.call(&[&bad]).unwrap_err();
    assert!(matches!(root_of(&error), LumaError::ShapeMismatch { .. }), "{error}");
    assert_eq!(compiled.stats().traces, 1);
}

#[test]
fn arguments_of_two_contexts_are_refused() {
    let (one, other) = (ctx(), ctx());
    let compiled = trace(|inp| (&inp.get::<f32>(0) + &inp.get::<f32>(1)).into());
    let error = compiled.call(&[&on(&one, 8, 1), &on(&other, 8, 2)]).unwrap_err();
    assert!(matches!(root_of(&error), LumaError::ContextMismatch), "{error}");
}

#[test]
fn a_panic_in_the_function_leaves_no_variant_and_no_recording() {
    let ctx = ctx();
    let x = on(&ctx, 64, 39);
    let broken = trace(|inp| {
        let y = &inp.get::<f32>(0) * 2.0;
        y.eval().unwrap();
        panic!("the function failed");
    });
    let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| broken.call(&[&x])));
    assert!(panicked.is_err());
    assert!(broken.variants().is_empty());
    // This thread records again: nothing is left of the first trace.
    let fine = trace(|inp| (&inp.get::<f32>(0) + 1.0).into());
    assert_eq!(fine.call1::<f32>(&[&x]).unwrap().to_vec().unwrap(), x.to_vec().unwrap().iter().map(|v| v + 1.0).collect::<Vec<_>>());
}

// -- Variants ------------------------------------------------------------------------------------

#[test]
fn a_new_shape_retraces_and_keeps_the_old_variant() {
    let ctx = ctx();
    let compiled = trace(|inp| (&inp.get::<f32>(0) * 3.0).into());
    let (a, b) = (on(&ctx, 100, 1), on(&ctx, 200, 2));
    for x in [&a, &b, &a] {
        let got = compiled.call1::<f32>(&[x]).unwrap();
        assert_eq!(got.to_vec().unwrap(), x.to_vec().unwrap().iter().map(|v| v * 3.0).collect::<Vec<_>>());
    }
    let stats = compiled.stats();
    assert_eq!((stats.calls, stats.traces, stats.retraces, stats.variants), (3, 2, 1, 2));
    let variants = compiled.variants();
    assert_eq!(variants.len(), 2);
    assert_eq!((variants[0].shapes.clone(), variants[0].calls), (vec![Shape::from(100)], 2), "the most recently called first");
    assert_eq!((variants[1].shapes.clone(), variants[1].calls), (vec![Shape::from(200)], 1));
    assert_eq!(variants[0].elems, [ElemKind::F32]);
    assert_eq!(variants[0].classes, [ShapeClass::of(Shape::from(100))]);
    // An argument of another element type is another signature.
    let u = Array::<u32>::zeros_on(&ctx, 100).unwrap();
    let compiled = trace(|inp| (&inp.get::<u32>(0) + 1u32).into());
    compiled.call1::<u32>(&[&u]).unwrap();
    assert_eq!(compiled.variants()[0].elems, [ElemKind::U32]);
}

#[test]
fn strict_mode_errors_on_an_unknown_signature() {
    let ctx = ctx();
    let compiled = trace_with(TraceOptions::new().strict(), |inp| (&inp.get::<f32>(0) * 3.0).into());
    let (a, b) = (on(&ctx, 100, 1), on(&ctx, 200, 2));
    let error = compiled.call(&[&a]).unwrap_err();
    assert!(matches!(trace_error(&error), TraceError::Signature { expected, .. } if expected.is_empty()), "{error}");
    compiled.precompile(&[&a]).unwrap();
    assert_eq!(compiled.stats().traces, 1);
    assert_eq!(compiled.call1::<f32>(&[&a]).unwrap().len(), 100);
    let error = compiled.call(&[&b]).unwrap_err();
    let TraceError::Signature { expected, got } = trace_error(&error) else { panic!("{error}") };
    assert_eq!((expected.as_slice(), got.as_str()), (&["f32 [100, 1, 1, 1]".to_string()][..], "f32 [200, 1, 1, 1]"));
    let text = error.to_string();
    assert!(text.contains("f32 [100, 1, 1, 1]") && text.contains("f32 [200, 1, 1, 1]"), "{text}");
    assert_eq!(compiled.stats().traces, 1, "nothing was traced for the unknown signature");
}

#[test]
fn max_variants_evicts_the_least_recently_called() {
    let ctx = ctx();
    let compiled = trace_with(TraceOptions::new().max_variants(2), |inp| (&inp.get::<f32>(0) + 1.0).into());
    let (a, b, c) = (on(&ctx, 10, 1), on(&ctx, 20, 2), on(&ctx, 30, 3));
    for x in [&a, &b, &a, &c] {
        compiled.call1::<f32>(&[x]).unwrap();
    }
    let shapes: Vec<Shape> = compiled.variants().iter().map(|v| v.shapes[0]).collect();
    assert_eq!(shapes, [Shape::from(30), Shape::from(10)], "b, the least recently called, is gone");
    assert_eq!((compiled.stats().evicted, compiled.stats().variants), (1, 2));
    // Alternating over more shapes than variants retraces at every call.
    let before = compiled.stats().traces;
    for x in [&b, &a, &c, &b, &a] {
        compiled.call1::<f32>(&[x]).unwrap();
    }
    assert!(compiled.stats().traces >= before + 3, "{:?}", compiled.stats());
}

/// A function whose kernels depend on the extents of its argument (a shift and a transpose).
fn shifted(x: &Array<f32>) -> Array<f32> {
    (x.shift([1, 0]) + x.shift([0, 1]) + x.transpose().transpose() * 0.5).cache() * 2.0
}

#[test]
fn a_retrace_within_a_shape_class_builds_no_kernel() {
    let ctx = ctx();
    let compiled = trace(|inp| shifted(&inp.get::<f32>(0)).into());
    let first = on(&ctx, [1000, 1000], 1);
    compiled.call1::<f32>(&[&first]).unwrap();
    let traced = compiled.stats();
    assert!(traced.compiled_kernels > 0, "{traced:?}");
    let second = on(&ctx, [1001, 1001], 2);
    assert_eq!(ShapeClass::of(first.shape()), ShapeClass::of(second.shape()));
    let got = compiled.call1::<f32>(&[&second]).unwrap();
    assert_eq!(bits(&got), bits(&shifted(&second)));
    let stats = compiled.stats();
    assert_eq!((stats.traces, stats.retraces, stats.variants), (2, 1, 2));
    assert_eq!(stats.retrace_compiled_kernels, 0, "{stats:?}");
}

#[test]
fn a_retrace_across_a_class_builds_only_what_the_class_needs() {
    let ctx = ctx();
    let compiled = trace(|inp| shifted(&inp.get::<f32>(0)).into());
    compiled.call1::<f32>(&[&on(&ctx, [1000, 1000], 1)]).unwrap();
    let traced = compiled.stats().compiled_kernels;
    let pow2 = on(&ctx, [1024, 1024], 2);
    assert_ne!(ShapeClass::of(Shape::from([1000, 1000])), ShapeClass::of(pow2.shape()));
    let got = compiled.call1::<f32>(&[&pow2]).unwrap();
    assert_eq!(bits(&got), bits(&shifted(&pow2)));
    let across = compiled.stats().retrace_compiled_kernels;
    println!("{traced} kernels for the first trace, {across} for the retrace across the class");
    assert!(across <= traced, "a retrace needs at most what the first trace did: {across} > {traced}");
}

// -- Verification ---------------------------------------------------------------------------------

#[test]
fn verify_detects_a_function_that_changed() {
    let ctx = ctx();
    let counter = Arc::new(AtomicU32::new(0));
    let inside = Arc::clone(&counter);
    let compiled = trace_with(TraceOptions::new().verify_every(2), move |inp| {
        let k = inside.fetch_add(1, Ordering::Relaxed);
        (&inp.get::<f32>(0) + k as f32).into()
    });
    let x = f32s(&ctx, &[1.0; 8]);
    // The first call traces; the second is a verification, which traces again and finds a
    // function that adds another number now.
    compiled.call1::<f32>(&[&x]).unwrap();
    let error = compiled.call(&[&x]).unwrap_err();
    assert!(matches!(trace_error(&error), TraceError::StructureChanged), "{error}");
    assert!(compiled.variants().is_empty(), "the stale variant is dropped");
    assert_eq!(compiled.stats().verifications, 1);

    // A function that does not change passes every check.
    let steady = trace_with(TraceOptions::new().verify_every(1), |inp| (&inp.get::<f32>(0) * 2.0 + 1.0).into());
    for _ in 0..4 {
        assert_eq!(steady.call1::<f32>(&[&x]).unwrap().to_vec().unwrap(), [3.0; 8]);
    }
    assert_eq!(steady.stats().verifications, 4);
}

// -- Eager fallback and nesting ------------------------------------------------------------------

static WARNINGS: Mutex<Vec<String>> = Mutex::new(Vec::new());

struct Capture;

impl log::Log for Capture {
    fn enabled(&self, metadata: &log::Metadata<'_>) -> bool {
        metadata.level() <= log::Level::Warn
    }

    fn log(&self, record: &log::Record<'_>) {
        let text = record.args().to_string();
        if text.starts_with("luma::trace") {
            WARNINGS.lock().unwrap().push(text);
        }
    }

    fn flush(&self) {}
}

fn capture_warnings() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        let _ = log::set_logger(&Capture);
        log::set_max_level(log::LevelFilter::Warn);
    });
}

#[test]
fn eager_fallback_runs_the_function_and_warns_once() {
    capture_warnings();
    let ctx = ctx();
    let runs = Arc::new(AtomicUsize::new(0));
    let inside = Arc::clone(&runs);
    let compiled = trace_with(TraceOptions::new().eager_fallback(), move |inp| {
        let x = inp.get::<f32>(0);
        inside.fetch_add(1, Ordering::Relaxed);
        let total = x.sum().item().unwrap_or(f32::NAN);
        (&x + total).into()
    });
    let x = f32s(&ctx, &[1.0, 2.0, 3.0, 4.0]);
    let first = compiled.call1::<f32>(&[&x]).unwrap();
    assert_eq!(first.to_vec().unwrap(), [11.0, 12.0, 13.0, 14.0], "it read the sum of the real argument");
    let second = compiled.call1::<f32>(&[&f32s(&ctx, &[2.0; 8])]).unwrap();
    assert_eq!(second.to_vec().unwrap(), [18.0; 8]);
    let third = compiled.call1::<f32>(&[&x]).unwrap();
    assert_eq!(third.to_vec().unwrap(), [11.0, 12.0, 13.0, 14.0]);
    let stats = compiled.stats();
    assert_eq!((stats.eager_calls, stats.variants), (3, 0), "{stats:?}");
    // Each new signature runs the function once to trace it (which fails) and once eagerly; a
    // signature that is marked runs it eagerly only.
    assert_eq!(runs.load(Ordering::Relaxed), 2 + 2 + 1, "two failed traces and three eager runs");
    let warnings = WARNINGS.lock().unwrap().iter().filter(|w| w.contains("readback")).count();
    assert_eq!(warnings, 1, "one warning per callable");
}

#[test]
fn a_compiled_call_inside_iterate_all_is_refused_and_the_recording_survives() {
    let ctx = ctx();
    let compiled = trace(|inp| (inp.get::<f32>(0).shift([1]) * 0.5).into());
    let xs = values(512, 40, 0.0, 1.0);
    let x = f32s(&ctx, &xs);
    let refused = Mutex::new(Vec::new());
    let [out] = iterate_all([&x], 4, |[u]| {
        if let Err(error) = compiled.call1::<f32>(&[u]) {
            refused.lock().unwrap().push(error);
        }
        // The function itself, in the loop: the recording holds it.
        [u.shift([1]) * 0.5]
    });
    let refused = refused.into_inner().unwrap();
    assert!(!refused.is_empty());
    assert!(refused.iter().all(|error| matches!(trace_error(error), TraceError::Nested)), "{refused:?}");
    let mut want = xs;
    for _ in 0..4 {
        want = (0..want.len()).map(|i| want[(i + want.len() - 1) % want.len()] * 0.5).collect();
    }
    assert_close(&out.to_vec().unwrap(), &want, 1e-6, 1e-6);
    assert_eq!(compiled.stats().traces, 0, "nothing was traced inside the loop");
    // Outside the loop the callable works.
    assert_eq!(compiled.call1::<f32>(&[&x]).unwrap().len(), 512);
}

#[test]
fn a_trace_inside_another_trace_is_nested() {
    let ctx = ctx();
    let inner = trace(|inp| (&inp.get::<f32>(0) + 1.0).into());
    let seen = Arc::new(Mutex::new(Vec::new()));
    let log = Arc::clone(&seen);
    let outer = trace(move |inp| {
        let x = inp.get::<f32>(0);
        if let Err(error) = inner.call(&[&x]) {
            log.lock().unwrap().push(error);
        }
        (&x * 2.0).into()
    });
    let x = on(&ctx, 64, 41);
    assert_eq!(outer.call1::<f32>(&[&x]).unwrap().len(), 64);
    let seen = seen.lock().unwrap();
    assert_eq!(seen.len(), 1);
    assert!(matches!(trace_error(&seen[0]), TraceError::Nested), "{:?}", seen[0]);
}

#[test]
fn eager_fallback_inside_a_recording_runs_the_function_once() {
    capture_warnings();
    let ctx = ctx();
    let runs = Arc::new(AtomicUsize::new(0));
    let inside = Arc::clone(&runs);
    let compiled = trace_with(TraceOptions::new().eager_fallback(), move |inp| {
        inside.fetch_add(1, Ordering::Relaxed);
        (inp.get::<f32>(0).shift([1]) * 0.5).into()
    });
    let xs = values(256, 42, 0.0, 1.0);
    let x = f32s(&ctx, &xs);
    let [out] = iterate_all([&x], 3, |[u]| [compiled.call1::<f32>(&[u]).unwrap()]);
    let mut want = xs;
    for _ in 0..3 {
        want = (0..want.len()).map(|i| want[(i + want.len() - 1) % want.len()] * 0.5).collect();
    }
    assert_close(&out.to_vec().unwrap(), &want, 1e-6, 1e-6);
    let stats = compiled.stats();
    assert_eq!(runs.load(Ordering::Relaxed) as u64, stats.eager_calls, "one run of the function per call: {stats:?}");
    assert!(stats.eager_calls >= 1 && stats.variants == 0, "{stats:?}");
    assert_eq!(WARNINGS.lock().unwrap().iter().filter(|w| w.contains("inside a recording")).count(), 1);
}

// -- Threads -------------------------------------------------------------------------------------

#[test]
fn calls_from_several_threads_share_one_compiled() {
    let ctx = ctx();
    let compiled = trace(|inp| stencil(&inp.get::<f32>(0)).into());
    std::thread::scope(|scope| {
        for t in 0..4u64 {
            let (ctx, compiled) = (&ctx, &compiled);
            scope.spawn(move || {
                for round in 0..12 {
                    let x = on(ctx, 1000, 100 + t * 20 + round);
                    let got = compiled.call1::<f32>(&[&x]).unwrap();
                    assert_eq!(bits(&got), bits(&stencil(&x)), "thread {t} round {round}");
                }
            });
        }
    });
    assert_eq!(compiled.stats().variants, 1);
    assert_eq!(compiled.stats().calls, 48);
}

#[test]
fn two_traces_of_different_signatures_do_not_block_each_other() {
    let ctx = ctx();
    let gate = Arc::new((Mutex::new(0usize), Condvar::new()));
    let inside = Arc::clone(&gate);
    let compiled = trace(move |inp| {
        // Both traces are inside the function at once, or this gives up.
        let (count, changed) = &*inside;
        let mut count = count.lock().unwrap();
        *count += 1;
        changed.notify_all();
        let (count, timeout) = changed.wait_timeout_while(count, Duration::from_secs(30), |n| *n < 2).unwrap();
        assert!(!timeout.timed_out(), "the second trace never started: traces block each other");
        drop(count);
        (&inp.get::<f32>(0) + 1.0).into()
    });
    std::thread::scope(|scope| {
        for n in [100usize, 200] {
            let (ctx, compiled) = (&ctx, &compiled);
            scope.spawn(move || {
                let x = on(ctx, n, n as u64);
                assert_eq!(compiled.call1::<f32>(&[&x]).unwrap().len(), n);
            });
        }
    });
    assert_eq!(compiled.stats().variants, 2);
}

// -- Memory --------------------------------------------------------------------------------------

#[test]
fn a_compiled_counts_and_releases_its_buffers() {
    let ctx = ctx();
    let len = 1 << 18;
    let bytes = 4 * len as u64;
    let (x, w) = (on(&ctx, len, 50), on(&ctx, len, 51));
    ctx.sync().unwrap();
    let before = ctx.memory_stats().resident_bytes;
    let captured = w.clone();
    let compiled = trace(move |inp| ((&inp.get::<f32>(0) * &captured).cache() + 1.0).into());
    let out = compiled.call1::<f32>(&[&x]).unwrap();
    out.eval().unwrap();
    ctx.sync().unwrap();
    let stats = compiled.stats();
    assert!(stats.pinned_bytes >= bytes, "the captured weight is pinned: {stats:?}");
    assert!(stats.scratch_bytes >= bytes, "the cached intermediate is the plan's: {stats:?}");
    let during = ctx.memory_stats().resident_bytes;
    assert!(during >= before + bytes + stats.scratch_bytes, "the result and the plan's buffers count: {before} -> {during}, {stats:?}");
    drop(out);
    compiled.clear();
    let cleared = compiled.stats();
    assert_eq!((cleared.variants, cleared.pinned_bytes, cleared.scratch_bytes), (0, 0, 0));
    assert_eq!(ctx.memory_stats().resident_bytes, before, "the plan's buffers are released");
    // Dropping the callable releases them too.
    let captured = w.clone();
    let compiled = trace(move |inp| ((&inp.get::<f32>(0) * &captured).cache() + 1.0).into());
    drop(compiled.call1::<f32>(&[&x]).unwrap());
    drop(compiled);
    assert_eq!(ctx.memory_stats().resident_bytes, before);
}

#[test]
fn tracing_does_not_leave_full_size_probes_behind() {
    let ctx = ctx();
    let x = on(&ctx, 1 << 20, 52);
    ctx.sync().unwrap();
    let before = ctx.memory_stats().resident_bytes;
    let compiled = trace(|inp| (&inp.get::<f32>(0) * 2.0).sum().into());
    let total = compiled.call1::<f32>(&[&x]).unwrap();
    total.eval().unwrap();
    ctx.sync().unwrap();
    let after = ctx.memory_stats().resident_bytes;
    assert!(after < before + (1 << 16), "the 4 MB probe is gone: {before} -> {after}");
}

// -- Decisions -----------------------------------------------------------------------------------

#[test]
fn a_trace_makes_the_decisions_of_the_eager_run() {
    let ctx = ctx();
    let x = on(&ctx, 1 << 16, 53);
    let logged = |run: &dyn Fn()| {
        ctx.record_decisions(true);
        run();
        ctx.sync().unwrap();
        let mut decisions: Vec<String> = ctx.take_decisions().iter().map(|d| format!("{:?} {} {}", d.site, d.op, d.chosen)).collect();
        ctx.record_decisions(false);
        decisions.sort();
        decisions
    };
    let eager = logged(&|| stencil(&x).eval().unwrap());
    let compiled = trace(|inp| stencil(&inp.get::<f32>(0)).into());
    let traced = logged(&|| {
        compiled.call1::<f32>(&[&x]).unwrap().eval().unwrap();
    });
    assert!(!eager.is_empty());
    assert_eq!(traced, eager);
}
