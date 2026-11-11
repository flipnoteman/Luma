// wgpu's handle types nest deeply; see tests/batching.rs.
#![recursion_limit = "256"]

//! Elementwise operations are recorded lazily and fused into one kernel. These tests check what
//! the user can observe: the values of fused expressions against CPU references, errors and their
//! locations surfacing from the middle of lazy chains, and the semantics of lazy arrays when the
//! buffers they read are overwritten afterwards.
//!
//! The semantics: *a lazy array is the value its expression had when it was written down.* Its
//! inputs are snapshotted: `write`, compound assignment and kernels taking `&mut` on an array
//! the expression reads do not change the result of the expression.
//!
//! Each test that depends on pool reuse uses its own `Context`.

mod common;

use common::{assert_close, values};
use luma::*;

fn f32s_on(ctx: &Context, data: &[f32]) -> Array<f32> {
    Array::from_slice_on(ctx, data, data.len()).unwrap()
}

// -- Values --

#[test]
fn chains_of_every_length_match_the_cpu() {
    let ctx = Context::new().unwrap();
    let n = 10_001;
    let xs = values(n, 1, 0.5, 2.0);
    let ys = values(n, 2, 0.5, 2.0);
    let x = f32s_on(&ctx, &xs);
    let y = f32s_on(&ctx, &ys);

    for ops in [2usize, 5, 10, 20, 64, 65, 150, 400] {
        let mut acc = &x * 1.0;
        let mut want = xs.clone();
        for step in 0..ops {
            match step % 5 {
                0 => {
                    acc = &acc + &y;
                    for (w, b) in want.iter_mut().zip(&ys) {
                        *w += b;
                    }
                }
                1 => {
                    acc *= 0.99;
                    want.iter_mut().for_each(|w| *w *= 0.99);
                }
                2 => {
                    acc = acc.sqrt();
                    want.iter_mut().for_each(|w| *w = w.sqrt());
                }
                3 => {
                    acc = 3.0 - &acc;
                    want.iter_mut().for_each(|w| *w = 3.0 - *w);
                }
                _ => {
                    acc = acc.abs().maximum(0.5);
                    want.iter_mut().for_each(|w| *w = w.abs().max(0.5));
                }
            }
        }
        assert_close(&acc.to_vec().unwrap(), &want, 1e-4, 1e-4);
    }
}

#[test]
fn every_elementwise_operation_in_one_expression() {
    let ctx = Context::new().unwrap();
    let xs = values(2000, 3, 0.2, 2.0);
    let ys = values(2000, 4, 0.2, 2.0);
    let x = f32s_on(&ctx, &xs);
    let y = f32s_on(&ctx, &ys);

    let z = ((&x + &y) * 0.5 - &y / &x).abs().sqrt().exp().ln().abs()
        + (&x - 1.0).sin() * (&y - 1.0).cos()
        + (&x * 0.3).tan()
        + x.powf(1.5).clamp(0.1, 2.0)
        + x.minimum(&y).maximum(0.3)
        + (-&y);
    let want: Vec<f32> = xs
        .iter()
        .zip(&ys)
        .map(|(&x, &y)| {
            ((x + y) * 0.5 - y / x).abs().sqrt().exp().ln().abs()
                + (x - 1.0).sin() * (y - 1.0).cos()
                + (x * 0.3).tan()
                + x.powf(1.5).clamp(0.1, 2.0)
                + x.min(y).max(0.3)
                + (-y)
        })
        .collect();
    assert_close(&z.to_vec().unwrap(), &want, 1e-3, 1e-3);
}

#[test]
fn comparison_into_select_in_chains() {
    let ctx = Context::new().unwrap();
    let xs = values(4096, 5, -2.0, 2.0);
    let ys = values(4096, 6, -2.0, 2.0);
    let x = f32s_on(&ctx, &xs);
    let y = f32s_on(&ctx, &ys);

    // relu-like branches, with masks computed from expressions and combined.
    let scaled = &x * 2.0 - &y;
    let mask = scaled.gt(0.0);
    let picked = select(&mask, &(&x * &x), &(&y.abs() + 1.0));
    let again = select(&picked.lt(2.0), &picked, &(&picked * 0.5)).minimum(1.5);
    let want: Vec<f32> = xs
        .iter()
        .zip(&ys)
        .map(|(&x, &y)| {
            let picked = if x * 2.0 - y > 0.0 { x * x } else { y.abs() + 1.0 };
            (if picked < 2.0 { picked } else { picked * 0.5 }).min(1.5)
        })
        .collect();
    assert_close(&again.to_vec().unwrap(), &want, 1e-5, 1e-5);

    // The mask itself, all six comparisons, against scalars and arrays.
    let masks = [x.lt(&y), x.le(&y), x.gt(0.5), x.ge(0.5), x.eq(&x), x.ne(&y)];
    let want_masks: Vec<Vec<u32>> = vec![
        xs.iter().zip(&ys).map(|(a, b)| u32::from(a < b)).collect(),
        xs.iter().zip(&ys).map(|(a, b)| u32::from(a <= b)).collect(),
        xs.iter().map(|&a| u32::from(a > 0.5)).collect(),
        xs.iter().map(|&a| u32::from(a >= 0.5)).collect(),
        vec![1; xs.len()],
        xs.iter().zip(&ys).map(|(a, b)| u32::from(a != b)).collect(),
    ];
    for (mask, want) in masks.iter().zip(&want_masks) {
        assert_eq!(&mask.to_vec().unwrap(), want);
    }
}

#[test]
fn u32_chains() {
    let ctx = Context::new().unwrap();
    let data: Vec<u32> = (0..5000).map(|i| (i * 2654435761u64 % 1000) as u32).collect();
    let a = Array::from_slice_on(&ctx, &data, data.len()).unwrap();
    let b = (&a * 3u32 + 7u32 - 1u32) / 2u32;
    let c = b.minimum(500u32).maximum(&a);
    let got = (&c + &a).to_vec().unwrap();
    let want: Vec<u32> =
        data.iter().map(|&v| ((v * 3 + 7 - 1) / 2).min(500).max(v) + v).collect();
    assert_eq!(got, want);
}

#[test]
fn reductions_of_lazy_expressions() {
    let ctx = Context::new().unwrap();
    let n = 70_001;
    let xs = values(n, 7, -1.0, 1.0);
    let ys = values(n, 8, -1.0, 1.0);
    let x = f32s_on(&ctx, &xs);
    let y = f32s_on(&ctx, &ys);

    let dot = (&x * &y).sum().unwrap();
    let want: f64 = xs.iter().zip(&ys).map(|(&a, &b)| f64::from(a) * f64::from(b)).sum();
    assert!((f64::from(dot) - want).abs() < 1e-2, "{dot} vs {want}");

    let lazy = &x * 2.0 + &y;
    let (min, max) = (lazy.min().unwrap(), lazy.max().unwrap());
    let want_min = xs.iter().zip(&ys).map(|(&a, &b)| a * 2.0 + b).fold(f32::INFINITY, f32::min);
    let want_max = xs.iter().zip(&ys).map(|(&a, &b)| a * 2.0 + b).fold(f32::NEG_INFINITY, f32::max);
    assert_eq!((min, max), (want_min, want_max));
    // ... and the expression is still good for more than reducing.
    let want: Vec<f32> = xs.iter().zip(&ys).map(|(&a, &b)| a * 2.0 + b).collect();
    assert_close(&lazy.to_vec().unwrap(), &want, 1e-6, 1e-6);

    let positives = x.gt(0.0).sum().unwrap();
    assert_eq!(positives as usize, xs.iter().filter(|&&v| v > 0.0).count());
}

// -- Errors from the middle of lazy chains --

#[test]
fn a_faulty_operation_inside_a_lazy_chain_reports_its_own_location() {
    let a = Array::from_slice(&[1.0f32, 2.0, 3.0], 3).unwrap();
    let b = Array::from_slice(&[1.0f32, 2.0], 2).unwrap();

    // Everything before and after the faulty line is lazy.
    let healthy = (&a * 2.0).sqrt() + 1.0;
    let line = line!() + 1;
    let broken = &healthy - &b;
    let later = (broken.abs() * 3.0).exp().clamp(0.0, 1.0);
    let sum_of_later = &later + &a;

    for array in [&broken, &later, &sum_of_later] {
        let err = array.to_vec().unwrap_err();
        assert!(matches!(err.root(), LumaError::ShapeMismatch { lhs: [3, 1, 1, 1], rhs: [2, 1, 1, 1] }), "{err}");
        let location = err.location().expect("location");
        assert!(location.file().ends_with("fusion.rs"), "{}", location.file());
        assert_eq!(location.line(), line);
        assert!(err.to_string().contains("sub"), "{err}");
    }
    assert_eq!(later.error().unwrap().location().unwrap().line(), line);
    // Reductions and eval report it too.
    assert!(sum_of_later.sum().is_err());
    assert!(sum_of_later.norm().is_err());
    assert!(sum_of_later.eval().is_err());
    // The healthy part is unharmed, and was never poisoned by the later failure.
    let want: Vec<f32> = [1.0f32, 2.0, 3.0].iter().map(|v| (v * 2.0).sqrt() + 1.0).collect();
    assert_close(&healthy.to_vec().unwrap(), &want, 1e-6, 1e-6);
}

#[test]
fn select_and_comparison_errors_inside_lazy_chains() {
    let a = Array::from_slice(&[1.0f32, 2.0, 3.0], 3).unwrap();
    let b = Array::from_slice(&[1.0f32, 2.0], 2).unwrap();
    let mask = (&a * 2.0).gt(1.0);
    let line = line!() + 1;
    let bad = select(&mask, &(&a + 1.0), &b);
    let after = bad * 2.0;
    let err = after.to_vec().unwrap_err();
    assert_eq!(err.location().unwrap().line(), line);
    assert!(err.to_string().contains("select"), "{err}");

    // A comparison between different shapes.
    let line = line!() + 1;
    let bad_mask = (&a + 1.0).lt(&b);
    let err = select(&bad_mask, &a, &a).to_vec().unwrap_err();
    assert_eq!(err.location().unwrap().line(), line);
}

#[test]
fn compound_assignment_on_a_lazy_array_reports_errors_like_before() {
    let a = Array::from_slice(&[1.0f32, 2.0, 3.0], 3).unwrap();
    let b = Array::from_slice(&[1.0f32, 2.0], 2).unwrap();

    let mut y = &a * 2.0; // lazy
    y += 1.0;
    let line = line!() + 1;
    y -= &b;
    let err = y.error().expect("poisoned at the failing operator");
    assert_eq!(err.location().unwrap().line(), line);
    assert!(err.to_string().contains("sub_assign"), "{err}");
    y *= 2.0;
    assert_eq!(y.to_vec().unwrap_err().location().unwrap().line(), line);

    // A poisoned right-hand side poisons a lazy left-hand side.
    let mut x = &a * 2.0;
    let bad = &a + &b;
    x += &bad;
    assert!(x.error().is_some());
}

#[test]
fn try_methods_on_lazy_arrays_fail_immediately() {
    let a = Array::from_slice(&[1.0f32, 2.0, 3.0], 3).unwrap();
    let b = Array::from_slice(&[1.0f32, 2.0], 2).unwrap();
    let lazy = &a * 2.0;
    let err = lazy.try_add(&b).unwrap_err();
    assert!(matches!(err, LumaError::ShapeMismatch { .. }), "{err}");
    assert!(err.location().is_none());
    assert_eq!(lazy.try_add(1.0).unwrap().to_vec().unwrap(), vec![3.0, 5.0, 7.0]);
    assert_eq!(lazy.try_sub(&a).unwrap().to_vec().unwrap(), vec![1.0, 2.0, 3.0]);
}

#[test]
fn lazy_arrays_of_different_contexts_cannot_be_combined() {
    let one = Context::new().unwrap();
    let two = Context::new().unwrap();
    let a = &f32s_on(&one, &[1.0, 2.0]) * 2.0;
    let b = &f32s_on(&two, &[1.0, 2.0]) * 2.0;
    let err = (&a + &b).to_vec().unwrap_err();
    assert!(matches!(err.root(), LumaError::ContextMismatch), "{err}");
    assert_eq!(a.to_vec().unwrap(), vec![2.0, 4.0]);
}

// -- Hazards: buffers overwritten after lazy arrays were recorded against them --

#[test]
fn write_does_not_change_pending_expressions() {
    let ctx = Context::new().unwrap();
    let old: Vec<f32> = (0..300).map(|i| i as f32).collect();
    let mut x = f32s_on(&ctx, &old);

    let direct = &x * 2.0 + 1.0;
    let nested = (&direct + 3.0).sqrt().abs(); // an expression on top of an expression
    let sum_view = &x + &x;

    x.write(&vec![-5.0; 300]).unwrap();
    let after = &x * 2.0 + 1.0; // recorded after the write: sees the new data

    assert_eq!(direct.to_vec().unwrap(), old.iter().map(|v| v * 2.0 + 1.0).collect::<Vec<_>>());
    assert_close(
        &nested.to_vec().unwrap(),
        &old.iter().map(|v| (v * 2.0 + 1.0 + 3.0).sqrt().abs()).collect::<Vec<_>>(),
        1e-5,
        1e-5,
    );
    assert_eq!(sum_view.to_vec().unwrap(), old.iter().map(|v| v + v).collect::<Vec<_>>());
    assert_eq!(after.to_vec().unwrap(), vec![-9.0; 300]);
}

#[test]
fn many_overwrites_between_many_pending_expressions() {
    let ctx = Context::new().unwrap();
    let mut x = f32s_on(&ctx, &[0.0; 100]);
    let mut outputs = Vec::new();
    for i in 0..80 {
        x.write(&vec![i as f32; 100]).unwrap();
        outputs.push((&x * 10.0 + 1.0).sqrt());
        x += 1.0; // in place, after the expression above was recorded
        outputs.push(&x - 0.5);
    }
    for i in 0..80 {
        assert_close(
            &outputs[2 * i].to_vec().unwrap(),
            &vec![(i as f32 * 10.0 + 1.0).sqrt(); 100],
            1e-5,
            1e-5,
        );
        assert_eq!(outputs[2 * i + 1].to_vec().unwrap(), vec![i as f32 + 0.5; 100]);
    }
}

#[test]
fn compound_assignment_does_not_change_pending_expressions() {
    let ctx = Context::new().unwrap();
    let mut x = f32s_on(&ctx, &[1.0, 2.0, 3.0, 4.0]);
    let before = &x * 10.0;
    let before_nested = &before + 1.0;
    x += 100.0;
    x *= &f32s_on(&ctx, &[1.0, 2.0, 3.0, 4.0]);
    let after = &x * 10.0;
    assert_eq!(before.to_vec().unwrap(), vec![10.0, 20.0, 30.0, 40.0]);
    assert_eq!(before_nested.to_vec().unwrap(), vec![11.0, 21.0, 31.0, 41.0]);
    assert_eq!(x.to_vec().unwrap(), vec![101.0, 204.0, 309.0, 416.0]);
    assert_eq!(after.to_vec().unwrap(), vec![1010.0, 2040.0, 3090.0, 4160.0]);

    // The right-hand side reads the array that is being updated.
    let mut y = f32s_on(&ctx, &[1.0, 2.0, 3.0]);
    let doubled = &y * 2.0;
    y += &doubled;
    assert_eq!(y.to_vec().unwrap(), vec![3.0, 6.0, 9.0]);
    assert_eq!(doubled.to_vec().unwrap(), vec![2.0, 4.0, 6.0]);
}

#[test]
fn views_updated_in_place_do_not_change_pending_expressions() {
    let ctx = Context::new().unwrap();
    let x = f32s_on(&ctx, &[1.0, 2.0, 3.0, 4.0]);
    let pending = &x * 2.0;
    let mut view = x.reshape([2, 2]);
    view += 10.0; // shares x's buffer
    assert_eq!(pending.to_vec().unwrap(), vec![2.0, 4.0, 6.0, 8.0]);
    assert_eq!(x.to_vec().unwrap(), vec![11.0, 12.0, 13.0, 14.0]);
    // The view of a lazy array is evaluated first and then shares its buffer.
    let lazy = &x * 2.0;
    let mut lazy_view = lazy.reshape([4]);
    lazy_view += 1.0;
    assert_eq!(lazy.to_vec().unwrap(), vec![23.0, 25.0, 27.0, 29.0]);
}

const TRIPLE: &str = r#"
@group(1) @binding(0) var<storage, read_write> data: array<f32>;
@compute @workgroup_size(32)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x < luma.size.x) { data[gid.x] = data[gid.x] * 3.0; }
}"#;

const COPY: &str = r#"
@group(1) @binding(0) var<storage, read> src: array<f32>;
@group(1) @binding(1) var<storage, read_write> dst: array<f32>;
@compute @workgroup_size(32)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x < luma.size.x) { dst[gid.x] = src[gid.x]; }
}"#;

#[test]
fn custom_kernels_writing_an_array_do_not_change_pending_expressions() {
    let ctx = Context::new().unwrap();
    let triple = ctx.kernel(TRIPLE).build().unwrap();
    let mut x = f32s_on(&ctx, &[1.0, 2.0, 3.0, 4.0]);
    let pending = (&x + 1.0) * 2.0;
    let nested = pending.sqrt();
    triple.launch([4], args![&mut x]).unwrap();
    let after = &x + 1.0;
    assert_eq!(pending.to_vec().unwrap(), vec![4.0, 6.0, 8.0, 10.0]);
    assert_close(&nested.to_vec().unwrap(), &[2.0, 6.0f32.sqrt(), 8.0f32.sqrt(), 10.0f32.sqrt()], 1e-6, 1e-6);
    assert_eq!(x.to_vec().unwrap(), vec![3.0, 6.0, 9.0, 12.0]);
    assert_eq!(after.to_vec().unwrap(), vec![4.0, 7.0, 10.0, 13.0]);
}

#[test]
fn lazy_arrays_as_kernel_arguments() {
    let ctx = Context::new().unwrap();
    let copy = ctx.kernel(COPY).build().unwrap();
    let triple = ctx.kernel(TRIPLE).build().unwrap();
    let x = f32s_on(&ctx, &[1.0, 2.0, 3.0, 4.0]);

    // Read-only: the expression is evaluated when it is passed.
    let src = &x * 2.0;
    let mut dst = Array::<f32>::zeros_on(&ctx, 4).unwrap();
    copy.launch([4], args![&src, &mut dst]).unwrap();
    assert_eq!(dst.to_vec().unwrap(), vec![2.0, 4.0, 6.0, 8.0]);

    // Read_write: the kernel updates the evaluated buffer; arrays derived from the old value
    // keep it.
    let mut target = &x + 1.0;
    let snapshot = &target * 1.0;
    triple.launch([4], args![&mut target]).unwrap();
    assert_eq!(target.to_vec().unwrap(), vec![6.0, 9.0, 12.0, 15.0]);
    assert_eq!(snapshot.to_vec().unwrap(), vec![2.0, 3.0, 4.0, 5.0]);
    let later = &target + 0.0;
    assert_eq!(later.to_vec().unwrap(), vec![6.0, 9.0, 12.0, 15.0]);
}

#[test]
fn writing_a_lazy_array_does_not_change_expressions_built_on_it() {
    let ctx = Context::new().unwrap();
    let x = f32s_on(&ctx, &[1.0, 2.0, 3.0]);
    let mut t = &x * 2.0; // lazy
    let built_on_t = &t + 1.0;
    t.write(&[0.0, 0.0, 0.0]).unwrap();
    assert_eq!(built_on_t.to_vec().unwrap(), vec![3.0, 5.0, 7.0]);
    assert_eq!(t.to_vec().unwrap(), vec![0.0, 0.0, 0.0]);
    assert_eq!(x.to_vec().unwrap(), vec![1.0, 2.0, 3.0]);
}

#[test]
fn inputs_of_lazy_arrays_survive_being_dropped() {
    let ctx = Context::new().unwrap();
    let old: Vec<f32> = (0..1000).map(|i| i as f32).collect();
    let x = f32s_on(&ctx, &old);
    let pending = &x * 2.0 - 1.0;
    drop(x);
    // These would take the buffer of `x` if it had been recycled.
    let others: Vec<Array<f32>> = (0..6).map(|i| f32s_on(&ctx, &vec![-(i as f32); 1000])).collect();
    let written = others.iter().map(|o| o * 2.0).collect::<Vec<_>>();
    assert_eq!(pending.to_vec().unwrap(), old.iter().map(|v| v * 2.0 - 1.0).collect::<Vec<_>>());
    for (i, w) in written.iter().enumerate() {
        assert_eq!(w.to_vec().unwrap(), vec![-2.0 * i as f32; 1000]);
    }
}

#[test]
fn sync_runs_lazy_arrays_that_are_alive() {
    let ctx = Context::new().unwrap();
    let x = f32s_on(&ctx, &[4.0, 9.0]);
    let y = x.sqrt() + 1.0;
    ctx.sync().unwrap();
    assert_eq!(y.to_vec().unwrap(), vec![3.0, 4.0]);
    y.eval().unwrap();
}

#[test]
fn shifts_and_convolutions_take_lazy_arrays() {
    let ctx = Context::new().unwrap();
    let data = values(64, 20, -1.0, 1.0);
    let x = f32s_on(&ctx, &data);
    let lazy = &x * 2.0 + 1.0;
    let shifted = lazy.shift(3);
    let n = data.len();
    let want: Vec<f32> = (0..n).map(|i| data[(i + n - 3) % n] * 2.0 + 1.0).collect();
    assert_close(&shifted.to_vec().unwrap(), &want, 1e-6, 1e-6);

    let kernel = f32s_on(&ctx, &[0.5, 0.25, 0.25]);
    let conv = (&x * 2.0).conv1d(&kernel, 1, 0);
    let scaled: Vec<f32> = data.iter().map(|v| v * 2.0).collect();
    let want: Vec<f32> =
        (0..n - 2).map(|i| scaled[i] * 0.5 + scaled[i + 1] * 0.25 + scaled[i + 2] * 0.25).collect();
    assert_close(&conv.to_vec().unwrap(), &want, 1e-5, 1e-5);
    // A stencil: lazy expressions on both sides of a shift.
    let stencil = (&lazy.shift(1) + &lazy.shift(-1) - &lazy * 2.0) * 0.5;
    let want: Vec<f32> = (0..n)
        .map(|i| {
            let f = |j: usize| data[j % n] * 2.0 + 1.0;
            (f(i + n - 1) + f(i + 1) - f(i) * 2.0) * 0.5
        })
        .collect();
    assert_close(&stencil.to_vec().unwrap(), &want, 1e-5, 1e-5);
}

#[test]
fn lazy_arrays_from_many_threads() {
    let ctx = Context::new().unwrap();
    let x = f32s_on(&ctx, &values(512, 30, 0.0, 1.0));
    let xs = x.to_vec().unwrap();
    std::thread::scope(|scope| {
        for t in 0..4 {
            let (x, xs) = (&x, &xs);
            scope.spawn(move || {
                for i in 0..50 {
                    let s = (t * 100 + i) as f32;
                    let y = (x * s + 1.0).sqrt();
                    let want: Vec<f32> = xs.iter().map(|v| (v * s + 1.0).sqrt()).collect();
                    assert_close(&y.to_vec().unwrap(), &want, 1e-4, 1e-4);
                }
            });
        }
    });
}

#[test]
fn iterate_matches_the_step_by_step_loop() {
    let ctx = Context::new().unwrap();
    // 2-D five-point heat step with a fixed source term, on a grid that is not a multiple of
    // the tile, for more steps than one launch runs.
    let (w, h) = (77usize, 45usize);
    let data = values(w * h, 60, 0.0, 1.0);
    let source = values(w * h, 61, -0.01, 0.01);
    let u0 = Array::from_slice_on(&ctx, &data, [w, h]).unwrap();
    let s = Array::from_slice_on(&ctx, &source, [w, h]).unwrap();
    let step = |u: &Array<f32>| {
        let lap = u.shift([1isize, 0]) + u.shift([-1isize, 0]) + u.shift([0isize, 1]) + u.shift([0isize, -1]) - u * 4.0;
        u + lap * 0.2 + &s
    };
    let mut want = data.clone();
    for _ in 0..23 {
        let at = |x: isize, y: isize| want[x.rem_euclid(w as isize) as usize + w * y.rem_euclid(h as isize) as usize];
        want = (0..w * h)
            .map(|i| {
                let (x, y) = ((i % w) as isize, (i / w) as isize);
                let lap = at(x - 1, y) + at(x + 1, y) + at(x, y - 1) + at(x, y + 1) - 4.0 * at(x, y);
                at(x, y) + lap * 0.2 + source[i]
            })
            .collect();
    }
    let got = u0.iterate(23, step).to_vec().unwrap();
    assert_close(&got, &want, 1e-4, 1e-4);
    // The start is unchanged.
    assert_eq!(u0.to_vec().unwrap(), data);

    // 1-D, radius 2, one step only.
    let xs = values(5000, 62, -1.0, 1.0);
    let x = Array::from_slice_on(&ctx, &xs, xs.len()).unwrap();
    let got = x.iterate(1, |u| (u.shift(2) + u.shift(-2)) * 0.5 - u * 0.1).to_vec().unwrap();
    let n = xs.len();
    let want: Vec<f32> = (0..n).map(|i| (xs[(i + n - 2) % n] + xs[(i + 2) % n]) * 0.5 - xs[i] * 0.1).collect();
    assert_close(&got, &want, 1e-6, 1e-6);

    // A step that is not a stencil (a convolution) still iterates correctly.
    let kernel = Array::from_slice_on(&ctx, &[0.25f32, 0.5, 0.25], 3).unwrap();
    let got = x.iterate(2, |u| u.conv1d(&kernel, 1, 1)).to_vec().unwrap();
    let once: Vec<f32> = (0..n)
        .map(|i| 0.25 * if i > 0 { xs[i - 1] } else { 0.0 } + 0.5 * xs[i] + 0.25 * xs.get(i + 1).copied().unwrap_or(0.0))
        .collect();
    let twice: Vec<f32> = (0..n)
        .map(|i| 0.25 * if i > 0 { once[i - 1] } else { 0.0 } + 0.5 * once[i] + 0.25 * once.get(i + 1).copied().unwrap_or(0.0))
        .collect();
    assert_close(&got, &twice, 1e-5, 1e-5);

    // A step that materializes an intermediate result on `u` before combining with `u.shift`
    // must not treat that intermediate buffer as a fixed stencil input.
    let small = Array::from_slice_on(&ctx, &[1.0f32, 2.0, 3.0, 4.0], 4).unwrap();
    let got = small
        .iterate(3, |u| u.shift(1) + u.try_add(1.0).unwrap())
        .to_vec()
        .unwrap();
    let mut want = vec![1.0f32, 2.0, 3.0, 4.0];
    for _ in 0..3 {
        want = (0..4).map(|i| want[(i + 3) % 4] + (want[i] + 1.0)).collect();
    }
    assert_close(&got, &want, 1e-6, 1e-6);
}
