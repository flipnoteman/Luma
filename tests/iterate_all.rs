//! `iterate_all` runs pointwise multi-array loops as one kernel and anything else step by step.

mod common;

use common::{assert_close, values};
use luma::*;

fn f32s_on(ctx: &Context, data: &[f32]) -> Array<f32> {
    Array::from_slice_on(ctx, data, data.len()).unwrap()
}

fn mandelbrot_step(cx: &Array<f32>, cy: &Array<f32>, [zx, zy, count]: &[Array<f32>; 3]) -> [Array<f32>; 3] {
    let inside = (zx * zx + zy * zy).le(4.0);
    let nzx = zx * zx - zy * zy + cx;
    let nzy = zx * zy * 2.0 + cy;
    [select(&inside, &nzx, zx), select(&inside, &nzy, zy), count + select(&inside, 1.0, 0.0)]
}

fn mandelbrot_cpu(cx: &[f32], cy: &[f32], steps: usize) -> [Vec<f32>; 3] {
    let n = cx.len();
    let (mut zx, mut zy, mut count) = (vec![0.0f32; n], vec![0.0f32; n], vec![0.0f32; n]);
    for _ in 0..steps {
        for i in 0..n {
            if zx[i] * zx[i] + zy[i] * zy[i] <= 4.0 {
                let x = zx[i] * zx[i] - zy[i] * zy[i] + cx[i];
                zy[i] = zx[i] * zy[i] * 2.0 + cy[i];
                zx[i] = x;
                count[i] += 1.0;
            }
        }
    }
    [zx, zy, count]
}

/// Orbits near the boundary are chaotic, so rounding decides some escape counts.
fn assert_counts(got: &[f32], want: &[f32]) {
    let differ = got.iter().zip(want).filter(|(a, b)| a != b).count();
    assert!(differ * 100 <= got.len(), "{differ} of {} escape counts differ", got.len());
}

#[test]
fn a_three_state_loop_matches_the_cpu_and_the_hand_written_loop() {
    let ctx = Context::new().unwrap();
    let n = 4096;
    let (cx, cy) = (values(n, 1, -2.0, 0.5), values(n, 2, -1.2, 1.2));
    let (gcx, gcy) = (f32s_on(&ctx, &cx), f32s_on(&ctx, &cy));
    let zero = Array::<f32>::zeros_on(&ctx, n).unwrap();
    for steps in [0, 1, 7, 64] {
        let looped = iterate_all([&zero, &zero, &zero], steps, |s| mandelbrot_step(&gcx, &gcy, s));
        let mut by_hand = [&zero * 1.0, &zero * 1.0, &zero * 1.0];
        for _ in 0..steps {
            by_hand = mandelbrot_step(&gcx, &gcy, &by_hand);
        }
        let want = mandelbrot_cpu(&cx, &cy, steps);
        let count = looped[2].to_vec().unwrap();
        assert_counts(&count, &want[2]);
        assert_counts(&count, &by_hand[2].to_vec().unwrap());
    }
}

#[test]
fn the_register_loop_is_one_launch_for_any_step_count() {
    let ctx = Context::new().unwrap();
    let n = 1024;
    let (cx, cy) = (values(n, 3, -2.0, 0.5), values(n, 4, -1.2, 1.2));
    let (gcx, gcy) = (f32s_on(&ctx, &cx), f32s_on(&ctx, &cy));
    let zero = Array::<f32>::zeros_on(&ctx, n).unwrap();
    for steps in [5, 300] {
        let (states, profile) = ctx
            .profile(|| {
                let states = iterate_all([&zero, &zero, &zero], steps, |s| mandelbrot_step(&gcx, &gcy, s));
                eval_all(&states.each_ref()).unwrap();
                states
            })
            .unwrap();
        assert_eq!(profile.launches(), 1, "{steps} steps");
        assert_counts(&states[2].to_vec().unwrap(), &mandelbrot_cpu(&cx, &cy, steps)[2]);
    }
}

#[test]
fn one_and_four_states_with_scalars() {
    let ctx = Context::new().unwrap();
    let n = 2048;
    let xs = values(n, 5, -1.0, 1.0);
    let x = f32s_on(&ctx, &xs);
    let [single] = iterate_all([&x], 10, |[u]| [u * 0.9 + 0.1]);
    let mut want = xs.clone();
    for _ in 0..10 {
        want.iter_mut().for_each(|v| *v = *v * 0.9 + 0.1);
    }
    assert_close(&single.to_vec().unwrap(), &want, 1e-5, 1e-5);

    let four = iterate_all([&x, &x, &x, &x], 3, |[a, b, c, d]| [b * 1.0, c * 1.0, d * 1.0, a + b]);
    let mut s = [xs.clone(), xs.clone(), xs.clone(), xs.clone()];
    for _ in 0..3 {
        let sum: Vec<f32> = s[0].iter().zip(&s[1]).map(|(a, b)| a + b).collect();
        s = [s[1].clone(), s[2].clone(), s[3].clone(), sum];
    }
    for k in 0..4 {
        assert_close(&four[k].to_vec().unwrap(), &s[k], 1e-5, 1e-5);
    }
}

#[test]
fn a_step_with_a_shift_runs_step_by_step() {
    let ctx = Context::new().unwrap();
    let n = 512;
    let xs = values(n, 6, 0.0, 1.0);
    let x = f32s_on(&ctx, &xs);
    let [u, v] = iterate_all([&x, &x], 4, |[u, v]| [(u.shift([1]) + v) * 0.5, u * 1.0]);
    let (mut a, mut b) = (xs.clone(), xs.clone());
    for _ in 0..4 {
        let next: Vec<f32> = (0..n).map(|i| (a[(i + n - 1) % n] + b[i]) * 0.5).collect();
        b = a;
        a = next;
    }
    assert_close(&u.to_vec().unwrap(), &a, 1e-5, 1e-5);
    assert_close(&v.to_vec().unwrap(), &b, 1e-5, 1e-5);
}

#[test]
fn chunked_arrays_compute_correctly() {
    let ctx = Context::with_options(ContextOptions::default().chunk_bytes(1 << 12)).unwrap();
    let n = 10_000;
    let (cx, cy) = (values(n, 7, -2.0, 0.5), values(n, 8, -1.2, 1.2));
    let (gcx, gcy) = (f32s_on(&ctx, &cx), f32s_on(&ctx, &cy));
    let zero = Array::<f32>::zeros_on(&ctx, n).unwrap();
    let states = iterate_all([&zero, &zero, &zero], 20, |s| mandelbrot_step(&gcx, &gcy, s));
    assert_counts(&states[2].to_vec().unwrap(), &mandelbrot_cpu(&cx, &cy, 20)[2]);
}

#[test]
fn the_initial_arrays_keep_their_values() {
    let ctx = Context::new().unwrap();
    let xs = values(256, 9, 0.0, 1.0);
    let x = f32s_on(&ctx, &xs);
    let [y] = iterate_all([&x], 3, |[u]| [u * 2.0]);
    let want: Vec<f32> = xs.iter().map(|v| v * 8.0).collect();
    assert_close(&y.to_vec().unwrap(), &want, 1e-6, 1e-6);
    assert_eq!(x.to_vec().unwrap(), xs);
}

#[test]
fn many_captured_arrays_stay_correct() {
    let ctx = Context::new().unwrap();
    let n = 256;
    let inputs: Vec<Vec<f32>> = (0..40).map(|k| values(n, 100 + k, 0.0, 1.0)).collect();
    let arrays: Vec<Array<f32>> = inputs.iter().map(|data| f32s_on(&ctx, data)).collect();
    let zero = Array::<f32>::zeros_on(&ctx, n).unwrap();
    let [a, b] = iterate_all([&zero, &zero], 2, |[a, b]| {
        let mut na = a * 1.0;
        let mut nb = b * 1.0;
        for k in 0..20 {
            na += &arrays[k];
            nb += &arrays[20 + k];
        }
        [na, nb]
    });
    let sum = |range: std::ops::Range<usize>| -> Vec<f32> {
        (0..n).map(|i| 2.0 * range.clone().map(|k| inputs[k][i]).sum::<f32>()).collect()
    };
    assert_close(&a.to_vec().unwrap(), &sum(0..20), 1e-4, 1e-4);
    assert_close(&b.to_vec().unwrap(), &sum(20..40), 1e-4, 1e-4);
}

#[test]
fn a_state_stops_only_when_its_bits_stop_changing() {
    let ctx = Context::new().unwrap();
    let zero = Array::<f32>::zeros_on(&ctx, 64).unwrap();
    // -0.0 == 0.0, but the next step from -0.0 is +0.0 again: two steps end at +0.0.
    let [u] = iterate_all([&zero], 2, |[u]| [-u]);
    assert!(u.to_vec().unwrap().iter().all(|v| v.to_bits() == 0.0f32.to_bits()));
    let [v] = iterate_all([&zero], 3, |[u]| [-u]);
    assert!(v.to_vec().unwrap().iter().all(|v| v.to_bits() == (-0.0f32).to_bits()));
}

#[test]
fn a_state_that_cycles_never_stops_early() {
    let ctx = Context::new().unwrap();
    let start = Array::<f32>::full_on(&ctx, 64, 0.25).unwrap();
    for steps in [7, 8, 9, 16, 17, 33] {
        let [u] = iterate_all([&start], steps, |[u]| [1.0 - u]);
        let want = if steps.is_multiple_of(2) { 0.25 } else { 0.75 };
        assert!(u.to_vec().unwrap().iter().all(|&v| v == want), "{steps} steps");
    }
}

/// `steps` applications of `step` written as a host loop.
fn by_hand<const N: usize>(
    init: [&Array<f32>; N],
    steps: usize,
    step: impl Fn(&[Array<f32>; N]) -> [Array<f32>; N],
) -> [Array<f32>; N] {
    let mut state = init.map(|array| array * 1.0);
    for _ in 0..steps {
        state = step(&state);
    }
    state
}

#[test]
fn a_step_that_caches_in_the_middle_is_replayed() {
    let ctx = Context::new().unwrap();
    let x = f32s_on(&ctx, &values(200, 11, 0.0, 1.0));
    let step = |[u]: &[Array<f32>; 1]| {
        let smooth = (u * 0.5 + 0.1).cache();
        [smooth.shift(1) * 0.25 + smooth.shift(-1) * 0.25 + u * 0.5]
    };
    let [got] = iterate_all([&x], 37, step);
    let [want] = by_hand([&x], 37, step);
    assert_close(&got.to_vec().unwrap(), &want.to_vec().unwrap(), 1e-5, 1e-6);
}

#[test]
fn a_step_that_assigns_a_boundary_into_a_padded_temporary_is_replayed() {
    let ctx = Context::new().unwrap();
    let n = 24;
    let x = f32s_on(&ctx, &values(n, 12, 0.0, 1.0));
    let calls = std::cell::Cell::new(0);
    let step = |[u]: &[Array<f32>; 1]| {
        calls.set(calls.get() + 1);
        let mut w = u.pad([(1, 1)], PadMode::Edge);
        w.slice_mut(0, ..1).assign(0.0).unwrap();
        [w.slice(0, 1..n + 1) * 0.5 + w.slice(0, 0..n) * 0.25 + w.slice(0, 2..n + 2) * 0.25]
    };
    let [got] = iterate_all([&x], 37, step);
    // Traced once, and for an unrolled body of up to 16 steps where that makes fewer launches.
    assert!((1..=17).contains(&calls.get()), "traced {} times", calls.get());
    let [want] = by_hand([&x], 37, step);
    assert_close(&got.to_vec().unwrap(), &want.to_vec().unwrap(), 1e-5, 1e-6);
    // The state the loop started from is untouched.
    assert_eq!(x.to_vec().unwrap(), values(n, 12, 0.0, 1.0));
}

#[test]
fn a_step_that_reads_back_gives_the_host_loop_result() {
    let ctx = Context::new().unwrap();
    let x = f32s_on(&ctx, &values(50, 13, 1.0, 2.0));
    let y = f32s_on(&ctx, &values(50, 14, 1.0, 2.0));
    let step = |[a, b]: &[Array<f32>; 2]| {
        let m = a.max().item().unwrap();
        [a * (1.0 / m) + b * 0.25, b.shift(1) * 0.5 + a * 0.25]
    };
    let [ga, gb] = iterate_all([&x, &y], 9, step);
    let [wa, wb] = by_hand([&x, &y], 9, step);
    assert_close(&ga.to_vec().unwrap(), &wa.to_vec().unwrap(), 1e-5, 1e-6);
    assert_close(&gb.to_vec().unwrap(), &wb.to_vec().unwrap(), 1e-5, 1e-6);
}

#[test]
fn iterate_replays_a_step_with_a_cache_and_an_assignment() {
    let ctx = Context::new().unwrap();
    let n = 30;
    let x = f32s_on(&ctx, &values(n, 15, 0.0, 1.0));
    let step = |u: &Array<f32>| {
        let mut w = u.pad([(1, 1)], PadMode::Mirror);
        w.slice_mut(0, n + 1..).assign(0.5).unwrap();
        (w.slice(0, 1..n + 1) * 0.5 + w.slice(0, 2..n + 2) * 0.25 + u.cache().shift(1) * 0.25) * 0.9
    };
    let got = x.iterate(21, step);
    let mut want = &x * 1.0;
    for _ in 0..21 {
        want = step(&want);
    }
    assert_close(&got.to_vec().unwrap(), &want.to_vec().unwrap(), 1e-5, 1e-6);
}

/// The step reads a reduction of its state (`sum` of one element plus a captured array). The
/// reduction must run at every step: flattening the step for a register loop used to evaluate it
/// once on the placeholder, whose buffer holds unspecified data.
#[test]
fn a_step_reading_a_reduction_of_its_state_computes_it_at_every_step() {
    let ctx = Context::new().unwrap();
    let c = f32s_on(&ctx, &[20.0]);
    for _ in 0..3 {
        // Leave different data in the pool each time: the placeholder is a pooled buffer.
        let junk = f32s_on(&ctx, &[-3.5]);
        junk.to_vec().unwrap();
        drop(junk);
        let s = f32s_on(&ctx, &[20.0]);
        let [r] = iterate_all([&s], 3, |s| [s[0].sum() + &c]);
        assert_eq!(values_of(&r), [80.0]);
    }
    // The captured reduction does not depend on the state: still correct.
    let x = f32s_on(&ctx, &[1.0, 2.0, 3.0]);
    let s = f32s_on(&ctx, &[0.0]);
    let [r] = iterate_all([&s], 4, |s| [s[0].clone() + x.sum()]);
    assert_eq!(values_of(&r), [24.0]);
}

/// A step that returns its state unchanged returns an array of its own, not the placeholder that
/// borrows the state's buffers (`state *= result` bound one buffer twice).
#[test]
fn an_identity_step_over_a_chunked_state_returns_an_independent_array() {
    let ctx = Context::with_options(ContextOptions::from_env().chunk_bytes(1024)).unwrap();
    let data: Vec<f32> = (0..300).map(|i| i as f32).collect();
    let mut s = f32s_on(&ctx, &data);
    let [r] = iterate_all([&s], 2, |p| [p[0].clone()]);
    s *= &r;
    let want: Vec<f32> = data.iter().map(|v| v * v).collect();
    assert_eq!(values_of(&s), want);
}

fn values_of(array: &Array<f32>) -> Vec<f32> {
    array.to_vec().unwrap()
}

/// `steps` steps of the circular stencil `(u[i-1] + u[i+1]) / 4 + u / 2` on the host, in `f64`.
fn stencil_host(x: &[f32], steps: usize) -> Vec<f32> {
    let n = x.len();
    let mut u: Vec<f64> = x.iter().map(|&v| f64::from(v)).collect();
    for _ in 0..steps {
        u = (0..n).map(|i| (u[(i + n - 1) % n] + u[(i + 1) % n]) * 0.25 + u[i] * 0.5).collect();
    }
    u.iter().map(|&v| v as f32).collect()
}

fn stencil_step(u: &Array<f32>) -> Array<f32> {
    (u.shift([1]) + u.shift([-1])) * 0.25 + u * 0.5
}

#[test]
fn a_shifted_step_matches_the_step_by_step_loop_for_every_remainder() {
    // The step is traced as an unrolled body of several steps where that makes fewer launches, and the
    // steps that do not fill a body run one by one: whatever the count, the result is the loop's.
    let ctx = Context::new().unwrap();
    let xs = values(1000, 21, 0.0, 1.0);
    let x = f32s_on(&ctx, &xs);
    for steps in [1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 47, 100] {
        let [got] = iterate_all([&x], steps, |[u]| [stencil_step(u)]);
        assert_close(&got.to_vec().unwrap(), &stencil_host(&xs, steps), 2e-4, 2e-4);
    }
    assert_eq!(x.to_vec().unwrap(), xs, "the initial array changed");
}

#[test]
fn an_unrolled_loop_over_several_states_and_captured_arrays_matches_the_step_by_step_loop() {
    let ctx = Context::new().unwrap();
    let n = 777;
    let (us, vs, cs) = (values(n, 22, 0.0, 1.0), values(n, 23, -1.0, 1.0), values(n, 24, 0.0, 0.1));
    let (u0, v0, c) = (f32s_on(&ctx, &us), f32s_on(&ctx, &vs), f32s_on(&ctx, &cs));
    let step = |[u, v]: &[Array<f32>; 2]| [stencil_step(u) + &c * 0.5 + v * 0.01, v * 0.9 + u.shift([1]) * 0.1];
    for steps in [2, 5, 16, 17, 40] {
        let got = iterate_all([&u0, &v0], steps, step);
        let want = by_hand([&u0, &v0], steps, step);
        for (got, want) in got.iter().zip(&want) {
            assert_close(&got.to_vec().unwrap(), &want.to_vec().unwrap(), 5e-4, 5e-4);
        }
    }
}

#[test]
fn a_shifted_step_runs_in_fewer_launches_than_steps_where_the_planner_fuses_them() {
    let ctx = Context::new().unwrap();
    if ctx.backend() == BackendKind::Cpu {
        return;
    }
    let steps = 64;
    let xs = values(1 << 16, 25, 0.0, 1.0);
    let x = f32s_on(&ctx, &xs);
    let (got, profile) = ctx
        .profile(|| {
            let [u] = iterate_all([&x], steps, |[u]| [stencil_step(u)]);
            u.eval().unwrap();
            u
        })
        .unwrap();
    assert!(profile.launches() < steps as u32 / 2, "{} launches for {steps} steps", profile.launches());
    assert_close(&got.to_vec().unwrap(), &stencil_host(&xs, steps), 2e-4, 2e-4);
}
