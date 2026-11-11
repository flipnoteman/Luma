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
