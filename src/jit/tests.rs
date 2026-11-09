//! Unit tests of the fusion machinery: they look at how many kernels were generated, reused and
//! launched (the counters of [`JitState`](super::JitState)). What the results of fused
//! expressions must be is tested through the public API in `tests/fusion.rs`.

use super::StatsSnapshot;
use crate::array::Array;
use crate::context::Context;
use crate::ops::select;

fn values(n: usize, seed: u64, lo: f32, hi: f32) -> Vec<f32> {
    let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let unit = ((state >> 40) as f32) / (1u64 << 24) as f32;
            lo + unit * (hi - lo)
        })
        .collect()
}

fn assert_close(got: &[f32], want: &[f32], rel: f32, abs: f32) {
    assert_eq!(got.len(), want.len(), "length");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let tolerance = abs.max(rel * w.abs());
        assert!((g - w).abs() <= tolerance, "index {i}: got {g}, want {w} (tolerance {tolerance})");
    }
}

fn f32s(ctx: &Context, data: &[f32]) -> Array<f32> {
    Array::from_slice_on(ctx, data, data.len()).unwrap()
}

fn delta(before: StatsSnapshot, after: StatsSnapshot) -> StatsSnapshot {
    StatsSnapshot {
        built: after.built - before.built,
        hits: after.hits - before.hits,
        elementwise_launches: after.elementwise_launches - before.elementwise_launches,
        reduce_launches: after.reduce_launches - before.reduce_launches,
    }
}

#[test]
fn a_chain_of_operations_is_one_kernel_and_matches_the_cpu() {
    let ctx = Context::new().unwrap();
    let xs = values(5000, 1, 0.5, 2.0);
    let ys = values(5000, 2, -1.0, 1.0);
    let x = f32s(&ctx, &xs);
    let y = f32s(&ctx, &ys);

    let before = ctx.jit().stats();
    let z = ((&x * 2.0 + &y).sqrt() - 0.25).abs().minimum(&x) / 1.5 + (-&y).clamp(-0.5, 0.5);
    // Nothing has run: recording the expression launched no kernel.
    assert_eq!(delta(before, ctx.jit().stats()).elementwise_launches, 0);
    assert_eq!(ctx.pending_dispatches(), 0);

    let got = z.to_vec().unwrap();
    let d = delta(before, ctx.jit().stats());
    assert_eq!(d.elementwise_launches, 1, "the whole chain is one dispatch");
    assert_eq!(d.built, 1);

    let want: Vec<f32> = xs
        .iter()
        .zip(&ys)
        .map(|(&x, &y)| ((x * 2.0 + y).sqrt() - 0.25).abs().min(x) / 1.5 + (-y).clamp(-0.5, 0.5))
        .collect();
    assert_close(&got, &want, 1e-5, 1e-5);
}

#[test]
fn fused_results_match_the_unfused_evaluation_of_the_same_expression() {
    let xs = values(3001, 3, 0.1, 3.0);
    let ys = values(3001, 4, 0.1, 3.0);
    let run = |max_nodes: usize| {
        let ctx = Context::new().unwrap();
        ctx.jit().set_max_nodes(max_nodes);
        let x = f32s(&ctx, &xs);
        let y = f32s(&ctx, &ys);
        let before = ctx.jit().stats();
        let mut acc = &x * &y;
        for i in 0..20 {
            acc = (&acc + &x * (i as f32 * 0.1)).sqrt() * 1.1 - &y / 7.0 + 0.5;
        }
        let out = acc.to_vec().unwrap();
        (out, delta(before, ctx.jit().stats()).elementwise_launches)
    };
    let (fused, fused_launches) = run(64);
    let (unfused, unfused_launches) = run(1);
    // 141 operations: three kernels of at most 64, or one per operation.
    assert_eq!(fused_launches, 3);
    assert_eq!(unfused_launches, 141);
    assert_close(&fused, &unfused, 1e-4, 1e-4);
}

#[test]
fn comparison_into_select_stays_in_one_kernel() {
    let ctx = Context::new().unwrap();
    let xs = values(2048, 5, -1.0, 1.0);
    let x = f32s(&ctx, &xs);

    let before = ctx.jit().stats();
    // f32 inputs, a u32 mask in the middle, f32 out.
    let mask = x.gt(0.25);
    let out = select(&mask, &(&x * 3.0), &(&x - 1.0)) + 1.0;
    let got = out.to_vec().unwrap();
    let d = delta(before, ctx.jit().stats());
    assert_eq!(d.elementwise_launches, 1);

    let want: Vec<f32> =
        xs.iter().map(|&x| if x > 0.25 { x * 3.0 } else { x - 1.0 } + 1.0).collect();
    assert_close(&got, &want, 1e-6, 1e-6);

    // A u32 chain through a comparison: counts elements above a threshold per mask.
    let ints: Vec<u32> = (0..1000).collect();
    let u = Array::from_slice_on(&ctx, &ints, 1000).unwrap();
    let flags = (&u + 3u32).ge(500u32);
    let got = (&flags * 7u32 + 1u32).to_vec().unwrap();
    let want: Vec<u32> = ints.iter().map(|&v| u32::from(v + 3 >= 500) * 7 + 1).collect();
    assert_eq!(got, want);
}

#[test]
fn scalar_values_do_not_change_the_cache_key() {
    let ctx = Context::new().unwrap();
    let xs = values(1000, 6, -2.0, 2.0);
    let x = f32s(&ctx, &xs);

    let before = ctx.jit().stats();
    for i in 0..8 {
        let a = 0.5 + i as f32;
        let b = 3.0 - i as f32 * 0.25;
        let got = (&x * a + b).clamp(-b, a).to_vec().unwrap();
        let want: Vec<f32> = xs.iter().map(|&v| (v * a + b).clamp(-b, a)).collect();
        assert_close(&got, &want, 1e-5, 1e-5);
    }
    let d = delta(before, ctx.jit().stats());
    assert_eq!(d.elementwise_launches, 8);
    assert_eq!(d.built, 1, "one pipeline for eight different scalar values");
    assert_eq!(d.hits, 7);

    // A different structure (another operation) is another kernel.
    let _ = (&x * 2.0 - 1.0).clamp(-1.0, 1.0).to_vec().unwrap();
    assert_eq!(delta(before, ctx.jit().stats()).built, 2);
}

#[test]
fn reductions_fuse_their_input_expression() {
    let ctx = Context::new().unwrap();
    let n = 100_000;
    let xs = values(n, 7, -1.0, 1.0);
    let ys = values(n, 8, -1.0, 1.0);
    let a = f32s(&ctx, &xs);
    let b = f32s(&ctx, &ys);

    let before = ctx.jit().stats();
    let sum = (&a * &b).sum().unwrap();
    let d = delta(before, ctx.jit().stats());
    assert_eq!(d.elementwise_launches, 0, "the product is never materialized");
    assert_eq!(d.reduce_launches, 1);
    let want: f64 = xs.iter().zip(&ys).map(|(&x, &y)| f64::from(x) * f64::from(y)).sum();
    assert!((f64::from(sum) - want).abs() < 1e-2, "{sum} vs {want}");

    let before = ctx.jit().stats();
    let norm = (&a - &b).norm().unwrap();
    let min = (&a * 2.0 - 1.0).min().unwrap();
    let max = (&a * 2.0 - 1.0).max().unwrap();
    let mean = (&a + &b).mean().unwrap();
    let d = delta(before, ctx.jit().stats());
    assert_eq!(d.elementwise_launches, 0);
    assert_eq!(d.reduce_launches, 4);
    let want_norm = xs
        .iter()
        .zip(&ys)
        .map(|(&x, &y)| f64::from(x - y) * f64::from(x - y))
        .sum::<f64>()
        .sqrt();
    assert!((f64::from(norm) - want_norm).abs() < 1e-3 * want_norm, "{norm} vs {want_norm}");
    let want_min = xs.iter().map(|&x| x * 2.0 - 1.0).fold(f32::INFINITY, f32::min);
    let want_max = xs.iter().map(|&x| x * 2.0 - 1.0).fold(f32::NEG_INFINITY, f32::max);
    assert_eq!(min, want_min);
    assert_eq!(max, want_max);
    let want_mean = xs.iter().zip(&ys).map(|(&x, &y)| f64::from(x + y)).sum::<f64>() / n as f64;
    assert!((f64::from(mean) - want_mean).abs() < 1e-4, "{mean} vs {want_mean}");

    // u32: counting a mask.
    let count = a.gt(0.0).sum().unwrap();
    assert_eq!(count as usize, xs.iter().filter(|&&x| x > 0.0).count());
    assert_eq!(delta(before, ctx.jit().stats()).elementwise_launches, 0);

    // Sizes around the block boundaries, one-element arrays included.
    for len in [1usize, 2, 255, 1023, 1024, 1025, 4096, 4097, 1024 * 1024 + 3] {
        let data = values(len, len as u64, 0.0, 1.0);
        let x = f32s(&ctx, &data);
        let got = (&x * 2.0).sum().unwrap();
        let want: f64 = data.iter().map(|&v| f64::from(v) * 2.0).sum();
        assert!((f64::from(got) - want).abs() <= 1e-4 * want.abs().max(1.0), "{len}: {got} vs {want}");
        let got_max = (&x + 1.0).max().unwrap();
        let want_max = data.iter().map(|&v| v + 1.0).fold(f32::NEG_INFINITY, f32::max);
        assert_eq!(got_max, want_max, "{len}");
    }
}

#[test]
fn reducing_an_expression_does_not_evaluate_it() {
    let ctx = Context::new().unwrap();
    let xs = values(777, 9, 0.0, 1.0);
    let a = f32s(&ctx, &xs);
    let t = &a * 3.0;

    let before = ctx.jit().stats();
    let s = t.sum().unwrap();
    assert_eq!(delta(before, ctx.jit().stats()).elementwise_launches, 0);
    // The expression is still usable afterwards, and gets evaluated on its own.
    let want: Vec<f32> = xs.iter().map(|&v| v * 3.0).collect();
    assert_close(&t.to_vec().unwrap(), &want, 1e-6, 1e-6);
    let d = delta(before, ctx.jit().stats());
    assert_eq!((d.elementwise_launches, d.reduce_launches), (1, 1));
    let want_sum: f64 = want.iter().map(|&v| f64::from(v)).sum();
    assert!((f64::from(s) - want_sum).abs() < 1e-3);

    // Once evaluated, reductions read its buffer.
    let before = ctx.jit().stats();
    let again = t.sum().unwrap();
    assert_eq!(again, s);
    assert_eq!(delta(before, ctx.jit().stats()).reduce_launches, 0);
}

#[test]
fn deep_chains_are_split_at_the_node_limit() {
    let ctx = Context::new().unwrap();
    ctx.jit().set_max_nodes(64);
    let xs = values(500, 10, 0.5, 1.5);
    let x = f32s(&ctx, &xs);

    let before = ctx.jit().stats();
    // Two operations per round, 200 in total.
    let mut y = &x * 1.0001 + 0.001;
    let mut want: Vec<f32> = xs.iter().map(|&v| v * 1.0001 + 0.001).collect();
    for i in 0..99 {
        let s = 1.0 + (i % 3) as f32 * 0.01;
        y = y * s + 0.001;
        for w in &mut want {
            *w = *w * s + 0.001;
        }
    }
    // The graph was cut while it was built; the rest runs at to_vec.
    let cut = delta(before, ctx.jit().stats()).elementwise_launches;
    assert_eq!(cut, 3, "64-operation pieces were evaluated early");
    assert_close(&y.to_vec().unwrap(), &want, 1e-4, 1e-4);
    let d = delta(before, ctx.jit().stats());
    assert_eq!(d.elementwise_launches, 4);
    // Three full pieces share one pipeline, the remainder is another one.
    assert_eq!(d.built, 2);
    assert_eq!(d.hits, 2);

    // With a limit of one, every operation is a kernel of its own.
    let ctx = Context::new().unwrap();
    ctx.jit().set_max_nodes(1);
    let x = f32s(&ctx, &xs);
    let before = ctx.jit().stats();
    let y = (&x * 2.0 + 1.0).sqrt().exp();
    let got = y.to_vec().unwrap();
    assert_eq!(delta(before, ctx.jit().stats()).elementwise_launches, 4);
    let want: Vec<f32> = xs.iter().map(|&v| (v * 2.0 + 1.0).sqrt().exp()).collect();
    assert_close(&got, &want, 1e-4, 1e-4);
}

#[test]
fn expressions_over_many_buffers_are_split_by_the_binding_limit() {
    let ctx = Context::new().unwrap();
    let max_inputs = ctx.jit().max_inputs;
    let count = max_inputs * 2 + 3;
    let data: Vec<Vec<f32>> = (0..count).map(|i| values(300, 100 + i as u64, -1.0, 1.0)).collect();
    let arrays: Vec<Array<f32>> = data.iter().map(|d| f32s(&ctx, d)).collect();

    let before = ctx.jit().stats();
    let mut acc = &arrays[0] + &arrays[1];
    for a in &arrays[2..] {
        acc = acc + a;
    }
    let got = acc.to_vec().unwrap();
    let want: Vec<f32> = (0..300).map(|j| data.iter().map(|d| d[j]).sum()).collect();
    assert_close(&got, &want, 1e-4, 1e-4);
    let launches = delta(before, ctx.jit().stats()).elementwise_launches;
    assert!(launches >= 3, "{count} buffers with at most {max_inputs} inputs per kernel: {launches}");

    // The same input many times is one binding: 81 operations, one kernel.
    ctx.jit().set_max_nodes(128);
    let before = ctx.jit().stats();
    let mut acc = &arrays[0] * 1.0;
    for _ in 0..40 {
        acc = &acc + &arrays[0] * 0.5;
    }
    let got = acc.to_vec().unwrap();
    assert_close(&got, &data[0].iter().map(|v| v * 21.0).collect::<Vec<_>>(), 1e-4, 1e-4);
    assert_eq!(delta(before, ctx.jit().stats()).elementwise_launches, 1);
}

#[test]
fn a_node_with_two_consumers_is_computed_once() {
    let ctx = Context::new().unwrap();
    let xs = values(400, 11, 0.5, 2.0);
    let ys = values(400, 12, 0.5, 2.0);
    let x = f32s(&ctx, &xs);
    let y = f32s(&ctx, &ys);

    let before = ctx.jit().stats();
    let t = (&x * &y).sqrt();
    let u = &t + 1.0;
    let v = &t * 2.0; // second consumer, in another kernel: t is evaluated on its own
    let got_u = u.to_vec().unwrap();
    let got_v = v.to_vec().unwrap();
    let d = delta(before, ctx.jit().stats());
    assert_eq!(d.elementwise_launches, 3, "t once, u once, v once");

    let t_ref: Vec<f32> = xs.iter().zip(&ys).map(|(&a, &b)| (a * b).sqrt()).collect();
    assert_close(&got_u, &t_ref.iter().map(|v| v + 1.0).collect::<Vec<_>>(), 1e-5, 1e-5);
    assert_close(&got_v, &t_ref.iter().map(|v| v * 2.0).collect::<Vec<_>>(), 1e-5, 1e-5);
    assert_close(&t.to_vec().unwrap(), &t_ref, 1e-5, 1e-5);
    assert_eq!(delta(before, ctx.jit().stats()).elementwise_launches, 3);

    // If the other consumer is gone, the node is just part of the one kernel.
    let before = ctx.jit().stats();
    let t = (&x * &y).sqrt();
    let u = &t + 1.0;
    let v = &t * 2.0;
    drop(v);
    let got_u = u.to_vec().unwrap();
    assert_eq!(delta(before, ctx.jit().stats()).elementwise_launches, 1);
    assert_close(&got_u, &t_ref.iter().map(|v| v + 1.0).collect::<Vec<_>>(), 1e-5, 1e-5);
}

#[test]
fn sharing_inside_one_expression_is_one_instruction() {
    let ctx = Context::new().unwrap();
    let xs = values(1000, 15, 0.5, 2.0);
    let ys = values(1000, 16, 0.5, 2.0);
    let x = f32s(&ctx, &xs);
    let y = f32s(&ctx, &ys);

    let before = ctx.jit().stats();
    // `s` is used by three different operations, all inside the final expression.
    let s = (&x + &y).sin();
    let z = &s * &s + &s * 0.5 - (&s + 1.0).sqrt();
    let got = z.to_vec().unwrap();
    assert_eq!(delta(before, ctx.jit().stats()).elementwise_launches, 1);
    let want: Vec<f32> = xs
        .iter()
        .zip(&ys)
        .map(|(&a, &b)| {
            let s = (a + b).sin();
            s * s + s * 0.5 - (s + 1.0).sqrt()
        })
        .collect();
    assert_close(&got, &want, 1e-4, 1e-5);

    // A value-and-derivative style update that uses the previous value twice in each step.
    let before = ctx.jit().stats();
    let mut acc = &x * 1.0;
    let mut want = xs.clone();
    for _ in 0..10 {
        acc = &acc + &(&acc * 0.01);
        want.iter_mut().for_each(|w| *w += *w * 0.01);
    }
    assert_close(&acc.to_vec().unwrap(), &want, 1e-5, 1e-5);
    assert_eq!(delta(before, ctx.jit().stats()).elementwise_launches, 1);
}

#[test]
fn compound_assignment_on_a_lazy_array_fuses() {
    let ctx = Context::new().unwrap();
    let xs = values(333, 13, -1.0, 1.0);
    let x = f32s(&ctx, &xs);
    let before = ctx.jit().stats();
    let mut y = &x * 2.0;
    y += 1.0;
    y *= &x;
    y -= 0.5;
    let got = y.to_vec().unwrap();
    assert_eq!(delta(before, ctx.jit().stats()).elementwise_launches, 1);
    let want: Vec<f32> = xs.iter().map(|&v| (v * 2.0 + 1.0) * v - 0.5).collect();
    assert_close(&got, &want, 1e-5, 1e-5);

    // An array that already has a buffer is updated in place, one dispatch per operator.
    let mut z = f32s(&ctx, &xs);
    let before = ctx.jit().stats();
    z += 1.0;
    z *= 2.0;
    assert_eq!(ctx.pending_dispatches(), 2);
    assert_eq!(delta(before, ctx.jit().stats()).elementwise_launches, 0);
    let want: Vec<f32> = xs.iter().map(|&v| (v + 1.0) * 2.0).collect();
    assert_close(&z.to_vec().unwrap(), &want, 1e-6, 1e-6);
}

#[test]
fn nothing_runs_for_arrays_that_are_dropped_and_sync_runs_the_live_ones() {
    let ctx = Context::new().unwrap();
    let x = f32s(&ctx, &[1.0, 2.0, 3.0, 4.0]);

    let before = ctx.jit().stats();
    let unused = &x * 2.0 + 1.0;
    drop(unused);
    ctx.sync().unwrap();
    assert_eq!(delta(before, ctx.jit().stats()).elementwise_launches, 0);

    let kept = (&x * 2.0).sqrt();
    ctx.sync().unwrap();
    // `sync` evaluated it (one kernel for the whole expression) ...
    assert_eq!(delta(before, ctx.jit().stats()).elementwise_launches, 1);
    // ... so reading it back launches nothing more.
    assert_close(&kept.to_vec().unwrap(), &[2.0f32, 4.0, 6.0, 8.0].map(f32::sqrt), 1e-6, 1e-6);
    assert_eq!(delta(before, ctx.jit().stats()).elementwise_launches, 1);
}

#[test]
fn lazy_arrays_hold_their_inputs_alive() {
    let ctx = Context::new().unwrap();
    let data = values(1000, 14, 0.0, 1.0);
    let x = f32s(&ctx, &data);
    let y = &x * 2.0 + 1.0;
    drop(x);
    // The buffer of `x` must not be handed to these.
    let others: Vec<Array<f32>> = (0..8).map(|i| f32s(&ctx, &vec![i as f32; 1000])).collect();
    let want: Vec<f32> = data.iter().map(|&v| v * 2.0 + 1.0).collect();
    assert_close(&y.to_vec().unwrap(), &want, 1e-6, 1e-6);
    for (i, other) in others.iter().enumerate() {
        assert_eq!(other.to_vec().unwrap(), vec![i as f32; 1000]);
    }
}

#[test]
fn many_pending_readers_of_one_buffer_are_pruned() {
    let ctx = Context::new().unwrap();
    let x = f32s(&ctx, &[1.0; 64]);
    // A million-style loop of expressions that die unevaluated must not pile up in the
    // bookkeeping of `x`.
    for i in 0..10_000 {
        let _ = &x * (i as f32);
    }
    let kept = &x * 3.0;
    assert_eq!(kept.to_vec().unwrap(), vec![3.0; 64]);
    let readers = x.storage().ok().map(|s| s.readers.len()).unwrap_or(usize::MAX);
    assert!(readers < 200, "{readers}");
}

#[test]
fn eval_does_not_materialize_the_temporaries_of_the_expression() {
    let ctx = Context::new().unwrap();
    let x = f32s(&ctx, &values(256, 40, 0.0, 1.0));
    let before = ctx.jit().stats();
    // The intermediate arrays are temporaries alive until the end of the statement.
    ((&x * 2.0 + 1.0).sqrt() - 1.0).eval().unwrap();
    assert_eq!(delta(before, ctx.jit().stats()).elementwise_launches, 1);

    let before = ctx.jit().stats();
    let product = &x * &x;
    product.sum().unwrap();
    // Reduced without being materialized; flushing does not compute it afterwards.
    ctx.sync().unwrap();
    assert_eq!(delta(before, ctx.jit().stats()).elementwise_launches, 0);
}

#[test]
fn shifts_fuse_into_the_expression_that_reads_them() {
    let ctx = Context::new().unwrap();
    let xs = values(1000, 41, -1.0, 1.0);
    let n = xs.len();
    let x = f32s(&ctx, &xs);
    let u = &x * 2.0;
    let before = ctx.jit().stats();
    let step = &u + (u.shift(1) + u.shift(-1) - &u * 2.0) * 0.1;
    let got = step.to_vec().unwrap();
    // `u` is evaluated once for the shifts, then the whole step is one kernel.
    assert_eq!(delta(before, ctx.jit().stats()).elementwise_launches, 2);
    let f = |j: usize| xs[j % n] * 2.0;
    let want: Vec<f32> =
        (0..n).map(|i| f(i) + (f(i + n - 1) + f(i + 1) - f(i) * 2.0) * 0.1).collect();
    assert_close(&got, &want, 1e-5, 1e-5);

    // A shifted read inside a reduction's first pass.
    let before = ctx.jit().stats();
    let lag = (&x * &x.shift(3)).sum().unwrap();
    assert_eq!(delta(before, ctx.jit().stats()).elementwise_launches, 0);
    let want: f32 = (0..n).map(|i| xs[i] * xs[(i + n - 3) % n]).sum();
    assert!((lag - want).abs() < 1e-3, "{lag} vs {want}");

    // Multi-dimensional shifts of two arrays in one kernel.
    let (nx, ny) = (7usize, 5usize);
    let a = Array::from_slice_on(&ctx, &xs[..nx * ny], [nx, ny]).unwrap();
    let got = (a.shift([1isize, -2]) - a.shift([-3isize, 0])).to_vec().unwrap();
    let at = |cx: isize, cy: isize| {
        xs[cx.rem_euclid(nx as isize) as usize + nx * cy.rem_euclid(ny as isize) as usize]
    };
    let want: Vec<f32> = (0..nx * ny)
        .map(|i| {
            let (cx, cy) = ((i % nx) as isize, (i / nx) as isize);
            at(cx - 1, cy + 2) - at(cx + 3, cy)
        })
        .collect();
    assert_close(&got, &want, 1e-6, 1e-6);
}

#[test]
fn arrays_evaluated_together_share_one_kernel() {
    let ctx = Context::new().unwrap();
    let xs = values(3000, 50, 0.5, 2.0);
    let ys = values(3000, 51, 0.5, 2.0);
    let (x, y) = (f32s(&ctx, &xs), f32s(&ctx, &ys));

    let before = ctx.jit().stats();
    let ratio = &x / &y;
    let call = ratio.ln() * 2.0 + &x;
    let put = ratio.ln() * 3.0 - &y;
    // `ratio` is used by both: computed once, inside the same kernel.
    crate::ops::eval_all(&[&call, &put]).unwrap();
    assert_eq!(delta(before, ctx.jit().stats()).elementwise_launches, 1);
    let want_call: Vec<f32> = xs.iter().zip(&ys).map(|(x, y)| (x / y).ln() * 2.0 + x).collect();
    let want_put: Vec<f32> = xs.iter().zip(&ys).map(|(x, y)| (x / y).ln() * 3.0 - y).collect();
    assert_close(&call.to_vec().unwrap(), &want_call, 1e-5, 1e-5);
    assert_close(&put.to_vec().unwrap(), &want_put, 1e-5, 1e-5);
    assert_eq!(delta(before, ctx.jit().stats()).elementwise_launches, 1);

    // Arrays still alive at a sync are grouped the same way; a mask output has its own type.
    let before = ctx.jit().stats();
    let scaled = &x * 4.0;
    let mask = x.gt(1.0);
    let other_len = f32s(&ctx, &[1.0, 2.0]) * 2.0;
    ctx.sync().unwrap();
    assert_eq!(delta(before, ctx.jit().stats()).elementwise_launches, 2);
    assert_close(&scaled.to_vec().unwrap(), &xs.iter().map(|v| v * 4.0).collect::<Vec<_>>(), 1e-6, 1e-6);
    assert_eq!(mask.to_vec().unwrap(), xs.iter().map(|&v| u32::from(v > 1.0)).collect::<Vec<_>>());
    assert_eq!(other_len.to_vec().unwrap(), vec![2.0, 4.0]);
}

#[test]
fn several_reductions_share_one_pass() {
    use crate::ops::Reduction;
    let ctx = Context::new().unwrap();
    // More than one workgroup's block, so the later (segmented) passes run too.
    let xs = values(300_001, 52, -3.0, 2.0);
    let x = f32s(&ctx, &xs);
    let lazy = &x * 2.0;
    let before = ctx.jit().stats();
    let [sum, lo, hi, squares] =
        lazy.reduce([Reduction::Sum, Reduction::Min, Reduction::Max, Reduction::SumOfSquares]).unwrap();
    let d = delta(before, ctx.jit().stats());
    assert_eq!((d.reduce_launches, d.elementwise_launches), (1, 0));
    let doubled: Vec<f64> = xs.iter().map(|&v| f64::from(v) * 2.0).collect();
    let want_sum: f64 = doubled.iter().sum();
    let want_squares: f64 = doubled.iter().map(|v| v * v).sum();
    assert!((f64::from(sum) - want_sum).abs() < 1e-3 * want_squares.sqrt(), "{sum} vs {want_sum}");
    assert!((f64::from(squares) - want_squares).abs() < 1e-4 * want_squares, "{squares} vs {want_squares}");
    assert_eq!(lo, xs.iter().fold(f32::MAX, |m, &v| m.min(v * 2.0)));
    assert_eq!(hi, xs.iter().fold(f32::MIN, |m, &v| m.max(v * 2.0)));

    // On a materialized array, and an op repeated.
    let [a, b] = x.reduce([Reduction::Max, Reduction::Max]).unwrap();
    assert_eq!((a, b), (hi / 2.0, hi / 2.0));
}
