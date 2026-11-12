//! Unit tests of the fusion machinery: they look at how many kernels were generated, reused and
//! launched (the counters of [`JitState`](super::JitState)). What the results of fused
//! expressions must be is tested through the public API in `tests/fusion.rs`.

use super::StatsSnapshot;
use crate::array::Array;
use crate::context::{Context, ContextOptions};
use crate::jit::cost::CostModel;
use crate::ops::select;

/// A context with the calibration off and the cost model pinned, so that kernel counts do not
/// depend on the machine.
fn ctx() -> Context {
    let ctx = Context::with_options(ContextOptions::from_env().calibrate(false)).unwrap();
    ctx.set_cost_model(CostModel::TEST);
    ctx
}

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
    let ctx = ctx();
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
        let ctx = ctx();
        ctx.jit().set_max_nodes(max_nodes);
        let x = f32s(&ctx, &xs);
        let y = f32s(&ctx, &ys);
        let before = ctx.jit().stats();
        let mut acc = &x * &y;
        for i in 0..20 {
            // A different divisor each round: identical operations would be one node.
            acc = (&acc + &x * (i as f32 * 0.1)).sqrt() * 1.1 - &y / (7.0 + i as f32) + 0.5;
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
    let ctx = ctx();
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
    let ctx = ctx();
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
    let ctx = ctx();
    let n = 100_000;
    let xs = values(n, 7, -1.0, 1.0);
    let ys = values(n, 8, -1.0, 1.0);
    let a = f32s(&ctx, &xs);
    let b = f32s(&ctx, &ys);

    let before = ctx.jit().stats();
    let sum = (&a * &b).sum().item().unwrap();
    let d = delta(before, ctx.jit().stats());
    assert_eq!(d.elementwise_launches, 0, "the product is never materialized");
    assert_eq!(d.reduce_launches, 1);
    let want: f64 = xs.iter().zip(&ys).map(|(&x, &y)| f64::from(x) * f64::from(y)).sum();
    assert!((f64::from(sum) - want).abs() < 1e-2, "{sum} vs {want}");

    let before = ctx.jit().stats();
    let norm = (&a - &b).norm().item().unwrap();
    let min = (&a * 2.0 - 1.0).min().item().unwrap();
    let max = (&a * 2.0 - 1.0).max().item().unwrap();
    let mean = (&a + &b).mean().item().unwrap();
    let d = delta(before, ctx.jit().stats());
    // The expression is never materialized; `norm` and `mean` finish with a one-element kernel each.
    assert_eq!(d.elementwise_launches, 2);
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
    let count = a.gt(0.0).sum().item().unwrap();
    assert_eq!(count as usize, xs.iter().filter(|&&x| x > 0.0).count());
    assert_eq!(delta(before, ctx.jit().stats()).elementwise_launches, 2);

    // Sizes around the block boundaries, one-element arrays included.
    for len in [1usize, 2, 255, 1023, 1024, 1025, 4096, 4097, 1024 * 1024 + 3] {
        let data = values(len, len as u64, 0.0, 1.0);
        let x = f32s(&ctx, &data);
        let got = (&x * 2.0).sum().item().unwrap();
        let want: f64 = data.iter().map(|&v| f64::from(v) * 2.0).sum();
        assert!((f64::from(got) - want).abs() <= 1e-4 * want.abs().max(1.0), "{len}: {got} vs {want}");
        let got_max = (&x + 1.0).max().item().unwrap();
        let want_max = data.iter().map(|&v| v + 1.0).fold(f32::NEG_INFINITY, f32::max);
        assert_eq!(got_max, want_max, "{len}");
    }
}

#[test]
fn reducing_an_expression_does_not_evaluate_it() {
    let ctx = ctx();
    let xs = values(777, 9, 0.0, 1.0);
    let a = f32s(&ctx, &xs);
    let t = &a * 3.0;

    let before = ctx.jit().stats();
    let s = t.sum().item().unwrap();
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
    let again = t.sum().item().unwrap();
    assert_eq!(again, s);
    assert_eq!(delta(before, ctx.jit().stats()).reduce_launches, 0);
}

#[test]
fn deep_chains_are_split_at_the_node_limit() {
    let ctx = ctx();
    ctx.jit().set_max_nodes(64);
    let xs = values(500, 10, 0.5, 1.5);
    let x = f32s(&ctx, &xs);

    let before = ctx.jit().stats();
    // Two operations per round, 200 in total.
    let mut y = &x * 1.0001 + 0.001;
    let mut want: Vec<f32> = xs.iter().map(|&v| v * 1.0001 + 0.001).collect();
    for i in 0..99 {
        // Never exactly 1: `y * 1.0` would be simplified away and the chain would be shorter.
        let s = 1.0 + (i % 3 + 1) as f32 * 0.01;
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
    // The first piece reads `x`, which stays alive, so it writes a new buffer; the later pieces
    // read a dying temporary and write over it, which is a different kernel. Of those, the
    // second and third full pieces share one pipeline, the remainder is another one.
    assert_eq!(d.built, 3);
    assert_eq!(d.hits, 1);

    // With a limit of one, every operation is a kernel of its own.
    let ctx = self::ctx();
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
    let ctx = ctx();
    let max_inputs = ctx.jit().max_inputs();
    let count = max_inputs * 2 + 3;
    let data: Vec<Vec<f32>> = (0..count).map(|i| values(300, 100 + i as u64, -1.0, 1.0)).collect();
    let arrays: Vec<Array<f32>> = data.iter().map(|d| f32s(&ctx, d)).collect();

    let before = ctx.jit().stats();
    let mut acc = &arrays[0] + &arrays[1];
    for a in &arrays[2..] {
        acc += a;
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
    let ctx = ctx();
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
    let ctx = ctx();
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
    let ctx = ctx();
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

    // An array that already has a buffer and no other holder is lazy too; evaluating it runs one
    // kernel for both operators, in the array's own buffer.
    let mut z = f32s(&ctx, &xs);
    let before = ctx.jit().stats();
    z += 1.0;
    z *= 2.0;
    assert_eq!(ctx.pending_dispatches(), 0);
    assert_eq!(delta(before, ctx.jit().stats()).elementwise_launches, 0);
    let want: Vec<f32> = xs.iter().map(|&v| (v + 1.0) * 2.0).collect();
    assert_close(&z.to_vec().unwrap(), &want, 1e-6, 1e-6);
    assert_eq!(delta(before, ctx.jit().stats()).elementwise_launches, 1);
}

#[test]
fn compound_assignment_on_a_solely_owned_array_reuses_its_buffer() {
    let ctx = ctx();
    let xs = values(1000, 15, -1.0, 1.0);
    let mut z = f32s(&ctx, &xs);
    let buffer = z.buffer();
    let before = ctx.jit().stats();
    z += 1.0;
    z *= 2.0;
    let got = z.to_vec().unwrap();
    assert_eq!(delta(before, ctx.jit().stats()).elementwise_launches, 1);
    let want: Vec<f32> = xs.iter().map(|&v| (v + 1.0) * 2.0).collect();
    assert_close(&got, &want, 1e-6, 1e-6);
    assert_eq!(z.buffer(), buffer, "the result lives in the old buffer");
}

#[test]
fn compound_assignment_on_a_shared_buffer_is_one_in_place_kernel() {
    let ctx = ctx();
    let xs = values(500, 16, -1.0, 1.0);
    let a = f32s(&ctx, &xs);
    let b = f32s(&ctx, &values(500, 17, -1.0, 1.0));
    let x = f32s(&ctx, &xs);
    let view = x.reshape(500);
    let mut y = x.reshape(500);
    drop(x);

    let before = ctx.jit().stats();
    y += &(&a * &b);
    // The product is part of the kernel, not a buffer of its own.
    assert_eq!(delta(before, ctx.jit().stats()).elementwise_launches, 1);
    let want: Vec<f32> = xs
        .iter()
        .zip(a.to_vec().unwrap().iter().zip(b.to_vec().unwrap()))
        .map(|(&x, (&p, q))| x + p * q)
        .collect();
    assert_close(&y.to_vec().unwrap(), &want, 1e-6, 1e-6);
    assert_close(&view.to_vec().unwrap(), &want, 1e-6, 1e-6);
    assert_eq!(delta(before, ctx.jit().stats()).elementwise_launches, 1);
}

#[test]
fn compound_assignment_does_not_change_earlier_expressions() {
    let ctx = ctx();
    let xs = values(64, 18, -1.0, 1.0);
    let x = f32s(&ctx, &xs);
    let mut x2 = x.reshape(64);
    let p = &x * 2.0;
    x2 += 10.0;
    let want: Vec<f32> = xs.iter().map(|&v| v * 2.0).collect();
    assert_close(&p.to_vec().unwrap(), &want, 1e-6, 1e-6);
    let want: Vec<f32> = xs.iter().map(|&v| v + 10.0).collect();
    assert_close(&x.to_vec().unwrap(), &want, 1e-6, 1e-6);
}

#[test]
fn a_dying_input_read_through_a_shift_is_not_overwritten() {
    let ctx = ctx();
    let xs = values(257, 19, -1.0, 1.0);
    let mut z = f32s(&ctx, &xs);
    // `z` is the only holder of its buffer once the old array is replaced, but the expression
    // reads it at other elements than the one it writes.
    z = z.shift(1) + &z;
    let n = xs.len();
    let want: Vec<f32> = (0..n).map(|i| xs[(i + n - 1) % n] + xs[i]).collect();
    assert_close(&z.to_vec().unwrap(), &want, 1e-6, 1e-6);
}

#[test]
fn nothing_runs_for_arrays_that_are_dropped_and_sync_runs_the_live_ones() {
    let ctx = ctx();
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
    let ctx = ctx();
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
    let ctx = ctx();
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
    let ctx = ctx();
    let x = f32s(&ctx, &values(256, 40, 0.0, 1.0));
    let before = ctx.jit().stats();
    // The intermediate arrays are temporaries alive until the end of the statement.
    ((&x * 2.0 + 1.0).sqrt() - 1.0).eval().unwrap();
    assert_eq!(delta(before, ctx.jit().stats()).elementwise_launches, 1);

    let before = ctx.jit().stats();
    let product = &x * &x;
    product.sum().item().unwrap();
    // Reduced without being materialized; flushing does not compute it afterwards.
    ctx.sync().unwrap();
    assert_eq!(delta(before, ctx.jit().stats()).elementwise_launches, 0);
}

#[test]
fn shifts_fuse_into_the_expression_that_reads_them() {
    let ctx = ctx();
    let xs = values(1000, 41, -1.0, 1.0);
    let n = xs.len();
    let x = f32s(&ctx, &xs);
    let u = &x * 2.0;
    let before = ctx.jit().stats();
    let step = &u + (u.shift(1) + u.shift(-1) - &u * 2.0) * 0.1;
    let got = step.to_vec().unwrap();
    // `u` is small: recomputed at the shifted elements, so the whole step is one kernel.
    assert_eq!(delta(before, ctx.jit().stats()).elementwise_launches, 1);
    let f = |j: usize| xs[j % n] * 2.0;
    let want: Vec<f32> =
        (0..n).map(|i| f(i) + (f(i + n - 1) + f(i + 1) - f(i) * 2.0) * 0.1).collect();
    assert_close(&got, &want, 1e-5, 1e-5);

    // An expression beyond the staging limit is evaluated once for its shifts instead.
    let mut big = &x * 1.0;
    for _ in 0..40 {
        big = &big * 0.5 + 0.25;
    }
    let before = ctx.jit().stats();
    let got = (big.shift(1) - &big).to_vec().unwrap();
    assert_eq!(delta(before, ctx.jit().stats()).elementwise_launches, 2);
    let g = |v: f32| (0..40).fold(v, |a, _| a * 0.5 + 0.25);
    let want: Vec<f32> = (0..n).map(|i| g(xs[(i + n - 1) % n]) - g(xs[i])).collect();
    assert_close(&got, &want, 1e-5, 1e-5);

    // Shifts of shifted expressions nest.
    let nested = ((&x * 2.0).shift(1) + 1.0).shift(1).to_vec().unwrap();
    let want: Vec<f32> = (0..n).map(|i| xs[(i + 2 * n - 2) % n] * 2.0 + 1.0).collect();
    assert_close(&nested, &want, 1e-6, 1e-6);


    // A shifted expression shared with an outside consumer is evaluated once instead of being
    // recomputed inside the shift's kernel.
    let before = ctx.jit().stats();
    let shared = &x * 3.0 + 1.0;
    let shifted = shared.shift(1);
    let other = &shared - 2.0;
    let got_shifted = shifted.to_vec().unwrap();
    let got_other = other.to_vec().unwrap();
    assert_eq!(delta(before, ctx.jit().stats()).elementwise_launches, 3);
    let want_shifted: Vec<f32> = (0..n).map(|i| xs[(i + n - 1) % n] * 3.0 + 1.0).collect();
    let want_other: Vec<f32> = xs.iter().map(|&v| v * 3.0 - 1.0).collect();
    assert_close(&got_shifted, &want_shifted, 1e-5, 1e-5);
    assert_close(&got_other, &want_other, 1e-5, 1e-5);
    // A shifted read inside a reduction's first pass.
    let before = ctx.jit().stats();
    let lag = (&x * &x.shift(3)).sum().item().unwrap();
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
fn a_life_step_recomputes_through_its_shifts() {
    let ctx = ctx();
    let (w, h) = (64usize, 64usize);
    let cells: Vec<f32> = values(w * h, 7, 0.0, 1.0).into_iter().map(|v| if v > 0.5 { 1.0 } else { 0.0 }).collect();
    let step = |u: &Array<f32>| {
        let mut n = u.shift([1isize, 1]) + u.shift([1isize, 0]) + u.shift([1isize, -1]);
        n = n + u.shift([0isize, 1]) + u.shift([0isize, -1]);
        n = n + u.shift([-1isize, 1]) + u.shift([-1isize, 0]) + u.shift([-1isize, -1]);
        let survive = select(&n.eq(2.0), u, 0.0);
        select(&n.eq(3.0), 1.0, &survive)
    };
    let u = Array::from_slice_on(&ctx, &cells, [w, h]).unwrap();
    let before = ctx.jit().stats();
    let got = step(&step(&u)).to_vec().unwrap();
    assert_eq!(delta(before, ctx.jit().stats()).elementwise_launches, 1);

    let life = |g: &[f32]| -> Vec<f32> {
        (0..w * h)
            .map(|i| {
                let (x, y) = ((i % w) as isize, (i / w) as isize);
                let mut n = 0.0;
                for dy in -1isize..=1 {
                    for dx in -1isize..=1 {
                        if (dx, dy) != (0, 0) {
                            let j = (x + dx).rem_euclid(w as isize) as usize + w * (y + dy).rem_euclid(h as isize) as usize;
                            n += g[j];
                        }
                    }
                }
                if n == 3.0 || (n == 2.0 && g[i] == 1.0) { 1.0 } else { 0.0 }
            })
            .collect()
    };
    assert_close(&got, &life(&life(&cells)), 0.0, 0.0);
}

fn life_step(u: &Array<f32>) -> Array<f32> {
    let mut n = u.shift([1isize, 1]) + u.shift([1isize, 0]) + u.shift([1isize, -1]);
    n = n + u.shift([0isize, 1]) + u.shift([0isize, -1]);
    n = n + u.shift([-1isize, 1]) + u.shift([-1isize, 0]) + u.shift([-1isize, -1]);
    let survive = select(&n.eq(2.0), u, 0.0);
    select(&n.eq(3.0), 1.0, &survive)
}

fn life_cpu(g: &[f32], w: usize, h: usize) -> Vec<f32> {
    (0..w * h)
        .map(|i| {
            let (x, y) = ((i % w) as isize, (i / w) as isize);
            let mut n = 0.0;
            for dy in -1isize..=1 {
                for dx in -1isize..=1 {
                    if (dx, dy) != (0, 0) {
                        n += g[(x + dx).rem_euclid(w as isize) as usize + w * (y + dy).rem_euclid(h as isize) as usize];
                    }
                }
            }
            if n == 3.0 || (n == 2.0 && g[i] == 1.0) { 1.0 } else { 0.0 }
        })
        .collect()
}

#[test]
fn a_lazy_chain_of_life_steps_is_staged_in_workgroup_memory() {
    let ctx = ctx();
    for (w, h) in [(64usize, 64usize), (37, 29)] {
        let cells: Vec<f32> = values(w * h, 9, 0.0, 1.0).into_iter().map(|v| if v > 0.5 { 1.0 } else { 0.0 }).collect();
        let mut want = cells.clone();
        for steps in 1..=6 {
            want = life_cpu(&want, w, h);
            let mut u = Array::from_slice_on(&ctx, &cells, [w, h]).unwrap();
            let before = ctx.jit().stats();
            for _ in 0..steps {
                u = life_step(&u);
            }
            let got = u.to_vec().unwrap();
            if steps == 3 {
                assert_eq!(delta(before, ctx.jit().stats()).elementwise_launches, 1, "{w}x{h}: three levels in one kernel");
            }
            assert_close(&got, &want, 0.0, 0.0);
        }
    }
}

#[test]
fn a_shifted_expression_also_read_directly_is_computed_once() {
    let ctx = ctx();
    let cells = values(64 * 64, 10, 0.0, 1.0);
    let u = Array::from_slice_on(&ctx, &cells, [64, 64]).unwrap();
    // The life step reads `u` through eight shifts and once at the cell itself.
    let two = life_step(&life_step(&u));
    let node = two.pending().unwrap();
    let fused = super::program::flatten(&[(std::sync::Arc::clone(node), node.operands().unwrap())], super::program::Tail::Store);
    let fused = fused.ok().unwrap();
    assert_eq!(fused.key.subs.len(), 1);
    // The outer step plus one read of the inner step's staged value, not a second inner step.
    assert_eq!(fused.key.instrs.len(), fused.key.subs[0].len() + 1);
}

#[test]
fn a_lazy_chain_of_heat_steps_matches_iterate() {
    let ctx = ctx();
    let cells = values(64 * 64, 21, 0.0, 1.0);
    let g = Array::from_slice_on(&ctx, &cells, [64, 64]).unwrap();
    for steps in 1..=10 {
        let mut u = Array::from_slice_on(&ctx, &cells, [64, 64]).unwrap();
        for _ in 0..steps {
            u = heat_step(&u);
        }
        assert_close(&u.to_vec().unwrap(), &g.iterate(steps, heat_step).to_vec().unwrap(), 1e-5, 1e-5);
    }

    let n = 1000;
    let xs = values(n, 22, 0.0, 1.0);
    let step = |x: &Array<f32>| x + (x.shift(1) + x.shift(-1) - x * 2.0) * 0.25;
    let mut want = xs.clone();
    for steps in 1..=8 {
        want = (0..n).map(|i| want[i] + (want[(i + n - 1) % n] + want[(i + 1) % n] - want[i] * 2.0) * 0.25).collect();
        let mut u = f32s(&ctx, &xs);
        let before = ctx.jit().stats();
        for _ in 0..steps {
            u = step(&u);
        }
        let got = u.to_vec().unwrap();
        if steps == 3 {
            assert_eq!(delta(before, ctx.jit().stats()).elementwise_launches, 1);
        }
        assert_close(&got, &want, 1e-5, 1e-5);
    }
}

#[test]
fn a_shift_wider_than_the_staging_limit_still_gives_the_right_values() {
    let ctx = ctx();
    let n = 1000;
    let xs = values(n, 31, 0.0, 1.0);
    let x = f32s(&ctx, &xs);
    let got = ((&x * 2.0 + 1.0).shift(3) * 0.5 + (&x * 3.0).shift(-1)).to_vec().unwrap();
    let want: Vec<f32> = (0..n).map(|i| (xs[(i + n - 3) % n] * 2.0 + 1.0) * 0.5 + xs[(i + 1) % n] * 3.0).collect();
    assert_close(&got, &want, 1e-6, 1e-6);

    let (nx, ny, nz) = (4usize, 3usize, 5usize);
    let cube = Array::from_slice_on(&ctx, &xs[..nx * ny * nz], [nx, ny, nz]).unwrap();
    let got = ((&cube * 2.0).shift([1isize, 0, -1]) + 1.0).to_vec().unwrap();
    let want: Vec<f32> = (0..nx * ny * nz)
        .map(|i| {
            let (cx, cy, cz) = (i % nx, (i / nx) % ny, i / (nx * ny));
            xs[(cx + nx - 1) % nx + nx * (cy + ny * ((cz + 1) % nz))] * 2.0 + 1.0
        })
        .collect();
    assert_close(&got, &want, 1e-6, 1e-6);
}

#[test]
fn mixed_types_and_scalars_in_a_staged_expression() {
    let ctx = ctx();
    let (w, h) = (23usize, 14usize);
    let xs = values(w * h, 41, 0.0, 2.0);
    let x = Array::from_slice_on(&ctx, &xs, [w, h]).unwrap();
    let scale = 1.5f32;
    let mask = x.gt(1.0);
    let got = (select(&mask.shift([1isize, 0]), (&x * scale).shift([0isize, 1]), &x) + select(&mask.shift([-1isize, 1]), 1.0, 0.0))
        .to_vec()
        .unwrap();
    let at = |cx: isize, cy: isize| (cx.rem_euclid(w as isize) as usize) + w * cy.rem_euclid(h as isize) as usize;
    let want: Vec<f32> = (0..w * h)
        .map(|i| {
            let (cx, cy) = ((i % w) as isize, (i / w) as isize);
            let first = if xs[at(cx - 1, cy)] > 1.0 { xs[at(cx, cy - 1)] * scale } else { xs[i] };
            first + if xs[at(cx + 1, cy - 1)] > 1.0 { 1.0 } else { 0.0 }
        })
        .collect();
    assert_close(&got, &want, 1e-6, 1e-6);
}

#[test]
fn arrays_evaluated_together_share_one_kernel() {
    let ctx = ctx();
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
    use crate::reduce::Reduction;
    let ctx = ctx();
    // More than one workgroup's block, so the later (segmented) passes run too.
    let xs = values(300_001, 52, -3.0, 2.0);
    let x = f32s(&ctx, &xs);
    let lazy = &x * 2.0;
    let before = ctx.jit().stats();
    let [sum, lo, hi, squares] =
        lazy.reduce([Reduction::Sum, Reduction::Min, Reduction::Max, Reduction::SumOfSquares]).to_array().unwrap();
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
    let [a, b] = x.reduce([Reduction::Max, Reduction::Max]).to_array().unwrap();
    assert_eq!((a, b), (hi / 2.0, hi / 2.0));
}

#[test]
fn arrays_advancing_together_are_split_into_multi_output_kernels() {
    let n = 600;
    let cxs = values(n, 20, -0.3, 0.3);
    let cys = values(n, 21, -0.3, 0.3);
    let ctx = ctx();
    ctx.jit().set_max_nodes(64);
    let (cx, cy) = (f32s(&ctx, &cxs), f32s(&ctx, &cys));
    let zero = f32s(&ctx, &vec![0.0; n]);
    let one = f32s(&ctx, &vec![1.0; n]);

    let before = ctx.jit().stats();
    let (mut zx, mut zy, mut cnt) = (&zero * 1.0, &zero * 1.0, &zero * 1.0);
    let iterations = 40;
    for _ in 0..iterations {
        let inside = (&zx * &zx + &zy * &zy).le(4.0);
        let nzx = &zx * &zx - &zy * &zy + &cx;
        let nzy = &zx * &zy * 2.0 + &cy;
        zx = select(&inside, &nzx, &zx);
        zy = select(&inside, &nzy, &zy);
        cnt = &cnt + select(&inside, &one, &zero);
    }
    let got = [zx.to_vec().unwrap(), zy.to_vec().unwrap(), cnt.to_vec().unwrap()];
    let launches = delta(before, ctx.jit().stats()).elementwise_launches;

    let (mut wx, mut wy, mut wc) = (vec![0.0f32; n], vec![0.0f32; n], vec![0.0f32; n]);
    for _ in 0..iterations {
        for j in 0..n {
            let (x, y) = (wx[j], wy[j]);
            if x * x + y * y <= 4.0 {
                wx[j] = x * x - y * y + cxs[j];
                wy[j] = x * y * 2.0 + cys[j];
                wc[j] += 1.0;
            }
        }
    }
    assert_close(&got[0], &wx, 1e-3, 1e-3);
    assert_close(&got[1], &wy, 1e-3, 1e-3);
    assert_close(&got[2], &wc, 0.0, 0.0);
    // About 14 operations per iteration across three results: a few kernels per 64 operations,
    // not one single-output kernel per operand.
    // (The operations the three results share, like `inside`, are one node each and computed on
    // their own where several kernels read them, which costs a few more kernels than recomputing.)
    assert!(launches <= 32, "{launches} kernels");
}

#[test]
fn arrays_advancing_together_for_many_iterations_split_into_few_kernels() {
    let n = 600;
    let cxs = values(n, 20, -0.3, 0.3);
    let cys = values(n, 21, -0.3, 0.3);
    let ctx = ctx();
    let (cx, cy) = (f32s(&ctx, &cxs), f32s(&ctx, &cys));
    let zero = f32s(&ctx, &vec![0.0; n]);
    let one = f32s(&ctx, &vec![1.0; n]);

    let before = ctx.jit().stats();
    let (mut zx, mut zy, mut cnt) = (&zero * 1.0, &zero * 1.0, &zero * 1.0);
    let iterations = 90;
    // No `eval` while building: the limits of `create` alone cut the chains. The three arrays
    // are evaluated together at the end (`flush` computes the live arrays that share work in
    // multi-output kernels; reading them one by one would force `zx` alone, which makes every
    // node that `zy` and `cnt` also read a kernel of its own).
    for _ in 0..iterations {
        let inside = (&zx * &zx + &zy * &zy).le(4.0);
        let nzx = &zx * &zx - &zy * &zy + &cx;
        let nzy = &zx * &zy * 2.0 + &cy;
        zx = select(&inside, &nzx, &zx);
        zy = select(&inside, &nzy, &zy);
        cnt = &cnt + select(&inside, &one, &zero);
    }
    ctx.flush().unwrap();
    let got = [zx.to_vec().unwrap(), zy.to_vec().unwrap(), cnt.to_vec().unwrap()];
    let launches = delta(before, ctx.jit().stats()).elementwise_launches;

    let (mut wx, mut wy, mut wc) = (vec![0.0f32; n], vec![0.0f32; n], vec![0.0f32; n]);
    for _ in 0..iterations {
        for j in 0..n {
            let (x, y) = (wx[j], wy[j]);
            if x * x + y * y <= 4.0 {
                wx[j] = x * x - y * y + cxs[j];
                wy[j] = x * y * 2.0 + cys[j];
                wc[j] += 1.0;
            }
        }
    }
    assert_close(&got[0], &wx, 1e-3, 1e-3);
    assert_close(&got[1], &wy, 1e-3, 1e-3);
    assert_close(&got[2], &wc, 0.0, 0.0);
    assert!(launches <= 16, "{launches} kernels");
}

#[test]
fn the_node_limit_counts_distinct_nodes_only() {
    let ctx = ctx();
    ctx.jit().set_max_nodes(61);
    let xs = values(300, 40, 0.5, 1.5);
    let x = f32s(&ctx, &xs);

    // Two chains of `rounds` steps (two nodes each) that both start at `common` and are built
    // alternately, so their node ids interleave. `common` is read by both chains and by the
    // last operation, yet counts once: `4 * rounds + 4` distinct nodes lie below the result.
    let run = |rounds: usize| {
        let before = ctx.jit().stats();
        let common = &x * 1.1;
        let (mut a, mut b) = (&common + 0.5, &common - 0.5);
        for _ in 0..rounds {
            a = &a * 1.01 * 1.02;
            b = &b * 0.99 * 0.98;
        }
        let z = &a + &b + &common;
        let got = z.to_vec().unwrap();
        let want: Vec<f32> = xs
            .iter()
            .map(|&v| {
                let common = v * 1.1;
                let (mut a, mut b) = (common + 0.5, common - 0.5);
                for _ in 0..rounds {
                    a = a * 1.01 * 1.02;
                    b = b * 0.99 * 0.98;
                }
                a + b + common
            })
            .collect();
        assert_close(&got, &want, 1e-4, 1e-4);
        delta(before, ctx.jit().stats()).elementwise_launches
    };
    // 60 distinct nodes fit the limit of 61, counted by what is shared and not by its paths.
    assert_eq!(run(14), 1);
    // 64 do not.
    assert!(run(15) >= 2);
}

#[test]
fn a_long_register_loop_is_split_into_launches_with_the_same_result() {
    let ctx = ctx();
    let n = 1000;
    let xs = values(n, 11, -1.0, 1.0);
    let x = f32s(&ctx, &xs);
    let step = |[u, v]: &[Array<f32>; 2]| [u * 0.5 + v, v * 0.9];
    let [u0, v0] = crate::iterate::iterate_all([&x, &x], 9, step);
    let (u0, v0) = (u0.to_vec().unwrap(), v0.to_vec().unwrap());

    // Three steps of this program per launch.
    ctx.jit().set_loop_work(n as u64 * 4 * 3);
    let before = ctx.jit().stats();
    let [u1, v1] = crate::iterate::iterate_all([&x, &x], 9, step);
    let (u1, v1) = (u1.to_vec().unwrap(), v1.to_vec().unwrap());
    assert_eq!(delta(before, ctx.jit().stats()).elementwise_launches, 3);
    assert_eq!(u1, u0);
    assert_eq!(v1, v0);
}

#[test]
fn requesting_an_array_many_times_keeps_one_entry() {
    let ctx = ctx();
    let x = f32s(&ctx, &values(64, 12, 0.0, 1.0));
    let y = &x * 2.0 + 1.0;
    let node = y.pending().unwrap().clone();
    for _ in 0..1000 {
        ctx.jit().request(&node);
    }
    assert_eq!(ctx.jit().requested_len(), 1);
    ctx.jit().evaluate_requested();
    assert_eq!(ctx.jit().requested_len(), 0);
    assert!(y.pending().is_none());
}

/// `x = x * 0.5 + y; y = y * 0.9 + 0.1` on the CPU; the states after every step.
fn elementwise_steps(xs: &[f32], ys: &[f32], steps: usize) -> Vec<(Vec<f32>, Vec<f32>)> {
    let (mut x, mut y) = (xs.to_vec(), ys.to_vec());
    let mut states = Vec::new();
    for _ in 0..steps {
        for (x, y) in x.iter_mut().zip(&y) {
            *x = *x * 0.5 + y;
        }
        y.iter_mut().for_each(|y| *y = *y * 0.9 + 0.1);
        states.push((x.clone(), y.clone()));
    }
    states
}

#[test]
fn an_eval_loop_of_an_elementwise_step_runs_as_one_loop() {
    let ctx = ctx();
    ctx.set_assume_busy(true);
    let (xs, ys) = (values(4096, 60, 0.5, 1.5), values(4096, 61, 0.5, 1.5));
    let (mut x, mut y) = (f32s(&ctx, &xs), f32s(&ctx, &ys));

    let before = ctx.jit().stats();
    for _ in 0..50 {
        x = &x * 0.5 + &y;
        y = &y * 0.9 + 0.1;
        crate::ops::eval_all(&[&x, &y]).unwrap();
    }
    ctx.sync().unwrap();
    let launches = delta(before, ctx.jit().stats()).elementwise_launches;
    assert!(launches <= 3, "{launches} kernels for 50 steps");

    let (want_x, want_y) = elementwise_steps(&xs, &ys, 50).pop().unwrap();
    assert_close(&x.to_vec().unwrap(), &want_x, 1e-5, 1e-5);
    assert_close(&y.to_vec().unwrap(), &want_y, 1e-5, 1e-5);
}

fn heat_step(u: &Array<f32>) -> Array<f32> {
    let lap = u.shift([1isize, 0]) + u.shift([-1isize, 0]) + u.shift([0isize, 1]) + u.shift([0isize, -1]) - u * 4.0;
    u + lap * 0.2
}

#[test]
fn an_eval_loop_of_a_shift_step_runs_time_blocked() {
    let ctx = ctx();
    ctx.set_assume_busy(true);
    let g = Array::from_slice_on(&ctx, &values(64 * 64, 62, 0.0, 1.0), [64, 64]).unwrap();

    let before = ctx.jit().stats();
    let mut u = &g * 1.0;
    for _ in 0..100 {
        u = heat_step(&u);
        u.eval().unwrap();
    }
    ctx.sync().unwrap();
    let got = u.to_vec().unwrap();
    let launches = delta(before, ctx.jit().stats()).elementwise_launches;
    assert!(launches < 20, "{launches} kernels for 100 steps");

    let want = g.iterate(100, heat_step).to_vec().unwrap();
    assert_close(&got, &want, 1e-6, 1e-6);
}

#[test]
fn a_kept_intermediate_state_keeps_its_value() {
    let ctx = ctx();
    ctx.set_assume_busy(true);
    let (xs, ys) = (values(1024, 63, 0.5, 1.5), values(1024, 64, 0.5, 1.5));
    let (mut x, mut y) = (f32s(&ctx, &xs), f32s(&ctx, &ys));

    let mut kept = None;
    for i in 0..50 {
        x = &x * 0.5 + &y;
        y = &y * 0.9 + 0.1;
        crate::ops::eval_all(&[&x, &y]).unwrap();
        if i == 9 {
            kept = Some(&x * 1.0);
        }
    }
    let states = elementwise_steps(&xs, &ys, 50);
    assert_close(&kept.unwrap().to_vec().unwrap(), &states[9].0, 1e-5, 1e-5);
    assert_close(&x.to_vec().unwrap(), &states[49].0, 1e-5, 1e-5);
    assert_close(&y.to_vec().unwrap(), &states[49].1, 1e-5, 1e-5);
}

#[test]
fn writing_a_fixed_input_after_a_compressed_loop_does_not_change_it() {
    let ctx = ctx();
    ctx.set_assume_busy(true);
    let (xs, cs) = (values(512, 65, 0.5, 1.5), values(512, 66, 0.5, 1.5));
    let mut c = f32s(&ctx, &cs);
    let mut x = f32s(&ctx, &xs);
    for _ in 0..30 {
        x = &x * 0.5 + &c;
        x.eval().unwrap();
    }
    c.write(&vec![1.0; 512]).unwrap();

    let want: Vec<f32> = xs.iter().zip(&cs).map(|(&x, &c)| (0..30).fold(x, |x, _| x * 0.5 + c)).collect();
    assert_close(&x.to_vec().unwrap(), &want, 1e-5, 1e-5);
    assert_eq!(c.to_vec().unwrap(), vec![1.0; 512]);
}

#[test]
fn a_scalar_that_changes_every_step_still_gives_the_right_result() {
    let ctx = ctx();
    ctx.set_assume_busy(true);
    let xs = values(777, 67, 0.5, 1.5);
    let mut x = f32s(&ctx, &xs);
    let mut want = xs.clone();
    for k in 0..40 {
        let dt = 0.5 + k as f32 * 0.001;
        x = &x * dt + 1.0;
        x.eval().unwrap();
        want.iter_mut().for_each(|w| *w = *w * dt + 1.0);
    }
    assert_close(&x.to_vec().unwrap(), &want, 1e-5, 1e-5);
}

#[test]
fn kernels_built_stay_bounded_for_long_eval_loops() {
    let ctx = ctx();
    let w = 128;
    let (dx, dy) = (3.2 / w as f32, 2.8 / w as f32);
    let cxs: Vec<f32> = (0..w * w).map(|i| -2.2 + (i % w) as f32 * dx).collect();
    let cys: Vec<f32> = (0..w * w).map(|i| -1.4 + (i / w) as f32 * dy).collect();
    let iters = 200;
    let cx = Array::from_slice_on(&ctx, &cxs, [w, w]).unwrap();
    let cy = Array::from_slice_on(&ctx, &cys, [w, w]).unwrap();
    let zero = Array::<f32>::full_on(&ctx, [w, w], 0.0).unwrap();

    let before = ctx.jit().stats();
    let (mut zx, mut zy, mut cnt) = (&zero * 1.0, &zero * 1.0, &zero * 1.0);
    for _ in 0..iters {
        let inside = (&zx * &zx + &zy * &zy).le(4.0);
        let nzx = &zx * &zx - &zy * &zy + &cx;
        let nzy = &zx * &zy * 2.0 + &cy;
        zx = select(&inside, &nzx, &zx);
        zy = select(&inside, &nzy, &zy);
        cnt = &cnt + select(&inside, 1.0, 0.0);
        crate::ops::eval_all(&[&zx, &zy, &cnt]).unwrap();
    }
    let got = cnt.to_vec().unwrap();
    let built = delta(before, ctx.jit().stats()).built;
    assert!(built <= 8, "{built} kernels built for {iters} steps");

    let want: Vec<f32> = cxs
        .iter()
        .zip(&cys)
        .map(|(&cx, &cy)| {
            let (mut x, mut y, mut count) = (0.0f32, 0.0f32, 0.0f32);
            for _ in 0..iters {
                if x * x + y * y <= 4.0 {
                    (x, y) = (x * x - y * y + cx, x * y * 2.0 + cy);
                    count += 1.0;
                }
            }
            count
        })
        .collect();
    let mismatched = got.iter().zip(&want).filter(|(g, w)| g != w).count();
    assert!((mismatched as f32) < want.len() as f32 * 0.01, "{mismatched} of {} pixels differ", want.len());
}

#[test]
fn an_eval_loop_of_a_shift_step_over_two_states_matches_the_cpu() {
    let ctx = ctx();
    ctx.set_assume_busy(true);
    let n = 500;
    let (xs, ys) = (values(n, 70, 0.0, 1.0), values(n, 71, 0.0, 1.0));
    let (mut a, mut b) = (f32s(&ctx, &xs), f32s(&ctx, &ys));
    let (mut want_a, mut want_b) = (xs.clone(), ys.clone());
    for _ in 0..30 {
        let (next_a, next_b) = (&a + b.shift(1) * 0.1, &b * 0.99 + a.shift(-1) * 0.05);
        (a, b) = (next_a, next_b);
        crate::ops::eval_all(&[&a, &b]).unwrap();
        let (old_a, old_b) = (want_a.clone(), want_b.clone());
        for i in 0..n {
            want_a[i] = old_a[i] + old_b[(i + n - 1) % n] * 0.1;
            want_b[i] = old_b[i] * 0.99 + old_a[(i + 1) % n] * 0.05;
        }
    }
    assert_close(&a.to_vec().unwrap(), &want_a, 1e-5, 1e-5);
    assert_close(&b.to_vec().unwrap(), &want_b, 1e-5, 1e-5);
}

#[test]
fn an_eval_loop_over_chunked_arrays_matches_the_cpu() {
    let options = crate::ContextOptions::default().chunk_bytes(1 << 12);
    let ctx = Context::with_options(options).unwrap();
    ctx.set_assume_busy(true);
    let n = 10_000;
    let (xs, ys) = (values(n, 72, 0.5, 1.5), values(n, 73, 0.5, 1.5));
    let (mut x, mut y) = (f32s(&ctx, &xs), f32s(&ctx, &ys));
    for _ in 0..30 {
        x = &x * 0.5 + &y;
        y = &y * 0.9 + 0.1;
        crate::ops::eval_all(&[&x, &y]).unwrap();
    }
    let (want_x, want_y) = elementwise_steps(&xs, &ys, 30).pop().unwrap();
    assert_close(&x.to_vec().unwrap(), &want_x, 1e-5, 1e-5);
    assert_close(&y.to_vec().unwrap(), &want_y, 1e-5, 1e-5);
}

#[test]
fn reading_back_in_the_middle_of_an_eval_loop_gives_the_right_values() {
    let ctx = ctx();
    ctx.set_assume_busy(true);
    let start = values(32 * 32, 74, 0.0, 1.0);
    let g = Array::from_slice_on(&ctx, &start, [32, 32]).unwrap();
    let mut u = &g * 1.0;
    let mut want = start;
    for i in 0..60 {
        u = heat_step(&u);
        u.eval().unwrap();
        let old = want.clone();
        for (j, w) in want.iter_mut().enumerate() {
            let (x, y) = (j % 32, j / 32);
            let at = |x: usize, y: usize| old[x % 32 + (y % 32) * 32];
            let lap = at(x + 31, y) + at(x + 1, y) + at(x, y + 31) + at(x, y + 1) - old[j] * 4.0;
            *w = old[j] + lap * 0.2;
        }
        // A read-back ends the compressed loop; the next steps start another one.
        if i % 7 == 0 {
            assert_close(&u.to_vec().unwrap(), &want, 1e-4, 1e-4);
        }
    }
    assert_close(&u.to_vec().unwrap(), &want, 1e-4, 1e-4);
}

#[test]
fn an_eval_loop_of_a_step_too_big_to_recompute_still_runs_as_one_loop() {
    let ctx = ctx();
    ctx.set_assume_busy(true);
    let extra = ctx.tuning().max_recomputed;
    // Longer than the recompute limit: a shift of the previous state evaluates it.
    let step = |u: &Array<f32>| (0..extra).fold(heat_step(u), |r, _| r * 1.0 + 0.0);
    let g = Array::from_slice_on(&ctx, &values(64 * 64, 65, 0.0, 1.0), [64, 64]).unwrap();

    let before = ctx.jit().stats();
    let mut u = &g * 1.0;
    for _ in 0..100 {
        u = step(&u);
        u.eval().unwrap();
    }
    ctx.sync().unwrap();
    let got = u.to_vec().unwrap();
    let launches = delta(before, ctx.jit().stats()).elementwise_launches;
    assert!(launches < 20, "{launches} kernels for 100 steps");

    let want = g.iterate(100, step).to_vec().unwrap();
    assert_close(&got, &want, 1e-6, 1e-6);
}

/// Runs `next = step(&u); err = (&next - &u).abs().max().item()?; u = next` on a 32×32 torus with a
/// fixed source `c`, `alpha(k)` as the diffusion of step `k`, and `c` overwritten before step
/// `rewrite_at`, and checks every `err` and the final state against a CPU loop.
fn check_readback_loop(ctx: &Context, steps: usize, alpha: impl Fn(usize) -> f32, rewrite_at: Option<usize>) {
    const W: usize = 32;
    let (u0, c0, c1) = (values(W * W, 80, 0.0, 1.0), values(W * W, 81, 0.0, 1.0), values(W * W, 82, 0.0, 1.0));
    let mut c = Array::from_slice_on(ctx, &c0, [W, W]).unwrap();
    let mut u = &Array::from_slice_on(ctx, &u0, [W, W]).unwrap() * 1.0;
    let (mut cpu, mut source) = (u0, c0);
    for k in 0..steps {
        if rewrite_at == Some(k) {
            c.write(&c1).unwrap();
            source = c1.clone();
        }
        let a = alpha(k);
        let lap = u.shift([1isize, 0]) + u.shift([-1isize, 0]) + u.shift([0isize, 1]) + u.shift([0isize, -1]) - &u * 4.0;
        let next = &u + lap * a + &c * 0.01;
        let err = (&next - &u).abs().max().item().unwrap();
        u = next;

        let at = |i: usize, j: usize| cpu[i % W * W + j % W];
        let mut new = vec![0.0f32; W * W];
        for i in 0..W {
            for j in 0..W {
                let lap = at(i + 1, j) + at(i + W - 1, j) + at(i, j + 1) + at(i, j + W - 1) - at(i, j) * 4.0;
                new[i * W + j] = at(i, j) + lap * a + source[i * W + j] * 0.01;
            }
        }
        let want = new.iter().zip(&cpu).map(|(n, o)| (n - o).abs()).fold(0.0, f32::max);
        assert!((err - want).abs() <= 1e-4 * want + 1e-6, "step {k}: err {err}, want {want}");
        cpu = new;
    }
    assert_close(&u.to_vec().unwrap(), &cpu, 1e-4, 1e-5);
}

#[test]
fn a_readback_loop_gives_the_same_values_with_speculation() {
    let ctx = ctx();
    check_readback_loop(&ctx, 60, |_| 0.2, None);
    assert!(ctx.jit().speculation_hits() > 0, "no readback was answered ahead of time");
}

#[test]
fn a_step_that_changes_turns_speculation_off_and_stays_correct() {
    let ctx = ctx();
    check_readback_loop(&ctx, 40, |k| 0.2 - k as f32 * 0.002, None);
    assert_eq!(ctx.jit().speculation_hits(), 0);
}

#[test]
fn writing_an_input_between_readbacks_discards_the_prediction() {
    let ctx = ctx();
    check_readback_loop(&ctx, 40, |_| 0.2, Some(20));
    assert!(ctx.jit().speculation_hits() > 0);
}

#[test]
fn a_readback_loop_does_not_speculate_when_it_is_off() {
    let ctx = ctx();
    ctx.jit().set_speculate(false);
    check_readback_loop(&ctx, 40, |_| 0.2, None);
    assert_eq!(ctx.jit().speculation_hits(), 0);
}

#[test]
fn a_difference_of_slices_is_one_kernel() {
    let ctx = ctx();
    let data = values(9, 21, -1.0, 1.0);
    let f = f32s(&ctx, &data);
    let before = ctx.jit().stats();
    let got = (f.slice(0, 1..) - f.slice(0, ..8)).to_vec().unwrap();
    let want: Vec<f32> = (0..8).map(|i| data[i + 1] - data[i]).collect();
    assert_close(&got, &want, 1e-6, 1e-6);
    assert_eq!(delta(before, ctx.jit().stats()).elementwise_launches, 1);

    // A slice of a lazy expression is computed inside the kernel of its user.
    let before = ctx.jit().stats();
    let e = &f * 2.0 + 1.0;
    let got = (e.slice(0, 1..) - e.slice(0, ..8)).to_vec().unwrap();
    let want: Vec<f32> = (0..8).map(|i| (data[i + 1] * 2.0 + 1.0) - (data[i] * 2.0 + 1.0)).collect();
    assert_close(&got, &want, 1e-5, 1e-6);
    assert_eq!(delta(before, ctx.jit().stats()).elementwise_launches, 1);
}

#[test]
fn a_boundary_assignment_reading_the_target_is_one_kernel() {
    let ctx = ctx();
    let (ni, nj) = (8usize, 6usize);
    let data = values(ni * nj, 40, -1.0, 1.0);
    let want_ghost: Vec<f32> = (0..nj).flat_map(|j| [-data[3 + ni * j], -data[2 + ni * j]]).collect();
    let check = |w: &Array<f32>| {
        let got = w.to_vec().unwrap();
        for j in 0..nj {
            assert_eq!(&got[ni * j..ni * j + 2], &want_ghost[2 * j..2 * j + 2]);
            assert_eq!(&got[ni * j + 2..ni * (j + 1)], &data[ni * j + 2..ni * (j + 1)]);
        }
    };

    // By value: the expression is private to the write, so it all fuses into it.
    let w = Array::from_slice_on(&ctx, &data, [ni, nj]).unwrap();
    let before = ctx.jit().stats();
    w.slice(0, ..2).assign(w.slice(0, 2..4).flip(0) * -1.0).unwrap();
    assert_eq!(delta(before, ctx.jit().stats()).elementwise_launches, 1);
    check(&w);

    // By reference: the expression stays pending and reads the columns the write leaves alone.
    let w = Array::from_slice_on(&ctx, &data, [ni, nj]).unwrap();
    let g = w.slice(0, 2..4).flip(0) * -1.0;
    let before = ctx.jit().stats();
    w.slice(0, ..2).assign(&g).unwrap();
    assert_eq!(delta(before, ctx.jit().stats()).elementwise_launches, 1);
    check(&w);
    let want_g: Vec<f32> = (0..nj).flat_map(|j| [-data[3 + ni * j], -data[2 + ni * j]]).collect();
    assert_eq!(g.to_vec().unwrap(), want_g);
}

#[test]
fn assigning_an_expression_of_the_whole_target_is_one_in_place_kernel() {
    let ctx = ctx();
    let data = values(500, 41, -1.0, 1.0);
    let w = f32s(&ctx, &data);
    let before = ctx.jit().stats();
    w.assign((&w * 2.0 + 1.0) * 0.5 - &w * 0.25).unwrap();
    assert_eq!(delta(before, ctx.jit().stats()).elementwise_launches, 1);
    let want: Vec<f32> = data.iter().map(|v| (v * 2.0 + 1.0) * 0.5 - v * 0.25).collect();
    assert_close(&w.to_vec().unwrap(), &want, 1e-5, 1e-6);
}

#[test]
fn a_record_ghost_assignment_is_one_kernel() {
    use crate::fields::Fields;
    let ctx = ctx();
    let (ni, nj) = (8usize, 6usize);
    let data: Vec<Vec<f32>> = (0..4).map(|k| values(ni * nj, 50 + k, -1.0, 1.0)).collect();
    let record = |data: &[Vec<f32>]| -> [Array<f32>; 4] {
        std::array::from_fn(|k| Array::from_slice_on(&ctx, &data[k], [ni, nj]).unwrap())
    };
    // ghost field k combines inner fields k and k + 1.
    let want: Vec<Vec<f32>> = (0..4)
        .map(|k| {
            (0..nj)
                .flat_map(|j| {
                    let inner = |x: usize| data[k][x + ni * j] * 2.0 + data[(k + 1) % 4][x + ni * j];
                    [inner(3), inner(2)]
                })
                .collect()
        })
        .collect();
    let check = |w: &[Array<f32>; 4]| {
        for k in 0..4 {
            let got = w[k].to_vec().unwrap();
            for j in 0..nj {
                assert_eq!(&got[ni * j..ni * j + 2], &want[k][2 * j..2 * j + 2], "field {k} row {j}");
                assert_eq!(&got[ni * j + 2..ni * (j + 1)], &data[k][ni * j + 2..ni * (j + 1)]);
            }
        }
    };

    let w = record(&data);
    let inner = w.slice(0, 2..4).flip(0);
    let ghost: [Array<f32>; 4] = std::array::from_fn(|k| &inner[k] * 2.0 + &inner[(k + 1) % 4]);
    drop(inner);
    let before = ctx.jit().stats();
    w.slice(0, ..2).assign(ghost).unwrap();
    assert_eq!(delta(before, ctx.jit().stats()).elementwise_launches, 1);
    check(&w);

    // Kept alive and by reference: same values.
    let w = record(&data);
    let inner = w.slice(0, 2..4).flip(0);
    let ghost: [Array<f32>; 4] = std::array::from_fn(|k| &inner[k] * 2.0 + &inner[(k + 1) % 4]);
    w.slice(0, ..2).assign(&ghost).unwrap();
    check(&w);
}

#[test]
fn a_wide_expression_of_hundreds_of_operations_is_one_kernel() {
    let ctx = ctx();
    let n = 777;
    let (xs, ys, zs) = (values(n, 90, 0.5, 1.5), values(n, 91, 0.5, 1.5), values(n, 92, 0.5, 1.5));
    let (x, y, z) = (f32s(&ctx, &xs), f32s(&ctx, &ys), f32s(&ctx, &zs));
    let terms = 150;

    let before = ctx.jit().stats();
    // Four operations per term; every constant differs, so no two terms share a node.
    let mut acc = &x * 0.0;
    for i in 0..terms {
        let c = 0.01 * (i + 1) as f32;
        acc = acc + ((&x * c + &y) * &z - c * 0.5);
    }
    let got = acc.to_vec().unwrap();
    assert_eq!(delta(before, ctx.jit().stats()).elementwise_launches, 1);

    let want: Vec<f32> = (0..n)
        .map(|j| {
            let mut sum = xs[j] * 0.0;
            for i in 0..terms {
                let c = 0.01 * (i + 1) as f32;
                sum += (xs[j] * c + ys[j]) * zs[j] - c * 0.5;
            }
            sum
        })
        .collect();
    assert_close(&got, &want, 1e-4, 1e-3);
}

#[test]
fn a_fusion_limit_scope_splits_the_expressions_written_in_it() {
    let ctx = ctx();
    let xs = values(500, 93, 0.5, 1.5);
    let x = f32s(&ctx, &xs);
    let chain = |x: &Array<f32>| {
        let mut acc = x * 1.0;
        for _ in 0..20 {
            acc = &acc * 1.01 + 0.5;
        }
        acc
    };

    let before = ctx.jit().stats();
    let limited = crate::fusion_limit(8, || chain(&x));
    let a = limited.to_vec().unwrap();
    assert!(delta(before, ctx.jit().stats()).elementwise_launches > 1);

    let before = ctx.jit().stats();
    let b = chain(&x).to_vec().unwrap();
    assert_eq!(delta(before, ctx.jit().stats()).elementwise_launches, 1);
    assert_close(&a, &b, 1e-5, 1e-6);
}

#[test]
fn a_very_deep_chain_is_cut_and_evaluates_on_a_small_stack() {
    let handle = std::thread::Builder::new()
        // The walk is bounded by `MAX_DEPTH`; unoptimized frames are several times larger, so a debug
        // build gets the default stack of a test thread for the same depth.
        .stack_size(if cfg!(debug_assertions) { 2 << 20 } else { 256 * 1024 })
        .spawn(|| {
            let ctx = ctx();
            let x = f32s(&ctx, &[1.0, 2.0, 3.0]);
            let mut acc = &x * 1.0;
            for _ in 0..300 {
                acc = &acc + 1.0;
            }
            assert_eq!(acc.to_vec().unwrap(), vec![301.0, 302.0, 303.0]);
        })
        .unwrap();
    handle.join().unwrap();
}

#[test]
fn identical_lazy_operations_are_one_node() {
    let ctx = ctx();
    let xs = values(200, 94, 0.0, 2.0);
    let mut x = f32s(&ctx, &xs);
    let f = |a: &Array<f32>| (a * 2.0 + 1.0).sqrt();

    let got = (f(&x) + f(&x)).to_vec().unwrap();
    // Multiply, add, square root, the second (held, so unshared) square root and the sum.
    assert_eq!(ctx.jit().last_program_len(), 5);
    let want: Vec<f32> = xs.iter().map(|v| 2.0 * (v * 2.0 + 1.0).sqrt()).collect();
    assert_close(&got, &want, 1e-5, 1e-6);

    // After a write the same expression is a new value.
    let fresh = values(200, 95, 0.0, 2.0);
    x.write(&fresh).unwrap();
    let want: Vec<f32> = fresh.iter().map(|v| (v * 2.0 + 1.0).sqrt()).collect();
    assert_close(&f(&x).to_vec().unwrap(), &want, 1e-5, 1e-6);
}

#[test]
fn held_nodes_are_not_shared_but_their_temporaries_are() {
    let ctx = ctx();
    let xs = values(200, 97, 0.0, 2.0);
    let x = f32s(&ctx, &xs);
    let f = |a: &Array<f32>| (a * 2.0 + 1.0).sqrt();
    let u = f(&x);
    let v = f(&x);
    crate::ops::eval_all(&[&u, &v]).unwrap();
    // Multiply and add are shared; the two held square roots are not.
    assert!(ctx.jit().last_program_len() <= 5, "{}", ctx.jit().last_program_len());
    let want: Vec<f32> = xs.iter().map(|v| (v * 2.0 + 1.0).sqrt()).collect();
    assert_close(&u.to_vec().unwrap(), &want, 1e-5, 1e-6);
    assert_close(&v.to_vec().unwrap(), &want, 1e-5, 1e-6);
}

#[test]
fn cuts_of_an_expression_are_noted_with_their_reason_and_place() {
    use crate::profile::SplitReason;
    let ctx = ctx();
    ctx.jit().set_max_nodes(16);
    ctx.jit().record_splits();
    let x = f32s(&ctx, &values(100, 96, 0.5, 1.5));
    let mut acc = &x * 1.0;
    for _ in 0..20 {
        acc = &acc * 1.01 + 0.5;
    }
    acc.eval().unwrap();
    let splits = ctx.jit().take_splits();
    let first = splits.first().expect("the chain exceeds the limit");
    assert!(matches!(first.reason, SplitReason::Nodes { limit: 16, .. }), "{first:?}");
    assert!(first.location.file().ends_with("tests.rs"), "{}", first.location);
    assert!(first.to_string().contains("operations > 16"), "{first}");
}

#[test]
fn a_program_with_many_live_values_is_split_in_the_middle() {
    use crate::profile::SplitReason;
    let ctx = ctx();
    ctx.jit().set_max_live(4);
    ctx.jit().record_splits();
    let xs = values(300, 97, 0.5, 2.0);
    let x = f32s(&ctx, &xs);
    let count = 12;
    // The first value stays live until the last sum: `v1 + (v2 + (v3 + ..))`.
    let terms: Vec<Array<f32>> = (1..=count).map(|k| (&x * k as f32).sqrt()).collect();
    let total = terms.into_iter().rev().reduce(|acc, v| v + acc).unwrap();
    drop(x);
    let got = total.to_vec().unwrap();
    let splits = ctx.jit().take_splits();
    assert!(splits.iter().any(|split| matches!(split.reason, SplitReason::Live { limit: 4, .. })), "{splits:?}");

    let want: Vec<f32> = xs.iter().map(|v| (1..=count).map(|k| (v * k as f32).sqrt()).rev().fold(0.0, |acc, t| t + acc)).collect();
    assert_close(&got, &want, 1e-4, 1e-5);
}

#[test]
fn a_third_view_reader_recomputes_the_expression_inside_the_one_kernel() {
    let ctx = ctx();
    let n = 400;
    let xs = values(n, 98, 0.5, 2.0);
    let x = f32s(&ctx, &xs);
    let e = (&x * 2.0 + 1.0).sqrt();
    let host: Vec<f32> = xs.iter().map(|v| (v * 2.0 + 1.0).sqrt()).collect();

    // Two readers recompute the three operations at their offsets.
    let before = ctx.jit().stats();
    let got = (e.slice(0, 1..) - e.slice(0, ..n - 1)).to_vec().unwrap();
    assert_eq!(delta(before, ctx.jit().stats()).elementwise_launches, 1);
    let want: Vec<f32> = (0..n - 1).map(|i| host[i + 1] - host[i]).collect();
    assert_close(&got, &want, 1e-5, 1e-6);

    // A third reader recomputes it too (once per distinct index, inside the one kernel): views
    // of a pending expression no longer evaluate it by reader count.
    let e = (&x * 2.5 + 1.0).sqrt();
    let host: Vec<f32> = xs.iter().map(|v| (v * 2.5 + 1.0).sqrt()).collect();
    let before = ctx.jit().stats();
    let got = (e.slice(0, 2..) - e.slice(0, 1..n - 1) + e.slice(0, ..n - 2)).to_vec().unwrap();
    assert_eq!(delta(before, ctx.jit().stats()).elementwise_launches, 1);
    let want: Vec<f32> = (0..n - 2).map(|i| host[i + 2] - host[i + 1] + host[i]).collect();
    assert_close(&got, &want, 1e-5, 1e-6);
}

/// A 40-operation expression of `x` and its host values.
fn long_expression(x: &Array<f32>, xs: &[f32]) -> (Array<f32>, Vec<f32>) {
    let mut e = x.clone();
    let mut host = xs.to_vec();
    for _ in 0..20 {
        e = &e * 0.9 + 0.2;
        for v in &mut host {
            *v = *v * 0.9 + 0.2;
        }
    }
    (e, host)
}

#[test]
fn prohibitive_launches_keep_a_view_read_at_three_offsets_in_one_kernel() {
    let ctx = ctx();
    ctx.set_cost_model(CostModel { launch_us: 1e6, ..CostModel::TEST });
    let n = 1 << 16;
    let xs = values(n, 101, 0.5, 2.0);
    let x = f32s(&ctx, &xs);
    let (e, host) = long_expression(&x, &xs);

    let before = ctx.jit().stats();
    let got = (e.slice(0, 0..n - 2) + e.slice(0, 1..n - 1) + e.slice(0, 2..n)).to_vec().unwrap();
    let launches = delta(before, ctx.jit().stats()).elementwise_launches;
    assert_eq!(launches, 1, "{:?}", ctx.jit().last_judgement());
    let want: Vec<f32> = (0..n - 2).map(|i| host[i] + host[i + 1] + host[i + 2]).collect();
    assert_close(&got, &want, 1e-4, 1e-5);
}

#[test]
fn prohibitive_arithmetic_stores_a_view_read_at_three_offsets() {
    let ctx = ctx();
    ctx.set_cost_model(CostModel { f32_ops_per_ns: 1e-6, ..CostModel::TEST });
    // Staging the expression once per cell would win on this model too; this is about the
    // choice between recomputing and storing.
    ctx.jit().set_staging(false);
    ctx.jit().record_splits();
    let n = 1 << 16;
    let xs = values(n, 102, 0.5, 2.0);
    let x = f32s(&ctx, &xs);
    let (e, host) = long_expression(&x, &xs);

    let before = ctx.jit().stats();
    let got = (e.slice(0, 0..n - 2) + e.slice(0, 1..n - 1) + e.slice(0, 2..n)).to_vec().unwrap();
    let launches = delta(before, ctx.jit().stats()).elementwise_launches;
    assert_eq!(launches, 2, "{:?}", ctx.jit().last_judgement());
    let splits = ctx.jit().take_splits();
    assert!(splits.iter().any(|split| split.reason == crate::profile::SplitReason::Materialised { instances: 3 }), "{splits:?}");
    let want: Vec<f32> = (0..n - 2).map(|i| host[i] + host[i + 1] + host[i + 2]).collect();
    assert_close(&got, &want, 1e-4, 1e-5);
}

/// A cost model on which arithmetic is slow enough for computing an expression once per cell to
/// beat recomputing it at every read, but a second launch still costs more than the recomputation
/// it saves over staging.
fn arithmetic_bound() -> CostModel {
    CostModel { f32_ops_per_ns: 10.0, ..CostModel::TEST }
}

/// `e` read through an edge pad at the five offsets of the five-point stencil of an
/// `nx x ny` array.
fn five_point(e: &Array<f32>, nx: usize, ny: usize) -> Array<f32> {
    use crate::view::PadMode;
    let p = e.pad([(1, 1), (1, 1)], PadMode::Edge);
    let at = |x: usize, y: usize| p.slice(0, x..x + nx).slice(1, y..y + ny);
    at(0, 1) + at(2, 1) + at(1, 0) + at(1, 2) - at(1, 1) * 4.0
}

/// The host values of [`five_point`] of `e`.
fn five_point_host(e: &[f32], nx: usize, ny: usize) -> Vec<f32> {
    let at = |x: isize, y: isize| e[x.clamp(0, nx as isize - 1) as usize + nx * y.clamp(0, ny as isize - 1) as usize];
    let mut out = Vec::with_capacity(nx * ny);
    for y in 0..ny as isize {
        for x in 0..nx as isize {
            out.push(at(x - 1, y) + at(x + 1, y) + at(x, y - 1) + at(x, y + 1) - at(x, y) * 4.0);
        }
    }
    out
}

/// A 2-D array of random values, a 40-operation expression of it and the expression's host values.
fn expression_2d(ctx: &Context, nx: usize, ny: usize, seed: u64) -> (Array<f32>, Vec<f32>) {
    let xs = values(nx * ny, seed, 0.2, 1.2);
    let x = Array::from_slice_on(ctx, &xs, [nx, ny]).unwrap();
    let mut e = x.clone();
    let mut host = xs;
    for _ in 0..10 {
        e = (&e * 0.9 + 0.2).sqrt() * 1.1 - 0.05;
        for v in &mut host {
            *v = (*v * 0.9 + 0.2).sqrt() * 1.1 - 0.05;
        }
    }
    (e, host)
}

/// Runs the five-point stencil of an expression over `nx x ny` cells with the staging on and
/// checks the kernel stages the padded expression, one launch, against the host.
fn stencil5_staged(nx: usize, ny: usize, seed: u64) {
    let ctx = ctx();
    ctx.set_cost_model(arithmetic_bound());
    let (e, host) = expression_2d(&ctx, nx, ny, seed);
    let before = ctx.jit().stats();
    let got = five_point(&e, nx, ny).to_vec().unwrap();
    let launches = delta(before, ctx.jit().stats()).elementwise_launches;
    assert_eq!(launches, 1, "{:?}", ctx.jit().last_judgement());
    let staging = ctx.jit().last_staging().unwrap_or_else(|| panic!("not staged: {:?}", ctx.jit().last_judgement()));
    // `e` read through the clamping pad is recomputed per cell of the padded expression's region.
    assert_eq!(staging.subs.len(), 1, "{staging:?}");
    let [tx, ty] = staging.tile;
    assert_eq!((staging.subs[0].lo, staging.subs[0].hi), ([0, 0], [tx as i32 + 1, ty as i32 + 1]), "{staging:?}");
    assert_eq!(staging.subs[0].dims, [nx as u32 + 2, ny as u32 + 2]);
    assert_close(&got, &five_point_host(&host, nx, ny), 1e-4, 1e-4);
}

#[test]
fn a_five_point_stencil_of_an_expression_stages_the_padded_expression() {
    stencil5_staged(64, 48, 201);
}

#[test]
fn staging_covers_tiles_that_do_not_divide_the_array() {
    stencil5_staged(61, 47, 202);
}

#[test]
fn the_stencil_without_staging_is_one_launch_with_the_same_values() {
    let ctx = ctx();
    ctx.jit().set_staging(false);
    let (nx, ny) = (64, 48);
    // Small enough for the lowered program to stay under every device's instruction cap.
    let xs = values(nx * ny, 201, 0.2, 1.2);
    let e = (&Array::from_slice_on(&ctx, &xs, [nx, ny]).unwrap() * 1.5 + 0.25).sqrt();
    let host: Vec<f32> = xs.iter().map(|v| (v * 1.5 + 0.25).sqrt()).collect();
    let before = ctx.jit().stats();
    let got = five_point(&e, nx, ny).to_vec().unwrap();
    assert_eq!(delta(before, ctx.jit().stats()).elementwise_launches, 1, "{:?}", ctx.jit().last_judgement());
    assert_eq!(ctx.jit().last_staging(), None);
    assert_close(&got, &five_point_host(&host, nx, ny), 1e-4, 1e-4);
}

#[test]
fn a_1d_expression_read_at_three_offsets_is_staged() {
    let ctx = ctx();
    ctx.set_cost_model(arithmetic_bound());
    let n = 1000;
    let xs = values(n, 203, 0.5, 2.0);
    let x = f32s(&ctx, &xs);
    let (e, host) = long_expression(&x, &xs);
    let before = ctx.jit().stats();
    let got = (e.slice(0, 0..n - 2) + e.slice(0, 1..n - 1) + e.slice(0, 2..n)).to_vec().unwrap();
    assert_eq!(delta(before, ctx.jit().stats()).elementwise_launches, 1, "{:?}", ctx.jit().last_judgement());
    let staging = ctx.jit().last_staging().unwrap_or_else(|| panic!("not staged: {:?}", ctx.jit().last_judgement()));
    assert_eq!(staging.dims, [n as u32 - 2, 1]);
    assert_eq!(staging.subs.len(), 1);
    assert_eq!((staging.subs[0].lo, staging.subs[0].hi), ([0, 0], [staging.tile[0] as i32 + 1, 0]));
    let want: Vec<f32> = (0..n - 2).map(|i| host[i] + host[i + 1] + host[i + 2]).collect();
    assert_close(&got, &want, 1e-4, 1e-5);
}

#[test]
fn a_df32_expression_read_at_five_offsets_is_staged() {
    use crate::view::PadMode;
    use crate::Df32;
    let ctx = ctx();
    ctx.set_cost_model(arithmetic_bound());
    let (nx, ny) = (64, 48);
    let xs = values(nx * ny, 204, 0.2, 1.2);
    let mut e = Array::from_slice_on(&ctx, &xs, [nx, ny]).unwrap().to_df32();
    let mut host: Vec<f64> = xs.iter().map(|&v| f64::from(v)).collect();
    for _ in 0..6 {
        e = (&e * Df32::from(0.9) + Df32::from(0.2)).sqrt();
        for v in &mut host {
            *v = (*v * 0.9 + 0.2).sqrt();
        }
    }
    let p = e.pad([(1, 1), (1, 1)], PadMode::Edge);
    let at = |x: usize, y: usize| p.slice(0, x..x + nx).slice(1, y..y + ny);
    let r = at(0, 1) + at(2, 1) + at(1, 0) + at(1, 2) - at(1, 1) * Df32::from(4.0);
    let before = ctx.jit().stats();
    let got: Vec<f64> = r.to_vec().unwrap().into_iter().map(Df32::to_f64).collect();
    assert_eq!(delta(before, ctx.jit().stats()).elementwise_launches, 1, "{:?}", ctx.jit().last_judgement());
    assert_eq!(ctx.jit().last_staging().map(|staging| staging.subs.len()), Some(1), "{:?}", ctx.jit().last_judgement());
    let hat = |x: isize, y: isize| host[x.clamp(0, nx as isize - 1) as usize + nx * y.clamp(0, ny as isize - 1) as usize];
    for y in 0..ny {
        for x in 0..nx {
            let (xi, yi) = (x as isize, y as isize);
            let want = hat(xi - 1, yi) + hat(xi + 1, yi) + hat(xi, yi - 1) + hat(xi, yi + 1) - 4.0 * hat(xi, yi);
            let g = got[x + nx * y];
            assert!((g - want).abs() <= 1e-6, "cell ({x}, {y}): got {g}, want {want}");
        }
    }
}

#[test]
fn a_staged_expression_read_by_another_staged_expression_is_computed_a_level_earlier() {
    let ctx = ctx();
    ctx.set_cost_model(arithmetic_bound());
    let n = 1000;
    let xs = values(n, 205, 0.5, 2.0);
    let x = f32s(&ctx, &xs);
    let (p, host) = long_expression(&x, &xs);
    // A flux of two reads of `p`, and the difference of two reads of the flux.
    let g = |a: Array<f32>| &a * &a * 0.5 + &a;
    let m = n - 1;
    let f = g(p.slice(0, 0..m)) - g(p.slice(0, 1..m + 1));
    let r = f.slice(0, 0..m - 1) + f.slice(0, 1..m);
    let before = ctx.jit().stats();
    let got = r.to_vec().unwrap();
    assert_eq!(delta(before, ctx.jit().stats()).elementwise_launches, 1, "{:?}", ctx.jit().last_judgement());
    let staging = ctx.jit().last_staging().unwrap_or_else(|| panic!("not staged: {:?}", ctx.jit().last_judgement()));
    assert_eq!(staging.subs.len(), 2, "{staging:?}");
    assert_eq!(staging.levels, vec![0, 1, 2], "{staging:?}");
    let gh = |a: f32| a * a * 0.5 + a;
    let fh: Vec<f32> = (0..m).map(|i| gh(host[i]) - gh(host[i + 1])).collect();
    let want: Vec<f32> = (0..m - 1).map(|i| fh[i] + fh[i + 1]).collect();
    assert_close(&got, &want, 1e-3, 1e-4);
}

#[test]
fn cache_evaluates_an_expression_once_for_every_later_read() {
    let ctx = ctx();
    let n = 400;
    let xs = values(n, 99, 0.5, 2.0);
    let x = f32s(&ctx, &xs);
    let host: Vec<f32> = xs.iter().map(|v| (v * 2.0 + 1.0).sqrt()).collect();

    let before = ctx.jit().stats();
    let f = (&x * 2.0 + 1.0).sqrt().cache();
    assert_eq!(delta(before, ctx.jit().stats()).elementwise_launches, 1);
    let got = (f.slice(0, 1..) - f.slice(0, ..n - 1) + f.slice(0, 1..)).to_vec().unwrap();
    // The expression ran once; the readers of `f` are one more kernel.
    assert_eq!(delta(before, ctx.jit().stats()).elementwise_launches, 2);
    let want: Vec<f32> = (0..n - 1).map(|i| host[i + 1] - host[i] + host[i + 1]).collect();
    assert_close(&got, &want, 1e-5, 1e-6);
}

#[test]
fn a_scalar_operand_shares_one_kernel_whatever_its_value() {
    let ctx = ctx();
    let xs = values(64, 7, 0.0, 1.0);
    let x = f32s(&ctx, &xs);
    let s = crate::scalar::Scalar::new(2.0f32);
    let before = ctx.jit().stats();
    let a = &x * &s;
    s.set(3.0);
    let b = &x * &s;
    let (a, b) = (a.to_vec().unwrap(), b.to_vec().unwrap());
    assert_eq!(delta(before, ctx.jit().stats()).built, 1);
    assert_eq!(a, xs.iter().map(|v| v * 2.0).collect::<Vec<_>>());
    assert_eq!(b, xs.iter().map(|v| v * 3.0).collect::<Vec<_>>());
}

#[test]
fn a_reduction_used_in_an_expression_is_one_reduce_launch_and_no_readback() {
    let ctx = ctx();
    let xs = values(500, 8, 0.5, 1.5);
    let x = f32s(&ctx, &xs);
    let before = ctx.jit().stats();
    let total = (&x * 2.0).sum();
    // Used as a broadcast operand: recorded into a buffer, nothing is read back.
    let normalized = &x / &total;
    let d = delta(before, ctx.jit().stats());
    assert_eq!(d.reduce_launches, 1);
    let sum: f32 = xs.iter().map(|v| v * 2.0).sum();
    let want: Vec<f32> = xs.iter().map(|v| v / sum).collect();
    assert_close(&normalized.to_vec().unwrap(), &want, 1e-4, 1e-7);
    assert_eq!(delta(before, ctx.jit().stats()).reduce_launches, 1);
}

#[test]
fn a_cached_step_is_traced_once_and_builds_no_kernels_per_step() {
    let ctx = ctx();
    let x = f32s(&ctx, &values(128, 21, 0.0, 1.0));
    let calls = std::cell::Cell::new(0);
    let step = |[u]: &[Array<f32>; 1]| {
        calls.set(calls.get() + 1);
        let smooth = (u * 0.5 + 0.1).cache();
        [smooth.shift(1) * 0.25 + smooth.shift(-1) * 0.25 + u * 0.5]
    };
    let [short] = crate::iterate_all([&x], 5, step);
    short.eval().unwrap();
    assert_eq!(calls.get(), 1);
    let before = ctx.jit().stats();
    let [long] = crate::iterate_all([&x], 50, step);
    long.eval().unwrap();
    assert_eq!(calls.get(), 2, "one trace per call");
    assert_eq!(delta(before, ctx.jit().stats()).built, 0, "the replayed kernels are reused");
    // Same values as a host loop.
    let mut want = &x * 1.0;
    for _ in 0..50 {
        let [next] = step(&[want]);
        want = next;
    }
    assert_close(&long.to_vec().unwrap(), &want.to_vec().unwrap(), 1e-5, 1e-6);
}

#[test]
fn nested_views_of_a_lazy_expression_are_one_kernel_computing_each_index_once() {
    let ctx = ctx();
    let n = 20;
    let xs = values(n, 101, 0.5, 2.0);
    let x = f32s(&ctx, &xs);
    let e = (&x * 2.0 + 1.0).sqrt();
    let f = e.slice(0, 1..) - e.slice(0, ..n - 1);
    let g = f.slice(0, 1..) * f.slice(0, ..n - 2) + f.slice(0, 1..);

    let before = ctx.jit().stats();
    let got = g.to_vec().unwrap();
    assert_eq!(delta(before, ctx.jit().stats()).elementwise_launches, 1);

    let eh: Vec<f32> = xs.iter().map(|v| (v * 2.0 + 1.0).sqrt()).collect();
    let fh: Vec<f32> = (0..n - 1).map(|i| eh[i + 1] - eh[i]).collect();
    let want: Vec<f32> = (0..n - 2).map(|i| fh[i + 1] * fh[i] + fh[i + 1]).collect();
    assert_close(&got, &want, 1e-5, 1e-6);

    // `e` is read at the offsets 2, 1 (twice: `f` shifted by 1 then `e` by 0, and `f` by 0 then
    // `e` by 1: the same context) and 0, so its three operations exist at 3 indices, not 4:
    // 9. `f` (its subtraction) exists at the 2 offsets `g` reads it at: 2. `g` adds the product
    // and the sum: 2. The shifts themselves are aliases of the values they read.
    assert_eq!(ctx.jit().last_inlined_len(), 3 * 3 + 2 + 2);
}

#[test]
fn a_kernel_over_the_inlining_cap_is_cut_by_evaluating_a_sub_program() {
    use crate::profile::SplitReason;
    let ctx = ctx();
    ctx.jit().set_max_inlined(16);
    // Staged, the three levels fit the cap; the cap is what cuts the program otherwise.
    ctx.jit().set_staging(false);
    ctx.jit().record_splits();
    let n = 64;
    let xs = values(n, 102, 0.5, 2.0);
    let x = f32s(&ctx, &xs);
    // Six operations, read at 7 offsets by three levels of three-point stencils.
    let e = (((&x * 1.5 + 0.5).sqrt() * 2.0) - 0.25).abs();
    let level = |a: &Array<f32>| {
        let m = a.shape().len();
        a.slice(0, ..m - 2) + a.slice(0, 1..m - 1) + a.slice(0, 2..)
    };
    let c = level(&level(&level(&e)));

    let before = ctx.jit().stats();
    let got = c.to_vec().unwrap();
    assert!(delta(before, ctx.jit().stats()).elementwise_launches > 1);

    let eh: Vec<f32> = xs.iter().map(|v| (((v * 1.5 + 0.5).sqrt() * 2.0) - 0.25).abs()).collect();
    let stencil = |a: &[f32]| -> Vec<f32> { (0..a.len() - 2).map(|i| a[i] + a[i + 1] + a[i + 2]).collect() };
    assert_close(&got, &stencil(&stencil(&stencil(&eh))), 1e-4, 1e-5);
    let splits = ctx.jit().take_splits();
    assert!(splits.iter().any(|split| matches!(split.reason, SplitReason::Inlined { limit: 16, .. })), "{splits:?}");
}

#[test]
fn views_with_every_edge_of_a_lazy_expression_nest() {
    use crate::view::PadMode;
    let ctx = ctx();
    let n = 10;
    let xs = values(n, 103, 0.5, 2.0);
    let x = f32s(&ctx, &xs);
    let e = &x * 2.0 + 1.0;
    let eh: Vec<f32> = xs.iter().map(|v| v * 2.0 + 1.0).collect();
    let (ni, mirror) = (n as isize, |c: isize| if c < 0 { -c - 1 } else if c >= n as isize { 2 * n as isize - 1 - c } else { c });
    let a_val = |j: usize| eh[(j as isize - 1).clamp(0, ni - 1) as usize];
    let b_val = |j: usize| eh[mirror(j as isize - 2) as usize];

    let a = e.pad([(1, 2)], PadMode::Edge);
    let b = e.pad([(2, 1)], PadMode::Mirror);
    let sum = a.slice(0, 1..n + 2).slice(0, 1..n) + b.slice(0, ..n + 1).slice(0, 2..n + 1);
    let r = sum.shift_fill(2, 9.0);

    let before = ctx.jit().stats();
    let got = r.to_vec().unwrap();
    assert_eq!(delta(before, ctx.jit().stats()).elementwise_launches, 1);
    let want: Vec<f32> = (0..n - 1).map(|m| if m >= 2 { a_val(m) + b_val(m) } else { 9.0 }).collect();
    assert_close(&got, &want, 1e-5, 1e-6);
}

#[test]
fn a_gather_of_a_lazy_expression_of_a_lazy_slice() {
    let ctx = ctx();
    let n = 16;
    let xs = values(n, 104, -1.0, 1.0);
    let x = f32s(&ctx, &xs);
    let e = x.slice(0, 2..) * 3.0 + 1.0;
    let idx_host = [3u32, 0, 0, 99, 13, 7];
    let idx = Array::from_slice_on(&ctx, &idx_host, idx_host.len()).unwrap();
    let t = e.take(&idx);

    let before = ctx.jit().stats();
    let got = t.to_vec().unwrap();
    assert_eq!(delta(before, ctx.jit().stats()).elementwise_launches, 1);
    let want: Vec<f32> = idx_host.iter().map(|&i| xs[2 + (i as usize).min(n - 3)] * 3.0 + 1.0).collect();
    assert_close(&got, &want, 1e-5, 1e-6);
}

#[test]
fn a_reduction_over_nested_lazy_views_is_one_pass_and_no_store_kernel() {
    let ctx = ctx();
    let n = 1000;
    let xs = values(n, 105, -1.0, 1.0);
    let x = f32s(&ctx, &xs);
    let f = &x * 2.0 + 1.0;
    let g = f.slice(0, 1..) - f.slice(0, ..n - 1);
    let m = (g.slice(0, 1..) - g.slice(0, ..n - 2)).abs().max();

    let before = ctx.jit().stats();
    let got = m.item().unwrap();
    let after = delta(before, ctx.jit().stats());
    assert_eq!((after.reduce_launches, after.elementwise_launches), (1, 0));
    let fh: Vec<f32> = xs.iter().map(|v| v * 2.0 + 1.0).collect();
    let gh: Vec<f32> = (0..n - 1).map(|i| fh[i + 1] - fh[i]).collect();
    let want = (0..n - 2).map(|i| (gh[i + 1] - gh[i]).abs()).fold(0.0f32, f32::max);
    assert!((got - want).abs() <= 1e-4, "{got} vs {want}");
}

#[test]
fn a_padded_record_assigned_through_launches_nothing_until_it_is_read() {
    use crate::fields::Fields;
    use crate::view::PadMode;
    let ctx = ctx();
    let (ni, nj) = (8usize, 5usize);
    let data: Vec<Vec<f32>> = (0..4).map(|k| values(ni * nj, 110 + k, -1.0, 1.0)).collect();
    let q: [Array<f32>; 4] = std::array::from_fn(|k| Array::from_slice_on(&ctx, &data[k], [ni, nj]).unwrap());
    let p: [Array<f32>; 4] = q.map(|a| &a * 2.0 + 1.0);
    let w = p.pad([(2, 0)], PadMode::Edge);
    let ghost: [Array<f32>; 4] = std::array::from_fn(|k| w[k].slice(0, 2..4).flip(0) * -1.0);

    let before = ctx.jit().stats();
    w.slice(0, ..2).assign(ghost).unwrap();
    // The padded fields are still expressions: the write launches nothing.
    assert_eq!(delta(before, ctx.jit().stats()).elementwise_launches, 0);
    for k in 0..4 {
        let got = w[k].to_vec().unwrap();
        let p_at = |x: usize, y: usize| data[k][x.saturating_sub(2) + ni * y] * 2.0 + 1.0;
        for y in 0..nj {
            for x in 0..ni + 2 {
                let want = match x {
                    0 => -p_at(3, y),
                    1 => -p_at(2, y),
                    _ => p_at(x, y),
                };
                assert!((got[x + (ni + 2) * y] - want).abs() <= 1e-5, "field {k} ({x}, {y})");
            }
        }
    }
}

#[test]
fn a_view_that_grows_beyond_one_buffer_evaluates_the_lazy_expression_it_reads_first() {
    use crate::view::PadMode;
    let ctx = Context::with_options(crate::ContextOptions::default().chunk_bytes(1024)).unwrap();
    let xs = values(100, 106, -1.0, 1.0);
    let x = f32s(&ctx, &xs);
    // 100 elements fit one buffer, the 500 of the padded array do not.
    let p = x.pad([(0, 400)], PadMode::Edge) * 2.0 + 1.0;
    let got = p.to_vec().unwrap();
    let want: Vec<f32> = (0..500).map(|j| xs[j.min(99)] * 2.0 + 1.0).collect();
    assert_close(&got, &want, 1e-6, 1e-6);

    let e = &x * 3.0 - 1.0;
    let q = e.pad([(0, 400)], PadMode::Edge) + 1.0;
    let got = q.to_vec().unwrap();
    let want: Vec<f32> = (0..500).map(|j| xs[j.min(99)] * 3.0 - 1.0 + 1.0).collect();
    assert_close(&got, &want, 1e-6, 1e-6);
}

/// A ghost-cell style expression of `len` elements: `depth` elementwise steps over `x`.
fn long_chain(x: &Array<f32>, depth: usize) -> Array<f32> {
    let mut acc = x * 1.5;
    for k in 0..depth {
        acc = (&acc * 0.99 + x) * 0.5 - (k as f32) * 0.001;
    }
    acc
}

fn long_chain_host(x: &[f32], depth: usize) -> Vec<f32> {
    x.iter()
        .map(|&v| {
            let mut acc = v * 1.5;
            for k in 0..depth {
                acc = (acc * 0.99 + v) * 0.5 - (k as f32) * 0.001;
            }
            acc
        })
        .collect()
}

#[test]
fn concat_of_lazy_parts_is_one_kernel() {
    let ctx = ctx();
    let (xs, ys) = (values(30, 301, -1.0, 1.0), values(20, 302, -1.0, 1.0));
    let (x, y) = (f32s(&ctx, &xs), f32s(&ctx, &ys));

    // Small expressions, and one with more operations than a shifted read recomputes.
    for depth in [1, 3, 12] {
        let before = ctx.jit().stats();
        let joined = crate::concat(0, [&long_chain(&x, depth), &(long_chain(&y, depth) * 2.0)]);
        let got = joined.to_vec().unwrap();
        assert_eq!(delta(before, ctx.jit().stats()).elementwise_launches, 1, "depth {depth}");
        let mut want = long_chain_host(&xs, depth);
        want.extend(long_chain_host(&ys, depth).iter().map(|v| v * 2.0));
        assert_close(&got, &want, 1e-5, 1e-5);
    }

    // A part that is itself a stencil of a lazy expression.
    let before = ctx.jit().stats();
    let stencil = (&x * 2.0).shift(1) + (&x * 2.0).shift(-1);
    let joined = crate::concat(0, [&stencil, &(&y + 1.0)]);
    let got = joined.to_vec().unwrap();
    assert_eq!(delta(before, ctx.jit().stats()).elementwise_launches, 1);
    let n = xs.len();
    let mut want: Vec<f32> = (0..n).map(|i| xs[(i + n - 1) % n] * 2.0 + xs[(i + 1) % n] * 2.0).collect();
    want.extend(ys.iter().map(|v| v + 1.0));
    assert_close(&got, &want, 1e-5, 1e-5);

    // Three parts along the second axis, parts that are views of lazy expressions, and the join
    // read through a shift of its own.
    let a = Array::from_slice_on(&ctx, &values(12, 303, -1.0, 1.0), [4, 3]).unwrap();
    let before = ctx.jit().stats();
    let (p, q, r) = (&a * 2.0 + 1.0, a.slice(1, ..2).flip(0) * -1.0, a.transpose().slice(0, ..2).transpose() + 0.5);
    let joined = crate::concat(1, [&p, &q, &r]);
    let out = joined.slice(1, 1..7) * 3.0;
    out.eval().unwrap();
    out.to_vec().unwrap();
    assert_eq!(delta(before, ctx.jit().stats()).elementwise_launches, 1);
}

#[test]
fn a_clone_of_a_lazy_array_shares_its_evaluation() {
    let ctx = ctx();
    let xs = values(300, 310, -1.0, 1.0);
    let x = f32s(&ctx, &xs);
    let want: Vec<f32> = xs.iter().map(|&v| v * 2.0 + 1.0).collect();

    let e = &x * 2.0 + 1.0;
    let c = e.clone();
    let before = ctx.jit().stats();
    assert_close(&c.to_vec().unwrap(), &want, 1e-6, 1e-6);
    assert_close(&e.to_vec().unwrap(), &want, 1e-6, 1e-6);
    assert_eq!(delta(before, ctx.jit().stats()).elementwise_launches, 1, "evaluated once, by whichever clone came first");

    // Both clones in one expression are one operand: still one kernel, computing the node once.
    let e = (&x * 3.0).abs().sqrt();
    let c = e.clone();
    let before = ctx.jit().stats();
    let sum = &e + &c;
    let got = sum.to_vec().unwrap();
    assert_eq!(delta(before, ctx.jit().stats()).elementwise_launches, 1);
    let want: Vec<f32> = xs.iter().map(|&v| 2.0 * (v * 3.0).abs().sqrt()).collect();
    assert_close(&got, &want, 1e-5, 1e-6);
}

#[test]
fn an_input_buffer_is_not_reused_in_place_while_a_clone_of_its_array_lives() {
    let ctx = ctx();
    let xs = values(400, 311, -1.0, 1.0);
    let x = f32s(&ctx, &xs);
    let doubled: Vec<f32> = xs.iter().map(|&v| v * 2.0).collect();

    // Control: the only holder of an evaluated buffer lets the kernel that consumes it write over it.
    let t = &x * 2.0;
    t.eval().unwrap();
    let buffer = t.buffer();
    let u = &t * 3.0;
    drop(t);
    u.to_vec().unwrap();
    assert_eq!(u.buffer(), buffer, "without a clone the buffer is reused");

    // A live clone (of an evaluated lazy array, and of a valid one) forbids it.
    let t = &x * 2.0;
    t.eval().unwrap();
    let buffer = t.buffer();
    let kept = t.clone();
    let u = &t * 3.0;
    drop(t);
    let got = u.to_vec().unwrap();
    assert_ne!(u.buffer(), buffer);
    assert_close(&kept.to_vec().unwrap(), &doubled, 1e-6, 1e-6);
    let want: Vec<f32> = xs.iter().map(|&v| v * 6.0).collect();
    assert_close(&got, &want, 1e-6, 1e-6);

    let v = f32s(&ctx, &xs);
    let kept = v.clone();
    let u = &v * 3.0;
    drop(v);
    u.to_vec().unwrap();
    assert_eq!(kept.to_vec().unwrap(), xs);
    assert_ne!(u.buffer(), kept.buffer());
}

type F = Array<f32>;

/// A chain of `depth` mixed operations over `xs`.
fn mix(xs: &[&F], seed: usize, depth: usize) -> F {
    let mut acc = xs[seed % xs.len()] * 1.0;
    for k in 0..depth {
        let y = xs[(seed + k + 1) % xs.len()];
        acc = match k % 4 {
            0 => &acc * y + 0.1,
            1 => (&acc - y).abs() + 0.2,
            2 => &acc / (y + 1.5),
            _ => &acc + y * 0.5,
        };
    }
    acc
}

/// A solver step staged with `cache()` over 14 distinct buffers, on a device that binds 16 per kernel,
/// stays at the launches its stages need (4 `l`, 4 `r`, 4 `f`, and 3 `s` when they are staged too). Before,
/// the expression of `f` was cut for "21 input buffers > 16" and became 68 launches: the nodes it shares with
/// the other `f` were evaluated first, which made each of them one more buffer to bind (see
/// `program::evaluation_fits`).
#[test]
fn cached_arrays_do_not_count_the_buffers_below_them() {
    let ctx = ctx();
    ctx.jit().set_max_inputs(16);
    let n = 2000usize;
    let w: Vec<F> = (0..4)
        .map(|k| f32s(&ctx, &(0..n + 4).map(|i| 1.0 + 0.1 * ((i + k) as f32 * 0.37).sin()).collect::<Vec<_>>()))
        .collect();
    let (nx, nr) = (f32s(&ctx, &vec![0.5f32; n]), f32s(&ctx, &vec![0.7f32; n]));
    for cache_s in [false, true] {
        let build = || {
            let m1: Vec<F> = w.iter().map(|a| a.slice(0, 1..n + 1)).collect();
            let p0: Vec<F> = w.iter().map(|a| a.slice(0, 2..n + 2)).collect();
            let views: Vec<&F> = m1.iter().chain(p0.iter()).collect();
            let l: Vec<F> = (0..4).map(|k| mix(&views, k, 12).cache()).collect();
            let r: Vec<F> = (0..4).map(|k| mix(&views, k + 4, 12).cache()).collect();
            let lr: Vec<&F> = l.iter().chain(r.iter()).chain([&nx, &nr]).collect();
            let s: Vec<F> = (0..3).map(|k| if cache_s { mix(&lr, k, 16).cache() } else { mix(&lr, k, 16) }).collect();
            let lrs: Vec<&F> = lr.iter().copied().chain(s.iter()).collect();
            let f: Vec<F> = (0..4).map(|k| mix(&lrs, k, 24)).collect();
            let t = (&p0[3] - &m1[3]).abs() / m1[3].minimum(&p0[3]);
            f.iter().map(|f| (f * &t).cache()).collect::<Vec<_>>()
        };
        drop(build());
        ctx.sync().unwrap();
        ctx.jit().record_splits();
        let before = ctx.jit().stats();
        let out = build();
        ctx.sync().unwrap();
        let launches = delta(before, ctx.jit().stats()).elementwise_launches;
        let splits = ctx.jit().take_splits();
        // 4 l, 4 r and the 4 f; the 3 s as well when they are cached.
        let want = if cache_s { 15 } else { 12 };
        assert_eq!(launches, want, "s cached: {cache_s}, splits: {splits:?}");
        assert!(splits.iter().all(|split| !matches!(split.reason, crate::profile::SplitReason::Inputs { .. })), "{splits:?}");
        drop(out);
    }
}

/// What an expression reads is counted from what is pending now, not from what its operands read when
/// they were created: `p` was recorded over the 12 buffers of `x`, `x` was evaluated since, so `p + other`
/// reads one buffer of `x` and 8 of `other` and is one kernel on a device that binds 16.
#[test]
fn buffers_below_an_evaluated_part_of_an_expression_are_not_counted() {
    let ctx = ctx();
    ctx.jit().set_max_inputs(16);
    let n = 1000usize;
    let data: Vec<Vec<f32>> = (0..20).map(|i| values(n, 500 + i as u64, 0.5, 1.5)).collect();
    let b: Vec<Array<f32>> = data.iter().map(|d| f32s(&ctx, d)).collect();
    let sum = |arrays: &[Array<f32>]| arrays.iter().skip(1).fold(&arrays[0] * 1.0, |acc, a| acc + a);

    let x = sum(&b[..12]);
    let p = &x * 2.0;
    let before = ctx.jit().stats();
    x.eval().unwrap();
    let other = sum(&b[12..]);
    let z = &p + &other;
    let got = z.to_vec().unwrap();
    // `x` once, then `p + other` in one kernel.
    assert_eq!(delta(before, ctx.jit().stats()).elementwise_launches, 2);
    let want: Vec<f32> = (0..n)
        .map(|i| 2.0 * (0..12).map(|k| data[k][i]).sum::<f32>() + (12..20).map(|k| data[k][i]).sum::<f32>())
        .collect();
    assert_close(&got, &want, 1e-5, 1e-5);
}

/// `flush`, `sync` and the evaluation of requested arrays only compute what the calling thread
/// created: another thread may be flattening it that moment, or tracing it into a recording whose
/// buffers only the recording thread's launches may fill. Whoever reads an array computes it.
#[test]
fn sweeps_over_every_lazy_array_leave_the_arrays_of_other_threads_alone() {
    let ctx = ctx();
    let xs = values(500, 7, -1.0, 1.0);
    let x = f32s(&ctx, &xs);
    let lazy = &x * 2.0 + 1.0;
    let requested = &x * 3.0;
    ctx.jit().request(requested.pending().unwrap());

    let before = ctx.jit().stats();
    std::thread::scope(|scope| {
        scope.spawn(|| {
            // Its own lazy array is computed by these, the main thread's is not.
            let mine = &f32s(&ctx, &xs) * 5.0;
            ctx.flush().unwrap();
            ctx.sync().unwrap();
            ctx.jit().evaluate_requested();
            assert!(mine.pending().is_none());
        });
    });
    assert_eq!(delta(before, ctx.jit().stats()).elementwise_launches, 1, "only the thread's own array");
    assert!(lazy.pending().is_some());
    assert!(requested.pending().is_some());

    // The thread that created them computes them, here or when it reads them.
    ctx.flush().unwrap();
    assert!(lazy.pending().is_none());
    assert!(requested.pending().is_none());
    assert_close(&lazy.to_vec().unwrap(), &xs.iter().map(|v| v * 2.0 + 1.0).collect::<Vec<_>>(), 1e-6, 1e-6);
    assert_close(&requested.to_vec().unwrap(), &xs.iter().map(|v| v * 3.0).collect::<Vec<_>>(), 1e-6, 1e-6);
}

#[test]
fn a_ghost_assignment_into_a_pending_array_launches_nothing() {
    use crate::view::PadMode;
    let ctx = ctx();
    let (ni, nj) = (8usize, 5usize);
    let data = values(ni * nj, 120, -1.0, 1.0);
    let x = Array::from_slice_on(&ctx, &data, [ni, nj]).unwrap();
    let w = (&x * 2.0 + 1.0).pad([(2, 0)], PadMode::Edge);

    let before = ctx.jit().stats();
    w.slice(0, ..2).assign(w.slice(0, 2..4).flip(0) * -1.0).unwrap();
    assert_eq!(delta(before, ctx.jit().stats()).elementwise_launches, 0);

    let n = ni + 2;
    let before = ctx.jit().stats();
    let got = (w.slice(0, 1..) - w.slice(0, ..n - 1)).to_vec().unwrap();
    assert_eq!(delta(before, ctx.jit().stats()).elementwise_launches, 1);
    let p = |x: usize, y: usize| data[x.saturating_sub(2) + ni * y] * 2.0 + 1.0;
    let cell = |x: usize, y: usize| match x {
        0 => -p(3, y),
        1 => -p(2, y),
        _ => p(x, y),
    };
    let want: Vec<f32> = (0..nj).flat_map(|y| (0..n - 1).map(move |x| (x, y))).map(|(x, y)| cell(x + 1, y) - cell(x, y)).collect();
    assert_close(&got, &want, 1e-5, 1e-5);
}

#[cfg(test)]
mod euler {
    use super::*;
    use crate::fields::Fields;
    use crate::view::PadMode;

    type A = Array<f32>;

    #[derive(crate::Fields)]
    pub struct R {
        pub a: A,
        pub b: A,
        pub c: A,
        pub d: A,
    }

    fn prim(q: &R) -> R {
        let u = &q.b / &q.a;
        let v = &q.c / &q.a;
        let p = 0.4 * (&q.d - 0.5 * &q.a * (&u * &u + &v * &v));
        R { a: &q.a * 1.0, b: u, c: v, d: p }
    }

    fn cons(w: &R) -> R {
        let ke = 0.5 * &w.a * (&w.b * &w.b + &w.c * &w.c);
        R { a: &w.a * 1.0, b: &w.a * &w.b, c: &w.a * &w.c, d: &w.d / 0.4 + ke }
    }

    fn flux(w: &R) -> R {
        let q = cons(w);
        R { a: &q.b * 1.0, b: &q.b * &w.b + &w.d, c: &q.c * &w.b, d: (&q.d + &w.d) * &w.b }
    }

    fn rusanov(l: &R, r: &R) -> R {
        let speed = |w: &R| w.b.abs() + (1.4 * &w.d / &w.a).sqrt();
        let smax = speed(l).maximum(speed(r));
        let (ul, ur) = (cons(l), cons(r));
        let (fl, fr) = (flux(l), flux(r));
        0.5 * (&fl + &fr) - 0.5 * &smax * (&ur - &ul)
    }

    /// One first-order step with a mirror wall on the low side of the first axis; `cache_w`
    /// evaluates the padded state before the ghost cells are assigned.
    fn step(q: &R, ni: usize, nj: usize, cache_w: bool) -> R {
        step_cached(q, ni, nj, cache_w, false)
    }

    /// [`step`] that can also evaluate the padded state after the ghost cells were assigned.
    fn step_cached(q: &R, ni: usize, nj: usize, cache_before: bool, cache_after: bool) -> R {
        let padded = prim(q).pad([(1, 1), (1, 1)], PadMode::Edge);
        let w = if cache_before { padded.cache() } else { padded };
        let g = w.slice(0, 1..2);
        let g = R { a: &g.a * 1.0, b: -&g.b, c: &g.c * 1.0, d: &g.d * 1.0 };
        w.slice(0, ..1).assign(g).unwrap();
        let w = if cache_after { w.cache() } else { w };
        let wi = w.slice(1, 1..nj + 1);
        let fi = rusanov(&wi.slice(0, 0..ni + 1), &wi.slice(0, 1..ni + 2));
        let wj = w.slice(0, 1..ni + 1);
        let fj = rusanov(&wj.slice(1, 0..nj + 1), &wj.slice(1, 1..nj + 2));
        let r = -(fi.slice(0, 1..) - fi.slice(0, ..ni)) - (fj.slice(1, 1..) - fj.slice(1, ..nj));
        R { a: &q.a + 0.01 * &r.a, b: &q.b + 0.01 * &r.b, c: &q.c + 0.01 * &r.c, d: &q.d + 0.01 * &r.d }
    }

    fn state(ctx: &Context, ni: usize, nj: usize) -> R {
        let up = |k: u64, lo: f32, hi: f32| Array::from_slice_on(ctx, &values(ni * nj, 130 + k, lo, hi), [ni, nj]).unwrap();
        R { a: up(0, 0.8, 1.2), b: up(1, -0.3, 0.3), c: up(2, -0.3, 0.3), d: up(3, 2.0, 3.0) }
    }

    #[test]
    fn a_step_with_a_wall_condition_on_a_padded_state_is_one_launch_with_four_outputs() {
        let ctx = ctx();
        // The software rasterizer's tuning limits kernels far below what this step lowers to.
        ctx.jit().set_max_inlined(4096);
        let (ni, nj) = (16usize, 12usize);
        let q = state(&ctx, ni, nj);
        let next = step(&q, ni, nj, false);
        let before = ctx.jit().stats();
        next.eval().unwrap();
        ctx.sync().unwrap();
        assert_eq!(delta(before, ctx.jit().stats()).elementwise_launches, 1);

        // The same step with the padded state evaluated before the ghost cells are written.
        let want = step(&state(&ctx, ni, nj), ni, nj, true);
        for (got, want) in next.arrays().into_iter().zip(want.arrays()) {
            assert_close(&got.to_vec().unwrap(), &want.to_vec().unwrap(), 1e-5, 1e-5);
        }
    }

    /// Steps evaluated one after the other compile to the same kernels every time: one per step
    /// (two with the padded state evaluated before the ghost cells are read), whatever else is
    /// pending. The mask of the assigned region reads no buffer and is recomputed, not evaluated,
    /// and what the padded state's primitives are read through (nested views) belongs to the kernel.
    #[test]
    fn per_step_loops_compile_to_one_kernel_per_step_or_two_with_a_cached_padded_state() {
        let ctx = ctx();
        ctx.jit().set_max_inlined(4096);
        let (ni, nj, steps) = (24usize, 16usize, 6usize);
        for (cache, per_step) in [(false, 1u64), (true, 2)] {
            let mut q = state(&ctx, ni, nj);
            ctx.sync().unwrap();
            let before = ctx.jit().stats();
            for _ in 0..steps {
                q = step_cached(&q, ni, nj, false, cache);
                q.eval().unwrap();
            }
            ctx.sync().unwrap();
            assert_eq!(delta(before, ctx.jit().stats()).elementwise_launches, per_step * steps as u64, "cache {cache}");
        }

        // The same with the evaluation of every step postponed until the last is requested.
        let before = ctx.jit().stats();
        let mut q = state(&ctx, ni, nj);
        for _ in 0..steps {
            q = step_cached(&q, ni, nj, false, true);
            for array in q.arrays() {
                ctx.jit().request(array.pending().unwrap());
            }
        }
        ctx.jit().evaluate_requested();
        ctx.sync().unwrap();
        assert_eq!(delta(before, ctx.jit().stats()).elementwise_launches, 2 * steps as u64, "postponed");
    }

    #[test]
    fn stencil_steps_requested_one_after_the_other_are_a_chain_and_compute_in_order() {
        let ctx = ctx();
        let (ni, nj) = (16usize, 12usize);
        let mut chained = state(&ctx, ni, nj);
        let mut eager = state(&ctx, ni, nj);
        for _ in 0..3 {
            chained = step(&chained, ni, nj, false);
            chained.eval().unwrap();
            eager = step(&eager, ni, nj, true);
            eager.eval().unwrap();
            ctx.sync().unwrap();
        }
        for (got, want) in chained.arrays().into_iter().zip(eager.arrays()) {
            assert_close(&got.to_vec().unwrap(), &want.to_vec().unwrap(), 1e-4, 1e-4);
        }

        // Two steps requested while nothing ran in between: a stencil reading a postponed one.
        let first = step(&state(&ctx, ni, nj), ni, nj, false);
        ctx.jit().request(first.a.pending().unwrap());
        assert!(!ctx.jit().requested_stencil_chain());
        let second = step(&first, ni, nj, false);
        ctx.jit().request(second.a.pending().unwrap());
        assert!(ctx.jit().requested_stencil_chain());
        ctx.jit().evaluate_requested();
        ctx.sync().unwrap();
        assert!(!ctx.jit().requested_stencil_chain());
        let want = step(&step(&state(&ctx, ni, nj), ni, nj, true), ni, nj, true);
        assert_close(&second.a.to_vec().unwrap(), &want.a.to_vec().unwrap(), 1e-4, 1e-4);

        // An elementwise chain is not one that must be cut.
        let x = f32s(&ctx, &values(64, 140, 0.5, 1.5));
        let (y, z) = (&x * 2.0, &x * 3.0);
        ctx.jit().request(y.pending().unwrap());
        let z2 = &y + &z;
        ctx.jit().request(z2.pending().unwrap());
        assert!(!ctx.jit().requested_stencil_chain());
        ctx.jit().evaluate_requested();
    }
}

/// Operations without operands of their own (the same generator, say) are one node only within a thread:
/// a node shared by two threads is flattened, recorded and waited for by both.
#[test]
fn another_threads_pending_node_is_not_shared_by_hash_consing() {
    let ctx = ctx();
    let here = Array::<u32>::range_on(&ctx, 100, 0).unwrap();
    let node = std::sync::Arc::clone(here.pending().unwrap());
    // The array is gone, the node lives on in the expression that reads it.
    let user = &here + 1u32;
    drop(here);
    let again = Array::<u32>::range_on(&ctx, 100, 0).unwrap();
    assert!(std::sync::Arc::ptr_eq(&node, again.pending().unwrap()), "within a thread it is one node");
    drop(again);
    let there = std::thread::scope(|scope| scope.spawn(|| Array::<u32>::range_on(&ctx, 100, 0).unwrap()).join().unwrap());
    assert!(!std::sync::Arc::ptr_eq(&node, there.pending().unwrap()), "another thread's is its own");
    assert_eq!(user.to_vec().unwrap()[..3], [1, 2, 3]);
}

/// A lazy array created before a traced loop and read through a captured handle is evaluated
/// once, before the loop: it must not be recorded with the step (and replayed against the buffers
/// the step overwrites), nor take over its operand's buffer inside the recording.
#[test]
fn lazy_arrays_captured_by_a_traced_loop_are_not_replayed_with_it() {
    let ctx = ctx();
    let xs: Vec<f32> = (0..12).map(|i| i as f32 / 8.0 + 0.25).collect();
    let x = Array::<f32>::from_slice_on(&ctx, &xs, 12).unwrap();
    let s = Array::<f32>::from_slice_on(&ctx, &xs, 12).unwrap();
    let y = &x * 2.0;
    let mut z = Array::<f32>::from_slice_on(&ctx, &xs, 12).unwrap();
    z *= -2.0;
    let [r] = crate::iterate_all([&s], 3, |s| {
        let mut shared = x.clone();
        shared += 1.0; // overwrites x: `y` must have been computed from the old contents
        [s[0].cumsum(0) + z.cumsum(0)]
    });
    let host = |f: &dyn Fn(f32) -> f32| xs.iter().map(|&v| f(v)).collect::<Vec<_>>();
    assert_eq!(y.to_vec().unwrap(), host(&|v| v * 2.0));
    assert_eq!(x.to_vec().unwrap(), host(&|v| v + 3.0));
    assert_eq!(z.to_vec().unwrap(), host(&|v| v * -2.0));
    r.to_vec().unwrap();
}

/// An array that was only computed inside the kernel of another array (here `n`, inside the
/// kernel of `held`) is still lazy, and its operand's buffer is only held by it: a traced loop that
/// reads `n` must not evaluate it inside the recording, where its kernel would take over that
/// buffer and be replayed at every step (`n` would be squared again at every step).
#[test]
fn an_array_computed_inside_another_arrays_kernel_is_evaluated_before_a_traced_loop() {
    let ctx = ctx();
    let xs: Vec<f32> = (0..12).map(|i| i as f32 / 8.0 + 0.25).collect();
    let x = Array::<f32>::from_slice_on(&ctx, &xs, 12).unwrap();
    let mid = (&x * 2.0).maximum(1.0);
    mid.to_vec().unwrap(); // evaluated: its buffer is held by `mid` only
    let n = &mid * &mid + 1.0;
    drop(mid); // now `n` is the only holder of that buffer
    let held = &n + 1.0;
    let s = Array::<f32>::from_slice_on(&ctx, &xs, 12).unwrap();
    let [r] = crate::iterate_all([&s], 3, |s| [s[0].cumsum(0) + n.cumsum(0)]);

    let host_n: Vec<f32> = xs.iter().map(|&v| (v * 2.0).max(1.0)).map(|m| m * m + 1.0).collect();
    let cumsum = |v: &[f32]| -> Vec<f32> { v.iter().scan(0.0f32, |acc, &e| { *acc += e; Some(*acc) }).collect() };
    let mut host_r = xs.clone();
    for _ in 0..3 {
        host_r = cumsum(&host_r).iter().zip(cumsum(&host_n)).map(|(a, b)| a + b).collect();
    }
    let close = |got: Vec<f32>, want: &[f32]| {
        for (g, w) in got.iter().zip(want) {
            assert!((g - w).abs() <= 1e-4 * w.abs().max(1.0), "{got:?} != {want:?}");
        }
    };
    close(held.to_vec().unwrap(), &host_n.iter().map(|v| v + 1.0).collect::<Vec<_>>());
    close(n.to_vec().unwrap(), &host_n);
    close(r.to_vec().unwrap(), &host_r);
}

// ---- Risk areas (plan step 6d.4): supersede chains, postponed stencil chains, writes between speculated readbacks.

/// One step of a padded state whose clones are assigned through slices: the second assignment
/// supersedes the expression the first clone holds (`Held::latest`), the first clone is read after both.
fn clone_assign_step(u: &Array<f32>) -> Array<f32> {
    use crate::view::PadMode;
    let n = u.len();
    let w = u.pad([(1, 1)], PadMode::Edge);
    let (left, right) = (w.clone(), w.clone());
    left.slice(0, ..1).assign(0.5).unwrap();
    right.slice(0, n + 1..).assign(0.25).unwrap();
    (w.slice(0, 1..n + 1) * 0.5 + left.slice(0, ..n) * 0.25 + right.slice(0, 2..n + 2) * 0.25) * 0.99 + 0.001
}

fn clone_assign_host(u: &[f32]) -> Vec<f32> {
    let n = u.len();
    let w: Vec<f32> = std::iter::once(0.5).chain(u.iter().copied()).chain(std::iter::once(0.25)).collect();
    (0..n).map(|i| (0.5 * w[i + 1] + 0.25 * w[i] + 0.25 * w[i + 2]) * 0.99 + 0.001).collect()
}

/// Clones of a lazy padded state, assigned through slices inside a per-step `eval` loop (with the
/// evaluation postponed or not, and under tight node limits) and inside a traced `iterate_all`.
#[test]
fn clones_of_a_lazy_state_assigned_through_slices_in_loop_steps_keep_their_values() {
    let xs = values(40, 150, 0.0, 1.0);
    for (busy, max_nodes) in [(false, 1024), (true, 1024), (true, 4), (false, 2)] {
        let ctx = ctx();
        ctx.set_assume_busy(busy);
        ctx.jit().set_max_nodes(max_nodes);
        let mut u = f32s(&ctx, &xs);
        let mut w = xs.clone();
        for step in 0..30 {
            u = clone_assign_step(&u);
            u.eval().unwrap();
            w = clone_assign_host(&w);
            if step % 9 == 8 {
                assert_close(&u.to_vec().unwrap(), &w, 1e-4, 1e-5);
            }
        }
        assert_close(&u.to_vec().unwrap(), &w, 1e-4, 1e-5);

        let [r] = crate::iterate_all([&f32s(&ctx, &xs)], 12, |[u]| [clone_assign_step(u)]);
        let mut w = xs.clone();
        for _ in 0..12 {
            w = clone_assign_host(&w);
        }
        assert_close(&r.to_vec().unwrap(), &w, 1e-4, 1e-5);
    }
}

/// A 1-D stencil step through slices and shifts of the previous (postponed) result, two states.
fn chain_step(u: &Array<f32>, v: &Array<f32>) -> (Array<f32>, Array<f32>) {
    let nu = (u.shift(1) + u.shift(-1)) * 0.25 + u * 0.5 + v * 0.01;
    let nv = v * 0.9 + nu.slice(0, 1..).pad([(0, 1)], crate::view::PadMode::Edge) * 0.1;
    (nu, nv)
}

fn chain_host(u: &[f32], v: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let n = u.len();
    let nu: Vec<f32> = (0..n).map(|i| (u[(i + 1) % n] + u[(i + n - 1) % n]) * 0.25 + u[i] * 0.5 + v[i] * 0.01).collect();
    let nv = (0..n).map(|i| v[i] * 0.9 + nu[(i + 1).min(n - 1)] * 0.1).collect();
    (nu, nv)
}

/// With the device assumed busy every `eval` is postponed, so each step's stencil reads the previous
/// step's result while it is still pending: the chain must be cut and computed in order, however
/// the pending steps pile up, and whatever readbacks and node limits interrupt them.
#[test]
fn postponed_stencil_chains_compute_in_order() {
    let (xs, ys) = (values(48, 160, 0.0, 1.0), values(48, 161, 0.0, 1.0));
    for max_nodes in [1024, 8, 2] {
        for read_every in [0usize, 1, 5, 7] {
            let ctx = ctx();
            ctx.set_assume_busy(true);
            ctx.jit().set_max_nodes(max_nodes);
            let (mut u, mut v) = (f32s(&ctx, &xs), f32s(&ctx, &ys));
            let (mut hu, mut hv) = (xs.clone(), ys.clone());
            for step in 1..=25 {
                (u, v) = chain_step(&u, &v);
                u.eval().unwrap();
                v.eval().unwrap();
                (hu, hv) = chain_host(&hu, &hv);
                if read_every != 0 && step % read_every == 0 {
                    assert_close(&u.to_vec().unwrap(), &hu, 1e-4, 1e-5);
                }
            }
            assert_close(&u.to_vec().unwrap(), &hu, 1e-4, 1e-5);
            assert_close(&v.to_vec().unwrap(), &hv, 1e-4, 1e-5);
        }
    }
}

/// Two-dimensional steps requested while nothing ran in between with the euler step's views.
#[test]
fn postponed_2d_stencil_steps_read_through_views_compute_in_order() {
    let ctx = ctx();
    ctx.set_assume_busy(true);
    let g = f32s(&ctx, &values(24 * 20, 162, 0.0, 1.0));
    let g = g.reshape([24, 20]);
    let mut u = &g * 1.0;
    for _ in 0..30 {
        u = heat_step(&u);
        u.eval().unwrap();
    }
    let want = g.iterate(30, heat_step).to_vec().unwrap();
    assert_close(&u.to_vec().unwrap(), &want, 1e-5, 1e-5);
}

/// What happens to the input `c` of a step between two readbacks.
#[derive(Clone, Copy, Debug)]
enum Rewrite {
    Write,
    AssignSlice,
    AssignScalar,
}

/// `check_readback_loop` where `c` is rewritten in several ways at `events` (step, rewrite) once the
/// speculation has had steps to start; every error value and the final state are compared to the host.
fn check_readback_loop_with_writes(ctx: &Context, steps: usize, events: &[(usize, Rewrite)]) {
    const W: usize = 32;
    let u0 = values(W * W, 90, 0.0, 1.0);
    let mut source = values(W * W, 91, 0.0, 1.0);
    let c = Array::from_slice_on(ctx, &source, [W, W]).unwrap();
    let mut u = &Array::from_slice_on(ctx, &u0, [W, W]).unwrap() * 1.0;
    let mut cpu = u0;
    for k in 0..steps {
        for &(at, how) in events.iter().filter(|e| e.0 == k) {
            match how {
                Rewrite::Write => {
                    source = values(W * W, 92 + at as u64, 0.0, 1.0);
                    c.clone().write(&source).unwrap();
                }
                Rewrite::AssignSlice => {
                    c.slice(0, 4..8).slice(1, 2..30).assign(0.75).unwrap();
                    for i in 4..8 {
                        for j in 2..30 {
                            source[j * W + i] = 0.75; // axis 0 is the contiguous one
                        }
                    }
                }
                Rewrite::AssignScalar => {
                    c.assign(0.125).unwrap();
                    source.iter_mut().for_each(|s| *s = 0.125);
                }
            }
        }
        let lap = u.shift([1isize, 0]) + u.shift([-1isize, 0]) + u.shift([0isize, 1]) + u.shift([0isize, -1]) - &u * 4.0;
        let next = &u + lap * 0.2 + &c * 0.01;
        let err = (&next - &u).abs().max().item().unwrap();
        u = next;

        let at = |i: usize, j: usize| cpu[i % W * W + j % W];
        let mut new = vec![0.0f32; W * W];
        for i in 0..W {
            for j in 0..W {
                let lap = at(i + 1, j) + at(i + W - 1, j) + at(i, j + 1) + at(i, j + W - 1) - at(i, j) * 4.0;
                new[i * W + j] = at(i, j) + lap * 0.2 + source[i * W + j] * 0.01;
            }
        }
        let want = new.iter().zip(&cpu).map(|(n, o)| (n - o).abs()).fold(0.0, f32::max);
        assert!((err - want).abs() <= 1e-4 * want + 1e-6, "step {k} ({events:?}): err {err}, want {want}");
        cpu = new;
    }
    assert_close(&u.to_vec().unwrap(), &cpu, 1e-4, 1e-5);
}

/// A write, a slice assignment or a fill of an input of the step between two readbacks, after the
/// speculation has started answering readbacks ahead of time: the prediction made before the write
/// must not be used for the readback after it, and the speculation may start again afterwards.
#[test]
fn writes_between_speculated_readbacks_are_seen_by_the_next_readback() {
    for events in [
        &[(20, Rewrite::AssignSlice)][..],
        &[(20, Rewrite::AssignScalar)],
        &[(15, Rewrite::Write), (16, Rewrite::AssignSlice), (30, Rewrite::AssignScalar)],
        &[(12, Rewrite::AssignSlice), (13, Rewrite::AssignSlice), (14, Rewrite::Write), (15, Rewrite::Write)],
        &[(10, Rewrite::Write), (20, Rewrite::Write), (30, Rewrite::Write)],
    ] {
        let ctx = ctx();
        check_readback_loop_with_writes(&ctx, 45, events);
        assert!(ctx.jit().speculation_hits() > 0, "{events:?}: no readback was answered ahead of time");
    }
}
