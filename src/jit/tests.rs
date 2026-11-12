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
    // The first piece reads `x`, which stays alive, so it writes a new buffer; the later pieces
    // read a dying temporary and write over it, which is a different kernel. Of those, the
    // second and third full pieces share one pipeline, the remainder is another one.
    assert_eq!(d.built, 3);
    assert_eq!(d.hits, 1);

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
    let ctx = Context::new().unwrap();
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
    let ctx = Context::new().unwrap();
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
    let ctx = Context::new().unwrap();
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
    let ctx = Context::new().unwrap();
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
    // `u` is small: recomputed at the shifted elements, so the whole step is one kernel.
    assert_eq!(delta(before, ctx.jit().stats()).elementwise_launches, 1);
    let f = |j: usize| xs[j % n] * 2.0;
    let want: Vec<f32> =
        (0..n).map(|i| f(i) + (f(i + n - 1) + f(i + 1) - f(i) * 2.0) * 0.1).collect();
    assert_close(&got, &want, 1e-5, 1e-5);

    // A large expression is evaluated once for its shifts instead.
    let mut big = &x * 1.0;
    for _ in 0..20 {
        big = &big * 0.5 + 0.25;
    }
    let before = ctx.jit().stats();
    let got = (big.shift(1) - &big).to_vec().unwrap();
    assert_eq!(delta(before, ctx.jit().stats()).elementwise_launches, 2);
    let g = |v: f32| (0..20).fold(v, |a, _| a * 0.5 + 0.25);
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
fn a_life_step_recomputes_through_its_shifts() {
    let ctx = Context::new().unwrap();
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
    use crate::reduce::Reduction;
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

#[test]
fn arrays_advancing_together_are_split_into_multi_output_kernels() {
    let n = 600;
    let cxs = values(n, 20, -0.3, 0.3);
    let cys = values(n, 21, -0.3, 0.3);
    let ctx = Context::new().unwrap();
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
    assert!(launches <= 20, "{launches} kernels");
}

#[test]
fn the_node_limit_counts_distinct_nodes_only() {
    let ctx = Context::new().unwrap();
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
    let ctx = Context::new().unwrap();
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
    let ctx = Context::new().unwrap();
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
    let ctx = Context::new().unwrap();
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
    let ctx = Context::new().unwrap();
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
    let ctx = Context::new().unwrap();
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
    let ctx = Context::new().unwrap();
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
    let ctx = Context::new().unwrap();
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
    let ctx = Context::new().unwrap();
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
    let ctx = Context::new().unwrap();
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
    let ctx = Context::new().unwrap();
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
    let ctx = Context::new().unwrap();
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
