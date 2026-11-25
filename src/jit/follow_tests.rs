//! Tests of trace following (see `follow`): a loop that builds the same step again and again replays
//! its tape, every way the step can stop being the same is a divergence that changes nothing a
//! program can see, and what a followed step computes is bit-identical to what it computes without.

use super::cost::CostModel;
use super::follow::StatsSnapshot;
use crate::array::Array;
use crate::context::{Context, ContextOptions};
use crate::ops::eval_all;

type A = Array<f32>;

const N: usize = 96;

fn ctx(follow: bool) -> Context {
    let ctx = Context::with_options(ContextOptions::from_env().calibrate(false)).unwrap();
    ctx.set_cost_model(CostModel::TEST);
    ctx.jit().set_follow(follow);
    ctx
}

/// Whether the backend evaluates when `eval` asks (a GPU postpones while it is busy, so how many steps
/// replay depends on timing there; what they compute never does).
fn on_cpu() -> bool {
    matches!(ctx(false).backend(), luma_core::BackendKind::Cpu)
}

fn data(seed: u32) -> Vec<f32> {
    (0..N).map(|i| 0.5 + ((i as u32).wrapping_mul(2654435761).wrapping_add(seed.wrapping_mul(40503)) % 1000) as f32 * 1e-3).collect()
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|v| v.to_bits()).collect()
}

/// `1.0009 + i * 1.7e-3 + salt * 1.1e-4`: scalars that differ from step to step and that no rewrite looks for.
fn scalar(i: usize, salt: usize) -> f32 {
    1.0009 + i as f32 * 1.7e-3 + salt as f32 * 1.1e-4
}

fn chain(x: &A, k: usize, salt: usize) -> A {
    let mut y = x.clone();
    for i in 0..k {
        let c = scalar(i, salt);
        y = match i % 4 {
            0 => &y * c,
            1 => &y + c * 1e-3,
            2 => y.abs(),
            _ => &y - c * 1e-3,
        };
    }
    y
}

/// Runs `scenario` with following off and on (the same program on two contexts), checks that the
/// bits agree, and returns what following did.
fn both(scenario: impl Fn(&Context) -> Vec<u32>) -> StatsSnapshot {
    let off = ctx(false);
    let want = scenario(&off);
    assert_eq!(off.jit().follow_stats(), StatsSnapshot::default(), "a context with following off recorded or followed");
    let on = ctx(true);
    let got = scenario(&on);
    assert_eq!(got, want, "following changed what the program computes");
    assert_eq!(on.jit().pending_nodes.load(std::sync::atomic::Ordering::Relaxed), 0, "pending nodes were left behind");
    on.jit().follow_stats()
}

#[test]
fn equal_steps_replay_their_tape() {
    let stats = both(|ctx| {
        let x = A::from_slice_on(ctx, &data(1), N).unwrap();
        let mut out = Vec::new();
        for salt in 0..14 {
            let y = chain(&x, 20, salt);
            y.eval().unwrap();
            out.extend(bits(&y.to_vec().unwrap()));
        }
        out
    });
    if on_cpu() {
        assert_eq!(stats.recorded, 1, "{stats:?}");
    }
    if on_cpu() {
        assert!(stats.replayed >= 11, "{stats:?}");
    }
    if on_cpu() {
        assert_eq!(stats.diverged, 0, "{stats:?}");
    }
}

#[test]
fn a_step_that_ends_with_a_read_replays_too() {
    let stats = both(|ctx| {
        let x = A::from_slice_on(ctx, &data(1), N).unwrap();
        let mut out = Vec::new();
        for salt in 0..10 {
            out.extend(bits(&chain(&x, 12, salt).to_vec().unwrap()));
            let y = chain(&x, 12, salt + 50);
            y.eval().unwrap();
            out.extend(bits(&y.to_vec().unwrap()));
        }
        out
    });
    if on_cpu() {
        assert!(stats.replayed >= 8, "{stats:?}");
    }
}

#[test]
fn wide_steps_with_shared_leaves_and_state_replay() {
    let stats = both(|ctx| {
        let x = A::from_slice_on(ctx, &data(1), N).unwrap();
        let z = A::from_slice_on(ctx, &data(2), N).unwrap();
        let mut cur = A::from_slice_on(ctx, &data(3), N).unwrap();
        for salt in 0..12 {
            let mut acc: Option<A> = None;
            for i in 0..6 {
                let c = scalar(i, salt);
                let base = if i % 2 == 0 { &x } else { &z };
                let t = &(&(base * c) + c * 1e-3).abs() * c;
                acc = Some(match acc {
                    None => t,
                    Some(a) => a + t,
                });
            }
            let next = &(acc.unwrap() * 1e-3) + &(&cur * 0.5);
            next.eval().unwrap();
            cur = next;
        }
        bits(&cur.to_vec().unwrap())
    });
    if on_cpu() {
        assert!(stats.replayed >= 8, "{stats:?}");
    }
}

#[test]
fn an_operation_that_changes_mid_step_diverges() {
    let stats = both(|ctx| {
        let x = A::from_slice_on(ctx, &data(1), N).unwrap();
        let mut out = Vec::new();
        for s in 0..16 {
            let c = scalar(1, s);
            let y = if (s / 5) % 2 == 0 { &(&x * c) + 1.5 } else { &(&x + c) * 2.5 };
            y.eval().unwrap();
            out.extend(bits(&y.to_vec().unwrap()));
        }
        out
    });
    if on_cpu() {
        assert!(stats.replayed >= 6 && stats.recorded >= 2, "{stats:?}");
    }
}

#[test]
fn a_step_that_is_cut_short_or_goes_on_diverges() {
    both(|ctx| {
        let x = A::from_slice_on(ctx, &data(1), N).unwrap();
        let mut out = Vec::new();
        for s in 0..24 {
            // The number of operations changes from time to time: the call after the last one, or one missing.
            let k = if s % 7 == 6 { 9 } else if s % 7 == 3 { 13 } else { 11 };
            let y = chain(&x, k, s);
            y.eval().unwrap();
            out.extend(bits(&y.to_vec().unwrap()));
        }
        out
    });
}

#[test]
fn a_write_between_steps_is_a_divergence_and_the_pending_step_keeps_the_old_values() {
    both(|ctx| {
        let mut xw = A::from_slice_on(ctx, &data(1), N).unwrap();
        let mut out = Vec::new();
        for s in 0..14 {
            let y = &(&xw * scalar(2, s)) + 1.0;
            if s % 4 == 3 {
                // A write while the step is pending: the pending expression keeps the old values.
                xw.write(&data(10 + s as u32)).unwrap();
            }
            y.eval().unwrap();
            out.extend(bits(&y.to_vec().unwrap()));
        }
        out
    });
}

#[test]
fn an_in_place_operation_in_the_step() {
    both(|ctx| {
        let mut xw = A::from_slice_on(ctx, &data(1), N).unwrap();
        let mut out = Vec::new();
        for s in 0..12 {
            let y = &xw * scalar(3, s);
            xw += 0.5;
            let w = &y + &xw;
            w.eval().unwrap();
            out.extend(bits(&w.to_vec().unwrap()));
        }
        out
    });
}

#[test]
fn another_pending_array_keeps_the_tape_from_replaying() {
    let stats = both(|ctx| {
        let x = A::from_slice_on(ctx, &data(1), N).unwrap();
        let z = A::from_slice_on(ctx, &data(2), N).unwrap();
        // Held and pending through every step: a plan made without it would not be the plan with it.
        let held = &z * 1.0009;
        let mut out = Vec::new();
        for s in 0..12 {
            let y = chain(&x, 10, s);
            y.eval().unwrap();
            out.extend(bits(&y.to_vec().unwrap()));
        }
        out.extend(bits(&held.to_vec().unwrap()));
        out
    });
    if on_cpu() {
        assert_eq!(stats.replayed, 0, "{stats:?}");
    }
}

#[test]
fn an_array_that_gets_pending_and_is_read_later() {
    both(|ctx| {
        let x = A::from_slice_on(ctx, &data(1), N).unwrap();
        let z = A::from_slice_on(ctx, &data(2), N).unwrap();
        let mut out = Vec::new();
        let mut held: Option<A> = None;
        for s in 0..16 {
            let y = chain(&x, 10, s);
            if s % 5 == 0 {
                held = Some(&z * scalar(4, s));
            }
            y.eval().unwrap();
            out.extend(bits(&y.to_vec().unwrap()));
            if s % 5 == 4
                && let Some(p) = &held
            {
                out.extend(bits(&p.to_vec().unwrap()));
            }
        }
        out
    });
}

#[test]
fn intermediates_that_are_read_or_kept_are_real_nodes() {
    both(|ctx| {
        let x = A::from_slice_on(ctx, &data(1), N).unwrap();
        let mut out = Vec::new();
        let mut kept: Vec<A> = Vec::new();
        for s in 0..16 {
            let t = &x * scalar(5, s);
            if s % 3 == 2 {
                // A read in the middle of the step ends it.
                out.extend(bits(&t.to_vec().unwrap()));
            }
            let u = &t + 1.0;
            let y = &u * 2.0;
            y.eval().unwrap();
            out.extend(bits(&y.to_vec().unwrap()));
            if s % 2 == 0 {
                // Held across the evaluation, used and written to afterwards.
                kept.push(u);
                let again = &kept[kept.len() - 1] * 3.0;
                out.extend(bits(&again.to_vec().unwrap()));
                out.extend(bits(&t.to_vec().unwrap()));
            }
        }
        for u in &kept {
            out.extend(bits(&u.to_vec().unwrap()));
        }
        out
    });
}

#[test]
fn a_node_a_replay_left_pending_has_what_create_would_have_made() {
    let on = ctx(true);
    let off = ctx(false);
    let probe = |ctx: &Context| {
        let x = A::from_slice_on(ctx, &data(1), N).unwrap();
        let mut seen = Vec::new();
        for s in 0..8 {
            let a = &x * scalar(6, s);
            let b = &a.abs() + &x;
            let c = &b * 0.5;
            // Held, and only fused into the kernel of `c`: still pending after the step.
            c.eval().unwrap();
            let d = &b + 2.0;
            if let Some(node) = d.pending() {
                seen.push((node.below_len(), node.inputs().len(), node.below_vec().len(), node.depth(), node.is_pending()));
            }
            let e = &d * &a;
            e.eval().unwrap();
            seen.push((e.to_vec().unwrap().len(), 0, 0, 0, false));
        }
        seen
    };
    assert_eq!(probe(&on), probe(&off));
    if on_cpu() {
        assert!(on.jit().follow_stats().replayed > 0, "{:?}", on.jit().follow_stats());
    }
}

#[test]
fn a_reduction_in_the_step_ends_it() {
    both(|ctx| {
        let x = A::from_slice_on(ctx, &data(1), N).unwrap();
        let mut out = Vec::new();
        for s in 0..12 {
            let t = &x * scalar(7, s);
            let m = t.sum();
            let y = &t / &m;
            y.eval().unwrap();
            out.extend(bits(&y.to_vec().unwrap()));
        }
        out
    });
}

#[test]
fn hash_consing_hits_and_misses_are_the_same_with_a_tape() {
    both(|ctx| {
        let x = A::from_slice_on(ctx, &data(1), N).unwrap();
        let mut out = Vec::new();
        for s in 0..16 {
            let c = scalar(8, s);
            let a = &x * c;
            let y = if s % 4 == 3 {
                // The first operand is held, so the second is not the same node.
                let keep = &x * c;
                &(&a + &(&x * c)) + &keep
            } else {
                &a + &(&x * c)
            };
            y.eval().unwrap();
            out.extend(bits(&y.to_vec().unwrap()));
        }
        out
    });
}

#[test]
fn scalars_that_rewrites_look_for_and_scalars_that_equal_one_another() {
    both(|ctx| {
        let x = A::from_slice_on(ctx, &data(1), N).unwrap();
        let mut out = Vec::new();
        let vals = [1.5f32, 1.0, 2.0, 2.0, 0.0, 1.0, 1.5, 3.0, 1.0, 1.0, 2.5, 0.0, N as f32, 1.25, -1.0, 0.0];
        for s in 0..vals.len() * 2 {
            let a = vals[s % vals.len()];
            let b = vals[(s + 3) % vals.len()];
            let y = &(&(&x * a) + b) * a;
            y.eval().unwrap();
            out.extend(bits(&y.to_vec().unwrap()));
        }
        out
    });
}

#[test]
fn leaves_that_are_the_same_buffer_or_not() {
    both(|ctx| {
        let x = A::from_slice_on(ctx, &data(1), N).unwrap();
        let z = A::from_slice_on(ctx, &data(2), N).unwrap();
        let mut out = Vec::new();
        for s in 0..18 {
            let y = if s % 3 == 0 { &x * &x } else { &x * &z };
            let w = &y + 1.0;
            w.eval().unwrap();
            out.extend(bits(&w.to_vec().unwrap()));
        }
        out
    });
}

#[test]
fn deep_chains_are_followed_through_their_cuts() {
    let stats = both(|ctx| {
        // Deeper than a kernel nests (256 in a build without optimisation): one cut, whose launch is
        // the first of the step and a plan can repeat (two cuts in a row take over the buffer the
        // first one wrote, which no replay does).
        let x = A::from_slice_on(ctx, &data(1), N).unwrap();
        let mut out = Vec::new();
        for s in 0..8 {
            let y = chain(&x, if cfg!(debug_assertions) { 400 } else { 700 }, s);
            y.eval().unwrap();
            out.extend(bits(&y.to_vec().unwrap()));
        }
        out
    });
    if on_cpu() {
        assert!(stats.replayed >= 5 && stats.cuts >= 5, "{stats:?}");
    }
}

#[test]
fn state_that_advances_through_a_step() {
    both(|ctx| {
        let mut cur = A::from_slice_on(ctx, &data(1), N).unwrap();
        let z = A::from_slice_on(ctx, &data(2), N).unwrap();
        for _ in 0..14 {
            let next = &(&cur * 0.99) + &(&z * 0.01);
            next.eval().unwrap();
            cur = next;
        }
        let mut stencil = A::from_slice_on(ctx, &data(4), N).unwrap();
        for _ in 0..14 {
            let next = &(&stencil.shift([1]) + &stencil.shift([-1])) * 0.5;
            next.eval().unwrap();
            stencil = next;
        }
        let mut out = bits(&cur.to_vec().unwrap());
        out.extend(bits(&stencil.to_vec().unwrap()));
        out
    });
}

#[test]
fn several_roots_in_one_evaluation() {
    let stats = both(|ctx| {
        let x = A::from_slice_on(ctx, &data(1), N).unwrap();
        let z = A::from_slice_on(ctx, &data(2), N).unwrap();
        let mut out = Vec::new();
        for s in 0..12 {
            let a = &x * scalar(1, s);
            let b = &z + scalar(2, s);
            let c = &a * &b;
            eval_all(&[&a, &b, &c]).unwrap();
            out.extend(bits(&c.to_vec().unwrap()));
        }
        out
    });
    if on_cpu() {
        assert!(stats.replayed >= 6, "{stats:?}");
    }
}

#[test]
fn a_flush_ends_the_step() {
    both(|ctx| {
        let x = A::from_slice_on(ctx, &data(1), N).unwrap();
        let mut out = Vec::new();
        for s in 0..12 {
            let y = chain(&x, 12, s);
            ctx.flush().unwrap();
            out.extend(bits(&y.to_vec().unwrap()));
        }
        out
    });
}

#[test]
fn steps_that_are_dropped_without_an_evaluation() {
    both(|ctx| {
        let x = A::from_slice_on(ctx, &data(1), N).unwrap();
        let mut out = Vec::new();
        for s in 0..20 {
            let y = chain(&x, 10, s);
            if s % 3 == 0 {
                drop(y);
                continue;
            }
            y.eval().unwrap();
            out.extend(bits(&y.to_vec().unwrap()));
        }
        out
    });
}

#[test]
fn a_context_that_two_threads_use_stops_following() {
    let on = ctx(true);
    let off = ctx(false);
    let run = |ctx: &Context| {
        let x = A::from_slice_on(ctx, &data(1), N).unwrap();
        let mut out = Vec::new();
        for s in 0..6 {
            let y = chain(&x, 10, s);
            y.eval().unwrap();
            out.extend(bits(&y.to_vec().unwrap()));
        }
        // Another thread writes: the arrays that the steps read could be overwritten under phantoms.
        let (ctx2, host) = (ctx.clone(), data(5));
        let theirs = std::thread::spawn(move || {
            let mut shared = A::from_slice_on(&ctx2, &host, N).unwrap();
            shared.write(&data(6)).unwrap();
            bits(&shared.to_vec().unwrap())
        })
        .join()
        .unwrap();
        out.extend(theirs);
        for s in 6..14 {
            let y = chain(&x, 10, s);
            y.eval().unwrap();
            out.extend(bits(&y.to_vec().unwrap()));
        }
        out
    };
    assert_eq!(run(&on), run(&off));
    let stats = on.jit().follow_stats();
    if on_cpu() {
        assert!(stats.replayed >= 2, "{stats:?}");
    }
}

#[test]
fn steps_built_while_a_recording_is_active_are_not_followed() {
    let ctx = ctx(true);
    let x = A::from_slice_on(&ctx, &data(1), N).unwrap();
    assert!(ctx.begin_recording(Vec::new()));
    for s in 0..8 {
        let y = chain(&x, 10, s);
        y.eval().unwrap();
    }
    let _ = ctx.end_recording();
    let stats = ctx.jit().follow_stats();
    if on_cpu() {
        assert_eq!(stats.replayed, 0, "{stats:?}");
    }
}

#[test]
fn the_tape_survives_steps_of_two_contexts() {
    let a = ctx(true);
    let b = ctx(true);
    let xa = A::from_slice_on(&a, &data(1), N).unwrap();
    let xb = A::from_slice_on(&b, &data(1), N).unwrap();
    let mut last = (Vec::new(), Vec::new());
    for s in 0..12 {
        let ya = chain(&xa, 10, s);
        let yb = chain(&xb, 10, s);
        ya.eval().unwrap();
        yb.eval().unwrap();
        last = (bits(&ya.to_vec().unwrap()), bits(&yb.to_vec().unwrap()));
    }
    assert_eq!(last.0, last.1);
}

#[test]
fn no_buffer_outlives_what_it_is_used_for() {
    let ctx = ctx(true);
    let x = A::from_slice_on(&ctx, &data(1), N).unwrap();
    for s in 0..6 {
        chain(&x, 20, s).eval().unwrap();
    }
    let before = ctx.memory_stats();
    for s in 6..40 {
        chain(&x, 20, s).eval().unwrap();
    }
    ctx.flush().unwrap();
    let after = ctx.memory_stats();
    assert!(after.resident_bytes <= before.resident_bytes + (1 << 16), "{before:?} -> {after:?}");
}

#[test]
fn another_thread_that_writes_a_buffer_a_phantom_reads() {
    // After the tape is made, a step reads `shared`; another thread overwrites it while that step is
    // pending (the write flushes what reads it, as it always did); the step sees the old values.
    let run = |ctx: &Context| {
        let x = A::from_slice_on(ctx, &data(1), N).unwrap();
        let mut out = Vec::new();
        for s in 0..6 {
            let y = chain(&x, 10, s);
            y.eval().unwrap();
            out.extend(bits(&y.to_vec().unwrap()));
        }
        let mut shared = A::from_slice_on(ctx, &data(7), N).unwrap();
        for s in 6..12 {
            let y = chain(&shared, 10, s);
            let (ctx2, mut theirs) = (ctx.clone(), shared.clone());
            if s == 10 {
                std::thread::spawn(move || {
                    let _ = &ctx2;
                    theirs.write(&data(8)).unwrap();
                })
                .join()
                .unwrap();
            }
            y.eval().unwrap();
            out.extend(bits(&y.to_vec().unwrap()));
        }
        let _ = &mut shared;
        out
    };
    let (on, off) = (ctx(true), ctx(false));
    assert_eq!(run(&on), run(&off));
}

#[test]
fn an_intermediate_sent_to_another_thread_and_written_mid_step() {
    // The step holds an intermediate `a` and builds `b` on it; another thread takes `a`, evaluates and
    // overwrites it before the step is evaluated: `b` was recorded against the old values.
    let run = |ctx: &Context| {
        let x = A::from_slice_on(ctx, &data(1), N).unwrap();
        let mut out = Vec::new();
        for s in 0..12 {
            let a = &x * scalar(1, s);
            let b = &a + 1.5;
            if s == 9 {
                let mut theirs = a.clone();
                std::thread::spawn(move || {
                    theirs.write(&data(9)).unwrap();
                })
                .join()
                .unwrap();
            }
            b.eval().unwrap();
            out.extend(bits(&b.to_vec().unwrap()));
            out.extend(bits(&a.to_vec().unwrap()));
        }
        out
    };
    let (on, off) = (ctx(true), ctx(false));
    assert_eq!(run(&on), run(&off));
}
