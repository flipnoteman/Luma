//! Failures of a commit: the outputs are poisoned and the error reaches everything that uses them.

use std::sync::atomic::{AtomicUsize, Ordering};

use super::SplitMode;
use crate::{Array, Context, ContextOptions};

/// Slab at which the next stage fails (`usize::MAX`: never). One test at a time sets it (they are serialised by `LOCK`).
static FAIL_AT: AtomicUsize = AtomicUsize::new(usize::MAX);
static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

pub(super) fn fail_here(slab: usize) -> bool {
    slab >= FAIL_AT.load(Ordering::Relaxed)
}

#[test]
fn a_failing_commit_poisons_the_outputs() {
    let _serial = LOCK.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    let ctx = Context::with_options(ContextOptions::from_env().chunk_bytes(1024).split(SplitMode::Auto)).expect("a device is available");
    let n = 5 * 256 + 3;
    let x = Array::from_slice_on(&ctx, &vec![1.0f32; n], n).unwrap();
    let y = &x * 2.0;
    FAIL_AT.store(3, Ordering::Relaxed);
    // The commit runs when the GPU is idle (`eval`) or when the data is wanted; the error comes back from the call that triggered it.
    let evaluated = y.eval();
    FAIL_AT.store(usize::MAX, Ordering::Relaxed);
    let wanted = y.to_vec();
    let error = evaluated.err().or(wanted.err()).expect("the commit failed");
    assert!(format!("{error}").contains("injected failure"), "{error}");
    // Everything the failed steps wrote stays failed, whoever asks.
    assert!(y.to_vec().is_err());
    // A dependent step fails its own commit the same way.
    let z = &y + 1.0;
    assert!(z.to_vec().is_err());
    // The input is intact and the context works.
    assert_eq!(x.to_vec().unwrap(), vec![1.0f32; n]);
    let fresh = Array::from_slice_on(&ctx, &vec![3.0f32; n], n).unwrap();
    assert_eq!((&fresh + 1.0).to_vec().unwrap(), vec![4.0f32; n]);
}

#[test]
fn dropping_the_last_handle_of_a_context_commits_its_schedule() {
    let ctx = Context::with_options(ContextOptions::from_env().chunk_bytes(1024).split(SplitMode::Auto)).expect("a device is available");
    let n = 5 * 256 + 3;
    let x = Array::from_slice_on(&ctx, &vec![1.0f32; n], n).unwrap();
    // An expression too long for one kernel is cut: the cuts are deferred, and nothing asks for the result.
    let mut y = x.clone();
    for k in 0..300 {
        y = (&y * 1.0009 + (k % 5) as f32 * 0.25).abs() - (&y * 0.001);
    }
    let before = super::COMMITS_ON_DROP.load(Ordering::Relaxed);
    drop(y);
    drop(x);
    drop(ctx);
    assert!(super::COMMITS_ON_DROP.load(Ordering::Relaxed) > before, "the schedule ran when the context was dropped");
}

// -- canonical blocks (D8) --

use std::sync::Arc;

use super::reduce::merge_tree;
use crate::builtin::{Builtin, ReduceList, ReduceOp};
use crate::context::{RawArg, STORAGE_USAGE};

/// The values `ops` folds out of `states` (`count` of them per op) on the device, through the balanced merge kernel.
fn merged_on_device(ctx: &Context, ops: &[ReduceOp], states: &[f32], count: usize) -> Vec<f32> {
    let list = ReduceList::new(ops).balanced();
    let core = ctx.builtin(Builtin::Reduce(list, true), "f32").unwrap();
    let input = ctx.alloc_buffer((states.len() * 4) as u64, STORAGE_USAGE).unwrap();
    ctx.upload(&input, bytemuck::cast_slice(states)).unwrap();
    // Two slots per op of a pretend carry, the result in the second: the stride and slot reach the output index.
    let out = ctx.alloc_buffer((ops.len() * 2 * 4) as u64, STORAGE_USAGE).unwrap();
    let words = [count as u32, 2, 1];
    ctx.launch(
        &core,
        [1, 1, 1],
        [count as u32, 1, 1],
        &[RawArg::write(&out, (ops.len() * 2 * 4) as u64), RawArg::read(&input, (states.len() * 4) as u64), RawArg::Uniform(bytemuck::cast_slice(&words))],
    )
    .unwrap();
    let values: Vec<f32> = ctx.read(&out, (ops.len() * 2 * 4) as u64).unwrap();
    (0..ops.len()).map(|j| values[j * 2 + 1]).collect()
}

#[test]
fn the_merge_tree_matches_the_host_reference() {
    let ctx = Context::new().expect("a device is available");
    let mut seed = 0x2545_f491_4f6c_dd1du64;
    let mut next = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    let counts = [1usize, 2, 3, 5, 7, 8, 9, 255, 256, 257, 300, 511, 512, 513, 1000, 1025, 4097, 9999, 33_333, 70_000];
    for count in counts {
        // Values of very different magnitudes make every association visible in the bits.
        let ops = [ReduceOp::Sum, ReduceOp::Prod, ReduceOp::SumSquares];
        let mut states = Vec::with_capacity(count * ops.len());
        for op in ops {
            for _ in 0..count {
                let unit = (next() >> 40) as f32 / (1u64 << 24) as f32;
                states.push(match op {
                    ReduceOp::Prod => 1.0 + (unit - 0.5) * 1e-3,
                    _ => (unit - 0.5) * 10f32.powi((next() % 9) as i32 - 4),
                });
            }
        }
        let got = merged_on_device(&ctx, &ops, &states, count);
        for (j, op) in ops.iter().enumerate() {
            let leaves = &states[j * count..(j + 1) * count];
            let want = match op {
                ReduceOp::Prod => merge_tree(leaves, &|a, b| a * b),
                _ => merge_tree(leaves, &|a, b| a + b),
            };
            assert_eq!(got[j].to_bits(), want.to_bits(), "{op:?} over {count} states: {} against {want}", got[j]);
        }
    }
}

fn float_data(n: usize) -> Vec<f32> {
    // Cancellation and magnitude spread: a different association gives different last bits.
    (0..n).map(|i| ((i as f32 * 0.618_034).fract() - 0.5) * 10f32.powi((i % 7) as i32 - 3)).collect()
}

fn blocks_context(chunk_bytes: u64) -> Context {
    Context::with_options(ContextOptions::from_env().chunk_bytes(chunk_bytes).split(SplitMode::Auto)).expect("a device is available")
}

#[test]
fn float_reductions_do_not_depend_on_the_chunk_size() {
    // Blocks are `min(2^16, segment)`: segments of at least 2^16 elements (256 KiB) all cut the data into the same blocks.
    let n = 3 * (1 << 20) + 12_345;
    let data = float_data(n);
    let mut seen: Vec<(f32, f32, f32, Vec<f32>)> = Vec::new();
    for chunk in [1u64 << 19, 1 << 20, 1 << 22] {
        let ctx = blocks_context(chunk);
        let x = Array::from_slice_on(&ctx, &data, n).unwrap();
        let reduced = x.reduce([crate::Reduction::Sum, crate::Reduction::Min, crate::Reduction::Max, crate::Reduction::SumOfSquares]).to_vec().unwrap();
        seen.push((x.sum().item().unwrap(), x.norm().item().unwrap(), x.mean().item().unwrap(), reduced));
        assert!(ctx.stream_stats().stages >= 4, "the reductions ran as canonical blocks: {:?}", ctx.stream_stats());
    }
    for other in &seen[1..] {
        assert_eq!(other.0.to_bits(), seen[0].0.to_bits(), "sum");
        assert_eq!(other.1.to_bits(), seen[0].1.to_bits(), "norm");
        assert_eq!(other.2.to_bits(), seen[0].2.to_bits(), "mean");
        assert_eq!(other.3.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), seen[0].3.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), "reduce");
    }
    let exact: f64 = data.iter().map(|&v| f64::from(v)).sum();
    assert!((f64::from(seen[0].0) - exact).abs() <= exact.abs() * 1e-4 + 1.0, "{} against {exact}", seen[0].0);
}
