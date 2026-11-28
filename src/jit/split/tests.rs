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

// -- re-selection (D9) --

use crate::array::Failure;
use crate::caps::Caps;
use crate::jit::structured::emit::Emit;
use crate::jit::structured::strategy::{BitClass, FamilyTag, FitsInputs, Lowered, NumClass, Output, Params, ReduceParams, PassMode, ResClass, Residencies, SplitClass, Strategy, StrategyId, Subject, Warm};

fn fits_device(_: &Subject<'_>, _: &Context, res: &Residencies) -> Option<Params> {
    (res.class == ResClass::Device).then_some(Params::Reduce(ReduceParams { pass: PassMode::Auto, groups: 1 }))
}

fn fits_streamed(_: &Subject<'_>, _: &Context, res: &Residencies) -> Option<Params> {
    (res.class == ResClass::Streamed).then_some(Params::Reduce(ReduceParams { pass: PassMode::Auto, groups: 1 }))
}

fn split_global(_: &Subject<'_>, _: &Params) -> SplitClass {
    SplitClass::Global
}

fn no_warm(_: &Subject<'_>, _: &Params, _: &Context) -> Result<Warm, Failure> {
    Ok(Warm { kernels: Vec::new() })
}

/// The one value the test family's second strategy makes.
fn lower_forty_two(_: &Subject<'_>, _: &Params, emit: &mut Emit<'_>) -> Result<Lowered, Failure> {
    let made = Array::<f32>::from_slice_on(emit.ctx(), &[42.0], 1)?;
    Ok(Lowered::Done(Output::Storage(Arc::clone(made.storage()?))))
}

const fn test_entry(at: u8, name: &'static str, fits: crate::jit::structured::strategy::FitsFn) -> Strategy {
    Strategy {
        id: StrategyId(FamilyTag::Reduce, at),
        name,
        needs: Caps::empty(),
        numerics: NumClass::Exact,
        bits: BitClass(at as u16),
        baseline: true,
        group: at,
        prologue: false,
        epilogue: false,
        reads: FitsInputs::NONE,
        fits,
        cost: None,
        split: split_global,
        warm: no_warm,
        lower: Some(lower_forty_two),
    }
}

/// A family of two strategies: the first admits resident operands only, the second admits streamed ones.
static TEST_FAMILY: [Strategy; 2] = [test_entry(100, "test-resident", fits_device), test_entry(101, "test-streamed", fits_streamed)];

#[test]
fn reselection_happens_at_most_once() {
    let _serial = LOCK.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    let ctx = blocks_context(1024);
    let n = 5 * 256;
    let x = Array::from_slice_on(&ctx, &vec![1.0f32; n], n).unwrap();
    let id = ctx.id();
    super::plan::INFEASIBLE.store(id, Ordering::Relaxed);
    *super::plan::TEST_ENTRIES.lock().unwrap() = Some((id, &TEST_FAMILY));
    // The stage of the reduction never fits: the hook replaces its strategy by the one that admits streamed operands, once, and the
    // stages formed again still do not fit but the step is not asked again.
    let sum = x.sum();
    let evaluated = sum.eval();
    let values = sum.to_vec();
    super::plan::INFEASIBLE.store(0, Ordering::Relaxed);
    *super::plan::TEST_ENTRIES.lock().unwrap() = None;
    evaluated.unwrap();
    assert_eq!(values.unwrap(), vec![42.0f32]);
    assert_eq!(ctx.stream_stats().reselections, 1, "{:?}", ctx.stream_stats());
    // A context that is not asked to fail is not touched.
    let other = blocks_context(1024);
    let y = Array::from_slice_on(&other, &vec![1.0f32; n], n).unwrap();
    assert_eq!(y.sum().to_vec().unwrap(), vec![n as f32]);
    assert_eq!(other.stream_stats().reselections, 0);
}

/// Slab order the stages visit (0: the planner's), set by one test at a time.
pub(super) static FORCE_ORDER: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

#[test]
fn float_reductions_do_not_depend_on_slab_order() {
    let _serial = LOCK.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    // Segments of 2^16 elements: 6 slabs, blocks of 2^16, and orders a ring would never choose.
    let n = 6 * (1 << 16) - 777;
    let data = float_data(n);
    let mut seen: Vec<Vec<u32>> = Vec::new();
    for order in [0u8, 1, 2, 3] {
        FORCE_ORDER.store(order, Ordering::Relaxed);
        let ctx = blocks_context(1 << 18);
        let x = Array::from_slice_on(&ctx, &data, n).unwrap();
        let y = &x * 2.0;
        let bits = |values: Vec<f32>| values.into_iter().map(f32::to_bits).collect::<Vec<_>>();
        let mut all = bits(x.reduce([crate::Reduction::Sum, crate::Reduction::Max, crate::Reduction::SumOfSquares]).to_vec().unwrap());
        all.extend(bits(y.reduce([crate::Reduction::Sum]).to_vec().unwrap()));
        seen.push(all);
        assert!(ctx.stream_stats().stages >= 2, "{:?}", ctx.stream_stats());
    }
    FORCE_ORDER.store(0, Ordering::Relaxed);
    assert!(seen.iter().all(|bits| *bits == seen[0]), "{seen:?}");
}

#[test]
fn integer_reductions_are_unchanged_by_deferral() {
    let n = 5 * 256 + 17;
    let data: Vec<u32> = (0..n as u32).map(|i| i.wrapping_mul(2_654_435_761)).collect();
    let mut results = Vec::new();
    for mode in [SplitMode::Off, SplitMode::Auto] {
        let ctx = Context::with_options(ContextOptions::from_env().chunk_bytes(1024).split(mode)).expect("a device is available");
        let x = Array::from_slice_on(&ctx, &data, n).unwrap();
        results.push((x.sum().item().unwrap(), x.max().item().unwrap(), x.min().item().unwrap()));
    }
    assert_eq!(results[0], results[1]);
}

#[test]
fn a_single_segment_reduction_keeps_its_bits() {
    let n = 3000;
    let data = float_data(n);
    let mut results = Vec::new();
    for mode in [SplitMode::Off, SplitMode::Auto] {
        let ctx = Context::with_options(ContextOptions::from_env().split(mode)).expect("a device is available");
        let x = Array::from_slice_on(&ctx, &data, n).unwrap();
        let reduced = x.reduce([crate::Reduction::Sum, crate::Reduction::SumOfSquares]).to_vec().unwrap();
        results.push((x.sum().item().unwrap().to_bits(), x.norm().item().unwrap().to_bits(), reduced.iter().map(|v| v.to_bits()).collect::<Vec<_>>()));
        assert_eq!(ctx.stream_stats().stages, 0);
    }
    assert_eq!(results[0], results[1]);
}
