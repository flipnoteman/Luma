//! Slab-major execution (`LUMA_SPLIT`): a chain of eight kernels the planner cannot fuse (each stage is cached) over an array of
//! 16 segments, on a context of its own with a memory budget set by the row. `chain8_fits` is half the budget, `chain8_2x` twice
//! the budget; each runs with the mode `off` (kernel-major: every kernel pages the whole array) and `auto` (one pass over the
//! slabs, the stages' intermediates never materialised). `sum_of_product_2x` is `(x * y).sum()` over two arrays whose sum is twice
//! the budget (the product never forms), and `chain8_cpu_blocking` is the chain over in-memory segments on the CPU (slab-major keeps a
//! slab's chain in cache). Each row prints its `StreamStats` beside the time.

use std::time::Instant;

use super::common::*;
use crate::harness::*;
use luma::*;

const STAGES: usize = 8;

/// The chain: each stage is a kernel of its own (`cache`), with arithmetic to hide transfers behind.
fn chain8(x: &A) -> A {
    let mut y = x.clone();
    for k in 0..STAGES {
        y = ((&y * (1.0 + 0.0009 * (k + 1) as f32) + 0.25 * (k % 3) as f32).sin().abs() + x * 0.5).tanh().cache();
    }
    y
}

fn fill(n: usize) -> Vec<f32> {
    (0..n).map(|i| (i % 1013) as f32 * 0.01).collect()
}

fn mib(bytes: u64) -> u64 {
    bytes >> 20
}

fn context(budget: u64, chunk: u64, mode: SplitMode) -> Option<Context> {
    Context::with_options(ContextOptions::from_env().memory_budget(budget).chunk_bytes(chunk).split(mode)).ok()
}

/// What a row moved per call (`StreamStats` deltas, in MiB).
fn print_stats(row: &str, label: &str, calls: u64, start: Instant, before: &(StreamStats, u64, u64), ctx: &Context) {
    let stats = ctx.stream_stats();
    let per = calls.max(1);
    println!(
        "  {row}_{label}: {:.1} s over {calls} calls; per call upload {} MiB download {} MiB (paged up {} MiB, spilled {} MiB), schedules {} stages {} slabs {} elided {} MiB, immediate reductions {}",
        start.elapsed().as_secs_f64(),
        mib((stats.upload_bytes - before.0.upload_bytes) / per),
        mib((stats.download_bytes - before.0.download_bytes) / per),
        mib((ctx.paging_stats().uploaded_bytes - before.1) / per),
        mib(ctx.spilled_bytes().saturating_sub(before.2) / per),
        (stats.schedules - before.0.schedules) / per,
        (stats.stages - before.0.stages) / per,
        (stats.slabs - before.0.slabs) / per,
        mib((stats.transient_bytes_elided - before.0.transient_bytes_elided) / per),
        (stats.immediate_reductions - before.0.immediate_reductions) / per,
    );
}

fn snapshot(ctx: &Context) -> (StreamStats, u64, u64) {
    (ctx.stream_stats(), ctx.paging_stats().uploaded_bytes, ctx.spilled_bytes())
}

pub fn run(h: &mut Harness) {
    if h.ctx.is_cpu() {
        cpu_blocking(h);
        return;
    }
    // (budget MiB, array MiB) of `chain8_fits` and `chain8_2x`.
    let budget = size(h, 128, 128, 512) as u64;
    h.guarded("split", |h| {
        // Verification at a small size: `auto` is bit-equal to `off`.
        {
            let n = (16 << 20) / 4;
            let xs = fill(n);
            let mut results = Vec::new();
            for mode in [SplitMode::Off, SplitMode::Auto] {
                let Some(ctx) = context(8 << 20, 1 << 20, mode) else { return };
                let x = A::from_slice_on(&ctx, &xs, n).unwrap();
                results.push(chain8(&x).to_vec().unwrap());
            }
            assert!(results[0].iter().zip(&results[1]).all(|(a, b)| a.to_bits() == b.to_bits()), "split/chain8: auto differs from off");
        }
        for (row, array_mib) in [("chain8_fits", budget / 2), ("chain8_2x", budget * 2)] {
            let n = (array_mib << 20) as usize / 4;
            let xs = fill(n);
            for (label, mode) in [("off", SplitMode::Off), ("auto", SplitMode::Auto)] {
                let Some(ctx) = context(budget << 20, (array_mib << 20) / 16, mode) else { continue };
                let x = A::from_slice_on(&ctx, &xs, n).unwrap();
                ctx.sync().unwrap();
                let before = snapshot(&ctx);
                let start = Instant::now();
                let mut calls = 0u64;
                h.transfer_case(&format!("split/{row}_{label}"), &format!("{array_mib}M/{budget}M"), Work::Bytes((2 * STAGES * n * 4) as f64), &mut || {
                    let y = chain8(&x);
                    y.eval().unwrap();
                    ctx.sync().unwrap();
                    calls += 1;
                });
                print_stats(row, label, calls, start, &before, &ctx);
            }
        }
        // `(x * y).sum()`: two arrays of the budget each, 16 segments each; the product is never formed in `auto`.
        {
            let array_mib = budget;
            let n = (array_mib << 20) as usize / 4;
            let (xs, ys) = (fill(n), fill(n).into_iter().rev().collect::<Vec<_>>());
            let mut sums = Vec::new();
            for (label, mode) in [("off", SplitMode::Off), ("auto", SplitMode::Auto)] {
                let Some(ctx) = context(budget << 20, (array_mib << 20) / 16, mode) else { continue };
                let (x, y) = (A::from_slice_on(&ctx, &xs, n).unwrap(), A::from_slice_on(&ctx, &ys, n).unwrap());
                ctx.sync().unwrap();
                let before = snapshot(&ctx);
                let start = Instant::now();
                let mut calls = 0u64;
                let mut last = 0.0f32;
                h.transfer_case(&format!("split/sum_of_product_2x_{label}"), &format!("2x{array_mib}M/{budget}M"), Work::Bytes((2 * n * 4) as f64), &mut || {
                    last = (&x * &y).sum().item().unwrap();
                    calls += 1;
                });
                print_stats("sum_of_product_2x", label, calls, start, &before, &ctx);
                sums.push(last);
            }
            if let [off, auto] = sums[..] {
                // Canonical blocks change the last bits of a multi-segment float sum.
                assert!(((off - auto) / off.abs().max(1.0)).abs() < 1e-3, "split/sum_of_product_2x: {off} against {auto}");
            }
        }
    });
}

/// `chain8` over in-memory segments on the CPU (no budget): kernel-major runs each stage over the whole array, slab-major runs the eight
/// stages over one segment before the next, which then stays in cache.
fn cpu_blocking(h: &mut Harness) {
    let array_mib = size(h, 32, 64, 64) as u64;
    h.guarded("split", |h| {
        let n = (array_mib << 20) as usize / 4;
        let xs = fill(n);
        let mut results = Vec::new();
        for (label, mode) in [("off", SplitMode::Off), ("auto", SplitMode::Auto)] {
            let Ok(ctx) = Context::with_options(ContextOptions::from_env().chunk_bytes((array_mib << 20) / 16).split(mode)) else { return };
            let x = A::from_slice_on(&ctx, &xs, n).unwrap();
            ctx.sync().unwrap();
            let before = snapshot(&ctx);
            let start = Instant::now();
            let mut calls = 0u64;
            h.transfer_case(&format!("split/chain8_cpu_blocking_{label}"), &format!("{array_mib}M"), Work::Bytes((2 * STAGES * n * 4) as f64), &mut || {
                let y = chain8(&x);
                y.eval().unwrap();
                ctx.sync().unwrap();
                calls += 1;
            });
            print_stats("chain8_cpu_blocking", label, calls, start, &before, &ctx);
            results.push(chain8(&x).to_vec().unwrap());
        }
        if let [off, auto] = &results[..] {
            assert!(off.iter().zip(auto).all(|(a, b)| a.to_bits() == b.to_bits()), "split/chain8_cpu_blocking: auto differs from off");
        }
    });
}
