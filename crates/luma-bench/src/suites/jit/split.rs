//! Slab-major execution (`LUMA_SPLIT`): a chain of eight kernels the planner cannot fuse (each stage is cached) over an array of
//! 16 segments, on a context of its own with a memory budget set by the row. `chain8_fits` is half the budget, `chain8_2x` twice
//! the budget; each runs with the mode `off` (kernel-major: every kernel pages the whole array) and `auto` (one pass over the
//! slabs, the stages' intermediates never materialised). The size column carries the transfers (`StreamStats`).

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

pub fn run(h: &mut Harness) {
    if h.ctx.is_cpu() {
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
                let before = (ctx.stream_stats(), ctx.paging_stats().uploaded_bytes, ctx.spilled_bytes());
                let start = Instant::now();
                let mut calls = 0u64;
                h.transfer_case(&format!("split/{row}_{label}"), &format!("{array_mib}M/{budget}M"), Work::Bytes((2 * STAGES * n * 4) as f64), &mut || {
                    let y = chain8(&x);
                    y.eval().unwrap();
                    ctx.sync().unwrap();
                    calls += 1;
                });
                let stats = ctx.stream_stats();
                let per = calls.max(1);
                println!(
                    "  {row}_{label}: {:.1} s over {calls} calls; per call upload {} MiB download {} MiB (paged up {} MiB, spilled {} MiB), stages {} slabs {} elided {} MiB",
                    start.elapsed().as_secs_f64(),
                    mib((stats.upload_bytes - before.0.upload_bytes) / per),
                    mib((stats.download_bytes - before.0.download_bytes) / per),
                    mib((ctx.paging_stats().uploaded_bytes - before.1) / per),
                    mib((ctx.spilled_bytes().saturating_sub(before.2)) / per),
                    (stats.stages - before.0.stages) / per,
                    (stats.slabs - before.0.slabs) / per,
                    mib((stats.transient_bytes_elided - before.0.transient_bytes_elided) / per),
                );
            }
        }
    });
}
