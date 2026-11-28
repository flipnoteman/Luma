//! Running a committed schedule: steps are compiled into stages, and a stage runs slab by slab.

use std::sync::atomic::Ordering;

use super::{Open, SegRef, StepKernel};
use crate::context::Context;
use crate::error::LumaError;
use crate::jit::launch::{launch_store, launch_store_slab, note_store, store_tiles};

/// Steps that share one pass over the slabs of their (common) iteration space.
struct Stage {
    steps: Vec<usize>,
}

/// Groups the steps into stages. One stage per step: the kernel-major order.
fn compile(open: &Open) -> Vec<Stage> {
    (0..open.steps.len()).map(|at| Stage { steps: vec![at] }).collect()
}

pub(super) fn run(ctx: &Context, open: &Open) -> Result<(), LumaError> {
    let stats = ctx.stream_counters();
    stats.schedules.fetch_add(1, Ordering::Relaxed);
    stats.steps.fetch_add(open.steps.len() as u64, Ordering::Relaxed);
    for stage in compile(open) {
        run_stage(ctx, open, &stage)?;
    }
    Ok(())
}

/// The arrays one step binds, as launches take them.
struct Bound<'a> {
    outs: Vec<SegRef<'a>>,
    ins: Vec<SegRef<'a>>,
}

fn run_stage(ctx: &Context, open: &Open, stage: &Stage) -> Result<(), LumaError> {
    let stats = ctx.stream_counters();
    let bound: Vec<Bound<'_>> = stage
        .steps
        .iter()
        .map(|&at| {
            let step = &open.steps[at];
            let seg = |bind: &super::Bind| open.arrays[bind.array as usize].as_seg(ctx);
            Bound { outs: step.binds[..step.n_out].iter().map(seg).collect(), ins: step.binds[step.n_out..].iter().map(seg).collect() }
        })
        .collect();
    let slabs = open.steps[stage.steps[0]].space.slabs();
    stats.stages.fetch_add(1, Ordering::Relaxed);
    for slab in 0..slabs {
        stats.slabs.fetch_add(1, Ordering::Relaxed);
        for (&at, bound) in stage.steps.iter().zip(&bound) {
            let step = &open.steps[at];
            let StepKernel::Store { core, key } = &step.kernel;
            if store_tiles(key).is_some() {
                // One launch over the one segment.
                launch_store(ctx, core, key, &bound.outs, &bound.ins, &step.scalars, step.shifted.as_ref())?;
            } else {
                launch_store_slab(ctx, core, key, &bound.outs, &bound.ins, &step.scalars, step.shifted.as_ref(), slab)?;
            }
        }
    }
    for &at in &stage.steps {
        let StepKernel::Store { key, .. } = &open.steps[at].kernel;
        if store_tiles(key).is_none() {
            note_store(ctx, key);
        }
    }
    Ok(())
}
