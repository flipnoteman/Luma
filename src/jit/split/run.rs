//! Running a committed schedule: steps are compiled into stages (see `plan`), and a stage runs slab by slab, each slab through every
//! step of the stage, so a segment is brought to the device once per chain.
//!
//! The slab loop of [`run_stage`] is where lanes plug in: a ring of slab buffers filled by upload transfers for streamed inputs
//! and drained by download transfers for streamed outputs.

use std::sync::atomic::Ordering;

use luma_core::hal::size_class;
use luma_core::Buffer;

use super::plan::{compile, SlabOrder, Stage};
use super::{Access, Bind, Open, Reach, SegRef, StepKernel};
use crate::context::{Context, STORAGE_USAGE};
use crate::error::LumaError;
use crate::gpu::hash::{FxMap, FxSet};
use crate::gpu::pager::Segment;
use crate::jit::launch::{launch_store, launch_store_slab, note_store, store_tiles};
use crate::profile::{Decision, DecisionSite};

/// Where the segments of the arrays are used again, for the eviction order of [`crate::gpu::pager::Pager::reserve`].
struct Future {
    owner: FxMap<*const Segment, (u32, u32)>,
    /// For every array, the stages that bind it, ascending.
    stages_of: Vec<Vec<u32>>,
}

impl Future {
    fn new(open: &Open, stages: &[Stage]) -> Future {
        let mut owner = FxMap::default();
        for (array, entry) in open.arrays.iter().enumerate() {
            for (j, segment) in entry.segments.iter().enumerate() {
                owner.insert(std::ptr::from_ref::<Segment>(segment), (array as u32, j as u32));
            }
        }
        let mut stages_of: Vec<Vec<u32>> = vec![Vec::new(); open.arrays.len()];
        for (at, stage) in stages.iter().enumerate() {
            for step in &open.steps[stage.steps.clone()] {
                for bind in step.binds.iter() {
                    let list = &mut stages_of[bind.array as usize];
                    if list.last() != Some(&(at as u32)) {
                        list.push(at as u32);
                    }
                }
            }
        }
        Future { owner, stages_of }
    }

    /// The position of the first use of `segment` by a stage after `stage`.
    fn next_use(&self, segment: &Segment, stage: usize) -> Option<u64> {
        let &(array, j) = self.owner.get(&std::ptr::from_ref(segment))?;
        let list = &self.stages_of[array as usize];
        let at = list.partition_point(|&s| s as usize <= stage);
        list.get(at).map(|&s| (u64::from(s) << 32) | u64::from(j))
    }
}

pub(super) fn run(ctx: &Context, open: &Open) -> Result<(), LumaError> {
    let stats = ctx.stream_counters();
    stats.schedules.fetch_add(1, Ordering::Relaxed);
    stats.steps.fetch_add(open.steps.len() as u64, Ordering::Relaxed);
    let stages = compile(open);
    let future = ctx.pager().budget().is_some().then(|| Future::new(open, &stages));
    let (up, down) = ctx.pager().traffic();
    let mut slabs = 0;
    let mut elided = 0;
    let result = (|| {
        for (at, stage) in stages.iter().enumerate() {
            slabs += stage.space(open).slabs();
            elided += stage.transients.iter().map(|t| open.arrays[t.array as usize].meta.layout.bytes(open.arrays[t.array as usize].meta.len)).sum::<u64>();
            run_stage(ctx, open, stage, at, future.as_ref())?;
        }
        Ok(())
    })();
    let (up_after, down_after) = ctx.pager().traffic();
    stats.upload_bytes.fetch_add(up_after - up, Ordering::Relaxed);
    stats.download_bytes.fetch_add(down_after - down, Ordering::Relaxed);
    if result.is_ok() && ctx.jit().recording_decisions() {
        let retained = open.arrays.iter().filter(|array| array.first_write.is_some()).count() - stages.iter().map(|stage| stage.transients.len()).sum::<usize>();
        let transients = stages.iter().map(|stage| stage.transients.len()).sum::<usize>();
        let chosen = format!("stages={} slabs={slabs} window=1 retain={retained} stream=0 elided={transients}", stages.len());
        let writer = open.arrays.iter().find_map(|array| array.writer);
        if let Some((op, location)) = writer {
            ctx.jit().note_decision(Decision::new(op, location, DecisionSite::Schedule, chosen));
        }
    }
    result
}

/// The arrays one step binds, as launches take them.
struct Bound<'a> {
    outs: Vec<SegRef<'a>>,
    ins: Vec<SegRef<'a>>,
}

/// Buffers of a stage's transients, given back to the pool however the stage ends.
struct StageBuffers<'a> {
    ctx: &'a Context,
    buffers: Vec<Buffer>,
}

impl Drop for StageBuffers<'_> {
    fn drop(&mut self) {
        for buffer in self.buffers.drain(..) {
            self.ctx.recycle(buffer);
        }
    }
}

fn run_stage(ctx: &Context, open: &Open, stage: &Stage, at: usize, future: Option<&Future>) -> Result<(), LumaError> {
    let stats = ctx.stream_counters();
    let steps = &open.steps[stage.steps.clone()];

    // Transients get a slab-sized buffer per colour. Making room for them evicts by next use (the schedule knows the future).
    let mut buffers = StageBuffers { ctx, buffers: Vec::with_capacity(stage.colours.len()) };
    let max = ctx.max_buffer_bytes();
    let total: u64 = stage.colours.iter().map(|&bytes| size_class(bytes, max).unwrap_or(bytes)).sum();
    let _reservation = if total > 0 {
        let keep: FxSet<*const Segment> = steps
            .iter()
            .flat_map(|step| step.binds.iter())
            .filter(|bind| !stage.transients.iter().any(|t| t.array == bind.array))
            .flat_map(|bind| open.arrays[bind.array as usize].segments.iter())
            .map(|segment| std::ptr::from_ref::<Segment>(segment))
            .collect();
        let next_use = |segment: &Segment| future.and_then(|future| future.next_use(segment, at));
        Some(ctx.pager().reserve(ctx, total, &keep, &next_use)?)
    } else {
        None
    };
    for &bytes in &stage.colours {
        buffers.buffers.push(ctx.alloc_buffer(bytes, STORAGE_USAGE)?);
    }
    let fixed = |array: u32| {
        stage.transients.iter().find(|t| t.array == array).map(|t| &buffers.buffers[t.colour])
    };

    let bound: Vec<Bound<'_>> = steps
        .iter()
        .map(|step| {
            let seg = |bind: &Bind| open.arrays[bind.array as usize].as_seg(ctx).fixed_to(fixed(bind.array));
            Bound { outs: step.binds[..step.n_out].iter().map(seg).collect(), ins: step.binds[step.n_out..].iter().map(seg).collect() }
        })
        .collect();

    let space = stage.space(open);
    let slabs = space.slabs();
    stats.stages.fetch_add(1, Ordering::Relaxed);
    for transient in &stage.transients {
        let meta = open.arrays[transient.array as usize].meta;
        stats.transient_bytes_elided.fetch_add(meta.layout.bytes(meta.len), Ordering::Relaxed);
        stats.transients.fetch_add(1, Ordering::Relaxed);
    }
    for slab in slab_order(open, stage, slabs) {
        #[cfg(test)]
        if super::tests::fail_here(slab) {
            return Err(LumaError::Gpu("injected failure".into()));
        }
        stats.slabs.fetch_add(1, Ordering::Relaxed);
        for (step, bound) in steps.iter().zip(&bound) {
            let StepKernel::Store { core, key } = &step.kernel;
            if store_tiles(key).is_some() {
                // One launch over the one segment.
                launch_store(ctx, core, key, &bound.outs, &bound.ins, &step.scalars, step.shifted.as_ref())?;
            } else {
                launch_store_slab(ctx, core, key, &bound.outs, &bound.ins, &step.scalars, step.shifted.as_ref(), slab)?;
            }
        }
    }
    for step in steps {
        let StepKernel::Store { key, .. } = &step.kernel;
        if store_tiles(key).is_none() {
            note_store(ctx, key);
        }
    }
    Ok(())
}

/// The slabs of `stage` in the order it visits them.
fn slab_order(open: &Open, stage: &Stage, slabs: usize) -> Vec<usize> {
    if stage.order == SlabOrder::Ascending || slabs < 2 {
        return (0..slabs).collect();
    }
    // Slabs whose point inputs are all on the device first: they are used before the others make room by evicting them.
    let inputs: Vec<&super::ArrayRef> = open.steps[stage.steps.clone()]
        .iter()
        .flat_map(|step| step.binds.iter())
        .filter(|bind| bind.access == Access::Read && bind.reach == Reach::Point)
        .map(|bind| &open.arrays[bind.array as usize])
        .filter(|array| array.segments.len() == slabs && array.first_write.is_none_or(|at| !stage.steps.contains(&(at as usize))))
        .collect();
    let resident = |slab: usize| inputs.iter().all(|array| array.segments[slab].on_device());
    let (mut first, rest): (Vec<usize>, Vec<usize>) = (0..slabs).partition(|&slab| resident(slab));
    first.extend(rest);
    first
}
