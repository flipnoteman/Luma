//! Running a committed schedule: steps are compiled into stages (see `plan`), and a stage runs slab by slab, each slab through every
//! step of the stage, so a segment is brought to the device once per chain.
//!
//! The slab loop of [`run_stage`] is where lanes plug in: a ring of slab buffers filled by upload transfers for streamed inputs
//! and drained by download transfers for streamed outputs.

use std::sync::atomic::Ordering;

use luma_core::hal::size_class;
use luma_core::Buffer;

use super::plan::{compile, cut_stage, floor_bytes, SlabOrder, Stage};
use super::ring::{choose, run_slabs, Choice, Dir};
use super::{Access, Bind, Open, Reach, SegRef, Step, StepKernel};
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
    let stages = fit(ctx, open, compile(open))?;
    let future = ctx.pager().budget().is_some().then(|| Future::new(open, &stages));
    let (up, down) = ctx.pager().traffic();
    let mut slabs = 0;
    let mut elided = 0;
    let mut window = 0;
    let (mut streamed_in, mut streamed_out) = (0, 0);
    let result = (|| {
        for (at, stage) in stages.iter().enumerate() {
            slabs += stage.space(open).slabs();
            elided += stage.transients.iter().map(|t| open.arrays[t.array as usize].meta.layout.bytes(open.arrays[t.array as usize].meta.len)).sum::<u64>();
            let info = run_stage(ctx, open, stage, at, future.as_ref())?;
            window = window.max(info.window);
            streamed_in += info.streamed_in;
            streamed_out += info.streamed_out;
        }
        Ok(())
    })();
    let (up_after, down_after) = ctx.pager().traffic();
    stats.upload_bytes.fetch_add(up_after - up, Ordering::Relaxed);
    stats.download_bytes.fetch_add(down_after - down, Ordering::Relaxed);
    if result.is_ok() && ctx.jit().recording_decisions() {
        let transients = stages.iter().map(|stage| stage.transients.len()).sum::<usize>();
        let retained = (open.arrays.iter().filter(|array| array.first_write.is_some()).count() - transients).saturating_sub(streamed_out);
        let chosen = format!(
            "stages={} slabs={slabs} window={} retain={retained} stream={} elided={transients}",
            stages.len(),
            window.max(1),
            streamed_in + streamed_out
        );
        let writer = open.arrays.iter().find_map(|array| array.writer);
        if let Some((op, location)) = writer {
            ctx.jit().note_decision(Decision::new(op, location, DecisionSite::Schedule, chosen));
        }
    }
    result
}

/// Cuts the stages that cannot run in one pass over their slabs even with every array streamed through a window of one (the last of
/// the fallbacks of D6 this part has: window 1 and streaming outputs happen when a stage runs, a re-selection needs a registry
/// family that gains from it, and shrinking the slab is W3-P4's). A single step that does not fit fails, naming it.
fn fit(ctx: &Context, open: &Open, stages: Vec<Stage>) -> Result<Vec<Stage>, LumaError> {
    let Some(budget) = ctx.pager().budget() else { return Ok(stages) };
    let mut work: std::collections::VecDeque<Stage> = stages.into();
    let mut fitted = Vec::with_capacity(work.len());
    while let Some(stage) = work.pop_front() {
        let floor = floor_bytes(open, &stage);
        if floor <= budget {
            fitted.push(stage);
            continue;
        }
        match cut_stage(open, &stage) {
            Some((first, second)) => {
                ctx.stream_counters().cuts.fetch_add(1, Ordering::Relaxed);
                work.push_front(second);
                work.push_front(first);
            }
            None => {
                let names: Vec<&str> = open.steps[stage.steps.clone()]
                    .iter()
                    .filter_map(|step| open.arrays[step.binds[0].array as usize].writer.map(|(name, _)| name))
                    .collect();
                return Err(LumaError::Gpu(format!(
                    "out of memory: the stage of steps {names:?} needs {floor} bytes for one slab of each array it binds, and the memory budget is {budget} bytes"
                )));
            }
        }
    }
    Ok(fitted)
}

/// The arrays one step binds, as launches take them.
pub(super) struct Bound<'a> {
    pub(super) outs: Vec<SegRef<'a>>,
    pub(super) ins: Vec<SegRef<'a>>,
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

/// What a stage did, for the decision log.
struct StageInfo {
    window: usize,
    streamed_in: usize,
    streamed_out: usize,
}

/// The arrays of `steps` as launches take them. `fixed` gives the buffer an array is bound as for every segment (a transient's
/// colour, a streamed array's ring slot), `None` for the arrays that are pinned.
pub(super) fn bind_all<'a>(ctx: &'a Context, open: &'a Open, steps: &'a [Step], fixed: &dyn Fn(u32) -> Option<&'a Buffer>) -> Vec<Bound<'a>> {
    steps
        .iter()
        .map(|step| {
            let seg = |bind: &Bind| open.arrays[bind.array as usize].as_seg(ctx).fixed_to(fixed(bind.array));
            Bound { outs: step.binds[..step.n_out].iter().map(seg).collect(), ins: step.binds[step.n_out..].iter().map(seg).collect() }
        })
        .collect()
}

/// One step on one slab: the launches the eager path makes, with the same arguments.
pub(super) fn launch_step(ctx: &Context, step: &Step, bound: &Bound<'_>, slab: usize) -> Result<(), LumaError> {
    let StepKernel::Store { core, key } = &step.kernel;
    if store_tiles(key).is_some() {
        // One launch over the one segment.
        launch_store(ctx, core, key, &bound.outs, &bound.ins, &step.scalars, step.shifted.as_ref())
    } else {
        launch_store_slab(ctx, core, key, &bound.outs, &bound.ins, &step.scalars, step.shifted.as_ref(), slab)
    }
}

fn run_stage(ctx: &Context, open: &Open, stage: &Stage, at: usize, future: Option<&Future>) -> Result<StageInfo, LumaError> {
    let stats = ctx.stream_counters();
    let steps = &open.steps[stage.steps.clone()];

    // What the stage streams through a ring instead of making resident (see `ring`).
    let next_stage = |array: u32| future.and_then(|future| future.stages_of[array as usize].iter().copied().find(|&s| s as usize > at)).map_or(u64::MAX, u64::from);
    let choice: Option<Choice> = choose(ctx, open, stage, &next_stage);
    let ring_bytes = choice.as_ref().map_or(0, Choice::ring_bytes);

    // Transients get a slab-sized buffer per colour, streamed arrays a ring. Making room for them evicts by next use (the schedule
    // knows the future).
    let mut buffers = StageBuffers { ctx, buffers: Vec::with_capacity(stage.colours.len()) };
    let max = ctx.max_buffer_bytes();
    let total: u64 = stage.colours.iter().map(|&bytes| size_class(bytes, max).unwrap_or(bytes)).sum::<u64>() + ring_bytes;
    let _reservation = if total > 0 {
        let keep: FxSet<*const Segment> = steps
            .iter()
            .flat_map(|step| step.binds.iter())
            .filter(|bind| !stage.transients.iter().any(|t| t.array == bind.array) && !choice.as_ref().is_some_and(|choice| choice.streams_array(bind.array)))
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
    let fixed = |array: u32| stage.transients.iter().find(|t| t.array == array).map(|t| &buffers.buffers[t.colour]);

    let space = stage.space(open);
    let slabs = space.slabs();
    stats.stages.fetch_add(1, Ordering::Relaxed);
    for transient in &stage.transients {
        let meta = open.arrays[transient.array as usize].meta;
        stats.transient_bytes_elided.fetch_add(meta.layout.bytes(meta.len), Ordering::Relaxed);
        stats.transients.fetch_add(1, Ordering::Relaxed);
    }
    let order = slab_order(open, stage, slabs);
    let mut info = StageInfo { window: 0, streamed_in: 0, streamed_out: 0 };
    // The single place a launch window (W3-STREAMS) opens around the stage's launches.
    if let Some(choice) = &choice {
        stats.streamed.fetch_add(choice.streams.len() as u64, Ordering::Relaxed);
        stats.ring_bytes_peak.fetch_max(ring_bytes, Ordering::Relaxed);
        if choice.window == 1 {
            stats.window1_fallbacks.fetch_add(1, Ordering::Relaxed);
        }
        info.window = choice.window;
        info.streamed_in = choice.streams.iter().filter(|s| s.dir == Dir::In).count();
        info.streamed_out = choice.streams.len() - info.streamed_in;
        #[cfg(test)]
        if super::tests::fail_here(order[0]) {
            return Err(LumaError::Gpu("injected failure".into()));
        }
        stats.slabs.fetch_add(slabs as u64, Ordering::Relaxed);
        run_slabs(ctx, open, stage, choice, &order, &fixed)?;
    } else {
        let bound = bind_all(ctx, open, steps, &fixed);
        for slab in order {
            #[cfg(test)]
            if super::tests::fail_here(slab) {
                return Err(LumaError::Gpu("injected failure".into()));
            }
            stats.slabs.fetch_add(1, Ordering::Relaxed);
            for (step, bound) in steps.iter().zip(&bound) {
                launch_step(ctx, step, bound, slab)?;
            }
        }
    }
    for step in steps {
        let StepKernel::Store { key, .. } = &step.kernel;
        if store_tiles(key).is_none() {
            note_store(ctx, key);
        }
    }
    Ok(info)
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
