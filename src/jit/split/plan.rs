//! Stage formation: which steps share one pass over the slabs, and which intermediates never need a buffer.

use std::sync::Arc;

use std::sync::atomic::Ordering;

use super::slab::{self, Chosen, Env, Facts};
use super::{Access, Open, Origin, Reach, SlabRange, Space, Step, StepKernel};
use crate::context::Context;
use crate::jit::launch::{store_tiles, GhostSlab};
use crate::jit::structured::select::select_in;
use crate::jit::structured::strategy::{Residencies, SplitClass, Strategy, Subject};
use crate::profile::{Decision, DecisionSite};

/// How a stage visits its slabs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum SlabOrder {
    /// Slabs whose inputs are all on the device first, then the rest, each ascending: what is resident is used before it is evicted.
    ResidentFirst,
    /// 0, 1, 2, ...: a step reads a whole buffer or writes in place.
    Ascending,
}

/// An array that is written and read inside one stage and nowhere else: it gets no buffer of its own, only a slab-sized one that
/// every slab reuses (shared with the transients of the same slab size whose live ranges do not overlap).
pub(super) struct Transient {
    pub(super) array: u32,
    pub(super) colour: usize,
}

/// A halo edge the stage contracts: the transient `array`, written by the step `producer` and read through a shift or view by the step
/// `consumer` (indices into the stage's steps), as input `k` of its kernel (`bind`: its place among the step's binds). Per slab the
/// producer runs over the regions of `array` the consumer reads (see `launch::GhostSlab::regions`) instead of over the slab.
pub(super) struct Expansion {
    pub(super) producer: usize,
    pub(super) consumer: usize,
    pub(super) array: u32,
    pub(super) k: usize,
}

/// Steps `steps` (a range of the recording order) that run slab by slab together.
pub(super) struct Stage {
    pub(super) steps: std::ops::Range<usize>,
    pub(super) order: SlabOrder,
    pub(super) transients: Vec<Transient>,
    /// Bytes of one buffer per colour (a slab of the arrays sharing it).
    pub(super) colours: Vec<u64>,
    /// The halo edges recomputed per slab instead of cut.
    pub(super) expansions: Vec<Expansion>,
    /// Elements one offset alignment covers for the least dense array of the stage (slabs and regions start on a multiple of it).
    pub(super) align: usize,
    /// Elements of a slab: every step runs over `[s * slab, (s + 1) * slab)` before the next slab starts.
    pub(super) slab: usize,
    /// What the planner chose and why (for the decision record); `None` for a stage that runs by segments.
    pub(super) choice: Option<Chosen>,
}

impl Stage {
    pub(super) fn space(&self, open: &Open) -> Space {
        open.steps[self.steps.start].space
    }

    /// Slabs of the stage.
    pub(super) fn slabs(&self, open: &Open) -> usize {
        self.space(open).len.div_ceil(self.slab.max(1))
    }
}

/// Splits the steps, in recording order, into stages. The recording order is a valid sequential semantics, so a step may join the
/// current stage iff the slab-major order computes what the sequential one does:
/// - it has the stage's iteration space;
/// - it reads no array the stage wrote through anything but its own element (a halo, a whole buffer, data-dependent indices: those
///   edges are cut, which is the kernel-major behaviour for that edge only);
/// - it writes no array a step of the stage read through such a reach (the neighbour slab would read the new value).
pub(super) fn compile(ctx: &Context, open: &mut Open) -> Vec<Stage> {
    let mut stages = form(ctx, open);
    // Re-selection (D9): a stage that cannot fit asks the registry for another strategy for each structured step in it, once per
    // step; the stages are then formed again, because the class of a step decides which edges cut.
    let mut changed = false;
    for stage in &stages {
        if fits(ctx, open, stage) {
            continue;
        }
        for step in &mut open.steps[stage.steps.clone()] {
            changed |= reselect(ctx, step);
        }
    }
    if changed {
        stages = form(ctx, open);
    }
    stages
}

/// Whether `stage` can run in one pass over its slabs: the floor of its device memory (one slab of every array it binds, see
/// [`floor_bytes`]) is within the budget. The same signal as the cut of `run::fit`, which comes after: a stage below it re-selects
/// first. Without a budget everything fits.
fn fits(ctx: &Context, open: &Open, stage: &Stage) -> bool {
    #[cfg(test)]
    if INFEASIBLE.load(Ordering::Relaxed) == ctx.id() {
        return false;
    }
    ctx.pager().budget().is_none_or(|budget| floor_bytes(open, stage) <= budget)
}

/// The context whose stages are infeasible (tests; 0: none).
#[cfg(test)]
pub(super) static INFEASIBLE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// The strategies re-selection chooses among on one context, instead of the family's table (tests).
#[cfg(test)]
pub(super) static TEST_ENTRIES: std::sync::Mutex<Option<(u64, &'static [Strategy])>> = std::sync::Mutex::new(None);

/// The table re-selection looks in for `family` on `ctx`.
pub(super) fn entries_for(ctx: &Context, family: crate::jit::structured::strategy::FamilyTag) -> &'static [Strategy] {
    #[cfg(test)]
    if let Some((id, entries)) = *TEST_ENTRIES.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
        && id == ctx.id()
    {
        return entries;
    }
    let _ = ctx;
    crate::jit::structured::strategy::table(family)
}

/// D9: asks `select` for a strategy of the step's family that admits streamed operands and differs from the current one, and puts
/// it in the step's place: its `origin`, its `class`, and a kernel that runs the strategy whole when the stage is reached. At most
/// once per step and commit. Only for a structured step of a registered value.
fn reselect(ctx: &Context, step: &mut Step) -> bool {
    let Origin::Structured { value: Some(value), strategy, params: _ } = &step.origin else { return false };
    if step.reselected {
        return false;
    }
    let family = strategy.family();
    let entries = entries_for(ctx, family);
    let subject = Subject::Comp(value);
    let Some(selection) = select_in(entries, ctx, &subject, &Residencies::STREAMED, None) else { return false };
    let current = *strategy;
    let Some((id, chosen)) = std::iter::once((selection.id, selection.params)).chain(selection.rest.iter()).find(|&(id, _)| id != current) else { return false };
    let Some(entry) = entries.iter().find(|entry| entry.id == id) else { return false };
    let class = (entry.split)(&subject, &chosen);
    let (op, location) = subject.site();
    if ctx.jit().recording_decisions() {
        ctx.jit().note_decision(Decision::new(op, location, DecisionSite::Schedule, format!("reselect {} instead of {}", entry.name, current_name(entries, current))));
    }
    ctx.stream_counters().reselections.fetch_add(1, Ordering::Relaxed);
    step.origin = Origin::Structured { value: Some(std::sync::Arc::clone(value)), strategy: id, params: chosen };
    step.class = class;
    step.kernel = StepKernel::Relowered;
    step.reselected = true;
    true
}

fn current_name(entries: &[Strategy], id: crate::jit::structured::strategy::StrategyId) -> &'static str {
    entries.iter().find(|entry| entry.id == id).map_or("canonical-blocks", |entry| entry.name)
}

/// The stages of the steps as they are (see [`compile`]).
fn form(ctx: &Context, open: &Open) -> Vec<Stage> {
    let mut stages: Vec<Stage> = Vec::new();
    let mut written: Vec<bool> = vec![false; open.arrays.len()];
    let mut writer: Vec<usize> = vec![0; open.arrays.len()];
    let mut far_read: Vec<bool> = vec![false; open.arrays.len()];
    let mut start = 0;
    for (at, step) in open.steps.iter().enumerate() {
        // A step run whole by its strategy (re-selection) is a stage of its own, and so is a step that needs the whole array.
        let alone = |step: &Step| matches!(step.kernel, StepKernel::Relowered) || step.class == SplitClass::Global;
        let joins = at > start
            && !alone(step)
            && !alone(&open.steps[start])
            && step.space == open.steps[start].space
            && step.binds.iter().enumerate().all(|(at_bind, bind)| {
                let array = bind.array as usize;
                match bind.access {
                    // A halo read of what the stage wrote joins when the producer can be recomputed over the halo (checked again, with
                    // the slab length, when the stage is finished).
                    Access::Read => {
                        bind.reach == Reach::Point
                            || !written[array]
                            || (bind.reach == Reach::Halo && expandable(ctx, open, writer[array], step, at_bind))
                    }
                    Access::Write | Access::ReadWrite => !far_read[array],
                }
            });
        if at > start && !joins {
            stages.extend(finish(ctx, open, start..at));
            written.fill(false);
            far_read.fill(false);
            start = at;
        }
        for bind in step.binds.iter() {
            let array = bind.array as usize;
            match bind.access {
                Access::Read => far_read[array] |= bind.reach != Reach::Point,
                Access::Write | Access::ReadWrite => {
                    written[array] = true;
                    writer[array] = at;
                }
            }
        }
    }
    if start < open.steps.len() {
        stages.extend(finish(ctx, open, start..open.steps.len()));
    }
    stages
}

/// The ghost inputs of a step: `(k, index among the step's binds)` of every input of its kernel read through a window and a ghost.
fn ghost_inputs(step: &Step) -> Vec<(usize, usize)> {
    let StepKernel::Store { key, .. } = &step.kernel else { return Vec::new() };
    let mut found = Vec::new();
    let mut at = step.n_out;
    for (k, span) in key.spans.iter().enumerate() {
        if key.in_place.iter().any(|&(paired, _)| paired as usize == k) {
            continue;
        }
        if span.is_ghost() {
            found.push((k, at));
        }
        at += 1;
    }
    found
}

/// Whether step `writer` can be recomputed over the halo that `step` reads of its output through its bind `at_bind`: the producer is a
/// plain store of one output over points (a producer with shifts of its own cuts: radii would add) and the consumer reads the array
/// through a ghost. Both are stores: a reduction or a re-selected step never takes part in an expansion.
fn expandable(ctx: &Context, open: &Open, writer: usize, step: &Step, at_bind: usize) -> bool {
    let producer = &open.steps[writer];
    let (StepKernel::Store { key: pkey, .. }, StepKernel::Store { key: ckey, .. }) = (&producer.kernel, &step.kernel) else { return false };
    ctx.slab_expand()
        && producer.n_out == 1
        && producer.shifted.is_none()
        && pkey.in_place.is_empty()
        && store_tiles(pkey).is_none()
        && store_tiles(ckey).is_none()
        && producer.space == step.space
        && producer.binds[producer.n_out..].iter().all(|bind| matches!(bind.reach, Reach::Point | Reach::Whole))
        && step.shifted.is_some()
        && ghost_inputs(step).iter().any(|&(_, bind)| bind == at_bind)
}

/// The slab order and the transients of the stages of `steps`: one stage, or several where a halo edge has to be cut after all.
pub(super) fn finish(ctx: &Context, open: &Open, steps: std::ops::Range<usize>) -> Vec<Stage> {
    match finish_with_edges(ctx, open, steps.clone()) {
        Ok(stage) => vec![stage],
        Err(at) => {
            let mut stages = finish(ctx, open, steps.start..at);
            stages.extend(finish(ctx, open, at..steps.end));
            stages
        }
    }
}

/// [`finish`] of one stage, or the step at which it has to be cut: a consumer whose halo edge cannot be contracted.
fn finish_with_edges(ctx: &Context, open: &Open, steps: std::ops::Range<usize>) -> Result<Stage, usize> {
    let space = open.steps[steps.start].space;
    // The halo edges: a consumer reads through a ghost an array that an earlier step of the stage wrote.
    let mut expansions: Vec<Expansion> = Vec::new();
    for consumer in steps.clone() {
        let step = &open.steps[consumer];
        for (k, at_bind) in ghost_inputs(step) {
            let array = step.binds[at_bind].array;
            if step.binds[at_bind].reach != Reach::Halo {
                continue;
            }
            let Some(producer) = (steps.start..consumer).find(|&q| open.steps[q].binds[..open.steps[q].n_out].iter().any(|bind| bind.array == array)) else { continue };
            let entry = &open.arrays[array as usize];
            // The array has no other use than the producer's write and this read, and nothing holds it.
            let uses = open.steps.iter().filter(|other| other.binds.iter().any(|bind| bind.array == array)).count();
            let alone = uses == 2
                && entry.first_use as usize == producer
                && entry.last_use as usize == consumer
                && entry.first_write == Some(producer as u32)
                && !entry.read_write
                && entry.meta.len == space.len
                && entry.segments.iter().all(|segment| Arc::strong_count(segment) == 1 && segment.home().is_none());
            if !alone || !expandable(ctx, open, producer, step, at_bind) {
                return Err(consumer);
            }
            expansions.push(Expansion { producer: producer - steps.start, consumer: consumer - steps.start, array, k });
        }
    }
    // A stage with a reduction or a re-selected step runs its slabs as segments (the carries are per segment, and the canonical blocks
    // of a float reduction tile the segment), so no halo edge is contracted in it: the consumer starts a stage of its own.
    let stores_only = open.steps[steps.clone()].iter().all(|step| matches!(step.kernel, StepKernel::Store { .. }));
    if !stores_only && let Some(first) = expansions.first() {
        return Err(steps.start + first.consumer);
    }
    let whole_or_in_place = open.steps[steps.clone()]
        .iter()
        .flat_map(|step| step.binds.iter())
        .any(|bind| bind.reach == Reach::Whole || bind.access == Access::ReadWrite);
    let order = if whole_or_in_place { SlabOrder::Ascending } else { SlabOrder::ResidentFirst };

    // Transients: first used by a write inside the stage, last used inside it, never read through a neighbour (those edges are cut),
    // and held by nothing but the schedule. A segment with a home (a file) is never one: dropping its array must leave the data in the file.
    let mut candidates: Vec<(u32, crate::gpu_type::ElemLayout, (u32, u32))> = Vec::new();
    for (index, array) in open.arrays.iter().enumerate() {
        let Some(first_write) = array.first_write else { continue };
        let inside = steps.contains(&(array.first_use as usize)) && steps.contains(&(array.last_use as usize));
        if !inside
            || array.read_write
            || array.first_use != first_write
            || array.meta.len != space.len
            || array.meta.seg_len != space.seg_len
            || array.segments.iter().any(|segment| Arc::strong_count(segment) != 1 || segment.home().is_some())
        {
            continue;
        }
        candidates.push((index as u32, array.meta.layout, (array.first_use, array.last_use)));
    }
    // The slab length. A stage whose arrays fit the budget together runs resident: its slab is the planner's (below the segment when
    // that pays). A stage that needs a ring (`ring::choose`) streams segment by segment, and so does one with a reduction, a
    // re-selected step or a tiled kernel: its slab is the segment (the ring slots and uploads are segment-sized).
    let tiled = open.steps[steps.clone()].iter().any(|step| matches!(&step.kernel, StepKernel::Store { key, .. } if store_tiles(key).is_some()));
    let facts = facts(ctx, open, &steps, &candidates, tiled);
    let segment_slab = space.seg_len.min(space.len).max(1);
    let by_segment = !stores_only || tiled || (facts.uniform && !resident(ctx, open, &steps, &candidates, segment_slab));
    let (slab, choice) = if by_segment {
        (segment_slab, None)
    } else {
        let env = Env {
            cost: ctx.cost_model(),
            memory: ctx.memory_model(),
            budget: ctx.pager().budget(),
            max_buffer_bytes: ctx.max_buffer_bytes(),
            pinned_bytes: ctx.slab_bytes(),
            pinned_len: ctx.pinned_slab_len(),
        };
        let choice = slab::choose(&facts, &env);
        (choice.len, Some(choice))
    };
    // What the producers of the halo edges compute per slab: the body, its halos and the far strips of wrapping shifts, each started on
    // the alignment. An edge whose strips cost more than recomputing at most 75% over the body is cut (the halo is wider than a quarter
    // of the slab, or the shift reaches across the array).
    let mut expanded_elems: Vec<usize> = Vec::with_capacity(expansions.len());
    for e in &expansions {
        let step = &open.steps[steps.start + e.consumer];
        let Some(shifted) = step.shifted.as_ref() else { return Err(steps.start + e.consumer) };
        let input = open.arrays[e.array as usize].as_seg(ctx);
        let (mut total, mut widest) = (0usize, 0usize);
        for a in (0..space.len).step_by(slab.max(1)) {
            let plan = GhostSlab::plan(shifted, e.k, input, SlabRange { a, n: slab.min(space.len - a) }, true, facts.align_elems, true);
            total += plan.used();
            widest = widest.max(plan.used());
        }
        if total > space.len + space.len * 3 / 4 + space.len.div_ceil(slab.max(1)) * 2 * facts.align_elems {
            return Err(steps.start + e.consumer);
        }
        expanded_elems.push(widest);
    }
    // One colour per group of non-overlapping live ranges of the same buffer size: the first colour whose last range ended before this
    // one starts (visiting by start gives the fewest colours).
    candidates.sort_by_key(|&(index, _, (start, _))| (start, index));
    let mut colours: Vec<(u64, u32)> = Vec::new();
    let mut transients = Vec::with_capacity(candidates.len());
    for (array, layout, (start, end)) in candidates {
        let elems = expansions.iter().zip(&expanded_elems).find(|(e, _)| e.array == array).map_or(slab.min(space.len), |(_, &widest)| widest);
        let bytes = layout.bytes(elems);
        match colours.iter().position(|&(size, last)| size == bytes && last < start) {
            Some(colour) => {
                colours[colour].1 = end;
                transients.push(Transient { array, colour });
            }
            None => {
                colours.push((bytes, end));
                transients.push(Transient { array, colour: colours.len() - 1 });
            }
        }
    }
    Ok(Stage { steps, order, transients, colours: colours.into_iter().map(|(bytes, _)| bytes).collect(), expansions, align: facts.align_elems, slab, choice })
}

/// Whether the arrays of a stage fit the device budget together with segment-sized buffers for its transients: the stage then needs no
/// ring (`ring::choose` returns `None`) and runs resident. Without a budget everything does.
fn resident(ctx: &Context, open: &Open, steps: &std::ops::Range<usize>, transients: &[(u32, crate::gpu_type::ElemLayout, (u32, u32))], slab: usize) -> bool {
    let Some(budget) = ctx.pager().budget() else { return true };
    let max = ctx.max_buffer_bytes();
    let class = |bytes: u64| luma_core::hal::size_class(bytes, max).unwrap_or(bytes);
    let mut total: u64 = transients.iter().map(|&(_, layout, _)| class(layout.bytes(slab))).sum();
    let mut seen: Vec<bool> = vec![false; open.arrays.len()];
    for bind in open.steps[steps.clone()].iter().flat_map(|step| step.binds.iter()) {
        let array = bind.array as usize;
        if std::mem::replace(&mut seen[array], true) || transients.iter().any(|&(t, ..)| t == bind.array) {
            continue;
        }
        let entry = &open.arrays[array];
        total += entry.meta.layout.bytes(entry.meta.len);
    }
    total <= budget
}

/// What the slab chooser needs to know of a stage (see [`slab::Facts`]).
fn facts(ctx: &Context, open: &Open, steps: &std::ops::Range<usize>, transients: &[(u32, crate::gpu_type::ElemLayout, (u32, u32))], one_slab: bool) -> Facts {
    let len = open.steps[steps.start].space.len;
    let align = usize::try_from(ctx.hal().storage_offset_align()).unwrap_or(usize::MAX);
    let mut f = Facts {
        len,
        steps: steps.len(),
        seg_min: usize::MAX,
        max_stride: 4,
        align_elems: 1,
        transient_stride: 0,
        halo: 0,
        ghost_stride: 0,
        block: 1,
        one_slab,
        uniform: true,
    };
    for &(_, layout, _) in transients {
        f.transient_stride += layout.stride() as u64;
        f.max_stride = f.max_stride.max(layout.stride() as u64);
        f.align_elems = f.align_elems.max(align.saturating_mul(layout.per_word()));
    }
    for step in &open.steps[steps.clone()] {
        for bind in step.binds.iter() {
            let array = &open.arrays[bind.array as usize];
            if bind.reach == Reach::Whole || transients.iter().any(|&(t, ..)| t == bind.array) {
                continue;
            }
            f.seg_min = f.seg_min.min(array.meta.seg_len);
            f.uniform &= array.meta.seg_len == step.space.seg_len;
            f.max_stride = f.max_stride.max(array.meta.layout.stride() as u64);
            f.align_elems = f.align_elems.max(align.saturating_mul(array.meta.layout.per_word()));
        }
    }
    for step in &open.steps[steps.clone()] {
        let StepKernel::Store { key, .. } = &step.kernel else { continue };
        let (Some(shifted), true) = (step.shifted.as_ref(), key.spans.iter().any(|span| span.is_ghost())) else { continue };
        f.halo = f.halo.max(shifted.halo());
        let mut inputs = step.binds[step.n_out..].iter();
        for (k, span) in key.spans.iter().enumerate() {
            if key.in_place.iter().any(|&(paired, _)| paired as usize == k) {
                continue;
            }
            let bind = inputs.next();
            if let (true, Some(bind)) = (span.is_ghost(), bind) {
                f.ghost_stride += open.arrays[bind.array as usize].meta.layout.stride() as u64;
            }
        }
    }
    if align == 0 {
        // No offset bindings: a slab is a segment.
        f.one_slab = f.one_slab || f.seg_min == usize::MAX;
        f.align_elems = f.seg_min.min(len).max(1);
    }
    f
}

/// Bytes of the least device memory a stage needs: one slab of every array it binds (a ring of window 1 with everything streamed)
/// and the buffers of its transients. Below this the stage cannot run in one pass over the slabs.
pub(super) fn floor_bytes(open: &Open, stage: &Stage) -> u64 {
    let mut seen: Vec<bool> = vec![false; open.arrays.len()];
    let mut total: u64 = stage.colours.iter().sum();
    for bind in open.steps[stage.steps.clone()].iter().flat_map(|step| step.binds.iter()) {
        let array = bind.array as usize;
        if std::mem::replace(&mut seen[array], true) || stage.transients.iter().any(|t| t.array == bind.array) {
            continue;
        }
        let entry = &open.arrays[array];
        total += if entry.segments.len() == 1 { entry.meta.layout.bytes(entry.meta.len) } else { entry.meta.layout.bytes(entry.meta.seg_len) };
    }
    total
}

/// Cuts `stage` in two at the step boundary that leaves the smaller of the two floors (ties: the fewer arrays live across the
/// cut), or `None` for a stage of one step. What the first half wrote and the second reads is no longer a transient: it is retained,
/// which is the kernel-major behaviour for that edge only. A half whose halo edge cannot be contracted any more is several stages.
pub(super) fn cut_stage(ctx: &Context, open: &Open, stage: &Stage) -> Option<Vec<Stage>> {
    let range = stage.steps.clone();
    let mut best: Option<((u64, usize), Vec<Stage>)> = None;
    for at in range.start + 1..range.end {
        let mut halves = finish(ctx, open, range.start..at);
        halves.extend(finish(ctx, open, at..range.end));
        let live = open.arrays.iter().filter(|array| (array.first_use as usize) < at && (array.last_use as usize) >= at).count();
        let key = (halves.iter().map(|half| floor_bytes(open, half)).max().unwrap_or(0), live);
        if best.as_ref().is_none_or(|(known, _)| key < *known) {
            best = Some((key, halves));
        }
    }
    best.map(|(_, halves)| halves)
}
