//! Stage formation: which steps share one pass over the slabs, and which intermediates never need a buffer.

use std::sync::Arc;

use std::sync::atomic::Ordering;

use super::{Access, Open, Origin, Reach, Space, Step, StepKernel};
use crate::context::Context;
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

/// Steps `steps` (a range of the recording order) that run slab by slab together.
pub(super) struct Stage {
    pub(super) steps: std::ops::Range<usize>,
    pub(super) order: SlabOrder,
    pub(super) transients: Vec<Transient>,
    /// Bytes of one buffer per colour (a full segment of the arrays sharing it).
    pub(super) colours: Vec<u64>,
}

impl Stage {
    pub(super) fn space(&self, open: &Open) -> Space {
        open.steps[self.steps.start].space
    }
}

/// Splits the steps, in recording order, into stages. The recording order is a valid sequential semantics, so a step may join the
/// current stage iff the slab-major order computes what the sequential one does:
/// - it has the stage's iteration space;
/// - it reads no array the stage wrote through anything but its own element (a halo, a whole buffer, data-dependent indices: those
///   edges are cut, which is the kernel-major behaviour for that edge only);
/// - it writes no array a step of the stage read through such a reach (the neighbour slab would read the new value).
pub(super) fn compile(ctx: &Context, open: &mut Open) -> Vec<Stage> {
    let mut stages = form(open);
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
        stages = form(open);
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
fn form(open: &Open) -> Vec<Stage> {
    let mut stages: Vec<Stage> = Vec::new();
    let mut written: Vec<bool> = vec![false; open.arrays.len()];
    let mut far_read: Vec<bool> = vec![false; open.arrays.len()];
    let mut start = 0;
    for (at, step) in open.steps.iter().enumerate() {
        // A step run whole by its strategy (re-selection) is a stage of its own, and so is a step that needs the whole array.
        let alone = |step: &Step| matches!(step.kernel, StepKernel::Relowered) || step.class == SplitClass::Global;
        let joins = at > start
            && !alone(step)
            && !alone(&open.steps[start])
            && step.space == open.steps[start].space
            && step.binds.iter().all(|bind| {
                let array = bind.array as usize;
                match bind.access {
                    Access::Read => bind.reach == Reach::Point || !written[array],
                    Access::Write | Access::ReadWrite => !far_read[array],
                }
            });
        if at > start && !joins {
            stages.push(finish(open, start..at));
            written.fill(false);
            far_read.fill(false);
            start = at;
        }
        for bind in step.binds.iter() {
            let array = bind.array as usize;
            match bind.access {
                Access::Read => far_read[array] |= bind.reach != Reach::Point,
                Access::Write | Access::ReadWrite => written[array] = true,
            }
        }
    }
    if start < open.steps.len() {
        stages.push(finish(open, start..open.steps.len()));
    }
    stages
}

/// The slab order and the transients of the stage of `steps`.
pub(super) fn finish(open: &Open, steps: std::ops::Range<usize>) -> Stage {
    let space = open.steps[steps.start].space;
    let whole_or_in_place = open.steps[steps.clone()]
        .iter()
        .flat_map(|step| step.binds.iter())
        .any(|bind| bind.reach == Reach::Whole || bind.access == Access::ReadWrite);
    let order = if whole_or_in_place { SlabOrder::Ascending } else { SlabOrder::ResidentFirst };

    // Transients: first used by a write inside the stage, last used inside it, never read through a neighbour (those edges are cut),
    // and held by nothing but the schedule. A segment with a home (a file) is never one: dropping its array must leave the data in the file.
    let mut candidates: Vec<(u32, u64, (u32, u32))> = Vec::new();
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
        candidates.push((index as u32, array.meta.layout.bytes(space.len.min(space.seg_len)), (array.first_use, array.last_use)));
    }
    // One colour per group of non-overlapping live ranges of the same buffer size: the first colour whose last range ended before this
    // one starts (visiting by start gives the fewest colours).
    candidates.sort_by_key(|&(index, _, (start, _))| (start, index));
    let mut colours: Vec<(u64, u32)> = Vec::new();
    let mut transients = Vec::with_capacity(candidates.len());
    for (array, bytes, (start, end)) in candidates {
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
    Stage { steps, order, transients, colours: colours.into_iter().map(|(bytes, _)| bytes).collect() }
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
/// which is the kernel-major behaviour for that edge only.
pub(super) fn cut_stage(open: &Open, stage: &Stage) -> Option<(Stage, Stage)> {
    let range = stage.steps.clone();
    let mut best: Option<((u64, usize), Stage, Stage)> = None;
    for at in range.start + 1..range.end {
        let (a, b) = (finish(open, range.start..at), finish(open, at..range.end));
        let live = open.arrays.iter().filter(|array| (array.first_use as usize) < at && (array.last_use as usize) >= at).count();
        let key = (floor_bytes(open, &a).max(floor_bytes(open, &b)), live);
        if best.as_ref().is_none_or(|(known, ..)| key < *known) {
            best = Some((key, a, b));
        }
    }
    best.map(|(_, a, b)| (a, b))
}
