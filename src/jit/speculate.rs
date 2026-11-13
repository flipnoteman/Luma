//! Speculation at per-step readbacks.
//!
//! A loop `next = step(&u); e = f(&next, &u).max().item()?; u = next;` waits for the GPU once per step,
//! though the step does little work. Once the same step has been seen repeating, a readback also
//! computes the next few steps and their residuals and reads everything in one round trip. The
//! readbacks that follow are answered from memory, if the expression they reduce is the same
//! program over the predicted states and nothing it reads was overwritten meanwhile; the nodes
//! of their next states take the precomputed buffers. Anything else is a miss and runs as
//! usual, so values are those of the program that was written. `LUMA_SPECULATE=off` disables it.

use std::collections::VecDeque;
use std::sync::{Arc, Weak};
use std::time::{Duration, Instant};

use crate::array::{Failure, Storage};
use crate::builtin::{Finish, ReduceList};
use crate::context::{Context, Scratch};
use crate::error::{LumaError, Result};
use crate::reduce::finish_passes;

use super::program::{flatten_step, Fused, FusedKey, StateRef, Tail, Traced};
use super::{launch, ElemTy, Node, Src};

/// Most steps computed ahead by one readback.
const SPEC_MAX_DEPTH: usize = 16;
/// Most nodes looked at when looking for the states of a step.
const WALK_MAX: usize = 1024;
/// A round trip that is not hidden by speculation, assumed until measured.
const FLOOR: Duration = Duration::from_micros(1);
/// Readbacks that go without speculation after three misses in a row.
const COOLDOWN: u32 = 16;

/// The step program of the last readback that could be traced.
#[derive(Debug)]
struct Template {
    /// The reduction of the readback; inputs `0..states` are the states.
    key: FusedKey,
    scalars: Vec<u32>,
    states: usize,
    /// The inputs after the states. Weak: a finished loop must not keep its buffers alive.
    fixed: Vec<Weak<Storage>>,
    /// The node of each instruction. Weak, so that a freed address never compares equal.
    members: Vec<Weak<Node>>,
}

impl Template {
    /// Whether `fused` is this step: the same program with the same scalar values and the same
    /// buffers behind every input after the states.
    fn is_step(&self, fused: &Fused) -> bool {
        self.key == fused.key
            && self.scalars == fused.scalars
            && fused.inputs.len() == self.states + self.fixed.len()
            && fused.inputs[self.states..].iter().zip(&self.fixed).all(|(a, w)| std::ptr::eq(Arc::as_ptr(a), w.as_ptr()))
    }
}

/// What was computed ahead.
#[derive(Debug)]
struct Cache {
    /// The states the next readback is expected to read.
    cur: Vec<Arc<Storage>>,
    /// The states after each further step, and the values of the readbacks that follow.
    next: VecDeque<Vec<Arc<Storage>>>,
    values: VecDeque<Vec<u32>>,
    /// Every buffer a prediction was computed from or leads to, with its readers generation at
    /// the time: a write to any of them (see [`super::Readers::flush`]) voids the prediction.
    generations: Vec<(Arc<Storage>, u64)>,
}

/// What [`readback`] remembers of the readbacks before.
#[derive(Debug)]
pub(super) struct Tracker {
    pub(super) enabled: bool,
    /// The nodes the last readback's expression reached.
    prev_members: Vec<Weak<Node>>,
    template: Option<Template>,
    /// The instruction of `template` that computes each next state.
    outs: Option<Vec<u32>>,
    repeats: u32,
    pub(super) hits: u64,
    misses_in_row: u32,
    /// Readbacks left to skip after repeated misses.
    cooldown: u32,
    /// The shortest readback without speculation, and the shortest time per step computed ahead
    /// seen so far.
    wait: Option<Duration>,
    step: Option<Duration>,
    cache: Option<Cache>,
}

impl Tracker {
    pub(super) fn new(enabled: bool) -> Self {
        Tracker {
            enabled,
            prev_members: Vec::new(),
            template: None,
            outs: None,
            repeats: 0,
            hits: 0,
            misses_in_row: 0,
            cooldown: 0,
            wait: None,
            step: None,
            cache: None,
        }
    }

    pub(super) fn clear_cache(&mut self) {
        self.cache = None;
    }

    fn forget_step(&mut self) {
        self.template = None;
        self.outs = None;
        self.repeats = 0;
    }

    /// How many steps to compute ahead: enough to hide the round trip behind the work, and no
    /// more than a tenth of the confirmed steps, which bounds the work wasted where a loop ends.
    fn depth(&self) -> usize {
        let hidden = match (self.wait, self.step) {
            (Some(wait), Some(step)) => {
                let step = step.max(FLOOR);
                usize::try_from(wait.saturating_sub(step).as_nanos().div_ceil(step.as_nanos())).unwrap_or(usize::MAX)
            }
            _ => 1,
        };
        hidden.min((self.hits / 10).max(1) as usize).clamp(1, SPEC_MAX_DEPTH)
    }

    /// Whether the readback of `node` is the one `cache` predicted: the same program over its
    /// states, reading nothing that was overwritten since.
    fn confirm(&self, cache: &Cache, node: &Arc<Node>, tail: Tail) -> Option<Traced> {
        let template = self.template.as_ref()?;
        let outs = self.outs.as_ref()?;
        if cache.cur.len() != template.states
            || cache.generations.iter().any(|(storage, generation)| storage.readers.generation() != *generation)
        {
            return None;
        }
        let states: Vec<StateRef> = cache.cur.iter().cloned().map(StateRef::Buffer).collect();
        let traced = flatten_step(std::slice::from_ref(node), &states, tail)?;
        let known = template.is_step(&traced.fused)
            && outs.iter().all(|&out| traced.members.get(out as usize).is_some_and(Option::is_some));
        known.then_some(traced)
    }

    /// The readback of `node` without speculation.
    fn plain(
        &mut self,
        ctx: &Context,
        count: usize,
        reduce: impl FnOnce(&mut Scratch<'_>) -> Result<wgpu::Buffer>,
    ) -> Result<Vec<u32>> {
        let start = Instant::now();
        let mut scratch = Scratch::new(ctx);
        let buffer = reduce(&mut scratch)?;
        let words = read_words(ctx, &buffer, count)?;
        let waited = start.elapsed();
        self.wait = Some(self.wait.map_or(waited, |wait| wait.min(waited)));
        Ok(words)
    }

    /// The readback of `node`, whose expression is the template `traced` over its states, plus
    /// the following steps.
    fn speculate(
        &mut self,
        ctx: &Context,
        node: &Arc<Node>,
        ops: ReduceList,
        traced: Traced,
        reduce: impl FnOnce(&mut Scratch<'_>) -> Result<wgpu::Buffer>,
    ) -> Result<Vec<u32>> {
        let count = ops.as_slice().len();
        let snapshot = self.template.as_ref().zip(self.outs.as_ref());
        let Some((key, scalars, states, outs)) =
            snapshot.map(|(template, outs)| (template.key.clone(), template.scalars.clone(), template.states, outs.clone()))
        else {
            return self.plain(ctx, count, reduce);
        };
        let nexts: Option<Vec<Arc<Node>>> =
            outs.iter().map(|&out| traced.members.get(out as usize).cloned().flatten()).collect();
        let fixed: Vec<Arc<Storage>> = traced.fused.inputs.get(states..).unwrap_or_default().to_vec();
        let Some(nexts) = nexts.filter(|_| fixed.iter().all(|input| input.segments() == 1)) else {
            return self.plain(ctx, count, reduce);
        };
        drop(traced);

        let start = Instant::now();
        // The next states are computed once, as buffers; the readback then reads them.
        launch::evaluate_group(&nexts);
        let cur: Option<Vec<Arc<Storage>>> = nexts.iter().map(|next| next.evaluated().cloned()).collect();
        let Some(cur) = cur.filter(|cur| cur.iter().all(|storage| storage.segments() == 1)) else {
            return self.plain(ctx, count, reduce);
        };
        drop(nexts);
        let fused = Fused {
            key,
            inputs: cur.iter().chain(&fixed).cloned().collect(),
            scalars,
            fused_nodes: Vec::new(),
            instr_nodes: Vec::new(),
            sub_roots: Vec::new(),
        };

        let mut scratch = Scratch::new(ctx);
        let mut finals = vec![reduce(&mut scratch)?];
        let mut ahead: Vec<Vec<Arc<Storage>>> = Vec::new();
        let wgsl = node.op().out_ty().wgsl();
        for _ in 0..self.depth() {
            let from = ahead.last().unwrap_or(&cur);
            // A failure here only ends the speculation: the steps that were recorded stay valid.
            let Ok((next, residual)) = compute_ahead(ctx, &mut scratch, &fused, states, &outs, node.len(), ops, wgsl, from) else {
                break;
            };
            finals.push(residual);
            ahead.push(next);
        }

        let bytes = (count * 4) as u64;
        let gather = scratch.take(bytes * finals.len() as u64)?;
        for (i, buffer) in finals.iter().enumerate() {
            ctx.copy(buffer, 0, &gather, i as u64 * bytes, bytes)?;
        }
        let words = read_words(ctx, &gather, count * finals.len())?;
        drop(finals);
        drop(scratch);

        let steps = ahead.len();
        if steps > 0 {
            let per_step = start.elapsed().saturating_sub(self.wait.unwrap_or_default()) / steps as u32;
            self.step = Some(self.step.map_or(per_step, |step| step.min(per_step)));
        }
        let generations = fixed
            .iter()
            .chain(cur.iter())
            .chain(ahead.iter().flatten())
            .map(|storage| (Arc::clone(storage), storage.readers.generation()))
            .collect();
        let mut values: VecDeque<Vec<u32>> = words.chunks(count).map(<[u32]>::to_vec).collect();
        let first = values.pop_front().unwrap_or_default();
        self.cache = Some(Cache { cur, next: ahead.into(), values, generations });
        Ok(first)
    }
}

/// The `count` words at the start of `buffer`.
fn read_words(ctx: &Context, buffer: &wgpu::Buffer, count: usize) -> Result<Vec<u32>> {
    let words = ctx.read::<u32>(buffer, (count * 4) as u64)?;
    if words.len() != count {
        return Err(LumaError::Gpu("reduction results are missing".into()));
    }
    Ok(words)
}

/// `fused` with `states` as its first inputs, as a store of the instructions `outs` if given.
fn with_states(fused: &Fused, states: &[Arc<Storage>], outs: Option<&[u32]>) -> Fused {
    let mut fused = fused.clone();
    for (slot, state) in fused.inputs.iter_mut().zip(states) {
        *slot = Arc::clone(state);
    }
    if let Some(outs) = outs {
        fused.key.tail = Tail::Store;
        fused.key.outputs = outs.to_vec();
    }
    fused
}

/// Records one step of the template from `from`: the states after it and the buffer holding the
/// value of its readback.
#[allow(clippy::too_many_arguments)]
fn compute_ahead(
    ctx: &Context,
    scratch: &mut Scratch<'_>,
    template: &Fused,
    states: usize,
    outs: &[u32],
    len: usize,
    ops: ReduceList,
    wgsl: &'static str,
    from: &[Arc<Storage>],
) -> Result<(Vec<Arc<Storage>>, wgpu::Buffer)> {
    debug_assert_eq!(from.len(), states);
    let partials = launch::reduce_fused(ctx, len, with_states(template, from, None), ops, scratch, None, Finish::default()).map_err(Failure::into_error)?;
    let [(partial, groups)] = <[_; 1]>::try_from(partials).map_err(|_| LumaError::Gpu("speculated reduction spans segments".into()))?;
    let residual = finish_passes(ctx, scratch, ops, wgsl, ElemTy::F32.bytes(), partial, groups, true, None, Finish::default())?;
    let next = launch::run_store(ctx, len, with_states(template, from, Some(outs))).map_err(Failure::into_error)?;
    Ok((next.into_iter().map(Arc::new).collect(), residual))
}

/// What a walk over the expression of a readback found.
struct Walk<'a> {
    prev: &'a [Weak<Node>],
    seen: Vec<*const Node>,
    all: Vec<Arc<Node>>,
    /// Nodes of the previous readback's expression that this one's new operations read.
    frontier: Vec<Arc<Node>>,
}

impl Walk<'_> {
    fn in_prev(&self, node: *const Node) -> bool {
        self.prev.iter().any(|weak| std::ptr::eq(weak.as_ptr(), node))
    }

    fn found(&mut self, node: &Arc<Node>) {
        if !self.frontier.iter().any(|known| Arc::ptr_eq(known, node)) {
            self.frontier.push(Arc::clone(node));
        }
    }

    /// Visits `node` and what it reads, operands first, without evaluating anything. `None` if
    /// the expression is too large to be a step.
    fn visit(&mut self, node: &Arc<Node>) -> Option<()> {
        let at = Arc::as_ptr(node);
        if self.seen.contains(&at) {
            return Some(());
        }
        if self.seen.len() >= WALK_MAX {
            return None;
        }
        self.seen.push(at);
        self.all.push(Arc::clone(node));
        if !node.is_pending() {
            return Some(());
        }
        // A node without operands is being evaluated or is a loop leaf: a boundary.
        let Some(args) = node.operands() else { return Some(()) };
        let reader_is_old = self.in_prev(at);
        for arg in args.iter().take(node.op().kind.arity()) {
            match arg {
                Src::Node(operand) => {
                    self.visit(operand)?;
                    if !reader_is_old && self.in_prev(Arc::as_ptr(operand)) {
                        self.found(operand);
                    }
                }
                // An operand that was evaluated before this one was created: it is a buffer
                // here, but a node of the previous expression.
                Src::Buffer(storage) if !reader_is_old => {
                    let old = self
                        .prev
                        .iter()
                        .filter_map(Weak::upgrade)
                        .find(|old| old.evaluated().is_some_and(|result| Arc::ptr_eq(result, storage)));
                    if let Some(old) = old {
                        self.found(&old);
                    }
                }
                _ => {}
            }
        }
        Some(())
    }
}

/// Reads back the reduction `ops` of the expression behind `node`, speculating where the
/// readbacks form a loop. `reduce` runs the plain reduction and returns the buffer holding its
/// values. `Some` is the readback's outcome; `None` means this module does not handle it.
pub(crate) fn readback(
    node: &Arc<Node>,
    ops: ReduceList,
    reduce: impl FnOnce(&mut Scratch<'_>) -> Result<wgpu::Buffer>,
) -> Option<Result<Vec<u32>>> {
    let ctx = node.context();
    let mut guard = ctx.jit().speculation.try_lock().ok()?;
    let tracker = &mut *guard;
    if !tracker.enabled {
        return None;
    }
    if tracker.cooldown > 0 {
        tracker.cooldown -= 1;
        return None;
    }
    let tail = Tail::Reduce(ops);
    let count = ops.as_slice().len();

    if let Some(mut cache) = tracker.cache.take() {
        match tracker.confirm(&cache, node, tail) {
            Some(traced) => {
                let (Some(values), Some(next)) = (cache.values.pop_front(), cache.next.pop_front()) else {
                    // Everything computed ahead was used: go on from the last state.
                    return Some(tracker.speculate(ctx, node, ops, traced, reduce));
                };
                let outs = tracker.outs.as_ref()?;
                for (&out, storage) in outs.iter().zip(&next) {
                    if let Some(Some(state)) = traced.members.get(out as usize) {
                        state.finish_shared(Arc::clone(storage));
                    }
                }
                cache.cur = next;
                tracker.hits += 1;
                tracker.misses_in_row = 0;
                tracker.cache = Some(cache);
                return Some(Ok(values));
            }
            None => {
                if !cache.values.is_empty() {
                    tracker.misses_in_row += 1;
                    if tracker.misses_in_row >= 3 {
                        tracker.cooldown = COOLDOWN;
                        tracker.misses_in_row = 0;
                    }
                }
                tracker.forget_step();
                if tracker.cooldown > 0 {
                    return None;
                }
            }
        }
    }

    let mut walk = Walk { prev: &tracker.prev_members, seen: Vec::new(), all: Vec::new(), frontier: Vec::new() };
    if walk.visit(node).is_none() {
        tracker.prev_members.clear();
        tracker.forget_step();
        return None;
    }
    let (all, frontier) = (walk.all, walk.frontier);
    tracker.prev_members = all.iter().map(Arc::downgrade).collect();
    let traceable = !frontier.is_empty()
        && frontier.iter().all(|state| state.op().out_ty() == ElemTy::F32 && state.len() == node.len());
    let traced = traceable
        .then(|| {
            let states: Vec<StateRef> = frontier.iter().cloned().map(StateRef::Node).collect();
            flatten_step(std::slice::from_ref(node), &states, tail)
        })
        .flatten();
    let Some(traced) = traced else {
        tracker.forget_step();
        return None;
    };

    let before = tracker.template.take();
    let states = frontier.len();
    let same = before.as_ref().is_some_and(|before| before.states == states && before.is_step(&traced.fused));
    tracker.repeats = if same { tracker.repeats + 1 } else { 0 };
    // Where the previous template computed the states this readback reads.
    tracker.outs = before.as_ref().filter(|before| before.states == states).and_then(|before| {
        frontier
            .iter()
            .map(|state| {
                let at = before.members.iter().position(|weak| std::ptr::eq(weak.as_ptr(), Arc::as_ptr(state)))?;
                u32::try_from(at).ok()
            })
            .collect()
    });
    tracker.template = Some(Template {
        key: traced.fused.key.clone(),
        scalars: traced.fused.scalars.clone(),
        states,
        fixed: traced.fused.inputs[states..].iter().map(Arc::downgrade).collect(),
        members: traced.members.iter().map(|member| member.as_ref().map_or_else(Weak::new, Arc::downgrade)).collect(),
    });

    if tracker.repeats >= 2 && tracker.outs.is_some() {
        return Some(tracker.speculate(ctx, node, ops, traced, reduce));
    }
    Some(tracker.plain(ctx, count, reduce))
}
