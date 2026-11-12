//! Loops found while following `eval`.
//!
//! A program that advances arrays with `x = step(&x); x.eval()` builds the same expression over
//! the previous results again and again. [`JitState::step_boundary`] sees every `eval`: when the
//! requested arrays come from those of the previous call through the same step program (the same
//! structure, scalar values and fixed input buffers), they become *loop leaves*, nodes backed by
//! a [`LoopPlan`] that runs all the accumulated steps at once, as one register loop, one
//! time-blocked stencil or step by step. The graph that is pending stays one step deep however
//! long the loop runs, so the fusion limit never cuts it, and the kernels no longer depend on
//! how many steps were postponed.

use std::sync::{Arc, Mutex, OnceLock, Weak};

use crate::array::{Array, Failure, Storage};
use crate::context::Context;
use crate::error::LumaError;
use crate::shape::Shape;

use super::program::{flatten_step, same_program, Fused, FusedKey, StateRef, Tail};
use super::{launch, lock, stencil, JitState, Node, OpKind, Src};

/// Instructions per element after which a plan runs without waiting for the GPU to go idle, so
/// that the GPU computes these steps while the host records the next ones. Per element, so that
/// the states' memory traffic stays small against the work at any array size.
const PLAN_INSTRS: u64 = 1024;

/// What a plan runs, until it does.
struct Parts {
    /// The step, with one output per state; inputs `0..m` are the states, the others stay fixed.
    body: Fused,
    /// The state values before the first step: a buffer or a pending node.
    inits: Vec<Src>,
}

/// `steps` applications of a step program to states, evaluated lazily and once.
pub(super) struct LoopPlan {
    /// Released when the plan runs, so that a loop leaf that stays alive does not keep the
    /// initial values and fixed inputs of its loop alive.
    parts: Mutex<Option<Parts>>,
    pub(super) steps: u32,
    len: usize,
    shape: Shape,
    /// Sorted ids of the buffers the plan reads: the fixed inputs, the buffer states and those
    /// the pending states read.
    pub(super) input_ids: Vec<u64>,
    result: OnceLock<Result<(), Arc<LumaError>>>,
    /// The states after the last step, each handed out to the loop leaf it belongs to.
    outputs: Mutex<Vec<Option<Arc<Storage>>>>,
}

impl LoopPlan {
    fn new(body: Fused, inits: Vec<Src>, steps: u32, len: usize, shape: Shape) -> LoopPlan {
        let states = inits.len();
        let mut input_ids: Vec<u64> = body.inputs.iter().skip(states).map(|input| input.id()).collect();
        for init in &inits {
            match init {
                Src::Buffer(storage) => input_ids.push(storage.id()),
                Src::Node(node) => input_ids.extend_from_slice(node.inputs()),
                Src::Scalar(_) => {}
            }
        }
        input_ids.sort_unstable();
        input_ids.dedup();
        LoopPlan {
            parts: Mutex::new(Some(Parts { body, inits })),
            steps,
            len,
            shape,
            input_ids,
            result: OnceLock::new(),
            outputs: Mutex::new(Vec::new()),
        }
    }

    /// The plan for the step `body` applied to `states`, the values of the arrays `prev` that
    /// lead to the arrays of this `eval` (`None` if some of them are gone). If `prev` are the
    /// outputs of a plan that has not run and runs the same step, that plan gains one step;
    /// otherwise `states` are the initial values of a new one.
    fn following(prev: Option<&[Arc<Node>]>, states: &[StateRef], body: Fused, len: usize, shape: Shape) -> Option<Arc<LoopPlan>> {
        let count = states.len();
        if let Some(prev) = prev
            && let Some((plan, _)) = prev.first()?.looped.get()
            && plan.steps < u32::MAX
            && prev.iter().enumerate().all(|(j, node)| {
                matches!(node.looped.get(), Some((other, at)) if Arc::ptr_eq(other, plan) && *at == j as u32)
            })
            && plan.result.get().is_none()
            && let Some(parts) = lock(&plan.parts).as_ref()
            && same_program(&parts.body, &body, count)
        {
            return Some(Arc::new(LoopPlan::new(body, parts.inits.clone(), plan.steps + 1, len, shape)));
        }
        let inits = states
            .iter()
            .map(|state| match state {
                StateRef::Node(node) => node.as_src(),
                StateRef::Buffer(storage) => Ok(Src::Buffer(Arc::clone(storage))),
            })
            .collect::<Result<Vec<_>, _>>()
            .ok()?;
        Some(Arc::new(LoopPlan::new(body, inits, 1, len, shape)))
    }

    /// Makes `node` (output `j` of this plan) read what the plan reads, so that whatever is
    /// about to overwrite one of those buffers evaluates the node first.
    fn register(&self, node: &Arc<Node>) {
        let guard = lock(&self.parts);
        let Some(parts) = guard.as_ref() else { return };
        for storage in parts.body.inputs.iter().skip(parts.inits.len()) {
            storage.readers.add(node);
        }
        for init in &parts.inits {
            match init {
                Src::Buffer(storage) => storage.readers.add(node),
                Src::Node(init) => init.readers.add(node),
                Src::Scalar(_) => {}
            }
        }
    }

    /// Output `j`, running the plan if it has not run.
    pub(super) fn output(
        &self,
        ctx: &Context,
        j: u32,
        name: &'static str,
        location: &'static std::panic::Location<'static>,
    ) -> Result<Arc<Storage>, Failure> {
        let result = self.result.get_or_init(|| match self.execute(ctx) {
            Ok(states) => {
                *lock(&self.outputs) = states.into_iter().map(Some).collect();
                Ok(())
            }
            Err(Failure::Inherited(error)) => Err(error),
            Err(Failure::New(error)) => Err(Arc::new(LumaError::Op { op: name, location, source: Arc::new(error) })),
        });
        if let Err(error) = result {
            return Err(Failure::Inherited(Arc::clone(error)));
        }
        let taken = lock(&self.outputs).get_mut(j as usize).and_then(Option::take);
        taken.ok_or_else(|| Failure::New(LumaError::Gpu("loop output taken twice".into())))
    }

    /// Drops output `j`, for a loop leaf that went away without being evaluated.
    fn release(&self, j: u32) {
        if let Some(slot) = lock(&self.outputs).get_mut(j as usize) {
            *slot = None;
        }
    }

    fn execute(&self, ctx: &Context) -> Result<Vec<Arc<Storage>>, Failure> {
        let Some(Parts { body, inits }) = lock(&self.parts).take() else {
            return Err(Failure::New(LumaError::Gpu("loop plan ran without its inputs".into())));
        };
        let pending: Vec<Arc<Node>> = inits
            .iter()
            .filter_map(|init| match init {
                Src::Node(node) if node.is_pending() => Some(Arc::clone(node)),
                _ => None,
            })
            .collect();
        launch::evaluate_group(&pending);
        let mut states = Vec::with_capacity(inits.len());
        for init in &inits {
            states.push(match init {
                Src::Buffer(storage) => Arc::clone(storage),
                Src::Node(node) => match node.force() {
                    Ok(storage) => Arc::clone(storage),
                    Err(error) => return Err(Failure::Inherited(Arc::clone(error))),
                },
                Src::Scalar(_) => return Err(Failure::New(LumaError::Gpu("scalar as a loop state".into()))),
            });
        }

        let count = states.len();
        if !body.key.instrs.iter().any(|instr| matches!(instr.op.kind, OpKind::Shift(_))) {
            let state_of: Vec<Option<usize>> = (0..body.inputs.len()).map(|k| (k < count).then_some(k)).collect();
            if let Some(result) = launch::run_loop_fused(ctx, &body, &state_of, &states, self.steps) {
                return result;
            }
        } else if let [state] = states.as_slice()
            // Tiles load circular halos from one buffer.
            && state.segments() == 1
            && let Some(stencil) = stencil::plan_fused(body.clone(), Some(0), self.shape)
        {
            let out = stencil.run(&Array::from_storage(ctx, self.shape, Arc::clone(state)), self.steps as usize)?;
            return Ok(vec![Arc::clone(out.storage()?)]);
        }

        // Step by step; the states of later steps are ours alone, so a step may overwrite them.
        for _ in 0..self.steps {
            let mut inputs = body.inputs.clone();
            for (slot, state) in inputs.iter_mut().zip(states.drain(..)) {
                *slot = state;
            }
            let fused = Fused { key: body.key.clone(), inputs, scalars: body.scalars.clone(), fused_nodes: Vec::new(), instr_nodes: Vec::new(), sub_roots: Vec::new() };
            states = launch::run_store(ctx, self.len, fused)?.into_iter().map(Arc::new).collect();
        }
        Ok(states)
    }
}

impl Node {
    /// Makes this node output `j` of `plan`, releasing its operands (and with them the chain of
    /// steps below it). Nothing happens to a node that is being or was evaluated.
    fn redirect(self: &Arc<Self>, plan: &Arc<LoopPlan>, j: u32) {
        {
            let mut args = lock(&self.args);
            if args.is_none() || self.looped.set((Arc::clone(plan), j)).is_err() {
                return;
            }
            *args = None;
        }
        plan.register(self);
    }
}

impl Drop for Node {
    fn drop(&mut self) {
        if let Some((plan, j)) = self.looped.get() {
            plan.release(*j);
        }
    }
}

/// The last traced step. Its fixed input buffers are held weakly, so that a loop that ended does
/// not keep them alive; as long as a `Weak` exists the address of a buffer is not reused by
/// another one.
#[derive(Debug)]
struct Last {
    key: FusedKey,
    scalars: Vec<u32>,
    fixed: Vec<Weak<Storage>>,
}

impl Last {
    fn of(fused: &Fused, states: usize) -> Last {
        Last {
            key: fused.key.clone(),
            scalars: fused.scalars.clone(),
            fixed: fused.inputs.iter().skip(states).map(Arc::downgrade).collect(),
        }
    }

    /// Whether `fused` is the same step (see [`same_program`]).
    fn is_step_of(&self, fused: &Fused, states: usize) -> bool {
        self.key == fused.key
            && self.scalars == fused.scalars
            && self.fixed.len() + states == fused.inputs.len()
            && self.fixed.iter().zip(fused.inputs.iter().skip(states)).all(|(weak, input)| std::ptr::eq(weak.as_ptr(), Arc::as_ptr(input)))
    }
}

/// What [`JitState::step_boundary`] remembers of the previous `eval`.
#[derive(Debug, Default)]
pub(super) struct Tracker {
    /// The arrays of the previous `eval` with their node ids, and their shape.
    prev: Vec<(Weak<Node>, u64)>,
    prev_shape: Option<Shape>,
    /// The step that led to `prev`, if it could be traced.
    last: Option<Last>,
}

impl Tracker {
    /// Forgets the step and starts over from `nodes`.
    fn reset(&mut self, nodes: &[Arc<Node>], shape: Option<Shape>) {
        self.remember(nodes, shape);
        self.last = None;
    }

    fn remember(&mut self, nodes: &[Arc<Node>], shape: Option<Shape>) {
        self.prev = nodes.iter().map(|node| (Arc::downgrade(node), node.id)).collect();
        self.prev_shape = shape;
    }
}

/// The buffer computed by node `id` that the expressions of `roots` read, if any. A step whose
/// shifts cannot recompute the previous state evaluates it and drops its node, so the state is
/// only found through its buffer.
fn produced_by(roots: &[Arc<Node>], id: u64) -> Option<Arc<Storage>> {
    let mut stack: Vec<Arc<Node>> = roots.to_vec();
    let mut seen = super::program::Seen::default();
    while let Some(node) = stack.pop() {
        if seen.contains(Arc::as_ptr(&node)) {
            continue;
        }
        seen.push(Arc::as_ptr(&node));
        let Some(args) = node.operands() else { continue };
        for arg in args.iter().take(node.op().kind.arity()) {
            match arg {
                Src::Buffer(storage) if storage.producer() == id => return Some(Arc::clone(storage)),
                Src::Node(operand) if operand.is_pending() => stack.push(Arc::clone(operand)),
                _ => {}
            }
        }
    }
    None
}

impl JitState {
    /// Called by `eval_all` with the pending `nodes` it was asked for, all `f32` of one `shape`
    /// (no `nodes` if they are not). If they come from the previous call's arrays through the
    /// same step as the call before, they become loop leaves (see the module documentation).
    pub(crate) fn step_boundary(&self, nodes: &[Arc<Node>], shape: Shape) {
        let mut guard = lock(&self.tracker);
        let tracker = &mut *guard;
        if nodes.is_empty() {
            tracker.reset(nodes, None);
            return;
        }
        if tracker.prev_shape != Some(shape) || tracker.prev.len() != nodes.len() {
            tracker.reset(nodes, Some(shape));
            return;
        }
        let prev: Vec<(Option<Arc<Node>>, u64)> = tracker.prev.iter().map(|(weak, id)| (weak.upgrade(), *id)).collect();
        let mut states = Vec::with_capacity(prev.len());
        for (node, id) in &prev {
            let state = match node {
                Some(node) => match node.result.get() {
                    Some(Ok(storage)) => Some(StateRef::Buffer(Arc::clone(storage))),
                    Some(Err(_)) => {
                        tracker.reset(nodes, Some(shape));
                        return;
                    }
                    None => Some(StateRef::Node(Arc::clone(node))),
                },
                None => produced_by(nodes, *id).map(StateRef::Buffer),
            };
            let Some(state) = state else {
                // The array is gone and its buffer is not read: nothing to trace now, but the
                // step seen before still counts.
                tracker.remember(nodes, Some(shape));
                return;
            };
            states.push(state);
        }
        let Some(mut traced) = flatten_step(nodes, &states, Tail::Store) else {
            tracker.reset(nodes, Some(shape));
            return;
        };
        self.intern(&mut traced.fused.key);
        let last = Last::of(&traced.fused, states.len());
        let repeats = tracker.last.as_ref().is_some_and(|before| before.is_step_of(&traced.fused, states.len()));
        let alive: Option<Vec<Arc<Node>>> = prev.into_iter().map(|(node, _)| node).collect();
        let len = nodes[0].len();
        let instrs = traced.fused.key.instrs.len() as u64;
        let mut full = false;
        if repeats && let Some(plan) = LoopPlan::following(alive.as_deref(), &states, traced.fused, len, shape) {
            for (j, node) in nodes.iter().enumerate() {
                node.redirect(&plan, j as u32);
            }
            full = u64::from(plan.steps) * instrs >= PLAN_INSTRS;
        }
        tracker.remember(nodes, Some(shape));
        tracker.last = Some(last);
        drop(guard);
        if full {
            launch::evaluate_group(nodes);
            let _ = nodes[0].context().submit_now();
        }
    }
}
