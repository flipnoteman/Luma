//! Reductions forced together. A pending reduction is recorded when its value is first used or read
//! (see [`Reduced`]); programs compute many of them over the same data (the pressure that several
//! masked sums gather, the columns that a mean, a minimum and a maximum each traverse), and
//! recording each alone costs a launch and a pass over the inputs per reduction.
//!
//! Every reduction that can be recorded is registered when it is made. When one is recorded, the
//! others that are still pending, were made by the same thread, reduce the same number of elements
//! of the same type, and read what the group reads (a node or a buffer in common) or are small, are
//! taken along, as many as `Tuning::reduce_multi` accumulators and the device's bindings allow: one
//! kernel computes their operands together and every op keeps an accumulator (see
//! [`ReduceList::merged`](crate::builtin::ReduceList::merged)). Each reduction then holds a window
//! of the one buffer the kernel wrote. A reduction whose operand depends on another that is not
//! recorded yet (a variance on a mean) is not taken along: it waits for its own turn, when the
//! dependencies have run.
//!
//! Reductions along an axis (`Array::sum_axis` and the methods built on `reduce_axis`, `f32` on a GPU
//! backend outside a recording) are left pending the same way, one `Reduced` per op, and recorded
//! together when they reduce the same lines (`AxisGeom`): at most `scan::AXIS_MULTI` ops in one
//! kernel, whose first pass computes the distinct operands together (`jit::axis_plan`) and whose
//! ops each combine their line in the order a reduction alone does, so every result is the same
//! bits. A program that does not fit one kernel leaves them to record alone.

use std::collections::HashSet;
use std::sync::{Arc, Weak};

use super::{lock, overlap, union_into, IdSet, JitState, Node, Src, COMPUTING, FLATTENING};
use crate::reduce::Reduced;

/// Most nodes walked looking for the reductions an operand depends on.
const WALK_BUDGET: usize = 4096;

/// Most registered reductions looked at when one is recorded.
const EXAMINED: usize = 256;

/// Most nodes below the operands of one kernel.
const MAX_NODES: usize = 2048;

/// Operands of at most this many nodes are taken along without sharing anything.
const SMALL: usize = 16;

/// The reductions that the expression behind `root` is waiting for (leaves that stand for a reduction not
/// recorded yet), or `None` if it reads a node that cannot be told to run from here (another thread
/// evaluates it, or it is a loop's output) or is too big to walk.
pub(crate) fn unresolved_leaves(root: &Arc<Node>) -> Option<Vec<Arc<Node>>> {
    let mut seen: HashSet<*const Node> = HashSet::new();
    let mut stack = vec![Arc::clone(root)];
    let mut leaves = Vec::new();
    while let Some(node) = stack.pop() {
        if node.result.get().is_some() || !seen.insert(Arc::as_ptr(&node)) {
            continue;
        }
        if seen.len() > WALK_BUDGET {
            return None;
        }
        match node.operands() {
            Some(args) => {
                for arg in args.iter().take(node.op.kind.arity()) {
                    if let Src::Node(operand) = arg {
                        stack.push(Arc::clone(operand));
                    }
                }
            }
            None if node.deferred.get().is_some() => leaves.push(node),
            None => return None,
        }
    }
    Some(leaves)
}

impl JitState {
    /// Notes the reduction `reduced`, which is not recorded yet, as one that may be recorded together with another.
    pub(crate) fn register_reduction(&self, reduced: &Arc<Reduced>) {
        let mut list = lock(&self.reductions);
        // Dead or recorded ones are dropped as the list doubles.
        if list.len() >= 32 && list.len().is_power_of_two() {
            list.retain(|weak| weak.upgrade().is_some_and(|reduced| reduced.is_unforced()));
        }
        list.push(Arc::downgrade(reduced));
    }

    /// Whether some registered reduction could be recorded together with `seed`: what is cheap to tell, before the
    /// expressions are walked (see [`JitState::reduction_partners`]).
    pub(crate) fn may_have_partners(&self, seed: &Reduced) -> bool {
        lock(&self.reductions).iter().filter_map(Weak::upgrade).any(|candidate| !std::ptr::eq(Arc::as_ptr(&candidate), seed) && candidate.partner_of(seed))
    }

    /// The registered reductions to record together with `seed` (see the module documentation), oldest first.
    /// Nothing for a `seed` that cannot be merged.
    pub(crate) fn reduction_partners(&self, ctx: &crate::context::Context, seed: &Reduced) -> Vec<Arc<Reduced>> {
        let axis = seed.is_axis();
        let cap = if axis { ctx.tuning().reduce_multi.min(crate::scan::AXIS_MULTI) } else { ctx.tuning().reduce_multi };
        let mut accs = seed.ops().as_slice().len();
        if accs >= cap || !(if axis { seed.axis_mergeable(ctx) } else { seed.mergeable(ctx) }) {
            return Vec::new();
        }
        let list: Vec<Arc<Reduced>> = {
            let mut list = lock(&self.reductions);
            list.retain(|weak: &Weak<Reduced>| weak.strong_count() > 0);
            list.iter().filter_map(Weak::upgrade).collect()
        };
        let node = seed.operand();
        let max_inputs = self.max_inputs();
        let mut inputs = self.inputs_within(node, &[], false, max_inputs);
        let mut below = IdSet::of(node, true);
        let mut tmp = Vec::new();
        let mut partners = Vec::new();
        // A node that reads one this thread is computing or flattening right now cannot be computed before it (see
        // `JitState::evaluate_alongside`).
        let flying = FLATTENING.with_borrow(Clone::clone);
        let computing = COMPUTING.with_borrow(Clone::clone);
        for candidate in list.iter().take(EXAMINED) {
            if accs >= cap {
                break;
            }
            let ops = candidate.ops().as_slice().len();
            if std::ptr::eq(Arc::as_ptr(candidate), seed) || accs + ops > cap || !candidate.partner_of(seed) {
                continue;
            }
            let operand = candidate.operand();
            if flying.contains(&Arc::as_ptr(operand)) || computing.iter().any(|&id| operand.below_contains(id) || operand.id == id) {
                continue;
            }
            if !unresolved_leaves(operand).is_some_and(|leaves| leaves.is_empty()) {
                continue;
            }
            let (shared, new_below) = if below.contains(operand.id) { (1, 0) } else { below.overlap(operand) };
            let operand_inputs = self.inputs_within(operand, &inputs, false, max_inputs);
            let (shared_inputs, new_inputs) = overlap(&operand_inputs, &inputs);
            let related = shared > 0 || shared_inputs > 0 || operand.size() <= SMALL;
            // The kernel binds the partial results, the final values, the counter and the uniform besides (an axis reduction a buffer of
            // results for every op).
            let bound = if axis { accs + ops + 2 } else { 4 };
            if !related || inputs.len() + new_inputs + bound > max_inputs || below.len() + new_below >= MAX_NODES {
                continue;
            }
            union_into(&mut inputs, &mut tmp, &operand_inputs);
            below.add(operand, true);
            accs += ops;
            partners.push(Arc::clone(candidate));
        }
        partners
    }
}
