//! Running flattened programs: evaluation, in-place and shifted outputs, register loops and
//! the first pass of reductions.

use std::borrow::Cow;
use std::sync::atomic::Ordering;
use std::sync::Arc;

use luma_core::Buffer;
use crate::array::{Failure, Storage};
use crate::gpu_type::ElemLayout;
use crate::gpu::pager::Pin;
use crate::builtin::{Builtin, Finish, GateId, ReduceList, ReduceOp};
use crate::context::{RawArg, Scratch};
use crate::error::LumaError;
use crate::matmul::{MatmulOperand, MatmulTile};
use crate::solve::SolveSpec;
use crate::sort::RadixPass;
use crate::ops::{fold_workgroups, launch_1d, len_u32};

use super::codegen::direct;
use super::{lock, unresolved_leaves, Args, Edge, ElemTy, Node, OpKind, ShiftKind, Src, ViewMap};
use crate::profile::{Decision, DecisionSite, SplitReason};
use std::panic::Location;
use super::program::{flatten, flatten_inner, Fused, FusedKey, Ref, Tail};

/// Runs `steps` iterations of a traced step in one kernel that keeps the state in registers.
/// `outs[j]` is the next value of state `j`, computed from `probes[j]` (aliases of `inits[j]`).
/// `None` if the step is not a single-buffer elementwise program of `f32` (or `u32`) values; the caller
/// then runs the steps one by one.
pub(crate) fn run_loop(
    ctx: &crate::context::Context,
    probes: &[Arc<Storage>],
    inits: &[Arc<Storage>],
    outs: &[Arc<Node>],
    steps: u32,
) -> Option<Result<Vec<Arc<Storage>>, Failure>> {
    let (fused, states) = prepare_loop(probes, inits, outs)?;
    run_loop_fused(ctx, &fused, &states, inits, steps)
}

/// Flattens the traced step with next states `outs`, computed from `probes`, into a store program
/// with one output per state; `states[k]` is the state that input `k` reads (`None`: an input
/// that stays fixed). `None` if the step is not a single-buffer elementwise program.
pub(super) fn prepare_loop(
    probes: &[Arc<Storage>],
    inits: &[Arc<Storage>],
    outs: &[Arc<Node>],
) -> Option<(Fused, Vec<Option<usize>>)> {
    let len = outs.first()?.len();
    if outs.iter().any(|out| out.len() != len || !out.is_pending())
        || outs.iter().enumerate().any(|(j, out)| outs[..j].iter().any(|other| Arc::ptr_eq(other, out)))
        || probes.iter().any(|probe| probe.was_materialized())
        || probes.iter().chain(inits).any(|storage| storage.segments() != 1)
        || inits.iter().any(|init| init.layout() != ElemLayout::plain(4) || init.len() != len)
    {
        return None;
    }
    let roots: Vec<(Arc<Node>, Args)> =
        outs.iter().map(|out| out.operands().map(|args| (Arc::clone(out), args))).collect::<Option<_>>()?;
    let ids: Vec<u64> = probes.iter().map(|probe| probe.id()).collect();
    let fused = flatten_inner(&roots, Tail::Store, &ids).ok()?;
    drop(roots);
    if probes.iter().any(|probe| probe.was_materialized()) {
        return None;
    }
    let states: Vec<Option<usize>> =
        fused.inputs.iter().map(|input| probes.iter().position(|probe| Arc::ptr_eq(probe, input))).collect();
    Some((fused, states))
}

/// The weight of the store program `key` per element and step (see [`super::cost::op_weight`]) if
/// it can run as a register loop over `states` (`states[k]`: the state input `k` reads): a
/// single-buffer elementwise program of 4-byte values whose states are all `f32` or all `u32` and
/// that binds no more buffers than the device allows. `None` otherwise.
pub(super) fn loop_ops(ctx: &crate::context::Context, key: &FusedKey, states: &[Option<usize>]) -> Option<f64> {
    let shifts = key.instrs.iter().any(|instr| matches!(instr.op.kind, OpKind::Shift(_)));
    // The states are all `f32` or all `u32`, and what a state input reads is of that type.
    let state_ty = key.outputs.first().and_then(|&at| key.instrs.get(at as usize)).map(|i| i.op.out_ty());
    let typed = matches!(state_ty, Some(ElemTy::F32 | ElemTy::U32))
        && key.outputs.iter().all(|&at| key.instrs.get(at as usize).is_some_and(|i| Some(i.op.out_ty()) == state_ty))
        && states.iter().zip(&key.input_tys).all(|(state, ty)| state.is_none() || Some(*ty) == state_ty);
    if shifts || !key.subs.is_empty() || !typed || !key.plain_words() {
        return None;
    }
    let bindings = 2 * key.outputs.len() + states.iter().filter(|state| state.is_none()).count();
    if bindings > ctx.jit().max_inputs() + 1 {
        return None;
    }
    let cost = ctx.cost_model();
    Some(key.instrs.iter().map(|instr| super::cost::op_weight(&instr.op, &cost)).sum())
}

/// Runs `steps` iterations of the store program `fused` (one output per state, `outputs[j]` the
/// next value of state `j`) in one kernel that keeps the states in registers. `states[k]` is the
/// state that input `k` reads (`None`: an input that stays fixed) and `inits` the states before
/// the first step. `None` if the program is not a single-buffer elementwise program of `f32`
/// (or all `u32`) values.
pub(super) fn run_loop_fused(
    ctx: &crate::context::Context,
    fused: &Fused,
    states: &[Option<usize>],
    inits: &[Arc<Storage>],
    steps: u32,
) -> Option<Result<Vec<Arc<Storage>>, Failure>> {
    let ops = loop_ops(ctx, &fused.key, states)?;
    if fused.inputs.iter().any(|input| input.segments() != 1) {
        return None;
    }
    let fixed: Vec<Arc<Storage>> =
        fused.inputs.iter().zip(states).filter(|(_, state)| state.is_none()).map(|(input, _)| Arc::clone(input)).collect();
    Some(run_loop_program(ctx, &fused.key, &fused.scalars, states, &fixed, inits, steps, ops))
}

/// Runs the register loop `key` (checked by [`loop_ops`], which returned `ops`) for `steps`
/// steps over `inits`, with the inputs that stay fixed (`states[k] == None`, in input order) bound
/// to `fixed`. The steps are cut into launches by [`steps_per_launch`].
#[allow(clippy::too_many_arguments)]
pub(super) fn run_loop_program(
    ctx: &crate::context::Context,
    key: &FusedKey,
    scalars: &[u32],
    states: &[Option<usize>],
    fixed: &[Arc<Storage>],
    inits: &[Arc<Storage>],
    steps: u32,
    ops: f64,
) -> Result<Vec<Arc<Storage>>, Failure> {
    let len = inits.first().map_or(0, |init| init.len());
    let per_launch = steps_per_launch(ctx, len, ops, key.instrs.len());
    let core = ctx.jit().loop_kernel(ctx, key, states)?;
    let mut states_now: Vec<Arc<Storage>> = inits.to_vec();
    let mut remaining = steps;
    while remaining > 0 {
        let count = remaining.min(per_launch);
        let outputs = launch_loop(ctx, &core, scalars, fixed, &states_now, len, count)?;
        states_now = outputs.into_iter().map(Arc::new).collect();
        remaining -= count;
    }
    Ok(states_now)
}

/// Time one launch of a register loop is sized to take, by the device's [`CostModel`]. Long enough
/// that the launches and the states' round trips through memory vanish (a 1M-instance Keccak
/// permutation is half a millisecond, so one launch runs a few dozen of them), short enough that a
/// device several times slower than the model says stays far below the 2 s watchdog of Windows.
const LAUNCH_TARGET_NS: f64 = 40e6;

/// Nanoseconds one invocation takes per operation when nothing runs beside it: the dependent
/// chain of a step, which bounds the time of a loop over a few elements however little work it is.
const SERIAL_NS_PER_OP: f64 = 2.0;

/// How many steps of a register loop (`ops` per element and step by the cost model, `instrs`
/// instructions) one launch of `len` elements runs: as many as take about [`LAUNCH_TARGET_NS`],
/// at least one. A test that sets the work per launch (`JitState::set_loop_work`, in
/// element-instructions) gets that.
fn steps_per_launch(ctx: &crate::context::Context, len: usize, ops: f64, instrs: usize) -> u32 {
    let work = ctx.jit().loop_work.load(Ordering::Relaxed);
    if work > 0 {
        return u32::try_from(work / (len as u64 * instrs.max(1) as u64).max(1)).unwrap_or(u32::MAX).max(1);
    }
    let cost = ctx.cost_model();
    let ops = ops.max(1.0);
    let step_ns = (ops * len as f64 / cost.f32_ops_per_ns).max(ops * SERIAL_NS_PER_OP);
    (LAUNCH_TARGET_NS / step_ns).clamp(1.0, f64::from(u32::MAX)) as u32
}

fn launch_loop(
    ctx: &crate::context::Context,
    core: &crate::kernel::KernelCore,
    scalars: &[u32],
    fixed: &[Arc<Storage>],
    inits: &[Arc<Storage>],
    len: usize,
    steps: u32,
) -> Result<Vec<Storage>, Failure> {
    let n = len_u32(len)?;
    let bytes = u64::from(n) * 4;
    let outputs: Vec<Storage> = inits.iter().map(|_| Storage::new(ctx, len, ElemLayout::plain(4))).collect();
    let out_pins = outputs.iter().map(|out| out.pin(0)).collect::<Result<Vec<_>, _>>()?;
    let init_pins = inits.iter().map(|init| init.pin(0)).collect::<Result<Vec<_>, _>>()?;
    let in_pins = fixed.iter().map(|input| input.pin_read(0).map(|pin| (pin, input.byte_len()))).collect::<Result<Vec<_>, _>>()?;
    let mut words = Vec::with_capacity(scalars.len() + 1);
    words.push(steps);
    words.extend_from_slice(scalars);
    let mut raw: Vec<RawArg<'_>> = Vec::with_capacity(out_pins.len() * 2 + in_pins.len() + 1);
    raw.extend(out_pins.iter().map(|pin| RawArg::write(pin.buffer(), bytes)));
    raw.extend(init_pins.iter().map(|pin| RawArg::read(pin.buffer(), bytes)));
    raw.extend(in_pins.iter().map(|(pin, bytes)| RawArg::read(pin.buffer(), *bytes)));
    raw.push(RawArg::Uniform(bytemuck::cast_slice(&words)));
    launch_1d(ctx, core, n, &raw)?;
    ctx.jit().stats.elementwise_launches.fetch_add(1, Ordering::Relaxed);
    Ok(outputs)
}

/// Evaluates `node`: one fused kernel computing its whole expression into a new buffer.
pub(crate) fn evaluate(node: &Arc<Node>, args: Args) -> Result<Storage, Failure> {
    let (fused, args) = flatten_bounded(node, args, Tail::Store)?;
    drop(args);
    let mut outputs = run_store(node.context(), node.len(), fused)?;
    outputs.pop().ok_or_else(|| Failure::New(LumaError::Gpu("fused kernel without output".into())))
}

/// Rounds of splitting one program on register pressure (see [`live_split`]) before it runs as is.
const LIVE_ROUNDS: usize = 4;

/// Flattens the expression of `node` with operands `args`, which come back for the caller to drop.
/// Nodes inside it that were computed after it was measured are buffers now, so it can read more
/// buffers than the device binds; then its operands are computed first, one at a time. A program
/// that keeps more values live at once than the device's tuning allows is cut in the middle
/// (see [`live_split`]).
fn flatten_bounded(node: &Arc<Node>, mut args: Args, tail: Tail) -> Result<(Fused, Args), Failure> {
    let jit = node.context().jit();
    let max_inputs = jit.max_inputs();
    let max_live = jit.max_live(node.context().tuning());
    let mut rounds = 0;
    let mut repacked = false;
    loop {
        let roots = [(Arc::clone(node), args)];
        let mut fused = flatten(&roots, tail)?;
        let [(_, returned)] = roots;
        args = returned;
        // Buffers that a pack holds are read through one binding (see `pack`).
        if fused.inputs.len() > max_inputs && !repacked {
            repacked = true;
            if super::pack::repack_inputs(node.context(), &fused, fused.inputs.len() - max_inputs) {
                continue;
            }
        }
        if fused.inputs.len() > max_inputs
            && let Some(i) = (0..node.op().kind.arity()).find(|&i| matches!(args.get(i), Some(Src::Node(_))))
        {
            jit.note_cut(node.name, node.location, SplitReason::Inputs { inputs: fused.inputs.len(), limit: max_inputs });
            drop(fused);
            super::evaluate_operand(&mut args, i)?;
            continue;
        }
        // The scalars are fields of one uniform buffer. (The node limit lets an expression of
        // `f64` scalars, three words each, outgrow it.)
        let max_words = usize::try_from(node.context().max_uniform_bytes() / 4).unwrap_or(usize::MAX).saturating_sub(64);
        if fused.scalars.len() > max_words
            && let Some(i) = (0..node.op().kind.arity())
                .filter_map(|i| match args.get(i) {
                    Some(Src::Node(operand)) => Some((i, operand.size())),
                    _ => None,
                })
                .max_by_key(|&(_, size)| size)
                .map(|(i, _)| i)
        {
            jit.note_cut(node.name, node.location, SplitReason::Scalars { words: fused.scalars.len(), limit: max_words });
            drop(fused);
            super::evaluate_operand(&mut args, i)?;
            continue;
        }
        if rounds < LIVE_ROUNDS
            && let Some((outs, live)) = live_split(&fused, max_inputs, max_live)
        {
            rounds += 1;
            jit.note_cut(node.name, node.location, SplitReason::Live { live, limit: max_live });
            drop(fused);
            evaluate_group(&outs);
            continue;
        }
        // The nodes hold their operands, which a dying input must not be held by (see `run_store`).
        fused.instr_nodes = Vec::new();
        fused.sub_roots = Vec::new();
        return Ok((fused, args));
    }
}

/// Per instruction of the main program of `key`, the index of the last instruction reading its
/// value (its own index if none does, `usize::MAX` for an output).
pub(super) fn last_uses(key: &FusedKey) -> Vec<usize> {
    let mut last: Vec<usize> = (0..key.instrs.len()).collect();
    for (i, instr) in key.instrs.iter().enumerate() {
        for arg in &instr.args {
            if let Ref::Tmp(t) = arg
                && let Some(slot) = last.get_mut(*t as usize)
            {
                *slot = i;
            }
        }
    }
    for &out in &key.outputs {
        if let Some(slot) = last.get_mut(out as usize) {
            *slot = usize::MAX;
        }
    }
    last
}

/// For every instruction of the main program of `key`, the values live while it runs: those
/// computed so far that a later instruction (or an output) still reads, plus the inputs loaded
/// up front.
fn live_split(fused: &Fused, max_inputs: usize, max_live: usize) -> Option<(Vec<Arc<Node>>, usize)> {
    let key = &fused.key;
    let n = key.instrs.len();
    if n + key.input_tys.len() <= max_live || fused.instr_nodes.len() != n {
        return None;
    }
    let last = last_uses(key);
    let loaded = (0..key.input_tys.len() as u32).filter(|&k| direct(&key.instrs, k)).count() as u32;
    let mut dying = vec![0u32; n];
    let mut live = Vec::with_capacity(n);
    let mut now = 0u32;
    let mut peak = 0u32;
    for k in 0..n {
        now -= dying[k];
        if last[k] > k {
            now += 1;
            if let Some(slot) = dying.get_mut(last[k]) {
                *slot += 1;
            }
        }
        let here = now + loaded;
        peak = peak.max(here);
        live.push(here);
    }
    let peak = peak as usize;
    if peak <= max_live {
        return None;
    }
    let (lo, hi) = (n / 4, n * 3 / 4);
    let middle = n / 2;
    let cut = (lo..hi).min_by_key(|&k| (live[k], k.abs_diff(middle)))?;
    if live[cut] as usize >= peak {
        return None;
    }
    // Values computed up to the cut that something after it reads.
    let mut outs = Vec::new();
    for (j, &used) in last.iter().enumerate().take(cut + 1) {
        if used > cut {
            outs.push(Arc::clone(fused.instr_nodes[j].as_ref()?));
        }
    }
    (!outs.is_empty() && outs.len() <= max_inputs / 2).then_some((outs, peak))
}

/// Evaluates the pending ones of `nodes` (all of the same length) in one fused kernel with one
/// output per node. Failures are stored in the nodes, as [`Node::force`] does.
pub(crate) fn evaluate_group(nodes: &[Arc<Node>]) {
    if crate::host::try_eval_group(nodes) {
        return;
    }
    // A loop leaf has no operands to fuse: running its plan is its evaluation.
    for node in nodes.iter().filter(|node| node.is_pending() && (node.looped.get().is_some() || node.deferred.get().is_some())) {
        let _ = node.force();
    }
    let roots: Vec<(Arc<Node>, Args)> = nodes
        .iter()
        .filter(|node| node.is_pending())
        .filter_map(|node| node.operands().map(|args| (Arc::clone(node), args)))
        .collect();
    // A root that another root reads through views would be recomputed at every index it is read
    // at, and the chain of such steps grows past what a kernel takes: it goes first, alone.
    // (A root read at the element itself is computed once, as an output beside its reader.)
    let reads_a_root = reads_roots_through_views(&roots);
    if reads_a_root.contains(&true) {
        let (mut later, mut first) = (Vec::new(), Vec::new());
        for ((node, _), &reads) in roots.iter().zip(&reads_a_root) {
            if reads { later.push(Arc::clone(node)) } else { first.push(Arc::clone(node)) }
        }
        evaluate_group(&first);
        evaluate_group(&later);
        return;
    }
    let grouped = roots.len() > 1
        && roots.iter().all(|(node, _)| node.len() == roots[0].0.len())
        && match flatten_packed(&roots) {
            // Outputs that would make the kernel bind too many buffers share one (see `run_packed`).
            Ok(fused) if fused.inputs.len() + roots.len() > roots[0].0.context().jit().max_inputs() + 1 => run_packed(&roots, fused),
            Ok(fused) => {
                let (first, _) = &roots[0];
                match run_store(first.context(), first.len(), fused) {
                    Ok(outputs) => {
                        for ((node, _), storage) in roots.iter().zip(outputs) {
                            match check_finite(first.context(), &storage, node.op().out_ty()) {
                                Ok(()) => node.finish(storage),
                                Err(error) => node.fail(error),
                            }
                        }
                        true
                    }
                    Err(_) => false,
                }
            }
            Err(_) => false,
        };
    if !grouped {
        // One kernel each; this also reports every failure at its own node.
        for (node, _) in &roots {
            let _ = node.force();
        }
    }
}

/// [`flatten`] of `roots` as a store, after packing the buffers it reads if it binds more than a
/// kernel does (see `pack::repack_inputs`).
fn flatten_packed(roots: &[(Arc<Node>, Args)]) -> Result<Fused, Failure> {
    let fused = flatten(roots, Tail::Store)?;
    let ctx = roots[0].0.context();
    let max_inputs = ctx.jit().max_inputs();
    if fused.inputs.len() > max_inputs && super::pack::repack_inputs(ctx, &fused, fused.inputs.len() - max_inputs) {
        drop(fused);
        return flatten(roots, Tail::Store);
    }
    Ok(fused)
}

/// Whether `count` arrays of `len` elements of `ty` fit one pack (see [`run_packed`]): one buffer of
/// elements that sit one after the other.
pub(super) fn packable(ctx: &crate::context::Context, count: usize, len: usize, ty: ElemTy) -> bool {
    !ty.is_packed()
        && count > 1
        && len > 0
        && count.checked_mul(len).is_some_and(|total| u32::try_from(total).is_ok() && (total as u64).saturating_mul(ty.stride()) <= ctx.segment_bytes())
}

/// Runs the multi-output program `fused` of `roots` as one kernel that writes every output into
/// one new buffer, the pack, and makes the roots windows of it (see [`Storage::window_of`]): the
/// kernel binds one buffer for its outputs instead of one each, and so do the kernels that read
/// them. For a group whose outputs alone would exceed what a kernel binds. `false` if the
/// program cannot run like that (the outputs differ in type, or it reads an array split over
/// several buffers, or its inputs leave no binding for the pack, or a step is being recorded),
/// and nothing was evaluated.
fn run_packed(roots: &[(Arc<Node>, Args)], mut fused: Fused) -> bool {
    let (first, _) = &roots[0];
    let ctx = first.context();
    let (len, ty) = (first.len(), first.op().out_ty());
    // A recorded step (see `Trace`) is replayed with its results as buffers that its launches write:
    // windows of a pack, which a copy would later detach from it, are not.
    if ctx.recording_here() {
        return false;
    }
    if roots.iter().any(|(node, _)| node.op().out_ty() != ty)
        || !packable(ctx, roots.len(), len, ty)
        || fused.inputs.iter().any(|input| input.segments() != 1)
        || fused.inputs.len() + 1 > ctx.jit().max_inputs() + 1
    {
        return false;
    }
    let (Ok(total), Ok(block)) = (u32::try_from(roots.len()), u32::try_from(len)) else { return false };
    fused.key.tail = Tail::Stack { total, first: 0, block };
    let pack = Arc::new(Storage::new(ctx, roots.len() * len, ty.layout()));
    if run_mapped(ctx, fused, std::slice::from_ref(&pack), len).is_err() {
        return false;
    }
    let finite = check_finite(ctx, &pack, ty);
    for (j, (node, _)) in roots.iter().enumerate() {
        match &finite {
            Ok(()) => node.finish_shared(Arc::new(Storage::window_of(&pack, j * len, len))),
            Err(error) => node.fail(error.clone()),
        }
    }
    true
}

/// For every root of `roots`, whether its expression reads another of the roots through a shift,
/// view or gather (as the operand being shifted, not as the indices of a gather).
fn reads_roots_through_views(roots: &[(Arc<Node>, Args)]) -> Vec<bool> {
    roots
        .iter()
        .map(|(root, args)| {
            let others: Vec<&Arc<Node>> =
                roots.iter().map(|(other, _)| other).filter(|other| !Arc::ptr_eq(root, other) && root.below_contains(other.id)).collect();
            root.shift_depth() > 0 && !others.is_empty() && reaches_through_view(root, args, &others)
        })
        .collect()
}

/// Whether the expression of `root` (with operands `args`) reaches one of `targets` through a
/// shift, view or gather. Only the paths that lead to a target are followed.
fn reaches_through_view(root: &Arc<Node>, args: &Args, targets: &[&Arc<Node>]) -> bool {
    let leads_to_target = |node: &Arc<Node>| targets.iter().any(|target| Arc::ptr_eq(target, node) || node.below_contains(target.id));
    let mut seen: std::collections::HashSet<(u64, bool)> = std::collections::HashSet::new();
    let mut stack = vec![(Arc::clone(root), false)];
    while let Some((node, shifted)) = stack.pop() {
        let Some(operands) = (if Arc::ptr_eq(&node, root) { Some(args.clone()) } else { node.operands() }) else { continue };
        let kind = node.op().kind;
        for (i, src) in operands.iter().take(kind.arity()).enumerate() {
            let Src::Node(kid) = src else { continue };
            if !kid.is_pending() {
                continue;
            }
            let shifted = shifted || (i == 0 && matches!(kind, OpKind::Shift(_)));
            if shifted && targets.iter().any(|target| Arc::ptr_eq(target, kid)) {
                return true;
            }
            if leads_to_target(kid) && seen.insert((kid.id, shifted)) {
                stack.push((Arc::clone(kid), shifted));
            }
        }
    }
    false
}

/// Evaluates the pending ones of `nodes` like [`evaluate_group`], those of one length in a
/// multi-output kernel, and returns the first failure among all of them (as inherited from the
/// node that failed).
pub(crate) fn force_group(nodes: &[Arc<Node>]) -> Result<(), Failure> {
    let mut pending: Vec<Arc<Node>> = nodes.iter().filter(|node| node.is_pending()).cloned().collect();
    while let Some(first) = pending.first() {
        let len = first.len();
        let (same, rest): (Vec<_>, Vec<_>) = pending.into_iter().partition(|node| node.len() == len);
        evaluate_group(&same);
        pending = rest;
    }
    // A node another thread was evaluating is waited for here.
    for node in nodes {
        if let Err(error) = node.force() {
            return Err(Failure::Inherited(Arc::clone(error)));
        }
    }
    Ok(())
}

/// Whether input `k` of a store program can be the buffer of its output: every element is read
/// before it is overwritten, by the invocation that overwrites it. That fails when the program
/// has several outputs or types differ, and when `k` is read at another element: through a shift
/// or inside a sub-program.
fn in_place_eligible(key: &FusedKey, k: u32) -> bool {
    let Some(&input_ty) = key.input_tys.get(k as usize) else { return false };
    let out_ty = key.outputs.first().and_then(|&at| key.instrs.get(at as usize)).map(|instr| instr.op.out_ty());
    key.tail == Tail::Store
        && key.outputs.len() == 1
        && out_ty == Some(input_ty)
        && !key.instrs.iter().any(|instr| {
            matches!(instr.op.kind, OpKind::Shift(_)) && instr.args[0] == Ref::Input(k)
        })
        && !key.subs.iter().flatten().any(|instr| instr.args.contains(&Ref::Input(k)))
}

/// How a program with shifts runs on arrays spanning several segments (see [`spans`]).
pub(super) struct Shifted {
    /// Elements of the arrays: the shifts wrap around this length.
    len: usize,
    /// Per output segment and input, the ids of the segments bound as slots.
    ids: Vec<Vec<Vec<u32>>>,
}

/// The input `k` of `key` among `inputs`, which lack the in-place inputs of `key` (those are
/// never read through a shift of a split array, see [`in_place_eligible`] and [`run_mapped`]).
pub(super) fn input_at<'a>(key: &FusedKey, inputs: &'a [Arc<Storage>], k: usize) -> Option<&'a Arc<Storage>> {
    if key.in_place.iter().any(|&(paired, _)| paired as usize == k) {
        return None;
    }
    let before = key.in_place.iter().filter(|&&(paired, _)| (paired as usize) < k).count();
    inputs.get(k - before)
}

/// Decides whether `key` needs the segmented variant on arrays of `len` elements and, if it does,
/// sets `key.spans` and returns the segments to bind. `None` for arrays that fit one segment
/// (and read no split array through a shift) and programs without shifts: they run as plain
/// per-segment launches. `inputs` lack the in-place input of `key`.
fn plan_shifts(
    ctx: &crate::context::Context,
    key: &mut FusedKey,
    inputs: &[Arc<Storage>],
    scalars: &[u32],
    len: usize,
) -> Result<Option<Shifted>, LumaError> {
    let layout = layout_elem(key, inputs, len);
    let c = (ctx.segment_bytes() / layout) as usize;
    // Arrays of different element sizes do not share a segmentation: they only run unsplit.
    let outs = if matches!(key.tail, Tail::Reduce(_)) { Vec::new() } else { out_elems(key) };
    let widest = inputs.iter().map(|input| input.layout().stride() as u64).chain(outs.iter().copied()).max().map_or(layout, |w| w.max(layout));
    let mixed = inputs.iter().any(|input| input.layout().stride() as u64 != layout) || outs.iter().any(|&bytes| bytes != layout);
    if mixed && (len as u64 * widest > ctx.segment_bytes() || inputs.iter().any(|input| input.segments() > 1)) {
        return Err(LumaError::Chunked("mixed element sizes on arrays split over several buffers"));
    }
    // Only an input in several segments splits a program (checked first: the scan reads every instruction).
    let split = inputs.iter().any(|input| input.segments() > 1)
        && key.instrs.iter().chain(key.subs.iter().flatten()).any(|instr| {
            matches!(instr.op.kind, OpKind::Shift(_))
                && matches!(instr.args[0], Ref::Input(k) if input_at(key, inputs, k as usize).is_some_and(|input| input.segments() > 1))
        });
    if len <= c && !split {
        return Ok(None);
    }
    if !key.subs.is_empty() {
        return Err(LumaError::Chunked("a lazy expression read through a view or shift on arrays split over several buffers"));
    }
    let Some((counts, ids)) = spans(key, inputs, scalars, len, c, ctx.jit().max_inputs() + 1)? else {
        return Ok(None);
    };
    key.spans = counts;
    Ok(Some(Shifted { len, ids }))
}

/// `ElemLayout::stride` of every output of `key`: one per stored value, or the one value a
/// reduction folds (a `bool` is a word of partial results there, as it is for the segmentation).
fn out_elems(key: &FusedKey) -> Vec<u64> {
    let bytes = |at: u32| {
        key.instrs
            .get(at as usize)
            .map_or(4, |instr| instr.op.out_ty().stride())
    };
    match key.tail {
        Tail::Store
        | Tail::StoreMapped(_)
        | Tail::ParamStoreMapped { .. }
        | Tail::Matmul { .. }
        | Tail::Radix(_)
        | Tail::Solve(_) => key.outputs.iter().map(|&at| bytes(at)).collect(),
        Tail::Scatter { .. }
        | Tail::ParamScatter { .. }
        | Tail::ScatterAdd { .. }
        | Tail::Stack { .. }
        | Tail::ParamStack { .. } => key.outputs.iter().take(1).map(|&at| bytes(at)).collect(),
        Tail::Reduce(_) | Tail::AxisReduce(_) => vec![
            key.instrs
                .len()
                .checked_sub(1)
                .map_or(4, |at| bytes(at as u32)),
        ],
    }
}

/// Bytes per element of the arrays whose segmentation the launch follows: the outputs of a
/// store, the inputs of the length `len` of a reduction (a reduction writes only partial
/// results, so what it folds does not matter).
fn layout_elem(key: &FusedKey, inputs: &[Arc<Storage>], len: usize) -> u64 {
    let outs = out_elems(key);
    match key.tail {
        Tail::Reduce(_) => inputs.iter().find(|input| input.len() == len).map_or_else(|| outs.first().copied().unwrap_or(4), |input| input.layout().stride() as u64),
        _ => outs.first().copied().unwrap_or(4),
    }
}

/// Slot counts per input and, per output segment and input, the segment ids to bind (see [`spans`]).
type SegmentSpans = (Vec<u8>, Vec<Vec<Vec<u32>>>);

/// For a program with shifts on arrays of `len` elements in segments of `c`: how many segments
/// of every input its shifted reads touch at most (slots, 0 = the input is never shifted) and,
/// per output segment, the ids of the segments to bind in those slots (`ids[j][k]`, padded to
/// the slot count by repeating the last id). `None` if the program has no shift. Fails if the
/// binding count exceeds `max_bindings`. `inputs` lack the in-place input of `key`.
fn spans(
    key: &FusedKey,
    inputs: &[Arc<Storage>],
    scalars: &[u32],
    len: usize,
    c: usize,
    max_bindings: usize,
) -> Result<Option<SegmentSpans>, LumaError> {
    // Per shift instruction: its input and what it reads: the translations (read index -
    // written index) of a circular shift, over the wrapped and unwrapped parts of every shifted
    // axis, or a view.
    let mut shifts: Vec<(usize, Reach)> = Vec::new();
    for instr in key.instrs.iter() {
        let OpKind::Shift(kind) = instr.op.kind else { continue };
        let Ref::Input(k) = instr.args[0] else {
            return Err(LumaError::Gpu("segmented program with sub-programs".into()));
        };
        let reach = match kind {
            ShiftKind::View { map, edge } => Reach::View(map, edge),
            ShiftKind::Take { .. } | ShiftKind::ParamTake { .. } | ShiftKind::ParamView { .. } | ShiftKind::ParamDims { .. } => Reach::All,
            ShiftKind::Linear => {
                let amount = match instr.args[1] {
                    Ref::Scalar(q) => scalars.get(q as usize).copied(),
                    _ => None,
                };
                let Some(amount) = amount else {
                    return Err(LumaError::Gpu("shift without a scalar amount".into()));
                };
                let s = i64::from(amount);
                Reach::Deltas(if s == 0 { vec![0] } else { vec![-s, len as i64 - s] })
            }
            ShiftKind::Dims { dims, shifts } => {
                let [d0, d1, d2, _] = dims.map(i64::from);
                let strides = [1, d0, d0 * d1, d0 * d1 * d2];
                let mut deltas = vec![0i64];
                for axis in 0..4 {
                    let (dim, shift) = (i64::from(dims[axis]), i64::from(shifts[axis]));
                    if dim == 1 || shift == 0 {
                        continue;
                    }
                    let stride = strides[axis];
                    deltas = deltas
                        .iter()
                        .flat_map(|delta| [delta - shift * stride, delta + (dim - shift) * stride])
                        .collect();
                }
                Reach::Deltas(deltas)
            }
        };
        shifts.push((k as usize, reach));
    }
    if shifts.is_empty() {
        return Ok(None);
    }

    let input_count = key.input_tys.len();
    let segments = len.div_ceil(c);
    let last = len as i64 - 1;
    let mut ids: Vec<Vec<Vec<u32>>> = Vec::with_capacity(segments);
    for j in 0..segments {
        let (a, b) = ((j * c) as i64, (len.min((j + 1) * c)) as i64);
        let mut lists: Vec<Vec<u32>> = vec![Vec::new(); input_count];
        for (k, reach) in &shifts {
            let Some(list) = lists.get_mut(*k) else {
                return Err(LumaError::Gpu("shift of an unknown input".into()));
            };
            match reach {
                Reach::Deltas(deltas) => {
                    for delta in deltas {
                        let (lo, hi) = ((a + delta).max(0), (b - 1 + delta).min(last));
                        if lo <= hi {
                            list.extend((lo as usize / c..=hi as usize / c).map(|id| id as u32));
                        }
                    }
                }
                Reach::View(map, edge) => {
                    let Some(input) = input_at(key, inputs, *k) else {
                        return Err(LumaError::Gpu("view of an unknown input".into()));
                    };
                    let (lo, hi) = view_reach(map, *edge, a as usize, b as usize);
                    let cs = input.segment_len();
                    list.extend((lo / cs..=(hi / cs).min(input.segments() - 1)).map(|id| id as u32));
                }
                Reach::All => {
                    let Some(input) = input_at(key, inputs, *k) else {
                        return Err(LumaError::Gpu("take of an unknown input".into()));
                    };
                    list.extend(0..input.segments() as u32);
                }
            }
        }
        for list in &mut lists {
            list.sort_unstable();
            list.dedup();
        }
        ids.push(lists);
    }
    let mut counts = vec![0u8; input_count];
    for (k, count) in counts.iter_mut().enumerate() {
        let longest = ids.iter().map(|lists| lists[k].len()).max().unwrap_or(0);
        *count = u8::try_from(longest)
            .map_err(|_| LumaError::Chunked("a shift that reads this many buffers at once"))?;
    }
    for lists in &mut ids {
        for (list, &count) in lists.iter_mut().zip(&counts) {
            let pad = list.last().copied().unwrap_or(0);
            list.resize(usize::from(count), pad);
        }
    }

    let outputs = match key.tail {
        Tail::Store | Tail::StoreMapped(_) | Tail::ParamStoreMapped { .. } | Tail::Matmul { .. } | Tail::Solve(_) => key.outputs.len(),
        // The buffers of the sort itself: the counts, or two to write and two to read.
        Tail::Radix(_) => 4,
        Tail::Scatter { .. }
        | Tail::ParamScatter { .. }
        | Tail::ScatterAdd { .. }
        | Tail::Stack { .. }
        | Tail::ParamStack { .. }
        | Tail::Reduce(_)
        | Tail::AxisReduce(_) => 1,
    };
    let direct = (0..input_count)
        .filter(|&k| !key.in_place.iter().any(|&(paired, _)| paired as usize == k) && direct(&key.instrs, k as u32))
        .count();
    let slots: usize = counts.iter().map(|&count| usize::from(count)).sum();
    if outputs + direct + slots > max_bindings {
        return Err(LumaError::Chunked("a shift that reads this many buffers at once"));
    }
    Ok(Some((counts, ids)))
}

/// What a shift instruction reads, see [`spans`].
enum Reach {
    Deltas(Vec<i64>),
    View(ViewMap, Edge),
    /// Any element of the source (`take`).
    All,
}

/// The flat source indices `lo..=hi` that a view with `edge` reads for the output elements
/// `a..b` (a superset).
fn view_reach(map: &ViewMap, edge: Edge, a: usize, b: usize) -> (usize, usize) {
    let top = (0..4).rev().find(|&d| map.out[d] > 1).unwrap_or(0);
    let stride: usize = map.out[..top].iter().map(|&d| d as usize).product();
    let mut region = map.out.map(|d| (0, d.saturating_sub(1)));
    region[top] = ((a / stride) as u32, ((b - 1) / stride) as u32);
    let boxed = map.src_box_in(edge, region);
    let (mut lo, mut hi, mut step) = (0usize, 0usize, 1usize);
    for (&(first, last), &extent) in boxed.iter().zip(&map.src) {
        lo += first as usize * step;
        hi += last as usize * step;
        step *= extent as usize;
    }
    (lo, hi)
}

/// The slot ids of output segment `j` (see [`Shifted`]); `None` when the launch is plain.
fn segment_ids(shifted: Option<&Shifted>, j: usize) -> Result<Option<&[Vec<u32>]>, LumaError> {
    match shifted {
        None => Ok(None),
        Some(shifted) => shifted
            .ids
            .get(j)
            .map(|ids| Some(ids.as_slice()))
            .ok_or_else(|| LumaError::Gpu("segment without slot ids".into())),
    }
}

/// What a launch binds to read an input: the pinned buffer, the byte range of it, from `offset`.
pub(super) struct InPin {
    pin: Pin,
    offset: u64,
    bytes: u64,
}

impl InPin {
    fn new(pin: Pin, bytes: u64) -> Self {
        InPin { pin, offset: 0, bytes }
    }

    /// Pins the only segment of `input`. A window whose range starts where the device allows a
    /// binding to is bound as that range of its pack's buffer, instead of being copied out.
    fn of(input: &Storage) -> Result<Self, LumaError> {
        if let Some((pack, at)) = input.window() {
            let offset = pack.layout().bytes(at);
            if offset % input.ctx().info().limits.storage_offset_align.max(1) == 0 {
                return Ok(InPin { pin: pack.pin_read(0)?, offset, bytes: input.byte_len() });
            }
        }
        Ok(InPin::new(input.pin_read(0)?, input.byte_len()))
    }

    pub(super) fn raw(&self) -> RawArg<'_> {
        RawArg::storage_at(self.pin.buffer(), self.offset, self.bytes, false)
    }
}

/// Pins what segment `j` (of `n` elements) of a launch binds as inputs, in the binding order of
/// the generated source, each with the bytes to bind. `inputs` lack the in-place input of `key`.
/// With `ids` (the segmented variant) an input is bound as itself only if the program loads it
/// at the element being computed, then once per slot.
pub(super) fn pin_inputs(
    key: &FusedKey,
    inputs: &[Arc<Storage>],
    ids: Option<&[Vec<u32>]>,
    j: usize,
    n: u32,
) -> Result<Vec<InPin>, LumaError> {
    let bytes = |input: &Storage| input.layout().bytes(n as usize);
    let mut pins = Vec::with_capacity(inputs.len());
    let mut stored = inputs.iter();
    for k in 0..key.input_tys.len() {
        if key.in_place.iter().any(|&(paired, _)| paired as usize == k) {
            continue;
        }
        let Some(input) = stored.next() else {
            return Err(LumaError::Gpu("fused program with too few inputs".into()));
        };
        let Some(ids) = ids else {
            // An input of the length of the output is bound by segment; a single buffer of
            // another length (the source of a view) is bound whole.
            pins.push(if input.segments() > 1 {
                InPin::new(input.pin_read(j)?, input.segment_byte_len(j))
            } else {
                InPin::of(input)?
            });
            continue;
        };
        if direct(&key.instrs, k as u32) {
            pins.push(InPin::new(input.pin_read(j)?, bytes(input)));
        }
        for &id in ids.get(k).map_or(&[][..], Vec::as_slice) {
            let id = id as usize;
            pins.push(InPin::new(input.pin_read(id)?, input.segment_byte_len(id)));
        }
    }
    Ok(pins)
}

/// The words of the uniform of segment `j`: `n` (reductions), then, for the segmented variant,
/// the first element of the segment, the array length and the slot ids of every input, then the
/// scalars. Borrows `scalars` when nothing else goes in front of them.
pub(super) fn params<'a>(
    key: &FusedKey,
    ids: Option<&[Vec<u32>]>,
    n: Option<u32>,
    base: usize,
    len: usize,
    runs: Option<&Runs>,
    scalars: &'a [u32],
) -> Result<Cow<'a, [u32]>, LumaError> {
    let based = ids.is_none() && super::codegen::generates(key);
    if ids.is_none() && n.is_none() && !based && runs.is_none() {
        // A kernel that bakes the scalars in has no binding for the uniform (see `reads_scalars`).
        return Ok(Cow::Borrowed(if super::codegen::reads_scalars(key) { scalars } else { &[] }));
    }
    let mut words = Vec::with_capacity(scalars.len() + 3);
    words.extend(n);
    if let Some(ids) = ids {
        words.push(len_u32(base)?);
        words.push(len_u32(len)?);
        words.extend(ids.iter().flatten());
    } else if based {
        words.push(len_u32(base)?);
    }
    if let Some(runs) = runs {
        words.extend([runs.len, runs.words]);
    }
    words.extend_from_slice(scalars);
    Ok(Cow::Owned(words))
}

/// What a launch of the store kernel `key` binds for segment `j` of `outputs[0]`: `outputs`, then
/// the inputs as reads (see [`pin_inputs`]), then the uniform (see [`params`]).
pub(super) struct StoreArgs<'a> {
    pub(super) n: u32,
    outs: Vec<(Pin, u64)>,
    ins: Vec<InPin>,
    words: Cow<'a, [u32]>,
}

impl StoreArgs<'_> {
    pub(super) fn raw(&self) -> Vec<RawArg<'_>> {
        let mut raw: Vec<RawArg<'_>> = Vec::with_capacity(self.outs.len() + self.ins.len() + 1);
        raw.extend(self.outs.iter().map(|(pin, bytes)| RawArg::write(pin.buffer(), *bytes)));
        raw.extend(self.ins.iter().map(InPin::raw));
        if !self.words.is_empty() {
            raw.push(RawArg::Uniform(bytemuck::cast_slice(&self.words)));
        }
        raw
    }
}

/// The arguments of the launch of segment `j` of `outputs[0]`; `outputs` must not be empty.
pub(super) fn store_args<'a>(
    key: &FusedKey,
    outputs: &[&Storage],
    inputs: &[Arc<Storage>],
    scalars: &'a [u32],
    shifted: Option<&Shifted>,
    j: usize,
) -> Result<StoreArgs<'a>, LumaError> {
    let Some(first) = outputs.first() else {
        return Err(LumaError::Gpu("fused kernel without output".into()));
    };
    let range = first.range(j);
    let n = len_u32(range.len())?;
    let ids = segment_ids(shifted, j)?;
    let outs = outputs
        .iter()
        .map(|out| out.pin(j).map(|pin| (pin, out.segment_byte_len(j))))
        .collect::<Result<Vec<_>, _>>()?;
    let ins = pin_inputs(key, inputs, ids, j, n)?;
    let words = params(key, ids, None, range.start, shifted.map_or(0, |shifted| shifted.len), None, scalars)?;
    Ok(StoreArgs { n, outs, ins, words })
}

/// Launches the store kernel `core` of `key` once per segment of `outputs[0]`, with the arguments
/// of [`store_args`]. Counts as one elementwise launch.
fn launch_store(
    ctx: &crate::context::Context,
    core: &crate::kernel::KernelCore,
    key: &FusedKey,
    outputs: &[&Storage],
    inputs: &[Arc<Storage>],
    scalars: &[u32],
    shifted: Option<&Shifted>,
) -> Result<(), LumaError> {
    let Some(first) = outputs.first() else { return Ok(()) };
    let tiles = match (&key.tile, &key.staging) {
        (Some(tile), _) => Some(tile.tiles()),
        (None, Some(staging)) => Some(staging.tiles()),
        (None, None) => None,
    };
    if let Some(tiles) = tiles {
        if first.segments() != 1 {
            return Err(LumaError::Gpu("tiled program on an array of several segments".into()));
        }
        let args = store_args(key, outputs, inputs, scalars, shifted, 0)?;
        ctx.launch(core, fold_workgroups(ctx, tiles)?, [args.n, 1, 1], &args.raw())?;
    } else {
        for j in 0..first.segments() {
            let args = store_args(key, outputs, inputs, scalars, shifted, j)?;
            let groups = args.n.div_ceil(core.workgroup[0] * u32::from(key.coarsen.max(1)) * key.pack_factor());
            ctx.launch(core, fold_workgroups(ctx, groups)?, [args.n, 1, 1], &args.raw())?;
        }
    }
    ctx.jit().stats.elementwise_launches.fetch_add(1, Ordering::Relaxed);
    if key.coarsen > 1 {
        ctx.jit().stats.coarsened.fetch_add(1, Ordering::Relaxed);
    }
    Ok(())
}

/// The operation and location of the first output of `fused`, for the decision log (`None` for
/// traced steps, which keep no nodes).
fn origin_of(fused: &Fused) -> Option<(&'static str, &'static Location<'static>)> {
    let node = fused.instr_nodes.get(*fused.key.outputs.first()? as usize)?.as_ref()?;
    Some((node.name, node.location))
}

/// Records how many elements per invocation the launch of `key` computes, if it could have been
/// more than one.
fn note_coarsen(ctx: &crate::context::Context, key: &FusedKey, origin: (&'static str, &'static Location<'static>)) {
    let jit = ctx.jit();
    if jit.recording_decisions() && super::coarsen::coarsenable(key) {
        jit.note_decision(Decision::new(origin.0, origin.1, DecisionSite::Coarsen, format!("coarsen {}", key.coarsen.max(1))));
    }
}

/// Drops the staging the judgement chose (see `inline::judge`) if the launch cannot run it: the
/// kernel runs over segments (`key.spans`), or `len` is not the cells of the staged output.
fn settle_staging(key: &mut FusedKey, len: usize) {
    if let Some(staging) = &key.staging
        && (!key.spans.is_empty() || u64::from(staging.dims[0]) * u64::from(staging.dims[1]) != len as u64)
    {
        key.staging = None;
    }
}

/// Runs the store kernel of `fused` over `len` elements, one new buffer per output. A single
/// output takes over the buffer of an input that nothing else holds (see [`in_place_eligible`]),
/// instead of a new one.
pub(super) fn run_store(ctx: &crate::context::Context, len: usize, fused: Fused) -> Result<Vec<Storage>, Failure> {
    if !super::plancache::recording() {
        return run_store_planned(ctx, len, fused);
    }
    let step = super::plancache::begin_store(len, &fused);
    let result = run_store_planned(ctx, len, fused);
    super::plancache::end_store(step, &result);
    result
}

/// [`run_store`] without the plan cache's recording.
fn run_store_planned(ctx: &crate::context::Context, len: usize, fused: Fused) -> Result<Vec<Storage>, Failure> {
    fused.mark_fused();
    let origin = ctx.jit().recording_decisions().then(|| origin_of(&fused)).flatten().unwrap_or(("elementwise", Location::caller()));
    let Fused { mut key, mut inputs, mut scalars, fused_nodes, instr_nodes, sub_roots } = fused;
    // The fused nodes hold their operands; a dying input is only uniquely owned without them.
    drop((fused_nodes, instr_nodes, sub_roots));
    let count = key.outputs.len();

    let mut reused = None;
    if count == 1 {
        let candidate = (0..inputs.len()).find(|&k| {
            let input = &inputs[k];
            Arc::strong_count(input) == 1
                && !input.is_alias()
                && !input.is_window()
                && in_place_eligible(&key, k as u32)
        });
        if let Some(k) = candidate {
            match Arc::try_unwrap(inputs.remove(k)) {
                Ok(storage) => {
                    ctx.jit().note_write();
                    super::plancache::taint("a launch took over the buffer of an input");
                    key.in_place = vec![(k as u32, 0)];
                    reused = Some(storage);
                }
                Err(shared) => inputs.insert(k, shared),
            }
        }
    }
    if let Some(plan) = super::transpose::plan(ctx, &key, &inputs, len) {
        return Ok(vec![super::transpose::run(ctx, &plan, &inputs[0], len)?]);
    }
    if let Some(plan) = super::transpose::plan_fused(ctx, &key, &inputs, &scalars, len, origin.0, origin.1)? {
        return Ok(vec![super::transpose::run_fused(ctx, &plan, &inputs, len, reused)?]);
    }
    let shifted = plan_shifts(ctx, &mut key, &inputs, &scalars, len)?;
    settle_staging(&mut key, len);
    if shifted.is_none() && key.staging.is_none() {
        key.tile = super::codegen::plan_tile(&key, &scalars, len, ctx.tuning());
    }
    key.coarsen = super::coarsen::plan(ctx, &key, len);
    note_coarsen(ctx, &key, origin);
    let core = ctx.jit().kernel(ctx, &mut key, &mut scalars)?;

    let outputs = match reused {
        Some(storage) => vec![storage],
        None => {
            let mut outputs = Vec::with_capacity(count);
            for &at in &key.outputs {
                let ty = key.instrs.get(at as usize).map_or(ElemTy::F32, |instr| instr.op.out_ty());
                outputs.push(Storage::new(ctx, len, ty.layout()));
            }
            outputs
        }
    };
    let refs: Vec<&Storage> = outputs.iter().collect();
    launch_store(ctx, &core, &key, &refs, &inputs, &scalars, shifted.as_ref())?;
    Ok(outputs)
}

/// Evaluates `node` into `target`, one of its inputs, overwriting it in place: one fused kernel
/// that reads every element of `target` before writing it. When simplification removed every
/// read of `target` (`x -= x` under fast math, `x *= 0` on integers), the node no longer has it
/// as an input and the kernel is a plain store into `target`. The node counts as evaluated, with
/// `target` as its result. Callers must have evaluated every other pending operation that reads
/// `target` (see [`super::Readers::flush`]) and the program must allow in-place execution; the
/// compound assignment of an array guarantees both.
pub(crate) fn evaluate_into(node: &Arc<Node>, target: &Arc<Storage>) -> Result<(), Failure> {
    let Some(args) = lock(&node.args).take() else {
        return Err(Failure::New(LumaError::Gpu("lazy node evaluated recursively".into())));
    };
    let (fused, args) = flatten_bounded(node, args, Tail::Store)?;
    drop(args);
    let ctx = node.context();
    ctx.jit().note_write();
    fused.mark_fused();
    let Fused { mut key, mut inputs, mut scalars, fused_nodes, instr_nodes, sub_roots } = fused;
    drop((fused_nodes, instr_nodes, sub_roots));
    let origin = (node.name, node.location);
    let at = inputs.iter().position(|input| Arc::ptr_eq(input, target));
    if let Some(k) = at {
        if !in_place_eligible(&key, k as u32) {
            return Err(Failure::New(LumaError::Gpu("in-place assignment cannot run in place".into())));
        }
        inputs.remove(k);
        key.in_place = vec![(k as u32, 0)];
    }
    let shifted = plan_shifts(ctx, &mut key, &inputs, &scalars, node.len())?;
    settle_staging(&mut key, node.len());
    if shifted.is_none() && key.staging.is_none() {
        key.tile = super::codegen::plan_tile(&key, &scalars, node.len(), ctx.tuning());
    }
    key.coarsen = super::coarsen::plan(ctx, &key, node.len());
    note_coarsen(ctx, &key, origin);
    let core = ctx.jit().kernel(ctx, &mut key, &mut scalars)?;
    launch_store(ctx, &core, &key, &[&**target], &inputs, &scalars, shifted.as_ref())?;
    check_finite(ctx, target, node.op().out_ty())?;
    // Another thread cannot have evaluated this node: its operands were taken above.
    if node.result.set(Ok(Arc::clone(target))).is_ok() {
        node.settled();
    }
    Ok(())
}

/// How many storage buffers a launch of [`run_mapped`] binds: the `bases`, then the inputs that
/// are not one of them.
pub(super) fn mapped_bindings(inputs: &[Arc<Storage>], bases: &[Arc<Storage>]) -> usize {
    bases.len() + inputs.iter().filter(|input| !bases.iter().any(|base| Arc::ptr_eq(base, input))).count()
}

/// Workgroups of a private atomic scatter of integers (see [`Tail::ScatterAdd`]) on the reference
/// device ([`crate::tuning::REFERENCE_COMPUTE_UNITS`], 3 per multiprocessor), scaled to the device. A
/// histogram of 256 bins of 10M `f32` on an RTX 3090 took 0.0632 ms with 1024 workgroups, 0.0512
/// with 246 (2 to 4 per multiprocessor all took 0.050-0.051, 1 per multiprocessor 0.070).
const PRIVATE_GROUPS: u32 = 246;

/// Runs the program `fused` (tail [`Tail::StoreMapped`], [`Tail::Scatter`] or [`Tail::ScatterAdd`]) over `n` elements,
/// writing into the buffers of `bases` (output `j` is `bases[j]`): one launch, counted as an
/// elementwise launch. An input that is one of the bases is read from that buffer, before the
/// invocation overwrites it; every other input must fit one buffer. The caller has made sure no
/// element is read after another invocation wrote it (see `jit::assign`).
pub(super) fn run_mapped(
    ctx: &crate::context::Context,
    fused: Fused,
    bases: &[Arc<Storage>],
    n: usize,
) -> Result<(), Failure> {
    fused.mark_fused();
    let Fused { mut key, mut inputs, mut scalars, fused_nodes, instr_nodes, sub_roots } = fused;
    // The fused nodes hold their operands; dropping them lets the buffers go once the kernel ran.
    drop((fused_nodes, instr_nodes, sub_roots));
    for k in (0..inputs.len()).rev() {
        if let Some(j) = bases.iter().position(|base| Arc::ptr_eq(base, &inputs[k])) {
            inputs.remove(k);
            key.in_place.push((k as u32, j as u32));
        }
    }
    if inputs.iter().any(|input| input.segments() != 1) {
        return Err(LumaError::Chunked("assigning from an array split over several buffers").into());
    }
    if bases.iter().any(|base| base.segments() != 1) {
        return Err(LumaError::Chunked("assigning into an array split over several buffers").into());
    }
    let n = len_u32(n)?;
    // A region write of bits that are made of long runs of consecutive elements builds whole words.
    let runs = match key.tail {
        Tail::StoreMapped(map) if key.outputs.iter().any(|&at| key.instrs.get(at as usize).is_some_and(|instr| instr.op.out_ty() == ElemTy::Bool)) => Runs::plan(&map),
        _ => None,
    };
    key.runs = runs.is_some();
    let core = ctx.jit().kernel(ctx, &mut key, &mut scalars)?;
    let outs = bases.iter().map(|base| base.pin(0)).collect::<Result<Vec<_>, _>>()?;
    let ins = pin_inputs(&key, &inputs, None, 0, n)?;
    let words = params(&key, None, None, 0, n as usize, runs.as_ref(), &scalars)?;
    let mut raw: Vec<RawArg<'_>> = Vec::with_capacity(outs.len() + ins.len() + 1);
    raw.extend(outs.iter().zip(bases).map(|(pin, base)| RawArg::write(pin.buffer(), base.byte_len())));
    raw.extend(ins.iter().map(InPin::raw));
    if !words.is_empty() {
        raw.push(RawArg::Uniform(bytemuck::cast_slice(&words)));
    }
    match key.tail {
        // Workgroups that stay resident, each striding over the elements: merging the private
        // copies into the target (a global atomic per touched element per workgroup) is then cheap
        // and no wave of workgroups is left half empty. The merge of `f32` (a compare-and-exchange
        // loop on devices without native atomic adds) serialises on contended words, and its
        // accumulators want more workgroups to hide their latency: they keep 256.
        Tail::ScatterAdd { len, private: true, .. } => {
            let float = key.outputs.first().and_then(|&at| key.instrs.get(at as usize)).is_some_and(|instr| instr.op.out_ty() == ElemTy::F32);
            let cap = if float { 256 } else { ctx.tuning().scaled(PRIVATE_GROUPS) };
            ctx.launch(&core, fold_workgroups(ctx, (n / (4 * len.max(1))).clamp(1, cap))?, [n, 1, 1], &raw)?
        }
        _ => launch_1d(ctx, &core, runs.as_ref().map_or(n, |runs| runs.invocations), &raw)?,
    }
    ctx.jit().stats.elementwise_launches.fetch_add(1, Ordering::Relaxed);
    Ok(())
}

/// The runs of consecutive base elements a region write is made of (see `kirgen::Build::runs`).
pub(super) struct Runs {
    /// Elements of a run.
    len: u32,
    /// Words of the base a run spans, at most: the invocations of a run are the 32 bits of that many words.
    words: u32,
    /// The invocations of the launch: 32 for every word of every run.
    invocations: u32,
}

impl Runs {
    /// Runs shorter than this are written bit by bit: most of their words would be shared.
    const MIN_LEN: u64 = 64;
    /// Most runs whose alignment is looked at to size their words; with more the worst case is taken.
    const EXACT_RUNS: u64 = 4096;

    /// The runs of the region `map` writes, if they are long enough: along the output's first
    /// axis the region is the base's first axis at step 1, and so is the next axis whenever the
    /// first is whole, which makes a run of both (and so on).
    fn plan(map: &ViewMap) -> Option<Runs> {
        let axis = map.axes[0];
        if axis.from != 0 || axis.step != 1 {
            return None;
        }
        let mut len = u64::from(map.out[0]);
        for d in 0..3 {
            let whole = map.out[d] == map.src[d] && map.axes[d].start == 0;
            let next = map.axes[d + 1].from as usize == d + 1 && map.axes[d + 1].step == 1;
            if !(whole && next) {
                break;
            }
            len *= u64::from(map.out[d + 1]);
        }
        if len < Self::MIN_LEN {
            return None;
        }
        let count = map.out_len() as u64 / len;
        let words = if count <= Self::EXACT_RUNS {
            (0..count).map(|run| (Self::start(map, run * len) % 32 + len).div_ceil(32)).max()?
        } else {
            (31 + len).div_ceil(32)
        };
        let invocations = u32::try_from(count * words * 32).ok()?;
        Some(Runs { len: u32::try_from(len).ok()?, words: u32::try_from(words).ok()?, invocations })
    }

    /// The element of the base that element `o` of the region goes to.
    fn start(map: &ViewMap, o: u64) -> u64 {
        let (mut out_stride, mut src_stride) = ([1u64; 4], [1u64; 4]);
        for d in 1..4 {
            out_stride[d] = out_stride[d - 1] * u64::from(map.out[d - 1]);
            src_stride[d] = src_stride[d - 1] * u64::from(map.src[d - 1]);
        }
        (0..4)
            .map(|a| {
                let axis = map.axes[a];
                let from = axis.from as usize;
                let coord = (o / out_stride[from] % u64::from(map.out[from])) as i64 * i64::from(axis.step) + i64::from(axis.start);
                coord.max(0) as u64 * src_stride[a]
            })
            .sum()
    }
}

/// Runs the first pass of the reductions `ops` directly over the expression behind `node`,
/// without materializing it: every workgroup evaluates its strided blocks of `Tuning::reduce_block` elements and reduces them,
/// once per op. Returns the buffer of partial results (op `j` at `j * groups ..`) and how many
/// there are per op, or `None` if the node has already been evaluated or is better evaluated
/// first (reduce its buffer instead). With one segment and one group per op the partials are
/// the final values and are written into `target` when it is given; `finish` is applied to them
/// then, inside the pass (the caller sees that this happened: one segment and one group), and
/// not otherwise. With `Tuning::single_pass_reduce` a pass of one segment and several workgroups
/// finishes the reduction itself (see `ReduceList::single_pass`): the one partial result per op
/// is then the final value, with `finish` applied, in `target` when it is given.
pub(crate) fn reduce_first_pass(
    node: &Arc<Node>,
    ops: ReduceList,
    scratch: &mut Scratch<'_>,
    target: Option<&Buffer>,
    finish: Finish,
) -> Result<Option<Vec<(Buffer, u32)>>, Failure> {
    if !node.is_pending() {
        return Ok(None);
    }
    let Some(args) = node.operands() else { return Ok(None) };
    let (fused, args) = flatten_bounded(node, args, Tail::Reduce(ops))?;
    drop(args);
    // A reduction calls the sub-programs of circular shifts once per shift; nested or large ones
    // are cheaper staged by the store kernel of the node, whose buffer is then reduced. Programs
    // that read sub-programs through views compute each index once and are reduced directly.
    let max = node.context().tuning().max_recomputed;
    if !fused.key.inlines() && fused.key.subs.iter().any(|sub| sub.len() > max || sub.iter().any(|instr| matches!(instr.args[0], Ref::Sub(_)))) {
        return Ok(None);
    }
    reduce_fused(node.context(), node.len(), fused, ops, scratch, target, finish).map(Some)
}

/// The first pass of [`reduce_first_pass`] for an already flattened expression of `len` elements,
/// applying `finish` if the pass is the last. With [`Tuning::single_pass_reduce`], one segment
/// and more than one group, the pass finishes the reduction itself (see
/// [`ReduceList::single_pass`]) and the one partial result it returns is the final value.
pub(super) fn reduce_fused(
    ctx: &crate::context::Context,
    len: usize,
    fused: Fused,
    ops: ReduceList,
    scratch: &mut Scratch<'_>,
    target: Option<&Buffer>,
    finish: Finish,
) -> Result<Vec<(Buffer, u32)>, Failure> {
    fused.mark_fused();
    let Fused { mut key, inputs, mut scalars, .. } = fused;
    // The segments are those an array of this length has: the inputs may be of other lengths.
    let layout = layout_elem(&key, &inputs, len);
    let first = Storage::new(ctx, len, ElemLayout::plain(layout as usize));
    let segments = first.segments();
    let only_groups = if segments == 1 { Some(ctx.tuning().reduce_groups(len_u32(first.range(0).len())?)) } else { None };
    // One segment: a pass of one group, or one that finishes itself, leaves the final values, so
    // the epilogue goes in it.
    let single_pass = ctx.tuning().single_pass_reduce && only_groups.is_some_and(|groups| groups > 1);
    let last = !finish.epilogue.is_empty() && only_groups == Some(1);
    if single_pass {
        key.tail = Tail::Reduce(ops.with_epilogue(finish.epilogue).single_pass());
    } else if last {
        key.tail = Tail::Reduce(ops.with_epilogue(finish.epilogue));
    }
    let shifted = plan_shifts(ctx, &mut key, &inputs, &scalars, len)?;
    let core = ctx.jit().kernel(ctx, &mut key, &mut scalars)?;

    let cell = out_elems(&key).first().copied().unwrap_or(4) * ops.as_slice().len() as u64;
    let mut partials = Vec::with_capacity(segments);
    for j in 0..segments {
        let range = first.range(j);
        let n = len_u32(range.len())?;
        let groups = ctx.tuning().reduce_groups(n);
        let out_bytes = u64::from(groups) * cell;
        let out = match target {
            Some(target) if segments == 1 && groups == 1 => target.clone(),
            _ => scratch.take(out_bytes)?,
        };
        // The partial results of a pass that finishes itself go to `out`, the final values here.
        let values = if single_pass {
            Some(match target {
                Some(target) => target.clone(),
                None => scratch.take(cell)?,
            })
        } else {
            None
        };
        let ids = segment_ids(shifted.as_ref(), j)?;
        let mut params = params(&key, ids, Some(n), range.start, len, None, &scalars)?;
        if last || single_pass {
            params.to_mut().extend(finish.uniform_words());
        }
        let pins = pin_inputs(&key, &inputs, ids, j, n)?;
        let mut raw: Vec<RawArg<'_>> = Vec::with_capacity(pins.len() + 4);
        raw.push(RawArg::write(&out, out_bytes));
        raw.extend(pins.iter().map(InPin::raw));
        if let Some(values) = &values {
            raw.push(RawArg::write(ctx.reduce_counter()?, 4));
            raw.push(RawArg::write(values, cell));
        }
        raw.push(RawArg::Uniform(bytemuck::cast_slice(&params)));
        ctx.launch(&core, fold_workgroups(ctx, groups)?, [n, 1, 1], &raw)?;
        drop(raw);
        partials.push(match values {
            Some(values) => (values, 1),
            None => (out, groups),
        });
    }
    ctx.jit().stats.reduce_launches.fetch_add(1, Ordering::Relaxed);
    Ok(partials)
}

/// [`reduce_fused`] of one segment and one unsplit program that also stores the instructions
/// `store_outs` in the same kernel, one new buffer each. Returns them, the buffer of partial
/// results and how many there are (with `Tuning::single_pass_reduce` and several workgroups the
/// pass finishes the reduction itself: one value per op, the final one). `finish` is applied
/// inside the pass when it leaves the final values (one group, or a single pass: one partial
/// result is returned then), which go to `target` if it is given. With a `gate` (its id and the
/// arguments of its buffers, see `ReduceList::with_gate`) the pass also runs the gate on the
/// final value, and the pass must be one that leaves it: one group, or a single pass.
#[allow(clippy::too_many_arguments)]
pub(super) fn reduce_and_store_fused(
    ctx: &crate::context::Context,
    len: usize,
    mut fused: Fused,
    store_outs: &[u32],
    ops: ReduceList,
    scratch: &mut Scratch<'_>,
    target: Option<&Buffer>,
    finish: Finish,
    gate: Option<(GateId, &[RawArg<'_>])>,
) -> Result<(Vec<Storage>, Buffer, u32), Failure> {
    if store_outs.iter().any(|&at| fused.key.instrs.get(at as usize).is_none_or(|instr| instr.op.out_ty().is_packed())) {
        return Err(LumaError::Chunked("a fused store and reduction on a packed output").into());
    }
    let last = fused.key.instrs.len() as u32 - 1;
    fused.key.outputs = store_outs.iter().copied().chain(std::iter::once(last)).collect();
    fused.mark_fused();
    let Fused { mut key, inputs, mut scalars, .. } = fused;
    let layout = layout_elem(&key, &inputs, len);
    let first = Storage::new(ctx, len, ElemLayout::plain(layout as usize));
    if first.segments() != 1 {
        return Err(LumaError::Chunked("a fused store and reduction on an array of several segments").into());
    }
    let n = len_u32(first.range(0).len())?;
    let groups = ctx.tuning().reduce_groups(n);
    let single_pass = ctx.tuning().single_pass_reduce && groups > 1;
    let finished = !finish.epilogue.is_empty() && (single_pass || groups == 1);
    if gate.is_some() && !single_pass && groups != 1 {
        return Err(LumaError::Chunked("a gate on a reduction that does not finish in its first pass").into());
    }
    if single_pass {
        key.tail = Tail::Reduce(ops.with_epilogue(finish.epilogue).single_pass());
    } else if finished {
        key.tail = Tail::Reduce(ops.with_epilogue(finish.epilogue));
    }
    if let (Some((id, _)), Tail::Reduce(list)) = (&gate, key.tail) {
        key.tail = Tail::Reduce(list.with_gate(*id));
    }
    let shifted = plan_shifts(ctx, &mut key, &inputs, &scalars, len)?;
    if shifted.is_some() {
        return Err(LumaError::Chunked("a fused store and reduction on shifts over several buffers").into());
    }
    let core = ctx.jit().kernel(ctx, &mut key, &mut scalars)?;
    let stored: Vec<Storage> = store_outs.iter().map(|&at| Storage::new(ctx, len, key.instrs[at as usize].op.out_ty().layout())).collect();
    if stored.iter().any(|out| out.segments() != 1) {
        return Err(LumaError::Chunked("a fused store and reduction on an array of several segments").into());
    }

    let cell = out_elems(&key).first().copied().unwrap_or(4) * ops.as_slice().len() as u64;
    let out_bytes = u64::from(groups) * cell;
    let partial = match target {
        Some(target) if groups == 1 => target.clone(),
        _ => scratch.take(out_bytes)?,
    };
    let values = if single_pass {
        Some(match target {
            Some(target) => target.clone(),
            None => scratch.take(cell)?,
        })
    } else {
        None
    };
    let mut params = params(&key, None, Some(n), first.range(0).start, len, None, &scalars)?;
    if finished {
        params.to_mut().extend(finish.uniform_words());
    }
    let pins = pin_inputs(&key, &inputs, None, 0, n)?;
    let store_pins = stored.iter().map(|out| out.pin(0).map(|pin| (pin, out.segment_byte_len(0)))).collect::<Result<Vec<_>, _>>()?;
    let mut raw: Vec<RawArg<'_>> = Vec::with_capacity(pins.len() + store_pins.len() + 4);
    raw.push(RawArg::write(&partial, out_bytes));
    raw.extend(store_pins.iter().map(|(pin, bytes)| RawArg::write(pin.buffer(), *bytes)));
    raw.extend(pins.iter().map(InPin::raw));
    if let Some(values) = &values {
        raw.push(RawArg::write(ctx.reduce_counter()?, 4));
        raw.push(RawArg::write(values, cell));
    }
    if let Some((_, args)) = gate {
        debug_assert_eq!(args.len(), crate::builtin::GATE_BINDINGS);
        raw.extend_from_slice(args);
    }
    raw.push(RawArg::Uniform(bytemuck::cast_slice(&params)));
    ctx.launch(&core, fold_workgroups(ctx, groups)?, [n, 1, 1], &raw)?;
    drop(raw);
    let stats = &ctx.jit().stats;
    stats.reduce_launches.fetch_add(1, Ordering::Relaxed);
    stats.elementwise_launches.fetch_add(1, Ordering::Relaxed);
    Ok(match values {
        Some(values) => (stored, values, 1),
        None => (stored, partial, groups),
    })
}

/// One pending reduction that [`reduce_many`] runs with others: the pending node it reduces (of
/// `len` elements), its ops, and what is applied to their values.
pub(crate) struct Member<'a> {
    pub(crate) operand: &'a Arc<Node>,
    pub(crate) ops: ReduceList,
    pub(crate) finish: Finish,
}

/// What [`reduce_many`] does with the kernel it builds.
pub(crate) enum ManyMode<'a> {
    /// Launch it. `ahead` runs between the build and the launch (which waits for the compile) when the kernel was not
    /// built before: where the caller starts the compiles of the kernels it will need next. With
    /// `Tuning::reduce_multi_after`, a group is not launched (nothing is, `deferred` is set) before it has been recorded that many
    /// times, nor while its kernel is compiling: the caller records the reductions one by one.
    Run { ahead: &'a dyn Fn(), deferred: &'a std::cell::Cell<bool> },
    /// Only build it, which starts its compile in the background: nothing is launched, the result is `None`.
    BuildOnly,
}

/// Runs the reductions of `members` (pending nodes of one length, of element type `ty`: `f32`,
/// `u32` or `i32`) in one kernel: the program computes the operands together, once what they share,
/// and every op of every member has an accumulator (at most `max_accs` in all). The first pass is
/// one launch (two with several workgroups where the device cannot finish it in the same one, see
/// `Tuning::single_pass_reduce`); op `j` of the members taken in order is element `j` of the buffer
/// of the returned storage, with the member's epilogue applied. Every op combines its elements in the order a
/// reduction of the member alone does, so the values are the same bits.
///
/// `None` (nothing launched) where the program does not fit a single such kernel: sub-programs read
/// through shifts, arrays split over several buffers, too many bindings.
pub(crate) fn reduce_many(ctx: &crate::context::Context, members: &[Member<'_>], ty: ElemTy, max_accs: usize, mode: ManyMode<'_>) -> Result<Option<Storage>, Failure> {
    let Some(first) = members.first() else { return Ok(None) };
    let len = first.operand.len();
    let total: usize = members.iter().map(|member| member.ops.as_slice().len()).sum();
    if total > max_accs.min(crate::builtin::MAX_ACCS) || !matches!(ty, ElemTy::F32 | ElemTy::U32 | ElemTy::I32) {
        return Ok(None);
    }
    // One root per distinct operand.
    let mut operands: Vec<&Arc<Node>> = Vec::with_capacity(members.len());
    let mut member_root = Vec::with_capacity(members.len());
    for member in members {
        if member.operand.len() != len || member.ops.roots().is_some() {
            return Ok(None);
        }
        let at = match operands.iter().position(|known| Arc::ptr_eq(known, member.operand)) {
            Some(at) => at,
            None => {
                operands.push(member.operand);
                operands.len() - 1
            }
        };
        member_root.push(at);
    }
    let mut roots = Vec::with_capacity(operands.len());
    for node in &operands {
        let Some(args) = node.operands() else { return Ok(None) };
        roots.push((Arc::clone(node), args));
    }
    let Ok(fused) = flatten_inner(&roots, Tail::Reduce(ReduceList::new(&[ReduceOp::Sum])), &[]) else { return Ok(None) };
    drop(roots);
    let max = ctx.tuning().max_recomputed;
    if !fused.key.inlines() && fused.key.subs.iter().any(|sub| sub.len() > max || sub.iter().any(|instr| matches!(instr.args[0], Ref::Sub(_)))) {
        return Ok(None);
    }
    let fits = fused.key.outputs.len() == operands.len()
        && fused.key.outputs.iter().all(|&at| fused.key.instrs.get(at as usize).is_some_and(|instr| instr.op.out_ty() == ty))
        && (fused.key.subs.is_empty() || (fused.key.inlines() && fused.key.staging.is_none()))
        && fused.key.narrow()
        && fused.inputs.iter().all(|input| input.segments() == 1)
        && fused.inputs.len() + 4 <= ctx.jit().max_inputs();
    if !fits {
        return Ok(None);
    }
    // The program's outputs: equal values are one root.
    let mut outputs: Vec<u32> = Vec::with_capacity(operands.len());
    let mut root_of_operand = Vec::with_capacity(operands.len());
    for &at in &fused.key.outputs {
        let slot = match outputs.iter().position(|&known| known == at) {
            Some(slot) => slot,
            None => {
                outputs.push(at);
                outputs.len() - 1
            }
        };
        root_of_operand.push(slot);
    }
    let (mut ops, mut op_roots, mut epilogues, mut scalars_of_ops) = (Vec::with_capacity(total), Vec::with_capacity(total), Vec::with_capacity(total), Vec::new());
    for (member, &root) in members.iter().zip(&member_root) {
        for &op in member.ops.as_slice() {
            ops.push(op);
            op_roots.push(u8::try_from(root_of_operand[root]).map_err(|_| LumaError::Chunked("too many roots"))?);
            epilogues.push(member.finish.epilogue);
            scalars_of_ops.extend(member.finish.uniform_words());
        }
    }
    let mut key = fused.key.clone();
    let (inputs, mut scalars) = (&fused.inputs, fused.scalars.clone());
    key.outputs = outputs;
    let first_pass = ReduceList::merged(&ops, &op_roots, &epilogues);
    let probe = Storage::new(ctx, len, ElemLayout::plain(ty.stride() as usize));
    if probe.segments() != 1 {
        return Ok(None);
    }
    let n = len_u32(probe.range(0).len())?;
    let groups = ctx.tuning().reduce_groups(n);
    let single_pass = ctx.tuning().single_pass_reduce && groups > 1;
    let last = groups == 1;
    key.tail = Tail::Reduce(match (single_pass, last) {
        (true, _) => first_pass.finished(Default::default()).single_pass(),
        (false, true) => first_pass.finished(Default::default()),
        (false, false) => first_pass,
    });
    if plan_shifts(ctx, &mut key, inputs, &scalars, len)?.is_some() {
        return Ok(None);
    }
    let after = ctx.tuning().reduce_multi_after;
    // The structure of the group (not the values of its scalars, which are not in the key): what is counted.
    let structure = if after > 0 { super::fast_key_hash(&key) } else { 0 };
    match &mode {
        ManyMode::BuildOnly if after > 0 && !ctx.jit().merged_due(structure, after, false) => return Ok(None),
        ManyMode::Run { deferred, .. } if after > 0 && !ctx.jit().merged_due(structure, after, true) => {
            deferred.set(true);
            return Ok(None);
        }
        _ => {}
    }
    let built_before = ctx.jit().stats.built.load(Ordering::Relaxed);
    let core = ctx.jit().kernel(ctx, &mut key, &mut scalars)?;
    match mode {
        ManyMode::BuildOnly => return Ok(None),
        ManyMode::Run { ahead, deferred } => {
            if ctx.jit().stats.built.load(Ordering::Relaxed) != built_before {
                ahead();
            }
            // Never a launch that waits for the compile of a kernel nothing needs yet.
            if after > 0 && !core.raw.is_ready() {
                deferred.set(true);
                return Ok(None);
            }
        }
    }

    let elem = ty.stride();
    let cell = elem * total as u64;
    let pack = Storage::new(ctx, total, ElemLayout::plain(elem as usize));
    let pack_pin = pack.pin(0)?;
    let target = pack_pin.buffer();
    let mut scratch = Scratch::new(ctx);
    let out_bytes = u64::from(groups) * cell;
    let out = if last { target.clone() } else { scratch.take(out_bytes)? };
    let mut params = params(&key, None, Some(n), probe.range(0).start, len, None, &scalars)?;
    if last || single_pass {
        params.to_mut().extend(scalars_of_ops.iter().copied());
    }
    let pins = pin_inputs(&key, inputs, None, 0, n)?;
    let mut raw: Vec<RawArg<'_>> = Vec::with_capacity(pins.len() + 4);
    raw.push(RawArg::write(&out, out_bytes));
    raw.extend(pins.iter().map(InPin::raw));
    if single_pass {
        raw.push(RawArg::write(ctx.reduce_counter()?, 4));
        raw.push(RawArg::write(target, cell));
    }
    raw.push(RawArg::Uniform(bytemuck::cast_slice(&params)));
    ctx.launch(&core, fold_workgroups(ctx, groups)?, [n, 1, 1], &raw)?;
    drop(raw);
    ctx.jit().stats.reduce_launches.fetch_add(1, Ordering::Relaxed);
    if !single_pass && !last {
        crate::reduce::finish_passes_merged(ctx, &mut scratch, first_pass, ty.wgsl(), elem, out, groups, target, &scalars_of_ops)?;
    }
    drop(pack_pin);
    fused.mark_fused();
    Ok(Some(pack))
}

/// The finite check (see `ContextOptions::check_finite`): fails with [`LumaError::NonFinite`] if
/// `storage`, holding elements of type `ty`, contains a NaN or an infinity. Does nothing unless
/// the option is on and the elements are `f32`, `Df32` or `C32`. Waits for the GPU (once per segment).
pub(crate) fn check_finite(ctx: &crate::context::Context, storage: &Storage, ty: ElemTy) -> Result<(), LumaError> {
    if !ctx.check_finite() || !matches!(ty, ElemTy::F32 | ElemTy::Df32 | ElemTy::C32) {
        return Ok(());
    }
    let core = ctx.builtin(Builtin::NonFinite, "u32")?;
    let max = ReduceList::new(&[ReduceOp::Max]);
    let mut scratch = Scratch::new(ctx);
    for j in 0..storage.segments() {
        let bytes = storage.segment_byte_len(j);
        let words = len_u32((bytes / 4) as usize)?;
        let pin = storage.pin_read(j)?;
        let flags = scratch.take(u64::from(words) * 4)?;
        launch_1d(ctx, &core, words, &[RawArg::write(&flags, u64::from(words) * 4), RawArg::read(pin.buffer(), bytes)])?;
        let result = crate::reduce::finish_passes(ctx, &mut scratch, max, "u32", 4, flags, words, false, None, Finish::default())?;
        if ctx.read::<u32>(&result, 4)?.first().is_some_and(|&flag| flag != 0) {
            return Err(LumaError::NonFinite);
        }
    }
    Ok(())
}

/// One operand of the product [`matmul`] computes.
pub(crate) enum MatmulSide {
    /// An evaluated array: a pin on its buffer and the bytes of it the array holds.
    Ready(Pin, u64),
    /// A lazy expression, fused into the loads of the kernel unless it cannot be (see
    /// [`fusable`]).
    Pending(Arc<Node>),
}

/// What the source of the kernel of a product depends on (see `matmul::body`).
#[derive(Clone, Copy)]
pub(crate) struct MatmulKernel {
    pub(crate) tile: MatmulTile,
    pub(crate) aligned: bool,
    pub(crate) split: bool,
}

/// Whether the product kernel can run `fused`, the program of the operands that are expressions,
/// with `unfused` more operands bound as buffers: sub-programs of circular shifts are evaluated
/// before a kernel that reads every element many times calls them (those read through views
/// are computed at the index they are read at, see `inline`), arrays split over several
/// buffers cannot be bound, and the buffers and scalars have to fit what a kernel binds.
fn fusable(ctx: &crate::context::Context, fused: &Fused, unfused: usize) -> bool {
    let max_words = usize::try_from(ctx.max_uniform_bytes() / 4).unwrap_or(usize::MAX).saturating_sub(64 + 4);
    (fused.key.subs.is_empty() || fused.key.inlines())
        && fused.inputs.iter().all(|input| input.segments() == 1)
        && fused.inputs.len() + unfused <= ctx.jit().max_inputs()
        && fused.scalars.len() <= max_words
}

/// Whether the matrix `node` (of `dims`) is a view that reads its source transposed: the
/// elements that are neighbours along axis 1 of the matrix are neighbours in memory, those along
/// axis 0 are not. The kernel then divides the slices of that operand differently (see
/// [`MatmulOperand::Transposed`]).
fn stored_transposed(node: &Node, dims: [u32; 2]) -> bool {
    let OpKind::Shift(ShiftKind::View { map, .. }) = node.op().kind else { return false };
    if map.out != [dims[0], dims[1], 1, 1] {
        return false;
    }
    // The distance in the source buffer of the elements that are neighbours along the axes.
    let mut distance = [0i64; 2];
    let mut below = 1i64;
    for (axis, &size) in map.axes.iter().zip(&map.src) {
        if let Some(d) = distance.get_mut(usize::from(axis.from)).filter(|_| size > 1) {
            *d += i64::from(axis.step) * below;
        }
        below *= i64::from(size);
    }
    distance[1].abs() == 1 && distance[0].abs() != 1
}

/// Whether the elements of the left operand `side` (`[K, M]`) that are neighbours along `K` are
/// neighbours in memory: false for a view that reads its source transposed.
pub(crate) fn stored_along_k(side: &MatmulSide, dims: [u32; 2]) -> bool {
    match side {
        MatmulSide::Pending(node) => !stored_transposed(node, dims),
        MatmulSide::Ready(..) => true,
    }
}

/// Flattens the operands in `slots` (the lazy ones, by side) into one product program, or `None`
/// if there are none or they cannot be fused: those that cannot be fused alone (all, if only
/// together they are too much) are then evaluated into `ready` and the rest is tried again.
fn flatten_operands(
    ctx: &crate::context::Context,
    kernel: MatmulKernel,
    transposed: [bool; 2],
    slots: &mut [Option<Arc<Node>>; 2],
    ready: &mut [Option<(Pin, u64)>; 2],
) -> Result<Option<Fused>, Failure> {
    let attempt = |slots: &[Option<Arc<Node>>; 2], sides: &[usize]| -> Result<Option<Fused>, Failure> {
        let mut roots = Vec::with_capacity(2);
        for &s in sides {
            let Some(node) = &slots[s] else { return Ok(None) };
            // A node another thread evaluates meanwhile has no operands: it is read as a buffer.
            let Some(args) = node.operands() else { return Ok(None) };
            roots.push((Arc::clone(node), args));
        }
        let operand = |s: usize| match (sides.contains(&s), transposed[s]) {
            (false, _) => MatmulOperand::Buffer,
            (true, false) => MatmulOperand::Fused,
            (true, true) => MatmulOperand::Transposed,
        };
        let tail = Tail::Matmul { tile: kernel.tile, aligned: kernel.aligned, split: kernel.split, lhs: operand(0), rhs: operand(1) };
        let fused = flatten(&roots, tail)?;
        drop(roots);
        Ok(fusable(ctx, &fused, 2 - sides.len()).then_some(fused))
    };
    loop {
        let live: Vec<usize> = (0..2).filter(|&s| slots[s].is_some()).collect();
        if live.is_empty() {
            return Ok(None);
        }
        if let Some(fused) = attempt(slots, &live)? {
            return Ok(Some(fused));
        }
        let mut evaluate = Vec::with_capacity(2);
        for &s in &live {
            if attempt(slots, &[s])?.is_none() {
                evaluate.push(s);
            }
        }
        if evaluate.is_empty() {
            evaluate = live;
        }
        for s in evaluate {
            if let Some(node) = slots[s].take() {
                let storage = node.force().as_ref().map_err(|error| Failure::Inherited(Arc::clone(error)))?;
                ready[s] = Some((storage.pin_single()?, storage.byte_len()));
            }
        }
    }
}

/// What a kernel that reads every element of a product operand many times would recompute at each
/// load: `node` itself if it is a pending expression, what it reads if it is a view (which stays
/// fused: a transposed view of a stored array costs a few index operations). `None` for a view of a
/// view, an array that is stored, or an expression of one buffer (the labels of a one-hot matrix),
/// which costs about a load per evaluation, as much as reading it stored. A reduction that is not
/// recorded yet (the mean and deviation of `(x - mean) / std`) is a buffer read once recorded, so it
/// counts like one.
fn recomputed(node: &Arc<Node>) -> Option<Arc<Node>> {
    if !node.is_pending() {
        return None;
    }
    let target = if node.is_pending_view() {
        match node.operands() {
            Some([Src::Node(inner), _, _]) if inner.is_pending() && !matches!(inner.op().kind, OpKind::Shift(ShiftKind::View { .. })) => inner,
            _ => return None,
        }
    } else {
        Arc::clone(node)
    };
    let deferred = unresolved_leaves(&target).map_or(0, |leaves| leaves.len());
    (target.inputs().len() + deferred >= 2).then_some(target)
}

/// Whether a product operand is an expression of several buffers that a kernel reading each of its
/// elements many times would compute at every load (see [`evaluate_for_reuse`]).
pub(crate) fn is_recomputed(node: &Arc<Node>) -> bool {
    recomputed(node).is_some()
}

/// Evaluates what a product operand computes, so that a kernel that reads every element many times
/// (a skinny product, see `matmul::skinny`) loads it from a buffer instead of recomputing it at
/// each load (see [`recomputed`]). Returns whether anything was evaluated.
pub(crate) fn evaluate_for_reuse(node: &Arc<Node>) -> Result<bool, Failure> {
    let Some(target) = recomputed(node) else { return Ok(false) };
    target.force().as_ref().map_err(|error| Failure::Inherited(Arc::clone(error)))?;
    Ok(true)
}

/// Records the product of `sides` (`[lhs, rhs]`) into the buffer `out`: one launch of the GEMM
/// kernel of `kernel` over `groups` workgroups. `words` are `m, n, k, kc` (see `matmul::body`).
/// Operands that are lazy expressions are fused into the kernel when they can be (see
/// [`fusable`]); otherwise they are evaluated and the kernel reads their buffers.
pub(crate) fn matmul(
    ctx: &crate::context::Context,
    sides: [MatmulSide; 2],
    kernel: MatmulKernel,
    words: [u32; 4],
    groups: [u32; 3],
    out: RawArg<'_>,
) -> Result<(), Failure> {
    let mut slots: [Option<Arc<Node>>; 2] = [None, None];
    let mut ready: [Option<(Pin, u64)>; 2] = [None, None];
    for (s, side) in sides.into_iter().enumerate() {
        match side {
            MatmulSide::Ready(pin, bytes) => ready[s] = Some((pin, bytes)),
            MatmulSide::Pending(node) => slots[s] = Some(node),
        }
    }
    let size = [words[1], words[0], 1];
    // `lhs` is `[K, M]`, `rhs` `[N, K]`.
    let dims = [[words[2], words[0]], [words[1], words[2]]];
    let transposed = [0, 1].map(|s| slots[s].as_ref().is_some_and(|node| stored_transposed(node, dims[s])));
    let fused = flatten_operands(ctx, kernel, transposed, &mut slots, &mut ready)?;
    let Some(fused) = fused else {
        let core = ctx.builtin(Builtin::Matmul { tile: kernel.tile, aligned: kernel.aligned, split: kernel.split }, "f32")?;
        let (Some((a, a_bytes)), Some((b, b_bytes))) = (&ready[0], &ready[1]) else {
            return Err(Failure::New(LumaError::Gpu("matrix product operand neither evaluated nor fused".into())));
        };
        ctx.launch(&core, groups, size, &[out, RawArg::read(a.buffer(), *a_bytes), RawArg::read(b.buffer(), *b_bytes), RawArg::Uniform(bytemuck::cast_slice(&words))])?;
        return Ok(());
    };
    fused.mark_fused();
    let Fused { mut key, inputs, mut scalars, fused_nodes, instr_nodes, sub_roots } = fused;
    drop((fused_nodes, instr_nodes, sub_roots));
    let core = ctx.jit().kernel(ctx, &mut key, &mut scalars)?;
    let pins = pin_inputs(&key, &inputs, None, 0, 0)?;
    let mut uniform = Vec::with_capacity(words.len() + scalars.len());
    uniform.extend_from_slice(&words);
    uniform.extend_from_slice(&scalars);
    let mut raw: Vec<RawArg<'_>> = Vec::with_capacity(pins.len() + 4);
    raw.push(out);
    raw.extend(pins.iter().map(InPin::raw));
    raw.extend(ready.iter().flatten().map(|(pin, bytes)| RawArg::read(pin.buffer(), *bytes)));
    raw.push(RawArg::Uniform(bytemuck::cast_slice(&uniform)));
    ctx.launch(&core, groups, size, &raw)?;
    Ok(())
}

/// Runs the reducing pass `pass` over the cells of the expression behind `node`, computed
/// inside the kernel (the program becomes the load `luma_eval(q)`, see `scan::reduce_body`):
/// `geometry` are the first fields of the uniform (the scalars of the expression follow),
/// `outs` the buffers it writes. Returns `false`, having launched nothing, when the expression
/// cannot be fused (see [`fusable`]); the caller then reduces its evaluated buffer. `reads` are buffers
/// the pass reads besides the expression's (bound after them, before the uniform).
#[allow(clippy::too_many_arguments)]
pub(crate) fn axis_reduce(
    ctx: &crate::context::Context,
    node: &Arc<Node>,
    pass: crate::scan::AxisPass,
    geometry: [u32; 5],
    workgroups: [u32; 3],
    size: [u32; 3],
    outs: &[RawArg<'_>],
    reads: &[RawArg<'_>],
) -> Result<bool, Failure> {
    let Some(args) = node.operands() else { return Ok(false) };
    let (fused, args) = flatten_bounded(node, args, Tail::AxisReduce(pass))?;
    drop(args);
    if !fusable(ctx, &fused, outs.len() + reads.len()) {
        return Ok(false);
    }
    fused.mark_fused();
    let Fused { mut key, inputs, mut scalars, fused_nodes, instr_nodes, sub_roots } = fused;
    drop((fused_nodes, instr_nodes, sub_roots));
    let core = ctx.jit().kernel(ctx, &mut key, &mut scalars)?;
    let pins = pin_inputs(&key, &inputs, None, 0, 0)?;
    let mut uniform = Vec::with_capacity(geometry.len() + scalars.len());
    uniform.extend_from_slice(&geometry);
    uniform.extend_from_slice(&scalars);
    let mut raw: Vec<RawArg<'_>> = Vec::with_capacity(outs.len() + pins.len() + reads.len() + 1);
    raw.extend_from_slice(outs);
    raw.extend(pins.iter().map(InPin::raw));
    raw.extend_from_slice(reads);
    raw.push(RawArg::Uniform(bytemuck::cast_slice(&uniform)));
    ctx.launch(&core, workgroups, size, &raw)?;
    ctx.jit().stats.reduce_launches.fetch_add(1, Ordering::Relaxed);
    Ok(true)
}

/// The first pass of axis reductions of several operands (pending nodes of one length, `f32`) in one program that
/// computes them together, once what they share (see [`axis_plan`]).
pub(crate) struct AxisPlan {
    /// Taken by the launch.
    fused: std::cell::RefCell<Option<Fused>>,
    /// The program's outputs: the distinct roots.
    outputs: Vec<u32>,
    /// The root every operand reduces.
    roots: Vec<u8>,
}

impl AxisPlan {
    /// The root (output of the program) that operand `k` reduces.
    pub(crate) fn roots(&self) -> &[u8] {
        &self.roots
    }
}

/// Plans the first pass that reduces every operand of `operands` along an axis, with `outs` buffers bound for its results
/// (the passes after it read what it writes): `None` if the program does not fit one kernel (see [`fusable`]), or an
/// operand is not an `f32` expression that is still pending.
pub(crate) fn axis_plan(ctx: &crate::context::Context, operands: &[&Arc<Node>], outs: usize) -> Option<AxisPlan> {
    // One root per distinct operand.
    let mut distinct: Vec<&Arc<Node>> = Vec::with_capacity(operands.len());
    let mut operand_root = Vec::with_capacity(operands.len());
    for node in operands {
        if node.op().out_ty() != ElemTy::F32 {
            return None;
        }
        let at = match distinct.iter().position(|known| Arc::ptr_eq(known, node)) {
            Some(at) => at,
            None => {
                distinct.push(node);
                distinct.len() - 1
            }
        };
        operand_root.push(at);
    }
    let mut roots = Vec::with_capacity(distinct.len());
    for node in &distinct {
        roots.push((Arc::clone(node), node.operands()?));
    }
    let placeholder = crate::scan::AxisPass {
        pass: crate::scan::Pass::Fold,
        ops: ReduceList::new(&[ReduceOp::Sum]),
        partial: false,
        split: false,
        line_major: false,
    };
    let fused = flatten_inner(&roots, Tail::AxisReduce(placeholder), &[]).ok()?;
    drop(roots);
    let fits = fused.key.outputs.len() == distinct.len()
        && fused.key.outputs.iter().all(|&at| fused.key.instrs.get(at as usize).is_some_and(|instr| instr.op.out_ty() == ElemTy::F32))
        && (fused.key.subs.is_empty() || (fused.key.inlines() && fused.key.staging.is_none()))
        && fused.key.narrow()
        && fusable(ctx, &fused, outs);
    if !fits {
        return None;
    }
    // Equal values are one root.
    let mut outputs: Vec<u32> = Vec::with_capacity(distinct.len());
    let mut slot_of_root = Vec::with_capacity(distinct.len());
    for &at in &fused.key.outputs {
        let slot = match outputs.iter().position(|&known| known == at) {
            Some(slot) => slot,
            None => {
                outputs.push(at);
                outputs.len() - 1
            }
        };
        slot_of_root.push(slot);
    }
    let roots = operand_root.iter().map(|&root| u8::try_from(slot_of_root[root]).ok()).collect::<Option<Vec<u8>>>()?;
    Some(AxisPlan { fused: std::cell::RefCell::new(Some(fused)), outputs, roots })
}

/// Runs the reducing pass `pass` (its ops reduce the roots of `plan`, see [`ReduceList::merged`]) over the cells of the
/// planned program, as [`axis_reduce`] does for one expression.
pub(crate) fn axis_reduce_planned(
    ctx: &crate::context::Context,
    plan: &AxisPlan,
    pass: crate::scan::AxisPass,
    geometry: [u32; 5],
    workgroups: [u32; 3],
    size: [u32; 3],
    outs: &[RawArg<'_>],
) -> Result<(), Failure> {
    let Some(fused) = plan.fused.borrow_mut().take() else {
        return Err(Failure::New(LumaError::Gpu("an axis reduction was planned once and launched twice".into())));
    };
    fused.mark_fused();
    let Fused { mut key, inputs, mut scalars, fused_nodes, instr_nodes, sub_roots } = fused;
    drop((fused_nodes, instr_nodes, sub_roots));
    key.outputs.clone_from(&plan.outputs);
    key.tail = Tail::AxisReduce(pass);
    let core = ctx.jit().kernel(ctx, &mut key, &mut scalars)?;
    let pins = pin_inputs(&key, &inputs, None, 0, 0)?;
    let mut uniform = Vec::with_capacity(geometry.len() + scalars.len());
    uniform.extend_from_slice(&geometry);
    uniform.extend_from_slice(&scalars);
    let mut raw: Vec<RawArg<'_>> = Vec::with_capacity(outs.len() + pins.len() + 1);
    raw.extend_from_slice(outs);
    raw.extend(pins.iter().map(InPin::raw));
    raw.push(RawArg::Uniform(bytemuck::cast_slice(&uniform)));
    ctx.launch(&core, workgroups, size, &raw)?;
    ctx.jit().stats.reduce_launches.fetch_add(1, Ordering::Relaxed);
    Ok(())
}

/// Records one kernel of the first radix sort pass over the expression behind `node`, computing
/// the keys in the kernel instead of reading a buffer: the kernel of `pass` (counting or
/// scattering) launched on `groups` workgroups with the uniform `words` (see `sort::PARAMS`)
/// followed by the expression's scalars. Binds `writes`, the buffers of the expression, then
/// `reads`. Returns `false`, having recorded nothing, if the node has been evaluated already or
/// the expression cannot be fused (see [`fusable`]): the caller evaluates it and runs the kernel
/// over its buffer. `bindings` is how many buffers the kernels of the whole pass bind in the
/// worst case, so that both kernels fuse or neither does.
#[allow(clippy::too_many_arguments)]
pub(crate) fn radix_pass(
    ctx: &crate::context::Context,
    node: &Arc<Node>,
    pass: RadixPass,
    words: [u32; 3],
    groups: [u32; 3],
    writes: &[RawArg<'_>],
    reads: &[RawArg<'_>],
    bindings: usize,
) -> Result<bool, Failure> {
    if !node.is_pending() {
        return Ok(false);
    }
    let Some(args) = node.operands() else { return Ok(false) };
    let (fused, args) = flatten_bounded(node, args, Tail::Radix(pass))?;
    drop(args);
    if !fusable(ctx, &fused, bindings) {
        return Ok(false);
    }
    fused.mark_fused();
    let Fused { mut key, inputs, mut scalars, fused_nodes, instr_nodes, sub_roots } = fused;
    drop((fused_nodes, instr_nodes, sub_roots));
    let core = ctx.jit().kernel(ctx, &mut key, &mut scalars)?;
    let pins = pin_inputs(&key, &inputs, None, 0, 0)?;
    let mut uniform = Vec::with_capacity(words.len() + scalars.len());
    uniform.extend_from_slice(&words);
    uniform.extend_from_slice(&scalars);
    let mut raw: Vec<RawArg<'_>> = Vec::with_capacity(pins.len() + writes.len() + reads.len() + 1);
    raw.extend_from_slice(writes);
    raw.extend(pins.iter().map(InPin::raw));
    raw.extend_from_slice(reads);
    raw.push(RawArg::Uniform(bytemuck::cast_slice(&uniform)));
    ctx.launch(&core, groups, [words[0], 1, 1], &raw)?;
    Ok(true)
}

/// One part of a line solve (see `solve::SolveSpec`).
pub(crate) enum PartSide {
    /// An evaluated array: a pin on its buffer and the bytes of it the array holds.
    Ready(Pin, u64),
    /// A lazy expression, fused into the kernel unless it cannot be (see [`fusable`]).
    Pending(Arc<Node>),
}

/// Records the line solve of `spec` over `parts` (in the order of its parts): one launch of
/// `groups` workgroups, with `writes` (its outputs and scratch) bound first, then the buffers of
/// the parts, `starts` (if the spec has them) and the uniform `words` (`n, stride, lines`)
/// followed by the scalars of the fused parts. Parts that are lazy expressions are computed in the
/// kernel, one program for all of them. Returns `false`, having launched nothing, when they
/// cannot be (see [`fusable`], or a part that has been evaluated meanwhile): the caller then
/// solves their evaluated buffers.
#[allow(clippy::too_many_arguments)]
pub(crate) fn solve(
    ctx: &crate::context::Context,
    parts: Vec<PartSide>,
    mut spec: SolveSpec,
    words: [u32; 3],
    groups: [u32; 3],
    size: [u32; 3],
    writes: &[RawArg<'_>],
    starts: Option<RawArg<'_>>,
) -> Result<bool, Failure> {
    let mut roots = Vec::new();
    let mut ready: Vec<(Pin, u64)> = Vec::new();
    for (part, side) in parts.into_iter().enumerate() {
        match side {
            PartSide::Ready(pin, bytes) => ready.push((pin, bytes)),
            PartSide::Pending(node) => {
                let Some(args) = node.operands() else { return Ok(false) };
                spec.fuse(part);
                roots.push((node, args));
            }
        }
    }
    let buffers = writes.len() + ready.len() + usize::from(starts.is_some());
    if roots.is_empty() {
        if buffers > ctx.jit().max_inputs() {
            return Ok(false);
        }
        let core = ctx.builtin(Builtin::SolveParts(spec), "f32")?;
        let mut raw: Vec<RawArg<'_>> = Vec::with_capacity(writes.len() + ready.len() + 2);
        raw.extend_from_slice(writes);
        raw.extend(ready.iter().map(|(pin, bytes)| RawArg::read(pin.buffer(), *bytes)));
        raw.extend(starts);
        raw.push(RawArg::Uniform(bytemuck::cast_slice(&words)));
        ctx.launch(&core, groups, size, &raw)?;
        return Ok(true);
    }
    let fused = flatten(&roots, Tail::Solve(spec))?;
    drop(roots);
    if !fusable(ctx, &fused, buffers) {
        return Ok(false);
    }
    fused.mark_fused();
    let Fused { mut key, inputs, mut scalars, fused_nodes, instr_nodes, sub_roots } = fused;
    drop((fused_nodes, instr_nodes, sub_roots));
    let core = ctx.jit().kernel(ctx, &mut key, &mut scalars)?;
    let pins = pin_inputs(&key, &inputs, None, 0, 0)?;
    let mut uniform = Vec::with_capacity(words.len() + scalars.len());
    uniform.extend_from_slice(&words);
    uniform.extend_from_slice(&scalars);
    let mut raw: Vec<RawArg<'_>> = Vec::with_capacity(writes.len() + pins.len() + ready.len() + 2);
    raw.extend_from_slice(writes);
    raw.extend(pins.iter().map(InPin::raw));
    raw.extend(ready.iter().map(|(pin, bytes)| RawArg::read(pin.buffer(), *bytes)));
    raw.extend(starts);
    raw.push(RawArg::Uniform(bytemuck::cast_slice(&uniform)));
    ctx.launch(&core, groups, size, &raw)?;
    Ok(true)
}
