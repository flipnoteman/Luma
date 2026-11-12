//! Running flattened programs: evaluation, in-place and shifted outputs, register loops and
//! the first pass of reductions.

use std::borrow::Cow;
use std::sync::atomic::Ordering;
use std::sync::Arc;

use crate::array::{Failure, Storage};
use crate::gpu::pager::Pin;
use crate::builtin::{Builtin, ReduceList, ReduceOp};
use crate::context::{RawArg, Scratch};
use crate::error::LumaError;
use crate::ops::{fold_workgroups, launch_1d};

use super::codegen::direct;
use super::{lock, Args, Edge, ElemTy, Node, OpKind, ShiftKind, Src, ViewMap};
use crate::profile::SplitReason;
use super::program::{flatten, flatten_inner, Fused, FusedKey, Ref, Tail};

/// Runs `steps` iterations of a traced step in one kernel that keeps the state in registers.
/// `outs[j]` is the next value of state `j`, computed from `probes[j]` (aliases of `inits[j]`).
/// `None` if the step is not a single-buffer elementwise program of `f32` values; the caller
/// then runs the steps one by one.
pub(crate) fn run_loop(
    ctx: &crate::context::Context,
    probes: &[Arc<Storage>],
    inits: &[Arc<Storage>],
    outs: &[Arc<Node>],
    steps: u32,
) -> Option<Result<Vec<Arc<Storage>>, Failure>> {
    let len = outs.first()?.len();
    if outs.iter().any(|out| out.len() != len || !out.is_pending())
        || outs.iter().enumerate().any(|(j, out)| outs[..j].iter().any(|other| Arc::ptr_eq(other, out)))
        || probes.iter().any(|probe| probe.was_materialized())
        || probes.iter().chain(inits).any(|storage| storage.segments() != 1)
        || inits.iter().any(|init| init.elem() != 4 || init.len() != len)
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
    run_loop_fused(ctx, &fused, &states, inits, steps)
}

/// Runs `steps` iterations of the store program `fused` (one output per state, `outputs[j]` the
/// next value of state `j`) in one kernel that keeps the states in registers. `states[k]` is the
/// state that input `k` reads (`None`: an input that stays fixed) and `inits` the states before
/// the first step. `None` if the program is not a single-buffer elementwise program of `f32`
/// values.
pub(super) fn run_loop_fused(
    ctx: &crate::context::Context,
    fused: &Fused,
    states: &[Option<usize>],
    inits: &[Arc<Storage>],
    steps: u32,
) -> Option<Result<Vec<Arc<Storage>>, Failure>> {
    let key = &fused.key;
    let len = inits.first()?.len();
    let shifts = key.instrs.iter().any(|instr| matches!(instr.op.kind, OpKind::Shift(_)));
    let all_f32 = key.outputs.iter().all(|&at| key.instrs.get(at as usize).is_some_and(|i| i.op.out_ty() == ElemTy::F32));
    if shifts || !key.subs.is_empty() || !all_f32 || !key.narrow() || fused.inputs.iter().any(|input| input.segments() != 1) {
        return None;
    }
    let bindings = 2 * key.outputs.len() + states.iter().filter(|state| state.is_none()).count();
    if bindings > ctx.jit().max_inputs() + 1 {
        return None;
    }
    let budget = ctx.jit().loop_work.load(Ordering::Relaxed);
    let per_launch = budget / (len as u64 * key.instrs.len().max(1) as u64).max(1);
    let per_launch = u32::try_from(per_launch).unwrap_or(u32::MAX).max(1);
    Some((|| {
        let core = ctx.jit().loop_kernel(ctx, key, states)?;
        let mut states_now: Vec<Arc<Storage>> = inits.to_vec();
        let mut remaining = steps;
        while remaining > 0 {
            let count = remaining.min(per_launch);
            let outputs = launch_loop(ctx, &core, fused, states, &states_now, len, count)?;
            states_now = outputs.into_iter().map(Arc::new).collect();
            remaining -= count;
        }
        Ok(states_now)
    })())
}

/// Most element-instructions (elements x instructions x steps) one launch of a register loop
/// runs: about 9 ms on an RTX 3080 Laptop, so a GPU 100 times slower still stays below the
/// 2 s watchdog of Windows. Longer loops are split into several launches.
pub(super) const LOOP_WORK_PER_LAUNCH: u64 = 1 << 36;

fn launch_loop(
    ctx: &crate::context::Context,
    core: &crate::kernel::KernelCore,
    fused: &Fused,
    states: &[Option<usize>],
    inits: &[Arc<Storage>],
    len: usize,
    steps: u32,
) -> Result<Vec<Storage>, Failure> {
    let n = len_u32(len)?;
    let bytes = u64::from(n) * 4;
    let outputs: Vec<Storage> = inits.iter().map(|_| Storage::new(ctx, len, 4)).collect();
    let out_pins = outputs.iter().map(|out| out.pin(0)).collect::<Result<Vec<_>, _>>()?;
    let init_pins = inits.iter().map(|init| init.pin(0)).collect::<Result<Vec<_>, _>>()?;
    let in_pins = fused
        .inputs
        .iter()
        .zip(states)
        .filter(|(_, state)| state.is_none())
        .map(|(input, _)| input.pin(0).map(|pin| (pin, input.byte_len())))
        .collect::<Result<Vec<_>, _>>()?;
    let mut words = Vec::with_capacity(fused.scalars.len() + 1);
    words.push(steps);
    words.extend_from_slice(&fused.scalars);
    let mut raw: Vec<RawArg<'_>> = Vec::with_capacity(out_pins.len() * 2 + in_pins.len() + 1);
    raw.extend(out_pins.iter().map(|pin| RawArg::write(pin.buffer(), bytes)));
    raw.extend(init_pins.iter().map(|pin| RawArg::read(pin.buffer(), bytes)));
    raw.extend(in_pins.iter().map(|(pin, bytes)| RawArg::read(pin.buffer(), *bytes)));
    raw.push(RawArg::Uniform(bytemuck::cast_slice(&words)));
    launch_1d(ctx, core, n, &raw)?;
    ctx.jit().stats.elementwise_launches.fetch_add(1, Ordering::Relaxed);
    Ok(outputs)
}

pub(super) fn len_u32(len: usize) -> Result<u32, LumaError> {
    u32::try_from(len).map_err(|_| LumaError::InvalidShape(format!("{len} elements exceed u32")))
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
    loop {
        let roots = [(Arc::clone(node), args)];
        let mut fused = flatten(&roots, tail)?;
        let [(_, returned)] = roots;
        args = returned;
        if fused.inputs.len() > max_inputs
            && let Some(i) = (0..node.op().kind.arity()).find(|&i| matches!(args.get(i), Some(Src::Node(_))))
        {
            jit.note_split(node.name, node.location, SplitReason::Inputs { inputs: fused.inputs.len(), limit: max_inputs });
            drop(fused);
            super::evaluate_operand(&mut args, i)?;
            continue;
        }
        if rounds < LIVE_ROUNDS
            && let Some((outs, live)) = live_split(&fused, max_inputs, max_live)
        {
            rounds += 1;
            jit.note_split(node.name, node.location, SplitReason::Live { live, limit: max_live });
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
fn last_uses(key: &FusedKey) -> Vec<usize> {
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
fn live_profile(key: &FusedKey) -> Vec<u32> {
    let n = key.instrs.len();
    let last = last_uses(key);
    let loaded = (0..key.input_tys.len() as u32).filter(|&k| direct(&key.instrs, k)).count() as u32;
    let mut dying = vec![0u32; n];
    let mut live = Vec::with_capacity(n);
    let mut now = 0u32;
    for k in 0..n {
        now -= dying[k];
        if last[k] > k {
            now += 1;
            if let Some(slot) = dying.get_mut(last[k]) {
                *slot += 1;
            }
        }
        live.push(now + loaded);
    }
    live
}

/// If the main program of `fused` keeps more than `max_live` values live at once, the nodes to
/// evaluate first so that the program splits where fewest values are live, and the peak. The cut
/// is made in the middle half of the program; nothing is returned if no point there is better
/// than the peak, the values crossing it are not all nodes, or there are more of them than one
/// kernel can output.
fn live_split(fused: &Fused, max_inputs: usize, max_live: usize) -> Option<(Vec<Arc<Node>>, usize)> {
    let key = &fused.key;
    let n = key.instrs.len();
    if fused.instr_nodes.len() != n {
        return None;
    }
    let live = live_profile(key);
    let peak = live.iter().copied().max()? as usize;
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
    let last = last_uses(key);
    let mut outs = Vec::new();
    for j in 0..=cut {
        if last[j] > cut {
            outs.push(Arc::clone(fused.instr_nodes[j].as_ref()?));
        }
    }
    (!outs.is_empty() && outs.len() <= max_inputs / 2).then_some((outs, peak))
}

/// Evaluates the pending ones of `nodes` (all of the same length) in one fused kernel with one
/// output per node. Failures are stored in the nodes, as [`Node::force`] does.
pub(crate) fn evaluate_group(nodes: &[Arc<Node>]) {
    // A loop leaf has no operands to fuse: running its plan is its evaluation.
    for node in nodes.iter().filter(|node| node.is_pending() && node.looped.get().is_some()) {
        let _ = node.force();
    }
    let roots: Vec<(Arc<Node>, Args)> = nodes
        .iter()
        .filter(|node| node.is_pending())
        .filter_map(|node| node.operands().map(|args| (Arc::clone(node), args)))
        .collect();
    // A root that another root reads through views would be recomputed at every index it is read
    // at, and the chain of such steps grows past what a kernel takes: it goes first, alone.
    let reads_a_root = |node: &Arc<Node>| {
        node.shift_depth() > 0 && roots.iter().any(|(other, _)| !Arc::ptr_eq(node, other) && node.below_contains(other.id))
    };
    if roots.iter().any(|(node, _)| reads_a_root(node)) {
        let (later, first): (Vec<_>, Vec<_>) = roots.iter().map(|(node, _)| Arc::clone(node)).partition(reads_a_root);
        evaluate_group(&first);
        evaluate_group(&later);
        return;
    }
    let grouped = roots.len() > 1
        && roots.iter().all(|(node, _)| node.len() == roots[0].0.len())
        && match flatten(&roots, Tail::Store) {
            Ok(fused) if fused.inputs.len() + roots.len() <= roots[0].0.context().jit().max_inputs() + 1 => {
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
            _ => false,
        };
    if !grouped {
        // One kernel each; this also reports every failure at its own node.
        for (node, _) in &roots {
            let _ = node.force();
        }
    }
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
fn input_at<'a>(key: &FusedKey, inputs: &'a [Arc<Storage>], k: usize) -> Option<&'a Arc<Storage>> {
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
    let widest = inputs.iter().map(|input| input.elem() as u64).chain(outs.iter().copied()).max().map_or(layout, |w| w.max(layout));
    let mixed = inputs.iter().any(|input| input.elem() as u64 != layout) || outs.iter().any(|&bytes| bytes != layout);
    if mixed && (len as u64 * widest > ctx.segment_bytes() || inputs.iter().any(|input| input.segments() > 1)) {
        return Err(LumaError::Chunked("mixed element sizes on arrays split over several buffers"));
    }
    let split = key.instrs.iter().chain(key.subs.iter().flatten()).any(|instr| {
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

/// Bytes per element of every output of `key`: one per stored value, or the one value a
/// reduction folds.
fn out_elems(key: &FusedKey) -> Vec<u64> {
    let bytes = |at: u32| key.instrs.get(at as usize).map_or(4, |instr| instr.op.out_ty().bytes());
    match key.tail {
        Tail::Store | Tail::StoreMapped(_) => key.outputs.iter().map(|&at| bytes(at)).collect(),
        Tail::Scatter { .. } => key.outputs.iter().take(1).map(|&at| bytes(at)).collect(),
        Tail::Reduce(_) => vec![key.instrs.len().checked_sub(1).map_or(4, |at| bytes(at as u32))],
    }
}

/// Bytes per element of the arrays whose segmentation the launch follows: the outputs of a
/// store, the inputs of the length `len` of a reduction (a reduction writes only partial
/// results, so what it folds does not matter).
fn layout_elem(key: &FusedKey, inputs: &[Arc<Storage>], len: usize) -> u64 {
    let outs = out_elems(key);
    match key.tail {
        Tail::Reduce(_) => inputs.iter().find(|input| input.len() == len).map_or_else(|| outs.first().copied().unwrap_or(4), |input| input.elem() as u64),
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
            ShiftKind::Take { .. } => Reach::All,
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
        Tail::Store | Tail::StoreMapped(_) => key.outputs.len(),
        Tail::Scatter { .. } | Tail::Reduce(_) => 1,
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
    for axis in 0..4 {
        lo += boxed[axis].0 as usize * step;
        hi += boxed[axis].1 as usize * step;
        step *= map.src[axis] as usize;
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
) -> Result<Vec<(Pin, u64)>, LumaError> {
    let bytes = |input: &Storage| u64::from(n) * input.elem() as u64;
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
                (input.pin(j)?, input.range(j).len() as u64 * input.elem() as u64)
            } else {
                (input.pin(0)?, input.byte_len())
            });
            continue;
        };
        if direct(&key.instrs, k as u32) {
            pins.push((input.pin(j)?, bytes(input)));
        }
        for &id in ids.get(k).map_or(&[][..], Vec::as_slice) {
            let id = id as usize;
            pins.push((input.pin(id)?, input.range(id).len() as u64 * input.elem() as u64));
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
    scalars: &'a [u32],
) -> Result<Cow<'a, [u32]>, LumaError> {
    let based = ids.is_none() && super::codegen::generates(key);
    if ids.is_none() && n.is_none() && !based {
        return Ok(Cow::Borrowed(scalars));
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
    words.extend_from_slice(scalars);
    Ok(Cow::Owned(words))
}

/// What a launch of the store kernel `key` binds for segment `j` of `outputs[0]`: `outputs`, then
/// the inputs as reads (see [`pin_inputs`]), then the uniform (see [`params`]).
pub(super) struct StoreArgs<'a> {
    pub(super) n: u32,
    outs: Vec<(Pin, u64)>,
    ins: Vec<(Pin, u64)>,
    words: Cow<'a, [u32]>,
}

impl StoreArgs<'_> {
    pub(super) fn raw(&self) -> Vec<RawArg<'_>> {
        let mut raw: Vec<RawArg<'_>> = Vec::with_capacity(self.outs.len() + self.ins.len() + 1);
        raw.extend(self.outs.iter().map(|(pin, bytes)| RawArg::write(pin.buffer(), *bytes)));
        raw.extend(self.ins.iter().map(|(pin, bytes)| RawArg::read(pin.buffer(), *bytes)));
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
        .map(|out| out.pin(j).map(|pin| (pin, range.len() as u64 * out.elem() as u64)))
        .collect::<Result<Vec<_>, _>>()?;
    let ins = pin_inputs(key, inputs, ids, j, n)?;
    let words = params(key, ids, None, range.start, shifted.map_or(0, |shifted| shifted.len), scalars)?;
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
        (Some(tile), _) => Some(tile.dims.0.div_ceil(tile.tile.0) * tile.dims.1.div_ceil(tile.tile.1)),
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
            launch_1d(ctx, core, args.n, &args.raw())?;
        }
    }
    ctx.jit().stats.elementwise_launches.fetch_add(1, Ordering::Relaxed);
    Ok(())
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
    fused.mark_fused();
    let Fused { mut key, mut inputs, scalars, fused_nodes, instr_nodes, sub_roots } = fused;
    // The fused nodes hold their operands; a dying input is only uniquely owned without them.
    drop((fused_nodes, instr_nodes, sub_roots));
    let count = key.outputs.len();

    let mut reused = None;
    if count == 1 {
        let candidate = (0..inputs.len()).find(|&k| {
            let input = &inputs[k];
            in_place_eligible(&key, k as u32)
                && Arc::strong_count(input) == 1
                && !input.is_alias()
        });
        if let Some(k) = candidate {
            match Arc::try_unwrap(inputs.remove(k)) {
                Ok(storage) => {
                    ctx.jit().note_write();
                    key.in_place = vec![(k as u32, 0)];
                    reused = Some(storage);
                }
                Err(shared) => inputs.insert(k, shared),
            }
        }
    }
    let shifted = plan_shifts(ctx, &mut key, &inputs, &scalars, len)?;
    settle_staging(&mut key, len);
    if shifted.is_none() && key.staging.is_none() {
        key.tile = super::codegen::plan_tile(&key, &scalars, len, ctx.tuning());
    }
    let core = ctx.jit().kernel(ctx, &key)?;

    let outputs = match reused {
        Some(storage) => vec![storage],
        None => {
            let mut outputs = Vec::with_capacity(count);
            for bytes in out_elems(&key) {
                outputs.push(Storage::new(ctx, len, bytes as usize));
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
    let Fused { mut key, mut inputs, scalars, fused_nodes, instr_nodes, sub_roots } = fused;
    drop((fused_nodes, instr_nodes, sub_roots));
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
    let core = ctx.jit().kernel(ctx, &key)?;
    launch_store(ctx, &core, &key, &[&**target], &inputs, &scalars, shifted.as_ref())?;
    check_finite(ctx, target, node.op().out_ty())?;
    // Another thread cannot have evaluated this node: its operands were taken above.
    let _ = node.result.set(Ok(Arc::clone(target)));
    Ok(())
}

/// How many storage buffers a launch of [`run_mapped`] binds: the `bases`, then the inputs that
/// are not one of them.
pub(super) fn mapped_bindings(inputs: &[Arc<Storage>], bases: &[Arc<Storage>]) -> usize {
    bases.len() + inputs.iter().filter(|input| !bases.iter().any(|base| Arc::ptr_eq(base, input))).count()
}

/// Runs the program `fused` (tail [`Tail::StoreMapped`] or [`Tail::Scatter`]) over `n` elements,
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
    let Fused { mut key, mut inputs, scalars, fused_nodes, instr_nodes, sub_roots } = fused;
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
    let core = ctx.jit().kernel(ctx, &key)?;
    let outs = bases.iter().map(|base| base.pin(0)).collect::<Result<Vec<_>, _>>()?;
    let ins = pin_inputs(&key, &inputs, None, 0, n)?;
    let words = params(&key, None, None, 0, n as usize, &scalars)?;
    let mut raw: Vec<RawArg<'_>> = Vec::with_capacity(outs.len() + ins.len() + 1);
    raw.extend(outs.iter().zip(bases).map(|(pin, base)| RawArg::write(pin.buffer(), base.byte_len())));
    raw.extend(ins.iter().map(|(pin, bytes)| RawArg::read(pin.buffer(), *bytes)));
    if !words.is_empty() {
        raw.push(RawArg::Uniform(bytemuck::cast_slice(&words)));
    }
    launch_1d(ctx, &core, n, &raw)?;
    ctx.jit().stats.elementwise_launches.fetch_add(1, Ordering::Relaxed);
    Ok(())
}

/// Runs the first pass of the reductions `ops` directly over the expression behind `node`,
/// without materializing it: every workgroup evaluates its strided blocks of `Tuning::reduce_block` elements and reduces them,
/// once per op. Returns the buffer of partial results (op `j` at `j * groups ..`) and how many
/// there are per op, or `None` if the node has already been evaluated or is better evaluated
/// first (reduce its buffer instead). With one segment and one group per op the partials are
/// the final values and are written into `target` when it is given.
pub(crate) fn reduce_first_pass(
    node: &Arc<Node>,
    ops: ReduceList,
    scratch: &mut Scratch<'_>,
    target: Option<&wgpu::Buffer>,
) -> Result<Option<Vec<(wgpu::Buffer, u32)>>, Failure> {
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
    reduce_fused(node.context(), node.len(), fused, ops, scratch, target).map(Some)
}

/// The first pass of [`reduce_first_pass`] for an already flattened expression of `len` elements.
pub(super) fn reduce_fused(
    ctx: &crate::context::Context,
    len: usize,
    fused: Fused,
    ops: ReduceList,
    scratch: &mut Scratch<'_>,
    target: Option<&wgpu::Buffer>,
) -> Result<Vec<(wgpu::Buffer, u32)>, Failure> {
    fused.mark_fused();
    let Fused { mut key, inputs, scalars, .. } = fused;
    // The segments are those an array of this length has: the inputs may be of other lengths.
    let layout = layout_elem(&key, &inputs, len);
    let first = Storage::new(ctx, len, layout as usize);
    let segments = first.segments();
    let shifted = plan_shifts(ctx, &mut key, &inputs, &scalars, len)?;
    let core = ctx.jit().kernel(ctx, &key)?;

    let mut partials = Vec::with_capacity(segments);
    for j in 0..segments {
        let range = first.range(j);
        let n = len_u32(range.len())?;
        let groups = ctx.tuning().reduce_groups(n);
        let out_bytes = u64::from(groups) * out_elems(&key).first().copied().unwrap_or(4) * ops.as_slice().len() as u64;
        let out = match target {
            Some(target) if segments == 1 && groups == 1 => target.clone(),
            _ => scratch.take(out_bytes)?,
        };
        let ids = segment_ids(shifted.as_ref(), j)?;
        let params = params(&key, ids, Some(n), range.start, len, &scalars)?;
        let pins = pin_inputs(&key, &inputs, ids, j, n)?;
        let mut raw: Vec<RawArg<'_>> = Vec::with_capacity(pins.len() + 2);
        raw.push(RawArg::write(&out, out_bytes));
        raw.extend(pins.iter().map(|(pin, bytes)| RawArg::read(pin.buffer(), *bytes)));
        raw.push(RawArg::Uniform(bytemuck::cast_slice(&params)));
        ctx.launch(&core, fold_workgroups(ctx, groups)?, [n, 1, 1], &raw)?;
        drop(raw);
        partials.push((out, groups));
    }
    ctx.jit().stats.reduce_launches.fetch_add(1, Ordering::Relaxed);
    Ok(partials)
}

/// The finite check (see `ContextOptions::check_finite`): fails with [`LumaError::NonFinite`] if
/// `storage`, holding elements of type `ty`, contains a NaN or an infinity. Does nothing unless
/// the option is on and the elements are `f32` or `Df32`. Waits for the GPU (once per segment).
pub(crate) fn check_finite(ctx: &crate::context::Context, storage: &Storage, ty: ElemTy) -> Result<(), LumaError> {
    if !ctx.check_finite() || !matches!(ty, ElemTy::F32 | ElemTy::Df32) {
        return Ok(());
    }
    let core = ctx.builtin(Builtin::NonFinite, "u32")?;
    let max = ReduceList::new(&[ReduceOp::Max]);
    let mut scratch = Scratch::new(ctx);
    for j in 0..storage.segments() {
        let bytes = storage.range(j).len() as u64 * storage.elem() as u64;
        let words = len_u32((bytes / 4) as usize)?;
        let pin = storage.pin(j)?;
        let flags = scratch.take(u64::from(words) * 4)?;
        launch_1d(ctx, &core, words, &[RawArg::write(&flags, u64::from(words) * 4), RawArg::read(pin.buffer(), bytes)])?;
        let result = crate::reduce::finish_passes(ctx, &mut scratch, max, "u32", 4, flags, words, false, None)?;
        if ctx.read::<u32>(&result, 4)?.first().is_some_and(|&flag| flag != 0) {
            return Err(LumaError::NonFinite);
        }
    }
    Ok(())
}
