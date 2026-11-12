//! Running flattened programs: evaluation, in-place and shifted outputs, register loops and
//! the first pass of reductions.

use std::borrow::Cow;
use std::sync::atomic::Ordering;
use std::sync::Arc;

use crate::array::{Failure, Storage};
use crate::gpu::pager::Pin;
use crate::builtin::ReduceList;
use crate::context::{RawArg, Scratch};
use crate::error::LumaError;
use crate::ops::{fold_workgroups, launch_1d};

use super::codegen::direct;
use super::{lock, Args, ElemTy, Node, OpKind, ShiftKind, Src};
use super::program::{flatten, flatten_inner, Fused, FusedKey, Ref, Tail, ELEMENT_BYTES};

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
        || inits.iter().any(|init| init.byte_len() != len as u64 * ELEMENT_BYTES)
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
    let len = usize::try_from(inits.first()?.byte_len() / ELEMENT_BYTES).ok()?;
    let shifts = key.instrs.iter().any(|instr| matches!(instr.op.kind, OpKind::Shift(_)));
    let all_f32 = key.outputs.iter().all(|&at| key.instrs.get(at as usize).is_some_and(|i| i.op.out_ty() == ElemTy::F32));
    if shifts || !key.subs.is_empty() || !all_f32 || fused.inputs.iter().any(|input| input.segments() != 1) {
        return None;
    }
    let bindings = 2 * key.outputs.len() + states.iter().filter(|state| state.is_none()).count();
    if bindings > ctx.jit().max_inputs + 1 {
        return None;
    }
    let source = super::codegen::loop_source(key, states, ctx.tuning());
    let label = format!("Loop{}", key.label().trim_start_matches("Fused"));
    let budget = ctx.jit().loop_work.load(Ordering::Relaxed);
    let per_launch = budget / (len as u64 * key.instrs.len().max(1) as u64).max(1);
    let per_launch = u32::try_from(per_launch).unwrap_or(u32::MAX).max(1);
    Some((|| {
        let core = ctx.jit().stencil_kernel(ctx, source, &label)?;
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
    let bytes = u64::from(n) * ELEMENT_BYTES;
    let outputs: Vec<Storage> = inits.iter().map(|_| Storage::new(ctx, len, ELEMENT_BYTES as usize)).collect();
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

/// Flattens the expression of `node` with operands `args`, which come back for the caller to drop.
/// Nodes inside it that were computed after it was measured are buffers now, so it can read more
/// buffers than the device binds; then its operands are computed first, one at a time.
fn flatten_bounded(node: &Arc<Node>, mut args: Args, tail: Tail) -> Result<(Fused, Args), Failure> {
    let max_inputs = node.context().jit().max_inputs;
    loop {
        let roots = [(Arc::clone(node), args)];
        let fused = flatten(&roots, tail)?;
        let [(_, returned)] = roots;
        args = returned;
        if fused.inputs.len() > max_inputs
            && let Some(i) = (0..node.op().kind.arity()).find(|&i| matches!(args.get(i), Some(Src::Node(_))))
        {
            drop(fused);
            super::evaluate_operand(&mut args, i)?;
            continue;
        }
        return Ok((fused, args));
    }
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
    let grouped = roots.len() > 1
        && roots.iter().all(|(node, _)| node.len() == roots[0].0.len())
        && match flatten(&roots, Tail::Store) {
            Ok(fused) if fused.inputs.len() + roots.len() <= roots[0].0.context().jit().max_inputs + 1 => {
                let (first, _) = &roots[0];
                match run_store(first.context(), first.len(), fused) {
                    Ok(outputs) => {
                        for ((node, _), storage) in roots.iter().zip(outputs) {
                            node.finish(storage);
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

/// Decides whether `key` needs the segmented variant on arrays of `len` elements and, if it does,
/// sets `key.spans` and returns the segments to bind. `None` for arrays that fit one segment and
/// programs without shifts: they run as plain per-segment launches.
fn plan_shifts(
    ctx: &crate::context::Context,
    key: &mut FusedKey,
    scalars: &[u32],
    len: usize,
) -> Result<Option<Shifted>, LumaError> {
    let c = (ctx.segment_bytes() / ELEMENT_BYTES) as usize;
    if len <= c {
        return Ok(None);
    }
    if !key.subs.is_empty() {
        return Err(LumaError::Gpu("segmented program with sub-programs".into()));
    }
    let Some((counts, ids)) = spans(key, scalars, len, c, ctx.jit().max_inputs + 1)? else {
        return Ok(None);
    };
    key.spans = counts;
    Ok(Some(Shifted { len, ids }))
}

/// Slot counts per input and, per output segment and input, the segment ids to bind (see [`spans`]).
type SegmentSpans = (Vec<u8>, Vec<Vec<Vec<u32>>>);

/// For a program with shifts on arrays of `len` elements in segments of `c`: how many segments
/// of every input its shifted reads touch at most (slots, 0 = the input is never shifted) and,
/// per output segment, the ids of the segments to bind in those slots (`ids[j][k]`, padded to
/// the slot count by repeating the last id). `None` if the program has no shift. Fails if the
/// binding count exceeds `max_bindings`.
fn spans(
    key: &FusedKey,
    scalars: &[u32],
    len: usize,
    c: usize,
    max_bindings: usize,
) -> Result<Option<SegmentSpans>, LumaError> {
    // Per shift instruction: its input and the translations (read index - written index) it
    // applies, over the wrapped and unwrapped parts of every shifted axis.
    let mut shifts: Vec<(usize, Vec<i64>)> = Vec::new();
    for instr in &key.instrs {
        let OpKind::Shift(kind) = instr.op.kind else { continue };
        let Ref::Input(k) = instr.args[0] else {
            return Err(LumaError::Gpu("segmented program with sub-programs".into()));
        };
        let deltas = match kind {
            ShiftKind::Linear => {
                let amount = match instr.args[1] {
                    Ref::Scalar(q) => scalars.get(q as usize).copied(),
                    _ => None,
                };
                let Some(amount) = amount else {
                    return Err(LumaError::Gpu("shift without a scalar amount".into()));
                };
                let s = i64::from(amount);
                if s == 0 { vec![0] } else { vec![-s, len as i64 - s] }
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
                deltas
            }
        };
        shifts.push((k as usize, deltas));
    }
    if shifts.is_empty() {
        return Ok(None);
    }

    let inputs = key.input_tys.len();
    let segments = len.div_ceil(c);
    let last = len as i64 - 1;
    let mut ids: Vec<Vec<Vec<u32>>> = Vec::with_capacity(segments);
    for j in 0..segments {
        let (a, b) = ((j * c) as i64, (len.min((j + 1) * c)) as i64);
        let mut lists: Vec<Vec<u32>> = vec![Vec::new(); inputs];
        for (k, deltas) in &shifts {
            let Some(list) = lists.get_mut(*k) else {
                return Err(LumaError::Gpu("shift of an unknown input".into()));
            };
            for delta in deltas {
                let (lo, hi) = ((a + delta).max(0), (b - 1 + delta).min(last));
                if lo <= hi {
                    list.extend((lo as usize / c..=hi as usize / c).map(|id| id as u32));
                }
            }
        }
        for list in &mut lists {
            list.sort_unstable();
            list.dedup();
        }
        ids.push(lists);
    }
    let mut counts = vec![0u8; inputs];
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
        Tail::Store => key.outputs.len(),
        Tail::Reduce(_) => 1,
    };
    let direct = (0..inputs)
        .filter(|&k| key.in_place != Some(k as u32) && direct(&key.instrs, k as u32))
        .count();
    let slots: usize = counts.iter().map(|&count| usize::from(count)).sum();
    if outputs + direct + slots > max_bindings {
        return Err(LumaError::Chunked("a shift that reads this many buffers at once"));
    }
    Ok(Some((counts, ids)))
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
    let bytes = u64::from(n) * ELEMENT_BYTES;
    let mut pins = Vec::with_capacity(inputs.len());
    let mut stored = inputs.iter();
    for k in 0..key.input_tys.len() {
        if key.in_place == Some(k as u32) {
            continue;
        }
        let Some(input) = stored.next() else {
            return Err(LumaError::Gpu("fused program with too few inputs".into()));
        };
        let Some(ids) = ids else {
            pins.push((input.pin(j)?, bytes));
            continue;
        };
        if direct(&key.instrs, k as u32) {
            pins.push((input.pin(j)?, bytes));
        }
        for &id in ids.get(k).map_or(&[][..], Vec::as_slice) {
            let id = id as usize;
            pins.push((input.pin(id)?, input.range(id).len() as u64 * ELEMENT_BYTES));
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
    bytes: u64,
    outs: Vec<Pin>,
    ins: Vec<(Pin, u64)>,
    words: Cow<'a, [u32]>,
}

impl StoreArgs<'_> {
    pub(super) fn raw(&self) -> Vec<RawArg<'_>> {
        let mut raw: Vec<RawArg<'_>> = Vec::with_capacity(self.outs.len() + self.ins.len() + 1);
        raw.extend(self.outs.iter().map(|pin| RawArg::write(pin.buffer(), self.bytes)));
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
    let bytes = u64::from(n) * ELEMENT_BYTES;
    let ids = segment_ids(shifted, j)?;
    let outs = outputs.iter().map(|out| out.pin(j)).collect::<Result<Vec<_>, _>>()?;
    let ins = pin_inputs(key, inputs, ids, j, n)?;
    let words = params(key, ids, None, range.start, shifted.map_or(0, |shifted| shifted.len), scalars)?;
    Ok(StoreArgs { n, bytes, outs, ins, words })
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
    for j in 0..first.segments() {
        let args = store_args(key, outputs, inputs, scalars, shifted, j)?;
        launch_1d(ctx, core, args.n, &args.raw())?;
    }
    ctx.jit().stats.elementwise_launches.fetch_add(1, Ordering::Relaxed);
    Ok(())
}

/// Runs the store kernel of `fused` over `len` elements, one new buffer per output. A single
/// output takes over the buffer of an input that nothing else holds (see [`in_place_eligible`]),
/// instead of a new one.
pub(super) fn run_store(ctx: &crate::context::Context, len: usize, fused: Fused) -> Result<Vec<Storage>, Failure> {
    fused.mark_fused();
    let Fused { mut key, mut inputs, scalars, fused_nodes } = fused;
    // The fused nodes hold their operands; a dying input is only uniquely owned without them.
    drop(fused_nodes);
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
                    key.in_place = Some(k as u32);
                    reused = Some(storage);
                }
                Err(shared) => inputs.insert(k, shared),
            }
        }
    }
    let shifted = plan_shifts(ctx, &mut key, &scalars, len)?;
    let core = ctx.jit().kernel(ctx, &key)?;

    let outputs = match reused {
        Some(storage) => vec![storage],
        None => {
            let mut outputs = Vec::with_capacity(count);
            for _ in 0..count {
                outputs.push(Storage::new(ctx, len, ELEMENT_BYTES as usize));
            }
            outputs
        }
    };
    let refs: Vec<&Storage> = outputs.iter().collect();
    launch_store(ctx, &core, &key, &refs, &inputs, &scalars, shifted.as_ref())?;
    Ok(outputs)
}

/// Evaluates `node` into `target`, one of its inputs, overwriting it in place: one fused kernel
/// that reads every element of `target` before writing it. The node counts as evaluated, with
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
    fused.mark_fused();
    let Fused { mut key, mut inputs, scalars, fused_nodes } = fused;
    drop(fused_nodes);
    let at = inputs.iter().position(|input| Arc::ptr_eq(input, target));
    let Some(k) = at.filter(|&k| in_place_eligible(&key, k as u32)) else {
        return Err(Failure::New(LumaError::Gpu("in-place assignment cannot run in place".into())));
    };
    inputs.remove(k);
    key.in_place = Some(k as u32);
    let shifted = plan_shifts(ctx, &mut key, &scalars, node.len())?;
    let core = ctx.jit().kernel(ctx, &key)?;
    launch_store(ctx, &core, &key, &[&**target], &inputs, &scalars, shifted.as_ref())?;
    // Another thread cannot have evaluated this node: its operands were taken above.
    let _ = node.result.set(Ok(Arc::clone(target)));
    Ok(())
}

/// Runs the first pass of the reductions `ops` directly over the expression behind `node`,
/// without materializing it: every workgroup evaluates its strided blocks of `Tuning::reduce_block` elements and reduces them,
/// once per op. Returns the buffer of partial results (op `j` at `j * groups ..`) and how many
/// there are per op, or `None` if the node has already been evaluated (reduce its buffer
/// instead).
pub(crate) fn reduce_first_pass(
    node: &Arc<Node>,
    ops: ReduceList,
    scratch: &mut Scratch<'_>,
) -> Result<Option<Vec<(wgpu::Buffer, u32)>>, Failure> {
    if !node.is_pending() {
        return Ok(None);
    }
    let Some(args) = node.operands() else { return Ok(None) };
    let (fused, args) = flatten_bounded(node, args, Tail::Reduce(ops))?;
    drop(args);
    reduce_fused(node.context(), node.len(), fused, ops, scratch).map(Some)
}

/// The first pass of [`reduce_first_pass`] for an already flattened expression of `len` elements.
pub(super) fn reduce_fused(
    ctx: &crate::context::Context,
    len: usize,
    fused: Fused,
    ops: ReduceList,
    scratch: &mut Scratch<'_>,
) -> Result<Vec<(wgpu::Buffer, u32)>, Failure> {
    fused.mark_fused();
    let Fused { mut key, inputs, scalars, .. } = fused;
    // Without inputs (only generators) the segments are those an array of this length would have.
    let layout;
    let first = match inputs.first() {
        Some(first) => first,
        None => {
            layout = Storage::new(ctx, len, ELEMENT_BYTES as usize);
            &layout
        }
    };
    let segments = first.segments();
    let shifted = plan_shifts(ctx, &mut key, &scalars, len)?;
    let core = ctx.jit().kernel(ctx, &key)?;

    let mut partials = Vec::with_capacity(segments);
    for j in 0..segments {
        let range = first.range(j);
        let n = len_u32(range.len())?;
        let groups = ctx.tuning().reduce_groups(n);
        let out_bytes = u64::from(groups) * ELEMENT_BYTES * ops.as_slice().len() as u64;
        let out = scratch.take(out_bytes)?;
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
