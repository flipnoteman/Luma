//! `Array::iterate_until`: blocks of steps recorded at once. Every launch of a step takes its
//! workgroup counts from an argument buffer that a gate kernel, run after the step, either
//! copies from a template or zeroes once the residual is within the tolerance, so the steps after
//! convergence dispatch no workgroups and the host waits once per block instead of once per step.
//! The first block is [`FIRST_BLOCK`] steps and every block that ends without a stop doubles the
//! next one, up to [`MAX_BLOCK`]; the result does not depend on the schedule. Where a reduction
//! finishes in its first launch (CUDA and CPU) the residual's kernel runs the gate itself, in its last
//! workgroup ([`UNTIL_GATE`]): a step is one launch.

use std::sync::Arc;

use luma_core::{Buffer, BufferUsage};

use crate::array::{Array, Failure, Storage};
use crate::gpu_type::{ElemLayout, GpuType};
use crate::builtin::{Builtin, Finish, GateId, ReduceList, ReduceOp, GATE_BINDINGS};
use crate::context::{Context, RawArg, Scratch, STORAGE_USAGE};
use crate::error::LumaError;
use crate::ops::{fold_workgroups, launch_1d, len_u32, workgroups_1d};
use crate::reduce::finish_passes_gated;

use super::launch::{self, params, pin_inputs, store_args};
use super::program::{flatten, flatten_inner, Fused, Tail};
use super::{ElemTy, Node};

/// Steps of the first block recorded before the host waits.
pub(crate) const FIRST_BLOCK: usize = 16;

/// Largest block of the growing schedule.
pub(crate) const MAX_BLOCK: usize = 128;

/// Steps of the block after one of `current` steps that ended without a stop: `fixed` if given,
/// else double, up to `MAX_BLOCK`.
pub(crate) fn next_block(current: usize, fixed: Option<usize>) -> usize {
    fixed.unwrap_or_else(|| (current * 2).min(MAX_BLOCK))
}

/// Bytes of one word of the argument buffers.
const WORD: u64 = 4;

/// Bytes of the gate's state.
const GATE_STATE_BYTES: u64 = 8 * WORD;

/// The gate of `iterate_until` as the WGSL a reduction kernel runs on its final value (see
/// `builtin::reduce_main`): what the `UntilGate` kernel does, over the state `st`
/// (`[stopped, steps run, steps so far, words, own launch, 0, tolerance, 0]`) and
/// the arguments `args` of the next step; the other buffers it is given are not used.
const UNTIL_GATE: &str = "fn luma_gate(v: f32) {
    let i = st[2u];
    let hit = (bitcast<u32>(v) & 0x7fffffffu) <= 0x7f800000u && v <= bitcast<f32>(st[6u]);
    st[1u] = i + 1u;
    st[2u] = i + 1u;
    if (hit) { st[0u] = 1u; }
    for (var w = 0u; w < st[3u]; w++) { args[w] = select(tmpl[w], 0u, hit && w / 3u != st[4u]); }
    hist[0u] = hist[0u];
    agg[0u] = agg[0u];
    stepv[0u] = stepv[0u];
}
";

/// Bytes of one element of the `f32` states this machinery handles.
const ELEMENT_BYTES: u64 = 4;

// Tests: run the body and the check as two kernels even where one program does both.
#[cfg(test)]
thread_local! {
    pub(crate) static SEPARATE_KERNELS: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

// Tests: run the gate of `iterate_until` and `iterate_until_all` as a launch of its own.
#[cfg(test)]
thread_local! {
    pub(crate) static SEPARATE_GATE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// Launches a step of the last `iterate_until` made.
    pub(crate) static STEP_LAUNCHES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

// Tests: trace the record of `iterate_until_all` after its step, as two kernels, even where one program does both.
#[cfg(test)]
thread_local! {
    pub(crate) static SEPARATE_RECORD: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// A gate (see [`ReduceList::with_gate`]) that [`force_with_record`] may fold into its kernel.
pub(crate) struct FoldGate<'a> {
    pub(crate) id: GateId,
    /// The arguments of the gate's buffers, `GATE_BINDINGS` of them.
    pub(crate) args: &'a [RawArg<'a>],
    /// The storage of the step index, which the gate writes: a kernel that reads it cannot.
    pub(crate) stepv: u64,
}

/// What [`force_with_record`] did.
pub(crate) struct Grouped {
    /// The index in the recording of the launch that runs the gate, if it was folded in.
    pub(crate) folded: Option<usize>,
}

/// Evaluates the arrays `next` (one step of a loop, traced) and the reduction `record` (of the
/// state the step reads, as `iterate_until_all` records it) in one launch that stores the first
/// and reduces the second into `value`, one `f32` buffer of one element: the record is what
/// the next step computes from its input, shared in the program, so that the step's launch does
/// both. `None` unless that is possible (the arrays are pending expressions of one length,
/// the record is a reduction of `f32`s of that length, the program is one unsplit kernel); nothing
/// has been launched then, though flattening the program may have evaluated sub-programs it
/// cannot keep in it first (their launches are recorded like the step's own). With a `gate` the kernel also runs it once the value is
/// final, where the reduction finishes in its first pass and the kernel does not read the
/// gate's step index.
pub(crate) fn force_with_record<T: GpuType>(
    ctx: &Context,
    next: &[&Array<T>],
    record: &Array<f32>,
    value: &Buffer,
    gate: Option<FoldGate<'_>>,
) -> Option<Result<Grouped, Failure>> {
    #[cfg(test)]
    if SEPARATE_RECORD.with(std::cell::Cell::get) {
        return None;
    }
    let (operand, ops, finish) = record.reduced()?.fusable()?;
    if ops.as_slice().len() != 1 || ctx.check_finite() || !operand.is_pending() {
        return None;
    }
    let nodes: Vec<&Arc<Node>> = next.iter().map(|array| array.pending()).collect::<Option<_>>()?;
    let len = operand.len();
    let distinct = nodes.iter().enumerate().all(|(i, node)| !Arc::ptr_eq(node, operand) && nodes[..i].iter().all(|other| !Arc::ptr_eq(other, node)));
    if nodes.is_empty() || !distinct || nodes.iter().any(|node| node.len() != len) {
        return None;
    }
    let mut roots = Vec::with_capacity(nodes.len() + 1);
    for node in nodes.iter().copied().chain(std::iter::once(operand)) {
        roots.push((Arc::clone(node), node.operands()?));
    }
    let fused = flatten_inner(&roots, Tail::Reduce(ops), &[]).ok()?;
    drop(roots);
    let key = &fused.key;
    let (&reduced, stored) = key.outputs.split_last()?;
    let f32_at = |at: u32| key.instrs.get(at as usize).is_some_and(|instr| instr.op.out_ty() == ElemTy::F32);
    let max_inputs = ctx.jit().max_inputs();
    let fits = key.outputs.len() == nodes.len() + 1
        && reduced as usize + 1 == key.instrs.len()
        && f32_at(reduced)
        && stored.iter().enumerate().all(|(i, &at)| at != reduced && !stored[..i].contains(&at))
        // Sub-programs read through views are written out where they are read (see `inline`), which
        // a reduction does; those staged in workgroup memory need a store kernel's tiles.
        && (key.subs.is_empty() || (key.inlines() && key.staging.is_none()))
        && key.narrow()
        && fused.inputs.iter().all(|input| input.segments() == 1)
        && fused.inputs.len() + stored.len() < max_inputs;
    if !fits {
        return None;
    }
    // The gate runs in the first pass where that one finishes the reduction (one group, or a single
    // pass), else in the last pass over its partial results, if the device binds that many buffers.
    let single_pass = ctx.tuning().single_pass_reduce;
    let finishes = single_pass || ctx.tuning().reduce_groups(len_u32(len).ok()?) == 1;
    let gate = gate.filter(|gate| {
        fused.inputs.iter().all(|input| input.id() != gate.stepv)
            && if finishes { fused.inputs.len() + stored.len() + GATE_BINDINGS + 1 < max_inputs } else { 2 + GATE_BINDINGS + 1 < max_inputs }
    });
    let stored = stored.to_vec();
    let mut scratch = Scratch::new(ctx);
    let mut own = ctx.recorded();
    let first_gate = gate.as_ref().filter(|_| finishes).map(|gate| (gate.id, gate.args));
    let (storages, partial, groups) = launch::reduce_and_store_fused(ctx, len, fused, &stored, ops, &mut scratch, Some(value), finish, first_gate).ok()?;
    // Past a first pass that did not finish (several groups, no single pass): the rest, ending in `value`.
    let rest = if groups > 1 { finish } else { Finish::default() };
    let last_gate = gate.as_ref().filter(|_| !finishes).map(|gate| (gate.id, gate.args));
    if let Err(error) = finish_passes_gated(ctx, &mut scratch, ops, "f32", ELEMENT_BYTES, partial, groups, true, Some(value), rest, last_gate) {
        return Some(Err(Failure::New(error)));
    }
    if last_gate.is_some() {
        own = ctx.recorded() - 1;
    }
    for (node, storage) in nodes.iter().zip(storages) {
        node.finish_shared(Arc::new(storage));
    }
    Some(Ok(Grouped { folded: gate.is_some().then_some(own) }))
}


/// What a step of the loop launches: the body and the residual's first pass as two kernels, or
/// one program that stores the step and reduces its residual.
#[derive(Clone, Copy)]
enum Step<'a> {
    Separate { body: &'a Fused, check: &'a Fused },
    Fused(&'a Fused),
}

/// One program computing `first` (stored as the next state) and the maximum of `residual`
/// (which reads `first`, computed once), as [`launch::reduce_and_store_fused`](super::launch)
/// runs for the speculation: `None` where that cannot run, or the values are not plain `f32`s
/// of one buffer each.
fn fuse(ctx: &Context, first: &Arc<Node>, residual: &Arc<Node>, probe: &Arc<Storage>, max_ops: ReduceList) -> Option<Fused> {
    #[cfg(test)]
    if SEPARATE_KERNELS.with(std::cell::Cell::get) {
        return None;
    }
    let roots = [(Arc::clone(first), first.operands()?), (Arc::clone(residual), residual.operands()?)];
    let fused = flatten_inner(&roots, Tail::Reduce(max_ops), &[probe.id()]).ok()?;
    let key = &fused.key;
    let [stored, reduced] = key.outputs[..] else { return None };
    let f32_at = |at: u32| key.instrs.get(at as usize).is_some_and(|instr| instr.op.out_ty() == ElemTy::F32);
    let fits = reduced as usize + 1 == key.instrs.len()
        && stored != reduced
        && f32_at(stored)
        && f32_at(reduced)
        && key.subs.is_empty()
        && key.narrow()
        && fused.inputs.iter().all(|input| input.segments() == 1)
        && fused.inputs.len() + 1 < ctx.jit().max_inputs();
    fits.then_some(fused)
}

/// Runs `first` (one step from `probe`, an alias of `from`) repeatedly, each time from the state
/// the last step wrote, until the largest `|next - prev|` (`residual`, computed from `probe`
/// too) is at most `tol` or `max_steps` steps ran. Returns the last state, or `None` if the step
/// is not a single-buffer program of `f32` values; the caller then loops on the host.
pub(crate) fn run_until(
    ctx: &Context,
    from: &Arc<Storage>,
    probe: &Arc<Storage>,
    first: &Arc<Node>,
    residual: &Arc<Node>,
    max_steps: usize,
    tol: f32,
) -> Option<Result<Arc<Storage>, Failure>> {
    let len = first.len();
    if max_steps == 0
        || max_steps > u32::MAX as usize
        || residual.len() != len
        || !first.is_pending()
        || !residual.is_pending()
        || probe.was_materialized()
        || from.segments() != 1
        || from.byte_len() != len as u64 * ELEMENT_BYTES
    {
        return None;
    }
    let max_ops = ReduceList::new(&[ReduceOp::Max]);
    if let Some(fused) = fuse(ctx, first, residual, probe, max_ops) {
        if probe.was_materialized() {
            return None;
        }
        return Some(run(ctx, from, probe, Step::Fused(&fused), len, max_steps, tol));
    }
    let body = flatten_inner(&[(Arc::clone(first), first.operands()?)], Tail::Store, &[probe.id()]).ok()?;
    let check = flatten(&[(Arc::clone(residual), residual.operands()?)], Tail::Reduce(max_ops)).ok()?;
    let one_buffer = |fused: &Fused| {
        fused.key.subs.is_empty()
            && fused.inputs.iter().all(|input| input.segments() == 1)
            && fused.inputs.len() < ctx.jit().max_inputs()
    };
    let stores_f32 = body
        .key
        .outputs
        .first()
        .and_then(|&at| body.key.instrs.get(at as usize))
        .is_some_and(|instr| instr.op.out_ty() == ElemTy::F32);
    let reads_f32 = check.key.instrs.last().is_some_and(|instr| instr.op.out_ty() == ElemTy::F32);
    if probe.was_materialized()
        || body.key.outputs.len() != 1
        || !stores_f32
        || !reads_f32
        || !one_buffer(&body)
        || !body.key.narrow()
        || !check.key.narrow()
        || !one_buffer(&check)
    {
        return None;
    }
    Some(run(ctx, from, probe, Step::Separate { body: &body, check: &check }, len, max_steps, tol))
}

#[allow(clippy::too_many_arguments)]
fn run(
    ctx: &Context,
    from: &Arc<Storage>,
    probe: &Arc<Storage>,
    step: Step<'_>,
    len: usize,
    max_steps: usize,
    tol: f32,
) -> Result<Arc<Storage>, Failure> {
    let n = len_u32(len)?;
    let max_ops = ReduceList::new(&[ReduceOp::Max]);
    let (body, check) = match step {
        Step::Separate { body, check } => (Some(body), check),
        Step::Fused(fused) => (None, fused),
    };
    // With `Tuning::single_pass_reduce` the check kernel also folds its partial results (the last
    // workgroup to finish does): one launch per step, its value in `value`. A step after the
    // gate has zeroed the dispatch arguments runs no workgroup, takes no ticket and leaves the
    // counter at zero.
    let single = ctx.tuning().single_pass_reduce && ctx.tuning().reduce_groups(n) > 1;
    // On such a backend (and for a residual of one workgroup) the check kernel also runs the gate,
    // so that a step is that one launch (see `UNTIL_GATE`): the workgroups of a step after the
    // stop find the loop stopped and leave at once, the first of them zeroing the dispatch
    // arguments of the next. The check kernel then is the step's last launch.
    let fold = ctx.tuning().single_pass_reduce && check.inputs.len() + GATE_BINDINGS + 4 <= ctx.jit().max_inputs() && {
        #[cfg(test)]
        {
            !SEPARATE_GATE.with(std::cell::Cell::get)
        }
        #[cfg(not(test))]
        {
            true
        }
    };
    let mut body_parts = body.map(|body| (body.key.clone(), body.scalars.clone()));
    let (mut check_key, mut check_scalars) = (check.key.clone(), check.scalars.clone());
    if single {
        check_key.tail = Tail::Reduce(max_ops.single_pass());
    }
    if let (true, Tail::Reduce(list)) = (fold, check_key.tail) {
        check_key.tail = Tail::Reduce(list.with_gate(GateId::register(UNTIL_GATE)));
    }
    let body_core = match &mut body_parts {
        Some((key, scalars)) => Some(ctx.jit().kernel(ctx, key, scalars)?),
        None => None,
    };
    let check_core = ctx.jit().kernel(ctx, &mut check_key, &mut check_scalars)?;
    let gate_core = if fold { None } else { Some(ctx.builtin(Builtin::UntilGate, "u32")?) };

    // The dispatches of one step, in order: the body (unless the check stores the step itself),
    // the first pass of the residual, then the passes that fold its partial results into one
    // value.
    let mut scratch = Scratch::new(ctx);
    let mut dispatches = Vec::new();
    if let Some(body_core) = &body_core {
        dispatches.push(workgroups_1d(ctx, body_core, n)?);
    }
    let check_at = dispatches.len();
    let mut groups = ctx.tuning().reduce_groups(n);
    dispatches.push(fold_workgroups(ctx, groups)?);
    let first_fold = dispatches.len();
    let mut partials = vec![(scratch.take(u64::from(groups) * ELEMENT_BYTES)?, u64::from(groups) * ELEMENT_BYTES)];
    let mut folds = Vec::new();
    while groups > 1 && !single {
        let from_groups = groups;
        groups = ctx.tuning().reduce_groups(from_groups);
        dispatches.push(fold_workgroups(ctx, groups)?);
        partials.push((scratch.take(u64::from(groups) * ELEMENT_BYTES)?, u64::from(groups) * ELEMENT_BYTES));
        folds.push((from_groups, groups));
    }
    let single_value = if single { Some(scratch.take(ELEMENT_BYTES)?) } else { None };
    let finish_core = if folds.is_empty() { None } else { Some(ctx.builtin(Builtin::Reduce(max_ops.rest(), true), "f32")?) };
    let counter = if single { Some(ctx.reduce_counter()?) } else { None };
    let Some(value) = single_value.as_ref().or_else(|| partials.last().map(|(buffer, _)| buffer)) else {
        return Err(Failure::New(LumaError::Gpu("residual without a partial result".into())));
    };

    let template: Vec<u32> = dispatches.iter().flatten().copied().collect();
    let words = len_u32(template.len())?;
    #[cfg(test)]
    STEP_LAUNCHES.with(|launches| launches.set(dispatches.len() + usize::from(gate_core.is_some())));
    let args_bytes = u64::from(words) * WORD;
    let indirect = STORAGE_USAGE | BufferUsage::INDIRECT;
    let mut owned = Vec::new();
    // Counted as resident while the run uses them.
    let mut held = crate::gpu::pager::Accounted::new(ctx, 0);
    let mut take = |bytes: u64, usage: BufferUsage| -> Result<Buffer, LumaError> {
        let buffer = ctx.alloc_buffer(bytes, usage)?;
        held.add(buffer.size());
        owned.push(buffer.clone());
        Ok(buffer)
    };
    let outcome = (|| -> Result<Arc<Storage>, Failure> {
        let tmpl = take(args_bytes, STORAGE_USAGE)?;
        ctx.upload(&tmpl, bytemuck::cast_slice(&template))?;
        // `[stopped, steps run, ..]`; a folded gate's state is that of `UNTIL_GATE`.
        let state = take(GATE_STATE_BYTES, STORAGE_USAGE)?;
        let gate_dummies = if fold { Some([take(16, STORAGE_USAGE)?, take(16, STORAGE_USAGE)?, take(16, STORAGE_USAGE)?]) } else { None };
        let mut steps: Vec<Buffer> = Vec::new();
        let ping = [Arc::new(Storage::new(ctx, len, ElemLayout::plain(ELEMENT_BYTES as usize))), Arc::new(Storage::new(ctx, len, ElemLayout::plain(ELEMENT_BYTES as usize)))];
        let body_state = body.and_then(|body| body.inputs.iter().position(|input| Arc::ptr_eq(input, probe)));
        let check_state = check.inputs.iter().position(|input| Arc::ptr_eq(input, probe));
        let states = [(from, &ping[0]), (&ping[0], &ping[1]), (&ping[1], &ping[0])];
        let mut body_args = Vec::with_capacity(3);
        let mut check_pins = Vec::with_capacity(3);
        let mut next_pins = Vec::with_capacity(3);
        for &(current, next) in &states {
            if let (Some(body), Some((body_key, body_scalars))) = (body, &body_parts) {
                let mut inputs = body.inputs.clone();
                if let Some(at) = body_state {
                    inputs[at] = Arc::clone(current);
                }
                body_args.push(store_args(body_key, &[&**next], &inputs, body_scalars, None, 0)?);
            } else {
                // The check stores the state itself.
                next_pins.push(next.pin(0)?);
            }

            let mut inputs = check.inputs.clone();
            if let Some(at) = check_state {
                inputs[at] = Arc::clone(current);
            }
            check_pins.push(pin_inputs(&check_key, &inputs, None, 0, n)?);
        }
        let body_raws: [Vec<RawArg<'_>>; 3] = std::array::from_fn(|s| body_args.get(s).map_or_else(Vec::new, |args| args.raw()));
        let check_uniform = params(&check_key, None, Some(n), 0, len, None, &check_scalars)?;
        let (first_out, first_bytes) = &partials[0];
        let check_raws: [Vec<RawArg<'_>>; 3] = std::array::from_fn(|s| {
            let pins = &check_pins[s];
            let mut raw: Vec<RawArg<'_>> = Vec::with_capacity(pins.len() + 5);
            raw.push(RawArg::write(first_out, *first_bytes));
            if let Some(next) = next_pins.get(s) {
                raw.push(RawArg::write(next.buffer(), len as u64 * ELEMENT_BYTES));
            }
            raw.extend(pins.iter().map(|(pin, bytes)| RawArg::read(pin.buffer(), *bytes)));
            if let (Some(counter), Some(result)) = (counter, &single_value) {
                raw.push(RawArg::write(counter, 4));
                raw.push(RawArg::write(result, ELEMENT_BYTES));
            }
            raw.push(RawArg::Uniform(bytemuck::cast_slice(&check_uniform)));
            raw
        });
        let fold_counts: Vec<[u32; 1]> = folds.iter().map(|&(count, _)| [count]).collect();

        let mut base = 0;
        let mut block = FIRST_BLOCK;
        loop {
            let in_block = block.min(max_steps - base);
            while steps.len() <= in_block {
                steps.push(take(args_bytes, indirect)?);
            }
            // `[stopped, steps run, steps before the block, words, the check kernel's launch, 0, tolerance, 0]`.
            let words_of_state = [0u32, 0, len_u32(base)?, words, check_at as u32, 0, tol.to_bits(), 0];
            ctx.upload(&state, bytemuck::cast_slice(&words_of_state))?;
            ctx.upload(&steps[0], bytemuck::cast_slice(&template))?;
            if body.is_some() {
                ctx.jit().stats.elementwise_launches.fetch_add(in_block as u64, std::sync::atomic::Ordering::Relaxed);
            }
            ctx.jit().stats.reduce_launches.fetch_add(in_block as u64, std::sync::atomic::Ordering::Relaxed);
            for k in 0..in_block {
                let i = base + k;
                let s = if i == 0 { 0 } else if i % 2 == 1 { 1 } else { 2 };

                if let Some(body_core) = &body_core {
                    ctx.launch_indirect(body_core, &steps[k], 0, [n, 1, 1], dispatches[0], &body_raws[s])?;
                }
                if let Some([hist, agg, stepv]) = &gate_dummies {
                    let mut raw = check_raws[s].clone();
                    let uniform = raw.pop();
                    raw.extend([
                        RawArg::write(&state, GATE_STATE_BYTES),
                        RawArg::write(&steps[k + 1], args_bytes),
                        RawArg::read(&tmpl, args_bytes),
                        RawArg::write(hist, 16),
                        RawArg::write(agg, 16),
                        RawArg::write(stepv, 16),
                    ]);
                    raw.extend(uniform);
                    ctx.launch_indirect(&check_core, &steps[k], WORD * 3 * check_at as u64, [n, 1, 1], dispatches[check_at], &raw)?;
                } else {
                    ctx.launch_indirect(&check_core, &steps[k], WORD * 3 * check_at as u64, [n, 1, 1], dispatches[check_at], &check_raws[s])?;
                }

                for (pass, count) in fold_counts.iter().enumerate() {
                    let Some(finish_core) = &finish_core else { break };
                    let (input, input_bytes) = &partials[pass];
                    let (output, output_bytes) = &partials[pass + 1];
                    ctx.launch_indirect(
                        finish_core,
                        &steps[k],
                        WORD * 3 * (first_fold + pass) as u64,
                        [count[0], 1, 1],
                        dispatches[first_fold + pass],
                        &[
                            RawArg::write(output, *output_bytes),
                            RawArg::read(input, *input_bytes),
                            RawArg::Uniform(bytemuck::bytes_of(count)),
                        ],
                    )?;
                }
                if let Some(gate_core) = &gate_core {
                    let gate = [tol.to_bits(), len_u32(i)?, words];
                    launch_1d(
                        ctx,
                        gate_core,
                        words,
                        &[
                            RawArg::read(value, WORD),
                            RawArg::write(&state, 2 * WORD),
                            RawArg::write(&steps[k + 1], args_bytes),
                            RawArg::read(&tmpl, args_bytes),
                            RawArg::Uniform(bytemuck::cast_slice(&gate)),
                        ],
                    )?;
                }
            }
            let [done, count] = ctx.read::<u32>(&state, 2 * WORD)?[..] else {
                return Err(Failure::New(LumaError::Gpu("iterate_until state is missing".into())));
            };
            base += in_block;
            block = next_block(in_block, None);
            if done != 0 || base >= max_steps {
                let steps_run = if done != 0 { count as usize } else { base };
                return Ok(Arc::clone(&ping[(steps_run - 1) % 2]));
            }
        }
    })();
    for buffer in owned {
        ctx.recycle(buffer);
    }
    outcome
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::ContextOptions;

    fn single_pass_ctx() -> Context {
        use luma_core::{BackendKind, DeviceKind};
        let options = ContextOptions { backend: BackendKind::Cpu, device: DeviceKind::Cpu, ..ContextOptions::from_env() }.calibrate(false);
        let ctx = Context::with_tuning(options, |tuning| tuning.single_pass_reduce = true).unwrap();
        ctx.set_cost_model(crate::jit::cost::CostModel::TEST);
        ctx
    }

    fn data(n: usize, seed: usize) -> Vec<f32> {
        (0..n).map(|i| (((i + seed) * 7919) % 1009) as f32 / 1009.0 - 0.5).collect()
    }

    #[test]
    fn the_gate_of_iterate_until_runs_in_the_residual_kernel_with_the_results_of_a_launch_of_its_own() {
        for n in [100usize, 16384] {
            let ctx = single_pass_ctx();
            assert_eq!(ctx.tuning().reduce_groups(n as u32) > 1, n > 100);
            let (init, source) = (data(n, 1), data(n, 2));
            let c = Array::from_slice_on(&ctx, &source, n).unwrap();
            let run = |separate_gate: bool, separate_kernels: bool, tol: f32, max: usize| {
                SEPARATE_GATE.with(|flag| flag.set(separate_gate));
                SEPARATE_KERNELS.with(|flag| flag.set(separate_kernels));
                let x = Array::from_slice_on(&ctx, &init, n).unwrap();
                let got = x.iterate_until(max, tol, |u| &(u * 0.9f32) + &c).to_vec().unwrap();
                SEPARATE_GATE.with(|flag| flag.set(false));
                SEPARATE_KERNELS.with(|flag| flag.set(false));
                (got, STEP_LAUNCHES.with(std::cell::Cell::get))
            };
            // Converging after a few blocks, never, at the first step, at the last of few, and after one step only.
            for (tol, max) in [(1e-4f32, 300), (0.0, 40), (1e9, 300), (1e-4, 3), (1e-4, 1)] {
                let (folded, fold_launches) = run(false, false, tol, max);
                let (separate, separate_launches) = run(true, false, tol, max);
                assert_eq!(folded, separate, "n = {n}, tol = {tol}, max = {max}");
                // The step and its residual are one launch, and the gate is no other.
                assert_eq!((fold_launches, separate_launches), (1, 2), "n = {n}");
                let (kernels, kernel_launches) = run(false, true, tol, max);
                assert_eq!(kernels, separate, "n = {n}, tol = {tol}, max = {max}, the step as a kernel of its own");
                assert_eq!(kernel_launches, 2, "n = {n}");
            }
            let (converged, _) = run(false, false, 1e-4, 300);
            let (not_yet, _) = run(false, false, 1e-4, 3);
            assert_ne!(converged, not_yet);
        }
    }
}
