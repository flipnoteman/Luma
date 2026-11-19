//! `Array::iterate_until`: blocks of steps recorded at once. Every launch of a step takes its
//! workgroup counts from an argument buffer that a gate kernel, run after the step, either
//! copies from a template or zeroes once the residual is within the tolerance, so the steps after
//! convergence dispatch no workgroups and the host waits once per block instead of once per step.
//! The first block is [`FIRST_BLOCK`] steps and every block that ends without a stop doubles the
//! next one, up to [`MAX_BLOCK`]; the result does not depend on the schedule.

use std::sync::Arc;

use luma_core::{Buffer, BufferUsage};

use crate::array::{Failure, Storage};
use crate::gpu_type::ElemLayout;
use crate::builtin::{Builtin, ReduceList, ReduceOp};
use crate::context::{Context, RawArg, Scratch, STORAGE_USAGE};
use crate::error::LumaError;
use crate::ops::{fold_workgroups, launch_1d, len_u32, workgroups_1d};

use super::launch::{params, pin_inputs, store_args};
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

/// Bytes of one element of the `f32` states this machinery handles.
const ELEMENT_BYTES: u64 = 4;

// Tests: run the body and the check as two kernels even where one program does both.
#[cfg(test)]
thread_local! {
    pub(crate) static SEPARATE_KERNELS: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
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
    let mut body_parts = body.map(|body| (body.key.clone(), body.scalars.clone()));
    let (mut check_key, mut check_scalars) = (check.key.clone(), check.scalars.clone());
    if single {
        check_key.tail = Tail::Reduce(max_ops.single_pass());
    }
    let body_core = match &mut body_parts {
        Some((key, scalars)) => Some(ctx.jit().kernel(ctx, key, scalars)?),
        None => None,
    };
    let check_core = ctx.jit().kernel(ctx, &mut check_key, &mut check_scalars)?;
    let gate_core = ctx.builtin(Builtin::UntilGate, "u32")?;

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
        let state = take(2 * WORD, STORAGE_USAGE)?;
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
            ctx.upload(&state, bytemuck::cast_slice(&[0u32, 0]))?;
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
                ctx.launch_indirect(&check_core, &steps[k], WORD * 3 * check_at as u64, [n, 1, 1], dispatches[check_at], &check_raws[s])?;

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
                let gate = [tol.to_bits(), len_u32(i)?, words];
                launch_1d(
                    ctx,
                    &gate_core,
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
