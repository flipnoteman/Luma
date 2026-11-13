//! `Array::iterate_until`: blocks of steps recorded at once. Every launch of a step takes its
//! workgroup counts from an argument buffer that a gate kernel, run after the step, either
//! copies from a template or zeroes once the residual is within the tolerance, so the steps after
//! convergence dispatch no workgroups and the host waits once per block instead of once per step.
//! The first block is [`FIRST_BLOCK`] steps and every block that ends without a stop doubles the
//! next one, up to [`MAX_BLOCK`]; the result does not depend on the schedule.

use std::sync::Arc;

use wgpu::BufferUsages;

use crate::array::{Failure, Storage};
use crate::builtin::{Builtin, ReduceList, ReduceOp};
use crate::context::{Context, RawArg, Scratch, STORAGE_USAGE};
use crate::error::LumaError;
use crate::ops::{fold_workgroups, launch_1d, workgroups_1d};

use super::launch::{len_u32, params, pin_inputs, store_args};
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
    Some(run(ctx, from, probe, &body, &check, len, max_steps, tol))
}

#[allow(clippy::too_many_arguments)]
fn run(
    ctx: &Context,
    from: &Arc<Storage>,
    probe: &Arc<Storage>,
    body: &Fused,
    check: &Fused,
    len: usize,
    max_steps: usize,
    tol: f32,
) -> Result<Arc<Storage>, Failure> {
    let n = len_u32(len)?;
    let max_ops = ReduceList::new(&[ReduceOp::Max]);
    let (mut body_key, mut body_scalars) = (body.key.clone(), body.scalars.clone());
    let (mut check_key, mut check_scalars) = (check.key.clone(), check.scalars.clone());
    let body_core = ctx.jit().kernel(ctx, &mut body_key, &mut body_scalars)?;
    let check_core = ctx.jit().kernel(ctx, &mut check_key, &mut check_scalars)?;
    let gate_core = ctx.builtin(Builtin::UntilGate, "u32")?;
    let finish_core = ctx.builtin(Builtin::Reduce(max_ops.rest(), true), "f32")?;

    // The dispatches of one step, in order: the body, the first pass of the residual, then the
    // passes that fold its partial results into one value.
    let mut scratch = Scratch::new(ctx);
    let mut dispatches = vec![workgroups_1d(ctx, &body_core, n)?];
    let mut groups = ctx.tuning().reduce_groups(n);
    dispatches.push(fold_workgroups(ctx, groups)?);
    let mut partials = vec![(scratch.take(u64::from(groups) * ELEMENT_BYTES)?, u64::from(groups) * ELEMENT_BYTES)];
    let mut folds = Vec::new();
    while groups > 1 {
        let from_groups = groups;
        groups = ctx.tuning().reduce_groups(from_groups);
        dispatches.push(fold_workgroups(ctx, groups)?);
        partials.push((scratch.take(u64::from(groups) * ELEMENT_BYTES)?, u64::from(groups) * ELEMENT_BYTES));
        folds.push((from_groups, groups));
    }
    let Some((value, _)) = partials.last() else {
        return Err(Failure::New(LumaError::Gpu("residual without a partial result".into())));
    };

    let template: Vec<u32> = dispatches.iter().flatten().copied().collect();
    let words = len_u32(template.len())?;
    let args_bytes = u64::from(words) * WORD;
    let indirect = STORAGE_USAGE | BufferUsages::INDIRECT;
    let mut owned = Vec::new();
    let mut take = |bytes: u64, usage: BufferUsages| -> Result<wgpu::Buffer, LumaError> {
        let buffer = ctx.alloc_buffer(bytes, usage)?;
        owned.push(buffer.clone());
        Ok(buffer)
    };
    let outcome = (|| -> Result<Arc<Storage>, Failure> {
        let tmpl = take(args_bytes, STORAGE_USAGE)?;
        ctx.upload(&tmpl, bytemuck::cast_slice(&template))?;
        let state = take(2 * WORD, STORAGE_USAGE)?;
        let mut steps: Vec<wgpu::Buffer> = Vec::new();
        let ping = [Arc::new(Storage::new(ctx, len, ELEMENT_BYTES as usize)), Arc::new(Storage::new(ctx, len, ELEMENT_BYTES as usize))];
        let body_state = body.inputs.iter().position(|input| Arc::ptr_eq(input, probe));
        let check_state = check.inputs.iter().position(|input| Arc::ptr_eq(input, probe));

        let mut base = 0;
        let mut block = FIRST_BLOCK;
        loop {
            let in_block = block.min(max_steps - base);
            while steps.len() <= in_block {
                steps.push(take(args_bytes, indirect)?);
            }
            ctx.upload(&state, bytemuck::cast_slice(&[0u32, 0]))?;
            ctx.upload(&steps[0], bytemuck::cast_slice(&template))?;
            for k in 0..in_block {
                let i = base + k;
                let current = if i == 0 { from } else { &ping[(i - 1) % 2] };
                let next = &ping[i % 2];

                let mut inputs = body.inputs.clone();
                if let Some(at) = body_state {
                    inputs[at] = Arc::clone(current);
                }
                let args = store_args(&body_key, &[&**next], &inputs, &body_scalars, None, 0)?;
                ctx.jit().stats.elementwise_launches.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                ctx.launch_indirect(&body_core, &steps[k], 0, [n, 1, 1], &args.raw())?;

                let mut inputs = check.inputs.clone();
                if let Some(at) = check_state {
                    inputs[at] = Arc::clone(current);
                }
                let pins = pin_inputs(&check_key, &inputs, None, 0, n)?;
                let uniform = params(&check_key, None, Some(n), 0, len, &check_scalars)?;
                let (first_out, first_bytes) = &partials[0];
                let mut raw: Vec<RawArg<'_>> = Vec::with_capacity(pins.len() + 2);
                raw.push(RawArg::write(first_out, *first_bytes));
                raw.extend(pins.iter().map(|(pin, bytes)| RawArg::read(pin.buffer(), *bytes)));
                raw.push(RawArg::Uniform(bytemuck::cast_slice(&uniform)));
                ctx.jit().stats.reduce_launches.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                ctx.launch_indirect(&check_core, &steps[k], WORD * 3, [n, 1, 1], &raw)?;
                drop(raw);

                for (pass, &(count, _)) in folds.iter().enumerate() {
                    let (input, input_bytes) = &partials[pass];
                    let (output, output_bytes) = &partials[pass + 1];
                    ctx.launch_indirect(
                        &finish_core,
                        &steps[k],
                        WORD * 3 * (2 + pass as u64),
                        [count, 1, 1],
                        &[
                            RawArg::write(output, *output_bytes),
                            RawArg::read(input, *input_bytes),
                            RawArg::Uniform(bytemuck::bytes_of(&[count])),
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
