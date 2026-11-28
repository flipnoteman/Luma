//! Float reductions over arrays of several segments as canonical blocks (W3-U6 D8).
//!
//! The elements are cut into blocks of `B = min(2^16, C)` (`C`: the elements of a full segment, a power of two). The first pass of a
//! slab leaves the state of every block (workgroup `g` reduces exactly the elements of block `g`); a balanced merge folds the states
//! of the slab into the slab's slot of the stage's *carry*; after the last slab the same merge folds the slots (in slab order, whatever
//! order the slabs were visited in) and applies the epilogue into the result. A slab is a power-of-two range of blocks, so every slab
//! state is a node of the one balanced tree over all blocks, and the result is a function of the data and `B` alone: not of the slab
//! size, the slab order or the memory available.

use std::sync::Arc;

use luma_core::Buffer;

use super::{defer, Open, Origin, Reach, SegRef, Space, Step, StepKernel};
use crate::array::{Failure, Storage};
use crate::builtin::{Builtin, ReduceList};
use crate::jit::structured::epilogue::{EpiId, EpiProgram};
use crate::context::{Context, RawArg, STORAGE_USAGE};
use crate::error::LumaError;
use crate::jit::launch::{canonical_block, reduce_blocks_slab, BlockSource};
use crate::jit::program::FusedKey;
use crate::jit::structured::strategy::SplitClass;
use crate::kernel::KernelCore;
use crate::ops::fold_workgroups;

/// How the first pass reads its elements.
pub(crate) enum First {
    /// The program of a pending expression, with the inputs of the step.
    Fused { core: Arc<KernelCore>, key: Arc<FusedKey> },
    /// The elements of an evaluated array (the one input of the step).
    Buffer { core: Arc<KernelCore> },
}

/// What a `ReduceBlocks` step runs per slab and at the end of its stage.
pub(crate) struct Blocks {
    first: First,
    /// The first pass's list (it names the block).
    ops: ReduceList,
    /// The epilogue the last fold applies (a whole reduction's program, scalars only).
    epi: Option<Arc<EpiProgram>>,
    wgsl: &'static str,
    /// Bytes of one state (of one op).
    cell: u64,
}

impl Blocks {
    fn block(&self) -> u32 {
        self.ops.block().expect("a block first pass")
    }
}

/// What the reduction is of, besides where its elements come from.
#[derive(Clone)]
pub(crate) struct Spec {
    pub(crate) ops: ReduceList,
    pub(crate) wgsl: &'static str,
    /// Bytes of one value of one op.
    pub(crate) cell: u64,
    pub(crate) epi: Option<Arc<EpiProgram>>,
}

/// Whether a reduction over `len` elements of `layout` in `ctx` is done in canonical blocks: the context splits (`LUMA_SPLIT` is not
/// `off`) and the elements span several segments. The partial of a slab of an integer reduction is exact (the operations are associative
/// and commutative on integers, wrapping included), so the same blocks and the same tree give the bits the single pass gives; a float
/// reduction's bits are the tree's (see the module documentation). Bit-packed and single-segment reductions keep their kernels.
pub(crate) fn applies(ctx: &Context, len: usize, layout: crate::gpu_type::ElemLayout) -> bool {
    ctx.split_mode() != super::SplitMode::Off && Storage::segment_count_of(ctx, len, layout) > 1
}

/// Records the canonical-block reduction `spec` of the elements of `fused` (or, when the expression is not fused, of the storage
/// `storage()` returns) into `result` (one cell per op): in this thread's schedule when `deferred`, else it runs before this returns.
#[allow(clippy::too_many_arguments)]
pub(crate) fn record(
    ctx: &Context,
    fused: Option<BlockSource>,
    storage: impl FnOnce() -> Result<Arc<Storage>, Failure>,
    spec: Spec,
    result: &Arc<Storage>,
    deferred: bool,
    origin: Origin,
    writer: (&'static str, &'static std::panic::Location<'static>),
) -> Result<(), Failure> {
    #[allow(clippy::type_complexity)]
    let (first, inputs, reaches, scalars, shifted, space, ops): (First, Vec<Arc<Storage>>, Vec<Reach>, Vec<u32>, _, Space, ReduceList);
    match fused {
        Some(source) => {
            let BlockSource { core, key, inputs: ins, scalars: sc, shifted: sh, len, seg_len, ops: blocked } = source;
            reaches = (0..key.input_tys.len() as u32).filter(|&k| !key.in_place.iter().any(|&(paired, _)| paired == k)).map(|k| super::reach_of(&key, k)).collect();
            first = First::Fused { core, key };
            inputs = ins;
            scalars = sc;
            shifted = sh;
            space = Space { len, seg_len };
            ops = blocked;
        }
        None => {
            let array = storage()?;
            let meta = array.as_seg().meta();
            let blocked = spec.ops.blocked(canonical_block(meta.seg_len));
            let core = ctx.builtin(Builtin::Reduce(blocked, false), spec.wgsl)?;
            first = First::Buffer { core };
            inputs = vec![array];
            reaches = vec![Reach::Point];
            scalars = Vec::new();
            shifted = None;
            space = Space { len: meta.len, seg_len: meta.seg_len };
            ops = blocked;
        }
    }
    let blocks = Blocks { first, ops, epi: spec.epi, wgsl: spec.wgsl, cell: spec.cell };
    let step = Step {
        origin,
        class: SplitClass::Reduce,
        kernel: StepKernel::ReduceBlocks(Box::new(blocks)),
        space,
        binds: Box::default(),
        n_out: 1,
        scalars: scalars.into_boxed_slice(),
        shifted,
        reselected: false,
    };
    let segments = space.slabs();
    let outputs = [&**result];
    if deferred {
        defer(ctx, step, &outputs, &inputs, &reaches, &|_| false, segments, writer)?;
    } else {
        let mut open = Open::immediate();
        open.push(step, &outputs, &inputs, &reaches, &|_| false, segments, writer, 0);
        ctx.stream_counters().immediate_reductions.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        super::run::run(ctx, &mut open)?;
    }
    Ok(())
}

/// Runs a step whose strategy the re-selection hook chose, whole: the strategy lowers the value as it does anywhere else and its
/// values are copied into the step's result.
pub(super) fn run_relowered(ctx: &Context, open: &Open, step: &Step) -> Result<(), LumaError> {
    use crate::jit::structured::strategy::{Lowered, Output, Subject};
    let Origin::Structured { value: Some(value), strategy, params } = &step.origin else { return Err(LumaError::Gpu("a re-selected step without its value".into())) };
    let family = strategy.family();
    let entries = super::plan::entries_for(ctx, family);
    let lower = entries.iter().find(|entry| entry.id == *strategy).and_then(|entry| entry.lower).ok_or_else(|| LumaError::Gpu("a re-selected strategy that does not lower".into()))?;
    let subject = Subject::Comp(value);
    let mut emit = crate::jit::structured::emit::Emit::now(ctx);
    let Lowered::Done(Output::Storage(made)) = lower(&subject, params, &mut emit).map_err(Failure::into_error)? else {
        return Err(LumaError::Unsupported("the re-selected strategy declined".into()));
    };
    let out = open.arrays[step.binds[0].array as usize].as_seg(ctx);
    let src = made.pin(0)?;
    let dst = out.pin_overwrite(0)?;
    ctx.copy(src.buffer(), 0, dst.buffer(), 0, out.segment_byte_len(0))?;
    Ok(())
}

/// The buffers a stage's reductions work in: per `ReduceBlocks` step (in stage order) the states of one slab's blocks and the carry.
pub(super) struct Carries<'a> {
    ctx: &'a Context,
    items: Vec<Carry>,
}

struct Carry {
    /// Block states of a slab, op `j` at `j * blocks ..`.
    blocks: Buffer,
    /// The slab states, op `j` at `j * slabs ..`.
    carry: Buffer,
    merge: Arc<KernelCore>,
    last: Arc<KernelCore>,
    slabs: u32,
    /// Blocks of a full slab.
    full: u32,
}

impl<'a> Carries<'a> {
    pub(super) fn new(ctx: &'a Context, steps: &[Step]) -> Result<Carries<'a>, LumaError> {
        let mut carries = Carries { ctx, items: Vec::new() };
        for step in steps {
            let StepKernel::ReduceBlocks(blocks) = &step.kernel else { continue };
            let count = blocks.ops.as_slice().len() as u64;
            let full = (step.space.seg_len.min(step.space.len) as u64).div_ceil(u64::from(blocks.block())) as u32;
            let slabs = step.space.slabs() as u32;
            let merge = ctx.builtin(Builtin::Reduce(blocks.ops.balanced(), true), blocks.wgsl)?;
            let last = ctx.builtin(Builtin::Reduce(blocks.ops.balanced().with_epilogue(EpiId::of(blocks.epi.as_deref())), true), blocks.wgsl)?;
            let blocks_buffer = ctx.alloc_buffer(count * u64::from(full) * blocks.cell, STORAGE_USAGE)?;
            // Pushed before the next allocation so that a failure gives both back.
            let carry = ctx.alloc_buffer(count * u64::from(slabs) * blocks.cell, STORAGE_USAGE);
            match carry {
                Ok(carry) => carries.items.push(Carry { blocks: blocks_buffer, carry, merge, last, slabs, full }),
                Err(error) => {
                    ctx.recycle(blocks_buffer);
                    return Err(error);
                }
            }
        }
        Ok(carries)
    }

    /// The first pass and the merge of slab `slab` of the step `k`-th of the stage's reductions, whose inputs are `ins`.
    pub(super) fn slab(&self, k: usize, step: &Step, ins: &[SegRef<'_>], slab: usize) -> Result<(), LumaError> {
        let ctx = self.ctx;
        let StepKernel::ReduceBlocks(blocks) = &step.kernel else { unreachable!("a reduction step") };
        let carry = &self.items[k];
        let Space { len, seg_len } = step.space;
        let start = slab * seg_len;
        let n = (start + seg_len).min(len) - start;
        let nblocks = (n as u64).div_ceil(u64::from(blocks.block())) as u32;
        let count = blocks.ops.as_slice().len() as u64;
        let bytes = count * u64::from(carry.full) * blocks.cell;
        match &blocks.first {
            First::Fused { core, key } => reduce_blocks_slab(ctx, core, key, ins, &step.scalars, step.shifted.as_ref(), (len, seg_len, blocks.block()), slab, &carry.blocks, bytes)?,
            First::Buffer { core } => {
                let input = ins[0];
                let pin = input.pin_read(slab)?;
                let mut words = vec![n as u32];
                if blocks.wgsl == "df32" {
                    words.push(0);
                }
                ctx.launch(
                    core,
                    fold_workgroups(ctx, nblocks)?,
                    [n as u32, 1, 1],
                    &[
                        RawArg::write(&carry.blocks, bytes),
                        RawArg::read(pin.buffer(), input.layout().bytes(n)),
                        RawArg::Uniform(bytemuck::cast_slice(&words)),
                    ],
                )?;
            }
        }
        ctx.jit().note_reduce_pass();
        let mut words = vec![nblocks, carry.slabs, slab as u32];
        if blocks.wgsl == "df32" {
            words.push(0);
        }
        ctx.launch(
            &carry.merge,
            [1, 1, 1],
            [nblocks, 1, 1],
            &[
                RawArg::write(&carry.carry, count * u64::from(carry.slabs) * blocks.cell),
                RawArg::read(&carry.blocks, count * u64::from(nblocks) * blocks.cell),
                RawArg::Uniform(bytemuck::cast_slice(&words)),
            ],
        )?;
        ctx.jit().note_reduce_pass();
        Ok(())
    }

    /// After the last slab: folds the slab states of the step `k`-th of the stage's reductions into `out`, with its epilogue.
    pub(super) fn finish(&self, k: usize, step: &Step, out: &SegRef<'_>) -> Result<(), LumaError> {
        let ctx = self.ctx;
        let StepKernel::ReduceBlocks(blocks) = &step.kernel else { unreachable!("a reduction step") };
        let carry = &self.items[k];
        let count = blocks.ops.as_slice().len() as u64;
        let pin = out.pin_overwrite(0)?;
        let mut words = vec![carry.slabs, 1, 0];
        if blocks.wgsl == "df32" {
            words.push(0);
        } else {
            words.extend(blocks.epi.iter().flat_map(|epi| epi.words()));
        }
        ctx.launch(
            &carry.last,
            [1, 1, 1],
            [carry.slabs, 1, 1],
            &[
                RawArg::write(pin.buffer(), out.segment_byte_len(0)),
                RawArg::read(&carry.carry, count * u64::from(carry.slabs) * blocks.cell),
                RawArg::Uniform(bytemuck::cast_slice(&words)),
            ],
        )?;
        ctx.jit().note_reduce_pass();
        ctx.jit().stats.reduce_launches.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(())
    }
}

impl Drop for Carries<'_> {
    fn drop(&mut self) {
        for item in self.items.drain(..) {
            self.ctx.recycle(item.blocks);
            self.ctx.recycle(item.carry);
        }
    }
}

/// The reference the merge kernel is tested against: the balanced binary tree over `states` (state `i` pairs with `i ^ 1`, then the
/// pairs with their neighbours, ...), a node without a right child being its left child.
#[cfg(test)]
pub(crate) fn merge_tree<T: Copy>(states: &[T], combine: &impl Fn(T, T) -> T) -> T {
    let mut level: Vec<T> = states.to_vec();
    while level.len() > 1 {
        level = level.chunks(2).map(|pair| if let [left, right] = pair { combine(*left, *right) } else { pair[0] }).collect();
    }
    level[0]
}
