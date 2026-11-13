//! [`crate::stack`]: equal-length expressions written interleaved into one buffer by one fused
//! kernel. Every invocation computes one element of every part and stores them next to each
//! other (a stored layout, not a select between the parts: each output cell is computed once,
//! by the invocation that writes it).
//!
//! Parts that cannot be written like that (inputs split over several buffers, a result larger
//! than a buffer) are left to the caller, which joins them lazily.

use std::ops::Range;
use std::panic::Location;
use std::sync::atomic::Ordering;
use std::sync::Arc;

use crate::array::{Failure, Storage};
use crate::builtin::UnOp;
use crate::context::Context;
use crate::error::LumaError;

use super::program::{flatten, Tail};
use super::{launch, lock, Args, ElemTy, Node, Op, OpKind, Src};

/// Writes `parts` (each `len` elements of `ty`) interleaved in blocks of `block` elements into a
/// new buffer (see [`Tail::Stack`]) and returns it, or `None` if the kernel cannot run on these
/// operands (the caller then joins the parts lazily).
///
/// The parts are fused into the kernels that write them, `LUMA_FUSION_MAX_LIVE / 4` of them to a
/// kernel (all of them for the usual number), each reading the buffers its expressions read.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run(
    ctx: &Context,
    parts: &[Src],
    ty: ElemTy,
    len: usize,
    block: usize,
    name: &'static str,
    location: &'static Location<'static>,
) -> Result<Option<Arc<Storage>>, Failure> {
    let total = parts.len();
    let (Some(out_len), Ok(block32)) = (total.checked_mul(len), u32::try_from(block)) else { return Ok(None) };
    let fits = u32::try_from(out_len).is_ok()
        && u32::try_from(total).is_ok()
        && (out_len as u64).saturating_mul(ty.bytes()) <= ctx.segment_bytes();
    if !fits || block == 0 || len == 0 {
        return Ok(None);
    }
    let base = Arc::new(Storage::new(ctx, out_len, ty.bytes() as usize));
    let jit = ctx.jit();
    let group = (jit.max_live(ctx.tuning()) / 4).max(1);
    let mut work: Vec<Range<usize>> = (0..total).step_by(group).map(|first| first..(first + group).min(total)).rev().collect();
    while let Some(range) = work.pop() {
        let tail = Tail::Stack { total: total as u32, first: range.start as u32, block: block32 };
        match run_group(ctx, &parts[range.clone()], ty, len, tail, &base, name, location)? {
            Group::Done => {}
            Group::Unsupported => return Ok(None),
            Group::TooManyInputs => {
                let middle = range.start + range.len() / 2;
                work.push(middle..range.end);
                work.push(range.start..middle);
            }
        }
    }
    super::check_finite(ctx, &base, ty)?;
    Ok(Some(base))
}

enum Group {
    Done,
    /// The program reads an array split over several buffers.
    Unsupported,
    /// A group of more than one part reads more buffers than a kernel binds: split it.
    TooManyInputs,
}

/// One kernel writing `parts` into `base`.
#[allow(clippy::too_many_arguments)]
fn run_group(
    ctx: &Context,
    parts: &[Src],
    ty: ElemTy,
    len: usize,
    tail: Tail,
    base: &Arc<Storage>,
    name: &'static str,
    location: &'static Location<'static>,
) -> Result<Group, Failure> {
    let jit = ctx.jit();
    let mut stores: Vec<Arc<Node>> = Vec::with_capacity(parts.len());
    for part in parts {
        let args = [part.clone(), Src::unused(), Src::unused()];
        let store = super::create(ctx, len, Op::new(OpKind::Unary(UnOp::Id), ty), args, name, location)?;
        // Only this function evaluates a store (into the buffer of the stack): marked as fused,
        // no multi-output evaluation picks it up as a live array.
        store.fused.store(true, Ordering::Relaxed);
        stores.push(store);
    }
    let bases = [Arc::clone(base)];
    let result = (|| {
        let mut forced = false;
        loop {
            let mut roots: Vec<(Arc<Node>, Args)> = Vec::with_capacity(stores.len());
            for store in &stores {
                let Some(args) = store.operands() else {
                    return Err(Failure::New(LumaError::Gpu("stack evaluated twice".into())));
                };
                roots.push((Arc::clone(store), args));
            }
            let fused = flatten(&roots, tail)?;
            drop(roots);
            if fused.inputs.iter().any(|input| input.segments() != 1) {
                return Ok(Group::Unsupported);
            }
            if launch::mapped_bindings(&fused.inputs, &bases) <= jit.max_inputs() + 1 {
                launch::run_mapped(ctx, fused, &bases, len)?;
                return Ok(Group::Done);
            }
            drop(fused);
            if stores.len() > 1 {
                return Ok(Group::TooManyInputs);
            }
            // One part reading too many buffers: its expression is computed first.
            if forced {
                return Err(Failure::New(LumaError::Chunked("a stack of an expression that reads this many buffers at once")));
            }
            forced = true;
            if let Src::Node(node) = &parts[0]
                && let Err(error) = node.force()
            {
                return Err(Failure::Inherited(Arc::clone(error)));
            }
        }
    })();
    for store in &stores {
        lock(&store.args).take();
    }
    result
}
