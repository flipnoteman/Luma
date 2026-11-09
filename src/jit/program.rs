//! Flattening an expression graph into a [`FusedKey`] (the structure a kernel is generated and
//! cached by) plus the runtime data (buffers, scalar values) it runs with.

use std::sync::atomic::Ordering;
use std::sync::Arc;

use crate::array::{Failure, Storage};
use crate::builtin::{ReduceOp, REDUCE_BLOCK};
use crate::context::{RawArg, Scratch, STORAGE_USAGE};
use crate::error::LumaError;
use crate::ops::{fold_workgroups, launch_1d};

use super::{Args, ElemTy, Node, Op, Src};

/// Size of every element the fused kernels handle (`f32` and `u32`).
const ELEMENT_BYTES: u64 = 4;

/// Where an instruction finds a value.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum Ref {
    /// Slot of an operation that has fewer than three operands.
    Unused,
    /// The element of an input buffer.
    Input(u32),
    /// A field of the uniform with the scalars.
    Scalar(u32),
    /// The result of an earlier instruction.
    Tmp(u32),
}

/// One operation of a program; instruction `k` defines `t{k}`.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub(crate) struct Instr {
    pub(crate) op: Op,
    pub(crate) args: [Ref; 3],
}

/// What happens to the value of the last instruction.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum Tail {
    /// It is stored: an elementwise kernel.
    Store,
    /// It is reduced: the first pass of a reduction.
    Reduce(ReduceOp),
}

/// Everything the source of a fused kernel depends on, and nothing it does not (buffers and
/// scalar *values* are not part of it).
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub(crate) struct FusedKey {
    pub(crate) instrs: Vec<Instr>,
    pub(crate) input_tys: Vec<ElemTy>,
    pub(crate) scalar_tys: Vec<ElemTy>,
    pub(crate) tail: Tail,
}

impl FusedKey {
    /// A profile label naming the operations fused into the kernel, for example
    /// `Fused(Mul, Add, Sqrt) f32` or `FusedReduce(Sum: Mul, Add) f32`. Long programs are cut
    /// short.
    pub(crate) fn label(&self) -> String {
        use super::OpKind;
        const SHOWN: usize = 6;
        let mut names: Vec<String> = self
            .instrs
            .iter()
            .take(SHOWN)
            .map(|instr| match instr.op.kind {
                OpKind::Unary(op) => format!("{op:?}"),
                OpKind::Binary(op) => format!("{op:?}"),
                OpKind::Compare(op) => format!("{op:?}"),
                other => format!("{other:?}"),
            })
            .collect();
        if self.instrs.len() > SHOWN {
            names.push(format!("+{} more", self.instrs.len() - SHOWN));
        }
        let ty = self.instrs.last().map_or("f32", |instr| instr.op.out_ty().wgsl());
        let names = names.join(", ");
        match self.tail {
            Tail::Store => format!("Fused({names}) {ty}"),
            Tail::Reduce(op) => format!("FusedReduce({op:?}: {names}) {ty}"),
        }
    }
}

/// A flattened expression ready to run.
struct Fused {
    key: FusedKey,
    inputs: Vec<Arc<Storage>>,
    /// Bit patterns of the scalars, in the order of `key.scalar_tys`.
    scalars: Vec<u32>,
}

/// A pending node below the root of an evaluation, with its operands.
struct Member {
    node: Arc<Node>,
    args: Args,
}

/// Collects the pending nodes below `src` into `members`, operands before the nodes using them.
/// `seen` holds the nodes already collected (each is collected once, so shared nodes are shared in
/// the program too). Evaluated nodes are leaves.
fn collect(
    src: &Src,
    members: &mut Vec<Member>,
    seen: &mut Vec<*const Node>,
) -> Result<(), Failure> {
    let Src::Node(node) = src else { return Ok(()) };
    let key = Arc::as_ptr(node);
    if seen.contains(&key) {
        return Ok(());
    }
    match node.result.get() {
        Some(Ok(_)) => return Ok(()),
        Some(Err(error)) => return Err(Failure::Inherited(Arc::clone(error))),
        None => {}
    }
    let Some(args) = node.operands() else {
        // Another thread is evaluating it right now; wait for its buffer.
        return match node.force() {
            Ok(_) => Ok(()),
            Err(error) => Err(Failure::Inherited(Arc::clone(error))),
        };
    };
    for arg in args.iter().take(node.op().kind.arity()) {
        collect(arg, members, seen)?;
    }
    seen.push(key);
    members.push(Member { node: Arc::clone(node), args });
    Ok(())
}

#[derive(Default)]
struct Flattener {
    instrs: Vec<Instr>,
    input_tys: Vec<ElemTy>,
    inputs: Vec<Arc<Storage>>,
    scalar_tys: Vec<ElemTy>,
    scalars: Vec<u32>,
    /// Where the value of every member emitted so far lives.
    emitted: Vec<(*const Node, Ref)>,
}

impl Flattener {
    /// Emits the instruction for `op` on `args`, whose pending operands were emitted before, and
    /// returns where the result lives.
    fn emit(&mut self, op: Op, args: &[Src; 3]) -> Result<Ref, Failure> {
        let mut refs = [Ref::Unused; 3];
        for (index, (slot, arg)) in refs.iter_mut().zip(args).enumerate().take(op.kind.arity()) {
            *slot = self.operand(arg, op.operand_ty(index))?;
        }
        let at = u32::try_from(self.instrs.len()).unwrap_or(u32::MAX);
        self.instrs.push(Instr { op, args: refs });
        Ok(Ref::Tmp(at))
    }

    fn operand(&mut self, src: &Src, ty: ElemTy) -> Result<Ref, Failure> {
        match src {
            Src::Scalar(scalar) => {
                let at = u32::try_from(self.scalars.len()).unwrap_or(u32::MAX);
                self.scalar_tys.push(scalar.ty);
                self.scalars.push(scalar.bits);
                Ok(Ref::Scalar(at))
            }
            Src::Buffer(storage) => Ok(self.input(storage, ty)),
            Src::Node(node) => {
                let key = Arc::as_ptr(node);
                if let Some(&(_, found)) = self.emitted.iter().find(|(seen, _)| *seen == key) {
                    return Ok(found);
                }
                // Not a member: it was evaluated when the members were collected.
                match node.force() {
                    Ok(storage) => Ok(self.input(storage, ty)),
                    Err(error) => Err(Failure::Inherited(Arc::clone(error))),
                }
            }
        }
    }

    fn input(&mut self, storage: &Arc<Storage>, ty: ElemTy) -> Ref {
        let at = match self.inputs.iter().position(|known| Arc::ptr_eq(known, storage)) {
            Some(at) => at,
            None => {
                self.inputs.push(Arc::clone(storage));
                self.input_tys.push(ty);
                self.inputs.len() - 1
            }
        };
        Ref::Input(u32::try_from(at).unwrap_or(u32::MAX))
    }

    fn finish(self, tail: Tail) -> Fused {
        Fused {
            key: FusedKey {
                instrs: self.instrs,
                input_tys: self.input_tys,
                scalar_tys: self.scalar_tys,
                tail,
            },
            inputs: self.inputs,
            scalars: self.scalars,
        }
    }
}

/// Flattens the expression of `root` (with operands `args`) into a program.
///
/// A pending node that other pending operations (outside this expression) also use is evaluated
/// first instead of being recomputed by each of them; shared nodes *within* the expression
/// are one instruction.
fn flatten(root: &Arc<Node>, args: &[Src; 3], tail: Tail) -> Result<Fused, Failure> {
    let arity = root.op().kind.arity();
    let members = loop {
        let mut members = Vec::new();
        let mut seen = Vec::new();
        for arg in args.iter().take(arity) {
            collect(arg, &mut members, &mut seen)?;
        }
        seen.push(Arc::as_ptr(root));
        let shared: Vec<Arc<Node>> = members
            .iter()
            .filter(|member| member.node.has_consumer_outside(&seen))
            .map(|member| Arc::clone(&member.node))
            .collect();
        if shared.is_empty() {
            break members;
        }
        for node in shared {
            if let Err(error) = node.force() {
                return Err(Failure::Inherited(Arc::clone(error)));
            }
        }
    };

    let mut flattener = Flattener::default();
    for member in &members {
        let at = flattener.emit(member.node.op(), &member.args)?;
        flattener.emitted.push((Arc::as_ptr(&member.node), at));
    }
    flattener.emit(root.op(), args)?;
    for member in &members {
        member.node.fused.store(true, Ordering::Relaxed);
    }
    if matches!(tail, Tail::Reduce(_)) {
        root.fused.store(true, Ordering::Relaxed);
    }
    Ok(flattener.finish(tail))
}

fn len_u32(len: usize) -> Result<u32, LumaError> {
    u32::try_from(len).map_err(|_| LumaError::InvalidShape(format!("{len} elements exceed u32")))
}

/// Evaluates `node`: one fused kernel computing its whole expression into a new buffer.
pub(crate) fn evaluate(node: &Arc<Node>, args: Args) -> Result<Storage, Failure> {
    let ctx = node.context();
    let fused = flatten(node, &args, Tail::Store)?;
    drop(args);

    let n = len_u32(node.len())?;
    let bytes = u64::from(n) * ELEMENT_BYTES;
    let Fused { key, inputs, scalars } = fused;
    let core = ctx.jit().kernel(ctx, key)?;

    let out = Storage::new(ctx, ctx.alloc_buffer(bytes, STORAGE_USAGE)?);
    let mut raw: Vec<RawArg<'_>> = Vec::with_capacity(inputs.len() + 2);
    raw.push(RawArg::write(out.buffer(), bytes));
    raw.extend(inputs.iter().map(|input| RawArg::read(input.buffer(), bytes)));
    if !scalars.is_empty() {
        raw.push(RawArg::Uniform(bytemuck::cast_slice(&scalars)));
    }
    launch_1d(ctx, &core, n, &raw)?;
    ctx.jit().stats.elementwise_launches.fetch_add(1, Ordering::Relaxed);
    Ok(out)
}

/// Runs the first pass of a reduction directly over the expression behind `node`, without
/// materializing it: every workgroup evaluates `REDUCE_BLOCK` elements and reduces them. Returns
/// the buffer of partial results and how many there are, or `None` if the node has already been
/// evaluated (reduce its buffer instead).
pub(crate) fn reduce_first_pass(
    node: &Arc<Node>,
    op: ReduceOp,
    scratch: &mut Scratch<'_>,
) -> Result<Option<(wgpu::Buffer, u32)>, Failure> {
    if !node.is_pending() {
        return Ok(None);
    }
    let Some(args) = node.operands() else { return Ok(None) };
    let ctx = node.context();
    let fused = flatten(node, &args, Tail::Reduce(op))?;
    drop(args);

    let n = len_u32(node.len())?;
    let bytes = u64::from(n) * ELEMENT_BYTES;
    let groups = n.div_ceil(REDUCE_BLOCK);
    let Fused { key, inputs, scalars } = fused;
    let core = ctx.jit().kernel(ctx, key)?;

    let out = scratch.take(u64::from(groups) * ELEMENT_BYTES)?;
    let mut params = Vec::with_capacity(scalars.len() + 1);
    params.push(n);
    params.extend_from_slice(&scalars);
    let mut raw: Vec<RawArg<'_>> = Vec::with_capacity(inputs.len() + 2);
    raw.push(RawArg::write(&out, u64::from(groups) * ELEMENT_BYTES));
    raw.extend(inputs.iter().map(|input| RawArg::read(input.buffer(), bytes)));
    raw.push(RawArg::Uniform(bytemuck::cast_slice(&params)));
    ctx.launch(&core, fold_workgroups(ctx, groups)?, [n, 1, 1], &raw)?;
    ctx.jit().stats.reduce_launches.fetch_add(1, Ordering::Relaxed);
    Ok(Some((out, groups)))
}
