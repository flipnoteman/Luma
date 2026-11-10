//! Flattening an expression graph into a [`FusedKey`] (the structure a kernel is generated and
//! cached by) plus the runtime data (buffers, scalar values) it runs with.

use std::sync::atomic::Ordering;
use std::sync::Arc;

use crate::array::{Failure, Storage};
use crate::builtin::ReduceList;
use crate::context::{RawArg, Scratch, STORAGE_USAGE};
use crate::error::LumaError;
use crate::ops::{fold_workgroups, launch_1d};

use super::{Args, ElemTy, Node, Op, OpKind, Src};

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
    /// A sub-program, evaluated at the element a shift reads (operand 0 of a shift only).
    Sub(u32),
}

/// One operation of a program; instruction `k` defines `t{k}`.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub(crate) struct Instr {
    pub(crate) op: Op,
    pub(crate) args: [Ref; 3],
}

/// What happens to the values the program computes.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum Tail {
    /// The outputs are stored: an elementwise kernel.
    Store,
    /// The last instruction is reduced, once per op: the first pass of the reductions.
    Reduce(ReduceList),
}

/// Everything the source of a fused kernel depends on, and nothing it does not (buffers and
/// scalar *values* are not part of it).
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub(crate) struct FusedKey {
    pub(crate) instrs: Vec<Instr>,
    pub(crate) input_tys: Vec<ElemTy>,
    pub(crate) scalar_tys: Vec<ElemTy>,
    /// The instructions whose values a store kernel writes, one output buffer each.
    pub(crate) outputs: Vec<u32>,
    /// Programs of expressions read through shifts; sub-program `k` defines `sub{k}(i)`, its
    /// last instruction being the value.
    pub(crate) subs: Vec<Vec<Instr>>,
    pub(crate) tail: Tail,
}

impl FusedKey {
    /// A profile label naming the operations fused into the kernel, for example
    /// `Fused(Mul, Add, Sqrt) f32` or `FusedReduce(Sum: Mul, Add) f32`. Long programs are cut
    /// short.
    pub(crate) fn label(&self) -> String {
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
            Tail::Store if self.outputs.len() > 1 => format!("Fused[{}]({names}) {ty}", self.outputs.len()),
            Tail::Store => format!("Fused({names}) {ty}"),
            Tail::Reduce(ops) => {
                let ops: Vec<String> = ops.as_slice().iter().map(|op| format!("{op:?}")).collect();
                format!("FusedReduce({}: {names}) {ty}", ops.join("+"))
            }
        }
    }
}

/// A flattened expression ready to run.
pub(super) struct Fused {
    pub(super) key: FusedKey,
    pub(super) inputs: Vec<Arc<Storage>>,
    /// Bit patterns of the scalars, in the order of `key.scalar_tys`.
    pub(super) scalars: Vec<u32>,
    /// Nodes computed inside this program (not as outputs of their own); marked fused once run.
    fused_nodes: Vec<Arc<Node>>,
}

impl Fused {
    pub(super) fn mark_fused(&self) {
        for node in &self.fused_nodes {
            node.fused.store(true, Ordering::Relaxed);
        }
    }
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
    for (index, arg) in args.iter().enumerate().take(node.op().kind.arity()) {
        // A shifted expression is a sub-program of its own (see `Flattener::sub`).
        if index == 0 && matches!(node.op().kind, OpKind::Shift(_)) {
            continue;
        }
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
    /// Programs of the expressions read through shifts, and the node of each.
    subs: Vec<Vec<Instr>>,
    sub_nodes: Vec<*const Node>,
    fused_nodes: Vec<Arc<Node>>,
}

impl Flattener {
    /// Emits the instruction for `op` on `args`, whose pending operands were emitted before, and
    /// returns where the result lives.
    fn emit(&mut self, op: Op, args: &[Src; 3]) -> Result<Ref, Failure> {
        let mut refs = [Ref::Unused; 3];
        for (index, (slot, arg)) in refs.iter_mut().zip(args).enumerate().take(op.kind.arity()) {
            *slot = match (op.kind, index, arg) {
                (OpKind::Shift(_), 0, Src::Node(node)) if node.is_pending() => self.sub(node)?,
                _ => self.operand(arg, op.operand_ty(index))?,
            };
        }
        let at = u32::try_from(self.instrs.len()).unwrap_or(u32::MAX);
        self.instrs.push(Instr { op, args: refs });
        Ok(Ref::Tmp(at))
    }

    /// The expression of `node` as a sub-program evaluated at another element (the operand of a
    /// shift), recomputed instead of materialized. Shares the inputs and scalars of the program.
    fn sub(&mut self, node: &Arc<Node>) -> Result<Ref, Failure> {
        let key = Arc::as_ptr(node);
        if let Some(at) = self.sub_nodes.iter().position(|&seen| seen == key) {
            return Ok(Ref::Sub(at as u32));
        }
        let Some(args) = node.operands() else {
            return self.operand(&Src::Node(Arc::clone(node)), node.op().out_ty());
        };
        let (mut members, mut seen) = (Vec::new(), vec![key]);
        for arg in args.iter().take(node.op().kind.arity()) {
            collect(arg, &mut members, &mut seen)?;
        }
        let outer_instrs = std::mem::take(&mut self.instrs);
        let outer_emitted = std::mem::take(&mut self.emitted);
        let result = (|| {
            for member in &members {
                let at = self.emit(member.node.op(), &member.args)?;
                self.emitted.push((Arc::as_ptr(&member.node), at));
            }
            self.emit(node.op(), &args)
        })();
        let instrs = std::mem::replace(&mut self.instrs, outer_instrs);
        self.emitted = outer_emitted;
        result?;
        self.fused_nodes.extend(members.into_iter().map(|m| m.node));
        self.fused_nodes.push(Arc::clone(node));
        self.subs.push(instrs);
        self.sub_nodes.push(key);
        Ok(Ref::Sub(u32::try_from(self.subs.len() - 1).unwrap_or(u32::MAX)))
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

    fn finish(self, tail: Tail, outputs: Vec<u32>) -> Fused {
        Fused {
            key: FusedKey {
                instrs: self.instrs,
                input_tys: self.input_tys,
                scalar_tys: self.scalar_tys,
                subs: self.subs,
                outputs,
                tail,
            },
            inputs: self.inputs,
            scalars: self.scalars,
            fused_nodes: self.fused_nodes,
        }
    }
}

/// Flattens the expressions of `roots` (each with its operands) into one program computing all
/// of them; a root that another root uses is computed once. Returns the program with the
/// instruction of every root, in order.
///
/// A pending node that other pending operations (outside these expressions) also use is
/// evaluated first instead of being recomputed by each of them; shared nodes *within* them are
/// one instruction.
pub(super) fn flatten(roots: &[(Arc<Node>, Args)], tail: Tail) -> Result<Fused, Failure> {
    flatten_inner(roots, tail, None)
}

pub(super) fn flatten_stencil(
    node: &Arc<Node>,
    args: Args,
    probe_id: u64,
) -> Result<Fused, Failure> {
    flatten_inner(&[(Arc::clone(node), args)], Tail::Store, Some(probe_id))
}

fn flatten_inner(
    roots: &[(Arc<Node>, Args)],
    tail: Tail,
    probe_id: Option<u64>,
) -> Result<Fused, Failure> {
    let members = loop {
        let mut members = Vec::new();
        let mut seen = Vec::new();
        for (root, args) in roots {
            for (index, arg) in args.iter().enumerate().take(root.op().kind.arity()) {
                if index == 0 && matches!(root.op().kind, OpKind::Shift(_)) {
                    continue;
                }
                collect(arg, &mut members, &mut seen)?;
            }
        }
        for (root, args) in roots {
            if !seen.contains(&Arc::as_ptr(root)) {
                seen.push(Arc::as_ptr(root));
                members.push(Member { node: Arc::clone(root), args: args.clone() });
            }
        }
        let mut sub_members = Vec::new();
        let mut all_seen = seen.clone();
        for member in &members {
            if matches!(member.node.op().kind, OpKind::Shift(_)) {
                if let Src::Node(sub_root) = &member.args[0] {
                    collect(&Src::Node(Arc::clone(sub_root)), &mut sub_members, &mut all_seen)?;
                }
            }
        }
        let is_root = |node: &Arc<Node>| roots.iter().any(|(root, _)| Arc::ptr_eq(root, node));
        let shared: Vec<Arc<Node>> = members
            .iter()
            .chain(&sub_members)
            .filter(|member| !is_root(&member.node) && member.node.has_consumer_outside(&all_seen))
            .map(|member| Arc::clone(&member.node))
            .collect();
        if shared.is_empty() {
            break members;
        }
        if let Some(id) = probe_id {
            if shared.iter().any(|node| node.depends_on(id)) {
                return Err(Failure::New(LumaError::Gpu("stencil probe shared outside step".into())));
            }
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
    let mut outputs = Vec::with_capacity(roots.len());
    for (root, _) in roots {
        let found = flattener.emitted.iter().find(|(node, _)| *node == Arc::as_ptr(root));
        match found {
            Some(&(_, Ref::Tmp(at))) => outputs.push(at),
            _ => return Err(Failure::New(LumaError::Gpu("fused root without an instruction".into()))),
        }
    }
    for member in &members {
        let root = roots.iter().any(|(root, _)| Arc::ptr_eq(root, &member.node));
        if !root || matches!(tail, Tail::Reduce(_)) {
            flattener.fused_nodes.push(Arc::clone(&member.node));
        }
    }
    Ok(flattener.finish(tail, outputs))
}

fn len_u32(len: usize) -> Result<u32, LumaError> {
    u32::try_from(len).map_err(|_| LumaError::InvalidShape(format!("{len} elements exceed u32")))
}

/// Evaluates `node`: one fused kernel computing its whole expression into a new buffer.
pub(crate) fn evaluate(node: &Arc<Node>, args: Args) -> Result<Storage, Failure> {
    let roots = [(Arc::clone(node), args)];
    let fused = flatten(&roots, Tail::Store)?;
    drop(roots);
    let mut outputs = run_store(node.context(), node.len(), fused)?;
    outputs.pop().ok_or_else(|| Failure::New(LumaError::Gpu("fused kernel without output".into())))
}

/// Evaluates the pending ones of `nodes` (all of the same length) in one fused kernel with one
/// output per node. Failures are stored in the nodes, as [`Node::force`] does.
pub(crate) fn evaluate_group(nodes: &[Arc<Node>]) {
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

/// Runs the store kernel of `fused` over `len` elements, one new buffer per output.
fn run_store(ctx: &crate::context::Context, len: usize, fused: Fused) -> Result<Vec<Storage>, Failure> {
    let n = len_u32(len)?;
    let bytes = u64::from(n) * ELEMENT_BYTES;
    fused.mark_fused();
    let Fused { key, inputs, scalars, .. } = fused;
    let count = key.outputs.len();
    let core = ctx.jit().kernel(ctx, key)?;

    let mut outputs = Vec::with_capacity(count);
    for _ in 0..count {
        outputs.push(Storage::new(ctx, ctx.alloc_buffer(bytes, STORAGE_USAGE)?));
    }
    let mut raw: Vec<RawArg<'_>> = Vec::with_capacity(count + inputs.len() + 1);
    raw.extend(outputs.iter().map(|out| RawArg::write(out.buffer(), bytes)));
    raw.extend(inputs.iter().map(|input| RawArg::read(input.buffer(), bytes)));
    if !scalars.is_empty() {
        raw.push(RawArg::Uniform(bytemuck::cast_slice(&scalars)));
    }
    launch_1d(ctx, &core, n, &raw)?;
    ctx.jit().stats.elementwise_launches.fetch_add(1, Ordering::Relaxed);
    Ok(outputs)
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
) -> Result<Option<(wgpu::Buffer, u32)>, Failure> {
    if !node.is_pending() {
        return Ok(None);
    }
    let Some(args) = node.operands() else { return Ok(None) };
    let ctx = node.context();
    let roots = [(Arc::clone(node), args)];
    let fused = flatten(&roots, Tail::Reduce(ops))?;
    drop(roots);

    let n = len_u32(node.len())?;
    let bytes = u64::from(n) * ELEMENT_BYTES;
    let groups = ctx.tuning().reduce_groups(n);
    let out_bytes = u64::from(groups) * ELEMENT_BYTES * ops.as_slice().len() as u64;
    fused.mark_fused();
    let Fused { key, inputs, scalars, .. } = fused;
    let core = ctx.jit().kernel(ctx, key)?;

    let out = scratch.take(out_bytes)?;
    let mut params = Vec::with_capacity(scalars.len() + 1);
    params.push(n);
    params.extend_from_slice(&scalars);
    let mut raw: Vec<RawArg<'_>> = Vec::with_capacity(inputs.len() + 2);
    raw.push(RawArg::write(&out, out_bytes));
    raw.extend(inputs.iter().map(|input| RawArg::read(input.buffer(), bytes)));
    raw.push(RawArg::Uniform(bytemuck::cast_slice(&params)));
    ctx.launch(&core, fold_workgroups(ctx, groups)?, [n, 1, 1], &raw)?;
    ctx.jit().stats.reduce_launches.fetch_add(1, Ordering::Relaxed);
    Ok(Some((out, groups)))
}
