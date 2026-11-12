//! Flattening an expression graph into a [`FusedKey`] (the structure a kernel is generated and
//! cached by) plus the runtime data (buffers, scalar values) it runs with.

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
use super::{lock, Args, ElemTy, Node, Op, OpKind, Scalar, ShiftKind, Src};

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
    /// Input `k` is not bound separately: it is the buffer of output 0, read at element `i`
    /// before being overwritten (see [`in_place_eligible`]).
    pub(crate) in_place: Option<u32>,
    /// Per input, how many segments of it the program reads through shifts (0 = never): the
    /// program is the segmented variant of the generated kernel, run once per segment of arrays
    /// spanning several buffers. Empty when the program runs unsegmented or has no shift.
    pub(crate) spans: Vec<u8>,
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
                in_place: None,
                spans: Vec::new(),
            },
            inputs: self.inputs,
            scalars: self.scalars,
            fused_nodes: self.fused_nodes,
        }
    }
}

/// What an operand of an instruction being emitted by [`Plain`] stands for.
enum Resolved {
    /// Slot beyond the operation's arity.
    Unused,
    Scalar(Scalar),
    /// A buffer that is not an input of the program yet when this was resolved.
    Buffer(Arc<Storage>),
    /// Input `k` of the program.
    Input(u32),
    /// The result of instruction `k`.
    Tmp(u32),
    /// A pending node whose instruction is not emitted yet.
    Pending(Arc<Node>),
}

/// Why [`Plain`] gave up.
enum Abandoned {
    Failed(Failure),
    /// A shift reads a pending expression: it needs sub-programs, see [`Flattener::sub`].
    Shift,
}

impl From<Failure> for Abandoned {
    fn from(failure: Failure) -> Self {
        Abandoned::Failed(failure)
    }
}

/// Flattens the expression of one root without shifted sub-expressions in a single walk: the
/// instruction of a node is emitted right after those of its operands, which is the program
/// [`collect`] and [`Flattener`] build in two passes. Operands are looked up by their position
/// in `seen` (instruction `k` is member `k`), and a node's operands are borrowed under its lock
/// instead of being copied out, which saves most of the reference counting.
struct Plain {
    flat: Flattener,
    /// Addresses of the nodes emitted so far, in instruction order.
    seen: Vec<*const Node>,
    /// The nodes below the root, in instruction order.
    members: Vec<Arc<Node>>,
    /// How many emitted nodes read member `k`: with [`Node::consumers`] this tells that nothing
    /// outside reads it, without looking at its readers.
    readers: Vec<u32>,
}

impl Plain {
    fn with_capacity(n: usize) -> Self {
        let mut flat = Flattener::default();
        flat.instrs.reserve(n);
        Plain {
            flat,
            seen: Vec::with_capacity(n),
            members: Vec::with_capacity(n),
            readers: Vec::with_capacity(n),
        }
    }

    /// What `arg` is, without recursing.
    fn resolve(&self, arg: &Src, shift_operand: bool) -> Result<Resolved, Abandoned> {
        Ok(match arg {
            Src::Scalar(scalar) => Resolved::Scalar(*scalar),
            Src::Buffer(storage) => self.buffer(storage),
            Src::Node(node) => {
                let key = Arc::as_ptr(node);
                if let Some(at) = self.seen.iter().position(|&seen| seen == key) {
                    return Ok(Resolved::Tmp(at as u32));
                }
                match node.result.get() {
                    Some(Ok(storage)) => self.buffer(storage),
                    Some(Err(error)) => return Err(Failure::Inherited(Arc::clone(error)).into()),
                    None if shift_operand => return Err(Abandoned::Shift),
                    None => Resolved::Pending(Arc::clone(node)),
                }
            }
        })
    }

    fn buffer(&self, storage: &Arc<Storage>) -> Resolved {
        match self.flat.inputs.iter().position(|known| Arc::ptr_eq(known, storage)) {
            Some(at) => Resolved::Input(at as u32),
            None => Resolved::Buffer(Arc::clone(storage)),
        }
    }

    /// What the operands of `node` are: those `given`, else its own. `None` if it has none (another
    /// thread is evaluating it). Looks under its lock and takes nothing but references out.
    fn operands(&self, node: &Node, given: Option<&Args>) -> Result<Option<[Resolved; 3]>, Abandoned> {
        let op = node.op();
        let resolve_all = |args: &Args| {
            let mut operands = [Resolved::Unused, Resolved::Unused, Resolved::Unused];
            for (index, (slot, arg)) in operands.iter_mut().zip(args).enumerate().take(op.kind.arity()) {
                *slot = self.resolve(arg, index == 0 && matches!(op.kind, OpKind::Shift(_)))?;
            }
            Ok(Some(operands))
        };
        match given {
            Some(args) => resolve_all(args),
            None => lock(&node.args).as_ref().map_or(Ok(None), resolve_all),
        }
    }

    /// Emits the instruction of the pending `node` with the operands `given` (its own if `None`)
    /// after those of the nodes it reads, and says where its value lives. A node that another
    /// thread is evaluating is waited for and becomes a buffer. The root is `given` its operands
    /// and not kept as a member.
    fn visit(&mut self, node: Arc<Node>, given: Option<&Args>) -> Result<Resolved, Abandoned> {
        let Some(mut operands) = self.operands(&node, given)? else {
            // Another thread is evaluating it right now; wait for its buffer.
            return match node.force() {
                Ok(storage) => Ok(Resolved::Buffer(Arc::clone(storage))),
                Err(error) => Err(Failure::Inherited(Arc::clone(error)).into()),
            };
        };
        for slot in &mut operands {
            if let Resolved::Pending(operand) = slot {
                let operand = Arc::clone(operand);
                // The same node twice (`x * x`) or below an earlier operand was emitted meanwhile.
                let key = Arc::as_ptr(&operand);
                *slot = match self.seen.iter().position(|&seen| seen == key) {
                    Some(at) => Resolved::Tmp(at as u32),
                    None => self.visit(operand, None)?,
                };
            }
        }
        self.emit(node, &operands, given.is_none())
    }

    /// Appends the instruction of `node`, whose operands are all emitted (`member`: keep the node).
    fn emit(&mut self, node: Arc<Node>, operands: &[Resolved; 3], member: bool) -> Result<Resolved, Abandoned> {
        let op = node.op();
        let mut refs = [Ref::Unused; 3];
        for (index, (slot, operand)) in refs.iter_mut().zip(operands).enumerate().take(op.kind.arity()) {
            let ty = op.operand_ty(index);
            *slot = match operand {
                Resolved::Scalar(scalar) => self.flat.operand(&Src::Scalar(*scalar), ty)?,
                Resolved::Buffer(storage) => self.flat.input(storage, ty),
                Resolved::Input(at) => Ref::Input(*at),
                Resolved::Tmp(at) => {
                    let repeated = operands.iter().take(index).any(|earlier| matches!(earlier, Resolved::Tmp(k) if k == at));
                    if !repeated && let Some(count) = self.readers.get_mut(*at as usize) {
                        *count += 1;
                    }
                    Ref::Tmp(*at)
                }
                Resolved::Unused | Resolved::Pending(_) => Ref::Unused,
            };
        }
        let at = u32::try_from(self.flat.instrs.len()).unwrap_or(u32::MAX);
        self.flat.instrs.push(Instr { op, args: refs });
        self.seen.push(Arc::as_ptr(&node));
        if member {
            self.members.push(node);
            self.readers.push(0);
        }
        Ok(Resolved::Tmp(at))
    }
}

/// [`flatten_inner`] for one root without shifted sub-expressions, in one walk (see [`Plain`]).
/// `None` if the expression has one after all.
fn flatten_plain(root: &Arc<Node>, args: &Args, tail: Tail) -> Result<Option<Fused>, Failure> {
    loop {
        let mut plain = Plain::with_capacity(root.size());
        let outcome = plain.visit(Arc::clone(root), Some(args));
        match outcome {
            Ok(Resolved::Tmp(_)) => {}
            Ok(_) => return Err(Failure::New(LumaError::Gpu("fused root without an instruction".into()))),
            Err(Abandoned::Failed(failure)) => return Err(failure),
            Err(Abandoned::Shift) => return Ok(None),
        }
        let Plain { mut flat, seen, members, readers } = plain;
        // A node that something outside also reads is evaluated first instead of being
        // recomputed by both. Most have all their readers inside, which is known without
        // looking at them.
        let shared: Vec<Arc<Node>> = members
            .iter()
            .zip(&readers)
            .filter(|&(member, &inside)| {
                member.consumers.load(Ordering::Relaxed) > inside && member.has_consumer_outside(&seen)
            })
            .map(|(member, _)| Arc::clone(member))
            .collect();
        if !shared.is_empty() {
            for node in shared {
                if let Err(error) = node.force() {
                    return Err(Failure::Inherited(Arc::clone(error)));
                }
            }
            continue;
        }
        let outputs = vec![u32::try_from(flat.instrs.len() - 1).unwrap_or(u32::MAX)];
        flat.fused_nodes = members;
        if matches!(tail, Tail::Reduce(_)) {
            flat.fused_nodes.push(Arc::clone(root));
        }
        return Ok(Some(flat.finish(tail, outputs)));
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
    flatten_inner(roots, tail, &[])
}

pub(super) fn flatten_stencil(
    node: &Arc<Node>,
    args: Args,
    probe_id: u64,
) -> Result<Fused, Failure> {
    flatten_inner(&[(Arc::clone(node), args)], Tail::Store, &[probe_id])
}

fn flatten_inner(
    roots: &[(Arc<Node>, Args)],
    tail: Tail,
    probes: &[u64],
) -> Result<Fused, Failure> {
    if let [(root, args)] = roots
        && probes.is_empty()
        && !root.recomputes
        && let Some(fused) = flatten_plain(root, args, tail)?
    {
        return Ok(fused);
    }
    let expected: usize = roots.iter().map(|(root, _)| root.size()).sum();
    let members = loop {
        let mut members = Vec::with_capacity(expected);
        let mut seen = Vec::with_capacity(expected);
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
        // Only expressions read through shifts add to what was seen.
        let mut with_subs: Option<Vec<*const Node>> = None;
        for member in &members {
            if matches!(member.node.op().kind, OpKind::Shift(_))
                && let Src::Node(sub_root) = &member.args[0] {
                    let all_seen = with_subs.get_or_insert_with(|| seen.clone());
                    collect(&Src::Node(Arc::clone(sub_root)), &mut sub_members, all_seen)?;
                }
        }
        let all_seen = with_subs.as_deref().unwrap_or(&seen);
        let is_root = |node: &Arc<Node>| roots.iter().any(|(root, _)| Arc::ptr_eq(root, node));
        let shared: Vec<Arc<Node>> = members
            .iter()
            .chain(&sub_members)
            .filter(|member| !is_root(&member.node) && member.node.has_consumer_outside(all_seen))
            .map(|member| Arc::clone(&member.node))
            .collect();
        if shared.is_empty() {
            break members;
        }
        if shared.iter().any(|node| probes.iter().any(|&id| node.depends_on(id))) {
            return Err(Failure::New(LumaError::Gpu("stencil probe shared outside step".into())));
        }
        for node in shared {
            if let Err(error) = node.force() {
                return Err(Failure::Inherited(Arc::clone(error)));
            }
        }
    };

    let mut flattener = Flattener::default();
    flattener.instrs.reserve(members.len());
    flattener.emitted.reserve(members.len());
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
    flattener.fused_nodes.reserve(members.len());
    for member in members {
        let root = roots.iter().any(|(root, _)| Arc::ptr_eq(root, &member.node));
        if !root || matches!(tail, Tail::Reduce(_)) {
            flattener.fused_nodes.push(member.node);
        }
    }
    Ok(flattener.finish(tail, outputs))
}

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
    let key = &fused.key;
    let shifts = key.instrs.iter().any(|instr| matches!(instr.op.kind, OpKind::Shift(_)));
    let all_f32 = key.outputs.iter().all(|&at| key.instrs.get(at as usize).is_some_and(|i| i.op.out_ty() == ElemTy::F32));
    if probes.iter().any(|probe| probe.was_materialized())
        || shifts
        || !key.subs.is_empty()
        || !all_f32
        || fused.inputs.iter().any(|input| input.segments() != 1)
    {
        return None;
    }
    let states: Vec<Option<usize>> =
        fused.inputs.iter().map(|input| probes.iter().position(|probe| Arc::ptr_eq(probe, input))).collect();
    let bindings = 2 * outs.len() + states.iter().filter(|state| state.is_none()).count();
    if bindings > ctx.jit().max_inputs + 1 {
        return None;
    }
    let source = super::codegen::loop_source(key, &states, ctx.tuning());
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
            let outputs = launch_loop(ctx, &core, &fused, &states, &states_now, len, count)?;
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

fn len_u32(len: usize) -> Result<u32, LumaError> {
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
struct Shifted {
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
fn pin_inputs(
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
fn params<'a>(
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

/// Launches the store kernel `core` of `key` once per segment of `outputs[0]`: `outputs`, then
/// the inputs as reads (see [`pin_inputs`]), then the uniform (see [`params`]). Counts as one
/// elementwise launch.
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
        let range = first.range(j);
        let n = len_u32(range.len())?;
        let bytes = u64::from(n) * ELEMENT_BYTES;
        let ids = segment_ids(shifted, j)?;
        let out_pins = outputs.iter().map(|out| out.pin(j)).collect::<Result<Vec<_>, _>>()?;
        let in_pins = pin_inputs(key, inputs, ids, j, n)?;
        let words = params(key, ids, None, range.start, shifted.map_or(0, |shifted| shifted.len), scalars)?;
        let mut raw: Vec<RawArg<'_>> = Vec::with_capacity(outputs.len() + in_pins.len() + 1);
        raw.extend(out_pins.iter().map(|pin| RawArg::write(pin.buffer(), bytes)));
        raw.extend(in_pins.iter().map(|(pin, bytes)| RawArg::read(pin.buffer(), *bytes)));
        if !words.is_empty() {
            raw.push(RawArg::Uniform(bytemuck::cast_slice(&words)));
        }
        launch_1d(ctx, core, n, &raw)?;
    }
    ctx.jit().stats.elementwise_launches.fetch_add(1, Ordering::Relaxed);
    Ok(())
}

/// Runs the store kernel of `fused` over `len` elements, one new buffer per output. A single
/// output takes over the buffer of an input that nothing else holds (see [`in_place_eligible`]),
/// instead of a new one.
fn run_store(ctx: &crate::context::Context, len: usize, fused: Fused) -> Result<Vec<Storage>, Failure> {
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
    let ctx = node.context();
    let (fused, args) = flatten_bounded(node, args, Tail::Reduce(ops))?;
    drop(args);

    fused.mark_fused();
    let Fused { mut key, inputs, scalars, .. } = fused;
    // Without inputs (only generators) the segments are those an array of this length would have.
    let layout;
    let first = match inputs.first() {
        Some(first) => first,
        None => {
            layout = Storage::new(ctx, node.len(), ELEMENT_BYTES as usize);
            &layout
        }
    };
    let segments = first.segments();
    let shifted = plan_shifts(ctx, &mut key, &scalars, node.len())?;
    let core = ctx.jit().kernel(ctx, &key)?;

    let mut partials = Vec::with_capacity(segments);
    for j in 0..segments {
        let range = first.range(j);
        let n = len_u32(range.len())?;
        let groups = ctx.tuning().reduce_groups(n);
        let out_bytes = u64::from(groups) * ELEMENT_BYTES * ops.as_slice().len() as u64;
        let out = scratch.take(out_bytes)?;
        let ids = segment_ids(shifted.as_ref(), j)?;
        let params = params(&key, ids, Some(n), range.start, node.len(), &scalars)?;
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
    Ok(Some(partials))
}
