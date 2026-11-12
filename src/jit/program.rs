//! Flattening an expression graph into a [`FusedKey`] (the structure a kernel is generated and
//! cached by) plus the runtime data (buffers, scalar values) it runs with.

use std::sync::atomic::Ordering;
use std::sync::Arc;

use crate::array::{Failure, Storage};
use crate::builtin::ReduceList;
use crate::error::LumaError;

use super::codegen::Tile;
use super::{lock, Args, ElemTy, Node, Op, OpKind, Scalar, Src};

/// Size of every element the fused kernels handle (`f32` and `u32`).
pub(super) const ELEMENT_BYTES: u64 = 4;

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
    /// before being overwritten (see `launch::in_place_eligible`).
    pub(crate) in_place: Option<u32>,
    /// Per input, how many segments of it the program reads through shifts (0 = never): the
    /// program is the segmented variant of the generated kernel, run once per segment of arrays
    /// spanning several buffers. Empty when the program runs unsegmented or has no shift.
    pub(crate) spans: Vec<u8>,
    /// Set by `launch::run_store` like `spans` and `in_place`: the source stages the sub-programs
    /// in workgroup memory (see [`Tile`]) instead of calling them per shift.
    pub(crate) tile: Option<Tile>,
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
#[derive(Clone, Debug)]
pub(super) struct Fused {
    pub(super) key: FusedKey,
    pub(super) inputs: Vec<Arc<Storage>>,
    /// Bit patterns of the scalars, in the order of `key.scalar_tys`.
    pub(super) scalars: Vec<u32>,
    /// Nodes computed inside this program (not as outputs of their own); marked fused once run.
    pub(super) fused_nodes: Vec<Arc<Node>>,
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
/// the program too). Evaluated nodes are leaves. With `trace`, a node that cannot be read
/// without waiting for another thread's evaluation aborts instead of being forced.
fn collect(
    src: &Src,
    members: &mut Vec<Member>,
    seen: &mut Vec<*const Node>,
    trace: bool,
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
        if trace {
            return Err(Failure::New(LumaError::Gpu("traced step reads a node under evaluation".into())));
        }
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
        collect(arg, members, seen, trace)?;
    }
    seen.push(key);
    members.push(Member { node: Arc::clone(node), args });
    Ok(())
}

#[derive(Default)]
struct Flattener {
    /// Flattening for [`flatten_step`]: nothing may be evaluated, and a shifted expression is
    /// not a sub-program.
    trace: bool,
    /// Pending nodes (by address) that stand for an input and are not flattened.
    boundary: Vec<*const Node>,
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
    pub(super) fused_nodes: Vec<Arc<Node>>,
}

impl Flattener {
    /// Emits the instruction for `op` on `args`, whose pending operands were emitted before, and
    /// returns where the result lives.
    fn emit(&mut self, op: Op, args: &[Src; 3]) -> Result<Ref, Failure> {
        let mut refs = [Ref::Unused; 3];
        for (index, (slot, arg)) in refs.iter_mut().zip(args).enumerate().take(op.kind.arity()) {
            *slot = match (op.kind, index, arg) {
                (OpKind::Shift(_), 0, Src::Node(node)) if node.is_pending() && !self.boundary.contains(&Arc::as_ptr(node)) => self.sub(node)?,
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
        if self.trace {
            return Err(Failure::New(LumaError::Gpu("traced step shifts an unevaluated expression".into())));
        }
        let key = Arc::as_ptr(node);
        if let Some(at) = self.sub_nodes.iter().position(|&seen| seen == key) {
            return Ok(Ref::Sub(at as u32));
        }
        let Some(args) = node.operands() else {
            return self.operand(&Src::Node(Arc::clone(node)), node.op().out_ty());
        };
        let (mut members, mut seen) = (Vec::new(), vec![key]);
        for arg in args.iter().take(node.op().kind.arity()) {
            collect(arg, &mut members, &mut seen, false)?;
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
                if self.trace && node.is_pending() {
                    return Err(Failure::New(LumaError::Gpu("traced step reads an unevaluated node".into())));
                }
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
                tile: None,
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

pub(super) fn flatten_inner(
    roots: &[(Arc<Node>, Args)],
    tail: Tail,
    probes: &[u64],
) -> Result<Fused, Failure> {
    if let [(root, args)] = roots
        && probes.is_empty()
        && root.shift_depth() == 0
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
                collect(arg, &mut members, &mut seen, false)?;
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
                    collect(&Src::Node(Arc::clone(sub_root)), &mut sub_members, all_seen, false)?;
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

/// What a traced step reads as state `j`: a pending node (matched by address) or a buffer.
pub(crate) enum StateRef {
    Node(Arc<Node>),
    Buffer(Arc<Storage>),
}

/// A step program computing `roots` from `states`; inputs `0..states.len()` are the states.
pub(super) struct Traced {
    pub(super) fused: Fused,
    /// The node whose value instruction `k` computes, roots included. Callers that keep a traced
    /// step use it to find the nodes of its instructions.
    pub(super) members: Vec<Option<Arc<Node>>>,
}

/// Flattens the expressions of `roots` as a function of `states` without evaluating or
/// otherwise changing any node, for finding out whether a step repeats. A pending state is
/// a boundary: the walk stops there and its uses read input `j`, shifts included. `None` if the
/// expressions cannot be traced that way: they read an unevaluated node that is not a state
/// (shifted ones included), share a node with an operation outside them, have a failed node, a
/// root twice or a root that is a state, or leave a state unread.
pub(super) fn flatten_step(roots: &[Arc<Node>], states: &[StateRef], tail: Tail) -> Option<Traced> {
    let ctx = roots.first()?.context();
    let mut flat = Flattener { trace: true, ..Flattener::default() };
    let mut seen: Vec<*const Node> = Vec::new();
    for (j, state) in states.iter().enumerate() {
        let (storage, node) = match state {
            StateRef::Buffer(storage) => (Arc::clone(storage), None),
            StateRef::Node(node) => match node.result.get() {
                Some(Ok(storage)) => (Arc::clone(storage), Some(node)),
                Some(Err(_)) => return None,
                // Nothing is allocated for it unless something pins it.
                None => (Arc::new(Storage::new(ctx, node.len(), ELEMENT_BYTES as usize)), Some(node)),
            },
        };
        let at = flat.input(&storage, ElemTy::F32);
        if at != Ref::Input(u32::try_from(j).ok()?) {
            return None;
        }
        if let Some(node) = node {
            let key = Arc::as_ptr(node);
            flat.emitted.push((key, at));
            flat.boundary.push(key);
            seen.push(key);
        }
    }
    if roots.iter().enumerate().any(|(i, root)| {
        seen.contains(&Arc::as_ptr(root)) || roots[..i].iter().any(|other| Arc::ptr_eq(other, root))
    }) {
        return None;
    }

    let mut members: Vec<Member> = Vec::new();
    let mut root_args = Vec::with_capacity(roots.len());
    for root in roots {
        if !root.is_pending() {
            return None;
        }
        let args = root.operands()?;
        for (index, arg) in args.iter().enumerate().take(root.op().kind.arity()) {
            if index == 0 && matches!(root.op().kind, OpKind::Shift(_)) {
                continue;
            }
            collect(arg, &mut members, &mut seen, true).ok()?;
        }
        root_args.push(args);
    }
    for (root, args) in roots.iter().zip(root_args) {
        if !seen.contains(&Arc::as_ptr(root)) {
            seen.push(Arc::as_ptr(root));
            members.push(Member { node: Arc::clone(root), args });
        }
    }
    let is_root = |node: &Arc<Node>| roots.iter().any(|root| Arc::ptr_eq(root, node));
    if members.iter().any(|member| !is_root(&member.node) && member.node.has_consumer_outside(&seen)) {
        return None;
    }

    let mut nodes = Vec::with_capacity(members.len());
    flat.instrs.reserve(members.len());
    for member in &members {
        let at = flat.emit(member.node.op(), &member.args).ok()?;
        flat.emitted.push((Arc::as_ptr(&member.node), at));
        nodes.push(Some(Arc::clone(&member.node)));
    }
    let mut outputs = Vec::with_capacity(roots.len());
    for root in roots {
        match flat.emitted.iter().find(|(node, _)| *node == Arc::as_ptr(root)) {
            Some(&(_, Ref::Tmp(at))) => outputs.push(at),
            _ => return None,
        }
    }
    let read = |j: usize| flat.instrs.iter().any(|instr| instr.args.contains(&Ref::Input(j as u32)));
    if !(0..states.len()).all(read) {
        return None;
    }
    Some(Traced { fused: flat.finish(tail, outputs), members: nodes })
}

/// Whether `a` and `b` are the same step: the same program with the same scalar values and the
/// same buffers behind every input after the first `states` (which are the states).
pub(super) fn same_program(a: &Fused, b: &Fused, states: usize) -> bool {
    a.key == b.key
        && a.scalars == b.scalars
        && a.inputs.len() == b.inputs.len()
        && a.inputs.iter().zip(&b.inputs).skip(states).all(|(x, y)| Arc::ptr_eq(x, y))
}
