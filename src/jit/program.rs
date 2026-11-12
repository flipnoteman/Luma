//! Flattening an expression graph into a [`FusedKey`] (the structure a kernel is generated and
//! cached by) plus the runtime data (buffers, scalar values) it runs with.

use std::sync::atomic::Ordering;
use std::sync::Arc;

use crate::array::{Failure, Storage};
use crate::builtin::{BinOp, ReduceList, UnOp};
use crate::error::LumaError;
use crate::gpu::hash::FxMap;

use super::codegen::Tile;
use super::inline::Judgement;
use super::launch::force_group;
use super::staging::Staging;
use super::{lock, Args, Edge, Flattening, ElemTy, Node, Op, OpKind, Scalar, ShiftKind, Src, ViewMap};


/// Where an instruction finds a value.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
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

impl std::hash::Hash for Ref {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        let (tag, value) = match *self {
            Ref::Unused => (0u64, 0u32),
            Ref::Input(k) => (1, k),
            Ref::Scalar(k) => (2, k),
            Ref::Tmp(k) => (3, k),
            Ref::Sub(k) => (4, k),
        };
        state.write_u64(tag << 32 | u64::from(value));
    }
}

/// One operation of a program; instruction `k` defines `t{k}`.
#[derive(Clone, PartialEq, Eq, Debug)]
pub(crate) struct Instr {
    pub(crate) op: Op,
    pub(crate) args: [Ref; 3],
}

impl std::hash::Hash for Instr {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.op.hash(state);
        self.args[0].hash(state);
        self.args[1].hash(state);
        self.args[2].hash(state);
    }
}

/// What happens to the values the program computes.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum Tail {
    /// The outputs are stored: an elementwise kernel.
    Store,
    /// The last instruction is reduced, once per op: the first pass of the reductions.
    Reduce(ReduceList),
    /// Like [`Tail::Store`], but every output is written through the view at its own bound buffer,
    /// which is larger than the launch: a region write (see `jit::assign`).
    StoreMapped(ViewMap),
    /// `outputs == [value, index]`: element `i` writes `value` to element `index` of the one
    /// bound buffer of `len` elements if `index < len` (`Array::put`).
    Scatter { len: u32 },
}

/// Everything the source of a fused kernel depends on, and nothing it does not (buffers and
/// scalar *values* are not part of it).
///
/// The instructions are behind `Arc`s that [`JitState::intern`](super::JitState::intern) makes
/// the same for equal programs, so that looking a key up compares them by address (equal keys
/// are found, and kernels and judgements cached, once per program) and the hash of the
/// instructions is made once.
#[derive(Clone, PartialEq, Eq, Debug)]
pub(crate) struct FusedKey {
    /// Hash of `instrs` and `subs` (see [`shape_of`]); first so that unequal keys are told
    /// apart before their instructions are looked at.
    pub(super) shape: u64,
    pub(crate) instrs: Arc<Vec<Instr>>,
    pub(crate) input_tys: Vec<ElemTy>,
    pub(crate) scalar_tys: Vec<ElemTy>,
    /// The instructions whose values a store kernel writes, one output buffer each.
    pub(crate) outputs: Vec<u32>,
    /// Programs of expressions read through shifts; sub-program `k` defines `sub{k}(i)`, its
    /// last instruction being the value.
    pub(crate) subs: Arc<Vec<Vec<Instr>>>,
    pub(crate) tail: Tail,
    /// Pairs (input `k`, output `j`): input `k` is not bound separately, it is the buffer of
    /// output `j`, read at element `i` before being overwritten (see `launch::in_place_eligible`
    /// and `launch::run_mapped`).
    pub(crate) in_place: Vec<(u32, u32)>,
    /// Per input, how many segments of it the program reads through shifts (0 = never): the
    /// program is the segmented variant of the generated kernel, run once per segment of arrays
    /// spanning several buffers. Empty when the program runs unsegmented or has no shift.
    pub(crate) spans: Vec<u8>,
    /// Set by `launch::run_store` like `spans` and `in_place`: the source stages the sub-programs
    /// in workgroup memory (see [`Tile`]) instead of calling them per shift.
    pub(crate) tile: Option<Tile>,
    /// Set by `JitState::judgement` (and cleared by `launch::run_store` where staging cannot run):
    /// the source stages sub-programs read through views in workgroup memory (see [`Staging`]).
    pub(crate) staging: Option<Staging>,
    /// The uniform slot (a `u32`, always 0) the double-float library hides its operations behind
    /// (see `builtin::df_library`); set when the program runs arithmetic or casts of a `Df32`
    /// (`reads_guard`), or reduces one.
    pub(crate) guard: Option<u32>,
}

/// Whether the WGSL of `instr` calls a function of the double-float library
/// (`builtin::df_library`): the arithmetic, the ordering, and every cast involving a `Df32`.
pub(super) fn calls_df_library(instr: &Instr) -> bool {
    let df = instr.op.ty == ElemTy::Df32;
    match instr.op.kind {
        OpKind::Unary(op) => df && op.expr_for(instr.op.ty).is_some_and(|expr| expr.contains("df_")),
        OpKind::Binary(op) => df && op.expr_for(instr.op.ty).is_some_and(|expr| expr.contains("df_")),
        OpKind::Compare(_) | OpKind::Clamp => df,
        OpKind::Cast(to) => df || to == ElemTy::Df32,
        _ => false,
    }
}

/// Whether `instr` reads the guard uniform: the arithmetic and the casts, whose error-free
/// transformations go through `luma_o`. Ordering, minimum, maximum and absolute value never
/// call it: nothing there can be reassociated away.
pub(super) fn reads_guard(instr: &Instr) -> bool {
    calls_df_library(instr)
        && !matches!(
            instr.op.kind,
            OpKind::Unary(UnOp::Abs) | OpKind::Binary(BinOp::Min | BinOp::Max) | OpKind::Compare(_) | OpKind::Clamp
        )
}

impl std::hash::Hash for FusedKey {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.shape.hash(state);
        self.input_tys.hash(state);
        self.scalar_tys.hash(state);
        self.outputs.hash(state);
        self.tail.hash(state);
        self.in_place.hash(state);
        self.spans.hash(state);
        self.tile.hash(state);
        self.staging.hash(state);
        self.guard.hash(state);
    }
}

/// The hash of the instructions of a program and its sub-programs, as [`FusedKey::shape`] has it.
pub(super) fn shape_of(instrs: &[Instr], subs: &[Vec<Instr>]) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = crate::gpu::hash::FxHasher::default();
    instrs.hash(&mut hasher);
    subs.hash(&mut hasher);
    hasher.finish()
}

impl FusedKey {
    /// Whether every value the program reads, computes or writes is 4 bytes wide: the register
    /// loops, stencils and speculation only handle those.
    pub(crate) fn narrow(&self) -> bool {
        self.input_tys.iter().all(|ty| ty.bytes() == 4)
            && self.instrs.iter().chain(self.subs.iter().flatten()).all(|instr| instr.op.ty.bytes() == 4 && instr.op.out_ty().bytes() == 4)
    }

    /// Whether some instruction reads a sub-program through a view or a gather: the kernel is
    /// then generated from the lowered program of `jit::inline` instead of calling `sub{k}`
    /// functions. Keys whose sub-programs are read through circular shifts only keep the latter
    /// (and the workgroup-memory staging of `codegen::plan_tile`).
    pub(crate) fn inlines(&self) -> bool {
        self.instrs.iter().chain(self.subs.iter().flatten()).any(|instr| {
            matches!(instr.op.kind, OpKind::Shift(ShiftKind::View { .. } | ShiftKind::Take { .. })) && matches!(instr.args[0], Ref::Sub(_))
        })
    }

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
            Tail::StoreMapped(_) => format!("FusedAssign({names}) {ty}"),
            Tail::Scatter { .. } => format!("FusedPut({names}) {ty}"),
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
    /// The node whose value main-program instruction `k` computes (`None` for instructions made
    /// up while flattening). Empty for traced steps.
    pub(super) instr_nodes: Vec<Option<Arc<Node>>>,
    /// The node every `key.subs[k]` computes, the expression a view reads lazily. Callers that
    /// keep the program drop them like `fused_nodes`: they hold their operands.
    pub(super) sub_roots: Vec<Arc<Node>>,
}

impl Fused {
    pub(super) fn mark_fused(&self) {
        for node in &self.fused_nodes {
            node.fused.store(true, Ordering::Relaxed);
        }
    }
}

/// Expressions with more nodes than this look nodes up through a hash map.
const INDEXED_AT: usize = 64;

/// A list of nodes by address with fast lookup of a node's position: the linear scan that is
/// fastest for small expressions grows into a hash map once the list gets long.
#[derive(Clone, Default)]
pub(crate) struct Seen {
    list: Vec<*const Node>,
    index: Option<FxMap<*const Node, u32>>,
}

impl Seen {
    pub(super) fn with_capacity(n: usize) -> Self {
        Seen { list: Vec::with_capacity(n), index: None }
    }

    /// A list that holds `first` and has room for `n` nodes.
    pub(super) fn of_sized(first: *const Node, n: usize) -> Self {
        let mut seen = Seen::with_capacity(n + 1);
        seen.push(first);
        seen
    }

    pub(super) fn position(&self, ptr: *const Node) -> Option<u32> {
        match &self.index {
            Some(index) => index.get(&ptr).copied(),
            None => self.list.iter().position(|&known| known == ptr).map(|at| at as u32),
        }
    }

    pub(crate) fn contains(&self, ptr: *const Node) -> bool {
        self.position(ptr).is_some()
    }

    pub(super) fn push(&mut self, ptr: *const Node) {
        let at = u32::try_from(self.list.len()).unwrap_or(u32::MAX);
        self.list.push(ptr);
        match &mut self.index {
            Some(index) => {
                index.entry(ptr).or_insert(at);
            }
            None if self.list.len() > INDEXED_AT => {
                let mut index = FxMap::with_capacity_and_hasher(self.list.capacity(), Default::default());
                for (at, &known) in self.list.iter().enumerate() {
                    index.entry(known).or_insert(at as u32);
                }
                self.index = Some(index);
            }
            None => {}
        }
    }
}

/// Where the value of every node emitted so far lives, by address.
#[derive(Default)]
struct Emitted {
    list: Vec<(*const Node, Ref)>,
    index: Option<FxMap<*const Node, Ref>>,
}

impl Emitted {
    fn with_capacity(n: usize) -> Self {
        Emitted { list: Vec::with_capacity(n), index: None }
    }

    fn get(&self, ptr: *const Node) -> Option<Ref> {
        match &self.index {
            Some(index) => index.get(&ptr).copied(),
            None => self.list.iter().find(|(known, _)| *known == ptr).map(|&(_, at)| at),
        }
    }

    fn push(&mut self, ptr: *const Node, at: Ref) {
        self.list.push((ptr, at));
        match &mut self.index {
            // The first entry of an address wins, as with the linear search.
            Some(index) => {
                index.entry(ptr).or_insert(at);
            }
            None if self.list.len() > INDEXED_AT => {
                let mut index = FxMap::with_capacity_and_hasher(self.list.capacity(), Default::default());
                for &(known, at) in &self.list {
                    index.entry(known).or_insert(at);
                }
                self.index = Some(index);
            }
            None => {}
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
    seen: &mut Seen,
    trace: bool,
) -> Result<(), Failure> {
    let Src::Node(node) = src else { return Ok(()) };
    let key = Arc::as_ptr(node);
    if seen.contains(key) {
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

/// The nodes of `members` that a shift among them also reads, except `roots`, with the kind of
/// that shift. Read directly, they take the value of their sub-program at offset zero instead of
/// computing their expression a second time.
fn shifted_members(members: &[Member], roots: &[*const Node]) -> Vec<(*const Node, ShiftKind)> {
    let mut shifted: Vec<(*const Node, ShiftKind)> = Vec::new();
    for member in members {
        if let OpKind::Shift(kind) = member.node.op().kind
            && let Src::Node(target) = &member.args[0]
            && target.is_pending()
        {
            let at = Arc::as_ptr(target);
            let direct = members.iter().any(|other| Arc::as_ptr(&other.node) == at);
            if direct && !roots.contains(&at) && !shifted.iter().any(|&(known, _)| known == at) {
                shifted.push((at, zero_shift(kind)));
            }
        }
    }
    shifted
}

/// A shift of the same kind as `kind` by nothing.
fn zero_shift(kind: ShiftKind) -> ShiftKind {
    match kind {
        ShiftKind::Dims { dims, .. } => ShiftKind::Dims { dims, shifts: [0; 4] },
        ShiftKind::Linear => ShiftKind::Linear,
        ShiftKind::View { map, .. } => ShiftKind::View { map: ViewMap::identity(map.src), edge: Edge::None },
        ShiftKind::Take { len } => ShiftKind::View { map: ViewMap::identity([len, 1, 1, 1]), edge: Edge::None },
    }
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
    emitted: Emitted,
    /// Programs of the expressions read through shifts, and the node of each.
    subs: Vec<Vec<Instr>>,
    sub_nodes: Vec<*const Node>,
    /// The nodes of `sub_nodes`, kept alive for [`Fused::sub_roots`].
    sub_roots: Vec<Arc<Node>>,
    /// Nodes read both through shifts and directly (see [`shifted_members`]); a direct read is a
    /// shift by nothing of their sub-program.
    shifted: Vec<(*const Node, ShiftKind)>,
    pub(super) fused_nodes: Vec<Arc<Node>>,
    /// The node each main-program instruction computes (see [`Fused::instr_nodes`]).
    instr_nodes: Vec<Option<Arc<Node>>>,
    /// The uniform slot of the scalar operand `index` of a node, by address (see [`Flattener::emit`]).
    scalar_slots: FxMap<(*const Node, usize), u32>,
}

impl Flattener {
    /// Emits the instruction for `op` on `args`, whose pending operands were emitted before, and
    /// returns where the result lives. `node` is the node the instruction computes (`None` for
    /// instructions made up while flattening): the same node emitted in several programs (the
    /// main one and sub-programs) reads its scalars from the same uniform slots, so that its
    /// copies are equal instructions (see `inline`).
    fn emit(&mut self, op: Op, args: &[Src; 3], node: Option<*const Node>) -> Result<Ref, Failure> {
        let mut refs = [Ref::Unused; 3];
        for (index, (slot, arg)) in refs.iter_mut().zip(args).enumerate().take(op.kind.arity()) {
            *slot = match (op.kind, index, arg) {
                // A source evaluated since the members were collected (by another thread, say) is
                // read as its buffer: `emitted` may hold the instruction that computes it.
                (OpKind::Shift(_), 0, Src::Node(node)) if let Some(storage) = node.evaluated() => self.input(storage, op.operand_ty(0)),
                (OpKind::Shift(_), 0, Src::Node(node)) if node.is_pending() && !self.boundary.contains(&Arc::as_ptr(node)) => self.sub(node)?,
                (_, _, Src::Scalar(_)) if let Some(owner) = node => {
                    let known = self.scalar_slots.get(&(owner, index)).copied();
                    match known {
                        Some(at) => Ref::Scalar(at),
                        None => {
                            let operand = self.operand(arg, op.operand_ty(index))?;
                            if let Ref::Scalar(at) = operand {
                                self.scalar_slots.insert((owner, index), at);
                            }
                            operand
                        }
                    }
                }
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
            // Evaluated (or being evaluated by another thread, which `force` waits for), possibly
            // since it was collected: a shift reads its buffer, never the instruction that
            // computed it in this program.
            return match node.force() {
                Ok(storage) => Ok(self.input(storage, node.op().out_ty())),
                Err(error) => Err(Failure::Inherited(Arc::clone(error))),
            };
        };
        let (mut members, mut seen) = (Vec::with_capacity(node.size()), Seen::of_sized(key, node.size()));
        for arg in args.iter().take(node.op().kind.arity()) {
            collect(arg, &mut members, &mut seen, false)?;
        }
        let shifted = shifted_members(&members, &[key]);
        if !shifted.is_empty() {
            (members, seen) = (Vec::with_capacity(node.size()), Seen::of_sized(key, node.size()));
            for &(at, _) in &shifted {
                seen.push(at);
            }
            for arg in args.iter().take(node.op().kind.arity()) {
                collect(arg, &mut members, &mut seen, false)?;
            }
            self.shifted.extend(shifted);
        }
        let outer_instrs = std::mem::replace(&mut self.instrs, Vec::with_capacity(members.len() + 1));
        let outer_emitted = std::mem::replace(&mut self.emitted, Emitted::with_capacity(members.len() + 1));
        let result = (|| {
            for member in &members {
                let at = self.emit(member.node.op(), &member.args, Some(Arc::as_ptr(&member.node)))?;
                self.emitted.push(Arc::as_ptr(&member.node), at);
            }
            self.emit(node.op(), &args, Some(key))
        })();
        let instrs = std::mem::replace(&mut self.instrs, outer_instrs);
        self.emitted = outer_emitted;
        result?;
        self.fused_nodes.extend(members.into_iter().map(|m| m.node));
        self.fused_nodes.push(Arc::clone(node));
        self.subs.push(instrs);
        self.sub_nodes.push(key);
        self.sub_roots.push(Arc::clone(node));
        Ok(Ref::Sub(u32::try_from(self.subs.len() - 1).unwrap_or(u32::MAX)))
    }

    fn operand(&mut self, src: &Src, ty: ElemTy) -> Result<Ref, Failure> {
        match src {
            Src::Scalar(scalar) => {
                let at = u32::try_from(self.scalars.len()).unwrap_or(u32::MAX);
                // The uniform has `f32` fields: a wide scalar takes several (see `Scalar::words`).
                let slot_ty = if scalar.ty.bytes() == 4 { scalar.ty } else { ElemTy::F32 };
                for &word in scalar.words().iter().take(scalar.ty.slots()) {
                    self.scalar_tys.push(slot_ty);
                    self.scalars.push(word);
                }
                Ok(Ref::Scalar(at))
            }
            Src::Buffer(storage) => Ok(self.input(storage, ty)),
            Src::Node(node) => {
                let key = Arc::as_ptr(node);
                if let Some(found) = self.emitted.get(key) {
                    return Ok(found);
                }
                // Not a member: it was evaluated when the members were collected, or it is read
                // through a shift too.
                if let Some(&(_, kind)) = self.shifted.iter().find(|(seen, _)| *seen == key)
                    && node.operands().is_some()
                {
                    return self.unshifted(node, kind);
                }
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

    /// The value of `node`'s sub-program at the element itself: a shift of `kind` by nothing.
    fn unshifted(&mut self, node: &Arc<Node>, kind: ShiftKind) -> Result<Ref, Failure> {
        let sub = self.sub(node)?;
        let (amount, total) = match kind {
            ShiftKind::Linear => (
                self.operand(&Src::Scalar(Scalar::u32(0)), ElemTy::U32)?,
                self.operand(&Src::Scalar(Scalar::u32(node.len() as u32)), ElemTy::U32)?,
            ),
            ShiftKind::Dims { .. } | ShiftKind::View { .. } | ShiftKind::Take { .. } => (Ref::Unused, Ref::Unused),
        };
        let at = u32::try_from(self.instrs.len()).unwrap_or(u32::MAX);
        self.instrs.push(Instr { op: Op::new(OpKind::Shift(kind), node.op().out_ty()), args: [sub, amount, total] });
        self.emitted.push(Arc::as_ptr(node), Ref::Tmp(at));
        Ok(Ref::Tmp(at))
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

    fn finish(mut self, tail: Tail, outputs: Vec<u32>) -> Fused {
        // The guard is a uniform that only the arithmetic of the library reads (`luma_o`): only
        // programs that run it get one, because a kernel whose uniform no code reads would not
        // bind it (see `reads_guard`).
        let reduces_df = matches!(tail, Tail::Reduce(_)) && self.instrs.last().is_some_and(|instr| instr.op.out_ty() == ElemTy::Df32);
        let guard = (reduces_df || self.instrs.iter().chain(self.subs.iter().flatten()).any(reads_guard))
        .then(|| {
            self.scalar_tys.push(ElemTy::U32);
            self.scalars.push(0);
            u32::try_from(self.scalar_tys.len() - 1).unwrap_or(u32::MAX)
        });
        Fused {
            key: FusedKey {
                shape: shape_of(&self.instrs, &self.subs),
                instrs: Arc::new(self.instrs),
                input_tys: self.input_tys,
                scalar_tys: self.scalar_tys,
                subs: Arc::new(self.subs),
                outputs,
                tail,
                in_place: Vec::new(),
                spans: Vec::new(),
                tile: None,
                staging: None,
                guard,
            },
            inputs: self.inputs,
            scalars: self.scalars,
            fused_nodes: self.fused_nodes,
            instr_nodes: self.instr_nodes,
            sub_roots: self.sub_roots,
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
    seen: Seen,
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
            seen: Seen::with_capacity(n),
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
                if let Some(at) = self.seen.position(key) {
                    return Ok(Resolved::Tmp(at));
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
                *slot = match self.seen.position(key) {
                    Some(at) => Resolved::Tmp(at),
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

/// Whether the `shared` nodes of the expressions `roots` can be evaluated before them without
/// the program over them reading more buffers than a kernel binds. Each evaluated node is a
/// buffer it reads (besides those it reads below it when others read them too): past the limit
/// the shared nodes are recomputed in the program instead, which is a result just as right.
fn evaluation_fits<'a>(roots: impl Iterator<Item = (&'a Arc<Node>, &'a Args)> + Clone, shared: &[Arc<Node>]) -> bool {
    let Some((first, _)) = roots.clone().next() else { return true };
    let operands: Vec<Src> = roots.clone().flat_map(|(root, args)| args.iter().take(root.op().kind.arity()).cloned()).collect();
    super::count_inputs(&operands, shared).len() + roots.count() <= first.context().jit().max_inputs() + 1
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
        if !shared.is_empty() && evaluation_fits(std::iter::once((root, args)), &shared) {
            force_group(&shared)?;
            continue;
        }
        let outputs = vec![u32::try_from(flat.instrs.len() - 1).unwrap_or(u32::MAX)];
        flat.instr_nodes = members.iter().map(|member| Some(Arc::clone(member))).chain(Some(Some(Arc::clone(root)))).collect();
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

/// [`flatten_once`], then what a program that reads sub-programs needs to run: they only run on
/// arrays that fit one buffer (`launch::plan_shifts`), and a kernel that reads them through views
/// must not grow past what [`Tuning::max_inlined`](crate::tuning::Tuning) allows once they are
/// written out at every index they are read at (see `inline`). Otherwise the sub-programs'
/// nodes are evaluated and the expressions flattened again, which reads them as buffers; every
/// round evaluates a node that was pending, so it ends.
pub(super) fn flatten_inner(
    roots: &[(Arc<Node>, Args)],
    tail: Tail,
    probes: &[u64],
) -> Result<Fused, Failure> {
    let _flattening = Flattening::enter(roots.iter().map(|(root, _)| Arc::as_ptr(root)));
    loop {
        let mut fused = flatten_once(roots, tail, probes)?;
        if let Some((first, _)) = roots.first() {
            first.context().jit().intern(&mut fused.key);
        }
        let Some((first, _)) = roots.first().filter(|_| probes.is_empty() && !fused.key.subs.is_empty()) else {
            return Ok(fused);
        };
        let ctx = first.context();
        let jit = ctx.jit();
        // Arrays split over several buffers: the sub-programs cannot run in a kernel over them.
        let splits = roots.iter().any(|(root, _)| root.len() as u64 * root.op().out_ty().bytes() > ctx.segment_bytes())
            || fused.key.instrs.iter().chain(fused.key.subs.iter().flatten()).any(|instr| {
                matches!(instr.op.kind, OpKind::Shift(_))
                    && matches!(instr.args[0], Ref::Input(k) if fused.inputs.get(k as usize).is_some_and(|input| input.segments() > 1))
            });
        if splits {
            let sub_roots = fused.sub_roots.clone();
            drop(fused);
            force_group(&sub_roots)?;
            continue;
        }
        if fused.key.inlines() {
            let judgement = jit.judgement(ctx, &fused.key, first.len());
            match &*judgement {
                Judgement::Materialise { sub, reason } => {
                    jit.note_split(first.name, first.location, *reason);
                    let node = fused.sub_roots.get(*sub as usize).cloned();
                    drop(fused);
                    if let Some(node) = node {
                        jit.evaluate_alongside(Arc::clone(&node));
                        if let Err(error) = node.force() {
                            return Err(Failure::Inherited(Arc::clone(error)));
                        }
                    }
                    continue;
                }
                Judgement::Run { staging } => fused.key.staging = staging.clone(),
            }
        }
        return Ok(fused);
    }
}

/// Flattens `roots` once (see [`flatten`]).
fn flatten_once(
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
    let mut shifted;
    let members = loop {
        let gather = |seed: &[*const Node]| -> Result<(Vec<Member>, Seen), Failure> {
            let mut members = Vec::with_capacity(expected);
            let mut seen = Seen::with_capacity(expected + seed.len());
            for &ptr in seed {
                seen.push(ptr);
            }
            for (root, args) in roots {
                for (index, arg) in args.iter().enumerate().take(root.op().kind.arity()) {
                    if index == 0 && matches!(root.op().kind, OpKind::Shift(_)) {
                        continue;
                    }
                    collect(arg, &mut members, &mut seen, false)?;
                }
            }
            for (root, args) in roots {
                if !seen.contains(Arc::as_ptr(root)) {
                    seen.push(Arc::as_ptr(root));
                    members.push(Member { node: Arc::clone(root), args: args.clone() });
                }
            }
            Ok((members, seen))
        };
        let (mut members, mut seen) = gather(&[])?;
        let root_ptrs: Vec<*const Node> = roots.iter().map(|(root, _)| Arc::as_ptr(root)).collect();
        shifted = shifted_members(&members, &root_ptrs);
        let mut targets = Vec::new();
        if !shifted.is_empty() {
            let ptrs: Vec<*const Node> = shifted.iter().map(|&(at, _)| at).collect();
            targets = members.into_iter().filter(|member| ptrs.contains(&Arc::as_ptr(&member.node))).collect();
            (members, seen) = gather(&ptrs)?;
        }
        let mut sub_members = Vec::new();
        // Only expressions read through shifts add to what was seen.
        let mut with_subs: Option<Seen> = None;
        for member in &members {
            if matches!(member.node.op().kind, OpKind::Shift(_))
                && let Src::Node(sub_root) = &member.args[0] {
                    let all_seen = with_subs.get_or_insert_with(|| seen.clone());
                    collect(&Src::Node(Arc::clone(sub_root)), &mut sub_members, all_seen, false)?;
                }
        }
        // The shifted members are in `seen` already, so their expressions are collected here.
        for target in targets {
            let all_seen = with_subs.get_or_insert_with(|| seen.clone());
            for arg in target.args.iter().take(target.node.op().kind.arity()) {
                collect(arg, &mut sub_members, all_seen, false)?;
            }
            sub_members.push(target);
        }
        // Sub-programs read further sub-programs through shifts of their own: those expressions
        // are part of the kernel too, and nothing in them is read from outside it.
        let mut next = 0;
        while let Some(member) = sub_members.get(next) {
            next += 1;
            if matches!(member.node.op().kind, OpKind::Shift(_))
                && let Src::Node(sub_root) = &member.args[0]
            {
                let sub_root = Arc::clone(sub_root);
                let all_seen = with_subs.get_or_insert_with(|| seen.clone());
                collect(&Src::Node(sub_root), &mut sub_members, all_seen, false)?;
            }
        }
        let all_seen = with_subs.as_ref().unwrap_or(&seen);
        let is_root = |node: &Arc<Node>| roots.iter().any(|(root, _)| Arc::ptr_eq(root, node));
        // A node that `eval` asked for (and whose evaluation was postponed) is computed on its
        // own once anyway: read inside a sub-program, it would be recomputed at every instance.
        let requested = |member: &&Member| member.node.requested.load(Ordering::Relaxed) && member.node.is_pending();
        let shared: Vec<Arc<Node>> = members
            .iter()
            .chain(&sub_members)
            .filter(|member| !is_root(&member.node) && member.node.has_consumer_outside(all_seen))
            .chain(sub_members.iter().filter(|member| !is_root(&member.node)).filter(requested))
            .map(|member| Arc::clone(&member.node))
            .collect();
        if shared.is_empty() {
            break members;
        }
        if shared.iter().any(|node| probes.iter().any(|&id| node.depends_on(id))) {
            return Err(Failure::New(LumaError::Gpu("stencil probe shared outside step".into())));
        }
        if !evaluation_fits(roots.iter().map(|(root, args)| (root, args)), &shared) {
            break members;
        }
        force_group(&shared)?;
    };

    let mut flattener = Flattener { shifted, ..Flattener::default() };
    flattener.instrs.reserve(members.len());
    flattener.emitted.list.reserve(members.len());
    let mut member_at = Vec::with_capacity(members.len());
    for member in &members {
        let at = flattener.emit(member.node.op(), &member.args, Some(Arc::as_ptr(&member.node)))?;
        flattener.emitted.push(Arc::as_ptr(&member.node), at);
        member_at.push(at);
    }
    let mut instr_nodes: Vec<Option<Arc<Node>>> = vec![None; flattener.instrs.len()];
    for (member, at) in members.iter().zip(&member_at) {
        if let Ref::Tmp(k) = at
            && let Some(slot) = instr_nodes.get_mut(*k as usize)
        {
            *slot = Some(Arc::clone(&member.node));
        }
    }
    flattener.instr_nodes = instr_nodes;
    let mut outputs = Vec::with_capacity(roots.len());
    for (root, _) in roots {
        match flattener.emitted.get(Arc::as_ptr(root)) {
            Some(Ref::Tmp(at)) => outputs.push(at),
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
    let mut seen = Seen::default();
    for (j, state) in states.iter().enumerate() {
        let (storage, node) = match state {
            StateRef::Buffer(storage) => (Arc::clone(storage), None),
            StateRef::Node(node) => match node.result.get() {
                Some(Ok(storage)) => (Arc::clone(storage), Some(node)),
                Some(Err(_)) => return None,
                // Nothing is allocated for it unless something pins it.
                None => (Arc::new(Storage::new(ctx, node.len(), ElemTy::F32.bytes() as usize)), Some(node)),
            },
        };
        let at = flat.input(&storage, ElemTy::F32);
        if at != Ref::Input(u32::try_from(j).ok()?) {
            return None;
        }
        if let Some(node) = node {
            let key = Arc::as_ptr(node);
            flat.emitted.push(key, at);
            flat.boundary.push(key);
            seen.push(key);
        }
    }
    if roots.iter().enumerate().any(|(i, root)| {
        seen.contains(Arc::as_ptr(root)) || roots[..i].iter().any(|other| Arc::ptr_eq(other, root))
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
        if !seen.contains(Arc::as_ptr(root)) {
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
        let at = flat.emit(member.node.op(), &member.args, Some(Arc::as_ptr(&member.node))).ok()?;
        flat.emitted.push(Arc::as_ptr(&member.node), at);
        nodes.push(Some(Arc::clone(&member.node)));
    }
    let mut outputs = Vec::with_capacity(roots.len());
    for root in roots {
        match flat.emitted.get(Arc::as_ptr(root)) {
            Some(Ref::Tmp(at)) => outputs.push(at),
            _ => return None,
        }
    }
    let read = |j: usize| flat.instrs.iter().any(|instr| instr.args.contains(&Ref::Input(j as u32)));
    if !(0..states.len()).all(read) {
        return None;
    }
    let fused = flat.finish(tail, outputs);
    // Traced steps are for the 4-byte machinery (loops, speculation, stencils).
    fused.key.narrow().then_some(Traced { fused, members: nodes })
}

/// Whether `a` and `b` are the same step: the same program with the same scalar values and the
/// same buffers behind every input after the first `states` (which are the states).
pub(super) fn same_program(a: &Fused, b: &Fused, states: usize) -> bool {
    a.key == b.key
        && a.scalars == b.scalars
        && a.inputs.len() == b.inputs.len()
        && a.inputs.iter().zip(&b.inputs).skip(states).all(|(x, y)| Arc::ptr_eq(x, y))
}

/// Checks the invariants the code generator and the launcher rely on: references point at
/// earlier instructions, inputs and scalars that exist, sub-programs only at lower ones, views
/// stay inside their sources, in-place inputs are read at their own element only, and the
/// staging and tile layouts fit the program. Debug builds run it on every kernel built.
#[cfg(any(debug_assertions, test))]
pub(super) fn validate(key: &FusedKey, inputs: usize, scalars: usize) -> Result<(), String> {
    let programs = std::iter::once(&key.instrs[..]).chain(key.subs.iter().map(|sub| &sub[..]));
    for (p, program) in programs.enumerate() {
        let name = if p == 0 { "main program".to_string() } else { format!("sub-program {}", p - 1) };
        for (at, instr) in program.iter().enumerate() {
            let here = format!("{name}, instruction {at} ({:?})", instr.op.kind);
            let arity = instr.op.kind.arity();
            for (slot, &arg) in instr.args.iter().enumerate() {
                if (slot < arity) == (arg == Ref::Unused) {
                    return Err(format!("{here}: operand {slot} is {arg:?} but the arity is {arity}"));
                }
                match arg {
                    Ref::Unused => {}
                    Ref::Tmp(t) if t as usize >= at => return Err(format!("{here}: operand {slot} reads t{t}, not an earlier instruction")),
                    Ref::Tmp(_) => {}
                    Ref::Input(k) if k as usize >= inputs => return Err(format!("{here}: operand {slot} reads input {k} of {inputs}")),
                    Ref::Input(_) => {}
                    Ref::Scalar(s) => {
                        let end = s as usize + instr.op.operand_ty(slot).slots();
                        if end > scalars {
                            return Err(format!("{here}: operand {slot} reads uniform slots up to {end} of {scalars}"));
                        }
                    }
                    Ref::Sub(j) => {
                        if slot != 0 || !matches!(instr.op.kind, OpKind::Shift(_)) {
                            return Err(format!("{here}: operand {slot} is a sub-program but not operand 0 of a shift"));
                        }
                        if j as usize >= key.subs.len() {
                            return Err(format!("{here}: reads sub-program {j} of {}", key.subs.len()));
                        }
                        if p != 0 && j as usize >= p - 1 {
                            return Err(format!("{here}: reads sub-program {j}, not a lower one"));
                        }
                    }
                }
            }
            if let OpKind::Shift(kind) = instr.op.kind {
                if let ShiftKind::View { map, edge } = kind {
                    if !map.fits() {
                        return Err(format!("{here}: the view {map:?} does not fit the index arithmetic"));
                    }
                    if edge == Edge::None && !map.in_range() {
                        return Err(format!("{here}: the view {map:?} reads outside its source"));
                    }
                }
            }
        }
    }
    if key.instrs.is_empty() {
        return Err("the main program is empty".into());
    }
    if key.subs.iter().any(|sub| sub.is_empty()) {
        return Err("an empty sub-program".into());
    }
    if let Some(&at) = key.outputs.iter().find(|&&at| at as usize >= key.instrs.len()) {
        return Err(format!("output {at} of {} instructions", key.instrs.len()));
    }
    if matches!(key.tail, Tail::Scatter { .. }) && key.outputs.len() != 2 {
        return Err(format!("a scatter has {} outputs, not 2", key.outputs.len()));
    }
    for &(k, j) in &key.in_place {
        if k as usize >= inputs {
            return Err(format!("in-place input {k} of {inputs}"));
        }
        if j as usize >= key.outputs.len() {
            return Err(format!("in-place output {j} of {}", key.outputs.len()));
        }
        let shifted = key.instrs.iter().any(|instr| matches!(instr.op.kind, OpKind::Shift(_)) && instr.args[0] == Ref::Input(k));
        let in_sub = key.subs.iter().flatten().any(|instr| instr.args.contains(&Ref::Input(k)));
        // Mapped and scatter kernels (`run_mapped`) may read their base through shifts: the
        // caller of `jit::assign` proved no element is read after another invocation wrote it.
        if key.tail == Tail::Store && (shifted || in_sub) {
            return Err(format!("in-place input {k} is read through a shift or in a sub-program"));
        }
    }
    if let Some(tile) = &key.tile {
        if tile.offsets.len() != 1 + key.subs.len() {
            return Err(format!("tile offsets for {} programs, {} exist", tile.offsets.len(), 1 + key.subs.len()));
        }
        let lens = std::iter::once(key.instrs.len()).chain(key.subs.iter().map(Vec::len));
        for (p, (row, len)) in tile.offsets.iter().zip(lens).enumerate() {
            if row.len() != len {
                return Err(format!("tile offsets of program {p} have {} rows for {len} instructions", row.len()));
            }
        }
    }
    if let Some(layout) = &key.staging {
        super::staging::check(layout, key).map_err(|message| format!("invalid staging: {message}"))?;
    }
    Ok(())
}

#[cfg(test)]
mod validate_tests {
    use super::*;
    use crate::builtin::BinOp;

    fn instr(kind: OpKind, args: [Ref; 3]) -> Instr {
        Instr { op: Op::new(kind, ElemTy::F32), args }
    }

    fn key(instrs: Vec<Instr>, subs: Vec<Vec<Instr>>) -> FusedKey {
        let outputs = vec![instrs.len() as u32 - 1];
        FusedKey {
            shape: shape_of(&instrs, &subs),
            instrs: Arc::new(instrs),
            input_tys: vec![ElemTy::F32],
            scalar_tys: vec![ElemTy::F32],
            outputs,
            subs: Arc::new(subs),
            tail: Tail::Store,
            in_place: Vec::new(),
            spans: Vec::new(),
            tile: None,
            staging: None,
            guard: None,
        }
    }

    fn add(a: Ref, b: Ref) -> Instr {
        instr(OpKind::Binary(BinOp::Add), [a, b, Ref::Unused])
    }

    fn check(key: &FusedKey) -> Result<(), String> {
        validate(key, key.input_tys.len(), key.scalar_tys.len())
    }

    fn view(src: u32, start: i32, out: u32) -> Instr {
        let mut map = ViewMap::identity([src, 1, 1, 1]);
        map.out[0] = out;
        map.axes[0].start = start;
        instr(OpKind::Shift(ShiftKind::View { map, edge: Edge::None }), [Ref::Sub(0), Ref::Unused, Ref::Unused])
    }

    #[test]
    fn a_well_formed_key_is_accepted() {
        let sub = vec![add(Ref::Input(0), Ref::Scalar(0))];
        let main = vec![view(9, 1, 8), view(9, 0, 8), add(Ref::Tmp(0), Ref::Tmp(1))];
        assert_eq!(check(&key(main, vec![sub])), Ok(()));
    }

    #[test]
    fn a_forward_temporary_is_rejected() {
        let k = key(vec![add(Ref::Tmp(1), Ref::Input(0)), add(Ref::Tmp(0), Ref::Input(0))], vec![]);
        assert!(check(&k).unwrap_err().contains("earlier"));
    }

    #[test]
    fn an_input_or_scalar_out_of_range_is_rejected() {
        assert!(check(&key(vec![add(Ref::Input(1), Ref::Scalar(0))], vec![])).unwrap_err().contains("input 1"));
        assert!(check(&key(vec![add(Ref::Input(0), Ref::Scalar(1))], vec![])).unwrap_err().contains("uniform"));
    }

    #[test]
    fn a_sub_program_reading_a_later_one_is_rejected() {
        let first = vec![view(9, 1, 8)];
        let second = vec![add(Ref::Input(0), Ref::Scalar(0))];
        let main = vec![view(9, 0, 8)];
        assert!(check(&key(main.clone(), vec![first, second.clone()])).unwrap_err().contains("lower"));
        assert!(check(&key(vec![view(9, 0, 8)], vec![])).unwrap_err().contains("sub-program"));
    }

    #[test]
    fn a_view_outside_its_source_is_rejected() {
        let sub = vec![add(Ref::Input(0), Ref::Scalar(0))];
        let k = key(vec![view(9, 2, 8)], vec![sub]);
        assert!(check(&k).unwrap_err().contains("outside"));
    }

    #[test]
    fn a_wrong_arity_or_in_place_shift_read_is_rejected() {
        let k = key(vec![instr(OpKind::Binary(BinOp::Add), [Ref::Input(0), Ref::Unused, Ref::Unused])], vec![]);
        assert!(check(&k).unwrap_err().contains("arity"));
        let mut shifted = key(vec![instr(OpKind::Shift(ShiftKind::Linear), [Ref::Input(0), Ref::Scalar(0), Ref::Scalar(0)])], vec![]);
        shifted.in_place = vec![(0, 0)];
        assert!(check(&shifted).unwrap_err().contains("in-place"));
    }
}
