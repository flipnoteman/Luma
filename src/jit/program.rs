//! Flattening an expression graph into a [`FusedKey`] (the structure a kernel is generated and
//! cached by) plus the runtime data (buffers, scalar values) it runs with.

use std::cell::Cell;
use std::sync::atomic::Ordering;
use std::sync::Arc;

use crate::array::{Failure, Storage};
use crate::builtin::{BinOp, ReduceList, UnOp};
use crate::error::LumaError;
use crate::gpu::hash::FxMap;
use crate::profile::SplitReason;

use super::codegen::Tile;
use super::inline::Judgement;
use super::launch::force_group;
use super::staging::Staging;
use super::{lock, Args, Edge, Flattening, ElemTy, JitState, Node, Op, OpKind, Scalar, ShiftKind, Src, ViewMap};


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
#[derive(Clone, Debug)]
pub(crate) struct Instr {
    pub(crate) op: Op,
    pub(crate) args: [Ref; 3],
}

impl PartialEq for Instr {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.args == other.args && self.op == other.op
    }
}

impl Eq for Instr {}

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
    /// short. A trailing copy (the reader a reduction keeps of its operand, the store of an
    /// assignment) computes nothing and is not named unless it is the whole program.
    pub(crate) fn label(&self) -> String {
        const SHOWN: usize = 6;
        let copy = |instr: &Instr| matches!(instr.op.kind, OpKind::Unary(UnOp::Id));
        let named = match self.instrs.split_last() {
            Some((last, rest)) if copy(last) && !rest.is_empty() => rest,
            _ => &self.instrs[..],
        };
        let mut names: Vec<String> = named
            .iter()
            .take(SHOWN)
            .map(|instr| match instr.op.kind {
                OpKind::Unary(op) => format!("{op:?}"),
                OpKind::Binary(op) => format!("{op:?}"),
                OpKind::Compare(op) => format!("{op:?}"),
                other => format!("{other:?}"),
            })
            .collect();
        if named.len() > SHOWN {
            names.push(format!("+{} more", named.len() - SHOWN));
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

/// No item: an operand that is no pending node (a scalar, a buffer), or a node the walks treat
/// as a leaf (evaluated, evaluated by another thread, or a state of a traced step).
const NONE: u32 = u32::MAX;

/// A pending node met while flattening, with its operands read once.
struct Item {
    node: Arc<Node>,
    args: Args,
    /// The item of each operand that is a pending node, `NONE` for the others.
    kids: [u32; 3],
    /// The uniform slot of each scalar operand once emitted, `NONE` before: a node emitted in
    /// several programs (the main one and sub-programs) reads its scalars from the same slots,
    /// so that its copies are equal instructions (see `inline`).
    slots: [Cell<u32>; 3],
    /// The type each operand must have (see [`Op::operand_ty`]).
    tys: [ElemTy; 3],
    arity: u8,
    /// Whether the node is a shift: its operand 0 is read through a sub-program.
    shift: bool,
}

impl Item {
    fn new(node: &Arc<Node>, args: Args, kids: [u32; 3], arity: usize) -> Self {
        let op = &node.op;
        let tys = [op.operand_ty(0), op.operand_ty(1), op.operand_ty(2)];
        Item {
            node: Arc::clone(node),
            args,
            kids,
            slots: [Cell::new(NONE), Cell::new(NONE), Cell::new(NONE)],
            tys,
            arity: u8::try_from(arity).unwrap_or(3),
            shift: matches!(op.kind, OpKind::Shift(_)),
        }
    }

    /// The item a shift reads through a sub-program; `NONE` for any other node.
    fn source(&self) -> u32 {
        if self.shift { self.kids[0] } else { NONE }
    }
}

/// The pending nodes below some roots, each read once: a node's operands are copied out under
/// its lock, and the operands that are pending nodes become indices. Collecting the members of
/// the main program and of every sub-program, and emitting their instructions, then follow
/// indices instead of looking nodes up by address and locking them again (a node is a member of
/// the main program and of every sub-program that reads it).
struct Graph {
    items: Vec<Item>,
    index: Index,
    /// Flattening for [`flatten_step`]: nothing may be evaluated, and a shifted expression is
    /// not followed (it is not a sub-program).
    trace: bool,
    /// Pending nodes (by address) that stand for an input: leaves.
    boundary: Vec<*const Node>,
}

/// The item of a node, by the id of the node: an array over the ids that can be below the roots
/// (see [`Node::below_span`]), a hash map for any other.
struct Index {
    base: u64,
    dense: Vec<u32>,
    spill: FxMap<u64, u32>,
}

impl Index {
    fn new<'a>(roots: impl Iterator<Item = &'a Arc<Node>> + Clone, expected: usize) -> Self {
        let (mut lo, mut hi) = (u64::MAX, 0u64);
        for root in roots {
            let (first, last) = root.below_span().unwrap_or((root.id, root.id));
            lo = lo.min(first).min(root.id);
            hi = hi.max(last).max(root.id);
        }
        // Ids are handed out as nodes are made, so those of an expression are close together; one
        // that holds the ids of old nodes too would take a lot of room.
        let width = hi.checked_sub(lo).and_then(|span| usize::try_from(span).ok()).map_or(0, |span| span.saturating_add(1));
        let (base, dense) = if width != 0 && width <= expected.saturating_mul(64).saturating_add(1024) {
            (lo, vec![NONE; width])
        } else {
            (0, Vec::new())
        };
        Index { base, dense, spill: FxMap::default() }
    }

    fn get(&self, id: u64) -> u32 {
        match self.dense.get(id.wrapping_sub(self.base) as usize) {
            Some(&at) if id >= self.base => at,
            _ if self.spill.is_empty() => NONE,
            _ => self.spill.get(&id).copied().unwrap_or(NONE),
        }
    }

    fn insert(&mut self, id: u64, at: u32) {
        match self.dense.get_mut(id.wrapping_sub(self.base) as usize) {
            Some(slot) if id >= self.base => *slot = at,
            _ => {
                self.spill.insert(id, at);
            }
        }
    }
}

impl Graph {
    fn new<'a>(roots: impl Iterator<Item = &'a Arc<Node>> + Clone, trace: bool, boundary: Vec<*const Node>) -> Self {
        let expected: usize = roots.clone().map(|root| root.size()).sum();
        Graph { items: Vec::with_capacity(expected), index: Index::new(roots, expected), trace, boundary }
    }

    /// The items by the address of their nodes.
    fn by_address(&self) -> FxMap<*const Node, u32> {
        let mut map = FxMap::with_capacity_and_hasher(self.items.len(), Default::default());
        for (at, item) in self.items.iter().enumerate() {
            map.insert(Arc::as_ptr(&item.node), at as u32);
        }
        map
    }

    /// Adds a root with the operands it is `given` (a node being evaluated has taken its own),
    /// and says its item and whether it is new (a root twice is one item). Its operands are read
    /// by [`Graph::link_root`] once all roots are added, so that a root another one reads is
    /// that item.
    fn add_root(&mut self, node: &Arc<Node>, args: Args) -> (u32, bool) {
        let known = self.index.get(node.id);
        if known != NONE {
            return (known, false);
        }
        let at = u32::try_from(self.items.len()).unwrap_or(u32::MAX);
        self.index.insert(node.id, at);
        let arity = node.op.kind.arity();
        self.items.push(Item::new(node, args, [NONE; 3], arity));
        (at, true)
    }

    /// Reads the operands below the root `at`.
    fn link_root(&mut self, at: u32) -> Result<(), Failure> {
        let item = &mut self.items[at as usize];
        let node = Arc::clone(&item.node);
        let args = std::mem::replace(&mut item.args, [Src::unused(), Src::unused(), Src::unused()]);
        let kids = self.link(&node, &args, usize::from(self.items[at as usize].arity));
        let item = &mut self.items[at as usize];
        item.args = args;
        item.kids = kids?;
        Ok(())
    }

    /// The items of the operands `args` of `node` (which has `arity` of them), made if they are
    /// not there yet.
    fn link(&mut self, node: &Node, args: &Args, arity: usize) -> Result<[u32; 3], Failure> {
        let skip_source = self.trace && matches!(node.op.kind, OpKind::Shift(_));
        let mut kids = [NONE; 3];
        for (index, (kid, arg)) in kids.iter_mut().zip(args).enumerate().take(arity) {
            if skip_source && index == 0 {
                continue;
            }
            *kid = self.kid(arg)?;
        }
        Ok(kids)
    }

    /// The item of `src` if it is a pending node, `NONE` if it is something else or a leaf. An
    /// evaluated node is a leaf; so is one another thread evaluates right now (waited for); with
    /// `trace` that is an error instead.
    fn kid(&mut self, src: &Src) -> Result<u32, Failure> {
        let Src::Node(node) = src else { return Ok(NONE) };
        // Evaluated nodes are most of the operands: told from the node itself, not looked up.
        match node.result.get() {
            Some(Ok(_)) => return Ok(NONE),
            Some(Err(error)) => return Err(Failure::Inherited(Arc::clone(error))),
            None => {}
        }
        let known = self.index.get(node.id);
        if known != NONE {
            return Ok(known);
        }
        if !self.boundary.is_empty() && self.boundary.contains(&Arc::as_ptr(node)) {
            return Ok(NONE);
        }
        let Some(args) = node.operands() else {
            if self.trace {
                return Err(Failure::New(LumaError::Gpu("traced step reads a node under evaluation".into())));
            }
            // Another thread is evaluating it right now; wait for its buffer.
            return match node.force() {
                Ok(_) => Ok(NONE),
                Err(error) => Err(Failure::Inherited(Arc::clone(error))),
            };
        };
        let arity = node.op.kind.arity();
        let kids = self.link(node, &args, arity)?;
        let at = u32::try_from(self.items.len()).unwrap_or(u32::MAX);
        self.index.insert(node.id, at);
        self.items.push(Item::new(node, args, kids, arity));
        Ok(at)
    }

    /// Appends item `at` and the items below it to `members`, operands before the items using
    /// them. `seen` holds the items already collected (each is collected once, so shared nodes
    /// are shared in the program too).
    fn collect(&self, at: u32, members: &mut Vec<u32>, seen: &mut [bool]) {
        if seen[at as usize] {
            return;
        }
        self.collect_kids(at, true, members, seen);
        seen[at as usize] = true;
        members.push(at);
    }

    /// [`Graph::collect`] for the items below item `at` only. A shifted expression is a
    /// sub-program of its own (see [`Flattener::sub`]), so the walk of a program leaves the
    /// source of a shift out when `without_source`.
    fn collect_kids(&self, at: u32, without_source: bool, members: &mut Vec<u32>, seen: &mut [bool]) {
        let item = &self.items[at as usize];
        for (index, &kid) in item.kids.iter().enumerate() {
            if kid != NONE && !(without_source && index == 0 && item.shift) {
                self.collect(kid, members, seen);
            }
        }
    }

    /// The items of `members` that a shift among them also reads (`direct` tells the members),
    /// except `roots`, with the kind of that shift. Read directly, they take the value of their
    /// sub-program at offset zero instead of computing their expression a second time.
    fn shifted_members(&self, members: &[u32], direct: &[bool], roots: &[u32]) -> Vec<(u32, ShiftKind)> {
        let mut shifted: Vec<(u32, ShiftKind)> = Vec::new();
        for &at in members {
            let item = &self.items[at as usize];
            if let OpKind::Shift(kind) = item.node.op().kind
                && let Src::Node(target) = &item.args[0]
                && target.is_pending()
            {
                let target = item.kids[0];
                if target != NONE
                    && direct[target as usize]
                    && !roots.contains(&target)
                    && !shifted.iter().any(|&(known, _)| known == target)
                {
                    shifted.push((target, zero_shift(kind)));
                }
            }
        }
        shifted
    }
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
    /// Pending nodes (by address) that stand for an input and are not flattened, with it.
    boundary: Vec<(*const Node, Ref)>,
    instrs: Vec<Instr>,
    input_tys: Vec<ElemTy>,
    inputs: Vec<Arc<Storage>>,
    scalar_tys: Vec<ElemTy>,
    scalars: Vec<u32>,
    /// Where the value of every item emitted so far in the program being built lives, by item
    /// (`Ref::Unused`: nowhere yet).
    emitted: Vec<Ref>,
    /// Programs of the expressions read through shifts, and per item the one that computes it
    /// (`NONE` if none does).
    subs: Vec<Vec<Instr>>,
    sub_of: Vec<u32>,
    /// The items computed inside the sub-programs, in emission order, and the item of each
    /// sub-program (see [`Fused::fused_nodes`] and [`Fused::sub_roots`]).
    fused_items: Vec<u32>,
    sub_root_items: Vec<u32>,
    /// The nodes of the sub-programs, kept alive for [`Fused::sub_roots`].
    sub_roots: Vec<Arc<Node>>,
    /// Per item: for those read both through shifts and directly (see [`Graph::shifted_members`])
    /// the kind of the first shift; a direct read is a shift by nothing of their sub-program.
    shifted: Vec<Option<ShiftKind>>,
    pub(super) fused_nodes: Vec<Arc<Node>>,
    /// The node each main-program instruction computes (see [`Fused::instr_nodes`]).
    instr_nodes: Vec<Option<Arc<Node>>>,
    /// Notes where inputs and scalars come from (see [`Recipe`]), if asked to.
    record: Option<Record>,
}

/// Where a scalar of a program comes from.
#[derive(Clone, Copy)]
enum ScalarSrc {
    /// Operand `slot` of item `item`.
    Operand(u32, u8),
    /// A value the flattening made up.
    Const(Scalar),
}

/// What a [`Flattener`] notes about the inputs and scalars of the program it builds.
#[derive(Default)]
struct Record {
    /// The operand being emitted: (item, slot).
    cursor: (u32, u8),
    /// Per input, the operand that made it one.
    inputs: Vec<(u32, u8)>,
    /// Per scalar emitted, where it comes from.
    scalars: Vec<ScalarSrc>,
    /// Set when the program depends on something other than the graph: a node evaluated since the
    /// graph was read.
    impure: bool,
}

impl Flattener {
    /// Makes room for the items of a graph of `n`.
    fn begin(&mut self, n: usize) {
        self.emitted = vec![Ref::Unused; n];
        self.sub_of = vec![NONE; n];
        self.shifted = vec![None; n];
    }

    fn is_boundary(&self, ptr: *const Node) -> bool {
        self.boundary.iter().any(|&(known, _)| known == ptr)
    }

    /// Notes that the program is not a function of the graph alone.
    fn impure(&mut self) {
        if let Some(record) = &mut self.record {
            record.impure = true;
        }
    }

    /// Records where the value of item `at` lives in the program being built; the first stays.
    fn mark(&mut self, at: u32, value: Ref) {
        if let Some(slot) = self.emitted.get_mut(at as usize)
            && *slot == Ref::Unused
        {
            *slot = value;
        }
    }

    /// Emits the instruction of item `at`, whose pending operands were emitted before, and returns
    /// where the result lives.
    fn emit(&mut self, g: &Graph, at: u32) -> Result<Ref, Failure> {
        let item = &g.items[at as usize];
        let mut refs = [Ref::Unused; 3];
        for (index, (slot, arg)) in refs.iter_mut().zip(&item.args).enumerate().take(usize::from(item.arity)) {
            if let Some(record) = &mut self.record {
                record.cursor = (at, index as u8);
            }
            *slot = match arg {
                Src::Scalar(scalar) => match item.slots[index].get() {
                    NONE => {
                        let value = self.scalar(scalar);
                        if let Ref::Scalar(first) = value {
                            item.slots[index].set(first);
                        }
                        if let Some(record) = &mut self.record {
                            record.scalars.push(ScalarSrc::Operand(at, index as u8));
                        }
                        value
                    }
                    first => Ref::Scalar(first),
                },
                Src::Buffer(storage) => self.input(storage, item.tys[index]),
                // A source evaluated since the members were collected (by another thread, say) is
                // read as its buffer: `emitted` may hold the instruction that computes it.
                Src::Node(node) if item.shift && index == 0 && let Some(storage) = node.evaluated() => {
                    if item.kids[0] != NONE {
                        self.impure();
                    }
                    self.input(storage, item.tys[0])
                }
                Src::Node(node) if item.shift && index == 0 && node.is_pending() && !self.is_boundary(Arc::as_ptr(node)) => {
                    self.sub(g, item.kids[0], node)?
                }
                Src::Node(node) => self.node_operand(g, item.kids[index], node, item.tys[index])?,
            };
        }
        let value = u32::try_from(self.instrs.len()).unwrap_or(u32::MAX);
        self.instrs.push(Instr { op: item.node.op, args: refs });
        Ok(Ref::Tmp(value))
    }

    /// The expression of item `kid` (`node`) as a sub-program evaluated at another element (the
    /// operand of a shift), recomputed instead of materialized. Shares the inputs and scalars of
    /// the program.
    fn sub(&mut self, g: &Graph, kid: u32, node: &Arc<Node>) -> Result<Ref, Failure> {
        if self.trace {
            return Err(Failure::New(LumaError::Gpu("traced step shifts an unevaluated expression".into())));
        }
        if let Some(&done) = self.sub_of.get(kid as usize)
            && done != NONE
        {
            return Ok(Ref::Sub(done));
        }
        if kid == NONE || !node.has_operands() {
            // Evaluated (or being evaluated by another thread, which `force` waits for), possibly
            // since it was collected: a shift reads its buffer, never the instruction that
            // computed it in this program.
            self.impure();
            return match node.force() {
                Ok(storage) => Ok(self.input(storage, node.op().out_ty())),
                Err(error) => Err(Failure::Inherited(Arc::clone(error))),
            };
        }
        let n = g.items.len();
        let mut members = Vec::new();
        let mut seen = vec![false; n];
        seen[kid as usize] = true;
        g.collect_kids(kid, false, &mut members, &mut seen);
        let shifted = g.shifted_members(&members, &seen, &[kid]);
        if !shifted.is_empty() {
            members.clear();
            seen.fill(false);
            seen[kid as usize] = true;
            for &(at, _) in &shifted {
                seen[at as usize] = true;
            }
            g.collect_kids(kid, false, &mut members, &mut seen);
            for (at, kind) in shifted {
                if let Some(slot) = self.shifted.get_mut(at as usize)
                    && slot.is_none()
                {
                    *slot = Some(kind);
                }
            }
        }
        let outer_instrs = std::mem::replace(&mut self.instrs, Vec::with_capacity(members.len() + 1));
        let outer_emitted = std::mem::replace(&mut self.emitted, vec![Ref::Unused; n]);
        let result = (|| {
            for &at in &members {
                let value = self.emit(g, at)?;
                self.mark(at, value);
            }
            self.emit(g, kid)
        })();
        let instrs = std::mem::replace(&mut self.instrs, outer_instrs);
        self.emitted = outer_emitted;
        result?;
        self.fused_items.extend_from_slice(&members);
        self.fused_items.push(kid);
        self.subs.push(instrs);
        let done = u32::try_from(self.subs.len() - 1).unwrap_or(u32::MAX);
        if let Some(slot) = self.sub_of.get_mut(kid as usize) {
            *slot = done;
        }
        self.sub_root_items.push(kid);
        Ok(Ref::Sub(done))
    }

    /// The first uniform slot of `scalar`.
    fn scalar(&mut self, scalar: &Scalar) -> Ref {
        let at = u32::try_from(self.scalars.len()).unwrap_or(u32::MAX);
        // The uniform has `f32` fields: a wide scalar takes several (see `Scalar::words`).
        let slot_ty = if scalar.ty.bytes() == 4 { scalar.ty } else { ElemTy::F32 };
        for &word in scalar.words().iter().take(scalar.ty.slots()) {
            self.scalar_tys.push(slot_ty);
            self.scalars.push(word);
        }
        Ref::Scalar(at)
    }

    /// A scalar the flattening makes up.
    fn constant(&mut self, scalar: Scalar) -> Ref {
        if let Some(record) = &mut self.record {
            record.scalars.push(ScalarSrc::Const(scalar));
        }
        self.scalar(&scalar)
    }

    /// Where the pending node operand `node` (item `kid` if it was pending when the graph was
    /// read) of an instruction lives.
    fn node_operand(&mut self, g: &Graph, kid: u32, node: &Arc<Node>, ty: ElemTy) -> Result<Ref, Failure> {
        if kid != NONE {
            if let Some(&found) = self.emitted.get(kid as usize)
                && found != Ref::Unused
            {
                return Ok(found);
            }
            // Not a member: it was evaluated when the members were collected, or it is read
            // through a shift too.
            if let Some(&Some(kind)) = self.shifted.get(kid as usize)
                && node.has_operands()
            {
                return self.unshifted(g, kid, kind);
            }
            self.impure();
        } else if let Some(&(_, found)) = self.boundary.iter().find(|&&(known, _)| known == Arc::as_ptr(node)) {
            return Ok(found);
        }
        if self.trace && node.is_pending() {
            return Err(Failure::New(LumaError::Gpu("traced step reads an unevaluated node".into())));
        }
        if node.evaluated().is_none() {
            self.impure();
        }
        match node.force() {
            Ok(storage) => Ok(self.input(storage, ty)),
            Err(error) => Err(Failure::Inherited(Arc::clone(error))),
        }
    }

    /// The value of the sub-program of item `kid` at the element itself: a shift of `kind` by nothing.
    fn unshifted(&mut self, g: &Graph, kid: u32, kind: ShiftKind) -> Result<Ref, Failure> {
        let node = &g.items[kid as usize].node;
        let sub = self.sub(g, kid, node)?;
        let (amount, total) = match kind {
            ShiftKind::Linear => (self.constant(Scalar::u32(0)), self.constant(Scalar::u32(node.len() as u32))),
            ShiftKind::Dims { .. } | ShiftKind::View { .. } | ShiftKind::Take { .. } => (Ref::Unused, Ref::Unused),
        };
        let value = u32::try_from(self.instrs.len()).unwrap_or(u32::MAX);
        self.instrs.push(Instr { op: Op::new(OpKind::Shift(kind), node.op().out_ty()), args: [sub, amount, total] });
        self.mark(kid, Ref::Tmp(value));
        Ok(Ref::Tmp(value))
    }

    fn input(&mut self, storage: &Arc<Storage>, ty: ElemTy) -> Ref {
        let at = match self.inputs.iter().position(|known| Arc::ptr_eq(known, storage)) {
            Some(at) => at,
            None => {
                self.inputs.push(Arc::clone(storage));
                self.input_tys.push(ty);
                if let Some(record) = &mut self.record {
                    record.inputs.push(record.cursor);
                }
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
                Resolved::Scalar(scalar) => self.flat.scalar(scalar),
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

/// The nodes of `shared` that no other of them reads. Evaluating those first leaves the others
/// without a reader outside the expression (what made them shared) once the readers that made
/// them shared are evaluated: they are then computed inside those kernels, once, instead of in
/// kernels of their own that nothing is left to read.
fn outermost(shared: Vec<Arc<Node>>) -> Vec<Arc<Node>> {
    let outer: Vec<bool> = shared.iter().map(|node| !shared.iter().any(|other| !Arc::ptr_eq(other, node) && other.below_contains(node.id))).collect();
    shared.into_iter().zip(outer).filter_map(|(node, outer)| outer.then_some(node)).collect()
}

/// `shared` and the pending nodes among `candidates` that are its siblings: of the same length,
/// created by the same operation at the same place in the program (the other fields of a record
/// that a loop builds), reading nodes the shared node reads, and not related to it by reading it
/// or being read by it. A shared node evaluated alone would leave everything its siblings
/// share with it with readers outside its own expression, and each such node would be evaluated
/// first on its own, level by level; evaluated together they are one kernel.
fn with_siblings<'a>(shared: Vec<Arc<Node>>, candidates: impl Iterator<Item = &'a Arc<Node>> + Clone) -> Vec<Arc<Node>> {
    let mut forced = shared.clone();
    for node in &shared {
        let below = node.below_vec();
        for other in candidates.clone() {
            if other.len() == node.len()
                && other.name == node.name
                && std::ptr::eq(other.location, node.location)
                && other.is_pending()
                && !forced.iter().any(|known| Arc::ptr_eq(known, other))
                && !node.below_contains(other.id)
                && !other.below_contains(node.id)
                && other.below_vec().iter().any(|id| below.binary_search(id).is_ok())
            {
                forced.push(Arc::clone(other));
            }
        }
    }
    forced
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
                member.consumers.load(Ordering::Relaxed) > inside
                    && super::worth_storing(member)
                    && member.has_consumer_outside(|ptr| seen.contains(ptr))
            })
            .map(|(member, _)| Arc::clone(member))
            .collect();
        if !shared.is_empty() && !is_alone(root) && evaluation_fits(std::iter::once((root, args)), &shared) {
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

thread_local! {
    /// The node [`force_alone`] is evaluating on this thread, if any.
    static ALONE: std::cell::Cell<*const Node> = const { std::cell::Cell::new(std::ptr::null()) };
}

/// Whether `root` is being evaluated by [`force_alone`].
fn is_alone(root: &Arc<Node>) -> bool {
    ALONE.with(|alone| std::ptr::eq(alone.get(), Arc::as_ptr(root)))
}

/// Restores the previous [`ALONE`] node when dropped.
struct Alone(*const Node);

impl Drop for Alone {
    fn drop(&mut self) {
        ALONE.with(|alone| alone.set(self.0));
    }
}

/// Evaluates `node` as one kernel that recomputes the nodes its expression shares with other
/// pending operations, instead of evaluating those first: what `inline::judge` priced when it
/// chose to evaluate the sub-program `node` computes before the program reading it.
fn force_alone(node: &Arc<Node>) -> Result<(), Failure> {
    let _restore = Alone(ALONE.with(|alone| alone.replace(Arc::as_ptr(node))));
    match node.force() {
        Ok(_) => Ok(()),
        Err(error) => Err(Failure::Inherited(Arc::clone(error))),
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
                Judgement::Materialise { subs, reason } => {
                    jit.note_split(first.name, first.location, *reason);
                    let nodes: Vec<Arc<Node>> = subs.iter().filter_map(|&sub| fused.sub_roots.get(sub as usize).cloned()).collect();
                    drop(fused);
                    // A cut forced by the instruction cap takes along the live arrays that
                    // share work with the expression; one the cost model chose is priced as
                    // that expression alone, recomputing what it shares with the rest.
                    let forced = matches!(reason, SplitReason::Inlined { .. });
                    match nodes.as_slice() {
                        [] => {}
                        [node] if !forced => force_alone(node)?,
                        [node] => {
                            jit.evaluate_alongside(Arc::clone(node));
                            if let Err(error) = node.force() {
                                return Err(Failure::Inherited(Arc::clone(error)));
                            }
                        }
                        [first_node, ..] => {
                            if forced {
                                jit.evaluate_alongside(Arc::clone(first_node));
                            }
                            force_group(&nodes)?;
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
    let jit = roots.first().map(|(root, _)| root.context().jit());
    let (graph, root_at, passes, signature) = loop {
        // Every pending node below the roots is read once, here: the walks below only follow
        // indices (see [`Graph`]).
        let mut graph = Graph::new(roots.iter().map(|(root, _)| root), false, Vec::new());
        let added: Vec<(u32, bool)> = roots.iter().map(|(root, args)| graph.add_root(root, args.clone())).collect();
        for &(at, new) in &added {
            if new {
                graph.link_root(at)?;
            }
        }
        let root_at: Vec<u32> = added.iter().map(|&(at, _)| at).collect();
        let n = graph.items.len();
        // Something is shared with another operation only if a node has a reader outside the
        // program and its sub-programs, which are the nodes of the graph. Most nodes have all their
        // readers there, which counting tells without looking at them; only if one may not, the
        // nodes of each program are told apart (below).
        let alone = matches!(roots, [(root, _)] if is_alone(root));
        let root_count = root_at.iter().copied().max().map_or(0, |last| last as usize + 1);
        let mut readers = vec![0u32; n];
        for item in &graph.items {
            for (slot, &kid) in item.kids.iter().enumerate() {
                if kid != NONE && !item.kids[..slot].contains(&kid) {
                    readers[kid as usize] += 1;
                }
            }
        }
        let by_address_map = std::cell::OnceCell::new();
        let by_address = || by_address_map.get_or_init(|| graph.by_address());
        let in_graph = |ptr: *const Node| by_address().contains_key(&ptr);
        let may_share = graph.items.iter().enumerate().skip(root_count).any(|(at, item)| {
            let node = &item.node;
            (node.requested.load(Ordering::Relaxed) && node.is_pending())
                || (!alone && node.consumers.load(Ordering::Relaxed) > readers[at] && super::worth_storing(node) && node.has_consumer_outside(&in_graph))
        });
        if !may_share {
            // The program of a graph of the same shape as one flattened before is that program
            // over the buffers and scalars of this one.
            let signature = jit.and_then(|_| graph.signature(&root_at, tail));
            if let (Some(jit), Some(signature)) = (jit, &signature)
                && let Some(fused) = jit.recipe(signature).and_then(|recipe| recipe.instantiate(&graph, tail))
            {
                #[cfg(any(test, debug_assertions))]
                verify_recipe(&graph, &root_at, tail, &fused)?;
                return Ok(fused);
            }
            let passes = graph.passes(&root_at);
            break (graph, root_at, passes, signature);
        }
        let passes = graph.passes(&root_at);
        let mut sub_members = Vec::new();
        // Only expressions read through shifts add to what was seen.
        let mut with_subs: Option<Vec<bool>> = None;
        for &at in &passes.members {
            let kid = graph.items[at as usize].source();
            if kid != NONE {
                let all_seen = with_subs.get_or_insert_with(|| passes.seen.clone());
                graph.collect(kid, &mut sub_members, all_seen);
            }
        }
        // The shifted members are in `seen` already, so their expressions are collected here.
        for &at in &passes.targets {
            let all_seen = with_subs.get_or_insert_with(|| passes.seen.clone());
            graph.collect_kids(at, false, &mut sub_members, all_seen);
            sub_members.push(at);
        }
        // Sub-programs read further sub-programs through shifts of their own: those expressions
        // are part of the kernel too, and nothing in them is read from outside it.
        let mut next = 0;
        while let Some(&at) = sub_members.get(next) {
            next += 1;
            let kid = graph.items[at as usize].source();
            if kid != NONE {
                let all_seen = with_subs.get_or_insert_with(|| passes.seen.clone());
                graph.collect(kid, &mut sub_members, all_seen);
            }
        }
        let all_seen = with_subs.as_ref().unwrap_or(&passes.seen);
        let inside = |ptr: *const Node| by_address().get(&ptr).is_some_and(|&at| all_seen[at as usize]);
        let is_root = |at: u32| root_at.contains(&at);
        // A node that `eval` asked for (and whose evaluation was postponed) is computed on its
        // own once anyway: read inside a sub-program, it would be recomputed at every instance.
        let requested = |at: u32| {
            let node = &graph.items[at as usize].node;
            node.requested.load(Ordering::Relaxed) && node.is_pending()
        };
        let shared: Vec<Arc<Node>> = passes
            .members
            .iter()
            .chain(&sub_members)
            .filter(|&&at| {
                !alone && !is_root(at) && super::worth_storing(&graph.items[at as usize].node) && graph.items[at as usize].node.has_consumer_outside(&inside)
            })
            .chain(sub_members.iter().filter(|&&at| !is_root(at) && requested(at)))
            .map(|&at| Arc::clone(&graph.items[at as usize].node))
            .collect();
        if shared.is_empty() {
            break (graph, root_at, passes, None);
        }
        let shared = outermost(shared);
        if shared.iter().any(|node| probes.iter().any(|&id| node.depends_on(id))) {
            return Err(Failure::New(LumaError::Gpu("stencil probe shared outside step".into())));
        }
        if !evaluation_fits(roots.iter().map(|(root, args)| (root, args)), &shared) {
            break (graph, root_at, passes, None);
        }
        let candidates = passes.members.iter().chain(&sub_members).filter(|&&at| !is_root(at)).map(|&at| &graph.items[at as usize].node);
        let forced = with_siblings(shared, candidates);
        force_group(&forced)?;
    };

    let (mut fused, recorded) = emit_program(&graph, &root_at, &passes.members, passes.shifted, tail, signature.is_some())?;
    if let (Some(jit), Some(signature), Some(recorded)) = (jit, signature, recorded)
        && !recorded.record.impure
    {
        // The recipe keeps the program every equal one shares (see `JitState::intern`).
        jit.intern(&mut fused.key);
        if let Some(recipe) = Recipe::new(signature, &fused, recorded) {
            jit.keep_recipe(recipe);
        }
    }
    Ok(fused)
}

/// What the main program's walks found for a graph.
struct Passes {
    /// The pending nodes below the roots (not through shifts), each once, operands before the
    /// nodes using them, without the `shifted` ones.
    members: Vec<u32>,
    /// Which items the walk saw (members and the shifted ones).
    seen: Vec<bool>,
    /// The shifted members, with the kind of shift that reads them (see [`Graph::shifted_members`]).
    shifted: Vec<(u32, ShiftKind)>,
    /// The members the walk left out because a shift reads them.
    targets: Vec<u32>,
}

impl Graph {
    fn passes(&self, root_at: &[u32]) -> Passes {
        let n = self.items.len();
        let gather = |seed: &[(u32, ShiftKind)]| -> (Vec<u32>, Vec<bool>) {
            let mut members = Vec::with_capacity(n);
            let mut seen = vec![false; n];
            for &(at, _) in seed {
                seen[at as usize] = true;
            }
            for &at in root_at {
                self.collect_kids(at, true, &mut members, &mut seen);
            }
            for &at in root_at {
                if !seen[at as usize] {
                    seen[at as usize] = true;
                    members.push(at);
                }
            }
            (members, seen)
        };
        let (mut members, mut seen) = gather(&[]);
        let shifted = self.shifted_members(&members, &seen, root_at);
        let mut targets = Vec::new();
        if !shifted.is_empty() {
            let mut is_target = vec![false; n];
            for &(at, _) in &shifted {
                is_target[at as usize] = true;
            }
            targets = members.iter().copied().filter(|&at| is_target[at as usize]).collect();
            (members, seen) = gather(&shifted);
        }
        Passes { members, seen, shifted, targets }
    }
}

/// What [`emit_program`] noted besides the program.
struct Recorded {
    record: Record,
    /// The items whose nodes are computed inside the program, the one each main instruction
    /// computes (`NONE`: none), and the one each sub-program computes.
    fused_items: Vec<u32>,
    instr_items: Vec<u32>,
    sub_root_items: Vec<u32>,
}

/// The program of `members` (the main program) and of the sub-programs they read, over the items
/// of `graph`; `record` notes where its buffers and scalars come from.
fn emit_program(
    graph: &Graph,
    root_at: &[u32],
    members: &[u32],
    shifted: Vec<(u32, ShiftKind)>,
    tail: Tail,
    record: bool,
) -> Result<(Fused, Option<Recorded>), Failure> {
    let mut flattener = Flattener::default();
    if record {
        flattener.record = Some(Record::default());
    }
    flattener.begin(graph.items.len());
    for (at, kind) in shifted {
        flattener.shifted[at as usize] = Some(kind);
    }
    flattener.instrs.reserve(members.len());
    let mut member_at = Vec::with_capacity(members.len());
    for &at in members {
        let value = flattener.emit(graph, at)?;
        flattener.mark(at, value);
        member_at.push(value);
    }
    let mut instr_items = vec![NONE; flattener.instrs.len()];
    for (&at, value) in members.iter().zip(&member_at) {
        if let Ref::Tmp(k) = value
            && let Some(slot) = instr_items.get_mut(*k as usize)
        {
            *slot = at;
        }
    }
    let mut outputs = Vec::with_capacity(root_at.len());
    for &at in root_at {
        match flattener.emitted.get(at as usize) {
            Some(Ref::Tmp(value)) => outputs.push(*value),
            _ => return Err(Failure::New(LumaError::Gpu("fused root without an instruction".into()))),
        }
    }
    let mut fused_items = std::mem::take(&mut flattener.fused_items);
    for &at in members {
        if !root_at.contains(&at) || matches!(tail, Tail::Reduce(_)) {
            fused_items.push(at);
        }
    }
    let sub_root_items = std::mem::take(&mut flattener.sub_root_items);
    let node = |at: u32| Arc::clone(&graph.items[at as usize].node);
    flattener.fused_nodes = fused_items.iter().map(|&at| node(at)).collect();
    flattener.sub_roots = sub_root_items.iter().map(|&at| node(at)).collect();
    flattener.instr_nodes = instr_items.iter().map(|&at| (at != NONE).then(|| node(at))).collect();
    let record = flattener.record.take();
    let fused = flattener.finish(tail, outputs);
    Ok((fused, record.map(|record| Recorded { record, fused_items, instr_items, sub_root_items })))
}

/// Where an operand of a graph stands in the program built from it, by what that program
/// depends on: scalars by type (their values are not part of the program), buffers by the order
/// they are first read in (two reads of one buffer are one input), pending nodes by item.
#[derive(Clone, Copy, PartialEq, Eq)]
enum ArgSig {
    Unused,
    Scalar(ElemTy),
    Input(u32),
    Item(u32),
}

impl ArgSig {
    fn code(self) -> u64 {
        match self {
            ArgSig::Unused => 0,
            ArgSig::Scalar(ty) => 1 | (ty as u64) << 8,
            ArgSig::Input(k) => 2 | u64::from(k) << 8,
            ArgSig::Item(k) => 3 | u64::from(k) << 8,
        }
    }
}

#[derive(PartialEq)]
struct ItemSig {
    op: Op,
    len: usize,
    args: [ArgSig; 3],
}

/// Everything the program built from a graph is a function of, besides the buffers it reads and
/// the values of its scalars: two graphs with the same signature flatten to the same program.
#[derive(PartialEq)]
struct Signature {
    hash: u64,
    items: Vec<ItemSig>,
    roots: Vec<u32>,
    tail: Tail,
}

impl Graph {
    /// The signature of the graph of the roots `root_at`; `None` if an operand is a node that
    /// has no buffer (and is not an item), which the program then depends on.
    fn signature(&self, root_at: &[u32], tail: Tail) -> Option<Signature> {
        use std::hash::{Hash, Hasher};
        let mut buffers: Vec<*const Storage> = Vec::new();
        let mut canon = |storage: &Arc<Storage>| -> u32 {
            let ptr = Arc::as_ptr(storage);
            let at = buffers.iter().position(|&known| known == ptr).unwrap_or_else(|| {
                buffers.push(ptr);
                buffers.len() - 1
            });
            u32::try_from(at).unwrap_or(u32::MAX)
        };
        let mut hasher = crate::gpu::hash::FxHasher::default();
        let mut items = Vec::with_capacity(self.items.len());
        for item in &self.items {
            let mut args = [ArgSig::Unused; 3];
            for (index, (slot, arg)) in args.iter_mut().zip(&item.args).enumerate().take(usize::from(item.arity)) {
                *slot = match (arg, item.kids[index]) {
                    (Src::Scalar(scalar), _) => ArgSig::Scalar(scalar.ty),
                    (Src::Buffer(storage), _) => ArgSig::Input(canon(storage)),
                    (Src::Node(_), kid) if kid != NONE => ArgSig::Item(kid),
                    (Src::Node(node), _) => ArgSig::Input(canon(node.evaluated()?)),
                };
            }
            let (op, len) = (item.node.op, item.node.len);
            op.hash(&mut hasher);
            hasher.write_usize(len);
            for arg in args {
                hasher.write_u64(arg.code());
            }
            items.push(ItemSig { op, len, args });
        }
        root_at.hash(&mut hasher);
        tail.hash(&mut hasher);
        Some(Signature { hash: hasher.finish(), items, roots: root_at.to_vec(), tail })
    }
}

/// The program built from a graph, to be built again from another graph with the same
/// [`Signature`] without walking it: the program itself (the instructions every equal program
/// shares), and where the buffers, scalars and nodes of the program come from in the graph (by
/// item).
pub(super) struct Recipe {
    sig: Signature,
    instrs: Arc<Vec<Instr>>,
    subs: Arc<Vec<Vec<Instr>>>,
    input_tys: Vec<ElemTy>,
    scalar_tys: Vec<ElemTy>,
    outputs: Vec<u32>,
    guard: Option<u32>,
    shape: u64,
    inputs: Vec<(u32, u8)>,
    scalars: Vec<ScalarSrc>,
    fused: Vec<u32>,
    instr_items: Vec<u32>,
    sub_roots: Vec<u32>,
}

impl Recipe {
    fn new(sig: Signature, fused: &Fused, recorded: Recorded) -> Option<Recipe> {
        let Recorded { record, fused_items, instr_items, sub_root_items } = recorded;
        (record.inputs.len() == fused.inputs.len()).then(|| Recipe {
            sig,
            instrs: Arc::clone(&fused.key.instrs),
            subs: Arc::clone(&fused.key.subs),
            input_tys: fused.key.input_tys.clone(),
            scalar_tys: fused.key.scalar_tys.clone(),
            outputs: fused.key.outputs.clone(),
            guard: fused.key.guard,
            shape: fused.key.shape,
            inputs: record.inputs,
            scalars: record.scalars,
            fused: fused_items,
            instr_items,
            sub_roots: sub_root_items,
        })
    }

    /// The program over the buffers, scalars and nodes of `graph`, which has this recipe's
    /// signature. `None` if the graph does not have what the program reads (a node that is not
    /// evaluated any more).
    fn instantiate(&self, graph: &Graph, tail: Tail) -> Option<Fused> {
        let operand = |item: u32, slot: u8| graph.items.get(item as usize).and_then(|item| item.args.get(usize::from(slot)));
        let mut inputs = Vec::with_capacity(self.inputs.len());
        for &(item, slot) in &self.inputs {
            inputs.push(match operand(item, slot)? {
                Src::Buffer(storage) => Arc::clone(storage),
                Src::Node(node) => Arc::clone(node.evaluated()?),
                Src::Scalar(_) => return None,
            });
        }
        let mut scalars = Vec::with_capacity(self.scalar_tys.len());
        for source in &self.scalars {
            let scalar = match *source {
                ScalarSrc::Const(scalar) => scalar,
                ScalarSrc::Operand(item, slot) => match operand(item, slot)? {
                    Src::Scalar(scalar) => *scalar,
                    _ => return None,
                },
            };
            scalars.extend(scalar.words().iter().take(scalar.ty.slots()));
        }
        if self.guard.is_some() {
            scalars.push(0);
        }
        if scalars.len() != self.scalar_tys.len() {
            return None;
        }
        let node = |at: u32| graph.items.get(at as usize).map(|item| Arc::clone(&item.node));
        let fused_nodes: Option<Vec<Arc<Node>>> = self.fused.iter().map(|&at| node(at)).collect();
        let sub_roots: Option<Vec<Arc<Node>>> = self.sub_roots.iter().map(|&at| node(at)).collect();
        let instr_nodes: Option<Vec<Option<Arc<Node>>>> =
            self.instr_items.iter().map(|&at| if at == NONE { Some(None) } else { node(at).map(Some) }).collect();
        Some(Fused {
            key: FusedKey {
                shape: self.shape,
                instrs: Arc::clone(&self.instrs),
                input_tys: self.input_tys.clone(),
                scalar_tys: self.scalar_tys.clone(),
                subs: Arc::clone(&self.subs),
                outputs: self.outputs.clone(),
                tail,
                in_place: Vec::new(),
                spans: Vec::new(),
                tile: None,
                staging: None,
                guard: self.guard,
            },
            inputs,
            scalars,
            fused_nodes: fused_nodes?,
            instr_nodes: instr_nodes?,
            sub_roots: sub_roots?,
        })
    }
}

/// The recipes of a context by the hash of their signatures.
#[derive(Default)]
pub(super) struct RecipeTable {
    map: FxMap<u64, Vec<Arc<Recipe>>>,
    /// Items in the signatures kept.
    items: usize,
}

impl std::fmt::Debug for RecipeTable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "RecipeTable({} signatures, {} items)", self.map.len(), self.items)
    }
}

/// Signatures kept before the table starts again.
const RECIPE_ITEMS: usize = 60_000;

impl JitState {
    /// The recipe whose signature is `signature`, if one was kept.
    fn recipe(&self, signature: &Signature) -> Option<Arc<Recipe>> {
        let candidates = lock(&self.recipes).map.get(&signature.hash)?.clone();
        let found = candidates.into_iter().find(|recipe| recipe.sig == *signature);
        #[cfg(test)]
        if found.is_some() {
            self.recipe_hits.fetch_add(1, Ordering::Relaxed);
        }
        found
    }

    fn keep_recipe(&self, recipe: Recipe) {
        let mut table = lock(&self.recipes);
        if table.items + recipe.sig.items.len() > RECIPE_ITEMS {
            table.map.clear();
            table.items = 0;
        }
        table.items += recipe.sig.items.len();
        table.map.entry(recipe.sig.hash).or_default().push(Arc::new(recipe));
    }
}

/// Checks a program made from a recipe against the one flattening the graph builds.
#[cfg(any(test, debug_assertions))]
fn verify_recipe(graph: &Graph, root_at: &[u32], tail: Tail, fast: &Fused) -> Result<(), Failure> {
    let passes = graph.passes(root_at);
    let (slow, _) = emit_program(graph, root_at, &passes.members, passes.shifted, tail, false)?;
    let same_nodes = |a: &[Arc<Node>], b: &[Arc<Node>]| a.len() == b.len() && a.iter().zip(b).all(|(x, y)| Arc::ptr_eq(x, y));
    let same_inputs = fast.inputs.len() == slow.inputs.len() && fast.inputs.iter().zip(&slow.inputs).all(|(x, y)| Arc::ptr_eq(x, y));
    let same_instr_nodes = fast.instr_nodes.len() == slow.instr_nodes.len()
        && fast.instr_nodes.iter().zip(&slow.instr_nodes).all(|(x, y)| match (x, y) {
            (Some(x), Some(y)) => Arc::ptr_eq(x, y),
            (None, None) => true,
            _ => false,
        });
    assert!(
        fast.key == slow.key
            && fast.scalars == slow.scalars
            && same_inputs
            && same_nodes(&fast.fused_nodes, &slow.fused_nodes)
            && same_nodes(&fast.sub_roots, &slow.sub_roots)
            && same_instr_nodes,
        "a program made from a recipe differs from the flattened one:\nrecipe {fast:?}\nflattened {slow:?}"
    );
    Ok(())
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
            flat.boundary.push((Arc::as_ptr(node), at));
        }
    }
    if roots.iter().enumerate().any(|(i, root)| {
        flat.is_boundary(Arc::as_ptr(root)) || roots[..i].iter().any(|other| Arc::ptr_eq(other, root))
    }) {
        return None;
    }

    let mut graph = Graph::new(roots.iter(), true, flat.boundary.iter().map(|&(ptr, _)| ptr).collect());
    let mut root_at = Vec::with_capacity(roots.len());
    for root in roots {
        if !root.is_pending() {
            return None;
        }
        root_at.push(graph.add_root(root, root.operands()?).0);
    }
    for &at in &root_at {
        graph.link_root(at).ok()?;
    }
    let n = graph.items.len();
    let mut members = Vec::with_capacity(n);
    let mut seen = vec![false; n];
    for &at in &root_at {
        graph.collect_kids(at, true, &mut members, &mut seen);
    }
    for &at in &root_at {
        if !seen[at as usize] {
            seen[at as usize] = true;
            members.push(at);
        }
    }
    let by_address_map = std::cell::OnceCell::new();
    let inside = |ptr: *const Node| {
        flat.is_boundary(ptr) || by_address_map.get_or_init(|| graph.by_address()).get(&ptr).is_some_and(|&at| seen[at as usize])
    };
    if members.iter().any(|&at| !root_at.contains(&at) && graph.items[at as usize].node.has_consumer_outside(&inside)) {
        return None;
    }

    let mut nodes = Vec::with_capacity(members.len());
    flat.begin(n);
    flat.instrs.reserve(members.len());
    for &at in &members {
        let value = flat.emit(&graph, at).ok()?;
        flat.mark(at, value);
        nodes.push(Some(Arc::clone(&graph.items[at as usize].node)));
    }
    let mut outputs = Vec::with_capacity(roots.len());
    for &at in &root_at {
        match flat.emitted.get(at as usize) {
            Some(Ref::Tmp(value)) => outputs.push(*value),
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
