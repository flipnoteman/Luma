//! Lazy evaluation and fusion of elementwise operations.
//!
//! Elementwise operations (operators, math functions, comparisons, `select`, `powf`, `clamp`,
//! and shifts as shifted reads of an evaluated array) do not run when they are written. They add
//! a [`Node`] to a graph of pending expressions, and
//! the resulting array is *lazy*. Everything that can fail on shapes, contexts or poisoned
//! operands is still checked when the node is created, so errors and their source locations
//! behave as they would without laziness.
//!
//! A lazy array is evaluated when something needs its buffer: [`Array::to_vec`], `eval`, `write`,
//! reshaping, passing it to a kernel, a shift or convolution, a compound assignment into a
//! buffer, or [`Context::sync`]. Evaluating a node walks the expression behind it, assigns
//! every distinct input buffer a binding and every scalar a slot of one uniform, and generates a
//! single WGSL kernel for the whole expression ([`codegen`]). Reductions fold the expression into
//! their first pass instead, so `(&a * &b).sum()` never materializes the product.
//!
//! Kernels are cached per context by the *structure* of the expression (operations, types,
//! how values are wired); scalar values are uniforms and not part of the key, so
//! `&x * 2.0` and `&x * 3.0` share one pipeline ([`program`]).
//!
//! # Where fusion stops
//!
//! * An expression holds at most [`JitState::max_nodes`] operations (default
//!   [`DEFAULT_MAX_NODES`], `LUMA_FUSION_MAX_NODES` overrides it) and as many distinct input
//!   buffers as the device can bind. A new node that would exceed either evaluates its biggest
//!   operand first, which splits long chains into several kernels. The biggest operand is not
//!   evaluated alone if other live arrays *of the same length* share pending operations with
//!   it (they read nodes that its expression contains): those are computed in the same
//!   multi-output kernel, as far as the limits allow. Arrays that advance in lock step (`x`, `y`,
//!   `count` of an iteration) then split into a few multi-output kernels at the limit instead of
//!   one single-output kernel per operand, each recomputing what the others compute. Live arrays
//!   that share nothing stay lazy, so a single chain, or a chain beside unrelated arrays,
//!   splits as before. Only if the expression is still too big afterwards is the next biggest
//!   operand evaluated.
//! * When an expression is evaluated, a pending node inside it that another pending, live
//!   operation outside the expression also reads is evaluated on its own first, instead of
//!   being recomputed in both kernels. Sharing within one expression is free: a shared node is
//!   one instruction. (`let t = &a * &b; let u = &t + 1.0; let v = &t * 2.0;` evaluates `t` once
//!   when `u` or `v` is read; if `v` is dropped first, `t` is simply part of `u`'s kernel.)
//!
//! # Hazards
//!
//! A lazy array behaves as if it had been computed when it was written down: it sees the
//! contents its input buffers had *then*. Buffers are overwritten by `write`, compound
//! assignment and custom kernels taking `&mut`; before any of those, every pending node that
//! reads the buffer is evaluated ([`Readers`]). Nodes hold their input buffers, so a buffer is
//! not recycled while a lazy array still needs it. Dropping a lazy array that was never used
//! costs nothing.
//!
//! # Writing in place
//!
//! A store kernel whose output replaces an input that nothing else holds (a dying temporary or
//! a solely owned array) reuses that buffer instead of allocating one, as long as the input is
//! read only at the element being written (not through a shift). Compound assignment on an array
//! whose buffer is shared is such a kernel too, forced onto the shared buffer.

mod codegen;
mod program;
mod stencil;
#[cfg(test)]
mod tests;

use std::panic::Location;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock, PoisonError, Weak};

use crate::array::{Failure, Storage};
use crate::builtin::{BinOp, CmpOp, UnOp};
use crate::context::Context;
use crate::error::{LumaError, Result};
use crate::gpu::hash::FxMap;
use crate::gpu_type::ScalarType;
use crate::kernel::{self, KernelCore, KernelKey};

pub(crate) use program::{evaluate, evaluate_group, evaluate_into, reduce_first_pass};
pub(crate) use stencil::plan as stencil_plan;

/// Length of the requested list at which entries of dropped or computed nodes are pruned.
const REQUESTED_PRUNE_AT: usize = 256;

/// Most operations one fused kernel contains unless configured otherwise.
pub(crate) const DEFAULT_MAX_NODES: usize = 128;
/// Upper bound of the configurable limit (the expression is walked recursively).
const MAX_NODES_LIMIT: usize = 256;
/// Most input buffers of one fused kernel, whatever the device allows.
const MAX_INPUTS: usize = 16;
/// Environment variable overriding [`DEFAULT_MAX_NODES`] for new contexts. `1` makes every
/// operation its own kernel.
const MAX_NODES_ENV: &str = "LUMA_FUSION_MAX_NODES";

/// Locks `mutex`, ignoring poisoning: the guarded lists stay consistent even if a holder
/// panicked.
fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Element type of a lazy operation's operands and results.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum ElemTy {
    F32,
    U32,
}

impl ElemTy {
    pub(crate) fn of<T: ScalarType>() -> std::result::Result<ElemTy, LumaError> {
        match T::wgsl_type() {
            "f32" => Ok(ElemTy::F32),
            "u32" => Ok(ElemTy::U32),
            other => Err(LumaError::InvalidShape(format!("Unsupported element type {other}"))),
        }
    }

    pub(crate) fn wgsl(self) -> &'static str {
        match self {
            ElemTy::F32 => "f32",
            ElemTy::U32 => "u32",
        }
    }
}

/// What a node computes per element.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum OpKind {
    Unary(UnOp),
    Binary(BinOp),
    /// Comparison; the result is a `u32` mask.
    Compare(CmpOp),
    /// `pow(x, exponent)`; operands: x, exponent (a scalar).
    Powf,
    /// `clamp(x, lo, hi)`; operands: x, lo, hi (scalars).
    Clamp,
    /// `mask != 0 ? a : b`; operands: mask (`u32`), a, b.
    Select,
    /// Circular shift of a materialized array; operand 0 is its buffer (see [`ShiftKind`]).
    Shift(ShiftKind),
}

/// Which element a shift reads for element `i`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum ShiftKind {
    /// `(i - s) mod n` over the flat array; `s` (in `0..n`) is the scalar operand 1.
    Linear,
    /// Every dimension of an array of `dims` shifted circularly by `shifts` (each in
    /// `0..dim`). Part of the kernel's key, so the index math works on constants.
    Dims { dims: [u32; 4], shifts: [u32; 4] },
}

impl OpKind {
    pub(crate) fn arity(self) -> usize {
        match self {
            OpKind::Unary(_) | OpKind::Shift(ShiftKind::Dims { .. }) => 1,
            OpKind::Binary(_)
            | OpKind::Compare(_)
            | OpKind::Powf
            | OpKind::Shift(ShiftKind::Linear) => 2,
            OpKind::Clamp | OpKind::Select => 3,
        }
    }
}

/// An operation together with the element type of its operands.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) struct Op {
    pub(crate) kind: OpKind,
    pub(crate) ty: ElemTy,
}

impl Op {
    pub(crate) fn new(kind: OpKind, ty: ElemTy) -> Self {
        Op { kind, ty }
    }

    pub(crate) fn out_ty(self) -> ElemTy {
        match self.kind {
            OpKind::Compare(_) => ElemTy::U32,
            _ => self.ty,
        }
    }

    /// The element type operand `index` must have.
    pub(crate) fn operand_ty(self, index: usize) -> ElemTy {
        match (self.kind, index) {
            (OpKind::Select, 0) | (OpKind::Shift(_), 1) => ElemTy::U32,
            _ => self.ty,
        }
    }
}

/// A scalar operand: a uniform of the fused kernel, never part of its cache key.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Scalar {
    pub(crate) ty: ElemTy,
    pub(crate) bits: u32,
}

impl Scalar {
    pub(crate) fn f32(value: f32) -> Self {
        Scalar { ty: ElemTy::F32, bits: value.to_bits() }
    }

    pub(crate) fn of<T: ScalarType>(value: T) -> std::result::Result<Self, LumaError> {
        let ty = ElemTy::of::<T>()?;
        let bits = bytemuck::try_pod_read_unaligned::<u32>(bytemuck::bytes_of(&value))
            .map_err(|e| LumaError::InvalidShape(format!("Scalar is not 32 bits wide: {e}")))?;
        Ok(Scalar { ty, bits })
    }
}

/// An operand of a node.
#[derive(Clone, Debug)]
pub(crate) enum Src {
    /// A materialized array.
    Buffer(Arc<Storage>),
    /// A pending (or just evaluated) expression.
    Node(Arc<Node>),
    Scalar(Scalar),
}

impl Src {
    /// Filler for the operand slots an operation does not use.
    pub(crate) fn unused() -> Self {
        Src::Scalar(Scalar { ty: ElemTy::U32, bits: 0 })
    }
}

/// Up to three operands; slots beyond the operation's arity are [`Src::unused`].
pub(crate) type Args = [Src; 3];

type NodeResult = std::result::Result<Arc<Storage>, Arc<LumaError>>;

static NEXT_NODE_ID: AtomicU64 = AtomicU64::new(0);

/// One pending elementwise operation.
pub(crate) struct Node {
    ctx: Context,
    op: Op,
    /// Number of elements of the result (and of every array operand).
    len: usize,
    /// The operands. Taken (and so released) when the node is evaluated.
    args: Mutex<Option<Args>>,
    /// Set once the node was evaluated: its buffer, or why that failed.
    result: OnceLock<NodeResult>,
    /// What failures of this node are reported as.
    name: &'static str,
    location: &'static Location<'static>,
    /// Unique per node.
    id: u64,
    /// Ids of the distinct nodes in the expression below this one (the first `below_len`),
    /// followed by the ids of the distinct buffers the expression reads, each part sorted. Upper
    /// bounds once parts of the expression have been evaluated. One allocation for both.
    ids: Box<[u64]>,
    below_len: usize,
    /// Number of nodes this one is an operand of (a pre-filter for sharing checks).
    consumers: AtomicU32,
    /// Set once the node's value was computed inside another node's kernel (or a reduction).
    /// Flushing skips such nodes: they are usually the temporaries of a fused expression.
    fused: AtomicBool,
    /// Whether the expression reads a shifted lazy expression (recomputed, see
    /// [`recompute_shifted`]).
    recomputes: bool,
    /// Pending nodes that read this node's result; handed to its storage when it is evaluated.
    readers: Readers,
}

impl std::fmt::Debug for Node {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Node")
            .field("name", &self.name)
            .field("size", &self.size())
            .field("pending", &self.is_pending())
            .finish()
    }
}

impl Node {
    /// Operations in the expression behind this node, itself included.
    fn size(&self) -> usize {
        self.below_len + 1
    }

    /// Ids of the distinct nodes in the expression below this one, sorted.
    fn below(&self) -> &[u64] {
        self.ids.get(..self.below_len).unwrap_or(&[])
    }

    /// Ids of the distinct buffers the expression reads, sorted.
    fn inputs(&self) -> &[u64] {
        self.ids.get(self.below_len..).unwrap_or(&[])
    }

    pub(crate) fn is_pending(&self) -> bool {
        self.result.get().is_none()
    }

    /// The buffer of a node that was evaluated successfully.
    pub(crate) fn evaluated(&self) -> Option<&Arc<Storage>> {
        match self.result.get() {
            Some(Ok(storage)) => Some(storage),
            _ => None,
        }
    }

    /// The error of a failed evaluation.
    pub(crate) fn failure(&self) -> Option<&Arc<LumaError>> {
        match self.result.get() {
            Some(Err(error)) => Some(error),
            _ => None,
        }
    }

    /// This node as an operand: its buffer if it was evaluated, itself otherwise.
    pub(crate) fn as_src(self: &Arc<Self>) -> std::result::Result<Src, Failure> {
        match self.result.get() {
            Some(Ok(storage)) => Ok(Src::Buffer(Arc::clone(storage))),
            Some(Err(error)) => Err(Failure::Inherited(Arc::clone(error))),
            None => Ok(Src::Node(Arc::clone(self))),
        }
    }

    /// Evaluates the node (once): records its fused kernel into the context's batch and returns
    /// the buffer it writes. Does not wait for the GPU.
    pub(crate) fn force(self: &Arc<Self>) -> &NodeResult {
        self.result.get_or_init(|| self.compute())
    }

    fn compute(self: &Arc<Self>) -> NodeResult {
        let args = lock(&self.args).take();
        let outcome = match args {
            Some(args) => evaluate(self, args),
            None => Err(Failure::New(LumaError::Gpu("lazy node evaluated recursively".into()))),
        };
        match outcome {
            Ok(storage) => {
                let storage = Arc::new(storage);
                storage.readers.absorb(&self.readers);
                Ok(storage)
            }
            // An operand failed earlier: its error already names the operation.
            Err(Failure::Inherited(error)) => Err(error),
            Err(Failure::New(error)) => Err(Arc::new(LumaError::Op {
                op: self.name,
                location: self.location,
                source: Arc::new(error),
            })),
        }
    }

    /// Stores `storage`, computed for this node inside a kernel with several outputs, as the
    /// node's result, unless another thread evaluated it meanwhile.
    fn finish(&self, storage: Storage) {
        let storage = Arc::new(storage);
        if self.result.set(Ok(Arc::clone(&storage))).is_ok() {
            storage.readers.absorb(&self.readers);
            lock(&self.args).take();
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.len
    }

    pub(crate) fn op(&self) -> Op {
        self.op
    }

    pub(crate) fn context(&self) -> &Context {
        &self.ctx
    }

    pub(crate) fn depends_on(&self, storage_id: u64) -> bool {
        self.inputs().binary_search(&storage_id).is_ok()
    }

    /// Whether another pending node outside `group` (node pointers) reads this node. Evaluating
    /// this node separately then beats recomputing it in every consumer's kernel.
    pub(crate) fn has_consumer_outside(&self, group: &[*const Node]) -> bool {
        // Cheap filter: nodes with a single consumer cannot be shared.
        if self.consumers.load(Ordering::Relaxed) < 2 {
            return false;
        }
        // Most readers are inside the group: tell by address and only upgrade the others.
        lock(&self.readers.list).nodes.iter().any(|weak| {
            !group.contains(&weak.as_ptr()) && weak.upgrade().is_some_and(|reader| reader.is_pending())
        })
    }

    /// A copy of the operands of a node that has not been evaluated.
    fn operands(&self) -> Option<Args> {
        lock(&self.args).clone()
    }
}

fn insert_sorted(ids: &mut Vec<u64>, id: u64) {
    if let Err(at) = ids.binary_search(&id) {
        ids.insert(at, id);
    }
}

/// Merges the sorted `other` into the sorted `ids` (a set union), through `tmp`.
fn union_into(ids: &mut Vec<u64>, tmp: &mut Vec<u64>, other: &[u64]) {
    let (Some(&first), Some(&last)) = (other.first(), other.last()) else { return };
    // Operands are mostly nested or disjoint: nothing to interleave.
    if ids.last().is_none_or(|&top| top < first) {
        ids.extend_from_slice(other);
        return;
    }
    if ids.first().is_some_and(|&bottom| bottom > last) {
        tmp.clear();
        tmp.extend_from_slice(other);
        tmp.extend_from_slice(ids);
        std::mem::swap(ids, tmp);
        return;
    }
    tmp.clear();
    tmp.reserve(ids.len() + other.len());
    let (mut a, mut b) = (ids.as_slice(), other);
    while let (Some(&x), Some(&y)) = (a.first(), b.first()) {
        match x.cmp(&y) {
            std::cmp::Ordering::Less => {
                tmp.push(x);
                a = a.get(1..).unwrap_or(&[]);
            }
            std::cmp::Ordering::Greater => {
                tmp.push(y);
                b = b.get(1..).unwrap_or(&[]);
            }
            std::cmp::Ordering::Equal => {
                tmp.push(x);
                a = a.get(1..).unwrap_or(&[]);
                b = b.get(1..).unwrap_or(&[]);
            }
        }
    }
    tmp.extend_from_slice(a);
    tmp.extend_from_slice(b);
    std::mem::swap(ids, tmp);
}

/// Working memory of [`measure`], kept per thread so that measuring allocates nothing.
#[derive(Default)]
struct Measured {
    below: Vec<u64>,
    inputs: Vec<u64>,
    tmp: Vec<u64>,
}

thread_local! {
    static MEASURED: std::cell::RefCell<Measured> = std::cell::RefCell::default();
}

/// The number of distinct nodes below and of distinct buffers read by an operation on `args`.
/// The sorted ids of both stay in this thread's [`Measured`] until the next call, for
/// [`measured_ids`].
fn measure(args: &[Src]) -> (usize, usize) {
    if let Some(node) = only_node(args) {
        return (node.below_len + 1, node.inputs().len());
    }
    MEASURED.with_borrow_mut(|m| {
        let Measured { below, inputs, tmp } = m;
        below.clear();
        inputs.clear();
        let mut seen: Option<&Arc<Node>> = None;
        for arg in args {
            match arg {
                Src::Scalar(_) => {}
                Src::Buffer(storage) => insert_sorted(inputs, storage.id()),
                Src::Node(node) => {
                    // `x * x`: the second operand adds nothing.
                    if seen.is_some_and(|first| Arc::ptr_eq(first, node)) {
                        continue;
                    }
                    seen = seen.or(Some(node));
                    union_into(below, tmp, node.below());
                    insert_sorted(below, node.id);
                    union_into(inputs, tmp, node.inputs());
                }
            }
        }
        (below.len(), inputs.len())
    })
}

/// The node that is the only non-scalar operand (possibly several times) of an operation on
/// `args`, whose ids then are its own plus the node's: nothing to merge.
fn only_node(args: &[Src]) -> Option<&Arc<Node>> {
    let mut found: Option<&Arc<Node>> = None;
    for arg in args {
        match arg {
            Src::Scalar(_) => {}
            Src::Buffer(_) => return None,
            Src::Node(node) => match found {
                Some(first) if !Arc::ptr_eq(first, node) => return None,
                _ => found = Some(node),
            },
        }
    }
    found
}

/// The ids found by the last [`measure`] of `args` on this thread, as stored in a [`Node`]:
/// nodes, then buffers.
fn measured_ids(args: &[Src]) -> Box<[u64]> {
    if let Some(node) = only_node(args) {
        // The operand is the newest node of its expression: its id goes last.
        return node.below().iter().copied().chain(Some(node.id)).chain(node.inputs().iter().copied()).collect();
    }
    MEASURED.with_borrow(|m| m.below.iter().chain(&m.inputs).copied().collect())
}

/// Replaces `args[i]` and every operand that is the same node by its evaluated buffer.
fn evaluate_operand(args: &mut [Src], i: usize) -> std::result::Result<(), Failure> {
    let node = match args.get(i) {
        Some(Src::Node(node)) => Arc::clone(node),
        _ => return Ok(()),
    };
    let storage = match node.force() {
        Ok(storage) => Arc::clone(storage),
        Err(error) => return Err(Failure::Inherited(Arc::clone(error))),
    };
    for arg in args.iter_mut() {
        if matches!(arg, Src::Node(n) if Arc::ptr_eq(n, &node)) {
            *arg = Src::Buffer(Arc::clone(&storage));
        }
    }
    Ok(())
}

/// Adds a lazy operation `op` on `args` (checked by the caller) producing `len` elements.
///
/// This is where fusion is limited: operands that already have a consumer, and the biggest operand
/// while the expression is too large, are evaluated first (see the module documentation).
pub(crate) fn create(
    ctx: &Context,
    len: usize,
    op: Op,
    mut args: Args,
    name: &'static str,
    location: &'static Location<'static>,
) -> std::result::Result<Arc<Node>, Failure> {
    let jit = ctx.jit();
    let arity = op.kind.arity();

    // Operands evaluated since they were created become plain buffers.
    for i in 0..arity {
        let evaluated = match args.get(i) {
            Some(Src::Node(node)) => match node.result.get() {
                Some(Ok(_)) => true,
                Some(Err(error)) => return Err(Failure::Inherited(Arc::clone(error))),
                None => false,
            },
            _ => false,
        };
        if evaluated {
            evaluate_operand(&mut args, i)?;
        }
    }

    let max_nodes = jit.max_nodes();
    let max_inputs = jit.max_inputs;
    // The first time the expression is over a limit, its biggest operand is evaluated together
    // with the live arrays that share pending work with it (see the module documentation)
    // instead of alone.
    let mut batched = false;
    let mut cut = false;
    let (below_len, _) = loop {
        let (below, inputs) = measure(args.get(..arity).unwrap_or(&[]));
        if below < max_nodes && inputs <= max_inputs {
            break (below, inputs);
        }
        if !batched {
            batched = true;
            let seed = args
                .iter()
                .take(arity)
                .filter_map(|arg| match arg {
                    Src::Node(node) => Some(node),
                    _ => None,
                })
                .max_by_key(|node| node.size())
                .map(Arc::clone);
            if let Some(seed) = seed {
                cut = true;
                jit.evaluate_alongside(seed);
                for i in 0..arity {
                    if matches!(args.get(i), Some(Src::Node(node)) if node.result.get().is_some()) {
                        evaluate_operand(&mut args, i)?;
                    }
                }
                continue;
            }
        }
        let biggest = args
            .iter()
            .take(arity)
            .enumerate()
            .filter_map(|(i, a)| match a {
                Src::Node(n) => Some((i, n.size())),
                _ => None,
            })
            .max_by_key(|&(_, size)| size);
        match biggest {
            Some((i, _)) => {
                cut = true;
                evaluate_operand(&mut args, i)?;
            }
            // Only buffers and scalars left: at most three operands, within any sane limit.
            None => break (below, inputs),
        }
    };
    if cut {
        // Without this, a long chain built without `eval` keeps the GPU idle until its end.
        let _ = ctx.submit();
    }
    let recomputes = args.iter().take(arity).enumerate().any(|(i, arg)| match arg {
        Src::Node(operand) => operand.recomputes || (i == 0 && matches!(op.kind, OpKind::Shift(_))),
        _ => false,
    });

    let node = Arc::new(Node {
        ctx: ctx.clone(),
        op,
        len,
        args: Mutex::new(Some(args.clone())),
        result: OnceLock::new(),
        name,
        location,
        id: NEXT_NODE_ID.fetch_add(1, Ordering::Relaxed),
        ids: measured_ids(args.get(..arity).unwrap_or(&[])),
        below_len,
        consumers: AtomicU32::new(0),
        fused: AtomicBool::new(false),
        recomputes,
        readers: Readers::default(),
    });

    for (i, arg) in args.iter().enumerate().take(arity) {
        let repeated = |other: &Src| match (arg, other) {
            (Src::Buffer(a), Src::Buffer(b)) => Arc::ptr_eq(a, b),
            (Src::Node(a), Src::Node(b)) => Arc::ptr_eq(a, b),
            _ => false,
        };
        if args.iter().take(i).any(repeated) {
            continue;
        }
        match arg {
            Src::Buffer(storage) => storage.readers.add(&node),
            Src::Node(operand) => {
                operand.consumers.fetch_add(1, Ordering::Relaxed);
                operand.readers.add(&node);
            }
            Src::Scalar(_) => {}
        }
    }
    jit.register(&node);
    Ok(node)
}

/// Pending lazy nodes that read one buffer (or the result of another pending node).
///
/// Whoever is about to overwrite the buffer calls [`Readers::flush`] first: that evaluates those
/// nodes, recording their reads ahead of the write.
#[derive(Debug, Default)]
pub(crate) struct Readers {
    list: Mutex<ReaderList>,
}

/// Weak references to nodes. Most nodes are read by a single operation, so the first reference
/// lives in the list itself: no allocation until a second one arrives.
#[derive(Debug, Default)]
struct Weaks {
    first: Option<Weak<Node>>,
    rest: Vec<Weak<Node>>,
}

impl Weaks {
    fn push(&mut self, node: Weak<Node>) {
        if self.first.is_none() {
            self.first = Some(node);
        } else {
            self.rest.push(node);
        }
    }

    fn len(&self) -> usize {
        usize::from(self.first.is_some()) + self.rest.len()
    }

    fn iter(&self) -> impl DoubleEndedIterator<Item = &Weak<Node>> {
        self.first.iter().chain(&self.rest)
    }

    fn retain(&mut self, mut keep: impl FnMut(&Weak<Node>) -> bool) {
        if self.first.as_ref().is_some_and(|weak| !keep(weak)) {
            self.first = None;
        }
        self.rest.retain(|weak| keep(weak));
        // `first` leads the list: refill it so that a single reader never needs the vector.
        if self.first.is_none() && !self.rest.is_empty() {
            self.first = Some(self.rest.remove(0));
        }
    }
}

impl IntoIterator for Weaks {
    type Item = Weak<Node>;
    type IntoIter = std::iter::Chain<std::option::IntoIter<Weak<Node>>, std::vec::IntoIter<Weak<Node>>>;

    fn into_iter(self) -> Self::IntoIter {
        self.first.into_iter().chain(self.rest)
    }
}

impl Extend<Weak<Node>> for Weaks {
    fn extend<I: IntoIterator<Item = Weak<Node>>>(&mut self, nodes: I) {
        nodes.into_iter().for_each(|node| self.push(node));
    }
}

#[derive(Debug, Default)]
struct ReaderList {
    nodes: Weaks,
    /// Length at which dead and evaluated entries are dropped next.
    prune_at: usize,
}

/// Most operations of a lazy expression that a shift recomputes at the shifted element instead
/// of evaluating it first. Every shift of it repeats them, but nothing goes through memory.
const MAX_RECOMPUTED: usize = 16;

/// Whether a shift of `node` reads its expression recomputed rather than evaluated. Only one
/// level: an expression that already recomputes a shifted one is evaluated, or every further
/// stencil step would multiply the work by its number of shifts.
pub(crate) fn recompute_shifted(node: &Node) -> bool {
    // A recomputed expression runs inside kernels of segmented programs only if it fits one
    // segment, so segmented programs have no sub-programs.
    node.size() <= MAX_RECOMPUTED
        && !node.recomputes
        && node.len() as u64 * 4 <= node.context().segment_bytes()
}

/// Smallest list length worth pruning.
const PRUNE_MIN: usize = 32;

impl Readers {
    pub(crate) fn add(&self, node: &Arc<Node>) {
        let mut list = lock(&self.list);
        list.nodes.push(Arc::downgrade(node));
        if list.nodes.len() > list.prune_at.max(PRUNE_MIN) {
            list.nodes.retain(|weak| weak.upgrade().is_some_and(|node| node.is_pending()));
            list.prune_at = list.nodes.len().saturating_mul(2);
        }
    }

    fn take(&self) -> Weaks {
        let mut list = lock(&self.list);
        list.prune_at = 0;
        std::mem::take(&mut list.nodes)
    }

    /// Takes over the readers of `other`: a node that was evaluated passes its readers on to the
    /// buffer it produced.
    pub(crate) fn absorb(&self, other: &Readers) {
        let moved = other.take();
        if moved.first.is_none() {
            return;
        }
        lock(&self.list).nodes.extend(moved);
    }

    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        lock(&self.list).nodes.len()
    }

    /// Evaluates every pending node registered here.
    pub(crate) fn flush(&self) {
        for weak in self.take() {
            if let Some(node) = weak.upgrade() {
                // A failure is stored in the node and reported when its array is used.
                let _ = node.force();
            }
        }
    }
}

/// Per-context state of the fusion machinery.
#[derive(Debug)]
pub(crate) struct JitState {
    /// Generated kernels by the structure of their program.
    kernels: Mutex<FxMap<program::FusedKey, Arc<KernelCore>>>,
    /// Time-blocked stencil kernels (see [`stencil`]) by their source.
    stencils: Mutex<FxMap<String, Arc<KernelCore>>>,
    /// Every node created and not yet known to be evaluated, oldest first, so that
    /// [`Context::sync`] can evaluate the live ones.
    live: Mutex<ReaderList>,
    /// Nodes `eval` asked for that are not computed yet; dropped ones are skipped.
    requested: Mutex<Vec<Weak<Node>>>,
    max_nodes: AtomicUsize,
    max_inputs: usize,
    stats: Stats,
}

/// Counters of the fusion machinery, read by tests.
#[derive(Debug, Default)]
struct Stats {
    /// Kernels generated and compiled.
    built: AtomicU64,
    /// Kernel lookups answered from the cache.
    hits: AtomicU64,
    /// Elementwise kernels launched.
    elementwise_launches: AtomicU64,
    /// Fused first passes of reductions launched.
    reduce_launches: AtomicU64,
}

/// A snapshot of the counters of a context's [`JitState`].
#[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct StatsSnapshot {
    pub(crate) built: u64,
    pub(crate) hits: u64,
    pub(crate) elementwise_launches: u64,
    pub(crate) reduce_launches: u64,
}

impl JitState {
    pub(crate) fn new(limits: &wgpu::Limits) -> Self {
        let max_nodes = std::env::var(MAX_NODES_ENV)
            .ok()
            .and_then(|value| value.trim().parse::<usize>().ok())
            .unwrap_or(DEFAULT_MAX_NODES)
            .clamp(1, MAX_NODES_LIMIT);
        // One storage binding of the stage is the output.
        let max_inputs = (limits.max_storage_buffers_per_shader_stage as usize)
            .saturating_sub(1)
            .clamp(1, MAX_INPUTS);
        JitState {
            kernels: Mutex::new(FxMap::default()),
            stencils: Mutex::new(FxMap::default()),
            live: Mutex::new(ReaderList::default()),
            requested: Mutex::new(Vec::new()),
            max_nodes: AtomicUsize::new(max_nodes),
            max_inputs,
            stats: Stats::default(),
        }
    }

    pub(crate) fn max_nodes(&self) -> usize {
        self.max_nodes.load(Ordering::Relaxed)
    }

    #[cfg(test)]
    pub(crate) fn set_max_nodes(&self, max_nodes: usize) {
        self.max_nodes.store(max_nodes.clamp(1, MAX_NODES_LIMIT), Ordering::Relaxed);
    }

    #[cfg(test)]
    pub(crate) fn stats(&self) -> StatsSnapshot {
        StatsSnapshot {
            built: self.stats.built.load(Ordering::Relaxed),
            hits: self.stats.hits.load(Ordering::Relaxed),
            elementwise_launches: self.stats.elementwise_launches.load(Ordering::Relaxed),
            reduce_launches: self.stats.reduce_launches.load(Ordering::Relaxed),
        }
    }

    fn register(&self, node: &Arc<Node>) {
        let mut live = lock(&self.live);
        live.nodes.push(Arc::downgrade(node));
        if live.nodes.len() > live.prune_at.max(PRUNE_MIN) {
            live.nodes.retain(|weak| weak.upgrade().is_some_and(|node| node.is_pending()));
            live.prune_at = live.nodes.len().saturating_mul(2);
        }
    }

    /// Evaluates every lazy array that is still alive, newest first (an expression is looked at
    /// before the operands it fused, which usually died with it). Arrays of the same length are
    /// computed together, as many per kernel as the bindings allow: inputs they share are read
    /// once, and one of them that another uses is computed once.
    pub(crate) fn request(&self, node: &Arc<Node>) {
        let mut requested = lock(&self.requested);
        if requested.len() >= REQUESTED_PRUNE_AT {
            requested.retain(|weak| weak.upgrade().is_some_and(|node| node.is_pending()));
        }
        requested.push(Arc::downgrade(node));
    }

    /// Computes the requested nodes that are still alive, those of one length together in
    /// multi-output kernels.
    pub(crate) fn evaluate_requested(&self) {
        let requested = std::mem::take(&mut *lock(&self.requested));
        let mut pending: Vec<Arc<Node>> =
            requested.into_iter().filter_map(|weak| weak.upgrade()).filter(|node| node.is_pending()).collect();
        while let Some(first) = pending.first() {
            let len = first.len;
            let (same, rest): (Vec<_>, Vec<_>) = pending.into_iter().partition(|node| node.len == len);
            evaluate_group(&same);
            pending = rest;
        }
    }

    pub(crate) fn evaluate_live(&self) {
        let nodes = {
            let mut live = lock(&self.live);
            live.prune_at = 0;
            std::mem::take(&mut live.nodes)
        };
        let mut pending: Vec<Arc<Node>> = nodes.into_iter().rev().filter_map(|weak| weak.upgrade()).collect();
        while let Some(first) = pending.first() {
            let len = first.len;
            let (same, mut rest): (Vec<_>, Vec<_>) = pending.into_iter().partition(|node| node.len == len);
            let mut group: Vec<Arc<Node>> = Vec::new();
            let (mut inputs, mut below) = (Vec::new(), Vec::new());
            for node in same {
                // Computed inside a kernel of an earlier group.
                if !node.is_pending() || node.fused.load(Ordering::Relaxed) {
                    continue;
                }
                // Part of an expression of the group: it is computed inside that kernel and only
                // evaluated on its own afterwards if it is still pending then.
                if below.binary_search(&node.id).is_ok() {
                    rest.push(node);
                    continue;
                }
                let mut more_inputs = inputs.clone();
                node.inputs().iter().for_each(|&id| insert_sorted(&mut more_inputs, id));
                let mut more_below = below.clone();
                node.below().iter().for_each(|&id| insert_sorted(&mut more_below, id));
                let fits = more_inputs.len() + group.len() < self.max_inputs + 1
                    && more_below.len() + group.len() < MAX_NODES_LIMIT;
                if fits || group.is_empty() {
                    (inputs, below) = (more_inputs, more_below);
                    group.push(node);
                } else {
                    rest.push(node);
                }
            }
            program::evaluate_group(&group);
            pending = rest;
        }
    }

    /// Evaluates `seed` (an operand of an expression that got too big) together with the live
    /// pending arrays of its length that read nodes of its expression, newest first, as many as
    /// one kernel takes: [`MAX_NODES_LIMIT`] operations (what [`Context::flush`] allows) and the
    /// bindings the device allows. Computing them together evaluates what they share once.
    fn evaluate_alongside(&self, seed: Arc<Node>) {
        let candidates: Vec<Weak<Node>> = lock(&self.live).nodes.iter().rev().cloned().collect();
        let mut inputs = seed.inputs().to_vec();
        let mut below = seed.below().to_vec();
        insert_sorted(&mut below, seed.id);
        let mut group = vec![seed];
        for weak in candidates {
            // Upgraded one at a time and dropped unless taken: a strong reference keeps operands
            // alive, and with them the buffers that a single-output kernel could overwrite.
            let Some(node) = weak.upgrade() else { continue };
            if node.len != group[0].len
                || !node.is_pending()
                || node.fused.load(Ordering::Relaxed)
                || below.binary_search(&node.id).is_ok()
                || !node.below().iter().any(|id| below.binary_search(id).is_ok())
            {
                continue;
            }
            let mut more_inputs = inputs.clone();
            node.inputs().iter().for_each(|&id| insert_sorted(&mut more_inputs, id));
            let mut more_below = below.clone();
            node.below().iter().for_each(|&id| insert_sorted(&mut more_below, id));
            if more_inputs.len() + group.len() < self.max_inputs + 1 && more_below.len() + group.len() < MAX_NODES_LIMIT {
                (inputs, below) = (more_inputs, more_below);
                insert_sorted(&mut below, node.id);
                group.push(node);
            }
        }
        match group.as_slice() {
            // On its own, nothing but the node holds its operands, which lets its kernel
            // overwrite a dying operand's buffer.
            [node] => {
                let _ = node.force();
            }
            _ => program::evaluate_group(&group),
        }
    }

    /// The kernel for `key`, generated and compiled on first use.
    fn kernel(&self, ctx: &Context, key: &program::FusedKey) -> Result<Arc<KernelCore>> {
        let mut cache = lock(&self.kernels);
        if let Some(core) = cache.get(key) {
            self.stats.hits.fetch_add(1, Ordering::Relaxed);
            return Ok(Arc::clone(core));
        }
        let segment_len = u32::try_from(ctx.segment_bytes() / 4).unwrap_or(u32::MAX);
        let source = codegen::source(key, ctx.tuning(), segment_len);
        let label = key.label();
        let core = Arc::new(kernel::build_core(
            ctx,
            &KernelKey::new(source, "main", None),
            Some(&label),
            true,
        )?);
        self.stats.built.fetch_add(1, Ordering::Relaxed);
        cache.insert(key.clone(), Arc::clone(&core));
        Ok(core)
    }

    /// The stencil kernel with WGSL `source`, compiled on first use.
    fn stencil_kernel(&self, ctx: &Context, source: String, label: &str) -> Result<Arc<KernelCore>> {
        let mut cache = lock(&self.stencils);
        if let Some(core) = cache.get(&source) {
            return Ok(Arc::clone(core));
        }
        let core = Arc::new(kernel::build_core(ctx, &KernelKey::new(source.clone(), "main", None), Some(label), true)?);
        cache.insert(source, Arc::clone(&core));
        Ok(core)
    }
}
