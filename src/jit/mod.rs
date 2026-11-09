//! Lazy evaluation and fusion of elementwise operations.
//!
//! Elementwise operations (operators, math functions, comparisons, `select`, `powf`, `clamp`)
//! do not run when they are written. They add a [`Node`] to a graph of pending expressions, and
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
//!   operand first, which splits long chains into several kernels.
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

mod codegen;
mod program;
#[cfg(test)]
mod tests;

use std::collections::HashMap;
use std::panic::Location;
use std::sync::atomic::{AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock, PoisonError, Weak};

use crate::array::{Failure, Storage};
use crate::builtin::{BinOp, CmpOp, UnOp};
use crate::context::Context;
use crate::error::{LumaError, Result};
use crate::gpu_type::ScalarType;
use crate::kernel::{self, KernelCore, KernelKey};

pub(crate) use program::{evaluate, reduce_first_pass};

/// Most operations one fused kernel contains unless configured otherwise.
pub(crate) const DEFAULT_MAX_NODES: usize = 64;
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
}

impl OpKind {
    pub(crate) fn arity(self) -> usize {
        match self {
            OpKind::Unary(_) => 1,
            OpKind::Binary(_) | OpKind::Compare(_) | OpKind::Powf => 2,
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
            (OpKind::Select, 0) => ElemTy::U32,
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
    /// Ids of the distinct nodes in the expression below this one, sorted. An upper bound once
    /// parts of the expression have been evaluated.
    below: Box<[u64]>,
    /// Ids of the distinct buffers the expression reads, sorted. An upper bound once parts of
    /// the expression have been evaluated.
    inputs: Box<[u64]>,
    /// Number of nodes this one is an operand of (a pre-filter for sharing checks).
    consumers: AtomicU32,
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
        self.below.len() + 1
    }

    pub(crate) fn is_pending(&self) -> bool {
        self.result.get().is_none()
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

    pub(crate) fn len(&self) -> usize {
        self.len
    }

    pub(crate) fn op(&self) -> Op {
        self.op
    }

    pub(crate) fn context(&self) -> &Context {
        &self.ctx
    }

    /// Whether another pending node outside `group` (node pointers) reads this node. Evaluating
    /// this node separately then beats recomputing it in every consumer's kernel.
    pub(crate) fn has_consumer_outside(&self, group: &[*const Node]) -> bool {
        // Cheap filter: nodes with a single consumer cannot be shared.
        if self.consumers.load(Ordering::Relaxed) < 2 {
            return false;
        }
        lock(&self.readers.list).nodes.iter().any(|weak| {
            weak.upgrade().is_some_and(|reader| {
                reader.is_pending() && !group.contains(&Arc::as_ptr(&reader))
            })
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

/// The distinct nodes below and the distinct buffers read by an operation on `args`: sorted ids
/// of both.
fn measure(args: &[Src]) -> (Vec<u64>, Vec<u64>) {
    let mut below: Vec<u64> = Vec::new();
    let mut inputs: Vec<u64> = Vec::new();
    for arg in args {
        match arg {
            Src::Scalar(_) => {}
            Src::Buffer(storage) => insert_sorted(&mut inputs, storage.id()),
            Src::Node(node) => {
                insert_sorted(&mut below, node.id);
                node.below.iter().for_each(|&id| insert_sorted(&mut below, id));
                node.inputs.iter().for_each(|&id| insert_sorted(&mut inputs, id));
            }
        }
    }
    (below, inputs)
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
    let (below, inputs) = loop {
        let (below, inputs) = measure(args.get(..arity).unwrap_or(&[]));
        if below.len() < max_nodes && inputs.len() <= max_inputs {
            break (below, inputs);
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
            Some((i, _)) => evaluate_operand(&mut args, i)?,
            // Only buffers and scalars left: at most three operands, within any sane limit.
            None => break (below, inputs),
        }
    };

    let node = Arc::new(Node {
        ctx: ctx.clone(),
        op,
        len,
        args: Mutex::new(Some(args.clone())),
        result: OnceLock::new(),
        name,
        location,
        id: NEXT_NODE_ID.fetch_add(1, Ordering::Relaxed),
        below: below.into_boxed_slice(),
        inputs: inputs.into_boxed_slice(),
        consumers: AtomicU32::new(0),
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

#[derive(Debug, Default)]
struct ReaderList {
    nodes: Vec<Weak<Node>>,
    /// Length at which dead and evaluated entries are dropped next.
    prune_at: usize,
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

    fn take(&self) -> Vec<Weak<Node>> {
        let mut list = lock(&self.list);
        list.prune_at = 0;
        std::mem::take(&mut list.nodes)
    }

    /// Takes over the readers of `other`: a node that was evaluated passes its readers on to the
    /// buffer it produced.
    fn absorb(&self, other: &Readers) {
        let moved = other.take();
        if moved.is_empty() {
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
    kernels: Mutex<HashMap<program::FusedKey, Arc<KernelCore>>>,
    /// Every node created and not yet known to be evaluated, oldest first, so that
    /// [`Context::sync`] can evaluate the live ones.
    live: Mutex<ReaderList>,
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
            kernels: Mutex::new(HashMap::new()),
            live: Mutex::new(ReaderList::default()),
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

    /// Evaluates every lazy array that is still alive, newest first: an expression is evaluated
    /// as a whole before the operands it fused (which usually died with it) are looked at.
    pub(crate) fn evaluate_live(&self) {
        let nodes = {
            let mut live = lock(&self.live);
            live.prune_at = 0;
            std::mem::take(&mut live.nodes)
        };
        for weak in nodes.into_iter().rev() {
            if let Some(node) = weak.upgrade() {
                let _ = node.force();
            }
        }
    }

    /// The kernel for `key`, generated and compiled on first use.
    fn kernel(&self, ctx: &Context, key: program::FusedKey) -> Result<Arc<KernelCore>> {
        let mut cache = lock(&self.kernels);
        if let Some(core) = cache.get(&key) {
            self.stats.hits.fetch_add(1, Ordering::Relaxed);
            return Ok(Arc::clone(core));
        }
        let source = codegen::source(&key);
        let core = Arc::new(kernel::build_core(
            ctx,
            &KernelKey::new(source, "main", None),
            Some("luma fused"),
        )?);
        self.stats.built.fetch_add(1, Ordering::Relaxed);
        cache.insert(key, Arc::clone(&core));
        Ok(core)
    }
}
