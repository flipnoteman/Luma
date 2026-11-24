//! Reductions: whole-array sums, extrema and norms, fused with the expression behind them. They
//! are lazy: the result is a one-element array, and the host reads it with [`Array::item`].

use std::panic::Location;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};

use crate::array::{Array, Failure, Storage};
use crate::builtin::{BinOp, Builtin, Epilogue, EpilogueStep, Finish, ReduceList, ReduceOp, UnOp, MAX_ACCS, MAX_EPILOGUE, MAX_REDUCTIONS};
use crate::context::{Context, RawArg, Scratch};
use crate::error::{LumaError, Result};
use crate::gpu_type::{ElemLayout, GpuType, Numeric, ScalarType};
use crate::jit::{self, ElemTy, Node, Op, OpKind, Src};
use crate::ops::{fold_workgroups, len_u32, run_select, Condition, Rhs};
use crate::shape::Shape;

/// A reduction of a pending expression that has not been recorded yet: the operand is kept as an
/// expression until the result is used on the GPU (then the passes are recorded into a buffer,
/// without reading anything back) or on the host (then it is read like a synchronous reduction,
/// which lets the speculation of [`crate::jit`] answer loops of readbacks).
///
/// Elementwise operations with scalars on the result (`(x.sum() / n).sqrt()`) do not create
/// expressions: they extend the reduction by an epilogue that its last pass applies, so the
/// whole is as many launches as the reduction alone (see [`Array::fuse_step`]). Such a reduction
/// shares the operand with the one it extends but never reads its buffer (which a write through
/// another array may change): if both are used they run in one kernel with the others that are
/// pending (see `jit::multireduce`), or separately where that is not possible.
pub(crate) struct Reduced {
    operand: Arc<Node>,
    ops: ReduceList,
    wgsl: &'static str,
    /// Bytes of one element of the partial results (a `bool` reduces as a `u32` word).
    elem: u64,
    /// How the results are stored: the layout of the element type.
    layout: ElemLayout,
    /// Whether the operand only reads a buffer (the reduction of an array that is already
    /// evaluated), which no speculation of readbacks concerns.
    buffer: bool,
    /// The operations applied to the reduction's values (none for a plain reduction).
    finish: Finish,
    result: OnceLock<std::result::Result<Arc<Storage>, Arc<LumaError>>>,
    /// The element type, and where the reduction was asked for (for the node that stands for its value).
    ty: ElemTy,
    name: &'static str,
    location: &'static Location<'static>,
    /// The nodes that stand for the value in expressions (see [`Reduced::leaf`]): they get the buffer when the passes are
    /// recorded for any other reason, so that what reads them is a reader of that buffer (a write into it evaluates it first).
    leaves: std::sync::Mutex<Vec<std::sync::Weak<Node>>>,
    /// The thread that made the reduction: only that one's reductions are recorded together (see [`crate::jit`]'s `multireduce`).
    thread: std::thread::ThreadId,
    /// Whether it was made while this thread was recording a step, whose reductions are never taken along.
    in_recording: bool,
    /// Set while the passes are being recorded.
    forcing: AtomicBool,
    /// Set when the group this reduction was recorded in was not recorded in one kernel because that was not due or ready (see
    /// `Tuning::reduce_multi_after`): it is recorded alone, so that the others do not form other groups.
    solo: AtomicBool,
    /// Where the values are when they were recorded together with other reductions' (see [`PackRead`]), and the first one's index there.
    pack: OnceLock<(Arc<PackRead>, usize)>,
    /// The lines of an axis reduction (made by [`Array::reduce_axis`] and the methods built on it); `None` for a reduction of
    /// the whole array. It has one op, and its buffer holds one value per line.
    axis: Option<AxisGeom>,
}

/// Most kernels of reductions whose compile [`Reduced::build_ahead`] starts at once.
const AHEAD_KERNELS: usize = 8;

/// The lines an axis reduction runs over: `lines` lines of `n` cells, `stride` apart.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct AxisGeom {
    pub(crate) n: u32,
    pub(crate) stride: u32,
    pub(crate) lines: u32,
}

/// The buffer the reductions recorded in one kernel share (see `Reduced::record_merged`), and its words once a reader has
/// copied them to the host: the first of those reductions that is read on the host reads them all, so that reading a dozen
/// of them costs one round trip.
struct PackRead {
    pack: Arc<Storage>,
    words: OnceLock<Vec<u32>>,
}

impl std::fmt::Debug for Reduced {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Reduced")
            .field("ops", &self.ops)
            .field("steps", &self.finish.epilogue)
            .field("forced", &self.result.get().is_some())
            .finish()
    }
}

impl Reduced {
    /// The node whose values the reduction reduces.
    pub(crate) fn operand(&self) -> &Arc<Node> {
        &self.operand
    }

    pub(crate) fn ops(&self) -> ReduceList {
        self.ops
    }

    pub(crate) fn finish(&self) -> Finish {
        self.finish
    }

    pub(crate) fn ty(&self) -> ElemTy {
        self.ty
    }

    pub(crate) fn layout(&self) -> ElemLayout {
        self.layout
    }

    /// What a kernel that reduces `operand` itself needs to run this reduction (see
    /// `jit::until::force_with_record`): the operand, the ops, what is applied to their values.
    /// `None` unless it is a reduction of `f32`s that has not been recorded, and has not failed.
    pub(crate) fn fusable(&self) -> Option<(&Arc<Node>, ReduceList, Finish)> {
        (self.axis.is_none() && self.wgsl == "f32" && self.ty == ElemTy::F32 && self.is_unforced() && self.failure().is_none()).then_some((&self.operand, self.ops, self.finish))
    }

    /// The error of the operand's evaluation or of the recorded passes, if any.
    pub(crate) fn failure(&self) -> Option<&Arc<LumaError>> {
        self.operand.failure().or_else(|| self.result.get().and_then(|result| result.as_ref().err()))
    }

    /// The buffer of the reduction if its passes have been recorded (or the error), without recording them.
    pub(crate) fn result_if_recorded(&self) -> Option<&std::result::Result<Arc<Storage>, Arc<LumaError>>> {
        self.result.get()
    }

    /// A node standing for the value of this reduction in expressions: evaluating it records the passes.
    pub(crate) fn leaf(self: &Arc<Self>, ctx: &Context, len: usize) -> Arc<Node> {
        let leaf = jit::deferred(ctx, len, self.ty, Arc::clone(self), self.name, self.location);
        let mut leaves = crate::gpu::lock(&self.leaves);
        // Dead leaves (their expressions were dropped) are pruned as the list doubles.
        if leaves.len() >= 32 && leaves.len().is_power_of_two() {
            leaves.retain(|weak| weak.strong_count() > 0);
        }
        leaves.push(Arc::downgrade(&leaf));
        leaf
    }

    /// Whether the passes have not been recorded into a buffer yet.
    pub(crate) fn is_unforced(&self) -> bool {
        self.result.get().is_none()
    }

    /// Whether this reduction may be recorded together with others: a plain `f32`, `u32` or `i32` reduction of an
    /// expression that is still pending, outside a recording.
    pub(crate) fn mergeable(&self, ctx: &Context) -> bool {
        ctx.tuning().reduce_multi >= 2 && !self.in_recording && !ctx.recording_here() && !ctx.check_finite() && self.eligible()
    }

    /// What [`Reduced::mergeable`] says without the context.
    fn eligible(&self) -> bool {
        self.axis.is_none()
            && matches!(self.ty, ElemTy::F32 | ElemTy::U32 | ElemTy::I32)
            && self.wgsl == self.ty.wgsl()
            && !self.layout.is_packed()
            && self.is_unforced()
            && !self.solo.load(Ordering::Relaxed)
            && self.operand.is_pending()
            && self.operand.has_operands()
            && self.failure().is_none()
    }

    /// Whether this reduction can be recorded together with `seed` (itself mergeable).
    pub(crate) fn partners_with(&self, seed: &Reduced) -> bool {
        self.thread == seed.thread
            && !self.in_recording
            && !self.forcing.load(Ordering::Relaxed)
            && self.ty == seed.ty
            && self.operand.len() == seed.operand.len()
            && self.eligible()
    }

    /// Whether this is an axis reduction (see [`Reduced::axis`]), which is recorded with its own kind only.
    pub(crate) fn is_axis(&self) -> bool {
        self.axis.is_some()
    }

    /// Whether this axis reduction may be recorded together with others of the same lines: a plain `f32` reduction of an
    /// expression that is still pending, outside a recording.
    pub(crate) fn axis_mergeable(&self, ctx: &Context) -> bool {
        ctx.tuning().reduce_multi >= 2 && !self.in_recording && !ctx.recording_here() && !ctx.check_finite() && self.axis_eligible()
    }

    /// What [`Reduced::axis_mergeable`] says without the context.
    fn axis_eligible(&self) -> bool {
        self.axis.is_some()
            && self.ty == ElemTy::F32
            && self.wgsl == "f32"
            && self.is_unforced()
            && self.operand.is_pending()
            && self.operand.has_operands()
            && self.failure().is_none()
    }

    /// Whether this axis reduction can be recorded together with the axis reduction `seed` (itself mergeable): the same lines.
    pub(crate) fn axis_partners_with(&self, seed: &Reduced) -> bool {
        self.thread == seed.thread && !self.in_recording && !self.forcing.load(Ordering::Relaxed) && self.axis == seed.axis && self.axis_eligible()
    }

    /// Whether this reduction can be recorded together with `seed`, which is mergeable (of whichever kind it is).
    pub(crate) fn partner_of(&self, seed: &Reduced) -> bool {
        if seed.axis.is_some() { self.axis_partners_with(seed) } else { self.partners_with(seed) }
    }

    /// The pending reductions to record together with this one, first running the reductions its operand
    /// waits for. Empty if it cannot be merged.
    pub(crate) fn merge_partners(&self, ctx: &Context) -> Vec<Arc<Reduced>> {
        let mergeable = if self.axis.is_some() { self.axis_mergeable(ctx) } else { self.mergeable(ctx) };
        if !mergeable || !ctx.jit().may_have_partners(self) {
            return Vec::new();
        }
        let Some(leaves) = jit::unresolved_leaves(&self.operand) else { return Vec::new() };
        if !leaves.is_empty() {
            for leaf in leaves {
                let _ = leaf.force();
            }
            // Running them may have left others (a reduction that depends on a reduction).
            if !jit::unresolved_leaves(&self.operand).is_some_and(|leaves| leaves.is_empty()) {
                return Vec::new();
            }
        }
        ctx.jit().reduction_partners(ctx, self, &[])
    }

    /// Records the reductions `partners` (see [`Reduced::merge_partners`]) and this one in one kernel: this reduction's
    /// buffer, once the others hold theirs. `None` (nothing recorded) if the program does not fit one kernel.
    fn record_merged(&self, ctx: &Context, partners: &[Arc<Reduced>]) -> Option<Arc<Storage>> {
        let all = || std::iter::once(self).chain(partners.iter().map(|partner| &**partner));
        let members: Vec<jit::Member<'_>> = all().map(|reduced| jit::Member { operand: &reduced.operand, ops: reduced.ops, finish: reduced.finish }).collect();
        let ahead = || self.build_ahead(ctx, partners);
        let deferred = std::cell::Cell::new(false);
        let pack = match jit::reduce_many(ctx, &members, self.ty, ctx.tuning().reduce_multi, jit::ManyMode::Run { ahead: &ahead, deferred: &deferred }) {
            Ok(Some(pack)) => Arc::new(pack),
            _ => {
                if deferred.get() {
                    for reduced in all() {
                        reduced.solo.store(true, Ordering::Relaxed);
                    }
                }
                return None;
            }
        };
        drop(members);
        let shared = Arc::new(PackRead { pack: Arc::clone(&pack), words: OnceLock::new() });
        let mut offset = 0;
        let mut own = None;
        for (k, reduced) in all().enumerate() {
            let count = reduced.ops.as_slice().len();
            let window = Arc::new(Storage::window_of(&pack, offset, count));
            let _ = reduced.pack.set((Arc::clone(&shared), offset));
            offset += count;
            if k == 0 {
                own = Some(window);
            } else {
                reduced.adopt(window);
            }
        }
        own
    }

    /// Whether this reduction would be the seed of a kernel recorded after the one of `first`, which it does not share a kernel with.
    pub(crate) fn follows(&self, ctx: &Context, first: &Reduced) -> bool {
        self.thread == first.thread && self.axis.is_none() && !self.in_recording && !self.forcing.load(Ordering::Relaxed) && self.mergeable(ctx)
    }

    /// Starts the compiles of the kernels the reductions pending after `first_group` (this one and its `partners`) will be
    /// recorded in, if they are used in the order they were made, so that the compiles run side by side: a kernel that is not
    /// compiled yet is waited for where it is launched, and the groups are recorded one after the other (a force monitor of
    /// several dozen sums is a few kernels, each compiled from scratch the first time). Runs only when the kernel of the
    /// first group was not built before, so it costs nothing where the kernels are known.
    fn build_ahead(&self, ctx: &Context, partners: &[Arc<Reduced>]) {
        let mut taken: Vec<*const Reduced> = std::iter::once(self as *const Reduced).chain(partners.iter().map(Arc::as_ptr)).collect();
        for _ in 0..AHEAD_KERNELS {
            let Some(seed) = ctx.jit().next_seed(ctx, self, &taken) else { return };
            taken.push(Arc::as_ptr(&seed));
            let others = ctx.jit().reduction_partners(ctx, &seed, &taken);
            taken.extend(others.iter().map(Arc::as_ptr));
            // Alone, a reduction is recorded by its own passes, not by this kind of kernel.
            if others.is_empty() {
                continue;
            }
            let members: Vec<jit::Member<'_>> =
                std::iter::once(&*seed).chain(others.iter().map(|other| &**other)).map(|reduced| jit::Member { operand: &reduced.operand, ops: reduced.ops, finish: reduced.finish }).collect();
            // A program that does not fit is left to the reductions' own turn.
            let _ = jit::reduce_many(ctx, &members, seed.ty, ctx.tuning().reduce_multi, jit::ManyMode::BuildOnly);
        }
    }

    /// Takes the buffer `storage` recorded together with another reduction as the result, if there is none yet.
    fn adopt(&self, storage: Arc<Storage>) {
        if self.result.set(Ok(Arc::clone(&storage))).is_ok() {
            for leaf in std::mem::take(&mut *crate::gpu::lock(&self.leaves)) {
                if let Some(leaf) = leaf.upgrade() {
                    leaf.finish_shared(Arc::clone(&storage));
                }
            }
        }
    }

    /// The values of a reduction recorded together with others, read from the host copy of the buffer they share
    /// (made by the first read of any of them); `None` if it was not, or its own buffer was made since (a window
    /// that is pinned gets a copy of its range, see [`Storage::window_of`]).
    pub(crate) fn packed_values<T: GpuType>(&self, ctx: &Context) -> Option<Result<Vec<T>>> {
        if std::mem::size_of::<T::Repr>() != 4 {
            return None;
        }
        let (shared, offset) = self.pack.get()?;
        if !self.result.get()?.as_ref().is_ok_and(|storage| storage.is_window()) {
            return None;
        }
        let words = match shared.words.get() {
            Some(words) => words,
            None => match ctx.read_storage::<u32>(&shared.pack) {
                Ok(words) => shared.words.get_or_init(|| words),
                Err(error) => return Some(Err(error)),
            },
        };
        let count = self.ops.as_slice().len();
        Some(Ok(T::from_repr(bytemuck::pod_collect_to_vec(words.get(*offset..offset + count)?), count)))
    }

    /// Records the passes into a new buffer (once) and returns it. Nothing is read back.
    pub(crate) fn force(&self, ctx: &Context) -> &std::result::Result<Arc<Storage>, Arc<LumaError>> {
        let result = self.record_once(ctx);
        if let Ok(storage) = result {
            for leaf in std::mem::take(&mut *crate::gpu::lock(&self.leaves)) {
                if let Some(leaf) = leaf.upgrade() {
                    leaf.finish_shared(Arc::clone(storage));
                }
            }
        }
        result
    }

    /// [`Reduced::force`] for the node that stands for the value: it takes the buffer itself.
    pub(crate) fn record_once(&self, ctx: &Context) -> &std::result::Result<Arc<Storage>, Arc<LumaError>> {
        self.result.get_or_init(|| match crate::host::try_eval_reduced(self, ctx) {
            Some(storage) => Ok(storage),
            None => {
                self.forcing.store(true, Ordering::Relaxed);
                let partners = self.merge_partners(ctx);
                let result = self.record(ctx, &partners).map_err(Arc::new);
                self.forcing.store(false, Ordering::Relaxed);
                result
            }
        })
    }

    /// The buffer of the reduction computed on the host, or the one already recorded; `None` if
    /// it cannot run on the host and has not been recorded.
    pub(crate) fn host_result(&self, ctx: &Context) -> Option<Arc<Storage>> {
        if self.result.get().is_none()
            && let Some(storage) = crate::host::try_eval_reduced(self, ctx)
        {
            let _ = self.result.set(Ok(storage));
        }
        self.result.get()?.as_ref().ok().cloned()
    }

    fn record(&self, ctx: &Context, partners: &[Arc<Reduced>]) -> Result<Arc<Storage>> {
        if let Some(geom) = self.axis {
            return self.record_axis(ctx, geom, partners);
        }
        if !partners.is_empty()
            && let Some(storage) = self.record_merged(ctx, partners)
        {
            return Ok(storage);
        }
        let count = self.ops.as_slice().len();
        let storage = Arc::new(Storage::new(ctx, count, self.layout));
        let pin = storage.pin(0)?;
        let array = Array::<u32>::from_node(ctx, Shape::from(self.operand.len()), Arc::clone(&self.operand));
        let mut scratch = Scratch::new(ctx);
        reduce_into(&array, self.ty == ElemTy::Bool, self.elem, self.wgsl, self.ops, &mut scratch, Some(pin.buffer()), self.finish)?;
        Ok(storage)
    }

    /// Records this axis reduction into a buffer of one value per line, in one kernel with the axis reductions `partners` of the
    /// same lines (see [`Reduced::merge_partners`]) where the program of their operands fits one: each of them then holds its own buffer.
    fn record_axis(&self, ctx: &Context, geom: AxisGeom, partners: &[Arc<Reduced>]) -> Result<Arc<Storage>> {
        let all: Vec<&Reduced> = std::iter::once(self).chain(partners.iter().map(|partner| &**partner)).collect();
        let bytes = geom.lines as u64 * 4;
        let plan = if all.len() > 1 {
            let operands: Vec<&Arc<Node>> = all.iter().map(|reduced| &reduced.operand).collect();
            jit::axis_plan(ctx, &operands, all.len())
        } else {
            None
        };
        // Without a plan the others are left for their own turn.
        let all = if plan.is_some() { all } else { vec![self] };
        let storages: Vec<Arc<Storage>> = all.iter().map(|reduced| Arc::new(Storage::new(ctx, geom.lines as usize, reduced.layout))).collect();
        let pins = storages.iter().map(|storage| storage.pin(0)).collect::<Result<Vec<_>>>()?;
        let outs: Vec<crate::scan::Bound<'_>> = pins.iter().map(|pin| crate::scan::Bound { buffer: pin.buffer(), bytes }).collect();
        match plan {
            Some(plan) => {
                let ops: Vec<ReduceOp> = all.iter().map(|reduced| reduced.ops.as_slice()[0]).collect();
                crate::scan::reduce_axis_planned(ctx, &plan, geom, &ops, &outs).map_err(Failure::into_error)?;
                drop(outs);
                drop(pins);
                for (reduced, storage) in all.iter().zip(&storages).skip(1) {
                    reduced.adopt(Arc::clone(storage));
                }
            }
            None => crate::scan::reduce_axis_node(ctx, &self.operand, geom, self.ops.as_slice()[0], outs[0]).map_err(Failure::into_error)?,
        }
        Ok(Arc::clone(&storages[0]))
    }

    /// The values on the host, without keeping them on the GPU.
    pub(crate) fn read<T: GpuType>(&self, ctx: &Context) -> Result<Vec<T>> {
        if let Some(geom) = self.axis {
            // One value per line: recorded (with the axis reductions that are pending) and read like any array.
            let storage = self.force(ctx).as_ref().map_err(|error| LumaError::clone(error))?;
            return Ok(T::from_repr(ctx.read_storage::<T::Repr>(storage)?, geom.lines as usize));
        }
        if std::mem::size_of::<T::Repr>() == 4
            && let Some(words) = crate::host::reduced_words(self, ctx)
        {
            return Ok(T::from_repr(bytemuck::pod_collect_to_vec(&words), self.ops.as_slice().len()));
        }
        // Read with others that are pending, in one kernel, which records them all (before the speculation, which
        // records one reduction at a time).
        if self.is_unforced() && std::mem::size_of::<T::Repr>() == 4 {
            let partners = self.merge_partners(ctx);
            if !partners.is_empty() {
                self.forcing.store(true, Ordering::Relaxed);
                let result = self.result.get_or_init(|| self.record(ctx, &partners).map_err(Arc::new));
                self.forcing.store(false, Ordering::Relaxed);
                let storage = result.as_ref().map_err(|error| LumaError::clone(error))?;
                let values = match self.packed_values::<T>(ctx) {
                    Some(values) => values?,
                    None => T::from_repr(ctx.read_storage::<T::Repr>(storage)?, self.ops.as_slice().len()),
                };
                // The leaves that stand for this value in expressions get its buffer, as `force` does.
                self.force(ctx);
                return Ok(values);
            }
        }
        let array = Array::<T>::from_node(ctx, Shape::from(self.operand.len()), Arc::clone(&self.operand));
        reduce_array(&array, self.wgsl, self.ops, self.finish, !self.buffer)
    }
}

impl<T: Numeric> Array<T> {
    /// The sum of all elements, as a lazy one-element array: [`Array::item`] reads the value on
    /// the host, and using the array in an expression with `self` (`&x / &x.sum()`) keeps it on
    /// the GPU. Float sums combine in trees and over at most a few hundred sequential steps per
    /// invocation, so the error stays far below that of a sequential sum. `u32` sums wrap around.
    #[track_caller]
    pub fn sum(&self) -> Array<T> {
        self.reduce_at([Reduction::Sum], "sum", Location::caller())
    }

    /// The smallest element, as a lazy one-element array (see [`Array::sum`]).
    #[track_caller]
    pub fn min(&self) -> Array<T> {
        self.reduce_at([Reduction::Min], "min", Location::caller())
    }

    /// The largest element, as a lazy one-element array (see [`Array::sum`]).
    #[track_caller]
    pub fn max(&self) -> Array<T> {
        self.reduce_at([Reduction::Max], "max", Location::caller())
    }

    /// The product of all elements, as a lazy one-element array (see [`Array::sum`]): NaN if any
    /// element is NaN, and (as for the sum) `u32` and `i32` products wrap around. Float products combine in trees like sums do, so the rounding error stays that
    /// of `n` multiplications; a partial product that leaves the range of the type becomes
    /// infinity (or 0) even where later factors would bring it back, and which partial products
    /// there are depends on the grouping. Fuses with the expression behind it
    /// (`(&x + 1.0).prod()` is one pass) and with scalar operations on the result.
    #[track_caller]
    pub fn prod(&self) -> Array<T> {
        self.reduce_at([Reduction::Product], "prod", Location::caller())
    }

    /// The product of all elements: the same as [`Array::prod`] (ArrayFire's name).
    #[track_caller]
    pub fn product(&self) -> Array<T> {
        self.reduce_at([Reduction::Product], "product", Location::caller())
    }

    /// Several reductions of the array in one pass over it (up to four), as an array of `N`
    /// elements: `let [lo, hi] = x.reduce([Reduction::Min, Reduction::Max]).to_array()?;`.
    /// Reading the data once instead of once per reduction is what makes it faster than
    /// separate calls on large arrays. Lazy like [`Array::sum`].
    #[track_caller]
    pub fn reduce<const N: usize>(&self, ops: [Reduction; N]) -> Array<T> {
        self.reduce_at(ops, "reduce", Location::caller())
    }

    /// The sum of the elements where `cond` is true (an `Array<bool>`, or an `Array<u32>` of
    /// non-zero elements): `select(cond, self, 0).sum()` in one pass. Lazy like [`Array::sum`].
    #[track_caller]
    pub fn sum_where(&self, cond: &impl Condition) -> Array<T> {
        let location = Location::caller();
        let cond = cond.condition();
        let selected = run_select(cond, Rhs::Array(self), Rhs::Scalar(T::zeroed()), location);
        let selected = Array::resolve(self.context(), cond.shape(), selected, "sum_where", location);
        selected.reduce_at([Reduction::Sum], "sum_where", location)
    }
}

impl Array<bool> {
    /// The number of true elements, as a lazy one-element `u32` array (NumPy's `sum` of a
    /// boolean array is an integer too; see [`Array::sum`]). An array that has been evaluated
    /// is counted a word of 32 elements at a time.
    #[track_caller]
    pub fn sum(&self) -> Array<u32> {
        self.count_at("sum", Location::caller())
    }

    /// The number of true elements: the same as [`Array<bool>::sum`](Array::sum).
    #[track_caller]
    pub fn count(&self) -> Array<u32> {
        self.count_at("count", Location::caller())
    }

    fn count_at(&self, name: &'static str, location: &'static Location<'static>) -> Array<u32> {
        if self.pending().is_some() {
            return self.cast_at::<u32>(location).reduce_at([Reduction::Sum], name, location);
        }
        let result = self.run_words(ReduceOp::Sum);
        Array::resolve(self.context(), Shape::from(1usize), result, name, location)
    }

    /// The one-element reduction `op` of the words of this evaluated array, as an array of `R`
    /// (`u32` for the count, `bool` otherwise): recorded at once.
    fn run_words<R: GpuType>(&self, op: ReduceOp) -> std::result::Result<Array<R>, Failure> {
        let ctx = self.context();
        self.src()?;
        let storage = Arc::new(Storage::new(ctx, 1, R::LAYOUT));
        let pin = storage.pin(0)?;
        let mut scratch = Scratch::new(ctx);
        reduce_into(self, true, R::LAYOUT.stride() as u64, "u32", ReduceList::new(&[op]), &mut scratch, Some(pin.buffer()), Finish::default())?;
        Ok(Array::from_storage(ctx, Shape::from(1usize), storage))
    }

    /// The smallest element (whether all are true), as a lazy one-element `bool` array.
    #[track_caller]
    pub fn min(&self) -> Array<bool> {
        self.reduce_at([Reduction::Min], "min", Location::caller())
    }

    /// The largest element (whether any is true), as a lazy one-element `bool` array.
    #[track_caller]
    pub fn max(&self) -> Array<bool> {
        self.reduce_at([Reduction::Max], "max", Location::caller())
    }

    /// Whether any element is true, as a lazy one-element `bool` array. Read it with
    /// [`Array::item`].
    #[track_caller]
    pub fn any(&self) -> Array<bool> {
        self.reduce_at([Reduction::Max], "any", Location::caller())
    }

    /// Whether every element is true, as a lazy one-element `bool` array. Read it with
    /// [`Array::item`].
    #[track_caller]
    pub fn all(&self) -> Array<bool> {
        self.reduce_at([Reduction::Min], "all", Location::caller())
    }
}

impl<T: ScalarType> Array<T> {
    pub(crate) fn reduce_at<const N: usize>(&self, ops: [Reduction; N], name: &'static str, location: &'static Location<'static>) -> Array<T> {
        const { assert!(N >= 1 && N <= MAX_REDUCTIONS, "Array::reduce takes one to four reductions") };
        let result = self.run_reduce(ReduceList::new(&ops.map(Reduction::op)), name, location);
        Array::resolve(self.context(), Shape::from(N), result, name, location)
    }

    fn run_reduce(&self, ops: ReduceList, name: &'static str, location: &'static Location<'static>) -> std::result::Result<Array<T>, Failure> {
        let ctx = self.context();
        let shape = Shape::from(ops.as_slice().len());
        let elem = T::LAYOUT.stride() as u64;
        let wgsl = T::reduce_type();
        // Poisoned operands pass their error on.
        let src = self.src()?;
        // An `f32` reduction stays lazy even on a buffer, so that steps on its result fuse into
        // its last pass (see `Reduced`); other types have no such steps and record at once.
        let operand = match (self.pending(), src) {
            (Some(node), _) => Some((Src::Node(Arc::clone(node)), false)),
            (None, src @ Src::Buffer(_)) if wgsl == "f32" => Some((src, true)),
            _ => None,
        };
        if let Some((operand, buffer)) = operand {
            // The reduction reads the operand later, so it is registered with it like any other
            // reader: a write into the operand's buffer (or into what the operand reads)
            // evaluates the readers first, which records the reduction's read ahead of the
            // write. The reader is a copy of the operand that the reduction's first pass fuses.
            let args = [operand, Src::unused(), Src::unused()];
            let reader = jit::create(ctx, self.len(), Op::new(OpKind::Unary(UnOp::Id), ElemTy::of::<T>()?), args, name, location)?;
            let reduced = Reduced {
                operand: reader,
                ops,
                wgsl,
                elem,
                layout: T::LAYOUT,
                buffer,
                finish: Finish::default(),
                result: OnceLock::new(),
                ty: ElemTy::of::<T>()?,
                name,
                location,
                leaves: Default::default(),
                thread: std::thread::current().id(),
                in_recording: ctx.recording_here(),
                forcing: AtomicBool::new(false),
                solo: AtomicBool::new(false),
                pack: OnceLock::new(),
                axis: None,
            };
            let reduced = Arc::new(reduced);
            ctx.jit().register_reduction(&reduced);
            return Ok(Array::from_reduced(ctx, shape, reduced));
        }
        // Nothing is left to fuse: record the passes now.
        let storage = Arc::new(Storage::new(ctx, shape.len(), T::LAYOUT));
        let pin = storage.pin(0)?;
        let mut scratch = Scratch::new(ctx);
        reduce_into(self, T::LAYOUT.is_packed(), elem, wgsl, ops, &mut scratch, Some(pin.buffer()), Finish::default())?;
        Ok(Array::from_storage(ctx, shape, storage))
    }

    /// The reductions `ops` along the lines `geom` of this array as lazy arrays of `shape` (one value per line), `None` where they
    /// cannot be left pending: only `f32` reductions of an array on a GPU backend (of at least `Tuning::axis_defer_cells` cells; on the CPU
    /// backend of an expression of at least `Tuning::axis_defer_weight` operations), outside a recording, with the finite check off.
    /// Recorded when one is used, together with the others that are pending over the same lines (see `jit::multireduce`), so that
    /// each of them is the value a reduction of its own gives (the same bits).
    pub(crate) fn defer_axis(&self, ops: &[ReduceOp], geom: AxisGeom, shape: Shape, name: &'static str, location: &'static Location<'static>) -> Option<Vec<Array<T>>> {
        let ctx = self.context();
        if ElemTy::of::<T>().ok()? != ElemTy::F32
            || T::reduce_type() != "f32"
            || ctx.tuning().reduce_multi < 2
            || ctx.recording_here()
            || ctx.check_finite()
            || ctx.is_profiling()
            || self.error().is_some()
            || (ctx.can_run_on_host() && self.len() <= crate::host::MAX_HOST_ELEMS)
            || self.len() < ctx.tuning().axis_defer_cells
            || (ctx.tuning().axis_defer_weight > 0 && self.pending().is_none_or(|node| node.weight() < f64::from(ctx.tuning().axis_defer_weight)))
        {
            return None;
        }
        // The reductions read the operand later, so they are registered with it like any other reader (see `run_reduce`);
        // the ops of one call read one copy of it.
        let args = [self.src().ok()?, Src::unused(), Src::unused()];
        let reader = jit::create(ctx, self.len(), Op::new(OpKind::Unary(UnOp::Id), ElemTy::F32), args, name, location).ok()?;
        let arrays = ops
            .iter()
            .map(|&op| {
                let reduced = Arc::new(Reduced {
                    operand: Arc::clone(&reader),
                    ops: ReduceList::new(&[op]),
                    wgsl: "f32",
                    elem: 4,
                    layout: T::LAYOUT,
                    buffer: false,
                    finish: Finish::default(),
                    result: OnceLock::new(),
                    ty: ElemTy::F32,
                    name,
                    location,
                    leaves: Default::default(),
                    thread: std::thread::current().id(),
                    in_recording: false,
                    forcing: AtomicBool::new(false),
                    solo: AtomicBool::new(false),
                    pack: OnceLock::new(),
                    axis: Some(geom),
                });
                ctx.jit().register_reduction(&reduced);
                Array::from_reduced(ctx, shape, reduced)
            })
            .collect();
        Some(arrays)
    }

    /// This array, a reduction that has not been recorded yet, extended by `step` (with the
    /// value `word` of its scalar operand, as `f32` bits), as an array of `O`; `None` if the
    /// step cannot run inside the reduction's last pass: the array is not such a reduction of
    /// `f32`, or it has run or failed, or it has [`MAX_EPILOGUE`] steps already.
    pub(crate) fn fuse_step<O: ScalarType>(&self, step: EpilogueStep, word: u32) -> Option<Array<O>> {
        let reduced = self.reduced()?;
        if reduced.axis.is_some() || reduced.wgsl != "f32" || !reduced.is_unforced() || reduced.failure().is_some() {
            return None;
        }
        if ElemTy::of::<T>().ok()? != ElemTy::F32 || ElemTy::of::<O>().ok()? != ElemTy::F32 || step == EpilogueStep::Unary(UnOp::Id) {
            return None;
        }
        let helper = match step {
            EpilogueStep::Unary(op) => op.needs_helper(),
            EpilogueStep::Binary { op, .. } => op.needs_helper() || op == BinOp::Mod,
        };
        if helper {
            return None;
        }
        let mut finish = reduced.finish;
        let at = finish.epilogue.steps().count();
        finish.epilogue = finish.epilogue.then(step)?;
        debug_assert!(at < MAX_EPILOGUE);
        finish.words[at] = word;
        let derived = Reduced {
            operand: Arc::clone(&reduced.operand),
            ops: reduced.ops,
            wgsl: reduced.wgsl,
            elem: reduced.elem,
            layout: reduced.layout,
            buffer: reduced.buffer,
            finish,
            result: OnceLock::new(),
            ty: reduced.ty,
            name: reduced.name,
            location: reduced.location,
            leaves: Default::default(),
            thread: std::thread::current().id(),
            in_recording: self.context().recording_here(),
            forcing: AtomicBool::new(false),
            solo: AtomicBool::new(false),
            pack: OnceLock::new(),
            axis: None,
        };
        let derived = Arc::new(derived);
        self.context().jit().register_reduction(&derived);
        Some(Array::from_reduced(self.context(), self.shape(), derived))
    }
}

impl<T: GpuType> Array<T> {
    /// The value of a one-element array on the host (for example of a reduction). Synchronizes.
    pub fn item(&self) -> Result<T> {
        if self.len() != 1 {
            return Err(LumaError::InvalidShape(format!("item() of an array of shape {}", self.shape())));
        }
        Ok(self.to_vec()?[0])
    }

    /// The `N` values of an `N`-element array on the host, for example of
    /// [`Array::reduce`]. Synchronizes.
    pub fn to_array<const N: usize>(&self) -> Result<[T; N]> {
        if self.len() != N {
            return Err(LumaError::LengthMismatch { expected: N, actual: self.len() });
        }
        let values = self.to_vec()?;
        let mut out = [T::zeroed(); N];
        out.copy_from_slice(&values);
        Ok(out)
    }
}

impl Array<f32> {
    /// The mean of all elements, as a lazy one-element array: the sum divided by the length. The
    /// division happens in the last pass of the reduction.
    #[track_caller]
    pub fn mean(&self) -> Array<f32> {
        &self.sum() / (self.len() as f32)
    }

    /// The Euclidean (L2) norm: the square root of the sum of squares, as a lazy one-element
    /// array. The square root happens in the last pass of the reduction, so a norm is two
    /// launches for a large array and one for a small one. Operations with scalars on any
    /// reduction's result (`(x.sum() / n).sqrt()`, up to four) do the same.
    #[track_caller]
    pub fn norm(&self) -> Array<f32> {
        self.reduce([Reduction::SumOfSquares]).sqrt()
    }
}

/// A reduction for [`Array::reduce`].
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Reduction {
    /// The sum of the elements.
    Sum,
    /// The smallest element.
    Min,
    /// The largest element.
    Max,
    /// The sum of the squared elements.
    SumOfSquares,
    /// The product of the elements (a NaN anywhere gives NaN, integers wrap around).
    Product,
}

impl Reduction {
    pub(crate) fn op(self) -> ReduceOp {
        match self {
            Reduction::Sum => ReduceOp::Sum,
            Reduction::Min => ReduceOp::Min,
            Reduction::Max => ReduceOp::Max,
            Reduction::SumOfSquares => ReduceOp::SumSquares,
            Reduction::Product => ReduceOp::Prod,
        }
    }
}

fn reduce_n<T: GpuType, const N: usize>(array: &Array<T>, wgsl: &'static str, ops: [Reduction; N]) -> Result<[T; N]> {
    const { assert!(N >= 1 && N <= MAX_REDUCTIONS, "Array::reduce takes one to four reductions") };
    let values = reduce_array(array, wgsl, ReduceList::new(&ops.map(Reduction::op)), Finish::default(), true)?;
    let mut out = [T::zeroed(); N];
    for (slot, value) in out.iter_mut().zip(values) {
        *slot = value;
    }
    Ok(out)
}

/// Reduces the whole array on the GPU, once per op of `ops`: the first pass leaves at most
/// `Tuning::reduce_max_groups` partial results per op, the next one workgroup's worth of them,
/// until one element per op is left; those are read back together, with `finish` applied.
/// The first pass reads the data once for all ops; if the array is a pending lazy expression, it
/// computes it on the fly instead of reading a materialized buffer. `wgsl` is the WGSL type of
/// the elements; for vectors (which are never lazy) the operations act on every component
/// separately. A reduction without `finish` of an expression may be answered by the speculation
/// of [`crate::jit`] if `speculate`. Synchronizes.
fn reduce_array<T: GpuType>(array: &Array<T>, wgsl: &'static str, ops: ReduceList, finish: Finish, speculate: bool) -> Result<Vec<T>> {
    let ctx = array.context();
    let elem = T::LAYOUT.stride() as u64;
    let bits = T::LAYOUT.is_packed();
    let count = ops.as_slice().len() as u64;
    // The results of a bit-packed type are the words of a one-element array (see `Reduced`).
    debug_assert!(!bits || count == 1);
    if let Some(values) = speculated(array, wgsl, ops, finish, speculate) {
        return values;
    }
    let mut scratch = Scratch::new(ctx);
    let current = reduce_into(array, bits, elem, wgsl, ops, &mut scratch, None, finish)?;
    let values = T::from_repr(ctx.read::<T::Repr>(&current, elem * count)?, count as usize);
    if values.len() as u64 != count {
        return Err(LumaError::Gpu("reduction results are missing".into()));
    }
    Ok(values)
}

/// The values of the reduction of `array` that the speculation of [`crate::jit`] has ready (see [`reduce_array`]), if it has.
fn speculated<T: GpuType>(array: &Array<T>, wgsl: &'static str, ops: ReduceList, finish: Finish, speculate: bool) -> Option<Result<Vec<T>>> {
    let elem = T::LAYOUT.stride() as u64;
    let bits = T::LAYOUT.is_packed();
    let count = ops.as_slice().len();
    if let Some(node) = array.pending()
        && speculate
        && finish.epilogue.is_empty()
        && matches!(wgsl, "f32" | "u32")
        && let Some(words) = jit::speculate::readback(node, ops, |scratch| reduce_into(array, bits, elem, wgsl, ops, scratch, None, finish))
    {
        return Some(words.map(|words| T::from_repr(bytemuck::cast_slice(&words).to_vec(), count)));
    }
    None
}

/// Runs passes over `current` (`n` elements, or `n` per op once `per_op`) until one element per
/// op is left, and returns the buffer holding them. The last pass applies `finish`. A first pass
/// over more than one workgroup's share is one launch that also finishes the reduction with
/// [`Tuning::single_pass_reduce`] (see [`ReduceList::single_pass`]).
#[allow(clippy::too_many_arguments)]
pub(crate) fn finish_passes(
    ctx: &Context,
    scratch: &mut Scratch<'_>,
    ops: ReduceList,
    wgsl: &'static str,
    elem: u64,
    current: luma_core::Buffer,
    n: u32,
    per_op: bool,
    target: Option<&luma_core::Buffer>,
    finish: Finish,
) -> Result<luma_core::Buffer> {
    finish_passes_gated(ctx, scratch, ops, wgsl, elem, current, n, per_op, target, finish, None)
}

/// [`finish_passes`] where the last pass also runs a gate (its id and the arguments of its
/// buffers, see `ReduceList::with_gate`) on the final value; one op of `f32`s over partial results
/// (`per_op`) that are more than one value, so that the last pass is a pass of one group.
#[allow(clippy::too_many_arguments)]
pub(crate) fn finish_passes_gated(
    ctx: &Context,
    scratch: &mut Scratch<'_>,
    ops: ReduceList,
    wgsl: &'static str,
    elem: u64,
    current: luma_core::Buffer,
    n: u32,
    per_op: bool,
    target: Option<&luma_core::Buffer>,
    finish: Finish,
    gate: Option<(crate::builtin::GateId, &[RawArg<'_>])>,
) -> Result<luma_core::Buffer> {
    let mut scalars = [0u32; MAX_EPILOGUE];
    let mut used = 0;
    for word in finish.uniform_words() {
        scalars[used] = word;
        used += 1;
    }
    passes(ctx, scratch, ops, wgsl, elem, current, n, per_op, target, finish.epilogue, &scalars[..used], gate)
}

/// [`finish_passes`] for the partial results of a first pass of a [merged](ReduceList::merged)
/// list (`n` per op, `ops` the list of the first pass): the last pass applies the epilogues of the
/// list, the scalars of which are `scalars`, in the order of its uniform fields.
#[allow(clippy::too_many_arguments)]
pub(crate) fn finish_passes_merged(
    ctx: &Context,
    scratch: &mut Scratch<'_>,
    ops: ReduceList,
    wgsl: &'static str,
    elem: u64,
    current: luma_core::Buffer,
    n: u32,
    target: &luma_core::Buffer,
    scalars: &[u32],
) -> Result<luma_core::Buffer> {
    passes(ctx, scratch, ops, wgsl, elem, current, n, true, Some(target), Epilogue::default(), scalars, None)
}

/// The passes of [`finish_passes`]: the last one applies `epilogue` (the epilogues of a merged list)
/// with the scalar operands `scalars`, and runs `gate` if there is one (see
/// [`finish_passes_gated`]).
#[allow(clippy::too_many_arguments)]
fn passes(
    ctx: &Context,
    scratch: &mut Scratch<'_>,
    ops: ReduceList,
    wgsl: &'static str,
    elem: u64,
    mut current: luma_core::Buffer,
    mut n: u32,
    mut per_op: bool,
    target: Option<&luma_core::Buffer>,
    epilogue: Epilogue,
    scalars: &[u32],
    gate: Option<(crate::builtin::GateId, &[RawArg<'_>])>,
) -> Result<luma_core::Buffer> {
    debug_assert!(gate.is_none() || (per_op && n > 1 && ops.as_slice().len() == 1), "a gate runs in a last pass over partial results");
    let count = ops.as_slice().len() as u64;
    while !per_op || n > 1 {
        let groups = ctx.tuning().reduce_groups(n);
        let list = if per_op { ops.rest() } else { ops };
        // The pass with one group is the last: its output is the final value; so is that of a
        // first pass that finishes the reduction itself.
        let single_pass = !per_op && groups > 1 && ctx.tuning().single_pass_reduce;
        let last = groups == 1 || single_pass;
        let list = if last { list.finished(epilogue) } else { list };
        let list = if single_pass { list.single_pass() } else { list };
        let list = match gate {
            Some((id, _)) if last => list.with_gate(id),
            _ => list,
        };
        let in_bytes = u64::from(n) * elem * if per_op { count } else { 1 };
        // The last pass leaves one element per op: into `target` if there is one.
        let out = match target {
            Some(target) if groups == 1 => target.clone(),
            _ => scratch.take(u64::from(groups) * elem * count)?,
        };
        let core = ctx.builtin(Builtin::Reduce(list, per_op), wgsl)?;
        // The uniform: `n`; the double-float pass has a second field, the guard of its arithmetic
        // (always 0); the last pass the scalars of the epilogue.
        let mut words = [0u32; 2 + MAX_ACCS * MAX_EPILOGUE];
        words[0] = n;
        let mut used = if wgsl == "df32" { 2 } else { 1 };
        if last {
            words[used..used + scalars.len()].copy_from_slice(scalars);
            used += scalars.len();
        }
        ctx.jit().note_reduce_pass();
        if single_pass {
            // The partial results go to `out`, the final values to `target` or a buffer of their own.
            let values = match target {
                Some(target) => target.clone(),
                None => scratch.take(elem * count)?,
            };
            ctx.launch(
                &core,
                fold_workgroups(ctx, groups)?,
                [n, 1, 1],
                &[
                    RawArg::write(&out, u64::from(groups) * elem * count),
                    RawArg::read(&current, in_bytes),
                    RawArg::write(ctx.reduce_counter()?, 4),
                    RawArg::write(&values, elem * count),
                    RawArg::Uniform(bytemuck::cast_slice(&words[..used])),
                ],
            )?;
            return Ok(values);
        }
        let mut raw = vec![RawArg::write(&out, u64::from(groups) * elem * count), RawArg::read(&current, in_bytes)];
        if let Some((_, args)) = gate.filter(|_| last) {
            raw.extend_from_slice(args);
        }
        raw.push(RawArg::Uniform(bytemuck::cast_slice(&words[..used])));
        ctx.launch(&core, fold_workgroups(ctx, groups)?, [n, 1, 1], &raw)?;
        drop(raw);
        current = out;
        n = groups;
        per_op = true;
    }
    Ok(current)
}

/// The reduction of `array` as a buffer holding one element per op; see [`reduce_array`]. `elem`
/// is the size of an element in bytes. The values end up in `target` when it is given (a buffer
/// of at least `elem` bytes per op), without any copy. `finish` is applied by the last pass.
/// With `bits` the array holds bit-packed `bool`s: an evaluated one is reduced a word at a time
/// (see [`Builtin::ReduceBits`]) by the one op of `ops`.
#[allow(clippy::too_many_arguments)]
fn reduce_into<T: GpuType>(
    array: &Array<T>,
    bits: bool,
    elem: u64,
    wgsl: &'static str,
    ops: ReduceList,
    scratch: &mut Scratch<'_>,
    target: Option<&luma_core::Buffer>,
    finish: Finish,
) -> Result<luma_core::Buffer> {
    let ctx = array.context();
    let count = ops.as_slice().len() as u64;
    let run_passes = |scratch: &mut Scratch<'_>, current: luma_core::Buffer, n: u32, per_op: bool, target: Option<&luma_core::Buffer>, finish: Finish| {
        finish_passes(ctx, scratch, ops, wgsl, elem, current, n, per_op, target, finish)
    };

    let fused = match array.pending() {
        Some(node) => jit::reduce_first_pass(node, ops, scratch, target, finish).map_err(Failure::into_error)?,
        None => None,
    };
    let mut results = Vec::new();
    match fused {
        Some(partials) => {
            // With several segments only the final collecting pass writes `target`.
            let target = target.filter(|_| partials.len() == 1);
            let single = partials.len() == 1;
            for (partial, groups) in partials {
                // A first pass with one segment and one group applied `finish` itself.
                let rest = if single && groups != 1 { finish } else { Finish::default() };
                results.push(run_passes(scratch, partial, groups, true, target, rest)?);
            }
        }
        None => {
            let storage = array.storage().map_err(Failure::into_error)?;
            let single = storage.segments() == 1;
            let target = target.filter(|_| single);
            for j in 0..storage.segments() {
                let n = len_u32(storage.range(j).len())?;
                // Wgpu buffers are reference counted; the pin keeps it resident while it is bound.
                let pin = storage.pin_read(j)?;
                let rest = if single { finish } else { Finish::default() };
                if bits {
                    let (words, groups) = bits_first_pass(ctx, scratch, ops, pin.buffer(), n, target)?;
                    results.push(run_passes(scratch, words, groups, true, target, rest)?);
                } else {
                    results.push(run_passes(scratch, pin.buffer().clone(), n, false, target, rest)?);
                }
            }
        }
    }

    if results.len() == 1 {
        return Ok(results.swap_remove(0));
    }
    let segments = results.len() as u64;
    let collect = scratch.take(segments * count * elem)?;
    for i in 0..count {
        for (j, partial) in results.iter().enumerate() {
            ctx.copy(partial, i * elem, &collect, (i * segments + j as u64) * elem, elem)?;
        }
    }
    run_passes(scratch, collect, len_u32(results.len())?, true, target, finish)
}

/// The first pass of the reduction `ops` (one op) of the `n` bit-packed elements in `words`: the
/// buffer holding the partial results and how many there are. With one group they are the final
/// value, which is written into `target` if it is given.
fn bits_first_pass(
    ctx: &Context,
    scratch: &mut Scratch<'_>,
    ops: ReduceList,
    words: &luma_core::Buffer,
    n: u32,
    target: Option<&luma_core::Buffer>,
) -> Result<(luma_core::Buffer, u32)> {
    let count = n.div_ceil(32);
    let groups = ctx.tuning().reduce_groups(count);
    let out = match target {
        Some(target) if groups == 1 => target.clone(),
        _ => scratch.take(u64::from(groups) * 4)?,
    };
    let core = ctx.builtin(Builtin::ReduceBits(ops), "u32")?;
    ctx.jit().note_reduce_pass();
    ctx.launch(
        &core,
        fold_workgroups(ctx, groups)?,
        [count, 1, 1],
        &[
            RawArg::write(&out, u64::from(groups) * 4),
            RawArg::read(words, u64::from(count) * 4),
            RawArg::Uniform(bytemuck::cast_slice(&[count, n])),
        ],
    )?;
    Ok((out, groups))
}

/// Per-component reductions of `f32` vector arrays: the same tree reduction as the scalar
/// reductions, run on all components at once, so the accuracy is that of the scalar sum.
macro_rules! vector_reductions {
    ($n:literal, $wgsl:literal) => {
        impl Array<[f32; $n]> {
            /// The sum of all elements, per component, with the accuracy of [`Array::sum`].
            /// Synchronizes.
            pub fn sum(&self) -> Result<[f32; $n]> {
                Ok(self.reduce([Reduction::Sum])?[0])
            }

            /// The smallest value of every component. Synchronizes.
            pub fn min(&self) -> Result<[f32; $n]> {
                Ok(self.reduce([Reduction::Min])?[0])
            }

            /// The largest value of every component. Synchronizes.
            pub fn max(&self) -> Result<[f32; $n]> {
                Ok(self.reduce([Reduction::Max])?[0])
            }

            /// The Euclidean (L2) norm of every component taken over the whole array: the square
            /// root of the sum of squares, per component. Synchronizes.
            pub fn norm(&self) -> Result<[f32; $n]> {
                Ok(self.reduce([Reduction::SumOfSquares])?[0].map(f32::sqrt))
            }

            /// Several per-component reductions in one pass over the array (up to four), like
            /// [`Array::reduce`]. Synchronizes.
            pub fn reduce<const N: usize>(&self, ops: [Reduction; N]) -> Result<[[f32; $n]; N]> {
                reduce_n(self, $wgsl, ops)
            }
        }
    };
}

vector_reductions!(2, "vec2<f32>");
vector_reductions!(4, "vec4<f32>");

#[cfg(test)]
mod tests {
    //! Reductions that finish in their first launch (see `ReduceList::single_pass`) against the
    //! two-launch ones, on the CPU backend, whose atomics order memory across workgroups (the
    //! flag is on by default only for CUDA).

    use luma_core::{BackendKind, DeviceKind};

    use super::*;
    use crate::context::ContextOptions;
    use crate::gpu_type::Df32;

    /// A CPU context whose reductions finish in their first launch or not.
    fn cpu(single_pass: bool) -> Context {
        let options = ContextOptions { backend: BackendKind::Cpu, device: DeviceKind::Cpu, ..ContextOptions::from_env() }.calibrate(false);
        Context::with_tuning(options, |tuning| tuning.single_pass_reduce = single_pass).unwrap()
    }

    /// One workgroup, a few, the cap of workgroups with and without a strided second round.
    const LENGTHS: [usize; 6] = [1, 2047, 2048, 2049, 1 << 20, (1 << 21) + 17];

    fn words(n: usize, seed: u64) -> Vec<u32> {
        let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (0..n)
            .map(|_| {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                (state >> 32) as u32
            })
            .collect()
    }

    fn floats(n: usize, seed: u64) -> Vec<f32> {
        words(n, seed).into_iter().map(|w| (w >> 8) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0).collect()
    }

    /// Reductions of `data` of every kind, the whole set twice in a row (the counter must be zero
    /// again after a launch), as bits.
    fn float_reductions(ctx: &Context, data: &[f32]) -> Vec<u32> {
        let x = Array::from_slice_on(ctx, data, data.len()).unwrap();
        let lazy = || &x * 2.0f32 + 1.0f32;
        let mut out = Vec::new();
        for _ in 0..2 {
            let mut keep = |value: f32| out.push(value.to_bits());
            keep(x.sum().item().unwrap());
            keep(x.min().item().unwrap());
            keep(x.max().item().unwrap());
            keep(x.norm().item().unwrap());
            keep(x.mean().item().unwrap());
            let all = x.reduce([Reduction::Sum, Reduction::Min, Reduction::Max, Reduction::SumOfSquares]).to_array::<4>().unwrap();
            all.into_iter().for_each(&mut keep);
            keep(lazy().sum().item().unwrap());
            keep(lazy().max().item().unwrap());
            keep(lazy().norm().item().unwrap());
            keep(lazy().mean().item().unwrap());
            // Launches recorded one after the other before anything is read.
            let (sum, max) = (x.sum(), lazy().max());
            keep((&sum + &max).item().unwrap());
        }
        out
    }

    #[test]
    fn float_reductions_match_the_two_launch_ones_bit_for_bit() {
        let (two, one) = (cpu(false), cpu(true));
        for (k, &n) in LENGTHS.iter().enumerate() {
            let data = floats(n, k as u64 + 1);
            let (want, got) = (float_reductions(&two, &data), float_reductions(&one, &data));
            assert_eq!(got, want, "n = {n}");
            // Twice in a row: the second round saw a counter that was reset.
            let (first, second) = got.split_at(got.len() / 2);
            assert_eq!(first, second, "n = {n}, second round");

            let as_f64: Vec<f64> = data.iter().map(|&v| f64::from(v)).collect();
            let (sum, abs): (f64, f64) = (as_f64.iter().sum(), as_f64.iter().map(|v| v.abs()).sum());
            let squares: f64 = as_f64.iter().map(|v| v * v).sum();
            let value = |at: usize| f64::from(f32::from_bits(got[at]));
            let tolerance = 1e-4 * abs + 1e-6;
            assert!((value(0) - sum).abs() <= tolerance, "n = {n}: sum {} vs {sum}", value(0));
            assert_eq!(value(1), as_f64.iter().copied().fold(f64::INFINITY, f64::min), "n = {n}: min");
            assert_eq!(value(2), as_f64.iter().copied().fold(f64::NEG_INFINITY, f64::max), "n = {n}: max");
            assert!((value(3) - squares.sqrt()).abs() <= 1e-4 * squares.sqrt() + 1e-6, "n = {n}: norm {} vs {}", value(3), squares.sqrt());
            assert!((value(4) - sum / n as f64).abs() <= tolerance / n as f64 + 1e-6, "n = {n}: mean");
            assert!((value(9) - (2.0 * sum + n as f64)).abs() <= 2.0 * tolerance + 1e-3, "n = {n}: lazy sum {} vs {}", value(9), 2.0 * sum + n as f64);
        }
    }

    #[test]
    fn integer_reductions_match_the_two_launch_ones() {
        let (two, one) = (cpu(false), cpu(true));
        for (k, &n) in LENGTHS.iter().enumerate() {
            let data = words(n, k as u64 + 11);
            let signed: Vec<i32> = data.iter().map(|&w| w as i32).collect();
            for ctx in [&two, &one] {
                for _ in 0..2 {
                    let x = Array::from_slice_on(ctx, &data, n).unwrap();
                    assert_eq!(x.sum().item().unwrap(), data.iter().fold(0u32, |acc, &w| acc.wrapping_add(w)), "u32 sum, n = {n}");
                    assert_eq!(x.min().item().unwrap(), data.iter().copied().min().unwrap(), "u32 min, n = {n}");
                    assert_eq!(x.max().item().unwrap(), data.iter().copied().max().unwrap(), "u32 max, n = {n}");
                    let lazy = &x + 1u32;
                    assert_eq!(lazy.max().item().unwrap(), data.iter().map(|w| w.wrapping_add(1)).max().unwrap(), "u32 lazy max, n = {n}");
                    let y = Array::from_slice_on(ctx, &signed, n).unwrap();
                    assert_eq!(y.sum().item().unwrap(), signed.iter().fold(0i32, |acc, &w| acc.wrapping_add(w)), "i32 sum, n = {n}");
                    assert_eq!(y.min().item().unwrap(), signed.iter().copied().min().unwrap(), "i32 min, n = {n}");
                    assert_eq!(y.max().item().unwrap(), signed.iter().copied().max().unwrap(), "i32 max, n = {n}");
                    let [lo, hi] = y.reduce([Reduction::Min, Reduction::Max]).to_array::<2>().unwrap();
                    assert_eq!((lo, hi), (signed.iter().copied().min().unwrap(), signed.iter().copied().max().unwrap()), "i32 min and max, n = {n}");
                }
            }
        }
    }

    #[test]
    fn double_float_reductions_match_the_two_launch_ones() {
        let (two, one) = (cpu(false), cpu(true));
        for (k, &n) in LENGTHS.iter().take(5).enumerate() {
            let data: Vec<Df32> = floats(n, k as u64 + 21).into_iter().map(|v| Df32 { hi: v, lo: v * 1e-8 }).collect();
            let run = |ctx: &Context| {
                let x = Array::from_slice_on(ctx, &data, n).unwrap();
                let mut out = Vec::new();
                for _ in 0..2 {
                    out.push(x.sum().item().unwrap());
                    out.push(x.min().item().unwrap());
                    out.push(x.max().item().unwrap());
                    out.push(x.minimum(Df32::from(0.5)).sum().item().unwrap());
                }
                out.into_iter().flat_map(|d| [d.hi.to_bits(), d.lo.to_bits()]).collect::<Vec<u32>>()
            };
            let (want, got) = (run(&two), run(&one));
            assert_eq!(got, want, "n = {n}");
            let lowest = data.iter().map(|d| d.hi).fold(f32::INFINITY, f32::min);
            assert_eq!(f32::from_bits(got[2]), lowest, "n = {n}: min");
        }
    }

    #[test]
    fn vector_reductions_match_the_two_launch_ones() {
        let (two, one) = (cpu(false), cpu(true));
        let n = 1 << 20;
        let data: Vec<[f32; 2]> = floats(2 * n, 31).chunks(2).map(|c| [c[0], c[1]]).collect();
        let run = |ctx: &Context| {
            let x = Array::from_slice_on(ctx, &data, n).unwrap();
            let [sum, lo, hi] = x.reduce([Reduction::Sum, Reduction::Min, Reduction::Max]).unwrap();
            [sum, lo, hi].into_iter().flatten().map(f32::to_bits).collect::<Vec<u32>>()
        };
        assert_eq!(run(&one), run(&two));
    }

    #[test]
    fn a_reduction_over_many_workgroups_is_one_launch_with_the_flag() {
        // (flag, launches of a reduction over several workgroups); one workgroup is one launch.
        for (single_pass, many) in [(false, 2), (true, 1)] {
            let ctx = cpu(single_pass);
            let launches = |run: &dyn Fn()| {
                let before = ctx.jit().stats();
                run();
                let after = ctx.jit().stats();
                (after.reduce_launches - before.reduce_launches) + (after.reduce_passes - before.reduce_passes)
            };
            let big = Array::<u32>::from_slice_on(&ctx, &vec![1u32; 1 << 20], 1 << 20).unwrap();
            let small = Array::<u32>::from_slice_on(&ctx, &vec![1u32; 1000], 1000).unwrap();
            assert_eq!(launches(&|| assert_eq!(big.sum().item().unwrap(), 1 << 20)), many, "evaluated buffer, single_pass = {single_pass}");
            assert_eq!(launches(&|| assert_eq!(small.sum().item().unwrap(), 1000)), 1, "one workgroup, single_pass = {single_pass}");
            let floats = Array::<f32>::from_slice_on(&ctx, &vec![0.5f32; 1 << 20], 1 << 20).unwrap();
            assert_eq!(launches(&|| assert_eq!((&floats * 2.0f32).sum().item().unwrap(), (1 << 20) as f32)), many, "lazy expression, single_pass = {single_pass}");
        }
    }

    /// The gate loop of `iterate_until` folds the residual in the check kernel (one launch per
    /// step); with the dispatch arguments zeroed after convergence the steps left in a block run
    /// no workgroup, so no ticket is taken and the counter stays zero for the next run.
    #[test]
    fn iterate_until_stops_at_the_same_step_with_single_pass_residuals() {
        let (two, one) = (cpu(false), cpu(true));
        for n in [4096usize, 1 << 16] {
            let data = floats(n, 41);
            for (max_steps, tol) in [(300usize, 1e-3f32), (300, 1e-6), (300, 1e-16), (20, 0.0)] {
                // Halving is exact: the state after `k` steps is `data * 0.5^k`, and the change of
                // step `k` is its largest element's half.
                let mut host = data.clone();
                for _ in 0..max_steps {
                    let change = host.iter().map(|&v| (v - v * 0.5).abs()).fold(0.0f32, f32::max);
                    host.iter_mut().for_each(|v| *v *= 0.5);
                    if change <= tol {
                        break;
                    }
                }
                for ctx in [&two, &one] {
                    // Twice: the second run starts from the counter the first left.
                    for _ in 0..2 {
                        let x = Array::from_slice_on(ctx, &data, n).unwrap();
                        let got = x.iterate_until(max_steps, tol, |u| u * 0.5f32).to_vec().unwrap();
                        assert_eq!(got, host, "n = {n}, tol = {tol}, single_pass = {}", ctx.tuning().single_pass_reduce);
                    }
                }
            }
        }
    }
}
