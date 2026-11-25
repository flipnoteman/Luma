//! Reductions: whole-array sums, extrema and norms, fused with the expression behind them. They
//! are lazy: the result is a one-element array, and the host reads it with [`Array::item`].

use std::panic::Location;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use crate::array::{Array, Failure, Storage};
use crate::builtin::{BinOp, Builtin, Epilogue, EpilogueStep, Finish, ReduceList, ReduceOp, UnOp, MAX_ACCS, MAX_EPILOGUE, MAX_REDUCTIONS};
use crate::context::{Context, RawArg, Scratch};
use crate::error::{LumaError, Result};
use crate::gpu_type::{ElemLayout, GpuType, Numeric, ScalarType};
use crate::jit::structured::emit::Emit;
use crate::jit::structured::select;
use crate::jit::structured::strategy::{Output, PassMode, Subject};
use crate::gpu::lock;
use crate::jit::comprehension::{Algebra, Comprehension, OpTag};
use crate::jit::rules;
use crate::jit::structured::{Comp, Family, Launch, Meta, RunState, Structured};
use crate::jit::{self, ElemTy, Node, Op, OpKind, Src};
use crate::ops::{fold_workgroups, len_u32, run_select, Condition, Rhs};
use crate::shape::Shape;

// A reduction of a pending expression that has not been recorded yet is a [`Structured`] value of the reduce family: the
// operand is kept as an expression until the result is used on the GPU (then the passes are recorded into a buffer,
// without reading anything back) or on the host (then it is read like a synchronous reduction,
// which lets the speculation of [`crate::jit`] answer loops of readbacks).
//
// Elementwise operations with scalars on the result (`(x.sum() / n).sqrt()`) do not create
// expressions: they extend the reduction by an epilogue that its last pass applies, so the
// whole is as many launches as the reduction alone (see [`Array::fuse_step`]). Such a reduction
// shares the operand with the one it extends but never reads its buffer (which a write through
// another array may change): if both are used they run in one kernel with the others that are
// pending (see `jit::multireduce`), or separately where that is not possible.
//
// The methods below are the reduce family's: the predicates and recording of a [`Structured`] whose payload is a [`Comp`].

/// Most kernels of reductions whose compile [`Structured::build_ahead`] starts at once.
const AHEAD_KERNELS: usize = 8;

mod strategies;
pub(crate) use strategies::REDUCE_STRATEGIES;

/// The lines an axis reduction runs over: `lines` lines of `n` cells, `stride` apart.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct AxisGeom {
    pub(crate) n: u32,
    pub(crate) stride: u32,
    pub(crate) lines: u32,
}

/// The buffer the reductions recorded in one kernel share (see `Structured::record_merged`), and its words once a reader has
/// copied them to the host: the first of those reductions that is read on the host reads them all, so that reading a dozen
/// of them costs one round trip.
pub(crate) struct PackRead {
    pack: Arc<Storage>,
    words: OnceLock<Vec<u32>>,
}

impl Structured {
    /// The node whose values the reduction reduces.
    pub(crate) fn operand(&self) -> &Arc<Node> {
        let RunState::Reduce { operand, .. } = &self.comp().run;
        operand
    }

    /// The ops of the reduction (a `dot` is a sum).
    pub(crate) fn ops(&self) -> ReduceList {
        match &self.comp().meta.tag {
            OpTag::Reduce(ops) | OpTag::ReduceAxis(ops) => *ops,
            OpTag::Dot => ReduceList::new(&[ReduceOp::Sum]),
            tag => unreachable!("a structured value is a reduction, not {tag:?}"),
        }
    }

    /// The form of the reduction (`None` under `LUMA_ALGEBRA=off`), built by the builder of its tag from what the value holds the
    /// first time it is asked for and kept: the registered reader node as the one access, the steps of `finish` as the epilogue.
    /// A copy: the lock is not held while a rule or the recogniser works on it.
    pub(crate) fn form(&self, ctx: &Context) -> Option<Comprehension> {
        if !Algebra::builds_forms() {
            return None;
        }
        let comp = self.comp();
        let mut slot = lock(&comp.form);
        // The rules of tier 1 for comprehensions run once, where a context first meets the form (the builders are pure and have none).
        let form = slot.get_or_insert_with(|| Box::new(rules::t1(self.build_form(), ctx)));
        Some(Comprehension::clone(form))
    }

    fn build_form(&self) -> Comprehension {
        let comp = self.comp();
        let RunState::Reduce { operand, finish, .. } = &comp.run;
        let (src, len, ty) = (Src::Node(Arc::clone(operand)), comp.meta.len, comp.meta.ty);
        let form = match (&comp.meta.tag, comp.meta.geom) {
            (OpTag::Reduce(ops), None) => Comprehension::reduce(src, len, *ops, ty),
            (OpTag::Dot, None) => Comprehension::dot(src, len, ty),
            (OpTag::ReduceAxis(ops), Some(geom)) => Comprehension::reduce_axis(src, geom, ops.as_slice()[0], ty),
            (tag, geom) => unreachable!("a structured value is a reduction, not {tag:?} over {geom:?}"),
        };
        let form = form.with_steps(*finish);
        // The guarantee of 07 §4.1: an API form is well formed and of the class its tag says.
        debug_assert!(form.validate().is_ok(), "the form of {:?}: {:?}", comp.meta.tag, form.validate());
        form
    }

    /// The lines of an axis reduction; `None` for a reduction of the whole array.
    pub(crate) fn geom(&self) -> Option<AxisGeom> {
        self.comp().meta.geom
    }

    /// The WGSL type of the elements the passes work on.
    pub(crate) fn wgsl(&self) -> &'static str {
        let RunState::Reduce { wgsl, .. } = &self.comp().run;
        wgsl
    }

    pub(crate) fn finish(&self) -> Finish {
        let RunState::Reduce { finish, .. } = &self.comp().run;
        *finish
    }

    pub(crate) fn layout(&self) -> ElemLayout {
        let RunState::Reduce { layout, .. } = &self.comp().run;
        *layout
    }

    /// What a kernel that reduces `operand` itself needs to run this reduction (see
    /// `jit::until::force_with_record`): the operand, the ops, what is applied to their values.
    /// `None` unless it is a reduction of `f32`s that has not been recorded, and has not failed.
    pub(crate) fn fusable(&self) -> Option<(&Arc<Node>, ReduceList, Finish)> {
        let comp = self.comp();
        let RunState::Reduce { operand, wgsl, finish, .. } = &comp.run;
        (comp.meta.tag.is_whole_reduce() && *wgsl == "f32" && comp.meta.ty == ElemTy::F32 && self.is_unforced() && self.failure().is_none()).then(|| (operand, self.ops(), *finish))
    }

    /// Whether this reduction may be recorded together with others: a plain `f32`, `u32` or `i32` reduction of an
    /// expression that is still pending, outside a recording.
    pub(crate) fn mergeable(&self, ctx: &Context) -> bool {
        ctx.tuning().reduce_multi >= 2 && !self.in_recording && !ctx.recording_here() && !ctx.check_finite() && self.eligible()
    }

    /// What [`Structured::mergeable`] says without the context.
    fn eligible(&self) -> bool {
        let comp = self.comp();
        let RunState::Reduce { operand, wgsl, layout, solo, .. } = &comp.run;
        comp.meta.tag.is_whole_reduce()
            && matches!(comp.meta.ty, ElemTy::F32 | ElemTy::U32 | ElemTy::I32)
            && *wgsl == comp.meta.ty.wgsl()
            && !layout.is_packed()
            && self.is_unforced()
            && !solo.load(Ordering::Relaxed)
            && operand.is_pending()
            && operand.has_operands()
            && self.failure().is_none()
    }

    /// Whether this reduction can be recorded together with `seed` (itself mergeable).
    pub(crate) fn partners_with(&self, seed: &Structured) -> bool {
        self.thread == seed.thread
            && !self.in_recording
            && !self.forcing.load(Ordering::Relaxed)
            && self.comp().meta.ty == seed.comp().meta.ty
            && self.comp().meta.len == seed.comp().meta.len
            && self.eligible()
    }

    /// Whether this is an axis reduction (see [`Meta::geom`]), which is recorded with its own kind only.
    pub(crate) fn is_axis(&self) -> bool {
        matches!(self.comp().meta.tag, OpTag::ReduceAxis(_))
    }

    /// Whether this axis reduction may be recorded together with others of the same lines: a plain `f32` reduction of an
    /// expression that is still pending, outside a recording.
    pub(crate) fn axis_mergeable(&self, ctx: &Context) -> bool {
        ctx.tuning().reduce_multi >= 2 && !self.in_recording && !ctx.recording_here() && !ctx.check_finite() && self.axis_eligible()
    }

    /// What [`Structured::axis_mergeable`] says without the context.
    fn axis_eligible(&self) -> bool {
        let comp = self.comp();
        let RunState::Reduce { operand, .. } = &comp.run;
        matches!(comp.meta.tag, OpTag::ReduceAxis(_))
            && comp.meta.ty == ElemTy::F32
            && self.wgsl() == "f32"
            && self.is_unforced()
            && operand.is_pending()
            && operand.has_operands()
            && self.failure().is_none()
    }

    /// Whether this axis reduction can be recorded together with the axis reduction `seed` (itself mergeable): the same lines.
    pub(crate) fn axis_partners_with(&self, seed: &Structured) -> bool {
        self.thread == seed.thread && !self.in_recording && !self.forcing.load(Ordering::Relaxed) && self.comp().meta.geom == seed.comp().meta.geom && self.axis_eligible()
    }

    /// Whether this reduction can be recorded together with `seed`, which is mergeable (of whichever kind it is).
    pub(crate) fn partner_of(&self, seed: &Structured) -> bool {
        if seed.is_axis() { self.axis_partners_with(seed) } else { self.partners_with(seed) }
    }

    /// The pending reductions to record together with this one, first running the reductions its operand
    /// waits for. Empty if it cannot be merged.
    pub(crate) fn merge_partners(&self, ctx: &Context) -> Vec<Arc<Structured>> {
        let mergeable = if self.is_axis() { self.axis_mergeable(ctx) } else { self.mergeable(ctx) };
        if !mergeable || !ctx.jit().may_have_partners(self) {
            return Vec::new();
        }
        self.take_partners(ctx)
    }

    /// [`Structured::merge_partners`] once its checks hold (the merged strategies' `fits` made them): forces the reductions the
    /// operand waits for, which is a side effect, hence it runs in `lower` and never in `fits`.
    pub(crate) fn take_partners(&self, ctx: &Context) -> Vec<Arc<Structured>> {
        let operand = self.operand();
        let Some(leaves) = jit::unresolved_leaves(operand) else { return Vec::new() };
        if !leaves.is_empty() {
            for leaf in leaves {
                let _ = leaf.force();
            }
            // Running them may have left others (a reduction that depends on a reduction).
            if !jit::unresolved_leaves(operand).is_some_and(|leaves| leaves.is_empty()) {
                return Vec::new();
            }
        }
        ctx.jit().reduction_partners(ctx, self, &[])
    }

    /// Records the reductions `partners` (see [`Structured::merge_partners`]) and this one in one kernel: this reduction's
    /// buffer, once the others hold theirs. `None` (nothing recorded) if the program does not fit one kernel, or the kernel is
    /// not due or not ready yet (then every member is marked to run alone).
    pub(crate) fn record_merged(&self, emit: &mut Emit<'_>, partners: &[Arc<Structured>]) -> Option<Arc<Storage>> {
        let ctx = emit.ctx();
        let all = || std::iter::once(self).chain(partners.iter().map(|partner| &**partner));
        let members: Vec<jit::Member<'_>> = all().map(|reduced| jit::Member { operand: reduced.operand(), ops: reduced.ops(), finish: reduced.finish() }).collect();
        let ahead = || self.build_ahead(ctx, partners);
        let deferred = std::cell::Cell::new(false);
        let pack = match jit::reduce_many(emit, &members, self.ty(), ctx.tuning().reduce_multi, PassMode::Auto, jit::ManyMode::Run { ahead: &ahead, deferred: &deferred }) {
            Ok(Some(pack)) => Arc::new(pack),
            _ => {
                if deferred.get() {
                    for reduced in all() {
                        let RunState::Reduce { solo, .. } = &reduced.comp().run;
                        solo.store(true, Ordering::Relaxed);
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
            let count = reduced.ops().as_slice().len();
            let window = Arc::new(Storage::window_of(&pack, offset, count));
            let RunState::Reduce { pack: slot, .. } = &reduced.comp().run;
            let _ = slot.set((Arc::clone(&shared), offset));
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
    pub(crate) fn follows(&self, ctx: &Context, first: &Structured) -> bool {
        self.thread == first.thread && self.comp().meta.tag.is_whole_reduce() && !self.in_recording && !self.forcing.load(Ordering::Relaxed) && self.mergeable(ctx)
    }

    /// Starts the compiles of the kernels the reductions pending after `first_group` (this one and its `partners`) will be
    /// recorded in, if they are used in the order they were made, so that the compiles run side by side: a kernel that is not
    /// compiled yet is waited for where it is launched, and the groups are recorded one after the other (a force monitor of
    /// several dozen sums is a few kernels, each compiled from scratch the first time). Runs only when the kernel of the
    /// first group was not built before, so it costs nothing where the kernels are known.
    fn build_ahead(&self, ctx: &Context, partners: &[Arc<Structured>]) {
        let mut taken: Vec<*const Structured> = std::iter::once(self as *const Structured).chain(partners.iter().map(Arc::as_ptr)).collect();
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
                std::iter::once(&*seed).chain(others.iter().map(|other| &**other)).map(|reduced| jit::Member { operand: reduced.operand(), ops: reduced.ops(), finish: reduced.finish() }).collect();
            // A program that does not fit is left to the reductions' own turn.
            let _ = jit::reduce_many(&mut Emit::now(ctx), &members, seed.ty(), ctx.tuning().reduce_multi, PassMode::Auto, jit::ManyMode::BuildOnly);
        }
    }

    /// The values of a reduction recorded together with others, read from the host copy of the buffer they share
    /// (made by the first read of any of them); `None` if it was not, or its own buffer was made since (a window
    /// that is pinned gets a copy of its range, see [`Storage::window_of`]).
    pub(crate) fn packed_values<T: GpuType>(&self, ctx: &Context) -> Option<Result<Vec<T>>> {
        if std::mem::size_of::<T::Repr>() != 4 {
            return None;
        }
        let RunState::Reduce { pack, .. } = &self.comp().run;
        let (shared, offset) = pack.get()?;
        if !self.result_if_recorded()?.as_ref().is_ok_and(|storage| storage.is_window()) {
            return None;
        }
        let words = match shared.words.get() {
            Some(words) => words,
            None => match ctx.read_storage::<u32>(&shared.pack) {
                Ok(words) => shared.words.get_or_init(|| words),
                Err(error) => return Some(Err(error)),
            },
        };
        let count = self.ops().as_slice().len();
        Some(Ok(T::from_repr(bytemuck::pod_collect_to_vec(words.get(*offset..offset + count)?), count)))
    }

    /// Records the passes into the buffer of a reduction that was not recorded yet: the strategy `select` picks runs (see
    /// [`strategies`]): on the host if it can run there, else on the GPU together with the reductions that are pending.
    pub(crate) fn record_reduce(&self, ctx: &Context) -> std::result::Result<Arc<Storage>, Arc<LumaError>> {
        self.forcing.store(true, Ordering::Relaxed);
        // The rewrite engine plans the value first (one load and a branch while no rule of tier 2 is allowed).
        let planned = if ctx.jit().rules.t2_active { rules::plan(ctx, self) } else { None };
        let result = match planned {
            Some(planned) => rules::run_planned(ctx, self, &planned),
            None => select::execute(ctx, &Subject::Comp(self)).map(Output::into_storage).map_err(|failure| Arc::new(failure.into_error())),
        };
        self.forcing.store(false, Ordering::Relaxed);
        result
    }

    /// Records this reduction with the `partners` a read found (non-axis; see [`Structured::read`]): in one kernel with them where
    /// that works, else alone.
    fn record_listed(&self, ctx: &Context, partners: &[Arc<Structured>]) -> Result<Arc<Storage>> {
        let mut emit = Emit::now(ctx);
        if !partners.is_empty()
            && let Some(storage) = self.record_merged(&mut emit, partners)
        {
            return Ok(storage);
        }
        self.record_alone(&mut emit, PassMode::Auto, self.comp().meta.ty == ElemTy::Bool).map_err(Failure::into_error)
    }

    /// Records this reduction's own passes into a buffer: `bits` for a packed `bool` value.
    pub(crate) fn record_alone(&self, emit: &mut Emit<'_>, mode: PassMode, bits: bool) -> std::result::Result<Arc<Storage>, Failure> {
        let ctx = emit.ctx();
        let comp = self.comp();
        let RunState::Reduce { operand, wgsl, elem, layout, finish, .. } = &comp.run;
        let ops = self.ops();
        let storage = Arc::new(Storage::new(ctx, ops.as_slice().len(), *layout));
        let pin = storage.pin(0)?;
        let array = Array::<u32>::from_node(ctx, Shape::from(operand.len()), Arc::clone(operand));
        let mut scratch = emit.scratch();
        reduce_into(emit, &array, bits, *elem, wgsl, ops, mode, &mut scratch, Some(pin.buffer()), *finish)?;
        Ok(storage)
    }

    /// Records this axis reduction into a buffer of one value per line, in one kernel with the axis reductions `partners` of the
    /// same lines (see [`Structured::merge_partners`]) where the program of their operands fits one: each of them then holds its own
    /// buffer. `None` (nothing recorded) if there are no partners or the program does not fit.
    pub(crate) fn record_axis_merged(&self, emit: &mut Emit<'_>, geom: AxisGeom, partners: &[Arc<Structured>]) -> std::result::Result<Option<Arc<Storage>>, Failure> {
        if partners.is_empty() {
            return Ok(None);
        }
        let ctx = emit.ctx();
        let all: Vec<&Structured> = std::iter::once(self).chain(partners.iter().map(|partner| &**partner)).collect();
        let operands: Vec<&Arc<Node>> = all.iter().map(|reduced| reduced.operand()).collect();
        let Some(plan) = jit::axis_plan(ctx, &operands, all.len()) else { return Ok(None) };
        let bytes = geom.lines as u64 * 4;
        let storages: Vec<Arc<Storage>> = all.iter().map(|reduced| Arc::new(Storage::new(ctx, geom.lines as usize, reduced.layout()))).collect();
        let pins = storages.iter().map(|storage| storage.pin(0)).collect::<Result<Vec<_>>>()?;
        let outs: Vec<crate::scan::Bound<'_>> = pins.iter().map(|pin| crate::scan::Bound { buffer: pin.buffer(), bytes }).collect();
        let ops: Vec<ReduceOp> = all.iter().map(|reduced| reduced.ops().as_slice()[0]).collect();
        crate::scan::reduce_axis_planned(emit, &plan, geom, &ops, &outs)?;
        drop(outs);
        drop(pins);
        for (reduced, storage) in all.iter().zip(&storages).skip(1) {
            reduced.adopt(Arc::clone(storage));
        }
        Ok(Some(Arc::clone(&storages[0])))
    }

    /// Records this axis reduction alone, into a buffer of one value per line.
    pub(crate) fn record_axis_lines(&self, emit: &mut Emit<'_>, geom: AxisGeom) -> std::result::Result<Arc<Storage>, Failure> {
        let ctx = emit.ctx();
        let storage = Arc::new(Storage::new(ctx, geom.lines as usize, self.layout()));
        let pin = storage.pin(0)?;
        let out = crate::scan::Bound { buffer: pin.buffer(), bytes: geom.lines as u64 * 4 };
        crate::scan::reduce_axis_node(emit, self.operand(), geom, self.ops().as_slice()[0], out)?;
        drop(pin);
        Ok(storage)
    }

    /// The first pass's workgroups, for the label of a Decision.
    pub(crate) fn first_groups(&self, ctx: &Context) -> u32 {
        ctx.tuning().reduce_groups(u32::try_from(self.comp().meta.len).unwrap_or(u32::MAX))
    }

    /// The shape that picks the strategy: the operand's length, or an axis reduction's lines.
    pub(crate) fn dims(&self) -> [u32; 4] {
        let comp = self.comp();
        match comp.meta.geom {
            Some(geom) => [geom.n, geom.stride, geom.lines, 1],
            None => [u32::try_from(comp.meta.len).unwrap_or(u32::MAX), 1, 1, 1],
        }
    }

    /// What identifies the family's value for the ranking cache: the ops, the element type, packed or not, axis or not.
    pub(crate) fn signature(&self, hasher: &mut impl std::hash::Hasher) {
        use std::hash::Hash;
        let comp = self.comp();
        comp.meta.tag.hash(hasher);
        comp.meta.ty.hash(hasher);
        self.layout().is_packed().hash(hasher);
    }

    /// The values on the host, without keeping them on the GPU.
    pub(crate) fn read<T: GpuType>(&self, ctx: &Context) -> Result<Vec<T>> {
        let comp = self.comp();
        if let Some(geom) = comp.meta.geom {
            // One value per line: recorded (with the axis reductions that are pending) and read like any array.
            let storage = self.force(ctx).as_ref().map_err(|error| LumaError::clone(error))?;
            return Ok(T::from_repr(ctx.read_storage::<T::Repr>(storage)?, geom.lines as usize));
        }
        if std::mem::size_of::<T::Repr>() == 4
            && let Some(words) = crate::host::reduced_words(self, ctx)
        {
            return Ok(T::from_repr(bytemuck::pod_collect_to_vec(&words), self.ops().as_slice().len()));
        }
        // Read with others that are pending, in one kernel, which records them all (before the speculation, which
        // records one reduction at a time).
        if self.is_unforced() && std::mem::size_of::<T::Repr>() == 4 {
            let partners = self.merge_partners(ctx);
            if !partners.is_empty() {
                self.forcing.store(true, Ordering::Relaxed);
                let result = self.record_with(|| self.record_listed(ctx, &partners).map_err(Arc::new));
                self.forcing.store(false, Ordering::Relaxed);
                let storage = result.as_ref().map_err(|error| LumaError::clone(error))?;
                let values = match self.packed_values::<T>(ctx) {
                    Some(values) => values?,
                    None => T::from_repr(ctx.read_storage::<T::Repr>(storage)?, self.ops().as_slice().len()),
                };
                // The leaves that stand for this value in expressions get its buffer, as `force` does.
                self.force(ctx);
                return Ok(values);
            }
        }
        let RunState::Reduce { operand, wgsl, buffer, finish, .. } = &comp.run;
        let array = Array::<T>::from_node(ctx, Shape::from(operand.len()), Arc::clone(operand));
        reduce_array(&array, wgsl, self.ops(), *finish, !*buffer)
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
        if self.is_wide() {
            return Array::resolve(self.context(), Shape::from(1usize), Err(self.too_wide().into()), name, location);
        }
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
        reduce_into(&mut Emit::now(ctx), self, true, R::LAYOUT.stride() as u64, "u32", ReduceList::new(&[op]), PassMode::Auto, &mut scratch, Some(pin.buffer()), Finish::default())?;
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

impl<T: GpuType> Array<T> {
    /// Whether the array has more than `u32::MAX` elements (only a file array can: see
    /// [`Array::open`]); the operations that need 32-bit counts or indices refuse it.
    pub(crate) fn is_wide(&self) -> bool {
        self.len() > u32::MAX as usize
    }

    /// The error of an operation that does not run on a wide array.
    pub(crate) fn too_wide(&self) -> LumaError {
        LumaError::DimensionTooLarge { shape: self.shape().dims() }
    }
}

impl<T: ScalarType> Array<T> {
    pub(crate) fn reduce_at<const N: usize>(&self, ops: [Reduction; N], name: &'static str, location: &'static Location<'static>) -> Array<T> {
        const { assert!(N >= 1 && N <= MAX_REDUCTIONS, "Array::reduce takes one to four reductions") };
        let result = self.run_reduce(OpTag::Reduce(ReduceList::new(&ops.map(Reduction::op))), name, location);
        Array::resolve(self.context(), Shape::from(N), result, name, location)
    }

    /// A reduction of the whole array that is `tag` (`Reduce(ops)`, or `Dot` for the sum of a product).
    pub(crate) fn reduce_tagged(&self, tag: OpTag, name: &'static str, location: &'static Location<'static>) -> Array<T> {
        let shape = Shape::from(tag.op_count());
        let result = self.run_reduce(tag, name, location);
        Array::resolve(self.context(), shape, result, name, location)
    }

    fn run_reduce(&self, tag: OpTag, name: &'static str, location: &'static Location<'static>) -> std::result::Result<Array<T>, Failure> {
        let ctx = self.context();
        let shape = Shape::from(tag.op_count());
        let elem = T::LAYOUT.stride() as u64;
        let wgsl = T::reduce_type();
        // Poisoned operands pass their error on.
        let src = self.src()?;
        // A file array of more than 2^32 elements: only the whole-array reductions that need no count or index.
        let elements = self.len();
        if elements > u32::MAX as usize && !matches!(name, "sum" | "prod" | "product" | "min" | "max" | "any" | "all") {
            return Err(self.too_wide().into());
        }
        // An `f32` reduction stays lazy even on a buffer, so that steps on its result fuse into
        // its last pass (see `Structured`); other types have no such steps and record at once.
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
            let reader = jit::create(ctx, elements, Op::new(OpKind::Unary(UnOp::Id), ElemTy::of::<T>()?), args, name, location)?;
            let ty = ElemTy::of::<T>()?;
            let len = reader.len();
            let reduced = Arc::new(Structured::new(
                Family::Comprehension(Comp {
                    form: Mutex::new(None),
                    meta: Meta { ty, len, tag, geom: None, launch: Launch::Now },
                    run: RunState::Reduce { operand: reader, wgsl, elem, layout: T::LAYOUT, buffer, finish: Finish::default(), solo: AtomicBool::new(false), pack: OnceLock::new() },
                }),
                ctx.recording_here(),
                name,
                location,
            ));
            ctx.jit().register_reduction(&reduced);
            return Ok(Array::from_structured(ctx, shape, reduced));
        }
        // Nothing is left to fuse: record the passes now.
        let ops = tag.reduce_list().expect("a reduction has ops");
        let storage = Arc::new(Storage::new(ctx, shape.len(), T::LAYOUT));
        let pin = storage.pin(0)?;
        let mut scratch = Scratch::new(ctx);
        reduce_into(&mut Emit::now(ctx), self, T::LAYOUT.is_packed(), elem, wgsl, ops, PassMode::Auto, &mut scratch, Some(pin.buffer()), Finish::default())?;
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
                let ops = ReduceList::new(&[op]);
                let comp = Comp {
                    form: Mutex::new(None),
                    meta: Meta { ty: ElemTy::F32, len: reader.len(), tag: OpTag::ReduceAxis(ops), geom: Some(geom), launch: Launch::Now },
                    run: RunState::Reduce { operand: Arc::clone(&reader), wgsl: "f32", elem: 4, layout: T::LAYOUT, buffer: false, finish: Finish::default(), solo: AtomicBool::new(false), pack: OnceLock::new() },
                };
                let reduced = Arc::new(Structured::new(Family::Comprehension(comp), false, name, location));
                ctx.jit().register_reduction(&reduced);
                Array::from_structured(ctx, shape, reduced)
            })
            .collect();
        Some(arrays)
    }

    /// This array, a reduction that has not been recorded yet, extended by `step` (with the
    /// value `word` of its scalar operand, as `f32` bits), as an array of `O`; `None` if the
    /// step cannot run inside the reduction's last pass: the array is not such a reduction of
    /// `f32`, or it has run or failed, or it has [`MAX_EPILOGUE`] steps already.
    pub(crate) fn fuse_step<O: ScalarType>(&self, step: EpilogueStep, word: u32) -> Option<Array<O>> {
        let reduced = self.structured()?;
        let comp = reduced.comp();
        let RunState::Reduce { operand, wgsl, elem, layout, buffer, finish, .. } = &comp.run;
        if !comp.meta.tag.is_whole_reduce() || *wgsl != "f32" || !reduced.is_unforced() || reduced.failure().is_some() {
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
        let mut finish = *finish;
        let at = finish.epilogue.steps().count();
        finish.epilogue = finish.epilogue.then(step)?;
        debug_assert!(at < MAX_EPILOGUE);
        finish.words[at] = word;
        let derived = Comp {
            form: Mutex::new(None),
            meta: Meta { ty: comp.meta.ty, len: comp.meta.len, tag: comp.meta.tag.clone(), geom: None, launch: Launch::Now },
            run: RunState::Reduce { operand: Arc::clone(operand), wgsl, elem: *elem, layout: *layout, buffer: *buffer, finish, solo: AtomicBool::new(false), pack: OnceLock::new() },
        };
        let derived = Arc::new(Structured::new(Family::Comprehension(derived), self.context().recording_here(), reduced.name, reduced.location));
        self.context().jit().register_reduction(&derived);
        Some(Array::from_structured(self.context(), self.shape(), derived))
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
        if self.is_wide() {
            return Array::resolve(self.context(), Shape::from(1usize), Err(self.too_wide().into()), "mean", Location::caller());
        }
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
    // The results of a bit-packed type are the words of a one-element array (see `Structured`).
    debug_assert!(!bits || count == 1);
    if let Some(values) = speculated(array, wgsl, ops, finish, speculate) {
        return values;
    }
    let mut scratch = Scratch::new(ctx);
    let current = reduce_into(&mut Emit::now(ctx), array, bits, elem, wgsl, ops, PassMode::Auto, &mut scratch, None, finish)?;
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
        && let Some(words) = jit::speculate::readback(node, ops, |scratch| reduce_into(&mut Emit::now(array.context()), array, bits, elem, wgsl, ops, PassMode::Auto, scratch, None, finish))
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
    emit: &mut Emit<'_>,
    mode: PassMode,
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
    finish_passes_gated(emit, mode, scratch, ops, wgsl, elem, current, n, per_op, target, finish, None)
}

/// [`finish_passes`] where the last pass also runs a gate (its id and the arguments of its
/// buffers, see `ReduceList::with_gate`) on the final value; one op of `f32`s over partial results
/// (`per_op`) that are more than one value, so that the last pass is a pass of one group.
#[allow(clippy::too_many_arguments)]
pub(crate) fn finish_passes_gated(
    emit: &mut Emit<'_>,
    mode: PassMode,
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
    passes(emit, mode, scratch, ops, wgsl, elem, current, n, per_op, target, finish.epilogue, &scalars[..used], gate)
}

/// [`finish_passes`] for the partial results of a first pass of a [merged](ReduceList::merged)
/// list (`n` per op, `ops` the list of the first pass): the last pass applies the epilogues of the
/// list, the scalars of which are `scalars`, in the order of its uniform fields.
#[allow(clippy::too_many_arguments)]
pub(crate) fn finish_passes_merged(
    emit: &mut Emit<'_>,
    scratch: &mut Scratch<'_>,
    ops: ReduceList,
    wgsl: &'static str,
    elem: u64,
    current: luma_core::Buffer,
    n: u32,
    target: &luma_core::Buffer,
    scalars: &[u32],
) -> Result<luma_core::Buffer> {
    passes(emit, PassMode::Auto, scratch, ops, wgsl, elem, current, n, true, Some(target), Epilogue::default(), scalars, None)
}

/// The passes of [`finish_passes`]: the last one applies `epilogue` (the epilogues of a merged list)
/// with the scalar operands `scalars`, and runs `gate` if there is one (see
/// [`finish_passes_gated`]).
#[allow(clippy::too_many_arguments)]
fn passes(
    emit: &mut Emit<'_>,
    mode: PassMode,
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
    let ctx = emit.ctx();
    let count = ops.as_slice().len() as u64;
    while !per_op || n > 1 {
        let groups = ctx.tuning().reduce_groups(n);
        let list = if per_op { ops.rest() } else { ops };
        // The pass with one group is the last: its output is the final value; so is that of a
        // first pass that finishes the reduction itself.
        let single_pass = !per_op && groups > 1 && mode.single(ctx);
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
            emit.launch(
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
        emit.launch(&core, fold_workgroups(ctx, groups)?, [n, 1, 1], &raw)?;
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
    emit: &mut Emit<'_>,
    array: &Array<T>,
    bits: bool,
    elem: u64,
    wgsl: &'static str,
    ops: ReduceList,
    mode: PassMode,
    scratch: &mut Scratch<'_>,
    target: Option<&luma_core::Buffer>,
    finish: Finish,
) -> Result<luma_core::Buffer> {
    let count = ops.as_slice().len() as u64;
    let run_passes = |emit: &mut Emit<'_>, scratch: &mut Scratch<'_>, current: luma_core::Buffer, n: u32, per_op: bool, target: Option<&luma_core::Buffer>, finish: Finish| {
        finish_passes(emit, mode, scratch, ops, wgsl, elem, current, n, per_op, target, finish)
    };

    let fused = match array.pending() {
        Some(node) => jit::reduce_first_pass(emit, mode, node, ops, scratch, target, finish).map_err(Failure::into_error)?,
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
                results.push(run_passes(emit, scratch, partial, groups, true, target, rest)?);
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
                    let (words, groups) = bits_first_pass(emit, scratch, ops, pin.buffer(), n, target)?;
                    results.push(run_passes(emit, scratch, words, groups, true, target, rest)?);
                } else {
                    results.push(run_passes(emit, scratch, pin.buffer().clone(), n, false, target, rest)?);
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
            emit.copy(partial, i * elem, &collect, (i * segments + j as u64) * elem, elem)?;
        }
    }
    run_passes(emit, scratch, collect, len_u32(results.len())?, true, target, finish)
}

/// The first pass of the reduction `ops` (one op) of the `n` bit-packed elements in `words`: the
/// buffer holding the partial results and how many there are. With one group they are the final
/// value, which is written into `target` if it is given.
fn bits_first_pass(
    emit: &mut Emit<'_>,
    scratch: &mut Scratch<'_>,
    ops: ReduceList,
    words: &luma_core::Buffer,
    n: u32,
    target: Option<&luma_core::Buffer>,
) -> Result<(luma_core::Buffer, u32)> {
    let ctx = emit.ctx();
    let count = n.div_ceil(32);
    let groups = ctx.tuning().reduce_groups(count);
    let out = match target {
        Some(target) if groups == 1 => target.clone(),
        _ => scratch.take(u64::from(groups) * 4)?,
    };
    let core = ctx.builtin(Builtin::ReduceBits(ops), "u32")?;
    ctx.jit().note_reduce_pass();
    emit.launch(
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
