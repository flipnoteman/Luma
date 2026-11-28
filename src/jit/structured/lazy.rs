//! The lazy product (W2-LAZY): a [`Structured`] value of the comprehension family whose `run` is [`RunState::Contraction`] and whose
//! launch is [`Launch::AtForce`]. `matmul` and its variants, and `matmul_batched` where one launch computes the batch, make one when
//! their call site is lazy (see [`super::sites`]); everything else about the product is as if it had launched at the call.
//!
//! **Creation** checks what depends on nothing but shapes, types, contexts and buffer sizes, and poisons the result with the error the
//! launch would have given (`ContractionView::check`, [`ContractionView::check_resources`]): those are the errors a test or a user can see
//! without using the value. Resource and device errors (an allocation, a grid, a compile, an operand that fails when it is evaluated)
//! surface when the value is **forced**, tagged with the operation and the call that made it, as `Array::resolve` tags them.
//!
//! The value is held by the arrays that stand for it together with its **leaf**, the node that stands for it in expressions: the leaf
//! registers as a reader of every operand (the buffer of an evaluated one, the pending node of an expression, the leaf of another
//! product), so a write into an operand evaluates the leaf, which forces the product before the write (`Readers`, the protocol of
//! `reduce::run_reduce` with one node for all operands). The arrays hold the leaf, and the leaf holds the value: dropping every array and
//! expression that reads it drops the product unevaluated, and with it the operands it pinned.

use std::panic::Location;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use smallvec::SmallVec;

use super::contraction::{self, ContractionAttrs, ContractionView, Geometry};
use super::select;
use super::sites::{LazyMode, Laziness, SiteSlot};
use super::strategy::{Output, Subject};
use super::{Comp, Family, Launch, Meta, RunState, Structured};
use crate::array::{Array, Failure, Storage};
use crate::bmm;
use crate::context::Context;
use crate::error::LumaError;
use crate::gpu::lock;
use crate::jit::comprehension::{Algebra, Comprehension, OpTag};
use crate::jit::rules;
use crate::jit::{Computing, ElemTy, Node, Src};
use crate::shape::Shape;

/// What a lazy product keeps until it is forced.
pub(crate) struct ContractionRun {
    /// `lhs` and `rhs`; taken when the product is forced, so that a forced product no longer holds its operands.
    operands: Mutex<Option<[Array<f32>; 2]>>,
    kind: ContractionKind,
    /// Where the call site's state is (see [`super::sites::Sites::note_forced_alone`]).
    slot: SiteSlot,
    /// The error of an operand once one was seen to have failed, so that [`ContractionRun::operand_failure`] hands out a reference.
    inherited: OnceLock<Arc<LumaError>>,
    /// The id of the leaf that stands for the product in expressions: what the thread is computing while it records the product, so that
    /// the expressions waiting to be evaluated that read it are left for after.
    leaf_id: AtomicU64,
    /// How many unrecorded products this one is at the end of a chain of (itself included): recording it records them first, one
    /// call deeper each, and dropping it drops them the same way.
    depth: u16,
}

/// The longest chain of unrecorded products. A longer one records the products it would read when it is made: recording and
/// dropping a chain recurse one level per product, and a loop that never uses its products must not grow the stack.
const MAX_CHAIN: u16 = 64;

/// The chain a product is the end of, if `array` is one that has not been recorded (0 for anything else).
fn chain_depth(array: &Array<f32>) -> u16 {
    match array.structured() {
        Some(value) if value.is_unforced() => match &value.comp().run {
            RunState::Contraction(run) => run.depth,
            RunState::Reduce { .. } => 0,
        },
        _ => 0,
    }
}

/// Which product it is.
pub(crate) enum ContractionKind {
    /// `matmul`, `matmul_tf32`, `matmul_f32` and the transposed forms: the registry's strategies run it.
    Matmul { attrs: ContractionAttrs },
    /// `matmul_batched` as one launch: [`bmm::route`] chose it at the call.
    Batched { geom: bmm::Geometry, launch: bmm::Launch },
}

// -- creation --

/// The product of `lhs` and `rhs` (a `matmul` of any precision): launched now, or a value that launches when it is used. `shape` is
/// the shape a failure keeps (see `Array::resolve`).
pub(crate) fn product(lhs: &Array<f32>, rhs: &Array<f32>, attrs: ContractionAttrs, shape: Shape, location: &'static Location<'static>) -> Array<f32> {
    let ctx = lhs.context();
    // The rules of tier 1 (W3-R4R5): a few tests of the operands say whether any of them may apply.
    let mut tagged = None;
    if rules::needs_t1(ctx, lhs, rhs) {
        match rules::product_hook(ctx, lhs, rhs, attrs, location) {
            rules::Product::Replaced(array) => return array,
            rules::Product::Props(props) => tagged = Some(props),
            rules::Product::Unchanged => {}
        }
    }
    let product = launch_product(lhs, rhs, attrs, shape, location);
    match tagged {
        Some(props) => product.with_props(props),
        None => product,
    }
}

#[inline(always)]
fn launch_product(lhs: &Array<f32>, rhs: &Array<f32>, attrs: ContractionAttrs, shape: Shape, location: &'static Location<'static>) -> Array<f32> {
    let ctx = lhs.context();
    let tuning = ctx.tuning();
    // The default mode never reads the site: the eager path is what it was before sites existed (unless the numerics want the
    // product unforced for the rules of tier 2).
    if tuning.lazy != LazyMode::Never || super::sites::wants_unforced(ctx.numerics_ref()) {
        let view = ContractionView { lhs, rhs, attrs, location, epilogue: None };
        match view.check() {
            Err(failure) => return Array::resolve(ctx, shape, Err(failure), "matmul", location),
            Ok(geom) => {
                let (laziness, slot) = ctx.jit().sites.decide(location, tuning, ctx.recording_here(), ctx.numerics_ref());
                if laziness == Laziness::Lazy {
                    return lazy_product(ctx, &view, geom, slot, shape);
                }
            }
        }
    }
    Array::resolve(ctx, shape, contraction::run(lhs, rhs, attrs, location), "matmul", location)
}

fn lazy_product(ctx: &Context, view: &ContractionView<'_>, geom: Geometry, slot: SiteSlot, shape: Shape) -> Array<f32> {
    if let Err(failure) = view.check_resources(ctx, geom) {
        return Array::resolve(ctx, shape, Err(failure), "matmul", view.location);
    }
    let out = [geom.n as usize, geom.m as usize, 1, 1];
    let kind = ContractionKind::Matmul { attrs: view.attrs };
    let (value, leaf) = Structured::contraction(ctx, view.lhs, view.rhs, kind, geom.n as usize * geom.m as usize, slot, view.location);
    Array::from_product(ctx, out.into(), value, leaf)
}

/// The batched product of `lhs` and `rhs` as a lazy value, or `None` where the product launches at the call (the default mode, a
/// recording, an eager site, a batch that [`bmm::route`] sends to the loop of products or fails, an operand that is chunked): the
/// caller then runs it as it always did. `shape` is what a failure keeps.
pub(crate) fn batched(lhs: &Array<f32>, rhs: &Array<f32>, g: &bmm::Geometry, shape: Shape, location: &'static Location<'static>) -> Option<Array<f32>> {
    let ctx = lhs.context();
    let tuning = ctx.tuning();
    if tuning.lazy == LazyMode::Never && !super::sites::wants_unforced(ctx.numerics_ref()) {
        return None;
    }
    let Ok(Some(launch)) = bmm::plan(lhs, rhs, g) else { return None };
    let (laziness, slot) = ctx.jit().sites.decide(location, tuning, ctx.recording_here(), ctx.numerics_ref());
    if laziness == Laziness::Eager {
        return None;
    }
    let fail = |failure: Failure| Some(Array::resolve(ctx, shape, Err(failure), "matmul_batched", location));
    // The errors `run` gives before it launches, in its order: an operand's own, a context mismatch, an array of several buffers.
    for array in [lhs, rhs] {
        if let Some(error) = array.failed() {
            return fail(Failure::Inherited(Arc::clone(error)));
        }
    }
    if !ctx.same_as(rhs.context()) {
        return fail(LumaError::ContextMismatch.into());
    }
    let out = g.out_shape();
    let len = Shape::from(out).len();
    if contraction::split_over_buffers(ctx, [lhs, rhs], len) {
        return fail(contraction::chunked());
    }
    let kind = ContractionKind::Batched { geom: *g, launch };
    let (value, leaf) = Structured::contraction(ctx, lhs, rhs, kind, len, slot, location);
    Some(Array::from_product(ctx, out.into(), value, leaf))
}

impl Structured {
    /// A lazy product of `lhs` and `rhs` (`len` elements) made at `location`, and its leaf. Records nothing: the leaf is registered as a
    /// reader of the operands, so that a write into one of them forces the product first.
    pub(crate) fn contraction(ctx: &Context, lhs: &Array<f32>, rhs: &Array<f32>, kind: ContractionKind, len: usize, slot: SiteSlot, location: &'static Location<'static>) -> (Arc<Structured>, Arc<Node>) {
        let mut depth = 1 + chain_depth(lhs).max(chain_depth(rhs));
        if depth > MAX_CHAIN {
            // The chain is recorded now, as a run of eager products would have been; a failure is the products' own, read again by
            // whatever reads them.
            let _ = lhs.storage();
            let _ = rhs.storage();
            depth = 1;
        }
        // A reduction that is not recorded yet has no node that lives to register with: it is recorded now, as the call would have.
        lhs.record_for_reader();
        rhs.record_for_reader();
        let (name, tag) = match &kind {
            ContractionKind::Matmul { attrs, .. } => ("matmul", OpTag::Matmul { tf32: attrs.tf32.request() }),
            ContractionKind::Batched { .. } => ("matmul_batched", OpTag::MatmulBatched),
        };
        let run = ContractionRun { operands: Mutex::new(Some([lhs.held_clone(), rhs.held_clone()])), kind, slot, inherited: OnceLock::new(), leaf_id: AtomicU64::new(0), depth };
        debug_assert!(run.form(lhs, rhs, location).is_none_or(|form| form.validate().is_ok()), "the form of an API product is well formed");
        let comp = Comp { form: Mutex::new(None), meta: Meta { ty: ElemTy::F32, len, tag, geom: None, launch: Launch::AtForce }, run: RunState::Contraction(Box::new(run)) };
        let value = Arc::new(Structured::new(Family::Comprehension(comp), false, name, location));
        let leaf = value.leaf(ctx, len);
        if let RunState::Contraction(run) = &value.comp().run {
            run.leaf_id.store(leaf.id(), Ordering::Relaxed);
        }
        let reader = Arc::downgrade(&leaf);
        lhs.add_reader(&reader);
        rhs.add_reader(&reader);
        (value, leaf)
    }

    /// Records the product into a new buffer: the operands are taken (so they are released when it returns), the strategy the registry
    /// selects runs, and a failure is tagged with the operation and the call that made the value. Nothing is read back.
    pub(super) fn record_contraction(&self, ctx: &Context, run: &ContractionRun) -> Result<Arc<Storage>, Arc<LumaError>> {
        // The rewrite engine plans the value first (one load and a branch while no rule of tier 2 is allowed): a chain of products in
        // another order, a reduction of a product that is never formed. The operands are taken once the plan ran.
        if ctx.jit().rules.t2_active
            && matches!(run.kind, ContractionKind::Matmul { .. })
            && lock(&run.operands).is_some()
            && let Some(planned) = {
                self.forcing.store(true, Ordering::Relaxed);
                let planned = rules::plan(ctx, self);
                self.forcing.store(false, Ordering::Relaxed);
                planned
            }
        {
            self.forcing.store(true, Ordering::Relaxed);
            let leaf_id = run.leaf_id.load(Ordering::Relaxed);
            let held = ctx.jit().hold_requested_reading(leaf_id);
            let computing = Computing::enter(leaf_id);
            let outcome = rules::run_planned(ctx, self, &planned);
            drop(computing);
            ctx.jit().release_requested(held);
            self.forcing.store(false, Ordering::Relaxed);
            drop(lock(&run.operands).take());
            drop(lock(&self.comp().form).take());
            ctx.jit().sites.note_forced_alone(run.slot);
            return outcome.map_err(|error| Arc::new(LumaError::Op { op: self.name, location: self.location, source: error }));
        }
        let Some([lhs, rhs]) = lock(&run.operands).take() else {
            return Err(Arc::new(LumaError::Op { op: self.name, location: self.location, source: Arc::new(LumaError::Gpu("a lazy product was recorded twice".into())) }));
        };
        self.forcing.store(true, Ordering::Relaxed);
        let leaf_id = run.leaf_id.load(Ordering::Relaxed);
        // Expressions waiting to be evaluated that read this product stay where they are until it is recorded: recording asks for the
        // others, and one that reads the product would wait for it forever.
        let held = ctx.jit().hold_requested_reading(leaf_id);
        let computing = Computing::enter(leaf_id);
        let outcome = match &run.kind {
            ContractionKind::Matmul { attrs, .. } => {
                let view = ContractionView { lhs: &lhs, rhs: &rhs, attrs: *attrs, location: self.location, epilogue: None };
                select::execute(ctx, &Subject::Contraction(&view)).map(Output::into_storage)
            }
            ContractionKind::Batched { geom, launch } => bmm::execute(ctx, &lhs, &rhs, geom, *launch),
        };
        drop(computing);
        ctx.jit().release_requested(held);
        self.forcing.store(false, Ordering::Relaxed);
        // The form holds sources of the operands: a forced product keeps neither.
        drop(lock(&self.comp().form).take());
        ctx.jit().sites.note_forced_alone(run.slot);
        outcome.map_err(|failure| match failure {
            Failure::Inherited(error) => error,
            Failure::New(error) => Arc::new(LumaError::Op { op: self.name, location: self.location, source: Arc::new(error) }),
        })
    }

    /// What a leaf standing for this value reads when it is recorded: the sources of its operands, `None` where they are not known
    /// (the value is being recorded). For the walks that look for the buffers a pending value depends on.
    pub(crate) fn sources(&self) -> Option<SmallVec<[Src; 2]>> {
        match &self.comp().run {
            RunState::Reduce { operand, .. } => {
                let mut sources = SmallVec::new();
                sources.push(Src::Node(Arc::clone(operand)));
                Some(sources)
            }
            RunState::Contraction(run) => {
                let operands = lock(&run.operands);
                let [lhs, rhs] = operands.as_ref()?;
                Some(SmallVec::from_buf([lhs.src().ok()?, rhs.src().ok()?]))
            }
        }
    }
}

impl ContractionRun {
    /// The error of an operand that failed, if one has.
    pub(super) fn operand_failure(&self) -> Option<&Arc<LumaError>> {
        if let Some(error) = self.inherited.get() {
            return Some(error);
        }
        let operands = lock(&self.operands);
        let [lhs, rhs] = operands.as_ref()?;
        let error = lhs.failed().or_else(|| rhs.failed())?;
        Some(self.inherited.get_or_init(|| Arc::clone(error)))
    }

    /// Whether the host strategy takes this product (a small one on a backend whose memory the host reads).
    pub(super) fn fits_host(&self, ctx: &Context, location: &'static Location<'static>) -> bool {
        let ContractionKind::Matmul { attrs, .. } = &self.kind else { return false };
        let operands = lock(&self.operands);
        let Some([lhs, rhs]) = operands.as_ref() else { return false };
        contraction::host_fits(&ContractionView { lhs, rhs, attrs: *attrs, location, epilogue: None }, ctx)
    }

    /// The form of the product of `lhs` and `rhs` made at `location`: [`ContractionView::form`] for a `matmul`, the batched builder for
    /// `matmul_batched`.
    pub(crate) fn form(&self, lhs: &Array<f32>, rhs: &Array<f32>, location: &'static Location<'static>) -> Option<Comprehension> {
        match &self.kind {
            ContractionKind::Matmul { attrs, .. } => rules::with_operand_props(ContractionView { lhs, rhs, attrs: *attrs, location, epilogue: None }.form(), lhs, rhs),
            ContractionKind::Batched { geom, .. } => {
                if !Algebra::builds_forms() {
                    return None;
                }
                let word = |extent: usize| u32::try_from(extent).ok();
                let dims = [word(geom.k)?, word(geom.m)?, word(geom.n)?, word(geom.batch())?];
                Some(Comprehension::contract_batched(lhs.src().ok()?, rhs.src().ok()?, dims, geom.lhs_batched, geom.rhs_batched))
            }
        }
    }

    /// [`ContractionRun::form`] of the operands the value still holds; `None` once it was forced.
    pub(crate) fn build_form(&self, location: &'static Location<'static>) -> Option<Comprehension> {
        let operands = lock(&self.operands);
        let [lhs, rhs] = operands.as_ref()?;
        self.form(lhs, rhs, location)
    }
}

impl ContractionView<'_> {
    /// The form of the product as the API call meant it (07 §2.1): `n` and `m` parallel, `k` reduce, `x0 * x1` added; the operands are
    /// the arrays as handed over (a transposed operand is a view node, with the same maps). The one function that says what a `matmul`
    /// is as a form: the lazy product's form is this. `None` under `LUMA_ALGEBRA=off`, or where the operands do not make a product
    /// (check [`ContractionView::check`]) or cannot be read as sources.
    pub(crate) fn form(&self) -> Option<Comprehension> {
        if !Algebra::builds_forms() {
            return None;
        }
        let geom = self.check().ok()?;
        Some(Comprehension::contract(self.lhs.src().ok()?, self.rhs.src().ok()?, geom, self.attrs))
    }

    /// The errors a launch would give that depend on buffer sizes: an operand that is evaluated in several buffers, a result that needs
    /// several. [`ContractionView::check`] has the pending operands'. They are creation-time errors of a lazy product (a test reads them
    /// without using the value) with the launch's own message.
    pub(crate) fn check_resources(&self, ctx: &Context, geom: Geometry) -> Result<(), Failure> {
        let len = u64::from(geom.n) * u64::from(geom.m);
        if contraction::split_over_buffers(ctx, [self.lhs, self.rhs], usize::try_from(len).unwrap_or(usize::MAX)) {
            return Err(contraction::chunked());
        }
        Ok(())
    }
}
