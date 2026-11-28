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
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock, Weak};

use smallvec::SmallVec;

use super::epilogue::{EpiInput, EpiMap, EpiProgram, StepOperand};
use super::contraction::{self, ContractionAttrs, ContractionView, Geometry, Partner};
use super::select;
use super::sites::{LazyMode, Laziness, SiteSlot};
use super::strategy::{Output, Subject};
use super::{Comp, Family, Launch, Meta, RunState, Structured};
use crate::array::{Array, Failure, Storage};
use crate::builtin::{BinOp, UnOp};
use crate::gpu_type::{GpuType, ScalarType};
use crate::jit::program::Ref;
use crate::jit::{self, OpKind, Scalar};
use crate::ops::Rhs;
use crate::profile::{Decision, DecisionSite};
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
    /// The epilogue absorbed into this value (W3-U4), if it is a derived one.
    epilogue: Option<Absorbed>,
    /// Set when a consumer was absorbed from this value: forcing it is not a lone force for its site (see
    /// [`super::sites::Sites::note_forced_alone`]).
    absorbed_from: AtomicBool,
}

/// What an elementwise consumer of a lazy product added to it: the program the product's strategy applies to each output element
/// (`Absorbed::program`, relative to the plain product), the arrays its inputs read, and how to take the last step alone when the
/// value it was derived from is still in use.
pub(crate) struct Absorbed {
    pub(crate) program: Arc<EpiProgram>,
    /// One array per input of `program`, as the consumer read them (before any broadcast).
    pub(crate) arrays: SmallVec<[Array<f32>; 3]>,
    /// The shape of the output.
    shape: Shape,
    /// The value this one was derived from, if it is still alive: it was used elsewhere (or is yet), so the product is computed once,
    /// for both, and the last step runs over its buffer.
    parent: Weak<Structured>,
    /// The last step over the parent's value (`Ref::Acc`), with its input array if it has one.
    delta: Arc<EpiProgram>,
    delta_arrays: SmallVec<[Array<f32>; 1]>,
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
        if let Some(scaled) = rules::scaled_product(ctx, lhs, rhs, attrs, shape, location) {
            return scaled;
        }
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
        // An eager call is `contraction::run`, which checks the operands itself: only a lazy one checks them here.
        // A recording makes every call eager (`Sites::decide`), asked only of a call that would be lazy: the lookup is not free.
        let (laziness, slot) = ctx.jit().sites.decide(location, tuning, false, ctx.numerics_ref());
        if laziness == Laziness::Lazy && !ctx.recording_here() {
            let view = ContractionView { lhs, rhs, attrs, location, epilogue: None, partners: &[] };
            match view.check() {
                Err(failure) => return Array::resolve(ctx, shape, Err(failure), "matmul", location),
                Ok(geom) => return lazy_product(ctx, &view, geom, slot, shape),
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
    let (laziness, slot) = ctx.jit().sites.decide(location, tuning, false, ctx.numerics_ref());
    if laziness == Laziness::Eager || ctx.recording_here() {
        return None;
    }
    let Ok(Some(launch)) = bmm::plan(lhs, rhs, g) else { return None };
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
        let run = ContractionRun { operands: Mutex::new(Some([lhs.held_clone(), rhs.held_clone()])), kind, slot, inherited: OnceLock::new(), leaf_id: AtomicU64::new(0), depth, epilogue: None, absorbed_from: AtomicBool::new(false) };
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
        if contraction::TAKES_PARTNERS {
            ctx.jit().register_product(&value);
        }
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
            ContractionKind::Matmul { attrs, .. } => self.run_product(ctx, [&lhs, &rhs], *attrs, run),
            ContractionKind::Batched { geom, launch } => bmm::execute(ctx, &lhs, &rhs, geom, *launch),
        };
        drop(computing);
        ctx.jit().release_requested(held);
        self.forcing.store(false, Ordering::Relaxed);
        // The form holds sources of the operands: a forced product keeps neither.
        drop(lock(&self.comp().form).take());
        if run.epilogue.is_none() && !run.absorbed_from.load(Ordering::Relaxed) {
            ctx.jit().sites.note_forced_alone(run.slot);
        }
        outcome.map_err(|failure| match failure {
            Failure::Inherited(error) => error,
            Failure::New(error) => Arc::new(LumaError::Op { op: self.name, location: self.location, source: Arc::new(error) }),
        })
    }

    /// For the plan cache's walk (W3-U5): hashes what makes the value a structure into `hasher` (which operation, its types and extents:
    /// what `select` reads, since the key holds no buffer and no scalar value) and returns the operands it reads; `None` where a plan
    /// cannot hold the value: one that is being recorded or has been, a product with an absorbed epilogue or known properties, a batched
    /// product.
    pub(crate) fn plan_sources(&self, hasher: &mut impl std::hash::Hasher) -> Option<SmallVec<[Src; 2]>> {
        use std::hash::Hash;
        if !self.is_unforced() || self.forcing.load(Ordering::Relaxed) || self.in_recording {
            return None;
        }
        let comp = self.comp();
        comp.meta.tag.hash(hasher);
        comp.meta.ty.hash(hasher);
        comp.meta.len.hash(hasher);
        match &comp.run {
            RunState::Reduce { operand, wgsl, buffer, finish, layout, epi, .. } => {
                if let Some(geom) = comp.meta.geom {
                    [geom.n, geom.stride, geom.lines].hash(hasher);
                }
                wgsl.hash(hasher);
                buffer.hash(hasher);
                layout.is_packed().hash(hasher);
                finish.epilogue.is_empty().hash(hasher);
                let mut sources = SmallVec::new();
                sources.push(Src::Node(Arc::clone(operand)));
                // The epilogue of an axis reduction (W3-U4 D1): its structure, what it reads (the arrays, as they are read) and its scalars (as classes).
                epi.is_some().hash(hasher);
                if let Some(program) = epi {
                    program.instrs.hash(hasher);
                    program.outputs.hash(hasher);
                    for input in &program.inputs {
                        input.map.hash(hasher);
                        input.ty.hash(hasher);
                        sources.push(input.src.clone());
                    }
                    for scalar in &program.scalars {
                        scalar_class(*scalar).hash(hasher);
                        sources.push(Src::Scalar(*scalar));
                    }
                }
                Some(sources)
            }
            RunState::Contraction(run) => {
                let ContractionKind::Matmul { attrs } = &run.kind else { return None };
                if run.absorbed_from.load(Ordering::Relaxed) || run.inherited.get().is_some() {
                    return None;
                }
                let operands = lock(&run.operands);
                let [lhs, rhs] = operands.as_ref()?;
                if lhs.has_props() || rhs.has_props() || lhs.failed().is_some() || rhs.failed().is_some() {
                    return None;
                }
                attrs.hash(hasher);
                lhs.shape().dims().hash(hasher);
                rhs.shape().dims().hash(hasher);
                let mut sources = SmallVec::from_buf([lhs.src().ok()?, rhs.src().ok()?]);
                // An absorbed epilogue: its structure, and what it reads (the arrays, as they are read, and the scalars, as classes).
                if let Some(absorbed) = &run.epilogue {
                    // A value derived from one that is alive is computed with it (one product for both): not a plan of its own.
                    if absorbed.parent.strong_count() > 0 || absorbed.program.outputs.len() != 1 || absorbed.arrays.len() != absorbed.program.inputs.len() {
                        return None;
                    }
                    absorbed.program.instrs.hash(hasher);
                    absorbed.program.outputs.hash(hasher);
                    absorbed.shape.dims().hash(hasher);
                    for (array, input) in absorbed.arrays.iter().zip(&absorbed.program.inputs) {
                        if array.has_props() || array.failed().is_some() {
                            return None;
                        }
                        input.map.hash(hasher);
                        array.shape().dims().hash(hasher);
                        sources.push(array.src().ok()?);
                    }
                    // The passes of the epilogue are made when the value is forced, and the simplifier reads the values of their scalars then
                    // (`x * 1`, `x + 0`, `powf(x, 2)`, constants that fold): the values it reads are part of the key, and the others are classes.
                    for scalar in &absorbed.program.scalars {
                        scalar_class(*scalar).hash(hasher);
                        sources.push(Src::Scalar(*scalar));
                    }
                }
                Some(sources)
            }
        }
    }

    /// What a leaf standing for this value reads when it is recorded: the sources of its operands, `None` where they are not known
    /// (the value is being recorded). For the walks that look for the buffers a pending value depends on.
    pub(crate) fn sources(&self) -> Option<SmallVec<[Src; 2]>> {
        match &self.comp().run {
            RunState::Reduce { operand, epi, .. } => {
                let mut sources = SmallVec::new();
                sources.push(Src::Node(Arc::clone(operand)));
                if let Some(program) = epi {
                    sources.extend(program.inputs.iter().map(|input| input.src.clone()));
                }
                Some(sources)
            }
            RunState::Contraction(run) => {
                let operands = lock(&run.operands);
                let [lhs, rhs] = operands.as_ref()?;
                let mut sources = SmallVec::from_buf([lhs.src().ok()?, rhs.src().ok()?]);
                if let Some(absorbed) = &run.epilogue {
                    for array in &absorbed.arrays {
                        sources.push(array.src().ok()?);
                    }
                }
                Some(sources)
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
        let error = lhs.failed().or_else(|| rhs.failed()).or_else(|| self.epilogue.as_ref().and_then(|absorbed| absorbed.arrays.iter().find_map(|array| array.failed())))?;
        Some(self.inherited.get_or_init(|| Arc::clone(error)))
    }

    /// Whether the host strategy takes this product (a small one on a backend whose memory the host reads).
    pub(super) fn fits_host(&self, ctx: &Context, location: &'static Location<'static>) -> bool {
        let ContractionKind::Matmul { attrs, .. } = &self.kind else { return false };
        let operands = lock(&self.operands);
        let Some([lhs, rhs]) = operands.as_ref() else { return false };
        contraction::host_fits(&ContractionView { lhs, rhs, attrs: *attrs, location, epilogue: None, partners: &[] }, ctx)
    }

    /// The form of the product of `lhs` and `rhs` made at `location`: [`ContractionView::form`] for a `matmul`, the batched builder for
    /// `matmul_batched`.
    pub(crate) fn form(&self, lhs: &Array<f32>, rhs: &Array<f32>, location: &'static Location<'static>) -> Option<Comprehension> {
        match &self.kind {
            ContractionKind::Matmul { attrs, .. } => rules::with_operand_props(ContractionView { lhs, rhs, attrs: *attrs, location, epilogue: None, partners: &[] }.form(), lhs, rhs),
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
        // A product with an absorbed epilogue has no form of its own yet: the rewrite engine would read it as the plain product.
        if self.epilogue.is_some() {
            return None;
        }
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

// -- absorption (W3-U4) --

impl Structured {
    /// Runs a product that has no value to wait for. With an absorbed epilogue: if the value it was derived from is alive its product
    /// is computed once for both and the last step runs over its buffer (the cost of the unfused code); otherwise the strategy gets the
    /// whole program, and one that does not take it returns the plain product, over which the program runs as one elementwise pass (bit
    /// for bit what the consumers would have computed). Pending products that share the left operand go along if a strategy takes them.
    fn run_product(&self, ctx: &Context, [lhs, rhs]: [&Array<f32>; 2], attrs: ContractionAttrs, run: &ContractionRun) -> Result<Arc<Storage>, Failure> {
        let absorbed = run.epilogue.as_ref();
        if let Some(absorbed) = absorbed
            && let Some(parent) = absorbed.parent.upgrade()
        {
            let acc = match parent.force(ctx) {
                Ok(storage) => Arc::clone(storage),
                Err(error) => return Err(Failure::Inherited(Arc::clone(error))),
            };
            return apply_separately(ctx, &absorbed.delta, &absorbed.delta_arrays, Src::Buffer(acc), absorbed.shape, self.location);
        }
        let candidates = if contraction::TAKES_PARTNERS { self.partner_candidates(ctx, lhs, attrs, absorbed) } else { SmallVec::new() };
        let partners: SmallVec<[Partner; 3]> = candidates.iter().map(|candidate| candidate.partner.clone()).collect();
        let view = ContractionView { lhs, rhs, attrs, location: self.location, epilogue: absorbed.map(|absorbed| &*absorbed.program), partners: &partners };
        match select::execute(ctx, &Subject::Contraction(&view))? {
            Output::Epilogued(storages) => Ok(storages.into_iter().next().expect("one output")),
            Output::Group(storages) if storages.len() == candidates.len() + 1 => {
                let mut storages = storages.into_iter();
                let own = storages.next().expect("the subject's output");
                for (candidate, storage) in candidates.iter().zip(storages) {
                    candidate.value.finish_member(storage);
                    ctx.jit().sites.note_partnered(candidate.slot);
                }
                if let RunState::Contraction(run) = &self.comp().run {
                    ctx.jit().sites.note_partnered(run.slot);
                }
                // Rule 4 (`shared_operand`) fired: the products of one left operand ran as one.
                if ctx.jit().recording_decisions() {
                    ctx.jit().note_decision(Decision::new("matmul", self.location, DecisionSite::Rewrite, "shared_operand"));
                }
                Ok(own)
            }
            output => {
                let storage = output.into_storage();
                match absorbed {
                    None => Ok(storage),
                    Some(absorbed) => {
                        if ctx.jit().recording_decisions() {
                            ctx.jit().note_decision(Decision::new("epilogue", self.location, DecisionSite::Epilogue, "separate: no epilogue kernel"));
                        }
                        // The plan cache records the passes that follow as part of this value's step, and replays them.
                        match crate::jit::plancache::separate_pass(ctx, self, &storage) {
                            crate::jit::plancache::Pass::Run => apply_separately(ctx, &absorbed.program, &absorbed.arrays, Src::Buffer(storage), absorbed.shape, self.location),
                            crate::jit::plancache::Pass::Replayed(result) => result,
                        }
                    }
                }
            }
        }
    }

    /// The pending products that could be launched with this one (the subject, whose operands were taken: `lhs` is its left one): of
    /// this thread and context, not in a recording, with the same left operand (the same buffer or node), the same attributes and the
    /// same epilogue instructions or none, at most [`MAX_PARTNERS`]. A product that waits for a value it was derived from is not one.
    pub(super) fn partner_candidates(&self, ctx: &Context, lhs: &Array<f32>, attrs: ContractionAttrs, absorbed: Option<&Absorbed>) -> SmallVec<[Candidate; 3]> {
        let mut found = SmallVec::new();
        if self.in_recording || ctx.recording_here() {
            return found;
        }
        let Ok(mine) = lhs.src() else { return found };
        for value in lock(&ctx.jit().products).iter().filter_map(Weak::upgrade) {
            if found.len() + 1 >= MAX_PARTNERS {
                break;
            }
            let RunState::Contraction(run) = &value.comp().run else { continue };
            let ContractionKind::Matmul { attrs: theirs } = &run.kind else { continue };
            if std::ptr::eq(Arc::as_ptr(&value), self) || *theirs != attrs || !value.is_unforced() || value.forcing.load(Ordering::Relaxed) || value.in_recording || value.thread != self.thread || value.failure().is_some() {
                continue;
            }
            let same_epilogue = match (&run.epilogue, absorbed) {
                (None, None) => true,
                (Some(theirs), Some(mine)) => theirs.parent.strong_count() == 0 && theirs.program.instrs == mine.program.instrs && theirs.program.outputs == mine.program.outputs,
                _ => false,
            };
            if !same_epilogue {
                continue;
            }
            let operands = lock(&run.operands);
            let Some([their_lhs, their_rhs]) = operands.as_ref() else { continue };
            if their_lhs.shape() != lhs.shape() || their_lhs.src().ok().is_none_or(|theirs| !same_source(&theirs, &mine)) {
                continue;
            }
            let partner = Partner { rhs: their_rhs.hold(), epilogue: run.epilogue.as_ref().map(|theirs| (Arc::clone(&theirs.program), theirs.arrays.clone())) };
            let slot = run.slot;
            drop(operands);
            found.push(Candidate { value, partner, slot });
        }
        found
    }

    /// Takes `storage`, which a group launch computed together with another value's product, as the result: the operands are released
    /// as if the value had been recorded alone.
    fn finish_member(&self, storage: Arc<Storage>) {
        if let RunState::Contraction(run) = &self.comp().run {
            drop(lock(&run.operands).take());
        }
        drop(lock(&self.comp().form).take());
        self.adopt(storage);
    }
}

/// What the simplifier (`jit::simplify`) can tell of a scalar: its bits where a rule reads its value (zeros, ones, the exponents of
/// `powf`, values whose products, sums and reciprocals may leave the range of `f32`), `0` for every other value of `f32`.
fn scalar_class(scalar: Scalar) -> u64 {
    if scalar.ty == ElemTy::F32 {
        let value = f32::from_bits(scalar.bits as u32).abs();
        if value.is_finite() && (2f32.powi(-60)..=2f32.powi(60)).contains(&value) && ![0.5, 1.0, 2.0].contains(&value) {
            return 0;
        }
    }
    scalar.bits | 1 << 63
}

/// The most products one launch takes (the subject and its partners).
pub(crate) const MAX_PARTNERS: usize = 4;

/// A pending product that may be launched with the one being recorded.
pub(super) struct Candidate {
    value: Arc<Structured>,
    partner: Partner,
    slot: SiteSlot,
}

/// Whether two operands are the same buffer or the same pending node.
fn same_source(a: &Src, b: &Src) -> bool {
    match (a, b) {
        (Src::Buffer(a), Src::Buffer(b)) => Arc::ptr_eq(a, b),
        (Src::Node(a), Src::Node(b)) => Arc::ptr_eq(a, b),
        _ => false,
    }
}

impl crate::jit::JitState {
    /// Lists the lazy product `value` for the partner search (see [`Structured::partner_candidates`]).
    pub(crate) fn register_product(&self, value: &Arc<Structured>) {
        let mut list = lock(&self.products);
        // Dead or recorded ones are dropped as the list doubles.
        if list.len() >= 32 && list.len().is_power_of_two() {
            list.retain(|weak| weak.upgrade().is_some_and(|value| value.is_unforced()));
        }
        list.push(Arc::downgrade(value));
    }
}

/// `program` evaluated over `acc` as one fused elementwise kernel: the backend-neutral way to run an epilogue, the instructions
/// built as the nodes the consumers would have been. `arrays` are the program's inputs.
pub(crate) fn apply_separately(ctx: &Context, program: &EpiProgram, arrays: &[Array<f32>], acc: Src, shape: Shape, location: &'static Location<'static>) -> Result<Arc<Storage>, Failure> {
    let mut temps: SmallVec<[Arc<Node>; 8]> = SmallVec::new();
    for instr in program.instrs.iter() {
        let mut args = [Src::unused(), Src::unused(), Src::unused()];
        for (slot, &arg) in args.iter_mut().zip(&instr.args) {
            *slot = match arg {
                Ref::Acc => acc.clone(),
                Ref::Input(j) => arrays[j as usize].broadcast_operand(shape, location)?,
                Ref::Scalar(k) => Src::Scalar(program.scalars[k as usize]),
                Ref::Tmp(t) => Src::Node(Arc::clone(&temps[t as usize])),
                Ref::Unused | Ref::Sub(_) => Src::unused(),
            };
        }
        temps.push(jit::create(ctx, shape.len(), instr.op, args, "epilogue", location)?);
    }
    let last = temps.pop().expect("an epilogue has an instruction");
    let out = Array::<f32>::from_node(ctx, shape, last);
    out.storage().cloned()
}

/// The array that is `this` with `op` applied, if `this` is a lazy product that can take it into its epilogue.
#[inline(always)]
pub(crate) fn absorb_unary<T: ScalarType, O: ScalarType>(this: &Array<T>, op: UnOp, location: &'static Location<'static>) -> Option<Array<O>> {
    match this.structured() {
        Some(value) if matches!(value.comp().run, RunState::Contraction(_)) => absorb(this, value, StepOperand::None, OpKind::Unary(op), None, location),
        _ => None,
    }
}

/// [`absorb_unary`] for `this op rhs`.
#[inline(always)]
pub(crate) fn absorb_binary<T: ScalarType, O: ScalarType>(this: &Array<T>, rhs: &Rhs<'_, T>, op: BinOp, scalar_left: bool, location: &'static Location<'static>) -> Option<Array<O>> {
    let value = this.structured()?;
    if !matches!(value.comp().run, RunState::Contraction(_)) || op.needs_helper() {
        return None;
    }
    match rhs {
        Rhs::Scalar(x) => {
            let scalar = Scalar::of(*x).ok()?;
            absorb(this, value, StepOperand::Scalar { value: scalar, scalar_left }, OpKind::Binary(op), None, location)
        }
        Rhs::Array(other) if !scalar_left => absorb(this, value, StepOperand::None, OpKind::Binary(op), Some(*other as &dyn AnyArray), location),
        Rhs::Array(_) => None,
    }
}

/// An array of some type, as the epilogue sees an operand.
pub(crate) trait AnyArray {
    fn as_f32_array(&self) -> Option<Array<f32>>;
}

impl<T: GpuType> AnyArray for Array<T> {
    fn as_f32_array(&self) -> Option<Array<f32>> {
        self.as_f32()
    }
}

#[cold]
#[inline(never)]
fn absorb<T: GpuType, O: ScalarType>(this: &Array<T>, base: &Arc<Structured>, operand: StepOperand, kind: OpKind, other: Option<&dyn AnyArray>, location: &'static Location<'static>) -> Option<Array<O>> {
    let ctx = this.context();
    let RunState::Contraction(run) = &base.comp().run else { return None };
    let ContractionKind::Matmul { attrs } = &run.kind else { return None };
    if !base.is_unforced() || base.failure().is_some() || base.in_recording || ctx.recording_here() || base.thread != std::thread::current().id() {
        return None;
    }
    if this.has_props() || this.view().is_some() || this.as_f32().is_none() || ElemTy::of::<O>().ok()? != ElemTy::F32 {
        return None;
    }
    if let OpKind::Unary(UnOp::Id) = kind {
        return None;
    }
    if let OpKind::Unary(op) = kind
        && op.needs_helper()
    {
        return None;
    }
    let note = |chosen: String| {
        if ctx.jit().recording_decisions() {
            ctx.jit().note_decision(Decision::new("epilogue", location, DecisionSite::Epilogue, chosen));
        }
    };
    let shape = this.shape();
    // The other operand: an `f32` array of the output's shape, or one that varies along one axis only.
    let mut arrays: SmallVec<[Array<f32>; 1]> = SmallVec::new();
    let operand = match (operand, other) {
        (operand, None) => operand,
        (_, Some(other)) => {
            let array = other.as_f32_array()?;
            if !ctx.same_as(array.context()) || array.has_props() || array.is_read_view() {
                return None;
            }
            let (out, mine) = (shape.dims(), array.shape().dims());
            let map = if mine == out {
                EpiMap::Point
            } else if mine == [out[0], 1, 1, 1] {
                EpiMap::AlongCols
            } else if mine == [1, out[1], 1, 1] {
                EpiMap::AlongRows
            } else if mine == [1, 1, 1, 1] {
                EpiMap::Scalar
            } else {
                note("separate: operand shape".to_owned());
                return None;
            };
            let input = EpiInput { src: array.src().ok()?, map, ty: ElemTy::F32 };
            arrays.push(array);
            StepOperand::Input(input)
        }
    };
    let delta = Arc::new(EpiProgram::step(kind, operand));
    let program = match &run.epilogue {
        Some(parent) => parent.program.then(&delta),
        None => Some(EpiProgram::clone(&delta)),
    };
    let Some(program) = program else {
        note("separate: limit".to_owned());
        return None;
    };
    let operands = lock(&run.operands).as_ref().map(|[lhs, rhs]| [lhs.hold(), rhs.hold()])?;
    let mut all: SmallVec<[Array<f32>; 3]> = run.epilogue.as_ref().map(|parent| parent.arrays.clone()).unwrap_or_default();
    all.extend(arrays.iter().cloned());
    let count = program.instrs.len();
    for array in operands.iter().chain(&all) {
        array.record_for_reader();
    }
    let derived = ContractionRun {
        operands: Mutex::new(Some([operands[0].hold(), operands[1].hold()])),
        kind: ContractionKind::Matmul { attrs: *attrs },
        slot: run.slot,
        inherited: OnceLock::new(),
        leaf_id: AtomicU64::new(0),
        depth: run.depth,
        epilogue: Some(Absorbed { program: Arc::new(program), arrays: all.clone(), shape, parent: Arc::downgrade(base), delta, delta_arrays: arrays }),
        absorbed_from: AtomicBool::new(false),
    };
    let comp = Comp { form: Mutex::new(None), meta: Meta { ty: ElemTy::F32, len: base.comp().meta.len, tag: base.comp().meta.tag.clone(), geom: None, launch: Launch::AtForce }, run: RunState::Contraction(Box::new(derived)) };
    let value = Arc::new(Structured::new(Family::Comprehension(comp), false, base.name, base.location));
    let leaf = value.leaf(ctx, base.comp().meta.len);
    let RunState::Contraction(made) = &value.comp().run else { unreachable!("a product") };
    made.leaf_id.store(leaf.id(), Ordering::Relaxed);
    let reader = Arc::downgrade(&leaf);
    for array in operands.iter().chain(&all) {
        array.add_reader(&reader);
    }
    if contraction::TAKES_PARTNERS {
        ctx.jit().register_product(&value);
    }
    run.absorbed_from.store(true, Ordering::Relaxed);
    ctx.jit().sites.note_absorbed(run.slot);
    note(format!("absorbed {count} ops"));
    Some(Array::from_product(ctx, shape, value, leaf))
}
