//! Custom ops with strategies: the public extension surface of the planner.
//!
//! A custom op is declared once per [`Context`] ([`Context::custom_op`]): the element types of its
//! inputs and outputs, how the shapes of the outputs follow from the inputs ([`OutShape`]) and one
//! or more [`Strategy`]s, each with a `fits` test, an optional `cost` estimate and a `lower`
//! function that launches kernels. [`Op::apply`] selects a strategy as the planner selects
//! between its own (device features, the context's numerics policy, `fits`, then the lowest
//! `cost`, else registration order), records the choice as a [`Decision`](crate::Decision) of
//! site [`DecisionSite::Extension`](crate::DecisionSite::Extension) and runs it. `LUMA_STRATEGY=ext:<op>/<strategy>`
//! forces one.
//!
//! A custom op is an eager boundary: its inputs are evaluated first (an elementwise chain that
//! feeds it runs as one launch before it), its outputs are buffers that later expressions read
//! and fuse around. A custom op does not fuse with its neighbours; an elementwise function that
//! should is [`ElementwiseFn`]. Its launches are recorded in order with the
//! rest of the batch, so recordings and traces see them.
//!
//! # Safety contract
//!
//! - Memory safety is the kernel's trust level: kernels are `Checked` unless the author calls
//!   the `unsafe` [`KernelBuilder::unchecked_bounds`](crate::KernelBuilder::unchecked_bounds).
//!   `Trusted` cannot be reached. A strategy cannot obtain a backend, a buffer, a device pointer
//!   or a pin: [`Lower`] binds buffers by [`Bind`] position and owns the pins for the duration of
//!   the call. Scratch memory is charged to the memory budget and released after the op.
//! - A declared [`Reach`] is trusted for scheduling and verified for shape only. It cannot cause
//!   memory unsafety except together with `unchecked_bounds`.
//! - Closures must be `Send + Sync`, may run on any thread that applies the op, must not call back
//!   into the same context's recording or flush machinery except through [`Lower`], and `lower`
//!   must be idempotent and free of side effects apart from its launches: a later version may
//!   call it again with the same views. A panic in `lower` never unwinds through planner locks
//!   (it is caught at the call and returned as [`LumaError::Extension`]). Closures stay alive
//!   as long as the context.
//! - Names: building the same name with an identical declaration (inputs, outputs, strategy
//!   names) returns the same [`Op`]; a different declaration under an existing name is an error.
//! - Nothing persists: the decision log and `LUMA_STRATEGY` use `name/strategy` strings, which
//!   are stable across runs.

mod apply;
mod function;
#[cfg(test)]
mod function_tests;
#[cfg(test)]
mod tests;

use std::sync::{Arc, RwLock};

use crate::context::Context;
use crate::erased::{AnyArray, Dyn, ElemKind};
use crate::error::{LumaError, Result};
use crate::gpu::hash::FxMap;
use crate::gpu_type::GpuType;
use crate::kernel::{Reach, UniformValue};
use crate::shape::Shape;
use luma_core::{CostModel, Features};

pub use apply::{Bind, Fit, Lower, ScratchId};
pub use function::{elementwise_fn, ElemFnType, ElementwiseFn};

use crate::array::Array;
use crate::numerics::Numerics;

/// The element type and shape of one operand of a custom op, as a strategy sees it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct ArgInfo {
    /// The element type.
    pub elem: ElemKind,
    /// The shape.
    pub shape: Shape,
}

impl ArgInfo {
    /// An operand of `elem` and `shape`.
    pub fn new(elem: ElemKind, shape: Shape) -> ArgInfo {
        ArgInfo { elem, shape }
    }
}

/// How the shape of an output of a custom op follows from its inputs.
#[derive(Clone)]
#[non_exhaustive]
pub enum OutShape {
    /// The shape of input `i`.
    Same(usize),
    /// This shape.
    Fixed(Shape),
    /// Computed from the inputs and the `params` bytes of the call.
    #[allow(clippy::type_complexity)]
    With(Arc<dyn Fn(&[ArgInfo], &[u8]) -> Result<Shape> + Send + Sync>),
}

impl OutShape {
    /// A shape computed by `f` from the inputs and the `params` bytes.
    pub fn with(f: impl Fn(&[ArgInfo], &[u8]) -> Result<Shape> + Send + Sync + 'static) -> OutShape {
        OutShape::With(Arc::new(f))
    }

    /// Two declarations agree: `With` closures cannot be compared, any two are taken to agree.
    fn same_rule(&self, other: &OutShape) -> bool {
        match (self, other) {
            (OutShape::Same(a), OutShape::Same(b)) => a == b,
            (OutShape::Fixed(a), OutShape::Fixed(b)) => a == b,
            (OutShape::With(_), OutShape::With(_)) => true,
            _ => false,
        }
    }
}

impl std::fmt::Debug for OutShape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OutShape::Same(i) => write!(f, "Same({i})"),
            OutShape::Fixed(shape) => write!(f, "Fixed({shape:?})"),
            OutShape::With(_) => f.write_str("With(..)"),
        }
    }
}

/// How a strategy's results compare with the other strategies' of the same op: what the context's
/// numerics policy has to allow for the strategy to be chosen.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum NumericsClass {
    /// Bit-equal to the other strategies: always allowed.
    Exact,
    /// Sums in another order: needs [`Numerics::reassociate`].
    Reassociating,
    /// Computes at reduced precision (TF32, fast math): needs a policy with a reduced-precision feature.
    Reduced,
}

type FitsFn = Arc<dyn Fn(&Fit<'_>) -> bool + Send + Sync>;
type CostFn = Arc<dyn Fn(&Fit<'_>) -> f64 + Send + Sync>;
type LowerFn = Arc<dyn Fn(&mut Lower<'_>) -> Result<()> + Send + Sync>;

/// One way to run a custom op: when it applies, what it is estimated to cost, and how it launches.
#[derive(Clone)]
#[non_exhaustive]
pub struct Strategy {
    pub(crate) name: &'static str,
    pub(crate) needs: Features,
    pub(crate) numerics: NumericsClass,
    pub(crate) reach: Reach,
    pub(crate) fits: Option<FitsFn>,
    pub(crate) cost: Option<CostFn>,
    pub(crate) lower: LowerFn,
}

impl Strategy {
    /// A strategy `name` that runs `lower`. `lower` launches kernels through [`Lower::launch`]; it
    /// may return [`LumaError::Declined`] before it launched anything to let the next strategy
    /// run. It must be idempotent and free of side effects apart from its launches.
    pub fn new(name: &'static str, lower: impl Fn(&mut Lower<'_>) -> Result<()> + Send + Sync + 'static) -> Self {
        Strategy { name, needs: Features::NONE, numerics: NumericsClass::Exact, reach: Reach::Whole, fits: None, cost: None, lower: Arc::new(lower) }
    }

    /// Device features that must be present.
    #[must_use]
    pub fn needs(mut self, features: Features) -> Self {
        self.needs = features;
        self
    }

    /// The numerics class: `Exact` (the default), `Reassociating` or `Reduced`.
    #[must_use]
    pub fn numerics(mut self, class: NumericsClass) -> Self {
        self.numerics = class;
        self
    }

    /// How this strategy's launches relate to their arrays (default [`Reach::Whole`]); checked
    /// against the shapes at every launch as for a [`Kernel`](crate::Kernel).
    #[must_use]
    pub fn reach(mut self, reach: Reach) -> Self {
        self.reach = reach;
        self
    }

    /// Whether the strategy applies to these operands (default: always).
    #[must_use]
    pub fn fits(mut self, f: impl Fn(&Fit<'_>) -> bool + Send + Sync + 'static) -> Self {
        self.fits = Some(Arc::new(f));
        self
    }

    /// The estimated cost in nanoseconds. Called only when several strategies fit and all of them
    /// have a cost; otherwise registration order decides.
    #[must_use]
    pub fn cost(mut self, f: impl Fn(&Fit<'_>) -> f64 + Send + Sync + 'static) -> Self {
        self.cost = Some(Arc::new(f));
        self
    }
}

impl std::fmt::Debug for Strategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Strategy").field("name", &self.name).field("numerics", &self.numerics).field("reach", &self.reach).finish_non_exhaustive()
    }
}

/// What a registered op is.
pub(crate) struct OpDef {
    name: &'static str,
    inputs: Vec<ElemKind>,
    outputs: Vec<(ElemKind, OutShape)>,
    params: usize,
    strategies: Vec<Strategy>,
}

impl OpDef {
    fn same_declaration(&self, other: &OpDef) -> bool {
        self.inputs == other.inputs
            && self.params == other.params
            && self.outputs.len() == other.outputs.len()
            && self.outputs.iter().zip(&other.outputs).all(|(a, b)| a.0 == b.0 && a.1.same_rule(&b.1))
            && self.strategies.len() == other.strategies.len()
            && self.strategies.iter().zip(&other.strategies).all(|(a, b)| a.name == b.name)
    }
}

/// The custom ops of a context, and the strategies `LUMA_STRATEGY=ext:<op>/<strategy>` forces.
pub(crate) struct Registry {
    ops: RwLock<FxMap<&'static str, Arc<OpDef>>>,
    forced: Vec<(String, String)>,
    /// The elementwise functions registered on the context, by name: the id of the source each name holds.
    functions: std::sync::Mutex<FxMap<String, u16>>,
}

impl std::fmt::Debug for Registry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Registry").field("forced", &self.forced).finish_non_exhaustive()
    }
}

impl Registry {
    /// The registry of a context opened with the `LUMA_STRATEGY` text `strategy`: its `ext:` items.
    pub(crate) fn new(strategy: Option<&str>) -> Registry {
        let forced = strategy
            .into_iter()
            .flat_map(|text| text.split(','))
            .filter_map(|item| item.trim().strip_prefix("ext:"))
            .filter_map(|rest| rest.split_once('/'))
            .map(|(op, strategy)| (op.trim().to_owned(), strategy.trim().to_owned()))
            .collect();
        Registry { ops: RwLock::new(FxMap::default()), forced, functions: std::sync::Mutex::default() }
    }

    fn forced(&self, op: &str) -> Option<&str> {
        self.forced.iter().find(|(name, _)| name == op).map(|(_, strategy)| strategy.as_str())
    }
}

/// Warns once per text.
fn warn_once(text: String) {
    static WARNED: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());
    let mut warned = crate::gpu::lock(&WARNED);
    if !warned.contains(&text) {
        log::warn!("{text}");
        warned.push(text);
    }
}

impl Context {
    /// Starts declaring the custom op `name` (see the [module](crate::ext)).
    pub fn custom_op(&self, name: &str) -> OpBuilder<'_> {
        OpBuilder { ctx: self, name: name.to_owned(), inputs: Vec::new(), outputs: Vec::new(), params: 0, strategies: Vec::new() }
    }
}

/// Declares a custom op; created by [`Context::custom_op`].
#[must_use]
pub struct OpBuilder<'c> {
    ctx: &'c Context,
    name: String,
    inputs: Vec<ElemKind>,
    outputs: Vec<(ElemKind, OutShape)>,
    params: usize,
    strategies: Vec<Strategy>,
}

impl OpBuilder<'_> {
    /// One array operand of element type `T`.
    pub fn input<T: GpuType>(mut self) -> Self {
        self.inputs.push(ElemKind::of::<T>());
        self
    }

    /// One result of element type `T` and the shape `shape` says. At least one.
    pub fn output<T: GpuType>(mut self, shape: OutShape) -> Self {
        self.outputs.push((ElemKind::of::<T>(), shape));
        self
    }

    /// The size in bytes of the uniform block [`Op::apply`] takes (0: none); see [`params_bytes`].
    pub fn params(mut self, bytes: usize) -> Self {
        self.params = bytes;
        self
    }

    /// Adds a strategy; at least one. Registration order is the tie-break of the selection.
    pub fn strategy(mut self, strategy: Strategy) -> Self {
        self.strategies.push(strategy);
        self
    }

    /// Registers the op in the context (or returns the op that is registered under this name with
    /// the same declaration).
    pub fn build(self) -> Result<Op> {
        let OpBuilder { ctx, name, inputs, outputs, params, strategies } = self;
        if name.is_empty() {
            return Err(LumaError::InvalidShape("a custom op needs a name".into()));
        }
        if outputs.is_empty() {
            return Err(LumaError::InvalidShape(format!("custom op `{name}` declares no output")));
        }
        if strategies.is_empty() {
            return Err(LumaError::InvalidShape(format!("custom op `{name}` declares no strategy")));
        }
        for (_, rule) in &outputs {
            if let OutShape::Same(i) = rule
                && *i >= inputs.len()
            {
                return Err(LumaError::InvalidShape(format!("custom op `{name}`: an output takes the shape of input {i}, but there are {} inputs", inputs.len())));
            }
        }
        for (i, strategy) in strategies.iter().enumerate() {
            if strategies[..i].iter().any(|earlier| earlier.name == strategy.name) {
                return Err(LumaError::InvalidShape(format!("custom op `{name}`: two strategies are named `{}`", strategy.name)));
            }
        }
        let registry = ctx.ext_registry();
        let name: &'static str = crate::context::intern(name);
        let def = Arc::new(OpDef { name, inputs, outputs, params, strategies });
        {
            let mut ops = registry.ops.write().unwrap_or_else(std::sync::PoisonError::into_inner);
            if let Some(existing) = ops.get(name) {
                if existing.same_declaration(&def) {
                    return Ok(Op { ctx: ctx.clone(), def: Arc::clone(existing) });
                }
                return Err(LumaError::InvalidShape(format!("op `{name}` is already registered differently")));
            }
            ops.insert(name, Arc::clone(&def));
        }
        if let Some(forced) = registry.forced(name)
            && !def.strategies.iter().any(|strategy| strategy.name == forced)
        {
            let known: Vec<&str> = def.strategies.iter().map(|strategy| strategy.name).collect();
            warn_once(format!("luma: LUMA_STRATEGY forces ext:{name}/{forced}, but the op has no such strategy (known: {})", known.join(", ")));
        }
        Ok(Op { ctx: ctx.clone(), def })
    }
}

impl std::fmt::Debug for OpBuilder<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpBuilder").field("name", &self.name).field("inputs", &self.inputs).field("outputs", &self.outputs).finish_non_exhaustive()
    }
}

/// A custom op registered in a context; cheap to clone. Created by [`OpBuilder::build`].
#[derive(Clone)]
pub struct Op {
    ctx: Context,
    def: Arc<OpDef>,
}

impl std::fmt::Debug for Op {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Op").field("name", &self.def.name).field("inputs", &self.def.inputs).field("outputs", &self.def.outputs).finish_non_exhaustive()
    }
}

impl Op {
    /// The name the op was registered under.
    pub fn name(&self) -> &str {
        self.def.name
    }

    /// Runs the op on `inputs` with the uniform block `params` (as many bytes as the op declared;
    /// see [`params_bytes`]) and returns its outputs.
    ///
    /// Inputs still lazy are evaluated first (an op is a fusion barrier on its inputs); an
    /// input that is poisoned makes the call fail with its error; arrays split over several
    /// buffers fail with [`LumaError::Chunked`]. The strategy is selected as the
    /// [module](crate::ext) says; no strategy that fits is [`LumaError::NoStrategy`]. A strategy
    /// whose `lower` fails or panics makes the call fail with [`LumaError::Extension`] and its
    /// outputs are dropped; the inputs are untouched.
    #[track_caller]
    pub fn apply(&self, inputs: &[&dyn AnyArray], params: &[u8]) -> Result<Vec<Dyn>> {
        let location = std::panic::Location::caller();
        apply::apply(&self.ctx, &self.def, inputs, params, location)
    }

    /// [`Op::apply`] for an op with one output of type `T`.
    #[track_caller]
    pub fn apply1<T: GpuType>(&self, inputs: &[&dyn AnyArray], params: &[u8]) -> Result<Array<T>> {
        if self.def.outputs.len() != 1 {
            return Err(LumaError::InvalidShape(format!("op `{}` has {} outputs, apply1 needs one", self.def.name, self.def.outputs.len())));
        }
        let mut outputs = self.apply(inputs, params)?;
        match outputs.pop() {
            Some(output) => output.take::<T>(),
            None => Err(LumaError::InvalidShape(format!("op `{}` returned no output", self.def.name))),
        }
    }
}

/// The bytes of `value` laid out as `uniform(&value)` binds it: the `params` of [`Op::apply`].
///
/// Fails when encase cannot lay the value out (a runtime-sized array).
pub fn params_bytes<T: UniformValue>(value: &T) -> Result<Vec<u8>> {
    crate::kernel::uniform_bytes(value)
}

/// What the numerics policy of the context allows of a strategy of `class`.
pub(crate) fn numerics_allowed(class: NumericsClass, policy: &Numerics) -> bool {
    match class {
        NumericsClass::Exact => true,
        NumericsClass::Reassociating => policy.reassociate,
        NumericsClass::Reduced => policy.reduced != crate::numerics::Reduced::NONE,
    }
}

/// The cost model a [`Fit`] hands out.
pub(crate) fn cost_model(ctx: &Context) -> CostModel {
    ctx.cost_model()
}
