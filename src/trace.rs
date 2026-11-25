//! Compiled callables: [`trace`] records the launches a piece of array code makes once and
//! returns a [`Compiled`] that replays them over new input arrays. A call builds no graph, plans
//! nothing and fuses nothing: its host cost is the encoding of the recorded launches.
//!
//! A trace is the machinery of `iterate_all` pointed at a function instead of a loop step: the
//! function runs once on placeholder arrays (probes) while the context records its launches (see
//! `context::record`), and the recorded launches are replayed with the probes' buffers swapped for
//! the real arguments and the results' buffers for new ones (see `jit::replay`). What a replay
//! bakes (the extents, the workgroup counts, the uniform words) is why a *variant* is keyed by the
//! exact shapes of the arguments; the shape class only says how much a retrace costs.

use std::fmt;
use std::panic::Location;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, PoisonError, RwLock};

use luma_core::Buffer;

use crate::array::{Array, Failure, Storage};
use crate::context::{default_context, Cmd, CmdArg, Context};
use crate::erased::{AnyArray, Dyn, ElemKind, Erased, StorageRef};
use crate::error::{LumaError, Result};
use crate::gpu::hash::FxMap;
use crate::gpu::lock;
use crate::gpu::pager::{Accounted, Pin};
use crate::gpu_type::{ElemLayout, GpuType};
use crate::jit::{self, Forced, Mix, Plan, Trace};
use crate::shape::Shape;

/// The environment variable that sets [`TraceOptions::verify_every`] for every callable.
const VERIFY_ENV: &str = "LUMA_TRACE_VERIFY";

/// Signatures a callable remembers as running eagerly (see [`TraceOptions::eager_fallback`]).
const MAX_EAGER: usize = 64;

// -- Options -------------------------------------------------------------------------------------

/// How a [`Compiled`] behaves. Build one with [`TraceOptions::new`] and the methods below.
///
/// ```
/// use luma::TraceOptions;
/// let opts = TraceOptions::new().strict().max_variants(2);
/// # let _ = opts;
/// ```
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub struct TraceOptions {
    strict: bool,
    max_variants: usize,
    eager_fallback: bool,
    verify_every: u32,
}

impl Default for TraceOptions {
    fn default() -> Self {
        TraceOptions::new()
    }
}

impl TraceOptions {
    /// The defaults: a new signature retraces, at most 8 variants, a function that cannot be
    /// traced is an error, and verification as `LUMA_TRACE_VERIFY` says (off when it is not set).
    pub fn new() -> Self {
        let verify_every = std::env::var(VERIFY_ENV).ok().and_then(|text| text.trim().parse().ok()).unwrap_or(0);
        TraceOptions { strict: false, max_variants: 8, eager_fallback: false, verify_every }
    }

    /// A signature that has no variant is an error ([`TraceError::Signature`]) instead of a
    /// retrace: use it with [`Compiled::precompile`] for a callable whose shapes are known.
    pub fn strict(mut self) -> Self {
        self.strict = true;
        self
    }

    /// Keeps at most `n` variants (default 8, at least 1): a new one drops the least recently
    /// called. A loop that alternates over more shapes than variants retraces at every call.
    pub fn max_variants(mut self, n: usize) -> Self {
        self.max_variants = n.max(1);
        self
    }

    /// A function that cannot be recorded ([`TraceError::Aborted`], [`TraceError::Nested`],
    /// [`TraceError::Segmented`], [`TraceError::Empty`]) makes the call run the function
    /// eagerly on the real arguments instead of failing, with a one-time warning (through the
    /// `log` crate) naming the reason. The signature is then run eagerly from the next call on.
    pub fn eager_fallback(mut self) -> Self {
        self.eager_fallback = true;
        self
    }

    /// Every `n`-th call of a variant traces the function again into a scratch variant and
    /// compares the two: a function whose structure changed (it reads a counter, a `Scalar` that was
    /// set in between, ...) gives [`TraceError::StructureChanged`] and its variant is dropped.
    /// 0 turns verification off; `LUMA_TRACE_VERIFY=<n>` sets the default.
    pub fn verify_every(mut self, n: u32) -> Self {
        self.verify_every = n;
        self
    }
}

// -- Errors --------------------------------------------------------------------------------------

/// Why a function could not be traced or a [`Compiled`] could not be called. Carried by
/// [`LumaError::Trace`].
#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum TraceError {
    /// The arguments match none of the traced variants and the callable is
    /// [strict](TraceOptions::strict), or an input was asked for with the wrong index or type
    /// ([`Inputs::get`]). `expected` lists what exists, `got` what was asked.
    Signature {
        /// The signatures the callable has a variant for (or the input that exists).
        expected: Vec<String>,
        /// The signature of the call (or the request).
        got: String,
    },
    /// The function did something a recording cannot hold: read data back, uploaded host data,
    /// synchronised, cleared a buffer, ... Create constants outside the traced function and
    /// capture them.
    Aborted {
        /// What the function did.
        reason: &'static str,
    },
    /// This thread is already recording (a trace inside `iterate_all`, inside another trace, or a
    /// compiled call inside the step of a loop). Call [`Compiled::precompile`] before entering the
    /// outer recording, or call the function itself there.
    Nested,
    /// The function writes into input `index`. Return the new value instead.
    WritesInput {
        /// The argument.
        index: usize,
    },
    /// The function writes into an array it captured. Return the new value instead.
    WritesCaptured,
    /// The function launched nothing and returned none of its inputs.
    Empty,
    /// An argument or output spans several buffers (see [`LumaError::Chunked`]).
    Segmented {
        /// Which one.
        what: &'static str,
    },
    /// Verification traced the function again and found another structure than the stored one
    /// (see [`TraceOptions::verify_every`]).
    StructureChanged,
}

impl fmt::Display for TraceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TraceError::Signature { expected, got } if expected.is_empty() => write!(f, "no traced variant for ({got}): the callable has none"),
            TraceError::Signature { expected, got } => write!(f, "no traced variant for ({got}); it has: {}", expected.iter().map(|sig| format!("({sig})")).collect::<Vec<_>>().join(", ")),
            TraceError::Aborted { reason } => write!(
                f,
                "the function cannot be traced: it did something that needs the data on the host or in order ({reason}); create constants outside the traced function and capture them"
            ),
            TraceError::Nested => f.write_str(
                "a compiled callable cannot be traced or called while this thread records a loop step or another trace; call `precompile` before entering the outer recording, or call the function itself there",
            ),
            TraceError::WritesInput { index } => write!(f, "the traced function writes into input {index}; return the new value instead"),
            TraceError::WritesCaptured => f.write_str("the traced function writes into an array it captured; return the new value instead"),
            TraceError::Empty => f.write_str("the traced function launched nothing and returned none of its inputs"),
            TraceError::Segmented { what } => write!(f, "{what} spans several buffers; a compiled callable needs arrays that fit one buffer"),
            TraceError::StructureChanged => f.write_str("the traced function no longer has the structure it was traced with (it depends on state that changed between calls)"),
        }
    }
}

impl std::error::Error for TraceError {}

impl TraceError {
    /// Whether [`TraceOptions::eager_fallback`] runs the function eagerly instead of failing.
    fn falls_back(&self) -> bool {
        matches!(self, TraceError::Aborted { .. } | TraceError::Nested | TraceError::Segmented { .. } | TraceError::Empty)
    }
}

fn refused(error: TraceError) -> Failure {
    Failure::New(LumaError::Trace(error))
}

/// The error a public method returns for a failure inside it.
fn report(failure: Failure, op: &'static str, location: &'static Location<'static>) -> LumaError {
    match failure {
        Failure::Inherited(error) => LumaError::clone(&error),
        Failure::New(error) => LumaError::Op { op, location, source: Arc::new(error) },
    }
}

// -- Shape classes and variant information -------------------------------------------------------

/// How one extent of a shape reaches the generated kernels: an extent of 1 is not an axis, a
/// power of two divides with shifts and masks, any other extent divides with a runtime divisor.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[non_exhaustive]
pub enum DimClass {
    /// An extent of 1.
    One,
    /// A power of two of 2 or more.
    Pow2,
    /// Any other extent.
    Other,
}

/// The classes of the extents of a shape that change the kernels generated for it. Extents are
/// uniform words, not kernel text, so retracing a function at another shape of the same class
/// compiles no kernel (the kernel cache answers); across a class boundary only the kernels the
/// new class needs are built.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[non_exhaustive]
pub struct ShapeClass {
    /// The class of each of the four extents.
    pub dims: [DimClass; 4],
    /// Whether the array has `2^23` elements or more (its indices need the wide division).
    pub wide: bool,
}

impl ShapeClass {
    /// The class of `shape`.
    pub fn of(shape: Shape) -> ShapeClass {
        let dims = shape.dims().map(|extent| match extent {
            1 => DimClass::One,
            e if e.is_power_of_two() => DimClass::Pow2,
            _ => DimClass::Other,
        });
        ShapeClass { dims, wide: shape.len() >= 1 << 23 }
    }
}

/// One traced variant of a [`Compiled`], as [`Compiled::variants`] lists it.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct VariantInfo {
    /// The shape of each argument.
    pub shapes: Vec<Shape>,
    /// The element type of each argument.
    pub elems: Vec<ElemKind>,
    /// The class of each argument's shape.
    pub classes: Vec<ShapeClass>,
    /// Launches one call encodes.
    pub launches: usize,
    /// Calls this variant has served.
    pub calls: u64,
    /// Bytes of the captured arrays the variant keeps resident.
    pub pinned_bytes: u64,
    /// Bytes of the buffers the variant made up for its intermediates and placeholders.
    pub scratch_bytes: u64,
}

/// What a [`Compiled`] has done so far, see [`Compiled::stats`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct TraceStats {
    /// Calls, of every kind.
    pub calls: u64,
    /// Traces made, the first one and the retraces.
    pub traces: u64,
    /// Traces made after the first: a new signature, or a variant that was dropped.
    pub retraces: u64,
    /// Variants dropped for being the least recently called when `max_variants` was reached.
    pub evicted: u64,
    /// Calls that ran the function eagerly ([`TraceOptions::eager_fallback`]).
    pub eager_calls: u64,
    /// Verification traces made ([`TraceOptions::verify_every`]).
    pub verifications: u64,
    /// Kernels the context compiled during traces, all of them.
    pub compiled_kernels: u64,
    /// Of those, the ones compiled during retraces (it counts the kernels the whole context built
    /// meanwhile): 0 for a retrace within a shape class.
    pub retrace_compiled_kernels: u64,
    /// Variants alive now.
    pub variants: usize,
    /// Bytes of captured arrays the variants keep resident.
    pub pinned_bytes: u64,
    /// Bytes of the buffers the variants made up for their intermediates and placeholders.
    pub scratch_bytes: u64,
}

// -- Inputs, outputs, results --------------------------------------------------------------------

/// The arrays a traced function is called with: the probes of one trace (or, when it runs
/// eagerly, the real arguments).
pub struct Inputs<'a> {
    ctx: &'a Context,
    probes: &'a [Erased],
}

impl Inputs<'_> {
    /// Number of arguments.
    pub fn len(&self) -> usize {
        self.probes.len()
    }

    /// Whether the callable was called without arguments.
    pub fn is_empty(&self) -> bool {
        self.probes.is_empty()
    }

    /// Argument `i` as an `Array<T>`: a poisoned array ([`TraceError::Signature`], at the next
    /// synchronisation point) if there is no such argument or it holds another element type.
    #[track_caller]
    pub fn get<T: GpuType>(&self, i: usize) -> Array<T> {
        let location = Location::caller();
        let problem = match self.probes.get(i) {
            Some(probe) => match probe.clone().downcast::<T>() {
                Ok(array) => return array,
                Err(probe) => TraceError::Signature { expected: vec![format!("input {i}: {} {}", probe.elem(), probe.shape())], got: format!("input {i}: {}", T::ELEM_KIND) },
            },
            None => TraceError::Signature {
                expected: (0..self.probes.len()).map(|j| format!("input {j}: {} {}", self.probes[j].elem(), self.probes[j].shape())).collect(),
                got: format!("input {i}"),
            },
        };
        let error = LumaError::Op { op: "Inputs::get", location, source: Arc::new(LumaError::Trace(problem)) };
        Array::poisoned(self.ctx, Shape::from(1), Arc::new(error))
    }
}

/// What a traced function returns: any number of arrays of any element types.
///
/// Build it with `.into()` from one array, a tuple of up to eight, an array or a `Vec` of arrays
/// of one type, or with [`Outputs::new`] and [`Outputs::push`].
#[derive(Debug, Default)]
pub struct Outputs {
    items: Vec<Dyn>,
}

impl Outputs {
    /// No outputs yet.
    pub fn new() -> Self {
        Outputs::default()
    }

    /// Appends `array`.
    pub fn push<T: GpuType>(mut self, array: Array<T>) -> Self {
        self.items.push(array.into());
        self
    }
}

impl<T: GpuType> From<Array<T>> for Outputs {
    fn from(array: Array<T>) -> Self {
        Outputs { items: vec![array.into()] }
    }
}

impl<T: GpuType, const N: usize> From<[Array<T>; N]> for Outputs {
    fn from(arrays: [Array<T>; N]) -> Self {
        Outputs { items: arrays.into_iter().map(Dyn::from).collect() }
    }
}

impl<T: GpuType> From<Vec<Array<T>>> for Outputs {
    fn from(arrays: Vec<Array<T>>) -> Self {
        Outputs { items: arrays.into_iter().map(Dyn::from).collect() }
    }
}

macro_rules! outputs_from_tuple {
    ($($name:ident),+) => {
        impl<$($name: GpuType),+> From<($(Array<$name>,)+)> for Outputs {
            #[allow(non_snake_case)]
            fn from(($($name,)+): ($(Array<$name>,)+)) -> Self {
                Outputs { items: vec![$(Dyn::from($name)),+] }
            }
        }
    };
}

outputs_from_tuple!(A);
outputs_from_tuple!(A, B);
outputs_from_tuple!(A, B, C);
outputs_from_tuple!(A, B, C, D);
outputs_from_tuple!(A, B, C, D, E);
outputs_from_tuple!(A, B, C, D, E, F);
outputs_from_tuple!(A, B, C, D, E, F, G);
outputs_from_tuple!(A, B, C, D, E, F, G, H);

/// What a call of a [`Compiled`] returned: the outputs of the traced function, in order. Each is
/// an array of its own (nothing is shared with another call's).
#[derive(Debug)]
pub struct Results {
    items: Vec<Option<Dyn>>,
}

impl Results {
    /// Number of outputs the function returns.
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// Whether the function returns nothing.
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// Output `i` as an `Array<T>`, once: an output that was taken is gone. Fails if there is no
    /// such output, it was taken, or it holds another element type (it stays in place then).
    pub fn take<T: GpuType>(&mut self, i: usize) -> Result<Array<T>> {
        let count = self.items.len();
        let Some(slot) = self.items.get_mut(i) else {
            return Err(LumaError::InvalidShape(format!("output {i} does not exist: the call returned {count} outputs")));
        };
        let Some(item) = slot.as_ref() else {
            return Err(LumaError::InvalidShape(format!("output {i} was already taken")));
        };
        if item.elem() != T::ELEM_KIND {
            return Err(LumaError::InvalidShape(format!("output {i} holds {}, not {}", item.elem(), T::ELEM_KIND)));
        }
        slot.take().expect("the slot was just looked at").take::<T>()
    }

    /// The outputs that were not taken, in order, without their element types.
    pub fn into_vec(self) -> Vec<Dyn> {
        self.items.into_iter().flatten().collect()
    }
}

// -- The callable --------------------------------------------------------------------------------

/// A function traced once per signature and replayed: see [`trace`].
///
/// Cloning is cheap and shares the variants; any thread may call it.
#[derive(Clone)]
pub struct Compiled {
    inner: Arc<Callable>,
}

/// Records the launches `f` makes once and returns a [`Compiled`] that replays them over new
/// arguments, with no graph building, planning or fusion work per call.
///
/// ```no_run
/// use luma::{trace, Array};
///
/// let step = trace(|inp| {
///     let (x, w) = (inp.get::<f32>(0), inp.get::<f32>(1));
///     let y = x.matmul(&w).maximum(0.0);
///     (y.sum(), y).into()
/// });
/// let x = Array::from_slice(&[1.0f32; 64], [8, 8])?;
/// let w = Array::from_slice(&[0.5f32; 64], [8, 8])?;
/// let mut out = step.call(&[&x, &w])?; // the first call traces, then runs
/// let (loss, y) = (out.take::<f32>(0)?, out.take::<f32>(1)?);
/// # let _ = (loss, y);
/// # Ok::<(), luma::LumaError>(())
/// ```
///
/// `f` is kept, because a new signature retraces, and runs once per trace on probe arrays, never
/// per call. It may build arrays from its [`Inputs`] and from arrays it captured, call any
/// operation that does not read data back, and assign into arrays it creates itself. It may not
/// read back, upload, synchronise, write into an input or a captured array, call `iterate_all`
/// or `iterate_until_all`, or depend on state that changes between calls (a counter, a changing
/// [`Scalar`](crate::Scalar)): the trace freezes it. A scalar that changes per call is passed
/// as a one-element array.
///
/// Captured arrays are bound by buffer: one updated in place between calls is seen by the next
/// call, one that is replaced is not. The variants keep the captured arrays resident until
/// [`Compiled::clear`] or the drop of the last clone.
pub fn trace<F>(f: F) -> Compiled
where
    F: Fn(&Inputs) -> Outputs + Send + Sync + 'static,
{
    trace_with(TraceOptions::new(), f)
}

/// [`trace`] with [`TraceOptions`].
pub fn trace_with<F>(opts: TraceOptions, f: F) -> Compiled
where
    F: Fn(&Inputs) -> Outputs + Send + Sync + 'static,
{
    Compiled { inner: Arc::new(Callable::new(opts, Box::new(f))) }
}

impl Compiled {
    /// Calls the function on `args`: runs the variant traced for their element types and shapes,
    /// tracing it first if there is none. Returns the function's outputs as arrays of their own.
    ///
    /// Nothing is read back: the call only records launches, like any operation on arrays.
    #[track_caller]
    pub fn call(&self, args: &[&dyn AnyArray]) -> Result<Results> {
        let location = Location::caller();
        self.inner.call(args).map_err(|failure| report(failure, "Compiled::call", location))
    }

    /// [`Compiled::call`] for a function with exactly one output, of element type `T`.
    #[track_caller]
    pub fn call1<T: GpuType>(&self, args: &[&dyn AnyArray]) -> Result<Array<T>> {
        let location = Location::caller();
        let mut results = self.inner.call(args).map_err(|failure| report(failure, "Compiled::call1", location))?;
        if results.len() != 1 {
            let error = LumaError::InvalidShape(format!("call1 needs a function with one output, this one returns {}", results.len()));
            return Err(LumaError::Op { op: "Compiled::call1", location, source: Arc::new(error) });
        }
        results.take::<T>(0).map_err(|error| LumaError::Op { op: "Compiled::call1", location, source: Arc::new(error) })
    }

    /// Traces the function for the signature of `args` without running it (and without a
    /// retrace later), so that a loop or a [strict](TraceOptions::strict) callable starts warm.
    /// A signature that has a variant already is left alone.
    #[track_caller]
    pub fn precompile(&self, args: &[&dyn AnyArray]) -> Result<()> {
        let location = Location::caller();
        self.inner.precompile(args).map_err(|failure| report(failure, "Compiled::precompile", location))
    }

    /// What the callable has done so far.
    pub fn stats(&self) -> TraceStats {
        self.inner.stats()
    }

    /// The traced variants, most recently called first.
    pub fn variants(&self) -> Vec<VariantInfo> {
        self.inner.variants()
    }

    /// Drops every variant: the buffers of the plans are freed and the captured arrays unpinned.
    /// The next call traces again.
    pub fn clear(&self) {
        self.inner.clear();
    }
}

impl fmt::Debug for Compiled {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Compiled").field("variants", &self.inner.read().len()).field("options", &self.inner.opts).finish_non_exhaustive()
    }
}

type Function = Box<dyn Fn(&Inputs<'_>) -> Outputs + Send + Sync>;

#[derive(Default)]
struct Counters {
    calls: AtomicU64,
    traces: AtomicU64,
    retraces: AtomicU64,
    evicted: AtomicU64,
    eager_calls: AtomicU64,
    verifications: AtomicU64,
    compiled_kernels: AtomicU64,
    retrace_compiled_kernels: AtomicU64,
}

struct Callable {
    f: Function,
    opts: TraceOptions,
    variants: RwLock<Vec<Arc<Variant>>>,
    /// Signatures that run eagerly (only with `eager_fallback`).
    eager: Mutex<Vec<Signature>>,
    /// Orders the calls: a variant's `used` is the tick of its last call.
    clock: AtomicU64,
    warned: AtomicBool,
    counters: Counters,
}

/// What a variant was traced for.
struct Signature {
    ctx: Context,
    stamp: u64,
    args: Box<[(ElemKind, Shape)]>,
}

impl Signature {
    fn of(ctx: &Context, stamp: u64, args: &[&dyn AnyArray]) -> Signature {
        Signature { ctx: ctx.clone(), stamp, args: args.iter().map(|arg| (arg.elem(), arg.shape())).collect() }
    }

    fn matches(&self, ctx: &Context, stamp: u64, args: &[&dyn AnyArray]) -> bool {
        self.stamp == stamp && self.args.len() == args.len() && self.ctx.same_as(ctx) && args.iter().zip(&*self.args).all(|(arg, &(elem, shape))| arg.elem() == elem && arg.shape() == shape)
    }

    fn render(&self) -> String {
        render(self.args.iter().map(|&(elem, shape)| (elem, shape)))
    }
}

fn render(args: impl Iterator<Item = (ElemKind, Shape)>) -> String {
    args.map(|(elem, shape)| format!("{elem} {shape}")).collect::<Vec<_>>().join(", ")
}

/// What a call does for one output.
enum Out {
    /// A clone of argument `i` (the function handed an input back).
    Arg(usize),
    /// A clone of an array that existed before the trace and that the plan does not write.
    Captured(Erased),
    /// The array that output slot `slot` holds this call.
    Slot { slot: usize, kind: ElemKind, shape: Shape },
}

/// A storage a call allocates for the outputs it computes.
struct Slot {
    len: usize,
    layout: ElemLayout,
    /// The placeholder the plan names where it names this output's buffer; swapped for the new
    /// storage's buffer. `None` for a window (see `window`).
    placeholder: Option<Buffer>,
    /// A range of a pack buffer that the plan writes and keeps: copied out after the replay.
    window: Option<Window>,
}

struct Window {
    pack: Buffer,
    offset: u64,
    bytes: u64,
}

/// One trace of the function: the plan and what it takes and gives.
struct Variant {
    sig: Signature,
    plan: Plan,
    /// The placeholders of the arguments, in order.
    probes: Vec<Buffer>,
    slots: Vec<Slot>,
    outputs: Vec<Out>,
    classes: Vec<ShapeClass>,
    /// Hash of the structure of the plan and of what it takes and gives (see [`fingerprint`]).
    fingerprint: (u64, u64),
    /// Bytes counted for the placeholders (see `counted`).
    placeholder_bytes: u64,
    /// Counts the placeholders against the memory budget while the variant lives.
    #[allow(dead_code)]
    counted: Accounted,
    /// Serialises the encoding of one variant: its intermediates are shared by every call.
    encode: Mutex<()>,
    calls: AtomicU64,
    used: AtomicU64,
}

impl Callable {
    fn new(opts: TraceOptions, f: Function) -> Callable {
        Callable {
            f,
            opts,
            variants: RwLock::new(Vec::new()),
            eager: Mutex::new(Vec::new()),
            clock: AtomicU64::new(0),
            warned: AtomicBool::new(false),
            counters: Counters::default(),
        }
    }

    fn read(&self) -> std::sync::RwLockReadGuard<'_, Vec<Arc<Variant>>> {
        self.variants.read().unwrap_or_else(PoisonError::into_inner)
    }

    fn write(&self) -> std::sync::RwLockWriteGuard<'_, Vec<Arc<Variant>>> {
        self.variants.write().unwrap_or_else(PoisonError::into_inner)
    }

    fn call(&self, args: &[&dyn AnyArray]) -> std::result::Result<Results, Failure> {
        let owned;
        let ctx: &Context = match args.first() {
            Some(first) => first.context(),
            None => {
                owned = default_context()?;
                &owned
            }
        };
        if args.iter().any(|arg| !ctx.same_as(arg.context())) {
            return Err(Failure::New(LumaError::ContextMismatch));
        }
        self.counters.calls.fetch_add(1, Ordering::Relaxed);
        if ctx.recording_open() {
            // The recording this thread has would have to hold a variant's launches and intermediates: not supported.
            if self.opts.eager_fallback {
                self.warn_once("it was called inside a recording (a loop step or another trace)");
                return self.eager(ctx, args);
            }
            return Err(refused(TraceError::Nested));
        }
        let stamp = jit::stamp(ctx);
        let variant = match self.find(ctx, stamp, args) {
            Some(variant) => variant,
            None => match self.missing(ctx, stamp, args, true)? {
                Some(variant) => variant,
                None => return self.eager(ctx, args),
            },
        };
        let calls = variant.calls.fetch_add(1, Ordering::Relaxed) + 1;
        if self.opts.verify_every > 0 && calls.is_multiple_of(u64::from(self.opts.verify_every)) {
            self.verify(&variant, ctx, stamp, args)?;
        }
        variant.run(args)
    }

    fn precompile(&self, args: &[&dyn AnyArray]) -> std::result::Result<(), Failure> {
        let owned;
        let ctx: &Context = match args.first() {
            Some(first) => first.context(),
            None => {
                owned = default_context()?;
                &owned
            }
        };
        if args.iter().any(|arg| !ctx.same_as(arg.context())) {
            return Err(Failure::New(LumaError::ContextMismatch));
        }
        if ctx.recording_open() {
            return Err(refused(TraceError::Nested));
        }
        let stamp = jit::stamp(ctx);
        if self.find(ctx, stamp, args).is_some() {
            return Ok(());
        }
        self.missing(ctx, stamp, args, false).map(|_| ())
    }

    /// The variant of the signature of `args`, marked as called now.
    fn find(&self, ctx: &Context, stamp: u64, args: &[&dyn AnyArray]) -> Option<Arc<Variant>> {
        let variants = self.read();
        let variant = variants.iter().find(|variant| variant.sig.matches(ctx, stamp, args))?;
        variant.used.store(self.clock.fetch_add(1, Ordering::Relaxed) + 1, Ordering::Relaxed);
        Some(Arc::clone(variant))
    }

    /// What a call finds when it has no variant: a new one (`Ok(Some)`), the order to run the
    /// function eagerly (`Ok(None)`, with `eager_fallback`), or the error. `call` is whether this
    /// is a call (a strict callable refuses those, not `precompile`).
    fn missing(&self, ctx: &Context, stamp: u64, args: &[&dyn AnyArray], call: bool) -> std::result::Result<Option<Arc<Variant>>, Failure> {
        if self.opts.eager_fallback && self.is_eager(ctx, stamp, args) {
            return Ok(None);
        }
        if call && self.opts.strict {
            let expected = self.read().iter().map(|variant| variant.sig.render()).collect();
            return Err(refused(TraceError::Signature { expected, got: render(args.iter().map(|arg| (arg.elem(), arg.shape()))) }));
        }
        match self.trace(ctx, stamp, args) {
            Ok(variant) => Ok(Some(self.insert(variant))),
            Err(Failure::New(LumaError::Trace(error))) if self.opts.eager_fallback && error.falls_back() => {
                self.warn_once(&error.to_string());
                let mut eager = lock(&self.eager);
                if eager.len() >= MAX_EAGER {
                    eager.remove(0);
                }
                eager.push(Signature::of(ctx, stamp, args));
                Ok(None)
            }
            Err(error) => Err(error),
        }
    }

    fn is_eager(&self, ctx: &Context, stamp: u64, args: &[&dyn AnyArray]) -> bool {
        lock(&self.eager).iter().any(|sig| sig.matches(ctx, stamp, args))
    }

    fn warn_once(&self, reason: &str) {
        if !self.warned.swap(true, Ordering::Relaxed) {
            log::warn!("luma::trace: running the function eagerly: {reason}");
        }
    }

    /// Runs the function on the real arguments, as plain code.
    fn eager(&self, ctx: &Context, args: &[&dyn AnyArray]) -> std::result::Result<Results, Failure> {
        self.counters.eager_calls.fetch_add(1, Ordering::Relaxed);
        let probes: Vec<Erased> = args.iter().map(|arg| arg.erased()).collect();
        let outputs = (self.f)(&Inputs { ctx, probes: &probes });
        Ok(Results { items: outputs.items.into_iter().map(Some).collect() })
    }

    /// Adds `variant` (or returns the one another thread added for the same signature first),
    /// dropping the least recently called if there are too many.
    fn insert(&self, variant: Variant) -> Arc<Variant> {
        let mut variants = self.write();
        if let Some(known) = variants.iter().find(|known| known.sig.stamp == variant.sig.stamp && known.sig.ctx.same_as(&variant.sig.ctx) && known.sig.args == variant.sig.args) {
            return Arc::clone(known);
        }
        while variants.len() >= self.opts.max_variants {
            let Some(oldest) = variants.iter().enumerate().min_by_key(|(_, variant)| variant.used.load(Ordering::Relaxed)).map(|(at, _)| at) else { break };
            variants.swap_remove(oldest);
            self.counters.evicted.fetch_add(1, Ordering::Relaxed);
        }
        variant.used.store(self.clock.fetch_add(1, Ordering::Relaxed) + 1, Ordering::Relaxed);
        let variant = Arc::new(variant);
        variants.push(Arc::clone(&variant));
        variant
    }

    /// Traces the function again and compares the structure with `variant`'s.
    fn verify(&self, variant: &Arc<Variant>, ctx: &Context, stamp: u64, args: &[&dyn AnyArray]) -> std::result::Result<(), Failure> {
        self.counters.verifications.fetch_add(1, Ordering::Relaxed);
        let scratch = self.trace(ctx, stamp, args);
        let same = scratch.as_ref().is_ok_and(|scratch| scratch.fingerprint == variant.fingerprint);
        if same {
            return Ok(());
        }
        // The stale variant goes; the next call traces again.
        self.write().retain(|known| !Arc::ptr_eq(known, variant));
        match scratch {
            Ok(_) => Err(refused(TraceError::StructureChanged)),
            Err(error) => Err(error),
        }
    }

    /// T3: runs the function once on probes while the context records, and keeps the plan.
    fn trace(&self, ctx: &Context, stamp: u64, args: &[&dyn AnyArray]) -> std::result::Result<Variant, Failure> {
        // 1. The arguments: one context, none poisoned, each in one buffer.
        let mut inputs: Vec<Erased> = Vec::with_capacity(args.len());
        for arg in args {
            let erased = arg.erased();
            if let Some(error) = erased.error() {
                return Err(Failure::Inherited(Arc::new(error.clone())));
            }
            inputs.push(erased);
        }
        // 2. Lazy arrays that are alive elsewhere run first: the function may read one, and the
        //    recording must not hold the launch that evaluates it.
        ctx.jit().evaluate_held();
        for input in &inputs {
            if input.storage()?.segments() != 1 {
                return Err(refused(TraceError::Segmented { what: "an argument" }));
            }
        }
        // 3. Probes: arrays of the arguments' shapes whose contents never matter.
        let mut probes: Vec<Erased> = Vec::with_capacity(inputs.len());
        let mut probe_storages: Vec<Arc<Storage>> = Vec::with_capacity(inputs.len());
        let mut probe_buffers: Vec<Buffer> = Vec::with_capacity(inputs.len());
        for input in &inputs {
            let probe = input.fresh_probe()?;
            let storage = Arc::clone(probe.storage()?);
            probe_buffers.push(storage.pin_quiet(0)?.buffer().clone());
            probe_storages.push(storage);
            probes.push(probe);
        }
        drop(inputs);
        let built = ctx.jit().kernels_built();
        let Some(mut trace) = Trace::start_unseeded(ctx, probe_storages.clone()) else {
            return Err(refused(TraceError::Nested));
        };
        // 4. The function, as plain code: a panic unwinds through `Trace::drop`, which ends the recording.
        let outputs = (self.f)(&Inputs { ctx, probes: &probes });
        // 5. The outputs, evaluated together; then what each one is.
        let outs: Vec<Erased> = outputs.items.into_iter().map(Dyn::into_erased).collect();
        for out in &outs {
            if let Some(error) = out.error() {
                return Err(Failure::Inherited(Arc::new(error.clone())));
            }
            if !ctx.same_as(out.context()) {
                return Err(Failure::New(LumaError::ContextMismatch));
            }
        }
        let refs: Vec<&Erased> = outs.iter().collect();
        let forced = jit::force_traced(&refs, &probe_storages)?;
        if let Some(reason) = ctx.abort_reason() {
            return Err(refused(TraceError::Aborted { reason }));
        }
        let mut classified = Classified::default();
        for (out, forced) in outs.iter().zip(&forced) {
            classified.classify(ctx, out, forced, &probe_buffers)?;
        }
        // 6. End the recording; what the function did to its inputs and to what it captured.
        let writes_outside = ctx.recording_writes_outside();
        let reason = ctx.abort_reason();
        let plan = trace.finish_plan();
        drop(trace);
        let Some(mut plan) = plan else {
            return Err(refused(TraceError::Aborted { reason: reason.unwrap_or("the recording could not be completed") }));
        };
        if let Some(index) = probe_buffers.iter().position(|probe| plan.writes(probe)) {
            return Err(refused(TraceError::WritesInput { index }));
        }
        if writes_outside {
            return Err(refused(TraceError::WritesCaptured));
        }
        if plan.launches() == 0 && !classified.outputs.iter().any(|out| matches!(out, Out::Arg(_))) {
            return Err(refused(TraceError::Empty));
        }
        // 7. As a stepper requires: the buffers a call swaps are written by the plan and are not intermediates.
        for slot in &classified.slots {
            if let Some((buffer, _)) = &slot.placeholder_of {
                if plan.is_transient(buffer) {
                    return Err(refused(TraceError::Aborted { reason: "an output aliases an intermediate" }));
                }
                if !plan.writes(buffer) {
                    return Err(refused(TraceError::Aborted { reason: "an output the function never wrote" }));
                }
            }
        }
        // 8. The probes and results of the trace are full size and only name buffers in the plan:
        //    small placeholders take their names and the memory goes with the arrays.
        let mut named: Vec<Buffer> = probe_buffers.clone();
        named.extend(classified.slots.iter().filter_map(|slot| slot.placeholder_of.as_ref().map(|(buffer, _)| buffer.clone())));
        let placeholders = named.iter().map(|buffer| ctx.alloc_buffer(16, buffer.usage())).collect::<Result<Vec<Buffer>>>()?;
        plan.forget_probes(&named, &placeholders);
        let mut counted = Accounted::new(ctx, 0);
        let placeholder_bytes: u64 = placeholders.iter().map(Buffer::size).sum();
        counted.add(placeholder_bytes);
        let mut spare = placeholders[args.len()..].iter();
        let slots: Vec<Slot> = classified
            .slots
            .into_iter()
            .map(|slot| Slot { len: slot.len, layout: slot.layout, placeholder: slot.placeholder_of.map(|_| spare.next().cloned().expect("a placeholder per result")), window: slot.window })
            .collect();
        let probes = placeholders[..args.len()].to_vec();
        // 9. The structure, and the variant.
        let fingerprint = fingerprint(&plan, &probes, &slots, &classified.outputs);
        let sig = Signature::of(ctx, stamp, args);
        let classes = sig.args.iter().map(|&(_, shape)| ShapeClass::of(shape)).collect();
        let first = self.counters.traces.fetch_add(1, Ordering::Relaxed) == 0;
        let kernels = ctx.jit().kernels_built().saturating_sub(built);
        self.counters.compiled_kernels.fetch_add(kernels, Ordering::Relaxed);
        if !first {
            self.counters.retraces.fetch_add(1, Ordering::Relaxed);
            self.counters.retrace_compiled_kernels.fetch_add(kernels, Ordering::Relaxed);
        }
        Ok(Variant {
            sig,
            plan,
            probes,
            slots,
            outputs: classified.outputs,
            classes,
            fingerprint,
            placeholder_bytes,
            counted,
            encode: Mutex::new(()),
            calls: AtomicU64::new(0),
            used: AtomicU64::new(0),
        })
    }

    fn stats(&self) -> TraceStats {
        let variants = self.read();
        TraceStats {
            calls: self.counters.calls.load(Ordering::Relaxed),
            traces: self.counters.traces.load(Ordering::Relaxed),
            retraces: self.counters.retraces.load(Ordering::Relaxed),
            evicted: self.counters.evicted.load(Ordering::Relaxed),
            eager_calls: self.counters.eager_calls.load(Ordering::Relaxed),
            verifications: self.counters.verifications.load(Ordering::Relaxed),
            compiled_kernels: self.counters.compiled_kernels.load(Ordering::Relaxed),
            retrace_compiled_kernels: self.counters.retrace_compiled_kernels.load(Ordering::Relaxed),
            variants: variants.len(),
            pinned_bytes: variants.iter().map(|variant| variant.plan.pinned_bytes()).sum(),
            scratch_bytes: variants.iter().map(|variant| variant.scratch_bytes()).sum(),
        }
    }

    fn variants(&self) -> Vec<VariantInfo> {
        let mut variants: Vec<Arc<Variant>> = self.read().clone();
        variants.sort_by_key(|variant| std::cmp::Reverse(variant.used.load(Ordering::Relaxed)));
        variants
            .iter()
            .map(|variant| VariantInfo {
                shapes: variant.sig.args.iter().map(|&(_, shape)| shape).collect(),
                elems: variant.sig.args.iter().map(|&(elem, _)| elem).collect(),
                classes: variant.classes.clone(),
                launches: variant.plan.launches(),
                calls: variant.calls.load(Ordering::Relaxed),
                pinned_bytes: variant.plan.pinned_bytes(),
                scratch_bytes: variant.scratch_bytes(),
            })
            .collect()
    }

    fn clear(&self) {
        // Dropped outside the lock: freeing a plan can take other locks.
        let dropped = std::mem::take(&mut *self.write());
        drop(dropped);
        lock(&self.eager).clear();
    }
}

/// The outputs of a trace, sorted into what a call does for each, and the storages a call allocates.
#[derive(Default)]
struct Classified {
    outputs: Vec<Out>,
    slots: Vec<TraceSlot>,
}

/// A [`Slot`] while the trace still has the buffers themselves.
struct TraceSlot {
    len: usize,
    layout: ElemLayout,
    /// The buffer the plan writes for this output, and the id it is looked up by.
    placeholder_of: Option<(Buffer, u64)>,
    window: Option<Window>,
}

impl Classified {
    /// Notes what a call does for `out`, which `forced` says is a probe handed back or evaluated.
    fn classify(&mut self, ctx: &Context, out: &Erased, forced: &Forced, probe_buffers: &[Buffer]) -> std::result::Result<(), Failure> {
        let (kind, shape) = (out.elem(), out.shape());
        let storage = match forced {
            Forced::Probe(i) => {
                self.outputs.push(Out::Arg(*i));
                return Ok(());
            }
            Forced::Storage(storage) => storage,
        };
        if storage.segments() != 1 {
            return Err(refused(TraceError::Segmented { what: "an output" }));
        }
        // Looking at a buffer must not pin a window: that copies it, which ends the recording.
        if let Some((pack, at)) = storage.window() {
            let pack_buffer = pack.pin_quiet(0)?.buffer().clone();
            if !ctx.recording_made(&pack_buffer) {
                self.outputs.push(Out::Captured(out.clone()));
                return Ok(());
            }
            let window = Window { pack: pack_buffer, offset: pack.layout().bytes(at), bytes: storage.byte_len() };
            let found = self.slots.iter().position(|slot| slot.window.as_ref().is_some_and(|known| known.pack == window.pack && known.offset == window.offset && known.bytes == window.bytes));
            let slot = found.unwrap_or_else(|| {
                self.slots.push(TraceSlot { len: storage.len(), layout: storage.layout(), placeholder_of: None, window: Some(window) });
                self.slots.len() - 1
            });
            self.outputs.push(Out::Slot { slot, kind, shape });
            return Ok(());
        }
        let buffer = storage.pin_quiet(0)?.buffer().clone();
        if probe_buffers.contains(&buffer) {
            return Err(refused(TraceError::Aborted { reason: "an output aliases an input in a form that cannot be replayed" }));
        }
        if !ctx.recording_made(&buffer) {
            self.outputs.push(Out::Captured(out.clone()));
            return Ok(());
        }
        let found = self.slots.iter().position(|slot| slot.placeholder_of.as_ref().is_some_and(|(known, _)| *known == buffer));
        let slot = match found {
            Some(slot) => {
                if self.slots[slot].len != storage.len() || self.slots[slot].layout != storage.layout() {
                    return Err(refused(TraceError::Aborted { reason: "two outputs share a buffer but not a layout" }));
                }
                slot
            }
            None => {
                let id = buffer.id();
                self.slots.push(TraceSlot { len: storage.len(), layout: storage.layout(), placeholder_of: Some((buffer, id)), window: None });
                self.slots.len() - 1
            }
        };
        self.outputs.push(Out::Slot { slot, kind, shape });
        Ok(())
    }
}

/// T6: a hash of what a call does, independent of which buffers it does it with: per command the
/// kernel, workgroups, grid, uniform bytes and for each storage argument its role and size, where
/// a role is the argument `i`, the result `k`, an intermediate or a captured array (the last two
/// numbered by first appearance); then what each output is.
fn fingerprint(plan: &Plan, probes: &[Buffer], slots: &[Slot], outputs: &[Out]) -> (u64, u64) {
    const PROBE: u64 = 1;
    const RESULT: u64 = 2;
    const PACK: u64 = 3;
    const TRANSIENT: u64 = 4;
    const CAPTURED: u64 = 5;
    let mut roles: FxMap<u64, (u64, u64)> = FxMap::default();
    for (i, probe) in probes.iter().enumerate() {
        roles.insert(probe.id(), (PROBE, i as u64));
    }
    for (k, slot) in slots.iter().enumerate() {
        if let Some(placeholder) = &slot.placeholder {
            roles.insert(placeholder.id(), (RESULT, k as u64));
        }
    }
    let (mut next_pack, mut next_transient, mut next_captured) = (0, 0, 0);
    let mut mix = Mix::new(plan.commands().len() as u64);
    for Cmd { kernel, workgroups, size, args } in plan.commands() {
        mix.word(kernel.id);
        workgroups.iter().chain(size).for_each(|&word| mix.word(u64::from(word)));
        mix.word(args.len() as u64);
        for arg in args {
            match arg {
                CmdArg::Storage { buffer, bytes, writable } => {
                    let (tag, number) = *roles.entry(buffer.id()).or_insert_with(|| {
                        let (tag, next) = if slots.iter().any(|slot| slot.window.as_ref().is_some_and(|window| window.pack == *buffer)) {
                            (PACK, &mut next_pack)
                        } else if plan.is_transient(buffer) {
                            (TRANSIENT, &mut next_transient)
                        } else {
                            (CAPTURED, &mut next_captured)
                        };
                        *next += 1;
                        (tag, *next - 1)
                    });
                    mix.word(0xA);
                    mix.word(tag);
                    mix.word(number);
                    mix.word(*bytes);
                    mix.word(u64::from(*writable));
                }
                CmdArg::Uniform(data) => {
                    mix.word(0xB);
                    mix.word(data.len() as u64);
                    for chunk in data.chunks(8) {
                        mix.word(chunk.iter().rev().fold(0u64, |word, &byte| (word << 8) | u64::from(byte)));
                    }
                }
            }
        }
    }
    mix.word(0xC);
    for out in outputs {
        match out {
            Out::Arg(i) => {
                mix.word(6);
                mix.word(*i as u64);
            }
            Out::Captured(array) => {
                mix.word(7);
                mix.word(array.elem() as u64);
                array.shape().dims().iter().for_each(|&extent| mix.word(extent as u64));
            }
            Out::Slot { slot, kind, shape } => {
                mix.word(8);
                mix.word(*slot as u64);
                mix.word(*kind as u64);
                shape.dims().iter().for_each(|&extent| mix.word(extent as u64));
            }
        }
    }
    (mix.a, mix.b)
}

impl Variant {
    fn scratch_bytes(&self) -> u64 {
        self.plan.transient_bytes() + self.placeholder_bytes
    }

    /// T5: encodes the recorded launches over `args` and new buffers for the results.
    fn run(&self, args: &[&dyn AnyArray]) -> std::result::Result<Results, Failure> {
        let ctx = &self.sig.ctx;
        let mut pins: Vec<Pin> = Vec::with_capacity(args.len() + self.slots.len());
        let mut subst: Vec<(Buffer, Buffer)> = Vec::with_capacity(args.len() + self.slots.len());
        // The arguments run first (a lazy one is evaluated here) and stay pinned until the launches are encoded.
        for (arg, probe) in args.iter().zip(&self.probes) {
            let StorageRef(storage) = arg.storage_ref().map_err(Failure::Inherited)?;
            if storage.segments() != 1 {
                return Err(refused(TraceError::Segmented { what: "an argument" }));
            }
            let pin = storage.pin_read(0)?;
            subst.push((probe.clone(), pin.buffer().clone()));
            pins.push(pin);
        }
        // New storages for what the plan computes, with pooled buffers (their contents are written before they are read).
        let mut fresh: Vec<Arc<Storage>> = Vec::with_capacity(self.slots.len());
        for slot in &self.slots {
            let storage = Arc::new(Storage::new(ctx, slot.len, slot.layout));
            let pin = storage.pin(0)?;
            if let Some(placeholder) = &slot.placeholder {
                subst.push((placeholder.clone(), pin.buffer().clone()));
            }
            pins.push(pin);
            fresh.push(storage);
        }
        {
            let _encoding = lock(&self.encode);
            self.plan.replay(&subst, None)?;
            for (slot, pin) in self.slots.iter().zip(&pins[args.len()..]) {
                if let Some(Window { pack, offset, bytes }) = &slot.window {
                    ctx.copy(pack, *offset, pin.buffer(), 0, *bytes)?;
                }
            }
        }
        drop(pins);
        let items = self
            .outputs
            .iter()
            .map(|out| {
                Some(Dyn::new(match out {
                    Out::Arg(i) => args[*i].erased(),
                    Out::Captured(array) => array.clone(),
                    Out::Slot { slot, kind, shape } => Erased::from_storage(*kind, ctx, *shape, Arc::clone(&fresh[*slot])),
                }))
            })
            .collect();
        Ok(Results { items })
    }
}
