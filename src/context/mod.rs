use std::sync::{Arc, Mutex, OnceLock};

use luma_core::{Backend, BackendKind, Buffer, BufferUsage, CostModel, DeviceInfo, Decision, DecisionSite, DeviceKind, Features, KernelTrust, MatmulSpec, MemoryModel, Profile};

use crate::builtin::{Builtin, BuiltinKernel};
use crate::caps::Caps;
use crate::numerics::Reduced;
use crate::numerics::Numerics;
use crate::error::{LumaError, Result};
use crate::jit::cost;
use crate::jit::JitState;
use crate::jit::split::{SplitMode, StreamCounters, StreamStats};
use crate::jit::structured::select::StrategyOverride;
use crate::jit::structured::strategy::{FamilyTag, StrategyId};
use crate::gpu::pager::{Accounted, Pager};
use crate::gpu::built::BuildCache;
use crate::gpu::hash::FxMap;
use crate::gpu::lock;
use crate::gpu::pool::BufferPool;
use luma_core::hal::size_class;
use crate::kernel::{self, KernelBuilder, KernelCore, KernelKey};
use crate::tuning::{Tuning, TuningProfile};

mod batch;
mod record;
mod transfer;
#[cfg(test)]
mod tests;

pub(crate) use record::{Cmd, CmdArg, Recording};
pub(crate) use luma_core::RawArg;
use transfer::WaitHistory;

/// Usage of every array storage buffer (and the scratch buffers of multi-pass operations, so they
/// share a pool class).
pub(crate) const STORAGE_USAGE: BufferUsage = BufferUsage::STORAGE
    .union(BufferUsage::COPY_DST)
    .union(BufferUsage::COPY_SRC);

/// Kernels of library modules by (module, variant, entry point).
type ModuleKernels = std::collections::HashMap<(&'static str, usize, &'static str), Arc<KernelCore>>;

#[derive(Debug)]
struct Inner {
    /// Built-in kernels, built on first use of their (operation, element type).
    builtins: BuildCache<(Builtin, &'static str), Arc<KernelCore>>,
    /// The tables of Bluestein's algorithm by (length, inverse), see [`crate::spectral`].
    chirps: Mutex<std::collections::HashMap<(u32, bool), Arc<crate::spectral::ChirpTables>>>,
    /// Lazy-evaluation state: fused kernels and the pending expressions.
    jit: JitState,
    /// Custom kernels, built on first use of their (source, entry, workgroup).
    kernels: BuildCache<KernelKey, Arc<KernelCore>>,
    /// Kernels of library modules that write their WGSL themselves (see [`Context::cached_kernel`]),
    /// by (module, variant, entry point).
    module_kernels: Mutex<ModuleKernels>,
    /// Kernels of the dense linear algebra by a key that determines their source (see
    /// [`Context::keyed_kernel`]).
    keyed_kernels: Mutex<std::collections::HashMap<String, Arc<KernelCore>>>,
    pool: Mutex<BufferPool>,
    /// Buffers the pool released; the backend drops what it caches for them (bind groups) at the
    /// next submission, which frees them.
    released: Mutex<Vec<Buffer>>,
    /// Bytes of a full segment of an array, see [`Storage`](crate::array::Storage).
    segment_bytes: u64,
    /// Whether results are checked for NaN and Inf (see [`ContextOptions::check_finite`]).
    check_finite: bool,
    /// Whether kernels over streamed arrays are deferred into schedules (see [`ContextOptions::split`]).
    split: SplitMode,
    /// Schedules opened and not yet committed, on any thread: zero (the common case) makes every commit trigger a load.
    schedules_open: std::sync::atomic::AtomicU32,
    /// Held while a schedule runs: one runs at a time.
    schedule_lease: Mutex<()>,
    /// What schedules did.
    streams: StreamCounters,
    /// The numerics policy (see [`ContextOptions::numerics`]).
    numerics: Numerics,
    /// What the device can do, computed once when the context opens.
    caps: Caps,
    /// The strategies forced by `LUMA_STRATEGY` (see [`ContextOptions::strategy`]).
    strategy_override: StrategyOverride,
    /// The custom ops declared on this context (see [`crate::ext`]).
    ext: crate::ext::Registry,
    /// Which segments of arrays hold a device buffer.
    pager: Pager,
    /// Whether `Array::from_slice` uploads into mappable storage buffers (unified memory).
    zero_copy: bool,
    /// Kernel parameters for this GPU.
    tuning: Tuning,
    /// What launches, memory traffic and arithmetic cost on this GPU, set once the context exists.
    cost: OnceLock<CostModel>,
    /// How bytes move inside the device (see [`Context::memory_model`]), set once the context exists.
    memory: OnceLock<Option<MemoryModel>>,
    /// The one-word counter of the reductions that finish in their first launch (see
    /// [`Tuning::single_pass_reduce`]), zero between launches; made on first use.
    reduce_counter: OnceLock<Buffer>,
    /// Tells this context from the others of the process (the recordings of a thread are kept by it, see [`record`]).
    id: u64,
    /// How long recent waits took, which sets how long the next one polls before it blocks.
    waits: WaitHistory,
    /// Host memory of finished transfers, kept for the next stage (pinned memory is slow to allocate and fresh heap pages fault).
    stages: Mutex<Vec<luma_core::hal::Stage>>,
    /// Report the GPU as busy regardless of the queue (see `Context::set_assume_busy`).
    assume_busy: std::sync::atomic::AtomicBool,
    /// Report the GPU as idle regardless of the queue (see `Context::set_assume_idle`).
    assume_idle: std::sync::atomic::AtomicBool,
    /// Tests: bytes of device buffers created so far (see [`Context::created_bytes`]).
    #[cfg(test)]
    created_bytes: std::sync::atomic::AtomicU64,
    /// Largest buffer that can be allocated *and* bound as a storage buffer.
    max_buffer_bytes: u64,
    max_uniform_bytes: u64,
    max_workgroups_per_dimension: u32,
    /// Whether the device was opened with timestamp queries (see [`Context::profile`]).
    timestamps: bool,
    /// The device. Must stay the last field: fields drop in declaration order, and the backend
    /// hands its device to the next context, which must only happen once everything of this one
    /// (buffer pools, kernels) is gone.
    backend: Box<dyn Backend>,
}

/// Handle to a GPU device and everything needed to run work on it: compiled kernels, a pool of
/// reusable buffers and the batch of commands waiting to be submitted.
///
/// A `Context` is cheap to [`Clone`] (it is reference counted) and can be shared between threads.
/// Every [`Array`](crate::Array) holds a clone of the context it was created on. When the last
/// handle (context or array) is dropped, the context's kernels and buffers are freed; the GPU
/// device itself stays open and is reused by the next [`Context::new`]. Arrays can only be
/// combined with arrays of the same
/// context. Most programs never create one: [`default_context`] opens the GPU on first use and
/// the constructors without an `_on` suffix use it.
///
/// A context runs on one backend: NVIDIA CUDA, Apple Metal or WebGPU through wgpu (see
/// [`BackendKind`]; [`Context::backend`] says which).
///
/// # Batching
///
/// Operations are recorded into a pending command buffer and not submitted one by one. The batch
/// is submitted when the host needs a result ([`Array::to_vec`](crate::Array::to_vec),
/// [`Array::eval`](crate::Array::eval), reductions, [`Context::sync`]) and whenever it grows past
/// a few hundred dispatches. [`Context::flush`] submits it on request *without* waiting, so the
/// GPU can start working while the host keeps recording the next operations. Because the whole batch is one ordered stream of commands, no
/// ordering is lost: host data is uploaded through staging buffers *inside* the stream, buffers
/// released by a dropped array can be handed out again immediately (every use of the new owner is
/// recorded after every use of the old one), and every uniform value lives in its own slot until
/// its batch is submitted. The launches of a batch are recorded into one compute pass, and
/// everything that is already bound (pipeline, grid uniform, bind groups) is not bound again.
///
/// # Memory
/// Array data beyond the device's memory budget (80% of the device-local memory on Vulkan GPUs,
/// or [`ContextOptions::memory_budget`] / `LUMA_MEMORY_BUDGET`) or beyond what the driver will
/// allocate is not an error: the least recently used buffers move to host memory and come back
/// when an operation needs them, with a one-time warning. [`Context::spilled_bytes`] counts it.
///
/// # Example
/// ```no_run
/// let ctx = luma::Context::new()?;
/// let a = luma::Array::from_slice_on(&ctx, &[1u32, 2, 3], 3)?;
/// let b = &a * 2;
/// assert_eq!(b.to_vec()?, vec![2, 4, 6]);
/// # Ok::<(), luma::LumaError>(())
/// ```
#[derive(Clone, Debug)]
pub struct Context {
    inner: Arc<Inner>,
}

impl Drop for Context {
    fn drop(&mut self) {
        // A schedule this thread left open holds handles of its own (and its segments do): when they are all that keeps the context
        // alive, nothing can ask for its steps any more, so they run now (trigger 8).
        if self.inner.schedules_open.load(std::sync::atomic::Ordering::Relaxed) != 0 {
            crate::jit::split::context_dropped(self);
        }
    }
}

static DEFAULT_CONTEXT: OnceLock<Result<Context>> = OnceLock::new();

/// The process-wide default [`Context`], opened on first use.
///
/// The result of the first attempt is kept: if no GPU could be opened, every call returns that
/// error instead of trying again. All constructors without an `_on` suffix
/// ([`Array::from_slice`](crate::Array::from_slice), [`Kernel::new`](crate::Kernel::new), ...)
/// use it.
pub fn default_context() -> Result<Context> {
    DEFAULT_CONTEXT.get_or_init(Context::new).clone()
}

/// [`Context::flush`] on the [default context](default_context): submits the pending batch
/// without waiting for it.
pub fn flush() -> Result<()> {
    default_context()?.flush()
}

/// How a [`Context`] is opened.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[allow(deprecated)]
#[non_exhaustive]
pub struct ContextOptions {
    /// The backend to open: CUDA, Metal or wgpu; `Auto` tries them in that order and takes the
    /// first that has a device.
    pub backend: BackendKind,
    /// The kind of device to open.
    pub device: DeviceKind,
    /// Bytes of array data kept on the device. `None`: 80% of the device's memory where the
    /// backend reports it (never on CPU devices), otherwise unlimited.
    pub memory_budget: Option<u64>,
    /// Bytes of host memory the context may use for host copies of device data: the segments it
    /// moved off the device plus the copies it keeps of segments that were brought back (see
    /// [`crate::PagingStats::host_copy_bytes`]). Kept copies are dropped first when it is exceeded.
    /// `None`: half of the available host memory at open, at most half of the cgroup limit.
    /// `LUMA_HOST_BUDGET`.
    pub host_budget: Option<u64>,
    /// Largest buffer an array is split into: a power of two from 1024 up to the largest
    /// bindable size. `None`: that size, or less when a memory budget is set.
    pub chunk_bytes: Option<u64>,
    /// Debug mode: every array an operation computes is checked for NaN and infinities right
    /// after its kernel, and the operation fails with [`LumaError::NonFinite`] if it holds one.
    /// The check synchronizes with the GPU once per evaluated array, so it is for finding the
    /// operation that first produces a non-finite value, not for production runs.
    pub check_finite: bool,
    /// Measure the device's throughputs when the context opens (about 150 ms on a GPU), once
    /// per machine: a clean result is kept in the cache directory (see `LUMA_CACHE`) and read by
    /// later processes. Each kernel is timed as the best of several runs, and a result worse
    /// than the profile's numbers by more than 4 times in any throughput (a device shared with
    /// other processes during the measurement) is measured once more and, if still so, dropped:
    /// the process then uses the profile's fixed numbers and nothing is stored. `false` uses
    /// the profile's fixed numbers (see [`CostModel`]).
    pub calibrate: bool,
    /// The numerics policy: which departures from exact `f32` semantics the context allows (see
    /// [`Numerics`]). The default has `matmul` run in TF32 on tensor cores where the device has
    /// them (compute capability 8.0 and up on CUDA; about `2^-11` of the magnitudes summed in
    /// accuracy, up to 1.7x on large products and several times on small ones);
    /// [`Numerics::strict`] makes every product `f32`, as does
    /// [`Array::matmul_f32`](crate::Array::matmul_f32) for one product.
    ///
    /// `Reduced::FAST_MATH` (in [`Numerics::fast`]) allows rewrites of expressions that are not
    /// exact: `x + 0` becomes `x` (a `-0.0` stays `-0.0` instead of becoming `0.0`), `x * 0`
    /// becomes `0` and `x - x` becomes `0` (wrong for NaN and infinity), `x / x` becomes `1`,
    /// `x <= x`, `x >= x`, `x == x` become true and `x != x` false (wrong for NaN), `(x * a) * b`
    /// and `(x + a) + b` combine their constants (different rounding), `x / a` for a constant `a`
    /// becomes `x * (1 / a)`, and `powf(x, e)` with `e` of 1, 2 or 0.5 becomes `x`, `x * x` or
    /// `sqrt(x)`. `Df32` additions and subtractions take the cheaper algorithm whose relative
    /// error grows with the cancellation (below 2^-47 of the larger operand instead of 2^-46 of
    /// the result), except in reductions. Rewrites that keep every bit (`x * 1`, `-(-x)`,
    /// `min(x, x)` and the like) always run. `Reduced::TF32_LINALG` runs the trailing-update
    /// products of the factorisations in TF32 (CUDA, compute capability 8.0 and up).
    /// `LUMA_NUMERICS` sets it from the environment (see [`Numerics::parse`]).
    pub numerics: Numerics,
    /// Allows the value-changing rewrites of expressions: set, it adds `Reduced::FAST_MATH` to
    /// [`ContextOptions::numerics`] when the context opens (`LUMA_FAST_MATH` still sets it,
    /// with a warning).
    #[deprecated(since = "0.5.2", note = "use `ContextOptions::numerics` with `Numerics::fast()` (`LUMA_NUMERICS=fast`); removed in 0.6")]
    pub fast_math: bool,
    /// Runs [`Array::matmul`](crate::Array::matmul) in TF32 on tensor cores where it can
    /// (default): `false` clears `Reduced::TF32_MATMUL` from [`ContextOptions::numerics`] when the
    /// context opens (`LUMA_TF32_MATMUL=0` still sets it, with a warning).
    #[deprecated(since = "0.5.2", note = "use `ContextOptions::numerics` with `Numerics::strict()` (`LUMA_NUMERICS=strict`); removed in 0.6")]
    pub tf32_matmul: bool,
    /// Whether the device's `fma` is a fused multiply-add, which makes the products of `Df32`
    /// arithmetic several times cheaper (see `Df32` in the crate docs). `None`: found out by a
    /// small test kernel when the device is first opened (`true` on Vulkan GPUs that fuse,
    /// `false` where `fma` is a multiplication and an addition, as on some software
    /// rasterizers); `Some(false)` forces the splitting everywhere, `Some(true)` trusts the
    /// caller. `LUMA_DF_FMA=on|off`. The results are the same either way.
    pub df_fma: Option<bool>,
    /// Forces strategies of the registry: `family:strategy[,family:strategy...]` (`reduce:two-pass`), as `LUMA_STRATEGY` says.
    /// A strategy that does not fit an operation is not used for it (the default selection runs, and the profile's decisions say
    /// so); an unknown family or strategy fails [`Context::with_options`] with [`LumaError::InvalidOption`]. For tests and
    /// investigations: it can change which kernels run, never what a strategy of the same bit class computes.
    #[doc(hidden)]
    pub strategy: Option<&'static str>,
    /// Defers kernels over arrays of several segments into schedules that run segment by segment (`LUMA_SPLIT` = `off` | `segment` |
    /// `auto`). Results are bit-identical to kernel-major launches. Removed before 0.6.0.
    #[doc(hidden)]
    pub split: SplitMode,
}

/// The value of the removed environment variable `name`, with a warning (once per variable) that
/// it is read until 0.6 and `replacement` is the way to say it.
fn deprecated_env(name: &'static str, replacement: &str) -> Option<String> {
    static WARNED: Mutex<Vec<&'static str>> = Mutex::new(Vec::new());
    let value = std::env::var(name).ok()?;
    let mut warned = lock(&WARNED);
    if !warned.contains(&name) {
        warned.push(name);
        log::warn!("luma: {name} is deprecated and read until luma 0.6; use {replacement}");
    }
    Some(value)
}

#[allow(deprecated)]
impl Default for ContextOptions {
    fn default() -> Self {
        ContextOptions {
            backend: BackendKind::default(),
            device: DeviceKind::default(),
            memory_budget: None,
            host_budget: None,
            chunk_bytes: None,
            check_finite: false,
            calibrate: true,
            numerics: Numerics::default(),
            fast_math: false,
            tf32_matmul: true,
            df_fma: None,
            strategy: None,
            split: SplitMode::Off,
        }
    }
}

impl ContextOptions {
    /// Turns calibration of the cost model on or off (see [`ContextOptions::calibrate`]).
    pub fn calibrate(mut self, on: bool) -> Self {
        self.calibrate = on;
        self
    }

    /// Forces the `Df32` library's products to use `fma` (`true`) or Dekker's splitting
    /// (`false`) instead of what the device's test says (see [`ContextOptions::df_fma`]).
    pub fn df_fma(mut self, fused: bool) -> Self {
        self.df_fma = Some(fused);
        self
    }

    /// Sets [`ContextOptions::split`].
    #[doc(hidden)]
    pub fn split(mut self, mode: SplitMode) -> Self {
        self.split = mode;
        self
    }

    /// Forces strategies of the registry (see [`ContextOptions::strategy`]).
    #[doc(hidden)]
    pub fn strategy(mut self, forced: &'static str) -> Self {
        self.strategy = Some(forced);
        self
    }

    /// Sets the backend to open.
    pub fn backend(mut self, kind: BackendKind) -> Self {
        self.backend = kind;
        self
    }

    /// Sets the kind of device to open.
    pub fn device(mut self, kind: DeviceKind) -> Self {
        self.device = kind;
        self
    }

    /// Sets the number of bytes of array data kept on the device.
    pub fn memory_budget(mut self, bytes: u64) -> Self {
        self.memory_budget = Some(bytes);
        self
    }

    /// Sets the bytes of host memory for host copies of device data (see
    /// [`ContextOptions::host_budget`]); `None` is the default.
    pub fn host_budget(mut self, bytes: Option<u64>) -> Self {
        self.host_budget = bytes;
        self
    }

    /// Sets the largest buffer an array is split into (a power of two from 1024 up to the
    /// largest bindable size; checked by [`Context::with_options`]).
    pub fn chunk_bytes(mut self, bytes: u64) -> Self {
        self.chunk_bytes = Some(bytes);
        self
    }

    /// Turns the finite check on or off (see [`ContextOptions::check_finite`]).
    pub fn check_finite(mut self, on: bool) -> Self {
        self.check_finite = on;
        self
    }

    /// Sets the numerics policy (see [`ContextOptions::numerics`]).
    pub fn numerics(mut self, numerics: Numerics) -> Self {
        self.numerics = numerics;
        self
    }

    /// Allows the value-changing rewrites of expressions (see [`Numerics::fast`]): the
    /// `fast_math` field is applied over [`ContextOptions::numerics`] when the context opens.
    #[deprecated(since = "0.5.2", note = "use `ContextOptions::numerics(Numerics::fast())` (or `.with_reduced(Reduced::FAST_MATH, true)`); removed in 0.6")]
    #[allow(deprecated)]
    pub fn fast_math(mut self, on: bool) -> Self {
        self.fast_math = on;
        self
    }

    /// Runs the matrix products on tensor cores in TF32 where the device has them (the default),
    /// or with `false` always in `f32`: the `tf32_matmul` field is applied over
    /// [`ContextOptions::numerics`] when the context opens (`false` clears
    /// `Reduced::TF32_MATMUL`; `true` changes nothing).
    #[deprecated(since = "0.5.2", note = "use `ContextOptions::numerics(Numerics::strict())` (or `.with_reduced(Reduced::TF32_MATMUL, false)`); removed in 0.6")]
    #[allow(deprecated)]
    pub fn tf32_matmul(mut self, on: bool) -> Self {
        self.tf32_matmul = on;
        self
    }

    /// The numerics policy a context opened with these options gets: [`ContextOptions::numerics`]
    /// with the deprecated `fast_math` and `tf32_matmul` fields applied.
    #[allow(deprecated)]
    pub(crate) fn effective_numerics(&self) -> Numerics {
        let mut numerics = self.numerics;
        if self.fast_math {
            // Before the class flags existed `fast_math` was the one switch of the algebraic rewrites too.
            numerics = numerics.with_reduced(Reduced::FAST_MATH, true);
            numerics.exact_finite = true;
            numerics.reassociate = true;
            numerics.reformulate = true;
        }
        if !self.tf32_matmul {
            numerics = numerics.with_reduced(Reduced::TF32_MATMUL, false);
        }
        numerics
    }

    /// Options from the environment: `LUMA_BACKEND` = `cuda` | `metal` | `wgpu` | `cpu` | `auto`;
    /// `LUMA_DEVICE` = `gpu` | `cpu` (anything else: `Auto`), which also takes a backend name
    /// (`cuda` | `metal` | `wgpu`) when `LUMA_BACKEND` is not set;
    /// `LUMA_MEMORY_BUDGET` and `LUMA_CHUNK_BYTES` as decimal byte counts, `LUMA_CHECK_FINITE` =
    /// `1` | `on` | `true` for the finite check, `LUMA_NUMERICS` =
    /// `strict` | `default` | `algebraic` | `fast` with modifiers (see [`Numerics::parse`]) for
    /// [`ContextOptions::numerics`], `LUMA_CALIBRATE` = `0` |
    /// `off` | `false` to skip calibration.
    ///
    /// # Panics
    ///
    /// If `LUMA_NUMERICS` is set to something [`Numerics::parse`] rejects.
    #[allow(deprecated)]
    pub fn from_env() -> ContextOptions {
        let var = |name: &str| std::env::var(name).ok().map(|v| v.trim().to_ascii_lowercase());
        let backend_named = |name: &str| match name {
            "cuda" => Some(BackendKind::Cuda),
            "metal" => Some(BackendKind::Metal),
            "wgpu" => Some(BackendKind::Wgpu),
            "cpu" => Some(BackendKind::Cpu),
            _ => None,
        };
        let device_var = var("LUMA_DEVICE");
        let device = match device_var.as_deref() {
            Some("gpu") => DeviceKind::Gpu,
            Some("cpu") => DeviceKind::Cpu,
            _ => DeviceKind::Auto,
        };
        let backend = match var("LUMA_BACKEND").as_deref() {
            Some("auto") => BackendKind::Auto,
            Some(name) => backend_named(name).unwrap_or_default(),
            None => match device_var.as_deref() {
                Some("cpu") => BackendKind::Auto,
                Some(name) => backend_named(name).unwrap_or_default(),
                None => BackendKind::Auto,
            },
        };
        let bytes = |name: &str| std::env::var(name).ok().and_then(|v| v.trim().parse::<u64>().ok());
        ContextOptions {
            backend,
            device,
            memory_budget: bytes("LUMA_MEMORY_BUDGET"),
            host_budget: bytes("LUMA_HOST_BUDGET"),
            chunk_bytes: bytes("LUMA_CHUNK_BYTES"),
            check_finite: std::env::var("LUMA_CHECK_FINITE")
                .is_ok_and(|v| matches!(v.trim().to_ascii_lowercase().as_str(), "1" | "on" | "true")),
            calibrate: !std::env::var("LUMA_CALIBRATE")
                .is_ok_and(|v| matches!(v.trim().to_ascii_lowercase().as_str(), "0" | "off" | "false")),
            numerics: {
                let numerics = match std::env::var("LUMA_NUMERICS") {
                    Ok(text) => Numerics::parse(&text).unwrap_or_else(|why| panic!("LUMA_NUMERICS={text:?}: {why}")),
                    Err(_) => Numerics::default(),
                };
                // The 0.5 name of `+tf32-linalg`, read until 0.6.
                if deprecated_env("LUMA_LINALG_TF32", "LUMA_NUMERICS=...+tf32-linalg").is_some_and(|v| v.trim() == "1") {
                    numerics.with_reduced(Reduced::TF32_LINALG, true)
                } else {
                    numerics
                }
            },
            fast_math: deprecated_env("LUMA_FAST_MATH", "LUMA_NUMERICS=fast (or +fast-math)")
                .is_some_and(|v| matches!(v.trim().to_ascii_lowercase().as_str(), "1" | "on" | "true")),
            tf32_matmul: !deprecated_env("LUMA_TF32_MATMUL", "LUMA_NUMERICS=strict (or -tf32-matmul)")
                .is_some_and(|v| matches!(v.trim().to_ascii_lowercase().as_str(), "0" | "off" | "false")),
            df_fma: std::env::var("LUMA_DF_FMA").ok().and_then(|v| match v.trim().to_ascii_lowercase().as_str() {
                "1" | "on" | "true" => Some(true),
                "0" | "off" | "false" => Some(false),
                _ => None,
            }),
            strategy: std::env::var("LUMA_STRATEGY").ok().filter(|text| !text.trim().is_empty()).map(intern),
            split: std::env::var("LUMA_SPLIT").ok().and_then(|text| SplitMode::parse(&text)).unwrap_or_default(),
        }
    }
}

/// `text` as a `&'static str`, leaked once per distinct text (the options are `Copy`).
pub(crate) fn intern(text: String) -> &'static str {
    static KNOWN: Mutex<Vec<&'static str>> = Mutex::new(Vec::new());
    let mut known = lock(&KNOWN);
    if let Some(&found) = known.iter().find(|&&known| known == text) {
        return found;
    }
    let leaked: &'static str = Box::leak(text.into_boxed_str());
    known.push(leaked);
    leaked
}

/// Which device a [`Context`] runs on, from [`Context::device`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeviceSummary {
    /// The backend the context runs on.
    pub backend: BackendKind,
    /// The device's name.
    pub name: String,
    /// Whether the device is the host CPU (the `cpu` backend) rather than a GPU.
    pub software: bool,
}

/// The largest power of two not above `x` (1 for `x` of 0 or 1): the segment size that fits a
/// buffer limit or a fraction of the memory budget.
fn pow2_floor(x: u64) -> u64 {
    1 << (63 - x.max(1).leading_zeros())
}

/// The error of a backend this build has no code for.
#[allow(dead_code)]
fn not_built(feature: &str) -> LumaError {
    LumaError::NoDevice(match feature {
        "cuda" => "luma was built without the `cuda` feature",
        "metal" => "luma was built without the `metal` feature",
        _ => "luma was built without the `wgpu` feature",
    })
}

/// Why a CPU device cannot be had from a GPU backend.
const CPU_NEEDS_CPU_BACKEND: &str = "a CPU device needs the `cpu` backend; GPU backends never run on software adapters";

#[cfg(feature = "cuda")]
fn open_cuda(device: DeviceKind) -> Result<Box<dyn Backend>> {
    if device == DeviceKind::Cpu {
        return Err(LumaError::NoDevice(CPU_NEEDS_CPU_BACKEND));
    }
    Ok(luma_cuda::open()?)
}

#[cfg(not(feature = "cuda"))]
fn open_cuda(_: DeviceKind) -> Result<Box<dyn Backend>> {
    Err(not_built("cuda"))
}

#[cfg(feature = "metal")]
fn open_metal(device: DeviceKind) -> Result<Box<dyn Backend>> {
    if device == DeviceKind::Cpu {
        return Err(LumaError::NoDevice(CPU_NEEDS_CPU_BACKEND));
    }
    Ok(luma_metal::open()?)
}

#[cfg(not(feature = "metal"))]
fn open_metal(_: DeviceKind) -> Result<Box<dyn Backend>> {
    Err(not_built("metal"))
}

#[cfg(feature = "wgpu")]
fn open_wgpu(device: DeviceKind) -> Result<Box<dyn Backend>> {
    if device == DeviceKind::Cpu {
        return Err(LumaError::NoDevice(CPU_NEEDS_CPU_BACKEND));
    }
    Ok(luma_wgpu::open(device)?)
}

#[cfg(not(feature = "wgpu"))]
fn open_wgpu(device: DeviceKind) -> Result<Box<dyn Backend>> {
    if device == DeviceKind::Cpu {
        return Err(LumaError::NoDevice(CPU_NEEDS_CPU_BACKEND));
    }
    Err(not_built("wgpu"))
}

fn open_cpu(device: DeviceKind) -> Result<Box<dyn Backend>> {
    if device == DeviceKind::Gpu {
        return Err(LumaError::NoDevice("the cpu backend only provides a software (CPU) device"));
    }
    Ok(luma_cpu::open()?)
}

#[cfg(test)]
std::thread_local! {
    /// The workgroups per grid dimension the next context this thread opens allows at most (see [`Context::with_workgroup_limit`]).
    static WORKGROUP_LIMIT: std::cell::Cell<Option<u32>> = const { std::cell::Cell::new(None) };
}

/// The most workgroups per grid dimension the context allows: the device's, or in a test the limit it asked for.
#[inline(always)]
fn workgroup_limit(device: u32) -> u32 {
    #[cfg(test)]
    if let Some(max) = WORKGROUP_LIMIT.with(std::cell::Cell::get) {
        return device.min(max);
    }
    device
}

/// Opens the backend `backend` names. `Auto` tries CUDA, then Metal, then wgpu for GPU devices
/// and falls back to the native CPU backend when needed. A CPU device is the native CPU backend only.
fn open_backend(backend: BackendKind, device: DeviceKind) -> Result<Box<dyn Backend>> {
    match backend {
        BackendKind::Cuda => open_cuda(device),
        BackendKind::Metal => open_metal(device),
        BackendKind::Wgpu => open_wgpu(device),
        BackendKind::Cpu => open_cpu(device),
        BackendKind::Auto => {
            if device == DeviceKind::Cpu {
                return open_cpu(DeviceKind::Cpu);
            }
            type Open = fn(DeviceKind) -> Result<Box<dyn Backend>>;
            for (name, open) in [("cuda", open_cuda as Open), ("metal", open_metal as Open), ("wgpu", open_wgpu as Open)] {
                let req = if name == "wgpu" && device == DeviceKind::Auto { DeviceKind::Gpu } else { device };
                match open(req) {
                    Ok(backend) => return Ok(backend),
                    Err(error) => log::debug!("luma: the {name} backend is not available: {error}"),
                }
            }
            if device == DeviceKind::Auto {
                return open_cpu(DeviceKind::Cpu);
            }
            open_wgpu(device)
        }
        _ => Err(LumaError::Unsupported("unknown backend kind".into())),
    }
}

/// Whether the `fma` of `backend`'s device is fused: the answer of an earlier process from the
/// disk cache, else the result of the test kernel (kept for later processes). A test that could
/// not run says `false` and is not kept. Known once per device and process.
fn device_fma(backend: &dyn Backend) -> bool {
    static KNOWN: Mutex<Option<FxMap<String, bool>>> = Mutex::new(None);
    let disk = backend.disk_cache();
    if let Some(&known) = lock(&KNOWN).as_ref().and_then(|known| known.get(disk.key())) {
        return known;
    }
    let found = disk.load_fma().or_else(|| {
        let fused = crate::gpu::fma::probe(backend)?;
        disk.store_fma(fused);
        Some(fused)
    });
    if let Some(fused) = found {
        lock(&KNOWN).get_or_insert_with(FxMap::default).insert(disk.key().to_owned(), fused);
    }
    found.unwrap_or(false)
}

impl Context {
    /// Opens the device chosen by `LUMA_BACKEND` (`cuda` | `metal` | `wgpu` | `cpu` | `auto`) and
    /// `LUMA_DEVICE` (`gpu` | `cpu`; by default the first available of CUDA, Metal and the
    /// high-performance wgpu GPU, or the native CPU backend when no GPU exists). Blocks until the
    /// device is ready.
    pub fn new() -> Result<Context> {
        Context::with_options(ContextOptions::from_env())
    }

    /// Opens the native CPU backend (`cpu`), whatever GPUs exist and whatever `LUMA_BACKEND` says.
    pub fn cpu() -> Result<Context> {
        Context::with_options(ContextOptions { backend: BackendKind::Cpu, device: DeviceKind::Cpu, ..ContextOptions::from_env() })
    }

    /// Opens a device of the CUDA backend (NVIDIA GPUs).
    pub fn cuda() -> Result<Context> {
        Context::with_options(ContextOptions { backend: BackendKind::Cuda, ..ContextOptions::from_env() })
    }

    /// Opens a device of the Metal backend (Apple GPUs).
    pub fn metal() -> Result<Context> {
        Context::with_options(ContextOptions { backend: BackendKind::Metal, ..ContextOptions::from_env() })
    }

    /// Opens a device of the wgpu backend (Vulkan, Metal, DX12 through WebGPU).
    pub fn wgpu() -> Result<Context> {
        Context::with_options(ContextOptions { backend: BackendKind::Wgpu, ..ContextOptions::from_env() })
    }

    /// The backend this context runs on.
    pub fn backend(&self) -> BackendKind {
        self.inner.backend.info().backend
    }

    /// Which device this context runs on: its backend, name and whether it is a software (CPU) device.
    pub fn device(&self) -> DeviceSummary {
        let info = self.inner.backend.info();
        DeviceSummary { backend: info.backend, name: info.name.clone(), software: info.cpu }
    }

    /// The device's name (the GPU's marketing name, or a description of the CPU for the `cpu` backend).
    pub fn device_name(&self) -> &str {
        &self.inner.backend.info().name
    }

    /// The numerics policy of this context (see [`ContextOptions::numerics`]); fixed when it opens.
    pub fn numerics(&self) -> Numerics {
        self.inner.numerics
    }

    /// Whether [`Array::matmul`](crate::Array::matmul) runs on tensor cores in TF32 where it can.
    #[deprecated(since = "0.5.2", note = "use `Context::numerics().tf32_matmul()`; removed in 0.6")]
    pub fn tf32_matmul(&self) -> bool {
        self.inner.numerics.tf32_matmul()
    }

    /// What the device can do.
    pub(crate) fn caps(&self) -> Caps {
        self.inner.caps
    }

    /// The strategy `LUMA_STRATEGY` forces for `family`, if any.
    #[inline]
    pub(crate) fn strategy_override(&self, family: FamilyTag) -> Option<StrategyId> {
        self.inner.strategy_override.of(family)
    }

    /// The custom ops of this context.
    pub(crate) fn ext_registry(&self) -> &crate::ext::Registry {
        &self.inner.ext
    }

    /// Whether value-changing algebraic rewrites are allowed.
    pub(crate) fn fast_math(&self) -> bool {
        self.inner.numerics.fast_math()
    }

    /// Whether the numerics allow a rule of a flagged pass of `simplify` (anything but the exact rules).
    #[inline(always)]
    pub(crate) fn rules_flagged(&self) -> bool {
        let n = &self.inner.numerics;
        n.exact_finite || n.reassociate || n.reformulate
    }

    /// The numerics policy, by reference: what the hot paths read.
    #[inline(always)]
    pub(crate) fn numerics_ref(&self) -> &Numerics {
        &self.inner.numerics
    }

    /// Whether this context runs on a software (CPU) device.
    pub fn is_cpu(&self) -> bool {
        self.inner.backend.info().cpu
    }

    /// Opens a device as `options` say, without looking at the environment. Blocks until the
    /// device is ready.
    pub fn with_options(options: ContextOptions) -> Result<Context> {
        Context::open(options, |_| {}, Caps::empty())
    }

    /// [`Context::with_options`] with `tune` changing the kernel parameters of the device (tests
    /// run kernels the device would not choose).
    #[cfg(test)]
    pub(crate) fn with_tuning(options: ContextOptions, tune: impl FnOnce(&mut Tuning)) -> Result<Context> {
        Context::open(options, tune, Caps::empty())
    }

    /// [`Context::with_tuning`] on a device that allows at most `max` workgroups per grid dimension (tests of the resource errors that
    /// depend on it, which no real device can be made to give with arrays that fit a test).
    #[cfg(test)]
    pub(crate) fn with_workgroup_limit(options: ContextOptions, max: u32, tune: impl FnOnce(&mut Tuning)) -> Result<Context> {
        WORKGROUP_LIMIT.with(|limit| limit.set(Some(max)));
        let opened = Context::open(options, tune, Caps::empty());
        WORKGROUP_LIMIT.with(|limit| limit.set(None));
        opened
    }

    /// [`Context::with_options`] reporting the device without the capabilities of `removed`.
    #[cfg(test)]
    pub(crate) fn with_caps_removed(options: ContextOptions, removed: Caps) -> Result<Context> {
        Context::open(options, |_| {}, removed)
    }

    fn open(options: ContextOptions, tune: impl FnOnce(&mut Tuning), removed: Caps) -> Result<Context> {
        let backend = open_backend(options.backend, options.device)?;
        let info = backend.info();
        let limits = info.limits;
        let max_buffer_bytes = limits.max_buffer_bytes;
        let limit = pow2_floor(max_buffer_bytes & !3); // largest power of two <= the bindable size
        // 80% of the device's memory, except on CPU devices, whose "device memory" is host RAM.
        let budget = options.memory_budget.or(if info.cpu { None } else { info.memory_bytes.map(|m| m / 5 * 4) });
        let segment_bytes = match options.chunk_bytes {
            Some(b) if b.is_power_of_two() && (1024..=limit).contains(&b) => b,
            Some(b) => {
                return Err(LumaError::InvalidShape(format!(
                    "chunk_bytes must be a power of two between 1024 and {limit}, got {b}"
                )));
            }
            None => budget.map_or(limit, |b| pow2_floor(b / 8).clamp(1024, limit)),
        };
        // Idle buffer caches scale with the budget (half of it): they are given back when the device
        // runs out of memory, so they cannot crowd out array data. A program that cycles through
        // gigabytes of temporaries (a batch of large arrays per call) reuses them instead of
        // allocating them again, which costs about as much as writing them ten times over.
        let cap = budget.map_or(1 << 30, |b| b / 2);

        // The test of the device's `fma` runs only where the backend does not know the answer
        // and no earlier process stored it.
        let mut tuning = Tuning::for_device(info);
        tuning.df_fma = match options.df_fma {
            Some(fused) => fused,
            None => match info.hardware_fma {
                Some(fused) => fused,
                None => device_fma(&*backend),
            },
        };
        tuning.df_sloppy_add = options.effective_numerics().fast_math();
        tuning.native_solve = Tuning::native_solve_for(backend.native_line_solve());
        tuning.native_gemm = Tuning::native_gemm_for(backend.native_matmul());
        tune(&mut tuning);
        let caps = Caps::of(info, &*backend, &tuning).without(removed);
        let strategy_override = match options.strategy {
            Some(text) => StrategyOverride::parse(text)?,
            None => StrategyOverride::default(),
        };

        let ctx = Context {
            inner: Arc::new(Inner {
                max_uniform_bytes: limits.max_uniform_bytes,
                max_workgroups_per_dimension: workgroup_limit(limits.max_workgroups_per_dim),
                timestamps: info.features.contains(Features::TIMESTAMP_QUERY),
                zero_copy: info.zero_copy,
                tuning,
                cost: OnceLock::new(),
                memory: OnceLock::new(),
                reduce_counter: OnceLock::new(),
                builtins: BuildCache::default(),
                chirps: Mutex::new(std::collections::HashMap::new()),
                jit: JitState::new(&limits, &options.effective_numerics()),
                kernels: BuildCache::default(),
                module_kernels: Mutex::default(),
                keyed_kernels: Mutex::default(),
                pool: Mutex::new(BufferPool::new(cap)),
                released: Mutex::new(Vec::new()),
                segment_bytes,
                check_finite: options.check_finite,
                split: options.split,
                schedules_open: std::sync::atomic::AtomicU32::new(0),
                schedule_lease: Mutex::new(()),
                streams: StreamCounters::default(),
                numerics: options.effective_numerics(),
                caps,
                strategy_override,
                ext: crate::ext::Registry::new(options.strategy),
                pager: Pager::new(budget, options.host_budget.unwrap_or_else(crate::gpu::pager::default_host_budget)),
                id: record::next_context_id(),
                waits: WaitHistory::new(),
                stages: Mutex::new(Vec::new()),
                assume_busy: std::sync::atomic::AtomicBool::new(false),
                assume_idle: std::sync::atomic::AtomicBool::new(false),
                #[cfg(test)]
                created_bytes: std::sync::atomic::AtomicU64::new(0),
                max_buffer_bytes,
                backend,
            }),
        };
        let profile = ctx.inner.tuning.profile;
        let model = if options.calibrate {
            *ctx.inner.backend.device_cost().get_or_init(|| {
                // Measured once per machine: a stored result for this device and driver wins,
                // and only a measurement that does not look contended is stored (see
                // `cost::contended`); otherwise this process runs on the profile's numbers, and so
                // do the processes of the next day (see `DiskCache::recently_contended`).
                let disk = ctx.disk_cache();
                let fallback = CostModel::fallback(profile);
                match disk.load_calibration().filter(|(stored, _)| !cost::contended(stored, &fallback)) {
                    Some((stored, memory)) => {
                        disk.set_memory_model(memory);
                        stored
                    }
                    None => {
                        if disk.recently_contended() {
                            return fallback;
                        }
                        match cost::calibrate_clean(&ctx, &fallback) {
                            Ok(Some((measured, memory))) => {
                                disk.store_calibration(&measured, memory.as_ref());
                                disk.set_memory_model(memory);
                                measured
                            }
                            Ok(None) => {
                                disk.note_contended();
                                fallback
                            }
                            Err(_) => fallback,
                        }
                    }
                }
            })
        } else {
            CostModel::fallback(profile)
        };
        let _ = ctx.inner.cost.set(model);
        let _ = ctx.inner.memory.set(if options.calibrate { ctx.disk_cache().memory_model() } else { None });
        Ok(ctx)
    }

    /// How bytes move inside the device (see [`MemoryModel`]): measured by the calibration that
    /// measured the cost model (see [`ContextOptions::calibrate`]) and stored with it; `None`
    /// where calibration is off, failed or looked contended, and where the device has no
    /// numbers. Nothing plans with it yet.
    pub fn memory_model(&self) -> Option<MemoryModel> {
        self.inner.memory.get().copied().flatten()
    }

    /// The device's cost model: measured when the context opened (see
    /// [`ContextOptions::calibrate`]), otherwise the fixed numbers of its tuning profile.
    pub fn cost_model(&self) -> CostModel {
        #[cfg(test)]
        if let Some(model) = self.jit().cost_override() {
            return model;
        }
        self.inner.cost.get().copied().unwrap_or_else(|| CostModel::fallback(self.inner.tuning.profile))
    }

    /// Replaces the cost model (tests).
    #[cfg(test)]
    pub(crate) fn set_cost_model(&self, model: CostModel) {
        self.jit().set_cost_override(model);
    }

    /// The cost model cell of the device, shared by every context on it (tests).
    #[cfg(test)]
    pub(crate) fn device_cost(&self) -> Arc<OnceLock<CostModel>> {
        Arc::clone(self.inner.backend.device_cost())
    }

    /// Starts building a custom compute kernel from WGSL source.
    ///
    /// See [`KernelBuilder`] for the options and [`Kernel`](crate::Kernel) for the binding
    /// conventions.
    pub fn kernel(&self, wgsl_src: &str) -> KernelBuilder<'_> {
        KernelBuilder::new(self, wgsl_src)
    }

    /// Submits everything recorded so far and waits until the GPU has executed it. Reports the
    /// first error the GPU raised while running luma's commands.
    pub fn sync(&self) -> Result<()> {
        self.abort_recording_for("sync")?;
        self.flush()?;
        let waited = self.hal().sync(self.inner.waits.budget())?;
        self.inner.waits.record(waited);
        self.jit().telemetry.drain_pending(self);
        Ok(())
    }

    pub(crate) fn jit(&self) -> &JitState {
        &self.inner.jit
    }

    /// Switches the plan cache of eager evaluation on or off (it is on unless
    /// `LUMA_PLAN_CACHE=off`), dropping the plans it holds. Results are bit-identical either way.
    #[doc(hidden)]
    pub fn set_plan_cache(&self, on: bool) {
        self.jit().set_plan_cache(on);
    }

    /// Switches the recording of the planner's decisions on or off (also on at open with
    /// `LUMA_RECORD_DECISIONS=1`). It sets nothing else: host paths, deferred axis reductions, the
    /// plan cache and graph capture stay as they are, unlike [`Context::enable_profiling`], which
    /// records decisions too. Read them with [`Context::take_decisions`]. Switching off drops
    /// the decisions not yet taken.
    pub fn record_decisions(&self, on: bool) {
        if on {
            self.jit().start_decisions();
        } else {
            drop(self.jit().stop_decisions());
        }
    }

    /// Returns the decisions recorded since the last call (or since recording began) and keeps
    /// recording if it is on.
    pub fn take_decisions(&self) -> Vec<Decision> {
        self.jit().drain_decisions()
    }

    /// How many evaluations ran from a cached plan so far.
    #[doc(hidden)]
    pub fn plan_cache_hits(&self) -> u64 {
        self.jit().plan_stats().hits
    }

    /// Submits the pending batch to the GPU and returns without waiting for it, so the GPU
    /// starts working while the host keeps recording. Everything recorded afterwards runs after
    /// it, in order. Does nothing if nothing is pending.
    ///
    /// Lazy arrays that are still alive count as recorded work: `flush` evaluates them first (each
    /// as one fused kernel, so a chain is only cut at the flush point), which makes the GPU start
    /// on elementwise expressions too. Expressions whose arrays were already dropped are never
    /// evaluated, nor are arrays already computed inside another node's fused kernel (usually
    /// the temporaries of an expression). It only evaluates the arrays the calling thread created:
    /// those of other threads using the context at the same time are theirs to evaluate (reading an
    /// array always does).
    ///
    /// Only [`Context::sync`] and the operations that read results back
    /// ([`Array::to_vec`](crate::Array::to_vec), [`Array::item`](crate::Array::item))
    /// wait, and only they report an error the GPU raises while running the batch.
    /// A typical time-stepping loop records a few steps, calls `flush`, records the next few
    /// while the GPU runs the first, and reads results back once in a while:
    ///
    /// ```no_run
    /// # let ctx = luma::Context::new()?;
    /// let mut x = luma::Array::<f32>::zeros_on(&ctx, 1 << 20)?;
    /// for _ in 0..100 {
    ///     for _ in 0..10 {
    ///         x += 1.0;
    ///     }
    ///     ctx.flush()?; // the GPU runs this chunk while the next one is recorded
    /// }
    /// println!("{}", x.sum().item()?);
    /// # Ok::<(), luma::LumaError>(())
    /// ```
    pub fn flush(&self) -> Result<()> {
        self.jit().evaluate_requested();
        self.jit().evaluate_live();
        self.commit_schedules()?;
        self.flush_batch()
    }

    /// The backend the context runs on.
    pub(crate) fn hal(&self) -> &dyn Backend {
        &*self.inner.backend
    }

    /// What the backend knows about the device: limits, features, name.
    pub(crate) fn info(&self) -> &DeviceInfo {
        self.inner.backend.info()
    }

    /// The on-disk cache of the device (compiled kernels, calibration).
    pub(crate) fn disk_cache(&self) -> &luma_core::DiskCache {
        self.inner.backend.disk_cache()
    }

    /// Bytes of a full segment of an array: the largest buffer an array is split into.
    pub(crate) fn segment_bytes(&self) -> u64 {
        self.inner.segment_bytes
    }

    /// Whether the finite check is on (see [`ContextOptions::check_finite`]).
    pub(crate) fn check_finite(&self) -> bool {
        self.inner.check_finite
    }

    /// Whether small jobs on host-resident data may run on the host CPU: a GPU context whose
    /// device memory the host addresses directly ([`Context::zero_copy`]) that is not recording,
    /// profiling or checking results, with `LUMA_HOST_FAST` not `off`.
    ///
    /// A discrete GPU (CUDA, Vulkan) is left out: its arrays live in device buffers from the
    /// moment they are created, which keeps the memory budget, [`Context::memory_stats`] and
    /// spilling exact, and the traces of `iterate` replayable. Arrays that stayed in host memory
    /// until a kernel needed them would sit outside all of that, and uploading one while a step
    /// is being traced ends the recording.
    pub(crate) fn can_run_on_host(&self) -> bool {
        #[cfg(test)]
        if self.inner.assume_busy.load(std::sync::atomic::Ordering::Relaxed) || std::env::var("LUMA_HOST_FAST").as_deref() != Ok("on") {
            return false;
        }
        !self.is_cpu()
            && self.zero_copy()
            && !self.recording_here()
            && !self.check_finite()
            && self.segment_bytes() >= 65536
            && !self.is_profiling()
            && std::env::var("LUMA_HOST_FAST").as_deref() != Ok("off")
    }

    /// Largest buffer that can be allocated and bound as a storage buffer.
    pub(crate) fn max_buffer_bytes(&self) -> u64 {
        self.inner.max_buffer_bytes
    }

    /// Whether kernels over streamed arrays are deferred (see [`ContextOptions::split`]).
    #[inline(always)]
    pub(crate) fn split_mode(&self) -> SplitMode {
        self.inner.split
    }

    pub(crate) fn schedules_open(&self) -> &std::sync::atomic::AtomicU32 {
        &self.inner.schedules_open
    }

    pub(crate) fn schedule_lease(&self) -> &Mutex<()> {
        &self.inner.schedule_lease
    }

    pub(crate) fn stream_counters(&self) -> &StreamCounters {
        &self.inner.streams
    }

    /// Handles to the context's state: this one, the segments' and a schedule's.
    pub(crate) fn handles(&self) -> usize {
        Arc::strong_count(&self.inner)
    }

    /// Tells this context from the others of the process.
    pub(crate) fn id(&self) -> u64 {
        self.inner.id
    }

    /// What schedules did on this context (see `LUMA_SPLIT`).
    #[doc(hidden)]
    pub fn stream_stats(&self) -> StreamStats {
        self.inner.streams.snapshot()
    }

    /// Which segments of arrays currently hold a device buffer.
    pub(crate) fn pager(&self) -> &Pager {
        &self.inner.pager
    }

    pub(crate) fn max_workgroups_per_dimension(&self) -> u32 {
        self.inner.max_workgroups_per_dimension
    }

    pub(crate) fn max_uniform_bytes(&self) -> u64 {
        self.inner.max_uniform_bytes
    }

    /// `true` if both handles refer to the same device.
    pub(crate) fn same_as(&self, other: &Context) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }

    /// The kernel `entry` of the WGSL `source()` of a library module (`name`, one of `variant`s),
    /// built on the first call: later calls find it by the short key without generating, cloning
    /// or hashing the source, which for a source of kilobytes is a few microseconds of every launch.
    pub(crate) fn cached_kernel(&self, name: &'static str, variant: usize, entry: &'static str, source: impl FnOnce() -> String) -> Result<crate::Kernel> {
        let key = (name, variant, entry);
        let cached = self.inner.module_kernels.lock().unwrap_or_else(std::sync::PoisonError::into_inner).get(&key).cloned();
        if let Some(core) = cached {
            return Ok(crate::Kernel::from_core(self, core));
        }
        let kernel = self.kernel(&source()).entry(entry).build()?;
        self.inner.module_kernels.lock().unwrap_or_else(std::sync::PoisonError::into_inner).insert(key, kernel.core());
        Ok(kernel)
    }

    /// The kernel of source `make()` that `key` names (the key must determine the source), built
    /// on the first call with `label`: later calls find it by the key. The cache belongs to this
    /// context and holds the compiled kernels only, never a handle to a context, so it dies with
    /// it and a finished context gives its device back to the pool.
    pub(crate) fn keyed_kernel(&self, key: &str, label: &str, make: impl FnOnce() -> String) -> Result<crate::Kernel> {
        self.keyed_kernel_with(key, make, |source| self.kernel(source).label(label).build())
    }

    /// [`Context::keyed_kernel`] for a kernel whose every access is in bounds by construction: built
    /// without the per-access bounds checks.
    ///
    /// # Safety
    ///
    /// The caller vouches that every storage and workgroup access of the kernel is in bounds.
    pub(crate) unsafe fn keyed_kernel_unchecked(&self, key: &str, label: &str, make: impl FnOnce() -> String) -> Result<crate::Kernel> {
        // SAFETY: the caller vouches for the bounds of every access of the kernel.
        self.keyed_kernel_with(key, make, |source| unsafe { self.kernel(source).label(label).unchecked_bounds() }.build())
    }

    fn keyed_kernel_with(&self, key: &str, make: impl FnOnce() -> String, build: impl FnOnce(&str) -> Result<crate::Kernel>) -> Result<crate::Kernel> {
        let cached = self.inner.keyed_kernels.lock().unwrap_or_else(std::sync::PoisonError::into_inner).get(key).cloned();
        if let Some(core) = cached {
            return Ok(crate::Kernel::from_core(self, core));
        }
        let kernel = build(&make())?;
        self.inner.keyed_kernels.lock().unwrap_or_else(std::sync::PoisonError::into_inner).insert(key.to_owned(), kernel.core());
        Ok(kernel)
    }

    /// The cache of custom kernels (see [`KernelBuilder::build`]).
    pub(crate) fn kernel_cache(&self) -> &BuildCache<KernelKey, Arc<KernelCore>> {
        &self.inner.kernels
    }

    /// The cached built-in kernel for `op` on elements of WGSL type `ty`, compiled on first use.
    pub(crate) fn builtin(&self, op: Builtin, ty: &'static str) -> Result<Arc<KernelCore>> {
        self.inner
            .builtins
            .get_or_build(&(op, ty), || {
                let label = format!("{op:?} {ty}");
                match op.kernel(ty, &self.inner.tuning) {
                    BuiltinKernel::Module(module) => kernel::build_module(self, module, Some(&label), KernelTrust::Trusted),
                    BuiltinKernel::Wgsl(source) => {
                        let native = match op {
                            Builtin::Matmul { tile: crate::matmul::MatmulTile::Native { edge, lhs_t, rhs_t }, .. } => {
                                Some(kernel::Native::Matmul(MatmulSpec { edge, lhs_transposed: lhs_t, rhs_transposed: rhs_t, half: false }))
                            }
                            Builtin::Matmul { tile: crate::matmul::MatmulTile::NativeHalf { edge, lhs_t, rhs_t }, .. } => {
                                Some(kernel::Native::Matmul(MatmulSpec { edge, lhs_transposed: lhs_t, rhs_transposed: rhs_t, half: true }))
                            }
                            Builtin::Matmul { tile: crate::matmul::MatmulTile::GemmHalf { rows, cols, lhs_t, rhs_t, exact, lanes }, .. } => {
                                Some(kernel::Native::Gemm(luma_core::GemmSpec::new(rows, cols, lhs_t, rhs_t, exact).with_lanes(lanes).with_half()))
                            }
                            Builtin::Matmul { tile: crate::matmul::MatmulTile::Gemm { rows, cols, lhs_t, rhs_t, exact }, .. } => {
                                Some(kernel::Native::Gemm(luma_core::GemmSpec::new(rows, cols, lhs_t, rhs_t, exact)))
                            }
                            Builtin::Matmul { tile: crate::matmul::MatmulTile::GemmRagged { rows, cols, lhs_t, rhs_t, exact, lanes }, .. } => {
                                Some(kernel::Native::Gemm(luma_core::GemmSpec::new(rows, cols, lhs_t, rhs_t, exact).with_lanes(lanes)))
                            }
                            Builtin::Bmm(_) if self.inner.tuning.native_gemm > 0 => Some(kernel::Native::BatchedMatmul),
                            _ => op.line_solve().filter(|_| self.inner.tuning.native_solve).map(kernel::Native::LineSolve),
                        };
                        kernel::build_core_native(self, &KernelKey::new(source.into_owned(), "main", None), Some(&label), KernelTrust::Trusted, native)
                    }
                }
                .map(Arc::new)
            })
            .map(|(core, _)| core)
    }

    /// Takes a buffer of at least `bytes` bytes with exactly `usage` from the pool, creating one
    /// if none is idle. The buffer's size is rounded up to a power of two and its contents are
    /// unspecified. Give it back with [`Context::recycle`].
    pub(crate) fn alloc_buffer(&self, bytes: u64, usage: BufferUsage) -> Result<Buffer> {
        if let Some(buffer) = self.take_recorded(bytes, usage) {
            return Ok(buffer);
        }
        let max = self.inner.max_buffer_bytes;
        let class = size_class(bytes, max)
            .ok_or(LumaError::BufferTooLarge { requested: bytes, max })?;

        let pooled = lock(&self.inner.pool).take(class, usage);
        let buffer = match pooled {
            Some(buffer) => buffer,
            None => self.new_buffer(class, usage, false)?,
        };
        self.note_fresh(&buffer);
        Ok(buffer)
    }

    /// A buffer for `bytes` bytes with `usage` that this thread's recording is done with (see
    /// `Recording::reuse`), if it is recording and has one: no new memory is needed for it.
    pub(crate) fn take_recorded(&self, bytes: u64, usage: BufferUsage) -> Option<Buffer> {
        let class = size_class(bytes, self.inner.max_buffer_bytes)?;
        self.reuse_recorded(class, usage)
    }

    /// Runs `attempt` (`None`: the device is out of memory) and, when the device is full, gives
    /// back every idle buffer and the caches pinning them, then tries again; when it is still
    /// full, moves array data to host memory, one segment at a time, until it fits.
    pub(crate) fn relieving<T>(&self, requested: u64, mut attempt: impl FnMut() -> Result<Option<T>>) -> Result<T> {
        if let Some(done) = attempt()? {
            return Ok(done);
        }
        self.reclaim_idle_buffers()?;
        if let Some(done) = attempt()? {
            return Ok(done);
        }
        while self.evict_one()? {
            self.reclaim_idle_buffers()?;
            if let Some(done) = attempt()? {
                return Ok(done);
            }
        }
        Err(LumaError::OutOfMemory { requested })
    }

    /// A new buffer; when the device is out of memory, gives back every idle buffer and the bind
    /// groups pinning them, then tries once more.
    fn new_buffer(&self, size: u64, usage: BufferUsage, mapped: bool) -> Result<Buffer> {
        #[cfg(test)]
        self.inner.created_bytes.fetch_add(size, std::sync::atomic::Ordering::Relaxed);
        self.relieving(size, || Ok(self.hal().create_buffer(size, usage, mapped)))
    }

    /// Tests: bytes of device buffers created so far.
    #[cfg(test)]
    pub(crate) fn created_bytes(&self) -> u64 {
        self.inner.created_bytes.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Releases the idle buffers the context keeps for reuse (see [`MemoryStats::pooled_bytes`]),
    /// after the work recorded so far, and has the backend drop what it caches for them. Array
    /// data and recorded plans keep their memory. The pool fills again as arrays are dropped;
    /// call this after a phase whose temporaries the next one cannot reuse (another grid
    /// shape, say) to give the memory back to the device.
    ///
    /// [`MemoryStats::pooled_bytes`]: crate::MemoryStats::pooled_bytes
    pub fn trim(&self) -> Result<()> {
        luma_core::hal::trim_host_pool();
        self.inner.stages.lock().unwrap_or_else(std::sync::PoisonError::into_inner).clear();
        self.reclaim_idle_buffers()
    }

    /// Bytes of the idle buffers in the pool.
    pub(crate) fn pooled_bytes(&self) -> u64 {
        lock(&self.inner.pool).pooled_bytes()
    }

    /// Gives back every idle buffer: queues them for release, has the backend submit and wait
    /// for the device and drop what it caches for them (and its own idle buffers).
    fn reclaim_idle_buffers(&self) -> Result<()> {
        let idle = lock(&self.inner.pool).drain();
        lock(&self.inner.released).extend(idle);
        let released = std::mem::take(&mut *lock(&self.inner.released));
        self.hal().notify_released(&released);
        // Dropped before the backend reclaims: the memory they free is part of what it gives
        // back (the CUDA pool returns to the driver only what was freed by then).
        drop(released);
        Ok(self.hal().reclaim_idle()?)
    }

    /// Returns a buffer from [`Context::alloc_buffer`] to the pool.
    ///
    /// This is safe while recorded or submitted GPU work still uses the buffer. All of it is
    /// part of the one ordered command stream, and whoever takes the buffer next (to upload into
    /// it, to run a kernel on it) records that after everything recorded so far. Nothing writes
    /// to pooled buffers outside the stream.
    pub(crate) fn recycle(&self, buffer: Buffer) {
        let Some(buffer) = self.retire(buffer) else { return };
        if buffer.usage().contains(BufferUsage::MAP_WRITE) {
            // May still be used by the unsubmitted batch, which mapping would invalidate: the
            // backend makes it writable again after the next submission. This must not take
            // anything the caller may hold.
            self.hal().recycle_mappable(buffer);
            return;
        }
        let released = lock(&self.inner.pool).put(buffer);
        if !released.is_empty() {
            lock(&self.inner.released).extend(released);
        }
    }

    /// The kernel parameter set chosen for this context's GPU (see [`TuningProfile`]).
    pub fn tuning_profile(&self) -> TuningProfile {
        self.inner.tuning.profile
    }

    /// The cache of Bluestein tables.
    pub(crate) fn chirps(&self) -> &Mutex<std::collections::HashMap<(u32, bool), Arc<crate::spectral::ChirpTables>>> {
        &self.inner.chirps
    }

    pub(crate) fn tuning(&self) -> &Tuning {
        &self.inner.tuning
    }

    /// The counter of the reductions that finish in their first launch: a zeroed word that every
    /// such launch leaves zero. One serves the context, as its launches run in order.
    pub(crate) fn reduce_counter(&self) -> Result<&Buffer> {
        if let Some(counter) = self.inner.reduce_counter.get() {
            return Ok(counter);
        }
        let made = self.alloc_filled(&[0u8; 4])?;
        let bytes = made.size();
        if let Err(lost) = self.inner.reduce_counter.set(made) {
            // Another thread made it first; this one is only recycled.
            self.recycle(lost);
        } else {
            self.pager().count_for_life(bytes);
        }
        Ok(self.inner.reduce_counter.get().expect("the counter was set above or by the thread that won"))
    }

    /// The features the device was opened with. Custom kernels may use the shader features
    /// among them (`SUBGROUP`, `SHADER_F16`, `SHADER_F64`, `EXPERIMENTAL_COOPERATIVE_MATRIX`) when
    /// present. `SHADER_F64` is also what `f64` arrays need.
    pub fn features(&self) -> Features {
        self.info().features
    }

    // -- Profiling --

    /// Whether the GPU supports the timestamp queries [`Context::profile`] needs.
    pub fn profiling_supported(&self) -> bool {
        self.inner.timestamps
    }

    /// Measures the GPU time of every kernel launch recorded from now on, until
    /// [`Context::finish_profiling`]. Does nothing if profiling is already enabled.
    ///
    /// Fails with [`LumaError::Profiling`] if the device has no timestamp queries (see
    /// [`Context::profiling_supported`]).
    pub fn enable_profiling(&self) -> Result<()> {
        if !self.inner.timestamps {
            return Err(LumaError::Profiling(
                "the GPU does not support timestamp queries (feature TIMESTAMP_QUERY)".into(),
            ));
        }
        if !self.hal().is_profiling()? {
            let enabled = self.hal().enable_profiling();
            // `can_run_on_host` reads whether the device is profiling.
            self.jit().selcache.bump();
            enabled?;
            self.jit().start_decisions();
        }
        Ok(())
    }

    /// Stops profiling and returns the GPU durations of the launches measured since
    /// [`Context::enable_profiling`], by kernel label. Submits the pending batch and waits for
    /// it, so this is also a sync point.
    ///
    /// [`Profile::cuts`] lists where the planner cut expressions into several kernels. Every
    /// decision the planner made in the profiled region (cuts, inlining, staging, plans) is kept
    /// for the next [`Context::take_decisions`], which returns them.
    ///
    /// Fails with [`LumaError::Profiling`] if profiling is not enabled.
    #[allow(deprecated)]
    pub fn finish_profiling(&self) -> Result<Profile> {
        self.abort_recording_for("sync")?;
        if !self.hal().is_profiling()? {
            return Err(LumaError::Profiling("profiling is not enabled".into()));
        }
        self.jit().follow_report();
        // Lazy arrays still alive were computed inside the profiled region; record their fused
        // kernels while the profiler is still on.
        self.jit().evaluate_live();
        let decisions = self.jit().stop_decisions();
        self.notify_released();
        let splits = decisions
            .iter()
            .filter_map(|d| match d.site {
                DecisionSite::Cut(reason) => Some(luma_core::FusionSplit { op: d.op, location: d.location, reason }),
                _ => None,
            })
            .collect();
        let finished = self.hal().finish_profiling(splits);
        self.jit().selcache.bump();
        let profile = finished?;
        self.jit().park_decisions(decisions);
        Ok(profile)
    }

    /// Runs `f` and measures the GPU time of every kernel launch recorded during it (and, since
    /// the batch is waited for at the end, of everything that was still pending), by kernel
    /// label.
    ///
    /// Needs the GPU's timestamp queries; fails with [`LumaError::Profiling`] otherwise (see
    /// [`Context::profiling_supported`]). While profiling, every launch runs in a compute pass of
    /// its own, which costs host time and lets the GPU overlap less; use it to find out where
    /// the GPU time goes, not to measure total throughput. Built-in operations are measured
    /// too, under labels like `Binary(Add, Arrays) f32`; custom kernels under the label given to
    /// [`KernelBuilder::label`](crate::KernelBuilder::label) (default: the entry point name).
    ///
    /// ```no_run
    /// # let ctx = luma::Context::new()?;
    /// let x = luma::Array::from_slice_on(&ctx, &[1.0f32; 1 << 20], 1 << 20)?;
    /// let (sum, profile) = ctx.profile(|| (&x * 2.0).sum().item())?;
    /// println!("sum = {}\n{profile}", sum?);
    /// # Ok::<(), luma::LumaError>(())
    /// ```
    pub fn profile<R>(&self, f: impl FnOnce() -> R) -> Result<(R, Profile)> {
        self.enable_profiling()?;
        let result = f();
        let profile = self.finish_profiling()?;
        Ok((result, profile))
    }
}

/// Temporary buffers of one multi-pass operation (FFT work space, partial reductions). Taken
/// from the context's pool and handed back when this guard drops, right after the operation's
/// commands were recorded. That is safe while they have not run: every later use of such a buffer
/// is recorded after them (see [`Context::recycle`]).
pub(crate) struct Scratch<'a> {
    ctx: &'a Context,
    held: Vec<Buffer>,
    /// Counts the bytes of `held` as resident until the buffers are given back (when `Scratch`
    /// drops, after `Drop::drop` has recycled them: a recording then counts them itself).
    counted: Accounted,
}

impl<'a> Scratch<'a> {
    pub(crate) fn new(ctx: &'a Context) -> Self {
        Scratch { ctx, held: Vec::new(), counted: Accounted::new(ctx, 0) }
    }

    /// A buffer of at least `bytes` bytes with unspecified contents.
    pub(crate) fn take(&mut self, bytes: u64) -> Result<Buffer> {
        let buffer = self.ctx.alloc_buffer(bytes, STORAGE_USAGE)?;
        self.counted.add(buffer.size());
        self.held.push(buffer.clone());
        Ok(buffer)
    }
}

impl Drop for Scratch<'_> {
    fn drop(&mut self) {
        for buffer in self.held.drain(..) {
            self.ctx.recycle(buffer);
        }
    }
}
