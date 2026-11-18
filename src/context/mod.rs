use std::sync::{Arc, Mutex, OnceLock};

use luma_core::{Backend, BackendKind, Buffer, BufferUsage, CostModel, DeviceInfo, DeviceKind, Features, Profile};

use crate::builtin::{Builtin, BuiltinKernel};
use crate::error::{LumaError, Result};
use crate::jit::cost;
use crate::jit::JitState;
use crate::gpu::pager::Pager;
use crate::gpu::built::BuildCache;
use crate::gpu::hash::FxMap;
use crate::gpu::lock;
use crate::gpu::pool::{self, BufferPool};
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

#[derive(Debug)]
struct Inner {
    /// Built-in kernels, built on first use of their (operation, element type).
    builtins: BuildCache<(Builtin, &'static str), Arc<KernelCore>>,
    /// Lazy-evaluation state: fused kernels and the pending expressions.
    jit: JitState,
    /// Custom kernels, built on first use of their (source, entry, workgroup).
    kernels: BuildCache<KernelKey, Arc<KernelCore>>,
    pool: Mutex<BufferPool>,
    /// Buffers the pool released; the backend drops what it caches for them (bind groups) at the
    /// next submission, which frees them.
    released: Mutex<Vec<Buffer>>,
    /// Bytes of a full segment of an array, see [`Storage`](crate::array::Storage).
    segment_bytes: u64,
    /// Whether results are checked for NaN and Inf (see [`ContextOptions::check_finite`]).
    check_finite: bool,
    /// Whether value-changing algebraic rewrites are allowed (see [`ContextOptions::fast_math`]).
    fast_math: bool,
    /// Which segments of arrays hold a device buffer.
    pager: Pager,
    /// Whether `Array::from_slice` uploads into mappable storage buffers (unified memory).
    zero_copy: bool,
    /// Kernel parameters for this GPU.
    tuning: Tuning,
    /// What launches, memory traffic and arithmetic cost on this GPU, set once the context exists.
    cost: OnceLock<CostModel>,
    /// Tells this context from the others of the process (the recordings of a thread are kept by it, see [`record`]).
    id: u64,
    /// How long recent waits took, which sets how long the next one polls before it blocks.
    waits: WaitHistory,
    /// Tests: report the GPU as busy regardless of the queue (see `Context::set_assume_busy`).
    #[cfg(test)]
    assume_busy: std::sync::atomic::AtomicBool,
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
    /// Allows rewrites of expressions that are not exact: `x + 0` becomes `x` (a `-0.0`
    /// stays `-0.0` instead of becoming `0.0`), `x * 0` becomes `0` and `x - x` becomes `0` (wrong
    /// for NaN and infinity), `x / x` becomes `1`, `x <= x`, `x >= x`, `x == x` become true and
    /// `x != x` false (wrong for NaN), `(x * a) * b` and `(x + a) + b` combine their constants
    /// (different rounding), `x / a` for a constant `a` becomes `x * (1 / a)`, and `powf(x, e)`
    /// with `e` of 1, 2 or 0.5 becomes `x`, `x * x` or `sqrt(x)`. `Df32` additions and
    /// subtractions take the cheaper algorithm whose relative error grows with the cancellation
    /// (below 2^-47 of the larger operand instead of 2^-46 of the result), except in reductions.
    /// Rewrites that keep every bit (`x * 1`, `-(-x)`, `min(x, x)` and the like) always run.
    pub fast_math: bool,
    /// Whether the device's `fma` is a fused multiply-add, which makes the products of `Df32`
    /// arithmetic several times cheaper (see `Df32` in the crate docs). `None`: found out by a
    /// small test kernel when the device is first opened (`true` on Vulkan GPUs that fuse,
    /// `false` where `fma` is a multiplication and an addition, as on some software
    /// rasterizers); `Some(false)` forces the splitting everywhere, `Some(true)` trusts the
    /// caller. `LUMA_DF_FMA=on|off`. The results are the same either way.
    pub df_fma: Option<bool>,
}

impl Default for ContextOptions {
    fn default() -> Self {
        ContextOptions {
            backend: BackendKind::default(),
            device: DeviceKind::default(),
            memory_budget: None,
            chunk_bytes: None,
            check_finite: false,
            calibrate: true,
            fast_math: false,
            df_fma: None,
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

    /// Allows the value-changing rewrites of expressions (see [`ContextOptions::fast_math`]).
    pub fn fast_math(mut self, on: bool) -> Self {
        self.fast_math = on;
        self
    }

    /// Options from the environment: `LUMA_BACKEND` = `cuda` | `metal` | `wgpu` | `auto`;
    /// `LUMA_DEVICE` = `gpu` | `cpu` (anything else: `Auto`), which also takes a backend name
    /// (`cuda` | `metal` | `wgpu`) when `LUMA_BACKEND` is not set;
    /// `LUMA_MEMORY_BUDGET` and `LUMA_CHUNK_BYTES` as decimal byte counts, `LUMA_CHECK_FINITE` =
    /// `1` | `on` | `true` for the finite check, `LUMA_FAST_MATH` likewise for
    /// [`ContextOptions::fast_math`], `LUMA_CALIBRATE` = `0` | `off` | `false` to skip
    /// calibration.
    pub fn from_env() -> ContextOptions {
        let var = |name: &str| std::env::var(name).ok().map(|v| v.trim().to_ascii_lowercase());
        let backend_named = |name: &str| match name {
            "cuda" => Some(BackendKind::Cuda),
            "metal" => Some(BackendKind::Metal),
            "wgpu" => Some(BackendKind::Wgpu),
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
            None => device_var.as_deref().and_then(backend_named).unwrap_or_default(),
        };
        let bytes = |name: &str| std::env::var(name).ok().and_then(|v| v.trim().parse::<u64>().ok());
        ContextOptions {
            backend,
            device,
            memory_budget: bytes("LUMA_MEMORY_BUDGET"),
            chunk_bytes: bytes("LUMA_CHUNK_BYTES"),
            check_finite: std::env::var("LUMA_CHECK_FINITE")
                .is_ok_and(|v| matches!(v.trim().to_ascii_lowercase().as_str(), "1" | "on" | "true")),
            calibrate: !std::env::var("LUMA_CALIBRATE")
                .is_ok_and(|v| matches!(v.trim().to_ascii_lowercase().as_str(), "0" | "off" | "false")),
            fast_math: std::env::var("LUMA_FAST_MATH")
                .is_ok_and(|v| matches!(v.trim().to_ascii_lowercase().as_str(), "1" | "on" | "true")),
            df_fma: std::env::var("LUMA_DF_FMA").ok().and_then(|v| match v.trim().to_ascii_lowercase().as_str() {
                "1" | "on" | "true" => Some(true),
                "0" | "off" | "false" => Some(false),
                _ => None,
            }),
        }
    }
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

/// Why a CPU device cannot be had from a backend other than wgpu.
#[cfg(any(feature = "cuda", feature = "metal", not(feature = "wgpu")))]
const CPU_NEEDS_WGPU: &str = "CPU fallback requires the wgpu backend";

#[cfg(feature = "cuda")]
fn open_cuda(device: DeviceKind) -> Result<Box<dyn Backend>> {
    if device == DeviceKind::Cpu {
        return Err(LumaError::NoDevice(CPU_NEEDS_WGPU));
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
        return Err(LumaError::NoDevice(CPU_NEEDS_WGPU));
    }
    Ok(luma_metal::open()?)
}

#[cfg(not(feature = "metal"))]
fn open_metal(_: DeviceKind) -> Result<Box<dyn Backend>> {
    Err(not_built("metal"))
}

#[cfg(feature = "wgpu")]
fn open_wgpu(device: DeviceKind) -> Result<Box<dyn Backend>> {
    Ok(luma_wgpu::open(device)?)
}

#[cfg(not(feature = "wgpu"))]
fn open_wgpu(device: DeviceKind) -> Result<Box<dyn Backend>> {
    if device == DeviceKind::Cpu {
        return Err(LumaError::NoDevice(CPU_NEEDS_WGPU));
    }
    Err(not_built("wgpu"))
}

/// Opens the backend `backend` names. `Auto` tries CUDA, then Metal, then wgpu and takes the first
/// that opens (a CPU device only exists through wgpu).
fn open_backend(backend: BackendKind, device: DeviceKind) -> Result<Box<dyn Backend>> {
    match backend {
        BackendKind::Cuda => open_cuda(device),
        BackendKind::Metal => open_metal(device),
        BackendKind::Wgpu => open_wgpu(device),
        BackendKind::Auto => {
            if device == DeviceKind::Cpu {
                return open_wgpu(device);
            }
            type Open = fn(DeviceKind) -> Result<Box<dyn Backend>>;
            for (name, open) in [("cuda", open_cuda as Open), ("metal", open_metal as Open)] {
                match open(device) {
                    Ok(backend) => return Ok(backend),
                    Err(error) => log::debug!("luma: the {name} backend is not available: {error}"),
                }
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
    /// Opens the device chosen by `LUMA_BACKEND` (`cuda` | `metal` | `wgpu` | `auto`) and
    /// `LUMA_DEVICE` (`gpu` | `cpu`; by default the first available of CUDA, Metal and the
    /// high-performance wgpu GPU, or the CPU through a software driver such as Mesa llvmpipe when
    /// no GPU exists). Blocks until the device is ready.
    pub fn new() -> Result<Context> {
        Context::with_options(ContextOptions::from_env())
    }

    /// Opens a software (CPU) device, for example Mesa llvmpipe, whatever GPUs exist. Only the
    /// wgpu backend has one.
    pub fn cpu() -> Result<Context> {
        Context::with_options(ContextOptions { backend: BackendKind::Wgpu, device: DeviceKind::Cpu, ..ContextOptions::from_env() })
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

    /// Whether value-changing algebraic rewrites are allowed.
    pub(crate) fn fast_math(&self) -> bool {
        self.inner.fast_math
    }

    /// Whether this context runs on a software (CPU) device.
    pub fn is_cpu(&self) -> bool {
        self.inner.backend.info().cpu
    }

    /// Opens a device as `options` say, without looking at the environment. Blocks until the
    /// device is ready.
    pub fn with_options(options: ContextOptions) -> Result<Context> {
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
        tuning.df_sloppy_add = options.fast_math;

        let ctx = Context {
            inner: Arc::new(Inner {
                max_uniform_bytes: limits.max_uniform_bytes,
                max_workgroups_per_dimension: limits.max_workgroups_per_dim,
                timestamps: info.features.contains(Features::TIMESTAMP_QUERY),
                zero_copy: info.zero_copy,
                tuning,
                cost: OnceLock::new(),
                builtins: BuildCache::default(),
                jit: JitState::new(&limits),
                kernels: BuildCache::default(),
                pool: Mutex::new(BufferPool::new(cap)),
                released: Mutex::new(Vec::new()),
                segment_bytes,
                check_finite: options.check_finite,
                fast_math: options.fast_math,
                pager: Pager::new(budget),
                id: record::next_context_id(),
                waits: WaitHistory::new(),
                #[cfg(test)]
                assume_busy: std::sync::atomic::AtomicBool::new(false),
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
                disk.load_cost().filter(|stored| !cost::contended(stored, &fallback)).unwrap_or_else(|| {
                    if disk.recently_contended() {
                        return fallback;
                    }
                    match cost::calibrate_clean(&ctx, &fallback) {
                        Ok(Some(measured)) => {
                            disk.store_cost(&measured);
                            measured
                        }
                        Ok(None) => {
                            disk.note_contended();
                            fallback
                        }
                        Err(_) => fallback,
                    }
                })
            })
        } else {
            CostModel::fallback(profile)
        };
        let _ = ctx.inner.cost.set(model);
        Ok(ctx)
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
        self.abort_recording()?;
        self.flush()?;
        let waited = self.hal().sync(self.inner.waits.budget())?;
        self.inner.waits.record(waited);
        Ok(())
    }

    pub(crate) fn jit(&self) -> &JitState {
        &self.inner.jit
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

    /// Largest buffer that can be allocated and bound as a storage buffer.
    pub(crate) fn max_buffer_bytes(&self) -> u64 {
        self.inner.max_buffer_bytes
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
                    BuiltinKernel::Module(module) => kernel::build_module(self, module, Some(&label), true),
                    BuiltinKernel::Wgsl(source) => {
                        kernel::build_core(self, &KernelKey::new(source.into_owned(), "main", None), Some(&label), true)
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
        let max = self.inner.max_buffer_bytes;
        let class = pool::size_class(bytes, max)
            .ok_or(LumaError::BufferTooLarge { requested: bytes, max })?;

        if let Some(buffer) = lock(&self.inner.pool).take(class, usage) {
            return Ok(buffer);
        }
        self.new_buffer(class, usage, false)
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
        self.relieving(size, || Ok(self.hal().create_buffer(size, usage, mapped)))
    }

    /// Gives back every idle buffer: queues them for release, has the backend submit and wait
    /// for the device and drop what it caches for them (and its own idle buffers).
    fn reclaim_idle_buffers(&self) -> Result<()> {
        let idle = lock(&self.inner.pool).drain();
        lock(&self.inner.released).extend(idle);
        let released = std::mem::take(&mut *lock(&self.inner.released));
        self.hal().notify_released(&released);
        let reclaimed = self.hal().reclaim_idle();
        drop(released);
        Ok(reclaimed?)
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

    pub(crate) fn tuning(&self) -> &Tuning {
        &self.inner.tuning
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
            self.hal().enable_profiling()?;
            self.jit().start_splits();
        }
        Ok(())
    }

    /// Stops profiling and returns the GPU durations of the launches measured since
    /// [`Context::enable_profiling`], by kernel label. Submits the pending batch and waits for
    /// it, so this is also a sync point.
    ///
    /// Fails with [`LumaError::Profiling`] if profiling is not enabled.
    pub fn finish_profiling(&self) -> Result<Profile> {
        self.abort_recording()?;
        if !self.hal().is_profiling()? {
            return Err(LumaError::Profiling("profiling is not enabled".into()));
        }
        // Lazy arrays still alive were computed inside the profiled region; record their fused
        // kernels while the profiler is still on.
        self.jit().evaluate_live();
        let splits = self.jit().take_split_notes();
        self.notify_released();
        Ok(self.hal().finish_profiling(splits)?)
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
}

impl<'a> Scratch<'a> {
    pub(crate) fn new(ctx: &'a Context) -> Self {
        Scratch { ctx, held: Vec::new() }
    }

    /// A buffer of at least `bytes` bytes with unspecified contents.
    pub(crate) fn take(&mut self, bytes: u64) -> Result<Buffer> {
        let buffer = self.ctx.alloc_buffer(bytes, STORAGE_USAGE)?;
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
