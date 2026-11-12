use std::hash::Hash;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use wgpu::{Buffer, BufferUsages};

use crate::builtin::Builtin;
use crate::error::{LumaError, Result};
use crate::jit::cost::{self, CostModel};
use crate::jit::JitState;
use crate::gpu::pager::Pager;
use crate::gpu::handle;
use crate::gpu::built::BuildCache;
use crate::gpu::lock;
use crate::gpu::pool::{self, BufferPool};
use crate::kernel::{self, KernelBuilder, KernelCore, KernelKey};
use crate::profile::{Profile, Profiler};
use crate::tuning::{Tuning, TuningProfile};

mod batch;
mod record;
mod transfer;
mod uniforms;
#[cfg(test)]
mod tests;

use batch::{Batch, BindGroupCache, Bound};
pub(crate) use record::{Cmd, CmdArg, Recording};
pub(crate) use batch::RawArg;
use transfer::{FreeUploads, WaitHistory};
use uniforms::Arena;
pub(crate) use transfer::copy_parallel;

/// Usage of every array storage buffer (and the scratch buffers of multi-pass operations, so they
/// share a pool class).
pub(crate) const STORAGE_USAGE: BufferUsages = BufferUsages::STORAGE
    .union(BufferUsages::COPY_DST)
    .union(BufferUsages::COPY_SRC);

/// Usage of the buffers uniform values are written into.
const UNIFORM_USAGE: BufferUsages = BufferUsages::UNIFORM.union(BufferUsages::COPY_DST);

/// Usage of storage buffers the host writes directly (zero-copy uploads).
const MAPPABLE_STORAGE_USAGE: BufferUsages = STORAGE_USAGE.union(BufferUsages::MAP_WRITE);

#[derive(Debug)]
struct Inner {
    /// Built-in kernels, built on first use of their (operation, element type).
    builtins: BuildCache<(Builtin, &'static str), Arc<KernelCore>>,
    /// Lazy-evaluation state: fused kernels and the pending expressions.
    jit: JitState,
    /// Custom kernels, built on first use of their (source, entry, workgroup).
    kernels: BuildCache<KernelKey, Arc<KernelCore>>,
    pool: Mutex<BufferPool>,
    /// Buffers the pool released; the cached bind groups still holding them are dropped at the
    /// next submission, which frees them.
    released: Arc<Mutex<Vec<Buffer>>>,
    /// Mapped upload staging buffers, refilled by the map callbacks of submitted ones.
    uploads: Arc<Mutex<FreeUploads>>,
    mappable: Arc<Mutex<FreeUploads>>,
    /// Bytes of a full segment of an array, see [`Storage`](crate::array::Storage).
    segment_bytes: u64,
    /// Whether results are checked for NaN and Inf (see [`ContextOptions::check_finite`]).
    check_finite: bool,
    /// Whether value-changing algebraic rewrites are allowed (see [`ContextOptions::fast_math`]).
    fast_math: bool,
    /// Which segments of arrays hold a device buffer.
    pager: Pager,
    /// Recycled mappable storage buffers waiting for the next submission before they are mapped.
    unmapped: Mutex<Vec<Buffer>>,
    /// Whether `Array::from_slice` uploads into mappable storage buffers (unified memory).
    zero_copy: bool,
    /// Kernel parameters for this GPU.
    tuning: Tuning,
    /// What launches, memory traffic and arithmetic cost on this GPU, set once the context exists.
    cost: OnceLock<CostModel>,
    batch: Mutex<Batch>,
    /// `Batch::dispatches`, kept where threads that only decide whether to submit can read it
    /// without the lock of the batch.
    dispatches_pending: std::sync::atomic::AtomicU32,
    /// Tells this context from the others of the process (the recordings of a thread are kept by it, see [`record`]).
    id: u64,
    /// How long recent waits took, which sets how long the next one polls before it blocks.
    waits: WaitHistory,
    /// The first error the GPU reported outside of a call that could return it.
    gpu_error: Arc<Mutex<Option<String>>>,
    /// Tests: report the GPU as busy regardless of the queue (see `Context::set_assume_busy`).
    #[cfg(test)]
    assume_busy: std::sync::atomic::AtomicBool,
    /// Layout of `@group(0)`: the grid uniform, at a dynamic offset into the arena.
    grid_layout: wgpu::BindGroupLayout,
    /// Largest buffer that can be allocated *and* bound as a storage buffer.
    max_buffer_bytes: u64,
    max_uniform_bytes: u64,
    uniform_align: u64,
    max_workgroups_per_dimension: u32,
    max_dynamic_uniforms: u32,
    /// Whether the device was opened with timestamp queries (see [`Context::profile`]).
    timestamps: bool,
    /// The leased GPU device. Must stay the last field: fields drop in declaration order, and the
    /// lease hands the device to the next context, which must only happen once everything of
    /// this one (open pass and encoder, buffer pools, bind groups, kernels) is gone.
    gpu: Lease,
}

/// A [`Gpu`](handle::Gpu) borrowed from the process-wide pool; returned to it when dropped.
/// The device is never destroyed, see [`handle`].
#[derive(Debug)]
struct Lease(handle::Gpu);

impl Drop for Lease {
    fn drop(&mut self) {
        handle::release(self.0.clone());
    }
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

/// Which kind of device a [`Context`] runs on.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum DeviceKind {
    /// A GPU if one opens, otherwise the CPU through a software driver. Resolved once per process.
    #[default]
    Auto,
    /// A hardware GPU only; [`LumaError::NoDevice`] if only a software adapter exists.
    Gpu,
    /// A software (CPU) adapter such as Mesa llvmpipe.
    Cpu,
}

/// How a [`Context`] is opened.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct ContextOptions {
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
    /// Measure the device's throughputs when the context opens (about 100 ms on a GPU, once per
    /// device and process); `false` uses the profile's fixed numbers (see [`CostModel`]).
    pub calibrate: bool,
    /// Allows rewrites of expressions that are not exact: `x + 0` becomes `x` (a `-0.0`
    /// stays `-0.0` instead of becoming `0.0`), `x * 0` becomes `0` and `x - x` becomes `0` (wrong
    /// for NaN and infinity), `x / x` becomes `1`, `x <= x`, `x >= x`, `x == x` become true and
    /// `x != x` false (wrong for NaN), `(x * a) * b` and `(x + a) + b` combine their constants
    /// (different rounding), `x / a` for a constant `a` becomes `x * (1 / a)`, and `powf(x, e)`
    /// with `e` of 1, 2 or 0.5 becomes `x`, `x * x` or `sqrt(x)`. Rewrites that keep every bit
    /// (`x * 1`, `-(-x)`, `min(x, x)` and the like) always run.
    pub fast_math: bool,
}

impl Default for ContextOptions {
    fn default() -> Self {
        ContextOptions {
            device: DeviceKind::default(),
            memory_budget: None,
            chunk_bytes: None,
            check_finite: false,
            calibrate: true,
            fast_math: false,
        }
    }
}

impl ContextOptions {
    /// Turns calibration of the cost model on or off (see [`ContextOptions::calibrate`]).
    pub fn calibrate(mut self, on: bool) -> Self {
        self.calibrate = on;
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

    /// Options from the environment: `LUMA_DEVICE` = `gpu` | `cpu` (anything else: `Auto`),
    /// `LUMA_MEMORY_BUDGET` and `LUMA_CHUNK_BYTES` as decimal byte counts, `LUMA_CHECK_FINITE` =
    /// `1` | `on` | `true` for the finite check, `LUMA_FAST_MATH` likewise for
    /// [`ContextOptions::fast_math`], `LUMA_CALIBRATE` = `0` | `off` | `false` to skip
    /// calibration.
    pub fn from_env() -> ContextOptions {
        let device = match std::env::var("LUMA_DEVICE").ok().as_deref().map(str::trim) {
            Some("gpu") => DeviceKind::Gpu,
            Some("cpu") => DeviceKind::Cpu,
            _ => DeviceKind::Auto,
        };
        let bytes = |name: &str| std::env::var(name).ok().and_then(|v| v.trim().parse::<u64>().ok());
        ContextOptions {
            device,
            memory_budget: bytes("LUMA_MEMORY_BUDGET"),
            chunk_bytes: bytes("LUMA_CHUNK_BYTES"),
            check_finite: std::env::var("LUMA_CHECK_FINITE")
                .is_ok_and(|v| matches!(v.trim().to_ascii_lowercase().as_str(), "1" | "on" | "true")),
            calibrate: !std::env::var("LUMA_CALIBRATE")
                .is_ok_and(|v| matches!(v.trim().to_ascii_lowercase().as_str(), "0" | "off" | "false")),
            fast_math: std::env::var("LUMA_FAST_MATH")
                .is_ok_and(|v| matches!(v.trim().to_ascii_lowercase().as_str(), "1" | "on" | "true")),
        }
    }
}

/// The largest power of two not above `x` (1 for `x` of 0 or 1): the segment size that fits a
/// buffer limit or a fraction of the memory budget.
fn pow2_floor(x: u64) -> u64 {
    1 << (63 - x.max(1).leading_zeros())
}

impl Context {
    /// Opens the device chosen by `LUMA_DEVICE` (`gpu` | `cpu`; by default the high-performance
    /// GPU, or the CPU through a software driver such as Mesa llvmpipe when no GPU exists).
    /// Blocks until the device is ready.
    pub fn new() -> Result<Context> {
        Context::with_options(ContextOptions::from_env())
    }

    /// Opens a software (CPU) device, for example Mesa llvmpipe, whatever GPUs exist.
    pub fn cpu() -> Result<Context> {
        Context::with_options(ContextOptions { device: DeviceKind::Cpu, ..ContextOptions::from_env() })
    }

    /// Whether value-changing algebraic rewrites are allowed.
    pub(crate) fn fast_math(&self) -> bool {
        self.inner.fast_math
    }

    /// Whether this context runs on a software (CPU) device.
    pub fn is_cpu(&self) -> bool {
        self.inner.gpu.0.cpu
    }

    /// Opens a device as `options` say, without looking at the environment. Blocks until the
    /// device is ready.
    pub fn with_options(options: ContextOptions) -> Result<Context> {
        // Leased at once: an early `Err` below must return the device to the pool, not destroy it.
        let gpu = Lease(handle::acquire(options.device)?);
        let device = gpu.0.device.clone();
        let limits = device.limits();
        let max_buffer_bytes = limits
            .max_buffer_size
            .min(limits.max_storage_buffer_binding_size);
        let limit = pow2_floor(max_buffer_bytes & !3); // largest power of two <= the bindable size
        // 80% of the device's memory, except on CPU devices, whose "device memory" is host RAM.
        let budget = options.memory_budget.or(if gpu.0.cpu { None } else { gpu.0.memory.map(|m| m / 5 * 4) });
        let segment_bytes = match options.chunk_bytes {
            Some(b) if b.is_power_of_two() && (1024..=limit).contains(&b) => b,
            Some(b) => {
                return Err(LumaError::InvalidShape(format!(
                    "chunk_bytes must be a power of two between 1024 and {limit}, got {b}"
                )));
            }
            None => budget.map_or(limit, |b| pow2_floor(b / 8).clamp(1024, limit)),
        };
        // Idle buffer caches scale with the budget so that they cannot crowd out array data.
        let cap = budget.map_or(1 << 30, |b| (b / 8).min(1 << 30));

        // wgpu panics on errors nobody handles. Record the first one instead and report it as an
        // `Err` at the next synchronization point.
        let gpu_error = Arc::new(Mutex::new(None));
        let sink = Arc::clone(&gpu_error);
        // Registered for every context: it replaces the previous owner's handler, so a reused device reports errors to its current context only.
        device.on_uncaptured_error(Arc::new(move |error: wgpu::Error| {
            lock(&sink).get_or_insert_with(|| error.to_string());
        }));

        let grid_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("luma grid"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let batch = Batch {
            pass: None,
            encoder: None,
            prefix: None,
            bound: Bound::default(),
            dispatches: 0,
            arena: Arena::new(&device),
            grid_slots: Vec::new(),
            bind_groups: BindGroupCache::default(),
            offsets: Vec::new(),
            uploads: Vec::new(),
            profiler: None,
        };

        let ctx = Context {
            inner: Arc::new(Inner {
                max_uniform_bytes: limits.max_uniform_buffer_binding_size,
                uniform_align: u64::from(limits.min_uniform_buffer_offset_alignment).max(4),
                max_workgroups_per_dimension: limits.max_compute_workgroups_per_dimension,
                max_dynamic_uniforms: limits.max_dynamic_uniform_buffers_per_pipeline_layout,
                timestamps: device.features().contains(wgpu::Features::TIMESTAMP_QUERY),
                zero_copy: device.features().contains(wgpu::Features::MAPPABLE_PRIMARY_BUFFERS),
                tuning: Tuning::for_adapter(&gpu.0.adapter, &limits, device.features()),
                cost: OnceLock::new(),
                gpu,
                builtins: BuildCache::default(),
                jit: JitState::new(&limits),
                kernels: BuildCache::default(),
                pool: Mutex::new(BufferPool::new(cap)),
                released: Arc::new(Mutex::new(Vec::new())),
                uploads: Arc::new(Mutex::new(FreeUploads::new(cap))),
                mappable: Arc::new(Mutex::new(FreeUploads::new(cap))),
                segment_bytes,
                check_finite: options.check_finite,
                fast_math: options.fast_math,
                pager: Pager::new(budget),
                unmapped: Mutex::new(Vec::new()),
                batch: Mutex::new(batch),
                dispatches_pending: std::sync::atomic::AtomicU32::new(0),
                id: record::next_context_id(),
                waits: WaitHistory::new(),
                gpu_error,
                #[cfg(test)]
                assume_busy: std::sync::atomic::AtomicBool::new(false),
                grid_layout,
                max_buffer_bytes,
            }),
        };
        let profile = ctx.inner.tuning.profile;
        let model = if options.calibrate {
            *ctx.inner.gpu.0.cost.get_or_init(|| {
                cost::calibrate(&ctx).unwrap_or_else(|_| CostModel::fallback(profile))
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
        Arc::clone(&self.inner.gpu.0.cost)
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
        let start = Instant::now();
        if !self.spin_until_idle() {
            self.device()
                .poll(wgpu::PollType::wait_indefinitely())
                .map_err(|e| LumaError::Poll(Arc::new(e)))?;
        }
        self.inner.waits.record(start.elapsed());
        self.take_gpu_error()
    }

    /// Polls without blocking for up to the spin budget ([`WaitHistory`]); `true` once the queue
    /// is empty. A thread that notices the end of a short kernel while running returns sooner
    /// than one that was blocked in `poll` and has to be woken up.
    fn spin_until_idle(&self) -> bool {
        let deadline = Instant::now() + self.inner.waits.budget();
        loop {
            match self.device().poll(wgpu::PollType::Poll) {
                Ok(status) if status.is_queue_empty() => return true,
                Ok(_) => {}
                // The blocking wait reports it.
                Err(_) => return false,
            }
            if Instant::now() >= deadline {
                return false;
            }
            std::hint::spin_loop();
        }
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
        let mut batch = self.batch()?;
        self.flush_locked(&mut batch);
        Ok(())
    }

    pub(crate) fn device(&self) -> &wgpu::Device {
        &self.inner.gpu.0.device
    }

    pub(crate) fn queue(&self) -> &wgpu::Queue {
        &self.inner.gpu.0.queue
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

    pub(crate) fn max_dynamic_uniforms(&self) -> u32 {
        self.inner.max_dynamic_uniforms
    }

    pub(crate) fn grid_layout(&self) -> &wgpu::BindGroupLayout {
        &self.inner.grid_layout
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
                let key = KernelKey::new(op.source(ty, &self.inner.tuning).into_owned(), "main", None);
                let label = format!("{op:?} {ty}");
                kernel::build_core(self, &key, Some(&label), true).map(Arc::new)
            })
            .map(|(core, _)| core)
    }

    /// Takes a buffer of at least `bytes` bytes with exactly `usage` from the pool, creating one
    /// if none is idle. The buffer's size is rounded up to a power of two and its contents are
    /// unspecified. Give it back with [`Context::recycle`].
    pub(crate) fn alloc_buffer(&self, bytes: u64, usage: BufferUsages) -> Result<Buffer> {
        let max = self.inner.max_buffer_bytes;
        let class = pool::size_class(bytes, max)
            .ok_or(LumaError::BufferTooLarge { requested: bytes, max })?;

        if let Some(buffer) = lock(&self.inner.pool).take(class, usage) {
            return Ok(buffer);
        }
        self.new_buffer(class, usage, false)
    }

    /// A new buffer; when the device is out of memory, gives back every idle buffer and the bind
    /// groups pinning them, then tries once more.
    fn new_buffer(&self, size: u64, usage: BufferUsages, mapped: bool) -> Result<Buffer> {
        if let Some(buffer) = self.create_buffer(size, usage, mapped) {
            return Ok(buffer);
        }
        let mut idle = lock(&self.inner.pool).drain();
        idle.extend(lock(&self.inner.mappable).drain());
        idle.extend(lock(&self.inner.uploads).drain());
        lock(&self.inner.released).extend(idle);
        self.flush_locked(&mut *self.batch()?);
        self.device()
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| LumaError::Poll(Arc::new(e)))?;
        let mut late = lock(&self.inner.mappable).drain();
        late.extend(lock(&self.inner.uploads).drain());
        if !late.is_empty() {
            let mut batch = self.batch()?;
            batch.pass = None;
            batch.bound = Bound::default();
            batch.bind_groups.forget(&late);
        }
        if let Some(buffer) = self.create_buffer(size, usage, mapped) {
            return Ok(buffer);
        }
        // Still full: move array data to host memory, one segment at a time, until it fits.
        while self.evict_one()? {
            let mut idle = lock(&self.inner.pool).drain();
            idle.extend(lock(&self.inner.mappable).drain());
            idle.extend(lock(&self.inner.uploads).drain());
            lock(&self.inner.released).extend(idle);
            self.flush_locked(&mut *self.batch()?);
            self.device()
                .poll(wgpu::PollType::wait_indefinitely())
                .map_err(|e| LumaError::Poll(Arc::new(e)))?;
            let mut late = lock(&self.inner.mappable).drain();
            late.extend(lock(&self.inner.uploads).drain());
            if !late.is_empty() {
                let mut batch = self.batch()?;
                batch.pass = None;
                batch.bound = Bound::default();
                batch.bind_groups.forget(&late);
            }
            if let Some(buffer) = self.create_buffer(size, usage, mapped) {
                return Ok(buffer);
            }
        }
        Err(LumaError::OutOfMemory { requested: size })
    }

    /// A new buffer, or `None` if the device is out of memory.
    fn create_buffer(&self, size: u64, usage: BufferUsages, mapped: bool) -> Option<Buffer> {
        let scope = self.device().push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        let buffer = self.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("luma buffer"),
            size,
            usage,
            mapped_at_creation: mapped,
        });
        pollster::block_on(scope.pop()).is_none().then_some(buffer)
    }

    /// Returns a buffer from [`Context::alloc_buffer`] to the pool.
    ///
    /// This is safe while recorded or submitted GPU work still uses the buffer. All of it is
    /// part of the one ordered command stream, and whoever takes the buffer next (to upload into
    /// it, to run a kernel on it) records that after everything recorded so far. Nothing writes
    /// to pooled buffers outside the stream (`Queue::write_buffer` is only used for uniform
    /// slots, which are never pooled).
    pub(crate) fn recycle(&self, buffer: Buffer) {
        let Some(buffer) = self.retire(buffer) else { return };
        if buffer.usage().contains(BufferUsages::MAP_WRITE) {
            // May still be used by the unsubmitted batch, which mapping would invalidate; it is
            // mapped after the next submission. This must not take the batch lock: it can be
            // held by the caller.
            lock(&self.inner.mappable).in_flight += buffer.size();
            lock(&self.inner.unmapped).push(buffer);
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
    pub fn features(&self) -> wgpu::Features {
        self.device().features()
    }

    // -- Profiling --

    /// Whether the GPU supports the timestamp queries [`Context::profile`] needs.
    pub fn profiling_supported(&self) -> bool {
        self.inner.timestamps
    }

    /// Measures the GPU time of every kernel launch recorded from now on, until
    /// [`Context::finish_profiling`]. Does nothing if profiling is already enabled.
    ///
    /// Fails with [`LumaError::Profiling`] if the adapter has no timestamp queries (see
    /// [`Context::profiling_supported`]).
    pub fn enable_profiling(&self) -> Result<()> {
        if !self.inner.timestamps {
            return Err(LumaError::Profiling(
                "the GPU does not support timestamp queries (wgpu feature TIMESTAMP_QUERY)".into(),
            ));
        }
        let mut batch = self.batch()?;
        if batch.profiler.is_none() {
            batch.profiler = Some(Profiler::new());
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
        if self.batch()?.profiler.is_none() {
            return Err(LumaError::Profiling("profiling is not enabled".into()));
        }
        // Lazy arrays still alive were computed inside the profiled region; record their fused
        // kernels while the profiler is still on.
        self.jit().evaluate_live();
        let profiler = {
            let mut batch = self.batch()?;
            let Some(profiler) = batch.profiler.take() else {
                return Err(LumaError::Profiling("profiling is not enabled".into()));
            };
            self.flush_locked(&mut batch);
            profiler
        };
        let splits = self.jit().take_split_notes();
        self.sync()?;

        // Everything profiled has completed; only now are its timestamps reliably written.
        let mut encoder = self
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("luma timestamps") });
        let (staging, labels) = profiler.resolve(self.device(), &mut encoder);
        self.queue().submit(Some(encoder.finish()));
        let stamps: Vec<u64> = self.map_read(&staging, staging.size())?;

        let period = f64::from(self.queue().get_timestamp_period());
        let mut samples = Vec::new();
        for (label, &[begin, end]) in labels.iter().zip(stamps.as_chunks::<2>().0) {
            let nanos = end.saturating_sub(begin) as f64 * period;
            samples.push((Arc::clone(label), Duration::from_secs_f64(nanos * 1e-9)));
        }
        let mut profile = Profile::from_samples(samples);
        profile.splits = splits;
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
