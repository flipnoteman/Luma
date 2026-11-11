use std::hash::{Hash, Hasher};
use std::sync::mpsc;
use std::sync::{Arc, Mutex, MutexGuard, OnceLock, PoisonError};
use std::time::{Duration, Instant};

use bytemuck::Pod;

use wgpu::{BindGroupEntry, BindingResource, Buffer, BufferBinding, BufferUsages};

use crate::builtin::Builtin;
use crate::error::{LumaError, Result};
use crate::jit::JitState;
use crate::array::Storage;
use crate::gpu::pager::Pager;
use crate::gpu::handle;
use crate::gpu::hash::{FxHasher, FxMap};
use crate::gpu::pool::{self, BufferPool};
use crate::kernel::{self, KernelBuilder, KernelCore, KernelKey};
use crate::profile::{Profile, Profiler};
use crate::tuning::{Tuning, TuningProfile};

/// Usage of every array storage buffer (and the scratch buffers of multi-pass operations, so they
/// share a pool class).
pub(crate) const STORAGE_USAGE: BufferUsages = BufferUsages::STORAGE
    .union(BufferUsages::COPY_DST)
    .union(BufferUsages::COPY_SRC);

/// Usage of the buffers uniform values are written into.
const UNIFORM_USAGE: BufferUsages = BufferUsages::UNIFORM.union(BufferUsages::COPY_DST);

/// Usage of readback buffers.
const STAGING_USAGE: BufferUsages = BufferUsages::MAP_READ.union(BufferUsages::COPY_DST);

/// Usage of upload staging buffers.
const UPLOAD_USAGE: BufferUsages = BufferUsages::MAP_WRITE.union(BufferUsages::COPY_SRC);

/// Readbacks larger than twice this go in pieces of this size, see [`Context::read`].
const READ_CHUNK: u64 = 32 << 20;

/// Usage of storage buffers the host writes directly (zero-copy uploads).
const MAPPABLE_STORAGE_USAGE: BufferUsages = STORAGE_USAGE.union(BufferUsages::MAP_WRITE);

/// Mapped upload staging buffers ready to be written.
#[derive(Debug)]
struct FreeUploads {
    buffers: Vec<Buffer>,
    bytes: u64,
    /// Bytes of buffers in the unsubmitted batch or waiting for a `map_async` callback.
    in_flight: u64,
    /// Most bytes of idle buffers kept for reuse; scales with the memory budget.
    cap: u64,
}

impl FreeUploads {
    fn new(cap: u64) -> Self {
        FreeUploads { buffers: Vec::new(), bytes: 0, in_flight: 0, cap }
    }

    fn take(&mut self, size: u64) -> Option<Buffer> {
        let at = self.buffers.iter().position(|buffer| buffer.size() == size)?;
        self.bytes -= size;
        Some(self.buffers.swap_remove(at))
    }

    /// Keeps `buffer` unless that would exceed the cap; returns it otherwise.
    fn put(&mut self, buffer: Buffer) -> Option<Buffer> {
        if self.bytes + buffer.size() <= self.cap {
            self.bytes += buffer.size();
            self.buffers.push(buffer);
            None
        } else {
            Some(buffer)
        }
    }

    fn drain(&mut self) -> Vec<Buffer> {
        self.bytes = 0;
        std::mem::take(&mut self.buffers)
    }
}

/// How long [`Context::sync`] polls for the GPU to finish before it blocks in the driver. Bounds
/// the CPU time spent spinning on a long kernel; kernels that finish inside it are noticed
/// without waking a blocked thread.
const SYNC_SPIN: Duration = Duration::from_micros(100);

/// Number of dispatches after which the pending batch is submitted without waiting for a sync
/// point. Bounds how much work piles up in one command buffer.
pub(crate) const BATCH_LIMIT: u32 = 256;

/// Size of one uniform arena chunk. Uniform values of a batch are packed into it, one
/// `min_uniform_buffer_offset_alignment` slot each.
const ARENA_CHUNK: u64 = 256 * 1024;

/// Bind groups cached per context before the cache is emptied (at the next flush).
const MAX_CACHED_BIND_GROUPS: usize = 2048;

/// Problem sizes whose grid uniform slot is remembered within a batch, so that launches with a
/// size seen before share its slot (and its bind group) instead of writing a new one.
const MAX_GRID_SLOTS: usize = 16;

/// One argument of a kernel launch, already validated: what to bind at `@group(1)` in order.
#[derive(Clone, Copy)]
pub(crate) enum RawArg<'a> {
    /// The first `bytes` bytes of a storage buffer.
    Storage { buffer: &'a Buffer, bytes: u64, writable: bool },
    /// A uniform value, copied at launch time.
    Uniform(&'a [u8]),
}

impl<'a> RawArg<'a> {
    pub(crate) fn read(buffer: &'a Buffer, bytes: u64) -> Self {
        RawArg::Storage { buffer, bytes, writable: false }
    }

    pub(crate) fn write(buffer: &'a Buffer, bytes: u64) -> Self {
        RawArg::Storage { buffer, bytes, writable: true }
    }

    /// What is bound: a buffer and how many of its bytes. Uniforms live in `uniforms`, the arena
    /// chunk this launch's uniform slots were allocated from.
    fn binding(&self, uniforms: &'a Buffer) -> (&'a Buffer, u64) {
        match *self {
            RawArg::Storage { buffer, bytes, .. } => (buffer, bytes),
            RawArg::Uniform(data) => (uniforms, data.len() as u64),
        }
    }
}

/// One buffer of the uniform arena together with the bytes that will be written into it.
#[derive(Debug)]
struct Chunk {
    buffer: Buffer,
    /// Host copy of the slots handed out so far (`data.len()` is the used size).
    data: Vec<u8>,
    /// Bind group of `@group(0)` for this chunk (the grid uniform at a dynamic offset).
    grid_group: Option<wgpu::BindGroup>,
}

/// Uniform values of the pending batch. Every value gets its own slot in a chunk that stays
/// untouched until the batch executed, so each launch sees the value it was given even though
/// the data is copied into the chunk (from a mapped staging buffer, once per chunk when the
/// batch is submitted) before the batch's commands run: the copies are recorded into a command
/// buffer of their own that goes first in the same submission. Slots are only reused after the
/// batch holding them was submitted; the copy into a reused slot is ordered after that
/// submission on the queue.
#[derive(Debug)]
struct Arena {
    /// The chunk slots are currently handed out from.
    current: Chunk,
    /// Chunks of this batch that filled up; dropped when the batch is submitted.
    full: Vec<Chunk>,
    /// Counts the chunks that were started; identifies `current` in the bound-state tracking.
    epoch: u64,
}

fn new_chunk(device: &wgpu::Device, size: u64) -> Chunk {
    Chunk {
        buffer: device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("luma uniforms"),
            size,
            usage: UNIFORM_USAGE,
            mapped_at_creation: false,
        }),
        data: Vec::new(),
        grid_group: None,
    }
}

/// Bytes a uniform value of `len` bytes occupies in the arena.
fn slot_size(align: u64, len: usize) -> u64 {
    (len as u64).next_multiple_of(wgpu::COPY_BUFFER_ALIGNMENT).max(4).next_multiple_of(align)
}

impl Arena {
    fn new(device: &wgpu::Device) -> Self {
        Arena { current: new_chunk(device, ARENA_CHUNK), full: Vec::new(), epoch: 0 }
    }

    /// `true` if `bytes` more bytes of slots fit into the current chunk.
    fn has_room(&self, bytes: u64) -> bool {
        self.current.data.len() as u64 + bytes <= self.current.buffer.size()
    }

    /// Starts a new chunk with room for at least `bytes` bytes; the old one is kept until the
    /// batch is submitted.
    fn roll(&mut self, device: &wgpu::Device, bytes: u64) {
        let fresh = new_chunk(device, bytes.max(ARENA_CHUNK));
        self.full.push(std::mem::replace(&mut self.current, fresh));
        self.epoch += 1;
    }

    /// Reserves a fresh slot holding `data` in the current chunk and returns its offset; the
    /// caller has made room for it ([`Arena::has_room`]). The bytes reach the GPU in
    /// [`Arena::upload`].
    fn push(&mut self, align: u64, data: &[u8]) -> u64 {
        let slot = slot_size(align, data.len());
        let chunk = &mut self.current;
        let offset = chunk.data.len() as u64;
        chunk.data.extend_from_slice(data);
        chunk.data.resize((offset + slot) as usize, 0);
        offset
    }

    /// Copies the slots of every chunk to the GPU and empties the arena for the next batch,
    /// keeping the current chunk. Must run right before the batch is submitted: the returned
    /// commands (the copies out of the staging buffers) have to be submitted ahead of the
    /// batch's own, and so run after everything submitted earlier.
    ///
    /// The staging buffers are the reusable mapped ones of [`Context::upload`] (pushed to
    /// `staged`, which the caller maps again after the submission), not `Queue::write_buffer`:
    /// that one has wgpu create a staging buffer per call, which allocates and maps a fresh
    /// block of host-visible memory whenever nothing else lives in it, and frees it again at
    /// the next poll, on every submission.
    fn upload(
        &mut self,
        ctx: &Context,
        staged: &mut Vec<Buffer>,
        released: &mut Vec<Buffer>,
    ) -> Option<wgpu::CommandBuffer> {
        let mut encoder: Option<wgpu::CommandEncoder> = None;
        for chunk in self.full.iter().chain(std::iter::once(&self.current)) {
            if chunk.data.is_empty() {
                continue;
            }
            match ctx.stage(&chunk.data) {
                Some(staging) => {
                    encoder
                        .get_or_insert_with(|| {
                            ctx.device().create_command_encoder(&wgpu::CommandEncoderDescriptor {
                                label: Some("luma uniforms"),
                            })
                        })
                        .copy_buffer_to_buffer(&staging, 0, &chunk.buffer, 0, chunk.data.len() as u64);
                    staged.push(staging);
                }
                // No staging buffer to be had (the device is out of memory): `write_buffer`
                // brings its own and is ordered the same way.
                None => ctx.queue().write_buffer(&chunk.buffer, 0, &chunk.data),
            }
        }
        released.extend(self.full.drain(..).map(|chunk| chunk.buffer));
        self.current.data.clear();
        encoder.map(wgpu::CommandEncoder::finish)
    }
}

/// A bind group of `@group(1)` and what it was created for.
#[derive(Debug)]
struct CachedGroup {
    kernel: u64,
    /// What is bound, in binding order. Uniforms are identified by their arena chunk (the slot
    /// is a dynamic offset). Holding the buffers keeps them alive as long as the group is cached,
    /// so a buffer's identity cannot be reused by another one.
    entries: Box<[(Buffer, u64)]>,
    group: wgpu::BindGroup,
}

impl CachedGroup {
    fn matches<'a>(&self, kernel: u64, entries: impl ExactSizeIterator<Item = (&'a Buffer, u64)>) -> bool {
        self.kernel == kernel
            && self.entries.len() == entries.len()
            && self.entries.iter().zip(entries).all(|((a, a_bytes), (b, b_bytes))| a == b && *a_bytes == b_bytes)
    }
}

/// Bind groups of `@group(1)` by (kernel, bound buffers). Looked up with a hash computed from
/// borrowed handles, so a hit allocates and clones nothing.
#[derive(Debug, Default)]
struct BindGroupCache {
    buckets: FxMap<u64, Vec<CachedGroup>>,
    len: usize,
}

impl BindGroupCache {
    /// Drops the groups binding any of `buffers`, so that the cache does not keep them alive.
    /// Invalidates indices into buckets: only call it while nothing is bound.
    fn forget(&mut self, buffers: &[Buffer]) {
        self.buckets.retain(|_, bucket| {
            bucket.retain(|group| !group.entries.iter().any(|(b, _)| buffers.contains(b)));
            !bucket.is_empty()
        });
        self.len = self.buckets.values().map(Vec::len).sum();
    }
}

/// What the open compute pass has bound; launches skip what is already in place.
#[derive(Debug, Default, Clone, Copy)]
struct Bound {
    /// The kernel whose pipeline is set.
    kernel: Option<u64>,
    /// The grid uniform at `@group(0)`: arena chunk epoch and slot offset.
    grid: Option<(u64, u32)>,
    /// The bind group at `@group(1)`: bucket hash and index in it.
    args: Option<(u64, usize)>,
}

/// The commands recorded since the last submission.
#[derive(Debug)]
struct Batch {
    /// The compute pass consecutive launches are recorded into (one pass per batch instead of
    /// one per launch). Commands that need the encoder itself end it first. Declared before the
    /// encoder so that it is dropped (ended) first.
    pass: Option<wgpu::ComputePass<'static>>,
    encoder: Option<wgpu::CommandEncoder>,
    bound: Bound,
    dispatches: u32,
    arena: Arena,
    /// Slots of the grid uniform written to `arena.current` in this batch, by problem size.
    grid_slots: Vec<([u32; 3], u32)>,
    bind_groups: BindGroupCache,
    /// Dynamic offsets of the launch being recorded (kept to reuse its allocation).
    offsets: Vec<u32>,
    /// Upload staging buffers copied from in this batch; mapped again once it is submitted.
    uploads: Vec<Buffer>,
    /// Present while profiling is enabled.
    profiler: Option<Profiler>,
}

#[derive(Debug)]
struct Inner {
    /// Built-in kernels, built on first use of their (operation, element type).
    builtins: Mutex<FxMap<(Builtin, &'static str), Arc<KernelCore>>>,
    /// Lazy-evaluation state: fused kernels and the pending expressions.
    jit: JitState,
    /// Custom kernels, built on first use of their (source, entry, workgroup).
    kernels: Mutex<FxMap<KernelKey, Arc<KernelCore>>>,
    pool: Mutex<BufferPool>,
    /// Buffers the pool released; the cached bind groups still holding them are dropped at the
    /// next submission, which frees them.
    released: Arc<Mutex<Vec<Buffer>>>,
    /// Mapped upload staging buffers, refilled by the map callbacks of submitted ones.
    uploads: Arc<Mutex<FreeUploads>>,
    mappable: Arc<Mutex<FreeUploads>>,
    /// Bytes of a full segment of an array, see [`Storage`](crate::array::Storage).
    segment_bytes: u64,
    /// Which segments of arrays hold a device buffer.
    pager: Pager,
    /// Recycled mappable storage buffers waiting for the next submission before they are mapped.
    unmapped: Mutex<Vec<Buffer>>,
    /// Whether `Array::from_slice` uploads into mappable storage buffers (unified memory).
    zero_copy: bool,
    /// Kernel parameters for this GPU.
    tuning: Tuning,
    batch: Mutex<Batch>,
    /// The first error the GPU reported outside of a call that could return it.
    gpu_error: Arc<Mutex<Option<String>>>,
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

/// Locks `mutex`, ignoring poisoning: the guarded caches and free lists stay consistent even if
/// a holder panicked.
fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
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
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
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
}

impl ContextOptions {
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

    /// Options from the environment: `LUMA_DEVICE` = `gpu` | `cpu` (anything else: `Auto`),
    /// `LUMA_MEMORY_BUDGET` and `LUMA_CHUNK_BYTES` as decimal byte counts.
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
            bound: Bound::default(),
            dispatches: 0,
            arena: Arena::new(&device),
            grid_slots: Vec::new(),
            bind_groups: BindGroupCache::default(),
            offsets: Vec::new(),
            uploads: Vec::new(),
            profiler: None,
        };

        Ok(Context {
            inner: Arc::new(Inner {
                max_uniform_bytes: limits.max_uniform_buffer_binding_size,
                uniform_align: u64::from(limits.min_uniform_buffer_offset_alignment).max(4),
                max_workgroups_per_dimension: limits.max_compute_workgroups_per_dimension,
                max_dynamic_uniforms: limits.max_dynamic_uniform_buffers_per_pipeline_layout,
                timestamps: device.features().contains(wgpu::Features::TIMESTAMP_QUERY),
                zero_copy: device.features().contains(wgpu::Features::MAPPABLE_PRIMARY_BUFFERS),
                tuning: Tuning::for_adapter(&gpu.0.adapter, &limits, device.features()),
                gpu,
                builtins: Mutex::new(FxMap::default()),
                jit: JitState::new(&limits),
                kernels: Mutex::new(FxMap::default()),
                pool: Mutex::new(BufferPool::new(cap)),
                released: Arc::new(Mutex::new(Vec::new())),
                uploads: Arc::new(Mutex::new(FreeUploads::new(cap))),
                mappable: Arc::new(Mutex::new(FreeUploads::new(cap))),
                segment_bytes,
                pager: Pager::new(budget),
                unmapped: Mutex::new(Vec::new()),
                batch: Mutex::new(batch),
                gpu_error,
                grid_layout,
                max_buffer_bytes,
            }),
        })
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
        self.flush()?;
        if !self.spin_until_idle() {
            self.device()
                .poll(wgpu::PollType::wait_indefinitely())
                .map_err(|e| LumaError::Poll(Arc::new(e)))?;
        }
        self.take_gpu_error()
    }

    /// Polls without blocking for up to [`SYNC_SPIN`]; `true` once the queue is empty. A thread
    /// that notices the end of a short kernel while running returns about 20 us sooner than one
    /// that was blocked in `poll` and has to be woken up.
    fn spin_until_idle(&self) -> bool {
        let deadline = Instant::now() + SYNC_SPIN;
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
    /// the temporaries of an expression).
    ///
    /// Only [`Context::sync`] and the operations that read results back
    /// ([`Array::to_vec`](crate::Array::to_vec), [`Array::eval`](crate::Array::eval),
    /// reductions) wait, and only they report an error the GPU raises while running the batch.
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
    /// println!("{}", x.sum()?);
    /// # Ok::<(), luma::LumaError>(())
    /// ```
    pub fn flush(&self) -> Result<()> {
        self.jit().evaluate_live();
        self.submit()
    }

    /// Submits the pending batch without waiting for it, like [`Context::flush`] but without
    /// evaluating the lazy arrays that are still alive: only what was recorded so far goes to
    /// the GPU. For [`Array::submit`](crate::Array::submit) and [`submit_all`](crate::submit_all),
    /// which record exactly what they want submitted.
    pub(crate) fn submit(&self) -> Result<()> {
        let mut batch = self.batch()?;
        self.flush_locked(&mut batch);
        Ok(())
    }

    /// Number of dispatches recorded but not yet submitted.
    #[cfg(test)]
    pub(crate) fn pending_dispatches(&self) -> u32 {
        self.batch().map(|b| b.dispatches).unwrap_or(0)
    }

    fn take_gpu_error(&self) -> Result<()> {
        match lock(&self.inner.gpu_error).take() {
            Some(message) => Err(LumaError::Gpu(message)),
            None => Ok(()),
        }
    }

    fn batch(&self) -> Result<MutexGuard<'_, Batch>> {
        self.inner.batch.lock().map_err(|_| LumaError::LockPoisoned)
    }

    /// Submits the pending batch (if there is one) and prepares the next. Never waits.
    fn flush_locked(&self, batch: &mut Batch) -> Option<wgpu::SubmissionIndex> {
        // The open pass has to end before the encoder can be finished.
        batch.pass = None;
        batch.bound = Bound::default();
        batch.dispatches = 0;
        // The slots of the arena are reused by the next batch.
        batch.grid_slots.clear();
        let mut released = std::mem::take(&mut *lock(&self.inner.released));
        let uniforms = batch.arena.upload(self, &mut batch.uploads, &mut released);
        if batch.bind_groups.len > MAX_CACHED_BIND_GROUPS {
            // In-flight work keeps what it uses alive; this only drops our references.
            batch.bind_groups = BindGroupCache::default();
        } else if !released.is_empty() {
            batch.bind_groups.forget(&released);
        }
        // Everything recorded that uses these buffers is in the batch submitted below.
        let unmapped = std::mem::take(&mut *lock(&self.inner.unmapped));
        // The uniform copies go first: the batch's dispatches read what they write.
        let commands = batch.encoder.take().map(|encoder| encoder.finish());
        let submission = (uniforms.is_some() || commands.is_some())
            .then(|| self.queue().submit(uniforms.into_iter().chain(commands)));
        for staging in batch.uploads.drain(..) {
            let free = Arc::clone(&self.inner.uploads);
            let buffer = staging.clone();
            staging.slice(..).map_async(wgpu::MapMode::Write, move |result| {
                let mut free = lock(&free);
                free.in_flight = free.in_flight.saturating_sub(buffer.size());
                if result.is_ok() {
                    free.put(buffer);
                }
            });
        }
        for storage in unmapped {
            let free = Arc::clone(&self.inner.mappable);
            let released = Arc::clone(&self.inner.released);
            let buffer = storage.clone();
            storage.slice(..).map_async(wgpu::MapMode::Write, move |result| {
                let dropped = {
                    let mut free = lock(&free);
                    free.in_flight = free.in_flight.saturating_sub(buffer.size());
                    if result.is_ok() { free.put(buffer) } else { None }
                };
                if let Some(dropped) = dropped {
                    lock(&released).push(dropped);
                }
            });
        }
        submission
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

    pub(crate) fn kernel_cache(&self) -> MutexGuard<'_, FxMap<KernelKey, Arc<KernelCore>>> {
        lock(&self.inner.kernels)
    }

    /// The cached built-in kernel for `op` on elements of WGSL type `ty`, compiled on first use.
    pub(crate) fn builtin(&self, op: Builtin, ty: &'static str) -> Result<Arc<KernelCore>> {
        let mut cache = lock(&self.inner.builtins);
        if let Some(core) = cache.get(&(op, ty)) {
            return Ok(Arc::clone(core));
        }
        let key = KernelKey::new(op.source(ty, &self.inner.tuning).into_owned(), "main", None);
        let label = format!("{op:?} {ty}");
        let core = Arc::new(kernel::build_core(self, &key, Some(&label), true)?);
        cache.insert((op, ty), Arc::clone(&core));
        Ok(core)
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
            self.batch()?.bind_groups.forget(&late);
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
                self.batch()?.bind_groups.forget(&late);
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

    /// Records a copy of `data` into the start of `dst`, ordered after everything recorded so
    /// far. `data.len()` must be a multiple of 4.
    ///
    /// The bytes go through a mappable staging buffer. Those are reused: once the batch that
    /// copies out of one is submitted it is mapped again, and the next upload of its size class
    /// writes straight into it.
    pub(crate) fn upload(&self, dst: &Buffer, data: &[u8]) -> Result<()> {
        let bytes = data.len() as u64;
        let max = self.inner.max_buffer_bytes;
        let class = pool::size_class(bytes, max)
            .ok_or(LumaError::BufferTooLarge { requested: bytes, max })?;
        let staging = self.take_upload_buffer(&self.inner.uploads, class, UPLOAD_USAGE)?;
        {
            let mut view = staging.get_mapped_range_mut(..bytes).map_err(LumaError::MapRange)?;
            write_parallel(data, view.slice(..));
        }
        staging.unmap();

        let mut guard = self.batch()?;
        let encoder = self.encoder(&mut guard);
        encoder.copy_buffer_to_buffer(&staging, 0, dst, 0, bytes);
        lock(&self.inner.uploads).in_flight += staging.size();
        guard.uploads.push(staging);
        Ok(())
    }

    /// Whether `from_slice` writes straight into mappable storage buffers (unified memory)
    /// instead of copying out of a staging buffer.
    pub(crate) fn zero_copy(&self) -> bool {
        self.inner.zero_copy
    }

    /// A storage buffer holding `data` at its start, written by the host without a GPU copy.
    /// Only with [`Context::zero_copy`]; give it back with [`Context::recycle`]. `data.len()`
    /// must be a multiple of 4.
    pub(crate) fn alloc_filled(&self, data: &[u8]) -> Result<Buffer> {
        let bytes = data.len() as u64;
        let max = self.inner.max_buffer_bytes;
        let class = pool::size_class(bytes, max)
            .ok_or(LumaError::BufferTooLarge { requested: bytes, max })?;
        let buffer = self.take_upload_buffer(&self.inner.mappable, class, MAPPABLE_STORAGE_USAGE)?;
        {
            let mut view = buffer.get_mapped_range_mut(..bytes).map_err(LumaError::MapRange)?;
            write_parallel(data, view.slice(..));
        }
        buffer.unmap();
        Ok(buffer)
    }

    /// Takes a mapped `MAP_WRITE` buffer of `class` bytes from `pool`, polling completed
    /// submissions (and waiting if more bytes than the idle cap are in flight) before
    /// allocating a fresh one.
    fn take_upload_buffer(
        &self,
        pool: &Mutex<FreeUploads>,
        class: u64,
        usage: BufferUsages,
    ) -> Result<Buffer> {
        if let Some(buffer) = self.take_idle(pool, class) {
            return Ok(buffer);
        }
        let (in_flight, cap) = {
            let uploads = lock(pool);
            (uploads.in_flight, uploads.cap)
        };
        if in_flight > cap {
            self.flush_locked(&mut *self.batch()?);
            self.device()
                .poll(wgpu::PollType::wait_indefinitely())
                .map_err(|e| LumaError::Poll(Arc::new(e)))?;
            if let Some(buffer) = lock(pool).take(class) {
                return Ok(buffer);
            }
        }
        self.new_buffer(class, usage, true)
    }

    /// An idle mapped buffer of `class` bytes from `pool`. Polls completed submissions first,
    /// which hands back the buffers whose `map_async` callback is due. Never submits or waits.
    fn take_idle(&self, pool: &Mutex<FreeUploads>, class: u64) -> Option<Buffer> {
        if let Some(buffer) = lock(pool).take(class) {
            return Some(buffer);
        }
        let _ = self.device().poll(wgpu::PollType::Poll);
        lock(pool).take(class)
    }

    /// A staging buffer holding `data` at its start, unmapped and ready to be copied out of, for
    /// the uniform slots of [`Arena::upload`]. It counts as in flight: give it to
    /// [`Batch::uploads`], which maps it again once the batch that copies out of it is
    /// submitted. `None` if no staging buffer can be had (the device is out of memory).
    ///
    /// This runs while the batch is being submitted, with the batch locked, so unlike
    /// [`Context::take_upload_buffer`] it must not flush or wait: it reuses an idle buffer or
    /// creates a mapped one.
    fn stage(&self, data: &[u8]) -> Option<Buffer> {
        let bytes = data.len() as u64;
        let class = pool::size_class(bytes, self.inner.max_buffer_bytes)?;
        let staging = match self.take_idle(&self.inner.uploads, class) {
            Some(buffer) => buffer,
            None => self.create_buffer(class, UPLOAD_USAGE, true)?,
        };
        {
            // A plain copy: the slots of one chunk are kilobytes, and `write_parallel` asks the
            // OS for the thread count first, which costs more than copying them.
            let mut view = staging.get_mapped_range_mut(..bytes).ok()?;
            view.slice(..).copy_from_slice(data);
        }
        staging.unmap();
        lock(&self.inner.uploads).in_flight += staging.size();
        Some(staging)
    }

    /// Records zeroing the first `bytes` bytes of `buffer`.
    pub(crate) fn clear(&self, buffer: &Buffer, bytes: u64) -> Result<()> {
        let mut guard = self.batch()?;
        self.encoder(&mut guard).clear_buffer(buffer, 0, Some(bytes));
        Ok(())
    }

    /// Records copying `bytes` bytes from `src` at `src_offset` to `dst` at `dst_offset`, for
    /// gathering the partial results of several segments into one buffer.
    pub(crate) fn copy(&self, src: &Buffer, src_offset: u64, dst: &Buffer, dst_offset: u64, bytes: u64) -> Result<()> {
        let mut guard = self.batch()?;
        self.encoder(&mut guard).copy_buffer_to_buffer(src, src_offset, dst, dst_offset, bytes);
        Ok(())
    }

    /// The batch's command encoder, for commands that are not dispatches. The open compute pass
    /// (which borrows the encoder) ends first, so the commands recorded here run after the
    /// launches recorded before and before those recorded after.
    fn encoder<'a>(&self, batch: &'a mut Batch) -> &'a mut wgpu::CommandEncoder {
        batch.pass = None;
        batch.bound = Bound::default();
        batch.encoder.get_or_insert_with(|| {
            self.device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("luma batch") })
        })
    }

    /// Copies the first `bytes` bytes of `buffer` into a new vector, see [`Context::read_into`].
    pub(crate) fn read<T: Pod>(&self, buffer: &Buffer, bytes: u64) -> Result<Vec<T>> {
        self.read_vec(bytes, |dst| self.read_into(buffer, dst))
    }

    /// Copies the data of `storage` into a new vector, segment by segment.
    pub(crate) fn read_storage<T: Pod>(&self, storage: &Storage) -> Result<Vec<T>> {
        self.read_vec(storage.byte_len(), |dst| self.read_storage_into(storage, dst))
    }

    /// Copies the data of `storage` into `dst` (exactly `storage.byte_len()` bytes), after
    /// everything recorded so far; on success every byte of `dst` is written.
    pub(crate) fn read_storage_into(&self, storage: &Storage, dst: &mut [std::mem::MaybeUninit<u8>]) -> Result<()> {
        let elem = storage.elem();
        let mut rest = dst;
        for j in 0..storage.segments() {
            let (piece, tail) = rest.split_at_mut(storage.range(j).len() * elem);
            storage.segment(j).read_into(piece)?;
            rest = tail;
        }
        Ok(())
    }

    /// A vector of `bytes / size_of::<T>()` elements, in memory advised to use huge pages, that
    /// `fill` writes completely.
    fn read_vec<T: Pod>(
        &self,
        bytes: u64,
        fill: impl FnOnce(&mut [std::mem::MaybeUninit<u8>]) -> Result<()>,
    ) -> Result<Vec<T>> {
        let len = usize::try_from(bytes)
            .map_err(|_| LumaError::InvalidShape(format!("{bytes} bytes do not fit in host memory")))?
            / std::mem::size_of::<T>();
        let mut out: Vec<T> = Vec::with_capacity(len);
        let spare = out.spare_capacity_mut();
        // SAFETY: a `MaybeUninit<T>` slice viewed as the same bytes; `T: Pod` has no padding or
        // invalid bit patterns, so any initialized bytes make valid elements.
        let dst = unsafe {
            std::slice::from_raw_parts_mut(spare.as_mut_ptr().cast::<std::mem::MaybeUninit<u8>>(), len * std::mem::size_of::<T>())
        };
        advise_huge_pages(dst);
        fill(dst)?;
        // SAFETY: `fill` wrote all `len * size_of::<T>()` bytes.
        unsafe { out.set_len(len) };
        Ok(out)
    }

    /// Copies the first `dst.len()` bytes of `buffer` into `dst`, after everything recorded so
    /// far; on success every byte of `dst` is written. Large reads go in [`READ_CHUNK`] pieces,
    /// each its own submission, so copying one piece out of its staging buffer overlaps the
    /// transfer of the next.
    pub(crate) fn read_into(&self, buffer: &Buffer, dst: &mut [std::mem::MaybeUninit<u8>]) -> Result<()> {
        let bytes = dst.len() as u64;
        let chunk = if bytes > 2 * READ_CHUNK { READ_CHUNK } else { bytes.max(1) };
        let mut staged = Vec::new();
        for offset in (0..bytes).step_by(chunk as usize) {
            let size = chunk.min(bytes - offset);
            match self.alloc_buffer(size, STAGING_USAGE) {
                Ok(staging) => staged.push((offset, size, staging)),
                Err(err) => {
                    for (_, _, s) in staged {
                        self.recycle(s);
                    }
                    return Err(err);
                }
            }
        }
        let mut maps = Vec::with_capacity(staged.len());
        match self.batch() {
            Ok(mut guard) => {
                for (offset, size, staging) in &staged {
                    self.encoder(&mut guard).copy_buffer_to_buffer(buffer, *offset, staging, 0, *size);
                    let submission = self.flush_locked(&mut guard);
                    maps.push((submission, map_for_read(staging.slice(..*size))));
                }
            }
            Err(err) => {
                for (_, _, s) in staged {
                    self.recycle(s);
                }
                return Err(err);
            }
        }

        let mut staged_iter = staged.into_iter().zip(maps);
        while let Some(((offset, size, staging), (submission, mapped))) = staged_iter.next() {
            let step = self.wait_map(submission, &mapped).and_then(|()| {
                let view = staging.slice(..size).get_mapped_range().map_err(LumaError::MapRange)?;
                copy_parallel(&view, &mut dst[offset as usize..(offset + size) as usize]);
                Ok(())
            });
            staging.unmap();
            if let Err(err) = step {
                for ((_, _, rest), _) in staged_iter {
                    rest.unmap();
                }
                return Err(err);
            }
            self.recycle(staging);
        }
        Ok(())
    }

    /// Waits for the GPU and returns the first `bytes` bytes of `staging`, a `MAP_READ` buffer
    /// whose contents were copied in by a submitted batch.
    fn map_read<T: Pod>(&self, staging: &Buffer, bytes: u64) -> Result<Vec<T>> {
        let slice = staging.slice(..bytes);
        self.wait_map(None, &map_for_read(slice))?;
        let result = slice.get_mapped_range().map(|view| bytemuck::pod_collect_to_vec(&view));
        staging.unmap();
        result.map_err(LumaError::MapRange)
    }

    /// Waits until `submission` (the latest one if `None`) has run and the mapping `mapped`
    /// reports on is done.
    fn wait_map(&self, submission: Option<wgpu::SubmissionIndex>, mapped: &MapReceiver) -> Result<()> {
        self.device()
            .poll(wgpu::PollType::Wait { submission_index: submission, timeout: None })
            .map_err(|e| LumaError::Poll(Arc::new(e)))?;
        // The callback runs on whichever thread polls the device first, so it may still be
        // in flight on another thread after our poll returns; wait for it.
        let mapped = mapped
            .recv()
            .map_err(|_| LumaError::Map(wgpu::BufferAsyncError))?
            .map_err(LumaError::Map);
        // An error the GPU raised while running the batch explains a failed map better.
        self.take_gpu_error()?;
        mapped
    }

    /// Records a dispatch of `kernel` and returns without waiting for it.
    ///
    /// `args` are bound to `@group(1)` in order; `size` is what `luma.size` is in the shader.
    /// Arguments must already be validated against the kernel's bindings.
    pub(crate) fn launch(
        &self,
        kernel: &KernelCore,
        workgroups: [u32; 3],
        size: [u32; 3],
        args: &[RawArg<'_>],
    ) -> Result<()> {
        check_aliasing(args)?;

        let mut guard = self.batch()?;
        self.record_launch(&mut guard, kernel, workgroups, size, args)?;
        if guard.dispatches >= BATCH_LIMIT {
            self.flush_locked(&mut guard);
        }
        Ok(())
    }

    /// The body of [`Context::launch`]: allocates the arena slots, finds or creates the bind
    /// group, and records the dispatch. After the first launch of a kernel with a given set of
    /// buffers this allocates nothing.
    fn record_launch(
        &self,
        batch: &mut Batch,
        kernel: &KernelCore,
        workgroups: [u32; 3],
        size: [u32; 3],
        args: &[RawArg<'_>],
    ) -> Result<()> {
        let inner = &*self.inner;
        let device = self.device();
        let align = inner.uniform_align;
        let Batch { encoder, pass, bound, dispatches, arena, grid_slots, bind_groups, offsets, profiler, .. } =
            batch;

        // Uniform values go into fresh arena slots, all of this launch's in one chunk. The grid
        // uniform shares the slot of an earlier launch of the same size.
        let uniform_bytes: u64 = args
            .iter()
            .map(|arg| match arg {
                RawArg::Uniform(data) => slot_size(align, data.len()),
                RawArg::Storage { .. } => 0,
            })
            .sum();
        let grid_bytes = slot_size(align, GRID_BYTES as usize);
        let mut grid_offset =
            grid_slots.iter().find(|(known, _)| *known == size).map(|&(_, offset)| offset);
        let needed = uniform_bytes + if grid_offset.is_some() { 0 } else { grid_bytes };
        if !arena.has_room(needed) {
            arena.roll(device, uniform_bytes + grid_bytes);
            // The remembered slots belong to the chunk that was left.
            grid_slots.clear();
            grid_offset = None;
        }
        let grid_offset = match grid_offset {
            Some(offset) => offset,
            None => {
                let offset = dynamic_offset(
                    arena.push(align, bytemuck::bytes_of(&[size[0], size[1], size[2], 0u32])),
                )?;
                if grid_slots.len() < MAX_GRID_SLOTS {
                    grid_slots.push((size, offset));
                }
                offset
            }
        };
        offsets.clear();
        for arg in args {
            if let RawArg::Uniform(data) = arg {
                offsets.push(dynamic_offset(arena.push(align, data))?);
            }
        }

        // Bind groups: the grid's per chunk, the arguments' per (kernel, buffers).
        let Chunk { buffer: uniforms, grid_group, .. } = &mut arena.current;
        let uniforms: &Buffer = uniforms;
        let grid_group = &*grid_group.get_or_insert_with(|| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &inner.grid_layout,
                entries: &[BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::Buffer(BufferBinding {
                        buffer: uniforms,
                        offset: 0,
                        size: std::num::NonZeroU64::new(GRID_BYTES),
                    }),
                }],
            })
        });

        let bindings = || args.iter().map(|arg| arg.binding(uniforms));
        let mut hasher = FxHasher::default();
        kernel.id.hash(&mut hasher);
        for (buffer, bytes) in bindings() {
            buffer.hash(&mut hasher);
            bytes.hash(&mut hasher);
        }
        let key = hasher.finish();
        let bucket = bind_groups.buckets.entry(key).or_default();
        let index = match bucket.iter().position(|cached| cached.matches(kernel.id, bindings())) {
            Some(index) => index,
            None => {
                let group_entries: Vec<BindGroupEntry<'_>> = bindings()
                    .enumerate()
                    .map(|(binding, (buffer, bytes))| BindGroupEntry {
                        binding: binding as u32,
                        resource: BindingResource::Buffer(BufferBinding {
                            buffer,
                            offset: 0,
                            size: std::num::NonZeroU64::new(bytes),
                        }),
                    })
                    .collect();
                let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout: &kernel.args_layout,
                    entries: &group_entries,
                });
                bucket.push(CachedGroup {
                    kernel: kernel.id,
                    entries: bindings().map(|(buffer, bytes)| (buffer.clone(), bytes)).collect(),
                    group,
                });
                bind_groups.len += 1;
                bucket.len().saturating_sub(1)
            }
        };
        let Some(args_group) = bucket.get(index).map(|cached| &cached.group) else {
            return Err(LumaError::Gpu("bind group cache entry vanished".into()));
        };

        // Record. Launches share one compute pass; a profiled launch gets a pass of its own that
        // carries the timestamp writes.
        let profiled = profiler.is_some();
        if profiled {
            *pass = None;
        }
        let encoder = encoder.get_or_insert_with(|| {
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("luma batch") })
        });
        let pass_slot = pass;
        let pass = pass_slot.get_or_insert_with(|| {
            *bound = Bound::default();
            let timestamp_writes = profiler.as_mut().map(|p| p.writes(device, &kernel.label));
            encoder
                .begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes })
                .forget_lifetime()
        });
        if bound.kernel != Some(kernel.id) {
            pass.set_pipeline(&kernel.pipeline);
            bound.kernel = Some(kernel.id);
            bound.args = None;
        }
        let grid = Some((arena.epoch, grid_offset));
        if bound.grid != grid {
            pass.set_bind_group(0, grid_group, &[grid_offset]);
            bound.grid = grid;
        }
        // A group with uniforms is bound at new dynamic offsets by every launch.
        if !offsets.is_empty() || bound.args != Some((key, index)) {
            pass.set_bind_group(1, args_group, offsets.as_slice());
            bound.args = Some((key, index));
        }
        pass.dispatch_workgroups(workgroups[0], workgroups[1], workgroups[2]);
        if profiled {
            *pass_slot = None;
        }
        *dispatches += 1;
        Ok(())
    }

    /// The kernel parameter set chosen for this context's GPU (see [`TuningProfile`]).
    pub fn tuning_profile(&self) -> TuningProfile {
        self.inner.tuning.profile
    }

    pub(crate) fn tuning(&self) -> &Tuning {
        &self.inner.tuning
    }

    /// The features the device was opened with. Custom kernels may use the shader features
    /// among them (`SUBGROUP`, `SHADER_F16`, `EXPERIMENTAL_COOPERATIVE_MATRIX`) when present.
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
        }
        Ok(())
    }

    /// Stops profiling and returns the GPU durations of the launches measured since
    /// [`Context::enable_profiling`], by kernel label. Submits the pending batch and waits for
    /// it, so this is also a sync point.
    ///
    /// Fails with [`LumaError::Profiling`] if profiling is not enabled.
    pub fn finish_profiling(&self) -> Result<Profile> {
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
        Ok(Profile::from_samples(samples))
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
    /// let (sum, profile) = ctx.profile(|| (&x * 2.0).sum())?;
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

/// Size of the `luma` grid uniform in bytes (`LumaGrid`).
const GRID_BYTES: u64 = 16;

/// Copies below this size stay on the calling thread.
const PARALLEL_COPY_MIN: usize = 8 << 20;

/// `dst.copy_from_slice(src)` split over several threads for large copies: one core cannot
/// saturate unified memory, and the page faults of a fresh destination spread out too. The
/// threads store past the cache (see [`copy_streaming`]).
pub(crate) fn copy_parallel(src: &[u8], dst: &mut [std::mem::MaybeUninit<u8>]) {
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get().min(8));
    if src.len() < PARALLEL_COPY_MIN || threads == 1 {
        dst.write_copy_of_slice(src);
        return;
    }
    let chunk = src.len().div_ceil(threads).next_multiple_of(4096);
    std::thread::scope(|scope| {
        for (s, d) in src.chunks(chunk).zip(dst.chunks_mut(chunk)) {
            scope.spawn(move || copy_streaming(s, d));
        }
    });
}

/// `dst.copy_from_slice(src)` with non-temporal stores: the destination lines are not read into
/// the cache before being overwritten, which a large readback (bound by host memory bandwidth
/// while the GPU's transfer writes the next chunk) would pay for with a third of its traffic.
#[cfg(target_arch = "x86_64")]
fn copy_streaming(src: &[u8], dst: &mut [std::mem::MaybeUninit<u8>]) {
    use std::arch::x86_64::{__m128i, _mm_loadu_si128, _mm_sfence, _mm_stream_si128};
    const LINE: usize = 64;
    assert_eq!(src.len(), dst.len(), "source and destination lengths differ");
    let head = dst.as_ptr().align_offset(LINE).min(dst.len());
    let (dst_head, dst_body) = dst.split_at_mut(head);
    let (src_head, src_body) = src.split_at(head);
    dst_head.write_copy_of_slice(src_head);
    let (src_lines, src_tail) = src_body.as_chunks::<LINE>();
    let (dst_lines, dst_tail) = dst_body.as_chunks_mut::<LINE>();
    for (s, d) in src_lines.iter().zip(dst_lines) {
        for part in (0..LINE).step_by(16) {
            // SAFETY: `part + 16 <= LINE`, so both pointers stay inside their 64-byte chunks;
            // the load is unaligned, and `d` is 64-byte aligned (`head`), as the store requires.
            // SSE2 is part of the x86_64 baseline.
            unsafe {
                let value = _mm_loadu_si128(s.as_ptr().add(part).cast::<__m128i>());
                _mm_stream_si128(d.as_mut_ptr().add(part).cast::<__m128i>(), value);
            }
        }
    }
    // SAFETY: SSE is part of the x86_64 baseline. Orders the streaming stores before whatever
    // reads the destination next, on any thread.
    unsafe { _mm_sfence() };
    dst_tail.write_copy_of_slice(src_tail);
}

#[cfg(not(target_arch = "x86_64"))]
fn copy_streaming(src: &[u8], dst: &mut [std::mem::MaybeUninit<u8>]) {
    dst.write_copy_of_slice(src);
}

/// The outcome of a `map_async` request, see [`map_for_read`].
type MapReceiver = mpsc::Receiver<std::result::Result<(), wgpu::BufferAsyncError>>;

/// Requests a read mapping of `slice`; [`Context::wait_map`] waits for it.
fn map_for_read(slice: wgpu::BufferSlice<'_>) -> MapReceiver {
    let (sender, receiver) = mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    receiver
}

/// Backs a large readback destination with transparent huge pages: faulting in, and later
/// freeing, its 4 KiB pages one by one costs about as much as the copy itself.
#[cfg(target_os = "linux")]
fn advise_huge_pages(dst: &mut [std::mem::MaybeUninit<u8>]) {
    const PAGE: usize = 4096;
    if dst.len() < PARALLEL_COPY_MIN {
        return;
    }
    let start = (dst.as_mut_ptr() as usize).next_multiple_of(PAGE);
    let end = (dst.as_mut_ptr() as usize + dst.len()) & !(PAGE - 1);
    // SAFETY: the page-aligned range lies within `dst`; `madvise` only changes how the kernel
    // backs it, never its contents, and failure is harmless.
    unsafe { libc::madvise(start as *mut libc::c_void, end - start, libc::MADV_HUGEPAGE) };
}

#[cfg(not(target_os = "linux"))]
fn advise_huge_pages(_: &mut [std::mem::MaybeUninit<u8>]) {}

/// [`copy_parallel`] into write-only mapped memory.
fn write_parallel(src: &[u8], mut dst: wgpu::WriteOnly<'_, [u8]>) {
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get().min(8));
    if src.len() < PARALLEL_COPY_MIN || threads == 1 {
        dst.copy_from_slice(src);
        return;
    }
    /// A disjoint part of the mapping, handed to one thread.
    struct Part(std::ptr::NonNull<[u8]>);
    // SAFETY: parts are disjoint and outlive the scope the threads run in.
    unsafe impl Send for Part {}

    assert_eq!(src.len(), dst.len(), "source and destination lengths differ");
    let base = dst.as_raw_ptr().cast::<u8>();
    let chunk = src.len().div_ceil(threads).next_multiple_of(4096);
    std::thread::scope(|scope| {
        for (i, s) in src.chunks(chunk).enumerate() {
            // SAFETY: `i * chunk + s.len() <= dst.len()`, so the part lies inside the mapping.
            let part = Part(std::ptr::NonNull::slice_from_raw_parts(unsafe { base.add(i * chunk) }, s.len()));
            scope.spawn(move || {
                let part = part;
                // SAFETY: the part is mapped for writing and no other thread touches it.
                unsafe { wgpu::WriteOnly::new(part.0) }.copy_from_slice(s);
            });
        }
    });
}

fn dynamic_offset(offset: u64) -> Result<u32> {
    u32::try_from(offset).map_err(|_| LumaError::Gpu("uniform arena offset exceeds u32".into()))
}

/// A buffer bound twice to one launch, with at least one writable binding, is a wgpu validation
/// error (usage conflict). Arrays sharing a buffer through `reshape` make that reachable.
fn check_aliasing(args: &[RawArg<'_>]) -> Result<()> {
    for (second, arg) in args.iter().enumerate() {
        let RawArg::Storage { buffer, writable, .. } = arg else { continue };
        for (first, earlier) in args.iter().enumerate().take(second) {
            let RawArg::Storage { buffer: other, writable: other_writable, .. } = earlier else {
                continue;
            };
            if buffer == other && (*writable || *other_writable) {
                return Err(LumaError::KernelArgAlias { first, second });
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::Array;

    #[test]
    fn recycled_buffers_are_reused_per_size_class_and_usage() {
        let ctx = Context::new().unwrap();

        let a = ctx.alloc_buffer(2400, STORAGE_USAGE).unwrap();
        assert_eq!(a.size(), 4096);
        ctx.recycle(a.clone());

        // Same class and usage: the recycled buffer comes back, even for a different byte size.
        let b = ctx.alloc_buffer(4000, STORAGE_USAGE).unwrap();
        assert_eq!(a, b);

        // Different usage or different class: a new buffer.
        ctx.recycle(b.clone());
        assert_ne!(ctx.alloc_buffer(4000, STAGING_USAGE).unwrap(), a);
        assert_ne!(ctx.alloc_buffer(4100, STORAGE_USAGE).unwrap(), a);
        // ... while the recycled one is still waiting in the pool.
        assert_eq!(ctx.alloc_buffer(4000, STORAGE_USAGE).unwrap(), a);
    }

    #[test]
    fn dropping_an_array_returns_its_buffer_to_the_pool() {
        let ctx = Context::new().unwrap();

        let array = Array::<u32>::zeros_on(&ctx, 600).unwrap();
        let storage = array.buffer().unwrap();
        drop(array);

        let reused = Array::<u32>::zeros_on(&ctx, 1000).unwrap();
        assert_eq!(reused.buffer().unwrap(), storage);
    }

    #[test]
    fn dropped_buffer_is_reused_while_its_readers_are_still_pending() {
        let ctx = Context::new().unwrap();

        let a = Array::from_slice_on(&ctx, &[1.0f32, 2.0, 3.0], 3).unwrap();
        let storage = a.buffer().unwrap();
        let doubled = &a * 2.0;
        // Operators are lazy: record the multiplication (without waiting for it) so that it is
        // pending in the batch, reading a's buffer.
        assert!(doubled.ready().is_ok());
        assert!(ctx.pending_dispatches() > 0, "the multiplication must still be pending");
        drop(a);

        // The new array takes over a's buffer, although the multiplication has not run yet. (A
        // zero-copy buffer is only reusable after the next submission.)
        let b = Array::from_slice_on(&ctx, &[10.0f32, 20.0, 30.0], 3).unwrap();
        if !ctx.zero_copy() {
            assert_eq!(b.buffer().unwrap(), storage);
        }
        assert!(ctx.pending_dispatches() > 0);

        assert_eq!(doubled.to_vec().unwrap(), vec![2.0, 4.0, 6.0]);
        assert_eq!(b.to_vec().unwrap(), vec![10.0, 20.0, 30.0]);
    }

    #[test]
    fn flush_submits_the_pending_batch_and_keeps_the_results() {
        let ctx = Context::new().unwrap();
        ctx.flush().unwrap(); // nothing pending
        assert_eq!(ctx.pending_dispatches(), 0);

        let mut x = Array::<f32>::zeros_on(&ctx, 16).unwrap();
        // A view sharing the buffer makes every compound assignment a dispatch of its own.
        let view = x.reshape(16);
        x += 2.0;
        x *= 3.0;
        assert_eq!(ctx.pending_dispatches(), 2);
        ctx.flush().unwrap();
        assert_eq!(ctx.pending_dispatches(), 0);

        x += 1.0; // recorded after the flush, runs after it
        assert_eq!(ctx.pending_dispatches(), 1);
        assert_eq!(view.to_vec().unwrap(), vec![7.0; 16]);
    }

    #[test]
    fn submit_hands_over_one_array_and_leaves_other_lazy_arrays_lazy() {
        let ctx = Context::new().unwrap();
        let x = Array::<f32>::full_on(&ctx, 16, 1.0).unwrap();
        let wanted = &x + 1.0;
        let unrelated = &x * 3.0;
        wanted.submit().unwrap();
        // `wanted` went to the GPU; unlike `flush`, `submit` did not evaluate the other array.
        assert_eq!(ctx.pending_dispatches(), 0);
        assert!(wanted.pending().is_none());
        assert!(unrelated.pending().is_some());
        ctx.flush().unwrap();
        assert!(unrelated.pending().is_none());
        assert_eq!(wanted.to_vec().unwrap(), vec![2.0; 16]);
        assert_eq!(unrelated.to_vec().unwrap(), vec![3.0; 16]);
    }

    #[test]
    fn one_pass_per_batch_and_bound_state_survive_interleaved_commands() {
        // Uploads and clears between launches end the pass; the launches after them start a new
        // one and must bind everything again.
        let ctx = Context::new().unwrap();
        let mut x = Array::<f32>::zeros_on(&ctx, 8).unwrap();
        for i in 0..50 {
            x += 1.0;
            let y = Array::from_slice_on(&ctx, &[i as f32; 8], 8).unwrap();
            x += &y;
            let z = Array::<f32>::zeros_on(&ctx, 8).unwrap();
            x += &z;
        }
        let want: f32 = (0..50).map(|i| 1.0 + i as f32).sum();
        assert_eq!(x.to_vec().unwrap(), vec![want; 8]);
    }

    #[test]
    fn batch_is_submitted_at_the_dispatch_limit() {
        let ctx = Context::new().unwrap();
        let mut x = Array::<f32>::zeros_on(&ctx, 8).unwrap();
        // A second array shares the buffer, so every `+=` is a dispatch of its own.
        let _view = x.reshape(8);
        for i in 0..BATCH_LIMIT * 2 + 5 {
            x += 1.0;
            assert!(ctx.pending_dispatches() < BATCH_LIMIT, "after {i} operations");
        }
        assert_eq!(x.to_vec().unwrap(), vec![(BATCH_LIMIT * 2 + 5) as f32; 8]);
        assert_eq!(ctx.pending_dispatches(), 0);
    }
}
