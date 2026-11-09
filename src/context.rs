use std::hash::{Hash, Hasher};
use std::sync::mpsc;
use std::sync::{Arc, Mutex, MutexGuard, OnceLock, PoisonError};
use std::time::Duration;

use wgpu::{BindGroupEntry, BindingResource, Buffer, BufferBinding, BufferUsages};

use crate::builtin::Builtin;
use crate::error::{LumaError, Result};
use crate::gpu::handle;
use crate::gpu::hash::{FxHasher, FxMap};
use crate::gpu::pool::{self, BufferPool};
use crate::kernel::{self, KernelBuilder, KernelCore, KernelKey};
use crate::profile::{Profile, Profiler, TIMESTAMP_BYTES};

/// Usage of every array storage buffer (and the scratch buffers of multi-pass operations, so they
/// share a pool class).
pub(crate) const STORAGE_USAGE: BufferUsages = BufferUsages::STORAGE
    .union(BufferUsages::COPY_DST)
    .union(BufferUsages::COPY_SRC);

/// Usage of the buffers uniform values are written into.
const UNIFORM_USAGE: BufferUsages = BufferUsages::UNIFORM.union(BufferUsages::COPY_DST);

/// Usage of readback buffers.
const STAGING_USAGE: BufferUsages = BufferUsages::MAP_READ.union(BufferUsages::COPY_DST);

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
/// the data is written (`Queue::write_buffer`, once per chunk when the batch is submitted) before
/// the batch's commands run. Slots are only reused after the batch holding them was submitted; a
/// write into a reused slot is ordered after that submission on the queue.
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

    /// Writes the slots of every chunk to the GPU and empties the arena for the next batch,
    /// keeping the current chunk. Must run right before the batch is submitted: the writes are
    /// executed ahead of the submission's commands, and after everything submitted earlier.
    fn upload(&mut self, queue: &wgpu::Queue) {
        for chunk in self.full.iter().chain(std::iter::once(&self.current)) {
            if !chunk.data.is_empty() {
                queue.write_buffer(&chunk.buffer, 0, &chunk.data);
            }
        }
        self.full.clear();
        self.current.data.clear();
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
    /// Present while profiling is enabled.
    profiler: Option<Profiler>,
}

#[derive(Debug)]
struct Inner {
    device: wgpu::Device,
    queue: wgpu::Queue,
    /// Built-in kernels, built on first use of their (operation, element type).
    builtins: Mutex<FxMap<(Builtin, &'static str), Arc<KernelCore>>>,
    /// Custom kernels, built on first use of their (source, entry, workgroup).
    kernels: Mutex<FxMap<KernelKey, Arc<KernelCore>>>,
    pool: Mutex<BufferPool>,
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
}

/// Handle to a GPU device and everything needed to run work on it: compiled kernels, a pool of
/// reusable buffers and the batch of commands waiting to be submitted.
///
/// A `Context` is cheap to [`Clone`] (it is reference counted) and can be shared between threads.
/// Every [`Array`](crate::Array) holds a clone of the context it was created on, so the device
/// lives as long as any of its arrays. Arrays can only be combined with arrays of the same
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

impl Context {
    /// Opens the default high-performance GPU. Blocks until the device is ready.
    pub fn new() -> Result<Context> {
        let (device, queue) = pollster::block_on(handle::request_device())?;
        let limits = device.limits();
        let max_buffer_bytes = limits
            .max_buffer_size
            .min(limits.max_storage_buffer_binding_size);

        // wgpu panics on errors nobody handles. Record the first one instead and report it as an
        // `Err` at the next synchronization point.
        let gpu_error = Arc::new(Mutex::new(None));
        let sink = Arc::clone(&gpu_error);
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
            profiler: None,
        };

        Ok(Context {
            inner: Arc::new(Inner {
                max_uniform_bytes: limits.max_uniform_buffer_binding_size,
                uniform_align: u64::from(limits.min_uniform_buffer_offset_alignment).max(4),
                max_workgroups_per_dimension: limits.max_compute_workgroups_per_dimension,
                max_dynamic_uniforms: limits.max_dynamic_uniform_buffers_per_pipeline_layout,
                timestamps: device.features().contains(wgpu::Features::TIMESTAMP_QUERY),
                device,
                queue,
                builtins: Mutex::new(FxMap::default()),
                kernels: Mutex::new(FxMap::default()),
                pool: Mutex::new(BufferPool::default()),
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
        self.device()
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| LumaError::Poll(Arc::new(e)))?;
        self.take_gpu_error()
    }

    /// Submits the pending batch to the GPU and returns without waiting for it, so the GPU
    /// starts working while the host keeps recording. Everything recorded afterwards runs after
    /// it, in order. Does nothing if nothing is pending.
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
    fn flush_locked(&self, batch: &mut Batch) {
        // The open pass has to end before the encoder can be finished.
        batch.pass = None;
        batch.bound = Bound::default();
        batch.dispatches = 0;
        // The slots of the arena are reused by the next batch.
        batch.grid_slots.clear();
        batch.arena.upload(self.queue());
        if batch.bind_groups.len > MAX_CACHED_BIND_GROUPS {
            // In-flight work keeps what it uses alive; this only drops our references.
            batch.bind_groups = BindGroupCache::default();
        }
        if let Some(mut encoder) = batch.encoder.take() {
            if let Some(profiler) = &mut batch.profiler {
                profiler.resolve(self.device(), &mut encoder);
            }
            self.queue().submit(Some(encoder.finish()));
        }
    }

    pub(crate) fn device(&self) -> &wgpu::Device {
        &self.inner.device
    }

    pub(crate) fn queue(&self) -> &wgpu::Queue {
        &self.inner.queue
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
        let key = KernelKey::new(op.source(ty).into_owned(), "main", None);
        let label = format!("{op:?} {ty}");
        let core = Arc::new(kernel::build_core(self, &key, Some(&label))?);
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
        let scope = self.device().push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        let buffer = self.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("luma buffer"),
            size: class,
            usage,
            mapped_at_creation: false,
        });
        if pollster::block_on(scope.pop()).is_some() {
            return Err(LumaError::OutOfMemory { requested: class });
        }
        Ok(buffer)
    }

    /// Returns a buffer from [`Context::alloc_buffer`] to the pool.
    ///
    /// This is safe while recorded or submitted GPU work still uses the buffer. All of it is
    /// part of the one ordered command stream, and whoever takes the buffer next (to upload into
    /// it, to run a kernel on it) records that after everything recorded so far. Nothing writes
    /// to pooled buffers outside the stream (`Queue::write_buffer` is only used for uniform
    /// slots, which are never pooled).
    pub(crate) fn recycle(&self, buffer: Buffer) {
        lock(&self.inner.pool).put(buffer);
    }

    /// Records a copy of `data` into the start of `dst`, ordered after everything recorded so
    /// far. `data.len()` must be a multiple of 4.
    pub(crate) fn upload(&self, dst: &Buffer, data: &[u8]) -> Result<()> {
        let staging = self.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("luma upload"),
            size: data.len() as u64,
            usage: BufferUsages::COPY_SRC,
            mapped_at_creation: true,
        });
        {
            let mut view = staging.get_mapped_range_mut(..).map_err(LumaError::MapRange)?;
            view.copy_from_slice(data);
        }
        staging.unmap();

        let mut guard = self.batch()?;
        let encoder = self.encoder(&mut guard);
        encoder.copy_buffer_to_buffer(&staging, 0, dst, 0, data.len() as u64);
        Ok(())
    }

    /// Records zeroing the first `bytes` bytes of `buffer`.
    pub(crate) fn clear(&self, buffer: &Buffer, bytes: u64) -> Result<()> {
        let mut guard = self.batch()?;
        self.encoder(&mut guard).clear_buffer(buffer, 0, Some(bytes));
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

    /// Records a copy of the first `bytes` bytes of `buffer`, submits the batch and waits for the
    /// bytes to arrive.
    pub(crate) fn read_bytes(&self, buffer: &Buffer, bytes: u64) -> Result<Vec<u8>> {
        let staging = self.alloc_buffer(bytes, STAGING_USAGE)?;
        {
            let mut guard = self.batch()?;
            self.encoder(&mut guard).copy_buffer_to_buffer(buffer, 0, &staging, 0, bytes);
            self.flush_locked(&mut guard);
        }
        let result = self.map_read(&staging, bytes)?;
        self.recycle(staging);
        Ok(result)
    }

    /// Waits for the GPU and returns the first `bytes` bytes of `staging`, a `MAP_READ` buffer
    /// whose contents were copied in by a submitted batch.
    fn map_read(&self, staging: &Buffer, bytes: u64) -> Result<Vec<u8>> {
        let slice = staging.slice(..bytes);
        let (sender, receiver) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        self.device()
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| LumaError::Poll(Arc::new(e)))?;
        // The callback runs on whichever thread polls the device first, so it may still be
        // in flight on another thread after our poll returns; wait for it.
        let mapped = receiver
            .recv()
            .map_err(|_| LumaError::Map(wgpu::BufferAsyncError))?
            .map_err(LumaError::Map);
        // An error the GPU raised while running the batch explains a failed map better.
        self.take_gpu_error()?;
        mapped?;

        let result = slice.get_mapped_range().map(|view| view.to_vec());
        staging.unmap();
        result.map_err(LumaError::MapRange)
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
        if guard.profiler.as_ref().is_some_and(Profiler::is_full) {
            self.flush_locked(&mut guard);
        }
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
        let device = &inner.device;
        let align = inner.uniform_align;
        let Batch { encoder, pass, bound, dispatches, arena, grid_slots, bind_groups, offsets, profiler } =
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
            let timestamp_writes = profiler.as_mut().map(|p| p.writes(&kernel.label));
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
            batch.profiler = Some(Profiler::new(self.device()));
        }
        Ok(())
    }

    /// Stops profiling and returns the GPU durations of the launches measured since
    /// [`Context::enable_profiling`], by kernel label. Submits the pending batch and waits for
    /// it, so this is also a sync point.
    ///
    /// Fails with [`LumaError::Profiling`] if profiling is not enabled.
    pub fn finish_profiling(&self) -> Result<Profile> {
        let resolved = {
            let mut batch = self.batch()?;
            if batch.profiler.is_none() {
                return Err(LumaError::Profiling("profiling is not enabled".into()));
            }
            self.flush_locked(&mut batch);
            match batch.profiler.take() {
                Some(profiler) => profiler.resolved,
                None => Vec::new(),
            }
        };
        self.sync()?;

        let period = f64::from(self.queue().get_timestamp_period());
        let mut samples = Vec::new();
        for batch in resolved {
            let bytes = self.map_read(&batch.staging, batch.labels.len() as u64 * 2 * TIMESTAMP_BYTES)?;
            let stamps = bytemuck::pod_collect_to_vec::<u8, u64>(&bytes);
            for (label, &[begin, end]) in batch.labels.iter().zip(stamps.as_chunks::<2>().0) {
                let nanos = end.saturating_sub(begin) as f64 * period;
                samples.push((Arc::clone(label), Duration::from_secs_f64(nanos * 1e-9)));
            }
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
        let storage = array.buffer().unwrap().clone();
        drop(array);

        let reused = Array::<u32>::zeros_on(&ctx, 1000).unwrap();
        assert_eq!(reused.buffer().unwrap(), &storage);
    }

    #[test]
    fn dropped_buffer_is_reused_while_its_readers_are_still_pending() {
        let ctx = Context::new().unwrap();

        let a = Array::from_slice_on(&ctx, &[1.0f32, 2.0, 3.0], 3).unwrap();
        let storage = a.buffer().unwrap().clone();
        let doubled = &a * 2.0;
        assert!(ctx.pending_dispatches() > 0, "the multiplication must still be pending");
        drop(a);

        // The new array takes over a's buffer, although the multiplication has not run yet.
        let b = Array::from_slice_on(&ctx, &[10.0f32, 20.0, 30.0], 3).unwrap();
        assert_eq!(b.buffer().unwrap(), &storage);
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
        x += 2.0;
        x *= 3.0;
        assert_eq!(ctx.pending_dispatches(), 2);
        ctx.flush().unwrap();
        assert_eq!(ctx.pending_dispatches(), 0);

        x += 1.0; // recorded after the flush, runs after it
        assert_eq!(ctx.pending_dispatches(), 1);
        assert_eq!(x.to_vec().unwrap(), vec![7.0; 16]);
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
        for i in 0..BATCH_LIMIT * 2 + 5 {
            x += 1.0;
            assert!(ctx.pending_dispatches() < BATCH_LIMIT, "after {i} operations");
        }
        assert_eq!(x.to_vec().unwrap(), vec![(BATCH_LIMIT * 2 + 5) as f32; 8]);
        assert_eq!(ctx.pending_dispatches(), 0);
    }
}
