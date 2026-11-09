use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::num::NonZeroU64;
use std::sync::mpsc;
use std::sync::{Arc, Mutex, MutexGuard, OnceLock, PoisonError};

use wgpu::{BindGroupEntry, BindingResource, Buffer, BufferBinding, BufferUsages};

use crate::builtin::Builtin;
use crate::error::{LumaError, Result};
use crate::jit::JitState;
use crate::gpu::handle;
use crate::gpu::pool::{self, BufferPool};
use crate::kernel::{self, KernelBuilder, KernelCore, KernelKey};

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

/// One argument of a kernel launch, already validated: what to bind at `@group(1)` in order.
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
}

/// One buffer of the uniform arena together with the bytes that will be written into it.
#[derive(Debug)]
struct Chunk {
    buffer: Buffer,
    /// Host copy of the slots handed out so far (`data.len()` is the used size).
    data: Vec<u8>,
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
    }
}

impl Arena {
    fn new(device: &wgpu::Device) -> Self {
        Arena { current: new_chunk(device, ARENA_CHUNK), full: Vec::new() }
    }

    /// Reserves a fresh slot holding `data` and returns its chunk's buffer and the slot's offset.
    /// The bytes reach the GPU in [`Arena::upload`].
    fn alloc(&mut self, device: &wgpu::Device, align: u64, data: &[u8]) -> (Buffer, u64) {
        let padded = (data.len() as u64).next_multiple_of(wgpu::COPY_BUFFER_ALIGNMENT).max(4);
        let slot = padded.next_multiple_of(align);
        if self.current.data.len() as u64 + slot > self.current.buffer.size() {
            let fresh = new_chunk(device, slot.max(ARENA_CHUNK));
            self.full.push(std::mem::replace(&mut self.current, fresh));
        }
        let chunk = &mut self.current;
        let offset = chunk.data.len() as u64;
        chunk.data.extend_from_slice(data);
        chunk.data.resize((offset + slot) as usize, 0);
        (chunk.buffer.clone(), offset)
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

/// Identifies a bind group of `@group(1)`: the kernel and what is bound, in binding order.
/// Uniforms are identified by their arena chunk (the slot is a dynamic offset).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct BindKey {
    kernel: u64,
    entries: Vec<(Buffer, u64)>,
}

/// The commands recorded since the last submission.
#[derive(Debug)]
struct Batch {
    encoder: Option<wgpu::CommandEncoder>,
    dispatches: u32,
    arena: Arena,
    bind_groups: HashMap<BindKey, wgpu::BindGroup>,
    grid_groups: HashMap<Buffer, wgpu::BindGroup>,
}

#[derive(Debug)]
struct Inner {
    device: wgpu::Device,
    queue: wgpu::Queue,
    /// Built-in kernels, built on first use of their (operation, element type).
    builtins: Mutex<HashMap<(Builtin, &'static str), Arc<KernelCore>>>,
    /// Lazy-evaluation state: fused kernels and the pending expressions.
    jit: JitState,
    /// Custom kernels, built on first use of their (source, entry, workgroup).
    kernels: Mutex<HashMap<KernelKey, Arc<KernelCore>>>,
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
/// a few hundred dispatches. Because the whole batch is one ordered stream of commands, no
/// ordering is lost: host data is uploaded through staging buffers *inside* the stream, buffers
/// released by a dropped array can be handed out again immediately (every use of the new owner is
/// recorded after every use of the old one), and every uniform value lives in its own slot until
/// its batch is submitted.
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
            encoder: None,
            dispatches: 0,
            arena: Arena::new(&device),
            bind_groups: HashMap::new(),
            grid_groups: HashMap::new(),
        };

        Ok(Context {
            inner: Arc::new(Inner {
                max_uniform_bytes: limits.max_uniform_buffer_binding_size,
                uniform_align: u64::from(limits.min_uniform_buffer_offset_alignment).max(4),
                max_workgroups_per_dimension: limits.max_compute_workgroups_per_dimension,
                max_dynamic_uniforms: limits.max_dynamic_uniform_buffers_per_pipeline_layout,
                device,
                queue,
                builtins: Mutex::new(HashMap::new()),
                jit: JitState::new(&limits),
                kernels: Mutex::new(HashMap::new()),
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
        // Lazy arrays that are still alive are part of "everything recorded so far".
        self.jit().evaluate_live();
        self.flush()?;
        self.device()
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| LumaError::Poll(Arc::new(e)))?;
        self.take_gpu_error()
    }

    pub(crate) fn jit(&self) -> &JitState {
        &self.inner.jit
    }

    /// Submits the pending batch without waiting for it.
    pub(crate) fn flush(&self) -> Result<()> {
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

    fn flush_locked(&self, batch: &mut Batch) {
        batch.dispatches = 0;
        batch.arena.upload(self.queue());
        if batch.bind_groups.len() > MAX_CACHED_BIND_GROUPS {
            // In-flight work keeps what it uses alive; this only drops our references.
            batch.bind_groups.clear();
            batch.grid_groups.clear();
        }
        if let Some(encoder) = batch.encoder.take() {
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

    pub(crate) fn kernel_cache(&self) -> MutexGuard<'_, HashMap<KernelKey, Arc<KernelCore>>> {
        lock(&self.inner.kernels)
    }

    /// The cached built-in kernel for `op` on elements of WGSL type `ty`, compiled on first use.
    pub(crate) fn builtin(&self, op: Builtin, ty: &'static str) -> Result<Arc<KernelCore>> {
        let mut cache = lock(&self.inner.builtins);
        if let Some(core) = cache.get(&(op, ty)) {
            return Ok(Arc::clone(core));
        }
        let key = KernelKey::new(op.source(ty).into_owned(), "main", None);
        let core = Arc::new(kernel::build_core(self, &key, Some("luma builtin"))?);
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

    fn encoder<'a>(&self, batch: &'a mut Batch) -> &'a mut wgpu::CommandEncoder {
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

        let result = {
            let view = slice.get_mapped_range().map_err(LumaError::MapRange)?;
            view.to_vec()
        };
        staging.unmap();
        self.recycle(staging);
        Ok(result)
    }

    /// Records one compute pass running `kernel` and returns without waiting for it.
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

        let inner = &*self.inner;
        let mut guard = self.batch()?;
        let flush = {
            let Batch { encoder, dispatches, arena, bind_groups, grid_groups } = &mut *guard;
            let device = &inner.device;
            let align = inner.uniform_align;

            // Uniform values first: they go into fresh arena slots.
            let (grid_chunk, grid_offset) =
                arena.alloc(device, align, bytemuck::bytes_of(&[size[0], size[1], size[2], 0u32]));
            let mut entries: Vec<(Buffer, u64)> = Vec::with_capacity(args.len());
            let mut offsets: Vec<u32> = Vec::new();
            for arg in args {
                match arg {
                    RawArg::Storage { buffer, bytes, .. } => entries.push(((*buffer).clone(), *bytes)),
                    RawArg::Uniform(data) => {
                        let (chunk, offset) = arena.alloc(device, align, data);
                        offsets.push(dynamic_offset(offset)?);
                        entries.push((chunk, data.len() as u64));
                    }
                }
            }
            let grid_offset = dynamic_offset(grid_offset)?;

            let args_group = match bind_groups.entry(BindKey { kernel: kernel.id, entries }) {
                Entry::Occupied(entry) => entry.into_mut(),
                Entry::Vacant(entry) => {
                    let group_entries: Vec<BindGroupEntry<'_>> = entry
                        .key()
                        .entries
                        .iter()
                        .enumerate()
                        .map(|(binding, (buffer, bytes))| BindGroupEntry {
                            binding: binding as u32,
                            resource: BindingResource::Buffer(BufferBinding {
                                buffer,
                                offset: 0,
                                size: NonZeroU64::new(*bytes),
                            }),
                        })
                        .collect();
                    let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: None,
                        layout: &kernel.args_layout,
                        entries: &group_entries,
                    });
                    entry.insert(group)
                }
            };
            let grid_group = match grid_groups.entry(grid_chunk) {
                Entry::Occupied(entry) => entry.into_mut(),
                Entry::Vacant(entry) => {
                    let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: None,
                        layout: &inner.grid_layout,
                        entries: &[BindGroupEntry {
                            binding: 0,
                            resource: BindingResource::Buffer(BufferBinding {
                                buffer: entry.key(),
                                offset: 0,
                                size: NonZeroU64::new(GRID_BYTES),
                            }),
                        }],
                    });
                    entry.insert(group)
                }
            };

            let encoder = encoder.get_or_insert_with(|| {
                device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("luma batch"),
                })
            });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: None,
                    timestamp_writes: None,
                });
                pass.set_pipeline(&kernel.pipeline);
                pass.set_bind_group(0, &*grid_group, &[grid_offset]);
                pass.set_bind_group(1, &*args_group, &offsets);
                pass.dispatch_workgroups(workgroups[0], workgroups[1], workgroups[2]);
            }
            *dispatches += 1;
            *dispatches >= BATCH_LIMIT
        };
        if flush {
            self.flush_locked(&mut guard);
        }
        Ok(())
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
        // Operators are lazy: record the multiplication (without waiting for it) so that it is
        // pending in the batch, reading a's buffer.
        assert!(doubled.ready().is_ok());
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
