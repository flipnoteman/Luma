use std::collections::HashMap;
use std::num::NonZeroU64;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use wgpu::{BindGroupEntry, BindingResource, Buffer, BufferBinding, BufferUsages};

use crate::array::Array;
use crate::error::{LumaError, Result};
use crate::gpu::handle;
use crate::gpu::pool::{self, BufferPool};
use crate::gpu_type::GpuType;
use crate::kernel::{KernelBuilder, KernelCore, KernelKey};
use crate::operations::{self, Access, Operation, ShaderResources};

/// Usage of every array storage buffer (and the FFT scratch buffers, so they share a pool class).
pub(crate) const STORAGE_USAGE: BufferUsages = BufferUsages::STORAGE
    .union(BufferUsages::COPY_DST)
    .union(BufferUsages::COPY_SRC);

/// Usage of dims and kernel-parameter buffers.
pub(crate) const PARAMS_USAGE: BufferUsages = BufferUsages::STORAGE.union(BufferUsages::COPY_DST);

/// Usage of uniform buffers (the kernel grid and user uniforms).
pub(crate) const UNIFORM_USAGE: BufferUsages = BufferUsages::UNIFORM.union(BufferUsages::COPY_DST);

/// Usage of readback buffers.
pub(crate) const STAGING_USAGE: BufferUsages = BufferUsages::MAP_READ.union(BufferUsages::COPY_DST);

/// A compute pipeline together with the bind group layout it was created with.
#[derive(Debug)]
pub(crate) struct Pipeline {
    pub(crate) pipeline: wgpu::ComputePipeline,
    pub(crate) layout: wgpu::BindGroupLayout,
}

#[derive(Debug)]
struct Inner {
    device: wgpu::Device,
    queue: wgpu::Queue,
    shaders: ShaderResources,
    /// One pipeline per (operation, element type), created on first use.
    pipelines: Mutex<HashMap<(Operation, &'static str), Arc<Pipeline>>>,
    /// Custom kernels, built on first use of their (source, entry, workgroup).
    kernels: Mutex<HashMap<KernelKey, Arc<KernelCore>>>,
    pool: Mutex<BufferPool>,
    /// Largest buffer that can be allocated *and* bound as a storage buffer.
    max_buffer_bytes: u64,
    max_workgroups_per_dimension: u32,
}

/// Handle to a GPU device and everything needed to run work on it: compiled shaders, cached
/// compute pipelines and a pool of reusable buffers.
///
/// A `Context` is cheap to [`Clone`] (it is reference counted) and can be shared between threads.
/// Every [`Array`] holds a clone of the context it was created on, so the device lives as long as
/// any of its arrays. Arrays can only be combined with arrays of the same context.
///
/// # Example
/// ```no_run
/// let ctx = luma::Context::new()?;
/// let a = ctx.from_slice(&[3, 1, 1, 1], &[1u32, 2, 3])?;
/// let b = a.double()?;
/// assert_eq!(b.read()?, vec![2, 4, 6]);
/// # Ok::<(), luma::LumaError>(())
/// ```
#[derive(Clone, Debug)]
pub struct Context {
    inner: Arc<Inner>,
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    // The guarded state (caches, free lists) stays consistent even if a holder panicked.
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

impl Context {
    /// Opens the default high-performance GPU and compiles the embedded shaders.
    /// Blocks until the device is ready.
    pub fn new() -> Result<Context> {
        let (device, queue) = pollster::block_on(handle::request_device())?;
        let shaders = operations::create_shader_modules(&device);
        let limits = device.limits();
        let max_buffer_bytes = limits
            .max_buffer_size
            .min(u64::from(limits.max_storage_buffer_binding_size));

        Ok(Context {
            inner: Arc::new(Inner {
                device,
                queue,
                shaders,
                pipelines: Mutex::new(HashMap::new()),
                kernels: Mutex::new(HashMap::new()),
                pool: Mutex::new(BufferPool::default()),
                max_buffer_bytes,
                max_workgroups_per_dimension: limits.max_compute_workgroups_per_dimension,
            }),
        })
    }

    /// Creates an array of the given shape and uploads `data` (row-major) into it.
    ///
    /// Every dimension must be non-zero and `data.len()` must equal their product.
    pub fn from_slice<T: GpuType>(&self, shape: &[usize; 4], data: &[T]) -> Result<Array<T>> {
        let mut array = Array::alloc(self, shape)?;
        array.write(data)?;
        Ok(array)
    }

    /// Creates an array of the given shape filled with zeros.
    pub fn zeros<T: GpuType>(&self, shape: &[usize; 4]) -> Result<Array<T>> {
        let array = Array::<T>::alloc(self, shape)?;
        let mut encoder = self
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        // Pooled buffers hold stale data, so zeroing is required.
        encoder.clear_buffer(array.storage(), 0, Some(array.byte_len()));
        self.queue().submit(Some(encoder.finish()));
        Ok(array)
    }

    /// Starts building a custom compute kernel from WGSL source.
    ///
    /// See [`KernelBuilder`] for the options and [`Kernel`](crate::Kernel) for the binding
    /// conventions.
    pub fn kernel(&self, wgsl_src: &str) -> KernelBuilder<'_> {
        KernelBuilder::new(self, wgsl_src)
    }

    pub(crate) fn kernel_cache(&self) -> MutexGuard<'_, HashMap<KernelKey, Arc<KernelCore>>> {
        lock(&self.inner.kernels)
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

    /// `true` if both handles refer to the same device.
    pub(crate) fn same_as(&self, other: &Context) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
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
        Ok(self.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("luma buffer"),
            size: class,
            usage,
            mapped_at_creation: false,
        }))
    }

    /// Returns a buffer from [`Context::alloc_buffer`] to the pool.
    ///
    /// This is safe while submitted GPU work still uses the buffer: later writes to it are
    /// ordered after that work on the queue.
    pub(crate) fn recycle(&self, buffer: Buffer) {
        lock(&self.inner.pool).put(buffer);
    }

    /// Returns the cached pipeline for `op` on elements with `type_suffix`, creating it (and its
    /// bind group layout) on first use.
    fn pipeline(&self, op: Operation, type_suffix: &'static str) -> Result<Arc<Pipeline>> {
        let mut cache = lock(&self.inner.pipelines);
        if let Some(pipeline) = cache.get(&(op, type_suffix)) {
            return Ok(Arc::clone(pipeline));
        }

        let name = operations::shader_name(op, type_suffix);
        let shader = self
            .inner
            .shaders
            .get(&name)
            .ok_or(LumaError::ShaderNotFound(name))?;

        let device = self.device();
        let layout_entries: Vec<wgpu::BindGroupLayoutEntry> = op
            .bindings()
            .iter()
            .enumerate()
            .map(|(binding, access)| wgpu::BindGroupLayoutEntry {
                binding: binding as u32,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage {
                        read_only: *access == Access::ReadOnly,
                    },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            })
            .collect();
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(op.name()),
            entries: &layout_entries,
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(op.name()),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(op.name()),
            layout: Some(&pipeline_layout),
            module: shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let pipeline = Arc::new(Pipeline { pipeline, layout });
        cache.insert((op, type_suffix), Arc::clone(&pipeline));
        Ok(pipeline)
    }

    /// Records one compute pass running `op` into `encoder`.
    ///
    /// `buffers` are bound to `@group(0)` in order and must match [`Operation::bindings`].
    /// `workgroups` is the `(x, y)` dispatch size.
    pub(crate) fn dispatch<const N: usize>(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        op: Operation,
        type_suffix: &'static str,
        buffers: [BufferBinding<'_>; N],
        workgroups: (u32, u32),
    ) -> Result<()> {
        debug_assert_eq!(N, op.bindings().len());
        let pipeline = self.pipeline(op, type_suffix)?;

        let entries: [BindGroupEntry<'_>; N] = std::array::from_fn(|i| BindGroupEntry {
            binding: i as u32,
            resource: BindingResource::Buffer(buffers[i].clone()),
        });
        let bind_group = self.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.layout,
            entries: &entries,
        });

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(workgroups.0, workgroups.1, 1);
        Ok(())
    }
}

/// Binds the first `bytes` bytes of `buffer`. Pooled buffers can be larger than the data they
/// hold; binding only the logical size keeps shaders within it and within the device's
/// storage-binding limit.
pub(crate) fn bind(buffer: &Buffer, bytes: u64) -> BufferBinding<'_> {
    BufferBinding {
        buffer,
        offset: 0,
        size: NonZeroU64::new(bytes),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        assert_ne!(ctx.alloc_buffer(4000, PARAMS_USAGE).unwrap(), a);
        assert_ne!(ctx.alloc_buffer(4100, STORAGE_USAGE).unwrap(), a);
        // ... while the recycled one is still waiting in the pool.
        assert_eq!(ctx.alloc_buffer(4000, STORAGE_USAGE).unwrap(), a);
    }

    #[test]
    fn dropping_an_array_returns_its_buffers_to_the_pool() {
        let ctx = Context::new().unwrap();

        let array = Array::<u32>::alloc(&ctx, &[600, 1, 1, 1]).unwrap();
        let storage = array.storage().clone();
        let dims = array.dims_buffer().clone();
        drop(array);

        let reused = Array::<u32>::alloc(&ctx, &[1000, 1, 1, 1]).unwrap();
        assert_eq!(reused.storage(), &storage);
        assert_eq!(reused.dims_buffer(), &dims);
    }
}
