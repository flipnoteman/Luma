use std::marker::PhantomData;
use std::sync::mpsc;

use crate::context::{Context, PARAMS_USAGE, STAGING_USAGE, STORAGE_USAGE};
use crate::error::{LumaError, Result};
use crate::gpu::pipeline;
use crate::gpu_type::{GpuType, ScalarType};
use crate::operations::Operation;

/// Size of the dims buffer: the shape as `array<u32, 4>`.
pub(crate) const DIMS_BYTES: u64 = std::mem::size_of::<[u32; 4]>() as u64;

/// Checks `shape` against what the GPU kernels support and returns it as the `array<u32, 4>`
/// they read, together with the total element count.
///
/// Every dimension must be non-zero, and every dimension and the element count must fit in `u32`.
pub(crate) fn validate_shape(shape: &[usize; 4]) -> Result<([u32; 4], usize)> {
    let too_large = || LumaError::DimensionTooLarge { shape: *shape };
    let mut dims = [0u32; 4];
    let mut len = 1u32;
    for (dim, &extent) in dims.iter_mut().zip(shape) {
        if extent == 0 {
            return Err(LumaError::InvalidShape(format!(
                "Shape {shape:?} has a zero-sized dimension"
            )));
        }
        *dim = u32::try_from(extent).map_err(|_| too_large())?;
        len = len.checked_mul(*dim).ok_or_else(too_large)?;
    }
    Ok((dims, len as usize))
}

/// GPU-resident array. The data lives in a GPU buffer owned by the array and only comes back to
/// the host through [`Array::read`], the one operation that waits for the GPU.
///
/// Arrays are created from a [`Context`]. Operations return new arrays and are submitted to the
/// GPU without waiting; the GPU executes them in submission order. Dropping an array returns
/// its buffer to the context's pool for reuse.
///
/// # Example
/// ```no_run
/// let ctx = luma::Context::new()?;
/// let a = ctx.from_slice(&[3, 1, 1, 1], &[1u32, 2, 3])?;
/// let b = a.double()?;
/// let result = b.read()?; // [2, 4, 6]
/// # Ok::<(), luma::LumaError>(())
/// ```
#[derive(Debug)]
pub struct Array<T: GpuType> {
    ctx: Context,
    /// May be larger than `len * size_of::<T>()`; only the first `byte_len()` bytes are the array.
    storage: wgpu::Buffer,
    /// The shape as `array<u32, 4>`, read by the elementwise kernels to get the element count.
    dims: wgpu::Buffer,
    shape: [usize; 4],
    len: usize,
    _marker: PhantomData<T>,
}

impl<T: GpuType> Drop for Array<T> {
    fn drop(&mut self) {
        // Cloning a wgpu buffer only bumps a reference count; it is still the same buffer.
        self.ctx.recycle(self.storage.clone());
        self.ctx.recycle(self.dims.clone());
    }
}

impl<T: GpuType> Array<T> {
    /// Allocates an array of `shape` whose contents are unspecified (the dims are initialized).
    pub(crate) fn alloc(ctx: &Context, shape: &[usize; 4]) -> Result<Self> {
        let (dims, len) = validate_shape(shape)?;
        let bytes = len as u64 * std::mem::size_of::<T>() as u64;

        let storage = ctx.alloc_buffer(bytes, STORAGE_USAGE)?;
        let dims_buffer = ctx.alloc_buffer(DIMS_BYTES, PARAMS_USAGE)?;
        ctx.queue()
            .write_buffer(&dims_buffer, 0, bytemuck::cast_slice(&dims));

        Ok(Array {
            ctx: ctx.clone(),
            storage,
            dims: dims_buffer,
            shape: *shape,
            len,
            _marker: PhantomData,
        })
    }

    /// The [`Context`] this array lives on.
    pub fn context(&self) -> &Context {
        &self.ctx
    }

    pub(crate) fn storage(&self) -> &wgpu::Buffer {
        &self.storage
    }

    pub(crate) fn dims_buffer(&self) -> &wgpu::Buffer {
        &self.dims
    }

    /// Size of the array's data in bytes (not the size of the possibly larger buffer).
    pub(crate) fn byte_len(&self) -> u64 {
        self.len as u64 * std::mem::size_of::<T>() as u64
    }

    /// The shape of the array.
    pub fn shape(&self) -> &[usize; 4] {
        &self.shape
    }

    /// Total number of elements (the product of the shape).
    pub fn len(&self) -> usize {
        self.len
    }

    /// Copies the array back to the host. Blocks until all GPU work submitted so far, including
    /// the operations this array was computed by, has finished. This is the only way data leaves
    /// the GPU.
    pub fn read(&self) -> Result<Vec<T>> {
        let bytes = self.byte_len();
        let device = self.ctx.device();
        let staging = self.ctx.alloc_buffer(bytes, STAGING_USAGE)?;

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.copy_buffer_to_buffer(&self.storage, 0, &staging, 0, bytes);
        self.ctx.queue().submit(Some(encoder.finish()));

        let slice = staging.slice(..bytes);
        let (sender, receiver) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(LumaError::Poll)?;
        // The callback runs on whichever thread polls the device first, so it may still be
        // in flight on another thread after our poll returns; wait for it.
        receiver
            .recv()
            .map_err(|_| LumaError::Map(wgpu::BufferAsyncError))?
            .map_err(LumaError::Map)?;

        let result = {
            let view = slice.get_mapped_range().map_err(LumaError::MapRange)?;
            bytemuck::pod_collect_to_vec::<u8, T>(&view)
        };
        staging.unmap();
        self.ctx.recycle(staging);
        Ok(result)
    }

    /// Overwrites the array's contents with `data`, which must have exactly [`Array::len`]
    /// elements. Like operations, the upload is queued and does not wait for the GPU.
    pub fn write(&mut self, data: &[T]) -> Result<()> {
        if data.len() != self.len {
            return Err(LumaError::LengthMismatch {
                expected: self.len,
                actual: data.len(),
            });
        }
        self.ctx
            .queue()
            .write_buffer(&self.storage, 0, bytemuck::cast_slice(data));
        Ok(())
    }

}

impl<T: ScalarType> Array<T> {
    // -- Unary operations --

    pub fn double(&self) -> Result<Array<T>> {
        pipeline::unary(self, Operation::DOUBLE)
    }

    // -- Binary operations (array-array) --

    pub fn add(&self, rhs: &Array<T>) -> Result<Array<T>> {
        self.binary_op(rhs, Operation::ADD)
    }

    pub fn subtract(&self, rhs: &Array<T>) -> Result<Array<T>> {
        self.binary_op(rhs, Operation::SUBTRACT)
    }

    pub fn multiply(&self, rhs: &Array<T>) -> Result<Array<T>> {
        self.binary_op(rhs, Operation::MULTIPLY)
    }

    pub fn divide(&self, rhs: &Array<T>) -> Result<Array<T>> {
        self.binary_op(rhs, Operation::DIVIDE)
    }

    // -- Scalar operations --

    pub fn add_scalar(&self, scalar: T) -> Result<Array<T>> {
        self.scalar_op(scalar, Operation::ADD)
    }

    pub fn subtract_scalar(&self, scalar: T) -> Result<Array<T>> {
        self.scalar_op(scalar, Operation::SUBTRACT)
    }

    pub fn multiply_scalar(&self, scalar: T) -> Result<Array<T>> {
        self.scalar_op(scalar, Operation::MULTIPLY)
    }

    pub fn divide_scalar(&self, scalar: T) -> Result<Array<T>> {
        self.scalar_op(scalar, Operation::DIVIDE)
    }

    // -- Internal helpers --

    fn binary_op(&self, rhs: &Array<T>, op: Operation) -> Result<Array<T>> {
        if self.shape != rhs.shape {
            return Err(LumaError::ShapeMismatch {
                lhs: self.shape,
                rhs: rhs.shape,
            });
        }
        pipeline::binary(self, rhs, op)
    }

    /// Applies `op` with a single-element right-hand side, broadcast over the whole array.
    fn scalar_op(&self, scalar: T, op: Operation) -> Result<Array<T>> {
        let rhs = self.ctx.from_slice(&[1, 1, 1, 1], &[scalar])?;
        pipeline::binary(self, &rhs, op)
    }
}

impl Array<f32> {
    /// 1D convolution with the given kernel, stride, and zero-padding.
    ///
    /// Output length = floor((input_len - kernel_len + 2 * padding) / stride) + 1
    pub fn conv1d(&self, kernel: &Array<f32>, stride: usize, padding: usize) -> Result<Array<f32>> {
        pipeline::conv1d(self, kernel, stride, padding)
    }

    /// FFT-based 1D convolution. Returns the full linear convolution of
    /// length `input_len + kernel_len - 1`. No stride or padding — those
    /// are spatial-domain concepts; use `conv1d()` for that.
    ///
    /// This is O(n log n) and faster than spatial `conv1d` for large kernels.
    pub fn conv1d_fft(&self, kernel: &Array<f32>) -> Result<Array<f32>> {
        pipeline::conv1d_fft(self, kernel)
    }
}
