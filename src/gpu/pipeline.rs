use crate::array::{Array, DIMS_BYTES};
use crate::context::{bind, Context, PARAMS_USAGE, STORAGE_USAGE};
use crate::error::{LumaError, Result};
use crate::gpu_type::GpuType;
use crate::operations::Operation;

/// Workgroup size used by every shader (`@workgroup_size(64)`).
const WORKGROUP_SIZE: u32 = 64;

/// Splits `element_count` elements into a 2D dispatch that stays within the
/// per-dimension workgroup limit. Shaders linearize with `gid.x + gid.y * nwg.x * 64`.
fn dispatch_dims(element_count: u32, max_per_dim: u32) -> (u32, u32) {
    let groups = element_count.div_ceil(WORKGROUP_SIZE).max(1);
    let x = groups.min(max_per_dim);
    let y = groups.div_ceil(x);
    (x, y)
}

/// 1D dispatch for shaders that index with `global_invocation_id.x` only.
fn dispatch_1d(element_count: u32) -> (u32, u32) {
    (element_count.div_ceil(WORKGROUP_SIZE), 1)
}

fn new_encoder(ctx: &Context) -> wgpu::CommandEncoder {
    ctx.device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None })
}

fn same_context<T: GpuType, U: GpuType>(a: &Array<T>, b: &Array<U>) -> Result<()> {
    if a.context().same_as(b.context()) {
        Ok(())
    } else {
        Err(LumaError::ContextMismatch)
    }
}

/// Temporary buffers of one operation (scratch space, kernel parameters). Taken from the
/// context's pool and handed back when this guard drops, i.e. right after the operation's
/// commands were submitted (or it failed). That is safe while the GPU still runs them: any
/// later reuse of a buffer is ordered after the work already submitted.
struct Scratch<'a> {
    ctx: &'a Context,
    held: Vec<wgpu::Buffer>,
}

impl<'a> Scratch<'a> {
    fn new(ctx: &'a Context) -> Self {
        Scratch { ctx, held: Vec::new() }
    }

    fn take(&mut self, bytes: u64, usage: wgpu::BufferUsages) -> Result<wgpu::Buffer> {
        let buffer = self.ctx.alloc_buffer(bytes, usage)?;
        self.held.push(buffer.clone());
        Ok(buffer)
    }

    /// A scratch buffer of `bytes` bytes with unspecified contents.
    fn storage(&mut self, bytes: u64) -> Result<wgpu::Buffer> {
        self.take(bytes, STORAGE_USAGE)
    }

    /// A buffer holding `words`, returned with the number of bytes to bind. Every call hands out a
    /// distinct buffer, so a single submission can use one parameter buffer per pass.
    fn params(&mut self, words: &[u32]) -> Result<(wgpu::Buffer, u64)> {
        let bytes = std::mem::size_of_val(words) as u64;
        let buffer = self.take(bytes, PARAMS_USAGE)?;
        self.ctx
            .queue()
            .write_buffer(&buffer, 0, bytemuck::cast_slice(words));
        Ok((buffer, bytes))
    }
}

impl Drop for Scratch<'_> {
    fn drop(&mut self) {
        for buffer in self.held.drain(..) {
            self.ctx.recycle(buffer);
        }
    }
}

/// Runs a unary operation on the GPU into a new array. The source array is untouched.
pub(crate) fn unary<T: GpuType>(src: &Array<T>, operation: Operation) -> Result<Array<T>> {
    let ctx = src.context();
    let out = Array::<T>::alloc(ctx, src.shape())?;
    let bytes = src.byte_len();
    let groups = dispatch_dims(src.len() as u32, ctx.max_workgroups_per_dimension());

    // The kernel works in place, so copy the source into the output first.
    let mut encoder = new_encoder(ctx);
    encoder.copy_buffer_to_buffer(src.storage(), 0, out.storage(), 0, bytes);
    ctx.dispatch(
        &mut encoder,
        operation,
        T::type_suffix(),
        [bind(out.storage(), bytes), bind(out.dims_buffer(), DIMS_BYTES)],
        groups,
    )?;
    ctx.queue().submit(Some(encoder.finish()));

    Ok(out)
}

/// Runs a binary operation on the GPU into a new array shaped like `lhs`. Both operands are
/// untouched. `rhs` is either shaped like `lhs` or a single element broadcast over `lhs`.
pub(crate) fn binary<T: GpuType>(
    lhs: &Array<T>,
    rhs: &Array<T>,
    operation: Operation,
) -> Result<Array<T>> {
    same_context(lhs, rhs)?;
    let ctx = lhs.context();
    let out = Array::<T>::alloc(ctx, lhs.shape())?;
    let bytes = lhs.byte_len();
    let groups = dispatch_dims(lhs.len() as u32, ctx.max_workgroups_per_dimension());

    // The kernel works in place on its output, so copy lhs into it first.
    let mut encoder = new_encoder(ctx);
    encoder.copy_buffer_to_buffer(lhs.storage(), 0, out.storage(), 0, bytes);
    ctx.dispatch(
        &mut encoder,
        operation,
        T::type_suffix(),
        [
            bind(out.storage(), bytes),
            bind(rhs.storage(), rhs.byte_len()),
            bind(out.dims_buffer(), DIMS_BYTES),
            bind(rhs.dims_buffer(), DIMS_BYTES),
        ],
        groups,
    )?;
    ctx.queue().submit(Some(encoder.finish()));

    Ok(out)
}

fn check_kernel(kernel: &Array<f32>) -> Result<()> {
    if kernel.shape()[1..] != [1, 1, 1] {
        return Err(LumaError::InvalidShape("Kernel must be 1D".into()));
    }
    Ok(())
}

fn to_u32(value: usize, what: &str) -> Result<u32> {
    u32::try_from(value)
        .map_err(|_| LumaError::InvalidShape(format!("{what} {value} exceeds u32::MAX")))
}

/// Runs a 1D convolution on the GPU (f32 only) into a new array.
pub(crate) fn conv1d(
    input: &Array<f32>,
    kernel: &Array<f32>,
    stride: usize,
    padding: usize,
) -> Result<Array<f32>> {
    same_context(input, kernel)?;
    check_kernel(kernel)?;
    let ctx = input.context();

    let input_len = input.len();
    let kernel_len = kernel.len();
    if stride == 0 {
        return Err(LumaError::InvalidShape("Stride must be non-zero".into()));
    }
    let padded_len = padding
        .checked_mul(2)
        .and_then(|doubled| doubled.checked_add(input_len))
        .ok_or_else(|| LumaError::InvalidShape("Padding is too large".into()))?;
    if padded_len < kernel_len {
        return Err(LumaError::InvalidShape("Kernel is larger than padded input".into()));
    }
    let output_len = (padded_len - kernel_len) / stride + 1;

    let output = Array::<f32>::alloc(ctx, &[output_len, 1, 1, 1])?;

    // Buffers can be larger than their arrays, so the kernel gets every length explicitly.
    let mut scratch = Scratch::new(ctx);
    let (params, params_bytes) = scratch.params(&[
        kernel_len as u32,
        to_u32(stride, "Stride")?,
        to_u32(padding, "Padding")?,
        input_len as u32,
        output_len as u32,
    ])?;

    let mut encoder = new_encoder(ctx);
    ctx.dispatch(
        &mut encoder,
        Operation::CONV1D,
        f32::type_suffix(),
        [
            bind(output.storage(), output.byte_len()),
            bind(input.storage(), input.byte_len()),
            bind(kernel.storage(), kernel.byte_len()),
            bind(&params, params_bytes),
        ],
        dispatch_1d(output_len as u32),
    )?;
    ctx.queue().submit(Some(encoder.finish()));

    Ok(output)
}

/// Runs an FFT-based 1D convolution on the GPU (f32 only) into a new array.
/// The result is the full linear convolution, of length `input_len + kernel_len - 1`.
pub(crate) fn conv1d_fft(input: &Array<f32>, kernel: &Array<f32>) -> Result<Array<f32>> {
    same_context(input, kernel)?;
    check_kernel(kernel)?;
    let ctx = input.context();

    let input_len = input.len() as u32;
    let kernel_len = kernel.len() as u32;
    let output_len = input.len() + kernel.len() - 1;
    let n = u32::try_from(output_len)
        .ok()
        .and_then(u32::checked_next_power_of_two)
        .ok_or(LumaError::DimensionTooLarge { shape: [output_len, 1, 1, 1] })?;
    let log2n = n.trailing_zeros();

    let output = Array::<f32>::alloc(ctx, &[output_len, 1, 1, 1])?;

    let mut scratch = Scratch::new(ctx);

    // Complex buffers (interleaved re/im, so 2*N f32s each). Every pass writes all N entries of
    // the buffer it produces, so stale pool contents never leak into the result.
    let complex_bytes = u64::from(n) * 2 * 4;
    let input_fft = scratch.storage(complex_bytes)?;
    let kernel_fft = scratch.storage(complex_bytes)?;
    let product = scratch.storage(complex_bytes)?;

    let suffix = f32::type_suffix();
    let n_groups = dispatch_1d(n);
    let half_n_groups = dispatch_1d(n / 2);
    let mut encoder = new_encoder(ctx);

    // --- Pass 1: Bit-reverse input ---
    let (params, params_bytes) = scratch.params(&[n, input_len])?;
    ctx.dispatch(
        &mut encoder,
        Operation::FFT_BIT_REVERSE,
        suffix,
        [
            bind(&input_fft, complex_bytes),
            bind(input.storage(), input.byte_len()),
            bind(&params, params_bytes),
        ],
        n_groups,
    )?;

    // --- Pass 2: Bit-reverse kernel ---
    let (params, params_bytes) = scratch.params(&[n, kernel_len])?;
    ctx.dispatch(
        &mut encoder,
        Operation::FFT_BIT_REVERSE,
        suffix,
        [
            bind(&kernel_fft, complex_bytes),
            bind(kernel.storage(), kernel.byte_len()),
            bind(&params, params_bytes),
        ],
        n_groups,
    )?;

    // --- Pass 3 + 4: FFT butterfly on input, then kernel (log2(N) stages each) ---
    for fft_buffer in [&input_fft, &kernel_fft] {
        for stage in 0..log2n {
            let (params, params_bytes) = scratch.params(&[n, stage])?;
            ctx.dispatch(
                &mut encoder,
                Operation::FFT_BUTTERFLY,
                suffix,
                [bind(fft_buffer, complex_bytes), bind(&params, params_bytes)],
                half_n_groups,
            )?;
        }
    }

    // --- Pass 5: Complex multiply ---
    let (params, params_bytes) = scratch.params(&[n])?;
    ctx.dispatch(
        &mut encoder,
        Operation::FFT_COMPLEX_MUL,
        suffix,
        [
            bind(&product, complex_bytes),
            bind(&input_fft, complex_bytes),
            bind(&kernel_fft, complex_bytes),
            bind(&params, params_bytes),
        ],
        n_groups,
    )?;

    // --- Pass 6: Complex bit-reverse product -> input_fft (reused) for IFFT ---
    let (params, params_bytes) = scratch.params(&[n])?;
    ctx.dispatch(
        &mut encoder,
        Operation::FFT_COMPLEX_BIT_REVERSE,
        suffix,
        [
            bind(&input_fft, complex_bytes),
            bind(&product, complex_bytes),
            bind(&params, params_bytes),
        ],
        n_groups,
    )?;

    // --- Pass 7: IFFT butterfly on input_fft (log2(N) stages) ---
    for stage in 0..log2n {
        let is_last = u32::from(stage == log2n - 1);
        let (params, params_bytes) = scratch.params(&[n, stage, is_last])?;
        ctx.dispatch(
            &mut encoder,
            Operation::IFFT_BUTTERFLY,
            suffix,
            [bind(&input_fft, complex_bytes), bind(&params, params_bytes)],
            half_n_groups,
        )?;
    }

    // --- Pass 8: Extract real ---
    let (params, params_bytes) = scratch.params(&[output_len as u32])?;
    ctx.dispatch(
        &mut encoder,
        Operation::FFT_EXTRACT_REAL,
        suffix,
        [
            bind(output.storage(), output.byte_len()),
            bind(&input_fft, complex_bytes),
            bind(&params, params_bytes),
        ],
        dispatch_1d(output_len as u32),
    )?;

    ctx.queue().submit(Some(encoder.finish()));

    Ok(output)
}
