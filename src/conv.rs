//! 1D convolutions: a direct (spatial) one and an FFT-based one, as sequences of built-in
//! kernels recorded into the context's batch.

use std::panic::Location;

use crate::array::{Array, Failure, Storage};
use crate::builtin::{conv_tile, Builtin, CONV_MAX_STRIDE};
use crate::context::{Context, RawArg, Scratch};
use crate::error::LumaError;
use crate::fft;
use crate::ops::{fold_workgroups, launch_1d};
use crate::shape::Shape;

type Outcome<T> = std::result::Result<T, Failure>;

impl Array<f32> {
    /// 1D convolution (cross-correlation) with the given kernel, stride and zero-padding.
    ///
    /// Output length = floor((input_len - kernel_len + 2 * padding) / stride) + 1
    ///
    /// Like operators, failures (kernel larger than the padded input, zero stride, ...) give a
    /// poisoned array.
    #[track_caller]
    pub fn conv1d(&self, kernel: &Array<f32>, stride: usize, padding: usize) -> Array<f32> {
        let location = Location::caller();
        let result = run_conv1d(self, kernel, stride, padding);
        Array::resolve(self.context(), self.shape(), result, "conv1d", location)
    }

    /// FFT-based 1D convolution. Returns the full linear convolution of
    /// length `input_len + kernel_len - 1`. No stride or padding — those
    /// are spatial-domain concepts; use `conv1d()` for that.
    ///
    /// This is O(n log n) and faster than spatial `conv1d` for large kernels.
    #[track_caller]
    pub fn conv1d_fft(&self, kernel: &Array<f32>) -> Array<f32> {
        let location = Location::caller();
        let result = run_conv1d_fft(self, kernel);
        Array::resolve(self.context(), self.shape(), result, "conv1d_fft", location)
    }
}

fn check_operands(input: &Array<f32>, kernel: &Array<f32>) -> Outcome<()> {
    input.storage()?;
    kernel.storage()?;
    if !input.context().same_as(kernel.context()) {
        return Err(LumaError::ContextMismatch.into());
    }
    if kernel.shape().dims()[1..] != [1, 1, 1] {
        return Err(LumaError::InvalidShape("Kernel must be 1D".into()).into());
    }
    Ok(())
}

fn to_u32(value: usize, what: &str) -> Result<u32, LumaError> {
    u32::try_from(value)
        .map_err(|_| LumaError::InvalidShape(format!("{what} {value} exceeds u32::MAX")))
}

fn run_conv1d(
    input: &Array<f32>,
    kernel: &Array<f32>,
    stride: usize,
    padding: usize,
) -> Outcome<Array<f32>> {
    check_operands(input, kernel)?;
    let ctx = input.context();

    let input_len = input.len();
    let kernel_len = kernel.len();
    if stride == 0 {
        return Err(LumaError::InvalidShape("Stride must be non-zero".into()).into());
    }
    let padded_len = padding
        .checked_mul(2)
        .and_then(|doubled| doubled.checked_add(input_len))
        .ok_or_else(|| LumaError::InvalidShape("Padding is too large".into()))?;
    if padded_len < kernel_len {
        return Err(LumaError::InvalidShape("Kernel is larger than padded input".into()).into());
    }
    let output_len = (padded_len - kernel_len) / stride + 1;

    let output = Array::<f32>::alloc(ctx, Shape::from(output_len))?;
    let input_storage = input.storage()?;
    let output_storage = output.storage()?;
    if input_storage.segments() == 1 && output_storage.segments() == 1 {
        // Buffers can be larger than their arrays, so the kernel gets every length explicitly.
        let params = [
            to_u32(kernel_len, "Kernel length")?,
            to_u32(stride, "Stride")?,
            to_u32(padding, "Padding")?,
            to_u32(input_len, "Input length")?,
            to_u32(output_len, "Output length")?,
        ];
        let (out_pin, out_bytes) = output.ready()?;
        let (in_pin, in_bytes) = input.ready()?;
        let (k_pin, k_bytes) = kernel.ready()?;
        launch_conv(ctx, params, (out_pin.buffer(), out_bytes), (in_pin.buffer(), in_bytes), (k_pin.buffer(), k_bytes))?;
        return Ok(output);
    }

    let (k_pin, k_bytes) = kernel.ready()?;
    let kernel_buf = (k_pin.buffer(), k_bytes);
    let c = input_storage.segment_len();
    if kernel_len > c / 2 {
        return Err(LumaError::Chunked("conv1d with a kernel longer than half a buffer").into());
    }
    let (n, k, s, pad) = (to_i64(input_len)?, to_i64(kernel_len)?, to_i64(stride)?, to_i64(padding)?);
    let per_piece = c.min((c - kernel_len) / stride + 1);
    for a in (0..output_len).step_by(per_piece) {
        let b = output_len.min(a + per_piece);
        let (ai, bi) = (to_i64(a)?, to_i64(b)?);
        let w0 = (ai * s - pad).max(0);
        let w1 = n.min((bi - 1) * s + k - pad);
        let pieces = b - a;
        let mut scratch = Scratch::new(ctx);
        let out = scratch.take(pieces as u64 * 4)?;
        if w1 <= w0 {
            // The piece lies entirely in the padding.
            ctx.clear(&out, pieces as u64 * 4)?;
        } else {
            let window = usize::try_from(w1 - w0)
                .map_err(|_| LumaError::InvalidShape("Window length out of range".into()))?;
            let start = usize::try_from(w0)
                .map_err(|_| LumaError::InvalidShape("Window start out of range".into()))?;
            let win = scratch.take(window as u64 * 4)?;
            gather(ctx, input_storage, start, window, &win)?;
            let left = usize::try_from(pad + w0 - ai * s)
                .map_err(|_| LumaError::InvalidShape("Piece padding out of range".into()))?;
            let params = [
                to_u32(kernel_len, "Kernel length")?,
                to_u32(stride, "Stride")?,
                to_u32(left, "Padding")?,
                to_u32(window, "Input length")?,
                to_u32(pieces, "Output length")?,
            ];
            launch_conv(ctx, params, (&out, pieces as u64 * 4), (&win, window as u64 * 4), kernel_buf)?;
        }
        scatter(ctx, &out, 0, output_storage, a, pieces)?;
    }
    Ok(output)
}

/// Converts a length to `i64` for the signed window arithmetic of the piecewise convolution.
fn to_i64(value: usize) -> Result<i64, LumaError> {
    i64::try_from(value).map_err(|_| LumaError::InvalidShape(format!("Length {value} is out of range")))
}

/// Records the direct convolution `params = [kernel, stride, padding, input, output]` lengths on
/// whole buffers; shared by the single-buffer path and the pieces of the chunked path.
fn launch_conv(
    ctx: &Context,
    params: [u32; 5],
    out: (&wgpu::Buffer, u64),
    input: (&wgpu::Buffer, u64),
    kernel: (&wgpu::Buffer, u64),
) -> Result<(), LumaError> {
    let args = [
        RawArg::write(out.0, out.1),
        RawArg::read(input.0, input.1),
        RawArg::read(kernel.0, kernel.1),
        RawArg::Uniform(bytemuck::bytes_of(&params)),
    ];
    if params[1] <= CONV_MAX_STRIDE {
        let (wg, per_thread) = (ctx.tuning().conv_workgroup, ctx.tuning().conv_per_thread);
        let core = ctx.builtin(Builtin::Conv1dTiled(params[1], per_thread), "f32")?;
        let groups = fold_workgroups(ctx, params[4].div_ceil(conv_tile(params[1], per_thread, wg)))?;
        ctx.launch(&core, groups, [params[4], 1, 1], &args)?;
    } else {
        let core = ctx.builtin(Builtin::Conv1d, "f32")?;
        launch_1d(ctx, &core, params[4], &args)?;
    }
    Ok(())
}

/// Records a copy of elements `start..start + len` of `storage` into the start of `dst`, walking
/// the segments that range overlaps. Needed because a chunked array has no single buffer to read.
fn gather(ctx: &Context, storage: &Storage, start: usize, len: usize, dst: &wgpu::Buffer) -> Result<(), LumaError> {
    let end = start + len;
    for j in segment_span(storage, start, len) {
        let range = storage.range(j);
        let (lo, hi) = (start.max(range.start), end.min(range.end));
        let pin = storage.pin(j)?;
        ctx.copy(
            pin.buffer(),
            (lo - range.start) as u64 * 4,
            dst,
            (lo - start) as u64 * 4,
            (hi - lo) as u64 * 4,
        )?;
    }
    Ok(())
}

/// Records a copy of `len` elements of `src`, from element `src_start`, into elements
/// `start..` of `storage`, walking the segments that range overlaps. The inverse of [`gather`].
fn scatter(
    ctx: &Context,
    src: &wgpu::Buffer,
    src_start: usize,
    storage: &Storage,
    start: usize,
    len: usize,
) -> Result<(), LumaError> {
    let end = start + len;
    for j in segment_span(storage, start, len) {
        let range = storage.range(j);
        let (lo, hi) = (start.max(range.start), end.min(range.end));
        let pin = storage.pin(j)?;
        ctx.copy(
            src,
            (src_start + lo - start) as u64 * 4,
            pin.buffer(),
            (lo - range.start) as u64 * 4,
            (hi - lo) as u64 * 4,
        )?;
    }
    Ok(())
}

/// The segment indices overlapping elements `start..start + len` (none when `len` is 0).
fn segment_span(storage: &Storage, start: usize, len: usize) -> std::ops::Range<usize> {
    if len == 0 {
        return 0..0;
    }
    let c = storage.segment_len();
    start / c..(start + len - 1) / c + 1
}

fn run_conv1d_fft(input: &Array<f32>, kernel: &Array<f32>) -> Outcome<Array<f32>> {
    check_operands(input, kernel)?;
    let ctx = input.context();

    let input_len = to_u32(input.len(), "Input length")?;
    let kernel_len = to_u32(kernel.len(), "Kernel length")?;
    let output_len = input.len() + kernel.len() - 1;

    let output = Array::<f32>::alloc(ctx, Shape::from(output_len))?;
    let (k_pin, k_bytes) = kernel.ready()?;
    let input_storage = input.storage()?;
    let output_storage = output.storage()?;
    if input_storage.segments() == 1 && output_storage.segments() == 1 {
        let (out_pin, out_bytes) = output.ready()?;
        let (in_pin, in_bytes) = input.ready()?;
        let operands = fft::Operands {
            input: (in_pin.buffer(), in_bytes, input_len),
            kernel: (k_pin.buffer(), k_bytes, kernel_len),
            output: (out_pin.buffer(), out_bytes, to_u32(output_len, "Output length")?),
        };
        fft::convolve(ctx, &operands, &mut Scratch::new(ctx))?;
        return Ok(output);
    }

    let c = input_storage.segment_len();
    let k = kernel.len();
    if k > c / 16 {
        return Err(LumaError::Chunked("conv1d_fft with a kernel longer than a sixteenth of a buffer").into());
    }
    let n = input.len();
    let per_piece = c / 8;
    for a in (0..output_len).step_by(per_piece) {
        let b = output_len.min(a + per_piece);
        let w0 = a.saturating_sub(k - 1);
        let w1 = n.min(b);
        let window = w1 - w0;
        let tmp_len = window + k - 1;
        let mut scratch = Scratch::new(ctx);
        let win = scratch.take(window as u64 * 4)?;
        let tmp = scratch.take(tmp_len as u64 * 4)?;
        gather(ctx, input_storage, w0, window, &win)?;
        let operands = fft::Operands {
            input: (&win, window as u64 * 4, to_u32(window, "Window length")?),
            kernel: (k_pin.buffer(), k_bytes, kernel_len),
            output: (&tmp, tmp_len as u64 * 4, to_u32(tmp_len, "Window length")?),
        };
        fft::convolve(ctx, &operands, &mut scratch)?;
        scatter(ctx, &tmp, a - w0, output_storage, a, b - a)?;
    }
    Ok(output)
}
