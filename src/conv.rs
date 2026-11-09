//! 1D convolutions: a direct (spatial) one and an FFT-based one, as sequences of built-in
//! kernels recorded into the context's batch.

use std::panic::Location;

use crate::array::{Array, Failure};
use crate::builtin::{Builtin, CONV_TILE};
use crate::context::{RawArg, Scratch};
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
    input.ready()?;
    kernel.ready()?;
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

    // Buffers can be larger than their arrays, so the kernel gets every length explicitly.
    let params = [
        to_u32(kernel_len, "Kernel length")?,
        to_u32(stride, "Stride")?,
        to_u32(padding, "Padding")?,
        to_u32(input_len, "Input length")?,
        to_u32(output_len, "Output length")?,
    ];
    let (out_buf, out_bytes) = output.ready()?;
    let (in_buf, in_bytes) = input.ready()?;
    let (k_buf, k_bytes) = kernel.ready()?;
    let args = [
        RawArg::write(out_buf, out_bytes),
        RawArg::read(in_buf, in_bytes),
        RawArg::read(k_buf, k_bytes),
        RawArg::Uniform(bytemuck::bytes_of(&params)),
    ];
    if stride == 1 {
        let core = ctx.builtin(Builtin::Conv1dTiled, "f32")?;
        let groups = fold_workgroups(ctx, params[4].div_ceil(CONV_TILE))?;
        ctx.launch(&core, groups, [params[4], 1, 1], &args)?;
    } else {
        let core = ctx.builtin(Builtin::Conv1d, "f32")?;
        launch_1d(ctx, &core, params[4], &args)?;
    }
    Ok(output)
}

fn run_conv1d_fft(input: &Array<f32>, kernel: &Array<f32>) -> Outcome<Array<f32>> {
    check_operands(input, kernel)?;
    let ctx = input.context();

    let input_len = to_u32(input.len(), "Input length")?;
    let kernel_len = to_u32(kernel.len(), "Kernel length")?;
    let output_len = input.len() + kernel.len() - 1;

    let output = Array::<f32>::alloc(ctx, Shape::from(output_len))?;
    let (out_buf, out_bytes) = output.ready()?;
    let (in_buf, in_bytes) = input.ready()?;
    let (k_buf, k_bytes) = kernel.ready()?;
    let operands = fft::Operands {
        input: (in_buf, in_bytes, input_len),
        kernel: (k_buf, k_bytes, kernel_len),
        output: (out_buf, out_bytes, to_u32(output_len, "Output length")?),
    };
    fft::convolve(ctx, &operands, &mut Scratch::new(ctx))?;
    Ok(output)
}
