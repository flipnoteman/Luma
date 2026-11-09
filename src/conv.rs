//! 1D convolutions: a direct (spatial) one and an FFT-based one, as sequences of built-in
//! kernels recorded into the context's batch.

use std::panic::Location;

use crate::array::{Array, Failure};
use crate::builtin::Builtin;
use crate::context::{RawArg, Scratch};
use crate::error::LumaError;
use crate::ops::launch_1d;
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
    let core = ctx.builtin(Builtin::Conv1d, "f32")?;
    launch_1d(
        ctx,
        &core,
        params[4],
        &[
            RawArg::write(out_buf, out_bytes),
            RawArg::read(in_buf, in_bytes),
            RawArg::read(k_buf, k_bytes),
            RawArg::Uniform(bytemuck::bytes_of(&params)),
        ],
    )?;
    Ok(output)
}

fn run_conv1d_fft(input: &Array<f32>, kernel: &Array<f32>) -> Outcome<Array<f32>> {
    check_operands(input, kernel)?;
    let ctx = input.context();

    let input_len = to_u32(input.len(), "Input length")?;
    let kernel_len = to_u32(kernel.len(), "Kernel length")?;
    let output_len = input.len() + kernel.len() - 1;
    let n = u32::try_from(output_len)
        .ok()
        .and_then(u32::checked_next_power_of_two)
        .ok_or(LumaError::DimensionTooLarge { shape: [output_len, 1, 1, 1] })?;
    let log2n = n.trailing_zeros();

    let output = Array::<f32>::alloc(ctx, Shape::from(output_len))?;
    let (out_buf, out_bytes) = output.ready()?;
    let (in_buf, in_bytes) = input.ready()?;
    let (k_buf, k_bytes) = kernel.ready()?;

    let mut scratch = Scratch::new(ctx);

    // Complex buffers (interleaved re/im, so 2*N f32s each). Every pass writes all N entries of
    // the buffer it produces, so stale pool contents never leak into the result.
    let complex_bytes = u64::from(n) * 2 * 4;
    let input_fft = scratch.take(complex_bytes)?;
    let kernel_fft = scratch.take(complex_bytes)?;
    let product = scratch.take(complex_bytes)?;

    let bit_reverse = ctx.builtin(Builtin::FftBitReverse, "f32")?;
    let butterfly = ctx.builtin(Builtin::FftButterfly, "f32")?;
    let complex_mul = ctx.builtin(Builtin::FftComplexMul, "f32")?;
    let complex_bit_reverse = ctx.builtin(Builtin::FftComplexBitReverse, "f32")?;
    let ibutterfly = ctx.builtin(Builtin::IfftButterfly, "f32")?;
    let extract_real = ctx.builtin(Builtin::FftExtractReal, "f32")?;

    // --- Pass 1: Bit-reverse input ---
    launch_1d(
        ctx,
        &bit_reverse,
        n,
        &[
            RawArg::write(&input_fft, complex_bytes),
            RawArg::read(in_buf, in_bytes),
            RawArg::Uniform(bytemuck::bytes_of(&[n, input_len])),
        ],
    )?;

    // --- Pass 2: Bit-reverse kernel ---
    launch_1d(
        ctx,
        &bit_reverse,
        n,
        &[
            RawArg::write(&kernel_fft, complex_bytes),
            RawArg::read(k_buf, k_bytes),
            RawArg::Uniform(bytemuck::bytes_of(&[n, kernel_len])),
        ],
    )?;

    // --- Pass 3 + 4: FFT butterfly on input, then kernel (log2(N) stages each) ---
    for fft_buffer in [&input_fft, &kernel_fft] {
        for stage in 0..log2n {
            launch_1d(
                ctx,
                &butterfly,
                n / 2,
                &[
                    RawArg::write(fft_buffer, complex_bytes),
                    RawArg::Uniform(bytemuck::bytes_of(&[n, stage])),
                ],
            )?;
        }
    }

    // --- Pass 5: Complex multiply ---
    launch_1d(
        ctx,
        &complex_mul,
        n,
        &[
            RawArg::write(&product, complex_bytes),
            RawArg::read(&input_fft, complex_bytes),
            RawArg::read(&kernel_fft, complex_bytes),
            RawArg::Uniform(bytemuck::bytes_of(&[n])),
        ],
    )?;

    // --- Pass 6: Complex bit-reverse product -> input_fft (reused) for IFFT ---
    launch_1d(
        ctx,
        &complex_bit_reverse,
        n,
        &[
            RawArg::write(&input_fft, complex_bytes),
            RawArg::read(&product, complex_bytes),
            RawArg::Uniform(bytemuck::bytes_of(&[n])),
        ],
    )?;

    // --- Pass 7: IFFT butterfly on input_fft (log2(N) stages) ---
    for stage in 0..log2n {
        let is_last = u32::from(stage + 1 == log2n);
        launch_1d(
            ctx,
            &ibutterfly,
            n / 2,
            &[
                RawArg::write(&input_fft, complex_bytes),
                RawArg::Uniform(bytemuck::bytes_of(&[n, stage, is_last])),
            ],
        )?;
    }

    // --- Pass 8: Extract real ---
    launch_1d(
        ctx,
        &extract_real,
        to_u32(output_len, "Output length")?,
        &[
            RawArg::write(out_buf, out_bytes),
            RawArg::read(&input_fft, complex_bytes),
            RawArg::Uniform(bytemuck::bytes_of(&[output_len as u32])),
        ],
    )?;

    Ok(output)
}
