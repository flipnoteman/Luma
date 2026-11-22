# Fourier transforms

`Array<C32>` (complex `f32`, see [`C32`](https://docs.rs/luma)) has FFTs of any length along any
axis, in one to three dimensions, batched over the remaining axes. They run on every backend as a
few passes over the data, each transforming tiles in workgroup memory, and write the result in
natural order.

```rust
use luma::*;

let x = Array::from_slice(&[C32::new(1.0, 0.0), C32::new(2.0, 0.0), C32::new(3.0, 0.0), C32::new(4.0, 0.0)], 4)?;
let spectrum = x.fft();                      // X[k] = sum_n x[n] e^(-2 pi i n k / N)
assert_eq!(spectrum.to_vec()?[0], C32::new(10.0, 0.0));
let back = spectrum.ifft();                  // divided by N: back == x
# Ok::<(), luma::LumaError>(())
```

| Method | Transform |
| --- | --- |
| `fft()`, `ifft()` | axis 0, batched over the others |
| `fft_axis(axis, norm)`, `ifft_axis(axis, norm)` | one axis (0 to 3) |
| `fft2()`, `ifft2()`, `fft3()`, `ifft3()` | axes 0 and 1, or 0, 1 and 2 |
| `fft_axes(&[..], norm)`, `ifft_axes(&[..], norm)` | any axes, in the order given |
| `Array<f32>::fft()`, `fft_axes(..)` | the complex spectrum of real input |
| `Array<f32>::rfft()`, `rfft_axis`, `rfft2()`, `rfft3()`, `rfft_axes` | real input: only the `N / 2 + 1` non-redundant frequencies along the first axis (ArrayFire's `fftR2C`) |
| `Array<C32>::irfft(n)`, `irfft_axis`, `irfft2(n0)`, `irfft3(n0)`, `irfft_axes` | the inverse of `rfft`: `n` real points from the half spectrum |
| `Array<f32>::fft_convolve1/2/3(&kernel, mode)` | convolution by FFTs (`ConvMode::Same` or `Full`) |

## Normalisation

`FftNorm` says how a transform is scaled. The default of `fft`/`ifft`/... (without a `norm`
argument) is `Backward`, the convention of NumPy and ArrayFire's `ifft`:

| `FftNorm` | forward | inverse |
| --- | --- | --- |
| `Backward` | 1 | 1 / N |
| `Forward` | 1 / N | 1 |
| `Ortho` | 1 / sqrt(N) | 1 / sqrt(N) |
| `None` | 1 | 1 |
| `Scale(s)` | `s` | `s` (once, however many axes) |

`N` is the number of points of the transformed axis; each axis of a multi-axis transform is
scaled by its own length.

## Lengths

Any length works. Lengths whose prime factors are at most 31 (powers of two, 3, 5, 7, ...) are
transformed directly in as many passes as needed (one pass up to 4096 points per line, two up to
about 16 million); other lengths use Bluestein's algorithm, a circular convolution of the next
power of two at least `2N - 1`, which costs several times as much. Even-length real transforms are
half-size complex transforms plus one pass that packs or unpacks the spectrum.

## Convolution

`fft_convolve2(&kernel, ConvMode::Same)` zero-pads signal and kernel to a fast length, multiplies
their spectra and transforms back. `Same` returns the signal's shape with the kernel centred as
ArrayFire does (output `i` is element `i + k / 2` of the full convolution); `Full` returns
`signal + kernel - 1` along each convolved axis. The kernel must have extent 1 along the batch
axes (2 and 3 for `fft_convolve2`). `conv1d_fft` is the one-dimensional full convolution with its
own, convolution-specific passes.

## Performance

Per call on an RTX 3090 (CUDA), against ArrayFire 3.10.1 (cuFFT) in `tools/afcompare` (interleaved,
warm, synchronised; ratio = ArrayFire / luma, above 1 is faster):

| transform | size | luma ms | ArrayFire ms | ratio |
| --- | --- | ---: | ---: | ---: |
| `fft` | 1,000,000 | 0.072 | 0.084 | 1.17 |
| `fft` | 16,000,000 | 1.11 | 1.37 | 1.24 |
| `fft` | 2^20 / 2^24 | 0.066 / 1.00 | 0.079 / 1.49 | 1.20 / 1.50 |
| `fft` (prime, Bluestein) | 1,000,003 | 0.30 | 0.39 | 1.30 |
| `ifft` | 1,000,000 / 16,000,000 | 0.071 / 1.11 | 0.107 / 1.72 | 1.52 / 1.55 |
| `rfft` | 1,000,000 / 16,000,000 | 0.040 / 0.60 | 0.046 / 0.74 | 1.15 / 1.22 |
| `rfft` | 2^20 / 2^24 | 0.036 / 0.55 | 0.036 / 0.71 | 1.00 / 1.29 |
| `fft2` / `ifft2` | 2048 x 2048 | 0.239 / 0.236 | 0.281 / 0.372 | 1.18 / 1.57 |
| `fft3` | 256^3 | 0.985 | 1.38 | 1.40 |
| `fft_convolve2` | 2048^2 with 15^2 | 0.74 | 1.93 | 2.61 |
| `fft_convolve2` | 4096^2 with 31^2 | 4.9 | 9.5 | 1.95 |
| `fft_convolve3` | 256^3 with 9^3 | 3.1 | 16.4 | 5.38 |

A one-pass transform moves the array once in and once out; a transform of up to about 4096 points
per line is one pass, up to about 16 million two or three. Results are accurate to a few `f32`
roundings: against an `f64` reference the rms error is 3e-7 to 8e-7 of the rms of the spectrum (up to
about 2e-6 at the worst point), a little above cuFFT's.

## Notes

* The transforms are computed when called (they are not fused with surrounding lazy operations);
  their inputs are evaluated first.
* A transform needs the whole array in one buffer: an array split over several buffers (see
  [Performance](performance.md)) fails with `LumaError::Chunked`.
* Errors poison the result like any other operation: `fft_axis(4, ..)` gives a poisoned array.
