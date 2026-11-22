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

## Notes

* The transforms are computed when called (they are not fused with surrounding lazy operations);
  their inputs are evaluated first.
* A transform needs the whole array in one buffer: an array split over several buffers (see
  [Performance](performance.md)) fails with `LumaError::Chunked`.
* Errors poison the result like any other operation: `fft_axis(4, ..)` gives a poisoned array.
