# Images and filters

luma has the usual 2-D and 3-D image operations as built-in kernels that run on every backend:
convolutions, rank and morphological filters, derivatives, the bilateral filter, summed-area
tables, colour conversions and `unwrap` / `wrap` (im2col / col2im).

An image is an `[H, W]` array (**axis 0, the height, varies fastest**, as in ArrayFire) or an
`[H, W, C]` array with the channels in planes. The axes after the ones an operation works on
are a batch: `conv2` filters every `[H, W]` plane of an `[H, W, N, M]` array, `conv3` every
`[D0, D1, D2]` volume of an `[D0, D1, D2, N]` array. Colour conversions take `[H, W, 3]`
(and `[H, W, 3, N]`).

Like every operation they record work and poison the result on an error (a zero stride, a window
larger than the padded image, a kernel of the wrong shape). They need arrays that fit one
buffer.

## Convolution

`conv2(&kernel, spec)` and `conv3` are *cross-correlations* (like `conv1d`): the kernel is not
flipped; flip it for a true convolution, which is what ArrayFire's `convolve2` computes.
`ConvSpec` gives the padding (`Same`, the default: the output has the shape of the input;
`Valid`; `Full`; or explicit cells before and after every axis), the stride and the border
([`Border`]: `Zero`, `Edge`, `Mirror`, `Wrap`).

```rust
use luma::*;

let img = Array::from_slice(&(0..12).map(|i| i as f32).collect::<Vec<_>>(), [4, 3])?;
let k = gaussian_kernel(3, 3, 1.0, 1.0)?;
let blurred = img.conv2(&k, ConvSpec::new().border(Border::Edge));
let small = img.conv2(&k, ConvSpec::new().padding(Padding::Valid).stride([2, 1]));
assert_eq!(blurred.shape(), img.shape());
assert_eq!(small.shape(), Shape::from([1, 1]));
# Ok::<(), luma::LumaError>(())
```

For a separable kernel `conv2_sep(&along0, &along1, spec)` does `k0 + k1` multiply-adds per
output instead of `k0 * k1`; `gaussian_kernel_1d(len, sigma)` gives the taps of a Gaussian.

A workgroup stages a tile of the image together with the halo its windows need in workgroup
memory (the border is applied while staging) and every invocation computes several outputs
from it, so the image is read from memory about once whatever the kernel size.

## Rank and morphological filters

`medfilt1(len, border)`, `medfilt2([k0, k1], border)`, `minfilt` / `maxfilt([k0, k1], border)`
(`minfilt3` / `maxfilt3` for volumes) work on `f32`, `i32` and `u32`. `dilate(&mask)` and
`erode(&mask)` (`dilate3`, `erode3`) take any structuring element: the window cells where the
mask is not zero. With `Border::Ignore` (the default of the morphological filters) cells outside
the image do not take part.

```rust
use luma::*;

let img = Array::from_slice(&[1u32, 5, 2, 9, 3, 7, 4, 8, 6], [3, 3])?;
let m = img.medfilt2([3, 3], Border::Edge).to_vec()?;
assert_eq!(m[4], 5); // the median of the whole image
let cross = Array::from_slice(&[0u32, 1, 0, 1, 1, 1, 0, 1, 0], [3, 3])?;
assert_eq!(img.dilate(&cross).to_vec()?[4], 9);
# Ok::<(), luma::LumaError>(())
```

## Derivatives, bilateral filter

`sobel()` returns the Sobel derivatives `(d0, d1)` along axis 0 and axis 1 from one pass over the
image, `sobel_magnitude()` their magnitude, `grad()` the central-difference gradient (one-sided
at the borders). `bilateral(spatial_sigma, range_sigma)` and `bilateral_color(..)` (for
`[H, W, 3]`) are the edge-preserving smoothing filter.

## Summed-area tables, moments, histograms

`sat()` is the double prefix sum (four lookups give the sum of any rectangle), `moments()` the raw
moments `M00 M01 M10 M11` of every image, `hist_equal(&hist)` equalises an image with the
histogram `hist` of its (integer valued) pixels.

## Colour

`rgb2gray`, `gray2rgb`, `rgb2hsv`, `hsv2rgb`, `rgb2ycbcr(YccStd)` and `ycbcr2rgb(YccStd)` convert
`[H, W, 3]` images (RGB in `0 ..= 1`, hue in `0 .. 1`, Y'CbCr in the 8-bit studio range);
`color_space(from, to)` chains them.

## im2col and col2im

`unwrap(window, stride, pad, column)` lays the windows of an image out as the columns (or rows)
of a matrix, so that a convolution is a `matmul`; `wrap(..)` sums them back (every output cell
gathers its windows, so the sum is deterministic).
