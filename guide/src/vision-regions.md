# Connected components, confidence connected components, deconvolution

`Array<bool>::regions` labels the connected components of a binary image and
`Array<f32>::confidence_cc` grows a region from seed pixels by the statistics of the pixels
around them: ArrayFire's `regions` and `confidenceCC`. Both take an `[H, W]` image (axis 0, the
height, varies fastest), are built-in kernels that run on every backend, need an array that fits
one buffer, and poison their result on an error. Neither synchronizes.

## `regions`

```rust
use luma::{Array, Connectivity};

// The 3 x 3 image, columns [1, 0, 1], [0, 0, 0], [0, 1, 1] (axis 0 fastest).
let img = Array::from_slice(&[true, false, true, false, false, false, false, true, true], [3, 3])?;
assert_eq!(img.regions(Connectivity::Four).to_vec()?, [1, 0, 2, 0, 0, 0, 0, 3, 3]);
// With 8-connectivity the pixels touching at a corner are one component.
let img = Array::from_slice(&[true, false, false, false, true, false, false, false, true], [3, 3])?;
assert_eq!(img.regions(Connectivity::Four).to_vec()?, [1, 0, 0, 0, 2, 0, 0, 0, 3]);
assert_eq!(img.regions(Connectivity::Eight).to_vec()?, [1, 0, 0, 0, 1, 0, 0, 0, 1]);
# Ok::<(), luma::LumaError>(())
```

The result is an `Array<u32>`: 0 for the background, `1..=N` for the components, **numbered in
the order of their first pixel in memory order** (axis 0 fastest). That is the numbering of
both ArrayFire backends (probed on the CPU and CUDA ones), so the labels equal ArrayFire's,
not only the partition. Differences from ArrayFire: the labels are `u32` (convert with
`to_f32()` and the like for ArrayFire's `f32`, `i32`, `u16`, `f64` outputs) and the image must
have fewer than 2^32 - 1 pixels. Like ArrayFire's, the input has to be two-dimensional.

### How it works

A lock-free union-find over the pixels: a set's root is always its smallest pixel (flat index), so
the roots in ascending order are the components in ascending order of their first pixel.

1. A workgroup labels a tile of 32 x 8 pixels in workgroup memory (every foreground pixel is
   unioned with its backward neighbours in the tile) and stores, for every pixel, the flat index of
   the root of its piece of the component.
2. One thread per pixel along a tile boundary unions the roots on both sides (`atomicMin` hooks the
   larger root under the smaller one).
3. Every pixel's entry becomes the final root; a workgroup takes 1024 consecutive pixels, counts
   their roots and ranks them.
4. A tiny scan turns the counts of the chunks into offsets.
5. The entry of a pixel becomes `offset + rank + 1` of its root.

There is no iteration over the image, so the time does not depend on the shape of the components
(a one pixel wide spiral costs the same as noise).

## `confidence_cc`

```rust
use luma::Array;

// A bright 2 x 2 block (10) on a dark background (1); a seed inside it.
let mut data = vec![1.0f32; 16];
for (i, j) in [(1, 1), (2, 1), (1, 2), (2, 2)] {
    data[i + 4 * j] = 10.0;
}
let img = Array::from_slice(&data, [4, 4])?;
let seg = img.confidence_cc_at(&[1], &[1], 0, 1, 1, 255.0)?;
assert_eq!(seg.to_vec()?.iter().filter(|&&v| v == 255.0).count(), 4);
# Ok::<(), luma::LumaError>(())
```

`confidence_cc(&seeds_x, &seeds_y, radius, multiplier, iterations, segmented_value)` (also
`confidence_cc_seeds` with an `[N, 2]` array and `confidence_cc_at` with host slices) returns an
`f32` image holding `segmented_value` in the segmentation and 0 elsewhere. **`seeds_x` indexes
axis 0 and `seeds_y` axis 1**, as in ArrayFire's `confidenceCC` (the `x` of keypoints is the column:
pass `keypoints.y` as `seeds_x`).

The algorithm is ArrayFire's: the image is scaled linearly to `[1, 2]`; the mean and standard
deviation of the pixels in the `(2 radius + 1)^2` windows around the seeds give the interval
`[mean - multiplier * sd, mean + multiplier * sd]` (widened to contain the seeds' own values); the
first segmentation is the set of pixels in the interval that are 8-connected to a seed through
such pixels. Each of the `iterations` further rounds takes mean and standard deviation of the
pixels of the *current segmentation*, widens the interval likewise and segments again from the
seeds. The rounds stop early when the segmentation is empty or a variance is below 1e-6.

The whole loop runs on the device without a read-back: the control words (interval, "done" flag)
live in a small buffer, a segmentation is the union-find labelling of [`regions`](#regions) over
the pixels in the interval followed by marking the roots the seeds fall into, and the launches of
rounds after the last one are no-ops that exit at once.

### Differences from ArrayFire

ArrayFire's `confidenceCC` is only well defined for a **single seed whose window is inside the
image**; probing it (CPU backend; the shared code behind it is the same on CUDA) showed:

* A window reaching over the border reads outside ArrayFire's summed-area table: the statistics
  are garbage, usually a segmentation of the seed alone or of the whole image. luma clips the
  window to the image.
* With several seeds ArrayFire sums the windows over all *pairs* of seed rows and columns (it
  indexes its summed-area table with two index arrays) but divides by the number of seeds times the
  window size: the variance comes out negative, the interval is NaN and the first segmentation is
  just the seeds. luma uses the union of the windows of the seeds. The widening to the seeds'
  values uses the values at all row/column pairs in ArrayFire, the seeds' own in luma.
* Seeds outside the image are ignored.
* ArrayFire's CPU backend adds stray components not connected to the seed to the segmentation of
  the later iterations (seen with identical intervals); luma's segmentation is always the
  component of the seeds.
* ArrayFire takes the window sums from an `f32` summed-area table of the whole image, which has no
  precision left on a big image (on 4K its segmentation is the seed alone). luma adds up the
  pixels of the windows from values shifted by their mean, so its intervals are exact; a pixel
  exactly on the border of an interval may go either way.

For a single seed with its window inside the image the segmentation is the one of ArrayFire's CUDA
backend (identical in 31 of 31 random probes; the CPU backend adds stray components) up to the
pixels where ArrayFire's imprecise interval is wrong on a big image (the `afcompare` rows count
them).

## Deconvolution

`iterative_deconv(&psf, iterations, relaxation, algo)` and `inverse_deconv(&psf, gamma, algo)`
(ArrayFire's `iterativeDeconv` and `inverseDeconv`) restore an `[H, W]` image blurred by a known
point spread function `[PH, PW]` whose centre is `(PH / 2, PW / 2)`.

```rust
use luma::{Array, InverseDeconvAlgo, IterativeDeconvAlgo};

let img = Array::from_slice(&(0..20).map(|i| 1.0 + i as f32).collect::<Vec<_>>(), [4, 5])?;
// A psf that is a single 1 at its centre does not blur.
let psf = Array::from_slice(&[0.0f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], [3, 3])?;
let same = img.iterative_deconv(&psf, 3, 0.5, IterativeDeconvAlgo::RichardsonLucy).to_vec()?;
assert!(same.iter().zip(img.to_vec()?).all(|(a, b)| (a - b).abs() < 1e-4));
// Tikhonov divides the spectrum by |P|^2 + gamma = 1 + gamma.
let tik = img.inverse_deconv(&psf, 0.5, InverseDeconvAlgo::Tikhonov).to_vec()?;
assert!((tik[0] - 1.0 / 1.5).abs() < 1e-4);
# Ok::<(), luma::LumaError>(())
```

* **Landweber** (`IterativeDeconvAlgo::Landweber`, ArrayFire's default): `x += a P^T (y - P x)`
  per iteration, starting from the blurred image `y`.
* **Richardson-Lucy**: `x *= P^T (y / (P x))`; `relaxation` is unused (but must be positive, as in
  ArrayFire). Needs positive data.
* **Tikhonov** (`InverseDeconvAlgo::Tikhonov`): `X = Y conj(P) / (|P|^2 + gamma)` per frequency.

The problem is solved on the image padded to `(H + PH) x (W + PW)`, each size rounded up to one whose
prime factors are all at most 7 (the rule of ArrayFire's CUDA backend, probed; its CPU backend
rounds to factors of at most 13 and so pads differently, which changes the border of the result), with the nearest
pixel; the psf is zero padded to the same size and rolled so that its centre is the origin. The
convolutions are circular on the padded image, so the border sees a replicated, not a periodic,
image. The result is the central `[H, W]` part.

Landweber's step is diagonal in the frequency domain (`X += a conj(P) (Y - P X)`), so any number of
iterations costs one forward transform, one inverse transform and one elementwise pass per
iteration; Richardson-Lucy needs two forward and two inverse real transforms per iteration.

### Differences from ArrayFire

* ArrayFire's Landweber and Tikhonov results are `N0 * N1` times larger than the estimate (the
  padded pixel count: its last inverse transform is not scaled; its Richardson-Lucy result is not
  affected). luma returns the estimate; multiply by `N0 * N1` for ArrayFire's numbers.
  Everything else (padding, centring, update) is ArrayFire's, to float rounding: the `afcompare`
  rows and a test compare the results with ArrayFire's.
* Where the blurred estimate of Richardson-Lucy is not positive the ratio is taken as 0.
  ArrayFire's division gives inf or NaN there, which spreads over the whole image through the next
  transform (any zero pixel of the input spoils its result).
* The input is `f32` (ArrayFire also takes `u8`, `u16`, `i16`; convert with `to_f32`).
* Like ArrayFire, `iterations`, `relaxation` and `gamma` must be positive and the arrays two-dimensional;
  the point spread function may be larger than the image.
