# Vision: Harris and SUSAN corners, difference of Gaussians

`harris`, `susan` and `dog` are ArrayFire's functions of the same names (`vision.h`). The
detectors take an `[H, W]` image of `f32` (axis 0, the height, is the fastest axis; a feature's
`x` is the index along axis 1 and `y` the one along axis 0) and return `Option<Keypoints>`: `None`
if there is no corner, since luma has no arrays without elements. The number of corners depends
on the data, so a detector reads it back and **synchronizes** once (like `fast` and `filter`);
the corners themselves stay on the device. `orientation` is 0 and `size` is 1.

ArrayFire's `harris` takes `f32` and `f64`, its `susan` and `dog` every image type; luma takes
`f32` only, like the rest of the image operations. The detectors take one `[H, W]` image and
return an error for a batch (`[H, W, N]`); ArrayFire's `harris` and `susan` reject those as well
(the documentation of `susan` mentions batches, the code does not take them). A detector gives
`None` for an image too small to have a corner. `dog` takes batches like every 2-D filter.

## Harris

```rust
use luma::Array;

// A bright square on a dark image has four corners.
let mut data = vec![0.0f32; 32 * 32];
for j in 12..20 {
    for i in 12..20 {
        data[i + 32 * j] = 100.0;
    }
}
let img = Array::from_slice(&data, [32, 32])?;
let corners = img.harris(10, 0.0, 1.0, 0, 0.04)?.expect("corners");
assert_eq!(corners.count(), 4);
# Ok::<(), luma::LumaError>(())
```

`harris(max_corners, min_response, sigma, block_size, k_thr)`:

* The derivatives are the central differences `(in[i + 1] - in[i - 1]) / 2` along both axes.
  The structure tensor `M` sums their products with a separable window, and the response of a
  pixel is `det(M) - k_thr * trace(M)^2`. (Not normalised in any way: the scale is that of the
  pixel values to the fourth power, the default `min_response` of 1e5 suits pixels of 0 to 255.)
* The window has `floor(6 * sigma)` cells, one less if that is even, with Gaussian weights
  (`block_size == 0`, `sigma` in `[0.5, 5]`: 3 cells for `sigma` 0.5 to 0.8, 5 for 0.9 to 1.1, ...).
  With `block_size > 0` the window has that many cells (1 to 32): Gaussian weights of `sigma`
  if `sigma >= 0.5` and equal weights otherwise. ArrayFire's documentation only mentions the box
  window and `block_size` 3 to 31; its code does what is described here, and so does luma.
* A pixel is a corner if its response is positive and strictly larger than the responses of its
  eight neighbours. The response only exists where the window lies inside the image, so
  corners are at least `window / 2 + 2` cells away from the border.
* With `max_corners > 0` the corners are ranked and the strongest `max_corners` are returned,
  strongest first (`min_response` is ignored). With `max_corners == 0` every corner of response
  at least `min_response` (which must then be positive) is returned. `k_thr` is at least 0.01.

luma computes all of this in one kernel per tile of 32 x 8 pixels (the tile of pixels, the
derivatives, the window passes and the decision in workgroup memory; the response is never
stored), then scans the per-tile counts and gathers the corners: three launches and one read of
the count. The ranking is `topk` of the corners' responses.

Differences from ArrayFire, besides the element types: the corners that are not ranked are
listed tile by tile (ArrayFire's CPU backend by column, its CUDA backend in the order its
blocks finish), the corner set can differ from ArrayFire's by pixels whose response ties with a
neighbour up to float rounding (ArrayFire's own CPU and CUDA backends differ the same way), and
an image too small for any corner is `None`.

## SUSAN

```rust
use luma::Array;

let mut data = vec![0.0f32; 32 * 32];
for j in 10..20 {
    for i in 8..18 {
        data[i + 32 * j] = 200.0;
    }
}
let img = Array::from_slice(&data, [32, 32])?;
let corners = img.susan(3, 32.0, 10.0, 0.5, 3)?.expect("corners");
assert_eq!(corners.count(), 4);
# Ok::<(), luma::LumaError>(())
```

`susan(radius, diff_thr, geom_thr, feature_ratio, edge)`: the area of similar pixels around a
pixel is `n = sum exp(-((c - v) / diff_thr)^6)` over the pixels `v` of the disc `i^2 + j^2 <
radius^2` around it (the centre `c` included). A pixel with `n < geom_thr` has the response
`geom_thr - n`; corners are the strict maxima of the response among their eight neighbours.
Pixels closer than `edge` to the border have no response (so corners are at least `edge + 1`
away). `radius` is at most 9 and at most `edge`; the thresholds are positive; `feature_ratio` in
`(0, 1]` limits the corners to `floor(feature_ratio * H * W)`. If there are more, the first ones
in memory order (axis 0 fastest) are kept: that is what ArrayFire's CPU backend does, its CUDA
backend keeps an arbitrary subset.

Differences from ArrayFire: **`x` is the column and `y` the row**, like every detector of luma;
ArrayFire's `susan` (CPU and CUDA backend alike) returns the row as `x`, which makes its
coordinates unusable for the plots of the other detectors. The corners below the limit come in
tile order, not in memory order. An `edge` larger than the image gives `None` (ArrayFire raises
an error from half the larger dimension on).

## Difference of Gaussians

```rust
use luma::Array;

let img = Array::from_slice(&[1.0f32; 25], [5, 5])?;
// A constant image has no detail away from the (zero padded) border.
let d = img.dog(1, 2).to_vec()?;
assert!(d[12].abs() < 1e-6);
# Ok::<(), luma::LumaError>(())
```

`dog(radius1, radius2)` is the image blurred with the Gaussian kernel of `2 * radius1 + 1` cells
a side minus the image blurred with the one of `2 * radius2 + 1` (the kernel of `gaussian_kernel`
with the standard deviation `0.25 * cells`), zero border, same shape; it filters every `[H, W]`
plane of a batch. Each blur is a separable pass pair, the result is lazy and poisons on an
error. A radius of 0 is the identity; equal radii give zeros. ArrayFire takes signed radii (a
negative one gives garbage), luma unsigned ones, and ArrayFire's CUDA backend fails for kernels
of more than 17 cells, luma has no such limit.
