# Vision: ORB, SIFT and GLOH

Three feature detectors with descriptors, the functions of ArrayFire's `vision.h` of the same
names. Each takes an `[H, W]` `f32` image (axis 0 is the row) and returns the detected points as
`Keypoints` with the descriptors of the points next to them, or `None` if the image has none
(luma has no empty arrays). The count of points depends on the data, so each call **synchronizes
once** to read it back; everything before that is recorded without waiting.

| function | descriptor | type | shape |
|---|---|---|---|
| `orb` | 256-bit rotated BRIEF | `u32` | `[8, N]` |
| `sift` | 4 x 4 x 8 histograms of gradients | `f32` (integers `0..=255`) | `[128, N]` |
| `gloh` | gradient location and orientation histogram | `f32` (integers `0..=255`) | `[272, N]` |

`Keypoints.x` is the position along axis 1 (the column) and `y` the one along axis 0, in
the coordinates of the input image. Descriptor `i` is column `i` of the descriptor array.

## ORB

`img.orb(fast_thr, max_feat, scale_factor, levels, blur_img)` builds a pyramid of at most
`levels` images, each `scale_factor` times smaller than the previous one (bilinear resize), finds
FAST corners (arc of 9) on every level, keeps the strongest by Harris response (about
`max_feat / scale^level` on level `level`), measures the orientation of the 31 x 31 patch around
each corner by its intensity centroid and computes 256 pair tests of OpenCV's BRIEF pattern,
rotated by that orientation. With `blur_img` the tests read the level image smoothed with a 9-tap
Gaussian of sigma 2. The result has `score` the Harris response (scaled by 1e-12),
`orientation` in radians, `size` the patch size `31 * scale^level`.

```rust
use luma::Array;
# fn main() -> Result<(), luma::LumaError> {
let mut data = vec![0.0f32; 96 * 96];
for x in 40..60 { for y in 40..60 { data[y + 96 * x] = 200.0; } }
let img = Array::from_slice(&data, [96, 96])?;
// (A square this small has corners only at the finest level of a 2-level pyramid.)
if let Some((points, desc)) = img.orb(20.0, 100, 1.5, 2, false)? {
    assert_eq!(desc.shape().dims()[1], points.count());
}
# Ok(()) }
```

Two descriptors are compared by the Hamming distance of their 256 bits: the number of `1` bits of
the XOR of the eight words, `a.zip(b).map(|(p, q)| (p ^ q).count_ones()).sum()`.

## SIFT and GLOH

`img.sift(n_layers, contrast_thr, edge_thr, init_sigma, double_input, intensity_scale, feature_ratio)`
and `img.gloh(...)` with the same arguments (ArrayFire's defaults are `3, 0.04, 10.0, 1.6, true, 1/256,
0.05`) detect the extrema of the difference of Gaussians of a pyramid of `floor(log2(min side)) - 2`
octaves, refine them, reject low-contrast and edge-like ones, give every remaining point one
orientation per peak of a 36-bin histogram and describe it. SIFT describes a point by 4 x 4
histograms of 8 orientations, GLOH by a log-polar grid (a disc and two rings of 8 sectors, 16
orientations each: 272 values). `intensity_scale` is one over the largest value of the image (the
contrast thresholds are in units of the scaled image). `feature_ratio` bounds the extrema per octave
by that fraction of its pixels.

`orientation` is in radians in `[0, 2 pi)` (ArrayFire returns degrees), `size` the diameter of the
point in input pixels, `score` the absolute interpolated contrast.

```rust
use luma::Array;
# fn main() -> Result<(), luma::LumaError> {
// A Gaussian blob is a scale-space extremum.
let mut data = vec![0.0f32; 96 * 96];
for x in 0..96 { for y in 0..96 {
    let (dx, dy) = (x as f32 - 48.0, y as f32 - 48.0);
    data[y + 96 * x] = 200.0 * (-(dx * dx + dy * dy) / 50.0).exp();
} }
let img = Array::from_slice(&data, [96, 96])?;
let (points, sift) = img.sift(3, 0.04, 10.0, 1.6, true, 1.0 / 256.0, 0.05)?.expect("a blob");
let (_, gloh) = img.gloh(3, 0.04, 10.0, 1.6, true, 1.0 / 256.0, 0.05)?.expect("a blob");
assert_eq!(sift.shape().dims()[0], 128);
assert_eq!(gloh.shape().dims()[0], 272);
assert_eq!(sift.shape().dims()[1], points.count());
# Ok(()) }
```

## Differences from ArrayFire

Where ArrayFire's CPU and CUDA backends differ, luma follows CUDA. The points of luma agree with
ArrayFire's (the same points with the same orientations and descriptors up to rounding), the
following differs on purpose or by construction:

* **Order.** ArrayFire's order depends on its CUDA blocks (atomics) and on a sort. luma returns the
  levels (`orb`) or octaves (`sift`, `gloh`) finest first. Within an ORB level the points are sorted by
  decreasing Harris response; within a SIFT octave their order is not specified. Compare results
  as sets.
* **ORB.** The features of a level survive the cap of `0.14 * H * W` FAST candidates in the order of
  luma's tiles, not ArrayFire's. ArrayFire blurs its input image in place when `blur_img` is set,
  luma does not touch it. The last level takes what is left of `max_feat`, saturating at 0 where
  ArrayFire's unsigned arithmetic wraps. The kernel launches of all levels are queued without a
  readback; the Harris ranking uses a histogram to find the cut and ranks the candidates at the cut
  exactly.
* **SIFT and GLOH** reproduce ArrayFire's algorithm including its quirks: the 3 x 3 solver of the
  sub-pixel refinement leaves the offset along the scale at 0 (so the size of a point is the one of
  its layer), the cross derivative `d2D/dy ds` mixes in the pixel `(x - 1, y - 1)`, the first
  extremum threshold is applied to the unscaled image. Duplicates (two extrema that refine to the
  same location) are removed with a hash table, not a sort. Descriptor histograms are accumulated with
  integer atomics scaled by `65536 * intensity_scale` instead of float atomics: entries can differ
  by one from ArrayFire's, and an image whose range is far above `1 / intensity_scale` can overflow
  them.
* **Errors.** Invalid arguments (`levels == 0`, `scale_factor <= 1`, `n_layers == 0`, images smaller
  than 7 x 7 for ORB and 15 x 15 for SIFT) are `LumaError`s, not ArrayFire exceptions. Only `f32`
  images are supported (ArrayFire also takes `f64`).


