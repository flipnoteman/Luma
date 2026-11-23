# Features, matching and image analysis

`luma::vision` has the feature detectors, descriptors and matchers of ArrayFire's `vision.h`
and the analysis functions of `image.h` that are not filters (those are in
[Images and filters](images.md)). They are built-in kernels: every backend runs them, and
they take `[H, W]` images (`H` along axis 0, the fastest axis, as in ArrayFire).

The chapters that follow cover the rest: [Harris and SUSAN corners, difference of
Gaussians](vision-corners.md), [template matching and the nearest-neighbour and Hamming
matchers](vision-matching.md), [Canny edges, anisotropic diffusion and mean shift](vision-edges.md),
[connected components and deconvolution](vision-regions.md) and [ORB, SIFT and
GLOH](vision-descriptors.md). This chapter has `fast` and `homography`.

## Points: `Keypoints`

A detector returns [`Keypoints`], ArrayFire's `features`: five one-dimensional `f32` arrays `x`,
`y`, `score`, `orientation` and `size` of the same length. As in ArrayFire `x` is the index along
**axis 1** (the column) and `y` the index along **axis 0** (the row). Detectors that have no use
for the orientation and size fill them with 0 and 1.

The number of points depends on the data: a detector counts them on the device, writes them
at their final place and then reads the count back, so it **synchronizes once** (about as fast
as a `filter`) and returns `Result<Option<Keypoints>>`: `None` for an image without a point (luma has
no arrays without elements), an `Err` for a bad argument or a failed device.

## `fast`

`image.fast(threshold, arc_length, non_max, feature_ratio, edge)` is the FAST corner detector
(ArrayFire's `fast`): a pixel is a corner when `arc_length` (9 to 16) consecutive pixels of the
circle of radius 3 around it are all brighter than the pixel plus `threshold` or all darker than
it minus `threshold`. The score is the larger of the sums of `|ring - pixel| - threshold` over the
brighter and over the darker ring pixels; `non_max` keeps a corner only if it scores strictly more
than its eight neighbours; `edge` (at least 3) is the width of the border without corners and
`feature_ratio` bounds the number of points to that share of the pixels.

```rust
use luma::*;

// A bright dot on a dark image is one corner.
let mut data = vec![0.0f32; 16 * 16];
data[8 + 16 * 8] = 100.0;
let img = Array::from_slice(&data, [16, 16])?;
let corners = img.fast(20.0, 9, true, 0.1, 3)?.expect("one corner");
assert_eq!(corners.count(), 1);
assert_eq!(corners.x.to_vec()?, [8.0]);
assert_eq!(corners.y.to_vec()?, [8.0]);
# Ok::<(), luma::LumaError>(())
```

Two launches over tiles of 32 x 8 pixels do the work. The first stages a tile with its halo in
workgroup memory, scores it (a bit-mask arc test with an early reject on the four cardinal
pixels), suppresses the non-maxima, writes the surviving scores to an image and adds the tile's
count to the counts of its block of tiles and to a total by atomics. The second launch gives every
tile the offset at which it writes (the counts of the blocks before its block plus those of the
tiles before it, summed by the workgroup itself, so there is no scan launch) and ranks the tile's
points by the bit masks of its rows; a tile without points exits after reading its count. The only
readback is the total. The points come in the order of the tiles (tiles along axis 0 first),
within a tile axis 0 first; the limit `feature_ratio` keeps the first points of that order.
Scores and points equal ArrayFire's `fast` on 99.4 % of its points; the rest differ in the last
bits or lie in the border band. On an RTX 3080 Laptop (CUDA) `fast` of a 1080p image takes
0.22 ms against ArrayFire's 0.29 ms and of a 4K image 0.94 ms against 1.09 ms.

Differences from ArrayFire: its `fast` limits the number of corner *candidates* before the
non-maximal suppression and lists the points row by row; on its CPU backend the suppression
leaves out the pixels within `edge + 2` of the border and mixes up the two axes on images that
are not square. luma applies the limit after the suppression and suppresses everywhere inside
`edge`.

## `homography`

`homography(x_src, y_src, x_dst, y_dst, method, inlier_thr, iterations)` estimates the perspective
transform between matching points (ArrayFire's `homography`): every one of `iterations`
hypotheses is the direct linear transform of four random point pairs and is scored over all the
points, RANSAC by the number of points that land within `inlier_thr` of their destination, least
median of squares (`HomographyMethod::Lmeds`) by the median of the squared errors. It returns the
`[3, 3]` matrix (its nine values in memory order are the rows of `H`, `(x', y', 1) ~ H (x, y, 1)`,
scaled so that `H[2][2] = 1`) and the inlier count as a one-element `u32` array: nothing is read
back until you ask. On an RTX 3080 Laptop (CUDA) RANSAC of 1000 points and 1000 iterations takes
0.05 ms against ArrayFire's 1.3 ms.

```rust
use luma::*;

let x: Vec<f32> = (0..8).map(|i| (i * 7 % 11) as f32).collect();
let y: Vec<f32> = (0..8).map(|i| (i * 5 % 13) as f32).collect();
let (xd, yd): (Vec<f32>, Vec<f32>) = (x.iter().map(|v| v + 3.0).collect(), y.iter().map(|v| v - 2.0).collect());
let up = |v: &[f32]| Array::from_slice(v, v.len());
let (h, inliers) = homography(&up(&x)?, &up(&y)?, &up(&xd)?, &up(&yd)?, HomographyMethod::Ransac, 0.5, 50);
assert_eq!(inliers.item()?, 8);
assert!((h.to_vec()?[2] - 3.0).abs() < 1e-3);
# Ok::<(), luma::LumaError>(())
```

One workgroup scores one hypothesis (an 8 x 8 direct linear transform on normalised coordinates);
a second launch picks the best. The samples come from a fixed hash of the hypothesis number,
so a call is deterministic (ArrayFire's differ: compare the recovered transform, not the
matrices). As in ArrayFire the matrix is the best sample's, not refined over its inliers.
