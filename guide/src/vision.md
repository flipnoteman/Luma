# Features, matching and image analysis

`luma::vision` has the feature detectors, descriptors and matchers of ArrayFire's `vision.h`
and the analysis functions of `image.h` that are not filters (those are in
[Images and filters](images.md)). They are built-in kernels: every backend runs them, and
they take `[H, W]` images (`H` along axis 0, the fastest axis, as in ArrayFire).

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

Two passes over tiles of 32 x 8 pixels do the work: the first counts the corners of every tile
(nothing of the score image is stored), a one-workgroup scan turns the counts into offsets and the
second recomputes the scores and writes every tile's points at its offset. The points come in the
order of the tiles (tiles along axis 0 first), within a tile axis 0 first; the limit
`feature_ratio` keeps the first points of that order.

Differences from ArrayFire: its `fast` limits the number of corner *candidates* before the
non-maximal suppression and lists the points row by row; on its CPU backend the suppression
leaves out the pixels within `edge + 2` of the border and mixes up the two axes on images that
are not square. luma applies the limit after the suppression and suppresses everywhere inside
`edge`.
