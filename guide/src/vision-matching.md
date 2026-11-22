# Template matching, nearest-neighbour and Hamming matchers

`Array::match_template`, `nearest_neighbour` and `hamming_matcher` are ArrayFire's
`matchTemplate`, `nearestNeighbour` and `hammingMatcher`. They are built-in kernels that run on
every backend and need arrays that fit one buffer.

## Template matching

`image.match_template(&template, MatchType)` scores, for every pixel of the `[H, W]` image, the
window of the template's size that **starts** at that pixel (axis 0 first); pixels past the image
count as 0. The result has the shape of the image, and axes 2 and 3 of the image are a batch (the
template is one `[h, w]` array for all planes). The metrics, with `w` the window, `t` the template
and `mean` taken over the `h * w` pixels of the (zero-padded) window and of the template:

| `MatchType` | value |
| --- | --- |
| `Sad`, `Ssd` | `sum abs(w - t)`, `sum (w - t)^2` |
| `Zsad`, `Zssd` | the same with `w - mean(w)` and `t - mean(t)` |
| `Lsad`, `Lssd` | the same with `t` scaled by `mean(w) / mean(t)` |
| `Ncc` | `sum(w t) / sqrt(sum w^2 * sum t^2)` |
| `Zncc` | `Ncc` of `w - mean(w)` and `t - mean(t)` |

```rust
use luma::{Array, MatchType};

let image = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], [3, 2])?;
let template = Array::from_slice(&[2.0f32, 3.0], [2, 1])?;
// The window at the last row reaches one pixel past the image (0).
let sad = image.match_template(&template, MatchType::Sad);
assert_eq!(sad.to_vec()?, [2.0, 0.0, 4.0, 4.0, 6.0, 7.0]);
# Ok::<(), luma::LumaError>(())
```

Differences from ArrayFire (both of its CPU and CUDA backends agree on the formulas above, which
were checked against them):

* `Ncc` and `Zncc` work; ArrayFire rejects them with an invalid argument error. A window whose
  denominator is 0 scores 0.
* A template of a single row or column works (ArrayFire wants a 2-D template); an image with a
  fourth axis is matched plane by plane (ArrayFire computes only the first plane).
* A template larger than the image is accepted, as in ArrayFire.
* Float sums are accumulated in another order: results agree to a few ulps per term.

A workgroup of 32 x 8 invocations computes a tile of 128 x 32 outputs (64 x 32 on devices with
16 KB of workgroup memory), 16 per invocation. It stages the tile's window of the image and the
template in workgroup memory (a template too large for it is staged in slabs, one rectangle after
the other) and every invocation slides down the template's columns keeping the image values of its
outputs in registers: one shared-memory read of the image and one broadcast read of the template
serve four accumulations. The metrics that subtract the mean of the window sum it first with the
same walk.

## Nearest neighbours

`nearest_neighbour(&query, &train, dist_dim, n_dist, DistType)` returns `(index, distance)`, both
`[n_dist, nq]`: column `j` lists the `n_dist` train features nearest to query `j`, closest first.
With `dist_dim` 0 a feature is a column of `[len, n]`, with 1 a row of `[n, len]`. `DistType` is
`Sad`, `Ssd` or (for `u32` features) `Shd`, the Hamming distance; `hamming_matcher` is the last
one on `u32` descriptors (256-bit ORB descriptors are `[8, n]`).

```rust
use luma::{hamming_matcher, nearest_neighbour, Array, DistType};

let train = Array::from_slice(&[0.0f32, 1.0, 4.0, 4.0, 9.0, 9.0], [2, 3])?;
let query = Array::from_slice(&[0.0f32, 0.0, 5.0, 5.0], [2, 2])?;
let (index, dist) = nearest_neighbour(&query, &train, 0, 2, DistType::Ssd)?;
assert_eq!(index.to_vec()?, [0, 1, 1, 2]);
assert_eq!(dist.to_vec()?, [1.0, 32.0, 2.0, 32.0]);

let words = Array::from_slice(&[0u32, 0, 0xFF, 0, 0xF, 1], [2, 3])?;
let probe = Array::from_slice(&[0x7u32, 0], [2, 1])?;
let (index, dist) = hamming_matcher(&probe, &words, 0, 2)?;
assert_eq!((index.to_vec()?, dist.to_vec()?), (vec![2, 0], vec![2, 3]));
# Ok::<(), luma::LumaError>(())
```

Features are `f32` (`f32` distances), `u32` (`u32` distances, also for `Sad` and `Ssd`; unsigned
differences, sums wrap) or `i32` (`i32` distances). Differences from ArrayFire:

* ArrayFire also takes `u8`, `u16`, `i16` and `f64` features; luma has no arrays of the first
  three, and `f64` is not supported here.
* `Shd` is only defined for `u32`, as in ArrayFire (which rejects `f32`/`i32` for it).
* `n_dist` is at most 256 and at most the number of train features (as in ArrayFire).
* Ties are broken by the smaller train index. ArrayFire's CUDA backend does the same; its CPU
  backend returns equal distances in an arbitrary order.
* NaN distances sort after all others.
* The results are `Result`s read from arrays that are already computed; the operation does not
  synchronize, but its errors are returned directly, not as a poisoned array.

For `n_dist <= 8` one kernel does everything: a workgroup takes 64 queries and a range of train
blocks (the ranges split over workgroups when there are few queries), computes the distances 64 x
64 at a time (operands staged 16 elements at a time in workgroup memory, 4 x 4 distances per
invocation in registers, like a small matrix product) and keeps the sorted list of the smallest
ones of every invocation in registers; no distance is ever written. A second launch merges the
lists. For larger `n_dist` the distances of a chunk of queries are written to a scratch matrix and
a workgroup per query keeps a sorted list in workgroup memory, merging 256 candidates at a time
with a bitonic network (only the blocks that contain a candidate better than the list's worst are
sorted).
