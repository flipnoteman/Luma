# Interpolation, image transforms and filters

These functions follow ArrayFire's `approx1`, `approx2`, `resize`, `rotate`, `translate`, `scale`,
`skew`, `transform`, `transformCoordinates`, `fir` and `iir`, with its conventions, on `f32`
arrays. The first dimension is the fastest, as everywhere in Luma. They run on every backend.

## Interpolation

`x.approx1(&pos, method, off_grid)` reads the samples of `x` along dimension 0 as the values at
the grid points `0, 1, .., n - 1` and interpolates them at the positions `pos`. Positions below
`0`, above `n - 1` and NaN give `off_grid`. `approx2` does the same along dimensions 0 and 1
with two position arrays. `Interp` is `Nearest`, `Lower` (floor),
`Linear`, `Cosine`, `Cubic` and `CubicSpline` (Catmull-Rom); in two dimensions `Linear` is
bilinear and `Cubic` bicubic. Further dimensions are batches: positions with extent 1 along
them are shared by the whole batch. `approx1_uniform` and `approx2_uniform` take the dimension
and the `start` and `step` of the grid (the sample `i` sits at `start + step * i`).

```rust
use luma::*;

let x = Array::from_slice(&[1.0f32, 4.0, 2.0], 3)?;
let pos = Array::from_slice(&[0.5f32, 1.0, 1.5, 2.5], 4)?;
assert_eq!(x.approx1(&pos, Interp::Linear, -1.0).to_vec()?, vec![2.5, 4.0, 3.0, -1.0]);
assert_eq!(x.approx1(&pos, Interp::Nearest, -1.0).to_vec()?, vec![4.0, 4.0, 2.0, -1.0]);
# Ok::<(), luma::LumaError>(())
```

Each call is one kernel that finds the coordinates, gathers the taps of the method and combines
them in registers. The input and the positions are evaluated first and must fit one buffer.

### Reading a source through the texture units

A kernel that reads a two-dimensional image at positions it computes (`approx2`, `resize`,
`rotate`, `transform`) can fetch the four texels of a bilinear footprint, or the sixteen of a
bicubic one as four such blocks, with the texture unit's *gather* (`textureGather` in WGSL,
`tld4` in CUDA) instead of one load per texel. A gather is exact: it returns the texels, unfiltered,
and the weights are computed by the kernel as before, so the results are bit for bit those of the
loads on every backend. Luma decides by its cost model ([`read_path`](performance.md#reading-a-source-through-a-texture)):
on an RTX 3090 and an RTX 3080 Laptop (the CUDA backend reads the source's own buffer as a pitched
texture, a view that costs nothing and is kept with the array) the gather takes 7-10 % off `approx2`
bicubic, 12-20 % off a bilinear `rotate` / `transform` and `resize` upwards, and it is chosen for
those; memory-bound kernels (nearest, `approx2` bilinear) and the bicubic `resize`, whose loads
the column of an output block shares, keep the loads. A backend that has to copy the source to
make a texture (wgpu) reads the buffer, as does the CPU backend, which has no textures.
Nothing in the API changes and the choice is not configurable.

`Interp::LinearHardware` is the explicit opt-in to the texture unit's own bilinear *filter*: one
fetch per output, but the hardware keeps 8 bits of the fraction of a position (the weights are
multiples of 1/256), so along each axis a result is off from `Interp::Linear`'s by up to the
difference of the two neighbouring samples over 256. Luma never chooses it. Where the source cannot
be a texture (more than two dimensions, the CPU backend, a width the device cannot read in place on
a copying backend) it is exactly `Interp::Linear`.

## Images

An image is an array whose dimensions 0 and 1 are the image axes; the dimensions after them
(colour channels, a batch) all get the same transform. `resize(d0, d1, method)` (or
`resize_by(scale0, scale1, method)`) samples the input at `x / (d0 / n0)`, so the first pixel is the
input's first and there is no half-pixel offset. `rotate(theta, crop, method)` turns about the
centre (the bounding box when `crop` is false), `translate`, `scale`, `skew` and `transform`
give every output pixel the input value at a position that is an affine or projective function
of the output pixel's coordinates (a 2 x 3 or 3 x 3 matrix, the output of
`transform_coordinates` for the corners). Pixels that come from outside the image are 0.

```rust
use luma::*;

let img = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], [3, 3])?;
// An affine matrix [a, b, tx, c, d, ty]: with `inverse == false` it maps input to output
// coordinates, so the content moves one pixel along dimension 0.
let moved = img.transform([1.0, 0.0, 1.0, 0.0, 1.0, 0.0], 3, 3, Interp::Nearest, false);
assert_eq!(moved.to_vec()?, vec![0.0, 1.0, 2.0, 0.0, 4.0, 5.0, 0.0, 7.0, 8.0]);
let big = img.resize(6, 6, Interp::Linear);
assert_eq!(big.shape().dims(), [6, 6, 1, 1]);
# Ok::<(), luma::LumaError>(())
```

Where ArrayFire's behaviour is not what its documentation says, Luma follows ArrayFire's
results: `translate(t0, t1)` reads the input at `(x + t0, y + t1)` (the content moves by `-t`),
ArrayFire's `rotate` rounds the cosine and sine to three decimals (Luma does not, so the two differ by a fraction of a pixel),
the inverse of a general affine matrix in `transform(.., inverse = false)` is the true inverse where ArrayFire's is
only right for rotations, scalings and translations, and `Interp::Cosine` in two dimensions
applies the cosine smoothing to both axes where ArrayFire's `BILINEAR_COSINE` is plain bilinear.

## Filters

`x.fir(&b)` is the finite impulse response filter `y[i] = b[0] x[i] + b[1] x[i-1] + ..`, with the
signal's shape; batches (dimensions 1 to 3) are filtered independently, with one filter for all or one
per signal. Filters up to 256 taps are one kernel; longer ones are blocks of taps, and from 384 taps
on a single long vector uses the FFT convolution.

`x.iir(&b, &a)` is the recursive filter
`a[0] y[i] = b[0] x[i] + .. - a[1] y[i-1] - ..` (host coefficient slices, feedback order up to
`IIR_MAX_ORDER`), and `x.iir_sos(&sections)` a cascade of biquads `[b0, b1, b2, a0, a1, a2]`. The
recursion is evaluated in parallel over the length of the signal: the signal is cut in chunks,
the state at the end of each chunk is found from zero, the chunk transitions (constant matrix
powers) are folded into the state entering every chunk, and the chunks run again from there.
A single signal of 16M samples therefore uses the whole GPU, and a batch of signals
runs side by side.

```rust
use luma::*;

let x = Array::from_slice(&[1.0f32, 0.0, 0.0, 0.0], 4)?;
// y[i] = x[i] + 0.5 y[i-1]: the impulse response is 1, 0.5, 0.25, ..
assert_eq!(x.iir(&[1.0], &[1.0, -0.5]).to_vec()?, vec![1.0, 0.5, 0.25, 0.125]);
let b = Array::from_slice(&[0.5f32, 0.25], 2)?;
assert_eq!(x.fir(&b).to_vec()?, vec![0.5, 0.25, 0.0, 0.0]);
# Ok::<(), luma::LumaError>(())
```

`fir` and `iir` read evaluated arrays (a pending expression is computed first) and need arrays
that fit one buffer.
