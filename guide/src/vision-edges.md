# Vision: Canny edges, anisotropic diffusion, mean shift

Three image operations of ArrayFire's `image.h`, built-in kernels that run on every backend and
follow ArrayFire's **CUDA** kernels step by step (its CPU backend disagrees with its CUDA
backend in several places; where it does, luma does what the CUDA backend does). All three take
`[H, W]` images (axis 0, the height, varies fastest), batch over axes 2 and 3 (mean shift of a
colour image takes `[H, W, 3]` and batches over axis 3), are lazy, record no synchronisation
and poison their result on an error.

## `canny`

```rust
use luma::{Array, CannyThreshold};

// A soft vertical edge near column 9 (a perfectly symmetric profile would tie between two
// columns and have no strict maximum).
let data: Vec<f32> = (0..400).map(|k| 100.0 * (1.0 + (((k / 20) as f32 - 9.2) / 1.5).tanh())).collect();
let img = Array::from_slice(&data, [20, 20])?;
let edges = img.canny(CannyThreshold::Manual, 0.2, 0.5, 3, false).to_vec()?;
assert!(edges[10 + 20 * 9] && !edges[10 + 20 * 5]);
# Ok::<(), luma::LumaError>(())
```

`canny(threshold, low, high, sobel_window, fast) -> Array<bool>` runs ArrayFire's pipeline:

1. a 5-tap Gaussian (`[0.11021, 0.23691, 0.30576, 0.23691, 0.11021]`, zero outside the image) in
   both directions, then the 3 x 3 Sobel derivatives of the result (the border mirrored without
   repeating the edge pixel, ArrayFire's `reflect101`) and the magnitude `sqrt(gx² + gy²)`, or
   `|gx| + |gy|` with `fast`;
2. non-maximal suppression of the interior pixels: a pixel keeps its magnitude if it is
   *strictly* larger than the two neighbours along the gradient direction, which fall between
   pixels and are linearly interpolated; everything else (and the outermost ring) becomes 0;
3. thresholds. With `CannyThreshold::Manual` `low` and `high` are **ratios of the largest
   suppressed magnitude of the whole array** (all planes of a batch share it). With
   `CannyThreshold::AutoOtsu` the high threshold comes from Otsu's method on the histogram of the
   suppressed magnitudes of each plane (`floor(max)` bins between 0 and the array's maximum, the
   zeros counted, the first maximum of the between-class variance; at most 16384 bins) and `low`
   is the *ratio of the high threshold* the low one has; `high` is ignored. The bin count follows
   `floor(max)`, so the result depends on the scale of the image like in ArrayFire: use images in
   `0..255`;
4. hysteresis: pixels at or above the high threshold are edges, and so are the pixels at or above
   the low threshold that are 8-connected to an edge through such pixels.

Only `sobel_window == 3` exists (ArrayFire rejects every other value as well).

**The hysteresis on the GPU.** Edge pixels are rare, so the work is organised around them. A first
kernel finds the connected components of the candidates inside every 32 x 8 tile in workgroup
memory and records which of them contain a strong pixel; a second links the components of
neighbouring tiles with a lock-free union-find (the larger root goes under the smaller one); a
third gives the final roots the strong flag and the last one writes the packed bits of the result.
The last three visit one invocation per 32 pixels and skip the empty words, so their cost follows
the number of edge pixels, and the whole detector is 6 launches (7 with Otsu) with nothing read
back. ArrayFire needs about twenty kernels and a host round trip per hysteresis sweep.

Differences from ArrayFire: its CPU backend marks the whole outer ring and mixes up the axes of
images that are not square (luma follows the CUDA backend: no ring, any shape). ArrayFire needs
images of at least 5 x 5; luma accepts any shape (an image without an interior pixel has no
edges). A constant image makes ArrayFire fail; luma gives the edges near the border that the zero
padding of the smoothing creates. Pixels whose suppressed magnitudes tie in exact arithmetic (the
sides of a perfectly flat square, say) can fall on either side in `f32`; luma mirrors the order of
ArrayFire's operations to land on the same side in most cases, but not all. The same goes for
Otsu thresholds when several bins tie (empty bins between the classes): ArrayFire's rounding
picks one, luma the lowest, which moves the weak threshold slightly.

## `anisotropic_diffusion`

```rust
use luma::{Array, DiffusionEq, FluxFunction};

let mut data = vec![0.0f32; 16 * 16];
data[8 + 16 * 8] = 80.0;
let img = Array::from_slice(&data, [16, 16])?;
let out = img.anisotropic_diffusion(0.1, 5.0, 3, FluxFunction::Exponential, DiffusionEq::Gradient).to_vec()?;
assert!(out[8 + 16 * 8] < 80.0 && out[7 + 16 * 8] > 0.0);
# Ok::<(), luma::LumaError>(())
```

`anisotropic_diffusion(timestep, conductance, iterations, flux, equation)` runs `iterations`
steps of Perona-Malik diffusion (`DiffusionEq::Gradient`) or modified curvature diffusion
(`DiffusionEq::Mcde`, which always uses the exponential flux). The conductance `K` is relative:
every step the exponent factor is `-1 / (2 K² mean(g²))` with `g` the gradient of the current
image and the mean over every element of the array. The flux through a pixel face is `c(x) *
(difference)` with `c = exp(x)` (`FluxFunction::Exponential`) or `c = 1 / (1 + x)`
(`FluxFunction::Quadratic`) of `x = factor * |grad u|²` at the face; outside the image the
nearest pixel is used.

ArrayFire's quadratic function has the sign of `x` inverted (the factor is negative), which makes
it exceed 1 and blow up where the gradient is large; luma reproduces it so that the results are
the same: prefer the exponential flux. There is no stability check of the time step (it is stable
up to about 0.125), as in ArrayFire, which rejects only `iterations == 0` and a conductance of 0
(luma too). A constant image comes back unchanged (ArrayFire's is NaN).

Every step is three launches: the gradient energy per tile, one workgroup that adds the tiles up
and writes the factor (ArrayFire reads the sum back to the host after every step), and the update
from one image into the other. luma reads and writes different images, so the result does not
depend on the order of the pixels; ArrayFire's CPU backend updates in place pixel by pixel.

## `mean_shift`

```rust
use luma::Array;

let mut data = vec![10.0f32; 8 * 8];
for j in 4..8 { for i in 0..8 { data[i + 8 * j] = 200.0; } }
let img = Array::from_slice(&data, [8, 8])?;
let out = img.mean_shift(1.0, 20.0, 2, false).to_vec()?;
assert_eq!((out[0], out[7 + 8 * 7]), (10.0, 200.0));
# Ok::<(), luma::LumaError>(())
```

`mean_shift(spatial_sigma, chromatic_sigma, iterations, is_color)` starts a walk at every pixel.
The window is `2 r + 1` pixels a side with `r = max(floor(1.5 * spatial_sigma), 1)`, cut at the
image. A step collects the window pixels whose squared colour distance (summed over the channels,
`[H, W, 3]` for `is_color`) from the walk's current colour is at most `chromatic_sigma²`, and
moves to the mean position and the mean colour of those, **both truncated to integers**
(ArrayFire's `__float2int_rz`). The walk stops early when the position did not move, or moved by
at most one pixel with a colour change of at most one. The result is the colour the walk ends
with. The truncation makes the filter meant for images of `0..255`: an image in `0..1` comes out
all zero, like in ArrayFire.

A workgroup stages the pixels around its 16 x 16 tile (as far as workgroup memory allows, up to
`r`) and the first step of every walk reads only from there; later steps move the window and read
global memory through the cache, so any number of steps stays exact. The result is bit-identical
to ArrayFire's (both backends of ArrayFire agree here); the reciprocal of the pixel count, whose
rounding decides the truncations, is corrected to the correctly rounded value on every backend.
