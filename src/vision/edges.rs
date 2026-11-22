//! Canny edges, anisotropic diffusion and mean shift: ArrayFire's `canny`,
//! `anisotropicDiffusion` and `meanShift`.
//!
//! * [`Array::canny`](crate::Array::canny): smoothing, Sobel gradient, non-maximal suppression
//!   and hysteresis as a handful of launches, with a GPU-side union-find for the hysteresis and
//!   no synchronisation (see `canny`);
//! * [`Array::anisotropic_diffusion`](crate::Array::anisotropic_diffusion): Perona-Malik and
//!   modified curvature diffusion, three launches per iteration (see `diffusion`);
//! * [`Array::mean_shift`](crate::Array::mean_shift): one mean-shift walk per pixel with the
//!   window staged in workgroup memory (see `meanshift`).
//!
//! All three take `[H, W]` images and batch over axes 2 and 3 as ArrayFire does (mean shift of a
//! colour image takes `[H, W, 3]` and batches over axis 3).

mod canny;
mod diffusion;
mod meanshift;

pub use canny::CannyThreshold;
pub use diffusion::{DiffusionEq, FluxFunction};

use crate::array::Failure;
use crate::context::Context;
use crate::ops::Outcome;
use crate::shape::Shape;

/// One kernel of this group (key of its cache entry in the context).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum Kernel {
    Canny(canny::Kernel),
    Diffusion(diffusion::Kernel),
    MeanShift(meanshift::Kernel),
}

pub(crate) fn source(kernel: Kernel) -> String {
    match kernel {
        Kernel::Canny(k) => canny::source(k),
        Kernel::Diffusion(k) => diffusion::source(k),
        Kernel::MeanShift(k) => meanshift::source(k),
    }
}

/// `[H, W]` and the number of planes (axes 2 and 3 together) of an image.
pub(super) fn geometry(shape: Shape) -> Outcome<(u32, u32, u32)> {
    let d = shape.dims();
    Ok((super::to_u32(d[0], "height")?, super::to_u32(d[1], "width")?, super::to_u32(d[2] * d[3], "plane count")?))
}

/// The workgroups of a launch over `tiles_x` by `tiles_y` tiles of every plane: the planes are
/// stacked along the second dimension (a kernel numbers its tile `wid.y % tiles_y` and its plane
/// `wid.y / tiles_y`).
pub(super) fn plane_grid(ctx: &Context, tiles_x: u32, tiles_y: u32, planes: u32) -> Outcome<[u32; 3]> {
    let rows = tiles_y.checked_mul(planes).ok_or_else(|| Failure::from(crate::error::LumaError::InvalidShape("too many tiles".into())))?;
    super::tile_grid(ctx, tiles_x, rows)
}
