//! Interpolation on a regular grid: [`Array::approx1`] and [`Array::approx2`], and the sampler
//! the image transforms of [`crate::image`] share.
//!
//! The sampling runs in one kernel ([`crate::sampler`]) on every backend: a method is a set of taps
//! per axis (one, two or four neighbouring grid points) and weights that are polynomials in the
//! fractional position, and a 2-D method is the product of two axes. The input and the position
//! arrays are evaluated first and must fit one buffer each.

use std::panic::Location;

use crate::array::{Array, Failure};
use crate::error::LumaError;
use crate::ops::Outcome;
use crate::sampler::{broadcast_strides, Grid, Sampling, Source};
use crate::shape::Shape;

/// How [`Array::approx1`], [`Array::approx2`] and the image transforms interpolate between grid
/// points. The names follow ArrayFire's `af_interp_type`; the 2-D methods are the products of the
/// 1-D ones, so [`Interp::BILINEAR`] is [`Interp::Linear`] and [`Interp::BICUBIC`] is
/// [`Interp::Cubic`] (ArrayFire's `BILINEAR` and `BICUBIC`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Interp {
    /// The nearest grid point: `floor(x + 0.5)`, halves round up.
    Nearest,
    /// The grid point at or below the position: `floor(x)`.
    Lower,
    /// Linear interpolation between the two neighbours (bilinear in two dimensions).
    Linear,
    /// Linear interpolation with the fraction smoothed by a cosine: `(1 - cos(pi t)) / 2`.
    Cosine,
    /// Cubic interpolation of the four neighbours with ArrayFire's `AF_INTERP_CUBIC` polynomial
    /// `a0 t^3 + a1 t^2 + a2 t + y1`, `a0 = y3 - y2 - y0 + y1`, `a1 = y0 - y1 - a0`,
    /// `a2 = y2 - y0` (bicubic in two dimensions). It passes through the grid points but its
    /// slope there is not continuous.
    Cubic,
    /// Cubic interpolation with Catmull-Rom splines (`AF_INTERP_CUBIC_SPLINE`): the same four
    /// neighbours, continuous slope.
    CubicSpline,
}

#[allow(non_upper_case_globals)]
impl Interp {
    /// ArrayFire's `AF_INTERP_BILINEAR`: [`Interp::Linear`] in two dimensions.
    pub const BILINEAR: Interp = Interp::Linear;
    /// ArrayFire's `AF_INTERP_BICUBIC`: [`Interp::Cubic`] in two dimensions.
    pub const BICUBIC: Interp = Interp::Cubic;
    /// ArrayFire's `AF_INTERP_BILINEAR_COSINE`: [`Interp::Cosine`] in two dimensions.
    pub const BILINEAR_COSINE: Interp = Interp::Cosine;
    /// ArrayFire's `AF_INTERP_BICUBIC_SPLINE`: [`Interp::CubicSpline`] in two dimensions.
    pub const BICUBIC_SPLINE: Interp = Interp::CubicSpline;
}

/// The shape `a` and `b` broadcast to along every axis (`None` where both are larger than 1 and
/// differ).
fn broadcast(a: [usize; 4], b: [usize; 4]) -> Option<[usize; 4]> {
    let mut out = a;
    for k in 0..4 {
        out[k] = match (a[k], b[k]) {
            (x, y) if x == y => x,
            (1, y) => y,
            (x, 1) => x,
            _ => return None,
        };
    }
    Some(out)
}

fn invalid<T: crate::gpu_type::GpuType>(ctx: &crate::Context, shape: Shape, what: String, op: &'static str, location: &'static Location<'static>) -> Array<T> {
    let result: Outcome<Array<T>> = Err(Failure::New(LumaError::InvalidShape(what)));
    Array::resolve(ctx, shape, result, op, location)
}

impl Array<f32> {
    /// Linear (or other, see [`Interp`]) interpolation of this array along its first dimension at
    /// the positions `pos`, like ArrayFire's `approx1`.
    ///
    /// The array holds samples at the grid points `0, 1, .., n - 1` of dimension 0. `pos` gives
    /// the positions to interpolate at, in the same units. The result has `pos`'s extent along
    /// dimension 0; along every other dimension the extent is this array's, or `pos`'s where this
    /// array has extent 1 (batches: `pos` extents must be 1 or match this array's, and an extent
    /// of 1 repeats along the batch). Positions below `0`, above `n - 1` or NaN give `off_grid`.
    /// Neighbours past the ends repeat the end samples.
    ///
    /// Fails (poisoned array, [`LumaError::InvalidShape`]) on batch extents that are neither equal
    /// nor 1.
    ///
    /// ```
    /// use luma::{Array, Interp};
    /// let x = Array::from_slice(&[1.0f32, 4.0, 2.0], 3)?;
    /// let pos = Array::from_slice(&[0.5f32, 1.0, 1.5, 2.5], 4)?;
    /// let y = x.approx1(&pos, Interp::Linear, -1.0).to_vec()?;
    /// assert_eq!(y, vec![2.5, 4.0, 3.0, -1.0]);
    /// # Ok::<(), luma::LumaError>(())
    /// ```
    #[track_caller]
    pub fn approx1(&self, pos: &Array<f32>, method: Interp, off_grid: f32) -> Array<f32> {
        self.approx1_at(pos, 0, 0.0, 1.0, method, off_grid, Location::caller(), "approx1")
    }

    /// [`Array::approx1`] along dimension `dim`, for samples that sit at the positions
    /// `start, start + step, start + 2 step, ..` (ArrayFire's `approx1` with `interp_dim`,
    /// `idx_start` and `idx_step`): the grid coordinate of `pos` is `(pos - start) / step`.
    ///
    /// Fails (poisoned array) if `dim >= 4` or `step` is zero or not finite.
    #[track_caller]
    pub fn approx1_uniform(&self, pos: &Array<f32>, dim: usize, start: f32, step: f32, method: Interp, off_grid: f32) -> Array<f32> {
        self.approx1_at(pos, dim, start, step, method, off_grid, Location::caller(), "approx1_uniform")
    }

    #[allow(clippy::too_many_arguments)]
    fn approx1_at(&self, pos: &Array<f32>, dim: usize, start: f32, step: f32, method: Interp, off_grid: f32, location: &'static Location<'static>, op: &'static str) -> Array<f32> {
        let ctx = self.context();
        let (a, p) = (self.shape().dims(), pos.shape().dims());
        let fallback = pos.shape();
        if dim >= 4 || step == 0.0 || !step.is_finite() {
            return invalid(ctx, fallback, format!("{op}: dimension {dim}, step {step}"), op, location);
        }
        // Along `dim` the output follows `pos`; elsewhere the two broadcast.
        let (mut ea, mut ep) = (a, p);
        ea[dim] = 1;
        ep[dim] = 1;
        let Some(mut out) = broadcast(ea, ep) else {
            return invalid(ctx, fallback, format!("{op}: batch extents {a:?} of the input and {p:?} of the positions do not match"), op, location);
        };
        out[dim] = p[dim];
        let out = Shape::from(out);
        let result = self.approx_sample(pos, pos, out, [Grid { dim, start, step }, Grid { dim, start, step }], 1, p, method, off_grid);
        Array::resolve(ctx, out, result, op, location)
    }

    #[allow(clippy::too_many_arguments)]
    fn approx_sample(&self, pos0: &Array<f32>, pos1: &Array<f32>, out: Shape, grid: [Grid; 2], ndims: usize, pos_dims: [usize; 4], method: Interp, off_grid: f32) -> Outcome<Array<f32>> {
        let ctx = self.context();
        self.src()?;
        pos0.src()?;
        pos1.src()?;
        if !ctx.same_as(pos0.context()) || !ctx.same_as(pos1.context()) {
            return Err(LumaError::ContextMismatch.into());
        }
        let sampling = Sampling {
            src: self,
            pos: [pos0, pos1],
            out,
            source: Source::Positions,
            method,
            grid: &grid[..ndims],
            pos_strides: broadcast_strides(pos_dims),
            off_grid,
            scale: [1.0; 2],
            matrix: [0.0; 9],
            perspective: false,
        };
        Ok(sampling.run(ctx)?)
    }

    /// Interpolation along the first two dimensions at the positions `(pos0, pos1)`, like
    /// ArrayFire's `approx2`: output element `i` is this array sampled at coordinate
    /// `pos0[i]` along dimension 0 and `pos1[i]` along dimension 1. `pos0` and `pos1` have the
    /// same shape; the result has the extents of `pos0` along dimensions 0 and 1, and along dimensions 2 and 3
    /// this array's (a position batch of extent 1 repeats over this array's batch; positions with
    /// a larger extent where this array has 1 give that extent). Positions outside
    /// `0..=n - 1` of either dimension, or NaN, give `off_grid`. [`Interp::Linear`] is bilinear
    /// and [`Interp::Cubic`] bicubic here.
    ///
    /// Fails (poisoned array) if the shapes of `pos0` and `pos1` differ or the batches do not
    /// match.
    #[track_caller]
    pub fn approx2(&self, pos0: &Array<f32>, pos1: &Array<f32>, method: Interp, off_grid: f32) -> Array<f32> {
        self.approx2_at(pos0, pos1, [0, 1], [0.0, 0.0], [1.0, 1.0], method, off_grid, Location::caller(), "approx2")
    }

    /// [`Array::approx2`] along dimensions `dims`, for samples at `start[k] + step[k] * i` along
    /// dimension `dims[k]` (ArrayFire's `approx2` with `interp_dim0/1`, `idx_start_dim0/1` and
    /// `idx_step_dim0/1`). Fails (poisoned array) if the dimensions are equal or not below 4, or a
    /// step is zero or not finite.
    #[track_caller]
    #[allow(clippy::too_many_arguments)]
    pub fn approx2_uniform(&self, pos0: &Array<f32>, pos1: &Array<f32>, dims: [usize; 2], start: [f32; 2], step: [f32; 2], method: Interp, off_grid: f32) -> Array<f32> {
        self.approx2_at(pos0, pos1, dims, start, step, method, off_grid, Location::caller(), "approx2_uniform")
    }

    #[allow(clippy::too_many_arguments)]
    fn approx2_at(
        &self,
        pos0: &Array<f32>,
        pos1: &Array<f32>,
        dims: [usize; 2],
        start: [f32; 2],
        step: [f32; 2],
        method: Interp,
        off_grid: f32,
        location: &'static Location<'static>,
        op: &'static str,
    ) -> Array<f32> {
        let ctx = self.context();
        let fallback = pos0.shape();
        if pos0.shape() != pos1.shape() {
            let result: Outcome<Array<f32>> = Err(LumaError::ShapeMismatch { lhs: pos0.shape().dims(), rhs: pos1.shape().dims() }.into());
            return Array::resolve(ctx, fallback, result, op, location);
        }
        if dims[0] >= 4 || dims[1] >= 4 || dims[0] == dims[1] || step.iter().any(|s| *s == 0.0 || !s.is_finite()) {
            return invalid(ctx, fallback, format!("{op}: dimensions {dims:?}, steps {step:?}"), op, location);
        }
        let (a, p) = (self.shape().dims(), pos0.shape().dims());
        let (mut ea, mut ep) = (a, p);
        for d in dims {
            ea[d] = 1;
            ep[d] = 1;
        }
        let Some(mut out) = broadcast(ea, ep) else {
            return invalid(ctx, fallback, format!("{op}: batch extents {a:?} of the input and {p:?} of the positions do not match"), op, location);
        };
        for d in dims {
            out[d] = p[d];
        }
        let out = Shape::from(out);
        let grid = [Grid { dim: dims[0], start: start[0], step: step[0] }, Grid { dim: dims[1], start: start[1], step: step[1] }];
        let result = self.approx_sample(pos0, pos1, out, grid, 2, p, method, off_grid);
        Array::resolve(ctx, out, result, op, location)
    }
}
