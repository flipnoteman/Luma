//! Interpolation on a regular grid: [`Array::approx1`] and [`Array::approx2`], and the sampler
//! the image transforms of [`crate::image`] share.
//!
//! Everything is built from lazy elementwise operations and gathers ([`Array::take`]), so it runs
//! on every backend, fuses into the expressions around it (positions computed from other arrays,
//! a result that is scaled or reduced right away) and reads each input element once per tap.
//! A method is a set of taps per axis (one, two or four neighbouring grid points) and weights
//! that are polynomials in the fractional position; a 2-D method is the product of two axes.

use std::panic::Location;
use std::sync::Arc;

use crate::array::{Array, Failure};
use crate::error::LumaError;
use crate::ops::{select, Outcome};
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

/// The taps of one axis: grid indices (in elements of that axis, clamped into the axis) and the
/// weight of each. No weights means a single tap of weight one.
struct Axis {
    idx: Vec<Array<u32>>,
    w: Vec<Array<f32>>,
}

/// The weights of the four taps `y0..y3` of a cubic method at fraction `t`.
fn cubic_weights(t: &Array<f32>, spline: bool) -> Vec<Array<f32>> {
    let t2 = t * t;
    let t3 = &t2 * t;
    if spline {
        // Catmull-Rom.
        vec![
            (&t2 * 2.0 - &t3 - t) * 0.5,
            (&t3 * 3.0 - &t2 * 5.0 + 2.0) * 0.5,
            (&t2 * 4.0 - &t3 * 3.0 + t) * 0.5,
            (&t3 - &t2) * 0.5,
        ]
    } else {
        // `a0 t^3 + a1 t^2 + a2 t + y1`, expanded in `y0..y3`.
        vec![&t2 * 2.0 - &t3 - t, &t3 - &t2 * 2.0 + 1.0, &t2 + t - &t3, &t3 - &t2]
    }
}

/// The taps and weights of `method` at the positions `x` (in grid units) of an axis of `n` points.
/// Neighbours outside the axis repeat the edge point.
fn axis(x: &Array<f32>, n: usize, method: Interp) -> Axis {
    let hi = (n - 1) as f32;
    let index = |v: Array<f32>| v.clamp(0.0, hi).to_u32();
    match method {
        Interp::Nearest => Axis { idx: vec![index((x + 0.5).floor())], w: vec![] },
        Interp::Lower => Axis { idx: vec![index(x.floor())], w: vec![] },
        Interp::Linear | Interp::Cosine => {
            let f = x.floor();
            let t = x - &f;
            let t = if method == Interp::Cosine { (&(&t * std::f32::consts::PI).cos() * -1.0 + 1.0) * 0.5 } else { t };
            Axis { idx: vec![index(f.clone()), index(&f + 1.0)], w: vec![&t * -1.0 + 1.0, t] }
        }
        Interp::Cubic | Interp::CubicSpline => {
            let f = x.floor();
            let t = x - &f;
            let w = cubic_weights(&t, method == Interp::CubicSpline);
            Axis { idx: vec![index(&f - 1.0), index(f.clone()), index(&f + 1.0), index(&f + 2.0)], w }
        }
    }
}

/// `src` sampled with `method` at the coordinates `coords`: for every element of the output
/// shape `out`, the interpolated value at `(coords[0][i], coords[1][i], ..)` along the grid
/// dimensions `dims`, within that element's own position along the other dimensions of `src`
/// (batch dimensions of extent 1 in `src` are repeated). Every coordinate array has the shape
/// `out`. The result is not masked: positions outside the grid read repeated edge points.
pub(crate) fn sample(src: &Array<f32>, out: Shape, grid: &[(usize, &Array<f32>)], method: Interp) -> Array<f32> {
    let ctx = src.context();
    let dims = src.shape().dims();
    let stride = |d: usize| -> u32 { dims[..d].iter().product::<usize>() as u32 };
    // The offset of the element's batch position: the dimensions that are not on the grid.
    let mut base: Option<Array<u32>> = None;
    for d in 0..4 {
        if dims[d] > 1 && out.dims()[d] > 1 && grid.iter().all(|&(g, _)| g != d) {
            let term = match Array::<u32>::range_on(ctx, out, d) {
                Ok(r) => &r * stride(d),
                Err(e) => return Array::poisoned(ctx, out, Arc::new(e)),
            };
            base = Some(match base {
                Some(b) => b + term,
                None => term,
            });
        }
    }
    let axes: Vec<(Axis, u32)> = grid.iter().map(|&(d, x)| (axis(x, dims[d], method), stride(d))).collect();
    // Taps of the first grid dimension are summed innermost.
    let (first, rest) = axes.split_first().expect("sample needs a grid dimension");
    let line = |offset: Option<Array<u32>>| -> Array<f32> {
        let (ax, st) = first;
        let mut sum: Option<Array<f32>> = None;
        for (k, i) in ax.idx.iter().enumerate() {
            let mut at = if *st == 1 { i.clone() } else { i * *st };
            if let Some(o) = &offset {
                at = at + o;
            }
            let v = src.take(&at);
            let v = if ax.w.is_empty() { v } else { &ax.w[k] * &v };
            sum = Some(match sum {
                Some(s) => s + v,
                None => v,
            });
        }
        sum.expect("an axis has taps")
    };
    match rest {
        [] => line(base),
        [(second, st2)] => {
            let mut sum: Option<Array<f32>> = None;
            for (k, j) in second.idx.iter().enumerate() {
                let row = j * *st2;
                let offset = match &base {
                    Some(b) => row + b,
                    None => row,
                };
                let v = line(Some(offset));
                let v = if second.w.is_empty() { v } else { &second.w[k] * &v };
                sum = Some(match sum {
                    Some(s) => s + v,
                    None => v,
                });
            }
            sum.expect("an axis has taps")
        }
        _ => unreachable!("at most two grid dimensions"),
    }
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

/// Where `x` lies inside `0..=n-1` (NaN does not).
fn inside(x: &Array<f32>, n: usize) -> Array<bool> {
    x.ge(0.0) & x.le((n - 1) as f32)
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
        let x = if start == 0.0 && step == 1.0 { pos.clone() } else { (pos - start) / step };
        let x = x.broadcast_to(out);
        let ok = inside(&x, a[dim]);
        let v = sample(self, out, &[(dim, &x)], method);
        select(&ok, v, off_grid)
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
        let coord = |pos: &Array<f32>, k: usize| {
            let x = if start[k] == 0.0 && step[k] == 1.0 { pos.clone() } else { (pos - start[k]) / step[k] };
            x.broadcast_to(out)
        };
        let (x, y) = (coord(pos0, 0), coord(pos1, 1));
        let ok = inside(&x, a[dims[0]]) & inside(&y, a[dims[1]]);
        let v = sample(self, out, &[(dims[0], &x), (dims[1], &y)], method);
        select(&ok, v, off_grid)
    }
}
