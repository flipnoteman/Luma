//! Geometric transforms of images: [`Array::resize`], [`Array::rotate`], [`Array::translate`],
//! [`Array::scale`], [`Array::skew`], [`Array::transform`] and [`transform_coordinates`].
//!
//! An image is an array whose dimensions 0 and 1 are the two image axes (`x` along dimension 0,
//! `y` along dimension 1, as in ArrayFire); dimensions 2 and 3 are channels or batches that all
//! undergo the same transform. Every output pixel looks up where it comes from in the input
//! (the *inverse* map, an affine or projective function of the pixel coordinates), and
//! interpolates there with an [`Interp`] method. The functions are lazy elementwise expressions
//! over gathers (see [`crate::interp`]): they run on every backend and fuse with the
//! operations before and after them. They follow ArrayFire's `resize`, `rotate`, `translate`,
//! `scale`, `skew`, `transform` and `transformCoordinates`, with its conventions for the output
//! sizes, centres and edges.

use std::panic::Location;
use std::sync::Arc;

use crate::array::{Array, Failure};
use crate::context::{Context, default_context};
use crate::error::{LumaError, Result};
use crate::interp::{sample, Interp};
use crate::ops::{select, Outcome};
use crate::shape::Shape;

/// A 2-D affine (2 x 3) or projective (3 x 3) matrix for [`Array::transform`]. Implemented for
/// `[f32; 6]` and `[[f32; 3]; 2]` (the rows `[a, b, tx]` and `[c, d, ty]` of an affine map:
/// `x' = a x + b y + tx`, `y' = c x + d y + ty`) and for `[f32; 9]` and `[[f32; 3]; 3]` (a third
/// row `[g, h, i]` divides: `w = g x + h y + i`, `x' = (a x + b y + tx) / w`, ..), always in
/// row-major order, the order of ArrayFire's `transform` matrices in memory.
pub trait TransformMatrix: matrix_sealed::Sealed {
    #[doc(hidden)]
    fn into_matrix(self) -> Matrix3;
}

/// A 3 x 3 matrix in row-major order; the last row of an affine matrix is `[0, 0, 1]`.
#[doc(hidden)]
#[derive(Clone, Copy, Debug)]
pub struct Matrix3 {
    m: [f64; 9],
    perspective: bool,
}

mod matrix_sealed {
    pub trait Sealed {}
}

macro_rules! matrix_impl {
    ($t:ty, $perspective:expr, |$v:ident| $flat:expr) => {
        impl matrix_sealed::Sealed for $t {}
        impl TransformMatrix for $t {
            fn into_matrix(self) -> Matrix3 {
                let $v = self;
                let flat: Vec<f32> = $flat;
                let mut m = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
                for (k, v) in flat.iter().enumerate() {
                    m[k] = f64::from(*v);
                }
                Matrix3 { m, perspective: $perspective }
            }
        }
    };
}

matrix_impl!([f32; 6], false, |v| v.to_vec());
matrix_impl!([f32; 9], true, |v| v.to_vec());
matrix_impl!([[f32; 3]; 2], false, |v| v.concat());
matrix_impl!([[f32; 3]; 3], true, |v| v.concat());

impl Matrix3 {
    /// The inverse matrix, `None` if it is singular.
    fn inverse(&self) -> Option<Matrix3> {
        let m = &self.m;
        let cof = [
            m[4] * m[8] - m[5] * m[7],
            -(m[3] * m[8] - m[5] * m[6]),
            m[3] * m[7] - m[4] * m[6],
        ];
        let det = m[0] * cof[0] + m[1] * cof[1] + m[2] * cof[2];
        if det == 0.0 || !det.is_finite() {
            return None;
        }
        let inv = [
            cof[0],
            -(m[1] * m[8] - m[2] * m[7]),
            m[1] * m[5] - m[2] * m[4],
            cof[1],
            m[0] * m[8] - m[2] * m[6],
            -(m[0] * m[5] - m[2] * m[3]),
            cof[2],
            -(m[0] * m[7] - m[1] * m[6]),
            m[0] * m[4] - m[1] * m[3],
        ]
        .map(|v| v / det);
        Some(Matrix3 { m: inv, perspective: self.perspective })
    }

    /// The point `(x, y)` mapped, in `f64`.
    fn apply(&self, x: f64, y: f64) -> (f64, f64) {
        let m = &self.m;
        let w = m[6] * x + m[7] * y + m[8];
        ((m[0] * x + m[1] * y + m[2]) / w, (m[3] * x + m[4] * y + m[5]) / w)
    }
}

fn failure<T: crate::gpu_type::GpuType>(ctx: &Context, shape: Shape, error: LumaError, op: &'static str, location: &'static Location<'static>) -> Array<T> {
    let result: Outcome<Array<T>> = Err(Failure::New(error));
    Array::resolve(ctx, shape, result, op, location)
}

/// The pixel coordinates `(x, y)` along dimensions 0 and 1 of an output of `d0 x d1` pixels, as
/// `f32` arrays of that shape.
fn grid(ctx: &Context, d0: usize, d1: usize) -> Result<(Array<f32>, Array<f32>)> {
    let shape = Shape::from([d0, d1]);
    Ok((Array::<u32>::range_on(ctx, shape, 0)?.to_f32(), Array::<u32>::range_on(ctx, shape, 1)?.to_f32()))
}

/// The output extent of `n` scaled by `s`: truncated, as ArrayFire does.
fn scaled(n: usize, s: f32) -> usize {
    (n as f64 * f64::from(s)) as usize
}

impl Array<f32> {
    /// The image resized to `d0 x d1` pixels (dimensions 0 and 1; the others keep their extents),
    /// like ArrayFire's `resize`. Output pixel `(x, y)` samples the input at
    /// `(x * n0 / d0, y * n1 / d1)` (computed as `x / (d0 / n0)` in `f32`, as ArrayFire does), so the first pixel of each axis is the input's first and the
    /// grid is not centred; neighbours past the last input pixel repeat it. Use
    /// [`Interp::Nearest`], [`Interp::Lower`], [`Interp::Linear`] (bilinear), [`Interp::Cosine`],
    /// [`Interp::Cubic`] (bicubic) or [`Interp::CubicSpline`].
    ///
    /// Fails (poisoned array) if `d0` or `d1` is zero.
    ///
    /// ```
    /// use luma::{Array, Interp};
    /// let img = Array::from_slice(&[0.0f32, 1.0, 2.0, 3.0], [2, 2])?;
    /// let big = img.resize(4, 4, Interp::Linear);
    /// assert_eq!(big.shape().dims(), [4, 4, 1, 1]);
    /// assert_eq!(big.to_vec()?[1], 0.5);
    /// # Ok::<(), luma::LumaError>(())
    /// ```
    #[track_caller]
    pub fn resize(&self, d0: usize, d1: usize, method: Interp) -> Array<f32> {
        self.resize_at(d0, d1, method, Location::caller())
    }

    /// [`Array::resize`] to `scale0 * n0` by `scale1 * n1` pixels (each rounded down), like
    /// ArrayFire's `resize(scale0, scale1, in)`. Fails (poisoned array) unless the scales are
    /// positive and finite and the result is not empty.
    #[track_caller]
    pub fn resize_by(&self, scale0: f32, scale1: f32, method: Interp) -> Array<f32> {
        let location = Location::caller();
        if !(scale0 > 0.0 && scale1 > 0.0 && scale0.is_finite() && scale1.is_finite()) {
            let error = LumaError::InvalidShape(format!("resize_by: scales {scale0} and {scale1} must be positive and finite"));
            return failure(self.context(), self.shape(), error, "resize_by", location);
        }
        let dims = self.shape().dims();
        self.resize_at(scaled(dims[0], scale0), scaled(dims[1], scale1), method, location)
    }

    fn resize_at(&self, d0: usize, d1: usize, method: Interp, location: &'static Location<'static>) -> Array<f32> {
        let ctx = self.context();
        let dims = self.shape().dims();
        let out = Shape::from([d0, d1, dims[2], dims[3]]);
        if d0 == 0 || d1 == 0 {
            let error = LumaError::InvalidShape(format!("resize to {d0} x {d1} pixels: an image has at least one pixel"));
            return failure(ctx, out, error, "resize", location);
        }
        if self.shape().validate().is_err() || out.validate().is_err() {
            return failure(ctx, out, LumaError::DimensionTooLarge { shape: out.dims() }, "resize", location);
        }
        let axis = |d: usize, n: usize, to: usize| -> Result<Array<f32>> {
            let mut shape = [1; 4];
            shape[d] = to;
            let r = Array::<u32>::range_on(ctx, shape, d)?.to_f32();
            Ok((&r / (to as f32 / n as f32)).broadcast_to(out))
        };
        match (axis(0, dims[0], d0), axis(1, dims[1], d1)) {
            (Ok(x), Ok(y)) => sample(self, out, &[(0, &x), (1, &y)], method),
            (Err(e), _) | (_, Err(e)) => Array::poisoned(ctx, out, Arc::new(e)),
        }
    }

    /// The image warped by the map `m` from output to input pixel coordinates, into `d0 x d1`
    /// pixels: output pixel `(x, y)` takes the input at `m(x, y)` (`m` is `matrix` itself, or its
    /// inverse for `inverse == false`; see [`Array::transform`]). `d0 == 0 && d1 == 0` means the
    /// input's size. Pixels whose source lies outside the input are 0.
    fn warp(&self, m: Matrix3, d0: usize, d1: usize, method: Interp, op: &'static str, location: &'static Location<'static>) -> Array<f32> {
        let ctx = self.context();
        let dims = self.shape().dims();
        let (d0, d1) = if d0 == 0 && d1 == 0 { (dims[0], dims[1]) } else { (d0, d1) };
        let out = Shape::from([d0, d1, dims[2], dims[3]]);
        if d0 == 0 || d1 == 0 {
            let error = LumaError::InvalidShape(format!("{op}: output of {d0} x {d1} pixels: give both extents or neither"));
            return failure(ctx, out, error, op, location);
        }
        if m.m.iter().any(|v| !v.is_finite()) {
            let error = LumaError::InvalidShape(format!("{op}: the transformation matrix has a NaN or infinite entry"));
            return failure(ctx, out, error, op, location);
        }
        if out.validate().is_err() {
            return failure(ctx, out, LumaError::DimensionTooLarge { shape: out.dims() }, op, location);
        }
        let (x, y) = match grid(ctx, d0, d1) {
            Ok(g) => g,
            Err(e) => return failure(ctx, out, e, op, location),
        };
        let f = m.m.map(|v| v as f32);
        let affine = |row: usize| (&x * f[row * 3] + &y * f[row * 3 + 1]) + f[row * 3 + 2];
        let (mut xi, mut yi) = (affine(0), affine(1));
        if m.perspective {
            let w = (&x * f[6] + &y * f[7]) + f[8];
            xi = &xi / &w;
            yi = &yi / &w;
        }
        let (n0, n1) = (dims[0] as f32, dims[1] as f32);
        // Sources inside the input; for `Nearest` the pixel it rounds to must exist.
        let reach = if method == Interp::Nearest { 0.5 } else { 0.0 };
        let ok = xi.ge(0.0) & yi.ge(0.0) & (&xi + reach).lt(n0) & (&yi + reach).lt(n1);
        let (xi, yi) = (xi.broadcast_to(out), yi.broadcast_to(out));
        select(&ok.broadcast_to(out), sample(self, out, &[(0, &xi), (1, &yi)], method), 0.0)
    }

    /// The image transformed by `matrix` into `d0 x d1` pixels (both 0: the input's size), like
    /// ArrayFire's `transform`. `matrix` is an affine (2 x 3) or projective (3 x 3) map, see
    /// [`TransformMatrix`]. With `inverse == true` it maps output pixel coordinates to input
    /// coordinates: output pixel `(x, y)` is the input interpolated at `matrix * (x, y, 1)`;
    /// with `inverse == false` it maps input to output coordinates, and the output samples the
    /// matrix's inverse (so `[1, 0, 2, 0, 1, 1]` moves the content by `+2` along dimension 0 and
    /// `+1` along dimension 1). Output pixels whose source is outside the input are 0: a source
    /// coordinate is inside for `0 <= x < n`, except for [`Interp::Nearest`], where the pixel it
    /// rounds to must exist (`x < n - 0.5`). Neighbours past the last pixel repeat it. All
    /// channels and batches use the same matrix.
    ///
    /// Fails (poisoned array) if the matrix has a NaN or infinite entry, or is singular and
    /// `inverse == false`, or exactly one of `d0` and `d1` is 0.
    ///
    /// ```
    /// use luma::{Array, Interp};
    /// let img = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], [3, 3])?;
    /// // Content moves one pixel along dimension 0.
    /// let moved = img.transform([1.0, 0.0, 1.0, 0.0, 1.0, 0.0], 3, 3, Interp::Nearest, false);
    /// assert_eq!(moved.to_vec()?, vec![0.0, 1.0, 2.0, 0.0, 4.0, 5.0, 0.0, 7.0, 8.0]);
    /// # Ok::<(), luma::LumaError>(())
    /// ```
    #[track_caller]
    pub fn transform(&self, matrix: impl TransformMatrix, d0: usize, d1: usize, method: Interp, inverse: bool) -> Array<f32> {
        let location = Location::caller();
        let m = matrix.into_matrix();
        self.transform_at(m, d0, d1, method, inverse, "transform", location)
    }

    #[allow(clippy::too_many_arguments)]
    fn transform_at(&self, m: Matrix3, d0: usize, d1: usize, method: Interp, inverse: bool, op: &'static str, location: &'static Location<'static>) -> Array<f32> {
        let m = if inverse {
            m
        } else {
            match m.inverse() {
                Some(m) => m,
                None => {
                    let error = LumaError::InvalidShape(format!("{op}: the transformation matrix is singular"));
                    return failure(self.context(), self.shape(), error, op, location);
                }
            }
        };
        self.warp(m, d0, d1, method, op, location)
    }

    /// The image rotated by `theta` radians about its centre, like ArrayFire's `rotate`: output
    /// pixel `(x, y)` takes the input at `R (x - cx', y - cy') + (cx, cy)` with `R` the rotation
    /// by `theta` (`cos` / `sin` of `theta` in `[c s; -s c]` order), `(cx, cy)` the input's centre
    /// `((n0 - 1) / 2, (n1 - 1) / 2)` and `(cx', cy')` the output's. With `crop` the output has
    /// the input's size; without, the bounding box of the rotated image (extents truncated).
    /// Pixels that come from outside the input are 0.
    #[track_caller]
    pub fn rotate(&self, theta: f32, crop: bool, method: Interp) -> Array<f32> {
        let location = Location::caller();
        let dims = self.shape().dims();
        let (c, s) = ((-theta).cos(), (-theta).sin());
        let (n0, n1) = (dims[0] as f32, dims[1] as f32);
        let (o0, o1) = if crop { (n0, n1) } else { ((n0 * c.abs() + n1 * s.abs()).trunc(), (n1 * c.abs() + n0 * s.abs()).trunc()) };
        let (nx, ny) = (0.5 * (n0 - 1.0), 0.5 * (n1 - 1.0));
        let (mx, my) = (0.5 * (o0 - 1.0), 0.5 * (o1 - 1.0));
        let (sx, sy) = (mx * c + my * -s, mx * s + my * c);
        let (tx, ty) = (-(sx - nx), -(sy - ny));
        let m = [c, -s, tx, s, c, ty].into_matrix();
        self.warp(m, o0 as usize, o1 as usize, method, "rotate", location)
    }

    /// The image translated, like ArrayFire's `translate`: output pixel `(x, y)` takes the input
    /// at `(x + t0, y + t1)` (so the content moves by `-t0` along dimension 0 and `-t1` along
    /// dimension 1), into `d0 x d1` pixels (both 0: the input's size). Pixels that come from
    /// outside the input are 0.
    #[track_caller]
    pub fn translate(&self, t0: f32, t1: f32, d0: usize, d1: usize, method: Interp) -> Array<f32> {
        self.warp([1.0, 0.0, t0, 0.0, 1.0, t1].into_matrix(), d0, d1, method, "translate", Location::caller())
    }

    /// The image scaled by `s0` along dimension 0 and `s1` along dimension 1, like ArrayFire's
    /// `scale`: output pixel `(x, y)` takes the input at `(x / s0, y / s1)`, into `d0 x d1` pixels
    /// (both 0: the input's extents times the scales, truncated). Pixels that come from outside
    /// the input are 0. Fails (poisoned array) unless the scales are finite and not 0.
    #[track_caller]
    pub fn scale(&self, s0: f32, s1: f32, d0: usize, d1: usize, method: Interp) -> Array<f32> {
        let location = Location::caller();
        if s0 == 0.0 || s1 == 0.0 || !s0.is_finite() || !s1.is_finite() {
            let error = LumaError::InvalidShape(format!("scale: factors {s0} and {s1} must be finite and not 0"));
            return failure(self.context(), self.shape(), error, "scale", location);
        }
        let dims = self.shape().dims();
        let (d0, d1) = if d0 == 0 && d1 == 0 { (scaled(dims[0], s0.abs()), scaled(dims[1], s1.abs())) } else { (d0, d1) };
        self.warp([1.0 / s0, 0.0, 0.0, 0.0, 1.0 / s1, 0.0].into_matrix(), d0, d1, method, "scale", location)
    }

    /// The image skewed by the angles `skew0` and `skew1` (radians), like ArrayFire's `skew`.
    /// With `ta = tan(skew0)` and `tb = tan(skew1)`, output pixel `(x, y)` takes the input at
    /// `(x + tb y, ta x + y)` if `inverse == false`, and at `(x + tb y, ta x + y) / (1 - ta tb)`
    /// if `inverse == true`, into `d0 x d1` pixels (both 0: the input's size). Pixels that come
    /// from outside the input are 0.
    #[track_caller]
    pub fn skew(&self, skew0: f32, skew1: f32, d0: usize, d1: usize, inverse: bool, method: Interp) -> Array<f32> {
        let (ta, tb) = (skew0.tan(), skew1.tan());
        let c = if inverse { 1.0 / (1.0 - ta * tb) } else { 1.0 };
        self.warp([c, c * tb, 0.0, c * ta, c, 0.0].into_matrix(), d0, d1, method, "skew", Location::caller())
    }
}

/// Where the corners of an image go under `matrix`, like ArrayFire's `transformCoordinates`: a
/// `[4, 2]` array whose rows are the images of the points `(0, 0)`, `(0, d0)`, `(d1, d0)` and
/// `(d1, 0)` (column 0: the `x` results, column 1: the `y` results), with the 3 x 3 (or 2 x 3)
/// `matrix` applied to `(x, y, 1)` and the projective division where it has a third row. Runs on
/// the host and uploads the eight values to the default context.
///
/// ```
/// let corners = luma::transform_coordinates([1.0f32, 0.0, 5.0, 0.0, 1.0, 7.0], 4.0, 3.0)?;
/// assert_eq!(corners.to_vec()?, vec![5.0, 5.0, 8.0, 8.0, 7.0, 11.0, 11.0, 7.0]);
/// # Ok::<(), luma::LumaError>(())
/// ```
pub fn transform_coordinates(matrix: impl TransformMatrix, d0: f32, d1: f32) -> Result<Array<f32>> {
    transform_coordinates_on(&default_context()?, matrix, d0, d1)
}

/// [`transform_coordinates`] on an explicit context.
pub fn transform_coordinates_on(ctx: &Context, matrix: impl TransformMatrix, d0: f32, d1: f32) -> Result<Array<f32>> {
    let m = matrix.into_matrix();
    let corners = [(0.0, 0.0), (0.0, f64::from(d0)), (f64::from(d1), f64::from(d0)), (f64::from(d1), 0.0)];
    let mut out = [0.0f32; 8];
    for (k, (x, y)) in corners.into_iter().enumerate() {
        let (u, v) = m.apply(x, y);
        out[k] = u as f32;
        out[4 + k] = v as f32;
    }
    Array::from_slice_on(ctx, &out, [4, 2])
}
