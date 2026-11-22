//! Structural operations of ArrayFire's array API: [`Array::tile`], diagonals ([`Array::diag`],
//! [`Array::diag_matrix`]), [`Array::identity`], triangles ([`Array::lower`], [`Array::upper`]),
//! [`Array::replace`] and differences ([`Array::diff1`], [`Array::diff2`]).
//!
//! None of them is a kernel of its own: each is a lazy index map ([`Array::tile`], the diagonals,
//! the differences read the array through a view) or an elementwise expression of coordinate
//! generators and a select ([`Array::identity`], the triangles), so each fuses into the
//! expression that uses it and moves no data until something reads it.
//!
//! **Dimension order.** A luma [`Shape`] lists the extents with the first dimension varying
//! fastest, exactly like ArrayFire's `af::dim4` (column-major): dimension 0 of luma is dimension
//! 0 of ArrayFire, so `af::tile(in, x, y, z, w)` is `a.tile([x, y, z, w])` and element `(i, j)`
//! (`i` along dimension 0, the "row" of a matrix) is where the diagonals and triangles below are
//! counted from: the diagonal `k` holds the elements with `j - i = k`.

use std::panic::Location;

use crate::array::{Array, Failure};
use crate::builtin::BinOp;
use crate::context::Context;
use crate::error::{LumaError, Result};
use crate::gpu_type::{Numeric, ScalarType};
use crate::generate::constant;
use crate::jit::{Edge, ViewMap};
use crate::ops::{run_select, Cond, Condition, Operand, Outcome, Rhs};
use crate::shape::Shape;
use crate::view::dims32;

/// `select(mask, a, b)` as the lazy elementwise operation, `a` and `b` of the shape of `mask`.
fn pick<T: ScalarType>(mask: &Array<bool>, a: Rhs<'_, T>, b: Rhs<'_, T>, location: &'static Location<'static>) -> Outcome<Array<T>> {
    run_select(Cond::Bool(mask), a, b, location)
}

impl<T: ScalarType> Array<T> {
    /// The array repeated `reps[k]` times along dimension `k` (a missing entry repeats once),
    /// like `af::tile(in, x, y, z, w)`: the result has extent `shape[k] * reps[k]` and element `x`
    /// along dimension `k` is element `x % shape[k]` of this array.
    ///
    /// ```
    /// # use luma::Array;
    /// # fn main() -> luma::Result<()> {
    /// let a = Array::from_slice(&[1.0f32, 2.0, 3.0], 3)?;
    /// assert_eq!(a.tile([2, 2]).to_vec()?, [1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
    /// # Ok(()) }
    /// ```
    ///
    /// Lazy like [`Array::pad`] with [`PadMode::Wrap`](crate::PadMode::Wrap): the result is a
    /// read of this array through an index map that fuses into the expression using it. Works
    /// for every element type.
    ///
    /// Fails (poisoned array) if a repetition is zero or the tiled shape does not fit `u32`.
    #[track_caller]
    pub fn tile(&self, reps: impl Into<Shape>) -> Array<T> {
        self.tile_at(reps.into(), Location::caller())
    }

    /// [`Array::tile`] reporting failures at `location`.
    pub(crate) fn tile_at(&self, reps: Shape, location: &'static Location<'static>) -> Array<T> {
        let (dims, reps_dims) = (self.shape().dims(), reps.dims());
        let out = Shape::new(std::array::from_fn(|k| dims[k].saturating_mul(reps_dims[k])));
        let result = self.run_tile(reps, out, location);
        Array::resolve(self.context(), out, result, "tile", location)
    }

    fn run_tile(&self, reps: Shape, out: Shape, location: &'static Location<'static>) -> Outcome<Array<T>> {
        self.src()?;
        reps.validate()?;
        let mut map = ViewMap::identity(dims32(self.shape())?);
        map.out = dims32(out)?;
        self.run_view(map, Edge::Wrap, None, out, "tile", location)
    }

    /// The elements of the diagonal `k` of every matrix (dimensions 0 and 1; dimensions 2 and 3
    /// are batches) as a column, like `af::diag(in, k, true)`: for `k >= 0` element `i` of the
    /// result is element `(i, i + k)`, for `k < 0` it is `(i - k, i)`. The result has shape
    /// `[len, 1, shape[2], shape[3]]` with `len = min(shape[0], shape[1] - k)` for `k >= 0` and
    /// `min(shape[0] + k, shape[1])` for `k < 0`.
    ///
    /// A lazy read through an index map, like a slice, so a diagonal of an expression is
    /// computed at the diagonal elements only. Works for every element type.
    ///
    /// Fails (poisoned array, [`LumaError::InvalidShape`]) if the diagonal lies outside the
    /// matrix (it would be empty: luma has no empty arrays).
    #[track_caller]
    pub fn diag(&self, k: isize) -> Array<T> {
        self.diag_at(k, Location::caller())
    }

    /// [`Array::diag`] reporting failures at `location`.
    pub(crate) fn diag_at(&self, k: isize, location: &'static Location<'static>) -> Array<T> {
        let [m, n, b2, b3] = self.shape().dims();
        let len = diag_len(m, n, k).unwrap_or(1);
        let out = Shape::new([len, 1, b2, b3]);
        let result = self.run_diag(k, out, location);
        Array::resolve(self.context(), out, result, "diag", location)
    }

    fn run_diag(&self, k: isize, out: Shape, location: &'static Location<'static>) -> Outcome<Array<T>> {
        self.src()?;
        let [m, n, ..] = self.shape().dims();
        if diag_len(m, n, k).is_none() {
            return Err(LumaError::InvalidShape(format!("diagonal {k} lies outside a matrix of {m} x {n}")).into());
        }
        let too_large = || LumaError::DimensionTooLarge { shape: self.shape().dims() };
        let offset = i32::try_from(k.unsigned_abs()).map_err(|_| too_large())?;
        let mut map = ViewMap::identity(dims32(self.shape())?);
        map.out = dims32(out)?;
        // Both coordinates of the source follow output axis 0; the unused output axis 1 is 1.
        map.axes[0].start = if k < 0 { offset } else { 0 };
        map.axes[1].start = if k > 0 { offset } else { 0 };
        map.axes[1].from = 0;
        self.run_view(map, Edge::None, None, out, "diag", location)
    }

    /// The matrices that have the columns of this array on their diagonal `k`, like
    /// `af::diag(in, k, false)`: this array is a column `[n, 1, b2, b3]` and the result has shape
    /// `[n + |k|, n + |k|, b2, b3]` with element `i` of the column at `(i, i + k)` for `k >= 0`
    /// and at `(i - k, i)` for `k < 0`, and zeros (`false` for `bool`) elsewhere. Fused: the
    /// zeros are not read from anywhere.
    ///
    /// ```
    /// # use luma::Array;
    /// # fn main() -> luma::Result<()> {
    /// let v = Array::from_slice(&[1.0f32, 2.0], 2)?;
    /// assert_eq!(v.diag_matrix(0).to_vec()?, [1.0, 0.0, 0.0, 2.0]);
    /// assert_eq!(v.diag_matrix(1).to_vec()?, [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0, 0.0]);
    /// # Ok(()) }
    /// ```
    ///
    /// Fails (poisoned array, [`LumaError::InvalidShape`]) if this array has more than one
    /// column.
    #[track_caller]
    pub fn diag_matrix(&self, k: isize) -> Array<T> {
        self.diag_matrix_at(k, Location::caller())
    }

    /// [`Array::diag_matrix`] reporting failures at `location`.
    pub(crate) fn diag_matrix_at(&self, k: isize, location: &'static Location<'static>) -> Array<T> {
        let [n, _, b2, b3] = self.shape().dims();
        let side = n.saturating_add(k.unsigned_abs());
        let out = Shape::new([side, side, b2, b3]);
        let result = self.run_diag_matrix(k, out, location);
        Array::resolve(self.context(), out, result, "diag_matrix", location)
    }

    fn run_diag_matrix(&self, k: isize, out: Shape, location: &'static Location<'static>) -> Outcome<Array<T>> {
        self.src()?;
        if self.shape().dims()[1] != 1 {
            return Err(LumaError::InvalidShape(format!("diag_matrix of {}: the array must be a column", self.shape())).into());
        }
        let ctx = self.context();
        out.validate()?;
        let offset = u32::try_from(k.unsigned_abs()).map_err(|_| LumaError::DimensionTooLarge { shape: out.dims() })?;
        let rows = Array::<u32>::range_on(ctx, out, 0).map_err(Failure::New)?;
        let cols = Array::<u32>::range_on(ctx, out, 1).map_err(Failure::New)?;
        // Where the diagonal is: column = row + k, i.e. `cols == rows + k` or `rows == cols + |k|`.
        let on = if k >= 0 { cols.eq(&rows + offset) } else { rows.eq(&cols + offset) };
        // The element on the diagonal is `v[min(row, col)]`: row for k >= 0, row + k = col else.
        let mut map = ViewMap::identity(dims32(self.shape())?);
        map.out = dims32(out)?;
        map.axes[0].start = if k < 0 { -i32::try_from(offset).map_err(|_| LumaError::DimensionTooLarge { shape: out.dims() })? } else { 0 };
        map.axes[1].step = 0;
        let diagonal = self.run_view(map, Edge::Clamp, None, out, "diag_matrix", location)?;
        pick(&on, Rhs::Array(&diagonal), Rhs::Scalar(T::zeroed()), location)
    }

    /// The matrices of this array (dimensions 0 and 1; dimensions 2 and 3 are batches) with the
    /// elements above the diagonal set to zero (`false` for `bool`), like `af::lower`. With
    /// `unit_diagonal` the diagonal itself is one. Works on non-square matrices: element `(i, j)`
    /// is kept where `j <= i`.
    ///
    /// Fused elementwise: the triangle of an expression is computed where it is kept.
    #[track_caller]
    pub fn lower(&self, unit_diagonal: bool) -> Array<T> {
        self.triangle(true, unit_diagonal, "lower", Location::caller())
    }

    /// The matrices of this array with the elements below the diagonal set to zero, like
    /// `af::upper`: element `(i, j)` is kept where `j >= i`. See [`Array::lower`].
    #[track_caller]
    pub fn upper(&self, unit_diagonal: bool) -> Array<T> {
        self.triangle(false, unit_diagonal, "upper", Location::caller())
    }

    fn triangle(&self, lower: bool, unit_diagonal: bool, name: &'static str, location: &'static Location<'static>) -> Array<T> {
        let result = self.run_triangle(lower, unit_diagonal, location);
        Array::resolve(self.context(), self.shape(), result, name, location)
    }

    fn run_triangle(&self, lower: bool, unit_diagonal: bool, location: &'static Location<'static>) -> Outcome<Array<T>> {
        self.src()?;
        let (ctx, shape) = (self.context(), self.shape());
        let rows = Array::<u32>::range_on(ctx, shape, 0).map_err(Failure::New)?;
        let cols = Array::<u32>::range_on(ctx, shape, 1).map_err(Failure::New)?;
        let outside = if lower { cols.gt(&rows) } else { cols.lt(&rows) };
        let zeroed = pick(&outside, Rhs::Scalar(T::zeroed()), Rhs::Array(self), location)?;
        if !unit_diagonal {
            return Ok(zeroed);
        }
        let on = cols.eq(&rows);
        pick(&on, Rhs::Scalar(constant::<T>(1.0)?), Rhs::Array(&zeroed), location)
    }

    /// This array where `cond` is true and `b` elsewhere, like `af::replace(a, cond, b)` but
    /// returning a new array (this one is left alone): `b` is an array of this shape (or with
    /// extent 1 along the axes where it differs, read at every position) or a single value
    /// (`f64` arrays take arrays only, like [`select`](crate::select)). Fused elementwise.
    ///
    /// `cond` must have this array's shape; a mismatch poisons the result with
    /// [`LumaError::ShapeMismatch`].
    #[track_caller]
    pub fn replace(&self, cond: &impl Condition, b: impl Operand<T>) -> Array<T> {
        let location = Location::caller();
        let result = self.run_replace(cond.condition(), b.rhs(), location);
        Array::resolve(self.context(), self.shape(), result, "replace", location)
    }

    fn run_replace(&self, cond: Cond<'_>, b: Rhs<'_, T>, location: &'static Location<'static>) -> Outcome<Array<T>> {
        self.src()?;
        if !self.context().same_as(cond.context()) {
            return Err(LumaError::ContextMismatch.into());
        }
        if cond.shape() != self.shape() {
            return Err(LumaError::ShapeMismatch { lhs: self.shape().dims(), rhs: cond.shape().dims() }.into());
        }
        run_select(cond, Rhs::Array(self), b, location)
    }
}

/// The length of the diagonal `k` of an `m x n` matrix, `None` if it is empty.
fn diag_len(m: usize, n: usize, k: isize) -> Option<usize> {
    let shift = k.unsigned_abs();
    let len = if k >= 0 { m.min(n.checked_sub(shift)?) } else { m.checked_sub(shift)?.min(n) };
    (len > 0).then_some(len)
}

impl<T: ScalarType> Array<T> {
    /// The identity matrix of dimensions 0 and 1 of `shape` (ones where the two coordinates are
    /// equal, zeros elsewhere), repeated over dimensions 2 and 3, like `af::identity`, on the
    /// default context. Not necessarily square. It is generated inside the kernels that read it
    /// and occupies no memory unless evaluated on its own (see [`Array::range`]).
    #[track_caller]
    pub fn identity(shape: impl Into<Shape>) -> Result<Self> {
        Self::identity_on(&crate::default_context()?, shape)
    }

    /// [`Array::identity`] on an explicit context.
    #[track_caller]
    pub fn identity_on(ctx: &Context, shape: impl Into<Shape>) -> Result<Self> {
        let location = Location::caller();
        let shape = shape.into();
        shape.validate()?;
        let rows = Array::<u32>::range_on(ctx, shape, 0)?;
        let cols = Array::<u32>::range_on(ctx, shape, 1)?;
        let on = cols.eq(&rows);
        pick(&on, Rhs::Scalar(constant::<T>(1.0)?), Rhs::Scalar(T::zeroed()), location).map_err(Failure::into_error)
    }
}

impl<T: Numeric> Array<T> {
    /// The first difference along `axis`, like `af::diff1(in, dim)`: element `i` of the result is
    /// `self[i + 1] - self[i]` along that axis, which is one shorter. Integers wrap like `-`.
    ///
    /// A fused elementwise difference of two lazy slices of this array: it reads the array once.
    ///
    /// Fails (poisoned array, [`LumaError::InvalidShape`]) if `axis >= 4` or the extent along it
    /// is below 2 (luma has no empty arrays).
    #[track_caller]
    pub fn diff1(&self, axis: usize) -> Array<T> {
        self.diff_at(axis, 1, Location::caller())
    }

    /// The second difference along `axis`, like `af::diff2(in, dim)`: element `i` of the result
    /// is `(self[i + 2] - self[i + 1]) - (self[i + 1] - self[i])`, which is two shorter; the same
    /// as `diff1(axis).diff1(axis)`, in one expression. See [`Array::diff1`].
    #[track_caller]
    pub fn diff2(&self, axis: usize) -> Array<T> {
        self.diff_at(axis, 2, Location::caller())
    }

    fn diff_at(&self, axis: usize, order: usize, location: &'static Location<'static>) -> Array<T> {
        let name = if order == 1 { "diff1" } else { "diff2" };
        let mut dims = self.shape().dims();
        let result = self.run_diff(axis, order, location);
        if let Some(extent) = dims.get_mut(axis) {
            *extent = extent.saturating_sub(order).max(1);
        }
        Array::resolve(self.context(), Shape::new(dims), result, name, location)
    }

    fn run_diff(&self, axis: usize, order: usize, location: &'static Location<'static>) -> Outcome<Array<T>> {
        self.src()?;
        let name = if order == 1 { "diff1" } else { "diff2" };
        let Some(&extent) = self.shape().dims().get(axis) else {
            return Err(LumaError::InvalidShape(format!("{name} along axis {axis} of a 4-dimensional array")).into());
        };
        if extent <= order {
            return Err(LumaError::InvalidShape(format!("{name} along axis {axis} of extent {extent} would be empty")).into());
        }
        let part = |from: usize| self.slice_at(axis, from..extent - order + from, false, location);
        let sub = |a: &Array<T>, b: &Array<T>| a.binary_at(Rhs::Array(b), BinOp::Sub, false, name, location);
        let (x0, x1) = (part(0), part(1));
        let first = sub(&x1, &x0);
        if order == 1 {
            return Ok(first);
        }
        let second = sub(&part(2), &x1);
        Ok(sub(&second, &first))
    }
}
