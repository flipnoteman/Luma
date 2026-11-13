//! Views (slices, padding, flips, transposes, non-circular shifts, gathers) and the writes
//! through them. A view is a lazy read of another array through an index map (see
//! [`crate::jit`]'s `ViewMap`): it moves no data, and fuses into the expression that uses it
//! like a shift does. [`Array::slice`], [`Array::flip`] and [`Array::transpose`] are read-only;
//! [`Array::slice_mut`], [`Array::flip_mut`] and [`Array::transpose_mut`] give a [`ViewMut`]
//! that writes into the array it was taken from ([`ViewMut::assign`], `+=`), and
//! [`Array::put`] scatters.

use std::ops::{
    AddAssign, BitAndAssign, BitOrAssign, BitXorAssign, Bound, DivAssign, MulAssign, RangeBounds, RemAssign, ShlAssign,
    ShrAssign, SubAssign,
};
use std::panic::Location;
use std::sync::Arc;

use crate::array::{Array, Failure, Storage};
use crate::builtin::{BinOp, CmpOp, UnOp};
use crate::context::Context;
use crate::error::{LumaError, Result};
use crate::fields::{FieldOp, Fields};
use crate::gpu::pager::Pin;
use crate::gpu_type::{Df32, GpuType, ScalarType};
use crate::jit::assign::Region;
use crate::jit::{self, AxisMap, Edge, ElemTy, Held, Op, OpKind, Scalar, ShiftKind, Src, ViewMap};
use crate::ops::{run_select, Operand, Outcome, Rhs};
use crate::shape::{Offsets, Shape};

/// What [`Array::pad`] puts in the cells outside the array.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PadMode {
    /// The nearest element of the array.
    Edge,
    /// The array mirrored about its boundary face: `-1` reads `0`, `-2` reads `1`, `n` reads
    /// `n - 1` (the ghost cells of a finite-volume wall).
    Mirror,
    /// The array repeated periodically.
    Wrap,
    /// Zeros.
    Zero,
}

/// An array that is an unevaluated-or-evaluated injective view of another array's data.
#[derive(Debug)]
pub(crate) struct ViewOf {
    /// The array the view reads: a buffer or a pending expression, never a scalar.
    pub(crate) base: Src,
    /// The claim on `base` when it is a node: a write through the view must not reach another
    /// array that the hash-consing table would otherwise hand the node to.
    pub(crate) held: Option<Held>,
    /// How the view reads `base`.
    pub(crate) map: ViewMap,
}

/// The dimensions of `shape` as `u32`.
fn dims32(shape: Shape) -> Result<[u32; 4], LumaError> {
    let dims = shape.dims();
    let mut out = [1u32; 4];
    for (slot, &dim) in out.iter_mut().zip(&dims) {
        *slot = u32::try_from(dim).map_err(|_| LumaError::DimensionTooLarge { shape: dims })?;
    }
    Ok(out)
}

impl<T: ScalarType> Array<T> {
    /// The view of this array through `map` as an array of `shape`: element `o` of the result is
    /// the element of `self` that `map` gives, and `edge` says what a coordinate outside `self`
    /// reads (`fill` is the value of [`Edge::Fill`]).
    pub(crate) fn run_view(
        &self,
        map: ViewMap,
        edge: Edge,
        fill: Option<Scalar>,
        shape: Shape,
        name: &'static str,
        location: &'static Location<'static>,
    ) -> Outcome<Array<T>> {
        self.run_view_with(map, edge, fill, shape, false, name, location)
    }

    /// [`Array::run_view`], the result writing through to what it reads if `writable` (a view
    /// for [`ViewMut`]) and rejecting writes otherwise.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn run_view_with(
        &self,
        map: ViewMap,
        edge: Edge,
        fill: Option<Scalar>,
        shape: Shape,
        writable: bool,
        name: &'static str,
        location: &'static Location<'static>,
    ) -> Outcome<Array<T>> {
        self.src()?;
        if writable && self.is_read_view() {
            return Err(LumaError::InvalidShape(
                "cannot take a mutable view of a read-only view; use slice_mut, flip_mut or transpose_mut on the array".into(),
            )
            .into());
        }
        shape.validate()?;
        if !map.fits() {
            return Err(LumaError::DimensionTooLarge { shape: shape.dims() }.into());
        }
        if edge == Edge::None && !map.in_range() {
            return Err(LumaError::InvalidShape(format!("view of {} outside of the array", self.shape())).into());
        }

        // What the node reads, and through which map. A view of a view that was not evaluated
        // reads what that view's node reads, through the composed map. (Not what `view.base`
        // holds: the node may read a snapshot of it, a view evaluated before a write into its
        // base, and an assignment through the view may have replaced its expression.)
        let inner = match (self.is_read_view() || self.view().is_some(), edge, self.pending()) {
            (true, Edge::None, Some(node)) => node.operands().and_then(|args| match node.op().kind {
                OpKind::Shift(ShiftKind::View { map: inner, edge: Edge::None }) => Some((args[0].clone(), inner)),
                _ => None,
            }),
            _ => None,
        };
        let (operand, read_map) = match (inner, self.pending()) {
            (Some((read, inner)), _) => {
                let base = match read {
                    Src::Node(node) if node.is_pending() && jit::recompute_viewed(&node) => Src::Node(node),
                    Src::Node(node) => {
                        node.context().jit().evaluate_alongside(Arc::clone(&node));
                        match node.force() {
                            Ok(storage) => Src::Buffer(Arc::clone(storage)),
                            Err(error) => return Err(Failure::Inherited(Arc::clone(error))),
                        }
                    }
                    other => other,
                };
                (base, inner.then(&map))
            }
            (None, Some(node)) if jit::recompute_viewed(node) => (Src::Node(Arc::clone(node)), map),
            _ => (Src::Buffer(Arc::clone(self.storage_alongside()?)), map),
        };

        let fill = fill.map_or_else(Src::unused, Src::Scalar);
        let args = [operand.clone(), fill, Src::unused()];
        let op = Op::new(OpKind::Shift(ShiftKind::View { map: read_map, edge }), ElemTy::of::<T>()?);
        let node = jit::create(self.context(), shape.len(), op, args, name, location)?;
        let mut result = Array::from_node(self.context(), shape, node);

        // What an assignment through the result writes to. A read-only view keeps no claim on
        // the node it reads: a temporary slice must not make it look shared.
        if edge == Edge::None {
            if writable {
                let (base, held, full) = match self.view() {
                    Some(view) => (view.base.clone(), view.held.as_ref().map(Held::share), view.map.then(&map)),
                    None => {
                        let held = match &operand {
                            Src::Node(node) => Some(Held::new(Arc::clone(node))),
                            _ => None,
                        };
                        (operand, held, map)
                    }
                };
                if full.is_injective() {
                    result = result.with_view(ViewOf { base, held, map: full });
                }
            } else if read_map.is_injective() {
                result = result.into_read_view();
            }
        }
        Ok(result)
    }

    /// This array read at every coordinate of an array of `shape` as an operand: along every
    /// axis where this array has extent 1 and `shape` more, the one element is repeated, the
    /// other axes are read as they are. Every extent of `shape` must be this array's or this
    /// array's must be 1.
    pub(crate) fn broadcast_axis(&self, shape: Shape, location: &'static Location<'static>) -> Outcome<Src> {
        self.broadcast_view(shape, location)?.src()
    }

    fn broadcast_view(&self, shape: Shape, location: &'static Location<'static>) -> Outcome<Array<T>> {
        let src = dims32(self.shape())?;
        let out = dims32(shape)?;
        let axes = std::array::from_fn(|k| AxisMap { from: k as u8, start: 0, step: if src[k] == 1 && out[k] > 1 { 0 } else { 1 } });
        let map = ViewMap { src, out, axes };
        self.run_view(map, Edge::None, None, shape, "broadcast", location)
    }

    /// This array as an operand of an operation working at `target`: its own source if the
    /// shapes agree, else read through [`Array::broadcast_axis`]; fails with `ShapeMismatch`
    /// (`lhs` the target) unless every extent is the target's or 1.
    pub(crate) fn broadcast_operand(&self, target: Shape, location: &'static Location<'static>) -> Outcome<Src> {
        let operand = self.src()?;
        if self.shape() == target {
            return Ok(operand);
        }
        let (mine, theirs) = (self.shape().dims(), target.dims());
        if (0..4).any(|k| mine[k] != theirs[k] && mine[k] != 1) {
            return Err(LumaError::ShapeMismatch { lhs: theirs, rhs: mine }.into());
        }
        self.broadcast_axis(target, location)
    }

    /// This array repeated along its extent-1 axes to `shape`: along every axis the extent
    /// is `shape`'s or 1. A lazy read like a slice that fuses into the expression using it, so
    /// `col.broadcast_to([n, m]) * &row.broadcast_to([n, m])` is an outer product in one kernel
    /// (operators and [`select`](crate::select) broadcast the same way by themselves).
    ///
    /// Fails (poisoned array, [`LumaError::ShapeMismatch`]) if an extent differs from `shape`'s
    /// and is not 1. The result is an ordinary array: it is not a view that writes through.
    #[track_caller]
    pub fn broadcast_to(&self, shape: impl Into<Shape>) -> Array<T> {
        self.broadcast_to_at(shape.into(), Location::caller())
    }

    /// [`Array::broadcast_to`] reporting failures at `location`.
    pub(crate) fn broadcast_to_at(&self, shape: Shape, location: &'static Location<'static>) -> Array<T> {
        let result = self.src().and_then(|_| {
            shape.validate()?;
            let (mine, theirs) = (self.shape().dims(), shape.dims());
            if (0..4).any(|k| mine[k] != theirs[k] && mine[k] != 1) {
                return Err(LumaError::ShapeMismatch { lhs: mine, rhs: theirs }.into());
            }
            self.broadcast_view(shape, location).map(Array::without_view)
        });
        Array::resolve(self.context(), shape, result, "broadcast_to", location)
    }

    /// The part `range` of the extent along `axis` (0 is the fastest dimension), as an array of
    /// that smaller shape: element `(x, y)` of `a.slice(0, 1..4)` is `a[(x + 1, y)]`.
    ///
    /// A slice moves no data: it is a lazy read that fuses into the expression using it (the
    /// difference of two slices of one array is one kernel reading the array once). It is
    /// read-only: [`Array::assign`], [`Array::put`], [`Array::write`] and `+=` on it fail with
    /// [`LumaError::InvalidShape`]; [`Array::slice_mut`] is the slice that writes through.
    ///
    /// Fails (poisoned array, [`LumaError::InvalidShape`]) if the range is empty or exceeds the
    /// extent, or `axis >= 4`.
    #[track_caller]
    pub fn slice(&self, axis: usize, range: impl RangeBounds<usize>) -> Array<T> {
        self.slice_at(axis, range, false, Location::caller())
    }

    /// [`Array::slice`] that writes through: the part `range` along `axis` of this array as a
    /// [`ViewMut`], which [`ViewMut::assign`] and the compound operators (`+=`, ...) write into
    /// this array. `w.slice_mut(1, ..1).assign(0.0)` zeros the first row of `w` for every array
    /// sharing its buffer. An invalid range, or a read-only array, poisons the view: its writes
    /// fail.
    #[track_caller]
    pub fn slice_mut(&mut self, axis: usize, range: impl RangeBounds<usize>) -> ViewMut<'_, Array<T>> {
        let inner = self.slice_at(axis, range, true, Location::caller());
        ViewMut { inner, owner: self }
    }

    /// [`Array::slice`] (or the view of [`Array::slice_mut`] if `writable`) reporting failures
    /// at `location`.
    pub(crate) fn slice_at(
        &self,
        axis: usize,
        range: impl RangeBounds<usize>,
        writable: bool,
        location: &'static Location<'static>,
    ) -> Array<T> {
        let (start, end) = (range.start_bound().cloned(), range.end_bound().cloned());
        let result = self.run_slice(axis, start, end, writable, location);
        Array::resolve(self.context(), self.slice_shape(axis, start, end), result, "slice", location)
    }

    /// The `start..end` of a slice along `axis`, or `None` if it is out of range.
    fn slice_bounds(&self, axis: usize, start: Bound<usize>, end: Bound<usize>) -> Option<(usize, usize)> {
        let extent = *self.shape().dims().get(axis)?;
        let start = match start {
            Bound::Included(s) => s,
            Bound::Excluded(s) => s.checked_add(1)?,
            Bound::Unbounded => 0,
        };
        let end = match end {
            Bound::Included(e) => e.checked_add(1)?,
            Bound::Excluded(e) => e,
            Bound::Unbounded => extent,
        };
        (start < end && end <= extent).then_some((start, end))
    }

    /// The shape of a slice for the error value of a failed one (the shape itself if invalid).
    fn slice_shape(&self, axis: usize, start: Bound<usize>, end: Bound<usize>) -> Shape {
        let mut dims = self.shape().dims();
        if let Some((start, end)) = self.slice_bounds(axis, start, end) {
            dims[axis] = end - start;
        }
        Shape::new(dims)
    }

    fn run_slice(
        &self,
        axis: usize,
        start: Bound<usize>,
        end: Bound<usize>,
        writable: bool,
        location: &'static Location<'static>,
    ) -> Outcome<Array<T>> {
        self.src()?;
        let Some((start, end)) = self.slice_bounds(axis, start, end) else {
            let range = format!(
                "{}..{}",
                match start {
                    Bound::Included(s) => s.to_string(),
                    Bound::Excluded(s) => (s + 1).to_string(),
                    Bound::Unbounded => "0".to_owned(),
                },
                match end {
                    Bound::Included(e) => (e + 1).to_string(),
                    Bound::Excluded(e) => e.to_string(),
                    Bound::Unbounded => self.shape().dims().get(axis).map_or_else(|| "?".to_owned(), ToString::to_string),
                }
            );
            return Err(LumaError::InvalidShape(format!("slice {range} of axis {axis} of shape {}", self.shape())).into());
        };
        let src = dims32(self.shape())?;
        let mut out = src;
        out[axis] = u32::try_from(end - start).map_err(|_| LumaError::DimensionTooLarge { shape: self.shape().dims() })?;
        let mut map = ViewMap::identity(src);
        map.out = out;
        map.axes[axis].start = i32::try_from(start).map_err(|_| LumaError::DimensionTooLarge { shape: self.shape().dims() })?;
        let mut dims = self.shape().dims();
        dims[axis] = end - start;
        self.run_view_with(map, Edge::None, None, Shape::new(dims), writable, "slice", location)
    }

    /// A shift that does not wrap: element `i` moves to `i + offset` in every dimension (as
    /// [`Array::shift`]) and the cells that no element moves into read `value`.
    /// `[1, 2, 3, 4].shift_fill(1, 9.0)` is `[9, 1, 2, 3]`. Lazy like a slice.
    #[track_caller]
    pub fn shift_fill(&self, offsets: impl Into<Offsets>, value: T) -> Array<T> {
        self.shift_fill_at(offsets.into(), value, Location::caller())
    }

    /// [`Array::shift_fill`] reporting failures at `location`.
    pub(crate) fn shift_fill_at(&self, offsets: Offsets, value: T, location: &'static Location<'static>) -> Array<T> {
        let result = Scalar::of(value)
            .map_err(Into::into)
            .and_then(|fill| self.run_edge_shift(offsets, Edge::Fill, Some(fill), "shift_fill", location));
        Array::resolve(self.context(), self.shape(), result, "shift_fill", location)
    }

    /// A shift that does not wrap: cells that no element moves into repeat the nearest element.
    /// `[1, 2, 3, 4].shift_clamp(-1)` is `[2, 3, 4, 4]`. Lazy like a slice.
    #[track_caller]
    pub fn shift_clamp(&self, offsets: impl Into<Offsets>) -> Array<T> {
        self.shift_clamp_at(offsets.into(), Location::caller())
    }

    /// [`Array::shift_clamp`] reporting failures at `location`.
    pub(crate) fn shift_clamp_at(&self, offsets: Offsets, location: &'static Location<'static>) -> Array<T> {
        let result = self.run_edge_shift(offsets, Edge::Clamp, None, "shift_clamp", location);
        Array::resolve(self.context(), self.shape(), result, "shift_clamp", location)
    }

    /// A shift that does not wrap: cells that no element moves into mirror the array about its
    /// boundary face (`-1` reads `0`, `-2` reads `1`, `n` reads `n - 1`), the ghost-cell
    /// convention of finite-volume methods. `[1, 2, 3, 4].shift_mirror(2)` is `[2, 1, 1, 2]`.
    /// Lazy like a slice.
    #[track_caller]
    pub fn shift_mirror(&self, offsets: impl Into<Offsets>) -> Array<T> {
        self.shift_mirror_at(offsets.into(), Location::caller())
    }

    /// [`Array::shift_mirror`] reporting failures at `location`.
    pub(crate) fn shift_mirror_at(&self, offsets: Offsets, location: &'static Location<'static>) -> Array<T> {
        let result = self.run_edge_shift(offsets, Edge::Mirror, None, "shift_mirror", location);
        Array::resolve(self.context(), self.shape(), result, "shift_mirror", location)
    }

    fn run_edge_shift(
        &self,
        offsets: Offsets,
        edge: Edge,
        fill: Option<Scalar>,
        name: &'static str,
        location: &'static Location<'static>,
    ) -> Outcome<Array<T>> {
        self.src()?;
        let mut map = ViewMap::identity(dims32(self.shape())?);
        for (axis, &offset) in map.axes.iter_mut().zip(&offsets.0) {
            axis.start = i32::try_from(-(offset as i64))
                .map_err(|_| LumaError::DimensionTooLarge { shape: self.shape().dims() })?;
        }
        self.run_view(map, edge, fill, self.shape(), name, location)
    }
}

impl<T: ScalarType> Array<T> {
    /// The array with `before` / `after` extra cells on both sides of axis `k` for every
    /// `widths[k]` (axes beyond `N` are not padded), the new cells filled as `mode` says.
    /// Lazy like a slice: `a.pad([(2, 2)], PadMode::Edge)` reads `a` through an index map.
    ///
    /// Fails (poisoned array) if a padded extent does not fit `u32`.
    #[track_caller]
    pub fn pad<const N: usize>(&self, widths: [(usize, usize); N], mode: PadMode) -> Array<T> {
        self.pad_at(widths, mode, Location::caller())
    }

    /// [`Array::pad`] reporting failures at `location`.
    pub(crate) fn pad_at<const N: usize>(
        &self,
        widths: [(usize, usize); N],
        mode: PadMode,
        location: &'static Location<'static>,
    ) -> Array<T> {
        const { assert!(N <= 4) };
        let mut dims = self.shape().dims();
        for (dim, &(before, after)) in dims.iter_mut().zip(&widths) {
            *dim = dim.saturating_add(before).saturating_add(after);
        }
        let result = self.run_pad(&widths, mode, location);
        Array::resolve(self.context(), Shape::new(dims), result, "pad", location)
    }

    fn run_pad(
        &self,
        widths: &[(usize, usize)],
        mode: PadMode,
        location: &'static Location<'static>,
    ) -> Outcome<Array<T>> {
        self.src()?;
        let too_large = || LumaError::DimensionTooLarge { shape: self.shape().dims() };
        let mut dims = self.shape().dims();
        let mut map = ViewMap::identity(dims32(self.shape())?);
        for (k, &(before, after)) in widths.iter().enumerate() {
            dims[k] = dims[k].checked_add(before).and_then(|d| d.checked_add(after)).ok_or_else(too_large)?;
            map.out[k] = u32::try_from(dims[k]).map_err(|_| too_large())?;
            map.axes[k].start = i32::try_from(before).map(|b| -b).map_err(|_| too_large())?;
        }
        let (edge, fill) = match mode {
            PadMode::Edge => (Edge::Clamp, None),
            PadMode::Mirror => (Edge::Mirror, None),
            PadMode::Wrap => (Edge::Wrap, None),
            PadMode::Zero => (Edge::Fill, Some(Scalar::of(T::zeroed())?)),
        };
        self.run_view(map, edge, fill, Shape::new(dims), "pad", location)
    }

    /// The array reversed along `axis`: element `i` of the result is element `n - 1 - i` of
    /// the source. Lazy like a slice, and read-only like one ([`Array::flip_mut`] writes
    /// through). Fails (poisoned array) if `axis >= 4`.
    #[track_caller]
    pub fn flip(&self, axis: usize) -> Array<T> {
        self.flip_at(axis, false, Location::caller())
    }

    /// [`Array::flip`] that writes through, as a [`ViewMut`] (see [`Array::slice_mut`]).
    #[track_caller]
    pub fn flip_mut(&mut self, axis: usize) -> ViewMut<'_, Array<T>> {
        let inner = self.flip_at(axis, true, Location::caller());
        ViewMut { inner, owner: self }
    }

    /// [`Array::flip`] (or the view of [`Array::flip_mut`] if `writable`) reporting failures at
    /// `location`.
    pub(crate) fn flip_at(&self, axis: usize, writable: bool, location: &'static Location<'static>) -> Array<T> {
        let result = self.run_flip(axis, writable, location);
        Array::resolve(self.context(), self.shape(), result, "flip", location)
    }

    fn run_flip(&self, axis: usize, writable: bool, location: &'static Location<'static>) -> Outcome<Array<T>> {
        self.src()?;
        if axis >= 4 {
            return Err(LumaError::InvalidShape(format!("flip of axis {axis} of shape {}", self.shape())).into());
        }
        let src = dims32(self.shape())?;
        let mut map = ViewMap::identity(src);
        let last = i32::try_from(src[axis] - 1).map_err(|_| LumaError::DimensionTooLarge { shape: self.shape().dims() })?;
        map.axes[axis] = AxisMap { from: axis as u8, start: last, step: -1 };
        self.run_view_with(map, Edge::None, None, self.shape(), writable, "flip", location)
    }

    /// The array with its first two axes swapped: element `(x, y)` of the result is element
    /// `(y, x)` of the source. Lazy like a slice, and read-only like one
    /// ([`Array::transpose_mut`] writes through).
    #[track_caller]
    pub fn transpose(&self) -> Array<T> {
        self.transpose_at(false, Location::caller())
    }

    /// [`Array::transpose`] that writes through, as a [`ViewMut`] (see [`Array::slice_mut`]).
    #[track_caller]
    pub fn transpose_mut(&mut self) -> ViewMut<'_, Array<T>> {
        let inner = self.transpose_at(true, Location::caller());
        ViewMut { inner, owner: self }
    }

    /// [`Array::transpose`] (or the view of [`Array::transpose_mut`] if `writable`) reporting
    /// failures at `location`.
    pub(crate) fn transpose_at(&self, writable: bool, location: &'static Location<'static>) -> Array<T> {
        let [d0, d1, d2, d3] = self.shape().dims();
        let shape = Shape::new([d1, d0, d2, d3]);
        let result = self.run_transpose(shape, writable, location);
        Array::resolve(self.context(), shape, result, "transpose", location)
    }

    fn run_transpose(&self, shape: Shape, writable: bool, location: &'static Location<'static>) -> Outcome<Array<T>> {
        self.src()?;
        let src = dims32(self.shape())?;
        let mut map = ViewMap::identity(src);
        map.out = [src[1], src[0], src[2], src[3]];
        map.axes[0].from = 1;
        map.axes[1].from = 0;
        self.run_view_with(map, Edge::None, None, shape, writable, "transpose", location)
    }

    /// Each element's index along `axis` (see [`Array::range`]) for this array's shape.
    #[track_caller]
    pub fn index(&self, axis: usize) -> Array<u32> {
        let location = Location::caller();
        let result = Array::<u32>::range_on(self.context(), self.shape(), axis).map_err(Failure::New);
        Array::resolve(self.context(), self.shape(), result, "index", location)
    }

    /// Gathers: element `i` of the result is element `min(indices[i], len - 1)` of this array
    /// (flat index, the first dimension varies fastest). The result has the shape of `indices`.
    /// Lazy like a slice: the gather fuses into the expression that reads it.
    #[track_caller]
    pub fn take(&self, indices: &Array<u32>) -> Array<T> {
        self.take_at(indices, Location::caller())
    }

    /// [`Array::take`] reporting failures at `location`.
    pub(crate) fn take_at(&self, indices: &Array<u32>, location: &'static Location<'static>) -> Array<T> {
        let result = self.run_take(indices, location);
        Array::resolve(self.context(), indices.shape(), result, "take", location)
    }

    fn run_take(&self, indices: &Array<u32>, location: &'static Location<'static>) -> Outcome<Array<T>> {
        let ctx = self.context();
        self.src()?;
        let index = indices.src()?;
        if !ctx.same_as(indices.context()) {
            return Err(LumaError::ContextMismatch.into());
        }
        let len = u32::try_from(self.len()).map_err(|_| LumaError::DimensionTooLarge { shape: self.shape().dims() })?;
        let operand = match self.pending() {
            Some(node) if jit::recompute_viewed(node) => Src::Node(Arc::clone(node)),
            _ => Src::Buffer(Arc::clone(self.storage_alongside()?)),
        };
        let op = Op::new(OpKind::Shift(ShiftKind::Take { len }), ElemTy::of::<T>()?);
        let node = jit::create(ctx, indices.len(), op, [operand, index, Src::unused()], "take", location)?;
        Ok(Array::from_node(ctx, indices.shape(), node))
    }

    /// This array evaluated: its expression is recorded as a kernel now (together with the other
    /// arrays that share work with it), and the result is an ordinary array that later reads
    /// take from its buffer instead of recomputing the expression. It does not wait for the GPU.
    ///
    /// Fusion otherwise computes the expression inside every kernel that reads it through a view
    /// (slices, pads, gathers), once per distinct index, and evaluates it only when such a
    /// kernel would get too big; `cache` is for when you know better, for example a flux that
    /// many kernels read, or an expression whose arithmetic outweighs a buffer read.
    /// `a.cache()` of an array that is evaluated already is free.
    #[track_caller]
    pub fn cache(&self) -> Array<T> {
        self.cache_at(Location::caller())
    }

    /// [`Array::cache`] reporting failures at `location`.
    pub(crate) fn cache_at(&self, location: &'static Location<'static>) -> Array<T> {
        let result = self.storage_alongside().map(|storage| Array::from_storage(self.context(), self.shape(), Arc::clone(storage)));
        Array::resolve(self.context(), self.shape(), result, "cache", location)
    }

    // -- Writes --

    /// Writes `rhs` into this array in place: `w.assign(0.0)` fills the buffer of `w` for every
    /// array sharing it. `rhs` is an array of this one's shape (`&expr`, or `expr` by value) or
    /// a single value. Through a slice, flip or transpose, write with [`Array::slice_mut`],
    /// [`Array::flip_mut`] or [`Array::transpose_mut`] ([`ViewMut::assign`]); `assign` on a
    /// read-only view fails with [`LumaError::InvalidShape`].
    ///
    /// Only views that keep every element distinct write through. For any other array (a
    /// temporary expression, a shift, a pad, a gather) the write goes into the array's own
    /// buffer, so what it was computed from is unchanged.
    ///
    /// If the array a [`ViewMut`] was taken from is still a lazy expression (a padded state,
    /// say), nothing is written and nothing is launched: from then on that array *is* the
    /// expression with the region replaced, and fuses into whatever reads it. The values are
    /// the same as when the array has a buffer.
    ///
    /// The expression may read the target: `w.slice_mut(0, ..2).assign(rhs)` with
    /// `rhs = w.slice(0, 2..4).flip(0) * -1.0` fills ghost cells from interior ones. Lazy arrays
    /// recorded before the write keep the values they were recorded against. The whole
    /// expression fuses into the write (one kernel) when it does not read the cells being
    /// written, or when it is passed by value and no other array still holds a part of it that
    /// reads them; otherwise those parts are computed first. The result is the same either way.
    ///
    /// Fails with [`LumaError::ReadOnly`] on the state passed to an `iterate` step and with
    /// [`LumaError::Chunked`] on arrays split over several buffers.
    #[track_caller]
    pub fn assign(&mut self, rhs: impl Operand<T>) -> Result<()> {
        self.assign_in(rhs, Location::caller())
    }

    /// [`Array::assign`] through a shared reference (what a [`ViewMut`] holds).
    pub(crate) fn assign_in(&self, rhs: impl Operand<T>, location: &'static Location<'static>) -> Result<()> {
        if self.is_read_view() {
            let checked = self.src().and_then(|_| -> Outcome<()> {
                Err(LumaError::InvalidShape(
                    "cannot assign through a read-only view; use slice_mut, flip_mut or transpose_mut".into(),
                )
                .into())
            });
            return finish(checked, "assign", location);
        }
        let region = match self.assign_lazy(rhs.rhs(), location) {
            Ok(true) => Ok(None),
            Ok(false) => self.region(rhs.rhs()).map(Some),
            Err(error) => Err(error),
        };
        // The right-hand side is only held by the region now: a temporary expression is private
        // to the kernel that stores it.
        drop(rhs);
        let result = match region {
            Ok(Some(region)) => jit::assign::assign(self.context(), vec![region], "assign", location),
            Ok(None) => Ok(()),
            Err(error) => Err(error),
        };
        finish(result, "assign", location)
    }

    /// Fails if `rhs` cannot be written into this array (poisoned, other context or shape).
    pub(crate) fn check_write(&self, rhs: Rhs<'_, T>) -> Outcome<()> {
        self.write_operand(&rhs).map(|_| ())
    }

    /// The operand `rhs` stands for in a write into this array: checked against this array's
    /// context and shape.
    fn write_operand(&self, rhs: &Rhs<'_, T>) -> Outcome<Src> {
        match rhs {
            Rhs::Array(other) => {
                let operand = other.src()?;
                if !self.context().same_as(other.context()) {
                    return Err(LumaError::ContextMismatch.into());
                }
                if self.shape() != other.shape() {
                    return Err(LumaError::ShapeMismatch { lhs: self.shape().dims(), rhs: other.shape().dims() }.into());
                }
                Ok(operand)
            }
            Rhs::Scalar(value) => Ok(Src::Scalar(Scalar::of(*value)?)),
        }
    }

    /// The region that writing `rhs` into this array overwrites.
    pub(crate) fn region(&self, rhs: Rhs<'_, T>) -> Outcome<Region> {
        let ctx = self.context();
        let (base, map, receiver, claim) = match self.view() {
            Some(view) => {
                self.src()?;
                let base = view.storage(ctx)?;
                (base, view.map, self.pending().cloned(), self.held().map(Held::share))
            }
            None => (Arc::clone(self.storage()?), ViewMap::identity(dims32(self.shape())?), None, None),
        };
        let rhs = self.write_operand(&rhs)?;
        Ok(Region { base, map, rhs, ty: ElemTy::of::<T>()?, receiver, claim })
    }

    /// The write of `rhs` into this array if the array it is a view of is a pending expression
    /// that is read through steps of one cell: `Ok(true)` when done, `Ok(false)` when this write
    /// needs the buffer (see [`Array::region`]).
    ///
    /// The base then has no buffer to protect and nothing is launched: from now on the arrays
    /// holding it denote `select(in_region, rhs, old base)`, `old` being the base's
    /// expression and `rhs` read at base coordinates through the inverse of the view. Expressions
    /// recorded earlier keep reading the old base, and the receiver, pending or evaluated,
    /// becomes a view of the new base.
    pub(crate) fn assign_lazy(&self, rhs: Rhs<'_, T>, location: &'static Location<'static>) -> Outcome<bool> {
        let Some(view) = self.view() else { return Ok(false) };
        let Some(base) = &view.held else { return Ok(false) };
        let (old, map) = (base.latest(), view.map);
        if !jit::can_supersede(old) || map.axes.iter().any(|axis| axis.step.abs() != 1) {
            return Ok(false);
        }
        self.src()?;
        self.write_operand(&rhs)?;
        let ctx = self.context();
        let ty = ElemTy::of::<T>()?;
        let too_large = || LumaError::DimensionTooLarge { shape: self.shape().dims() };

        // The cells of the base the region covers: a box, as a mask over the base.
        let base_shape = Shape::new(map.src.map(|d| d as usize));
        let mut mask: Option<Array<u32>> = None;
        for a in 0..4 {
            let (first, last) = map.ends(a);
            let (lo, hi) = (first.min(last), first.max(last));
            if lo == 0 && hi == i64::from(map.src[a]) - 1 {
                continue;
            }
            let at = Array::<u32>::range_on(ctx, base_shape, a)?;
            // The side that touches the edge of the base needs no comparison.
            let below = (lo > 0).then(|| at.compare_at(Rhs::Scalar(lo as u32), CmpOp::Ge, "ge", location));
            let above = (hi < i64::from(map.src[a]) - 1).then(|| at.compare_at(Rhs::Scalar(hi as u32), CmpOp::Le, "le", location));
            let inside = match (below, above) {
                (Some(below), Some(above)) => below.binary_at(Rhs::Array(&above), BinOp::Mul, false, "mul", location),
                (Some(side), None) | (None, Some(side)) => side,
                (None, None) => continue,
            };
            mask = Some(match mask {
                Some(rest) => rest.binary_at(Rhs::Array(&inside), BinOp::Mul, false, "mul", location),
                None => inside,
            });
        }

        // The right-hand side at base coordinates: base axis `a` reads region axis `from` at
        // `step * (c - start)`.
        let back = match &rhs {
            Rhs::Scalar(value) => Src::Scalar(Scalar::of(*value)?),
            Rhs::Array(other) => {
                let mut inverse = ViewMap::identity(map.src);
                inverse.src = map.out;
                for (a, axis) in map.axes.iter().enumerate() {
                    let start = -i64::from(axis.step) * i64::from(axis.start);
                    inverse.axes[axis.from as usize] =
                        AxisMap { from: a as u8, start: i32::try_from(start).map_err(|_| too_large())?, step: axis.step };
                }
                other.run_view(inverse, Edge::Clamp, None, base_shape, "assign", location)?.src()?
            }
        };

        let new = match &mask {
            Some(mask) => {
                let args = [mask.src()?, back, Src::Node(Arc::clone(old))];
                jit::create(ctx, old.len(), Op::new(OpKind::Select, ty), args, "assign", location)?
            }
            // The whole base: its new value is the right-hand side.
            None => match back {
                Src::Node(node) => node,
                other => {
                    let args = [other, Src::unused(), Src::unused()];
                    jit::create(ctx, old.len(), Op::new(OpKind::Unary(UnOp::Id), ty), args, "assign", location)?
                }
            },
        };

        // What the receiver shows afterwards: its region of the new base, whether it was still
        // pending or an evaluation (by `eval`, or together with the arrays sharing work with
        // another one) gave it a buffer with the data from before the write.
        let viewed = match self.held() {
            Some(receiver) => {
                let op = Op::new(OpKind::Shift(ShiftKind::View { map, edge: Edge::None }), ty);
                let args = [Src::Node(Arc::clone(&new)), Src::unused(), Src::unused()];
                Some((receiver, jit::create(ctx, self.len(), op, args, "assign", location)?))
            }
            None => None,
        };
        jit::supersede(base, new);
        if let Some((receiver, node)) = viewed {
            // An array evaluated before the write keeps its buffer, which is overwritten (arrays
            // sharing it, a `cache()`, show the written data); a pending one is replaced.
            if let Some(node) = jit::assign::write_evaluated_receiver(ctx, receiver, node, "assign", location)? {
                jit::supersede(receiver, node);
            }
        }
        Ok(true)
    }

    /// Scatters: `self[indices[i]] = values[i]` for every `i`, in place. Indices of `len` or
    /// more are skipped; of several equal indices one (unspecified) wins. `values` is an array
    /// of the shape of `indices` or a single value. Must be called on the array itself, not on a
    /// view of another one. Lazy arrays recorded before the write keep their values.
    #[track_caller]
    pub fn put(&mut self, indices: &Array<u32>, values: impl Operand<T>) -> Result<()> {
        let location = Location::caller();
        let prepared = self.scatter_args(indices, values.rhs());
        drop(values);
        let result = prepared.and_then(|(base, index, values)| {
            jit::assign::scatter(self.context(), &base, index, values, ElemTy::of::<T>()?, indices.len(), "put", location)
        });
        finish(result, "put", location)
    }

    /// The target buffer, the index operand and the value operand of [`Array::put`].
    fn scatter_args(&self, indices: &Array<u32>, values: Rhs<'_, T>) -> Outcome<(Arc<Storage>, Src, Src)> {
        let ctx = self.context();
        if self.is_read_view() {
            self.src()?;
            return Err(LumaError::InvalidShape("cannot put into a read-only view; use slice_mut, flip_mut or transpose_mut".into()).into());
        }
        if self.view().is_some() {
            self.src()?;
            return Err(LumaError::InvalidShape("put through a view; call it on the array".into()).into());
        }
        let base = Arc::clone(self.storage()?);
        let index = indices.src()?;
        if !ctx.same_as(indices.context()) {
            return Err(LumaError::ContextMismatch.into());
        }
        let values = match values {
            Rhs::Array(other) => {
                let operand = other.src()?;
                if !ctx.same_as(other.context()) {
                    return Err(LumaError::ContextMismatch.into());
                }
                if indices.shape() != other.shape() {
                    return Err(LumaError::ShapeMismatch { lhs: indices.shape().dims(), rhs: other.shape().dims() }.into());
                }
                operand
            }
            Rhs::Scalar(value) => Src::Scalar(Scalar::of(value)?),
        };
        Ok((base, index, values))
    }
}

/// Turns the outcome of a write into the error the caller gets: a failure is tagged with the
/// operation and its call site, an inherited one passed on untouched.
pub(crate) fn finish(result: Outcome<()>, op: &'static str, location: &'static Location<'static>) -> Result<()> {
    match result {
        Ok(()) => Ok(()),
        Err(Failure::Inherited(error)) => Err(LumaError::clone(&error)),
        Err(Failure::New(error)) => Err(LumaError::Op { op, location, source: Arc::new(error) }),
    }
}

/// Joins `parts` along `axis` (0 varies fastest) into one array. The parts must share a
/// context and every extent but the one along `axis`; otherwise the result is poisoned
/// ([`LumaError::ShapeMismatch`], [`LumaError::ContextMismatch`], [`LumaError::InvalidShape`]
/// for `axis >= 4`). Lazy: every part is read through a view and the pieces are selected by the
/// coordinate along `axis`, so the result fuses into the expression using it. The parts may be
/// lazy expressions too (ghost cells computed from the state, say): they are not evaluated
/// first but computed inside the kernel that reads the join, which evaluates only the part the
/// coordinate selects at every cell. Only an expression too large to unfold
/// there (more than `LUMA_FUSION_MAX_INLINED` instructions in all) is evaluated into a buffer
/// first; [`Array::cache`] makes that choice explicit. To join many expressions into an array
/// that is read as a whole (the entries of a block matrix), [`stack`] writes them in one kernel.
#[track_caller]
pub fn concat<T: ScalarType, const N: usize>(axis: usize, parts: [&Array<T>; N]) -> Array<T> {
    const { assert!(N >= 1) };
    let location = Location::caller();
    let first = parts[0];
    let result = run_concat(axis, &parts, location);
    Array::resolve(first.context(), first.shape(), result, "concat", location)
}

fn run_concat<T: ScalarType>(
    axis: usize,
    parts: &[&Array<T>],
    location: &'static Location<'static>,
) -> Outcome<Array<T>> {
    let first = parts[0];
    let ctx = first.context();
    for part in parts {
        part.src()?;
        if !ctx.same_as(part.context()) {
            return Err(LumaError::ContextMismatch.into());
        }
    }
    if axis >= 4 {
        return Err(LumaError::InvalidShape(format!("concat along axis {axis} of shape {}", first.shape())).into());
    }
    let mut dims = first.shape().dims();
    for part in &parts[1..] {
        let other = part.shape().dims();
        if (0..4).any(|a| a != axis && other[a] != dims[a]) {
            return Err(LumaError::ShapeMismatch { lhs: first.shape().dims(), rhs: other }.into());
        }
        dims[axis] += other[axis];
    }
    let shape = Shape::new(dims);
    shape.validate()?;
    let out = dims32(shape)?;

    let mut views = Vec::with_capacity(parts.len());
    let mut offsets = Vec::with_capacity(parts.len());
    let mut offset = 0usize;
    for part in parts {
        let mut map = ViewMap::identity(dims32(part.shape())?);
        map.out = out;
        map.axes[axis].start = i32::try_from(offset).map(|o| -o).map_err(|_| LumaError::DimensionTooLarge { shape: dims })?;
        views.push(part.run_view(map, Edge::Clamp, None, shape, "concat", location)?);
        offsets.push(offset);
        offset += part.shape().dims()[axis];
    }
    select_parts(shape, axis, &views, &offsets, location)
}

/// The array of `shape` whose element is that of `views[j]`, `j` being the last part that starts
/// at or below its coordinate along `axis` (`offsets[j]` is where part `j` starts): a chain of
/// selects on the coordinate.
fn select_parts<T: ScalarType>(
    shape: Shape,
    axis: usize,
    views: &[Array<T>],
    offsets: &[usize],
    location: &'static Location<'static>,
) -> Outcome<Array<T>> {
    let ctx = views[0].context();
    let coord = Array::<u32>::range_on(ctx, shape, axis)?;
    let mut acc = views.last().ok_or_else(|| LumaError::InvalidShape("join of no arrays".into()))?.clone();
    for j in (0..views.len() - 1).rev() {
        let below = coord.lt(offsets[j + 1] as u32);
        acc = run_select(&below, Rhs::Array(&views[j]), Rhs::Array(&acc), location)?;
    }
    Ok(acc)
}

/// Joins equal-shaped `parts` along a new axis inserted at position `axis` (0 varies fastest):
/// element `k` of the result along `axis` is `parts[k]`, so stacking `[ni, nj]` arrays along
/// axis 0 gives a `[parts, ni, nj]` array (and `[ni, nj, parts]` along axis 2). The parts must
/// share a context and a shape whose last extent is 1 (the new axis takes its place), and
/// `axis <= 3`; otherwise the result is poisoned ([`LumaError::ShapeMismatch`],
/// [`LumaError::ContextMismatch`], [`LumaError::InvalidShape`]).
///
/// The parts may be lazy expressions: they are fused into one kernel that computes one element
/// of every part and writes them side by side in the stacked layout, so every output element is
/// computed once and the cost is close to writing the result. The result is an evaluated array
/// (the kernel is recorded when `stack` is called, as for [`Array::cache`]); a result too large
/// for one buffer, or parts that read arrays split over several buffers, are joined lazily
/// instead, like [`concat()`] does.
#[track_caller]
pub fn stack<T: ScalarType, const N: usize>(axis: usize, parts: [&Array<T>; N]) -> Array<T> {
    const { assert!(N >= 1) };
    let location = Location::caller();
    stack_of(axis, &parts, location)
}

/// [`stack`] of the arrays of a slice. An empty slice gives a poisoned array
/// ([`LumaError::InvalidShape`]) on the [`default_context`](crate::default_context).
#[track_caller]
pub fn stack_slice<T: ScalarType>(axis: usize, parts: &[Array<T>]) -> Array<T> {
    let location = Location::caller();
    let refs: Vec<&Array<T>> = parts.iter().collect();
    stack_of(axis, &refs, location)
}

fn stack_of<T: ScalarType>(axis: usize, parts: &[&Array<T>], location: &'static Location<'static>) -> Array<T> {
    let Some(&first) = parts.first() else {
        let error = LumaError::InvalidShape("stack of no arrays".into());
        let error = LumaError::Op { op: "stack", location, source: Arc::new(error) };
        let ctx = crate::default_context().unwrap_or_else(|e| panic!("stack of no arrays: no default context: {e}"));
        return Array::poisoned(&ctx, Shape::new([1; 4]), Arc::new(error));
    };
    let mut dims = first.shape().dims();
    let at = axis.min(3);
    dims.copy_within(at..3, at + 1);
    dims[at] = parts.len();
    let result = run_stack(axis, parts, location);
    Array::resolve(first.context(), Shape::new(dims), result, "stack", location)
}

fn run_stack<T: ScalarType>(
    axis: usize,
    parts: &[&Array<T>],
    location: &'static Location<'static>,
) -> Outcome<Array<T>> {
    let first = parts[0];
    let ctx = first.context();
    let mut srcs = Vec::with_capacity(parts.len());
    for part in parts {
        srcs.push(part.src()?);
        if !ctx.same_as(part.context()) {
            return Err(LumaError::ContextMismatch.into());
        }
    }
    let dims = first.shape().dims();
    if axis >= 4 || dims[3] != 1 {
        return Err(LumaError::InvalidShape(format!("stack along axis {axis} of shape {}", first.shape())).into());
    }
    for part in &parts[1..] {
        if part.shape() != first.shape() {
            return Err(LumaError::ShapeMismatch { lhs: dims, rhs: part.shape().dims() }.into());
        }
    }
    let mut out_dims = dims;
    out_dims.copy_within(axis..3, axis + 1);
    out_dims[axis] = parts.len();
    let shape = Shape::new(out_dims);
    shape.validate()?;

    let block: usize = dims[..axis].iter().product();
    let ty = ElemTy::of::<T>()?;
    if let Some(storage) = jit::stack::run(ctx, &srcs, ty, first.len(), block, "stack", location)? {
        return Ok(Array::from_storage(ctx, shape, storage));
    }

    // The part `k` read as an array of the stacked shape: its own coordinates, the new axis
    // skipped.
    let out = dims32(shape)?;
    let mut views = Vec::with_capacity(parts.len());
    for part in parts {
        let mut map = ViewMap::identity(dims32(part.shape())?);
        map.out = out;
        for (a, slot) in map.axes.iter_mut().enumerate() {
            *slot = match a {
                _ if a < axis => AxisMap { from: a as u8, start: 0, step: 1 },
                0..=2 => AxisMap { from: a as u8 + 1, start: 0, step: 1 },
                _ => AxisMap { from: 0, start: 0, step: 0 },
            };
        }
        views.push(part.run_view(map, Edge::None, None, shape, "stack", location)?);
    }
    let offsets: Vec<usize> = (0..parts.len()).collect();
    select_parts(shape, axis, &views, &offsets, location)
}

impl ViewOf {
    /// The buffer the view reads and writes: the base itself, or the buffer of the base
    /// expression (evaluated now). An assignment that replaced the base expression made the
    /// arrays that held it denote another one: that is what is written.
    pub(crate) fn storage(&self, ctx: &Context) -> Outcome<Arc<Storage>> {
        match &self.base {
            Src::Buffer(storage) => Ok(Arc::clone(storage)),
            Src::Node(node) => {
                let node = self.held.as_ref().map_or(node, Held::latest);
                if node.is_pending() {
                    ctx.jit().evaluate_alongside(Arc::clone(node));
                }
                match node.force() {
                    Ok(storage) => Ok(Arc::clone(storage)),
                    Err(error) => Err(Failure::Inherited(Arc::clone(error))),
                }
            }
            Src::Scalar(_) => Err(LumaError::InvalidShape("view of a scalar".into()).into()),
        }
    }
}

/// The index map of a strided view of a buffer, as a custom kernel reads it (`LumaView`):
/// element `o` of the view is `buffer[offset + sum_k stride[k] * o_k]`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct RawViewDesc {
    pub offset: u32,
    pub len: u32,
    pub _pad: [u32; 2],
    pub stride: [i32; 4],
    pub dims: [u32; 4],
}

// SAFETY: `#[repr(C)]` over 12 four-byte integers: no padding, every bit pattern valid.
unsafe impl bytemuck::Zeroable for RawViewDesc {}
// SAFETY: see above.
unsafe impl bytemuck::Pod for RawViewDesc {}

impl RawViewDesc {
    /// The descriptor of the view `map` of a contiguous source.
    pub(crate) fn from_map(map: &ViewMap) -> Self {
        let s = map.src.map(i64::from);
        let src_stride = [1, s[0], s[0] * s[1], s[0] * s[1] * s[2]];
        let offset = (0..4).map(|a| src_stride[a] * i64::from(map.axes[a].start)).sum::<i64>() as u32;
        // An axis of extent 1 is only ever read at 0, whatever its step.
        let stride = std::array::from_fn(|k| {
            (0..4)
                .filter(|&a| map.axes[a].from as usize == k && map.out[k] > 1)
                .map(|a| src_stride[a] * i64::from(map.axes[a].step))
                .sum::<i64>() as i32
        });
        RawViewDesc { offset, len: map.out_len() as u32, _pad: [0; 2], stride, dims: map.out }
    }

    /// The descriptor of a whole array of `shape`.
    pub(crate) fn contiguous(shape: Shape) -> Result<Self, LumaError> {
        let dims = dims32(shape)?;
        let d = dims.map(i64::from);
        let stride = [1, d[0], d[0] * d[1], d[0] * d[1] * d[2]].map(|s| s as i32);
        Ok(RawViewDesc { offset: 0, len: shape.len() as u32, _pad: [0; 2], stride, dims })
    }
}

impl<T: GpuType> Array<T> {
    /// A pin on the buffer this array reads, the buffer's size in bytes and the strided view
    /// of it that is the array: a slice, flip or transpose is read in place through its index
    /// map (nothing is copied), any other array as a contiguous one (evaluated first).
    pub(crate) fn ready_view(&self) -> std::result::Result<(Pin, u64, RawViewDesc), Failure> {
        if let Some(node) = self.pending()
            && let OpKind::Shift(ShiftKind::View { map, edge: Edge::None }) = node.op().kind
            && dims32(self.shape())? == map.out
            && let Some(args) = node.operands()
        {
            let storage = match &args[0] {
                Src::Buffer(storage) => Arc::clone(storage),
                Src::Node(inner) => {
                    if inner.is_pending() {
                        self.context().jit().evaluate_alongside(Arc::clone(inner));
                    }
                    match inner.force() {
                        Ok(storage) => Arc::clone(storage),
                        Err(error) => return Err(Failure::Inherited(Arc::clone(error))),
                    }
                }
                Src::Scalar(_) => return Err(LumaError::InvalidShape("view of a scalar".into()).into()),
            };
            return Ok((storage.pin_single()?, storage.byte_len(), RawViewDesc::from_map(&map)));
        }
        let (pin, bytes) = self.ready()?;
        Ok((pin, bytes, RawViewDesc::contiguous(self.shape())?))
    }
}

/// A slice, flip or transpose of an array (or of a record of arrays) that writes through to
/// what it was taken from: [`Array::slice_mut`], [`Array::flip_mut`], [`Array::transpose_mut`]
/// and the same on records ([`Fields`]). It borrows the array mutably for as long as it lives.
///
/// [`ViewMut::assign`] writes an expression (or a value) into the region, and the compound
/// operators write `region op rhs` into it: `w.slice_mut(0, 1..3) += 10.0`. They are lazy like
/// every write: nothing waits for the GPU. A failed compound operator poisons the array the
/// view was taken from, like `+=` on an array does; `assign` returns its error.
/// [`ViewMut::slice`], [`ViewMut::flip`] and [`ViewMut::transpose`] refine the region further.
/// Custom kernels take `&mut view` as an argument and write the region in place.
pub struct ViewMut<'a, F: Fields> {
    /// The region, as arrays that write through.
    pub(crate) inner: F,
    /// What the region was taken from, poisoned when a compound operator fails.
    pub(crate) owner: &'a mut F,
}

impl<'a, F: Fields> ViewMut<'a, F> {
    /// The region's part `range` along `axis` (see [`Array::slice_mut`]).
    #[track_caller]
    pub fn slice(self, axis: usize, range: impl RangeBounds<usize>) -> Self {
        let location = Location::caller();
        let bounds = (range.start_bound().cloned(), range.end_bound().cloned());
        let inner = self.inner.map(|a| a.slice_at(axis, bounds, true, location));
        ViewMut { inner, owner: self.owner }
    }

    /// The region reversed along `axis` (see [`Array::flip_mut`]).
    #[track_caller]
    pub fn flip(self, axis: usize) -> Self {
        let location = Location::caller();
        let inner = self.inner.map(|a| a.flip_at(axis, true, location));
        ViewMut { inner, owner: self.owner }
    }

    /// The region with its first two axes swapped (see [`Array::transpose_mut`]).
    #[track_caller]
    pub fn transpose(self) -> Self {
        let location = Location::caller();
        let inner = self.inner.map(|a| a.transpose_at(true, location));
        ViewMut { inner, owner: self.owner }
    }

    /// The shape of the region.
    pub fn shape(&self) -> Shape {
        self.inner.arrays()[0].shape()
    }

    /// Number of elements of the region (per field).
    pub fn len(&self) -> usize {
        self.shape().len()
    }

    /// Always `false`: arrays have at least one element.
    pub fn is_empty(&self) -> bool {
        false
    }

    /// Writes `rhs` into the region: an array of its shape (`&expr`, or `expr` by value), or a
    /// single value for a region of one array; for a record, a record (`&rec` or `rec`). See
    /// [`Array::assign`] for how the write is recorded.
    #[track_caller]
    pub fn assign<R: AssignRhs<F>>(&mut self, rhs: R) -> Result<()> {
        rhs.assign_into(&self.inner, Location::caller())
    }

    fn poison_owner(&mut self, error: LumaError) {
        let error = Arc::new(error);
        self.owner.each_mut(&mut |a| a.poison(Arc::clone(&error)));
    }

    #[track_caller]
    fn compound_fields(&mut self, rhs: &F, op: FieldOp) {
        let location = Location::caller();
        let new = self.inner.binary(rhs, op);
        if let Err(error) = crate::fields::assign_fields(&self.inner, new, location) {
            self.poison_owner(error);
        }
    }
}

impl<'a, T: ScalarType> ViewMut<'a, Array<T>> {
    /// The [`Context`] the array lives on.
    pub fn context(&self) -> &Context {
        self.inner.context()
    }

    /// The region as a read-only array: a lazy read of the data as it is when the array is
    /// used, fusing into the expression using it like [`Array::slice`].
    #[track_caller]
    pub fn to_array(&self) -> Array<T> {
        let location = Location::caller();
        let shape = self.inner.shape();
        let result = dims32(shape).map_err(Failure::from).and_then(|dims| {
            self.inner.run_view(ViewMap::identity(dims), Edge::None, None, shape, "to_array", location)
        });
        Array::resolve(self.inner.context(), shape, result, "to_array", location)
    }

    /// The buffer the region lives in, its size in bytes and the strided view of it. A
    /// `writable` request fails with [`LumaError::ReadOnly`] on the state passed to an
    /// `iterate` step, and lets lazy operations that still read the buffer run first.
    pub(crate) fn ready_view(&self, writable: bool) -> std::result::Result<(Pin, u64, RawViewDesc), Failure> {
        self.inner.src()?;
        let Some(view) = self.inner.view() else {
            return Err(LumaError::InvalidShape("this view does not write through to an array".into()).into());
        };
        let ctx = self.inner.context();
        let storage = view.storage(ctx)?;
        if writable {
            if storage.is_alias() {
                return Err(LumaError::ReadOnly("the state passed to an iteration step").into());
            }
            storage.readers.flush();
            ctx.jit().note_write();
        }
        Ok((storage.pin_single()?, storage.byte_len(), RawViewDesc::from_map(&view.map)))
    }

    /// `region = region op rhs`.
    #[track_caller]
    fn compound(&mut self, rhs: Rhs<'_, T>, op: BinOp, name: &'static str) {
        let location = Location::caller();
        let new = self.inner.binary_at(rhs, op, false, name, location);
        if let Err(error) = self.inner.assign_in(new, location) {
            self.poison_owner(error);
        }
    }
}

/// What [`ViewMut::assign`] accepts for a region of `F`: a record (`F` or `&F`) and, for a
/// region of one array, a single element value or host [`Scalar`](crate::Scalar).
pub trait AssignRhs<F: Fields> {
    #[doc(hidden)]
    fn assign_into(self, target: &F, location: &'static Location<'static>) -> Result<()>;
}

impl<F: Fields> AssignRhs<F> for F {
    fn assign_into(self, target: &F, location: &'static Location<'static>) -> Result<()> {
        crate::fields::assign_fields(target, self, location)
    }
}

impl<F: Fields> AssignRhs<F> for &F {
    fn assign_into(self, target: &F, location: &'static Location<'static>) -> Result<()> {
        crate::fields::assign_fields(target, self, location)
    }
}

macro_rules! assign_rhs_value {
    ($t:ty) => {
        impl AssignRhs<Array<$t>> for $t {
            fn assign_into(self, target: &Array<$t>, location: &'static Location<'static>) -> Result<()> {
                target.assign_in(self, location)
            }
        }
    };
}

macro_rules! assign_rhs_host {
    ($t:ty) => {
        impl AssignRhs<Array<$t>> for &crate::scalar::Scalar<$t> {
            fn assign_into(self, target: &Array<$t>, location: &'static Location<'static>) -> Result<()> {
                target.assign_in(self, location)
            }
        }
    };
}

assign_rhs_value!(f32);
assign_rhs_value!(u32);
assign_rhs_value!(Df32);
assign_rhs_host!(f32);
assign_rhs_host!(u32);

macro_rules! view_fields_assign {
    ($Trait:ident, $method:ident, $op:expr) => {
        impl<F: Fields> $Trait<&F> for ViewMut<'_, F> {
            #[track_caller]
            fn $method(&mut self, rhs: &F) {
                self.compound_fields(rhs, $op)
            }
        }

        impl<F: Fields> $Trait<F> for ViewMut<'_, F> {
            #[track_caller]
            fn $method(&mut self, rhs: F) {
                self.compound_fields(&rhs, $op)
            }
        }
    };
}

view_fields_assign!(AddAssign, add_assign, FieldOp::Add);
view_fields_assign!(SubAssign, sub_assign, FieldOp::Sub);
view_fields_assign!(MulAssign, mul_assign, FieldOp::Mul);
view_fields_assign!(DivAssign, div_assign, FieldOp::Div);

macro_rules! view_value_assign {
    ($t:ty, $Trait:ident, $method:ident, $op:expr, $name:literal) => {
        impl $Trait<$t> for ViewMut<'_, Array<$t>> {
            #[track_caller]
            fn $method(&mut self, rhs: $t) {
                self.compound(Rhs::Scalar(rhs), $op, $name)
            }
        }
    };
}

macro_rules! view_host_assign {
    ($t:ty, $Trait:ident, $method:ident, $op:expr, $name:literal) => {
        impl $Trait<&crate::scalar::Scalar<$t>> for ViewMut<'_, Array<$t>> {
            #[track_caller]
            fn $method(&mut self, rhs: &crate::scalar::Scalar<$t>) {
                self.compound(Rhs::Scalar(rhs.get()), $op, $name)
            }
        }
    };
}

macro_rules! view_array_assign {
    ($Trait:ident, $method:ident, $op:expr, $name:literal) => {
        impl $Trait<&Array<u32>> for ViewMut<'_, Array<u32>> {
            #[track_caller]
            fn $method(&mut self, rhs: &Array<u32>) {
                self.compound(Rhs::Array(rhs), $op, $name)
            }
        }

        impl $Trait<Array<u32>> for ViewMut<'_, Array<u32>> {
            #[track_caller]
            fn $method(&mut self, rhs: Array<u32>) {
                self.compound(Rhs::Array(&rhs), $op, $name)
            }
        }
    };
}

macro_rules! view_arithmetic_values {
    ($t:ty) => {
        view_value_assign!($t, AddAssign, add_assign, BinOp::Add, "add_assign");
        view_value_assign!($t, SubAssign, sub_assign, BinOp::Sub, "sub_assign");
        view_value_assign!($t, MulAssign, mul_assign, BinOp::Mul, "mul_assign");
        view_value_assign!($t, DivAssign, div_assign, BinOp::Div, "div_assign");
    };
}

macro_rules! view_arithmetic_host {
    ($t:ty) => {
        view_host_assign!($t, AddAssign, add_assign, BinOp::Add, "add_assign");
        view_host_assign!($t, SubAssign, sub_assign, BinOp::Sub, "sub_assign");
        view_host_assign!($t, MulAssign, mul_assign, BinOp::Mul, "mul_assign");
        view_host_assign!($t, DivAssign, div_assign, BinOp::Div, "div_assign");
    };
}

view_arithmetic_values!(f32);
view_arithmetic_values!(u32);
view_arithmetic_values!(f64);
view_arithmetic_values!(Df32);
view_arithmetic_host!(f32);
view_arithmetic_host!(u32);

macro_rules! view_u32_assign {
    ($Trait:ident, $method:ident, $op:expr, $name:literal) => {
        view_value_assign!(u32, $Trait, $method, $op, $name);
        view_host_assign!(u32, $Trait, $method, $op, $name);
        view_array_assign!($Trait, $method, $op, $name);
    };
}

view_u32_assign!(RemAssign, rem_assign, BinOp::Rem, "rem_assign");
view_u32_assign!(BitAndAssign, bitand_assign, BinOp::BitAnd, "bitand_assign");
view_u32_assign!(BitOrAssign, bitor_assign, BinOp::BitOr, "bitor_assign");
view_u32_assign!(BitXorAssign, bitxor_assign, BinOp::BitXor, "bitxor_assign");
view_u32_assign!(ShlAssign, shl_assign, BinOp::Shl, "shl_assign");
view_u32_assign!(ShrAssign, shr_assign, BinOp::Shr, "shr_assign");

/// [`concat()`] of the arrays of a slice. An empty slice gives a poisoned array
/// ([`LumaError::InvalidShape`]) on the [`default_context`](crate::default_context).
#[track_caller]
pub fn concat_slice<T: ScalarType>(axis: usize, parts: &[Array<T>]) -> Array<T> {
    let location = Location::caller();
    let Some(first) = parts.first() else {
        let error = LumaError::InvalidShape("concat of no arrays".into());
        let error = LumaError::Op { op: "concat", location, source: Arc::new(error) };
        let ctx = crate::default_context().unwrap_or_else(|e| panic!("concat of no arrays: no default context: {e}"));
        return Array::poisoned(&ctx, Shape::new([1; 4]), Arc::new(error));
    };
    let refs: Vec<&Array<T>> = parts.iter().collect();
    let result = run_concat(axis, &refs, location);
    Array::resolve(first.context(), first.shape(), result, "concat", location)
}
