//! Operators and math functions on arrays, recorded lazily and fused by [`crate::jit`]. The
//! `run_*` functions return failures; the public methods turn them into poisoned arrays tagged
//! with the caller's location, or into `Err` for the `try_*` methods.

use std::ops::{
    Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Div, DivAssign, Mul, MulAssign,
    Neg, Not, Rem, RemAssign, Shl, ShlAssign, Shr, ShrAssign, Sub, SubAssign,
};
use std::panic::Location;
use std::sync::Arc;

use crate::array::{Array, Failure};
use crate::builtin::{BinOp, Builtin, Class, CmpOp, EpilogueStep, UnOp};
use crate::context::{Context, RawArg};
use crate::error::{LumaError, Result};
use crate::gpu_type::{Df32, GpuType, ScalarType};
use crate::jit::{self, ElemTy, Op, OpKind, Scalar, ShiftKind, Src};
use crate::kernel::KernelCore;
use crate::shape::{Offsets, Shape};

pub(crate) type Outcome<T> = std::result::Result<T, Failure>;

/// The right-hand side of a binary operation.
#[doc(hidden)]
pub enum Rhs<'a, T: ScalarType> {
    Array(&'a Array<T>),
    Scalar(T),
}

/// Seals [`Operand`]: only the types below are right-hand sides.
pub(crate) mod operand_sealed {
    pub trait Sealed {}
}

/// A right-hand side accepted by the `try_*` methods, comparisons and `minimum`/`maximum`:
/// an `&Array<T>` or `Array<T>` or a single `T` applied to every element. An array broadcasts
/// along its extent-1 axes ([`Array::broadcast_to`]) to the shape the operation works at.
pub trait Operand<T: ScalarType>: operand_sealed::Sealed {
    #[doc(hidden)]
    fn rhs(&self) -> Rhs<'_, T>;
}

macro_rules! impl_value_operand {
    ($t:ty) => {
        impl operand_sealed::Sealed for $t {}

        impl Operand<$t> for $t {
            fn rhs(&self) -> Rhs<'_, $t> {
                Rhs::Scalar(*self)
            }
        }
    };
}

macro_rules! impl_scalar_operand {
    ($t:ty) => {
        impl operand_sealed::Sealed for &crate::scalar::Scalar<$t> {}

        impl Operand<$t> for &crate::scalar::Scalar<$t> {
            fn rhs(&self) -> Rhs<'_, $t> {
                Rhs::Scalar(self.get())
            }
        }
    };
}

impl<T: ScalarType> operand_sealed::Sealed for &Array<T> {}
impl<T: ScalarType> operand_sealed::Sealed for Array<T> {}

impl<T: ScalarType> Operand<T> for &Array<T> {
    fn rhs(&self) -> Rhs<'_, T> {
        Rhs::Array(*self)
    }
}

impl<T: ScalarType> Operand<T> for Array<T> {
    fn rhs(&self) -> Rhs<'_, T> {
        Rhs::Array(self)
    }
}

impl_value_operand!(f32);
impl_value_operand!(u32);
impl_value_operand!(Df32);
// There is deliberately no `Operand<f64> for f64`: a second float type among the scalar
// operands would make `select(&mask, 1.0, 0.0)` and every other call that only has float
// literals to go by infer `f64` instead of `f32`. The operators (`&x * 2.0`) take `f64`
// scalars; comparisons, `minimum`/`maximum` and `select` take `f64` arrays.
impl_scalar_operand!(f32);
impl_scalar_operand!(u32);

/// Splits `count` workgroups into a 2D grid within the per-dimension limit; shaders recombine
/// them as `x + y * num_workgroups.x`.
pub(crate) fn fold_workgroups(ctx: &Context, count: u32) -> Result<[u32; 3]> {
    let max = ctx.max_workgroups_per_dimension();
    let [x, y, _] = fold_grid(count, max);
    if y > max {
        return Err(LumaError::KernelGridTooLarge { workgroups: [x, y, 1], max });
    }
    Ok([x, y, 1])
}

/// The rows of the grid are as even as possible: filling the first dimension up to `max` would
/// launch a nearly empty second row for 65536 groups with the usual limit of 65535, and the
/// groups past the end that exit at once cost a measurable share of a bandwidth-bound kernel
/// (about 35 us of 340 for 2^24 elements). Fewer than `y` groups at the end are idle.
fn fold_grid(count: u32, max: u32) -> [u32; 3] {
    let groups = count.max(1);
    let y = groups.div_ceil(max.max(1));
    [groups.div_ceil(y), y, 1]
}

/// The workgroups of a launch with one invocation per element for `n` elements.
pub(crate) fn workgroups_1d(ctx: &Context, kernel: &KernelCore, n: u32) -> Result<[u32; 3]> {
    fold_workgroups(ctx, n.div_ceil(kernel.workgroup[0]))
}

/// Launches a kernel with one invocation per element for `n` elements (`luma.size.x = n`).
pub(crate) fn launch_1d(
    ctx: &Context,
    kernel: &KernelCore,
    n: u32,
    args: &[RawArg<'_>],
) -> Result<()> {
    ctx.launch(kernel, workgroups_1d(ctx, kernel, n)?, [n, 1, 1], args)
}

pub(crate) fn len_u32(len: usize) -> Result<u32> {
    // Array lengths were validated to fit u32 when the array was created.
    u32::try_from(len).map_err(|_| LumaError::InvalidShape(format!("{len} elements exceed u32")))
}

/// Records the lazy operation `op` on `args` as an array of `shape`. Nothing runs yet.
pub(crate) fn lazy<O: GpuType>(
    ctx: &Context,
    shape: Shape,
    op: Op,
    args: [Src; 3],
    name: &'static str,
    location: &'static Location<'static>,
) -> Outcome<Array<O>> {
    let node = jit::create(ctx, shape.len(), op, args, name, location)?;
    Ok(Array::from_node(ctx, shape, node))
}

/// The shape two operands of shapes `lhs` and `rhs` broadcast to: per axis the common extent, or
/// the other one where one side has extent 1. `None` if some axis has two different extents
/// other than 1.
pub(crate) fn broadcast_shape(lhs: Shape, rhs: Shape) -> Option<Shape> {
    let (l, r) = (lhs.dims(), rhs.dims());
    let mut out = l;
    for k in 0..4 {
        out[k] = match (l[k], r[k]) {
            (a, b) if a == b => a,
            (1, b) => b,
            (a, 1) => a,
            _ => return None,
        };
    }
    Some(Shape::new(out))
}

impl<T: ScalarType> Array<T> {
    // -- Lazy operations --

    /// `f(self)` for a unary operation.
    fn run_unary(
        &self,
        op: UnOp,
        location: &'static Location<'static>,
    ) -> Outcome<Array<T>> {
        // A reduction that has not run takes the operation into its last pass.
        if let Some(fused) = self.fuse_step::<T>(EpilogueStep::Unary(op), 0) {
            return Ok(fused);
        }
        let operand = self.src()?;
        let op_kind = Op::new(OpKind::Unary(op), ElemTy::of::<T>()?);
        lazy(
            self.context(),
            self.shape(),
            op_kind,
            [operand, Src::unused(), Src::unused()],
            op.name(),
            location,
        )
    }

    /// The test `class` of every element, as a `u32` mask (see [`Class`]).
    fn classify(&self, class: Class, location: &'static Location<'static>) -> Array<u32> {
        let result = self.src().and_then(|operand| {
            let op = Op::new(OpKind::Classify(class), ElemTy::of::<T>()?);
            lazy(self.context(), self.shape(), op, [operand, Src::unused(), Src::unused()], class.name(), location)
        });
        Array::resolve(self.context(), self.shape(), result, class.name(), location)
    }

    /// `-self` reporting failures at `location`.
    pub(crate) fn neg_at(&self, location: &'static Location<'static>) -> Array<T> {
        let result = self.run_unary(UnOp::Neg, location);
        Array::resolve(self.context(), self.shape(), result, "neg", location)
    }

    /// A lazy copy of the array's current contents (not a shared handle), reporting failures at
    /// the caller.
    #[track_caller]
    pub(crate) fn copied(&self) -> Array<T> {
        let location = Location::caller();
        let result = self.run_unary(UnOp::Id, location);
        Array::resolve(self.context(), self.shape(), result, "copy", location)
    }

    /// `self op rhs` (or `rhs op self` if `scalar_left`, which only exists for scalars),
    /// producing elements of type `O`.
    fn run_binary<O: GpuType>(
        &self,
        rhs: Rhs<'_, T>,
        scalar_left: bool,
        kind: OpKind,
        name: &'static str,
        location: &'static Location<'static>,
    ) -> Outcome<Array<O>> {
        let ctx = self.context();
        let this = self.src()?;
        let (this, other, shape) = match rhs {
            Rhs::Array(other) => {
                let operand = other.src()?;
                if !ctx.same_as(other.context()) {
                    return Err(LumaError::ContextMismatch.into());
                }
                if self.shape() == other.shape() {
                    (this, operand, self.shape())
                } else {
                    let mismatch = || LumaError::ShapeMismatch { lhs: self.shape().dims(), rhs: other.shape().dims() };
                    let out = broadcast_shape(self.shape(), other.shape()).ok_or_else(mismatch)?;
                    let this = if self.shape() == out { this } else { self.broadcast_axis(out, location)? };
                    let operand = if other.shape() == out { operand } else { other.broadcast_axis(out, location)? };
                    (this, operand, out)
                }
            }
            Rhs::Scalar(value) => (this, Src::Scalar(Scalar::of(value)?), self.shape()),
        };
        let (left, right) = if scalar_left { (other, this) } else { (this, other) };
        lazy(
            ctx,
            shape,
            Op::new(kind, ElemTy::of::<T>()?),
            [left, right, Src::unused()],
            name,
            location,
        )
    }

    /// `self = self op rhs` over the array's buffer, which other arrays or pending operations
    /// share, as a fused kernel that writes it in place.
    fn run_in_place(
        &self,
        rhs: Rhs<'_, T>,
        op: BinOp,
        name: &'static str,
        location: &'static Location<'static>,
    ) -> Outcome<()> {
        let ctx = self.context();
        let storage = Arc::clone(self.storage()?);
        let other = match rhs {
            Rhs::Array(other) => {
                if !ctx.same_as(other.context()) {
                    return Err(LumaError::ContextMismatch.into());
                }
                other.broadcast_operand(self.shape(), location)?
            }
            Rhs::Scalar(value) => Src::Scalar(Scalar::of(value)?),
        };
        // Lazy operations recorded earlier (the right-hand side's among them) still read the old
        // contents: evaluate them before the buffer changes. Afterwards no pending expression
        // reads it, so the new node sees it as a plain buffer.
        storage.readers.flush();
        ctx.jit().note_write();
        let node = jit::create(
            ctx,
            self.len(),
            Op::new(OpKind::Binary(op), ElemTy::of::<T>()?),
            [Src::Buffer(Arc::clone(&storage)), other, Src::unused()],
            name,
            location,
        )?;
        jit::evaluate_into(&node, &storage)
    }

    // -- Result plumbing --

    #[track_caller]
    pub(crate) fn binary(
        &self,
        rhs: Rhs<'_, T>,
        op: BinOp,
        scalar_left: bool,
        name: &'static str,
    ) -> Array<T> {
        self.binary_at(rhs, op, scalar_left, name, Location::caller())
    }

    /// [`Array::binary`] reporting failures at `location`.
    pub(crate) fn binary_at(
        &self,
        rhs: Rhs<'_, T>,
        op: BinOp,
        scalar_left: bool,
        name: &'static str,
        location: &'static Location<'static>,
    ) -> Array<T> {
        // A shift or rotation by a constant is an operation of its own: its amount is in the kernel.
        if let Rhs::Scalar(value) = &rhs
            && !scalar_left
            && let Ok(scalar) = Scalar::of(*value)
            && scalar.ty == ElemTy::U32
            && let Some(by) = UnOp::by_constant(op, scalar.bits as u32)
        {
            let result = self.run_unary(by, location);
            return Array::resolve(self.context(), self.shape(), result, name, location);
        }
        if let Rhs::Scalar(value) = &rhs
            && let Ok(scalar) = Scalar::of(*value)
            && let Some(fused) = self.fuse_step::<T>(EpilogueStep::Binary { op, scalar_left }, scalar.words()[0])
        {
            return fused;
        }
        let result = self.run_binary(rhs, scalar_left, OpKind::Binary(op), name, location);
        Array::resolve(self.context(), self.shape(), result, name, location)
    }

    #[track_caller]
    fn compare(&self, rhs: Rhs<'_, T>, op: CmpOp, name: &'static str) -> Array<u32> {
        self.compare_at(rhs, op, name, Location::caller())
    }

    /// [`Array::compare`] reporting failures at `location`.
    pub(crate) fn compare_at(
        &self,
        rhs: Rhs<'_, T>,
        op: CmpOp,
        name: &'static str,
        location: &'static Location<'static>,
    ) -> Array<u32> {
        let result = self.run_binary(rhs, false, OpKind::Compare(op), name, location);
        Array::resolve(self.context(), self.shape(), result, name, location)
    }

    /// Like [`Array::binary`], but fails immediately with the plain error, and evaluates the
    /// result right away so that failures of the evaluation itself (out of memory, ...) are
    /// reported here too.
    #[track_caller]
    fn try_binary(&self, rhs: Rhs<'_, T>, op: BinOp, name: &'static str) -> Result<Array<T>> {
        let location = Location::caller();
        let out = self
            .run_binary(rhs, false, OpKind::Binary(op), name, location)
            .map_err(Failure::into_error)?;
        out.storage().map_err(|failure| failure.into_error().root().clone())?;
        Ok(out)
    }

    #[track_caller]
    pub(crate) fn assign_op(&mut self, rhs: Rhs<'_, T>, op: BinOp, name: &'static str) {
        let location = Location::caller();
        let result = if self.is_read_view() {
            self.src().and_then(|_| -> Outcome<()> {
                Err(LumaError::InvalidShape("cannot mutate a read-only view; use slice_mut, flip_mut or transpose_mut".into()).into())
            })
        } else if (self.pending().is_some() && !self.expression_shared()) || self.sole_owner() {
            // The target is an expression without a buffer, or nothing else can observe its
            // buffer changing: it simply becomes the expression `self op rhs`, stays fusable, and
            // reuses its buffer when it is evaluated.
            self.run_binary(rhs, false, OpKind::Binary(op), name, location).and_then(|new: Array<T>| {
                // The target never grows to the right-hand side's shape.
                if new.shape() != self.shape() {
                    return Err(LumaError::ShapeMismatch { lhs: self.shape().dims(), rhs: new.shape().dims() }.into());
                }
                *self = new;
                Ok(())
            })
        } else {
            self.run_in_place(rhs, op, name, location)
        };
        match result {
            Ok(()) => {}
            // Already poisoned (by itself or by the right-hand side): keep that error.
            Err(Failure::Inherited(error)) => {
                if self.error().is_none() {
                    self.poison(error);
                }
            }
            Err(Failure::New(error)) => {
                self.poison(Arc::new(LumaError::Op { op: name, location, source: Arc::new(error) }));
            }
        }
    }

    // -- Eager arithmetic --

    /// `self + rhs`, failing immediately instead of returning a poisoned array.
    #[track_caller]
    pub fn try_add(&self, rhs: impl Operand<T>) -> Result<Array<T>> {
        self.try_binary(rhs.rhs(), BinOp::Add, "add")
    }

    /// `self - rhs`, failing immediately instead of returning a poisoned array.
    #[track_caller]
    pub fn try_sub(&self, rhs: impl Operand<T>) -> Result<Array<T>> {
        self.try_binary(rhs.rhs(), BinOp::Sub, "sub")
    }

    /// `self * rhs`, failing immediately instead of returning a poisoned array.
    #[track_caller]
    pub fn try_mul(&self, rhs: impl Operand<T>) -> Result<Array<T>> {
        self.try_binary(rhs.rhs(), BinOp::Mul, "mul")
    }

    /// `self / rhs`, failing immediately instead of returning a poisoned array.
    #[track_caller]
    pub fn try_div(&self, rhs: impl Operand<T>) -> Result<Array<T>> {
        self.try_binary(rhs.rhs(), BinOp::Div, "div")
    }

    // -- Elementwise min / max --

    /// Elementwise minimum with an array of the same shape or a scalar.
    ///
    /// (Named `minimum` because [`Array::min`] is the reduction over the whole array.)
    #[track_caller]
    pub fn minimum(&self, rhs: impl Operand<T>) -> Array<T> {
        self.binary(rhs.rhs(), BinOp::Min, false, "minimum")
    }

    /// Elementwise maximum with an array of the same shape or a scalar.
    #[track_caller]
    pub fn maximum(&self, rhs: impl Operand<T>) -> Array<T> {
        self.binary(rhs.rhs(), BinOp::Max, false, "maximum")
    }

    // -- Comparisons: masks of 1 (true) and 0 (false) --

    /// `self < rhs` as a mask of ones and zeros.
    #[track_caller]
    pub fn lt(&self, rhs: impl Operand<T>) -> Array<u32> {
        self.compare(rhs.rhs(), CmpOp::Lt, "lt")
    }

    /// `self <= rhs` as a mask of ones and zeros.
    #[track_caller]
    pub fn le(&self, rhs: impl Operand<T>) -> Array<u32> {
        self.compare(rhs.rhs(), CmpOp::Le, "le")
    }

    /// `self > rhs` as a mask of ones and zeros.
    #[track_caller]
    pub fn gt(&self, rhs: impl Operand<T>) -> Array<u32> {
        self.compare(rhs.rhs(), CmpOp::Gt, "gt")
    }

    /// `self >= rhs` as a mask of ones and zeros.
    #[track_caller]
    pub fn ge(&self, rhs: impl Operand<T>) -> Array<u32> {
        self.compare(rhs.rhs(), CmpOp::Ge, "ge")
    }

    /// `self == rhs` as a mask of ones and zeros.
    #[track_caller]
    pub fn eq(&self, rhs: impl Operand<T>) -> Array<u32> {
        self.compare(rhs.rhs(), CmpOp::Eq, "eq")
    }

    /// `self != rhs` as a mask of ones and zeros.
    #[track_caller]
    pub fn ne(&self, rhs: impl Operand<T>) -> Array<u32> {
        self.compare(rhs.rhs(), CmpOp::Ne, "ne")
    }

    // -- Creation --

    /// An array of the given shape with every element set to `value`, on the default context.
    pub fn full(shape: impl Into<crate::Shape>, value: T) -> Result<Array<T>> {
        Self::full_on(&crate::default_context()?, shape, value)
    }

    /// [`Array::full`] on an explicit context.
    pub fn full_on(ctx: &Context, shape: impl Into<crate::Shape>, value: T) -> Result<Array<T>> {
        let shape = shape.into();
        if T::wgsl_type() == "f64" {
            // A uniform cannot hold an `f64`: upload the values instead.
            let len = shape.validate()?;
            return Array::from_slice_on(ctx, &vec![value; len], shape);
        }
        let out = Array::<T>::alloc(ctx, shape)?;
        let storage = out.storage().map_err(Failure::into_error)?;
        let core = ctx.builtin(Builtin::Fill, T::wgsl_type())?;
        for j in 0..storage.segments() {
            let n = storage.range(j).len();
            let pin = storage.pin(j)?;
            launch_1d(
                ctx,
                &core,
                len_u32(n)?,
                &[
                    RawArg::write(pin.buffer(), n as u64 * std::mem::size_of::<T>() as u64),
                    RawArg::Uniform(bytemuck::bytes_of(&value)),
                ],
            )?;
        }
        Ok(out)
    }

    // -- Shifts --

    /// Circular shift like `af::shift`: element `i` moves to `i + offset` in every dimension,
    /// wrapping around (`[1, 2, 3, 4].shift(1)` is `[4, 1, 2, 3]`; negative offsets shift the
    /// other way). Offsets can be larger than a dimension.
    ///
    /// The shift itself is lazy: it fuses into the expression that uses it as a shifted read,
    /// so a stencil like `u.shift(1) + u.shift(-1) - &u * 2.0` is one kernel reading `u` once. A
    /// small lazy expression is not even evaluated: the kernel computes it at the shifted
    /// element instead (`(&a * &b).shift(1)` reads `a` and `b` shifted), staged in workgroup
    /// memory for small shifts; a larger one is evaluated first.
    #[track_caller]
    pub fn shift(&self, offsets: impl Into<Offsets>) -> Array<T> {
        self.shift_at(offsets.into(), Location::caller())
    }

    /// [`Array::shift`] reporting failures at `location`.
    pub(crate) fn shift_at(&self, offsets: Offsets, location: &'static Location<'static>) -> Array<T> {
        let result = self.run_shift(offsets, location);
        Array::resolve(self.context(), self.shape(), result, "shift", location)
    }

    fn run_shift(&self, offsets: Offsets, location: &'static Location<'static>) -> Outcome<Array<T>> {
        let dims = self.shape().dims();
        let operand = match self.pending() {
            Some(node) if jit::recompute_shifted(node, dims, offsets.0) => Src::Node(Arc::clone(node)),
            _ => Src::Buffer(Arc::clone(self.storage_alongside()?)),
        };
        let mut dims32 = [1u32; 4];
        let mut shifts = [0u32; 4];
        for axis in 0..4 {
            let dim = u32::try_from(dims[axis]).map_err(|_| LumaError::DimensionTooLarge { shape: dims })?;
            // Valid arrays have dim > 0; the remainder is in 0..dim and so fits u32.
            let shift = (offsets.0[axis] as i64).rem_euclid(i64::from(dim.max(1)));
            dims32[axis] = dim;
            shifts[axis] = u32::try_from(shift).map_err(|_| LumaError::DimensionTooLarge { shape: dims })?;
        }
        // A shift of the outermost non-trivial dimension only is a shift of the flat array.
        let top = shifts.iter().rposition(|&s| s != 0).unwrap_or(0);
        let linear = dims32[top + 1..].iter().all(|&d| d == 1)
            && shifts[..top].iter().all(|&s| s == 0);
        let (kind, amount, total) = if linear {
            let stride: u32 = dims32[..top].iter().product();
            (
                ShiftKind::Linear,
                Src::Scalar(Scalar::u32(shifts[top] * stride)),
                Src::Scalar(Scalar::u32(len_u32(self.len())?)),
            )
        } else {
            (ShiftKind::Dims { dims: dims32, shifts }, Src::unused(), Src::unused())
        };
        lazy(
            self.context(),
            self.shape(),
            Op::new(OpKind::Shift(kind), ElemTy::of::<T>()?),
            [operand, amount, total],
            "shift",
            location,
        )
    }
}

/// Evaluates several arrays without waiting, like [`Array::eval`] on each, but computes lazy
/// arrays of the same length together: one fused kernel with an output per array (as many as
/// the device's bindings allow), so inputs they share are read once, and an array that another
/// one uses is computed once. `let (call, put) = ...; eval_all(&[&call, &put])?` prices both
/// options in one pass over the market data. Postponed while the GPU is busy, like
/// [`Array::eval`]. A loop of `x = step(&x); eval_all(&[&x])?` over `f32` arrays is recognized
/// while evaluation is postponed: the steps are not chained into ever longer expressions but run
/// together when a result is needed, in one kernel that keeps the values in registers, or time
/// blocked for a stencil.
pub fn eval_all<T: GpuType>(arrays: &[&Array<T>]) -> Result<()> {
    let Some(first) = arrays.first() else { return Ok(()) };
    let ctx = first.context();
    if arrays.iter().any(|array| !ctx.same_as(array.context())) {
        return Err(LumaError::ContextMismatch);
    }
    for array in arrays {
        // A reduction is recorded when it is requested, whatever the postponement: the arrays asked
        // for before it are computed first, so that what it shares with them (the residual of a
        // step is summed, and is also what the update of the state reads) is computed once, by
        // their kernel, and not by the reduction alone with every intermediate evaluated on its
        // own.
        if array.reduced().is_some_and(|reduced| reduced.is_unforced()) {
            ctx.jit().evaluate_requested();
        }
        array.request()?;
    }
    // Consecutive calls that advance the same f32 arrays through the same step are a loop.
    let shape = first.shape();
    let nodes: Option<Vec<_>> = (std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>()
        && arrays.iter().all(|array| array.shape() == shape))
    .then(|| arrays.iter().map(|array| array.pending().cloned()).collect())
    .flatten();
    ctx.jit().step_boundary(nodes.as_deref().unwrap_or(&[]), shape);
    ctx.submit_requested()?;
    arrays.iter().try_for_each(|array| array.check())
}

/// Picks `a[i]` where `mask[i] != 0` and `b[i]` elsewhere. Either branch is an `&Array<T>` or a
/// single `T` applied to every element; a scalar is a uniform of the fused kernel, so no constant
/// array is materialized or read from memory:
///
/// ```ignore
/// let ones = select(&x.gt(0.0), 1.0, 0.0);
/// let clipped = select(&x.gt(0.0), &x, 0.0);
/// ```
///
/// The result takes the mask's shape and context. An array branch has the mask's shape or
/// extent 1 on the axes where it differs (a `[n, 1]` branch under an `[n, m]` mask is read at
/// every column, see [`Array::broadcast_to`]). A mismatch poisons the result with
/// `ShapeMismatch` (reported as `lhs` = the mask's shape, `rhs` = the branch's) or
/// `ContextMismatch`.
#[track_caller]
pub fn select<T: ScalarType>(mask: &Array<u32>, a: impl Operand<T>, b: impl Operand<T>) -> Array<T> {
    select_at(mask, a, b, Location::caller())
}

/// [`select`] reporting failures at `location`.
pub(crate) fn select_at<T: ScalarType>(
    mask: &Array<u32>,
    a: impl Operand<T>,
    b: impl Operand<T>,
    location: &'static Location<'static>,
) -> Array<T> {
    let result = run_select(mask, a.rhs(), b.rhs(), location);
    Array::resolve(mask.context(), mask.shape(), result, "select", location)
}

pub(crate) fn run_select<T: ScalarType>(
    mask: &Array<u32>,
    a: Rhs<'_, T>,
    b: Rhs<'_, T>,
    location: &'static Location<'static>,
) -> Outcome<Array<T>> {
    let ctx = mask.context();
    let mask_src = mask.src()?;
    let branch = |rhs: Rhs<'_, T>| -> Outcome<Src> {
        match rhs {
            Rhs::Array(other) => {
                if !ctx.same_as(other.context()) {
                    return Err(LumaError::ContextMismatch.into());
                }
                other.broadcast_operand(mask.shape(), location)
            }
            Rhs::Scalar(value) => Ok(Src::Scalar(Scalar::of(value)?)),
        }
    };
    let a_src = branch(a)?;
    let b_src = branch(b)?;
    lazy(
        ctx,
        mask.shape(),
        Op::new(OpKind::Select, ElemTy::of::<T>()?),
        [mask_src, a_src, b_src],
        "select",
        location,
    )
}

// -- Math functions on f32 arrays --

macro_rules! unary_methods {
    ($($(#[$doc:meta])* $name:ident => $op:expr;)*) => {
        impl Array<f32> {
            $(
                $(#[$doc])*
                #[track_caller]
                pub fn $name(&self) -> Array<f32> {
                    let location = Location::caller();
                    let op: UnOp = $op;
                    let result = self.run_unary(op, location);
                    Array::resolve(self.context(), self.shape(), result, op.name(), location)
                }
            )*
        }
    };
}

unary_methods! {
    /// Elementwise square root.
    sqrt => UnOp::Sqrt;
    /// Elementwise `e^x`.
    exp => UnOp::Exp;
    /// Elementwise natural logarithm.
    ln => UnOp::Ln;
    /// Elementwise absolute value.
    abs => UnOp::Abs;
    /// Elementwise sine (radians).
    sin => UnOp::Sin;
    /// Elementwise cosine (radians).
    cos => UnOp::Cos;
    /// Elementwise tangent (radians).
    tan => UnOp::Tan;
    /// Elementwise hyperbolic tangent.
    tanh => UnOp::Tanh;
    /// Elementwise sign: `-1`, `0` or `1`.
    sign => UnOp::Sign;
    /// Elementwise largest integer not above `x`.
    floor => UnOp::Floor;
    /// Elementwise smallest integer not below `x`.
    ceil => UnOp::Ceil;
}

macro_rules! impl_float_classes {
    ($($ty:ty),*) => {$(
        impl Array<$ty> {
            /// Elementwise: `1` where the element is neither infinite nor `NaN`, else `0`; a
            /// lazy `u32` array that fuses like any elementwise operation. The test reads the
            /// bits of the exponent, so it does not depend on how the device compares `NaN`
            /// (a compiler may assume no `NaN` exists and fold `x != x`). For [`Df32`] it is the
            /// high word that counts.
            #[track_caller]
            pub fn is_finite(&self) -> Array<u32> {
                self.classify(Class::Finite, Location::caller())
            }

            /// Elementwise: `1` where the element is `NaN`, else `0` (see [`Array::is_finite`]).
            #[track_caller]
            pub fn is_nan(&self) -> Array<u32> {
                self.classify(Class::Nan, Location::caller())
            }

            /// Elementwise: `1` where the element is positive or negative infinity, else `0`
            /// (see [`Array::is_finite`]).
            #[track_caller]
            pub fn is_inf(&self) -> Array<u32> {
                self.classify(Class::Inf, Location::caller())
            }
        }
    )*};
}

impl_float_classes!(f32, f64, Df32);

impl Array<f32> {
    /// Elementwise `atan2(self, x)`: the angle of the point `(x, self)`, `self` being `y`.
    #[track_caller]
    pub fn atan2(&self, x: impl Operand<f32>) -> Array<f32> {
        self.binary(x.rhs(), BinOp::Atan2, false, "atan2")
    }

    /// Elementwise `sqrt(self^2 + other^2)` without intermediate overflow.
    #[track_caller]
    pub fn hypot(&self, other: impl Operand<f32>) -> Array<f32> {
        self.binary(other.rhs(), BinOp::Hypot, false, "hypot")
    }

    /// Elementwise magnitude of `self` with the sign of `sign`.
    #[track_caller]
    pub fn copysign(&self, sign: impl Operand<f32>) -> Array<f32> {
        self.binary(sign.rhs(), BinOp::Copysign, false, "copysign")
    }

    /// Elementwise `x` raised to `exponent`.
    #[track_caller]
    pub fn powf(&self, exponent: f32) -> Array<f32> {
        let location = Location::caller();
        let result = self.run_scalars(OpKind::Powf, [Some(Scalar::f32(exponent)), None], "powf", location);
        Array::resolve(self.context(), self.shape(), result, "powf", location)
    }

    /// Elementwise clamp into `lo..=hi`.
    #[track_caller]
    pub fn clamp(&self, lo: f32, hi: f32) -> Array<f32> {
        let location = Location::caller();
        let result = self.run_scalars(
            OpKind::Clamp,
            [Some(Scalar::f32(lo)), Some(Scalar::f32(hi))],
            "clamp",
            location,
        );
        Array::resolve(self.context(), self.shape(), result, "clamp", location)
    }

    /// `f(self, scalars...)` for an operation whose other operands are scalars.
    fn run_scalars(
        &self,
        kind: OpKind,
        scalars: [Option<Scalar>; 2],
        name: &'static str,
        location: &'static Location<'static>,
    ) -> Outcome<Array<f32>> {
        let operand = self.src()?;
        let [first, second] = scalars.map(|s| s.map_or_else(Src::unused, Src::Scalar));
        lazy(self.context(), self.shape(), Op::new(kind, ElemTy::F32), [operand, first, second], name, location)
    }
}

impl Neg for &Array<f32> {
    type Output = Array<f32>;

    #[track_caller]
    fn neg(self) -> Array<f32> {
        self.neg_at(Location::caller())
    }
}

impl Neg for Array<f32> {
    type Output = Array<f32>;

    #[track_caller]
    fn neg(self) -> Array<f32> {
        -&self
    }
}

// -- Math functions on the 8-byte types --

macro_rules! wide_unary_methods {
    ($t:ty; $($(#[$doc:meta])* $name:ident => $op:expr;)*) => {
        impl Array<$t> {
            $(
                $(#[$doc])*
                #[track_caller]
                pub fn $name(&self) -> Array<$t> {
                    let location = Location::caller();
                    let op: UnOp = $op;
                    let result = self.run_unary(op, location);
                    Array::resolve(self.context(), self.shape(), result, op.name(), location)
                }
            )*
        }

        impl Neg for &Array<$t> {
            type Output = Array<$t>;

            #[track_caller]
            fn neg(self) -> Array<$t> {
                self.neg_at(Location::caller())
            }
        }

        impl Neg for Array<$t> {
            type Output = Array<$t>;

            #[track_caller]
            fn neg(self) -> Array<$t> {
                -&self
            }
        }
    };
}

wide_unary_methods! { f64;
    /// Elementwise square root.
    sqrt => UnOp::Sqrt;
    /// Elementwise absolute value.
    abs => UnOp::Abs;
    /// Elementwise sign: `-1`, `0` or `1`.
    sign => UnOp::Sign;
    /// Elementwise largest integer not above `x`.
    floor => UnOp::Floor;
    /// Elementwise smallest integer not below `x`.
    ceil => UnOp::Ceil;
}

wide_unary_methods! { Df32;
    /// Elementwise square root (accurate to about 44 bits).
    sqrt => UnOp::Sqrt;
    /// Elementwise absolute value.
    abs => UnOp::Abs;
}

// -- Operators --

macro_rules! impl_binary_operator {
    ($t:ty, $Trait:ident, $method:ident, $op:expr, $name:literal) => {
        impl $Trait<&Array<$t>> for &Array<$t> {
            type Output = Array<$t>;
            #[track_caller]
            fn $method(self, rhs: &Array<$t>) -> Array<$t> {
                self.binary(Rhs::Array(rhs), $op, false, $name)
            }
        }

        impl $Trait<Array<$t>> for Array<$t> {
            type Output = Array<$t>;
            #[track_caller]
            fn $method(self, rhs: Array<$t>) -> Array<$t> {
                self.binary(Rhs::Array(&rhs), $op, false, $name)
            }
        }

        impl $Trait<Array<$t>> for &Array<$t> {
            type Output = Array<$t>;
            #[track_caller]
            fn $method(self, rhs: Array<$t>) -> Array<$t> {
                self.binary(Rhs::Array(&rhs), $op, false, $name)
            }
        }

        impl $Trait<&Array<$t>> for Array<$t> {
            type Output = Array<$t>;
            #[track_caller]
            fn $method(self, rhs: &Array<$t>) -> Array<$t> {
                self.binary(Rhs::Array(rhs), $op, false, $name)
            }
        }

        impl $Trait<$t> for &Array<$t> {
            type Output = Array<$t>;
            #[track_caller]
            fn $method(self, rhs: $t) -> Array<$t> {
                self.binary(Rhs::Scalar(rhs), $op, false, $name)
            }
        }

        impl $Trait<$t> for Array<$t> {
            type Output = Array<$t>;
            #[track_caller]
            fn $method(self, rhs: $t) -> Array<$t> {
                self.binary(Rhs::Scalar(rhs), $op, false, $name)
            }
        }

        impl $Trait<&Array<$t>> for $t {
            type Output = Array<$t>;
            #[track_caller]
            fn $method(self, rhs: &Array<$t>) -> Array<$t> {
                rhs.binary(Rhs::Scalar(self), $op, true, $name)
            }
        }

        impl $Trait<Array<$t>> for $t {
            type Output = Array<$t>;
            #[track_caller]
            fn $method(self, rhs: Array<$t>) -> Array<$t> {
                rhs.binary(Rhs::Scalar(self), $op, true, $name)
            }
        }
    };
}

/// The operators with a shared host-side [`crate::Scalar`] (32-bit types only).
macro_rules! impl_binary_scalar {
    ($t:ty, $Trait:ident, $method:ident, $op:expr, $name:literal) => {
        impl $Trait<&crate::scalar::Scalar<$t>> for &Array<$t> {
            type Output = Array<$t>;
            #[track_caller]
            fn $method(self, rhs: &crate::scalar::Scalar<$t>) -> Array<$t> {
                self.binary(Rhs::Scalar(rhs.get()), $op, false, $name)
            }
        }

        impl $Trait<&crate::scalar::Scalar<$t>> for Array<$t> {
            type Output = Array<$t>;
            #[track_caller]
            fn $method(self, rhs: &crate::scalar::Scalar<$t>) -> Array<$t> {
                self.binary(Rhs::Scalar(rhs.get()), $op, false, $name)
            }
        }

        impl $Trait<&Array<$t>> for &crate::scalar::Scalar<$t> {
            type Output = Array<$t>;
            #[track_caller]
            fn $method(self, rhs: &Array<$t>) -> Array<$t> {
                rhs.binary(Rhs::Scalar(self.get()), $op, true, $name)
            }
        }

        impl $Trait<Array<$t>> for &crate::scalar::Scalar<$t> {
            type Output = Array<$t>;
            #[track_caller]
            fn $method(self, rhs: Array<$t>) -> Array<$t> {
                rhs.binary(Rhs::Scalar(self.get()), $op, true, $name)
            }
        }
    };
}

macro_rules! impl_assign_operator {
    ($t:ty, $Trait:ident, $method:ident, $op:expr, $name:literal) => {
        impl $Trait<&Array<$t>> for Array<$t> {
            #[track_caller]
            fn $method(&mut self, rhs: &Array<$t>) {
                self.assign_op(Rhs::Array(rhs), $op, $name)
            }
        }

        impl $Trait<$t> for Array<$t> {
            #[track_caller]
            fn $method(&mut self, rhs: $t) {
                self.assign_op(Rhs::Scalar(rhs), $op, $name)
            }
        }
    };
}

macro_rules! impl_assign_scalar {
    ($t:ty, $Trait:ident, $method:ident, $op:expr, $name:literal) => {
        impl $Trait<&crate::scalar::Scalar<$t>> for Array<$t> {
            #[track_caller]
            fn $method(&mut self, rhs: &crate::scalar::Scalar<$t>) {
                self.assign_op(Rhs::Scalar(rhs.get()), $op, $name)
            }
        }
    };
}

macro_rules! impl_operators {
    ($t:ty) => {
        impl_binary_operator!($t, Add, add, BinOp::Add, "add");
        impl_binary_operator!($t, Sub, sub, BinOp::Sub, "sub");
        impl_binary_operator!($t, Mul, mul, BinOp::Mul, "mul");
        impl_binary_operator!($t, Div, div, BinOp::Div, "div");
        impl_assign_operator!($t, AddAssign, add_assign, BinOp::Add, "add_assign");
        impl_assign_operator!($t, SubAssign, sub_assign, BinOp::Sub, "sub_assign");
        impl_assign_operator!($t, MulAssign, mul_assign, BinOp::Mul, "mul_assign");
        impl_assign_operator!($t, DivAssign, div_assign, BinOp::Div, "div_assign");
    };
}

macro_rules! impl_scalar_operators {
    ($t:ty) => {
        impl_binary_scalar!($t, Add, add, BinOp::Add, "add");
        impl_binary_scalar!($t, Sub, sub, BinOp::Sub, "sub");
        impl_binary_scalar!($t, Mul, mul, BinOp::Mul, "mul");
        impl_binary_scalar!($t, Div, div, BinOp::Div, "div");
        impl_assign_scalar!($t, AddAssign, add_assign, BinOp::Add, "add_assign");
        impl_assign_scalar!($t, SubAssign, sub_assign, BinOp::Sub, "sub_assign");
        impl_assign_scalar!($t, MulAssign, mul_assign, BinOp::Mul, "mul_assign");
        impl_assign_scalar!($t, DivAssign, div_assign, BinOp::Div, "div_assign");
    };
}

impl_operators!(f32);
impl_operators!(u32);
impl_operators!(f64);
impl_operators!(Df32);
impl_scalar_operators!(f32);
impl_scalar_operators!(u32);
impl_binary_operator!(u32, Rem, rem, BinOp::Rem, "rem");
impl_binary_scalar!(u32, Rem, rem, BinOp::Rem, "rem");
impl_assign_operator!(u32, RemAssign, rem_assign, BinOp::Rem, "rem_assign");
impl_assign_scalar!(u32, RemAssign, rem_assign, BinOp::Rem, "rem_assign");

// -- Bitwise operations on u32 arrays --

macro_rules! impl_bitwise_operator {
    ($Trait:ident, $method:ident, $AssignTrait:ident, $assign_method:ident, $op:expr, $name:literal, $assign_name:literal) => {
        impl_binary_operator!(u32, $Trait, $method, $op, $name);
        impl_binary_scalar!(u32, $Trait, $method, $op, $name);
        impl_assign_operator!(u32, $AssignTrait, $assign_method, $op, $assign_name);
        impl_assign_scalar!(u32, $AssignTrait, $assign_method, $op, $assign_name);
    };
}

impl_bitwise_operator!(BitAnd, bitand, BitAndAssign, bitand_assign, BinOp::BitAnd, "bitand", "bitand_assign");
impl_bitwise_operator!(BitOr, bitor, BitOrAssign, bitor_assign, BinOp::BitOr, "bitor", "bitor_assign");
impl_bitwise_operator!(BitXor, bitxor, BitXorAssign, bitxor_assign, BinOp::BitXor, "bitxor", "bitxor_assign");
impl_bitwise_operator!(Shl, shl, ShlAssign, shl_assign, BinOp::Shl, "shl", "shl_assign");
impl_bitwise_operator!(Shr, shr, ShrAssign, shr_assign, BinOp::Shr, "shr", "shr_assign");

impl Not for &Array<u32> {
    type Output = Array<u32>;

    /// Elementwise bitwise complement.
    ///
    /// ```
    /// # use luma::*;
    /// let x = Array::from_slice(&[0u32, 1, 0xFFFF_FFFF], 3)?;
    /// assert_eq!((!&x).to_vec()?, [0xFFFF_FFFF, 0xFFFF_FFFE, 0]);
    /// # Ok::<(), luma::LumaError>(())
    /// ```
    #[track_caller]
    fn not(self) -> Array<u32> {
        let location = Location::caller();
        let result = self.run_unary(UnOp::BitNot, location);
        Array::resolve(self.context(), self.shape(), result, "not", location)
    }
}

impl Not for Array<u32> {
    type Output = Array<u32>;

    #[track_caller]
    fn not(self) -> Array<u32> {
        !&self
    }
}

impl Array<u32> {
    /// Elementwise `self << amount`, with an array of the same shape or a single `u32` as the
    /// amount; the amount is taken modulo 32 (so a shift by 32 is no shift) and bits shifted
    /// out are lost. The operator `&x << n` is the same.
    ///
    /// ```
    /// # use luma::*;
    /// let x = Array::from_slice(&[1u32, 0x8000_0001, 3], 3)?;
    /// assert_eq!(x.shl(4u32).to_vec()?, [16, 16, 48]);
    /// assert_eq!(x.shl(33u32).to_vec()?, [2, 2, 6]);
    /// # Ok::<(), luma::LumaError>(())
    /// ```
    #[track_caller]
    pub fn shl(&self, amount: impl Operand<u32>) -> Array<u32> {
        self.binary(amount.rhs(), BinOp::Shl, false, "shl")
    }

    /// Elementwise logical `self >> amount` (zeros come in from the left), with an array of the
    /// same shape or a single `u32` as the amount; the amount is taken modulo 32. The operator
    /// `&x >> n` is the same.
    ///
    /// ```
    /// # use luma::*;
    /// let x = Array::from_slice(&[0x8000_0000u32, 0xF0, 3], 3)?;
    /// assert_eq!(x.shr(4u32).to_vec()?, [0x0800_0000, 0xF, 0]);
    /// assert_eq!(x.shr(32u32).to_vec()?, x.to_vec()?);
    /// # Ok::<(), luma::LumaError>(())
    /// ```
    #[track_caller]
    pub fn shr(&self, amount: impl Operand<u32>) -> Array<u32> {
        self.binary(amount.rhs(), BinOp::Shr, false, "shr")
    }

    /// Elementwise rotation of the 32 bits to the left by `amount` (an array of the same shape
    /// or a single `u32`), taken modulo 32.
    ///
    /// ```
    /// # use luma::*;
    /// let x = Array::from_slice(&[0x8000_0001u32, 0x1234_5678], 2)?;
    /// assert_eq!(x.rotate_left(4u32).to_vec()?, [0x0000_0018, 0x2345_6781]);
    /// assert_eq!(x.rotate_left(32u32).to_vec()?, x.to_vec()?);
    /// # Ok::<(), luma::LumaError>(())
    /// ```
    #[track_caller]
    pub fn rotate_left(&self, amount: impl Operand<u32>) -> Array<u32> {
        self.binary(amount.rhs(), BinOp::Rotl, false, "rotate_left")
    }

    /// Elementwise rotation of the 32 bits to the right by `amount` (an array of the same shape
    /// or a single `u32`), taken modulo 32: `rotate_left(32 - amount)`.
    ///
    /// ```
    /// # use luma::*;
    /// let x = Array::from_slice(&[0x8000_0001u32, 0x1234_5678], 2)?;
    /// assert_eq!(x.rotate_right(4u32).to_vec()?, [0x1800_0000, 0x8123_4567]);
    /// assert_eq!(x.rotate_right(0u32).to_vec()?, x.to_vec()?);
    /// # Ok::<(), luma::LumaError>(())
    /// ```
    #[track_caller]
    pub fn rotate_right(&self, amount: impl Operand<u32>) -> Array<u32> {
        let location = Location::caller();
        match amount.rhs() {
            Rhs::Scalar(n) => self.binary_at(Rhs::Scalar(32u32.wrapping_sub(n)), BinOp::Rotl, false, "rotate_right", location),
            Rhs::Array(n) => {
                let complement = n.binary_at(Rhs::Scalar(32), BinOp::Sub, true, "rotate_right", location);
                self.binary_at(Rhs::Array(&complement), BinOp::Rotl, false, "rotate_right", location)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::fold_grid;

    #[test]
    fn folded_grids_cover_the_groups_without_a_nearly_empty_row() {
        let max = 65535u32;
        let counts = [0, 1, 2, 65534, 65535, 65536, 65537, 100_000, 131_070, 131_071, 131_072, 262_144, 1 << 20, 4_000_000, 65535 * 65535];
        for count in counts {
            let [x, y, z] = fold_grid(count, max);
            let groups = count.max(1);
            assert_eq!(z, 1);
            assert!(x >= 1 && y >= 1 && x <= max && y <= max, "{count}: {x}x{y}");
            let (grid, groups) = (u64::from(x) * u64::from(y), u64::from(groups));
            assert!(grid >= groups, "{count}: {x}x{y} does not cover");
            assert!(grid - groups < u64::from(y), "{count}: {x}x{y} leaves a nearly empty row");
        }
        assert_eq!(fold_grid(65536, max), [32768, 2, 1]);
        assert_eq!(fold_grid(65535, max), [65535, 1, 1]);
    }
}
