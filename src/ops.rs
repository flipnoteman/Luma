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
use crate::builtin::{BinOp, Builtin, Class, CmpOp, EpilogueStep, Part, UnOp};
use crate::context::{Context, RawArg};
use crate::error::{LumaError, Result};
use crate::complex::C32;
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
impl_value_operand!(i32);
impl_value_operand!(bool);
impl_value_operand!(Df32);
impl_value_operand!(C32);
// There is deliberately no `Operand<f64> for f64`: a second float type among the scalar
// operands would make `select(&mask, 1.0, 0.0)` and every other call that only has float
// literals to go by infer `f64` instead of `f32`. The operators (`&x * 2.0`) take `f64`
// scalars; comparisons, `minimum`/`maximum` and `select` take `f64` arrays.
impl_scalar_operand!(f32);
impl_scalar_operand!(u32);
impl_scalar_operand!(i32);

/// Seals [`Condition`].
pub(crate) mod condition_sealed {
    pub trait Sealed {}
}

/// What [`select`], [`Array::sum_where`] and [`Fields::select`](crate::Fields::select) test: an
/// `Array<bool>`, or an `Array<u32>` whose non-zero elements are true (code written when
/// comparisons returned `u32` keeps working). Views are arrays, so they are conditions too.
pub trait Condition: condition_sealed::Sealed {
    #[doc(hidden)]
    fn condition(&self) -> Cond<'_>;
}

/// The condition of a [`select`]: the array of words it tests against 0.
#[doc(hidden)]
#[derive(Clone, Copy)]
pub enum Cond<'a> {
    Bool(&'a Array<bool>),
    Word(&'a Array<u32>),
}

impl Cond<'_> {
    pub(crate) fn context(&self) -> &Context {
        match self {
            Cond::Bool(array) => array.context(),
            Cond::Word(array) => array.context(),
        }
    }

    pub(crate) fn shape(&self) -> Shape {
        match self {
            Cond::Bool(array) => array.shape(),
            Cond::Word(array) => array.shape(),
        }
    }

    /// The condition as an operand: a word array is compared with 0 first (lazily, inside the
    /// kernel that selects), because the operand of a select is a `bool`, which is a bit.
    pub(crate) fn src(&self, location: &'static Location<'static>) -> Outcome<Src> {
        match self {
            Cond::Bool(array) => array.src(),
            Cond::Word(array) => array.compare_at(Rhs::Scalar(0u32), CmpOp::Ne, "select", location).src(),
        }
    }
}

impl condition_sealed::Sealed for Array<bool> {}
impl condition_sealed::Sealed for Array<u32> {}

impl Condition for Array<bool> {
    fn condition(&self) -> Cond<'_> {
        Cond::Bool(self)
    }
}

impl Condition for Array<u32> {
    fn condition(&self) -> Cond<'_> {
        Cond::Word(self)
    }
}

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
pub(crate) fn fold_grid(count: u32, max: u32) -> [u32; 3] {
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

    /// The test `class` of every element, as an `Array<bool>` (see [`Class`]).
    fn classify(&self, class: Class, location: &'static Location<'static>) -> Array<bool> {
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

    /// This array evaluated, and copied if it is (or became, `-(-x)`) a placeholder of an
    /// iteration step: the placeholder borrows the buffers of the state it stands for, and an
    /// array that is returned to the caller must not (a write into the state with it would bind
    /// one buffer twice). A failure shows when the array is used.
    #[track_caller]
    pub(crate) fn detached(self) -> Array<T> {
        if !self.storage().is_ok_and(|storage| storage.is_alias()) {
            return self;
        }
        let copy = self.copied();
        let _ = copy.storage();
        copy
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
        if storage.is_alias() {
            return Err(LumaError::ReadOnly("the state passed to an iteration step").into());
        }
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
            && matches!(scalar.ty, ElemTy::U32 | ElemTy::I32)
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
    fn compare(&self, rhs: Rhs<'_, T>, op: CmpOp, name: &'static str) -> Array<bool> {
        self.compare_at(rhs, op, name, Location::caller())
    }

    /// [`Array::compare`] reporting failures at `location`.
    pub(crate) fn compare_at(
        &self,
        rhs: Rhs<'_, T>,
        op: CmpOp,
        name: &'static str,
        location: &'static Location<'static>,
    ) -> Array<bool> {
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
        self.assign_op_at(rhs, op, name, Location::caller());
    }

    /// [`Array::assign_op`] reporting failures at `location`.
    pub(crate) fn assign_op_at(&mut self, rhs: Rhs<'_, T>, op: BinOp, name: &'static str, location: &'static Location<'static>) {
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

    // -- Comparisons --

    /// `self < rhs` as an `Array<bool>`.
    #[track_caller]
    pub fn lt(&self, rhs: impl Operand<T>) -> Array<bool> {
        self.compare(rhs.rhs(), CmpOp::Lt, "lt")
    }

    /// `self <= rhs` as an `Array<bool>`.
    #[track_caller]
    pub fn le(&self, rhs: impl Operand<T>) -> Array<bool> {
        self.compare(rhs.rhs(), CmpOp::Le, "le")
    }

    /// `self > rhs` as an `Array<bool>`.
    #[track_caller]
    pub fn gt(&self, rhs: impl Operand<T>) -> Array<bool> {
        self.compare(rhs.rhs(), CmpOp::Gt, "gt")
    }

    /// `self >= rhs` as an `Array<bool>`.
    #[track_caller]
    pub fn ge(&self, rhs: impl Operand<T>) -> Array<bool> {
        self.compare(rhs.rhs(), CmpOp::Ge, "ge")
    }

    /// `self == rhs` as an `Array<bool>`.
    #[track_caller]
    pub fn eq(&self, rhs: impl Operand<T>) -> Array<bool> {
        self.compare(rhs.rhs(), CmpOp::Eq, "eq")
    }

    /// `self != rhs` as an `Array<bool>`.
    #[track_caller]
    pub fn ne(&self, rhs: impl Operand<T>) -> Array<bool> {
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
        if T::LAYOUT.is_packed() {
            // The words of the bits: the kernel runs over words, and knows the elements.
            let core = ctx.builtin(Builtin::FillBits, "u32")?;
            for j in 0..storage.segments() {
                let (n, bytes) = (storage.range(j).len(), storage.segment_byte_len(j));
                let pin = storage.pin(j)?;
                let words = [bytemuck::cast::<_, u32>(value.into_repr()), len_u32(n)?];
                launch_1d(ctx, &core, len_u32((bytes / 4) as usize)?, &[RawArg::write(pin.buffer(), bytes), RawArg::Uniform(bytemuck::cast_slice(&words))])?;
            }
            return Ok(out);
        }
        let core = ctx.builtin(Builtin::Fill, T::wgsl_type())?;
        for j in 0..storage.segments() {
            let n = storage.range(j).len();
            let pin = storage.pin(j)?;
            launch_1d(
                ctx,
                &core,
                len_u32(n)?,
                &[
                    RawArg::write(pin.buffer(), storage.segment_byte_len(j)),
                    RawArg::Uniform(bytemuck::bytes_of(&value.into_repr())),
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

/// Picks `a[i]` where `cond[i]` is true and `b[i]` elsewhere. The condition is an `Array<bool>`
/// (or an `Array<u32>`, non-zero meaning true: see [`Condition`]). Either branch is an
/// `&Array<T>` or a single `T` applied to every element; a scalar is a uniform of the fused
/// kernel, so no constant array is materialized or read from memory:
///
/// ```ignore
/// let ones = select(&x.gt(0.0), 1.0, 0.0);
/// let clipped = select(&x.gt(0.0), &x, 0.0);
/// ```
///
/// The result takes the condition's shape and context. An array branch has the condition's
/// shape or extent 1 on the axes where it differs (a `[n, 1]` branch under an `[n, m]`
/// condition is read at every column, see [`Array::broadcast_to`]). A mismatch poisons the
/// result with `ShapeMismatch` (reported as `lhs` = the condition's shape, `rhs` = the branch's)
/// or `ContextMismatch`.
#[track_caller]
pub fn select<T: ScalarType>(cond: &impl Condition, a: impl Operand<T>, b: impl Operand<T>) -> Array<T> {
    select_at(cond, a, b, Location::caller())
}

/// [`select`] reporting failures at `location`.
pub(crate) fn select_at<T: ScalarType>(
    cond: &impl Condition,
    a: impl Operand<T>,
    b: impl Operand<T>,
    location: &'static Location<'static>,
) -> Array<T> {
    let cond = cond.condition();
    let result = run_select(cond, a.rhs(), b.rhs(), location);
    Array::resolve(cond.context(), cond.shape(), result, "select", location)
}

pub(crate) fn run_select<T: ScalarType>(
    cond: Cond<'_>,
    a: Rhs<'_, T>,
    b: Rhs<'_, T>,
    location: &'static Location<'static>,
) -> Outcome<Array<T>> {
    let ctx = cond.context();
    let shape = cond.shape();
    let cond_src = cond.src(location)?;
    let branch = |rhs: Rhs<'_, T>| -> Outcome<Src> {
        match rhs {
            Rhs::Array(other) => {
                if !ctx.same_as(other.context()) {
                    return Err(LumaError::ContextMismatch.into());
                }
                other.broadcast_operand(shape, location)
            }
            Rhs::Scalar(value) => Ok(Src::Scalar(Scalar::of(value)?)),
        }
    };
    let a_src = branch(a)?;
    let b_src = branch(b)?;
    lazy(ctx, shape, Op::new(OpKind::Select, ElemTy::of::<T>()?), [cond_src, a_src, b_src], "select", location)
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
    /// Elementwise integer part: `x` rounded toward zero.
    trunc => UnOp::Trunc;
    /// Elementwise nearest integer, halves to the even one (`0.5` -> `0`, `1.5` -> `2`, `2.5` ->
    /// `2`), which is what WGSL's `round` and NumPy's `round` both do.
    round => UnOp::Round;
    /// Elementwise `ln(1 + x)`, accurate for `x` near 0 where `(1.0 + x).ln()` loses the digits
    /// of `x`: a series in `x / (2 + x)` for `-0.5 < x < 1`, `ln(1 + x)` elsewhere (about
    /// `1e-6` relative error). `-inf` at `-1`, NaN below it.
    log1p => UnOp::Log1p;
    /// Elementwise `e^x - 1`, accurate for `x` near 0 where `x.exp() - 1.0` cancels: a Taylor
    /// series for `|x| < 1`, `exp(x) - 1` elsewhere.
    expm1 => UnOp::Expm1;
}

macro_rules! impl_float_classes {
    ($($ty:ty),*) => {$(
        impl Array<$ty> {
            /// Elementwise: `1` where the element is neither infinite nor `NaN`, else `0`; a
            /// lazy `bool` array that fuses like any elementwise operation. The test reads the
            /// bits of the exponent, so it does not depend on how the device compares `NaN`
            /// (a compiler may assume no `NaN` exists and fold `x != x`). For [`Df32`] it is the
            /// high word that counts.
            #[track_caller]
            pub fn is_finite(&self) -> Array<bool> {
                self.classify(Class::Finite, Location::caller())
            }

            /// Elementwise: `1` where the element is `NaN`, else `0` (see [`Array::is_finite`]).
            #[track_caller]
            pub fn is_nan(&self) -> Array<bool> {
                self.classify(Class::Nan, Location::caller())
            }

            /// Elementwise: `1` where the element is positive or negative infinity, else `0`
            /// (see [`Array::is_finite`]).
            #[track_caller]
            pub fn is_inf(&self) -> Array<bool> {
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
    /// Elementwise integer part: `x` rounded toward zero.
    trunc => UnOp::Trunc;
    /// Elementwise nearest integer, halves to the even one (as [`Array<f32>::round`](Array::round)).
    round => UnOp::Round;
}

wide_unary_methods! { i32;
    /// Elementwise absolute value (`i32::MIN` stays `i32::MIN`).
    abs => UnOp::Abs;
    /// Elementwise sign: `-1`, `0` or `1`.
    sign => UnOp::Sign;
}

wide_unary_methods! { C32;
}

wide_unary_methods! { Df32;
    /// Elementwise square root (accurate to about 44 bits).
    sqrt => UnOp::Sqrt;
    /// Elementwise `e^x`, accurate to about 44 bits; infinity above `88.72` and `0` below `-87.33`.
    /// A result below `2^-126` (where the low word stops being a normal `f32`) is `0`.
    exp => UnOp::Exp;
    /// Elementwise natural logarithm, accurate to about 44 bits (relative to the result, also near
    /// `1`); `-inf` at `0`, `NaN` below it.
    ln => UnOp::Ln;
    /// Elementwise hyperbolic tangent, accurate to about 44 bits (relative to the result, also
    /// near `0`); `+-1` from `|x| > 20`.
    tanh => UnOp::Tanh;
    /// Elementwise absolute value.
    abs => UnOp::Abs;
    /// Elementwise `ln(1 + x)`, accurate to about 44 bits relative to the result also for `x`
    /// near 0 (see [`Array<f32>::log1p`](Array::log1p)); `-inf` at `-1`, `NaN` below it.
    log1p => UnOp::Log1p;
    /// Elementwise `e^x - 1`, accurate to about 44 bits relative to the result also for `x` near
    /// 0; infinity above `88.72`, `-1` below `-87.33`.
    expm1 => UnOp::Expm1;
}

impl Array<Df32> {
    /// Elementwise `x` raised to `exponent`, accurate to about 44 bits for results in the range of
    /// `exp` and `|exponent * ln x|` up to a few hundred. `NaN` for a negative base, `1` for an
    /// exponent of `0` (also of `NaN`), and as `pow` for a base of `0` or infinity.
    #[track_caller]
    pub fn powf(&self, exponent: Df32) -> Array<Df32> {
        let location = Location::caller();
        let result = self.src().and_then(|operand| {
            lazy(
                self.context(),
                self.shape(),
                Op::new(OpKind::Powf, ElemTy::Df32),
                [operand, Src::Scalar(Scalar::of(exponent)?), Src::unused()],
                "powf",
                location,
            )
        });
        Array::resolve(self.context(), self.shape(), result, "powf", location)
    }
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
impl_operators!(i32);
impl_operators!(f64);
impl_operators!(Df32);
impl_operators!(C32);
impl_scalar_operators!(f32);
impl_scalar_operators!(u32);
impl_scalar_operators!(i32);
impl_binary_operator!(u32, Rem, rem, BinOp::Rem, "rem");
impl_binary_scalar!(u32, Rem, rem, BinOp::Rem, "rem");
impl_assign_operator!(u32, RemAssign, rem_assign, BinOp::Rem, "rem_assign");
impl_assign_scalar!(u32, RemAssign, rem_assign, BinOp::Rem, "rem_assign");
impl_binary_operator!(i32, Rem, rem, BinOp::Rem, "rem");
impl_binary_scalar!(i32, Rem, rem, BinOp::Rem, "rem");
impl_assign_operator!(i32, RemAssign, rem_assign, BinOp::Rem, "rem_assign");
impl_assign_scalar!(i32, RemAssign, rem_assign, BinOp::Rem, "rem_assign");

// -- Bitwise operations on u32 and i32 arrays --

macro_rules! impl_bitwise_operator {
    ($t:ty, $Trait:ident, $method:ident, $AssignTrait:ident, $assign_method:ident, $op:expr, $name:literal, $assign_name:literal) => {
        impl_binary_operator!($t, $Trait, $method, $op, $name);
        impl_binary_scalar!($t, $Trait, $method, $op, $name);
        impl_assign_operator!($t, $AssignTrait, $assign_method, $op, $assign_name);
        impl_assign_scalar!($t, $AssignTrait, $assign_method, $op, $assign_name);
    };
}

macro_rules! impl_bitwise_operators {
    ($t:ty) => {
        impl_bitwise_operator!($t, BitAnd, bitand, BitAndAssign, bitand_assign, BinOp::BitAnd, "bitand", "bitand_assign");
        impl_bitwise_operator!($t, BitOr, bitor, BitOrAssign, bitor_assign, BinOp::BitOr, "bitor", "bitor_assign");
        impl_bitwise_operator!($t, BitXor, bitxor, BitXorAssign, bitxor_assign, BinOp::BitXor, "bitxor", "bitxor_assign");
        impl_bitwise_operator!($t, Shl, shl, ShlAssign, shl_assign, BinOp::Shl, "shl", "shl_assign");
        impl_bitwise_operator!($t, Shr, shr, ShrAssign, shr_assign, BinOp::Shr, "shr", "shr_assign");
    };
}

impl_bitwise_operators!(u32);
impl_bitwise_operators!(i32);

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

impl Not for &Array<i32> {
    type Output = Array<i32>;

    /// Elementwise bitwise complement (`-x - 1`).
    #[track_caller]
    fn not(self) -> Array<i32> {
        let location = Location::caller();
        let result = self.run_unary(UnOp::BitNot, location);
        Array::resolve(self.context(), self.shape(), result, "not", location)
    }
}

impl Not for Array<i32> {
    type Output = Array<i32>;

    #[track_caller]
    fn not(self) -> Array<i32> {
        !&self
    }
}

impl_binary_operator!(bool, BitAnd, bitand, BinOp::BitAnd, "bitand");
impl_binary_operator!(bool, BitOr, bitor, BinOp::BitOr, "bitor");
impl_binary_operator!(bool, BitXor, bitxor, BinOp::BitXor, "bitxor");
impl_assign_operator!(bool, BitAndAssign, bitand_assign, BinOp::BitAnd, "bitand_assign");
impl_assign_operator!(bool, BitOrAssign, bitor_assign, BinOp::BitOr, "bitor_assign");
impl_assign_operator!(bool, BitXorAssign, bitxor_assign, BinOp::BitXor, "bitxor_assign");
// `bool * bool` is the logical and, so that products of comparisons keep their meaning.
impl_binary_operator!(bool, Mul, mul, BinOp::BitAnd, "mul");

impl Not for &Array<bool> {
    type Output = Array<bool>;

    /// Elementwise logical negation.
    #[track_caller]
    fn not(self) -> Array<bool> {
        let location = Location::caller();
        let result = self.run_unary(UnOp::BitNot, location);
        Array::resolve(self.context(), self.shape(), result, "not", location)
    }
}

impl Not for Array<bool> {
    type Output = Array<bool>;

    #[track_caller]
    fn not(self) -> Array<bool> {
        !&self
    }
}

// -- NumPy-style promotion of `bool` arrays --

/// The arithmetic operators between a `bool` array and a number.
#[derive(Clone, Copy)]
pub(crate) enum Promote {
    Mul,
    Add,
    Sub,
}

impl Promote {
    fn name(self) -> &'static str {
        match self {
            Promote::Mul => "mul",
            Promote::Add => "add",
            Promote::Sub => "sub",
        }
    }

    fn assign_name(self) -> &'static str {
        match self {
            Promote::Mul => "mul_assign",
            Promote::Add => "add_assign",
            Promote::Sub => "sub_assign",
        }
    }

    fn op(self) -> BinOp {
        match self {
            Promote::Mul => BinOp::Mul,
            Promote::Add => BinOp::Add,
            Promote::Sub => BinOp::Sub,
        }
    }
}

impl Array<bool> {
    /// `self op other` (`other op self` unless `bool_left`) with the booleans as the 0 and 1 of
    /// `other`'s type. A product is `select(self, other, 0)`; a sum or difference converts first.
    pub(crate) fn promote<N: ScalarType>(&self, other: Rhs<'_, N>, bool_left: bool, op: Promote, location: &'static Location<'static>) -> Array<N> {
        if let Promote::Mul = op {
            let result = self.run_masked(other, location);
            return Array::resolve(self.context(), self.shape(), result, op.name(), location);
        }
        let number = self.cast_at::<N>(location);
        match (bool_left, other) {
            (true, other) => number.binary_at(other, op.op(), false, op.name(), location),
            (false, Rhs::Scalar(value)) => number.binary_at(Rhs::Scalar(value), op.op(), true, op.name(), location),
            (false, Rhs::Array(array)) => array.binary_at(Rhs::Array(&number), op.op(), false, op.name(), location),
        }
    }

    /// `select(self, other, 0)`, broadcasting the shapes like a binary operation.
    fn run_masked<N: ScalarType>(&self, other: Rhs<'_, N>, location: &'static Location<'static>) -> Outcome<Array<N>> {
        let ctx = self.context();
        let mask = self.src()?;
        let (mask, value, shape) = match other {
            Rhs::Array(array) => {
                let value = array.src()?;
                if !ctx.same_as(array.context()) {
                    return Err(LumaError::ContextMismatch.into());
                }
                if self.shape() == array.shape() {
                    (mask, value, self.shape())
                } else {
                    let mismatch = || LumaError::ShapeMismatch { lhs: self.shape().dims(), rhs: array.shape().dims() };
                    let out = broadcast_shape(self.shape(), array.shape()).ok_or_else(mismatch)?;
                    let mask = if self.shape() == out { mask } else { self.broadcast_axis(out, location)? };
                    let value = if array.shape() == out { value } else { array.broadcast_axis(out, location)? };
                    (mask, value, out)
                }
            }
            Rhs::Scalar(value) => (mask, Src::Scalar(Scalar::of(value)?), self.shape()),
        };
        let zero = Src::Scalar(Scalar::of(N::zeroed())?);
        lazy(ctx, shape, Op::new(OpKind::Select, ElemTy::of::<N>()?), [mask, value, zero], "mul", location)
    }

    /// `self op= other`: the booleans as the 0 and 1 of the target's type.
    fn promote_assign<N: ScalarType>(target: &mut Array<N>, other: &Array<bool>, op: Promote, location: &'static Location<'static>) {
        if let Promote::Mul = op {
            // `x *= &mask` is `x = select(mask, x, 0)`, like `&x * &mask`.
            let product = other.promote(Rhs::Array(&*target), true, op, location);
            if product.shape() == target.shape() {
                *target = product;
            } else {
                target.poison(Arc::new(LumaError::ShapeMismatch { lhs: target.shape().dims(), rhs: other.shape().dims() }));
            }
            return;
        }
        let number = other.cast_at::<N>(location);
        target.assign_op_at(Rhs::Array(&number), op.op(), op.assign_name(), location);
    }
}

macro_rules! impl_promotion {
    ($Trait:ident, $method:ident, $op:expr, $n:ty) => {
        impl $Trait<&Array<$n>> for &Array<bool> {
            type Output = Array<$n>;
            #[track_caller]
            fn $method(self, rhs: &Array<$n>) -> Array<$n> {
                self.promote(Rhs::Array(rhs), true, $op, Location::caller())
            }
        }

        impl $Trait<Array<$n>> for &Array<bool> {
            type Output = Array<$n>;
            #[track_caller]
            fn $method(self, rhs: Array<$n>) -> Array<$n> {
                self.promote(Rhs::Array(&rhs), true, $op, Location::caller())
            }
        }

        impl $Trait<&Array<$n>> for Array<bool> {
            type Output = Array<$n>;
            #[track_caller]
            fn $method(self, rhs: &Array<$n>) -> Array<$n> {
                self.promote(Rhs::Array(rhs), true, $op, Location::caller())
            }
        }

        impl $Trait<Array<$n>> for Array<bool> {
            type Output = Array<$n>;
            #[track_caller]
            fn $method(self, rhs: Array<$n>) -> Array<$n> {
                self.promote(Rhs::Array(&rhs), true, $op, Location::caller())
            }
        }

        impl $Trait<&Array<bool>> for &Array<$n> {
            type Output = Array<$n>;
            #[track_caller]
            fn $method(self, rhs: &Array<bool>) -> Array<$n> {
                rhs.promote(Rhs::Array(self), false, $op, Location::caller())
            }
        }

        impl $Trait<Array<bool>> for &Array<$n> {
            type Output = Array<$n>;
            #[track_caller]
            fn $method(self, rhs: Array<bool>) -> Array<$n> {
                rhs.promote(Rhs::Array(self), false, $op, Location::caller())
            }
        }

        impl $Trait<&Array<bool>> for Array<$n> {
            type Output = Array<$n>;
            #[track_caller]
            fn $method(self, rhs: &Array<bool>) -> Array<$n> {
                rhs.promote(Rhs::Array(&self), false, $op, Location::caller())
            }
        }

        impl $Trait<Array<bool>> for Array<$n> {
            type Output = Array<$n>;
            #[track_caller]
            fn $method(self, rhs: Array<bool>) -> Array<$n> {
                rhs.promote(Rhs::Array(&self), false, $op, Location::caller())
            }
        }
    };
}

/// The same with a number: `bool_array * 2.0`, `1.0 - bool_array`.
macro_rules! impl_promotion_scalar {
    ($Trait:ident, $method:ident, $op:expr, $n:ty) => {
        impl $Trait<$n> for &Array<bool> {
            type Output = Array<$n>;
            #[track_caller]
            fn $method(self, rhs: $n) -> Array<$n> {
                self.promote(Rhs::Scalar(rhs), true, $op, Location::caller())
            }
        }

        impl $Trait<$n> for Array<bool> {
            type Output = Array<$n>;
            #[track_caller]
            fn $method(self, rhs: $n) -> Array<$n> {
                self.promote(Rhs::Scalar(rhs), true, $op, Location::caller())
            }
        }

        impl $Trait<&Array<bool>> for $n {
            type Output = Array<$n>;
            #[track_caller]
            fn $method(self, rhs: &Array<bool>) -> Array<$n> {
                rhs.promote(Rhs::Scalar(self), false, $op, Location::caller())
            }
        }

        impl $Trait<Array<bool>> for $n {
            type Output = Array<$n>;
            #[track_caller]
            fn $method(self, rhs: Array<bool>) -> Array<$n> {
                rhs.promote(Rhs::Scalar(self), false, $op, Location::caller())
            }
        }
    };
}

macro_rules! impl_promotion_assign {
    ($Trait:ident, $method:ident, $op:expr, $n:ty) => {
        impl $Trait<&Array<bool>> for Array<$n> {
            #[track_caller]
            fn $method(&mut self, rhs: &Array<bool>) {
                Array::<bool>::promote_assign(self, rhs, $op, Location::caller())
            }
        }
    };
}

macro_rules! impl_promotions {
    ($n:ty) => {
        impl_promotion!(Mul, mul, Promote::Mul, $n);
        impl_promotion!(Add, add, Promote::Add, $n);
        impl_promotion!(Sub, sub, Promote::Sub, $n);
        impl_promotion_assign!(MulAssign, mul_assign, Promote::Mul, $n);
        impl_promotion_assign!(AddAssign, add_assign, Promote::Add, $n);
        impl_promotion_assign!(SubAssign, sub_assign, Promote::Sub, $n);
    };
}

macro_rules! impl_promotions_scalar {
    ($n:ty) => {
        impl_promotion_scalar!(Mul, mul, Promote::Mul, $n);
        impl_promotion_scalar!(Add, add, Promote::Add, $n);
        impl_promotion_scalar!(Sub, sub, Promote::Sub, $n);
    };
}

impl_promotions!(f32);
impl_promotions!(u32);
impl_promotions!(i32);
impl_promotions!(f64);
impl_promotions!(Df32);
impl_promotions!(C32);
// No `f64` scalar: with two float types among the scalars, `bool_array * 2.0` would infer `f64`.
impl_promotions_scalar!(f32);
impl_promotions_scalar!(u32);
impl_promotions_scalar!(i32);
impl_promotions_scalar!(Df32);
impl_promotions_scalar!(C32);

impl Array<i32> {
    /// Elementwise `self << amount`, with an array of the same shape or a single `i32` as the
    /// amount; the amount's bits are taken modulo 32 and bits shifted out are lost. The operator
    /// `&x << n` is the same.
    #[track_caller]
    pub fn shl(&self, amount: impl Operand<i32>) -> Array<i32> {
        self.binary(amount.rhs(), BinOp::Shl, false, "shl")
    }

    /// Elementwise arithmetic `self >> amount` (the sign bit comes in from the left), with an
    /// array of the same shape or a single `i32` as the amount, taken modulo 32. The operator
    /// `&x >> n` is the same.
    #[track_caller]
    pub fn shr(&self, amount: impl Operand<i32>) -> Array<i32> {
        self.binary(amount.rhs(), BinOp::Shr, false, "shr")
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

// -- Complex arrays --

impl Array<C32> {
    /// The real quantity `part` of every element, reporting failures at `location`.
    fn part_at(&self, part: Part, location: &'static Location<'static>) -> Array<f32> {
        let result = self.src().and_then(|operand| {
            lazy(self.context(), self.shape(), Op::new(OpKind::Part(part), ElemTy::C32), [operand, Src::unused(), Src::unused()], part.name(), location)
        });
        Array::resolve(self.context(), self.shape(), result, part.name(), location)
    }

    /// Elementwise real part, an `Array<f32>`. Lazy and fused like every elementwise operation.
    ///
    /// ```
    /// # use luma::*;
    /// let z = Array::from_slice(&[C32::new(1.0, 2.0), C32::new(-3.0, 4.0)], 2)?;
    /// assert_eq!(z.real().to_vec()?, [1.0, -3.0]);
    /// # Ok::<(), luma::LumaError>(())
    /// ```
    #[track_caller]
    pub fn real(&self) -> Array<f32> {
        self.part_at(Part::Real, Location::caller())
    }

    /// Elementwise imaginary part, an `Array<f32>`.
    #[track_caller]
    pub fn imag(&self) -> Array<f32> {
        self.part_at(Part::Imag, Location::caller())
    }

    /// Elementwise modulus `sqrt(re^2 + im^2)`, an `Array<f32>`. Scaled by the larger component,
    /// so it neither overflows nor underflows where the result is representable (`|3e30 + 4e30 i|`
    /// is `5e30`); infinite if a component is infinite, `NaN` if one is and none is infinite.
    #[track_caller]
    pub fn abs(&self) -> Array<f32> {
        self.part_at(Part::Abs, Location::caller())
    }

    /// Elementwise argument (phase) `atan2(im, re)` in `[-pi, pi]`, an `Array<f32>`.
    #[track_caller]
    pub fn arg(&self) -> Array<f32> {
        self.part_at(Part::Arg, Location::caller())
    }

    /// Elementwise squared modulus `re^2 + im^2`, an `Array<f32>` (cheaper than [`Array::abs`]
    /// and without its scaling: it overflows above `|z| = 1.8e19`).
    #[track_caller]
    pub fn norm_sqr(&self) -> Array<f32> {
        self.part_at(Part::NormSqr, Location::caller())
    }

    /// Elementwise complex conjugate `re - im i`.
    #[track_caller]
    pub fn conj(&self) -> Array<C32> {
        let location = Location::caller();
        let result = self.run_unary(UnOp::Conj, location);
        Array::resolve(self.context(), self.shape(), result, "conj", location)
    }

    /// Elementwise `e^z = e^re (cos im + i sin im)`.
    #[track_caller]
    pub fn exp(&self) -> Array<C32> {
        let location = Location::caller();
        let result = self.run_unary(UnOp::Exp, location);
        Array::resolve(self.context(), self.shape(), result, "exp", location)
    }

    /// Elementwise principal square root (non-negative real part; the sign of the imaginary part
    /// follows that of `im`, also of a zero).
    #[track_caller]
    pub fn sqrt(&self) -> Array<C32> {
        let location = Location::caller();
        let result = self.run_unary(UnOp::Sqrt, location);
        Array::resolve(self.context(), self.shape(), result, "sqrt", location)
    }

    /// `re + im i` elementwise from two real arrays of the same shape (or shapes that broadcast,
    /// see [`Array::broadcast_to`]); a mismatch poisons the result like an operator does. The
    /// inverse of [`Array::real`] and [`Array::imag`]; a `from_parts` of the parts of one array is
    /// that array.
    ///
    /// ```
    /// # use luma::*;
    /// let re = Array::from_slice(&[1.0f32, 2.0], 2)?;
    /// let im = Array::from_slice(&[-1.0f32, 0.5], 2)?;
    /// let z = Array::from_parts(&re, &im);
    /// assert_eq!(z.to_vec()?, [C32::new(1.0, -1.0), C32::new(2.0, 0.5)]);
    /// # Ok::<(), luma::LumaError>(())
    /// ```
    #[track_caller]
    pub fn from_parts(re: &Array<f32>, im: &Array<f32>) -> Array<C32> {
        Self::from_parts_at(re, im, Location::caller())
    }

    fn from_parts_at(re: &Array<f32>, im: &Array<f32>, location: &'static Location<'static>) -> Array<C32> {
        let result = re.run_binary::<C32>(Rhs::Array(im), false, OpKind::Make, "from_parts", location);
        let shape = result.as_ref().map_or(re.shape(), |array| array.shape());
        Array::resolve(re.context(), shape, result, "from_parts", location)
    }

    /// `r (cos theta + i sin theta)` elementwise from the moduli `r` and the arguments `theta`
    /// (two real arrays of the same shape).
    #[track_caller]
    pub fn from_polar(r: &Array<f32>, theta: &Array<f32>) -> Array<C32> {
        let location = Location::caller();
        let (c, s) = (theta.cos(), theta.sin());
        let re = r.binary_at(Rhs::Array(&c), BinOp::Mul, false, "from_polar", location);
        let im = r.binary_at(Rhs::Array(&s), BinOp::Mul, false, "from_polar", location);
        Self::from_parts_at(&re, &im, location)
    }

    /// `self op s` (`s op self` if `scalar_left`) with the real number `s` as `s + 0i`, computed
    /// on the parts (so that no `0 * inf` appears where the real number has no imaginary part to
    /// multiply). `op` is `Add`, `Sub`, `Mul` or `Div`.
    fn with_real(&self, s: f32, op: BinOp, scalar_left: bool, location: &'static Location<'static>) -> Array<C32> {
        let (re, im) = (self.part_at(Part::Real, location), self.part_at(Part::Imag, location));
        let name = match op {
            BinOp::Add => "add",
            BinOp::Sub => "sub",
            BinOp::Mul => "mul",
            _ => "div",
        };
        match (op, scalar_left) {
            (BinOp::Add | BinOp::Sub, false) => Self::from_parts_at(&re.binary_at(Rhs::Scalar(s), op, false, name, location), &im, location),
            (BinOp::Add, true) => Self::from_parts_at(&re.binary_at(Rhs::Scalar(s), op, true, name, location), &im, location),
            (BinOp::Sub, true) => {
                let (re, im) = (re.binary_at(Rhs::Scalar(s), op, true, name, location), im.neg_at(location));
                Self::from_parts_at(&re, &im, location)
            }
            (BinOp::Mul, _) | (BinOp::Div, false) => {
                let (re, im) = (re.binary_at(Rhs::Scalar(s), op, false, name, location), im.binary_at(Rhs::Scalar(s), op, false, name, location));
                Self::from_parts_at(&re, &im, location)
            }
            // `s / z` is a complex quotient.
            _ => self.binary_at(Rhs::Scalar(C32::from(s)), BinOp::Div, true, name, location),
        }
    }

    /// The sum of all elements, as a lazy one-element array (see [`Array::sum`]): the sums of the
    /// real and of the imaginary parts.
    #[track_caller]
    pub fn sum(&self) -> Array<C32> {
        let location = Location::caller();
        Self::from_parts_at(&self.part_at(Part::Real, location).sum(), &self.part_at(Part::Imag, location).sum(), location)
    }

    /// The mean of all elements, as a lazy one-element array: the sum divided by the length.
    #[track_caller]
    pub fn mean(&self) -> Array<C32> {
        let location = Location::caller();
        self.sum().with_real(self.len() as f32, BinOp::Div, false, location)
    }

    /// The sum along `axis`; the result has extent 1 along it (see [`Array::sum_axis`]).
    #[track_caller]
    pub fn sum_axis(&self, axis: usize) -> Array<C32> {
        let location = Location::caller();
        Self::from_parts_at(&self.part_at(Part::Real, location).sum_axis(axis), &self.part_at(Part::Imag, location).sum_axis(axis), location)
    }

    /// The mean along `axis`; the result has extent 1 along it.
    #[track_caller]
    pub fn mean_axis(&self, axis: usize) -> Array<C32> {
        let location = Location::caller();
        let n = self.shape().dims().get(axis).copied().unwrap_or(1);
        self.sum_axis(axis).with_real(n as f32, BinOp::Div, false, location)
    }
}

impl Array<f32> {
    /// The array as complex numbers with zero imaginary parts, an `Array<C32>`.
    #[track_caller]
    pub fn to_complex(&self) -> Array<C32> {
        let location = Location::caller();
        let result = self.run_binary::<C32>(Rhs::Scalar(0.0f32), false, OpKind::Make, "to_complex", location);
        Array::resolve(self.context(), self.shape(), result, "to_complex", location)
    }
}

/// The operators between an `Array<C32>` and a real number (`f32`), which stands for `s + 0i`.
macro_rules! impl_real_operator {
    ($Trait:ident, $method:ident, $op:expr) => {
        impl $Trait<f32> for &Array<C32> {
            type Output = Array<C32>;
            #[track_caller]
            fn $method(self, rhs: f32) -> Array<C32> {
                self.with_real(rhs, $op, false, Location::caller())
            }
        }

        impl $Trait<f32> for Array<C32> {
            type Output = Array<C32>;
            #[track_caller]
            fn $method(self, rhs: f32) -> Array<C32> {
                self.with_real(rhs, $op, false, Location::caller())
            }
        }

        impl $Trait<&Array<C32>> for f32 {
            type Output = Array<C32>;
            #[track_caller]
            fn $method(self, rhs: &Array<C32>) -> Array<C32> {
                rhs.with_real(self, $op, true, Location::caller())
            }
        }

        impl $Trait<Array<C32>> for f32 {
            type Output = Array<C32>;
            #[track_caller]
            fn $method(self, rhs: Array<C32>) -> Array<C32> {
                rhs.with_real(self, $op, true, Location::caller())
            }
        }
    };
}

impl_real_operator!(Add, add, BinOp::Add);
impl_real_operator!(Sub, sub, BinOp::Sub);
impl_real_operator!(Mul, mul, BinOp::Mul);
impl_real_operator!(Div, div, BinOp::Div);

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
