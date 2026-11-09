//! Operators, math functions and reductions on arrays.
//!
//! Elementwise operations (operators, math functions, comparisons, `select`, `powf`, `clamp`)
//! are recorded as lazy expressions and fused into one generated kernel when their result is
//! needed (see [`crate::jit`]); reductions fold the expression behind their input into their
//! first pass. In-place operators, fills, shifts and the later reduction passes are plain
//! built-in kernels (see [`crate::builtin`]) recorded into the context's batch. Methods that
//! return arrays never fail loudly: they return poisoned arrays (see [`Array`]). The `run_*`
//! functions do the work and return the failure; the public methods turn it into a poisoned
//! array tagged with the caller's location, or into a plain `Err` for the `try_*` methods.

use std::ops::{
    Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign,
};
use std::panic::Location;
use std::sync::Arc;

use crate::array::{Array, Failure};
use crate::builtin::{BinOp, Builtin, CmpOp, ReduceOp, UnOp, REDUCE_BLOCK};
use crate::context::{Context, RawArg, Scratch};
use crate::error::{LumaError, Result};
use crate::gpu_type::{GpuType, ScalarType};
use crate::jit::{self, ElemTy, Op, OpKind, Scalar, Src};
use crate::kernel::KernelCore;
use crate::shape::{Offsets, Shape};

type Outcome<T> = std::result::Result<T, Failure>;

/// The right-hand side of a binary operation.
#[doc(hidden)]
pub enum Rhs<'a, T: ScalarType> {
    Array(&'a Array<T>),
    Scalar(T),
}

/// A right-hand side accepted by the `try_*` methods, comparisons and `minimum`/`maximum`:
/// an `&Array<T>` of the same shape or a single `T` applied to every element.
pub trait Operand<T: ScalarType> {
    #[doc(hidden)]
    fn rhs(&self) -> Rhs<'_, T>;
}

macro_rules! impl_operand {
    ($t:ty) => {
        impl Operand<$t> for $t {
            fn rhs(&self) -> Rhs<'_, $t> {
                Rhs::Scalar(*self)
            }
        }

        impl Operand<$t> for &Array<$t> {
            fn rhs(&self) -> Rhs<'_, $t> {
                Rhs::Array(*self)
            }
        }
    };
}

impl_operand!(f32);
impl_operand!(u32);

/// Splits `count` workgroups into a 2D grid within the per-dimension limit; shaders recombine
/// them as `x + y * num_workgroups.x`.
pub(crate) fn fold_workgroups(ctx: &Context, count: u32) -> Result<[u32; 3]> {
    let max = ctx.max_workgroups_per_dimension();
    let groups = count.max(1);
    let x = groups.min(max).max(1);
    let y = groups.div_ceil(x);
    if y > max {
        return Err(LumaError::KernelGridTooLarge { workgroups: [x, y, 1], max });
    }
    Ok([x, y, 1])
}

/// Launches a kernel with one invocation per element for `n` elements (`luma.size.x = n`).
pub(crate) fn launch_1d(
    ctx: &Context,
    kernel: &KernelCore,
    n: u32,
    args: &[RawArg<'_>],
) -> Result<()> {
    let groups = n.div_ceil(kernel.workgroup[0]);
    ctx.launch(kernel, fold_workgroups(ctx, groups)?, [n, 1, 1], args)
}

fn len_u32(len: usize) -> Result<u32> {
    // Array lengths were validated to fit u32 when the array was created.
    u32::try_from(len).map_err(|_| LumaError::InvalidShape(format!("{len} elements exceed u32")))
}

/// Records the lazy operation `op` on `args` as an array of `shape`. Nothing runs yet.
fn lazy<O: GpuType>(
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

impl<T: ScalarType> Array<T> {
    // -- Lazy operations --

    /// `f(self)` for a unary operation.
    fn run_unary(
        &self,
        op: UnOp,
        location: &'static Location<'static>,
    ) -> Outcome<Array<T>> {
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
        let other = match rhs {
            Rhs::Array(other) => {
                let operand = other.src()?;
                if !ctx.same_as(other.context()) {
                    return Err(LumaError::ContextMismatch.into());
                }
                if self.shape() != other.shape() {
                    return Err(LumaError::ShapeMismatch {
                        lhs: self.shape().dims(),
                        rhs: other.shape().dims(),
                    }
                    .into());
                }
                operand
            }
            Rhs::Scalar(value) => Src::Scalar(Scalar::of(value)?),
        };
        let (left, right) = if scalar_left { (other, this) } else { (this, other) };
        lazy(
            ctx,
            self.shape(),
            Op::new(kind, ElemTy::of::<T>()?),
            [left, right, Src::unused()],
            name,
            location,
        )
    }

    /// `self = self op rhs`, as a kernel running now over the array's buffer.
    fn run_in_place(&self, rhs: Rhs<'_, T>, op: BinOp) -> Outcome<()> {
        let ctx = self.context();
        let storage = self.storage()?;
        let (a, a_bytes) = self.ready()?;
        let n = len_u32(self.len())?;
        match rhs {
            Rhs::Array(other) => {
                let (b, b_bytes) = other.ready()?;
                if !ctx.same_as(other.context()) {
                    return Err(LumaError::ContextMismatch.into());
                }
                if self.shape() != other.shape() {
                    return Err(LumaError::ShapeMismatch {
                        lhs: self.shape().dims(),
                        rhs: other.shape().dims(),
                    }
                    .into());
                }
                // Lazy operations recorded earlier still read the old contents.
                storage.readers.flush();
                let core = ctx.builtin(Builtin::InPlaceArray(op), T::wgsl_type())?;
                launch_1d(
                    ctx,
                    &core,
                    n,
                    &[RawArg::write(a, a_bytes), RawArg::read(b, b_bytes)],
                )?;
            }
            Rhs::Scalar(scalar) => {
                storage.readers.flush();
                let core = ctx.builtin(Builtin::InPlaceScalar(op), T::wgsl_type())?;
                launch_1d(
                    ctx,
                    &core,
                    n,
                    &[
                        RawArg::write(a, a_bytes),
                        RawArg::Uniform(bytemuck::bytes_of(&scalar)),
                    ],
                )?;
            }
        }
        Ok(())
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
        let location = Location::caller();
        let result = self.run_binary(rhs, scalar_left, OpKind::Binary(op), name, location);
        Array::resolve(self.context(), self.shape(), result, name, location)
    }

    #[track_caller]
    fn compare(&self, rhs: Rhs<'_, T>, op: CmpOp, name: &'static str) -> Array<u32> {
        let location = Location::caller();
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
        out.ready().map_err(|failure| failure.into_error().root().clone())?;
        Ok(out)
    }

    #[track_caller]
    pub(crate) fn assign(&mut self, rhs: Rhs<'_, T>, op: BinOp, name: &'static str) {
        let location = Location::caller();
        let result = if self.pending().is_some() {
            // The target is an expression without a buffer, so nothing can observe it changing:
            // it simply becomes the expression `self op rhs`, and stays fusable.
            self.run_binary(rhs, false, OpKind::Binary(op), name, location).map(|new: Array<T>| {
                *self = new;
            })
        } else {
            self.run_in_place(rhs, op)
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
        let out = Array::<T>::alloc(ctx, shape.into())?;
        let (buffer, bytes) = out.ready().map_err(Failure::into_error)?;
        let core = ctx.builtin(Builtin::Fill, T::wgsl_type())?;
        launch_1d(
            ctx,
            &core,
            len_u32(out.len())?,
            &[RawArg::write(buffer, bytes), RawArg::Uniform(bytemuck::bytes_of(&value))],
        )?;
        Ok(out)
    }

    // -- Shifts --

    /// Circular shift like `af::shift`: element `i` moves to `i + offset` in every dimension,
    /// wrapping around (`[1, 2, 3, 4].shift(1)` is `[4, 1, 2, 3]`; negative offsets shift the
    /// other way). Offsets can be larger than a dimension.
    #[track_caller]
    pub fn shift(&self, offsets: impl Into<Offsets>) -> Array<T> {
        let location = Location::caller();
        let result = self.run_shift(offsets.into());
        Array::resolve(self.context(), self.shape(), result, "shift", location)
    }

    fn run_shift(&self, offsets: Offsets) -> Outcome<Array<T>> {
        let ctx = self.context();
        let (a, a_bytes) = self.ready()?;
        let dims = self.shape().dims();
        // [dims, shifts], each a vec4<u32>; shifts normalized into 0..dim.
        let mut params = [0u32; 8];
        let (dims_out, shifts_out) = params.split_at_mut(4);
        let slots = dims_out.iter_mut().zip(shifts_out.iter_mut());
        for ((dim_out, shift_out), (&dim, &offset)) in slots.zip(dims.iter().zip(&offsets.0)) {
            let dim = u32::try_from(dim).map_err(|_| LumaError::DimensionTooLarge { shape: dims })?;
            *dim_out = dim;
            // Valid arrays have dim > 0; the remainder is in 0..dim and so fits u32.
            let shift = (offset as i64).rem_euclid(i64::from(dim.max(1)));
            *shift_out = u32::try_from(shift)
                .map_err(|_| LumaError::DimensionTooLarge { shape: dims })?;
        }
        let out = Array::<T>::alloc(ctx, self.shape())?;
        let (out_buf, out_bytes) = out.ready()?;
        let core = ctx.builtin(Builtin::Shift, T::wgsl_type())?;
        launch_1d(
            ctx,
            &core,
            len_u32(self.len())?,
            &[
                RawArg::write(out_buf, out_bytes),
                RawArg::read(a, a_bytes),
                RawArg::Uniform(bytemuck::bytes_of(&params)),
            ],
        )?;
        Ok(out)
    }

    // -- Reductions --

    /// Reduces the whole array on the GPU, see [`reduce_array`].
    fn reduce(&self, first: ReduceOp, rest: ReduceOp) -> Result<T> {
        reduce_array(self, T::wgsl_type(), first, rest)
    }

    /// The sum of all elements. Float sums use a tree reduction, whose error grows with the
    /// logarithm of the length rather than the length itself. `u32` sums wrap around.
    /// Synchronizes.
    pub fn sum(&self) -> Result<T> {
        self.reduce(ReduceOp::Sum, ReduceOp::Sum)
    }

    /// The smallest element. Synchronizes.
    pub fn min(&self) -> Result<T> {
        self.reduce(ReduceOp::Min, ReduceOp::Min)
    }

    /// The largest element. Synchronizes.
    pub fn max(&self) -> Result<T> {
        self.reduce(ReduceOp::Max, ReduceOp::Max)
    }
}

/// Reduces the whole array on the GPU with a workgroup tree reduction: every pass combines
/// pairs, `first` for the first pass over the data and `rest` for the passes over partial
/// results, until one element is left, which is read back. If the array is a pending lazy
/// expression, the first pass computes it on the fly instead of reading a materialized buffer.
/// `wgsl` is the WGSL type of the elements; for vectors (which are never lazy) the operations
/// act on every component separately. Synchronizes.
fn reduce_array<T: GpuType>(
    array: &Array<T>,
    wgsl: &'static str,
    first: ReduceOp,
    rest: ReduceOp,
) -> Result<T> {
    let ctx = array.context();
    let elem = std::mem::size_of::<T>() as u64;
    let mut scratch = Scratch::new(ctx);

    let fused = match array.pending() {
        Some(node) => {
            jit::reduce_first_pass(node, first, &mut scratch).map_err(Failure::into_error)?
        }
        None => None,
    };
    let (mut current, mut n, mut op, mut remaining) = match fused {
        Some((partial, groups)) => (partial, groups, rest, groups > 1),
        None => {
            let (input, _) = array.ready().map_err(Failure::into_error)?;
            (input.clone(), len_u32(array.len())?, first, true)
        }
    };
    while remaining {
        let groups = n.div_ceil(REDUCE_BLOCK);
        let out = scratch.take(u64::from(groups) * elem)?;
        let core = ctx.builtin(Builtin::Reduce(op), wgsl)?;
        ctx.launch(
            &core,
            fold_workgroups(ctx, groups)?,
            [n, 1, 1],
            &[
                RawArg::write(&out, u64::from(groups) * elem),
                RawArg::read(&current, u64::from(n) * elem),
                RawArg::Uniform(bytemuck::bytes_of(&[n])),
            ],
        )?;
        current = out;
        n = groups;
        op = rest;
        remaining = groups > 1;
    }

    let bytes = ctx.read_bytes(&current, elem)?;
    bytemuck::try_pod_read_unaligned::<T>(&bytes)
        .map_err(|e| LumaError::Gpu(format!("reduction result has the wrong size: {e}")))
}

/// Per-component reductions of `f32` vector arrays: the same tree reduction as the scalar
/// reductions, run on all components at once, so the accuracy is that of the scalar sum.
macro_rules! vector_reductions {
    ($n:literal, $wgsl:literal) => {
        impl Array<[f32; $n]> {
            /// The sum of all elements, per component. Float sums use a tree reduction, whose
            /// error grows with the logarithm of the length rather than the length itself.
            /// Synchronizes.
            pub fn sum(&self) -> Result<[f32; $n]> {
                reduce_array(self, $wgsl, ReduceOp::Sum, ReduceOp::Sum)
            }

            /// The smallest value of every component. Synchronizes.
            pub fn min(&self) -> Result<[f32; $n]> {
                reduce_array(self, $wgsl, ReduceOp::Min, ReduceOp::Min)
            }

            /// The largest value of every component. Synchronizes.
            pub fn max(&self) -> Result<[f32; $n]> {
                reduce_array(self, $wgsl, ReduceOp::Max, ReduceOp::Max)
            }

            /// The Euclidean (L2) norm of every component taken over the whole array: the square
            /// root of the sum of squares, per component. Synchronizes.
            pub fn norm(&self) -> Result<[f32; $n]> {
                Ok(reduce_array(self, $wgsl, ReduceOp::SumSquares, ReduceOp::Sum)?.map(f32::sqrt))
            }
        }
    };
}

vector_reductions!(2, "vec2<f32>");
vector_reductions!(4, "vec4<f32>");

/// Picks `a[i]` where `mask[i] != 0` and `b[i]` elsewhere. All three arrays must have the same
/// shape (the mask comes from the comparison methods).
#[track_caller]
pub fn select<T: ScalarType>(mask: &Array<u32>, a: &Array<T>, b: &Array<T>) -> Array<T> {
    let location = Location::caller();
    let result = run_select(mask, a, b, location);
    Array::resolve(a.context(), a.shape(), result, "select", location)
}

fn run_select<T: ScalarType>(
    mask: &Array<u32>,
    a: &Array<T>,
    b: &Array<T>,
    location: &'static Location<'static>,
) -> Outcome<Array<T>> {
    let ctx = a.context();
    let a_src = a.src()?;
    let b_src = b.src()?;
    let mask_src = mask.src()?;
    if !ctx.same_as(b.context()) || !ctx.same_as(mask.context()) {
        return Err(LumaError::ContextMismatch.into());
    }
    for other in [b.shape(), mask.shape()] {
        if other != a.shape() {
            return Err(LumaError::ShapeMismatch { lhs: a.shape().dims(), rhs: other.dims() }.into());
        }
    }
    lazy(
        ctx,
        a.shape(),
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
}

impl Array<f32> {
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

    /// The mean of all elements (the sum computed on the GPU, divided on the host).
    /// Synchronizes.
    pub fn mean(&self) -> Result<f32> {
        let sum = self.sum()?;
        Ok((f64::from(sum) / self.len() as f64) as f32)
    }

    /// The Euclidean (L2) norm: the square root of the sum of squares. Synchronizes.
    pub fn norm(&self) -> Result<f32> {
        Ok(self.reduce(ReduceOp::SumSquares, ReduceOp::Sum)?.sqrt())
    }
}

impl Neg for &Array<f32> {
    type Output = Array<f32>;

    #[track_caller]
    fn neg(self) -> Array<f32> {
        let location = Location::caller();
        let result = self.run_unary(UnOp::Neg, location);
        Array::resolve(self.context(), self.shape(), result, "neg", location)
    }
}

impl Neg for Array<f32> {
    type Output = Array<f32>;

    #[track_caller]
    fn neg(self) -> Array<f32> {
        -&self
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

macro_rules! impl_assign_operator {
    ($t:ty, $Trait:ident, $method:ident, $op:expr, $name:literal) => {
        impl $Trait<&Array<$t>> for Array<$t> {
            #[track_caller]
            fn $method(&mut self, rhs: &Array<$t>) {
                self.assign(Rhs::Array(rhs), $op, $name)
            }
        }

        impl $Trait<$t> for Array<$t> {
            #[track_caller]
            fn $method(&mut self, rhs: $t) {
                self.assign(Rhs::Scalar(rhs), $op, $name)
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

impl_operators!(f32);
impl_operators!(u32);
