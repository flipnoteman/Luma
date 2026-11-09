//! Operators, math functions and reductions on arrays.
//!
//! Every operation is a built-in kernel (see [`crate::builtin`]) recorded into the context's
//! batch. Methods that return arrays never fail loudly: they return poisoned arrays (see
//! [`Array`]). The `run_*` functions do the work and return the failure; the public methods turn
//! it into a poisoned array tagged with the caller's location, or into a plain `Err` for the
//! `try_*` methods.

use std::ops::{
    Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign,
};
use std::panic::Location;
use std::sync::Arc;

use crate::array::{Array, Failure};
use crate::builtin::{
    BinOp, Builtin, CmpOp, Operands, ReduceOp, UnOp, REDUCE_BLOCK,
};
use crate::context::{Context, RawArg, Scratch};
use crate::error::{LumaError, Result};
use crate::gpu_type::{GpuType, ScalarType};
use crate::kernel::KernelCore;
use crate::shape::Offsets;

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
fn fold_workgroups(ctx: &Context, count: u32) -> Result<[u32; 3]> {
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

impl<T: ScalarType> Array<T> {
    // -- Kernel runners --

    /// `out = f(self)` for a unary kernel.
    fn run_unary(&self, op: Builtin) -> Outcome<Array<T>> {
        let (a, a_bytes) = self.ready()?;
        let out = Array::<T>::alloc(self.context(), self.shape())?;
        let (out_buf, out_bytes) = out.ready()?;
        let core = self.context().builtin(op, T::wgsl_type())?;
        launch_1d(
            self.context(),
            &core,
            len_u32(self.len())?,
            &[RawArg::write(out_buf, out_bytes), RawArg::read(a, a_bytes)],
        )?;
        Ok(out)
    }

    /// `out = self op rhs` (or `rhs op self` if `scalar_left`, which only exists for scalars),
    /// producing elements of type `O`.
    fn run_binary<O: GpuType>(
        &self,
        rhs: Rhs<'_, T>,
        scalar_left: bool,
        kernel: impl Fn(Operands) -> Builtin,
    ) -> Outcome<Array<O>> {
        let ctx = self.context();
        let (a, a_bytes) = self.ready()?;
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
                let out = Array::<O>::alloc(ctx, self.shape())?;
                let (out_buf, out_bytes) = out.ready()?;
                let core = ctx.builtin(kernel(Operands::Arrays), T::wgsl_type())?;
                launch_1d(
                    ctx,
                    &core,
                    len_u32(self.len())?,
                    &[
                        RawArg::write(out_buf, out_bytes),
                        RawArg::read(a, a_bytes),
                        RawArg::read(b, b_bytes),
                    ],
                )?;
                Ok(out)
            }
            Rhs::Scalar(scalar) => {
                let out = Array::<O>::alloc(ctx, self.shape())?;
                let (out_buf, out_bytes) = out.ready()?;
                let operands =
                    if scalar_left { Operands::ScalarArray } else { Operands::ArrayScalar };
                let core = ctx.builtin(kernel(operands), T::wgsl_type())?;
                launch_1d(
                    ctx,
                    &core,
                    len_u32(self.len())?,
                    &[
                        RawArg::write(out_buf, out_bytes),
                        RawArg::read(a, a_bytes),
                        RawArg::Uniform(bytemuck::bytes_of(&scalar)),
                    ],
                )?;
                Ok(out)
            }
        }
    }

    /// `self = self op rhs`.
    fn run_in_place(&self, rhs: Rhs<'_, T>, op: BinOp) -> Outcome<()> {
        let ctx = self.context();
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
                let core = ctx.builtin(Builtin::InPlaceArray(op), T::wgsl_type())?;
                launch_1d(
                    ctx,
                    &core,
                    n,
                    &[RawArg::write(a, a_bytes), RawArg::read(b, b_bytes)],
                )?;
            }
            Rhs::Scalar(scalar) => {
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
        let result = self.run_binary(rhs, scalar_left, |operands| Builtin::Binary(op, operands));
        Array::resolve(self.context(), self.shape(), result, name, location)
    }

    #[track_caller]
    fn compare(&self, rhs: Rhs<'_, T>, op: CmpOp, name: &'static str) -> Array<u32> {
        let location = Location::caller();
        let result = self.run_binary(rhs, false, |operands| Builtin::Compare(op, operands));
        Array::resolve(self.context(), self.shape(), result, name, location)
    }

    fn try_binary(&self, rhs: Rhs<'_, T>, op: BinOp) -> Result<Array<T>> {
        self.run_binary(rhs, false, |operands| Builtin::Binary(op, operands))
            .map_err(Failure::into_error)
    }

    #[track_caller]
    pub(crate) fn assign(&mut self, rhs: Rhs<'_, T>, op: BinOp, name: &'static str) {
        let location = Location::caller();
        match self.run_in_place(rhs, op) {
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
    pub fn try_add(&self, rhs: impl Operand<T>) -> Result<Array<T>> {
        self.try_binary(rhs.rhs(), BinOp::Add)
    }

    /// `self - rhs`, failing immediately instead of returning a poisoned array.
    pub fn try_sub(&self, rhs: impl Operand<T>) -> Result<Array<T>> {
        self.try_binary(rhs.rhs(), BinOp::Sub)
    }

    /// `self * rhs`, failing immediately instead of returning a poisoned array.
    pub fn try_mul(&self, rhs: impl Operand<T>) -> Result<Array<T>> {
        self.try_binary(rhs.rhs(), BinOp::Mul)
    }

    /// `self / rhs`, failing immediately instead of returning a poisoned array.
    pub fn try_div(&self, rhs: impl Operand<T>) -> Result<Array<T>> {
        self.try_binary(rhs.rhs(), BinOp::Div)
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
/// results, until one element is left, which is read back. `wgsl` is the WGSL type of the
/// elements; for vectors the operations act on every component separately. Synchronizes.
fn reduce_array<T: GpuType>(
    array: &Array<T>,
    wgsl: &'static str,
    first: ReduceOp,
    rest: ReduceOp,
) -> Result<T> {
    let ctx = array.context();
    let (input, _) = array.ready().map_err(Failure::into_error)?;
    let elem = std::mem::size_of::<T>() as u64;
    let mut scratch = Scratch::new(ctx);

    let mut current = input.clone();
    let mut n = len_u32(array.len())?;
    let mut op = first;
    loop {
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
        if groups == 1 {
            break;
        }
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
    let result = run_select(mask, a, b);
    Array::resolve(a.context(), a.shape(), result, "select", location)
}

fn run_select<T: ScalarType>(mask: &Array<u32>, a: &Array<T>, b: &Array<T>) -> Outcome<Array<T>> {
    let ctx = a.context();
    let (a_buf, a_bytes) = a.ready()?;
    let (b_buf, b_bytes) = b.ready()?;
    let (m_buf, m_bytes) = mask.ready()?;
    if !ctx.same_as(b.context()) || !ctx.same_as(mask.context()) {
        return Err(LumaError::ContextMismatch.into());
    }
    for other in [b.shape(), mask.shape()] {
        if other != a.shape() {
            return Err(LumaError::ShapeMismatch { lhs: a.shape().dims(), rhs: other.dims() }.into());
        }
    }
    let out = Array::<T>::alloc(ctx, a.shape())?;
    let (out_buf, out_bytes) = out.ready()?;
    let core = ctx.builtin(Builtin::Select, T::wgsl_type())?;
    launch_1d(
        ctx,
        &core,
        len_u32(a.len())?,
        &[
            RawArg::write(out_buf, out_bytes),
            RawArg::read(m_buf, m_bytes),
            RawArg::read(a_buf, a_bytes),
            RawArg::read(b_buf, b_bytes),
        ],
    )?;
    Ok(out)
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
                    let result = self.run_unary(Builtin::Unary(op));
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
        let result = self.run_with_params(Builtin::Powf, bytemuck::bytes_of(&exponent));
        Array::resolve(self.context(), self.shape(), result, "powf", location)
    }

    /// Elementwise clamp into `lo..=hi`.
    #[track_caller]
    pub fn clamp(&self, lo: f32, hi: f32) -> Array<f32> {
        let location = Location::caller();
        let result = self.run_with_params(Builtin::Clamp, bytemuck::bytes_of(&[lo, hi]));
        Array::resolve(self.context(), self.shape(), result, "clamp", location)
    }

    /// `out = f(self)` for a kernel taking one uniform.
    fn run_with_params(&self, op: Builtin, params: &[u8]) -> Outcome<Array<f32>> {
        let ctx = self.context();
        let (a, a_bytes) = self.ready()?;
        let out = Array::<f32>::alloc(ctx, self.shape())?;
        let (out_buf, out_bytes) = out.ready()?;
        let core = ctx.builtin(op, f32::wgsl_type())?;
        launch_1d(
            ctx,
            &core,
            len_u32(self.len())?,
            &[RawArg::write(out_buf, out_bytes), RawArg::read(a, a_bytes), RawArg::Uniform(params)],
        )?;
        Ok(out)
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
        let result = self.run_unary(Builtin::Unary(UnOp::Neg));
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
