//! Operators, math functions and reductions on arrays.
//!
//! Elementwise operations (operators, math functions, comparisons, `select`, `powf`, `clamp`,
//! shifts) are recorded as lazy expressions and fused into one generated kernel when their result
//! is needed (see [`crate::jit`]); reductions fold the expression behind their input into their
//! first pass. In-place operators, fills and the later reduction passes are plain
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
use crate::builtin::{BinOp, Builtin, CmpOp, ReduceList, ReduceOp, UnOp, MAX_REDUCTIONS};
use crate::context::{Context, RawArg, Scratch};
use crate::error::{LumaError, Result};
use crate::gpu_type::{GpuType, ScalarType};
use crate::jit::{self, ElemTy, Op, OpKind, Scalar, ShiftKind, Src};
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
    ///
    /// The shift itself is lazy: it fuses into the expression that uses it as a shifted read,
    /// so a stencil like `u.shift(1) + u.shift(-1) - &u * 2.0` is one kernel reading `u` once. A
    /// small lazy expression is not even evaluated: the kernel recomputes it at the shifted
    /// element (`(&a * &b).shift(1)` reads `a` and `b` shifted); a larger one is evaluated first.
    #[track_caller]
    pub fn shift(&self, offsets: impl Into<Offsets>) -> Array<T> {
        let location = Location::caller();
        let result = self.run_shift(offsets.into(), location);
        Array::resolve(self.context(), self.shape(), result, "shift", location)
    }

    fn run_shift(&self, offsets: Offsets, location: &'static Location<'static>) -> Outcome<Array<T>> {
        let operand = match self.pending() {
            Some(node) if jit::recompute_shifted(node) => Src::Node(Arc::clone(node)),
            _ => Src::Buffer(Arc::clone(self.storage()?)),
        };
        let dims = self.shape().dims();
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
        let (kind, amount) = if linear {
            let stride: u32 = dims32[..top].iter().product();
            (ShiftKind::Linear, Src::Scalar(Scalar { ty: ElemTy::U32, bits: shifts[top] * stride }))
        } else {
            (ShiftKind::Dims { dims: dims32, shifts }, Src::unused())
        };
        lazy(
            self.context(),
            self.shape(),
            Op::new(OpKind::Shift(kind), ElemTy::of::<T>()?),
            [operand, amount, Src::unused()],
            "shift",
            location,
        )
    }

    // -- Reductions --

    /// The sum of all elements. Float sums use a tree reduction, whose error grows with the
    /// logarithm of the length rather than the length itself. `u32` sums wrap around.
    /// Synchronizes.
    pub fn sum(&self) -> Result<T> {
        Ok(self.reduce([Reduction::Sum])?[0])
    }

    /// The smallest element. Synchronizes.
    pub fn min(&self) -> Result<T> {
        Ok(self.reduce([Reduction::Min])?[0])
    }

    /// The largest element. Synchronizes.
    pub fn max(&self) -> Result<T> {
        Ok(self.reduce([Reduction::Max])?[0])
    }

    /// Several reductions of the array in one pass over it (up to four), for example
    /// `let [lo, hi] = x.reduce([Reduction::Min, Reduction::Max])?;`. Reading the data once
    /// instead of once per reduction is what makes it faster than separate calls on large arrays.
    /// Synchronizes.
    pub fn reduce<const N: usize>(&self, ops: [Reduction; N]) -> Result<[T; N]> {
        reduce_n(self, T::wgsl_type(), ops)
    }
}

/// Evaluates several arrays and waits for them, like [`Array::eval`] on each, but computes lazy
/// arrays of the same length together: one fused kernel with an output per array (as many as
/// the device's bindings allow), so inputs they share are read once, and an array that another
/// one uses is computed once. `let (call, put) = ...; eval_all(&[&call, &put])?` prices both
/// options in one pass over the market data.
pub fn eval_all<T: GpuType>(arrays: &[&Array<T>]) -> Result<()> {
    let Some(first) = arrays.first() else { return Ok(()) };
    let ctx = first.context();
    if arrays.iter().any(|array| !ctx.same_as(array.context())) {
        return Err(LumaError::ContextMismatch);
    }
    let mut pending: Vec<_> = arrays.iter().filter_map(|array| array.pending().cloned()).collect();
    while let Some(node) = pending.first() {
        let len = node.len();
        let (same, rest): (Vec<_>, Vec<_>) = pending.into_iter().partition(|node| node.len() == len);
        jit::evaluate_group(&same);
        pending = rest;
    }
    for array in arrays {
        array.ready().map_err(Failure::into_error)?;
    }
    ctx.sync()
}

/// A reduction for [`Array::reduce`].
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Reduction {
    Sum,
    Min,
    Max,
    /// The sum of the squared elements.
    SumOfSquares,
}

impl Reduction {
    fn op(self) -> ReduceOp {
        match self {
            Reduction::Sum => ReduceOp::Sum,
            Reduction::Min => ReduceOp::Min,
            Reduction::Max => ReduceOp::Max,
            Reduction::SumOfSquares => ReduceOp::SumSquares,
        }
    }
}

fn reduce_n<T: GpuType, const N: usize>(array: &Array<T>, wgsl: &'static str, ops: [Reduction; N]) -> Result<[T; N]> {
    const { assert!(N >= 1 && N <= MAX_REDUCTIONS, "Array::reduce takes one to four reductions") };
    let values = reduce_array(array, wgsl, ReduceList::new(&ops.map(Reduction::op)))?;
    let mut out = [T::zeroed(); N];
    for (slot, value) in out.iter_mut().zip(values) {
        *slot = value;
    }
    Ok(out)
}

/// Reduces the whole array on the GPU with a workgroup tree reduction, once per op of `ops`:
/// every pass combines pairs, until one element per op is left, which are read back together.
/// The first pass reads the data once for all ops; if the array is a pending lazy expression, it
/// computes it on the fly instead of reading a materialized buffer. `wgsl` is the WGSL type of
/// the elements; for vectors (which are never lazy) the operations act on every component
/// separately. Synchronizes.
fn reduce_array<T: GpuType>(array: &Array<T>, wgsl: &'static str, ops: ReduceList) -> Result<Vec<T>> {
    let ctx = array.context();
    let elem = std::mem::size_of::<T>() as u64;
    let count = ops.as_slice().len() as u64;
    let mut scratch = Scratch::new(ctx);

    let fused = match array.pending() {
        Some(node) => jit::reduce_first_pass(node, ops, &mut scratch).map_err(Failure::into_error)?,
        None => None,
    };
    // `current` holds `n` elements per op, the ops' segments one after the other once
    // `segmented`.
    let (mut current, mut n, mut segmented) = match fused {
        Some((partial, groups)) => (partial, groups, true),
        None => {
            let (input, _) = array.ready().map_err(Failure::into_error)?;
            (input.clone(), len_u32(array.len())?, false)
        }
    };
    while !segmented || n > 1 {
        let groups = n.div_ceil(ctx.tuning().reduce_block());
        let list = if segmented { ops.rest() } else { ops };
        let in_bytes = u64::from(n) * elem * if segmented { count } else { 1 };
        let out = scratch.take(u64::from(groups) * elem * count)?;
        let core = ctx.builtin(Builtin::Reduce(list, segmented), wgsl)?;
        ctx.launch(
            &core,
            fold_workgroups(ctx, groups)?,
            [n, 1, 1],
            &[
                RawArg::write(&out, u64::from(groups) * elem * count),
                RawArg::read(&current, in_bytes),
                RawArg::Uniform(bytemuck::bytes_of(&[n])),
            ],
        )?;
        current = out;
        n = groups;
        segmented = true;
    }

    let values = ctx.read::<T>(&current, elem * count)?;
    if values.len() as u64 != count {
        return Err(LumaError::Gpu("reduction results are missing".into()));
    }
    Ok(values)
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
                Ok(self.reduce([Reduction::Sum])?[0])
            }

            /// The smallest value of every component. Synchronizes.
            pub fn min(&self) -> Result<[f32; $n]> {
                Ok(self.reduce([Reduction::Min])?[0])
            }

            /// The largest value of every component. Synchronizes.
            pub fn max(&self) -> Result<[f32; $n]> {
                Ok(self.reduce([Reduction::Max])?[0])
            }

            /// The Euclidean (L2) norm of every component taken over the whole array: the square
            /// root of the sum of squares, per component. Synchronizes.
            pub fn norm(&self) -> Result<[f32; $n]> {
                Ok(self.reduce([Reduction::SumOfSquares])?[0].map(f32::sqrt))
            }

            /// Several per-component reductions in one pass over the array (up to four), like
            /// [`Array::reduce`]. Synchronizes.
            pub fn reduce<const N: usize>(&self, ops: [Reduction; N]) -> Result<[[f32; $n]; N]> {
                reduce_n(self, $wgsl, ops)
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
        Ok(self.reduce([Reduction::SumOfSquares])?[0].sqrt())
    }

    /// `steps` applications of `step`: `u = step(u)`, starting from this array, like a loop but
    /// run time-blocked where possible. When `step` is a stencil — one elementwise expression of
    /// its argument, shifts of it by a few elements (circular, as [`Array::shift`]), scalars and
    /// other arrays of the same shape that do not change — each launch loads a tile into
    /// workgroup memory and runs several steps on it, so the data passes through memory once per
    /// several steps instead of once per step. Any other `step` simply runs in a loop.
    ///
    /// `step` is called with a placeholder first to inspect the expression it builds; it must
    /// compute its result from its argument and not keep it.
    ///
    /// ```no_run
    /// # let u0 = luma::Array::<f32>::zeros([512, 512])?;
    /// let alpha = 0.2;
    /// let u = u0.iterate(100, |u| {
    ///     let lap = u.shift([1, 0]) + u.shift([-1, 0]) + u.shift([0, 1]) + u.shift([0, -1]) - u * 4.0;
    ///     u + lap * alpha
    /// });
    /// # Ok::<(), luma::LumaError>(())
    /// ```
    #[track_caller]
    pub fn iterate(&self, steps: usize, step: impl Fn(&Array<f32>) -> Array<f32>) -> Array<f32> {
        let location = Location::caller();
        let result = self.run_iterate(steps, step);
        Array::resolve(self.context(), self.shape(), result, "iterate", location)
    }

    fn run_iterate(&self, steps: usize, step: impl Fn(&Array<f32>) -> Array<f32>) -> Outcome<Array<f32>> {
        if let Some(error) = self.error() {
            return Err(Failure::Inherited(Arc::new(error.clone())));
        }
        if steps == 0 {
            return Ok(self * 1.0);
        }
        let probe = Array::<f32>::alloc(self.context(), self.shape())?;
        let stencil = jit::stencil_plan(&probe, &step(&probe));
        match stencil {
            Some(stencil) => stencil.run(self, steps),
            None => {
                let mut u = step(self);
                for _ in 1..steps {
                    u = step(&u);
                }
                u.check()?;
                Ok(u)
            }
        }
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
