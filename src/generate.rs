//! Arrays computed from each element's index inside the kernels that read them: counter-based
//! random numbers, coordinates and ramps. They never occupy memory unless evaluated on their own.

use std::panic::Location;

use crate::array::{Array, Failure};
use crate::context::{default_context, Context};
use crate::error::{LumaError, Result};
use crate::gpu_type::ScalarType;
use crate::jit::{self, ElemTy, Gen, Op, OpKind, Scalar, Src};
use crate::shape::Shape;

fn generated<T: ScalarType>(
    ctx: &Context,
    shape: Shape,
    kind: Gen,
    args: [Src; 3],
    name: &'static str,
    location: &'static Location<'static>,
) -> Result<Array<T>> {
    let len = shape.validate()?;
    let op = Op::new(OpKind::Generate(kind), ElemTy::of::<T>()?);
    let node = jit::create(ctx, len, op, args, name, location).map_err(Failure::into_error)?;
    Ok(Array::from_node(ctx, shape, node))
}

fn seed_args(seed: u64) -> [Src; 3] {
    let word = |bits: u32| Src::Scalar(Scalar { ty: ElemTy::U32, bits });
    [word(seed as u32), word((seed >> 32) as u32), Src::unused()]
}

impl Array<f32> {
    /// Uniformly distributed values in `[0, 1)` with 24 random bits each, on the default context.
    /// Element `i` depends only on `seed` and `i` (Threefry-2x32-20 of the counter `(i, 0)`), so
    /// the values are the same however the array is fused, split or shifted.
    #[track_caller]
    pub fn uniform(shape: impl Into<Shape>, seed: u64) -> Result<Self> {
        Self::uniform_on(&default_context()?, shape, seed)
    }

    /// [`Array::uniform`] on an explicit context.
    #[track_caller]
    pub fn uniform_on(ctx: &Context, shape: impl Into<Shape>, seed: u64) -> Result<Self> {
        generated(ctx, shape.into(), Gen::Uniform, seed_args(seed), "uniform", Location::caller())
    }

    /// Standard normally distributed values (Box-Muller on the counter-based generator of
    /// [`Array::uniform`]), on the default context.
    #[track_caller]
    pub fn normal(shape: impl Into<Shape>, seed: u64) -> Result<Self> {
        Self::normal_on(&default_context()?, shape, seed)
    }

    /// [`Array::normal`] on an explicit context.
    #[track_caller]
    pub fn normal_on(ctx: &Context, shape: impl Into<Shape>, seed: u64) -> Result<Self> {
        generated(ctx, shape.into(), Gen::Normal, seed_args(seed), "normal", Location::caller())
    }

    /// `n` evenly spaced values from `start` to `end` (both included up to rounding), on the
    /// default context.
    #[track_caller]
    pub fn linspace(start: f32, end: f32, n: usize) -> Result<Self> {
        Self::linspace_on(&default_context()?, start, end, n)
    }

    /// [`Array::linspace`] on an explicit context.
    #[track_caller]
    pub fn linspace_on(ctx: &Context, start: f32, end: f32, n: usize) -> Result<Self> {
        let step = if n > 1 { (end - start) / (n - 1) as f32 } else { 0.0 };
        let args = [Src::Scalar(Scalar::f32(start)), Src::Scalar(Scalar::f32(step)), Src::unused()];
        generated(ctx, Shape::from(n), Gen::Linspace, args, "linspace", Location::caller())
    }
}

impl Array<u32> {
    /// 32 random bits per element from the generator of [`Array::uniform`], on the default
    /// context.
    #[track_caller]
    pub fn random_bits(shape: impl Into<Shape>, seed: u64) -> Result<Self> {
        Self::random_bits_on(&default_context()?, shape, seed)
    }

    /// [`Array::random_bits`] on an explicit context.
    #[track_caller]
    pub fn random_bits_on(ctx: &Context, shape: impl Into<Shape>, seed: u64) -> Result<Self> {
        generated(ctx, shape.into(), Gen::Bits, seed_args(seed), "random_bits", Location::caller())
    }
}

impl<T: ScalarType> Array<T> {
    /// Each element's index along dimension `dim` (0 varies fastest), on the default context.
    #[track_caller]
    pub fn range(shape: impl Into<Shape>, dim: usize) -> Result<Self> {
        Self::range_on(&default_context()?, shape, dim)
    }

    /// [`Array::range`] on an explicit context.
    #[track_caller]
    pub fn range_on(ctx: &Context, shape: impl Into<Shape>, dim: usize) -> Result<Self> {
        let shape = shape.into();
        let dims = shape.dims();
        let Some(&size) = dims.get(dim) else {
            return Err(LumaError::InvalidShape(format!("range along dimension {dim} of a 4-dimensional shape")));
        };
        let stride: usize = dims[..dim].iter().product();
        let fits = |v: usize| u32::try_from(v).map_err(|_| LumaError::InvalidShape(format!("{v} exceeds u32")));
        let kind = Gen::Range { stride: fits(stride)?, size: fits(size)? };
        let args = [Src::unused(), Src::unused(), Src::unused()];
        generated(ctx, shape, kind, args, "range", Location::caller())
    }
}
