//! Arrays computed from each element's index inside the kernels that read them: counter-based
//! random numbers, coordinates and ramps. They never occupy memory unless evaluated on their own.

use std::panic::Location;

use crate::array::{Array, Failure};
use crate::context::{default_context, Context};
use crate::error::{LumaError, Result};
use crate::builtin::BinOp;
use crate::gpu_type::{Df32, Numeric, ScalarType};
use crate::jit::{self, ElemTy, Gen, Op, OpKind, Scalar, Src};
use crate::ops::{Outcome, Rhs};
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
    let word = |bits: u32| Src::Scalar(Scalar::u32(bits));
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
        let len = shape.validate()?;
        let stride: usize = dims[..dim].iter().product();
        let fits = |v: usize| u32::try_from(v).map_err(|_| LumaError::InvalidShape(format!("{v} exceeds u32")));
        let (stride32, size32) = (fits(stride)?, fits(size)?);
        let top = (dim + 1..4).all(|d| dims[d] == 1);
        let (kind, args) = if stride32 == 1 && top {
            (Gen::Index, [Src::unused(), Src::unused(), Src::unused()])
        } else if stride32.is_power_of_two() && size32.is_power_of_two() {
            (
                Gen::RangePow2,
                [Src::Scalar(Scalar::u32(stride32.trailing_zeros())), Src::Scalar(Scalar::u32(size32 - 1)), Src::unused()],
            )
        } else {
            (Gen::RangeConst { stride: stride32, size: size32, wide: crate::jit::is_wide(len) }, [Src::unused(), Src::unused(), Src::unused()])
        };
        generated(ctx, shape, kind, args, "range", Location::caller())
    }
}

/// The flat (first dimension fastest) index within `dims` of every element of an array of
/// `dims * tiles`: `Gen::Index` itself if nothing is tiled, else the index of each coordinate
/// taken modulo `dims`, computed from coordinate generators in the same fused expression.
fn tiled_index(ctx: &Context, dims: Shape, tiles: Shape, location: &'static Location<'static>) -> Result<Array<u32>> {
    let (d, t) = (dims.dims(), tiles.dims());
    tiles.validate()?;
    dims.validate()?;
    let out = Shape::new(std::array::from_fn(|k| d[k].saturating_mul(t[k])));
    out.validate()?;
    if t == [1; 4] {
        return generated(ctx, dims, Gen::Index, [Src::unused(), Src::unused(), Src::unused()], "iota", location);
    }
    let mut flat: Option<Array<u32>> = None;
    let mut stride = 1u32;
    for k in 0..4 {
        if d[k] == 1 {
            continue;
        }
        // `d[k]` divides into `out`, which fits u32.
        let extent = d[k] as u32;
        let coordinate = Array::<u32>::range_on(ctx, out, k)?;
        let coordinate = if t[k] > 1 { &coordinate % extent } else { coordinate };
        let term = if stride > 1 { coordinate * stride } else { coordinate };
        flat = Some(match flat {
            Some(sum) => sum + term,
            None => term,
        });
        stride *= extent;
    }
    match flat {
        Some(flat) => Ok(flat),
        // Every dimension is 1: all zeros.
        None => Ok(Array::<u32>::range_on(ctx, out, 0)?.minimum(0u32)),
    }
}

/// The value `value` as a `T` (rounded to what `T` holds).
pub(crate) fn constant<T: ScalarType>(value: f64) -> std::result::Result<T, LumaError> {
    let word = |bytes: &[u8]| T::from_repr(vec![bytemuck::pod_read_unaligned::<T::Repr>(bytes)], 1)[0];
    Ok(match ElemTy::of::<T>()? {
        ElemTy::F32 => word(&(value as f32).to_ne_bytes()),
        ElemTy::F64 => word(&value.to_ne_bytes()),
        ElemTy::Df32 => {
            let Df32 { hi, lo } = Df32::from(value);
            word(&[hi.to_ne_bytes(), lo.to_ne_bytes()].concat())
        }
        ElemTy::U32 => word(&(value as u32).to_ne_bytes()),
        ElemTy::I32 => word(&(value as i32).to_ne_bytes()),
        ElemTy::Bool => word(&u32::from(value != 0.0).to_ne_bytes()),
        ElemTy::U8 => word(&u32::from(value as u8).to_ne_bytes()),
        ElemTy::I8 => word(&u32::from(value as i8 as u8).to_ne_bytes()),
        ElemTy::U16 => word(&u32::from(value as u16).to_ne_bytes()),
        ElemTy::I16 => word(&u32::from(value as i16 as u16).to_ne_bytes()),
        ElemTy::F16 => word(&u32::from(half::f16::from_f64(value).to_bits()).to_ne_bytes()),
        ElemTy::BF16 => word(&u32::from(half::bf16::from_f64(value).to_bits()).to_ne_bytes()),
        ElemTy::C32 => word(&[(value as f32).to_ne_bytes(), 0.0f32.to_ne_bytes()].concat()),
    })
}

/// The `u32` array `flat` as an array of `T` with every value exact: `u32`, `i32` directly, `f32`
/// rounded above `2^24`, `f64` and `Df32` from 16-bit halves (which `f32` holds exactly).
fn exact_from_u32<T: Numeric>(flat: Array<u32>, location: &'static Location<'static>) -> Outcome<Array<T>> {
    Ok(match ElemTy::of::<T>()? {
        ElemTy::F64 | ElemTy::Df32 => {
            let high = (&flat / 65536u32).cast_at::<f32>(location).cast_at::<T>(location);
            let low = (&flat % 65536u32).cast_at::<f32>(location).cast_at::<T>(location);
            high.binary_at(Rhs::Scalar(constant::<T>(65536.0)?), BinOp::Mul, false, "iota", location).binary_at(Rhs::Array(&low), BinOp::Add, false, "iota", location)
        }
        _ => flat.cast_at::<T>(location),
    })
}

impl<T: Numeric> Array<T> {
    /// Counts up the elements of `shape` in memory order (the first dimension fastest) and
    /// repeats that `tiles[k]` times along dimension `k`, like `af::iota(dims, tile)`: the result
    /// has shape `shape * tiles` and element `(x, y, z, w)` is the flat index
    /// `x' + shape[0] * (y' + shape[1] * (z' + shape[2] * w'))` of its coordinates reduced
    /// modulo `shape`. On the default context.
    ///
    /// Generated inside the kernels that read it (no memory unless evaluated on its own).
    /// `u32`, `i32`, `f64` and [`Df32`] count exactly all the way; `f32` rounds above `2^24`.
    /// (`f64` needs a device with 64-bit shader floats, see [`Array::to_f64`].)
    ///
    /// ```
    /// # use luma::Array;
    /// # fn main() -> luma::Result<()> {
    /// let a = Array::<u32>::iota([2, 2], [1, 2])?;
    /// assert_eq!(a.to_vec()?, [0, 1, 2, 3, 0, 1, 2, 3]);
    /// # Ok(()) }
    /// ```
    #[track_caller]
    pub fn iota(shape: impl Into<Shape>, tiles: impl Into<Shape>) -> Result<Self> {
        Self::iota_on(&default_context()?, shape, tiles)
    }

    /// [`Array::iota`] on an explicit context.
    #[track_caller]
    pub fn iota_on(ctx: &Context, shape: impl Into<Shape>, tiles: impl Into<Shape>) -> Result<Self> {
        let location = Location::caller();
        let flat = tiled_index(ctx, shape.into(), tiles.into(), location)?;
        exact_from_u32(flat, location).map_err(Failure::into_error)
    }
}
