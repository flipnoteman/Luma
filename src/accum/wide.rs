//! The wide element types states are kept in (`Df32`, `f64`), the arithmetic wrapper the merge
//! formulas are written in, and the conversions of the input element types.

use std::ops::{Add, Div, Mul, Sub};

use crate::array::Array;
use crate::context::Context;
use crate::error::{LumaError, Result};
use crate::gpu_type::{Df32, Numeric};
use crate::ops::select;
use crate::reduce::Reduction;

/// The types an accumulator keeps its state in: [`Df32`] (about 48 bits of mantissa, runs on every
/// device) and `f64` (devices with 64-bit shader floats: CUDA and the CPU backend).
///
/// The state of an accumulator is a handful of arrays of this type, one element per position; the
/// merge arithmetic runs in it, so counts stay exact to 2^48 (`Df32`) or 2^53 (`f64`) and means
/// and moments do not lose their low digits to the rounding of `f32` however many batches have
/// been folded in.
pub trait Wide: Numeric {
    #[doc(hidden)]
    const NAME: &'static str;
    #[doc(hidden)]
    fn add(a: &Array<Self>, b: &Array<Self>) -> Array<Self>;
    #[doc(hidden)]
    fn sub(a: &Array<Self>, b: &Array<Self>) -> Array<Self>;
    #[doc(hidden)]
    fn mul(a: &Array<Self>, b: &Array<Self>) -> Array<Self>;
    #[doc(hidden)]
    fn div(a: &Array<Self>, b: &Array<Self>) -> Array<Self>;
    #[doc(hidden)]
    fn addc(a: &Array<Self>, c: f64) -> Array<Self>;
    #[doc(hidden)]
    fn mulc(a: &Array<Self>, c: f64) -> Array<Self>;
    #[doc(hidden)]
    fn sqrt(a: &Array<Self>) -> Array<Self>;
    #[doc(hidden)]
    fn from_f32(a: &Array<f32>) -> Array<Self>;
    #[doc(hidden)]
    fn to_f32(a: &Array<Self>) -> Array<f32>;
    #[doc(hidden)]
    fn from_f64(c: f64) -> Self;
    #[doc(hidden)]
    fn to_f64(self) -> f64;
}

impl Wide for Df32 {
    const NAME: &'static str = "Df32";
    fn add(a: &Array<Self>, b: &Array<Self>) -> Array<Self> {
        a + b
    }
    fn sub(a: &Array<Self>, b: &Array<Self>) -> Array<Self> {
        a - b
    }
    fn mul(a: &Array<Self>, b: &Array<Self>) -> Array<Self> {
        a * b
    }
    fn div(a: &Array<Self>, b: &Array<Self>) -> Array<Self> {
        a / b
    }
    fn addc(a: &Array<Self>, c: f64) -> Array<Self> {
        a + Df32::from(c)
    }
    fn mulc(a: &Array<Self>, c: f64) -> Array<Self> {
        a * Df32::from(c)
    }
    fn sqrt(a: &Array<Self>) -> Array<Self> {
        a.sqrt()
    }
    fn from_f32(a: &Array<f32>) -> Array<Self> {
        a.to_df32()
    }
    fn to_f32(a: &Array<Self>) -> Array<f32> {
        a.to_f32()
    }
    fn from_f64(c: f64) -> Self {
        Df32::from(c)
    }
    fn to_f64(self) -> f64 {
        Df32::to_f64(self)
    }
}

impl Wide for f64 {
    const NAME: &'static str = "f64";
    fn add(a: &Array<Self>, b: &Array<Self>) -> Array<Self> {
        a + b
    }
    fn sub(a: &Array<Self>, b: &Array<Self>) -> Array<Self> {
        a - b
    }
    fn mul(a: &Array<Self>, b: &Array<Self>) -> Array<Self> {
        a * b
    }
    fn div(a: &Array<Self>, b: &Array<Self>) -> Array<Self> {
        a / b
    }
    fn addc(a: &Array<Self>, c: f64) -> Array<Self> {
        a + c
    }
    fn mulc(a: &Array<Self>, c: f64) -> Array<Self> {
        a * c
    }
    fn sqrt(a: &Array<Self>) -> Array<Self> {
        a.sqrt()
    }
    fn from_f32(a: &Array<f32>) -> Array<Self> {
        a.to_f64()
    }
    fn to_f32(a: &Array<Self>) -> Array<f32> {
        a.to_f32()
    }
    fn from_f64(c: f64) -> Self {
        c
    }
    fn to_f64(self) -> f64 {
        self
    }
}

/// `f64` states need a device with 64-bit shader floats.
pub fn check_wide<W: Wide>(ctx: &Context) -> Result<()> {
    if std::any::TypeId::of::<W>() == std::any::TypeId::of::<f64>() && !ctx.features().contains(luma_core::Features::SHADER_F64) {
        return Err(LumaError::Unsupported("an f64 accumulator needs a device with 64-bit shader floats (CUDA, the CPU backend); use Df32".into()));
    }
    Ok(())
}

/// `Df32` and `f64` arrays have no reductions along an axis, only over the whole array, so a
/// precise pass folds the whole batch.
pub fn check_precise_axes<W: Wide>(precise: bool, dims: [usize; 4], axes: &[usize]) -> Result<()> {
    if precise && !super::is_flat(dims, axes) {
        return Err(LumaError::Unsupported(format!("a precise pass folds the whole batch ({} arrays have no reductions along an axis); use the default pass for per-position states", W::NAME)));
    }
    Ok(())
}

/// An array of a [`Wide`] type with the arithmetic the merge formulas use.
#[derive(Clone)]
pub struct V<W: Wide>(pub Array<W>);

impl<W: Wide> V<W> {
    /// A one-element array holding `c`: broadcasts against any shape.
    pub fn constant(ctx: &Context, c: f64) -> Result<V<W>> {
        Ok(V(Array::<W>::full_on(ctx, 1usize, W::from_f64(c))?))
    }

    pub fn zeros(ctx: &Context, shape: crate::Shape) -> Result<V<W>> {
        Ok(V(Array::<W>::zeros_on(ctx, shape)?))
    }

    pub fn filled(ctx: &Context, shape: crate::Shape, c: f64) -> Result<V<W>> {
        Ok(V(Array::<W>::full_on(ctx, shape, W::from_f64(c))?))
    }

    pub fn array(&self) -> &Array<W> {
        &self.0
    }

    pub fn from_f32(a: &Array<f32>) -> V<W> {
        V(W::from_f32(a))
    }

    pub fn narrow(&self) -> Array<f32> {
        W::to_f32(&self.0)
    }

    /// The exact value of a `u32` array: two exact `f32` halves summed in the wide type.
    pub fn from_u32(a: &Array<u32>) -> V<W> {
        let hi = (a >> 16u32).to_f32() * 65536.0;
        let lo = (a & 0xFFFFu32).to_f32();
        V::from_f32(&hi) + V::from_f32(&lo)
    }

    /// The exact value of an `i32` array.
    pub fn from_i32(a: &Array<i32>) -> V<W> {
        let hi = (a >> 16i32).to_f32() * 65536.0;
        let lo = (a & 0xFFFFi32).to_f32();
        V::from_f32(&hi) + V::from_f32(&lo)
    }

    /// `if cond { a } else { b }` elementwise.
    pub fn select(cond: &Array<bool>, a: &V<W>, b: &V<W>) -> V<W> {
        V(select(cond, &a.0, &b.0))
    }

    pub fn gt(&self, other: &V<W>) -> Array<bool> {
        self.0.gt(&other.0)
    }

    pub fn lt(&self, other: &V<W>) -> Array<bool> {
        self.0.lt(&other.0)
    }

    pub fn minimum(&self, other: &V<W>) -> V<W> {
        V(self.0.minimum(&other.0))
    }

    pub fn maximum(&self, other: &V<W>) -> V<W> {
        V(self.0.maximum(&other.0))
    }

    pub fn sqrt(&self) -> V<W> {
        V(W::sqrt(&self.0))
    }

    pub fn sq(&self) -> V<W> {
        self * self
    }
}

macro_rules! v_binary {
    ($Trait:ident, $method:ident, $call:ident) => {
        impl<W: Wide> $Trait<&V<W>> for &V<W> {
            type Output = V<W>;
            fn $method(self, rhs: &V<W>) -> V<W> {
                V(W::$call(&self.0, &rhs.0))
            }
        }
        impl<W: Wide> $Trait<V<W>> for &V<W> {
            type Output = V<W>;
            fn $method(self, rhs: V<W>) -> V<W> {
                V(W::$call(&self.0, &rhs.0))
            }
        }
        impl<W: Wide> $Trait<&V<W>> for V<W> {
            type Output = V<W>;
            fn $method(self, rhs: &V<W>) -> V<W> {
                V(W::$call(&self.0, &rhs.0))
            }
        }
        impl<W: Wide> $Trait<V<W>> for V<W> {
            type Output = V<W>;
            fn $method(self, rhs: V<W>) -> V<W> {
                V(W::$call(&self.0, &rhs.0))
            }
        }
    };
}

v_binary!(Add, add, add);
v_binary!(Sub, sub, sub);
v_binary!(Mul, mul, mul);
v_binary!(Div, div, div);

impl<W: Wide> Add<f64> for &V<W> {
    type Output = V<W>;
    fn add(self, rhs: f64) -> V<W> {
        V(W::addc(&self.0, rhs))
    }
}

impl<W: Wide> Add<f64> for V<W> {
    type Output = V<W>;
    fn add(self, rhs: f64) -> V<W> {
        V(W::addc(&self.0, rhs))
    }
}

impl<W: Wide> Mul<f64> for &V<W> {
    type Output = V<W>;
    fn mul(self, rhs: f64) -> V<W> {
        V(W::mulc(&self.0, rhs))
    }
}

impl<W: Wide> Mul<f64> for V<W> {
    type Output = V<W>;
    fn mul(self, rhs: f64) -> V<W> {
        V(W::mulc(&self.0, rhs))
    }
}

/// The element types a batch can have. `u32`, `i32`, `f32`, `f64` (devices with 64-bit shader
/// floats) and [`Df32`] are implemented; integers are exact in counts, sums, extrema and the
/// difference from the pivot (see the module documentation).
pub trait AccumInput: Numeric {
    /// The value as `f32`.
    #[doc(hidden)]
    fn to_f32(a: &Array<Self>) -> Array<f32>;
    /// `a - pivot` as `f32`, computed in the element type where that is exact (integers, `f64`).
    #[doc(hidden)]
    fn shifted(a: &Array<Self>, pivot: &Array<f32>) -> Array<f32>;
    /// The exact value in the wide type.
    #[doc(hidden)]
    fn widen<W: Wide>(a: &Array<Self>) -> Array<W>;
    /// Pivots of integer types are integers.
    #[doc(hidden)]
    fn tidy_pivot(pivot: Array<f32>) -> Array<f32>;
    /// A one-element array of the largest (`highest`) or the smallest value: the identity of
    /// `min` and `max`.
    #[doc(hidden)]
    fn extreme(ctx: &Context, highest: bool) -> Result<Array<Self>>;
    /// `Some(x == x)` for types with NaN.
    #[doc(hidden)]
    fn not_nan(a: &Array<Self>) -> Option<Array<bool>>;
    /// The index of the smallest (largest) element along `axis` of every line.
    #[doc(hidden)]
    fn argmin_axis(a: &Array<Self>, axis: usize, max: bool) -> Result<Array<u32>>;
    /// The flat index of the smallest (largest) element.
    #[doc(hidden)]
    fn argmin_flat(a: &Array<Self>, max: bool) -> Result<Array<u32>>;
    /// The value as `f64` (exact) and back (rounded to the type).
    #[doc(hidden)]
    fn to_host_f64(self) -> f64;
    #[doc(hidden)]
    fn from_host_f64(v: f64) -> Self;
    /// An array starting at the identity of the minimum (maximum) for scatters.
    #[doc(hidden)]
    fn extreme_target(ctx: &Context, shape: crate::Shape, max: bool) -> Result<Array<Self>>;
    /// Scatter-min or -max of `values` into `target`.
    #[doc(hidden)]
    fn scatter_extreme(target: &mut Array<Self>, idx: &Array<u32>, values: &Array<Self>, max: bool) -> Result<()>;
}

fn unsupported<T>(what: &str) -> Result<T> {
    Err(LumaError::Unsupported(what.to_string()))
}

macro_rules! arg_impls_atomic {
    () => {
        fn argmin_axis(a: &Array<Self>, axis: usize, max: bool) -> Result<Array<u32>> {
            Ok(if max { a.argmax_axis(axis) } else { a.argmin_axis(axis) })
        }
        fn argmin_flat(a: &Array<Self>, max: bool) -> Result<Array<u32>> {
            Ok(if max { a.argmax() } else { a.argmin() })
        }
        fn extreme_target(ctx: &Context, shape: crate::Shape, max: bool) -> Result<Array<Self>> {
            use crate::scatter::AtomicElem;
            let fill = if max { <Self as AtomicElem>::MAX_IDENTITY } else { <Self as AtomicElem>::MIN_IDENTITY };
            Array::<Self>::full_on(ctx, shape, fill)
        }
        fn scatter_extreme(target: &mut Array<Self>, idx: &Array<u32>, values: &Array<Self>, max: bool) -> Result<()> {
            if max { target.put_max(idx, values) } else { target.put_min(idx, values) }
        }
    };
}

impl AccumInput for f32 {
    fn to_host_f64(self) -> f64 {
        self as f64
    }
    fn from_host_f64(v: f64) -> Self {
        v as f32
    }
    fn to_f32(a: &Array<Self>) -> Array<f32> {
        a.clone()
    }
    fn shifted(a: &Array<Self>, pivot: &Array<f32>) -> Array<f32> {
        a - pivot
    }
    fn widen<W: Wide>(a: &Array<Self>) -> Array<W> {
        W::from_f32(a)
    }
    fn tidy_pivot(pivot: Array<f32>) -> Array<f32> {
        pivot
    }
    fn extreme(ctx: &Context, highest: bool) -> Result<Array<Self>> {
        Array::full_on(ctx, 1usize, if highest { f32::INFINITY } else { f32::NEG_INFINITY })
    }
    fn not_nan(a: &Array<Self>) -> Option<Array<bool>> {
        Some(!a.is_nan())
    }
    arg_impls_atomic!();
}

impl AccumInput for u32 {
    fn to_host_f64(self) -> f64 {
        f64::from(self)
    }
    fn from_host_f64(v: f64) -> Self {
        v as u32
    }
    fn to_f32(a: &Array<Self>) -> Array<f32> {
        a.to_f32()
    }
    fn shifted(a: &Array<Self>, pivot: &Array<f32>) -> Array<f32> {
        (a - pivot.to_u32()).to_i32().to_f32()
    }
    fn widen<W: Wide>(a: &Array<Self>) -> Array<W> {
        V::<W>::from_u32(a).0
    }
    fn tidy_pivot(pivot: Array<f32>) -> Array<f32> {
        pivot.round().clamp(0.0, 4_294_967_040.0)
    }
    fn extreme(ctx: &Context, highest: bool) -> Result<Array<Self>> {
        Array::full_on(ctx, 1usize, if highest { u32::MAX } else { 0 })
    }
    fn not_nan(_: &Array<Self>) -> Option<Array<bool>> {
        None
    }
    arg_impls_atomic!();
}

impl AccumInput for i32 {
    fn to_host_f64(self) -> f64 {
        f64::from(self)
    }
    fn from_host_f64(v: f64) -> Self {
        v as i32
    }
    fn to_f32(a: &Array<Self>) -> Array<f32> {
        a.to_f32()
    }
    fn shifted(a: &Array<Self>, pivot: &Array<f32>) -> Array<f32> {
        (a - pivot.to_i32()).to_f32()
    }
    fn widen<W: Wide>(a: &Array<Self>) -> Array<W> {
        V::<W>::from_i32(a).0
    }
    fn tidy_pivot(pivot: Array<f32>) -> Array<f32> {
        pivot.round().clamp(-2_147_483_648.0, 2_147_483_520.0)
    }
    fn extreme(ctx: &Context, highest: bool) -> Result<Array<Self>> {
        Array::full_on(ctx, 1usize, if highest { i32::MAX } else { i32::MIN })
    }
    fn not_nan(_: &Array<Self>) -> Option<Array<bool>> {
        None
    }
    arg_impls_atomic!();
}

impl AccumInput for f64 {
    fn to_host_f64(self) -> f64 {
        self
    }
    fn from_host_f64(v: f64) -> Self {
        v
    }
    fn to_f32(a: &Array<Self>) -> Array<f32> {
        a.to_f32()
    }
    fn shifted(a: &Array<Self>, pivot: &Array<f32>) -> Array<f32> {
        (a - &pivot.to_f64()).to_f32()
    }
    fn widen<W: Wide>(a: &Array<Self>) -> Array<W> {
        // `f64` to `Df32` keeps 48 bits; to `f64` it is the identity.
        wide_from_f64::<W>(a)
    }
    fn tidy_pivot(pivot: Array<f32>) -> Array<f32> {
        pivot
    }
    fn extreme(ctx: &Context, highest: bool) -> Result<Array<Self>> {
        Array::full_on(ctx, 1usize, if highest { f64::INFINITY } else { f64::NEG_INFINITY })
    }
    fn not_nan(a: &Array<Self>) -> Option<Array<bool>> {
        Some(!a.is_nan())
    }
    fn argmin_axis(_: &Array<Self>, _: usize, _: bool) -> Result<Array<u32>> {
        unsupported("argmin / argmax of f64 batches")
    }
    fn argmin_flat(_: &Array<Self>, _: bool) -> Result<Array<u32>> {
        unsupported("argmin / argmax of f64 batches")
    }
    fn extreme_target(_: &Context, _: crate::Shape, _: bool) -> Result<Array<Self>> {
        unsupported("grouped extrema of f64 batches")
    }
    fn scatter_extreme(_: &mut Array<Self>, _: &Array<u32>, _: &Array<Self>, _: bool) -> Result<()> {
        unsupported("grouped extrema of f64 batches")
    }
}

impl AccumInput for Df32 {
    fn to_host_f64(self) -> f64 {
        Df32::to_f64(self)
    }
    fn from_host_f64(v: f64) -> Self {
        Df32::from(v)
    }
    fn to_f32(a: &Array<Self>) -> Array<f32> {
        a.to_f32()
    }
    fn shifted(a: &Array<Self>, pivot: &Array<f32>) -> Array<f32> {
        (a - &pivot.to_df32()).to_f32()
    }
    fn widen<W: Wide>(a: &Array<Self>) -> Array<W> {
        wide_from_df32::<W>(a)
    }
    fn tidy_pivot(pivot: Array<f32>) -> Array<f32> {
        pivot
    }
    fn extreme(ctx: &Context, highest: bool) -> Result<Array<Self>> {
        Array::full_on(ctx, 1usize, Df32 { hi: if highest { f32::INFINITY } else { f32::NEG_INFINITY }, lo: 0.0 })
    }
    fn not_nan(a: &Array<Self>) -> Option<Array<bool>> {
        Some(!a.is_nan())
    }
    fn argmin_axis(_: &Array<Self>, _: usize, _: bool) -> Result<Array<u32>> {
        unsupported("argmin / argmax of Df32 batches")
    }
    fn argmin_flat(_: &Array<Self>, _: bool) -> Result<Array<u32>> {
        unsupported("argmin / argmax of Df32 batches")
    }
    fn extreme_target(_: &Context, _: crate::Shape, _: bool) -> Result<Array<Self>> {
        unsupported("grouped extrema of Df32 batches")
    }
    fn scatter_extreme(_: &mut Array<Self>, _: &Array<u32>, _: &Array<Self>, _: bool) -> Result<()> {
        unsupported("grouped extrema of Df32 batches")
    }
}

/// `f64` batch to the wide type `W` (a cast through `TypeId`: both wide types are concrete).
fn wide_from_f64<W: Wide>(a: &Array<f64>) -> Array<W> {
    use std::any::Any;
    let boxed: Box<dyn Any> = if std::any::TypeId::of::<W>() == std::any::TypeId::of::<f64>() { Box::new(a.clone()) } else { Box::new(a.to_df32()) };
    *boxed.downcast::<Array<W>>().expect("a wide type is f64 or Df32")
}

fn wide_from_df32<W: Wide>(a: &Array<Df32>) -> Array<W> {
    use std::any::Any;
    let boxed: Box<dyn Any> = if std::any::TypeId::of::<W>() == std::any::TypeId::of::<f64>() { Box::new(a.to_f64()) } else { Box::new(a.clone()) };
    *boxed.downcast::<Array<W>>().expect("a wide type is f64 or Df32")
}

/// Reductions along several axes in sequence: `ops` over the first axis in one pass over `x`,
/// then each result over the next axis.
pub fn reduce_axes<T: Numeric>(x: &Array<T>, axes: &[usize], ops: &[Reduction]) -> Vec<Array<T>> {
    fn first<T: Numeric>(x: &Array<T>, axis: usize, ops: &[Reduction]) -> Vec<Array<T>> {
        match *ops {
            [a] => x.reduce_axis(axis, [a]).into(),
            [a, b] => x.reduce_axis(axis, [a, b]).into(),
            [a, b, c] => x.reduce_axis(axis, [a, b, c]).into(),
            [a, b, c, d] => x.reduce_axis(axis, [a, b, c, d]).into(),
            _ => unreachable!("one to four reductions"),
        }
    }
    let mut out: Vec<Array<T>> = Vec::with_capacity(ops.len());
    for chunk in ops.chunks(4) {
        out.extend(first(x, axes[0], chunk));
    }
    // The later axes combine the partial results: sums of squares add up like sums.
    for &axis in &axes[1..] {
        for (array, &op) in out.iter_mut().zip(ops) {
            let combine = if op == Reduction::SumOfSquares { Reduction::Sum } else { op };
            *array = array.reduce_axis(axis, [combine]).into_iter().next().expect("one reduction");
        }
    }
    out
}

/// Reductions over the whole array in one pass: one-element arrays.
pub fn reduce_flat<T: Numeric>(x: &Array<T>, ops: &[Reduction]) -> Vec<Array<T>> {
    let mut out = Vec::with_capacity(ops.len());
    for chunk in ops.chunks(4) {
        let packed = match *chunk {
            [a] => x.reduce([a]),
            [a, b] => x.reduce([a, b]),
            [a, b, c] => x.reduce([a, b, c]),
            [a, b, c, d] => x.reduce([a, b, c, d]),
            _ => unreachable!("one to four reductions"),
        };
        for j in 0..chunk.len() {
            out.push(packed.slice(0, j..j + 1));
        }
    }
    out
}
