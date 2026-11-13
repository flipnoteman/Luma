pub(crate) mod sealed {
    pub trait Sealed {}
}

impl sealed::Sealed for u32 {}
impl sealed::Sealed for f32 {}
impl sealed::Sealed for f64 {}
impl sealed::Sealed for Df32 {}
impl sealed::Sealed for [f32; 2] {}
impl sealed::Sealed for [f32; 4] {}
impl sealed::Sealed for [u32; 2] {}
impl sealed::Sealed for [u32; 4] {}

use bytemuck::{Pod, Zeroable};

/// Element types that can be stored in an [`Array`](crate::Array) and read by custom kernels.
///
/// Besides the scalars `u32` and `f32`, arrays can hold `[f32; 2]`, `[f32; 4]`, `[u32; 2]` and
/// `[u32; 4]`, whose layout matches WGSL's `vec2`/`vec4` (there is deliberately no `[f32; 3]`:
/// `vec3` is 16-byte aligned in WGSL, which a plain 12-byte array is not), and the 8-byte
/// scalars `f64` and [`Df32`] (see [`crate::Array::to_f64`] and [`crate::Array::to_df32`]).
///
/// The built-in math only exists for [`ScalarType`]s; `[f32; 2]` and `[f32; 4]` arrays also have
/// per-component reductions (`sum`, `min`, `max`, `norm`).
pub trait GpuType: Pod + std::fmt::Debug + sealed::Sealed {}

/// Element types the built-in operators, math functions and reductions are implemented for.
pub trait ScalarType: GpuType {
    /// The WGSL name of the type, used to generate the shaders of the built-in operations.
    #[doc(hidden)]
    fn wgsl_type() -> &'static str;

    /// The type name the reduction kernels are keyed by: the WGSL type, except for types whose
    /// arithmetic is a library (`"df32"` for [`Df32`]).
    #[doc(hidden)]
    fn reduce_type() -> &'static str {
        Self::wgsl_type()
    }
}

impl GpuType for u32 {}
impl GpuType for f32 {}
impl GpuType for f64 {}
impl GpuType for [f32; 2] {}
impl GpuType for [f32; 4] {}
impl GpuType for [u32; 2] {}
impl GpuType for [u32; 4] {}

impl ScalarType for u32 {
    fn wgsl_type() -> &'static str {
        "u32"
    }
}

impl ScalarType for f32 {
    fn wgsl_type() -> &'static str {
        "f32"
    }
}

impl ScalarType for f64 {
    fn wgsl_type() -> &'static str {
        "f64"
    }
}

/// A double-float: the unevaluated sum `hi + lo` of two `f32`, about 48 bits of mantissa,
/// computed with `f32` arithmetic only (so it needs no 64-bit float support). Arrays of `Df32`
/// have `+ - * /`, `sqrt`, `abs`, comparisons, `minimum`/`maximum`, `select`, views, reductions
/// and the casts [`crate::Array::to_f32`] / `to_f64`. See [`crate::Array::sum_compensated`].
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct Df32 {
    /// The leading part.
    pub hi: f32,
    /// What `hi` leaves out, at most half an ulp of `hi`.
    pub lo: f32,
}

// SAFETY: `#[repr(C)]` over two `f32`: no padding, every bit pattern valid.
unsafe impl Zeroable for Df32 {}
// SAFETY: see above.
unsafe impl Pod for Df32 {}

impl From<f32> for Df32 {
    fn from(value: f32) -> Self {
        Df32 { hi: value, lo: 0.0 }
    }
}

impl From<f64> for Df32 {
    fn from(value: f64) -> Self {
        let hi = value as f32;
        Df32 { hi, lo: (value - f64::from(hi)) as f32 }
    }
}

impl Df32 {
    /// The value as an `f64` (exact: `hi + lo` is representable).
    pub fn to_f64(self) -> f64 {
        f64::from(self.hi) + f64::from(self.lo)
    }
}

impl GpuType for Df32 {}

impl ScalarType for Df32 {
    fn wgsl_type() -> &'static str {
        "vec2<f32>"
    }

    fn reduce_type() -> &'static str {
        "df32"
    }
}
