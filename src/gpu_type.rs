pub(crate) mod sealed {
    pub trait Sealed {}
}

impl sealed::Sealed for u32 {}
impl sealed::Sealed for i32 {}
impl sealed::Sealed for bool {}
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
///
/// Host values and GPU words are related through the hidden associated type `Repr`: every
/// host-to-GPU and GPU-to-host copy goes through `to_repr` / `from_repr` (borrowed, no copy, for
/// every type but `bool`), and element sizes come from `BYTES`.
///
/// A `bool` is stored as one `u32` word holding 0 or 1 (a custom kernel sees an `array<u32>`).
pub trait GpuType: Zeroable + Copy + Send + Sync + std::fmt::Debug + sealed::Sealed + 'static {
    /// The plain-old-data form of the element that is uploaded to and read back from the GPU.
    #[doc(hidden)]
    type Repr: Pod;

    /// The bytes one element takes on the GPU.
    #[doc(hidden)]
    const BYTES: usize = std::mem::size_of::<Self::Repr>();

    /// The GPU words of `host`: borrowed when the representation is the type itself.
    #[doc(hidden)]
    fn to_repr(host: &[Self]) -> std::borrow::Cow<'_, [Self::Repr]>;

    /// The host values of GPU words read back.
    #[doc(hidden)]
    fn from_repr(repr: Vec<Self::Repr>) -> Vec<Self>;

    /// `host` viewed as GPU words, if the representation is the type itself (no conversion
    /// needed to read into it); `None` otherwise.
    #[doc(hidden)]
    fn repr_mut(host: &mut [Self]) -> Option<&mut [Self::Repr]>;

    /// The GPU word of one value.
    #[doc(hidden)]
    fn into_repr(self) -> Self::Repr;

    /// The value of one GPU word.
    #[doc(hidden)]
    fn one_from_repr(word: Self::Repr) -> Self;
}

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

/// `GpuType` for types that are their own GPU representation.
macro_rules! plain_gpu_type {
    ($($ty:ty),*) => {$(
        impl GpuType for $ty {
            type Repr = $ty;

            fn to_repr(host: &[Self]) -> std::borrow::Cow<'_, [Self::Repr]> {
                std::borrow::Cow::Borrowed(host)
            }

            fn from_repr(repr: Vec<Self::Repr>) -> Vec<Self> {
                repr
            }

            fn repr_mut(host: &mut [Self]) -> Option<&mut [Self::Repr]> {
                Some(host)
            }

            fn into_repr(self) -> Self::Repr {
                self
            }

            fn one_from_repr(word: Self::Repr) -> Self {
                word
            }
        }
    )*};
}

plain_gpu_type!(u32, i32, f32, f64, Df32, [f32; 2], [f32; 4], [u32; 2], [u32; 4]);

impl GpuType for bool {
    type Repr = u32;

    fn to_repr(host: &[Self]) -> std::borrow::Cow<'_, [Self::Repr]> {
        std::borrow::Cow::Owned(host.iter().map(|&b| u32::from(b)).collect())
    }

    fn from_repr(repr: Vec<Self::Repr>) -> Vec<Self> {
        repr.into_iter().map(|w| w != 0).collect()
    }

    fn repr_mut(_host: &mut [Self]) -> Option<&mut [Self::Repr]> {
        None
    }

    fn into_repr(self) -> Self::Repr {
        u32::from(self)
    }

    fn one_from_repr(word: Self::Repr) -> Self {
        word != 0
    }
}

/// The element types of arithmetic: `u32`, `i32`, `f32`, `f64` and [`Df32`]. Sums, running sums,
/// products and the generic reductions exist for these; `bool` has its own counting versions
/// (see [`Array<bool>`](crate::Array#impl-Array%3Cbool%3E)) because NumPy's sum of booleans is an integer.
pub trait Numeric: ScalarType {}

impl Numeric for u32 {}
impl Numeric for i32 {}
impl Numeric for f32 {}
impl Numeric for f64 {}
impl Numeric for Df32 {}

impl ScalarType for u32 {
    fn wgsl_type() -> &'static str {
        "u32"
    }
}

impl ScalarType for i32 {
    fn wgsl_type() -> &'static str {
        "i32"
    }
}

// The WGSL type of the words a `bool` array is stored in (`GpuType::Repr` is `u32`). The built-in
// kernels that move elements without computing on them (copy, take, put, fill, stack, concat)
// and the reductions use it as the storage type; the fused kernels go through the accessors
// `codegen::storage_ty`, `codegen::load` and `codegen::store`.
impl ScalarType for bool {
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

impl ScalarType for Df32 {
    fn wgsl_type() -> &'static str {
        "vec2<f32>"
    }

    fn reduce_type() -> &'static str {
        "df32"
    }
}
