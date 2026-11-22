pub(crate) mod sealed {
    pub trait Sealed {}
}

impl sealed::Sealed for u32 {}
impl sealed::Sealed for i32 {}
impl sealed::Sealed for bool {}
impl sealed::Sealed for u8 {}
impl sealed::Sealed for i8 {}
impl sealed::Sealed for u16 {}
impl sealed::Sealed for i16 {}
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
/// every type but `bool`), and the size of a buffer comes from `LAYOUT`.
///
/// A `bool` takes one bit: an array of `n` bools is `ceil(n / 32)` `u32` words, element `i` is
/// bit `i % 32` of word `i / 32`, and the bits past `n` in the last word are 0 (a custom kernel
/// sees an `array<u32>` of those words, see [`Kernel`](crate::Kernel)).
pub trait GpuType: Zeroable + Copy + Send + Sync + std::fmt::Debug + sealed::Sealed + 'static {
    /// The plain-old-data form of the elements that is uploaded to and read back from the GPU:
    /// one `Repr` per element, except for the bit-packed `bool` (32 elements to a `u32`).
    #[doc(hidden)]
    type Repr: Pod;

    /// How the elements sit in a buffer.
    #[doc(hidden)]
    const LAYOUT: ElemLayout = ElemLayout::plain(std::mem::size_of::<Self::Repr>());

    /// The GPU words of `host`: borrowed when the representation is the type itself.
    #[doc(hidden)]
    fn to_repr(host: &[Self]) -> std::borrow::Cow<'_, [Self::Repr]>;

    /// The `len` host values of GPU words read back.
    #[doc(hidden)]
    fn from_repr(repr: Vec<Self::Repr>, len: usize) -> Vec<Self>;

    /// `host` viewed as GPU words, if the representation is the type itself (no conversion
    /// needed to read into it); `None` otherwise.
    #[doc(hidden)]
    fn repr_mut(host: &mut [Self]) -> Option<&mut [Self::Repr]>;

    /// The GPU word of a one-element array holding this value.
    #[doc(hidden)]
    fn into_repr(self) -> Self::Repr;
}

/// How the elements of a type sit in a device buffer.
#[doc(hidden)]
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct ElemLayout {
    stride: usize,
    /// Elements to a `u32` word: 1 for elements of a word or more, 2 (`u16`, `i16`), 4 (`u8`,
    /// `i8`) or 32 (`bool`) for the types that share words.
    per_word: u8,
}

impl ElemLayout {
    /// Elements of `bytes` bytes each, one after the other.
    pub const fn plain(bytes: usize) -> Self {
        ElemLayout { stride: bytes, per_word: 1 }
    }

    /// One bit per element: 32 to a `u32` word, element `i` at bit `i % 32` of word `i / 32`.
    /// Arrays split over several buffers split like arrays of 4-byte elements (see
    /// [`ElemLayout::stride`]), so every buffer starts on a word.
    pub const PACKED: ElemLayout = ElemLayout { stride: 4, per_word: 32 };

    /// Two bytes per element: 2 to a `u32` word, element `i` in the bits `16 * (i % 2)..` of word
    /// `i / 2` (little endian, the layout of a plain array of 16-bit integers).
    pub const HALF: ElemLayout = ElemLayout { stride: 4, per_word: 2 };

    /// One byte per element: 4 to a `u32` word, element `i` in the bits `8 * (i % 4)..` of word
    /// `i / 4` (little endian, the layout of a plain array of bytes).
    pub const BYTE: ElemLayout = ElemLayout { stride: 4, per_word: 4 };

    /// The bytes that set the elements a segment of an array holds, `segment_bytes / stride`:
    /// the element size, or 4 for elements that share words (arrays of `bool`, `u8`, `i16`, ... and
    /// `f32` are split alike, so that an operation between them runs per segment).
    pub const fn stride(self) -> usize {
        self.stride
    }

    /// Whether several elements share a word (`bool`, 8-bit and 16-bit integers).
    pub const fn is_packed(self) -> bool {
        self.per_word > 1
    }

    /// Whether elements are bits of words.
    pub const fn is_bits(self) -> bool {
        self.per_word == 32
    }

    /// Elements to a word: 1 when they do not share words.
    pub const fn per_word(self) -> usize {
        self.per_word as usize
    }

    /// The bytes of a buffer holding `len` elements. Never compute `len * stride` for a packed
    /// layout.
    pub const fn bytes(self, len: usize) -> u64 {
        if self.is_packed() {
            len.div_ceil(self.per_word as usize) as u64 * 4
        } else {
            len as u64 * self.stride as u64
        }
    }

    /// The `Repr` words holding the elements `range` (which starts on a multiple of the
    /// elements per word if the layout is packed).
    pub const fn words(self, range: std::ops::Range<usize>) -> std::ops::Range<usize> {
        if self.is_packed() { range.start / self.per_word as usize..range.end.div_ceil(self.per_word as usize) } else { range }
    }
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

            fn from_repr(repr: Vec<Self::Repr>, _len: usize) -> Vec<Self> {
                repr
            }

            fn repr_mut(host: &mut [Self]) -> Option<&mut [Self::Repr]> {
                Some(host)
            }

            fn into_repr(self) -> Self::Repr {
                self
            }
        }
    )*};
}

plain_gpu_type!(u32, i32, f32, f64, Df32, [f32; 2], [f32; 4], [u32; 2], [u32; 4]);

impl GpuType for bool {
    type Repr = u32;

    const LAYOUT: ElemLayout = ElemLayout::PACKED;

    fn to_repr(host: &[Self]) -> std::borrow::Cow<'_, [Self::Repr]> {
        std::borrow::Cow::Owned(
            host.chunks(32).map(|chunk| chunk.iter().enumerate().fold(0u32, |word, (bit, &b)| word | u32::from(b) << bit)).collect(),
        )
    }

    fn from_repr(repr: Vec<Self::Repr>, len: usize) -> Vec<Self> {
        let mut out = Vec::with_capacity(len);
        let whole = len / 32;
        for &word in &repr[..whole] {
            out.extend((0..32).map(|bit| word >> bit & 1 != 0));
        }
        if !len.is_multiple_of(32) {
            out.extend((0..len % 32).map(|bit| repr[whole] >> bit & 1 != 0));
        }
        out
    }

    fn repr_mut(_host: &mut [Self]) -> Option<&mut [Self::Repr]> {
        None
    }

    fn into_repr(self) -> Self::Repr {
        u32::from(self)
    }
}

/// `GpuType` for the 8-bit and 16-bit integers: `4` or `2` elements to a `u32` word (little
/// endian, the bytes of the host slice), see [`ElemLayout::BYTE`] and [`ElemLayout::HALF`]. The
/// host values go to the device as they are (a copy of the bytes, padded to a whole word, no
/// widening) and come back the same way.
macro_rules! small_gpu_type {
    ($($ty:ty => $layout:ident),*) => {$(
        impl GpuType for $ty {
            type Repr = u32;

            const LAYOUT: ElemLayout = ElemLayout::$layout;

            fn to_repr(host: &[Self]) -> std::borrow::Cow<'_, [Self::Repr]> {
                let bytes: &[u8] = bytemuck::cast_slice(host);
                if bytes.len().is_multiple_of(4)
                    && let Ok(words) = bytemuck::try_cast_slice::<u8, u32>(bytes)
                {
                    return std::borrow::Cow::Borrowed(words);
                }
                let mut words = vec![0u32; bytes.len().div_ceil(4)];
                bytemuck::cast_slice_mut::<u32, u8>(&mut words)[..bytes.len()].copy_from_slice(bytes);
                std::borrow::Cow::Owned(words)
            }

            fn from_repr(repr: Vec<Self::Repr>, len: usize) -> Vec<Self> {
                let bytes: &[u8] = bytemuck::cast_slice(&repr);
                bytemuck::pod_collect_to_vec(&bytes[..len * std::mem::size_of::<$ty>()])
            }

            fn repr_mut(host: &mut [Self]) -> Option<&mut [Self::Repr]> {
                let bytes: &mut [u8] = bytemuck::cast_slice_mut(host);
                if !bytes.len().is_multiple_of(4) {
                    return None;
                }
                bytemuck::try_cast_slice_mut::<u8, u32>(bytes).ok()
            }

            fn into_repr(self) -> Self::Repr {
                // The word of a one-element array: the value in the low bits.
                (self as u32) & (u32::MAX >> (32 - 8 * std::mem::size_of::<$ty>()))
            }
        }
    )*};
}

small_gpu_type!(u8 => BYTE, i8 => BYTE, u16 => HALF, i16 => HALF);

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

// The WGSL type of a `bool` in registers and of the words of its buffers (32 bools to a `u32`,
// `GpuType::LAYOUT`). The fused kernels read and write the bits through the accessors
// `codegen::storage_ty`, `codegen::load` and `codegen::store`; the built-in kernels that touch
// bool buffers (fill, the reductions of words) know the layout themselves.
impl ScalarType for bool {
    fn wgsl_type() -> &'static str {
        "u32"
    }
}

// The WGSL type of the register holding a small integer, and of the words of its buffers
// (`GpuType::LAYOUT`): the kernels widen on load and narrow on store (`jit::kirgen`).
impl ScalarType for u8 {
    fn wgsl_type() -> &'static str {
        "u32"
    }
}

impl ScalarType for u16 {
    fn wgsl_type() -> &'static str {
        "u32"
    }
}

impl ScalarType for i8 {
    fn wgsl_type() -> &'static str {
        "i32"
    }
}

impl ScalarType for i16 {
    fn wgsl_type() -> &'static str {
        "i32"
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
/// have `+ - * /`, `sqrt`, `exp`, `ln`, `tanh`, `abs`, `log1p`, `expm1`, `powf`, comparisons,
/// `minimum`/`maximum`, `select`, views, reductions and the casts [`crate::Array::to_f32`] /
/// `to_f64`. See [`crate::Array::sum_compensated`].
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
