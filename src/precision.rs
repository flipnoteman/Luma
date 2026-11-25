//! Extended precision: casts between `f32`, `f64` and [`Df32`] arrays and the compensated
//! reductions built on them.
//!
//! `f64` arrays need a device with 64-bit shader floats (`luma_core::Features::SHADER_F64`: CUDA,
//! the native `cpu` backend, and some Vulkan GPUs) and have the operations `+ - * /`, `sqrt`,
//! `abs`, `sign`, `floor`, `ceil`, `minimum`, `maximum`, comparisons, `select`, views and
//! reductions; anything else (`exp`, trigonometry, ...) poisons the result with
//! [`LumaError::Unsupported`](crate::LumaError). [`Df32`] arrays (a pair of `f32`, about 48 bits of
//! mantissa) run everywhere with `+ - * /`, `sqrt`, `exp`, `ln`, `tanh`, `abs`, `log1p`, `expm1`,
//! `powf`, comparisons, `minimum`, `maximum`, `select`, views and reductions.

use std::panic::Location;

use crate::array::Array;
use crate::gpu_type::{Df32, ScalarType};
use crate::jit::{ElemTy, Op, OpKind, Src};
use crate::ops::{lazy, Outcome};

impl<T: ScalarType> Array<T> {
    /// The elements converted to `U`, as a lazy elementwise operation.
    fn run_cast<U: ScalarType>(&self, location: &'static Location<'static>) -> Outcome<Array<U>> {
        let operand = self.src()?;
        let op = Op::new(OpKind::Cast(ElemTy::of::<U>()?), ElemTy::of::<T>()?);
        lazy(self.context(), self.shape(), op, [operand, Src::unused(), Src::unused()], "cast", location)
    }

    /// The elements converted to `U`, failures reported at `location`.
    pub(crate) fn cast_at<U: ScalarType>(&self, location: &'static Location<'static>) -> Array<U> {
        let result = self.run_cast::<U>(location);
        Array::resolve(self.context(), self.shape(), result, "cast", location)
    }
}

macro_rules! casts {
    ($from:ty => $($(#[$doc:meta])* $name:ident -> $to:ty;)*) => {
        impl Array<$from> {
            $(
                $(#[$doc])*
                #[track_caller]
                pub fn $name(&self) -> Array<$to> {
                    let location = Location::caller();
                    let result = self.run_cast::<$to>(location);
                    Array::resolve(self.context(), self.shape(), result, stringify!($name), location)
                }
            )*
        }
    };
}

casts! { f32 =>
    /// The elements as `u32`: truncates toward zero and saturates to the nearest `f32` inside the
    /// range (`4294967040` at the top, `0` at the bottom); NaN becomes 0.
    to_u32 -> u32;
    /// The elements as `i32`: truncates toward zero and saturates to the nearest `f32` inside
    /// the range (`-2147483648..=2147483520`); NaN becomes 0.
    to_i32 -> i32;
    /// The elements as `f64`. Needs a device with 64-bit shader floats; otherwise the result is
    /// poisoned with [`LumaError::Unsupported`](crate::LumaError).
    to_f64 -> f64;
    /// The elements as double-floats (`lo` is zero).
    to_df32 -> Df32;
}

casts! { u32 =>
    /// The elements as `f32` (rounded to nearest above 2^24).
    to_f32 -> f32;
    /// The elements as `i32`: a two's-complement reinterpretation of the bits.
    to_i32 -> i32;
}

casts! { i32 =>
    /// The elements as `f32` (rounded to nearest above 2^24).
    to_f32 -> f32;
    /// The elements as `u32`: a two's-complement reinterpretation of the bits.
    to_u32 -> u32;
}

casts! { bool =>
    /// The elements as `u32`: 1 where true, 0 where false.
    to_u32 -> u32;
    /// The elements as `i32`: 1 where true, 0 where false.
    to_i32 -> i32;
    /// The elements as `f32`: 1.0 where true, 0.0 where false.
    to_f32 -> f32;
    /// The elements as `f64`: 1.0 where true, 0.0 where false. Needs a device with 64-bit shader
    /// floats; otherwise the result is poisoned with [`LumaError::Unsupported`](crate::LumaError).
    to_f64 -> f64;
    /// The elements as double-floats: 1 where true, 0 where false.
    to_df32 -> Df32;
}

casts! { f64 =>
    /// The elements rounded to `f32`.
    to_f32 -> f32;
    /// The elements as double-floats (exact up to 48 bits of mantissa).
    to_df32 -> Df32;
}

casts! { Df32 =>
    /// The elements rounded to `f32`.
    to_f32 -> f32;
    /// The elements as `f64` (exact). Needs a device with 64-bit shader floats.
    to_f64 -> f64;
}

// -- Casts to and from the half-precision floats --
//
// A conversion to `f16` or `bf16` rounds to nearest even (overflow gives infinity, a NaN the
// canonical quiet NaN); one from them is exact. The casts fuse into the kernels that read or
// write them: an `Array<f16>` read through `to_f32()` loads 2 bytes per element.

macro_rules! half_casts {
    ($($from:ty),*) => {$(
        casts! { $from =>
            /// The elements as `f16`, rounded to nearest even.
            to_f16 -> half::f16;
            /// The elements as `bf16`, rounded to nearest even.
            to_bf16 -> half::bf16;
        }
    )*};
}

half_casts!(f32, f64, i32, u32);

macro_rules! half_source_casts {
    ($($from:ty, $other:ty, $to_other:ident;)*) => {$(
        casts! { $from =>
            /// The elements as `f32` (exact).
            to_f32 -> f32;
            /// The elements as `f64` (exact). Needs a device with 64-bit shader floats;
            /// otherwise the result is poisoned with [`LumaError::Unsupported`](crate::LumaError).
            to_f64 -> f64;
            /// The elements as `i32`: truncates toward zero and saturates; NaN becomes 0.
            to_i32 -> i32;
            /// The elements as `u32`: truncates toward zero and saturates; NaN becomes 0.
            to_u32 -> u32;
            /// The elements in the other half-precision type, rounded to nearest even.
            $to_other -> $other;
        }
    )*};
}

half_source_casts! {
    half::f16, half::bf16, to_bf16;
    half::bf16, half::f16, to_f16;
}

// -- Casts to and from the 8-bit and 16-bit integers --
//
// A conversion *to* a small integer wraps an integer (the low bits are kept, like Rust's `as`:
// `300u32` is `44u8`, `-1i32` is `255u8` and `65535u16`) and saturates a float (a NaN is 0, like
// `as` on floats). A conversion *from* one is by value (`-1i8` is `-1i32`, `0xFFFF_FFFFu32` as
// `u32`, `-1.0f32`), exact for every target
// (every value of the small types is exactly representable in all of them). `to_bool` is `!= 0`. The casts fuse into the kernel that reads them:
// an `Array<i16>` read through `to_f32()` loads 2 bytes per element.

macro_rules! small_casts {
    ($($from:ty),*) => {$(
        casts! { $from =>
            /// The elements as `u8`: wraps an integer (keeps the low 8 bits), saturates a float
            /// (NaN becomes 0); `true` is 1.
            to_u8 -> u8;
            /// The elements as `i8`: wraps an integer (keeps the low 8 bits as two's complement),
            /// saturates a float (NaN becomes 0); `true` is 1.
            to_i8 -> i8;
            /// The elements as `u16`: wraps an integer (keeps the low 16 bits), saturates a float
            /// (NaN becomes 0); `true` is 1.
            to_u16 -> u16;
            /// The elements as `i16`: wraps an integer (keeps the low 16 bits as two's
            /// complement), saturates a float (NaN becomes 0); `true` is 1.
            to_i16 -> i16;
        }
    )*};
}

small_casts!(f32, f64, Df32, u32, i32, bool);

macro_rules! small_source_casts {
    ($($from:ty, $self_to:ident;)*) => {$(
        casts! { $from =>
            /// The elements as `f32` (exact).
            to_f32 -> f32;
            /// The elements as `f64` (exact). Needs a device with 64-bit shader floats;
            /// otherwise the result is poisoned with [`LumaError::Unsupported`](crate::LumaError).
            to_f64 -> f64;
            /// The elements as double-floats (exact).
            to_df32 -> Df32;
            /// The elements as `u32`, by value: a negative number becomes its two's complement
            /// (`-1i8` is `0xFFFF_FFFF`).
            to_u32 -> u32;
            /// The elements as `i32`, by value.
            to_i32 -> i32;
            /// `true` where the element is not 0.
            to_bool -> bool;
        }
    )*};
}

small_source_casts! {
    u8, to_u8;
    i8, to_i8;
    u16, to_u16;
    i16, to_i16;
}

casts! { u8 =>
    /// The elements as `i8`: the same bits.
    to_i8 -> i8;
    /// The elements as `u16`: zero-extended.
    to_u16 -> u16;
    /// The elements as `i16`: zero-extended.
    to_i16 -> i16;
}

casts! { i8 =>
    /// The elements as `u8`: the same bits.
    to_u8 -> u8;
    /// The elements as `u16`: sign-extended (`-1i8` is `0xFFFF`).
    to_u16 -> u16;
    /// The elements as `i16`: sign-extended.
    to_i16 -> i16;
}

casts! { u16 =>
    /// The elements as `u8`: the low 8 bits.
    to_u8 -> u8;
    /// The elements as `i8`: the low 8 bits as two's complement.
    to_i8 -> i8;
    /// The elements as `i16`: the same bits.
    to_i16 -> i16;
}

casts! { i16 =>
    /// The elements as `u8`: the low 8 bits.
    to_u8 -> u8;
    /// The elements as `i8`: the low 8 bits as two's complement.
    to_i8 -> i8;
    /// The elements as `u16`: the same bits.
    to_u16 -> u16;
}

impl Array<f32> {
    /// The sum of all elements, accumulated in double-float arithmetic and rounded to `f32` at
    /// the end: a lazy one-element array like [`Array::sum`]. The accumulator keeps roughly twice
    /// the precision of `f32`, so the result is the exact sum rounded to `f32` except for sums
    /// whose terms cancel almost completely; the order of the reduction tree (workgroup sizes,
    /// subgroups) can still change the bits below `f32` precision. Use it for residual norms and
    /// in regression tests. Costs several times the plain sum's arithmetic and reads the array
    /// once.
    #[track_caller]
    pub fn sum_compensated(&self) -> Array<f32> {
        self.to_df32().sum().to_f32()
    }

    /// The Euclidean norm with the squares accumulated in double-float arithmetic (see
    /// [`Array::sum_compensated`]), as a lazy one-element array.
    #[track_caller]
    pub fn norm_compensated(&self) -> Array<f32> {
        let d = self.to_df32();
        (&d * &d).sum().sqrt().to_f32()
    }
}
