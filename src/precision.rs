//! Extended precision: casts between `f32`, `f64` and [`Df32`] arrays and the compensated
//! reductions built on them.
//!
//! `f64` arrays need a device with 64-bit shader floats (`wgpu::Features::SHADER_F64`, in
//! practice Vulkan) and have the operations `+ - * /`, `sqrt`, `abs`, `sign`, `floor`, `ceil`,
//! `minimum`, `maximum`, comparisons, `select`, views and reductions; anything else (`exp`,
//! trigonometry, ...) poisons the result with [`LumaError::Unsupported`](crate::LumaError).
//! [`Df32`] arrays (a pair of `f32`, about 48 bits of mantissa) run everywhere with `+ - * /`,
//! `sqrt`, `abs`, comparisons, `minimum`, `maximum`, `select`, views and reductions.

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
    /// The elements as `f64`. Needs a device with 64-bit shader floats; otherwise the result is
    /// poisoned with [`LumaError::Unsupported`](crate::LumaError).
    to_f64 -> f64;
    /// The elements as double-floats (`lo` is zero).
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

impl Array<f32> {
    /// The sum of all elements, accumulated in double-float arithmetic and rounded to `f32`
    /// at the end: a lazy one-element array like [`Array::sum`]. The accumulation does not
    /// depend on the reduction tree (workgroup sizes, subgroups), so the result is the exact sum
    /// rounded to `f32` up to the last place; use it for residual norms and in regression tests.
    /// Costs several times the plain sum's arithmetic and reads the array once.
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
