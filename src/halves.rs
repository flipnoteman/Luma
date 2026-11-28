//! Reductions, matrix products and the unsupported operations of the half-precision float arrays
//! ([`f16`](half::f16) and [`bf16`](half::bf16)), built on the `f32` ones like the compact
//! integers of `small.rs`: the elements are widened by a cast that fuses into the first kernel
//! (an `Array<f16>` read by a sum is read as 2 bytes per element) and the result is rounded to the
//! element type once, at the end. The accumulator is `f32`, which is more accurate than rounding
//! to the element type after every addition; the sum of an `f16` array is exact to `f32` precision
//! until the final rounding (and overflows to infinity above 65504).

use std::panic::Location;

use half::{bf16, f16};

use crate::array::{Array, Failure};
use crate::error::LumaError;
use crate::gpu_type::GpuType;
use crate::search::Side;

impl<T: GpuType> Array<T> {
    /// The poisoned result of an operation half-precision arrays do not have (sorts, scans, set
    /// operations, searches and atomics), tagged with the call.
    fn unsupported_half<O: GpuType>(&self, shape: crate::Shape, ty: &str, op: &'static str, location: &'static Location<'static>) -> Array<O> {
        let error = LumaError::Unsupported(format!("{op} of {ty} arrays is not supported: convert with to_f32() first"));
        Array::resolve(self.context(), shape, Err(Failure::New(error)), op, location)
    }
}

macro_rules! half_array {
    ($($t:ident, $name:literal, $to:ident;)*) => {$(
        impl Array<$t> {
            /// The sum of all elements, accumulated in `f32` and rounded to the element type once
            /// at the end, as a lazy one-element array. The widening is fused into the reduction:
            /// the elements are read at their own size.
            #[track_caller]
            pub fn sum(&self) -> Array<$t> {
                self.to_f32().sum().$to()
            }

            /// The product of all elements, accumulated in `f32` and rounded once at the end.
            #[track_caller]
            pub fn prod(&self) -> Array<$t> {
                self.to_f32().prod().$to()
            }

            /// The product of all elements: the same as [`Array::prod`] (ArrayFire's name).
            #[track_caller]
            pub fn product(&self) -> Array<$t> {
                self.to_f32().product().$to()
            }

            /// The smallest element, as a lazy one-element array.
            #[track_caller]
            pub fn min(&self) -> Array<$t> {
                self.to_f32().min().$to()
            }

            /// The largest element, as a lazy one-element array.
            #[track_caller]
            pub fn max(&self) -> Array<$t> {
                self.to_f32().max().$to()
            }

            /// The mean of all elements: the `f32` sum divided by the length, rounded once.
            #[track_caller]
            pub fn mean(&self) -> Array<$t> {
                self.to_f32().mean().$to()
            }

            /// The Euclidean norm, accumulated in `f32` and rounded once.
            #[track_caller]
            pub fn norm(&self) -> Array<$t> {
                self.to_f32().norm().$to()
            }

            /// The sum along `axis`, accumulated in `f32` and rounded once; the result has
            /// extent 1 along it.
            #[track_caller]
            pub fn sum_axis(&self, axis: usize) -> Array<$t> {
                self.to_f32().sum_axis(axis).$to()
            }

            /// The product along `axis`, accumulated in `f32` and rounded once.
            #[track_caller]
            pub fn prod_axis(&self, axis: usize) -> Array<$t> {
                self.to_f32().prod_axis(axis).$to()
            }

            /// The minimum along `axis`; the result has extent 1 along it.
            #[track_caller]
            pub fn min_axis(&self, axis: usize) -> Array<$t> {
                self.to_f32().min_axis(axis).$to()
            }

            /// The maximum along `axis`; the result has extent 1 along it.
            #[track_caller]
            pub fn max_axis(&self, axis: usize) -> Array<$t> {
                self.to_f32().max_axis(axis).$to()
            }

            /// The mean along `axis`, accumulated in `f32` and rounded once.
            #[track_caller]
            pub fn mean_axis(&self, axis: usize) -> Array<$t> {
                self.to_f32().mean_axis(axis).$to()
            }

            /// The population variance along `axis`, computed in `f32` and rounded once.
            #[track_caller]
            pub fn var_axis(&self, axis: usize) -> Array<$t> {
                self.to_f32().var_axis(axis).$to()
            }

            /// The population standard deviation along `axis`, computed in `f32` and rounded once.
            #[track_caller]
            pub fn std_axis(&self, axis: usize) -> Array<$t> {
                self.to_f32().std_axis(axis).$to()
            }

            /// The flat index of the largest element (the first of equal ones), as a one-element
            /// `u32` array; see [`Array::argmax`] of an `f32` array.
            #[track_caller]
            pub fn argmax(&self) -> Array<u32> {
                self.to_f32().argmax()
            }

            /// The flat index of the smallest element (the first of equal ones).
            #[track_caller]
            pub fn argmin(&self) -> Array<u32> {
                self.to_f32().argmin()
            }

            /// The matrix product `self * rhs` (see [`Array::matmul`] of an `f32` array): the
            /// elements are widened to `f32` where the kernel loads them, the product is computed
            /// with the `f32` kernels (so it is accumulated in `f32`, with the precision of the
            /// context's numerics), and the result is rounded to the element type. Where the numerics
            /// allow [`Reduced::MATRIX_ACCUMULATE`](crate::Reduced::MATRIX_ACCUMULATE) (all but
            /// `strict`) and the device has half-precision tensor cores (CUDA, compute capability 8.0
            /// or higher), a product whose leading dimensions are multiples of 8 runs on them instead:
            /// the operands are multiplied exactly and accumulated in `f32` in hardware order, the
            /// result rounded once.
            #[track_caller]
            pub fn matmul(&self, rhs: &Array<$t>) -> Array<$t> {
                if let Some(product) = crate::matmul_half::product(self, rhs, false, false, Location::caller()) {
                    return product;
                }
                self.to_f32().matmul(&rhs.to_f32()).$to()
            }

            /// `self * rhs.transpose()` (see [`Array::matmul_nt`] of an `f32` array and
            /// [`Array::matmul`] of this type).
            #[track_caller]
            pub fn matmul_nt(&self, rhs: &Array<$t>) -> Array<$t> {
                if let Some(product) = crate::matmul_half::product(self, rhs, false, true, Location::caller()) {
                    return product;
                }
                self.to_f32().matmul_nt(&rhs.to_f32()).$to()
            }

            /// `self.transpose() * rhs` (see [`Array::matmul_tn`] of an `f32` array).
            #[track_caller]
            pub fn matmul_tn(&self, rhs: &Array<$t>) -> Array<$t> {
                if let Some(product) = crate::matmul_half::product(self, rhs, true, false, Location::caller()) {
                    return product;
                }
                self.to_f32().matmul_tn(&rhs.to_f32()).$to()
            }

            /// `self.transpose() * rhs.transpose()` (see [`Array::matmul_tt`] of an `f32` array).
            #[track_caller]
            pub fn matmul_tt(&self, rhs: &Array<$t>) -> Array<$t> {
                if let Some(product) = crate::matmul_half::product(self, rhs, true, true, Location::caller()) {
                    return product;
                }
                self.to_f32().matmul_tt(&rhs.to_f32()).$to()
            }

            /// The dot product with `rhs`, accumulated in `f32` and rounded once (see
            /// [`Array::dot`] of an `f32` array).
            #[track_caller]
            pub fn dot(&self, rhs: &Array<$t>) -> Array<$t> {
                self.to_f32().dot(&rhs.to_f32()).$to()
            }

            /// Not supported for half-precision arrays: the result is poisoned with
            /// [`LumaError::Unsupported`]. Convert with [`Array::to_f32`] first.
            #[track_caller]
            pub fn sort(&self) -> Array<$t> {
                self.unsupported_half(self.shape(), $name, "sort", Location::caller())
            }

            /// Not supported for half-precision arrays (see [`Array::sort`]).
            #[track_caller]
            pub fn sort_descending(&self) -> Array<$t> {
                self.unsupported_half(self.shape(), $name, "sort_descending", Location::caller())
            }

            /// Not supported for half-precision arrays (see [`Array::sort`]).
            #[track_caller]
            pub fn sort_axis(&self, _axis: usize) -> Array<$t> {
                self.unsupported_half(self.shape(), $name, "sort_axis", Location::caller())
            }

            /// Not supported for half-precision arrays (see [`Array::sort`]).
            #[track_caller]
            pub fn argsort(&self) -> Array<u32> {
                self.unsupported_half(self.shape(), $name, "argsort", Location::caller())
            }

            /// Not supported for half-precision arrays (see [`Array::sort`]).
            #[track_caller]
            pub fn argsort_descending(&self) -> Array<u32> {
                self.unsupported_half(self.shape(), $name, "argsort_descending", Location::caller())
            }

            /// Not supported for half-precision arrays (see [`Array::sort`]).
            #[track_caller]
            pub fn unique(&self) -> Array<$t> {
                self.unsupported_half(crate::Shape::from(1usize), $name, "unique", Location::caller())
            }

            /// Not supported for half-precision arrays (see [`Array::sort`]).
            #[track_caller]
            pub fn cumsum(&self, _axis: usize) -> Array<$t> {
                self.unsupported_half(self.shape(), $name, "cumsum", Location::caller())
            }

            /// Not supported for half-precision arrays (see [`Array::sort`]).
            #[track_caller]
            pub fn searchsorted(&self, values: &Array<$t>, _side: Side) -> Array<u32> {
                self.unsupported_half(values.shape(), $name, "searchsorted", Location::caller())
            }

            /// Not supported for half-precision arrays (see [`Array::sort`]).
            #[track_caller]
            pub fn topk(&self, k: usize) -> (Array<$t>, Array<u32>) {
                let location = Location::caller();
                let shape = crate::Shape::from(k.max(1));
                (self.unsupported_half(shape, $name, "topk", location), self.unsupported_half(shape, $name, "topk", location))
            }
        }
    )*};
}

half_array! {
    f16, "f16", to_f16;
    bf16, "bf16", to_bf16;
}
