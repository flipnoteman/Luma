//! Reductions, scans, searches, sorts and counting of the 8-bit and 16-bit integer arrays
//! (`u8`, `i8`, `u16`, `i16`), built on the 32-bit ones: the elements are widened by a cast that
//! fuses into the first kernel (an `Array<i16>` read by a sum is read as 2 bytes per element), so
//! the data is never stored widened.
//!
//! Sums, running sums and sums along an axis accumulate in `i32` (the signed types) or `u32` (the
//! unsigned ones) and wrap beyond 32 bits: `u8` sums are exact for up to 16.8 million elements of
//! 255, `u16` sums for 65 537 of 65 535 and `i16` sums for 65 536 of `-32768` (the same bound as
//! NumPy's `uint32` accumulator; NumPy's own default is 64-bit). Minima and maxima keep the type.
//! Means and variances are `f32`, accumulated in `f32` (the values are exact in `f32`; the sum
//! rounds like the sum of an `f32` array does).

use std::panic::Location;

use crate::array::Array;
use crate::ops::Outcome;
use crate::scatter::bincount;
use crate::sort::{radix_sort, KeyTransform, Keys, Payload, Sorted, Spec};

macro_rules! small_reductions {
    ($($t:ident, $wide:ident, $to_wide:ident, $back:ident;)*) => {$(
        impl Array<$t> {
            /// The sum of all elements, accumulated in 32 bits (`u32` for the unsigned types,
            /// `i32` for the signed ones; it wraps beyond that, see the module documentation), as
            /// a lazy one-element array. The widening is fused into the reduction: the elements are read at their
            /// own size.
            #[track_caller]
            pub fn sum(&self) -> Array<$wide> {
                self.$to_wide().sum()
            }

            /// The smallest element, as a lazy one-element array of the same type.
            #[track_caller]
            pub fn min(&self) -> Array<$t> {
                self.$to_wide().min().$back()
            }

            /// The largest element, as a lazy one-element array of the same type.
            #[track_caller]
            pub fn max(&self) -> Array<$t> {
                self.$to_wide().max().$back()
            }

            /// The mean of all elements as a lazy one-element `f32` array.
            #[track_caller]
            pub fn mean(&self) -> Array<f32> {
                self.to_f32().mean()
            }

            /// The sum along `axis` in the wide type; the result has extent 1 along it (see
            /// [`Array::sum`]).
            #[track_caller]
            pub fn sum_axis(&self, axis: usize) -> Array<$wide> {
                self.$to_wide().sum_axis(axis)
            }

            /// The minimum along `axis`, of the same type; the result has extent 1 along it.
            #[track_caller]
            pub fn min_axis(&self, axis: usize) -> Array<$t> {
                self.$to_wide().min_axis(axis).$back()
            }

            /// The maximum along `axis`, of the same type; the result has extent 1 along it.
            #[track_caller]
            pub fn max_axis(&self, axis: usize) -> Array<$t> {
                self.$to_wide().max_axis(axis).$back()
            }

            /// The mean along `axis` as `f32`; the result has extent 1 along it.
            #[track_caller]
            pub fn mean_axis(&self, axis: usize) -> Array<f32> {
                self.to_f32().mean_axis(axis)
            }

            /// The population variance along `axis` as `f32` (see [`Array::var_axis`] of an
            /// `f32` array, which this computes on the widened elements).
            #[track_caller]
            pub fn var_axis(&self, axis: usize) -> Array<f32> {
                self.to_f32().var_axis(axis)
            }

            /// The population standard deviation along `axis` as `f32`.
            #[track_caller]
            pub fn std_axis(&self, axis: usize) -> Array<f32> {
                self.to_f32().std_axis(axis)
            }

            /// The running sum along `axis` in the wide type (it wraps beyond 32 bits).
            #[track_caller]
            pub fn cumsum(&self, axis: usize) -> Array<$wide> {
                self.$to_wide().cumsum(axis)
            }

            /// The flat index of the largest element (the first of equal ones), as a one-element
            /// `u32` array; see [`Array::argmax`].
            #[track_caller]
            pub fn argmax(&self) -> Array<u32> {
                self.$to_wide().argmax()
            }

            /// The flat index of the smallest element (the first of equal ones).
            #[track_caller]
            pub fn argmin(&self) -> Array<u32> {
                self.$to_wide().argmin()
            }

            /// The index along `axis` of the largest element of every line.
            #[track_caller]
            pub fn argmax_axis(&self, axis: usize) -> Array<u32> {
                self.$to_wide().argmax_axis(axis)
            }

            /// The index along `axis` of the smallest element of every line.
            #[track_caller]
            pub fn argmin_axis(&self, axis: usize) -> Array<u32> {
                self.$to_wide().argmin_axis(axis)
            }

            /// Sorts every line along `axis` independently, keeping the shape (see
            /// [`Array::sort_axis`]).
            #[track_caller]
            pub fn sort_axis(&self, axis: usize) -> Array<$t> {
                self.$to_wide().sort_axis(axis).$back()
            }
        }
    )*};
}

small_reductions! {
    u8, u32, to_u32, to_u8;
    u16, u32, to_u32, to_u16;
    i8, i32, to_i32, to_i8;
    i16, i32, to_i32, to_i16;
}

/// The stable radix sort of `keys` (`bits` low bits order them; see `sort::Spec`), the first pass
/// computing the keys from the compact elements they are made of.
fn compact_radix(keys: &Array<u32>, bits: u32, descending: bool, index: bool) -> Outcome<Sorted> {
    let ctx = keys.context();
    let spec = Spec { transform: KeyTransform::Identity, descending, key_bits: bits, want_keys: !index };
    let payload = if index { Payload::Index } else { Payload::None };
    match keys.pending() {
        Some(node) => radix_sort(ctx, Keys::Pending(node), payload, spec),
        None => radix_sort(ctx, Keys::Ready(keys.storage()?), payload, spec),
    }
}

macro_rules! small_sorts {
    ($($t:ident, $bits:expr, $key:expr, $unkey:expr;)*) => {$(
        impl Array<$t> {
            /// The sorted keys (`u32`, see `small_sorts!`) or the sorted indices of the elements:
            /// a one-pass (8-bit) or two-pass (16-bit) radix sort, not the four of a 32-bit key.
            fn run_sort(&self, descending: bool, index: bool) -> Outcome<Array<u32>> {
                let ctx = self.context();
                let key: fn(&Array<$t>) -> Array<u32> = $key;
                let keys = key(self);
                if self.len() <= 1 {
                    keys.storage()?;
                    return if index { Ok(Array::<u32>::zeros_on(ctx, self.shape())?) } else { Ok(keys) };
                }
                let sorted = compact_radix(&keys, $bits, descending, index)?;
                let storage = if index { sorted.payload } else { sorted.keys };
                Ok(Array::from_storage(ctx, self.shape(), storage.expect("the sort wrote what was asked for")))
            }

            /// The elements in ascending order, with the shape of this array: a stable radix
            /// sort over the bits of the elements (one pass for `u8` and `i8`, two for `u16` and
            /// `i16`, where `u32` takes four), the first pass reading the elements as they are
            /// stored. See [`Array::sort`].
            #[track_caller]
            pub fn sort(&self) -> Array<$t> {
                let location = Location::caller();
                let unkey: fn(Array<u32>) -> Array<$t> = $unkey;
                let result = self.run_sort(false, false).map(unkey);
                Array::resolve(self.context(), self.shape(), result, "sort", location)
            }

            /// The elements in descending order, the exact reverse of [`sort`](Array::sort) by
            /// key: equal elements keep their input order (see [`Array::sort_descending`]).
            #[track_caller]
            pub fn sort_descending(&self) -> Array<$t> {
                let location = Location::caller();
                let unkey: fn(Array<u32>) -> Array<$t> = $unkey;
                let result = self.run_sort(true, false).map(unkey);
                Array::resolve(self.context(), self.shape(), result, "sort_descending", location)
            }

            /// The flat indices of the elements in ascending order, with the shape of this
            /// array: a stable sort, so equal elements keep their input order (see
            /// [`Array::argsort`]).
            #[track_caller]
            pub fn argsort(&self) -> Array<u32> {
                let location = Location::caller();
                let result = self.run_sort(false, true);
                Array::resolve(self.context(), self.shape(), result, "argsort", location)
            }

            /// The flat indices of the elements in descending order; ties in input order.
            #[track_caller]
            pub fn argsort_descending(&self) -> Array<u32> {
                let location = Location::caller();
                let result = self.run_sort(true, true);
                Array::resolve(self.context(), self.shape(), result, "argsort_descending", location)
            }
        }
    )*};
}

small_sorts! {
    u8, 8, |x| x.to_u32(), |k| k.to_u8();
    u16, 16, |x| x.to_u32(), |k| k.to_u16();
    i8, 8, |x| (x.to_i32() + 128).to_u32(), |k| (k.to_i32() - 128).to_i8();
    i16, 16, |x| (x.to_i32() + 32768).to_u32(), |k| (k.to_i32() - 32768).to_i16();
}

macro_rules! small_counts {
    ($($t:ident, $bins:expr;)*) => {$(
        impl Array<$t> {
            /// How many times each value in `0..bins` occurs, as a `u32` array of `bins`
            /// elements; values of `bins` or more are not counted (see [`bincount`]). The
            /// elements are read at their own size inside the counting kernel.
            #[track_caller]
            pub fn bincount(&self, bins: usize) -> Array<u32> {
                bincount(&self.to_u32(), bins)
            }

            /// The count of every possible value: a `u32` array of
            #[doc = concat!(stringify!($bins), " elements, element `v` the number of elements equal to `v`.")]
            #[track_caller]
            pub fn histogram(&self) -> Array<u32> {
                bincount(&self.to_u32(), $bins)
            }
        }
    )*};
}

small_counts! {
    u8, 256;
    u16, 65536;
}
