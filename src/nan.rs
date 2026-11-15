//! NaN-aware reductions: `nansum`, `nanmin`, `nanmax` (and `nanmean` for `f32`) and their
//! `_axis` forms (`f32`), NumPy's semantics. Built from `is_nan`, `select` and the ordinary
//! reductions, so the masking is computed inside the reduction's first pass, like any pending
//! operand.

use std::panic::Location;

use crate::array::Array;
use crate::gpu_type::Df32;
use crate::ops::{run_select, Cond, Rhs};

/// `array` with the NaNs replaced by `fill`, lazy.
fn without_nans<T: crate::gpu_type::ScalarType>(array: &Array<T>, is_nan: &Array<bool>, fill: T, location: &'static Location<'static>) -> Array<T> {
    let result = run_select(Cond::Bool(&!is_nan), Rhs::Array(array), Rhs::Scalar(fill), location);
    Array::resolve(array.context(), array.shape(), result, "nan", location)
}

/// `value` where `any` is true, else `fill`, lazy.
fn or_else<T: crate::gpu_type::ScalarType>(any: &Array<bool>, value: &Array<T>, fill: T, location: &'static Location<'static>) -> Array<T> {
    let result = run_select(Cond::Bool(any), Rhs::Array(value), Rhs::Scalar(fill), location);
    Array::resolve(value.context(), value.shape(), result, "nan", location)
}

macro_rules! nan_reductions {
    ($t:ty, $inf:expr, $neg_inf:expr, $nan:expr) => {
        impl Array<$t> {
            /// The sum of the elements that are not NaN, as a lazy one-element array (see
            /// [`Array::sum`]); `0` if all are NaN. One pass over the elements: the sum of
            /// `select(!is_nan, x, 0)`, computed inside the reduction.
            #[track_caller]
            pub fn nansum(&self) -> Array<$t> {
                self.sum_where(&!&self.is_nan())
            }

            /// The smallest element that is not NaN, as a lazy one-element array; NaN if every
            /// element is NaN (NumPy's `nanmin`). Two passes over the elements, both fused with
            /// the NaN tests: the minimum with the NaNs replaced by `+inf`, and whether any element
            /// is not NaN (which tells an `+inf` minimum from an all-NaN array).
            #[track_caller]
            pub fn nanmin(&self) -> Array<$t> {
                let location = Location::caller();
                let nan = self.is_nan();
                let min = without_nans(self, &nan, $inf, location).min();
                or_else(&(!&nan).any(), &min, $nan, location)
            }

            /// The largest element that is not NaN, as a lazy one-element array; NaN if every
            /// element is NaN (NumPy's `nanmax`). Two passes, as [`Array::nanmin`].
            #[track_caller]
            pub fn nanmax(&self) -> Array<$t> {
                let location = Location::caller();
                let nan = self.is_nan();
                let max = without_nans(self, &nan, $neg_inf, location).max();
                or_else(&(!&nan).any(), &max, $nan, location)
            }
        }
    };
}

nan_reductions!(f32, f32::INFINITY, f32::NEG_INFINITY, f32::NAN);
nan_reductions!(f64, f64::INFINITY, f64::NEG_INFINITY, f64::NAN);
nan_reductions!(Df32, Df32::from(f32::INFINITY), Df32::from(f32::NEG_INFINITY), Df32 { hi: f32::NAN, lo: 0.0 });

impl Array<f32> {
    /// The mean of the elements that are not NaN, as a lazy one-element array: the sum of
    /// those over their number; NaN if every element is NaN (NumPy's `nanmean`). Two passes over
    /// the elements (the sum and the count), both fused with the NaN test.
    #[track_caller]
    pub fn nanmean(&self) -> Array<f32> {
        let ok = !&self.is_nan();
        let sum = self.sum_where(&ok);
        sum / ok.count().to_f32()
    }

    /// The sum along `axis` of the elements that are not NaN (`0` for a line of NaNs); the
    /// result has extent 1 along it. The masking is computed inside the reduction, like
    /// [`Array::sum_axis`] does for any pending operand.
    #[track_caller]
    pub fn nansum_axis(&self, axis: usize) -> Array<f32> {
        let location = Location::caller();
        without_nans(self, &self.is_nan(), 0.0, location).sum_axis(axis)
    }

    /// The mean along `axis` of the elements that are not NaN (NaN for a line of NaNs); the
    /// result has extent 1 along it. Two passes over the lines (the sum and the count), both
    /// fused with the NaN test, and a lazy division.
    #[track_caller]
    pub fn nanmean_axis(&self, axis: usize) -> Array<f32> {
        let ok = !&self.is_nan();
        let sum = without_nans(self, &self.is_nan(), 0.0, Location::caller()).sum_axis(axis);
        sum / ok.sum_axis(axis).to_f32()
    }

    /// The smallest element along `axis` that is not NaN (NaN for a line of NaNs); the result
    /// has extent 1 along it. Two passes over the lines, as [`Array::nanmin`].
    #[track_caller]
    pub fn nanmin_axis(&self, axis: usize) -> Array<f32> {
        let location = Location::caller();
        let nan = self.is_nan();
        let min = without_nans(self, &nan, f32::INFINITY, location).min_axis(axis);
        or_else(&(!&nan).max_axis(axis), &min, f32::NAN, location)
    }

    /// The largest element along `axis` that is not NaN (NaN for a line of NaNs); the result
    /// has extent 1 along it. Two passes over the lines, as [`Array::nanmin`].
    #[track_caller]
    pub fn nanmax_axis(&self, axis: usize) -> Array<f32> {
        let location = Location::caller();
        let nan = self.is_nan();
        let max = without_nans(self, &nan, f32::NEG_INFINITY, location).max_axis(axis);
        or_else(&(!&nan).max_axis(axis), &max, f32::NAN, location)
    }
}
