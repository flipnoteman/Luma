//! Quantiles and the median of `f32` arrays.

use std::sync::Arc;

use crate::array::Array;
use crate::error::LumaError;
use crate::shape::Shape;

impl Array<f32> {
    /// The `q`-quantile of all elements as a one-element array, by NumPy's default `linear`
    /// method: with `h = q * (len - 1)` the sorted value at `floor(h)` plus `h - floor(h)` times
    /// the step to the value at `ceil(h)`. `q` must be in `[0, 1]`, else the result is poisoned
    /// with [`LumaError::InvalidShape`]. The array is sorted with [`Array::sort`] (evaluated), the
    /// interpolation is lazy.
    ///
    /// NaNs sort last, so any NaN makes the high quantiles NaN; NumPy returns NaN for every
    /// quantile then, this does not.
    #[track_caller]
    pub fn quantile(&self, q: f32) -> Array<f32> {
        let ctx = self.context();
        let n = self.len();
        match check_q(q, n) {
            Err(error) => Array::poisoned(ctx, Shape::from(1), Arc::new(error)),
            Ok((lo, hi, frac)) => {
                let sorted = self.sort().reshape(n);
                lerp(&sorted.slice(0, lo..lo + 1), &sorted.slice(0, hi..hi + 1), frac)
            }
        }
    }

    /// The median of all elements: [`Array::quantile`] at 0.5.
    #[track_caller]
    pub fn median(&self) -> Array<f32> {
        self.quantile(0.5)
    }

    /// The `q`-quantile of every line along `axis` (axis 0 varies fastest): the shape of this
    /// array with extent 1 along `axis`. As [`Array::quantile`] on [`Array::sort_axis`]; `axis >=
    /// 4` poisons the result with [`LumaError::InvalidShape`].
    #[track_caller]
    pub fn quantile_axis(&self, axis: usize, q: f32) -> Array<f32> {
        let ctx = self.context();
        let mut dims = self.shape().dims();
        if axis >= 4 {
            let error = LumaError::InvalidShape(format!("axis {axis} of shape {}", self.shape()));
            return Array::poisoned(ctx, self.shape(), Arc::new(error));
        }
        let n = dims[axis];
        dims[axis] = 1;
        match check_q(q, n) {
            Err(error) => Array::poisoned(ctx, Shape::from(dims), Arc::new(error)),
            Ok((lo, hi, frac)) => {
                let sorted = self.sort_axis(axis);
                lerp(&sorted.slice(axis, lo..lo + 1), &sorted.slice(axis, hi..hi + 1), frac)
            }
        }
    }

    /// The median of every line along `axis`: [`Array::quantile_axis`] at 0.5.
    #[track_caller]
    pub fn median_axis(&self, axis: usize) -> Array<f32> {
        self.quantile_axis(axis, 0.5)
    }
}

/// `lo + frac * (hi - lo)`, lazily.
fn lerp(lo: &Array<f32>, hi: &Array<f32>, frac: f32) -> Array<f32> {
    if frac == 0.0 {
        return lo.clone();
    }
    lo + &((hi - lo) * frac)
}

/// The sorted positions and the interpolation weight of the `q`-quantile of `n` values.
fn check_q(q: f32, n: usize) -> Result<(usize, usize, f32), LumaError> {
    if !(0.0..=1.0).contains(&q) {
        return Err(LumaError::InvalidShape(format!("quantile {q} is outside [0, 1]")));
    }
    let h = f64::from(q) * (n - 1) as f64;
    let lo = h.floor();
    Ok((lo as usize, h.ceil() as usize, (h - lo) as f32))
}
