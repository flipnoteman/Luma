//! Quantiles and the median of `f32` arrays.

use std::sync::Arc;

use crate::array::Array;
use crate::error::LumaError;
use crate::ops::select;
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

    /// The quantiles `qs` of all elements, as an array of `qs.len()` elements: entry `j` is
    /// [`Array::quantile`] of `qs[j]`, but the array is sorted once however many quantiles are
    /// asked for (a call per quantile sorts every time, the sort being almost all of the cost).
    /// Every `q` must be in `[0, 1]` and `qs` must not be empty, else the result is poisoned
    /// with [`LumaError::InvalidShape`].
    ///
    /// ```
    /// # use luma::*;
    /// # let ctx = Context::new()?;
    /// let x = Array::from_slice(&[4.0f32, 1.0, 3.0, 2.0], 4)?;
    /// assert_eq!(x.quantiles(&[0.0, 0.25, 1.0]).to_vec()?, vec![1.0, 1.75, 4.0]);
    /// # Ok::<(), luma::LumaError>(())
    /// ```
    #[track_caller]
    pub fn quantiles(&self, qs: &[f32]) -> Array<f32> {
        let ctx = self.context();
        let n = self.len();
        match positions(qs, n) {
            Err(error) => Array::poisoned(ctx, Shape::from(qs.len().max(1)), Arc::new(error)),
            Ok(pos) => {
                let sorted = self.sort().reshape(n);
                pick(&sorted, [pos.len(), 1, 1, 1], 0, n, &pos)
            }
        }
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

    /// The quantiles `qs` of every line along `axis` (axis 0 varies fastest): the shape of this
    /// array with `qs.len()` elements along `axis`, so that `slice(axis, j..j + 1)` of the
    /// result is [`Array::quantile_axis`] of `qs[j]`. The array is sorted once however many
    /// quantiles are asked for. Every `q` must be in `[0, 1]`, `qs` must not be empty and `axis`
    /// must be below 4, else the result is poisoned with [`LumaError::InvalidShape`].
    ///
    /// ```
    /// # use luma::*;
    /// # let ctx = Context::new()?;
    /// // Two columns of four values.
    /// let x = Array::from_slice(&[4.0f32, 1.0, 3.0, 2.0, 10.0, 40.0, 20.0, 30.0], [4, 2])?;
    /// let q = x.quantiles_axis(0, &[0.0, 0.5, 1.0]);
    /// assert_eq!(q.shape(), Shape::from([3, 2]));
    /// assert_eq!(q.to_vec()?, vec![1.0, 2.5, 4.0, 10.0, 25.0, 40.0]);
    /// # Ok::<(), luma::LumaError>(())
    /// ```
    #[track_caller]
    pub fn quantiles_axis(&self, axis: usize, qs: &[f32]) -> Array<f32> {
        let ctx = self.context();
        let mut dims = self.shape().dims();
        if axis >= 4 {
            let error = LumaError::InvalidShape(format!("axis {axis} of shape {}", self.shape()));
            return Array::poisoned(ctx, self.shape(), Arc::new(error));
        }
        let n = dims[axis];
        dims[axis] = qs.len().max(1);
        match positions(qs, n) {
            Err(error) => Array::poisoned(ctx, Shape::from(dims), Arc::new(error)),
            Ok(pos) => {
                let sorted = self.sort_axis(axis);
                pick(&sorted, dims, axis, n, &pos)
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

/// The quantiles at `pos` (see [`positions`]) of the lines along `axis` of `sorted`, which are
/// sorted and have `n` elements; `dims` is the shape of the result, `pos.len()` along `axis`.
fn pick(sorted: &Array<f32>, dims: [usize; 4], axis: usize, n: usize, pos: &[(u32, u32, f32)]) -> Array<f32> {
    if let [(lo, hi, frac)] = *pos {
        let (lo, hi) = (lo as usize, hi as usize);
        return lerp(&sorted.slice(axis, lo..lo + 1), &sorted.slice(axis, hi..hi + 1), frac);
    }
    let ctx = sorted.context();
    let (m, stride) = (pos.len(), dims[..axis].iter().product::<usize>());
    let table = |values: Vec<u32>| Array::from_slice_on(ctx, &values, m);
    let out = Shape::from(dims);
    let build = || -> crate::error::Result<Array<f32>> {
        // The result element `at` is line `s + o * stride`, quantile `j`: `at = s + j * stride + o * stride * m`.
        let at = Array::<u32>::range_on(ctx, Shape::from(out.len()), 0)?;
        let (m, stride, n) = (m as u32, stride as u32, n as u32);
        let (s, j, o) = (&at % stride, (&at / stride) % m, &at / (stride * m));
        let gather = |ks: Vec<u32>| -> crate::error::Result<Array<f32>> {
            let k = table(ks)?.take(&j);
            Ok(sorted.take(&(&s + k * stride + &o * (stride * n))).reshape(out))
        };
        let lo = gather(pos.iter().map(|p| p.0).collect())?;
        if pos.iter().all(|p| p.2 == 0.0) {
            return Ok(lo);
        }
        let hi = gather(pos.iter().map(|p| p.1).collect())?;
        let frac = Array::from_slice_on(ctx, &pos.iter().map(|p| p.2).collect::<Vec<f32>>(), m as usize)?.take(&j).reshape(out);
        let blend = &lo + &((&hi - &lo) * &frac);
        // An exact position is the value itself, even where `hi - lo` is not finite.
        Ok(if pos.iter().any(|p| p.2 == 0.0) { select(&frac.eq(0.0), &lo, &blend) } else { blend })
    };
    match build() {
        Ok(array) => array,
        Err(error) => Array::poisoned(ctx, out, Arc::new(error)),
    }
}

/// The sorted positions and the interpolation weight of every quantile of `qs` among `n` values.
fn positions(qs: &[f32], n: usize) -> Result<Vec<(u32, u32, f32)>, LumaError> {
    if qs.is_empty() {
        return Err(LumaError::InvalidShape("no quantiles asked for".into()));
    }
    qs.iter().map(|&q| check_q(q, n).map(|(lo, hi, frac)| (lo as u32, hi as u32, frac))).collect()
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
