//! `factorize` and `unique`: the distinct values of an array and the dense code of each element.

use std::panic::Location;
use std::sync::Arc;

use crate::array::Array;
use crate::ops::select;
use crate::error::Result;
use crate::shape::Shape;
use crate::sort::SortKey;

impl<T: SortKey> Array<T> {
    /// The sorted distinct values of the array (one dimension) and, shaped like this array, the
    /// index into them of every element: `uniques.take(&codes)` is the array again. Works on the
    /// flat array. Floats compare as in NumPy's `unique`: all NaNs are one value (the last) and
    /// `-0.0` and `+0.0` are one value (the unique is `-0.0`, which sorts first). The values are sorted with [`Array::argsort`], so the
    /// array is evaluated and the number of distinct values read back, which synchronizes once.
    /// Errors poison both results.
    #[track_caller]
    pub fn factorize(&self) -> (Array<T>, Array<u32>) {
        let location = Location::caller();
        match self.run_factorize() {
            Ok(pair) => pair,
            Err(error) => {
                let error = Arc::new(crate::LumaError::Op { op: "factorize", location, source: Arc::new(error) });
                let ctx = self.context();
                (Array::poisoned(ctx, Shape::from(1), error.clone()), Array::poisoned(ctx, self.shape(), error))
            }
        }
    }

    /// The sorted distinct values of the array: the first result of [`Array::factorize`].
    #[track_caller]
    pub fn unique(&self) -> Array<T> {
        self.factorize().0
    }

    fn run_factorize(&self) -> Result<(Array<T>, Array<u32>)> {
        let ctx = self.context();
        let n = self.len();
        let flat = self.reshape(n);
        let (sorted, perm) = flat.sort_with_indices()?;
        // A new value starts where the key differs from its predecessor and at element 0. Two
        // NaNs are not different; comparisons with NaN are unreliable on the GPU, so floats
        // classify them instead.
        let previous = sorted.shift(1);
        let mut changed = sorted.ne(&previous);
        if let (Some(cur), Some(prev)) = (as_f32(&sorted), as_f32(&previous)) {
            let (nan_cur, nan_prev) = (cur.is_nan(), prev.is_nan());
            changed = (&nan_cur ^ &nan_prev) | (changed & !(nan_cur | nan_prev));
        }
        let flags = sorted.index(0).eq(0u32) | changed;
        let ids = flags.to_u32().cumsum(0) - 1u32;
        let count = ids.slice(0, n - 1..).item()? as usize + 1;
        let mut uniques = Array::<T>::alloc(ctx, count.into())?;
        // Only the first element of a run writes, so that equal keys that differ in bits (`-0.0`
        // and `+0.0`) leave the first in sorted order, not whichever write lands last.
        uniques.put(&select(&flags, &ids, u32::MAX), &sorted)?;
        let mut codes = Array::<u32>::alloc(ctx, n.into())?;
        codes.put(&perm, &ids)?;
        Ok((uniques, codes.reshape(self.shape())))
    }
}

/// `array` as an `f32` array, if that is its type.
fn as_f32<T: SortKey>(array: &Array<T>) -> Option<&Array<f32>> {
    (array as &dyn std::any::Any).downcast_ref()
}
