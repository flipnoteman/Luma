//! `factorize` and `unique`: the distinct values of an array and the dense code of each element.

use std::any::Any;
use std::panic::Location;
use std::sync::Arc;

use crate::array::Array;
use crate::reduce::Reduction;
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
        if ctx.can_run_on_host()
            && n <= crate::host::MAX_SORT_ELEMS
            && self.error().is_none()
            && let Some(pair) = crate::host::try_factorize(self)
        {
            return Ok(pair);
        }
        if n > 0 && !ctx.recording_here() {
            let fast = if let Some(a) = (self as &dyn Any).downcast_ref::<Array<u32>>() {
                small_range(a)?.map(|(u, c)| (Box::new(u) as Box<dyn Any>, c))
            } else if let Some(a) = (self as &dyn Any).downcast_ref::<Array<i32>>() {
                small_range(a)?.map(|(u, c)| (Box::new(u) as Box<dyn Any>, c))
            } else {
                None
            };
            if let Some((uniques, codes)) = fast {
                let uniques = *uniques.downcast::<Array<T>>().expect("uniques have the type of the keys");
                return Ok((uniques, codes.reshape(self.shape())));
            }
        }
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

/// An integer key type the counting path handles.
trait IntKey: SortKey + crate::Numeric {
    /// The value as a wide integer.
    fn wide(self) -> i64;
    /// `x - lo` as `u32`; the caller guarantees it does not overflow.
    fn offsets(x: &Array<Self>, lo: Self) -> Array<u32>;
    /// The array of the keys whose bit patterns are `bits`.
    fn from_bits(bits: Array<u32>) -> Array<Self>;
}

impl IntKey for u32 {
    fn wide(self) -> i64 {
        self as i64
    }
    fn offsets(x: &Array<u32>, lo: u32) -> Array<u32> {
        x - lo
    }
    fn from_bits(bits: Array<u32>) -> Array<u32> {
        bits
    }
}

impl IntKey for i32 {
    fn wide(self) -> i64 {
        self as i64
    }
    fn offsets(x: &Array<i32>, lo: i32) -> Array<u32> {
        (x - lo).to_u32()
    }
    fn from_bits(bits: Array<u32>) -> Array<i32> {
        bits.to_i32()
    }
}

/// The widest range of keys the counting path takes: a presence table of that many words is
/// cleared, scanned and read in place of the radix sort's eight passes over the keys.
const MAX_SPAN: usize = 1 << 24;

/// The fewest keys worth the extra synchronization of reading the range back: below it the sort
/// is as fast (and wins when the range turns out wide; measured on wgpu, where a readback costs
/// about 200 us).
const MIN_KEYS: usize = 1 << 18;

/// [`Array::factorize`] of integer keys whose range `max - min + 1` is small compared with the
/// number of keys: a presence table over the range, ranked by a scan, in place of the sort. The
/// result is the same as the sort's. `None` if the range is too wide. Reads the range back
/// (and, later, the number of distinct values), so it is not used while recording.
fn small_range<K: IntKey>(keys: &Array<K>) -> Result<Option<(Array<K>, Array<u32>)>> {
    let ctx = keys.context();
    let n = keys.len();
    if n < MIN_KEYS {
        return Ok(None);
    }
    let flat = keys.reshape(n);
    let [lo, hi] = flat.reduce([Reduction::Min, Reduction::Max]).to_array()?;
    let span = (hi.wide() - lo.wide() + 1) as usize;
    if span > MAX_SPAN || span > n {
        return Ok(None);
    }
    let offsets = K::offsets(&flat, lo);
    let mut present = Array::<u32>::zeros_on(ctx, span)?;
    // Equal keys write the same value, so the order of the writes does not matter.
    present.put(&offsets, 1u32)?;
    let ranks = present.cumsum(0);
    let count = ranks.slice(0, span - 1..).item()? as usize;
    let ids = ranks - 1u32;
    let mut uniques = Array::<u32>::alloc(ctx, count.into())?;
    let values = Array::<u32>::range_on(ctx, span, 0)? + (lo.wide() as u32);
    uniques.put(&select(&present, &ids, u32::MAX), &values)?;
    let codes = ids.take(&offsets);
    Ok(Some((K::from_bits(uniques), codes)))
}
