//! Set operations on one-dimensional sorted-unique views of arrays: [`Array::set_unique`],
//! [`Array::set_union`] and [`Array::set_intersect`] (ArrayFire's `setUnique`, `setUnion` and
//! `setIntersect`).
//!
//! Elements are equal when they compare `==`, except that all NaNs are one value (as in
//! [`Array::factorize`]); `-0.0` and `+0.0` are equal and a zero in a result is `+0.0`. Results
//! are sorted ascending. The number of elements of a result depends on
//! the data, so the operations read it back and **synchronize** once. Inputs of any shape are
//! taken in flat order.

use std::any::Any;

use crate::array::{Array, Failure};
use crate::error::{LumaError, Result};
use crate::setmerge::Op;
use crate::sort::SortKey;

impl<T: SortKey> Array<T> {
    /// `self` and `previous` elementwise equal as set elements (NaNs equal each other).
    fn same_as_set_element(&self, previous: &Array<T>) -> Array<bool> {
        let equal = self.eq(previous);
        match ((self as &dyn Any).downcast_ref::<Array<f32>>(), (previous as &dyn Any).downcast_ref::<Array<f32>>()) {
            (Some(x), Some(p)) => {
                let both_nan = x.is_nan() & p.is_nan();
                equal | both_nan
            }
            _ => equal,
        }
    }

    /// The array with every `-0.0` turned into `+0.0` (floats; others are returned as they are), so
    /// that equal elements are equal bit for bit.
    fn zeros_positive(&self) -> Array<T> {
        match (self as &dyn Any).downcast_ref::<Array<f32>>() {
            Some(x) => {
                let positive: Box<dyn Any> = Box::new(x + 0.0f32);
                *positive.downcast::<Array<T>>().expect("an f32 array is an array of T")
            }
            None => self.clone(),
        }
    }

    /// The first of every run of equal elements of the sorted flat array `sorted`.
    fn unique_sorted(sorted: &Array<T>) -> Result<Array<T>> {
        let previous = sorted.shift(1);
        let first = sorted.index(0).eq(0u32);
        let keep = first | sorted.same_as_set_element(&previous).ne(true);
        Ok(sorted.filter(&keep)?.expect("the first element is kept"))
    }

    /// The distinct elements of the array (in flat order) sorted ascending, as a one-dimensional
    /// array (ArrayFire's `setUnique`). `is_sorted` says the elements are in ascending order
    /// already, which skips the sort (a wrong claim gives an unspecified result). Synchronizes.
    pub fn set_unique(&self, is_sorted: bool) -> Result<Array<T>> {
        let flat = self.reshape(self.len()).zeros_positive();
        let sorted = if is_sorted { flat } else { flat.sort() };
        Array::unique_sorted(&sorted)
    }

    /// Both arrays sorted, flat: sorted by a sort unless `is_sorted`.
    fn sorted_pair(&self, other: &Array<T>, is_sorted: bool) -> Result<(Array<T>, Array<T>)> {
        if !self.context().same_as(other.context()) {
            return Err(LumaError::ContextMismatch);
        }
        let flat = |a: &Array<T>| {
            let flat = a.reshape(a.len());
            if is_sorted { flat } else { flat.sort() }
        };
        Ok((flat(self), flat(other)))
    }

    /// The sorted union of the elements of `self` and `other` without duplicates, one-dimensional
    /// (ArrayFire's `setUnion`). `is_unique` says both arrays are sorted ascending and hold
    /// distinct elements already, which skips their sorts (a wrong claim gives an unspecified
    /// result; the merge itself drops duplicates whatever the inputs hold). Merge path: see
    /// `setmerge.rs`. Synchronizes.
    pub fn set_union(&self, other: &Array<T>, is_unique: bool) -> Result<Array<T>> {
        let (a, b) = self.sorted_pair(other, is_unique)?;
        let merged = a.merge_sorted(&b, Op::Union).map_err(Failure::into_error)?;
        Ok(merged.expect("a union of non-empty arrays is not empty"))
    }

    /// The sorted elements that both `self` and `other` hold, without duplicates,
    /// one-dimensional (ArrayFire's `setIntersect`); `None` if there are none (luma has no arrays
    /// without elements). `is_unique` as for [`Array::set_union`]. Synchronizes.
    pub fn set_intersect(&self, other: &Array<T>, is_unique: bool) -> Result<Option<Array<T>>> {
        let (a, b) = self.sorted_pair(other, is_unique)?;
        a.merge_sorted(&b, Op::Intersect).map_err(Failure::into_error)
    }
}
