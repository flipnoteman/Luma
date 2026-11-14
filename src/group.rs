//! Group reductions: combine the elements that share a group code (a groupby over dense codes).

use std::panic::Location;
use std::sync::Arc;

use crate::array::Array;
use crate::error::LumaError;
use crate::scatter::{bincount, AtomicElem};
use crate::shape::Shape;

impl<T: AtomicElem> Array<T> {
    /// The target of a group reduction starting from `fill`, or the poisoned result.
    fn group_scatter(
        &self,
        codes: &Array<u32>,
        groups: usize,
        fill: Option<T>,
        location: &'static Location<'static>,
        scatter: impl FnOnce(&mut Array<T>, &Array<u32>, &Array<T>, &'static Location<'static>) -> crate::Result<()>,
    ) -> Array<T> {
        let ctx = self.context();
        let target = match fill {
            Some(value) => Array::<T>::full_on(ctx, groups, value),
            None => Array::<T>::zeros_on(ctx, groups),
        };
        let mut target = match target {
            Ok(target) => target,
            Err(error) => return Array::poisoned(ctx, Shape::from(groups.max(1)), Arc::new(error)),
        };
        if let Some(error) = self.group_check(codes) {
            target.poison(Arc::new(error));
            return target;
        }
        if let Err(error) = scatter(&mut target, codes, self, location) {
            target.poison(Arc::new(error));
        }
        target
    }

    /// `ShapeMismatch` unless `codes` has the shape of `self`.
    fn group_check(&self, codes: &Array<u32>) -> Option<LumaError> {
        (self.shape() != codes.shape()).then(|| LumaError::ShapeMismatch { lhs: self.shape().dims(), rhs: codes.shape().dims() })
    }

    /// The sum of the elements of every group: element `g` of the result adds the elements `i`
    /// with `codes[i] == g`, as [`Array::put_add`] into a zero array of `groups` elements. Codes of
    /// `groups` or more are skipped. `codes` must have the shape of this array, else the result
    /// is poisoned with [`LumaError::ShapeMismatch`]. The values and codes are computed inside
    /// the scatter kernel. Integer sums wrap and are exact; `f32` sums are not reproducible to the
    /// last bits (see [`Array::put_add`]).
    #[track_caller]
    pub fn group_sum(&self, codes: &Array<u32>, groups: usize) -> Array<T> {
        self.group_scatter(codes, groups, None, Location::caller(), |target, codes, values, location| {
            target.put_with(codes, values, Some(crate::jit::AtomicOp::Add), "group_sum", location)
        })
    }

    /// The number of elements of every group, as [`bincount`] of `codes`; this array only has to
    /// have the shape of `codes` (else the result is poisoned with [`LumaError::ShapeMismatch`]).
    #[track_caller]
    pub fn group_count(&self, codes: &Array<u32>, groups: usize) -> Array<u32> {
        match self.group_check(codes) {
            Some(error) => Array::poisoned(self.context(), Shape::from(groups.max(1)), Arc::new(error)),
            None => bincount(codes, groups),
        }
    }

    /// The minimum of every group, as [`Array::put_min`] into an array of `groups` elements that
    /// starts at the largest value of the type (`f32`: `+inf`). Groups without elements keep that
    /// value. Otherwise as [`Array::group_sum`].
    #[track_caller]
    pub fn group_min(&self, codes: &Array<u32>, groups: usize) -> Array<T> {
        self.group_scatter(codes, groups, Some(T::MIN_IDENTITY), Location::caller(), |target, codes, values, location| {
            target.put_with(codes, values, Some(crate::jit::AtomicOp::Min), "group_min", location)
        })
    }

    /// The maximum of every group, as [`Array::put_max`] into an array of `groups` elements that
    /// starts at the smallest value of the type (`f32`: `-inf`). Groups without elements keep
    /// that value. Otherwise as [`Array::group_sum`].
    #[track_caller]
    pub fn group_max(&self, codes: &Array<u32>, groups: usize) -> Array<T> {
        self.group_scatter(codes, groups, Some(T::MAX_IDENTITY), Location::caller(), |target, codes, values, location| {
            target.put_with(codes, values, Some(crate::jit::AtomicOp::Max), "group_max", location)
        })
    }
}

impl Array<f32> {
    /// The mean of every group: [`Array::group_sum`] over [`Array::group_count`], a lazy
    /// division. Groups without elements are NaN (0 / 0).
    #[track_caller]
    pub fn group_mean(&self, codes: &Array<u32>, groups: usize) -> Array<f32> {
        self.group_sum(codes, groups) / self.group_count(codes, groups).to_f32()
    }
}
