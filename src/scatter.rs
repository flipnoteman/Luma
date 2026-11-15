//! Scatter-add, scatter-min/max, `bincount` and `histogram`: writes that combine duplicates.

use std::panic::Location;
use std::sync::Arc;

use crate::array::Array;
use crate::error::{LumaError, Result};
use crate::gpu_type::ScalarType;
use crate::jit::AtomicOp;
use crate::ops::Operand;

/// Element types the combining scatters ([`Array::put_add`], [`Array::put_min`],
/// [`Array::put_max`]) are defined for: `u32`, `i32` and `f32`.
pub trait AtomicElem: ScalarType {
    /// The value no element is greater than: where a minimum starts.
    #[doc(hidden)]
    const MIN_IDENTITY: Self;
    /// The value no element is less than: where a maximum starts.
    #[doc(hidden)]
    const MAX_IDENTITY: Self;
}

impl AtomicElem for u32 {
    const MIN_IDENTITY: u32 = u32::MAX;
    const MAX_IDENTITY: u32 = 0;
}
impl AtomicElem for i32 {
    const MIN_IDENTITY: i32 = i32::MAX;
    const MAX_IDENTITY: i32 = i32::MIN;
}
impl AtomicElem for f32 {
    const MIN_IDENTITY: f32 = f32::INFINITY;
    const MAX_IDENTITY: f32 = f32::NEG_INFINITY;
}

impl<T: AtomicElem> Array<T> {
    /// Scatters with addition: `self[indices[i]] += values[i]` for every `i`, in place, so equal
    /// indices all count. Indices of `len` or more are skipped. `values` is an array of the shape
    /// of `indices` or a single value. Must be called on the array itself, not on a view of
    /// another one; lazy arrays recorded before the write keep their values. `indices` and
    /// `values` are computed inside the kernel, not evaluated first. Integer sums wrap and are
    /// exact; `f32` sums are accumulated atomically in no fixed order, so the last bits of the
    /// result may differ between runs.
    #[track_caller]
    pub fn put_add(&mut self, indices: &Array<u32>, values: impl Operand<T>) -> Result<()> {
        self.put_with(indices, values, Some(AtomicOp::Add), "put_add", Location::caller())
    }

    /// Scatters with the minimum: `self[indices[i]] = min(self[indices[i]], values[i])`. Same
    /// rules as [`Array::put_add`]; the result does not depend on the order of the writes.
    #[track_caller]
    pub fn put_min(&mut self, indices: &Array<u32>, values: impl Operand<T>) -> Result<()> {
        self.put_with(indices, values, Some(AtomicOp::Min), "put_min", Location::caller())
    }

    /// Scatters with the maximum: `self[indices[i]] = max(self[indices[i]], values[i])`. Same
    /// rules as [`Array::put_add`]; the result does not depend on the order of the writes.
    #[track_caller]
    pub fn put_max(&mut self, indices: &Array<u32>, values: impl Operand<T>) -> Result<()> {
        self.put_with(indices, values, Some(AtomicOp::Max), "put_max", Location::caller())
    }
}

/// How many times each value in `0..bins` occurs in `indices`, as a `u32` array of `bins`
/// elements. Indices of `bins` or more are not counted. Errors poison the result.
#[track_caller]
pub fn bincount(indices: &Array<u32>, bins: usize) -> Array<u32> {
    let location = Location::caller();
    let ctx = indices.context();
    let mut counts = match Array::<u32>::zeros_on(ctx, bins) {
        Ok(counts) => counts,
        Err(error) => return Array::poisoned(ctx, crate::Shape::from(bins), Arc::new(error)),
    };
    if let Err(error) = counts.put_with(indices, 1u32, Some(AtomicOp::Add), "bincount", location) {
        counts.poison(Arc::new(error));
    }
    counts
}

impl Array<f32> {
    /// Counts the elements in `bins` equal-width bins over `[lo, hi)`: bin `k` holds
    /// `lo + k * w <= x < lo + (k + 1) * w` for `w = (hi - lo) / bins`. Values below `lo` and NaN
    /// go to bin 0, values of `hi` or more to the last bin. Lazy until counted: the bucket
    /// computation runs inside the scatter kernel. `bins == 0` or `hi.partial_cmp(&lo) != Some(std::cmp::Ordering::Greater)` poisons the
    /// result with [`LumaError::InvalidShape`].
    #[track_caller]
    pub fn histogram(&self, bins: usize, lo: f32, hi: f32) -> Array<u32> {
        let ctx = self.context();
        if bins == 0 || bins > u32::MAX as usize || hi.partial_cmp(&lo) != Some(std::cmp::Ordering::Greater) {
            let error = LumaError::InvalidShape(format!("histogram of {bins} bins over [{lo}, {hi})"));
            return Array::poisoned(ctx, crate::Shape::from(bins.max(1)), Arc::new(error));
        }
        let scale = bins as f32 / (hi - lo);
        bincount(&((self - lo) * scale).to_u32().minimum(bins as u32 - 1), bins)
    }
}
