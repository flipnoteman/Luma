//! `searchsorted`, `digitize` and `isin`: binary searches of arrays of values in a sorted array.
//!
//! The search is lazy and elementwise over the values: `ceil(log2(n + 1))` unrolled steps of a
//! branch-free binary search, each a gather from the sorted array ([`Array::take`]) and a
//! comparison, built from ordinary lazy operations. The values (and whatever expression they
//! are) are computed inside the one kernel that runs the steps, which reads the sorted array as
//! a buffer, so a lookup costs the values' bytes in and the positions' bytes out. The order is
//! the one of [`Array::sort`]: `NaN` is larger than everything, `-0.0` equals `+0.0`.

use std::any::Any;
use std::panic::Location;

use crate::array::{Array, Failure};
use crate::error::LumaError;
use crate::ops::select;
use crate::sort::SortKey;

/// Which insertion point [`Array::searchsorted`] returns where the value equals elements of the
/// sorted array.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Side {
    /// The first position where the value could be inserted keeping the order: the number of
    /// elements below it.
    Left,
    /// The last such position: the number of elements below or equal to it.
    Right,
}

/// `array` as an `f32` array, if that is its type.
fn as_f32<T: SortKey>(array: &Array<T>) -> Option<&Array<f32>> {
    (array as &dyn Any).downcast_ref()
}

impl<T: SortKey> Array<T> {
    /// The insertion points of `values` in this sorted array, NumPy's `searchsorted`: element `i`
    /// of the result is the number of elements of `self` that are below (`side` [`Side::Left`])
    /// or below or equal to (`Side::Right`) `values[i]`, so that inserting there keeps `self`
    /// sorted. It has the shape of `values`; `self` must be one-dimensional and sorted ascending
    /// (what [`Array::sort`] produces: NaNs last, which here are larger than any number, and
    /// equal to each other; `-0.0` equals `+0.0`), else the positions are meaningless.
    ///
    /// Lazy: one binary search per value, in a kernel of its own or fused into the expression
    /// that reads the result, with the values computed inside it. A pending `self` is evaluated
    /// first (the search reads it at scattered positions). The unrolled search is `ceil(log2(n + 1))`
    /// steps, so the kernel is compiled once per bit length of `n`. A `self` that is not
    /// one-dimensional poisons the result with [`LumaError::InvalidShape`], operands of different
    /// contexts with [`LumaError::ContextMismatch`].
    #[track_caller]
    pub fn searchsorted(&self, values: &Array<T>, side: Side) -> Array<u32> {
        self.search_at(values, side, "searchsorted", Location::caller())
    }

    /// [`Array::searchsorted`] on behalf of `name` at `location`.
    pub(crate) fn search_at(&self, values: &Array<T>, side: Side, name: &'static str, location: &'static Location<'static>) -> Array<u32> {
        let ctx = values.context();
        let result = if !ctx.same_as(self.context()) {
            Err(Failure::New(LumaError::ContextMismatch))
        } else if self.shape().dims()[1..].iter().any(|&d| d != 1) {
            Err(Failure::New(LumaError::InvalidShape(format!("{name} in an array of shape {}: it must be one-dimensional", self.shape()))))
        } else {
            Ok(search(&self.cache(), values, side))
        };
        Array::resolve(ctx, values.shape(), result, name, location)
    }

    /// The bin of every element among the increasing `edges`, NumPy's `digitize`: with `right`
    /// false the `i` with `edges[i - 1] <= x < edges[i]`, with `right` true the `i` with
    /// `edges[i - 1] < x <= edges[i]`. `0` is below the first edge, `edges.len()` at or above the
    /// last (above it with `right`), NaN is above all of them. A [`Array::searchsorted`] of the
    /// edges (`Side::Right`, or `Side::Left` with `right`), with the same laziness, shape and
    /// errors; decreasing edges are not supported.
    #[track_caller]
    pub fn digitize(&self, edges: &Array<T>, right: bool) -> Array<u32> {
        edges.search_at(self, if right { Side::Left } else { Side::Right }, "digitize", Location::caller())
    }

    /// Whether each element equals one of the elements of `set`, NumPy's `isin`, as a lazy
    /// `bool` array of this array's shape. `NaN` is in no set (it equals nothing), `-0.0` and
    /// `+0.0` are the same value. The set (of any shape) is sorted when `isin` is called
    /// ([`Array::sort`]: eager, no host synchronization), then searched for every element
    /// like [`Array::searchsorted`] does and the found element compared; the elements are computed
    /// inside the search. Sorting errors (more than one buffer in `set`) poison the result.
    #[track_caller]
    pub fn isin(&self, set: &Array<T>) -> Array<bool> {
        let location = Location::caller();
        let sorted = set.reshape(set.len()).sort();
        let positions = sorted.search_at(self, Side::Left, "isin", location);
        let found = sorted.take(&positions).eq(self);
        match as_f32(self) {
            // A NaN of the set sorts last and would equal itself where comparisons are unreliable.
            Some(floats) => found & !floats.is_nan(),
            None => found,
        }
    }
}

/// The steps of a search in `sorted` (evaluated, one-dimensional): `step` takes the values
/// `2^k, ..., 2, 1` for the largest `2^k <= n`, and `pos` is the largest `p` below `n` with every
/// element before `p` ordered before the value, found by trying to add each step. The first step
/// probes the one element `step - 1`, shared by all values.
fn search<T: SortKey>(sorted: &Array<T>, values: &Array<T>, side: Side) -> Array<u32> {
    let n = sorted.len();
    let mut step = 1u32 << (usize::BITS - 1 - n.leading_zeros());
    let before = Before::new(values, side);
    let first = sorted.slice(0, step as usize - 1..step as usize);
    let mut pos = select(&before.probe(&first), step, 0u32);
    while step > 1 {
        step /= 2;
        // The probe `pos + step - 1` is inside the array if the step fits; the gather clamps
        // the index, so the test must be done by the position.
        let index = &pos + (step - 1);
        let go = before.probe(&sorted.take(&index)) & index.lt(n as u32);
        pos = &pos + select(&go, step, 0u32);
    }
    pos
}

/// Whether an element of the sorted array is ordered before a value (strictly for
/// [`Side::Left`], or equal for [`Side::Right`]), with `NaN` the largest.
struct Before<'a, T: SortKey> {
    values: &'a Array<T>,
    side: Side,
    /// The `NaN`s among the values, for floats.
    nan: Option<Array<bool>>,
}

impl<'a, T: SortKey> Before<'a, T> {
    fn new(values: &'a Array<T>, side: Side) -> Self {
        Before { values, side, nan: as_f32(values).map(Array::<f32>::is_nan) }
    }

    fn probe(&self, probe: &Array<T>) -> Array<bool> {
        let ordered = match self.side {
            Side::Left => self.values.gt(probe),
            Side::Right => self.values.ge(probe),
        };
        match (&self.nan, as_f32(probe)) {
            // A NaN is above every other value and equal to a NaN, and the comparisons themselves
            // are not trusted with NaNs: for `Left` a NaN probe is never below, a NaN value is
            // above every probe that is not NaN; for `Right` a NaN value is above or equal to all.
            (Some(value_nan), Some(floats)) => match self.side {
                Side::Left => !floats.is_nan() & (value_nan | ordered),
                Side::Right => value_nan | (!floats.is_nan() & ordered),
            },
            _ => ordered,
        }
    }
}
