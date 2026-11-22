//! [`TopK`]: the `k` largest or smallest elements of a stream, with their global indices.

use std::sync::{Mutex, MutexGuard};

use crate::array::Array;
use crate::context::Context;
use crate::error::{LumaError, Result};
use crate::gpu_type::Df32;
use crate::ops::{eval_all, select};
use crate::shape::Shape;
use crate::sort::SortKey;
use crate::view::{concat, concat_slice};

use super::blob::{Reader, Writer};
use super::wide::{check_wide, AccumCore, Wide, V};
use super::{invalid, kept_dims, split_plan, strides};

const KIND: u8 = 6;

/// The most entries a per-position top-k keeps (it is `k` passes over the batch).
const MAX_K_ALONG: usize = 32;

fn ranges_is_whole(a: usize, b: usize, n: usize) -> bool {
    a == 0 && b == n
}

/// The `k` largest (or smallest) elements seen so far, best first, with their global indices:
/// the position of the element in the stream of all elements pushed (the elements of every batch
/// in memory order, batches one after the other; merging another `TopK` continues the indices
/// after this one's). Ties go to the earlier element. NaN elements and the elements a mask
/// leaves out are never selected; while fewer than `k` elements have been seen the remaining
/// slots hold index `-1` and the identity value (`-inf` / `i32::MIN` / `0`, or the opposite for
/// the smallest).
///
/// [`largest`](TopK::largest) and [`smallest`](TopK::smallest) treat the whole batch as one
/// stream: the pushes of up to 16 batches are sorted together once ([`Array::topk`] of their
/// concatenation) and merged with the state's `k` entries. [`largest_along`](TopK::largest_along)
/// and [`smallest_along`](TopK::smallest_along) keep the best `k` of every line along one axis
/// (per feature, say), with the *sample* index along that axis as the global index: `k` passes
/// of an argmax over the batch, for `k` of at most 32. `T` is `f32`, `u32` or `i32`.
///
/// ```no_run
/// use luma::{Array, Df32, TopK};
/// let top = TopK::<f32, Df32>::largest(3);
/// top.push(&Array::from_slice(&[1.0f32, 9.0, 3.0, 7.0], 4)?)?;
/// top.push(&Array::from_slice(&[8.0f32, 0.0], 2)?)?;
/// assert_eq!(top.values()?.to_vec()?, [9.0, 8.0, 7.0]);
/// assert_eq!(TopK::<f32, Df32>::to_f64(&top.indices()?)?, [1.0, 4.0, 3.0]);
/// # Ok::<(), luma::LumaError>(())
/// ```
pub struct TopK<T: SortKey + AccumCore, W: Wide = Df32> {
    k: usize,
    largest: bool,
    /// The axis whose lines are ranked (`None`: the whole batch is one stream).
    axis: Option<usize>,
    inner: Mutex<Inner<T, W>>,
}

struct Inner<T: SortKey + AccumCore, W: Wide> {
    state: Option<State<T, W>>,
}

struct State<T: SortKey + AccumCore, W: Wide> {
    ctx: Context,
    /// The batch's shape with the ranked axis set to 1 (per-position mode).
    kept: [usize; 4],
    values: Array<T>,
    /// Global indices as wide numbers, `-1` for an empty slot.
    indices: V<W>,
    /// Elements (samples along the axis in per-position mode) pushed so far, those of `pending`
    /// included (the base of the next batch's indices).
    pushed: u64,
    /// The batches (values with the elements that cannot win replaced by the identity, and the
    /// elements that may win) pushed since the last merge: sorted together, once.
    pending: Vec<(Array<T>, Option<Array<bool>>)>,
    pending_elements: usize,
}

/// Pushes recorded before they are sorted together, and the most elements held back.
const FLUSH_EVERY: usize = 16;
const FLUSH_ELEMENTS: usize = 1 << 24;

impl<T: SortKey + AccumCore, W: Wide> TopK<T, W> {
    /// The `k` largest elements of the stream of all pushed elements.
    pub fn largest(k: usize) -> TopK<T, W> {
        TopK { k, largest: true, axis: None, inner: Mutex::new(Inner { state: None }) }
    }

    /// The `k` smallest elements of the stream of all pushed elements.
    pub fn smallest(k: usize) -> TopK<T, W> {
        TopK { k, largest: false, axis: None, inner: Mutex::new(Inner { state: None }) }
    }

    /// The `k` largest elements of every line along `axis` (one stream per position of the other
    /// axes); the results have `k` entries along `axis`, and the indices count the samples along it.
    pub fn largest_along(k: usize, axis: usize) -> Result<TopK<T, W>> {
        Self::along(k, true, axis)
    }

    /// The `k` smallest elements of every line along `axis`, see [`largest_along`](Self::largest_along).
    pub fn smallest_along(k: usize, axis: usize) -> Result<TopK<T, W>> {
        Self::along(k, false, axis)
    }

    fn along(k: usize, largest: bool, axis: usize) -> Result<TopK<T, W>> {
        if axis >= 4 || k == 0 || k > MAX_K_ALONG {
            return Err(invalid(format!("a per-position top-{k} along axis {axis} (k is 1 to {MAX_K_ALONG}, axis below 4)")));
        }
        Ok(TopK { k, largest, axis: Some(axis), inner: Mutex::new(Inner { state: None }) })
    }

    fn lock(&self) -> MutexGuard<'_, Inner<T, W>> {
        self.inner.lock().unwrap_or_else(|p| p.into_inner())
    }

    /// The number of entries.
    pub fn k(&self) -> usize {
        self.k
    }

    /// Whether the `k` largest (rather than smallest) are kept.
    pub fn is_largest(&self) -> bool {
        self.largest
    }

    /// The identity: what loses against every element.
    fn identity(&self, ctx: &Context) -> Result<Array<T>> {
        T::extreme(ctx, !self.largest)
    }

    /// The state shape of a batch shape.
    fn state_dims(&self, kept: [usize; 4]) -> [usize; 4] {
        let mut dims = kept;
        match self.axis {
            Some(axis) => dims[axis] = self.k,
            None => dims = [self.k, 1, 1, 1],
        }
        dims
    }

    fn empty(&self, ctx: &Context, kept: [usize; 4]) -> Result<State<T, W>> {
        if self.k == 0 || self.k > u32::MAX as usize / 4 {
            return Err(invalid(format!("top-{} of a stream", self.k)));
        }
        let shape = Shape::new(self.state_dims(kept));
        Ok(State {
            ctx: ctx.clone(),
            kept,
            values: self.identity(ctx)?.broadcast_to(shape),
            indices: V::filled(ctx, shape, -1.0)?,
            pushed: 0,
            pending: Vec::new(),
            pending_elements: 0,
        })
    }

    /// Folds a batch in (the whole-batch kinds: any shape, its elements are one stream in memory
    /// order; the per-position kinds: the lines along the axis are the streams).
    pub fn push(&self, batch: &Array<T>) -> Result<()> {
        self.push_impl(batch, None)
    }

    /// Folds in the elements where `mask` is true (they still take up indices).
    pub fn push_where(&self, batch: &Array<T>, mask: &Array<bool>) -> Result<()> {
        self.push_impl(batch, Some(mask))
    }

    fn push_impl(&self, batch: &Array<T>, mask: Option<&Array<bool>>) -> Result<()> {
        batch.check()?;
        if let Some(mask) = mask {
            mask.check()?;
            if mask.shape() != batch.shape() {
                return Err(LumaError::ShapeMismatch { lhs: batch.shape().dims(), rhs: mask.shape().dims() });
            }
        }
        let ctx = batch.context();
        check_wide::<W>(ctx)?;
        let dims = batch.shape().dims();
        let kept = match self.axis {
            Some(axis) => kept_dims(dims, &[axis]),
            None => [1; 4],
        };
        let mut guard = self.lock();
        if guard.state.is_none() {
            guard.state = Some(self.empty(ctx, kept)?);
        }
        let state = guard.state.as_mut().expect("just made");
        if !ctx.same_as(&state.ctx) {
            return Err(LumaError::ContextMismatch);
        }
        if kept != state.kept {
            return Err(LumaError::ShapeMismatch { lhs: dims, rhs: state.kept });
        }
        match self.axis {
            Some(axis) => self.push_along(state, batch, mask, axis),
            None => self.push_stream(state, batch, mask),
        }
    }

    /// The batch with the elements that cannot win (NaN, masked out) replaced by the identity,
    /// and which elements may win.
    fn prepare(&self, ctx: &Context, batch: &Array<T>, mask: Option<&Array<bool>>) -> Result<(Array<T>, Option<Array<bool>>)> {
        let skip = match (T::not_nan(batch), mask) {
            (Some(not_nan), Some(mask)) => Some(&not_nan & mask),
            (Some(not_nan), None) => Some(not_nan),
            (None, Some(mask)) => Some(mask.clone()),
            (None, None) => None,
        };
        let values = match &skip {
            Some(keep) => select(keep, batch, &self.identity(ctx)?),
            None => batch.clone(),
        };
        Ok((values, skip))
    }

    fn push_stream(&self, state: &mut State<T, W>, batch: &Array<T>, mask: Option<&Array<bool>>) -> Result<()> {
        let ctx = state.ctx.clone();
        let n = batch.len();
        let (values, skip) = self.prepare(&ctx, &batch.reshape(n), mask.map(|m| m.reshape(n)).as_ref())?;
        state.pending.push((values, skip));
        state.pending_elements += n;
        state.pushed += n as u64;
        if state.pending.len() >= FLUSH_EVERY || state.pending_elements >= FLUSH_ELEMENTS {
            self.merge_pending(state)?;
        }
        Ok(())
    }

    /// Sorts the pushes recorded since the last merge together (one sort of their concatenation
    /// is cheaper than a sort per push) and merges their best into the state.
    fn merge_pending(&self, state: &mut State<T, W>) -> Result<()> {
        if state.pending.is_empty() {
            return Ok(());
        }
        let ctx = state.ctx.clone();
        let parts = std::mem::take(&mut state.pending);
        let n = std::mem::take(&mut state.pending_elements);
        let base = state.pushed - n as u64;
        let (values, keep) = if parts.len() == 1 {
            let (v, k) = parts.into_iter().next().expect("one part");
            (v, k)
        } else {
            let masked = parts.iter().any(|p| p.1.is_some());
            let values = concat_slice(0, &parts.iter().map(|p| p.0.clone()).collect::<Vec<_>>());
            let keep = if masked {
                let keeps: Vec<Array<bool>> = parts.iter().map(|p| p.1.clone().map_or_else(|| Array::<bool>::full_on(&ctx, p.0.len(), true), Ok)).collect::<Result<_>>()?;
                Some(concat_slice(0, &keeps))
            } else {
                None
            };
            (values, keep)
        };
        // Pushes that do not fit one buffer are sorted in pieces.
        let ranges = match split_plan(&ctx, [n, 1, 1, 1], T::LAYOUT.stride() as u64, Some(0)) {
            Some((_, ranges)) => ranges,
            None => vec![(0, n)],
        };
        for (a, b) in ranges {
            let (piece, keep) = if ranges_is_whole(a, b, n) { (values.clone(), keep.clone()) } else { (values.slice(0, a..b).cache(), keep.as_ref().map(|k| k.slice(0, a..b).cache())) };
            let take = self.k.min(b - a);
            let (top, local) = if self.largest { piece.topk(take) } else { piece.bottomk(take) };
            let mut global = &V::<W>::from_u32(&local) + (base + a as u64) as f64;
            if let Some(keep) = &keep {
                global = V::select(&keep.take(&local), &global, &V::constant(&ctx, -1.0)?);
            }
            self.fold(state, &top, &global)?;
        }
        Ok(())
    }

    /// Merges candidate entries (values and global indices, best first) into the state.
    fn fold(&self, state: &mut State<T, W>, values: &Array<T>, indices: &V<W>) -> Result<()> {
        let both = concat(0, [&state.values, values]);
        let (top, at) = if self.largest { both.topk(self.k) } else { both.bottomk(self.k) };
        let ids = concat(0, [&state.indices.0, &indices.0]).take(&at);
        eval_all(&[&ids])?;
        top.eval()?;
        state.values = top;
        state.indices = V(ids);
        Ok(())
    }

    // -- Per-position top-k --

    /// The `rounds` best elements along `axis` of every line of `x` (`identity` replaced elements
    /// last), best first: their values and their indices along the axis, with extent `rounds`
    /// along it. Each round is an argmax over `x` with the elements chosen so far masked out
    /// inside the pass (nothing is written), so `rounds` passes read `x` and no more.
    fn best_along(&self, ctx: &Context, x: &Array<T>, axis: usize, rounds: usize) -> Result<(Array<T>, Array<u32>)> {
        let dims = x.shape().dims();
        let identity = self.identity(ctx)?;
        let flat_of = |idx: &Array<u32>| -> Result<Array<u32>> { flat_index(ctx, idx, dims, axis) };
        let ruler = Array::<u32>::range_on(ctx, x.shape(), axis)?;
        let mut picked: Vec<Array<u32>> = Vec::with_capacity(rounds);
        let mut values: Vec<Array<T>> = Vec::with_capacity(rounds);
        for _ in 0..rounds {
            let remaining = match picked.split_first() {
                None => x.clone(),
                Some((first, rest)) => {
                    let mut taken = ruler.eq(first);
                    for p in rest {
                        taken = &taken | &ruler.eq(p);
                    }
                    select(&taken, &identity, x)
                }
            };
            let idx = T::argmin_axis(&remaining, axis, self.largest)?;
            values.push(x.take(&flat_of(&idx)?));
            picked.push(idx);
        }
        Ok((concat_slice(axis, &values), concat_slice(axis, &picked)))
    }

    fn push_along(&self, state: &mut State<T, W>, batch: &Array<T>, mask: Option<&Array<bool>>, axis: usize) -> Result<()> {
        let ctx = state.ctx.clone();
        let dims = batch.shape().dims();
        // A batch that does not fit one buffer is ranked in pieces along the axis.
        let ranges = match split_plan(&ctx, dims, T::LAYOUT.stride() as u64, Some(axis)) {
            Some((split, ranges)) if split == axis => ranges,
            _ => vec![(0, dims[axis])],
        };
        for (a, b) in ranges {
            let whole = a == 0 && b == dims[axis];
            let (piece, piece_mask) = if whole { (batch.clone(), mask.cloned()) } else { (batch.slice(axis, a..b).cache(), mask.map(|m| m.slice(axis, a..b).cache())) };
            let (values, keep) = self.prepare(&ctx, &piece, piece_mask.as_ref())?;
            let piece_dims = piece.shape().dims();
            let (top, local) = self.best_along(&ctx, &values, axis, self.k.min(piece_dims[axis]))?;
            let flat = flat_index(&ctx, &local, piece_dims, axis)?;
            let mut global = &V::<W>::from_u32(&local) + (state.pushed + a as u64) as f64;
            if let Some(keep) = &keep {
                global = V::select(&keep.take(&flat), &global, &V::constant(&ctx, -1.0)?);
            }
            self.fold_along(state, &top, &global, axis)?;
        }
        state.pushed += dims[axis] as u64;
        Ok(())
    }

    /// Merges candidate entries (values and global indices along `axis`) into the state.
    fn fold_along(&self, state: &mut State<T, W>, values: &Array<T>, indices: &V<W>, axis: usize) -> Result<()> {
        let ctx = state.ctx.clone();
        let both = concat(axis, [&state.values, values]);
        let ids = concat(axis, [&state.indices.0, &indices.0]);
        let (top, at) = self.best_along(&ctx, &both, axis, self.k)?;
        let flat = flat_index(&ctx, &at, both.shape().dims(), axis)?;
        let ids = ids.take(&flat);
        eval_all(&[&ids])?;
        top.eval()?;
        state.values = top;
        state.indices = V(ids);
        Ok(())
    }

    /// The values, best first: `k` entries (along the axis, for a per-position top-k).
    pub fn values(&self) -> Result<Array<T>> {
        let mut guard = self.lock();
        let state = guard.state.as_mut().ok_or_else(|| invalid("the top-k is empty: push a batch first"))?;
        self.merge_pending(state)?;
        Ok(state.values.clone())
    }

    /// The global indices of the values (`-1` in slots no element has filled), as wide numbers.
    pub fn indices(&self) -> Result<Array<W>> {
        let mut guard = self.lock();
        let state = guard.state.as_mut().ok_or_else(|| invalid("the top-k is empty: push a batch first"))?;
        self.merge_pending(state)?;
        Ok(state.indices.0.clone())
    }

    /// The values of an index array on the host as `f64`.
    pub fn to_f64(array: &Array<W>) -> Result<Vec<f64>> {
        Ok(array.to_vec()?.into_iter().map(W::to_f64).collect())
    }

    /// The number of elements (samples along the axis, per position) pushed so far: the next
    /// batch's first index.
    pub fn elements_seen(&self) -> u64 {
        self.lock().state.as_ref().map_or(0, |s| s.pushed)
    }

    /// Folds `other` in, as if its elements had been pushed after this one's.
    pub fn merge(&self, other: &TopK<T, W>) -> Result<()> {
        if self.k != other.k || self.largest != other.largest || self.axis != other.axis {
            return Err(invalid("top-k accumulators of different k, direction or axis cannot be merged"));
        }
        let theirs = {
            let mut guard = other.lock();
            let Some(state) = guard.state.as_mut() else { return Ok(()) };
            other.merge_pending(state)?;
            (state.ctx.clone(), state.kept, state.values.clone(), state.indices.clone(), state.pushed)
        };
        let mut guard = self.lock();
        let state = match guard.state.as_mut() {
            Some(state) => state,
            None => guard.state.insert(self.empty(&theirs.0, theirs.1)?),
        };
        self.merge_pending(state)?;
        if state.kept != theirs.1 {
            return Err(LumaError::ShapeMismatch { lhs: state.kept, rhs: theirs.1 });
        }
        let ctx = state.ctx.clone();
        let (values, indices) = if ctx.same_as(&theirs.0) {
            (theirs.2, theirs.3)
        } else {
            let values = Array::from_slice_on(&ctx, &theirs.2.to_vec()?, theirs.2.shape())?;
            let ids = Array::from_slice_on(&ctx, &theirs.3 .0.to_vec()?, theirs.3 .0.shape())?;
            (values, V(ids))
        };
        // Shift the other's indices past this stream's; empty slots stay -1.
        let zero = V::<W>::constant(&ctx, 0.0)?;
        let valid = indices.gt(&(&zero + (-0.5)));
        let shifted = V::select(&valid, &(&indices + state.pushed as f64), &indices);
        match self.axis {
            Some(axis) => self.fold_along(state, &values, &shifted, axis)?,
            None => self.fold(state, &values, &shifted)?,
        }
        state.pushed += theirs.4;
        Ok(())
    }

    /// Clears the state.
    pub fn reset(&self) {
        self.lock().state = None;
    }

    /// The state as bytes, see [`Accumulator::to_bytes`](super::Accumulator::to_bytes).
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        let mut w = Writer::new(KIND);
        w.u64(self.k as u64);
        w.u8(u8::from(self.largest));
        w.u8(self.axis.map_or(0xFF, |a| a as u8));
        let mut guard = self.lock();
        match guard.state.as_mut() {
            None => w.u8(0),
            Some(state) => {
                self.merge_pending(state)?;
                w.u8(1);
                w.u64(state.pushed);
                w.dims(state.kept);
                let values: Vec<f64> = state.values.to_vec()?.into_iter().map(|v| T::to_host_f64(v)).collect();
                w.f64s(&values);
                w.f64s(&state.indices.0.to_vec()?.into_iter().map(W::to_f64).collect::<Vec<_>>());
            }
        }
        Ok(w.0)
    }

    /// Reads a state written by [`to_bytes`](Self::to_bytes) onto the default context.
    pub fn from_bytes(bytes: &[u8]) -> Result<TopK<T, W>> {
        Self::from_bytes_on(&crate::default_context()?, bytes)
    }

    /// Reads a state written by [`to_bytes`](Self::to_bytes) onto `ctx`.
    pub fn from_bytes_on(ctx: &Context, bytes: &[u8]) -> Result<TopK<T, W>> {
        check_wide::<W>(ctx)?;
        let mut r = Reader::new(bytes, KIND)?;
        let k = r.u64()? as usize;
        let largest = r.u8()? != 0;
        let axis = match r.u8()? {
            0xFF => None,
            a => Some(a as usize),
        };
        let top = match axis {
            Some(axis) => Self::along(k, largest, axis)?,
            None => TopK { k, largest, axis: None, inner: Mutex::new(Inner { state: None }) },
        };
        if r.u8()? != 0 {
            let pushed = r.u64()?;
            let kept = r.dims()?;
            let shape = Shape::new(top.state_dims(kept));
            let values: Vec<T> = r.f64s(shape.len())?.into_iter().map(T::from_host_f64).collect();
            let indices: Vec<W> = r.f64s(shape.len())?.into_iter().map(W::from_f64).collect();
            r.finish()?;
            let indices = V(Array::from_slice_on(ctx, &indices, shape)?);
            indices.0.eval()?;
            top.lock().state = Some(State { ctx: ctx.clone(), kept, values: Array::from_slice_on(ctx, &values, shape)?, indices, pushed, pending: Vec::new(), pending_elements: 0 });
        } else {
            r.finish()?;
        }
        Ok(top)
    }
}

/// The flat index into an array of shape `dims` of the elements at index `idx` along `axis` of
/// every line (`idx` has extent 1 or more along `axis`, the shape of the other axes of `dims`).
fn flat_index(ctx: &Context, idx: &Array<u32>, dims: [usize; 4], axis: usize) -> Result<Array<u32>> {
    let st = strides(dims);
    let mut kept = dims;
    kept[axis] = 1;
    let kept_shape = Shape::new(kept);
    let mut flat = idx * st[axis];
    for k in 0..4 {
        if k != axis && dims[k] > 1 {
            flat = &flat + &(&Array::<u32>::range_on(ctx, kept_shape, k)? * st[k]);
        }
    }
    Ok(flat)
}

impl<T: SortKey + AccumCore, W: Wide> std::fmt::Debug for TopK<T, W> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TopK").field("k", &self.k).field("largest", &self.largest).field("axis", &self.axis).field("wide", &W::NAME).finish()
    }
}
