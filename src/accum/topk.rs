//! [`TopK`]: the `k` largest or smallest elements of a stream, with their global indices.

use std::sync::{Mutex, MutexGuard};

use crate::array::Array;
use crate::context::Context;
use crate::error::{LumaError, Result};
use crate::gpu_type::Df32;
use crate::ops::{eval_all, select};
use crate::sort::SortKey;
use crate::view::concat;

use super::blob::{Reader, Writer};
use super::wide::{check_wide, AccumInput, Wide, V};
use super::invalid;

const KIND: u8 = 6;

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
/// Every push is the top `k` of the batch (a sort of the batch) merged with the state's: the
/// cost is that of [`Array::topk`] of the batch plus one small sort of `2k` entries. `T` is
/// `f32`, `u32` or `i32`.
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
pub struct TopK<T: SortKey + AccumInput, W: Wide = Df32> {
    k: usize,
    largest: bool,
    inner: Mutex<Inner<T, W>>,
}

struct Inner<T: SortKey + AccumInput, W: Wide> {
    state: Option<State<T, W>>,
}

struct State<T: SortKey + AccumInput, W: Wide> {
    ctx: Context,
    values: Array<T>,
    /// Global indices as wide numbers, `-1` for an empty slot.
    indices: V<W>,
    /// Elements pushed so far, those of `pending` included (the base of the next batch's indices).
    pushed: u64,
    /// The batches (values with the elements that cannot win replaced by the identity, and the
    /// elements that may win) pushed since the last merge: sorted together, once.
    pending: Vec<(Array<T>, Option<Array<bool>>)>,
    pending_elements: usize,
}

/// Pushes recorded before they are sorted together, and the most elements held back.
const FLUSH_EVERY: usize = 16;
const FLUSH_ELEMENTS: usize = 1 << 24;

impl<T: SortKey + AccumInput, W: Wide> TopK<T, W> {
    /// The `k` largest elements.
    pub fn largest(k: usize) -> TopK<T, W> {
        TopK { k, largest: true, inner: Mutex::new(Inner { state: None }) }
    }

    /// The `k` smallest elements.
    pub fn smallest(k: usize) -> TopK<T, W> {
        TopK { k, largest: false, inner: Mutex::new(Inner { state: None }) }
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

    fn empty(&self, ctx: &Context) -> Result<State<T, W>> {
        if self.k == 0 || self.k > u32::MAX as usize / 4 {
            return Err(invalid(format!("top-{} of a stream", self.k)));
        }
        // The identity: what loses against every element.
        let fill = T::extreme(ctx, !self.largest)?;
        Ok(State { ctx: ctx.clone(), values: fill.broadcast_to(self.k), indices: V::filled(ctx, crate::Shape::from(self.k), -1.0)?, pushed: 0, pending: Vec::new(), pending_elements: 0 })
    }

    /// Folds a batch in (any shape: its elements are one stream in memory order).
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
        let mut guard = self.lock();
        if guard.state.is_none() {
            guard.state = Some(self.empty(ctx)?);
        }
        let state = guard.state.as_mut().expect("just made");
        if !ctx.same_as(&state.ctx) {
            return Err(LumaError::ContextMismatch);
        }
        let n = batch.len();
        let flat = batch.reshape(n);
        let skip = match (T::not_nan(&flat), mask) {
            (Some(not_nan), Some(mask)) => Some(&not_nan & &mask.reshape(n)),
            (Some(not_nan), None) => Some(not_nan),
            (None, Some(mask)) => Some(mask.reshape(n)),
            (None, None) => None,
        };
        // The elements that cannot win are replaced by the identity.
        let values = match &skip {
            Some(keep) => select(keep, &flat, &T::extreme(ctx, !self.largest)?),
            None => flat,
        };
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
            let values = crate::view::concat_slice(0, &parts.iter().map(|p| p.0.clone()).collect::<Vec<_>>());
            let keep = if masked {
                let keeps: Vec<Array<bool>> = parts.iter().map(|p| p.1.clone().map_or_else(|| Array::<bool>::full_on(&ctx, p.0.len(), true), Ok)).collect::<Result<_>>()?;
                Some(crate::view::concat_slice(0, &keeps))
            } else {
                None
            };
            (values, keep)
        };
        // Pushes that do not fit one buffer are sorted in pieces.
        let ranges = match super::split_plan(&ctx, [n, 1, 1, 1], T::LAYOUT.stride() as u64, Some(0)) {
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

    /// The values, best first, `k` entries.
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

    /// The number of elements pushed so far (the next batch's first index).
    pub fn elements_seen(&self) -> u64 {
        self.lock().state.as_ref().map_or(0, |s| s.pushed)
    }

    /// Folds `other` in, as if its elements had been pushed after this one's.
    pub fn merge(&self, other: &TopK<T, W>) -> Result<()> {
        if self.k != other.k || self.largest != other.largest {
            return Err(invalid("top-k accumulators of different k or direction cannot be merged"));
        }
        let theirs = {
            let mut guard = other.lock();
            let Some(state) = guard.state.as_mut() else { return Ok(()) };
            other.merge_pending(state)?;
            (state.ctx.clone(), state.values.clone(), state.indices.clone(), state.pushed)
        };
        let mut guard = self.lock();
        let state = match guard.state.as_mut() {
            Some(state) => state,
            None => guard.state.insert(self.empty(&theirs.0)?),
        };
        self.merge_pending(state)?;
        let ctx = state.ctx.clone();
        let (values, indices) = if ctx.same_as(&theirs.0) {
            (theirs.1, theirs.2)
        } else {
            let values = Array::from_slice_on(&ctx, &theirs.1.to_vec()?, theirs.1.shape())?;
            let ids = Array::from_slice_on(&ctx, &theirs.2 .0.to_vec()?, theirs.2 .0.shape())?;
            (values, V(ids))
        };
        // Shift the other's indices past this stream's; empty slots stay -1.
        let zero = V::<W>::constant(&ctx, 0.0)?;
        let valid = indices.gt(&(&zero + (-0.5)));
        let shifted = V::select(&valid, &(&indices + state.pushed as f64), &indices);
        self.fold(state, &values, &shifted)?;
        state.pushed += theirs.3;
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
        let mut guard = self.lock();
        match guard.state.as_mut() {
            None => w.u8(0),
            Some(state) => {
                self.merge_pending(state)?;
                w.u8(1);
                w.u64(state.pushed);
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
        let top = if largest { TopK::largest(k) } else { TopK::smallest(k) };
        if r.u8()? != 0 {
            let pushed = r.u64()?;
            let values: Vec<T> = r.f64s(k)?.into_iter().map(T::from_host_f64).collect();
            let indices: Vec<W> = r.f64s(k)?.into_iter().map(W::from_f64).collect();
            r.finish()?;
            top.lock().state = Some(State { ctx: ctx.clone(), values: Array::from_slice_on(ctx, &values, k)?, indices: V(Array::from_slice_on(ctx, &indices, k)?), pushed, pending: Vec::new(), pending_elements: 0 });
            eval_all(&[&top.lock().state.as_ref().expect("just set").indices.0])?;
        } else {
            r.finish()?;
        }
        Ok(top)
    }
}

impl<T: SortKey + AccumInput, W: Wide> std::fmt::Debug for TopK<T, W> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TopK").field("k", &self.k).field("largest", &self.largest).field("wide", &W::NAME).finish()
    }
}
