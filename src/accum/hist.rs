//! [`Histogram`] and [`Histogram2d`]: counts in fixed bins.

use std::sync::{Mutex, MutexGuard};

use crate::array::Array;
use crate::context::Context;
use crate::error::{LumaError, Result};
use crate::gpu_type::Df32;
use crate::ops::{eval_all, select};
use crate::shape::Shape;

use super::blob::{Reader, Writer};
use super::wide::{check_wide, AccumInput, Wide, V};
use super::{invalid, kept_dims, Fold};

const KIND_HIST: u8 = 4;
const KIND_HIST2: u8 = 5;
/// Elements counted into the `u32` table of one epoch before it is added to the wide state.
const EPOCH_ELEMENTS: u64 = 1 << 31;
const FLUSH_EVERY: usize = 16;

/// A table of counts: a wide state and the `u32` counts of the pushes since the last merge, which
/// the scatter-add of every push adds to in place.
struct Table<W: Wide> {
    ctx: Context,
    shape: Shape,
    state: V<W>,
    /// For a joint table of `bins` cells (and one more for what is outside): the marginals,
    /// summed from the `u32` counts of every epoch (`Df32` arrays have no axis sums).
    marginals: Option<Marginals<W>>,
    epoch: Array<u32>,
    pushes: usize,
    elements: u64,
}

struct Marginals<W: Wide> {
    bins: [usize; 2],
    x: V<W>,
    y: V<W>,
}

impl<W: Wide> Table<W> {
    fn new(ctx: &Context, shape: Shape) -> Result<Table<W>> {
        Ok(Table { ctx: ctx.clone(), shape, state: V::zeros(ctx, shape)?, marginals: None, epoch: Array::zeros_on(ctx, shape)?, pushes: 0, elements: 0 })
    }

    /// A table of `bins[0] x bins[1]` joint cells and one for the outside, with marginals.
    fn joint(ctx: &Context, bins: [usize; 2]) -> Result<Table<W>> {
        let mut table = Self::new(ctx, Shape::from(bins[0] * bins[1] + 1))?;
        table.marginals = Some(Marginals { bins, x: V::zeros(ctx, Shape::from([bins[0], 1]))?, y: V::zeros(ctx, Shape::from([1, bins[1]]))? });
        Ok(table)
    }

    /// Counts the elements whose flat slot is `idx` (slots of the table's length or more are skipped).
    fn add(&mut self, idx: &Array<u32>, elements: u64) -> Result<()> {
        self.epoch.put_add(idx, 1u32)?;
        self.pushes += 1;
        self.elements += elements;
        if self.pushes >= FLUSH_EVERY || self.elements >= EPOCH_ELEMENTS {
            self.merge()?;
        }
        Ok(())
    }

    fn merge(&mut self) -> Result<()> {
        if self.pushes == 0 {
            return Ok(());
        }
        self.state = &self.state + &V::from_u32(&self.epoch);
        if let Some(m) = self.marginals.as_mut() {
            let joint = self.epoch.slice(0, ..m.bins[0] * m.bins[1]).reshape([m.bins[0], m.bins[1]]);
            m.x = &m.x + &V::from_u32(&joint.sum_axis(1));
            m.y = &m.y + &V::from_u32(&joint.sum_axis(0));
            eval_all(&[&self.state.0, &m.x.0, &m.y.0])?;
        } else {
            eval_all(&[&self.state.0])?;
        }
        self.epoch = Array::zeros_on(&self.ctx, self.shape)?;
        self.pushes = 0;
        self.elements = 0;
        Ok(())
    }

    fn settled(&mut self) -> Result<V<W>> {
        self.merge()?;
        Ok(self.state.clone())
    }

    /// Adds another table's settled counts (and marginals).
    fn absorb(&mut self, other: &V<W>, marginals: Option<(V<W>, V<W>)>) -> Result<()> {
        self.merge()?;
        self.state = &self.state + other;
        if let (Some(m), Some((x, y))) = (self.marginals.as_mut(), marginals) {
            m.x = &m.x + &x;
            m.y = &m.y + &y;
            return eval_all(&[&self.state.0, &m.x.0, &m.y.0]);
        }
        eval_all(&[&self.state.0])
    }

    /// The marginals as of the last merge.
    fn marginals_of(&self) -> Option<(V<W>, V<W>)> {
        self.marginals.as_ref().map(|m| (m.x.clone(), m.y.clone()))
    }

    /// The marginals computed from joint counts on the host (a state read from bytes).
    fn set_state_from_host(&mut self, host: Vec<f64>) -> Result<()> {
        if let Some(m) = self.marginals.as_mut() {
            let [bx, by] = m.bins;
            let (mut x, mut y) = (vec![0.0; bx], vec![0.0; by]);
            for j in 0..by {
                for i in 0..bx {
                    x[i] += host[i + bx * j];
                    y[j] += host[i + bx * j];
                }
            }
            m.x = from_host(&self.ctx, x, Shape::from([bx, 1]))?;
            m.y = from_host(&self.ctx, y, Shape::from([1, by]))?;
        }
        self.state = from_host(&self.ctx, host, self.shape)?;
        Ok(())
    }
}

fn host_f64<W: Wide>(v: &V<W>) -> Result<Vec<f64>> {
    Ok(v.0.to_vec()?.into_iter().map(W::to_f64).collect())
}

fn from_host<W: Wide>(ctx: &Context, host: Vec<f64>, shape: Shape) -> Result<V<W>> {
    let host: Vec<W> = host.into_iter().map(W::from_f64).collect();
    Ok(V(Array::from_slice_on(ctx, &host, shape)?))
}

/// `[0, 0, ..]` strides of a state shape.
fn strides(dims: [usize; 4]) -> [u32; 4] {
    [1, dims[0] as u32, (dims[0] * dims[1]) as u32, (dims[0] * dims[1] * dims[2]) as u32]
}

// -------------------------------------------------------------------------------------------------
// Histogram
// -------------------------------------------------------------------------------------------------

/// Counts in `bins` equal-width bins over `[lo, hi]` of one stream, over the whole batch or per
/// position along chosen axes ([`Fold`]). Bin `k` holds `lo + k w <= x < lo + (k + 1) w` for
/// `w = (hi - lo) / bins`, and the last bin also `x == hi`. Values below `lo`, above `hi` and NaN
/// are counted separately ([`outliers`](Histogram::outliers)). Counts are exact to 2^48 (`Df32`) or
/// 2^53 (`f64` states).
///
/// The result arrays have `bins` entries along the first folded axis (all of the batch folded: a
/// one-dimensional array of `bins`).
///
/// ```no_run
/// use luma::{Array, Df32, Histogram};
/// let hist = Histogram::<Df32>::new(10, 0.0, 1.0)?;
/// hist.push(&Array::<f32>::from_slice(&[0.05, 0.15, 0.95, 1.0, 2.0, f32::NAN], 6)?)?;
/// assert_eq!(Histogram::<Df32>::to_f64(&hist.counts()?)?[0], 1.0);
/// # Ok::<(), luma::LumaError>(())
/// ```
pub struct Histogram<W: Wide = Df32> {
    bins: usize,
    lo: f32,
    hi: f32,
    fold: Fold,
    inner: Mutex<HistInner<W>>,
}

struct HistInner<W: Wide> {
    state: Option<HistState<W>>,
}

struct HistState<W: Wide> {
    kept: [usize; 4],
    first_axis: usize,
    table: Table<W>,
}

impl<W: Wide> Histogram<W> {
    /// A histogram of the whole batch.
    pub fn new(bins: usize, lo: f32, hi: f32) -> Result<Histogram<W>> {
        Self::with_fold(bins, lo, hi, Fold::All)
    }

    /// A histogram per position, folding the axes of `fold`.
    pub fn with_fold(bins: usize, lo: f32, hi: f32, fold: Fold) -> Result<Histogram<W>> {
        if bins == 0 || bins > (u32::MAX - 3) as usize || !(lo.is_finite() && hi.is_finite() && hi > lo) {
            return Err(invalid(format!("a histogram of {bins} bins over [{lo}, {hi}]")));
        }
        Ok(Histogram { bins, lo, hi, fold, inner: Mutex::new(HistInner { state: None }) })
    }

    fn lock(&self) -> MutexGuard<'_, HistInner<W>> {
        self.inner.lock().unwrap_or_else(|p| p.into_inner())
    }

    /// The number of bins.
    pub fn bins(&self) -> usize {
        self.bins
    }

    /// The `bins + 1` bin edges.
    pub fn edges(&self) -> Vec<f32> {
        let w = (f64::from(self.hi) - f64::from(self.lo)) / self.bins as f64;
        (0..=self.bins).map(|k| if k == self.bins { self.hi } else { (f64::from(self.lo) + w * k as f64) as f32 }).collect()
    }

    /// Counts a batch (of any element type; integers are binned by their `f32` value).
    pub fn push<T: AccumInput>(&self, batch: &Array<T>) -> Result<()> {
        self.push_impl(batch, None)
    }

    /// Counts the elements where `mask` is true.
    pub fn push_where<T: AccumInput>(&self, batch: &Array<T>, mask: &Array<bool>) -> Result<()> {
        self.push_impl(batch, Some(mask))
    }

    fn push_impl<T: AccumInput>(&self, batch: &Array<T>, mask: Option<&Array<bool>>) -> Result<()> {
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
        let axes = self.fold.resolve(dims)?;
        let kept = kept_dims(dims, &axes);
        let a0 = axes[0];
        let mut state_dims = kept;
        state_dims[a0] = self.bins + 3;
        let shape = Shape::new(state_dims);
        let mut guard = self.lock();
        if guard.state.is_none() {
            guard.state = Some(HistState { kept, first_axis: a0, table: Table::new(ctx, shape)? });
        }
        let state = guard.state.as_mut().expect("just made");
        if !ctx.same_as(&state.table.ctx) {
            return Err(LumaError::ContextMismatch);
        }
        if kept != state.kept || a0 != state.first_axis {
            return Err(LumaError::ShapeMismatch { lhs: dims, rhs: state.kept });
        }
        let bins = self.bins as u32;
        let x = T::to_f32(batch);
        let scale = self.bins as f32 / (self.hi - self.lo);
        let bin = ((&x - self.lo) * scale).to_u32().minimum(bins - 1);
        let slot = select(&x.is_nan(), bins + 2, &select(&x.lt(self.lo), bins, &select(&x.gt(self.hi), bins + 1, &bin)));
        let st = strides(state_dims);
        let mut idx = &slot * st[a0];
        for k in 0..4 {
            if !axes.contains(&k) && dims[k] > 1 {
                idx = &idx + &(&Array::<u32>::range_on(ctx, batch.shape(), k)? * st[k]);
            }
        }
        if let Some(mask) = mask {
            idx = select(mask, &idx, shape.len() as u32);
        }
        state.table.add(&idx, batch.len() as u64)
    }

    fn settled(&self) -> Result<(Shape, usize, V<W>)> {
        let mut guard = self.lock();
        let state = guard.state.as_mut().ok_or_else(|| invalid("the histogram is empty: push a batch first"))?;
        let v = state.table.settled()?;
        Ok((state.table.shape, state.first_axis, v))
    }

    /// The counts of the bins, `bins` entries along the first folded axis.
    pub fn counts(&self) -> Result<Array<W>> {
        let (_, axis, v) = self.settled()?;
        Ok(v.0.slice(axis, ..self.bins))
    }

    /// The counts below `lo`, above `hi` and of NaN: three entries along the first folded axis.
    pub fn outliers(&self) -> Result<Array<W>> {
        let (_, axis, v) = self.settled()?;
        Ok(v.0.slice(axis, self.bins..self.bins + 3))
    }

    /// The counts in `f32` over their sum: the frequency of every bin among the counted
    /// (in-range) elements. Only for a histogram of the whole batch.
    pub fn density(&self) -> Result<Array<f32>> {
        let counts = V(self.counts()?).narrow();
        Ok(&counts / &counts.sum())
    }

    /// The values of a result array on the host as `f64`.
    pub fn to_f64(array: &Array<W>) -> Result<Vec<f64>> {
        Ok(array.to_vec()?.into_iter().map(W::to_f64).collect())
    }

    /// Adds the counts of `other` (same bins, range and fold; it may live on another device).
    pub fn merge(&self, other: &Histogram<W>) -> Result<()> {
        if self.bins != other.bins || self.lo != other.lo || self.hi != other.hi || self.fold != other.fold {
            return Err(invalid("histograms with different bins, ranges or folds cannot be merged"));
        }
        let theirs = {
            let mut guard = other.lock();
            let Some(state) = guard.state.as_mut() else { return Ok(()) };
            (state.table.ctx.clone(), state.kept, state.first_axis, state.table.shape, state.table.settled()?)
        };
        let mut guard = self.lock();
        match guard.state.as_mut() {
            None => {
                let mut table = Table::new(&theirs.0, theirs.3)?;
                table.state = theirs.4;
                guard.state = Some(HistState { kept: theirs.1, first_axis: theirs.2, table });
                Ok(())
            }
            Some(state) => {
                if state.kept != theirs.1 || state.table.shape != theirs.3 {
                    return Err(LumaError::ShapeMismatch { lhs: state.table.shape.dims(), rhs: theirs.3.dims() });
                }
                let ctx = state.table.ctx.clone();
                let b = if ctx.same_as(&theirs.0) { theirs.4 } else { from_host(&ctx, host_f64(&theirs.4)?, theirs.3)? };
                state.table.absorb(&b, None)
            }
        }
    }

    /// Clears the counts.
    pub fn reset(&self) {
        self.lock().state = None;
    }

    /// The state as bytes, see [`Accumulator::to_bytes`](super::Accumulator::to_bytes).
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        let mut w = Writer::new(KIND_HIST);
        w.u64(self.bins as u64);
        w.f64(f64::from(self.lo));
        w.f64(f64::from(self.hi));
        w.u8(self.fold.code());
        let mut guard = self.lock();
        match guard.state.as_mut() {
            None => w.u8(0),
            Some(state) => {
                w.u8(1);
                w.dims(state.kept);
                w.u64(state.first_axis as u64);
                w.dims(state.table.shape.dims());
                w.f64s(&host_f64(&state.table.settled()?)?);
            }
        }
        Ok(w.0)
    }

    /// Reads a state written by [`to_bytes`](Self::to_bytes) onto the default context.
    pub fn from_bytes(bytes: &[u8]) -> Result<Histogram<W>> {
        Self::from_bytes_on(&crate::default_context()?, bytes)
    }

    /// Reads a state written by [`to_bytes`](Self::to_bytes) onto `ctx`.
    pub fn from_bytes_on(ctx: &Context, bytes: &[u8]) -> Result<Histogram<W>> {
        check_wide::<W>(ctx)?;
        let mut r = Reader::new(bytes, KIND_HIST)?;
        let bins = r.u64()? as usize;
        let (lo, hi) = (r.f64()? as f32, r.f64()? as f32);
        let hist = Histogram::with_fold(bins, lo, hi, Fold::from_code(r.u8()?))?;
        if r.u8()? != 0 {
            let kept = r.dims()?;
            let first_axis = r.u64()? as usize;
            let shape = Shape::new(r.dims()?);
            if first_axis >= 4 || shape.len() > u32::MAX as usize {
                return Err(invalid("serialized histogram state: bad shape"));
            }
            let host = r.f64s(shape.len())?;
            r.finish()?;
            let mut table = Table::new(ctx, shape)?;
            table.state = from_host(ctx, host, shape)?;
            hist.lock().state = Some(HistState { kept, first_axis, table });
        } else {
            r.finish()?;
        }
        Ok(hist)
    }
}

impl<W: Wide> std::fmt::Debug for Histogram<W> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Histogram").field("bins", &self.bins).field("range", &(self.lo, self.hi)).field("wide", &W::NAME).finish()
    }
}

// -------------------------------------------------------------------------------------------------
// Histogram2d
// -------------------------------------------------------------------------------------------------

/// Joint counts of two streams in `bins_x * bins_y` bins: the pairs `(x[i], y[i])` of two arrays
/// of one shape, binned as [`Histogram`] bins (the top edge belongs to the last bin). Pairs with
/// either value out of range or NaN are counted in [`outside`](Histogram2d::outside).
///
/// The marginal counts and the information measures that need only the joint counts
/// ([`entropy_x`](Histogram2d::entropy_x), [`mutual_information`](Histogram2d::mutual_information))
/// come from the counts, as lazy `f32` arrays.
pub struct Histogram2d<W: Wide = Df32> {
    bins: [usize; 2],
    range_x: (f32, f32),
    range_y: (f32, f32),
    inner: Mutex<Hist2Inner<W>>,
}

struct Hist2Inner<W: Wide> {
    table: Option<Table<W>>,
}

impl<W: Wide> Histogram2d<W> {
    /// Joint bins `bins.0 x bins.1` over `[range_x.0, range_x.1] x [range_y.0, range_y.1]`.
    pub fn new(bins: (usize, usize), range_x: (f32, f32), range_y: (f32, f32)) -> Result<Histogram2d<W>> {
        let ok = |r: (f32, f32)| r.0.is_finite() && r.1.is_finite() && r.1 > r.0;
        let cells = bins.0.checked_mul(bins.1);
        if bins.0 == 0 || bins.1 == 0 || cells.is_none_or(|c| c >= u32::MAX as usize) || !ok(range_x) || !ok(range_y) {
            return Err(invalid(format!("a joint histogram of {}x{} bins over {range_x:?} x {range_y:?}", bins.0, bins.1)));
        }
        Ok(Histogram2d { bins: [bins.0, bins.1], range_x, range_y, inner: Mutex::new(Hist2Inner { table: None }) })
    }

    fn lock(&self) -> MutexGuard<'_, Hist2Inner<W>> {
        self.inner.lock().unwrap_or_else(|p| p.into_inner())
    }

    fn cells(&self) -> usize {
        self.bins[0] * self.bins[1]
    }

    /// Counts the pairs `(x[i], y[i])`.
    pub fn push<T: AccumInput, U: AccumInput>(&self, x: &Array<T>, y: &Array<U>) -> Result<()> {
        self.push_impl(x, y, None)
    }

    /// Counts the pairs where `mask` is true.
    pub fn push_where<T: AccumInput, U: AccumInput>(&self, x: &Array<T>, y: &Array<U>, mask: &Array<bool>) -> Result<()> {
        self.push_impl(x, y, Some(mask))
    }

    fn push_impl<T: AccumInput, U: AccumInput>(&self, x: &Array<T>, y: &Array<U>, mask: Option<&Array<bool>>) -> Result<()> {
        x.check()?;
        y.check()?;
        if x.shape() != y.shape() {
            return Err(LumaError::ShapeMismatch { lhs: x.shape().dims(), rhs: y.shape().dims() });
        }
        if let Some(mask) = mask {
            mask.check()?;
            if mask.shape() != x.shape() {
                return Err(LumaError::ShapeMismatch { lhs: x.shape().dims(), rhs: mask.shape().dims() });
            }
        }
        let ctx = x.context();
        if !ctx.same_as(y.context()) {
            return Err(LumaError::ContextMismatch);
        }
        check_wide::<W>(ctx)?;
        let mut guard = self.lock();
        if guard.table.is_none() {
            guard.table = Some(Table::joint(ctx, self.bins)?);
        }
        let table = guard.table.as_mut().expect("just made");
        if !ctx.same_as(&table.ctx) {
            return Err(LumaError::ContextMismatch);
        }
        let bin = |a: &Array<f32>, bins: usize, (lo, hi): (f32, f32)| {
            let scale = bins as f32 / (hi - lo);
            let b = ((a - lo) * scale).to_u32().minimum(bins as u32 - 1);
            // In range: lo <= a <= hi (false for NaN).
            let inside = &a.ge(lo) & &a.le(hi);
            (b, inside)
        };
        let (bx, in_x) = bin(&T::to_f32(x), self.bins[0], self.range_x);
        let (by, in_y) = bin(&U::to_f32(y), self.bins[1], self.range_y);
        let joint = &bx + &(&by * self.bins[0] as u32);
        let ok = &in_x & &in_y;
        let cells = self.cells() as u32;
        let slot = select(&ok, &joint, cells);
        let slot = match mask {
            Some(mask) => select(mask, &slot, cells + 1),
            None => slot,
        };
        table.add(&slot, x.len() as u64)
    }

    fn settled(&self) -> Result<V<W>> {
        let mut guard = self.lock();
        guard.table.as_mut().ok_or_else(|| invalid("the histogram is empty: push a batch first"))?.settled()
    }

    /// The joint counts `[bins_x, bins_y]` (`x` varies fastest).
    pub fn counts(&self) -> Result<Array<W>> {
        Ok(self.settled()?.0.slice(0, ..self.cells()).reshape([self.bins[0], self.bins[1]]))
    }

    /// The number of pairs with a value out of range or NaN, a one-element array (pairs a mask
    /// leaves out are in no count).
    pub fn outside(&self) -> Result<Array<W>> {
        Ok(self.settled()?.0.slice(0, self.cells()..self.cells() + 1))
    }

    /// The counts of the bins of `x` (summed over `y`), `[bins_x, 1]`.
    pub fn marginal_x(&self) -> Result<Array<W>> {
        Ok(self.marginals()?.0)
    }

    /// The counts of the bins of `y` (summed over `x`), `[1, bins_y]`.
    pub fn marginal_y(&self) -> Result<Array<W>> {
        Ok(self.marginals()?.1)
    }

    fn marginals(&self) -> Result<(Array<W>, Array<W>)> {
        let mut guard = self.lock();
        let table = guard.table.as_mut().ok_or_else(|| invalid("the histogram is empty: push a batch first"))?;
        table.merge()?;
        let (x, y) = table.marginals_of().expect("a joint table has marginals");
        Ok((x.0, y.0))
    }

    fn probabilities(&self) -> Result<Array<f32>> {
        let c = V(self.counts()?).narrow();
        Ok(&c / &c.sum())
    }

    /// The entropy of `x` in nats, `-sum p ln p` over the bins, a one-element `f32` array.
    pub fn entropy_x(&self) -> Result<Array<f32>> {
        Ok(entropy(&self.probabilities()?.sum_axis(1)))
    }

    /// The entropy of `y` in nats.
    pub fn entropy_y(&self) -> Result<Array<f32>> {
        Ok(entropy(&self.probabilities()?.sum_axis(0)))
    }

    /// The mutual information of the binned `x` and `y` in nats,
    /// `sum p(x, y) ln(p(x, y) / (p(x) p(y)))`, a one-element `f32` array (0 for independent
    /// streams, up to `ln(bins)` for streams that determine each other).
    pub fn mutual_information(&self) -> Result<Array<f32>> {
        let p = self.probabilities()?;
        let (px, py) = (p.sum_axis(1), p.sum_axis(0));
        let ratio = &p / &(&px * &py);
        Ok(select(&p.gt(0.0), &(&p * &ratio.ln()), 0.0f32).sum())
    }

    /// Adds the counts of `other` (same bins and ranges; it may live on another device).
    pub fn merge(&self, other: &Histogram2d<W>) -> Result<()> {
        if self.bins != other.bins || self.range_x != other.range_x || self.range_y != other.range_y {
            return Err(invalid("histograms with different bins or ranges cannot be merged"));
        }
        let theirs = {
            let mut guard = other.lock();
            let Some(table) = guard.table.as_mut() else { return Ok(()) };
            let state = table.settled()?;
            (table.ctx.clone(), table.shape, state, table.marginals_of())
        };
        let mut guard = self.lock();
        match guard.table.as_mut() {
            None => {
                let mut table = Table::joint(&theirs.0, self.bins)?;
                table.state = theirs.2;
                if let (Some(m), Some((x, y))) = (table.marginals.as_mut(), theirs.3) {
                    m.x = x;
                    m.y = y;
                }
                guard.table = Some(table);
                Ok(())
            }
            Some(table) => {
                let ctx = table.ctx.clone();
                if ctx.same_as(&theirs.0) {
                    table.absorb(&theirs.2, theirs.3)
                } else {
                    let up = |v: &V<W>| -> Result<V<W>> { from_host(&ctx, host_f64(v)?, v.0.shape()) };
                    let marginals = match &theirs.3 {
                        Some((x, y)) => Some((up(x)?, up(y)?)),
                        None => None,
                    };
                    table.absorb(&up(&theirs.2)?, marginals)
                }
            }
        }
    }

    /// Clears the counts.
    pub fn reset(&self) {
        self.lock().table = None;
    }

    /// The state as bytes, see [`Accumulator::to_bytes`](super::Accumulator::to_bytes).
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        let mut w = Writer::new(KIND_HIST2);
        w.u64(self.bins[0] as u64);
        w.u64(self.bins[1] as u64);
        for v in [self.range_x.0, self.range_x.1, self.range_y.0, self.range_y.1] {
            w.f64(f64::from(v));
        }
        let mut guard = self.lock();
        match guard.table.as_mut() {
            None => w.u8(0),
            Some(table) => {
                w.u8(1);
                w.f64s(&host_f64(&table.settled()?)?);
            }
        }
        Ok(w.0)
    }

    /// Reads a state written by [`to_bytes`](Self::to_bytes) onto the default context.
    pub fn from_bytes(bytes: &[u8]) -> Result<Histogram2d<W>> {
        Self::from_bytes_on(&crate::default_context()?, bytes)
    }

    /// Reads a state written by [`to_bytes`](Self::to_bytes) onto `ctx`.
    pub fn from_bytes_on(ctx: &Context, bytes: &[u8]) -> Result<Histogram2d<W>> {
        check_wide::<W>(ctx)?;
        let mut r = Reader::new(bytes, KIND_HIST2)?;
        let bins = (r.u64()? as usize, r.u64()? as usize);
        let mut v = [0f32; 4];
        for x in &mut v {
            *x = r.f64()? as f32;
        }
        let hist = Histogram2d::new(bins, (v[0], v[1]), (v[2], v[3]))?;
        if r.u8()? != 0 {
            let shape = Shape::from(hist.cells() + 1);
            let host = r.f64s(shape.len())?;
            r.finish()?;
            let mut table = Table::joint(ctx, hist.bins)?;
            table.set_state_from_host(host)?;
            hist.lock().table = Some(table);
        } else {
            r.finish()?;
        }
        Ok(hist)
    }
}

impl<W: Wide> std::fmt::Debug for Histogram2d<W> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Histogram2d").field("bins", &self.bins).field("wide", &W::NAME).finish()
    }
}

/// `-sum p ln p` of an array of probabilities (zero entries count 0).
fn entropy(p: &Array<f32>) -> Array<f32> {
    let term = select(&p.gt(0.0), &(p * &p.ln()), 0.0f32);
    -term.sum()
}
