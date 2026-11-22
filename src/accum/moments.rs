//! [`Accumulator`]: count, sum, moments, extrema and their indices of one stream.

use std::any::TypeId;
use std::marker::PhantomData;
use std::sync::{Mutex, MutexGuard};

use crate::array::Array;
use crate::context::Context;
use crate::error::{LumaError, Result};
use crate::gpu_type::Df32;
use crate::ops::{eval_all, select};
use crate::reduce::Reduction;
use crate::shape::Shape;

use super::blob::{Reader, Writer};
use super::wide::{reduce_axes, reduce_flat, AccumInput, Wide, V};
use super::{folded_extent, invalid, is_flat, kept_dims, Fold, NanPolicy};

const KIND: u8 = 1;
const DEFAULT_FLUSH_EVERY: usize = 16;

#[derive(Clone, Debug, PartialEq)]
struct Config {
    fold: Fold,
    order: u8,
    sum: bool,
    min: bool,
    max: bool,
    argmin: bool,
    argmax: bool,
    nan: NanPolicy,
    groups: Option<usize>,
    flush_every: usize,
    precise: bool,
}

impl Config {
    fn track_mean(&self) -> bool {
        self.order >= 1 || self.sum
    }

    /// The state needs the extrema arrays.
    fn wants_min(&self) -> bool {
        self.min || self.argmin
    }

    fn wants_max(&self) -> bool {
        self.max || self.argmax
    }

    fn flags(&self) -> u8 {
        u8::from(self.sum) | u8::from(self.min) << 1 | u8::from(self.max) << 2 | u8::from(self.argmin) << 3 | u8::from(self.argmax) << 4 | u8::from(self.nan == NanPolicy::Skip) << 5 | u8::from(self.precise) << 6
    }
}

/// Configures an [`Accumulator`].
#[derive(Clone, Debug)]
pub struct AccumulatorBuilder<W: Wide = Df32> {
    cfg: Config,
    _wide: PhantomData<W>,
}

impl<W: Wide> AccumulatorBuilder<W> {
    /// Tracks moments up to `order`: 1 the mean, 2 the variance (default), 3 the skewness, 4 the
    /// kurtosis; 0 only the count (and what else is asked for). Every order adds a reduction to
    /// the pass over the batch.
    #[must_use]
    pub fn moments(mut self, order: u8) -> Self {
        self.cfg.order = order.min(4);
        self
    }

    /// Tracks the sum (exact where the arithmetic is, see the module documentation).
    #[must_use]
    pub fn sum(mut self) -> Self {
        self.cfg.sum = true;
        self
    }

    /// Tracks the minimum.
    #[must_use]
    pub fn min(mut self) -> Self {
        self.cfg.min = true;
        self
    }

    /// Tracks the maximum.
    #[must_use]
    pub fn max(mut self) -> Self {
        self.cfg.max = true;
        self
    }

    /// Tracks the minimum and the maximum.
    #[must_use]
    pub fn min_max(self) -> Self {
        self.min().max()
    }

    /// Tracks the global index of the first minimum (a second pass over the batch). Indices count
    /// the samples folded so far along the folded axis (or the elements, folding the whole
    /// batch), and merging another accumulator continues them after this one's. `u32`, `i32` and
    /// `f32` batches.
    #[must_use]
    pub fn argmin(mut self) -> Self {
        self.cfg.argmin = true;
        self
    }

    /// Tracks the global index of the first maximum, see [`argmin`](Self::argmin).
    #[must_use]
    pub fn argmax(mut self) -> Self {
        self.cfg.argmax = true;
        self
    }

    /// Tracks both indices.
    #[must_use]
    pub fn arg_min_max(self) -> Self {
        self.argmin().argmax()
    }

    /// What a push does with NaN, see [`NanPolicy`].
    #[must_use]
    pub fn nan_policy(mut self, nan: NanPolicy) -> Self {
        self.cfg.nan = nan;
        self
    }

    /// Keeps a state per group label (`0..groups`) and position: pushed with
    /// [`Accumulator::push_grouped`]. The result arrays have `groups` entries along the first
    /// folded axis. Folding one axis or the whole batch.
    #[must_use]
    pub fn groups(mut self, groups: usize) -> Self {
        self.cfg.groups = Some(groups);
        self
    }

    /// Reduces the batches in the wide type instead of `f32`: the sums are as exact as the state
    /// (Df32: about 1e-14 relative, f64: 1e-16), at the cost of the wide arithmetic in the pass
    /// over the batch (Df32 is about ten times the `f32` operations per element; f64 runs at the
    /// device's f64 rate). The default pass is `f32` about a pivot, accurate to a few `f32`
    /// roundings relative to the spread of the data. Not available with groups.
    #[must_use]
    pub fn precise(mut self) -> Self {
        self.cfg.precise = true;
        self
    }

    /// The number of pushes recorded before their sums are merged into the state (default 16;
    /// the first push is merged at once so that the pivot of the later ones is the mean of
    /// data). Pushes in between keep their batches alive: more is fewer launches, less is less
    /// memory held.
    #[must_use]
    pub fn flush_every(mut self, pushes: usize) -> Self {
        self.cfg.flush_every = pushes.max(1);
        self
    }

    /// The accumulator, empty until the first push.
    pub fn build(self) -> Result<Accumulator<W>> {
        let cfg = self.cfg;
        if cfg.groups == Some(0) {
            return Err(invalid("an accumulator needs at least one group"));
        }
        if cfg.groups.is_some() && cfg.precise {
            return Err(LumaError::Unsupported("precise passes with groups (the group sums are scatter-adds in f32)".into()));
        }
        if cfg.groups.is_some() && (cfg.argmin || cfg.argmax) {
            return Err(LumaError::Unsupported("argmin / argmax with groups".into()));
        }
        if let Fold::Axes(axes) = &cfg.fold {
            if cfg.groups.is_some() && axes.len() != 1 {
                return Err(invalid("a grouped accumulator folds one axis or the whole batch"));
            }
        }
        Ok(Accumulator { cfg, inner: Mutex::new(Inner { state: None }) })
    }
}

/// The state of a stream: count, sum, mean, variance, higher moments, extrema and their indices,
/// over the whole batch or per position, see the [module documentation](crate::accum).
///
/// All methods take `&self` (the state sits behind a lock), so an accumulator can be shared
/// between threads; pushes from several threads fold in some order.
///
/// ```no_run
/// use luma::{Accumulator, Array, Fold};
/// // per-feature statistics of batches of 256 rows of 8 features ([features, rows]: fold axis 1)
/// let acc = Accumulator::new(Fold::axis(1));
/// for step in 0..100 {
///     let batch = Array::<f32>::from_slice(&vec![step as f32; 8 * 256], [8, 256])?;
///     acc.push(&batch)?;
/// }
/// let mean = acc.mean()?.to_f32().to_vec()?;
/// let var = acc.var()?.to_f32().to_vec()?;
/// # let _ = (mean, var);
/// # Ok::<(), luma::LumaError>(())
/// ```
pub struct Accumulator<W: Wide = Df32> {
    cfg: Config,
    inner: Mutex<Inner<W>>,
}

struct Inner<W: Wide> {
    state: Option<State<W>>,
}

/// The state arrays of a stream, committed: all of the state's shape.
#[derive(Clone)]
struct Fields<W: Wide> {
    n: V<W>,
    mean: Option<V<W>>,
    sum: Option<V<W>>,
    m2: Option<V<W>>,
    m3: Option<V<W>>,
    m4: Option<V<W>>,
    min: Option<V<W>>,
    max: Option<V<W>>,
    imin: Option<V<W>>,
    imax: Option<V<W>>,
}

const FIELD_NAMES: [&str; 10] = ["n", "mean", "sum", "m2", "m3", "m4", "min", "max", "imin", "imax"];

impl<W: Wide> Fields<W> {
    fn slots(&self) -> [Option<&V<W>>; 10] {
        [
            Some(&self.n),
            self.mean.as_ref(),
            self.sum.as_ref(),
            self.m2.as_ref(),
            self.m3.as_ref(),
            self.m4.as_ref(),
            self.min.as_ref(),
            self.max.as_ref(),
            self.imin.as_ref(),
            self.imax.as_ref(),
        ]
    }

    fn from_slots(mut slots: [Option<V<W>>; 10]) -> Fields<W> {
        let [n, mean, sum, m2, m3, m4, min, max, imin, imax] = std::array::from_fn(|i| slots[i].take());
        Fields { n: n.expect("the count is always tracked"), mean, sum, m2, m3, m4, min, max, imin, imax }
    }

    fn arrays(&self) -> Vec<&Array<W>> {
        self.slots().into_iter().flatten().map(|v| &v.0).collect()
    }

    fn zeros(cfg: &Config, ctx: &Context, shape: Shape) -> Result<Fields<W>> {
        let zeros = || V::<W>::zeros(ctx, shape);
        let opt = |on: bool, fill: f64| -> Result<Option<V<W>>> { on.then(|| V::<W>::filled(ctx, shape, fill)).transpose() };
        Ok(Fields {
            n: zeros()?,
            mean: opt(cfg.track_mean(), 0.0)?,
            sum: opt(cfg.sum, 0.0)?,
            m2: opt(cfg.order >= 2, 0.0)?,
            m3: opt(cfg.order >= 3, 0.0)?,
            m4: opt(cfg.order >= 4, 0.0)?,
            min: opt(cfg.wants_min(), f64::INFINITY)?,
            max: opt(cfg.wants_max(), f64::NEG_INFINITY)?,
            imin: opt(cfg.argmin, 0.0)?,
            imax: opt(cfg.argmax, 0.0)?,
        })
    }

    /// The same state on another context, through the host.
    fn transfer(&self, ctx: &Context) -> Result<Fields<W>> {
        let mut out: [Option<V<W>>; 10] = Default::default();
        for (slot, field) in out.iter_mut().zip(self.slots()) {
            if let Some(v) = field {
                let host = v.0.to_vec()?;
                *slot = Some(V(Array::from_slice_on(ctx, &host, v.0.shape())?));
            }
        }
        Ok(Fields::from_slots(out))
    }
}

/// One push, reduced: its count and its sums about the epoch's pivot, its extrema.
struct Partial<W: Wide> {
    n: Count,
    /// `S1 ..= S_order` (or `S1` alone for a sum without moments).
    s: Vec<V<W>>,
    min: Option<V<W>>,
    max: Option<V<W>>,
    imin: Option<V<W>>,
    imax: Option<V<W>>,
}

enum Count {
    /// The same count at every position.
    Uniform(f64),
    Per(Array<u32>),
}

struct Epoch<W: Wide> {
    ty: TypeId,
    /// The rounded `f32` pivot of the default pass.
    k_f32: Option<Array<f32>>,
    k_w: V<W>,
}

struct State<W: Wide> {
    ctx: Context,
    /// The batch's shape with the folded axes set to 1.
    kept: [usize; 4],
    /// The shape of every state array (`kept`, with the group count on the first folded axis).
    shape: Shape,
    fields: Fields<W>,
    /// The first element of the first batch at every position (`kept` shaped).
    pivot0: Array<f32>,
    /// The current pivot as of the last merge (`shape` shaped), unrounded.
    pivot: V<W>,
    /// The pushes recorded since the last merge share a pivot (and an element type).
    epoch: Option<Epoch<W>>,
    pending: Vec<Partial<W>>,
    /// Samples folded so far along the folded axes (the base of the global indices).
    pushed: u64,
    merged_once: bool,
}

impl<W: Wide> Accumulator<W> {
    /// An accumulator folding `fold` with the count, sum, mean and variance (the default
    /// [`builder`](Accumulator::builder) settings).
    ///
    /// # Panics
    /// Never: the default settings are valid for every fold except invalid axes, which fail at the
    /// first push.
    pub fn new(fold: Fold) -> Accumulator<W> {
        Self::builder(fold).sum().build().expect("default accumulator settings are valid")
    }

    /// A builder with the variance tracked (`moments(2)`) and nothing else.
    pub fn builder(fold: Fold) -> AccumulatorBuilder<W> {
        AccumulatorBuilder {
            cfg: Config { fold, order: 2, sum: false, min: false, max: false, argmin: false, argmax: false, nan: NanPolicy::Propagate, groups: None, flush_every: DEFAULT_FLUSH_EVERY, precise: false },
            _wide: PhantomData,
        }
    }

    fn lock(&self) -> MutexGuard<'_, Inner<W>> {
        self.inner.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Folds a batch in. Nothing is read back; the reductions are recorded (fused with the
    /// expression that computes the batch) and merged into the state every
    /// [`flush_every`](AccumulatorBuilder::flush_every) pushes. The batch may have any size and
    /// may be split over several buffers; the shape along the position axes must be the same
    /// at every push.
    pub fn push<T: AccumInput>(&self, batch: &Array<T>) -> Result<()> {
        self.push_impl(batch, None, None)
    }

    /// Folds in the elements where `mask` (of the batch's shape) is true.
    pub fn push_where<T: AccumInput>(&self, batch: &Array<T>, mask: &Array<bool>) -> Result<()> {
        self.push_impl(batch, Some(mask), None)
    }

    /// Folds a batch into per-group states: `labels` has one `u32` label per sample (the extent
    /// of the folded axis, or one per element when the whole batch is folded). Labels of
    /// [`groups`](AccumulatorBuilder::groups) or more are skipped.
    pub fn push_grouped<T: AccumInput>(&self, batch: &Array<T>, labels: &Array<u32>) -> Result<()> {
        self.push_impl(batch, None, Some(labels))
    }

    /// [`push_grouped`](Self::push_grouped) of the elements where `mask` is true.
    pub fn push_grouped_where<T: AccumInput>(&self, batch: &Array<T>, labels: &Array<u32>, mask: &Array<bool>) -> Result<()> {
        self.push_impl(batch, Some(mask), Some(labels))
    }

    fn push_impl<T: AccumInput>(&self, batch: &Array<T>, user_mask: Option<&Array<bool>>, labels: Option<&Array<u32>>) -> Result<()> {
        batch.check()?;
        if let Some(mask) = user_mask {
            mask.check()?;
            if mask.shape() != batch.shape() {
                return Err(LumaError::ShapeMismatch { lhs: batch.shape().dims(), rhs: mask.shape().dims() });
            }
        }
        if let Some(labels) = labels {
            labels.check()?;
        }
        if self.cfg.groups.is_some() != labels.is_some() {
            return Err(invalid("`push_grouped` is for accumulators built with `groups`, and only for them"));
        }
        let ctx = batch.context();
        super::wide::check_wide::<W>(ctx)?;
        let dims = batch.shape().dims();
        let axes = self.cfg.fold.resolve(dims)?;
        super::wide::check_precise_axes::<W>(self.cfg.precise, dims, &axes)?;
        let kept = kept_dims(dims, &axes);
        let mut guard = self.lock();
        if guard.state.is_none() {
            guard.state = Some(State::new(&self.cfg, ctx, batch, &axes, kept)?);
        }
        let state = guard.state.as_mut().expect("the state was just made");
        if !ctx.same_as(&state.ctx) {
            return Err(LumaError::ContextMismatch);
        }
        if kept != state.kept {
            return Err(LumaError::ShapeMismatch { lhs: dims, rhs: state.kept });
        }
        let ty = TypeId::of::<T>();
        if state.epoch.as_ref().is_some_and(|e| e.ty != ty) {
            state.merge_pending(&self.cfg)?;
        }
        if state.epoch.is_none() {
            state.epoch = Some(if self.cfg.precise {
                state.pivot.0.eval()?;
                Epoch { ty, k_f32: None, k_w: state.pivot.clone() }
            } else {
                let k = T::tidy_pivot(state.pivot.narrow());
                k.eval()?;
                Epoch { ty, k_w: V::from_f32(&k), k_f32: Some(k) }
            });
        }
        let mask = effective_mask(&self.cfg, batch, user_mask);
        let partial = state.partial(&self.cfg, batch, mask.as_ref(), labels, &axes)?;
        state.pending.push(partial);
        state.pushed += folded_extent(dims, &axes);
        let limit = if state.merged_once { self.cfg.flush_every } else { 1 };
        if state.pending.len() >= limit {
            state.merge_pending(&self.cfg)?;
        }
        Ok(())
    }

    /// Merges the sums of the pushes recorded so far into the state (done by every result method
    /// and by `merge`, `to_bytes`; useful to bound the memory of held batches).
    pub fn flush(&self) -> Result<()> {
        let mut guard = self.lock();
        if let Some(state) = guard.state.as_mut() {
            state.merge_pending(&self.cfg)?;
        }
        Ok(())
    }

    /// Clears the state; the accumulator is empty again and takes batches of any position shape.
    pub fn reset(&self) {
        self.lock().state = None;
    }

    /// Whether anything has been pushed or merged since the last reset.
    pub fn is_empty(&self) -> bool {
        self.lock().state.is_none()
    }

    /// The number of samples folded so far along the folded axes (elements, folding the whole
    /// batch), masked out or not.
    pub fn samples_seen(&self) -> u64 {
        self.lock().state.as_ref().map_or(0, |s| s.pushed)
    }

    /// The committed state: `(context, shape, fields)`.
    fn settled(&self) -> Result<(Context, Shape, Fields<W>)> {
        let mut guard = self.lock();
        let state = guard.state.as_mut().ok_or_else(|| invalid("the accumulator is empty: push a batch first"))?;
        state.merge_pending(&self.cfg)?;
        Ok((state.ctx.clone(), state.shape, state.fields.clone()))
    }

    fn field(&self, what: &'static str, pick: impl Fn(&Fields<W>) -> Option<&V<W>>) -> Result<V<W>> {
        let (_, _, fields) = self.settled()?;
        pick(&fields).cloned().ok_or_else(|| invalid(format!("{what} is not tracked by this accumulator")))
    }

    /// The number of samples at every position (exact), in the wide type.
    pub fn count(&self) -> Result<Array<W>> {
        Ok(self.settled()?.2.n.0)
    }

    /// The sum at every position; the accumulator must have been built with `sum()`.
    pub fn sum(&self) -> Result<Array<W>> {
        Ok(self.field("the sum", |f| f.sum.as_ref())?.0)
    }

    /// The mean at every position (NaN at positions without samples).
    pub fn mean(&self) -> Result<Array<W>> {
        let (ctx, _, f) = self.settled()?;
        let _ = ctx;
        f.mean.map(|m| m.0).ok_or_else(|| invalid("the mean is not tracked (`moments(0)` without `sum()`)"))
    }

    /// The population variance (divisor `n`).
    pub fn var(&self) -> Result<Array<W>> {
        self.var_ddof(0)
    }

    /// The variance with divisor `n - ddof` (1 for the unbiased sample variance).
    pub fn var_ddof(&self, ddof: u32) -> Result<Array<W>> {
        let (ctx, _, f) = self.settled()?;
        let m2 = f.m2.ok_or_else(|| invalid("the variance needs `moments(2)` or more"))?;
        let denom = &f.n + (-f64::from(ddof));
        let _ = ctx;
        Ok((&m2 / &denom).0)
    }

    /// The population standard deviation.
    pub fn std(&self) -> Result<Array<W>> {
        Ok(V(self.var()?).sqrt().0)
    }

    /// The skewness `m3 / m2^1.5` (needs `moments(3)`).
    pub fn skewness(&self) -> Result<Array<W>> {
        let (_, _, f) = self.settled()?;
        let (m2, m3) = f.m2.zip(f.m3).ok_or_else(|| invalid("the skewness needs `moments(3)`"))?;
        let n = &f.n;
        let var = &m2 / n;
        Ok(((&m3 / n) / (&var * &var.sqrt())).0)
    }

    /// The excess kurtosis `m4 / m2^2 - 3` (needs `moments(4)`).
    pub fn kurtosis(&self) -> Result<Array<W>> {
        let (_, _, f) = self.settled()?;
        let (m2, m4) = f.m2.zip(f.m4).ok_or_else(|| invalid("the kurtosis needs `moments(4)`"))?;
        let n = &f.n;
        let var = &m2 / n;
        Ok((((&m4 / n) / (&var * &var)) + (-3.0)).0)
    }

    /// The minimum (`+inf` at positions without samples).
    pub fn min(&self) -> Result<Array<W>> {
        Ok(self.field("the minimum", |f| f.min.as_ref())?.0)
    }

    /// The maximum (`-inf` at positions without samples).
    pub fn max(&self) -> Result<Array<W>> {
        Ok(self.field("the maximum", |f| f.max.as_ref())?.0)
    }

    /// The global sample index of the first minimum.
    pub fn argmin(&self) -> Result<Array<W>> {
        Ok(self.field("the index of the minimum", |f| f.imin.as_ref())?.0)
    }

    /// The global sample index of the first maximum.
    pub fn argmax(&self) -> Result<Array<W>> {
        Ok(self.field("the index of the maximum", |f| f.imax.as_ref())?.0)
    }

    /// The values of a result array on the host as `f64` (exact for both wide types).
    pub fn to_f64(array: &Array<W>) -> Result<Vec<f64>> {
        Ok(array.to_vec()?.into_iter().map(W::to_f64).collect())
    }

    /// Folds the state of `other` in, as if its samples had been pushed after this accumulator's.
    /// `other` may live on another device (the state travels through the host). Both must have
    /// the same settings and position shape; merging an empty accumulator changes nothing.
    pub fn merge(&self, other: &Accumulator<W>) -> Result<()> {
        if self.cfg != other.cfg {
            return Err(invalid("accumulators with different settings cannot be merged"));
        }
        let snapshot = if std::ptr::eq(self, other) { self.snapshot()? } else { other.snapshot()? };
        let Some(snapshot) = snapshot else { return Ok(()) };
        let mut guard = self.lock();
        match guard.state.as_mut() {
            None => {
                guard.state = Some(State::from_snapshot(&self.cfg, snapshot)?);
                Ok(())
            }
            Some(state) => {
                state.merge_pending(&self.cfg)?;
                state.absorb(&self.cfg, snapshot)
            }
        }
    }

    fn snapshot(&self) -> Result<Option<Snapshot<W>>> {
        let mut guard = self.lock();
        let Some(state) = guard.state.as_mut() else { return Ok(None) };
        state.merge_pending(&self.cfg)?;
        Ok(Some(Snapshot { ctx: state.ctx.clone(), kept: state.kept, shape: state.shape, fields: state.fields.clone(), pivot0: state.pivot0.clone(), pushed: state.pushed }))
    }

    /// The state as bytes, for [`from_bytes`](Self::from_bytes) on any machine and either wide type.
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        let mut w = Writer::new(KIND);
        let cfg = &self.cfg;
        w.u8(cfg.fold.code());
        w.u8(cfg.order);
        w.u8(cfg.flags());
        w.u64(cfg.groups.map_or(0, |g| g as u64 + 1));
        w.u64(cfg.flush_every as u64);
        let snapshot = self.snapshot()?;
        w.u8(u8::from(snapshot.is_some()));
        if let Some(s) = snapshot {
            w.dims(s.kept);
            w.dims(s.shape.dims());
            w.u64(s.pushed);
            w.f64s(&s.pivot0.to_vec()?.into_iter().map(f64::from).collect::<Vec<_>>());
            for (_, field) in FIELD_NAMES.iter().zip(s.fields.slots()) {
                if let Some(v) = field {
                    w.f64s(&v.0.to_vec()?.into_iter().map(W::to_f64).collect::<Vec<_>>());
                }
            }
        }
        Ok(w.0)
    }

    /// Reads a state written by [`to_bytes`](Self::to_bytes) onto the default context.
    pub fn from_bytes(bytes: &[u8]) -> Result<Accumulator<W>> {
        Self::from_bytes_on(&crate::default_context()?, bytes)
    }

    /// Reads a state written by [`to_bytes`](Self::to_bytes) onto `ctx`.
    pub fn from_bytes_on(ctx: &Context, bytes: &[u8]) -> Result<Accumulator<W>> {
        super::wide::check_wide::<W>(ctx)?;
        let mut r = Reader::new(bytes, KIND)?;
        let fold = Fold::from_code(r.u8()?);
        let order = r.u8()?;
        let flags = r.u8()?;
        let groups = r.u64()?;
        let flush_every = r.u64()? as usize;
        let cfg = Config {
            fold,
            order,
            sum: flags & 1 != 0,
            min: flags & 2 != 0,
            max: flags & 4 != 0,
            argmin: flags & 8 != 0,
            argmax: flags & 16 != 0,
            nan: if flags & 32 != 0 { NanPolicy::Skip } else { NanPolicy::Propagate },
            groups: (groups > 0).then(|| (groups - 1) as usize),
            flush_every: flush_every.max(1),
            precise: flags & 64 != 0,
        };
        let acc = AccumulatorBuilder { cfg: cfg.clone(), _wide: PhantomData }.build()?;
        if r.u8()? != 0 {
            let kept = r.dims()?;
            let shape = Shape::new(r.dims()?);
            let pushed = r.u64()?;
            let len = shape.len();
            let kept_len: usize = kept.iter().product();
            let pivot0: Vec<f32> = r.f64s(kept_len)?.into_iter().map(|v| v as f32).collect();
            let mut slots: [Option<V<W>>; 10] = Default::default();
            let present = Fields::<W>::zeros(&cfg, ctx, Shape::from(1usize))?;
            for (slot, field) in slots.iter_mut().zip(present.slots()) {
                if field.is_some() {
                    let host: Vec<W> = r.f64s(len)?.into_iter().map(W::from_f64).collect();
                    *slot = Some(V(Array::from_slice_on(ctx, &host, shape)?));
                }
            }
            r.finish()?;
            let fields = Fields::from_slots(slots);
            let pivot0 = Array::from_slice_on(ctx, &pivot0, Shape::new(kept))?;
            let snapshot = Snapshot { ctx: ctx.clone(), kept, shape, fields, pivot0, pushed };
            acc.lock().state = Some(State::from_snapshot(&cfg, snapshot)?);
        } else {
            r.finish()?;
        }
        Ok(acc)
    }
}

impl<W: Wide> Clone for Accumulator<W> {
    /// A copy with its own state (the arrays are shared handles of values that are never
    /// modified): the two are pushed to independently.
    fn clone(&self) -> Self {
        let state = {
            let mut guard = self.lock();
            guard.state.as_mut().map(|s| {
                let _ = s.merge_pending(&self.cfg);
                State {
                    ctx: s.ctx.clone(),
                    kept: s.kept,
                    shape: s.shape,
                    fields: s.fields.clone(),
                    pivot0: s.pivot0.clone(),
                    pivot: s.pivot.clone(),
                    epoch: None,
                    pending: Vec::new(),
                    pushed: s.pushed,
                    merged_once: s.merged_once,
                }
            })
        };
        Accumulator { cfg: self.cfg.clone(), inner: Mutex::new(Inner { state }) }
    }
}

impl<W: Wide> std::fmt::Debug for Accumulator<W> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Accumulator").field("wide", &W::NAME).field("config", &self.cfg).field("samples", &self.samples_seen()).finish()
    }
}

struct Snapshot<W: Wide> {
    ctx: Context,
    kept: [usize; 4],
    shape: Shape,
    fields: Fields<W>,
    pivot0: Array<f32>,
    pushed: u64,
}

/// The user's mask and the NaN policy as one mask.
fn effective_mask<T: AccumInput>(cfg: &Config, batch: &Array<T>, mask: Option<&Array<bool>>) -> Option<Array<bool>> {
    let not_nan = (cfg.nan == NanPolicy::Skip).then(|| T::not_nan(batch)).flatten();
    match (mask, not_nan) {
        (Some(m), Some(n)) => Some(m & &n),
        (Some(m), None) => Some(m.clone()),
        (None, n) => n,
    }
}

/// Reductions of `x` over the folded axes (the reduction of one array, or per line).
pub(super) fn reduce_folded<T: crate::gpu_type::Numeric>(x: &Array<T>, axes: &[usize], ops: &[Reduction]) -> Vec<Array<T>> {
    let dims = x.shape().dims();
    if is_flat(dims, axes) {
        return reduce_flat(x, ops);
    }
    let real: Vec<usize> = axes.iter().copied().filter(|&a| dims[a] > 1).collect();
    // Nothing to reduce but extent-1 axes: reducing one of them is the identity (and the square).
    let real = if real.is_empty() { vec![axes[0]] } else { real };
    reduce_axes(x, &real, ops)
}

impl<W: Wide> State<W> {
    fn new<T: AccumInput>(cfg: &Config, ctx: &Context, batch: &Array<T>, axes: &[usize], kept: [usize; 4]) -> Result<State<W>> {
        let mut dims_g = kept;
        if let Some(groups) = cfg.groups {
            dims_g[axes[0]] = groups;
        }
        let shape = Shape::new(dims_g);
        let first = axes.iter().fold(batch.clone(), |a, &ax| a.slice(ax, ..1));
        let first = T::to_f32(&first);
        let pivot0 = T::tidy_pivot(select(&first.is_finite(), &first, 0.0f32));
        pivot0.eval()?;
        let pivot = V::from_f32(&pivot0.broadcast_to(shape));
        Ok(State {
            ctx: ctx.clone(),
            kept,
            shape,
            fields: Fields::zeros(cfg, ctx, shape)?,
            pivot0,
            pivot,
            epoch: None,
            pending: Vec::new(),
            pushed: 0,
            merged_once: false,
        })
    }

    fn from_snapshot(cfg: &Config, s: Snapshot<W>) -> Result<State<W>> {
        let mut state = State {
            ctx: s.ctx.clone(),
            kept: s.kept,
            shape: s.shape,
            fields: s.fields,
            pivot0: s.pivot0.clone(),
            pivot: V::from_f32(&s.pivot0.broadcast_to(s.shape)),
            epoch: None,
            pending: Vec::new(),
            pushed: s.pushed,
            merged_once: true,
        };
        state.refresh_pivot(cfg)?;
        Ok(state)
    }

    /// The pivot of the next pushes: the mean where there are samples, the first element elsewhere.
    fn refresh_pivot(&mut self, cfg: &Config) -> Result<()> {
        if cfg.track_mean() {
            let mean = self.fields.mean.as_ref().expect("the mean is tracked");
            let have = self.fields.n.gt(&V::constant(&self.ctx, 0.0)?);
            self.pivot = V::select(&have, mean, &V::from_f32(&self.pivot0));
            self.pivot.0.eval()?;
        }
        self.epoch = None;
        Ok(())
    }

    /// One push's reductions.
    fn partial<T: AccumInput>(&self, cfg: &Config, batch: &Array<T>, mask: Option<&Array<bool>>, labels: Option<&Array<u32>>, axes: &[usize]) -> Result<Partial<W>> {
        match labels {
            Some(labels) => self.partial_grouped(cfg, batch, mask, labels, axes),
            None => self.partial_plain(cfg, batch, mask, axes),
        }
    }

    fn partial_plain<T: AccumInput>(&self, cfg: &Config, batch: &Array<T>, mask: Option<&Array<bool>>, axes: &[usize]) -> Result<Partial<W>> {
        let ctx = &self.ctx;
        let dims = batch.shape().dims();
        let epoch = self.epoch.as_ref().expect("the epoch is open");
        // Sums of powers of the shifted values.
        let mut s: Vec<V<W>> = Vec::new();
        let order = cfg.order.max(1);
        if cfg.track_mean() && cfg.precise {
            let mut d = &V(T::widen::<W>(batch)) - &epoch.k_w;
            if let Some(mask) = mask {
                d = V::select(mask, &d, &V::constant(ctx, 0.0)?);
            }
            let d2 = &d * &d;
            let mut terms = vec![d.clone(), d2.clone()];
            terms.truncate(order.min(2) as usize);
            if order >= 3 {
                terms.push(&d2 * &d);
            }
            if order >= 4 {
                terms.push(&d2 * &d2);
            }
            for term in &terms {
                s.push(V(reduce_folded(&term.0, axes, &[Reduction::Sum]).remove(0)));
            }
        } else if cfg.track_mean() {
            let pivot = epoch.k_f32.as_ref().expect("the default pass has an f32 pivot");
            let mut d = T::shifted(batch, pivot);
            if let Some(mask) = mask {
                d = select(mask, &d, 0.0f32);
            }
            let mut ops = vec![Reduction::Sum];
            if order >= 2 {
                ops.push(Reduction::SumOfSquares);
            }
            s.extend(reduce_folded(&d, axes, &ops).iter().map(V::from_f32));
            if order >= 3 {
                let d2 = &d * &d;
                s.extend(reduce_folded(&(&d2 * &d), axes, &[Reduction::Sum]).iter().map(V::from_f32));
                if order >= 4 {
                    s.extend(reduce_folded(&d2, axes, &[Reduction::SumOfSquares]).iter().map(V::from_f32));
                }
            }
        }
        let n = match mask {
            Some(mask) => Count::Per(reduce_folded(&mask.to_u32(), axes, &[Reduction::Sum]).remove(0)),
            None => Count::Uniform(folded_extent(dims, axes) as f64),
        };
        // Extrema.
        let extreme = |highest: bool| -> Result<Array<T>> {
            Ok(match mask {
                Some(mask) => select(mask, batch, &T::extreme(ctx, highest)?),
                None => batch.clone(),
            })
        };
        let widen = |a: &Array<T>| V(T::widen::<W>(a));
        let (mut min, mut max) = (None, None);
        let (mut imin, mut imax) = (None, None);
        let base = self.pushed as f64;
        let global = |local: Array<u32>| -> V<W> { &V::<W>::from_u32(&local) + base };
        let shape_idx = |max_: bool, hi: &Array<T>| -> Result<Array<u32>> {
            let real: Vec<usize> = axes.iter().copied().filter(|&a| dims[a] > 1).collect();
            if is_flat(dims, axes) {
                T::argmin_flat(hi, max_)
            } else if real.len() == 1 {
                T::argmin_axis(hi, real[0], max_)
            } else if real.is_empty() {
                Array::zeros_on(ctx, Shape::new(kept_dims(dims, axes)))
            } else {
                Err(LumaError::Unsupported("argmin / argmax over several axes that are not the whole batch".into()))
            }
        };
        if cfg.wants_min() || cfg.wants_max() {
            let (lo_src, hi_src) = if mask.is_some() { (Some(extreme(true)?), Some(extreme(false)?)) } else { (None, None) };
            let lo_x = lo_src.as_ref().unwrap_or(batch);
            let hi_x = hi_src.as_ref().unwrap_or(batch);
            if mask.is_none() && cfg.wants_min() && cfg.wants_max() {
                let mm = reduce_folded(batch, axes, &[Reduction::Min, Reduction::Max]);
                min = Some(widen(&mm[0]));
                max = Some(widen(&mm[1]));
            } else {
                if cfg.wants_min() {
                    min = Some(widen(&reduce_folded(lo_x, axes, &[Reduction::Min]).remove(0)));
                }
                if cfg.wants_max() {
                    max = Some(widen(&reduce_folded(hi_x, axes, &[Reduction::Max]).remove(0)));
                }
            }
            if cfg.argmin {
                imin = Some(global(shape_idx(false, lo_x)?));
            }
            if cfg.argmax {
                imax = Some(global(shape_idx(true, hi_x)?));
            }
        }
        Ok(Partial { n, s, min, max, imin, imax })
    }

    fn partial_grouped<T: AccumInput>(&self, cfg: &Config, batch: &Array<T>, mask: Option<&Array<bool>>, labels: &Array<u32>, axes: &[usize]) -> Result<Partial<W>> {
        let ctx = &self.ctx;
        let dims = batch.shape().dims();
        let groups = cfg.groups.expect("grouped push");
        let ga = axes[0];
        let pivot = self.epoch.as_ref().expect("the epoch is open").k_f32.as_ref().expect("grouped pushes use the f32 pass");
        // The batch-shaped flat index into the state of every element.
        let strides: [u32; 4] = {
            let sd = self.shape.dims();
            [1, sd[0] as u32, (sd[0] * sd[1]) as u32, (sd[0] * sd[1] * sd[2]) as u32]
        };
        let whole = axes.len() == 4;
        let expected = if whole { batch.len() } else { dims[ga] };
        if labels.len() != expected {
            return Err(invalid(format!("{} labels for {} samples", labels.len(), expected)));
        }
        let label_shape = if whole {
            batch.shape()
        } else {
            let mut d = [1usize; 4];
            d[ga] = dims[ga];
            Shape::new(d)
        };
        let labels = labels.reshape(label_shape);
        let mut idx = &labels * strides[ga];
        for k in 0..4 {
            if !axes.contains(&k) && dims[k] > 1 {
                idx = &idx + &(&Array::<u32>::range_on(ctx, batch.shape(), k)? * strides[k]);
            }
        }
        let idx = idx.broadcast_to(batch.shape());
        let mut ok = labels.lt(groups as u32).broadcast_to(batch.shape());
        if let Some(mask) = mask {
            ok = &ok & mask;
        }
        let idx = select(&ok, &idx, self.shape.len() as u32);
        let mut s = Vec::new();
        if cfg.track_mean() {
            let d = T::shifted(batch, &pivot.take(&idx));
            let order = cfg.order.max(1);
            let mut powers = vec![d.clone()];
            if order >= 2 {
                powers.push(&d * &d);
            }
            if order >= 3 {
                powers.push(&(&d * &d) * &d);
            }
            if order >= 4 {
                let d2 = &d * &d;
                powers.push(&d2 * &d2);
            }
            for p in &powers {
                let mut target = Array::<f32>::zeros_on(ctx, self.shape)?;
                target.put_add(&idx, p)?;
                s.push(V::from_f32(&target));
            }
        }
        let mut counts = Array::<u32>::zeros_on(ctx, self.shape)?;
        counts.put_add(&idx, 1u32)?;
        let (mut min, mut max) = (None, None);
        if cfg.wants_min() {
            min = Some(V(T::widen::<W>(&T::scatter_extreme(ctx, self.shape, &idx, batch, false)?)));
        }
        if cfg.wants_max() {
            max = Some(V(T::widen::<W>(&T::scatter_extreme(ctx, self.shape, &idx, batch, true)?)));
        }
        Ok(Partial { n: Count::Per(counts), s, min, max, imin: None, imax: None })
    }

    /// Folds the recorded pushes into the committed state.
    fn merge_pending(&mut self, cfg: &Config) -> Result<()> {
        if self.pending.is_empty() {
            return Ok(());
        }
        let ctx = self.ctx.clone();
        let shape = self.shape;
        let pending = std::mem::take(&mut self.pending);
        let pivot = self.epoch.as_ref().map(|e| e.k_w.clone());
        let zero = V::<W>::constant(&ctx, 0.0)?;
        // Counts.
        let mut uniform = 0.0;
        let mut per: Option<V<W>> = None;
        for p in &pending {
            match &p.n {
                Count::Uniform(c) => uniform += c,
                Count::Per(a) => {
                    let v = V::<W>::from_u32(a);
                    per = Some(match per {
                        Some(acc) => &acc + &v,
                        None => v,
                    });
                }
            }
        }
        let n_b = match per {
            Some(per) => &per + uniform,
            None => V::<W>::filled(&ctx, shape, uniform)?,
        };
        let f = &self.fields;
        let n = &f.n + &n_b;
        let mut next = f.clone();
        next.n = n.clone();
        if cfg.track_mean() {
            let pivot = pivot.expect("a push opened the epoch");
            let order = cfg.order.max(1) as usize;
            let mut s: Vec<V<W>> = Vec::with_capacity(order);
            for k in 0..order.min(pending[0].s.len()) {
                let mut acc: Option<V<W>> = None;
                for p in &pending {
                    let v = p.s[k].clone();
                    acc = Some(match acc {
                        Some(a) => &a + &v,
                        None => v,
                    });
                }
                s.push(acc.expect("a pending push"));
            }
            let n_safe = n_b.maximum(&V::constant(&ctx, 1.0)?);
            let m = &s[0] / &n_safe;
            let mean_b = &pivot + &m;
            if cfg.sum {
                next.sum = Some(f.sum.as_ref().expect("sum tracked") + &(&(&n_b * &pivot) + &s[0]));
            }
            // Centered moments of the pushes.
            let m2_b = (order >= 2).then(|| &s[1] - &(&s[0] * &m));
            let m3_b = (order >= 3).then(|| {
                let m2 = &m * &m;
                &(&s[2] - &(&(&m * &s[1]) * 3.0)) + &(&(&m2 * &s[0]) * 2.0)
            });
            let m4_b = (order >= 4).then(|| {
                let m2 = &m * &m;
                let m3 = &m2 * &m;
                &(&(&s[3] - &(&(&m * &s[2]) * 4.0)) + &(&(&m2 * &s[1]) * 6.0)) - &(&(&m3 * &s[0]) * 3.0)
            });
            let b = Central { n: n_b.clone(), mean: mean_b, m2: m2_b, m3: m3_b, m4: m4_b };
            let a = Central { n: f.n.clone(), mean: f.mean.clone().expect("mean tracked"), m2: f.m2.clone(), m3: f.m3.clone(), m4: f.m4.clone() };
            let c = chan(&a, &b, &ctx)?;
            next.mean = Some(c.mean);
            next.m2 = c.m2;
            next.m3 = c.m3;
            next.m4 = c.m4;
        }
        // Extrema, partial by partial in push order.
        if cfg.wants_min() || cfg.wants_max() {
            let mut run = f.n.clone();
            for p in &pending {
                let n_i = match &p.n {
                    Count::Uniform(c) => V::<W>::filled(&ctx, shape, *c)?,
                    Count::Per(a) => V::<W>::from_u32(a),
                };
                if let (Some(b), Some(cur)) = (&p.min, next.min.as_ref()) {
                    let (v, i) = extreme_step(&zero, &run, &n_i, cur, next.imin.as_ref(), b, p.imin.as_ref(), false);
                    next.min = Some(v);
                    next.imin = i.or(next.imin.take());
                }
                if let (Some(b), Some(cur)) = (&p.max, next.max.as_ref()) {
                    let (v, i) = extreme_step(&zero, &run, &n_i, cur, next.imax.as_ref(), b, p.imax.as_ref(), true);
                    next.max = Some(v);
                    next.imax = i.or(next.imax.take());
                }
                run = &run + &n_i;
            }
        }
        self.fields = next;
        let arrays = self.fields.arrays();
        eval_all(&arrays)?;
        self.merged_once = true;
        self.refresh_pivot(cfg)
    }

    /// Folds another committed state in (its samples come after this one's).
    fn absorb(&mut self, cfg: &Config, other: Snapshot<W>) -> Result<()> {
        if other.kept != self.kept || other.shape != self.shape {
            return Err(LumaError::ShapeMismatch { lhs: self.shape.dims(), rhs: other.shape.dims() });
        }
        let ctx = self.ctx.clone();
        let theirs = if ctx.same_as(&other.ctx) { other.fields } else { other.fields.transfer(&ctx)? };
        let ours = self.fields.clone();
        let mut next = ours.clone();
        next.n = &ours.n + &theirs.n;
        if cfg.track_mean() {
            let a = Central { n: ours.n.clone(), mean: ours.mean.clone().expect("mean tracked"), m2: ours.m2.clone(), m3: ours.m3.clone(), m4: ours.m4.clone() };
            let b = Central { n: theirs.n.clone(), mean: theirs.mean.clone().expect("mean tracked"), m2: theirs.m2.clone(), m3: theirs.m3.clone(), m4: theirs.m4.clone() };
            let c = chan(&a, &b, &ctx)?;
            next.mean = Some(c.mean);
            next.m2 = c.m2;
            next.m3 = c.m3;
            next.m4 = c.m4;
        }
        if cfg.sum {
            next.sum = Some(ours.sum.as_ref().expect("sum tracked") + theirs.sum.as_ref().expect("sum tracked"));
        }
        let zero = V::<W>::constant(&ctx, 0.0)?;
        let offset = self.pushed as f64;
        if let (Some(cur), Some(b)) = (ours.min.as_ref(), theirs.min.as_ref()) {
            let bi = theirs.imin.as_ref().map(|i| i + offset);
            let (v, i) = extreme_step(&zero, &ours.n, &theirs.n, cur, ours.imin.as_ref(), b, bi.as_ref(), false);
            next.min = Some(v);
            next.imin = i.or(next.imin.take());
        }
        if let (Some(cur), Some(b)) = (ours.max.as_ref(), theirs.max.as_ref()) {
            let bi = theirs.imax.as_ref().map(|i| i + offset);
            let (v, i) = extreme_step(&zero, &ours.n, &theirs.n, cur, ours.imax.as_ref(), b, bi.as_ref(), true);
            next.max = Some(v);
            next.imax = i.or(next.imax.take());
        }
        self.fields = next;
        self.pushed += other.pushed;
        let arrays = self.fields.arrays();
        eval_all(&arrays)?;
        self.merged_once = true;
        self.refresh_pivot(cfg)
    }
}

/// The central moments of a set of samples.
struct Central<W: Wide> {
    n: V<W>,
    mean: V<W>,
    m2: Option<V<W>>,
    m3: Option<V<W>>,
    m4: Option<V<W>>,
}

/// Chan et al.'s pairwise merge of the moments of two sets of samples (either may be empty).
fn chan<W: Wide>(a: &Central<W>, b: &Central<W>, ctx: &Context) -> Result<Central<W>> {
    let n = &a.n + &b.n;
    let n_safe = n.maximum(&V::constant(ctx, 1.0)?);
    let ra = &a.n / &n_safe;
    let rb = &b.n / &n_safe;
    let delta = &b.mean - &a.mean;
    let mean = &a.mean + &(&delta * &rb);
    let d2 = &delta * &delta;
    let m2 = match (&a.m2, &b.m2) {
        (Some(a2), Some(b2)) => Some(&(a2 + b2) + &(&(&d2 * &a.n) * &rb)),
        _ => None,
    };
    let m3 = match (&a.m3, &b.m3, &a.m2, &b.m2) {
        (Some(a3), Some(b3), Some(a2), Some(b2)) => {
            let d3 = &d2 * &delta;
            let t1 = &(&(&d3 * &a.n) * &rb) * &(&ra - &rb);
            let t2 = &(&delta * &(&(&ra * b2) - &(&rb * a2))) * 3.0;
            Some(&(&(a3 + b3) + &t1) + &t2)
        }
        _ => None,
    };
    let m4 = match (&a.m4, &b.m4, &a.m3, &b.m3, &a.m2, &b.m2) {
        (Some(a4), Some(b4), Some(a3), Some(b3), Some(a2), Some(b2)) => {
            let d4 = &d2 * &d2;
            let t1 = &(&(&d4 * &a.n) * &rb) * &(&(&(&ra * &ra) - &(&ra * &rb)) + &(&rb * &rb));
            let t2 = &(&d2 * &(&(&(&ra * &ra) * b2) + &(&(&rb * &rb) * a2))) * 6.0;
            let t3 = &(&delta * &(&(&ra * b3) - &(&rb * a3))) * 4.0;
            Some(&(&(&(a4 + b4) + &t1) + &t2) + &t3)
        }
        _ => None,
    };
    Ok(Central { n, mean, m2, m3, m4 })
}

/// One step of a running extremum: `cur` (with `n_run` samples behind it, index `cur_i`) against
/// `b` (with `n_b` samples, index `b_i`): `b` wins if it has samples and is strictly better, or
/// if `cur` has none.
#[allow(clippy::too_many_arguments)]
fn extreme_step<W: Wide>(zero: &V<W>, n_run: &V<W>, n_b: &V<W>, cur: &V<W>, cur_i: Option<&V<W>>, b: &V<W>, b_i: Option<&V<W>>, max: bool) -> (V<W>, Option<V<W>>) {
    let better = if max { b.gt(cur) } else { b.lt(cur) };
    let empty = !n_run.gt(zero);
    let take = &n_b.gt(zero) & &(&empty | &better);
    let value = V::select(&take, b, cur);
    let index = match (cur_i, b_i) {
        (Some(ci), Some(bi)) => Some(V::select(&take, bi, ci)),
        _ => None,
    };
    (value, index)
}
