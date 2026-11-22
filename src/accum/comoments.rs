//! [`CoMoments`] (two streams, per position) and [`Covariance`] (the features of one stream).

use std::any::TypeId;
use std::marker::PhantomData;
use std::sync::{Mutex, MutexGuard};

use crate::array::Array;
use crate::context::Context;
use crate::error::{LumaError, Result};
use crate::gpu_type::Df32;
use crate::matmul::matmul_f32;
use crate::ops::{eval_all, select};
use crate::reduce::Reduction;
use crate::shape::Shape;

use super::blob::{Reader, Writer};
use super::moments::reduce_folded;
use super::wide::{check_wide, AccumCore, AccumInput, Wide, V};
use super::{folded_extent, invalid, kept_dims, Fold, NanPolicy};

const KIND_CO: u8 = 2;
const KIND_COV: u8 = 3;

// -------------------------------------------------------------------------------------------------
// CoMoments
// -------------------------------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq)]
struct CoConfig {
    fold: Fold,
    nan: NanPolicy,
    flush_every: usize,
    precise: bool,
}

/// Configures a [`CoMoments`].
#[derive(Clone, Debug)]
pub struct CoMomentsBuilder<W: Wide = Df32> {
    cfg: CoConfig,
    _wide: PhantomData<W>,
}

impl<W: Wide> CoMomentsBuilder<W> {
    /// What a push does with NaN: [`NanPolicy::Skip`] leaves out the pairs in which either value
    /// is NaN.
    #[must_use]
    pub fn nan_policy(mut self, nan: NanPolicy) -> Self {
        self.cfg.nan = nan;
        self
    }

    /// Reduces in the wide type, see [`AccumulatorBuilder::precise`](super::AccumulatorBuilder::precise).
    #[must_use]
    pub fn precise(mut self) -> Self {
        self.cfg.precise = true;
        self
    }

    /// Pushes recorded before their sums are merged, see
    /// [`AccumulatorBuilder::flush_every`](super::AccumulatorBuilder::flush_every).
    #[must_use]
    pub fn flush_every(mut self, pushes: usize) -> Self {
        self.cfg.flush_every = pushes.max(1);
        self
    }

    /// The accumulator, empty until the first push.
    pub fn build(self) -> CoMoments<W> {
        CoMoments { cfg: self.cfg, inner: Mutex::new(CoInner { state: None }) }
    }
}

/// The joint state of two streams `x` and `y` of the same shape, per position: count, both means
/// and variances, the covariance and the correlation (Chan's pairwise update for the sum of
/// products too, with the same pivoting as [`Accumulator`](super::Accumulator)). Pushing `x` and
/// `y` of one array each over the folded axes gives the covariance between the streams; pushing
/// one array against another of the same shape gives per-position covariances.
pub struct CoMoments<W: Wide = Df32> {
    cfg: CoConfig,
    inner: Mutex<CoInner<W>>,
}

struct CoInner<W: Wide> {
    state: Option<CoState<W>>,
}

#[derive(Clone)]
struct CoFields<W: Wide> {
    n: V<W>,
    mx: V<W>,
    my: V<W>,
    m2x: V<W>,
    m2y: V<W>,
    cxy: V<W>,
}

impl<W: Wide> CoFields<W> {
    fn list(&self) -> [&V<W>; 6] {
        [&self.n, &self.mx, &self.my, &self.m2x, &self.m2y, &self.cxy]
    }

    fn from_list(v: [V<W>; 6]) -> CoFields<W> {
        let [n, mx, my, m2x, m2y, cxy] = v;
        CoFields { n, mx, my, m2x, m2y, cxy }
    }
}

struct CoPartial<W: Wide> {
    n: Count,
    /// `Sx, Sy, Sxx, Syy, Sxy`.
    s: Vec<V<W>>,
}

enum Count {
    Uniform(f64),
    Per(Array<u32>),
}

struct CoEpoch<W: Wide> {
    ty: (TypeId, TypeId),
    k: Option<(Array<f32>, Array<f32>)>,
    kw: (V<W>, V<W>),
}

struct CoState<W: Wide> {
    ctx: Context,
    kept: [usize; 4],
    pivot0: (Array<f32>, Array<f32>),
    pivot: (V<W>, V<W>),
    fields: CoFields<W>,
    epoch: Option<CoEpoch<W>>,
    pending: Vec<CoPartial<W>>,
    merged_once: bool,
}

fn first_sample<T: AccumCore>(batch: &Array<T>, axes: &[usize]) -> Array<f32> {
    let first = axes.iter().fold(batch.clone(), |a, &ax| a.slice(ax, ..1));
    let first = T::to_f32(&first);
    T::tidy_pivot(select(&first.is_finite(), &first, 0.0f32))
}

impl<W: Wide> CoMoments<W> {
    /// An accumulator folding `fold`.
    pub fn new(fold: Fold) -> CoMoments<W> {
        Self::builder(fold).build()
    }

    /// A builder.
    pub fn builder(fold: Fold) -> CoMomentsBuilder<W> {
        CoMomentsBuilder { cfg: CoConfig { fold, nan: NanPolicy::Propagate, flush_every: 16, precise: false }, _wide: PhantomData }
    }

    fn lock(&self) -> MutexGuard<'_, CoInner<W>> {
        self.inner.lock().unwrap_or_else(|p| p.into_inner())
    }

    /// Folds in the pairs `(x[i], y[i])`; the arrays have one shape.
    pub fn push<T: AccumInput, U: AccumInput>(&self, x: &Array<T>, y: &Array<U>) -> Result<()> {
        self.push_impl(&T::to_core(x), &U::to_core(y), None)
    }

    /// Folds in the pairs where `mask` is true.
    pub fn push_where<T: AccumInput, U: AccumInput>(&self, x: &Array<T>, y: &Array<U>, mask: &Array<bool>) -> Result<()> {
        self.push_impl(&T::to_core(x), &U::to_core(y), Some(mask))
    }

    fn push_impl<T: AccumCore, U: AccumCore>(&self, x: &Array<T>, y: &Array<U>, user_mask: Option<&Array<bool>>) -> Result<()> {
        x.check()?;
        y.check()?;
        if x.shape() != y.shape() {
            return Err(LumaError::ShapeMismatch { lhs: x.shape().dims(), rhs: y.shape().dims() });
        }
        if let Some(mask) = user_mask {
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
        let dims = x.shape().dims();
        let axes = self.cfg.fold.resolve(dims)?;
        super::wide::check_precise_axes::<W>(self.cfg.precise, dims, &axes)?;
        let kept = kept_dims(dims, &axes);
        let mut guard = self.lock();
        if guard.state.is_none() {
            let shape = Shape::new(kept);
            let (p0x, p0y) = (first_sample(x, &axes), first_sample(y, &axes));
            p0x.eval()?;
            p0y.eval()?;
            guard.state = Some(CoState {
                ctx: ctx.clone(),
                kept,
                pivot: (V::from_f32(&p0x), V::from_f32(&p0y)),
                pivot0: (p0x, p0y),
                fields: CoFields::from_list(std::array::from_fn(|_| V::zeros(ctx, shape).expect("a small array"))),
                epoch: None,
                pending: Vec::new(),
                merged_once: false,
            });
        }
        let state = guard.state.as_mut().expect("just made");
        if !ctx.same_as(&state.ctx) {
            return Err(LumaError::ContextMismatch);
        }
        if kept != state.kept {
            return Err(LumaError::ShapeMismatch { lhs: dims, rhs: state.kept });
        }
        let ty = (TypeId::of::<T>(), TypeId::of::<U>());
        if state.epoch.as_ref().is_some_and(|e| e.ty != ty) {
            state.merge_pending(&self.cfg)?;
        }
        if state.epoch.is_none() {
            state.epoch = Some(if self.cfg.precise {
                state.pivot.0 .0.eval()?;
                state.pivot.1 .0.eval()?;
                CoEpoch { ty, k: None, kw: state.pivot.clone() }
            } else {
                let kx = T::tidy_pivot(state.pivot.0.narrow());
                let ky = U::tidy_pivot(state.pivot.1.narrow());
                kx.eval()?;
                ky.eval()?;
                CoEpoch { ty, kw: (V::from_f32(&kx), V::from_f32(&ky)), k: Some((kx, ky)) }
            });
        }
        let mut mask = user_mask.cloned();
        if self.cfg.nan == NanPolicy::Skip {
            for nn in [T::not_nan(x), U::not_nan(y)].into_iter().flatten() {
                mask = Some(match mask {
                    Some(m) => &m & &nn,
                    None => nn,
                });
            }
        }
        let partial = state.partial(x, y, mask.as_ref(), &axes)?;
        state.pending.push(partial);
        let limit = if state.merged_once { self.cfg.flush_every } else { 1 };
        if state.pending.len() >= limit {
            state.merge_pending(&self.cfg)?;
        }
        Ok(())
    }

    /// Merges the recorded pushes into the state.
    pub fn flush(&self) -> Result<()> {
        if let Some(state) = self.lock().state.as_mut() {
            state.merge_pending(&self.cfg)?;
        }
        Ok(())
    }

    /// Clears the state.
    pub fn reset(&self) {
        self.lock().state = None;
    }

    fn settled(&self) -> Result<(Context, CoFields<W>)> {
        let mut guard = self.lock();
        let state = guard.state.as_mut().ok_or_else(|| invalid("the accumulator is empty: push a batch first"))?;
        state.merge_pending(&self.cfg)?;
        Ok((state.ctx.clone(), state.fields.clone()))
    }

    /// The number of pairs at every position.
    pub fn count(&self) -> Result<Array<W>> {
        Ok(self.settled()?.1.n.0)
    }

    /// The mean of `x`.
    pub fn mean_x(&self) -> Result<Array<W>> {
        Ok(self.settled()?.1.mx.0)
    }

    /// The mean of `y`.
    pub fn mean_y(&self) -> Result<Array<W>> {
        Ok(self.settled()?.1.my.0)
    }

    /// The population variance of `x`.
    pub fn var_x(&self) -> Result<Array<W>> {
        let (_, f) = self.settled()?;
        Ok((&f.m2x / &f.n).0)
    }

    /// The population variance of `y`.
    pub fn var_y(&self) -> Result<Array<W>> {
        let (_, f) = self.settled()?;
        Ok((&f.m2y / &f.n).0)
    }

    /// The population covariance (divisor `n`).
    pub fn cov(&self) -> Result<Array<W>> {
        self.cov_ddof(0)
    }

    /// The covariance with divisor `n - ddof`.
    pub fn cov_ddof(&self, ddof: u32) -> Result<Array<W>> {
        let (_, f) = self.settled()?;
        Ok((&f.cxy / &(&f.n + (-f64::from(ddof)))).0)
    }

    /// The Pearson correlation `cov / (std_x * std_y)` (NaN where a variance is 0).
    pub fn corr(&self) -> Result<Array<W>> {
        let (_, f) = self.settled()?;
        Ok((&f.cxy / &(&f.m2x * &f.m2y).sqrt()).0)
    }

    /// The least-squares slope of `y` on `x`, `cov / var_x`.
    pub fn slope(&self) -> Result<Array<W>> {
        let (_, f) = self.settled()?;
        Ok((&f.cxy / &f.m2x).0)
    }

    /// Folds `other`'s state in. Both must have the same settings and position shape; `other` may
    /// live on another device.
    pub fn merge(&self, other: &CoMoments<W>) -> Result<()> {
        if (CoConfig { flush_every: other.cfg.flush_every, ..self.cfg.clone() }) != other.cfg {
            return Err(invalid("accumulators with different settings cannot be merged"));
        }
        let theirs = {
            let mut guard = other.lock();
            let Some(state) = guard.state.as_mut() else { return Ok(()) };
            state.merge_pending(&other.cfg)?;
            (state.ctx.clone(), state.kept, state.fields.clone(), state.pivot0.clone())
        };
        let mut guard = self.lock();
        match guard.state.as_mut() {
            None => {
                let mut state = CoState {
                    ctx: theirs.0.clone(),
                    kept: theirs.1,
                    pivot: (V::from_f32(&theirs.3 .0), V::from_f32(&theirs.3 .1)),
                    pivot0: theirs.3,
                    fields: theirs.2,
                    epoch: None,
                    pending: Vec::new(),
                    merged_once: true,
                };
                state.refresh_pivot()?;
                guard.state = Some(state);
                Ok(())
            }
            Some(state) => {
                state.merge_pending(&self.cfg)?;
                if theirs.1 != state.kept {
                    return Err(LumaError::ShapeMismatch { lhs: state.kept, rhs: theirs.1 });
                }
                let ctx = state.ctx.clone();
                let b = if ctx.same_as(&theirs.0) { theirs.2 } else { transfer(&theirs.2, &ctx)? };
                state.fields = co_chan(&state.fields, &b, &ctx)?;
                eval_all(&state.fields.list().map(|v| &v.0))?;
                state.refresh_pivot()
            }
        }
    }

    /// The state as bytes, see [`Accumulator::to_bytes`](super::Accumulator::to_bytes).
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        let mut w = Writer::new(KIND_CO);
        w.u8(self.cfg.fold.code());
        w.u8(u8::from(self.cfg.nan == NanPolicy::Skip) | u8::from(self.cfg.precise) << 1);
        w.u64(self.cfg.flush_every as u64);
        let mut guard = self.lock();
        match guard.state.as_mut() {
            None => w.u8(0),
            Some(state) => {
                state.merge_pending(&self.cfg)?;
                w.u8(1);
                w.dims(state.kept);
                for p in [&state.pivot0.0, &state.pivot0.1] {
                    w.f64s(&p.to_vec()?.into_iter().map(f64::from).collect::<Vec<_>>());
                }
                for v in state.fields.list() {
                    w.f64s(&v.0.to_vec()?.into_iter().map(W::to_f64).collect::<Vec<_>>());
                }
            }
        }
        Ok(w.0)
    }

    /// Reads a state written by [`to_bytes`](Self::to_bytes) onto the default context.
    pub fn from_bytes(bytes: &[u8]) -> Result<CoMoments<W>> {
        Self::from_bytes_on(&crate::default_context()?, bytes)
    }

    /// Reads a state written by [`to_bytes`](Self::to_bytes) onto `ctx`.
    pub fn from_bytes_on(ctx: &Context, bytes: &[u8]) -> Result<CoMoments<W>> {
        check_wide::<W>(ctx)?;
        let mut r = Reader::new(bytes, KIND_CO)?;
        let fold = Fold::from_code(r.u8()?);
        let flags = r.u8()?;
        let flush_every = (r.u64()? as usize).max(1);
        let cfg = CoConfig { fold, nan: if flags & 1 != 0 { NanPolicy::Skip } else { NanPolicy::Propagate }, flush_every, precise: flags & 2 != 0 };
        let acc = CoMoments { cfg, inner: Mutex::new(CoInner { state: None }) };
        if r.u8()? != 0 {
            let kept = r.dims()?;
            let len: usize = kept.iter().product();
            let shape = Shape::new(kept);
            let mut p0 = Vec::new();
            for _ in 0..2 {
                let host: Vec<f32> = r.f64s(len)?.into_iter().map(|v| v as f32).collect();
                p0.push(Array::from_slice_on(ctx, &host, shape)?);
            }
            let mut fields = Vec::new();
            for _ in 0..6 {
                let host: Vec<W> = r.f64s(len)?.into_iter().map(W::from_f64).collect();
                fields.push(V(Array::from_slice_on(ctx, &host, shape)?));
            }
            r.finish()?;
            let pivot0 = (p0.remove(0), p0.remove(0));
            let mut state = CoState {
                ctx: ctx.clone(),
                kept,
                pivot: (V::from_f32(&pivot0.0), V::from_f32(&pivot0.1)),
                pivot0,
                fields: CoFields::from_list(fields.try_into().map_err(|_| invalid("state fields"))?),
                epoch: None,
                pending: Vec::new(),
                merged_once: true,
            };
            state.refresh_pivot()?;
            acc.lock().state = Some(state);
        } else {
            r.finish()?;
        }
        Ok(acc)
    }
}

impl<W: Wide> std::fmt::Debug for CoMoments<W> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CoMoments").field("wide", &W::NAME).field("config", &self.cfg).finish()
    }
}

fn transfer<W: Wide>(f: &CoFields<W>, ctx: &Context) -> Result<CoFields<W>> {
    let mut out = Vec::new();
    for v in f.list() {
        out.push(V(Array::from_slice_on(ctx, &v.0.to_vec()?, v.0.shape())?));
    }
    Ok(CoFields::from_list(out.try_into().map_err(|_| invalid("state fields"))?))
}

/// Chan's merge of the joint moments.
fn co_chan<W: Wide>(a: &CoFields<W>, b: &CoFields<W>, ctx: &Context) -> Result<CoFields<W>> {
    let n = &a.n + &b.n;
    let n_safe = n.maximum(&V::constant(ctx, 1.0)?);
    let rb = &b.n / &n_safe;
    let dx = &b.mx - &a.mx;
    let dy = &b.my - &a.my;
    let w = &a.n * &rb;
    Ok(CoFields {
        n,
        mx: &a.mx + &(&dx * &rb),
        my: &a.my + &(&dy * &rb),
        m2x: &(&a.m2x + &b.m2x) + &(&(&dx * &dx) * &w),
        m2y: &(&a.m2y + &b.m2y) + &(&(&dy * &dy) * &w),
        cxy: &(&a.cxy + &b.cxy) + &(&(&dx * &dy) * &w),
    })
}

impl<W: Wide> CoState<W> {
    fn refresh_pivot(&mut self) -> Result<()> {
        let have = self.fields.n.gt(&V::constant(&self.ctx, 0.0)?);
        self.pivot = (V::select(&have, &self.fields.mx, &V::from_f32(&self.pivot0.0)), V::select(&have, &self.fields.my, &V::from_f32(&self.pivot0.1)));
        self.pivot.0 .0.eval()?;
        self.pivot.1 .0.eval()?;
        self.epoch = None;
        Ok(())
    }

    fn partial<T: AccumCore, U: AccumCore>(&self, x: &Array<T>, y: &Array<U>, mask: Option<&Array<bool>>, axes: &[usize]) -> Result<CoPartial<W>> {
        let ctx = &self.ctx;
        let dims = x.shape().dims();
        let epoch = self.epoch.as_ref().expect("the epoch is open");
        let mut s: Vec<V<W>> = Vec::new();
        if let Some((kx, ky)) = &epoch.k {
            let (mut dx, mut dy) = (T::shifted(x, kx), U::shifted(y, ky));
            if let Some(mask) = mask {
                dx = select(mask, &dx, 0.0f32);
                dy = select(mask, &dy, 0.0f32);
            }
            let r = reduce_folded(&dx, axes, &[Reduction::Sum, Reduction::SumOfSquares]);
            s.extend(r.iter().map(V::from_f32));
            let r = reduce_folded(&dy, axes, &[Reduction::Sum, Reduction::SumOfSquares]);
            s.extend(r.iter().map(V::from_f32));
            // Order: Sx, Sxx, Sy, Syy; Sxy last.
            s.swap(1, 2);
            s.extend(reduce_folded(&(&dx * &dy), axes, &[Reduction::Sum]).iter().map(V::from_f32));
        } else {
            let (mut dx, mut dy) = (&V(T::widen::<W>(x)) - &epoch.kw.0, &V(U::widen::<W>(y)) - &epoch.kw.1);
            if let Some(mask) = mask {
                let zero = V::constant(ctx, 0.0)?;
                dx = V::select(mask, &dx, &zero);
                dy = V::select(mask, &dy, &zero);
            }
            for term in [dx.clone(), dy.clone(), &dx * &dx, &dy * &dy, &dx * &dy] {
                s.push(V(reduce_folded(&term.0, axes, &[Reduction::Sum]).remove(0)));
            }
        }
        let n = match mask {
            Some(mask) => Count::Per(reduce_folded(&mask.to_u32(), axes, &[Reduction::Sum]).remove(0)),
            None => Count::Uniform(folded_extent(dims, axes) as f64),
        };
        Ok(CoPartial { n, s })
    }

    fn merge_pending(&mut self, _cfg: &CoConfig) -> Result<()> {
        if self.pending.is_empty() {
            return Ok(());
        }
        let ctx = self.ctx.clone();
        let pending = std::mem::take(&mut self.pending);
        let epoch = self.epoch.as_ref().expect("a push opened the epoch");
        let (kx, ky) = (epoch.kw.0.clone(), epoch.kw.1.clone());
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
            None => V::<W>::constant(&ctx, uniform)?,
        };
        let mut s: Vec<V<W>> = Vec::with_capacity(5);
        for k in 0..5 {
            let mut acc: Option<V<W>> = None;
            for p in &pending {
                acc = Some(match acc {
                    Some(a) => &a + &p.s[k],
                    None => p.s[k].clone(),
                });
            }
            s.push(acc.expect("a pending push"));
        }
        let n_safe = n_b.maximum(&V::constant(&ctx, 1.0)?);
        let (mx, my) = (&s[0] / &n_safe, &s[1] / &n_safe);
        // `s` is Sx, Sy, Sxx, Syy, Sxy (both passes, see `partial`).
        let b = CoFields {
            n: n_b,
            mx: &kx + &mx,
            my: &ky + &my,
            m2x: &s[2] - &(&s[0] * &mx),
            m2y: &s[3] - &(&s[1] * &my),
            cxy: &s[4] - &(&s[0] * &my),
        };
        self.fields = co_chan(&self.fields, &b, &ctx)?;
        eval_all(&self.fields.list().map(|v| &v.0))?;
        self.merged_once = true;
        self.refresh_pivot()
    }
}

// -------------------------------------------------------------------------------------------------
// Covariance
// -------------------------------------------------------------------------------------------------

/// The covariance and correlation matrix between the features of a stream of samples, folded in
/// batches of `[features, samples]` arrays (`samples` rows of `features`, row-major): the state
/// is the count, the mean of every feature and the `features x features` matrix of sums of
/// products about the mean, merged with Chan's update. One push is a full-precision `f32`
/// matrix product of the pivot-centered batch with itself (`matmul_f32`, never TF32), merged
/// into a [`Df32`] or `f64` state.
///
/// ```no_run
/// use luma::{Array, Covariance, Df32};
/// let cov = Covariance::<Df32>::new();
/// for _ in 0..10 {
///     cov.push(&Array::<f32>::from_slice(&vec![0.5; 4 * 100], [4, 100])?)?;
/// }
/// let matrix = cov.cov()?; // [4, 4]
/// # let _ = matrix;
/// # Ok::<(), luma::LumaError>(())
/// ```
pub struct Covariance<W: Wide = Df32> {
    flush_every: usize,
    inner: Mutex<CovInner<W>>,
}

struct CovInner<W: Wide> {
    state: Option<CovState<W>>,
}

struct CovState<W: Wide> {
    ctx: Context,
    features: usize,
    n: V<W>,
    mean: V<W>,
    c: V<W>,
    pivot0: Array<f32>,
    pivot: V<W>,
    /// The `f32` pivot of the pushes recorded since the last merge, `[features, 1]`.
    epoch: Option<(Array<f32>, V<W>)>,
    pending: Vec<(f64, V<W>, V<W>)>,
    merged_once: bool,
}

impl<W: Wide> Covariance<W> {
    /// An empty accumulator.
    pub fn new() -> Covariance<W> {
        Covariance { flush_every: 16, inner: Mutex::new(CovInner { state: None }) }
    }

    /// An accumulator merging every `pushes` pushes (default 16).
    pub fn with_flush_every(pushes: usize) -> Covariance<W> {
        Covariance { flush_every: pushes.max(1), inner: Mutex::new(CovInner { state: None }) }
    }

    fn lock(&self) -> MutexGuard<'_, CovInner<W>> {
        self.inner.lock().unwrap_or_else(|p| p.into_inner())
    }

    /// Folds a `[features, samples]` batch (an `f32` array, or an expression producing one); the
    /// number of features must be the same at every push, the number of samples may vary.
    pub fn push(&self, batch: &Array<f32>) -> Result<()> {
        batch.check()?;
        let dims = batch.shape().dims();
        if dims[2] != 1 || dims[3] != 1 {
            return Err(invalid("a covariance batch is a [features, samples] matrix"));
        }
        let (features, samples) = (dims[0], dims[1]);
        let ctx = batch.context();
        check_wide::<W>(ctx)?;
        let mut guard = self.lock();
        if guard.state.is_none() {
            let first = batch.slice(1, ..1);
            let pivot0 = select(&first.is_finite(), &first, 0.0f32);
            pivot0.eval()?;
            guard.state = Some(CovState {
                ctx: ctx.clone(),
                features,
                n: V::zeros(ctx, Shape::from(1usize))?,
                mean: V::zeros(ctx, Shape::from([features, 1]))?,
                c: V::zeros(ctx, Shape::from([features, features]))?,
                pivot: V::from_f32(&pivot0),
                pivot0,
                epoch: None,
                pending: Vec::new(),
                merged_once: false,
            });
        }
        let state = guard.state.as_mut().expect("just made");
        if !ctx.same_as(&state.ctx) {
            return Err(LumaError::ContextMismatch);
        }
        if features != state.features {
            return Err(LumaError::ShapeMismatch { lhs: dims, rhs: [state.features, samples, 1, 1] });
        }
        if state.epoch.is_none() {
            let k = state.pivot.narrow();
            k.eval()?;
            state.epoch = Some((k.clone(), V::from_f32(&k)));
        }
        let k = &state.epoch.as_ref().expect("opened").0;
        let d = batch - k;
        // The product `d d^T`, as the transpose of `d` times `d` in full `f32`.
        let gram = matmul_f32(&d.transpose(), &d);
        let s = d.sum_axis(1);
        state.pending.push((samples as f64, V::from_f32(&s), V::from_f32(&gram)));
        let limit = if state.merged_once { self.flush_every } else { 1 };
        if state.pending.len() >= limit {
            state.merge_pending()?;
        }
        Ok(())
    }

    /// Merges the recorded pushes into the state.
    pub fn flush(&self) -> Result<()> {
        if let Some(state) = self.lock().state.as_mut() {
            state.merge_pending()?;
        }
        Ok(())
    }

    /// Clears the state.
    pub fn reset(&self) {
        self.lock().state = None;
    }

    fn settled(&self) -> Result<(V<W>, V<W>, V<W>)> {
        let mut guard = self.lock();
        let state = guard.state.as_mut().ok_or_else(|| invalid("the accumulator is empty: push a batch first"))?;
        state.merge_pending()?;
        Ok((state.n.clone(), state.mean.clone(), state.c.clone()))
    }

    /// The number of samples, a one-element array.
    pub fn count(&self) -> Result<Array<W>> {
        Ok(self.settled()?.0 .0)
    }

    /// The mean of every feature, `[features, 1]`.
    pub fn mean(&self) -> Result<Array<W>> {
        Ok(self.settled()?.1 .0)
    }

    /// The population covariance matrix (divisor `n`), `[features, features]`.
    pub fn cov(&self) -> Result<Array<W>> {
        self.cov_ddof(0)
    }

    /// The covariance matrix with divisor `n - ddof`.
    pub fn cov_ddof(&self, ddof: u32) -> Result<Array<W>> {
        let (n, _, c) = self.settled()?;
        Ok((&c / &(&n + (-f64::from(ddof)))).0)
    }

    /// The variance of every feature, `[features, 1]`.
    pub fn var(&self) -> Result<Array<W>> {
        let (n, _, c) = self.settled()?;
        let f = c.0.shape().dims()[0];
        let diag = c.0.take(&(Array::<u32>::range_on(c.0.context(), f, 0)? * (f as u32 + 1))).reshape([f, 1]);
        Ok((&V(diag) / &n).0)
    }

    /// The correlation matrix `cov[i][j] / sqrt(var[i] var[j])`; the diagonal is 1 for features
    /// with variance, NaN for constant ones.
    pub fn corr(&self) -> Result<Array<W>> {
        let (_, _, c) = self.settled()?;
        let f = c.0.shape().dims()[0];
        let diag = c.0.take(&(Array::<u32>::range_on(c.0.context(), f, 0)? * (f as u32 + 1)));
        let (col, row) = (V(diag.reshape([f, 1])), V(diag.reshape([1, f])));
        Ok((&c / &(&col * &row).sqrt()).0)
    }

    /// Folds `other`'s state in (same number of features; `other` may live on another device).
    pub fn merge(&self, other: &Covariance<W>) -> Result<()> {
        let theirs = {
            let mut guard = other.lock();
            let Some(state) = guard.state.as_mut() else { return Ok(()) };
            state.merge_pending()?;
            (state.ctx.clone(), state.features, state.n.clone(), state.mean.clone(), state.c.clone(), state.pivot0.clone())
        };
        let mut guard = self.lock();
        match guard.state.as_mut() {
            None => {
                let mut state = CovState {
                    ctx: theirs.0,
                    features: theirs.1,
                    n: theirs.2,
                    mean: theirs.3,
                    c: theirs.4,
                    pivot: V::from_f32(&theirs.5),
                    pivot0: theirs.5,
                    epoch: None,
                    pending: Vec::new(),
                    merged_once: true,
                };
                state.refresh_pivot()?;
                guard.state = Some(state);
                Ok(())
            }
            Some(state) => {
                state.merge_pending()?;
                if theirs.1 != state.features {
                    return Err(invalid("covariances of different feature counts cannot be merged"));
                }
                let ctx = state.ctx.clone();
                let (n_b, mean_b, c_b) = if ctx.same_as(&theirs.0) {
                    (theirs.2, theirs.3, theirs.4)
                } else {
                    let up = |v: &V<W>| -> Result<V<W>> { Ok(V(Array::from_slice_on(&ctx, &v.0.to_vec()?, v.0.shape())?)) };
                    (up(&theirs.2)?, up(&theirs.3)?, up(&theirs.4)?)
                };
                state.absorb(n_b, mean_b, c_b)
            }
        }
    }

    /// The state as bytes, see [`Accumulator::to_bytes`](super::Accumulator::to_bytes).
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        let mut w = Writer::new(KIND_COV);
        w.u64(self.flush_every as u64);
        let mut guard = self.lock();
        match guard.state.as_mut() {
            None => w.u8(0),
            Some(state) => {
                state.merge_pending()?;
                w.u8(1);
                w.u64(state.features as u64);
                w.f64s(&state.pivot0.to_vec()?.into_iter().map(f64::from).collect::<Vec<_>>());
                for v in [&state.n, &state.mean, &state.c] {
                    w.f64s(&v.0.to_vec()?.into_iter().map(W::to_f64).collect::<Vec<_>>());
                }
            }
        }
        Ok(w.0)
    }

    /// Reads a state written by [`to_bytes`](Self::to_bytes) onto the default context.
    pub fn from_bytes(bytes: &[u8]) -> Result<Covariance<W>> {
        Self::from_bytes_on(&crate::default_context()?, bytes)
    }

    /// Reads a state written by [`to_bytes`](Self::to_bytes) onto `ctx`.
    pub fn from_bytes_on(ctx: &Context, bytes: &[u8]) -> Result<Covariance<W>> {
        check_wide::<W>(ctx)?;
        let mut r = Reader::new(bytes, KIND_COV)?;
        let flush_every = (r.u64()? as usize).max(1);
        let acc = Covariance::with_flush_every(flush_every);
        if r.u8()? != 0 {
            let f = r.u64()? as usize;
            let p0: Vec<f32> = r.f64s(f)?.into_iter().map(|v| v as f32).collect();
            let up = |host: Vec<f64>, shape: Shape| -> Result<V<W>> {
                let host: Vec<W> = host.into_iter().map(W::from_f64).collect();
                Ok(V(Array::from_slice_on(ctx, &host, shape)?))
            };
            let n = up(r.f64s(1)?, Shape::from(1usize))?;
            let mean = up(r.f64s(f)?, Shape::from([f, 1]))?;
            let c = up(r.f64s(f * f)?, Shape::from([f, f]))?;
            r.finish()?;
            let pivot0 = Array::from_slice_on(ctx, &p0, Shape::from([f, 1]))?;
            let mut state = CovState { ctx: ctx.clone(), features: f, n, mean, c, pivot: V::from_f32(&pivot0), pivot0, epoch: None, pending: Vec::new(), merged_once: true };
            state.refresh_pivot()?;
            acc.lock().state = Some(state);
        } else {
            r.finish()?;
        }
        Ok(acc)
    }
}

impl<W: Wide> Default for Covariance<W> {
    fn default() -> Self {
        Covariance::new()
    }
}

impl<W: Wide> std::fmt::Debug for Covariance<W> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Covariance").field("wide", &W::NAME).finish()
    }
}

impl<W: Wide> CovState<W> {
    fn refresh_pivot(&mut self) -> Result<()> {
        let have = self.n.gt(&V::constant(&self.ctx, 0.0)?).broadcast_to(self.mean.0.shape());
        self.pivot = V::select(&have, &self.mean, &V::from_f32(&self.pivot0));
        self.pivot.0.eval()?;
        self.epoch = None;
        Ok(())
    }

    fn merge_pending(&mut self) -> Result<()> {
        if self.pending.is_empty() {
            return Ok(());
        }
        let ctx = self.ctx.clone();
        let pending = std::mem::take(&mut self.pending);
        let k = self.epoch.as_ref().expect("a push opened the epoch").1.clone();
        let n_b: f64 = pending.iter().map(|p| p.0).sum();
        let mut s: Option<V<W>> = None;
        let mut g: Option<V<W>> = None;
        for (_, si, gi) in &pending {
            s = Some(match s {
                Some(a) => &a + si,
                None => si.clone(),
            });
            g = Some(match g {
                Some(a) => &a + gi,
                None => gi.clone(),
            });
        }
        let (s, g) = (s.expect("a pending push"), g.expect("a pending push"));
        let m = &s * (1.0 / n_b);
        let mean_b = &k + &m;
        // The centered sums of products of the pushes: G - s s^T / n.
        let c_b = &g - &(&(&s * &V(s.0.reshape([1, s.0.len()]))) * (1.0 / n_b));
        let n_b = V::<W>::constant(&ctx, n_b)?;
        self.absorb(n_b, mean_b, c_b)
    }

    /// Chan's merge of another set of samples (count, mean, centered sums of products).
    fn absorb(&mut self, n_b: V<W>, mean_b: V<W>, c_b: V<W>) -> Result<()> {
        let ctx = self.ctx.clone();
        let n = &self.n + &n_b;
        let n_safe = n.maximum(&V::constant(&ctx, 1.0)?);
        let rb = &n_b / &n_safe;
        let delta = &mean_b - &self.mean;
        let w = &self.n * &rb;
        let f = self.features;
        let outer = &delta * &V(delta.0.reshape([1, f]));
        self.c = &(&self.c + &c_b) + &(&outer * &w);
        self.mean = &self.mean + &(&delta * &rb);
        self.n = n;
        eval_all(&[&self.n.0, &self.mean.0, &self.c.0])?;
        self.merged_once = true;
        self.refresh_pivot()
    }
}
