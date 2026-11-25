//! Refined solves: an `f32` factorisation, then iterative refinement whose residual is accumulated with a compensated dot product
//! (Dot2, Ogita-Rump-Oishi) and whose iterate is kept as a [`Df32`] pair.
//!
//! Reached only through [`Array::solve_refined`] and its siblings, never by a selection: a refined result differs from the plain one in
//! the low bits (that is its purpose), so `solve` and `solve_spd` do not change. The call synchronises (one readback per chunk of
//! iterations) and returns plain values, so a stall is visible to the caller without profiling.

use std::panic::Location;

use super::lu::{factor, solve_matrix, Lu};
use super::tri::rhs_matrix;
use super::{chol, fail, grid, launch, Dims};
use crate::array::Array;
use crate::context::Context;
use crate::error::LumaError;
use crate::gpu_type::Df32;
use crate::jit::structured::boundary::FactorKind;
use crate::jit::structured::families::factor::{boundary_of, refined_stage};
use crate::jit::structured::strategy::{first_candidate_in, note_in, StrategyId};
use crate::kernel::{uniform, view, KernelArg};
use crate::ops::select;
use crate::profile::{Decision, DecisionSite, Verdict};
use crate::shape::Shape;
use crate::view::concat_slice;

/// What [`Array::solve_refined`] and its siblings return.
#[derive(Debug)]
#[non_exhaustive]
pub struct Refined {
    /// The solution, an `f32` array of the shape of the right-hand side. Errors poison it, as everywhere; the two scalars are then
    /// `NaN` and `false`.
    pub x: Array<f32>,
    /// The relative error of the returned solution, estimated by the size of the last correction: `max ||dx||_inf / ||x||_inf` over
    /// the columns and matrices of the call. An estimate, not a guarantee (see [`Array::solve_refined`]).
    pub achieved_rel_err: f64,
    /// Every column of every matrix met the target. `false` after a stall (the factorisation is too poor for the matrix: its condition
    /// number times the rounding unit of the factorisation is about 1 or more), an iteration cap or a non-finite iterate.
    pub converged: bool,
}

/// The least relative error an `f32` result can store (half an ulp, `2^-24`).
const BEST: f64 = 1.0 / 16_777_216.0;
/// Corrections applied at most.
const MAX_ITERS: u32 = 6;
/// Iterations recorded before the first readback (a well-conditioned system converges in two: one sync).
const CHUNK: u32 = 2;
/// A correction that is not smaller than this fraction of the one before it is a stall.
const STALL: f64 = 0.5;

impl Array<f32> {
    /// Solves `A X = B` (`self` is a square `A`, or a batch of them) with iterative refinement: an `f32` LU factorisation, then
    /// corrections computed from a residual accumulated in compensated arithmetic (Dot2: the dot product as if in twice the working
    /// precision), the iterate kept as a [`Df32`] pair. For a matrix whose condition number is up to about `1e6` the result is the
    /// correctly rounded `f32` solution (the relative error of [`Array::solve`] is about the condition number times `2^-24`).
    ///
    /// `b` is `[k, n]` (`[k, n, batch]`) or a vector, as for [`Array::solve`]. Returns a [`Refined`]: the solution, the relative error
    /// the last correction estimates, and whether every column met the target `2^-24`.
    ///
    /// This is an explicit call: nothing makes [`Array::solve`] refine. It synchronises (the iteration reads the size of its
    /// corrections back), so it is not for a step that is recorded: inside `iterate`, a trace or a graph capture the result is
    /// poisoned with [`LumaError::Unsupported`].
    ///
    /// What it does not promise: `achieved_rel_err` is an estimate (reliable when the corrections contract by half or better and
    /// `A` times the rounding unit of the factorisation is well below 1), not `||x - x_true||`; the matrix is the `f32` matrix as
    /// stored, not the user's decimal data; at least one correction is applied, so the result differs from [`Array::solve`] in the
    /// low bits. A system that does not converge (condition number of `1e7` and up, or a singular one) returns the best iterate it
    /// saw, never worse than the plain solve, with `converged == false`; a singular matrix gives the `NaN`/`Inf` entries of
    /// [`Array::solve`].
    ///
    /// ```
    /// # use luma::*;
    /// # let ctx = Context::new()?;
    /// let a = Array::from_slice(&[4.0f32, 1.0, 1.0, 3.0], [2, 2])?;
    /// let b = Array::from_slice(&[1.0f32, 2.0], [1, 2])?;
    /// let solved = a.solve_refined(&b);
    /// assert!(solved.converged);
    /// let x = solved.x.to_vec()?;
    /// assert!((x[0] - 1.0 / 11.0).abs() < 1e-7 && (x[1] - 7.0 / 11.0).abs() < 1e-7);
    /// # Ok::<(), luma::LumaError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// `x` is poisoned if `A` is not square, `b` does not match it, or the call is made inside a recording.
    #[track_caller]
    pub fn solve_refined(&self, b: &Array<f32>) -> Refined {
        run(self, b, false, BEST, "solve_refined", Location::caller())
    }

    /// [`Array::solve_refined`] with the target `max(target_rel_err, 2^-24)` (an `f32` result cannot store more).
    ///
    /// # Errors
    ///
    /// As [`Array::solve_refined`]; `x` is also poisoned if `target_rel_err` is not finite or not in `(0, 1)`.
    #[track_caller]
    pub fn solve_refined_to(&self, b: &Array<f32>, target_rel_err: f64) -> Refined {
        run(self, b, false, target_rel_err, "solve_refined_to", Location::caller())
    }

    /// [`Array::solve_refined`] for a symmetric positive definite `A` (`self`, only its lower triangle is read), by Cholesky.
    ///
    /// # Errors
    ///
    /// As [`Array::solve_refined`].
    #[track_caller]
    pub fn solve_spd_refined(&self, b: &Array<f32>) -> Refined {
        run(self, b, true, BEST, "solve_spd_refined", Location::caller())
    }

    /// [`Array::solve_refined_to`] for a symmetric positive definite `A`.
    ///
    /// # Errors
    ///
    /// As [`Array::solve_refined_to`].
    #[track_caller]
    pub fn solve_spd_refined_to(&self, b: &Array<f32>, target_rel_err: f64) -> Refined {
        run(self, b, true, target_rel_err, "solve_spd_refined_to", Location::caller())
    }
}

/// The factorisation the corrections reuse.
enum Factors {
    Lu(Lu),
    Cholesky(Array<f32>),
}

impl Factors {
    /// `F⁻¹ v` for `[k, n, batch]` right-hand sides.
    fn solve(&self, v: &Array<f32>, op: &'static str, loc: &'static Location<'static>) -> Array<f32> {
        match self {
            Factors::Lu(f) => solve_matrix(f, v, op, loc),
            Factors::Cholesky(l) => chol::cholesky_solve(l, v, loc),
        }
    }
}

/// What became of one column of one matrix.
#[derive(Clone, Copy, PartialEq, Debug)]
enum Column {
    Active,
    /// Met the target at iteration `k`.
    Converged(u32),
    /// Stopped contracting (or hit the cap) at iteration `k`.
    Stalled(u32),
    NonFinite,
}

/// The poisoned outcome of a call that failed before it iterated.
fn failed(ctx: &Context, shape: Shape, error: LumaError, op: &'static str, loc: &'static Location<'static>) -> Refined {
    Refined { x: fail(ctx, shape, error, op, loc), achieved_rel_err: f64::NAN, converged: false }
}

fn run(a: &Array<f32>, b: &Array<f32>, spd: bool, target: f64, op: &'static str, loc: &'static Location<'static>) -> Refined {
    let ctx = a.context();
    if !target.is_finite() || target <= 0.0 || target >= 1.0 {
        return failed(ctx, b.shape(), LumaError::InvalidOption(format!("{op}: the target relative error {target} is not in (0, 1)")), op, loc);
    }
    let d = match Dims::square(a) {
        Ok(d) => d,
        Err(e) => return failed(ctx, b.shape(), e, op, loc),
    };
    if !ctx.same_as(b.context()) {
        return failed(ctx, b.shape(), LumaError::ContextMismatch, op, loc);
    }
    let (rhs, vector) = rhs_matrix(a, b);
    let [k, n, batch, last] = rhs.shape().dims();
    if n != d.rows || batch != d.batch || last != 1 {
        return failed(ctx, b.shape(), LumaError::ShapeMismatch { lhs: a.shape().dims(), rhs: b.shape().dims() }, op, loc);
    }
    if n > ctx.max_workgroups_per_dimension() as usize {
        return failed(ctx, b.shape(), LumaError::DimensionTooLarge { shape: a.shape().dims() }, op, loc);
    }
    let boundary = boundary_of(if spd { FactorKind::SolveSpdRefined } else { FactorKind::SolveRefined }, op, loc, d, k);
    let Some((id, params)) = first_candidate_in(refined_stage(), ctx, &boundary) else {
        let error = LumaError::Unsupported(format!("{op} reads back; call it outside a recording"));
        return failed(ctx, b.shape(), error, op, loc);
    };
    debug_assert!(id == StrategyId::LU_REFINED_F32 || id == StrategyId::CHOL_REFINED_F32);
    note_in(ctx, &boundary, refined_stage(), id, &params, &[]);
    match iterate(ctx, a, &rhs, spd, target, Dims { rows: n, cols: k, batch }, op, loc) {
        Ok(done) => {
            if ctx.jit().recording_decisions() {
                if done.stalled {
                    ctx.jit().note_decision(Decision::new(op, loc, DecisionSite::FactorPlan, "factor::refine_stalled").candidate(id.name(), Verdict::Rejected("kappa*u >= 1")));
                }
                let label = format!("factor::refine_result[iters={} converged={} achieved_rel_err={:.3e}]", done.iters, done.converged, done.achieved_rel_err);
                ctx.jit().note_decision(Decision::new(op, loc, DecisionSite::FactorPlan, label));
            }
            let x = if vector { done.x.reshape(b.shape()) } else { done.x };
            Refined { x, achieved_rel_err: done.achieved_rel_err, converged: done.converged }
        }
        Err(e) => failed(ctx, b.shape(), e, op, loc),
    }
}

struct Done {
    x: Array<f32>,
    achieved_rel_err: f64,
    converged: bool,
    stalled: bool,
    iters: u32,
}

/// The infinity norms of `v` along its rows, one per column of every matrix (`[k, 1, batch]`); a non-finite entry makes the norm `Inf`.
fn norms(v: &Array<f32>) -> Array<f32> {
    select(&v.is_finite(), v.abs(), f32::INFINITY).max_axis(1)
}

/// The iteration of the module doc, for `[k, n, batch]` right-hand sides `rhs` (`dims` is `n`, `k` and the batch: `cols` is the number
/// of right-hand-side columns here).
#[allow(clippy::too_many_arguments)]
fn iterate(ctx: &Context, a: &Array<f32>, rhs: &Array<f32>, spd: bool, target: f64, dims: Dims, op: &'static str, loc: &'static Location<'static>) -> Result<Done, LumaError> {
    let (k, batch) = (dims.cols, dims.batch);
    let factors = if spd { Factors::Cholesky(chol::lower(a, loc)) } else { Factors::Lu(factor(a, op, loc)) };
    let columns = k * batch;
    // Iterate 0 is the plain solve; `iterates[j]` is `x_j`.
    let mut iterates: Vec<Array<Df32>> = vec![factors.solve(rhs, op, loc).to_df32().cache()];
    // The statistics of iteration `j >= 1` (`dx_j`, `||x_j||`, `||r_j||` of the residual of `x_{j-1}`), one per column, read back in chunks.
    let mut dx: Vec<Vec<f32>> = vec![Vec::new()];
    let mut xs: Vec<Vec<f32>> = vec![Vec::new()];
    let mut rs: Vec<Vec<f32>> = vec![Vec::new()];
    let mut pending: Vec<Array<f32>> = Vec::new();
    let mut state = vec![Column::Active; columns];
    let mut iters = 0u32;
    while iters < MAX_ITERS {
        let x = iterates.last().expect("iterate 0 exists");
        let r = residual(ctx, a, x, rhs, spd, dims)?;
        let correction = factors.solve(&r, op, loc);
        let next = (x + &correction.to_df32()).cache();
        let rounded = next.to_f32();
        pending.push(crate::view::concat(1, [&norms(&correction), &norms(&rounded), &norms(&r)]));
        iterates.push(next);
        iters += 1;
        if iters < CHUNK {
            continue;
        }
        // One sync: the statistics of every iteration since the last readback.
        let rows = 3 * pending.len();
        let flat = concat_slice(1, &pending).to_vec()?;
        let first = iters as usize - pending.len() + 1;
        for (at, _) in pending.iter().enumerate() {
            let (mut a_dx, mut a_x, mut a_r) = (vec![0.0f32; columns], vec![0.0f32; columns], vec![0.0f32; columns]);
            for mat in 0..batch {
                for col in 0..k {
                    let base = col + k * (3 * at + rows * mat);
                    let c = col + k * mat;
                    (a_dx[c], a_x[c], a_r[c]) = (flat[base], flat[base + k], flat[base + 2 * k]);
                }
            }
            debug_assert_eq!(dx.len(), first + at);
            dx.push(a_dx);
            xs.push(a_x);
            rs.push(a_r);
        }
        pending.clear();
        judge(&mut state, &dx, &xs, first as u32, iters, target);
        if state.iter().all(|s| *s != Column::Active) {
            break;
        }
    }
    // The returned iterate and its estimate, per column.
    let mut choice = vec![0usize; columns];
    let (mut worst, mut converged, mut stalled) = (0.0f64, true, false);
    for (c, column) in state.iter().enumerate() {
        let estimate = |j: usize| {
            let (d, x) = (f64::from(dx[j][c]), f64::from(xs[j][c]));
            if x == 0.0 { if d == 0.0 { 0.0 } else { f64::INFINITY } } else { d / x }
        };
        match *column {
            Column::Converged(at) => {
                choice[c] = at as usize;
                worst = worst.max(estimate(at as usize));
            }
            Column::Stalled(at) => {
                // The smallest residual seen: `rs[j + 1]` is that of `x_j`, which the iterate `at` has no residual for yet.
                let best = (0..at as usize).min_by(|&p, &q| rs[p + 1][c].total_cmp(&rs[q + 1][c])).unwrap_or(0);
                choice[c] = best;
                worst = worst.max(estimate(best + 1));
                converged = false;
                stalled = true;
            }
            Column::NonFinite => {
                worst = f64::NAN;
                converged = false;
            }
            Column::Active => unreachable!("the loop ends when no column is active or at the cap, which stalls them"),
        }
    }
    let x = assemble(ctx, &iterates, &choice, dims)?;
    Ok(Done { x, achieved_rel_err: worst, converged, stalled, iters })
}

/// Decides every active column from the iterations `first..=last` it has statistics for.
fn judge(state: &mut [Column], dx: &[Vec<f32>], xs: &[Vec<f32>], first: u32, last: u32, target: f64) {
    for (c, column) in state.iter_mut().enumerate() {
        for k in first..=last {
            if *column != Column::Active {
                break;
            }
            let (d, x) = (f64::from(dx[k as usize][c]), f64::from(xs[k as usize][c]));
            if !d.is_finite() || !x.is_finite() {
                *column = Column::NonFinite;
            } else if d <= target * x && (k == 1 || d <= STALL * f64::from(dx[k as usize - 1][c])) {
                *column = Column::Converged(k);
            } else if (k >= 2 && d > STALL * f64::from(dx[k as usize - 1][c])) || k == MAX_ITERS {
                *column = Column::Stalled(k);
            }
        }
    }
}

/// The solution that takes iterate `choice[c]` for column `c` (a column of a matrix, `c = col + k * matrix`), rounded to `f32`.
fn assemble(ctx: &Context, iterates: &[Array<Df32>], choice: &[usize], dims: Dims) -> Result<Array<f32>, LumaError> {
    let first = choice[0];
    if choice.iter().all(|&j| j == first) {
        return Ok(iterates[first].to_f32());
    }
    let (n, k, batch) = (dims.rows, dims.cols, dims.batch);
    let picks: Vec<u32> = choice.iter().map(|&j| j as u32).collect();
    let which = Array::<u32>::from_slice_on(ctx, &picks, Shape::from([k, 1, batch]))?.broadcast_to(Shape::from([k, n, batch]));
    let mut distinct: Vec<usize> = choice.to_vec();
    distinct.sort_unstable();
    distinct.dedup();
    let mut x = iterates[distinct[0]].to_f32();
    for &j in &distinct[1..] {
        x = select(&which.eq(j as u32), iterates[j].to_f32(), &x);
    }
    Ok(x)
}

// -- The residual --

/// Columns of the right-hand sides a workgroup of [`residual`] takes: it reads a row of `A` once for all of them.
const KC: usize = 8;
/// Invocations of a workgroup of [`residual`]; a power of two (the reduction is a fixed tree over them).
const THREADS: usize = 256;

#[derive(encase::ShaderType)]
struct ResidualParams {
    n: u32,
    k: u32,
    batch: u32,
    /// A runtime zero: the `luma_o` of the double-float library `or`s it in, which keeps the compiler from reassociating the
    /// compensation away.
    guard: u32,
    /// Groups of `KC` columns per matrix.
    ncg: u32,
}

/// The WGSL of [`residual`]: `r = b - A (x_hi + x_lo)` with Dot2. A workgroup is a row `i` of `A` and a group of up to `KC` columns of
/// one matrix; invocation `t` strides over `j = t, t + THREADS, ...` along the row and keeps a Dot2 pair `(s, c)` per column. The
/// invocations' pairs are combined by a fixed tree (`(s, q) = two_sum(s0, s1)`, `c = c0 + c1 + q`), so the result is the same on every
/// run of a device. `sym` reads `A(max(i, j), min(i, j))`: the lower triangle stands for a symmetric matrix.
///
/// Only `df_two_sum` and `df_two_prod` are used, never `df_add`, which `fast_math` makes sloppy: a sloppy sum would remove the
/// compensation.
fn residual_source(sym: bool, fma: bool) -> String {
    let element = if sym { "a[luma_view_idx3(av, min(i, j), max(i, j), m)]" } else { "a[luma_view_idx3(av, j, i, m)]" };
    let library = crate::builtin::df_library("p.guard", fma, false);
    format!(
        "struct Params {{ n: u32, k: u32, batch: u32, guard: u32, ncg: u32 }}
@group(1) @binding(0) var<storage, read> a: array<f32>;
@group(1) @binding(1) var<uniform> av: LumaView;
@group(1) @binding(2) var<storage, read> x: array<vec2<f32>>;
@group(1) @binding(3) var<storage, read> b: array<f32>;
@group(1) @binding(4) var<uniform> bv: LumaView;
@group(1) @binding(5) var<storage, read_write> r: array<f32>;
@group(1) @binding(6) var<uniform> p: Params;
{library}
var<workgroup> ws: array<f32, {size}>;
var<workgroup> wc: array<f32, {size}>;

@compute @workgroup_size({THREADS})
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_index) t: u32) {{
    let i = wid.x;
    let item = wid.y + wid.z * nwg.y;
    let m = item / p.ncg;
    if (m >= p.batch) {{ return; }}
    let col0 = (item % p.ncg) * {KC}u;
    var s: array<f32, {KC}>;
    var c: array<f32, {KC}>;
    for (var q = 0u; q < {KC}u; q = q + 1u) {{
        let col = col0 + q;
        s[q] = select(0.0, b[luma_view_idx3(bv, col, i, m)], (t == 0u) && (col < p.k));
        c[q] = 0.0;
    }}
    for (var j = t; j < p.n; j = j + {THREADS}u) {{
        let aij = -{element};
        for (var q = 0u; q < {KC}u; q = q + 1u) {{
            if (col0 + q < p.k) {{
                let xv = x[(m * p.n + j) * p.k + col0 + q];
                let pr = df_two_prod(aij, xv.x);
                let sm = df_two_sum(s[q], pr.x);
                s[q] = sm.x;
                c[q] = luma_o(c[q] + luma_o(luma_o(sm.y + pr.y) + luma_o(aij * xv.y)));
            }}
        }}
    }}
    for (var q = 0u; q < {KC}u; q = q + 1u) {{
        ws[t * {KC}u + q] = s[q];
        wc[t * {KC}u + q] = c[q];
    }}
    workgroupBarrier();
    for (var h = {half}u; h > 0u; h = h >> 1u) {{
        if (t < h) {{
            for (var q = 0u; q < {KC}u; q = q + 1u) {{
                let u = df_two_sum(ws[t * {KC}u + q], ws[(t + h) * {KC}u + q]);
                ws[t * {KC}u + q] = u.x;
                wc[t * {KC}u + q] = luma_o(luma_o(wc[t * {KC}u + q] + wc[(t + h) * {KC}u + q]) + u.y);
            }}
        }}
        workgroupBarrier();
    }}
    if (t == 0u) {{
        for (var q = 0u; q < {KC}u; q = q + 1u) {{
            if (col0 + q < p.k) {{ r[(m * p.n + i) * p.k + col0 + q] = luma_o(ws[q] + wc[q]); }}
        }}
    }}
}}
",
        size = THREADS * KC,
        half = THREADS / 2,
    )
}

/// `r = b - A (x_hi + x_lo)` of `[k, n, batch]` right-hand sides, accumulated in Dot2 and rounded once to `f32`; `sym` reads only the
/// lower triangle of `a`. One launch: `n` rows of `A` times groups of `KC` columns times the batch of workgroups.
fn residual(ctx: &Context, a: &Array<f32>, x: &Array<Df32>, b: &Array<f32>, sym: bool, dims: Dims) -> Result<Array<f32>, LumaError> {
    let (n, k, batch) = (dims.rows, dims.cols, dims.batch);
    let mut r = Array::<f32>::alloc(ctx, Shape::from([k, n, batch]))?;
    let ncg = k.div_ceil(KC);
    let params = ResidualParams { n: n as u32, k: k as u32, batch: batch as u32, guard: 0, ncg: ncg as u32 };
    let fma = ctx.tuning().df_fma;
    let key = format!("refine_residual/{}/{}", u8::from(sym), u8::from(fma));
    let args: [KernelArg<'_>; 5] = [view(a), x.into(), view(b), (&mut r).into(), uniform(&params)];
    launch(ctx, &key, "refine_residual", || residual_source(sym, fma), grid(ctx, n, ncg * batch), &args)?;
    Ok(r)
}

/// The residual of the call, for the tests of this module (`b - A x`, Dot2) on evaluated arrays.
#[cfg(test)]
pub(crate) fn residual_of(a: &Array<f32>, x: &Array<Df32>, b: &Array<f32>, sym: bool) -> Result<Array<f32>, LumaError> {
    let d = Dims::of(a)?;
    let [k, ..] = b.shape().dims();
    residual(a.context(), a, x, b, sym, Dims { rows: d.rows, cols: k, batch: d.batch })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::ContextOptions;
    use crate::BackendKind;

    /// Deterministic values in `[-1, 1)`.
    struct Lcg(u64);

    impl Lcg {
        fn next(&mut self) -> f64 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (self.0 >> 11) as f64 / (1u64 << 52) as f64 - 1.0
        }
    }

    fn cpu(fma: bool) -> Context {
        Context::with_options(ContextOptions::default().backend(BackendKind::Cpu).calibrate(false).df_fma(fma)).unwrap()
    }

    /// The residual kernel in each of its forms (fused or Dekker products, the whole matrix or its lower triangle, 1 to 33 columns):
    /// against the `f64` residual of the same operands it is within the Dot2 bound (`u |r| + (n u)^2 sum |a||x|`), and far closer to it
    /// than the plain `f32` residual `b - A x_hi`, which loses the whole of `r` to rounding where `b` is the correctly rounded `A x`.
    #[test]
    fn the_residual_beats_the_f32_residual() {
        let n = 64usize;
        let u = 1.0 / 16_777_216.0f64;
        for fma in [true, false] {
            let ctx = cpu(fma);
            for sym in [false, true] {
                for k in [1usize, 8, 9, 33] {
                    let mut rng = Lcg(0x5eed + (k as u64) * 31 + u64::from(sym));
                    let mut a: Vec<f32> = (0..n * n).map(|_| rng.next() as f32).collect();
                    let junk = a.clone();
                    if sym {
                        // The strict upper triangle must not be read: make the matrix symmetric from the lower one, then spoil the upper.
                        for i in 0..n {
                            for j in 0..i {
                                a[j * n + i] = a[i * n + j];
                            }
                        }
                    }
                    let given: Vec<f32> = if sym { (0..n * n).map(|e| if e % n > e / n { junk[e] * 7.0 } else { a[e] }).collect() } else { a.clone() };
                    let x_true: Vec<f64> = (0..n * k).map(|_| rng.next()).collect();
                    let x_df: Vec<Df32> = x_true.iter().map(|&v| Df32::from(v)).collect();
                    // b = f32(A x_true), column c of the right-hand sides is x[.. * k + c].
                    let mut b = vec![0.0f32; n * k];
                    for i in 0..n {
                        for c in 0..k {
                            b[i * k + c] = (0..n).map(|j| f64::from(a[i * n + j]) * x_true[j * k + c]).sum::<f64>() as f32;
                        }
                    }
                    let (a_arr, b_arr) = (Array::from_slice_on(&ctx, &given, [n, n]).unwrap(), Array::from_slice_on(&ctx, &b, [k, n]).unwrap());
                    let x_arr = Array::<Df32>::from_slice_on(&ctx, &x_df, [k, n]).unwrap();
                    let r = residual_of(&a_arr, &x_arr, &b_arr, sym).unwrap().to_vec().unwrap();
                    let (mut worst_kernel, mut worst_plain, mut worst_r) = (0.0f64, 0.0f64, 0.0f64);
                    for i in 0..n {
                        for c in 0..k {
                            let exact: f64 = f64::from(b[i * k + c]) - (0..n).map(|j| f64::from(a[i * n + j]) * x_df[j * k + c].to_f64()).sum::<f64>();
                            let scale: f64 = (0..n).map(|j| f64::from(a[i * n + j]).abs() * x_df[j * k + c].to_f64().abs()).sum();
                            let plain = b[i * k + c] - (0..n).map(|j| a[i * n + j] * x_df[j * k + c].hi).sum::<f32>();
                            let kernel_err = (f64::from(r[i * k + c]) - exact).abs();
                            let bound = u * exact.abs() + 1.1 * (n as f64 * u).powi(2) * scale;
                            assert!(kernel_err <= bound, "fma={fma} sym={sym} k={k} ({i}, {c}): error {kernel_err:e} above the Dot2 bound {bound:e}");
                            worst_kernel = worst_kernel.max(kernel_err);
                            worst_plain = worst_plain.max((f64::from(plain) - exact).abs());
                            worst_r = worst_r.max(exact.abs());
                        }
                    }
                    assert!(worst_plain > 0.1 * worst_r, "fma={fma} sym={sym} k={k}: the f32 residual ({worst_plain:e}) is not lost in rounding ({worst_r:e}): the case is too easy");
                    assert!(worst_kernel < 0.01 * worst_plain, "fma={fma} sym={sym} k={k}: kernel {worst_kernel:e}, plain {worst_plain:e}");
                }
            }
        }
    }
}
