//! Property-aware routing (W3-R6): what a matrix known to be diagonal, triangular or banded is solved with, and the explicit
//! banded solve.
//!
//! The automatic routes are rules of class `Reformulating` (they run under [`Numerics::reformulate`](crate::Numerics), which
//! `algebraic()` and `fast()` set): a forward substitution does not pivot by magnitude like the LU of `solve` does, and the
//! product of a diagonal is not `exp(sum(ln))`. They fire only for a matrix whose handle carries the property (see
//! [`crate::props`]); anything in doubt, a shape the routes do not cover, runs the dense path. [`Array::solve_banded`] is the one
//! opt-in for a band: its line solvers do not pivot across rows, so `solve` never routes a band by itself.

use std::panic::Location;
use std::sync::{Arc, Mutex, Weak};

use crate::jit::Node;

use super::tri::rhs_matrix;
use super::{fail, Dims};
use crate::array::Array;
use crate::context::Context;
use crate::error::LumaError;
use crate::jit::comprehension::Props;
use crate::profile::{Decision, DecisionSite, Verdict};
use crate::shape::Shape;
use crate::ops::select;
use crate::view::PadMode;
use crate::Triangle;

/// Records which route a call took, when decisions are being recorded.
fn note_route(ctx: &Context, op: &'static str, location: &'static Location<'static>, chosen: &'static str) {
    if ctx.jit().recording_decisions() {
        ctx.jit().note_decision(Decision::new(op, location, DecisionSite::Rewrite, chosen));
    }
}

/// The diagonals each matrix made by [`Array::from_diagonals`] was given, by the expression that is the matrix (weakly: an
/// entry dies with its matrix). `solve_banded` reads them back instead of recomputing them out of the expression for every column.
static BANDS: Mutex<Vec<(Weak<Node>, Diagonals)>> = Mutex::new(Vec::new());

/// `(offset, values)` pairs.
type Diagonals = Vec<(isize, Array<f32>)>;

impl Array<f32> {
    /// The `(offset, values)` pairs `self` was made from by [`Array::from_diagonals`], while it is that pending expression.
    fn remembered_diagonals(&self) -> Option<Diagonals> {
        let node = self.lazy_node()?;
        let bands = BANDS.lock().ok()?;
        bands.iter().find(|(held, _)| std::ptr::eq(held.as_ptr(), Arc::as_ptr(node))).map(|(_, vectors)| vectors.clone())
    }

    /// The matrix with the given diagonals: `diagonals` lists `(offset, values)` pairs, the values of the diagonal `offset` (above
    /// the main one for a positive offset, in the sense of [`Array::diag_matrix`]) as a column of `n - |offset|` elements, `n`
    /// being the same for every pair. Everything else is zero. Equal offsets add up.
    ///
    /// The matrix is lazy and knows its band, so [`Array::solve_banded`] and the banded product take it as one.
    ///
    /// # Errors
    ///
    /// The result is poisoned if there are no diagonals, the vectors are not columns, or the sizes `len + |offset|` differ.
    #[track_caller]
    pub fn from_diagonals(diagonals: &[(isize, &Array<f32>)]) -> Array<f32> {
        let location = Location::caller();
        let Some(&(_, first)) = diagonals.first() else {
            let ctx = crate::default_context().expect("a default context");
            let error = LumaError::InvalidShape("from_diagonals: no diagonals".into());
            return fail(&ctx, Shape::from([1, 1]), error, "from_diagonals", location);
        };
        let ctx = first.context();
        let sizes: Vec<usize> = diagonals.iter().map(|&(k, v)| v.shape().dims()[0].saturating_add(k.unsigned_abs())).collect();
        let n = sizes[0];
        let shape = Shape::from([n, n]);
        if sizes.iter().any(|&size| size != n) || diagonals.iter().any(|&(_, v)| v.shape().dims()[1..].iter().any(|&d| d != 1)) {
            let error = LumaError::InvalidShape(format!("from_diagonals: every diagonal must be a column of n - |offset| elements for one n (sizes {sizes:?})"));
            return fail(ctx, shape, error, "from_diagonals", location);
        }
        let mut sum: Option<Array<f32>> = None;
        for &(offset, values) in diagonals {
            let term = values.diag_matrix_at(offset, location);
            sum = Some(match sum {
                None => term,
                Some(acc) => acc.binary_at(crate::ops::Rhs::Array(&term), crate::builtin::BinOp::Add, false, "from_diagonals", location),
            });
        }
        let sum = sum.expect("at least one diagonal");
        if let (Some(node), Ok(mut bands)) = (sum.lazy_node(), BANDS.lock()) {
            bands.retain(|(held, _)| held.strong_count() > 0);
            // A vector in memory may be written through its handle later, which the matrix (an expression) is protected from; the
            // copy kept here is not written by anyone. An expression is a value already.
            let copy = |v: &Array<f32>| if v.lazy_node().is_some() { v.clone() } else { (v * 1.0f32).cache() };
            bands.push((Arc::downgrade(node), diagonals.iter().map(|&(k, v)| (k, copy(v))).collect()));
        }
        sum
    }

    /// Solves `A X = B` for a matrix that carries a band (one made by [`Array::from_diagonals`], [`Array::diag_matrix`] and the
    /// sums and products that keep it), without pivoting across rows: `A` must be diagonally dominant or positive definite for
    /// the result to be accurate (the dense [`Array::solve`] pivots and is the safe choice for the rest). A tridiagonal (or
    /// bidiagonal) `A` of a single matrix runs the tridiagonal line solver (`O(n)` per right-hand side column), fused with the
    /// expressions that make the diagonals; a wider band, and a batch, run the dense solve.
    ///
    /// Needs no numerics policy: calling it is the consent.
    ///
    /// # Errors
    ///
    /// The result is poisoned if `A` is not square, carries no band (a matrix read from memory or made by anything the property
    /// rules do not list has none), or `b` does not match it.
    #[track_caller]
    pub fn solve_banded(&self, b: &Array<f32>) -> Array<f32> {
        let location = Location::caller();
        let ctx = self.context();
        let d = match Dims::square(self) {
            Ok(d) => d,
            Err(e) => return fail(ctx, b.shape(), e, "solve_banded", location),
        };
        let Some((lo, hi)) = self.props_used().band else {
            let error = LumaError::InvalidShape("solve_banded: the matrix carries no band (build it with from_diagonals or diag_matrix)".into());
            return fail(ctx, b.shape(), error, "solve_banded", location);
        };
        let (rhs, vector) = rhs_matrix(self, b);
        let [k, rows, rb2, rb3] = rhs.shape().dims();
        let fits_line = d.batch == 1 && d.rows >= 2 && rows == d.rows && rb2 == 1 && rb3 == 1;
        let width = (lo.max(hi) as usize).max(1);
        let wide = (lo + hi + 1) as usize;
        let reason = if !fits_line {
            Some("batch or shape the line solver does not take")
        } else if width > 8 {
            Some("band wider than 8")
        } else if width > 1 && d.rows.saturating_mul(wide * wide) >= d.rows.saturating_mul(d.rows).saturating_mul(d.rows) / 3 {
            Some("band too wide for its size (dense is no slower)")
        } else {
            None
        };
        if let Some(reason) = reason {
            if ctx.jit().recording_decisions() {
                ctx.jit().note_decision(Decision::new("solve_banded", location, DecisionSite::Rewrite, "dense").candidate("block-tridiagonal", Verdict::Rejected(reason)));
            }
            return self.solve(b);
        }
        if width > 1 {
            note_route(ctx, "solve_banded", location, "block-tridiagonal");
            let x = self.solve_block_banded(&rhs, (lo as usize, hi as usize), width, location);
            return if vector { x.reshape(b.shape()) } else { x };
        }
        note_route(ctx, "solve_banded", location, "tridiagonal");
        let n = d.rows;
        let shape = Shape::from([k, n]);
        // Row `r` of the system: `lo[r] x[r-1] + diag[r] x[r] + up[r] x[r+1]`. In this crate's orientation the entry below the
        // diagonal of row `r` is `diag(1)[r - 1]` and the one above it `diag(-1)[r]` (see `crate::props`).
        let (lower, diag, upper) = match self.remembered_diagonals() {
            // Made by `from_diagonals`: the diagonals are the vectors it was given, read as the line solver reads its own.
            Some(vectors) => {
                let find = |offset: isize, len: usize| vectors.iter().filter(|(k, _)| *k == offset).map(|(_, v)| v.clone()).reduce(|a, b| &a + &b).unwrap_or_else(|| Array::<f32>::full_on(ctx, [len, 1], 0.0).expect("a small array"));
                let column = |v: Array<f32>| v.reshape([1, n]).broadcast_to(shape);
                (column(find(1, n - 1).pad_at([(1, 0)], PadMode::Zero, location)), column(find(0, n)), column(find(-1, n - 1).pad_at([(0, 1)], PadMode::Zero, location)))
            }
            // Any other banded matrix: the three diagonals are made once, in a pass over `3n` elements, and the solver reads them from
            // memory; left as expressions of the matrix (a sum of selects) each would be recomputed for every right-hand side column.
            None => {
                let storage = self.band_storage((1, 1), n, location).cache();
                let column = |w: usize| storage.slice(1, w..w + 1).reshape([1, n]).broadcast_to(shape);
                (column(0), column(1), column(2))
            }
        };
        let x = crate::linalg::solve_tridiagonal(1, &lower, &diag, &upper, &rhs);
        if vector { x.reshape(b.shape()) } else { x }
    }

    /// The band of `self` in diagonal storage: `[n', W]` (`W = lo + hi + 1`), entry `(r, w)` the element `(c = r + w - lo, r)` of the
    /// matrix, zero outside it, the matrix padded to `n'` rows with ones on the diagonal.
    fn band_storage(&self, (lo, hi): (usize, usize), padded: usize, location: &'static Location<'static>) -> Array<f32> {
        let n = self.shape().dims()[1];
        let extra = padded - n;
        let mut parts = Vec::with_capacity(lo + hi + 1);
        for s in -(lo as isize)..=(hi as isize) {
            // The entries `(r + s, r)` of every row `r`: `diag(-s)` is indexed by the row for `s >= 0`, `diag(|s|)` by the column.
            let mut v = if s >= 0 {
                self.diag_at(-s, location).pad_at([(0, s as usize + extra)], PadMode::Zero, location)
            } else {
                self.diag_at(-s, location).pad_at([(s.unsigned_abs(), extra)], PadMode::Zero, location)
            };
            if s == 0 && extra > 0 {
                let ones = Array::<f32>::full_on(self.context(), [extra, 1], 1.0).expect("a small array");
                v = &v + &ones.pad_at([(n, 0)], PadMode::Zero, location);
            }
            parts.push(v);
        }
        crate::concat_slice(1, &parts)
    }

    /// `solve_banded` for a band of width `b` in `2..=8`: the system as a block-tridiagonal one of `b x b` blocks (the matrix padded
    /// with identity rows to a multiple of `b`), the blocks read from the band storage by a gather. `rhs` is `[k, n]`.
    fn solve_block_banded(&self, rhs: &Array<f32>, band: (usize, usize), b: usize, location: &'static Location<'static>) -> Array<f32> {
        let ctx = self.context();
        let [k, n, ..] = rhs.shape().dims();
        let groups = n.div_ceil(b);
        let padded = groups * b;
        let wide = band.0 + band.1 + 1;
        let storage = self.band_storage(band, padded, location);
        let shape = Shape::from([b * b, 1, groups, 1]);
        let (Ok(pq), Ok(g)) = (Array::<u32>::range_on(ctx, shape, 0), Array::<u32>::range_on(ctx, shape, 2)) else {
            return fail(ctx, rhs.shape(), LumaError::InvalidShape("solve_banded: grid too large".into()), "solve_banded", location);
        };
        let (p, q) = (&pq / b as u32, &pq % b as u32);
        let row = &(&g * b as u32) + &p;
        let blocks = [-1isize, 0, 1].map(|d| {
            // Entry `(p, q)` of the block `d` columns of blocks from the diagonal one: the matrix element `(g b + q + d b, g b + p)`,
            // on the diagonal `s = d b + q - p`, stored at `w = s + lo` (offset by `2b` to stay unsigned).
            let top = (2 * b) as u32;
            let w2 = &(&q + (d * b as isize + band.0 as isize + 2 * b as isize) as u32) - &p;
            let inside = select(&w2.ge(top), select(&w2.lt(top + wide as u32), 1.0f32, 0.0f32), 0.0f32);
            let clamped = select(&w2.lt(top), top, select(&w2.ge(top + wide as u32), top, w2.clone()));
            let index = &row + &(&(&clamped - top) * padded as u32);
            let taken = storage.take(&index);
            (&taken * &inside).broadcast_to([b * b, k, groups, 1])
        });
        // Unknowns `(p, g)` of the right-hand side column `c`: `[k, n]` is `[k, b, groups]` and the solver wants `[b, k, groups]`.
        let padded_rhs = if padded > n { rhs.pad_at([(0, 0), (0, padded - n)], PadMode::Zero, location) } else { rhs.clone() };
        let grid = padded_rhs.reshape([k, b, groups]).transpose().reshape([b, k, groups, 1]);
        let x = crate::linalg::solve_block_tridiagonal(b, 2, &blocks[0], &blocks[1], &blocks[2], &grid);
        let x = x.reshape([b, k, groups]).transpose().reshape([k, padded]);
        if padded > n { x.slice(1, 0..n) } else { x }
    }

    /// What `solve` does for a matrix whose handle says it is diagonal or triangular, under `Numerics::reformulate`: `None` for
    /// every other call (the dense LU path then runs).
    pub(crate) fn route_solve(&self, b: &Array<f32>, location: &'static Location<'static>) -> Option<Array<f32>> {
        let props = self.props_used();
        let ctx = self.context();
        if Dims::square(self).is_err() || !ctx.same_as(b.context()) {
            return None;
        }
        let [n, _, a2, a3] = self.shape().dims();
        if props.has(Props::PERMUTATION) && ctx.numerics().exact_finite && a2 == 1 && a3 == 1 {
            // `P x = b` is `x = Pᵀ b`: row `j` of `x` is row `i` of `b` where `P[j, i]` is the one.
            let (rhs, vector) = rhs_matrix(self, b);
            if rhs.shape().dims()[1..] == [n, 1, 1] {
                note_route(ctx, "solve", location, "permutation");
                let inverse = self.argmax_axis(1).reshape([n, 1, 1]);
                let x = super::gather_rows(&rhs.reshape([rhs.shape().dims()[0], n, 1]), &inverse, location);
                return Some(if vector { x.reshape(b.shape()) } else { x.reshape(rhs.shape()) });
            }
        }
        if props.has(Props::DIAGONAL) {
            // `exact_finite`: LU of a diagonal matrix does not pivot and has zero multipliers, so for finite inputs it is `b / d`.
            if !ctx.numerics().exact_finite {
                return None;
            }
            // `x[c, r] = b[c, r] / d[r]`: a division that fuses into whatever reads it.
            let (rhs, vector) = rhs_matrix(self, b);
            if rhs.shape().dims()[1..] != [n, a2, a3] {
                return None;
            }
            note_route(ctx, "solve", location, "diagonal");
            let d = self.diag_at(0, location).reshape([1, n, a2, a3]).broadcast_to(rhs.shape());
            let x = &rhs / &d;
            return Some(if vector { x.reshape(b.shape()) } else { x });
        }
        if !ctx.numerics().reformulate {
            return None;
        }
        let triangle = if props.has(Props::LOWER) {
            Triangle::Lower
        } else if props.has(Props::UPPER) {
            Triangle::Upper
        } else {
            return None;
        };
        note_route(ctx, "solve", location, "triangular");
        Some(self.solve_triangular(b, triangle, false, props.has(Props::UNIT_DIAG)))
    }

    /// `det` of a matrix known to be triangular or diagonal, under `Numerics::reformulate`: the product of its diagonal.
    pub(crate) fn route_det(&self, location: &'static Location<'static>) -> Option<Array<f32>> {
        let props = self.props_used();
        let ctx = self.context();
        if !ctx.numerics().reformulate || Dims::square(self).is_err() || !(props.has(Props::LOWER) || props.has(Props::UPPER)) {
            return None;
        }
        note_route(ctx, "det", location, "diagonal-product");
        Some(self.diag_at(0, location).prod_axis(0))
    }
}

/// Terms of the fused sum of shifts, whatever the cost model says: the expression has to stay one kernel (33 diagonals run as one
/// on the GPUs, measured).
const MAX_SHIFT_TERMS: usize = 33;

/// Rows per diagonal at which the fused sum of shifts costs what the GEMM does: a term of the sum is an elementwise multiply-add
/// with the diagonal's index arithmetic (about 60 times the time of a multiply-add of the GEMM on the CPU backend, `1024 x 64`: 9
/// diagonals 0.108 ms against 0.183 dense, 33 equal; `512`: 9 diagonals 0.062 against 0.034), and on the GPUs 25 to 125 times (RTX 3090:
/// 9 diagonals of `4096 x 64` in 0.032 ms against 0.32; the shader GEMM of wgpu: 33 diagonals 0.575 against 0.573). One figure for all
/// of them, the CPU's, which is the highest of the backends the GEMM is fast on. The GEMM does `rows` multiply-adds per output, the sum
/// `width`.
const SHIFT_TERM_COST: usize = 60;

/// Whether the fused sum of `width` shifted copies of the right-hand side of a product with a banded `rows x rows` matrix costs less than the
/// GEMM of the dense matrix: `width * SHIFT_TERM_COST < rows` (the right-hand side's columns scale both alike).
fn banded_shifts_win(width: usize, rows: usize) -> bool {
    width <= MAX_SHIFT_TERMS && width.saturating_mul(SHIFT_TERM_COST) < rows
}

/// `lhs * rhs` for a left operand known to be a permutation or banded, under `Numerics::exact_finite` (the dropped terms are
/// products with exact zeros): a row gather, or a sum of shifted, scaled copies of `rhs` as one fused expression where that costs less than
/// the GEMM ([`banded_shifts_win`]).
/// `None` for every other product (the GEMM runs).
pub(crate) fn route_product(lhs: &Array<f32>, rhs: &Array<f32>, location: &'static Location<'static>) -> Option<Array<f32>> {
    let ctx = lhs.context();
    if !ctx.numerics().exact_finite || !ctx.same_as(rhs.context()) {
        return None;
    }
    let [k, m, l2, l3] = lhs.shape().dims();
    let [n, rk, r2, r3] = rhs.shape().dims();
    if k != m || rk != k || [l2, l3, r2, r3] != [1, 1, 1, 1] || k < 2 {
        return None;
    }
    let props = lhs.props_used();
    if props.has(Props::PERMUTATION) {
        // Row `i` of `P X` is row `perm[i]` of `X`, `perm[i]` being the column of the one in row `i` of `P`.
        note_route(ctx, "matmul", location, "permutation-gather");
        let perm = lhs.argmax_axis(0).reshape([k, 1, 1]);
        return Some(super::gather_rows(&rhs.reshape([n, k, 1]), &perm, location).reshape([n, k]));
    }
    let (lo, hi) = props.band?;
    let (lo, hi) = (lo as usize, hi as usize);
    if lo >= k || hi >= k || !banded_shifts_win(lo + hi + 1, k) {
        return None;
    }
    note_route(ctx, "matmul", location, "banded-shifts");
    // `(A X)[c, r] = sum over s of A[r + s, r] X[c, r + s]`, `s` from `-lo` to `hi`: the diagonal `s` of `A` (a vector over the
    // rows) times `X` moved up by `s` rows, the rows that come from outside zero.
    let mut sum: Option<Array<f32>> = None;
    for s in -(lo as isize)..=(hi as isize) {
        let diagonal = if s >= 0 {
            lhs.diag_at(-s, location).pad_at([(0, s as usize)], PadMode::Zero, location)
        } else {
            lhs.diag_at(-s, location).pad_at([(s.unsigned_abs(), 0)], PadMode::Zero, location)
        };
        let coefficient = diagonal.reshape([1, k]).broadcast_to(rhs.shape());
        let term = &coefficient * &rhs.shift_fill([0, -s], 0.0);
        sum = Some(match sum {
            None => term,
            Some(acc) => &acc + &term,
        });
    }
    sum
}
