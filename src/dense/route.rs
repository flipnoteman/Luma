//! Property-aware routing (W3-R6): what a matrix known to be diagonal, triangular or banded is solved with, and the explicit
//! banded solve.
//!
//! The automatic routes are rules of class `Reformulating` (they run under [`Numerics::reformulate`](crate::Numerics), which
//! `algebraic()` and `fast()` set): a forward substitution does not pivot by magnitude like the LU of `solve` does, and the
//! product of a diagonal is not `exp(sum(ln))`. They fire only for a matrix whose handle carries the property (see
//! [`crate::props`]); anything in doubt, a shape the routes do not cover, runs the dense path. [`Array::solve_banded`] is the one
//! opt-in for a band: its line solvers do not pivot across rows, so `solve` never routes a band by itself.

use std::panic::Location;

use super::tri::rhs_matrix;
use super::{fail, Dims};
use crate::array::Array;
use crate::context::Context;
use crate::error::LumaError;
use crate::jit::comprehension::Props;
use crate::profile::{Decision, DecisionSite, Verdict};
use crate::shape::Shape;
use crate::view::PadMode;
use crate::Triangle;

/// Records which route a call took, when decisions are being recorded.
fn note_route(ctx: &Context, op: &'static str, location: &'static Location<'static>, chosen: &'static str) {
    if ctx.jit().recording_decisions() {
        ctx.jit().note_decision(Decision::new(op, location, DecisionSite::Rewrite, chosen));
    }
}

impl Array<f32> {
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
        sum.expect("at least one diagonal")
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
        if lo > 1 || hi > 1 || d.batch != 1 || d.rows < 2 || rows != d.rows || rb2 != 1 || rb3 != 1 {
            let reason = if lo > 1 || hi > 1 { "band wider than 1" } else { "batch or shape the line solver does not take" };
            if ctx.jit().recording_decisions() {
                ctx.jit().note_decision(Decision::new("solve_banded", location, DecisionSite::Rewrite, "dense").candidate("tridiagonal", Verdict::Rejected(reason)));
            }
            return self.solve(b);
        }
        note_route(ctx, "solve_banded", location, "tridiagonal");
        let n = d.rows;
        let shape = Shape::from([k, n]);
        // Row `r` of the system: `lo[r] x[r-1] + diag[r] x[r] + up[r] x[r+1]`. In this crate's orientation the entry below the
        // diagonal of row `r` is `diag(1)[r - 1]` and the one above it `diag(-1)[r]` (see `crate::props`).
        let column = |v: Array<f32>| v.reshape([1, n]).broadcast_to(shape);
        let diag = column(self.diag_at(0, location));
        let lower = column(self.diag_at(1, location).pad_at([(1, 0)], PadMode::Zero, location));
        let upper = column(self.diag_at(-1, location).pad_at([(0, 1)], PadMode::Zero, location));
        let x = crate::linalg::solve_tridiagonal(1, &lower, &diag, &upper, &rhs);
        if vector { x.reshape(b.shape()) } else { x }
    }

    /// What `solve` does for a matrix whose handle says it is diagonal or triangular, under `Numerics::reformulate`: `None` for
    /// every other call (the dense LU path then runs).
    pub(crate) fn route_solve(&self, b: &Array<f32>, location: &'static Location<'static>) -> Option<Array<f32>> {
        let props = self.props_used();
        let ctx = self.context();
        if !ctx.numerics().reformulate || Dims::square(self).is_err() || !ctx.same_as(b.context()) {
            return None;
        }
        let [n, _, a2, a3] = self.shape().dims();
        if props.has(Props::DIAGONAL) {
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
