//! The symmetric eigenproblem.
//!
//! `eigh(A)` reads the lower triangle of `A`, shifts it by `c I` with `c = ||A||_F >= |lambda|` so
//! that it is positive semi-definite, and takes the one-sided Jacobi SVD of the result (see
//! [`svd`](super::svd)): the singular values of a positive semi-definite matrix are its
//! eigenvalues and `U = V`, so `V` holds the eigenvectors, with no sign ambiguity and no `±lambda`
//! mixing; `lambda = sigma - c`, sorted ascending. The matrix is symmetrised, shifted and scaled
//! inside the decomposition kernel (a matrix that fits workgroup memory, a batch of them
//! included, is a single launch).
//!
//! Accuracy: the eigenvalues carry an *absolute* error of a few `eps ||A||_F` (the shift costs
//! the relative accuracy of eigenvalues much smaller than the norm), the eigenvectors are
//! orthonormal to a few `eps` and `||A v - lambda v|| <= c eps ||A||_F`.

use std::panic::Location;

use super::svd::{decompose, Kind};
use crate::array::Array;

/// The eigendecomposition of a symmetric matrix, see [`Array::eigh`].
#[derive(Clone, Debug)]
pub struct Eigh {
    /// The eigenvalues in ascending order, `[n, 1]` (`[n, 1, batch]`).
    pub values: Array<f32>,
    /// The eigenvectors as columns, `[n, n]` (`[n, n, batch]`): column `k` is the eigenvector of
    /// `values[k]`, `A V = V diag(values)`, `Vᵀ V = I`.
    pub vectors: Array<f32>,
}

impl Array<f32> {
    /// The eigenvalues and eigenvectors of a symmetric matrix `A` (`self`, `[n, n]` or a batch
    /// `[n, n, batch]`); only the lower triangle is read. See [`Eigh`] for the layout and the
    /// module notes for the method and the accuracy.
    ///
    /// ```
    /// # use luma::*;
    /// # let ctx = Context::new()?;
    /// // [[2, 1], [1, 2]] has eigenvalues 1 and 3.
    /// let a = Array::from_slice(&[2.0f32, 1.0, 1.0, 2.0], [2, 2])?;
    /// let e = a.eigh().values.to_vec()?;
    /// assert!((e[0] - 1.0).abs() < 1e-5 && (e[1] - 3.0).abs() < 1e-5);
    /// # Ok::<(), luma::LumaError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// The result is poisoned if the matrices are not square. A matrix with a `NaN` gives `NaN`
    /// values; nothing hangs.
    #[track_caller]
    pub fn eigh(&self) -> Eigh {
        let o = decompose(self, Kind::Eigh, true, "eigh", Location::caller());
        Eigh { values: o.s, vectors: o.vt.expect("eigh computes the vectors") }
    }

    /// The eigenvalues of a symmetric matrix alone, ascending, `[n, 1]` (`[n, 1, batch]`):
    /// cheaper than [`Array::eigh`] (no eigenvectors).
    ///
    /// # Errors
    ///
    /// As [`Array::eigh`].
    #[track_caller]
    pub fn eigvalsh(&self) -> Array<f32> {
        decompose(self, Kind::Eigh, false, "eigvalsh", Location::caller()).s
    }
}
