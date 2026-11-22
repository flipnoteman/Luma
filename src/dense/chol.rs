//! Cholesky factorisation and SPD solves.
//!
//! A big matrix is factored recursively, `A = [A11 A21ᵀ; A21 A22]`: `L11 = chol(A11)`,
//! `L21ᵀ = L11⁻¹ A21ᵀ` (a triangular solve, see [`tri`](super::tri)), `L22 = chol(A22 - L21 L21ᵀ)`
//! with the update one [`matmul_f32`]; the big update happens once at the top, so the elementwise
//! passes over the trailing matrix stay `O(n²)` in all. Up to 64 rows, and for every matrix of a
//! batch, one workgroup factors a matrix in workgroup memory.

use std::panic::Location;

use super::tri::{lower_left, rhs_matrix, split};
use super::{block, fail, launch, nth, per_matrix, stack_batch, Dims, NAN};
use crate::array::Array;
use crate::context::Context;
use crate::kernel::{uniform, KernelArg};
use crate::matmul::matmul_f32;
use crate::view::concat;

/// Rows of the matrix a workgroup factors in workgroup memory, at most: 64 where the device has the
/// 16.5 KiB that takes, else 48.
fn leaf(ctx: &Context) -> usize {
    if ctx.info().limits.max_workgroup_storage_bytes >= 64 * 65 * 4 + 64 * 4 { 64 } else { 48 }
}

#[derive(encase::ShaderType)]
struct Params {
    n: u32,
    batch: u32,
}

/// The leaf kernel: a workgroup of `LEAF` invocations factors one matrix of up to `leaf` rows in
/// workgroup memory, an invocation per row. Column `j` of every row `i >= j` is
/// `(a_ij - sum_k l_ik l_jk) / l_jj`, where the pivot `l_jj` is computed by every invocation from
/// row `j` (a few more flops, one barrier per column instead of two); the diagonal goes to its own
/// array so that the row `j` the others read is not overwritten under them.
fn source(leaf: usize) -> String {
    let pitch = leaf | 1;
    format!(
        "struct Params {{ n: u32, batch: u32 }}
@group(1) @binding(0) var<storage, read> a: array<f32>;
@group(1) @binding(1) var<storage, read_write> l: array<f32>;
@group(1) @binding(2) var<uniform> p: Params;
var<workgroup> s: array<f32, {size}>;
var<workgroup> dv: array<f32, {leaf}>;

@compute @workgroup_size({leaf})
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_index) t: u32) {{
    let m = wid.x + wid.y * nwg.x;
    let n = select(0u, p.n, m < p.batch);
    let base = m * p.n * p.n;
    for (var idx = t; idx < n * n; idx = idx + {leaf}u) {{
        let i = idx / n;
        let j = idx - i * n;
        s[i * {pitch}u + j] = select(0.0, a[base + idx], j <= i);
    }}
    workgroupBarrier();
    let ri = t * {pitch}u;
    for (var j = 0u; j < n; j = j + 1u) {{
        let rj = j * {pitch}u;
        var v = s[ri + j];
        var d = s[rj + j];
        var k = 0u;
        for (; k + 4u <= j; k = k + 4u) {{
            let b0 = s[rj + k];
            let b1 = s[rj + k + 1u];
            let b2 = s[rj + k + 2u];
            let b3 = s[rj + k + 3u];
            let a0 = s[ri + k];
            let a1 = s[ri + k + 1u];
            let a2 = s[ri + k + 2u];
            let a3 = s[ri + k + 3u];
            v = fma(-a0, b0, v);
            v = fma(-a1, b1, v);
            v = fma(-a2, b2, v);
            v = fma(-a3, b3, v);
            d = fma(-b0, b0, d);
            d = fma(-b1, b1, d);
            d = fma(-b2, b2, d);
            d = fma(-b3, b3, d);
        }}
        for (; k < j; k = k + 1u) {{
            let lj = s[rj + k];
            v = fma(-s[ri + k], lj, v);
            d = fma(-lj, lj, d);
        }}
        let r = select({NAN}, sqrt(d), d > 0.0);
        if (t > j && t < n) {{
            s[ri + j] = v / r;
        }}
        if (t == j) {{
            dv[j] = r;
        }}
        workgroupBarrier();
    }}
    for (var idx = t; idx < n * n; idx = idx + {leaf}u) {{
        let i = idx / n;
        let j = idx - i * n;
        l[base + idx] = select(select(0.0, dv[i], i == j), s[i * {pitch}u + j], j < i);
    }}
}}
",
        size = leaf * pitch,
    )
}

/// One launch over the batch.
fn small(a: &Array<f32>, d: Dims, loc: &'static Location<'static>) -> Array<f32> {
    let ctx = a.context();
    let mut out = match Array::<f32>::alloc(ctx, a.shape()) {
        Ok(out) => out,
        Err(e) => return fail(ctx, a.shape(), e, "cholesky", loc),
    };
    let params = Params { n: d.rows as u32, batch: d.batch as u32 };
    let args: [KernelArg<'_>; 3] = [a.into(), (&mut out).into(), uniform(&params)];
    match launch(ctx, &format!("chol/{}", leaf(ctx)), "cholesky", || source(leaf(ctx)), per_matrix(ctx, d.batch), &args) {
        Ok(()) => out,
        Err(e) => fail(ctx, a.shape(), e, "cholesky", loc),
    }
}

/// The recursion over one `[n, n]` matrix.
fn recurse(a: &Array<f32>, loc: &'static Location<'static>) -> Array<f32> {
    let [n, ..] = a.shape().dims();
    if n <= leaf(a.context()) {
        return small(a, Dims { rows: n, cols: n, batch: 1 }, loc);
    }
    let h = split(n);
    let l11 = recurse(&block(a, 0..h, 0..h), loc);
    // Y = L21ᵀ = L11⁻¹ A21ᵀ, `[n - h, h]`.
    let y = lower_left(&l11, &block(a, h..n, 0..h).transpose(), false, "cholesky", loc);
    let y = y.cache();
    let yt = y.transpose();
    let s = &block(a, h..n, h..n) - &matmul_f32(&yt, &y);
    let l22 = recurse(&s, loc);
    let zeros = Array::<f32>::zeros_on(a.context(), [n - h, h]);
    let upper = match zeros {
        Ok(z) => concat(0, [&l11, &z]),
        Err(e) => return fail(a.context(), a.shape(), e, "cholesky", loc),
    };
    concat(1, [&upper, &concat(0, [&yt, &l22])])
}

/// The lower Cholesky factor of the lower triangle of every matrix of `a`.
pub(crate) fn lower(a: &Array<f32>, loc: &'static Location<'static>) -> Array<f32> {
    let d = match Dims::square(a) {
        Ok(d) => d,
        Err(e) => return fail(a.context(), a.shape(), e, "cholesky", loc),
    };
    if d.rows <= leaf(a.context()) {
        return small(a, d, loc);
    }
    if d.batch > 1 {
        let parts: Vec<Array<f32>> = (0..d.batch).map(|i| recurse(&nth(a, i), loc)).collect();
        return stack_batch(&parts);
    }
    recurse(a, loc)
}

impl Array<f32> {
    /// The Cholesky factor `L` of a symmetric positive definite matrix `A` (`self`, `[n, n]` or a
    /// batch `[n, n, batch]`), `A = L Lᵀ` with `L` lower triangular (the strict upper triangle is
    /// zero). Only the lower triangle of `A` is read.
    ///
    /// A matrix that is not positive definite (a pivot that is not positive, or `NaN`) leaves a
    /// `NaN` on the diagonal of its factor from the failing column on; nothing is checked or
    /// raised. Matrices of up to 64 rows (and every matrix of a batch) are factored by one
    /// workgroup each in one launch; bigger ones recursively with [`matmul_f32`] products.
    ///
    /// ```
    /// # use luma::*;
    /// # let ctx = Context::new()?;
    /// // [[4, 2], [2, 5]] = L Lᵀ with L = [[2, 0], [1, 2]]
    /// let a = Array::from_slice(&[4.0f32, 2.0, 2.0, 5.0], [2, 2])?;
    /// assert_eq!(a.cholesky().to_vec()?, vec![2.0, 0.0, 1.0, 2.0]);
    /// # Ok::<(), luma::LumaError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// The result is poisoned if the matrices are not square.
    #[track_caller]
    pub fn cholesky(&self) -> Array<f32> {
        lower(self, Location::caller())
    }

    /// The upper Cholesky factor `U`, `A = Uᵀ U`, reading the upper triangle of `A`: the
    /// transpose of the lower factor of `Aᵀ`.
    ///
    /// # Errors
    ///
    /// As [`Array::cholesky`].
    #[track_caller]
    pub fn cholesky_upper(&self) -> Array<f32> {
        lower(&self.transpose(), Location::caller()).transpose()
    }

    /// Solves `A X = B` for a symmetric positive definite `A` (`self`, only its lower triangle is
    /// read) by Cholesky: `L Lᵀ X = B`. `b` is `[k, n]` (`[k, n, batch]`), or a vector (see
    /// [`Array::solve_triangular`]); the result has its shape. A matrix that is not positive
    /// definite gives `NaN`.
    ///
    /// # Errors
    ///
    /// The result is poisoned if `A` is not square or `b` does not match it.
    #[track_caller]
    pub fn solve_spd(&self, b: &Array<f32>) -> Array<f32> {
        let loc = Location::caller();
        let l = lower(self, loc);
        cholesky_solve(&l, b, loc)
    }

    /// Solves `L Lᵀ X = B` given the lower Cholesky factor `L` (`self`, as [`Array::cholesky`]
    /// returns it): the two triangular solves of [`Array::solve_spd`].
    ///
    /// # Errors
    ///
    /// The result is poisoned if `L` is not square or `b` does not match it.
    #[track_caller]
    pub fn cholesky_solve(&self, b: &Array<f32>) -> Array<f32> {
        cholesky_solve(self, b, Location::caller())
    }
}

fn cholesky_solve(l: &Array<f32>, b: &Array<f32>, loc: &'static Location<'static>) -> Array<f32> {
    let (rhs, vector) = rhs_matrix(l, b);
    let y = lower_left(l, &rhs, false, "cholesky_solve", loc);
    // Lᵀ X = Y: the transpose is upper, solved on its flip.
    let lt = l.transpose().flip(0).flip(1);
    let x = lower_left(&lt, &y.flip(1), false, "cholesky_solve", loc).flip(1);
    if vector { x.reshape(b.shape()) } else { x }
}


