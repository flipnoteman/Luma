//! CSR sparse matrix-vector products: a gather of `x` at the column indices, a product with the
//! values and a segmented sum over the rows, four ways.
//!
//! * `atomic`: `group_sum` of the products by the row of every entry (one atomic add per entry);
//! * `running`: one `cumsum` over all entries and the difference at the row boundaries (no
//!   atomics, but it needs a long running sum and its rounding error scales with it);
//! * `ell`: rows padded to the longest, products reduced with `sum_axis` over equal lines;
//! * `split`: rows cut into pieces of 16 entries, `sum_axis` over the pieces, then a `group_sum`
//!   of the piece sums by row (atomics only for the rows longer than a piece);
//! * `segments`: `sum_segments` of the products over the row pointer (entries split evenly over
//!   the workgroups whatever the row lengths, no atomics, two launches).
//!
//! Values and `x` are multiples of 1/16, so every sum is exact in `f32` in any order and the
//! results are compared for equality with the `f64` host product.

use luma::*;

use super::data::{self, Csr};
use super::*;

/// A CSR matrix on the device, with the row of every entry expanded (COO rows).
pub struct Dev {
    pub rows: usize,
    pub nnz: usize,
    pub ptr: Array<u32>,
    pub col: Array<u32>,
    pub val: Array<f32>,
    pub row: Array<u32>,
}

pub fn upload(h: &Harness, a: &Csr) -> Dev {
    Dev {
        rows: a.rows,
        nnz: a.nnz(),
        ptr: up_u32(h, &a.ptr, a.rows + 1),
        col: up_u32(h, &a.col, a.nnz()),
        val: up_f32(h, &a.val, a.nnz()),
        row: up_u32(h, &a.row_of(), a.nnz()),
    }
}

pub struct DevEll {
    pub w: usize,
    pub rows: usize,
    pub col: Array<u32>,
    pub val: Array<f32>,
}

pub fn upload_ell(h: &Harness, a: &Csr) -> DevEll {
    let e = data::ell(a);
    DevEll { w: e.w, rows: a.rows, col: up_u32(h, &e.col, [e.w, a.rows]), val: up_f32(h, &e.val, [e.w, a.rows]) }
}

pub struct DevPieces {
    pub rows: usize,
    pub col: Array<u32>,
    pub val: Array<f32>,
    pub row: Array<u32>,
}

pub fn upload_pieces(h: &Harness, a: &Csr, l: usize) -> DevPieces {
    let p = data::pieces(a, l);
    let count = p.row.len();
    DevPieces { rows: a.rows, col: up_u32(h, &p.col, [l, count]), val: up_f32(h, &p.val, [l, count]), row: up_u32(h, &p.row, [1, count]) }
}

/// `y = A x` with an atomic `group_sum` by row.
pub fn atomic(a: &Dev, x: &Array<f32>) -> Array<f32> {
    (&a.val * x.take(&a.col)).group_sum(&a.row, a.rows)
}

/// `y = A x` with `sum_segments` over the row pointer.
pub fn segments(a: &Dev, x: &Array<f32>) -> Array<f32> {
    (&a.val * x.take(&a.col)).sum_segments(&a.ptr)
}

/// `y = A x` from one running sum of the products.
pub fn running(a: &Dev, x: &Array<f32>) -> Array<f32> {
    let p = (&a.val * x.take(&a.col)).cumsum(0);
    let lo = a.ptr.slice(0, 0..a.rows);
    let hi = a.ptr.slice(0, 1..=a.rows);
    // The running sum up to (excluding) `end`.
    let upto = |end: &Array<u32>| select(&end.eq(0u32), 0.0f32, p.take(&(end - 1u32)));
    upto(&hi) - upto(&lo)
}

/// `y = A x` from an ELLPACK matrix: the result has shape `[1, rows]`.
pub fn ell(a: &DevEll, x: &Array<f32>) -> Array<f32> {
    (&a.val * x.take(&a.col)).sum_axis(0)
}

/// `y = A x` from rows split in pieces.
pub fn split(a: &DevPieces, x: &Array<f32>) -> Array<f32> {
    (&a.val * x.take(&a.col)).sum_axis(0).group_sum(&a.row, a.rows)
}

/// Verifies `f()` against `want`, then times it.
fn variant(h: &mut Harness, name: &str, size: &str, nnz: usize, want: &[f64], f: &dyn Fn() -> Array<f32>) {
    if !h.want(name) {
        return;
    }
    check_flat(name, &f().to_vec().unwrap(), want, 0.0, 0.0, 0.0);
    case_trim(h, name, size, Work::Items(nnz as f64, "Mnnz/s"), &mut || f().eval().unwrap());
}

pub fn run(h: &mut Harness) {
    let large = scale(h, 65_536, 262_144, 2_000_000);
    let mut sizes = vec![16_384];
    if large != 16_384 {
        sizes.push(large);
    }
    for rows in sizes {
        h.guarded("spmv", |h| banded(h, rows));
        h.guarded("spmv", |h| uniform(h, rows));
        h.guarded("spmv", |h| power_law(h, rows));
    }
    h.guarded("spmv", |h| empty_rows(h, 16_384));
}

fn size_of(a: &Csr) -> String {
    format!("{} rows {} nnz", label(a.rows), label(a.nnz()))
}

fn banded(h: &mut Harness, rows: usize) {
    if !wants(h, "spmv/banded") {
        return;
    }
    let a = data::banded(rows, 2, 11);
    let xh = data::quantized(a.cols, 12);
    let want = data::spmv_host(&a, &xh);
    let (dev, e, x) = (upload(h, &a), upload_ell(h, &a), up_f32(h, &xh, a.cols));
    let size = size_of(&a);
    variant(h, "spmv/banded_atomic", &size, a.nnz(), &want, &|| atomic(&dev, &x));
    variant(h, "spmv/banded_running", &size, a.nnz(), &want, &|| running(&dev, &x));
    variant(h, "spmv/banded_segments", &size, a.nnz(), &want, &|| segments(&dev, &x));
    variant(h, "spmv/banded_ell", &size, a.nnz(), &want, &|| ell(&e, &x));
}

fn uniform(h: &mut Harness, rows: usize) {
    if !wants(h, "spmv/uniform") {
        return;
    }
    let a = data::uniform(rows, 8, 24, 21);
    let xh = data::quantized(a.cols, 22);
    let want = data::spmv_host(&a, &xh);
    let (dev, e, x) = (upload(h, &a), upload_ell(h, &a), up_f32(h, &xh, a.cols));
    let size = size_of(&a);
    variant(h, "spmv/uniform_atomic", &size, a.nnz(), &want, &|| atomic(&dev, &x));
    variant(h, "spmv/uniform_running", &size, a.nnz(), &want, &|| running(&dev, &x));
    variant(h, "spmv/uniform_segments", &size, a.nnz(), &want, &|| segments(&dev, &x));
    variant(h, "spmv/uniform_ell", &size, a.nnz(), &want, &|| ell(&e, &x));
}

fn power_law(h: &mut Harness, rows: usize) {
    if !wants(h, "spmv/powerlaw") {
        return;
    }
    let a = data::power_law(rows, 4, 2.5, rows.min(1 << 17), 31);
    let xh = data::quantized(a.cols, 32);
    let want = data::spmv_host(&a, &xh);
    let (dev, x) = (upload(h, &a), up_f32(h, &xh, a.cols));
    let size = format!("{} (longest row {})", size_of(&a), a.max_row());
    variant(h, "spmv/powerlaw_atomic", &size, a.nnz(), &want, &|| atomic(&dev, &x));
    variant(h, "spmv/powerlaw_running", &size, a.nnz(), &want, &|| running(&dev, &x));
    variant(h, "spmv/powerlaw_segments", &size, a.nnz(), &want, &|| segments(&dev, &x));
    let p = upload_pieces(h, &a, 16);
    variant(h, "spmv/powerlaw_split", &size, a.nnz(), &want, &|| split(&p, &x));
}

/// Empty rows (the first ones too), which a segmented sum from row boundaries must get right.
fn empty_rows(h: &mut Harness, rows: usize) {
    if !wants(h, "spmv/emptyrows") {
        return;
    }
    let a = data::with_empty_rows(rows, 41);
    let xh = data::quantized(a.cols, 42);
    let want = data::spmv_host(&a, &xh);
    let (dev, e, x) = (upload(h, &a), upload_ell(h, &a), up_f32(h, &xh, a.cols));
    let p = upload_pieces(h, &a, 16);
    let size = size_of(&a);
    variant(h, "spmv/emptyrows_atomic", &size, a.nnz(), &want, &|| atomic(&dev, &x));
    variant(h, "spmv/emptyrows_running", &size, a.nnz(), &want, &|| running(&dev, &x));
    variant(h, "spmv/emptyrows_segments", &size, a.nnz(), &want, &|| segments(&dev, &x));
    variant(h, "spmv/emptyrows_ell", &size, a.nnz(), &want, &|| ell(&e, &x));
    variant(h, "spmv/emptyrows_split", &size, a.nnz(), &want, &|| split(&p, &x));
}
