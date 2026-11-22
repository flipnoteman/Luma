//! The [`Sparse`] type: products with a vector and with a few dense columns in CSR, CSC and COO
//! (plain and transposed), conversions between the formats, construction from triplets and
//! from a dense matrix, `to_dense`, and elementwise operations with a dense matrix. The matrices
//! are the banded and power-law ones of the `spmv` cases (1M to 10M entries on a GPU); the
//! same matrices and operations are timed against ArrayFire in `tools/afcompare`.
//!
//! Values and operands are multiples of small powers of two, so sums are exact in any order and
//! every result is compared for equality with the `f64` host reference.

use luma::*;

use super::data::{self, Csr};
use super::*;

/// The matrix as `Sparse` CSR / COO / CSC, built from the arrays of the `spmv` upload.
struct Mats {
    csr: Sparse,
    coo: Sparse,
    csc: Sparse,
}

fn build(h: &Harness, a: &Csr) -> Mats {
    let dev = spmv::upload(h, a);
    let csr = Sparse::from_csr(a.rows, a.cols, &dev.ptr, &dev.col, &dev.val);
    let (coo, csc) = (csr.to_format(SparseFormat::Coo), csr.to_format(SparseFormat::Csc));
    for s in [&csr, &coo, &csc] {
        settle(s);
    }
    Mats { csr, coo, csc }
}

/// Evaluates every array of `s`.
fn settle(s: &Sparse) {
    let info = s.info();
    if let Some(v) = &info.values {
        v.eval().unwrap();
    }
    for i in [&info.row_idx, &info.col_idx].into_iter().flatten() {
        i.eval().unwrap();
    }
}

/// `y = A^T x` in `f64`.
fn spmv_t_host(a: &Csr, x: &[f32]) -> Vec<f64> {
    let mut y = vec![0.0f64; a.cols];
    for r in 0..a.rows {
        for k in a.ptr[r] as usize..a.ptr[r + 1] as usize {
            y[a.col[k] as usize] += f64::from(a.val[k]) * f64::from(x[r]);
        }
    }
    y
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
    for rows in [large / 10, large] {
        h.guarded("matrix", |h| products(h, rows, "banded"));
        h.guarded("matrix", |h| products(h, rows / 2, "powerlaw"));
        h.guarded("matrix", |h| products(h, rows / 2, "uniform"));
    }
    h.guarded("matrix", |h| conversions(h, large / 2));
    h.guarded("matrix", spmm);
    h.guarded("matrix", dense_forms);
}

fn matrix_of(kind: &str, rows: usize) -> Csr {
    match kind {
        "banded" => data::banded(rows, 2, 11),
        "uniform" => data::uniform(rows, 8, 24, 21),
        _ => data::power_law(rows, 4, 2.5, rows.min(1 << 17), 31),
    }
}

fn products(h: &mut Harness, rows: usize, kind: &str) {
    let prefix = format!("matrix/{kind}");
    if !wants(h, &prefix) {
        return;
    }
    let a = matrix_of(kind, rows);
    let m = build(h, &a);
    let nnz = a.nnz();
    let xh = data::quantized(a.cols, 12);
    let x = up_f32(h, &xh, a.cols);
    let want = data::spmv_host(&a, &xh);
    let want_t = spmv_t_host(&a, &xh);
    let size = format!("{} rows {} nnz", label(rows), label(nnz));
    variant(h, &format!("{prefix}_csr_spmv"), &size, nnz, &want, &|| m.csr.matmul(&x));
    variant(h, &format!("{prefix}_coo_spmv"), &size, nnz, &want, &|| m.coo.matmul(&x));
    variant(h, &format!("{prefix}_csc_spmv"), &size, nnz, &want, &|| m.csc.matmul(&x));
    variant(h, &format!("{prefix}_csr_spmv_t"), &size, nnz, &want_t, &|| m.csr.matmul_t(&x));
    variant(h, &format!("{prefix}_csc_spmv_t"), &size, nnz, &want_t, &|| m.csc.matmul_t(&x));
}

/// Building and converting: COO to CSR and CSC (sorts), CSR to CSC, CSR to COO, the transpose.
fn conversions(h: &mut Harness, rows: usize) {
    if !wants(h, "matrix/convert") {
        return;
    }
    let a = data::power_law(rows, 4, 2.5, rows.min(1 << 17), 33);
    let nnz = a.nnz();
    let dev = spmv::upload(h, &a);
    let size = format!("{} rows {} nnz", label(rows), label(nnz));
    let work = Work::Items(nnz as f64, "Mnnz/s");
    let (r, c, v) = (dev.row.clone(), dev.col.clone(), dev.val.clone());
    let csr = Sparse::from_csr(rows, rows, &dev.ptr, &dev.col, &dev.val);
    let coo = Sparse::from_coo(rows, rows, &r, &c, &v);

    let name = "matrix/convert_triplets_to_csr";
    if h.want(name) {
        let s = Sparse::from_triplets(rows, rows, &r, &c, &v, SparseFormat::Csr);
        exact(name, &s.row_idx().unwrap().to_vec().unwrap(), &a.ptr);
        exact(name, &s.col_idx().unwrap().to_vec().unwrap(), &a.col);
        exact(name, &s.values().unwrap().to_vec().unwrap(), &a.val);
        case_trim(h, name, &size, work, &mut || settle(&Sparse::from_triplets(rows, rows, &r, &c, &v, SparseFormat::Csr)));
    }
    let name = "matrix/convert_csr_to_csc";
    if h.want(name) {
        // A stable counting sort of the entries by column, host side.
        let s = csr.to_format(SparseFormat::Csc);
        let mut ptr = vec![0u32; rows + 1];
        a.col.iter().for_each(|&c| ptr[c as usize + 1] += 1);
        for i in 0..rows {
            ptr[i + 1] += ptr[i];
        }
        let mut next = ptr.clone();
        let (mut want_row, mut want_val) = (vec![0u32; nnz], vec![0f32; nnz]);
        for (k, &r) in a.row_of().iter().enumerate() {
            let slot = &mut next[a.col[k] as usize];
            want_row[*slot as usize] = r;
            want_val[*slot as usize] = a.val[k];
            *slot += 1;
        }
        exact(name, &s.col_idx().unwrap().to_vec().unwrap(), &ptr);
        exact(name, &s.row_idx().unwrap().to_vec().unwrap(), &want_row);
        exact(name, &s.values().unwrap().to_vec().unwrap(), &want_val);
        case_trim(h, name, &size, work, &mut || settle(&csr.to_format(SparseFormat::Csc)));
    }
    let name = "matrix/convert_csr_to_coo";
    if h.want(name) {
        let s = csr.to_format(SparseFormat::Coo);
        exact(name, &s.row_idx().unwrap().to_vec().unwrap(), &a.row_of());
        // The cached expansion is built once per matrix; time it from a fresh matrix.
        case_trim(h, name, &size, work, &mut || settle(&Sparse::from_csr(rows, rows, &dev.ptr, &dev.col, &dev.val).to_format(SparseFormat::Coo)));
    }
    let name = "matrix/convert_coo_to_csc";
    if h.want(name) {
        case_trim(h, name, &size, work, &mut || settle(&coo.to_format(SparseFormat::Csc)));
    }
}

/// Sparse times `c` dense columns.
fn spmm(h: &mut Harness) {
    if !wants(h, "matrix/spmm") {
        return;
    }
    let rows = scale(h, 16_384, 65_536, 262_144);
    let a = data::uniform(rows, 8, 24, 91);
    let m = build(h, &a);
    for c in [4usize, 16, 64] {
        let xh = data::quantized(rows * c, 92 + c as u64);
        let x = up_f32(h, &xh, [c, rows]);
        let mut want = vec![0.0f64; rows * c];
        for r in 0..rows {
            for k in a.ptr[r] as usize..a.ptr[r + 1] as usize {
                for j in 0..c {
                    want[r * c + j] += f64::from(a.val[k]) * f64::from(xh[a.col[k] as usize * c + j]);
                }
            }
        }
        let size = format!("{} rows {} nnz, {c} columns", label(rows), label(a.nnz()));
        let work = Work::Flops(2.0 * a.nnz() as f64 * c as f64);
        for (tag, s) in [("csr", &m.csr), ("coo", &m.coo), ("csc", &m.csc)] {
            let name = format!("matrix/spmm_{tag}_c{c}");
            if h.want(&name) {
                check_flat(&name, &s.matmul(&x).to_vec().unwrap(), &want, 0.0, 0.0, 0.0);
                case_trim(h, &name, &size, work, &mut || s.matmul(&x).eval().unwrap());
            }
        }
    }
}

/// Dense forms: a square dense matrix with 2 % entries to sparse and back, and the elementwise
/// product, sum and difference with a dense matrix.
fn dense_forms(h: &mut Harness) {
    if !wants(h, "matrix/dense") {
        return;
    }
    let n = scale(h, 1024, 2048, 4096);
    let a = data::uniform(n, n / 100, n / 50, 71);
    let mut dh = vec![0.0f32; n * n];
    for r in 0..n {
        for k in a.ptr[r] as usize..a.ptr[r + 1] as usize {
            dh[r * n + a.col[k] as usize] += a.val[k];
        }
    }
    // Entries cancelling to zero are not entries: the sparse matrix has what the dense one holds.
    let held = dh.iter().filter(|&&v| v != 0.0).count();
    let d = up_f32(h, &dh, [n, n]);
    let oh = data::quantized(n * n, 72);
    let o = up_f32(h, &oh, [n, n]);
    let size = format!("{n} x {n} dense, {} nnz", label(held));
    let items = Work::Items((n * n) as f64, "Mel/s");

    let name = "matrix/dense_to_csr";
    if h.want(name) {
        let s = Sparse::from_dense(&d, SparseFormat::Csr);
        assert_eq!(s.nnz(), held, "{name}: nnz");
        exact(name, &s.to_dense().to_vec().unwrap(), &dh);
        case_trim(h, name, &size, items, &mut || settle(&Sparse::from_dense(&d, SparseFormat::Csr)));
    }
    let s = Sparse::from_dense(&d, SparseFormat::Csr);
    settle(&s);
    let name = "matrix/csr_to_dense";
    if h.want(name) {
        case_trim(h, name, &size, items, &mut || s.to_dense().eval().unwrap());
    }
    let name = "matrix/mul_dense";
    if h.want(name) {
        let want: Vec<f32> = dh.iter().zip(&oh).map(|(a, b)| a * b).collect();
        exact(name, &s.mul_dense(&o).to_dense().to_vec().unwrap(), &want);
        case_trim(h, name, &size, Work::Items(held as f64, "Mnnz/s"), &mut || settle(&s.mul_dense(&o)));
    }
    let name = "matrix/add_dense";
    if h.want(name) {
        let want: Vec<f32> = dh.iter().zip(&oh).map(|(a, b)| a + b).collect();
        exact(name, &s.add_dense(&o).to_vec().unwrap(), &want);
        case_trim(h, name, &size, items, &mut || s.add_dense(&o).eval().unwrap());
    }
    let name = "matrix/scale";
    if h.want(name) {
        case_trim(h, name, &size, Work::Items(held as f64, "Mnnz/s"), &mut || settle(&s.mul_scalar(2.0)));
    }
}
