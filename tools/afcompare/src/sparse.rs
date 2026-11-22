//! Sparse matrices: luma's `Sparse` against ArrayFire's sparse arrays (cuSPARSE underneath) on
//! the banded and power-law matrices of `luma-bench`'s `sparse` suite (1M and 10M entries):
//! products with a vector and with 16 dense columns, the transposed product, the conversions,
//! construction from triplets and from a dense matrix, `to_dense`, and elementwise operations
//! with a dense matrix.
//!
//! The matrices have sorted, distinct columns in every row (what cuSPARSE expects), values that
//! are multiples of 1/16 and vectors of multiples of 1/16, so every product is exact in `f32`
//! (up to the length of the longest row) and both libraries are checked against an `f64` host
//! product. ArrayFire stores `i32` indices and supports `matmul` with a CSR left operand only,
//! so the CSC / COO products of luma have no row. Dense operands: a `rows x cols` dense matrix is
//! `[cols, rows]` in luma and `(rows, cols)` column-major in ArrayFire (its sparse arrays take the
//! row and column indices as they are); a dense right-hand side of `c` columns is `[c, n]` in
//! luma (the `c` values of an index adjacent, the natural gather layout) and `(n, c)` in
//! ArrayFire, so the data is transposed on the host before upload and results are compared in
//! their own layout after a host transpose.

use arrayfire as af;
use luma::{Array as LA, Sparse, SparseFormat};

use crate::afx;
use crate::bench::Bench;
use crate::cases::{attempt, human, A, L};
use crate::lumax;
use crate::util::*;

/// A CSR matrix on the host.
pub struct Csr {
    pub rows: usize,
    pub cols: usize,
    pub ptr: Vec<u32>,
    pub col: Vec<u32>,
    pub val: Vec<f32>,
}

impl Csr {
    pub fn nnz(&self) -> usize {
        self.col.len()
    }

    pub fn row_of(&self) -> Vec<u32> {
        let mut row = Vec::with_capacity(self.nnz());
        for r in 0..self.rows {
            row.extend(std::iter::repeat_n(r as u32, (self.ptr[r + 1] - self.ptr[r]) as usize));
        }
        row
    }

    /// `y = A x` (or `A^T x`) in `f64` for `c` columns of `x` stored `[n, c]` row-major.
    pub fn product(&self, x: &[f32], c: usize, transposed: bool) -> Vec<f64> {
        let out = if transposed { self.cols } else { self.rows };
        let mut y = vec![0.0f64; out * c];
        for r in 0..self.rows {
            for k in self.ptr[r] as usize..self.ptr[r + 1] as usize {
                let col = self.col[k] as usize;
                let (src, dst) = if transposed { (r, col) } else { (col, r) };
                for j in 0..c {
                    y[dst * c + j] += f64::from(self.val[k]) * f64::from(x[src * c + j]);
                }
            }
        }
        y
    }
}

/// `below(n)` and a value that is a multiple of 1/16 in `[-1, 1]`, as luma-bench's generator.
struct Gen(Rng);

impl Gen {
    fn new(seed: u64) -> Gen {
        Gen(Rng::new(seed))
    }
    fn below(&mut self, n: usize) -> usize {
        ((self.0.next_u64() >> 11) % n.max(1) as u64) as usize
    }
    fn unit(&mut self) -> f64 {
        (self.0.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    fn quant(&mut self) -> f32 {
        (self.below(33) as f32 - 16.0) / 16.0
    }
}

fn finish(rows: usize, cols: usize, g: &mut Gen, mut row_cols: impl FnMut(usize, &mut Gen, &mut Vec<u32>)) -> Csr {
    let mut a = Csr { rows, cols, ptr: Vec::with_capacity(rows + 1), col: Vec::new(), val: Vec::new() };
    a.ptr.push(0);
    let mut scratch = Vec::new();
    for r in 0..rows {
        scratch.clear();
        row_cols(r, g, &mut scratch);
        scratch.sort_unstable();
        scratch.dedup();
        a.col.extend_from_slice(&scratch);
        a.ptr.push(a.col.len() as u32);
    }
    a.val = (0..a.col.len()).map(|_| g.quant()).collect();
    a
}

/// `2 * half + 1` diagonals.
pub fn banded(rows: usize, half: usize, seed: u64) -> Csr {
    finish(rows, rows, &mut Gen::new(seed), |r, _, out| out.extend(r.saturating_sub(half) as u32..=(r + half).min(rows - 1) as u32))
}

/// Pareto row lengths (`dmin` 4, exponent 2.5, at most `cap`) at random columns: hub rows next to
/// rows of a handful of entries.
pub fn power_law(rows: usize, cap: usize, seed: u64) -> Csr {
    finish(rows, rows, &mut Gen::new(seed), |_, g, out| {
        let u = 1.0 - g.unit();
        let d = ((4.0 * u.powf(-1.0 / 1.5)) as usize).clamp(4, cap);
        out.extend((0..d).map(|_| g.below(rows) as u32));
    })
}

fn up_i32(v: &[u32]) -> Vec<i32> {
    v.iter().map(|&x| x as i32).collect()
}

/// A matrix on both sides, in CSR (and COO for the luma triplets).
struct Both {
    l_csr: Sparse,
    l_coo: Sparse,
    l_csc: Sparse,
    a_csr: A,
}

fn build(ctx: &luma::Context, a: &Csr) -> Both {
    let (ptr, col, val) = (lumax::arr(ctx, &a.ptr, a.rows + 1), lumax::arr(ctx, &a.col, a.nnz()), lumax::arr(ctx, &a.val, a.nnz()));
    let l_csr = Sparse::from_csr(a.rows, a.cols, &ptr, &col, &val);
    let l_coo = l_csr.to_format(SparseFormat::Coo);
    let l_csc = l_csr.to_format(SparseFormat::Csc);
    for s in [&l_csr, &l_coo, &l_csc] {
        settle(s);
    }
    ctx.sync().unwrap();
    let (ap, ac, av) = (afx::arr(&up_i32(&a.ptr), &[a.rows as u64 + 1]), afx::arr(&up_i32(&a.col), &[a.nnz() as u64]), afx::arr(&a.val, &[a.nnz() as u64]));
    let a_csr = af::sparse(a.rows as u64, a.cols as u64, &av, &ap, &ac, af::SparseFormat::CSR);
    a_csr.eval();
    afx::sync();
    Both { l_csr, l_coo, l_csc, a_csr }
}

fn settle(s: &Sparse) {
    let info = s.info();
    if let Some(v) = &info.values {
        v.eval().unwrap();
    }
    for i in [&info.row_idx, &info.col_idx].into_iter().flatten() {
        i.eval().unwrap();
    }
}

fn transposed(x: &[f32], n: usize, c: usize) -> Vec<f32> {
    let mut t = vec![0.0f32; x.len()];
    for i in 0..n {
        for j in 0..c {
            t[j * n + i] = x[i * c + j];
        }
    }
    t
}

fn size_label(a: &Csr) -> String {
    format!("{} rows, {} nnz", human(a.rows), human(a.nnz()))
}

pub fn sparse(b: &mut Bench) {
    b.guarded("sparse", |b| {
        let ctx = b.ctx.clone();
        let mut mats: Vec<(&str, Csr)> = vec![("banded", banded(200_000, 2, 11)), ("powerlaw", power_law(100_000, 1 << 17, 31))];
        if !b.quick {
            mats.push(("banded", banded(2_000_000, 2, 12)));
            mats.push(("powerlaw", power_law(800_000, 1 << 17, 32)));
        }
        for (kind, a) in &mats {
            products(b, &ctx, kind, a);
        }
        // Conversions and constructors on the big power-law matrix.
        let (_, big) = mats.last().unwrap();
        conversions(b, &ctx, big);
        dense_forms(b, &ctx);
    });
}

fn products(b: &mut Bench, ctx: &luma::Context, kind: &str, a: &Csr) {
    let m = build(ctx, a);
    let size = size_label(a);
    let n = a.cols;
    let xh: Vec<f32> = {
        let mut g = Gen::new(12);
        (0..n).map(|_| g.quant()).collect()
    };
    let (xl, xa) = (lumax::arr(ctx, &xh, n), afx::arr(&xh, &[n as u64]));
    let want = a.product(&xh, 1, false);
    let want_t = a.product(&xh, 1, true);
    for transposed in [false, true] {
        let (name, want) = if transposed { (format!("sparse {kind} CSR A^T x (SpMV)"), &want_t) } else { (format!("sparse {kind} CSR A x (SpMV)"), &want) };
        let luma_run = |l: &Sparse, x: &L| if transposed { l.matmul_t(x) } else { l.matmul(x) };
        let lo = lumax::host(&luma_run(&m.l_csr, &xl));
        let v = compare_ref("luma", &lo, want, 1e-5, 1e-4);
        let af_run = |x: &A| af::matmul(&m.a_csr, x, if transposed { af::MatProp::TRANS } else { af::MatProp::NONE }, af::MatProp::NONE);
        let af_out = if af_ok(if transposed { "matmul_t" } else { "matmul" }) { attempt(|| afx::host(&af_run(&xa))) } else { None };
        match af_out {
            Some(ao) => {
                let v = v.and(compare_ref("AF", &ao, want, 1e-5, 1e-4));
                b.run(
                    &name,
                    &size,
                    v,
                    "",
                    &mut || {
                        luma_run(&m.l_csr, &xl).eval().unwrap();
                        ctx.sync().unwrap();
                    },
                    &mut || {
                        af_run(&xa).eval();
                        afx::sync();
                    },
                );
            }
            None => b.run_one(&name, &size, Some(&mut || {
                luma_run(&m.l_csr, &xl).eval().unwrap();
                ctx.sync().unwrap();
            }), None, "ArrayFire's sparse matmul crashes (segmentation fault in cusparseCreateSpMatDescr) or refuses the product: no AF row"),
        }
    }
    // The other luma formats have no ArrayFire product (matmul takes a CSR left operand only).
    for (tag, s) in [("COO", &m.l_coo), ("CSC", &m.l_csc)] {
        let name = format!("sparse {kind} {tag} A x (SpMV)");
        let lo = lumax::host(&s.matmul(&xl));
        let v = compare_ref("luma", &lo, &want, 1e-5, 1e-4);
        if !v.ok {
            println!("!!!!!! MISMATCH in {name} [{size}]: {}", v.text);
        }
        b.run_one(&name, &size, Some(&mut || {
            s.matmul(&xl).eval().unwrap();
            ctx.sync().unwrap();
        }), None, "ArrayFire matmul needs a CSR left operand");
    }

    // SpMM: 16 dense columns.
    let c = 16usize;
    let xh: Vec<f32> = {
        let mut g = Gen::new(13);
        (0..n * c).map(|_| g.quant()).collect()
    };
    let xt = transposed(&xh, n, c);
    let (xl, xa) = (lumax::arr(ctx, &xh, [c, n]), afx::arr(&xt, &[n as u64, c as u64]));
    let want = a.product(&xh, c, false);
    let lo = lumax::host(&m.l_csr.matmul(&xl));
    let v = compare_ref("luma", &lo, &want, 1e-5, 1e-4);
    let name = format!("sparse {kind} CSR A X (SpMM, {c} columns)");
    let af_run = |x: &A| af::matmul(&m.a_csr, x, af::MatProp::NONE, af::MatProp::NONE);
    match if af_ok("spmm") { attempt(|| afx::host(&af_run(&xa))) } else { None } {
        Some(ao) => {
            // ArrayFire's result is `(rows, c)` column-major: transpose it to luma's `[c, rows]`.
            let ao = transposed(&ao, c, a.rows);
            let ao = {
                // `transposed(x, n, c)` maps `[n, c]` row-major to `[c, n]`; here the input is `[c, rows]` -> use the inverse.
                let mut back = vec![0.0f32; ao.len()];
                for j in 0..c {
                    for r in 0..a.rows {
                        back[r * c + j] = ao[j * a.rows + r];
                    }
                }
                back
            };
            let v = v.and(compare_ref("AF", &ao, &want, 1e-5, 1e-4));
            b.run(
                &name,
                &size,
                v,
                "luma X is [c, n] (row-major), AF X is (n, c) column-major",
                &mut || {
                    m.l_csr.matmul(&xl).eval().unwrap();
                    ctx.sync().unwrap();
                },
                &mut || {
                    af_run(&xa).eval();
                    afx::sync();
                },
            );
        }
        None => b.run_one(&name, &size, Some(&mut || {
            m.l_csr.matmul(&xl).eval().unwrap();
            ctx.sync().unwrap();
        }), None, "ArrayFire's sparse matmul crashes (segmentation fault in cusparseCreateSpMatDescr) or refuses the product: no AF row"),
    }
}

/// The host arrays of an ArrayFire sparse array as `(values, row_idx, col_idx)`.
fn af_parts(s: &A) -> (Vec<f32>, Vec<i32>, Vec<i32>) {
    let (v, r, c, _) = af::sparse_get_info(s);
    (afx::host(&v), afx::host(&r), afx::host(&c))
}

fn l_parts(s: &Sparse) -> (Vec<f32>, Vec<i32>, Vec<i32>) {
    let to_i32 = |a: Option<LA<u32>>| a.map(|a| a.to_vec().unwrap().into_iter().map(|x| x as i32).collect()).unwrap_or_default();
    (s.values().map(|v| v.to_vec().unwrap()).unwrap_or_default(), to_i32(s.row_idx()), to_i32(s.col_idx()))
}

fn same_parts(l: (Vec<f32>, Vec<i32>, Vec<i32>), a: (Vec<f32>, Vec<i32>, Vec<i32>)) -> Verdict {
    let ok = l.0 == a.0 && l.1 == a.1 && l.2 == a.2;
    if ok { Verdict::yes("yes (identical arrays)") } else { Verdict::no(format!("NO: arrays differ (values {}, row {}, col {})", l.0 == a.0, l.1 == a.1, l.2 == a.2)) }
}

fn conversions(b: &mut Bench, ctx: &luma::Context, a: &Csr) {
    let size = size_label(a);
    let m = build(ctx, a);
    let row = a.row_of();
    let (rl, cl, vl) = (lumax::arr(ctx, &row, a.nnz()), lumax::arr(ctx, &a.col, a.nnz()), lumax::arr(ctx, &a.val, a.nnz()));
    let (ra, ca, va) = (afx::arr(&up_i32(&row), &[a.nnz() as u64]), afx::arr(&up_i32(&a.col), &[a.nnz() as u64]), afx::arr(&a.val, &[a.nnz() as u64]));

    // CSR -> CSC and CSR -> COO.
    for (name, lf, af_f) in [("CSR -> CSC", SparseFormat::Csc, af::SparseFormat::CSC), ("CSR -> COO", SparseFormat::Coo, af::SparseFormat::COO)] {
        let lo = m.l_csr.to_format(lf);
        settle(&lo);
        let probe_name = if lf == SparseFormat::Csc { "convert_csc" } else { "convert_coo" };
        let v = match if af_ok(probe_name) { attempt(|| af_parts(&af::sparse_convert_to(&m.a_csr, af_f))) } else { None } {
            Some(ap) => same_parts(l_parts(&lo), ap),
            None => Verdict::no("ArrayFire conversion failed"),
        };
        let name = format!("sparse convert {name}");
        match if af_ok(probe_name) { attempt(|| {
            af::sparse_convert_to(&m.a_csr, af_f).eval();
            afx::sync();
        }) } else { None } {
            Some(()) => b.run(
                &name,
                &size,
                v,
                "luma: from a fresh clone of the CSR matrix (a CSR -> COO expansion is not cached across clones)",
                &mut || {
                    let fresh = Sparse::from_csr(a.rows, a.cols, &m.l_csr.row_idx().unwrap(), &m.l_csr.col_idx().unwrap(), &m.l_csr.values().unwrap());
                    settle(&fresh.to_format(lf));
                    ctx.sync().unwrap();
                },
                &mut || {
                    af::sparse_convert_to(&m.a_csr, af_f).eval();
                    afx::sync();
                },
            ),
            None => b.run_one(&name, &size, Some(&mut || {
                settle(&m.l_csr.to_format(lf));
                ctx.sync().unwrap();
            }), None, "ArrayFire conversion refused: no AF row"),
        }
    }

    // COO -> CSR (construction from triplets).
    let lo = Sparse::from_triplets(a.rows, a.cols, &rl, &cl, &vl, SparseFormat::Csr);
    settle(&lo);
    let af_build = || {
        let coo = af::sparse(a.rows as u64, a.cols as u64, &va, &ra, &ca, af::SparseFormat::COO);
        af::sparse_convert_to(&coo, af::SparseFormat::CSR)
    };
    let af_ran = if af_ok("coo_to_csr") { attempt(|| af_parts(&af_build())) } else { None };
    let luma_build = &mut || {
        settle(&Sparse::from_triplets(a.rows, a.cols, &rl, &cl, &vl, SparseFormat::Csr));
        ctx.sync().unwrap();
    };
    match af_ran {
        Some(ap) => b.run(
            "sparse from triplets (COO -> CSR)",
            &size,
            same_parts(l_parts(&lo), ap),
            "luma: from_triplets into CSR; AF: sparse() COO then sparse_convert_to CSR",
            luma_build,
            &mut || {
                af_build().eval();
                afx::sync();
            },
        ),
        None => b.run_one("sparse from triplets (COO -> CSR)", &size, Some(luma_build), None, "ArrayFire COO -> CSR crashes or fails: no AF row"),
    }
}

fn dense_forms(b: &mut Bench, ctx: &luma::Context) {
    let n = 4096usize;
    // 2 % entries at random positions, multiples of 1/16.
    let mut g = Gen::new(71);
    let mut dh = vec![0.0f32; n * n];
    for _ in 0..n * n / 50 {
        let at = g.below(n * n);
        dh[at] = g.quant();
    }
    let held = dh.iter().filter(|&&v| v != 0.0).count();
    let size = format!("{n} x {n} dense, {} nnz", human(held));
    let (dl, da) = (lumax::arr(ctx, &dh, [n, n]), afx::arr(&dh, &[n as u64, n as u64]));
    let ls = Sparse::from_dense(&dl, SparseFormat::Csr);
    settle(&ls);
    let oh: Vec<f32> = {
        let mut g = Gen::new(72);
        (0..n * n).map(|_| g.quant()).collect()
    };
    let (ol, oa) = (lumax::arr(ctx, &oh, [n, n]), afx::arr(&oh, &[n as u64, n as u64]));

    // dense -> CSR. The dense array is `[cols, rows]` in both libraries; ArrayFire reads it
    // column-major, so its CSR matrix is the transpose of luma's (same work): the check compares
    // the multisets of stored values.
    let luma_from = &mut || {
        settle(&Sparse::from_dense(&dl, SparseFormat::Csr));
        ctx.sync().unwrap();
    };
    let name = "sparse from dense (dense -> CSR)";
    match if af_ok("from_dense") { attempt(|| af_parts(&af::sparse_from_dense(&da, af::SparseFormat::CSR)).0) } else { None } {
        Some(mut av) => {
            let mut lv = l_parts(&ls).0;
            lv.sort_by(f32::total_cmp);
            av.sort_by(f32::total_cmp);
            let v = if lv == av { Verdict::yes("yes (same multiset of values; AF holds the transpose)") } else { Verdict::no("NO: stored values differ") };
            b.run(name, &size, v, "AF reads the dense array column-major: its CSR matrix is the transpose of luma's", luma_from, &mut || {
                af::sparse_from_dense(&da, af::SparseFormat::CSR).eval();
                afx::sync();
            });
        }
        None => b.run_one(name, &size, Some(luma_from), None, "ArrayFire sparse(dense) crashes or fails: no AF row"),
    }

    // CSR -> dense.
    let name = "sparse to dense (CSR -> dense)";
    let luma_to = &mut || {
        ls.to_dense().eval().unwrap();
        ctx.sync().unwrap();
    };
    let v = exact(&lumax::host(&ls.to_dense()), &dh);
    let a_sparse = if af_ok("from_dense") { attempt(|| {
        let s = af::sparse_from_dense(&da, af::SparseFormat::CSR);
        s.eval();
        afx::sync();
        s
    }) } else { None };
    match a_sparse.as_ref().filter(|_| af_ok("to_dense")) {
        Some(sp) => {
            let v = v.and(match attempt(|| afx::host(&af::sparse_to_dense(sp))) {
                Some(ad) => exact(&ad, &dh),
                None => Verdict::no("ArrayFire dense() failed"),
            });
            b.run(name, &size, v, "", luma_to, &mut || {
                af::sparse_to_dense(sp).eval();
                afx::sync();
            });
        }
        None => b.run_one(name, &size, Some(luma_to), None, "ArrayFire dense() crashes or fails: no AF row"),
    }

    // Elementwise with a dense matrix: product (pattern kept) and sum (dense result).
    let want_mul: Vec<f32> = dh.iter().zip(&oh).map(|(a, b)| a * b).collect();
    let v = exact(&lumax::host(&ls.mul_dense(&ol).to_dense()), &want_mul);
    let luma_mul = &mut || {
        settle(&ls.mul_dense(&ol));
        ctx.sync().unwrap();
    };
    let af_mul = a_sparse.as_ref().filter(|_| af_ok("mul")).and_then(|sp| attempt(|| afx::host(&af::mul(sp, &oa, false))));
    match (af_mul, a_sparse.as_ref()) {
        (Some(ao), Some(sp)) if ao.len() == want_mul.len() => {
            let v = v.and(exact(&ao, &want_mul));
            b.run("sparse .* dense", &size, v, "luma: sparse result (pattern of A); AF: af::mul of a sparse and a dense array", luma_mul, &mut || {
                af::mul(sp, &oa, false).eval();
                afx::sync();
            });
        }
        _ => b.run_one("sparse .* dense", &size, Some(luma_mul), None, "ArrayFire sparse .* dense crashes, fails or has no such operation: no AF row"),
    }
    let want_add: Vec<f32> = dh.iter().zip(&oh).map(|(a, b)| a + b).collect();
    let v = exact(&lumax::host(&ls.add_dense(&ol)), &want_add);
    let luma_add = &mut || {
        ls.add_dense(&ol).eval().unwrap();
        ctx.sync().unwrap();
    };
    let af_add = a_sparse.as_ref().filter(|_| af_ok("add")).and_then(|sp| attempt(|| afx::host(&af::add(sp, &oa, false))));
    match (af_add, a_sparse.as_ref()) {
        (Some(ao), Some(sp)) if ao.len() == want_add.len() => {
            let v = v.and(exact(&ao, &want_add));
            b.run("sparse + dense", &size, v, "dense result", luma_add, &mut || {
                af::add(sp, &oa, false).eval();
                afx::sync();
            });
        }
        _ => b.run_one("sparse + dense", &size, Some(luma_add), None, "ArrayFire sparse + dense crashes, fails or has no such operation: no AF row"),
    }
}

/// ArrayFire 3.10.1's sparse paths can fail with a segmentation fault (the CUDA backend's
/// `cusparseCreateSpMatDescr` call inside `matmul` does on the machines this was measured on),
/// which no `catch_unwind` survives: every ArrayFire sparse operation is first tried on a small
/// matrix in a child process (`afcompare sparse-probe OP`), and the rows of the ones that crash
/// have no ArrayFire side.
fn af_ok(op: &str) -> bool {
    use std::sync::Mutex;
    static CACHE: Mutex<Vec<(String, bool)>> = Mutex::new(Vec::new());
    if let Some((_, ok)) = CACHE.lock().unwrap().iter().find(|(o, _)| o == op) {
        return *ok;
    }
    let out = std::process::Command::new(std::env::current_exe().unwrap()).args(["sparse-probe", op]).output().expect("cannot run the probe");
    let ok = out.status.success();
    println!("ArrayFire sparse probe `{op}`: {}", if ok { "works".to_string() } else { format!("FAILS ({})", out.status) });
    CACHE.lock().unwrap().push((op.to_string(), ok));
    ok
}

/// The child process of [`af_ok`]: one ArrayFire sparse operation on a small matrix.
pub fn probe(op: &str) {
    afx::init();
    let a = banded(2000, 2, 5);
    let (ap, ac, av) = (afx::arr(&up_i32(&a.ptr), &[a.rows as u64 + 1]), afx::arr(&up_i32(&a.col), &[a.nnz() as u64]), afx::arr(&a.val, &[a.nnz() as u64]));
    let csr = af::sparse(a.rows as u64, a.cols as u64, &av, &ap, &ac, af::SparseFormat::CSR);
    let x = afx::arr(&vec![1.0f32; a.cols], &[a.cols as u64]);
    let x4 = afx::arr(&vec![1.0f32; a.cols * 4], &[a.cols as u64, 4]);
    let dense = afx::arr(&vec![1.0f32; 64 * 64], &[64, 64]);
    let row = a.row_of();
    let (ra, ca) = (afx::arr(&up_i32(&row), &[a.nnz() as u64]), afx::arr(&up_i32(&a.col), &[a.nnz() as u64]));
    match op {
        "matmul" => af::matmul(&csr, &x, af::MatProp::NONE, af::MatProp::NONE).eval(),
        "matmul_t" => af::matmul(&csr, &x, af::MatProp::TRANS, af::MatProp::NONE).eval(),
        "spmm" => af::matmul(&csr, &x4, af::MatProp::NONE, af::MatProp::NONE).eval(),
        "convert_csc" => af::sparse_convert_to(&csr, af::SparseFormat::CSC).eval(),
        "convert_coo" => af::sparse_convert_to(&csr, af::SparseFormat::COO).eval(),
        "coo_to_csr" => af::sparse_convert_to(&af::sparse(a.rows as u64, a.cols as u64, &av, &ra, &ca, af::SparseFormat::COO), af::SparseFormat::CSR).eval(),
        "from_dense" => af::sparse_from_dense(&dense, af::SparseFormat::CSR).eval(),
        "to_dense" => af::sparse_to_dense(&csr).eval(),
        "mul" => af::mul(&csr, &afx::arr(&vec![2.0f32; a.rows * a.cols], &[a.rows as u64, a.cols as u64]), false).eval(),
        "add" => af::add(&csr, &afx::arr(&vec![2.0f32; a.rows * a.cols], &[a.rows as u64, a.cols as u64]), false).eval(),
        other => panic!("unknown probe {other}"),
    }
    afx::sync();
}
