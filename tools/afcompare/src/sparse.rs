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
use crate::cusparse::{self, AI};
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

/// Rows of `lo..=hi` entries at random columns.
pub fn uniform(rows: usize, lo: usize, hi: usize, seed: u64) -> Csr {
    finish(rows, rows, &mut Gen::new(seed), |_, g, out| {
        let d = lo + g.below(hi - lo + 1);
        out.extend((0..d).map(|_| g.below(rows) as u32));
    })
}

fn up_i32(v: &[u32]) -> Vec<i32> {
    v.iter().map(|&x| x as i32).collect()
}

/// A matrix on both sides: luma's three formats, and ArrayFire's CSR arrays.
struct Both {
    l_csr: Sparse,
    l_coo: Sparse,
    l_csc: Sparse,
    a_ptr: AI,
    a_col: AI,
    a_val: A,
}

impl Both {
    /// ArrayFire's sparse array over the CSR arrays (no copy). Creating it works; the operations
    /// on it are what may crash.
    fn af_csr(&self, a: &Csr) -> A {
        let s = af::sparse(a.rows as u64, a.cols as u64, &self.a_val, &self.a_ptr, &self.a_col, af::SparseFormat::CSR);
        s.eval();
        afx::sync();
        s
    }
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
    let (a_ptr, a_col, a_val) = (afx::arr(&up_i32(&a.ptr), &[a.rows as u64 + 1]), afx::arr(&up_i32(&a.col), &[a.nnz() as u64]), afx::arr(&a.val, &[a.nnz() as u64]));
    Both { l_csr, l_coo, l_csc, a_ptr, a_col, a_val }
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
        let cs = cusparse::Cusparse::open();
        match &cs {
            Ok(_) => println!("cuSPARSE reference: libcusparse.so.12 loaded"),
            Err(e) => println!("cuSPARSE reference unavailable: {e}"),
        }
        let cs = cs.ok();
        let mut mats: Vec<(&str, Csr)> = vec![("banded", banded(200_000, 2, 11)), ("powerlaw", power_law(100_000, 1 << 17, 31)), ("uniform", uniform(60_000, 8, 24, 34))];
        if !b.quick {
            mats.push(("banded", banded(2_000_000, 2, 12)));
            mats.push(("powerlaw", power_law(800_000, 1 << 17, 32)));
            mats.push(("uniform", uniform(600_000, 8, 24, 33)));
        }
        for (kind, a) in &mats {
            products(b, &ctx, cs.as_ref(), kind, a);
        }
        // Conversions and constructors on the big power-law matrix.
        let (_, big) = mats.iter().rev().find(|(k, _)| *k == "powerlaw").unwrap();
        conversions(b, &ctx, cs.as_ref(), big);
        dense_forms(b, &ctx);
    });
}

/// The reference side of a product row: ArrayFire's `matmul` when it survives its probe, else
/// cuSPARSE called directly (what `matmul` calls). Returns the closures that run it once
/// (synchronised) and read the result back as `(rows, c)` row-major (luma's layout), plus the
/// note for the row.
type Side<'a> = (Box<dyn FnMut() + 'a>, Box<dyn Fn() -> Vec<f32> + 'a>, &'static str);

#[allow(clippy::too_many_arguments)]
fn reference_product<'a>(
    m: &'a Both,
    a: &Csr,
    cs: Option<&'a cusparse::Cusparse>,
    xa: &'a A,
    c: usize,
    transposed_op: bool,
    af_sparse: Option<&'a A>,
) -> Option<Side<'a>> {
    let (n_out, n_in) = if transposed_op { (a.cols, a.rows) } else { (a.rows, a.cols) };
    let _ = n_in;
    // ArrayFire's own product.
    if let Some(sp) = af_sparse {
        let run = move || af::matmul(sp, xa, if transposed_op { af::MatProp::TRANS } else { af::MatProp::NONE }, af::MatProp::NONE);
        let read = move || {
            let out = afx::host(&run());
            from_col_major(&out, n_out, c)
        };
        return Some((
            Box::new(move || {
                run().eval();
                afx::sync();
            }),
            Box::new(read),
            "AF = ArrayFire matmul",
        ));
    }
    // cuSPARSE directly.
    let cs = cs?;
    let y: &'a A = Box::leak(Box::new(afx::arr(&vec![0.0f32; n_out * c], &[n_out as u64, c as u64])));
    let prod = cs.product(cs.csr(a.rows, a.cols, &m.a_ptr, &m.a_col, &m.a_val), xa, y, c, transposed_op);
    let prod: &'a cusparse::Product = Box::leak(Box::new(prod));
    let cs: &'a cusparse::Cusparse = cs;
    Some((
        Box::new(move || {
            cs.run(prod);
            cs.sync();
        }),
        Box::new(move || {
            cs.run(prod);
            cs.sync();
            from_col_major(&afx::host(y), n_out, c)
        }),
        "AF column = cuSPARSE SpMV/SpMM called directly (generic API, default algorithm; ArrayFire's own sparse matmul segfaults on this install)",
    ))
}

/// `(n, c)` column-major to `[c, n]` row-major (the layout of luma's dense operands).
fn from_col_major(v: &[f32], n: usize, c: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; v.len()];
    for j in 0..c {
        for r in 0..n {
            out[r * c + j] = v[j * n + r];
        }
    }
    out
}

fn products(b: &mut Bench, ctx: &luma::Context, cs: Option<&cusparse::Cusparse>, kind: &str, a: &Csr) {
    let m = build(ctx, a);
    let size = size_label(a);
    let af_sparse = if af_ok("matmul") && af_ok("matmul_t") && af_ok("spmm") { Some(m.af_csr(a)) } else { None };
    let n = a.cols;
    for (c, transposed_op) in [(1usize, false), (1, true), (16, false)] {
        let mut g = Gen::new(12 + c as u64);
        let n_in = if transposed_op { a.rows } else { a.cols };
        let xh: Vec<f32> = (0..n_in * c).map(|_| g.quant()).collect();
        let want = a.product(&xh, c, transposed_op);
        let name = match (c, transposed_op) {
            (1, false) => format!("sparse {kind} CSR A x (SpMV)"),
            (1, true) => format!("sparse {kind} CSR A^T x (SpMV)"),
            _ => format!("sparse {kind} CSR A X (SpMM, {c} columns)"),
        };
        // luma's operand is `[c, n]`, the reference's `(n, c)` column-major.
        let xl = if c == 1 { lumax::arr(ctx, &xh, n_in) } else { lumax::arr(ctx, &xh, [c, n_in]) };
        let xa = afx::arr(&transposed(&xh, n_in, c), &[n_in as u64, c as u64]);
        let luma_run = |x: &L| if transposed_op { m.l_csr.matmul_t(x) } else { m.l_csr.matmul(x) };
        let v = compare_ref("luma", &lumax::host(&luma_run(&xl)), &want, 1e-5, 1e-4);
        let luma_closure = &mut || {
            luma_run(&xl).eval().unwrap();
            ctx.sync().unwrap();
        };
        match reference_product(&m, a, cs, &xa, c, transposed_op, af_sparse.as_ref()) {
            Some((mut run, read, note)) => {
                let v = v.and(compare_ref("AF side", &read(), &want, 1e-5, 1e-4));
                b.run(&name, &size, v, note, luma_closure, &mut *run);
            }
            None => b.run_one(&name, &size, Some(luma_closure), None, "no reference: ArrayFire's sparse matmul crashes and cuSPARSE could not be loaded"),
        }
    }
    // The other luma formats have no ArrayFire product (matmul takes a CSR left operand only).
    let mut g = Gen::new(12);
    let xh: Vec<f32> = (0..n).map(|_| g.quant()).collect();
    let want = a.product(&xh, 1, false);
    let xl = lumax::arr(ctx, &xh, n);
    for (tag, s) in [("COO", &m.l_coo), ("CSC", &m.l_csc)] {
        let name = format!("sparse {kind} {tag} A x (SpMV)");
        let v = compare_ref("luma", &lumax::host(&s.matmul(&xl)), &want, 1e-5, 1e-4);
        if !v.ok {
            println!("!!!!!! MISMATCH in {name} [{size}]: {}", v.text);
        }
        b.run_one(
            &name,
            &size,
            Some(&mut || {
                s.matmul(&xl).eval().unwrap();
                ctx.sync().unwrap();
            }),
            None,
            "ArrayFire matmul needs a CSR left operand",
        );
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

fn conversions(b: &mut Bench, ctx: &luma::Context, cs: Option<&cusparse::Cusparse>, a: &Csr) {
    let size = size_label(a);
    let m = build(ctx, a);
    let row = a.row_of();
    let (rl, cl, vl) = (lumax::arr(ctx, &row, a.nnz()), lumax::arr(ctx, &a.col, a.nnz()), lumax::arr(ctx, &a.val, a.nnz()));
    let fresh = || Sparse::from_csr(a.rows, a.cols, &m.l_csr.row_idx().unwrap(), &m.l_csr.col_idx().unwrap(), &m.l_csr.values().unwrap());

    // CSR -> CSC. Reference: ArrayFire's `sparseConvertTo` if it survives its probe, else
    // `cusparseCsr2cscEx2` (what it calls). The arrays must be identical (a stable transposition
    // of sorted, distinct CSR).
    let luma_csc = &mut || {
        settle(&m.l_csr.to_format(SparseFormat::Csc));
        ctx.sync().unwrap();
    };
    let lcsc = l_parts(&m.l_csr.to_format(SparseFormat::Csc));
    if af_ok("convert_csc") {
        let sp = m.af_csr(a);
        let v = same_parts(lcsc, af_parts(&af::sparse_convert_to(&sp, af::SparseFormat::CSC)));
        b.run("sparse convert CSR -> CSC", &size, v, "AF sparseConvertTo", luma_csc, &mut || {
            af::sparse_convert_to(&sp, af::SparseFormat::CSC).eval();
            afx::sync();
        });
    } else if let Some(cs) = cs {
        let descr = cs.csr(a.rows, a.cols, &m.a_ptr, &m.a_col, &m.a_val);
        let out = (afx::arr(&vec![0.0f32; a.nnz()], &[a.nnz() as u64]), afx::arr(&vec![0i32; a.cols + 1], &[a.cols as u64 + 1]), afx::arr(&vec![0i32; a.nnz()], &[a.nnz() as u64]));
        let buffer = cs.csr_to_csc_setup(&descr, &m.a_ptr, &m.a_col, &m.a_val, &out);
        cs.csr_to_csc(&descr, &m.a_ptr, &m.a_col, &m.a_val, &out, &buffer);
        cs.sync();
        // luma: (values, row indices, column pointer) as `l_parts` lists them for CSC = (values, row_idx = rows, col_idx = pointer).
        let got = (afx::host(&out.0), afx::host(&out.2), afx::host(&out.1));
        let v = same_parts(lcsc, got);
        b.run(
            "sparse convert CSR -> CSC",
            &size,
            v,
            "AF column = cusparseCsr2cscEx2 called directly (ArrayFire's sparseConvertTo fails on this install)",
            luma_csc,
            &mut || {
                cs.csr_to_csc(&descr, &m.a_ptr, &m.a_col, &m.a_val, &out, &buffer);
                cs.sync();
            },
        );
    } else {
        b.run_one("sparse convert CSR -> CSC", &size, Some(luma_csc), None, "no reference available");
    }

    // CSR -> COO: luma expands the pointer (a fresh matrix, as the expansion is cached per matrix).
    b.run_one(
        "sparse convert CSR -> COO",
        &size,
        Some(&mut || {
            settle(&fresh().to_format(SparseFormat::Coo));
            ctx.sync().unwrap();
        }),
        None,
        "ArrayFire sparseConvertTo CSR -> COO segfaults on this install; no reference measured",
    );

    // COO -> CSR (construction from triplets): a stable sort by row, no reference (ArrayFire's
    // `sparse(...)` COO conversion segfaults).
    let lo = Sparse::from_triplets(a.rows, a.cols, &rl, &cl, &vl, SparseFormat::Csr);
    let (lv, lp, lc) = l_parts(&lo);
    let v = if lp == up_i32(&a.ptr) && lc == up_i32(&a.col) && lv == a.val { Verdict::yes("yes (the host CSR)") } else { Verdict::no("NO: differs from the host CSR") };
    if !v.ok {
        println!("!!!!!! MISMATCH in sparse from triplets: {}", v.text);
    }
    b.run_one(
        "sparse from triplets (COO -> CSR)",
        &size,
        Some(&mut || {
            settle(&Sparse::from_triplets(a.rows, a.cols, &rl, &cl, &vl, SparseFormat::Csr));
            ctx.sync().unwrap();
        }),
        None,
        "ArrayFire's COO -> CSR conversion segfaults on this install; no reference measured",
    );
}

fn same_parts(l: (Vec<f32>, Vec<i32>, Vec<i32>), a: (Vec<f32>, Vec<i32>, Vec<i32>)) -> Verdict {
    let ok = l.0 == a.0 && l.1 == a.1 && l.2 == a.2;
    if ok { Verdict::yes("yes (identical arrays)") } else { Verdict::no(format!("NO: arrays differ (values {}, row {}, col {})", l.0 == a.0, l.1 == a.1, l.2 == a.2)) }
}

fn dense_forms(b: &mut Bench, ctx: &luma::Context) {
    let n = 4096usize;
    // 2 % entries at random positions, multiples of 1/16; a square matrix, so that ArrayFire's
    // column-major view of the same memory is the transpose.
    let mut g = Gen::new(71);
    let mut dh = vec![0.0f32; n * n];
    for _ in 0..n * n / 50 {
        let at = g.below(n * n);
        dh[at] = g.quant();
    }
    let held = dh.iter().filter(|&&v| v != 0.0).count();
    let size = format!("{n} x {n} dense, {} nnz", human(held));
    let dl = lumax::arr(ctx, &dh, [n, n]);
    let ls = Sparse::from_dense(&dl, SparseFormat::Csr);
    settle(&ls);
    let oh: Vec<f32> = {
        let mut g = Gen::new(72);
        (0..n * n).map(|_| g.quant()).collect()
    };
    let ol = lumax::arr(ctx, &oh, [n, n]);
    let flip = |v: &[f32]| transposed(v, n, n);

    // dense -> CSR and CSR -> dense: ArrayFire's `sparse(dense)` and `dense(sparse)` crash (they
    // call cusparse's dense conversions): luma only.
    let name = "sparse from dense (dense -> CSR)";
    let v = exact(&lumax::host(&ls.to_dense()), &dh);
    if !v.ok {
        println!("!!!!!! MISMATCH in {name}: {}", v.text);
    }
    b.run_one(
        name,
        &size,
        Some(&mut || {
            settle(&Sparse::from_dense(&dl, SparseFormat::Csr));
            ctx.sync().unwrap();
        }),
        None,
        "ArrayFire sparse(dense) segfaults on this install; no reference measured",
    );
    b.run_one(
        "sparse to dense (CSR -> dense)",
        &size,
        Some(&mut || {
            ls.to_dense().eval().unwrap();
            ctx.sync().unwrap();
        }),
        None,
        "ArrayFire dense(sparse) segfaults on this install; no reference measured",
    );

    // Elementwise with a dense matrix, against ArrayFire's operators on a sparse array made from
    // the CSR arrays of the same matrix (that works): the product (pattern kept) and the sum.
    let want_mul: Vec<f32> = dh.iter().zip(&oh).map(|(a, b)| a * b).collect();
    let want_add: Vec<f32> = dh.iter().zip(&oh).map(|(a, b)| a + b).collect();
    let luma_mul = &mut || {
        settle(&ls.mul_dense(&ol));
        ctx.sync().unwrap();
    };
    let luma_add = &mut || {
        ls.add_dense(&ol).eval().unwrap();
        ctx.sync().unwrap();
    };
    let v_mul = exact(&lumax::host(&ls.mul_dense(&ol).to_dense()), &want_mul);
    let v_add = exact(&lumax::host(&ls.add_dense(&ol)), &want_add);
    let a_sparse = (af_ok("mul") && af_ok("add")).then(|| {
        let (mut ptr, mut col, mut val) = (vec![0i32], Vec::new(), Vec::new());
        // ArrayFire reads `dh` column-major, so it holds the transpose: build the CSR of the
        // matrix with ArrayFire's own (row, column) meaning from the flipped data.
        let dt = flip(&dh);
        for r in 0..n {
            for c in 0..n {
                if dt[r * n + c] != 0.0 {
                    col.push(c as i32);
                    val.push(dt[r * n + c]);
                }
            }
            ptr.push(col.len() as i32);
        }
        let s = af::sparse(n as u64, n as u64, &afx::arr(&val, &[val.len() as u64]), &afx::arr(&ptr, &[ptr.len() as u64]), &afx::arr(&col, &[col.len() as u64]), af::SparseFormat::CSR);
        s.eval();
        afx::sync();
        s
    });
    // ArrayFire's `(n, n)` column-major dense operand holding `oh` read as the matrix of luma:
    // element (r, c) of luma's operand is at r + n * c, i.e. the flipped memory.
    let oa = afx::arr(&flip(&oh), &[n as u64, n as u64]);
    match &a_sparse {
        Some(sp) => {
            let got = attempt(|| afx::host(&af::mul(sp, &oa, false)));
            let v = v_mul.and(match got {
                Some(g) => exact(&flip(&g), &want_mul),
                None => Verdict::no("af::mul failed"),
            });
            b.run("sparse .* dense", &size, v, "AF: af::mul(sparse, dense), dense result; luma: sparse result (pattern of A)", luma_mul, &mut || {
                af::mul(sp, &oa, false).eval();
                afx::sync();
            });
            let got = attempt(|| afx::host(&af::add(sp, &oa, false)));
            let v = v_add.and(match got {
                Some(g) => exact(&flip(&g), &want_add),
                None => Verdict::no("af::add failed"),
            });
            b.run("sparse + dense", &size, v, "dense result", luma_add, &mut || {
                af::add(sp, &oa, false).eval();
                afx::sync();
            });
        }
        None => {
            b.run_one("sparse .* dense", &size, Some(luma_mul), None, "ArrayFire sparse .* dense unavailable: no AF row");
            b.run_one("sparse + dense", &size, Some(luma_add), None, "ArrayFire sparse + dense unavailable: no AF row");
        }
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
        "mul" | "add" => {
            let d = afx::arr(&vec![2.0f32; a.rows * a.cols], &[a.rows as u64, a.cols as u64]);
            let r = if op == "mul" { af::mul(&csr, &d, false) } else { af::add(&csr, &d, false) };
            // The result must be a dense array that can be read back.
            assert!(af::sparse_get_format(&r) == af::SparseFormat::DENSE, "the result is sparse");
            assert_eq!(afx::host(&r).len(), a.rows * a.cols);
        }
        other => panic!("unknown probe {other}"),
    }
    afx::sync();
}
