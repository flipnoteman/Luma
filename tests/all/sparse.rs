//! `Sparse<f32>` (CSR, CSC, COO) against dense host references: construction from dense, triplets
//! and compressed arrays, conversions between the formats, the transpose, products with vectors
//! and dense matrices (plain and transposed, every format), elementwise operations with dense
//! operands and scalars, empty matrices, duplicates, NaN / inf, and errors. `f32` is the only
//! element type of `Sparse`.
//!
//! Values and dense operands are multiples of 1/16 that keep every sum exact in `f32`, so the
//! results are compared for equality (with an `f64` reference).

use luma::{sparse, Array, LumaError, Sparse, SparseFormat};

const FORMATS: [SparseFormat; 3] = [SparseFormat::Csr, SparseFormat::Csc, SparseFormat::Coo];

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }

    fn below(&mut self, n: usize) -> usize {
        self.next() as usize % n
    }

    /// A multiple of 1/16 in `[-2, 2]`, never 0.
    fn quant(&mut self) -> f32 {
        let v = (self.next() % 65) as i32 - 32;
        if v == 0 { 1.0 / 16.0 } else { v as f32 / 16.0 }
    }
}

/// A host matrix as triplets (duplicates allowed) and as the dense sum.
struct Host {
    rows: usize,
    cols: usize,
    r: Vec<u32>,
    c: Vec<u32>,
    v: Vec<f32>,
}

impl Host {
    /// `nnz` random entries (duplicates possible when `dups`; otherwise distinct positions).
    fn random(rng: &mut Rng, rows: usize, cols: usize, nnz: usize, dups: bool) -> Host {
        let mut h = Host { rows, cols, r: vec![], c: vec![], v: vec![] };
        let mut taken = std::collections::HashSet::new();
        let nnz = if dups { nnz } else { nnz.min(rows * cols) };
        while h.v.len() < nnz {
            let (r, c) = (rng.below(rows), rng.below(cols));
            if !dups && !taken.insert((r, c)) {
                continue;
            }
            h.r.push(r as u32);
            h.c.push(c as u32);
            h.v.push(rng.quant());
        }
        h
    }

    fn dense(&self) -> Vec<f32> {
        let mut d = vec![0.0f32; self.rows * self.cols];
        for k in 0..self.v.len() {
            d[self.r[k] as usize * self.cols + self.c[k] as usize] += self.v[k];
        }
        d
    }

    fn arrays(&self) -> (Array<u32>, Array<u32>, Array<f32>) {
        (Array::from_slice(&self.r, self.r.len()).unwrap(), Array::from_slice(&self.c, self.c.len()).unwrap(), Array::from_slice(&self.v, self.v.len()).unwrap())
    }

    fn sparse(&self, format: SparseFormat) -> Sparse {
        let (r, c, v) = self.arrays();
        Sparse::from_triplets(self.rows, self.cols, &r, &c, &v, format)
    }
}

fn dense_array(d: &[f32], rows: usize, cols: usize) -> Array<f32> {
    Array::from_slice(d, [cols, rows]).unwrap()
}

/// `rows x cols` dense `a` times the `cols x c` row-major `x`, in `f64`.
fn host_matmul(a: &[f32], rows: usize, cols: usize, x: &[f32], c: usize) -> Vec<f32> {
    let mut y = vec![0.0f64; rows * c];
    for r in 0..rows {
        for k in 0..cols {
            let v = f64::from(a[r * cols + k]);
            if v != 0.0 {
                for j in 0..c {
                    y[r * c + j] += v * f64::from(x[k * c + j]);
                }
            }
        }
    }
    y.into_iter().map(|v| v as f32).collect()
}

fn transpose(a: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut t = vec![0.0f32; a.len()];
    for r in 0..rows {
        for c in 0..cols {
            t[c * rows + r] = a[r * cols + c];
        }
    }
    t
}

fn quantized(rng: &mut Rng, n: usize) -> Vec<f32> {
    (0..n).map(|_| (rng.next() % 65) as f32 / 16.0 - 2.0).collect()
}

fn dense_of(s: &Sparse) -> Vec<f32> {
    s.to_dense().to_vec().unwrap()
}

/// Sizes that straddle workgroup and tile edges, with vectors and single entries.
const SIZES: [(usize, usize, usize); 9] = [(1, 1, 1), (1, 9, 4), (9, 1, 5), (7, 13, 30), (33, 64, 200), (130, 257, 3000), (1000, 1000, 5000), (64, 64, 4096), (300, 50, 2000)];

#[test]
fn from_dense_every_format() {
    let mut rng = Rng(1);
    for (rows, cols, nnz) in SIZES {
        let h = Host::random(&mut rng, rows, cols, nnz, false);
        let d = h.dense();
        let a = dense_array(&d, rows, cols);
        for format in FORMATS {
            let s = sparse(&a, format);
            assert!(s.error().is_none(), "{format:?} {rows}x{cols}: {:?}", s.error());
            assert_eq!((s.rows(), s.cols(), s.nnz(), s.format()), (rows, cols, h.v.len(), format));
            assert_eq!(dense_of(&s), d, "{format:?} {rows}x{cols} to_dense");
            // The host layout the arrays must have.
            let mut entries: Vec<(usize, usize, f32)> = (0..h.v.len()).map(|k| (h.r[k] as usize, h.c[k] as usize, h.v[k])).collect();
            let info = s.info();
            let (want_r, want_c, want_v): (Vec<u32>, Vec<u32>, Vec<f32>) = match format {
                SparseFormat::Csc => {
                    entries.sort_by_key(|&(r, c, _)| (c, r));
                    (entries.iter().map(|e| e.0 as u32).collect(), entries.iter().map(|e| e.1 as u32).collect(), entries.iter().map(|e| e.2).collect())
                }
                _ => {
                    entries.sort_by_key(|&(r, c, _)| (r, c));
                    (entries.iter().map(|e| e.0 as u32).collect(), entries.iter().map(|e| e.1 as u32).collect(), entries.iter().map(|e| e.2).collect())
                }
            };
            assert_eq!(info.values.unwrap().to_vec().unwrap(), want_v, "{format:?} values");
            let ptr_of = |keys: &[u32], n: usize| {
                let mut p = vec![0u32; n + 1];
                keys.iter().for_each(|&k| p[k as usize + 1] += 1);
                for i in 0..n {
                    p[i + 1] += p[i];
                }
                p
            };
            match format {
                SparseFormat::Csr => {
                    assert_eq!(info.row_idx.unwrap().to_vec().unwrap(), ptr_of(&want_r, rows));
                    assert_eq!(info.col_idx.unwrap().to_vec().unwrap(), want_c);
                }
                SparseFormat::Csc => {
                    assert_eq!(info.col_idx.unwrap().to_vec().unwrap(), ptr_of(&want_c, cols));
                    assert_eq!(info.row_idx.unwrap().to_vec().unwrap(), want_r);
                }
                SparseFormat::Coo => {
                    assert_eq!(info.row_idx.unwrap().to_vec().unwrap(), want_r);
                    assert_eq!(info.col_idx.unwrap().to_vec().unwrap(), want_c);
                }
            }
        }
    }
}

#[test]
fn from_dense_special_values() {
    // NaN and inf are entries, zero and -0.0 are not.
    let d = [0.0f32, f32::NAN, -0.0, f32::INFINITY, 0.0, f32::NEG_INFINITY, 2.0, 0.0, 0.0];
    let a = dense_array(&d, 3, 3);
    for format in FORMATS {
        let s = sparse(&a, format);
        assert_eq!(s.nnz(), 4, "{format:?}");
        let back = dense_of(&s);
        for (g, w) in back.iter().zip(&d) {
            assert!(g.to_bits() == w.to_bits() || (g.is_nan() && w.is_nan()) || (*g == 0.0 && *w == 0.0), "{format:?}: {g} vs {w}");
        }
    }
    // Vectors: shape [n] is a 1 x n row, [1, n] an n x 1 column.
    let row = Array::from_slice(&[0.0f32, 3.0, 0.0, 4.0], 4).unwrap();
    let s = sparse(&row, SparseFormat::Csr);
    assert_eq!((s.rows(), s.cols(), s.nnz()), (1, 4, 2));
    let col = Array::from_slice(&[0.0f32, 3.0, 0.0, 4.0], [1, 4]).unwrap();
    let s = sparse(&col, SparseFormat::Csc);
    assert_eq!((s.rows(), s.cols(), s.nnz()), (4, 1, 2));
    assert_eq!(s.to_dense().to_vec().unwrap(), [0.0, 3.0, 0.0, 4.0]);
}

#[test]
fn all_zero_matrix_is_empty() {
    let a = Array::<f32>::zeros([5, 3]).unwrap();
    for format in FORMATS {
        let s = sparse(&a, format);
        assert!(s.error().is_none());
        assert_eq!((s.rows(), s.cols(), s.nnz()), (3, 5, 0));
        assert!(s.values().is_none());
        assert_eq!(dense_of(&s), vec![0.0; 15]);
        let info = s.info();
        assert!(info.values.is_none());
        match format {
            SparseFormat::Csr => {
                assert_eq!(info.row_idx.unwrap().to_vec().unwrap(), [0, 0, 0, 0]);
                assert!(info.col_idx.is_none());
            }
            SparseFormat::Csc => {
                assert_eq!(info.col_idx.unwrap().to_vec().unwrap(), [0; 6]);
                assert!(info.row_idx.is_none());
            }
            SparseFormat::Coo => assert!(info.row_idx.is_none() && info.col_idx.is_none()),
        }
        // Products are zero, conversions and transposes stay empty, elementwise keeps the pattern.
        let x = Array::from_slice(&[f32::NAN, 1.0, f32::INFINITY, 2.0, 3.0], 5).unwrap();
        assert_eq!(s.matmul(&x).to_vec().unwrap(), [0.0; 3]);
        let xm = Array::from_slice(&[1.0f32; 10], [2, 5]).unwrap();
        assert_eq!(s.matmul(&xm).shape().dims(), [2, 3, 1, 1]);
        assert_eq!(s.matmul(&xm).to_vec().unwrap(), [0.0; 6]);
        let y = Array::from_slice(&[1.0f32; 3], 3).unwrap();
        assert_eq!(s.matmul_t(&y).to_vec().unwrap(), [0.0; 5]);
        for to in FORMATS {
            let t = s.to_format(to);
            assert_eq!((t.format(), t.nnz(), t.error().is_none()), (to, 0, true));
            assert_eq!(dense_of(&t), vec![0.0; 15]);
        }
        let t = s.transpose();
        assert_eq!((t.rows(), t.cols(), t.nnz()), (5, 3, 0));
        assert_eq!(dense_of(&t), vec![0.0; 15]);
        let m = s.mul_dense(&Array::from_slice(&[7.0f32; 15], [5, 3]).unwrap());
        assert_eq!(m.nnz(), 0);
        assert_eq!(s.add_dense(&Array::from_slice(&[7.0f32; 15], [5, 3]).unwrap()).to_vec().unwrap(), vec![7.0; 15]);
        assert_eq!(s.mul_scalar(3.0).nnz(), 0);
        assert_eq!(s.add_scalar(2.0).to_vec().unwrap(), vec![2.0; 15]);
    }
    for format in FORMATS {
        let s = Sparse::empty(4, 6, format).unwrap();
        assert_eq!((s.rows(), s.cols(), s.nnz(), s.format()), (4, 6, 0, format));
        assert_eq!(dense_of(&s), vec![0.0; 24]);
    }
    assert!(Sparse::empty(0, 3, SparseFormat::Csr).is_err());
}

#[test]
fn from_triplets_with_duplicates_and_any_order() {
    let mut rng = Rng(2);
    for (rows, cols, nnz) in SIZES {
        // More entries than positions in the small cases: many duplicates.
        let h = Host::random(&mut rng, rows, cols, nnz.max(rows * cols), true);
        let d = h.dense();
        for format in FORMATS {
            let s = h.sparse(format);
            assert!(s.error().is_none());
            assert_eq!((s.rows(), s.cols(), s.nnz()), (rows, cols, h.v.len()));
            assert_eq!(dense_of(&s), d, "{format:?} {rows}x{cols}");
            // The products see the duplicates as one summed entry.
            let x = quantized(&mut rng, cols);
            let want = host_matmul(&d, rows, cols, &x, 1);
            assert_eq!(s.matmul(&Array::from_slice(&x, cols).unwrap()).to_vec().unwrap(), want, "{format:?} spmv");
        }
    }
}

#[test]
fn csr_and_csc_sorted_by_construction() {
    let mut rng = Rng(3);
    let h = Host::random(&mut rng, 40, 50, 600, true);
    let csr = h.sparse(SparseFormat::Csr);
    let ptr = csr.row_idx().unwrap().to_vec().unwrap();
    let col = csr.col_idx().unwrap().to_vec().unwrap();
    let val = csr.values().unwrap().to_vec().unwrap();
    assert_eq!(ptr.len(), 41);
    assert_eq!(*ptr.last().unwrap() as usize, 600);
    // Stable by row: the entries of every row keep their input order.
    for r in 0..40 {
        let want: Vec<(u32, f32)> = (0..600).filter(|&k| h.r[k] as usize == r).map(|k| (h.c[k], h.v[k])).collect();
        let got: Vec<(u32, f32)> = (ptr[r] as usize..ptr[r + 1] as usize).map(|k| (col[k], val[k])).collect();
        assert_eq!(got, want, "row {r}");
    }
    let csc = h.sparse(SparseFormat::Csc);
    let ptr = csc.col_idx().unwrap().to_vec().unwrap();
    let row = csc.row_idx().unwrap().to_vec().unwrap();
    for c in 0..50 {
        let want: Vec<u32> = (0..600).filter(|&k| h.c[k] as usize == c).map(|k| h.r[k]).collect();
        assert_eq!(&row[ptr[c] as usize..ptr[c + 1] as usize], &want[..], "column {c}");
    }
}

#[test]
fn from_csr_and_from_csc_arrays() {
    // [[1, 0, 2], [0, 0, 0], [0, 3, 0], [4, 5, 6]]
    let ptr = Array::from_slice(&[0u32, 2, 2, 3, 6], 5).unwrap();
    let col = Array::from_slice(&[0u32, 2, 1, 0, 1, 2], 6).unwrap();
    let val = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], 6).unwrap();
    let want = [1.0f32, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 4.0, 5.0, 6.0];
    let a = Sparse::from_csr(4, 3, &ptr, &col, &val);
    assert_eq!((a.rows(), a.cols(), a.nnz(), a.format()), (4, 3, 6, SparseFormat::Csr));
    assert_eq!(dense_of(&a), want);
    let x = Array::from_slice(&[1.0f32, 10.0, 100.0], 3).unwrap();
    assert_eq!(a.matmul(&x).to_vec().unwrap(), [201.0, 0.0, 30.0, 654.0]);
    let y = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], 4).unwrap();
    assert_eq!(a.matmul_t(&y).to_vec().unwrap(), [17.0, 29.0, 26.0]);
    // The same matrix by columns.
    let cptr = Array::from_slice(&[0u32, 2, 4, 6], 4).unwrap();
    let rows = Array::from_slice(&[0u32, 3, 2, 3, 0, 3], 6).unwrap();
    let cval = Array::from_slice(&[1.0f32, 4.0, 3.0, 5.0, 2.0, 6.0], 6).unwrap();
    let b = Sparse::from_csc(4, 3, &cptr, &rows, &cval);
    assert_eq!(dense_of(&b), want);
    assert_eq!(b.matmul(&x).to_vec().unwrap(), [201.0, 0.0, 30.0, 654.0]);
    assert_eq!(b.matmul_t(&y).to_vec().unwrap(), [17.0, 29.0, 26.0]);
    // The arrays come back as they went in.
    assert_eq!(a.row_idx().unwrap().to_vec().unwrap(), [0, 2, 2, 3, 6]);
    assert_eq!(a.col_idx().unwrap().to_vec().unwrap(), [0, 2, 1, 0, 1, 2]);
    assert_eq!(a.values().unwrap().to_vec().unwrap(), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    assert_eq!(b.col_idx().unwrap().to_vec().unwrap(), [0, 2, 4, 6]);
    assert_eq!(b.row_idx().unwrap().to_vec().unwrap(), [0, 3, 2, 3, 0, 3]);
}

#[test]
fn conversions_between_all_formats() {
    let mut rng = Rng(4);
    for (rows, cols, nnz) in SIZES {
        let h = Host::random(&mut rng, rows, cols, nnz, true);
        let d = h.dense();
        for from in FORMATS {
            let s = h.sparse(from);
            for to in FORMATS {
                let t = s.to_format(to);
                assert!(t.error().is_none());
                assert_eq!((t.format(), t.nnz(), t.rows(), t.cols()), (to, s.nnz(), rows, cols));
                assert_eq!(dense_of(&t), d, "{from:?} -> {to:?} {rows}x{cols}");
                // And back: a round trip keeps the matrix.
                assert_eq!(dense_of(&t.to_format(from)), d, "{from:?} -> {to:?} -> {from:?}");
            }
        }
    }
}

#[test]
fn csr_csc_conversion_keeps_order_within_groups() {
    let mut rng = Rng(5);
    // No duplicates: sorted by (row, column) after dense, and CSR -> CSC -> CSR is the identity.
    let h = Host::random(&mut rng, 70, 90, 800, false);
    let d = dense_array(&h.dense(), 70, 90);
    let csr = sparse(&d, SparseFormat::Csr);
    let csc = csr.to_format(SparseFormat::Csc);
    let from_dense_csc = sparse(&d, SparseFormat::Csc);
    for (a, b) in [(csc.row_idx(), from_dense_csc.row_idx()), (csc.col_idx(), from_dense_csc.col_idx())] {
        assert_eq!(a.unwrap().to_vec().unwrap(), b.unwrap().to_vec().unwrap());
    }
    assert_eq!(csc.values().unwrap().to_vec().unwrap(), from_dense_csc.values().unwrap().to_vec().unwrap());
    let back = csc.to_format(SparseFormat::Csr);
    assert_eq!(back.row_idx().unwrap().to_vec().unwrap(), csr.row_idx().unwrap().to_vec().unwrap());
    assert_eq!(back.col_idx().unwrap().to_vec().unwrap(), csr.col_idx().unwrap().to_vec().unwrap());
    assert_eq!(back.values().unwrap().to_vec().unwrap(), csr.values().unwrap().to_vec().unwrap());
    // Same format is a clone.
    assert_eq!(csr.to_format(SparseFormat::Csr).nnz(), csr.nnz());
}

#[test]
fn transpose_every_format() {
    let mut rng = Rng(6);
    for (rows, cols, nnz) in SIZES {
        let h = Host::random(&mut rng, rows, cols, nnz, true);
        let want = transpose(&h.dense(), rows, cols);
        for format in FORMATS {
            let t = h.sparse(format).transpose();
            assert_eq!((t.rows(), t.cols(), t.nnz()), (cols, rows, h.v.len()));
            assert_eq!(dense_of(&t), want, "{format:?} {rows}x{cols}");
            // The transpose of a CSR matrix is CSC and vice versa; COO stays COO.
            let expect = match format {
                SparseFormat::Csr => SparseFormat::Csc,
                SparseFormat::Csc => SparseFormat::Csr,
                SparseFormat::Coo => SparseFormat::Coo,
            };
            assert_eq!(t.format(), expect);
            assert_eq!(dense_of(&t.transpose()), h.dense());
        }
    }
}

#[test]
fn products_with_vectors_and_matrices() {
    let mut rng = Rng(7);
    for (rows, cols, nnz) in SIZES {
        let h = Host::random(&mut rng, rows, cols, nnz, rows * cols < nnz);
        let d = h.dense();
        let dt = transpose(&d, rows, cols);
        for format in FORMATS {
            let s = h.sparse(format);
            for c in [1usize, 2, 3, 5, 16] {
                let x = quantized(&mut rng, cols * c);
                let xt = quantized(&mut rng, rows * c);
                let (xa, xta) = if c == 1 {
                    (Array::from_slice(&x, cols).unwrap(), Array::from_slice(&xt, rows).unwrap())
                } else {
                    (Array::from_slice(&x, [c, cols]).unwrap(), Array::from_slice(&xt, [c, rows]).unwrap())
                };
                let y = s.matmul(&xa);
                assert_eq!(y.shape().dims(), if c == 1 { [rows, 1, 1, 1] } else { [c, rows, 1, 1] });
                assert_eq!(y.to_vec().unwrap(), host_matmul(&d, rows, cols, &x, c), "{format:?} {rows}x{cols} c={c}");
                let yt = s.matmul_t(&xta);
                assert_eq!(yt.shape().dims(), if c == 1 { [cols, 1, 1, 1] } else { [c, cols, 1, 1] });
                assert_eq!(yt.to_vec().unwrap(), host_matmul(&dt, cols, rows, &xt, c), "{format:?} {rows}x{cols} c={c} transposed");
            }
        }
    }
}

#[test]
fn products_with_a_column_matrix_of_one_column() {
    // `[1, cols]` is a one-column matrix: the result keeps the matrix shape `[1, rows]`.
    let mut rng = Rng(8);
    let h = Host::random(&mut rng, 20, 30, 100, false);
    let x = quantized(&mut rng, 30);
    let want = host_matmul(&h.dense(), 20, 30, &x, 1);
    for format in FORMATS {
        let y = h.sparse(format).matmul(&Array::from_slice(&x, [1, 30]).unwrap());
        assert_eq!(y.shape().dims(), [1, 20, 1, 1]);
        assert_eq!(y.to_vec().unwrap(), want);
    }
}

#[test]
fn products_with_skewed_rows_and_empty_rows() {
    let mut rng = Rng(9);
    // A power-law matrix: one row holds 5000 entries (many tiles), many rows are empty, the
    // first and the last among them.
    let (rows, cols) = (3000usize, 6000usize);
    let mut h = Host { rows, cols, r: vec![], c: vec![], v: vec![] };
    for r in 1..rows - 1 {
        let len = if r == 17 {
            5000
        } else if r % 3 == 0 {
            0
        } else {
            1 + rng.below(12)
        };
        let mut seen = std::collections::HashSet::new();
        while seen.len() < len {
            let c = rng.below(cols);
            if seen.insert(c) {
                h.r.push(r as u32);
                h.c.push(c as u32);
                h.v.push(rng.quant());
            }
        }
    }
    let d = h.dense();
    let x = quantized(&mut rng, cols);
    let y = quantized(&mut rng, rows);
    for format in FORMATS {
        let s = h.sparse(format);
        assert_eq!(s.matmul(&Array::from_slice(&x, cols).unwrap()).to_vec().unwrap(), host_matmul(&d, rows, cols, &x, 1), "{format:?}");
        assert_eq!(s.matmul_t(&Array::from_slice(&y, rows).unwrap()).to_vec().unwrap(), host_matmul(&transpose(&d, rows, cols), cols, rows, &y, 1), "{format:?} t");
    }
}

#[test]
fn products_do_not_read_unstored_columns() {
    // x has NaN and inf where the matrix has no entries: they must not reach the result.
    let a = Array::from_slice(&[2.0f32, 0.0, 0.0, 0.0, 0.0, 3.0], [3, 2]).unwrap();
    let x = Array::from_slice(&[10.0f32, f32::NAN, 100.0], 3).unwrap();
    for format in FORMATS {
        assert_eq!(sparse(&a, format).matmul(&x).to_vec().unwrap(), [20.0, 300.0], "{format:?}");
    }
    let y = Array::from_slice(&[f32::INFINITY, 1.0], 2).unwrap();
    for format in FORMATS {
        let t = sparse(&a, format).matmul_t(&y).to_vec().unwrap();
        assert_eq!((t[0], t[1], t[2]), (f32::INFINITY, 0.0, 3.0), "{format:?}");
    }
    // NaN in a stored position does propagate, along its row only.
    let x = Array::from_slice(&[f32::NAN, 1.0, 1.0], 3).unwrap();
    for format in FORMATS {
        let r = sparse(&a, format).matmul(&x).to_vec().unwrap();
        assert!(r[0].is_nan() && r[1] == 3.0, "{format:?}: {r:?}");
    }
}

#[test]
fn products_take_pending_operands() {
    let mut rng = Rng(10);
    let h = Host::random(&mut rng, 50, 60, 400, false);
    let x = quantized(&mut rng, 60);
    let xa = Array::from_slice(&x, 60).unwrap();
    // `x * 2 + 1` is not evaluated before the product.
    let pending = &xa * 2.0f32 + 1.0f32;
    let host: Vec<f32> = x.iter().map(|v| v * 2.0 + 1.0).collect();
    for format in FORMATS {
        assert_eq!(h.sparse(format).matmul(&pending).to_vec().unwrap(), host_matmul(&h.dense(), 50, 60, &host, 1), "{format:?}");
    }
}

#[test]
fn products_against_dense_matmul() {
    // The same product as luma's dense matmul (f32 precision) on a mid-size matrix.
    let mut rng = Rng(11);
    let h = Host::random(&mut rng, 96, 80, 1500, false);
    let d = dense_array(&h.dense(), 96, 80);
    let x = Array::from_slice(&quantized(&mut rng, 80 * 8), [8, 80]).unwrap();
    let want = d.matmul_f32(&x).to_vec().unwrap();
    for format in FORMATS {
        assert_eq!(h.sparse(format).matmul(&x).to_vec().unwrap(), want, "{format:?}");
    }
}

#[test]
fn elementwise_with_dense() {
    let mut rng = Rng(12);
    for (rows, cols, nnz) in [(1usize, 1usize, 1usize), (7, 13, 30), (130, 257, 3000)] {
        let h = Host::random(&mut rng, rows, cols, nnz, true);
        let d = h.dense();
        let o = quantized(&mut rng, rows * cols);
        let oa = dense_array(&o, rows, cols);
        // Dense forms of the results.
        let sum: Vec<f32> = d.iter().zip(&o).map(|(a, b)| a + b).collect();
        let diff: Vec<f32> = d.iter().zip(&o).map(|(a, b)| a - b).collect();
        let rdiff: Vec<f32> = d.iter().zip(&o).map(|(a, b)| b - a).collect();
        // The product with a dense matrix: entry by entry, then summed (duplicates multiply separately).
        let mut prod = vec![0.0f32; rows * cols];
        for k in 0..h.v.len() {
            let at = h.r[k] as usize * cols + h.c[k] as usize;
            prod[at] += h.v[k] * o[at];
        }
        for format in FORMATS {
            let s = h.sparse(format);
            assert_eq!(s.add_dense(&oa).to_vec().unwrap(), sum, "{format:?} add");
            assert_eq!(s.sub_dense(&oa).to_vec().unwrap(), diff, "{format:?} sub");
            assert_eq!(s.rsub_dense(&oa).to_vec().unwrap(), rdiff, "{format:?} rsub");
            let m = s.mul_dense(&oa);
            assert_eq!((m.format(), m.nnz()), (format, s.nnz()));
            assert_eq!(dense_of(&m), prod, "{format:?} mul");
            // The operands are untouched.
            assert_eq!(oa.to_vec().unwrap(), o);
            assert_eq!(dense_of(&s), d);
        }
    }
}

#[test]
fn division_by_dense_and_scalars() {
    let mut rng = Rng(13);
    let h = Host::random(&mut rng, 20, 30, 150, false);
    let d = h.dense();
    // Divisors that are powers of two keep the quotients exact.
    let o: Vec<f32> = (0..600).map(|i| [0.5f32, 1.0, 2.0, 4.0][i % 4]).collect();
    let oa = dense_array(&o, 20, 30);
    for format in FORMATS {
        let s = h.sparse(format);
        let q = s.div_dense(&oa);
        assert_eq!(q.nnz(), s.nnz());
        assert_eq!(dense_of(&q), d.iter().zip(&o).map(|(a, b)| a / b).collect::<Vec<_>>(), "{format:?} div");
        assert_eq!(dense_of(&s.mul_scalar(4.0)), d.iter().map(|v| v * 4.0).collect::<Vec<_>>());
        assert_eq!(dense_of(&s.div_scalar(2.0)), d.iter().map(|v| v / 2.0).collect::<Vec<_>>());
        assert_eq!(dense_of(&s.neg()), d.iter().map(|v| -v).collect::<Vec<_>>());
        assert_eq!(dense_of(&(&s * 8.0)), d.iter().map(|v| v * 8.0).collect::<Vec<_>>());
        assert_eq!(dense_of(&(&s / 4.0)), d.iter().map(|v| v / 4.0).collect::<Vec<_>>());
        assert_eq!(dense_of(&(-&s)), d.iter().map(|v| -v).collect::<Vec<_>>());
        assert_eq!(s.mul_scalar(2.0).format(), format);
        assert_eq!(s.add_scalar(0.5).to_vec().unwrap(), d.iter().map(|v| v + 0.5).collect::<Vec<_>>());
        assert_eq!(s.sub_scalar(0.5).to_vec().unwrap(), d.iter().map(|v| v - 0.5).collect::<Vec<_>>());
    }
}

#[test]
fn elementwise_mul_ignores_unstored_infinities() {
    // 0 * inf is not computed where there is no entry.
    let a = Array::from_slice(&[2.0f32, 0.0, 0.0, 3.0], [2, 2]).unwrap();
    let o = Array::from_slice(&[4.0f32, f32::INFINITY, f32::NAN, 5.0], [2, 2]).unwrap();
    for format in FORMATS {
        assert_eq!(dense_of(&sparse(&a, format).mul_dense(&o)), [8.0, 0.0, 0.0, 15.0], "{format:?}");
    }
}

#[test]
fn errors_poison() {
    let r = Array::from_slice(&[0u32, 1, 2], 3).unwrap();
    let c2 = Array::from_slice(&[0u32, 1], 2).unwrap();
    let v = Array::from_slice(&[1.0f32, 2.0, 3.0], 3).unwrap();
    let bad = Sparse::from_triplets(3, 3, &r, &c2, &v, SparseFormat::Csr);
    assert!(matches!(bad.error().map(LumaError::root), Some(LumaError::LengthMismatch { .. })), "{:?}", bad.error());
    // Everything computed from it fails with that error.
    assert!(bad.to_dense().to_vec().is_err());
    assert!(bad.matmul(&Array::from_slice(&[1.0f32; 3], 3).unwrap()).to_vec().is_err());
    assert!(bad.transpose().error().is_some());
    assert!(bad.to_format(SparseFormat::Coo).error().is_some());
    assert!(bad.mul_scalar(2.0).error().is_some());
    assert!(bad.mul_dense(&Array::from_slice(&[1.0f32; 9], [3, 3]).unwrap()).error().is_some());
    assert!(bad.add_dense(&Array::from_slice(&[1.0f32; 9], [3, 3]).unwrap()).to_vec().is_err());
    assert!(bad.values().is_none());

    let zero_dim = Sparse::from_triplets(0, 3, &r, &r, &v, SparseFormat::Coo);
    assert!(zero_dim.error().is_some());
    let ptr = Array::from_slice(&[0u32, 1, 3], 3).unwrap();
    let wrong_ptr = Sparse::from_csr(3, 3, &ptr, &r, &v);
    assert!(matches!(wrong_ptr.error().map(LumaError::root), Some(LumaError::LengthMismatch { expected: 4, actual: 3 })));
    let wrong_vals = Sparse::from_csc(3, 3, &Array::from_slice(&[0u32, 1, 2, 3], 4).unwrap(), &c2, &v);
    assert!(wrong_vals.error().is_some());
    let wide = Array::from_slice(&[0.0f32; 4], [2, 2, 1]).unwrap();
    assert!(sparse(&Array::from_slice(&[1.0f32; 8], [2, 2, 2]).unwrap(), SparseFormat::Csr).error().is_some());
    assert!(sparse(&wide, SparseFormat::Csr).error().is_none());

    // A dense operand of the wrong shape.
    let good = Sparse::from_triplets(3, 3, &r, &r, &v, SparseFormat::Csr);
    assert!(good.error().is_none());
    let wrong = Array::from_slice(&[1.0f32; 4], 4).unwrap();
    let e = good.matmul(&wrong);
    assert!(matches!(e.error().map(LumaError::root), Some(LumaError::ShapeMismatch { .. })), "{:?}", e.error());
    assert!(good.matmul_t(&wrong).to_vec().is_err());
    assert!(good.matmul(&Array::from_slice(&[1.0f32; 12], [3, 2, 2]).unwrap()).to_vec().is_err());
    assert!(good.mul_dense(&wrong).error().is_some());
    assert!(good.add_dense(&wrong).to_vec().is_err());
    assert!(good.sub_dense(&Array::from_slice(&[1.0f32; 9], [9]).unwrap()).to_vec().is_err());
    // A poisoned dense array poisons its sparse form.
    let poisoned = &Array::from_slice(&[1.0f32; 4], 4).unwrap() + &Array::from_slice(&[1.0f32; 3], 3).unwrap();
    assert!(sparse(&poisoned, SparseFormat::Csr).error().is_some());
}

#[test]
fn clones_share_and_stay_independent() {
    let mut rng = Rng(14);
    let h = Host::random(&mut rng, 30, 30, 200, false);
    let s = h.sparse(SparseFormat::Csr);
    let t = s.clone();
    let x = Array::from_slice(&quantized(&mut rng, 30), 30).unwrap();
    // A product that builds the expanded rows on one clone does not change the other.
    assert_eq!(t.matmul_t(&Array::from_slice(&quantized(&mut rng, 30), 30).unwrap()).to_vec().unwrap().len(), 30);
    assert_eq!(s.matmul(&x).to_vec().unwrap(), t.matmul(&x).to_vec().unwrap());
    let scaled = s.mul_scalar(2.0);
    assert_eq!(dense_of(&s), h.dense());
    assert_eq!(dense_of(&scaled), h.dense().iter().map(|v| v * 2.0).collect::<Vec<_>>());
}

#[test]
fn large_random_matrix_round_trips() {
    let mut rng = Rng(15);
    let (rows, cols) = (20_000usize, 20_000usize);
    let h = Host::random(&mut rng, rows, cols, 200_000, true);
    let x = quantized(&mut rng, cols);
    let xa = Array::from_slice(&x, cols).unwrap();
    // The reference product in f64 straight from the triplets.
    let mut want = vec![0.0f64; rows];
    for k in 0..h.v.len() {
        want[h.r[k] as usize] += f64::from(h.v[k]) * f64::from(x[h.c[k] as usize]);
    }
    let want: Vec<f32> = want.into_iter().map(|v| v as f32).collect();
    for format in FORMATS {
        let s = h.sparse(format);
        assert_eq!(s.matmul(&xa).to_vec().unwrap(), want, "{format:?}");
        for to in FORMATS {
            assert_eq!(s.to_format(to).matmul(&xa).to_vec().unwrap(), want, "{format:?} -> {to:?}");
        }
    }
}
