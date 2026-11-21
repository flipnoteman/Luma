//! Format conversions: COO to CSR (stable argsort of the rows, gathers of columns and values, a
//! count and a running sum for the row pointer), CSR to COO rows (`searchsorted` of the entry
//! numbers in the row pointer) and CSR transpose (the same sort keyed by column).

use luma::*;

use super::data::{self, Rng};
use super::*;

/// COO to CSR on the device: the end of every row (the running sum of the row counts), and the
/// columns and values in row order (entries of a row keep their COO order).
fn coo_to_csr(rows: &Array<u32>, cols: &Array<u32>, vals: &Array<f32>, n: usize) -> (Array<u32>, Array<u32>, Array<f32>) {
    let perm = rows.argsort();
    (bincount(rows, n).cumsum(0), cols.take(&perm), vals.take(&perm))
}

/// COO to CSR on the host by a stable counting sort: row pointer, columns, values.
fn coo_to_csr_host(rows: &[u32], cols: &[u32], vals: &[f32], n: usize) -> (Vec<u32>, Vec<u32>, Vec<f32>) {
    let mut ptr = vec![0u32; n + 1];
    rows.iter().for_each(|&r| ptr[r as usize + 1] += 1);
    for i in 0..n {
        ptr[i + 1] += ptr[i];
    }
    let mut next = ptr.clone();
    let (mut col, mut val) = (vec![0u32; rows.len()], vec![0f32; rows.len()]);
    for k in 0..rows.len() {
        let slot = &mut next[rows[k] as usize];
        col[*slot as usize] = cols[k];
        val[*slot as usize] = vals[k];
        *slot += 1;
    }
    (ptr, col, val)
}

fn eval3(ptr: &Array<u32>, col: &Array<u32>, val: &Array<f32>) {
    ptr.eval().unwrap();
    col.eval().unwrap();
    val.eval().unwrap();
}

pub fn run(h: &mut Harness) {
    h.guarded("convert", coo_cases);
    h.guarded("convert", csr_cases);
}

fn coo_cases(h: &mut Harness) {
    let n = scale(h, 65_536, 262_144, 2_000_000);
    let nnz = 8 * n;
    let mut rng = Rng::new(51);
    for (tag, skew) in [("uniform", false), ("skewed", true)] {
        let name = format!("convert/coo_to_csr_{tag}");
        if !h.want(&name) {
            continue;
        }
        let rows: Vec<u32> = (0..nnz).map(|_| if skew { ((rng.unit().powi(2) * n as f64) as usize).min(n - 1) } else { rng.below(n) } as u32).collect();
        let cols: Vec<u32> = (0..nnz).map(|_| rng.below(n) as u32).collect();
        let vals = values(nnz, 52, -1.0, 1.0);
        let (r, c, v) = (up_u32(h, &rows, nnz), up_u32(h, &cols, nnz), up_f32(h, &vals, nnz));
        let (hp, hc, hv) = coo_to_csr_host(&rows, &cols, &vals, n);
        let (ptr, col, val) = coo_to_csr(&r, &c, &v, n);
        exact(&name, &ptr.to_vec().unwrap(), &hp[1..]);
        exact(&name, &col.to_vec().unwrap(), &hc);
        exact(&name, &val.to_vec().unwrap().iter().map(|v| v.to_bits()).collect::<Vec<_>>(), &hv.iter().map(|v| v.to_bits()).collect::<Vec<_>>());
        drop((ptr, col, val));
        let size = format!("{} rows {} nnz", label(n), label(nnz));
        case_trim(h, &name, &size, Work::Items(nnz as f64, "Mnnz/s"), &mut || {
            let (p, c2, v2) = coo_to_csr(&r, &c, &v, n);
            eval3(&p, &c2, &v2);
        });
    }
}

fn csr_cases(h: &mut Harness) {
    let n = scale(h, 65_536, 262_144, 2_000_000);
    if !wants(h, "convert/csr_") {
        return;
    }
    let a = data::uniform(n, 4, 12, 53);
    let nnz = a.nnz();
    let dev = spmv::upload(h, &a);
    let size = format!("{} rows {} nnz", label(n), label(nnz));

    let name = "convert/csr_to_coo";
    if h.want(name) {
        let end = dev.ptr.slice(0, 1..=n);
        let ctx = h.ctx.clone();
        let expand = || end.searchsorted(&Array::<u32>::range_on(&ctx, nnz, 0).unwrap(), Side::Right);
        exact(name, &expand().to_vec().unwrap(), &a.row_of());
        case_trim(h, name, &size, Work::Items(nnz as f64, "Mnnz/s"), &mut || expand().eval().unwrap());
    }

    let name = "convert/csr_transpose";
    if h.want(name) {
        let (hp, hc, hv) = coo_to_csr_host(&a.col, &a.row_of(), &a.val, n);
        let (ptr, col, val) = coo_to_csr(&dev.col, &dev.row, &dev.val, n);
        exact(name, &ptr.to_vec().unwrap(), &hp[1..]);
        exact(name, &col.to_vec().unwrap(), &hc);
        exact(name, &val.to_vec().unwrap().iter().map(|v| v.to_bits()).collect::<Vec<_>>(), &hv.iter().map(|v| v.to_bits()).collect::<Vec<_>>());
        drop((ptr, col, val));
        case_trim(h, name, &size, Work::Items(nnz as f64, "Mnnz/s"), &mut || {
            let (p, c, v) = coo_to_csr(&dev.col, &dev.row, &dev.val, n);
            eval3(&p, &c, &v);
        });
    }

    let name = "convert/csr_prune";
    if h.want(name) {
        // Drop the entries below 1/4 in magnitude: three `filter`s with one mask (each counts and
        // reads back on its own), then the row pointer of what is left.
        let prune = || {
            let keep = dev.val.abs().ge(0.25f32);
            let rows = dev.row.filter(&keep).unwrap().unwrap();
            let col = dev.col.filter(&keep).unwrap().unwrap();
            let val = dev.val.filter(&keep).unwrap().unwrap();
            (bincount(&rows, n).cumsum(0), col, val)
        };
        let kept: Vec<usize> = (0..nnz).filter(|&k| a.val[k].abs() >= 0.25).collect();
        let mut counts = vec![0u32; n];
        let row = a.row_of();
        kept.iter().for_each(|&k| counts[row[k] as usize] += 1);
        let want_ptr: Vec<u32> = counts.iter().scan(0u32, |s, &c| { *s += c; Some(*s) }).collect();
        let (ptr, col, val) = prune();
        exact(name, &ptr.to_vec().unwrap(), &want_ptr);
        exact(name, &col.to_vec().unwrap(), &kept.iter().map(|&k| a.col[k]).collect::<Vec<_>>());
        exact(name, &val.to_vec().unwrap(), &kept.iter().map(|&k| a.val[k]).collect::<Vec<_>>());
        drop((ptr, col, val));
        case_trim(h, name, &size, Work::Items(nnz as f64, "Mnnz/s"), &mut || {
            let (p, c, v) = prune();
            eval3(&p, &c, &v);
        });
    }
}
