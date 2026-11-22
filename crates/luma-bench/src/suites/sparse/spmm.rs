//! Sparse matrix times a few dense columns, `Y = A X` with `X` of shape `[c, n]` (the `c` values
//! of a row are adjacent, so a gather reads `c` consecutive floats): an atomic `group_sum` by
//! output element and an ELLPACK `sum_axis`, both against a dense `matmul` of the same matrix,
//! and one graph-convolution layer `relu(A (H W))` (a `matmul` and an SpMM).
//!
//! Values are multiples of small powers of two, so sums are exact in any order.

use luma::*;

use super::data::{self, Csr};
use super::*;

/// `Y = A X` with an atomic `group_sum` of the `[c, nnz]` products by output element.
fn atomic(a: &spmv::Dev, x: &Array<f32>, c: usize, ctx: &Context) -> Array<f32> {
    let nnz = a.nnz;
    let j = Array::<u32>::range_on(ctx, [c, nnz], 0).unwrap();
    let wide = |ids: &Array<u32>| ids.reshape([1, nnz]).broadcast_to([c, nnz]) * c as u32 + &j;
    let prod = a.val.reshape([1, nnz]).broadcast_to([c, nnz]) * x.take(&wide(&a.col));
    prod.group_sum(&wide(&a.row), a.rows * c)
}

/// `Y = A X` from ELLPACK: `[c, w, rows]` products summed over the `w` slots; shape `[c, 1, rows]`.
fn ell(a: &spmv::DevEll, x: &Array<f32>, c: usize, ctx: &Context) -> Array<f32> {
    let (w, rows) = (a.w, a.rows);
    let j = Array::<u32>::range_on(ctx, [c, w, rows], 0).unwrap();
    let col = a.col.reshape([1, w, rows]).broadcast_to([c, w, rows]) * c as u32 + &j;
    (a.val.reshape([1, w, rows]).broadcast_to([c, w, rows]) * x.take(&col)).sum_axis(1)
}

/// `Y = A X` in `f64`, with `X` of shape `[c, n]` as a flat slice.
fn spmm_host(a: &Csr, x: &[f32], c: usize) -> Vec<f64> {
    let mut y = vec![0.0f64; a.rows * c];
    for r in 0..a.rows {
        for k in a.ptr[r] as usize..a.ptr[r + 1] as usize {
            let (v, col) = (f64::from(a.val[k]), a.col[k] as usize);
            for j in 0..c {
                y[r * c + j] += v * f64::from(x[col * c + j]);
            }
        }
    }
    y
}

pub fn run(h: &mut Harness) {
    h.guarded("spmm", columns);
    h.guarded("spmm", dense_vs_sparse);
    h.guarded("spmm", gcn);
}

fn columns(h: &mut Harness) {
    if !wants(h, "spmm/csr") && !wants(h, "spmm/ell") {
        return;
    }
    let rows = scale(h, 16_384, 65_536, 262_144);
    let a = data::uniform(rows, 8, 24, 91);
    let (dev, e) = (spmv::upload(h, &a), spmv::upload_ell(h, &a));
    let ctx = h.ctx.clone();
    for c in [4usize, 16, 64] {
        let xh = data::quantized(rows * c, 92 + c as u64);
        let want = spmm_host(&a, &xh, c);
        let x = up_f32(h, &xh, [c, rows]);
        let size = format!("{} rows {} nnz, {c} columns", label(rows), label(a.nnz()));
        let work = Work::Flops(2.0 * a.nnz() as f64 * c as f64);
        let name = format!("spmm/csr_c{c}");
        if h.want(&name) {
            check_flat(&name, &atomic(&dev, &x, c, &ctx).to_vec().unwrap(), &want, 0.0, 0.0, 0.0);
            case_trim(h, &name, &size, work, &mut || atomic(&dev, &x, c, &ctx).eval().unwrap());
        }
        let name = format!("spmm/ell_c{c}");
        if h.want(&name) {
            check_flat(&name, &ell(&e, &x, c, &ctx).to_vec().unwrap(), &want, 0.0, 0.0, 0.0);
            case_trim(h, &name, &size, work, &mut || ell(&e, &x, c, &ctx).eval().unwrap());
        }
    }
}

/// The same matrix (about 64 entries per row) as a dense `matmul` operand and as CSR: where the
/// dense product's arithmetic rate outweighs its 25 to 100 times larger operand.
fn dense_vs_sparse(h: &mut Harness) {
    if !wants(h, "spmm/dense") && !wants(h, "spmm/sparse_same") {
        return;
    }
    let (m, c) = (scale(h, 1024, 2048, 4096), 64);
    let a = data::uniform(m, 56, 72, 93);
    let xh = data::quantized(m * c, 94);
    let want = spmm_host(&a, &xh, c);
    let mut dense = vec![0.0f32; m * m];
    for r in 0..m {
        for k in a.ptr[r] as usize..a.ptr[r + 1] as usize {
            dense[r * m + a.col[k] as usize] += a.val[k];
        }
    }
    let (da, x) = (up_f32(h, &dense, [m, m]), up_f32(h, &xh, [c, m]));
    let ctx = h.ctx.clone();
    let size = format!("{m} rows {} nnz ({:.1}% dense), {c} columns", label(a.nnz()), 100.0 * a.nnz() as f64 / (m * m) as f64);

    let name = "spmm/dense_matmul";
    if h.want(name) {
        check_flat(name, &da.matmul(&x).to_vec().unwrap(), &want, 1e-5, 1e-4, 1e-3);
        case_trim(h, name, &size, Work::Flops(2.0 * (m * m * c) as f64), &mut || da.matmul(&x).eval().unwrap());
    }
    let name = "spmm/sparse_same";
    if h.want(name) {
        let dev = spmv::upload(h, &a);
        check_flat(name, &atomic(&dev, &x, c, &ctx).to_vec().unwrap(), &want, 0.0, 0.0, 0.0);
        case_trim(h, name, &size, Work::Flops(2.0 * a.nnz() as f64 * c as f64), &mut || atomic(&dev, &x, c, &ctx).eval().unwrap());
    }
}

/// One graph-convolution layer `relu(A (H W))`: `H` is `[c_in, n]`, `W` is `[c_out, c_in]`.
fn gcn(h: &mut Harness) {
    if !wants(h, "spmm/gcn") {
        return;
    }
    let (rows, c_in, c_out) = (scale(h, 16_384, 65_536, 262_144), 64, 16);
    let a = data::uniform(rows, 8, 24, 95);
    let hh = data::quantized(rows * c_in, 96);
    let wh: Vec<f32> = data::quantized(c_out * c_in, 97).iter().map(|v| v / 4.0).collect();
    // Host: (H W)[r, j] = sum_k H[r, k] W[j, k], then A, then relu.
    let mut hw = vec![0.0f32; rows * c_out];
    for r in 0..rows {
        for j in 0..c_out {
            hw[r * c_out + j] = (0..c_in).map(|k| f64::from(hh[r * c_in + k]) * f64::from(wh[j * c_in + k])).sum::<f64>() as f32;
        }
    }
    let want: Vec<f64> = spmm_host(&a, &hw, c_out).into_iter().map(|v| v.max(0.0)).collect();
    let (dev, hd, w) = (spmv::upload(h, &a), up_f32(h, &hh, [c_in, rows]), up_f32(h, &wh, [c_in, c_out]));
    let ctx = h.ctx.clone();
    let layer = || atomic(&dev, &hd.matmul(&w.transpose()), c_out, &ctx).maximum(0.0f32);
    let name = "spmm/gcn_layer";
    check_flat(name, &layer().to_vec().unwrap(), &want, 1e-5, 1e-4, 1e-3);
    let size = format!("{} nodes {} nnz, {c_in} -> {c_out} features", label(rows), label(a.nnz()));
    let flops = 2.0 * (rows * c_in * c_out) as f64 + 2.0 * (a.nnz() * c_out) as f64;
    case_trim(h, name, &size, Work::Flops(flops), &mut || layer().eval().unwrap());
}
