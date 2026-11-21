//! Embedding tables: row gather (`take`), scatter-add gradients (`put_add`, atomics, with
//! uniform and Zipf-like hot ids), sparse in-place updates, bags and a gather feeding a matmul.
//!
//! A table of `V` rows of width `D` is `[D, V]`; the flat index of element `d` of row `id` is
//! `id * D + d`, built on the device from the id array.

use super::common::*;
use crate::harness::*;
use luma::*;

/// Token ids: uniform in `0..v`, or hot ones (a few ids take most of the draws).
fn token_ids(n: usize, v: usize, hot: bool) -> Vec<u32> {
    let raw = ints(n, 51, v as u32);
    if hot {
        raw.iter().map(|&x| ((u64::from(x).pow(3)) / (v as u64 * v as u64)) as u32).collect()
    } else {
        raw
    }
}

/// `[d, ...]` flat indices of the rows `ids` (shape `[1, ...]`).
fn flat_index(ctx: &Context, ids: &Array<u32>, d: usize, shape: impl Into<Shape>) -> Array<u32> {
    ids * d as u32 + &Array::<u32>::range_on(ctx, shape, 0).unwrap()
}

/// `(v, d, t)` scaled down on the CPU.
fn dims(h: &Harness, v: usize, d: usize, t: usize) -> (usize, usize, usize) {
    if h.ctx.is_cpu() {
        (v / 4, d, t / 4)
    } else if h.quick {
        (v / 2, d, t / 2)
    } else {
        (v, d, t)
    }
}

fn gather(h: &mut Harness, v: usize, d: usize, t: usize) {
    let (v, d, t) = dims(h, v, d, t);
    let case = "embed/gather";
    let size = format!("V{v} D{d} T{t}");
    h.guarded(&format!("{case} {size}"), |h| {
        let ctx = h.ctx.clone();
        let table = values(v * d, 52, -1.0, 1.0);
        let ids = token_ids(t, v, false);
        let (tab, id) = (arr(h, &table, [d, v]), arr_u32(h, &ids, [1, t]));
        let run = || tab.take(&flat_index(&ctx, &id, d, [d, t]));
        let got = run().to_vec().unwrap();
        let want: Vec<f32> = ids.iter().flat_map(|&i| table[i as usize * d..][..d].iter().copied()).collect();
        assert_eq!(got, want, "{case}: rows differ");
        h.case(case, &size, Work::Bytes(8.0 * (t * d) as f64 + 4.0 * t as f64), &mut || run().eval().unwrap());
    });
}

/// The gradient of the table: `grad[ids[t]] += g[t]`, into a fresh zero table every call.
fn scatter_add(h: &mut Harness, v: usize, d: usize, t: usize, hot: bool) {
    let (v, d, t) = dims(h, v, d, t);
    let case = if hot { "embed/scatter_add_hot" } else { "embed/scatter_add" };
    let size = format!("V{v} D{d} T{t}");
    h.guarded(&format!("{case} {size}"), |h| {
        let ctx = h.ctx.clone();
        let g = values(t * d, 53, -1.0, 1.0);
        let ids = token_ids(t, v, hot);
        let (grad, id) = (arr(h, &g, [d, t]), arr_u32(h, &ids, [1, t]));
        let run = || {
            let mut table = Array::<f32>::zeros_on(&ctx, [d, v]).unwrap();
            table.put_add(&flat_index(&ctx, &id, d, [d, t]), &grad).unwrap();
            table
        };
        let mut want = vec![0.0f64; v * d];
        for (row, &i) in ids.iter().enumerate() {
            for c in 0..d {
                want[i as usize * d + c] += f64::from(g[row * d + c]);
            }
        }
        // Atomic float adds accumulate in no fixed order.
        check64(case, &run().to_vec().unwrap(), &want, 1e-3, 1e-3);
        h.case(case, &size, Work::Bytes(4.0 * (v * d) as f64 + 8.0 * (t * d) as f64), &mut || run().eval().unwrap());
    });
}

/// Sparse SGD on the table in place: `table[ids[t]] -= lr * g[t]`.
fn sparse_update(h: &mut Harness, v: usize, d: usize, t: usize) {
    let (v, d, t) = dims(h, v, d, t);
    let case = "embed/sparse_update";
    let size = format!("V{v} D{d} T{t}");
    h.guarded(&format!("{case} {size}"), |h| {
        let ctx = h.ctx.clone();
        let table0 = values(v * d, 54, -1.0, 1.0);
        let g = values(t * d, 55, -1.0, 1.0);
        let ids = token_ids(t, v, true);
        let (grad, id) = (arr(h, &g, [d, t]), arr_u32(h, &ids, [1, t]));
        let mut table = arr(h, &table0, [d, v]);
        let lr = 0.01f32;
        let step = |table: &mut Array<f32>| {
            table.put_add(&flat_index(&ctx, &id, d, [d, t]), &(&grad * -lr)).unwrap();
            table.eval().unwrap();
        };
        step(&mut table);
        let mut want: Vec<f64> = table0.iter().map(|&x| f64::from(x)).collect();
        for (row, &i) in ids.iter().enumerate() {
            for c in 0..d {
                want[i as usize * d + c] -= f64::from(lr) * f64::from(g[row * d + c]);
            }
        }
        check64(case, &table.to_vec().unwrap(), &want, 1e-3, 1e-3);
        h.case(case, &size, Work::Bytes(8.0 * (t * d) as f64 + 4.0 * t as f64), &mut || step(&mut table));
    });
}

/// Embedding bags: the sum of `len` rows per bag (a gather fused into a reduction).
fn bag(h: &mut Harness, v: usize, d: usize, bags: usize, len: usize) {
    let (v, d, bags) = dims(h, v, d, bags);
    let case = "embed/bag_sum";
    let size = format!("V{v} D{d} {bags}x{len}");
    h.guarded(&format!("{case} {size}"), |h| {
        let ctx = h.ctx.clone();
        let table = values(v * d, 56, -1.0, 1.0);
        let ids = token_ids(bags * len, v, false);
        let (tab, id) = (arr(h, &table, [d, v]), arr_u32(h, &ids, [1, len, bags]));
        let run = || tab.take(&flat_index(&ctx, &id, d, [d, len, bags])).sum_axis(1);
        let got = run().to_vec().unwrap();
        for b in sample(bags, 6) {
            for c in sample(d, 6) {
                let want: f64 = (0..len).map(|l| f64::from(table[ids[b * len + l] as usize * d + c])).sum();
                check_scalar(case, f64::from(got[b * d + c]), want, 1e-3);
            }
        }
        h.case(case, &size, Work::Bytes(4.0 * (bags * len * d) as f64 + 4.0 * (bags * d) as f64), &mut || run().eval().unwrap());
    });
}

/// A gather that feeds a linear layer: `embedding(ids) W`.
fn lookup_linear(h: &mut Harness, v: usize, d: usize, t: usize, n: usize) {
    let (v, d, t) = dims(h, v, d, t);
    let flops = 2.0 * (t * d * n) as f64;
    if !fits(h, flops) {
        return;
    }
    let case = "embed/lookup_linear";
    let size = format!("V{v} D{d} T{t} N{n}");
    h.guarded(&format!("{case} {size}"), |h| {
        let ctx = h.ctx.clone();
        let table = values(v * d, 57, -1.0, 1.0);
        let s = 1.0 / (d as f32).sqrt();
        let ws = values(d * n, 58, -s, s);
        let ids = token_ids(t, v, false);
        let (tab, w, id) = (arr(h, &table, [d, v]), arr(h, &ws, [n, d]), arr_u32(h, &ids, [1, t]));
        let run = || matmul(&tab.take(&flat_index(&ctx, &id, d, [d, t])), &w);
        let picked = sample(t, 6);
        let rows: Vec<f32> = ids.iter().flat_map(|&i| table[i as usize * d..][..d].iter().copied()).collect();
        let want = gemm_rows(&rows, &ws, d, n, &picked);
        check64(case, &gather_rows(&run().to_vec().unwrap(), n, &picked), &want, 2e-3, 2e-3);
        h.case(case, &size, Work::Flops(flops), &mut || run().eval().unwrap());
    });
}

pub fn run(h: &mut Harness) {
    gather(h, 32768, 128, 4096);
    gather(h, 50000, 256, 16384);
    gather(h, 100000, 64, 65536);
    scatter_add(h, 32768, 128, 4096, false);
    scatter_add(h, 50000, 256, 16384, false);
    scatter_add(h, 50000, 256, 16384, true);
    scatter_add(h, 100000, 64, 65536, true);
    sparse_update(h, 50000, 256, 16384);
    bag(h, 50000, 128, 4096, 32);
    lookup_linear(h, 50000, 256, 4096, 512);
}
