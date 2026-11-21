//! The `sparse` suite: sparse linear algebra and graph workloads written with luma's array API
//! (gathers with `take`, scatters with `put_add` / `put_min`, `group_sum`, scans, sort / argsort,
//! `searchsorted`, `nonzero`, `iterate_all` / `iterate_until_all` loops and `matmul`).
//!
//! * `spmv`: CSR `y = A x` as a gather plus a segmented sum, on banded, uniform-random and
//!   power-law matrices, with the segmented sum made by an atomic `group_sum`, by one running
//!   sum (`cumsum`), by an ELLPACK `sum_axis` and by split rows (`sum_axis` over equal pieces
//!   followed by a small `group_sum`);
//! * `cg`: conjugate gradients on the 2-D Poisson matrix in CSR (atomic and ELL) and matrix-free
//!   (shifts), as fixed-count `iterate_all` loops and as an `iterate_until_all` loop;
//! * `convert`: COO to CSR (argsort, count, scan), CSR to COO (`searchsorted`), CSR transpose and
//!   pruning small entries (`filter`);
//! * `pagerank`: power iterations on a power-law graph (gather + `put_add` per iteration), with
//!   the edges in random and in destination order;
//! * `bfs`: breadth-first levels from a hub, as dense edge-parallel levels (host-checked and fixed
//!   count in a device loop) and with compacted frontiers (`nonzero`, `cumsum`, `searchsorted`);
//! * `cc`: connected components by min-label propagation (`put_min` until no label changes),
//!   plain and with pointer jumping; `sssp`: Bellman-Ford shortest paths (`f32` `put_min` rounds);
//! * `degree`, `khop`: degree counts, histograms and wedge counts of a bipartite (triangle-free)
//!   graph, and neighbourhood sizes of several seeds after k hops;
//! * `spmm`: sparse times a few dense columns (atomic and ELL), against a dense `matmul` of the
//!   same matrix, and a graph-convolution layer (`matmul` then SpMM).
//!
//! Every case is checked against a host reference (`f64` where sums matter) before it is timed.
//! Sizes: the large size is about 2M rows (10M-30M entries) on a GPU, 256K on the CPU backend and
//! 64K with `--quick`.

mod convert;
mod data;
mod graph;
mod solve;
mod spmm;
mod spmv;

use std::fmt::Debug;

use crate::harness::*;
use luma::*;

pub fn run(h: &mut Harness) {
    spmv::run(h);
    solve::run(h);
    convert::run(h);
    graph::run(h);
    spmm::run(h);
}

/// `quick` in `--quick` runs, `cpu` on the CPU backend, else `gpu`.
fn scale(h: &Harness, quick: usize, cpu: usize, gpu: usize) -> usize {
    if h.quick {
        quick
    } else if h.ctx.is_cpu() {
        cpu
    } else {
        gpu
    }
}

/// Whether any case of `prefix` (`"spmv/banded"`) can be selected by `--only`.
fn wants(h: &Harness, prefix: &str) -> bool {
    let full = format!("{}/{}", h.suite, prefix);
    h.only.as_deref().is_none_or(|only| full.contains(only) || only.starts_with(&full))
}

fn label(n: usize) -> String {
    if n >= 1_000_000 && n % 100_000 == 0 {
        format!("{}M", n as f64 / 1e6)
    } else if n >= 1_000 && n % 100 == 0 {
        format!("{}K", n as f64 / 1e3)
    } else {
        n.to_string()
    }
}

fn up_f32(h: &Harness, data: &[f32], shape: impl Into<Shape>) -> Array<f32> {
    let a = Array::from_slice_on(&h.ctx, data, shape).unwrap();
    a.eval().unwrap();
    a
}

fn up_u32(h: &Harness, data: &[u32], shape: impl Into<Shape>) -> Array<u32> {
    let a = Array::from_slice_on(&h.ctx, data, shape).unwrap();
    a.eval().unwrap();
    a
}

/// [`Harness::case`] after releasing the pooled idle buffers of the earlier cases, so that the
/// `mem MiB` column is this case's own footprint. With `LUMA_BENCH_KERNELS=1` the kernels of one
/// call are listed under the case row (label, launches, GPU time).
fn case_trim(h: &mut Harness, name: &str, size: &str, work: Work, f: &mut dyn FnMut()) {
    if h.want(name) {
        h.ctx.trim().unwrap();
    }
    h.case(name, size, work, f);
    if h.want(name)
        && std::env::var_os("LUMA_BENCH_KERNELS").is_some()
        && let Ok(((), p)) = h.ctx.profile(&mut *f)
    {
        let _ = h.ctx.sync();
        for k in p.kernels.iter().take(10) {
            println!("    {:<64.64} {:>5} x {:>9.3} ms", k.label, k.launches, k.total.as_secs_f64() * 1e3);
        }
    }
}

fn exact<T: PartialEq + Debug>(name: &str, got: &[T], want: &[T]) {
    assert_eq!(got.len(), want.len(), "{name}: length");
    if let Some(i) = got.iter().zip(want).position(|(g, w)| g != w) {
        panic!("{name}: index {i}: got {:?}, want {:?}", got[i], want[i]);
    }
}

/// Panics unless every `got[i]` is within `rel * scale + abs` of the `f64` reference `want[i]`.
fn check_flat(name: &str, got: &[f32], want: &[f64], scale: f64, rel: f64, abs: f64) {
    assert_eq!(got.len(), want.len(), "{name}: length");
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        let tol = rel * scale + abs;
        assert!((f64::from(g) - w).abs() <= tol, "{name}: index {i}: got {g}, want {w} (tol {tol})");
    }
}

fn to_f32(v: &[f64]) -> Vec<f32> {
    v.iter().map(|&x| x as f32).collect()
}
