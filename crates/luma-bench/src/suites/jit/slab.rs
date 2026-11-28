//! Slabs below segments (W3-P4): arrays at twice the memory budget, at the default segment rule (`budget / 8`), run by the schedules of
//! `LUMA_SPLIT=segment` (slabs are segments) and `auto` (the planner's slabs, ghosts for shifts). Each row has both modes as cases
//! (`slab/<row>/segment`, `slab/<row>/auto`, measured again after the other as `_b`); the acceptance is `auto` at least as fast as `segment` on every row.

use super::common::*;
use crate::harness::*;
use luma::*;

/// A context of `mode` with a memory budget of `budget` bytes (segments of `budget / 8`).
fn context(mode: SplitMode, budget: u64) -> Context {
    Context::with_options(ContextOptions::from_env().memory_budget(budget).split(mode)).expect("a device is available")
}

type Row = fn(&A, usize) -> A;

fn map_k8(x: &A, _: usize) -> A {
    (0..8).fold(x.clone(), |acc, k| (&acc * 1.0001 + 0.5 * (k as f32 + 1.0)).abs())
}

fn shift1(x: &A, _: usize) -> A {
    x.shift(1) + x
}

fn stencil5(x: &A, side: usize) -> A {
    let e = (x * 0.5 + 1.0).abs();
    let g = e.reshape([side, side]);
    (g.shift([1, 0]) + g.shift([-1, 0]) + g.shift([0, 1]) + g.shift([0, -1]) - &g * 4.0).reshape([side * side])
}

fn pad_edge(x: &A, _: usize) -> A {
    x.pad([(16, 16)], PadMode::Edge).slice(0, 0..x.len()) + x
}

pub fn run(h: &mut Harness) {
    h.guarded("slab", |h| {
        let budget: u64 = if h.quick { 8 << 20 } else { 64 << 20 };
        let side = ((2 * budget / 4) as f64).sqrt() as usize;
        let n = side * side;
        let xh = values(n, 21, 0.0, 1.0);
        let rows: [(&str, Row); 4] = [("map_k8_2x", map_k8), ("shift1_2x", shift1), ("stencil5_2x", stencil5), ("pad_edge_2x", pad_edge)];
        let mut reference: Vec<Vec<f32>> = Vec::new();
        for (r, (name, row)) in rows.iter().enumerate() {
            let modes = [(SplitMode::Segment, "segment"), (SplitMode::Auto, "auto")];
            let runs: Vec<(Context, A)> = modes
                .iter()
                .map(|&(mode, _)| {
                    let ctx = context(mode, budget);
                    let x = A::from_slice_on(&ctx, &xh, n).unwrap();
                    (ctx, x)
                })
                .collect();
            for ((_, x), (_, label)) in runs.iter().zip(modes) {
                let got = row(x, side).to_vec().unwrap();
                match reference.get(r) {
                    Some(want) => assert!(got.iter().zip(want).all(|(a, b)| a.to_bits() == b.to_bits()), "slab/{name}: {label} differs from segment"),
                    None => reference.push(got),
                }
            }
            // ABBA, so that the order of the modes (memory state, clocks) does not decide the comparison: the second pair has a `_b` suffix.
            for (m, suffix) in [(0, ""), (1, ""), (1, "_b"), (0, "_b")] {
                let (ctx, x) = &runs[m];
                let case = format!("slab/{name}/{}{suffix}", modes[m].1);
                let stats_before = ctx.stream_stats();
                // The harness profiles its own context: it must be the one the row runs on.
                let saved = std::mem::replace(&mut h.ctx, ctx.clone());
                h.case(&case, &format!("n={n}"), Work::Bytes(8.0 * n as f64), &mut || {
                    force(&row(x, side));
                    ctx.sync().unwrap();
                });
                h.ctx = saved;
                let stats = ctx.stream_stats();
                println!("   {case}: slabs {} ghost_bytes {} upload {} download {}", stats.slabs - stats_before.slabs, stats.ghost_bytes - stats_before.ghost_bytes, stats.upload_bytes - stats_before.upload_bytes, stats.download_bytes - stats_before.download_bytes);
            }
        }
    });
}
