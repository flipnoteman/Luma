//! Thread coarsening: a plain store kernel whose invocations each compute several elements.
//!
//! A fused elementwise kernel runs one invocation per output element. With coarsening
//! (`Tuning::coarsen`, `LUMA_COARSEN=off|auto|<n>`) it runs `C` (1, 2, 4 or 8) elements per
//! invocation, spaced by the number of invocations of the launch (the grid-stride layout): element
//! `i`, `i + T`, `i + 2T`, ... with `T = workgroups * workgroup size`, so that the lanes of a warp
//! stay on neighbouring elements and every access coalesces. The launch has `C` times fewer
//! workgroups. The body is written out `C` times with all the loads before the first store
//! (see `kirgen::Build::plain`), which gives the memory system more independent requests per
//! invocation and pays the per-invocation work (the element index, the grid size, the uniform
//! reads) once for `C` elements. Each element runs the same arithmetic as without coarsening, so
//! the results are bit-identical.

use crate::context::Context;
use crate::tuning::Coarsen;

use super::program::{FusedKey, Tail};

/// The most elements one invocation computes.
pub(crate) const MAX_COARSEN: u8 = 8;

/// Whether the kernel of `key` can run several elements per invocation: the plain store kernel of
/// a program over one buffer per array. The kernels that stage values in workgroup memory, run
/// per segment, write the bits of `bool` arrays (invocations share a word), build runs, or are
/// not a plain store keep one element per invocation.
pub(super) fn coarsenable(key: &FusedKey) -> bool {
    key.tail == Tail::Store
        && key.tile.is_none()
        && key.staging.is_none()
        && !key.runs
        && key.spans.is_empty()
        && !key.outputs.is_empty()
        && !key.writes_bits()
}

/// The elements per invocation of the launch of `key` over `len` elements.
pub(super) fn plan(ctx: &Context, key: &FusedKey, len: usize) -> u8 {
    if !coarsenable(key) || len > (1 << 30) {
        return 1;
    }
    match ctx.tuning().coarsen {
        Coarsen::Off => 1,
        Coarsen::Fixed(n) => n.clamp(1, MAX_COARSEN),
        Coarsen::Auto => auto(ctx, key, len),
    }
}

fn auto(_ctx: &Context, _key: &FusedKey, _len: usize) -> u8 {
    1
}

#[cfg(test)]
mod tests {
    use crate::array::Array;
    use crate::context::{Context, ContextOptions};
    use crate::jit::cost::CostModel;
    use crate::ops::{eval_all, select};
    use crate::tuning::Coarsen;
    use crate::Df32;

    fn context(coarsen: Coarsen) -> Context {
        let ctx = Context::with_tuning(ContextOptions::from_env().calibrate(false), |tuning| tuning.coarsen = coarsen).unwrap();
        ctx.set_cost_model(CostModel::TEST);
        ctx
    }

    fn values(n: usize, seed: u64, lo: f32, hi: f32) -> Vec<f32> {
        let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (0..n)
            .map(|_| {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                lo + ((state >> 40) as f32 / (1u64 << 24) as f32) * (hi - lo)
            })
            .collect()
    }

    fn ints(n: usize, seed: u64) -> Vec<i32> {
        values(n, seed, -1000.0, 1000.0).into_iter().map(|v| v as i32).collect()
    }

    /// Runs `program` with every count of elements per invocation and demands the bits of the
    /// results of one per invocation, and that some launch really was coarsened.
    fn same_bits(program: impl Fn(&Context) -> Vec<Vec<u64>>) {
        let one = context(Coarsen::Fixed(1));
        let want = program(&one);
        assert_eq!(one.jit().stats().coarsened, 0);
        for count in [2u8, 4, 8] {
            let ctx = context(Coarsen::Fixed(count));
            let got = program(&ctx);
            assert!(ctx.jit().stats().coarsened > 0, "no launch ran {count} elements per invocation");
            assert_eq!(got, want, "{count} elements per invocation");
        }
        let off = context(Coarsen::Off);
        assert_eq!(program(&off), want);
        assert_eq!(off.jit().stats().coarsened, 0);
    }

    fn f32_bits(array: &Array<f32>) -> Vec<u64> {
        array.to_vec().unwrap().into_iter().map(|v| u64::from(v.to_bits())).collect()
    }

    const LENGTHS: [usize; 12] = [1, 2, 3, 7, 63, 64, 65, 255, 513, 1000, 4097, 70_001];

    #[test]
    fn odd_lengths_match_one_element_per_invocation() {
        for n in LENGTHS {
            same_bits(|ctx| {
                let x = Array::from_slice_on(ctx, &values(n, 1, -2.0, 2.0), n).unwrap();
                let y = Array::from_slice_on(ctx, &values(n, 2, 0.5, 3.0), n).unwrap();
                vec![f32_bits(&((&x * &y + 1.0).sqrt() - (&x * 0.5).exp() / &y))]
            });
        }
    }

    #[test]
    fn views_broadcasts_and_shifts_match() {
        for n in [1usize, 5, 100, 1001, 5000] {
            same_bits(|ctx| {
                let x = Array::from_slice_on(ctx, &values(n, 3, -1.0, 1.0), n).unwrap();
                let shifted = x.shift(1) + x.shift(-2) - &x * 2.0;
                let tail = if n > 3 { (x.slice(0, 1..) - x.slice(0, ..n - 1)).to_vec().unwrap() } else { Vec::new() };
                let (rows, cols) = (n.min(37), 11);
                let col = Array::from_slice_on(ctx, &values(rows, 4, 0.0, 1.0), [rows, 1]).unwrap();
                let row = Array::from_slice_on(ctx, &values(cols, 5, 0.0, 1.0), [1, cols]).unwrap();
                let grid = Array::from_slice_on(ctx, &values(rows * cols, 6, -1.0, 1.0), [rows, cols]).unwrap();
                let broadcast = &col * &row + &grid;
                let moved = grid.shift([1isize, -2]) + grid.transpose().transpose() + grid.flip(1);
                vec![
                    f32_bits(&shifted),
                    tail.into_iter().map(|v| u64::from(v.to_bits())).collect(),
                    f32_bits(&broadcast),
                    f32_bits(&moved),
                ]
            });
        }
    }

    #[test]
    fn several_outputs_match() {
        for n in [1usize, 9, 333, 4096, 9999] {
            same_bits(|ctx| {
                let x = Array::from_slice_on(ctx, &values(n, 7, -1.0, 1.0), n).unwrap();
                let y = Array::from_slice_on(ctx, &values(n, 8, 0.1, 1.0), n).unwrap();
                let common = (&x * &x + &y).sqrt();
                let (a, b, c) = (&common * &x, &common / &y, select(&x.gt(0.0), &common, &(&y - 1.0)));
                eval_all(&[&a, &b, &c]).unwrap();
                vec![f32_bits(&a), f32_bits(&b), f32_bits(&c)]
            });
        }
    }

    #[test]
    fn integers_match() {
        for n in [1usize, 6, 257, 5000] {
            same_bits(|ctx| {
                let a = Array::from_slice_on(ctx, &ints(n, 9), n).unwrap();
                let b = Array::from_slice_on(ctx, &ints(n, 10), n).unwrap();
                let r = (&a * 3 + &b) ^ &a;
                let m = select(&a.gt(&b), &(&a - &b), &(&b * 2));
                let u = Array::from_slice_on(ctx, &ints(n, 11).into_iter().map(|v| v as u32).collect::<Vec<_>>(), n).unwrap();
                let w = (&u * 2654435761u32) ^ (&u + 7u32);
                vec![
                    r.to_vec().unwrap().into_iter().map(|v| v as u32 as u64).collect(),
                    m.to_vec().unwrap().into_iter().map(|v| v as u32 as u64).collect(),
                    w.to_vec().unwrap().into_iter().map(u64::from).collect(),
                ]
            });
        }
    }

    #[test]
    fn double_floats_match() {
        for n in [1usize, 31, 1000, 3001] {
            same_bits(|ctx| {
                let x = Array::from_slice_on(ctx, &values(n, 12, 0.5, 2.0), n).unwrap().to_df32();
                let r = (&x * Df32::from(0.9) + Df32::from(0.2)).sqrt() * &x;
                vec![r.to_vec().unwrap().into_iter().map(|v| v.to_f64().to_bits()).collect()]
            });
        }
    }

    #[test]
    fn an_array_written_in_place_matches() {
        for n in [5usize, 777, 5000] {
            same_bits(|ctx| {
                let mut x = Array::from_slice_on(ctx, &values(n, 13, -1.0, 1.0), n).unwrap();
                let y = Array::from_slice_on(ctx, &values(n, 14, -1.0, 1.0), n).unwrap();
                x += &(&y * 2.0);
                x *= &(&y + 3.0);
                vec![f32_bits(&x)]
            });
        }
    }

    #[test]
    fn the_planner_keeps_one_element_per_invocation_where_it_cannot_do_more() {
        let ctx = context(Coarsen::Fixed(4));
        let x = Array::from_slice_on(&ctx, &values(1000, 15, -1.0, 1.0), 1000).unwrap();
        // A `bool` result is written by words: one invocation per word, never coarsened.
        let before = ctx.jit().stats().coarsened;
        let mask = x.gt(0.0);
        mask.eval().unwrap();
        assert_eq!(ctx.jit().stats().coarsened, before);
        let off = context(Coarsen::Off);
        let y = Array::from_slice_on(&off, &values(1000, 15, -1.0, 1.0), 1000).unwrap();
        (&y + 1.0).eval().unwrap();
        assert_eq!(off.jit().stats().coarsened, 0);
    }
}
