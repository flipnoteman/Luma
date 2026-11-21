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

use crate::builtin::UnOp;
use crate::context::Context;
use crate::tuning::Coarsen;

use super::program::{FusedKey, Tail};
use super::{ElemTy, OpKind, ShiftKind};

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
    let tuning = ctx.tuning();
    match tuning.coarsen {
        Coarsen::Off => 1,
        Coarsen::Fixed(n) => n.clamp(1, MAX_COARSEN),
        Coarsen::Auto => {
            // The grid is judged before the program is looked at: a program is scanned only for
            // launches big enough to be coarsened.
            let waves = len as f64 / (f64::from(tuning.compute_units.max(1)) * RESIDENT_PER_UNIT);
            if waves < MIN_WAVES {
                return 1;
            }
            pick(&Features::of(key), tuning.max_live)
        }
    }
}

/// Invocations a multiprocessor keeps resident (1536 on the NVIDIA GPUs of the Ampere generation,
/// 2048 and 1024 on others): what a wave of the grid is made of.
const RESIDENT_PER_UNIT: f64 = 1536.0;

/// Fewest waves of invocations a launch has before it is coarsened: with less, the invocations
/// that remain do not hide the latency of the longer chain each of them runs (a chain of 256
/// operations over a million elements ran 12 % slower with 4 elements per invocation).
const MIN_WAVES: f64 = 16.0;

/// What of a program decides its coarsening (see [`pick`]).
#[derive(Debug, Default, PartialEq)]
struct Features {
    /// Arithmetic operations on `f64`: the double-precision pipe is a fraction of the single-precision
    /// one and a chain of dependent operations leaves it idle with one element per invocation.
    wide_ops: usize,
    /// Gathers (`take`): the load of the element waits for the load of its index.
    gathers: usize,
    /// Most values live at once in the main program.
    live: usize,
}

impl Features {
    fn of(key: &FusedKey) -> Features {
        let mut features = Features { live: peak_live(key), ..Features::default() };
        for instr in key.instrs.iter().chain(key.subs.iter().flatten()) {
            match instr.op.kind {
                OpKind::Unary(op) if op != UnOp::Id && instr.op.ty == ElemTy::F64 => features.wide_ops += 1,
                OpKind::Binary(_) | OpKind::Clamp | OpKind::Powf if instr.op.ty == ElemTy::F64 => features.wide_ops += 1,
                OpKind::Shift(ShiftKind::Take { .. } | ShiftKind::ParamTake { .. }) => features.gathers += 1,
                _ => {}
            }
        }
        features
    }
}

/// Most values the main program of `key` keeps live at once, the loaded inputs and the outputs it
/// holds until the end included.
fn peak_live(key: &FusedKey) -> usize {
    let last = super::launch::last_uses(key);
    let mut dying = vec![0usize; key.instrs.len()];
    let (mut now, mut peak) = (0usize, 0usize);
    for k in 0..key.instrs.len() {
        now -= dying[k];
        if last[k] > k {
            now += 1;
            if let Some(slot) = dying.get_mut(last[k]) {
                *slot += 1;
            }
        }
        peak = peak.max(now);
    }
    peak + key.input_tys.len()
}

/// The elements per invocation for a program with `features` over a grid that is big enough (see [`MIN_WAVES`]).
///
/// Measured on an RTX 3090 and an RTX 3080 Laptop (see the changelog): a coarsened kernel gains where
/// the invocations wait: a chain of `f64` operations (the pipe is fed by independent elements, not
/// by one element's chain: 4.2x at 8 elements for 16 operations) and gathers (the loads of several
/// elements are issued together: bilinear resampling 13 % faster). A kernel bound by memory
/// bandwidth runs the same speed, so the planner leaves those alone rather than pay for the longer
/// kernels and the registers of every copy: the most elements it takes keep `copies * live` within
/// the registers `max_live` allows.
fn pick(features: &Features, max_live: usize) -> u8 {
    let wanted: u8 = if features.wide_ops >= WIDE_OPS {
        8
    } else if features.gathers >= GATHERS {
        4
    } else {
        1
    };
    let mut count = wanted;
    while count > 1 && usize::from(count) * features.live.max(1) > max_live {
        count /= 2;
    }
    count
}

/// Operations on `f64` from which a kernel is coarsened.
const WIDE_OPS: usize = 8;
/// Gathers from which a kernel is coarsened.
const GATHERS: usize = 2;

#[cfg(test)]
mod tests {
    use super::{pick, Features};
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
    fn the_planner_picks_by_what_the_kernel_waits_for() {
        let features = |wide_ops, gathers, live| Features { wide_ops, gathers, live };
        assert_eq!(pick(&features(0, 0, 4), 96), 1, "a streaming kernel is left alone");
        assert_eq!(pick(&features(7, 1, 4), 96), 1);
        assert_eq!(pick(&features(16, 0, 4), 96), 8, "a chain of f64 operations");
        assert_eq!(pick(&features(0, 4, 4), 96), 4, "gathers");
        assert_eq!(pick(&features(16, 4, 4), 96), 8);
        assert_eq!(pick(&features(16, 0, 30), 96), 2, "registers for no more copies than 96 values allow");
        assert_eq!(pick(&features(16, 0, 100), 96), 1);
    }

    /// A tuning that takes a launch of 100 000 elements for many waves.
    fn auto_context() -> Context {
        let ctx = Context::with_tuning(ContextOptions::from_env().calibrate(false), |tuning| {
            tuning.coarsen = Coarsen::Auto;
            tuning.compute_units = 1;
        })
        .unwrap();
        ctx.set_cost_model(CostModel::TEST);
        ctx
    }

    #[test]
    fn auto_coarsens_a_chain_of_double_floats_and_not_a_streaming_kernel() {
        let ctx = auto_context();
        let n = 100_000;
        let xs: Vec<f64> = values(n, 20, 0.5, 1.5).into_iter().map(f64::from).collect();
        let x = Array::<f64>::from_slice_on(&ctx, &xs, n).unwrap();
        let before = ctx.jit().stats().coarsened;
        let mut z = x.clone();
        for i in 0..12 {
            z = if i % 2 == 0 { &z * 0.999 } else { &z + 0.01 };
        }
        let got = z.to_vec().unwrap();
        assert_eq!(ctx.jit().stats().coarsened, before + 1);
        let want: Vec<f64> = xs.iter().map(|&v| (0..12).fold(v, |v, i| if i % 2 == 0 { v * 0.999 } else { v + 0.01 })).collect();
        assert_eq!(got, want);
        let before = ctx.jit().stats().coarsened;
        let y = Array::from_slice_on(&ctx, &values(n, 21, 0.0, 1.0), n).unwrap();
        (&y * 2.0 + 1.0).eval().unwrap();
        assert_eq!(ctx.jit().stats().coarsened, before, "a streaming kernel keeps one element per invocation");
        // Below the waves a launch needs, nothing is coarsened.
        let small = Array::<f64>::from_slice_on(&ctx, &xs[..1000], 1000).unwrap();
        let before = ctx.jit().stats().coarsened;
        let mut z = small.clone();
        for i in 0..12 {
            z = if i % 2 == 0 { &z * 0.999 } else { &z + 0.01 };
        }
        z.eval().unwrap();
        assert_eq!(ctx.jit().stats().coarsened, before);
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
