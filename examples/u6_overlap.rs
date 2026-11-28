//! Measures a chain over an array above the memory budget, kernel-major (`LUMA_SPLIT=off` path) against schedules with rings.
//! `cargo run --release --example u6_overlap [MiB of array] [budget MiB] [rounds]`
use std::time::Instant;

use luma::{Array, Context, ContextOptions, SplitMode};

fn chain(ctx: &Context, xs: &[f32], rounds: usize) -> (f64, Vec<f32>) {
    let n = xs.len();
    let x = Array::from_slice_on(ctx, xs, n).unwrap();
    ctx.sync().unwrap();
    let start = Instant::now();
    let mut y = x.clone();
    for k in 0..rounds {
        y = (&y * 1.0009 + (k % 5) as f32 * 0.25).sin().abs() + &x * 0.5;
    }
    let out = y.to_vec().unwrap();
    (start.elapsed().as_secs_f64(), out)
}

fn main() {
    let args: Vec<usize> = std::env::args().skip(1).filter_map(|a| a.parse().ok()).collect();
    let (mib, budget, rounds) = (args.first().copied().unwrap_or(1024), args.get(1).copied().unwrap_or(256), args.get(2).copied().unwrap_or(8));
    let n = mib * (1 << 20) / 4;
    let xs: Vec<f32> = (0..n).map(|i| (i % 1013) as f32 * 0.01).collect();
    let mut reference = None;
    for (label, mode) in [("off", SplitMode::Off), ("auto", SplitMode::Auto)] {
        let options = ContextOptions::from_env().memory_budget((budget as u64) << 20).split(mode);
        let ctx = Context::with_options(options).expect("a device");
        let mut times = Vec::new();
        for _ in 0..3 {
            let (t, out) = chain(&ctx, &xs, rounds);
            times.push(t);
            match &reference {
                None => reference = Some(out),
                Some(want) => assert!(want.iter().zip(&out).all(|(a, b)| a.to_bits() == b.to_bits()), "{label} differs"),
            }
        }
        times.sort_by(f64::total_cmp);
        let stats = ctx.stream_stats();
        let paging = ctx.paging_stats();
        println!(
            "{label:5} median {:.3}s  array {mib} MiB budget {budget} MiB rounds {rounds}  uploaded {} MiB spilled {} MiB  {stats:?}",
            times[1],
            paging.uploaded_bytes >> 20,
            ctx.spilled_bytes() >> 20
        );
    }
}
