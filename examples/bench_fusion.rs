//! Elementwise fusion: chains of 2/5/10/20 dependent operations on 1M and 16M elements, run as
//! one kernel per operation ("eager": a context whose fusion limit is one operation per kernel,
//! `LUMA_FUSION_MAX_NODES=1`) and fused into a single kernel (the default).
//!
//! Every operation reads one array and writes one array, so an eager chain of `k` operations
//! moves `2 * k * N * 4` bytes through memory and the fused chain `2 * N * 4`. The GB/s columns
//! are those *actual* byte counts divided by the measured time; "eff." divides the eager byte
//! count by the fused time, i.e. what bandwidth an unfused library would need to match it.

use luma::*;
use std::time::Instant;

/// `ops` dependent elementwise operations on `x` (scalar arithmetic and math functions,
/// chosen to keep values finite).
fn chain(x: &Array<f32>, ops: usize) -> Array<f32> {
    let mut y = x * 1.0001;
    for i in 1..ops {
        y = match i % 5 {
            0 => y.abs(),
            1 => y + 0.5,
            2 => y * 0.9999,
            3 => y.sqrt(),
            _ => y.maximum(0.25),
        };
    }
    y
}

/// Median time in milliseconds to build the chain and evaluate it (waiting for the GPU).
fn time_ms(x: &Array<f32>, ops: usize, iterations: usize) -> Result<f64> {
    for _ in 0..3 {
        chain(x, ops).eval()?; // compile pipelines, warm the buffer pool
    }
    let mut samples = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let start = Instant::now();
        let y = chain(x, ops);
        y.eval()?;
        y.context().sync()?;
        samples.push(start.elapsed().as_secs_f64() * 1e3);
    }
    samples.sort_by(f64::total_cmp);
    Ok(samples.get(samples.len() / 2).copied().unwrap_or(f64::NAN))
}

fn gb_per_s(bytes: f64, ms: f64) -> f64 {
    bytes / (ms * 1e-3) / 1e9
}

fn main() -> Result<()> {
    // The fusion limit is read when a context is created.
    // SAFETY: no other thread exists yet, so nothing reads the environment concurrently.
    unsafe { std::env::set_var("LUMA_FUSION_MAX_NODES", "1") };
    let eager_ctx = Context::new()?;
    // SAFETY: the only other threads are the first context's, which do not read the environment.
    unsafe { std::env::remove_var("LUMA_FUSION_MAX_NODES") };
    let fused_ctx = Context::new()?;

    println!("Elementwise fusion: eager (one kernel per operation) vs fused (one kernel per chain)");
    println!();
    let line = "=".repeat(112);
    println!("{line}");
    println!(
        "{:<10} {:<5} {:>12} {:>12} {:>9} {:>14} {:>14} {:>14} {:>9}",
        "Elements", "Ops", "Eager (ms)", "Fused (ms)", "Speedup", "Eager GB/s", "Fused GB/s", "Fused eff.", "Max diff"
    );
    println!("{}", "-".repeat(112));

    for &len in &[1_048_576usize, 16_777_216] {
        let data: Vec<f32> = (0..len).map(|i| (i % 97) as f32 * 0.1 + 0.2).collect();
        let eager_x = Array::from_slice_on(&eager_ctx, &data, len)?;
        let fused_x = Array::from_slice_on(&fused_ctx, &data, len)?;
        let iterations = if len > 4_000_000 { 9 } else { 25 };
        for &ops in &[2usize, 5, 10, 20] {
            let eager_ms = time_ms(&eager_x, ops, iterations)?;
            let fused_ms = time_ms(&fused_x, ops, iterations)?;

            // Both must compute the same thing (fusion may change rounding slightly).
            let (a, b) = (chain(&eager_x, ops).to_vec()?, chain(&fused_x, ops).to_vec()?);
            let max_diff = a.iter().zip(&b).map(|(p, q)| (p - q).abs()).fold(0.0f32, f32::max);

            let elem_bytes = len as f64 * 4.0;
            let eager_bytes = 2.0 * ops as f64 * elem_bytes;
            let fused_bytes = 2.0 * elem_bytes;
            println!(
                "{:<10} {:<5} {:>12.3} {:>12.3} {:>8.2}x {:>14.1} {:>14.1} {:>14.1} {:>9.1e}",
                len,
                ops,
                eager_ms,
                fused_ms,
                eager_ms / fused_ms,
                gb_per_s(eager_bytes, eager_ms),
                gb_per_s(fused_bytes, fused_ms),
                gb_per_s(eager_bytes, fused_ms),
                max_diff
            );
        }
    }
    println!("{line}");
    Ok(())
}
