//! Cost of dispatching many small elementwise operations: waiting for the GPU after every
//! operation (how a library without batching behaves) vs. recording them all and waiting once.

use luma::*;
use std::time::Instant;

/// `ops` dependent operations on `x`, synchronizing after every one of them.
fn sync_each(x: &Array<f32>, ops: usize) -> Result<f32> {
    let mut y = x * 1.0;
    for i in 0..ops {
        y += (i % 3) as f32 * 0.001 + 0.5;
        y.eval()?;
    }
    y.sum()
}

/// The same operations, synchronizing once at the end.
fn batched(x: &Array<f32>, ops: usize) -> Result<f32> {
    let mut y = x * 1.0;
    for i in 0..ops {
        y += (i % 3) as f32 * 0.001 + 0.5;
    }
    y.sum()
}

fn time(iterations: u32, mut f: impl FnMut() -> Result<f32>) -> Result<(f64, f32)> {
    let mut last = f()?; // warm up pipelines and the buffer pool
    let start = Instant::now();
    for _ in 0..iterations {
        last = f()?;
    }
    Ok((start.elapsed().as_secs_f64() / f64::from(iterations), last))
}

fn main() -> Result<()> {
    println!("Batching benchmark: N small elementwise ops per synchronization");
    println!();
    println!("{}", "=".repeat(86));
    println!(
        "{:<12} {:<8} {:<18} {:<18} {:<10}",
        "Elements", "Ops", "Sync each (us/op)", "Batched (us/op)", "Speedup"
    );
    println!("{}", "-".repeat(86));

    for &len in &[1_024usize, 65_536, 1_048_576] {
        let data: Vec<f32> = (0..len).map(|i| (i % 7) as f32).collect();
        let x = Array::from_slice(&data, len)?;
        for &ops in &[10usize, 100, 1000] {
            let iterations = if ops >= 1000 { 3 } else { 10 };
            let (each, a) = time(iterations, || sync_each(&x, ops))?;
            let (batch, b) = time(iterations, || batched(&x, ops))?;
            // Both orders of work do the same arithmetic.
            assert!(((a - b) / a).abs() < 1e-3, "results differ: {a} vs {b}");
            println!(
                "{:<12} {:<8} {:<18.2} {:<18.2} {:.1}x",
                len,
                ops,
                each / ops as f64 * 1e6,
                batch / ops as f64 * 1e6,
                each / batch
            );
        }
    }
    println!("{}", "=".repeat(86));
    Ok(())
}
