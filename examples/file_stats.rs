//! Mean and standard deviation of a file that is larger than the memory budget of the device.
//!
//! The file holds `LEN` (default 2^26, 256 MiB) `f32` values. The context keeps at most 64 MiB of
//! array data on the device, so every pass reads the file in pieces: nothing but one pass's
//! segments is ever resident. The array has fewer than 2^32 elements, so `mean` and `std` work
//! (a larger one allows `sum`, `min` and `max` only; see the guide chapter "File-backed arrays").
//!
//! `LUMA_BACKEND=cpu cargo run --release --example file_stats [LEN]`

use std::io::Write;

use luma::{Array, Bias, Context, ContextOptions, FileOptions};

fn main() -> luma::Result<()> {
    let len: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(1 << 26);
    let path = std::env::temp_dir().join(format!("luma-file-stats-{}.f32", std::process::id()));

    // The data: a slow sine wave with an offset, written in 16 MiB pieces.
    let value = |i: usize| 3.0 + (i as f64 * 1e-4).sin();
    let (mut sum, mut square) = (0.0f64, 0.0f64);
    {
        let mut out = std::io::BufWriter::new(std::fs::File::create(&path).map_err(|e| luma::LumaError::Gpu(e.to_string()))?);
        for start in (0..len).step_by(1 << 22) {
            let piece: Vec<f32> = (start..len.min(start + (1 << 22))).map(|i| value(i) as f32).collect();
            for &v in &piece {
                sum += f64::from(v);
                square += f64::from(v) * f64::from(v);
            }
            out.write_all(bytemuck::cast_slice(&piece)).map_err(|e| luma::LumaError::Gpu(e.to_string()))?;
        }
        out.flush().map_err(|e| luma::LumaError::Gpu(e.to_string()))?;
    }

    let ctx = Context::with_options(ContextOptions::from_env().memory_budget(64 << 20))?;
    let x = Array::<f32>::open_on(&ctx, &path, FileOptions::raw().shape(len))?;
    println!("{} on {:?}: {len} values ({} MiB) through a {} MiB budget", ctx.backend(), ctx.device(), (len * 4) >> 20, 64);

    let started = std::time::Instant::now();
    let (mean, var) = x.meanvar(Bias::Population);
    let (mean, std) = (mean.item()?, var.sqrt().item()?);
    let elapsed = started.elapsed();

    let host_mean = sum / len as f64;
    let host_std = (square / len as f64 - host_mean * host_mean).sqrt();
    let stats = ctx.paging_stats();
    println!("mean {mean:.6} (host {host_mean:.6}), std {std:.6} (host {host_std:.6})");
    println!("{:.2} s, {} MiB read from the file, {} clean segments dropped", elapsed.as_secs_f64(), stats.io_read_bytes >> 20, stats.home_drops);
    assert!((f64::from(mean) - host_mean).abs() < 1e-3 && (f64::from(std) - host_std).abs() < 1e-3, "the statistics disagree with the host's");

    drop(x);
    let _ = std::fs::remove_file(&path);
    Ok(())
}
