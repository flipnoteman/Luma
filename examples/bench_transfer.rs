//! Host <-> device transfer timings from 1 KB to 1 GB, and the fixed latency of tiny readbacks.
//! `cargo run --release --example bench_transfer -- [filter] [max_bytes] [min_bytes]`
//!
//! Every case reports the first call ("cold": pinned staging memory and device buffers are still
//! to be allocated) and the median of the later ones ("warm"). The host data is allocated once,
//! like a NumPy array a program uploads repeatedly.

use luma::*;
use luma_core::{Backend, Buffer, StageDir};
use std::hint::black_box;
use std::time::{Duration, Instant};

#[path = "../tests/all/common/lanes.rs"]
mod lanes;

/// Runs `f` once ("cold"), then until about `budget` has passed and at least `min_reps` more times.
fn measure(budget: Duration, min_reps: usize, mut f: impl FnMut()) -> (f64, f64) {
    let t = Instant::now();
    f();
    let cold = t.elapsed().as_secs_f64() * 1e3;
    let mut samples = Vec::new();
    let start = Instant::now();
    while samples.len() < min_reps || (start.elapsed() < budget && samples.len() < 4000) {
        let t = Instant::now();
        f();
        samples.push(t.elapsed().as_secs_f64() * 1e3);
    }
    samples.sort_by(f64::total_cmp);
    (cold, samples[samples.len() / 2])
}

fn human(bytes: usize) -> String {
    match bytes {
        b if b >= 1 << 30 => format!("{} GB", b >> 30),
        b if b >= 1 << 20 => format!("{} MB", b >> 20),
        b if b >= 1 << 10 => format!("{} KB", b >> 10),
        b => format!("{b} B"),
    }
}

fn line(case: &str, bytes: usize, (cold, med): (f64, f64)) {
    let gbs = bytes as f64 / (med * 1e-3) / 1e9;
    println!("{case:<14} {:>7}  cold {cold:>9.3} ms   med {med:>9.3} ms   {gbs:>7.2} GB/s", human(bytes));
}

fn main() -> Result<()> {
    let filter = std::env::args().nth(1).unwrap_or_default();
    let max: usize = std::env::args().nth(2).map_or(1 << 30, |a| a.parse().unwrap());
    let min: usize = std::env::args().nth(3).map_or(1 << 10, |a| a.parse().unwrap());
    let want = |name: &str| filter.is_empty() || name.contains(&filter);
    let ctx = default_context()?;
    println!("device: {}", ctx.device_name());
    let budget = Duration::from_millis(600);

    if want("small") {
        let n = 1 << 16;
        let xs: Vec<f32> = (0..n).map(|i| (i % 97) as f32 * 0.01).collect();
        let x = Array::from_slice_on(&ctx, &xs, n)?;
        x.eval()?;
        ctx.sync()?;
        let mut c = 1.0f32;
        let r = measure(budget, 200, || {
            c += 1e-3;
            black_box((&x * c).sum().item().unwrap());
        });
        println!("{:<14} {:>7}  cold {:>9.3} ms   med {:>9.3} ms", "item(sum)", "64K f32", r.0, r.1);
        let y = Array::from_slice_on(&ctx, &xs[..16], 16)?;
        y.eval()?;
        ctx.sync()?;
        let r = measure(budget, 200, || {
            black_box(y.to_vec().unwrap());
        });
        println!("{:<14} {:>7}  cold {:>9.3} ms   med {:>9.3} ms", "to_vec", "16 f32", r.0, r.1);
        let z = Array::from_slice_on(&ctx, &xs[..4096], 4096)?;
        z.eval()?;
        ctx.sync()?;
        let r = measure(budget, 200, || {
            black_box(z.to_vec().unwrap());
        });
        println!("{:<14} {:>7}  cold {:>9.3} ms   med {:>9.3} ms", "to_vec", "4096 f32", r.0, r.1);
        let r = measure(budget, 200, || {
            c += 1e-3;
            (&z + c).eval().unwrap();
            ctx.sync().unwrap();
        });
        println!("{:<14} {:>7}  cold {:>9.3} ms   med {:>9.3} ms", "op+sync", "4096 f32", r.0, r.1);
        let r = measure(budget, 200, || {
            black_box(Array::from_slice_on(&ctx, &xs[..16], 16).unwrap().to_vec().unwrap());
        });
        println!("{:<14} {:>7}  cold {:>9.3} ms   med {:>9.3} ms", "up+to_vec", "16 f32", r.0, r.1);
    }

    if want("busy") {
        // An upload (and a readback) queued behind a long kernel: how much of the host's staging overlaps the GPU's work.
        let n = 1 << 24;
        let xs: Vec<f32> = (0..n).map(|i| (i % 97) as f32 * 0.01 + 0.2).collect();
        let x = Array::from_slice_on(&ctx, &xs, n)?;
        x.eval()?;
        ctx.sync()?;
        let up: usize = 64 << 20;
        let host: Vec<f32> = (0..up / 4).map(|i| (i % 89) as f32 * 0.02 + 0.1).collect();
        let mut c = 1.0f32;
        let heavy = |c: f32| {
            let mut y = &x * c;
            for _ in 0..60 {
                y = (&y * 1.0001 + 0.5).sin();
            }
            y.eval().unwrap();
            ctx.flush().unwrap();
            y
        };
        let h = measure(budget, 9, || {
            c += 1e-3;
            black_box(heavy(c));
            ctx.sync().unwrap();
        });
        println!("{:<14} {:>7}  cold {:>9.3} ms   med {:>9.3} ms", "busy:kernel", "", h.0, h.1);
        let u = measure(budget, 9, || {
            black_box(Array::from_slice_on(&ctx, &host, up / 4).unwrap());
            ctx.sync().unwrap();
        });
        println!("{:<14} {:>7}  cold {:>9.3} ms   med {:>9.3} ms", "busy:upload", human(up), u.0, u.1);
        let hu = measure(budget, 9, || {
            c += 1e-3;
            let y = heavy(c);
            black_box(Array::from_slice_on(&ctx, &host, up / 4).unwrap());
            ctx.sync().unwrap();
            black_box(y);
        });
        println!("{:<14} {:>7}  cold {:>9.3} ms   med {:>9.3} ms   (kernel + upload alone: {:.3} ms)", "busy:both", human(up), hu.0, hu.1, h.1 + u.1);
        let hr = measure(budget, 9, || {
            c += 1e-3;
            let y = heavy(c);
            black_box(y.to_vec().unwrap());
        });
        println!("{:<14} {:>7}  cold {:>9.3} ms   med {:>9.3} ms", "busy:to_vec", human(n * 4), hr.0, hr.1);
    }

    let mut bytes = min;
    while bytes <= max {
        let n = bytes / 4;
        let host: Vec<f32> = (0..n).map(|i| (i % 97) as f32 * 0.01 + 0.2).collect();
        let reps = if bytes >= 256 << 20 { 5 } else { 9 };
        if want("upload") {
            line("upload", bytes, measure(budget, reps, || {
                let a = Array::from_slice_on(&ctx, &host, n).unwrap();
                a.eval().unwrap();
                ctx.sync().unwrap();
                black_box(&a);
            }));
        }
        if want("to_vec") || want("read_into") {
            let dev = Array::from_slice_on(&ctx, &host, n)?;
            dev.eval()?;
            ctx.sync()?;
            if want("to_vec") {
                line("to_vec", bytes, measure(budget, reps, || {
                    black_box(dev.to_vec().unwrap());
                }));
            }
            if want("read_into") {
                let mut out = vec![0f32; n];
                line("read_into", bytes, measure(budget, reps, || {
                    dev.read_into(&mut out).unwrap();
                    black_box(&out);
                }));
            }
        }
        if want("roundtrip") {
            // LumaPy-like: asarray, compute, to_numpy of the result.
            let mut c = 1.0f32;
            line("roundtrip", 2 * bytes, measure(budget, reps, || {
                c += 1e-3;
                let a = Array::from_slice_on(&ctx, &host, n).unwrap();
                black_box((&a * c + 1.0).to_vec().unwrap());
            }));
        }
        if want("up+item") {
            // montecarlo-like: upload, a reduction, one scalar back.
            let mut c = 1.0f32;
            line("up+item", bytes, measure(budget, reps, || {
                c += 1e-3;
                let a = Array::from_slice_on(&ctx, &host, n).unwrap();
                black_box((&a * c).mean().item().unwrap());
            }));
        }
        bytes *= 4;
    }
    if want("overlap") {
        overlap(max.min(256 << 20));
    }
    Ok(())
}

/// Wall time of `f`.
fn wall(f: impl FnOnce()) -> Duration {
    let start = Instant::now();
    f();
    start.elapsed()
}

/// The transfer lane against the compute lane: upload slab k+1 while a kernel runs on slab k.
/// Prints the wall time of the pipeline against the sum of the two run alone (and against the
/// ideal, the longer of the two). Runs on the backend `LUMA_BACKEND` names, through the HAL.
fn overlap(slab: usize) {
    let Some(backend) = lanes::open() else { return println!("overlap: no backend") };
    let backend: &dyn Backend = &*backend;
    let words = (slab / 4) as u32;
    let bytes = u64::from(words) * 4;
    let buffers = [0, 1].map(|_| backend.create_buffer_unordered(bytes, lanes::USAGE).expect("a slab buffer"));
    let out = backend.create_buffer_unordered(bytes, lanes::USAGE).expect("an output buffer");
    let mut stage = backend.stage_alloc(bytes, StageDir::ToDevice).expect("a stage");
    backend.stage_ptr(&mut stage).fill(1);
    let upload = |dst: &Buffer, after: &[luma_core::Ticket]| backend.transfer_in(&stage, dst, 0, bytes, after).unwrap();
    let wait = |ticket: &luma_core::Ticket| backend.ticket_wait(ticket, Duration::ZERO).unwrap();
    let best = |mut f: Box<dyn FnMut() + '_>| (0..5).map(|_| wall(&mut f)).min().unwrap();

    // The transfer alone.
    wait(&upload(&buffers[0], &[]));
    let up = best(Box::new(|| wait(&upload(&buffers[0], &[]))));
    // A kernel as long as the upload (calibrated on a short one).
    let probe_iters = 200;
    let probe = lanes::affine_kernel(backend, probe_iters);
    let run = |kernel: &luma_core::CompiledKernel| {
        lanes::launch(backend, kernel, &buffers[0], &out, words);
        backend.sync(Duration::ZERO).unwrap();
    };
    run(&probe);
    let probe_time = best(Box::new(|| run(&probe)));
    let iters = ((f64::from(probe_iters) * up.as_secs_f64() / probe_time.as_secs_f64()) as u32).max(1);
    let kernel = lanes::affine_kernel(backend, iters);
    run(&kernel);
    let alone = best(Box::new(|| run(&kernel)));

    // The pipeline: the upload of slab k+1 (into the buffer slab k-1 used, after the kernel that
    // read it) runs while the kernel of slab k does.
    const SLABS: usize = 8;
    let pipeline = || {
        let mut loaded = upload(&buffers[0], &[]);
        let mut finished: Option<luma_core::Ticket> = None;
        for k in 0..SLABS {
            wait(&loaded);
            backend.barrier(std::slice::from_ref(&loaded)).unwrap();
            lanes::launch(backend, &kernel, &buffers[k % 2], &out, words);
            let done = backend.barrier(&[]).unwrap();
            if k + 1 < SLABS {
                let after: Vec<_> = finished.iter().cloned().collect();
                loaded = upload(&buffers[(k + 1) % 2], &after);
            }
            finished = Some(done);
        }
        backend.sync(Duration::ZERO).unwrap();
    };
    pipeline();
    let both = best(Box::new(pipeline));
    let serial = (up + alone) * SLABS as u32;
    let ideal = up.max(alone) * SLABS as u32;
    println!("overlap: {SLABS} slabs of {}", human(slab));
    println!("  upload alone  {:>9.3} ms/slab   kernel alone {:>9.3} ms/slab", up.as_secs_f64() * 1e3, alone.as_secs_f64() * 1e3);
    println!(
        "  pipeline      {:>9.3} ms   serial sum {:>9.3} ms   ideal {:>9.3} ms   wall/serial {:.2}",
        both.as_secs_f64() * 1e3,
        serial.as_secs_f64() * 1e3,
        ideal.as_secs_f64() * 1e3,
        both.as_secs_f64() / serial.as_secs_f64()
    );
}
