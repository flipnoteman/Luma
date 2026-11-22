//! Cold compile cost of one kernel per class (elementwise, stencil, reduction, scan, matmul): IR size,
//! NVVM time and module load time per kernel, as luma-cuda logs them, and the first-call time of each.
//! Run with an empty `LUMA_CACHE_DIR` and `CUDA_CACHE_DISABLE=1` for cold caches, `LUMA_CUDA_DUMP=<dir>` to
//! keep the IR and PTX.
//!
//! `cargo run --release --example bench_cudair`

use luma::{Array, PadMode};
use std::time::Instant;

struct Print;

impl log::Log for Print {
    fn enabled(&self, m: &log::Metadata) -> bool {
        m.level() <= log::Level::Debug
    }
    fn log(&self, r: &log::Record) {
        let line = r.args().to_string();
        if line.starts_with("luma-cuda:") {
            println!("  {line}");
        }
    }
    fn flush(&self) {}
}

fn case(name: &str, f: impl FnOnce() -> Vec<f32>) {
    println!("{name}");
    let t0 = Instant::now();
    let v = f();
    println!("  first call {:.1} ms (value {})", t0.elapsed().as_secs_f64() * 1e3, v.first().copied().unwrap_or(0.0));
}

fn main() {
    let _ = log::set_logger(&Print).map(|()| log::set_max_level(log::LevelFilter::Debug));
    let ctx = luma::default_context().unwrap();
    let n = 1 << 20;
    let host: Vec<f32> = (0..n).map(|i| 1.0 + 0.1 * (i as f32 * 0.37).sin()).collect();
    let x = Array::from_slice_on(&ctx, &host, [n]).unwrap();
    let m = 512;
    let a = Array::from_slice_on(&ctx, &host[..m * m], [m, m]).unwrap();
    let b = Array::from_slice_on(&ctx, &host[m * m..2 * m * m], [m, m]).unwrap();
    ctx.sync().unwrap();
    case("elementwise", || {
        let mut acc = &x * 1.5;
        for j in 0..12 {
            acc = match j % 4 {
                0 => acc.sin() + &x,
                1 => &acc * &x + 0.25,
                2 => (&acc * &acc + 1.0).sqrt(),
                _ => acc.abs() - &x * 0.5,
            };
        }
        acc.to_vec().unwrap()
    });
    case("stencil", || {
        let l = x.slice(0, 0..n - 2);
        let c = x.slice(0, 1..n - 1);
        let r = x.slice(0, 2..n);
        (&l + &c * 2.0 + &r).pad([(1, 1)], PadMode::Edge).to_vec().unwrap()
    });
    case("reduction (sum)", || x.sum().to_vec().unwrap());
    case("reduction (max of sin)", || x.sin().max().to_vec().unwrap());
    case("reduction (sum axis 1)", || a.sum_axis(1).to_vec().unwrap());
    case("scan (cumsum)", || x.cumsum(0).to_vec().unwrap());
    case("matmul 512", || a.matmul(&b).to_vec().unwrap());
}
