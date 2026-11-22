//! luma against ArrayFire 3.10 (CUDA), case by case, on the same GPU. See README.md.

mod afx;
mod bench;
mod cases;
mod cold;
mod linalg;
mod linalg_svd;
mod lumax;
mod mlp;
mod util;

use std::time::{Duration, Instant};

fn wake(ctx: &luma::Context) {
    // Keep both libraries' kernels busy for a moment so the first case does not time the clock ramp-up.
    let x = luma::Array::<f32>::linspace_on(ctx, 0.0, 1.0, 1 << 24).unwrap();
    let start = Instant::now();
    while start.elapsed() < Duration::from_millis(500) {
        let mut z = x.clone();
        for _ in 0..6 {
            z = (z * 1.0001f32 + 0.1f32).tan().cos();
        }
        z.eval().unwrap();
        ctx.sync().unwrap();
    }
}

fn main() {
    let mut args = std::env::args().skip(1);
    let (mut quick, mut only, mut out, mut reps) = (false, None::<String>, String::from("out"), 5usize);
    let mut mode = "run".to_string();
    let mut rest = Vec::new();
    while let Some(a) = args.next() {
        match a.as_str() {
            "--quick" => quick = true,
            "--only" => only = args.next(),
            "--out" => out = args.next().expect("--out DIR"),
            "--reps" => reps = args.next().and_then(|v| v.parse().ok()).expect("--reps N"),
            "cold" | "child" => mode = a,
            _ => rest.push(a),
        }
    }
    match mode.as_str() {
        "child" => return cold::child(&rest[0], &rest[1]),
        "cold" => {
            std::fs::create_dir_all(&out).unwrap();
            let md = cold::orchestrate(reps, only.as_deref());
            println!("\n{md}");
            std::fs::write(format!("{out}/cold.md"), md).unwrap();
            return;
        }
        _ => {}
    }
    afx::init();
    let ctx = lumax::open();
    let gpu = ctx.device_name().to_string();
    println!("luma: backend {:?}, device {gpu}", ctx.backend());
    println!("{}", afx::info());
    std::fs::create_dir_all(&out).unwrap();
    let mut b = bench::Bench::new(ctx.clone(), only, quick, Some(&format!("{out}/results.jsonl")));
    wake(&ctx);
    let groups: [(&str, fn(&mut bench::Bench)); 15] = [
        ("ew", cases::elementwise),
        ("small", cases::small_ops),
        ("reduce", cases::reductions),
        ("matmul", cases::matmuls),
        ("bmm", cases::batched),
        ("fft", cases::fft),
        ("sort", cases::sorts),
        ("scan", cases::scans),
        ("hist", cases::histograms),
        ("gather", cases::gathers),
        ("stencil", cases::stencil),
        ("filter", cases::filters),
        ("mlp", mlp::mlp),
        ("linalg", linalg::linalg),
        ("end", |_| ()),
    ];
    for (_, g) in groups {
        g(&mut b);
        // Return pooled memory of both libraries between groups so that one case's big buffers
        // cannot starve the next one.
        let _ = ctx.trim();
        arrayfire::device_gc();
    }
    let md = b.markdown(&format!("{gpu} - luma vs ArrayFire"));
    println!("\n{md}");
    std::fs::write(format!("{out}/results.md"), &md).unwrap();
    let bad: Vec<_> = b.rows.iter().filter(|r| !r.ok).collect();
    if !bad.is_empty() || !b.failures.is_empty() {
        println!("\n!!!!!! {} mismatching row(s), {} failed group(s):", bad.len(), b.failures.len());
        for r in bad {
            println!("!!!!!!   {} [{}]: {}", r.case, r.size, r.agree);
        }
        for (g, m) in &b.failures {
            println!("!!!!!!   group {g}: {m}");
        }
    }
}
