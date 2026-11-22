//! First-call (cold) times: the first call of an operation in a fresh process, which includes
//! building kernels (luma: tracing, NVVM, module load; ArrayFire: NVRTC for its JIT and its own
//! kernels, cuBLAS handle creation) and first-use allocations.
//!
//! Every measurement is its own process: `afcompare cold` spawns `afcompare child <lib> <case>`
//! processes, interleaved (luma, ArrayFire, luma, ...), and takes the median over repetitions.
//! A child opens its library, runs an unrelated warm-up kernel for 400 ms (so the GPU clocks are
//! up and the library's own initialisation is not in the number), uploads its inputs and times
//! one call of the case including the final synchronisation. Two cache states:
//!
//! * `cold`: every on-disk cache is empty or off (fresh `LUMA_CACHE_DIR` and
//!   `AF_JIT_KERNEL_CACHE_DIRECTORY`, `CUDA_CACHE_DISABLE=1`);
//! * `disk-warm`: the caches were filled by an earlier identical run (what a second launch of an
//!   application sees).

use std::path::PathBuf;
use std::process::Command;
use std::time::{Duration, Instant};

use arrayfire as af;

use crate::cases::{chain_a, chain_l, jacobi_a, jacobi_l, A, L, STENCIL_STEPS};
use crate::{afx, lumax, util};

pub const CASES: [(&str, &str); 4] = [
    ("ew8", "1 elementwise chain, 8 fused ops, 16M"),
    ("mm_f32_4096", "4 matmul f32, 4096 x 4096 x 4096"),
    ("mm_tf32_4096", "4 matmul TF32, 4096 x 4096 x 4096"),
    ("stencil", "11 stencil: 5-pt Jacobi, 2048 x 2048, 100 iterations"),
];

fn wake_luma(ctx: &luma::Context) {
    let x = luma::Array::<f32>::linspace_on(ctx, 0.0, 1.0, 1 << 24).unwrap();
    let start = Instant::now();
    while start.elapsed() < Duration::from_millis(400) {
        let mut z = x.clone();
        for _ in 0..6 {
            z = (z * 1.0001f32 + 0.1f32).tan().cos();
        }
        z.eval().unwrap();
        ctx.sync().unwrap();
    }
}

fn wake_af() {
    let x = af::iota::<f32>(afx::dims(&[1 << 24]), afx::dims(&[1]));
    let x = &x * (1.0f32 / (1 << 24) as f32);
    x.eval();
    let start = Instant::now();
    while start.elapsed() < Duration::from_millis(400) {
        let mut z = x.clone();
        for _ in 0..6 {
            z = af::cos(&af::tan(&(&(&z * 1.0001f32) + 0.1f32)));
        }
        z.eval();
        afx::sync();
    }
}

fn timed(f: &mut dyn FnMut()) -> f64 {
    let t = Instant::now();
    f();
    t.elapsed().as_secs_f64() * 1e3
}

/// The child: one first call of `case` in `lib`, printed as a `COLD` line.
pub fn child(lib: &str, case: &str) {
    let ms = match lib {
        "luma" => {
            let ctx = lumax::open();
            wake_luma(&ctx);
            let c = ctx.clone();
            match case {
                "ew8" => {
                    let x = lumax::arr(&ctx, &util::uniform(16_000_000, 1, 0.0, 1.0), 16_000_000usize);
                    timed(&mut || {
                        chain_l(&x, 8).eval().unwrap();
                        c.sync().unwrap();
                    })
                }
                "mm_f32_4096" | "mm_tf32_4096" => {
                    let n = 4096;
                    let a = lumax::arr(&ctx, &util::uniform(n * n, 11, -1.0, 1.0), [n, n]);
                    let b = lumax::arr(&ctx, &util::uniform(n * n, 12, -1.0, 1.0), [n, n]);
                    let f: fn(&L, &L) -> L = if case == "mm_f32_4096" { luma::matmul_f32 } else { luma::matmul_tf32 };
                    timed(&mut || {
                        f(&a, &b).eval().unwrap();
                        c.sync().unwrap();
                    })
                }
                "stencil" => {
                    let n = 2048;
                    let u = lumax::arr(&ctx, &util::uniform(n * n, 91, 0.0, 1.0), [n, n]);
                    let r = lumax::arr(&ctx, &util::uniform(n * n, 92, -0.005, 0.005), [n, n]);
                    timed(&mut || {
                        let [o] = luma::iterate_all([&u], STENCIL_STEPS, |s| [jacobi_l(&s[0], &r)]);
                        o.eval().unwrap();
                        c.sync().unwrap();
                    })
                }
                _ => panic!("unknown case {case}"),
            }
        }
        "af" => {
            afx::init();
            wake_af();
            match case {
                "ew8" => {
                    let x = afx::arr(&util::uniform(16_000_000, 1, 0.0, 1.0), &[16_000_000]);
                    timed(&mut || {
                        chain_a(&x, 8).eval();
                        afx::sync();
                    })
                }
                "mm_f32_4096" | "mm_tf32_4096" => {
                    let n = 4096u64;
                    let a: A = afx::arr(&util::uniform((n * n) as usize, 11, -1.0, 1.0), &[n, n]);
                    let b: A = afx::arr(&util::uniform((n * n) as usize, 12, -1.0, 1.0), &[n, n]);
                    let tf32 = case == "mm_tf32_4096";
                    timed(&mut || {
                        if tf32 {
                            afx::set_math_mode(afx::CUBLAS_TF32_TENSOR_OP_MATH).expect("math mode");
                        }
                        af::matmul(&b, &a, af::MatProp::NONE, af::MatProp::NONE).eval();
                        afx::sync();
                    })
                }
                "stencil" => {
                    let n = 2048u64;
                    let u = afx::arr(&util::uniform((n * n) as usize, 91, 0.0, 1.0), &[n, n]);
                    let r = afx::arr(&util::uniform((n * n) as usize, 92, -0.005, 0.005), &[n, n]);
                    timed(&mut || {
                        let mut cur = jacobi_a(&u, &r);
                        cur.eval();
                        for _ in 1..STENCIL_STEPS {
                            let next = jacobi_a(&cur, &r);
                            next.eval();
                            cur = next;
                        }
                        afx::sync();
                    })
                }
                _ => panic!("unknown case {case}"),
            }
        }
        _ => panic!("unknown lib {lib}"),
    };
    println!("COLD\t{lib}\t{case}\t{ms}");
}

fn run_child(exe: &PathBuf, lib: &str, case: &str, cache: &PathBuf, disable_cuda_cache: bool) -> Option<f64> {
    let mut cmd = Command::new(exe);
    cmd.args(["child", lib, case]).env("LUMA_BACKEND", "cuda").env("LUMA_CACHE_DIR", cache.join("luma")).env("AF_JIT_KERNEL_CACHE_DIRECTORY", cache.join("af"));
    if disable_cuda_cache {
        cmd.env("CUDA_CACHE_DISABLE", "1");
    }
    let out = cmd.output().expect("spawn child");
    let text = String::from_utf8_lossy(&out.stdout);
    let ms = text.lines().find_map(|l| l.strip_prefix("COLD\t")).and_then(|l| l.split('\t').nth(2)).and_then(|v| v.parse().ok());
    if ms.is_none() {
        eprintln!("child {lib} {case} failed: {}\n{}", String::from_utf8_lossy(&out.stderr).lines().last().unwrap_or(""), text.lines().last().unwrap_or(""));
    }
    ms
}

fn median(mut v: Vec<f64>) -> Option<f64> {
    if v.is_empty() {
        return None;
    }
    v.sort_by(f64::total_cmp);
    Some(v[v.len() / 2])
}

/// Spawns the children and returns the markdown table.
pub fn orchestrate(reps: usize, only: Option<&str>) -> String {
    let exe = std::env::current_exe().expect("current_exe");
    let base = std::env::temp_dir().join(format!("afcompare-cold-{}", std::process::id()));
    let mut table = String::new();
    table += "| case | luma cold ms | ArrayFire cold ms | ratio (AF/luma) | luma disk-warm ms | ArrayFire disk-warm ms | ratio (AF/luma) |\n|---|---:|---:|---:|---:|---:|---:|\n";
    for (case, label) in CASES {
        if only.is_some_and(|o| !case.contains(o)) {
            continue;
        }
        println!("== cold: {label}");
        let warm_dir = base.join(format!("warm-{case}"));
        std::fs::create_dir_all(&warm_dir).unwrap();
        // Fill the warm caches (and the driver's JIT cache) once, untimed.
        for lib in ["luma", "af"] {
            run_child(&exe, lib, case, &warm_dir, false);
        }
        let mut t = [Vec::new(), Vec::new(), Vec::new(), Vec::new()]; // luma cold, af cold, luma warm, af warm
        for rep in 0..reps {
            // Rotate the order of the four children so that no state always follows another.
            for k in 0..4 {
                let slot = (k + rep) % 4;
                let lib = if slot % 2 == 0 { "luma" } else { "af" };
                let ms = if slot < 2 {
                    let fresh = base.join(format!("cold-{case}-{slot}-{rep}"));
                    std::fs::create_dir_all(&fresh).unwrap();
                    let r = run_child(&exe, lib, case, &fresh, true);
                    let _ = std::fs::remove_dir_all(&fresh);
                    r
                } else {
                    run_child(&exe, lib, case, &warm_dir, false)
                };
                if let Some(ms) = ms {
                    t[slot].push(ms);
                }
            }
        }
        let m: Vec<Option<f64>> = t.iter().map(|v| median(v.clone())).collect();
        println!("  runs: luma cold {:?}\n        AF cold {:?}\n        luma disk-warm {:?}\n        AF disk-warm {:?}", t[0], t[1], t[2], t[3]);
        let f = |v: Option<f64>| v.map_or("n/a".to_string(), crate::bench::fmt_ms);
        let r = |a: Option<f64>, l: Option<f64>| match (a, l) {
            (Some(a), Some(l)) => format!("{:.2}", a / l),
            _ => "n/a".into(),
        };
        table += &format!("| {label} | {} | {} | {} | {} | {} | {} |\n", f(m[0]), f(m[1]), r(m[1], m[0]), f(m[2]), f(m[3]), r(m[3], m[2]));
    }
    let _ = std::fs::remove_dir_all(&base);
    format!(
        "Cold = first call in a fresh process, after the library's own initialisation and an unrelated 400 ms warm-up kernel; median of {reps} processes per cell, children interleaved. \
         `cold`: all on-disk kernel caches empty or off; `disk-warm`: caches filled by an earlier identical run. AF TF32 includes the cuBLAS math-mode call.\n\n{table}"
    )
}
