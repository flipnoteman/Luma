//! `luma-bench`: workload benchmark suites for Luma.
//!
//! ```text
//! cargo run --release -p luma-bench -- [--quick] [--suite ml,cfd,...] [--only <substring>] [--json out.jsonl] [--iters N]
//! ```
//!
//! `--iters N` times every selected case with exactly N calls (no calibration, no profile run); it is for
//! `tools/hosttime`, which counts instructions under callgrind.
//!
//! `LUMA_BENCH_DECISIONS=<path>` also writes the planner's decisions of every case as JSONL, cold and
//! warm, after the timed runs (pinned `LUMA_CALIBRATE=0`, `LUMA_TELEMETRY=off`); `tools/decisiondiff`
//! compares two such files.
//!
//! The backend comes from the environment (`LUMA_BACKEND=cuda|wgpu|metal|cpu`, `LUMA_DEVICE`).
//! Every case is verified against a host reference before it is timed. `tools/benchcmp.py A B`
//! compares two JSON files and flags regressions; `tools/benchref.sh` runs a luma ref through
//! the GPU queue.

mod harness;
mod suites;

use luma::*;

fn main() {
    let mut quick = false;
    let mut only = None;
    let mut json = None;
    let mut suites: Option<Vec<String>> = None;
    let mut iters: Option<usize> = None;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--quick" => quick = true,
            "--only" => only = args.next(),
            "--json" => json = args.next(),
            "--suite" => suites = args.next().map(|s| s.split(',').map(str::to_string).collect()),
            "--iters" => iters = Some(args.next().and_then(|v| v.parse().ok()).filter(|n| *n > 0).expect("--iters takes a positive integer")),
            "--list" => {
                for (name, _) in suites::ALL {
                    println!("{name}");
                }
                return;
            }
            other => {
                eprintln!("unknown argument {other}");
                std::process::exit(2);
            }
        }
    }
    let decisions = std::env::var("LUMA_BENCH_DECISIONS").ok().filter(|p| !p.is_empty());
    if decisions.is_some() {
        // Pinned, so that choices near a threshold cannot flip between two runs being compared.
        // SAFETY: nothing else runs yet (no threads have been started).
        unsafe {
            std::env::set_var("LUMA_CALIBRATE", "0");
            std::env::set_var("LUMA_TELEMETRY", "off");
        }
    }
    let ctx = Context::with_options(ContextOptions::from_env()).expect("no device");
    println!("luma-bench: backend {:?}, device {}, quick {quick}", ctx.backend(), ctx.device_name());
    let mut h = harness::Harness::new(ctx, quick, only, json.as_deref());
    h.iters = iters;
    if let Some(path) = decisions {
        h.log_decisions(&path);
    }
    for (name, run) in suites::ALL {
        if suites.as_ref().is_some_and(|s| !s.iter().any(|x| x == name)) {
            continue;
        }
        h.suite = name.to_string();
        run(&mut h);
    }
    if !h.skipped.is_empty() {
        println!("\n{} case(s) skipped:", h.skipped.len());
        for (name, why) in &h.skipped {
            println!("  {name}: {why}");
        }
        std::process::exit(1);
    }
}
