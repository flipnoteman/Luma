//! The measuring harness every suite uses.
//!
//! A case is verified once against a host reference by the suite (before it calls
//! [`Harness::case`]), then timed: the first call on its own (it includes building kernels,
//! tracing and first-use allocation), two warm-up calls, then timed calls (median and minimum),
//! then one profiled call for GPU time and launch count. Each case prints one table row and, with
//! `--json`, one JSON line (see [`Record`]); `tools/benchcmp.py` compares two such files.

use std::fmt::Write as _;
use std::io::Write as _;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::Mutex;
use std::time::Instant;

use luma::*;

/// The last panic message, so a failed case is reported instead of aborting the run.
pub static PANIC: Mutex<String> = Mutex::new(String::new());

/// What one call of a case does, for the throughput column.
#[derive(Clone, Copy, Debug)]
pub enum Work {
    /// Bytes an ideal implementation reads plus writes -> GB/s.
    Bytes(f64),
    /// Floating-point operations -> GFLOP/s.
    Flops(f64),
    /// Cell updates (cells x steps) -> Mcell-steps/s.
    Cells(f64),
    /// `n` items of the named kind -> M items/s.
    Items(f64, &'static str),
    /// No meaningful rate (latency cases).
    None,
}

/// One measured case, as written to the JSON file.
#[derive(Clone, Debug)]
pub struct Record {
    pub suite: String,
    pub case: String,
    pub size: String,
    /// The first call, which pays for kernel builds, tracing and first-use allocations.
    pub first_ms: f64,
    pub median_ms: f64,
    pub min_ms: f64,
    pub reps: usize,
    pub rate: Option<f64>,
    pub unit: &'static str,
    pub gpu_ms: Option<f64>,
    pub launches: Option<u32>,
    /// Device memory the context holds after the case (resident plus pooled), MiB.
    pub mem_mib: f64,
}

pub struct Harness {
    pub ctx: Context,
    pub quick: bool,
    /// Substring filter on `suite/case`.
    pub only: Option<String>,
    /// The suite being run; set by the runner.
    pub suite: String,
    pub records: Vec<Record>,
    pub skipped: Vec<(String, String)>,
    json: Option<std::io::BufWriter<std::fs::File>>,
    header_printed: bool,
}

impl Harness {
    pub fn new(ctx: Context, quick: bool, only: Option<String>, json: Option<&str>) -> Harness {
        let json = json.map(|p| std::io::BufWriter::new(std::fs::File::create(p).expect("cannot create the JSON file")));
        std::panic::set_hook(Box::new(|info| {
            let msg = info.payload().downcast_ref::<&str>().map(|s| s.to_string()).or_else(|| info.payload().downcast_ref::<String>().cloned()).unwrap_or_default();
            let loc = info.location().map(|l| format!(" at {}:{}", l.file(), l.line())).unwrap_or_default();
            *PANIC.lock().unwrap() = format!("{msg}{loc}");
        }));
        Harness { ctx, quick, only, suite: String::new(), records: Vec::new(), skipped: Vec::new(), json, header_printed: false }
    }

    fn full(&self, case: &str) -> String {
        format!("{}/{}", self.suite, case)
    }

    /// Whether `case` (of the current suite) is selected by `--only`.
    pub fn want(&self, case: &str) -> bool {
        let full = self.full(case);
        self.only.as_deref().is_none_or(|f| full.contains(f))
    }

    /// Runs `f` (which verifies and times one or more cases) if `group` is selected; a panic is
    /// reported as a `SKIPPED` line and the run goes on. Case names inside a group should start
    /// with the group name (`"attention/forward"` in group `"attention"`), so that `--only`
    /// picks a group by `suite/group` or one case by its full name.
    pub fn guarded(&mut self, group: &str, f: impl FnOnce(&mut Harness)) {
        if let Some(only) = self.only.as_deref() {
            let full = self.full(group);
            if !full.contains(only) && !only.starts_with(&full) {
                return;
            }
        }
        let result = catch_unwind(AssertUnwindSafe(|| f(self)));
        if result.is_err() {
            let msg = PANIC.lock().unwrap().clone().replace('\n', " | ");
            let name = self.full(group);
            println!("SKIPPED: {name}: {msg}");
            self.skipped.push((name, msg));
            let _ = self.ctx.finish_profiling();
            let _ = self.ctx.sync();
        }
    }

    /// Records that `case` was not run, and why.
    pub fn skip(&mut self, case: &str, why: &str) {
        if self.want(case) {
            let name = self.full(case);
            println!("SKIPPED: {name}: {why}");
            self.skipped.push((name, why.to_string()));
        }
    }

    fn header(&mut self) {
        if !self.header_printed {
            println!("{:<40} {:>14} {:>10} {:>10} {:>10} {:>11} {:<10} {:>6} {:>9} {:>8}", "case", "size", "first ms", "median ms", "min ms", "rate", "unit", "launch", "gpu ms", "mem MiB");
            self.header_printed = true;
        }
    }

    /// Times `f` as case `case` of size `size` if selected. `f` must leave its results on the
    /// device or read them back itself; the harness syncs after every call.
    pub fn case(&mut self, case: &str, size: &str, work: Work, f: &mut dyn FnMut()) {
        if !self.want(case) {
            return;
        }
        self.header();
        let ctx = self.ctx.clone();
        let run = |f: &mut dyn FnMut()| {
            let t = Instant::now();
            f();
            ctx.sync().expect("sync");
            t.elapsed().as_secs_f64() * 1e3
        };
        let first = run(f);
        let warm = run(f).max(run(f));
        // A GPU that idled between cases runs at low clocks (NVIDIA drops to 210 MHz within
        // ~50 ms, which made a 12 us kernel take 150 us in a case); keep calling the case for
        // a while so the clocks are up when the timed calls start.
        let started = Instant::now();
        while warm < 5.0 && started.elapsed().as_millis() < 150 {
            run(f);
        }
        let reps = if let Some(r) = std::env::var("LUMA_BENCH_REPS").ok().and_then(|v| v.parse().ok()) {
            r
        } else if warm > 1000.0 {
            3
        } else if self.quick {
            5
        } else if warm < 10.0 {
            21
        } else {
            9
        };
        let mut times: Vec<f64> = (0..reps).map(|_| run(f)).collect();
        times.sort_by(f64::total_cmp);
        let (median, min) = (times[times.len() / 2], times[0]);
        let (rate, unit) = match work {
            Work::Bytes(b) => (Some(b / (median * 1e-3) / 1e9), "GB/s"),
            Work::Flops(n) => (Some(n / (median * 1e-3) / 1e9), "GFLOP/s"),
            Work::Cells(n) => (Some(n / (median * 1e-3) / 1e6), "Mcell-st/s"),
            Work::Items(n, u) => (Some(n / (median * 1e-3) / 1e6), u),
            Work::None => (None, ""),
        };
        let kernels = std::env::var_os("LUMA_BENCH_KERNELS").is_some();
        let (launches, gpu_ms) = match self.ctx.profile(&mut *f) {
            Ok(((), p)) => {
                if kernels {
                    println!("  kernels of {} {size}:", self.full(case));
                    for k in &p.kernels {
                        println!("    {:>9.1} us x{:<3} {}", k.total.as_secs_f64() * 1e6, k.launches, k.label.chars().take(110).collect::<String>());
                    }
                }
                (Some(p.launches()), Some(p.total().as_secs_f64() * 1e3))
            }
            Err(_) => (None, None),
        };
        let _ = self.ctx.sync();
        let stats = self.ctx.memory_stats();
        let mem_mib = (stats.resident_bytes + stats.pooled_bytes) as f64 / (1 << 20) as f64;
        let rec = Record {
            suite: self.suite.clone(),
            case: case.to_string(),
            size: size.to_string(),
            first_ms: first,
            median_ms: median,
            min_ms: min,
            reps,
            rate,
            unit,
            gpu_ms,
            launches,
            mem_mib,
        };
        let fmt = |v: Option<f64>| v.map_or("-".to_string(), |v| if v < 10.0 { format!("{v:.3}") } else { format!("{v:.1}") });
        println!(
            "{:<40} {:>14} {:>10.3} {:>10.3} {:>10.3} {:>11} {:<10} {:>6} {:>9} {:>8.1}",
            self.full(case),
            size,
            first,
            median,
            min,
            fmt(rate),
            unit,
            launches.map_or("-".to_string(), |l| l.to_string()),
            fmt(gpu_ms),
            mem_mib
        );
        if let Some(json) = self.json.as_mut() {
            let _ = writeln!(json, "{}", rec.to_json(&self.ctx));
            let _ = json.flush();
        }
        self.records.push(rec);
    }
}

fn esc(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

impl Record {
    pub fn to_json(&self, ctx: &Context) -> String {
        let mut s = String::from("{");
        let num = |v: Option<f64>| v.map_or("null".to_string(), |v| format!("{v:.6}"));
        let _ = write!(
            s,
            "\"suite\":\"{}\",\"case\":\"{}\",\"size\":\"{}\",\"backend\":\"{:?}\",\"device\":\"{}\",\"first_ms\":{:.6},\"median_ms\":{:.6},\"min_ms\":{:.6},\"reps\":{},\"rate\":{},\"unit\":\"{}\",\"gpu_ms\":{},\"launches\":{},\"mem_mib\":{:.3},\"ref\":\"{}\"",
            esc(&self.suite),
            esc(&self.case),
            esc(&self.size),
            ctx.backend(),
            esc(ctx.device_name()),
            self.first_ms,
            self.median_ms,
            self.min_ms,
            self.reps,
            num(self.rate),
            self.unit,
            num(self.gpu_ms),
            self.launches.map_or("null".to_string(), |l| l.to_string()),
            self.mem_mib,
            esc(&std::env::var("LUMA_BENCH_REF").unwrap_or_default())
        );
        s.push('}');
        s
    }
}

// -------------------------------------------------------------------------------------------------
// Data and verification helpers shared by the suites
// -------------------------------------------------------------------------------------------------

/// Deterministic values in `lo..hi`.
pub fn values(n: usize, seed: u64, lo: f32, hi: f32) -> Vec<f32> {
    let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let unit = ((state >> 40) as f32) / (1u64 << 24) as f32;
            lo + unit * (hi - lo)
        })
        .collect()
}

/// Deterministic integers in `0..m`.
pub fn ints(n: usize, seed: u64, m: u32) -> Vec<u32> {
    let mut s = seed.wrapping_mul(0x9E3779B97F4A7C15) | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s % u64::from(m.max(1))) as u32
        })
        .collect()
}

/// Panics unless every `got[i]` is within `max(abs, rel * |want[i]|)` of `want[i]`.
pub fn check(name: &str, got: &[f32], want: &[f32], rel: f32, abs: f32) {
    assert_eq!(got.len(), want.len(), "{name}: length");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let tol = abs.max(rel * w.abs());
        assert!((g - w).abs() <= tol || (g.is_nan() && w.is_nan()), "{name}: index {i}: got {g}, want {w} (tol {tol})");
    }
}

/// Panics unless `got` is within `rel` (relative) of `want`.
pub fn check_scalar(name: &str, got: f64, want: f64, rel: f64) {
    assert!((got - want).abs() <= rel * want.abs().max(1e-30), "{name}: got {got}, want {want} (rel {rel})");
}
