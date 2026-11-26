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
    /// The wall time of the call before the harness syncs (the host time of recording and submitting it), over its launches.
    pub host_ms_per_launch: Option<f64>,
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
    /// `--iters N`: every case runs exactly N timed calls, no calibration or profile run.
    pub iters: Option<usize>,
    /// `--group suite/group,...`: run only these units (see [`Harness::unit_of`]); `tools/benchab.sh` runs one unit per process.
    pub groups: Option<Vec<String>>,
    /// `--list-groups`: run nothing, collect the units of the selected suites in [`Harness::listed`].
    pub list: bool,
    /// The units seen in list mode, in order, without repeats.
    pub listed: std::cell::RefCell<Vec<String>>,
    /// Cases timed (or skipped) so far: a run whose `--only` / `--group` selected none is an error (see `main`).
    pub selected: usize,
    /// Inside a [`Harness::guarded`] group that was selected.
    in_group: bool,
    warmed: bool,
    json: Option<std::io::BufWriter<std::fs::File>>,
    /// `LUMA_BENCH_DECISIONS`: the decision log (see [`Harness::log_decisions`]).
    decisions: Option<std::io::BufWriter<std::fs::File>>,
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
        Harness { ctx, quick, only, suite: String::new(), records: Vec::new(), skipped: Vec::new(), iters: None, groups: None, list: false, listed: Default::default(), selected: 0, in_group: false, warmed: false, json, decisions: None, header_printed: false }
    }

    /// Writes the decision log of every case that runs from now on to `path` (see
    /// [`Harness::record_decisions`]).
    pub fn log_decisions(&mut self, path: &str) {
        self.decisions = Some(std::io::BufWriter::new(std::fs::File::create(path).expect("cannot create the decision log")));
    }

    /// Records the decisions of `f` on the harness context, after the timed runs: with the plan cache
    /// emptied (`set_plan_cache(true)` drops every plan) one call is the **cold** pass, every planning
    /// decision made and logged; the next call is the **warm** pass, which replays the plan (`PlanReplay`).
    /// `kernels` is the harness's profile run of the case (the launches per kernel label), `None` where
    /// profiling is unsupported. One JSONL line per decision and one `kernels` line per case.
    fn record_decisions(&mut self, case: &str, f: &mut dyn FnMut(), kernels: Option<&Profile>) {
        if self.decisions.is_none() {
            return;
        }
        let name = self.full(case);
        let ctx = self.ctx.clone();
        ctx.set_plan_cache(true);
        ctx.record_decisions(true);
        let mut lines = String::new();
        for pass in ["cold", "warm"] {
            f();
            ctx.sync().expect("sync");
            for d in ctx.take_decisions() {
                let _ = writeln!(
                    lines,
                    "{{\"case\":\"{}\",\"pass\":\"{pass}\",\"site\":\"{}\",\"op\":\"{}\",\"location\":\"{}:{}:{}\",\"chosen\":\"{}\"}}",
                    esc(&name),
                    esc(&format!("{:?}", d.site)),
                    esc(d.op),
                    esc(d.location.file()),
                    d.location.line(),
                    d.location.column(),
                    esc(&d.chosen)
                );
            }
        }
        ctx.record_decisions(false);
        if let Some(p) = kernels {
            let mut counts: Vec<(&str, u32)> = p.kernels.iter().map(|k| (k.label.as_str(), k.launches)).collect();
            counts.sort_unstable();
            let list: Vec<String> = counts.iter().map(|(label, n)| format!("[\"{}\",{n}]", esc(label))).collect();
            let _ = writeln!(lines, "{{\"case\":\"{}\",\"kernels\":[{}]}}", esc(&name), list.join(","));
        }
        if let Some(out) = self.decisions.as_mut() {
            let _ = out.write_all(lines.as_bytes());
            let _ = out.flush();
        }
    }

    fn full(&self, case: &str) -> String {
        format!("{}/{}", self.suite, case)
    }

    /// The unit `tools/benchab.sh` alternates between the two binaries: the `suite/group` of a [`Harness::guarded`] group, and for a
    /// case outside any group its full name without the last `/` component.
    fn unit_of(full: &str) -> &str {
        full.rsplit_once('/').map_or(full, |(unit, _)| unit)
    }

    fn note_unit(&self, unit: &str) {
        let mut listed = self.listed.borrow_mut();
        if !listed.iter().any(|u| u == unit) {
            listed.push(unit.to_string());
        }
    }

    /// Whether `case` (of the current suite) is selected by `--only` (a substring match, or an exact
    /// match of the full name when the filter ends in `$`, what `tools/hosttime` passes) and `--group`.
    /// In list mode nothing is selected: a case outside a group registers its unit instead.
    pub fn want(&self, case: &str) -> bool {
        let full = self.full(case);
        let by_only = self.only.as_deref().is_none_or(|f| {
            f.split(',').any(|f| match f.strip_suffix('$') {
                Some(exact) => full == exact,
                None => full.contains(f),
            })
        });
        if !by_only {
            return false;
        }
        if self.list {
            self.note_unit(Self::unit_of(&full));
            return false;
        }
        self.in_group || self.groups.as_ref().is_none_or(|g| g.iter().any(|g| g == Self::unit_of(&full)))
    }

    /// Whether `--only` selects the group `group` of the current suite. A filter selects a group by (a substring of) its
    /// `suite/group` name, by a name that starts with it (`compute/matmul/square`, a case of the group), or by the short
    /// `group/case` form (`matmul/square`), since the cases of a group are named `group/...`.
    fn group_wanted(&self, group: &str) -> bool {
        let full = self.full(group);
        self.only.as_deref().is_none_or(|only| {
            only.split(',').any(|f| {
                let f = f.trim_end_matches('$');
                full.contains(f) || f.starts_with(&full) || f.strip_prefix(group).is_some_and(|rest| rest.is_empty() || rest.starts_with('/'))
            })
        })
    }

    /// Runs `f` (which verifies and times one or more cases) if `group` is selected; a panic is
    /// reported as a `SKIPPED` line and the run goes on. Case names inside a group should start
    /// with the group name (`"attention/forward"` in group `"attention"`), so that `--only`
    /// picks a group by `suite/group` or one case by `suite/group/case` or `group/case`; `--only a,b` takes several.
    pub fn guarded(&mut self, group: &str, f: impl FnOnce(&mut Harness)) {
        if !self.group_wanted(group) {
            return;
        }
        let name = self.full(group);
        if self.list {
            self.note_unit(&name);
            return;
        }
        if self.groups.as_ref().is_some_and(|g| !g.contains(&name)) {
            return;
        }
        let outer = std::mem::replace(&mut self.in_group, true);
        let result = catch_unwind(AssertUnwindSafe(|| f(self)));
        self.in_group = outer;
        if result.is_err() {
            let msg = PANIC.lock().unwrap().clone().replace('\n', " | ");
            println!("SKIPPED: {name}: {msg}");
            self.skipped.push((name, msg));
            let _ = self.ctx.finish_profiling();
            let _ = self.ctx.sync();
        }
    }

    /// Keeps the GPU busy for a moment (once per process) so that the first cases do not time its clock ramp-up. The
    /// kernel shares nothing with the cases (so their first-call costs stay theirs).
    pub fn warm_up(&mut self) {
        if self.warmed || self.list || self.ctx.is_cpu() {
            return;
        }
        self.warmed = true;
        let x = Array::<f32>::linspace_on(&self.ctx, 0.0, 1.0, 1 << 24).unwrap();
        let start = Instant::now();
        while start.elapsed() < std::time::Duration::from_millis(400) {
            let mut z = x.clone();
            for _ in 0..6 {
                z = (z * 1.0001 + 0.1).tan().cos();
            }
            z.eval().unwrap();
            self.ctx.sync().unwrap();
        }
    }

    /// Records that `case` was not run, and why.
    pub fn skip(&mut self, case: &str, why: &str) {
        if self.want(case) {
            self.selected += 1;
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
    ///
    /// The profiled call must launch at least one kernel: a case whose work is never evaluated
    /// (an `Array::eval` postpones work, and an array dropped before it is evaluated never runs)
    /// times nothing, so it fails the run. Cases that only build lazy graphs use [`Harness::build_case`].
    pub fn case(&mut self, case: &str, size: &str, work: Work, f: &mut dyn FnMut()) {
        self.run_case(case, size, work, f, false);
    }

    /// Like [`Harness::case`] for a case that only builds a lazy graph and drops it
    /// (`*_build_step`, `converge/points_compile_only`): launching no kernel is expected.
    pub fn build_case(&mut self, case: &str, size: &str, work: Work, f: &mut dyn FnMut()) {
        self.run_case(case, size, work, f, true);
    }

    /// Like [`Harness::case`] for a case that does its work without a kernel launch on the harness
    /// context: a pure transfer (`readback/to_vec_*`, `upload_*`), or work on a context of its own
    /// (`compile/newctx_*`, `threads/private*`), which the profile of the harness context cannot see.
    /// The case still evaluates and flushes inside `f`.
    pub fn transfer_case(&mut self, case: &str, size: &str, work: Work, f: &mut dyn FnMut()) {
        self.run_case(case, size, work, f, true);
    }

    fn run_case(&mut self, case: &str, size: &str, work: Work, f: &mut dyn FnMut(), build_only: bool) {
        if !self.want(case) {
            return;
        }
        self.selected += 1;
        self.header();
        let ctx = self.ctx.clone();
        // (the call and its sync, the call alone: what the host spent recording and submitting)
        let run_both = |f: &mut dyn FnMut()| {
            let t = Instant::now();
            f();
            let host = t.elapsed().as_secs_f64() * 1e3;
            ctx.sync().expect("sync");
            (t.elapsed().as_secs_f64() * 1e3, host)
        };
        let run = |f: &mut dyn FnMut()| run_both(f).0;
        let started = Instant::now();
        let first = run(f);
        if let Some(n) = self.iters {
            // Fixed iteration count for `tools/hosttime` (callgrind): no calibration, no profile run.
            let t = Instant::now();
            for _ in 0..n {
                run(f);
            }
            println!("{:<40} {:>14} {n} iterations, {:.3} ms each", self.full(case), size, t.elapsed().as_secs_f64() * 1e3 / n as f64);
            self.record_decisions(case, f, None);
            return;
        }
        let warm = run(f).max(run(f));
        // A GPU that idled between cases runs at low clocks (NVIDIA drops to 210 MHz within
        // ~50 ms, which made a 12 us kernel take 150 us in a case, and a laptop GPU can start a
        // 1 ms case in P8 and take 12 ms); keep calling the case until it has run for 250 ms so
        // the clocks are up when the timed calls start, whatever the case's length.
        // A decision log needs the same calls before the recording whatever the speed of the run
        // (cases that vary their program per call, adaptive loops), so it skips the timed warm-up.
        while self.decisions.is_none() && started.elapsed().as_millis() < 250 {
            run(f);
        }
        let reps = if self.decisions.is_some() {
            3
        } else if let Some(r) = std::env::var("LUMA_BENCH_REPS").ok().and_then(|v| v.parse().ok()) {
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
        let both: Vec<(f64, f64)> = (0..reps).map(|_| run_both(f)).collect();
        let mut times: Vec<f64> = both.iter().map(|t| t.0).collect();
        let mut hosts: Vec<f64> = both.iter().map(|t| t.1).collect();
        times.sort_by(f64::total_cmp);
        hosts.sort_by(f64::total_cmp);
        let (median, min) = (times[times.len() / 2], times[0]);
        let host_median = hosts[hosts.len() / 2];
        let (rate, unit) = match work {
            Work::Bytes(b) => (Some(b / (median * 1e-3) / 1e9), "GB/s"),
            Work::Flops(n) => (Some(n / (median * 1e-3) / 1e9), "GFLOP/s"),
            Work::Cells(n) => (Some(n / (median * 1e-3) / 1e6), "Mcell-st/s"),
            Work::Items(n, u) => (Some(n / (median * 1e-3) / 1e6), u),
            Work::None => (None, ""),
        };
        let kernels = std::env::var_os("LUMA_BENCH_KERNELS").is_some();
        let mut profile = None;
        // `eval` postpones while the GPU is busy and the arrays the case drops are never computed, so
        // the profiled call would launch nothing whenever the previous call's kernels still ran: treat
        // the GPU as idle for this call only (a decision log has it so already for the whole run).
        let pin = self.decisions.is_none();
        if pin {
            self.ctx.set_assume_idle(true);
        }
        let profiled = self.ctx.profile(&mut *f);
        if pin {
            self.ctx.set_assume_idle(false);
        }
        let (launches, gpu_ms) = match profiled {
            Ok(((), p)) => {
                if kernels {
                    println!("  kernels of {} {size}:", self.full(case));
                    for k in &p.kernels {
                        println!("    {:>9.1} us x{:<3} {}", k.total.as_secs_f64() * 1e6, k.launches, k.label.chars().take(110).collect::<String>());
                    }
                }
                let measured = (Some(p.launches()), Some(p.total().as_secs_f64() * 1e3));
                profile = Some(p);
                measured
            }
            Err(_) => (None, None),
        };
        let _ = self.ctx.sync();
        self.record_decisions(case, f, profile.as_ref());
        if !build_only && launches == Some(0) {
            let why = "the profiled call launched no kernel: the case does not evaluate and flush its work (build_case and transfer_case register cases that launch none by design)";
            println!("FAILED: {}: {why}", self.full(case));
            self.skipped.push((self.full(case), why.to_string()));
        }
        let stats = self.ctx.memory_stats();
        let mem_mib = (stats.resident_bytes + stats.pooled_bytes) as f64 / (1 << 20) as f64;
        let host_ms_per_launch = launches.filter(|&l| l > 0).map(|l| host_median / f64::from(l));
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
            host_ms_per_launch,
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
            "\"suite\":\"{}\",\"case\":\"{}\",\"size\":\"{}\",\"backend\":\"{:?}\",\"device\":\"{}\",\"first_ms\":{:.6},\"median_ms\":{:.6},\"min_ms\":{:.6},\"reps\":{},\"rate\":{},\"unit\":\"{}\",\"gpu_ms\":{},\"launches\":{},\"host_ms_per_launch\":{},\"mem_mib\":{:.3},\"ref\":\"{}\"",
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
            num(self.host_ms_per_launch),
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
