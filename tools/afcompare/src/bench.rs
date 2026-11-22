//! The interleaved measuring harness and the report.
//!
//! One steady-state measurement of a case: the two sides were verified against each other
//! (by the case, before it calls [`Bench::run`]); each side is then warmed up for at least
//! 250 ms (as `luma-bench` does, so that the GPU clocks are up), and the timed repetitions
//! alternate in ABBA / BAAB groups (L A A L, A L L A, ...), every repetition ending in a full
//! device synchronisation inside the closure. The reported time is the median.

use std::fmt::Write as _;
use std::io::Write as _;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::Mutex;
use std::time::Instant;

use crate::util::Verdict;

pub static PANIC: Mutex<String> = Mutex::new(String::new());

#[derive(Clone, Debug)]
pub struct Row {
    pub case: String,
    pub size: String,
    pub luma_ms: Option<f64>,
    pub af_ms: Option<f64>,
    pub luma_min: Option<f64>,
    pub af_min: Option<f64>,
    pub reps: usize,
    pub agree: String,
    pub ok: bool,
    pub note: String,
}

impl Row {
    pub fn ratio(&self) -> Option<f64> {
        Some(self.af_ms? / self.luma_ms?)
    }
}

pub struct Bench {
    pub ctx: luma::Context,
    pub filter: Option<String>,
    pub quick: bool,
    pub rows: Vec<Row>,
    pub failures: Vec<(String, String)>,
    json: Option<std::io::BufWriter<std::fs::File>>,
}

fn time(f: &mut dyn FnMut()) -> f64 {
    let t = Instant::now();
    f();
    t.elapsed().as_secs_f64() * 1e3
}

/// Calls `f` for at least 250 ms (and twice); returns the mean call time in ms.
fn warm(f: &mut dyn FnMut()) -> f64 {
    let start = Instant::now();
    let mut n = 0u32;
    while n < 2 || start.elapsed().as_millis() < 250 {
        f();
        n += 1;
    }
    start.elapsed().as_secs_f64() * 1e3 / f64::from(n)
}

/// A kernel label without the debug dump of its view specs: every `Name { .. }` block is cut to `Name`.
fn compact(label: &str) -> String {
    let mut out = String::new();
    let mut depth = 0usize;
    let mut chars = label.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '{' => depth += 1,
            '}' => depth = depth.saturating_sub(1),
            _ if depth == 0 => out.push(c),
            _ => {}
        }
    }
    out.chars().take(200).collect()
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

fn reps_per_side(est_ms: f64) -> usize {
    match est_ms {
        e if e < 1.0 => 200,
        e if e < 10.0 => 60,
        e if e < 100.0 => 24,
        e if e < 1000.0 => 10,
        _ => 6,
    }
}

pub fn fmt_ms(v: f64) -> String {
    if v >= 100.0 {
        format!("{v:.1}")
    } else if v >= 10.0 {
        format!("{v:.2}")
    } else if v >= 1.0 {
        format!("{v:.3}")
    } else {
        format!("{v:.4}")
    }
}

impl Bench {
    pub fn new(ctx: luma::Context, filter: Option<String>, quick: bool, json: Option<&str>) -> Bench {
        std::panic::set_hook(Box::new(|info| {
            let msg = info.payload().downcast_ref::<&str>().map(|s| s.to_string()).or_else(|| info.payload().downcast_ref::<String>().cloned()).unwrap_or_default();
            let loc = info.location().map(|l| format!(" at {}:{}", l.file(), l.line())).unwrap_or_default();
            *PANIC.lock().unwrap() = format!("{msg}{loc}");
        }));
        let json = json.map(|p| std::io::BufWriter::new(std::fs::File::create(p).expect("cannot create the JSON file")));
        Bench { ctx, filter, quick, rows: Vec::new(), failures: Vec::new(), json }
    }

    /// Whether a case group named `group` is selected by `--only` (substring of the name).
    pub fn want(&self, group: &str) -> bool {
        self.filter.as_deref().is_none_or(|f| group.contains(f) || f.contains(group))
    }

    /// Runs `f` if `group` is selected; a panic (an ArrayFire or luma error) is reported and the run goes on.
    pub fn guarded(&mut self, group: &str, f: impl FnOnce(&mut Bench)) {
        if !self.want(group) {
            return;
        }
        println!("== {group}");
        let result = catch_unwind(AssertUnwindSafe(|| f(self)));
        if result.is_err() {
            let msg = PANIC.lock().unwrap().clone().replace('\n', " | ");
            println!("!!!!!! FAILED: {group}: {msg}");
            self.failures.push((group.to_string(), msg));
            let _ = self.ctx.sync();
            let _ = catch_unwind(|| arrayfire::sync(0));
        }
    }

    /// Times both sides interleaved. `luma` and `af` do one call of the case including the final
    /// synchronisation of their library.
    pub fn run(&mut self, case: &str, size: &str, verdict: Verdict, note: &str, luma: &mut dyn FnMut(), af: &mut dyn FnMut()) {
        if !verdict.ok {
            println!("!!!!!! MISMATCH in {case} [{size}]: {}", verdict.text);
            eprintln!("!!!!!! MISMATCH in {case} [{size}]: {}", verdict.text);
        }
        let est_l = warm(luma);
        let est_a = warm(af);
        let n = reps_per_side(est_l.max(est_a));
        let (mut tl, mut ta) = (Vec::with_capacity(n), Vec::with_capacity(n));
        for g in 0..n / 2 {
            if g % 2 == 0 {
                tl.push(time(luma));
                ta.push(time(af));
                ta.push(time(af));
                tl.push(time(luma));
            } else {
                ta.push(time(af));
                tl.push(time(luma));
                tl.push(time(luma));
                ta.push(time(af));
            }
        }
        let (lmin, amin) = (tl.iter().copied().fold(f64::INFINITY, f64::min), ta.iter().copied().fold(f64::INFINITY, f64::min));
        // One profiled call of luma (GPU time by kernel timestamps, number of launches): a sanity check
        // that the timed closure does the work (luma does not compute arrays dropped before the GPU got to them).
        let kernels = std::env::var_os("AFCOMPARE_KERNELS").is_some();
        let prof = self.ctx.profile(&mut *luma).ok().map(|(_, p)| {
            if kernels {
                println!("  luma kernels of {case} [{size}]:");
                for k in &p.kernels {
                    println!("    {:>9.1} us x{:<3} {}", k.total.as_secs_f64() * 1e6, k.launches, compact(&k.label));
                }
            }
            format!("luma: {} launches, {:.3} ms GPU", p.launches(), p.total().as_secs_f64() * 1e3)
        });
        let _ = self.ctx.sync();
        let note = match prof {
            Some(p) if note.is_empty() => p,
            Some(p) => format!("{note}; {p}"),
            None => note.to_string(),
        };
        let row = Row {
            case: case.to_string(),
            size: size.to_string(),
            luma_ms: Some(median(&mut tl)),
            af_ms: Some(median(&mut ta)),
            luma_min: Some(lmin),
            af_min: Some(amin),
            reps: n,
            agree: verdict.text,
            ok: verdict.ok,
            note,
        };
        self.push(row);
    }

    /// A row with only one side measured (the other has no equivalent): `luma` or `af` is `None`.
    pub fn run_one(&mut self, case: &str, size: &str, luma: Option<&mut dyn FnMut()>, af: Option<&mut dyn FnMut()>, note: &str) {
        let measure = |f: &mut dyn FnMut()| {
            let est = warm(f);
            let n = reps_per_side(est);
            let mut t: Vec<f64> = (0..n).map(|_| time(f)).collect();
            let min = t.iter().copied().fold(f64::INFINITY, f64::min);
            (median(&mut t), min, n)
        };
        let l = luma.map(|f| measure(f));
        let a = af.map(|f| measure(f));
        let row = Row {
            case: case.to_string(),
            size: size.to_string(),
            luma_ms: l.map(|x| x.0),
            af_ms: a.map(|x| x.0),
            luma_min: l.map(|x| x.1),
            af_min: a.map(|x| x.1),
            reps: l.or(a).map_or(0, |x| x.2),
            agree: "n/a (one side only)".into(),
            ok: true,
            note: note.to_string(),
        };
        self.push(row);
    }

    fn push(&mut self, row: Row) {
        println!(
            "{:<34} {:>22}  luma {:>9} ms  AF {:>9} ms  AF/luma {:>6}  {}",
            row.case,
            row.size,
            row.luma_ms.map_or("-".into(), fmt_ms),
            row.af_ms.map_or("-".into(), fmt_ms),
            row.ratio().map_or("-".into(), |r| format!("{r:.2}")),
            row.agree
        );
        if let Some(j) = self.json.as_mut() {
            let esc = |s: &str| s.replace('\\', "\\\\").replace('"', "\\\"");
            let num = |v: Option<f64>| v.map_or("null".to_string(), |x| format!("{x}"));
            let _ = writeln!(
                j,
                "{{\"case\":\"{}\",\"size\":\"{}\",\"luma_ms\":{},\"af_ms\":{},\"luma_min_ms\":{},\"af_min_ms\":{},\"ratio\":{},\"reps_per_side\":{},\"agree\":\"{}\",\"ok\":{},\"note\":\"{}\"}}",
                esc(&row.case),
                esc(&row.size),
                num(row.luma_ms),
                num(row.af_ms),
                num(row.luma_min),
                num(row.af_min),
                num(row.ratio()),
                row.reps,
                esc(&row.agree),
                row.ok,
                esc(&row.note)
            );
            let _ = j.flush();
        }
        self.rows.push(row);
    }

    /// The markdown table of every row so far.
    pub fn markdown(&self, title: &str) -> String {
        let mut s = String::new();
        let _ = writeln!(s, "### {title}\n");
        let _ = writeln!(s, "Steady-state per-call time, median of the timed repetitions of an interleaved (ABBA) run; ratio = AF / luma (> 1: luma faster).\n");
        let _ = writeln!(s, "| case | size | luma ms | ArrayFire ms | ratio (AF/luma) | results agree? | note |");
        let _ = writeln!(s, "|---|---|---:|---:|---:|---|---|");
        for r in &self.rows {
            let agree = if r.ok { r.agree.clone() } else { format!("**MISMATCH** {}", r.agree) };
            let _ = writeln!(
                s,
                "| {} | {} | {} | {} | {} | {} | {} |",
                r.case,
                r.size,
                r.luma_ms.map_or("n/a".into(), fmt_ms),
                r.af_ms.map_or("n/a".into(), fmt_ms),
                r.ratio().map_or("n/a".into(), |x| format!("{x:.2}")),
                agree,
                r.note
            );
        }
        if !self.failures.is_empty() {
            let _ = writeln!(s, "\nCases that failed with an error (no row):\n");
            for (g, m) in &self.failures {
                let _ = writeln!(s, "- `{g}`: {m}");
            }
        }
        s
    }
}
