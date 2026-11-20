//! `iterate_until_all`: a loop over a record of arrays whose stop rule is evaluated on the GPU.
//!
//! The step is traced once (its launches recorded) and replayed; between two steps a gate kernel
//! stores the recorded scalars in a history, evaluates the stop conditions on it and, once one
//! holds, zeroes the dispatch arguments of the steps after it. The host waits once per block of
//! steps: the first block is 16 steps and every block that ends without a stop doubles the next
//! one, up to 128 (or every block is [`Until::check_every`] steps); the result never depends on
//! the schedule. On a backend that captures graphs, a block (but its first step, which reads the
//! initial state) is one graph replay: the gate counts the steps and passes the dispatch
//! arguments from step to step on the device, so nothing in a block changes from one run to the
//! next (see `GateLoop`). A step that cannot be replayed (it reads back, spans several buffers,
//! ...) runs on the host instead, with the same stop rule evaluated by `eval_host`; both
//! evaluations agree.
//!
//! With one record that is a reduction of one array expression (a norm, a sum, a maximum), the
//! record of the state a step reads is traced together with the step, so that the step's launch
//! stores the next state and reduces the record in one program, which shares what the two
//! compute (the record of step `j` is then the output of launch `j`, and the loop makes one launch
//! more than it has steps, whose result nobody sees: see `GateLoop`). Where a reduction finishes
//! in its first launch (CUDA and CPU, see `Tuning::single_pass_reduce`) and the step does not read
//! its index, the last workgroup of that launch also runs the gate: a step is one launch. Several
//! records, or records that are not one reduction, keep the records after the step and the gate as
//! a launch of its own; the values are the same.

use std::fmt::Write as _;
use std::marker::PhantomData;
use std::ops::{Add, Div, Index, Mul, Neg, Sub};
use std::panic::Location;
use std::sync::Arc;

use luma_core::{Buffer, BufferUsage, Dispatch, ExecGraph};

use crate::array::{Array, Failure, Storage};
use crate::builtin::{GATE_BINDINGS, GATE_STOPPED, GateId};
use crate::gpu_type::GpuType;
use crate::context::{Context, RawArg, STORAGE_USAGE};
use crate::error::{LumaError, Result};
use crate::fields::Fields;
use crate::gpu::pager::{Accounted, Pin};
use crate::jit::{self, GraphBuilder, GraphCache, Plan, Stepper, Trace};
use crate::kernel::KernelCore;
use crate::ops::{Outcome, launch_1d, workgroups_1d};
use crate::shape::Shape;

type Record<'a, S> = Box<dyn Fn(&S) -> Array<f32> + 'a>;
type Stop<'a> = Box<dyn Fn(&History) -> Cond + 'a>;

/// What a loop run by [`iterate_until_all`] records, and when it stops.
///
/// ```
/// # use luma::*;
/// # fn main() -> Result<()> {
/// let x = Array::from_slice(&[1.0f32; 100], 100)?;
/// let run = iterate_until_all(
///     (x,),
///     Until::new(100)
///         .record("max", |s: &(Array<f32>,)| s.0.max())
///         .stop_when(|h| h["max"].last().le(1e-3)),
///     |s, _| (&s.0 * 0.5,),
/// )?;
/// assert_eq!(run.steps, 10);
/// assert_eq!(run.reason, StopReason::Predicate(0));
/// # Ok(()) }
/// ```
pub struct Until<'a, S> {
    max_steps: usize,
    check_every: Option<usize>,
    records: Vec<(String, Record<'a, S>)>,
    stops: Vec<Stop<'a>>,
}

impl<'a, S: Fields> Until<'a, S> {
    /// A loop of at most `max_steps` steps with no records and no stop rule.
    pub fn new(max_steps: usize) -> Self {
        Until { max_steps, check_every: None, records: Vec::new(), stops: Vec::new() }
    }

    /// A scalar (an array of one element) computed from the state after every step, kept as
    /// history under `name`. Records are `f32` whatever the element type of the state: a
    /// `Df32` state records `s.u.max().to_f32()`.
    pub fn record(mut self, name: &str, f: impl Fn(&S) -> Array<f32> + 'a) -> Self {
        self.records.push((name.to_string(), Box::new(f)));
        self
    }

    /// Stops after the first step for which the condition holds. Conditions are checked in the
    /// order they were added; [`StopReason::Predicate`] holds the index of the first that held.
    pub fn stop_when(mut self, f: impl Fn(&History) -> Cond + 'a) -> Self {
        self.stops.push(Box::new(f));
        self
    }

    /// Records a fixed block of `steps` steps between two waits of the host. By default the first
    /// block is 16 steps and every block that ends without a stop doubles the next one, up to
    /// 128. Steps, reason, histories and state do not depend on this.
    pub fn check_every(mut self, steps: usize) -> Self {
        self.check_every = Some(steps.max(1));
        self
    }
}

/// The recorded scalars of a run, as seen by a stop rule. A stop rule is called once, before the
/// loop starts, and builds a [`Cond`] from series of this history.
#[derive(Debug)]
pub struct History {
    names: Vec<String>,
    series: Vec<Series>,
}

impl History {
    fn new(names: Vec<String>) -> History {
        let series = (0..names.len()).map(|index| Series { index }).collect();
        History { names, series }
    }

    /// The number of steps done so far (1 after the first step).
    pub fn step(&self) -> Stat {
        Stat::new(StatNode::Step)
    }
}

impl Index<&str> for History {
    type Output = Series;

    /// The series recorded under `name`; panics on an unknown name, like a map index.
    fn index(&self, name: &str) -> &Series {
        match self.names.iter().position(|known| known == name) {
            Some(at) => &self.series[at],
            None => panic!("no record named `{name}` (records: {:?})", self.names),
        }
    }
}

/// A recorded scalar over the steps so far.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Series {
    index: usize,
}

impl Series {
    /// Its value after the latest step.
    pub fn last(&self) -> Stat {
        Stat::new(StatNode::Last(self.index))
    }

    /// Its value after the first step.
    pub fn first(&self) -> Stat {
        Stat::new(StatNode::First(self.index))
    }

    /// The largest value over all steps so far.
    pub fn peak(&self) -> Stat {
        Stat::new(StatNode::Peak(self.index))
    }

    /// The smallest value over all steps so far.
    pub fn trough(&self) -> Stat {
        Stat::new(StatNode::Trough(self.index))
    }

    /// The largest value over the last `steps` steps (fewer at the start).
    pub fn window_max(&self, steps: usize) -> Stat {
        Stat::new(StatNode::WindowMax(self.index, steps.max(1)))
    }

    /// The smallest value over the last `steps` steps (fewer at the start).
    pub fn window_min(&self, steps: usize) -> Stat {
        Stat::new(StatNode::WindowMin(self.index, steps.max(1)))
    }

    /// The arithmetic mean over the last `steps` steps (fewer at the start); NaN if any of them is.
    pub fn window_mean(&self, steps: usize) -> Stat {
        Stat::new(StatNode::WindowMean(self.index, steps.max(1)))
    }

    /// Its value `steps_ago` steps before the latest one (the first step's value when the loop
    /// has not run that long); `prev(0)` is [`Series::last`].
    pub fn prev(&self, steps_ago: usize) -> Stat {
        Stat::new(StatNode::Prev(self.index, steps_ago))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Arith {
    Add,
    Sub,
    Mul,
    Div,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Compare {
    Lt,
    Le,
    Gt,
    Ge,
}

#[derive(Debug)]
enum StatNode {
    Const(f32),
    Last(usize),
    First(usize),
    Peak(usize),
    Trough(usize),
    WindowMax(usize, usize),
    WindowMin(usize, usize),
    WindowMean(usize, usize),
    Prev(usize, usize),
    Step,
    Bin(Arith, Stat, Stat),
    Abs(Stat),
}

/// A number computed from the history of a run, evaluated on the GPU after every step. Combine
/// with `+ - * /` (with other stats and with `f32`), [`Stat::abs`], and compare with
/// [`Stat::lt`], [`Stat::le`], [`Stat::gt`] and [`Stat::ge`].
#[derive(Clone, Debug)]
pub struct Stat(Arc<StatNode>);

impl Stat {
    fn new(node: StatNode) -> Stat {
        Stat(Arc::new(node))
    }

    /// The absolute value.
    pub fn abs(&self) -> Stat {
        Stat::new(StatNode::Abs(self.clone()))
    }

    fn compare(&self, op: Compare, rhs: impl Into<Stat>) -> Cond {
        Cond(Arc::new(CondNode::Cmp(op, self.clone(), rhs.into())))
    }

    /// `self < rhs`.
    pub fn lt(&self, rhs: impl Into<Stat>) -> Cond {
        self.compare(Compare::Lt, rhs)
    }

    /// `self <= rhs`.
    pub fn le(&self, rhs: impl Into<Stat>) -> Cond {
        self.compare(Compare::Le, rhs)
    }

    /// `self > rhs`.
    pub fn gt(&self, rhs: impl Into<Stat>) -> Cond {
        self.compare(Compare::Gt, rhs)
    }

    /// `self >= rhs`.
    pub fn ge(&self, rhs: impl Into<Stat>) -> Cond {
        self.compare(Compare::Ge, rhs)
    }
}

impl From<f32> for Stat {
    fn from(value: f32) -> Stat {
        Stat::new(StatNode::Const(value))
    }
}

impl From<&Stat> for Stat {
    fn from(stat: &Stat) -> Stat {
        stat.clone()
    }
}

fn arith(op: Arith, a: &Stat, b: &Stat) -> Stat {
    Stat::new(StatNode::Bin(op, a.clone(), b.clone()))
}

macro_rules! stat_operator {
    ($trait:ident, $method:ident, $op:expr) => {
        impl $trait<Stat> for Stat {
            type Output = Stat;
            fn $method(self, rhs: Stat) -> Stat {
                arith($op, &self, &rhs)
            }
        }
        impl $trait<&Stat> for Stat {
            type Output = Stat;
            fn $method(self, rhs: &Stat) -> Stat {
                arith($op, &self, rhs)
            }
        }
        impl $trait<Stat> for &Stat {
            type Output = Stat;
            fn $method(self, rhs: Stat) -> Stat {
                arith($op, self, &rhs)
            }
        }
        impl $trait<&Stat> for &Stat {
            type Output = Stat;
            fn $method(self, rhs: &Stat) -> Stat {
                arith($op, self, rhs)
            }
        }
        impl $trait<f32> for Stat {
            type Output = Stat;
            fn $method(self, rhs: f32) -> Stat {
                arith($op, &self, &Stat::from(rhs))
            }
        }
        impl $trait<f32> for &Stat {
            type Output = Stat;
            fn $method(self, rhs: f32) -> Stat {
                arith($op, self, &Stat::from(rhs))
            }
        }
        impl $trait<Stat> for f32 {
            type Output = Stat;
            fn $method(self, rhs: Stat) -> Stat {
                arith($op, &Stat::from(self), &rhs)
            }
        }
        impl $trait<&Stat> for f32 {
            type Output = Stat;
            fn $method(self, rhs: &Stat) -> Stat {
                arith($op, &Stat::from(self), rhs)
            }
        }
    };
}

stat_operator!(Add, add, Arith::Add);
stat_operator!(Sub, sub, Arith::Sub);
stat_operator!(Mul, mul, Arith::Mul);
stat_operator!(Div, div, Arith::Div);

impl Neg for Stat {
    type Output = Stat;
    fn neg(self) -> Stat {
        arith(Arith::Sub, &Stat::from(0.0), &self)
    }
}

impl Neg for &Stat {
    type Output = Stat;
    fn neg(self) -> Stat {
        arith(Arith::Sub, &Stat::from(0.0), self)
    }
}

#[derive(Debug)]
enum CondNode {
    Cmp(Compare, Stat, Stat),
    And(Cond, Cond),
    Or(Cond, Cond),
    Not(Cond),
}

/// A condition on the history of a run, see [`Until::stop_when`].
#[derive(Clone, Debug)]
pub struct Cond(Arc<CondNode>);

impl Cond {
    /// Both conditions hold.
    pub fn and(self, other: Cond) -> Cond {
        Cond(Arc::new(CondNode::And(self, other)))
    }

    /// At least one of the conditions holds.
    pub fn or(self, other: Cond) -> Cond {
        Cond(Arc::new(CondNode::Or(self, other)))
    }

    /// The condition does not hold.
    #[allow(clippy::should_implement_trait)]
    pub fn not(self) -> Cond {
        Cond(Arc::new(CondNode::Not(self)))
    }
}

/// Why a loop run by [`iterate_until_all`] ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StopReason {
    /// `max_steps` steps ran without any condition holding.
    MaxSteps,
    /// The condition added by the `stop_when` call with this index held after the last step.
    Predicate(usize),
}

/// The result of [`iterate_until_all`].
#[derive(Debug)]
pub struct Run<S> {
    /// The state after the last step.
    pub state: S,
    /// Steps run.
    pub steps: usize,
    /// Why the loop ended.
    pub reason: StopReason,
    history: Vec<(String, Vec<f32>)>,
}

impl<S> Run<S> {
    /// The values recorded under `name`, one per step run.
    pub fn history(&self, name: &str) -> Option<&[f32]> {
        self.history.iter().find(|(known, _)| known == name).map(|(_, values)| values.as_slice())
    }
}

// -- The two evaluations of a stop rule. --

/// `f32::max` that propagates NaN, like the gate kernel.
fn nan_max(a: f32, b: f32) -> f32 {
    if a.is_nan() || b.is_nan() { f32::NAN } else { a.max(b) }
}

/// `f32::min` that propagates NaN, like the gate kernel.
fn nan_min(a: f32, b: f32) -> f32 {
    if a.is_nan() || b.is_nan() { f32::NAN } else { a.min(b) }
}

fn eval_stat(stat: &Stat, histories: &[Vec<f32>], done: usize) -> f32 {
    let window = |r: usize, k: usize| &histories[r][done - k.min(done)..done];
    match &*stat.0 {
        StatNode::Const(c) => *c,
        StatNode::Last(r) => histories[*r][done - 1],
        StatNode::First(r) => histories[*r][0],
        StatNode::Peak(r) => histories[*r][..done].iter().copied().fold(f32::NEG_INFINITY, nan_max),
        StatNode::Trough(r) => histories[*r][..done].iter().copied().fold(f32::INFINITY, nan_min),
        StatNode::WindowMax(r, k) => window(*r, *k).iter().copied().fold(f32::NEG_INFINITY, nan_max),
        StatNode::WindowMin(r, k) => window(*r, *k).iter().copied().fold(f32::INFINITY, nan_min),
        StatNode::WindowMean(r, k) => {
            let w = window(*r, *k);
            if w.iter().any(|x| x.is_nan()) {
                f32::NAN
            } else {
                w.iter().sum::<f32>() / w.len() as f32
            }
        }
        StatNode::Prev(r, back) => histories[*r][(done - 1).saturating_sub(*back)],
        StatNode::Step => done as f32,
        StatNode::Bin(op, a, b) => {
            let (a, b) = (eval_stat(a, histories, done), eval_stat(b, histories, done));
            match op {
                Arith::Add => a + b,
                Arith::Sub => a - b,
                Arith::Mul => a * b,
                Arith::Div => a / b,
            }
        }
        StatNode::Abs(a) => eval_stat(a, histories, done).abs(),
    }
}

/// Whether `cond` holds after `steps_done` steps, `histories[r]` holding at least that many
/// values of record `r`.
fn eval_host(cond: &Cond, histories: &[Vec<f32>], steps_done: usize) -> bool {
    match &*cond.0 {
        CondNode::Cmp(op, a, b) => {
            let (a, b) = (eval_stat(a, histories, steps_done), eval_stat(b, histories, steps_done));
            match op {
                Compare::Lt => a < b,
                Compare::Le => a <= b,
                Compare::Gt => a > b,
                Compare::Ge => a >= b,
            }
        }
        CondNode::And(a, b) => eval_host(a, histories, steps_done) && eval_host(b, histories, steps_done),
        CondNode::Or(a, b) => eval_host(a, histories, steps_done) || eval_host(b, histories, steps_done),
        CondNode::Not(a) => !eval_host(a, histories, steps_done),
    }
}

/// The source of `stat` in the gate. Constants are not part of it: the `k`th one is read from the
/// gate's state (`st[GATE_STATE_WORDS + k]`, see `GateLoop::drive`) and pushed to `constants`, so
/// that a threshold that differs from one run to the next does not make another kernel.
fn stat_source(stat: &Stat, constants: &mut Vec<f32>) -> String {
    match &*stat.0 {
        StatNode::Const(c) => {
            constants.push(*c);
            format!("bitcast<f32>(st[{}u])", GATE_STATE_WORDS as usize + constants.len() - 1)
        }
        StatNode::Last(r) => format!("v{r}"),
        StatNode::First(r) => format!("hist[{r}u * st[5u]]"),
        StatNode::Peak(r) => format!("agg[{}u]", 2 * r),
        StatNode::Trough(r) => format!("agg[{}u]", 2 * r + 1),
        StatNode::WindowMax(r, k) => format!("lg_wmax(i, {r}u, {k}u)"),
        StatNode::WindowMin(r, k) => format!("lg_wmin(i, {r}u, {k}u)"),
        StatNode::WindowMean(r, k) => format!("lg_wmean(i, {r}u, {k}u)"),
        StatNode::Prev(r, back) => format!("hist[{r}u * st[5u] + (i - min({back}u, i))]"),
        StatNode::Step => "f32(i + 1u)".to_string(),
        StatNode::Bin(op, a, b) => {
            let symbol = match op {
                Arith::Add => "+",
                Arith::Sub => "-",
                Arith::Mul => "*",
                Arith::Div => "/",
            };
            let (a, b) = (stat_source(a, constants), stat_source(b, constants));
            format!("({a} {symbol} {b})")
        }
        StatNode::Abs(a) => format!("abs({})", stat_source(a, constants)),
    }
}

fn cond_source(cond: &Cond, constants: &mut Vec<f32>) -> String {
    match &*cond.0 {
        CondNode::Cmp(op, a, b) => {
            let symbol = match op {
                Compare::Lt => "<",
                Compare::Le => "<=",
                Compare::Gt => ">",
                Compare::Ge => ">=",
            };
            // A comparison with a NaN never holds, even where fast math ignores NaNs.
            let (a, b) = (stat_source(a, constants), stat_source(b, constants));
            format!("(!lg_isn({a}) && !lg_isn({b}) && ({a} {symbol} {b}))")
        }
        CondNode::And(a, b) => {
            let (a, b) = (cond_source(a, constants), cond_source(b, constants));
            format!("({a} && {b})")
        }
        CondNode::Or(a, b) => {
            let (a, b) = (cond_source(a, constants), cond_source(b, constants));
            format!("({a} || {b})")
        }
        CondNode::Not(a) => format!("!{}", cond_source(a, constants)),
    }
}

/// Words of the gate's state buffer `st` before the constants of the stop rule:
/// `[stopped, steps, reason, words, own, max_steps, 0, 0]` (`words` those of the plan's dispatch
/// arguments, `own` the launch that keeps its workgroups once the loop stopped, `max_steps` the
/// stride of the history).
const GATE_STATE_WORDS: u64 = 8;

/// The WGSL of the gate, over the buffers `st`, `args` (the dispatch arguments of the next step,
/// written), `tmpl` (their template), `hist`, `agg` and `stepv`: `luma_gate(v0, v1, ..)` takes
/// the recorded scalars, stores them, evaluates the conditions and either copies the template to
/// `args` or, once one holds, zeroes it (but for the launch `st[4]`); `luma_stopped()` is what a
/// gate run after the loop stopped does. It counts the steps itself (`st[1]`), so that its launch
/// is the same for every step. `shifted` for the records of the state a step *reads* (the
/// launch `j` of the loop records step `j`, and the launch 0, which reads the initial state,
/// records nothing: see [`GateLoop`]); else the launch `i` records step `i + 1`.
fn gate_logic(records: usize, conds: &[Cond], shifted: bool) -> (String, Vec<f32>) {
    let mut constants = Vec::new();
    let mut s = String::new();
    s.push_str(
        "fn lg_isn(x: f32) -> bool { return (bitcast<u32>(x) & 0x7fffffffu) > 0x7f800000u; }
fn lg_fmax(a: f32, b: f32) -> f32 { return select(select(max(a, b), a, lg_isn(a)), b, lg_isn(b)); }
fn lg_fmin(a: f32, b: f32) -> f32 { return select(select(min(a, b), a, lg_isn(a)), b, lg_isn(b)); }
fn lg_wmax(i: u32, r: u32, k: u32) -> f32 {
    var m = hist[r * st[5u] + i];
    for (var j = i + 1u - min(k, i + 1u); j < i; j++) { m = lg_fmax(m, hist[r * st[5u] + j]); }
    return m;
}
fn lg_wmin(i: u32, r: u32, k: u32) -> f32 {
    var m = hist[r * st[5u] + i];
    for (var j = i + 1u - min(k, i + 1u); j < i; j++) { m = lg_fmin(m, hist[r * st[5u] + j]); }
    return m;
}
fn lg_wmean(i: u32, r: u32, k: u32) -> f32 {
    let n = min(k, i + 1u);
    var s = 0.0;
    var bad = false;
    for (var j = i + 1u - n; j <= i; j++) {
        let v = hist[r * st[5u] + j];
        bad = bad || lg_isn(v);
        s += v;
    }
    return select(s / f32(n), bitcast<f32>(0x7fc00000u), bad);
}
",
    );
    let params: Vec<String> = (0..records).map(|r| format!("v{r}: f32")).collect();
    let _ = writeln!(s, "fn luma_gate({}) {{", params.join(", "));
    if shifted {
        // Launch 0 reads the initial state: nothing to record, only the next launch to set up.
        s.push_str(
            "    let j = st[1u];
    if (j == 0u) {
        st[1u] = 1u;
        for (var w = 0u; w < st[3u]; w++) { args[w] = tmpl[w]; }
        stepv[0u] = 1.0;
        return;
    }
    let i = j - 1u;
",
        );
    } else {
        s.push_str("    let i = st[1u];\n");
    }
    if records == 0 {
        // Every binding must be used for the argument numbering to be reflected.
        s.push_str("    hist[0u] = hist[0u];\n    agg[0u] = agg[0u];\n");
    }
    for r in 0..records {
        let _ = writeln!(s, "    hist[{r}u * st[5u] + i] = v{r};");
        let _ = writeln!(s, "    agg[{}u] = lg_fmax(agg[{}u], v{r});", 2 * r, 2 * r);
        let _ = writeln!(s, "    agg[{}u] = lg_fmin(agg[{}u], v{r});", 2 * r + 1, 2 * r + 1);
    }
    s.push_str("    var reason = 0u;\n");
    for (c, cond) in conds.iter().enumerate() {
        let _ = writeln!(s, "    if (reason == 0u && {}) {{ reason = {}u; }}", cond_source(cond, &mut constants), c + 1);
    }
    // The step count the next gate starts from: one past this step's, unless the loop stops here.
    let (count, next_step) = if shifted { ("select(i + 2u, i + 1u, reason != 0u)", "f32(i + 2u)") } else { ("i + 1u", "f32(i + 1u)") };
    let _ = write!(
        s,
        "    st[1u] = {count};
    if (reason != 0u) {{ st[0u] = 1u; st[2u] = reason; }}
    for (var w = 0u; w < st[3u]; w++) {{ args[w] = select(tmpl[w], 0u, reason != 0u && w / 3u != st[4u]); }}
    stepv[0u] = {next_step};
}}
"
    );
    (s, constants)
}

/// The gate as a kernel of its own, run after the step: the bindings of `gate_logic`, then the
/// recorded scalars `rec0..`, which it reads. Returns the constants of the stop rule too.
fn gate_source(records: usize, conds: &[Cond], shifted: bool) -> (String, Vec<f32>) {
    let mut s = String::new();
    s.push_str("@group(1) @binding(0) var<storage, read_write> st: array<u32>;\n");
    s.push_str("@group(1) @binding(1) var<storage, read_write> args: array<u32>;\n");
    s.push_str("@group(1) @binding(2) var<storage, read> tmpl: array<u32>;\n");
    s.push_str("@group(1) @binding(3) var<storage, read_write> hist: array<f32>;\n");
    s.push_str("@group(1) @binding(4) var<storage, read_write> agg: array<f32>;\n");
    s.push_str("@group(1) @binding(5) var<storage, read_write> stepv: array<f32>;\n");
    for r in 0..records {
        let _ = writeln!(s, "@group(1) @binding({}) var<storage, read> rec{r}: array<f32>;", 6 + r);
    }
    let (logic, constants) = gate_logic(records, conds, shifted);
    s.push_str(&logic);
    s.push_str(GATE_STOPPED);
    let values: Vec<String> = (0..records).map(|r| format!("rec{r}[0u]")).collect();
    let _ = write!(
        s,
        "@compute @workgroup_size(1)
fn main() {{
    if (st[0u] != 0u) {{
        luma_stopped();
        return;
    }}
    luma_gate({});
}}
",
        values.join(", ")
    );
    (s, constants)
}

/// What the words of the gate's state say: the steps that ran and why the loop stopped. The gate
/// wrote them, so a stop at a step outside `1..=max_steps`, or by a condition that does not
/// exist, can only be a device or driver that returned something else than it was given: that is
/// an error, not an index into the host's arrays.
fn gate_outcome(
    done: u32,
    count: u32,
    reason: u32,
    base: usize,
    max_steps: usize,
    conditions: usize,
) -> Outcome<(usize, StopReason)> {
    if done == 0 {
        return Ok((base, StopReason::MaxSteps));
    }
    let steps = count as usize;
    match (reason as usize).checked_sub(1) {
        Some(condition) if condition < conditions && (1..=max_steps).contains(&steps) => {
            Ok((steps, StopReason::Predicate(condition)))
        }
        _ => Err(Failure::New(LumaError::Gpu(format!(
            "iterate_until_all read a state back that the GPU cannot have written: stopped={done} \
             steps={count} condition={reason} (at most {max_steps} steps and {conditions} conditions)"
        )))),
    }
}

// -- The loop. --

fn state_from<S: Fields>(arrays: Vec<Array<S::Elem>>) -> S {
    let mut arrays = arrays.into_iter();
    S::build(&mut || arrays.next().unwrap_or_else(|| unreachable!("a record builds exactly COUNT arrays")))
}

/// Runs `step` on `init` at most `until.max_steps` times, recording scalars of the state after
/// every step and stopping at the first step after which a [`Until::stop_when`] condition
/// holds. `step` gets the state and the 0-based index of the step as an `f32` array of one
/// element (convert it with `to_df32()` / `to_f64()` to use it in a state of those types). The
/// state is any record, `f32`, [`Df32`](crate::Df32) or `f64` (see [`Fields::Elem`]); the
/// recorded values and their history are always `f32`.
///
/// The step is called once to trace it; its recorded kernels are replayed for every step, and
/// the stop rule is evaluated by a kernel on the GPU between steps, so the host waits only once
/// per block of steps (16 steps, doubling after every block without a stop up to 128, or a fixed
/// [`Until::check_every`] steps). A step that reads results back, or states larger than one
/// buffer, run step by step with the stop rule evaluated on the host instead. The step must
/// compute its results from its arguments and captured arrays only. `init` is not modified.
///
/// ```
/// # use luma::*;
/// # fn main() -> Result<()> {
/// let u = Array::from_slice(&[4.0f32, 2.0, 1.0, 0.5], 4)?;
/// let run = iterate_until_all(
///     (u,),
///     Until::new(500)
///         .record("max", |s: &(Array<f32>,)| s.0.max())
///         .stop_when(|h| (h["max"].peak() / h["max"].last()).ge(1e6)),
///     |s, _| (&s.0 * 0.5,),
/// )?;
/// println!("{} steps, {:?}", run.steps, run.reason);
/// # Ok(()) }
/// ```
#[track_caller]
pub fn iterate_until_all<S: Fields>(
    init: S,
    until: Until<'_, S>,
    step: impl Fn(&S, &Array<f32>) -> S,
) -> Result<Run<S>> {
    let location = Location::caller();
    run(init, &until, &step).map_err(|failure| match failure {
        Failure::Inherited(error) => LumaError::clone(&error),
        Failure::New(error) => LumaError::Op { op: "iterate_until_all", location, source: Arc::new(error) },
    })
}

fn run<S: Fields>(init: S, until: &Until<'_, S>, step: &impl Fn(&S, &Array<f32>) -> S) -> Outcome<Run<S>> {
    let arrays = init.arrays();
    let ctx = arrays[0].context().clone();
    if arrays.iter().any(|array| !ctx.same_as(array.context())) {
        return Err(Failure::New(LumaError::ContextMismatch));
    }
    for array in &arrays {
        if let Some(error) = array.error() {
            return Err(Failure::Inherited(Arc::new(error.clone())));
        }
    }
    let shapes: Vec<Shape> = arrays.iter().map(|array| array.shape()).collect();
    let names: Vec<String> = until.records.iter().map(|(name, _)| name.clone()).collect();
    let history = History::new(names.clone());
    let conds: Vec<Cond> = until.stops.iter().map(|stop| stop(&history)).collect();
    if until.max_steps == 0 {
        let history = names.into_iter().map(|name| (name, Vec::new())).collect();
        return Ok(Run { state: init.copy(), steps: 0, reason: StopReason::MaxSteps, history });
    }

    // The step index: a one-element array that the gate advances.
    let mut stepv = Array::<f32>::from_slice_on(&ctx, &[0.0], 1)?;
    let mut single = true;
    for array in &arrays {
        single &= array.storage()?.segments() == 1;
    }

    let apply_records = |next: &S| records_of(until, &shapes, next);

    // Trace one step.
    let mut trace = None;
    let constants = gate_logic(0, &conds, false).1;
    let mut gate_buffers = if single { fold_buffers(&ctx, until, constants.len()) } else { None };
    let (next, records) = if single {
        let (started, probes) = Trace::start(&arrays)?;
        trace = Some(started);
        let probes: S = state_from(probes);
        let next = step(&probes, &stepv);
        let records = trace_records(&ctx, until, &shapes, &probes, &next, &stepv, &conds, gate_buffers.as_ref())?;
        (next, records)
    } else {
        let next = step(&init, &stepv);
        jit::force(&next.arrays())?;
        let recs = apply_records(&next)?;
        (next, Records::After(recs))
    };
    drop(arrays);

    let plan = {
        jit::force(&next.arrays())?;
        records.force()?;
        trace.as_mut().and_then(Trace::finish_plan)
    };

    let rec_storages = records.storages();
    let stepper = match (plan, trace.as_ref(), &rec_storages) {
        (Some(plan), Some(trace), Some(storages)) => gated(&ctx, plan, trace, &next, storages, until.max_steps, records.fold()),
        (plan, _, _) => Err(plan.map(Box::new)),
    };
    let stepper = match stepper {
        Ok(stepper) => Some(stepper),
        Err(plan) => {
            if let Some(plan) = plan {
                plan.run_once()?;
            }
            None
        }
    };

    if let (Some(stepper), Some(rec_storages)) = (stepper, rec_storages) {
        let gate = gate_kernel(&ctx, &conds, rec_storages.len(), &records)?;
        let spec = Spec { conditions: conds.len(), capacity: until.max_steps, fused: records.fused(), fold: records.fold(), constants };
        let mut gate_loop = GateLoop::new(&ctx, stepper, gate, stepv, rec_storages, spec, gate_buffers.take())?;
        return gate_loop.run(&shapes, names, until.max_steps, until.check_every, false);
    }

    // The host loop.
    let mut histories: Vec<Vec<f32>> = vec![Vec::new(); until.records.len()];
    let mut state = init;
    let first_records = match records {
        Records::After(recs) => recs,
        // The step's launch reduced the record of the state it read: the new state's is not there.
        Records::Fused { .. } => apply_records(&next)?,
    };
    let mut first = Some((next, first_records));
    let mut reason = StopReason::MaxSteps;
    let mut steps = 0;
    for t in 0..until.max_steps {
        let (next, recs) = match first.take() {
            Some(pair) => pair,
            None => {
                stepv.write(&[t as f32])?;
                let next = step(&state, &stepv);
                jit::force(&next.arrays())?;
                let recs = apply_records(&next)?;
                (next, recs)
            }
        };
        for (history, rec) in histories.iter_mut().zip(&recs) {
            history.push(rec.item()?);
        }
        state = next;
        steps = t + 1;
        if let Some(c) = conds.iter().position(|cond| eval_host(cond, &histories, steps)) {
            reason = StopReason::Predicate(c);
            break;
        }
    }
    for array in state.arrays() {
        array.check()?;
    }
    Ok(Run { state, steps, reason, history: names.into_iter().zip(histories).collect() })
}

/// Makes a stepper of the traced plan if the gate can drive it; `Err` gives the plan back (if
/// any) to be run once. `fold` is the launch of the plan that runs the gate, if it was folded in.
#[allow(clippy::result_large_err)]
fn gated(
    ctx: &Context,
    plan: Plan,
    trace: &Trace,
    next: &impl Fields,
    recs: &[Arc<Storage>],
    max_steps: usize,
    fold: Option<usize>,
) -> std::result::Result<Stepper, Option<Box<Plan>>> {
    let records = recs.len();
    let fits = recs.iter().all(|rec| rec.segments() == 1)
        && 6 + records <= ctx.jit().max_inputs() + 1
        && (records.max(1) * max_steps * 4) as u64 <= ctx.max_buffer_bytes()
        // The buffers a folded gate names were made for plans up to this many launches, and what
        // it rewrites (the dispatch arguments of the next launch) is read by the launches after it.
        && fold.is_none_or(|own| plan.launches() <= FOLD_LAUNCHES && own + 1 == plan.launches());
    if !fits {
        return Err(Some(Box::new(plan)));
    }
    let outs: std::result::Result<Vec<Arc<Storage>>, _> =
        next.arrays().into_iter().map(|array| array.storage().cloned()).collect();
    match outs {
        Ok(outs) => Stepper::new(ctx, plan, &trace.init, &trace.probe_storages, outs).map_err(Some),
        Err(_) => Err(Some(Box::new(plan))),
    }
}

/// The records of a step's result, each one element and the state the shapes it started with.
fn records_of<S: Fields>(
    until: &Until<'_, S>,
    shapes: &[Shape],
    next: &S,
) -> Outcome<Vec<Array<f32>>> {
    let mut recs = Vec::with_capacity(until.records.len());
    for (_, record) in &until.records {
        let value = record(next);
        if value.len() != 1 {
            return Err(Failure::New(LumaError::InvalidShape(
                "recorded values must have one element".into(),
            )));
        }
        recs.push(value);
    }
    check_shapes(next, shapes)?;
    Ok(recs)
}

/// Whether the arrays of `next` have the shapes the state started with.
fn check_shapes<S: Fields>(next: &S, shapes: &[Shape]) -> Outcome<()> {
    for (a, shape) in next.arrays().into_iter().zip(shapes) {
        if a.shape() != *shape {
            return Err(Failure::New(LumaError::ShapeMismatch {
                lhs: a.shape().dims(),
                rhs: shape.dims(),
            }));
        }
    }
    Ok(())
}

// Tests: the launches of the last traced step (see `crate::jit::SEPARATE_GATE` and `SEPARATE_RECORD` for
// running the gate as a launch of its own, and tracing the record after the step).
#[cfg(test)]
thread_local! {
    /// Launches of the plan of the last loop that was traced, and the launch that runs its gate if folded.
    pub(crate) static LAST_PLAN: std::cell::Cell<Option<(usize, Option<usize>)>> = const { std::cell::Cell::new(None) };
}

/// Launches of a traced step that the buffers of a folded gate are made for (three words of
/// dispatch arguments each).
const FOLD_LAUNCHES: usize = 256;

/// How the records of a traced step are computed.
enum Records {
    /// Reductions of the new state, evaluated after the step (the general case).
    After(Vec<Array<f32>>),
    /// The one record, of the state the step reads, reduced by a launch of the step itself into
    /// `value`, so that the record of step `j` is the output of launch `j` (see [`GateLoop`]).
    /// With `fold` that launch also runs the gate.
    Fused { value: Arc<Storage>, fold: Option<usize> },
}

impl Records {
    fn force(&self) -> Outcome<()> {
        match self {
            Records::After(recs) => jit::force(&recs.iter().collect::<Vec<_>>()).map(drop),
            Records::Fused { .. } => Ok(()),
        }
    }

    /// The buffers the records are in, `None` if some failed.
    fn storages(&self) -> Option<Vec<Arc<Storage>>> {
        match self {
            Records::After(recs) => recs.iter().map(|rec| rec.storage().ok().cloned()).collect(),
            Records::Fused { value, .. } => Some(vec![Arc::clone(value)]),
        }
    }

    fn fused(&self) -> bool {
        matches!(self, Records::Fused { .. })
    }

    fn fold(&self) -> Option<usize> {
        match self {
            Records::Fused { fold, .. } => *fold,
            Records::After(_) => None,
        }
    }
}

/// Evaluates the step `next` (traced from the state `probes`) and its records: the one record
/// of the state the step reads goes into the step's own launch where it can (and the gate with it,
/// given `buffers`), else the records of the new state are evaluated after it. A step that writes
/// an array it captured keeps its records after it: the launch of the step that finds the loop
/// stopped runs in full, so the writes would be applied once more than the steps there are.
#[allow(clippy::too_many_arguments)]
fn trace_records<S: Fields>(
    ctx: &Context,
    until: &Until<'_, S>,
    shapes: &[Shape],
    probes: &S,
    next: &S,
    stepv: &Array<f32>,
    conds: &[Cond],
    buffers: Option<&GateBuffers>,
) -> Outcome<Records> {
    if until.records.len() == 1 && !ctx.recording_writes_outside() {
        check_shapes(next, shapes)?;
        if let Some(rec) = records_of(until, shapes, probes)?.pop() {
            let value = Arc::new(Storage::new(ctx, 1, <f32 as GpuType>::LAYOUT));
            let pin = value.pin(0)?;
            let stepv_storage = stepv.storage()?;
            let step_pin = stepv_storage.pin_quiet(0)?;
            // A launch that is not recorded runs the gate for real, on the state of this run: only
            // a recorded one is replayed (with that state set up, see `GateLoop::drive`).
            let args = buffers.filter(|_| ctx.recording_here()).and_then(|buffers| buffers.fold_args(step_pin.buffer()));
            let gate = args.as_ref().map(|args| jit::FoldGate { id: GateId::register(&gate_logic(1, conds, true).0), args, stepv: stepv_storage.id() });
            if let Some(grouped) = jit::force_with_record(ctx, &next.arrays(), &rec, pin.buffer(), gate) {
                return Ok(Records::Fused { value, fold: grouped?.folded });
            }
        }
    }
    // The records read the new state (a residual of it, through slices): evaluated first, they
    // read its buffers instead of recomputing the step at every index they read.
    jit::force(&next.arrays())?;
    Ok(Records::After(records_of(until, shapes, next)?))
}

/// The buffers of a gate folded into the step's last launch, which the recorded commands name:
/// made before the step is traced, for [`FOLD_LAUNCHES`] launches and the `constants` of the stop
/// rule, with `None` if it cannot be.
fn fold_buffers<S: Fields>(ctx: &Context, until: &Until<'_, S>, constants: usize) -> Option<GateBuffers> {
    #[cfg(test)]
    if crate::jit::SEPARATE_GATE.with(std::cell::Cell::get) {
        return None;
    }
    if until.records.len() != 1 || !ctx.tuning().single_pass_reduce {
        return None;
    }
    GateBuffers::new(ctx, 1, until.max_steps, 3 * FOLD_LAUNCHES, constants, true).ok()
}

/// The gate's kernel, unless the step's last launch runs the gate.
fn gate_kernel(ctx: &Context, conds: &[Cond], records: usize, traced: &Records) -> Outcome<Option<Arc<KernelCore>>> {
    if traced.fold().is_some() {
        return Ok(None);
    }
    Ok(Some(ctx.jit().stencil_kernel(ctx, gate_source(records, conds, traced.fused()).0, "UntilAllGate")?))
}

/// What distinguishes the loops a [`GateLoop`] runs.
struct Spec {
    conditions: usize,
    /// Steps the history has room for: its stride.
    capacity: usize,
    /// Whether the records are those of the states the launches read (see [`Records::Fused`]).
    fused: bool,
    /// The launch of the plan that runs the gate, if it does.
    fold: Option<usize>,
    /// The constants of the stop rule, in the order of its source (see [`stat_source`]).
    constants: Vec<f32>,
}

/// Allocates a buffer from `ctx`'s pool and notes it in `owned`, to be given back, and in `held`,
/// to be counted as resident meanwhile.
fn take(ctx: &Context, owned: &mut Vec<Buffer>, held: &mut Accounted, bytes: u64, usage: BufferUsage) -> Result<Buffer> {
    let buffer = ctx.alloc_buffer(bytes, usage)?;
    held.add(buffer.size());
    owned.push(buffer.clone());
    Ok(buffer)
}

/// The buffers the gate works in, taken from the pool and given back with the loop.
struct GateBuffers {
    ctx: Context,
    /// The template of the dispatch arguments on the device, `words` of them at most.
    tmpl: Buffer,
    /// The gate's state, see [`GATE_STATE_WORDS`], then the constants of the stop rule.
    st: Buffer,
    /// The recorded scalars, `capacity` per record.
    hist: Buffer,
    /// The peak and trough of every record.
    agg: Buffer,
    /// Stands for the dispatch arguments of the next step in the commands of a plan whose last
    /// launch runs the gate (see [`Spec::fold`]).
    next_args: Option<Buffer>,
    /// Words of the template and of every buffer of dispatch arguments.
    words: usize,
    /// Words of `st`.
    state_words: usize,
    records: usize,
    capacity: usize,
    /// Everything taken from the pool, to be given back.
    owned: Vec<Buffer>,
    /// Counts the bytes of `owned` while the loop lives.
    #[allow(dead_code)]
    held: Accounted,
}

impl GateBuffers {
    /// Buffers for a loop of `records` scalars and at most `capacity` steps, whose plan has `words` words of
    /// dispatch arguments and whose stop rule has `constants` constants; with `next_args` a placeholder
    /// for the arguments the folded gate writes.
    fn new(ctx: &Context, records: usize, capacity: usize, words: usize, constants: usize, next_args: bool) -> Result<GateBuffers> {
        let records = records.max(1);
        let state_words = GATE_STATE_WORDS as usize + constants;
        let mut owned = Vec::new();
        let mut held = Accounted::new(ctx, 0);
        let made = (|| -> Result<(Buffer, Buffer, Buffer, Buffer, Option<Buffer>)> {
            let tmpl = take(ctx, &mut owned, &mut held, words as u64 * 4, STORAGE_USAGE)?;
            let st = take(ctx, &mut owned, &mut held, state_words as u64 * 4, STORAGE_USAGE)?;
            let hist = take(ctx, &mut owned, &mut held, (records * capacity) as u64 * 4, STORAGE_USAGE)?;
            let agg = take(ctx, &mut owned, &mut held, 2 * records as u64 * 4, STORAGE_USAGE)?;
            let next = if next_args { Some(take(ctx, &mut owned, &mut held, words as u64 * 4, STORAGE_USAGE | BufferUsage::INDIRECT)?) } else { None };
            Ok((tmpl, st, hist, agg, next))
        })();
        // A gate that runs in a launch that was not recorded finds nothing to do in a state of zeros.
        let made = made.and_then(|(tmpl, st, hist, agg, next)| ctx.upload(&st, bytemuck::cast_slice(&vec![0u32; state_words])).map(|()| (tmpl, st, hist, agg, next)));
        match made {
            Ok((tmpl, st, hist, agg, next_args)) => Ok(GateBuffers { ctx: ctx.clone(), tmpl, st, hist, agg, next_args, words, state_words, records, capacity, owned, held }),
            Err(error) => {
                for buffer in owned {
                    ctx.recycle(buffer);
                }
                Err(error)
            }
        }
    }

    fn args_bytes(&self) -> u64 {
        self.words as u64 * 4
    }

    fn state_bytes(&self) -> u64 {
        self.state_words as u64 * 4
    }

    fn hist_bytes(&self) -> u64 {
        (self.records * self.capacity) as u64 * 4
    }

    fn agg_bytes(&self) -> u64 {
        2 * self.records as u64 * 4
    }

    /// The arguments a gate folded into a launch binds, in the order of its bindings (see
    /// `builtin::reduce_main`): `None` without the placeholder of the next dispatch arguments.
    fn fold_args<'a>(&'a self, stepv: &'a Buffer) -> Option<[RawArg<'a>; GATE_BINDINGS]> {
        let next = self.next_args.as_ref()?;
        Some([
            RawArg::write(&self.st, self.state_bytes()),
            RawArg::write(next, self.args_bytes()),
            RawArg::read(&self.tmpl, self.args_bytes()),
            RawArg::write(&self.hist, self.hist_bytes()),
            RawArg::write(&self.agg, self.agg_bytes()),
            RawArg::write(stepv, 4),
        ])
    }
}

impl Drop for GateBuffers {
    fn drop(&mut self) {
        for buffer in self.owned.drain(..) {
            self.ctx.recycle(buffer);
        }
    }
}

/// The device side of a gated loop: the traced step, the gate kernel and the buffers the gate
/// works in, and the graphs of blocks of steps captured from them.
///
/// Launch `i` (0-based) replays the plan with every command dispatched from `args[k]` (`k` counts
/// from the start of the span of launches being run) at byte `12 * l` for command `l`, then runs
/// the gate. The gate reads the launches it has seen from `st[1]`, records the step's scalars,
/// evaluates the stop conditions and writes the dispatch arguments of the next launch into
/// `args[(k + 1) % len]` (the template, or zeros once the loop stopped), `len` being the
/// launches of the span. Nothing in a launch depends on the host, so a span of launches from the
/// second on (the first reads the run's `init`) is the same work every time it runs: it is
/// captured once, keyed by its length and the parity of its first launch (which of the two state
/// buffer sets it reads), and replayed as one graph. The last gate of a span writes `args[0]`,
/// where the next span starts.
///
/// Launch `i` is step `i + 1`, and the gate after it records step `i + 1`, unless the records
/// are [fused](Records::Fused): then launch `j` reduces the record of the state it reads, which
/// is that of step `j` (the state after `j` steps), so the loop makes one launch more than
/// it has steps and the gate after launch `j` records step `j` and decides whether the loop
/// stops with the state launch `j` read (the buffers the launch wrote hold a step ahead that no
/// one sees); the gate after launch 0, which reads `init`, only sets the next launch up. With a
/// gate [folded](Records::Fused) into the last launch of the plan there is no launch of the gate,
/// and once the loop stopped that launch keeps its workgroups (which leave at once, workgroup 0
/// after it wrote the zeros of the next dispatch arguments) so as to pass the stop on.
///
/// The buffers outlive a run, so a [`CompiledUntil`] replays its graphs run after run; they are
/// forgotten when a buffer they name is replaced.
struct GateLoop {
    ctx: Context,
    stepper: Stepper,
    /// The gate as a launch of its own, unless it is folded into the plan.
    gate: Option<Arc<KernelCore>>,
    conditions: usize,
    /// Steps the history has room for: its stride.
    capacity: usize,
    /// The storages of the recorded scalars, kept alive and resident.
    recs: Vec<Arc<Storage>>,
    rec_pins: Vec<Pin>,
    /// The step index the step reads, one element.
    #[allow(dead_code)]
    stepv: Array<f32>,
    step_pin: Pin,
    /// The workgroup counts of every launch of the plan, three words each.
    template: Vec<u32>,
    /// Whether the records are those of the states the launches read.
    fused: bool,
    /// The launch of the plan that runs the gate, if it does.
    fold: Option<usize>,
    /// The constants of the stop rule as the words of the gate's state.
    constants: Vec<u32>,
    buffers: GateBuffers,
    /// Dispatch arguments, as many as the longest span of launches run so far.
    args: Vec<Buffer>,
    /// Captured spans of launches, by (launches, whether the first is an even launch).
    graphs: GraphCache<(usize, bool)>,
    /// A compiled loop is run again and again: its spans are captured on first use. A one-shot
    /// loop's are captured the second time a span runs (its first spans run once).
    compiled: bool,
}

impl GateLoop {
    /// A loop of `stepper` under `gate` (made for `recs`, none if it is folded into the plan),
    /// in the buffers the folded gate named if it was, else new ones.
    fn new(
        ctx: &Context,
        stepper: Stepper,
        gate: Option<Arc<KernelCore>>,
        stepv: Array<f32>,
        recs: Vec<Arc<Storage>>,
        spec: Spec,
        buffers: Option<GateBuffers>,
    ) -> Outcome<GateLoop> {
        let template = stepper.plan().template();
        u32::try_from(template.len()).map_err(|_| LumaError::Gpu("too many recorded launches".into()))?;
        let rec_pins = recs.iter().map(|rec| rec.pin_quiet(0)).collect::<Result<Vec<_>>>()?;
        let step_pin = stepv.storage()?.pin_quiet(0)?;
        let buffers = match buffers {
            Some(buffers) if buffers.words >= template.len() && buffers.state_words == GATE_STATE_WORDS as usize + spec.constants.len() => buffers,
            // A folded gate's buffers are for plans of up to `FOLD_LAUNCHES` launches (see `gated`).
            _ => GateBuffers::new(ctx, recs.len(), spec.capacity, template.len(), spec.constants.len(), false)?,
        };
        ctx.upload(&buffers.tmpl, bytemuck::cast_slice(&template))?;
        #[cfg(test)]
        LAST_PLAN.with(|last| last.set(Some((stepper.plan().launches(), spec.fold))));
        Ok(GateLoop {
            ctx: ctx.clone(),
            stepper,
            gate,
            conditions: spec.conditions,
            capacity: spec.capacity,
            recs,
            rec_pins,
            stepv,
            step_pin,
            template,
            fused: spec.fused,
            fold: spec.fold,
            constants: spec.constants.iter().map(|c| c.to_bits()).collect(),
            buffers,
            args: Vec::new(),
            graphs: GraphCache::new(ctx),
            compiled: false,
        })
    }

    /// Points the first step at a new initial state, see [`Stepper::rebind`]. The captured
    /// graphs never run the first step, so they stay valid.
    fn rebind(&mut self, init: &[Arc<Storage>]) -> bool {
        self.stepper.rebind(init)
    }

    /// Launches a run makes for `steps` steps.
    fn launches(&self, steps: usize) -> usize {
        steps + usize::from(self.fused)
    }

    /// Makes sure `len` dispatch argument buffers exist.
    fn ensure_args(&mut self, len: usize) -> Result<()> {
        let bytes = self.buffers.args_bytes();
        while self.args.len() < len {
            let buffers = &mut self.buffers;
            let buffer = take(&self.ctx, &mut buffers.owned, &mut buffers.held, bytes, STORAGE_USAGE | BufferUsage::INDIRECT)?;
            self.args.push(buffer);
        }
        Ok(())
    }

    /// The buffers of the recorded scalars as the step swapped by `swaps` leaves them.
    fn resolved(&self, swaps: &[(Buffer, Buffer)]) -> Vec<Buffer> {
        self.rec_pins
            .iter()
            .map(|pin| {
                let buffer = pin.buffer();
                swaps.iter().find(|(from, _)| from == buffer).map_or_else(|| buffer.clone(), |(_, to)| to.clone())
            })
            .collect()
    }

    /// The arguments of the gate launch writing the dispatch arguments of the next step to `next`.
    fn gate_args<'a>(&'a self, next: &'a Buffer, resolved: &'a [Buffer]) -> Vec<RawArg<'a>> {
        let buffers = &self.buffers;
        let mut raw: Vec<RawArg<'_>> = Vec::with_capacity(6 + resolved.len());
        raw.push(RawArg::write(&buffers.st, buffers.state_bytes()));
        raw.push(RawArg::write(next, buffers.args_bytes()));
        raw.push(RawArg::read(&buffers.tmpl, buffers.args_bytes()));
        raw.push(RawArg::write(&buffers.hist, buffers.hist_bytes()));
        raw.push(RawArg::write(&buffers.agg, buffers.agg_bytes()));
        raw.push(RawArg::write(self.step_pin.buffer(), 4));
        raw.extend(resolved.iter().map(|buffer| RawArg::read(buffer, 4)));
        raw
    }

    /// `swaps` with the placeholder of the next dispatch arguments (see [`GateBuffers::next_args`])
    /// standing for `next`, if the gate is folded into the plan.
    fn with_next(&self, swaps: &[(Buffer, Buffer)], next: &Buffer) -> Vec<(Buffer, Buffer)> {
        let mut all = swaps.to_vec();
        if let Some(placeholder) = self.fold.and(self.buffers.next_args.as_ref()) {
            all.push((placeholder.clone(), next.clone()));
        }
        all
    }

    /// Launches `i0..i0 + len` one by one, the dispatch arguments cycling through `args[..len]`.
    fn replay_span(&self, i0: usize, len: usize) -> Outcome<()> {
        let parities = [self.stepper.subst(2), self.stepper.subst(3)];
        let resolved_parities = [self.resolved(&parities[0]), self.resolved(&parities[1])];
        for k in 0..len {
            let t = i0 + k + 1;
            let first = (t == 1).then(|| {
                let swaps = self.stepper.subst(1);
                let resolved = self.resolved(&swaps);
                (swaps, resolved)
            });
            let (swaps, resolved) = match &first {
                Some((s, r)) => (s.as_slice(), r.as_slice()),
                None => (&parities[t & 1][..], &resolved_parities[t & 1][..]),
            };
            let next = &self.args[(k + 1) % len];
            match &self.gate {
                Some(gate) => {
                    self.stepper.plan().replay(swaps, Some(&self.args[k]))?;
                    let raw = self.gate_args(next, resolved);
                    launch_1d(&self.ctx, gate, 1, &raw)?;
                }
                None => self.stepper.plan().replay(&self.with_next(swaps, next), Some(&self.args[k]))?,
            }
        }
        Ok(())
    }

    /// Launches `i0..i0 + len` (`i0 > 0`) as one graph: captured on first use in a compiled loop,
    /// which runs its spans again and again, else the second time such a span runs (a one-shot
    /// loop's first spans run once); `None` to launch them one by one.
    fn graph(&self, i0: usize, len: usize) -> Result<Option<Arc<dyn ExecGraph>>> {
        let grid = match &self.gate {
            Some(gate) => workgroups_1d(&self.ctx, gate, 1)?,
            None => [1, 1, 1],
        };
        let key = (len, (i0 + 1).is_multiple_of(2));
        let replays = if self.compiled { jit::PAYING_REPLAYS } else { 1 };
        Ok(self.graphs.get_or_build(key, replays, || {
            let swaps = [self.stepper.subst(2), self.stepper.subst(3)];
            let resolved = [self.resolved(&swaps[0]), self.resolved(&swaps[1])];
            let steps: Vec<Vec<(Buffer, Buffer)>> = (0..len).map(|k| self.with_next(&swaps[(i0 + k + 1) & 1], &self.args[(k + 1) % len])).collect();
            let mut builder = GraphBuilder::new();
            for k in 0..len {
                let p = (i0 + k + 1) & 1;
                match &self.gate {
                    Some(gate) => {
                        builder.plan(self.stepper.plan(), &swaps[p], Some(&self.args[k]));
                        let raw = self.gate_args(&self.args[(k + 1) % len], &resolved[p]);
                        builder.launch(gate, Dispatch::Direct(grid), [1, 1, 1], raw);
                    }
                    None => builder.plan(self.stepper.plan(), &steps[k], Some(&self.args[k])),
                }
            }
            builder.capture(&self.ctx)
        }))
    }

    /// Runs launches `i0..i0 + len` (the first reads `init` when `i0` is 0, and always runs on its
    /// own), as a graph when the backend has them and nothing needs the launches one by one.
    fn run_span(&mut self, i0: usize, len: usize) -> Outcome<()> {
        self.ensure_args(len)?;
        if i0 > 0
            && !self.ctx.recording_here()
            && !self.ctx.is_profiling()
            && let Some(graph) = self.graph(i0, len)?
        {
            let launches = len * (self.stepper.plan().launches() + usize::from(self.gate.is_some()));
            self.ctx.launch_graph(&*graph, u32::try_from(launches).unwrap_or(u32::MAX))?;
            return Ok(());
        }
        self.replay_span(i0, len)
    }

    /// Runs the launches of a run of `steps` steps, one block of them between two looks at the
    /// gate's state, and returns the steps that ran, why the loop stopped and the history.
    fn drive(&mut self, steps: usize, check_every: Option<usize>) -> Outcome<(usize, StopReason, Vec<Vec<f32>>)> {
        let ctx = self.ctx.clone();
        let records = self.recs.len();
        let total = self.launches(steps);
        // A run starts at step 0 with nothing stopped or aggregated, whatever an earlier run left.
        let own = self.fold.map_or(u32::MAX, |own| own as u32);
        let mut state = vec![0, 0, 0, self.template.len() as u32, own, self.capacity as u32, 0, 0];
        state.extend(&self.constants);
        ctx.upload(self.step_pin.buffer(), bytemuck::cast_slice(&[0.0f32]))?;
        ctx.upload(&self.buffers.st, bytemuck::cast_slice(&state))?;
        let limits: Vec<f32> = (0..records.max(1)).flat_map(|_| [f32::NEG_INFINITY, f32::INFINITY]).collect();
        ctx.upload(&self.buffers.agg, bytemuck::cast_slice(&limits))?;
        self.ensure_args(1)?;
        ctx.upload(&self.args[0], bytemuck::cast_slice(&self.template))?;

        let mut base = 0;
        let mut block = check_every.unwrap_or(crate::jit::FIRST_BLOCK);
        loop {
            let in_block = block.min(total - base);
            let end = base + in_block;
            let mut at = base;
            if at == 0 {
                // The first launch reads `init`, which no graph does: a span of its own, whose gate
                // writes `args[0]` for the rest.
                self.run_span(0, 1)?;
                at = 1;
            }
            if end > at {
                self.run_span(at, end - at)?;
            }
            let [done, count, reason, ..] = ctx.read::<u32>(&self.buffers.st, self.buffers.state_bytes())?[..] else {
                return Err(Failure::New(LumaError::Gpu("iterate_until_all state is missing".into())));
            };
            base = end;
            block = crate::jit::next_block(in_block, check_every);
            if done != 0 || base >= total {
                let (ran, reason) = gate_outcome(done, count, reason, base - usize::from(self.fused), steps, self.conditions)?;
                let histories = if records == 0 {
                    Vec::new()
                } else {
                    // Record `r` has its values at `r * capacity`: read up to the last one wanted.
                    let wanted = ((records - 1) * self.capacity + ran) as u64 * 4;
                    let all = ctx.read::<f32>(&self.buffers.hist, wanted)?;
                    (0..records).map(|r| all[r * self.capacity..r * self.capacity + ran].to_vec()).collect()
                };
                return Ok((ran, reason, histories));
            }
        }
    }

    /// Runs the loop for up to `steps` steps. With `fresh` the returned state is a copy that
    /// shares no buffer with the stepper, so that it may be the next run's `init`.
    fn run<S: Fields>(
        &mut self,
        shapes: &[Shape],
        names: Vec<String>,
        steps: usize,
        check_every: Option<usize>,
        fresh: bool,
    ) -> Outcome<Run<S>> {
        let ctx = self.ctx.clone();
        let (steps, reason, histories) = self.drive(steps, check_every)?;

        // Detaching the even state gives its buffers away, which a captured graph names: the
        // graphs would have to be captured again on the next run, which costs far more (a block is
        // thousands of launches) than the copy the odd case makes anyway.
        let detached = if fresh && !self.graphs.holds_graphs() { self.stepper.detach_even(&ctx, steps) } else { None };
        let arrays: Vec<Array<S::Elem>> = match detached {
            Some(storages) => storages
                .into_iter()
                .zip(shapes)
                .map(|(storage, shape)| Array::from_storage(&ctx, *shape, storage))
                .collect(),
            None => {
                let refs: Vec<Array<S::Elem>> = self
                    .stepper
                    .state_after(steps)
                    .iter()
                    .zip(shapes)
                    .map(|(storage, shape)| Array::from_storage(&ctx, *shape, Arc::clone(storage)))
                    .collect();
                if fresh {
                    let copies: Vec<Array<S::Elem>> = refs.iter().map(Array::copy).collect();
                    jit::force(&copies.iter().collect::<Vec<_>>())?;
                    copies
                } else {
                    refs
                }
            }
        };
        Ok(Run {
            state: state_from(arrays),
            steps,
            reason,
            history: names.into_iter().zip(histories).collect(),
        })
    }
}

impl Drop for GateLoop {
    fn drop(&mut self) {
        // The graphs name the buffers (given back when the fields go): they go first.
        self.graphs.clear();
    }
}

/// A loop traced once by [`compile_until_all`] and run any number of times on new initial states.
///
/// Running a loop in chunks with [`iterate_until_all`] traces the step again for every chunk; a
/// `CompiledUntil` keeps the traced step, the stop rule's kernel, the buffers and, on a backend
/// that captures graphs, the graphs of blocks of steps captured from them, so that a chunk costs
/// only its launches. Every run is a run of the compiled loop: its step index and history
/// start at 0 and its stop rule sees only the steps of that run.
pub struct CompiledUntil<S: Fields> {
    ctx: Context,
    shapes: Vec<Shape>,
    names: Vec<String>,
    max_steps: usize,
    check_every: Option<usize>,
    gate_loop: GateLoop,
    state: PhantomData<fn() -> S>,
}

impl<S: Fields> CompiledUntil<S> {
    /// Runs the compiled loop from `init` for at most the steps it was compiled with. `init` is
    /// not modified; the state returned shares no buffer with the compiled loop and may be passed
    /// straight to the next run.
    #[track_caller]
    pub fn run(&mut self, init: S) -> Result<Run<S>> {
        self.run_steps(init, self.max_steps)
    }

    /// [`CompiledUntil::run`] for at most `max_steps.min(compiled steps)` steps: a shorter last
    /// chunk needs no new compilation.
    #[track_caller]
    pub fn run_steps(&mut self, init: S, max_steps: usize) -> Result<Run<S>> {
        let location = Location::caller();
        self.run_inner(init, max_steps.min(self.max_steps))
            .map_err(|failure| match failure {
                Failure::Inherited(error) => LumaError::clone(&error),
                Failure::New(error) => LumaError::Op {
                    op: "CompiledUntil::run",
                    location,
                    source: Arc::new(error),
                },
            })
    }

    fn run_inner(&mut self, init: S, steps: usize) -> Outcome<Run<S>> {
        let arrays = init.arrays();
        if arrays.len() != self.shapes.len() {
            return Err(Failure::New(LumaError::InvalidShape(
                "the state is not the one compiled".into(),
            )));
        }
        for (array, shape) in arrays.iter().zip(&self.shapes) {
            if !self.ctx.same_as(array.context()) {
                return Err(Failure::New(LumaError::ContextMismatch));
            }
            if let Some(error) = array.error() {
                return Err(Failure::Inherited(Arc::new(error.clone())));
            }
            if array.shape() != *shape {
                return Err(Failure::New(LumaError::ShapeMismatch {
                    lhs: array.shape().dims(),
                    rhs: shape.dims(),
                }));
            }
        }
        if steps == 0 {
            let history = self
                .names
                .iter()
                .map(|name| (name.clone(), Vec::new()))
                .collect();
            return Ok(Run {
                state: init.copy(),
                steps: 0,
                reason: StopReason::MaxSteps,
                history,
            });
        }
        let storages = jit::force(&arrays)?;
        if !self.gate_loop.rebind(&storages) {
            // Buffers the loop writes (or a state spread over several buffers): run on a copy.
            let copies: Vec<Array<S::Elem>> = arrays.iter().map(|array| (*array).copy()).collect();
            let storages = jit::force(&copies.iter().collect::<Vec<_>>())?;
            if !self.gate_loop.rebind(&storages) {
                return Err(Failure::New(LumaError::Unsupported(
                    "CompiledUntil needs states of single buffers, shaped like the compiled one"
                        .into(),
                )));
            }
        }
        self.gate_loop
            .run(&self.shapes, self.names.clone(), steps, self.check_every, true)
    }
}

/// Traces the loop of [`iterate_until_all`] once, for `init`'s shapes, and returns it ready to run
/// on any number of states of those shapes: a solver that steps in chunks (to look at the state
/// between chunks) pays for the trace once instead of once per chunk.
///
/// `step` is called once, here, and not kept, so it may borrow. Unlike [`iterate_until_all`] there
/// is no host loop to fall back to: a step that reads back, or states larger than one buffer,
/// give [`LumaError::Unsupported`]; so does `until` with 0 steps.
#[track_caller]
pub fn compile_until_all<S: Fields>(
    init: &S,
    until: Until<'_, S>,
    step: impl Fn(&S, &Array<f32>) -> S,
) -> Result<CompiledUntil<S>> {
    let location = Location::caller();
    compile(init, &until, &step).map_err(|failure| match failure {
        Failure::Inherited(error) => LumaError::clone(&error),
        Failure::New(error) => LumaError::Op {
            op: "compile_until_all",
            location,
            source: Arc::new(error),
        },
    })
}

fn compile<S: Fields>(
    init: &S,
    until: &Until<'_, S>,
    step: &impl Fn(&S, &Array<f32>) -> S,
) -> Outcome<CompiledUntil<S>> {
    let unsupported = || {
        Failure::New(LumaError::Unsupported(
            "compile_until_all requires single-buffer states without host readbacks in the step"
                .into(),
        ))
    };
    let arrays = init.arrays();
    let ctx = arrays[0].context().clone();
    if arrays.iter().any(|array| !ctx.same_as(array.context())) {
        return Err(Failure::New(LumaError::ContextMismatch));
    }
    for array in &arrays {
        if let Some(error) = array.error() {
            return Err(Failure::Inherited(Arc::new(error.clone())));
        }
    }
    if until.max_steps == 0 {
        return Err(Failure::New(LumaError::Unsupported(
            "compile_until_all needs at least one step".into(),
        )));
    }
    for array in &arrays {
        if array.storage()?.segments() != 1 {
            return Err(unsupported());
        }
    }
    let shapes: Vec<Shape> = arrays.iter().map(|array| array.shape()).collect();
    let names: Vec<String> = until.records.iter().map(|(name, _)| name.clone()).collect();
    let history = History::new(names.clone());
    let conds: Vec<Cond> = until.stops.iter().map(|stop| stop(&history)).collect();

    let stepv = Array::<f32>::from_slice_on(&ctx, &[0.0], 1)?;
    let constants = gate_logic(0, &conds, false).1;
    let mut gate_buffers = fold_buffers(&ctx, until, constants.len());
    let (mut trace, probes) = Trace::start(&arrays)?;
    let probes: S = state_from(probes);
    let next = step(&probes, &stepv);
    let records = trace_records(&ctx, until, &shapes, &probes, &next, &stepv, &conds, gate_buffers.as_ref())?;
    jit::force(&next.arrays())?;
    records.force()?;
    let plan = trace.finish_plan().ok_or_else(unsupported)?;
    let rec_storages = records.storages().ok_or_else(unsupported)?;
    let stepper = gated(&ctx, plan, &trace, &next, &rec_storages, until.max_steps, records.fold()).map_err(|_| unsupported())?;
    let gate = gate_kernel(&ctx, &conds, rec_storages.len(), &records)?;
    let spec = Spec { conditions: conds.len(), capacity: until.max_steps, fused: records.fused(), fold: records.fold(), constants };
    let mut gate_loop = GateLoop::new(&ctx, stepper, gate, stepv, rec_storages, spec, gate_buffers.take())?;
    gate_loop.compiled = true;
    Ok(CompiledUntil {
        ctx,
        shapes,
        names,
        max_steps: until.max_steps,
        check_every: until.check_every,
        gate_loop,
        state: PhantomData,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hist() -> History {
        History::new(vec!["a".into(), "b".into()])
    }

    #[test]
    fn the_gate_state_read_back_is_checked_before_it_indexes_host_arrays() {
        // A loop that ran all its steps, and one a condition (the second of two) stopped at step 3.
        assert_eq!(gate_outcome(0, 0, 0, 16, 16, 2).ok(), Some((16, StopReason::MaxSteps)));
        assert_eq!(gate_outcome(1, 3, 2, 16, 16, 2).ok(), Some((3, StopReason::Predicate(1))));
        assert_eq!(gate_outcome(1, 16, 1, 16, 16, 2).ok(), Some((16, StopReason::Predicate(0))));
        // Words the gate cannot write: float bit patterns seen in the wild, a step past the maximum,
        // step 0, a condition of 0 (the gate stores `index + 1`) or beyond the last one.
        for (count, reason) in [(4_294_967_231, 1), (1_065_353_216, 1), (17, 1), (0, 1), (3, 0), (3, 3)] {
            let Err(failure) = gate_outcome(1, count, reason, 16, 16, 2) else { panic!("a corrupt state is an error") };
            let message = failure.into_error().to_string();
            assert!(message.contains("cannot have written"), "{message}");
        }
    }

    #[test]
    fn host_evaluation_follows_the_definitions() {
        let h = hist();
        let histories = vec![vec![1.0, 4.0, 2.0, 3.0], vec![5.0, 5.0, 5.0, 1.0]];
        let a = &h["a"];
        let cond = (a.peak() / a.last()).ge(1.3).and(a.window_max(2).le(3.0));
        // Peak 4, last 3: ratio 1.33; the last two values are 2 and 3.
        assert!(eval_host(&cond, &histories, 4));
        assert!(!eval_host(&cond, &histories, 2));
        assert!(eval_host(&(h["b"].trough() - h["b"].first()).lt(0.0), &histories, 4));
        assert!(eval_host(&h.step().ge(4.0), &histories, 4));
        assert!(eval_host(&a.first().gt(0.5).not().or(a.last().ge(3.0)), &histories, 4));
    }

    #[test]
    fn the_gate_source_names_every_record_and_condition() {
        let h = hist();
        let conds = [h["a"].window_min(3).lt(h["b"].last()), h["b"].peak().ge(2.0)];
        let source = gate_source(2, &conds, false).0;
        assert!(source.contains("rec1"));
        assert!(source.contains("wmin(i, 0u, 3u)"));
        assert!(source.contains("reason = 2u"));
    }

    #[test]
    fn host_window_mean_and_prev() {
        let h = hist();
        let a = &h["a"];
        let histories = vec![vec![1.0, 4.0, 2.0, 3.0], vec![1.0, f32::NAN, 3.0, 4.0]];
        assert_eq!(eval_stat(&a.window_mean(2), &histories, 4), 2.5);
        // Fewer values than the window: all of them.
        assert_eq!(eval_stat(&a.window_mean(10), &histories, 2), 2.5);
        assert_eq!(eval_stat(&a.window_mean(0), &histories, 4), 3.0);
        assert_eq!(eval_stat(&a.prev(0), &histories, 4), 3.0);
        assert_eq!(eval_stat(&a.prev(1), &histories, 4), 2.0);
        assert_eq!(eval_stat(&a.prev(9), &histories, 4), 1.0);
        let b = &h["b"];
        assert!(eval_stat(&b.window_mean(2), &histories, 3).is_nan());
        assert_eq!(eval_stat(&b.window_mean(2), &histories, 1), 1.0);
        assert_eq!(eval_stat(&b.window_mean(2), &histories, 4), 3.5);
        let source = gate_source(2, &[a.window_mean(3).lt(a.prev(2))], true).0;
        assert!(source.contains("wmean(i, 0u, 3u)"));
        assert!(source.contains("(i - min(2u, i))"));
    }

    /// The first step after which `cond` holds on `history`, as the host evaluates it.
    fn host_stop(cond: &Cond, history: &[f32]) -> Option<usize> {
        let histories = vec![history.to_vec()];
        (1..=history.len()).find(|&done| eval_host(cond, &histories, done))
    }

    #[test]
    fn the_gpu_evaluates_window_mean_and_prev_like_the_host() {
        let ctx = Context::new().unwrap();
        let k = 4;
        let make = || {
            let rule = move |h: &History| {
                let m = &h["max"];
                m.window_mean(k)
                    .lt(0.01)
                    .and((m.prev(1) - m.last()).abs().le(1.0))
            };
            let stationary = move |h: &History| {
                let m = &h["max"];
                (m.window_mean(2 * k) - m.window_mean(k))
                    .abs()
                    .le(1e-6)
                    .and(m.prev(1).ge(m.last()))
            };
            Until::new(60)
                .record("max", |s: &(Array<f32>,)| s.0.max())
                .stop_when(rule)
                .stop_when(stationary)
        };
        let init = (Array::from_slice_on(&ctx, &[1.0f32; 16], 16).unwrap(),);
        let run = iterate_until_all(init, make(), |s, _| (&s.0 * 0.5,)).unwrap();
        let history = run.history("max").unwrap();
        let h = History::new(vec!["max".into()]);
        let conds: Vec<Cond> = make().stops.iter().map(|stop| stop(&h)).collect();
        let first = conds.iter().map(|c| host_stop(c, history)).enumerate();
        let (index, steps) = first
            .filter_map(|(i, s)| s.map(|s| (i, s)))
            .min_by_key(|&(i, s)| (s, i))
            .unwrap();
        assert_eq!(
            (run.steps, run.reason),
            (steps, StopReason::Predicate(index))
        );

        // A NaN in the window stops nothing, on either side.
        let nan = (Array::from_slice_on(&ctx, &[f32::NAN; 4], 4).unwrap(),);
        let run = iterate_until_all(
            nan,
            Until::new(5)
                .record("max", |s: &(Array<f32>,)| s.0.max())
                .stop_when(|h| h["max"].window_mean(2).ge(-1e30)),
            |s, _| (&s.0 * 0.5,),
        )
        .unwrap();
        assert_eq!((run.steps, run.reason), (5, StopReason::MaxSteps));
    }

    #[test]
    fn a_compiled_loop_runs_chunks_like_iterate_until_all() {
        type One = (Array<f32>,);
        let ctx = Context::new().unwrap();
        let scale = 1.0f32;
        let step = |s: &One, k: &Array<f32>| (&(&s.0 + k) * scale,);
        let until = |n: usize| Until::new(n).record("max", |s: &One| s.0.max());
        let start = || (Array::from_slice_on(&ctx, &[0.0f32; 8], 8).unwrap(),);

        let mut compiled = compile_until_all(&start(), until(8), step).unwrap();
        let mut want = start();
        let mut got = start();
        for (n, shorter) in [(8, false), (8, false), (3, true)] {
            let expected = iterate_until_all(want, until(n), step).unwrap();
            // The state of a run goes straight into the next one.
            let run = if shorter {
                compiled.run_steps(got, 3)
            } else {
                compiled.run(got)
            }
            .unwrap();
            assert_eq!((run.steps, run.reason), (expected.steps, expected.reason));
            assert_eq!(run.history("max"), expected.history("max"));
            assert_eq!(
                run.state.0.to_vec().unwrap(),
                expected.state.0.to_vec().unwrap()
            );
            want = expected.state;
            got = run.state;
        }
        // 0 + 1 + .. + 7 twice, then 0 + 1 + 2.
        assert_eq!(got.0.to_vec().unwrap(), vec![59.0; 8]);
    }

    /// A CPU context whose reductions over several workgroups finish in their first launch or not.
    fn cpu_ctx(single_pass: bool) -> Context {
        use luma_core::{BackendKind, DeviceKind};
        let options = crate::context::ContextOptions { backend: BackendKind::Cpu, device: DeviceKind::Cpu, ..crate::context::ContextOptions::from_env() }.calibrate(false);
        let ctx = Context::with_tuning(options, |tuning| tuning.single_pass_reduce = single_pass).unwrap();
        ctx.set_cost_model(crate::jit::cost::CostModel::TEST);
        ctx
    }

    fn spike(n: usize) -> Vec<f32> {
        (0..n).map(|i| if i % 97 == 0 { 1.0 } else { ((i * 7919) % 101) as f32 / 101.0 }).collect()
    }

    type One = (Array<f32>,);
    type Rule = Box<dyn Fn() -> Until<'static, One>>;

    /// A step of the heat equation on a ring.
    fn heat(s: &One, _: &Array<f32>) -> One {
        let lap = &(&s.0.shift(1) + &s.0.shift(-1)) - &(&s.0 * 2.0f32);
        (&s.0 + &(&lap * 0.25f32),)
    }

    /// How a loop was run: the steps, why, the history and the state, and what its step launched.
    #[derive(Debug, PartialEq)]
    struct Outcome1 {
        steps: usize,
        reason: StopReason,
        norm: Vec<f32>,
        state: Vec<f32>,
    }

    /// Launches of a step of the last loop run: its plan's, and one more for a gate of its own.
    fn step_launches() -> (usize, bool) {
        let (launches, fold) = LAST_PLAN.with(|last| last.get()).expect("the loop ran a traced step");
        (launches + usize::from(fold.is_none()), fold.is_some())
    }

    /// Runs the heat loop with the record of the state traced after the step (`separate_record`),
    /// the gate as a launch of its own (`separate_gate`), or as it comes.
    fn run_heat(ctx: &Context, data: &[f32], until: Until<'_, One>, separate_record: bool, separate_gate: bool) -> (Outcome1, (usize, bool)) {
        crate::jit::SEPARATE_RECORD.with(|flag| flag.set(separate_record));
        crate::jit::SEPARATE_GATE.with(|flag| flag.set(separate_gate));
        LAST_PLAN.with(|last| last.set(None));
        let run = iterate_until_all((Array::from_slice_on(ctx, data, data.len()).unwrap(),), until, heat);
        crate::jit::SEPARATE_RECORD.with(|flag| flag.set(false));
        crate::jit::SEPARATE_GATE.with(|flag| flag.set(false));
        let run = run.unwrap();
        let launches = step_launches();
        let norm = run.history("norm").unwrap().to_vec();
        (Outcome1 { steps: run.steps, reason: run.reason, norm, state: run.state.0.to_vec().unwrap() }, launches)
    }

    fn norm_until<'a>(max: usize) -> Until<'a, One> {
        Until::new(max).record("norm", |s: &One| s.0.norm())
    }

    #[test]
    fn a_step_and_the_record_of_its_input_run_as_one_launch_with_the_results_of_two() {
        // 16384 elements: several workgroups for a reduction, so that a single pass is one
        // launch that finishes it; 100: one.
        for single_pass in [false, true] {
            let ctx = cpu_ctx(single_pass);
            for n in [100usize, 16384] {
                assert_eq!(ctx.tuning().reduce_groups(n as u32) > 1, n > 100);
                let data = spike(n);
                // The norm after the 7th step, which a rule stops at: the first step it is at most
                // that is the 7th, as the norm only falls.
                let (all, _) = run_heat(&ctx, &data, norm_until(12), true, true);
                let at_7 = all.norm[6];
                let stops: Vec<(&str, Rule)> = vec![
                    ("norm at most the 7th", Box::new(move || norm_until(12).stop_when(move |h| h["norm"].last().le(at_7)))),
                    ("after the first step", Box::new(|| norm_until(12).stop_when(|h| h.step().ge(1.0)))),
                    ("after the second step", Box::new(|| norm_until(12).stop_when(|h| h.step().ge(2.0)))),
                    ("at the last step", Box::new(|| norm_until(5).stop_when(|h| h.step().ge(5.0)))),
                    ("never, one step", Box::new(|| norm_until(1))),
                    ("never, five steps", Box::new(|| norm_until(5))),
                    ("in the middle of a block", Box::new(|| norm_until(20).stop_when(|h| h.step().ge(3.0)).check_every(2))),
                    ("by a window", Box::new(|| norm_until(30).stop_when(|h| (h["norm"].window_max(3) - h["norm"].window_min(3)).lt(0.2)).check_every(4))),
                    ("by the second rule", Box::new(move || norm_until(12).stop_when(|h| h["norm"].last().lt(0.0)).stop_when(move |h| h["norm"].last().le(at_7)))),
                ];
                for (name, until) in &stops {
                    let (old, old_launches) = run_heat(&ctx, &data, until(), true, true);
                    let (fused, fused_launches) = run_heat(&ctx, &data, until(), false, true);
                    let (folded, folded_launches) = run_heat(&ctx, &data, until(), false, false);
                    let what = format!("{name}, {n} elements, single pass {single_pass}");
                    assert_eq!(fused, old, "{what}: the record in the step's launch");
                    assert_eq!(folded, old, "{what}: and the gate");
                    assert!(old.steps >= 1, "{what}");
                    // A launch for the step and one for its record (two with a pass to finish it), and the gate.
                    let record_launches = if n > 100 && !single_pass { 2 } else { 1 };
                    assert_eq!(old_launches, (1 + record_launches + 1, false), "{what}");
                    assert_eq!(fused_launches, (record_launches + 1, false), "{what}");
                    // The gate runs in the step's launch where the backend finishes a reduction in one.
                    assert_eq!(folded_launches, if single_pass { (1, true) } else { fused_launches }, "{what}");
                }
                let (stopped, _) = run_heat(&ctx, &data, norm_until(12).stop_when(move |h| h["norm"].last().le(at_7)), false, false);
                assert_eq!((stopped.steps, stopped.reason), (7, StopReason::Predicate(0)));
                assert_eq!(stopped.norm, all.norm[..7]);
            }
        }
    }

    #[test]
    fn the_state_a_loop_returns_is_that_of_the_step_it_stopped_at_not_of_the_speculated_one() {
        let ctx = cpu_ctx(true);
        let data = spike(16384);
        let mut state = Array::from_slice_on(&ctx, &data, data.len()).unwrap();
        let mut norms = Vec::new();
        for _ in 0..9 {
            state = heat(&(state,), &Array::from_slice_on(&ctx, &[0.0], 1).unwrap()).0;
            norms.push(state.norm().item().unwrap());
        }
        for steps in 1..=9usize {
            let until = norm_until(12).stop_when(move |h| h.step().ge(steps as f32));
            let (run, _) = run_heat(&ctx, &data, until, false, false);
            assert_eq!(run.steps, steps);
            assert_eq!(run.norm, norms[..steps]);
            let mut want = Array::from_slice_on(&ctx, &data, data.len()).unwrap();
            for _ in 0..steps {
                want = heat(&(want,), &Array::from_slice_on(&ctx, &[0.0], 1).unwrap()).0;
            }
            assert_eq!(run.state, want.to_vec().unwrap(), "{steps} steps");
        }
    }

    #[test]
    fn a_step_that_reads_its_index_keeps_the_gate_in_a_launch_of_its_own() {
        // The gate writes the index the step reads: not in one kernel.
        let ctx = cpu_ctx(true);
        let data = spike(16384);
        let step = |s: &One, k: &Array<f32>| (&(&s.0 * 0.5f32) + k,);
        let run = |separate_record: bool| {
            crate::jit::SEPARATE_RECORD.with(|flag| flag.set(separate_record));
            LAST_PLAN.with(|last| last.set(None));
            let run = iterate_until_all(
                (Array::from_slice_on(&ctx, &data, data.len()).unwrap(),),
                Until::new(9).record("norm", |s: &One| s.0.norm()).stop_when(|h| h.step().ge(6.0)),
                step,
            );
            crate::jit::SEPARATE_RECORD.with(|flag| flag.set(false));
            let run = run.unwrap();
            (run.steps, run.history("norm").unwrap().to_vec(), run.state.0.to_vec().unwrap(), step_launches())
        };
        let (old, fused) = (run(true), run(false));
        assert_eq!(old.0, 6);
        assert_eq!((fused.0, &fused.1, &fused.2), (old.0, &old.1, &old.2));
        assert_eq!((old.3, fused.3), ((3, false), (2, false)));
    }

    #[test]
    fn records_that_are_not_one_reduction_of_the_step_keep_the_gate_and_the_records_after_the_step() {
        let ctx = cpu_ctx(true);
        let data = spike(16384);
        let until = || {
            Until::new(8)
                .record("norm", |s: &One| s.0.norm())
                .record("max", |s: &One| s.0.max())
                .stop_when(|h| h["max"].last().lt(0.9))
        };
        LAST_PLAN.with(|last| last.set(None));
        let two = iterate_until_all((Array::from_slice_on(&ctx, &data, data.len()).unwrap(),), until(), heat).unwrap();
        let (launches, fold) = LAST_PLAN.with(|last| last.get()).unwrap();
        assert_eq!(fold, None);
        // The step, the two records (the gate is a launch of its own).
        assert_eq!(launches, 3);
        let mut state = Array::from_slice_on(&ctx, &data, data.len()).unwrap();
        for t in 0..two.steps {
            state = heat(&(state,), &Array::from_slice_on(&ctx, &[0.0], 1).unwrap()).0;
            assert_eq!(two.history("max").unwrap()[t], state.max().item().unwrap());
            assert_eq!(two.history("norm").unwrap()[t], state.norm().item().unwrap());
        }
        assert_eq!(two.state.0.to_vec().unwrap(), state.to_vec().unwrap());
        // A record that is a reduction of something else than the state's step (an arithmetic
        // of two reductions) is not one reduction either.
        let sum = iterate_until_all(
            (Array::from_slice_on(&ctx, &data, data.len()).unwrap(),),
            Until::new(4).record("spread", |s: &One| &s.0.max() - &s.0.min()),
            heat,
        )
        .unwrap();
        assert_eq!(sum.history("spread").unwrap().len(), 4);
    }

    #[test]
    fn a_step_that_reads_back_runs_on_the_host_with_the_records_of_the_states_it_made() {
        // The readback aborts the recording of the step: nothing can be replayed, and the launch that
        // reduced the record of the state the step read (and maybe ran a gate) ran for real.
        let ctx = cpu_ctx(true);
        let data = spike(16384);
        let step = |s: &One, _: &Array<f32>| {
            let total = s.0.sum().item().unwrap();
            (&s.0 * (1.0 / (1.0 + total.abs())).max(0.5f32),)
        };
        let run = |separate_record: bool| {
            crate::jit::SEPARATE_RECORD.with(|flag| flag.set(separate_record));
            LAST_PLAN.with(|last| last.set(None));
            let run = iterate_until_all(
                (Array::from_slice_on(&ctx, &data, data.len()).unwrap(),),
                Until::new(6).record("norm", |s: &One| s.0.norm()).stop_when(|h| h.step().ge(4.0)),
                step,
            );
            crate::jit::SEPARATE_RECORD.with(|flag| flag.set(false));
            let run = run.unwrap();
            (run.steps, run.history("norm").unwrap().to_vec(), run.state.0.to_vec().unwrap())
        };
        let (old, fused) = (run(true), run(false));
        assert_eq!(old.0, 4);
        assert_eq!(fused, old);
        assert_eq!(LAST_PLAN.with(|last| last.get()), None, "the step was replayed");
    }

    #[test]
    fn a_state_of_several_arrays_is_stored_by_the_launch_that_reduces_the_record() {
        type Two = (Array<f32>, Array<f32>);
        let ctx = cpu_ctx(true);
        let (a, b) = (spike(16384), spike(16384).into_iter().rev().collect::<Vec<_>>());
        let start = || (Array::from_slice_on(&ctx, &a, a.len()).unwrap(), Array::from_slice_on(&ctx, &b, b.len()).unwrap());
        let rotate = |s: &Two, _: &Array<f32>| (&s.0 + &(&s.1 * 0.1f32), &s.1 - &(&s.0 * 0.1f32));
        // The second array passes through: not a step to store, the record follows the step.
        let passes = |s: &Two, _: &Array<f32>| (&s.0 * 0.9f32, s.1.clone());
        let until = || Until::new(10).record("norm", |s: &Two| s.0.norm()).stop_when(|h| h["norm"].last().gt(1.0e9));
        type Step<'a> = &'a dyn Fn(&Two, &Array<f32>) -> Two;
        let run = |step: Step<'_>, separate_record: bool| {
            crate::jit::SEPARATE_RECORD.with(|flag| flag.set(separate_record));
            LAST_PLAN.with(|last| last.set(None));
            let run = iterate_until_all(start(), until(), step);
            crate::jit::SEPARATE_RECORD.with(|flag| flag.set(false));
            let run = run.unwrap();
            let launches = LAST_PLAN.with(|last| last.get());
            ((run.steps, run.reason, run.history("norm").unwrap().to_vec(), run.state.0.to_vec().unwrap(), run.state.1.to_vec().unwrap()), launches)
        };
        let (old, old_launches) = run(&rotate, true);
        let (fused, fused_launches) = run(&rotate, false);
        assert_eq!(fused, old);
        assert_eq!(old.0, 10);
        // Both arrays and the record in one launch; before, a launch to store both and one for the record.
        assert_eq!((old_launches, fused_launches), (Some((2, None)), Some((1, Some(0)))));
        let (old, old_launches) = run(&passes, true);
        let (fused, fused_launches) = run(&passes, false);
        assert_eq!(fused, old);
        assert_eq!(fused_launches, old_launches);
    }

    #[test]
    fn a_step_that_writes_a_captured_array_applies_its_writes_once_per_step() {
        // The launch that finds the loop stopped (or the last one) runs the whole step, so a step
        // with effects on arrays outside its state keeps its records after it.
        let ctx = cpu_ctx(true);
        let data = spike(16384);
        let index = Array::<u32>::from_slice_on(&ctx, &[0], 1).unwrap();
        for stop_at in [4.0f32, 100.0] {
            for separate_record in [true, false] {
                let counted = std::cell::RefCell::new(Array::from_slice_on(&ctx, &[0.0f32, 0.0], 2).unwrap());
                let sliced = std::cell::RefCell::new(Array::from_slice_on(&ctx, &[1.0f32, 1.0, 1.0, 1.0], 4).unwrap());
                let step = |s: &One, _: &Array<f32>| {
                    counted.borrow_mut().put_add(&index, 1.0f32).unwrap();
                    let current = sliced.borrow().slice(0, 2..4);
                    sliced.borrow_mut().slice_mut(0, 2..4).assign(&current + 1.0f32).unwrap();
                    (&s.0 * 0.5f32,)
                };
                crate::jit::SEPARATE_RECORD.with(|flag| flag.set(separate_record));
                LAST_PLAN.with(|last| last.set(None));
                let run = iterate_until_all(
                    (Array::from_slice_on(&ctx, &data, data.len()).unwrap(),),
                    Until::new(6).record("norm", |s: &One| s.0.norm()).stop_when(move |h| h.step().ge(stop_at)),
                    step,
                );
                crate::jit::SEPARATE_RECORD.with(|flag| flag.set(false));
                let run = run.unwrap();
                let applied = counted.borrow().to_vec().unwrap()[0];
                assert_eq!(applied, run.steps as f32, "stop at {stop_at}, record separate: {separate_record}");
                assert_eq!(sliced.borrow().to_vec().unwrap(), vec![1.0, 1.0, 1.0 + run.steps as f32, 1.0 + run.steps as f32], "stop at {stop_at}, record separate: {separate_record}");
                let launched = LAST_PLAN.with(|last| last.get());
                assert!(launched.is_some_and(|(_, fold)| fold.is_none()), "{launched:?}");
            }
        }
    }

    #[test]
    fn a_step_that_reads_a_pending_expression_through_a_view_is_still_one_launch() {
        // The expression is written out where the view reads it, in the program that reduces the record.
        let ctx = cpu_ctx(true);
        let data = spike(16384);
        let step = |s: &One, _: &Array<f32>| {
            let e = &(&s.0 * 0.5f32) + 0.25f32;
            (&e.flip(0) + &e,)
        };
        let run = |separate_record: bool| {
            crate::jit::SEPARATE_RECORD.with(|flag| flag.set(separate_record));
            LAST_PLAN.with(|last| last.set(None));
            let run = iterate_until_all(
                (Array::from_slice_on(&ctx, &data, data.len()).unwrap(),),
                Until::new(7).record("norm", |s: &One| s.0.norm()).stop_when(|h| h.step().ge(5.0)),
                step,
            );
            crate::jit::SEPARATE_RECORD.with(|flag| flag.set(false));
            let run = run.unwrap();
            ((run.steps, run.history("norm").unwrap().to_vec(), run.state.0.to_vec().unwrap()), LAST_PLAN.with(|last| last.get()))
        };
        let (old, old_launches) = run(true);
        let (fused, fused_launches) = run(false);
        assert_eq!(fused, old);
        assert_eq!(old.0, 5);
        assert_eq!(fused_launches, Some((1, Some(0))), "{old_launches:?}");
    }

    #[test]
    fn thresholds_of_a_stop_rule_are_state_of_the_gate_not_part_of_its_kernels() {
        let ctx = cpu_ctx(true);
        let data = spike(16384);
        let (all, _) = run_heat(&ctx, &data, norm_until(12), true, true);
        let rule = |at: f32, steps: f32| norm_until(12).stop_when(move |h| h["norm"].last().le(at).or(h.step().ge(steps)));
        for separate_gate in [false, true] {
            let built = |ctx: &Context| ctx.jit().stats().built;
            let (first, _) = run_heat(&ctx, &data, rule(all.norm[6], 99.0), false, separate_gate);
            let before = built(&ctx);
            let (second, _) = run_heat(&ctx, &data, rule(all.norm[3], 99.0), false, separate_gate);
            let (third, _) = run_heat(&ctx, &data, rule(0.0, 2.0), false, separate_gate);
            assert_eq!(built(&ctx), before, "a kernel was built for other thresholds (gate of its own: {separate_gate})");
            assert_eq!((first.steps, second.steps, third.steps), (7, 4, 2));
            assert_eq!(second.norm, all.norm[..4]);
        }
    }

    #[test]
    fn a_compiled_loop_folds_its_gate_and_runs_chunks_like_iterate_until_all() {
        let ctx = cpu_ctx(true);
        let data = spike(16384);
        // Stops in the middle of a block, so that the launches after the stop find the loop stopped.
        let until = |n: usize| norm_until(n).stop_when(|h| h.step().ge(4.0));
        let start = || (Array::from_slice_on(&ctx, &data, data.len()).unwrap(),);
        LAST_PLAN.with(|last| last.set(None));
        let mut compiled = compile_until_all(&start(), until(6), heat).unwrap();
        assert_eq!(LAST_PLAN.with(|last| last.get()), Some((1, Some(0))));
        let (mut want, mut got) = (start(), start());
        // Runs of a span of steps again and again: it is captured as a graph, which runs the gate
        // folded into the step's launch like any other.
        for n in [6, 6, 6, 6, 5, 2, 6] {
            let expected = iterate_until_all(want, until(n), heat).unwrap();
            let run = compiled.run_steps(got, n).unwrap();
            assert_eq!((run.steps, run.reason), (expected.steps, expected.reason));
            assert_eq!(run.history("norm"), expected.history("norm"));
            assert_eq!(run.state.0.to_vec().unwrap(), expected.state.0.to_vec().unwrap());
            want = expected.state;
            got = run.state;
        }
        assert!(compiled.gate_loop.graphs.holds_graphs(), "no span of launches was captured");
    }
}
