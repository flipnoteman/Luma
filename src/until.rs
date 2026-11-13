//! `iterate_until_all`: a loop over a record of arrays whose stop rule is evaluated on the GPU.
//!
//! The step is traced once (its launches recorded) and replayed; between two steps a gate kernel
//! stores the recorded scalars in a history, evaluates the stop conditions on it and, once one
//! holds, zeroes the dispatch arguments of the steps after it. The host waits once per block of
//! steps: the first block is 16 steps and every block that ends without a stop doubles the next
//! one, up to 128 (or every block is [`Until::check_every`] steps); the result never depends on
//! the schedule. A step that cannot be replayed (it reads back, spans several buffers, ...) runs on the
//! host instead, with the same stop rule evaluated by `eval_host`; both evaluations agree.

use std::fmt::Write as _;
use std::ops::{Add, Div, Index, Mul, Neg, Sub};
use std::panic::Location;
use std::sync::Arc;

use wgpu::BufferUsages;

use crate::array::{Array, Failure, Storage};
use crate::context::{Context, RawArg, STORAGE_USAGE};
use crate::error::{LumaError, Result};
use crate::fields::Fields;
use crate::jit::{self, Plan, Stepper, Trace};
use crate::ops::{launch_1d, Outcome};
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

fn stat_source(stat: &Stat) -> String {
    match &*stat.0 {
        StatNode::Const(c) => format!("bitcast<f32>({}u)", c.to_bits()),
        StatNode::Last(r) => format!("v{r}"),
        StatNode::First(r) => format!("hist[{r}u * p.max_steps]"),
        StatNode::Peak(r) => format!("agg[{}u]", 2 * r),
        StatNode::Trough(r) => format!("agg[{}u]", 2 * r + 1),
        StatNode::WindowMax(r, k) => format!("wmax(i, {r}u, {k}u)"),
        StatNode::WindowMin(r, k) => format!("wmin(i, {r}u, {k}u)"),
        StatNode::Step => "f32(i + 1u)".to_string(),
        StatNode::Bin(op, a, b) => {
            let symbol = match op {
                Arith::Add => "+",
                Arith::Sub => "-",
                Arith::Mul => "*",
                Arith::Div => "/",
            };
            format!("({} {symbol} {})", stat_source(a), stat_source(b))
        }
        StatNode::Abs(a) => format!("abs({})", stat_source(a)),
    }
}

fn cond_source(cond: &Cond) -> String {
    match &*cond.0 {
        CondNode::Cmp(op, a, b) => {
            let symbol = match op {
                Compare::Lt => "<",
                Compare::Le => "<=",
                Compare::Gt => ">",
                Compare::Ge => ">=",
            };
            // A comparison with a NaN never holds, even where fast math ignores NaNs.
            let (a, b) = (stat_source(a), stat_source(b));
            format!("(!isn({a}) && !isn({b}) && ({a} {symbol} {b}))")
        }
        CondNode::And(a, b) => format!("({} && {})", cond_source(a), cond_source(b)),
        CondNode::Or(a, b) => format!("({} || {})", cond_source(a), cond_source(b)),
        CondNode::Not(a) => format!("!{}", cond_source(a)),
    }
}

/// The gate kernel run after every step: records the scalars, evaluates the conditions and
/// either copies the dispatch arguments of the next step from the template or zeroes them.
fn gate_source(records: usize, conds: &[Cond]) -> String {
    let mut s = String::new();
    s.push_str("struct Params { step: u32, words: u32, max_steps: u32, _pad: u32 }\n");
    s.push_str("@group(1) @binding(0) var<storage, read_write> st: array<u32>;\n");
    s.push_str("@group(1) @binding(1) var<storage, read_write> args: array<u32>;\n");
    s.push_str("@group(1) @binding(2) var<storage, read> tmpl: array<u32>;\n");
    s.push_str("@group(1) @binding(3) var<storage, read_write> hist: array<f32>;\n");
    s.push_str("@group(1) @binding(4) var<storage, read_write> agg: array<f32>;\n");
    s.push_str("@group(1) @binding(5) var<storage, read_write> stepv: array<f32>;\n");
    for r in 0..records {
        let _ = writeln!(s, "@group(1) @binding({}) var<storage, read> rec{r}: array<f32>;", 6 + r);
    }
    let _ = writeln!(s, "@group(1) @binding({}) var<uniform> p: Params;", 6 + records);
    s.push_str(
        "fn isn(x: f32) -> bool { return (bitcast<u32>(x) & 0x7fffffffu) > 0x7f800000u; }
fn fmax(a: f32, b: f32) -> f32 { return select(select(max(a, b), a, isn(a)), b, isn(b)); }
fn fmin(a: f32, b: f32) -> f32 { return select(select(min(a, b), a, isn(a)), b, isn(b)); }
fn wmax(i: u32, r: u32, k: u32) -> f32 {
    var m = hist[r * p.max_steps + i];
    for (var j = i + 1u - min(k, i + 1u); j < i; j++) { m = fmax(m, hist[r * p.max_steps + j]); }
    return m;
}
fn wmin(i: u32, r: u32, k: u32) -> f32 {
    var m = hist[r * p.max_steps + i];
    for (var j = i + 1u - min(k, i + 1u); j < i; j++) { m = fmin(m, hist[r * p.max_steps + j]); }
    return m;
}
@compute @workgroup_size(1)
fn main() {
    if (st[0] != 0u) {
        for (var w = 0u; w < p.words; w++) { args[w] = 0u; }
        return;
    }
    let i = p.step;
",
    );
    if records == 0 {
        // Every binding must be used for the argument numbering to be reflected.
        s.push_str("    hist[0] = hist[0];\n    agg[0] = agg[0];\n");
    }
    for r in 0..records {
        let _ = writeln!(s, "    let v{r} = rec{r}[0];");
        let _ = writeln!(s, "    hist[{r}u * p.max_steps + i] = v{r};");
        let _ = writeln!(s, "    agg[{}u] = fmax(agg[{}u], v{r});", 2 * r, 2 * r);
        let _ = writeln!(s, "    agg[{}u] = fmin(agg[{}u], v{r});", 2 * r + 1, 2 * r + 1);
    }
    s.push_str("    var reason = 0u;\n");
    for (c, cond) in conds.iter().enumerate() {
        let _ = writeln!(s, "    if (reason == 0u && {}) {{ reason = {}u; }}", cond_source(cond), c + 1);
    }
    s.push_str(
        "    st[1] = i + 1u;
    if (reason != 0u) { st[0] = 1u; st[2] = reason; }
    for (var w = 0u; w < p.words; w++) { args[w] = select(tmpl[w], 0u, reason != 0u); }
    stepv[0] = f32(i + 1u);
}
",
    );
    s
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

    let apply_records = |next: &S| -> Outcome<Vec<Array<f32>>> {
        let mut recs = Vec::with_capacity(until.records.len());
        for (_, record) in &until.records {
            let value = record(next);
            if value.len() != 1 {
                return Err(Failure::New(LumaError::InvalidShape("recorded values must have one element".into())));
            }
            recs.push(value);
        }
        for (a, shape) in next.arrays().into_iter().zip(&shapes) {
            if a.shape() != *shape {
                return Err(Failure::New(LumaError::ShapeMismatch { lhs: a.shape().dims(), rhs: shape.dims() }));
            }
        }
        Ok(recs)
    };

    // Trace one step.
    let mut trace = None;
    let (next, recs) = if single {
        let (started, probes) = Trace::start(&arrays)?;
        trace = Some(started);
        let probes: S = state_from(probes);
        let next = step(&probes, &stepv);
        // The records read the new state (a residual of it, through slices): evaluated first, they
        // read its buffers instead of recomputing the step at every index they read.
        jit::force(&next.arrays())?;
        let recs = apply_records(&next)?;
        (next, recs)
    } else {
        let next = step(&init, &stepv);
        jit::force(&next.arrays())?;
        let recs = apply_records(&next)?;
        (next, recs)
    };
    drop(arrays);

    let plan = {
        jit::force(&next.arrays())?;
        jit::force(&recs.iter().collect::<Vec<_>>())?;
        trace.as_mut().and_then(Trace::finish_plan)
    };

    let stepper = match (plan, trace.as_ref()) {
        (Some(plan), Some(trace)) => gated(&ctx, plan, trace, &next, &recs, until.max_steps),
        _ => Err(None),
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

    if let Some(stepper) = stepper {
        let rec_storages = recs.iter().map(|rec| rec.storage().cloned()).collect::<std::result::Result<Vec<_>, _>>()?;
        return gated_run(&ctx, &shapes, until, &conds, names, stepper, &stepv, &rec_storages);
    }

    // The host loop.
    let mut histories: Vec<Vec<f32>> = vec![Vec::new(); until.records.len()];
    let mut state = init;
    let mut first = Some((next, recs));
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
/// any) to be run once.
fn gated(
    ctx: &Context,
    plan: Plan,
    trace: &Trace,
    next: &impl Fields,
    recs: &[Array<f32>],
    max_steps: usize,
) -> std::result::Result<Stepper, Option<Plan>> {
    let records = recs.len();
    let fits = recs.iter().all(|rec| rec.storage().is_ok_and(|storage| storage.segments() == 1))
        && 6 + records <= ctx.jit().max_inputs() + 1
        && (records.max(1) * max_steps * 4) as u64 <= ctx.max_buffer_bytes();
    if !fits {
        return Err(Some(plan));
    }
    let outs: std::result::Result<Vec<Arc<Storage>>, _> =
        next.arrays().into_iter().map(|array| array.storage().cloned()).collect();
    match outs {
        Ok(outs) => Stepper::new(ctx, plan, &trace.init, &trace.probe_storages, outs).map_err(Some),
        Err(_) => Err(Some(plan)),
    }
}

#[allow(clippy::too_many_arguments)]
fn gated_run<S: Fields>(
    ctx: &Context,
    shapes: &[Shape],
    until: &Until<'_, S>,
    conds: &[Cond],
    names: Vec<String>,
    stepper: Stepper,
    stepv: &Array<f32>,
    recs: &[Arc<Storage>],
) -> Outcome<Run<S>> {
    let records = recs.len();
    let max_steps = until.max_steps;
    let gate = ctx.jit().stencil_kernel(ctx, gate_source(records, conds), "UntilAllGate")?;
    let template = stepper.plan().template();
    let words = u32::try_from(template.len()).map_err(|_| LumaError::Gpu("too many recorded launches".into()))?;
    let args_bytes = u64::from(words) * 4;

    let rec_pins = recs.iter().map(|rec| rec.pin_quiet(0)).collect::<Result<Vec<_>>>()?;
    let step_pin = stepv.storage()?.pin_quiet(0)?;

    let mut owned = Vec::new();
    let mut take = |bytes: u64, usage: BufferUsages| -> Result<wgpu::Buffer> {
        let buffer = ctx.alloc_buffer(bytes, usage)?;
        owned.push(buffer.clone());
        Ok(buffer)
    };
    let outcome = (|| -> Outcome<(usize, StopReason, Vec<Vec<f32>>)> {
        let tmpl = take(args_bytes, STORAGE_USAGE)?;
        ctx.upload(&tmpl, bytemuck::cast_slice(&template))?;
        let st = take(16, STORAGE_USAGE)?;
        let indirect = STORAGE_USAGE | BufferUsages::INDIRECT;
        let mut args: Vec<wgpu::Buffer> = Vec::new();
        let hist_bytes = (records.max(1) * max_steps) as u64 * 4;
        let hist = take(hist_bytes, STORAGE_USAGE)?;
        let agg_bytes = 2 * records.max(1) as u64 * 4;
        let agg = take(agg_bytes, STORAGE_USAGE)?;
        let limits: Vec<f32> = (0..records.max(1)).flat_map(|_| [f32::NEG_INFINITY, f32::INFINITY]).collect();
        ctx.upload(&agg, bytemuck::cast_slice(&limits))?;

        let mut base = 0;
        let mut block = until.check_every.unwrap_or(crate::jit::FIRST_BLOCK);
        loop {
            let in_block = block.min(max_steps - base);
            while args.len() <= in_block {
                args.push(take(args_bytes, indirect)?);
            }
            ctx.upload(&st, bytemuck::cast_slice(&[0u32; 4]))?;
            ctx.upload(&args[0], bytemuck::cast_slice(&template))?;
            for k in 0..in_block {
                let i = base + k;
                let swaps = stepper.subst(i + 1);
                stepper.plan().replay(&swaps, Some(&args[k]))?;
                let uniform = [i as u32, words, max_steps as u32, 0];
                let resolved: Vec<wgpu::Buffer> = rec_pins
                    .iter()
                    .map(|pin| {
                        let buffer = pin.buffer();
                        swaps.iter().find(|(from, _)| from == buffer).map_or_else(|| buffer.clone(), |(_, to)| to.clone())
                    })
                    .collect();
                let mut raw: Vec<RawArg<'_>> = Vec::with_capacity(7 + records);
                raw.push(RawArg::write(&st, 16));
                raw.push(RawArg::write(&args[k + 1], args_bytes));
                raw.push(RawArg::read(&tmpl, args_bytes));
                raw.push(RawArg::write(&hist, hist_bytes));
                raw.push(RawArg::write(&agg, agg_bytes));
                raw.push(RawArg::write(step_pin.buffer(), 4));
                raw.extend(resolved.iter().map(|buffer| RawArg::read(buffer, 4)));
                raw.push(RawArg::Uniform(bytemuck::cast_slice(&uniform)));
                launch_1d(ctx, &gate, 1, &raw)?;
            }
            let [done, count, reason, _] = ctx.read::<u32>(&st, 16)?[..] else {
                return Err(Failure::New(LumaError::Gpu("iterate_until_all state is missing".into())));
            };
            base += in_block;
            block = crate::jit::next_block(in_block, until.check_every);
            if done != 0 || base >= max_steps {
                let steps = if done != 0 { count as usize } else { base };
                let reason =
                    if done != 0 { StopReason::Predicate(reason as usize - 1) } else { StopReason::MaxSteps };
                let all: Vec<f32> = ctx.read::<f32>(&hist, hist_bytes)?;
                let histories = (0..records).map(|r| all[r * max_steps..r * max_steps + steps].to_vec()).collect();
                return Ok((steps, reason, histories));
            }
        }
    })();
    for buffer in owned {
        ctx.recycle(buffer);
    }
    let (steps, reason, histories) = outcome?;

    let arrays = stepper
        .state_after(steps)
        .iter()
        .zip(shapes)
        .map(|(storage, shape)| Array::from_storage(ctx, *shape, Arc::clone(storage)))
        .collect();
    Ok(Run { state: state_from(arrays), steps, reason, history: names.into_iter().zip(histories).collect() })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hist() -> History {
        History::new(vec!["a".into(), "b".into()])
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
        let source = gate_source(2, &conds);
        assert!(source.contains("rec1"));
        assert!(source.contains("wmin(i, 0u, 3u)"));
        assert!(source.contains("reason = 2u"));
    }
}
