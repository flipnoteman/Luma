//! Random array programs for the fuzzer.
//!
//! A program is a list of statements over numbered variables plus an engine configuration. The
//! strategy draws a *tape* of random words and a deterministic builder turns it into a program
//! that is valid by construction: it tracks the type, shape and a bound on the values of every
//! variable, so no statement fails except the deliberately invalid ones (an error on both
//! sides is expected). Shrinking the tape shrinks the program.
//!
//! Two value domains: in the **exact** domain every value is a dyadic number with at most 22
//! significant bits (the builder tracks the bits), so every `f32` operation is exact on the
//! GPU too (no rounding, fused multiply-add changes nothing) and results compare bit for bit.
//! In the **math** domain inputs are arbitrary floats, `/ sqrt exp ...` are allowed, values stay
//! in `[-1e4, 1e4]` away from the poles, and results compare with a tolerance; comparisons,
//! `floor` and `select` masks need exact operands.

use luma::PadMode;
use proptest::prelude::*;

use super::oracle::{BinOp, CmpOp, Dims, RedOp, ScanOp, Ty, UnOp};

/// An operand: a variable or a constant (a scalar of the statement's element type).
#[derive(Clone, Debug, PartialEq)]
pub enum Opd {
    Var(usize),
    Const(f32),
}

/// The views `View` statements create.
#[derive(Clone, Debug, PartialEq)]
pub enum ViewOp {
    Slice { axis: usize, lo: usize, hi: usize },
    Pad { widths: Vec<(usize, usize)>, mode: PadMode },
    Flip { axis: usize },
    Transpose,
    ShiftCirc { offs: Vec<isize> },
    ShiftFill { offs: Vec<isize>, value: f32 },
    ShiftClamp { offs: Vec<isize> },
    ShiftMirror { offs: Vec<isize> },
    Take { idx: usize },
    Concat { axis: usize, others: Vec<usize> },
}

/// The views a write goes through: they keep every element distinct.
#[derive(Clone, Debug, PartialEq)]
pub enum InjView {
    InjSlice { axis: usize, lo: usize, hi: usize },
    InjFlip { axis: usize },
    InjTranspose,
}

/// How a loop runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LoopKind {
    /// A host loop that evaluates every step.
    PerStepEval,
    /// `iterate_all`.
    IterateAll,
    /// `iterate_until_all` stopping when the first state's maximum exceeds `1e9`.
    IterateUntilAll,
}

/// A statement. Statements that create arrays append one variable each (a loop appends its
/// final states); a statement that fails on both sides creates none.
#[derive(Clone, Debug, PartialEq)]
pub enum Stmt {
    Input { dims: Dims, ty: Ty, seed: u64 },
    Full { dims: Dims, ty: Ty, value: f32 },
    Range { dims: Dims, axis: usize, ty: Ty },
    Linspace { start: f32, end: f32, n: usize },
    Unary { src: usize, op: UnOp },
    Binary { lhs: Opd, rhs: Opd, op: BinOp },
    Compare { lhs: usize, rhs: Opd, op: CmpOp },
    Select { mask: usize, a: Opd, b: Opd, ty: Ty },
    View { src: usize, view: ViewOp },
    Reduce { src: usize, op: RedOp },
    SumWhere { src: usize, mask: usize },
    Scan { src: usize, axis: usize, op: ScanOp },
    AxisReduce { src: usize, axis: usize, op: RedOp },
    Assign { target: usize, path: Vec<InjView>, rhs: Opd },
    Put { target: usize, idx: Vec<u32>, vals: Opd },
    Compound { target: usize, op: BinOp, rhs: Opd },
    Write { target: usize, seed: u64 },
    Cache { src: usize },
    EvalVars { vars: Vec<usize> },
    Alias { src: usize },
    Release { var: usize },
    Fetch { var: usize },
    /// Runs `body` `steps` times on `states` (variables of one type); `next` are the body's
    /// variables (states and locals count from the first new variable) that become the next
    /// states. Appends the final states.
    Loop { kind: LoopKind, steps: usize, states: Vec<usize>, body: Vec<Stmt>, next: Vec<usize> },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CostChoice {
    Test,
    SlowLaunch,
    SlowArithmetic,
}

/// The engine configuration a program runs under.
#[derive(Clone, Debug, PartialEq)]
pub struct Config {
    pub max_nodes: usize,
    pub max_inlined: Option<usize>,
    pub max_live: Option<usize>,
    pub max_inputs: Option<usize>,
    pub assume_busy: bool,
    pub speculate: bool,
    pub simplify: bool,
    pub fast_math: bool,
    pub chunk_bytes: Option<u64>,
    pub memory_budget: Option<u64>,
    pub cost: CostChoice,
}

#[derive(Clone, PartialEq)]
pub struct Program {
    pub config: Config,
    /// Exact or math value domain (see the module docs); selects the comparison tolerance.
    pub math: bool,
    pub stmts: Vec<Stmt>,
}

impl Config {
    /// The configuration of the threaded runs: the engine's defaults.
    pub fn default_engine() -> Config {
        Config {
            max_nodes: 1024,
            max_inlined: None,
            max_live: None,
            max_inputs: None,
            assume_busy: false,
            speculate: true,
            simplify: true,
            fast_math: false,
            chunk_bytes: None,
            memory_budget: None,
            cost: CostChoice::Test,
        }
    }
}

const VEC_FIELDS: [&str; 10] = ["idx", "vars", "states", "body", "next", "path", "widths", "others", "offs", "stmts"];

impl Program {
    /// The program as a Rust literal for `regressions.rs` (one top-level statement per line).
    pub fn rust(&self) -> String {
        let vecs = |s: String| {
            let mut s = s;
            for name in &VEC_FIELDS {
                s = s.replace(&format!("{name}: ["), &format!("{name}: vec!["));
            }
            s
        };
        let stmts: Vec<String> = self.stmts.iter().map(|s| format!("        {},", vecs(format!("{s:?}")))).collect();
        format!(
            "Program {{\n    config: {:?},\n    math: {},\n    stmts: vec![\n{}\n    ],\n}}",
            self.config,
            self.math,
            stmts.join("\n")
        )
    }
}

/// The Rust literal, so that the failing program proptest prints can be pasted into a test.
impl std::fmt::Debug for Program {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.rust())
    }
}

/// `n` values of `ty` from `seed`: integers below 21 for `u32`, otherwise values of the domain.
pub fn values(seed: u64, ty: Ty, n: usize, math: bool) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let r = (s >> 32) as u32;
            match ty {
                Ty::U32 => (r % 21) as f32,
                _ if math => 0.5 + (r % 1500) as f32 / 1000.0,
                _ => ((r % 65) as f32 - 32.0) / 8.0,
            }
        })
        .collect()
}

// -- The tape --

struct Tape {
    words: Vec<u32>,
    pos: usize,
}

impl Tape {
    fn next(&mut self) -> u32 {
        let w = self.words.get(self.pos).copied().unwrap_or(0);
        self.pos += 1;
        w
    }

    fn below(&mut self, n: usize) -> usize {
        self.next() as usize % n.max(1)
    }

    /// True with probability `num / den`.
    fn chance(&mut self, num: usize, den: usize) -> bool {
        self.below(den) < num
    }

    fn pick<T: Copy>(&mut self, items: &[T]) -> T {
        items[self.below(items.len())]
    }

    fn done(&self) -> bool {
        self.pos >= self.words.len()
    }
}

// -- Value bounds --

/// What is known about the values of an array: they are multiples of `2^lsb` within `lo..=hi`,
/// and `exact` says that every `f32` operation on them is exact (few enough bits).
#[derive(Clone, Copy, Debug)]
struct Info {
    lsb: i32,
    lo: f64,
    hi: f64,
    exact: bool,
}

impl Info {
    fn mag(&self) -> f64 {
        self.lo.abs().max(self.hi.abs())
    }

    /// Significant bits of the largest value.
    fn bits(&self) -> i32 {
        let m = self.mag();
        if m == 0.0 {
            0
        } else {
            m.log2().floor() as i32 + 1 - self.lsb
        }
    }

    /// Bounds of an integer array (wrapping arithmetic: nothing to track).
    fn int() -> Info {
        Info { lsb: 0, lo: 0.0, hi: 4e9, exact: true }
    }

    fn of_const(c: f32) -> Info {
        let mut v = f64::from(c).abs();
        let mut lsb = 0;
        if v == 0.0 {
            return Info { lsb: 0, lo: 0.0, hi: 0.0, exact: true };
        }
        while v.fract() != 0.0 {
            v *= 2.0;
            lsb -= 1;
        }
        while v % 2.0 == 0.0 {
            v /= 2.0;
            lsb += 1;
        }
        Info { lsb, lo: f64::from(c), hi: f64::from(c), exact: true }
    }

    fn join(a: Info, b: Info) -> Info {
        Info { lsb: a.lsb.min(b.lsb), lo: a.lo.min(b.lo), hi: a.hi.max(b.hi), exact: a.exact && b.exact }
    }

    /// The values of `n` of them summed.
    fn summed(self, n: usize) -> Info {
        let n = n as f64;
        Info { lo: (self.lo * n).min(self.lo), hi: (self.hi * n).max(self.hi), ..self }
    }
}

fn limit(ty: Ty) -> i32 {
    match ty {
        Ty::F32 => 22,
        Ty::Df32 => 44,
        Ty::U32 => i32::MAX,
    }
}

/// Floats of `ty` bounded and exact enough, or `None` if the domain does not allow the value.
fn finish(ty: Ty, math: bool, mut info: Info) -> Option<Info> {
    if ty == Ty::U32 {
        return Some(Info::int());
    }
    if !(info.lo.is_finite() && info.hi.is_finite()) {
        return None;
    }
    let bound = if math { 1e4 } else { 268_435_456.0 };
    if info.mag() > bound {
        return None;
    }
    if info.bits() > limit(ty) {
        info.exact = false;
    }
    if !math && !info.exact {
        return None;
    }
    Some(info)
}

fn corners(a: Info, b: Info, f: impl Fn(f64, f64) -> f64) -> (f64, f64) {
    let c = [f(a.lo, b.lo), f(a.lo, b.hi), f(a.hi, b.lo), f(a.hi, b.hi)];
    (c.iter().copied().fold(f64::INFINITY, f64::min), c.iter().copied().fold(f64::NEG_INFINITY, f64::max))
}

/// The bounds of `a op b` for `ty` (before `finish`); `b_const` is the right operand if it is one.
fn bin_info(ty: Ty, math: bool, op: BinOp, a: Info, b: Info, b_const: Option<f32>) -> Option<Info> {
    if ty == Ty::U32 {
        return Some(Info::int());
    }
    let both = a.exact && b.exact;
    let info = match op {
        BinOp::Add | BinOp::Sub => {
            let (lo, hi) = corners(a, b, |x, y| if op == BinOp::Add { x + y } else { x - y });
            Info { lsb: a.lsb.min(b.lsb), lo, hi, exact: both }
        }
        BinOp::Mul => {
            let (lo, hi) = corners(a, b, |x, y| x * y);
            Info { lsb: a.lsb + b.lsb, lo, hi, exact: both }
        }
        BinOp::Minimum => Info { lsb: a.lsb.min(b.lsb), lo: a.lo.min(b.lo), hi: a.hi.min(b.hi), exact: both },
        BinOp::Maximum => Info { lsb: a.lsb.min(b.lsb), lo: a.lo.max(b.lo), hi: a.hi.max(b.hi), exact: both },
        BinOp::Div => {
            // Dividing by a power of two is exact; in the math domain any divisor away from zero.
            let pow2 = b_const.filter(|c| *c != 0.0 && (f64::from(c.abs()).log2().fract() == 0.0));
            if ty == Ty::Df32 && !math {
                return None;
            }
            if let Some(c) = pow2 {
                let k = f64::from(c.abs()).log2() as i32;
                let (lo, hi) = (a.lo / f64::from(c), a.hi / f64::from(c));
                Info { lsb: a.lsb - k, lo: lo.min(hi), hi: lo.max(hi), exact: a.exact }
            } else if math && (b.lo > 0.25 || b.hi < -0.25) {
                let (lo, hi) = corners(a, b, |x, y| x / y);
                Info { lsb: -60, lo, hi, exact: false }
            } else {
                return None;
            }
        }
        BinOp::Hypot if math => {
            let m = a.mag().hypot(b.mag());
            Info { lsb: -60, lo: 0.0, hi: m, exact: false }
        }
        BinOp::Atan2 if math && b.lo > 0.25 => Info { lsb: -60, lo: -1.58, hi: 1.58, exact: false },
        BinOp::Copysign if math && (b.lo > 0.05 || b.hi < -0.05) => {
            let m = a.mag();
            Info { lsb: -60, lo: -m, hi: m, exact: false }
        }
        BinOp::Rem => return None,
        _ => return None,
    };
    finish(ty, math, info)
}

/// The bounds of `op(a)` for `ty`.
fn un_info(ty: Ty, math: bool, op: UnOp, a: Info) -> Option<Info> {
    if ty == Ty::U32 {
        return None;
    }
    let inexact = |lo: f64, hi: f64| Info { lsb: -60, lo, hi, exact: false };
    let info = match op {
        UnOp::Neg => Info { lo: -a.hi, hi: -a.lo, ..a },
        UnOp::Abs => {
            let lo = if a.lo <= 0.0 && a.hi >= 0.0 { 0.0 } else { a.lo.abs().min(a.hi.abs()) };
            Info { lo, hi: a.mag(), ..a }
        }
        UnOp::Floor | UnOp::Ceil if a.exact => {
            let (lo, hi) = if op == UnOp::Floor { (a.lo.floor(), a.hi.floor()) } else { (a.lo.ceil(), a.hi.ceil()) };
            Info { lsb: a.lsb.max(0), lo, hi, exact: true }
        }
        UnOp::Sign if a.exact => Info { lsb: 0, lo: if a.lo < 0.0 { -1.0 } else { 0.0 }, hi: if a.hi > 0.0 { 1.0 } else { 0.0 }, exact: true },
        UnOp::Clamp { lo, hi } => {
            let (l, h) = (Info::of_const(lo), Info::of_const(hi));
            Info { lsb: a.lsb.min(l.lsb).min(h.lsb), lo: a.lo.max(f64::from(lo)).min(f64::from(hi)), hi: a.hi.min(f64::from(hi)).max(f64::from(lo)), exact: a.exact }
        }
        UnOp::ToDf32 | UnOp::ToF32 => a,
        UnOp::Sqrt if math && a.lo >= 0.0 => inexact(a.lo.sqrt(), a.hi.sqrt()),
        UnOp::Exp if math && ty == Ty::F32 && a.hi <= 11.0 => inexact(a.lo.exp(), a.hi.exp()),
        UnOp::Ln if math && ty == Ty::F32 && a.lo >= 0.05 => inexact(a.lo.ln(), a.hi.ln()),
        UnOp::Sin | UnOp::Cos if math && ty == Ty::F32 && a.mag() <= 100.0 => inexact(-1.0, 1.0),
        UnOp::Tan if math && ty == Ty::F32 && a.mag() <= 1.2 => inexact(a.lo.tan(), a.hi.tan()),
        UnOp::Tanh if math && ty == Ty::F32 => inexact(a.lo.tanh(), a.hi.tanh()),
        UnOp::Powf(e) if math && ty == Ty::F32 && a.lo >= 0.25 => {
            let (x, y) = (a.lo.powf(f64::from(e)), a.hi.powf(f64::from(e)));
            inexact(x.min(y), x.max(y))
        }
        _ => return None,
    };
    finish(ty, math, info)
}

// -- Scope --

#[derive(Clone, Debug)]
struct Var {
    ty: Ty,
    dims: Dims,
    cell: usize,
    /// The cell an assignment through this array writes (set for slices, flips, transposes).
    root: Option<usize>,
}

#[derive(Clone, Debug, Default)]
struct Scope {
    vars: Vec<Option<Var>>,
    cells: Vec<Info>,
    /// Cells below this index belong to an enclosing scope of a loop body: read only there.
    local_from: usize,
}

enum CellRef {
    Fresh(Info),
    Shared(usize),
}

struct New {
    ty: Ty,
    dims: Dims,
    cell: CellRef,
    root: Option<usize>,
}

#[derive(Default)]
struct Effect {
    new: Vec<New>,
    set: Vec<(usize, Info)>,
    kill: Vec<usize>,
}

fn len_of(d: Dims) -> usize {
    d.iter().product()
}

impl Scope {
    fn var(&self, i: usize) -> Option<&Var> {
        self.vars.get(i)?.as_ref()
    }

    fn info(&self, v: &Var) -> Info {
        self.cells[v.cell]
    }

    fn commit(&mut self, e: Effect) {
        for (cell, info) in e.set {
            self.cells[cell] = info;
        }
        for new in e.new {
            let cell = match new.cell {
                CellRef::Fresh(info) => {
                    self.cells.push(info);
                    self.cells.len() - 1
                }
                CellRef::Shared(c) => c,
            };
            self.vars.push(Some(Var { ty: new.ty, dims: new.dims, cell, root: new.root }));
        }
        for k in e.kill {
            self.vars[k] = None;
        }
    }

    fn fresh(ty: Ty, dims: Dims, info: Info) -> New {
        New { ty, dims, cell: CellRef::Fresh(info), root: None }
    }

    /// The info and typing of an operand of type `ty`.
    fn opd(&self, o: &Opd, ty: Ty) -> Option<(Option<Dims>, Info, Option<f32>)> {
        match o {
            Opd::Var(i) => {
                let v = self.var(*i)?;
                (v.ty == ty).then(|| (Some(v.dims), self.info(v), None))
            }
            Opd::Const(c) => Some((None, if ty == Ty::U32 { Info::int() } else { Info::of_const(*c) }, Some(*c))),
        }
    }

    /// Whether writes into cell `c` (and through view roots) are allowed in this scope.
    fn writable(&self, v: &Var) -> bool {
        v.cell >= self.local_from && v.root.is_none_or(|r| r >= self.local_from)
    }

    /// The effect of `stmt`, or `None` if it is not valid here (in the value domain `math`).
    fn effect(&self, stmt: &Stmt, math: bool) -> Option<Effect> {
        let mut e = Effect::default();
        match stmt {
            Stmt::Input { dims, ty, seed } => {
                let info = match ty {
                    Ty::U32 => Info::int(),
                    _ => {
                        let vals = values(*seed, *ty, len_of(*dims), math);
                        let lo = vals.iter().copied().fold(f32::INFINITY, f32::min);
                        let hi = vals.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                        Info { lsb: if math { -23 } else { -3 }, lo: f64::from(lo), hi: f64::from(hi), exact: !math }
                    }
                };
                e.new.push(Scope::fresh(*ty, *dims, info));
            }
            Stmt::Full { dims, ty, value } => {
                let info = if *ty == Ty::U32 { Info::int() } else { Info::of_const(*value) };
                e.new.push(Scope::fresh(*ty, *dims, info));
            }
            Stmt::Range { dims, axis, ty } => {
                let hi = dims[*axis] as f64 - 1.0;
                let info = if *ty == Ty::U32 { Info::int() } else { Info { lsb: 0, lo: 0.0, hi, exact: true } };
                e.new.push(Scope::fresh(*ty, *dims, info));
            }
            Stmt::Linspace { start, end, n } => {
                let (s, t) = (f64::from(*start), f64::from(*end));
                // Exact when the step `(end - start) / (n - 1)` is a dyadic number.
                let info = if math {
                    Info { lsb: -23, lo: s.min(t), hi: s.max(t), exact: false }
                } else {
                    if !(*n - 1).is_power_of_two() {
                        return None;
                    }
                    Info { lsb: -3 - (*n - 1).ilog2() as i32, lo: s.min(t), hi: s.max(t), exact: true }
                };
                e.new.push(Scope::fresh(Ty::F32, [*n, 1, 1, 1], finish(Ty::F32, math, info)?));
            }
            Stmt::Unary { src, op } => {
                let v = self.var(*src)?;
                let ty = match op {
                    UnOp::ToDf32 if v.ty == Ty::F32 => Ty::Df32,
                    UnOp::ToF32 if v.ty == Ty::Df32 => Ty::F32,
                    UnOp::ToDf32 | UnOp::ToF32 => return None,
                    _ => v.ty,
                };
                if v.ty == Ty::Df32 && !matches!(op, UnOp::Neg | UnOp::Abs | UnOp::Sqrt | UnOp::ToF32) {
                    return None;
                }
                let info = un_info(v.ty, math, *op, self.info(v))?;
                // A cast to `f32` rounds: exact only if the value fits.
                let info = if ty == v.ty { info } else { finish(ty, math, info)? };
                e.new.push(Scope::fresh(ty, v.dims, info));
            }
            Stmt::Binary { lhs, rhs, op } => {
                let ty = match (lhs, rhs) {
                    (Opd::Var(i), _) | (_, Opd::Var(i)) => self.var(*i)?.ty,
                    _ => return None,
                };
                let (ld, li, _) = self.opd(lhs, ty)?;
                let (rd, ri, rc) = self.opd(rhs, ty)?;
                if matches!(lhs, Opd::Const(_)) && !matches!(op, BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div) {
                    return None;
                }
                if matches!(op, BinOp::Rem | BinOp::Div) && ty == Ty::U32 && !matches!(rc, Some(c) if c >= 1.0) {
                    return None;
                }
                if (ty != Ty::F32 && matches!(op, BinOp::Atan2 | BinOp::Hypot | BinOp::Copysign)) || (ty != Ty::U32 && *op == BinOp::Rem) {
                    return None;
                }
                let dims = bcast(ld, rd)?;
                let info = bin_info(ty, math, *op, li, ri, rc)?;
                e.new.push(Scope::fresh(ty, dims, info));
            }
            Stmt::Compare { lhs, rhs, op: _ } => {
                let v = self.var(*lhs)?;
                let (rd, ri, _) = self.opd(rhs, v.ty)?;
                // Comparisons of rounded values would differ between the GPU and the oracle.
                if v.ty != Ty::U32 && !(self.info(v).exact && ri.exact) {
                    return None;
                }
                let dims = bcast(Some(v.dims), rd)?;
                e.new.push(Scope::fresh(Ty::U32, dims, Info::int()));
            }
            Stmt::Select { mask, a, b, ty } => {
                let m = self.var(*mask)?;
                if m.ty != Ty::U32 {
                    return None;
                }
                let (ad, ai, _) = self.opd(a, *ty)?;
                let (bd, bi, _) = self.opd(b, *ty)?;
                for d in [ad, bd].into_iter().flatten() {
                    if d != m.dims && len_of(d) != 1 {
                        return None;
                    }
                }
                let info = finish(*ty, math, Info::join(ai, bi))?;
                e.new.push(Scope::fresh(*ty, m.dims, info));
            }
            Stmt::View { src, view } => {
                let v = self.var(*src)?;
                let info = self.info(v);
                let root = v.root.unwrap_or(v.cell);
                match view {
                    ViewOp::Slice { axis, lo, hi } => {
                        if *axis >= 4 || lo >= hi || *hi > v.dims[*axis] {
                            return None;
                        }
                        let mut dims = v.dims;
                        dims[*axis] = hi - lo;
                        e.new.push(New { root: Some(root), ..Scope::fresh(v.ty, dims, info) });
                    }
                    ViewOp::Flip { axis } => {
                        if *axis >= 4 {
                            return None;
                        }
                        e.new.push(New { root: Some(root), ..Scope::fresh(v.ty, v.dims, info) });
                    }
                    ViewOp::Transpose => {
                        let d = v.dims;
                        e.new.push(New { root: Some(root), ..Scope::fresh(v.ty, [d[1], d[0], d[2], d[3]], info) });
                    }
                    ViewOp::Pad { widths, mode } => {
                        let mut dims = v.dims;
                        for (k, (b, a)) in widths.iter().enumerate() {
                            dims[k] += b + a;
                        }
                        if len_of(dims) > 4096 {
                            return None;
                        }
                        let info = if *mode == PadMode::Zero { Info::join(info, Info::of_const(0.0)) } else { info };
                        e.new.push(Scope::fresh(v.ty, dims, finish(v.ty, math, info)?));
                    }
                    ViewOp::ShiftCirc { .. } | ViewOp::ShiftClamp { .. } | ViewOp::ShiftMirror { .. } => {
                        e.new.push(Scope::fresh(v.ty, v.dims, info));
                    }
                    ViewOp::ShiftFill { value, .. } => {
                        let info = if v.ty == Ty::U32 { info } else { Info::join(info, Info::of_const(*value)) };
                        e.new.push(Scope::fresh(v.ty, v.dims, finish(v.ty, math, info)?));
                    }
                    ViewOp::Take { idx } => {
                        let i = self.var(*idx)?;
                        if i.ty != Ty::U32 {
                            return None;
                        }
                        e.new.push(Scope::fresh(v.ty, i.dims, info));
                    }
                    ViewOp::Concat { axis, others } => {
                        let mut dims = v.dims;
                        let mut joined = info;
                        for o in others {
                            let p = self.var(*o)?;
                            if p.ty != v.ty || *axis >= 4 || (0..4).any(|a| a != *axis && p.dims[a] != dims[a]) {
                                return None;
                            }
                            dims[*axis] += p.dims[*axis];
                            joined = Info::join(joined, self.info(p));
                        }
                        if len_of(dims) > 4096 {
                            return None;
                        }
                        e.new.push(Scope::fresh(v.ty, dims, finish(v.ty, math, joined)?));
                    }
                }
            }
            Stmt::Reduce { src, op } => {
                let v = self.var(*src)?;
                let info = self.info(v);
                let info = if *op == RedOp::Sum { info.summed(len_of(v.dims)) } else { info };
                e.new.push(Scope::fresh(v.ty, [1, 1, 1, 1], finish(v.ty, math, info)?));
            }
            Stmt::SumWhere { src, mask } => {
                let v = self.var(*src)?;
                let m = self.var(*mask)?;
                if m.ty != Ty::U32 || m.dims != v.dims {
                    return None;
                }
                let info = Info::join(self.info(v), Info::of_const(0.0)).summed(len_of(v.dims));
                e.new.push(Scope::fresh(v.ty, [1, 1, 1, 1], finish(v.ty, math, info)?));
            }
            Stmt::Scan { src, axis, op } => {
                let v = self.var(*src)?;
                if *axis >= 4 {
                    return None;
                }
                let info = self.info(v);
                let info = if *op == ScanOp::CumSum { info.summed(v.dims[*axis]) } else { info };
                e.new.push(Scope::fresh(v.ty, v.dims, finish(v.ty, math, info)?));
            }
            Stmt::AxisReduce { src, axis, op } => {
                let v = self.var(*src)?;
                if *axis >= 4 {
                    return None;
                }
                let info = self.info(v);
                let info = if *op == RedOp::Sum { info.summed(v.dims[*axis]) } else { info };
                let mut dims = v.dims;
                dims[*axis] = 1;
                e.new.push(Scope::fresh(v.ty, dims, finish(v.ty, math, info)?));
            }
            Stmt::Assign { target, path, rhs } => {
                let t = self.var(*target)?;
                let mut dims = t.dims;
                for step in path {
                    match step {
                        InjView::InjSlice { axis, lo, hi } => {
                            if *axis >= 4 || lo >= hi || *hi > dims[*axis] {
                                return None;
                            }
                            dims[*axis] = hi - lo;
                        }
                        InjView::InjFlip { axis } => {
                            if *axis >= 4 {
                                return None;
                            }
                        }
                        InjView::InjTranspose => dims = [dims[1], dims[0], dims[2], dims[3]],
                    }
                }
                let (rd, ri, _) = self.opd(rhs, t.ty)?;
                if rd.is_some_and(|d| d != dims) {
                    return None;
                }
                let viewed = !path.is_empty() || t.root.is_some();
                let root = t.root.unwrap_or(t.cell);
                if viewed {
                    // The root is written; a named receiver also shows the new data.
                    if root < self.local_from || (path.is_empty() && t.cell < self.local_from) {
                        return None;
                    }
                    let joined = finish(t.ty, math, Info::join(self.cells[root], ri))?;
                    e.set.push((root, joined));
                    if path.is_empty() {
                        e.set.push((t.cell, ri));
                    }
                } else {
                    if t.cell < self.local_from {
                        return None;
                    }
                    e.set.push((t.cell, ri));
                }
            }
            Stmt::Put { target, idx, vals } => {
                let t = self.var(*target)?;
                if t.root.is_some() || !self.writable(t) {
                    return None;
                }
                let (vd, vi, _) = self.opd(vals, t.ty)?;
                if vd.is_some_and(|d| len_of(d) != idx.len()) {
                    return None;
                }
                let joined = finish(t.ty, math, Info::join(self.info(t), vi))?;
                e.set.push((t.cell, joined));
            }
            Stmt::Compound { target, op, rhs } => {
                let t = self.var(*target)?;
                if t.root.is_some() || !self.writable(t) {
                    return None;
                }
                let (rd, ri, rc) = self.opd(rhs, t.ty)?;
                if rd.is_some_and(|d| d != t.dims && len_of(d) != 1) {
                    return None;
                }
                if !matches!(op, BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Rem) || (t.ty != Ty::U32 && *op == BinOp::Rem) {
                    return None;
                }
                if matches!(op, BinOp::Rem | BinOp::Div) && t.ty == Ty::U32 && !matches!(rc, Some(c) if c >= 1.0) {
                    return None;
                }
                let info = bin_info(t.ty, math, *op, self.info(t), ri, rc)?;
                e.set.push((t.cell, info));
            }
            Stmt::Write { target, seed } => {
                let t = self.var(*target)?;
                if t.root.is_some() || !self.writable(t) {
                    return None;
                }
                let info = if t.ty == Ty::U32 {
                    Info::int()
                } else {
                    let vals = values(*seed, t.ty, len_of(t.dims), math);
                    let lo = vals.iter().copied().fold(f32::INFINITY, f32::min);
                    let hi = vals.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    finish(t.ty, math, Info { lsb: if math { -23 } else { -3 }, lo: f64::from(lo), hi: f64::from(hi), exact: !math })?
                };
                e.set.push((t.cell, info));
            }
            Stmt::Cache { src } => {
                let v = self.var(*src)?;
                e.new.push(New { ty: v.ty, dims: v.dims, cell: CellRef::Shared(v.cell), root: None });
            }
            Stmt::Alias { src } => {
                let v = self.var(*src)?;
                e.new.push(New { ty: v.ty, dims: v.dims, cell: CellRef::Shared(v.cell), root: v.root });
            }
            Stmt::EvalVars { vars } => {
                for v in vars {
                    self.var(*v)?;
                }
            }
            Stmt::Release { var } => {
                self.var(*var)?;
                e.kill.push(*var);
            }
            Stmt::Fetch { var } => {
                self.var(*var)?;
            }
            Stmt::Loop { .. } => return None,
        }
        Some(e)
    }
}

/// The shape of a binary operation on operands of shapes `a` and `b` (a constant has none).
fn bcast(a: Option<Dims>, b: Option<Dims>) -> Option<Dims> {
    match (a, b) {
        (Some(a), Some(b)) if a == b => Some(a),
        (Some(a), Some(b)) if len_of(b) == 1 => Some(a),
        (Some(a), Some(b)) if len_of(a) == 1 => Some(b),
        (Some(_), Some(_)) => None,
        (Some(d), None) | (None, Some(d)) => Some(d),
        (None, None) => None,
    }
}

// -- The builder --

#[derive(Clone, Copy, PartialEq, Eq)]
enum Ctx {
    Top,
    /// The body of a per-step loop (readbacks and evaluations are fine).
    HostLoop,
    /// A traced body (`iterate_all`, `iterate_until_all`): runs once, so no readbacks.
    Traced,
}

struct Builder {
    tape: Tape,
    scope: Scope,
    math: bool,
}

const SHAPES: [Dims; 6] = [[12, 1, 1, 1], [7, 5, 1, 1], [9, 6, 1, 1], [8, 8, 1, 1], [300, 1, 1, 1], [24, 20, 1, 1]];
const EXACT_CONSTS: [f32; 8] = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 4.0];
const MATH_CONSTS: [f32; 7] = [0.5, 1.0, 1.5, 2.0, 3.0, -1.0, 0.25];
const INT_CONSTS: [f32; 6] = [0.0, 1.0, 2.0, 3.0, 5.0, 7.0];
const POW2_CONSTS: [f32; 5] = [-2.0, -0.5, 0.5, 2.0, 4.0];

impl Builder {
    fn alive(&self) -> impl Iterator<Item = (usize, &Var)> {
        self.scope.vars.iter().enumerate().filter_map(|(i, v)| v.as_ref().map(|v| (i, v)))
    }

    /// A random alive variable satisfying `ok`, preferring recent ones.
    fn pick(&mut self, ok: impl Fn(&Var) -> bool) -> Option<usize> {
        let all: Vec<usize> = self.alive().filter(|(_, v)| ok(v)).map(|(i, _)| i).collect();
        if all.is_empty() {
            return None;
        }
        let recent = self.tape.chance(2, 3);
        let pool = if recent && all.len() > 6 { &all[all.len() - 6..] } else { &all[..] };
        Some(pool[self.tape.below(pool.len())])
    }

    fn pick_ty(&mut self, ty: Ty) -> Option<usize> {
        self.pick(|v| v.ty == ty)
    }

    fn any_ty(&mut self) -> Ty {
        // Mostly f32; U32 for indices and masks; some double-floats.
        match self.tape.below(20) {
            0..=1 => Ty::Df32,
            2..=6 => Ty::U32,
            _ => Ty::F32,
        }
    }

    fn constant(&mut self, ty: Ty) -> f32 {
        match ty {
            Ty::U32 => self.tape.pick(&INT_CONSTS),
            _ if self.math => self.tape.pick(&MATH_CONSTS),
            _ => self.tape.pick(&EXACT_CONSTS),
        }
    }

    /// A value of the domain from a random word.
    fn float_from(math: bool, word: u32) -> f32 {
        if math {
            0.5 + (word % 1500) as f32 / 1000.0
        } else {
            ((word % 65) as f32 - 32.0) / 8.0
        }
    }

    fn float_val(&mut self) -> f32 {
        let word = self.tape.next();
        Self::float_from(self.math, word)
    }

    /// An operand for an operation on `ty` with `like` the shape it must fit: a variable of the
    /// type with that shape (or one element), or a constant.
    fn operand(&mut self, ty: Ty, like: Option<Dims>, const_chance: (usize, usize)) -> Opd {
        if !self.tape.chance(const_chance.0, const_chance.1) {
            let pick = self.pick(|v| v.ty == ty && like.is_none_or(|d| v.dims == d || len_of(v.dims) == 1));
            if let Some(i) = pick {
                return Opd::Var(i);
            }
        }
        Opd::Const(self.constant(ty))
    }

    fn try_stmt(&mut self, stmt: Stmt) -> Option<Stmt> {
        let effect = self.scope.effect(&stmt, self.math)?;
        self.scope.commit(effect);
        Some(stmt)
    }

    fn offsets(&mut self, dims: Dims, ndim: usize) -> Vec<isize> {
        (0..ndim)
            .map(|a| {
                let n = dims[a] as isize;
                self.tape.below(2 * n.min(4) as usize + 1) as isize - n.min(4)
            })
            .collect()
    }

    fn ndim(dims: Dims) -> usize {
        if dims[1] > 1 { 2 } else { 1 }
    }

    // -- One statement --

    fn stmt(&mut self, ctx: Ctx) -> Option<Stmt> {
        let kind = self.tape.below(100);
        let loops = ctx == Ctx::Top;
        let host = ctx != Ctx::Traced;
        let stmt = match kind {
            0..=4 if ctx != Ctx::Traced => self.g_input(),
            5..=6 => self.g_fill(),
            7..=10 => self.g_range(),
            11..=12 => self.g_linspace(),
            13..=24 => self.g_unary(),
            25..=40 => self.g_binary(),
            41..=44 => self.g_compare(),
            45..=49 => self.g_select(),
            50..=65 => self.g_view(),
            66..=70 => self.g_reduce(),
            71..=72 => self.g_sum_where(),
            73..=75 => self.g_scan(),
            76..=77 => self.g_axis_reduce(),
            78..=85 => self.g_assign(),
            86..=87 => self.g_put(),
            88..=90 => self.g_compound(),
            91 if ctx != Ctx::Traced => self.g_write(),
            92 => self.g_cache(),
            93..=94 if host => self.g_eval(),
            95 => self.g_alias(),
            96 if loops => self.g_release(),
            97..=99 if host => self.g_fetch(),
            _ => None,
        }?;
        self.try_stmt(stmt)
    }

    fn g_input(&mut self) -> Option<Stmt> {
        let dims = self.tape.pick(&SHAPES);
        let ty = self.any_ty();
        let seed = u64::from(self.tape.next()) << 32 | u64::from(self.tape.next());
        Some(Stmt::Input { dims, ty, seed })
    }

    fn g_fill(&mut self) -> Option<Stmt> {
        let dims = self.tape.pick(&SHAPES);
        let ty = self.any_ty();
        let value = self.constant(ty);
        Some(Stmt::Full { dims, ty, value })
    }

    fn g_range(&mut self) -> Option<Stmt> {
        let dims = self.tape.pick(&SHAPES);
        let axis = self.tape.below(Self::ndim(dims));
        let ty = if self.tape.chance(1, 2) { Ty::U32 } else { Ty::F32 };
        Some(Stmt::Range { dims, axis, ty })
    }

    fn g_linspace(&mut self) -> Option<Stmt> {
        let n = if self.math { 2 + self.tape.below(40) } else { self.tape.pick(&[2usize, 3, 5, 9, 17]) };
        let (start, end) = (self.float_val(), self.float_val());
        Some(Stmt::Linspace { start, end, n })
    }

    fn g_unary(&mut self) -> Option<Stmt> {
        let src = self.pick(|v| v.ty != Ty::U32)?;
        let ty = self.scope.var(src)?.ty;
        let op = if ty == Ty::Df32 {
            match self.tape.below(5) {
                0 => UnOp::Neg,
                1 => UnOp::Abs,
                2 if self.math => UnOp::Sqrt,
                3 | 4 => UnOp::ToF32,
                _ => UnOp::Neg,
            }
        } else if self.math {
            match self.tape.below(16) {
                0 => UnOp::Neg,
                1 => UnOp::Abs,
                2 => UnOp::Sqrt,
                3 => UnOp::Exp,
                4 => UnOp::Ln,
                5 => UnOp::Sin,
                6 => UnOp::Cos,
                7 => UnOp::Tan,
                8 => UnOp::Tanh,
                9 => UnOp::Powf(self.tape.pick(&[-1.5f32, -1.0, -0.5, 0.5, 1.5, 2.0, 3.0])),
                10 => UnOp::Clamp { lo: -0.5, hi: 1.5 },
                11 => UnOp::Floor,
                12 => UnOp::Ceil,
                13 => UnOp::Sign,
                _ => UnOp::ToDf32,
            }
        } else {
            match self.tape.below(10) {
                0 | 1 => UnOp::Neg,
                2 | 3 => UnOp::Abs,
                4 => UnOp::Floor,
                5 => UnOp::Ceil,
                6 => UnOp::Sign,
                7 => {
                    let lo = self.tape.pick(&[-2.0f32, -1.0, -0.5, 0.0]);
                    UnOp::Clamp { lo, hi: self.tape.pick(&[0.5f32, 1.0, 2.0, 4.0]) }
                }
                _ => UnOp::ToDf32,
            }
        };
        Some(Stmt::Unary { src, op })
    }

    fn g_binary(&mut self) -> Option<Stmt> {
        let ty = self.any_ty();
        let ops: &[BinOp] = match (ty, self.math) {
            (Ty::U32, _) => &[BinOp::Add, BinOp::Sub, BinOp::Mul, BinOp::Div, BinOp::Rem, BinOp::Minimum, BinOp::Maximum],
            (Ty::Df32, false) => &[BinOp::Add, BinOp::Sub, BinOp::Mul, BinOp::Minimum, BinOp::Maximum],
            (Ty::Df32, true) => &[BinOp::Add, BinOp::Sub, BinOp::Mul, BinOp::Div, BinOp::Minimum, BinOp::Maximum],
            (Ty::F32, false) => &[BinOp::Add, BinOp::Sub, BinOp::Mul, BinOp::Minimum, BinOp::Maximum, BinOp::Div],
            (Ty::F32, true) => &[
                BinOp::Add,
                BinOp::Sub,
                BinOp::Mul,
                BinOp::Div,
                BinOp::Minimum,
                BinOp::Maximum,
                BinOp::Hypot,
                BinOp::Atan2,
                BinOp::Copysign,
            ],
        };
        let op = self.tape.pick(ops);
        let lhs_const = matches!(op, BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div) && self.tape.chance(1, 8);
        let lhs = if lhs_const {
            Opd::Const(self.constant(ty))
        } else {
            Opd::Var(self.pick_ty(ty)?)
        };
        let like = match &lhs {
            Opd::Var(i) => self.scope.var(*i).map(|v| v.dims),
            Opd::Const(_) => None,
        };
        let rhs = if lhs_const {
            Opd::Var(self.pick_ty(ty)?)
        } else if (op == BinOp::Div && ty == Ty::U32) || (op == BinOp::Rem) {
            Opd::Const(self.tape.pick(&[1.0f32, 2.0, 3.0, 7.0]))
        } else if op == BinOp::Div && !self.math {
            Opd::Const(self.tape.pick(&POW2_CONSTS))
        } else {
            self.operand(ty, like, (1, 4))
        };
        Some(Stmt::Binary { lhs, rhs, op })
    }

    fn g_compare(&mut self) -> Option<Stmt> {
        let ty = self.any_ty();
        let lhs = self.pick_ty(ty)?;
        let like = self.scope.var(lhs).map(|v| v.dims);
        let rhs = self.operand(ty, like, (1, 3));
        let op = self.tape.pick(&[CmpOp::Lt, CmpOp::Le, CmpOp::Gt, CmpOp::Ge, CmpOp::Equal, CmpOp::NotEqual]);
        Some(Stmt::Compare { lhs, rhs, op })
    }

    fn g_select(&mut self) -> Option<Stmt> {
        let mask = self.pick_ty(Ty::U32)?;
        let dims = self.scope.var(mask)?.dims;
        let ty = self.any_ty();
        let a = self.operand(ty, Some(dims), (1, 4));
        let b = self.operand(ty, Some(dims), (1, 3));
        Some(Stmt::Select { mask, a, b, ty })
    }

    fn g_view(&mut self) -> Option<Stmt> {
        let src = self.pick(|_| true)?;
        let v = self.scope.var(src)?.clone();
        let nd = Self::ndim(v.dims);
        let view = match self.tape.below(14) {
            0..=2 => {
                let axis = self.tape.below(nd);
                let n = v.dims[axis];
                let lo = self.tape.below(n);
                let hi = lo + 1 + self.tape.below(n - lo);
                ViewOp::Slice { axis, lo, hi }
            }
            3..=4 => {
                let widths = (0..nd)
                    .map(|a| (self.tape.below(v.dims[a].min(3) + 1), self.tape.below(v.dims[a].min(3) + 1)))
                    .collect();
                let mode = self.tape.pick(&[PadMode::Edge, PadMode::Mirror, PadMode::Wrap, PadMode::Zero]);
                ViewOp::Pad { widths, mode }
            }
            5 => ViewOp::Flip { axis: self.tape.below(nd) },
            6 => ViewOp::Transpose,
            7 => ViewOp::ShiftCirc { offs: self.offsets(v.dims, nd) },
            8 => {
                let value = if v.ty == Ty::U32 { self.tape.below(5) as f32 } else { self.constant(v.ty) };
                ViewOp::ShiftFill { offs: self.offsets(v.dims, nd), value }
            }
            9 => ViewOp::ShiftClamp { offs: self.offsets(v.dims, nd) },
            10 => ViewOp::ShiftMirror { offs: self.offsets(v.dims, nd) },
            11 | 12 => ViewOp::Take { idx: self.pick_ty(Ty::U32)? },
            _ => {
                let axis = self.tape.below(nd);
                let dims = v.dims;
                let count = 1 + self.tape.below(2);
                let others: Vec<usize> = (0..count)
                    .filter_map(|_| self.pick(|p| p.ty == v.ty && (0..4).all(|a| a == axis || p.dims[a] == dims[a])))
                    .collect();
                if others.is_empty() {
                    return None;
                }
                ViewOp::Concat { axis, others }
            }
        };
        Some(Stmt::View { src, view })
    }

    fn g_reduce(&mut self) -> Option<Stmt> {
        let src = self.pick(|_| true)?;
        let op = self.tape.pick(&[RedOp::Sum, RedOp::Sum, RedOp::Min, RedOp::Max]);
        Some(Stmt::Reduce { src, op })
    }

    fn g_sum_where(&mut self) -> Option<Stmt> {
        let mask = self.pick_ty(Ty::U32)?;
        let dims = self.scope.var(mask)?.dims;
        let src = self.pick(|v| v.dims == dims)?;
        Some(Stmt::SumWhere { src, mask })
    }

    fn g_scan(&mut self) -> Option<Stmt> {
        let src = self.pick(|_| true)?;
        let axis = self.tape.below(Self::ndim(self.scope.var(src)?.dims));
        let op = self.tape.pick(&[ScanOp::CumSum, ScanOp::CumSum, ScanOp::ScanMax, ScanOp::ScanMin]);
        Some(Stmt::Scan { src, axis, op })
    }

    fn g_axis_reduce(&mut self) -> Option<Stmt> {
        let src = self.pick(|_| true)?;
        let axis = self.tape.below(Self::ndim(self.scope.var(src)?.dims));
        let op = self.tape.pick(&[RedOp::Sum, RedOp::Min, RedOp::Max]);
        Some(Stmt::AxisReduce { src, axis, op })
    }

    fn g_assign(&mut self) -> Option<Stmt> {
        let local_from = self.scope.local_from;
        let target = self.pick(|v| v.cell >= local_from)?;
        let t = self.scope.var(target)?.clone();
        let mut dims = t.dims;
        let mut path = Vec::new();
        for _ in 0..self.tape.below(3) {
            let nd = Self::ndim(dims);
            match self.tape.below(4) {
                0 | 1 => {
                    let axis = self.tape.below(nd);
                    let lo = self.tape.below(dims[axis]);
                    let hi = lo + 1 + self.tape.below(dims[axis] - lo);
                    dims[axis] = hi - lo;
                    path.push(InjView::InjSlice { axis, lo, hi });
                }
                2 => path.push(InjView::InjFlip { axis: self.tape.below(nd) }),
                _ => {
                    dims = [dims[1], dims[0], dims[2], dims[3]];
                    path.push(InjView::InjTranspose);
                }
            }
        }
        let rhs = if self.tape.chance(1, 4) {
            Opd::Const(self.constant(t.ty))
        } else {
            match self.pick(|v| v.ty == t.ty && v.dims == dims) {
                Some(i) => Opd::Var(i),
                None => Opd::Const(self.constant(t.ty)),
            }
        };
        Some(Stmt::Assign { target, path, rhs })
    }

    fn g_put(&mut self) -> Option<Stmt> {
        let local_from = self.scope.local_from;
        let target = self.pick(|v| v.root.is_none() && v.cell >= local_from)?;
        let t = self.scope.var(target)?.clone();
        let len = len_of(t.dims);
        let vals = if self.tape.chance(1, 3) {
            Opd::Const(self.constant(t.ty))
        } else {
            Opd::Var(self.pick(|v| v.ty == t.ty && len_of(v.dims) <= len + 4)?)
        };
        let n = match &vals {
            Opd::Var(i) => len_of(self.scope.var(*i)?.dims),
            Opd::Const(_) => 1 + self.tape.below(len.min(8)),
        };
        // Distinct indices, some past the end (skipped).
        let mut pool: Vec<u32> = (0..(len + 3) as u32).collect();
        let mut idx = Vec::with_capacity(n);
        for _ in 0..n {
            if pool.is_empty() {
                return None;
            }
            let k = self.tape.below(pool.len());
            idx.push(pool.swap_remove(k));
        }
        Some(Stmt::Put { target, idx, vals })
    }

    fn g_compound(&mut self) -> Option<Stmt> {
        let local_from = self.scope.local_from;
        let target = self.pick(|v| v.root.is_none() && v.cell >= local_from)?;
        let t = self.scope.var(target)?.clone();
        let op = match t.ty {
            Ty::U32 => self.tape.pick(&[BinOp::Add, BinOp::Sub, BinOp::Mul, BinOp::Div, BinOp::Rem]),
            _ => self.tape.pick(&[BinOp::Add, BinOp::Sub, BinOp::Mul, BinOp::Div]),
        };
        let rhs = if (t.ty == Ty::U32 && matches!(op, BinOp::Div | BinOp::Rem)) || (op == BinOp::Div && !self.math) {
            let pool: &[f32] = if t.ty == Ty::U32 { &[1.0, 2.0, 3.0, 7.0] } else { &POW2_CONSTS };
            Opd::Const(self.tape.pick(pool))
        } else {
            // A one-element target is never broadcast to a larger right-hand side.
            let like = Some(t.dims);
            self.operand(t.ty, like, (1, 3))
        };
        Some(Stmt::Compound { target, op, rhs })
    }

    fn g_write(&mut self) -> Option<Stmt> {
        let local_from = self.scope.local_from;
        let target = self.pick(|v| v.root.is_none() && v.cell >= local_from)?;
        let _t = self.scope.var(target)?.clone();
        let seed = u64::from(self.tape.next()) << 32 | u64::from(self.tape.next());
        Some(Stmt::Write { target, seed })
    }

    fn g_cache(&mut self) -> Option<Stmt> {
        Some(Stmt::Cache { src: self.pick(|_| true)? })
    }

    fn g_eval(&mut self) -> Option<Stmt> {
        let n = 1 + self.tape.below(3);
        let mut vars = Vec::new();
        for _ in 0..n {
            let v = self.pick(|_| true)?;
            if !vars.contains(&v) {
                vars.push(v);
            }
        }
        Some(Stmt::EvalVars { vars })
    }

    fn g_alias(&mut self) -> Option<Stmt> {
        Some(Stmt::Alias { src: self.pick(|_| true)? })
    }

    fn g_release(&mut self) -> Option<Stmt> {
        Some(Stmt::Release { var: self.pick(|_| true)? })
    }

    fn g_fetch(&mut self) -> Option<Stmt> {
        Some(Stmt::Fetch { var: self.pick(|_| true)? })
    }

    /// A statement that fails on both sides, and creates nothing.
    fn invalid(&mut self) -> Option<Stmt> {
        match self.tape.below(6) {
            0 => {
                let ty = self.any_ty();
                let a = self.pick_ty(ty)?;
                let da = self.scope.var(a)?.dims;
                let b = self.pick(|v| v.ty == ty && v.dims != da && len_of(v.dims) != 1 && len_of(da) != 1)?;
                Some(Stmt::Binary { lhs: Opd::Var(a), rhs: Opd::Var(b), op: BinOp::Add })
            }
            1 => {
                let src = self.pick(|_| true)?;
                let dims = self.scope.var(src)?.dims;
                let axis = self.tape.below(Self::ndim(dims));
                let hi = dims[axis] + 1 + self.tape.below(3);
                Some(Stmt::View { src, view: ViewOp::Slice { axis, lo: self.tape.below(2), hi } })
            }
            2 => {
                let local_from = self.scope.local_from;
                let target = self.pick(|v| v.cell >= local_from)?;
                let t = self.scope.var(target)?.clone();
                let rhs = self.pick(|v| v.ty == t.ty && v.dims != t.dims)?;
                Some(Stmt::Assign { target, path: vec![], rhs: Opd::Var(rhs) })
            }
            3 => {
                let target = self.pick(|v| v.root.is_some())?;
                let ty = self.scope.var(target)?.ty;
                Some(Stmt::Put { target, idx: vec![0], vals: Opd::Const(self.constant(ty)) })
            }
            4 => {
                let mask = self.pick_ty(Ty::U32)?;
                let dims = self.scope.var(mask)?.dims;
                let ty = self.any_ty();
                let b = self.pick(|v| v.ty == ty && v.dims != dims && len_of(v.dims) != 1)?;
                Some(Stmt::Select { mask, a: Opd::Var(b), b: Opd::Const(self.constant(ty)), ty })
            }
            _ => {
                let src = self.pick(|_| true)?;
                let v = self.scope.var(src)?.clone();
                let axis = self.tape.below(Self::ndim(v.dims));
                // Another axis differs.
                let other = self.pick(|p| p.ty == v.ty && (0..4).any(|a| a != axis && p.dims[a] != v.dims[a]))?;
                Some(Stmt::View { src, view: ViewOp::Concat { axis, others: vec![other] } })
            }
        }
    }

    // -- Blocks and loops --

    fn block(&mut self, count: usize, ctx: Ctx) -> Vec<Stmt> {
        let mut out = Vec::new();
        let mut attempts = 0;
        while out.len() < count && attempts < count * 6 && !self.tape.done() {
            attempts += 1;
            // Start with a few inputs.
            let stmt = if ctx == Ctx::Top && self.scope.vars.len() < 3 {
                self.g_input().and_then(|s| self.try_stmt(s))
            } else if ctx != Ctx::Traced && self.tape.chance(1, 40) {
                // Validity is not tracked: nothing is created, nothing changes.
                self.invalid()
            } else if ctx == Ctx::Top && self.tape.chance(1, 8) {
                self.g_loop()
            } else {
                self.stmt(ctx)
            };
            out.extend(stmt);
        }
        out
    }

    fn g_loop(&mut self) -> Option<Stmt> {
        let kind = self.tape.pick(&[LoopKind::PerStepEval, LoopKind::PerStepEval, LoopKind::IterateAll, LoopKind::IterateAll, LoopKind::IterateUntilAll]);
        let first = self.pick(|v| kind != LoopKind::IterateUntilAll || v.ty == Ty::F32)?;
        let (ty, first_cell) = {
            let v = self.scope.var(first)?;
            (v.ty, v.cell)
        };
        let mut states = vec![first];
        if self.tape.chance(1, 3)
            && let Some(second) = self.pick(|v| v.ty == ty && v.cell != first_cell)
        {
            states.push(second);
        }
        let requested = 1 + self.tape.below(12);

        // The body is generated against the state variables appended to the scope; the scope
        // is restored afterwards.
        let outer = self.scope.clone();
        let base = outer.vars.len();
        let n = states.len();
        let mut state_infos = Vec::new();
        for &s in &states {
            let v = outer.var(s)?.clone();
            let info = outer.info(&v);
            state_infos.push(info);
            self.scope.commit(Effect { new: vec![Scope::fresh(v.ty, v.dims, info)], ..Effect::default() });
        }
        self.scope.local_from = self.scope.cells.len();
        let proto = self.scope.clone();
        let ctx = if kind == LoopKind::PerStepEval { Ctx::HostLoop } else { Ctx::Traced };
        let count = 1 + self.tape.below(5);
        let body = self.block(count, ctx);

        // The next states: computed variables of the body (fresh data: not views, not aliases
        // of the states or of arrays outside), one per state, no two sharing data.
        let mut next: Vec<usize> = Vec::new();
        let mut used: Vec<usize> = Vec::new();
        for k in 0..n {
            let want = proto.var(base + k)?.clone();
            let local_from = self.scope.local_from;
            let candidates: Vec<usize> = self
                .alive()
                .filter(|(i, v)| {
                    *i >= base + n && v.ty == want.ty && v.dims == want.dims && v.root.is_none() && v.cell >= local_from && !used.contains(&v.cell)
                })
                .map(|(i, _)| i)
                .collect();
            if candidates.is_empty() {
                self.scope = outer;
                return None;
            }
            let pick = candidates[candidates.len() - 1 - self.tape.below(candidates.len().min(3))];
            used.push(self.scope.var(pick)?.cell);
            next.push(pick);
        }

        // The step count the bounds allow: run the body on the bounds, step by step.
        let mut infos = state_infos;
        let mut joined: Option<Vec<Info>> = None;
        let mut steps = 0;
        'steps: for _ in 0..requested {
            let mut s = proto.clone();
            for (k, info) in infos.iter().enumerate() {
                let cell = s.vars[base + k].as_ref()?.cell;
                s.cells[cell] = *info;
            }
            for stmt in &body {
                match s.effect(stmt, self.math) {
                    Some(e) => s.commit(e),
                    None => break 'steps,
                }
            }
            let mut after = Vec::new();
            for &x in &next {
                match s.var(x) {
                    Some(v) => after.push(s.info(v)),
                    None => break 'steps,
                }
            }
            joined = Some(match joined {
                None => after.clone(),
                Some(j) => j.iter().zip(&after).map(|(a, b)| Info::join(*a, *b)).collect(),
            });
            infos = after;
            steps += 1;
        }
        let Some(finals) = joined.filter(|_| steps > 0) else {
            self.scope = outer;
            return None;
        };

        // Back to the outer scope, with the final states appended (the body wrote nothing
        // outside).
        self.scope = outer;
        for (k, &s) in states.iter().enumerate() {
            let v = self.scope.var(s)?.clone();
            self.scope.commit(Effect { new: vec![Scope::fresh(v.ty, v.dims, finals[k])], ..Effect::default() });
        }
        Some(Stmt::Loop { kind, steps, states, body, next })
    }
}

// -- Minimizing --

impl Opd {
    fn remap(&self, f: &dyn Fn(usize) -> Option<usize>) -> Option<Opd> {
        Some(match self {
            Opd::Var(i) => Opd::Var(f(*i)?),
            Opd::Const(c) => Opd::Const(*c),
        })
    }
}

impl Stmt {
    /// Every variable number the statement mentions (a loop's body included).
    pub fn mentions(&self) -> Vec<usize> {
        let seen = std::cell::RefCell::new(Vec::new());
        let _ = self.remap(&|i| {
            seen.borrow_mut().push(i);
            Some(i)
        });
        seen.into_inner()
    }

    /// Whether the statement only creates variables: it writes no array and reads nothing back.
    pub fn creates_only(&self) -> bool {
        !matches!(
            self,
            Stmt::Assign { .. }
                | Stmt::Put { .. }
                | Stmt::Compound { .. }
                | Stmt::Write { .. }
                | Stmt::EvalVars { .. }
                | Stmt::Release { .. }
                | Stmt::Fetch { .. }
                | Stmt::Loop { .. }
        )
    }

    /// The statement with every variable number `i` replaced by `f(i)`; `None` if `f` has none.
    fn remap(&self, f: &dyn Fn(usize) -> Option<usize>) -> Option<Stmt> {
        let all = |v: &[usize]| v.iter().map(|&i| f(i)).collect::<Option<Vec<_>>>();
        Some(match self {
            Stmt::Input { .. } | Stmt::Full { .. } | Stmt::Range { .. } | Stmt::Linspace { .. } => self.clone(),
            Stmt::Unary { src, op } => Stmt::Unary { src: f(*src)?, op: *op },
            Stmt::Binary { lhs, rhs, op } => Stmt::Binary { lhs: lhs.remap(f)?, rhs: rhs.remap(f)?, op: *op },
            Stmt::Compare { lhs, rhs, op } => Stmt::Compare { lhs: f(*lhs)?, rhs: rhs.remap(f)?, op: *op },
            Stmt::Select { mask, a, b, ty } => Stmt::Select { mask: f(*mask)?, a: a.remap(f)?, b: b.remap(f)?, ty: *ty },
            Stmt::View { src, view } => {
                let view = match view {
                    ViewOp::Take { idx } => ViewOp::Take { idx: f(*idx)? },
                    ViewOp::Concat { axis, others } => ViewOp::Concat { axis: *axis, others: all(others)? },
                    other => other.clone(),
                };
                Stmt::View { src: f(*src)?, view }
            }
            Stmt::Reduce { src, op } => Stmt::Reduce { src: f(*src)?, op: *op },
            Stmt::SumWhere { src, mask } => Stmt::SumWhere { src: f(*src)?, mask: f(*mask)? },
            Stmt::Scan { src, axis, op } => Stmt::Scan { src: f(*src)?, axis: *axis, op: *op },
            Stmt::AxisReduce { src, axis, op } => Stmt::AxisReduce { src: f(*src)?, axis: *axis, op: *op },
            Stmt::Assign { target, path, rhs } => Stmt::Assign { target: f(*target)?, path: path.clone(), rhs: rhs.remap(f)? },
            Stmt::Put { target, idx, vals } => Stmt::Put { target: f(*target)?, idx: idx.clone(), vals: vals.remap(f)? },
            Stmt::Compound { target, op, rhs } => Stmt::Compound { target: f(*target)?, op: *op, rhs: rhs.remap(f)? },
            Stmt::Write { target, seed } => Stmt::Write { target: f(*target)?, seed: *seed },
            Stmt::Cache { src } => Stmt::Cache { src: f(*src)? },
            Stmt::EvalVars { vars } => Stmt::EvalVars { vars: all(vars)? },
            Stmt::Alias { src } => Stmt::Alias { src: f(*src)? },
            Stmt::Release { var } => Stmt::Release { var: f(*var)? },
            Stmt::Fetch { var } => Stmt::Fetch { var: f(*var)? },
            Stmt::Loop { kind, steps, states, body, next } => Stmt::Loop {
                kind: *kind,
                steps: *steps,
                states: all(states)?,
                body: body.iter().map(|s| s.remap(f)).collect::<Option<Vec<_>>>()?,
                next: all(next)?,
            },
        })
    }
}

impl Program {
    /// The program without statement `k`, later variable numbers shifted; `None` if a later
    /// statement reads a variable `k` creates. `created` is how many variables each statement
    /// creates (`created_counts` in the runner).
    pub fn without(&self, k: usize, created: &[usize]) -> Option<Program> {
        let start: usize = created[..k].iter().sum();
        let end = start + created[k];
        let shift = created[k];
        let f = move |i: usize| if i < start { Some(i) } else if i >= end { Some(i - shift) } else { None };
        let mut stmts = Vec::with_capacity(self.stmts.len() - 1);
        for (j, stmt) in self.stmts.iter().enumerate() {
            if j != k {
                stmts.push(if j < k { stmt.clone() } else { stmt.remap(&f)? });
            }
        }
        Some(Program { stmts, ..self.clone() })
    }
}

/// Builds the program a tape stands for.
pub fn build(config: Config, math: bool, words: Vec<u32>) -> Program {
    let mut b = Builder { tape: Tape { words, pos: 0 }, scope: Scope::default(), math };
    let stmts = b.block(48, Ctx::Top);
    let mut config = config;
    config.fast_math &= !math;
    Program { config, math, stmts }
}

// -- The strategy --

fn config_strategy() -> impl Strategy<Value = Config> {
    let limits = (
        prop::sample::select(vec![2usize, 8, 64, 1024]),
        prop::sample::select(vec![Some(16usize), Some(64), None]),
        prop::sample::select(vec![Some(8usize), None]),
        prop::sample::select(vec![Some(4usize), Some(8), None]),
    );
    let flags = (prop::bool::weighted(0.3), prop::bool::weighted(0.7), prop::bool::weighted(0.8), prop::bool::weighted(0.3));
    let memory = (
        prop::sample::select(vec![None, None, Some(1024u64)]),
        prop::sample::select(vec![None, None, Some(1u64 << 20)]),
        prop::sample::select(vec![CostChoice::Test, CostChoice::SlowLaunch, CostChoice::SlowArithmetic]),
    );
    (limits, flags, memory).prop_map(|(limits, flags, memory)| Config {
        max_nodes: limits.0,
        max_inlined: limits.1,
        max_live: limits.2,
        max_inputs: limits.3,
        assume_busy: flags.0,
        speculate: flags.1,
        simplify: flags.2,
        fast_math: flags.3,
        chunk_bytes: memory.0,
        memory_budget: memory.1,
        cost: memory.2,
    })
}

/// Programs under random configurations.
pub fn programs() -> impl Strategy<Value = Program> {
    (config_strategy(), any::<bool>(), prop::collection::vec(any::<u32>(), 40..500)).prop_map(|(config, math, tape)| build(config, math, tape))
}
