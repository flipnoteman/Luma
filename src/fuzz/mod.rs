//! A differential fuzzer: random array programs run on luma, under random engine
//! configurations, and on an eager CPU oracle ([`oracle`]); the results are compared at every
//! `Fetch` statement and for every live variable at the end.
//!
//! * `LUMA_FUZZ_CASES`: number of programs per entry point (default 256).
//! * `LUMA_FUZZ_SEED`: fixed seed of the random number generator (default: random; proptest
//!   prints the seed of a failing run and persists failures in `proptest-regressions/`).
//!
//! A failure prints the shrunk program as a Rust literal; paste it into `regressions.rs`. See
//! [`gen`] for the programs and the value domains, [`oracle`] for the reference semantics.
//!
//! Where luma documents no behaviour, the choices of the generator are:
//! * `+=`, `write` and `put` only target arrays that are not views (what a view does after
//!   such a write is unspecified); `assign` goes through views, which keep their link to the
//!   array they were taken from.
//! * Nothing writes an array that a loop body did not create (iterate steps are traced once
//!   and must compute from their arguments), and no traced step uploads data or reads back.
//! * An operation that fails on luma with `Chunked` or `Unsupported` (documented limits of
//!   arrays split over several buffers, of 8-byte element types) ends the program early.

#[path = "gen.rs"]
mod generator;
mod oracle;
mod regressions;

use std::cell::RefCell;
use std::sync::atomic::{AtomicUsize, Ordering};

use proptest::prelude::*;
use proptest::test_runner::{TestError, TestRunner};
use proptest::test_runner::{FileFailurePersistence, RngSeed};

use crate::builtin::BinOp as LBin;
use crate::ops::{Operand, Rhs};
use crate::{
    concat, eval_all, iterate_all, iterate_until_all, select, Array, Context, ContextOptions, CostModel, Df32, LumaError, ScalarType, Shape,
    Until,
};
use generator::{CostChoice, Config, InjView, LoopKind, Opd, Program, Stmt, ViewOp};
use oracle::{BinOp, CmpOp, Dims, Edge, ErrKind, HostArray, RedOp, Rhs as HRhs, ScanOp, Ty, UnOp};

/// Programs that ended early because luma reported a documented limit.
static ENDED_EARLY: AtomicUsize = AtomicUsize::new(0);

/// Statements skipped because luma refused them for a documented limit.
static SKIPPED: AtomicUsize = AtomicUsize::new(0);

// -- Element types --

/// An element type of the fuzzer, with its conversions.
trait Num: ScalarType + Operand<Self> {
    const TY: Ty;
    fn from_f32(v: f32) -> Self;
    fn to_f64(self) -> f64;
    fn get(a: &Arr) -> Option<&Array<Self>>;
    fn wrap(a: Array<Self>) -> Arr;
}

impl Num for f32 {
    const TY: Ty = Ty::F32;
    fn from_f32(v: f32) -> f32 {
        v
    }
    fn to_f64(self) -> f64 {
        f64::from(self)
    }
    fn get(a: &Arr) -> Option<&Array<f32>> {
        if let Arr::F(x) = a { Some(x) } else { None }
    }
    fn wrap(a: Array<f32>) -> Arr {
        Arr::F(a)
    }
}

impl Num for u32 {
    const TY: Ty = Ty::U32;
    fn from_f32(v: f32) -> u32 {
        v as u32
    }
    fn to_f64(self) -> f64 {
        f64::from(self)
    }
    fn get(a: &Arr) -> Option<&Array<u32>> {
        if let Arr::U(x) = a { Some(x) } else { None }
    }
    fn wrap(a: Array<u32>) -> Arr {
        Arr::U(a)
    }
}

impl Num for Df32 {
    const TY: Ty = Ty::Df32;
    fn from_f32(v: f32) -> Df32 {
        Df32::from(v)
    }
    fn to_f64(self) -> f64 {
        Df32::to_f64(self)
    }
    fn get(a: &Arr) -> Option<&Array<Df32>> {
        if let Arr::D(x) = a { Some(x) } else { None }
    }
    fn wrap(a: Array<Df32>) -> Arr {
        Arr::D(a)
    }
}

/// An array of any of the types.
#[derive(Clone)]
enum Arr {
    F(Array<f32>),
    U(Array<u32>),
    D(Array<Df32>),
}

/// `$body` with `$a` bound to the array inside an [`Arr`].
macro_rules! with_arr {
    ($arr:expr, $a:ident => $body:expr) => {
        match $arr {
            Arr::F($a) => $body,
            Arr::U($a) => $body,
            Arr::D($a) => $body,
        }
    };
}

/// `$body` with the type alias `$t` standing for the element type `$ty` names.
macro_rules! by_ty {
    ($ty:expr, $t:ident => $body:expr) => {
        match $ty {
            Ty::F32 => {
                type $t = f32;
                $body
            }
            Ty::U32 => {
                type $t = u32;
                $body
            }
            Ty::Df32 => {
                type $t = Df32;
                $body
            }
        }
    };
}

impl Arr {
    fn ty(&self) -> Ty {
        with_arr!(self, a => {
            let _ = a;
            match self {
                Arr::F(_) => Ty::F32,
                Arr::U(_) => Ty::U32,
                Arr::D(_) => Ty::Df32,
            }
        })
    }

    fn dims(&self) -> Dims {
        with_arr!(self, a => a.shape().dims())
    }

    fn check(&self) -> Result<(), LumaError> {
        with_arr!(self, a => a.check())
    }

    fn read(&self) -> Result<Vec<f64>, LumaError> {
        with_arr!(self, a => Ok(a.to_vec()?.into_iter().map(Num::to_f64).collect()))
    }
}

/// An operand as the array operations take it.
#[derive(Clone, Copy)]
enum R<'a, T: ScalarType> {
    A(&'a Array<T>),
    V(T),
}

impl<T: ScalarType> Operand<T> for R<'_, T> {
    fn rhs(&self) -> Rhs<'_, T> {
        match self {
            R::A(a) => Rhs::Array(a),
            R::V(v) => Rhs::Scalar(*v),
        }
    }
}

fn kind_of(error: &LumaError) -> ErrKind {
    match error.root() {
        LumaError::ShapeMismatch { .. } => ErrKind::ShapeMismatch,
        LumaError::InvalidShape(_) => ErrKind::InvalidShape,
        LumaError::LengthMismatch { .. } => ErrKind::LengthMismatch,
        LumaError::Chunked(_) => ErrKind::Chunked,
        LumaError::Unsupported(_) => ErrKind::Unsupported,
        _ => ErrKind::Other,
    }
}

// -- The luma side --

type Env = Vec<Option<Arr>>;

/// What statements run with: the context, and the value domain of the program (it selects the
/// values of inputs).
struct Cx {
    ctx: Context,
    math: bool,
}

fn get<T: Num>(env: &Env, i: usize) -> &Array<T> {
    env.get(i).and_then(Option::as_ref).and_then(T::get).unwrap_or_else(|| panic!("variable {i} is not a live {:?}", T::TY))
}

fn ty_of(env: &Env, i: usize) -> Ty {
    env.get(i).and_then(Option::as_ref).unwrap_or_else(|| panic!("variable {i} is not live")).ty()
}

fn take<T: Num>(env: &mut Env, i: usize) -> Array<T> {
    let arr = env.get_mut(i).and_then(Option::take).unwrap_or_else(|| panic!("variable {i} is not live"));
    T::get(&arr).unwrap_or_else(|| panic!("variable {i} is not a {:?}", T::TY)).clone()
}

fn operand<'a, T: Num>(env: &'a Env, o: &Opd) -> R<'a, T> {
    match o {
        Opd::Var(i) => R::A(get::<T>(env, *i)),
        Opd::Const(c) => R::V(T::from_f32(*c)),
    }
}

fn opd_ty(env: &Env, a: &Opd, b: &Opd) -> Ty {
    match (a, b) {
        (Opd::Var(i), _) | (_, Opd::Var(i)) => ty_of(env, *i),
        _ => panic!("operation on two constants"),
    }
}

fn lbin(op: BinOp) -> LBin {
    match op {
        BinOp::Add => LBin::Add,
        BinOp::Sub => LBin::Sub,
        BinOp::Mul => LBin::Mul,
        BinOp::Div => LBin::Div,
        BinOp::Rem => LBin::Rem,
        BinOp::Minimum => LBin::Min,
        BinOp::Maximum => LBin::Max,
        BinOp::Atan2 => LBin::Atan2,
        BinOp::Hypot => LBin::Hypot,
        BinOp::Copysign => LBin::Copysign,
        BinOp::BitAnd => LBin::BitAnd,
        BinOp::BitOr => LBin::BitOr,
        BinOp::BitXor => LBin::BitXor,
        BinOp::Shl => LBin::Shl,
        BinOp::Shr => LBin::Shr,
        BinOp::Rotl => LBin::Rotl,
    }
}

fn push(env: &mut Env, arr: Arr) -> Result<(), LumaError> {
    arr.check()?;
    env.push(Some(arr));
    Ok(())
}

fn unary(a: &Arr, op: UnOp) -> Arr {
    match (a, op) {
        (Arr::F(x), UnOp::Neg) => Arr::F(-x),
        (Arr::F(x), UnOp::Abs) => Arr::F(x.abs()),
        (Arr::F(x), UnOp::Floor) => Arr::F(x.floor()),
        (Arr::F(x), UnOp::Ceil) => Arr::F(x.ceil()),
        (Arr::F(x), UnOp::Sign) => Arr::F(x.sign()),
        (Arr::F(x), UnOp::Sqrt) => Arr::F(x.sqrt()),
        (Arr::F(x), UnOp::Exp) => Arr::F(x.exp()),
        (Arr::F(x), UnOp::Ln) => Arr::F(x.ln()),
        (Arr::F(x), UnOp::Sin) => Arr::F(x.sin()),
        (Arr::F(x), UnOp::Cos) => Arr::F(x.cos()),
        (Arr::F(x), UnOp::Tan) => Arr::F(x.tan()),
        (Arr::F(x), UnOp::Tanh) => Arr::F(x.tanh()),
        (Arr::F(x), UnOp::Clamp { lo, hi }) => Arr::F(x.clamp(lo, hi)),
        (Arr::F(x), UnOp::Powf(e)) => Arr::F(x.powf(e)),
        (Arr::F(x), UnOp::ToDf32) => Arr::D(x.to_df32()),
        (Arr::U(x), UnOp::Not) => Arr::U(!x),
        (Arr::D(x), UnOp::Neg) => Arr::D(-x),
        (Arr::D(x), UnOp::Abs) => Arr::D(x.abs()),
        (Arr::D(x), UnOp::Sqrt) => Arr::D(x.sqrt()),
        (Arr::D(x), UnOp::ToF32) => Arr::F(x.to_f32()),
        _ => panic!("{op:?} is not defined for {:?}", a.ty()),
    }
}

fn binary<T: Num>(env: &Env, lhs: &Opd, rhs: &Opd, op: BinOp) -> Array<T> {
    let op = lbin(op);
    match (lhs, rhs) {
        (Opd::Var(i), r) => get::<T>(env, *i).binary(operand::<T>(env, r).rhs(), op, false, "fuzz"),
        (Opd::Const(c), Opd::Var(j)) => get::<T>(env, *j).binary(Rhs::Scalar(T::from_f32(*c)), op, true, "fuzz"),
        _ => panic!("operation on two constants"),
    }
}

fn compare<T: Num>(env: &Env, lhs: usize, rhs: &Opd, op: CmpOp) -> Array<u32> {
    let (a, r) = (get::<T>(env, lhs), operand::<T>(env, rhs));
    match op {
        CmpOp::Lt => a.lt(r),
        CmpOp::Le => a.le(r),
        CmpOp::Gt => a.gt(r),
        CmpOp::Ge => a.ge(r),
        CmpOp::Equal => a.eq(r),
        CmpOp::NotEqual => a.ne(r),
    }
}

fn view<T: Num>(env: &Env, a: &Array<T>, view: &ViewOp) -> Array<T> {
    match view {
        ViewOp::Slice { axis, lo, hi } => a.slice(*axis, *lo..*hi),
        ViewOp::Pad { widths, mode } => match widths[..] {
            [w0] => a.pad([w0], *mode),
            [w0, w1] => a.pad([w0, w1], *mode),
            _ => panic!("pad of {} axes", widths.len()),
        },
        ViewOp::Flip { axis } => a.flip(*axis),
        ViewOp::Transpose => a.transpose(),
        ViewOp::ShiftCirc { offs } => match offs[..] {
            [o0] => a.shift([o0]),
            [o0, o1] => a.shift([o0, o1]),
            _ => panic!("shift of {} axes", offs.len()),
        },
        ViewOp::ShiftFill { offs, value } => match offs[..] {
            [o0] => a.shift_fill([o0], T::from_f32(*value)),
            [o0, o1] => a.shift_fill([o0, o1], T::from_f32(*value)),
            _ => panic!("shift of {} axes", offs.len()),
        },
        ViewOp::ShiftClamp { offs } => match offs[..] {
            [o0] => a.shift_clamp([o0]),
            [o0, o1] => a.shift_clamp([o0, o1]),
            _ => panic!("shift of {} axes", offs.len()),
        },
        ViewOp::ShiftMirror { offs } => match offs[..] {
            [o0] => a.shift_mirror([o0]),
            [o0, o1] => a.shift_mirror([o0, o1]),
            _ => panic!("shift of {} axes", offs.len()),
        },
        ViewOp::Take { idx } => a.take(get::<u32>(env, *idx)),
        ViewOp::Concat { axis, others } => match others[..] {
            [b] => concat(*axis, [a, get::<T>(env, b)]),
            [b, c] => concat(*axis, [a, get::<T>(env, b), get::<T>(env, c)]),
            _ => panic!("concat of {} parts", others.len() + 1),
        },
    }
}

fn reduce<T: Num>(a: &Array<T>, op: RedOp) -> Array<T> {
    match op {
        RedOp::Sum => a.sum(),
        RedOp::Min => a.min(),
        RedOp::Max => a.max(),
    }
}

fn scan<T: Num>(a: &Array<T>, axis: usize, op: ScanOp) -> Array<T> {
    match op {
        ScanOp::CumSum => a.cumsum(axis),
        ScanOp::ScanMax => a.scan(axis, |x, y| x.maximum(y)),
        ScanOp::ScanMin => a.scan(axis, |x, y| x.minimum(y)),
    }
}

fn axis_reduce<T: Num>(a: &Array<T>, axis: usize, op: RedOp) -> Array<T> {
    match op {
        RedOp::Sum => a.sum_axis(axis),
        RedOp::Min => a.min_axis(axis),
        RedOp::Max => a.max_axis(axis),
    }
}

fn eval_group<T: Num>(env: &Env, vars: &[usize]) -> Result<(), LumaError> {
    let arrays: Vec<&Array<T>> = vars.iter().filter_map(|&i| env.get(i).and_then(Option::as_ref).and_then(T::get)).collect();
    if arrays.is_empty() { Ok(()) } else { eval_all(&arrays) }
}

/// Runs a statement on luma; creations push their variables, a failure creates none.
fn luma_stmt(cx: &Cx, env: &mut Env, stmt: &Stmt) -> Result<(), LumaError> {
    let ctx = &cx.ctx;
    match stmt {
        Stmt::Input { dims, ty, seed } => by_ty!(*ty, T => {
            let data: Vec<T> = generator::values(*seed, *ty, oracle::len_of(*dims), cx.math).into_iter().map(T::from_f32).collect();
            push(env, T::wrap(Array::<T>::from_slice_on(ctx, &data, Shape::new(*dims))?))
        }),
        Stmt::Full { dims, ty, value } => by_ty!(*ty, T => {
            push(env, T::wrap(Array::<T>::full_on(ctx, Shape::new(*dims), T::from_f32(*value))?))
        }),
        Stmt::Range { dims, axis, ty } => by_ty!(*ty, T => {
            push(env, T::wrap(Array::<T>::range_on(ctx, Shape::new(*dims), *axis)?))
        }),
        Stmt::Linspace { start, end, n } => push(env, Arr::F(Array::<f32>::linspace_on(ctx, *start, *end, *n)?)),
        Stmt::Unary { src, op } => {
            let out = unary(env[*src].as_ref().expect("live"), *op);
            push(env, out)
        }
        Stmt::Binary { lhs, rhs, op } => by_ty!(opd_ty(env, lhs, rhs), T => {
            let out = binary::<T>(env, lhs, rhs, *op);
            push(env, T::wrap(out))
        }),
        Stmt::Compare { lhs, rhs, op } => by_ty!(ty_of(env, *lhs), T => {
            let out = compare::<T>(env, *lhs, rhs, *op);
            push(env, Arr::U(out))
        }),
        Stmt::Select { mask, a, b, ty } => by_ty!(*ty, T => {
            let out = select::<T>(get::<u32>(env, *mask), operand::<T>(env, a), operand::<T>(env, b));
            push(env, T::wrap(out))
        }),
        Stmt::View { src, view: v } => by_ty!(ty_of(env, *src), T => {
            let out = view::<T>(env, get::<T>(env, *src), v);
            push(env, T::wrap(out))
        }),
        Stmt::Reduce { src, op } => by_ty!(ty_of(env, *src), T => {
            let out = reduce::<T>(get::<T>(env, *src), *op);
            push(env, T::wrap(out))
        }),
        Stmt::SumWhere { src, mask } => by_ty!(ty_of(env, *src), T => {
            let out = get::<T>(env, *src).sum_where(get::<u32>(env, *mask));
            push(env, T::wrap(out))
        }),
        Stmt::Scan { src, axis, op } => by_ty!(ty_of(env, *src), T => {
            let out = scan::<T>(get::<T>(env, *src), *axis, *op);
            push(env, T::wrap(out))
        }),
        Stmt::AxisReduce { src, axis, op } => by_ty!(ty_of(env, *src), T => {
            let out = axis_reduce::<T>(get::<T>(env, *src), *axis, *op);
            push(env, T::wrap(out))
        }),
        Stmt::Assign { target, path, rhs } => by_ty!(ty_of(env, *target), T => {
            let t = get::<T>(env, *target);
            let mut receiver: Option<Array<T>> = None;
            for step in path {
                let base = receiver.as_ref().unwrap_or(t);
                receiver = Some(match step {
                    InjView::InjSlice { axis, lo, hi } => base.slice(*axis, *lo..*hi),
                    InjView::InjFlip { axis } => base.flip(*axis),
                    InjView::InjTranspose => base.transpose(),
                });
            }
            receiver.as_ref().unwrap_or(t).assign(operand::<T>(env, rhs))
        }),
        Stmt::Put { target, idx, vals } => by_ty!(ty_of(env, *target), T => {
            let r = operand::<T>(env, vals);
            let shape = match vals {
                Opd::Var(i) => get::<T>(env, *i).shape(),
                Opd::Const(_) => Shape::new([idx.len(), 1, 1, 1]),
            };
            let indices = Array::<u32>::from_slice_on(ctx, idx, shape)?;
            get::<T>(env, *target).put(&indices, r)
        }),
        Stmt::Compound { target, op, rhs } => by_ty!(ty_of(env, *target), T => {
            let mut t = take::<T>(env, *target);
            let own = matches!(rhs, Opd::Var(i) if i == target).then(|| t.clone());
            let r = match &own {
                Some(copy) => R::A(copy),
                None => operand::<T>(env, rhs),
            };
            t.assign_op(r.rhs(), lbin(*op), "fuzz_assign_op");
            let failed = t.check();
            env[*target] = Some(T::wrap(t));
            failed
        }),
        Stmt::Write { target, seed } => by_ty!(ty_of(env, *target), T => {
            let mut t = take::<T>(env, *target);
            let n = t.len();
            let data: Vec<T> = generator::values(*seed, T::TY, n, cx.math).into_iter().map(T::from_f32).collect();
            let done = t.write(&data);
            env[*target] = Some(T::wrap(t));
            done
        }),
        Stmt::Cache { src } => by_ty!(ty_of(env, *src), T => {
            let out = get::<T>(env, *src).cache();
            push(env, T::wrap(out))
        }),
        Stmt::Alias { src } => {
            let out = env[*src].clone().expect("live");
            push(env, out)
        }
        Stmt::EvalVars { vars } => {
            eval_group::<f32>(env, vars)?;
            eval_group::<u32>(env, vars)?;
            eval_group::<Df32>(env, vars)
        }
        Stmt::Release { var } => {
            env[*var] = None;
            Ok(())
        }
        Stmt::Fetch { .. } => Ok(()),
        Stmt::Loop { .. } => unreachable!("loops are run by the machine"),
    }
}

/// `iterate_all` over `N` states of one type; the body runs on luma only.
fn luma_iterate_all<T: Num, const N: usize>(
    cx: &Cx,
    env: &mut Env,
    steps: usize,
    states: &[usize],
    body: &[Stmt],
    next: &[usize],
) -> Result<Vec<Arr>, LumaError> {
    let init: Vec<Array<T>> = states.iter().map(|&s| get::<T>(env, s).clone()).collect();
    let init: [&Array<T>; N] = std::array::from_fn(|k| &init[k]);
    let base = env.len();
    let cell = RefCell::new(std::mem::take(env));
    let failure: RefCell<Option<LumaError>> = RefCell::new(None);
    let out = iterate_all(init, steps, |st: &[Array<T>; N]| {
        let mut e = cell.borrow_mut();
        step_body::<T, N>(cx, &mut e, base, st, body, next, &failure)
    });
    *env = cell.into_inner();
    if let Some(error) = failure.into_inner() {
        return Err(error);
    }
    out.iter().try_for_each(Array::check)?;
    Ok(out.into_iter().map(T::wrap).collect())
}

/// One step of a traced loop: the states become the variables from `base`, the body runs, the
/// `next` variables come out.
fn step_body<T: Num, const N: usize>(
    cx: &Cx,
    env: &mut Env,
    base: usize,
    st: &[Array<T>; N],
    body: &[Stmt],
    next: &[usize],
    failure: &RefCell<Option<LumaError>>,
) -> [Array<T>; N] {
    env.truncate(base);
    for s in st {
        env.push(Some(T::wrap(s.clone())));
    }
    for stmt in body {
        if let Err(error) = luma_stmt(cx, env, stmt) {
            failure.borrow_mut().get_or_insert(error);
            env.truncate(base);
            return st.clone();
        }
    }
    let out = std::array::from_fn(|k| get::<T>(env, next[k]).clone());
    env.truncate(base);
    out
}

/// `iterate_until_all` over `N` `f32` states, stopping when the first state's maximum exceeds
/// `1e9`; returns the final states and the number of steps run.
fn luma_iterate_until<const N: usize>(
    cx: &Cx,
    env: &mut Env,
    steps: usize,
    states: &[usize],
    body: &[Stmt],
    next: &[usize],
) -> Result<(Vec<Arr>, usize), LumaError> {
    let init: [Array<f32>; N] = std::array::from_fn(|k| get::<f32>(env, states[k]).clone());
    let base = env.len();
    let cell = RefCell::new(std::mem::take(env));
    let failure: RefCell<Option<LumaError>> = RefCell::new(None);
    let until = Until::new(steps).record("max", |s: &[Array<f32>; N]| s[0].max()).stop_when(|h| h["max"].last().gt(1e9));
    let run = iterate_until_all(init, until, |st: &[Array<f32>; N], _step| {
        let mut e = cell.borrow_mut();
        step_body::<f32, N>(cx, &mut e, base, st, body, next, &failure)
    });
    *env = cell.into_inner();
    if let Some(error) = failure.into_inner() {
        return Err(error);
    }
    let run = run?;
    run.state.iter().try_for_each(Array::check)?;
    Ok((run.state.into_iter().map(Arr::F).collect(), run.steps))
}

// -- The oracle side --

type HostEnv = Vec<Option<HostArray>>;

fn host_get(env: &HostEnv, i: usize) -> &HostArray {
    env.get(i).and_then(Option::as_ref).unwrap_or_else(|| panic!("oracle variable {i} is not live"))
}

fn host_rhs<'a>(env: &'a HostEnv, o: &Opd, ty: Ty) -> HRhs<'a> {
    match o {
        Opd::Var(i) => HRhs::Array(host_get(env, *i)),
        Opd::Const(c) => HRhs::Val(match ty {
            Ty::U32 => f64::from(*c as u32),
            _ => f64::from(*c),
        }),
    }
}

fn host_view(env: &HostEnv, a: &HostArray, view: &ViewOp) -> Result<HostArray, ErrKind> {
    Ok(match view {
        ViewOp::Slice { axis, lo, hi } => a.slice(*axis, *lo, *hi)?,
        ViewOp::Pad { widths, mode } => a.pad(widths, *mode),
        ViewOp::Flip { axis } => a.flip(*axis)?,
        ViewOp::Transpose => a.transpose(),
        ViewOp::ShiftCirc { offs } => a.shift(offs, Edge::Wrap),
        ViewOp::ShiftFill { offs, value } => a.shift(offs, Edge::Fill(oracle::store(a.ty, f64::from(*value)))),
        ViewOp::ShiftClamp { offs } => a.shift(offs, Edge::Clamp),
        ViewOp::ShiftMirror { offs } => a.shift(offs, Edge::Mirror),
        ViewOp::Take { idx } => a.take(host_get(env, *idx)),
        ViewOp::Concat { axis, others } => {
            let mut parts = vec![a];
            parts.extend(others.iter().map(|&o| host_get(env, o)));
            HostArray::concat(*axis, &parts)?
        }
    })
}

/// Runs a statement on the oracle; creations push their variables, a failure creates none.
fn host_stmt(env: &mut HostEnv, stmt: &Stmt, math: bool) -> Result<(), ErrKind> {
    let made: Option<HostArray> = match stmt {
        Stmt::Input { dims, ty, seed } => {
            Some(HostArray::new(*dims, *ty, generator::values(*seed, *ty, oracle::len_of(*dims), math).into_iter().map(f64::from).collect()))
        }
        Stmt::Full { dims, ty, value } => Some(HostArray::full(*dims, *ty, f64::from(*value))),
        Stmt::Range { dims, axis, ty } => Some(HostArray::range(*dims, *ty, *axis)?),
        Stmt::Linspace { start, end, n } => Some(HostArray::linspace(*start, *end, *n)),
        Stmt::Unary { src, op } => Some(host_get(env, *src).unary(*op)),
        Stmt::Binary { lhs, rhs, op } => Some(match (lhs, rhs) {
            (Opd::Var(i), r) => {
                let a = host_get(env, *i);
                a.binary(*op, host_rhs(env, r, a.ty), false)?
            }
            (Opd::Const(c), Opd::Var(j)) => {
                let b = host_get(env, *j);
                b.binary(*op, host_rhs(env, &Opd::Const(*c), b.ty), true)?
            }
            _ => panic!("operation on two constants"),
        }),
        Stmt::Compare { lhs, rhs, op } => {
            let a = host_get(env, *lhs);
            Some(a.compare(*op, host_rhs(env, rhs, a.ty))?)
        }
        Stmt::Select { mask, a, b, ty } => Some(host_get(env, *mask).select(host_rhs(env, a, *ty), host_rhs(env, b, *ty), *ty)?),
        Stmt::View { src, view } => Some(host_view(env, host_get(env, *src), view)?),
        Stmt::Reduce { src, op } => Some(host_get(env, *src).reduce(*op)),
        Stmt::SumWhere { src, mask } => Some(host_get(env, *src).sum_where(host_get(env, *mask))?),
        Stmt::Scan { src, axis, op } => Some(host_get(env, *src).scan(*axis, *op)),
        Stmt::AxisReduce { src, axis, op } => Some(host_get(env, *src).reduce_axis(*axis, *op)),
        Stmt::Assign { target, path, rhs } => {
            let t = host_get(env, *target);
            let mut receiver = t.clone();
            for step in path {
                receiver = match step {
                    InjView::InjSlice { axis, lo, hi } => receiver.slice(*axis, *lo, *hi)?,
                    InjView::InjFlip { axis } => receiver.flip(*axis)?,
                    InjView::InjTranspose => receiver.transpose(),
                };
            }
            receiver.assign(host_rhs(env, rhs, t.ty))?;
            None
        }
        Stmt::Put { target, idx, vals } => {
            let t = host_get(env, *target);
            let dims = match vals {
                Opd::Var(i) => host_get(env, *i).dims,
                Opd::Const(_) => [idx.len(), 1, 1, 1],
            };
            let indices = HostArray::new(dims, Ty::U32, idx.iter().map(|&i| f64::from(i)).collect());
            t.put(&indices, host_rhs(env, vals, t.ty))?;
            None
        }
        Stmt::Compound { target, op, rhs } => {
            let t = host_get(env, *target);
            t.compound(*op, host_rhs(env, rhs, t.ty))?;
            None
        }
        Stmt::Write { target, seed } => {
            let t = host_get(env, *target);
            t.write(&generator::values(*seed, t.ty, t.len(), math).into_iter().map(f64::from).collect::<Vec<_>>())?;
            None
        }
        Stmt::Cache { src } => Some(host_get(env, *src).cached()),
        Stmt::Alias { src } => Some(host_get(env, *src).clone()),
        Stmt::EvalVars { .. } | Stmt::Fetch { .. } => None,
        Stmt::Release { var } => {
            env[*var] = None;
            None
        }
        Stmt::Loop { .. } => unreachable!("loops are run by the machine"),
    };
    env.extend(made.map(Some));
    Ok(())
}

/// The oracle's `iterate_all` / `iterate_until_all`: the final states and the steps run.
fn host_iterate(
    env: &mut HostEnv,
    steps: usize,
    states: &[usize],
    body: &[Stmt],
    next: &[usize],
    until: bool,
    math: bool,
) -> Result<(Vec<HostArray>, usize), ErrKind> {
    let base = env.len();
    let mut current: Vec<HostArray> = states.iter().map(|&s| host_get(env, s).copied()).collect();
    let mut done = 0;
    for _ in 0..steps {
        env.truncate(base);
        env.extend(current.iter().map(|c| Some(c.clone())));
        for stmt in body {
            host_stmt(env, stmt, math)?;
        }
        current = next.iter().map(|&i| host_get(env, i).copied()).collect();
        done += 1;
        if until && current[0].reduce(RedOp::Max).data()[0] > 1e9 {
            break;
        }
    }
    env.truncate(base);
    Ok((current, done))
}

/// How many variables each top-level statement creates, from a run of the oracle alone (a
/// statement that fails creates none).
fn created_counts(program: &Program) -> Vec<usize> {
    let mut env: HostEnv = Vec::new();
    let math = program.math;
    program
        .stmts
        .iter()
        .map(|stmt| {
            let before = env.len();
            match stmt {
                Stmt::Loop { kind: LoopKind::PerStepEval, steps, states, body, next } => {
                    let base = env.len();
                    for &s in states {
                        let copy = host_get(&env, s).copied();
                        env.push(Some(copy));
                    }
                    for _ in 0..*steps {
                        env.truncate(base + states.len());
                        for stmt in body {
                            let _ = host_stmt(&mut env, stmt, math);
                        }
                        let finals: Vec<Option<HostArray>> = next.iter().map(|&i| env[i].clone()).collect();
                        env.truncate(base);
                        env.extend(finals);
                    }
                }
                Stmt::Loop { kind, steps, states, body, next } => {
                    let until = *kind == LoopKind::IterateUntilAll;
                    if let Ok((finals, _)) = host_iterate(&mut env, *steps, states, body, next, until, math) {
                        env.extend(finals.into_iter().map(Some));
                    }
                }
                other => {
                    let _ = host_stmt(&mut env, other, math);
                }
            }
            env.len() - before
        })
        .collect()
}

// -- Comparison --

/// The first element where luma and the oracle disagree.
fn first_difference(math: bool, ty: Ty, luma: &[f64], oracle: &[f64]) -> Option<(usize, f64, f64)> {
    luma.iter().zip(oracle).enumerate().find_map(|(k, (&l, &h))| {
        let same = if l.is_nan() || h.is_nan() {
            l.is_nan() && h.is_nan()
        } else if l == h {
            true
        } else if math || ty == Ty::Df32 {
            let (rel, abs) = if math { (2e-3, 1e-4) } else { (1e-12, 1e-12) };
            (l - h).abs() <= abs + rel * h.abs()
        } else {
            false
        };
        (!same).then_some((k, l, h))
    })
}

enum Fail {
    Mismatch(String),
    /// Luma reported a documented limit: the program ends here.
    Limit(String),
}

fn limit_or_fail(error: &LumaError, what: &str) -> Fail {
    match kind_of(error) {
        ErrKind::Chunked | ErrKind::Unsupported => Fail::Limit(format!("{what}: {error}")),
        _ => Fail::Mismatch(format!("{what}: luma failed: {error}")),
    }
}

// -- The machine --

struct Machine {
    cx: Cx,
    luma: Env,
    host: HostEnv,
}

impl Machine {
    fn new(ctx: Context, math: bool) -> Machine {
        Machine { cx: Cx { ctx, math }, luma: Vec::new(), host: Vec::new() }
    }

    fn truncate(&mut self, len: usize) {
        self.luma.truncate(len);
        self.host.truncate(len);
    }

    /// Compares variable `i` on both sides.
    fn check_var(&self, i: usize, at: &str) -> Result<(), Fail> {
        let (Some(l), Some(h)) = (self.luma.get(i).and_then(Option::as_ref), self.host.get(i).and_then(Option::as_ref)) else {
            return Ok(());
        };
        let describe = format!("{at}: variable {i} ({:?} {:?})", h.ty, h.dims);
        if l.dims() != h.dims || l.ty() != h.ty {
            return Err(Fail::Mismatch(format!("{describe}: luma is {:?} {:?}", l.ty(), l.dims())));
        }
        let got = l.read().map_err(|e| limit_or_fail(&e, &format!("{describe}: reading")))?;
        match first_difference(self.cx.math, h.ty, &got, &h.data()) {
            Some((k, l, h)) => Err(Fail::Mismatch(format!("{describe}: element {k} is {l} on luma, {h} on the oracle"))),
            None => Ok(()),
        }
    }

    fn check_all(&self, at: &str) -> Result<(), Fail> {
        (0..self.luma.len()).try_for_each(|i| self.check_var(i, at))
    }

    /// Runs a top-level statement (`created` is the number of variables it creates when it
    /// works) or a statement of a loop body (`None`).
    ///
    /// A statement that only creates arrays and that luma refuses for a documented limit (a
    /// scan of a double-float array, an operation on an array split over several buffers)
    /// makes dead variables instead: later statements that read one are skipped too.
    fn step(&mut self, stmt: &Stmt, at: &str, created: Option<usize>) -> Result<(), Fail> {
        if let Some(n) = created
            && stmt.mentions().iter().any(|&i| i < self.luma.len() && self.luma[i].is_none())
        {
            self.dead(n);
            return Ok(());
        }
        match stmt {
            Stmt::Loop { kind, steps, states, body, next } => self.run_loop(*kind, *steps, states, body, next, at),
            Stmt::Fetch { var } => self.check_var(*var, at),
            _ => {
                let before = self.luma.len();
                let host = host_stmt(&mut self.host, stmt, self.cx.math);
                let luma = luma_stmt(&self.cx, &mut self.luma, stmt);
                if let (Some(n), Ok(()), Err(error)) = (created, &host, &luma)
                    && stmt.creates_only()
                    && matches!(kind_of(error), ErrKind::Chunked | ErrKind::Unsupported)
                {
                    self.truncate(before);
                    self.dead(n);
                    return Ok(());
                }
                self.compare_outcomes(host, luma, at)
            }
        }
    }

    /// `n` variables that do not exist (luma refused what creates them).
    fn dead(&mut self, n: usize) {
        SKIPPED.fetch_add(1, Ordering::Relaxed);
        for _ in 0..n {
            self.luma.push(None);
            self.host.push(None);
        }
    }

    fn compare_outcomes(&self, host: Result<(), ErrKind>, luma: Result<(), LumaError>, at: &str) -> Result<(), Fail> {
        if let Err(e) = &luma
            && matches!(kind_of(e), ErrKind::Chunked | ErrKind::Unsupported)
        {
            return Err(Fail::Limit(format!("{at}: {e}")));
        }
        match (host, luma) {
            (Ok(()), Ok(())) => {
                if self.host.len() != self.luma.len() {
                    return Err(Fail::Mismatch(format!("{at}: the sides created different numbers of variables")));
                }
                Ok(())
            }
            (Err(h), Err(l)) if kind_of(&l) == h => Ok(()),
            (Err(h), Err(l)) => Err(Fail::Mismatch(format!("{at}: luma failed with {l}, the oracle with {h:?}"))),
            (Err(h), Ok(())) => Err(Fail::Mismatch(format!("{at}: luma succeeded where the oracle fails with {h:?}"))),
            (Ok(()), Err(l)) => Err(Fail::Mismatch(format!("{at}: luma failed where the oracle succeeds: {l}"))),
        }
    }

    fn run_loop(&mut self, kind: LoopKind, steps: usize, states: &[usize], body: &[Stmt], next: &[usize], at: &str) -> Result<(), Fail> {
        match kind {
            LoopKind::PerStepEval => self.per_step(steps, states, body, next, at),
            LoopKind::IterateAll | LoopKind::IterateUntilAll => self.iterate(kind, steps, states, body, next, at),
        }
    }

    /// A host loop: the states are copies, every step runs the body and evaluates the new
    /// states, in lockstep with the oracle.
    fn per_step(&mut self, steps: usize, states: &[usize], body: &[Stmt], next: &[usize], at: &str) -> Result<(), Fail> {
        let base = self.luma.len();
        let n = states.len();
        for &s in states {
            let copy = with_arr!(self.luma[s].as_ref().expect("live"), a => Arr::from_copy(a));
            self.luma.push(Some(copy));
            let host = host_get(&self.host, s).copied();
            self.host.push(Some(host));
        }
        for t in 0..steps {
            self.truncate(base + n);
            for (k, stmt) in body.iter().enumerate() {
                self.step(stmt, &format!("{at}.body[{k}] (step {t})"), None)?;
            }
            let luma: Vec<Option<Arr>> = next.iter().map(|&i| self.luma[i].clone()).collect();
            let host: Vec<Option<HostArray>> = next.iter().map(|&i| self.host[i].clone()).collect();
            self.truncate(base);
            self.luma.extend(luma);
            self.host.extend(host);
            let vars: Vec<usize> = (base..base + n).collect();
            let evaluated = eval_group::<f32>(&self.luma, &vars)
                .and_then(|()| eval_group::<u32>(&self.luma, &vars))
                .and_then(|()| eval_group::<Df32>(&self.luma, &vars));
            evaluated.map_err(|e| limit_or_fail(&e, &format!("{at}: eval (step {t})")))?;
        }
        Ok(())
    }

    /// `iterate_all` / `iterate_until_all`: the body runs once on luma (traced), `steps` times
    /// on the oracle.
    fn iterate(&mut self, kind: LoopKind, steps: usize, states: &[usize], body: &[Stmt], next: &[usize], at: &str) -> Result<(), Fail> {
        let until = kind == LoopKind::IterateUntilAll;
        let (finals, host_steps) = host_iterate(&mut self.host, steps, states, body, next, until, self.cx.math)
            .map_err(|e| Fail::Mismatch(format!("{at}: the oracle failed in the loop body: {e:?} (generator bug)")))?;
        let ty = ty_of(&self.luma, states[0]);
        let cx = &self.cx;
        let env = &mut self.luma;
        let run: Result<(Vec<Arr>, Option<usize>), LumaError> = match (until, ty, states.len()) {
            (false, Ty::F32, 1) => luma_iterate_all::<f32, 1>(cx, env, steps, states, body, next).map(|r| (r, None)),
            (false, Ty::F32, 2) => luma_iterate_all::<f32, 2>(cx, env, steps, states, body, next).map(|r| (r, None)),
            (false, Ty::U32, 1) => luma_iterate_all::<u32, 1>(cx, env, steps, states, body, next).map(|r| (r, None)),
            (false, Ty::U32, 2) => luma_iterate_all::<u32, 2>(cx, env, steps, states, body, next).map(|r| (r, None)),
            (false, Ty::Df32, 1) => luma_iterate_all::<Df32, 1>(cx, env, steps, states, body, next).map(|r| (r, None)),
            (false, Ty::Df32, 2) => luma_iterate_all::<Df32, 2>(cx, env, steps, states, body, next).map(|r| (r, None)),
            (true, Ty::F32, 1) => luma_iterate_until::<1>(cx, env, steps, states, body, next).map(|(r, s)| (r, Some(s))),
            (true, Ty::F32, 2) => luma_iterate_until::<2>(cx, env, steps, states, body, next).map(|(r, s)| (r, Some(s))),
            _ => panic!("loop over {} {ty:?} states", states.len()),
        };
        let (results, luma_steps) = run.map_err(|e| limit_or_fail(&e, &format!("{at}: loop")))?;
        if let Some(luma_steps) = luma_steps
            && luma_steps != host_steps
        {
            return Err(Fail::Mismatch(format!("{at}: iterate_until_all ran {luma_steps} steps, the oracle {host_steps}")));
        }
        self.luma.extend(results.into_iter().map(Some));
        self.host.extend(finals.into_iter().map(Some));
        Ok(())
    }
}

impl Arr {
    /// A new array holding the current values (`Array::copied`).
    fn from_copy<T: Num>(a: &Array<T>) -> Arr {
        T::wrap(a.copied())
    }
}

// -- Programs --

/// The context of a program under its configuration; `None` if the device refuses it.
fn engine(config: &Config) -> Option<Context> {
    let mut options = ContextOptions::from_env().calibrate(false).fast_math(config.fast_math);
    if config.df_split {
        options = options.df_fma(false);
    }
    if let Some(bytes) = config.chunk_bytes {
        options = options.chunk_bytes(bytes);
    }
    if let Some(bytes) = config.memory_budget {
        options = options.memory_budget(bytes);
    }
    let ctx = Context::with_options(options).ok()?;
    let jit = ctx.jit();
    jit.set_max_nodes(config.max_nodes);
    if let Some(n) = config.max_inlined {
        jit.set_max_inlined(n);
    }
    if let Some(n) = config.max_live {
        jit.set_max_live(n);
    }
    if let Some(n) = config.max_inputs {
        jit.set_max_inputs(n);
    }
    ctx.set_assume_busy(config.assume_busy);
    jit.set_speculate(config.speculate);
    jit.set_simplify(config.simplify);
    jit.set_staging(config.staging);
    ctx.set_cost_model(match config.cost {
        CostChoice::Test => CostModel::TEST,
        CostChoice::SlowLaunch => CostModel { launch_us: 1e6, ..CostModel::TEST },
        CostChoice::SlowArithmetic => CostModel { f32_ops_per_ns: 1e-6, ..CostModel::TEST },
    });
    Some(ctx)
}

/// A failure recorded in `fuzz-findings.md` and as an `#[ignore]`d test in `regressions.rs`. A
/// random program that fails like this is counted, not reported, so that the runs go on to
/// find other failures; nothing in the generator avoids them. A failure is matched by a piece
/// of its message, or (when it is a wrong value) by `shape`, a predicate on the *minimized*
/// failing program.
struct Known {
    id: &'static str,
    message: Option<&'static str>,
    shape: Option<fn(&Program) -> bool>,
}

const KNOWN: &[Known] = &[

];

/// Programs that failed in a way listed in [`KNOWN`], by finding.
static KNOWN_HITS: std::sync::Mutex<Vec<(&'static str, usize)>> = std::sync::Mutex::new(Vec::new());

fn note_known(id: &'static str) {
    let mut hits = KNOWN_HITS.lock().unwrap_or_else(|e| e.into_inner());
    match hits.iter_mut().find(|(known, _)| *known == id) {
        Some((_, n)) => *n += 1,
        None => hits.push((id, 1)),
    }
}

/// Which known failures a run lets pass.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Filter {
    /// None: every failure is reported (the regression tests).
    Off,
    /// Those matched by their message.
    Messages,
    /// Those matched by message or by the shape of the minimized program.
    Shapes,
}

/// Runs `program` on `ctx` and the oracle; `Err` describes the first difference.
fn run_on(ctx: Context, program: &Program, filter: Filter) -> Result<(), String> {
    let mut machine = Machine::new(ctx, program.math);
    let created = created_counts(program);
    let outcome = program
        .stmts
        .iter()
        .enumerate()
        .try_for_each(|(k, stmt)| machine.step(stmt, &format!("statement {k}"), Some(created[k])))
        .and_then(|()| machine.check_all("at the end"));
    match outcome {
        Ok(()) => Ok(()),
        Err(Fail::Limit(what)) => {
            if std::env::var_os("LUMA_FUZZ_VERBOSE").is_some() {
                eprintln!("fuzz: program ended early: {what}");
            }
            ENDED_EARLY.fetch_add(1, Ordering::Relaxed);
            Ok(())
        }
        Err(Fail::Mismatch(what)) => {
            if filter != Filter::Off
                && let Some(known) = KNOWN.iter().find(|k| k.message.is_some_and(|m| what.contains(m)))
            {
                note_known(known.id);
                return Ok(());
            }
            // Minimizing only removes statements: a shape the program does not have, the minimum lacks.
            if filter == Filter::Shapes && KNOWN.iter().any(|k| k.shape.is_some_and(|shape| shape(program))) {
                let minimal = minimize(program.clone(), true);
                if let Some(known) = KNOWN.iter().find(|k| k.shape.is_some_and(|shape| shape(&minimal))) {
                    note_known(known.id);
                    return Ok(());
                }
            }
            Err(what)
        }
    }
}

/// Runs `program` under its own configuration on a fresh context with the given filter.
fn run_filtered(program: &Program, filter: Filter) -> Result<(), String> {
    match engine(&program.config) {
        Some(ctx) => run_on(ctx, program, filter),
        None => Ok(()),
    }
}

/// Runs `program` reporting every failure (the regression tests).
pub(crate) fn run(program: &Program) -> Result<(), String> {
    run_filtered(program, Filter::Off)
}

/// [`run`] for the random runs: known failures are counted instead.
fn run_random(program: &Program) -> Result<(), String> {
    run_filtered(program, Filter::Shapes)
}

/// `LUMA_FUZZ_CASES`, by default 256 (24 in debug builds, which are slow).
fn cases() -> u32 {
    let default = if cfg!(debug_assertions) { 24 } else { 256 };
    std::env::var("LUMA_FUZZ_CASES").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(default)
}

fn config(file: &'static str) -> ProptestConfig {
    let seed = std::env::var("LUMA_FUZZ_SEED").ok().and_then(|v| v.trim().parse::<u64>().ok());
    ProptestConfig {
        cases: cases(),
        max_shrink_iters: 2000,
        failure_persistence: Some(Box::new(FileFailurePersistence::Direct(file))),
        rng_seed: seed.map_or(RngSeed::Random, RngSeed::Fixed),
        ..ProptestConfig::default()
    }
}

/// The part of a failure message that identifies the failure: without the statement number
/// and the numbers in it.
fn failure_class(message: &str) -> String {
    message.split_once(": ").map_or(message, |(_, rest)| rest).chars().filter(|c| !c.is_ascii_digit()).collect()
}

/// The smallest program that fails in the same way, found by dropping statements and
/// resetting configuration switches while the failure stays. `quick` stops after one pass over
/// the statements (for the filter of known failures, which minimizes many programs).
fn minimize(program: Program, quick: bool) -> Program {
    let Err(first) = run_filtered(&program, Filter::Messages) else { return program };
    let class = failure_class(&first);
    // A candidate that no longer makes sense may panic: it does not fail "in the same way".
    let fails = |candidate: &Program| {
        let hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| run_filtered(candidate, Filter::Messages)));
        std::panic::set_hook(hook);
        matches!(outcome, Ok(Err(m)) if failure_class(&m) == class)
    };
    let mut best = program;
    loop {
        let mut changed = false;
        for k in (0..best.stmts.len()).rev() {
            if let Some(candidate) = best.without(k, &created_counts(&best))
                && fails(&candidate)
            {
                best = candidate;
                changed = true;
            }
        }
        if quick {
            return best;
        }
        let default = Config::default_engine();
        let resets: [fn(&mut Config, &Config); 12] = [
            |c, d| c.max_nodes = d.max_nodes,
            |c, d| c.max_inlined = d.max_inlined,
            |c, d| c.max_live = d.max_live,
            |c, d| c.max_inputs = d.max_inputs,
            |c, d| c.assume_busy = d.assume_busy,
            |c, d| c.speculate = d.speculate,
            |c, d| c.simplify = d.simplify,
            |c, d| c.fast_math = d.fast_math,
            |c, d| c.df_split = d.df_split,
            |c, d| c.chunk_bytes = d.chunk_bytes,
            |c, d| c.memory_budget = d.memory_budget,
            |c, d| c.cost = d.cost,
        ];
        for reset in resets {
            let mut candidate = best.clone();
            reset(&mut candidate.config, &default);
            if candidate != best && fails(&candidate) {
                best = candidate;
                changed = true;
            }
        }
        if !changed {
            return best;
        }
    }
}

/// Runs `body` on `cases()` random values of `strategy`, then reports how many programs ended
/// early and how many failures were known ones. Panics with the shrunk failure, described by
/// `describe`.
fn drive<S: Strategy>(
    file: &'static str,
    strategy: S,
    describe: impl Fn(S::Value) -> String,
    body: impl Fn(S::Value) -> Result<(), String>,
) {
    let mut runner = TestRunner::new(config(file));
    let result = runner.run(&strategy, |value| body(value).map_err(TestCaseError::fail));
    eprintln!(
        "fuzz: {} cases, {} programs ended early and {} statements were skipped on documented limits, known failures: {:?}",
        cases(),
        ENDED_EARLY.swap(0, Ordering::Relaxed),
        SKIPPED.swap(0, Ordering::Relaxed),
        std::mem::take(&mut *KNOWN_HITS.lock().unwrap_or_else(|e| e.into_inner()))
    );
    match result {
        Ok(()) => {}
        Err(TestError::Fail(reason, value)) => panic!("{reason}\n\nminimized input:\n{}", describe(value)),
        Err(other) => panic!("{other}"),
    }
}

/// Random programs under random configurations, one at a time.
#[test]
fn programs() {
    let describe = |program: Program| {
        let minimal = minimize(program, false);
        format!("{}\n{}", run_filtered(&minimal, Filter::Messages).err().unwrap_or_default(), minimal.rust())
    };
    drive("proptest-regressions/fuzz_programs.txt", generator::programs(), describe, |program| run_random(&program));
}

/// Runs each of `programs` (under the default engine configuration) on its own thread, all on
/// `ctx`, and returns the failure of every one that failed (a panic is a failure).
fn run_concurrently(ctx: &Context, programs: &[Program]) -> Vec<(usize, String)> {
    let outcomes: Vec<Result<(), String>> = std::thread::scope(|scope| {
        let handles: Vec<_> = programs
            .iter()
            .map(|program| {
                let ctx = ctx.clone();
                scope.spawn(move || {
                    let mut program = program.clone();
                    program.config = Config::default_engine();
                    run_on(ctx, &program, Filter::Messages)
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap_or_else(|_| Err("a thread panicked".into()))).collect()
    });
    outcomes.into_iter().enumerate().filter_map(|(k, outcome)| outcome.err().map(|failure| (k, failure))).collect()
}

/// Whether `program` fails when it runs by itself on `ctx` (a panic counts as failing).
fn fails_alone(ctx: &Context, program: &Program) -> bool {
    let mut program = program.clone();
    program.config = Config::default_engine();
    let hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| run_on(ctx.clone(), &program, Filter::Off)));
    std::panic::set_hook(hook);
    !matches!(outcome, Ok(Ok(())))
}

/// Four programs at once on four threads sharing the default context, each against its own
/// oracle; the engine runs with its defaults. Only what the concurrency changes fails here: a
/// program that fails in the concurrent run is run again by itself on the same context, and if
/// it fails then too it is a failure of `programs` (counted, not reported).
#[test]
fn threads() {
    let set = (generator::programs(), generator::programs(), generator::programs(), generator::programs());
    drive("proptest-regressions/fuzz_threads.txt", set, |set| format!("{set:?}"), |set| {
        let ctx = crate::default_context().map_err(|e| e.to_string())?;
        let programs = [set.0, set.1, set.2, set.3];
        for (k, failure) in run_concurrently(&ctx, &programs) {
            if fails_alone(&ctx, &programs[k]) {
                note_known("fails alone too (reported by fuzz::programs)");
            } else {
                return Err(format!("only when run concurrently: {failure}"));
            }
        }
        Ok(())
    });
}

/// The generator produces every kind of statement, loops of every kind, both domains and
/// invalid statements; and the programs mostly run to the end.
#[test]
fn generator_covers_the_statements() {
    use proptest::strategy::{Strategy, ValueTree};
    use std::collections::BTreeMap;

    fn walk(stmts: &[Stmt], out: &mut BTreeMap<String, usize>) {
        for stmt in stmts {
            let name = format!("{stmt:?}");
            let name = name.split(|c: char| !c.is_alphanumeric()).next().unwrap_or_default().to_owned();
            *out.entry(name).or_default() += 1;
            if let Stmt::Loop { kind, body, .. } = stmt {
                *out.entry(format!("Loop/{kind:?}")).or_default() += 1;
                walk(body, out);
            }
            match stmt {
                Stmt::Binary { op, .. } | Stmt::Compound { op, .. } if op.bitwise() => *out.entry("Bitwise".to_owned()).or_default() += 1,
                Stmt::Unary { op: UnOp::Not, .. } => *out.entry("Not".to_owned()).or_default() += 1,
                _ => {}
            }
            if let Stmt::View { view, .. } = stmt {
                let name = format!("{view:?}");
                let name = name.split(|c: char| !c.is_alphanumeric()).next().unwrap_or_default().to_owned();
                *out.entry(format!("View/{name}")).or_default() += 1;
            }
        }
    }

    let mut runner = proptest::test_runner::TestRunner::deterministic();
    let strategy = generator::programs();
    let mut counts = BTreeMap::new();
    let (mut stmts, mut math, mut total) = (0, 0, 0);
    for _ in 0..400 {
        let program = strategy.new_tree(&mut runner).unwrap().current();
        walk(&program.stmts, &mut counts);
        stmts += program.stmts.len();
        math += usize::from(program.math);
        total += 1;
    }
    eprintln!("{total} programs, {stmts} statements, {math} math programs\n{counts:#?}");
    for kind in [
        "Input", "Full", "Range", "Linspace", "Unary", "Binary", "Compare", "Select", "View", "Reduce", "SumWhere", "Scan", "AxisReduce",
        "Assign", "Put", "Compound", "Write", "Cache", "EvalVars", "Alias", "Release", "Fetch", "Loop/PerStepEval", "Loop/IterateAll",
        "Loop/IterateUntilAll", "View/Slice", "View/Pad", "View/Flip", "View/Transpose", "View/ShiftCirc", "View/ShiftFill",
        "View/ShiftClamp", "View/ShiftMirror", "View/Take", "View/Concat", "Bitwise", "Not",
    ] {
        assert!(counts.get(kind).copied().unwrap_or(0) >= 5, "{kind} is rare: {counts:?}");
    }
    assert!(stmts / total >= 15, "programs are short: {} statements on average", stmts / total);
}
