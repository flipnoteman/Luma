//! The rule checker: ℤ/p fingerprints, the class check against exact-`f32` semantics, and the
//! rule-application check against `simplify.rs`.
//!
//! Every context is pinned to exact `f32` semantics: no fast math, no TF32 (`Numerics::strict()`; the flagged passes of the
//! rule-application check run in a context that only adds the three class flags).

use super::table::*;
use crate::jit::UnOp;
use super::*;
use crate::array::Array;
use crate::context::{Context, ContextOptions};
use crate::ops::select;
use proptest::prelude::*;
use proptest::strategy::ValueTree;
use proptest::test_runner::TestRunner;

// ---------------------------------------------------------------------------------------------
// Tensors and the two scalar fields.
// ---------------------------------------------------------------------------------------------

/// A row-major matrix (luma's shape `[cols, rows]`: the first dimension varies fastest); a vector
/// is `1 × n`, a constant `1 × 1` (broadcast by the elementwise operations).
#[derive(Clone, Debug)]
struct T<V> {
    rows: usize,
    cols: usize,
    data: Vec<V>,
}

impl<V: Copy> T<V> {
    /// The element at row `r`, column `c` of this matrix broadcast to a larger one: an extent of 1 repeats.
    fn at(&self, r: usize, c: usize) -> V {
        self.data[(if self.rows == 1 { 0 } else { r }) * self.cols + if self.cols == 1 { 0 } else { c }]
    }
}

trait Num: Copy {
    fn add(self, o: Self) -> Self;
    fn sub(self, o: Self) -> Self;
    fn mul(self, o: Self) -> Self;
    fn div(self, o: Self) -> Self;
    fn neg(self) -> Self;
    fn zero() -> Self;
    /// `None`: the constant is not representable.
    fn cst(c: f64) -> Option<Self>;
    /// Operations outside the field; `None` makes the rule not fingerprintable.
    fn min(self, o: Self) -> Option<Self>;
    fn max(self, o: Self) -> Option<Self>;
    fn cmp(self, c: Cmp, o: Self) -> Option<Self>;
    fn select(self, a: Self, b: Self) -> Option<Self>;
    fn unary(self, op: UnOp) -> Option<Self>;
    fn powf(self, p: f64) -> Option<Self>;
}

impl Num for f64 {
    fn add(self, o: f64) -> f64 {
        self + o
    }
    fn sub(self, o: f64) -> f64 {
        self - o
    }
    fn mul(self, o: f64) -> f64 {
        self * o
    }
    fn div(self, o: f64) -> f64 {
        self / o
    }
    fn neg(self) -> f64 {
        -self
    }
    fn zero() -> f64 {
        0.0
    }
    fn cst(c: f64) -> Option<f64> {
        Some(c)
    }
    fn min(self, o: f64) -> Option<f64> {
        Some(f64::min(self, o))
    }
    fn max(self, o: f64) -> Option<f64> {
        Some(f64::max(self, o))
    }
    fn cmp(self, c: Cmp, o: f64) -> Option<f64> {
        let t = match c {
            Cmp::Lt => self < o,
            Cmp::Le => self <= o,
            Cmp::Gt => self > o,
            Cmp::Ge => self >= o,
            Cmp::Eq => self == o,
            Cmp::Ne => self != o,
        };
        Some(f64::from(u8::from(t)))
    }
    fn select(self, a: f64, b: f64) -> Option<f64> {
        Some(if self != 0.0 { a } else { b })
    }
    fn unary(self, op: UnOp) -> Option<f64> {
        Some(match op {
            UnOp::Neg => -self,
            UnOp::Abs => self.abs(),
            UnOp::Floor => self.floor(),
            UnOp::Ceil => self.ceil(),
            UnOp::Trunc => self.trunc(),
            UnOp::Sqrt => self.sqrt(),
            UnOp::Sign => {
                if self.is_nan() {
                    self
                } else if self > 0.0 {
                    1.0
                } else if self < 0.0 {
                    -1.0
                } else {
                    0.0
                }
            }
            _ => panic!("unary {op:?} has no f64 reference"),
        })
    }
    fn powf(self, p: f64) -> Option<f64> {
        Some(f64::powf(self, p))
    }
}

/// An element of ℤ/p, p = 2³¹ − 1.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Zp(u64);

const P: u64 = (1 << 31) - 1;

impl Zp {
    fn pow(self, mut e: u64) -> Zp {
        let (mut base, mut acc) = (self, Zp(1));
        while e > 0 {
            if e & 1 == 1 {
                acc = acc.mul(base);
            }
            base = base.mul(base);
            e >>= 1;
        }
        acc
    }
    fn inv(self) -> Zp {
        self.pow(P - 2)
    }
}

impl Num for Zp {
    fn add(self, o: Zp) -> Zp {
        Zp((self.0 + o.0) % P)
    }
    fn sub(self, o: Zp) -> Zp {
        Zp((self.0 + P - o.0) % P)
    }
    fn mul(self, o: Zp) -> Zp {
        Zp(self.0 * o.0 % P)
    }
    fn div(self, o: Zp) -> Zp {
        self.mul(o.inv())
    }
    fn neg(self) -> Zp {
        Zp((P - self.0) % P)
    }
    fn zero() -> Zp {
        Zp(0)
    }
    /// A dyadic rational `m / 2^s` (every finite `f64` is one, if `m` is small enough).
    fn cst(c: f64) -> Option<Zp> {
        for s in 0..=60u32 {
            let scaled = c * (1u64 << s) as f64;
            if scaled.fract() == 0.0 && scaled.abs() < 9e15 {
                let m = scaled as i64;
                let magnitude = Zp(m.unsigned_abs() % P);
                let value = if m < 0 { magnitude.neg() } else { magnitude };
                return Some(value.mul(Zp(2).pow(u64::from(s)).inv()));
            }
        }
        None
    }
    fn min(self, _: Zp) -> Option<Zp> {
        None
    }
    fn max(self, _: Zp) -> Option<Zp> {
        None
    }
    fn cmp(self, _: Cmp, _: Zp) -> Option<Zp> {
        None
    }
    fn select(self, _: Zp, _: Zp) -> Option<Zp> {
        None
    }
    fn unary(self, op: UnOp) -> Option<Zp> {
        if op == UnOp::Neg { Some(self.neg()) } else { None }
    }
    fn powf(self, p: f64) -> Option<Zp> {
        if p == -1.0 {
            Some(self.inv())
        } else if p >= 0.0 && p.fract() == 0.0 {
            Some(self.pow(p as u64))
        } else {
            None
        }
    }
}

/// Evaluates `e` over `vars`; `None` if it leaves the field (ℤ/p) or a constant is unrepresentable.
fn eval<N: Num>(e: &E, vars: &[T<N>]) -> Option<T<N>> {
    fn zip<N: Num>(a: T<N>, b: T<N>, f: impl Fn(N, N) -> Option<N>) -> Option<T<N>> {
        let (rows, cols) = (a.rows.max(b.rows), a.cols.max(b.cols));
        let at = |t: &T<N>, i: usize| t.at(i / cols, i % cols);
        let data = (0..rows * cols).map(|i| f(at(&a, i), at(&b, i))).collect::<Option<Vec<_>>>()?;
        Some(T { rows, cols, data })
    }
    fn map<N: Num>(a: T<N>, f: impl Fn(N) -> Option<N>) -> Option<T<N>> {
        let data = a.data.iter().map(|&v| f(v)).collect::<Option<Vec<_>>>()?;
        Some(T { rows: a.rows, cols: a.cols, data })
    }
    let go = |e: &E| eval(e, vars);
    match e {
        E::Var(i) => Some(vars[usize::from(*i)].clone()),
        E::Const(c) => Some(T { rows: 1, cols: 1, data: vec![N::cst(*c)?] }),
        E::Neg(a) => map(go(a)?, |v| Some(v.neg())),
        E::Add(a, b) => zip(go(a)?, go(b)?, |p, q| Some(p.add(q))),
        E::Sub(a, b) => zip(go(a)?, go(b)?, |p, q| Some(p.sub(q))),
        E::Mul(a, b) => zip(go(a)?, go(b)?, |p, q| Some(p.mul(q))),
        E::Div(a, b) => zip(go(a)?, go(b)?, |p, q| Some(p.div(q))),
        E::Min(a, b) => zip(go(a)?, go(b)?, N::min),
        E::Max(a, b) => zip(go(a)?, go(b)?, N::max),
        E::Cmp(c, a, b) => zip(go(a)?, go(b)?, |p, q| p.cmp(*c, q)),
        E::Select(c, a, b) => {
            let (c, a, b) = (go(c)?, go(a)?, go(b)?);
            let (rows, cols) = (c.rows.max(a.rows).max(b.rows), c.cols.max(a.cols).max(b.cols));
            let at = |t: &T<N>, i: usize| t.at(i / cols, i % cols);
            let data = (0..rows * cols).map(|i| at(&c, i).select(at(&a, i), at(&b, i))).collect::<Option<Vec<_>>>()?;
            Some(T { rows, cols, data })
        }
        E::Unary(op, a) => map(go(a)?, |v| v.unary(*op)),
        E::Powf(a, p) => map(go(a)?, |v| v.powf(*p)),
        E::Sum { axis, of } => {
            let a = go(of)?;
            // luma's axis 0 is the fastest one: the columns of a row.
            Some(if *axis == 0 {
                let data = (0..a.rows).map(|i| (0..a.cols).fold(N::zero(), |s, j| s.add(a.data[i * a.cols + j]))).collect();
                T { rows: a.rows, cols: 1, data }
            } else {
                let data = (0..a.cols).map(|j| (0..a.rows).fold(N::zero(), |s, i| s.add(a.data[i * a.cols + j]))).collect();
                T { rows: 1, cols: a.cols, data }
            })
        }
        E::Matmul(a, b) => {
            let (a, b) = (go(a)?, go(b)?);
            assert_eq!(a.cols, b.rows, "matmul shapes");
            let mut data = Vec::with_capacity(a.rows * b.cols);
            for i in 0..a.rows {
                for j in 0..b.cols {
                    data.push((0..a.cols).fold(N::zero(), |s, l| s.add(a.data[i * a.cols + l].mul(b.data[l * b.cols + j]))));
                }
            }
            Some(T { rows: a.rows, cols: b.cols, data })
        }
        E::Transpose(a) => {
            let a = go(a)?;
            let data = (0..a.cols).flat_map(|j| (0..a.rows).map(move |i| (i, j))).map(|(i, j)| a.data[i * a.cols + j]).collect();
            Some(T { rows: a.cols, cols: a.rows, data })
        }
        E::Diag(a) => {
            let a = go(a)?;
            let n = a.rows.min(a.cols);
            Some(T { rows: n, cols: 1, data: (0..n).map(|i| a.data[i * a.cols + i]).collect() })
        }
        E::Eye(i) => {
            let n = vars[usize::from(*i)].rows;
            let (one, zero) = (N::cst(1.0)?, N::zero());
            Some(T { rows: n, cols: n, data: (0..n * n).map(|p| if p / n == p % n { one } else { zero }).collect() })
        }
        E::Bcast { of, .. } => go(of),
        E::Trace(a) => {
            let a = go(a)?;
            let n = a.rows.min(a.cols);
            Some(T { rows: 1, cols: 1, data: vec![(0..n).fold(N::zero(), |s, i| s.add(a.data[i * a.cols + i]))] })
        }
        E::Cat { axis, a, b } => {
            let (a, b) = (go(a)?, go(b)?);
            Some(if *axis == 0 {
                assert_eq!(a.rows, b.rows, "cat shapes");
                let data = (0..a.rows).flat_map(|i| a.data[i * a.cols..(i + 1) * a.cols].iter().chain(&b.data[i * b.cols..(i + 1) * b.cols]).copied().collect::<Vec<_>>()).collect();
                T { rows: a.rows, cols: a.cols + b.cols, data }
            } else {
                assert_eq!(a.cols, b.cols, "cat shapes");
                T { rows: a.rows + b.rows, cols: a.cols, data: a.data.iter().chain(&b.data).copied().collect() }
            })
        }
    }
}

// ---------------------------------------------------------------------------------------------
// The luma program of an expression.
// ---------------------------------------------------------------------------------------------

enum Val {
    A(Array<f32>),
    B(Array<bool>),
    K(f32),
}

fn arith(op: char, a: Val, b: Val) -> Val {
    use Val::{A, K};
    match (a, b) {
        (K(p), K(q)) => K(match op {
            '+' => p + q,
            '-' => p - q,
            '*' => p * q,
            _ => p / q,
        }),
        (A(p), A(q)) => A(match op {
            '+' => &p + &q,
            '-' => &p - &q,
            '*' => &p * &q,
            _ => &p / &q,
        }),
        (A(p), K(q)) => A(match op {
            '+' => &p + q,
            '-' => &p - q,
            '*' => &p * q,
            _ => &p / q,
        }),
        (K(p), A(q)) => A(match op {
            '+' => p + &q,
            '-' => p - &q,
            '*' => p * &q,
            _ => p / &q,
        }),
        _ => panic!("arithmetic on a comparison"),
    }
}

fn compare(c: Cmp, a: Val, b: Val) -> Val {
    use Val::{A, B, K};
    let flip = |c: Cmp| match c {
        Cmp::Lt => Cmp::Gt,
        Cmp::Gt => Cmp::Lt,
        Cmp::Le => Cmp::Ge,
        Cmp::Ge => Cmp::Le,
        c => c,
    };
    macro_rules! by {
        ($c:expr, $p:expr, $q:expr) => {
            match $c {
                Cmp::Lt => $p.lt($q),
                Cmp::Le => $p.le($q),
                Cmp::Gt => $p.gt($q),
                Cmp::Ge => $p.ge($q),
                Cmp::Eq => $p.eq($q),
                Cmp::Ne => $p.ne($q),
            }
        };
    }
    match (a, b) {
        (A(p), A(q)) => B(by!(c, p, &q)),
        (A(p), K(q)) => B(by!(c, p, q)),
        (K(p), A(q)) => B(by!(flip(c), q, p)),
        _ => panic!("comparison of constants"),
    }
}

fn to_val(e: &E, vars: &[Array<f32>]) -> Val {
    use Val::{A, B, K};
    let go = |e: &E| to_val(e, vars);
    match e {
        E::Var(i) => A(vars[usize::from(*i)].clone()),
        E::Const(c) => K(*c as f32),
        E::Neg(a) => match go(a) {
            A(a) => A(-&a),
            K(k) => K(-k),
            B(_) => panic!("negated comparison"),
        },
        E::Add(a, b) => arith('+', go(a), go(b)),
        E::Sub(a, b) => arith('-', go(a), go(b)),
        E::Mul(a, b) => arith('*', go(a), go(b)),
        E::Div(a, b) => arith('/', go(a), go(b)),
        E::Min(a, b) => match (go(a), go(b)) {
            (A(p), A(q)) => A(p.minimum(&q)),
            (A(p), K(q)) | (K(q), A(p)) => A(p.minimum(q)),
            _ => panic!("min of constants"),
        },
        E::Max(a, b) => match (go(a), go(b)) {
            (A(p), A(q)) => A(p.maximum(&q)),
            (A(p), K(q)) | (K(q), A(p)) => A(p.maximum(q)),
            _ => panic!("max of constants"),
        },
        E::Cmp(c, a, b) => compare(*c, go(a), go(b)),
        E::Select(c, a, b) => {
            let (A(a), A(b)) = (go(a), go(b)) else { panic!("select of constants") };
            match go(c) {
                B(m) => A(select(&m, &a, &b)),
                A(c) => A(select(&c.ne(0.0), &a, &b)),
                K(_) => panic!("select on a constant"),
            }
        }
        E::Unary(op, a) => {
            let A(a) = go(a) else { panic!("unary of a constant") };
            A(match op {
                UnOp::Abs => a.abs(),
                UnOp::Floor => a.floor(),
                UnOp::Ceil => a.ceil(),
                UnOp::Sign => a.sign(),
                UnOp::Sqrt => a.sqrt(),
                _ => panic!("unary {op:?} has no array form in the harness"),
            })
        }
        E::Powf(a, p) => {
            let A(a) = go(a) else { panic!("powf of a constant") };
            A(a.powf(*p as f32))
        }
        E::Sum { axis, of } => {
            let A(a) = go(of) else { panic!("sum of a constant") };
            A(a.sum_axis(usize::from(*axis)))
        }
        E::Matmul(a, b) => {
            let (A(a), A(b)) = (go(a), go(b)) else { panic!("matmul of constants") };
            A(a.matmul(&b))
        }
        E::Transpose(a) => {
            let A(a) = go(a) else { panic!("transpose of a constant") };
            A(a.transpose())
        }
        E::Diag(a) => {
            let A(a) = go(a) else { panic!("diag of a constant") };
            A(a.diag(0))
        }
        E::Eye(i) => {
            let [_, rows, ..] = vars[usize::from(*i)].shape().dims();
            A(Array::identity_on(vars[usize::from(*i)].context(), [rows, rows]).unwrap())
        }
        E::Bcast { of, .. } => go(of),
        E::Trace(a) => {
            let A(a) = go(a) else { panic!("trace of a constant") };
            A(a.diag(0).sum())
        }
        E::Cat { axis, a, b } => {
            let (A(a), A(b)) = (go(a), go(b)) else { panic!("cat of constants") };
            A(crate::concat(usize::from(*axis), [&a, &b]))
        }
    }
}

/// The program `e` builds over `vars` (the spec's `to_array`; a comparison stays a bool array).
fn to_array(e: &E, vars: &[Array<f32>]) -> Array<f32> {
    match to_val(e, vars) {
        Val::A(a) => a,
        Val::K(c) => Array::full_on(vars[0].context(), vars[0].shape(), c).unwrap(),
        Val::B(_) => panic!("to_array of a comparison: use eval_bits"),
    }
}

fn canonical(v: f32) -> u32 {
    if v.is_nan() { 0x7fc0_0000 } else { v.to_bits() }
}

// ---------------------------------------------------------------------------------------------
// Inputs.
// ---------------------------------------------------------------------------------------------

const SPECIALS: [f32; 22] = [
    0.0,
    -0.0,
    1e-40,
    -1e-40,
    f32::MIN_POSITIVE,
    -f32::MIN_POSITIVE,
    f32::MAX,
    -f32::MAX,
    f32::INFINITY,
    f32::NEG_INFINITY,
    f32::NAN,
    1.0,
    -1.0,
    2.0,
    -2.0,
    3.0,
    0.5,
    -0.5,
    7.0,
    -100.3,
    1.5,
    -3.0,
];

/// Bit patterns of every kind: the special values, any bit pattern, small integers, random.
fn special_heavy(finite: bool) -> BoxedStrategy<f32> {
    let raw = prop_oneof![
        3 => proptest::sample::select(SPECIALS.to_vec()),
        2 => any::<u32>().prop_map(f32::from_bits),
        2 => -8.0f32..8.0f32,
        2 => (-8i32..=8).prop_map(|v| v as f32),
    ];
    if finite { raw.prop_map(|v| if v.is_finite() { v } else { 0.5 }).boxed() } else { raw.boxed() }
}

/// Random finite values away from zero (the points `Reformulating` guards exclude).
fn random_finite() -> BoxedStrategy<f32> {
    (-4.0f32..4.0f32).prop_map(|v| if v.abs() < 0.05 { 1.5 } else { v }).boxed()
}

fn dims(shape: &[usize], len: usize) -> (usize, usize) {
    match shape {
        [0] => (1, len),
        [r, c] => (*r, *c),
        _ => panic!("rule shapes are [0] or [rows, cols]"),
    }
}

/// One set of inputs: a vector of values per variable.
type Batch = Vec<Vec<f32>>;

const LEN: usize = 256;

fn batches(rule: &Rule, kind: Kind) -> Vec<Batch> {
    let vars = rule.lhs.vars().max(rule.rhs.vars());
    assert_eq!(vars, rule.shapes.len(), "{}: shapes for every variable", rule.name);
    let mut runner = TestRunner::deterministic();
    let mut out = Vec::new();
    let elementwise = rule.shapes.iter().all(|s| s == &[0]);
    if kind != Kind::Random && elementwise {
        let pool: Vec<f32> = match (vars, kind) {
            (..=2, Kind::Specials) => SPECIALS.to_vec(),
            (..=2, _) => SPECIALS.iter().copied().filter(|v| v.is_finite()).collect(),
            (_, Kind::Specials) => vec![0.0, -0.0, 1.0, -1.0, f32::NAN, f32::INFINITY, 0.5, 3.0],
            _ => vec![0.0, -0.0, 1.0, -1.0, 0.5, 3.0],
        };
        let n = pool.len().pow(vars as u32);
        let mut batch = vec![Vec::with_capacity(n); vars];
        for i in 0..n {
            let mut rest = i;
            for column in batch.iter_mut() {
                column.push(pool[rest % pool.len()]);
                rest /= pool.len();
            }
        }
        out.push(batch);
    }
    if kind != Kind::Random && !elementwise {
        // A matrix rule: every special value meets every other in some entry, whatever the random batches draw.
        let pool: Vec<f32> = SPECIALS.iter().copied().filter(|v| kind == Kind::Specials || v.is_finite()).collect();
        for shift in [0, 1, 5, 11] {
            out.push(
                rule.shapes
                    .iter()
                    .enumerate()
                    .map(|(v, s)| {
                        let (r, c) = dims(s, LEN);
                        (0..r * c).map(|i| pool[(i * (2 * v + 1) + shift * (v + 1) + v) % pool.len()]).collect()
                    })
                    .collect(),
            );
        }
    }
    let strategy = match kind {
        Kind::Specials => special_heavy(false),
        Kind::Finite => special_heavy(true),
        Kind::Random => random_finite(),
    };
    for _ in 0..4 {
        let batch = rule
            .shapes
            .iter()
            .map(|s| {
                let (r, c) = dims(s, LEN);
                proptest::collection::vec(strategy.clone(), r * c).new_tree(&mut runner).unwrap().current()
            })
            .collect();
        out.push(batch);
    }
    out
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    /// Every `f32`, special values included.
    Specials,
    /// Every finite `f32`.
    Finite,
    /// Finite and away from zero.
    Random,
}

// ---------------------------------------------------------------------------------------------
// The harness.
// ---------------------------------------------------------------------------------------------

pub(super) struct Harness {
    /// Exact-`f32` semantics: no fast math, every class flag off.
    pub(super) strict: Context,
    /// The same with the three class flags set: for the flagged passes of the rule-application check.
    pub(super) fast: Context,
    /// The backend flushes subnormal results of arithmetic to zero (wgpu): a property of the
    /// device, so the bit checks read a subnormal as a zero of its sign.
    pub(super) ftz: bool,
}

/// A subnormal as a signed zero.
fn flush(b: u32) -> u32 {
    if b & 0x7f80_0000 == 0 { b & 0x8000_0000 } else { b }
}

impl Harness {
    pub(super) fn new() -> Harness {
        // Both are exact-`f32` contexts (no TF32, no fast math); the flagged one allows every class of rule.
        let flagged = crate::Numerics { exact_finite: true, reassociate: true, reformulate: true, ..crate::Numerics::strict() };
        let open = |fast: bool| Context::with_options(ContextOptions::from_env().numerics(if fast { flagged } else { crate::Numerics::strict() })).unwrap();
        let strict = open(false);
        strict.jit().set_simplify(false);
        let probe = Array::from_slice_on(&strict, &[1e-40f32], [1, 1]).unwrap();
        let ftz = (&probe * 1.0).to_vec().unwrap()[0].to_bits() & 0x7f80_0000 == 0;
        strict.jit().set_simplify(true);
        Harness { strict, fast: open(true), ftz }
    }

    /// Builds the lhs of `rule` in `ctx` with simplification on and the variables pending (what the application check does), so
    /// that the rules that fire in it are counted by the trace.
    pub(super) fn run_lhs(&self, ctx: &Context, rule: &Rule) {
        let kind = if matches!(rule.gate, Gate::Flagged { .. }) { Kind::Random } else { Kind::Specials };
        let batch = batches(rule, kind).swap_remove(0);
        let _ = self.bits(ctx, &rule.lhs, rule, &batch, true, true);
    }

    /// The values of `e` over `batch` as the bits of its elements (a comparison: 0 or 1), with
    /// the length of the last program. `pending` makes every variable a pending `x * 2`.
    fn bits(&self, ctx: &Context, e: &E, rule: &Rule, batch: &Batch, simplify: bool, pending: bool) -> (Vec<u32>, usize) {
        let as_bool = is_bool(&rule.lhs);
        ctx.jit().set_simplify(simplify);
        let vars: Vec<Array<f32>> = batch
            .iter()
            .zip(rule.shapes)
            .map(|(data, s)| {
                let (r, c) = dims(s, data.len());
                let a = Array::from_slice_on(ctx, data, [c, r]).unwrap();
                if pending { &a * 2.0 } else { a }
            })
            .collect();
        let bits = match to_val(e, &vars) {
            Val::A(a) => a.to_vec().unwrap().into_iter().map(canonical).collect(),
            Val::B(b) => b.to_u32().to_vec().unwrap(),
            Val::K(c) => {
                let n = batch.first().map_or(1, |v| v.len());
                vec![if as_bool { c as u32 } else { canonical(c) }; n]
            }
        };
        let len = ctx.jit().last_program_len();
        ctx.jit().set_simplify(true);
        (bits, len)
    }

    fn reference(&self, e: &E, rule: &Rule, batch: &Batch) -> Vec<f64> {
        let vars: Vec<T<f64>> = batch
            .iter()
            .zip(rule.shapes)
            .map(|(data, s)| {
                let (rows, cols) = dims(s, data.len());
                T { rows, cols, data: data.iter().map(|&v| f64::from(v)).collect() }
            })
            .collect();
        eval(e, &vars).expect("f64 reference").data
    }
}

fn is_bool(e: &E) -> bool {
    matches!(e, E::Cmp(..))
}

/// Check 1: lhs ≡ rhs over ℤ/p at 8 random points.
fn check_zp(rule: &Rule) {
    let mut seed = rule.name.bytes().fold(0xcbf2_9ce4_8422_2325u64, |h, b| (h ^ u64::from(b)).wrapping_mul(0x100_0000_01b3));
    let mut next = || {
        seed = seed.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = seed;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        (z ^ (z >> 31)) % P
    };
    for _ in 0..8 {
        let vars: Vec<T<Zp>> = rule
            .shapes
            .iter()
            .map(|s| {
                let (rows, cols) = dims(s, 4);
                T { rows, cols, data: (0..rows * cols).map(|_| Zp(next())).collect() }
            })
            .collect();
        match (eval(&rule.lhs, &vars), eval(&rule.rhs, &vars)) {
            (Some(l), Some(r)) => {
                // A constant side is one value, broadcast.
                let n = l.data.len().max(r.data.len());
                let at = |t: &T<Zp>, i: usize| t.data[if t.data.len() == 1 { 0 } else { i }];
                assert!((0..n).all(|i| at(&l, i) == at(&r, i)), "{}: lhs and rhs differ over Z/p ({:?} vs {:?})", rule.name, l.data, r.data);
            }
            _ => {
                // A rule outside the field (a comparison, a rounding, a root) has no fingerprint:
                // its class check (exhaustive over special values for the bit-equal classes, the
                // f64 reference for the others) is what stands for it.
                return;
            }
        }
    }
}

/// Check 2: the class, against exact-`f32` semantics, with simplify off.
fn check_class(h: &Harness, rule: &Rule) {
    match rule.class {
        Class::Exact | Class::ExactFinite => {
            let kind = if rule.class == Class::Exact { Kind::Specials } else { Kind::Finite };
            for batch in batches(rule, kind) {
                let (l, _) = h.bits(&h.strict, &rule.lhs, rule, &batch, false, false);
                let (r, _) = h.bits(&h.strict, &rule.rhs, rule, &batch, false, false);
                let same = |a: u32, b: u32| a == b || (h.ftz && flush(a) == flush(b));
                if let Some(i) = (0..l.len()).find(|&i| !same(l[i], r[i])) {
                    let at: Vec<f32> = batch.iter().map(|v| v[if v.len() == l.len() { i } else { 0 }]).collect();
                    panic!("{}: {:?} bit check failed at element {i}, inputs {at:?}: lhs {:#010x}, rhs {:#010x}", rule.name, rule.class, l[i], r[i]);
                }
            }
        }
        Class::Reassociating | Class::Reformulating => {
            let tol = rule.tol.unwrap_or_else(|| panic!("{}: a {:?} rule needs a tolerance", rule.name, rule.class));
            if rule.class == Class::Reformulating {
                assert!(rule.guard.is_some(), "{}: a Reformulating rule needs a guard", rule.name);
            }
            for batch in batches(rule, Kind::Random) {
                let reference = h.reference(&rule.lhs, rule, &batch);
                for (side, e) in [("lhs", &rule.lhs), ("rhs", &rule.rhs)] {
                    let (bits, _) = h.bits(&h.strict, e, rule, &batch, false, false);
                    for (i, (b, want)) in bits.iter().zip(&reference).enumerate() {
                        let got = f64::from(f32::from_bits(*b));
                        // The `+ 1` is the absolute floor: a result near zero of operands of order one.
                        assert!((got - want).abs() <= tol * (want.abs() + 1.0), "{}: error bound failed on the {side} at element {i}: {got} against {want} (tol {tol})", rule.name);
                    }
                }
            }
        }
    }
}

/// Check 3: what `simplify.rs` makes of the lhs.
fn check_application(h: &Harness, rule: &Rule) {
    let (fast, shortens) = match rule.gate {
        Gate::None => return,
        Gate::Exact { shortens } => (false, shortens),
        Gate::Flagged { shortens } => (true, shortens),
    };
    let as_bool = is_bool(&rule.lhs);
    let ctx = if fast { &h.fast } else { &h.strict };
    let kind = if fast { Kind::Random } else { Kind::Specials };
    let batch = batches(rule, kind).swap_remove(0);
    let (plain, plain_len) = h.bits(&h.strict, &rule.lhs, rule, &batch, false, true);
    let (got, len) = h.bits(ctx, &rule.lhs, rule, &batch, true, true);
    if fast {
        // Without the flag the rule does not run.
        let (off, off_len) = h.bits(&h.strict, &rule.lhs, rule, &batch, true, true);
        assert_eq!((off_len, &off), (plain_len, &plain), "{}: rewritten without fast math", rule.name);
        let tol = 2.0 * rule.tol.unwrap_or(1e-6);
        for (i, (g, p)) in got.iter().zip(&plain).enumerate() {
            if as_bool {
                assert_eq!(g, p, "{}: simplified comparison differs at {i}", rule.name);
            } else {
                let (g, p) = (f64::from(f32::from_bits(*g)), f64::from(f32::from_bits(*p)));
                assert!((g - p).abs() <= tol * (p.abs() + 1.0), "{}: simplified value differs at {i}: {g} against {p}", rule.name);
            }
        }
    } else {
        // A reformulated exact-gate rule may differ where the backend flushes a subnormal.
        let flushed = |v: &[u32]| -> Vec<u32> { v.iter().map(|&b| flush(b)).collect() };
        if rule.class == Class::Reformulating || h.ftz {
            assert_eq!(flushed(&got), flushed(&plain), "{}: simplified bits differ", rule.name);
        } else {
            assert_eq!(got, plain, "{}: simplified bits differ", rule.name);
        }
    }
    if shortens {
        assert!(len < plain_len, "{}: program of {len} instructions, {plain_len} unsimplified", rule.name);
    } else {
        assert!(len <= plain_len, "{}: program of {len} instructions, {plain_len} unsimplified", rule.name);
    }
    if !matches!(rule.rhs, E::Const(_)) {
        let (_, rhs_len) = h.bits(&h.strict, &rule.rhs, rule, &batch, false, true);
        // An identity leaves a copy (`Unary(Id)`) of the operand it keeps: one instruction over
        // the bare operand the rhs is.
        let copy = usize::from(matches!(rule.rhs, E::Var(_)));
        assert!(len <= rhs_len + copy, "{}: simplified program of {len} instructions, the rhs builds {rhs_len}", rule.name);
    }
}

pub(super) fn check(h: &Harness, rule: &Rule) {
    check_zp(rule);
    check_class(h, rule);
    check_application(h, rule);
}

// ---------------------------------------------------------------------------------------------
// Typed cases: the rules whose operands are not `f32` (unsigned integers, complex numbers, a
// cast through `f64`, a select with a negated bool mask). They are `Exact` rules checked by bit
// equality with simplify off against on, and by the instruction-count contract.
// ---------------------------------------------------------------------------------------------

pub(super) struct Typed {
    pub(super) name: &'static str,
    /// The rule makes the program strictly shorter; a non-rule keeps its length exactly.
    shortens: bool,
    pub(super) rule: bool,
    pub(super) run: fn(&Context) -> Vec<u32>,
}

const X: [f32; 6] = [0.0, -0.0, f32::NAN, f32::INFINITY, -1.5, 3.0];
const Y: [f32; 6] = [1.0, -2.0, 0.5, 3.0, f32::NEG_INFINITY, -0.0];

fn mask(array: &Array<u32>) -> Vec<u32> {
    array.to_vec().unwrap()
}

/// A pending `u32` array of the interesting values (`+ 5` keeps it pending and wraps one).
fn pending_u32(ctx: &Context) -> Array<u32> {
    let data = [0u32, 1, 2, 7, 4_000_000_000, 3];
    &Array::from_slice_on(ctx, &data, data.len()).unwrap() + 5u32
}

fn pending_f32(ctx: &Context, data: &[f32; 6]) -> Array<f32> {
    &Array::from_slice_on(ctx, data, data.len()).unwrap() * 2.0
}

fn complex_bits(values: Vec<crate::complex::C32>) -> Vec<u32> {
    values.iter().flat_map(|c| [canonical(c.re), canonical(c.im)]).collect()
}

fn pending_complex(ctx: &Context) -> Array<crate::complex::C32> {
    use crate::complex::C32;
    let data = [C32::new(f32::INFINITY, 1.0), C32::new(-0.0, 2.0), C32::new(f32::NAN, -3.0), C32::new(1.5, f32::NEG_INFINITY)];
    Array::from_slice_on(ctx, &data, data.len()).unwrap().conj()
}

#[allow(clippy::erasing_op, clippy::modulo_one, clippy::eq_op)]
pub(super) fn typed_cases() -> Vec<Typed> {
    use crate::complex::C32;
    let t = |name, shortens, run: fn(&Context) -> Vec<u32>| Typed { name, shortens, rule: true, run };
    let not_a_rule = |name, run: fn(&Context) -> Vec<u32>| Typed { name, shortens: false, rule: false, run };
    vec![
        t("u32 add zero", false, |c| mask(&(&pending_u32(c) + 0u32))),
        t("u32 sub zero", false, |c| mask(&(&pending_u32(c) - 0u32))),
        t("u32 mul one", false, |c| mask(&(&pending_u32(c) * 1u32))),
        t("u32 div one", false, |c| mask(&(&pending_u32(c) / 1u32))),
        t("u32 mul zero", true, |c| mask(&(&pending_u32(c) * 0u32))),
        t("u32 rem one", false, |c| mask(&(&pending_u32(c) % 1u32))),
        t("u32 xor zero", false, |c| mask(&(&pending_u32(c) ^ 0u32))),
        t("u32 zero xor", false, |c| mask(&(0u32 ^ &pending_u32(c)))),
        t("u32 or zero", false, |c| mask(&(&pending_u32(c) | 0u32))),
        t("u32 and all ones", false, |c| mask(&(&pending_u32(c) & u32::MAX))),
        t("u32 all ones and", false, |c| mask(&(u32::MAX & &pending_u32(c)))),
        t("u32 and zero", true, |c| mask(&(&pending_u32(c) & 0u32))),
        t("u32 shl zero", false, |c| mask(&(&pending_u32(c) << 0u32))),
        t("u32 shr 32", false, |c| mask(&(&pending_u32(c) >> 32u32))),
        t("u32 rotl 0", false, |c| mask(&pending_u32(c).rotate_left(0u32))),
        t("u32 rotl 64", false, |c| mask(&pending_u32(c).rotate_left(64u32))),
        t("u32 rotr 0", false, |c| mask(&pending_u32(c).rotate_right(0u32))),
        t("u32 shl shl", true, |c| mask(&(&(&pending_u32(c) << 1u32) << 2u32))),
        t("u32 shr shr", true, |c| mask(&(&(&pending_u32(c) >> 1u32) >> 2u32))),
        t("u32 rotl rotl", true, |c| mask(&pending_u32(c).rotate_left(5u32).rotate_left(30u32))),
        t("u32 xor self", true, |c| {
            let u = pending_u32(c);
            mask(&(&u ^ &u))
        }),
        t("u32 and self", false, |c| {
            let u = pending_u32(c);
            mask(&(&u & &u))
        }),
        t("u32 or self", false, |c| {
            let u = pending_u32(c);
            mask(&(&u | &u))
        }),
        t("u32 sub self", true, |c| {
            let u = pending_u32(c);
            mask(&(&u - &u))
        }),
        t("u32 not not", true, |c| mask(&!&!&pending_u32(c))),
        t("u32 le self", true, |c| {
            let u = pending_u32(c);
            u.le(&u).to_u32().to_vec().unwrap()
        }),
        t("u32 ne self", true, |c| {
            let u = pending_u32(c);
            u.ne(&u).to_u32().to_vec().unwrap()
        }),
        // Not rules: the amounts and masks that change the value stay operations.
        not_a_rule("u32 shl one", |c| mask(&(&pending_u32(c) << 1u32))),
        not_a_rule("u32 rotl 33", |c| mask(&pending_u32(c).rotate_left(33u32))),
        not_a_rule("u32 and mask", |c| mask(&(&pending_u32(c) & 0xFF00u32))),
        t("select of a negated mask", true, |c| {
            let (b, d) = (pending_f32(c, &X), pending_f32(c, &Y));
            let m = d.lt(0.5);
            select(&!&m, &b, &d).to_vec().unwrap().into_iter().map(canonical).collect()
        }),
        t("complex make of the parts", true, |c| {
            let w = pending_complex(c);
            complex_bits(Array::from_parts(&w.real(), &w.imag()).to_vec().unwrap())
        }),
        t("complex real of make", true, |c| {
            let (re, im) = (pending_f32(c, &X), pending_f32(c, &Y));
            Array::from_parts(&re, &im).real().to_vec().unwrap().into_iter().map(canonical).collect()
        }),
        t("complex imag of make", true, |c| {
            let (re, im) = (pending_f32(c, &X), pending_f32(c, &Y));
            Array::from_parts(&re, &im).imag().to_vec().unwrap().into_iter().map(canonical).collect()
        }),
        t("complex make of the parts under a conj", true, |c| {
            let w = pending_complex(c);
            complex_bits(Array::from_parts(&w.real(), &w.imag()).conj().to_vec().unwrap())
        }),
        t("complex conj conj", true, |c| complex_bits(pending_complex(c).conj().conj().to_vec().unwrap())),
        not_a_rule("complex times one", |c| complex_bits((&pending_complex(c) * C32::ONE).to_vec().unwrap())),
        t("cast through f64", true, |c| {
            if !c.features().contains(luma_core::Features::SHADER_F64) {
                return Vec::new();
            }
            pending_f32(c, &X).to_f64().to_f32().to_vec().unwrap().into_iter().map(canonical).collect()
        }),
    ]
}

pub(super) fn check_typed(ctx: &Context, case: &Typed) {
    let mut results = Vec::new();
    for simplify in [false, true] {
        ctx.jit().set_simplify(simplify);
        let bits = (case.run)(ctx);
        results.push((bits, ctx.jit().last_program_len()));
    }
    ctx.jit().set_simplify(true);
    let [(plain, plain_len), (got, len)] = [results[0].clone(), results[1].clone()];
    if plain.is_empty() {
        return; // the device lacks the type
    }
    assert_eq!(got, plain, "{}: bits differ", case.name);
    if case.shortens {
        assert!(len < plain_len, "{}: program of {len} instructions, {plain_len} unsimplified", case.name);
    } else if !case.rule {
        assert_eq!(len, plain_len, "{}: not a rule", case.name);
    } else {
        assert!(len <= plain_len, "{}: program of {len} instructions, {plain_len} unsimplified", case.name);
    }
}

// ---------------------------------------------------------------------------------------------
// Tests.
// ---------------------------------------------------------------------------------------------

#[test]
fn every_rule_of_the_table_holds() {
    let h = Harness::new();
    let failures: Vec<String> = rules()
        .iter()
        .filter_map(|rule| {
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| check(&h, rule))).err().map(|e| {
                e.downcast_ref::<String>().cloned().or_else(|| e.downcast_ref::<&str>().map(|s| (*s).to_string())).unwrap_or_default()
            })
        })
        .collect();
    assert!(failures.is_empty(), "{} rules failed:\n{}", failures.len(), failures.join("\n"));
}

#[test]
fn typed_cases_hold() {
    let h = Harness::new();
    for case in typed_cases() {
        check_typed(&h.strict, &case);
    }
}

/// A rule seeded with a flaw, for the tests that prove the harness bites.
fn seeded(name: &'static str, class: Class, lhs: E, rhs: E) -> Rule {
    rule(name, class, lhs, rhs, V1, Some(1e-6), Some("seeded"), Gate::None)
}

fn run_seeded(rule: Rule) {
    check_zp(&rule);
    check_class(&Harness::new(), &rule);
}

#[test]
#[should_panic(expected = "Exact bit check failed")]
fn seeded_add_zero_fails_exact() {
    run_seeded(seeded("x + 0", Class::Exact, add(x(0), k(0.0)), x(0)));
}

#[test]
#[should_panic(expected = "ExactFinite bit check failed")]
fn seeded_add_zero_fails_exact_finite() {
    run_seeded(seeded("x + 0", Class::ExactFinite, add(x(0), k(0.0)), x(0)));
}

#[test]
fn seeded_add_zero_passes_reformulating() {
    run_seeded(seeded("x + 0", Class::Reformulating, add(x(0), k(0.0)), x(0)));
}

#[test]
#[should_panic(expected = "Exact bit check failed")]
fn seeded_mul_zero_fails_exact() {
    run_seeded(seeded("x * 0", Class::Exact, mul(x(0), k(0.0)), k(0.0)));
}

#[test]
#[should_panic(expected = "ExactFinite bit check failed")]
fn seeded_mul_zero_fails_exact_finite() {
    run_seeded(seeded("x * 0", Class::ExactFinite, mul(x(0), k(0.0)), k(0.0)));
}

#[test]
fn seeded_mul_zero_passes_reformulating() {
    run_seeded(seeded("x * 0", Class::Reformulating, mul(x(0), k(0.0)), k(0.0)));
}

#[test]
#[should_panic(expected = "ExactFinite bit check failed")]
fn seeded_div_same_fails_exact_finite() {
    run_seeded(seeded("x / x", Class::ExactFinite, div(x(0), x(0)), k(1.0)));
}

#[test]
fn seeded_div_same_passes_reformulating() {
    run_seeded(seeded("x / x", Class::Reformulating, div(x(0), x(0)), k(1.0)));
}

/// A matrix rule seeded with a flaw: the shapes are the rule's own.
fn seeded_matrix(name: &'static str, class: Class, lhs: E, rhs: E, shapes: &'static [&'static [usize]]) -> Rule {
    rule(name, class, lhs, rhs, shapes, Some(2e-5), Some("seeded"), Gate::None)
}

#[test]
#[should_panic(expected = "Exact bit check failed")]
fn seeded_contract_k1_without_the_plus_zero_fails_on_a_negative_zero() {
    // Without the `+ 0.0` a product of `-0` stays `-0`; the kernels' sum starts at `+0.0`.
    run_seeded(seeded_matrix("k1 without + 0", Class::Exact, E::Matmul(bx(x(0)), bx(x(1))), mul(x(0), x(1)), &[&[3, 1], &[1, 2]]));
}

#[test]
#[should_panic(expected = "Exact bit check failed")]
fn seeded_contract_identity_tagged_exact_fails_on_a_nan() {
    run_seeded(seeded_matrix("a identity exact", Class::Exact, E::Matmul(bx(x(0)), bx(E::Eye(0))), x(0), &[&[3, 3]]));
}

#[test]
#[should_panic(expected = "ExactFinite bit check failed")]
fn seeded_contract_identity_tagged_exact_finite_fails_on_a_negative_zero() {
    run_seeded(seeded_matrix("a identity exact finite", Class::ExactFinite, E::Matmul(bx(x(0)), bx(E::Eye(0))), x(0), &[&[3, 3]]));
}

#[test]
#[should_panic(expected = "Exact bit check failed")]
fn seeded_contract_diag_tagged_exact_fails_on_a_nan() {
    let v = || E::Bcast { axis: 1, of: bx(x(1)) };
    run_seeded(seeded_matrix("a diag exact", Class::Exact, E::Matmul(bx(x(0)), bx(mul(E::Eye(0), v()))), mul(x(0), v()), &[&[3, 3], &[1, 3]]));
}

#[test]
#[should_panic(expected = "differ over Z/p")]
fn seeded_wrong_identity_fails_zp() {
    run_seeded(seeded("x + 1", Class::Reformulating, add(x(0), k(1.0)), x(0)));
}


// ---------------------------------------------------------------------------------------------
// Golden programs: the length of the program and a hash of the bits of every rule's lhs and of
// some compositions, under the default numerics and under fast math, as `integ` (c6140eb) made them:
// the port of `simplify` to class passes must not change a program of any preset.
// ---------------------------------------------------------------------------------------------

fn fnv(bits: &[u32]) -> u64 {
    bits.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, b| (h ^ u64::from(*b)).wrapping_mul(0x100_0000_01b3))
}

type Case = (&'static str, fn(&Array<f32>, &Array<f32>) -> Array<f32>);

/// Compositions that make several rules compete: the order of the passes decides.
#[allow(clippy::eq_op)]
fn golden_cases() -> Vec<Case> {
    vec![
        ("(x+1)+0", |x, _| &(x + 1.0) + 0.0),
        ("(x+1)+(-0)", |x, _| &(x + 1.0) + -0.0),
        ("0+(x+1)", |x, _| 0.0 + &(x + 1.0)),
        ("(x*3)*0", |x, _| &(x * 3.0) * 0.0),
        ("(x*3)*0.5", |x, _| &(x * 3.0) * 0.5),
        ("((x+1)+2)-0", |x, _| &(&(x + 1.0) + 2.0) - 0.0),
        ("((x-1)+3)+0", |x, _| &(&(x - 1.0) + 3.0) + 0.0),
        ("(x-1)-2", |x, _| &(x - 1.0) - 2.0),
        ("(x+5)-2", |x, _| &(x + 5.0) - 2.0),
        ("x/x", |x, _| x / x),
        ("(x*2)/(x*2)", |x, _| &(x * 2.0) / &(x * 2.0)),
        ("(x/3)*2", |x, _| &(x / 3.0) * 2.0),
        ("x/3/2", |x, _| &(x / 3.0) / 2.0),
        ("x/1", |x, _| x / 1.0),
        ("(x/1)+0", |x, _| &(x / 1.0) + 0.0),
        ("x-x", |x, _| x - x),
        ("(x-x)+y", |x, y| &(x - x) + y),
        ("x+(-y)", |x, y| x + &(-y)),
        ("(-x)+y", |x, y| &(-x) + y),
        ("x-(-y)", |x, y| x - &(-y)),
        ("(-x)+(-y)", |x, y| &(-x) + &(-y)),
        ("-(-x)", |x, _| -&(-x)),
        ("x.powf(2)", |x, _| x.powf(2.0)),
        ("x.powf(-1)", |x, _| x.powf(-1.0)),
        ("x.powf(0.5)", |x, _| x.powf(0.5)),
        ("x.powf(1)", |x, _| x.powf(1.0)),
        ("x.powf(0)", |x, _| x.powf(0.0)),
        ("x.powf(3)", |x, _| x.powf(3.0)),
        ("(x.powf(2))*0", |x, _| &x.powf(2.0) * 0.0),
        ("x<=x", |x, _| select(&x.le(x), &(x * 2.0), &(x * 3.0))),
        ("x!=x", |x, _| select(&x.ne(x), &(x * 2.0), &(x * 3.0))),
        ("x<x", |x, _| select(&x.lt(x), &(x * 2.0), &(x * 3.0))),
        ("min(x,x)+0", |x, _| &x.minimum(x) + 0.0),
        ("(x*1)+0", |x, _| &(x * 1.0) + 0.0),
        ("(x*-1)+y", |x, y| &(x * -1.0) + y),
        ("((x*2)*3)*4", |x, _| &(&(x * 2.0) * 3.0) * 4.0),
        ("(x+y)-(x+y)", |x, y| &(x + y) - &(x + y)),
        ("(x*0)+y", |x, y| &(x * 0.0) + y),
        ("(x/x)*y", |x, y| &(x / x) * y),
    ]
}

/// `name|numerics|program length|hash of the bits`, one line per rule lhs (both gates' contexts) and per composition.
pub(super) fn golden_dump() -> Vec<String> {
    let h = Harness::new();
    let mut lines = Vec::new();
    for rule in rules().iter().filter(|r| format!("{:?}", r.gate) != "None") {
        let flagged = format!("{:?}", rule.gate).starts_with('F');
        let batch = batches(rule, if flagged { Kind::Random } else { Kind::Specials }).swap_remove(0);
        for (label, ctx) in [("default", &h.strict), ("fast", &h.fast)] {
            let (bits, len) = h.bits(ctx, &rule.lhs, rule, &batch, true, true);
            lines.push(format!("rule {}|{label}|{len}|{:016x}", rule.name, fnv(&bits)));
        }
    }
    for case in typed_cases().iter().filter(|c| c.rule) {
        for (label, ctx) in [("default", &h.strict), ("fast", &h.fast)] {
            ctx.jit().set_simplify(true);
            let bits = (case.run)(ctx);
            lines.push(format!("typed {}|{label}|{}|{:016x}", case.name, ctx.jit().last_program_len(), fnv(&bits)));
        }
    }
    let open = |numerics: crate::Numerics| Context::with_options(ContextOptions::from_env().numerics(numerics)).unwrap();
    for (label, ctx) in [("default", open(crate::Numerics::default())), ("fast", open(crate::Numerics::fast()))] {
        let x = Array::from_slice_on(&ctx, &[0.5f32, -1.5, 3.0, 0.25, -2.0, 7.0, 0.0, -0.0], 8).unwrap();
        let y = Array::from_slice_on(&ctx, &[1.0f32, 2.0, -3.0, 0.5, 4.0, -7.0, 9.0, 0.125], 8).unwrap();
        for (name, case) in golden_cases() {
            let out = case(&(&x * 2.0), &(&y * 2.0));
            let bits: Vec<u32> = out.to_vec().unwrap().into_iter().map(canonical).collect();
            lines.push(format!("case {name}|{label}|{}|{:016x}", ctx.jit().last_program_len(), fnv(&bits)));
        }
    }
    lines
}

#[test]
fn programs_are_what_integ_made_them() {
    let want: Vec<&str> = include_str!("golden.txt").lines().collect();
    let mut got = golden_dump();
    // The hash of the bits is the CPU backend's (a device's division and `pow` round differently); the program length is every backend's.
    let cpu = Harness::new().strict.is_cpu();
    let key = |line: &str| if cpu { line.to_string() } else { line.rsplit_once('|').map_or(line, |(head, _)| head).to_string() };
    let want: Vec<String> = want.iter().map(|w| key(w)).collect();
    got.iter_mut().for_each(|g| *g = key(g));
    let want: Vec<&str> = want.iter().map(String::as_str).collect();
    assert_eq!(got.len(), want.len(), "the number of cases");
    let wrong: Vec<String> = got.iter().zip(&want).filter(|(g, w)| g != w).map(|(g, w)| format!("{g}   (integ: {w})")).collect();
    assert!(wrong.is_empty(), "{} programs differ from integ:\n{}", wrong.len(), wrong.join("\n"));
}

#[test]
#[should_panic(expected = "ExactFinite bit check failed")]
fn seeded_scale_through_tagged_exact_finite_fails() {
    // The factor applied once after the sum rounds differently from the factor applied to each term.
    run_seeded(seeded_matrix("scale exact finite", Class::ExactFinite, E::Matmul(bx(mul(k(1.1), x(0))), bx(x(1))), mul(k(1.1), E::Matmul(bx(x(0)), bx(x(1)))), &[&[3, 2], &[2, 4]]));
}

#[test]
#[should_panic(expected = "Exact bit check failed")]
fn seeded_factor_invariant_tagged_exact_fails() {
    run_seeded(seeded_matrix("factor exact", Class::Exact, E::Sum { axis: 0, of: bx(mul(E::Bcast { axis: 0, of: bx(x(1)) }, x(0))) }, mul(x(1), E::Sum { axis: 0, of: bx(x(0)) }), &[&[3, 4], &[3, 1]]));
}
