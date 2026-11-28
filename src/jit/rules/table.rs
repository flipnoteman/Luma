//! The rules of `simplify.rs` as data the harness checks (tests only): two expressions equal over the reals, a class, and the
//! pass of `simplify` that implements them.

use super::Class;
use crate::jit::UnOp;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Cmp {
    Lt,
    Le,
    Gt,
    Ge,
    Eq,
    Ne,
}

/// A binder-free rule expression. A comparison yields 1 or 0; `Select(c, a, b)` is `a` where
/// `c` is non-zero. `Sum` keeps the summed axis with extent 1 (as `Array::sum_axis` does; axis 0 is the fastest
/// dimension). A matrix is written `[rows, cols]` in `shapes`; luma's own shape is `[cols, rows]`.
#[derive(Clone, Debug)]
pub(crate) enum E {
    Var(u8),
    Const(f64),
    Neg(Box<E>),
    Add(Box<E>, Box<E>),
    Sub(Box<E>, Box<E>),
    Mul(Box<E>, Box<E>),
    Div(Box<E>, Box<E>),
    Min(Box<E>, Box<E>),
    Max(Box<E>, Box<E>),
    Cmp(Cmp, Box<E>, Box<E>),
    Select(Box<E>, Box<E>, Box<E>),
    /// Transcendental or rounding: excluded from the fingerprint over ℤ/p.
    Unary(UnOp, Box<E>),
    /// `x` to a constant power (a polynomial for a non-negative integer power).
    Powf(Box<E>, f64),
    Sum { axis: u8, of: Box<E> },
    Matmul(Box<E>, Box<E>),
    Transpose(Box<E>),
    Diag(Box<E>),
    /// The identity matrix whose side is the number of rows of variable `u8` (a rule that multiplies by one writes a square
    /// shape for that variable).
    Eye(u8),
    /// `of` broadcast along `axis` (axis 0 is the fastest dimension) to the extent of whatever it meets: the elementwise
    /// operations broadcast an extent-1 axis, so this is the value of `of`, written where a rule says "broadcast".
    Bcast { axis: u8, of: Box<E> },
    /// The sum of the diagonal, as a 1 x 1 matrix (the diagonal added in ascending order from `+0.0`).
    Trace(Box<E>),
    /// `a` and `b` joined along `axis` (axis 0 is the fastest dimension: the columns of a row).
    Cat { axis: u8, a: Box<E>, b: Box<E> },
}

/// Which pass of `simplify` implements a rule, and whether it makes the program strictly shorter.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Gate {
    /// Not a rewrite of `simplify.rs`.
    None,
    /// `simplify::exact`: runs always.
    Exact { shortens: bool },
    /// A flagged pass of `simplify`: runs only where the context's numerics allow the rule's gate.
    Flagged { shortens: bool },
}

pub(crate) struct Rule {
    pub name: &'static str,
    pub class: Class,
    pub lhs: E,
    pub rhs: E,
    /// The shape of each variable; `[0]` is a vector whose length the harness picks.
    pub shapes: &'static [&'static [usize]],
    /// The relative error bound (`Reassociating` and `Reformulating`).
    pub tol: Option<f64>,
    /// The excluded points, in words (required for `Reformulating`; documentation only).
    pub guard: Option<&'static str>,
    pub gate: Gate,
}

pub(super) const V1: &[&[usize]] = &[&[0]];
pub(super) const V2: &[&[usize]] = &[&[0], &[0]];

pub(super) fn bx(e: E) -> Box<E> {
    Box::new(e)
}
pub(super) fn x(i: u8) -> E {
    E::Var(i)
}
pub(super) fn k(c: f64) -> E {
    E::Const(c)
}
pub(super) fn neg(a: E) -> E {
    E::Neg(bx(a))
}
pub(super) fn add(a: E, b: E) -> E {
    E::Add(bx(a), bx(b))
}
pub(super) fn sub(a: E, b: E) -> E {
    E::Sub(bx(a), bx(b))
}
pub(super) fn mul(a: E, b: E) -> E {
    E::Mul(bx(a), bx(b))
}
pub(super) fn div(a: E, b: E) -> E {
    E::Div(bx(a), bx(b))
}
pub(super) fn un(op: UnOp, a: E) -> E {
    E::Unary(op, bx(a))
}
pub(super) fn cmp(c: Cmp, a: E, b: E) -> E {
    E::Cmp(c, bx(a), bx(b))
}
pub(super) fn pw(a: E, p: f64) -> E {
    E::Powf(bx(a), p)
}

impl E {
    /// The number of variables (the largest index plus one).
    pub(crate) fn vars(&self) -> usize {
        match self {
            E::Var(i) => usize::from(*i) + 1,
            E::Const(_) => 0,
            E::Neg(a) | E::Unary(_, a) | E::Powf(a, _) | E::Transpose(a) | E::Diag(a) | E::Trace(a) | E::Bcast { of: a, .. } | E::Sum { of: a, .. } => a.vars(),
            E::Eye(i) => usize::from(*i) + 1,
            E::Cat { a, b, .. } => a.vars().max(b.vars()),
            E::Add(a, b) | E::Sub(a, b) | E::Mul(a, b) | E::Div(a, b) | E::Min(a, b) | E::Max(a, b) | E::Cmp(_, a, b) | E::Matmul(a, b) => a.vars().max(b.vars()),
            E::Select(a, b, c) => a.vars().max(b.vars()).max(c.vars()),
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn rule(
    name: &'static str,
    class: Class,
    lhs: E,
    rhs: E,
    shapes: &'static [&'static [usize]],
    tol: Option<f64>,
    guard: Option<&'static str>,
    gate: Gate,
) -> Rule {
    Rule { name, class, lhs, rhs, shapes, tol, guard, gate }
}

/// Today's rules of `simplify.rs`, then a few matrix rules that exercise the structured forms.
pub(crate) fn rules() -> Vec<Rule> {
    use Class::{Exact, ExactFinite, Reassociating, Reformulating};
    let ex = |shortens| Gate::Exact { shortens };
    let fa = |shortens| Gate::Flagged { shortens };
    let mm: &'static [&'static [usize]] = &[&[2, 3], &[3, 2]];
    let sq: &'static [&'static [usize]] = &[&[3, 3]];
    let pair: &'static [&'static [usize]] = &[&[3, 2], &[3, 2]];
    vec![
        // `simplify::exact`
        rule("mul by one", Exact, mul(x(0), k(1.0)), x(0), V1, None, None, ex(false)),
        rule("one by mul", Exact, mul(k(1.0), x(0)), x(0), V1, None, None, ex(false)),
        rule("mul by minus one", Exact, mul(x(0), k(-1.0)), neg(x(0)), V1, None, None, ex(false)),
        rule("minus one by mul", Exact, mul(k(-1.0), x(0)), neg(x(0)), V1, None, None, ex(false)),
        rule("div by one", Reformulating, div(x(0), k(1.0)), x(0), V1, Some(1e-6), Some("x subnormal (the backend division flushes it to zero; x * 1 keeps it)"), ex(false)),
        rule("sub plus zero", Exact, sub(x(0), k(0.0)), x(0), V1, None, None, ex(false)),
        rule("add minus zero", Exact, add(x(0), k(-0.0)), x(0), V1, None, None, ex(false)),
        rule("minus zero add", Exact, add(k(-0.0), x(0)), x(0), V1, None, None, ex(false)),
        rule("neg neg", Exact, neg(neg(x(0))), x(0), V1, None, None, ex(true)),
        rule("abs abs", Exact, un(UnOp::Abs, un(UnOp::Abs, x(0))), un(UnOp::Abs, x(0)), V1, None, None, ex(true)),
        rule("abs neg", Exact, un(UnOp::Abs, neg(x(0))), un(UnOp::Abs, x(0)), V1, None, None, ex(true)),
        rule("floor floor", Exact, un(UnOp::Floor, un(UnOp::Floor, x(0))), un(UnOp::Floor, x(0)), V1, None, None, ex(true)),
        rule("floor ceil", Exact, un(UnOp::Floor, un(UnOp::Ceil, x(0))), un(UnOp::Ceil, x(0)), V1, None, None, ex(true)),
        rule("ceil ceil", Exact, un(UnOp::Ceil, un(UnOp::Ceil, x(0))), un(UnOp::Ceil, x(0)), V1, None, None, ex(true)),
        rule("ceil floor", Exact, un(UnOp::Ceil, un(UnOp::Floor, x(0))), un(UnOp::Floor, x(0)), V1, None, None, ex(true)),
        rule("sign sign", Exact, un(UnOp::Sign, un(UnOp::Sign, x(0))), un(UnOp::Sign, x(0)), V1, None, None, ex(true)),
        rule("min same", Exact, E::Min(bx(x(0)), bx(x(0))), x(0), V1, None, None, ex(false)),
        rule("max same", Exact, E::Max(bx(x(0)), bx(x(0))), x(0), V1, None, None, ex(false)),
        rule("select same", Exact, E::Select(bx(cmp(Cmp::Lt, x(1), k(0.5))), bx(x(0)), bx(x(0))), x(0), V2, None, None, ex(false)),
        rule("lt same", Exact, cmp(Cmp::Lt, x(0), x(0)), k(0.0), V1, None, None, ex(true)),
        rule("gt same", Exact, cmp(Cmp::Gt, x(0), x(0)), k(0.0), V1, None, None, ex(true)),
        // `simplify::fast`
        rule("add zero", Reformulating, add(x(0), k(0.0)), x(0), V1, Some(1e-6), Some("x = -0 (-0 + 0 is +0)"), fa(false)),
        rule("zero add", Reformulating, add(k(0.0), x(0)), x(0), V1, Some(1e-6), Some("x = -0 (0 + -0 is +0)"), fa(false)),
        rule("mul zero", Reformulating, mul(x(0), k(0.0)), k(0.0), V1, Some(1e-6), Some("x infinite or NaN (NaN), x negative (-0)"), fa(true)),
        rule("zero mul", Reformulating, mul(k(0.0), x(0)), k(0.0), V1, Some(1e-6), Some("x infinite or NaN (NaN), x negative (-0)"), fa(true)),
        rule("sub same", ExactFinite, sub(x(0), x(0)), k(0.0), V1, None, None, fa(true)),
        rule("add neg", Exact, add(x(0), neg(x(1))), sub(x(0), x(1)), V2, None, None, fa(true)),
        rule("neg add", Exact, add(neg(x(0)), x(1)), sub(x(1), x(0)), V2, None, None, fa(true)),
        rule("sub neg", Exact, sub(x(0), neg(x(1))), add(x(0), x(1)), V2, None, None, fa(true)),
        rule("div same", Reformulating, div(x(0), x(0)), k(1.0), V1, Some(1e-6), Some("x = 0 (NaN), x infinite or NaN (NaN)"), fa(true)),
        rule("div by constant", Reassociating, div(x(0), k(3.0)), mul(x(0), div(k(1.0), k(3.0))), V1, Some(4e-7), None, fa(false)),
        rule("le same", ExactFinite, cmp(Cmp::Le, x(0), x(0)), k(1.0), V1, None, None, fa(true)),
        rule("ge same", ExactFinite, cmp(Cmp::Ge, x(0), x(0)), k(1.0), V1, None, None, fa(true)),
        rule("eq same", ExactFinite, cmp(Cmp::Eq, x(0), x(0)), k(1.0), V1, None, None, fa(true)),
        rule("ne same", ExactFinite, cmp(Cmp::Ne, x(0), x(0)), k(0.0), V1, None, None, fa(true)),
        rule("mul chain", Reassociating, mul(mul(x(0), k(3.0)), k(0.5)), mul(x(0), k(1.5)), V1, Some(2e-6), None, fa(true)),
        rule("add chain", Reassociating, add(add(x(0), k(1.0)), k(2.0)), add(x(0), k(3.0)), V1, Some(2e-6), None, fa(true)),
        rule("sub chain", Reassociating, sub(sub(x(0), k(1.0)), k(2.0)), sub(x(0), k(3.0)), V1, Some(2e-6), None, fa(true)),
        rule("sub then add", Reassociating, add(sub(x(0), k(1.0)), k(3.0)), add(x(0), k(2.0)), V1, Some(2e-6), None, fa(true)),
        rule("add then sub", Reassociating, sub(add(x(0), k(5.0)), k(2.0)), add(x(0), k(3.0)), V1, Some(2e-6), None, fa(true)),
        rule("powf zero", Reformulating, pw(x(0), 0.0), k(1.0), V1, Some(1e-5), Some("x <= 0 (the device pow of a negative base)"), fa(false)),
        rule("powf one", Reassociating, pw(x(0), 1.0), x(0), V1, Some(1e-5), None, fa(false)),
        rule("powf two", Reassociating, pw(x(0), 2.0), mul(x(0), x(0)), V1, Some(1e-5), None, fa(false)),
        rule("powf minus one", Reassociating, pw(x(0), -1.0), div(k(1.0), x(0)), V1, Some(1e-5), None, fa(false)),
        rule("powf half", Reassociating, pw(un(UnOp::Abs, x(0)), 0.5), un(UnOp::Sqrt, un(UnOp::Abs, x(0))), V1, Some(1e-5), None, fa(false)),
        // Not rewrites of `simplify.rs`: the matrix forms wave 2 builds on.
        rule("transpose of a product", Reassociating, E::Transpose(bx(E::Matmul(bx(x(0)), bx(x(1))))), E::Matmul(bx(E::Transpose(bx(x(1)))), bx(E::Transpose(bx(x(0))))), mm, Some(2e-5), None, Gate::None),
        rule("sum of a sum", Reassociating, E::Sum { axis: 0, of: bx(add(x(0), x(1))) }, add(E::Sum { axis: 0, of: bx(x(0)) }, E::Sum { axis: 0, of: bx(x(1)) }), pair, Some(2e-5), None, Gate::None),
        rule("diag of a transpose", Exact, E::Diag(bx(E::Transpose(bx(x(0))))), E::Diag(bx(x(0))), sq, None, None, Gate::None),
    ]
    .into_iter()
    .chain(index_rules())
    .chain(reform_rules())
    .collect()
}

pub(super) fn matmul(a: E, b: E) -> E {
    E::Matmul(bx(a), bx(b))
}

pub(super) fn tr(a: E) -> E {
    E::Transpose(bx(a))
}

/// The rules of W3-R4R5 whose lhs and rhs are programs of matrices (`contract_view` ... `contract_diag`). The rules of tier 2
/// (sum of a product, invariant factors, chain order) follow in [`reform_rules`].
pub(super) fn index_rules() -> Vec<Rule> {
    use Class::{Exact, Reassociating, Reformulating};
    let none = Gate::None;
    let sq: &'static [&'static [usize]] = &[&[3, 3]];
    vec![
        // 1: `aᵀ·b` read through the view, or the product of the swapped operands transposed: every entry adds the same products in
        // the same order of `k`.
        rule("contract_view", Exact, matmul(tr(x(0)), x(1)), tr(matmul(tr(x(1)), x(0))), &[&[3, 2], &[3, 4]], None, None, none),
        // 3: a product of one term: `a·b + 0.0` (the sum starts at `+0.0`, so a `-0` product is `+0`).
        rule("contract_k1", Exact, matmul(x(0), x(1)), add(mul(x(0), x(1)), k(0.0)), &[&[3, 1], &[1, 2]], None, None, none),
        // 6b: the diagonal of a product is one sum of products per element.
        rule("diag_of_product", Exact, E::Diag(bx(matmul(x(0), x(1)))), E::Sum { axis: 0, of: bx(mul(x(0), tr(x(1)))) }, &[&[3, 4], &[4, 3]], None, None, none),
        // 11: `aᵀa` is its own transpose, bit for bit.
        rule("gram_symmetric", Exact, matmul(tr(x(0)), x(0)), tr(matmul(tr(x(0)), x(0))), &[&[4, 3]], None, None, none),
        // 12: a scalar factor of an operand scales the product (the rounding differs: the scale is applied once, after the sum).
        rule("scale_through", Reassociating, matmul(mul(k(1.5), x(0)), x(1)), mul(k(1.5), matmul(x(0), x(1))), &[&[3, 2], &[2, 4]], Some(2e-5), None, none),
        // 5: `a·I`.
        rule("contract_identity", Reformulating, matmul(x(0), E::Eye(0)), x(0), sq, Some(2e-5), Some("a NaN or infinity in a spreads over a row of the product (x * 0); -0 becomes +0"), none),
        // 6a: `a·diag(v)` with `v` a row vector of the diagonal.
        rule(
            "contract_diag",
            Reformulating,
            matmul(x(0), mul(E::Eye(0), E::Bcast { axis: 1, of: bx(x(1)) })),
            mul(x(0), E::Bcast { axis: 1, of: bx(x(1)) }),
            &[&[3, 3], &[1, 3]],
            Some(2e-5),
            Some("a NaN or infinity in a spreads over a row of the product (x * 0); -0 becomes +0"),
            none,
        ),
    ]
}

/// The rules of tier 2 of W3-R4R5 as expressions: `sum_of_product` and `chain_order`.
pub(super) fn reform_rules() -> Vec<Rule> {
    use Class::{Exact, Reassociating, Reformulating};
    let sum = |axis, of| E::Sum { axis, of: bx(of) };
    vec![
        // 7: the sum of all the entries of a product is the product of the column sums of `a` and the row sums of `b`.
        rule(
            "sum_of_product",
            Reformulating,
            sum(0, sum(1, matmul(x(0), x(1)))),
            matmul(sum(1, x(0)), sum(0, x(1))),
            &[&[3, 4], &[4, 5]],
            Some(1e-4),
            Some("an infinity or NaN in a or b reaches the result through fewer terms (x * 0); -0 sums differ"),
            Gate::None,
        ),
        // 8: a factor that does not depend on the summed index leaves the sum: `Σ_k s(i)·x(i,k)` is `s(i)·Σ_k x(i,k)`.
        rule(
            "factor_invariant",
            Reformulating,
            sum(0, mul(E::Bcast { axis: 0, of: bx(x(1)) }, x(0))),
            mul(x(1), sum(0, x(0))),
            &[&[3, 4], &[3, 1]],
            Some(2e-5),
            Some("an infinity or NaN in a sum that s multiplies at 0 (s * inf, 0 * NaN), the order of the roundings; -0 sums differ"),
            Gate::None,
        ),
        // 4: one product over the two right operands side by side gives the two products (every output sums the same terms in the
        // same order of `k` when the plan keeps `kc` and does not split K).
        rule("shared_operand", Exact, E::Cat { axis: 0, a: bx(matmul(x(0), x(1))), b: bx(matmul(x(0), x(2))) }, matmul(x(0), E::Cat { axis: 0, a: bx(x(1)), b: bx(x(2)) }), &[&[3, 4], &[4, 2], &[4, 5]], None, None, Gate::None),
        // 9: `(a·b)·v` is `a·(b·v)`; the rounding differs.
        rule("chain_order", Reassociating, matmul(matmul(x(0), x(1)), x(2)), matmul(x(0), matmul(x(1), x(2))), &[&[3, 4], &[4, 5], &[5, 1]], Some(1e-4), None, Gate::None),
    ]
}
