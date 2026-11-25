//! Algebraic simplification of an operation about to be created (see [`rewrite`]).
//!
//! The rules in [`exact`] never change a result bit (up to the payload of a NaN). The rules in
//! [`fast`] assume finite values and ignore the sign of zero and rounding, so they only run with
//! [`Reduced::FAST_MATH`](crate::Reduced::FAST_MATH).
//!
//! A rewrite never returns an existing node: an identity becomes a `Unary(Id)` of the surviving
//! operand and a constant a `Unary(Id)` of a scalar, because a node an array holds must not
//! become a second array's node (see [`Held`](super::Held)).
//!
//! A `Unary(Id)` is never folded into the operation it reads (`Id(Id(x))` stays): the stores of
//! `assign` and the readers of reductions are `Id` nodes whose edge to the expression they copy
//! is what the hazard analysis walks, and a copy that read `x` directly would not see that the
//! expression below it has to be evaluated before the write.

use super::{Args, BinOp, CmpOp, ElemTy, Node, Op, OpKind, Part, Scalar, Src, UnOp};

/// The operation and operands to create instead of `op` on `args`, if a rule applies.
#[inline(always)]
pub(super) fn rewrite(op: &Op, args: &Args, fast: bool) -> Option<(Op, Args)> {
    exact(op, args).or_else(|| if fast { self::fast(op, args) } else { None })
}

fn is_float(ty: ElemTy) -> bool {
    matches!(ty, ElemTy::F32 | ElemTy::F64 | ElemTy::Df32 | ElemTy::C32 | ElemTy::F16 | ElemTy::BF16)
}

/// The bit pattern of the constant `value` as a scalar of type `ty` (`Df32`: `hi` is `value`,
/// `lo` is 0; `C32`: the real part is `value`, the imaginary part 0).
fn const_bits(ty: ElemTy, value: f64) -> u64 {
    match ty {
        ElemTy::F32 | ElemTy::Df32 | ElemTy::C32 => u64::from((value as f32).to_bits()),
        // The register of a half holds the value rounded to the type, widened to `f32`.
        ElemTy::F16 => u64::from(half::f16::from_f64(value).to_f32().to_bits()),
        ElemTy::BF16 => u64::from(half::bf16::from_f64(value).to_f32().to_bits()),
        ElemTy::F64 => value.to_bits(),
        ElemTy::U32 | ElemTy::Bool | ElemTy::U8 | ElemTy::U16 => u64::from(value as u32),
        ElemTy::I32 | ElemTy::I8 | ElemTy::I16 => u64::from(value as i32 as u32),
    }
}

/// Whether a scalar operand of type `ty` with these bits can make a rule of [`exact`] fire (every
/// rule that looks at the *value* of a scalar looks for one of these). A step that makes the same
/// operations on scalars that are not sensitive gets the same rewrites (none, or those the
/// operands decide), which is what trace following relies on to skip [`rewrite`] (see
/// `LUMA_TRACE_FOLLOW`). The fast rules read any value: they are not covered, following always runs
/// [`rewrite`] under fast math.
pub(super) fn sensitive(ty: ElemTy, bits: u64) -> bool {
    let bits = bits & 0xffff_ffff;
    let shifts = matches!(ty, ElemTy::U32 | ElemTy::I32 | ElemTy::U8 | ElemTy::U16 | ElemTy::I8 | ElemTy::I16) && bits & 31 == 0;
    shifts || [0.0, 1.0, -1.0, -0.0, f64::from(u32::MAX)].into_iter().any(|value| bits == const_bits(ty, value) & 0xffff_ffff)
}

/// Whether `s` is the constant `value` of type `ty`, by bit pattern (so `-0.0` is not `0.0`).
fn is_value(s: Option<Scalar>, ty: ElemTy, value: f64) -> bool {
    s.is_some_and(|s| {
        let bits = if s.ty.slots() == 1 { s.bits & 0xffff_ffff } else { s.bits };
        s.ty == ty && bits == const_bits(ty, value)
    })
}

fn scalar(src: &Src) -> Option<Scalar> {
    match src {
        Src::Scalar(s) => Some(*s),
        _ => None,
    }
}

/// Both operands are the same array: the same node or the same buffer.
fn same(a: &Src, b: &Src) -> bool {
    match (a, b) {
        (Src::Node(a), Src::Node(b)) => std::sync::Arc::ptr_eq(a, b),
        (Src::Buffer(a), Src::Buffer(b)) => std::sync::Arc::ptr_eq(a, b),
        _ => false,
    }
}

/// The operation and operands of a pending, not yet evaluated node operand whose operation
/// `wanted` accepts (the operands are only copied then, which takes the node's lock).
fn inner(src: &Src, wanted: fn(Op) -> bool) -> Option<(Op, Args)> {
    let Src::Node(node) = src else { return None };
    if !wanted(*node.op()) || !node.is_pending() || is_looped(node) {
        return None;
    }
    node.operands().map(|args| (*node.op(), args))
}

fn is_looped(node: &Node) -> bool {
    node.looped.get().is_some()
}

fn args_of(a: Src, b: Src, c: Src) -> Args {
    [a, b, c]
}

/// `Id(x)`: the operation yields its operand.
fn identity(op: &Op, x: &Src) -> Option<(Op, Args)> {
    Some((Op::new(OpKind::Unary(UnOp::Id), op.out_ty()), args_of(x.clone(), Src::unused(), Src::unused())))
}

/// `Id` of the constant `value` of the operation's result type.
fn constant(op: &Op, value: f64) -> Option<(Op, Args)> {
    let ty = op.out_ty();
    let bits = const_bits(ty, value);
    Some((
        Op::new(OpKind::Unary(UnOp::Id), ty),
        args_of(Src::Scalar(Scalar { ty, bits }), Src::unused(), Src::unused()),
    ))
}

fn unary(kind: UnOp, ty: ElemTy, x: &Src) -> Option<(Op, Args)> {
    Some((Op::new(OpKind::Unary(kind), ty), args_of(x.clone(), Src::unused(), Src::unused())))
}

fn binary(kind: BinOp, ty: ElemTy, x: &Src, y: Src) -> Option<(Op, Args)> {
    Some((Op::new(OpKind::Binary(kind), ty), args_of(x.clone(), y, Src::unused())))
}

/// The rewrites that keep every result bit.
fn exact(op: &Op, args: &Args) -> Option<(Op, Args)> {
    let [a, b, c] = args;
    let ty = op.ty;
    match op.kind {
        // Not one identity of a real number holds bit for bit for a complex one: `z * 1` has
        // `inf * 0` in it, and `z + (-0)` turns a `-0` imaginary part into `+0`.
        OpKind::Binary(_) if ty == ElemTy::C32 => None,
        // arm: complex-part-of-make
        OpKind::Part(Part::Real | Part::Imag) => {
            let (_, iargs) = inner(a, |inside| inside.kind == OpKind::Make)?;
            identity(op, &iargs[usize::from(op.kind == OpKind::Part(Part::Imag))])
        }
        // arm: complex-make-of-parts
        OpKind::Make => {
            let (_, real) = inner(a, |inside| inside.kind == OpKind::Part(Part::Real))?;
            let (_, imag) = inner(b, |inside| inside.kind == OpKind::Part(Part::Imag))?;
            if same(&real[0], &imag[0]) { identity(op, &real[0]) } else { None }
        }
        OpKind::Binary(bin) => {
            let (sa, sb) = (scalar(a), scalar(b));
            let float = is_float(ty);
            let u32s = matches!(ty, ElemTy::U32 | ElemTy::I32);
            // All bits set: `-1` for `i32` (a larger constant saturates to `i32::MAX`).
            let all_ones = if ty == ElemTy::I32 { -1.0 } else { f64::from(u32::MAX) };
            match bin {
                // arm: mul-one, mul-minus-one, mul-zero-int
                BinOp::Mul => {
                    if is_value(sb, ty, 1.0) {
                        identity(op, a)
                    } else if is_value(sa, ty, 1.0) {
                        identity(op, b)
                    } else if (float || ty == ElemTy::I32) && is_value(sb, ty, -1.0) {
                        unary(UnOp::Neg, ty, a)
                    } else if (float || ty == ElemTy::I32) && is_value(sa, ty, -1.0) {
                        unary(UnOp::Neg, ty, b)
                    } else if u32s && (is_value(sa, ty, 0.0) || is_value(sb, ty, 0.0)) {
                        constant(op, 0.0)
                    } else {
                        None
                    }
                }
                // arm: div-one
                BinOp::Div if is_value(sb, ty, 1.0) => identity(op, a),
                // arm: sub-zero
                BinOp::Sub if is_value(sb, ty, 0.0) => identity(op, a),
                // arm: sub-same-int
                BinOp::Sub if u32s && same(a, b) => constant(op, 0.0),
                // arm: add-neutral
                BinOp::Add => {
                    if float && is_value(sb, ty, -0.0) || u32s && is_value(sb, ty, 0.0) {
                        identity(op, a)
                    } else if float && is_value(sa, ty, -0.0) || u32s && is_value(sa, ty, 0.0) {
                        identity(op, b)
                    } else {
                        None
                    }
                }
                // arm: min-max-same
                BinOp::Min | BinOp::Max if same(a, b) => identity(op, a),
                // arm: rem-one-int
                BinOp::Rem if u32s && is_value(sb, ty, 1.0) => constant(op, 0.0),
                // arm: xor-or-zero
                BinOp::BitXor | BinOp::BitOr if is_value(sb, ty, 0.0) => identity(op, a),
                BinOp::BitXor | BinOp::BitOr if is_value(sa, ty, 0.0) => identity(op, b),
                // arm: xor-same
                BinOp::BitXor if same(a, b) => constant(op, 0.0),
                // arm: or-and-same
                BinOp::BitOr | BinOp::BitAnd if same(a, b) => identity(op, a),
                // arm: and-zero
                BinOp::BitAnd if is_value(sa, ty, 0.0) || is_value(sb, ty, 0.0) => constant(op, 0.0),
                // arm: and-all-ones
                BinOp::BitAnd if is_value(sb, ty, all_ones) => identity(op, a),
                BinOp::BitAnd if is_value(sa, ty, all_ones) => identity(op, b),
                // arm: binary-shift-multiple-of-32
                // The amount is taken modulo 32: a multiple of 32 moves nothing.
                BinOp::Shl | BinOp::Shr | BinOp::Rotl if sb.is_some_and(|s| s.ty == ty && s.bits & 31 == 0) => identity(op, a),
                _ => None,
            }
        }
        // arm: unary-shift-zero
        OpKind::Unary(UnOp::Shl(0) | UnOp::Shr(0) | UnOp::Rotl(0)) => identity(op, a),
        // arm: involution, abs-chain, rounding-chain, shift-chain, sign-chain
        OpKind::Unary(un) => {
            if !matches!(
                un,
                UnOp::Neg | UnOp::Conj | UnOp::Abs | UnOp::Floor | UnOp::Ceil | UnOp::Trunc | UnOp::Round | UnOp::Sign | UnOp::BitNot | UnOp::Shl(_) | UnOp::Shr(_) | UnOp::Rotl(_)
            ) {
                return None;
            }
            let (iop, iargs) = inner(a, |inside| matches!(inside.kind, OpKind::Unary(_)))?;
            let OpKind::Unary(inner_un) = iop.kind else { return None };
            match (un, inner_un) {
                (UnOp::Neg, UnOp::Neg) | (UnOp::Conj, UnOp::Conj) | (UnOp::BitNot, UnOp::BitNot) => identity(op, &iargs[0]),
                (UnOp::Abs, UnOp::Abs | UnOp::Neg) => unary(UnOp::Abs, ty, &iargs[0]),
                (UnOp::Floor | UnOp::Ceil | UnOp::Trunc | UnOp::Round, UnOp::Floor | UnOp::Ceil | UnOp::Trunc | UnOp::Round) => {
                    unary(inner_un, ty, &iargs[0])
                }
                (UnOp::Shl(x), UnOp::Shl(y)) if u32::from(x) + u32::from(y) < 32 => unary(UnOp::Shl(x + y), ty, &iargs[0]),
                (UnOp::Shr(x), UnOp::Shr(y)) if u32::from(x) + u32::from(y) < 32 => unary(UnOp::Shr(x + y), ty, &iargs[0]),
                (UnOp::Rotl(x), UnOp::Rotl(y)) => unary(UnOp::Rotl(((u32::from(x) + u32::from(y)) & 31) as u8), ty, &iargs[0]),
                (UnOp::Sign, UnOp::Sign) => unary(UnOp::Sign, ty, &iargs[0]),
                _ => None,
            }
        }
        // arm: select-scalar, select-not, select-same
        OpKind::Select => {
            if let Some(m) = scalar(a) {
                identity(op, if m.bits & 0xffff_ffff != 0 { b } else { c })
            } else if let Some((_, iargs)) = inner(a, |inside| inside.ty == ElemTy::Bool && inside.kind == OpKind::Unary(UnOp::BitNot)) {
                Some((*op, args_of(iargs[0].clone(), c.clone(), b.clone())))
            } else if same(b, c) {
                identity(op, b)
            } else {
                None
            }
        }
        // arm: compare-same
        OpKind::Compare(cmp) if same(a, b) && cmp.exists_for(ty) => match cmp {
            CmpOp::Lt | CmpOp::Gt => constant(op, 0.0),
            CmpOp::Le | CmpOp::Ge | CmpOp::Eq if !is_float(ty) => constant(op, 1.0),
            CmpOp::Ne if !is_float(ty) => constant(op, 0.0),
            _ => None,
        },
        // arm: cast-f64-round-trip
        OpKind::Cast(ElemTy::F32) if ty == ElemTy::F64 => {
            let (iop, iargs) = inner(a, |inside| matches!(inside.kind, OpKind::Cast(_)))?;
            match iop.kind {
                OpKind::Cast(through) if through == ty && iop.ty == ElemTy::F32 => {
                    identity(op, &iargs[0])
                }
                _ => None,
            }
        }
        _ => None,
    }
}

/// The value of a scalar of `F32` or `F64` type.
fn value_of(s: Scalar) -> Option<f64> {
    match s.ty {
        ElemTy::F32 => Some(f64::from(f32::from_bits(s.bits as u32))),
        ElemTy::F64 => Some(f64::from_bits(s.bits)),
        _ => None,
    }
}

/// A scalar of type `ty` (`F32` or `F64`) holding `value`, rounded to it.
fn scalar_of(ty: ElemTy, value: f64) -> Src {
    Src::Scalar(Scalar { ty, bits: const_bits(ty, value) })
}

/// Whether `value` stays finite (and non-zero, if `nonzero`) when rounded to `ty`.
fn survives(ty: ElemTy, value: f64, nonzero: bool) -> bool {
    let rounded = if ty == ElemTy::F32 { f64::from(value as f32) } else { value };
    rounded.is_finite() && (!nonzero || rounded != 0.0)
}

/// The rewrites that may change rounding, the sign of zero or the handling of NaN and infinity.
fn fast(op: &Op, args: &Args) -> Option<(Op, Args)> {
    let [a, b, _] = args;
    let ty = op.ty;
    let float = is_float(ty);
    match op.kind {
        OpKind::Binary(bin) => {
            let (sa, sb) = (scalar(a), scalar(b));
            match bin {
                // arm: fast-add-zero
                BinOp::Add if float && is_value(sb, ty, 0.0) => identity(op, a),
                BinOp::Add if float && is_value(sa, ty, 0.0) => identity(op, b),
                // arm: fast-mul-zero
                BinOp::Mul if float && (is_value(sa, ty, 0.0) || is_value(sb, ty, 0.0)) => constant(op, 0.0),
                // arm: fast-sub-same
                BinOp::Sub if float && same(a, b) => constant(op, 0.0),
                // arm: fast-add-neg, fast-sub-neg
                BinOp::Add if float && inner(b, |inside| inside.kind == OpKind::Unary(UnOp::Neg)).is_some() => {
                    let (_, iargs) = inner(b, |inside| inside.kind == OpKind::Unary(UnOp::Neg))?;
                    binary(BinOp::Sub, ty, a, iargs[0].clone())
                }
                BinOp::Add if float && inner(a, |inside| inside.kind == OpKind::Unary(UnOp::Neg)).is_some() => {
                    let (_, iargs) = inner(a, |inside| inside.kind == OpKind::Unary(UnOp::Neg))?;
                    binary(BinOp::Sub, ty, b, iargs[0].clone())
                }
                BinOp::Sub if float && inner(b, |inside| inside.kind == OpKind::Unary(UnOp::Neg)).is_some() => {
                    let (_, iargs) = inner(b, |inside| inside.kind == OpKind::Unary(UnOp::Neg))?;
                    binary(BinOp::Add, ty, a, iargs[0].clone())
                }
                // arm: fast-div-same
                BinOp::Div if float && same(a, b) => constant(op, 1.0),
                // arm: fast-div-const
                BinOp::Div if matches!(ty, ElemTy::F32 | ElemTy::F64) => {
                    let divisor = sb.and_then(value_of).filter(|v| *v != 0.0)?;
                    let recip = 1.0 / divisor;
                    if !survives(ty, recip, true) {
                        return None;
                    }
                    binary(BinOp::Mul, ty, a, scalar_of(ty, recip))
                }
                // arm: fast-reassociate
                _ => reassociate(op, args),
            }
        }
        // arm: fast-compare-same
        OpKind::Compare(cmp) if same(a, b) => match cmp {
            CmpOp::Le | CmpOp::Ge | CmpOp::Eq => constant(op, 1.0),
            CmpOp::Ne => constant(op, 0.0),
            CmpOp::Lt | CmpOp::Gt => None,
        },
        // arm: fast-powf
        OpKind::Powf => match scalar(b).and_then(value_of)? {
            0.0 => constant(op, 1.0),
            -1.0 => Some((Op::new(OpKind::Binary(BinOp::Div), ty), args_of(scalar_of(ty, 1.0), a.clone(), Src::unused()))),
            1.0 => identity(op, a),
            2.0 => Some((Op::new(OpKind::Binary(BinOp::Mul), ty), args_of(a.clone(), a.clone(), Src::unused()))),
            0.5 => unary(UnOp::Sqrt, ty, a),
            _ => None,
        },
        _ => None,
    }
}

/// `(x * a) * b`, `(x + a) + b`, `(x - a) + b` and `(x + a) - b` with constants `a` and `b`
/// become one operation with a combined constant.
fn reassociate(op: &Op, args: &Args) -> Option<(Op, Args)> {
    let ty = op.ty;
    if !matches!(ty, ElemTy::F32 | ElemTy::F64) {
        return None;
    }
    let OpKind::Binary(outer) = op.kind else { return None };
    let [a, b, _] = args;
    let (operand, constant) = match outer {
        BinOp::Mul | BinOp::Add => match (scalar(a), scalar(b)) {
            (_, Some(s)) => (a, s),
            (Some(s), None) => (b, s),
            _ => return None,
        },
        BinOp::Sub => (a, scalar(b)?),
        _ => return None,
    };
    let (iop, iargs) = inner(operand, |inside| matches!(inside.kind, OpKind::Binary(_)))?;
    let OpKind::Binary(inside) = iop.kind else { return None };
    if iop.ty != ty {
        return None;
    }
    let (x, k) = match inside {
        BinOp::Mul | BinOp::Add => match (scalar(&iargs[0]), scalar(&iargs[1])) {
            (None, Some(s)) => (&iargs[0], s),
            (Some(s), None) => (&iargs[1], s),
            _ => return None,
        },
        BinOp::Sub => (&iargs[0], scalar(&iargs[1]).filter(|_| scalar(&iargs[0]).is_none())?),
        _ => return None,
    };
    let (k, c) = (value_of(k)?, value_of(constant)?);
    let (kind, folded) = match (outer, inside) {
        (BinOp::Mul, BinOp::Mul) => (BinOp::Mul, k * c),
        (BinOp::Add, BinOp::Add) => (BinOp::Add, k + c),
        (BinOp::Add, BinOp::Sub) => (BinOp::Add, c - k),
        (BinOp::Sub, BinOp::Add) => (BinOp::Add, k - c),
        (BinOp::Sub, BinOp::Sub) => (BinOp::Sub, k + c),
        _ => return None,
    };
    // A constant that overflows or underflows in `ty` would turn finite results into infinities
    // or zeros.
    if !survives(ty, folded, kind == BinOp::Mul && k != 0.0 && c != 0.0) {
        return None;
    }
    binary(kind, ty, x, scalar_of(ty, folded))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::Array;
    use crate::context::{Context, ContextOptions};
    use crate::gpu_type::Df32;

    fn context(fast: bool) -> Context {
        Context::with_options(ContextOptions::from_env().numerics(if fast { crate::Numerics::fast() } else { crate::Numerics::default() })).unwrap()
    }

    /// Bit patterns with every NaN as one NaN.
    fn bits(values: &[f32]) -> Vec<u32> {
        values.iter().map(|v| if v.is_nan() { 0x7fc0_0000 } else { v.to_bits() }).collect()
    }

    const X: [f32; 6] = [0.0, -0.0, f32::NAN, f32::INFINITY, -1.5, 3.0];
    const Y: [f32; 6] = [1.0, -2.0, 0.5, 3.0, f32::NEG_INFINITY, -0.0];
    /// Finite and non-zero: the fast rules are only exact on such values.
    const W: [f32; 6] = [0.5, -1.5, 3.0, 0.25, -2.0, 7.0];

    fn host(array: &Array<f32>) -> Vec<u32> {
        bits(&array.to_vec().unwrap())
    }

    /// Evaluates `case` on `x` and `y` (pending `* 2` of the data, which keeps `-0.0`); returns
    /// the result bits and the length of the program of the last kernel.
    fn run(ctx: &Context, simplify: bool, data: (&[f32], &[f32]), case: fn(&Array<f32>, &Array<f32>) -> Vec<u32>) -> (Vec<u32>, usize) {
        ctx.jit().set_simplify(simplify);
        let x = Array::from_slice_on(ctx, data.0, data.0.len()).unwrap();
        let y = Array::from_slice_on(ctx, data.1, data.1.len()).unwrap();
        let out = case(&(&x * 2.0), &(&y * 2.0));
        let len = ctx.jit().last_program_len();
        ctx.jit().set_simplify(true);
        (out, len)
    }

    #[test]
    fn df32_round_trip_matches_the_unevaluated_sum() {
        let ctx = context(false);
        let data = [-0.0f32, 0.0, -2.0, 3.0];
        let expected = data.map(|x| ((x * 2.0) + 0.0).to_bits());
        for simplify in [false, true] {
            let (got, _) = run(&ctx, simplify, (&data, &data), |b, _| host(&b.to_df32().to_f32()));
            assert_eq!(got, expected);
        }
    }

    #[test]
    fn simplify_multiplying_by_one_is_a_copy() {
        let ctx = context(false);
        let x = Array::from_slice_on(&ctx, &X, X.len()).unwrap();
        let out = &x * 1.0;
        assert_eq!(bits(&out.to_vec().unwrap()), bits(&X));
        assert_eq!(ctx.jit().last_program_len(), 1);
        // A pending operand too: no multiplication is left.
        let y = (&x + 1.0) * 1.0;
        let want = bits(&X.map(|v| v + 1.0));
        assert_eq!(bits(&y.to_vec().unwrap()), want);
        assert_eq!(ctx.jit().last_program_len(), 2);
    }

    #[allow(clippy::erasing_op)]
    #[test]
    fn simplify_constants_of_every_type_evaluate() {
        let ctx = context(true);
        let u = Array::from_slice_on(&ctx, &[3u32, 4, 5], 3).unwrap();
        assert_eq!((&u * 0u32).to_vec().unwrap(), [0, 0, 0]);
        let f = Array::from_slice_on(&ctx, &[3.0f32, 4.0, 5.0], 3).unwrap();
        assert_eq!((&f * 0.0).to_vec().unwrap(), [0.0, 0.0, 0.0]);
        assert_eq!((&f / &f).to_vec().unwrap(), [1.0, 1.0, 1.0]);
        let d = f.to_df32();
        let zero = d.try_mul(Df32::from(0.0)).unwrap();
        assert!(zero.to_vec().unwrap().iter().all(|v| v.hi == 0.0 && v.lo == 0.0));
        let one = d.try_div(&d).unwrap();
        assert!(one.to_vec().unwrap().iter().all(|v| v.hi == 1.0 && v.lo == 0.0));
        let kept = d.try_mul(Df32::from(1.0)).unwrap();
        assert_eq!(kept.to_vec().unwrap(), d.to_vec().unwrap());
    }

    #[test]
    fn simplify_scalar_select_picks_a_branch() {
        let ctx = context(false);
        let f = Array::from_slice_on(&ctx, &W, 6).unwrap();
        let g = Array::from_slice_on(&ctx, &Y, 6).unwrap();
        let args = |m: u32, a: &Array<f32>, b: &Array<f32>| -> Args {
            let node = |x: &Array<f32>| Src::Buffer(std::sync::Arc::clone(x.storage().ok().unwrap()));
            [Src::Scalar(Scalar::u32(m)), node(a), node(b)]
        };
        let op = Op::new(OpKind::Select, ElemTy::F32);
        let (taken, operands) = rewrite(&op, &args(1, &f, &g), false).unwrap();
        assert_eq!(taken.kind, OpKind::Unary(UnOp::Id));
        assert!(same(&operands[0], &args(1, &f, &g)[1]));
        let (_, operands) = rewrite(&op, &args(0, &f, &g), false).unwrap();
        assert!(same(&operands[0], &args(0, &f, &g)[2]));
    }

    #[test]
    fn simplify_division_by_a_constant_becomes_a_product() {
        let ctx = context(true);
        let x = Array::from_slice_on(&ctx, &W, 6).unwrap();
        let buffer = Src::Buffer(std::sync::Arc::clone(x.storage().ok().unwrap()));
        let op = Op::new(OpKind::Binary(BinOp::Div), ElemTy::F32);
        let args = [buffer, Src::Scalar(Scalar::f32(3.0)), Src::unused()];
        assert!(rewrite(&op, &args, false).is_none());
        let (product, operands) = rewrite(&op, &args, true).unwrap();
        assert_eq!(product.kind, OpKind::Binary(BinOp::Mul));
        let Src::Scalar(factor) = operands[1] else { panic!("no constant") };
        assert_eq!(factor.bits, u64::from((1.0f32 / 3.0).to_bits()));
        let zero = [args[0].clone(), Src::Scalar(Scalar::f32(0.0)), Src::unused()];
        assert!(rewrite(&op, &zero, true).is_none());
    }

    #[test]
    fn simplify_fast_rules_keep_finite_values_finite() {
        let ctx = context(true);
        let x = Array::from_slice_on(&ctx, &W, 6).unwrap();
        let buffer = Src::Buffer(std::sync::Arc::clone(x.storage().ok().unwrap()));
        let div = Op::new(OpKind::Binary(BinOp::Div), ElemTy::F32);
        // Dividing by infinity or a subnormal whose reciprocal overflows `f32` must not fold.
        let by_inf = [buffer.clone(), Src::Scalar(Scalar::f32(f32::INFINITY)), Src::unused()];
        assert!(rewrite(&div, &by_inf, true).is_none());
        let by_subnormal = [buffer, Src::Scalar(Scalar::f32(1e-40)), Src::unused()];
        assert!(rewrite(&div, &by_subnormal, true).is_none());
        // A combined constant outside the range of `f32` must stay two operations (`fusion_limit(1)`
        // keeps the shader compiler's own fast-math from reassociating across the two steps).
        let near = |got: Vec<f32>, want: [f32; 2]| {
            assert!(got.iter().zip(want).all(|(g, w)| w != 0.0 && g.is_finite() && (g - w).abs() <= w.abs() * 1e-5), "{got:?} vs {want:?}");
        };
        let large = Array::from_slice_on(&ctx, &[1e20f32, 3e20], 2).unwrap();
        near(crate::fusion_limit(1, || &(&large * 1e-25) * 1e-25).to_vec().unwrap(), [1e-30, 3e-30]);
        let tiny = Array::from_slice_on(&ctx, &[1e-30f32, 2e-30], 2).unwrap();
        near(crate::fusion_limit(1, || &(&tiny * 1e30) * 1e30).to_vec().unwrap(), [1e30, 2e30]);
    }

    /// The binary shift by a multiple of 32 (an array method turns a constant amount into a
    /// unary shift first, so only a pending binary operation reaches this arm).
    #[test]
    fn simplify_binary_shift_by_a_multiple_of_32() {
        let ctx = context(false);
        let u = Array::from_slice_on(&ctx, &[1u32, 2, 3], 3).unwrap();
        let buffer = Src::Buffer(std::sync::Arc::clone(u.storage().ok().unwrap()));
        let args = |amount: u32| [buffer.clone(), Src::Scalar(Scalar::u32(amount)), Src::unused()];
        for bin in [BinOp::Shl, BinOp::Shr, BinOp::Rotl] {
            let op = Op::new(OpKind::Binary(bin), ElemTy::U32);
            let (kept, operands) = rewrite(&op, &args(64), false).unwrap();
            assert_eq!(kept.kind, OpKind::Unary(UnOp::Id));
            assert!(same(&operands[0], &buffer));
            assert!(rewrite(&op, &args(33), false).is_none());
        }
    }
}
