//! Algebraic simplification of an operation about to be created (see [`rewrite`]).
//!
//! The rules in [`exact`] never change a result bit (up to the payload of a NaN). The rules in
//! [`fast`] assume finite values and ignore the sign of zero and rounding, so they only run with
//! [`ContextOptions::fast_math`](crate::context::ContextOptions::fast_math).
//!
//! A rewrite never returns an existing node: an identity becomes a `Unary(Id)` of the surviving
//! operand and a constant a `Unary(Id)` of a scalar, because a node an array holds must not
//! become a second array's node (see [`Held`](super::Held)).
//!
//! A `Unary(Id)` is never folded into the operation it reads (`Id(Id(x))` stays): the stores of
//! `assign` and the readers of reductions are `Id` nodes whose edge to the expression they copy
//! is what the hazard analysis walks, and a copy that read `x` directly would not see that the
//! expression below it has to be evaluated before the write.

use super::{Args, BinOp, CmpOp, ElemTy, Node, Op, OpKind, Scalar, Src, UnOp};

/// The operation and operands to create instead of `op` on `args`, if a rule applies.
pub(super) fn rewrite(op: Op, args: &Args, fast: bool) -> Option<(Op, Args)> {
    exact(op, args).or_else(|| if fast { self::fast(op, args) } else { None })
}

fn is_float(ty: ElemTy) -> bool {
    matches!(ty, ElemTy::F32 | ElemTy::F64 | ElemTy::Df32)
}

/// The bit pattern of the constant `value` as a scalar of type `ty` (`Df32`: `hi` is `value`,
/// `lo` is 0).
fn const_bits(ty: ElemTy, value: f64) -> u64 {
    match ty {
        ElemTy::F32 | ElemTy::Df32 => u64::from((value as f32).to_bits()),
        ElemTy::F64 => value.to_bits(),
        ElemTy::U32 => u64::from(value as u32),
    }
}

/// Whether `s` is the constant `value` of type `ty`, by bit pattern (so `-0.0` is not `0.0`).
fn is_value(s: Option<Scalar>, ty: ElemTy, value: f64) -> bool {
    s.is_some_and(|s| {
        let bits = if s.ty.bytes() == 4 { s.bits & 0xffff_ffff } else { s.bits };
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
    if !wanted(node.op()) || !node.is_pending() || is_looped(node) {
        return None;
    }
    node.operands().map(|args| (node.op(), args))
}

fn is_looped(node: &Node) -> bool {
    node.looped.get().is_some()
}

fn args_of(a: Src, b: Src, c: Src) -> Args {
    [a, b, c]
}

/// `Id(x)`: the operation yields its operand.
fn identity(op: Op, x: &Src) -> Option<(Op, Args)> {
    Some((Op::new(OpKind::Unary(UnOp::Id), op.out_ty()), args_of(x.clone(), Src::unused(), Src::unused())))
}

/// `Id` of the constant `value` of the operation's result type.
fn constant(op: Op, value: f64) -> Option<(Op, Args)> {
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
fn exact(op: Op, args: &Args) -> Option<(Op, Args)> {
    let [a, b, c] = args;
    let ty = op.ty;
    match op.kind {
        OpKind::Binary(bin) => {
            let (sa, sb) = (scalar(a), scalar(b));
            let float = is_float(ty);
            let u32s = ty == ElemTy::U32;
            match bin {
                BinOp::Mul => {
                    if is_value(sb, ty, 1.0) {
                        identity(op, a)
                    } else if is_value(sa, ty, 1.0) {
                        identity(op, b)
                    } else if float && is_value(sb, ty, -1.0) {
                        unary(UnOp::Neg, ty, a)
                    } else if float && is_value(sa, ty, -1.0) {
                        unary(UnOp::Neg, ty, b)
                    } else if u32s && (is_value(sa, ty, 0.0) || is_value(sb, ty, 0.0)) {
                        constant(op, 0.0)
                    } else {
                        None
                    }
                }
                BinOp::Div if is_value(sb, ty, 1.0) => identity(op, a),
                BinOp::Sub if is_value(sb, ty, 0.0) => identity(op, a),
                BinOp::Add => {
                    if float && is_value(sb, ty, -0.0) || u32s && is_value(sb, ty, 0.0) {
                        identity(op, a)
                    } else if float && is_value(sa, ty, -0.0) || u32s && is_value(sa, ty, 0.0) {
                        identity(op, b)
                    } else {
                        None
                    }
                }
                BinOp::Min | BinOp::Max if same(a, b) => identity(op, a),
                BinOp::Rem if u32s && is_value(sb, ty, 1.0) => constant(op, 0.0),
                _ => None,
            }
        }
        OpKind::Unary(un) => {
            if !matches!(un, UnOp::Neg | UnOp::Abs | UnOp::Floor | UnOp::Ceil | UnOp::Sign) {
                return None;
            }
            let (iop, iargs) = inner(a, |inside| matches!(inside.kind, OpKind::Unary(_)))?;
            let OpKind::Unary(inner_un) = iop.kind else { return None };
            match (un, inner_un) {
                (UnOp::Neg, UnOp::Neg) => identity(op, &iargs[0]),
                (UnOp::Abs, UnOp::Abs | UnOp::Neg) => unary(UnOp::Abs, ty, &iargs[0]),
                (UnOp::Floor, UnOp::Floor | UnOp::Ceil) | (UnOp::Ceil, UnOp::Ceil | UnOp::Floor) => {
                    unary(inner_un, ty, &iargs[0])
                }
                (UnOp::Sign, UnOp::Sign) => unary(UnOp::Sign, ty, &iargs[0]),
                _ => None,
            }
        }
        OpKind::Select => {
            if let Some(m) = scalar(a) {
                identity(op, if m.bits & 0xffff_ffff != 0 { b } else { c })
            } else if same(b, c) {
                identity(op, b)
            } else {
                None
            }
        }
        OpKind::Compare(CmpOp::Lt | CmpOp::Gt) if same(a, b) => constant(op, 0.0),
        OpKind::Cast(ElemTy::F32) if matches!(ty, ElemTy::F64 | ElemTy::Df32) => {
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

/// The rewrites that may change rounding, the sign of zero or the handling of NaN and infinity.
fn fast(op: Op, args: &Args) -> Option<(Op, Args)> {
    let [a, b, _] = args;
    let ty = op.ty;
    let float = is_float(ty);
    match op.kind {
        OpKind::Binary(bin) => {
            let (sa, sb) = (scalar(a), scalar(b));
            match bin {
                BinOp::Add if float && is_value(sb, ty, 0.0) => identity(op, a),
                BinOp::Add if float && is_value(sa, ty, 0.0) => identity(op, b),
                BinOp::Mul if float && (is_value(sa, ty, 0.0) || is_value(sb, ty, 0.0)) => constant(op, 0.0),
                BinOp::Sub if float && same(a, b) => constant(op, 0.0),
                BinOp::Div if float && same(a, b) => constant(op, 1.0),
                BinOp::Div if matches!(ty, ElemTy::F32 | ElemTy::F64) => {
                    let divisor = sb.and_then(value_of).filter(|v| *v != 0.0)?;
                    binary(BinOp::Mul, ty, a, scalar_of(ty, 1.0 / divisor))
                }
                _ => reassociate(op, args),
            }
        }
        OpKind::Compare(cmp) if same(a, b) => match cmp {
            CmpOp::Le | CmpOp::Ge | CmpOp::Eq => constant(op, 1.0),
            CmpOp::Ne => constant(op, 0.0),
            CmpOp::Lt | CmpOp::Gt => None,
        },
        OpKind::Powf => match scalar(b).and_then(value_of)? {
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
fn reassociate(op: Op, args: &Args) -> Option<(Op, Args)> {
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
        _ => return None,
    };
    binary(kind, ty, x, scalar_of(ty, folded))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::Array;
    use crate::context::{Context, ContextOptions};
    use crate::gpu_type::Df32;
    use crate::ops::select;

    fn context(fast: bool) -> Context {
        Context::with_options(ContextOptions::from_env().fast_math(fast)).unwrap()
    }

    /// Bit patterns with every NaN as one NaN.
    fn bits(values: &[f32]) -> Vec<u32> {
        values.iter().map(|v| if v.is_nan() { 0x7fc0_0000 } else { v.to_bits() }).collect()
    }

    const X: [f32; 6] = [0.0, -0.0, f32::NAN, f32::INFINITY, -1.5, 3.0];
    const Y: [f32; 6] = [1.0, -2.0, 0.5, 3.0, f32::NEG_INFINITY, -0.0];
    /// Finite and non-zero: the fast rules are only exact on such values.
    const W: [f32; 6] = [0.5, -1.5, 3.0, 0.25, -2.0, 7.0];

    type FloatCase = fn(&Array<f32>, &Array<f32>) -> Vec<u32>;

    fn host(array: &Array<f32>) -> Vec<u32> {
        bits(&array.to_vec().unwrap())
    }

    fn mask(array: &Array<u32>) -> Vec<u32> {
        array.to_vec().unwrap()
    }

    /// Evaluates `case` on `x` and `y` (pending `* 2` of the data, which keeps `-0.0`); returns
    /// the result bits and the length of the program of the last kernel.
    fn run(ctx: &Context, simplify: bool, data: (&[f32], &[f32]), case: FloatCase) -> (Vec<u32>, usize) {
        ctx.jit().set_simplify(simplify);
        let x = Array::from_slice_on(ctx, data.0, data.0.len()).unwrap();
        let y = Array::from_slice_on(ctx, data.1, data.1.len()).unwrap();
        let out = case(&(&x * 2.0), &(&y * 2.0));
        let len = ctx.jit().last_program_len();
        ctx.jit().set_simplify(true);
        (out, len)
    }

    /// Exact rules: same bits as without simplification, and (when `shorter`) fewer instructions.
    fn exact_cases() -> Vec<(&'static str, bool, FloatCase)> {
        vec![
            ("mul by one", false, |b, _| host(&(b * 1.0))),
            ("div by one", false, |b, _| host(&(b / 1.0))),
            ("mul by minus one", false, |b, _| host(&(b * -1.0))),
            ("sub plus zero", false, |b, _| host(&(b - 0.0))),
            ("add minus zero", false, |b, _| host(&(b + -0.0))),
            ("neg neg", true, |b, _| host(&-(-b))),
            ("abs abs", true, |b, _| host(&b.abs().abs())),
            ("abs neg", true, |b, _| host(&(-b).abs())),
            ("floor floor", true, |b, _| host(&b.floor().floor())),
            ("floor ceil", true, |b, _| host(&b.ceil().floor())),
            ("ceil ceil", true, |b, _| host(&b.ceil().ceil())),
            ("ceil floor", true, |b, _| host(&b.floor().ceil())),
            ("sign sign", true, |b, _| host(&b.sign().sign())),
            ("min same", false, |b, _| host(&b.minimum(b))),
            ("max same", false, |b, _| host(&b.maximum(b))),
            ("select same", false, |b, c| {
                let m = c.lt(0.5);
                host(&select(&m, b, b))
            }),
            ("lt same", true, |b, _| mask(&b.lt(b))),
            ("gt same", true, |b, _| mask(&b.gt(b))),
            ("through df32", true, |b, _| host(&b.to_df32().to_f32())),
        ]
    }

    #[test]
    fn simplify_exact_float_rules_keep_every_bit() {
        let ctx = context(false);
        for (name, shorter, case) in exact_cases() {
            for data in [(&X[..], &Y[..]), (&Y[..], &X[..])] {
                let (plain, plain_len) = run(&ctx, false, data, case);
                let (got, len) = run(&ctx, true, data, case);
                assert_eq!(got, plain, "{name}: bits differ");
                if shorter {
                    assert!(len < plain_len, "{name}: program of {len} instructions, {plain_len} unsimplified");
                } else {
                    assert!(len <= plain_len, "{name}: program of {len} instructions, {plain_len} unsimplified");
                }
            }
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

    #[allow(clippy::type_complexity, clippy::erasing_op, clippy::modulo_one)]
    #[test]
    fn simplify_exact_u32_rules() {
        let ctx = context(false);
        let data = [0u32, 1, 2, 7, 4_000_000_000, 3];
        let cases: Vec<(&str, bool, fn(&Array<u32>) -> Vec<u32>)> = vec![
            ("add zero", false, |u| mask(&(u + 0u32))),
            ("sub zero", false, |u| mask(&(u - 0u32))),
            ("mul one", false, |u| mask(&(u * 1u32))),
            ("div one", false, |u| mask(&(u / 1u32))),
            ("mul zero", true, |u| mask(&(u * 0u32))),
            ("rem one", false, |u| mask(&(u % 1u32))),
        ];
        for (name, shorter, case) in cases {
            let mut results = Vec::new();
            for simplify in [false, true] {
                ctx.jit().set_simplify(simplify);
                let u = Array::from_slice_on(&ctx, &data, data.len()).unwrap();
                let out = case(&(&u + 5u32));
                results.push((out, ctx.jit().last_program_len()));
            }
            ctx.jit().set_simplify(true);
            assert_eq!(results[0].0, results[1].0, "{name}");
            if shorter {
                assert!(results[1].1 < results[0].1, "{name}: {} vs {}", results[1].1, results[0].1);
            }
        }
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
        let (taken, operands) = rewrite(op, &args(1, &f, &g), false).unwrap();
        assert_eq!(taken.kind, OpKind::Unary(UnOp::Id));
        assert!(same(&operands[0], &args(1, &f, &g)[1]));
        let (_, operands) = rewrite(op, &args(0, &f, &g), false).unwrap();
        assert!(same(&operands[0], &args(0, &f, &g)[2]));
    }

    #[test]
    fn simplify_cast_through_f64() {
        let ctx = context(false);
        if !ctx.features().contains(wgpu::Features::SHADER_F64) {
            return;
        }
        for simplify in [false, true] {
            ctx.jit().set_simplify(simplify);
            let x = Array::from_slice_on(&ctx, &X, X.len()).unwrap();
            let back = (&x * 2.0).to_f64().to_f32();
            assert_eq!(bits(&back.to_vec().unwrap()), bits(&X.map(|v| v * 2.0)));
            assert_eq!(ctx.jit().last_program_len(), if simplify { 2 } else { 3 });
        }
    }

    /// Fast rules: not applied without the flag, applied with it (checked by program length or
    /// by the exact operation they turn into).
    #[allow(clippy::type_complexity, clippy::erasing_op, clippy::eq_op)]
    fn fast_cases() -> Vec<(&'static str, bool, FloatCase, FloatCase)> {
        vec![
            ("add zero", false, |b, _| host(&(b + 0.0)), |b, _| host(b)),
            ("mul zero", true, |b, _| host(&(b * 0.0)), |b, _| host(&(b - b))),
            ("sub same", true, |b, _| host(&(b - b)), |b, _| host(&(b * 0.0 + 0.0))),
            ("div same", true, |b, _| host(&(b / b)), |b, _| host(&(b * 0.0 + 1.0))),
            ("le same", true, |b, _| mask(&b.le(b)), |b, _| mask(&b.ge(b))),
            ("ne same", true, |b, _| mask(&b.ne(b)), |b, _| mask(&b.lt(b))),
            ("mul chain", true, |b, _| host(&(&(b * 3.0) * 0.5)), |b, _| host(&(b * 1.5))),
            ("add chain", true, |b, _| host(&(&(b + 1.0) + 2.0)), |b, _| host(&(b + 3.0))),
            ("sub then add", true, |b, _| host(&(&(b - 1.0) + 3.0)), |b, _| host(&(b + 2.0))),
            ("add then sub", true, |b, _| host(&(&(b + 5.0) - 2.0)), |b, _| host(&(b + 3.0))),
            ("powf one", false, |b, _| host(&b.powf(1.0)), |b, _| host(b)),
            ("powf two", false, |b, _| host(&b.powf(2.0)), |b, _| host(&(b * b))),
            ("powf half", false, |b, _| host(&b.abs().powf(0.5)), |b, _| host(&b.abs().sqrt())),
            ("div by constant", false, |b, _| host(&(b / 3.0)), |b, _| host(&(b * 0.333_333_34))),
        ]
    }

    #[test]
    fn simplify_fast_rules_only_with_the_flag() {
        let strict = context(false);
        let fast = context(true);
        for (name, shorter, case, equivalent) in fast_cases() {
            let data = (&W[..], &W[..]);
            let (plain, plain_len) = run(&strict, false, data, case);
            let (strict_out, strict_len) = run(&strict, true, data, case);
            assert_eq!(strict_len, plain_len, "{name}: rewritten without fast math");
            assert_eq!(strict_out, plain, "{name}: bits differ without fast math");
            let (fast_out, fast_len) = run(&fast, true, data, case);
            let (want, _) = run(&strict, true, data, equivalent);
            if shorter {
                assert!(fast_len < plain_len, "{name}: {fast_len} vs {plain_len}");
            }
            if name.contains("chain") || name.contains("then") {
                let (got, expected) = (f32s(&fast_out), f32s(&want));
                for (g, e) in got.iter().zip(&expected) {
                    assert!((g - e).abs() <= 1e-5 * e.abs().max(1.0), "{name}: {g} vs {e}");
                }
            } else {
                assert_eq!(fast_out, want, "{name}: not the operation the rule gives");
            }
        }
    }

    #[test]
    fn simplify_division_by_a_constant_becomes_a_product() {
        let ctx = context(true);
        let x = Array::from_slice_on(&ctx, &W, 6).unwrap();
        let buffer = Src::Buffer(std::sync::Arc::clone(x.storage().ok().unwrap()));
        let op = Op::new(OpKind::Binary(BinOp::Div), ElemTy::F32);
        let args = [buffer, Src::Scalar(Scalar::f32(3.0)), Src::unused()];
        assert!(rewrite(op, &args, false).is_none());
        let (product, operands) = rewrite(op, &args, true).unwrap();
        assert_eq!(product.kind, OpKind::Binary(BinOp::Mul));
        let Src::Scalar(factor) = operands[1] else { panic!("no constant") };
        assert_eq!(factor.bits, u64::from((1.0f32 / 3.0).to_bits()));
        let zero = [args[0].clone(), Src::Scalar(Scalar::f32(0.0)), Src::unused()];
        assert!(rewrite(op, &zero, true).is_none());
    }

    fn f32s(bits: &[u32]) -> Vec<f32> {
        bits.iter().map(|&b| f32::from_bits(b)).collect()
    }
}
