//! Scalar operations as NVVM IR, one lane at a time. Math goes through libdevice (`__nv_*`).

use luma_core::kir::*;

use super::fb::{Fb, Op, llty};

fn inv(msg: String) -> ! {
    panic!("luma-cuda codegen invariant: {msg}")
}

pub(crate) fn f32_lit(v: f32) -> Op {
    if v.is_nan() {
        "0x7FF8000000000000".into()
    } else {
        format!("0x{:016X}", (v as f64).to_bits())
    }
}

pub(crate) fn f64_lit(v: f64) -> Op {
    if v.is_nan() {
        "0x7FF8000000000000".into()
    } else {
        format!("0x{:016X}", v.to_bits())
    }
}

pub(crate) fn zero(s: ScalarTy) -> Op {
    match s {
        ScalarTy::Bool => "false".into(),
        ScalarTy::U32 | ScalarTy::I32 | ScalarTy::F16 => "0".into(),
        ScalarTy::F32 => f32_lit(0.0),
        ScalarTy::F64 => f64_lit(0.0),
    }
}

fn flit(s: ScalarTy, v: f64) -> Op {
    if s == ScalarTy::F64 { f64_lit(v) } else { f32_lit(v as f32) }
}

fn from_half(fb: &mut Fb, a: &Op) -> Op {
    fb.declare("declare float @llvm.convert.from.fp16.f32(i16)");
    fb.def(&format!("call float @llvm.convert.from.fp16.f32(i16 {a})"))
}

fn to_half(fb: &mut Fb, a: &Op) -> Op {
    fb.declare("declare i16 @llvm.convert.to.fp16.f32(float)");
    fb.def(&format!("call i16 @llvm.convert.to.fp16.f32(float {a})"))
}

/// libdevice name: `__nv_{base}f` for F32, `__nv_{base}` for F64.
fn lib(base: &str, s: ScalarTy) -> String {
    if s == ScalarTy::F32 { format!("__nv_{base}f") } else { format!("__nv_{base}") }
}

fn call(fb: &mut Fb, name: &str, ret: &str, args: &[(&str, &Op)]) -> Op {
    let tys: Vec<&str> = args.iter().map(|(t, _)| *t).collect();
    fb.declare(&format!("declare {ret} @{name}({})", tys.join(", ")));
    let a: Vec<String> = args.iter().map(|(t, v)| format!("{t} {v}")).collect();
    fb.def(&format!("call {ret} @{name}({})", a.join(", ")))
}

fn math1(fb: &mut Fb, base: &str, s: ScalarTy, a: &Op) -> Op {
    let t = llty(s);
    call(fb, &lib(base, s), t, &[(t, a)])
}

fn math2(fb: &mut Fb, base: &str, s: ScalarTy, a: &Op, b: &Op) -> Op {
    let t = llty(s);
    call(fb, &lib(base, s), t, &[(t, a), (t, b)])
}

fn sel(fb: &mut Fb, c: &Op, t: &str, a: &Op, b: &Op) -> Op {
    fb.def(&format!("select i1 {c}, {t} {a}, {t} {b}"))
}

fn int_intr(fb: &mut Fb, name: &str, a: &Op, flag: bool) -> Op {
    match name {
        "ctpop" => {
            fb.declare("declare i32 @llvm.ctpop.i32(i32)");
            fb.def(&format!("call i32 @llvm.ctpop.i32(i32 {a})"))
        }
        _ => {
            fb.declare(&format!("declare i32 @llvm.{name}.i32(i32, i1)"));
            fb.def(&format!("call i32 @llvm.{name}.i32(i32 {a}, i1 {flag})"))
        }
    }
}

fn first_bit(fb: &mut Fb, v: &Op, leading: bool) -> Op {
    let z = fb.def(&format!("icmp eq i32 {v}, 0"));
    let r = if leading {
        let c = int_intr(fb, "ctlz", v, false);
        fb.def(&format!("sub i32 31, {c}"))
    } else {
        int_intr(fb, "cttz", v, false)
    };
    sel(fb, &z, "i32", &"-1".to_string(), &r)
}

pub(crate) fn unary(fb: &mut Fb, op: IrUnOp, s: ScalarTy, a: &Op) -> (ScalarTy, Op) {
    use IrUnOp::*;
    use ScalarTy::*;
    match (op, s) {
        (F32ToF16Bits, F32) => {
            let h = to_half(fb, a);
            return (U32, fb.def(&format!("zext i16 {h} to i32")));
        }
        (RoundF16, F32) => {
            let h = to_half(fb, a);
            return (F32, from_half(fb, &h));
        }
        (F16BitsToF32, U32) => {
            let h = fb.def(&format!("trunc i32 {a} to i16"));
            return (F32, from_half(fb, &h));
        }
        _ => {}
    }
    if s == F16 {
        let f = from_half(fb, a);
        let (r, v) = unary(fb, op, F32, &f);
        return if r == F32 { (F16, to_half(fb, &v)) } else { (r, v) };
    }
    let t = llty(s);
    match (op, s) {
        (Not, Bool) => (Bool, fb.def(&format!("xor i1 {a}, true"))),
        (Neg, F32 | F64) => (s, fb.def(&format!("fsub {t} 0x8000000000000000, {a}"))),
        (Neg, U32 | I32) => (s, fb.def(&format!("sub i32 0, {a}"))),
        (BitNot, U32 | I32) => (s, fb.def(&format!("xor i32 {a}, -1"))),
        (Abs, F32 | F64) => (s, math1(fb, "fabs", s, a)),
        (Abs, I32) => {
            let c = fb.def(&format!("icmp slt i32 {a}, 0"));
            let n = fb.def(&format!("sub i32 0, {a}"));
            (I32, sel(fb, &c, "i32", &n, a))
        }
        (Abs, U32) => (U32, a.clone()),
        (Sqrt | Exp | Ln | Sin | Cos | Tan | Tanh | Floor | Ceil | Trunc | InverseSqrt | Exp2 | Log2 | Asin | Acos | Atan | Sinh | Cosh | Asinh | Acosh | Atanh | Log10 | Cbrt | Erf | Erfc | Lgamma | Tgamma | Log1p | Expm1 | Round, F32 | F64) => {
            let base = match op {
                Sqrt => "sqrt",
                Exp => "exp",
                Ln => "log",
                Sin => "sin",
                Cos => "cos",
                Tan => "tan",
                Tanh => "tanh",
                Floor => "floor",
                Ceil => "ceil",
                Trunc => "trunc",
                InverseSqrt => "rsqrt",
                Exp2 => "exp2",
                Log2 => "log2",
                Asin => "asin",
                Acos => "acos",
                Atan => "atan",
                Sinh => "sinh",
                Cosh => "cosh",
                Asinh => "asinh",
                Acosh => "acosh",
                Atanh => "atanh",
                Log10 => "log10",
                Cbrt => "cbrt",
                Erf => "erf",
                Erfc => "erfc",
                Lgamma => "lgamma",
                Tgamma => "tgamma",
                Log1p => "log1p",
                Expm1 => "expm1",
                _ => "rint",
            };
            // f32 `exp`, `log`, `log2` and the trigonometric functions use the hardware
            // approximations, within WGSL's precision rules, as the wgpu and Metal backends'
            // compilers do; f64 and the other functions keep libdevice's full precision. f32
            // `sqrt` and `inverseSqrt` are the approximations that flush subnormals (WGSL allows
            // it, the Vulkan driver does it): the forms that keep them add range fix-ups.
            if s == F32 && matches!(op, Sqrt | InverseSqrt) {
                let name = if op == Sqrt { "llvm.nvvm.sqrt.approx.ftz.f" } else { "llvm.nvvm.rsqrt.approx.ftz.f" };
                fb.declare(&format!("declare float @{name}(float)"));
                return (s, fb.def(&format!("call float @{name}(float {a})")));
            }
            let fast = s == F32 && matches!(op, Exp | Ln | Log2 | Sin | Cos | Tan);
            let name = if fast { format!("__nv_fast_{base}f") } else { lib(base, s) };
            (s, call(fb, &name, t, &[(t, a)]))
        }
        (Sign, F32 | F64) => {
            let z = zero(s);
            let gt = fb.def(&format!("fcmp ogt {t} {a}, {z}"));
            let lt = fb.def(&format!("fcmp olt {t} {a}, {z}"));
            let inner = sel(fb, &lt, t, &flit(s, -1.0), &z);
            (s, sel(fb, &gt, t, &flit(s, 1.0), &inner))
        }
        (Sign, I32) => {
            let gt = fb.def(&format!("icmp sgt i32 {a}, 0"));
            let lt = fb.def(&format!("icmp slt i32 {a}, 0"));
            let g = fb.def(&format!("zext i1 {gt} to i32"));
            let l = fb.def(&format!("zext i1 {lt} to i32"));
            (I32, fb.def(&format!("sub i32 {g}, {l}")))
        }
        (Sign, U32) => {
            let c = fb.def(&format!("icmp ne i32 {a}, 0"));
            (U32, fb.def(&format!("zext i1 {c} to i32")))
        }
        (CountOneBits, U32 | I32) => (s, int_intr(fb, "ctpop", a, false)),
        (CountLeadingZeros, U32 | I32) => (s, int_intr(fb, "ctlz", a, false)),
        (CountTrailingZeros, U32 | I32) => (s, int_intr(fb, "cttz", a, false)),
        (FirstTrailingBit, U32 | I32) => (s, first_bit(fb, a, false)),
        (FirstLeadingBit, U32) => (U32, first_bit(fb, a, true)),
        (FirstLeadingBit, I32) => {
            let neg = fb.def(&format!("icmp slt i32 {a}, 0"));
            let inv_a = fb.def(&format!("xor i32 {a}, -1"));
            let v = sel(fb, &neg, "i32", &inv_a, a);
            (I32, first_bit(fb, &v, true))
        }
        (IsNan, F32 | F64) => (Bool, fb.def(&format!("fcmp uno {t} {a}, {a}"))),
        (IsInf, F32 | F64) => {
            let f = math1(fb, "fabs", s, a);
            (Bool, fb.def(&format!("fcmp oeq {t} {f}, 0x7FF0000000000000")))
        }
        (OpaqueGuard, _) => (s, opaque(fb, s, a)),
        _ => inv(format!("unary {op:?} on `{}` has no CUDA lowering", s.wgsl_name())),
    }
}

fn guarded_divisor(fb: &mut Fb, s: ScalarTy, a: &Op, b: &Op) -> Op {
    let z = fb.def(&format!("icmp eq i32 {b}, 0"));
    let c = if s == ScalarTy::I32 {
        let am = fb.def(&format!("icmp eq i32 {a}, -2147483648"));
        let bm = fb.def(&format!("icmp eq i32 {b}, -1"));
        let o = fb.def(&format!("and i1 {am}, {bm}"));
        fb.def(&format!("or i1 {z}, {o}"))
    } else {
        z
    };
    sel(fb, &c, "i32", &"1".to_string(), b)
}

pub(crate) fn binary(fb: &mut Fb, op: IrBinOp, l: ScalarTy, r: ScalarTy, a: &Op, b: &Op) -> (ScalarTy, Op) {
    use IrBinOp::*;
    use ScalarTy::*;
    if l == F16 {
        let fa = from_half(fb, a);
        let fb_ = if r == F16 { from_half(fb, b) } else { b.clone() };
        let (res, v) = binary(fb, op, F32, F32, &fa, &fb_);
        return if res == F32 { (F16, to_half(fb, &v)) } else { (res, v) };
    }
    let t = llty(l);
    if op.is_comparison() {
        let name = match (op, l) {
            (Eq, F32 | F64) => "fcmp oeq",
            (Ne, F32 | F64) => "fcmp une",
            (Lt, F32 | F64) => "fcmp olt",
            (Le, F32 | F64) => "fcmp ole",
            (Gt, F32 | F64) => "fcmp ogt",
            (Ge, F32 | F64) => "fcmp oge",
            (Eq, _) => "icmp eq",
            (Ne, _) => "icmp ne",
            (Lt, I32) => "icmp slt",
            (Le, I32) => "icmp sle",
            (Gt, I32) => "icmp sgt",
            (Ge, I32) => "icmp sge",
            (Lt, _) => "icmp ult",
            (Le, _) => "icmp ule",
            (Gt, _) => "icmp ugt",
            _ => "icmp uge",
        };
        return (Bool, fb.def(&format!("{name} {t} {a}, {b}")));
    }
    let flt = matches!(l, F32 | F64);
    let int = matches!(l, U32 | I32);
    let (o, v) = match op {
        Add | Sub | Mul if flt || int => {
            let n = match (op, flt) {
                (Add, true) => "fadd",
                (Sub, true) => "fsub",
                (Mul, true) => "fmul",
                (Add, false) => "add",
                (Sub, false) => "sub",
                _ => "mul",
            };
            (l, fb.def(&format!("{n} {t} {a}, {b}")))
        }
        // f32: `div.approx.ftz` (a reciprocal and a product, 2 ulp for divisors of magnitude in
        // [2^-126, 2^126]): the precision WGSL specifies and what the Vulkan driver emits for
        // `/`. `div.full` adds about five instructions of range scaling per division, and the
        // form that keeps subnormals two more.
        Div if l == F32 => {
            fb.declare("declare float @llvm.nvvm.div.approx.ftz.f(float, float)");
            (l, fb.def(&format!("call float @llvm.nvvm.div.approx.ftz.f(float {a}, float {b})")))
        }
        Div if flt => (l, fb.def(&format!("fdiv {t} {a}, {b}"))),
        Div if int => {
            let d = guarded_divisor(fb, l, a, b);
            let n = if l == I32 { "sdiv" } else { "udiv" };
            (l, fb.def(&format!("{n} i32 {a}, {d}")))
        }
        Rem if flt => (l, math2(fb, "fmod", l, a, b)),
        Rem if int => {
            let d = guarded_divisor(fb, l, a, b);
            let n = if l == I32 { "srem" } else { "urem" };
            (l, fb.def(&format!("{n} i32 {a}, {d}")))
        }
        AddExact | SubExact | MulExact if flt => {
            let k = match op {
                AddExact => "add",
                SubExact => "sub",
                _ => "mul",
            };
            let name = if l == F32 { format!("__nv_f{k}_rn") } else { format!("__nv_d{k}_rn") };
            (l, call(fb, &name, t, &[(t, a), (t, b)]))
        }
        Min if flt => (l, math2(fb, "fmin", l, a, b)),
        Max if flt => (l, math2(fb, "fmax", l, a, b)),
        Min | Max if int => {
            let p = match (op, l) {
                (Min, I32) => "slt",
                (Min, _) => "ult",
                (_, I32) => "sgt",
                _ => "ugt",
            };
            let c = fb.def(&format!("icmp {p} i32 {a}, {b}"));
            (l, sel(fb, &c, "i32", a, b))
        }
        Pow if flt && l == F32 => (l, call(fb, "__nv_fast_powf", "float", &[("float", a), ("float", b)])),
        Pow if flt => (l, math2(fb, "pow", l, a, b)),
        Atan2 if flt => (l, math2(fb, "atan2", l, a, b)),
        Copysign if flt => (l, math2(fb, "copysign", l, a, b)),
        BitAnd | BitOr | BitXor if int || l == Bool => {
            let n = match op {
                BitAnd => "and",
                BitOr => "or",
                _ => "xor",
            };
            (l, fb.def(&format!("{n} {t} {a}, {b}")))
        }
        Shl | Shr | Rotl if int => {
            let m = fb.def(&format!("and i32 {b}, 31"));
            match op {
                Shl => (l, fb.def(&format!("shl i32 {a}, {m}"))),
                Shr => {
                    let n = if l == I32 { "ashr" } else { "lshr" };
                    (l, fb.def(&format!("{n} i32 {a}, {m}")))
                }
                _ => {
                    let hi = fb.def(&format!("shl i32 {a}, {m}"));
                    let k = fb.def(&format!("sub i32 32, {m}"));
                    let k = fb.def(&format!("and i32 {k}, 31"));
                    let lo = fb.def(&format!("lshr i32 {a}, {k}"));
                    (l, fb.def(&format!("or i32 {hi}, {lo}")))
                }
            }
        }
        MulHiU32 if l == U32 => (U32, call(fb, "__nv_umulhi", "i32", &[("i32", a), ("i32", b)])),
        And if l == Bool => (Bool, fb.def(&format!("and i1 {a}, {b}"))),
        Or if l == Bool => (Bool, fb.def(&format!("or i1 {a}, {b}"))),
        _ => inv(format!("binary {op:?} on `{}`, `{}` has no CUDA lowering", l.wgsl_name(), r.wgsl_name())),
    };
    (o, v)
}

pub(crate) fn fma(fb: &mut Fb, s: ScalarTy, a: &Op, b: &Op, c: &Op) -> Op {
    match s {
        ScalarTy::F16 => {
            let (fa, fb_, fc) = (from_half(fb, a), from_half(fb, b), from_half(fb, c));
            let v = fma(fb, ScalarTy::F32, &fa, &fb_, &fc);
            to_half(fb, &v)
        }
        ScalarTy::F32 => call(fb, "__nv_fmaf_rn", "float", &[("float", a), ("float", b), ("float", c)]),
        ScalarTy::F64 => call(fb, "__nv_fma_rn", "double", &[("double", a), ("double", b), ("double", c)]),
        _ => inv(format!("fma on `{}`", s.wgsl_name())),
    }
}

pub(crate) fn clamp(fb: &mut Fb, s: ScalarTy, a: &Op, lo: &Op, hi: &Op) -> Op {
    let (_, m) = binary(fb, IrBinOp::Max, s, s, a, lo);
    binary(fb, IrBinOp::Min, s, s, &m, hi).1
}

pub(crate) fn convert(fb: &mut Fb, kind: CastKind, ss: ScalarTy, ds: ScalarTy, v: &Op) -> Op {
    use ScalarTy::*;
    if ss == ds {
        return v.clone();
    }
    if ss == F16 {
        let f = from_half(fb, v);
        return convert(fb, kind, F32, ds, &f);
    }
    if ds == F16 {
        let f = convert(fb, CastKind::Convert, ss, F32, v);
        return to_half(fb, &f);
    }
    let st = llty(ss);
    let dt = llty(ds);
    if kind == CastKind::FloatToIntSat && matches!(ss, F32 | F64) && matches!(ds, U32 | I32) {
        let nan = fb.def(&format!("fcmp uno {st} {v}, {v}"));
        if ds == I32 {
            let conv = fb.def(&format!("fptosi {st} {v} to i32"));
            let lo = fb.def(&format!("fcmp ole {st} {v}, {}", flit(ss, -2147483648.0)));
            let s1 = sel(fb, &lo, "i32", &"-2147483648".to_string(), &conv);
            let hi = fb.def(&format!("fcmp oge {st} {v}, {}", flit(ss, 2147483648.0)));
            let s2 = sel(fb, &hi, "i32", &"2147483647".to_string(), &s1);
            return sel(fb, &nan, "i32", &"0".to_string(), &s2);
        }
        let conv = fb.def(&format!("fptoui {st} {v} to i32"));
        let hi = fb.def(&format!("fcmp oge {st} {v}, {}", flit(ss, 4294967296.0)));
        let s1 = sel(fb, &hi, "i32", &"-1".to_string(), &conv);
        let lo = fb.def(&format!("fcmp ole {st} {v}, {}", flit(ss, 0.0)));
        let s2 = sel(fb, &lo, "i32", &"0".to_string(), &s1);
        return sel(fb, &nan, "i32", &"0".to_string(), &s2);
    }
    match (ss, ds) {
        (F32 | F64, Bool) => fb.def(&format!("fcmp une {st} {v}, {}", zero(ss))),
        (U32 | I32, Bool) => fb.def(&format!("icmp ne i32 {v}, 0")),
        (Bool, U32 | I32) => fb.def(&format!("zext i1 {v} to i32")),
        (Bool, F32 | F64) => fb.def(&format!("uitofp i1 {v} to {dt}")),
        (U32, I32) | (I32, U32) => v.clone(),
        (U32, F32 | F64) => fb.def(&format!("uitofp i32 {v} to {dt}")),
        (I32, F32 | F64) => fb.def(&format!("sitofp i32 {v} to {dt}")),
        (F32, F64) => fb.def(&format!("fpext float {v} to double")),
        (F64, F32) => fb.def(&format!("fptrunc double {v} to float")),
        (F32 | F64, I32) => fb.def(&format!("fptosi {st} {v} to i32")),
        (F32 | F64, U32) => fb.def(&format!("fptoui {st} {v} to i32")),
        _ => inv(format!("conversion `{}` -> `{}`", ss.wgsl_name(), ds.wgsl_name())),
    }
}

pub(crate) fn bitcast_scalar(fb: &mut Fb, ss: ScalarTy, ds: ScalarTy, v: &Op) -> Option<Op> {
    use ScalarTy::*;
    if ss == ds {
        return Some(v.clone());
    }
    match (ss, ds) {
        (U32, I32) | (I32, U32) => Some(v.clone()),
        (F32, U32 | I32) => Some(fb.def(&format!("bitcast float {v} to i32"))),
        (U32 | I32, F32) => Some(fb.def(&format!("bitcast i32 {v} to float"))),
        _ => None,
    }
}

pub(crate) fn opaque(fb: &mut Fb, s: ScalarTy, a: &Op) -> Op {
    let (t, c) = match s {
        ScalarTy::Bool => return a.clone(),
        ScalarTy::F32 => ("float", "=f,0"),
        ScalarTy::F64 => ("double", "=d,0"),
        ScalarTy::U32 | ScalarTy::I32 => ("i32", "=r,0"),
        ScalarTy::F16 => ("i16", "=h,0"),
    };
    fb.def(&format!("call {t} asm sideeffect \"\", \"{c}\"({t} {a})"))
}

fn atomic_ty(s: ScalarTy) -> &'static str {
    match s {
        ScalarTy::Bool | ScalarTy::F16 => inv(format!("atomic access on `{}` is not a valid KIR atomic type", s.wgsl_name())),
        _ => llty(s),
    }
}

pub(crate) fn atomic_load(fb: &mut Fb, s: ScalarTy, ptr: &Op) -> Op {
    let t = atomic_ty(s);
    let p = fb.scalar_ptr(ptr, 0, s);
    fb.def(&format!("load volatile {t}, {t}* {p}, align {}", s.size_bytes()))
}

pub(crate) fn atomic_store(fb: &mut Fb, s: ScalarTy, ptr: &Op, v: &Op) {
    let t = atomic_ty(s);
    let p = fb.scalar_ptr(ptr, 0, s);
    fb.ins(&format!("store volatile {t} {v}, {t}* {p}, align {}", s.size_bytes()));
}

/// Registers `@luma_atomic_{fmin,fmax,dmin,dmax}`: the CAS loop of the old `PRELUDE_ATOMIC`.
fn atomic_minmax_helper(fb: &mut Fb, s: ScalarTy, max: bool) -> String {
    let (name, t, i, a, f) = if s == ScalarTy::F32 {
        (if max { "luma_atomic_fmax" } else { "luma_atomic_fmin" }, "float", "i32", 4, if max { "__nv_fmaxf" } else { "__nv_fminf" })
    } else {
        (if max { "luma_atomic_dmax" } else { "luma_atomic_dmin" }, "double", "i64", 8, if max { "__nv_fmax" } else { "__nv_fmin" })
    };
    fb.declare(&format!("declare {t} @{f}({t}, {t})"));
    if !fb.helpers.contains_key(name) {
        let text = format!(
            "define internal {t} @{name}(i8* %p, {t} %v) alwaysinline nounwind {{\n\
entry:\n\
  %u = bitcast i8* %p to {i}*\n\
  %o = load volatile {i}, {i}* %u, align {a}\n\
  br label %loop\n\
loop:\n\
  %assumed = phi {i} [ %o, %entry ], [ %old, %try ]\n\
  %af = bitcast {i} %assumed to {t}\n\
  %nv = call {t} @{f}({t} %af, {t} %v)\n\
  %nb = bitcast {t} %nv to {i}\n\
  %same = icmp eq {i} %nb, %assumed\n\
  br i1 %same, label %done, label %try\n\
try:\n\
  %r = cmpxchg {i}* %u, {i} %assumed, {i} %nb seq_cst seq_cst\n\
  %old = extractvalue {{ {i}, i1 }} %r, 0\n\
  %ok = extractvalue {{ {i}, i1 }} %r, 1\n\
  br i1 %ok, label %done, label %loop\n\
done:\n\
  ret {t} %af\n\
}}\n"
        );
        fb.helpers.insert(name.to_string(), text);
    }
    name.to_string()
}

pub(crate) fn atomic_rmw(fb: &mut Fb, op: IrAtomicOp, s: ScalarTy, ptr: &Op, v: &Op) -> Op {
    use IrAtomicOp::*;
    use ScalarTy::*;
    let int_op = |op: IrAtomicOp, s: ScalarTy| -> &'static str {
        match (op, s) {
            (Add, _) => "add",
            (Sub, _) => "sub",
            (And, _) => "and",
            (Or, _) => "or",
            (Xor, _) => "xor",
            (Exchange, _) => "xchg",
            (Min, I32) => "min",
            (Min, _) => "umin",
            (Max, I32) => "max",
            (Max, _) => "umax",
        }
    };
    match s {
        U32 | I32 => {
            let p = fb.typed_ptr(ptr, 0, "i32", 4);
            fb.def(&format!("atomicrmw {} i32* {p}, i32 {v} seq_cst", int_op(op, s)))
        }
        F32 | F64 => {
            let (t, i, bits_op) = if s == F32 { ("float", "i32", "f32.p0f32") } else { ("double", "i64", "f64.p0f64") };
            match op {
                Add | Sub => {
                    let name = format!("llvm.nvvm.atomic.load.add.{bits_op}");
                    fb.declare(&format!("declare {t} @{name}({t}*, {t})"));
                    let val = if op == Sub { fb.def(&format!("fsub {t} 0x8000000000000000, {v}")) } else { v.clone() };
                    let p = fb.scalar_ptr(ptr, 0, s);
                    fb.def(&format!("call {t} @{name}({t}* {p}, {t} {val})"))
                }
                Min | Max => {
                    let name = atomic_minmax_helper(fb, s, op == Max);
                    fb.def(&format!("call {t} @{name}(i8* {ptr}, {t} {v})"))
                }
                Exchange | And | Or | Xor => {
                    let p = fb.typed_ptr(ptr, 0, i, s.size_bytes());
                    let bits = fb.def(&format!("bitcast {t} {v} to {i}"));
                    let old = fb.def(&format!("atomicrmw {} {i}* {p}, {i} {bits} seq_cst", int_op(op, s)));
                    fb.def(&format!("bitcast {i} {old} to {t}"))
                }
            }
        }
        Bool | F16 => inv(format!("atomic {op:?} on `{}` is not a valid KIR atomic type", s.wgsl_name())),
    }
}

pub(crate) fn atomic_cas(fb: &mut Fb, s: ScalarTy, ptr: &Op, cmp: &Op, val: &Op) -> (Op, Op) {
    use ScalarTy::*;
    let (i, t, float) = match s {
        U32 | I32 => ("i32", "i32", false),
        F32 => ("i32", "float", true),
        F64 => ("i64", "double", true),
        _ => inv(format!("atomic compare-exchange on `{}` is not a valid KIR atomic type", s.wgsl_name())),
    };
    let p = fb.typed_ptr(ptr, 0, i, s.size_bytes());
    let (c, n) = if float {
        (fb.def(&format!("bitcast {t} {cmp} to {i}")), fb.def(&format!("bitcast {t} {val} to {i}")))
    } else {
        (cmp.clone(), val.clone())
    };
    let r = fb.def(&format!("cmpxchg {i}* {p}, {i} {c}, {i} {n} seq_cst seq_cst"));
    let prev_bits = fb.def(&format!("extractvalue {{ {i}, i1 }} {r}, 0"));
    let exchanged = fb.def(&format!("icmp eq {i} {prev_bits}, {c}"));
    let prev = if float { fb.def(&format!("bitcast {i} {prev_bits} to {t}")) } else { prev_bits };
    (prev, exchanged)
}

// ----- subgroup ------------------------------------------------------------------------------

pub(crate) fn lane_id(fb: &mut Fb) -> Op {
    fb.def("call i32 asm sideeffect \"mov.u32 $0, %laneid;\", \"=r\"()")
}

/// Every subgroup instruction is `convergent`: the optimizer must not move one into a branch
/// (or out of one, or copy it into the predecessors of its block), as that changes which lanes
/// execute it together.
pub(crate) fn active_mask(fb: &mut Fb) -> Op {
    fb.def("call i32 asm sideeffect \"activemask.b32 $0;\", \"=r\"() convergent")
}

/// Waits until every lane of `mask` has got here (`bar.warp.sync`).
pub(crate) fn warp_sync(fb: &mut Fb, mask: &Op) {
    fb.ins(&format!("call void asm sideeffect \"bar.warp.sync $0;\", \"r\"(i32 {mask}) convergent"));
}

/// The mask of the lanes a warp-wide operation involves: all of them when `converged` (see
/// `conv`), else the active ones.
fn warp_mask(fb: &mut Fb, converged: bool) -> Op {
    if converged { "-1".to_string() } else { active_mask(fb) }
}

pub(crate) fn ballot(fb: &mut Fb, pred: Option<&Op>, converged: bool) -> Op {
    let mask = warp_mask(fb, converged);
    let p = match pred {
        Some(p) => fb.def(&format!("zext i1 {p} to i32")),
        None => "1".to_string(),
    };
    fb.def(&format!(
        "call i32 asm sideeffect \"{{ .reg .pred p; setp.ne.b32 p, $1, 0; vote.sync.ballot.b32 $0, p, $2; }}\", \"=r,r,r\"(i32 {p}, i32 {mask}) convergent"
    ))
}

/// 32-bit `shfl.sync.{kind}`; `c` is 31 for `idx`/`bfly`, 0 for `up`.
fn shfl32(fb: &mut Fb, kind: &str, v: &Op, lane: &Op, mask: &Op) -> Op {
    let c = if kind == "up" { "0" } else { "31" };
    fb.def(&format!(
        "call i32 asm sideeffect \"shfl.sync.{kind}.b32 $0, $1, $2, $3, $4;\", \"=r,r,r,r,r\"(i32 {v}, i32 {lane}, i32 {c}, i32 {mask}) convergent"
    ))
}

fn shfl(fb: &mut Fb, kind: &str, s: ScalarTy, v: &Op, lane: &Op, mask: &Op) -> Op {
    use ScalarTy::*;
    match s {
        U32 | I32 => shfl32(fb, kind, v, lane, mask),
        F32 => {
            let b = fb.def(&format!("bitcast float {v} to i32"));
            let r = shfl32(fb, kind, &b, lane, mask);
            fb.def(&format!("bitcast i32 {r} to float"))
        }
        F64 => {
            let b = fb.def(&format!("bitcast double {v} to i64"));
            let lo = fb.def(&format!("trunc i64 {b} to i32"));
            let sh = fb.def(&format!("lshr i64 {b}, 32"));
            let hi = fb.def(&format!("trunc i64 {sh} to i32"));
            let lo = shfl32(fb, kind, &lo, lane, mask);
            let hi = shfl32(fb, kind, &hi, lane, mask);
            let lo = fb.def(&format!("zext i32 {lo} to i64"));
            let hi = fb.def(&format!("zext i32 {hi} to i64"));
            let hi = fb.def(&format!("shl i64 {hi}, 32"));
            let r = fb.def(&format!("or i64 {hi}, {lo}"));
            fb.def(&format!("bitcast i64 {r} to double"))
        }
        F16 => {
            let z = fb.def(&format!("zext i16 {v} to i32"));
            let r = shfl32(fb, kind, &z, lane, mask);
            fb.def(&format!("trunc i32 {r} to i16"))
        }
        Bool => {
            let z = fb.def(&format!("zext i1 {v} to i32"));
            let r = shfl32(fb, kind, &z, lane, mask);
            fb.def(&format!("icmp ne i32 {r}, 0"))
        }
    }
}

/// Combine rule for `Add`/`Min`/`Max` (`a` accumulated, `b` new).
fn combine(fb: &mut Fb, which: IrSubgroupOp, s: ScalarTy, a: &Op, b: &Op) -> Op {
    use ScalarTy::*;
    let op = match which {
        IrSubgroupOp::Add | IrSubgroupOp::ExclusiveAdd => IrBinOp::Add,
        IrSubgroupOp::Min => IrBinOp::Min,
        _ => IrBinOp::Max,
    };
    match (op, s) {
        (IrBinOp::Min | IrBinOp::Max, U32 | I32) => {
            let p = match (op, s) {
                (IrBinOp::Min, I32) => "slt",
                (IrBinOp::Min, _) => "ult",
                (_, I32) => "sgt",
                _ => "ugt",
            };
            let c = fb.def(&format!("icmp {p} i32 {b}, {a}"));
            sel(fb, &c, "i32", b, a)
        }
        (_, Bool) => inv("subgroup reduction on `bool`".to_string()),
        _ => binary(fb, op, s, s, a, b).1,
    }
}

fn cttz_true(fb: &mut Fb, m: &Op) -> Op {
    fb.declare("declare i32 @llvm.cttz.i32(i32, i1)");
    fb.def(&format!("call i32 @llvm.cttz.i32(i32 {m}, i1 true)"))
}

/// Loop over the set bits of `m` (`rest`), accumulating with `step(fb, acc, lane)`;
/// returns the accumulated value on exit. `h` must be positioned in a block that has not been
/// terminated; `pre` is that block's label.
fn lane_loop(h: &mut Fb, pre: &str, t: &str, m: &Op, init: &Op, mut step: impl FnMut(&mut Fb, &Op, &Op) -> Op) -> Op {
    let m1 = h.def(&format!("sub i32 {m}, 1"));
    let rest0 = h.def(&format!("and i32 {m}, {m1}"));
    let (lp, bd, ex) = (h.label(), h.label(), h.label());
    let (restp, accp, rest1) = (h.fresh(), h.fresh(), h.fresh());
    h.start(&lp);
    h.ins(&format!("{restp} = phi i32 [ {rest0}, %{pre} ], [ {rest1}, %{bd} ]"));
    h.ins(&format!("{accp} = phi {t} [ {init}, %{pre} ], [ @@ACC@@, %{bd} ]"));
    let done = h.def(&format!("icmp eq i32 {restp}, 0"));
    h.cond_br(&done, &ex, &bd);
    h.start(&bd);
    let l = cttz_true(h, &restp);
    let r1 = h.def(&format!("sub i32 {restp}, 1"));
    h.ins(&format!("{rest1} = and i32 {restp}, {r1}"));
    let n = step(h, &accp, &l);
    h.body = h.body.replacen("@@ACC@@", &n, 1);
    h.br(&lp);
    h.start(&ex);
    accp
}

fn sg_name(op: IrSubgroupOp, s: ScalarTy, has_lane: bool, converged: bool) -> String {
    let k = match (op, has_lane) {
        (IrSubgroupOp::Add, _) => "add",
        (IrSubgroupOp::Min, _) => "min",
        (IrSubgroupOp::Max, _) => "max",
        (IrSubgroupOp::ExclusiveAdd, _) => "xadd",
        (IrSubgroupOp::Shuffle, true) => "shuffle",
        (IrSubgroupOp::Shuffle, false) => "first",
        (IrSubgroupOp::Ballot, _) => inv("ballot is not a subgroup helper".to_string()),
    };
    format!("luma_sg_{k}_{}{}", if s == ScalarTy::F16 { "f16" } else { s.wgsl_name() }, if converged { "_c" } else { "" })
}

fn build_sg(op: IrSubgroupOp, s: ScalarTy, has_lane: bool, converged: bool, name: &str) -> (String, Fb) {
    let t = llty(s);
    let mut h = Fb::new();
    let mask = warp_mask(&mut h, converged);
    let v: Op = "%v".into();
    match (op, has_lane) {
        (IrSubgroupOp::Shuffle, true) => {
            let r = shfl(&mut h, "idx", s, &v, &"%l".to_string(), &mask);
            h.ret(&format!("ret {t} {r}"));
        }
        (IrSubgroupOp::Shuffle, false) => {
            // The first active lane; lane 0 when every lane is.
            let l = if converged { "0".to_string() } else { cttz_true(&mut h, &mask) };
            let r = shfl(&mut h, "idx", s, &v, &l, &mask);
            h.ret(&format!("ret {t} {r}"));
        }
        (IrSubgroupOp::Add | IrSubgroupOp::Min | IrSubgroupOp::Max, _) => {
            let (fl, pl) = (h.label(), h.label());
            if !converged {
                let full = h.def(&format!("icmp eq i32 {mask}, -1"));
                h.cond_br(&full, &fl, &pl);
                h.start(&fl);
            }
            // Full warp: xor butterfly.
            let mut x = v.clone();
            for o in [16, 8, 4, 2, 1] {
                let sh = shfl(&mut h, "bfly", s, &x, &o.to_string(), &mask);
                x = combine(&mut h, op, s, &x, &sh);
            }
            h.ret(&format!("ret {t} {x}"));
            if converged {
                let params = if has_lane { format!("{t} %v, i32 %l") } else { format!("{t} %v") };
                return (format!("define internal {t} @{name}({params}) alwaysinline nounwind convergent {{\n"), h);
            }
            // Partial warp: fold the active lanes in ascending order.
            h.start(&pl);
            let f = cttz_true(&mut h, &mask);
            let init = shfl(&mut h, "idx", s, &v, &f, &mask);
            let acc = lane_loop(&mut h, &pl, t, &mask, &init, |h, acc, l| {
                let x = shfl(h, "idx", s, &v, l, &mask);
                combine(h, op, s, acc, &x)
            });
            h.ret(&format!("ret {t} {acc}"));
        }
        (IrSubgroupOp::ExclusiveAdd, _) => {
            let lane = lane_id(&mut h);
            let (fl, pl) = (h.label(), h.label());
            if !converged {
                let full = h.def(&format!("icmp eq i32 {mask}, -1"));
                h.cond_br(&full, &fl, &pl);
                h.start(&fl);
            }
            // Full warp: Hillis-Steele inclusive scan, then shift by one lane.
            let mut sc = v.clone();
            for o in [1, 2, 4, 8, 16] {
                let tt = shfl(&mut h, "up", s, &sc, &o.to_string(), &mask);
                let ge = h.def(&format!("icmp uge i32 {lane}, {o}"));
                let c = combine(&mut h, op, s, &sc, &tt);
                sc = sel(&mut h, &ge, t, &c, &sc);
            }
            let ex = shfl(&mut h, "up", s, &sc, &"1".to_string(), &mask);
            let l0 = h.def(&format!("icmp eq i32 {lane}, 0"));
            let r = sel(&mut h, &l0, t, &zero(s), &ex);
            h.ret(&format!("ret {t} {r}"));
            if converged {
                return (format!("define internal {t} @{name}({}) alwaysinline nounwind convergent {{\n", if has_lane { format!("{t} %v, i32 %l") } else { format!("{t} %v") }), h);
            }
            // Partial warp: sum of the active lanes below this one.
            h.start(&pl);
            let acc = lane_loop(&mut h, &pl, t, &mask, &zero(s), |h, acc, l| {
                let x = shfl(h, "idx", s, &v, l, &mask);
                let lt = h.def(&format!("icmp ult i32 {l}, {lane}"));
                let c = combine(h, op, s, acc, &x);
                sel(h, &lt, t, &c, acc)
            });
            h.ret(&format!("ret {t} {acc}"));
        }
        (IrSubgroupOp::Ballot, _) => unreachable!(),
    }
    let params = if has_lane { format!("{t} %v, i32 %l") } else { format!("{t} %v") };
    let head = format!("define internal {t} @{name}({params}) alwaysinline nounwind convergent {{\n");
    (head, h)
}

pub(crate) fn subgroup(fb: &mut Fb, op: IrSubgroupOp, s: ScalarTy, v: &Op, lane: Option<&Op>, converged: bool) -> Op {
    let has_lane = lane.is_some();
    let name = sg_name(op, s, has_lane, converged);
    if !fb.helpers.contains_key(&name) {
        let (head, h) = build_sg(op, s, has_lane, converged, &name);
        fb.decls.extend(h.decls.iter().cloned());
        fb.helpers.extend(h.helpers.clone());
        let text = format!("{head}{}}}\n", h.finish());
        fb.helpers.insert(name.clone(), text);
    }
    let t = llty(s);
    match lane {
        Some(l) => fb.def(&format!("call {t} @{name}({t} {v}, i32 {l}) convergent")),
        None => fb.def(&format!("call {t} @{name}({t} {v}) convergent")),
    }
}

/// A fetch from the texture object whose handle is held in `ptr` (an `i8*` made of the `i64`
/// parameter): the four channels as `f32` lanes. `x` and `y` are `i32` texel coordinates for
/// `Load` and `f32` normalized coordinates otherwise.
pub(crate) fn texture(fb: &mut Fb, op: TextureOp, ptr: &Op, x: &Op, y: &Op) -> Vec<Op> {
    let handle = fb.def(&format!("ptrtoint i8* {ptr} to i64"));
    let (name, ct) = match op {
        TextureOp::Load => ("llvm.nvvm.tex.unified.2d.v4f32.s32".to_string(), "i32"),
        TextureOp::Sample => ("llvm.nvvm.tex.unified.2d.v4f32.f32".to_string(), "float"),
        TextureOp::Gather(c) => (format!("llvm.nvvm.tld4.unified.{}.2d.v4f32.f32", ["r", "g", "b", "a"][usize::from(c & 3)]), "float"),
    };
    fb.declare(&format!("declare {{ float, float, float, float }} @{name}(i64, {ct}, {ct})"));
    let r = fb.def(&format!("call {{ float, float, float, float }} @{name}(i64 {handle}, {ct} {x}, {ct} {y})"));
    (0..4).map(|k| fb.def(&format!("extractvalue {{ float, float, float, float }} {r}, {k}"))).collect()
}
