//! The WGSL printer of the kernel IR.
//!
//! [`emit`] turns a [`KernelModule`] into one WGSL module in the shape luma's launches expect: the
//! grid uniform at `@group(0) @binding(0)`, the kernel's arguments at `@group(1)` under their
//! original binding numbers and one compute entry point, [`ENTRY`]. WebGPU's gaps are filled
//! with small helper functions the module carries only when it uses them (the high word of a
//! 32-bit product, rotates, a `tanh` that saturates, exact `f64` rounding), and `f32` atomics
//! become compare-and-exchange loops over the bits.
//!
//! Every identifier of the IR is printed with a numeric tag (`l3_x`, `b0_out`, `S1_Params`), so
//! nothing the IR names can collide with a WGSL keyword or with another name.

use std::cell::{Cell, RefCell};
use std::collections::BTreeMap;
use std::fmt::Write as _;

use luma_core::hal::BindingKind;
use luma_core::kir::*;

/// The name of the compute entry point of every emitted module.
pub const ENTRY: &str = "luma_main";

const LANES: [char; 4] = ['x', 'y', 'z', 'w'];

/// `module` as WGSL, declaring every binding.
pub fn emit(module: &KernelModule) -> String {
    emit_with(module, None, false)
}

/// `module` as WGSL, declaring only the bindings `used` marks (the entry point's binding mask, see
/// `CompiledKernel::used`), each under its original binding number. With `hardware_f32_atomics`
/// (the device has `SHADER_FLOAT32_ATOMIC`) the storage bindings of `f32` atomics that are only
/// ever added to are declared `atomic<f32>` and `atomicAdd`ed directly instead of through a
/// compare-and-swap loop on their bits.
pub fn emit_with(module: &KernelModule, used: Option<&[bool]>, hardware_f32_atomics: bool) -> String {
    let hardware = if hardware_f32_atomics { hardware_f32_bindings(module) } else { vec![false; module.bindings.len()] };
    let em = Em { m: module, helpers: RefCell::new(BTreeMap::new()), f16: Cell::new(false), temps: Cell::new(0), hardware };
    em.module(used)
}

/// Which storage bindings of `module` hold `f32` atomics that nothing but `atomicAdd` and
/// `atomicSub` touches: those can use the device's native `f32` atomics (WGSL has no
/// compare-exchange, min or max on them, nor a way to see their bits).
fn hardware_f32_bindings(module: &KernelModule) -> Vec<bool> {
    let mut hardware: Vec<bool> = module.bindings.iter().map(|d| d.kind == BindingKind::StorageReadWrite && matches!(d.elem_ty, KirType::Atomic(ScalarTy::F32))).collect();
    if !hardware.contains(&true) {
        return hardware;
    }
    let mut spoil = |target: &AtomicTarget| {
        if let AtomicTarget::Storage { binding, .. } = target {
            hardware[*binding as usize] = false;
        }
    };
    let mut scan = |block: &[Stmt]| {
        walk_stmts(block, &mut |stmt| {
            match stmt {
                Stmt::AtomicStore { target, .. } | Stmt::AtomicCompareExchange { target, .. } => spoil(target),
                Stmt::AtomicRmw { op, target, .. } if !matches!(op, IrAtomicOp::Add | IrAtomicOp::Sub) => spoil(target),
                _ => {}
            }
            stmt.for_each_expr(&mut |e| {
                e.walk(&mut |x| {
                    if let Expr::AtomicLoad { target, .. } = x {
                        spoil(target);
                    }
                })
            });
        })
    };
    scan(&module.entry_body);
    module.functions.iter().for_each(|f| scan(&f.body));
    hardware
}

fn ident(s: &str) -> String {
    s.chars().map(|c| if c.is_ascii_alphanumeric() || c == '_' { c } else { '_' }).collect()
}

fn local_name(id: LocalId, locals: &[(String, KirType, bool)]) -> String {
    format!("l{}_{}", id.0, ident(locals.get(id.0 as usize).map_or("", |l| l.0.as_str())))
}

/// What a function body can see: its locals and parameters.
struct Scope<'a> {
    locals: &'a [(String, KirType, bool)],
    params: &'a [(String, KirType)],
}

struct Em<'m> {
    m: &'m KernelModule,
    helpers: RefCell<BTreeMap<String, String>>,
    f16: Cell<bool>,
    temps: Cell<u32>,
    /// Per binding: whether it is an `array<atomic<f32>>` of the device's native `f32` atomics.
    hardware: Vec<bool>,
}

fn lit_i32(v: i32) -> String {
    if v == i32::MIN {
        "(-2147483647i - 1i)".to_owned()
    } else if v < 0 {
        format!("({v}i)")
    } else {
        format!("{v}i")
    }
}

impl Em<'_> {
    // -- Names and types --

    fn scalar(&self, s: ScalarTy) -> &'static str {
        if s == ScalarTy::F16 {
            self.f16.set(true);
        }
        s.wgsl_name()
    }

    fn ty(&self, t: &KirType) -> String {
        match t {
            KirType::Scalar(s) => self.scalar(*s).to_owned(),
            KirType::Vec { elem, lanes } => format!("vec{lanes}<{}>", self.scalar(*elem)),
            KirType::Array { elem, len, stride } => {
                let e = self.array_elem(elem, *stride);
                if *len == 0 { format!("array<{e}>") } else { format!("array<{e}, {len}>") }
            }
            KirType::Struct(i) => self.struct_name(*i),
            KirType::Atomic(s) => format!("atomic<{}>", self.scalar(if *s == ScalarTy::F32 { ScalarTy::U32 } else { *s })),
        }
    }

    /// The `u32` type with the lanes of `t`.
    fn uint_like(&self, t: &KirType) -> String {
        match t.lanes() {
            1 => "u32".to_owned(),
            n => format!("vec{n}<u32>"),
        }
    }

    fn struct_name(&self, i: u32) -> String {
        format!("S{i}_{}", ident(&self.m.structs[i as usize].name))
    }

    fn field_name(&self, s: u32, f: u32) -> String {
        format!("m{f}_{}", ident(&self.m.structs[s as usize].fields[f as usize].name))
    }

    fn binding_name(&self, b: u32) -> String {
        format!("b{b}_{}", ident(&self.m.bindings[b as usize].name))
    }

    fn shared_name(&self, s: u32) -> String {
        format!("w{s}_{}", ident(&self.m.shared[s as usize].name))
    }

    fn private_name(&self, p: u32) -> String {
        format!("p{p}_{}", ident(&self.m.privates[p as usize].name))
    }

    fn function_name(&self, f: FnId) -> String {
        format!("f{}_{}", f.0, ident(&self.m.functions[f.0 as usize].name))
    }

    fn binding_root_ty(&self, b: u32) -> KirType {
        let d = &self.m.bindings[b as usize];
        if d.is_unsized_array {
            KirType::Array { elem: Box::new(d.elem_ty.clone()), len: 0, stride: d.element_stride.unwrap_or(0) }
        } else {
            d.elem_ty.clone()
        }
    }

    fn shared_root_ty(&self, s: u32) -> KirType {
        let d = &self.m.shared[s as usize];
        match d.len {
            Some(len) => {
                let (size, align) = d.elem_ty.layout(&self.m.structs);
                KirType::Array { elem: Box::new(d.elem_ty.clone()), len, stride: size.next_multiple_of(align) }
            }
            None => d.elem_ty.clone(),
        }
    }

    /// A float literal of `s` for `v`; non-finite values come from the bits of an integer the
    /// compiler cannot fold (`luma._pad`, always 0), as WGSL has no literal for them.
    fn float_lit(&self, s: ScalarTy, v: f64) -> String {
        if !v.is_finite() {
            let bits = if v.is_nan() {
                "0x7fc00000u"
            } else if v > 0.0 {
                "0x7f800000u"
            } else {
                "0xff800000u"
            };
            let base = format!("bitcast<f32>(luma._pad | {bits})");
            return match s {
                ScalarTy::F64 => format!("f64({base})"),
                ScalarTy::F16 => {
                    self.f16.set(true);
                    format!("f16({base})")
                }
                _ => base,
            };
        }
        let text = match s {
            ScalarTy::F16 => {
                self.f16.set(true);
                format!("{:?}h", v as f32)
            }
            ScalarTy::F64 => format!("{v:?}lf"),
            _ => format!("{:?}f", v as f32),
        };
        if v.is_sign_negative() { format!("({text})") } else { text }
    }

    // -- Helper functions --

    /// Registers the helper `name` (defined by `define`) and returns its name.
    fn helper(&self, name: String, define: impl FnOnce(&str) -> String) -> String {
        if !self.helpers.borrow().contains_key(&name) {
            // The borrow is released first: `define` may register helpers of its own.
            let text = define(&name);
            self.helpers.borrow_mut().entry(name.clone()).or_insert(text);
        }
        name
    }

    fn type_tag(&self, ty: &str) -> String {
        ident(ty)
    }

    /// Whether elements of `elem` held `stride` bytes apart are further apart than their
    /// natural stride, which WGSL arrays cannot express without a wrapper.
    fn padded_stride(&self, elem: &KirType, stride: u32) -> bool {
        let (size, align) = elem.layout(&self.m.structs);
        stride > size.next_multiple_of(align)
    }

    /// The element type of an array with `stride`: `elem`, or a struct padded to `stride` bytes
    /// holding it as its `value` field.
    fn array_elem(&self, elem: &KirType, stride: u32) -> String {
        let et = self.ty(elem);
        if !self.padded_stride(elem, stride) {
            return et;
        }
        self.helper(format!("LumaPad_{}_{stride}", self.type_tag(&et)), |name| format!("struct {name} {{ @size({stride}) value: {et}, }}\n"))
    }

    /// What turns an element access of the array type `arr` into the element itself.
    fn value_suffix(&self, arr: &KirType) -> &'static str {
        match arr {
            KirType::Array { elem, stride, .. } if self.padded_stride(elem, *stride) => ".value",
            _ => "",
        }
    }

    // -- Types of expressions --

    fn ty_of(&self, sc: &Scope, e: &Expr) -> KirType {
        match e {
            Expr::LitBool(_) => KirType::BOOL,
            Expr::LitU32(_) | Expr::GridSize(_) | Expr::ArrayLength { .. } => KirType::U32,
            Expr::LitI32(_) => KirType::I32,
            Expr::LitF16(_) => KirType::Scalar(ScalarTy::F16),
            Expr::LitF32(_) => KirType::F32,
            Expr::LitF64(_) => KirType::Scalar(ScalarTy::F64),
            Expr::ZeroVal(t) => t.clone(),
            Expr::Local(id) => sc.locals.get(id.0 as usize).map_or(KirType::U32, |l| l.1.clone()),
            Expr::FnParam(i) => sc.params.get(*i as usize).map_or(KirType::U32, |p| p.1.clone()),
            Expr::Private(p) => self.m.privates[*p as usize].ty.clone(),
            Expr::UniformRead { ty, .. }
            | Expr::Load { ty, .. }
            | Expr::StorageRead { ty, .. }
            | Expr::SharedLoad { ty, .. }
            | Expr::SharedRead { ty, .. }
            | Expr::Access { ty, .. }
            | Expr::Swizzle { ty, .. }
            | Expr::Compose { ty, .. }
            | Expr::Unary { ty, .. }
            | Expr::Binary { ty, .. }
            | Expr::Ternary { ty, .. }
            | Expr::Subgroup { ty, .. } => ty.clone(),
            Expr::Splat { elem, lanes, .. } => KirType::vec(*elem, *lanes),
            Expr::Cast { dst_ty, .. } => dst_ty.clone(),
            Expr::Call { ty, .. } => ty.clone().unwrap_or(KirType::U32),
            Expr::AtomicLoad { ty, .. } => KirType::Scalar(*ty),
        }
    }

    // -- Access paths --

    fn step(&self, sc: &Scope, cur: &KirType, step: &AccessStep) -> (String, KirType) {
        match step {
            AccessStep::Field(i) => match cur {
                KirType::Struct(s) => (format!(".{}", self.field_name(*s, *i)), self.m.structs[*s as usize].fields[*i as usize].ty.clone()),
                KirType::Vec { elem, .. } => (format!(".{}", LANES[(*i as usize).min(3)]), KirType::Scalar(*elem)),
                other => (format!(".m{i}"), other.clone()),
            },
            AccessStep::Index(e) => {
                let elem = match cur {
                    KirType::Array { elem, .. } => (**elem).clone(),
                    KirType::Vec { elem, .. } => KirType::Scalar(*elem),
                    other => other.clone(),
                };
                (format!("[{}]{}", self.expr(sc, e), self.value_suffix(cur)), elem)
            }
        }
    }

    fn path(&self, sc: &Scope, mut cur: KirType, path: &[AccessStep]) -> (String, KirType) {
        let mut out = String::new();
        for step in path {
            let (text, next) = self.step(sc, &cur, step);
            out.push_str(&text);
            cur = next;
        }
        (out, cur)
    }

    fn lvalue(&self, sc: &Scope, lv: &LValue) -> (String, KirType) {
        match lv {
            LValue::Local(id) => (local_name(*id, sc.locals), sc.locals[id.0 as usize].1.clone()),
            LValue::Private(p) => (self.private_name(*p), self.m.privates[*p as usize].ty.clone()),
            LValue::Storage { binding, path } => {
                let (text, ty) = self.path(sc, self.binding_root_ty(*binding), path);
                (format!("{}{text}", self.binding_name(*binding)), ty)
            }
            LValue::Shared { shared, path } => {
                let (text, ty) = self.path(sc, self.shared_root_ty(*shared), path);
                (format!("{}{text}", self.shared_name(*shared)), ty)
            }
            LValue::Index { base, index } => {
                let (b, ty) = self.lvalue(sc, base);
                let (text, ty) = self.step(sc, &ty, &AccessStep::Index(index.clone()));
                (format!("{b}{text}"), ty)
            }
            LValue::Field { base, field } => {
                let (b, ty) = self.lvalue(sc, base);
                let (text, ty) = self.step(sc, &ty, &AccessStep::Field(*field));
                (format!("{b}{text}"), ty)
            }
        }
    }

    fn atomic_target(&self, sc: &Scope, t: &AtomicTarget) -> String {
        let (name, index) = match t {
            AtomicTarget::Storage { binding, index } => (self.binding_name(*binding), index),
            AtomicTarget::Shared { shared, index } => (self.shared_name(*shared), index),
        };
        let suffix = match t {
            AtomicTarget::Storage { binding, .. } => self.value_suffix(&self.binding_root_ty(*binding)),
            AtomicTarget::Shared { shared, .. } => self.value_suffix(&self.shared_root_ty(*shared)),
        };
        match index {
            Some(i) => format!("&{name}[{}]{suffix}", self.expr(sc, i)),
            None => format!("&{name}"),
        }
    }

    // -- Expressions --

    fn call(&self, f: &str, args: &[String]) -> String {
        format!("{f}({})", args.join(", "))
    }

    fn expr(&self, sc: &Scope, e: &Expr) -> String {
        match e {
            Expr::LitBool(b) => b.to_string(),
            Expr::LitU32(v) => format!("{v}u"),
            Expr::LitI32(v) => lit_i32(*v),
            Expr::LitF16(v) => self.float_lit(ScalarTy::F16, f64::from(*v)),
            Expr::LitF32(v) => self.float_lit(ScalarTy::F32, f64::from(*v)),
            Expr::LitF64(v) => self.float_lit(ScalarTy::F64, *v),
            Expr::ZeroVal(t) => format!("{}()", self.ty(t)),
            Expr::Local(id) => local_name(*id, sc.locals),
            Expr::FnParam(i) => format!("a{i}"),
            Expr::Private(p) => self.private_name(*p),
            Expr::GridSize(k) => format!("luma.size.{}", LANES[(*k as usize).min(2)]),
            Expr::UniformRead { binding, path, .. } | Expr::StorageRead { binding, path, .. } => {
                let (text, _) = self.path(sc, self.binding_root_ty(*binding), path);
                format!("{}{text}", self.binding_name(*binding))
            }
            Expr::SharedRead { shared, path, .. } => {
                let (text, _) = self.path(sc, self.shared_root_ty(*shared), path);
                format!("{}{text}", self.shared_name(*shared))
            }
            Expr::Load { binding, index, .. } => {
                format!("{}[{}]{}", self.binding_name(*binding), self.expr(sc, index), self.value_suffix(&self.binding_root_ty(*binding)))
            }
            Expr::SharedLoad { shared, index, .. } => {
                format!("{}[{}]{}", self.shared_name(*shared), self.expr(sc, index), self.value_suffix(&self.shared_root_ty(*shared)))
            }
            Expr::Access { base, step, .. } => {
                let base_ty = self.ty_of(sc, base);
                let (text, _) = self.step(sc, &base_ty, step);
                format!("{}{text}", self.expr(sc, base))
            }
            Expr::Swizzle { base, indices, len, .. } => {
                let lanes: String = indices.iter().take(usize::from(*len)).map(|&i| LANES[usize::from(i).min(3)]).collect();
                format!("{}.{lanes}", self.expr(sc, base))
            }
            Expr::Compose { ty, comps } => {
                let wrapper = match ty {
                    KirType::Array { elem, stride, .. } if self.padded_stride(elem, *stride) => Some(self.array_elem(elem, *stride)),
                    _ => None,
                };
                let comps: Vec<String> = comps
                    .iter()
                    .map(|c| {
                        let c = self.expr(sc, c);
                        match &wrapper {
                            Some(w) => format!("{w}({c})"),
                            None => c,
                        }
                    })
                    .collect();
                self.call(&self.ty(ty), &comps)
            }
            Expr::Splat { elem, lanes, val } => format!("vec{lanes}<{}>({})", self.scalar(*elem), self.expr(sc, val)),
            Expr::Unary { op, arg, .. } => self.unary(sc, *op, arg),
            Expr::Binary { op, lhs, rhs, ty } => self.binary(sc, *op, lhs, rhs, ty),
            Expr::Ternary { op, a, b, c, .. } => {
                let (a, b, c) = (self.expr(sc, a), self.expr(sc, b), self.expr(sc, c));
                match op {
                    IrTernaryOp::Fma => format!("fma({a}, {b}, {c})"),
                    IrTernaryOp::Clamp => format!("clamp({a}, {b}, {c})"),
                    IrTernaryOp::Select => format!("select({c}, {b}, {a})"),
                }
            }
            Expr::Cast { kind, expr, src_ty, dst_ty } => {
                let x = self.expr(sc, expr);
                match kind {
                    CastKind::Bitcast => format!("bitcast<{}>({x})", self.ty(dst_ty)),
                    CastKind::Convert => format!("{}({x})", self.ty(dst_ty)),
                    CastKind::FloatToIntSat => self.float_to_int_sat(&x, src_ty, dst_ty),
                }
            }
            Expr::Call { func, args, .. } => {
                let args: Vec<String> = args.iter().map(|a| self.expr(sc, a)).collect();
                self.call(&self.function_name(*func), &args)
            }
            Expr::AtomicLoad { target, ty } => {
                let load = format!("atomicLoad({})", self.atomic_target(sc, target));
                if *ty == ScalarTy::F32 { format!("bitcast<f32>({load})") } else { load }
            }
            Expr::Subgroup { op, arg, lane, .. } => {
                let a = arg.as_ref().map(|a| self.expr(sc, a));
                match op {
                    IrSubgroupOp::Ballot => format!("subgroupBallot({})", a.unwrap_or_else(|| "true".to_owned())),
                    IrSubgroupOp::Add => format!("subgroupAdd({})", a.unwrap_or_default()),
                    IrSubgroupOp::Min => format!("subgroupMin({})", a.unwrap_or_default()),
                    IrSubgroupOp::Max => format!("subgroupMax({})", a.unwrap_or_default()),
                    IrSubgroupOp::ExclusiveAdd => format!("subgroupExclusiveAdd({})", a.unwrap_or_default()),
                    IrSubgroupOp::Shuffle => match lane {
                        Some(l) => format!("subgroupShuffle({}, {})", a.unwrap_or_default(), self.expr(sc, l)),
                        None => format!("subgroupBroadcastFirst({})", a.unwrap_or_default()),
                    },
                }
            }
            Expr::ArrayLength { binding } => format!("arrayLength(&{})", self.binding_name(*binding)),
        }
    }

    /// An exact-rounding barrier: the value is passed through a bit-or with a zero the compiler
    /// does not know (`luma._pad`), so no pass reassociates, contracts or folds across it.
    fn guard(&self, x: &str, ty: &KirType) -> String {
        let t = self.ty(ty);
        let u = self.uint_like(ty);
        let pad = if ty.lanes() == 1 { "luma._pad".to_owned() } else { format!("{u}(luma._pad)") };
        format!("bitcast<{t}>(bitcast<{u}>({x}) | {pad})")
    }

    /// A float to `i32`/`u32` conversion that saturates and maps NaN to 0, as a helper function
    /// per (source, destination) type so the operand is evaluated once and vectors work lane by
    /// lane. NaN is found from the bits of the value (of its scale into `f32` for `f64`), made
    /// opaque with `luma._pad` so no compiler folds the test away under fast-math; the value is
    /// then clamped to the range the conversion is defined on, and what lies beyond the largest
    /// representable float below the limit gets the integer limit itself. Invalid source or
    /// destination types violate the KIR construction invariant, so this emitter rejects them
    /// before producing shader text rather than leaving an invalid expression for the compiler.
    fn float_to_int_sat(&self, x: &str, src: &KirType, dst: &KirType) -> String {
        let (st, dt) = (self.ty(src), self.ty(dst));
        let invalid = || panic!("luma-wgpu codegen invariant: invalid saturating conversion from {src:?} to {dst:?}");
        let shaped = |t: &KirType| matches!(t, KirType::Scalar(_) | KirType::Vec { .. });
        let (Some(se), Some(de)) = (
            src.elem_scalar().filter(|s| s.is_float()),
            dst.elem_scalar().filter(|d| matches!(d, ScalarTy::U32 | ScalarTy::I32)),
        ) else {
            return invalid();
        };
        if !shaped(src) || !shaped(dst) || src.lanes() != dst.lanes() {
            return invalid();
        }
        let lanes = src.lanes();
        let f32t = if lanes == 1 { "f32".to_owned() } else { format!("vec{lanes}<f32>") };
        let ut = self.uint_like(src);
        let signed = de == ScalarTy::I32;
        let name = self.helper(format!("luma_sat_{}_{}", self.type_tag(&st), self.type_tag(&dt)), |name| {
            let pad = if lanes == 1 { "luma._pad".to_owned() } else { format!("{ut}(luma._pad)") };
            // `probe` is the `f32` whose bits classify the value, `value` what is converted.
            let (pre, probe, value) = match se {
                ScalarTy::F16 => (format!("    let y = {f32t}(x);\n"), "y", "y"),
                ScalarTy::F64 => {
                    let scale = self.float_lit(ScalarTy::F64, 9.332636185032189e-302);
                    (format!("    let y = {f32t}(x * {st}({scale}));\n"), "y", "x")
                }
                _ => (String::new(), "x", "x"),
            };
            let nan = format!("((bitcast<{ut}>({probe}) | {pad}) & {ut}(0x7fffffffu)) > {ut}(0x7f800000u)");
            let body = if se == ScalarTy::F64 {
                let (lo, hi) = if signed { (-2147483648.0, 2147483647.0) } else { (0.0, 4294967295.0) };
                let lit = |v: f64| format!("{st}({})", self.float_lit(ScalarTy::F64, v));
                format!(
                    "    let nan = {nan};\n    let s = select(x, {z}, nan);\n    return {dt}(clamp(s, {lo}, {hi}));\n",
                    z = lit(0.0),
                    lo = lit(lo),
                    hi = lit(hi)
                )
            } else {
                let (lo, hi, big, max) = if signed {
                    (-2147483648.0, 2147483520.0, 2147483648.0, "2147483647i")
                } else {
                    (0.0, 4294967040.0, 4294967296.0, "4294967295u")
                };
                let lit = |v: f64| format!("{f32t}({})", self.float_lit(ScalarTy::F32, v));
                format!(
                    "    let nan = {nan};\n    let s = select({value}, {z}, nan);\n    let r = {dt}(clamp(s, {lo}, {hi}));\n    return select(r, {dt}({max}), s >= {big});\n",
                    z = lit(0.0),
                    lo = lit(lo),
                    hi = lit(hi),
                    big = lit(big)
                )
            };
            format!("fn {name}(x: {st}) -> {dt} {{\n{pre}{body}}}\n")
        });
        format!("{name}({x})")
    }

    fn unary(&self, sc: &Scope, op: IrUnOp, arg: &Expr) -> String {
        let a = self.expr(sc, arg);
        let arg_ty = self.ty_of(sc, arg);
        let elem = arg_ty.elem_scalar().unwrap_or(ScalarTy::F32);
        let plain = |f: &str| format!("{f}({a})");
        match op {
            IrUnOp::Neg => {
                if elem == ScalarTy::U32 {
                    format!("({}(0u) - {a})", self.ty(&arg_ty))
                } else {
                    format!("(-{a})")
                }
            }
            IrUnOp::Not => format!("(!{a})"),
            IrUnOp::BitNot => format!("(~{a})"),
            IrUnOp::Abs => plain("abs"),
            IrUnOp::Sqrt => plain("sqrt"),
            IrUnOp::Exp => plain("exp"),
            IrUnOp::Ln => plain("log"),
            IrUnOp::Sin => plain("sin"),
            IrUnOp::Cos => plain("cos"),
            IrUnOp::Tan => plain("tan"),
            IrUnOp::Tanh => {
                let t = self.ty(&arg_ty);
                let name = self.helper(format!("luma_tanh_{}", self.type_tag(&t)), |name| {
                    let lit = |v: f64| format!("{t}({})", self.float_lit(elem, v));
                    format!(
                        "fn {name}(x: {t}) -> {t} {{ return select(tanh(x), select({}, {}, x > {}), abs(x) > {}); }}\n",
                        lit(-1.0),
                        lit(1.0),
                        lit(0.0),
                        lit(15.0)
                    )
                });
                format!("{name}({a})")
            }
            IrUnOp::Sign => plain("sign"),
            IrUnOp::Floor => plain("floor"),
            IrUnOp::Ceil => plain("ceil"),
            IrUnOp::Trunc if elem == ScalarTy::F64 => {
                // Exact on the magnitude, the sign put back, and a zero (either sign) or NaN left
                // as it is; in (-1, 0) the result is -0.
                let t = self.ty(&arg_ty);
                let name = self.helper(format!("luma_trunc_{}", self.type_tag(&t)), |name| {
                    let zero = format!("{t}({})", self.float_lit(elem, 0.0));
                    format!("fn {name}(x: {t}) -> {t} {{ return select(x, select(floor(abs(x)), -floor(abs(x)), x < {zero}), x != {zero}); }}\n")
                });
                format!("{name}({a})")
            }
            IrUnOp::Round if elem == ScalarTy::F64 => {
                // Half to even on the magnitude, with the sign and zeros handled as for trunc.
                let t = self.ty(&arg_ty);
                let name = self.helper(format!("luma_round_{}", self.type_tag(&t)), |name| {
                    let lit = |v: f64| format!("{t}({})", self.float_lit(elem, v));
                    format!(
                        "fn {name}(x: {t}) -> {t} {{\n    let a = abs(x);\n    let f = floor(a);\n    let d = a - f;\n    let up = (d > {h}) | ((d == {h}) & ((f - {two} * floor(f * {h})) == {one}));\n    let r = f + select({z}, {one}, up);\n    return select(x, select(r, -r, x < {z}), x != {z});\n}}\n",
                        h = lit(0.5),
                        two = lit(2.0),
                        one = lit(1.0),
                        z = lit(0.0)
                    )
                });
                format!("{name}({a})")
            }
            IrUnOp::Trunc => plain("trunc"),
            IrUnOp::Round => plain("round"),
            IrUnOp::CountOneBits => plain("countOneBits"),
            IrUnOp::FirstTrailingBit => plain("firstTrailingBit"),
            IrUnOp::FirstLeadingBit => plain("firstLeadingBit"),
            IrUnOp::CountLeadingZeros => plain("countLeadingZeros"),
            IrUnOp::CountTrailingZeros => plain("countTrailingZeros"),
            IrUnOp::IsNan => {
                if elem == ScalarTy::F32 {
                    let u = self.uint_like(&arg_ty);
                    format!("((bitcast<{u}>({a}) & {u}(0x7fffffffu)) > {u}(0x7f800000u))")
                } else {
                    format!("({a} != {a})")
                }
            }
            IrUnOp::IsInf => {
                if elem == ScalarTy::F32 {
                    let u = self.uint_like(&arg_ty);
                    format!("((bitcast<{u}>({a}) & {u}(0x7fffffffu)) == {u}(0x7f800000u))")
                } else {
                    let t = self.ty(&arg_ty);
                    let max = if elem == ScalarTy::F64 { "1.7976931348623157e308lf" } else { "65504.0h" };
                    format!("(abs({a}) > {t}({max}))")
                }
            }
            IrUnOp::OpaqueGuard => self.guard(&a, &arg_ty),
            IrUnOp::InverseSqrt => plain("inverseSqrt"),
            IrUnOp::Exp2 => plain("exp2"),
            IrUnOp::Log2 => plain("log2"),
            IrUnOp::Asin => plain("asin"),
            IrUnOp::Acos => plain("acos"),
            IrUnOp::Atan => plain("atan"),
            IrUnOp::All => plain("all"),
            IrUnOp::Any => plain("any"),
        }
    }

    fn binary(&self, sc: &Scope, op: IrBinOp, lhs: &Expr, rhs: &Expr, ty: &KirType) -> String {
        let l = self.expr(sc, lhs);
        let r = self.expr(sc, rhs);
        let infix = |sym: &str| format!("({l} {sym} {r})");
        let lhs_ty = self.ty_of(sc, lhs);
        match op {
            IrBinOp::Add => infix("+"),
            IrBinOp::Sub => infix("-"),
            IrBinOp::Mul => infix("*"),
            IrBinOp::Div => infix("/"),
            IrBinOp::Rem => infix("%"),
            IrBinOp::AddExact => self.guard(&infix("+"), ty),
            IrBinOp::SubExact => self.guard(&infix("-"), ty),
            IrBinOp::MulExact => self.guard(&infix("*"), ty),
            IrBinOp::Min => format!("min({l}, {r})"),
            IrBinOp::Max => format!("max({l}, {r})"),
            IrBinOp::Pow => format!("pow({l}, {r})"),
            IrBinOp::Atan2 => format!("atan2({l}, {r})"),
            IrBinOp::Copysign => {
                let t = self.ty(ty);
                let elem = ty.elem_scalar().unwrap_or(ScalarTy::F32);
                let name = self.helper(format!("luma_copysign_{}", self.type_tag(&t)), |name| {
                    if elem == ScalarTy::F32 {
                        let u = self.uint_like(ty);
                        format!(
                            "fn {name}(a: {t}, b: {t}) -> {t} {{ return select(abs(a), -abs(a), (bitcast<{u}>(b) & {u}(0x80000000u)) != {u}(0u)); }}\n"
                        )
                    } else {
                        let lit = |v: f64| format!("{t}({})", self.float_lit(elem, v));
                        format!(
                            "fn {name}(a: {t}, b: {t}) -> {t} {{ return select(abs(a), -abs(a), (b < {z}) | ((b == {z}) & (({o} / b) < {z}))); }}\n",
                            z = lit(0.0),
                            o = lit(1.0)
                        )
                    }
                });
                format!("{name}({l}, {r})")
            }
            IrBinOp::BitAnd => infix("&"),
            IrBinOp::BitOr => infix("|"),
            IrBinOp::BitXor => infix("^"),
            IrBinOp::Shl | IrBinOp::Shr => {
                let count_ty = self.ty_of(sc, rhs);
                let want = self.uint_like(&lhs_ty);
                let r = if count_ty.elem_scalar() == Some(ScalarTy::U32) && count_ty.lanes() == lhs_ty.lanes() { r } else { format!("{want}({r})") };
                format!("({l} {} {r})", if op == IrBinOp::Shl { "<<" } else { ">>" })
            }
            IrBinOp::Rotl => {
                let u = self.uint_like(&lhs_ty);
                let name = self.helper(format!("luma_rotl_{}", self.type_tag(&u)), |name| {
                    format!(
                        "fn {name}(a: {u}, n: {u}) -> {u} {{ return (a << (n & {u}(31u))) | (a >> ((({u}(32u)) - n) & {u}(31u))); }}\n"
                    )
                });
                let count = format!("{u}({r})");
                if lhs_ty.elem_scalar() == Some(ScalarTy::U32) {
                    format!("{name}({l}, {count})")
                } else {
                    format!("bitcast<{}>({name}(bitcast<{u}>({l}), {count}))", self.ty(&lhs_ty))
                }
            }
            IrBinOp::MulHiU32 => {
                let name = self.helper("luma_mulhi".to_owned(), |name| {
                    format!(
                        "fn {name}(a: u32, b: u32) -> u32 {{\n    let a0 = a & 0xFFFFu; let a1 = a >> 16u;\n    let b0 = b & 0xFFFFu; let b1 = b >> 16u;\n    let m1 = a1 * b0; let m2 = a0 * b1;\n    let carry = (((a0 * b0) >> 16u) + (m1 & 0xFFFFu) + (m2 & 0xFFFFu)) >> 16u;\n    return a1 * b1 + (m1 >> 16u) + (m2 >> 16u) + carry;\n}}\n"
                    )
                });
                format!("{name}({l}, {r})")
            }
            IrBinOp::Eq => infix("=="),
            IrBinOp::Ne => infix("!="),
            IrBinOp::Lt => infix("<"),
            IrBinOp::Le => infix("<="),
            IrBinOp::Gt => infix(">"),
            IrBinOp::Ge => infix(">="),
            IrBinOp::And => infix(if ty.lanes() == 1 { "&&" } else { "&" }),
            IrBinOp::Or => infix(if ty.lanes() == 1 { "||" } else { "|" }),
        }
    }

    // -- Statements --

    fn temp(&self, prefix: &str) -> String {
        let n = self.temps.get();
        self.temps.set(n + 1);
        format!("luma_{prefix}{n}")
    }

    fn block(&self, sc: &Scope, stmts: &[Stmt], ind: usize, out: &mut String) {
        for s in stmts {
            self.stmt(sc, s, ind, out);
        }
    }

    fn stmt(&self, sc: &Scope, s: &Stmt, ind: usize, out: &mut String) {
        let pad = "    ".repeat(ind);
        match s {
            Stmt::Let { local, init } => {
                let _ = writeln!(out, "{pad}let {} = {};", local_name(*local, sc.locals), self.expr(sc, init));
            }
            Stmt::Var { local, init } => {
                let name = local_name(*local, sc.locals);
                let ty = self.ty(&sc.locals[local.0 as usize].1);
                match init {
                    Some(init) => {
                        let _ = writeln!(out, "{pad}var {name}: {ty} = {};", self.expr(sc, init));
                    }
                    None => {
                        let _ = writeln!(out, "{pad}var {name}: {ty};");
                    }
                }
            }
            Stmt::Assign { dst, val } => {
                let (lv, _) = self.lvalue(sc, dst);
                let _ = writeln!(out, "{pad}{lv} = {};", self.expr(sc, val));
            }
            Stmt::Store { binding, index, val } => {
                let suffix = self.value_suffix(&self.binding_root_ty(*binding));
                let _ = writeln!(out, "{pad}{}[{}]{suffix} = {};", self.binding_name(*binding), self.expr(sc, index), self.expr(sc, val));
            }
            Stmt::SharedStore { shared, index, val } => {
                let suffix = self.value_suffix(&self.shared_root_ty(*shared));
                let _ = writeln!(out, "{pad}{}[{}]{suffix} = {};", self.shared_name(*shared), self.expr(sc, index), self.expr(sc, val));
            }
            Stmt::If { cond, then_body, else_body } => {
                let _ = writeln!(out, "{pad}if ({}) {{", self.expr(sc, cond));
                self.block(sc, then_body, ind + 1, out);
                if else_body.is_empty() {
                    let _ = writeln!(out, "{pad}}}");
                } else {
                    let _ = writeln!(out, "{pad}}} else {{");
                    self.block(sc, else_body, ind + 1, out);
                    let _ = writeln!(out, "{pad}}}");
                }
            }
            Stmt::Loop { body, continuing, break_if } => {
                let _ = writeln!(out, "{pad}loop {{");
                self.block(sc, body, ind + 1, out);
                if !continuing.is_empty() || break_if.is_some() {
                    let _ = writeln!(out, "{pad}    continuing {{");
                    self.block(sc, continuing, ind + 2, out);
                    if let Some(cond) = break_if {
                        let _ = writeln!(out, "{pad}        break if ({});", self.expr(sc, cond));
                    }
                    let _ = writeln!(out, "{pad}    }}");
                }
                let _ = writeln!(out, "{pad}}}");
            }
            Stmt::Switch { selector, cases, default } => {
                let _ = writeln!(out, "{pad}switch ({}) {{", self.expr(sc, selector));
                // WGSL has no fall-through: cases that fall into the next share its clause, and a
                // case with code that falls through runs that code first in the same clause.
                let mut selectors: Vec<String> = Vec::new();
                let mut bodies: Vec<&[Stmt]> = Vec::new();
                for (value, body, fall_through) in cases {
                    selectors.push(match value {
                        SwitchCase::I32(v) => lit_i32(*v),
                        SwitchCase::U32(v) => format!("{v}u"),
                    });
                    bodies.push(body);
                    if !*fall_through {
                        self.clause(sc, &selectors, false, &bodies, ind + 1, out);
                        selectors.clear();
                        bodies.clear();
                    }
                }
                if selectors.is_empty() {
                    self.clause(sc, &selectors, true, &[default.as_slice()], ind + 1, out);
                } else {
                    bodies.push(default);
                    self.clause(sc, &selectors, true, &bodies, ind + 1, out);
                }
                let _ = writeln!(out, "{pad}}}");
            }
            Stmt::Break => {
                let _ = writeln!(out, "{pad}break;");
            }
            Stmt::Continue => {
                let _ = writeln!(out, "{pad}continue;");
            }
            Stmt::Return(None) => {
                let _ = writeln!(out, "{pad}return;");
            }
            Stmt::Return(Some(e)) => {
                let _ = writeln!(out, "{pad}return {};", self.expr(sc, e));
            }
            Stmt::Barrier(scope) => match scope {
                BarrierScope::Workgroup => {
                    let _ = writeln!(out, "{pad}workgroupBarrier();");
                }
                BarrierScope::Storage => {
                    let _ = writeln!(out, "{pad}storageBarrier();");
                }
                BarrierScope::StorageAndWorkgroup => {
                    let _ = writeln!(out, "{pad}storageBarrier();\n{pad}workgroupBarrier();");
                }
            },
            Stmt::AtomicStore { target, val } => {
                let ty = self.ty_of(sc, val);
                let mut v = self.expr(sc, val);
                if ty.elem_scalar() == Some(ScalarTy::F32) {
                    v = format!("bitcast<u32>({v})");
                }
                let _ = writeln!(out, "{pad}atomicStore({}, {v});", self.atomic_target(sc, target));
            }
            Stmt::AtomicRmw { dst, op, elem_ty, target, val } => self.atomic_rmw(sc, *dst, *op, *elem_ty, target, val, &pad, out),
            Stmt::AtomicCompareExchange { dst_old, dst_exchanged, elem_ty, target, cmp, val } => {
                let float = *elem_ty == ScalarTy::F32;
                let (mut c, mut v) = (self.expr(sc, cmp), self.expr(sc, val));
                if float {
                    c = format!("bitcast<u32>({c})");
                    v = format!("bitcast<u32>({v})");
                }
                let r = self.temp("cas");
                let _ = writeln!(out, "{pad}let {r} = atomicCompareExchangeWeak({}, {c}, {v});", self.atomic_target(sc, target));
                let old = if float { format!("bitcast<f32>({r}.old_value)") } else { format!("{r}.old_value") };
                let _ = writeln!(out, "{pad}let {} = {old};", local_name(*dst_old, sc.locals));
                let _ = writeln!(out, "{pad}let {} = {r}.exchanged;", local_name(*dst_exchanged, sc.locals));
            }
            Stmt::Discard(e) => match e {
                Expr::Call { ty: None, .. } => {
                    let _ = writeln!(out, "{pad}{};", self.expr(sc, e));
                }
                _ => {
                    let _ = writeln!(out, "{pad}_ = {};", self.expr(sc, e));
                }
            },
        }
    }

    fn clause(&self, sc: &Scope, selectors: &[String], with_default: bool, bodies: &[&[Stmt]], ind: usize, out: &mut String) {
        let pad = "    ".repeat(ind);
        let mut labels: Vec<&str> = selectors.iter().map(String::as_str).collect();
        if with_default {
            labels.push("default");
        }
        let label = if labels.len() == 1 && with_default { "default".to_owned() } else { format!("case {}", labels.join(", ")) };
        let _ = writeln!(out, "{pad}{label}: {{");
        for body in bodies {
            self.block(sc, body, ind + 1, out);
        }
        let _ = writeln!(out, "{pad}}}");
    }

    #[allow(clippy::too_many_arguments)]
    fn atomic_rmw(&self, sc: &Scope, dst: Option<LocalId>, op: IrAtomicOp, elem_ty: ScalarTy, target: &AtomicTarget, val: &Expr, pad: &str, out: &mut String) {
        let t = self.atomic_target(sc, target);
        let v = self.expr(sc, val);
        let direct = |name: &str, value: &str| format!("{name}({t}, {value})");
        let fun = |op: IrAtomicOp| match op {
            IrAtomicOp::Add => "atomicAdd",
            IrAtomicOp::Sub => "atomicSub",
            IrAtomicOp::Min => "atomicMin",
            IrAtomicOp::Max => "atomicMax",
            IrAtomicOp::And => "atomicAnd",
            IrAtomicOp::Or => "atomicOr",
            IrAtomicOp::Xor => "atomicXor",
            IrAtomicOp::Exchange => "atomicExchange",
        };
        let float = elem_ty == ScalarTy::F32;
        let native = matches!(target, AtomicTarget::Storage { binding, .. } if self.hardware[*binding as usize]);
        if !float || native {
            let call = direct(fun(op), &v);
            match dst {
                Some(d) => {
                    let _ = writeln!(out, "{pad}let {} = {call};", local_name(d, sc.locals));
                }
                None => {
                    let _ = writeln!(out, "{pad}_ = {call};");
                }
            }
            return;
        }
        // The element holds the bits of an `f32`.
        let combine = match op {
            IrAtomicOp::Add => Some("bitcast<f32>(old) + v"),
            IrAtomicOp::Sub => Some("bitcast<f32>(old) - v"),
            IrAtomicOp::Min => Some("min(bitcast<f32>(old), v)"),
            IrAtomicOp::Max => Some("max(bitcast<f32>(old), v)"),
            _ => None,
        };
        let Some(combine) = combine else {
            // Bit operations and exchange work on the bits as they are.
            let call = direct(fun(op), &format!("bitcast<u32>({v})"));
            match dst {
                Some(d) => {
                    let _ = writeln!(out, "{pad}let {} = bitcast<f32>({call});", local_name(d, sc.locals));
                }
                None => {
                    let _ = writeln!(out, "{pad}_ = {call};");
                }
            }
            return;
        };
        let dst_name = dst.map(|d| local_name(d, sc.locals));
        if let Some(name) = &dst_name {
            let _ = writeln!(out, "{pad}var {name}: f32;");
        }
        let stable = matches!(op, IrAtomicOp::Min | IrAtomicOp::Max);
        let _ = writeln!(out, "{pad}{{");
        let _ = writeln!(out, "{pad}    let v = {v};");
        let _ = writeln!(out, "{pad}    let p = {t};");
        let _ = writeln!(out, "{pad}    var old = atomicLoad(p);");
        let _ = writeln!(out, "{pad}    loop {{");
        let _ = writeln!(out, "{pad}        let nv = bitcast<u32>({combine});");
        if stable {
            let _ = writeln!(out, "{pad}        if (nv == old) {{ break; }}");
        }
        let _ = writeln!(out, "{pad}        let r = atomicCompareExchangeWeak(p, old, nv);");
        let _ = writeln!(out, "{pad}        if (r.exchanged) {{ break; }}");
        let _ = writeln!(out, "{pad}        old = r.old_value;");
        let _ = writeln!(out, "{pad}    }}");
        if let Some(name) = &dst_name {
            let _ = writeln!(out, "{pad}    {name} = bitcast<f32>(old);");
        }
        let _ = writeln!(out, "{pad}}}");
    }

    // -- Declarations --

    fn struct_decl(&self, i: u32) -> String {
        let s = &self.m.structs[i as usize];
        let mut out = format!("struct {} {{\n", self.struct_name(i));
        let n = s.fields.len();
        for (k, f) in s.fields.iter().enumerate() {
            let (size, _) = f.ty.layout(&self.m.structs);
            let end = if k + 1 < n { s.fields[k + 1].offset } else { s.size };
            let mut attrs = String::new();
            if k == 0 && s.align > s.fields.iter().map(|f| f.ty.layout(&self.m.structs).1).max().unwrap_or(1) {
                let _ = write!(attrs, "@align({}) ", s.align);
            }
            if end > f.offset && end - f.offset != size && end - f.offset > size {
                let _ = write!(attrs, "@size({}) ", end - f.offset);
            }
            let _ = writeln!(out, "    {attrs}{}: {},", self.field_name(i, k as u32), self.ty(&f.ty));
        }
        out.push_str("}\n");
        out
    }

    /// Struct indices in an order where each comes after the structs it holds.
    fn struct_order(&self) -> Vec<u32> {
        fn visit(m: &KernelModule, t: &KirType, seen: &mut Vec<bool>, out: &mut Vec<u32>) {
            match t {
                KirType::Struct(i) => {
                    if seen[*i as usize] {
                        return;
                    }
                    seen[*i as usize] = true;
                    for f in &m.structs[*i as usize].fields {
                        visit(m, &f.ty, seen, out);
                    }
                    out.push(*i);
                }
                KirType::Array { elem, .. } => visit(m, elem, seen, out),
                _ => {}
            }
        }
        let mut seen = vec![false; self.m.structs.len()];
        let mut out = Vec::new();
        for i in 0..self.m.structs.len() {
            visit(self.m, &KirType::Struct(i as u32), &mut seen, &mut out);
        }
        out
    }

    fn function(&self, f: &KirFunction, id: usize) -> String {
        let sc = Scope { locals: &f.locals, params: &f.params };
        let params: Vec<String> = f.params.iter().enumerate().map(|(i, (_, t))| format!("a{i}: {}", self.ty(t))).collect();
        let name = format!("f{id}_{}", ident(&f.name));
        let mut out = match &f.ret_ty {
            Some(r) => format!("fn {name}({}) -> {} {{\n", params.join(", "), self.ty(r)),
            None => format!("fn {name}({}) {{\n", params.join(", ")),
        };
        self.block(&sc, &f.body, 1, &mut out);
        out.push_str("}\n");
        out
    }

    fn entry(&self) -> String {
        let m = self.m;
        let sc = Scope { locals: &m.entry_locals, params: &[] };
        let params: Vec<String> = m
            .entry_builtins
            .iter()
            .map(|(var, local)| {
                let attr = match var {
                    BuiltinVar::GlobalInvocationId => "global_invocation_id",
                    BuiltinVar::LocalInvocationId => "local_invocation_id",
                    BuiltinVar::LocalInvocationIndex => "local_invocation_index",
                    BuiltinVar::WorkgroupId => "workgroup_id",
                    BuiltinVar::NumWorkgroups => "num_workgroups",
                    BuiltinVar::SubgroupInvocationId => "subgroup_invocation_id",
                    BuiltinVar::SubgroupSize => "subgroup_size",
                };
                format!("@builtin({attr}) {}: {}", local_name(*local, &m.entry_locals), self.ty(&var.ty()))
            })
            .collect();
        let [x, y, z] = m.workgroup_size;
        let mut out = format!("@compute @workgroup_size({x}, {y}, {z})\nfn {ENTRY}({}) {{\n", params.join(", "));
        self.block(&sc, &m.entry_body, 1, &mut out);
        out.push_str("}\n");
        out
    }

    fn module(&self, used: Option<&[bool]>) -> String {
        let m = self.m;
        // The mask is authoritative: a binding it leaves out is not declared, so code that
        // reaches one is a compile error rather than a layout mismatch.
        let declared = |i: usize| used.is_none_or(|u| u.get(i).copied().unwrap_or(false));

        // Bodies first: they decide which helpers and extensions the module needs.
        let mut body = String::new();
        for i in self.struct_order() {
            body.push_str(&self.struct_decl(i));
        }
        for (i, d) in m.bindings.iter().enumerate() {
            if !declared(i) {
                continue;
            }
            let space = match d.kind {
                BindingKind::Uniform => "uniform",
                BindingKind::StorageRead => "storage, read",
                BindingKind::StorageReadWrite => "storage, read_write",
            };
            let mut ty = self.ty(&self.binding_root_ty_decl(d));
            if self.hardware[i] {
                ty = ty.replace("atomic<u32>", "atomic<f32>");
            }
            let _ = writeln!(body, "@group(1) @binding({i}) var<{space}> {}: {ty};", self.binding_name(i as u32));
        }
        for i in 0..m.shared.len() {
            let ty = self.ty(&self.shared_root_ty(i as u32));
            let _ = writeln!(body, "var<workgroup> {}: {ty};", self.shared_name(i as u32));
        }
        for (i, p) in m.privates.iter().enumerate() {
            let sc = Scope { locals: &[], params: &[] };
            match &p.init {
                Some(init) => {
                    let _ = writeln!(body, "var<private> {}: {} = {};", self.private_name(i as u32), self.ty(&p.ty), self.expr(&sc, init));
                }
                None => {
                    let _ = writeln!(body, "var<private> {}: {};", self.private_name(i as u32), self.ty(&p.ty));
                }
            }
        }
        let mut functions = String::new();
        let live = m.reachable_functions();
        for (i, f) in m.functions.iter().enumerate() {
            if live[i] {
                functions.push_str(&self.function(f, i));
            }
        }
        let entry = self.entry();

        let mut out = String::new();
        if self.f16.get() {
            out.push_str("enable f16;\n");
        }
        out.push_str("struct LumaGrid { size: vec3<u32>, _pad: u32 }\n@group(0) @binding(0) var<uniform> luma: LumaGrid;\n");
        out.push_str(&body);
        for text in self.helpers.borrow().values() {
            out.push_str(text);
        }
        out.push_str(&functions);
        out.push_str(&entry);
        out
    }

    fn binding_root_ty_decl(&self, d: &luma_core::kir::BindingDecl) -> KirType {
        if d.is_unsized_array {
            KirType::Array { elem: Box::new(d.elem_ty.clone()), len: 0, stride: d.element_stride.unwrap_or(0) }
        } else {
            d.elem_ty.clone()
        }
    }
}

#[cfg(test)]
mod tests {
    use luma_core::hal::Features;

    use super::*;

    fn caps() -> naga::valid::Capabilities {
        use naga::valid::Capabilities as C;
        C::SUBGROUP | C::SUBGROUP_BARRIER | C::SHADER_FLOAT16 | C::FLOAT64
    }

    /// Parses and validates `src` with naga, as the device would.
    fn validate(src: &str) -> naga::Module {
        let module = naga::front::wgsl::parse_str(src).unwrap_or_else(|e| panic!("{}\n{src}", e.emit_to_string(src)));
        naga::valid::Validator::new(naga::valid::ValidationFlags::all(), caps())
            .validate(&module)
            .unwrap_or_else(|e| panic!("{}\n{src}", e.emit_to_string(src)));
        module
    }

    type Parsed = (naga::Module, KernelModule);

    fn parse(src: &str, trusted: bool) -> Parsed {
        let (naga, m, ..) = parse_wgsl(src, "main", None, Features::SUBGROUP | Features::SHADER_F16, [256, 256, 64], 256, 8, trusted)
            .unwrap_or_else(|e| panic!("{e:?}\n{src}"));
        (naga, m)
    }

    fn round_trip(src: &str) -> String {
        let (_, m) = parse(src, false);
        let out = emit(&m);
        validate(&out);
        out
    }

    #[test]
    fn saxpy_round_trips_with_original_binding_numbers() {
        let out = round_trip(
            "struct P { a: f32, n: u32 }
@group(1) @binding(0) var<storage, read> x: array<f32>;
@group(1) @binding(1) var<storage, read_write> y: array<f32>;
@group(1) @binding(2) var<uniform> p: P;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p.n) { return; }
    y[i] = p.a * x[i] + y[i];
}",
        );
        assert!(out.contains("@group(0) @binding(0) var<uniform> luma: LumaGrid;"));
        assert!(out.contains("@group(1) @binding(2) var<uniform>"));
        assert!(out.contains("@workgroup_size(64, 1, 1)"));
    }

    #[test]
    fn math_vectors_arrays_and_helpers() {
        round_trip(
            "fn f(x: f32) -> array<f32, 4> { return array<f32, 4>(x, x + 1.0, x + 2.0, x + 3.0); }
@group(1) @binding(0) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(1) fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    let a = f(1.0);
    let v = vec3<f32>(1.0, 2.0, 3.0);
    var w = vec4<f32>(0.0);
    w.y = 3.0;
    out[luma_idx2(g.x, g.y)] = a[3] + dot(v, v) + select(1.0, 2.0, a[0] > 0.0) + f32(u32(a[1])) + tanh(a[2]) + w.y
        + length(v) + mix(1.0f, 2.0f, 0.25f) + smoothstep(0.0f, 1.0f, 0.5f) + fract(1.5f) + f32(luma.size.y) + f32(arrayLength(&out));
}",
        );
    }

    #[test]
    fn integer_ops_and_shifts() {
        round_trip(
            "@group(1) @binding(0) var<storage, read_write> out: array<u32>;
@compute @workgroup_size(1) fn main() {
    var a = 5u; var b = 3i;
    a = (a << 3u) | (a >> 1u) ^ (a & 7u);
    b = b / 2i + b % 3i - -b;
    out[0] = a + u32(b) + countOneBits(a) + firstTrailingBit(a) + firstLeadingBit(a) + countLeadingZeros(a) + countTrailingZeros(a);
    let v = vec2<u32>(1u, 2u) << vec2<u32>(1u, 1u);
    out[1] = v.x + v.y + bitcast<u32>(1.5) + u32(any(vec2<bool>(true, false))) + u32(all(vec2<bool>(true, false)));
}",
        );
    }

    #[test]
    fn loops_continuing_and_switch() {
        let out = round_trip(
            "@group(1) @binding(0) var<storage, read_write> out: array<u32>;
@compute @workgroup_size(1) fn main() {
    var i = 0u;
    loop {
        if (i >= 10u) { break; }
        switch (i) {
            case 1u, 2u: { out[0] = 1u; }
            case 3u: { out[0] = 3u; }
            default: { out[0] = 9u; }
        }
        continuing {
            i = i + 1u;
            break if i > 20u;
        }
    }
    for (var j = 0u; j < 4u; j++) {
        if (j == 2u) { continue; }
        out[1] = j;
    }
    while (i < 100u) { i = i + 7u; }
}",
        );
        assert!(out.contains("continuing"));
        assert!(out.contains("break if"));
        assert!(out.contains("case 1u, 2u:"));
    }

    #[test]
    fn atomics_shared_memory_and_subgroups() {
        let out = round_trip(
            "var<workgroup> hist: array<atomic<u32>, 32>;
@group(1) @binding(0) var<storage, read_write> out: array<u32>;
@group(1) @binding(1) var<storage, read_write> acc: array<atomic<u32>>;
@compute @workgroup_size(32) fn main(@builtin(local_invocation_index) li: u32) {
    atomicStore(&hist[li], 0u);
    workgroupBarrier();
    storageBarrier();
    let old = atomicAdd(&hist[li % 4u], 1u);
    let ballot = subgroupBallot(li < 5u);
    let sh = subgroupShuffle(old, 0u);
    let first = subgroupBroadcastFirst(old);
    let sum = subgroupAdd(old) + subgroupExclusiveAdd(old) + subgroupMin(old) + subgroupMax(old);
    let cas = atomicCompareExchangeWeak(&acc[0], old, 7u);
    if (cas.exchanged) { out[1] = cas.old_value; }
    out[li] = ballot.x + sh + sum + first + atomicLoad(&hist[0]) + atomicMax(&acc[1], li);
}",
        );
        assert!(out.contains("workgroupBarrier();"));
    }

    #[test]
    fn compare_exchange_loop_over_float_bits() {
        round_trip(
            "@group(1) @binding(0) var<storage, read_write> acc: array<atomic<u32>>;
@compute @workgroup_size(1) fn main() {
    var old = atomicLoad(&acc[0]);
    loop {
        let new_v = bitcast<u32>(bitcast<f32>(old) + 1.0);
        let r = atomicCompareExchangeWeak(&acc[0], old, new_v);
        if (r.exchanged) { break; }
        old = r.old_value;
    }
}",
        );
    }

    #[test]
    fn luma_helpers_survive_as_functions() {
        let out = round_trip(
            "@group(1) @binding(0) var<storage, read> x: array<f32>;
@group(1) @binding(1) var<storage, read_write> out: array<f32>;
@group(1) @binding(2) var<storage, read_write> hi: array<u32>;
fn luma_o(x: f32) -> f32 { return bitcast<f32>(bitcast<u32>(x) | luma._pad); }
fn luma_mulhi(a: u32, b: u32) -> u32 {
    let a0 = a & 0xFFFFu; let a1 = a >> 16u;
    let b0 = b & 0xFFFFu; let b1 = b >> 16u;
    let m1 = a1 * b0; let m2 = a0 * b1;
    let carry = (((a0 * b0) >> 16u) + (m1 & 0xFFFFu) + (m2 & 0xFFFFu)) >> 16u;
    return a1 * b1 + (m1 >> 16u) + (m2 >> 16u) + carry;
}
@compute @workgroup_size(1) fn main() {
    let a = x[0]; let b = x[1];
    out[0] = luma_o(a + b);
    out[1] = luma_o(luma_o(a * b) - a);
    hi[0] = luma_mulhi(3u, 5u);
}",
        );
        assert!(out.contains("luma_o"));
    }

    #[test]
    fn uniform_struct_layout_matches_the_declaration() {
        let (_, m) = parse(
            "struct Inner { a: f32, b: vec3<f32> }
struct Padded { a: f32, v: vec3<f32>, c: f32, inner: Inner, t: array<vec4<f32>, 2> }
@group(1) @binding(0) var<storage, read_write> out: array<f32>;
@group(1) @binding(1) var<uniform> p: Padded;
@compute @workgroup_size(1) fn main() { out[0] = p.a + p.v.x + p.v.z + p.c + p.inner.b.y + p.t[1].w; }",
            false,
        );
        let module = validate(&emit(&m));
        for decl in &m.structs {
            let Some((_, ty)) = module.types.iter().find(|(_, t)| t.name.as_deref().is_some_and(|n| n.ends_with(&format!("_{}", decl.name)))) else {
                panic!("struct {} was not emitted", decl.name)
            };
            let naga::TypeInner::Struct { members, span } = &ty.inner else { panic!("not a struct") };
            assert_eq!(*span, decl.size, "{}", decl.name);
            assert_eq!(members.iter().map(|m| m.offset).collect::<Vec<_>>(), decl.fields.iter().map(|f| f.offset).collect::<Vec<_>>(), "{}", decl.name);
        }
    }

    #[test]
    fn explicit_padding_is_reproduced() {
        let mut mb = ModuleBuilder::new("t", "main", [1, 1, 1]);
        let s = mb.add_struct(StructDecl {
            name: "Gap".into(),
            fields: vec![
                StructField { name: "a".into(), ty: KirType::U32, offset: 0 },
                StructField { name: "b".into(), ty: KirType::U32, offset: 16 },
            ],
            size: 32,
            align: 16,
        });
        let p = mb.add_uniform("p", KirType::Struct(s));
        let out = mb.add_storage_read_write("out", KirType::U32);
        let mut b = mb.entry_builder();
        let zero = b.lit_u32(0);
        let a = b.uniform_field(p, 0, KirType::U32);
        let c = b.uniform_field(p, 1, KirType::U32);
        b.store(out, zero, a + c);
        b.finish();
        let m = mb.finish();
        let module = validate(&emit(&m));
        let (_, ty) = module.types.iter().find(|(_, t)| matches!(t.inner, naga::TypeInner::Struct { .. }) && t.name.as_deref().is_some_and(|n| n.ends_with("Gap"))).unwrap();
        let naga::TypeInner::Struct { members, span } = &ty.inner else { unreachable!() };
        assert_eq!((*span, members[1].offset), (32, 16));
    }

    /// A builder module touching the operations WGSL has no spelling for.
    fn builder_module() -> KernelModule {
        let mut mb = ModuleBuilder::new("t", "main", [8, 1, 1]);
        let out = mb.add_storage_read_write("out", KirType::F32);
        let bits = mb.add_storage_read_write("bits", KirType::U32);
        let acc = mb.add_storage_read_write("acc", KirType::Atomic(ScalarTy::U32));
        let hist = mb.add_shared("hist", KirType::Atomic(ScalarTy::U32), 4);
        let mut b = mb.entry_builder();
        let gid = b.builtin(BuiltinVar::GlobalInvocationId);
        let i = b.field(gid, 0, KirType::U32);
        let x = b.load(out, i);
        let hi = b.mul_hi_u32(i, i);
        let guarded = b.opaque_guard(x + x);
        let t = b.unary(IrUnOp::Tanh, guarded);
        let cs = b.binary(IrBinOp::Copysign, t, x);
        let exact = b.binary(IrBinOp::MulExact, cs, x);
        let rot = b.binary(IrBinOp::Rotl, hi, i);
        let nan = b.unary(IrUnOp::IsNan, exact);
        let inf = b.unary(IrUnOp::IsInf, exact);
        let both = b.or(nan, inf);
        let picked = b.select(both, exact, x);
        b.store(out, i, picked);
        b.store(bits, i, rot);
        let target = AtomicTarget::Storage { binding: acc, index: Some(i.expr().boxed()) };
        let d = b.var_decl("old", KirType::F32, None);
        let _ = d;
        b.atomic_rmw(None, IrAtomicOp::Add, ScalarTy::F32, target.clone(), x);
        b.atomic_rmw(None, IrAtomicOp::Min, ScalarTy::F32, target.clone(), x);
        b.atomic_rmw(None, IrAtomicOp::Max, ScalarTy::F32, target, x);
        let shared_target = AtomicTarget::Shared { shared: hist, index: Some(i.expr().boxed()) };
        b.atomic_rmw(None, IrAtomicOp::Add, ScalarTy::U32, shared_target, i);
        b.barrier(BarrierScope::StorageAndWorkgroup);
        assert!(b.finish().is_none());
        mb.finish()
    }

    #[test]
    fn builder_modules_use_helpers_and_cas_loops() {
        let m = builder_module();
        let out = emit(&m);
        validate(&out);
        assert!(out.contains("fn luma_mulhi"));
        assert!(out.contains("select(tanh(x), select("));
        assert!(out.contains("luma._pad"));
        assert!(out.contains("atomicCompareExchangeWeak"));
        assert!(out.contains("if (nv == old) { break; }"));
        assert!(out.contains("fn luma_rotl_u32"));
        assert!(out.contains("storageBarrier();"));
    }

    /// A module adding `1.0` to `acc[i]` (and, with `min`, also taking the minimum into it).
    fn f32_atomic_module(min: bool) -> KernelModule {
        let mut mb = ModuleBuilder::new("t", "main", [8, 1, 1]);
        let acc = mb.add_storage_read_write("acc", KirType::Atomic(ScalarTy::F32));
        let mut b = mb.entry_builder();
        let gid = b.builtin(BuiltinVar::GlobalInvocationId);
        let i = b.field(gid, 0, KirType::U32);
        let one = b.lit_f32(1.0);
        let target = AtomicTarget::Storage { binding: acc, index: Some(i.expr().boxed()) };
        b.atomic_rmw(None, IrAtomicOp::Add, ScalarTy::F32, target.clone(), one);
        if min {
            b.atomic_rmw(None, IrAtomicOp::Min, ScalarTy::F32, target, one);
        }
        assert!(b.finish().is_none());
        mb.finish()
    }

    #[test]
    fn f32_atomics_are_native_only_when_nothing_needs_their_bits() {
        let validate_native = |src: &str| {
            let module = naga::front::wgsl::parse_str(src).unwrap_or_else(|e| panic!("{}\n{src}", e.emit_to_string(src)));
            naga::valid::Validator::new(naga::valid::ValidationFlags::all(), caps() | naga::valid::Capabilities::SHADER_FLOAT32_ATOMIC)
                .validate(&module)
                .unwrap_or_else(|e| panic!("{}\n{src}", e.emit_to_string(src)));
        };
        let add = emit_with(&f32_atomic_module(false), None, true);
        validate_native(&add);
        assert!(add.contains("array<atomic<f32>>") && add.contains("atomicAdd(") && !add.contains("atomicCompareExchangeWeak"), "{add}");
        let min = emit_with(&f32_atomic_module(true), None, true);
        validate(&min);
        assert!(min.contains("array<atomic<u32>>") && min.contains("atomicCompareExchangeWeak"), "{min}");
        let fallback = emit_with(&f32_atomic_module(false), None, false);
        validate(&fallback);
        assert!(fallback.contains("array<atomic<u32>>") && fallback.contains("atomicCompareExchangeWeak"), "{fallback}");
    }

    #[test]
    fn f64_rounding_and_f16_need_no_native_trunc() {
        let mut mb = ModuleBuilder::new("t", "main", [1, 1, 1]);
        let out = mb.add_storage_read_write("out", KirType::Scalar(ScalarTy::F64));
        let half = mb.add_storage_read_write("half", KirType::Scalar(ScalarTy::F16));
        let mut b = mb.entry_builder();
        let zero = b.lit_u32(0);
        let x = b.lit_f64(2.5);
        let r = b.unary(IrUnOp::Round, x);
        let t = b.unary(IrUnOp::Trunc, x);
        b.store(out, zero, r + t);
        let half_in = b.lit_f32(1.5);
        let h = b.convert(half_in, KirType::Scalar(ScalarTy::F16));
        b.store(half, zero, h);
        b.finish();
        let m = mb.finish();
        let text = emit(&m);
        validate(&text);
        assert!(text.contains("fn luma_round_f64"));
        assert!(text.contains("fn luma_trunc_f64"));
        assert!(text.contains("enable f16;"));
    }

    #[test]
    fn float_to_int_helpers_validate_for_scalars_vectors_and_every_source_width() {
        let mut mb = ModuleBuilder::new("t", "main", [1, 1, 1]);
        let f = mb.add_storage_read("f", KirType::F32);
        let v = mb.add_storage_read("v", KirType::vec(ScalarTy::F32, 4));
        let h = mb.add_storage_read("h", KirType::Scalar(ScalarTy::F16));
        let d = mb.add_storage_read("d", KirType::Scalar(ScalarTy::F64));
        let dv = mb.add_storage_read("dv", KirType::vec(ScalarTy::F64, 2));
        let oi = mb.add_storage_read_write("oi", KirType::I32);
        let ou = mb.add_storage_read_write("ou", KirType::U32);
        let ovi = mb.add_storage_read_write("ovi", KirType::vec(ScalarTy::I32, 4));
        let ovu = mb.add_storage_read_write("ovu", KirType::vec(ScalarTy::U32, 2));
        let ohi = mb.add_storage_read_write("ohi", KirType::I32);
        let mut b = mb.entry_builder();
        let zero = b.lit_u32(0);
        let (xf, xv, xh, xd, xdv) = (b.load(f, zero), b.load(v, zero), b.load(h, zero), b.load(d, zero), b.load(dv, zero));
        let r = b.float_to_int_sat(xf, KirType::I32);
        b.store(oi, zero, r);
        let r = b.float_to_int_sat(xf, KirType::U32);
        b.store(ou, zero, r);
        let r = b.float_to_int_sat(xv, KirType::vec(ScalarTy::I32, 4));
        b.store(ovi, zero, r);
        let r = b.float_to_int_sat(xh, KirType::I32);
        b.store(ohi, zero, r);
        let r = b.float_to_int_sat(xd, KirType::U32);
        b.store(ou, zero, r);
        let r = b.float_to_int_sat(xdv, KirType::vec(ScalarTy::U32, 2));
        b.store(ovu, zero, r);
        let r = b.float_to_int_sat(xdv, KirType::vec(ScalarTy::I32, 2));
        let _ = r;
        b.finish();
        let text = emit(&mb.finish());
        validate(&text);
        for helper in ["fn luma_sat_f32_i32(", "fn luma_sat_f32_u32(", "fn luma_sat_f16_i32(", "fn luma_sat_f64_u32("] {
            assert!(text.contains(helper), "{helper}\n{text}");
        }
        assert!(text.contains("luma._pad"), "the NaN test is opaque to the compiler");
        assert!(text.contains("2147483520.0") || text.contains("2147483500.0"), "the largest float below 2^31 is the clamp limit");
    }


    #[test]
    fn non_finite_literals_are_made_of_bits() {
        let mut mb = ModuleBuilder::new("t", "main", [1, 1, 1]);
        let out = mb.add_storage_read_write("out", KirType::F32);
        let mut b = mb.entry_builder();
        let zero = b.lit_u32(0);
        let (n, i, m) = (b.lit_f32(f32::NAN), b.lit_f32(f32::INFINITY), b.lit_f32(-0.0));
        b.store(out, zero, n + i + m + b.lit_f32(-1.5));
        b.finish();
        let text = emit(&mb.finish());
        validate(&text);
    }

    #[test]
    fn only_used_bindings_are_declared_under_their_original_numbers() {
        let mut mb = ModuleBuilder::new("t", "main", [1, 1, 1]);
        let _a = mb.add_storage_read("a", KirType::F32);
        let _b = mb.add_storage_read("b", KirType::F32);
        let c = mb.add_storage_read_write("c", KirType::F32);
        let mut bld = mb.entry_builder();
        let zero = bld.lit_u32(0);
        let one = bld.lit_f32(1.0);
        bld.store(c, zero, one);
        bld.finish();
        let m = mb.finish();
        let (_, mask) = used_bindings(&m);
        let mask = mask.expect("two bindings are unread");
        let text = emit_with(&m, Some(&mask), false);
        let module = validate(&text);
        let numbers: Vec<u32> = module.global_variables.iter().filter_map(|(_, v)| v.binding.filter(|b| b.group == 1).map(|b| b.binding)).collect();
        assert_eq!(numbers, vec![2]);
        assert!(emit(&m).contains("@binding(0)"));
    }

    #[test]
    fn trusted_dead_helpers_do_not_break_emission() {
        let (_, m) = parse(
            "@group(1) @binding(0) var<storage, read_write> out: array<u32>;
fn dead() -> u32 { return 7u; }
@compute @workgroup_size(1) fn main() { out[0] = 1u; }",
            true,
        );
        validate(&emit(&m));
    }
}
