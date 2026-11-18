//! Kernel IR to Metal Shading Language.
//!
//! Pure Rust and OS independent: the MSL text is produced anywhere and only compiled by the Metal
//! runtime compiler on Apple hardware.
//!
//! ABI of the emitted `luma_main` kernel:
//!
//! - `buffer(0)`: `LumaGrid` (16 bytes: `packed_uint3 size; uint _pad`).
//! - [`MslAbi::Direct`]: the *active* bindings (those `used` selects, ascending) at `buffer(1..=m)`
//!   and, when the kernel calls `arrayLength`, one `constant uint*` of element counts (ascending
//!   binding order, see [`MslTranslated::array_lengths`]) at `buffer(m + 1)`. Used when the total
//!   buffer slot count is at most 31.
//! - [`MslAbi::ArgBuffer`]: `constant LumaArgs& _luma_args` at `buffer(1)`, holding one 64-bit
//!   pointer per active binding followed by the `uint` element counts.
//!
//! Every name in the output is numeric-prefixed (`b3`, `l7`, `f0`, `S2`, `m1`, ...) so no WGSL name
//! can collide with an MSL or C++ keyword.

use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::fmt::Write as _;

use luma_core::BindingKind;
use luma_core::kir::*;

/// How the active bindings reach the kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MslAbi {
    Direct,
    ArgBuffer,
}

/// The result of lowering one [`KernelModule`].
#[derive(Clone, Debug)]
pub struct MslTranslated {
    pub source: String,
    pub entry_name: String,
    pub abi: MslAbi,
    /// Bindings whose element count the kernel reads (`arrayLength`), ascending. The runtime packs
    /// `bytes / element_stride` of each, in this order, into the length buffer (Direct) or after the
    /// pointers of `LumaArgs` (ArgBuffer).
    pub array_lengths: Vec<u32>,
}

const LANES: [&str; 4] = ["x", "y", "z", "w"];
const MIN_I32: &str = "(-2147483647 - 1)";

fn scalar_name(s: ScalarTy) -> &'static str {
    match s {
        ScalarTy::Bool => "bool",
        ScalarTy::U32 => "uint",
        ScalarTy::I32 => "int",
        ScalarTy::F16 => "half",
        ScalarTy::F32 => "float",
        // MSL has no double: the compiler rejects it loudly.
        ScalarTy::F64 => "double",
    }
}

fn vec_name(s: ScalarTy, lanes: u8) -> String {
    if lanes <= 1 { scalar_name(s).to_owned() } else { format!("{}{}", scalar_name(s), lanes) }
}

fn uname(t: &KirType) -> String {
    vec_name(ScalarTy::U32, t.lanes())
}

fn f32_lit(v: f32) -> String {
    if v.is_nan() {
        "as_type<float>(0x7fc00000u)".into()
    } else if v.is_infinite() {
        if v > 0.0 { "as_type<float>(0x7f800000u)".into() } else { "as_type<float>(0xff800000u)".into() }
    } else if v.is_sign_negative() {
        format!("({v:?}f)")
    } else {
        format!("{v:?}f")
    }
}

fn is_vec3(t: &KirType) -> bool {
    matches!(t, KirType::Vec { elem, lanes: 3 } if *elem != ScalarTy::Bool)
}

fn pad(out: &mut String, ind: usize) {
    for _ in 0..ind {
        out.push_str("    ");
    }
}

/// Declared at the top level of `body`, so visible to the `continuing` block of the loop.
fn top_decls(body: &[Stmt]) -> Vec<LocalId> {
    let mut v = Vec::new();
    for s in body {
        match s {
            Stmt::Let { local, .. } | Stmt::Var { local, .. } => v.push(*local),
            Stmt::AtomicRmw { dst: Some(d), .. } => v.push(*d),
            Stmt::AtomicCompareExchange { dst_old, dst_exchanged, .. } => {
                v.push(*dst_old);
                v.push(*dst_exchanged);
            }
            _ => {}
        }
    }
    v
}

/// A `continue` that targets the loop whose body this is (nested loops excluded).
fn has_continue(body: &[Stmt]) -> bool {
    body.iter().any(|s| match s {
        Stmt::Continue => true,
        Stmt::If { then_body, else_body, .. } => has_continue(then_body) || has_continue(else_body),
        Stmt::Switch { cases, default, .. } => cases.iter().any(|(_, b, _)| has_continue(b)) || has_continue(default),
        _ => false,
    })
}

#[derive(Default, Clone)]
struct Res {
    grid: bool,
    binds: BTreeSet<u32>,
    lens: bool,
    shared: BTreeSet<u32>,
    privs: bool,
    calls: BTreeSet<u32>,
}

fn lvalue_res(l: &LValue, r: &mut Res) {
    match l {
        LValue::Local(_) => {}
        LValue::Private(_) => r.privs = true,
        LValue::Storage { binding, .. } => {
            r.binds.insert(*binding);
        }
        LValue::Shared { shared, .. } => {
            r.shared.insert(*shared);
        }
        LValue::Index { base, .. } | LValue::Field { base, .. } => lvalue_res(base, r),
    }
}

fn atomic_res(t: &AtomicTarget, r: &mut Res) {
    match t {
        AtomicTarget::Storage { binding, .. } => {
            r.binds.insert(*binding);
        }
        AtomicTarget::Shared { shared, .. } => {
            r.shared.insert(*shared);
        }
    }
}

struct Em<'a> {
    res: Vec<Res>,
    m: &'a KernelModule,
    /// Position among the active bindings, per binding number.
    rank: Vec<Option<usize>>,
    lens: Vec<u32>,
    helpers: BTreeMap<String, String>,
    strided: bool,
    locals: Vec<(String, KirType, bool)>,
    params: Vec<(String, KirType)>,
    hoisted: HashSet<u32>,
    labels: u32,
}

impl<'a> Em<'a> {
    // ---- types -------------------------------------------------------------------------------

    fn has_bool(&self, t: &KirType) -> bool {
        match t {
            KirType::Scalar(ScalarTy::Bool) | KirType::Vec { elem: ScalarTy::Bool, .. } => true,
            KirType::Array { elem, .. } => self.has_bool(elem),
            KirType::Struct(i) => self.m.structs[*i as usize].fields.iter().any(|f| self.has_bool(&f.ty)),
            _ => false,
        }
    }

    /// `(size, align)` of the MSL type [`Self::ty`] prints.
    fn msl_layout(&self, t: &KirType, packed: bool) -> (u32, u32) {
        match t {
            KirType::Scalar(ScalarTy::Bool) => (1, 1),
            KirType::Scalar(s) | KirType::Atomic(s) => (s.size_bytes(), s.align_bytes()),
            KirType::Vec { elem, lanes } => {
                let e = if *elem == ScalarTy::Bool { 1 } else { elem.size_bytes() };
                let n = u32::from(*lanes);
                if packed && n == 3 {
                    (e * 3, e)
                } else if n == 3 {
                    (e * 4, e * 4)
                } else {
                    (e * n, e * n)
                }
            }
            KirType::Array { elem, len, stride } => {
                let (s, a) = self.msl_layout(elem, false);
                let natural = s.next_multiple_of(a);
                (len * if *stride > natural && !self.has_bool(elem) { *stride } else { natural }, a)
            }
            KirType::Struct(i) => {
                let s = &self.m.structs[*i as usize];
                (s.size, s.align)
            }
        }
    }

    /// Bytes of padding to append to each element so the array stride is `stride`.
    fn stride_pad(&self, elem: &KirType, stride: u32) -> Option<u32> {
        if self.has_bool(elem) {
            return None;
        }
        let (s, a) = self.msl_layout(elem, false);
        (stride > s.next_multiple_of(a)).then(|| stride - s)
    }

    fn elem_str(&mut self, elem: &KirType, stride: u32) -> String {
        let e = self.ty(elem, false);
        match self.stride_pad(elem, stride) {
            Some(p) => {
                self.strided = true;
                format!("_LumaStrided<{e}, {p}>")
            }
            None => e,
        }
    }

    fn ty(&mut self, t: &KirType, packed: bool) -> String {
        match t {
            KirType::Scalar(s) => scalar_name(*s).to_owned(),
            KirType::Vec { elem, lanes } => {
                if packed && *lanes == 3 && *elem != ScalarTy::Bool {
                    format!("packed_{}", vec_name(*elem, 3))
                } else {
                    vec_name(*elem, *lanes)
                }
            }
            KirType::Array { elem, len, stride } => {
                let e = self.elem_str(elem, *stride);
                format!("array<{e}, {len}>")
            }
            KirType::Struct(i) => format!("S{i}"),
            KirType::Atomic(ScalarTy::I32) => "atomic_int".into(),
            KirType::Atomic(_) => "atomic_uint".into(),
        }
    }

    fn zero(&mut self, t: &KirType) -> String {
        match t {
            KirType::Scalar(ScalarTy::Bool) => "false".into(),
            KirType::Vec { elem: ScalarTy::Bool, .. } => format!("{}(false)", self.ty(t, false)),
            KirType::Scalar(_) | KirType::Vec { .. } => format!("{}(0)", self.ty(t, false)),
            _ => format!("{}{{}}", self.ty(t, false)),
        }
    }

    fn natural_stride(&self, t: &KirType) -> u32 {
        let (s, a) = self.msl_layout(t, false);
        s.next_multiple_of(a)
    }

    fn struct_text(&mut self, i: usize, out: &mut String) {
        let m = self.m;
        let st = &m.structs[i];
        let mut s = format!("struct alignas({}) S{i} {{\n", st.align.max(1));
        let mut cursor = 0u32;
        let mut pads = 0;
        for (j, f) in st.fields.iter().enumerate() {
            if f.offset > cursor {
                let _ = writeln!(s, "    uchar _pad{pads}[{}];", f.offset - cursor);
                pads += 1;
                cursor = f.offset;
            }
            let t = self.ty(&f.ty, true);
            let _ = writeln!(s, "    {t} m{j};");
            cursor += self.msl_layout(&f.ty, true).0;
        }
        if st.size > cursor {
            let _ = writeln!(s, "    uchar _pad{pads}[{}];", st.size - cursor);
        }
        s.push_str("};\n");
        if !st.fields.iter().any(|f| self.has_bool(&f.ty)) {
            let _ = writeln!(s, "static_assert(sizeof(S{i}) == {}, \"S{i} layout\");", st.size);
        }
        out.push_str(&s);
    }

    /// Struct constructor helper used to lower `Compose` of a struct.
    fn struct_ctor(&mut self, i: u32) -> String {
        let name = format!("_luma_mk{i}");
        let key = format!("mk_{i:06}");
        if !self.helpers.contains_key(&key) {
            let m = self.m;
            let st = &m.structs[i as usize];
            let mut params = Vec::new();
            let mut body = String::new();
            for (j, f) in st.fields.iter().enumerate() {
                let t = self.ty(&f.ty, false);
                params.push(format!("{t} v{j}"));
                let _ = write!(body, " r.m{j} = v{j};");
            }
            let text = format!("static inline __attribute__((always_inline)) S{i} {name}({}) {{ S{i} r{{}};{body} return r; }}\n", params.join(", "));
            self.helpers.insert(key, text);
        }
        name
    }

    // ---- type inference ----------------------------------------------------------------------

    fn ety(&self, e: &Expr) -> KirType {
        match e {
            Expr::LitBool(_) => KirType::BOOL,
            Expr::LitU32(_) | Expr::GridSize(_) | Expr::ArrayLength { .. } => KirType::U32,
            Expr::LitI32(_) => KirType::I32,
            Expr::LitF16(_) => KirType::Scalar(ScalarTy::F16),
            Expr::LitF32(_) => KirType::F32,
            Expr::LitF64(_) => KirType::Scalar(ScalarTy::F64),
            Expr::ZeroVal(t) => t.clone(),
            Expr::Local(l) => self.locals.get(l.0 as usize).map(|l| l.1.clone()).unwrap_or(KirType::F32),
            Expr::FnParam(i) => self.params.get(*i as usize).map(|p| p.1.clone()).unwrap_or(KirType::F32),
            Expr::Private(i) => self.m.privates[*i as usize].ty.clone(),
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
            Expr::Splat { elem, lanes, .. } => KirType::Vec { elem: *elem, lanes: *lanes },
            Expr::Cast { dst_ty, .. } => dst_ty.clone(),
            Expr::Call { ty, .. } => ty.clone().unwrap_or(KirType::U32),
            Expr::AtomicLoad { ty, .. } => KirType::Scalar(*ty),
        }
    }

    // ---- places ------------------------------------------------------------------------------

    fn binding_root(&mut self, b: u32) -> (String, KirType) {
        let d = &self.m.bindings[b as usize];
        let base = format!("b{b}");
        if d.is_unsized_array {
            let elem = d.elem_ty.clone();
            let stride = d.element_stride.unwrap_or_else(|| self.natural_stride(&elem));
            (base, KirType::Array { elem: Box::new(elem), len: 0, stride })
        } else {
            (base, d.elem_ty.clone())
        }
    }

    fn shared_root(&mut self, s: u32) -> (String, KirType) {
        let d = &self.m.shared[s as usize];
        let base = format!("s{s}");
        match d.len {
            Some(n) => {
                let elem = d.elem_ty.clone();
                let stride = self.natural_stride(&elem);
                (base, KirType::Array { elem: Box::new(elem), len: n, stride })
            }
            None => (base, d.elem_ty.clone()),
        }
    }

    fn index_step(&mut self, cur: &KirType, idx: &Expr) -> (String, KirType) {
        let i = self.expr(idx);
        match cur {
            KirType::Array { elem, stride, .. } => {
                let suffix = if self.stride_pad(elem, *stride).is_some() { ".v" } else { "" };
                (format!("[{i}]{suffix}"), (**elem).clone())
            }
            KirType::Vec { elem, .. } => (format!("[{i}]"), KirType::Scalar(*elem)),
            other => (format!("[{i}]"), other.clone()),
        }
    }

    fn field_step(&self, cur: &KirType, f: u32) -> (String, KirType) {
        match cur {
            KirType::Struct(s) => (format!(".m{f}"), self.m.structs[*s as usize].fields[f as usize].ty.clone()),
            KirType::Vec { elem, .. } => (format!(".{}", LANES[(f as usize).min(3)]), KirType::Scalar(*elem)),
            other => (format!(".m{f}"), other.clone()),
        }
    }

    fn step(&mut self, cur: &KirType, st: &AccessStep) -> (String, KirType) {
        match st {
            AccessStep::Field(f) => self.field_step(cur, *f),
            AccessStep::Index(e) => self.index_step(cur, e),
        }
    }

    fn path(&mut self, mut s: String, mut cur: KirType, steps: &[AccessStep]) -> (String, KirType) {
        for st in steps {
            let (t, n) = self.step(&cur, st);
            s.push_str(&t);
            cur = n;
        }
        (s, cur)
    }

    fn wrap_vec3(&mut self, s: String, t: &KirType) -> String {
        if is_vec3(t) { format!("{}({s})", self.ty(t, false)) } else { s }
    }

    fn lvalue(&mut self, lv: &LValue) -> (String, KirType) {
        match lv {
            LValue::Local(l) => (format!("l{}", l.0), self.locals[l.0 as usize].1.clone()),
            LValue::Private(p) => (format!("_pv.p{p}"), self.m.privates[*p as usize].ty.clone()),
            LValue::Storage { binding, path } => {
                let (b, r) = self.binding_root(*binding);
                self.path(b, r, path)
            }
            LValue::Shared { shared, path } => {
                let (b, r) = self.shared_root(*shared);
                self.path(b, r, path)
            }
            LValue::Index { base, index } => {
                let (s, t) = self.lvalue(base);
                let (a, n) = self.index_step(&t, index);
                (s + &a, n)
            }
            LValue::Field { base, field } => {
                let (s, t) = self.lvalue(base);
                let (a, n) = self.field_step(&t, *field);
                (s + &a, n)
            }
        }
    }

    // ---- atomics -----------------------------------------------------------------------------

    /// `(pointer expression, address space)` of an atomic of scalar `elem`.
    fn atomic_ptr(&mut self, t: &AtomicTarget, elem: ScalarTy) -> (String, &'static str) {
        let (place, declared, space) = match t {
            AtomicTarget::Storage { binding, index } => {
                let d = &self.m.bindings[*binding as usize];
                let declared = d.elem_ty.clone();
                let p = match index {
                    Some(i) => format!("b{binding}[{}]", self.expr(i)),
                    None => format!("b{binding}"),
                };
                (p, declared, "device")
            }
            AtomicTarget::Shared { shared, index } => {
                let declared = self.m.shared[*shared as usize].elem_ty.clone();
                let p = match index {
                    Some(i) => format!("s{shared}[{}]", self.expr(i)),
                    None => format!("s{shared}"),
                };
                (p, declared, "threadgroup")
            }
        };
        if matches!(declared, KirType::Atomic(_)) {
            (format!("(&{place})"), space)
        } else {
            let a = if elem == ScalarTy::I32 { "atomic_int" } else { "atomic_uint" };
            (format!("reinterpret_cast<{space} {a}*>(&{place})"), space)
        }
    }

    fn f32_atomic_helper(&mut self, op: IrAtomicOp, space: &str) -> String {
        let name = match op {
            IrAtomicOp::Add => "add",
            IrAtomicOp::Sub => "sub",
            IrAtomicOp::Min => "min",
            _ => "max",
        };
        let fname = format!("_luma_af32_{name}_{space}");
        let key = format!("af32_{name}_{space}");
        self.helpers.entry(key).or_insert_with(|| {
            let combine = match op {
                IrAtomicOp::Add => "cur + v",
                IrAtomicOp::Sub => "cur - v",
                IrAtomicOp::Min => "fmin(cur, v)",
                _ => "fmax(cur, v)",
            };
            format!(
                "static inline __attribute__((always_inline)) float {fname}({space} atomic_uint* p, float v) {{\n    uint old = atomic_load_explicit(p, memory_order_relaxed);\n    for (;;) {{\n        float cur = as_type<float>(old);\n        if (atomic_compare_exchange_weak_explicit(p, &old, as_type<uint>({combine}), memory_order_relaxed, memory_order_relaxed)) {{ break; }}\n    }}\n    return as_type<float>(old);\n}}\n"
            )
        });
        fname
    }

    fn rmw_expr(&mut self, op: IrAtomicOp, elem: ScalarTy, ptr: &str, space: &str, val: &str) -> String {
        let order = "memory_order_relaxed";
        if elem == ScalarTy::F32 {
            return match op {
                IrAtomicOp::Add | IrAtomicOp::Sub | IrAtomicOp::Min | IrAtomicOp::Max => {
                    let f = self.f32_atomic_helper(op, space);
                    format!("{f}({ptr}, {val})")
                }
                IrAtomicOp::Exchange => format!("as_type<float>(atomic_exchange_explicit({ptr}, as_type<uint>({val}), {order}))"),
                IrAtomicOp::And => format!("as_type<float>(atomic_fetch_and_explicit({ptr}, as_type<uint>({val}), {order}))"),
                IrAtomicOp::Or => format!("as_type<float>(atomic_fetch_or_explicit({ptr}, as_type<uint>({val}), {order}))"),
                IrAtomicOp::Xor => format!("as_type<float>(atomic_fetch_xor_explicit({ptr}, as_type<uint>({val}), {order}))"),
            };
        }
        let f = match op {
            IrAtomicOp::Add => "atomic_fetch_add_explicit",
            IrAtomicOp::Sub => "atomic_fetch_sub_explicit",
            IrAtomicOp::Min => "atomic_fetch_min_explicit",
            IrAtomicOp::Max => "atomic_fetch_max_explicit",
            IrAtomicOp::And => "atomic_fetch_and_explicit",
            IrAtomicOp::Or => "atomic_fetch_or_explicit",
            IrAtomicOp::Xor => "atomic_fetch_xor_explicit",
            IrAtomicOp::Exchange => "atomic_exchange_explicit",
        };
        format!("{f}({ptr}, {val}, {order})")
    }

    // ---- helper functions --------------------------------------------------------------------

    /// Registers `scalar_text` as the 1-lane overload and generates 2..=4 lane overloads that apply
    /// it per lane.
    #[allow(clippy::too_many_arguments)]
    fn lanes_helper(&mut self, family: &str, fname: &str, elem: ScalarTy, ret: ScalarTy, nargs: usize, lanes: u8, scalar_text: impl Fn(&str, &str) -> String) {
        let k1 = format!("{family}_1");
        self.helpers.entry(k1).or_insert_with(|| {
            let t = scalar_name(elem);
            let r = scalar_name(ret);
            scalar_text(t, r)
        });
        if lanes > 1 {
            let key = format!("{family}_{lanes}");
            self.helpers.entry(key).or_insert_with(|| {
                let t = vec_name(elem, lanes);
                let r = vec_name(ret, lanes);
                let params: Vec<String> = (0..nargs).map(|i| format!("{t} a{i}")).collect();
                let comps: Vec<String> = (0..lanes as usize)
                    .map(|l| {
                        let args: Vec<String> = (0..nargs).map(|i| format!("a{i}.{}", LANES[l])).collect();
                        format!("{fname}({})", args.join(", "))
                    })
                    .collect();
                format!("static inline __attribute__((always_inline)) {r} {fname}({}) {{ return {r}({}); }}\n", params.join(", "), comps.join(", "))
            });
        }
    }

    fn div_rem(&mut self, rem: bool, elem: ScalarTy, lanes: u8) -> String {
        let fname = if rem { "_luma_rem" } else { "_luma_div" };
        let signed = elem == ScalarTy::I32;
        let family = format!("{}_{}", if rem { "rem" } else { "div" }, if signed { "i" } else { "u" });
        let op = if rem { "%" } else { "/" };
        self.lanes_helper(&family, fname, elem, elem, 2, lanes, |t, _| {
            let safe = if signed {
                format!("((b == 0 || (a == {MIN_I32} && b == -1)) ? 1 : b)")
            } else {
                "(b == 0u ? 1u : b)".to_owned()
            };
            format!("static inline __attribute__((always_inline)) {t} {fname}({t} a, {t} b) {{ return a {op} {safe}; }}\n")
        });
        fname.to_owned()
    }

    fn sat_conv(&mut self, dst: ScalarTy, lanes: u8) -> String {
        let (fname, family) = if dst == ScalarTy::I32 { ("_luma_f2i", "f2i") } else { ("_luma_f2u", "f2u") };
        self.lanes_helper(family, fname, ScalarTy::F32, dst, 1, lanes, |_, r| {
            if dst == ScalarTy::I32 {
                format!("static inline __attribute__((always_inline)) {r} {fname}(float x) {{ return (as_type<uint>(x) & 0x7fffffffu) > 0x7f800000u ? 0 : (x <= -2147483648.0f ? {MIN_I32} : (x >= 2147483648.0f ? 2147483647 : int(x))); }}\n")
            } else {
                format!("static inline __attribute__((always_inline)) {r} {fname}(float x) {{ return (as_type<uint>(x) & 0x7fffffffu) > 0x7f800000u ? 0u : (x <= 0.0f ? 0u : (x >= 4294967296.0f ? 0xffffffffu : uint(x))); }}\n")
            }
        });
        fname.to_owned()
    }

    fn tanh_fn(&mut self, lanes: u8) -> String {
        self.lanes_helper("tanh", "_luma_tanh", ScalarTy::F32, ScalarTy::F32, 1, lanes, |_, _| {
            "static inline __attribute__((always_inline)) float _luma_tanh(float x) { return select(tanh(x), select(-1.0f, 1.0f, x > 0.0f), fabs(x) > 15.0f); }\n".to_owned()
        });
        "_luma_tanh".into()
    }

    // ---- expressions -------------------------------------------------------------------------

    fn expr(&mut self, e: &Expr) -> String {
        match e {
            Expr::LitBool(b) => b.to_string(),
            Expr::LitU32(v) => format!("{v}u"),
            Expr::LitI32(v) => {
                if *v == i32::MIN {
                    MIN_I32.to_owned()
                } else if *v < 0 {
                    format!("({v})")
                } else {
                    v.to_string()
                }
            }
            Expr::LitF16(v) => format!("half({})", f32_lit(*v)),
            Expr::LitF32(v) => f32_lit(*v),
            Expr::LitF64(v) => format!("({v:?})"),
            Expr::ZeroVal(t) => self.zero(t),
            Expr::Local(l) => format!("l{}", l.0),
            Expr::FnParam(i) => format!("a{i}"),
            Expr::Private(i) => format!("_pv.p{i}"),
            Expr::GridSize(a) => format!("luma.size.{}", LANES[(*a as usize).min(2)]),
            Expr::UniformRead { binding, path, ty } | Expr::StorageRead { binding, path, ty } => {
                let (b, r) = self.binding_root(*binding);
                let (s, _) = self.path(b, r, path);
                self.wrap_vec3(s, ty)
            }
            Expr::Load { binding, index, ty } => {
                let (b, r) = self.binding_root(*binding);
                let (a, _) = self.index_step(&r, index);
                self.wrap_vec3(b + &a, ty)
            }
            Expr::SharedLoad { shared, index, ty } => {
                let (b, r) = self.shared_root(*shared);
                let (a, _) = self.index_step(&r, index);
                self.wrap_vec3(b + &a, ty)
            }
            Expr::SharedRead { shared, path, ty } => {
                let (b, r) = self.shared_root(*shared);
                let (s, _) = self.path(b, r, path);
                self.wrap_vec3(s, ty)
            }
            Expr::Access { base, step, ty } => {
                let bt = self.ety(base);
                let b = self.expr(base);
                let (a, _) = self.step(&bt, step);
                let s = format!("({b}){a}");
                self.wrap_vec3(s, ty)
            }
            Expr::Swizzle { base, indices, len, .. } => {
                let b = self.expr(base);
                let sw: String = indices.iter().take(*len as usize).map(|i| LANES[(*i as usize).min(3)]).collect();
                format!("({b}).{sw}")
            }
            Expr::Compose { ty, comps } => {
                let parts: Vec<String> = comps.iter().map(|c| self.expr(c)).collect();
                match ty {
                    KirType::Struct(i) => {
                        let f = self.struct_ctor(*i);
                        format!("{f}({})", parts.join(", "))
                    }
                    KirType::Array { elem, stride, .. } => {
                        let t = self.ty(ty, false);
                        if self.stride_pad(elem, *stride).is_some() {
                            let inner: Vec<String> = parts.iter().map(|p| format!("{{{p}}}")).collect();
                            format!("{t}{{{}}}", inner.join(", "))
                        } else {
                            format!("{t}{{{}}}", parts.join(", "))
                        }
                    }
                    _ => format!("{}({})", self.ty(ty, false), parts.join(", ")),
                }
            }
            Expr::Splat { elem, lanes, val } => {
                let v = self.expr(val);
                format!("{}({v})", vec_name(*elem, *lanes))
            }
            Expr::Unary { op, arg, ty } => self.unary(*op, arg, ty),
            Expr::Binary { op, lhs, rhs, ty } => self.binary(*op, lhs, rhs, ty),
            Expr::Ternary { op, a, b, c, ty } => {
                let ct = self.ety(a);
                let (sa, sb, sc) = (self.expr(a), self.expr(b), self.expr(c));
                match op {
                    IrTernaryOp::Fma => format!("fma({sa}, {sb}, {sc})"),
                    IrTernaryOp::Clamp => format!("clamp({sa}, {sb}, {sc})"),
                    IrTernaryOp::Select => {
                        if ct.lanes() > 1 {
                            format!("select({sc}, {sb}, {sa})")
                        } else {
                            let _ = ty;
                            format!("(({sa}) ? ({sb}) : ({sc}))")
                        }
                    }
                }
            }
            Expr::Cast { kind, expr, src_ty, dst_ty } => {
                let s = self.expr(expr);
                let d = self.ty(dst_ty, false);
                match kind {
                    CastKind::Convert => format!("{d}({s})"),
                    CastKind::Bitcast => {
                        if src_ty == dst_ty { s } else { format!("as_type<{d}>({s})") }
                    }
                    CastKind::FloatToIntSat => {
                        let dst = dst_ty.elem_scalar().unwrap_or(ScalarTy::U32);
                        let f = self.sat_conv(dst, dst_ty.lanes());
                        format!("{f}({}({s}))", vec_name(ScalarTy::F32, dst_ty.lanes()))
                    }
                }
            }
            Expr::Call { func, args, .. } => {
                let mut a = self.ctx_args(func.0 as usize);
                for x in args {
                    a.push(self.expr(x));
                }
                format!("f{}({})", func.0, a.join(", "))
            }
            Expr::AtomicLoad { target, ty } => {
                let (p, _) = self.atomic_ptr(target, *ty);
                if *ty == ScalarTy::F32 {
                    format!("as_type<float>(atomic_load_explicit({p}, memory_order_relaxed))")
                } else {
                    format!("atomic_load_explicit({p}, memory_order_relaxed)")
                }
            }
            Expr::Subgroup { op, arg, lane, .. } => {
                let a = match arg {
                    Some(a) => self.expr(a),
                    None => "true".to_owned(),
                };
                match op {
                    IrSubgroupOp::Ballot => format!("uint4(ulong(simd_ballot({a})), 0u, 0u, 0u)"),
                    IrSubgroupOp::Add => format!("simd_sum({a})"),
                    IrSubgroupOp::Min => format!("simd_min({a})"),
                    IrSubgroupOp::Max => format!("simd_max({a})"),
                    IrSubgroupOp::ExclusiveAdd => format!("simd_prefix_exclusive_sum({a})"),
                    IrSubgroupOp::Shuffle => match lane {
                        Some(l) => {
                            let l = self.expr(l);
                            format!("simd_shuffle({a}, ushort({l}))")
                        }
                        None => format!("simd_broadcast_first({a})"),
                    },
                }
            }
            Expr::ArrayLength { binding } => match self.lens.iter().position(|b| b == binding) {
                Some(i) => format!("_ll[{i}]"),
                None => "0u".to_owned(),
            },
        }
    }

    fn unary(&mut self, op: IrUnOp, arg: &Expr, ty: &KirType) -> String {
        let a = self.expr(arg);
        let t = self.ty(ty, false);
        let at = self.ety(arg);
        let elem = at.elem_scalar().unwrap_or(ScalarTy::F32);
        let lanes = at.lanes();
        let u = uname(&at);
        let is_f32 = elem == ScalarTy::F32;
        match op {
            IrUnOp::Neg => {
                if elem == ScalarTy::I32 {
                    format!("{t}({u}(0u) - {u}({a}))")
                } else {
                    format!("(-{a})")
                }
            }
            IrUnOp::Not => format!("(!{a})"),
            IrUnOp::BitNot => format!("(~{a})"),
            IrUnOp::Abs => match elem {
                ScalarTy::I32 => format!("{t}(abs({a}))"),
                ScalarTy::U32 => a,
                _ => format!("fabs({a})"),
            },
            IrUnOp::Sqrt => format!("sqrt({a})"),
            IrUnOp::Exp => format!("exp({a})"),
            IrUnOp::Ln => format!("log({a})"),
            IrUnOp::Sin => format!("sin({a})"),
            IrUnOp::Cos => format!("cos({a})"),
            IrUnOp::Tan => format!("tan({a})"),
            IrUnOp::Tanh => {
                let f = self.tanh_fn(lanes);
                if is_f32 {
                    format!("{f}({a})")
                } else {
                    format!("{t}({f}({}({a})))", vec_name(ScalarTy::F32, lanes))
                }
            }
            IrUnOp::Sign => {
                if elem.is_float() {
                    format!("sign({a})")
                } else {
                    format!("({t}({a} > {t}(0)) - {t}({a} < {t}(0)))")
                }
            }
            IrUnOp::Floor => format!("floor({a})"),
            IrUnOp::Ceil => format!("ceil({a})"),
            IrUnOp::Trunc => format!("trunc({a})"),
            IrUnOp::Round => format!("rint({a})"),
            IrUnOp::CountOneBits => format!("popcount({a})"),
            IrUnOp::FirstTrailingBit => format!("select(ctz({a}), {t}(0xffffffffu), ({a}) == {t}(0))"),
            IrUnOp::IsNan => {
                if is_f32 {
                    format!("(({u}(as_type<{u}>({a})) & {u}(0x7fffffffu)) > {u}(0x7f800000u))")
                } else {
                    format!("isnan({a})")
                }
            }
            IrUnOp::IsInf => {
                if is_f32 {
                    format!("(({u}(as_type<{u}>({a})) & {u}(0x7fffffffu)) == {u}(0x7f800000u))")
                } else {
                    format!("isinf({a})")
                }
            }
            IrUnOp::OpaqueGuard => format!("({a})"),
            IrUnOp::InverseSqrt => format!("rsqrt({a})"),
            IrUnOp::Exp2 => format!("exp2({a})"),
            IrUnOp::Log2 => format!("log2({a})"),
            IrUnOp::Asin => format!("asin({a})"),
            IrUnOp::Acos => format!("acos({a})"),
            IrUnOp::Atan => format!("atan({a})"),
            IrUnOp::FirstLeadingBit => {
                if elem == ScalarTy::I32 {
                    format!("({t}(31) - clz(select({a}, ~({a}), ({a}) < {t}(0))))")
                } else {
                    format!("({t}(31u) - clz({a}))")
                }
            }
            IrUnOp::CountLeadingZeros => format!("clz({a})"),
            IrUnOp::CountTrailingZeros => format!("ctz({a})"),
            IrUnOp::All => format!("all({a})"),
            IrUnOp::Any => format!("any({a})"),
        }
    }

    fn binary(&mut self, op: IrBinOp, lhs: &Expr, rhs: &Expr, ty: &KirType) -> String {
        let a = self.expr(lhs);
        let b = self.expr(rhs);
        let t = self.ty(ty, false);
        let elem = ty.elem_scalar().unwrap_or(ScalarTy::F32);
        let lanes = ty.lanes();
        let u = uname(ty);
        let signed_int = elem == ScalarTy::I32;
        let sym = |s: &str| format!("({a} {s} {b})");
        match op {
            IrBinOp::Add | IrBinOp::Sub | IrBinOp::Mul => {
                let s = match op {
                    IrBinOp::Add => "+",
                    IrBinOp::Sub => "-",
                    _ => "*",
                };
                if signed_int { format!("{t}({u}({a}) {s} {u}({b}))") } else { sym(s) }
            }
            IrBinOp::AddExact => sym("+"),
            IrBinOp::SubExact => sym("-"),
            IrBinOp::MulExact => sym("*"),
            IrBinOp::Div => {
                if elem.is_float() {
                    sym("/")
                } else {
                    let f = self.div_rem(false, elem, lanes);
                    format!("{f}({t}({a}), {t}({b}))")
                }
            }
            IrBinOp::Rem => {
                if elem.is_float() {
                    format!("fmod({t}({a}), {t}({b}))")
                } else {
                    let f = self.div_rem(true, elem, lanes);
                    format!("{f}({t}({a}), {t}({b}))")
                }
            }
            IrBinOp::Min => format!("{}({a}, {b})", if elem.is_float() { "fmin" } else { "min" }),
            IrBinOp::Max => format!("{}({a}, {b})", if elem.is_float() { "fmax" } else { "max" }),
            IrBinOp::Pow => format!("pow({a}, {b})"),
            IrBinOp::Atan2 => format!("atan2({a}, {b})"),
            IrBinOp::Copysign => format!("copysign({a}, {b})"),
            IrBinOp::BitAnd | IrBinOp::BitOr | IrBinOp::BitXor if elem == ScalarTy::Bool => {
                let s = match op {
                    IrBinOp::BitAnd => "&",
                    IrBinOp::BitOr => "|",
                    _ => "^",
                };
                format!("{t}({u}({a}) {s} {u}({b}))")
            }
            IrBinOp::BitAnd => sym("&"),
            IrBinOp::BitOr => sym("|"),
            IrBinOp::BitXor => sym("^"),
            IrBinOp::Shl => format!("{t}({u}({a}) << ({u}({b}) & {u}(31u)))"),
            IrBinOp::Shr => {
                if signed_int {
                    format!("({a} >> {t}({u}({b}) & {u}(31u)))")
                } else {
                    format!("({a} >> ({u}({b}) & {u}(31u)))")
                }
            }
            IrBinOp::Rotl => format!("rotate({a}, {t}({u}({b}) & {u}(31u)))"),
            IrBinOp::MulHiU32 => format!("mulhi({a}, {b})"),
            IrBinOp::Eq => sym("=="),
            IrBinOp::Ne => sym("!="),
            IrBinOp::Lt => sym("<"),
            IrBinOp::Le => sym("<="),
            IrBinOp::Gt => sym(">"),
            IrBinOp::Ge => sym(">="),
            IrBinOp::And => sym("&&"),
            IrBinOp::Or => sym("||"),
        }
    }

    // ---- statements --------------------------------------------------------------------------

    fn decl(&mut self, id: LocalId, init: Option<String>, konst: bool, out: &mut String, ind: usize) {
        pad(out, ind);
        let lt = self.locals[id.0 as usize].1.clone();
        let init = match init {
            Some(i) => i,
            None => self.zero(&lt),
        };
        if self.hoisted.contains(&id.0) {
            let _ = writeln!(out, "l{} = {init};", id.0);
            return;
        }
        let t = self.ty(&lt, false);
        let _ = writeln!(out, "{}{t} l{} = {init};", if konst { "const " } else { "" }, id.0);
    }

    fn undeclared(&mut self, id: LocalId, out: &mut String, ind: usize) {
        if self.hoisted.contains(&id.0) {
            return;
        }
        let lt = self.locals[id.0 as usize].1.clone();
        let t = self.ty(&lt, false);
        pad(out, ind);
        let _ = writeln!(out, "{t} l{};", id.0);
    }

    fn block(&mut self, body: &[Stmt], out: &mut String, ind: usize) {
        for s in body {
            self.stmt(s, out, ind);
        }
    }

    fn stmt(&mut self, s: &Stmt, out: &mut String, ind: usize) {
        match s {
            Stmt::Let { local, init } => {
                let i = self.expr(init);
                self.decl(*local, Some(i), true, out, ind);
            }
            Stmt::Var { local, init } => {
                let i = init.as_ref().map(|e| self.expr(e));
                self.decl(*local, i, false, out, ind);
            }
            Stmt::Assign { dst, val } => {
                let (d, _) = self.lvalue(dst);
                let v = self.expr(val);
                pad(out, ind);
                let _ = writeln!(out, "{d} = {v};");
            }
            Stmt::Store { binding, index, val } => {
                let (b, r) = self.binding_root(*binding);
                let (a, _) = self.index_step(&r, index);
                let v = self.expr(val);
                pad(out, ind);
                let _ = writeln!(out, "{b}{a} = {v};");
            }
            Stmt::SharedStore { shared, index, val } => {
                let (b, r) = self.shared_root(*shared);
                let (a, _) = self.index_step(&r, index);
                let v = self.expr(val);
                pad(out, ind);
                let _ = writeln!(out, "{b}{a} = {v};");
            }
            Stmt::If { cond, then_body, else_body } => {
                let c = self.expr(cond);
                pad(out, ind);
                let _ = writeln!(out, "if ({c}) {{");
                self.block(then_body, out, ind + 1);
                pad(out, ind);
                if else_body.is_empty() {
                    out.push_str("}\n");
                } else {
                    out.push_str("} else {\n");
                    self.block(else_body, out, ind + 1);
                    pad(out, ind);
                    out.push_str("}\n");
                }
            }
            Stmt::Loop { body, continuing, break_if } => {
                let hoist = !continuing.is_empty() || break_if.is_some();
                if hoist && has_continue(body) {
                    self.labels += 1;
                    let first = format!("_first{}", self.labels);
                    pad(out, ind);
                    out.push_str("{\n");
                    for id in top_decls(body) {
                        self.undeclared(id, out, ind + 1);
                        self.hoisted.insert(id.0);
                    }
                    pad(out, ind + 1);
                    let _ = writeln!(out, "bool {first} = true;");
                    pad(out, ind + 1);
                    out.push_str("for (;;) {\n");
                    pad(out, ind + 2);
                    let _ = writeln!(out, "if (!{first}) {{");
                    self.block(continuing, out, ind + 3);
                    if let Some(c) = break_if {
                        let c = self.expr(c);
                        pad(out, ind + 3);
                        let _ = writeln!(out, "if ({c}) {{ break; }}");
                    }
                    pad(out, ind + 2);
                    out.push_str("}\n");
                    pad(out, ind + 2);
                    let _ = writeln!(out, "{first} = false;");
                    self.block(body, out, ind + 2);
                    pad(out, ind + 1);
                    out.push_str("}\n");
                    pad(out, ind);
                    out.push_str("}\n");
                } else {
                    pad(out, ind);
                    out.push_str("for (;;) {\n");
                    if hoist {
                        for id in top_decls(body) {
                            self.undeclared(id, out, ind + 1);
                            self.hoisted.insert(id.0);
                        }
                    }
                    self.block(body, out, ind + 1);
                    self.block(continuing, out, ind + 1);
                    if let Some(c) = break_if {
                        let c = self.expr(c);
                        pad(out, ind + 1);
                        let _ = writeln!(out, "if ({c}) {{ break; }}");
                    }
                    pad(out, ind);
                    out.push_str("}\n");
                }
            }
            Stmt::Switch { selector, cases, default } => {
                let sel = self.expr(selector);
                pad(out, ind);
                let _ = writeln!(out, "switch ({sel}) {{");
                for (c, body, fall) in cases {
                    pad(out, ind + 1);
                    match c {
                        SwitchCase::I32(v) => {
                            let _ = writeln!(out, "case {v}: {{");
                        }
                        SwitchCase::U32(v) => {
                            let _ = writeln!(out, "case {v}u: {{");
                        }
                    }
                    self.block(body, out, ind + 2);
                    if !*fall {
                        pad(out, ind + 2);
                        out.push_str("break;\n");
                    }
                    pad(out, ind + 1);
                    out.push_str("}\n");
                }
                pad(out, ind + 1);
                out.push_str("default: {\n");
                self.block(default, out, ind + 2);
                pad(out, ind + 2);
                out.push_str("break;\n");
                pad(out, ind + 1);
                out.push_str("}\n");
                pad(out, ind);
                out.push_str("}\n");
            }
            Stmt::Break => {
                pad(out, ind);
                out.push_str("break;\n");
            }
            Stmt::Continue => {
                pad(out, ind);
                out.push_str("continue;\n");
            }
            Stmt::Return(v) => {
                pad(out, ind);
                match v {
                    Some(e) => {
                        let e = self.expr(e);
                        let _ = writeln!(out, "return {e};");
                    }
                    None => out.push_str("return;\n"),
                }
            }
            Stmt::Barrier(scope) => {
                pad(out, ind);
                out.push_str(match scope {
                    BarrierScope::Workgroup => "threadgroup_barrier(mem_flags::mem_threadgroup);\n",
                    BarrierScope::Storage => "threadgroup_barrier(mem_flags::mem_device);\n",
                    BarrierScope::StorageAndWorkgroup => "threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);\n",
                });
            }
            Stmt::AtomicStore { target, val } => {
                let elem = self.ety(val).scalar().unwrap_or(ScalarTy::U32);
                let (p, _) = self.atomic_ptr(target, elem);
                let v = self.expr(val);
                let v = if elem == ScalarTy::F32 { format!("as_type<uint>({v})") } else { v };
                pad(out, ind);
                let _ = writeln!(out, "atomic_store_explicit({p}, {v}, memory_order_relaxed);");
            }
            Stmt::AtomicRmw { dst, op, elem_ty, target, val } => {
                let (p, space) = self.atomic_ptr(target, *elem_ty);
                let v = self.expr(val);
                let call = self.rmw_expr(*op, *elem_ty, &p, space, &v);
                match dst {
                    Some(d) => self.decl(*d, Some(call), true, out, ind),
                    None => {
                        pad(out, ind);
                        let _ = writeln!(out, "(void){call};");
                    }
                }
            }
            Stmt::AtomicCompareExchange { dst_old, dst_exchanged, elem_ty, target, cmp, val } => {
                let (p, space) = self.atomic_ptr(target, *elem_ty);
                let c = self.expr(cmp);
                let v = self.expr(val);
                let f = *elem_ty == ScalarTy::F32;
                let (bits_c, bits_v) = if f { (format!("as_type<uint>({c})"), format!("as_type<uint>({v})")) } else { (c, v) };
                let et = if *elem_ty == ScalarTy::I32 { "int" } else { "uint" };
                let at = if *elem_ty == ScalarTy::I32 { "atomic_int" } else { "atomic_uint" };
                self.undeclared(*dst_old, out, ind);
                self.undeclared(*dst_exchanged, out, ind);
                pad(out, ind);
                let _ = writeln!(out, "{{");
                pad(out, ind + 1);
                let _ = writeln!(out, "{space} {at}* _target = {p}; {et} _cmp = {bits_c}; {et} _value = {bits_v}; {et} _e; bool _ok;");
                pad(out, ind + 1);
                let _ = writeln!(out, "do {{ _e = _cmp; _ok = atomic_compare_exchange_weak_explicit(_target, &_e, _value, memory_order_relaxed, memory_order_relaxed); }} while (!_ok && _e == _cmp);");
                pad(out, ind + 1);
                let old = if f { "as_type<float>(_e)" } else { "_e" };
                let _ = writeln!(out, "l{} = {old}; l{} = _ok;", dst_old.0, dst_exchanged.0);
                pad(out, ind);
                let _ = writeln!(out, "}}");
            }
            Stmt::Discard(e) => {
                let e = self.expr(e);
                pad(out, ind);
                let _ = writeln!(out, "{e};");
            }
        }
    }

    // ---- bindings and context ----------------------------------------------------------------

    fn binding_space(&self, b: u32) -> &'static str {
        match self.m.bindings[b as usize].kind {
            BindingKind::Uniform => "constant",
            BindingKind::StorageRead => "const device",
            BindingKind::StorageReadWrite => "device",
        }
    }

    fn binding_elem(&mut self, b: u32) -> String {
        let d = &self.m.bindings[b as usize];
        let (elem, unsized_, stride) = (d.elem_ty.clone(), d.is_unsized_array, d.element_stride);
        if unsized_ {
            let s = stride.unwrap_or_else(|| self.natural_stride(&elem));
            self.elem_str(&elem, s)
        } else {
            self.ty(&elem, false)
        }
    }

    fn binding_param(&mut self, b: u32) -> String {
        let space = self.binding_space(b);
        let e = self.binding_elem(b);
        let d = &self.m.bindings[b as usize];
        if d.is_unsized_array {
            let restrict = if matches!(d.elem_ty, KirType::Atomic(_)) { "" } else { " __restrict" };
            format!("{space} {e}*{restrict} b{b}")
        } else {
            format!("{space} {e}& b{b}")
        }
    }

    fn active(&self) -> Vec<u32> {
        (0..self.rank.len() as u32).filter(|b| self.rank[*b as usize].is_some()).collect()
    }

    fn shared_param(&mut self, s: u32) -> String {
        let d = &self.m.shared[s as usize];
        let (elem, len) = (d.elem_ty.clone(), d.len);
        let t = self.ty(&elem, false);
        if len.is_some() { format!("threadgroup {t}* s{s}") } else { format!("threadgroup {t}& s{s}") }
    }

    fn ctx_params(&mut self, fn_idx: usize) -> Vec<String> {
        let r = self.res[fn_idx].clone();
        let mut v = Vec::new();
        if r.grid {
            v.push("constant LumaGrid& luma".to_owned());
        }
        for b in self.active() {
            if r.binds.contains(&b) {
                v.push(self.binding_param(b));
            }
        }
        if r.lens && !self.lens.is_empty() {
            v.push("constant uint* _ll".to_owned());
        }
        for &s in &r.shared {
            v.push(self.shared_param(s));
        }
        if r.privs && !self.m.privates.is_empty() {
            v.push("thread _LumaPriv& _pv".to_owned());
        }
        v
    }

    fn ctx_args(&self, fn_idx: usize) -> Vec<String> {
        let r = &self.res[fn_idx];
        let mut v = Vec::new();
        if r.grid {
            v.push("luma".to_owned());
        }
        for b in self.active() {
            if r.binds.contains(&b) {
                v.push(format!("b{b}"));
            }
        }
        if r.lens && !self.lens.is_empty() {
            v.push("_ll".to_owned());
        }
        for s in &r.shared {
            v.push(format!("s{s}"));
        }
        if r.privs && !self.m.privates.is_empty() {
            v.push("_pv".to_owned());
        }
        v
    }
}

fn builtin_attr(b: BuiltinVar) -> &'static str {
    match b {
        BuiltinVar::GlobalInvocationId => "thread_position_in_grid",
        BuiltinVar::LocalInvocationId => "thread_position_in_threadgroup",
        BuiltinVar::LocalInvocationIndex => "thread_index_in_threadgroup",
        BuiltinVar::WorkgroupId => "threadgroup_position_in_grid",
        BuiltinVar::NumWorkgroups => "threadgroups_per_grid",
        BuiltinVar::SubgroupInvocationId => "thread_index_in_simdgroup",
        BuiltinVar::SubgroupSize => "threads_per_simdgroup",
    }
}

/// Lowers `module` to MSL. `used[b]` says whether binding `b` is passed to the kernel; `None`
/// passes all of them.
pub fn emit(module: &KernelModule, used: Option<&[bool]>) -> MslTranslated {
    emit_with_workgroup_init(module, used, true)
}

/// Like [`emit`]; `initialize` zero-fills workgroup memory at kernel start (WGSL semantics, needed
/// for untrusted kernels). Trusted kernels that initialize it themselves pass `false`.
pub fn emit_with_workgroup_init(module: &KernelModule, used: Option<&[bool]>, initialize: bool) -> MslTranslated {
    let m = module;
    let mut rank = vec![None; m.bindings.len()];
    let mut next = 0;
    for (b, r) in rank.iter_mut().enumerate() {
        if used.is_none_or(|u| u.get(b).copied().unwrap_or(false)) {
            *r = Some(next);
            next += 1;
        }
    }
    let active_count = next;

    // Element counts the kernel reads.
    let reachable = m.reachable_functions();
    let mut lens: Vec<u32> = Vec::new();
    let mut scan = |block: &[Stmt]| {
        walk_stmts(block, &mut |s| {
            s.for_each_expr(&mut |e| {
                e.walk(&mut |x| {
                    if let Expr::ArrayLength { binding } = x
                        && rank.get(*binding as usize).is_some_and(|r| r.is_some())
                        && !lens.contains(binding)
                    {
                        lens.push(*binding);
                    }
                })
            })
        });
    };
    scan(&m.entry_body);
    for (f, live) in m.functions.iter().zip(&reachable) {
        if *live {
            scan(&f.body);
        }
    }
    lens.sort_unstable();

    let slots = 1 + active_count + usize::from(!lens.is_empty());
    let abi = if slots <= 31 { MslAbi::Direct } else { MslAbi::ArgBuffer };

    let mut res: Vec<Res> = vec![Res::default(); m.functions.len()];
    for (f, r) in m.functions.iter().zip(res.iter_mut()) {
        walk_stmts(&f.body, &mut |s| {
            match s {
                Stmt::Store { binding, .. } => {
                    r.binds.insert(*binding);
                }
                Stmt::SharedStore { shared, .. } => {
                    r.shared.insert(*shared);
                }
                Stmt::Assign { dst, .. } => lvalue_res(dst, r),
                Stmt::AtomicStore { target, .. }
                | Stmt::AtomicRmw { target, .. }
                | Stmt::AtomicCompareExchange { target, .. } => atomic_res(target, r),
                _ => {}
            }
            s.for_each_expr(&mut |e| {
                e.walk(&mut |x| match x {
                    Expr::UniformRead { binding, .. } | Expr::Load { binding, .. } | Expr::StorageRead { binding, .. } => {
                        r.binds.insert(*binding);
                    }
                    Expr::SharedLoad { shared, .. } | Expr::SharedRead { shared, .. } => {
                        r.shared.insert(*shared);
                    }
                    Expr::Private(_) => r.privs = true,
                    Expr::GridSize(_) => r.grid = true,
                    Expr::ArrayLength { binding } => {
                        if rank.get(*binding as usize).is_some_and(|x| x.is_some()) {
                            r.lens = true;
                        }
                    }
                    Expr::AtomicLoad { target, .. } => atomic_res(target, r),
                    Expr::Call { func, .. } => {
                        r.calls.insert(func.0);
                    }
                    _ => {}
                })
            });
        });
    }
    loop {
        let mut changed = false;
        for i in 0..res.len() {
            for c in res[i].calls.clone() {
                let callee = res[c as usize].clone();
                let r = &mut res[i];
                let before = (r.grid, r.lens, r.privs, r.binds.len(), r.shared.len());
                r.grid |= callee.grid;
                r.lens |= callee.lens;
                r.privs |= callee.privs;
                r.binds.extend(callee.binds);
                r.shared.extend(callee.shared);
                changed |= before != (r.grid, r.lens, r.privs, r.binds.len(), r.shared.len());
            }
        }
        if !changed {
            break;
        }
    }

    let mut em = Em {
        res,
        m,
        rank,
        lens,
        helpers: BTreeMap::new(),
        strided: false,
        locals: Vec::new(),
        params: Vec::new(),
        hoisted: HashSet::new(),
        labels: 0,
    };

    // Structs, dependencies first.
    let mut structs = String::new();
    let mut done = vec![false; m.structs.len()];
    fn visit(em: &mut Em, i: usize, done: &mut Vec<bool>, out: &mut String) {
        if done[i] {
            return;
        }
        done[i] = true;
        let m = em.m;
        fn deps(t: &KirType, v: &mut Vec<usize>) {
            match t {
                KirType::Struct(s) => v.push(*s as usize),
                KirType::Array { elem, .. } => deps(elem, v),
                _ => {}
            }
        }
        let mut d = Vec::new();
        for f in &m.structs[i].fields {
            deps(&f.ty, &mut d);
        }
        for j in d {
            visit(em, j, done, out);
        }
        em.struct_text(i, out);
    }
    for i in 0..m.structs.len() {
        visit(&mut em, i, &mut done, &mut structs);
    }

    // Privates.
    let mut privs = String::new();
    if !m.privates.is_empty() {
        privs.push_str("struct _LumaPriv {\n    uint _d;\n");
        for (i, p) in m.privates.iter().enumerate() {
            let t = em.ty(&p.ty, false);
            let _ = writeln!(privs, "    {t} p{i};");
        }
        privs.push_str("};\n");
    }

    // Helper functions.
    let mut funcs = String::new();
    let mut prototypes = String::new();
    for (i, f) in m.functions.iter().enumerate() {
        if !reachable[i] {
            continue;
        }
        em.locals = f.locals.clone();
        em.params = f.params.clone();
        em.hoisted.clear();
        let mut ps = em.ctx_params(i);
        for (j, (_, t)) in f.params.iter().enumerate() {
            let t = em.ty(t, false);
            ps.push(format!("{t} a{j}"));
        }
        let ret = match &f.ret_ty {
            Some(t) => em.ty(t, false),
            None => "void".to_owned(),
        };
        let signature = format!("static inline {ret} f{i}({})", ps.join(", "));
        let _ = writeln!(prototypes, "{signature};");
        let _ = writeln!(funcs, "{signature} {{");
        em.block(&f.body, &mut funcs, 1);
        funcs.push_str("}\n");
    }

    // Kernel.
    em.locals = m.entry_locals.clone();
    em.params = Vec::new();
    em.hoisted.clear();
    let mut args: Vec<String> = vec!["constant LumaGrid& luma [[buffer(0)]]".to_owned()];
    let mut args_struct = String::new();
    let mut prologue = String::new();
    let active = em.active();
    match abi {
        MslAbi::Direct => {
            for (r, b) in active.iter().enumerate() {
                let p = em.binding_param(*b);
                args.push(format!("{p} [[buffer({})]]", r + 1));
            }
            if !em.lens.is_empty() {
                args.push(format!("constant uint* _luma_lengths [[buffer({})]]", active.len() + 1));
                prologue.push_str("    constant uint* _ll = _luma_lengths;\n");
            }
        }
        MslAbi::ArgBuffer => {
            args.push("constant LumaArgs& _luma_args [[buffer(1)]]".to_owned());
            args_struct.push_str("struct LumaArgs {\n");
            for b in &active {
                let space = em.binding_space(*b);
                let e = em.binding_elem(*b);
                let _ = writeln!(args_struct, "    {space} {e}* b{b};");
                let d = &m.bindings[*b as usize];
                if d.is_unsized_array {
                    let restrict = if matches!(d.elem_ty, KirType::Atomic(_)) { "" } else { " __restrict" };
                    let _ = writeln!(prologue, "    {space} {e}*{restrict} b{b} = _luma_args.b{b};");
                } else {
                    let _ = writeln!(prologue, "    {space} {e}& b{b} = *_luma_args.b{b};");
                }
            }
            if !em.lens.is_empty() {
                let _ = writeln!(args_struct, "    uint _lens[{}];", em.lens.len());
                prologue.push_str("    constant uint* _ll = _luma_args._lens;\n");
            }
            args_struct.push_str("};\n");
        }
    }
    for (k, (b, _)) in m.entry_builtins.iter().enumerate() {
        let t = em.ty(&b.ty(), false);
        args.push(format!("{t} _bi{k} [[{}]]", builtin_attr(*b)));
    }
    let reuse = m.entry_builtins.iter().position(|(b, _)| *b == BuiltinVar::LocalInvocationIndex);
    let init_shared = initialize && !m.shared.is_empty();
    let tidx = match reuse {
        Some(k) => format!("_bi{k}"),
        None => "_tidx".to_owned(),
    };
    if init_shared && reuse.is_none() {
        args.push("uint _tidx [[thread_index_in_threadgroup]]".to_owned());
    }

    let mut body = String::new();
    body.push_str(&prologue);
    for (k, (b, l)) in m.entry_builtins.iter().enumerate() {
        let t = em.ty(&b.ty(), false);
        let _ = writeln!(body, "    const {t} l{} = _bi{k};", l.0);
    }
    for (i, s) in m.shared.iter().enumerate() {
        let t = em.ty(&s.elem_ty, false);
        match s.len {
            Some(n) => {
                let _ = writeln!(body, "    threadgroup {t} s{i}[{n}];");
            }
            None => {
                let _ = writeln!(body, "    threadgroup {t} s{i};");
            }
        }
    }
    if init_shared {
        let wg: u64 = m.workgroup_size.iter().map(|&x| u64::from(x.max(1))).product();
        for (i, s) in m.shared.iter().enumerate() {
            let atomic = matches!(s.elem_ty, KirType::Atomic(_));
            let zero = em.zero(&s.elem_ty);
            let target = match s.len {
                Some(_) => format!("s{i}[_z]"),
                None => format!("s{i}"),
            };
            let stmt = if atomic {
                format!("atomic_store_explicit(&{target}, 0, memory_order_relaxed);")
            } else {
                format!("{target} = {zero};")
            };
            match s.len {
                Some(n) => {
                    let _ = writeln!(body, "    for (uint _z = {tidx}; _z < {n}u; _z += {wg}u) {{ {stmt} }}");
                }
                None => {
                    let _ = writeln!(body, "    if ({tidx} == 0u) {{ {stmt} }}");
                }
            }
        }
        body.push_str("    threadgroup_barrier(mem_flags::mem_threadgroup);\n");
    }
    if !m.privates.is_empty() {
        body.push_str("    _LumaPriv _pv{};\n");
    }
    for (i, p) in m.privates.iter().enumerate() {
        if let Some(init) = &p.init {
            let e = em.expr(init);
            let _ = writeln!(body, "    _pv.p{i} = {e};");
        }
    }
    em.block(&m.entry_body, &mut body, 1);

    let wg_total: u64 = m.workgroup_size.iter().map(|&x| u64::from(x.max(1))).product();
    let entry_name = "luma_main".to_owned();

    let mut src = String::new();
    src.push_str("#include <metal_stdlib>\nusing namespace metal;\n");
    if m.uses_exact_f32 {
        src.push_str("#pragma METAL fp math_mode(safe)\n#pragma METAL fp contract(off)\n");
    }
    src.push_str("struct LumaGrid { packed_uint3 size; uint _pad; };\n");
    src.push_str("static_assert(sizeof(LumaGrid) == 16, \"LumaGrid layout\");\n");
    if em.strided {
        src.push_str("template <typename T, uint PAD> struct _LumaStrided { T v; uchar _p[PAD]; };\n");
    }
    src.push_str(&structs);
    for h in em.helpers.values() {
        src.push_str(h);
    }
    src.push_str(&args_struct);
    src.push_str(&privs);
    src.push_str(&prototypes);
    src.push_str(&funcs);
    let _ = writeln!(src, "[[max_total_threads_per_threadgroup({wg_total})]]");
    let _ = writeln!(src, "kernel void {entry_name}(\n    {}) {{", args.join(",\n    "));
    src.push_str(&body);
    src.push_str("}\n");

    MslTranslated { source: src, entry_name, abi, array_lengths: em.lens }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn module() -> KernelModule {
        KernelModule {
            label: "t".into(),
            entry: "main".into(),
            workgroup_size: [64, 1, 1],
            uses_exact_f32: false,
            structs: Vec::new(),
            bindings: Vec::new(),
            shared: Vec::new(),
            privates: Vec::new(),
            functions: Vec::new(),
            entry_builtins: Vec::new(),
            entry_locals: Vec::new(),
            entry_body: Vec::new(),
        }
    }

    fn rw(name: &str, elem: KirType) -> BindingDecl {
        BindingDecl { name: name.into(), kind: BindingKind::StorageReadWrite, elem_ty: elem, is_unsized_array: true, element_stride: Some(4), min_size: None }
    }

    fn lit(v: f32) -> Expr {
        Expr::LitF32(v)
    }

    fn store(binding: u32, val: Expr) -> Stmt {
        Stmt::Store { binding, index: Expr::LitU32(0), val }
    }

    fn load(binding: u32, ty: KirType) -> Expr {
        Expr::Load { binding, index: Expr::LitU32(0).boxed(), ty }
    }

    #[test]
    fn padded_uniform_layout() {
        let mut m = module();
        m.structs.push(StructDecl {
            name: "Padded".into(),
            fields: vec![
                StructField { name: "a".into(), ty: KirType::F32, offset: 0 },
                StructField { name: "v".into(), ty: KirType::vec(ScalarTy::F32, 3), offset: 16 },
                StructField { name: "b".into(), ty: KirType::F32, offset: 28 },
            ],
            size: 32,
            align: 16,
        });
        m.bindings.push(BindingDecl {
            name: "p".into(),
            kind: BindingKind::Uniform,
            elem_ty: KirType::Struct(0),
            is_unsized_array: false,
            element_stride: None,
            min_size: Some(32),
        });
        m.bindings.push(rw("out", KirType::F32));
        let v = Expr::UniformRead { binding: 0, path: vec![AccessStep::Field(1)], ty: KirType::vec(ScalarTy::F32, 3) };
        m.entry_body.push(store(1, Expr::Access { base: v.boxed(), step: AccessStep::Field(0), ty: KirType::F32 }));
        let t = emit(&m, None);
        assert_eq!(t.abi, MslAbi::Direct);
        assert!(t.source.contains("struct LumaGrid { packed_uint3 size; uint _pad; };"));
        assert!(t.source.contains("struct alignas(16) S0"));
        assert!(t.source.contains("uchar _pad0[12];"));
        assert!(t.source.contains("packed_float3 m1;"));
        assert!(t.source.contains("float m2;"));
        assert!(t.source.contains("static_assert(sizeof(S0) == 32"));
        assert!(t.source.contains("constant S0& b0 [[buffer(1)]]"));
        assert!(t.source.contains("device float* __restrict b1 [[buffer(2)]]"));
    }

    #[test]
    fn mulhi_and_tanh() {
        let mut m = module();
        m.bindings.push(rw("o", KirType::U32));
        m.bindings.push(rw("f", KirType::F32));
        m.entry_body.push(store(
            0,
            Expr::Binary { op: IrBinOp::MulHiU32, lhs: Expr::LitU32(7).boxed(), rhs: Expr::LitU32(9).boxed(), ty: KirType::U32 },
        ));
        m.entry_body.push(store(1, Expr::Unary { op: IrUnOp::Tanh, arg: lit(0.5).boxed(), ty: KirType::F32 }));
        let t = emit(&m, None);
        assert!(t.source.contains("mulhi(7u, 9u)"));
        assert!(t.source.contains("select(tanh(x), select(-1.0f, 1.0f, x > 0.0f), fabs(x) > 15.0f)"));
    }

    #[test]
    fn select_on_float4() {
        let f4 = KirType::vec(ScalarTy::F32, 4);
        let b4 = KirType::vec(ScalarTy::Bool, 4);
        let mut m = module();
        m.bindings.push(rw("a", f4.clone()));
        m.bindings.push(rw("c", f4.clone()));
        let cond = Expr::Binary { op: IrBinOp::Lt, lhs: load(0, f4.clone()).boxed(), rhs: load(1, f4.clone()).boxed(), ty: b4 };
        let sel = Expr::Ternary { op: IrTernaryOp::Select, a: cond.boxed(), b: load(0, f4.clone()).boxed(), c: load(1, f4.clone()).boxed(), ty: f4.clone() };
        m.entry_body.push(Stmt::Store { binding: 0, index: Expr::LitU32(1), val: sel });
        let scalar = Expr::Ternary { op: IrTernaryOp::Select, a: Expr::LitBool(true).boxed(), b: load(0, f4.clone()).boxed(), c: load(1, f4.clone()).boxed(), ty: f4.clone() };
        m.entry_body.push(Stmt::Store { binding: 0, index: Expr::LitU32(2), val: scalar });
        let t = emit(&m, None);
        assert!(t.source.contains("select(b1[0u], b0[0u], (b0[0u] < b1[0u]))"));
        assert!(t.source.contains("((true) ? (b0[0u]) : (b1[0u]))"));
    }

    #[test]
    fn subgroup_ops() {
        let mut m = module();
        m.bindings.push(rw("o", KirType::U32));
        let u4 = KirType::vec(ScalarTy::U32, 4);
        let x = || Some(Expr::LitU32(3).boxed());
        let sg = |op, arg: Option<Box<Expr>>, lane: Option<Box<Expr>>, ty: KirType| Expr::Subgroup { op, arg, lane, ty };
        m.entry_body.push(store(0, Expr::Access { base: sg(IrSubgroupOp::Ballot, Some(Expr::LitBool(true).boxed()), None, u4).boxed(), step: AccessStep::Field(0), ty: KirType::U32 }));
        m.entry_body.push(store(0, sg(IrSubgroupOp::Add, x(), None, KirType::U32)));
        m.entry_body.push(store(0, sg(IrSubgroupOp::Min, x(), None, KirType::U32)));
        m.entry_body.push(store(0, sg(IrSubgroupOp::Max, x(), None, KirType::U32)));
        m.entry_body.push(store(0, sg(IrSubgroupOp::ExclusiveAdd, x(), None, KirType::U32)));
        m.entry_body.push(store(0, sg(IrSubgroupOp::Shuffle, x(), Some(Expr::LitU32(1).boxed()), KirType::U32)));
        let src = emit(&m, None).source;
        for s in [
            "uint4(ulong(simd_ballot(true)), 0u, 0u, 0u)",
            "simd_sum(3u)",
            "simd_min(3u)",
            "simd_max(3u)",
            "simd_prefix_exclusive_sum(3u)",
            "simd_shuffle(3u, ushort(1u))",
        ] {
            assert!(src.contains(s), "missing {s}");
        }
    }

    #[test]
    fn many_bindings_use_an_argument_buffer() {
        let mut m = module();
        for i in 0..80 {
            m.bindings.push(rw(&format!("b{i}"), KirType::F32));
            m.entry_body.push(store(i, lit(1.0)));
        }
        let t = emit(&m, None);
        assert_eq!(t.abi, MslAbi::ArgBuffer);
        assert!(t.source.contains("struct LumaArgs"));
        assert!(t.source.contains("device float* b79;"));
        assert!(t.source.contains("constant LumaArgs& _luma_args [[buffer(1)]]"));
        assert!(t.source.contains("device float* __restrict b79 = _luma_args.b79;"));
        // 30 active bindings fit directly (31 slots); 31 do not.
        let used30: Vec<bool> = (0..80).map(|i| i < 30).collect();
        assert_eq!(emit(&m, Some(&used30)).abi, MslAbi::Direct);
        let used31: Vec<bool> = (0..80).map(|i| i < 31).collect();
        assert_eq!(emit(&m, Some(&used31)).abi, MslAbi::ArgBuffer);
    }

    #[test]
    fn sparse_active_bindings_are_renumbered() {
        let mut m = module();
        for i in 0..3 {
            m.bindings.push(rw(&format!("b{i}"), KirType::F32));
        }
        m.entry_body.push(store(2, lit(1.0)));
        let t = emit(&m, Some(&[false, false, true]));
        assert!(t.source.contains("device float* __restrict b2 [[buffer(1)]]"));
        assert!(!t.source.contains("b0"));
    }

    #[test]
    fn array_length_abi() {
        let mut m = module();
        m.bindings.push(BindingDecl { kind: BindingKind::StorageRead, ..rw("a", KirType::F32) });
        m.bindings.push(rw("o", KirType::U32));
        m.entry_body.push(store(1, Expr::ArrayLength { binding: 0 }));
        let t = emit(&m, None);
        assert_eq!(t.array_lengths, vec![0]);
        assert_eq!(t.abi, MslAbi::Direct);
        assert!(t.source.contains("constant uint* _luma_lengths [[buffer(3)]]"));
        assert!(t.source.contains("_ll[0]"));

        // With the length slot the count is 1 + 30 + 1: the argument buffer takes over.
        let mut m = module();
        for i in 0..31 {
            m.bindings.push(rw(&format!("b{i}"), KirType::U32));
            m.entry_body.push(store(i, Expr::ArrayLength { binding: 0 }));
        }
        let t = emit(&m, Some(&(0..31).map(|i| i < 30).collect::<Vec<_>>()));
        assert_eq!(t.abi, MslAbi::ArgBuffer);
        assert_eq!(t.array_lengths, vec![0]);
        assert!(t.source.contains("uint _lens[1];"));
        assert!(t.source.contains("constant uint* _ll = _luma_args._lens;"));
    }

    #[test]
    fn loop_with_continuing_shares_body_locals() {
        let mut m = module();
        m.bindings.push(rw("o", KirType::U32));
        m.entry_locals.push(("i".into(), KirType::U32, true));
        m.entry_locals.push(("t".into(), KirType::U32, false));
        m.entry_body.push(Stmt::Var { local: LocalId(0), init: Some(Expr::LitU32(0)) });
        m.entry_body.push(Stmt::Loop {
            body: vec![
                Stmt::Let { local: LocalId(1), init: Expr::Local(LocalId(0)) },
                Stmt::If { cond: Expr::LitBool(true), then_body: vec![Stmt::Continue], else_body: vec![] },
                store(0, Expr::Local(LocalId(1))),
            ],
            continuing: vec![Stmt::Assign {
                dst: LValue::Local(LocalId(0)),
                val: Expr::Binary { op: IrBinOp::Add, lhs: Expr::Local(LocalId(1)).boxed(), rhs: Expr::LitU32(1).boxed(), ty: KirType::U32 },
            }],
            break_if: Some(Expr::Binary { op: IrBinOp::Ge, lhs: Expr::Local(LocalId(0)).boxed(), rhs: Expr::LitU32(8).boxed(), ty: KirType::BOOL }),
        });
        let src = emit(&m, None).source;
        assert!(src.contains("for (;;) {"));
        assert!(src.contains("uint l1;"));
        assert!(src.contains("l1 = l0;"));
        assert!(src.contains("bool _first1 = true;"));
        assert!(src.contains("if (!_first1) {"));
        assert!(!src.contains("goto"));
        assert!(src.contains("l0 = (l1 + 1u);"));
        assert!(src.contains("if ((l0 >= 8u)) { break; }"));
    }

    #[test]
    fn float_atomics_use_compare_exchange() {
        let mut m = module();
        m.bindings.push(BindingDecl { elem_ty: KirType::Atomic(ScalarTy::U32), ..rw("a", KirType::U32) });
        m.entry_body.push(Stmt::AtomicRmw {
            dst: None,
            op: IrAtomicOp::Add,
            elem_ty: ScalarTy::F32,
            target: AtomicTarget::Storage { binding: 0, index: Some(Expr::LitU32(0).boxed()) },
            val: lit(1.0),
        });
        let src = emit(&m, None).source;
        assert!(src.contains("atomic_compare_exchange_weak_explicit"));
        assert!(src.contains("_luma_af32_add_device("));
        assert!(src.contains("device atomic_uint* b0"));
    }

    #[test]
    fn exact_kernels_disable_contraction() {
        let mut m = module();
        m.uses_exact_f32 = true;
        let src = emit(&m, None).source;
        assert!(src.contains("fp contract(off)"));
        assert!(src.contains("kernel void luma_main("));
    }

    fn shared_module() -> KernelModule {
        let mut m = module();
        m.shared.push(SharedDecl { name: "h".into(), elem_ty: KirType::Scalar(ScalarTy::F16), len: Some(5) });
        m.shared.push(SharedDecl { name: "f".into(), elem_ty: KirType::BOOL, len: None });
        m.shared.push(SharedDecl { name: "a".into(), elem_ty: KirType::Atomic(ScalarTy::U32), len: Some(3) });
        m
    }

    #[test]
    fn workgroup_memory_is_initialized_per_element_type() {
        let src = emit(&shared_module(), None).source;
        assert!(src.contains("uint _tidx [[thread_index_in_threadgroup]]"));
        assert!(src.contains("for (uint _z = _tidx; _z < 5u; _z += 64u) { s0[_z] = half(0); }"));
        assert!(src.contains("if (_tidx == 0u) { s1 = false; }"));
        assert!(src.contains("atomic_store_explicit(&s2[_z], 0, memory_order_relaxed);"));
        assert!(!src.contains("sizeof(s"));
        assert!(src.contains("threadgroup_barrier(mem_flags::mem_threadgroup);"));
    }

    #[test]
    fn trusted_kernels_skip_workgroup_initialization() {
        let src = emit_with_workgroup_init(&shared_module(), None, false).source;
        assert!(!src.contains("_tidx"));
        assert!(!src.contains("threadgroup_barrier"));
        assert!(src.contains("threadgroup half s0[5];"));
    }

    #[test]
    fn workgroup_initialization_reuses_the_index_builtin() {
        let mut m = shared_module();
        m.entry_locals.push(("li".into(), KirType::U32, false));
        m.entry_builtins.push((BuiltinVar::LocalInvocationIndex, LocalId(0)));
        let src = emit(&m, None).source;
        assert!(!src.contains("_tidx"));
        assert_eq!(src.matches("thread_index_in_threadgroup").count(), 1);
        assert!(src.contains("_z = _bi0;"));
    }


    #[test]
    fn bool_vectors_use_componentwise_bit_ops() {
        let b4 = KirType::vec(ScalarTy::Bool, 4);
        let mut m = module();
        m.entry_locals.push(("x".into(), b4.clone(), false));
        let bin = |op| Expr::Binary { op, lhs: Expr::ZeroVal(b4.clone()).boxed(), rhs: Expr::ZeroVal(b4.clone()).boxed(), ty: b4.clone() };
        m.entry_body.push(Stmt::Let { local: LocalId(0), init: bin(IrBinOp::BitAnd) });
        m.entry_body.push(Stmt::Let { local: LocalId(0), init: bin(IrBinOp::BitOr) });
        let src = emit(&m, None).source;
        assert!(src.contains("bool4(uint4(bool4(false)) & uint4(bool4(false)))"));
        assert!(src.contains("bool4(uint4(bool4(false)) | uint4(bool4(false)))"));
        assert!(!src.contains("&&"));
    }
}
