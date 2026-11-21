//! Function builder for NVVM IR text: fresh names, blocks, byte-addressed memory helpers.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use luma_core::kir::*;

/// An SSA operand: `"%t12"` or a literal.
pub(crate) type Op = String;

#[derive(Clone, Debug)]
pub(crate) enum Val {
    /// A scalar (one op) or vector (one op per lane).
    Lanes { elem: ScalarTy, ops: Vec<Op> },
    /// A struct or array: `i8*` to a temporary nothing else writes.
    Mem { ptr: Op, ty: KirType },
}

/// SSA type of a scalar.
pub(crate) fn llty(s: ScalarTy) -> &'static str {
    match s {
        ScalarTy::Bool => "i1",
        ScalarTy::U32 | ScalarTy::I32 => "i32",
        ScalarTy::F16 => "i16",
        ScalarTy::F32 => "float",
        ScalarTy::F64 => "double",
    }
}

/// In-memory type of a scalar (`Bool` is a 0/1 word).
pub(crate) fn memty(s: ScalarTy) -> &'static str {
    match s {
        ScalarTy::Bool => "i32",
        _ => llty(s),
    }
}

/// Size in bytes of `ty` in memory.
pub(crate) fn size_of(ty: &KirType, structs: &[StructDecl]) -> u32 {
    match ty {
        KirType::Scalar(s) | KirType::Atomic(s) => s.size_bytes(),
        KirType::Vec { elem, lanes } => elem.size_bytes() * u32::from(*lanes),
        KirType::Array { len, stride, .. } => len * stride,
        KirType::Struct(i) => structs[*i as usize].size,
    }
}

/// Alignment in bytes of `ty` in memory, at least 1.
pub(crate) fn align_of(ty: &KirType, structs: &[StructDecl]) -> u32 {
    match ty {
        KirType::Scalar(s) | KirType::Atomic(s) => s.size_bytes(),
        KirType::Vec { elem, lanes } => match lanes {
            2 => 2 * elem.size_bytes(),
            4 => 4 * elem.size_bytes(),
            _ => elem.size_bytes(),
        },
        KirType::Array { elem, .. } => align_of(elem, structs),
        KirType::Struct(i) => structs[*i as usize].align.max(1),
    }
}

fn tbaa_tag(s: ScalarTy) -> u32 {
    match s {
        ScalarTy::U32 | ScalarTy::I32 => 7,
        ScalarTy::F32 => 8,
        ScalarTy::F64 => 9,
        ScalarTy::F16 => 10,
        ScalarTy::Bool => 11,
    }
}

/// How an `i8*` made by [`Fb::gep_const`] / [`Fb::gep_elem`] is reached from its base, so that an
/// access through it becomes one typed `getelementptr` (the byte arithmetic and casts that made it
/// are then dead and [`dce`] drops them).
#[derive(Clone)]
enum PtrDef {
    Off { base: Op, off: u64 },
    Elem { base: Op, idx: Op, stride: u32 },
}

pub(crate) struct Fb {
    /// The entry block's prologue: stack slots and the casts of pointers that are defined there.
    pub allocas: String,
    pub body: String,
    pub terminated: bool,
    /// The block instructions are appended to (for `phi` operands).
    pub cur: String,
    /// `declare …` lines this function needs.
    pub decls: BTreeSet<String>,
    /// Helper name -> full `define internal …` text.
    pub helpers: BTreeMap<String, String>,
    next: u32,
    ptrs: HashMap<Op, PtrDef>,
    /// Pointers that dominate every instruction (parameters, stack slots, prologue casts): their typed
    /// views are made once, in the prologue.
    entry: HashSet<Op>,
    views: HashMap<(Op, String), Op>,
}

impl Fb {
    pub fn new() -> Fb {
        Fb {
            allocas: String::new(),
            body: String::new(),
            terminated: false,
            cur: "entry".into(),
            decls: BTreeSet::new(),
            helpers: BTreeMap::new(),
            next: 0,
            ptrs: HashMap::new(),
            entry: HashSet::new(),
            views: HashMap::new(),
        }
    }

    pub fn fresh(&mut self) -> Op {
        let n = self.next;
        self.next += 1;
        format!("%t{n}")
    }

    /// A fresh block label, without `%`.
    pub fn label(&mut self) -> String {
        let n = self.next;
        self.next += 1;
        format!("b{n}")
    }

    pub fn ins(&mut self, text: &str) {
        if self.terminated {
            let l = self.label();
            self.start(&l);
        }
        self.body.push_str("  ");
        self.body.push_str(text);
        self.body.push('\n');
    }

    pub fn def(&mut self, rhs: &str) -> Op {
        if let Some(folded) = fold(rhs) {
            return folded;
        }
        let t = self.fresh();
        self.ins(&format!("{t} = {rhs}"));
        t
    }

    /// Defines a value in the entry prologue, where every later instruction sees it; for values
    /// that depend on nothing but parameters.
    pub fn def_entry(&mut self, rhs: &str) -> Op {
        let t = self.fresh();
        self.allocas.push_str(&format!("  {t} = {rhs}\n"));
        self.entry.insert(t.clone());
        t
    }

    /// Declares `op` (a parameter) as defined before the first instruction.
    pub fn mark_entry(&mut self, op: &Op) {
        self.entry.insert(op.clone());
    }

    pub fn br(&mut self, label: &str) {
        if self.terminated {
            return;
        }
        self.ins(&format!("br label %{label}"));
        self.terminated = true;
    }

    pub fn cond_br(&mut self, c: &Op, then_l: &str, else_l: &str) {
        self.ins(&format!("br i1 {c}, label %{then_l}, label %{else_l}"));
        self.terminated = true;
    }

    pub fn ret(&mut self, text: &str) {
        self.ins(text);
        self.terminated = true;
    }

    pub fn start(&mut self, label: &str) {
        self.br(label);
        self.body.push_str(label);
        self.body.push_str(":\n");
        self.terminated = false;
        self.cur = label.to_string();
    }

    /// An `i8*` to a fresh stack slot of `size` bytes (at least 1).
    pub fn alloca(&mut self, size: u32, align: u32) -> Op {
        let a = self.fresh();
        let size = size.max(1);
        let align = align.max(1);
        self.allocas.push_str(&format!("  {a} = alloca [{size} x i8], align {align}\n"));
        self.def_entry(&format!("bitcast [{size} x i8]* {a} to i8*"))
    }

    /// `ptr + off` bytes, as an `i8*`.
    pub fn gep(&mut self, ptr: &Op, off: &Op) -> Op {
        self.def(&format!("getelementptr inbounds i8, i8* {ptr}, i64 {off}"))
    }

    pub fn gep_const(&mut self, ptr: &Op, off: u64) -> Op {
        if off == 0 {
            return ptr.clone();
        }
        let t = self.def(&format!("getelementptr inbounds i8, i8* {ptr}, i64 {off}"));
        self.ptrs.insert(t.clone(), PtrDef::Off { base: ptr.clone(), off });
        t
    }

    /// `ptr + zext(idx) * stride` bytes, as an `i8*`; `idx` is an `i32`.
    pub fn gep_elem(&mut self, ptr: &Op, idx: &Op, stride: u32) -> Op {
        let z = self.def(&format!("zext i32 {idx} to i64"));
        if let Ok(v) = z.parse::<u64>() {
            return self.gep_const(ptr, v * u64::from(stride));
        }
        let off = if stride == 1 { z.clone() } else { self.def(&format!("mul nuw i64 {z}, {stride}")) };
        let t = self.gep(ptr, &off);
        self.ptrs.insert(t.clone(), PtrDef::Elem { base: ptr.clone(), idx: z, stride });
        t
    }

    /// `ptr` viewed as a pointer to `t`.
    fn view(&mut self, ptr: &Op, t: &str) -> Op {
        if self.entry.contains(ptr) {
            let key = (ptr.clone(), t.to_string());
            if let Some(v) = self.views.get(&key) {
                return v.clone();
            }
            let v = self.def_entry(&format!("bitcast i8* {ptr} to {t}*"));
            self.views.insert(key, v.clone());
            v
        } else {
            self.def(&format!("bitcast i8* {ptr} to {t}*"))
        }
    }

    /// A pointer to the `t` (of `size` bytes) at `ptr + off`: one typed `getelementptr` of the
    /// root `ptr` was computed from when its byte offsets are multiples of `size`.
    pub fn typed_ptr(&mut self, ptr: &Op, off: u32, t: &str, size: u32) -> Op {
        let (mut root, mut c, mut dynamic) = (ptr.clone(), u64::from(off), None::<(Op, u32)>);
        while let Some(d) = self.ptrs.get(&root) {
            match d.clone() {
                PtrDef::Off { base, off } => {
                    c += off;
                    root = base;
                }
                PtrDef::Elem { base, idx, stride } => {
                    if dynamic.is_some() {
                        break;
                    }
                    dynamic = Some((idx, stride));
                    root = base;
                }
            }
        }
        let size = u64::from(size);
        let aligned = c % size == 0 && dynamic.as_ref().is_none_or(|&(_, stride)| u64::from(stride) % size == 0);
        if !aligned {
            let p = self.gep_const(ptr, u64::from(off));
            return self.view(&p, t);
        }
        let base = self.view(&root, t);
        let mut at = c / size;
        match dynamic {
            None if at == 0 => base,
            None => self.def(&format!("getelementptr inbounds {t}, {t}* {base}, i64 {at}")),
            Some((idx, stride)) => {
                let k = u64::from(stride) / size;
                let i = if k == 1 { idx } else { self.def(&format!("mul nuw i64 {idx}, {k}")) };
                let i = if std::mem::take(&mut at) == 0 { i } else { self.def(&format!("add nuw i64 {i}, {}", c / size)) };
                self.def(&format!("getelementptr inbounds {t}, {t}* {base}, i64 {i}"))
            }
        }
    }

    /// A pointer to the scalar of type `s` at `ptr + off`.
    pub fn scalar_ptr(&mut self, ptr: &Op, off: u32, s: ScalarTy) -> Op {
        self.typed_ptr(ptr, off, memty(s), s.size_bytes())
    }

    pub fn load(&mut self, ptr: &Op, off: u32, s: ScalarTy, tbaa: bool) -> Op {
        let t = memty(s);
        let tp = self.typed_ptr(ptr, off, t, s.size_bytes());
        let tag = if tbaa { format!(", !tbaa !{}", tbaa_tag(s)) } else { String::new() };
        let v = self.def(&format!("load {t}, {t}* {tp}, align {}{tag}", s.size_bytes()));
        if s == ScalarTy::Bool {
            self.def(&format!("icmp ne i32 {v}, 0"))
        } else {
            v
        }
    }

    pub fn store(&mut self, ptr: &Op, off: u32, s: ScalarTy, v: &Op, tbaa: bool) {
        let t = memty(s);
        let tp = self.typed_ptr(ptr, off, t, s.size_bytes());
        let v = if s == ScalarTy::Bool { self.def(&format!("zext i1 {v} to i32")) } else { v.clone() };
        let tag = if tbaa { format!(", !tbaa !{}", tbaa_tag(s)) } else { String::new() };
        self.ins(&format!("store {t} {v}, {t}* {tp}, align {}{tag}", s.size_bytes()));
    }

    /// Whether a `lanes`-lane vector of `s` is read and written as one aligned vector access
    /// (`ld.v2`/`ld.v4`), as CUDA's naturally aligned `float2`/`float4` are.
    pub fn vector_access(s: ScalarTy, lanes: u32) -> bool {
        s != ScalarTy::Bool && matches!(lanes, 2 | 4)
    }

    /// Loads all lanes of a vector at `ptr` with one access; see [`Fb::vector_access`].
    pub fn load_vec(&mut self, ptr: &Op, s: ScalarTy, lanes: u32, tbaa: bool) -> Vec<Op> {
        let t = llty(s);
        let vt = format!("<{lanes} x {t}>");
        let tp = self.typed_ptr(ptr, 0, &vt, lanes * s.size_bytes());
        let tag = if tbaa { format!(", !tbaa !{}", tbaa_tag(s)) } else { String::new() };
        let v = self.def(&format!("load {vt}, {vt}* {tp}, align {}{tag}", lanes * s.size_bytes()));
        (0..lanes).map(|k| self.def(&format!("extractelement {vt} {v}, i32 {k}"))).collect()
    }

    /// Stores all lanes of a vector at `ptr` with one access; see [`Fb::vector_access`].
    pub fn store_vec(&mut self, ptr: &Op, s: ScalarTy, ops: &[Op], tbaa: bool) {
        let t = llty(s);
        let lanes = ops.len() as u32;
        let vt = format!("<{lanes} x {t}>");
        let mut v = "undef".to_string();
        if ops.iter().all(|o| !o.starts_with('%')) {
            // Constants: one vector constant instead of a chain of `insertelement`s.
            v = format!("<{}>", ops.iter().map(|o| format!("{t} {o}")).collect::<Vec<_>>().join(", "));
        } else {
            for (k, op) in ops.iter().enumerate() {
                v = self.def(&format!("insertelement {vt} {v}, {t} {op}, i32 {k}"));
            }
        }
        let tp = self.typed_ptr(ptr, 0, &vt, lanes * s.size_bytes());
        let tag = if tbaa { format!(", !tbaa !{}", tbaa_tag(s)) } else { String::new() };
        self.ins(&format!("store {vt} {v}, {vt}* {tp}, align {}{tag}", lanes * s.size_bytes()));
    }

    pub fn memcpy(&mut self, dst: &Op, src: &Op, size: u32, align: u32) {
        if dst == src || size == 0 {
            return;
        }
        self.declare("declare void @llvm.memcpy.p0i8.p0i8.i64(i8*, i8*, i64, i1)");
        let a = align.max(1);
        self.ins(&format!(
            "call void @llvm.memcpy.p0i8.p0i8.i64(i8* align {a} {dst}, i8* align {a} {src}, i64 {size}, i1 false)"
        ));
    }

    pub fn memzero(&mut self, dst: &Op, size: u32, align: u32) {
        if size == 0 {
            return;
        }
        self.declare("declare void @llvm.memset.p0i8.i64(i8*, i8, i64, i1)");
        let a = align.max(1);
        self.ins(&format!("call void @llvm.memset.p0i8.i64(i8* align {a} {dst}, i8 0, i64 {size}, i1 false)"));
    }

    pub fn declare(&mut self, decl: &str) {
        if !self.decls.contains(decl) {
            self.decls.insert(decl.to_string());
        }
    }

    /// The function body: the prologue, then the blocks, without the instructions nothing uses.
    pub fn finish(self) -> String {
        dce(&format!("entry:\n{}{}", self.allocas, self.body), self.next as usize)
    }
}

/// `%t<digits>` names that occur in `line` (not counting the one defined, at its start).
fn for_each_use(line: &str, mut f: impl FnMut(usize)) {
    let b = line.as_bytes();
    let mut i = 0;
    while i + 2 < b.len() {
        if b[i] == b'%' && b[i + 1] == b't' && b[i + 2].is_ascii_digit() && (i == 0 || !b[i - 1].is_ascii_alphanumeric()) {
            let mut j = i + 2;
            let mut n = 0usize;
            while j < b.len() && b[j].is_ascii_digit() {
                n = n * 10 + usize::from(b[j] - b'0');
                j += 1;
            }
            f(n);
            i = j;
        } else {
            i += 1;
        }
    }
}

/// Whether the instruction `rhs` (what follows `%tN = `) has no effect besides its value.
fn is_pure(rhs: &str) -> bool {
    let op = rhs.split(' ').next().unwrap_or("");
    match op {
        "add" | "sub" | "mul" | "udiv" | "sdiv" | "urem" | "srem" | "and" | "or" | "xor" | "shl" | "lshr" | "ashr" | "fadd" | "fsub" | "fmul" | "fdiv" | "icmp" | "fcmp" | "select" | "zext" | "sext" | "trunc"
        | "bitcast" | "addrspacecast" | "getelementptr" | "extractelement" | "insertelement" | "phi" | "alloca" | "fpext" | "fptrunc" | "fptosi" | "fptoui" | "sitofp" | "uitofp" => true,
        "load" => !rhs.starts_with("load volatile") && !rhs.starts_with("load atomic"),
        "call" => rhs.find('@').is_some_and(|at| {
            let name = &rhs[at + 1..];
            !rhs[..at].contains("asm")
                && ["llvm.nvvm.read.ptx.sreg.", "__nv_", "llvm.cttz.", "llvm.ctlz.", "llvm.ctpop.", "llvm.convert.", "llvm.nvvm.sqrt.", "llvm.nvvm.rsqrt.", "llvm.nvvm.div."].iter().any(|p| name.starts_with(p))
        }),
        _ => false,
    }
}

/// Removes the instructions (one per line, `  %tN = …`) whose value nothing uses, and repeats for
/// what that frees. `names` bounds the `%t` numbers. The emitter makes values it may not need
/// (casts, byte offsets behind a typed access, thread coordinates); NVVM would drop them but
/// parsing them is a large part of its time on small kernels.
fn dce(text: &str, names: usize) -> String {
    let lines: Vec<&str> = text.lines().collect();
    let mut uses = vec![0u32; names + 1];
    let defined = |line: &str| -> Option<(usize, usize)> {
        let rest = line.strip_prefix("  %t")?;
        let end = rest.find(' ')?;
        let n: usize = rest[..end].parse().ok()?;
        rest[end..].starts_with(" = ").then_some((n, 4 + end + 3))
    };
    for line in &lines {
        let from = defined(line).map_or(0, |(_, at)| at);
        for_each_use(&line[from..], |n| {
            if let Some(u) = uses.get_mut(n) {
                *u += 1;
            }
        });
    }
    let mut keep = vec![true; lines.len()];
    for (k, line) in lines.iter().enumerate().rev() {
        let Some((n, at)) = defined(line) else { continue };
        if uses[n] == 0 && is_pure(&line[at..]) {
            keep[k] = false;
            for_each_use(&line[at..], |m| {
                if let Some(u) = uses.get_mut(m) {
                    *u -= 1;
                }
            });
        }
    }
    let mut out = String::with_capacity(text.len());
    for (line, keep) in lines.iter().zip(keep) {
        if keep {
            out.push_str(line);
            out.push('\n');
        }
    }
    out
}

/// A literal integer operand (`i1` as `true`/`false`).
fn lit(op: &str) -> Option<i64> {
    match op {
        "true" => Some(1),
        "false" => Some(0),
        _ => op.parse().ok(),
    }
}

/// The value of `rhs` when it is a constant or an identity (`x + 0`, `x * 1`, `select true, a, b`):
/// the operand to use in its place.
fn fold(rhs: &str) -> Option<Op> {
    let w: Vec<&str> = rhs.split(' ').collect();
    match *w.first()? {
        "select" if w.len() == 7 => {
            // select i1 C, T A, T B
            let (c, a, b) = (w[2].trim_end_matches(','), w[4].trim_end_matches(','), w[6]);
            match lit(c) {
                Some(1) => Some(a.to_string()),
                Some(_) => Some(b.to_string()),
                None => (a == b).then(|| a.to_string()),
            }
        }
        "zext" if w.len() == 5 && w[1] == "i32" && w[3] == "to" && w[4] == "i64" => lit(w[2]).map(|v| (v as u32).to_string()),
        "icmp" if w.len() == 5 => {
            let (a, b) = (lit(w[3].trim_end_matches(','))?, lit(w[4])?);
            let bits = match w[2] {
                "i32" => 32,
                "i64" => 64,
                _ => return None,
            };
            let (ua, ub) = if bits == 32 { (u64::from(a as u32), u64::from(b as u32)) } else { (a as u64, b as u64) };
            let (sa, sb) = if bits == 32 { (i64::from(a as i32), i64::from(b as i32)) } else { (a, b) };
            let r = match w[1] {
                "eq" => ua == ub,
                "ne" => ua != ub,
                "ult" => ua < ub,
                "ule" => ua <= ub,
                "ugt" => ua > ub,
                "uge" => ua >= ub,
                "slt" => sa < sb,
                "sle" => sa <= sb,
                "sgt" => sa > sb,
                "sge" => sa >= sb,
                _ => return None,
            };
            Some(r.to_string())
        }
        op @ ("add" | "sub" | "mul" | "and" | "or" | "xor" | "shl" | "lshr" | "ashr" | "udiv" | "sdiv") => {
            // op [nuw|nsw|exact] TY A, B
            let mut k = 1;
            while matches!(w.get(k), Some(&("nuw" | "nsw" | "exact"))) {
                k += 1;
            }
            if w.len() != k + 3 {
                return None;
            }
            let ty = w[k];
            let (a, b) = (w[k + 1].trim_end_matches(','), w[k + 2]);
            if ty == "i1" {
                return match (op, lit(a), lit(b)) {
                    ("and", Some(0), _) | ("and", _, Some(0)) => Some("false".into()),
                    ("and", Some(1), _) => Some(b.to_string()),
                    ("and", _, Some(1)) => Some(a.to_string()),
                    ("or", Some(1), _) | ("or", _, Some(1)) => Some("true".into()),
                    ("or", Some(0), _) => Some(b.to_string()),
                    ("or", _, Some(0)) => Some(a.to_string()),
                    _ => None,
                };
            }
            let bits: u32 = match ty {
                "i32" => 32,
                "i64" => 64,
                _ => return None,
            };
            let mask = if bits == 32 { u64::from(u32::MAX) } else { u64::MAX };
            let wrap = |v: u64| -> Op {
                if bits == 32 { (v as u32 as i32).to_string() } else { (v as i64).to_string() }
            };
            let (la, lb) = (lit(a).map(|v| v as u64 & mask), lit(b).map(|v| v as u64 & mask));
            if let (Some(x), Some(y)) = (la, lb) {
                let sx = if bits == 32 { i64::from(x as u32 as i32) } else { x as i64 };
                let sy = if bits == 32 { i64::from(y as u32 as i32) } else { y as i64 };
                let r = match op {
                    "add" => x.wrapping_add(y),
                    "sub" => x.wrapping_sub(y),
                    "mul" => x.wrapping_mul(y),
                    "and" => x & y,
                    "or" => x | y,
                    "xor" => x ^ y,
                    "shl" if y < u64::from(bits) => x << y,
                    "lshr" if y < u64::from(bits) => x >> y,
                    "ashr" if y < u64::from(bits) => (sx >> y) as u64,
                    "udiv" if y != 0 => x / y,
                    "sdiv" if sy != 0 && !(sy == -1 && sx == i64::MIN >> (64 - bits)) => (sx / sy) as u64,
                    _ => return None,
                };
                return Some(wrap(r & mask));
            }
            match (op, la, lb) {
                ("add" | "or" | "xor", Some(0), _) => Some(b.to_string()),
                ("add" | "sub" | "or" | "xor" | "shl" | "lshr" | "ashr", _, Some(0)) => Some(a.to_string()),
                ("mul", Some(1), _) => Some(b.to_string()),
                ("mul" | "udiv" | "sdiv", _, Some(1)) => Some(a.to_string()),
                ("mul", Some(0), _) | ("mul", _, Some(0)) | ("and", Some(0), _) | ("and", _, Some(0)) => Some("0".into()),
                ("and", Some(m), _) if m == mask => Some(b.to_string()),
                ("and", _, Some(m)) if m == mask => Some(a.to_string()),
                _ => None,
            }
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constants_and_identities_fold() {
        let f = |rhs: &str| fold(rhs);
        assert_eq!(f("add i32 %t1, 0").as_deref(), Some("%t1"));
        assert_eq!(f("add i32 0, %t1").as_deref(), Some("%t1"));
        assert_eq!(f("sub i32 %t1, 0").as_deref(), Some("%t1"));
        assert_eq!(f("mul i32 %t1, 1").as_deref(), Some("%t1"));
        assert_eq!(f("mul i32 %t1, 0").as_deref(), Some("0"));
        assert_eq!(f("mul nuw i64 5, 4").as_deref(), Some("20"));
        assert_eq!(f("add i32 2147483647, 1").as_deref(), Some("-2147483648"));
        assert_eq!(f("and i32 %t1, -1").as_deref(), Some("%t1"));
        assert_eq!(f("and i32 %t1, 31").as_deref(), None);
        assert_eq!(f("and i32 70, 31").as_deref(), Some("6"));
        assert_eq!(f("udiv i32 %t1, 1").as_deref(), Some("%t1"));
        assert_eq!(f("icmp eq i32 16, 0").as_deref(), Some("false"));
        assert_eq!(f("icmp ult i32 -1, 1").as_deref(), Some("false"));
        assert_eq!(f("icmp slt i32 -1, 1").as_deref(), Some("true"));
        assert_eq!(f("select i1 true, i32 7, i32 %t2").as_deref(), Some("7"));
        assert_eq!(f("select i1 false, i32 7, i32 %t2").as_deref(), Some("%t2"));
        assert_eq!(f("select i1 %t3, i32 %t2, i32 %t2").as_deref(), Some("%t2"));
        assert_eq!(f("zext i32 -1 to i64").as_deref(), Some("4294967295"));
        assert_eq!(f("and i1 %t1, true").as_deref(), Some("%t1"));
        assert_eq!(f("or i1 %t1, true").as_deref(), Some("true"));
        // What is undefined or not constant stays.
        assert_eq!(f("shl i32 1, 32"), None);
        assert_eq!(f("udiv i32 1, 0"), None);
        assert_eq!(f("sdiv i32 -2147483648, -1"), None);
        assert_eq!(f("add i32 %t1, %t2"), None);
        assert_eq!(f("fadd float %t1, 0x0000000000000000"), None);
        assert_eq!(f("icmp eq i8* %t1, null"), None);
    }

    #[test]
    fn unused_instructions_go_and_what_they_used_with_them() {
        let text = "entry:\n  %t1 = bitcast i8* %p to float*\n  %t2 = getelementptr inbounds float, float* %t1, i64 3\n  %t3 = load float, float* %t2, align 4\n  %t4 = fadd float %t3, %t3\n  %t5 = load volatile float, float* %t2, align 4\n  %t6 = call i32 asm sideeffect \"mov.u32 $0, %laneid;\", \"=r\"()\n  store float 0x0, float* %t1, align 4\n  ret void\n";
        let out = dce(text, 6);
        assert!(!out.contains("%t3 =") && !out.contains("%t4 ="), "{out}");
        // The volatile load and the asm have effects; the pointers behind them stay.
        assert!(out.contains("%t5 =") && out.contains("%t6 =") && out.contains("%t2 =") && out.contains("%t1 ="), "{out}");
        assert!(out.contains("store float"), "{out}");
    }

    #[test]
    fn an_element_access_is_one_typed_getelementptr() {
        let mut fb = Fb::new();
        let buf: Op = "%buf0".into();
        fb.mark_entry(&buf);
        let idx: Op = "%i".into();
        let p = fb.gep_elem(&buf, &idx, 4);
        let a = fb.load(&p, 0, ScalarTy::F32, true);
        let q = fb.gep_const(&p, 4);
        fb.store(&q, 0, ScalarTy::F32, &a, true);
        let text = fb.finish();
        // The view of the buffer is made once, in the prologue; the byte arithmetic is gone.
        assert_eq!(text.matches("bitcast i8* %buf0 to float*").count(), 1, "{text}");
        assert!(!text.contains("getelementptr inbounds i8"), "{text}");
        assert!(!text.contains("mul nuw"), "{text}");
        assert!(text.contains("load float, float* %t"), "{text}");
        assert!(text.contains("add nuw i64"), "{text}");
    }

    #[test]
    fn a_misaligned_offset_keeps_the_byte_address() {
        let mut fb = Fb::new();
        let buf: Op = "%buf0".into();
        fb.mark_entry(&buf);
        let p = fb.gep_const(&buf, 2);
        let v = fb.load(&p, 0, ScalarTy::F32, true);
        fb.store(&buf, 0, ScalarTy::F32, &v, true);
        let text = fb.finish();
        assert!(text.contains("getelementptr inbounds i8, i8* %buf0, i64 2"), "{text}");
    }
}
