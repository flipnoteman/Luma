//! Function builder for NVVM IR text: fresh names, blocks, byte-addressed memory helpers.

use std::collections::{BTreeMap, BTreeSet};

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

pub(crate) struct Fb {
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
        let t = self.fresh();
        self.ins(&format!("{t} = {rhs}"));
        t
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
        let p = self.fresh();
        let size = size.max(1);
        let align = align.max(1);
        self.allocas.push_str(&format!("  {a} = alloca [{size} x i8], align {align}\n"));
        self.allocas.push_str(&format!("  {p} = bitcast [{size} x i8]* {a} to i8*\n"));
        p
    }

    pub fn gep(&mut self, ptr: &Op, off: &Op) -> Op {
        self.def(&format!("getelementptr inbounds i8, i8* {ptr}, i64 {off}"))
    }

    pub fn gep_const(&mut self, ptr: &Op, off: u64) -> Op {
        if off == 0 {
            return ptr.clone();
        }
        self.def(&format!("getelementptr inbounds i8, i8* {ptr}, i64 {off}"))
    }

    pub fn load(&mut self, ptr: &Op, off: u32, s: ScalarTy, tbaa: bool) -> Op {
        let p = self.gep_const(ptr, u64::from(off));
        let t = memty(s);
        let tp = self.def(&format!("bitcast i8* {p} to {t}*"));
        let tag = if tbaa { format!(", !tbaa !{}", tbaa_tag(s)) } else { String::new() };
        let v = self.def(&format!("load {t}, {t}* {tp}, align {}{tag}", s.size_bytes()));
        if s == ScalarTy::Bool {
            self.def(&format!("icmp ne i32 {v}, 0"))
        } else {
            v
        }
    }

    pub fn store(&mut self, ptr: &Op, off: u32, s: ScalarTy, v: &Op, tbaa: bool) {
        let p = self.gep_const(ptr, u64::from(off));
        let t = memty(s);
        let tp = self.def(&format!("bitcast i8* {p} to {t}*"));
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
        let tp = self.def(&format!("bitcast i8* {ptr} to <{lanes} x {t}>*"));
        let tag = if tbaa { format!(", !tbaa !{}", tbaa_tag(s)) } else { String::new() };
        let v = self.def(&format!("load <{lanes} x {t}>, <{lanes} x {t}>* {tp}, align {}{tag}", lanes * s.size_bytes()));
        (0..lanes).map(|k| self.def(&format!("extractelement <{lanes} x {t}> {v}, i32 {k}"))).collect()
    }

    /// Stores all lanes of a vector at `ptr` with one access; see [`Fb::vector_access`].
    pub fn store_vec(&mut self, ptr: &Op, s: ScalarTy, ops: &[Op], tbaa: bool) {
        let t = llty(s);
        let lanes = ops.len() as u32;
        let mut v = "undef".to_string();
        for (k, op) in ops.iter().enumerate() {
            v = self.def(&format!("insertelement <{lanes} x {t}> {v}, {t} {op}, i32 {k}"));
        }
        let tp = self.def(&format!("bitcast i8* {ptr} to <{lanes} x {t}>*"));
        let tag = if tbaa { format!(", !tbaa !{}", tbaa_tag(s)) } else { String::new() };
        self.ins(&format!("store <{lanes} x {t}> {v}, <{lanes} x {t}>* {tp}, align {}{tag}", lanes * s.size_bytes()));
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

    pub fn finish(self) -> String {
        format!("entry:\n{}{}", self.allocas, self.body)
    }
}
