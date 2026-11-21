//! Cranelift SIMD codegen for `luma-cpu`.
//!
//! Every barrier-free segment of the plan runs in 4-lane packets: each KIR scalar leaf is one
//! 128-bit vector (`F32X4` / `I32X4`, or a pair of `F64X2` for `f64`). `bool` lives as `-1`/`0`
//! lane masks. Divergence is handled with per-scope execution masks.

#![allow(deprecated)]

use std::collections::HashMap;
use std::sync::{Arc, LazyLock};

use cranelift_codegen::Context;
use cranelift_codegen::control::ControlPlane;
use cranelift_codegen::ir::condcodes::{FloatCC, IntCC};
use cranelift_codegen::ir::immediates::{Ieee32, Ieee64};
use cranelift_codegen::ir::types::{self, F32X4, F64X2, I32X4};
use cranelift_codegen::ir::{
    AbiParam, AtomicRmwOp, Block, BlockArg, ConstantData, Endianness, Function, InstBuilder,
    MemFlagsData, SigRef, Signature, Type, UserFuncName, Value,
};
use cranelift_codegen::isa::OwnedTargetIsa;
use cranelift_codegen::settings::{self, Configurable};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext, Variable};
use luma_core::kir::{
    AccessStep, AtomicTarget, BuiltinVar, CastKind, Expr, IrAtomicOp, IrBinOp, IrSubgroupOp,
    IrTernaryOp, IrUnOp, KernelModule, KirType, LValue, LocalId, ScalarTy, Stmt, StructDecl,
    SwitchCase, walk_stmts,
};
use luma_core::{BackendError, NativeKernel};

use crate::abi::{CpuMathTable, CpuPlan, PhaseStmt, WorkgroupCtx};
use crate::exec::{CostHint, ExecBlob};

/// A compiled workgroup function plus the plan it was lowered against.
#[derive(Debug)]
pub struct CompiledCpuKernel {
    pub exec: Arc<ExecBlob>,
    pub plan: CpuPlan,
    /// Learned run time, steering how many threads a launch uses.
    pub cost: CostHint,
}

impl NativeKernel for CompiledCpuKernel {}

fn build_isa(speed: bool) -> Result<OwnedTargetIsa, BackendError> {
    let err = |e: &dyn std::fmt::Display| BackendError::ShaderCompile(e.to_string());
    let mut flags = settings::builder();
    flags
        .set("opt_level", if speed { "speed" } else { "none" })
        .map_err(|e| err(&e))?;
    flags.set("is_pic", "true").map_err(|e| err(&e))?;
    flags
        .set("enable_verifier", if speed { "true" } else { "false" })
        .map_err(|e| err(&e))?;
    let isa_builder =
        cranelift_native::builder().map_err(|e| BackendError::ShaderCompile(e.to_string()))?;
    isa_builder
        .finish(settings::Flags::new(flags))
        .map_err(|e| err(&e))
}

static ISA_SPEED: LazyLock<Result<OwnedTargetIsa, BackendError>> =
    LazyLock::new(|| build_isa(true));
static ISA_FAST: LazyLock<Result<OwnedTargetIsa, BackendError>> =
    LazyLock::new(|| build_isa(false));

static ISA_KEY: LazyLock<String> = LazyLock::new(|| match ISA_SPEED.as_ref() {
    Ok(isa) => {
        let feats: Vec<String> = isa
            .isa_flags()
            .iter()
            .map(|v| format!("{}={}", v.name, v.value_string()))
            .collect();
        format!(
            "rev4-{}-{}-{}",
            isa.triple(),
            cranelift_codegen::VERSION,
            feats.join(",")
        )
    }
    Err(e) => format!("unsupported-{e}"),
});

/// Stable description of the host ISA and codegen version, for disk-cache keys.
pub fn host_isa_key() -> String {
    ISA_KEY.clone()
}

/// Revision of the generated code, part of the disk-cache key: bump it whenever the code this
/// module generates for the same kernel changes, so cached blobs from older builds are not reused.
pub(crate) const REVISION: u32 = 2;

// ---------------------------------------------------------------------------------------------
// Memory flags
// ---------------------------------------------------------------------------------------------

fn mf() -> MemFlagsData {
    MemFlagsData::new().with_notrap()
}

fn mf_ro() -> MemFlagsData {
    MemFlagsData::new()
        .with_notrap()
        .with_readonly()
        .with_can_move()
}

fn mf_bc() -> MemFlagsData {
    MemFlagsData::new().with_endianness(Endianness::Little)
}

// ---------------------------------------------------------------------------------------------
// Value model
// ---------------------------------------------------------------------------------------------

/// What is known about the lanes of a leaf: unknown, all equal, or `base + [0, 1, 2, 3]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    Gen,
    Uni,
    Lin,
}

/// One scalar component of a KIR value across the 4 lanes of a packet.
#[derive(Clone, Copy)]
struct Leaf {
    s: ScalarTy,
    a: Value,
    /// High pair (lanes 2, 3) of an `f64` leaf.
    b: Option<Value>,
    k: Kind,
}

#[derive(Clone)]
struct Val {
    ty: KirType,
    leaves: Vec<Leaf>,
}

type VarPair = (Variable, Option<Variable>);

struct LocalVars {
    ty: KirType,
    scs: Vec<ScalarTy>,
    vars: Vec<VarPair>,
    kinds: Vec<Kind>,
    /// A local that one `Let` defines and nothing else writes, and that no later segment reads:
    /// it is an SSA value (`direct`), not a `Variable`. Every `Variable` costs the frontend a map
    /// over all blocks, which for thousands of locals in a kernel of thousands of blocks is tens
    /// of gigabytes.
    let_only: bool,
    direct: Vec<Leaf>,
}

struct Frame {
    locals: Vec<LocalVars>,
    params: Vec<Val>,
    ret: Option<(KirType, Vec<ScalarTy>, Vec<VarPair>)>,
    entry: bool,
}

#[derive(Clone, Copy)]
enum ScopeKind {
    Root,
    CallRoot,
    Plain,
    Switch,
    Loop(Variable),
}

#[derive(Clone, Copy)]
struct Scope {
    var: Variable,
    full: bool,
    kind: ScopeKind,
}

struct PhaseLoop {
    cont: Block,
    exit: Block,
}

enum Step<'e> {
    Field(u32),
    Index(&'e Expr),
}

enum Root {
    Local(LocalId),
    Private(u32),
    Storage(u32),
    Shared(u32),
}

struct Builtins {
    li: Leaf,
    lid: [Leaf; 3],
    gid: [Leaf; 3],
    wid: [Leaf; 3],
    nwg: [Leaf; 3],
    sg_id: Leaf,
    sg_size: Leaf,
}

fn vty(s: ScalarTy) -> Type {
    match s {
        ScalarTy::F32 | ScalarTy::F16 => F32X4,
        ScalarTy::F64 => F64X2,
        _ => I32X4,
    }
}

fn leaf_scalars(ty: &KirType, structs: &[StructDecl], out: &mut Vec<ScalarTy>) {
    match ty {
        KirType::Scalar(s) | KirType::Atomic(s) => out.push(*s),
        KirType::Vec { elem, lanes } => out.extend(std::iter::repeat_n(*elem, usize::from(*lanes))),
        KirType::Array { elem, len, .. } => {
            for _ in 0..*len {
                leaf_scalars(elem, structs, out);
            }
        }
        KirType::Struct(id) => {
            for f in &structs[*id as usize].fields {
                leaf_scalars(&f.ty, structs, out);
            }
        }
    }
}

fn scalars_of(ty: &KirType, structs: &[StructDecl]) -> Vec<ScalarTy> {
    let mut v = Vec::new();
    leaf_scalars(ty, structs, &mut v);
    v
}

fn leaf_count(ty: &KirType, structs: &[StructDecl]) -> usize {
    match ty {
        KirType::Scalar(_) | KirType::Atomic(_) => 1,
        KirType::Vec { lanes, .. } => usize::from(*lanes),
        KirType::Array { elem, len, .. } => *len as usize * leaf_count(elem, structs),
        KirType::Struct(id) => structs[*id as usize]
            .fields
            .iter()
            .map(|f| leaf_count(&f.ty, structs))
            .sum(),
    }
}

/// `(byte offset, scalar)` of every leaf of `ty` under the WGSL host-shareable layout.
fn layout_leaves(ty: &KirType, structs: &[StructDecl], base: u32, out: &mut Vec<(u32, ScalarTy)>) {
    match ty {
        KirType::Scalar(s) | KirType::Atomic(s) => out.push((base, *s)),
        KirType::Vec { elem, lanes } => {
            for i in 0..u32::from(*lanes) {
                out.push((base + i * elem.size_bytes(), *elem));
            }
        }
        KirType::Array { elem, len, stride } => {
            for i in 0..*len {
                layout_leaves(elem, structs, base + i * stride, out);
            }
        }
        KirType::Struct(id) => {
            for f in &structs[*id as usize].fields {
                layout_leaves(&f.ty, structs, base + f.offset, out);
            }
        }
    }
}

/// Byte size of one leaf in a scratch slot (`f64` 8, everything else 4).
fn slot_leaf_bytes(s: ScalarTy) -> u32 {
    if s == ScalarTy::F64 { 8 } else { 4 }
}

fn lit_index(e: &Expr) -> Option<u32> {
    match e {
        Expr::LitU32(k) => Some(*k),
        Expr::LitI32(k) if *k >= 0 => Some(*k as u32),
        _ => None,
    }
}

fn lvalue_local(lv: &LValue) -> Option<LocalId> {
    match lv {
        LValue::Local(l) => Some(*l),
        LValue::Index { base, .. } | LValue::Field { base, .. } => lvalue_local(base),
        LValue::Private(_) | LValue::Storage { .. } | LValue::Shared { .. } => None,
    }
}

/// Which of the `n` locals of `body` are defined by exactly one `Let` and written by nothing else.
fn let_only_locals(n: usize, body: &[Stmt]) -> Vec<bool> {
    let mut lets = vec![0u32; n];
    let mut written = vec![false; n];
    walk_stmts(body, &mut |s| match s {
        Stmt::Let { local, .. } => lets[local.0 as usize] += 1,
        Stmt::Var { local, .. } => written[local.0 as usize] = true,
        Stmt::Assign { dst, .. } => {
            if let Some(l) = lvalue_local(dst) {
                written[l.0 as usize] = true;
            }
        }
        Stmt::AtomicRmw { dst: Some(d), .. } => written[d.0 as usize] = true,
        Stmt::AtomicCompareExchange {
            dst_old,
            dst_exchanged,
            ..
        } => {
            written[dst_old.0 as usize] = true;
            written[dst_exchanged.0 as usize] = true;
        }
        _ => {}
    });
    lets.iter()
        .zip(&written)
        .map(|(&c, &w)| c == 1 && !w)
        .collect()
}

fn expr_has_mem_or_div(e: &Expr) -> bool {
    let mut found = false;
    e.walk(&mut |x| match x {
        Expr::Load { .. }
        | Expr::StorageRead { .. }
        | Expr::SharedLoad { .. }
        | Expr::SharedRead { .. }
        | Expr::AtomicLoad { .. }
        | Expr::UniformRead { .. } => found = true,
        Expr::Binary {
            op: IrBinOp::Div | IrBinOp::Rem,
            ..
        }
        | Expr::Call { .. } => found = true,
        _ => {}
    });
    found
}

/// Which exits (return / break / continue) can reach the root scope of a statement list.
fn root_exits(stmts: &[Stmt], in_loop: bool, in_switch: bool, acc: &mut [bool; 3]) {
    for s in stmts {
        match s {
            Stmt::Return(_) => acc[0] = true,
            Stmt::Break if !in_loop && !in_switch => acc[1] = true,
            Stmt::Continue if !in_loop => acc[2] = true,
            Stmt::If {
                then_body,
                else_body,
                ..
            } => {
                root_exits(then_body, in_loop, in_switch, acc);
                root_exits(else_body, in_loop, in_switch, acc);
            }
            Stmt::Loop {
                body, continuing, ..
            } => {
                root_exits(body, true, in_switch, acc);
                root_exits(continuing, true, in_switch, acc);
            }
            Stmt::Switch { cases, default, .. } => {
                for (_, b, _) in cases {
                    root_exits(b, in_loop, true, acc);
                }
                root_exits(default, in_loop, true, acc);
            }
            _ => {}
        }
    }
}

/// Lowers `module` into position-independent machine code. Returns the kernel and its code bytes.
pub fn compile(
    module: &KernelModule,
    used: Option<&[bool]>,
    initialize_shared: bool,
) -> Result<(CompiledCpuKernel, Vec<u8>), BackendError> {
    let isa = ISA_SPEED.as_ref().map_err(Clone::clone)?;
    let ptr = isa.pointer_type();
    if ptr != types::I64 {
        return Err(BackendError::ShaderCompile(
            "luma-cpu requires a 64-bit host".into(),
        ));
    }
    let plan = crate::fission::plan(module, used, initialize_shared);

    let mut sig = Signature::new(isa.default_call_conv());
    sig.params.push(AbiParam::new(ptr));
    let mut func = Function::with_name_signature(UserFuncName::default(), sig);
    let mut fbc = FunctionBuilderContext::new();
    {
        let mut b = FunctionBuilder::new(&mut func, &mut fbc);
        let entry = b.create_block();
        b.append_block_params_for_function_params(entry);
        b.switch_to_block(entry);
        let ctx = b.block_params(entry)[0];
        let mut lower = Lower::new(b, module, &plan, ptr, ctx, entry);
        lower.emit_all();
        lower.b.seal_all_blocks();
        lower.b.finalize(isa.frontend_config());
    }

    let target_isa = if func.dfg.num_insts() > 2500 {
        ISA_FAST.as_ref().map_err(Clone::clone)?
    } else {
        isa
    };
    let mut cctx = Context::for_function(func);
    let compiled = cctx
        .compile(&**target_isa, &mut ControlPlane::default())
        .map_err(|e| BackendError::ShaderCompile(format!("{}: {e:?}", module.label)))?;
    if !compiled.buffer.relocs().is_empty() {
        return Err(BackendError::ShaderCompile(format!(
            "{}: generated code needs external relocations (host lacks required CPU features)",
            module.label
        )));
    }
    let bytes = compiled.code_buffer().to_vec();
    let blob = ExecBlob::new(&bytes)?;
    Ok((
        CompiledCpuKernel {
            exec: Arc::new(blob),
            plan,
            cost: CostHint::default(),
        },
        bytes,
    ))
}

struct Lower<'a, 'f> {
    b: FunctionBuilder<'f>,
    module: &'a KernelModule,
    plan: &'a CpuPlan,
    ptr: Type,
    ctx: Value,
    padded: u32,
    packets: u32,
    has_ret_multi: bool,
    pkt: Value,
    pkt_next: Block,
    active_var: Variable,
    broke_var: Variable,
    cont_var: Variable,
    bi: Option<Builtins>,
    frames: Vec<Frame>,
    scopes: Vec<Scope>,
    phase_loops: Vec<PhaseLoop>,
    sigs: HashMap<(u8, bool), SigRef>,
    /// Byte length (i64) of each storage binding pointer produced by `binding_ptr`.
    limits: HashMap<Value, Value>,
}

impl<'a, 'f> Lower<'a, 'f> {
    fn new(
        mut b: FunctionBuilder<'f>,
        module: &'a KernelModule,
        plan: &'a CpuPlan,
        ptr: Type,
        ctx: Value,
        entry: Block,
    ) -> Self {
        let padded = plan.invocations.next_multiple_of(4);
        let mut has_ret = false;
        walk_stmts(&module.entry_body, &mut |s| {
            has_ret |= matches!(s, Stmt::Return(_))
        });
        let pkt = b.ins().iconst(types::I32, 0);
        let active_var = b.declare_var(I32X4);
        let broke_var = b.declare_var(types::I32);
        let cont_var = b.declare_var(types::I32);
        let ones = b.ins().iconst(types::I32, -1);
        let ones = b.ins().splat(I32X4, ones);
        b.def_var(active_var, ones);
        b.def_var(broke_var, pkt);
        b.def_var(cont_var, pkt);
        Self {
            b,
            module,
            plan,
            ptr,
            ctx,
            padded,
            packets: plan.invocations.div_ceil(4),
            has_ret_multi: has_ret && plan.phases.len() > 1,
            pkt,
            pkt_next: entry,
            active_var,
            broke_var,
            cont_var,
            bi: None,
            frames: Vec::new(),
            scopes: Vec::new(),
            phase_loops: Vec::new(),
            sigs: HashMap::new(),
            limits: HashMap::new(),
        }
    }

    fn structs(&self) -> &'a [StructDecl] {
        &self.module.structs
    }

    fn emit_all(&mut self) {
        let module = self.module;
        let mut let_only = let_only_locals(module.entry_locals.len(), &module.entry_body);
        for (l, only) in let_only.iter_mut().enumerate() {
            *only &= self
                .plan
                .local_scratch_offsets
                .get(l)
                .is_none_or(Option::is_none);
        }
        for &(_, l) in &module.entry_builtins {
            let_only[l.0 as usize] = false;
        }
        let locals = module
            .entry_locals
            .iter()
            .zip(let_only)
            .map(|((_, ty, _), only)| self.make_local(ty, only))
            .collect();
        self.frames.push(Frame {
            locals,
            params: Vec::new(),
            ret: None,
            entry: true,
        });
        self.prologue();
        let plan = self.plan;
        self.phases(&plan.phases);
        self.b.ins().return_(&[]);
    }

    // -----------------------------------------------------------------------------------------
    // Constants and small helpers
    // -----------------------------------------------------------------------------------------

    fn iconst(&mut self, v: i32) -> Value {
        self.b.ins().iconst(types::I32, i64::from(v))
    }

    fn splat_i32(&mut self, v: i32) -> Value {
        let c = self.iconst(v);
        self.b.ins().splat(I32X4, c)
    }

    fn splat_u32(&mut self, v: u32) -> Value {
        self.splat_i32(v as i32)
    }

    fn splat_f32(&mut self, v: f32) -> Value {
        let c = self.b.ins().f32const(Ieee32::with_float(v));
        self.b.ins().splat(F32X4, c)
    }

    fn splat_f32_bits(&mut self, bits: u32) -> Value {
        let c = self.b.ins().f32const(Ieee32::with_bits(bits));
        self.b.ins().splat(F32X4, c)
    }

    fn splat_f64(&mut self, v: f64) -> Value {
        let c = self.b.ins().f64const(Ieee64::with_float(v));
        self.b.ins().splat(F64X2, c)
    }

    fn splat_f64_bits(&mut self, bits: u64) -> Value {
        let c = self.b.ins().f64const(Ieee64::with_bits(bits));
        self.b.ins().splat(F64X2, c)
    }

    fn vconst(&mut self, ty: Type, bytes: [u8; 16]) -> Value {
        let c = self
            .b
            .func
            .dfg
            .constants
            .insert(ConstantData::from(&bytes[..]));
        self.b.ins().vconst(ty, c)
    }

    fn iota(&mut self) -> Value {
        self.vconst(I32X4, [0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0])
    }

    fn zero_mask(&mut self) -> Value {
        self.splat_i32(0)
    }

    fn lf(&self, s: ScalarTy, a: Value) -> Leaf {
        Leaf {
            s,
            a,
            b: None,
            k: Kind::Gen,
        }
    }

    fn uni(&self, s: ScalarTy, a: Value) -> Leaf {
        Leaf {
            s,
            a,
            b: None,
            k: Kind::Uni,
        }
    }

    fn scalar_val(&self, s: ScalarTy, l: Leaf) -> Val {
        Val {
            ty: KirType::Scalar(s),
            leaves: vec![l],
        }
    }

    fn zero_leaf(&mut self, s: ScalarTy) -> Leaf {
        match s {
            ScalarTy::F64 => {
                let z = self.splat_f64(0.0);
                Leaf {
                    s,
                    a: z,
                    b: Some(z),
                    k: Kind::Uni,
                }
            }
            ScalarTy::F32 | ScalarTy::F16 => {
                let z = self.splat_f32(0.0);
                self.uni(s, z)
            }
            _ => {
                let z = self.splat_i32(0);
                self.uni(s, z)
            }
        }
    }

    fn zero_val(&mut self, ty: &KirType) -> Val {
        let scs = scalars_of(ty, self.structs());
        let leaves = scs.into_iter().map(|s| self.zero_leaf(s)).collect();
        Val {
            ty: ty.clone(),
            leaves,
        }
    }

    fn bitcast_ty(&mut self, ty: Type, v: Value) -> Value {
        if self.b.func.dfg.value_type(v) == ty {
            v
        } else {
            self.b.ins().bitcast(ty, mf_bc(), v)
        }
    }

    fn shl_n(&mut self, v: Value, n: i32) -> Value {
        let c = self.iconst(n);
        self.b.ins().ishl(v, c)
    }

    fn shr_n(&mut self, v: Value, n: i32) -> Value {
        let c = self.iconst(n);
        self.b.ins().ushr(v, c)
    }

    fn and_c(&mut self, v: Value, c: u32) -> Value {
        let k = self.splat_u32(c);
        self.b.ins().band(v, k)
    }

    fn icmp_c(&mut self, cc: IntCC, v: Value, c: u32) -> Value {
        let k = self.splat_u32(c);
        self.b.ins().icmp(cc, v, k)
    }

    /// Bitwise select with an `I32X4` lane mask, for any 128-bit vector type.
    fn bsel(&mut self, mask: Value, x: Value, y: Value) -> Value {
        let mask = self.bitcast_ty(I32X4, mask);
        let t = self.b.func.dfg.value_type(x);
        let xi = self.bitcast_ty(I32X4, x);
        let yi = self.bitcast_ty(I32X4, y);
        let r = self.b.ins().bitselect(mask, xi, yi);
        self.bitcast_ty(t, r)
    }

    /// Expands an `I32X4` lane mask to the two `I32X4`-shaped masks covering `f64` pairs.
    fn mask_pair(&mut self, m: Value) -> (Value, Value) {
        let l: Vec<Value> = (0..4u8).map(|i| self.b.ins().extractlane(m, i)).collect();
        let mut out = [m, m];
        for h in 0..2 {
            let v = self.b.ins().splat(I32X4, l[2 * h]);
            let v = self.b.ins().insertlane(v, l[2 * h + 1], 2u8);
            let v = self.b.ins().insertlane(v, l[2 * h + 1], 3u8);
            // Lanes 0,1 hold m[2h]; lanes 2,3 hold m[2h+1].
            out[h] = v;
        }
        (out[0], out[1])
    }

    fn select_leaf(&mut self, mask: Value, x: Leaf, y: Leaf) -> Leaf {
        if x.s == ScalarTy::F64 {
            let (ml, mh) = self.mask_pair(mask);
            let a = self.bsel(ml, x.a, y.a);
            let b = self.bsel(mh, x.b.unwrap(), y.b.unwrap());
            Leaf {
                s: x.s,
                a,
                b: Some(b),
                k: Kind::Gen,
            }
        } else {
            let a = self.bsel(mask, x.a, y.a);
            Leaf {
                s: x.s,
                a,
                b: None,
                k: Kind::Gen,
            }
        }
    }

    fn narrow_mask64(&mut self, lo: Value, hi: Value) -> Value {
        let mut out: Option<Value> = None;
        for i in 0..4u8 {
            let src = if i < 2 { lo } else { hi };
            let m = self.b.ins().extractlane(src, i % 2);
            let m = self.b.ins().ireduce(types::I32, m);
            out = Some(match out {
                None => self.b.ins().splat(I32X4, m),
                Some(v) => self.b.ins().insertlane(v, m, i),
            });
        }
        out.unwrap()
    }

    /// Applies a scalar function lane by lane and repacks the results into a vector of `ty`.
    fn lanewise(
        &mut self,
        ty: Type,
        srcs: &[Value],
        f: &dyn Fn(&mut FunctionBuilder<'f>, &[Value]) -> Value,
    ) -> Value {
        let lanes = ty.lane_count() as u8;
        let mut out: Option<Value> = None;
        for i in 0..lanes {
            let args: Vec<Value> = srcs
                .iter()
                .map(|&v| self.b.ins().extractlane(v, i))
                .collect();
            let r = f(&mut self.b, &args);
            out = Some(match out {
                None => self.b.ins().splat(ty, r),
                Some(v) => self.b.ins().insertlane(v, r, i),
            });
        }
        out.unwrap()
    }

    fn lane_scalars(&mut self, l: Leaf) -> [Value; 4] {
        if l.s == ScalarTy::F64 {
            let b = l.b.unwrap();
            [
                self.b.ins().extractlane(l.a, 0u8),
                self.b.ins().extractlane(l.a, 1u8),
                self.b.ins().extractlane(b, 0u8),
                self.b.ins().extractlane(b, 1u8),
            ]
        } else {
            [
                self.b.ins().extractlane(l.a, 0u8),
                self.b.ins().extractlane(l.a, 1u8),
                self.b.ins().extractlane(l.a, 2u8),
                self.b.ins().extractlane(l.a, 3u8),
            ]
        }
    }

    fn leaf_from_scalars(&mut self, s: ScalarTy, v: [Value; 4]) -> Leaf {
        if s == ScalarTy::F64 {
            let lo = self.b.ins().splat(F64X2, v[0]);
            let lo = self.b.ins().insertlane(lo, v[1], 1u8);
            let hi = self.b.ins().splat(F64X2, v[2]);
            let hi = self.b.ins().insertlane(hi, v[3], 1u8);
            Leaf {
                s,
                a: lo,
                b: Some(hi),
                k: Kind::Gen,
            }
        } else {
            let mut a = self.b.ins().splat(vty(s), v[0]);
            for (i, &x) in v.iter().enumerate().skip(1) {
                a = self.b.ins().insertlane(a, x, i as u8);
            }
            Leaf {
                s,
                a,
                b: None,
                k: Kind::Gen,
            }
        }
    }

    fn math_sig(&mut self, nargs: u8, is64: bool) -> SigRef {
        if let Some(&s) = self.sigs.get(&(nargs, is64)) {
            return s;
        }
        let t = if is64 { types::F64 } else { types::F32 };
        let mut sig = Signature::new(self.b.func.signature.call_conv);
        for _ in 0..nargs {
            sig.params.push(AbiParam::new(t));
        }
        sig.returns.push(AbiParam::new(t));
        let r = self.b.import_signature(sig);
        self.sigs.insert((nargs, is64), r);
        r
    }

    /// Calls a `CpuMathTable` routine: once per packet when there is a vectorised `f32` one, else once per lane.
    fn math_call(&mut self, off32: i32, off64: i32, is64: bool, args: &[Value]) -> Value {
        if !is64
            && args.len() == 1
            && let Some(poff) = CpuMathTable::packet_offset(off32)
        {
            // One call per packet: F32X4 in and out of `xmm0` (see `crate::vecmath`).
            let mut sig = Signature::new(self.b.func.signature.call_conv);
            sig.params.push(AbiParam::new(F32X4));
            sig.returns.push(AbiParam::new(F32X4));
            let sig = self.b.import_signature(sig);
            let table = self.b.ins().load(self.ptr, mf_ro(), self.ctx, WorkgroupCtx::OFFSET_MATH);
            let fp = self.b.ins().load(self.ptr, mf_ro(), table, poff);
            let call = self.b.ins().call_indirect(sig, fp, args);
            return self.b.inst_results(call)[0];
        }
        let sig = self.math_sig(args.len() as u8, is64);
        let table = self
            .b
            .ins()
            .load(self.ptr, mf_ro(), self.ctx, WorkgroupCtx::OFFSET_MATH);
        let fp = self
            .b
            .ins()
            .load(self.ptr, mf_ro(), table, if is64 { off64 } else { off32 });
        let ty = if is64 { F64X2 } else { F32X4 };
        let lanes = ty.lane_count() as u8;
        let mut out: Option<Value> = None;
        for i in 0..lanes {
            let a: Vec<Value> = args
                .iter()
                .map(|&v| self.b.ins().extractlane(v, i))
                .collect();
            let call = self.b.ins().call_indirect(sig, fp, &a);
            let r = self.b.inst_results(call)[0];
            out = Some(match out {
                None => self.b.ins().splat(ty, r),
                Some(v) => self.b.ins().insertlane(v, r, i),
            });
        }
        out.unwrap()
    }
}

// ---------------------------------------------------------------------------------------------
// Locals, scratch slots, packet setup, phases
// ---------------------------------------------------------------------------------------------

impl<'a, 'f> Lower<'a, 'f> {
    fn make_local(&mut self, ty: &KirType, let_only: bool) -> LocalVars {
        let scs = scalars_of(ty, self.structs());
        let mut vars = Vec::with_capacity(scs.len());
        let declared = if let_only { &[][..] } else { &scs[..] };
        for &s in declared {
            let z = self.zero_leaf(s);
            let v = self.b.declare_var(vty(s));
            self.b.def_var(v, z.a);
            let hi = z.b.map(|zb| {
                let h = self.b.declare_var(F64X2);
                self.b.def_var(h, zb);
                h
            });
            vars.push((v, hi));
        }
        let kinds = vec![Kind::Gen; scs.len()];
        LocalVars {
            ty: ty.clone(),
            scs,
            vars,
            kinds,
            let_only,
            direct: Vec::new(),
        }
    }

    fn frame(&self) -> &Frame {
        self.frames.last().unwrap()
    }

    fn read_local(&mut self, l: LocalId) -> Val {
        let lv = &self.frames.last().unwrap().locals[l.0 as usize];
        if lv.let_only {
            return Val {
                ty: lv.ty.clone(),
                leaves: lv.direct.clone(),
            };
        }
        let (ty, scs, vars, kinds) = (
            lv.ty.clone(),
            lv.scs.clone(),
            lv.vars.clone(),
            lv.kinds.clone(),
        );
        let mut leaves = Vec::with_capacity(vars.len());
        for i in 0..vars.len() {
            let a = self.b.use_var(vars[i].0);
            let b = vars[i].1.map(|h| self.b.use_var(h));
            leaves.push(Leaf {
                s: scs[i],
                a,
                b,
                k: kinds[i],
            });
        }
        Val { ty, leaves }
    }

    fn write_local_leaves(&mut self, l: LocalId, leaves: &[Leaf], kinds: bool) {
        let fi = self.frames.len() - 1;
        if self.frames[fi].locals[l.0 as usize].let_only {
            self.frames[fi].locals[l.0 as usize].direct = leaves.to_vec();
            return;
        }
        let vars = self.frames[fi].locals[l.0 as usize].vars.clone();
        for (i, leaf) in leaves.iter().enumerate() {
            self.b.def_var(vars[i].0, leaf.a);
            if let Some(h) = vars[i].1 {
                self.b.def_var(h, leaf.b.unwrap());
            }
            let lv = &mut self.frames[fi].locals[l.0 as usize];
            lv.kinds[i] = if kinds { leaf.k } else { Kind::Gen };
        }
        if self.frames[fi].entry {
            self.store_slot(l);
        }
    }

    fn slot_off(&self, l: LocalId) -> Option<u32> {
        if self.frames.len() == 1 {
            self.plan
                .local_scratch_offsets
                .get(l.0 as usize)
                .copied()
                .flatten()
        } else {
            None
        }
    }

    fn scratch_addr(&mut self, lb: u32) -> Value {
        let base = self
            .b
            .ins()
            .load(self.ptr, mf_ro(), self.ctx, WorkgroupCtx::OFFSET_SCRATCH);
        let pk = self.b.ins().uextend(self.ptr, self.pkt);
        let po = self.b.ins().imul_imm(pk, i64::from(4 * lb));
        self.b.ins().iadd(base, po)
    }

    fn scratch_load(&mut self, off: u32, scs: &[ScalarTy]) -> Vec<Leaf> {
        let mut out = Vec::with_capacity(scs.len());
        let mut prefix = 0u32;
        for &s in scs {
            let lb = slot_leaf_bytes(s);
            let addr = self.scratch_addr(lb);
            let at = (off + prefix * self.padded) as i32;
            let a = self.b.ins().load(vty(s), mf(), addr, at);
            let b = (s == ScalarTy::F64).then(|| self.b.ins().load(F64X2, mf(), addr, at + 16));
            out.push(Leaf {
                s,
                a,
                b,
                k: Kind::Gen,
            });
            prefix += lb;
        }
        out
    }

    fn scratch_store(&mut self, off: u32, leaves: &[Leaf]) {
        let mut prefix = 0u32;
        for l in leaves {
            let lb = slot_leaf_bytes(l.s);
            let addr = self.scratch_addr(lb);
            let at = (off + prefix * self.padded) as i32;
            self.b.ins().store(mf(), l.a, addr, at);
            if let Some(h) = l.b {
                self.b.ins().store(mf(), h, addr, at + 16);
            }
            prefix += lb;
        }
    }

    fn store_slot(&mut self, l: LocalId) {
        if let Some(off) = self.slot_off(l) {
            let v = self.read_local(l);
            self.scratch_store(off, &v.leaves);
        }
    }

    fn load_slot(&mut self, l: LocalId) {
        if let Some(off) = self.slot_off(l) {
            let scs = self.frames[0].locals[l.0 as usize].scs.clone();
            let leaves = self.scratch_load(off, &scs);
            let vars = self.frames[0].locals[l.0 as usize].vars.clone();
            for (i, leaf) in leaves.iter().enumerate() {
                self.b.def_var(vars[i].0, leaf.a);
                if let Some(h) = vars[i].1 {
                    self.b.def_var(h, leaf.b.unwrap());
                }
                self.frames[0].locals[l.0 as usize].kinds[i] = Kind::Gen;
            }
        }
    }

    fn kadd(a: Kind, b: Kind) -> Kind {
        match (a, b) {
            (Kind::Uni, Kind::Uni) => Kind::Uni,
            (Kind::Lin, Kind::Uni) | (Kind::Uni, Kind::Lin) => Kind::Lin,
            _ => Kind::Gen,
        }
    }

    fn ksub(a: Kind, b: Kind) -> Kind {
        match (a, b) {
            (Kind::Uni, Kind::Uni) => Kind::Uni,
            (Kind::Lin, Kind::Uni) => Kind::Lin,
            _ => Kind::Gen,
        }
    }

    fn kuni(ks: &[Kind]) -> Kind {
        if ks.iter().all(|&k| k == Kind::Uni) {
            Kind::Uni
        } else {
            Kind::Gen
        }
    }

    /// Computes the per-packet builtin vectors for packet `self.pkt`.
    fn setup_packet(&mut self) {
        let [wx, wy, wz] = self.plan.workgroup_size;
        let base = self.b.ins().imul_imm(self.pkt, 4);
        let base = self.b.ins().splat(I32X4, base);
        let io = self.iota();
        let li = self.b.ins().iadd(base, io);
        let li_leaf = Leaf {
            s: ScalarTy::U32,
            a: li,
            b: None,
            k: Kind::Lin,
        };
        let zero = self.splat_i32(0);
        let z = Leaf {
            s: ScalarTy::U32,
            a: zero,
            b: None,
            k: Kind::Uni,
        };
        let lid = if wx == 1 && wy == 1 && wz == 1 {
            [z, z, z]
        } else if wy == 1 && wz == 1 {
            [li_leaf, z, z]
        } else if wx == 1 && wz == 1 {
            [z, li_leaf, z]
        } else if wx == 1 && wy == 1 {
            [z, z, li_leaf]
        } else {
            let lx = self.lanewise(I32X4, &[li], &|b, a| b.ins().urem_imm(a[0], i64::from(wx)));
            let ly = self.lanewise(I32X4, &[li], &|b, a| {
                let t = b.ins().udiv_imm(a[0], i64::from(wx));
                b.ins().urem_imm(t, i64::from(wy))
            });
            let lz = self.lanewise(I32X4, &[li], &|b, a| {
                let t = b.ins().udiv_imm(a[0], i64::from(wx));
                b.ins().udiv_imm(t, i64::from(wy))
            });
            let (kx, kyz) = if wx.is_multiple_of(4) {
                (Kind::Lin, Kind::Uni)
            } else {
                (Kind::Gen, Kind::Gen)
            };
            [
                Leaf {
                    s: ScalarTy::U32,
                    a: lx,
                    b: None,
                    k: kx,
                },
                Leaf {
                    s: ScalarTy::U32,
                    a: ly,
                    b: None,
                    k: kyz,
                },
                Leaf {
                    s: ScalarTy::U32,
                    a: lz,
                    b: None,
                    k: kyz,
                },
            ]
        };
        let mut wid = [z; 3];
        let mut nwg = [z; 3];
        let mut gid = [z; 3];
        for i in 0..3 {
            let w = self.b.ins().load(
                types::I32,
                mf_ro(),
                self.ctx,
                WorkgroupCtx::OFFSET_WORKGROUP_ID + 4 * i as i32,
            );
            let n = self.b.ins().load(
                types::I32,
                mf_ro(),
                self.ctx,
                WorkgroupCtx::OFFSET_NUM_WORKGROUPS + 4 * i as i32,
            );
            let ws = self.b.ins().splat(I32X4, w);
            let ns = self.b.ins().splat(I32X4, n);
            wid[i] = Leaf {
                s: ScalarTy::U32,
                a: ws,
                b: None,
                k: Kind::Uni,
            };
            nwg[i] = Leaf {
                s: ScalarTy::U32,
                a: ns,
                b: None,
                k: Kind::Uni,
            };
            let off = self
                .b
                .ins()
                .imul_imm(w, i64::from(self.plan.workgroup_size[i]));
            let off = self.b.ins().splat(I32X4, off);
            let g = self.b.ins().iadd(off, lid[i].a);
            gid[i] = Leaf {
                s: ScalarTy::U32,
                a: g,
                b: None,
                k: Self::kadd(Kind::Uni, lid[i].k),
            };
        }
        let sg_id = Leaf {
            s: ScalarTy::U32,
            a: io,
            b: None,
            k: Kind::Lin,
        };
        let four = self.splat_i32(4);
        let sg_size = Leaf {
            s: ScalarTy::U32,
            a: four,
            b: None,
            k: Kind::Uni,
        };
        self.bi = Some(Builtins {
            li: li_leaf,
            lid,
            gid,
            wid,
            nwg,
            sg_id,
            sg_size,
        });
    }

    fn builtin_val(&self, bv: BuiltinVar) -> Val {
        let bi = self.bi.as_ref().unwrap();
        let ty = bv.ty();
        let leaves = match bv {
            BuiltinVar::GlobalInvocationId => bi.gid.to_vec(),
            BuiltinVar::LocalInvocationId => bi.lid.to_vec(),
            BuiltinVar::LocalInvocationIndex => vec![bi.li],
            BuiltinVar::WorkgroupId => bi.wid.to_vec(),
            BuiltinVar::NumWorkgroups => bi.nwg.to_vec(),
            BuiltinVar::SubgroupInvocationId => vec![bi.sg_id],
            BuiltinVar::SubgroupSize => vec![bi.sg_size],
        };
        Val { ty, leaves }
    }

    /// Defines builtin locals and reloads cross-segment locals at the start of a packet.
    fn init_packet_locals(&mut self) {
        let module = self.module;
        for i in 0..module.entry_locals.len() {
            let l = LocalId(i as u32);
            self.frames[0].locals[i]
                .kinds
                .iter_mut()
                .for_each(|k| *k = Kind::Gen);
            if self.slot_off(l).is_some() {
                self.load_slot(l);
            }
        }
        for &(bv, l) in &module.entry_builtins {
            if self.plan.local_scratch_offsets[l.0 as usize].is_none() {
                let v = self.builtin_val(bv);
                let n = self.frames[0].locals[l.0 as usize].vars.len();
                if v.leaves.len() == n {
                    self.write_local_leaves(l, &v.leaves, true);
                }
            }
        }
    }

    fn initial_active(&mut self) -> Value {
        if self.plan.invocations.is_multiple_of(4) {
            self.splat_i32(-1)
        } else {
            let li = self.bi.as_ref().unwrap().li.a;
            self.icmp_c(IntCC::UnsignedLessThan, li, self.plan.invocations)
        }
    }

    fn active_slot(&mut self) -> (Value, i32) {
        let addr = self.scratch_addr(4);
        (addr, (self.plan.scratch_bytes - self.padded * 4) as i32)
    }

    fn packet_loop(&mut self, f: &mut dyn FnMut(&mut Self)) {
        let pv = self.b.declare_var(types::I32);
        let z = self.iconst(0);
        self.b.def_var(pv, z);
        let header = self.b.create_block();
        let body = self.b.create_block();
        let next = self.b.create_block();
        let exit = self.b.create_block();
        self.b.ins().jump(header, &[]);
        self.b.switch_to_block(header);
        let p = self.b.use_var(pv);
        let lim = self.iconst(self.packets as i32);
        let c = self.b.ins().icmp(IntCC::UnsignedLessThan, p, lim);
        self.b.ins().brif(c, body, &[], exit, &[]);
        self.b.switch_to_block(body);
        self.pkt = p;
        let saved = std::mem::replace(&mut self.pkt_next, next);
        f(self);
        self.b.ins().jump(next, &[]);
        self.b.switch_to_block(next);
        let np = self.b.ins().iadd_imm(p, 1);
        self.b.def_var(pv, np);
        self.b.ins().jump(header, &[]);
        self.b.switch_to_block(exit);
        self.pkt_next = saved;
    }

    /// One-time per-workgroup initialisation of mutable builtins, privates and the return mask.
    fn prologue(&mut self) {
        let module = self.module;
        let slots: Vec<(BuiltinVar, u32)> = module
            .entry_builtins
            .iter()
            .filter_map(|&(bv, l)| self.plan.local_scratch_offsets[l.0 as usize].map(|o| (bv, o)))
            .collect();
        if slots.is_empty() && module.privates.is_empty() && !self.has_ret_multi {
            return;
        }
        self.packet_loop(&mut |s| {
            s.setup_packet();
            let root = s.b.declare_var(I32X4);
            let ones = s.splat_i32(-1);
            s.b.def_var(root, ones);
            s.scopes.clear();
            s.scopes.push(Scope {
                var: root,
                full: s.plan.invocations.is_multiple_of(4),
                kind: ScopeKind::Root,
            });
            for &(bv, off) in &slots {
                let v = s.builtin_val(bv);
                s.scratch_store(off, &v.leaves);
            }
            for (i, p) in module.privates.iter().enumerate() {
                let v = match &p.init {
                    Some(e) => s.expr(e),
                    None => s.zero_val(&p.ty),
                };
                s.scratch_store(s.plan.priv_scratch_offsets[i], &v.leaves);
            }
            if s.has_ret_multi {
                let act = s.initial_active();
                let (addr, off) = s.active_slot();
                s.b.ins().store(mf(), act, addr, off);
            }
            s.scopes.clear();
        });
    }

    fn phases(&mut self, phases: &[PhaseStmt]) {
        for p in phases {
            match p {
                PhaseStmt::Segment { stmts, .. } => self.segment(stmts),
                PhaseStmt::If {
                    cond,
                    then_body,
                    else_body,
                } => {
                    let c = self.phase_cond(cond);
                    let tb = self.b.create_block();
                    let eb = self.b.create_block();
                    let merge = self.b.create_block();
                    self.b.ins().brif(c, tb, &[], eb, &[]);
                    self.b.switch_to_block(tb);
                    self.phases(then_body);
                    self.b.ins().jump(merge, &[]);
                    self.b.switch_to_block(eb);
                    self.phases(else_body);
                    self.b.ins().jump(merge, &[]);
                    self.b.switch_to_block(merge);
                }
                PhaseStmt::Loop {
                    body,
                    continuing,
                    break_if,
                } => {
                    let header = self.b.create_block();
                    let cont = self.b.create_block();
                    let exit = self.b.create_block();
                    self.b.ins().jump(header, &[]);
                    self.b.switch_to_block(header);
                    self.phase_loops.push(PhaseLoop { cont, exit });
                    self.phases(body);
                    self.phase_loops.pop();
                    self.b.ins().jump(cont, &[]);
                    self.b.switch_to_block(cont);
                    self.phases(continuing);
                    match break_if {
                        Some(c) => {
                            let c = self.phase_cond(c);
                            self.b.ins().brif(c, exit, &[], header, &[]);
                        }
                        None => {
                            self.b.ins().jump(header, &[]);
                        }
                    }
                    self.b.switch_to_block(exit);
                }
            }
        }
    }

    /// Evaluates a workgroup-uniform condition on packet 0, lane 0.
    fn phase_cond(&mut self, cond: &Expr) -> Value {
        self.pkt = self.iconst(0);
        self.setup_packet();
        self.init_packet_locals();
        let m = self.vconst(
            I32X4,
            [0xff, 0xff, 0xff, 0xff, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        );
        let root = self.b.declare_var(I32X4);
        self.b.def_var(root, m);
        self.scopes.clear();
        self.scopes.push(Scope {
            var: root,
            full: false,
            kind: ScopeKind::Root,
        });
        let v = self.expr(cond);
        self.scopes.clear();
        let lane0 = self.b.ins().extractlane(v.leaves[0].a, 0u8);
        self.b.ins().icmp_imm(IntCC::NotEqual, lane0, 0)
    }

    fn segment(&mut self, stmts: &[Stmt]) {
        let mut exits = [false; 3];
        root_exits(stmts, false, false, &mut exits);
        let full = self.plan.invocations.is_multiple_of(4)
            && !self.has_ret_multi
            && !exits.iter().any(|&e| e);
        let zero = self.iconst(0);
        self.b.def_var(self.broke_var, zero);
        self.b.def_var(self.cont_var, zero);
        let alive_var = self.b.declare_var(types::I32);
        self.b.def_var(alive_var, zero);
        let root = self.b.declare_var(I32X4);
        self.packet_loop(&mut |s| {
            s.setup_packet();
            s.init_packet_locals();
            let act = if s.has_ret_multi {
                let (addr, off) = s.active_slot();
                s.b.ins().load(I32X4, mf(), addr, off)
            } else {
                s.initial_active()
            };
            s.b.def_var(s.active_var, act);
            s.b.def_var(root, act);
            s.scopes.clear();
            s.scopes.push(Scope {
                var: root,
                full,
                kind: ScopeKind::Root,
            });
            if s.has_ret_multi {
                let any = s.b.ins().vany_true(act);
                let go = s.b.create_block();
                s.b.ins().brif(any, go, &[], s.pkt_next, &[]);
                s.b.switch_to_block(go);
            }
            s.stmts(stmts);
            if s.has_ret_multi {
                let act = s.b.use_var(s.active_var);
                let (addr, off) = s.active_slot();
                s.b.ins().store(mf(), act, addr, off);
                let any = s.b.ins().vany_true(act);
                let any32 = s.b.ins().uextend(types::I32, any);
                let cur = s.b.use_var(alive_var);
                let next = s.b.ins().bor(cur, any32);
                s.b.def_var(alive_var, next);
            }
            s.scopes.clear();
        });
        if self.has_ret_multi {
            let al = self.b.use_var(alive_var);
            let cont_wg = self.b.create_block();
            let exit_wg = self.b.create_block();
            self.b.ins().brif(al, cont_wg, &[], exit_wg, &[]);
            self.b.switch_to_block(exit_wg);
            self.b.ins().return_(&[]);
            self.b.switch_to_block(cont_wg);
        }
        if let Some(pl) = self.phase_loops.last() {
            let (cont, exit) = (pl.cont, pl.exit);
            if exits[1] {
                let f = self.b.use_var(self.broke_var);
                let next = self.b.create_block();
                self.b.ins().brif(f, exit, &[], next, &[]);
                self.b.switch_to_block(next);
            }
            if exits[2] {
                let f = self.b.use_var(self.cont_var);
                let next = self.b.create_block();
                self.b.ins().brif(f, cont, &[], next, &[]);
                self.b.switch_to_block(next);
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Scopes and statements
// ---------------------------------------------------------------------------------------------

enum RStep {
    Field(u32),
    Const(u32),
    Dyn(Value),
}

impl<'a, 'f> Lower<'a, 'f> {
    fn exec(&mut self) -> Value {
        let v = self.scopes.last().unwrap().var;
        self.b.use_var(v)
    }

    fn is_full(&self) -> bool {
        self.scopes.last().is_some_and(|s| s.full)
    }

    fn push_scope(&mut self, mask: Value, kind: ScopeKind) {
        let var = self.b.declare_var(I32X4);
        self.b.def_var(var, mask);
        self.scopes.push(Scope {
            var,
            full: false,
            kind,
        });
    }

    fn clear_var(&mut self, var: Variable, m: Value) {
        let cur = self.b.use_var(var);
        let n = self.b.ins().band_not(cur, m);
        self.b.def_var(var, n);
    }

    fn at_root(&self) -> bool {
        matches!(
            self.scopes.last().unwrap().kind,
            ScopeKind::Root | ScopeKind::CallRoot
        )
    }

    fn stmts(&mut self, list: &[Stmt]) {
        for s in list {
            if self.stmt(s) {
                break;
            }
        }
    }

    fn do_break(&mut self) {
        let m = self.exec();
        let mut i = self.scopes.len() - 1;
        loop {
            let sc = self.scopes[i];
            self.clear_var(sc.var, m);
            match sc.kind {
                ScopeKind::Loop(live) => {
                    self.clear_var(live, m);
                    break;
                }
                ScopeKind::Switch => break,
                ScopeKind::Root | ScopeKind::CallRoot => {
                    if matches!(sc.kind, ScopeKind::Root) && !self.phase_loops.is_empty() {
                        let any = self.b.ins().vany_true(m);
                        let any = self.b.ins().uextend(types::I32, any);
                        let cur = self.b.use_var(self.broke_var);
                        let n = self.b.ins().bor(cur, any);
                        self.b.def_var(self.broke_var, n);
                    }
                    break;
                }
                ScopeKind::Plain => {}
            }
            i -= 1;
        }
    }

    fn do_continue(&mut self) {
        let m = self.exec();
        let mut i = self.scopes.len() - 1;
        loop {
            let sc = self.scopes[i];
            self.clear_var(sc.var, m);
            match sc.kind {
                ScopeKind::Loop(_) => break,
                ScopeKind::Root | ScopeKind::CallRoot => {
                    if matches!(sc.kind, ScopeKind::Root) && !self.phase_loops.is_empty() {
                        let any = self.b.ins().vany_true(m);
                        let any = self.b.ins().uextend(types::I32, any);
                        let cur = self.b.use_var(self.cont_var);
                        let n = self.b.ins().bor(cur, any);
                        self.b.def_var(self.cont_var, n);
                    }
                    break;
                }
                _ => {}
            }
            i -= 1;
        }
    }

    fn do_return(&mut self) {
        let m = self.exec();
        let mut i = self.scopes.len() - 1;
        loop {
            let sc = self.scopes[i];
            self.clear_var(sc.var, m);
            match sc.kind {
                ScopeKind::Loop(live) => self.clear_var(live, m),
                ScopeKind::Root => {
                    let a = self.active_var;
                    self.clear_var(a, m);
                    break;
                }
                ScopeKind::CallRoot => break,
                _ => {}
            }
            i -= 1;
        }
    }

    fn any_active(&mut self, mask: Value) -> Value {
        self.b.ins().vany_true(mask)
    }

    /// Lowers one statement. Returns `true` when the rest of the enclosing block is dead.
    fn stmt(&mut self, s: &Stmt) -> bool {
        match s {
            Stmt::Let { local, init } => {
                let v = self.expr(init);
                self.write_local_leaves(*local, &v.leaves, true);
            }
            Stmt::Var { local, init } => {
                let v = match init {
                    Some(e) => self.expr(e),
                    None => {
                        let ty = self.frames.last().unwrap().locals[local.0 as usize]
                            .ty
                            .clone();
                        self.zero_val(&ty)
                    }
                };
                self.write_local_leaves(*local, &v.leaves, false);
            }
            Stmt::Assign { dst, val } => {
                let v = self.expr(val);
                self.assign(dst, &v);
            }
            Stmt::Store {
                binding,
                index,
                val,
            } => {
                let idx = self.expr(index).leaves[0];
                let v = self.expr(val);
                if let Some(base) = self.binding_ptr(*binding) {
                    let stride = self.storage_stride(*binding);
                    self.store_elem(base, 0, idx, stride, &v);
                }
            }
            Stmt::SharedStore { shared, index, val } => {
                let idx = self.expr(index).leaves[0];
                let v = self.expr(val);
                let base = self.shared_ptr(*shared);
                let stride = self.module.shared[*shared as usize]
                    .elem_ty
                    .layout(self.structs())
                    .0;
                self.store_elem(base, 0, idx, stride, &v);
            }
            Stmt::If {
                cond,
                then_body,
                else_body,
            } => {
                let c = self.expr(cond).leaves[0].a;
                let exec = self.exec();
                let then_m = self.b.ins().band(exec, c);
                let else_m = self.b.ins().band_not(exec, c);
                self.guarded_scope(then_m, then_body);
                if !else_body.is_empty() {
                    self.guarded_scope(else_m, else_body);
                }
            }
            Stmt::Loop {
                body,
                continuing,
                break_if,
            } => self.lower_loop(body, continuing, break_if.as_ref()),
            Stmt::Switch {
                selector,
                cases,
                default,
            } => self.lower_switch(selector, cases, default),
            Stmt::Break => {
                self.do_break();
                return self.at_root();
            }
            Stmt::Continue => {
                self.do_continue();
                return self.at_root();
            }
            Stmt::Return(e) => {
                if let Some(e) = e {
                    let v = self.expr(e);
                    let exec = self.exec();
                    if let Some((_, _, vars)) = self.frames.last().unwrap().ret.clone() {
                        for (i, leaf) in v.leaves.iter().enumerate() {
                            let old = self.b.use_var(vars[i].0);
                            let oldb = vars[i].1.map(|h| self.b.use_var(h));
                            let cur = Leaf {
                                s: leaf.s,
                                a: old,
                                b: oldb,
                                k: Kind::Gen,
                            };
                            let n = self.select_leaf(exec, *leaf, cur);
                            self.b.def_var(vars[i].0, n.a);
                            if let Some(h) = vars[i].1 {
                                self.b.def_var(h, n.b.unwrap());
                            }
                        }
                    }
                }
                self.do_return();
                return self.at_root();
            }
            Stmt::Barrier(_) => {}
            Stmt::AtomicStore { target, val } => {
                let v = self.expr(val).leaves[0];
                let bits = if v.s == ScalarTy::F32 {
                    self.b.ins().bitcast(I32X4, mf_bc(), v.a)
                } else {
                    v.a
                };
                self.atomic_lanes(target, &mut |s, addr, lane| {
                    let x = s.b.ins().extractlane(bits, lane);
                    s.b.ins().atomic_store(MemFlagsData::trusted(), x, addr);
                });
            }
            Stmt::AtomicRmw {
                dst,
                op,
                elem_ty,
                target,
                val,
            } => {
                let v = self.expr(val).leaves[0];
                let acc = self.b.declare_var(I32X4);
                let z = self.splat_i32(0);
                self.b.def_var(acc, z);
                if *elem_ty == ScalarTy::F32
                    && matches!(
                        op,
                        IrAtomicOp::Add | IrAtomicOp::Sub | IrAtomicOp::Min | IrAtomicOp::Max
                    )
                {
                    let fop = *op;
                    self.atomic_lanes(target, &mut |s, addr, lane| {
                        let xf = s.b.ins().extractlane(v.a, lane);
                        let init =
                            s.b.ins()
                                .atomic_load(types::I32, MemFlagsData::trusted(), addr);
                        let exp_var = s.b.declare_var(types::I32);
                        s.b.def_var(exp_var, init);
                        let hdr = s.b.create_block();
                        let done = s.b.create_block();
                        s.b.ins().jump(hdr, &[]);
                        s.b.switch_to_block(hdr);
                        let exp_bits = s.b.use_var(exp_var);
                        let exp_f = s.b.ins().bitcast(types::F32, mf_bc(), exp_bits);
                        let next_f = match fop {
                            IrAtomicOp::Add => s.b.ins().fadd(exp_f, xf),
                            IrAtomicOp::Sub => s.b.ins().fsub(exp_f, xf),
                            IrAtomicOp::Min => {
                                let lt = s.b.ins().fcmp(FloatCC::LessThan, xf, exp_f);
                                s.b.ins().select(lt, xf, exp_f)
                            }
                            IrAtomicOp::Max => {
                                let gt = s.b.ins().fcmp(FloatCC::GreaterThan, xf, exp_f);
                                s.b.ins().select(gt, xf, exp_f)
                            }
                            _ => unreachable!(),
                        };
                        let next_bits = s.b.ins().bitcast(types::I32, mf_bc(), next_f);
                        let prev_bits = s.b.ins().atomic_cas(
                            MemFlagsData::trusted(),
                            addr,
                            exp_bits,
                            next_bits,
                        );
                        s.b.def_var(exp_var, prev_bits);
                        let eq = s.b.ins().icmp(IntCC::Equal, prev_bits, exp_bits);
                        s.b.ins().brif(eq, done, &[], hdr, &[]);
                        s.b.switch_to_block(done);
                        let old = s.b.use_var(exp_var);
                        let cur = s.b.use_var(acc);
                        let n = s.b.ins().insertlane(cur, old, lane);
                        s.b.def_var(acc, n);
                    });
                } else {
                    let bits = if *elem_ty == ScalarTy::F32 {
                        self.b.ins().bitcast(I32X4, mf_bc(), v.a)
                    } else {
                        v.a
                    };
                    let rop = match op {
                        IrAtomicOp::Add => AtomicRmwOp::Add,
                        IrAtomicOp::Sub => AtomicRmwOp::Sub,
                        IrAtomicOp::Min => {
                            if *elem_ty == ScalarTy::I32 {
                                AtomicRmwOp::Smin
                            } else {
                                AtomicRmwOp::Umin
                            }
                        }
                        IrAtomicOp::Max => {
                            if *elem_ty == ScalarTy::I32 {
                                AtomicRmwOp::Smax
                            } else {
                                AtomicRmwOp::Umax
                            }
                        }
                        IrAtomicOp::And => AtomicRmwOp::And,
                        IrAtomicOp::Or => AtomicRmwOp::Or,
                        IrAtomicOp::Xor => AtomicRmwOp::Xor,
                        IrAtomicOp::Exchange => AtomicRmwOp::Xchg,
                    };
                    self.atomic_lanes(target, &mut |s, addr, lane| {
                        let x = s.b.ins().extractlane(bits, lane);
                        let old =
                            s.b.ins()
                                .atomic_rmw(types::I32, MemFlagsData::trusted(), rop, addr, x);
                        let cur = s.b.use_var(acc);
                        let n = s.b.ins().insertlane(cur, old, lane);
                        s.b.def_var(acc, n);
                    });
                }
                if let Some(d) = dst {
                    let r = self.b.use_var(acc);
                    let r = if *elem_ty == ScalarTy::F32 {
                        self.b.ins().bitcast(F32X4, mf_bc(), r)
                    } else {
                        r
                    };
                    let leaf = Leaf {
                        s: *elem_ty,
                        a: r,
                        b: None,
                        k: Kind::Gen,
                    };
                    self.write_local_leaves(*d, &[leaf], false);
                }
            }
            Stmt::AtomicCompareExchange {
                dst_old,
                dst_exchanged,
                elem_ty,
                target,
                cmp,
                val,
            } => {
                let c = self.expr(cmp).leaves[0];
                let v = self.expr(val).leaves[0];
                let c_bits = if *elem_ty == ScalarTy::F32 {
                    self.b.ins().bitcast(I32X4, mf_bc(), c.a)
                } else {
                    c.a
                };
                let v_bits = if *elem_ty == ScalarTy::F32 {
                    self.b.ins().bitcast(I32X4, mf_bc(), v.a)
                } else {
                    v.a
                };
                let acc = self.b.declare_var(I32X4);
                let z = self.splat_i32(0);
                self.b.def_var(acc, z);
                self.atomic_lanes(target, &mut |s, addr, lane| {
                    let e = s.b.ins().extractlane(c_bits, lane);
                    let x = s.b.ins().extractlane(v_bits, lane);
                    let old = s.b.ins().atomic_cas(MemFlagsData::trusted(), addr, e, x);
                    let cur = s.b.use_var(acc);
                    let n = s.b.ins().insertlane(cur, old, lane);
                    s.b.def_var(acc, n);
                });
                let old_bits = self.b.use_var(acc);
                let ok = self.b.ins().icmp(IntCC::Equal, old_bits, c_bits);
                let exec = self.exec();
                let ok = self.b.ins().band(ok, exec);
                let old = if *elem_ty == ScalarTy::F32 {
                    self.b.ins().bitcast(F32X4, mf_bc(), old_bits)
                } else {
                    old_bits
                };
                self.write_local_leaves(
                    *dst_old,
                    &[Leaf {
                        s: *elem_ty,
                        a: old,
                        b: None,
                        k: Kind::Gen,
                    }],
                    false,
                );
                self.write_local_leaves(
                    *dst_exchanged,
                    &[Leaf {
                        s: ScalarTy::Bool,
                        a: ok,
                        b: None,
                        k: Kind::Gen,
                    }],
                    false,
                );
            }
            Stmt::Discard(e) => {
                let _ = self.expr(e);
            }
        }
        false
    }

    /// Runs `body` under `mask` only when some lane is active.
    fn guarded_scope(&mut self, mask: Value, body: &[Stmt]) {
        let any = self.any_active(mask);
        let tb = self.b.create_block();
        let merge = self.b.create_block();
        self.b.ins().brif(any, tb, &[], merge, &[]);
        self.b.switch_to_block(tb);
        self.push_scope(mask, ScopeKind::Plain);
        self.stmts(body);
        self.scopes.pop();
        self.b.ins().jump(merge, &[]);
        self.b.switch_to_block(merge);
    }

    fn lower_loop(&mut self, body: &[Stmt], continuing: &[Stmt], break_if: Option<&Expr>) {
        let outer = self.exec();
        let live = self.b.declare_var(I32X4);
        self.b.def_var(live, outer);
        let header = self.b.create_block();
        let exit = self.b.create_block();
        self.b.ins().jump(header, &[]);
        self.b.switch_to_block(header);
        let lm = self.b.use_var(live);
        self.push_scope(lm, ScopeKind::Loop(live));
        self.stmts(body);
        self.scopes.pop();
        let lm = self.b.use_var(live);
        self.push_scope(lm, ScopeKind::Plain);
        self.stmts(continuing);
        if let Some(c) = break_if {
            let cm = self.expr(c).leaves[0].a;
            let lm = self.b.use_var(live);
            let live_now = self.b.ins().band_not(lm, cm);
            self.b.def_var(live, live_now);
        }
        self.scopes.pop();
        let lm = self.b.use_var(live);
        let any = self.any_active(lm);
        self.b.ins().brif(any, header, &[], exit, &[]);
        self.b.switch_to_block(exit);
    }

    fn lower_switch(
        &mut self,
        selector: &Expr,
        cases: &[(SwitchCase, Vec<Stmt>, bool)],
        default: &[Stmt],
    ) {
        let sel = self.expr(selector).leaves[0].a;
        let exec = self.exec();
        let rem = self.b.declare_var(I32X4);
        self.b.def_var(rem, exec);
        let fall = self.b.declare_var(I32X4);
        let z = self.zero_mask();
        self.b.def_var(fall, z);
        for (case, body, falls) in cases {
            let k = match case {
                SwitchCase::I32(v) => *v,
                SwitchCase::U32(v) => *v as i32,
            };
            let kv = self.splat_i32(k);
            let eq = self.b.ins().icmp(IntCC::Equal, sel, kv);
            let r = self.b.use_var(rem);
            let hit = self.b.ins().band(r, eq);
            let f = self.b.use_var(fall);
            let m = self.b.ins().bor(hit, f);
            let r2 = self.b.ins().band_not(r, eq);
            self.b.def_var(rem, r2);
            self.switch_case(m, body, fall, *falls);
        }
        let r = self.b.use_var(rem);
        let f = self.b.use_var(fall);
        let m = self.b.ins().bor(r, f);
        let dummy = self.b.declare_var(I32X4);
        self.switch_case(m, default, dummy, false);
    }

    fn switch_case(&mut self, mask: Value, body: &[Stmt], fall: Variable, falls: bool) {
        let z = self.zero_mask();
        self.b.def_var(fall, z);
        let any = self.any_active(mask);
        let tb = self.b.create_block();
        let merge = self.b.create_block();
        self.b.ins().brif(any, tb, &[], merge, &[]);
        self.b.switch_to_block(tb);
        self.push_scope(mask, ScopeKind::Switch);
        self.stmts(body);
        let sc = self.scopes.pop().unwrap();
        if falls {
            let left = self.b.use_var(sc.var);
            self.b.def_var(fall, left);
        }
        self.b.ins().jump(merge, &[]);
        self.b.switch_to_block(merge);
    }

    // -----------------------------------------------------------------------------------------
    // Assignment through lvalues
    // -----------------------------------------------------------------------------------------

    fn flatten<'e>(&self, lv: &'e LValue, out: &mut Vec<Step<'e>>) -> Root {
        match lv {
            LValue::Local(l) => Root::Local(*l),
            LValue::Private(p) => Root::Private(*p),
            LValue::Storage { binding, path } => {
                for s in path {
                    out.push(match s {
                        AccessStep::Field(f) => Step::Field(*f),
                        AccessStep::Index(e) => Step::Index(e),
                    });
                }
                Root::Storage(*binding)
            }
            LValue::Shared { shared, path } => {
                for s in path {
                    out.push(match s {
                        AccessStep::Field(f) => Step::Field(*f),
                        AccessStep::Index(e) => Step::Index(e),
                    });
                }
                Root::Shared(*shared)
            }
            LValue::Index { base, index } => {
                let r = self.flatten(base, out);
                out.push(Step::Index(index));
                r
            }
            LValue::Field { base, field } => {
                let r = self.flatten(base, out);
                out.push(Step::Field(*field));
                r
            }
        }
    }

    fn resolve_steps(&mut self, steps: &[Step]) -> Vec<RStep> {
        steps
            .iter()
            .map(|s| match s {
                Step::Field(f) => RStep::Field(*f),
                Step::Index(e) => match lit_index(e) {
                    Some(k) => RStep::Const(k),
                    None => RStep::Dyn(self.expr(e).leaves[0].a),
                },
            })
            .collect()
    }

    fn assign(&mut self, dst: &LValue, v: &Val) {
        let mut raw = Vec::new();
        let root = self.flatten(dst, &mut raw);
        let steps = self.resolve_steps(&raw);
        match root {
            Root::Local(l) => {
                let cur = self.read_local(l);
                let cond = if self.is_full() {
                    None
                } else {
                    Some(self.exec())
                };
                let mut leaves = cur.leaves.clone();
                self.write_path(&mut leaves, &cur.ty, 0, &steps, v, cond);
                self.write_local_leaves(l, &leaves, false);
            }
            Root::Private(p) => {
                let ty = self.module.privates[p as usize].ty.clone();
                let scs = scalars_of(&ty, self.structs());
                let off = self.plan.priv_scratch_offsets[p as usize];
                let mut leaves = self.scratch_load(off, &scs);
                let cond = if self.is_full() {
                    None
                } else {
                    Some(self.exec())
                };
                self.write_path(&mut leaves, &ty, 0, &steps, v, cond);
                self.scratch_store(off, &leaves);
            }
            Root::Storage(b) => {
                let start = self.storage_start_ty(b);
                let (off, dyns, _) = self.mem_path(&start, &steps);
                if let Some(base) = self.binding_ptr(b) {
                    self.mem_write(base, off, &dyns, v);
                }
            }
            Root::Shared(s) => {
                let start = self.shared_start_ty(s);
                let (off, dyns, _) = self.mem_path(&start, &steps);
                let base = self.shared_ptr(s);
                self.mem_write(base, off, &dyns, v);
            }
        }
    }

    fn write_path(
        &mut self,
        leaves: &mut Vec<Leaf>,
        ty: &KirType,
        start: usize,
        steps: &[RStep],
        val: &Val,
        cond: Option<Value>,
    ) {
        let structs = self.structs();
        let Some((step, rest)) = steps.split_first() else {
            let n = leaf_count(ty, structs).min(val.leaves.len());
            for i in 0..n {
                let new = val.leaves[i];
                leaves[start + i] = match cond {
                    Some(m) => self.select_leaf(m, new, leaves[start + i]),
                    None => Leaf {
                        k: Kind::Gen,
                        ..new
                    },
                };
            }
            return;
        };
        match (ty, step) {
            (KirType::Struct(id), RStep::Field(i) | RStep::Const(i)) => {
                let decl = &structs[*id as usize];
                let mut off = start;
                for f in &decl.fields[..*i as usize] {
                    off += leaf_count(&f.ty, structs);
                }
                let sub = decl.fields[*i as usize].ty.clone();
                self.write_path(leaves, &sub, off, rest, val, cond);
            }
            (KirType::Vec { elem, lanes }, RStep::Field(i) | RStep::Const(i)) => {
                let i = (*i).min(u32::from(*lanes) - 1) as usize;
                self.write_path(leaves, &KirType::Scalar(*elem), start + i, rest, val, cond);
            }
            (KirType::Array { elem, len, .. }, RStep::Field(i) | RStep::Const(i)) => {
                let i = (*i).min(len.saturating_sub(1)) as usize;
                let n = leaf_count(elem, structs);
                self.write_path(leaves, elem, start + i * n, rest, val, cond);
            }
            (KirType::Vec { .. } | KirType::Array { .. }, RStep::Dyn(idx)) => {
                let (elem, len, n) = match ty {
                    KirType::Vec { elem, lanes } => (KirType::Scalar(*elem), u32::from(*lanes), 1),
                    KirType::Array { elem, len, .. } => {
                        ((**elem).clone(), *len, leaf_count(elem, structs))
                    }
                    _ => unreachable!(),
                };
                for k in 0..len {
                    let cc = if k + 1 == len {
                        IntCC::UnsignedGreaterThanOrEqual
                    } else {
                        IntCC::Equal
                    };
                    let mk = self.icmp_c(cc, *idx, k);
                    let mk = match cond {
                        Some(c) => self.b.ins().band(mk, c),
                        None => mk,
                    };
                    self.write_path(leaves, &elem, start + k as usize * n, rest, val, Some(mk));
                }
            }
            _ => {}
        }
    }

    /// Reads element `step` out of an aggregate value.
    fn access(&mut self, base: &Val, step: &RStep) -> Val {
        let structs = self.structs();
        match (&base.ty, step) {
            (KirType::Struct(id), RStep::Field(i) | RStep::Const(i)) => {
                let decl = &structs[*id as usize];
                let mut off = 0;
                for f in &decl.fields[..*i as usize] {
                    off += leaf_count(&f.ty, structs);
                }
                let sub = decl.fields[*i as usize].ty.clone();
                let n = leaf_count(&sub, structs);
                Val {
                    ty: sub,
                    leaves: base.leaves[off..off + n].to_vec(),
                }
            }
            (KirType::Vec { elem, lanes }, RStep::Field(i) | RStep::Const(i)) => {
                let i = (*i).min(u32::from(*lanes) - 1) as usize;
                Val {
                    ty: KirType::Scalar(*elem),
                    leaves: vec![base.leaves[i]],
                }
            }
            (KirType::Array { elem, len, .. }, RStep::Field(i) | RStep::Const(i)) => {
                let i = (*i).min(len.saturating_sub(1)) as usize;
                let n = leaf_count(elem, structs);
                Val {
                    ty: (**elem).clone(),
                    leaves: base.leaves[i * n..(i + 1) * n].to_vec(),
                }
            }
            (KirType::Vec { .. } | KirType::Array { .. }, RStep::Dyn(idx)) => {
                let (elem, len, n) = match &base.ty {
                    KirType::Vec { elem, lanes } => (KirType::Scalar(*elem), u32::from(*lanes), 1),
                    KirType::Array { elem, len, .. } => {
                        ((**elem).clone(), *len, leaf_count(elem, structs))
                    }
                    _ => unreachable!(),
                };
                let last = (len as usize - 1) * n;
                let mut out: Vec<Leaf> = base.leaves[last..last + n].to_vec();
                for k in (0..len.saturating_sub(1)).rev() {
                    let m = self.icmp_c(IntCC::Equal, *idx, k);
                    for j in 0..n {
                        let cand = base.leaves[k as usize * n + j];
                        out[j] = self.select_leaf(m, cand, out[j]);
                    }
                }
                Val {
                    ty: elem,
                    leaves: out,
                }
            }
            _ => base.clone(),
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Memory
// ---------------------------------------------------------------------------------------------

/// One leaf of a gathered value. Its accumulator vector (and, for `f64`, the vector of lanes 2
/// and 3) is carried through the lane blocks as block parameters, not as `Variable`s.
struct Acc {
    s: ScalarTy,
    off: u32,
}

impl<'a, 'f> Lower<'a, 'f> {
    fn binding_ptr(&mut self, b: u32) -> Option<Value> {
        let rank = self.plan.binding_ranks.get(b as usize).copied().flatten()?;
        let arr = self
            .b
            .ins()
            .load(self.ptr, mf_ro(), self.ctx, WorkgroupCtx::OFFSET_BINDINGS);
        let base = self.b.ins().load(self.ptr, mf_ro(), arr, (rank * 8) as i32);
        let sizes = self.b.ins().load(
            self.ptr,
            mf_ro(),
            self.ctx,
            WorkgroupCtx::OFFSET_BINDING_BYTES,
        );
        let bytes = self
            .b
            .ins()
            .load(types::I64, mf_ro(), sizes, (rank * 8) as i32);
        self.limits.insert(base, bytes);
        Some(base)
    }

    fn shared_ptr(&mut self, s: u32) -> Value {
        let base = self
            .b
            .ins()
            .load(self.ptr, mf_ro(), self.ctx, WorkgroupCtx::OFFSET_SHARED);
        let off = self.plan.shared_offsets[s as usize];
        if off == 0 {
            base
        } else {
            self.b.ins().iadd_imm(base, i64::from(off))
        }
    }

    fn storage_stride(&self, b: u32) -> u32 {
        let d = &self.module.bindings[b as usize];
        d.element_stride
            .unwrap_or_else(|| d.elem_ty.layout(self.structs()).0)
            .max(1)
    }

    fn storage_start_ty(&self, b: u32) -> KirType {
        let d = &self.module.bindings[b as usize];
        if d.is_unsized_array {
            KirType::Array {
                elem: Box::new(d.elem_ty.clone()),
                len: u32::MAX,
                stride: self.storage_stride(b),
            }
        } else {
            d.elem_ty.clone()
        }
    }

    fn shared_start_ty(&self, s: u32) -> KirType {
        let d = &self.module.shared[s as usize];
        match d.len {
            Some(n) => KirType::Array {
                elem: Box::new(d.elem_ty.clone()),
                len: n,
                stride: d.elem_ty.layout(self.structs()).0,
            },
            None => d.elem_ty.clone(),
        }
    }

    fn resolve_access(&mut self, path: &[AccessStep]) -> Vec<RStep> {
        path.iter()
            .map(|s| match s {
                AccessStep::Field(f) => RStep::Field(*f),
                AccessStep::Index(e) => match lit_index(e) {
                    Some(k) => RStep::Const(k),
                    None => RStep::Dyn(self.expr(e).leaves[0].a),
                },
            })
            .collect()
    }

    /// Walks `steps` through `start`, returning the constant byte offset, dynamic `(index, stride)` terms and result type.
    fn mem_path(&mut self, start: &KirType, steps: &[RStep]) -> (u32, Vec<(Value, u32)>, KirType) {
        let structs = self.structs();
        let mut off = 0u32;
        let mut dyns = Vec::new();
        let mut ty = start.clone();
        for st in steps {
            ty = match (&ty, st) {
                (KirType::Struct(id), RStep::Field(i) | RStep::Const(i)) => {
                    let f = &structs[*id as usize].fields[*i as usize];
                    off += f.offset;
                    f.ty.clone()
                }
                (KirType::Vec { elem, .. }, RStep::Field(i) | RStep::Const(i)) => {
                    off += i * elem.size_bytes();
                    KirType::Scalar(*elem)
                }
                (KirType::Array { elem, stride, .. }, RStep::Field(i) | RStep::Const(i)) => {
                    off += i * stride;
                    (**elem).clone()
                }
                (KirType::Vec { elem, .. }, RStep::Dyn(v)) => {
                    dyns.push((*v, elem.size_bytes()));
                    KirType::Scalar(*elem)
                }
                (KirType::Array { elem, stride, .. }, RStep::Dyn(v)) => {
                    dyns.push((*v, *stride));
                    (**elem).clone()
                }
                _ => ty,
            };
        }
        (off, dyns, ty)
    }

    fn lane_addr(&mut self, base: Value, off: u32, dyns: &[(Value, u32)], lane: u8) -> Value {
        let mut a = base;
        if off != 0 {
            a = self.b.ins().iadd_imm(a, i64::from(off));
        }
        for &(iv, stride) in dyns {
            let e = self.b.ins().extractlane(iv, lane);
            let e = self.b.ins().uextend(self.ptr, e);
            let e = if stride == 1 {
                e
            } else {
                self.b.ins().imul_imm(e, i64::from(stride))
            };
            a = self.b.ins().iadd(a, e);
        }
        a
    }

    /// Runs `f` only when `cond` (non-zero) holds, always when it is `None`. `carried` values go
    /// through: `f` gets them and returns their new values, the result being those or `carried`
    /// itself when `f` was skipped. One conditional branch, the values as block parameters.
    fn guarded(
        &mut self,
        cond: Option<Value>,
        carried: &[Value],
        f: &mut dyn FnMut(&mut Self, &[Value]) -> Vec<Value>,
    ) -> Vec<Value> {
        let Some(cond) = cond else {
            return f(self, carried);
        };
        let doit = self.b.create_block();
        let merge = self.b.create_block();
        for &v in carried {
            let ty = self.b.func.dfg.value_type(v);
            self.b.append_block_param(merge, ty);
        }
        let skip: Vec<BlockArg> = carried.iter().map(|&v| v.into()).collect();
        self.b.ins().brif(cond, doit, &[], merge, &skip);
        self.b.switch_to_block(doit);
        let done: Vec<BlockArg> = f(self, carried).into_iter().map(Into::into).collect();
        self.b.ins().jump(merge, &done);
        self.b.switch_to_block(merge);
        self.b.block_params(merge).to_vec()
    }

    /// Runs `f(addr, carried)` for `lane` only if it is active (always when the scope is
    /// statically full) and its `extent`-byte access at `base + off + dyns` falls inside the
    /// bound storage binding (speculative loads hoisted above a `Select`). The lane's result is
    /// what `f` returns, or `carried` when it was skipped.
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    fn lane_guard_mem(
        &mut self,
        exec: Value,
        full: bool,
        lane: u8,
        base: Value,
        off: u32,
        dyns: &[(Value, u32)],
        extent: u32,
        carried: &[Value],
        f: &mut dyn FnMut(&mut Self, Value, &[Value]) -> Vec<Value>,
    ) -> Vec<Value> {
        let mut cond = (!full).then(|| self.b.ins().extractlane(exec, lane));
        let addr = self.lane_addr(base, off, dyns, lane);
        if let Some(lim) = self.limits.get(&base).copied() {
            let rel = self.b.ins().isub(addr, base);
            let end = self.b.ins().iadd_imm(rel, i64::from(extent));
            let ok = self.b.ins().icmp(IntCC::UnsignedLessThanOrEqual, end, lim);
            let ok = self.b.ins().uextend(types::I32, ok);
            cond = Some(match cond {
                Some(m) => self.b.ins().band(m, ok),
                None => ok,
            });
        }
        self.guarded(cond, carried, &mut |sl, c| f(sl, addr, c))
    }

    fn load_leaf(&mut self, s: ScalarTy, addr: Value, at: u32) -> Value {
        let at = at as i32;
        match s {
            ScalarTy::F64 => self.b.ins().load(types::F64, mf(), addr, at),
            ScalarTy::F32 => self.b.ins().load(types::F32, mf(), addr, at),
            ScalarTy::F16 => self.b.ins().uload16(types::I32, mf(), addr, at),
            _ => self.b.ins().load(types::I32, mf(), addr, at),
        }
    }

    /// Reads the leaves at `base + off` when there is no per-lane index: every lane reads the
    /// same address, so it is one bounds check and one load per leaf, splat across the packet
    /// (inactive lanes included: WGSL leaves what a shuffle reads from them indeterminate).
    /// Out of bounds reads zero.
    fn uniform_gather(
        &mut self,
        base: Value,
        off: u32,
        extent: u32,
        accs: &[Acc],
        cur: &[Value],
    ) -> Vec<Value> {
        let cond = self.limits.get(&base).copied().map(|lim| {
            let end = self.b.ins().iconst(self.ptr, i64::from(off) + i64::from(extent));
            self.b.ins().icmp(IntCC::UnsignedLessThanOrEqual, end, lim)
        });
        self.guarded(cond, cur, &mut |sl, _| {
            let mut out = Vec::with_capacity(cur.len());
            // The offset is added in 64 bits: the load's own displacement is 32-bit.
            let addr = if off == 0 { base } else { sl.b.ins().iadd_imm(base, i64::from(off)) };
            for acc in accs {
                let x = sl.load_leaf(acc.s, addr, acc.off);
                match acc.s {
                    ScalarTy::F64 => {
                        let v = sl.b.ins().splat(F64X2, x);
                        out.extend([v, v]);
                    }
                    ScalarTy::F32 => out.push(sl.b.ins().splat(F32X4, x)),
                    _ => out.push(sl.b.ins().splat(I32X4, x)),
                }
            }
            out
        })
    }

    /// Reads the leaves at `base + off + dyns` lane by lane; lanes not read stay zero.
    fn lane_gather(
        &mut self,
        base: Value,
        off: u32,
        dyns: &[(Value, u32)],
        extent: u32,
        accs: &[Acc],
        mut cur: Vec<Value>,
    ) -> Vec<Value> {
        let exec = self.exec();
        let full = self.is_full();
        for lane in 0..4u8 {
            cur = self.lane_guard_mem(
                exec,
                full,
                lane,
                base,
                off,
                dyns,
                extent,
                &cur,
                &mut |sl, addr, cur| {
                    let mut out = cur.to_vec();
                    let mut k = 0;
                    for acc in accs {
                        let x = sl.load_leaf(acc.s, addr, acc.off);
                        if acc.s == ScalarTy::F64 {
                            let at = k + usize::from(lane >= 2);
                            out[at] = sl.b.ins().insertlane(cur[at], x, lane % 2);
                            k += 2;
                        } else {
                            out[k] = sl.b.ins().insertlane(cur[k], x, lane);
                            k += 1;
                        }
                    }
                    out
                },
            );
        }
        cur
    }

    fn mem_read(&mut self, base: Value, off: u32, dyns: &[(Value, u32)], ty: &KirType) -> Val {
        let mut lay = Vec::new();
        layout_leaves(ty, self.structs(), 0, &mut lay);
        let mut accs = Vec::with_capacity(lay.len());
        // `cur` holds the accumulator vectors, an `f64` leaf taking two.
        let mut cur: Vec<Value> = Vec::with_capacity(lay.len());
        for &(o, s) in &lay {
            let z = match s {
                ScalarTy::F32 => self.splat_f32(0.0),
                ScalarTy::F64 => self.splat_f64(0.0),
                _ => self.splat_i32(0),
            };
            cur.push(z);
            if s == ScalarTy::F64 {
                cur.push(z);
            }
            accs.push(Acc { s, off: o });
        }
        let extent = lay
            .iter()
            .map(|&(o, s)| {
                o + if s == ScalarTy::F64 {
                    8
                } else if s == ScalarTy::F16 {
                    2
                } else {
                    4
                }
            })
            .max()
            .unwrap_or(0);
        let cur = if dyns.is_empty() {
            self.uniform_gather(base, off, extent, &accs, &cur)
        } else {
            self.lane_gather(base, off, dyns, extent, &accs, cur)
        };
        let mut leaves = Vec::with_capacity(accs.len());
        let mut k = 0;
        for acc in &accs {
            let a = cur[k];
            let b = (acc.s == ScalarTy::F64).then(|| cur[k + 1]);
            k += if acc.s == ScalarTy::F64 { 2 } else { 1 };
            let leaf = match acc.s {
                ScalarTy::Bool => {
                    let z = self.splat_i32(0);
                    let m = self.b.ins().icmp(IntCC::NotEqual, a, z);
                    Leaf {
                        s: acc.s,
                        a: m,
                        b: None,
                        k: Kind::Gen,
                    }
                }
                ScalarTy::F16 => {
                    let f = self.f16bits_to_f32(a);
                    Leaf {
                        s: acc.s,
                        a: f,
                        b: None,
                        k: Kind::Gen,
                    }
                }
                s => Leaf {
                    s,
                    a,
                    b,
                    k: Kind::Gen,
                },
            };
            leaves.push(leaf);
        }
        Val {
            ty: ty.clone(),
            leaves,
        }
    }

    fn mem_write(&mut self, base: Value, off: u32, dyns: &[(Value, u32)], v: &Val) {
        let mut lay = Vec::new();
        layout_leaves(&v.ty, self.structs(), 0, &mut lay);
        let n = lay.len().min(v.leaves.len());
        // Convert leaves to their stored representation once, outside the lane loop.
        let mut stored: Vec<(u32, Leaf)> = Vec::with_capacity(n);
        for i in 0..n {
            let l = v.leaves[i];
            let l = match l.s {
                ScalarTy::Bool => {
                    let w = self.shr_n(l.a, 31);
                    Leaf { a: w, ..l }
                }
                ScalarTy::F16 => {
                    let h = self.f32_to_f16bits(l.a);
                    Leaf { a: h, ..l }
                }
                _ => l,
            };
            stored.push((lay[i].0, l));
        }
        let extent = lay
            .iter()
            .map(|&(o, s)| {
                o + if s == ScalarTy::F64 {
                    8
                } else if s == ScalarTy::F16 {
                    2
                } else {
                    4
                }
            })
            .max()
            .unwrap_or(0);
        let exec = self.exec();
        let full = self.is_full();
        for lane in 0..4u8 {
            self.lane_guard_mem(
                exec,
                full,
                lane,
                base,
                off,
                dyns,
                extent,
                &[],
                &mut |sl, addr, _| {
                    for &(o, l) in &stored {
                        match l.s {
                            ScalarTy::F64 => {
                                let src = if lane < 2 { l.a } else { l.b.unwrap() };
                                let x = sl.b.ins().extractlane(src, lane % 2);
                                sl.b.ins().store(mf(), x, addr, o as i32);
                            }
                            ScalarTy::F16 => {
                                let x = sl.b.ins().extractlane(l.a, lane);
                                sl.b.ins().istore16(mf(), x, addr, o as i32);
                            }
                            _ => {
                                let x = sl.b.ins().extractlane(l.a, lane);
                                sl.b.ins().store(mf(), x, addr, o as i32);
                            }
                        }
                    }
                    Vec::new()
                },
            );
        }
    }

    fn contig_scalar(ty: &KirType, stride: u32) -> Option<ScalarTy> {
        match ty {
            KirType::Scalar(s @ (ScalarTy::F32 | ScalarTy::U32 | ScalarTy::I32)) if stride == 4 => {
                Some(*s)
            }
            _ => None,
        }
    }

    fn contig_addr(&mut self, base: Value, idx: Leaf) -> Value {
        let e = self.b.ins().extractlane(idx.a, 0u8);
        let e = self.b.ins().uextend(self.ptr, e);
        let e = self.b.ins().imul_imm(e, 4);
        self.b.ins().iadd(base, e)
    }

    /// `binding[idx]` with 128-bit loads when the index is contiguous across the packet.
    fn load_elem(&mut self, base: Value, idx: Leaf, stride: u32, ty: &KirType) -> Val {
        let Some(s) = Self::contig_scalar(ty, stride).filter(|_| idx.k == Kind::Lin) else {
            return self.mem_read(base, 0, &[(idx.a, stride)], ty);
        };
        let vt = vty(s);
        let Some(cond) = self.fast_cond(base, idx) else {
            let addr = self.contig_addr(base, idx);
            let a = self.b.ins().load(vt, mf(), addr, 0);
            return self.scalar_val(s, self.lf(s, a));
        };
        let fast = self.b.create_block();
        let slow = self.b.create_block();
        let merge = self.b.create_block();
        let res = self.b.append_block_param(merge, vt);
        self.b.ins().brif(cond, fast, &[], slow, &[]);
        self.b.switch_to_block(fast);
        let addr = self.contig_addr(base, idx);
        let a = self.b.ins().load(vt, mf(), addr, 0);
        self.b.ins().jump(merge, &[a.into()]);
        self.b.switch_to_block(slow);
        let g = self.mem_read(base, 0, &[(idx.a, stride)], ty);
        self.b.ins().jump(merge, &[g.leaves[0].a.into()]);
        self.b.switch_to_block(merge);
        self.scalar_val(s, self.lf(s, res))
    }

    /// Condition under which the 128-bit contiguous fast path is valid: all four lanes active and the
    /// whole packet inside the bound binding. `None` when statically always true.
    fn fast_cond(&mut self, base: Value, idx: Leaf) -> Option<Value> {
        let mut cond = None;
        if !self.is_full() {
            let exec = self.exec();
            cond = Some(self.b.ins().vall_true(exec));
        }
        if let Some(&lim) = self.limits.get(&base) {
            let e = self.b.ins().extractlane(idx.a, 0u8);
            let e = self.b.ins().uextend(types::I64, e);
            let end = self.b.ins().imul_imm(e, 4);
            let end = self.b.ins().iadd_imm(end, 16);
            let ok = self.b.ins().icmp(IntCC::UnsignedLessThanOrEqual, end, lim);
            cond = Some(match cond {
                Some(c) => self.b.ins().band(c, ok),
                None => ok,
            });
        }
        cond
    }

    fn store_elem(&mut self, base: Value, off: u32, idx: Leaf, stride: u32, v: &Val) {
        let fast_ok =
            idx.k == Kind::Lin && off == 0 && Self::contig_scalar(&v.ty, stride).is_some();
        if !fast_ok {
            self.mem_write(base, off, &[(idx.a, stride)], v);
            return;
        }
        let x = v.leaves[0].a;
        let Some(cond) = self.fast_cond(base, idx) else {
            let addr = self.contig_addr(base, idx);
            self.b.ins().store(mf(), x, addr, 0);
            return;
        };
        let fast = self.b.create_block();
        let slow = self.b.create_block();
        let merge = self.b.create_block();
        self.b.ins().brif(cond, fast, &[], slow, &[]);
        self.b.switch_to_block(fast);
        let addr = self.contig_addr(base, idx);
        self.b.ins().store(mf(), x, addr, 0);
        self.b.ins().jump(merge, &[]);
        self.b.switch_to_block(slow);
        self.mem_write(base, off, &[(idx.a, stride)], v);
        self.b.ins().jump(merge, &[]);
        self.b.switch_to_block(merge);
    }

    /// Calls `f(addr, lane)` for each active lane of an atomic target.
    fn atomic_lanes(&mut self, target: &AtomicTarget, f: &mut dyn FnMut(&mut Self, Value, u8)) {
        let (base, idx, stride) = match target {
            AtomicTarget::Storage { binding, index } => {
                let Some(base) = self.binding_ptr(*binding) else {
                    return;
                };
                let idx = index.as_ref().map(|e| self.expr(e).leaves[0].a);
                (base, idx, self.storage_stride(*binding))
            }
            AtomicTarget::Shared { shared, index } => {
                let base = self.shared_ptr(*shared);
                let idx = index.as_ref().map(|e| self.expr(e).leaves[0].a);
                (
                    base,
                    idx,
                    self.module.shared[*shared as usize]
                        .elem_ty
                        .layout(self.structs())
                        .0,
                )
            }
        };
        let exec = self.exec();
        let full = self.is_full();
        let dyns: Vec<(Value, u32)> = idx.map(|i| (i, stride)).into_iter().collect();
        for lane in 0..4u8 {
            self.lane_guard_mem(exec, full, lane, base, 0, &dyns, 4, &[], &mut |sl, addr, _| {
                f(sl, addr, lane);
                Vec::new()
            });
        }
    }

    // -----------------------------------------------------------------------------------------
    // f16 conversion
    // -----------------------------------------------------------------------------------------

    fn f32_to_f16bits(&mut self, x: Value) -> Value {
        let bits = self.bitcast_ty(I32X4, x);
        let sh = self.shr_n(bits, 16);
        let sign = self.and_c(sh, 0x8000);
        let abs = self.and_c(bits, 0x7fff_ffff);
        let off = self.splat_u32(0x3800_0000);
        let t = self.b.ins().isub(abs, off);
        let s13 = self.shr_n(abs, 13);
        let lsb = self.and_c(s13, 1);
        let k = self.splat_u32(0xfff);
        let t = self.b.ins().iadd(t, k);
        let t = self.b.ins().iadd(t, lsb);
        let normal = self.shr_n(t, 13);
        let cap = self.splat_u32(0x7c00);
        let normal = self.b.ins().umin(normal, cap);
        let absf = self.bitcast_ty(F32X4, abs);
        let scale = self.splat_f32(16777216.0);
        let scaled = self.b.ins().fmul(absf, scale);
        let scaled = self.b.ins().nearest(scaled);
        let sub = self.b.ins().fcvt_to_sint_sat(I32X4, scaled);
        let is_sub = self.icmp_c(IntCC::UnsignedLessThan, abs, 0x3880_0000);
        let r = self.bsel(is_sub, sub, normal);
        let is_nan = self.icmp_c(IntCC::UnsignedGreaterThan, abs, 0x7f80_0000);
        let nan = self.splat_u32(0x7e00);
        let r = self.bsel(is_nan, nan, r);
        self.b.ins().bor(r, sign)
    }

    fn f16bits_to_f32(&mut self, h: Value) -> Value {
        let s = self.and_c(h, 0x8000);
        let sign = self.shl_n(s, 16);
        let e = self.shr_n(h, 10);
        let exp = self.and_c(e, 0x1f);
        let man = self.and_c(h, 0x3ff);
        let bias = self.splat_u32(112);
        let e2 = self.b.ins().iadd(exp, bias);
        let e2 = self.shl_n(e2, 23);
        let m13 = self.shl_n(man, 13);
        let normal = self.b.ins().bor(e2, m13);
        let mf32 = self.b.ins().fcvt_from_sint(F32X4, man);
        let sc = self.splat_f32(1.0 / 16777216.0);
        let subf = self.b.ins().fmul(mf32, sc);
        let sub = self.bitcast_ty(I32X4, subf);
        let inf = self.splat_u32(0x7f80_0000);
        let infnan = self.b.ins().bor(inf, m13);
        let is_zero = self.icmp_c(IntCC::Equal, exp, 0);
        let is_max = self.icmp_c(IntCC::Equal, exp, 31);
        let r = self.bsel(is_max, infnan, normal);
        let r = self.bsel(is_zero, sub, r);
        let r = self.b.ins().bor(r, sign);
        self.bitcast_ty(F32X4, r)
    }

    fn round_f16(&mut self, x: Value) -> Value {
        let h = self.f32_to_f16bits(x);
        self.f16bits_to_f32(h)
    }
}

// ---------------------------------------------------------------------------------------------
// Expressions
// ---------------------------------------------------------------------------------------------

impl<'a, 'f> Lower<'a, 'f> {
    fn expr(&mut self, e: &Expr) -> Val {
        match e {
            Expr::LitBool(v) => {
                let c = self.splat_i32(if *v { -1 } else { 0 });
                self.scalar_val(ScalarTy::Bool, self.uni(ScalarTy::Bool, c))
            }
            Expr::LitU32(v) => {
                let c = self.splat_u32(*v);
                self.scalar_val(ScalarTy::U32, self.uni(ScalarTy::U32, c))
            }
            Expr::LitI32(v) => {
                let c = self.splat_i32(*v);
                self.scalar_val(ScalarTy::I32, self.uni(ScalarTy::I32, c))
            }
            Expr::LitF16(v) => {
                let c = self.splat_f32(*v);
                self.scalar_val(ScalarTy::F16, self.uni(ScalarTy::F16, c))
            }
            Expr::LitF32(v) => {
                let c = self.splat_f32(*v);
                self.scalar_val(ScalarTy::F32, self.uni(ScalarTy::F32, c))
            }
            Expr::LitF64(v) => {
                let c = self.splat_f64(*v);
                self.scalar_val(
                    ScalarTy::F64,
                    Leaf {
                        s: ScalarTy::F64,
                        a: c,
                        b: Some(c),
                        k: Kind::Uni,
                    },
                )
            }
            Expr::ZeroVal(ty) => self.zero_val(ty),
            Expr::Local(l) => self.read_local(*l),
            Expr::FnParam(i) => self.frame().params[*i as usize].clone(),
            Expr::Private(p) => {
                let ty = self.module.privates[*p as usize].ty.clone();
                let scs = scalars_of(&ty, self.structs());
                let leaves = self.scratch_load(self.plan.priv_scratch_offsets[*p as usize], &scs);
                Val { ty, leaves }
            }
            Expr::GridSize(i) => {
                let v = self.b.ins().load(
                    types::I32,
                    mf_ro(),
                    self.ctx,
                    WorkgroupCtx::OFFSET_GRID_SIZE + 4 * i32::from(*i),
                );
                let v = self.b.ins().splat(I32X4, v);
                self.scalar_val(ScalarTy::U32, self.uni(ScalarTy::U32, v))
            }
            Expr::ArrayLength { binding } => {
                let v = match self
                    .plan
                    .array_lengths
                    .iter()
                    .position(|&(b, _)| b == *binding)
                {
                    Some(pos) => {
                        let lp = self.b.ins().load(
                            self.ptr,
                            mf_ro(),
                            self.ctx,
                            WorkgroupCtx::OFFSET_LENGTHS,
                        );
                        let n = self.b.ins().load(types::I32, mf_ro(), lp, (pos * 4) as i32);
                        self.b.ins().splat(I32X4, n)
                    }
                    None => self.splat_i32(0),
                };
                self.scalar_val(ScalarTy::U32, self.uni(ScalarTy::U32, v))
            }
            Expr::UniformRead { binding, path, ty } | Expr::StorageRead { binding, path, ty } => {
                let steps = self.resolve_access(path);
                let start = self.storage_start_ty(*binding);
                let (off, dyns, _) = self.mem_path(&start, &steps);
                match self.binding_ptr(*binding) {
                    Some(base) => self.mem_read(base, off, &dyns, ty),
                    None => self.zero_val(ty),
                }
            }
            Expr::SharedRead { shared, path, ty } => {
                let steps = self.resolve_access(path);
                let start = self.shared_start_ty(*shared);
                let (off, dyns, _) = self.mem_path(&start, &steps);
                let base = self.shared_ptr(*shared);
                self.mem_read(base, off, &dyns, ty)
            }
            Expr::Load { binding, index, ty } => {
                let idx = self.expr(index).leaves[0];
                match self.binding_ptr(*binding) {
                    Some(base) => {
                        let stride = self.storage_stride(*binding);
                        self.load_elem(base, idx, stride, ty)
                    }
                    None => self.zero_val(ty),
                }
            }
            Expr::SharedLoad { shared, index, ty } => {
                let idx = self.expr(index).leaves[0];
                let base = self.shared_ptr(*shared);
                let stride = self.module.shared[*shared as usize]
                    .elem_ty
                    .layout(self.structs())
                    .0;
                self.load_elem(base, idx, stride, ty)
            }
            Expr::Access { base, step, .. } => {
                let b = self.expr(base);
                let st = match step {
                    AccessStep::Field(f) => RStep::Field(*f),
                    AccessStep::Index(e) => match lit_index(e) {
                        Some(k) => RStep::Const(k),
                        None => RStep::Dyn(self.expr(e).leaves[0].a),
                    },
                };
                self.access(&b, &st)
            }
            Expr::Swizzle {
                base,
                indices,
                len,
                ty,
            } => {
                let b = self.expr(base);
                let leaves = indices[..usize::from(*len)]
                    .iter()
                    .map(|&i| b.leaves[usize::from(i)])
                    .collect();
                Val {
                    ty: ty.clone(),
                    leaves,
                }
            }
            Expr::Compose { ty, comps } => {
                let mut leaves = Vec::new();
                for c in comps {
                    leaves.extend(self.expr(c).leaves);
                }
                Val {
                    ty: ty.clone(),
                    leaves,
                }
            }
            Expr::Splat { elem, lanes, val } => {
                let v = self.expr(val).leaves[0];
                Val {
                    ty: KirType::Vec {
                        elem: *elem,
                        lanes: *lanes,
                    },
                    leaves: vec![v; usize::from(*lanes)],
                }
            }
            Expr::Unary { op, arg, ty } => {
                let a = self.expr(arg);
                self.unary(*op, a, ty)
            }
            Expr::Binary { op, lhs, rhs, ty } => {
                let l = self.expr(lhs);
                let r = self.expr(rhs);
                self.binary(*op, &l, &r, ty)
            }
            Expr::Ternary { op, a, b, c, ty } => self.ternary(*op, a, b, c, ty),
            Expr::Cast {
                kind, expr, dst_ty, ..
            } => {
                let v = self.expr(expr);
                self.cast(*kind, v, dst_ty)
            }
            Expr::Call { func, args, ty } => self.call(func.0 as usize, args, ty.as_ref()),
            Expr::AtomicLoad { target, ty } => {
                let acc = self.b.declare_var(I32X4);
                let z = self.splat_i32(0);
                self.b.def_var(acc, z);
                self.atomic_lanes(target, &mut |s, addr, lane| {
                    let x =
                        s.b.ins()
                            .atomic_load(types::I32, MemFlagsData::trusted(), addr);
                    let cur = s.b.use_var(acc);
                    let n = s.b.ins().insertlane(cur, x, lane);
                    s.b.def_var(acc, n);
                });
                let a = self.b.use_var(acc);
                let a = if *ty == ScalarTy::F32 {
                    self.b.ins().bitcast(F32X4, mf_bc(), a)
                } else {
                    a
                };
                self.scalar_val(*ty, self.lf(*ty, a))
            }
            Expr::Subgroup { op, arg, lane, ty } => {
                self.subgroup(*op, arg.as_deref(), lane.as_deref(), ty)
            }
        }
    }

    fn call(&mut self, func: usize, args: &[Expr], ty: Option<&KirType>) -> Val {
        let module = self.module;
        let callee = &module.functions[func];
        let argv: Vec<Val> = args.iter().map(|a| self.expr(a)).collect();
        let let_only = let_only_locals(callee.locals.len(), &callee.body);
        let locals: Vec<LocalVars> = callee
            .locals
            .iter()
            .zip(let_only)
            .map(|((_, t, _), only)| self.make_local(t, only))
            .collect();
        let ret = callee.ret_ty.as_ref().map(|t| {
            let lv = self.make_local(t, false);
            (lv.ty, lv.scs, lv.vars)
        });
        let exec = self.exec();
        let parent_full = self.is_full();
        let mut has_ret = false;
        walk_stmts(&callee.body, &mut |s| {
            has_ret |= matches!(s, Stmt::Return(_))
        });
        self.frames.push(Frame {
            locals,
            params: argv,
            ret,
            entry: false,
        });
        self.push_scope(exec, ScopeKind::CallRoot);
        self.scopes.last_mut().unwrap().full = parent_full && !has_ret;
        self.stmts(&callee.body);
        self.scopes.pop();
        let frame = self.frames.pop().unwrap();
        match frame.ret {
            Some((rty, scs, vars)) => {
                let mut leaves = Vec::with_capacity(vars.len());
                for (i, v) in vars.iter().enumerate() {
                    let a = self.b.use_var(v.0);
                    let b = v.1.map(|h| self.b.use_var(h));
                    leaves.push(Leaf {
                        s: scs[i],
                        a,
                        b,
                        k: Kind::Gen,
                    });
                }
                Val { ty: rty, leaves }
            }
            None => Val {
                ty: ty.cloned().unwrap_or(KirType::U32),
                leaves: Vec::new(),
            },
        }
    }

    // -----------------------------------------------------------------------------------------
    // Float helpers
    // -----------------------------------------------------------------------------------------

    fn fl(&mut self, a: Leaf, f: &mut dyn FnMut(&mut Self, Value) -> Value) -> Leaf {
        let x = f(self, a.a);
        let y = a.b.map(|h| f(self, h));
        let mut r = Leaf {
            s: a.s,
            a: x,
            b: y,
            k: Kind::Gen,
        };
        if a.s == ScalarTy::F16 {
            r.a = self.round_f16(r.a);
        }
        r
    }

    fn fl2(
        &mut self,
        a: Leaf,
        b: Leaf,
        f: &mut dyn FnMut(&mut Self, Value, Value) -> Value,
    ) -> Leaf {
        let x = f(self, a.a, b.a);
        let y = a.b.map(|h| f(self, h, b.b.unwrap()));
        let mut r = Leaf {
            s: a.s,
            a: x,
            b: y,
            k: Kind::Gen,
        };
        if a.s == ScalarTy::F16 {
            r.a = self.round_f16(r.a);
        }
        r
    }

    fn fminmax(&mut self, x: Value, y: Value, max: bool) -> Value {
        let nx = self.b.ins().fcmp(FloatCC::Unordered, x, x);
        let ny = self.b.ins().fcmp(FloatCC::Unordered, y, y);
        let m = if max {
            self.b.ins().fmax(x, y)
        } else {
            self.b.ins().fmin(x, y)
        };
        let r = self.bsel(ny, x, m);
        self.bsel(nx, y, r)
    }

    fn fcmp_leaf(&mut self, cc: FloatCC, a: Leaf, b: Leaf) -> Leaf {
        let lo = self.b.ins().fcmp(cc, a.a, b.a);
        let m = match a.b {
            Some(h) => {
                let hi = self.b.ins().fcmp(cc, h, b.b.unwrap());
                self.narrow_mask64(lo, hi)
            }
            None => lo,
        };
        Leaf {
            s: ScalarTy::Bool,
            a: m,
            b: None,
            k: Kind::Gen,
        }
    }

    fn binary(&mut self, op: IrBinOp, l: &Val, r: &Val, ty: &KirType) -> Val {
        let n = l.leaves.len().max(r.leaves.len());
        let mut leaves = Vec::with_capacity(n);
        for i in 0..n {
            let a = l.leaves[if l.leaves.len() == 1 { 0 } else { i }];
            let b = r.leaves[if r.leaves.len() == 1 { 0 } else { i }];
            leaves.push(self.bin_leaf(op, a, b));
        }
        Val {
            ty: ty.clone(),
            leaves,
        }
    }

    fn bin_leaf(&mut self, op: IrBinOp, a: Leaf, b: Leaf) -> Leaf {
        use IrBinOp::*;
        let s = a.s;
        if s.is_float() {
            let cc = match op {
                Eq => Some(FloatCC::Equal),
                Ne => Some(FloatCC::NotEqual),
                Lt => Some(FloatCC::LessThan),
                Le => Some(FloatCC::LessThanOrEqual),
                Gt => Some(FloatCC::GreaterThan),
                Ge => Some(FloatCC::GreaterThanOrEqual),
                _ => None,
            };
            if let Some(cc) = cc {
                return self.fcmp_leaf(cc, a, b);
            }
            let is64 = s == ScalarTy::F64;
            let mut r = self.fl2(a, b, &mut |sl, x, y| match op {
                Add | AddExact => sl.b.ins().fadd(x, y),
                Sub | SubExact => sl.b.ins().fsub(x, y),
                Mul | MulExact => sl.b.ins().fmul(x, y),
                Div => sl.b.ins().fdiv(x, y),
                Rem => {
                    let q = sl.b.ins().fdiv(x, y);
                    let t = sl.b.ins().trunc(q);
                    let m = sl.b.ins().fmul(t, y);
                    sl.b.ins().fsub(x, m)
                }
                Min => sl.fminmax(x, y, false),
                Max => sl.fminmax(x, y, true),
                Copysign => {
                    let sign = if is64 {
                        sl.splat_f64_bits(1 << 63)
                    } else {
                        sl.splat_f32_bits(0x8000_0000)
                    };
                    let sign = sl.bitcast_ty(I32X4, sign);
                    sl.bsel(sign, y, x)
                }
                Pow => sl.math_call(
                    CpuMathTable::OFFSET_POWF,
                    CpuMathTable::OFFSET_POW,
                    is64,
                    &[x, y],
                ),
                Atan2 => sl.math_call(
                    CpuMathTable::OFFSET_ATAN2F,
                    CpuMathTable::OFFSET_ATAN2,
                    is64,
                    &[x, y],
                ),
                _ => x,
            });
            r.k = Self::kuni(&[a.k, b.k]);
            return r;
        }
        // Integer and bool lanes.
        let signed = s == ScalarTy::I32;
        let cmp = |cc_s: IntCC, cc_u: IntCC| if signed { cc_s } else { cc_u };
        let (v, k) = match op {
            Add | AddExact => (self.b.ins().iadd(a.a, b.a), Self::kadd(a.k, b.k)),
            Sub | SubExact => (self.b.ins().isub(a.a, b.a), Self::ksub(a.k, b.k)),
            Mul | MulExact => (self.b.ins().imul(a.a, b.a), Self::kuni(&[a.k, b.k])),
            Div | Rem => {
                let is_div = op == Div;
                let v = self.lanewise(I32X4, &[a.a, b.a], &|bd, x| {
                    let zero = bd.ins().icmp_imm(IntCC::Equal, x[1], 0);
                    let bad = if signed {
                        let min = bd.ins().icmp_imm(IntCC::Equal, x[0], i64::from(i32::MIN));
                        let m1 = bd.ins().icmp_imm(IntCC::Equal, x[1], -1);
                        let both = bd.ins().band(min, m1);
                        bd.ins().bor(zero, both)
                    } else {
                        zero
                    };
                    let one = bd.ins().iconst(types::I32, 1);
                    let den = bd.ins().select(bad, one, x[1]);
                    match (signed, is_div) {
                        (true, true) => bd.ins().sdiv(x[0], den),
                        (true, false) => bd.ins().srem(x[0], den),
                        (false, true) => bd.ins().udiv(x[0], den),
                        (false, false) => bd.ins().urem(x[0], den),
                    }
                });
                (v, Kind::Gen)
            }
            Min => (
                if signed {
                    self.b.ins().smin(a.a, b.a)
                } else {
                    self.b.ins().umin(a.a, b.a)
                },
                Kind::Gen,
            ),
            Max => (
                if signed {
                    self.b.ins().smax(a.a, b.a)
                } else {
                    self.b.ins().umax(a.a, b.a)
                },
                Kind::Gen,
            ),
            BitAnd | And => (self.b.ins().band(a.a, b.a), Kind::Gen),
            BitOr | Or => (self.b.ins().bor(a.a, b.a), Kind::Gen),
            BitXor => (self.b.ins().bxor(a.a, b.a), Kind::Gen),
            Shl | Shr => {
                let v = if b.k == Kind::Uni {
                    let amt = self.b.ins().extractlane(b.a, 0u8);
                    match (op, signed) {
                        (Shl, _) => self.b.ins().ishl(a.a, amt),
                        (_, true) => self.b.ins().sshr(a.a, amt),
                        _ => self.b.ins().ushr(a.a, amt),
                    }
                } else {
                    self.lanewise(I32X4, &[a.a, b.a], &|bd, x| match (op, signed) {
                        (Shl, _) => bd.ins().ishl(x[0], x[1]),
                        (_, true) => bd.ins().sshr(x[0], x[1]),
                        _ => bd.ins().ushr(x[0], x[1]),
                    })
                };
                (v, Kind::Gen)
            }
            Rotl => (
                self.lanewise(I32X4, &[a.a, b.a], &|bd, x| bd.ins().rotl(x[0], x[1])),
                Kind::Gen,
            ),
            MulHiU32 => (
                self.lanewise(I32X4, &[a.a, b.a], &|bd, x| bd.ins().umulhi(x[0], x[1])),
                Kind::Gen,
            ),
            Eq => (self.b.ins().icmp(IntCC::Equal, a.a, b.a), Kind::Gen),
            Ne => (self.b.ins().icmp(IntCC::NotEqual, a.a, b.a), Kind::Gen),
            Lt => (
                self.b.ins().icmp(
                    cmp(IntCC::SignedLessThan, IntCC::UnsignedLessThan),
                    a.a,
                    b.a,
                ),
                Kind::Gen,
            ),
            Le => (
                self.b.ins().icmp(
                    cmp(IntCC::SignedLessThanOrEqual, IntCC::UnsignedLessThanOrEqual),
                    a.a,
                    b.a,
                ),
                Kind::Gen,
            ),
            Gt => (
                self.b.ins().icmp(
                    cmp(IntCC::SignedGreaterThan, IntCC::UnsignedGreaterThan),
                    a.a,
                    b.a,
                ),
                Kind::Gen,
            ),
            Ge => (
                self.b.ins().icmp(
                    cmp(
                        IntCC::SignedGreaterThanOrEqual,
                        IntCC::UnsignedGreaterThanOrEqual,
                    ),
                    a.a,
                    b.a,
                ),
                Kind::Gen,
            ),
            Pow | Atan2 | Copysign => (a.a, Kind::Gen),
        };
        let out_s = if op.is_comparison() {
            ScalarTy::Bool
        } else {
            s
        };
        let k = if k == Kind::Gen {
            Self::kuni(&[a.k, b.k])
        } else {
            k
        };
        Leaf {
            s: out_s,
            a: v,
            b: None,
            k,
        }
    }

    fn unary(&mut self, op: IrUnOp, a: Val, ty: &KirType) -> Val {
        use IrUnOp::*;
        if matches!(op, All | Any) {
            let mut acc = a.leaves[0].a;
            for l in &a.leaves[1..] {
                acc = if op == All {
                    self.b.ins().band(acc, l.a)
                } else {
                    self.b.ins().bor(acc, l.a)
                };
            }
            return Val {
                ty: ty.clone(),
                leaves: vec![Leaf {
                    s: ScalarTy::Bool,
                    a: acc,
                    b: None,
                    k: Kind::Gen,
                }],
            };
        }
        let leaves = a.leaves.iter().map(|&l| self.un_leaf(op, l)).collect();
        Val {
            ty: ty.clone(),
            leaves,
        }
    }

    fn un_leaf(&mut self, op: IrUnOp, a: Leaf) -> Leaf {
        use IrUnOp::*;
        let s = a.s;
        let is64 = s == ScalarTy::F64;
        if s.is_float() {
            match op {
                IsNan => {
                    let lo = self.b.ins().fcmp(FloatCC::Unordered, a.a, a.a);
                    let m = match a.b {
                        Some(h) => {
                            let hi = self.b.ins().fcmp(FloatCC::Unordered, h, h);
                            self.narrow_mask64(lo, hi)
                        }
                        None => lo,
                    };
                    return Leaf {
                        s: ScalarTy::Bool,
                        a: m,
                        b: None,
                        k: Kind::Gen,
                    };
                }
                IsInf => {
                    let inf = if is64 {
                        self.splat_f64(f64::INFINITY)
                    } else {
                        self.splat_f32(f32::INFINITY)
                    };
                    let abs = self.b.ins().fabs(a.a);
                    let lo = self.b.ins().fcmp(FloatCC::Equal, abs, inf);
                    let m = match a.b {
                        Some(h) => {
                            let abs = self.b.ins().fabs(h);
                            let hi = self.b.ins().fcmp(FloatCC::Equal, abs, inf);
                            self.narrow_mask64(lo, hi)
                        }
                        None => lo,
                    };
                    return Leaf {
                        s: ScalarTy::Bool,
                        a: m,
                        b: None,
                        k: Kind::Gen,
                    };
                }
                OpaqueGuard => return a,
                _ => {}
            }
            let mut r = self.fl(a, &mut |sl, x| match op {
                Neg => sl.b.ins().fneg(x),
                Abs => sl.b.ins().fabs(x),
                Sqrt => sl.b.ins().sqrt(x),
                InverseSqrt => {
                    let one = if is64 {
                        sl.splat_f64(1.0)
                    } else {
                        sl.splat_f32(1.0)
                    };
                    let q = sl.b.ins().sqrt(x);
                    sl.b.ins().fdiv(one, q)
                }
                Floor => sl.b.ins().floor(x),
                Ceil => sl.b.ins().ceil(x),
                Trunc => sl.b.ins().trunc(x),
                Round => sl.b.ins().nearest(x),
                Sign => {
                    let ty = if is64 { F64X2 } else { F32X4 };
                    sl.lanewise(ty, &[x], &|bd, v| {
                        let (z, one, mone) = if is64 {
                            (
                                bd.ins().f64const(Ieee64::with_float(0.0)),
                                bd.ins().f64const(Ieee64::with_float(1.0)),
                                bd.ins().f64const(Ieee64::with_float(-1.0)),
                            )
                        } else {
                            (
                                bd.ins().f32const(Ieee32::with_float(0.0)),
                                bd.ins().f32const(Ieee32::with_float(1.0)),
                                bd.ins().f32const(Ieee32::with_float(-1.0)),
                            )
                        };
                        let gt = bd.ins().fcmp(FloatCC::GreaterThan, v[0], z);
                        let lt = bd.ins().fcmp(FloatCC::LessThan, v[0], z);
                        let inner = bd.ins().select(lt, mone, z);
                        bd.ins().select(gt, one, inner)
                    })
                }
                Exp => sl.math_call(
                    CpuMathTable::OFFSET_EXPF,
                    CpuMathTable::OFFSET_EXP,
                    is64,
                    &[x],
                ),
                Ln => sl.math_call(
                    CpuMathTable::OFFSET_LOGF,
                    CpuMathTable::OFFSET_LOG,
                    is64,
                    &[x],
                ),
                Sin => sl.math_call(
                    CpuMathTable::OFFSET_SINF,
                    CpuMathTable::OFFSET_SIN,
                    is64,
                    &[x],
                ),
                Cos => sl.math_call(
                    CpuMathTable::OFFSET_COSF,
                    CpuMathTable::OFFSET_COS,
                    is64,
                    &[x],
                ),
                Tan => sl.math_call(
                    CpuMathTable::OFFSET_TANF,
                    CpuMathTable::OFFSET_TAN,
                    is64,
                    &[x],
                ),
                Tanh => sl.math_call(
                    CpuMathTable::OFFSET_TANHF,
                    CpuMathTable::OFFSET_TANH,
                    is64,
                    &[x],
                ),
                Exp2 => sl.math_call(
                    CpuMathTable::OFFSET_EXP2F,
                    CpuMathTable::OFFSET_EXP2,
                    is64,
                    &[x],
                ),
                Log2 => sl.math_call(
                    CpuMathTable::OFFSET_LOG2F,
                    CpuMathTable::OFFSET_LOG2,
                    is64,
                    &[x],
                ),
                Asin => sl.math_call(
                    CpuMathTable::OFFSET_ASINF,
                    CpuMathTable::OFFSET_ASIN,
                    is64,
                    &[x],
                ),
                Acos => sl.math_call(
                    CpuMathTable::OFFSET_ACOSF,
                    CpuMathTable::OFFSET_ACOS,
                    is64,
                    &[x],
                ),
                Atan => sl.math_call(
                    CpuMathTable::OFFSET_ATANF,
                    CpuMathTable::OFFSET_ATAN,
                    is64,
                    &[x],
                ),
                _ => x,
            });
            r.k = if a.k == Kind::Uni {
                Kind::Uni
            } else {
                Kind::Gen
            };
            return r;
        }
        let signed = s == ScalarTy::I32;
        let v = match op {
            Neg => self.b.ins().ineg(a.a),
            Not | BitNot => self.b.ins().bnot(a.a),
            Abs => {
                if signed {
                    self.b.ins().iabs(a.a)
                } else {
                    a.a
                }
            }
            Sign => {
                let z = self.splat_i32(0);
                if signed {
                    let lt = self.b.ins().icmp(IntCC::SignedLessThan, a.a, z);
                    let gt = self.b.ins().icmp(IntCC::SignedGreaterThan, a.a, z);
                    self.b.ins().isub(lt, gt)
                } else {
                    let nz = self.b.ins().icmp(IntCC::NotEqual, a.a, z);
                    self.shr_n(nz, 31)
                }
            }
            CountOneBits => self.lanewise(I32X4, &[a.a], &|bd, x| bd.ins().popcnt(x[0])),
            CountLeadingZeros => self.lanewise(I32X4, &[a.a], &|bd, x| bd.ins().clz(x[0])),
            CountTrailingZeros => self.lanewise(I32X4, &[a.a], &|bd, x| bd.ins().ctz(x[0])),
            FirstTrailingBit => self.lanewise(I32X4, &[a.a], &|bd, x| {
                let z = bd.ins().icmp_imm(IntCC::Equal, x[0], 0);
                let m1 = bd.ins().iconst(types::I32, -1);
                let t = bd.ins().ctz(x[0]);
                bd.ins().select(z, m1, t)
            }),
            FirstLeadingBit => self.lanewise(I32X4, &[a.a], &|bd, x| {
                let y = if signed {
                    let neg = bd.ins().icmp_imm(IntCC::SignedLessThan, x[0], 0);
                    let inv = bd.ins().bnot(x[0]);
                    bd.ins().select(neg, inv, x[0])
                } else {
                    x[0]
                };
                let z = bd.ins().icmp_imm(IntCC::Equal, y, 0);
                let m1 = bd.ins().iconst(types::I32, -1);
                let c = bd.ins().clz(y);
                let k31 = bd.ins().iconst(types::I32, 31);
                let t = bd.ins().isub(k31, c);
                bd.ins().select(z, m1, t)
            }),
            OpaqueGuard => a.a,
            _ => a.a,
        };
        let k = if matches!(op, Neg | Not | BitNot) && a.k == Kind::Uni {
            Kind::Uni
        } else {
            Kind::Gen
        };
        Leaf {
            s,
            a: v,
            b: None,
            k,
        }
    }

    fn ternary(&mut self, op: IrTernaryOp, a: &Expr, b: &Expr, c: &Expr, ty: &KirType) -> Val {
        match op {
            IrTernaryOp::Select => {
                let cond = self.expr(a);
                let narrow = expr_has_mem_or_div(b) || expr_has_mem_or_div(c);
                let (bv, cv) = if narrow {
                    let m0 = cond.leaves[0].a;
                    let exec = self.exec();
                    let tm = self.b.ins().band(exec, m0);
                    let em = self.b.ins().band_not(exec, m0);
                    self.push_scope(tm, ScopeKind::Plain);
                    let bv = self.expr(b);
                    self.scopes.pop();
                    self.push_scope(em, ScopeKind::Plain);
                    let cv = self.expr(c);
                    self.scopes.pop();
                    (bv, cv)
                } else {
                    let bv = self.expr(b);
                    let cv = self.expr(c);
                    (bv, cv)
                };
                let n = bv.leaves.len();
                let mut leaves = Vec::with_capacity(n);
                for i in 0..n {
                    let m = cond.leaves[if cond.leaves.len() == 1 { 0 } else { i }];
                    let mut l = self.select_leaf(m.a, bv.leaves[i], cv.leaves[i]);
                    l.k = Self::kuni(&[m.k, bv.leaves[i].k, cv.leaves[i].k]);
                    leaves.push(l);
                }
                Val {
                    ty: ty.clone(),
                    leaves,
                }
            }
            IrTernaryOp::Fma => {
                let x = self.expr(a);
                let y = self.expr(b);
                let z = self.expr(c);
                let mut leaves = Vec::new();
                for i in 0..x.leaves.len() {
                    let (p, q, r) = (x.leaves[i], y.leaves[i], z.leaves[i]);
                    let lo = self.b.ins().fma(p.a, q.a, r.a);
                    let hi = p.b.map(|h| self.b.ins().fma(h, q.b.unwrap(), r.b.unwrap()));
                    let mut l = Leaf {
                        s: p.s,
                        a: lo,
                        b: hi,
                        k: Kind::Gen,
                    };
                    if p.s == ScalarTy::F16 {
                        l.a = self.round_f16(l.a);
                    }
                    leaves.push(l);
                }
                Val {
                    ty: ty.clone(),
                    leaves,
                }
            }
            IrTernaryOp::Clamp => {
                let x = self.expr(a);
                let lo = self.expr(b);
                let hi = self.expr(c);
                let mut leaves = Vec::new();
                for i in 0..x.leaves.len() {
                    let l = lo.leaves[if lo.leaves.len() == 1 { 0 } else { i }];
                    let h = hi.leaves[if hi.leaves.len() == 1 { 0 } else { i }];
                    let m = self.bin_leaf(IrBinOp::Max, x.leaves[i], l);
                    leaves.push(self.bin_leaf(IrBinOp::Min, m, h));
                }
                Val {
                    ty: ty.clone(),
                    leaves,
                }
            }
        }
    }

    // -----------------------------------------------------------------------------------------
    // Casts
    // -----------------------------------------------------------------------------------------

    fn cast(&mut self, kind: CastKind, v: Val, dst: &KirType) -> Val {
        let d = dst.elem_scalar().unwrap_or(ScalarTy::U32);
        let leaves = v
            .leaves
            .iter()
            .map(|&l| self.cast_leaf(kind, l, d))
            .collect();
        Val {
            ty: dst.clone(),
            leaves,
        }
    }

    fn scalar_cast(&mut self, src: ScalarTy, dst: ScalarTy, x: Value) -> Value {
        match (src, dst) {
            (ScalarTy::F64, ScalarTy::F32 | ScalarTy::F16) => self.b.ins().fdemote(types::F32, x),
            (ScalarTy::F64, ScalarTy::I32) => self.b.ins().fcvt_to_sint_sat(types::I32, x),
            (ScalarTy::F64, ScalarTy::U32) => self.b.ins().fcvt_to_uint_sat(types::I32, x),
            (ScalarTy::F64, ScalarTy::Bool) => {
                let z = self.b.ins().f64const(Ieee64::with_float(0.0));
                let c = self.b.ins().fcmp(FloatCC::NotEqual, x, z);
                let t = self.b.ins().iconst(types::I32, -1);
                let f = self.b.ins().iconst(types::I32, 0);
                self.b.ins().select(c, t, f)
            }
            (ScalarTy::F32 | ScalarTy::F16, ScalarTy::F64) => self.b.ins().fpromote(types::F64, x),
            (ScalarTy::I32, ScalarTy::F64) => self.b.ins().fcvt_from_sint(types::F64, x),
            (ScalarTy::U32, ScalarTy::F64) => self.b.ins().fcvt_from_uint(types::F64, x),
            (ScalarTy::Bool, ScalarTy::F64) => {
                let one = self.b.ins().ushr_imm(x, 31);
                self.b.ins().fcvt_from_uint(types::F64, one)
            }
            _ => x,
        }
    }

    fn cast_leaf(&mut self, kind: CastKind, l: Leaf, d: ScalarTy) -> Leaf {
        let s = l.s;
        let keep = if matches!(d, ScalarTy::U32 | ScalarTy::I32)
            && matches!(s, ScalarTy::U32 | ScalarTy::I32)
        {
            l.k
        } else {
            Kind::Gen
        };
        if s == d {
            return l;
        }
        if s == ScalarTy::F64 || d == ScalarTy::F64 {
            let sc = self.lane_scalars(l);
            let out = [0usize, 1, 2, 3].map(|i| self.scalar_cast(s, d, sc[i]));
            return self.leaf_from_scalars(d, out);
        }
        let is_f = |t: ScalarTy| matches!(t, ScalarTy::F32 | ScalarTy::F16);
        let a = l.a;
        let v = if kind == CastKind::Bitcast
            && !matches!(s, ScalarTy::Bool)
            && !matches!(d, ScalarTy::Bool)
        {
            if is_f(d) {
                self.bitcast_ty(F32X4, a)
            } else {
                self.bitcast_ty(I32X4, a)
            }
        } else {
            match (s, d) {
                (ScalarTy::U32, ScalarTy::I32) | (ScalarTy::I32, ScalarTy::U32) => a,
                (ScalarTy::I32, f) if is_f(f) => self.b.ins().fcvt_from_sint(F32X4, a),
                (ScalarTy::U32, f) if is_f(f) => self.b.ins().fcvt_from_uint(F32X4, a),
                (f, ScalarTy::I32) if is_f(f) => self.b.ins().fcvt_to_sint_sat(I32X4, a),
                (f, ScalarTy::U32) if is_f(f) => self.b.ins().fcvt_to_uint_sat(I32X4, a),
                (ScalarTy::F32, ScalarTy::F16) => self.round_f16(a),
                (ScalarTy::F16, ScalarTy::F32) => a,
                (ScalarTy::Bool, ScalarTy::U32 | ScalarTy::I32) => self.shr_n(a, 31),
                (ScalarTy::Bool, f) if is_f(f) => {
                    let w = self.shr_n(a, 31);
                    self.b.ins().fcvt_from_sint(F32X4, w)
                }
                (ScalarTy::U32 | ScalarTy::I32, ScalarTy::Bool) => {
                    let z = self.splat_i32(0);
                    self.b.ins().icmp(IntCC::NotEqual, a, z)
                }
                (f, ScalarTy::Bool) if is_f(f) => {
                    let z = self.splat_f32(0.0);
                    self.b.ins().fcmp(FloatCC::NotEqual, a, z)
                }
                _ => a,
            }
        };
        Leaf {
            s: d,
            a: v,
            b: None,
            k: keep,
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Subgroups (one 4-lane packet is one subgroup)
// ---------------------------------------------------------------------------------------------

impl<'a, 'f> Lower<'a, 'f> {
    fn sc_zero(&mut self, s: ScalarTy) -> Value {
        match s {
            ScalarTy::F64 => self.b.ins().f64const(Ieee64::with_float(0.0)),
            ScalarTy::F32 | ScalarTy::F16 => self.b.ins().f32const(Ieee32::with_float(0.0)),
            _ => self.b.ins().iconst(types::I32, 0),
        }
    }

    /// Identity element of `min` (`max == false`) or `max` for scalar type `s`.
    fn sc_ident(&mut self, s: ScalarTy, max: bool) -> Value {
        match s {
            ScalarTy::F64 => self.b.ins().f64const(Ieee64::with_float(if max {
                f64::NEG_INFINITY
            } else {
                f64::INFINITY
            })),
            ScalarTy::F32 | ScalarTy::F16 => self.b.ins().f32const(Ieee32::with_float(if max {
                f32::NEG_INFINITY
            } else {
                f32::INFINITY
            })),
            ScalarTy::I32 => self
                .b
                .ins()
                .iconst(types::I32, i64::from(if max { i32::MIN } else { i32::MAX })),
            _ => self.b.ins().iconst(types::I32, if max { 0 } else { -1 }),
        }
    }

    fn sc_add(&mut self, s: ScalarTy, a: Value, b: Value) -> Value {
        if s.is_float() {
            self.b.ins().fadd(a, b)
        } else {
            self.b.ins().iadd(a, b)
        }
    }

    /// One step of `subgroupMin`/`subgroupMax`; floats combine as the `min`/`max` builtins do
    /// ([`Self::fminmax`]: a NaN operand yields the other), so the order lanes meet in is moot.
    fn sc_minmax(&mut self, s: ScalarTy, max: bool, a: Value, b: Value) -> Value {
        match (s.is_float(), s == ScalarTy::I32, max) {
            (true, _, _) => {
                let m = if max { self.b.ins().fmax(a, b) } else { self.b.ins().fmin(a, b) };
                let na = self.b.ins().fcmp(FloatCC::Unordered, a, a);
                let nb = self.b.ins().fcmp(FloatCC::Unordered, b, b);
                let r = self.b.ins().select(nb, a, m);
                self.b.ins().select(na, b, r)
            }
            (false, true, false) => self.b.ins().smin(a, b),
            (false, true, true) => self.b.ins().smax(a, b),
            (false, false, false) => self.b.ins().umin(a, b),
            (false, false, true) => self.b.ins().umax(a, b),
        }
    }

    fn subgroup(
        &mut self,
        op: IrSubgroupOp,
        arg: Option<&Expr>,
        lane: Option<&Expr>,
        ty: &KirType,
    ) -> Val {
        let exec = self.exec();
        if op == IrSubgroupOp::Ballot {
            let pred = match arg {
                Some(e) => self.expr(e).leaves[0].a,
                None => self.splat_i32(-1),
            };
            let m = self.b.ins().band(pred, exec);
            let mut bits: Option<Value> = None;
            for i in 0..4u8 {
                let l = self.b.ins().extractlane(m, i);
                let b = self.b.ins().band_imm(l, 1);
                let b = self.b.ins().ishl_imm(b, i64::from(i));
                bits = Some(match bits {
                    None => b,
                    Some(x) => self.b.ins().bor(x, b),
                });
            }
            let v = self.b.ins().splat(I32X4, bits.unwrap());
            let z = self.splat_i32(0);
            let mut leaves = vec![self.uni(ScalarTy::U32, v)];
            leaves.extend((1..ty.lanes().max(1)).map(|_| self.uni(ScalarTy::U32, z)));
            return Val {
                ty: ty.clone(),
                leaves,
            };
        }
        let a = self.expr(arg.expect("subgroup op needs an argument"));
        let lane_idx = lane.map(|e| self.expr(e).leaves[0].a);
        let act: Vec<Value> = (0..4u8)
            .map(|i| {
                let m = self.b.ins().extractlane(exec, i);
                self.b.ins().icmp_imm(IntCC::NotEqual, m, 0)
            })
            .collect();
        let mut leaves = Vec::with_capacity(a.leaves.len());
        for l in &a.leaves {
            let s = l.s;
            let sc = self.lane_scalars(*l);
            let out = match op {
                IrSubgroupOp::Add | IrSubgroupOp::ExclusiveAdd => {
                    let z = self.sc_zero(s);
                    let mx: Vec<Value> = (0..4)
                        .map(|i| self.b.ins().select(act[i], sc[i], z))
                        .collect();
                    if op == IrSubgroupOp::Add {
                        let mut r = mx[0];
                        for &x in &mx[1..] {
                            r = self.sc_add(s, r, x);
                        }
                        [r; 4]
                    } else {
                        let r1 = mx[0];
                        let r2 = self.sc_add(s, r1, mx[1]);
                        let r3 = self.sc_add(s, r2, mx[2]);
                        [z, r1, r2, r3]
                    }
                }
                IrSubgroupOp::Min | IrSubgroupOp::Max => {
                    let max = op == IrSubgroupOp::Max;
                    let id = self.sc_ident(s, max);
                    let mx: Vec<Value> = (0..4)
                        .map(|i| self.b.ins().select(act[i], sc[i], id))
                        .collect();
                    let mut r = mx[0];
                    for &x in &mx[1..] {
                        r = self.sc_minmax(s, max, r, x);
                    }
                    [r; 4]
                }
                IrSubgroupOp::Shuffle => {
                    let mut out = [sc[0]; 4];
                    for (i, slot) in out.iter_mut().enumerate() {
                        let src = match lane_idx {
                            Some(idx) => {
                                let li = self.b.ins().extractlane(idx, i as u8);
                                let q = self.b.ins().band_imm(li, 3);
                                let e0 = self.b.ins().icmp_imm(IntCC::Equal, q, 0);
                                let e1 = self.b.ins().icmp_imm(IntCC::Equal, q, 1);
                                let e2 = self.b.ins().icmp_imm(IntCC::Equal, q, 2);
                                let r = self.b.ins().select(e2, sc[2], sc[3]);
                                let r = self.b.ins().select(e1, sc[1], r);
                                self.b.ins().select(e0, sc[0], r)
                            }
                            None => {
                                let r = self.b.ins().select(act[2], sc[2], sc[3]);
                                let r = self.b.ins().select(act[1], sc[1], r);
                                self.b.ins().select(act[0], sc[0], r)
                            }
                        };
                        *slot = src;
                    }
                    out
                }
                IrSubgroupOp::Ballot => unreachable!(),
            };
            let mut leaf = self.leaf_from_scalars(s, out);
            if matches!(
                op,
                IrSubgroupOp::Add | IrSubgroupOp::Min | IrSubgroupOp::Max
            ) {
                leaf.k = Kind::Uni;
            }
            leaves.push(leaf);
        }
        Val {
            ty: ty.clone(),
            leaves,
        }
    }
}

#[cfg(test)]
mod tests {
    use luma_core::BindingKind;
    use luma_core::kir::{BarrierScope, BindingDecl, SharedDecl};

    use super::*;
    use crate::exec::{AlignedBuf, WorkgroupPool};

    fn bind(kind: BindingKind, ty: ScalarTy) -> BindingDecl {
        BindingDecl {
            name: "b".into(),
            kind,
            elem_ty: KirType::Scalar(ty),
            is_unsized_array: true,
            element_stride: Some(4),
            min_size: None,
        }
    }

    fn module(
        wg: u32,
        bindings: Vec<BindingDecl>,
        shared: Vec<SharedDecl>,
        extra_locals: Vec<KirType>,
        body: Vec<Stmt>,
    ) -> KernelModule {
        let mut locals = vec![
            ("gid".to_string(), KirType::vec(ScalarTy::U32, 3), false),
            ("li".to_string(), KirType::U32, false),
        ];
        locals.extend(
            extra_locals
                .into_iter()
                .enumerate()
                .map(|(i, t)| (format!("x{i}"), t, true)),
        );
        KernelModule {
            label: "t".into(),
            entry: "main".into(),
            workgroup_size: [wg, 1, 1],
            uses_exact_f32: true,
            structs: vec![],
            bindings,
            shared,
            privates: vec![],
            functions: vec![],
            entry_builtins: vec![
                (BuiltinVar::GlobalInvocationId, LocalId(0)),
                (BuiltinVar::LocalInvocationIndex, LocalId(1)),
            ],
            entry_locals: locals,
            entry_body: body,
        }
    }

    fn gid_x() -> Expr {
        Expr::Access {
            base: Box::new(Expr::Local(LocalId(0))),
            step: AccessStep::Field(0),
            ty: KirType::U32,
        }
    }

    fn li() -> Expr {
        Expr::Local(LocalId(1))
    }

    fn load(b: u32, idx: Expr, ty: ScalarTy) -> Expr {
        Expr::Load {
            binding: b,
            index: Box::new(idx),
            ty: KirType::Scalar(ty),
        }
    }

    fn bin(op: IrBinOp, l: Expr, r: Expr, ty: ScalarTy) -> Expr {
        Expr::Binary {
            op,
            lhs: Box::new(l),
            rhs: Box::new(r),
            ty: KirType::Scalar(ty),
        }
    }

    fn un(op: IrUnOp, a: Expr, ty: ScalarTy) -> Expr {
        Expr::Unary {
            op,
            arg: Box::new(a),
            ty: KirType::Scalar(ty),
        }
    }

    fn store(b: u32, idx: Expr, val: Expr) -> Stmt {
        Stmt::Store {
            binding: b,
            index: idx,
            val,
        }
    }

    /// Runs `m` over `groups` workgroups with the given zero-padded buffers; returns them afterwards.
    fn run(m: &KernelModule, bufs: &[Vec<u8>], groups: u32) -> Vec<Vec<u8>> {
        let (k, _) = compile(m, None, false).expect("compile");
        let alloc: Vec<AlignedBuf> = bufs
            .iter()
            .map(|d| {
                let b = AlignedBuf::new(d.len() as u64).unwrap();
                // SAFETY: freshly allocated buffer of at least `d.len()` bytes.
                unsafe { b.as_mut_slice(d.len()).copy_from_slice(d) };
                b
            })
            .collect();
        let ptrs: Vec<*mut u8> = alloc.iter().map(AlignedBuf::as_ptr).collect();
        let bytes: Vec<u64> = bufs.iter().map(|d| d.len() as u64).collect();
        let pool = WorkgroupPool::new(2);
        let n = groups * m.workgroup_size[0];
        pool.dispatch(
            k.exec.as_fn(),
            [n, 1, 1],
            [groups, 1, 1],
            &ptrs,
            &bytes,
            &[],
            k.plan.shared_bytes,
            k.plan.scratch_bytes,
            false,
            &k.cost,
        );
        // SAFETY: dispatch has completed; buffers are live.
        alloc
            .iter()
            .zip(bufs)
            .map(|(b, d)| unsafe { b.as_slice(d.len()).to_vec() })
            .collect()
    }

    fn f32s(v: &[f32]) -> Vec<u8> {
        v.iter().flat_map(|x| x.to_le_bytes()).collect()
    }

    fn u32s(v: &[u32]) -> Vec<u8> {
        v.iter().flat_map(|x| x.to_le_bytes()).collect()
    }

    fn rd_f32(b: &[u8]) -> Vec<f32> {
        b.as_chunks::<4>()
            .0
            .iter()
            .map(|&c| f32::from_le_bytes(c))
            .collect()
    }

    fn rd_u32(b: &[u8]) -> Vec<u32> {
        b.as_chunks::<4>()
            .0
            .iter()
            .map(|&c| u32::from_le_bytes(c))
            .collect()
    }

    /// `out[i] = f(in[i])` over one workgroup of 4.
    fn elementwise(
        in_ty: ScalarTy,
        out_ty: ScalarTy,
        f: impl Fn(Expr) -> Expr,
        input: Vec<u8>,
    ) -> Vec<u8> {
        let body = vec![
            Stmt::Let {
                local: LocalId(2),
                init: gid_x(),
            },
            store(
                1,
                Expr::Local(LocalId(2)),
                f(load(0, Expr::Local(LocalId(2)), in_ty)),
            ),
        ];
        let mut m = module(
            4,
            vec![
                bind(BindingKind::StorageRead, in_ty),
                bind(BindingKind::StorageReadWrite, out_ty),
            ],
            vec![],
            vec![],
            body,
        );
        m.entry_locals.push(("i".into(), KirType::U32, false));
        run(&m, &[input, vec![0u8; 16]], 1).remove(1)
    }

    #[test]
    fn saxpy_over_many_workgroups() {
        let body = vec![
            Stmt::Let {
                local: LocalId(2),
                init: gid_x(),
            },
            store(
                1,
                Expr::Local(LocalId(2)),
                bin(
                    IrBinOp::Add,
                    bin(
                        IrBinOp::Mul,
                        Expr::LitF32(2.0),
                        load(0, Expr::Local(LocalId(2)), ScalarTy::F32),
                        ScalarTy::F32,
                    ),
                    load(1, Expr::Local(LocalId(2)), ScalarTy::F32),
                    ScalarTy::F32,
                ),
            ),
        ];
        let mut m = module(
            64,
            vec![
                bind(BindingKind::StorageRead, ScalarTy::F32),
                bind(BindingKind::StorageReadWrite, ScalarTy::F32),
            ],
            vec![],
            vec![],
            body,
        );
        m.entry_locals.push(("i".into(), KirType::U32, false));
        let x: Vec<f32> = (0..256).map(|i| i as f32).collect();
        let y: Vec<f32> = (0..256).map(|i| 1000.0 - i as f32).collect();
        let out = rd_f32(&run(&m, &[f32s(&x), f32s(&y)], 4)[1]);
        for i in 0..256 {
            assert_eq!(out[i], 2.0 * x[i] + y[i], "lane {i}");
        }
    }

    #[test]
    fn partial_packet_does_not_write_past_the_workgroup() {
        let body = vec![
            Stmt::Let {
                local: LocalId(2),
                init: gid_x(),
            },
            store(
                0,
                Expr::Local(LocalId(2)),
                bin(
                    IrBinOp::Add,
                    Expr::Local(LocalId(2)),
                    Expr::LitU32(1),
                    ScalarTy::U32,
                ),
            ),
        ];
        let mut m = module(
            6,
            vec![bind(BindingKind::StorageReadWrite, ScalarTy::U32)],
            vec![],
            vec![],
            body,
        );
        m.entry_locals.push(("i".into(), KirType::U32, false));
        let out = rd_u32(&run(&m, &[vec![0u8; 32]], 1)[0]);
        assert_eq!(out, vec![1, 2, 3, 4, 5, 6, 0, 0]);
    }

    #[test]
    fn integer_and_saturating_ops() {
        let hi = elementwise(
            ScalarTy::U32,
            ScalarTy::U32,
            |x| {
                bin(
                    IrBinOp::MulHiU32,
                    x,
                    Expr::LitU32(0xffff_ffff),
                    ScalarTy::U32,
                )
            },
            u32s(&[0xffff_ffff, 0x8000_0000, 3, 0]),
        );
        assert_eq!(rd_u32(&hi), vec![0xffff_fffe, 0x7fff_ffff, 2, 0]);

        let sat = elementwise(
            ScalarTy::F32,
            ScalarTy::U32,
            |x| Expr::Cast {
                kind: CastKind::FloatToIntSat,
                expr: Box::new(x),
                src_ty: KirType::F32,
                dst_ty: KirType::U32,
            },
            f32s(&[f32::NAN, 1e20, -5.0, 3.7]),
        );
        assert_eq!(rd_u32(&sat), vec![0, u32::MAX, 0, 3]);

        let div = elementwise(
            ScalarTy::U32,
            ScalarTy::U32,
            |x| bin(IrBinOp::Div, Expr::LitU32(100), x, ScalarTy::U32),
            u32s(&[0, 1, 7, 100]),
        );
        assert_eq!(rd_u32(&div), vec![100, 100, 14, 1]);
    }

    #[test]
    fn exact_two_sum_is_not_reassociated() {
        // err = (a - (s - bb)) + (b - bb) with s = a + b, bb = s - a, written with exact ops.
        let two_sum = |a: Expr| {
            let b = Expr::LitF32(0.1);
            let s = un(
                IrUnOp::OpaqueGuard,
                bin(IrBinOp::AddExact, a.clone(), b.clone(), ScalarTy::F32),
                ScalarTy::F32,
            );
            let bb = un(
                IrUnOp::OpaqueGuard,
                bin(IrBinOp::SubExact, s.clone(), a.clone(), ScalarTy::F32),
                ScalarTy::F32,
            );
            let t = bin(
                IrBinOp::SubExact,
                a,
                bin(IrBinOp::SubExact, s, bb.clone(), ScalarTy::F32),
                ScalarTy::F32,
            );
            bin(
                IrBinOp::AddExact,
                t,
                bin(IrBinOp::SubExact, b, bb, ScalarTy::F32),
                ScalarTy::F32,
            )
        };
        let xs = [1.0e8f32, 3.0, -7.5e3, 0.0];
        let out = rd_f32(&elementwise(
            ScalarTy::F32,
            ScalarTy::F32,
            two_sum,
            f32s(&xs),
        ));
        for (i, &a) in xs.iter().enumerate() {
            let b = 0.1f32;
            let s = a + b;
            let bb = s - a;
            let want = (a - (s - bb)) + (b - bb);
            assert_eq!(out[i].to_bits(), want.to_bits(), "lane {i}");
        }
    }

    #[test]
    fn transcendentals_match_libm() {
        let xs = [0.5f32, -2.0, 20.0, -30.0];
        let out = rd_f32(&elementwise(
            ScalarTy::F32,
            ScalarTy::F32,
            |x| un(IrUnOp::Tanh, x, ScalarTy::F32),
            f32s(&xs),
        ));
        for (i, &x) in xs.iter().enumerate() {
            assert!(
                (out[i] - x.tanh()).abs() < 1e-6,
                "lane {i}: {} vs {}",
                out[i],
                x.tanh()
            );
        }
    }

    #[test]
    fn divergent_loop_breaks_per_lane() {
        // acc = 0; loop { if acc >= i { break } acc += 1 }  =>  out[i] = i
        let acc = Expr::Local(LocalId(3));
        let body = vec![
            Stmt::Let {
                local: LocalId(2),
                init: gid_x(),
            },
            Stmt::Var {
                local: LocalId(3),
                init: Some(Expr::LitU32(0)),
            },
            Stmt::Loop {
                body: vec![
                    Stmt::If {
                        cond: bin(
                            IrBinOp::Ge,
                            acc.clone(),
                            Expr::Local(LocalId(2)),
                            ScalarTy::Bool,
                        ),
                        then_body: vec![Stmt::Break],
                        else_body: vec![],
                    },
                    Stmt::Assign {
                        dst: LValue::Local(LocalId(3)),
                        val: bin(IrBinOp::Add, acc.clone(), Expr::LitU32(1), ScalarTy::U32),
                    },
                ],
                continuing: vec![],
                break_if: None,
            },
            store(0, Expr::Local(LocalId(2)), acc),
        ];
        let mut m = module(
            8,
            vec![bind(BindingKind::StorageReadWrite, ScalarTy::U32)],
            vec![],
            vec![],
            body,
        );
        m.entry_locals.push(("i".into(), KirType::U32, false));
        m.entry_locals.push(("acc".into(), KirType::U32, true));
        let out = rd_u32(&run(&m, &[vec![0u8; 32]], 1)[0]);
        assert_eq!(out, (0..8).collect::<Vec<u32>>());
    }

    #[test]
    fn shared_memory_barrier_and_subgroup_add() {
        let shared = vec![SharedDecl {
            name: "s".into(),
            elem_ty: KirType::F32,
            len: Some(8),
        }];
        let body = vec![
            Stmt::SharedStore {
                shared: 0,
                index: li(),
                val: load(0, li(), ScalarTy::F32),
            },
            Stmt::Barrier(BarrierScope::Workgroup),
            // out1[li] = shared[li ^ 1]
            store(
                1,
                li(),
                Expr::SharedLoad {
                    shared: 0,
                    index: Box::new(bin(IrBinOp::BitXor, li(), Expr::LitU32(1), ScalarTy::U32)),
                    ty: KirType::F32,
                },
            ),
            // out2[li] = subgroup_add(in[li])
            store(
                2,
                li(),
                Expr::Subgroup {
                    op: IrSubgroupOp::Add,
                    arg: Some(Box::new(load(0, li(), ScalarTy::F32))),
                    lane: None,
                    ty: KirType::F32,
                },
            ),
        ];
        let m = module(
            8,
            vec![
                bind(BindingKind::StorageRead, ScalarTy::F32),
                bind(BindingKind::StorageReadWrite, ScalarTy::F32),
                bind(BindingKind::StorageReadWrite, ScalarTy::F32),
            ],
            shared,
            vec![],
            body,
        );
        let xs: Vec<f32> = (1..=8).map(|i| i as f32).collect();
        let out = run(&m, &[f32s(&xs), vec![0u8; 32], vec![0u8; 32]], 1);
        let (swap, sums) = (rd_f32(&out[1]), rd_f32(&out[2]));
        for i in 0..8 {
            assert_eq!(swap[i], xs[i ^ 1], "swap {i}");
            let g = i / 4 * 4;
            assert_eq!(sums[i], xs[g..g + 4].iter().sum::<f32>(), "sum {i}");
        }
    }
}
