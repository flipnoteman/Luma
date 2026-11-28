//! The epilogue of a product as a KIR function (W3-U4 part C): `luma_epi(acc, row, col, n, scalars...) -> f32`, the program of an
//! [`EpiProgram`] lowered by the same [`Lower`] as the elementwise kernels, so that its operations are the unfused path's, bit for
//! bit. The backend compiles function 0 of the module and calls it (inlined) from the store stage of its native product, see
//! `luma_core::HookId`.

use luma_core::HookId;
use luma_core::kir::{BuiltinVar, FnBuilder, KernelModule, KirType, ModuleBuilder};

use super::kirgen::{ACC, Frame, Lower, helper_library, read_elem, reg};
use super::program::Ref;
use super::structured::epilogue::{EpiMap, EpiProgram};
use super::{ElemTy, OpKind};
use crate::error::{LumaError, Result};
use crate::gpu::lower::finish;
use crate::tuning::Tuning;

/// The parameters of the hook function before its scalars: `acc`, `row`, `col`, `n`.
const FIXED_PARAMS: u32 = 4;

/// The identity of the hook of `program`: a hash of its structure.
pub(crate) fn hook_id(program: &EpiProgram) -> HookId {
    use std::hash::Hasher;
    let mut hasher = std::hash::DefaultHasher::new();
    program.hash_structure(&mut hasher);
    HookId(hasher.finish())
}

/// Whether `program` is one the hook can compute: a single `f32` output, `f32` inputs and scalars, and the operations whose
/// operands are `f32` values (or `bool` tests). Pure.
pub(crate) fn hookable(program: &EpiProgram) -> bool {
    let tys_ok = |ty: ElemTy| matches!(ty, ElemTy::F32 | ElemTy::Bool);
    program.outputs.len() == 1
        && program.uses_acc()
        && program.validate(0).is_ok()
        && program.inputs.iter().all(|input| input.ty == ElemTy::F32)
        && program.scalars.iter().all(|scalar| matches!(scalar.ty, ElemTy::F32 | ElemTy::U32 | ElemTy::I32))
        && program.instrs.iter().all(|instr| {
            matches!(instr.op.kind, OpKind::Unary(_) | OpKind::Binary(_) | OpKind::Compare(_) | OpKind::Classify(_) | OpKind::Powf | OpKind::Clamp | OpKind::Select)
                && (0..3).all(|n| tys_ok(instr.op.operand_ty(n)))
                && tys_ok(instr.op.out_ty())
        })
        && program.instrs.get(program.outputs[0] as usize).is_some_and(|instr| instr.op.out_ty() == ElemTy::F32)
}

/// The KIR module of the hook of `program` (which must be [`hookable`]): function 0 is `luma_epi`, the bindings `0..e` its `e` extra
/// operands (read-only, indexed by the function), and one more read-write binding and an entry body that calls the function, so
/// that the module is a whole kernel that the optimizer keeps the function of.
pub(crate) fn hook_module(program: &EpiProgram, tuning: &Tuning) -> Result<KernelModule> {
    let programs: [&[super::program::Instr]; 1] = [program.instrs.as_slice()];
    let lib = helper_library(&programs, tuning, false, false)?;
    let mut env = Lower::new(lib.clone(), &[]);
    let mut mb = ModuleBuilder::new("luma_epi", "main", [1, 1, 1]);
    for (k, input) in program.inputs.iter().enumerate() {
        env.ins.push(Some(mb.add_storage_read(&format!("e{k}"), read_elem(input.ty))));
        env.segs.push(Vec::new());
    }
    let sink = mb.add_storage_read_write("sink", KirType::F32);
    let function = add_function(&mut mb, &mut env, program);
    {
        let mut fb = mb.entry_builder();
        let zero_f = fb.lit_f32(0.0);
        let zero = fb.lit_u32(0);
        let args: Vec<_> = [zero_f, zero, zero, zero].into_iter().chain(program.scalars.iter().map(|scalar| fb.zero(reg(scalar.ty)))).collect();
        let value = fb.call(function, &args, KirType::F32);
        fb.store(sink, zero, value);
        fb.finish();
    }
    if let Some(message) = env.error() {
        return Err(LumaError::Gpu(format!("epilogue hook: {message}")));
    }
    Ok(finish(mb, lib.as_deref()))
}

/// Invocations of the workgroup of [`tile_module`]: a tile row is covered by this many columns at a time.
const TILE_LANES: u32 = 32;

/// The epilogue of `program` over one tile of the output as a whole kernel (a backend's native product runs it after each tile's sum,
/// see `luma_core::Backend::compile_matmul_epilogue`): bindings `e0..` (the operands, read-only), `c` (the output, read-write) and the
/// uniform `p`, words `rows, cols, row, col, n`, three of padding and the scalars. One workgroup applies [`luma_epi`] in place to the
/// `rows x cols` tile at (`row`, `col`) of the `n`-wide output `c`, so its elements are the unfused pass's, bit for bit.
pub(crate) fn tile_module(program: &EpiProgram, tuning: &Tuning) -> Result<KernelModule> {
    let programs: [&[super::program::Instr]; 1] = [program.instrs.as_slice()];
    let lib = helper_library(&programs, tuning, false, false)?;
    let mut env = Lower::new(lib.clone(), &[]);
    let mut mb = ModuleBuilder::new("luma_epi_tile", "main", [TILE_LANES, 1, 1]);
    for (k, input) in program.inputs.iter().enumerate() {
        env.ins.push(Some(mb.add_storage_read(&format!("e{k}"), read_elem(input.ty))));
        env.segs.push(Vec::new());
    }
    let c = mb.add_storage_read_write("c", KirType::F32);
    let mut fields: Vec<(String, KirType)> = ["rows", "cols", "row", "col", "n", "pad0", "pad1", "pad2"].iter().map(|name| ((*name).to_owned(), KirType::U32)).collect();
    fields.extend(program.scalars.iter().enumerate().map(|(k, scalar)| (format!("s{k}"), reg(scalar.ty))));
    let layout: Vec<(&str, KirType)> = fields.iter().map(|(name, ty)| (name.as_str(), ty.clone())).collect();
    let params = mb.add_struct_layout("Params", &layout);
    let p = mb.add_uniform("p", KirType::Struct(params));
    let function = add_function(&mut mb, &mut env, program);
    {
        let mut fb = mb.entry_builder();
        let word = |fb: &FnBuilder<'_>, at: usize| fb.uniform_field(p, at as u32, fields[at].1.clone());
        let lid = fb.builtin(BuiltinVar::LocalInvocationId);
        let lane = fb.field(lid, 0, KirType::U32);
        let [rows, cols, row0, col0, n] = [0, 1, 2, 3, 4].map(|at| word(&fb, at));
        let scalars: Vec<_> = (0..program.scalars.len()).map(|k| word(&fb, 8 + k)).collect();
        let (zero, one, lanes) = (fb.lit_u32(0), fb.lit_u32(1), fb.lit_u32(TILE_LANES));
        fb.for_range("r", zero, rows, one, |b, r| {
            let row = b.let_val("row", row0 + r);
            b.for_range("q", lane, cols, lanes, |b, q| {
                let col = b.let_val("col", col0 + q);
                let at = b.let_val("at", row * n + col);
                let acc = b.load(c, at);
                let args: Vec<_> = [acc, row, col, n].into_iter().chain(scalars.iter().copied()).collect();
                let value = b.call(function, &args, KirType::F32);
                b.store(c, at, value);
            });
        });
        fb.finish();
    }
    if let Some(message) = env.error() {
        return Err(LumaError::Gpu(format!("epilogue tile: {message}")));
    }
    Ok(finish(mb, lib.as_deref()))
}

/// Adds `luma_epi(acc, row, col, n, scalars...) -> f32`, the program lowered, to `mb` whose operands are bindings `0..e` (declared in
/// `env`).
fn add_function(mb: &mut ModuleBuilder, env: &mut Lower, program: &EpiProgram) -> luma_core::kir::FnId {
    env.scalar_params = Some(FIXED_PARAMS);
    let mut params: Vec<(String, KirType)> =
        vec![("acc".into(), KirType::F32), ("row".into(), KirType::U32), ("col".into(), KirType::U32), ("n".into(), KirType::U32)];
    params.extend(program.scalars.iter().enumerate().map(|(k, scalar)| (format!("s{k}"), reg(scalar.ty))));
    let params: Vec<(&str, KirType)> = params.iter().map(|(name, ty)| (name.as_str(), ty.clone())).collect();
    mb.add_function("luma_epi", &params, Some(KirType::F32), |fb| {
        let (acc, row, col, n) = (fb.param(0), fb.param(1), fb.param(2), fb.param(3));
        let mut frame = Frame::with_len(program.instrs.len());
        frame.a.insert((ACC, 0), acc);
        for (k, input) in program.inputs.iter().enumerate() {
            // The element of the operand at the output (`row`, `col`) of an `[n, m]` output.
            let at = match input.map {
                EpiMap::Point => row * n + col,
                EpiMap::AlongRows => row,
                EpiMap::AlongCols => col,
                EpiMap::Scalar => fb.lit_u32(0),
            };
            if program.instrs.iter().any(|instr| instr.args.contains(&Ref::Input(k as u32))) {
                let value = env.input_load(fb, input.ty, k as u32, at, false);
                let value = fb.let_val(&format!("a{k}"), value);
                frame.a.insert((k as u32, 0), value);
            }
        }
        let zero = fb.lit_u32(0);
        for (k, instr) in program.instrs.iter().enumerate() {
            env.bind_instr(fb, &mut frame, k, instr, zero, false, false, None);
        }
        let result = frame.t(program.outputs[0] as usize);
        let result = result.unwrap_or_else(|| fb.lit_f32(0.0));
        fb.ret(Some(result));
    })
}
