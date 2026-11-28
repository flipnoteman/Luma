//! The native tensor-core products with an epilogue hook (W3-U4 part C): one kernel per (tile, lanes, orientation, epilogue
//! structure), built in the background by the backend (`Backend::compile_gemm` with the hook's KIR module) and kept in a table of
//! at most [`MAX_HOOKED`] entries per context, least recently used out.
//!
//! While a hooked kernel compiles (or if it cannot be built) the product runs as the plain native one followed by the elementwise
//! pass of the program: the same bits (the hook is the elementwise kernel's own lowering, called where the plain kernel stores), one
//! more pass. It never falls back to a kernel of another bit class.

use std::sync::Arc;

use luma_core::{BindingInfo, BindingKind, CompiledKernel, GemmSpec};

use super::epilogue::EpiProgram;
use crate::builtin::Builtin;
use crate::context::Context;
use crate::error::Result;
use crate::jit::epi_hook;
use crate::kernel::KernelCore;
use crate::matmul::MatmulTile;

/// Words of the uniform before the hook's scalars: the product's own (`gemm_copy::UNIFORM_BYTES / 4` in luma-cuda; 12 once W3-R6's
/// triangular word and the three reserved ones are in).
pub(crate) const PRODUCT_WORDS: usize = 8;

/// Hooked kernels per context.
pub(crate) const MAX_HOOKED: usize = 64;

/// What a hooked kernel is keyed on: the plain product it extends and the structure of the epilogue.
type Key = (Builtin, luma_core::HookId);

/// The hooked kernels of a context (`None`: the backend has none for that product), least recently used first.
#[derive(Default, Debug)]
pub(crate) struct Table {
    entries: Vec<(Key, Option<Arc<KernelCore>>)>,
}

impl Table {
    /// The entry of `key`, now the most recently used.
    fn touch(&mut self, key: &Key) -> Option<Option<Arc<KernelCore>>> {
        let at = self.entries.iter().position(|(known, _)| known == key)?;
        let entry = self.entries.remove(at);
        let found = entry.1.clone();
        self.entries.push(entry);
        Some(found)
    }

    /// Adds `key`, the least recently used entry out if the table is full. Returns the entry the key already had, if another thread
    /// built it meanwhile.
    fn insert(&mut self, key: Key, built: Option<Arc<KernelCore>>) -> Option<Arc<KernelCore>> {
        if let Some(found) = self.touch(&key) {
            return found;
        }
        if self.entries.len() >= MAX_HOOKED {
            self.entries.remove(0);
        }
        self.entries.push((key, built.clone()));
        built
    }
}

/// The tensor-core spec of a tile, `None` for a tile that is not a TF32 product of the backend.
fn spec_of(tile: MatmulTile) -> Option<GemmSpec> {
    match tile {
        MatmulTile::Gemm { rows, cols, lhs_t, rhs_t, exact: false } => Some(GemmSpec::new(rows, cols, lhs_t, rhs_t, false)),
        MatmulTile::GemmRagged { rows, cols, lhs_t, rhs_t, exact: false, lanes } => Some(GemmSpec::new(rows, cols, lhs_t, rhs_t, false).with_lanes(lanes)),
        _ => None,
    }
}

/// The kernel of the plain product `plain` (a `Builtin::Matmul` on a TF32 tile) with the epilogue `program`, started compiling on
/// first use. `None`: no such kernel (not a TF32 tile, the backend has no hooks, the hook does not compile on it, the program is not
/// one a hook computes). The caller checks `raw.is_ready()` before it launches: a kernel still compiling is not waited for.
pub(crate) fn kernel(ctx: &Context, plain: Builtin, program: &EpiProgram) -> Result<Option<Arc<KernelCore>>> {
    let Builtin::Matmul { tile, .. } = plain else { return Ok(None) };
    let Some(spec) = spec_of(tile) else { return Ok(None) };
    if !ctx.hal().gemm_hooks() || !epi_hook::hookable(program) {
        return Ok(None);
    }
    let id = epi_hook::hook_id(program);
    let key = (plain, id);
    if let Some(found) = crate::gpu::lock(ctx.hooked_gemms()).touch(&key) {
        return Ok(found);
    }
    let built = build(ctx, plain, spec, id, program)?;
    Ok(crate::gpu::lock(ctx.hooked_gemms()).insert(key, built))
}

fn build(ctx: &Context, plain: Builtin, spec: GemmSpec, id: luma_core::HookId, program: &EpiProgram) -> Result<Option<Arc<KernelCore>>> {
    // The shader of the plain product gives the bindings (`c`, `a`, `b`, the uniform) the hooked kernel keeps.
    let base = ctx.builtin(plain, "f32")?;
    let module = Arc::new(epi_hook::hook_module(program, ctx.tuning())?);
    let Some(raw) = ctx.hal().compile_gemm(&spec.with_hook(id), Some(&module)) else { return Ok(None) };
    let operands = program.inputs.len();
    let mut bindings: Vec<BindingInfo> = base.bindings[..3].to_vec();
    bindings.extend((0..operands).map(|k| BindingInfo { name: format!("e{k}"), kind: BindingKind::StorageRead, element_stride: Some(4), min_size: None }));
    let mut uniform = base.bindings[3].clone();
    uniform.min_size = Some(4 * (PRODUCT_WORDS + program.scalars.len()) as u32);
    bindings.push(uniform);
    let label: Arc<str> = Arc::from(format!("{} + epilogue {:08x}", base.label, id.0 as u32));
    Ok(Some(Arc::new(CompiledKernel::new(label, raw, bindings, 4 + operands, None, base.workgroup))))
}

#[cfg(test)]
mod tests {
    use std::panic::Location;
    use std::time::{Duration, Instant};

    use smallvec::smallvec;

    use super::*;
    use crate::array::Array;
    use crate::builtin::{BinOp, UnOp};
    use crate::jit::program::{Instr, Ref};
    use crate::jit::structured::contraction::{ContractionAttrs, ContractionView, Tf32Request};
    use crate::jit::structured::epilogue::{EpiInput, EpiMap};
    use crate::jit::structured::select;
    use crate::jit::structured::strategy::{Output, Subject};
    use crate::jit::{ElemTy, Op, OpKind, Scalar, Src};

    #[test]
    fn the_table_keeps_the_most_recently_used_sixty_four() {
        let mut table = Table::default();
        let key = |at: u64| (Builtin::MatmulSum, luma_core::HookId(at));
        for at in 0..MAX_HOOKED as u64 + 6 {
            table.insert(key(at), None);
            // Key 0 is used all along.
            table.touch(&key(0));
        }
        assert_eq!(table.entries.len(), MAX_HOOKED);
        assert!(table.touch(&key(0)).is_some(), "the entry in use stays");
        assert!(table.touch(&key(1)).is_none() && table.touch(&key(6)).is_none(), "the least recently used go");
        assert!(table.touch(&key(7)).is_some());
    }

    fn values(n: usize, seed: u64) -> Vec<f32> {
        let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (0..n)
            .map(|_| {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((state >> 40) as f32) / (1u64 << 24) as f32 * 2.0 - 1.0
            })
            .collect()
    }

    fn binary(op: BinOp, x: Ref, y: Ref) -> Instr {
        Instr { op: Op::new(OpKind::Binary(op), ElemTy::F32), args: [x, y, Ref::Unused] }
    }

    fn unary(op: UnOp, x: Ref) -> Instr {
        Instr { op: Op::new(OpKind::Unary(op), ElemTy::F32), args: [x, Ref::Unused, Ref::Unused] }
    }

    fn program(instrs: Vec<Instr>, inputs: &[(&Array<f32>, EpiMap)], scalars: &[f32]) -> EpiProgram {
        let last = instrs.len() as u32 - 1;
        EpiProgram {
            instrs: Arc::new(instrs),
            inputs: inputs.iter().map(|(array, map)| EpiInput { src: Src::Buffer(Arc::clone(array.storage().ok().expect("an evaluated array"))), map: *map, ty: ElemTy::F32 }).collect(),
            scalars: scalars.iter().map(|&value| Scalar::f32(value)).collect(),
            outputs: smallvec![last],
        }
    }

    /// A context on a device whose backend has hooks, or `None`.
    fn context() -> Option<Context> {
        let ctx = Context::new().ok()?;
        ctx.hal().gemm_hooks().then_some(ctx)
    }

    #[track_caller]
    fn product(ctx: &Context, lhs: &Array<f32>, rhs: &Array<f32>, epilogue: Option<&EpiProgram>) -> Output {
        let view = ContractionView { lhs, rhs, attrs: ContractionAttrs { tf32: Tf32Request::Tf32 }, location: Location::caller(), epilogue };
        select::execute(ctx, &Subject::Contraction(&view)).ok().expect("the product runs")
    }

    /// The values of `output` of an `[n, m]` array.
    fn read(ctx: &Context, output: Output, m: usize, n: usize) -> Vec<f32> {
        Array::<f32>::from_storage(ctx, [n, m, 1, 1].into(), output.into_storage()).to_vec().unwrap()
    }

    /// The hooked product: the plain one runs while the kernel compiles, so the call is repeated until the hook applies.
    #[track_caller]
    fn hooked(ctx: &Context, lhs: &Array<f32>, rhs: &Array<f32>, program: &EpiProgram, m: usize, n: usize) -> Vec<f32> {
        let start = Instant::now();
        loop {
            match product(ctx, lhs, rhs, Some(program)) {
                Output::Epilogued(storages) => return read(ctx, Output::Storage(Arc::clone(&storages[0])), m, n),
                Output::Storage(_) => {
                    assert!(start.elapsed() < Duration::from_secs(30), "the hooked kernel never became ready");
                    std::thread::sleep(Duration::from_millis(5));
                }
            }
        }
    }

    fn operands(ctx: &Context, (m, n, k): (usize, usize, usize)) -> [Array<f32>; 2] {
        [Array::from_slice_on(ctx, &values(m * k, 1), [k, m]).unwrap(), Array::from_slice_on(ctx, &values(k * n, 2), [n, k]).unwrap()]
    }

    #[test]
    fn a_hooked_relu_with_a_bias_is_the_plain_product_then_the_elementwise_pass_bit_for_bit() {
        let Some(ctx) = context() else { return };
        // Aligned, a ragged edge, and a shape that pads.
        for shape in [(128, 256, 96), (100, 130, 70), (33, 65, 17)] {
            let (m, n, _) = shape;
            let [lhs, rhs] = operands(&ctx, shape);
            let bias = Array::from_slice_on(&ctx, &values(n, 3), [n]).unwrap();
            let epi = program(vec![binary(BinOp::Add, Ref::Acc, Ref::Input(0)), binary(BinOp::Max, Ref::Tmp(0), Ref::Scalar(0))], &[(&bias, EpiMap::AlongCols)], &[0.0]);
            let plain = read(&ctx, product(&ctx, &lhs, &rhs, None), m, n);
            let bias = bias.to_vec().unwrap();
            let expected: Vec<u32> = plain.iter().enumerate().map(|(at, acc)| (acc + bias[at % n]).max(0.0).to_bits()).collect();
            let got: Vec<u32> = hooked(&ctx, &lhs, &rhs, &epi, m, n).iter().map(|v| v.to_bits()).collect();
            assert_eq!(got, expected, "{shape:?}");
        }
    }

    #[test]
    fn a_hooked_residual_and_a_scalar_and_the_identity() {
        let Some(ctx) = context() else { return };
        let shape = (96, 160, 64);
        let (m, n, _) = shape;
        let [lhs, rhs] = operands(&ctx, shape);
        let res = Array::from_slice_on(&ctx, &values(m * n, 4), [n, m]).unwrap();
        let plain = read(&ctx, product(&ctx, &lhs, &rhs, None), m, n);
        let epi = program(vec![binary(BinOp::Mul, Ref::Acc, Ref::Scalar(0)), binary(BinOp::Add, Ref::Tmp(0), Ref::Input(0))], &[(&res, EpiMap::Point)], &[0.5]);
        let res = res.to_vec().unwrap();
        let expected: Vec<u32> = plain.iter().zip(&res).map(|(acc, r)| (acc * 0.5 + r).to_bits()).collect();
        let got: Vec<u32> = hooked(&ctx, &lhs, &rhs, &epi, m, n).iter().map(|v| v.to_bits()).collect();
        // The hook boundary stops the contraction of the multiply and the add the unfused path never had.
        assert_eq!(got.len(), expected.len());
        let fused = plain.iter().zip(&res).map(|(acc, r)| acc.mul_add(0.5, *r).to_bits());
        assert!(got.iter().zip(&expected).zip(fused).all(|((g, e), f)| g == e || g == &f));
        // The identity program (`acc + 0`... as `Id`) stores the plain product.
        let id = program(vec![unary(UnOp::Id, Ref::Acc)], &[], &[]);
        let got = hooked(&ctx, &lhs, &rhs, &id, m, n);
        assert_eq!(got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), plain.iter().map(|v| v.to_bits()).collect::<Vec<_>>());
    }

    #[test]
    fn a_hook_the_expander_cannot_take_leaves_the_plain_product_running() {
        let Some(ctx) = context() else { return };
        let shape = (128, 128, 128);
        let (m, n, _) = shape;
        let [lhs, rhs] = operands(&ctx, shape);
        // `sin` reduces its argument through a table in local memory: if the hook cannot be built the plain product runs for good,
        // and if it can, it is the elementwise kernel's `sin` bit for bit.
        let epi = program(vec![unary(UnOp::Sin, Ref::Acc)], &[], &[]);
        let plain = Array::<f32>::from_storage(&ctx, [n, m, 1, 1].into(), product(&ctx, &lhs, &rhs, None).into_storage());
        let expected = plain.sin().to_vec().unwrap();
        let start = Instant::now();
        while start.elapsed() < Duration::from_secs(3) {
            match product(&ctx, &lhs, &rhs, Some(&epi)) {
                Output::Storage(_) => std::thread::sleep(Duration::from_millis(50)),
                Output::Epilogued(storages) => {
                    let got = read(&ctx, Output::Storage(Arc::clone(&storages[0])), m, n);
                    assert_eq!(got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>());
                    return;
                }
            }
        }
    }

    #[test]
    fn a_split_product_takes_no_epilogue() {
        let Some(ctx) = context() else { return };
        // A long K on a small output is cut into slices of K, which the epilogue could not follow.
        let shape = (64, 64, 32768);
        let [lhs, rhs] = operands(&ctx, shape);
        let epi = program(vec![unary(UnOp::Id, Ref::Acc)], &[], &[]);
        for _ in 0..40 {
            assert!(matches!(product(&ctx, &lhs, &rhs, Some(&epi)), Output::Storage(_)));
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    #[test]
    fn a_hooked_tanh_is_the_elementwise_kernels_tanh() {
        let Some(ctx) = context() else { return };
        let shape = (128, 128, 128);
        let (m, n, _) = shape;
        let [lhs, rhs] = operands(&ctx, shape);
        let plain = Array::<f32>::from_storage(&ctx, [n, m, 1, 1].into(), product(&ctx, &lhs, &rhs, None).into_storage());
        let expected = plain.tanh().to_vec().unwrap();
        let epi = program(vec![unary(UnOp::Tanh, Ref::Acc)], &[], &[]);
        let got = hooked(&ctx, &lhs, &rhs, &epi, m, n);
        assert_eq!(got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>());
    }
}
