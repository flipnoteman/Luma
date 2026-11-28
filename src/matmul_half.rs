//! Products of `f16` / `bf16` arrays on the half-precision matrix units (W3-R7): `Array<f16>::matmul` and its siblings run the
//! backend's `mma.m16n8k16` kernel on the stored operands, accumulate in `f32` and round the result once to the array's type (in the
//! kernel's store when K is not split, else after the sum of the slices), where the widening path (`to_f32().matmul(..).to_f16()`)
//! runs the `f32` kernels.
//!
//! A product takes the unit only where the numerics allow it ([`Reduced::MATRIX_ACCUMULATE`]: `default`, `algebraic`, `fast`; not
//! `strict`), the device has it (`Caps::F16_MMA` / `BF16_MMA`), the shape is one the kernels' lanes read (the leading dimensions
//! multiples of 8) and the variant passed its reference verification (`Context::unit_verified`, the first use of a variant waits for
//! it). Anything else returns `None` and the caller runs the widening path, so the answer of a call never depends on which of the
//! two ran for a reason other than those.

use std::panic::Location;
use std::sync::Arc;

use half::{bf16, f16};
use luma_core::{CompiledKernel, Decision, DecisionSite, GemmLanes, GemmOut, GemmSpec, GemmUnit};
use luma_core::{BindingInfo, BindingKind};

use crate::array::{Array, Storage};
use crate::builtin::Builtin;
use crate::caps::Caps;
use crate::context::{Context, RawArg, Scratch};
use crate::gpu_type::{ElemLayout, GpuType};
use crate::jit::comprehension::units::variant_label;
use crate::jit::structured::contraction::{gemm_plan, GEMM_HALF_TILES_PER_SM};
use crate::kernel::KernelCore;
use crate::numerics::Reduced;
use crate::shape::Shape;

/// A half-precision element type the units take.
pub(crate) trait HalfKind: GpuType {
    const UNIT: GemmUnit;
    const OUT: GemmOut;
    const CAPS: Caps;
    /// The rounding of an `f32` array to this type (lazy).
    fn round(array: &Array<f32>) -> Array<Self>;
}

impl HalfKind for f16 {
    const UNIT: GemmUnit = GemmUnit::F16;
    const OUT: GemmOut = GemmOut::F16;
    const CAPS: Caps = Caps::F16_MMA;
    fn round(array: &Array<f32>) -> Array<f16> {
        array.to_f16()
    }
}

impl HalfKind for bf16 {
    const UNIT: GemmUnit = GemmUnit::Bf16;
    const OUT: GemmOut = GemmOut::Bf16;
    const CAPS: Caps = Caps::BF16_MMA;
    fn round(array: &Array<f32>) -> Array<bf16> {
        array.to_bf16()
    }
}

/// The smallest extent of a product that is worth its own matrix-unit launch.
const MIN_EXTENT: usize = 16;

/// `lhs * rhs` (`lhs_t` / `rhs_t`: the array holds the transpose, see `Array::matmul_nt` ...) on the matrix unit of `T`, `None` where
/// the widening path runs instead.
pub(crate) fn product<T: HalfKind>(lhs: &Array<T>, rhs: &Array<T>, lhs_t: bool, rhs_t: bool, location: &'static Location<'static>) -> Option<Array<T>> {
    let ctx = lhs.context();
    if !ctx.same_as(rhs.context()) || !ctx.numerics().reduced.contains(Reduced::MATRIX_ACCUMULATE) || !ctx.caps().contains(T::CAPS) {
        return None;
    }
    let (ld, rd) = (lhs.shape().dims(), rhs.shape().dims());
    if ld[2..].iter().chain(&rd[2..]).any(|&extent| extent != 1) {
        return None;
    }
    let (m, k) = if lhs_t { (ld[0], ld[1]) } else { (ld[1], ld[0]) };
    let (n, k2) = if rhs_t { (rd[1], rd[0]) } else { (rd[0], rd[1]) };
    if k != k2 || m.min(n).min(k) < MIN_EXTENT || [m, n, k, ld[0], rd[0]].iter().any(|&x| u32::try_from(x).is_err()) {
        return None;
    }
    // The 16-byte copies read eight elements: the leading dimensions are multiples of 8, and the extents too for the lane without
    // edge guards.
    let lanes = if [m, n, k].iter().all(|x| x % 8 == 0) {
        GemmLanes::Vec4
    } else if ld[0] % 8 == 0 && rd[0] % 8 == 0 {
        GemmLanes::Vec4Tails
    } else {
        return None;
    };
    let (m32, n32, k32) = (m as u32, n as u32, k as u32);
    let tiles = ctx.hal().gemm_tiles(T::UNIT);
    let (groups, bytes) = (ctx.max_workgroups_per_dimension(), ctx.segment_bytes());
    let plan = gemm_plan(m32, n32, k32, ctx.tuning(), GEMM_HALF_TILES_PER_SM, tiles, [lhs_t, rhs_t], GemmUnit::Tf32, groups, bytes).ok()?;
    let crate::matmul::MatmulTile::Gemm { rows, cols, .. } = plan.tile else { return None };
    // A split product keeps its slices in `f32` and rounds the sum; one slice rounds in the store.
    let out = if plan.slices == 1 { T::OUT } else { GemmOut::F32 };
    let spec = GemmSpec::new(rows, cols, lhs_t, rhs_t, T::UNIT).with_lanes(lanes).with_out(out);
    if !ctx.unit_verified(&spec, "matmul", location) {
        return None;
    }
    let kernel = kernel(ctx, &spec)?;
    let kc = plan.kc;
    let (a, a_bytes) = lhs.ready().ok()?;
    let (b, b_bytes) = rhs.ready().ok()?;
    let shape = Shape::from([n, m, 1, 1]);
    let len = shape.validate().ok()?;
    ctx.jit().note_decision(Decision::new("matmul", location, DecisionSite::Unit, format!("{} slices={}", variant_label(&spec), plan.slices)));
    let launched = (|| -> Result<Arc<Storage>, crate::array::Failure> {
        let layout = if out == GemmOut::F32 { ElemLayout::plain(4) } else { ElemLayout::HALF };
        let storage = Arc::new(Storage::new(ctx, len, layout));
        let pin = storage.pin_single()?;
        let out_bytes = storage.byte_len();
        let mut scratch = Scratch::new(ctx);
        let partial = if plan.slices > 1 { Some(scratch.take(u64::from(plan.slices) * out_bytes)?) } else { None };
        let (target, target_bytes) = match &partial {
            Some(buffer) => (buffer, u64::from(plan.slices) * out_bytes),
            None => (pin.buffer(), out_bytes),
        };
        let words = [m32, n32, k32, kc, ld[0] as u32, rd[0] as u32, 0, 0];
        ctx.launch(&kernel, plan.groups, [n32, m32, 1], &[RawArg::write(target, target_bytes), RawArg::read(a.buffer(), a_bytes), RawArg::read(b.buffer(), b_bytes), RawArg::Uniform(bytemuck::cast_slice(&words))])?;
        if let Some(partial) = &partial {
            let sum = ctx.builtin(Builtin::MatmulSum, "f32")?;
            let slices = [plan.slices];
            let sum_groups = crate::ops::fold_workgroups(ctx, (m32 * n32).div_ceil(32))?;
            ctx.launch(&sum, sum_groups, [m32 * n32, 1, 1], &[RawArg::write(pin.buffer(), out_bytes), RawArg::read(partial, target_bytes), RawArg::Uniform(bytemuck::bytes_of(&slices))])?;
        }
        drop(scratch);
        Ok(storage)
    })();
    let storage = launched.ok()?;
    Some(if out == GemmOut::F32 { T::round(&Array::<f32>::from_storage(ctx, shape, storage)) } else { Array::<T>::from_storage(ctx, shape, storage) })
}

/// The compiled kernel of `spec`, kept by the context.
fn kernel(ctx: &Context, spec: &GemmSpec) -> Option<Arc<KernelCore>> {
    ctx.keyed_core(&format!("{spec:?}"), || {
        let raw = ctx.hal().compile_gemm(spec, None)?;
        let binding = |name: &str, kind| BindingInfo { name: name.into(), kind, element_stride: None, min_size: None };
        let bindings = vec![
            binding("c", BindingKind::StorageReadWrite),
            binding("a", BindingKind::StorageRead),
            binding("b", BindingKind::StorageRead),
            BindingInfo { min_size: Some(32), ..binding("u", BindingKind::Uniform) },
        ];
        Some(CompiledKernel::new(Arc::from(variant_label(spec)), raw, bindings, 4, None, [1, 1, 1]))
    })
}
