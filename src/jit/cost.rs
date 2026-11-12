//! The cost model: what a launch, a byte of memory traffic and an arithmetic operation cost on a
//! device. Measured once per device (see [`calibrate`]) or taken from the tuning profile's fixed
//! numbers ([`CostModel::fallback`]); the fusion decisions of [`super::inline`] weigh
//! recomputation against storing with it.

use std::time::Instant;

use crate::array::Storage;
use crate::builtin::{df_library, BinOp, CmpOp, UnOp};
use crate::context::{Context, RawArg};
use crate::error::Result;
use crate::kernel::{build_core, KernelKey};
use crate::ops::launch_1d;
use crate::tuning::TuningProfile;

use super::{Edge, ElemTy, Gen, Op, OpKind, ShiftKind};

/// Throughputs of a device, measured when a context opens (see
/// [`ContextOptions::calibrate`](crate::ContextOptions::calibrate)) or taken from a profile.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CostModel {
    /// Host + GPU time of one dispatch of a tiny kernel, microseconds.
    pub launch_us: f64,
    /// Device memory bandwidth, GB/s (read + write bytes per second / 1e9).
    pub bandwidth_gbs: f64,
    /// Simple f32 operations (add or mul) per nanosecond.
    pub f32_ops_per_ns: f64,
    /// Cost of one `Df32` add or mul in f32-operation units.
    pub df32_factor: f64,
    /// Cost of one `f64` add or mul in f32-operation units; `None` without `SHADER_F64`.
    pub f64_factor: Option<f64>,
}

impl CostModel {
    /// RTX 3090 (Vulkan), measured by [`calibrate`] (median of 5 runs).
    pub(crate) const NVIDIA: CostModel = CostModel {
        launch_us: 2.9,
        bandwidth_gbs: 740.0,
        f32_ops_per_ns: 16000.0,
        df32_factor: 36.0,
        f64_factor: Some(41.0),
    };
    /// Every other GPU, measured on a RADV Raphael iGPU (median of 5 runs).
    pub(crate) const GENERIC: CostModel = CostModel {
        launch_us: 2.0,
        bandwidth_gbs: 62.0,
        f32_ops_per_ns: 370.0,
        df32_factor: 44.0,
        f64_factor: Some(11.0),
    };
    /// Mesa llvmpipe on a Ryzen 9 7950X (median of 5 runs).
    pub(crate) const CPU: CostModel = CostModel {
        launch_us: 32.0,
        bandwidth_gbs: 31.0,
        f32_ops_per_ns: 30.0,
        df32_factor: 2.3,
        f64_factor: Some(1.1),
    };
    /// The numbers tests pin so that kernel counts do not depend on the machine.
    #[cfg(test)]
    pub(crate) const TEST: CostModel = CostModel::NVIDIA;

    /// The fixed numbers of a tuning profile, used when calibration is off or fails.
    pub fn fallback(profile: TuningProfile) -> CostModel {
        match profile {
            TuningProfile::Nvidia => CostModel::NVIDIA,
            TuningProfile::Cpu => CostModel::CPU,
            _ => CostModel::GENERIC,
        }
    }
}

/// The cost of `op` per element in f32-operation units.
pub(crate) fn op_weight(op: Op, cost: &CostModel) -> f64 {
    let base = match op.kind {
        OpKind::Unary(u) => match u {
            UnOp::Neg | UnOp::Abs | UnOp::Sign | UnOp::Floor | UnOp::Ceil | UnOp::Id => 1.0,
            UnOp::Sqrt => 4.0,
            UnOp::Exp | UnOp::Ln | UnOp::Sin | UnOp::Cos => 8.0,
            UnOp::Tan | UnOp::Tanh => 16.0,
        },
        OpKind::Binary(b) => match b {
            BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Min | BinOp::Max | BinOp::Copysign => 1.0,
            BinOp::Div | BinOp::Rem => 4.0,
            BinOp::Hypot => 6.0,
            BinOp::Atan2 => 16.0,
        },
        OpKind::Powf => 16.0,
        OpKind::Compare(c) => match c {
            CmpOp::Lt | CmpOp::Le | CmpOp::Gt | CmpOp::Ge | CmpOp::Eq | CmpOp::Ne => 1.0,
        },
        OpKind::Select => 1.0,
        OpKind::Clamp => 2.0,
        OpKind::Cast(to) => {
            if to == ElemTy::Df32 || op.ty == ElemTy::Df32 {
                4.0
            } else {
                1.0
            }
        }
        OpKind::Shift(ShiftKind::Dims { .. } | ShiftKind::Linear) => 4.0,
        OpKind::Shift(ShiftKind::View { edge, .. }) => 6.0 + if edge != Edge::None { 2.0 } else { 0.0 },
        OpKind::Shift(ShiftKind::Take { .. }) => 2.0,
        OpKind::Generate(Gen::Range { .. } | Gen::Linspace) => 2.0,
        OpKind::Generate(Gen::Bits | Gen::Uniform | Gen::Normal) => 60.0,
    };
    let arithmetic = matches!(op.kind, OpKind::Unary(_) | OpKind::Binary(_) | OpKind::Clamp | OpKind::Compare(_));
    match op.ty {
        ElemTy::Df32 if arithmetic => base * cost.df32_factor,
        ElemTy::F64 if arithmetic => base * cost.f64_factor.unwrap_or(32.0),
        _ => base,
    }
}

/// Estimated nanoseconds of one kernel moving `bytes` and doing `ops` f32-operation units of work.
pub(crate) fn kernel_ns(cost: &CostModel, bytes: f64, ops: f64) -> f64 {
    cost.launch_us * 1e3 + (bytes / cost.bandwidth_gbs).max(ops / cost.f32_ops_per_ns)
}

/// Times of calibration kernels are the best of this many measurements.
const BEST_OF: usize = 3;
/// Launches recorded before one `sync` in a measurement, amortising submission and the wait.
const REPS: u32 = 4;
/// Launches of the launch-overhead measurement.
const LAUNCHES: u32 = 64;
/// Invocations per workgroup of the calibration kernels.
const WORKGROUP: u32 = 64;

/// WGSL of a calibration kernel: reads `in0`, writes `out0`, loops `p.k` times over `loop_body`
/// on a value `x` of type `ty` initialised from `init`, stores `store`.
fn kernel_source(prelude: &str, ty: &str, init: &str, body: &str, store: &str) -> String {
    format!(
        "struct Params {{ k: u32, g: u32 }}
@group(1) @binding(0) var<storage, read> in0: array<f32>;
@group(1) @binding(1) var<storage, read_write> out0: array<f32>;
@group(1) @binding(2) var<uniform> p: Params;
{prelude}
@compute @workgroup_size({WORKGROUP})
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {{
    let i = gid.x + gid.y * nwg.x * {WORKGROUP}u;
    if (i >= luma.size.x) {{ return; }}
    var x: {ty} = {init};
    for (var k = 0u; k < p.k; k++) {{ {body} }}
    out0[i] = {store};
}}
"
    )
}

/// Best wall-clock nanoseconds per launch of a kernel over `n` elements with uniform `{k, 0}`:
/// one warm-up measurement, then the best of [`BEST_OF`], each `reps` launches and one `sync`.
fn time_kernel(ctx: &Context, source: String, label: &str, n: u32, k: u32, reps: u32) -> Result<f64> {
    let core = build_core(ctx, &KernelKey::new(source, "main", None), Some(label), true)?;
    let input = Storage::new(ctx, n as usize, 4);
    let output = Storage::new(ctx, n as usize, 4);
    let (in_pin, out_pin) = (input.pin(0)?, output.pin(0)?);
    let bytes = u64::from(n) * 4;
    let words = [k, 0u32];
    let raw = [
        RawArg::read(in_pin.buffer(), bytes),
        RawArg::write(out_pin.buffer(), bytes),
        RawArg::Uniform(bytemuck::cast_slice(&words)),
    ];
    let mut best = f64::INFINITY;
    for round in 0..=BEST_OF {
        let start = Instant::now();
        for _ in 0..reps {
            launch_1d(ctx, &core, n, &raw)?;
        }
        ctx.sync()?;
        let ns = start.elapsed().as_secs_f64() * 1e9 / f64::from(reps);
        if round > 0 {
            best = best.min(ns);
        }
    }
    Ok(best)
}

/// Measures the device: bandwidth, simple f32 throughput, the cost of `Df32` and `f64`
/// arithmetic relative to it, and the cost of a launch.
pub(crate) fn calibrate(ctx: &Context) -> Result<CostModel> {
    let cpu = ctx.is_cpu();
    let copy = || kernel_source("", "f32", "in0[i]", "", "x");

    let n_copy: u32 = if cpu { 1 << 20 } else { 1 << 24 };
    let t_copy = time_kernel(ctx, copy(), "CostCopy", n_copy, 0, REPS)?;
    let bandwidth_gbs = 8.0 * f64::from(n_copy) / t_copy;

    let n_alu: u32 = if cpu { 1 << 16 } else { 1 << 20 };
    let (k_alu, k_df, k_f64) = (512u32, 64u32, 128u32);
    let alu = kernel_source("", "f32", "in0[i]", "x = x * 1.0001 + 0.5;", "x");
    let t_alu = time_kernel(ctx, alu, "CostAlu", n_alu, k_alu, REPS)?;
    let f32_ops_per_ns = 2.0 * f64::from(k_alu) * f64::from(n_alu) / t_alu;
    let per_op_f32 = t_alu / (2.0 * f64::from(k_alu) * f64::from(n_alu));

    let df = kernel_source(
        &df_library("p.g"),
        "vec2<f32>",
        "vec2<f32>(in0[i], 0.0)",
        "x = df_add(df_mul(x, vec2<f32>(1.0001, 0.0)), vec2<f32>(0.5, 0.0));",
        "x.x + x.y",
    );
    let t_df = time_kernel(ctx, df, "CostDf32", n_alu, k_df, REPS)?;
    let df32_factor = t_df / (2.0 * f64::from(k_df) * f64::from(n_alu)) / per_op_f32;

    let f64_factor = if ctx.features().contains(wgpu::Features::SHADER_F64) {
        let wide = kernel_source("", "f64", "f64(in0[i])", "x = x * f64(1.0001) + f64(0.5);", "f32(x)");
        let t_f64 = time_kernel(ctx, wide, "CostF64", n_alu, k_f64, REPS)?;
        Some(t_f64 / (2.0 * f64::from(k_f64) * f64::from(n_alu)) / per_op_f32)
    } else {
        None
    };

    let t_launch = time_kernel(ctx, copy(), "CostLaunch", 1, 0, LAUNCHES)?;
    Ok(CostModel { launch_us: t_launch / 1000.0, bandwidth_gbs, f32_ops_per_ns, df32_factor, f64_factor })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weights_follow_the_table() {
        let cost = CostModel { df32_factor: 20.0, f64_factor: Some(3.0), ..CostModel::TEST };
        let mul = |ty| Op::new(OpKind::Binary(BinOp::Mul), ty);
        assert_eq!(op_weight(mul(ElemTy::F32), &cost), 1.0);
        assert_eq!(op_weight(mul(ElemTy::Df32), &cost), 20.0);
        assert_eq!(op_weight(mul(ElemTy::F64), &cost), 3.0);
        assert_eq!(op_weight(mul(ElemTy::F64), &CostModel { f64_factor: None, ..cost }), 32.0);
        assert_eq!(op_weight(Op::new(OpKind::Unary(UnOp::Sqrt), ElemTy::F32), &cost), 4.0);
        assert_eq!(op_weight(Op::new(OpKind::Unary(UnOp::Tanh), ElemTy::Df32), &cost), 320.0);
        assert_eq!(op_weight(Op::new(OpKind::Cast(ElemTy::Df32), ElemTy::F32), &cost), 4.0);
        assert_eq!(op_weight(Op::new(OpKind::Generate(Gen::Normal), ElemTy::F32), &cost), 60.0);
    }

    #[test]
    fn kernel_time_is_launch_plus_the_slower_of_memory_and_arithmetic() {
        let cost = CostModel { launch_us: 5.0, bandwidth_gbs: 100.0, f32_ops_per_ns: 1000.0, ..CostModel::TEST };
        assert_eq!(kernel_ns(&cost, 1e6, 0.0), 5000.0 + 1e4);
        assert_eq!(kernel_ns(&cost, 1e3, 1e8), 5000.0 + 1e5);
    }
}
