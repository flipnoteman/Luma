//! Zero-relocation runtime ABI and barrier-fission plan structures for `luma-cpu`.

use std::mem::offset_of;

use luma_core::kir::{Expr, Stmt};

/// Entry signature of every Cranelift-compiled workgroup function.
pub type WorkgroupFn = unsafe extern "C" fn(*const WorkgroupCtx);

extern "C" fn math_sinf(x: f32) -> f32 {
    x.sin()
}
extern "C" fn math_cosf(x: f32) -> f32 {
    x.cos()
}
extern "C" fn math_tanf(x: f32) -> f32 {
    x.tan()
}
extern "C" fn math_asinf(x: f32) -> f32 {
    x.asin()
}
extern "C" fn math_acosf(x: f32) -> f32 {
    x.acos()
}
extern "C" fn math_atanf(x: f32) -> f32 {
    x.atan()
}
extern "C" fn math_tanhf(x: f32) -> f32 {
    if x.abs() > 15.0 { x.signum() } else { x.tanh() }
}
extern "C" fn math_expf(x: f32) -> f32 {
    x.exp()
}
extern "C" fn math_exp2f(x: f32) -> f32 {
    x.exp2()
}
extern "C" fn math_logf(x: f32) -> f32 {
    x.ln()
}
extern "C" fn math_log2f(x: f32) -> f32 {
    x.log2()
}
extern "C" fn math_powf(x: f32, y: f32) -> f32 {
    x.powf(y)
}
extern "C" fn math_atan2f(y: f32, x: f32) -> f32 {
    y.atan2(x)
}

extern "C" fn math_sin(x: f64) -> f64 {
    x.sin()
}
extern "C" fn math_cos(x: f64) -> f64 {
    x.cos()
}
extern "C" fn math_tan(x: f64) -> f64 {
    x.tan()
}
extern "C" fn math_asin(x: f64) -> f64 {
    x.asin()
}
extern "C" fn math_acos(x: f64) -> f64 {
    x.acos()
}
extern "C" fn math_atan(x: f64) -> f64 {
    x.atan()
}
extern "C" fn math_tanh(x: f64) -> f64 {
    x.tanh()
}
extern "C" fn math_exp(x: f64) -> f64 {
    x.exp()
}
extern "C" fn math_exp2(x: f64) -> f64 {
    x.exp2()
}
extern "C" fn math_log(x: f64) -> f64 {
    x.ln()
}
extern "C" fn math_log2(x: f64) -> f64 {
    x.log2()
}
extern "C" fn math_pow(x: f64, y: f64) -> f64 {
    x.powf(y)
}
extern "C" fn math_atan2(y: f64, x: f64) -> f64 {
    y.atan2(x)
}

extern "C" fn math_sinhf(x: f32) -> f32 {
    x.sinh()
}
extern "C" fn math_coshf(x: f32) -> f32 {
    x.cosh()
}
extern "C" fn math_asinhf(x: f32) -> f32 {
    x.asinh()
}
extern "C" fn math_acoshf(x: f32) -> f32 {
    x.acosh()
}
extern "C" fn math_atanhf(x: f32) -> f32 {
    x.atanh()
}
extern "C" fn math_log10f(x: f32) -> f32 {
    x.log10()
}
extern "C" fn math_cbrtf(x: f32) -> f32 {
    x.cbrt()
}
extern "C" fn math_erff(x: f32) -> f32 {
    libm::erff(x)
}
extern "C" fn math_erfcf(x: f32) -> f32 {
    libm::erfcf(x)
}
extern "C" fn math_lgammaf(x: f32) -> f32 {
    libm::lgammaf(x)
}
extern "C" fn math_tgammaf(x: f32) -> f32 {
    libm::tgammaf(x)
}
extern "C" fn math_log1pf(x: f32) -> f32 {
    x.ln_1p()
}
extern "C" fn math_expm1f(x: f32) -> f32 {
    x.exp_m1()
}
extern "C" fn math_sinh(x: f64) -> f64 {
    x.sinh()
}
extern "C" fn math_cosh(x: f64) -> f64 {
    x.cosh()
}
extern "C" fn math_asinh(x: f64) -> f64 {
    x.asinh()
}
extern "C" fn math_acosh(x: f64) -> f64 {
    x.acosh()
}
extern "C" fn math_atanh(x: f64) -> f64 {
    x.atanh()
}
extern "C" fn math_log10(x: f64) -> f64 {
    x.log10()
}
extern "C" fn math_cbrt(x: f64) -> f64 {
    x.cbrt()
}
extern "C" fn math_erf(x: f64) -> f64 {
    libm::erf(x)
}
extern "C" fn math_erfc(x: f64) -> f64 {
    libm::erfc(x)
}
extern "C" fn math_lgamma(x: f64) -> f64 {
    libm::lgamma(x)
}
extern "C" fn math_tgamma(x: f64) -> f64 {
    libm::tgamma(x)
}
extern "C" fn math_log1p(x: f64) -> f64 {
    x.ln_1p()
}
extern "C" fn math_expm1(x: f64) -> f64 {
    x.exp_m1()
}

/// Function pointer table for transcendental math routines called from position-independent JIT
/// code without ELF/Mach-O relocations.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct CpuMathTable {
    pub sinf: extern "C" fn(f32) -> f32,
    pub cosf: extern "C" fn(f32) -> f32,
    pub tanf: extern "C" fn(f32) -> f32,
    pub asinf: extern "C" fn(f32) -> f32,
    pub acosf: extern "C" fn(f32) -> f32,
    pub atanf: extern "C" fn(f32) -> f32,
    pub tanhf: extern "C" fn(f32) -> f32,
    pub expf: extern "C" fn(f32) -> f32,
    pub exp2f: extern "C" fn(f32) -> f32,
    pub logf: extern "C" fn(f32) -> f32,
    pub log2f: extern "C" fn(f32) -> f32,
    pub powf: extern "C" fn(f32, f32) -> f32,
    pub atan2f: extern "C" fn(f32, f32) -> f32,
    pub sin: extern "C" fn(f64) -> f64,
    pub cos: extern "C" fn(f64) -> f64,
    pub tan: extern "C" fn(f64) -> f64,
    pub asin: extern "C" fn(f64) -> f64,
    pub acos: extern "C" fn(f64) -> f64,
    pub atan: extern "C" fn(f64) -> f64,
    pub tanh: extern "C" fn(f64) -> f64,
    pub exp: extern "C" fn(f64) -> f64,
    pub exp2: extern "C" fn(f64) -> f64,
    pub log: extern "C" fn(f64) -> f64,
    pub log2: extern "C" fn(f64) -> f64,
    pub pow: extern "C" fn(f64, f64) -> f64,
    pub atan2: extern "C" fn(f64, f64) -> f64,
    /// Four-lane `f32` routines taking and returning one packet (`xmm0`), see [`crate::vecmath`].
    #[cfg(all(target_arch = "x86_64", unix))]
    pub packet: PacketMath,
    // Appended after `packet`: compiled kernels cached on disk keep the offsets of everything above.
    pub sinhf: extern "C" fn(f32) -> f32,
    pub coshf: extern "C" fn(f32) -> f32,
    pub asinhf: extern "C" fn(f32) -> f32,
    pub acoshf: extern "C" fn(f32) -> f32,
    pub atanhf: extern "C" fn(f32) -> f32,
    pub log10f: extern "C" fn(f32) -> f32,
    pub cbrtf: extern "C" fn(f32) -> f32,
    pub erff: extern "C" fn(f32) -> f32,
    pub erfcf: extern "C" fn(f32) -> f32,
    pub lgammaf: extern "C" fn(f32) -> f32,
    pub tgammaf: extern "C" fn(f32) -> f32,
    pub log1pf: extern "C" fn(f32) -> f32,
    pub expm1f: extern "C" fn(f32) -> f32,
    pub sinh: extern "C" fn(f64) -> f64,
    pub cosh: extern "C" fn(f64) -> f64,
    pub asinh: extern "C" fn(f64) -> f64,
    pub acosh: extern "C" fn(f64) -> f64,
    pub atanh: extern "C" fn(f64) -> f64,
    pub log10: extern "C" fn(f64) -> f64,
    pub cbrt: extern "C" fn(f64) -> f64,
    pub erf: extern "C" fn(f64) -> f64,
    pub erfc: extern "C" fn(f64) -> f64,
    pub lgamma: extern "C" fn(f64) -> f64,
    pub tgamma: extern "C" fn(f64) -> f64,
    pub log1p: extern "C" fn(f64) -> f64,
    pub expm1: extern "C" fn(f64) -> f64,
}

/// One `extern "C" fn(__m128) -> __m128` per vectorised `f32` function.
#[cfg(all(target_arch = "x86_64", unix))]
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct PacketMath {
    pub sinf: PacketFn,
    pub cosf: PacketFn,
    pub tanhf: PacketFn,
    pub expf: PacketFn,
    pub exp2f: PacketFn,
    pub logf: PacketFn,
    pub log2f: PacketFn,
}

#[cfg(all(target_arch = "x86_64", unix))]
// `__m128` in `xmm0` (System V) is the calling convention the JIT emits for packet calls.
#[allow(improper_ctypes_definitions)]
pub type PacketFn = extern "C" fn(std::arch::x86_64::__m128) -> std::arch::x86_64::__m128;

impl CpuMathTable {
    pub const OFFSET_SINF: i32 = offset_of!(Self, sinf) as i32;
    pub const OFFSET_COSF: i32 = offset_of!(Self, cosf) as i32;
    pub const OFFSET_TANF: i32 = offset_of!(Self, tanf) as i32;
    pub const OFFSET_ASINF: i32 = offset_of!(Self, asinf) as i32;
    pub const OFFSET_ACOSF: i32 = offset_of!(Self, acosf) as i32;
    pub const OFFSET_ATANF: i32 = offset_of!(Self, atanf) as i32;
    pub const OFFSET_TANHF: i32 = offset_of!(Self, tanhf) as i32;
    pub const OFFSET_EXPF: i32 = offset_of!(Self, expf) as i32;
    pub const OFFSET_EXP2F: i32 = offset_of!(Self, exp2f) as i32;
    pub const OFFSET_LOGF: i32 = offset_of!(Self, logf) as i32;
    pub const OFFSET_LOG2F: i32 = offset_of!(Self, log2f) as i32;
    pub const OFFSET_POWF: i32 = offset_of!(Self, powf) as i32;
    pub const OFFSET_ATAN2F: i32 = offset_of!(Self, atan2f) as i32;
    pub const OFFSET_SIN: i32 = offset_of!(Self, sin) as i32;
    pub const OFFSET_COS: i32 = offset_of!(Self, cos) as i32;
    pub const OFFSET_TAN: i32 = offset_of!(Self, tan) as i32;
    pub const OFFSET_ASIN: i32 = offset_of!(Self, asin) as i32;
    pub const OFFSET_ACOS: i32 = offset_of!(Self, acos) as i32;
    pub const OFFSET_ATAN: i32 = offset_of!(Self, atan) as i32;
    pub const OFFSET_TANH: i32 = offset_of!(Self, tanh) as i32;
    pub const OFFSET_EXP: i32 = offset_of!(Self, exp) as i32;
    pub const OFFSET_EXP2: i32 = offset_of!(Self, exp2) as i32;
    pub const OFFSET_LOG: i32 = offset_of!(Self, log) as i32;
    pub const OFFSET_LOG2: i32 = offset_of!(Self, log2) as i32;
    pub const OFFSET_POW: i32 = offset_of!(Self, pow) as i32;
    pub const OFFSET_ATAN2: i32 = offset_of!(Self, atan2) as i32;
    pub const OFFSET_SINHF: i32 = offset_of!(Self, sinhf) as i32;
    pub const OFFSET_SINH: i32 = offset_of!(Self, sinh) as i32;
    pub const OFFSET_COSHF: i32 = offset_of!(Self, coshf) as i32;
    pub const OFFSET_COSH: i32 = offset_of!(Self, cosh) as i32;
    pub const OFFSET_ASINHF: i32 = offset_of!(Self, asinhf) as i32;
    pub const OFFSET_ASINH: i32 = offset_of!(Self, asinh) as i32;
    pub const OFFSET_ACOSHF: i32 = offset_of!(Self, acoshf) as i32;
    pub const OFFSET_ACOSH: i32 = offset_of!(Self, acosh) as i32;
    pub const OFFSET_ATANHF: i32 = offset_of!(Self, atanhf) as i32;
    pub const OFFSET_ATANH: i32 = offset_of!(Self, atanh) as i32;
    pub const OFFSET_LOG10F: i32 = offset_of!(Self, log10f) as i32;
    pub const OFFSET_LOG10: i32 = offset_of!(Self, log10) as i32;
    pub const OFFSET_CBRTF: i32 = offset_of!(Self, cbrtf) as i32;
    pub const OFFSET_CBRT: i32 = offset_of!(Self, cbrt) as i32;
    pub const OFFSET_ERFF: i32 = offset_of!(Self, erff) as i32;
    pub const OFFSET_ERF: i32 = offset_of!(Self, erf) as i32;
    pub const OFFSET_ERFCF: i32 = offset_of!(Self, erfcf) as i32;
    pub const OFFSET_ERFC: i32 = offset_of!(Self, erfc) as i32;
    pub const OFFSET_LGAMMAF: i32 = offset_of!(Self, lgammaf) as i32;
    pub const OFFSET_LGAMMA: i32 = offset_of!(Self, lgamma) as i32;
    pub const OFFSET_TGAMMAF: i32 = offset_of!(Self, tgammaf) as i32;
    pub const OFFSET_TGAMMA: i32 = offset_of!(Self, tgamma) as i32;
    pub const OFFSET_LOG1PF: i32 = offset_of!(Self, log1pf) as i32;
    pub const OFFSET_LOG1P: i32 = offset_of!(Self, log1p) as i32;
    pub const OFFSET_EXPM1F: i32 = offset_of!(Self, expm1f) as i32;
    pub const OFFSET_EXPM1: i32 = offset_of!(Self, expm1) as i32;

    /// Offset of the packet (four-lane) `f32` routine replacing the scalar one at `scalar_off32`,
    /// if there is one on this target.
    #[cfg(all(target_arch = "x86_64", unix))]
    pub fn packet_offset(scalar_off32: i32) -> Option<i32> {
        let base = offset_of!(Self, packet);
        let f = |o: usize| Some((base + o) as i32);
        match scalar_off32 {
            Self::OFFSET_SINF => f(offset_of!(PacketMath, sinf)),
            Self::OFFSET_COSF => f(offset_of!(PacketMath, cosf)),
            Self::OFFSET_TANHF => f(offset_of!(PacketMath, tanhf)),
            Self::OFFSET_EXPF => f(offset_of!(PacketMath, expf)),
            Self::OFFSET_EXP2F => f(offset_of!(PacketMath, exp2f)),
            Self::OFFSET_LOGF => f(offset_of!(PacketMath, logf)),
            Self::OFFSET_LOG2F => f(offset_of!(PacketMath, log2f)),
            _ => None,
        }
    }

    /// No packet routines on this target: every call is one scalar call per lane.
    #[cfg(not(all(target_arch = "x86_64", unix)))]
    pub fn packet_offset(_scalar_off32: i32) -> Option<i32> {
        None
    }
}

/// Static math dispatch table shared across all workgroup invocations.
pub static MATH_TABLE: CpuMathTable = CpuMathTable {
    sinf: math_sinf,
    cosf: math_cosf,
    tanf: math_tanf,
    asinf: math_asinf,
    acosf: math_acosf,
    atanf: math_atanf,
    tanhf: math_tanhf,
    expf: math_expf,
    exp2f: math_exp2f,
    logf: math_logf,
    log2f: math_log2f,
    powf: math_powf,
    atan2f: math_atan2f,
    sin: math_sin,
    cos: math_cos,
    tan: math_tan,
    asin: math_asin,
    acos: math_acos,
    atan: math_atan,
    tanh: math_tanh,
    exp: math_exp,
    exp2: math_exp2,
    log: math_log,
    log2: math_log2,
    pow: math_pow,
    atan2: math_atan2,
    #[cfg(all(target_arch = "x86_64", unix))]
    packet: PacketMath {
        sinf: crate::vecmath::sinf4,
        cosf: crate::vecmath::cosf4,
        tanhf: crate::vecmath::tanhf4,
        expf: crate::vecmath::expf4,
        exp2f: crate::vecmath::exp2f4,
        logf: crate::vecmath::logf4,
        log2f: crate::vecmath::log2f4,
    },
    sinhf: math_sinhf,
    sinh: math_sinh,
    coshf: math_coshf,
    cosh: math_cosh,
    asinhf: math_asinhf,
    asinh: math_asinh,
    acoshf: math_acoshf,
    acosh: math_acosh,
    atanhf: math_atanhf,
    atanh: math_atanh,
    log10f: math_log10f,
    log10: math_log10,
    cbrtf: math_cbrtf,
    cbrt: math_cbrt,
    erff: math_erff,
    erf: math_erf,
    erfcf: math_erfcf,
    erfc: math_erfc,
    lgammaf: math_lgammaf,
    lgamma: math_lgamma,
    tgammaf: math_tgammaf,
    tgamma: math_tgamma,
    log1pf: math_log1pf,
    log1p: math_log1p,
    expm1f: math_expm1f,
    expm1: math_expm1,
};

/// Context pointer passed as the sole argument (`r0`/`rdi`/`rcx`) to every compiled workgroup.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct WorkgroupCtx {
    /// Logical grid size (`luma.size.xyz`, padded to 4 `u32`s).
    pub grid_size: [u32; 4],
    /// Dispatched workgroup counts (`num_workgroups.xyz`, padded to 4 `u32`s).
    pub num_workgroups: [u32; 4],
    /// Current workgroup coordinates (`workgroup_id.xyz`, padded to 4 `u32`s).
    pub workgroup_id: [u32; 4],
    /// Base pointers of active bindings, indexed by active binding rank.
    pub bindings: *const *mut u8,
    /// Byte length of each active binding, indexed by active binding rank (bounds guard for speculative accesses).
    pub binding_bytes: *const u64,
    /// Element counts for `Expr::ArrayLength`, in `CpuPlan::array_lengths` order.
    pub lengths: *const u32,
    /// Thread-local workgroup shared memory buffer (`var<workgroup>`).
    pub shared: *mut u8,
    /// Thread-local per-workgroup scratch buffer for cross-barrier locals and `var<private>`.
    pub scratch: *mut u8,
    /// Pointer to [`MATH_TABLE`].
    pub math: *const CpuMathTable,
    /// The kernel's own data for a native kernel (see `crate::solve`); null for compiled code.
    pub native: *const u8,
}

impl WorkgroupCtx {
    pub const OFFSET_GRID_SIZE: i32 = offset_of!(Self, grid_size) as i32;
    pub const OFFSET_NUM_WORKGROUPS: i32 = offset_of!(Self, num_workgroups) as i32;
    pub const OFFSET_WORKGROUP_ID: i32 = offset_of!(Self, workgroup_id) as i32;
    pub const OFFSET_BINDINGS: i32 = offset_of!(Self, bindings) as i32;
    pub const OFFSET_BINDING_BYTES: i32 = offset_of!(Self, binding_bytes) as i32;
    pub const OFFSET_LENGTHS: i32 = offset_of!(Self, lengths) as i32;
    pub const OFFSET_SHARED: i32 = offset_of!(Self, shared) as i32;
    pub const OFFSET_SCRATCH: i32 = offset_of!(Self, scratch) as i32;
    pub const OFFSET_MATH: i32 = offset_of!(Self, math) as i32;
}

/// A barrier-split statement tree for a workgroup function.
#[derive(Clone, Debug, PartialEq)]
pub enum PhaseStmt {
    /// A maximal barrier-free sequence of statements executed across all invocations/packets of the workgroup.
    Segment {
        stmts: Vec<Stmt>,
        may_break_outer: bool,
    },
    /// Workgroup-uniform conditional containing a barrier in at least one arm.
    If {
        cond: Expr,
        then_body: Vec<PhaseStmt>,
        else_body: Vec<PhaseStmt>,
    },
    /// Workgroup-uniform loop containing at least one barrier.
    Loop {
        body: Vec<PhaseStmt>,
        continuing: Vec<PhaseStmt>,
        break_if: Option<Expr>,
    },
}

/// Lowering and memory-layout plan for a [`luma_core::kir::KernelModule`] on CPU.
#[derive(Clone, Debug)]
pub struct CpuPlan {
    /// Declared binding numbers that are active (`used`), in ascending order.
    pub active_bindings: Vec<u32>,
    /// Maps declared binding index -> active rank (`0..active_bindings.len()`).
    pub binding_ranks: Vec<Option<u32>>,
    /// `(binding, element_stride)` for every binding queried by `Expr::ArrayLength`.
    pub array_lengths: Vec<(u32, u32)>,
    /// Workgroup dimensions `[wx, wy, wz]`.
    pub workgroup_size: [u32; 3],
    /// Total invocations per workgroup (`wx * wy * wz`).
    pub invocations: u32,
    /// SIMD packet width (`4` when `invocations` is a multiple of 4, else `1`).
    pub simd_width: u32,
    /// Byte offset of each `SharedDecl` inside `WorkgroupCtx::shared`.
    pub shared_offsets: Vec<u32>,
    /// Total bytes required in `WorkgroupCtx::shared` (rounded up to 16).
    pub shared_bytes: u32,
    /// Per-`entry_locals` byte offset in `WorkgroupCtx::scratch` if the local crosses a barrier
    /// segment (`None` when the local lives purely in Cranelift SSA/SIMD registers within one segment).
    pub local_scratch_offsets: Vec<Option<u32>>,
    /// Per-`privates` byte offset in `WorkgroupCtx::scratch` (each slot holds `invocations` copies).
    pub priv_scratch_offsets: Vec<u32>,
    /// Total bytes required in `WorkgroupCtx::scratch` (rounded up to 16).
    pub scratch_bytes: u32,
    /// Whether `WorkgroupCtx::shared` must be zeroed at the start of each workgroup.
    pub initialize_shared: bool,
    /// Barrier-split phases of `entry_body`.
    pub phases: Vec<PhaseStmt>,
}
