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
}

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
