//! Kernel parameters per GPU family.
//!
//! Workgroup sizes and tile shapes that suit one GPU can cost another a lot (Apple GPUs, for
//! example, lose occupancy sharply above about 8 KiB of workgroup memory per workgroup). A
//! [`Tuning`] is chosen once per context from the adapter the device was opened on; the kernel
//! generator reads it instead of fixed constants. `LUMA_TUNING=apple`, `LUMA_TUNING=nvidia`,
//! `LUMA_TUNING=cpu` or `LUMA_TUNING=generic` in the environment overrides the choice for new
//! contexts.

/// Environment variable forcing a profile.
const TUNING_ENV: &str = "LUMA_TUNING";
/// `LUMA_SUBGROUPS=off` keeps the built-in kernels off subgroup operations.
const SUBGROUPS_ENV: &str = "LUMA_SUBGROUPS";
/// `LUMA_FUSION_MAX_LIVE=n` overrides [`Tuning::max_live`] (clamped to `8..=4096`).
const MAX_LIVE_ENV: &str = "LUMA_FUSION_MAX_LIVE";
/// `LUMA_FUSION_MAX_INLINED=n` overrides [`Tuning::max_inlined`] (clamped to `16..=16384`).
const MAX_INLINED_ENV: &str = "LUMA_FUSION_MAX_INLINED";
/// `LUMA_STAGING=off` keeps sub-programs read through views out of workgroup memory (see `jit::staging`).
const STAGING_ENV: &str = "LUMA_STAGING";
/// `LUMA_COARSEN=off|auto|<n>` sets [`Tuning::coarsen`] (see `jit::coarsen`).
const COARSEN_ENV: &str = "LUMA_COARSEN";
/// `LUMA_TRIDIAGONAL=thomas|cyclic|auto` picks the line solver of `solve_tridiagonal` (see [`LineSolver`]).
const LINE_SOLVER_ENV: &str = "LUMA_TRIDIAGONAL";
/// `LUMA_SINGLE_PASS_REDUCE=off|on` overrides [`Tuning::single_pass_reduce`] (`on` only takes
/// effect on the CUDA and CPU backends, whose atomics and fences order memory across workgroups).
const SINGLE_PASS_ENV: &str = "LUMA_SINGLE_PASS_REDUCE";
/// `LUMA_LAZY=never|site|always` sets [`Tuning::lazy`] (a dev switch of W2-LAZY, removed before 0.6.0).
const LAZY_ENV: &str = "LUMA_LAZY";
/// `LUMA_LAZY_PROBE=<n>` sets [`Tuning::lazy_probe_every`] (0, or a power of two up to 256: other values are rounded down to one).
const LAZY_PROBE_ENV: &str = "LUMA_LAZY_PROBE";
/// `LUMA_MULTI_REDUCE=off|n` overrides [`Tuning::reduce_multi`] (`off` or `1` keeps every reduction
/// in a kernel of its own; clamped to the most accumulators a kernel has).
const MULTI_REDUCE_ENV: &str = "LUMA_MULTI_REDUCE";
/// `LUMA_MULTI_REDUCE_AFTER=n` overrides [`Tuning::reduce_multi_after`].
const MULTI_REDUCE_AFTER_ENV: &str = "LUMA_MULTI_REDUCE_AFTER";
/// `LUMA_TELEMETRY=off|<n>` sets [`Tuning::telemetry_every`]: one execution of a strategy in `n` is timed on the device (`off` or 0
/// switches the sampling off).
const TELEMETRY_ENV: &str = "LUMA_TELEMETRY";
/// `LUMA_TELEMETRY_MIN=<n>` sets [`Tuning::telemetry_min`].
const TELEMETRY_MIN_ENV: &str = "LUMA_TELEMETRY_MIN";
/// `LUMA_EXPLORE=<m>` sets [`Tuning::explore_every`] (0 or `off`: never).
const EXPLORE_ENV: &str = "LUMA_EXPLORE";
/// `LUMA_VARIANT_BUDGET=<n>` sets [`Tuning::variant_budget`].
const VARIANT_BUDGET_ENV: &str = "LUMA_VARIANT_BUDGET";
/// `LUMA_LOOKBACK_SCAN=off|on` overrides [`Tuning::lookback_scan`] (`on` only takes effect on the
/// CUDA and CPU backends, for the same reason as `LUMA_SINGLE_PASS_REDUCE`; floating-point sums
/// keep two passes either way, to stay reproducible).
const LOOKBACK_ENV: &str = "LUMA_LOOKBACK_SCAN";
/// `LUMA_NATIVE_SOLVE=off` runs the line solves of a backend that has native ones (the CPU backend)
/// as compiled shaders, like the GPU backends (see [`Tuning::native_solve`]).
const NATIVE_SOLVE_ENV: &str = "LUMA_NATIVE_SOLVE";

/// `LUMA_NATIVE_GEMM=off` runs the matrix products of a backend that has native ones (the CPU
/// backend) as compiled tile kernels, like the GPU backends (see [`Tuning::native_gemm`]).
const NATIVE_GEMM_ENV: &str = "LUMA_NATIVE_GEMM";

/// How many output elements one invocation of a fused elementwise kernel computes (see `jit::coarsen`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum Coarsen {
    /// One element per invocation, always.
    Off,
    /// The planner picks 1, 2, 4 or 8 per launch from the size of the array and the cost of the program.
    Auto,
    /// This many (1, 2, 4 or 8) wherever the kernel can run it, whatever the size.
    Fixed(u8),
}

impl Coarsen {
    /// Reads `LUMA_COARSEN`: `off`, `auto` or a count (rounded down to 1, 2, 4 or 8); `default` if unset or not understood.
    fn from_env(default: Coarsen) -> Coarsen {
        match std::env::var(COARSEN_ENV).as_deref().map(str::trim) {
            Ok("off") => Coarsen::Off,
            Ok("auto") => Coarsen::Auto,
            Ok(value) => match value.parse::<u8>() {
                Ok(0) => Coarsen::Off,
                Ok(n) => Coarsen::Fixed(match n {
                    1 => 1,
                    2 | 3 => 2,
                    4..=7 => 4,
                    _ => 8,
                }),
                Err(_) => default,
            },
            Err(_) => default,
        }
    }
}

/// [`Tuning::axis_defer_cells`] of the wgpu backend: above the 131072 cells at which leaving the reductions pending lost 9 % and below the
/// 1048576 at which it won 7 % (an RTX 3090, `ml/train`); where in between they cross is not measured.
const AXIS_DEFER_CELLS: usize = 1 << 19;

/// The algorithm of the tridiagonal and block-tridiagonal line solves.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum LineSolver {
    /// Cyclic reduction for few, long lines when a line fits the workgroup memory budget
    /// ([`Tuning::line_bytes`], spilling the rest to global memory), Thomas otherwise.
    Auto,
    /// One invocation per line, always.
    Thomas,
    /// One workgroup per line whenever the device can run it.
    Cyclic,
}

pub(crate) use luma_core::TuningProfile;
use luma_core::{BackendKind, DeviceInfo, DeviceLimits, Features};
use crate::jit::structured::sites::{LazyMode, MAX_PROBE_EVERY};

/// Kernel parameters of a context.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct Tuning {
    pub(crate) profile: TuningProfile,
    /// Invocations per workgroup of elementwise kernels (fused and fills).
    pub(crate) elementwise_workgroup: u32,
    /// Output elements per invocation of fused elementwise kernels (`jit::coarsen`); `LUMA_COARSEN` overrides. Automatic on the
    /// NVIDIA profile, where it was measured; off elsewhere.
    pub(crate) coarsen: Coarsen,
    /// Tiled 1D convolution: invocations per workgroup, and outputs per invocation at stride 1.
    pub(crate) conv_workgroup: u32,
    pub(crate) conv_per_thread: u32,
    /// Time-blocked stencils: workgroup memory for the two tile buffers, and the output tile of
    /// 2D and 1D arrays.
    pub(crate) stencil_tile_bytes: u32,
    pub(crate) stencil_tile_2d: (u32, u32),
    pub(crate) stencil_tile_1d: u32,
    /// Elements each reduction invocation loads (and combines pairwise) per block.
    pub(crate) reduce_per_thread: u32,
    /// Most workgroups of a reduction pass; they stride over arrays larger than that many blocks.
    pub(crate) reduce_max_groups: u32,
    /// Most accumulators (reductions of one operand each, ops counted) one kernel computes when several
    /// pending reductions of the same length are forced together (see `jit::multireduce`); below 2 they are
    /// never merged. `LUMA_MULTI_REDUCE` overrides.
    pub(crate) reduce_multi: usize,
    /// How many times a group of pending reductions is recorded one by one (with the kernel all reductions of their shape
    /// share) before the kernel that computes the group together is compiled, in the background; the group is recorded in
    /// it once it is ready, and no launch waits for it. Compiling it costs 0.1 to 1 s on CUDA, which a group recorded a few
    /// times (a solver's final forces) never earns back. 0 records every group in its kernel at once, waiting for its compile.
    /// `LUMA_MULTI_REDUCE_AFTER` overrides; the default is 4 on CUDA and 0 elsewhere.
    pub(crate) reduce_multi_after: u32,
    /// Least weight (f32 operations per cell, see `jit::Node::weight`) of the operand of an `f32` axis reduction for it to be left
    /// pending to be recorded with its siblings (see `Array::defer_axis`). Merged, the reductions share the operands' work and the
    /// pass over the data, but the CPU backend returns the values of several roots as an array and combines them in lockstep, which
    /// costs more per cell than light operands save (a table's NaN-aware column statistics ran 1.5-1.7x slower merged, its
    /// 4-operation description and a layer norm gained less than they lost to the deferral); all-pairs forces, whose three
    /// components share a long chain, run 3x faster. Zero on GPU backends: always deferred.
    pub(crate) axis_defer_weight: u32,
    /// Fewest cells of the operand of an `f32` axis reduction for it to be left pending to be recorded with its siblings (see
    /// `Array::defer_axis`); smaller operands are reduced when the call is made. Recording a reduction late moves its kernel behind
    /// the ones recorded after its operands and, because the group has to wait for its last member, behind the work that follows
    /// it: where kernels are short that costs more than sharing the pass saves (`ml/train/sgd_medium` on an RTX 3090, three bias
    /// gradients of 131072 cells or fewer: 0.606 ms recorded at once, 0.662 ms left pending; at 1048576 cells, `sgd_large`, pending
    /// is 1.405 against 1.306 ms). Zero on the CPU backend, which judges by [`Tuning::axis_defer_weight`], and on CUDA and Metal, where it was not measured.
    pub(crate) axis_defer_cells: usize,
    /// Whether a reduction with more than one workgroup finishes in its first launch: the last
    /// workgroup to write its partial result combines them all (see `builtin::reduce_main`),
    /// instead of a second launch doing it. Needs a memory model that orders the partial results
    /// across workgroups, which WGSL does not give (wgpu, Metal): only the CUDA backend sets it.
    pub(crate) single_pass_reduce: bool,
    /// When library ops (the matrix product first) launch: at the call, or when the value is used (see `jit::structured::sites`).
    /// `LUMA_LAZY` overrides; `Site` since W3-U4 (a product can absorb an epilogue).
    pub(crate) lazy: LazyMode,
    /// A lazy site probes (makes one call lazy) at its first call and every this many calls after it; 0: never. At most 256 and a
    /// power of two. `LUMA_LAZY_PROBE` overrides.
    pub(crate) lazy_probe_every: u32,
    /// One execution of a strategy in this many is timed on the device (`jit::structured::telemetry`); 0: never.
    /// `LUMA_TELEMETRY` overrides.
    pub(crate) telemetry_every: u32,
    /// Samples of a strategy and shape class before their median counts as a measured cost. `LUMA_TELEMETRY_MIN` overrides.
    pub(crate) telemetry_min: u32,
    /// One execution in this many runs a rival strategy of the same bit class instead of the chosen one until it has
    /// `telemetry_min` samples; 0: never. `LUMA_EXPLORE` overrides.
    pub(crate) explore_every: u32,
    /// Most kernels of non-baseline strategies a family may build per context (the wgpu pipeline cache is dropped whole above
    /// its size cap). `LUMA_VARIANT_BUDGET` overrides.
    pub(crate) variant_budget: u32,
    /// Whether reductions combine within subgroups (the device has `Features::SUBGROUP`).
    pub(crate) subgroups: bool,
    /// Whether every subgroup has exactly 32 invocations (the radix sort ranks keys with ballots
    /// then; implies `subgroups`).
    pub(crate) subgroup32: bool,
    /// Bytes of workgroup memory a kernel may declare (the device's limit).
    pub(crate) workgroup_storage: u32,
    /// Most operations of a lazy expression a shift recomputes at the shifted element instead of evaluating it first.
    pub(crate) max_recomputed: usize,
    /// Most values a fused kernel keeps live at once before it is split in two (register pressure).
    pub(crate) max_live: usize,
    /// Most instructions of a kernel that reads sub-programs through views once they are written
    /// out at every index they are read at (see `jit::inline`); a bigger one is cut by evaluating
    /// one of its sub-programs first.
    pub(crate) max_inlined: usize,
    /// Whether sub-programs read through translation views at several offsets may be computed once
    /// per workgroup tile in workgroup memory (see `jit::staging`); `LUMA_STAGING=off` clears it.
    pub(crate) staging: bool,
    /// Whether the device's `fma` is a fused multiply-add (one rounding), so that the library's
    /// `Df32` products take the exact error of a product from it instead of splitting the
    /// factors. Measured when the device opens (see `gpu::fma`) or forced with
    /// `ContextOptions::df_fma`; the constants say `false` and the context fills it in.
    pub(crate) df_fma: bool,
    /// Whether `Df32` additions use the cheaper algorithm whose relative error grows with the
    /// cancellation (set by `Reduced::FAST_MATH`, see `builtin::df_library`).
    pub(crate) df_sloppy_add: bool,
    /// Workgroup memory the rows of one line of a cyclic-reduction tridiagonal solve may take; the
    /// rows beyond it live in global memory (see `cyclic`).
    pub(crate) line_bytes: u32,
    /// Most lines, weighted by `block^2` (lines along the contiguous axis) or `4 * block`
    /// (strided lines), the automatic choice still solves by cyclic reduction: with more lines the
    /// one-invocation-per-line Thomas solve fills the device and is faster (see `cyclic::plan`);
    /// 0 never chooses cyclic reduction on its own.
    pub(crate) line_cyclic_work: u32,
    /// The smallest stride (cells between neighbouring cells of a line) from which the Thomas
    /// solve of lines of blocks of 1 to 5 takes the wider workgroups and the tiled scratch of
    /// `builtin::thomas_plan`: neighbouring lines are then neighbouring cells and a warp's
    /// accesses coalesce. 0 keeps the narrow workgroups and the per-cell scratch.
    pub(crate) line_strided_from: u32,
    /// Which tridiagonal solver runs; `LUMA_TRIDIAGONAL` overrides.
    pub(crate) line_solver: LineSolver,
    /// Whether the backend runs the Thomas line solves as native code (`Backend::native_line_solve`,
    /// `LUMA_NATIVE_SOLVE=off` clears it): the planner then never takes cyclic reduction on its own and
    /// solves blocks of any size by the block Thomas algorithm (see `builtin::thomas_plan`).
    pub(crate) native_solve: bool,
    /// The cores a native matrix product spreads over (`Backend::native_matmul`; 0 if the backend has
    /// none or `LUMA_NATIVE_GEMM=off`): the products are then planned as output tiles for that many
    /// cores and their operands evaluated first (see `matmul::native_plan`).
    pub(crate) native_gemm: u32,
    /// Whether a scan of long contiguous lines is one kernel that reads its input once, with a
    /// decoupled look-back (each workgroup takes a chunk from an atomic counter, publishes the
    /// chunk's aggregate and waits for the prefix of the chunks before it, see
    /// `scan::Pass::ChunkLookback`), instead of chunk totals, a scan of the totals and a seeded
    /// second pass. Only for scans whose result does not depend on how the look-back groups the
    /// chunks before one (integers, `min`, `max`): the floating-point sums keep the two passes,
    /// so that scans are reproducible. Needs forward progress and memory ordering across
    /// workgroups, which WGSL does not give (wgpu, Metal): only the CUDA backend sets it.
    /// `LUMA_LOOKBACK_SCAN` overrides.
    pub(crate) lookback_scan: bool,
    /// Multiprocessors (compute units) of the device, 82 (the RTX 3090 the occupancy-derived plan
    /// constants were measured on) when the backend cannot tell. See [`Tuning::scaled`].
    pub(crate) compute_units: u32,
}

/// Multiprocessors of the RTX 3090, on which the plan constants that depend on how many workgroups
/// fill the device (`matmul::SPLIT_GROUPS`, `builtin::thomas_plan`)
/// were measured.
pub(crate) const REFERENCE_COMPUTE_UNITS: u32 = 82;

impl Tuning {
    /// [`scaled`](Self::scaled), but never below `reference`: for counts of workgroups that keep a
    /// device busy, where more than enough cost almost nothing (a K-split matmul on an RTX 3080
    /// Laptop was 2.7 % slower with the 599 workgroups of its 48 multiprocessors than with 1024).
    pub(crate) fn scaled_up(&self, reference: u32) -> u32 {
        self.scaled(reference).max(reference)
    }

    /// A count of workgroups or lines tuned on the reference device ([`REFERENCE_COMPUTE_UNITS`]),
    /// for this device: proportional to its multiprocessors, rounded to nearest, at least 1 for a
    /// nonzero count and unchanged at the reference.
    pub(crate) fn scaled(&self, reference: u32) -> u32 {
        let units = u64::from(self.compute_units.max(1));
        let scaled = (u64::from(reference) * units + u64::from(REFERENCE_COMPUTE_UNITS) / 2) / u64::from(REFERENCE_COMPUTE_UNITS);
        if reference == 0 { 0 } else { scaled.clamp(1, u64::from(u32::MAX)) as u32 }
    }

    const APPLE: Tuning = Tuning {
        profile: TuningProfile::Apple,
        elementwise_workgroup: 64,
        coarsen: Coarsen::Off,
        conv_workgroup: 128,
        conv_per_thread: 32,
        stencil_tile_bytes: 8 * 1024,
        stencil_tile_2d: (32, 16),
        // A wider 1D tile leaves no room in 8 KiB for the halo of a second step.
        stencil_tile_1d: 512,
        reduce_per_thread: 4,
        reduce_max_groups: 1024,
        reduce_multi: 16,
        reduce_multi_after: 0,
        axis_defer_weight: 0,
        axis_defer_cells: 0,
        single_pass_reduce: false,
        lazy: LazyMode::Site,
        lazy_probe_every: 256,
        telemetry_every: 32,
        telemetry_min: 8,
        explore_every: 0,
        variant_budget: 48,
        subgroups: false,
        subgroup32: false,
        workgroup_storage: 16384,
        max_recomputed: 24,
        max_live: 64,
        max_inlined: 1024,
        staging: true,
        df_fma: false,
        df_sloppy_add: false,
        line_bytes: 32 * 1024,
        // Cyclic reduction wins on parts along the contiguous axis and on few lines.
        line_cyclic_work: 8192,
        // 32-wide SIMD groups, like NVIDIA warps: see `builtin::thomas_plan`.
        line_strided_from: 16,
        line_solver: LineSolver::Auto,
        native_solve: false,
        native_gemm: 0,
        lookback_scan: false,
        compute_units: REFERENCE_COMPUTE_UNITS,
    };

    const NVIDIA: Tuning = Tuning {
        profile: TuningProfile::Nvidia,
        elementwise_workgroup: 256,
        coarsen: Coarsen::Auto,
        conv_workgroup: 256,
        conv_per_thread: 32,
        stencil_tile_bytes: 32 * 1024,
        stencil_tile_2d: (64, 32),
        stencil_tile_1d: 2048,
        reduce_per_thread: 8,
        reduce_max_groups: 2048,
        reduce_multi: 16,
        reduce_multi_after: 0,
        axis_defer_weight: 0,
        axis_defer_cells: 0,
        single_pass_reduce: false,
        lazy: LazyMode::Site,
        lazy_probe_every: 256,
        telemetry_every: 32,
        telemetry_min: 8,
        explore_every: 0,
        variant_budget: 48,
        subgroups: false,
        subgroup32: false,
        workgroup_storage: 16384,
        max_recomputed: 32,
        max_live: 96,
        max_inlined: 1536,
        staging: true,
        df_fma: false,
        df_sloppy_add: false,
        // All of the 48 KiB a workgroup may have: the rows of a line that fit it need no global
        // scratch (see `cyclic::plan`).
        line_bytes: 48 * 1024,
        // RTX 3090, see `cyclic::plan`.
        line_cyclic_work: 8192,
        // RTX 3090, see `builtin::thomas_plan`.
        line_strided_from: 16,
        line_solver: LineSolver::Auto,
        native_solve: false,
        native_gemm: 0,
        lookback_scan: false,
        compute_units: REFERENCE_COMPUTE_UNITS,
    };

    const GENERIC: Tuning = Tuning {
        profile: TuningProfile::Generic,
        elementwise_workgroup: 256,
        coarsen: Coarsen::Off,
        conv_workgroup: 256,
        conv_per_thread: 16,
        stencil_tile_bytes: 16 * 1024,
        stencil_tile_2d: (32, 16),
        stencil_tile_1d: 1024,
        reduce_per_thread: 4,
        reduce_max_groups: 1024,
        reduce_multi: 16,
        reduce_multi_after: 0,
        axis_defer_weight: 0,
        axis_defer_cells: 0,
        single_pass_reduce: false,
        lazy: LazyMode::Site,
        lazy_probe_every: 256,
        telemetry_every: 32,
        telemetry_min: 8,
        explore_every: 0,
        variant_budget: 48,
        subgroups: false,
        subgroup32: false,
        workgroup_storage: 16384,
        max_recomputed: 24,
        max_live: 96,
        max_inlined: 1024,
        staging: true,
        df_fma: false,
        df_sloppy_add: false,
        line_bytes: 16 * 1024,
        // Not measured on other GPUs (a RADV iGPU was slower, 8.7 vs 3.0 ms for 200 lines of 4096): Thomas unless forced.
        line_cyclic_work: 0,
        line_strided_from: 16,
        line_solver: LineSolver::Auto,
        native_solve: false,
        native_gemm: 0,
        lookback_scan: false,
        compute_units: REFERENCE_COMPUTE_UNITS,
    };

    /// Software rasterizers: the generic parameters with the smaller elementwise workgroups
    /// that measured faster on llvmpipe.
    const CPU: Tuning = Tuning {
        profile: TuningProfile::Cpu,
        elementwise_workgroup: 64,
        coarsen: Coarsen::Off,
        max_live: 256,
        max_inlined: 1024,
        line_cyclic_work: 0,
        ..Tuning::GENERIC
    };

    /// The parameters for the device `info` describes, unless the environment forces a profile.
    pub(crate) fn for_device(info: &DeviceInfo) -> Tuning {
        let limits = &info.limits;
        let forced = std::env::var(TUNING_ENV).ok();
        let profile = match forced.as_deref().map(str::trim) {
            Some("apple") => TuningProfile::Apple,
            Some("nvidia") => TuningProfile::Nvidia,
            Some("generic") => TuningProfile::Generic,
            Some("cpu") => TuningProfile::Cpu,
            _ => info.profile,
        };
        let tuning = match profile {
            TuningProfile::Apple => Tuning::APPLE,
            TuningProfile::Nvidia => Tuning::NVIDIA,
            TuningProfile::Generic => Tuning::GENERIC,
            TuningProfile::Cpu => Tuning::CPU,
            _ => Tuning::GENERIC,
        };
        let mut tuning = tuning.within(limits);
        if let Some(live) = std::env::var(MAX_LIVE_ENV).ok().and_then(|value| value.trim().parse::<usize>().ok()) {
            tuning.max_live = live.clamp(8, 4096);
        }
        if let Some(inlined) = std::env::var(MAX_INLINED_ENV).ok().and_then(|value| value.trim().parse::<usize>().ok()) {
            tuning.max_inlined = inlined.clamp(16, 16384);
        }
        tuning.subgroups = info.features.contains(Features::SUBGROUP) && std::env::var(SUBGROUPS_ENV).as_deref() != Ok("off");
        tuning.workgroup_storage = limits.max_workgroup_storage_bytes;
        tuning.subgroup32 = tuning.subgroups && info.subgroup_size == Some((32, 32));
        tuning.single_pass_reduce = match std::env::var(SINGLE_PASS_ENV).as_deref().map(str::trim) {
            Ok("off") => false,
            Ok("on") => matches!(info.backend, BackendKind::Cuda | BackendKind::Cpu),
            _ => info.backend == BackendKind::Cuda,
        };
        tuning.lazy = std::env::var(LAZY_ENV).ok().and_then(|value| LazyMode::parse(&value)).unwrap_or(tuning.lazy);
        tuning.lazy_probe_every = match std::env::var(LAZY_PROBE_ENV).ok().and_then(|value| value.trim().parse::<u32>().ok()) {
            Some(0) => 0,
            Some(every) => 1 << every.min(MAX_PROBE_EVERY).ilog2(),
            None => tuning.lazy_probe_every,
        };
        tuning.telemetry_every = match std::env::var(TELEMETRY_ENV).as_deref().map(str::trim) {
            Ok("off") => 0,
            Ok(value) => value.parse::<u32>().unwrap_or(tuning.telemetry_every),
            Err(_) => tuning.telemetry_every,
        };
        for (name, slot) in [(TELEMETRY_MIN_ENV, &mut tuning.telemetry_min), (EXPLORE_ENV, &mut tuning.explore_every), (VARIANT_BUDGET_ENV, &mut tuning.variant_budget)] {
            match std::env::var(name).as_deref().map(str::trim) {
                Ok("off") => *slot = 0,
                Ok(value) => *slot = value.parse::<u32>().unwrap_or(*slot),
                Err(_) => {}
            }
        }
        tuning.reduce_multi = match std::env::var(MULTI_REDUCE_ENV).as_deref().map(str::trim) {
            Ok("off") => 1,
            Ok(value) => value.parse::<usize>().map_or(tuning.reduce_multi, |n| n.clamp(1, crate::builtin::MAX_ACCS)),
            Err(_) => tuning.reduce_multi,
        };
        tuning.reduce_multi_after = std::env::var(MULTI_REDUCE_AFTER_ENV)
            .ok()
            .and_then(|value| value.trim().parse::<u32>().ok())
            .unwrap_or(if info.backend == BackendKind::Cuda { 4 } else { 0 });
        if info.backend == BackendKind::Cpu {
            tuning.axis_defer_weight = 12;
        } else if info.backend == BackendKind::Wgpu {
            tuning.axis_defer_cells = AXIS_DEFER_CELLS;
        }
        tuning.lookback_scan = match std::env::var(LOOKBACK_ENV).as_deref().map(str::trim) {
            Ok("off") => false,
            Ok("on") => matches!(info.backend, BackendKind::Cuda | BackendKind::Cpu),
            _ => info.backend == BackendKind::Cuda,
        };
        // Software rasterizers and the CPU backend keep the reference: their "units" are threads,
        // which the plans were never measured against.
        if let Some(units) = info.compute_units.filter(|_| profile != TuningProfile::Cpu) {
            tuning.compute_units = units.max(1);
        }
        tuning.coarsen = Coarsen::from_env(tuning.coarsen);
        tuning.staging = std::env::var(STAGING_ENV).as_deref().map(str::trim) != Ok("off");
        tuning.line_solver = match std::env::var(LINE_SOLVER_ENV).as_deref().map(str::trim) {
            Ok("thomas") => LineSolver::Thomas,
            Ok("cyclic") => LineSolver::Cyclic,
            _ => LineSolver::Auto,
        };
        tuning
    }

    /// Whether the line solves run natively, given whether the backend can: see
    /// [`Tuning::native_solve`].
    pub(crate) fn native_solve_for(supported: bool) -> bool {
        supported && std::env::var(NATIVE_SOLVE_ENV).as_deref().map(str::trim) != Ok("off")
    }

    /// The cores of [`Tuning::native_gemm`], given the cores of the backend's native product (if it has one).
    pub(crate) fn native_gemm_for(cores: Option<u32>) -> u32 {
        cores.filter(|_| std::env::var(NATIVE_GEMM_ENV).as_deref().map(str::trim) != Ok("off")).unwrap_or(0)
    }

    /// The generic parameters, for tests that need a `Tuning` without a device.
    #[cfg(test)]
    pub(crate) fn for_tests() -> Tuning {
        Tuning::GENERIC
    }

    /// Shrinks what the device cannot run.
    fn within(mut self, limits: &DeviceLimits) -> Tuning {
        let invocations = limits.max_workgroup_invocations.min(limits.max_workgroup_size[0]);
        self.elementwise_workgroup = self.elementwise_workgroup.min(invocations);
        self.conv_workgroup = self.conv_workgroup.min(invocations);
        while self.conv_per_thread > 4 && conv_workgroup_bytes(self.conv_workgroup, self.conv_per_thread) > limits.max_workgroup_storage_bytes {
            self.conv_per_thread /= 2;
        }
        self.stencil_tile_bytes = self.stencil_tile_bytes.min(limits.max_workgroup_storage_bytes);
        self.line_bytes = self.line_bytes.min(limits.max_workgroup_storage_bytes);
        // `stencil::layout` always runs at least one step, whatever the budget.
        while self.stencil_tile_1d > 32 && stencil_bytes((self.stencil_tile_1d, 1), (STENCIL_RADIUS, 0)) > self.stencil_tile_bytes {
            self.stencil_tile_1d /= 2;
        }
        let radius = (STENCIL_RADIUS, STENCIL_RADIUS);
        while self.stencil_tile_2d.0.max(self.stencil_tile_2d.1) > 4 && stencil_bytes(self.stencil_tile_2d, radius) > self.stencil_tile_bytes {
            let (x, y) = &mut self.stencil_tile_2d;
            if *x >= *y { *x /= 2 } else { *y /= 2 }
        }
        self
    }

    /// Elements one reduction workgroup consumes at a time.
    pub(crate) fn reduce_block(&self) -> u32 {
        crate::builtin::REDUCE_WORKGROUP * self.reduce_per_thread
    }

    /// Workgroups of a reduction pass over `n` elements.
    pub(crate) fn reduce_groups(&self, n: u32) -> u32 {
        n.div_ceil(self.reduce_block()).min(self.reduce_max_groups)
    }
}

/// Workgroup memory of the tiled convolution at stride 1: the input window (with one skew word
/// per 32) and the weights of a chunk (see `builtin::conv1d_tiled`).
fn conv_workgroup_bytes(workgroup: u32, per_thread: u32) -> u32 {
    let span = workgroup * per_thread + workgroup;
    (span + span / 32 + 1 + workgroup) * 4
}

/// Largest stencil radius (per axis) time-blocked; wider stencils iterate one kernel per step.
pub(crate) const STENCIL_RADIUS: u32 = 4;

/// Workgroup memory of the two buffers of a stencil tile of `tile` outputs with a halo of
/// `halo` cells per side and axis.
pub(crate) fn stencil_bytes(tile: (u32, u32), halo: (u32, u32)) -> u32 {
    (tile.0 + 2 * halo.0) * (tile.1 + 2 * halo.1) * 8
}
#[cfg(test)]
mod tests {
    use super::{conv_workgroup_bytes, stencil_bytes, DeviceLimits, Tuning, REFERENCE_COMPUTE_UNITS, STENCIL_RADIUS};

    #[test]
    fn scaling_is_the_identity_at_the_reference_device() {
        let at = |compute_units| Tuning { compute_units, ..Tuning::NVIDIA };
        for v in [0u32, 1, 8, 512, 1024, 8192, 1 << 20] {
            assert_eq!(at(REFERENCE_COMPUTE_UNITS).scaled(v), v);
        }
        assert_eq!(at(48).scaled(1024), 599);
        assert_eq!(at(48).scaled(8192), 4795);
        assert_eq!(at(1).scaled(1), 1);
        assert_eq!(at(48).scaled_up(1024), 1024);
        assert_eq!(at(164).scaled_up(1024), 2048);
        assert_eq!(at(164).scaled(1024), 2048);
        assert_eq!(at(48).scaled(0), 0);
    }

    #[test]
    fn tunings_fit_the_smallest_webgpu_limits() {
        let limits = DeviceLimits {
            max_buffer_bytes: 256 << 20,
            max_storage_buffers: 8,
            max_uniform_bytes: 16384,
            uniform_align: 256,
            max_dynamic_uniforms: 8,
            max_workgroups_per_dim: 65535,
            max_workgroup_size: [256, 256, 64],
            max_workgroup_invocations: 256,
            max_workgroup_storage_bytes: 16384,
        };
        for tuning in [Tuning::APPLE, Tuning::NVIDIA, Tuning::GENERIC, Tuning::CPU] {
            let t = tuning.within(&limits);
            assert!(conv_workgroup_bytes(t.conv_workgroup, t.conv_per_thread) <= limits.max_workgroup_storage_bytes, "{t:?}");
            assert!(t.stencil_tile_bytes <= limits.max_workgroup_storage_bytes);
            assert!(t.line_bytes <= limits.max_workgroup_storage_bytes);
            assert!(stencil_bytes((t.stencil_tile_1d, 1), (STENCIL_RADIUS, 0)) <= t.stencil_tile_bytes, "{t:?}");
            assert!(stencil_bytes(t.stencil_tile_2d, (STENCIL_RADIUS, STENCIL_RADIUS)) <= t.stencil_tile_bytes, "{t:?}");
            assert!(t.elementwise_workgroup <= limits.max_workgroup_invocations);
        }
    }
}
