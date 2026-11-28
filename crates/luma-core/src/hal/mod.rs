//! The backend hardware abstraction layer: device-independent types and the [`Backend`] trait
//! every GPU backend implements.

use std::any::Any;
use std::borrow::Cow;
use std::fmt;
use std::mem::MaybeUninit;
use std::panic::Location;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::Duration;

pub mod cache;
pub mod copy;
mod deferred;
pub mod hostbytes;
pub mod texture;
mod transfer;
pub mod topology;

pub use cache::DiskCache;
pub use copy::{advise_huge_pages, copy_parallel, copy_parallel_host, copy_streaming};
pub use texture::{TextureCaps, TextureDesc, TextureFilter, TextureFormat};
pub use hostbytes::{set_host_pool_cap, trim_host_pool, HostBytes};
pub use transfer::{NativeStage, Stage, StageDir, Ticket};

/// Which backend a context runs on.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum BackendKind {
    /// The best backend available: CUDA, then Metal, then wgpu.
    #[default]
    Auto,
    /// NVIDIA CUDA.
    Cuda,
    /// Apple Metal.
    Metal,
    /// WebGPU through wgpu.
    Wgpu,
    /// Native CPU (Cranelift SIMD).
    Cpu,
}

impl fmt::Display for BackendKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            BackendKind::Auto => "auto",
            BackendKind::Cuda => "cuda",
            BackendKind::Metal => "metal",
            BackendKind::Wgpu => "wgpu",
            BackendKind::Cpu => "cpu",
        })
    }
}

/// Which kind of device a context opens.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum DeviceKind {
    /// A GPU if there is one, otherwise the native CPU backend.
    #[default]
    Auto,
    /// A GPU only.
    Gpu,
    /// The native CPU backend only (software GPU adapters such as llvmpipe are never used).
    Cpu,
}

/// The family of hardware a device belongs to; selects tuning constants.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum TuningProfile {
    /// Apple silicon.
    Apple,
    /// NVIDIA GPUs.
    Nvidia,
    /// Every other GPU.
    Generic,
    /// A software device.
    Cpu,
}

/// What a kernel binding in `@group(1)` is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BindingKind {
    /// `var<storage, read>`: bound from `&Array`.
    StorageRead,
    /// `var<storage, read_write>`: bound from `&mut Array`.
    StorageReadWrite,
    /// `var<uniform>`: bound from a uniform value.
    Uniform,
    /// `texture_2d<f32>`: bound from a texture ([`Backend::create_texture`]).
    Texture2d,
    /// `sampler`: bound from the same texture as the `texture_2d<f32>` it samples. Its filtering
    /// and addressing are those of the texture ([`TextureDesc`]).
    Sampler,
}

impl fmt::Display for BindingKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            BindingKind::StorageRead => "read-only storage buffer",
            BindingKind::StorageReadWrite => "read_write storage buffer",
            BindingKind::Uniform => "uniform",
            BindingKind::Texture2d => "2-D texture",
            BindingKind::Sampler => "sampler",
        })
    }
}

/// One reflected `@group(1)` binding of a kernel; its index is the argument position.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BindingInfo {
    /// The variable's name in the shader.
    pub name: String,
    /// Whether the binding is a read-only storage buffer, a read-write storage buffer, a uniform,
    /// a texture or a sampler.
    pub kind: BindingKind,
    /// For storage bindings of `array<T>`: the byte stride of `T`. `None` when not an array.
    pub element_stride: Option<u32>,
    /// For uniforms: the byte size of the uniform's type. `None` for storage bindings.
    pub min_size: Option<u32>,
}

/// Optional device capabilities.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Features(u32);

impl Features {
    /// No features.
    pub const NONE: Self = Self(0);
    /// GPU timestamp queries.
    pub const TIMESTAMP_QUERY: Self = Self(1 << 0);
    /// Subgroup (warp, simdgroup) operations.
    pub const SUBGROUP: Self = Self(1 << 1);
    /// 16-bit floats in shaders.
    pub const SHADER_F16: Self = Self(1 << 2);
    /// 64-bit floats in shaders.
    pub const SHADER_F64: Self = Self(1 << 3);
    /// Cooperative matrix operations (wgpu only).
    pub const EXPERIMENTAL_COOPERATIVE_MATRIX: Self = Self(1 << 4);

    /// No features.
    pub const fn empty() -> Self {
        Self(0)
    }
    /// Whether every feature of `other` is set.
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
    /// The features of both.
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }
    /// Whether no feature is set.
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }
    /// The raw bits.
    pub const fn bits(self) -> u32 {
        self.0
    }
    /// Enabled features and their names.
    pub fn iter_names(self) -> impl Iterator<Item = (&'static str, Self)> {
        [
            ("TIMESTAMP_QUERY", Self::TIMESTAMP_QUERY),
            ("SUBGROUP", Self::SUBGROUP),
            ("SHADER_F16", Self::SHADER_F16),
            ("SHADER_F64", Self::SHADER_F64),
            ("EXPERIMENTAL_COOPERATIVE_MATRIX", Self::EXPERIMENTAL_COOPERATIVE_MATRIX),
        ]
        .into_iter()
        .filter(move |&(_, flag)| self.contains(flag))
    }
}

macro_rules! bit_ops {
    ($t:ident) => {
        impl std::ops::BitOr for $t {
            type Output = Self;
            fn bitor(self, rhs: Self) -> Self {
                Self(self.0 | rhs.0)
            }
        }
        impl std::ops::BitOrAssign for $t {
            fn bitor_assign(&mut self, rhs: Self) {
                self.0 |= rhs.0;
            }
        }
        impl std::ops::BitAnd for $t {
            type Output = Self;
            fn bitand(self, rhs: Self) -> Self {
                Self(self.0 & rhs.0)
            }
        }
        impl std::ops::BitAndAssign for $t {
            fn bitand_assign(&mut self, rhs: Self) {
                self.0 &= rhs.0;
            }
        }
    };
}
bit_ops!(Features);
bit_ops!(BufferUsage);

/// Throughputs of a device, measured when a context opens or taken from a profile.
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

/// How bytes move inside a device, measured by calibration next to the [`CostModel`] numbers.
/// Wave 1 only measures and exposes them; no planning decision reads them yet.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub struct MemoryModel {
    /// L2 cache size: the device attribute where the backend reports it (CUDA), else the
    /// measured knee of the copy bandwidth over power-of-two working sets, which is only an upper
    /// bound (the largest size still much faster than DRAM, within one doubling of the real size):
    /// consumers must not assume it is exact. `None` where no sharp knee was found.
    pub l2_bytes: Option<u64>,
    /// Whether `l2_bytes` is exact (a device attribute) rather than the power-of-two upper bound
    /// of the knee; `false` when `l2_bytes` is `None`.
    pub l2_exact: bool,
    /// Read bandwidth, GB/s, of a copy whose traffic wraps over a working set inside L2 (the
    /// largest measured set that is at least 1.5 times faster than DRAM).
    pub l2_gbs: f64,
    /// Workgroup-memory read bandwidth, GB/s; `None` on the CPU.
    pub shared_gbs: Option<f64>,
    /// Time per byte relative to coalesced access for strides of 2, 8 and 32 elements.
    pub strided_factor: [f64; 3],
    /// Time per byte of random 4-byte gathers relative to coalesced access.
    pub gather_factor: f64,
    /// Time per byte of a naive transposed read relative to coalesced access.
    pub transpose_factor: f64,
    /// Nanoseconds per dependent load of a one-workgroup chain.
    pub latency_ns: f64,
    /// What the transfer lanes move between host and device, `None` on backends whose lanes are the synchronous defaults.
    pub transfer: Option<TransferModel>,
}

/// How fast the transfer lanes of a device move bytes, and what a running transfer costs the kernels beside it.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub struct TransferModel {
    /// Host to device throughput of one transfer into pinned memory, GB/s.
    pub h2d_gbs: f64,
    /// Device to host throughput, GB/s.
    pub d2h_gbs: f64,
    /// The share of a copy kernel's throughput lost while a transfer runs beside it (0: none, 1: it stops).
    pub overlap_derate: f64,
}

impl TransferModel {
    /// A transfer model of the given numbers (see the fields).
    pub fn new(h2d_gbs: f64, d2h_gbs: f64, overlap_derate: f64) -> TransferModel {
        TransferModel { h2d_gbs, d2h_gbs, overlap_derate }
    }
}

impl MemoryModel {
    /// This model with the transfer lanes' numbers.
    pub fn with_transfer(mut self, transfer: Option<TransferModel>) -> MemoryModel {
        self.transfer = transfer;
        self
    }

    /// A memory model of the given numbers (see the fields).
    #[allow(clippy::too_many_arguments)]
    pub fn new(l2_bytes: Option<u64>, l2_exact: bool, l2_gbs: f64, shared_gbs: Option<f64>, strided_factor: [f64; 3], gather_factor: f64, transpose_factor: f64, latency_ns: f64) -> MemoryModel {
        MemoryModel { l2_bytes, l2_exact, l2_gbs, shared_gbs, strided_factor, gather_factor, transpose_factor, latency_ns, transfer: None }
    }
}

impl fmt::Display for MemoryModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.l2_bytes {
            Some(bytes) => write!(f, "L2 {}{:.2} MiB at {:.0} GB/s", if self.l2_exact { "" } else { "up to " }, bytes as f64 / (1 << 20) as f64, self.l2_gbs)?,
            None => write!(f, "L2 unknown size at {:.0} GB/s", self.l2_gbs)?,
        }
        match self.shared_gbs {
            Some(gbs) => write!(f, ", workgroup memory {gbs:.0} GB/s")?,
            None => write!(f, ", no workgroup memory figure")?,
        }
        let [s2, s8, s32] = self.strided_factor;
        write!(f, ", strided x{s2:.2} / x{s8:.2} / x{s32:.2} (2 / 8 / 32), gather x{:.2}, transpose x{:.2}, latency {:.0} ns", self.gather_factor, self.transpose_factor, self.latency_ns)?;
        if let Some(t) = self.transfer {
            write!(f, ", transfers {:.1} GB/s in / {:.1} GB/s out, {:.0}% of copy throughput lost beside one", t.h2d_gbs, t.d2h_gbs, t.overlap_derate * 100.0)?;
        }
        Ok(())
    }
}

impl CostModel {
    /// A cost model of the given measurements.
    pub const fn new(launch_us: f64, bandwidth_gbs: f64, f32_ops_per_ns: f64, df32_factor: f64, f64_factor: Option<f64>) -> CostModel {
        CostModel { launch_us, bandwidth_gbs, f32_ops_per_ns, df32_factor, f64_factor }
    }

    /// This model with `change` applied.
    pub fn with(mut self, change: impl FnOnce(&mut CostModel)) -> CostModel {
        change(&mut self);
        self
    }

    /// RTX 3090 (Vulkan), measured by calibration (best of several runs).
    pub const NVIDIA: CostModel = CostModel::new(2.9, 740.0, 16000.0, 27.0, Some(41.0));
    /// Every other GPU, measured on a RADV Raphael iGPU (best of 5 runs).
    pub const GENERIC: CostModel = CostModel::new(2.0, 62.0, 370.0, 44.0, Some(11.0));
    /// Mesa llvmpipe on a Ryzen 9 7950X (best of 5 runs).
    pub const CPU: CostModel = CostModel::new(32.0, 31.0, 30.0, 2.3, Some(1.1));
    /// The numbers tests pin so that kernel counts do not depend on the machine.
    #[doc(hidden)]
    pub const TEST: CostModel = CostModel::new(2.9, 740.0, 16000.0, 36.0, Some(41.0));

    /// The fixed numbers of a tuning profile, used when calibration is off or fails.
    pub fn fallback(profile: TuningProfile) -> CostModel {
        match profile {
            TuningProfile::Nvidia => CostModel::NVIDIA,
            TuningProfile::Cpu => CostModel::CPU,
            _ => CostModel::GENERIC,
        }
    }
}

/// The GPU time one kernel took, summed over all of its launches in a profile.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KernelTiming {
    /// The label of the kernel: the one given to the kernel builder, otherwise its entry point
    /// name. Built-in operations are labelled by operation and element type, for example
    /// `Fill f32`. Fused elementwise kernels are labelled by the operations they contain, for
    /// example `Fused(Mul, Add, Sqrt) f32`, or `FusedReduce(Sum: Mul, Add) f32` for a reduction
    /// whose first pass computes the expression.
    pub label: String,
    /// How many launches were measured.
    pub launches: u32,
    /// Sum of the durations of all launches.
    pub total: Duration,
    /// The shortest single launch.
    pub min: Duration,
    /// The longest single launch.
    pub max: Duration,
}

impl KernelTiming {
    /// The average duration of a launch.
    pub fn mean(&self) -> Duration {
        self.total.checked_div(self.launches).unwrap_or_default()
    }
}

/// Why the fusion engine ended a kernel before the expression did.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum SplitReason {
    /// The expression held more operations than the fusion limit allows.
    Nodes {
        /// Operations the expression would have held.
        nodes: usize,
        /// The limit in force.
        limit: usize,
    },
    /// The expression was nested deeper than the engine walks in one kernel.
    Depth {
        /// The limit on the nesting depth.
        limit: usize,
    },
    /// The expression read more distinct buffers than a kernel can bind.
    Inputs {
        /// Buffers the expression would have read.
        inputs: usize,
        /// Buffers a kernel can bind.
        limit: usize,
    },
    /// The kernel would have kept more values live at once than is good for register pressure.
    Live {
        /// Values live at the busiest point.
        live: usize,
        /// The threshold of the device's tuning (`LUMA_FUSION_MAX_LIVE` overrides it).
        limit: usize,
    },
    /// The scalar operands of the expression would not fit the uniform buffer of one kernel.
    Scalars {
        /// 32-bit words the scalars would have taken.
        words: usize,
        /// Words a kernel's uniform buffer holds.
        limit: usize,
    },
    /// A kernel reading expressions through views would have computed more operations than the
    /// device's tuning allows once every expression is written out at each index it is read at.
    Inlined {
        /// Operations the kernel would have computed.
        instructions: usize,
        /// The threshold of the device's tuning (`LUMA_FUSION_MAX_INLINED` overrides it).
        limit: usize,
    },
    /// An expression read at several indices (through views) is stored once and read back: the
    /// cost model estimates that computing it at every read costs more than the write and the
    /// extra kernel.
    Materialised {
        /// Times the kernel would have computed the expression.
        instances: usize,
    },
    /// An expression that other pending operations outside the kernel read too is stored once
    /// and read back by both: computing it in each costs more than the write and the read.
    Shared {
        /// Operations of the expression, in f32 operations.
        weight: usize,
    },
}

/// Where the planner chose between alternatives.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum DecisionSite {
    /// An expression was cut into several kernels (see [`SplitReason`]).
    Cut(SplitReason),
    /// The inliner judged whether a shared sub-expression is recomputed or stored.
    Inline,
    /// An operand was staged (through shared memory or registers) or read directly.
    Staging,
    /// Work per thread of an elementwise kernel.
    Coarsen,
    /// A transpose fused with its consumer or not.
    TransposeFusion,
    /// How a view is read (direct, texture, packed).
    ReadPath,
    /// Matrix product strategy.
    MatmulPlan,
    /// Sort strategy.
    SortPlan,
    /// Line solver (cyclic reduction) strategy.
    LineSolvePlan,
    /// QR panel strategy.
    QrPanel,
    /// Reduction strategy (host, merged, bits, single or two passes, axis lines).
    ReducePlan,
    /// Scan strategy (rows, warp, look-back, chunked, strided; by key).
    ScanPlan,
    /// Scatter strategy (host, plain, atomic with private or global bins).
    ScatterPlan,
    /// Binary search (the unrolled search expression).
    SearchPlan,
    /// Compaction strategy (`filter`, `nonzero`: count then scatter).
    CompactPlan,
    /// Merge strategy (merge of sorted arrays; `factorize`).
    MergePlan,
    /// Dense factorisation strategy (the leaf and recursion of LU, Cholesky, QR, triangular solves, SVD, eigh; refined solves).
    FactorPlan,
    /// FFT strategy (Stockham or Bluestein, the real-transform packings, the convolution).
    SpectralPlan,
    /// How a sampler reads its source (hardware filter, texture gather or buffer).
    SamplePlan,
    /// Convolution strategy (direct tiled or per element, FFT whole or overlap-save, the FIR forms).
    ConvPlan,
    /// Windowed image kernel strategy (tiled window, separable extremum, the median selections).
    WindowPlan,
    /// Ops declared on the registry with one strategy (vision, image, IIR, sparse products).
    OpaquePlan,
    /// A form without an API tag was specialised (`recognised-<form>`) or lowered by the generic strategies (`generic-contract`,
    /// `generic-reduce`, `generic-unrolled`). Never recorded for an API operation.
    Comprehension,
    /// The rewrite engine changed a form: `chosen` is the rule (`t1`, when a form was built, or the normaliser of a forced value),
    /// `before` and `after` the estimated nanoseconds of the scope where they were compared. Never recorded for an elementwise
    /// rewrite (`simplify`).
    Rewrite,
    /// A custom op of the public extension API chose one of its strategies (`op` is its name, `chosen` the strategy).
    Extension,
    /// A cached plan was replayed: no planning decision was made.
    PlanReplay,
    /// A schedule of deferred kernels over streamed arrays was committed: `chosen` counts its stages, slabs and arrays (never ids).
    Schedule,
    /// A consumer of a structured value was absorbed into its epilogue, or not (`absorbed <n> ops`, `separate: <reason>`).
    Epilogue,
    /// The slab length of a stage of a schedule: `chosen` is `L=2^k` with the elements and the ghost share, `candidates` the neighbouring
    /// powers with their estimated cost and the bound that excluded the next one (`L_mem`, `L_pin`, `seg_len`, `cap`).
    Slab,
}

/// What became of a candidate that was not chosen.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum Verdict {
    /// Its estimated cost.
    Estimate {
        /// Nanoseconds.
        ns: f64,
    },
    /// It did not apply, for this reason.
    Rejected(&'static str),
}

/// One choice of the planner between alternatives, recorded while profiling or after
/// [`Context::record_decisions`](https://docs.rs/luma). Labels are stable strings (no pointers or ids).
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct Decision {
    /// The operation whose creation (or evaluation) caused the decision.
    pub op: &'static str,
    /// Where that operation was written.
    pub location: &'static Location<'static>,
    /// What was decided.
    pub site: DecisionSite,
    /// The chosen alternative, e.g. `tiled 128x64 split=4`, `coarsen 4`, `texture`.
    pub chosen: Cow<'static, str>,
    /// The alternatives that were not chosen (may be empty).
    pub candidates: Vec<(Cow<'static, str>, Verdict)>,
    /// The choice came from a memo (judgement cache, recipes).
    pub cached: bool,
}

impl Decision {
    /// A decision with no candidates, not cached.
    pub fn new(op: &'static str, location: &'static Location<'static>, site: DecisionSite, chosen: impl Into<Cow<'static, str>>) -> Decision {
        Decision { op, location, site, chosen: chosen.into(), candidates: Vec::new(), cached: false }
    }

    /// Adds a rejected alternative.
    #[must_use]
    pub fn candidate(mut self, label: impl Into<Cow<'static, str>>, verdict: Verdict) -> Decision {
        self.candidates.push((label.into(), verdict));
        self
    }

    /// Sets whether the choice came from a memo.
    #[must_use]
    pub fn cached(mut self, cached: bool) -> Decision {
        self.cached = cached;
        self
    }
}

impl fmt::Display for Decision {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let DecisionSite::Cut(reason) = self.site else {
            return write!(f, "{:?} at {} (`{}`): {}", self.site, self.location, self.op, self.chosen);
        };
        write!(f, "split at {} (`{}`): ", self.location, self.op)?;
        match reason {
            SplitReason::Nodes { nodes, limit } => write!(f, "{nodes} operations > {limit}"),
            SplitReason::Depth { limit } => write!(f, "depth > {limit}"),
            SplitReason::Inputs { inputs, limit } => write!(f, "{inputs} input buffers > {limit}"),
            SplitReason::Live { live, limit } => write!(f, "{live} live values > {limit}"),
            SplitReason::Scalars { words, limit } => write!(f, "{words} scalar words > {limit}"),
            SplitReason::Inlined { instructions, limit } => write!(f, "{instructions} inlined operations > {limit}"),
            SplitReason::Materialised { instances } => write!(f, "{instances} recomputations of an expression cost more than storing it"),
            SplitReason::Shared { weight } => write!(f, "an expression of {weight} operations read by pending operations outside the kernel is stored, not recomputed in each"),
            #[allow(unreachable_patterns)]
            _ => write!(f, "{}", self.chosen),
        }
    }
}

/// A point where an expression was cut into several kernels, and why.
///
/// Deprecated with [`Profile::splits`]: [`Profile::cuts`] gives the same three values, and
/// `Context::take_decisions` every decision of the planner (`DecisionSite::Cut` is a cut).
#[deprecated(since = "0.5.2", note = "use `Profile::cuts()` (or `Decision` with `DecisionSite::Cut`); removed in 0.6")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FusionSplit {
    /// The operation whose creation (or evaluation) caused the cut.
    pub op: &'static str,
    /// Where that operation was written.
    pub location: &'static Location<'static>,
    /// Why the expression was cut.
    pub reason: SplitReason,
}

#[allow(deprecated)]
impl fmt::Display for FusionSplit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "split at {} (`{}`): ", self.location, self.op)?;
        match self.reason {
            SplitReason::Nodes { nodes, limit } => write!(f, "{nodes} operations > {limit}"),
            SplitReason::Depth { limit } => write!(f, "depth > {limit}"),
            SplitReason::Inputs { inputs, limit } => write!(f, "{inputs} input buffers > {limit}"),
            SplitReason::Live { live, limit } => write!(f, "{live} live values > {limit}"),
            SplitReason::Scalars { words, limit } => write!(f, "{words} scalar words > {limit}"),
            SplitReason::Inlined { instructions, limit } => write!(f, "{instructions} inlined operations > {limit}"),
            SplitReason::Materialised { instances } => write!(f, "{instances} recomputations of an expression cost more than storing it"),
            SplitReason::Shared { weight } => write!(f, "an expression of {weight} operations read by pending operations outside the kernel is stored, not recomputed in each"),
            #[allow(unreachable_patterns)]
            _ => write!(f, "{:?}", self.reason),
        }
    }
}

/// GPU durations per kernel, measured with timestamp queries.
///
/// A duration is the time from the start to the end of the compute pass of one launch, so it
/// includes the GPU's own scheduling of that pass but not the time between launches or any host
/// time. The planner's decisions of the profiled region are not part of it: they are read with
/// `Context::take_decisions` after `Context::finish_profiling`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[allow(deprecated)]
pub struct Profile {
    /// One entry per kernel label, the one with the most total time first.
    pub kernels: Vec<KernelTiming>,
    /// Where lazy expressions were cut into several kernels while profiling, in the order they
    /// happened (see [`SplitReason`]). Read them with [`Profile::cuts`].
    #[deprecated(since = "0.5.2", note = "use `Profile::cuts()` (or `Context::take_decisions()`); removed in 0.6")]
    pub splits: Vec<FusionSplit>,
}

impl Profile {
    /// The cuts of lazy expressions into several kernels, with their operation, location and
    /// reason, in the order they happened.
    #[allow(deprecated)]
    pub fn cuts(&self) -> impl Iterator<Item = (&'static str, &'static Location<'static>, SplitReason)> + '_ {
        self.splits.iter().map(|s| (s.op, s.location, s.reason))
    }

    /// The entry of the kernel with this label.
    pub fn get(&self, label: &str) -> Option<&KernelTiming> {
        self.kernels.iter().find(|k| k.label == label)
    }

    /// Total GPU time of all measured launches.
    pub fn total(&self) -> Duration {
        self.kernels.iter().map(|k| k.total).sum()
    }

    /// Number of measured launches.
    pub fn launches(&self) -> u32 {
        self.kernels.iter().map(|k| k.launches).sum()
    }

    /// Adds the measured launches to the per-kernel totals (merging into the kernels already
    /// there) and sorts the kernels by total time.
    pub fn add_samples(&mut self, samples: impl IntoIterator<Item = (Arc<str>, Duration)>) {
        for (label, duration) in samples {
            match self.kernels.iter_mut().find(|k| k.label == *label) {
                Some(k) => {
                    k.launches += 1;
                    k.total += duration;
                    k.min = k.min.min(duration);
                    k.max = k.max.max(duration);
                }
                None => self.kernels.push(KernelTiming {
                    label: label.to_string(),
                    launches: 1,
                    total: duration,
                    min: duration,
                    max: duration,
                }),
            }
        }
        self.kernels.sort_by_key(|k| std::cmp::Reverse(k.total));
    }
}

#[allow(deprecated)]
impl fmt::Display for Profile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let width = self.kernels.iter().map(|k| k.label.len()).max().unwrap_or(0).max(6);
        writeln!(f, "{:<width$} {:>9} {:>12} {:>12} {:>12} {:>12}", "kernel", "launches", "total us", "mean us", "min us", "max us")?;
        let us = |d: Duration| d.as_secs_f64() * 1e6;
        for k in &self.kernels {
            writeln!(f, "{:<width$} {:>9} {:>12.2} {:>12.2} {:>12.2} {:>12.2}", k.label, k.launches, us(k.total), us(k.mean()), us(k.min), us(k.max))?;
        }
        for split in &self.splits {
            writeln!(f, "{split}")?;
        }
        Ok(())
    }
}

/// An error of a backend; the root crate converts it into its own error type.
#[derive(Debug, Clone)]
pub enum BackendError {
    /// No suitable GPU adapter was found.
    NoAdapter(String),
    /// The requested kind of device is not available.
    NoDevice(&'static str),
    /// The adapter refused to create a device.
    RequestDevice(String),
    /// A kernel failed to parse, validate or compile; holds the compiler's message.
    ShaderCompile(String),
    /// A kernel declares resources that cannot be bound.
    KernelBinding(String),
    /// The requested entry point does not exist or is not a compute entry point.
    KernelEntryPoint(String),
    /// The workgroup size is invalid for the device or disagrees with the shader.
    KernelWorkgroup(String),
    /// A kernel was dispatched with the wrong number of arguments.
    KernelArgCount {
        /// Number of bindings the kernel declares.
        expected: usize,
        /// Number of arguments that were passed.
        actual: usize,
    },
    /// Waiting for the GPU to finish failed.
    Poll(String),
    /// Mapping a buffer failed.
    Map(String),
    /// Accessing a mapped buffer failed.
    MapRange(String),
    /// A buffer allocation failed because the device is out of memory.
    OutOfMemory {
        /// Size of the failed allocation in bytes.
        requested: u64,
    },
    /// GPU timing is not available or not enabled.
    Profiling(String),
    /// The GPU reported an error while running commands.
    Gpu(String),
    /// The operation does not exist on this device.
    Unsupported(String),
}

impl fmt::Display for BackendError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BackendError::NoAdapter(e) => write!(f, "Found no adapters: {e}"),
            BackendError::NoDevice(e) => write!(f, "No suitable device: {e}"),
            BackendError::RequestDevice(e) => write!(f, "Error requesting device: {e}"),
            BackendError::ShaderCompile(msg) => write!(f, "Shader failed to compile:\n{msg}"),
            BackendError::KernelBinding(msg) => write!(f, "Unsupported kernel binding: {msg}"),
            BackendError::KernelEntryPoint(msg) => f.write_str(msg),
            BackendError::KernelWorkgroup(msg) => write!(f, "Invalid workgroup size: {msg}"),
            BackendError::KernelArgCount { expected, actual } => write!(f, "Kernel takes {expected} arguments, got {actual}"),
            BackendError::Poll(e) => write!(f, "GPU poll failed: {e}"),
            BackendError::Map(e) => write!(f, "Failed to read data back from GPU: {e}"),
            BackendError::MapRange(e) => write!(f, "Failed to map staging buffer: {e}"),
            BackendError::OutOfMemory { requested } => write!(f, "Out of GPU memory allocating {requested} bytes"),
            BackendError::Profiling(msg) => write!(f, "GPU profiling: {msg}"),
            BackendError::Gpu(msg) => write!(f, "GPU error: {msg}"),
            BackendError::Unsupported(what) => f.write_str(what),
        }
    }
}

impl std::error::Error for BackendError {}

/// How a buffer may be used.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct BufferUsage(u8);

impl BufferUsage {
    /// No usage.
    pub const NONE: Self = Self(0);
    /// Bound as a storage buffer.
    pub const STORAGE: Self = Self(1 << 0);
    /// Source of copies.
    pub const COPY_SRC: Self = Self(1 << 1);
    /// Destination of copies.
    pub const COPY_DST: Self = Self(1 << 2);
    /// Bound as a uniform buffer.
    pub const UNIFORM: Self = Self(1 << 3);
    /// Holds indirect dispatch arguments.
    pub const INDIRECT: Self = Self(1 << 4);
    /// Mappable for reading by the host.
    pub const MAP_READ: Self = Self(1 << 5);
    /// Mappable for writing by the host.
    pub const MAP_WRITE: Self = Self(1 << 6);
    /// A texture ([`Backend::create_texture`]): only a texture or sampler binding takes it.
    pub const TEXTURE: Self = Self(1 << 7);

    /// No usage.
    pub const fn empty() -> Self {
        Self(0)
    }
    /// Whether every usage of `other` is set.
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
    /// The usages of both.
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }
    /// Whether no usage is set.
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }
    /// The raw bits.
    pub const fn bits(self) -> u8 {
        self.0
    }
}

/// Copies between buffers and writes into them are multiples of this many bytes.
pub const COPY_BUFFER_ALIGNMENT: u64 = 4;

/// Smallest buffer size class.
const MIN_BUFFER_SIZE: u64 = 16;

/// Buffers up to this many bytes round to a power of two; larger ones to an eighth-octave class.
const FINE_CLASS_START: u64 = 4096;

/// Size class for a buffer holding `bytes` bytes, the one rule every backend and the pool share.
///
/// Up to 4 KiB: the next power of two (at least 16 bytes). Above: the next multiple of
/// `next_power_of_two(bytes) / 16`, which cuts each octave into 8 steps (at most 12.5% waste,
/// every class a multiple of 512 bytes). Clamped to `max_size & !3`; `None` if `bytes` itself
/// exceeds that.
///
/// Buffers of a class may be larger than the data they hold, so users must track the logical
/// size themselves and never rely on `Buffer::size()` / WGSL `arrayLength`.
pub fn size_class(bytes: u64, max_size: u64) -> Option<u64> {
    // Keep copy sizes aligned even when the limit is not.
    let max_size = max_size & !(COPY_BUFFER_ALIGNMENT - 1);
    if bytes > max_size {
        return None;
    }
    let bytes = bytes.max(MIN_BUFFER_SIZE);
    let Some(pow2) = bytes.checked_next_power_of_two() else {
        return Some(max_size);
    };
    let class = if bytes <= FINE_CLASS_START {
        pow2
    } else {
        let step = pow2 / 16;
        bytes.next_multiple_of(step)
    };
    Some(class.min(max_size))
}

static NEXT_BUFFER_ID: AtomicU64 = AtomicU64::new(1);
static NEXT_KERNEL_ID: AtomicU64 = AtomicU64::new(1);

/// A backend's own buffer object, held type-erased by [`Buffer`].
pub trait NativeBuffer: Any + fmt::Debug + Send + Sync + 'static {}

/// Host memory a backend may show as a buffer without copying (see [`Backend::import_host`]).
pub struct HostRegion {
    /// First byte; 64-byte aligned for [`HostBytes`].
    pub ptr: *mut u8,
    /// Bytes.
    pub len: u64,
    /// Whether kernels may write the region.
    pub writable: bool,
    /// Held by the buffer until it is dropped: what owns the memory.
    pub keepalive: Arc<dyn Any + Send + Sync>,
}

// SAFETY: the pointer is plain data; `keepalive` owns the memory and the importer guarantees exclusive access while a writable buffer exists.
unsafe impl Send for HostRegion {}

const BORROWED_ID: u64 = 1 << 63;

/// A device buffer of any backend. Cloning shares the buffer; equality is identity (a view made
/// by [`Buffer::at`] is equal to the buffer it views).
#[derive(Clone, Debug)]
pub struct Buffer {
    id: u64,
    size: u64,
    usage: BufferUsage,
    native: Arc<dyn NativeBuffer>,
    offset: u64,
}

impl Buffer {
    /// Wraps a backend's buffer of `size` bytes.
    pub fn new(size: u64, usage: BufferUsage, native: impl NativeBuffer) -> Self {
        Buffer { id: NEXT_BUFFER_ID.fetch_add(1, Ordering::Relaxed), size, usage, native: Arc::new(native), offset: 0 }
    }
    /// Wraps a backend's buffer over host memory it does not own (see [`Backend::import_host`]):
    /// a pool must not take it back for reuse ([`Buffer::is_borrowed`]).
    pub fn new_borrowed(size: u64, usage: BufferUsage, native: impl NativeBuffer) -> Self {
        // Marked by the top bit of the id (ids count up from 0), so that the struct does not grow for it.
        let mut buffer = Self::new(size, usage, native);
        buffer.id |= BORROWED_ID;
        buffer
    }
    /// Whether the buffer shows memory the backend does not own (made by [`Buffer::new_borrowed`]).
    pub fn is_borrowed(&self) -> bool {
        self.id & BORROWED_ID != 0
    }
    /// A view of this buffer that a launch binds from `offset` bytes on (a storage argument of a
    /// backend whose [`Backend::storage_offset_align`] is not 0, with `offset` a multiple of it;
    /// [`CompiledKernel::check_offsets`] checks). The view is the same buffer for everything
    /// else: same id, same [`Buffer::size`] (the whole buffer), same memory.
    #[must_use]
    pub fn at(&self, offset: u64) -> Buffer {
        Buffer { offset: self.offset + offset, ..self.clone() }
    }
    /// Where a launch starts binding this buffer: 0 except for a view made by [`Buffer::at`].
    pub fn offset(&self) -> u64 {
        self.offset
    }
    /// Unique per buffer.
    pub fn id(&self) -> u64 {
        self.id
    }
    /// Size in bytes.
    pub fn size(&self) -> u64 {
        self.size
    }
    /// The usages the buffer was created with.
    pub fn usage(&self) -> BufferUsage {
        self.usage
    }
    /// The backend's buffer, if it is a `T`.
    pub fn downcast_ref<T: 'static>(&self) -> Option<&T> {
        (&*self.native as &dyn Any).downcast_ref::<T>()
    }
}

impl PartialEq for Buffer {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for Buffer {}

impl std::hash::Hash for Buffer {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

/// A backend's own compiled kernel object, held type-erased by [`RawKernel`].
pub trait NativeKernel: Any + fmt::Debug + Send + Sync + 'static {}

/// A compiled kernel of any backend, possibly still compiling (see [`RawKernel::spawn`]).
#[derive(Clone, Debug)]
pub struct RawKernel {
    id: u64,
    native: Arc<deferred::Slot>,
}

impl RawKernel {
    /// Wraps a backend's kernel.
    pub fn new(native: impl NativeKernel) -> Self {
        RawKernel { id: NEXT_KERNEL_ID.fetch_add(1, Ordering::Relaxed), native: Arc::new(deferred::Slot::ready(native)) }
    }
    /// A kernel `compile` produces on a worker thread (on the calling thread when the pool is off
    /// or its queue is full, see `LUMA_COMPILE_THREADS`). It returns at once; whatever needs the
    /// kernel to run calls [`resolve`](Self::resolve), which waits and reports the compile error.
    pub fn spawn<N: NativeKernel>(compile: impl FnOnce() -> Result<N, BackendError> + Send + 'static) -> Self {
        RawKernel { id: NEXT_KERNEL_ID.fetch_add(1, Ordering::Relaxed), native: deferred::spawn(compile) }
    }
    /// Unique per compiled kernel.
    pub fn id(&self) -> u64 {
        self.id
    }
    /// Whether the kernel is compiled (or failed to compile): a launch of it would not wait.
    pub fn is_ready(&self) -> bool {
        self.native.is_ready()
    }
    /// Whether the compilation has finished with an error (never waits).
    pub fn failed(&self) -> bool {
        self.native.is_ready() && self.native.wait().is_err()
    }
    /// The backend's kernel, once compiled: waits for the compilation, fails with its error, or
    /// with [`BackendError::Unsupported`] if the kernel is not a `T`.
    pub fn resolve<T: 'static>(&self) -> Result<&T, BackendError> {
        match self.native.wait() {
            Ok(native) => (&**native as &dyn Any).downcast_ref::<T>().ok_or_else(|| BackendError::Unsupported("the kernel belongs to another backend".into())),
            Err(error) => Err(error.clone()),
        }
    }
    /// The backend's kernel, if it compiled and is a `T`. Waits for the compilation.
    pub fn downcast_ref<T: 'static>(&self) -> Option<&T> {
        self.resolve().ok()
    }
}

/// Everything about a built kernel that does not depend on the caller.
#[derive(Clone, Debug)]
pub struct CompiledKernel {
    /// Unique per built kernel; identifies it in bind group caches.
    pub id: u64,
    /// Names the kernel in a [`Profile`].
    pub label: Arc<str>,
    /// The backend's kernel.
    pub raw: RawKernel,
    /// The bindings the entry point uses, by binding number.
    pub bindings: Vec<BindingInfo>,
    /// The arguments a launch passes: one per `@group(1)` variable the module declares.
    pub declared: usize,
    /// Which of the `declared` arguments the entry point reads, `None` when all of them. A
    /// variable no code reads has no binding in the pipeline layout, so a launch must not bind it.
    pub used: Option<Box<[bool]>>,
    /// The workgroup size.
    pub workgroup: [u32; 3],
}

static NEXT_COMPILED_ID: AtomicU64 = AtomicU64::new(1);

impl CompiledKernel {
    /// Wraps a built kernel, allocating its unique id.
    pub fn new(label: Arc<str>, raw: RawKernel, bindings: Vec<BindingInfo>, declared: usize, used: Option<Box<[bool]>>, workgroup: [u32; 3]) -> Self {
        CompiledKernel { id: NEXT_COMPILED_ID.fetch_add(1, Ordering::Relaxed), label, raw, bindings, declared, used, workgroup }
    }

    /// The arguments of a launch that the kernel binds, with their binding numbers.
    pub fn bound<'a, T>(&'a self, args: &'a [T]) -> impl Iterator<Item = (u32, &'a T)> + 'a {
        args.iter()
            .enumerate()
            .filter(|&(position, _)| self.used.as_ref().is_none_or(|used| used[position]))
            .map(|(position, arg)| (position as u32, arg))
    }

    /// Checks that a launch passes as many arguments as the module declares. A mismatch is an
    /// error of that launch; left to the device it would be a validation failure that
    /// invalidates the work of every thread sharing the batch.
    pub fn check_count(&self, passed: usize) -> Result<(), BackendError> {
        if passed != self.declared {
            return Err(BackendError::KernelArgCount { expected: self.declared, actual: passed });
        }
        Ok(())
    }

    /// Checks that each storage argument that is a view with an offset (see [`Buffer::at`]) lies
    /// within its buffer and starts at a multiple of `align` (the device's
    /// [`Backend::storage_offset_align`]; `0` allows no offset).
    pub fn check_offsets(&self, args: &[RawArg<'_>], align: u64) -> Result<(), BackendError> {
        // Almost no launch binds a view: one pass over the arguments that finds none.
        if args.iter().all(|arg| !matches!(arg, RawArg::Storage { buffer, .. } if buffer.offset() != 0)) {
            return Ok(());
        }
        self.check_offsets_of_views(args, align)
    }

    #[cold]
    #[inline(never)]
    fn check_offsets_of_views(&self, args: &[RawArg<'_>], align: u64) -> Result<(), BackendError> {
        for (position, arg) in self.bound(args) {
            let RawArg::Storage { buffer, bytes, .. } = arg else { continue };
            let offset = buffer.offset();
            if offset == 0 {
                continue;
            }
            if align == 0 || offset % align != 0 {
                return Err(BackendError::KernelBinding(format!(
                    "kernel `{}` argument {position}: offset {offset} is not a multiple of the device's storage offset alignment ({align}; 0: no offsets)",
                    self.label
                )));
            }
            if offset.checked_add(*bytes).is_none_or(|end| end > buffer.size()) {
                return Err(BackendError::KernelBinding(format!(
                    "kernel `{}` argument {position}: range {offset}+{bytes} exceeds the buffer ({} bytes)",
                    self.label,
                    buffer.size()
                )));
            }
        }
        Ok(())
    }

    /// Checks that each argument the kernel binds is of the kind of its binding, for the same
    /// reason as [`check_count`](Self::check_count).
    pub fn check_kinds(&self, args: &[RawArg<'_>]) -> Result<(), BackendError> {
        let mut infos = self.bindings.iter();
        for (position, arg) in args.iter().enumerate() {
            if self.used.as_ref().is_some_and(|used| !used[position]) {
                continue;
            }
            let Some(info) = infos.next() else { break };
            // Whether a storage buffer is passed for reading or writing does not matter to the
            // layout; a uniform where a storage buffer is expected (or the reverse) does.
            let fits = match arg {
                RawArg::Uniform(_) => info.kind == BindingKind::Uniform,
                RawArg::Storage { buffer, .. } if buffer.usage().contains(BufferUsage::TEXTURE) => matches!(info.kind, BindingKind::Texture2d | BindingKind::Sampler),
                RawArg::Storage { .. } => matches!(info.kind, BindingKind::StorageRead | BindingKind::StorageReadWrite),
            };
            if !fits {
                return Err(self.kind_error(position as u32, arg, info));
            }
        }
        Ok(())
    }

    #[cold]
    #[inline(never)]
    fn kind_error(&self, position: u32, arg: &RawArg<'_>, info: &BindingInfo) -> BackendError {
        let passed = match arg {
            RawArg::Uniform(_) => "uniform",
            RawArg::Storage { buffer, .. } if buffer.usage().contains(BufferUsage::TEXTURE) => "texture",
            RawArg::Storage { .. } => "storage buffer",
        };
        BackendError::KernelBinding(format!(
            "kernel `{}` argument {position} (`{}`) is a {}, but the launch passes a {passed}",
            self.label, info.name, info.kind
        ))
    }
}

/// One argument of a launch.
#[derive(Clone, Copy, Debug)]
pub enum RawArg<'a> {
    /// A storage buffer, of which the kernel may reach the first `bytes` bytes from the start
    /// of the buffer, or, for a view made by [`Buffer::at`], from the view's offset
    /// ([`Buffer::offset`]).
    Storage {
        /// The buffer (or a view of a range of it).
        buffer: &'a Buffer,
        /// Bytes bound.
        bytes: u64,
        /// Whether the kernel may write it.
        writable: bool,
    },
    /// Uniform data, copied at launch.
    Uniform(&'a [u8]),
}

impl<'a> RawArg<'a> {
    /// A read-only storage argument.
    pub fn read(buffer: &'a Buffer, bytes: u64) -> Self {
        RawArg::Storage { buffer, bytes, writable: false }
    }
    /// A writable storage argument.
    pub fn write(buffer: &'a Buffer, bytes: u64) -> Self {
        RawArg::Storage { buffer, bytes, writable: true }
    }
}

/// How many workgroups a launch runs.
#[derive(Clone, Copy, Debug)]
pub enum Dispatch<'a> {
    /// Workgroup counts known on the host.
    Direct([u32; 3]),
    /// Workgroup counts read from a buffer at `offset` (three `u32`s) when the GPU gets there.
    Indirect {
        /// The buffer holding the counts.
        buffer: &'a Buffer,
        /// Byte offset of the counts.
        offset: u64,
        /// An upper bound of the counts, for backends that must size the grid up front.
        max_workgroups: [u32; 3],
    },
}

/// One launch of a captured graph.
#[derive(Clone, Copy, Debug)]
pub struct GraphLaunch<'a> {
    /// The kernel.
    pub kernel: &'a CompiledKernel,
    /// How many workgroups run: counts known now, or counts read from a buffer when the graph
    /// is replayed (the buffer is kept alive by the graph and may be rewritten between and
    /// during replays, by earlier launches of the same graph too).
    pub dispatch: Dispatch<'a>,
    /// The logical grid size in invocations.
    pub size: [u32; 3],
    /// Arguments of the launch.
    pub args: &'a [RawArg<'a>],
}

/// A recorded sequence of launches a backend can replay cheaply.
pub trait ExecGraph: Any + fmt::Debug + Send + Sync + 'static {
    /// Bytes of device memory the graph holds itself (a copy of the uniforms it passes by
    /// pointer, say), not counting the buffers its launches name nor what the driver keeps
    /// inside the graph. Zero by default.
    fn device_bytes(&self) -> u64 {
        0
    }
}

impl dyn ExecGraph {
    /// The backend's graph, if it is a `T`.
    pub fn downcast_ref<T: 'static>(&self) -> Option<&T> {
        (self as &dyn Any).downcast_ref::<T>()
    }
}

/// Limits of a device.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DeviceLimits {
    /// Largest buffer, bytes.
    pub max_buffer_bytes: u64,
    /// Most storage buffers one kernel binds.
    pub max_storage_buffers: u32,
    /// Largest uniform binding, bytes.
    pub max_uniform_bytes: u64,
    /// Alignment of uniform offsets, bytes.
    pub uniform_align: u64,
    /// Most dynamic-offset uniform bindings.
    pub max_dynamic_uniforms: u32,
    /// Most workgroups per dispatch dimension.
    pub max_workgroups_per_dim: u32,
    /// Largest workgroup size per dimension.
    pub max_workgroup_size: [u32; 3],
    /// Most invocations per workgroup.
    pub max_workgroup_invocations: u32,
    /// Workgroup shared memory, bytes.
    pub max_workgroup_storage_bytes: u32,
}

/// Live device memory, from [`Backend::memory_now`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MemoryNow {
    /// Bytes free.
    pub free: u64,
    /// Bytes in total.
    pub total: u64,
}

/// What a backend knows about its device.
#[derive(Clone, Debug)]
pub struct DeviceInfo {
    /// The backend.
    pub backend: BackendKind,
    /// The device's name.
    pub name: String,
    /// Whether the device is a software one.
    pub cpu: bool,
    /// Whether the host and device share memory.
    pub zero_copy: bool,
    /// The tuning constants to use.
    pub profile: TuningProfile,
    /// Optional capabilities.
    pub features: Features,
    /// Smallest and largest subgroup size, if subgroups are supported.
    pub subgroup_size: Option<(u32, u32)>,
    /// Device memory, bytes, if known.
    pub memory_bytes: Option<u64>,
    /// Whether `f32` add/sub/mul are IEEE-exact (never fused or reassociated) without guards.
    pub ieee_f32_intrinsics: bool,
    /// Whether the device fuses `fma`, if known.
    pub hardware_fma: Option<bool>,
    /// The device's limits.
    pub limits: DeviceLimits,
    /// Multiprocessors (NVIDIA SMs, AMD compute units, Apple GPU cores) or, for the CPU backend,
    /// worker threads, if the backend can tell. Plan constants that depend on how many workgroups
    /// fill the device scale with it.
    pub compute_units: Option<u32>,
}

/// How much of a kernel's behaviour a backend may take on trust when it compiles it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum KernelTrust {
    /// Untrusted code (user WGSL): every memory access is checked against its bounds and
    /// workgroup memory starts zeroed.
    Checked,
    /// User code whose author vouches that every access is in bounds (see
    /// `KernelBuilder::unchecked_bounds`): no bounds checks, but workgroup memory still starts
    /// zeroed.
    UncheckedBounds,
    /// Luma's own kernels, in bounds by construction and writing workgroup memory before reading it.
    Trusted,
}

impl KernelTrust {
    /// Whether memory accesses are checked.
    pub fn checks_bounds(self) -> bool {
        self == KernelTrust::Checked
    }

    /// Whether workgroup memory is zeroed before the kernel runs.
    pub fn zero_workgroup_memory(self) -> bool {
        self != KernelTrust::Trusted
    }
}

/// A block-tridiagonal line solve (the block Thomas algorithm) that a backend may run as native
/// code instead of compiling the shader that describes it. The launch is the shader's: the same
/// bindings in the same order, the same uniform, one workgroup of `lines_per_group` invocations
/// per `lines_per_group` lines (the line `l` of invocation `i` of workgroup `g` is
/// `g * lines_per_group + i`, the groups of a folded grid numbered row by row).
///
/// Bindings: the outputs, the scratch (a buffer the native solve may ignore), the inputs, `starts`
/// if `starts`, then the uniform `{n, stride, lines}` (`u32`s). Stacked blocks (`parts` false) have
/// one output `out` of `block` floats per cell and the inputs `lo`, `diag`, `up` (`block^2` floats
/// per cell) and `rhs`. On `parts` every entry of every block and component is a buffer of its
/// own with one float per cell: the outputs `out{system * block + component}`, the inputs of every
/// system in the order `lo`, `diag`, `up` (entry `row * block + col`), `rhs`. Cell `q` of line
/// `l` is `l % stride + (l / stride) * stride * n + s * stride` for the row `s`; the lines of
/// `systems` systems are numbered one system after the other (`lines` per system in the uniform).
/// With `starts` row `s` of line `l` below `starts[l]` (clamped to `n`) is not solved and its
/// solution is 0.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct LineSolveSpec {
    /// The block size, `1..=8`.
    pub block: u8,
    /// Independent systems solved in one launch (1 unless `parts`).
    pub systems: u8,
    /// Whether the entries are separate buffers (see above) rather than stacked blocks.
    pub parts: bool,
    /// Whether every line begins at its own cell.
    pub starts: bool,
    /// Lines per workgroup.
    pub lines_per_group: u32,
}

/// A matrix product that a backend may run as native code instead of compiling the tiled shader
/// that describes it (`Builtin::Matmul` of `luma`). The launch is the shader's: the bindings `c`
/// (written), `a` and `b` (read) and the uniform `{m, n, k, kc, lda, ldb, a_off, b_off}` (`u32`s;
/// the four words after `kc` are those of [`GemmSpec`]: leading dimensions and element offsets of
/// the matrices in their buffers), one workgroup per
/// output tile of `edge` x `edge` (the tiles numbered row by row along the folded grid: tile
/// `x + y * num_workgroups.x`, column tile `tile % ceil(n / edge)`), the third grid dimension
/// slicing K: workgroup layer `z` multiplies `K` from `z * kc` to `min(k, (z + 1) * kc)` and
/// writes its tile of the `M x N` partial product at `z * m * n` of `c`. `a` is the row-major
/// `M x K` matrix (rows `lda` apart, from element `a_off`), `b` the row-major `K x N` one (rows
/// `ldb` apart, from `b_off`), `c` dense row-major `M x N`; `c` is overwritten. With
/// `lhs_transposed` the buffer `a` holds the row-major `K x M` matrix instead (the product reads
/// its transpose), with `rhs_transposed` `b` holds the row-major `N x K` one.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MatmulSpec {
    /// The edge of an output tile.
    pub edge: u32,
    /// Whether `a` holds the transpose of the left matrix.
    pub lhs_transposed: bool,
    /// Whether `b` holds the transpose of the right matrix.
    pub rhs_transposed: bool,
    /// A half-work product of a matrix with itself (`xᵀ x`, `x xᵀ`; square `c`): the grid is `[T (T + 1) / 2, 1, 1]` for `T` tile
    /// rows, workgroup `w` computing the tile `(ty, tx)`, `tx <= ty`, of the closed form [`triangle_tile`], and writing it together
    /// with its transpose at `(tx, ty)`; the output is bit-symmetric. Not combined with slices of K.
    pub half: bool,
}

/// The tile `(row, column)` of workgroup `w` of a half-work product ([`MatmulSpec::half`], [`GemmSpec::half`]): the lower
/// triangle of tiles row by row, `ty = floor((sqrt(8 w + 1) - 1) / 2)`, `tx = w - ty (ty + 1) / 2`, corrected in integers (the float
/// root is exact only below 2^24).
pub fn triangle_tile(w: u32) -> (u32, u32) {
    let w = w as u64;
    let mut ty = (((8.0 * w as f64 + 1.0).sqrt() - 1.0) / 2.0) as u64;
    while ty * (ty + 1) / 2 > w {
        ty -= 1;
    }
    while (ty + 1) * (ty + 2) / 2 <= w {
        ty += 1;
    }
    (ty as u32, (w - ty * (ty + 1) / 2) as u32)
}

/// A matrix product the backend runs as its own tiled kernel (`Builtin::Matmul` of `luma` with a
/// `Gemm` tile): on tensor cores with the operands rounded to TF32 (10 bits of mantissa) and
/// accumulated in `f32`, or with `exact` on the ordinary cores, every output an `f32` chain of
/// fused multiply-adds over `k` in ascending order (per slice of K). The launch is that of
/// [`MatmulSpec`] (the bindings, the slices of K), but one workgroup computes a tile of `rows` x
/// `cols` outputs and the grid is `[column tiles, row tiles, slices]` (never folded).
///
/// The uniform is eight `u32` words, `{m, n, k, kc, lda, ldb, a_off, b_off}`: `lda` and `ldb` are
/// the leading dimensions in elements of the matrices as the buffers hold them (the distance
/// between the starts of their rows: `k` for an `M x K` left matrix, `m` for its transpose, `n`
/// for a `K x N` right matrix, `k` for its transpose, in a dense product), `a_off` and `b_off`
/// the element offsets of their first element in the buffers. `c` is dense, `[m, n]` row-major.
/// The products of [`GemmLanes::Vec4`] need `m`, `n` and `k` and the leading dimensions and
/// offsets to be multiples of 4 (16-byte copies); [`GemmLanes::Vec4Tails`] only the leading
/// dimensions and offsets; [`GemmLanes::Scalar`] nothing. Every lane gives the same bits for the
/// same slices of K.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct GemmSpec {
    /// Rows of an output tile.
    pub rows: u32,
    /// Columns of an output tile.
    pub cols: u32,
    /// Whether `a` holds the transpose of the left matrix.
    pub lhs_transposed: bool,
    /// Whether `b` holds the transpose of the right matrix.
    pub rhs_transposed: bool,
    /// `f32` multiply-adds instead of TF32 tensor cores.
    pub exact: bool,
    /// How the kernel copies the operands, and so which shapes it takes.
    pub lanes: GemmLanes,
    /// A half-work product of a matrix with itself (see [`MatmulSpec::half`]): `rows == cols`, the grid `[T (T + 1) / 2, 1, 1]`,
    /// every output of the lower triangle of tiles computed as the full kernel does and stored with its transpose.
    pub half: bool,
}

/// How a [`GemmSpec`] kernel copies its operands from global memory.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum GemmLanes {
    /// 16-byte copies; `m`, `n`, `k`, the leading dimensions and the offsets are multiples of 4.
    Vec4,
    /// 16-byte copies whose chunks that cross the edge of the matrix are cut (the rest of the chunk is zeros), and stores
    /// that guard the edge; the leading dimensions and offsets are multiples of 4, the extents are any.
    Vec4Tails,
    /// 4-byte copies and stores, each guarded; any extents, leading dimensions and offsets.
    Scalar,
}

impl GemmSpec {
    /// A product on tiles of `rows` x `cols`, `lhs_transposed` / `rhs_transposed` if the operand's buffer holds the matrix
    /// transposed, in `f32` (`exact`) or TF32, on [`GemmLanes::Vec4`]. Build a spec with this, not with a struct literal: the struct grows fields.
    pub const fn new(rows: u32, cols: u32, lhs_transposed: bool, rhs_transposed: bool, exact: bool) -> Self {
        GemmSpec { rows, cols, lhs_transposed, rhs_transposed, exact, lanes: GemmLanes::Vec4, half: false }
    }

    /// The same product copying its operands with `lanes`.
    pub const fn with_lanes(self, lanes: GemmLanes) -> Self {
        GemmSpec { lanes, ..self }
    }

    /// The same product as a half-work one (see [`GemmSpec::half`]).
    pub const fn with_half(self) -> Self {
        GemmSpec { half: true, ..self }
    }
}

/// A GPU backend: allocates buffers, compiles kernels and records and runs work.
pub trait Backend: fmt::Debug + Send + Sync + 'static {
    /// What is known about the device.
    fn info(&self) -> &DeviceInfo;
    /// L2 cache size, bytes, if the backend can tell (CUDA); `None` by default.
    fn l2_cache_bytes(&self) -> Option<u64> {
        None
    }
    /// Alignment of the offset of a bound storage range, bytes: a [`Buffer::at`] view with an
    /// offset that is a multiple of it may be passed as a storage argument ([`RawArg`]). `0`
    /// (the default) means the backend binds a buffer only from its start and never gets a view
    /// with an offset.
    fn storage_offset_align(&self) -> u64 {
        0
    }
    /// Free and total device memory right now, where the API reports it (`None` otherwise).
    fn memory_now(&self) -> Option<MemoryNow> {
        None
    }
    /// The device's on-disk cache.
    fn disk_cache(&self) -> &Arc<DiskCache>;
    /// The device's cost model, set once calibrated.
    fn device_cost(&self) -> &Arc<OnceLock<CostModel>>;
    /// Allocates a buffer, `None` when the device is out of memory; `mapped` buffers start
    /// mapped for writing.
    fn create_buffer(&self, size: u64, usage: BufferUsage, mapped: bool) -> Option<Buffer>;
    /// A buffer that shows `region` (host memory the caller owns) without copying it, `None` when the
    /// backend cannot (the default), the address is not aligned for it, or the region is empty. The
    /// buffer keeps `region.keepalive` until it is dropped and is [borrowed](Buffer::is_borrowed).
    fn import_host(&self, _region: HostRegion) -> Option<Buffer> {
        None
    }
    /// What the backend has learned a launch of `workgroups` workgroups of `kernel` costs, in
    /// nanoseconds, `None` before it ran or when the backend does not learn (the default).
    fn learned_ns(&self, _kernel: &CompiledKernel, _workgroups: u64) -> Option<f64> {
        None
    }
    /// Allocates a buffer holding `data`.
    fn alloc_filled(&self, data: &[u8], max_buffer_bytes: u64) -> Result<Buffer, BackendError>;
    /// Allocates a buffer that is the destination or source of [`Backend::transfer_in`] /
    /// [`Backend::transfer_out`]: one whose free is not ordered on the compute lane (on CUDA a
    /// stream-ordered pool block could be freed while the transfer lane still copies). The
    /// default is [`Backend::create_buffer`].
    fn create_buffer_unordered(&self, size: u64, usage: BufferUsage) -> Option<Buffer> {
        self.create_buffer(size, usage, false)
    }
    /// Returns a `MAP_WRITE` buffer for reuse.
    fn recycle_mappable(&self, buffer: Buffer);
    /// Host pointer to the first `bytes` of `buffer` when directly CPU-accessible and idle (no
    /// pending or running GPU commands touch it).
    fn host_ptr(&self, _buffer: &Buffer, _bytes: u64) -> Option<*mut u8> {
        None
    }
    /// A mappable storage buffer of at least `bytes` bytes, uninitialized; recycled with
    /// [`Backend::recycle_mappable`]. Only called when [`DeviceInfo::zero_copy`] is true.
    fn alloc_host_buffer(&self, _bytes: u64, _max_buffer_bytes: u64) -> Result<Buffer, BackendError> {
        Err(BackendError::Unsupported("the backend has no mappable storage buffers".into()))
    }
    /// What the backend's textures can do, `None` when it has none (see [`Backend::create_texture`]).
    fn texture_caps(&self) -> Option<TextureCaps> {
        None
    }
    /// A texture holding the first `desc.width` x `desc.height` texels of the row-major data of
    /// `source` at byte `offset`, rows `pitch` bytes apart (a multiple of
    /// [`TextureCaps::pitch_align`], at least a row), in the order of the recorded work: it shows
    /// what the work recorded so far wrote to `source`. The texture is returned as a [`Buffer`]
    /// that only a kernel binding of kind [`BindingKind::Texture2d`] or [`BindingKind::Sampler`]
    /// accepts, and that a launch keeps alive until it has run. The backend may let the texture
    /// read `source` itself (the data must not change while a launch that samples the texture
    /// is pending) or copy it.
    fn create_texture(&self, _source: &Buffer, _offset: u64, _pitch: u64, _desc: &TextureDesc) -> Result<Buffer, BackendError> {
        Err(BackendError::Unsupported("the backend has no textures".into()))
    }
    /// Compiles a module; `used` is the entry point's binding mask, see [`CompiledKernel::used`].
    fn compile_kernel(&self, module: &crate::kir::KernelModule, used: Option<&[bool]>, label: &str, trust: KernelTrust) -> Result<RawKernel, BackendError>;
    /// Like [`compile_kernel`](Backend::compile_kernel), but the compilation may carry on in the
    /// background: a failure of the slow part then surfaces from the first use of the kernel (see
    /// [`RawKernel::resolve`]) instead of from here. For kernels luma builds itself, which no
    /// caller needs before a launch; the default compiles at once.
    fn compile_kernel_deferred(&self, module: crate::kir::KernelModule, used: Option<Box<[bool]>>, label: &str, trust: KernelTrust) -> Result<RawKernel, BackendError> {
        self.compile_kernel(&module, used.as_deref(), label, trust)
    }
    /// Compiles an already parsed naga module, `None` if the backend takes only [`KernelModule`](crate::kir::KernelModule)s.
    fn compile_naga(&self, _module: &naga::Module, _entry: &str, _bindings: &[BindingInfo], _workgroup: [u32; 3], _label: &str, _trust: KernelTrust) -> Option<Result<RawKernel, BackendError>> {
        None
    }
    /// Records a launch into the pending batch; returns how many dispatches are pending.
    fn record_launch(&self, kernel: &CompiledKernel, dispatch: Dispatch<'_>, size: [u32; 3], args: &[RawArg<'_>]) -> Result<u32, BackendError>;
    /// Records zeroing the first `bytes` of `buffer`.
    fn record_clear(&self, buffer: &Buffer, bytes: u64) -> Result<(), BackendError>;
    /// Records a buffer to buffer copy.
    fn record_copy(&self, src: &Buffer, src_offset: u64, dst: &Buffer, dst_offset: u64, bytes: u64) -> Result<(), BackendError>;
    /// Records writing `data` into `dst`.
    fn record_upload(&self, dst: &Buffer, data: &[u8], max_buffer_bytes: u64) -> Result<(), BackendError>;
    /// Records a replay of a captured graph counting as `dispatches` dispatches.
    fn record_graph(&self, graph: &dyn ExecGraph, dispatches: u32) -> Result<u32, BackendError>;
    /// Dispatches recorded and not yet submitted.
    fn pending_dispatches(&self) -> u32;
    /// Whether submitted work is still running.
    fn gpu_busy(&self) -> bool;
    /// Tells the backend that these buffers were released by their owners.
    fn notify_released(&self, buffers: &[Buffer]);
    /// Submits the pending batch.
    fn flush_batch(&self) -> Result<(), BackendError>;
    /// Submits and waits for all work; returns how long the wait took.
    fn sync(&self, spin_budget: Duration) -> Result<Duration, BackendError>;
    /// Reads `src` into `dst`; `spill` allows a larger-than-memory chunked readback.
    fn read_into(&self, src: &Buffer, dst: &mut [MaybeUninit<u8>], spill: bool, spin_budget: Duration) -> Result<Duration, BackendError>;
    /// Host memory of `bytes` bytes for transfers in direction `dir`, zeroed. The default is heap
    /// memory; CUDA hands out pinned memory held until the stage is dropped.
    fn stage_alloc(&self, bytes: u64, dir: StageDir) -> Result<Stage, BackendError> {
        Ok(Stage::heap(bytes, dir))
    }
    /// The bytes of `stage`, valid while the host owns it.
    fn stage_ptr<'a>(&self, stage: &'a mut Stage) -> &'a mut [u8] {
        stage.as_mut_slice()
    }
    /// Copies the first `bytes` bytes of `stage` into `dst` at `dst_offset` once every ticket of
    /// `after` has completed, without waiting for the copy. `dst` must come from
    /// [`Backend::create_buffer_unordered`]. Offset and length are multiples of 4. Compute work
    /// recorded afterwards does not wait for the copy unless it follows [`Backend::barrier`] with
    /// the returned ticket. Until the ticket completes the host leaves `stage` alone and `dst`
    /// alive.
    ///
    /// The default records an upload on the compute lane (which orders it after everything
    /// recorded, so `after` is already satisfied) and submits; the ticket is that submission.
    fn transfer_in(&self, stage: &Stage, dst: &Buffer, dst_offset: u64, bytes: u64, after: &[Ticket]) -> Result<Ticket, BackendError> {
        let _ = after;
        check_transfer(stage, StageDir::ToDevice, dst, dst_offset, bytes)?;
        let data = &stage.as_slice()[..bytes as usize];
        let max = self.info().limits.max_buffer_bytes;
        if dst_offset == 0 {
            self.record_upload(dst, data, max)?;
        } else {
            let scratch = self.create_buffer(bytes, BufferUsage::STORAGE | BufferUsage::COPY_SRC | BufferUsage::COPY_DST, false).ok_or(BackendError::OutOfMemory { requested: bytes })?;
            self.record_upload(&scratch, data, max)?;
            self.record_copy(&scratch, 0, dst, dst_offset, bytes)?;
        }
        self.flush_batch()?;
        Ok(Ticket::flushed())
    }
    /// Copies `bytes` bytes of `src` from `src_offset` into the start of `stage` once every
    /// ticket of `after` has completed. The default reads back synchronously, so the ticket is
    /// done on return.
    fn transfer_out(&self, src: &Buffer, src_offset: u64, stage: &mut Stage, bytes: u64, after: &[Ticket]) -> Result<Ticket, BackendError> {
        let _ = after;
        check_transfer(stage, StageDir::FromDevice, src, src_offset, bytes)?;
        let host = &mut stage.as_mut_slice()[..bytes as usize];
        // SAFETY: `MaybeUninit<u8>` has the layout of `u8`, and `read_into` only writes.
        let host = unsafe { std::slice::from_raw_parts_mut(host.as_mut_ptr().cast::<MaybeUninit<u8>>(), host.len()) };
        if src_offset == 0 {
            self.read_into(src, host, false, Duration::ZERO)?;
        } else {
            let scratch = self.create_buffer(bytes, BufferUsage::STORAGE | BufferUsage::COPY_SRC | BufferUsage::COPY_DST, false).ok_or(BackendError::OutOfMemory { requested: bytes })?;
            self.record_copy(src, src_offset, &scratch, 0, bytes)?;
            self.read_into(&scratch, host, false, Duration::ZERO)?;
        }
        Ok(Ticket::done())
    }
    /// Orders compute work recorded after this call behind every ticket of `after`, and returns
    /// a ticket for the compute work recorded so far (and `after`): with an empty `after` it is
    /// how a transfer is made to wait for a kernel. The default submits the pending batch (the
    /// compute lane is in order, so there is nothing to wait for).
    fn barrier(&self, after: &[Ticket]) -> Result<Ticket, BackendError> {
        let _ = after;
        self.flush_batch()?;
        Ok(Ticket::flushed())
    }
    /// Whether the work of `t` has completed.
    fn ticket_done(&self, t: &Ticket) -> Result<bool, BackendError> {
        if t.is_done_marker() {
            Ok(true)
        } else if t.is_flushed_marker() {
            Ok(!self.gpu_busy())
        } else {
            Err(BackendError::Unsupported("a ticket of another backend".into()))
        }
    }
    /// Waits for the work of `t`, polling for `spin` before blocking.
    fn ticket_wait(&self, t: &Ticket, spin: Duration) -> Result<(), BackendError> {
        if t.is_done_marker() {
            Ok(())
        } else if t.is_flushed_marker() {
            self.sync(spin).map(|_| ())
        } else {
            Err(BackendError::Unsupported("a ticket of another backend".into()))
        }
    }
    /// Releases cached resources that are idle.
    fn reclaim_idle(&self) -> Result<(), BackendError>;
    /// Starts timing launches.
    fn enable_profiling(&self) -> Result<(), BackendError>;
    /// Stops timing and returns the profile with `splits` attached.
    ///
    /// Deprecated: the `splits` argument and [`Profile::splits`] go in 0.6, where this takes no
    /// argument and a backend returns its `Profile` without them (the context reports the
    /// planner's cuts through [`Profile::cuts`] and every other decision through
    /// `Context::take_decisions`). Until then a backend attaches `splits` to the profile as it
    /// always did; the context passes the cuts of the profiled region.
    #[deprecated(since = "0.5.2", note = "0.6 drops the `splits` argument; the cuts are `Profile::cuts()`, every decision `Context::take_decisions()`")]
    #[allow(deprecated)]
    fn finish_profiling(&self, splits: Vec<FusionSplit>) -> Result<Profile, BackendError>;
    /// Whether profiling is enabled.
    fn is_profiling(&self) -> Result<bool, BackendError>;
    /// Whether the backend can time single launches outside profiling mode (see
    /// [`Backend::time_next_launch`]). The library samples one launch in a few dozen to learn what a
    /// strategy costs; a backend that cannot leaves the default and is never asked.
    fn can_time_launches(&self) -> bool {
        false
    }
    /// Asks for the next [`Backend::record_launch`] of the calling thread to be timed and reported
    /// under `tag` by [`Backend::drain_launch_times`]. A backend that cannot time launches ignores
    /// it. A tag that is armed and not consumed by a launch (the batch is flushed first, or the
    /// launch is captured into a graph instead) is dropped, never carried over to an unrelated launch.
    fn time_next_launch(&self, tag: u64) {
        let _ = tag;
    }
    /// Appends the timings of the armed launches that have finished to `out`, as `(tag, GPU time)`.
    /// Never waits for unfinished work and never synchronizes: a launch that is still running is
    /// reported by a later call.
    fn drain_launch_times(&self, out: &mut Vec<(u64, Duration)>) {
        let _ = out;
    }
    /// Captures launches into a replayable graph, `None` when unsupported.
    fn capture_graph(&self, launches: &[GraphLaunch<'_>]) -> Option<Arc<dyn ExecGraph>>;
    /// Whether [`Backend::compile_line_solve`] can run the line solves of luma: the library then
    /// plans them for it (the block Thomas algorithm, never cyclic reduction on its own).
    fn native_line_solve(&self) -> bool {
        false
    }
    /// The cores a native matrix product ([`Backend::compile_matmul`]) spreads over, `None` if the
    /// backend has none: the library then plans the products for it (output tiles that keep that
    /// many cores busy, see `matmul::native_plan`).
    fn native_matmul(&self) -> Option<u32> {
        None
    }
    /// The native code of a matrix product (always `None` unless [`Backend::native_matmul`]).
    fn compile_matmul(&self, _spec: &MatmulSpec) -> Option<RawKernel> {
        None
    }
    /// The output tiles (rows, columns) of the products ([`Backend::compile_gemm`]) the backend
    /// runs, largest first, on tensor cores in TF32 or (with `exact`) in `f32`; empty if it has
    /// none.
    fn gemm_tiles(&self, _exact: bool) -> &[(u32, u32)] {
        &[]
    }
    /// Whether [`Backend::compile_gemm`] takes a [`GemmSpec::half`] product.
    fn gemm_half(&self) -> bool {
        false
    }
    /// Whether [`Backend::compile_gemm`] takes the [`GemmLanes`] other than `Vec4` (any shape, any leading dimension and offset).
    fn gemm_ragged(&self) -> bool {
        false
    }
    /// The code of a matrix product with one of the tiles of [`Backend::gemm_tiles`].
    fn compile_gemm(&self, _spec: &GemmSpec) -> Option<RawKernel> {
        None
    }
    /// The native code of a batched matrix product (`Builtin::Bmm` of `luma`), `None` to compile
    /// the shader instead (always `None` unless [`Backend::native_matmul`]). The launch is the
    /// shader's: the bindings `c` (written), `a` and `b` (read) and the uniform `{m, n, k,
    /// matrices, a_stride, b_stride, _, _}` (`u32`s), one workgroup per matrix (numbered along
    /// the folded grid, `x + y * num_workgroups.x`; workgroups past `matrices` do nothing).
    /// Matrix `i` is the row-major product of the `M x K` matrix at `i * a_stride` of `a` and the
    /// `K x N` one at `i * b_stride` of `b`, written at `i * m * n` of `c` (a stride of 0 shares
    /// one operand).
    fn compile_batched_matmul(&self) -> Option<RawKernel> {
        None
    }
    /// The native code of a line solve, `None` to compile the shader instead (always `None` unless
    /// [`Backend::native_line_solve`]).
    fn compile_line_solve(&self, _spec: &LineSolveSpec) -> Option<RawKernel> {
        None
    }
}

/// Checks the direction and ranges of a transfer between `stage` and `buffer`.
pub fn check_transfer(stage: &Stage, dir: StageDir, buffer: &Buffer, offset: u64, bytes: u64) -> Result<(), BackendError> {
    if stage.dir() != dir {
        return Err(BackendError::Unsupported(format!("a stage for {:?} was used for {dir:?}", stage.dir())));
    }
    if bytes > stage.len() {
        return Err(BackendError::MapRange(format!("transferring {bytes} bytes through a stage of {}", stage.len())));
    }
    if offset.checked_add(bytes).is_none_or(|end| end > buffer.size()) {
        return Err(BackendError::MapRange(format!("transferring {bytes} bytes at {offset} of a buffer of {}", buffer.size())));
    }
    if !offset.is_multiple_of(4) || !bytes.is_multiple_of(4) {
        return Err(BackendError::MapRange(format!("a transfer of {bytes} bytes at {offset} is not a multiple of 4")));
    }
    Ok(())
}

/// Reusable mapped `MAP_WRITE` buffers, ready to be written: upload staging or mappable storage.
///
/// Buffers come in size classes, and idle ones are taken back by size: [`FreeUploads::take`]
/// wants an exact size (for pools whose buffers become storage that the caller sizes by class),
/// [`FreeUploads::take_fit`] the smallest idle buffer that is large enough (for staging, where
/// only a byte range of the buffer is used). A pool that served only exact classes filled up with
/// the large buffers of a big transfer and then rejected every small buffer handed back while
/// small requests never matched a large one: each small readback allocated and freed a staging
/// buffer (about 1 ms per 4-byte readback on CUDA). [`FreeUploads::put`] therefore makes room for
/// a small buffer by releasing larger idle ones.
#[derive(Debug)]
pub struct FreeUploads {
    buffers: Vec<Buffer>,
    bytes: u64,
    /// Bytes of buffers in the unsubmitted batch or waiting for a completion callback.
    pub in_flight: u64,
    /// Most bytes of idle buffers kept for reuse; scales with the memory budget.
    cap: u64,
}

impl FreeUploads {
    /// An empty pool keeping at most `cap` idle bytes.
    pub fn new(cap: u64) -> Self {
        FreeUploads { buffers: Vec::new(), bytes: 0, in_flight: 0, cap }
    }

    /// Most bytes of idle buffers kept for reuse.
    pub fn cap(&self) -> u64 {
        self.cap
    }

    /// An idle buffer of exactly `size` bytes.
    pub fn take(&mut self, size: u64) -> Option<Buffer> {
        let at = self.buffers.iter().position(|buffer| buffer.size() == size)?;
        self.bytes -= size;
        Some(self.buffers.swap_remove(at))
    }

    /// The smallest idle buffer of at least `size` bytes. The caller must only use the first
    /// `size` bytes (map, copy and read ranges of that length), never the buffer's own size.
    pub fn take_fit(&mut self, size: u64) -> Option<Buffer> {
        let at = self.buffers.iter().enumerate().filter(|(_, buffer)| buffer.size() >= size).min_by_key(|(_, buffer)| buffer.size()).map(|(at, _)| at)?;
        let buffer = self.buffers.swap_remove(at);
        self.bytes -= buffer.size();
        Some(buffer)
    }

    /// Keeps `buffer` unless that would exceed the cap; returns the buffer the caller must
    /// release, if any. When `buffer` does not fit, the largest idle buffer is released in its
    /// place if that one is larger than `buffer` (a small buffer is the likelier next request, and
    /// the large one it displaces is the costliest to hold); otherwise `buffer` itself is
    /// returned. One eviction always makes room: the idle bytes never exceed the cap, and the
    /// evicted buffer is larger than the incoming one.
    pub fn put(&mut self, buffer: Buffer) -> Option<Buffer> {
        let size = buffer.size();
        if self.bytes + size <= self.cap {
            self.bytes += size;
            self.buffers.push(buffer);
            return None;
        }
        let largest = self.buffers.iter().enumerate().max_by_key(|(_, idle)| idle.size()).filter(|(_, idle)| idle.size() > size).map(|(at, _)| at);
        let Some(at) = largest else {
            return Some(buffer);
        };
        let evicted = self.buffers.swap_remove(at);
        self.bytes = self.bytes - evicted.size() + size;
        self.buffers.push(buffer);
        Some(evicted)
    }

    /// Empties the pool.
    pub fn drain(&mut self) -> Vec<Buffer> {
        self.bytes = 0;
        std::mem::take(&mut self.buffers)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug)]
    struct Fake(u32);
    impl NativeBuffer for Fake {}
    impl NativeKernel for Fake {}

    #[test]
    fn features_and_usage_combine() {
        let f = Features::SUBGROUP | Features::SHADER_F16;
        assert!(f.contains(Features::SUBGROUP) && !f.contains(Features::SHADER_F64));
        assert!(Features::default().is_empty());
        let mut u = BufferUsage::STORAGE;
        u |= BufferUsage::COPY_SRC;
        assert_eq!(u.bits(), 0b11);
        assert!((u & BufferUsage::COPY_DST).is_empty());
    }

    #[test]
    fn buffers_compare_by_identity_and_downcast() {
        let a = Buffer::new(16, BufferUsage::STORAGE, Fake(7));
        let b = a.clone();
        let c = Buffer::new(16, BufferUsage::STORAGE, Fake(7));
        assert_eq!(a, b);
        assert_ne!(a, c);
        assert_eq!(a.downcast_ref::<Fake>().map(|f| f.0), Some(7));
        assert!(a.downcast_ref::<u32>().is_none());
    }

    #[test]
    fn kernels_check_counts_and_kinds() {
        let info = |kind| BindingInfo { name: "x".into(), kind, element_stride: None, min_size: None };
        let kernel = CompiledKernel::new(
            "k".into(),
            RawKernel::new(Fake(0)),
            vec![info(BindingKind::StorageRead), info(BindingKind::Uniform)],
            3,
            Some(vec![true, false, true].into_boxed_slice()),
            [1, 1, 1],
        );
        assert!(kernel.check_count(3).is_ok());
        assert!(matches!(kernel.check_count(2), Err(BackendError::KernelArgCount { expected: 3, actual: 2 })));
        let buf = Buffer::new(4, BufferUsage::STORAGE, Fake(0));
        let args = [RawArg::read(&buf, 4), RawArg::Uniform(&[0; 4]), RawArg::Uniform(&[0; 4])];
        assert_eq!(kernel.bound(&args).map(|(i, _)| i).collect::<Vec<_>>(), [0, 2]);
        assert!(kernel.check_kinds(&args).is_ok());
        let bad = [RawArg::Uniform(&[0; 4]), RawArg::Uniform(&[0; 4]), RawArg::Uniform(&[0; 4])];
        assert!(matches!(kernel.check_kinds(&bad), Err(BackendError::KernelBinding(_))));
    }

    #[test]
    fn profile_merges_samples() {
        let mut p = Profile::default();
        let label: Arc<str> = "k".into();
        p.add_samples([(label.clone(), Duration::from_micros(3)), (label, Duration::from_micros(1))]);
        assert_eq!(p.launches(), 2);
        assert_eq!(p.get("k").unwrap().min, Duration::from_micros(1));
        assert_eq!(p.total(), Duration::from_micros(4));
    }

    fn staging(size: u64) -> Buffer {
        Buffer::new(size, BufferUsage::MAP_WRITE, Fake(0))
    }

    /// Size of the buffer a `put` handed back for release.
    fn released(put: Option<Buffer>) -> Option<u64> {
        put.map(|buffer| buffer.size())
    }

    fn idle_sizes(pool: &mut FreeUploads) -> Vec<u64> {
        let mut sizes: Vec<u64> = pool.drain().iter().map(Buffer::size).collect();
        sizes.sort_unstable();
        sizes
    }

    #[test]
    fn free_uploads_respect_the_cap() {
        let mut pool = FreeUploads::new(16);
        assert_eq!(released(pool.put(staging(8))), None);
        // Over the cap and larger than everything idle: rejected, the idle one stays.
        assert_eq!(released(pool.put(staging(16))), Some(16));
        assert!(pool.take(16).is_none());
        assert!(pool.take(8).is_some());
        assert!(pool.drain().is_empty());
    }

    #[test]
    fn take_is_exact_and_take_fit_is_best_fit() {
        let mut pool = FreeUploads::new(1 << 20);
        for size in [4096, 64, 1024, 256] {
            assert_eq!(released(pool.put(staging(size))), None);
        }
        assert!(pool.take(100).is_none());
        assert_eq!(pool.take(256).map(|b| b.size()), Some(256));
        // The smallest buffer that is large enough, not the first or the largest.
        assert_eq!(pool.take_fit(100).map(|b| b.size()), Some(1024));
        assert_eq!(pool.take_fit(64).map(|b| b.size()), Some(64));
        assert!(pool.take_fit(8192).is_none());
        assert_eq!(pool.take_fit(1).map(|b| b.size()), Some(4096));
        assert!(pool.take_fit(1).is_none());
        // Taking gave the bytes back to the budget.
        assert_eq!(released(pool.put(staging(1 << 20))), None);
    }

    #[test]
    fn put_releases_a_larger_idle_buffer_to_keep_a_small_one() {
        let mut pool = FreeUploads::new(1000);
        for size in [400, 300, 200, 100] {
            assert_eq!(released(pool.put(staging(size))), None);
        }
        // Full: the incoming 50 is kept, the largest idle buffer goes.
        assert_eq!(released(pool.put(staging(50))), Some(400));
        assert!(pool.take(400).is_none());
        assert_eq!(pool.take_fit(1).map(|b| b.size()), Some(50));
        // 400 bytes of room again: 250 fits without releasing anything (900 idle).
        assert_eq!(released(pool.put(staging(250))), None);
        assert_eq!(released(pool.put(staging(250))), Some(300));
        // Nothing idle is larger than 300: rejected.
        assert_eq!(released(pool.put(staging(300))), Some(300));
        // 800 bytes idle, room for 200 more.
        assert_eq!(released(pool.put(staging(200))), None);
        // A 250 is larger than the incoming 150: it goes to make room for it.
        assert_eq!(released(pool.put(staging(150))), Some(250));
        assert_eq!(idle_sizes(&mut pool), [100, 150, 200, 200, 250]);
    }

    #[test]
    fn put_rejects_a_buffer_larger_than_everything_idle_when_over_the_cap() {
        let mut pool = FreeUploads::new(300);
        for size in [100, 100, 100] {
            assert_eq!(released(pool.put(staging(size))), None);
        }
        assert_eq!(released(pool.put(staging(200))), Some(200));
        assert_eq!(released(pool.put(staging(100))), Some(100));
        assert_eq!(idle_sizes(&mut pool), [100, 100, 100]);
        // Larger than the whole cap: rejected.
        let mut pool = FreeUploads::new(100);
        assert_eq!(released(pool.put(staging(101))), Some(101));
        // The incoming buffer is larger than the idle one: rejected, the idle one stays.
        assert_eq!(released(pool.put(staging(60))), None);
        assert_eq!(released(pool.put(staging(100))), Some(100));
        assert_eq!(idle_sizes(&mut pool), [60]);
    }
}
