//! The backend hardware abstraction layer: device-independent types and the [`Backend`] trait
//! every GPU backend implements.

use std::any::Any;
use std::fmt;
use std::mem::MaybeUninit;
use std::panic::Location;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::Duration;

pub mod cache;
pub mod copy;

pub use cache::DiskCache;
pub use copy::{advise_huge_pages, copy_parallel, copy_streaming};

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
}

impl fmt::Display for BindingKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            BindingKind::StorageRead => "read-only storage buffer",
            BindingKind::StorageReadWrite => "read_write storage buffer",
            BindingKind::Uniform => "uniform",
        })
    }
}

/// One reflected `@group(1)` binding of a kernel; its index is the argument position.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BindingInfo {
    /// The variable's name in the shader.
    pub name: String,
    /// Whether the binding is a read-only storage buffer, a read-write storage buffer or a uniform.
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

impl CostModel {
    /// RTX 3090 (Vulkan), measured by calibration (best of several runs).
    pub const NVIDIA: CostModel = CostModel { launch_us: 2.9, bandwidth_gbs: 740.0, f32_ops_per_ns: 16000.0, df32_factor: 27.0, f64_factor: Some(41.0) };
    /// Every other GPU, measured on a RADV Raphael iGPU (best of 5 runs).
    pub const GENERIC: CostModel = CostModel { launch_us: 2.0, bandwidth_gbs: 62.0, f32_ops_per_ns: 370.0, df32_factor: 44.0, f64_factor: Some(11.0) };
    /// Mesa llvmpipe on a Ryzen 9 7950X (best of 5 runs).
    pub const CPU: CostModel = CostModel { launch_us: 32.0, bandwidth_gbs: 31.0, f32_ops_per_ns: 30.0, df32_factor: 2.3, f64_factor: Some(1.1) };
    /// The numbers tests pin so that kernel counts do not depend on the machine.
    #[doc(hidden)]
    pub const TEST: CostModel = CostModel { launch_us: 2.9, bandwidth_gbs: 740.0, f32_ops_per_ns: 16000.0, df32_factor: 36.0, f64_factor: Some(41.0) };

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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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

/// A point where an expression was cut into several kernels, and why.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FusionSplit {
    /// The operation whose creation (or evaluation) caused the cut.
    pub op: &'static str,
    /// Where that operation was written.
    pub location: &'static Location<'static>,
    /// Why the expression was cut.
    pub reason: SplitReason,
}

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
        }
    }
}

/// GPU durations per kernel, measured with timestamp queries.
///
/// A duration is the time from the start to the end of the compute pass of one launch, so it
/// includes the GPU's own scheduling of that pass but not the time between launches or any host
/// time.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Profile {
    /// One entry per kernel label, the one with the most total time first.
    pub kernels: Vec<KernelTiming>,
    /// Where lazy expressions were cut into several kernels while profiling, in the order they
    /// happened (see [`SplitReason`]).
    pub splits: Vec<FusionSplit>,
}

impl Profile {
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

static NEXT_BUFFER_ID: AtomicU64 = AtomicU64::new(1);
static NEXT_KERNEL_ID: AtomicU64 = AtomicU64::new(1);

/// A backend's own buffer object, held type-erased by [`Buffer`].
pub trait NativeBuffer: Any + fmt::Debug + Send + Sync + 'static {}

/// A device buffer of any backend. Cloning shares the buffer; equality is identity.
#[derive(Clone, Debug)]
pub struct Buffer {
    id: u64,
    size: u64,
    usage: BufferUsage,
    native: Arc<dyn NativeBuffer>,
}

impl Buffer {
    /// Wraps a backend's buffer of `size` bytes.
    pub fn new(size: u64, usage: BufferUsage, native: impl NativeBuffer) -> Self {
        Buffer { id: NEXT_BUFFER_ID.fetch_add(1, Ordering::Relaxed), size, usage, native: Arc::new(native) }
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

/// A compiled kernel of any backend.
#[derive(Clone, Debug)]
pub struct RawKernel {
    id: u64,
    native: Arc<dyn NativeKernel>,
}

impl RawKernel {
    /// Wraps a backend's kernel.
    pub fn new(native: impl NativeKernel) -> Self {
        RawKernel { id: NEXT_KERNEL_ID.fetch_add(1, Ordering::Relaxed), native: Arc::new(native) }
    }
    /// Unique per compiled kernel.
    pub fn id(&self) -> u64 {
        self.id
    }
    /// The backend's kernel, if it is a `T`.
    pub fn downcast_ref<T: 'static>(&self) -> Option<&T> {
        (&*self.native as &dyn Any).downcast_ref::<T>()
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

    /// Checks that each argument the kernel binds is of the kind of its binding, for the same
    /// reason as [`check_count`](Self::check_count).
    pub fn check_kinds(&self, args: &[RawArg<'_>]) -> Result<(), BackendError> {
        for ((position, arg), info) in self.bound(args).zip(&self.bindings) {
            // Whether a storage buffer is passed for reading or writing does not matter to the
            // layout; a uniform where a storage buffer is expected (or the reverse) does.
            let (passed, fits) = match arg {
                RawArg::Uniform(_) => ("uniform", info.kind == BindingKind::Uniform),
                RawArg::Storage { .. } => ("storage buffer", info.kind != BindingKind::Uniform),
            };
            if !fits {
                return Err(BackendError::KernelBinding(format!(
                    "kernel `{}` argument {position} (`{}`) is a {}, but the launch passes a {passed}",
                    self.label, info.name, info.kind
                )));
            }
        }
        Ok(())
    }
}

/// One argument of a launch.
#[derive(Clone, Copy, Debug)]
pub enum RawArg<'a> {
    /// A storage buffer, of which the kernel may reach the first `bytes`.
    Storage {
        /// The buffer.
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
    /// Workgroup counts.
    pub workgroups: [u32; 3],
    /// The logical grid size in invocations.
    pub size: [u32; 3],
    /// Arguments of the launch.
    pub args: &'a [RawArg<'a>],
}

/// A recorded sequence of launches a backend can replay cheaply.
pub trait ExecGraph: Any + fmt::Debug + Send + Sync + 'static {}

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
}

/// A GPU backend: allocates buffers, compiles kernels and records and runs work.
pub trait Backend: fmt::Debug + Send + Sync + 'static {
    /// What is known about the device.
    fn info(&self) -> &DeviceInfo;
    /// The device's on-disk cache.
    fn disk_cache(&self) -> &Arc<DiskCache>;
    /// The device's cost model, set once calibrated.
    fn device_cost(&self) -> &Arc<OnceLock<CostModel>>;
    /// Allocates a buffer, `None` when the device is out of memory; `mapped` buffers start
    /// mapped for writing.
    fn create_buffer(&self, size: u64, usage: BufferUsage, mapped: bool) -> Option<Buffer>;
    /// Allocates a buffer holding `data`.
    fn alloc_filled(&self, data: &[u8], max_buffer_bytes: u64) -> Result<Buffer, BackendError>;
    /// Returns a `MAP_WRITE` buffer for reuse.
    fn recycle_mappable(&self, buffer: Buffer);
    /// Compiles a module; `used` is the entry point's binding mask, see [`CompiledKernel::used`].
    fn compile_kernel(&self, module: &crate::kir::KernelModule, used: Option<&[bool]>, label: &str, trusted: bool) -> Result<RawKernel, BackendError>;
    /// Compiles an already parsed naga module, `None` if the backend takes only [`KernelModule`](crate::kir::KernelModule)s.
    fn compile_naga(&self, _module: &naga::Module, _entry: &str, _bindings: &[BindingInfo], _workgroup: [u32; 3], _label: &str, _trusted: bool) -> Option<Result<RawKernel, BackendError>> {
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
    /// Releases cached resources that are idle.
    fn reclaim_idle(&self) -> Result<(), BackendError>;
    /// Starts timing launches.
    fn enable_profiling(&self) -> Result<(), BackendError>;
    /// Stops timing and returns the profile with `splits` attached.
    fn finish_profiling(&self, splits: Vec<FusionSplit>) -> Result<Profile, BackendError>;
    /// Whether profiling is enabled.
    fn is_profiling(&self) -> Result<bool, BackendError>;
    /// Captures launches into a replayable graph, `None` when unsupported.
    fn capture_graph(&self, launches: &[GraphLaunch<'_>]) -> Option<Arc<dyn ExecGraph>>;
}

/// Reusable mapped `MAP_WRITE` buffers, ready to be written: upload staging or mappable storage.
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

    /// Keeps `buffer` unless that would exceed the cap; returns it otherwise.
    pub fn put(&mut self, buffer: Buffer) -> Option<Buffer> {
        if self.bytes + buffer.size() <= self.cap {
            self.bytes += buffer.size();
            self.buffers.push(buffer);
            None
        } else {
            Some(buffer)
        }
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

    #[test]
    fn free_uploads_respect_the_cap() {
        let mut pool = FreeUploads::new(16);
        let a = Buffer::new(16, BufferUsage::MAP_WRITE, Fake(0));
        let b = Buffer::new(8, BufferUsage::MAP_WRITE, Fake(0));
        assert!(pool.put(a).is_none());
        assert!(pool.put(b).is_some());
        assert!(pool.take(8).is_none());
        assert!(pool.take(16).is_some());
        assert!(pool.drain().is_empty());
    }
}
