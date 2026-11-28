//! Runtime `Backend` implementation for `luma-cpu`.
//!
//! Buffers are 64-byte-aligned host allocations and launches run synchronously on the
//! [`WorkgroupPool`], so there is no batch to flush and nothing to wait for.

use std::borrow::Cow;
use std::fmt;
use std::mem::MaybeUninit;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, LazyLock, Mutex, MutexGuard, OnceLock, PoisonError};
use std::time::{Duration, Instant};

use luma_core::hal::{cache, copy_parallel_host, size_class};
use luma_core::kir::KernelModule;
use luma_core::{
    Backend, BackendError, BackendKind, Buffer, BufferUsage, CompiledKernel,
    CostModel, DeviceInfo, DeviceLimits, DiskCache, Dispatch, ExecGraph, Features, FreeUploads,
    GraphLaunch, LineSolveSpec, MatmulSpec, Profile, RawArg, RawKernel, TuningProfile,
};

use crate::codegen::CompiledCpuKernel;
use crate::exec::{AlignedBuf, CostHint, ExecBlob, WorkgroupPool};
use crate::fission;
use crate::gemm::{NativeBatchedMatmul, NativeMatmul};
use crate::solve::NativeLineSolve;

/// Most idle bytes of recycled mappable storage kept for reuse.
const MAPPABLE_CAP: u64 = 256 << 20;
const MAPPABLE_STORAGE_USAGE: BufferUsage = BufferUsage::STORAGE
    .union(BufferUsage::COPY_SRC)
    .union(BufferUsage::COPY_DST)
    .union(BufferUsage::MAP_WRITE);

/// Locks `mutex`, ignoring poisoning: every guarded value stays consistent if a holder panicked.
fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

fn raw(buffer: &Buffer) -> Result<&AlignedBuf, BackendError> {
    buffer
        .downcast_ref::<AlignedBuf>()
        .ok_or_else(|| BackendError::Unsupported("the buffer belongs to another backend".into()))
}

struct CpuShared {
    pool: Arc<WorkgroupPool>,
    info: DeviceInfo,
    disk: Arc<DiskCache>,
    cost: Arc<OnceLock<CostModel>>,
}

static SHARED: LazyLock<CpuShared> = LazyLock::new(|| {
    let pool = Arc::clone(WorkgroupPool::global());
    let info = DeviceInfo {
        backend: BackendKind::Cpu,
        name: format!(
            "CPU ({}, {} threads)",
            std::env::consts::ARCH,
            pool.threads()
        ),
        cpu: true,
        zero_copy: true,
        profile: TuningProfile::Cpu,
        features: Features::TIMESTAMP_QUERY
            | Features::SUBGROUP
            | Features::SHADER_F16
            | Features::SHADER_F64,
        subgroup_size: Some((4, 4)),
        memory_bytes: None,
        ieee_f32_intrinsics: true,
        hardware_fma: Some(true),
        limits: DeviceLimits {
            max_buffer_bytes: (1u64 << 31) & !3,
            max_storage_buffers: 256,
            max_uniform_bytes: 65536,
            uniform_align: 16,
            max_dynamic_uniforms: u32::MAX,
            max_workgroups_per_dim: 65535,
            max_workgroup_size: [1024, 1024, 64],
            max_workgroup_invocations: 1024,
            max_workgroup_storage_bytes: 65536,
        },
        compute_units: Some(pool.threads() as u32),
    };
    let key = format!(
        "cpu-{:016x}",
        cache::fnv1a(crate::codegen::host_isa_key().as_bytes())
    );
    CpuShared {
        pool,
        info,
        disk: DiskCache::open(cache::directory(), key),
        cost: Arc::new(OnceLock::new()),
    }
});

/// The native CPU backend.
type ProfileSamples = Vec<(Arc<str>, Duration)>;

pub struct CpuBackend {
    info: DeviceInfo,
    disk: Arc<DiskCache>,
    cost: Arc<OnceLock<CostModel>>,
    pool: Arc<WorkgroupPool>,
    dispatches: AtomicU32,
    mappable: Mutex<FreeUploads>,
    profiler: Mutex<Option<ProfileSamples>>,
    /// Whether `armed_tag` names the next launch, which `Backend::time_next_launch` asked to time.
    armed: AtomicBool,
    armed_tag: AtomicU64,
    /// Finished timings of armed launches, taken by `Backend::drain_launch_times`.
    launch_times: Mutex<Vec<(u64, Duration)>>,
}

impl fmt::Debug for CpuBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CpuBackend")
            .field("name", &self.info.name)
            .finish()
    }
}

/// Opens the native CPU backend.
pub fn open() -> Result<Box<dyn Backend>, BackendError> {
    let shared = &*SHARED;
    Ok(Box::new(CpuBackend {
        info: shared.info.clone(),
        disk: Arc::clone(&shared.disk),
        cost: Arc::clone(&shared.cost),
        pool: Arc::clone(&shared.pool),
        dispatches: AtomicU32::new(0),
        mappable: Mutex::new(FreeUploads::new(MAPPABLE_CAP)),
        profiler: Mutex::new(None),
        armed: AtomicBool::new(false),
        armed_tag: AtomicU64::new(0),
        launch_times: Mutex::new(Vec::new()),
    }))
}

/// Alignment of the offset of a bound storage range.
const STORAGE_OFFSET_ALIGN: u64 = 16;

/// One argument of a captured launch, owned.
#[derive(Debug)]
enum GraphArg {
    Storage {
        buffer: Buffer,
        bytes: u64,
        writable: bool,
    },
    Uniform(Box<[u8]>),
}

/// The workgroups of a captured launch; an indirect one reads its counts when the graph runs.
#[derive(Debug)]
enum CapturedDispatch {
    Direct([u32; 3]),
    Indirect {
        buffer: Buffer,
        offset: u64,
        max_workgroups: [u32; 3],
    },
}

impl CapturedDispatch {
    fn dispatch(&self) -> Dispatch<'_> {
        match self {
            CapturedDispatch::Direct(workgroups) => Dispatch::Direct(*workgroups),
            CapturedDispatch::Indirect {
                buffer,
                offset,
                max_workgroups,
            } => Dispatch::Indirect {
                buffer,
                offset: *offset,
                max_workgroups: *max_workgroups,
            },
        }
    }
}

#[derive(Debug)]
struct CapturedCpuLaunch {
    kernel: CompiledKernel,
    dispatch: CapturedDispatch,
    size: [u32; 3],
    args: Vec<GraphArg>,
}

#[derive(Debug)]
struct CpuExecGraph {
    launches: Vec<CapturedCpuLaunch>,
}

impl ExecGraph for CpuExecGraph {}

/// What a launched kernel runs: compiled shader code or a native solve.
enum CpuKernel<'a> {
    Compiled(&'a CompiledCpuKernel),
    LineSolve(&'a NativeLineSolve),
    Matmul(&'a NativeMatmul),
    BatchedMatmul(&'a NativeBatchedMatmul),
}

impl CpuBackend {
    /// The mask and workgroup size checks of a kernel, which fail before any compiling.
    fn check_module(&self, module: &KernelModule, used: Option<&[bool]>, label: &str) -> Result<(), BackendError> {
        if let Some(mask) = used
            && mask.len() != module.bindings.len()
        {
            return Err(BackendError::KernelBinding(format!(
                "kernel `{label}`: the used mask has {} entries for {} bindings",
                mask.len(),
                module.bindings.len()
            )));
        }
        if module.bindings.iter().any(|b| matches!(b.kind, luma_core::BindingKind::Texture2d | luma_core::BindingKind::Sampler)) {
            return Err(BackendError::Unsupported(format!("kernel `{label}` reads a texture; the CPU backend has no textures")));
        }
        let limits = &self.info.limits;
        let invocations: u64 = module
            .workgroup_size
            .iter()
            .map(|&x| u64::from(x.max(1)))
            .product();
        let too_wide = module
            .workgroup_size
            .iter()
            .zip(limits.max_workgroup_size)
            .any(|(&x, max)| x > max);
        if too_wide || invocations > u64::from(limits.max_workgroup_invocations) {
            return Err(BackendError::KernelWorkgroup(format!(
                "kernel `{label}` has workgroup size {:?} ({invocations} invocations), the device allows {:?} and {} invocations",
                module.workgroup_size, limits.max_workgroup_size, limits.max_workgroup_invocations
            )));
        }
        Ok(())
    }
}

/// Revision of the code this crate generates and the tables that code calls through: a hash of
/// the sources of the code generator, the ABI (the layout of the math table the machine code
/// indexes), the packet math routines, the planner and the runtime, so that a blob written by a
/// build that differs in any of them is never loaded.
fn code_revision() -> u64 {
    static REVISION: OnceLock<u64> = OnceLock::new();
    *REVISION.get_or_init(|| {
        let sources = [
            include_str!("codegen.rs"),
            include_str!("abi.rs"),
            include_str!("vecmath.rs"),
            include_str!("fission.rs"),
            include_str!("exec.rs"),
            include_str!("gemm.rs"),
            include_str!("solve.rs"),
            include_str!("backend.rs"),
        ];
        cache::source_revision(&sources)
    })
}

/// The file name of the compiled blob of the optimized kernel `kir` (its text) under `revision`.
fn cache_stem(kir: &str, used: Option<&[bool]>, zero_workgroup_memory: bool, revision: u64) -> String {
    let identity = format!(
        "{kir}\n{used:?}\n{zero_workgroup_memory}\n{}\n{}\n{revision:016x}",
        crate::codegen::host_isa_key(),
        crate::codegen::REVISION
    );
    format!("cpu-{:016x}", cache::fnv1a(identity.as_bytes()))
}

#[cfg(test)]
mod cache_tests {
    use super::*;

    /// A blob cached by one revision of the generated code is not found by another.
    #[test]
    fn a_blob_of_another_code_revision_is_not_used() {
        let dir = std::env::temp_dir().join(format!("luma-cpu-revision-test-{}", std::process::id()));
        let disk = DiskCache::open(Some(dir.clone()), "test".into());
        let old = cache_stem("kernel", None, false, 1);
        let new = cache_stem("kernel", None, false, 2);
        assert_ne!(old, new);
        assert!(disk.store_blob(&old, cache::CPU_MAGIC, b"machine code"));
        assert_eq!(disk.load_blob(&old, cache::CPU_MAGIC).as_deref(), Some(&b"machine code"[..]));
        assert!(disk.load_blob(&new, cache::CPU_MAGIC).is_none());
        // The same kernel under the same revision is the same file; a different kernel is not.
        assert_eq!(old, cache_stem("kernel", None, false, 1));
        assert_ne!(old, cache_stem("other", None, false, 1));
        // The revision of this build is stable within the process and depends on the sources.
        assert_eq!(code_revision(), code_revision());
        assert_ne!(cache::source_revision(&["a"]), cache::source_revision(&["b"]));
        let _ = std::fs::remove_dir_all(dir);
    }
}

/// Plans, compiles (or loads from the disk cache) the kernel of a checked `module`; runs on the
/// calling thread, which may be a compile worker.
fn compile_native(
    disk: &DiskCache,
    max_shared_bytes: u32,
    module: &KernelModule,
    used: Option<&[bool]>,
    label: &str,
    trust: luma_core::KernelTrust,
) -> Result<CompiledCpuKernel, BackendError> {
    let checked = if trust.checks_bounds() {
        luma_core::kir::opt::with_bounds_checks(module)
    } else {
        Cow::Borrowed(module)
    };
    let optimized = luma_core::kir::optimize_for_target(&checked, true);
    let plan = fission::plan(&optimized, used, trust.zero_workgroup_memory());
    if plan.shared_bytes > max_shared_bytes {
        return Err(BackendError::KernelWorkgroup(format!(
            "kernel `{label}` uses {} bytes of workgroup memory, the device allows {}",
            plan.shared_bytes, max_shared_bytes
        )));
    }
    let stem = cache_stem(&format!("{optimized:?}"), used, trust.zero_workgroup_memory(), code_revision());
    if let Some(bytes) = disk.load_blob(&stem, cache::CPU_MAGIC)
        && let Ok(blob) = ExecBlob::new(&bytes)
    {
        return Ok(CompiledCpuKernel { exec: Arc::new(blob), plan, cost: CostHint::default() });
    }
    let (compiled, bytes) =
        crate::codegen::compile(&optimized, used, trust.zero_workgroup_memory()).map_err(|e| match e {
            BackendError::ShaderCompile(m) => {
                BackendError::ShaderCompile(format!("kernel `{label}`: {m}"))
            }
            other => other,
        })?;
    disk.store_blob(&stem, cache::CPU_MAGIC, &bytes);
    Ok(compiled)
}

/// The kernel behind `kernel`, waiting for its compilation: a compile error is reported as is.
fn native(kernel: &CompiledKernel) -> Result<CpuKernel<'_>, BackendError> {
    match kernel.raw.resolve::<CompiledCpuKernel>() {
        Ok(compiled) => return Ok(CpuKernel::Compiled(compiled)),
        Err(BackendError::Unsupported(_)) => {}
        Err(error) => return Err(error),
    }
    if let Some(solve) = kernel.raw.downcast_ref::<NativeLineSolve>() {
        return Ok(CpuKernel::LineSolve(solve));
    }
    if let Some(product) = kernel.raw.downcast_ref::<NativeBatchedMatmul>() {
        return Ok(CpuKernel::BatchedMatmul(product));
    }
    kernel
        .raw
        .downcast_ref::<NativeMatmul>()
        .map(CpuKernel::Matmul)
        .ok_or_else(|| BackendError::Unsupported("the kernel belongs to another backend".into()))
}

fn invalid(what: &str) -> BackendError {
    BackendError::Unsupported(what.into())
}

/// The workgroup counts of a dispatch, read from its buffer when indirect.
fn workgroups_of(dispatch: Dispatch<'_>) -> Result<[u32; 3], BackendError> {
    match dispatch {
        Dispatch::Direct(workgroups) => Ok(workgroups),
        Dispatch::Indirect { buffer, offset, .. } => {
            let end = offset
                .checked_add(12)
                .filter(|&end| end <= buffer.size())
                .ok_or_else(|| invalid("indirect dispatch arguments lie outside their buffer"))?;
            let buf = raw(buffer)?;
            // SAFETY: `end <= buffer.size() <= buf.len()`; execution is synchronous so nothing writes concurrently.
            let bytes = unsafe { buf.as_slice(end as usize) };
            let word = |i: usize| {
                let at = offset as usize + 4 * i;
                u32::from_ne_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]])
            };
            Ok([word(0), word(1), word(2)])
        }
    }
}

impl CpuBackend {
    /// Runs one launch synchronously; returns the elapsed time.
    fn run(
        &self,
        kernel: &CompiledKernel,
        workgroups: [u32; 3],
        size: [u32; 3],
        args: &[RawArg<'_>],
    ) -> Result<Duration, BackendError> {
        kernel.check_count(args.len())?;
        kernel.check_kinds(args)?;
        kernel.check_offsets(args, STORAGE_OFFSET_ALIGN)?;
        let native = native(kernel)?;
        if workgroups.contains(&0) {
            return Ok(Duration::ZERO);
        }
        // Uniforms are copied into 16-byte-aligned storage padded to the declared type size.
        let mut sizes = Vec::new();
        for ((_, arg), info) in kernel.bound(args).zip(&kernel.bindings) {
            if let RawArg::Uniform(bytes) = arg {
                let min = info.min_size.map_or(0, |m| m as usize);
                sizes.push(bytes.len().max(min).max(16).next_multiple_of(16));
            }
        }
        let mut store = vec![0u128; sizes.iter().sum::<usize>() / 16];
        let store_ptr = store.as_mut_ptr().cast::<u8>();
        let mut bindings: Vec<*mut u8> = Vec::with_capacity(kernel.bindings.len());
        let mut binding_bytes: Vec<u64> = Vec::with_capacity(kernel.bindings.len());
        let (mut next_uniform, mut offset) = (0, 0);
        for (_, arg) in kernel.bound(args) {
            match arg {
                RawArg::Storage { buffer, bytes, .. } => {
                    // SAFETY: `check_offsets` kept `offset + bytes` within the buffer.
                    bindings.push(unsafe { raw(buffer)?.as_ptr().add(buffer.offset() as usize) });
                    binding_bytes.push(*bytes);
                }
                RawArg::Uniform(bytes) => {
                    // SAFETY: `offset + sizes[next_uniform]` stays within `store`, which holds the sum of all sizes.
                    let dst = unsafe { store_ptr.add(offset) };
                    // SAFETY: the destination region holds at least `bytes.len()` bytes.
                    unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), dst, bytes.len()) };
                    bindings.push(dst);
                    binding_bytes.push(sizes[next_uniform] as u64);
                    offset += sizes[next_uniform];
                    next_uniform += 1;
                }
            }
        }
        let compiled = match native {
            CpuKernel::Compiled(compiled) => compiled,
            CpuKernel::Matmul(product) => {
                let uniform = bindings.last().ok_or_else(|| invalid("a matrix product without its uniform"))?;
                // The uniform's storage is 16-byte aligned and padded to what the launch passed (see `sizes`): the words need 32 bytes.
                if binding_bytes.last().is_none_or(|&bytes| bytes < 4 * crate::gemm::UNIFORM_WORDS as u64) {
                    return Err(invalid("a matrix product's uniform of fewer than eight words"));
                }
                // SAFETY: just checked: the storage holds the eight words.
                let words = unsafe { uniform.cast::<[u32; crate::gemm::UNIFORM_WORDS]>().read() };
                product.validate(&binding_bytes, words, workgroups)?;
                let start = Instant::now();
                self.pool.dispatch_native(
                    product.func,
                    std::ptr::from_ref(product).cast(),
                    size,
                    workgroups,
                    &bindings,
                    &binding_bytes,
                    &[],
                    0,
                    0,
                    false,
                    &product.cost,
                );
                drop(store);
                return Ok(start.elapsed());
            }
            CpuKernel::BatchedMatmul(product) => {
                let uniform = bindings.last().ok_or_else(|| invalid("a batched matrix product without its uniform"))?;
                // SAFETY: the uniform's storage is 16-byte aligned and at least 32 bytes (the shader's `Params`).
                let words = unsafe { uniform.cast::<[u32; 8]>().read() };
                product.validate(&binding_bytes, words, workgroups)?;
                let start = Instant::now();
                self.pool.dispatch_native(
                    product.func,
                    std::ptr::from_ref(product).cast(),
                    size,
                    workgroups,
                    &bindings,
                    &binding_bytes,
                    &[],
                    0,
                    0,
                    false,
                    &product.cost,
                );
                drop(store);
                return Ok(start.elapsed());
            }
            CpuKernel::LineSolve(solve) => {
                let uniform = bindings.last().ok_or_else(|| invalid("a line solve without its uniform"))?;
                // SAFETY: the uniform's storage is 16-byte aligned and at least 16 bytes.
                let words = unsafe { uniform.cast::<[u32; 3]>().read() };
                solve.validate(&binding_bytes, words)?;
                let start = Instant::now();
                self.pool.dispatch_native(
                    solve.func,
                    std::ptr::from_ref(solve).cast(),
                    size,
                    workgroups,
                    &bindings,
                    &binding_bytes,
                    &[],
                    0,
                    0,
                    false,
                    &solve.cost,
                );
                drop(store);
                return Ok(start.elapsed());
            }
        };
        let lengths: Vec<u32> = compiled
            .plan
            .array_lengths
            .iter()
            .map(|&(binding, stride)| match args.get(binding as usize) {
                Some(RawArg::Storage { bytes, .. }) => {
                    u32::try_from(bytes / u64::from(stride.max(1))).unwrap_or(u32::MAX)
                }
                _ => 0,
            })
            .collect();
        let start = Instant::now();
        self.pool.dispatch(
            compiled.exec.as_fn(),
            size,
            workgroups,
            &bindings,
            &binding_bytes,
            &lengths,
            compiled.plan.shared_bytes,
            compiled.plan.scratch_bytes,
            compiled.plan.initialize_shared,
            &compiled.cost,
        );
        drop(store);
        Ok(start.elapsed())
    }

    fn sample(&self, kernel: &CompiledKernel, elapsed: Duration) {
        if let Some(samples) = lock(&self.profiler).as_mut() {
            samples.push((Arc::clone(&kernel.label), elapsed));
        }
    }
}

impl Backend for CpuBackend {
    fn info(&self) -> &DeviceInfo {
        &self.info
    }

    fn storage_offset_align(&self) -> u64 {
        STORAGE_OFFSET_ALIGN
    }

    fn disk_cache(&self) -> &Arc<DiskCache> {
        &self.disk
    }

    fn device_cost(&self) -> &Arc<OnceLock<CostModel>> {
        &self.cost
    }

    fn create_buffer(&self, size: u64, usage: BufferUsage, _mapped: bool) -> Option<Buffer> {
        if size > self.info.limits.max_buffer_bytes {
            return None;
        }
        AlignedBuf::new(size).map(|b| Buffer::new(size, usage, b))
    }

    fn import_host(&self, region: luma_core::hal::HostRegion) -> Option<Buffer> {
        let usage = BufferUsage::STORAGE.union(BufferUsage::COPY_SRC).union(BufferUsage::COPY_DST);
        // SAFETY: the importer guarantees the region and its exclusivity (see `HostRegion`); the buffer holds `keepalive`.
        let native = unsafe { AlignedBuf::borrowed(region.ptr, region.len, region.keepalive) }?;
        Some(Buffer::new_borrowed(region.len, usage, native))
    }

    fn learned_ns(&self, kernel: &CompiledKernel, workgroups: u64) -> Option<f64> {
        match native(kernel).ok()? {
            CpuKernel::Compiled(compiled) => compiled.cost.learned_ns(workgroups),
            _ => None,
        }
    }

    fn alloc_filled(&self, data: &[u8], max_buffer_bytes: u64) -> Result<Buffer, BackendError> {
        let bytes = data.len() as u64;
        let class = size_class(bytes, max_buffer_bytes)
            .ok_or(BackendError::OutOfMemory { requested: bytes })?;
        let reused = lock(&self.mappable).take(class);
        let buffer = match reused {
            Some(buffer) => buffer,
            None => AlignedBuf::new(class)
                .map(|b| Buffer::new(class, MAPPABLE_STORAGE_USAGE, b))
                .ok_or(BackendError::OutOfMemory { requested: class })?,
        };
        let buf = raw(&buffer)?;
        // SAFETY: `data.len() <= class == buf.len()`; the buffer is idle and exclusively ours.
        copy_parallel_host(data, unsafe { buf.as_uninit_slice_mut(data.len()) });
        Ok(buffer)
    }

    fn host_ptr(&self, buffer: &Buffer, bytes: u64) -> Option<*mut u8> {
        let native = raw(buffer).ok()?;
        // Launches run when they are recorded, but a batch that has not been flushed is still
        // pending work: reading before the flush would skip it (see `pending_dispatches`).
        if bytes > buffer.size() || self.dispatches.load(Ordering::Relaxed) != 0 {
            return None;
        }
        Some(native.as_ptr())
    }

    fn alloc_host_buffer(&self, bytes: u64, max_buffer_bytes: u64) -> Result<Buffer, BackendError> {
        let class = size_class(bytes, max_buffer_bytes).ok_or(BackendError::OutOfMemory { requested: bytes })?;
        let reused = lock(&self.mappable).take(class);
        match reused {
            Some(buffer) => Ok(buffer),
            None => AlignedBuf::new(class)
                .map(|b| Buffer::new(class, MAPPABLE_STORAGE_USAGE, b))
                .ok_or(BackendError::OutOfMemory { requested: class }),
        }
    }

    fn recycle_mappable(&self, buffer: Buffer) {
        // Execution is synchronous, so a recycled buffer is idle at once; over the cap it is dropped.
        let _ = lock(&self.mappable).put(buffer);
    }

    fn compile_kernel(
        &self,
        module: &KernelModule,
        used: Option<&[bool]>,
        label: &str,
        trust: luma_core::KernelTrust,
    ) -> Result<RawKernel, BackendError> {
        self.check_module(module, used, label)?;
        compile_native(&self.disk, self.info.limits.max_workgroup_storage_bytes, module, used, label, trust).map(RawKernel::new)
    }

    fn compile_kernel_deferred(
        &self,
        module: KernelModule,
        used: Option<Box<[bool]>>,
        label: &str,
        trust: luma_core::KernelTrust,
    ) -> Result<RawKernel, BackendError> {
        self.check_module(&module, used.as_deref(), label)?;
        let disk = Arc::clone(&self.disk);
        let max_shared = self.info.limits.max_workgroup_storage_bytes;
        let label = label.to_owned();
        Ok(RawKernel::spawn(move || compile_native(&disk, max_shared, &module, used.as_deref(), &label, trust)))
    }

    fn record_launch(
        &self,
        kernel: &CompiledKernel,
        dispatch: Dispatch<'_>,
        size: [u32; 3],
        args: &[RawArg<'_>],
    ) -> Result<u32, BackendError> {
        let workgroups = workgroups_of(dispatch)?;
        let elapsed = self.run(kernel, workgroups, size, args)?;
        if self.armed.load(Ordering::Relaxed) && self.armed.swap(false, Ordering::Acquire) {
            // An empty grid has no time worth reporting; its arm is consumed all the same.
            if !workgroups.contains(&0) {
                lock(&self.launch_times).push((self.armed_tag.load(Ordering::Relaxed), elapsed));
            }
        }
        if workgroups.contains(&0) {
            if let Some(samples) = lock(&self.profiler).as_mut() {
                samples.push((Arc::clone(&kernel.label), Duration::ZERO));
            }
            return Ok(self.dispatches.load(Ordering::Relaxed));
        }
        if let Some(samples) = lock(&self.profiler).as_mut() {
            samples.push((Arc::clone(&kernel.label), elapsed));
            self.dispatches.store(0, Ordering::Relaxed);
            return Ok(0);
        }
        Ok(self.dispatches.fetch_add(1, Ordering::Relaxed) + 1)
    }

    fn record_clear(&self, buffer: &Buffer, bytes: u64) -> Result<(), BackendError> {
        let n = bytes.min(buffer.size()) as usize;
        // SAFETY: `n <= buffer.size()`; execution is synchronous so no launch is running.
        unsafe { std::ptr::write_bytes(raw(buffer)?.as_ptr(), 0, n) };
        Ok(())
    }

    fn record_copy(
        &self,
        src: &Buffer,
        src_offset: u64,
        dst: &Buffer,
        dst_offset: u64,
        bytes: u64,
    ) -> Result<(), BackendError> {
        let in_range = |buffer: &Buffer, offset: u64| {
            offset
                .checked_add(bytes)
                .is_some_and(|end| end <= buffer.size())
        };
        if !in_range(src, src_offset) || !in_range(dst, dst_offset) {
            return Err(invalid("buffer copy out of range"));
        }
        let (s, d) = (raw(src)?, raw(dst)?);
        let n = bytes as usize;
        // SAFETY: both ranges were checked against the buffer sizes.
        unsafe {
            let from = s.as_ptr().add(src_offset as usize);
            let to = d.as_ptr().add(dst_offset as usize);
            if src.id() == dst.id() {
                std::ptr::copy(from, to, n);
            } else {
                copy_parallel_host(
                    std::slice::from_raw_parts(from, n),
                    std::slice::from_raw_parts_mut(to.cast::<MaybeUninit<u8>>(), n),
                );
            }
        }
        Ok(())
    }

    fn record_upload(
        &self,
        dst: &Buffer,
        data: &[u8],
        _max_buffer_bytes: u64,
    ) -> Result<(), BackendError> {
        if data.len() as u64 > dst.size() {
            return Err(invalid("upload larger than its destination buffer"));
        }
        let buf = raw(dst)?;
        // SAFETY: `data.len() <= dst.size()`; no launch is running.
        copy_parallel_host(data, unsafe { buf.as_uninit_slice_mut(data.len()) });
        Ok(())
    }

    fn record_graph(&self, graph: &dyn ExecGraph, _dispatches: u32) -> Result<u32, BackendError> {
        let graph = graph
            .downcast_ref::<CpuExecGraph>()
            .ok_or_else(|| invalid("the graph belongs to another backend"))?;
        for launch in &graph.launches {
            let args: Vec<RawArg<'_>> = launch
                .args
                .iter()
                .map(|arg| match arg {
                    GraphArg::Storage {
                        buffer,
                        bytes,
                        writable,
                    } => RawArg::Storage {
                        buffer,
                        bytes: *bytes,
                        writable: *writable,
                    },
                    GraphArg::Uniform(data) => RawArg::Uniform(data),
                })
                .collect();
            let workgroups = workgroups_of(launch.dispatch.dispatch())?;
            let elapsed = self.run(&launch.kernel, workgroups, launch.size, &args)?;
            self.sample(&launch.kernel, elapsed);
        }
        if lock(&self.profiler).is_some() {
            self.dispatches.store(0, Ordering::Relaxed);
            return Ok(0);
        }
        Ok(self.dispatches.fetch_add(_dispatches, Ordering::Relaxed) + _dispatches)
    }

    fn pending_dispatches(&self) -> u32 {
        self.dispatches.load(Ordering::Relaxed)
    }

    fn gpu_busy(&self) -> bool {
        false
    }

    fn notify_released(&self, _buffers: &[Buffer]) {}

    fn flush_batch(&self) -> Result<(), BackendError> {
        self.dispatches.store(0, Ordering::Relaxed);
        self.armed.store(false, Ordering::Relaxed);
        Ok(())
    }

    fn can_time_launches(&self) -> bool {
        true
    }

    fn time_next_launch(&self, tag: u64) {
        self.armed_tag.store(tag, Ordering::Relaxed);
        self.armed.store(true, Ordering::Release);
    }

    fn drain_launch_times(&self, out: &mut Vec<(u64, Duration)>) {
        out.append(&mut lock(&self.launch_times));
    }

    fn sync(&self, _spin_budget: Duration) -> Result<Duration, BackendError> {
        self.dispatches.store(0, Ordering::Relaxed);
        Ok(Duration::ZERO)
    }

    fn read_into(
        &self,
        src: &Buffer,
        dst: &mut [MaybeUninit<u8>],
        _spill: bool,
        _spin_budget: Duration,
    ) -> Result<Duration, BackendError> {
        if dst.len() as u64 > src.size() {
            return Err(invalid("readback larger than its source buffer"));
        }
        // SAFETY: `dst.len() <= src.size()`; no launch is running.
        copy_parallel_host(unsafe { raw(src)?.as_slice(dst.len()) }, dst);
        self.dispatches.store(0, Ordering::Relaxed);
        Ok(Duration::ZERO)
    }

    fn reclaim_idle(&self) -> Result<(), BackendError> {
        self.dispatches.store(0, Ordering::Relaxed);
        lock(&self.mappable).drain();
        Ok(())
    }

    fn enable_profiling(&self) -> Result<(), BackendError> {
        *lock(&self.profiler) = Some(Vec::new());
        Ok(())
    }

    #[allow(deprecated)]
    fn finish_profiling(&self, splits: Vec<luma_core::FusionSplit>) -> Result<Profile, BackendError> {
        self.dispatches.store(0, Ordering::Relaxed);
        let Some(samples) = lock(&self.profiler).take() else {
            return Err(BackendError::Profiling("profiling is not enabled".into()));
        };
        let mut profile = Profile { splits, ..Profile::default() };
        profile.add_samples(samples);
        Ok(profile)
    }

    fn is_profiling(&self) -> Result<bool, BackendError> {
        Ok(lock(&self.profiler).is_some())
    }

    fn native_line_solve(&self) -> bool {
        true
    }

    fn native_matmul(&self) -> Option<u32> {
        Some(u32::try_from(self.pool.threads()).unwrap_or(u32::MAX))
    }

    fn compile_matmul(&self, spec: &MatmulSpec) -> Option<RawKernel> {
        NativeMatmul::new(spec).map(RawKernel::new)
    }

    fn compile_matmul_epilogue(&self, spec: &MatmulSpec, tile: &Arc<KernelModule>) -> Option<RawKernel> {
        if spec.half || self.check_module(tile, None, &tile.label).is_err() {
            return None;
        }
        let (spec, tile, disk) = (*spec, Arc::clone(tile), Arc::clone(&self.disk));
        let max_shared = self.info.limits.max_workgroup_storage_bytes;
        // The tile kernel compiles on a worker thread, like the elementwise kernels: the launch that needs it waits (or runs the product
        // plain while it compiles, which is the caller's choice, see `RawKernel::is_ready`).
        Some(RawKernel::spawn(move || {
            let bindings = tile.bindings.len();
            // The uniform is the last binding; its scalars are the words after the eight of the tile's own.
            let scalars = match tile.bindings.last().map(|binding| &binding.elem_ty) {
                Some(luma_core::kir::KirType::Struct(at)) => tile.structs[*at as usize].fields.len().saturating_sub(8),
                _ => 0,
            };
            let compiled = compile_native(&disk, max_shared, &tile, None, &tile.label, luma_core::KernelTrust::Trusted)?;
            NativeMatmul::with_epilogue(&spec, compiled, bindings, scalars).ok_or_else(|| BackendError::Unsupported("the epilogue of a native product does not fit a tile".into()))
        }))
    }

    fn compile_batched_matmul(&self) -> Option<RawKernel> {
        Some(RawKernel::new(NativeBatchedMatmul::default()))
    }

    fn compile_line_solve(&self, spec: &LineSolveSpec) -> Option<RawKernel> {
        NativeLineSolve::new(spec).map(RawKernel::new)
    }

    fn capture_graph(&self, launches: &[GraphLaunch<'_>]) -> Option<Arc<dyn ExecGraph>> {
        // The launch an armed tag was meant for is captured, not run: the tag is dropped.
        self.armed.store(false, Ordering::Relaxed);
        let launches = launches
            .iter()
            .map(|l| CapturedCpuLaunch {
                kernel: l.kernel.clone(),
                dispatch: match l.dispatch {
                    Dispatch::Direct(workgroups) => CapturedDispatch::Direct(workgroups),
                    Dispatch::Indirect {
                        buffer,
                        offset,
                        max_workgroups,
                    } => CapturedDispatch::Indirect {
                        buffer: buffer.clone(),
                        offset,
                        max_workgroups,
                    },
                },
                size: l.size,
                args: l
                    .args
                    .iter()
                    .map(|arg| match *arg {
                        RawArg::Storage {
                            buffer,
                            bytes,
                            writable,
                        } => GraphArg::Storage {
                            buffer: buffer.clone(),
                            bytes,
                            writable,
                        },
                        RawArg::Uniform(data) => GraphArg::Uniform(data.into()),
                    })
                    .collect(),
            })
            .collect();
        Some(Arc::new(CpuExecGraph { launches }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn size_classes_use_eighth_octaves_and_clamp() {
        assert_eq!(size_class(1, 1 << 20), Some(16));
        assert_eq!(size_class(17, 1 << 20), Some(32));
        assert_eq!(size_class(5000, 1 << 20), Some(5120));
        assert_eq!(size_class((1 << 20) + 1, 1 << 20), None);
        assert_eq!(size_class(600, 1000), Some(1000 & !3));
    }

    #[test]
    fn disk_cache_persists_and_reloads_compiled_blob() {
        use luma_core::kir::{BuiltinVar, KirType, ModuleBuilder};
        let mut mb = ModuleBuilder::new("disk_cache_cpu_test", "main", [4, 1, 1]);
        let out = mb.add_storage_read_write("out", KirType::U32);
        let mut b = mb.entry_builder();
        let gid = b.builtin(BuiltinVar::GlobalInvocationId);
        let i = b.field(gid, 0, KirType::U32);
        b.store(out, i, i + b.lit_u32(7));
        b.finish();
        let m = mb.finish();

        let backend = open().unwrap();
        let k1 = backend
            .compile_kernel(&m, None, "disk_cache_cpu_test", luma_core::KernelTrust::Trusted)
            .unwrap();
        let k2 = backend
            .compile_kernel(&m, None, "disk_cache_cpu_test", luma_core::KernelTrust::Trusted)
            .unwrap();
        let c1 = k1.downcast_ref::<CompiledCpuKernel>().unwrap();
        let c2 = k2.downcast_ref::<CompiledCpuKernel>().unwrap();
        assert_eq!(c1.exec.bytes(), c2.exec.bytes());
    }

    /// A one-store kernel, compiled and launched through the backend API.
    fn store_kernel(backend: &dyn Backend) -> (CompiledKernel, Buffer) {
        use luma_core::kir::{BuiltinVar, KirType, ModuleBuilder};
        use luma_core::{BindingInfo, BindingKind, BufferUsage};
        let mut mb = ModuleBuilder::new("launch_timing_cpu_test", "main", [4, 1, 1]);
        let out = mb.add_storage_read_write("out", KirType::U32);
        let mut b = mb.entry_builder();
        let gid = b.builtin(BuiltinVar::GlobalInvocationId);
        let i = b.field(gid, 0, KirType::U32);
        b.store(out, i, i + b.lit_u32(1));
        b.finish();
        let m = mb.finish();
        let raw = backend
            .compile_kernel(&m, None, "launch_timing_cpu_test", luma_core::KernelTrust::Trusted)
            .unwrap();
        let bindings = vec![BindingInfo {
            name: "out".into(),
            kind: BindingKind::StorageReadWrite,
            element_stride: Some(4),
            min_size: None,
        }];
        let kernel = CompiledKernel::new("launch_timing_cpu_test".into(), raw, bindings, 1, None, [4, 1, 1]);
        let buffer = backend.create_buffer(16, BufferUsage::STORAGE, false).unwrap();
        (kernel, buffer)
    }

    fn launch(backend: &dyn Backend, kernel: &CompiledKernel, buffer: &Buffer) {
        backend
            .record_launch(kernel, luma_core::Dispatch::Direct([1, 1, 1]), [4, 1, 1], &[luma_core::RawArg::write(buffer, 16)])
            .unwrap();
    }

    #[test]
    fn an_armed_launch_reports_a_time() {
        let backend = open().unwrap();
        assert!(backend.can_time_launches());
        let (kernel, buffer) = store_kernel(&*backend);
        launch(&*backend, &kernel, &buffer);
        let mut out = Vec::new();
        backend.drain_launch_times(&mut out);
        assert!(out.is_empty(), "an unarmed launch reports nothing");
        backend.time_next_launch(7);
        launch(&*backend, &kernel, &buffer);
        backend.drain_launch_times(&mut out);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].0, 7);
        assert!(out[0].1 > Duration::ZERO);
        // The arm was consumed by the one launch.
        launch(&*backend, &kernel, &buffer);
        out.clear();
        backend.drain_launch_times(&mut out);
        assert!(out.is_empty());
    }

    #[test]
    fn a_flushed_arm_is_dropped() {
        let backend = open().unwrap();
        let (kernel, buffer) = store_kernel(&*backend);
        backend.time_next_launch(9);
        backend.flush_batch().unwrap();
        launch(&*backend, &kernel, &buffer);
        let mut out = Vec::new();
        backend.drain_launch_times(&mut out);
        assert!(out.is_empty());
    }
}
