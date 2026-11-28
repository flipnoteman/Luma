//! The CUDA runtime: a device pool, buffers, NVVM compilation (the driver's own NVVM library) with
//! a PTX disk cache, one ordered
//! stream per backend and CUDA graph capture and replay. Device memory comes from a dedicated
//! stream-ordered memory pool (`cuMemAllocFromPoolAsync` / `cuMemFreeAsync`) so that freeing a
//! buffer never synchronizes the device; devices without pools use `cuMemAlloc` / `cuMemFree`.
//!
//! Work goes onto the stream as soon as it is recorded; a "batch" is only the set of resources the
//! work since the last [`Backend::flush_batch`] uses. Flushing records an event after it, and
//! everything the batch used (buffers, kernels, pinned staging memory, uniform slots, graphs)
//! stays alive until that event completes.

use std::borrow::Cow;
use std::collections::VecDeque;
use std::ffi::c_void;
use std::fmt;
use std::mem::MaybeUninit;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, LazyLock, Mutex, MutexGuard, OnceLock, PoisonError, Weak};
use std::time::{Duration, Instant};

use cudarc::driver::{CudaContext, CudaEvent, CudaFunction, CudaGraph, CudaStream, DriverError, result, sys};
use cudarc::nvrtc::Ptx;
use luma_core::hal::cache::{self, PTX_MAGIC};
use luma_core::hal::{copy_parallel, size_class};
use luma_core::kir::{self, KernelModule};
use luma_core::{
    Backend, BackendError, BackendKind, Buffer, BufferUsage, CompiledKernel, CostModel, DeviceInfo, DeviceLimits, DiskCache, MemoryNow, Dispatch, ExecGraph, Features, GraphLaunch, TextureCaps, TextureDesc, TextureFilter,
    KernelTrust, NativeBuffer, NativeKernel, NativeStage, Profile, RawArg, RawKernel, GemmSpec, Stage, StageDir, Ticket, TuningProfile,
};
use luma_core::hal::check_transfer;

use crate::codegen::{self, CudaArgAbi};

/// Pieces an upload larger than this is staged in. The copy of one piece into pinned memory
/// overlaps the DMA of the previous one, and a piece's staging block is reused as soon as its
/// DMA is done, so a transfer of any size cycles through a few blocks. Measured on an RTX 3090
/// (PCIe 4.0 x16), 4 / 16 / 64 / 256 MiB uploads reach 17.8 / 23.2 / 24.7 / 22.4 GB/s with 2 MiB
/// pieces, 14.1 / 18.1 / 23.3 / 23.7 with 8 MiB ones and 13.9 / 14.3 / 18.0 / 21.3 with 32 MiB.
const UPLOAD_CHUNK: usize = 2 << 20;

/// Largest piece a readback is split into.
const READ_CHUNK: usize = 8 << 20;

/// Smallest piece of a split readback; one that is shorter than twice this is not split.
const READ_CHUNK_MIN: usize = 1 << 20;

/// A readback is split in about this many pieces (within the two limits above).
const READ_PIECES: usize = 8;

/// Readback pieces queued at once (copies overlap the host copying the previous one out).
const READ_WINDOW: usize = 4;

/// Pieces of a spilling readback (the device is full), one staging block at a time.
const SPILL_CHUNK: usize = 32 << 20;

/// Pinned staging memory held by unfinished work, above which a new request waits for it. New
/// pinned memory costs 0.4 ms per MiB, so waiting for the GPU to finish earlier copies is
/// cheaper than staging further ahead.
const PINNED_BUSY_CAP: usize = 128 << 20;

/// Most bytes of idle pinned staging memory kept for reuse.
const PINNED_IDLE_CAP: usize = 128 << 20;

/// Smallest pinned staging allocation.
const PINNED_MIN: usize = 4096;

/// An idle block serves a request up to this many times smaller than itself: small requests
/// get small blocks of their own instead of pinning down the big ones.
const PINNED_FIT: usize = 4;

/// Uploads whose staging block is at least this large record an event of their own, so that the
/// block is reused when its copy is done and not only when the whole batch is.
const MARK_MIN: usize = 256 << 10;

/// Size of the chunks that uniforms too large to pass by value are copied into.
const UNIFORM_CHUNK: usize = 1 << 20;

/// Alignment of a uniform slot in device memory.
const UNIFORM_ALIGN: usize = 256;

/// `LUMA_CUDA_DUMP`: a directory that receives the NVVM IR (`.ll`), the PTX (`.ptx`) and the label
/// (`.label`) of every kernel the backend builds or loads from the cache, named like the cache entry.
static DUMP_DIR: LazyLock<Option<std::path::PathBuf>> = LazyLock::new(|| std::env::var_os("LUMA_CUDA_DUMP").filter(|v| !v.is_empty()).map(Into::into));

/// Smallest device allocation.
const MIN_BUFFER: u64 = 16;

/// The entry point of every generated kernel.
const ENTRY: &str = "luma_main";

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

fn gpu(error: DriverError) -> BackendError {
    BackendError::Gpu(error.to_string())
}

fn poll(error: DriverError) -> BackendError {
    BackendError::Poll(error.to_string())
}

/// Makes `ctx` current on this thread, without the sticky-error check of
/// [`CudaContext::bind_to_thread`] (used on the paths that free memory and must not leak it).
fn bind_raw(ctx: &CudaContext) -> Result<(), DriverError> {
    if result::ctx::get_current()? != Some(ctx.cu_ctx()) {
        // SAFETY: `cu_ctx` is the live primary context `ctx` retains.
        unsafe { result::ctx::set_current(ctx.cu_ctx()) }?;
    }
    Ok(())
}

fn is_out_of_memory(error: &DriverError) -> bool {
    error.0 == sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY
}

/// A dedicated stream-ordered memory pool (the device's default pool is left alone: other
/// libraries of the process may use it). Its release threshold is unlimited, so memory freed
/// by buffers stays reserved for reuse instead of going back to the driver at every
/// synchronization.
#[derive(Debug)]
struct DevicePool {
    pool: sys::CUmemoryPool,
    ctx: Arc<CudaContext>,
}

// SAFETY: a pool handle is an opaque driver handle usable from any thread.
unsafe impl Send for DevicePool {}
unsafe impl Sync for DevicePool {}

impl DevicePool {
    /// Creates the pool on `ctx`'s device; `None` when the driver refuses, in which case the
    /// backend allocates with `cuMemAlloc`.
    fn create(ctx: &Arc<CudaContext>) -> Option<Arc<DevicePool>> {
        // SAFETY: an all-zero properties struct is valid (null pointers, zero enums, reserved
        // bytes); the fields that matter are set below.
        let mut props: sys::CUmemPoolProps = unsafe { std::mem::zeroed() };
        props.allocType = sys::CUmemAllocationType::CU_MEM_ALLOCATION_TYPE_PINNED;
        props.handleTypes = sys::CUmemAllocationHandleType::CU_MEM_HANDLE_TYPE_NONE;
        props.location.type_ = sys::CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE;
        props.location.id = ctx.cu_device();
        let mut pool: sys::CUmemoryPool = std::ptr::null_mut();
        // SAFETY: `props` is fully initialized and `pool` receives the handle.
        unsafe { sys::cuMemPoolCreate(&mut pool, &props) }.result().ok()?;
        let pool = DevicePool { pool, ctx: Arc::clone(ctx) };
        let mut threshold = u64::MAX;
        // SAFETY: the release threshold attribute takes one `cuuint64_t`.
        unsafe { sys::cuMemPoolSetAttribute(pool.pool, sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_RELEASE_THRESHOLD, (&raw mut threshold).cast()) }.result().ok()?;
        Some(Arc::new(pool))
    }
}

impl Drop for DevicePool {
    fn drop(&mut self) {
        if bind_raw(&self.ctx).is_ok() {
            // SAFETY: waits for queued frees, then destroys the pool once; the handle is not
            // used afterwards.
            unsafe {
                let _ = sys::cuCtxSynchronize();
                let _ = sys::cuMemPoolDestroy(self.pool);
            }
        }
    }
}

/// What a pool-allocated block needs to be freed in stream order.
#[derive(Debug)]
struct Ordered {
    stream: Arc<CudaStream>,
    /// Keeps the pool alive until the block is freed.
    _pool: Arc<DevicePool>,
}

/// A device allocation, freed when dropped.
#[derive(Debug)]
struct CudaBuffer {
    dptr: sys::CUdeviceptr,
    ctx: Arc<CudaContext>,
    /// The stream the block was allocated on (and is freed on) when it comes from the pool;
    /// `None` for a `cuMemAlloc` block.
    ordered: Option<Ordered>,
}

impl NativeBuffer for CudaBuffer {}

impl Drop for CudaBuffer {
    fn drop(&mut self) {
        if bind_raw(&self.ctx).is_ok() {
            // Luma drops a buffer only once no recorded work needs it, so ordering the free
            // after the stream's pending work is enough and the host never waits for it.
            let _ = match &self.ordered {
                // SAFETY: allocated from the pool on this stream and freed once.
                Some(ordered) => unsafe { sys::cuMemFreeAsync(self.dptr, ordered.stream.cu_stream()) }.result(),
                // SAFETY: allocated by `cuMemAlloc` and freed once; the driver orders the free
                // after work still using it, and unfinished work is anyway kept alive by its batch.
                None => unsafe { result::free_sync(self.dptr) },
            };
        }
    }
}

/// Page-locked host memory, freed when dropped.
#[derive(Debug)]
struct Pinned {
    ptr: *mut u8,
    cap: usize,
    ctx: Arc<CudaContext>,
}

// SAFETY: the allocation is plain host memory owned by this value; a freeing thread binds the
// context first.
unsafe impl Send for Pinned {}
unsafe impl Sync for Pinned {}

impl Drop for Pinned {
    fn drop(&mut self) {
        if bind_raw(&self.ctx).is_ok() {
            // SAFETY: allocated by `cuMemHostAlloc` and freed once.
            let _ = unsafe { result::free_host(self.ptr.cast()) };
        }
    }
}

/// A [`Stage`] on a pinned block taken out of the pool for the stage's lifetime.
#[derive(Debug)]
struct CudaStage {
    /// `None` only while dropping.
    pinned: Option<Pinned>,
    len: usize,
    /// The last copy that read or wrote the block: the block goes back to the pool only after it.
    last: Mutex<Option<Arc<CudaEvent>>>,
    state: Weak<Mutex<State>>,
}

impl NativeStage for CudaStage {
    fn bytes(&self) -> &[u8] {
        match &self.pinned {
            // SAFETY: the block holds at least `len` initialized bytes and lives as long as `self`.
            Some(pinned) => unsafe { std::slice::from_raw_parts(pinned.ptr, self.len) },
            None => &[],
        }
    }

    fn bytes_mut(&mut self) -> &mut [u8] {
        match &self.pinned {
            // SAFETY: as above, and `&mut self` makes the host the only user.
            Some(pinned) => unsafe { std::slice::from_raw_parts_mut(pinned.ptr, self.len) },
            None => &mut [],
        }
    }
}

impl Drop for CudaStage {
    fn drop(&mut self) {
        // Drains the lane: the block must not be reused (or freed) under a copy still running.
        if let Some(event) = lock(&self.last).take() {
            let _ = event.synchronize();
        }
        let Some(pinned) = self.pinned.take() else { return };
        match self.state.upgrade() {
            Some(state) => keep_pinned(&mut lock(&state), pinned),
            None => drop(pinned),
        }
    }
}

/// The completion of a copy on the transfer lane, or of the compute lane up to a barrier.
#[derive(Debug)]
struct CudaTicket(Arc<CudaEvent>);

/// Returns a pinned block to the idle pool. When it would push the pool over [`PINNED_IDLE_CAP`],
/// idle blocks larger than it are freed (largest first) to make room, and if it still does not
/// fit it is dropped: small blocks stay pooled, and big ones can still serve small requests.
fn keep_pinned(st: &mut State, pinned: Pinned) {
    while st.pinned_idle + pinned.cap > PINNED_IDLE_CAP {
        let Some(at) = st.pinned_free.iter().enumerate().filter(|(_, p)| p.cap > pinned.cap).max_by_key(|(_, p)| p.cap).map(|(at, _)| at) else {
            return;
        };
        let larger = st.pinned_free.swap_remove(at);
        st.pinned_idle -= larger.cap;
    }
    st.pinned_idle += pinned.cap;
    st.pinned_free.push(pinned);
}

fn check_mask(module: &KernelModule, used: Option<&[bool]>) -> Result<(), BackendError> {
    match used {
        Some(mask) if mask.len() != module.bindings.len() => {
            Err(BackendError::KernelBinding(format!("the used-binding mask has {} entries for {} bindings", mask.len(), module.bindings.len())))
        }
        _ => Ok(()),
    }
}

/// Where the time of building a kernel's function went (logged at debug level per kernel).
struct Phases {
    /// NVVM (IR to PTX); `None` when the PTX came from the disk cache.
    nvvm: Option<Duration>,
    /// `cuModuleLoadData` (PTX to SASS by the driver's JIT, or the driver's own cache) and the function lookup.
    load: Duration,
    ptx_bytes: usize,
}

impl fmt::Display for Phases {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.nvvm {
            Some(nvvm) => write!(f, "NVVM {:.1} ms ({} bytes of PTX), module load {:.1} ms", nvvm.as_secs_f64() * 1e3, self.ptx_bytes, self.load.as_secs_f64() * 1e3),
            None => write!(f, "PTX from the disk cache, module load {:.1} ms", self.load.as_secs_f64() * 1e3),
        }
    }
}

/// Generates, compiles and loads the kernel of `module`; runs on the calling thread, which may be
/// a compile worker: the device context is bound to it here.
fn compile_native(shared: &CudaShared, module: &KernelModule, used: Option<&[bool]>, label: &str, trust: KernelTrust) -> Result<CudaKernel, BackendError> {
    let started = Instant::now();
    let checked: Cow<'_, KernelModule> = if trust.checks_bounds() { kir::opt::with_bounds_checks(module) } else { Cow::Borrowed(module) };
    let optimized = kir::optimize_for_target(&checked, true);
    // `with_bounds_checks` already guards every workgroup-array index it cannot prove in range,
    // so the emitter's own clamp (a `min` per access) would only repeat it.
    let translated = codegen::emit_with_options(&optimized, used, trust.zero_workgroup_memory(), false);
    let codegen_time = started.elapsed();
    let strides: Vec<u32> = translated.array_lengths.iter().map(|&b| optimized.bindings.get(b as usize).and_then(|d| d.element_stride).unwrap_or(1)).collect();

    shared.ctx.bind_to_thread().map_err(gpu)?;
    let (func, phases) = shared.build_function(&translated.source, label)?;
    log::debug!(
        "luma-cuda: kernel `{label}`: codegen {:.1} ms ({} bytes of IR), {}",
        codegen_time.as_secs_f64() * 1e3,
        translated.source.len(),
        phases
    );
    Ok(CudaKernel { func, workgroup: optimized.workgroup_size, shared_bytes: 0, param_abis: translated.param_abis, array_lengths: translated.array_lengths, strides })
}

/// Sampled launches kept until they are drained.
const MAX_TIMED: usize = 64;

/// The output tiles of the tensor-core products, largest first (see [`crate::tensor`]).
const TENSOR_TILES: [(u32, u32); 4] = [(128, 128), (128, 64), (64, 128), (64, 64)];

/// The matrix product of `spec` (tensor cores, or `f32` on the CUDA cores), loaded from its PTX.
///
/// With `hook`, the module of `spec.hook` (see [`luma_core::HookId`]): its function is compiled by NVVM like any elementwise
/// kernel's body, and its PTX is expanded at the stores of the tensor-core kernel ([`crate::ptx::Hook`]), which then takes the hook's
/// operands as extra parameters and its scalars in the uniform.
fn compile_gemm_ptx(shared: &CudaShared, spec: &GemmSpec, hook: Option<&KernelModule>) -> Result<CudaKernel, BackendError> {
    let mut extra = (0u32, 0u32);
    let (source, threads, bytes) = if let Some(module) = hook {
        if spec.exact {
            return Err(BackendError::Unsupported("the exact product takes no epilogue hook".into()));
        }
        let started = Instant::now();
        let optimized = kir::optimize_for_target(module, true);
        let ir = codegen::emit_hook(&optimized).map_err(|e| BackendError::Unsupported(format!("epilogue hook: {e}")))?;
        let (major, minor) = shared.cc;
        let operands = module.bindings.len().saturating_sub(1) as u32;
        let scalars = module.functions.first().map_or(0, |f| f.params.len().saturating_sub(4)) as u32;
        let text = shared.nvvm.compile(&ir, major, minor).map_err(|e| BackendError::ShaderCompile(format!("epilogue hook of {spec:?}:\n{e}")))?;
        let hook = crate::ptx::Hook::parse(&text, operands, scalars).map_err(|e| BackendError::Unsupported(format!("epilogue hook: {e}")))?;
        let config = crate::tensor::Config::of(spec);
        let ptx = crate::tensor::ptx_hooked(&config, Some(&hook)).map_err(|e| BackendError::Unsupported(format!("epilogue hook: {e}")))?;
        log::debug!("luma-cuda: epilogue hook of {spec:?}: NVVM and splice {:.1} ms", started.elapsed().as_secs_f64() * 1e3);
        extra = (operands, scalars);
        (hook.reconcile(&ptx), config.threads(), config.shared_bytes())
    } else if spec.exact {
        let config = crate::sgemm::Config::of(spec);
        (crate::sgemm::ptx(&config), config.threads(), config.shared_bytes())
    } else {
        let config = crate::tensor::Config::of(spec);
        (crate::tensor::ptx(&config), config.threads(), config.shared_bytes())
    };
    if let Some(dir) = DUMP_DIR.as_deref() {
        // Best effort: a debugging aid must not fail a compile.
        let _ = std::fs::create_dir_all(dir);
        let lanes = match spec.lanes {
            luma_core::GemmLanes::Vec4 => "v4",
            luma_core::GemmLanes::Vec4Tails => "tails",
            _ => "scalar",
        };
        let name = format!("{}-{}x{}-{}{}-{lanes}{}.ptx", if spec.exact { "sgemm" } else { "tensor" }, spec.rows, spec.cols, u8::from(spec.lhs_transposed), u8::from(spec.rhs_transposed), spec.hook.map_or(String::new(), |id| format!("-hook{:08x}", id.0 as u32)));
        let _ = std::fs::write(dir.join(name), &source);
    }
    shared.ctx.bind_to_thread().map_err(gpu)?;
    let module = shared.ctx.load_module(Ptx::from_src(source)).map_err(gpu)?;
    let func = module.load_function("main").map_err(gpu)?;
    if bytes > 48 * 1024 {
        func.set_attribute(sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, bytes as i32).map_err(gpu)?;
    }
    if DUMP_DIR.is_some() {
        // A tuning aid: registers, spills and resident workgroups per multiprocessor.
        let spilled = func.get_attribute(sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES).unwrap_or(-1);
        eprintln!(
            "gemm-info {} {}x{} t{}{}: {} registers, {} bytes of local memory, {} bytes of shared memory, {:?} workgroups per multiprocessor",
            if spec.exact { "f32" } else { "tf32" },
            spec.rows,
            spec.cols,
            u8::from(spec.lhs_transposed),
            u8::from(spec.rhs_transposed),
            func.num_regs().unwrap_or(-1),
            spilled,
            bytes,
            func.occupancy_max_active_blocks_per_multiprocessor(threads, bytes as usize, None).ok()
        );
    }
    Ok(CudaKernel {
        func,
        workgroup: [threads, 1, 1],
        shared_bytes: bytes,
        param_abis: [CudaArgAbi::DevicePtr; 3]
            .into_iter()
            .chain(std::iter::repeat_n(CudaArgAbi::DevicePtr, extra.0 as usize))
            .chain([CudaArgAbi::InlineUniform { size: crate::gemm_copy::UNIFORM_BYTES + 4 * extra.1 as usize }])
            .collect(),
        array_lengths: Vec::new(),
        strides: Vec::new(),
    })
}

/// A compiled kernel.
#[derive(Debug)]
struct CudaKernel {
    func: CudaFunction,
    workgroup: [u32; 3],
    /// Dynamic shared memory of a workgroup.
    shared_bytes: u32,
    param_abis: Vec<CudaArgAbi>,
    /// The bindings whose length is passed, and the element stride dividing their bound bytes.
    array_lengths: Vec<u32>,
    strides: Vec<u32>,
}

impl NativeKernel for CudaKernel {}

/// What a graph keeps alive: everything its kernel nodes read, write or run.
#[derive(Debug)]
struct GraphRes {
    #[allow(dead_code)]
    buffers: Vec<Buffer>,
    #[allow(dead_code)]
    kernels: Vec<RawKernel>,
    /// Device copy of the uniforms too large to pass by value.
    #[allow(dead_code)]
    scratch: Option<CudaBuffer>,
    /// Size of `scratch`.
    scratch_bytes: u64,
}

struct GraphCell(CudaGraph);

// SAFETY: a `CudaGraph` must not be used from two threads at once; every access goes through the
// `Mutex` in `CudaExecGraph`, and the graph is only used with its own context, which is bound
// first.
unsafe impl Send for GraphCell {}

impl fmt::Debug for GraphCell {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("CudaGraph")
    }
}

/// A captured sequence of launches.
#[derive(Debug)]
struct CudaExecGraph {
    exec: Mutex<GraphCell>,
    res: Arc<GraphRes>,
    ctx: Arc<CudaContext>,
}

impl ExecGraph for CudaExecGraph {
    fn device_bytes(&self) -> u64 {
        self.res.scratch_bytes
    }
}

impl Drop for CudaExecGraph {
    fn drop(&mut self) {
        // The graph's own drop calls the driver; it needs the context current. An executing
        // graph is released by the driver once it completes.
        let _ = bind_raw(&self.ctx);
    }
}

/// Resources of the work since a flush.
#[derive(Default)]
struct Held {
    buffers: Vec<Buffer>,
    kernels: Vec<RawKernel>,
    pinned: Vec<Pinned>,
    scratch: Vec<CudaBuffer>,
    graphs: Vec<Arc<GraphRes>>,
}

/// A block of inline uniform bytes; keeps every inline uniform 16-byte aligned.
#[allow(dead_code)]
#[derive(Clone, Copy)]
#[repr(align(16))]
struct Aligned16([u8; 16]);

/// The kernel parameters of one launch, kept between launches to avoid allocating.
#[derive(Default)]
struct Params {
    grid: [u32; 4],
    indirect: u64,
    vals: Vec<u64>,
    lens: Vec<u32>,
    inline: Vec<Aligned16>,
    ptrs: Vec<*mut c_void>,
}


// SAFETY: `ptrs` points into the other fields of the same value and is rebuilt for every launch.
unsafe impl Send for Params {}

#[path = "texture.rs"]
mod texture;
use texture::CudaTexture;

/// The device address that tells the buffer apart from every other one a launch touches (a
/// texture: that of the buffer it reads, or of its own copy).
fn access_key(buffer: &Buffer) -> Result<u64, BackendError> {
    match buffer.downcast_ref::<CudaTexture>() {
        Some(texture) => Ok(texture.key()),
        None => device_ptr(buffer),
    }
}

fn device_ptr(buffer: &Buffer) -> Result<sys::CUdeviceptr, BackendError> {
    buffer.downcast_ref::<CudaBuffer>().map(|b| b.dptr).ok_or_else(|| BackendError::KernelBinding("a buffer of another backend was passed to a CUDA kernel".into()))
}

/// The grid a dispatch launches (an indirect one launches its upper bound; the kernel itself
/// retires the workgroups beyond the counts), the device address of its counts (0 when direct)
/// and the buffer holding them.
fn resolve_dispatch<'a>(dispatch: Dispatch<'a>) -> Result<([u32; 3], u64, Option<&'a Buffer>), BackendError> {
    match dispatch {
        Dispatch::Direct(workgroups) => Ok((workgroups, 0, None)),
        Dispatch::Indirect { buffer, offset, max_workgroups } => Ok((max_workgroups, device_ptr(buffer)? + offset, Some(buffer))),
    }
}

/// The accesses to one buffer since it was last written, while a graph is captured.
#[derive(Default)]
struct Touch {
    /// The launch that wrote it last.
    writer: Option<usize>,
    /// The launches that read it since.
    readers: Vec<usize>,
}

/// Orders the launches of a graph by the buffers they touch: a launch follows the last writer of
/// every buffer it touches (read after write, write after write) and, when it writes one, every
/// reader since (write after read). Launches that share no buffer, or only read one together,
/// get no edge between them and may run concurrently. Buffers are told apart by device address:
/// every buffer is a block of its own, and a pool buffer handed out again is the same address.
#[derive(Default)]
struct Hazards {
    touched: std::collections::HashMap<u64, Touch>,
    accesses: Vec<(u64, bool)>,
}

impl Hazards {
    /// Records the next launch (numbered `index`) and puts into `after` the earlier launches it
    /// must follow, sorted and without repeats. `args` are the buffers it binds (device address,
    /// whether it may write it), `counts` the buffer holding its workgroup counts, which it reads.
    fn follow(&mut self, index: usize, args: impl Iterator<Item = (u64, bool)>, counts: Option<u64>, after: &mut Vec<usize>) {
        self.accesses.clear();
        self.accesses.extend(args);
        self.accesses.extend(counts.map(|ptr| (ptr, false)));
        // One entry per buffer, a write if any binding of it is one.
        self.accesses.sort_unstable();
        self.accesses.dedup_by(|later, earlier| {
            let same = later.0 == earlier.0;
            if same {
                earlier.1 |= later.1;
            }
            same
        });
        after.clear();
        for &(ptr, writes) in &self.accesses {
            let touch = self.touched.entry(ptr).or_default();
            after.extend(touch.writer);
            if writes {
                after.extend(touch.readers.iter().copied());
                touch.writer = Some(index);
                touch.readers.clear();
            } else {
                touch.readers.push(index);
            }
        }
        after.sort_unstable();
        after.dedup();
    }
}

/// Alignment of the offset of a bound storage range (the rule for vector loads).
const STORAGE_OFFSET_ALIGN: u64 = 16;

/// Most launches looked at to find the edges a launch has through others (see [`reduce`]).
const REDUCE_BUDGET: usize = 512;

/// Drops from `after` (sorted) the launches another one in it already follows, through `preds`
/// (what each earlier launch follows): an edge to such a launch only costs the driver a wait.
/// Looks at no more than [`REDUCE_BUDGET`] launches; what it did not reach stays.
fn reduce(preds: &[Vec<usize>], after: &mut Vec<usize>, seen: &mut [u32], epoch: u32, stack: &mut Vec<usize>) {
    let Some(&low) = after.first() else { return };
    if after.len() < 2 {
        return;
    }
    stack.clear();
    for &d in after.iter() {
        stack.extend(preds[d].iter().copied().filter(|&p| p >= low));
    }
    let mut budget = REDUCE_BUDGET;
    while let Some(n) = stack.pop() {
        if seen[n] == epoch {
            continue;
        }
        if budget == 0 {
            break;
        }
        budget -= 1;
        seen[n] = epoch;
        stack.extend(preds[n].iter().copied().filter(|&p| p >= low));
    }
    after.retain(|&d| seen[d] != epoch);
}

/// How many launches a graph needs per launch of its longest chain for its launches to be
/// captured with edges of their own: a graph that is mostly a chain is captured as one (measured
/// on an RTX 3090: as fast as with edges, and without the driver's cost of branches).
const MIN_PARALLELISM: f64 = 1.5;

/// What every launch of `launches` that runs follows (indices among those that run), from the
/// buffers they touch; `None` when the launches have too little to run side by side to pay for
/// the edges (see [`MIN_PARALLELISM`]).
fn capture_order(launches: &[GraphLaunch<'_>], dispatches: &[([u32; 3], u64, Option<&Buffer>)]) -> Result<Option<Vec<Vec<usize>>>, BackendError> {
    let mut hazards = Hazards::default();
    let mut preds: Vec<Vec<usize>> = Vec::new();
    let mut level: Vec<u32> = Vec::new();
    let mut seen = vec![0u32; launches.len()];
    let (mut after, mut stack, mut accesses) = (Vec::new(), Vec::new(), Vec::new());
    for (launch, &(grid, _, counts)) in launches.iter().zip(dispatches) {
        if grid.contains(&0) {
            continue;
        }
        accesses.clear();
        for (_, arg) in launch.kernel.bound(launch.args) {
            if let RawArg::Storage { buffer, writable, .. } = arg {
                accesses.push((access_key(buffer)?, *writable));
            }
        }
        hazards.follow(preds.len(), accesses.iter().copied(), counts.map(device_ptr).transpose()?, &mut after);
        reduce(&preds, &mut after, &mut seen, preds.len() as u32 + 1, &mut stack);
        level.push(1 + after.iter().map(|&d| level[d]).max().unwrap_or(0));
        preds.push(after.clone());
    }
    let longest = level.iter().copied().max().unwrap_or(0);
    Ok((preds.len() as f64 >= MIN_PARALLELISM * f64::from(longest)).then_some(preds))
}

/// Makes the next launch captured on `stream` follow exactly the nodes `after` (indices into
/// `nodes`), `current` being what the stream's capture follows now.
///
/// # Safety
/// `stream` must be capturing and `nodes` must be nodes of its capture graph.
unsafe fn follow_nodes(stream: &CudaStream, nodes: &[sys::CUgraphNode], after: &[usize], current: &mut Vec<usize>) -> Result<(), BackendError> {
    if current == after {
        return Ok(());
    }
    let mut handles: Vec<sys::CUgraphNode> = after.iter().map(|&n| nodes[n]).collect();
    // SAFETY: per the caller's contract; the driver copies the handles.
    unsafe { sys::cuStreamUpdateCaptureDependencies(stream.cu_stream(), handles.as_mut_ptr(), handles.len(), sys::CUstreamUpdateCaptureDependencies_flags::CU_STREAM_SET_CAPTURE_DEPENDENCIES as u32) }
        .result()
        .map_err(gpu)?;
    current.clear();
    current.extend_from_slice(after);
    Ok(())
}

/// The node the last launch captured on `stream` created.
///
/// # Safety
/// `stream` must be capturing.
unsafe fn last_node(stream: &CudaStream) -> Result<sys::CUgraphNode, BackendError> {
    let mut status = sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE;
    let mut graph = std::ptr::null_mut();
    let mut deps: *const sys::CUgraphNode = std::ptr::null();
    let mut count = 0usize;
    // SAFETY: the out-pointers are live; the dependency array is read before the next call.
    unsafe { sys::cuStreamGetCaptureInfo_v2(stream.cu_stream(), &mut status, std::ptr::null_mut(), &mut graph, &mut deps, &mut count) }.result().map_err(gpu)?;
    if status != sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_ACTIVE || count != 1 {
        return Err(BackendError::Gpu(format!("a captured launch left {count} nodes to follow ({status:?})")));
    }
    // SAFETY: `count == 1` entries at `deps`.
    Ok(unsafe { *deps })
}

impl Params {
    /// Fills the parameter array of `luma_main` for a launch; `uniform` gives the device address
    /// of the copy of a uniform passed by pointer.
    fn build(
        &mut self,
        native: &CudaKernel,
        kernel: &CompiledKernel,
        grid: [u32; 4],
        indirect: u64,
        args: &[RawArg<'_>],
        uniform: &mut dyn FnMut(&[u8]) -> Result<u64, BackendError>,
    ) -> Result<(), BackendError> {
        kernel.check_offsets(args, STORAGE_OFFSET_ALIGN)?;
        self.vals.clear();
        self.lens.clear();
        self.inline.clear();
        self.ptrs.clear();
        self.grid = grid;
        self.indirect = indirect;

        let mut bound = 0usize;
        for ((position, arg), abi) in kernel.bound(args).zip(&native.param_abis) {
            bound += 1;
            match (abi, arg) {
                (CudaArgAbi::DevicePtr, RawArg::Storage { buffer, .. }) => self.vals.push(device_ptr(buffer)? + buffer.offset()),
                (CudaArgAbi::InlineUniform { size }, RawArg::Uniform(bytes)) => {
                    // Each uniform starts on a 16-byte boundary, zero-padded to its size.
                    let at = self.inline.len();
                    self.inline.resize(at + size.div_ceil(16), Aligned16([0; 16]));
                    let n = bytes.len().min(*size);
                    // SAFETY: `n` bytes fit in the blocks just added.
                    unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), self.inline.as_mut_ptr().add(at).cast::<u8>(), n) };
                }
                (CudaArgAbi::UniformPtr, RawArg::Uniform(bytes)) => {
                    let address = uniform(bytes)?;
                    self.vals.push(address);
                }
                (CudaArgAbi::Texture, RawArg::Storage { buffer, .. }) => self.vals.push(texture::object(buffer)?),
                (CudaArgAbi::Unbound, RawArg::Storage { .. }) => {}
                _ => return Err(BackendError::KernelBinding(format!("argument {position} does not match the kind of its binding"))),
            }
        }
        if bound != native.param_abis.len() {
            return Err(BackendError::KernelArgCount { expected: native.param_abis.len(), actual: bound });
        }
        for (&binding, &stride) in native.array_lengths.iter().zip(&native.strides) {
            let length = match args.get(binding as usize) {
                Some(RawArg::Storage { bytes, .. }) => (*bytes / u64::from(stride.max(1))) as u32,
                _ => 0,
            };
            self.lens.push(length);
        }

        self.ptrs.push(self.grid.as_mut_ptr().cast());
        self.ptrs.push((&mut self.indirect as *mut u64).cast());
        let (mut value, mut at) = (0usize, 0usize);
        for abi in &native.param_abis {
            match abi {
                CudaArgAbi::InlineUniform { size } => {
                    // SAFETY: `at` stays within the `inline` blocks laid out above.
                    self.ptrs.push(unsafe { self.inline.as_mut_ptr().add(at) }.cast());
                    at += size.div_ceil(16);
                }
                CudaArgAbi::Unbound => {}
                _ => {
                    self.ptrs.push((&mut self.vals[value] as *mut u64).cast());
                    value += 1;
                }
            }
        }
        for length in &mut self.lens {
            self.ptrs.push((length as *mut u32).cast());
        }
        Ok(())
    }
}

/// Start and end of a profiled launch.
struct Sample {
    label: Arc<str>,
    events: Option<(CudaEvent, CudaEvent)>,
}

/// A chunk of device memory uniforms are copied into, and the pinned chunk they come from.
struct UniformChunk<T> {
    block: T,
    cap: usize,
    used: usize,
}

struct State {
    /// Launches recorded since the last flush.
    dispatches: u32,
    /// Whether work was put on the stream since the last flush.
    dirty: bool,
    held: Held,
    last_kernel: u64,
    submitted: VecDeque<(Arc<CudaEvent>, Held)>,
    /// Staging blocks of uploads with the event that follows their copy, oldest first: they
    /// return to the pool when that copy is done, not when the batch they were queued in is.
    staging: VecDeque<(CudaEvent, Pinned)>,
    free_events: Vec<CudaEvent>,
    /// Events created with timing, for the sampled launches (`free_events` are not).
    free_timing_events: Vec<CudaEvent>,
    /// The tag the next launch of that thread is timed under (`Backend::time_next_launch`).
    armed: Option<(std::thread::ThreadId, u64)>,
    /// Timed launches not yet reported, oldest first: tag, start and end.
    timed: VecDeque<(u64, CudaEvent, CudaEvent)>,
    pinned_free: Vec<Pinned>,
    pinned_idle: usize,
    pinned_busy: usize,
    uniform_device: Option<UniformChunk<CudaBuffer>>,
    uniform_host: Option<UniformChunk<Pinned>>,
    /// Device chunks a rollover gave up, kept until the launch in progress has been queued.
    uniform_retired: Vec<CudaBuffer>,
    params: Params,
    profiler: Option<Vec<Sample>>,
}

/// The device and what is known about it, shared by every backend of the process.
struct CudaShared {
    ctx: Arc<CudaContext>,
    info: DeviceInfo,
    disk: Arc<DiskCache>,
    cost: Arc<OnceLock<CostModel>>,
    /// L2 cache size, bytes, from the device attribute.
    l2_cache_bytes: Option<u64>,
    cc: (i32, i32),
    nvvm: crate::nvvm::Nvvm,
    /// A dedicated stream-ordered memory pool, when the device supports them.
    pool: Option<Arc<DevicePool>>,
    /// What the texture units take, `None` when the device reports no 2-D textures.
    textures: Option<TextureCaps>,
}

static POOL: Mutex<Option<Arc<CudaShared>>> = Mutex::new(None);

impl CudaShared {
    fn acquire() -> Result<Arc<CudaShared>, BackendError> {
        let mut pool = lock(&POOL);
        if let Some(shared) = pool.as_ref() {
            return Ok(Arc::clone(shared));
        }
        let shared = Arc::new(CudaShared::open()?);
        *pool = Some(Arc::clone(&shared));
        Ok(shared)
    }

    /// The kernel compiled from NVVM IR `source`, through the disk cache.
    fn build_function(&self, source: &str, label: &str) -> Result<(CudaFunction, Phases), BackendError> {
        let (major, minor) = self.cc;
        // Last byte: emitter revision; bump whenever the emitted IR or the NVVM options change.
        let meta = [major as u8, minor as u8, self.nvvm.version.0 as u8, self.nvvm.version.1 as u8, 8];
        // The NVVM IR text is part of the key, so an emitter change that alters it changes the key; the
        // revision of the sources covers what the text does not show (the NVVM options, the PTX
        // post-processing).
        let revision = cuda_code_revision();
        let hash = cache::fnv1a(source.as_bytes()) ^ cache::fnv1a(&meta).wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ revision.rotate_left(29);
        let stem = format!("ptx-{hash:016x}");
        let load = |text: String| self.ctx.load_module(Ptx::from_src(text)).and_then(|m| m.load_function(ENTRY));
        let dump = |ptx: &str| {
            if let Some(dir) = DUMP_DIR.as_deref() {
                // Best effort: a debugging aid must not fail a compile.
                let _ = std::fs::create_dir_all(dir);
                let _ = std::fs::write(dir.join(format!("{stem}.ll")), source);
                let _ = std::fs::write(dir.join(format!("{stem}.ptx")), ptx);
                let _ = std::fs::write(dir.join(format!("{stem}.label")), label);
            }
        };
        if let Some(text) = self.disk.load_blob(&stem, PTX_MAGIC).and_then(|bytes| String::from_utf8(bytes).ok())
            && !text.contains('\0')
        {
            dump(&text);
            let started = Instant::now();
            match load(text) {
                Ok(func) => return Ok((func, Phases { nvvm: None, load: started.elapsed(), ptx_bytes: 0 })),
                // A stale cache entry: rebuild it.
                Err(_) => self.disk.remove_blob(&stem),
            }
        }
        let started = Instant::now();
        let text = self.nvvm.compile(source, major, minor).map_err(|e| BackendError::ShaderCompile(format!("kernel `{label}`:\n{e}")))?;
        let nvvm = started.elapsed();
        dump(&text);
        let started = Instant::now();
        let func = load(text.clone()).map_err(gpu)?;
        let phases = Phases { nvvm: Some(nvvm), load: started.elapsed(), ptx_bytes: text.len() };
        self.disk.store_blob(&stem, PTX_MAGIC, text.as_bytes());
        Ok((func, phases))
    }

    fn open() -> Result<CudaShared, BackendError> {
        // SAFETY: only looks for the libraries; without this check the first driver call panics
        // when they are missing.
        let present = unsafe { sys::is_culib_present() };
        if !present {
            return Err(BackendError::NoAdapter("NVIDIA CUDA driver is not installed".into()));
        }
        let count = CudaContext::device_count().map_err(|e| BackendError::NoAdapter(format!("CUDA initialization failed: {e}")))?;
        if count == 0 {
            return Err(BackendError::NoAdapter("no CUDA device".into()));
        }
        let ctx = CudaContext::new(0).map_err(|e| BackendError::RequestDevice(e.to_string()))?;
        // One ordered stream per backend and no `CudaSlice`s: nothing needs cudarc's cross-stream
        // event tracking, and it must not interfere with graph capture.
        // SAFETY: luma never creates tracked slices; its streams are synchronized by events it
        // records itself.
        unsafe { ctx.disable_event_tracking() };

        let attribute = |a: sys::CUdevice_attribute| ctx.attribute(a).map_err(|e| BackendError::RequestDevice(e.to_string()));
        let major = attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)?;
        let minor = attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)?;
        let threads = attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)?;
        let block = [
            attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X)?,
            attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y)?,
            attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z)?,
        ];
        let multiprocessors = attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)?;
        let l2_bytes = attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE)?;
        let shared_bytes = attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK)?;
        let name = ctx.name().map_err(|e| BackendError::RequestDevice(e.to_string()))?;
        ctx.bind_to_thread().map_err(|e| BackendError::RequestDevice(e.to_string()))?;
        let total = result::mem_get_info().map_err(|e| BackendError::RequestDevice(e.to_string()))?.1 as u64;

        let mut driver = 0;
        // SAFETY: writes one int.
        unsafe { sys::cuDriverGetVersion(&mut driver) }.result().map_err(|e| BackendError::RequestDevice(e.to_string()))?;
        let nvvm = crate::nvvm::Nvvm::open(major, minor).map_err(|e| BackendError::NoAdapter(format!("the NVIDIA driver's NVVM compiler is unusable: {e}")))?;

        let identity = format!("{name}\n{major}.{minor}\n{driver}");
        let key = format!("cuda-sm{major}{minor}-{:016x}", cache::fnv1a(identity.as_bytes()));
        let disk = DiskCache::open(cache::directory(), key);

        let info = DeviceInfo {
            backend: BackendKind::Cuda,
            name,
            cpu: false,
            zero_copy: false,
            profile: TuningProfile::Nvidia,
            features: Features::TIMESTAMP_QUERY | Features::SUBGROUP | Features::SHADER_F16 | Features::SHADER_F64,
            subgroup_size: Some((32, 32)),
            memory_bytes: Some(total),
            ieee_f32_intrinsics: true,
            hardware_fma: Some(true),
            limits: DeviceLimits {
                max_buffer_bytes: total.max(1 << 30) & !3,
                max_storage_buffers: 256,
                max_uniform_bytes: 65536,
                uniform_align: 16,
                max_dynamic_uniforms: u32::MAX,
                max_workgroups_per_dim: 65535,
                max_workgroup_size: block.map(|d| d as u32),
                max_workgroup_invocations: threads as u32,
                max_workgroup_storage_bytes: shared_bytes as u32,
            },
            compute_units: Some(multiprocessors as u32).filter(|&n| n > 0),
        };
        let pool = if attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED)? != 0 { DevicePool::create(&ctx) } else { None };
        let textures = texture::caps(&attribute);
        Ok(CudaShared { ctx, info, disk, cost: Arc::new(OnceLock::new()), l2_cache_bytes: u64::try_from(l2_bytes).ok().filter(|&n| n > 0), cc: (major, minor), nvvm, pool, textures })
    }
}

/// The CUDA backend.
pub struct CudaBackend {
    shared: Arc<CudaShared>,
    stream: Arc<CudaStream>,
    /// The transfer lane: [`Backend::transfer_in`] / [`Backend::transfer_out`] copy on it while
    /// kernels run on `stream`. Ordered against `stream` only through events.
    transfer: Arc<CudaStream>,
    /// The stream graphs are captured on, created on first use.
    capture: Mutex<Option<Arc<CudaStream>>>,
    pending: AtomicU32,
    /// Shared with the stages, which give their pinned block back to it when dropped.
    state: Arc<Mutex<State>>,
}

impl fmt::Debug for CudaBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CudaBackend").field("device", &self.shared.info.name).finish_non_exhaustive()
    }
}

impl Drop for CudaBackend {
    fn drop(&mut self) {
        // Everything the state holds is released only after both lanes have run dry.
        let _ = self.transfer.synchronize();
        let _ = self.stream.synchronize();
    }
}

impl CudaBackend {
    /// Opens the first CUDA device.
    pub fn new() -> Result<CudaBackend, BackendError> {
        let shared = CudaShared::acquire()?;
        let stream = shared.ctx.new_stream().map_err(|e| BackendError::RequestDevice(e.to_string()))?;
        let transfer = shared.ctx.new_stream().map_err(|e| BackendError::RequestDevice(e.to_string()))?;
        let state = State {
            dispatches: 0,
            dirty: false,
            held: Held::default(),
            last_kernel: 0,
            submitted: VecDeque::new(),
            staging: VecDeque::new(),
            free_events: Vec::new(),
            free_timing_events: Vec::new(),
            armed: None,
            timed: VecDeque::new(),
            pinned_free: Vec::new(),
            pinned_idle: 0,
            pinned_busy: 0,
            uniform_device: None,
            uniform_host: None,
            uniform_retired: Vec::new(),
            params: Params::default(),
            profiler: None,
        };
        Ok(CudaBackend { shared, stream, transfer, capture: Mutex::new(None), pending: AtomicU32::new(0), state: Arc::new(Mutex::new(state)) })
    }

    fn bind(&self) -> Result<(), BackendError> {
        self.shared.ctx.bind_to_thread().map_err(gpu)
    }

    /// A device block of at least `bytes` bytes. With a pool it is allocated in the stream's
    /// order (no implicit device synchronization on either allocation or free), so it must only
    /// be touched by work recorded on that stream.
    fn alloc_block(&self, bytes: u64) -> Result<CudaBuffer, BackendError> {
        let Some(pool) = self.shared.pool.as_ref() else { return self.alloc_block_sync(bytes) };
        self.bind()?;
        let mut dptr = 0;
        // SAFETY: the pool and stream are live and belong to the current context; the block
        // frees itself on the same stream.
        let allocated = unsafe { sys::cuMemAllocFromPoolAsync(&mut dptr, bytes.max(MIN_BUFFER) as usize, pool.pool, self.stream.cu_stream()) }.result();
        match allocated {
            Ok(()) => Ok(CudaBuffer { dptr, ctx: Arc::clone(&self.shared.ctx), ordered: Some(Ordered { stream: Arc::clone(&self.stream), _pool: Arc::clone(pool) }) }),
            Err(e) if is_out_of_memory(&e) => Err(BackendError::OutOfMemory { requested: bytes }),
            Err(e) => Err(gpu(e)),
        }
    }

    /// A device block from `cuMemAlloc`: usable outside the stream's order (host-synchronous
    /// copies, graph capture) and freed synchronously.
    fn alloc_block_sync(&self, bytes: u64) -> Result<CudaBuffer, BackendError> {
        self.bind()?;
        // SAFETY: plain allocation; the block frees itself.
        match unsafe { result::malloc_sync(bytes.max(MIN_BUFFER) as usize) } {
            Ok(dptr) => Ok(CudaBuffer { dptr, ctx: Arc::clone(&self.shared.ctx), ordered: None }),
            Err(e) if is_out_of_memory(&e) => Err(BackendError::OutOfMemory { requested: bytes }),
            Err(e) => Err(gpu(e)),
        }
    }

    /// The device address of a buffer of a transfer, which must not be a stream-ordered pool
    /// block: its free is queued on the compute stream and could run under the transfer lane.
    fn unordered_ptr(&self, buffer: &Buffer) -> Result<sys::CUdeviceptr, BackendError> {
        let block = buffer.downcast_ref::<CudaBuffer>().ok_or_else(|| BackendError::KernelBinding("a buffer of another backend was passed to a CUDA transfer".into()))?;
        if block.ordered.is_some() {
            return Err(BackendError::Unsupported("the buffer of a transfer must come from create_buffer_unordered".into()));
        }
        Ok(block.dptr)
    }

    /// Makes the transfer lane wait for the events of `after`.
    fn transfer_waits(&self, after: &[Ticket]) -> Result<(), BackendError> {
        for ticket in after {
            if let Some(CudaTicket(event)) = ticket.downcast_ref() {
                self.transfer.wait(event).map_err(gpu)?;
            } else if !ticket.is_done_marker() {
                return Err(BackendError::Unsupported("a ticket of another backend".into()));
            }
        }
        Ok(())
    }

    /// The ticket of the copy just queued on the transfer lane; the stage remembers it.
    fn transfer_ticket(&self, stage: &CudaStage) -> Result<Ticket, BackendError> {
        let event = Arc::new(self.shared.ctx.new_event(None).map_err(gpu)?);
        event.record(&self.transfer).map_err(gpu)?;
        *lock(&stage.last) = Some(Arc::clone(&event));
        Ok(Ticket::new(CudaTicket(event)))
    }

    /// Records an event after the work so far, taking the batch's resources with it. `None` when
    /// nothing was recorded since the last flush.
    fn flush_locked(&self, st: &mut State) -> Result<Option<Arc<CudaEvent>>, BackendError> {
        st.dispatches = 0;
        // A tag not consumed by a launch of this batch is dropped, never carried to the next.
        st.armed = None;
        self.pending.store(0, Ordering::Relaxed);
        if !st.dirty {
            return Ok(None);
        }
        self.bind()?;
        let event = self.event(st)?;
        event.record(&self.stream).map_err(gpu)?;
        let event = Arc::new(event);
        st.dirty = false;
        st.last_kernel = 0;
        let held = std::mem::take(&mut st.held);
        st.submitted.push_back((Arc::clone(&event), held));
        self.reap(st)?;
        Ok(Some(event))
    }

    /// An event, recycled when there is a spare one.
    fn event(&self, st: &mut State) -> Result<CudaEvent, BackendError> {
        match st.free_events.pop() {
            Some(event) => Ok(event),
            None => self.shared.ctx.new_event(None).map_err(gpu),
        }
    }

    /// An event that records time, recycled when there is a spare one.
    fn timing_event(&self, st: &mut State) -> Result<CudaEvent, BackendError> {
        match st.free_timing_events.pop() {
            Some(event) => Ok(event),
            None => self.shared.ctx.new_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT)).map_err(gpu),
        }
    }

    /// Releases the resources of every batch that has completed, and the staging blocks whose
    /// copy has.
    fn reap(&self, st: &mut State) -> Result<(), BackendError> {
        self.bind()?;
        while let Some((event, _)) = st.submitted.front() {
            if !event.try_is_complete().map_err(gpu)? {
                break;
            }
            let Some((event, held)) = st.submitted.pop_front() else { break };
            Self::release(st, held);
            if let Ok(event) = Arc::try_unwrap(event)
                && st.free_events.len() < 64
            {
                st.free_events.push(event);
            }
        }
        self.reap_staging(st)
    }

    /// Gives the staging blocks of uploads back to the pool once the stream has copied out of
    /// them (`State::staging`).
    fn reap_staging(&self, st: &mut State) -> Result<(), BackendError> {
        while let Some((event, _)) = st.staging.front() {
            if !event.try_is_complete().map_err(gpu)? {
                break;
            }
            let Some((event, pinned)) = st.staging.pop_front() else { break };
            st.pinned_busy -= pinned.cap;
            keep_pinned(st, pinned);
            if st.free_events.len() < 64 {
                st.free_events.push(event);
            }
        }
        Ok(())
    }

    fn release(st: &mut State, held: Held) {
        for pinned in held.pinned {
            st.pinned_busy -= pinned.cap;
            keep_pinned(st, pinned);
        }
    }

    /// Submits and waits for everything, then frees what completed.
    fn sync_locked(&self, st: &mut State) -> Result<(), BackendError> {
        self.flush_locked(st)?;
        if let Some((event, _)) = st.submitted.back() {
            event.synchronize().map_err(poll)?;
        }
        self.reap(st)
    }

    /// Waits for `event`, spinning for `budget` first. Returns how long it took.
    fn wait_event(&self, event: &CudaEvent, budget: Duration) -> Result<Duration, BackendError> {
        self.bind()?;
        let start = Instant::now();
        loop {
            if event.try_is_complete().map_err(poll)? {
                return Ok(start.elapsed());
            }
            if start.elapsed() >= budget {
                break;
            }
            std::hint::spin_loop();
        }
        event.synchronize().map_err(poll)?;
        Ok(start.elapsed())
    }

    /// Pinned memory of at least `bytes` bytes: an idle block when there is a fitting one, one
    /// that finished copying when the stream is ahead of the host, otherwise a new allocation
    /// (`cuMemHostAlloc` pins every page: 0.4 ms per MiB, and freeing costs a third of that),
    /// after waiting for earlier work when too much staging memory is in flight.
    fn take_pinned(&self, st: &mut State, bytes: usize) -> Result<Pinned, BackendError> {
        let class = size_class(bytes.max(PINNED_MIN) as u64, u64::MAX).map_or(bytes, |class| class as usize);
        // The smallest idle block that is large enough, and not much larger.
        let find = |st: &mut State| {
            let at = st.pinned_free.iter().enumerate().filter(|(_, p)| p.cap >= class && p.cap / class <= PINNED_FIT).min_by_key(|(_, p)| p.cap).map(|(at, _)| at)?;
            let pinned = st.pinned_free.swap_remove(at);
            st.pinned_idle -= pinned.cap;
            Some(pinned)
        };
        if let Some(pinned) = find(st) {
            return Ok(pinned);
        }
        if !st.submitted.is_empty() || !st.staging.is_empty() {
            self.reap(st)?;
            if let Some(pinned) = find(st) {
                return Ok(pinned);
            }
        }
        while st.pinned_busy + class > PINNED_BUSY_CAP && st.pinned_busy > 0 {
            if let Some((event, _)) = st.staging.front() {
                // A copy is in flight out of one of the blocks: its end frees it.
                event.synchronize().map_err(poll)?;
                self.reap(st)?;
                continue;
            }
            if st.submitted.is_empty() {
                self.flush_locked(st)?;
            }
            let Some((event, _)) = st.submitted.front() else { break };
            let event = Arc::clone(event);
            event.synchronize().map_err(poll)?;
            self.reap(st)?;
        }
        if let Some(pinned) = find(st) {
            return Ok(pinned);
        }
        self.bind()?;
        // SAFETY: plain allocation; `Pinned` frees it.
        let ptr = match unsafe { result::malloc_host(class, 0) } {
            Ok(ptr) => ptr,
            Err(_) => {
                st.pinned_free.clear();
                st.pinned_idle = 0;
                // SAFETY: as above.
                unsafe { result::malloc_host(class, 0) }.map_err(|e| if is_out_of_memory(&e) { BackendError::OutOfMemory { requested: class as u64 } } else { gpu(e) })?
            }
        };
        Ok(Pinned { ptr: ptr.cast(), cap: class, ctx: Arc::clone(&self.shared.ctx) })
    }

    /// Copies `bytes` into a fresh slot of the uniform chunks and returns the slot's device
    /// address; the copy is queued on the stream.
    fn uniform_slot(&self, st: &mut State, bytes: &[u8]) -> Result<u64, BackendError> {
        let slot = bytes.len().max(1).next_multiple_of(UNIFORM_ALIGN);
        if st.uniform_device.as_ref().is_none_or(|c| c.used + slot > c.cap) {
            if let Some(old) = st.uniform_device.take() {
                // Not released with the batch: the launch being built may already name it, and
                // `take_pinned` below can flush and reap.
                st.uniform_retired.push(old.block);
            }
            let cap = UNIFORM_CHUNK.max(slot);
            st.uniform_device = Some(UniformChunk { block: self.alloc_block(cap as u64)?, cap, used: 0 });
        }
        if st.uniform_host.as_ref().is_none_or(|c| c.used + slot > c.cap) {
            if let Some(old) = st.uniform_host.take() {
                st.pinned_busy += old.block.cap;
                st.held.pinned.push(old.block);
            }
            let block = self.take_pinned(st, UNIFORM_CHUNK.max(slot))?;
            let cap = block.cap;
            st.uniform_host = Some(UniformChunk { block, cap, used: 0 });
        }
        let (Some(device), Some(host)) = (st.uniform_device.as_mut(), st.uniform_host.as_mut()) else {
            return Err(BackendError::Gpu("uniform chunks missing".into()));
        };
        // SAFETY: the slot lies inside the pinned chunk and is written once; the copy reads it
        // only after this write, and the chunk outlives the queued copy (it is released after
        // the batch that rolled it over, which completes after every copy out of it).
        let staged = unsafe {
            let at = host.block.ptr.add(host.used);
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), at, bytes.len());
            std::slice::from_raw_parts(at, bytes.len())
        };
        let address = device.block.dptr + device.used as u64;
        // SAFETY: `address` is a slot of a live device chunk, `staged` is pinned memory that
        // stays untouched until the copy completes.
        unsafe { result::memcpy_htod_async(address, staged, self.stream.cu_stream()) }.map_err(gpu)?;
        device.used += slot;
        host.used += slot;
        st.dirty = true;
        Ok(address)
    }

    /// Frees device memory by waiting for the GPU and giving back idle staging memory.
    fn relieve(&self, st: &mut State) -> Result<(), BackendError> {
        self.sync_locked(st)?;
        st.pinned_free.clear();
        st.pinned_idle = 0;
        self.trim_pool()
    }

    /// Gives the memory the stream-ordered pool holds back to the driver. The stream is
    /// synchronized first so that the frees still queued on it have taken effect.
    fn trim_pool(&self) -> Result<(), BackendError> {
        let Some(pool) = self.shared.pool.as_ref() else { return Ok(()) };
        self.stream.synchronize().map_err(poll)?;
        self.bind()?;
        // SAFETY: `pool` is the live pool of this device; no allocation of it is in flight.
        unsafe { sys::cuMemPoolTrimTo(pool.pool, 0) }.result().map_err(gpu)
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_locked(
        &self,
        st: &mut State,
        native: &CudaKernel,
        kernel: &CompiledKernel,
        grid: [u32; 3],
        size: [u32; 3],
        indirect: u64,
        indirect_buffer: Option<&Buffer>,
        args: &[RawArg<'_>],
        params: &mut Params,
    ) -> Result<(), BackendError> {
        params.build(native, kernel, [size[0], size[1], size[2], 0], indirect, args, &mut |bytes| self.uniform_slot(st, bytes))?;
        // Everything the launch names is held before it is queued, and from here on nothing
        // flushes or reaps until it is.
        if st.last_kernel != kernel.raw.id() {
            st.last_kernel = kernel.raw.id();
            st.held.kernels.push(kernel.raw.clone());
        }
        for (_, arg) in kernel.bound(args) {
            if let RawArg::Storage { buffer, .. } = arg
                && st.held.buffers.last() != Some(*buffer)
            {
                st.held.buffers.push((*buffer).clone());
            }
        }
        if let Some(buffer) = indirect_buffer {
            st.held.buffers.push(buffer.clone());
        }
        // Both events exist before the launch, so that the end is recorded right after it: creating
        // it in between would add host time to a short kernel's measurement.
        let timing = if st.profiler.is_some() {
            let start = self.shared.ctx.new_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT)).map_err(gpu)?;
            let end = self.shared.ctx.new_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT)).map_err(gpu)?;
            start.record(&self.stream).map_err(gpu)?;
            st.dirty = true;
            Some((start, end))
        } else {
            None
        };
        // A sampled launch is bracketed by its own pair, outside the profiler's. The tag is taken
        // here whatever happens next, so that it never outlives the launch it was armed for.
        let sampled = match st.armed {
            Some((thread, tag)) if thread == std::thread::current().id() => {
                st.armed = None;
                let start = self.timing_event(st)?;
                let end = self.timing_event(st)?;
                start.record(&self.stream).map_err(gpu)?;
                st.dirty = true;
                Some((tag, start, end))
            }
            _ => None,
        };
        // SAFETY: the function belongs to a module kept alive by the kernel; `params.ptrs` point
        // to values that live until the call returns (the driver copies them), and every buffer
        // named stays alive in the batch until the work completes.
        let enqueued = unsafe {
            result::launch_kernel(
                native.func.cu_function(),
                (grid[0], grid[1], grid[2]),
                (native.workgroup[0], native.workgroup[1], native.workgroup[2]),
                native.shared_bytes,
                self.stream.cu_stream(),
                &mut params.ptrs[..],
            )
        };
        if enqueued.is_ok() {
            st.dirty = true;
            st.dispatches += 1;
        }
        enqueued.map_err(gpu)?;
        if let Some((tag, start, end)) = sampled {
            end.record(&self.stream).map_err(gpu)?;
            if st.timed.len() >= MAX_TIMED {
                // Nobody is draining: the oldest sample goes, its events are reused.
                if let Some((_, start, end)) = st.timed.pop_front() {
                    st.free_timing_events.push(start);
                    st.free_timing_events.push(end);
                }
            }
            st.timed.push_back((tag, start, end));
        }
        if let Some((start, end)) = timing {
            end.record(&self.stream).map_err(gpu)?;
            if let Some(samples) = st.profiler.as_mut() {
                samples.push(Sample { label: Arc::clone(&kernel.label), events: Some((start, end)) });
            }
        }
        Ok(())
    }

    /// Captures `launches` as one graph: with `edges`, the launches that touch no common buffer
    /// (or only read it) have no edge between them and may run concurrently, otherwise each one
    /// follows the one before it.
    fn try_capture(&self, launches: &[GraphLaunch<'_>], edges: bool) -> Result<Arc<dyn ExecGraph>, BackendError> {
        {
            let mut st = lock(&self.state);
            // The launches go into the graph, not through `record_launch`: a tag armed for one is dropped.
            st.armed = None;
            if st.profiler.is_some() {
                return Err(BackendError::Unsupported("graphs are not captured while profiling".into()));
            }
        }
        let natives: Vec<&CudaKernel> = launches
            .iter()
            .map(|l| l.kernel.raw.resolve::<CudaKernel>())
            .collect::<Result<_, _>>()?;
        let dispatches: Vec<([u32; 3], u64, Option<&Buffer>)> = launches.iter().map(|l| resolve_dispatch(l.dispatch)).collect::<Result<_, _>>()?;
        let order = if edges { capture_order(launches, &dispatches)? } else { None };

        // The uniforms passed by pointer are fixed for the life of the graph: one device copy.
        let mut staging: Vec<u8> = Vec::new();
        for ((launch, native), (grid, ..)) in launches.iter().zip(&natives).zip(&dispatches) {
            for ((_, arg), abi) in launch.kernel.bound(launch.args).zip(&native.param_abis) {
                if let (CudaArgAbi::UniformPtr, RawArg::Uniform(bytes)) = (abi, arg)
                    && !grid.contains(&0)
                {
                    staging.extend_from_slice(bytes);
                    staging.resize(staging.len().next_multiple_of(UNIFORM_ALIGN), 0);
                }
            }
        }
        let scratch = if staging.is_empty() {
            None
        } else {
            // Synchronous allocation: the copy below is not ordered on the stream.
            let block = self.alloc_block_sync(staging.len() as u64)?;
            // SAFETY: synchronous copy into the block just allocated.
            unsafe { result::memcpy_htod_sync(block.dptr, &staging[..]) }.map_err(gpu)?;
            Some(block)
        };
        let base = scratch.as_ref().map_or(0, |b| b.dptr);

        let mut capture = lock(&self.capture);
        self.bind()?;
        let stream = match &*capture {
            Some(stream) => Arc::clone(stream),
            None => {
                let stream = self.shared.ctx.new_stream().map_err(gpu)?;
                *capture = Some(Arc::clone(&stream));
                stream
            }
        };
        stream.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL).map_err(gpu)?;

        let mut params = Params::default();
        let mut offset = 0usize;
        let scratch_bytes = if scratch.is_some() { (staging.len() as u64).max(MIN_BUFFER) } else { 0 };
        let mut res = GraphRes { buffers: Vec::new(), kernels: Vec::new(), scratch, scratch_bytes };
        // Without an order every launch follows the one before it (the stream's order). With one,
        // `nodes` are the launches captured and `stream_after` what the stream's capture follows now.
        let (mut nodes, mut stream_after): (Vec<sys::CUgraphNode>, Vec<usize>) = (Vec::new(), Vec::new());
        let recorded = (|| {
            for ((launch, native), &(grid, indirect, indirect_buffer)) in launches.iter().zip(&natives).zip(&dispatches) {
                // A launch of no workgroups (an indirect one: of a bound of none) never runs.
                if grid.contains(&0) {
                    continue;
                }
                params.build(native, launch.kernel, [launch.size[0], launch.size[1], launch.size[2], 0], indirect, launch.args, &mut |bytes| {
                    let address = base + offset as u64;
                    offset += bytes.len().next_multiple_of(UNIFORM_ALIGN);
                    Ok(address)
                })?;
                if let Some(order) = &order {
                    // SAFETY: the stream is capturing and `nodes` are its nodes.
                    unsafe { follow_nodes(&stream, &nodes, &order[nodes.len()], &mut stream_after) }?;
                }
                // SAFETY: as in `launch_locked`; the captured node copies the parameters.
                unsafe {
                    result::launch_kernel(
                        native.func.cu_function(),
                        (grid[0], grid[1], grid[2]),
                        (native.workgroup[0], native.workgroup[1], native.workgroup[2]),
                        native.shared_bytes,
                        stream.cu_stream(),
                        &mut params.ptrs[..],
                    )
                }
                .map_err(gpu)?;
                for (_, arg) in launch.kernel.bound(launch.args) {
                    if let RawArg::Storage { buffer, .. } = arg {
                        res.buffers.push((*buffer).clone());
                    }
                }
                // The graph reads the counts when it runs: the buffer must outlive it.
                res.buffers.extend(indirect_buffer.cloned());
                res.kernels.push(launch.kernel.raw.clone());
                if order.is_some() {
                    // SAFETY: the launch just captured made a node of the capture on `stream`.
                    nodes.push(unsafe { last_node(&stream) }?);
                    stream_after.clear();
                    stream_after.push(nodes.len() - 1);
                }
            }
            Ok::<(), BackendError>(())
        })();
        let ended = stream.end_capture(sys::CUgraphInstantiate_flags(0));
        recorded?;
        let graph = ended.map_err(gpu)?.ok_or_else(|| BackendError::Gpu("the capture recorded nothing".into()))?;
        Ok(Arc::new(CudaExecGraph { exec: Mutex::new(GraphCell(graph)), res: Arc::new(res), ctx: Arc::clone(&self.shared.ctx) }))
    }
}

impl Backend for CudaBackend {
    fn memory_now(&self) -> Option<MemoryNow> {
        let (free, total) = result::mem_get_info().ok()?;
        Some(MemoryNow { free: free as u64, total: total as u64 })
    }

    fn info(&self) -> &DeviceInfo {
        &self.shared.info
    }

    fn l2_cache_bytes(&self) -> Option<u64> {
        self.shared.l2_cache_bytes
    }

    fn storage_offset_align(&self) -> u64 {
        STORAGE_OFFSET_ALIGN
    }

    fn disk_cache(&self) -> &Arc<DiskCache> {
        &self.shared.disk
    }

    fn device_cost(&self) -> &Arc<OnceLock<CostModel>> {
        &self.shared.cost
    }

    fn create_buffer(&self, size: u64, usage: BufferUsage, _mapped: bool) -> Option<Buffer> {
        let mut st = lock(&self.state);
        let block = match self.alloc_block(size) {
            Ok(block) => block,
            Err(BackendError::OutOfMemory { .. }) => {
                // Work still in flight pins freed buffers: wait for it and try once more.
                self.relieve(&mut st).ok()?;
                self.alloc_block(size).ok()?
            }
            Err(_) => return None,
        };
        // Not cleared: neither the pool (which hands back used buffers) nor any kernel reads a
        // buffer before it has been written.
        Some(Buffer::new(size, usage, block))
    }

    fn create_buffer_unordered(&self, size: u64, usage: BufferUsage) -> Option<Buffer> {
        let block = match self.alloc_block_sync(size) {
            Ok(block) => block,
            Err(BackendError::OutOfMemory { .. }) => {
                self.relieve(&mut lock(&self.state)).ok()?;
                self.alloc_block_sync(size).ok()?
            }
            Err(_) => return None,
        };
        Some(Buffer::new(size, usage, block))
    }

    fn alloc_filled(&self, data: &[u8], max_buffer_bytes: u64) -> Result<Buffer, BackendError> {
        let bytes = data.len() as u64;
        if bytes > max_buffer_bytes {
            return Err(BackendError::OutOfMemory { requested: bytes });
        }
        let usage = BufferUsage::STORAGE | BufferUsage::COPY_SRC | BufferUsage::COPY_DST;
        let buffer = self.create_buffer(bytes, usage, false).ok_or(BackendError::OutOfMemory { requested: bytes })?;
        self.record_upload(&buffer, data, max_buffer_bytes)?;
        Ok(buffer)
    }

    fn texture_caps(&self) -> Option<TextureCaps> {
        self.shared.textures
    }

    fn create_texture(&self, source: &Buffer, offset: u64, pitch: u64, desc: &TextureDesc) -> Result<Buffer, BackendError> {
        self.make_texture(source, offset, pitch, desc)
    }

    fn recycle_mappable(&self, buffer: Buffer) {
        // Nothing is mappable here; the work still using it keeps it alive.
        drop(buffer);
    }

    fn compile_kernel(&self, module: &KernelModule, used: Option<&[bool]>, label: &str, trust: KernelTrust) -> Result<RawKernel, BackendError> {
        check_mask(module, used)?;
        compile_native(&self.shared, module, used, label, trust).map(RawKernel::new)
    }

    fn gemm_tiles(&self, exact: bool) -> &[(u32, u32)] {
        match (self.shared.cc.0 >= 8, exact) {
            (false, _) => &[],
            (true, false) => &TENSOR_TILES,
            (true, true) => &crate::sgemm::TILES,
        }
    }

    fn gemm_ragged(&self) -> bool {
        self.shared.cc.0 >= 8
    }

    /// Both the `f32` kernels and the tensor-core ones have a half-work variant (`GemmSpec::half`).
    fn gemm_half(&self) -> bool {
        self.shared.cc.0 >= 8
    }

    fn gemm_hooks(&self) -> bool {
        self.shared.cc.0 >= 8
    }

    fn compile_gemm(&self, spec: &GemmSpec, hook: Option<&Arc<KernelModule>>) -> Option<RawKernel> {
        // A hooked product needs its module, and only the tensor-core kernels take one.
        if spec.hook.is_some() && (hook.is_none() || spec.exact || spec.half || !self.gemm_hooks()) {
            return None;
        }
        // Loaded on a worker thread (which binds the context in `compile_gemm_ptx`): a failure
        // arrives at the first resolve.
        let shared = Arc::clone(&self.shared);
        let spec = *spec;
        let hook = hook.cloned();
        Some(RawKernel::spawn(move || {
            compile_gemm_ptx(&shared, &spec, hook.as_deref()).inspect_err(|error| log::warn!("luma-cuda: the matrix product {spec:?} did not compile: {error}"))
        }))
    }

    fn compile_kernel_deferred(&self, module: KernelModule, used: Option<Box<[bool]>>, label: &str, trust: KernelTrust) -> Result<RawKernel, BackendError> {
        check_mask(&module, used.as_deref())?;
        let shared = Arc::clone(&self.shared);
        let label = label.to_owned();
        Ok(RawKernel::spawn(move || compile_native(&shared, &module, used.as_deref(), &label, trust)))
    }

    fn record_launch(&self, kernel: &CompiledKernel, dispatch: Dispatch<'_>, size: [u32; 3], args: &[RawArg<'_>]) -> Result<u32, BackendError> {
        kernel.check_count(args.len())?;
        kernel.check_kinds(args)?;
        let native = kernel.raw.resolve::<CudaKernel>()?;
        let (grid, indirect, indirect_buffer) = resolve_dispatch(dispatch)?;
        let mut st = lock(&self.state);
        if grid.contains(&0) {
            // Nothing runs, so nothing is timed.
            st.armed = None;
            if let Some(samples) = st.profiler.as_mut() {
                samples.push(Sample { label: Arc::clone(&kernel.label), events: None });
            }
            return Ok(st.dispatches);
        }
        self.bind()?;
        let mut params = std::mem::take(&mut st.params);
        let launched = self.launch_locked(&mut st, native, kernel, grid, size, indirect, indirect_buffer, args, &mut params);
        st.params = params;
        // The chunks a rollover gave up lived until the launch that named them was queued.
        let retired = std::mem::take(&mut st.uniform_retired);
        st.held.scratch.extend(retired);
        self.pending.store(st.dispatches, Ordering::Relaxed);
        launched?;
        Ok(st.dispatches)
    }

    fn record_clear(&self, buffer: &Buffer, bytes: u64) -> Result<(), BackendError> {
        let dptr = device_ptr(buffer)?;
        if bytes == 0 {
            return Ok(());
        }
        let mut st = lock(&self.state);
        self.bind()?;
        st.held.buffers.push(buffer.clone());
        // SAFETY: the caller clears the first `bytes` of a buffer at least that large.
        unsafe { result::memset_d8_async(dptr, 0, bytes as usize, self.stream.cu_stream()) }.map_err(gpu)?;
        st.dirty = true;
        Ok(())
    }

    fn record_copy(&self, src: &Buffer, src_offset: u64, dst: &Buffer, dst_offset: u64, bytes: u64) -> Result<(), BackendError> {
        let (from, to) = (device_ptr(src)?, device_ptr(dst)?);
        if bytes == 0 {
            return Ok(());
        }
        let mut st = lock(&self.state);
        self.bind()?;
        st.held.buffers.push(src.clone());
        st.held.buffers.push(dst.clone());
        // SAFETY: the caller keeps both ranges inside their buffers.
        unsafe { result::memcpy_dtod_async(to + dst_offset, from + src_offset, bytes as usize, self.stream.cu_stream()) }.map_err(gpu)?;
        st.dirty = true;
        Ok(())
    }

    fn record_upload(&self, dst: &Buffer, data: &[u8], max_buffer_bytes: u64) -> Result<(), BackendError> {
        let dptr = device_ptr(dst)?;
        let bytes = data.len() as u64;
        if bytes > dst.size().min(max_buffer_bytes) {
            return Err(BackendError::OutOfMemory { requested: bytes });
        }
        let mut st = lock(&self.state);
        self.bind()?;
        let chunk = UPLOAD_CHUNK;
        for (index, piece) in data.chunks(chunk).enumerate() {
            let pinned = self.take_pinned(&mut st, piece.len())?;
            // SAFETY: `piece.len() <= pinned.cap`; the pinned memory is kept until the copy out
            // of it has completed (the staging queue or the batch holds it).
            let staged = unsafe {
                copy_parallel(piece, std::slice::from_raw_parts_mut(pinned.ptr.cast::<MaybeUninit<u8>>(), piece.len()));
                std::slice::from_raw_parts(pinned.ptr, piece.len())
            };
            let to = dptr + (index * chunk) as u64;
            // Held after `take_pinned` (which can flush and reap) and before the copy is queued.
            if st.held.buffers.last() != Some(dst) {
                st.held.buffers.push(dst.clone());
            }
            // SAFETY: the destination range lies inside `dst`.
            let enqueued = unsafe { result::memcpy_htod_async(to, staged, self.stream.cu_stream()) };
            // The staging memory stays alive whether or not the call reports an error. A big
            // block is free again once the stream has copied out of it: the next piece (or the
            // next upload) reuses it instead of pinning more memory. A small one goes with the batch.
            let own_event = if enqueued.is_ok() && pinned.cap >= MARK_MIN { self.event(&mut st).ok().filter(|event| event.record(&self.stream).is_ok()) } else { None };
            st.pinned_busy += pinned.cap;
            st.dirty = true;
            match own_event {
                Some(event) => st.staging.push_back((event, pinned)),
                None => st.held.pinned.push(pinned),
            }
            enqueued.map_err(gpu)?;
        }
        Ok(())
    }

    fn record_graph(&self, graph: &dyn ExecGraph, dispatches: u32) -> Result<u32, BackendError> {
        let graph = graph.downcast_ref::<CudaExecGraph>().ok_or_else(|| BackendError::Unsupported("graph was captured by another backend".into()))?;
        let mut st = lock(&self.state);
        st.armed = None;
        if st.profiler.is_some() {
            return Err(BackendError::Unsupported("cannot replay a graph while profiling".into()));
        }
        self.bind()?;
        st.held.graphs.push(Arc::clone(&graph.res));
        {
            let cell = lock(&graph.exec);
            // SAFETY: the graph is instantiated and alive; the launch is serialized by the lock.
            unsafe { result::graph::launch(cell.0.cu_graph_exec(), self.stream.cu_stream()) }.map_err(gpu)?;
        }
        st.dirty = true;
        st.dispatches += dispatches;
        self.pending.store(st.dispatches, Ordering::Relaxed);
        Ok(st.dispatches)
    }

    fn pending_dispatches(&self) -> u32 {
        self.pending.load(Ordering::Relaxed)
    }

    fn gpu_busy(&self) -> bool {
        let mut st = lock(&self.state);
        self.reap(&mut st).is_ok() && !st.submitted.is_empty()
    }

    fn notify_released(&self, _buffers: &[Buffer]) {
        // Buffers the pool let go of stay alive in the batches that use them.
    }

    fn flush_batch(&self) -> Result<(), BackendError> {
        self.flush_locked(&mut lock(&self.state)).map(|_| ())
    }

    fn sync(&self, spin_budget: Duration) -> Result<Duration, BackendError> {
        let event = {
            let mut st = lock(&self.state);
            match self.flush_locked(&mut st)? {
                Some(event) => Some(event),
                None => st.submitted.back().map(|(event, _)| Arc::clone(event)),
            }
        };
        let waited = match event {
            Some(event) => self.wait_event(&event, spin_budget)?,
            None => Duration::ZERO,
        };
        self.reap(&mut lock(&self.state))?;
        Ok(waited)
    }

    fn read_into(&self, src: &Buffer, dst: &mut [MaybeUninit<u8>], spill: bool, spin_budget: Duration) -> Result<Duration, BackendError> {
        let from = device_ptr(src)?;
        let total = dst.len();
        if total as u64 > src.size() {
            return Err(BackendError::MapRange(format!("reading {total} bytes from a buffer of {} bytes", src.size())));
        }
        let mut longest = Duration::ZERO;
        if total == 0 {
            return Ok(longest);
        }
        // Chunks are staged and waited for one after the other when memory is short, otherwise
        // a few copies are queued first so copying one piece out overlaps the transfer of the
        // next.
        let window = if spill { 1 } else { READ_WINDOW };
        let chunk = if spill {
            SPILL_CHUNK.min(total)
        } else if total < 2 * READ_CHUNK_MIN {
            total
        } else {
            (total / READ_PIECES).clamp(READ_CHUNK_MIN, READ_CHUNK).next_multiple_of(1 << 16).min(total)
        };
        // Declared before the closure so that it is dropped (freeing the staging memory) only
        // after the failure path below has synchronized the stream.
        let mut queued: VecDeque<(usize, usize, Pinned, Option<Arc<CudaEvent>>)> = VecDeque::new();
        let mut offset = 0usize;
        let mut done = 0usize;
        let outcome = (|| {
            while done < total {
                while offset < total && queued.len() < window {
                    let size = chunk.min(total - offset);
                    let mut st = lock(&self.state);
                    let pinned = self.take_pinned(&mut st, size)?;
                    self.bind()?;
                    st.held.buffers.push(src.clone());
                    // SAFETY: the pinned chunk holds `size` bytes and stays alive in `queued`
                    // until the copy has been waited for; the source range lies inside `src`.
                    let enqueued = unsafe {
                        let host = std::slice::from_raw_parts_mut(pinned.ptr.cast::<MaybeUninit<u8>>(), size);
                        result::memcpy_dtoh_async(host, from + offset as u64, self.stream.cu_stream())
                    };
                    enqueued.map_err(gpu)?;
                    st.dirty = true;
                    // Queued before the flush, so that a failing flush cannot free memory a
                    // copy is writing to.
                    queued.push_back((offset, size, pinned, None));
                    let event = self.flush_locked(&mut st)?.ok_or_else(|| BackendError::Gpu("a readback recorded no event".into()))?;
                    if let Some(entry) = queued.back_mut() {
                        entry.3 = Some(event);
                    }
                    offset += size;
                }
                let (at, size, ptr, event) = match queued.front() {
                    Some((at, size, pinned, event)) => (*at, *size, pinned.ptr, event.clone()),
                    None => break,
                };
                let Some(event) = event else {
                    return Err(BackendError::Gpu("a readback recorded no event".into()));
                };
                longest = longest.max(self.wait_event(&event, spin_budget)?);
                // SAFETY: the copy completed, so the pinned bytes are initialized; the chunk is
                // still in `queued`.
                let staged = unsafe { std::slice::from_raw_parts(ptr, size) };
                copy_parallel(staged, &mut dst[at..at + size]);
                done += size;
                if let Some((_, _, pinned, _)) = queued.pop_front() {
                    keep_pinned(&mut lock(&self.state), pinned);
                }
            }
            Ok::<(), BackendError>(())
        })();
        if let Err(error) = outcome {
            // The staging memory must not be freed under a copy still running.
            let _ = self.stream.synchronize();
            return Err(error);
        }
        Ok(longest)
    }

    fn stage_alloc(&self, bytes: u64, dir: StageDir) -> Result<Stage, BackendError> {
        let len = usize::try_from(bytes).map_err(|_| BackendError::OutOfMemory { requested: bytes })?;
        // Waits on the compute lane when too much pinned memory is in flight: no lock the
        // completion path needs is held across it besides the state's own.
        let pinned = self.take_pinned(&mut lock(&self.state), len)?;
        // SAFETY: the block holds at least `len` bytes.
        unsafe { std::ptr::write_bytes(pinned.ptr, 0, len) };
        Ok(Stage::new(dir, CudaStage { pinned: Some(pinned), len, last: Mutex::new(None), state: Arc::downgrade(&self.state) }))
    }

    fn transfer_in(&self, stage: &Stage, dst: &Buffer, dst_offset: u64, bytes: u64, after: &[Ticket]) -> Result<Ticket, BackendError> {
        check_transfer(stage, StageDir::ToDevice, dst, dst_offset, bytes)?;
        let native = stage.downcast_ref::<CudaStage>().ok_or_else(|| BackendError::Unsupported("a stage of another backend".into()))?;
        let to = self.unordered_ptr(dst)?;
        self.bind()?;
        self.transfer_waits(after)?;
        let host = native.bytes();
        // SAFETY: the destination range lies inside `dst`, a block the transfer lane may use; the
        // pinned bytes stay untouched until the ticket completes (the stage waits for it on drop).
        unsafe { result::memcpy_htod_async(to + dst_offset, &host[..bytes as usize], self.transfer.cu_stream()) }.map_err(gpu)?;
        self.transfer_ticket(native)
    }

    fn transfer_out(&self, src: &Buffer, src_offset: u64, stage: &mut Stage, bytes: u64, after: &[Ticket]) -> Result<Ticket, BackendError> {
        check_transfer(stage, StageDir::FromDevice, src, src_offset, bytes)?;
        let from = self.unordered_ptr(src)?;
        self.bind()?;
        self.transfer_waits(after)?;
        let native = stage.downcast_ref::<CudaStage>().ok_or_else(|| BackendError::Unsupported("a stage of another backend".into()))?;
        let host = native.pinned.as_ref().ok_or_else(|| BackendError::Gpu("a stage without memory".into()))?.ptr;
        // SAFETY: the source range lies inside `src`; the stage holds `bytes` bytes of pinned memory
        // that the host leaves alone until the ticket completes.
        unsafe {
            let host = std::slice::from_raw_parts_mut(host.cast::<MaybeUninit<u8>>(), bytes as usize);
            result::memcpy_dtoh_async(host, from + src_offset, self.transfer.cu_stream())
        }
        .map_err(gpu)?;
        self.transfer_ticket(native)
    }

    fn barrier(&self, after: &[Ticket]) -> Result<Ticket, BackendError> {
        let mut st = lock(&self.state);
        self.bind()?;
        for ticket in after {
            if let Some(CudaTicket(event)) = ticket.downcast_ref() {
                self.stream.wait(event).map_err(gpu)?;
                st.dirty = true;
            } else if !ticket.is_done_marker() {
                return Err(BackendError::Unsupported("a ticket of another backend".into()));
            }
        }
        // The batch's event follows the waits just queued and everything recorded before them.
        match self.flush_locked(&mut st)? {
            Some(event) => Ok(Ticket::new(CudaTicket(event))),
            None => Ok(st.submitted.back().map_or_else(Ticket::done, |(event, _)| Ticket::new(CudaTicket(Arc::clone(event))))),
        }
    }

    fn ticket_done(&self, t: &Ticket) -> Result<bool, BackendError> {
        match t.downcast_ref() {
            Some(CudaTicket(event)) => {
                self.bind()?;
                event.try_is_complete().map_err(poll)
            }
            None if t.is_done_marker() => Ok(true),
            None => Err(BackendError::Unsupported("a ticket of another backend".into())),
        }
    }

    fn ticket_wait(&self, t: &Ticket, spin: Duration) -> Result<(), BackendError> {
        match t.downcast_ref() {
            Some(CudaTicket(event)) => self.wait_event(event, spin).map(|_| ()),
            None if t.is_done_marker() => Ok(()),
            None => Err(BackendError::Unsupported("a ticket of another backend".into())),
        }
    }

    fn reclaim_idle(&self) -> Result<(), BackendError> {
        let mut st = lock(&self.state);
        // The uniform chunks go only once the work copying out of them has completed, and
        // before the trim, so that it returns the device chunk too.
        self.sync_locked(&mut st)?;
        st.uniform_device = None;
        st.uniform_host = None;
        self.relieve(&mut st)
    }

    fn enable_profiling(&self) -> Result<(), BackendError> {
        let mut st = lock(&self.state);
        if st.profiler.is_none() {
            st.profiler = Some(Vec::new());
        }
        Ok(())
    }

    #[allow(deprecated)]
    fn finish_profiling(&self, splits: Vec<luma_core::FusionSplit>) -> Result<Profile, BackendError> {
        let samples = {
            let mut st = lock(&self.state);
            let Some(samples) = st.profiler.take() else {
                return Err(BackendError::Profiling("profiling is not enabled".into()));
            };
            self.sync_locked(&mut st)?;
            samples
        };
        let mut timings = Vec::with_capacity(samples.len());
        for sample in &samples {
            let duration = match &sample.events {
                Some((start, end)) => {
                    let ms = start.elapsed_ms(end).map_err(|e| BackendError::Profiling(e.to_string()))?;
                    Duration::from_secs_f64(f64::from(ms.max(0.0)) * 1e-3)
                }
                None => Duration::ZERO,
            };
            timings.push((Arc::clone(&sample.label), duration));
        }
        let mut profile = Profile { splits, ..Profile::default() };
        profile.add_samples(timings);
        Ok(profile)
    }

    fn can_time_launches(&self) -> bool {
        true
    }

    fn time_next_launch(&self, tag: u64) {
        lock(&self.state).armed = Some((std::thread::current().id(), tag));
    }

    fn drain_launch_times(&self, out: &mut Vec<(u64, Duration)>) {
        let mut st = lock(&self.state);
        if st.timed.is_empty() || self.bind().is_err() {
            return;
        }
        while let Some((_, _, end)) = st.timed.front() {
            // Never waits: an unfinished launch stays queued for a later drain.
            if !end.try_is_complete().unwrap_or(false) {
                break;
            }
            let Some((tag, start, end)) = st.timed.pop_front() else { break };
            // Both events are complete: `elapsed_ms` returns at once.
            if let Ok(ms) = start.elapsed_ms(&end) {
                out.push((tag, Duration::from_secs_f64(f64::from(ms.max(0.0)) * 1e-3)));
            }
            if st.free_timing_events.len() < 2 * MAX_TIMED {
                st.free_timing_events.push(start);
                st.free_timing_events.push(end);
            }
        }
    }

    fn is_profiling(&self) -> Result<bool, BackendError> {
        Ok(lock(&self.state).profiler.is_some())
    }

    fn capture_graph(&self, launches: &[GraphLaunch<'_>]) -> Option<Arc<dyn ExecGraph>> {
        if launches.is_empty() {
            return None;
        }
        let captured = self.try_capture(launches, true).or_else(|error| {
            log::debug!("luma: CUDA graph capture with edges failed, capturing a chain: {error}");
            self.try_capture(launches, false)
        });
        match captured {
            Ok(graph) => Some(graph),
            Err(error) => {
                log::debug!("luma: CUDA graph capture failed, replaying launches one by one: {error}");
                None
            }
        }
    }
}

/// Revision of the code generator, a hash of the sources that emit and compile the NVVM IR.
fn cuda_code_revision() -> u64 {
    static REVISION: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *REVISION.get_or_init(|| {
        cache::source_revision(&[
            include_str!("codegen/mod.rs"),
            include_str!("codegen/ops.rs"),
            include_str!("codegen/fb.rs"),
            include_str!("codegen/conv.rs"),
            include_str!("nvvm.rs"),
            include_str!("ptx.rs"),
            include_str!("backend.rs"),
        ])
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use luma_core::hal::{BindingInfo, BindingKind};

    /// A launch of a test: its `(buffer, writes)` arguments, a buffer given by a number, and the
    /// buffer of its workgroup counts.
    type Touches<'a> = (&'a [(u64, bool)], Option<u64>);

    /// What each launch of `launches` must follow.
    fn orders(launches: &[Touches<'_>]) -> Vec<Vec<usize>> {
        let mut hazards = Hazards::default();
        let mut after = Vec::new();
        launches
            .iter()
            .enumerate()
            .map(|(index, (args, counts))| {
                hazards.follow(index, args.iter().copied(), *counts, &mut after);
                after.clone()
            })
            .collect()
    }

    #[test]
    fn an_edge_through_another_launch_is_dropped() {
        // 1 and 2 follow 0, 3 follows both.
        let preds = vec![vec![], vec![0], vec![0], vec![1, 2]];
        let mut seen = vec![0u32; 5];
        let mut stack = Vec::new();
        let mut after = vec![0, 1, 2];
        reduce(&preds, &mut after, &mut seen, 1, &mut stack);
        assert_eq!(after, vec![1, 2]);
        let mut after = vec![0, 3];
        reduce(&preds, &mut after, &mut seen, 2, &mut stack);
        assert_eq!(after, vec![3]);
        let mut after = vec![1, 2];
        reduce(&preds, &mut after, &mut seen, 3, &mut stack);
        assert_eq!(after, vec![1, 2]);
    }

    #[test]
    fn launches_follow_the_buffers_they_share() {
        let (a, b, c, d) = (16, 32, 48, 64);
        let got = orders(&[
            (&[(a, false), (b, true)], None),               // 0: b = f(a)
            (&[(a, false), (c, true)], None),               // 1: reads a with 0, writes another buffer: independent
            (&[(b, false), (c, false), (d, true)], None),   // 2: reads what 0 and 1 wrote (RAW)
            (&[(a, true)], None),                           // 3: overwrites what 0 and 1 read (WAR)
            (&[(b, true), (b, false)], None),               // 4: writes what 0 wrote and 2 read (WAW, WAR); a repeat binding is one access
            (&[(a, false)], None),                          // 5: reads what 3 wrote (RAW) and nothing else
            (&[(d, false), (a, true)], Some(c)),            // 6: after the writer of d and of c, the readers of a since 3 (5), the writer of a (3)
        ]);
        assert_eq!(got, vec![vec![], vec![], vec![0, 1], vec![0, 1], vec![0, 2], vec![3], vec![1, 2, 3, 5]]);
    }

    #[test]
    fn a_missing_driver_is_no_adapter() {
        // SAFETY: only looks for the libraries.
        let present = unsafe { sys::is_culib_present() };
        match open_backend() {
            Ok(_) => assert!(present),
            Err(error) => {
                if !present {
                    assert!(matches!(error, BackendError::NoAdapter(_)), "{error:?}");
                }
            }
        }
    }

    fn open_backend() -> Result<CudaBackend, BackendError> {
        CudaBackend::new()
    }

    /// A device whose matrix product kernels exist, or `None` (the test is skipped).
    fn gemm_backend() -> Option<(CudaBackend, GemmSpec)> {
        let backend = match open_backend() {
            Ok(backend) => backend,
            Err(error) => {
                println!("SKIPPED: no CUDA device ({error})");
                return None;
            }
        };
        let Some(&(rows, cols)) = backend.gemm_tiles(true).first() else {
            println!("SKIPPED: the device has no native matrix product kernels");
            return None;
        };
        Some((backend, GemmSpec::new(rows, cols, false, false, true)))
    }

    fn wait_ready(kernel: &RawKernel) {
        let start = Instant::now();
        while !kernel.is_ready() {
            assert!(start.elapsed() < Duration::from_secs(60), "the kernel never finished compiling");
            std::thread::sleep(Duration::from_millis(1));
        }
    }

    /// The kernel of `spec` as a launchable one, once compiled.
    fn gemm_kernel(backend: &CudaBackend, spec: &GemmSpec) -> CompiledKernel {
        let raw = backend.compile_gemm(spec, None).expect("the backend plans the product");
        wait_ready(&raw);
        let binding = |name: &str, kind| BindingInfo { name: name.into(), kind, element_stride: None, min_size: None };
        let bindings = vec![
            binding("c", BindingKind::StorageReadWrite),
            binding("a", BindingKind::StorageRead),
            binding("b", BindingKind::StorageRead),
            BindingInfo { min_size: Some(32), ..binding("u", BindingKind::Uniform) },
        ];
        CompiledKernel::new("gemm".into(), raw, bindings, 4, None, [1, 1, 1])
    }

    /// Records one product of `m x k` by `k x n`, a single workgroup per output tile.
    fn launch_gemm(backend: &CudaBackend, kernel: &CompiledKernel, spec: &GemmSpec, m: u32, n: u32, k: u32) {
        let make = |bytes: u64| backend.create_buffer(bytes, BufferUsage::STORAGE, false).expect("a buffer");
        let (c, a, b) = (make(u64::from(m) * u64::from(n) * 4), make(u64::from(m) * u64::from(k) * 4), make(u64::from(k) * u64::from(n) * 4));
        let mut uniform = Vec::new();
        for value in [m, n, k, k, k, n, 0, 0] {
            uniform.extend_from_slice(&value.to_le_bytes());
        }
        let grid = [n.div_ceil(spec.cols), m.div_ceil(spec.rows), 1];
        let args = [RawArg::write(&c, c.size()), RawArg::read(&a, a.size()), RawArg::read(&b, b.size()), RawArg::Uniform(&uniform)];
        backend.record_launch(kernel, Dispatch::Direct(grid), [grid[0] * spec.cols, grid[1] * spec.rows, 1], &args).expect("the launch is recorded");
    }

    #[test]
    fn a_gemm_kernel_compiles_in_the_background() {
        let Some((backend, spec)) = gemm_backend() else { return };
        let raw = backend.compile_gemm(&spec, None).expect("the backend plans the product");
        // Loading the PTX takes a driver call on a worker thread: it is not done on return.
        assert!(!raw.is_ready(), "compile_gemm returned a loaded module");
        wait_ready(&raw);
        raw.resolve::<CudaKernel>().expect("the module loads");
    }

    #[test]
    fn an_armed_launch_reports_a_time() {
        let Some((backend, spec)) = gemm_backend() else { return };
        let kernel = gemm_kernel(&backend, &spec);
        assert!(backend.can_time_launches());
        // Not armed: nothing is reported.
        launch_gemm(&backend, &kernel, &spec, 1024, 1024, 1024);
        backend.sync(Duration::ZERO).unwrap();
        let mut out = Vec::new();
        backend.drain_launch_times(&mut out);
        assert!(out.is_empty(), "{out:?}");
        backend.time_next_launch(7);
        launch_gemm(&backend, &kernel, &spec, 1024, 1024, 1024);
        // The tag was taken by that launch: the next one is not timed.
        launch_gemm(&backend, &kernel, &spec, 1024, 1024, 1024);
        backend.sync(Duration::ZERO).unwrap();
        backend.drain_launch_times(&mut out);
        assert_eq!(out.len(), 1, "{out:?}");
        assert_eq!(out[0].0, 7);
        assert!(out[0].1 > Duration::ZERO, "{out:?}");
        backend.drain_launch_times(&mut out);
        assert_eq!(out.len(), 1);
    }

    #[test]
    fn draining_never_blocks_on_unfinished_work() {
        let Some((backend, spec)) = gemm_backend() else { return };
        let kernel = gemm_kernel(&backend, &spec);
        backend.time_next_launch(3);
        // Hundreds of GFLOP: far longer than the drain below.
        launch_gemm(&backend, &kernel, &spec, 4096, 4096, 8192);
        backend.flush_batch().unwrap();
        let mut out = Vec::new();
        let start = Instant::now();
        backend.drain_launch_times(&mut out);
        let drained = start.elapsed();
        assert!(out.is_empty(), "{out:?}");
        assert!(backend.pending.load(Ordering::Relaxed) == 0 && lock(&backend.state).timed.len() == 1);
        backend.sync(Duration::ZERO).unwrap();
        println!("an unfinished drain took {drained:?}");
        backend.drain_launch_times(&mut out);
        assert_eq!(out.len(), 1, "{out:?}");
        assert_eq!(out[0].0, 3);
        assert!(out[0].1 > Duration::ZERO);
    }

    #[test]
    fn a_graph_capture_is_never_armed() {
        let Some((backend, spec)) = gemm_backend() else { return };
        let kernel = gemm_kernel(&backend, &spec);
        let make = |bytes: u64| backend.create_buffer(bytes, BufferUsage::STORAGE, false).expect("a buffer");
        let (c, a, b) = (make(64 * 64 * 4), make(64 * 64 * 4), make(64 * 64 * 4));
        let mut uniform = Vec::new();
        for value in [64u32, 64, 64, 64, 64, 64, 0, 0] {
            uniform.extend_from_slice(&value.to_le_bytes());
        }
        let args = [RawArg::write(&c, c.size()), RawArg::read(&a, a.size()), RawArg::read(&b, b.size()), RawArg::Uniform(&uniform)];
        let grid = [64u32.div_ceil(spec.cols), 64u32.div_ceil(spec.rows), 1];
        let launch = GraphLaunch { kernel: &kernel, dispatch: Dispatch::Direct(grid), size: [64, 64, 1], args: &args };

        // Armed, then the launch goes into a graph: the tag is dropped.
        backend.time_next_launch(1);
        let graph = backend.capture_graph(&[launch]).expect("the graph is captured");
        assert!(lock(&backend.state).armed.is_none());
        launch_gemm(&backend, &kernel, &spec, 256, 256, 256);
        backend.record_graph(&*graph, 0).unwrap();
        // Armed, then the batch is flushed: the tag is dropped too.
        backend.time_next_launch(2);
        backend.flush_batch().unwrap();
        launch_gemm(&backend, &kernel, &spec, 256, 256, 256);
        backend.sync(Duration::ZERO).unwrap();
        let mut out = Vec::new();
        backend.drain_launch_times(&mut out);
        assert!(out.is_empty(), "{out:?}");
    }
}
