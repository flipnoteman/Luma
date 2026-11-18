//! The CUDA runtime: a device pool, buffers, NVVM compilation (the driver's own NVVM library) with
//! a PTX disk cache, one ordered
//! stream per backend and CUDA graph capture and replay.
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
use std::sync::{Arc, Mutex, MutexGuard, OnceLock, PoisonError};
use std::time::{Duration, Instant};

use cudarc::driver::{CudaContext, CudaEvent, CudaFunction, CudaGraph, CudaStream, DriverError, result, sys};
use cudarc::nvrtc::Ptx;
use luma_core::hal::cache::{self, PTX_MAGIC};
use luma_core::hal::copy_parallel;
use luma_core::kir::{self, KernelModule};
use luma_core::{
    Backend, BackendError, BackendKind, Buffer, BufferUsage, CompiledKernel, CostModel, DeviceInfo, DeviceLimits, DiskCache, Dispatch, ExecGraph, Features, FusionSplit, GraphLaunch,
    NativeBuffer, NativeKernel, Profile, RawArg, RawKernel, TuningProfile,
};

use crate::codegen::{self, CudaArgAbi};

/// Uploads and readbacks larger than this go in pieces of this size.
const TRANSFER_CHUNK: usize = 32 << 20;

/// Readback chunks queued at once (copies overlap the host copying the previous one out).
const READ_WINDOW: usize = 4;

/// Pinned staging memory held by unfinished work, above which a new request waits for it.
const PINNED_BUSY_CAP: usize = 512 << 20;

/// Most bytes of idle pinned staging memory kept for reuse.
const PINNED_IDLE_CAP: usize = 256 << 20;

/// Smallest pinned staging allocation.
const PINNED_MIN: usize = 4096;

/// Size of the chunks that uniforms too large to pass by value are copied into.
const UNIFORM_CHUNK: usize = 1 << 20;

/// Alignment of a uniform slot in device memory.
const UNIFORM_ALIGN: usize = 256;

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

/// A device allocation, freed when dropped.
#[derive(Debug)]
struct CudaBuffer {
    dptr: sys::CUdeviceptr,
    ctx: Arc<CudaContext>,
}

impl NativeBuffer for CudaBuffer {}

impl Drop for CudaBuffer {
    fn drop(&mut self) {
        if bind_raw(&self.ctx).is_ok() {
            // SAFETY: allocated by `cuMemAlloc` and freed once; the driver orders the free after
            // work still using it, and unfinished work is anyway kept alive by its batch.
            let _ = unsafe { result::free_sync(self.dptr) };
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

/// A compiled kernel.
#[derive(Debug)]
struct CudaKernel {
    func: CudaFunction,
    workgroup: [u32; 3],
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

impl ExecGraph for CudaExecGraph {}

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
                (CudaArgAbi::DevicePtr, RawArg::Storage { buffer, .. }) => self.vals.push(device_ptr(buffer)?),
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
    free_events: Vec<CudaEvent>,
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
    cc: (i32, i32),
    nvvm: crate::nvvm::Nvvm,
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
        };
        Ok(CudaShared { ctx, info, disk, cost: Arc::new(OnceLock::new()), cc: (major, minor), nvvm })
    }
}

/// The CUDA backend.
pub struct CudaBackend {
    shared: Arc<CudaShared>,
    stream: Arc<CudaStream>,
    /// The stream graphs are captured on, created on first use.
    capture: Mutex<Option<Arc<CudaStream>>>,
    pending: AtomicU32,
    state: Mutex<State>,
}

impl fmt::Debug for CudaBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CudaBackend").field("device", &self.shared.info.name).finish_non_exhaustive()
    }
}

impl Drop for CudaBackend {
    fn drop(&mut self) {
        // Everything the state holds is released only after the stream has run dry.
        let _ = self.stream.synchronize();
    }
}

impl CudaBackend {
    /// Opens the first CUDA device.
    pub fn new() -> Result<CudaBackend, BackendError> {
        let shared = CudaShared::acquire()?;
        let stream = shared.ctx.new_stream().map_err(|e| BackendError::RequestDevice(e.to_string()))?;
        let state = State {
            dispatches: 0,
            dirty: false,
            held: Held::default(),
            last_kernel: 0,
            submitted: VecDeque::new(),
            free_events: Vec::new(),
            pinned_free: Vec::new(),
            pinned_idle: 0,
            pinned_busy: 0,
            uniform_device: None,
            uniform_host: None,
            uniform_retired: Vec::new(),
            params: Params::default(),
            profiler: None,
        };
        Ok(CudaBackend { shared, stream, capture: Mutex::new(None), pending: AtomicU32::new(0), state: Mutex::new(state) })
    }

    fn bind(&self) -> Result<(), BackendError> {
        self.shared.ctx.bind_to_thread().map_err(gpu)
    }

    fn alloc_block(&self, bytes: u64) -> Result<CudaBuffer, BackendError> {
        self.bind()?;
        // SAFETY: plain allocation; the block frees itself.
        match unsafe { result::malloc_sync(bytes.max(MIN_BUFFER) as usize) } {
            Ok(dptr) => Ok(CudaBuffer { dptr, ctx: Arc::clone(&self.shared.ctx) }),
            Err(e) if is_out_of_memory(&e) => Err(BackendError::OutOfMemory { requested: bytes }),
            Err(e) => Err(gpu(e)),
        }
    }

    /// Records an event after the work so far, taking the batch's resources with it. `None` when
    /// nothing was recorded since the last flush.
    fn flush_locked(&self, st: &mut State) -> Result<Option<Arc<CudaEvent>>, BackendError> {
        st.dispatches = 0;
        self.pending.store(0, Ordering::Relaxed);
        if !st.dirty {
            return Ok(None);
        }
        self.bind()?;
        let event = match st.free_events.pop() {
            Some(event) => event,
            None => self.shared.ctx.new_event(None).map_err(gpu)?,
        };
        event.record(&self.stream).map_err(gpu)?;
        let event = Arc::new(event);
        st.dirty = false;
        st.last_kernel = 0;
        let held = std::mem::take(&mut st.held);
        st.submitted.push_back((Arc::clone(&event), held));
        self.reap(st)?;
        Ok(Some(event))
    }

    /// Releases the resources of every batch that has completed.
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
        Ok(())
    }

    fn release(st: &mut State, held: Held) {
        for pinned in held.pinned {
            st.pinned_busy -= pinned.cap;
            if st.pinned_idle + pinned.cap <= PINNED_IDLE_CAP {
                st.pinned_idle += pinned.cap;
                st.pinned_free.push(pinned);
            }
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

    /// Pinned memory of at least `bytes` bytes, from the idle pool when there is some.
    fn take_pinned(&self, st: &mut State, bytes: usize) -> Result<Pinned, BackendError> {
        let class = bytes.max(PINNED_MIN).next_power_of_two();
        let find = |st: &mut State| {
            let at = st.pinned_free.iter().position(|p| p.cap == class)?;
            st.pinned_idle -= class;
            Some(st.pinned_free.swap_remove(at))
        };
        if let Some(pinned) = find(st) {
            return Ok(pinned);
        }
        while st.pinned_busy + class > PINNED_BUSY_CAP && st.pinned_busy > 0 {
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
        Ok(())
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
        let timing = if st.profiler.is_some() {
            let start = self.shared.ctx.new_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT)).map_err(gpu)?;
            start.record(&self.stream).map_err(gpu)?;
            st.dirty = true;
            Some(start)
        } else {
            None
        };
        // SAFETY: the function belongs to a module kept alive by the kernel; `params.ptrs` point
        // to values that live until the call returns (the driver copies them), and every buffer
        // named stays alive in the batch until the work completes.
        let enqueued = unsafe {
            result::launch_kernel(
                native.func.cu_function(),
                (grid[0], grid[1], grid[2]),
                (native.workgroup[0], native.workgroup[1], native.workgroup[2]),
                0,
                self.stream.cu_stream(),
                &mut params.ptrs[..],
            )
        };
        if enqueued.is_ok() {
            st.dirty = true;
            st.dispatches += 1;
        }
        enqueued.map_err(gpu)?;
        if let Some(start) = timing {
            let end = self.shared.ctx.new_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT)).map_err(gpu)?;
            end.record(&self.stream).map_err(gpu)?;
            if let Some(samples) = st.profiler.as_mut() {
                samples.push(Sample { label: Arc::clone(&kernel.label), events: Some((start, end)) });
            }
        }
        Ok(())
    }

    /// The kernel compiled from NVVM IR `source`, through the disk cache.
    fn build_function(&self, source: &str, label: &str) -> Result<CudaFunction, BackendError> {
        let (major, minor) = self.shared.cc;
        // Last byte: emitter revision; bump whenever the emitted IR changes.
        let meta = [major as u8, minor as u8, self.shared.nvvm.version.0 as u8, self.shared.nvvm.version.1 as u8, 3];
        let hash = cache::fnv1a(source.as_bytes()) ^ cache::fnv1a(&meta).wrapping_mul(0x9e37_79b9_7f4a_7c15);
        let stem = format!("ptx-{hash:016x}");
        let load = |text: String| self.shared.ctx.load_module(Ptx::from_src(text)).and_then(|m| m.load_function(ENTRY));
        if let Some(text) = self.shared.disk.load_blob(&stem, PTX_MAGIC).and_then(|bytes| String::from_utf8(bytes).ok())
            && !text.contains('\0')
        {
            match load(text) {
                Ok(func) => return Ok(func),
                // A stale cache entry: rebuild it.
                Err(_) => self.shared.disk.remove_blob(&stem),
            }
        }
        let text = self.shared.nvvm.compile(source, major, minor).map_err(|e| BackendError::ShaderCompile(format!("kernel `{label}`:\n{e}")))?;
        let func = load(text.clone()).map_err(gpu)?;
        self.shared.disk.store_blob(&stem, PTX_MAGIC, text.as_bytes());
        Ok(func)
    }

    fn try_capture(&self, launches: &[GraphLaunch<'_>]) -> Result<Arc<dyn ExecGraph>, BackendError> {
        if lock(&self.state).profiler.is_some() {
            return Err(BackendError::Unsupported("graphs are not captured while profiling".into()));
        }
        let natives: Vec<&CudaKernel> = launches
            .iter()
            .map(|l| l.kernel.raw.downcast_ref::<CudaKernel>().ok_or_else(|| BackendError::Unsupported("kernel was compiled by another backend".into())))
            .collect::<Result<_, _>>()?;
        let dispatches: Vec<([u32; 3], u64, Option<&Buffer>)> = launches.iter().map(|l| resolve_dispatch(l.dispatch)).collect::<Result<_, _>>()?;

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
            let block = self.alloc_block(staging.len() as u64)?;
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
        let mut res = GraphRes { buffers: Vec::new(), kernels: Vec::new(), scratch };
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
                // SAFETY: as in `launch_locked`; the captured node copies the parameters.
                unsafe {
                    result::launch_kernel(
                        native.func.cu_function(),
                        (grid[0], grid[1], grid[2]),
                        (native.workgroup[0], native.workgroup[1], native.workgroup[2]),
                        0,
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
    fn info(&self) -> &DeviceInfo {
        &self.shared.info
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

    fn recycle_mappable(&self, buffer: Buffer) {
        // Nothing is mappable here; the work still using it keeps it alive.
        drop(buffer);
    }

    fn compile_kernel(&self, module: &KernelModule, used: Option<&[bool]>, label: &str, trusted: bool) -> Result<RawKernel, BackendError> {
        if let Some(mask) = used
            && mask.len() != module.bindings.len()
        {
            return Err(BackendError::KernelBinding(format!("the used-binding mask has {} entries for {} bindings", mask.len(), module.bindings.len())));
        }
        let checked: Cow<'_, KernelModule> = if trusted { Cow::Borrowed(module) } else { kir::opt::with_bounds_checks(module) };
        let optimized = kir::optimize_for_target(&checked, true);
        let translated = codegen::emit_with_workgroup_init(&optimized, used, !trusted);
        let strides: Vec<u32> = translated.array_lengths.iter().map(|&b| optimized.bindings.get(b as usize).and_then(|d| d.element_stride).unwrap_or(1)).collect();

        self.bind()?;
        let func = self.build_function(&translated.source, label)?;
        Ok(RawKernel::new(CudaKernel { func, workgroup: optimized.workgroup_size, param_abis: translated.param_abis, array_lengths: translated.array_lengths, strides }))
    }

    fn record_launch(&self, kernel: &CompiledKernel, dispatch: Dispatch<'_>, size: [u32; 3], args: &[RawArg<'_>]) -> Result<u32, BackendError> {
        kernel.check_count(args.len())?;
        kernel.check_kinds(args)?;
        let native = kernel.raw.downcast_ref::<CudaKernel>().ok_or_else(|| BackendError::Unsupported("kernel was compiled by another backend".into()))?;
        let (grid, indirect, indirect_buffer) = resolve_dispatch(dispatch)?;
        let mut st = lock(&self.state);
        if grid.contains(&0) {
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
        for (index, piece) in data.chunks(TRANSFER_CHUNK).enumerate() {
            let pinned = self.take_pinned(&mut st, piece.len())?;
            // SAFETY: `piece.len() <= pinned.cap`; the pinned memory is kept in the batch until
            // the copy out of it has completed.
            let staged = unsafe {
                copy_parallel(piece, std::slice::from_raw_parts_mut(pinned.ptr.cast::<MaybeUninit<u8>>(), piece.len()));
                std::slice::from_raw_parts(pinned.ptr, piece.len())
            };
            let to = dptr + (index * TRANSFER_CHUNK) as u64;
            // Held after `take_pinned` (which can flush and reap) and before the copy is queued.
            if st.held.buffers.last() != Some(dst) {
                st.held.buffers.push(dst.clone());
            }
            // SAFETY: the destination range lies inside `dst`.
            let enqueued = unsafe { result::memcpy_htod_async(to, staged, self.stream.cu_stream()) };
            // The staging memory stays with the batch whether or not the call reports an error.
            st.pinned_busy += pinned.cap;
            st.held.pinned.push(pinned);
            st.dirty = true;
            enqueued.map_err(gpu)?;
        }
        Ok(())
    }

    fn record_graph(&self, graph: &dyn ExecGraph, dispatches: u32) -> Result<u32, BackendError> {
        let graph = graph.downcast_ref::<CudaExecGraph>().ok_or_else(|| BackendError::Unsupported("graph was captured by another backend".into()))?;
        let mut st = lock(&self.state);
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
        let chunk = if total > 2 * TRANSFER_CHUNK || spill { TRANSFER_CHUNK.min(total) } else { total };
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
                    let mut st = lock(&self.state);
                    if st.pinned_idle + pinned.cap <= PINNED_IDLE_CAP {
                        st.pinned_idle += pinned.cap;
                        st.pinned_free.push(pinned);
                    }
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

    fn reclaim_idle(&self) -> Result<(), BackendError> {
        let mut st = lock(&self.state);
        self.relieve(&mut st)?;
        st.uniform_device = None;
        st.uniform_host = None;
        Ok(())
    }

    fn enable_profiling(&self) -> Result<(), BackendError> {
        let mut st = lock(&self.state);
        if st.profiler.is_none() {
            st.profiler = Some(Vec::new());
        }
        Ok(())
    }

    fn finish_profiling(&self, splits: Vec<FusionSplit>) -> Result<Profile, BackendError> {
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
        let mut profile = Profile::default();
        profile.add_samples(timings);
        profile.splits = splits;
        Ok(profile)
    }

    fn is_profiling(&self) -> Result<bool, BackendError> {
        Ok(lock(&self.state).profiler.is_some())
    }

    fn capture_graph(&self, launches: &[GraphLaunch<'_>]) -> Option<Arc<dyn ExecGraph>> {
        if launches.is_empty() {
            return None;
        }
        match self.try_capture(launches) {
            Ok(graph) => Some(graph),
            Err(error) => {
                log::debug!("luma: CUDA graph capture failed, replaying launches one by one: {error}");
                None
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
