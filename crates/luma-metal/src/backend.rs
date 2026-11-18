//! The Metal runtime: buffers, kernel compilation and one ordered command stream.
//!
//! Work is recorded into a single pending `MTLCommandBuffer`: consecutive launches share one
//! compute encoder, clears, copies and uploads go through blit encoders in between. Buffers are
//! `StorageModeShared`, so the host writes through `contents()` without a GPU copy; Metal's
//! hazard tracking orders the GPU's own reads and writes, but it never sees a host write. Every
//! host-written buffer (recycled mappable storage, upload staging, uniform ring chunks) is
//! therefore retired with the sequence number of the command buffer that may still read it and
//! only becomes writable again once that command buffer's completion handler has run.

#[cfg(not(target_vendor = "apple"))]
pub fn open() -> Result<Box<dyn luma_core::Backend>, luma_core::BackendError> {
    Err(luma_core::BackendError::NoAdapter("Metal is only supported on Apple platforms".into()))
}

#[cfg(target_vendor = "apple")]
pub use imp::{MetalBackend, open};

#[cfg(target_vendor = "apple")]
mod imp {
    use std::borrow::Cow;
    use std::collections::{BTreeSet, HashMap};
    use std::ffi::c_void;
    use std::fmt;
    use std::mem::MaybeUninit;
    use std::ptr::NonNull;
    use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
    use std::sync::{Arc, Condvar, Mutex, MutexGuard, OnceLock, PoisonError};
    use std::time::{Duration, Instant};

    use block2::RcBlock;
    use luma_core::hal::{cache, copy_parallel};
    use luma_core::kir::KernelModule;
    use luma_core::{
        Backend, BackendError, BackendKind, Buffer, BufferUsage, COPY_BUFFER_ALIGNMENT, CompiledKernel, CostModel, DeviceInfo, DeviceLimits,
        DiskCache, Dispatch, ExecGraph, Features, FreeUploads, FusionSplit, GraphLaunch, NativeBuffer, NativeKernel, Profile, RawArg,
        RawKernel, TuningProfile,
    };
    use objc2::rc::{Retained, autoreleasepool};
    use objc2::runtime::ProtocolObject;
    use objc2_foundation::{NSRange, NSString};
    use objc2_metal::{
        MTLArgumentBuffersTier, MTLBlitCommandEncoder, MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder,
        MTLCommandQueue, MTLCompileOptions, MTLComputeCommandEncoder, MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDevice,
        MTLGPUFamily, MTLLibrary, MTLResourceOptions, MTLResourceUsage, MTLSize,
    };

    use crate::codegen::{MslAbi, emit_with_workgroup_init};

    type MtlBuffer = Retained<ProtocolObject<dyn MTLBuffer>>;
    type MtlCommandBuffer = Retained<ProtocolObject<dyn MTLCommandBuffer>>;
    type MtlCompute = Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>;
    type MtlBlit = Retained<ProtocolObject<dyn MTLBlitCommandEncoder>>;

    /// Usage of storage buffers.
    const STORAGE_USAGE: BufferUsage = BufferUsage::STORAGE.union(BufferUsage::COPY_DST).union(BufferUsage::COPY_SRC);

    /// Usage of storage buffers the host writes directly (zero-copy uploads).
    const MAPPABLE_STORAGE_USAGE: BufferUsage = STORAGE_USAGE.union(BufferUsage::MAP_WRITE);

    /// Usage of upload staging buffers.
    const UPLOAD_USAGE: BufferUsage = BufferUsage::MAP_WRITE.union(BufferUsage::COPY_SRC);

    /// Smallest buffer the pools hand out.
    const MIN_BUFFER_SIZE: u64 = 16;

    /// Size of one uniform ring chunk. Uniform values above [`SET_BYTES_MAX`] are packed into it.
    const CHUNK_BYTES: u64 = 256 * 1024;

    /// Alignment of a uniform slot in a chunk: the offset alignment of `constant` buffers on macOS.
    const CHUNK_ALIGN: u64 = 256;

    /// Most idle uniform chunks kept for reuse.
    const MAX_FREE_CHUNKS: usize = 16;

    /// Largest value `setBytes` takes.
    const SET_BYTES_MAX: usize = 4096;

    /// Stands in for an empty uniform, which `setBytes` rejects.
    const ZEROS: [u8; 16] = [0; 16];

    /// Locks `mutex`, ignoring poisoning: every mutex of the crate guards data that stays
    /// consistent if a holder panicked.
    fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
        mutex.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Size class for a buffer holding `bytes` bytes: the next power of two (at least
    /// [`MIN_BUFFER_SIZE`]), clamped to `max_size`. `None` if `bytes` itself exceeds `max_size`.
    fn size_class(bytes: u64, max_size: u64) -> Option<u64> {
        let max_size = max_size & !(COPY_BUFFER_ALIGNMENT - 1);
        if bytes > max_size {
            return None;
        }
        let class = bytes.max(MIN_BUFFER_SIZE).checked_next_power_of_two().unwrap_or(max_size);
        Some(class.min(max_size))
    }

    // -- Native handles --

    /// A Metal buffer held by a [`Buffer`].
    #[derive(Debug)]
    struct MetalBuffer(MtlBuffer);

    // SAFETY: Metal buffers may be used from any thread. Creation, binding, `gpuAddress` and
    // `contents` need no external locking; the host writes through `contents` are ordered against
    // the GPU by the sequence numbers of the backend's completion handlers.
    unsafe impl Send for MetalBuffer {}
    // SAFETY: see above.
    unsafe impl Sync for MetalBuffer {}

    impl NativeBuffer for MetalBuffer {}

    /// A compiled pipeline held by a [`RawKernel`].
    #[derive(Debug)]
    struct MetalKernel {
        pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
        abi: MslAbi,
        /// `(binding, element stride)` of every binding whose element count the kernel reads,
        /// ascending, in the order the lengths are packed.
        array_lengths: Vec<(u32, u32)>,
    }

    // SAFETY: a compute pipeline state is immutable and Metal documents it as thread safe.
    unsafe impl Send for MetalKernel {}
    // SAFETY: see above.
    unsafe impl Sync for MetalKernel {}

    impl NativeKernel for MetalKernel {}

    fn raw(buffer: &Buffer) -> Result<&MetalBuffer, BackendError> {
        buffer.downcast_ref::<MetalBuffer>().ok_or_else(|| BackendError::Unsupported("the buffer belongs to another backend".into()))
    }

    // -- Completion tracking --

    #[derive(Debug, Default)]
    struct DoneState {
        /// Every command buffer up to this sequence number has completed.
        next: u64,
        /// Completed ones beyond `next + 1`.
        pending: BTreeSet<u64>,
    }

    /// Which submitted command buffers have completed, fed by their completion handlers.
    #[derive(Debug, Default)]
    struct Done {
        state: Mutex<DoneState>,
        cv: Condvar,
        /// Mirror of `DoneState::next` for lock-free polling.
        contiguous: AtomicU64,
        /// The first error a command buffer reported.
        error: Mutex<Option<String>>,
    }

    impl Done {
        fn complete(&self, seq: u64, error: Option<String>) {
            if let Some(error) = error {
                lock(&self.error).get_or_insert(error);
            }
            let mut guard = lock(&self.state);
            let state = &mut *guard;
            state.pending.insert(seq);
            while state.pending.remove(&(state.next + 1)) {
                state.next += 1;
            }
            self.contiguous.store(state.next, Ordering::Release);
            drop(guard);
            self.cv.notify_all();
        }

        fn reached(&self, seq: u64) -> bool {
            self.contiguous.load(Ordering::Acquire) >= seq
        }

        fn wait(&self, seq: u64) {
            let mut guard = lock(&self.state);
            while guard.next < seq {
                guard = self.cv.wait_timeout(guard, Duration::from_millis(50)).unwrap_or_else(PoisonError::into_inner).0;
            }
        }
    }

    // -- Pooled host-written buffers --

    /// A shared-memory chunk uniform values larger than `setBytes` takes are written into.
    #[derive(Debug)]
    struct Chunk {
        buf: MetalBuffer,
        size: u64,
        used: u64,
    }

    #[derive(Debug)]
    enum Item {
        Mappable(Buffer),
        Upload(Buffer),
        Chunk(Chunk),
    }

    /// A host-written buffer the GPU may still read; reusable once command buffer `ready_after`
    /// (and so every earlier one) has completed.
    #[derive(Debug)]
    struct Retired {
        ready_after: u64,
        item: Item,
    }

    // -- Recording state --

    /// What the open compute encoder has bound that a launch may skip rebinding.
    #[derive(Default)]
    struct Bound {
        /// `RawKernel::id` of the pipeline set, 0 for none.
        pipeline: u64,
        /// The `LumaGrid` at `buffer(0)`.
        grid: Option<[u32; 4]>,
    }

    struct Batch {
        cb: Option<MtlCommandBuffer>,
        compute: Option<MtlCompute>,
        blit: Option<MtlBlit>,
        bound: Bound,
        /// GPU addresses of buffers made resident in the open compute encoder, and whether for
        /// writing, for kernels that reach them through an argument buffer.
        resident: HashMap<u64, bool>,
        dispatches: u32,
        /// The uniform chunk being filled and the full ones of this command buffer.
        chunk: Option<Chunk>,
        chunks: Vec<Chunk>,
        /// Staging buffers the pending command buffer copies out of.
        uploads: Vec<Buffer>,
        scratch: Vec<u8>,
        /// Profiled launches, each in a command buffer of its own; `None` for an empty grid.
        profiler: Option<Vec<(Arc<str>, Option<MtlCommandBuffer>)>>,
    }

    // SAFETY: the encoders and the command buffer are only used under the mutex that owns the
    // batch, one thread at a time, which is all Metal requires of them.
    unsafe impl Send for Batch {}

    // -- Devices --

    /// A Metal device with what is derived from it once.
    struct MetalSharedDevice {
        device: Retained<ProtocolObject<dyn MTLDevice>>,
        info: DeviceInfo,
        disk: Arc<DiskCache>,
        cost: Arc<OnceLock<CostModel>>,
        /// Most bytes of idle buffers a backend keeps for reuse.
        cap: u64,
    }

    /// Idle devices of finished backends.
    static IDLE: Mutex<Vec<Arc<MetalSharedDevice>>> = Mutex::new(Vec::new());

    /// Serializes [`acquire`], so that a burst of threads opening backends reuses the devices that
    /// finished ones return in the meantime.
    static OPENING: Mutex<()> = Mutex::new(());

    fn open_shared() -> Result<MetalSharedDevice, BackendError> {
        autoreleasepool(|_| {
            let device = MTLCreateSystemDefaultDevice().ok_or_else(|| BackendError::NoAdapter("no Metal GPU device available".into()))?;
            let name = device.name().to_string();
            let apple_silicon = device.supportsFamily(MTLGPUFamily::Apple7);
            let memory_bytes = Some(device.recommendedMaxWorkingSetSize()).filter(|&bytes| bytes > 0);
            let threads = device.maxThreadsPerThreadgroup();
            let invocations = (threads.width as u64).saturating_mul(threads.height as u64).saturating_mul(threads.depth as u64).min(1024) as u32;
            let features = Features::TIMESTAMP_QUERY | Features::SHADER_F16 | if apple_silicon { Features::SUBGROUP } else { Features::NONE };
            let info = DeviceInfo {
                backend: BackendKind::Metal,
                name: name.clone(),
                cpu: false,
                zero_copy: device.hasUnifiedMemory(),
                profile: if apple_silicon { TuningProfile::Apple } else { TuningProfile::Generic },
                features,
                subgroup_size: apple_silicon.then_some((32, 32)),
                memory_bytes,
                ieee_f32_intrinsics: false,
                hardware_fma: apple_silicon.then_some(true),
                limits: DeviceLimits {
                    max_buffer_bytes: (device.maxBufferLength() as u64) & !3,
                    max_storage_buffers: if device.argumentBuffersSupport() == MTLArgumentBuffersTier::Tier2 { 256 } else { 28 },
                    max_uniform_bytes: 65536,
                    uniform_align: 16,
                    max_dynamic_uniforms: u32::MAX,
                    max_workgroups_per_dim: 65535,
                    max_workgroup_size: [threads.width as u32, threads.height as u32, threads.depth as u32],
                    max_workgroup_invocations: invocations,
                    max_workgroup_storage_bytes: device.maxThreadgroupMemoryLength() as u32,
                },
            };
            let identity = format!("{name}\n{}\n{apple_silicon}", std::env::consts::ARCH);
            let key = format!("metal-{:016x}", cache::fnv1a(identity.as_bytes()));
            let disk = DiskCache::open(cache::directory(), key);
            let cap = memory_bytes.map_or(1 << 30, |memory| memory / 5 * 4 / 2);
            Ok(MetalSharedDevice { device, info, disk, cost: Arc::new(OnceLock::new()), cap })
        })
    }

    /// An idle pooled device, or a newly opened one.
    fn acquire() -> Result<Arc<MetalSharedDevice>, BackendError> {
        let _opening = lock(&OPENING);
        if let Some(shared) = lock(&IDLE).pop() {
            return Ok(shared);
        }
        open_shared().map(Arc::new)
    }

    /// Returns the device to the pool when the backend is gone.
    struct Lease(Option<Arc<MetalSharedDevice>>);

    impl Drop for Lease {
        fn drop(&mut self) {
            if let Some(shared) = self.0.take() {
                lock(&IDLE).push(shared);
            }
        }
    }

    /// The Metal backend.
    pub struct MetalBackend {
        info: DeviceInfo,
        disk: Arc<DiskCache>,
        cost: Arc<OnceLock<CostModel>>,
        queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
        done: Arc<Done>,
        /// Sequence number of the last committed command buffer.
        submitted: AtomicU64,
        /// Whether the pending command buffer exists.
        open_work: AtomicBool,
        /// `Batch::dispatches`, readable without the lock of the batch.
        dispatches_pending: AtomicU32,
        retired: Mutex<Vec<Retired>>,
        /// Idle mappable storage buffers, safe to write.
        mappable: Mutex<FreeUploads>,
        /// Idle upload staging buffers, safe to write.
        uploads: Mutex<FreeUploads>,
        free_chunks: Mutex<Vec<Chunk>>,
        max_buffer_bytes: u64,
        batch: Mutex<Batch>,
        /// Must stay the last field: the device goes back to the pool only once everything of
        /// this backend is gone.
        lease: Lease,
    }

    impl fmt::Debug for MetalBackend {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.debug_struct("MetalBackend").field("device", &self.info.name).finish_non_exhaustive()
        }
    }

    impl Drop for MetalBackend {
        fn drop(&mut self) {
            // Submitted work must be finished before its buffers and the device are handed on.
            let _ = self.sync(Duration::ZERO);
        }
    }

    /// Opens the default Metal device and returns the backend running on it.
    pub fn open() -> Result<Box<dyn Backend>, BackendError> {
        MetalBackend::new().map(|backend| Box::new(backend) as Box<dyn Backend>)
    }

    /// Copies `data` into the start of `buffer` through its shared memory.
    fn host_write(buffer: &MetalBuffer, data: &[u8]) {
        if data.is_empty() {
            return;
        }
        let dst = buffer.0.contents().as_ptr().cast::<MaybeUninit<u8>>();
        // SAFETY: callers pass buffers of at least `data.len()` bytes (a size class of it) that no
        // GPU work reads or writes: pooled ones are retired until completion, fresh ones unused.
        let dst = unsafe { std::slice::from_raw_parts_mut(dst, data.len()) };
        copy_parallel(data, dst);
    }

    fn set_bytes(encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>, bytes: &[u8], index: usize) {
        let bytes = if bytes.is_empty() { &ZEROS[..] } else { bytes };
        // SAFETY: the pointer is valid for `bytes.len()` bytes and Metal copies them at once; the
        // binding index is below 31, which the code generator guarantees.
        unsafe { encoder.setBytes_length_atIndex(NonNull::from(bytes).cast::<c_void>(), bytes.len(), index) };
    }

    /// Makes `buf` resident in the open encoder for kernels that reach it through an argument
    /// buffer, and returns its GPU address.
    fn use_resource(
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        resident: &mut HashMap<u64, bool>,
        buf: &ProtocolObject<dyn MTLBuffer>,
        write: bool,
    ) -> u64 {
        let address = buf.gpuAddress();
        let have = resident.get(&address).copied();
        if have.is_none() || (write && have == Some(false)) {
            let usage = if write { MTLResourceUsage::Read | MTLResourceUsage::Write } else { MTLResourceUsage::Read };
            encoder.useResource_usage(ProtocolObject::from_ref(buf), usage);
            resident.insert(address, write || have == Some(true));
        }
        address
    }

    /// Appends the element counts of the bindings `native` reads the length of.
    fn push_lengths(native: &MetalKernel, args: &[RawArg<'_>], out: &mut Vec<u8>) {
        for &(binding, stride) in &native.array_lengths {
            let count = match args.get(binding as usize) {
                Some(RawArg::Storage { bytes, .. }) => u32::try_from(bytes / u64::from(stride.max(1))).unwrap_or(u32::MAX),
                _ => 0,
            };
            out.extend_from_slice(&count.to_ne_bytes());
        }
    }

    impl MetalBackend {
        /// Opens the default Metal device, from the process-wide pool when one is idle.
        pub fn new() -> Result<MetalBackend, BackendError> {
            // Leased at once: an early `Err` below must return the device to the pool.
            let lease = Lease(Some(acquire()?));
            let Some(shared) = lease.0.clone() else {
                return Err(BackendError::NoAdapter("no Metal GPU device available".into()));
            };
            let queue = autoreleasepool(|_| shared.device.newCommandQueue())
                .ok_or_else(|| BackendError::RequestDevice("the Metal device refused to create a command queue".into()))?;
            let batch = Batch {
                cb: None,
                compute: None,
                blit: None,
                bound: Bound::default(),
                resident: HashMap::new(),
                dispatches: 0,
                chunk: None,
                chunks: Vec::new(),
                uploads: Vec::new(),
                scratch: Vec::new(),
                profiler: None,
            };
            Ok(MetalBackend {
                info: shared.info.clone(),
                disk: Arc::clone(&shared.disk),
                cost: Arc::clone(&shared.cost),
                queue,
                done: Arc::new(Done::default()),
                submitted: AtomicU64::new(0),
                open_work: AtomicBool::new(false),
                dispatches_pending: AtomicU32::new(0),
                retired: Mutex::new(Vec::new()),
                mappable: Mutex::new(FreeUploads::new(shared.cap)),
                uploads: Mutex::new(FreeUploads::new(shared.cap)),
                free_chunks: Mutex::new(Vec::new()),
                max_buffer_bytes: shared.info.limits.max_buffer_bytes,
                batch: Mutex::new(batch),
                lease,
            })
        }

        fn device(&self) -> &ProtocolObject<dyn MTLDevice> {
            match &self.lease.0 {
                Some(shared) => &shared.device,
                // Only `Lease::drop` empties it, after the backend is gone.
                None => unreachable!("the lease of a live backend is held"),
            }
        }

        fn batch(&self) -> Result<MutexGuard<'_, Batch>, BackendError> {
            self.batch.lock().map_err(|_| BackendError::Gpu("the batch lock is poisoned".into()))
        }

        fn take_error(&self) -> Result<(), BackendError> {
            match lock(&self.done.error).take() {
                Some(message) => Err(BackendError::Gpu(message)),
                None => Ok(()),
            }
        }

        // -- Buffers --

        fn make_raw_buffer(&self, size: u64) -> Option<MetalBuffer> {
            let length = usize::try_from(size.max(MIN_BUFFER_SIZE)).ok()?;
            autoreleasepool(|_| self.device().newBufferWithLength_options(length, MTLResourceOptions::StorageModeShared)).map(MetalBuffer)
        }

        /// A new buffer, or `None` if the device is out of memory.
        fn make_buffer(&self, size: u64, usage: BufferUsage) -> Option<Buffer> {
            if size > self.max_buffer_bytes {
                return None;
            }
            self.make_raw_buffer(size).map(|native| Buffer::new(size, usage, native))
        }

        /// A new buffer; when the device is out of memory, gives back every idle buffer, then
        /// tries once more.
        fn new_buffer(&self, size: u64, usage: BufferUsage) -> Result<Buffer, BackendError> {
            if let Some(buffer) = self.make_buffer(size, usage) {
                return Ok(buffer);
            }
            self.reclaim_idle()?;
            self.make_buffer(size, usage).ok_or(BackendError::OutOfMemory { requested: size })
        }

        /// Moves retired buffers whose command buffer completed to their pools.
        fn reap(&self) {
            let done = self.done.contiguous.load(Ordering::Acquire);
            let ready = {
                let mut retired = lock(&self.retired);
                if retired.iter().all(|r| r.ready_after > done) {
                    return;
                }
                let mut ready = Vec::new();
                let mut i = 0;
                while i < retired.len() {
                    if retired[i].ready_after <= done {
                        ready.push(retired.swap_remove(i));
                    } else {
                        i += 1;
                    }
                }
                ready
            };
            let put = |pool: &Mutex<FreeUploads>, buffer: Buffer| {
                let mut pool = lock(pool);
                pool.in_flight = pool.in_flight.saturating_sub(buffer.size());
                // Over the cap it is dropped, which frees it.
                let _ = pool.put(buffer);
            };
            for retired in ready {
                match retired.item {
                    Item::Mappable(buffer) => put(&self.mappable, buffer),
                    Item::Upload(buffer) => put(&self.uploads, buffer),
                    Item::Chunk(mut chunk) => {
                        let mut free = lock(&self.free_chunks);
                        if chunk.size == CHUNK_BYTES && free.len() < MAX_FREE_CHUNKS {
                            chunk.used = 0;
                            free.push(chunk);
                        }
                    }
                }
            }
        }

        /// An idle buffer of `class` bytes from `pool`, or a new one; waits for the GPU first
        /// when more bytes than the idle cap are in flight.
        fn take_upload_buffer(&self, pool: &Mutex<FreeUploads>, class: u64, usage: BufferUsage) -> Result<Buffer, BackendError> {
            self.reap();
            if let Some(buffer) = lock(pool).take(class) {
                return Ok(buffer);
            }
            let (in_flight, cap) = {
                let pool = lock(pool);
                (pool.in_flight, pool.cap())
            };
            if in_flight > cap {
                self.sync(Duration::ZERO)?;
                self.reap();
                if let Some(buffer) = lock(pool).take(class) {
                    return Ok(buffer);
                }
            }
            self.new_buffer(class, usage)
        }

        /// A uniform chunk with room for `need` bytes.
        fn take_chunk(&self, need: u64) -> Result<Chunk, BackendError> {
            if need <= CHUNK_BYTES {
                self.reap();
                if let Some(chunk) = lock(&self.free_chunks).pop() {
                    return Ok(chunk);
                }
            }
            let size = need.max(CHUNK_BYTES);
            let buf = self.make_raw_buffer(size).ok_or(BackendError::OutOfMemory { requested: size })?;
            Ok(Chunk { buf, size, used: 0 })
        }

        /// Writes `bytes` into the batch's uniform chunks and returns the chunk's buffer and the
        /// byte offset of the value.
        fn uniform_slot(&self, batch: &mut Batch, bytes: &[u8]) -> Result<(MtlBuffer, u64), BackendError> {
            let need = (bytes.len() as u64).max(1).next_multiple_of(CHUNK_ALIGN);
            let mut chunk = match batch.chunk.take() {
                Some(chunk) if chunk.used + need <= chunk.size => chunk,
                other => {
                    if let Some(full) = other {
                        batch.chunks.push(full);
                    }
                    self.take_chunk(need)?
                }
            };
            let offset = chunk.used;
            let dst = chunk.buf.0.contents().as_ptr().cast::<u8>();
            // SAFETY: `offset + need <= size` and `bytes.len() <= need`, so the copy stays inside
            // the chunk; the GPU does not read a chunk slot before the command that uses it.
            unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), dst.add(offset as usize), bytes.len()) };
            chunk.used += need;
            let buf = chunk.buf.0.clone();
            batch.chunk = Some(chunk);
            Ok((buf, offset))
        }

        // -- Encoding --

        fn command_buffer(&self, batch: &mut Batch) -> Result<MtlCommandBuffer, BackendError> {
            if let Some(cb) = &batch.cb {
                return Ok(cb.clone());
            }
            let cb = autoreleasepool(|_| self.queue.commandBuffer()).ok_or_else(|| BackendError::Gpu("could not create a Metal command buffer".into()))?;
            batch.cb = Some(cb.clone());
            self.open_work.store(true, Ordering::Release);
            Ok(cb)
        }

        fn end_blit(batch: &mut Batch) {
            if let Some(blit) = batch.blit.take() {
                blit.endEncoding();
            }
        }

        fn end_compute(batch: &mut Batch) {
            if let Some(compute) = batch.compute.take() {
                compute.endEncoding();
            }
            batch.bound = Bound::default();
            batch.resident.clear();
        }

        /// The open compute encoder; a blit encoder in the way ends first.
        fn compute_encoder(&self, batch: &mut Batch) -> Result<MtlCompute, BackendError> {
            if let Some(compute) = &batch.compute {
                return Ok(compute.clone());
            }
            Self::end_blit(batch);
            let cb = self.command_buffer(batch)?;
            let compute = autoreleasepool(|_| cb.computeCommandEncoder()).ok_or_else(|| BackendError::Gpu("could not create a Metal compute encoder".into()))?;
            batch.compute = Some(compute.clone());
            Ok(compute)
        }

        /// The open blit encoder; a compute encoder in the way ends first, so that what is
        /// recorded here runs after the launches recorded before and before those recorded after.
        fn blit_encoder(&self, batch: &mut Batch) -> Result<MtlBlit, BackendError> {
            if let Some(blit) = &batch.blit {
                return Ok(blit.clone());
            }
            Self::end_compute(batch);
            let cb = self.command_buffer(batch)?;
            let blit = autoreleasepool(|_| cb.blitCommandEncoder()).ok_or_else(|| BackendError::Gpu("could not create a Metal blit encoder".into()))?;
            batch.blit = Some(blit.clone());
            Ok(blit)
        }

        /// Commits the pending command buffer, if there is one, and retires what it reads.
        fn commit_locked(&self, batch: &mut Batch) -> Option<MtlCommandBuffer> {
            Self::end_compute(batch);
            Self::end_blit(batch);
            batch.dispatches = 0;
            self.dispatches_pending.store(0, Ordering::Relaxed);
            let cb = batch.cb.take()?;
            let seq = self.submitted.load(Ordering::Relaxed) + 1;
            let done = Arc::clone(&self.done);
            let block = RcBlock::new(move |finished: NonNull<ProtocolObject<dyn MTLCommandBuffer>>| {
                // SAFETY: Metal passes the command buffer that completed, alive for the call.
                let finished = unsafe { finished.as_ref() };
                let error = (finished.status() == MTLCommandBufferStatus::Error).then(|| match finished.error() {
                    Some(error) => error.localizedDescription().to_string(),
                    None => "the command buffer failed without an error".to_owned(),
                });
                done.complete(seq, error);
            });
            // SAFETY: the block is a valid pointer and Metal copies it.
            unsafe { cb.addCompletedHandler(RcBlock::as_ptr(&block)) };
            // The sequence number is published before the commit and `open_work` cleared after
            // it, which `recycle_mappable` relies on to tag buffers with a number that is never
            // too small.
            self.submitted.store(seq, Ordering::Release);
            cb.commit();
            self.open_work.store(false, Ordering::Release);
            let mut retired = lock(&self.retired);
            for chunk in batch.chunks.drain(..).chain(batch.chunk.take()) {
                retired.push(Retired { ready_after: seq, item: Item::Chunk(chunk) });
            }
            for buffer in batch.uploads.drain(..) {
                retired.push(Retired { ready_after: seq, item: Item::Upload(buffer) });
            }
            Some(cb)
        }

        #[allow(clippy::too_many_arguments)]
        fn encode_launch(
            &self,
            batch: &mut Batch,
            encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
            kernel: &CompiledKernel,
            native: &MetalKernel,
            dispatch: Dispatch<'_>,
            size: [u32; 3],
            args: &[RawArg<'_>],
        ) -> Result<(), BackendError> {
            let id = kernel.raw.id();
            if batch.bound.pipeline != id {
                encoder.setComputePipelineState(&native.pipeline);
                batch.bound.pipeline = id;
            }
            let grid = [size[0], size[1], size[2], 0];
            if batch.bound.grid != Some(grid) {
                set_bytes(encoder, bytemuck::bytes_of(&grid), 0);
                batch.bound.grid = Some(grid);
            }

            match native.abi {
                MslAbi::Direct => {
                    let mut slot = 1;
                    for (_, arg) in kernel.bound(args) {
                        match arg {
                            RawArg::Storage { buffer, .. } => {
                                let native = raw(buffer)?;
                                // SAFETY: the offset 0 and the index below 31 are in range; the
                                // command buffer retains the buffer until it completes.
                                unsafe { encoder.setBuffer_offset_atIndex(Some(&*native.0), 0, slot) };
                            }
                            RawArg::Uniform(bytes) if bytes.len() <= SET_BYTES_MAX => set_bytes(encoder, bytes, slot),
                            RawArg::Uniform(bytes) => {
                                let (buf, offset) = self.uniform_slot(batch, bytes)?;
                                // SAFETY: `offset` is inside the chunk, 256-byte aligned.
                                unsafe { encoder.setBuffer_offset_atIndex(Some(&*buf), offset as usize, slot) };
                            }
                        }
                        slot += 1;
                    }
                    if !native.array_lengths.is_empty() {
                        let mut lengths = std::mem::take(&mut batch.scratch);
                        lengths.clear();
                        push_lengths(native, args, &mut lengths);
                        set_bytes(encoder, &lengths, slot);
                        batch.scratch = lengths;
                    }
                }
                MslAbi::ArgBuffer => {
                    let mut packed = std::mem::take(&mut batch.scratch);
                    packed.clear();
                    for (_, arg) in kernel.bound(args) {
                        let address = match arg {
                            RawArg::Storage { buffer, writable, .. } => use_resource(encoder, &mut batch.resident, &raw(buffer)?.0, *writable),
                            RawArg::Uniform(bytes) => {
                                let (buf, offset) = self.uniform_slot(batch, bytes)?;
                                use_resource(encoder, &mut batch.resident, &buf, false) + offset
                            }
                        };
                        packed.extend_from_slice(&address.to_ne_bytes());
                    }
                    push_lengths(native, args, &mut packed);
                    // `LumaArgs` is 8-byte aligned.
                    packed.resize(packed.len().next_multiple_of(8), 0);
                    if packed.len() <= SET_BYTES_MAX {
                        set_bytes(encoder, &packed, 1);
                    } else {
                        let (buf, offset) = self.uniform_slot(batch, &packed)?;
                        // SAFETY: `offset` is inside the chunk, 256-byte aligned.
                        unsafe { encoder.setBuffer_offset_atIndex(Some(&*buf), offset as usize, 1) };
                    }
                    batch.scratch = packed;
                }
            }

            let group = MTLSize { width: kernel.workgroup[0] as usize, height: kernel.workgroup[1] as usize, depth: kernel.workgroup[2] as usize };
            match dispatch {
                Dispatch::Direct([x, y, z]) => {
                    encoder.dispatchThreadgroups_threadsPerThreadgroup(MTLSize { width: x as usize, height: y as usize, depth: z as usize }, group);
                }
                Dispatch::Indirect { buffer, offset, .. } => {
                    let counts = raw(buffer)?;
                    // SAFETY: the gate kernels of the crate keep three `u32`s at `offset`.
                    unsafe { encoder.dispatchThreadgroupsWithIndirectBuffer_indirectBufferOffset_threadsPerThreadgroup(&counts.0, offset as usize, group) };
                }
            }
            Ok(())
        }

        // -- Waiting --

        fn spin_until(&self, target: u64, budget: Duration) -> bool {
            if self.done.reached(target) {
                return true;
            }
            if budget.is_zero() {
                return false;
            }
            let start = Instant::now();
            while start.elapsed() < budget {
                if self.done.reached(target) {
                    return true;
                }
                std::hint::spin_loop();
            }
            self.done.reached(target)
        }
    }

    impl Backend for MetalBackend {
        fn info(&self) -> &DeviceInfo {
            &self.info
        }

        fn disk_cache(&self) -> &Arc<DiskCache> {
            &self.disk
        }

        fn device_cost(&self) -> &Arc<OnceLock<CostModel>> {
            &self.cost
        }

        fn create_buffer(&self, size: u64, usage: BufferUsage, _mapped: bool) -> Option<Buffer> {
            self.make_buffer(size, usage)
        }

        /// A storage buffer holding `data` at its start, written by the host without a GPU copy;
        /// give it back with [`Backend::recycle_mappable`].
        fn alloc_filled(&self, data: &[u8], max_buffer_bytes: u64) -> Result<Buffer, BackendError> {
            let bytes = data.len() as u64;
            let class = size_class(bytes, max_buffer_bytes).ok_or(BackendError::OutOfMemory { requested: bytes })?;
            let buffer = self.take_upload_buffer(&self.mappable, class, MAPPABLE_STORAGE_USAGE)?;
            host_write(raw(&buffer)?, data);
            Ok(buffer)
        }

        fn recycle_mappable(&self, buffer: Buffer) {
            // The pending command buffer (committed as the next sequence number) may still read
            // the buffer. This must not take the batch lock: the caller may hold anything.
            let pending = self.open_work.load(Ordering::Acquire);
            let ready_after = self.submitted.load(Ordering::Acquire) + u64::from(pending);
            lock(&self.mappable).in_flight += buffer.size();
            lock(&self.retired).push(Retired { ready_after, item: Item::Mappable(buffer) });
        }

        fn compile_kernel(&self, module: &KernelModule, used: Option<&[bool]>, label: &str, trusted: bool) -> Result<RawKernel, BackendError> {
            if let Some(mask) = used
                && mask.len() != module.bindings.len()
            {
                return Err(BackendError::KernelBinding(format!("the used-binding mask has {} entries for {} bindings", mask.len(), module.bindings.len())));
            }
            let checked = if trusted { Cow::Borrowed(module) } else { luma_core::kir::opt::with_bounds_checks(module) };
            let optimized = luma_core::kir::optimize_for_target(&checked, false);
            let msl = emit_with_workgroup_init(&optimized, used, !trusted);
            let array_lengths = msl
                .array_lengths
                .iter()
                .map(|&binding| (binding, checked.bindings.get(binding as usize).and_then(|b| b.element_stride).unwrap_or(1)))
                .collect();

            let fail = |what: &str, message: String| BackendError::ShaderCompile(format!("kernel `{label}`: {what}: {message}"));
            let pipeline = autoreleasepool(|_| {
                let options = MTLCompileOptions::new();
                #[allow(deprecated)]
                options.setFastMathEnabled(!optimized.uses_exact_f32);
                let source = NSString::from_str(&msl.source);
                let library = self
                    .device()
                    .newLibraryWithSource_options_error(&source, Some(&options))
                    .map_err(|e| fail("Metal library", e.localizedDescription().to_string()))?;
                let name = NSString::from_str(&msl.entry_name);
                let function = library.newFunctionWithName(&name).ok_or_else(|| fail("entry point", format!("`{}` is missing", msl.entry_name)))?;
                self.device().newComputePipelineStateWithFunction_error(&function).map_err(|e| fail("pipeline", e.localizedDescription().to_string()))
            })?;

            let invocations: u64 = optimized.workgroup_size.iter().map(|&x| u64::from(x.max(1))).product();
            let limit = pipeline.maxTotalThreadsPerThreadgroup() as u64;
            if invocations > limit {
                return Err(BackendError::KernelWorkgroup(format!("kernel `{label}` runs {invocations} invocations per workgroup, the pipeline allows {limit}")));
            }
            Ok(RawKernel::new(MetalKernel { pipeline, abi: msl.abi, array_lengths }))
        }

        fn record_launch(&self, kernel: &CompiledKernel, dispatch: Dispatch<'_>, size: [u32; 3], args: &[RawArg<'_>]) -> Result<u32, BackendError> {
            kernel.check_count(args.len())?;
            kernel.check_kinds(args)?;
            let native = kernel
                .raw
                .downcast_ref::<MetalKernel>()
                .ok_or_else(|| BackendError::Unsupported("the kernel was compiled by another backend".into()))?;
            for (_, arg) in kernel.bound(args) {
                if let RawArg::Storage { buffer, .. } = arg {
                    raw(buffer)?;
                }
            }
            if let Dispatch::Indirect { buffer, .. } = dispatch {
                raw(buffer)?;
            }

            let mut guard = self.batch()?;
            let batch = &mut *guard;
            let profiled = batch.profiler.is_some();
            if matches!(dispatch, Dispatch::Direct(groups) if groups.contains(&0)) {
                if let Some(samples) = &mut batch.profiler {
                    samples.push((Arc::clone(&kernel.label), None));
                }
                return Ok(batch.dispatches);
            }
            if profiled {
                // The launch gets a command buffer of its own, so that its GPU time is its own.
                self.commit_locked(batch);
            }
            let encoder = self.compute_encoder(batch)?;
            self.encode_launch(batch, &encoder, kernel, native, dispatch, size, args)?;
            if profiled {
                let cb = self.commit_locked(batch);
                if let Some(samples) = &mut batch.profiler {
                    samples.push((Arc::clone(&kernel.label), cb));
                }
                return Ok(0);
            }
            batch.dispatches += 1;
            self.dispatches_pending.store(batch.dispatches, Ordering::Relaxed);
            Ok(batch.dispatches)
        }

        fn record_clear(&self, buffer: &Buffer, bytes: u64) -> Result<(), BackendError> {
            let native = raw(buffer)?;
            let bytes = bytes.min(buffer.size());
            if bytes == 0 {
                return Ok(());
            }
            let mut guard = self.batch()?;
            let blit = self.blit_encoder(&mut guard)?;
            blit.fillBuffer_range_value(&native.0, NSRange::new(0, bytes as usize), 0);
            Ok(())
        }

        fn record_copy(&self, src: &Buffer, src_offset: u64, dst: &Buffer, dst_offset: u64, bytes: u64) -> Result<(), BackendError> {
            let (src, dst) = (raw(src)?, raw(dst)?);
            if bytes == 0 {
                return Ok(());
            }
            let mut guard = self.batch()?;
            let blit = self.blit_encoder(&mut guard)?;
            // SAFETY: the callers keep the ranges inside both buffers.
            unsafe { blit.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(&src.0, src_offset as usize, &dst.0, dst_offset as usize, bytes as usize) };
            Ok(())
        }

        /// Records a copy of `data` into the start of `dst`, ordered after everything recorded so
        /// far. The bytes go through a shared staging buffer that is reused once the command
        /// buffer that copies out of it has completed.
        fn record_upload(&self, dst: &Buffer, data: &[u8], max_buffer_bytes: u64) -> Result<(), BackendError> {
            let target = raw(dst)?;
            let bytes = data.len() as u64;
            if bytes == 0 {
                return Ok(());
            }
            let class = size_class(bytes, max_buffer_bytes).ok_or(BackendError::OutOfMemory { requested: bytes })?;
            let staging = self.take_upload_buffer(&self.uploads, class, UPLOAD_USAGE)?;
            let source = raw(&staging)?;
            host_write(source, data);

            let mut guard = self.batch()?;
            let blit = self.blit_encoder(&mut guard)?;
            // SAFETY: the staging buffer holds `bytes` bytes and the caller sized `dst` for them.
            unsafe { blit.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(&source.0, 0, &target.0, 0, bytes as usize) };
            lock(&self.uploads).in_flight += staging.size();
            guard.uploads.push(staging);
            Ok(())
        }

        fn record_graph(&self, _graph: &dyn ExecGraph, _dispatches: u32) -> Result<u32, BackendError> {
            Err(BackendError::Unsupported("graph replay requires the CUDA backend".into()))
        }

        fn pending_dispatches(&self) -> u32 {
            self.dispatches_pending.load(Ordering::Relaxed)
        }

        fn gpu_busy(&self) -> bool {
            !self.done.reached(self.submitted.load(Ordering::Acquire))
        }

        fn notify_released(&self, _buffers: &[Buffer]) {
            // Nothing caches buffers by identity, and command buffers retain what they use.
        }

        fn flush_batch(&self) -> Result<(), BackendError> {
            let mut batch = self.batch()?;
            self.commit_locked(&mut batch);
            Ok(())
        }

        fn sync(&self, spin_budget: Duration) -> Result<Duration, BackendError> {
            self.flush_batch()?;
            let start = Instant::now();
            let target = self.submitted.load(Ordering::Acquire);
            if !self.spin_until(target, spin_budget) {
                self.done.wait(target);
            }
            let waited = start.elapsed();
            self.take_error()?;
            Ok(waited)
        }

        /// Copies the first `dst.len()` bytes of `src` into `dst`, after everything recorded so
        /// far. Buffers are shared memory, so once the GPU is done the bytes are read in place
        /// (also on a discrete GPU, where `Shared` storage lives in host memory).
        fn read_into(&self, src: &Buffer, dst: &mut [MaybeUninit<u8>], _spill: bool, spin_budget: Duration) -> Result<Duration, BackendError> {
            let native = raw(src)?;
            let bytes = dst.len();
            if bytes as u64 > src.size() {
                return Err(BackendError::MapRange(format!("reading {bytes} bytes of a buffer of {}", src.size())));
            }
            let waited = self.sync(spin_budget)?;
            if bytes > 0 {
                // SAFETY: the buffer holds at least `bytes` bytes and the GPU finished with it.
                let view = unsafe { std::slice::from_raw_parts(native.0.contents().as_ptr().cast::<u8>(), bytes) };
                copy_parallel(view, dst);
            }
            Ok(waited)
        }

        /// Gives back every idle buffer, after the GPU finished with them.
        fn reclaim_idle(&self) -> Result<(), BackendError> {
            self.sync(Duration::ZERO)?;
            self.reap();
            drop(lock(&self.mappable).drain());
            drop(lock(&self.uploads).drain());
            lock(&self.free_chunks).clear();
            Ok(())
        }

        fn enable_profiling(&self) -> Result<(), BackendError> {
            let mut batch = self.batch()?;
            if batch.profiler.is_none() {
                // Earlier work must not share a command buffer with the first profiled launch.
                self.commit_locked(&mut batch);
                batch.profiler = Some(Vec::new());
            }
            Ok(())
        }

        fn finish_profiling(&self, splits: Vec<FusionSplit>) -> Result<Profile, BackendError> {
            let samples = {
                let mut batch = self.batch()?;
                let Some(samples) = batch.profiler.take() else {
                    return Err(BackendError::Profiling("profiling is not enabled".into()));
                };
                self.commit_locked(&mut batch);
                samples
            };
            self.sync(Duration::ZERO)?;
            let mut timed = Vec::with_capacity(samples.len());
            for (label, cb) in samples {
                let duration = match cb {
                    None => Duration::ZERO,
                    Some(cb) => {
                        if cb.status() == MTLCommandBufferStatus::Error {
                            return Err(BackendError::Gpu(format!("profiled launch `{label}` failed")));
                        }
                        let seconds = (cb.GPUEndTime() - cb.GPUStartTime()).max(0.0);
                        if seconds.is_finite() { Duration::from_secs_f64(seconds) } else { Duration::ZERO }
                    }
                };
                timed.push((label, duration));
            }
            let mut profile = Profile::default();
            profile.add_samples(timed);
            profile.splits = splits;
            Ok(profile)
        }

        fn is_profiling(&self) -> Result<bool, BackendError> {
            Ok(self.batch()?.profiler.is_some())
        }

        fn capture_graph(&self, _launches: &[GraphLaunch<'_>]) -> Option<Arc<dyn ExecGraph>> {
            None
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn size_classes_round_up_to_powers_of_two_and_clamp() {
            assert_eq!(size_class(1, 1 << 20), Some(MIN_BUFFER_SIZE));
            assert_eq!(size_class(17, 1 << 20), Some(32));
            assert_eq!(size_class(1 << 20, 1 << 20), Some(1 << 20));
            assert_eq!(size_class((1 << 20) + 1, 1 << 20), None);
            assert_eq!(size_class(600, 1000), Some(1000 & !3));
        }

        #[test]
        fn completions_advance_only_over_a_contiguous_prefix() {
            let done = Done::default();
            done.complete(2, None);
            assert!(!done.reached(1));
            done.complete(1, None);
            assert!(done.reached(2));
            assert!(!done.reached(3));
            done.complete(3, Some("boom".into()));
            assert!(done.reached(3));
            assert_eq!(lock(&done.error).as_deref(), Some("boom"));
        }
    }
}
