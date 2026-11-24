//! [`WgpuBackend`]: buffers, kernels and the batch of commands recorded since the last
//! submission, on a wgpu device.
//!
//! Launches are recorded into one compute pass per batch; everything already bound (pipeline,
//! grid uniform, bind groups) is not bound again. Uniform values live in an arena of chunks that
//! stays untouched until the batch executed, host data goes up through reusable mapped staging
//! buffers inside the one ordered command stream, and readbacks go through pooled `MAP_READ`
//! buffers.

use std::borrow::Cow;
use std::collections::HashMap;
use std::fmt;
use std::hash::{BuildHasherDefault, Hash, Hasher};
use std::mem::MaybeUninit;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, LazyLock, Mutex, MutexGuard, OnceLock, mpsc};
use std::time::{Duration, Instant};

use luma_core::kir::KernelModule;
use luma_core::{
    Backend, BackendError, BackendKind, BindingInfo, BindingKind, Buffer, BufferUsage, COPY_BUFFER_ALIGNMENT, CompiledKernel, CostModel, DeviceInfo,
    DeviceKind, DeviceLimits, DiskCache, Dispatch, ExecGraph, Features, FreeUploads, FusionSplit, GraphLaunch, KernelTrust, NativeBuffer, NativeKernel,
    Profile, RawArg, RawKernel, TextureCaps, TextureDesc, TextureFilter, TextureFormat, TuningProfile,
};
use luma_core::hal::{check_transfer, copy_parallel};
use luma_core::{Stage, StageDir, Ticket};
use wgpu::{BindGroupEntry, BindingResource, BufferBinding};

use crate::codegen;

/// `LUMA_WGPU_DUMP`: a directory that receives the WGSL (`<label>-<hash>.wgsl`) of every kernel
/// the backend generates from luma's kernel IR.
static DUMP_DIR: LazyLock<Option<std::path::PathBuf>> = LazyLock::new(|| std::env::var_os("LUMA_WGPU_DUMP").filter(|v| !v.is_empty()).map(Into::into));

use crate::device::{self, Gpu, lock};
use crate::profiler::Profiler;

// -- Usages --

/// Usage of every storage buffer of luma's arrays.
const STORAGE_USAGE: BufferUsage = BufferUsage::STORAGE.union(BufferUsage::COPY_DST).union(BufferUsage::COPY_SRC);

/// Usage of the buffers uniform values are written into.
const UNIFORM_USAGE: BufferUsage = BufferUsage::UNIFORM.union(BufferUsage::COPY_DST);

/// Usage of storage buffers the host writes directly (zero-copy uploads).
const MAPPABLE_STORAGE_USAGE: BufferUsage = STORAGE_USAGE.union(BufferUsage::MAP_WRITE);

/// Usage of readback buffers.
const STAGING_USAGE: BufferUsage = BufferUsage::MAP_READ.union(BufferUsage::COPY_DST);

/// Usage of upload staging buffers.
const UPLOAD_USAGE: BufferUsage = BufferUsage::MAP_WRITE.union(BufferUsage::COPY_SRC);

fn wgpu_usage(usage: BufferUsage) -> wgpu::BufferUsages {
    let mut out = wgpu::BufferUsages::empty();
    for (flag, native) in [
        (BufferUsage::STORAGE, wgpu::BufferUsages::STORAGE),
        (BufferUsage::COPY_SRC, wgpu::BufferUsages::COPY_SRC),
        (BufferUsage::COPY_DST, wgpu::BufferUsages::COPY_DST),
        (BufferUsage::UNIFORM, wgpu::BufferUsages::UNIFORM),
        (BufferUsage::INDIRECT, wgpu::BufferUsages::INDIRECT),
        (BufferUsage::MAP_READ, wgpu::BufferUsages::MAP_READ),
        (BufferUsage::MAP_WRITE, wgpu::BufferUsages::MAP_WRITE),
    ] {
        if usage.contains(flag) {
            out |= native;
        }
    }
    out
}

// -- Native handles --

/// A wgpu buffer held by a [`Buffer`].
#[derive(Debug)]
struct WgpuBuffer(wgpu::Buffer);

/// The submission a transfer or barrier ticket waits for.
#[derive(Debug)]
struct WgpuTicket(wgpu::SubmissionIndex);

impl NativeBuffer for WgpuBuffer {}

/// A texture, its view and the sampler of its filter, held by a [`Buffer`] made by
/// [`Backend::create_texture`].
#[derive(Debug)]
struct WgpuTexture {
    /// Kept alive by the view; named for what it is.
    _texture: wgpu::Texture,
    view: wgpu::TextureView,
    sampler: wgpu::Sampler,
}

impl NativeBuffer for WgpuTexture {}

/// A compiled pipeline held by a [`RawKernel`].
#[derive(Debug)]
struct WgpuKernel {
    pipeline: wgpu::ComputePipeline,
    /// Layout of `@group(1)`.
    args_layout: wgpu::BindGroupLayout,
    /// Whether an argument is a texture or a sampler: such a launch builds its bind group anew
    /// (every texture is a new object, so the bind group cache could not hit).
    textured: bool,
}

impl NativeKernel for WgpuKernel {}

fn wrap(buffer: wgpu::Buffer, size: u64, usage: BufferUsage) -> Buffer {
    Buffer::new(size, usage, WgpuBuffer(buffer))
}

fn texture(buffer: &Buffer) -> Result<&WgpuTexture, BackendError> {
    buffer.downcast_ref::<WgpuTexture>().ok_or_else(|| BackendError::KernelBinding("a texture binding was given something that is not a wgpu texture".into()))
}

fn raw(buffer: &Buffer) -> Result<&wgpu::Buffer, BackendError> {
    buffer.downcast_ref::<WgpuBuffer>().map(|native| &native.0).ok_or_else(|| BackendError::Unsupported("the buffer belongs to another backend".into()))
}

/// The wgpu buffer of one of this backend's own pooled buffers.
fn own(buffer: &Buffer) -> &wgpu::Buffer {
    match raw(buffer) {
        Ok(native) => native,
        Err(_) => panic!("a buffer of another backend in the pools of the wgpu backend"),
    }
}

// -- Fast hashing of cache keys --

/// A fast, non-cryptographic hasher (the multiply-rotate scheme of rustc's `FxHasher`) for the
/// bind group cache, whose keys are small integers and buffer handles. The default `SipHash`
/// costs several times as much per lookup, which adds up on the launch path.
#[derive(Default, Clone, Copy)]
struct FxHasher(u64);

const SEED: u64 = 0x51_7c_c1_b7_27_22_0a_95;

impl FxHasher {
    #[inline]
    fn add(&mut self, word: u64) {
        self.0 = (self.0.rotate_left(5) ^ word).wrapping_mul(SEED);
    }
}

impl Hasher for FxHasher {
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        for chunk in bytes.chunks(8) {
            let word = chunk.iter().rev().fold(0u64, |word, &byte| (word << 8) | u64::from(byte));
            self.add(word);
        }
    }

    #[inline]
    fn write_u8(&mut self, i: u8) {
        self.add(u64::from(i));
    }

    #[inline]
    fn write_u32(&mut self, i: u32) {
        self.add(u64::from(i));
    }

    #[inline]
    fn write_u64(&mut self, i: u64) {
        self.add(i);
    }

    #[inline]
    fn write_usize(&mut self, i: usize) {
        self.add(i as u64);
    }

    /// The multiplication leaves the best-mixed bits at the top; hash tables index with the low
    /// ones, so rotate them down.
    #[inline]
    fn finish(&self) -> u64 {
        self.0.rotate_left(26)
    }
}

type FxMap<K, V> = HashMap<K, V, BuildHasherDefault<FxHasher>>;

// -- Sizes --

/// Smallest buffer the pools hand out.
const MIN_BUFFER_SIZE: u64 = 16;

/// Size class for a buffer holding `bytes` bytes: the next power of two (at least
/// [`MIN_BUFFER_SIZE`]), clamped to `max_size`. `None` if `bytes` itself exceeds `max_size`.
fn size_class(bytes: u64, max_size: u64) -> Option<u64> {
    // Keep copy sizes aligned even when the limit is not.
    let max_size = max_size & !(COPY_BUFFER_ALIGNMENT - 1);
    if bytes > max_size {
        return None;
    }
    let class = bytes.max(MIN_BUFFER_SIZE).checked_next_power_of_two().unwrap_or(max_size);
    Some(class.min(max_size))
}

/// How an idle staging buffer is matched to a request.
#[derive(Clone, Copy)]
enum Reuse {
    /// Only a buffer of exactly the requested size class: for buffers that outlive the transfer as
    /// storage, whose users see [`Buffer::size`].
    Exact,
    /// The smallest idle buffer that is at least as large: for staging buffers, which every user
    /// copies, maps and reads through an explicit byte range of the requested length, never
    /// through the buffer's own size.
    Fit,
}

impl Reuse {
    fn take(self, pool: &mut FreeUploads, class: u64) -> Option<Buffer> {
        match self {
            Reuse::Exact => pool.take(class),
            Reuse::Fit => pool.take_fit(class),
        }
    }
}

/// Readbacks larger than twice this go in pieces of this size, see [`WgpuBackend::read_into`].
const READ_CHUNK: u64 = 32 << 20;

/// Copies below this size stay on the calling thread.
const PARALLEL_COPY_MIN: usize = 8 << 20;

/// Size of the `luma` grid uniform in bytes (`LumaGrid`).
const GRID_BYTES: u64 = 16;

/// Bind groups cached per backend before the cache is emptied (at the next flush).
const MAX_CACHED_BIND_GROUPS: usize = 2048;

/// Problem sizes whose grid uniform slot is remembered within a batch, so that launches with a
/// size seen before share its slot (and its bind group) instead of writing a new one.
const MAX_GRID_SLOTS: usize = 16;

/// Size of one uniform arena chunk. Uniform values of a batch are packed into it, one
/// `min_uniform_buffer_offset_alignment` slot each.
const ARENA_CHUNK: u64 = 256 * 1024;

/// Idle uniform chunks kept for reuse (a few MiB at most); more are released, as Metal does.
const MAX_FREE_CHUNKS: usize = 16;

/// Bytes at the start of the batch's first uniform chunk that are copied by the batch's own
/// command buffer, as its first command. The slots of a handful of launches fit: the batch then
/// needs no command buffer of its own for uniforms (a second encoder, its `finish` and a second
/// command buffer in the submission cost the host about 2.5 us and delay the GPU by about 2 us
/// per submission). Slots beyond it go through the separate command buffer.
const UNIFORM_PREFIX: u64 = 4096;

fn dynamic_offset(offset: u64) -> Result<u32, BackendError> {
    u32::try_from(offset).map_err(|_| BackendError::Gpu("uniform arena offset exceeds u32".into()))
}

/// Bytes a uniform value of `len` bytes occupies in the arena.
fn slot_size(align: u64, len: usize) -> u64 {
    (len as u64).next_multiple_of(COPY_BUFFER_ALIGNMENT).max(4).next_multiple_of(align)
}

// -- Writes into mapped memory --

/// Copies `src` into write-only mapped memory, split over several threads for large copies: one
/// core cannot saturate unified memory, and the page faults of a fresh destination spread out
/// too.
fn write_parallel(src: &[u8], mut dst: wgpu::WriteOnly<'_, [u8]>) {
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get().min(8));
    if src.len() < PARALLEL_COPY_MIN || threads == 1 {
        dst.copy_from_slice(src);
        return;
    }
    /// A disjoint part of the mapping, handed to one thread.
    struct Part(std::ptr::NonNull<[u8]>);
    // SAFETY: parts are disjoint and outlive the scope the threads run in.
    unsafe impl Send for Part {}

    assert_eq!(src.len(), dst.len(), "source and destination lengths differ");
    let base = dst.as_raw_ptr().cast::<u8>();
    let chunk = src.len().div_ceil(threads).next_multiple_of(4096);
    std::thread::scope(|scope| {
        for (i, s) in src.chunks(chunk).enumerate() {
            // SAFETY: `i * chunk + s.len() <= dst.len()`, so the part lies inside the mapping.
            let part = Part(std::ptr::NonNull::slice_from_raw_parts(unsafe { base.add(i * chunk) }, s.len()));
            scope.spawn(move || {
                let part = part;
                // SAFETY: the part is mapped for writing and no other thread touches it.
                unsafe { wgpu::WriteOnly::new(part.0) }.copy_from_slice(s);
            });
        }
    });
}

/// The outcome of a `map_async` request, see [`map_for_read`].
type MapReceiver = mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>;

/// Requests a read mapping of `slice`; [`WgpuBackend::wait_map`] waits for it.
fn map_for_read(slice: wgpu::BufferSlice<'_>) -> MapReceiver {
    let (sender, receiver) = mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    receiver
}

// -- Uniform arena --

/// One buffer of the uniform arena together with the bytes that will be written into it.
#[derive(Debug)]
struct Chunk {
    buffer: Buffer,
    /// The arena epoch this chunk was started in.
    epoch: u64,
    /// Host copy of the slots handed out so far (`data.len()` is the used size).
    data: Vec<u8>,
    /// Bind group of `@group(0)` for this chunk (the grid uniform at a dynamic offset).
    grid_group: Option<wgpu::BindGroup>,
}

/// Uniform values of the pending batch. Every value gets its own slot in a chunk that stays
/// untouched until the batch executed, so each launch sees the value it was given even though
/// the data is copied into the chunk (from a mapped staging buffer, once per chunk when the
/// batch is submitted) before the batch's commands run: the copies are recorded into a command
/// buffer of their own that goes first in the same submission. Slots are only reused after the
/// batch holding them was submitted; the copy into a reused slot is ordered after that
/// submission on the queue.
#[derive(Debug)]
struct Arena {
    /// The chunk slots are currently handed out from.
    current: Chunk,
    /// Chunks of this batch that filled up; dropped when the batch is submitted.
    full: Vec<Chunk>,
    /// Counts the chunks that were started; identifies `current` in the bound-state tracking.
    epoch: u64,
}

fn new_chunk(device: &wgpu::Device, size: u64, epoch: u64) -> Chunk {
    let buffer = device.create_buffer(&wgpu::BufferDescriptor { label: Some("luma uniforms"), size, usage: wgpu_usage(UNIFORM_USAGE), mapped_at_creation: false });
    Chunk { epoch, buffer: wrap(buffer, size, UNIFORM_USAGE), data: Vec::new(), grid_group: None }
}

impl Arena {
    fn new(device: &wgpu::Device) -> Self {
        Arena { current: new_chunk(device, ARENA_CHUNK, 0), full: Vec::new(), epoch: 0 }
    }

    /// `true` if `bytes` more bytes of slots fit into the current chunk.
    fn has_room(&self, bytes: u64) -> bool {
        self.current.data.len() as u64 + bytes <= self.current.buffer.size()
    }

    /// Starts the chunk `fresh` (a recycled one or a new one, see [`WgpuBackend::take_chunk`]);
    /// the old one is kept until the batch is submitted.
    fn roll(&mut self, mut fresh: Chunk) {
        self.epoch += 1;
        fresh.epoch = self.epoch;
        self.full.push(std::mem::replace(&mut self.current, fresh));
    }

    /// Reserves a fresh slot holding `data` in the current chunk and returns its offset; the
    /// caller has made room for it ([`Arena::has_room`]). The bytes reach the GPU in
    /// [`Arena::upload`].
    fn push(&mut self, align: u64, data: &[u8]) -> u64 {
        let slot = slot_size(align, data.len());
        let chunk = &mut self.current;
        let offset = chunk.data.len() as u64;
        chunk.data.extend_from_slice(data);
        chunk.data.resize((offset + slot) as usize, 0);
        offset
    }

    /// Copies the slots of every chunk to the GPU and empties the arena for the next batch,
    /// keeping the current chunk. Must run right before the batch is submitted: the returned
    /// commands (the copies out of the staging buffers) have to be submitted ahead of the
    /// batch's own, and so run after everything submitted earlier. The chunks that filled up
    /// move to `retired`; they are recycled once the submission that reads them completed, see
    /// [`WgpuBackend::retire_chunks`].
    ///
    /// The first [`UNIFORM_PREFIX`] bytes of the chunk the batch started in are not part of
    /// them: `prefix` is the staging buffer the batch's own commands copy them out of (see
    /// [`WgpuBackend::copy_prefix`]); they are written into it here.
    ///
    /// The staging buffers are the reusable mapped ones of uploads (pushed to `staged`, which the
    /// caller maps again after the submission), not `Queue::write_buffer`: that one has wgpu
    /// create a staging buffer per call, which allocates and maps a fresh block of host-visible
    /// memory whenever nothing else lives in it, and frees it again at the next poll, on every
    /// submission.
    fn upload(&mut self, be: &WgpuBackend, prefix: Option<Prefix>, staged: &mut Vec<Buffer>, retired: &mut Vec<Chunk>) -> Option<wgpu::CommandBuffer> {
        let mut encoder: Option<wgpu::CommandEncoder> = None;
        // The staging buffer must be unmapped for the submission even if no slot went into it.
        let mut prefix = prefix.map(|prefix| (prefix, false));
        for chunk in self.full.iter().chain(std::iter::once(&self.current)) {
            if chunk.data.is_empty() {
                continue;
            }
            let mut from = 0;
            if let Some((prefix, written)) = prefix.as_mut().filter(|(prefix, _)| prefix.epoch == chunk.epoch) {
                from = chunk.data.len().min(UNIFORM_PREFIX as usize);
                be.fill(&prefix.staging, &chunk.data[..from]);
                *written = true;
            }
            let rest = &chunk.data[from..];
            if rest.is_empty() {
                continue;
            }
            match be.stage(rest) {
                Some(staging) => {
                    encoder
                        .get_or_insert_with(|| be.device().create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("luma uniforms") }))
                        .copy_buffer_to_buffer(own(&staging), 0, own(&chunk.buffer), from as u64, rest.len() as u64);
                    staged.push(staging);
                }
                // No staging buffer to be had (the device is out of memory): `write_buffer`
                // brings its own and is ordered the same way.
                None => be.queue().write_buffer(own(&chunk.buffer), from as u64, rest),
            }
        }
        if let Some((prefix, written)) = prefix {
            if !written {
                be.fill(&prefix.staging, &[]);
            }
            staged.push(prefix.staging);
        }
        retired.append(&mut self.full);
        self.current.data.clear();
        encoder.map(wgpu::CommandEncoder::finish)
    }
}

// -- The batch --

/// A bind group of `@group(1)` and what it was created for.
#[derive(Debug)]
struct CachedGroup {
    kernel: u64,
    /// What is bound, in binding order. Uniforms are identified by their arena chunk (the slot
    /// is a dynamic offset). Holding the buffers keeps them alive as long as the group is cached,
    /// so a buffer's identity cannot be reused by another one.
    entries: Box<[(Buffer, u64, u64)]>,
    group: wgpu::BindGroup,
}

impl CachedGroup {
    fn matches<'a>(&self, kernel: u64, mut entries: impl Iterator<Item = (&'a Buffer, u64, u64)>) -> bool {
        self.kernel == kernel
            && self.entries.iter().all(|(a, a_offset, a_bytes)| entries.next().is_some_and(|(b, b_offset, b_bytes)| a == b && *a_offset == b_offset && *a_bytes == b_bytes))
            && entries.next().is_none()
    }
}

/// Bind groups of `@group(1)` by (kernel, bound buffers). Looked up with a hash computed from
/// borrowed handles, so a hit allocates and clones nothing.
#[derive(Debug, Default)]
struct BindGroupCache {
    buckets: FxMap<u64, Vec<CachedGroup>>,
    len: usize,
}

impl BindGroupCache {
    /// Drops the groups binding any of `buffers`, so that the cache does not keep them alive.
    /// Invalidates indices into buckets: only call it while nothing is bound.
    fn forget(&mut self, buffers: &[Buffer]) {
        self.buckets.retain(|_, bucket| {
            bucket.retain(|group| !group.entries.iter().any(|(b, _, _)| buffers.contains(b)));
            !bucket.is_empty()
        });
        self.len = self.buckets.values().map(Vec::len).sum();
    }
}

/// What the open compute pass has bound; launches skip what is already in place.
#[derive(Debug, Default, Clone, Copy)]
struct Bound {
    /// The kernel whose pipeline is set.
    kernel: Option<u64>,
    /// The grid uniform at `@group(0)`: arena chunk epoch and slot offset.
    grid: Option<(u64, u32)>,
    /// The bind group at `@group(1)`: bucket hash and index in it.
    args: Option<(u64, usize)>,
}

/// The staging buffer the first [`UNIFORM_PREFIX`] bytes of a chunk are copied out of ahead of
/// the batch's first dispatch, and the chunk's arena epoch.
#[derive(Debug)]
struct Prefix {
    staging: Buffer,
    epoch: u64,
}

/// The commands recorded since the last submission.
#[derive(Debug)]
struct Batch {
    /// The compute pass consecutive launches are recorded into (one pass per batch instead of
    /// one per launch). Commands that need the encoder itself end it first. Declared before the
    /// encoder so that it is dropped (ended) first.
    pass: Option<wgpu::ComputePass<'static>>,
    encoder: Option<wgpu::CommandEncoder>,
    /// Present once the batch has a dispatch and a staging buffer was available for [`Prefix`].
    prefix: Option<Prefix>,
    bound: Bound,
    dispatches: u32,
    arena: Arena,
    /// Slots of the grid uniform written to `arena.current` in this batch, by problem size.
    grid_slots: Vec<([u32; 3], u32)>,
    bind_groups: BindGroupCache,
    /// Dynamic offsets of the launch being recorded (kept to reuse its allocation).
    offsets: Vec<u32>,
    /// Upload staging buffers copied from in this batch; mapped again once it is submitted.
    uploads: Vec<Buffer>,
    /// Present while profiling is enabled.
    profiler: Option<Profiler>,
}

/// What a kernel may use of the shader features the device was opened with.
fn shader_capabilities(features: wgpu::Features) -> naga::valid::Capabilities {
    use naga::valid::Capabilities as Caps;
    let mut caps = Caps::empty();
    if features.contains(wgpu::Features::SUBGROUP) {
        caps |= Caps::SUBGROUP | Caps::SUBGROUP_BARRIER;
    }
    if features.contains(wgpu::Features::SHADER_FLOAT32_ATOMIC) {
        caps |= Caps::SHADER_FLOAT32_ATOMIC;
    }
    if features.contains(wgpu::Features::SHADER_F16) {
        caps |= Caps::SHADER_FLOAT16;
    }
    if features.contains(wgpu::Features::SHADER_F64) {
        caps |= Caps::FLOAT64;
    }
    if features.contains(wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX) {
        caps |= Caps::COOPERATIVE_MATRIX;
    }
    caps
}

fn luma_features(features: wgpu::Features) -> Features {
    let mut out = Features::NONE;
    for (native, ours) in [
        (wgpu::Features::TIMESTAMP_QUERY, Features::TIMESTAMP_QUERY),
        (wgpu::Features::SUBGROUP, Features::SUBGROUP),
        (wgpu::Features::SHADER_F16, Features::SHADER_F16),
        (wgpu::Features::SHADER_F64, Features::SHADER_F64),
        (wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX, Features::EXPERIMENTAL_COOPERATIVE_MATRIX),
    ] {
        if features.contains(native) {
            out |= ours;
        }
    }
    out
}

const APPLE_VENDOR: u32 = 0x106B;
const NVIDIA_VENDOR: u32 = 0x10DE;

/// The family of hardware `info` describes, unless the environment forces one (`LUMA_TUNING`).
fn tuning_profile(info: &wgpu::AdapterInfo) -> TuningProfile {
    match std::env::var("LUMA_TUNING").ok().as_deref().map(str::trim) {
        Some("apple") => TuningProfile::Apple,
        Some("nvidia") => TuningProfile::Nvidia,
        Some("generic") => TuningProfile::Generic,
        Some("cpu") => TuningProfile::Cpu,
        _ if info.device_type == wgpu::DeviceType::Cpu => TuningProfile::Cpu,
        _ if info.backend == wgpu::Backend::Metal && (info.vendor == APPLE_VENDOR || info.name.starts_with("Apple")) => TuningProfile::Apple,
        _ if info.vendor == NVIDIA_VENDOR => TuningProfile::Nvidia,
        _ => TuningProfile::Generic,
    }
}

/// A [`Gpu`] borrowed from the process-wide pool; returned to it when dropped. The device is
/// never destroyed, see [`device`].
#[derive(Debug)]
struct Lease(Gpu);

impl Drop for Lease {
    fn drop(&mut self) {
        device::release(self.0.clone());
    }
}

/// The wgpu backend.
pub struct WgpuBackend {
    info: DeviceInfo,
    disk: Arc<DiskCache>,
    cost: Arc<OnceLock<CostModel>>,
    /// Mapped upload staging buffers, refilled by the map callbacks of submitted ones.
    uploads: Arc<Mutex<FreeUploads>>,
    /// Recycled mappable storage buffers (unified memory), mapped again for zero-copy uploads.
    mappable: Arc<Mutex<FreeUploads>>,
    /// Idle readback buffers.
    readbacks: Mutex<FreeUploads>,
    /// Buffers their owners released; the cached bind groups still holding them are dropped at
    /// the next submission, which frees them.
    released: Arc<Mutex<Vec<Buffer>>>,
    /// Uniform arena chunks whose last submission completed, ready for the next roll.
    free_chunks: Arc<Mutex<Vec<Chunk>>>,
    /// Recycled mappable storage buffers waiting for the next submission before they are mapped.
    unmapped: Mutex<Vec<Buffer>>,
    /// The first error the GPU reported outside of a call that could return it.
    gpu_error: Arc<Mutex<Option<String>>>,
    /// `Batch::dispatches`, kept where threads that only decide whether to submit can read it
    /// without the lock of the batch.
    dispatches_pending: AtomicU32,
    /// Layout of `@group(0)`: the grid uniform, at a dynamic offset into the arena.
    grid_layout: wgpu::BindGroupLayout,
    timestamps: bool,
    uniform_align: u64,
    storage_offset_align: u64,
    max_dynamic_uniforms: u32,
    /// Largest buffer that can be allocated *and* bound as a storage buffer.
    max_buffer_bytes: u64,
    batch: Mutex<Batch>,
    /// The leased GPU device. Must stay the last field: fields drop in declaration order, and the
    /// lease hands the device to the next backend, which must only happen once everything of
    /// this one (open pass and encoder, buffer pools, bind groups) is gone.
    gpu: Lease,
}

impl fmt::Debug for WgpuBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("WgpuBackend").field("adapter", &self.gpu.0.adapter.name).field("cpu", &self.info.cpu).finish_non_exhaustive()
    }
}

impl WgpuBackend {
    /// Opens a device of `kind`, from the process-wide pool when one is idle.
    pub fn new(kind: DeviceKind) -> Result<WgpuBackend, BackendError> {
        // Leased at once: an early `Err` below must return the device to the pool, not destroy it.
        let gpu = Lease(device::acquire(kind)?);
        let device = gpu.0.device.clone();
        let limits = device.limits();
        let features = device.features();
        let max_buffer_bytes = limits.max_buffer_size.min(limits.max_storage_buffer_binding_size) & !3;
        // Idle buffer caches scale with the memory the device holds (half of the 80% data
        // budget): they are given back when the device runs out of memory, so they cannot crowd
        // out array data.
        let cap = gpu.0.memory.map_or(1 << 30, |memory| memory / 5 * 4 / 2);

        // wgpu panics on errors nobody handles. Record the first one instead and report it as an
        // `Err` at the next synchronization point.
        let gpu_error = Arc::new(Mutex::new(None));
        let sink = Arc::clone(&gpu_error);
        // Registered for every backend: it replaces the previous owner's handler, so a reused
        // device reports errors to its current backend only.
        device.on_uncaptured_error(Arc::new(move |error: wgpu::Error| {
            lock(&sink).get_or_insert_with(|| error.to_string());
        }));

        let grid_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("luma grid"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: true, min_binding_size: None },
                count: None,
            }],
        });

        let adapter = &gpu.0.adapter;
        let subgroup = features.contains(wgpu::Features::SUBGROUP);
        let info = DeviceInfo {
            backend: BackendKind::Wgpu,
            name: adapter.name.clone(),
            cpu: false,
            zero_copy: features.contains(wgpu::Features::MAPPABLE_PRIMARY_BUFFERS),
            profile: tuning_profile(adapter),
            features: luma_features(features),
            subgroup_size: subgroup.then_some((adapter.subgroup_min_size, adapter.subgroup_max_size)),
            memory_bytes: gpu.0.memory,
            ieee_f32_intrinsics: false,
            hardware_fma: None,
            limits: DeviceLimits {
                max_buffer_bytes,
                max_storage_buffers: limits.max_storage_buffers_per_shader_stage,
                max_uniform_bytes: limits.max_uniform_buffer_binding_size,
                uniform_align: u64::from(limits.min_uniform_buffer_offset_alignment).max(4),
                storage_offset_align: u64::from(limits.min_storage_buffer_offset_alignment).max(4),
                max_dynamic_uniforms: limits.max_dynamic_uniform_buffers_per_pipeline_layout,
                max_workgroups_per_dim: limits.max_compute_workgroups_per_dimension,
                max_workgroup_size: [limits.max_compute_workgroup_size_x, limits.max_compute_workgroup_size_y, limits.max_compute_workgroup_size_z],
                max_workgroup_invocations: limits.max_compute_invocations_per_workgroup,
                max_workgroup_storage_bytes: limits.max_compute_workgroup_storage_size,
            },
            compute_units: gpu.0.compute_units,
            l2_cache_bytes: None,
        };

        let batch = Batch {
            pass: None,
            encoder: None,
            prefix: None,
            bound: Bound::default(),
            dispatches: 0,
            arena: Arena::new(&device),
            grid_slots: Vec::new(),
            bind_groups: BindGroupCache::default(),
            offsets: Vec::new(),
            uploads: Vec::new(),
            profiler: None,
        };

        Ok(WgpuBackend {
            disk: Arc::clone(&gpu.0.disk),
            cost: Arc::clone(&gpu.0.cost),
            uploads: Arc::new(Mutex::new(FreeUploads::new(cap))),
            mappable: Arc::new(Mutex::new(FreeUploads::new(cap))),
            readbacks: Mutex::new(FreeUploads::new(cap)),
            released: Arc::new(Mutex::new(Vec::new())),
            free_chunks: Arc::new(Mutex::new(Vec::new())),
            unmapped: Mutex::new(Vec::new()),
            gpu_error,
            dispatches_pending: AtomicU32::new(0),
            grid_layout,
            timestamps: features.contains(wgpu::Features::TIMESTAMP_QUERY),
            uniform_align: info.limits.uniform_align,
            storage_offset_align: info.limits.storage_offset_align,
            max_dynamic_uniforms: limits.max_dynamic_uniform_buffers_per_pipeline_layout,
            max_buffer_bytes,
            info,
            batch: Mutex::new(batch),
            gpu,
        })
    }

    fn device(&self) -> &wgpu::Device {
        &self.gpu.0.device
    }

    fn queue(&self) -> &wgpu::Queue {
        &self.gpu.0.queue
    }

    fn batch(&self) -> Result<MutexGuard<'_, Batch>, BackendError> {
        self.batch.lock().map_err(|_| BackendError::Gpu("the batch lock is poisoned".into()))
    }

    fn take_gpu_error(&self) -> Result<(), BackendError> {
        match lock(&self.gpu_error).take() {
            Some(message) => Err(BackendError::Gpu(message)),
            None => Ok(()),
        }
    }

    fn new_encoder(&self) -> wgpu::CommandEncoder {
        self.device().create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("luma batch") })
    }

    /// The batch's command encoder, for commands that are not dispatches. The open compute pass
    /// (which borrows the encoder) ends first, so the commands recorded here run after the
    /// launches recorded before and before those recorded after.
    fn encoder<'a>(&self, batch: &'a mut Batch) -> &'a mut wgpu::CommandEncoder {
        batch.pass = None;
        batch.bound = Bound::default();
        batch.encoder.get_or_insert_with(|| self.new_encoder())
    }

    // -- Buffers --

    /// A new buffer, or `None` if the device is out of memory.
    fn make_buffer(&self, size: u64, usage: BufferUsage, mapped: bool) -> Option<Buffer> {
        let scope = self.device().push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        let buffer = self.device().create_buffer(&wgpu::BufferDescriptor { label: Some("luma buffer"), size, usage: wgpu_usage(usage), mapped_at_creation: mapped });
        pollster::block_on(scope.pop()).is_none().then(|| wrap(buffer, size, usage))
    }

    /// A new buffer; when the device is out of memory, gives back every idle buffer and the bind
    /// groups pinning them, then tries once more.
    fn new_buffer(&self, size: u64, usage: BufferUsage, mapped: bool) -> Result<Buffer, BackendError> {
        if let Some(buffer) = self.make_buffer(size, usage, mapped) {
            return Ok(buffer);
        }
        self.reclaim_idle()?;
        self.make_buffer(size, usage, mapped).ok_or(BackendError::OutOfMemory { requested: size })
    }

    /// An idle mapped buffer from `pool` for `class` bytes, matched as `reuse` says. Polls
    /// completed submissions first, which hands back the buffers whose `map_async` callback is
    /// due. Never submits or waits.
    fn take_idle(&self, pool: &Mutex<FreeUploads>, class: u64, reuse: Reuse) -> Option<Buffer> {
        if let Some(buffer) = reuse.take(&mut lock(pool), class) {
            return Some(buffer);
        }
        let _ = self.device().poll(wgpu::PollType::Poll);
        reuse.take(&mut lock(pool), class)
    }

    /// Takes a mapped `MAP_WRITE` buffer for `class` bytes from `pool` (matched as `reuse` says),
    /// polling completed submissions (and waiting if more bytes than the idle cap are in flight)
    /// before allocating a fresh one.
    fn take_upload_buffer(&self, pool: &Mutex<FreeUploads>, class: u64, usage: BufferUsage, reuse: Reuse) -> Result<Buffer, BackendError> {
        if let Some(buffer) = self.take_idle(pool, class, reuse) {
            return Ok(buffer);
        }
        let (in_flight, cap) = {
            let uploads = lock(pool);
            (uploads.in_flight, uploads.cap())
        };
        if in_flight > cap {
            self.flush_locked(&mut *self.batch()?);
            self.device().poll(wgpu::PollType::wait_indefinitely()).map_err(|e| BackendError::Poll(e.to_string()))?;
            if let Some(buffer) = reuse.take(&mut lock(pool), class) {
                return Ok(buffer);
            }
        }
        self.new_buffer(class, usage, true)
    }

    /// A uniform chunk for the arena: an idle one whose last submission completed (so it is no
    /// longer read by the GPU, and its bind groups are still cached), or a new one. A chunk
    /// needing more than [`ARENA_CHUNK`] bytes is always new and never recycled.
    fn take_chunk(&self, bytes: u64) -> Chunk {
        if bytes > ARENA_CHUNK {
            return new_chunk(self.device(), bytes, 0);
        }
        // Completed submissions hand their chunks back from their callbacks, run by a poll.
        let mut idle = lock(&self.free_chunks).pop();
        if idle.is_none() {
            let _ = self.device().poll(wgpu::PollType::Poll);
            idle = lock(&self.free_chunks).pop();
        }
        idle.unwrap_or_else(|| new_chunk(self.device(), ARENA_CHUNK, 0))
    }

    /// Hands the arena chunks `chunks`, read by the submission just made, back for reuse once the
    /// GPU completed it: up to [`MAX_FREE_CHUNKS`] are kept, the rest are released. Without this
    /// every batch with more uniform data than one chunk holds created a buffer per extra chunk
    /// and dropped it again at its submission.
    ///
    /// Reuse waits for completion, as the staging pools do (their `map_async` callbacks), rather
    /// than relying on queue order alone: a chunk handed out while a pending submission still
    /// reads it would only be safe through the ordering of the queue, which nothing here checks.
    fn retire_chunks(&self, chunks: Vec<Chunk>) {
        let free = Arc::clone(&self.free_chunks);
        let released = Arc::clone(&self.released);
        self.queue().on_submitted_work_done(move || {
            let mut dropped = Vec::new();
            {
                let mut free = lock(&free);
                for mut chunk in chunks {
                    if chunk.buffer.size() == ARENA_CHUNK && free.len() < MAX_FREE_CHUNKS {
                        chunk.data.clear();
                        free.push(chunk);
                    } else {
                        dropped.push(chunk.buffer);
                    }
                }
            }
            if !dropped.is_empty() {
                lock(&released).extend(dropped);
            }
        });
    }

    /// An upload staging buffer for `bytes` bytes, still mapped: the smallest idle one that is
    /// large enough, or a new one of the size class. Only the first `bytes` bytes may be used.
    fn mapped_staging(&self, bytes: u64) -> Option<Buffer> {
        let class = size_class(bytes, self.max_buffer_bytes)?;
        match self.take_idle(&self.uploads, class, Reuse::Fit) {
            Some(buffer) => Some(buffer),
            None => self.make_buffer(class, UPLOAD_USAGE, true),
        }
    }

    /// A staging buffer holding `data` at its start, unmapped and ready to be copied out of, for
    /// the uniform slots of [`Arena::upload`]. It counts as in flight: give it to
    /// [`Batch::uploads`], which maps it again once the batch that copies out of it is submitted.
    /// `None` if no staging buffer can be had (the device is out of memory).
    ///
    /// This runs while the batch is being submitted, with the batch locked, so it must not flush
    /// or wait: it reuses an idle buffer or creates a mapped one.
    fn stage(&self, data: &[u8]) -> Option<Buffer> {
        let bytes = data.len() as u64;
        let staging = self.mapped_staging(bytes)?;
        {
            // A plain copy: the slots of one chunk are kilobytes, and `write_parallel` asks the
            // OS for the thread count first, which costs more than copying them.
            let mut view = own(&staging).get_mapped_range_mut(..bytes).ok()?;
            view.slice(..).copy_from_slice(data);
        }
        own(&staging).unmap();
        lock(&self.uploads).in_flight += staging.size();
        Some(staging)
    }

    /// Writes `data` at the start of the mapped staging buffer `staging` and unmaps it, so that
    /// the commands recorded to copy out of it can be submitted. Only [`Arena::upload`] calls
    /// this, for the buffer of a [`Prefix`], which is mapped for as long as it is not filled.
    fn fill(&self, staging: &Buffer, data: &[u8]) {
        let native = own(staging);
        if !data.is_empty() {
            match native.get_mapped_range_mut(..data.len() as u64) {
                Ok(mut view) => view.slice(..).copy_from_slice(data),
                Err(error) => {
                    lock(&self.gpu_error).get_or_insert_with(|| format!("uniform staging buffer: {error}"));
                }
            }
        }
        native.unmap();
    }

    /// Records copying the first [`UNIFORM_PREFIX`] bytes of the uniform chunk `chunk` (started in
    /// arena epoch `epoch`) out of a staging buffer that [`Arena::upload`] fills when the batch is
    /// submitted, so that the slots of a small batch need no command buffer of their own. Must
    /// be recorded before the batch's first dispatch. Without a staging buffer (the device is
    /// out of memory) everything goes through that separate buffer.
    fn copy_prefix(&self, encoder: &mut wgpu::CommandEncoder, prefix: &mut Option<Prefix>, chunk: &Buffer, epoch: u64) {
        if let Some(staging) = self.mapped_staging(UNIFORM_PREFIX) {
            lock(&self.uploads).in_flight += staging.size();
            encoder.copy_buffer_to_buffer(own(&staging), 0, own(chunk), 0, UNIFORM_PREFIX);
            *prefix = Some(Prefix { staging, epoch });
        }
    }

    /// An idle readback buffer of at least `bytes` bytes (the smallest there is), or a new one of
    /// the size class. Callers copy, map and read only the first `bytes` bytes, never the whole
    /// buffer, so a larger buffer serves as well as one of the exact class: with exact classes
    /// only, an idle pool full of the large buffers of an earlier transfer never matched a small
    /// request, and each small readback created and freed a buffer (on CUDA about 1 ms per 4-byte
    /// read).
    fn readback_buffer(&self, bytes: u64) -> Result<Buffer, BackendError> {
        let class = size_class(bytes, self.max_buffer_bytes).ok_or(BackendError::OutOfMemory { requested: bytes })?;
        if let Some(buffer) = lock(&self.readbacks).take_fit(class) {
            return Ok(buffer);
        }
        self.new_buffer(class, STAGING_USAGE, false)
    }

    /// Gives an unmapped readback buffer back to the idle pool.
    fn recycle_readback(&self, buffer: Buffer) {
        let dropped = lock(&self.readbacks).put(buffer);
        if let Some(dropped) = dropped {
            lock(&self.released).push(dropped);
        }
    }

    // -- Submission --

    /// Whether earlier submissions are still running. A failed poll counts as idle, so that the
    /// submission goes ahead and the waiting paths report the error.
    fn busy(&self) -> bool {
        matches!(self.device().poll(wgpu::PollType::Poll), Ok(status) if !status.is_queue_empty())
    }

    /// Records a copy of `data` into `dst` at `offset`, ordered after everything recorded so
    /// far. `data.len()` must be a multiple of 4.
    ///
    /// The bytes go through a mappable staging buffer. Those are reused: once the batch that
    /// copies out of one is submitted it is mapped again, and the next upload of its size class
    /// writes straight into it.
    fn upload_at(&self, dst: &Buffer, offset: u64, data: &[u8], max_buffer_bytes: u64) -> Result<(), BackendError> {
        let dst = raw(dst)?;
        let bytes = data.len() as u64;
        let class = size_class(bytes, max_buffer_bytes).ok_or(BackendError::OutOfMemory { requested: bytes })?;
        let staging = self.take_upload_buffer(&self.uploads, class, UPLOAD_USAGE, Reuse::Fit)?;
        {
            let mut view = own(&staging).get_mapped_range_mut(..bytes).map_err(|e| BackendError::MapRange(e.to_string()))?;
            write_parallel(data, view.slice(..));
        }
        own(&staging).unmap();

        let mut guard = self.batch()?;
        self.encoder(&mut guard).copy_buffer_to_buffer(own(&staging), 0, dst, offset, bytes);
        lock(&self.uploads).in_flight += staging.size();
        guard.uploads.push(staging);
        Ok(())
    }

    /// Submits the pending batch; the ticket is its submission (done when nothing was pending).
    fn submitted_ticket(&self) -> Ticket {
        match self.batch() {
            Ok(mut batch) => match self.flush_locked(&mut batch) {
                Some(index) => Ticket::new(WgpuTicket(index)),
                None => Ticket::done(),
            },
            Err(_) => Ticket::done(),
        }
    }

    /// Submits the pending batch (if there is one) and prepares the next. Never waits.
    fn flush_locked(&self, batch: &mut Batch) -> Option<wgpu::SubmissionIndex> {
        // The open pass has to end before the encoder can be finished.
        batch.pass = None;
        batch.bound = Bound::default();
        batch.dispatches = 0;
        self.dispatches_pending.store(0, Ordering::Relaxed);
        // The slots of the arena are reused by the next batch.
        batch.grid_slots.clear();
        let released = std::mem::take(&mut *lock(&self.released));
        let prefix = batch.prefix.take();
        let mut retired = Vec::new();
        let uniforms = batch.arena.upload(self, prefix, &mut batch.uploads, &mut retired);
        if batch.bind_groups.len > MAX_CACHED_BIND_GROUPS {
            // In-flight work keeps what it uses alive; this only drops our references.
            batch.bind_groups = BindGroupCache::default();
        } else if !released.is_empty() {
            batch.bind_groups.forget(&released);
        }
        // Everything recorded that uses these buffers is in the batch submitted below.
        let unmapped = std::mem::take(&mut *lock(&self.unmapped));
        // The uniform copies go first: the batch's dispatches read what they write.
        let commands = batch.encoder.take().map(|encoder| encoder.finish());
        let submission = (uniforms.is_some() || commands.is_some()).then(|| self.queue().submit(uniforms.into_iter().chain(commands)));
        if !retired.is_empty() {
            self.retire_chunks(retired);
        }
        for staging in batch.uploads.drain(..) {
            let free = Arc::clone(&self.uploads);
            let buffer = staging.clone();
            own(&staging).slice(..).map_async(wgpu::MapMode::Write, move |result| {
                let mut free = lock(&free);
                free.in_flight = free.in_flight.saturating_sub(buffer.size());
                if result.is_ok() {
                    free.put(buffer);
                }
            });
        }
        for storage in unmapped {
            let free = Arc::clone(&self.mappable);
            let released = Arc::clone(&self.released);
            let buffer = storage.clone();
            own(&storage).slice(..).map_async(wgpu::MapMode::Write, move |result| {
                let dropped = {
                    let mut free = lock(&free);
                    free.in_flight = free.in_flight.saturating_sub(buffer.size());
                    if result.is_ok() { free.put(buffer) } else { None }
                };
                if let Some(dropped) = dropped {
                    lock(&released).push(dropped);
                }
            });
        }
        submission
    }

    /// Submits the pending batch, waits for the GPU to finish it and resolves the timestamps of
    /// the profiler's query sets into its profile, releasing the sets. Called with the batch
    /// locked, so it waits on the device directly instead of through [`Backend::sync`].
    fn drain_timestamps(&self, batch: &mut Batch) -> Result<(), BackendError> {
        self.flush_locked(batch);
        self.device().poll(wgpu::PollType::wait_indefinitely()).map_err(|e| BackendError::Poll(e.to_string()))?;
        self.take_gpu_error()?;
        match batch.profiler.as_mut() {
            Some(profiler) => self.read_timestamps(profiler),
            None => Ok(()),
        }
    }

    /// Resolves the timestamps of every query set `profiler` holds into its profile and releases
    /// the sets. The work that wrote them must have completed already (Metal writes the samples
    /// late). Never takes the lock of the batch, so it may run while the caller holds it.
    fn read_timestamps(&self, profiler: &mut Profiler) -> Result<(), BackendError> {
        let mut encoder = self.device().create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("luma timestamps") });
        let Some((staging, labels)) = profiler.resolve(self.device(), &mut encoder) else {
            return Ok(());
        };
        let submission = self.queue().submit(Some(encoder.finish()));
        let slice = staging.slice(..);
        let mapped = map_for_read(slice);
        let read = self.wait_map(Some(submission), &mapped, Duration::ZERO).and_then(|_| {
            let view = slice.get_mapped_range().map_err(|e| BackendError::MapRange(e.to_string()))?;
            let stamps: Vec<u64> = bytemuck::pod_collect_to_vec(&view);
            let period = f64::from(self.queue().get_timestamp_period());
            profiler.add_samples(labels.iter().zip(stamps.as_chunks::<2>().0).map(|(label, &[begin, end])| {
                let nanos = end.saturating_sub(begin) as f64 * period;
                (Arc::clone(label), Duration::from_secs_f64(nanos * 1e-9))
            }));
            Ok(())
        });
        staging.unmap();
        read
    }

    /// Waits until `submission` (the latest one if `None`) has run and the mapping `mapped`
    /// reports on is done, and returns how long that took. Polls for up to `spin_budget` first.
    fn wait_map(&self, submission: Option<wgpu::SubmissionIndex>, mapped: &MapReceiver, spin_budget: Duration) -> Result<Duration, BackendError> {
        let start = Instant::now();
        let mapped = match self.spin_for_map(mapped, spin_budget) {
            Some(received) => received,
            None => {
                self.device()
                    .poll(wgpu::PollType::Wait { submission_index: submission, timeout: None })
                    .map_err(|e| BackendError::Poll(e.to_string()))?;
                // The callback runs on whichever thread polls the device first, so it may still
                // be in flight on another thread after our poll returns; wait for it.
                mapped.recv()
            }
        };
        let waited = start.elapsed();
        let mapped = mapped.map_err(|_| BackendError::Map("the map callback was dropped".into()))?.map_err(|e| BackendError::Map(e.to_string()));
        // An error the GPU raised while running the batch explains a failed map better.
        self.take_gpu_error()?;
        mapped.map(|()| waited)
    }

    /// Polls without blocking for up to `budget` until the mapping `mapped` reports on is done.
    /// `None` if it is not (or a poll failed, which the blocking wait reports).
    fn spin_for_map(&self, mapped: &MapReceiver, budget: Duration) -> Option<Result<Result<(), wgpu::BufferAsyncError>, mpsc::RecvError>> {
        let deadline = Instant::now() + budget;
        loop {
            self.device().poll(wgpu::PollType::Poll).ok()?;
            match mapped.try_recv() {
                Ok(result) => return Some(Ok(result)),
                Err(mpsc::TryRecvError::Disconnected) => return Some(Err(mpsc::RecvError)),
                Err(mpsc::TryRecvError::Empty) => {}
            }
            if Instant::now() >= deadline {
                return None;
            }
            std::hint::spin_loop();
        }
    }

    /// Polls without blocking for up to `budget`; `true` once the queue is empty. A thread that
    /// notices the end of a short kernel while running returns sooner than one that was blocked
    /// in `poll` and has to be woken up.
    fn spin_until_idle(&self, budget: Duration) -> bool {
        let deadline = Instant::now() + budget;
        loop {
            match self.device().poll(wgpu::PollType::Poll) {
                Ok(status) if status.is_queue_empty() => return true,
                Ok(_) => {}
                // The blocking wait reports it.
                Err(_) => return false,
            }
            if Instant::now() >= deadline {
                return false;
            }
            std::hint::spin_loop();
        }
    }

    // -- Kernels --

    /// Builds the pipeline of the entry point `entry` of `module`. `numbers` are the original
    /// `@group(1)` binding numbers of the kernel's `infos`, in order.
    #[allow(clippy::too_many_arguments)]
    fn build(&self, module: naga::Module, entry: &str, numbers: &[u32], infos: &[BindingInfo], workgroup: [u32; 3], label: &str, trust: KernelTrust) -> Result<RawKernel, BackendError> {
        let limits = &self.info.limits;
        let invocations = workgroup.iter().fold(1u64, |total, &d| total.saturating_mul(u64::from(d)));
        if workgroup.contains(&0)
            || workgroup.iter().zip(limits.max_workgroup_size).any(|(&d, max)| d > max)
            || invocations > u64::from(limits.max_workgroup_invocations)
        {
            return Err(BackendError::KernelWorkgroup(format!(
                "{workgroup:?} exceeds the device limits (per dimension {:?}, {} invocations in total)",
                limits.max_workgroup_size, limits.max_workgroup_invocations
            )));
        }
        // The grid and every user uniform are bound at dynamic offsets into the uniform arena.
        let uniforms = infos.iter().filter(|b| b.kind == BindingKind::Uniform).count() as u32;
        if uniforms + 1 > self.max_dynamic_uniforms {
            return Err(BackendError::KernelBinding(format!(
                "{uniforms} uniform arguments exceed the device limit of {} dynamic uniform buffers per kernel (one is used by luma's grid)",
                self.max_dynamic_uniforms.saturating_sub(1)
            )));
        }

        let device = self.device();
        let arg_entries: Vec<_> = infos
            .iter()
            .zip(numbers)
            .map(|(info, &binding)| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: match info.kind {
                    BindingKind::Texture2d => wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    BindingKind::Sampler => wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    kind => wgpu::BindingType::Buffer {
                        ty: match kind {
                            BindingKind::Uniform => wgpu::BufferBindingType::Uniform,
                            BindingKind::StorageRead => wgpu::BufferBindingType::Storage { read_only: true },
                            _ => wgpu::BufferBindingType::Storage { read_only: false },
                        },
                        has_dynamic_offset: kind == BindingKind::Uniform,
                        min_binding_size: None,
                    },
                },
                count: None,
            })
            .collect();
        let args_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some(label), entries: &arg_entries });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(label),
            bind_group_layouts: &[Some(&self.grid_layout), Some(&args_layout)],
            immediate_size: 0,
        });
        let shader_desc = wgpu::ShaderModuleDescriptor {
            label: Some(label),
            // The module luma parsed and validated, not its text: wgpu would parse it a second time.
            source: wgpu::ShaderSource::Naga(Cow::Owned(module)),
        };
        let shader = if !trust.checks_bounds() {
            // Integer division keeps its guards: WGSL defines `x / 0` and `x % 0`, and user data
            // reaches luma's kernels.
            let mut checks = wgpu::ShaderRuntimeChecks::checked();
            checks.bounds_checks = false;
            // User kernels keep the bound on their loops.
            checks.force_loop_bounding &= trust != KernelTrust::Trusted;
            // SAFETY: luma's built-in and generated kernels index storage buffers only below the
            // lengths they are launched with, which never exceed the bound ranges, and all of
            // their loops are bounded. A user kernel is only built this way on its author's word
            // (`KernelTrust::UncheckedBounds`, see `KernelBuilder::unchecked_bounds`).
            unsafe { device.create_shader_module_trusted(shader_desc, checks) }
        } else {
            device.create_shader_module(shader_desc)
        };
        let store = &self.gpu.0.store;
        let pipeline = crate::exit::driver_call(|| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some(entry),
                // Trusted kernels write their workgroup memory before reading it.
                compilation_options: wgpu::PipelineCompilationOptions { zero_initialize_workgroup_memory: trust.zero_workgroup_memory(), ..Default::default() },
                cache: store.cache(),
            })
        });
        store.built();
        let textured = infos.iter().any(|b| matches!(b.kind, BindingKind::Texture2d | BindingKind::Sampler));
        Ok(RawKernel::new(WgpuKernel { pipeline, args_layout, textured }))
    }
}

impl Backend for WgpuBackend {
    fn info(&self) -> &DeviceInfo {
        &self.info
    }

    fn disk_cache(&self) -> &Arc<DiskCache> {
        &self.disk
    }

    fn device_cost(&self) -> &Arc<OnceLock<CostModel>> {
        &self.cost
    }

    fn create_buffer(&self, size: u64, usage: BufferUsage, mapped: bool) -> Option<Buffer> {
        self.make_buffer(size, usage, mapped)
    }

    /// A storage buffer holding `data` at its start, written by the host without a GPU copy.
    /// Only on unified memory; give it back with [`Backend::recycle_mappable`].
    fn alloc_filled(&self, data: &[u8], max_buffer_bytes: u64) -> Result<Buffer, BackendError> {
        let bytes = data.len() as u64;
        let class = size_class(bytes, max_buffer_bytes).ok_or(BackendError::OutOfMemory { requested: bytes })?;
        let buffer = self.take_upload_buffer(&self.mappable, class, MAPPABLE_STORAGE_USAGE, Reuse::Exact)?;
        {
            let mut view = own(&buffer).get_mapped_range_mut(..bytes).map_err(|e| BackendError::MapRange(e.to_string()))?;
            write_parallel(data, view.slice(..));
        }
        own(&buffer).unmap();
        Ok(buffer)
    }

    fn texture_caps(&self) -> Option<TextureCaps> {
        self.device().features().contains(wgpu::Features::FLOAT32_FILTERABLE).then(|| TextureCaps {
            max_dim: self.device().limits().max_texture_dimension_2d,
            pitch_align: wgpu::COPY_BYTES_PER_ROW_ALIGNMENT,
            aliases: false,
        })
    }

    fn create_texture(&self, source: &Buffer, offset: u64, pitch: u64, desc: &TextureDesc) -> Result<Buffer, BackendError> {
        let caps = self.texture_caps().ok_or_else(|| BackendError::Unsupported("the device cannot filter f32 textures".into()))?;
        let (w, h) = (u64::from(desc.width), u64::from(desc.height));
        let row = w * u64::from(desc.format.texel_bytes());
        if w == 0 || h == 0 || desc.width > caps.max_dim || desc.height > caps.max_dim {
            return Err(BackendError::Unsupported(format!("a {w}x{h} texture (up to {0}x{0} are possible)", caps.max_dim)));
        }
        if pitch < row || !pitch.is_multiple_of(u64::from(caps.pitch_align)) || !offset.is_multiple_of(4) || offset + pitch * (h - 1) + row > source.size() {
            return Err(BackendError::KernelBinding(format!("a {w}x{h} texture does not fit {} bytes at offset {offset} with rows {pitch} bytes apart", source.size())));
        }
        let format = match desc.format {
            TextureFormat::R32Float => wgpu::TextureFormat::R32Float,
            TextureFormat::Rgba32Float => wgpu::TextureFormat::Rgba32Float,
        };
        let size = wgpu::Extent3d { width: desc.width, height: desc.height, depth_or_array_layers: 1 };
        let device = self.device();
        let scope = device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("luma texture"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        if pollster::block_on(scope.pop()).is_some() {
            return Err(BackendError::OutOfMemory { requested: row * h });
        }
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let mode = match desc.filter {
            TextureFilter::Nearest => wgpu::FilterMode::Nearest,
            TextureFilter::Linear => wgpu::FilterMode::Linear,
        };
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("luma sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: mode,
            min_filter: mode,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });
        let mut guard = self.batch()?;
        self.encoder(&mut guard).copy_buffer_to_texture(
            wgpu::TexelCopyBufferInfo {
                buffer: raw(source)?,
                layout: wgpu::TexelCopyBufferLayout { offset, bytes_per_row: Some(pitch as u32), rows_per_image: Some(desc.height) },
            },
            wgpu::TexelCopyTextureInfo { texture: &texture, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            size,
        );
        drop(guard);
        Ok(Buffer::new(row * h, BufferUsage::TEXTURE, WgpuTexture { _texture: texture, view, sampler }))
    }

    fn recycle_mappable(&self, buffer: Buffer) {
        // May still be used by the unsubmitted batch, which mapping would invalidate; it is
        // mapped after the next submission. This must not take the batch lock: it can be held by
        // the caller.
        lock(&self.mappable).in_flight += buffer.size();
        lock(&self.unmapped).push(buffer);
    }

    fn compile_kernel(&self, module: &KernelModule, used: Option<&[bool]>, label: &str, trust: KernelTrust) -> Result<RawKernel, BackendError> {
        if let Some(mask) = used
            && mask.len() != module.bindings.len()
        {
            return Err(BackendError::KernelBinding(format!("the used-binding mask has {} entries for {} bindings", mask.len(), module.bindings.len())));
        }
        let numbers: Vec<u32> = (0..module.bindings.len() as u32).filter(|&i| used.is_none_or(|u| u[i as usize])).collect();
        let infos: Vec<BindingInfo> = numbers.iter().map(|&i| module.bindings[i as usize].info()).collect();
        let optimized = luma_core::kir::optimize_for_target(module, false);
        let source = codegen::emit_with(optimized.as_ref(), used, self.device().features().contains(wgpu::Features::SHADER_FLOAT32_ATOMIC));
        if let Some(dir) = DUMP_DIR.as_deref() {
            let name: String = label.chars().take(100).map(|c| if c.is_ascii_alphanumeric() || c == '_' { c } else { '_' }).collect();
            let _ = std::fs::create_dir_all(dir);
            let _ = std::fs::write(dir.join(format!("{name}-{:016x}.wgsl", luma_core::hal::cache::fnv1a(source.as_bytes()))), &source);
        }
        let parsed = naga::front::wgsl::parse_str(&source).map_err(|e| BackendError::ShaderCompile(e.emit_to_string(&source)))?;
        naga::valid::Validator::new(naga::valid::ValidationFlags::all(), shader_capabilities(self.device().features()))
            .validate(&parsed)
            .map_err(|e| BackendError::ShaderCompile(e.emit_to_string(&source)))?;
        self.build(parsed, codegen::ENTRY, &numbers, &infos, module.workgroup_size, label, trust)
    }

    fn compile_naga(
        &self,
        module: &naga::Module,
        entry: &str,
        bindings: &[BindingInfo],
        workgroup: [u32; 3],
        label: &str,
        trust: KernelTrust,
    ) -> Option<Result<RawKernel, BackendError>> {
        Some((|| {
            let info = naga::valid::Validator::new(naga::valid::ValidationFlags::all(), shader_capabilities(self.device().features()))
                .validate(module)
                .map_err(|e| BackendError::ShaderCompile(format!("{e:?}")))?;
            let (index, _) = module
                .entry_points
                .iter()
                .enumerate()
                .find(|(_, ep)| ep.name == entry && ep.stage == naga::ShaderStage::Compute)
                .ok_or_else(|| BackendError::KernelEntryPoint(format!("No @compute entry point `{entry}`")))?;
            // The original binding numbers of the variables the entry point reaches: for luma's
            // own sparse kernels they are not 0..n.
            let reached = info.get_entry_point(index);
            let mut numbers: Vec<u32> = module
                .global_variables
                .iter()
                .filter_map(|(handle, var)| var.binding.filter(|rb| rb.group == 1 && !reached[handle].is_empty()).map(|rb| rb.binding))
                .collect();
            numbers.sort_unstable();
            if numbers.len() != bindings.len() {
                return Err(BackendError::KernelBinding(format!(
                    "entry point `{entry}` reaches {} @group(1) bindings but {} were reflected",
                    numbers.len(),
                    bindings.len()
                )));
            }
            self.build(module.clone(), entry, &numbers, bindings, workgroup, label, trust)
        })())
    }

    /// The body of a launch: allocates the arena slots, finds or creates the bind group, and
    /// records the dispatch. After the first launch of a kernel with a given set of buffers this
    /// allocates nothing.
    fn record_launch(&self, kernel: &CompiledKernel, dispatch: Dispatch<'_>, size: [u32; 3], args: &[RawArg<'_>]) -> Result<u32, BackendError> {
        // Before anything is allocated or recorded: a launch that does not fit the kernel is an
        // error of its caller, not of the batch.
        kernel.check_count(args.len())?;
        kernel.check_kinds(args)?;
        kernel.check_offsets(args, self.storage_offset_align)?;
        let native = kernel.raw.downcast_ref::<WgpuKernel>().ok_or_else(|| BackendError::Unsupported("the kernel belongs to another backend".into()))?;
        let indirect = match dispatch {
            Dispatch::Direct(_) => None,
            Dispatch::Indirect { buffer, offset, .. } => Some((raw(buffer)?, offset)),
        };

        let mut guard = self.batch()?;
        let batch: &mut Batch = &mut guard;
        // Before the launch may open a query set past the profiler's limit of live sets.
        if batch.profiler.as_ref().is_some_and(Profiler::at_capacity) {
            self.drain_timestamps(batch)?;
        }
        let device = self.device();
        let align = self.uniform_align;
        let Batch { encoder, prefix, pass, bound, dispatches, arena, grid_slots, bind_groups, offsets, profiler, .. } = batch;

        // Uniform values go into fresh arena slots, all of this launch's in one chunk. The grid
        // uniform shares the slot of an earlier launch of the same size.
        let uniform_bytes: u64 = kernel
            .bound(args)
            .map(|(_, arg)| match arg {
                RawArg::Uniform(data) => slot_size(align, data.len()),
                RawArg::Storage { .. } => 0,
            })
            .sum();
        let grid_bytes = slot_size(align, GRID_BYTES as usize);
        let mut grid_offset = grid_slots.iter().find(|(known, _)| *known == size).map(|&(_, offset)| offset);
        let needed = uniform_bytes + if grid_offset.is_some() { 0 } else { grid_bytes };
        if !arena.has_room(needed) {
            arena.roll(self.take_chunk(uniform_bytes + grid_bytes));
            // The remembered slots belong to the chunk that was left.
            grid_slots.clear();
            grid_offset = None;
        }
        let grid_offset = match grid_offset {
            Some(offset) => offset,
            None => {
                let offset = dynamic_offset(arena.push(align, bytemuck::bytes_of(&[size[0], size[1], size[2], 0u32])))?;
                if grid_slots.len() < MAX_GRID_SLOTS {
                    grid_slots.push((size, offset));
                }
                offset
            }
        };
        offsets.clear();
        for (_, arg) in kernel.bound(args) {
            if let RawArg::Uniform(data) = arg {
                offsets.push(dynamic_offset(arena.push(align, data))?);
            }
        }

        // Bind groups: the grid's per chunk, the arguments' per (kernel, buffers).
        let epoch = arena.epoch;
        let Chunk { buffer: uniforms, grid_group, .. } = &mut arena.current;
        let uniforms: &Buffer = uniforms;
        let grid_group = &*grid_group.get_or_insert_with(|| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.grid_layout,
                entries: &[BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::Buffer(BufferBinding { buffer: own(uniforms), offset: 0, size: std::num::NonZeroU64::new(GRID_BYTES) }),
                }],
            })
        });

        // What is bound: a buffer and how many of its bytes. Uniforms live in the arena chunk
        // this launch's slots were allocated from.
        let bindings = || {
            kernel.bound(args).map(|(_, arg)| match *arg {
                RawArg::Storage { buffer, offset, bytes, .. } => (buffer, offset, bytes),
                RawArg::Uniform(data) => (uniforms, 0, data.len() as u64),
            })
        };
        let mut hasher = FxHasher::default();
        kernel.id.hash(&mut hasher);
        for (buffer, offset, bytes) in bindings() {
            buffer.hash(&mut hasher);
            offset.hash(&mut hasher);
            bytes.hash(&mut hasher);
        }
        let key = hasher.finish();
        let textured_group;
        let (args_group, cached) = if native.textured {
            let mut group_entries: Vec<BindGroupEntry<'_>> = Vec::with_capacity(args.len());
            for (((binding, _), (buffer, offset, bytes)), info) in kernel.bound(args).zip(bindings()).zip(&kernel.bindings) {
                let resource = match info.kind {
                    BindingKind::Texture2d => BindingResource::TextureView(&texture(buffer)?.view),
                    BindingKind::Sampler => BindingResource::Sampler(&texture(buffer)?.sampler),
                    _ => BindingResource::Buffer(BufferBinding { buffer: raw(buffer)?, offset, size: std::num::NonZeroU64::new(bytes) }),
                };
                group_entries.push(BindGroupEntry { binding, resource });
            }
            textured_group = device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &native.args_layout, entries: &group_entries });
            (&textured_group, None)
        } else {
            let bucket = bind_groups.buckets.entry(key).or_default();
            let index = match bucket.iter().position(|cached| cached.matches(kernel.id, bindings())) {
                Some(index) => index,
                None => {
                    let mut group_entries: Vec<BindGroupEntry<'_>> = Vec::with_capacity(args.len());
                    for ((binding, _), (buffer, offset, bytes)) in kernel.bound(args).zip(bindings()) {
                        group_entries.push(BindGroupEntry {
                            binding,
                            resource: BindingResource::Buffer(BufferBinding { buffer: raw(buffer)?, offset, size: std::num::NonZeroU64::new(bytes) }),
                        });
                    }
                    let group = device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &native.args_layout, entries: &group_entries });
                    bucket.push(CachedGroup { kernel: kernel.id, entries: bindings().map(|(buffer, offset, bytes)| (buffer.clone(), offset, bytes)).collect(), group });
                    bind_groups.len += 1;
                    bucket.len().saturating_sub(1)
                }
            };
            let Some(args_group) = bucket.get(index).map(|cached| &cached.group) else {
                return Err(BackendError::Gpu("bind group cache entry vanished".into()));
            };
            (args_group, Some((key, index)))
        };

        // Record. Launches share one compute pass; a profiled launch gets a pass of its own that
        // carries the timestamp writes.
        let profiled = profiler.is_some();
        if profiled {
            *pass = None;
        }
        let encoder = encoder.get_or_insert_with(|| self.new_encoder());
        if *dispatches == 0 && prefix.is_none() {
            self.copy_prefix(encoder, prefix, uniforms, epoch);
        }
        let pass_slot = pass;
        let pass = pass_slot.get_or_insert_with(|| {
            *bound = Bound::default();
            let timestamp_writes = profiler.as_mut().map(|p| p.writes(device, &kernel.label));
            encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes }).forget_lifetime()
        });
        if bound.kernel != Some(kernel.id) {
            pass.set_pipeline(&native.pipeline);
            bound.kernel = Some(kernel.id);
            bound.args = None;
        }
        let grid = Some((epoch, grid_offset));
        if bound.grid != grid {
            pass.set_bind_group(0, grid_group, &[grid_offset]);
            bound.grid = grid;
        }
        // A group with uniforms is bound at new dynamic offsets by every launch.
        if !offsets.is_empty() || cached.is_none() || bound.args != cached {
            pass.set_bind_group(1, args_group, offsets.as_slice());
            bound.args = cached;
        }
        match (dispatch, indirect) {
            (Dispatch::Direct([x, y, z]), _) => pass.dispatch_workgroups(x, y, z),
            (Dispatch::Indirect { .. }, Some((buffer, offset))) => pass.dispatch_workgroups_indirect(buffer, offset),
            (Dispatch::Indirect { .. }, None) => return Err(BackendError::Gpu("an indirect dispatch lost its buffer".into())),
        }
        if profiled {
            *pass_slot = None;
        }
        *dispatches += 1;
        self.dispatches_pending.store(*dispatches, Ordering::Relaxed);
        Ok(*dispatches)
    }

    fn record_clear(&self, buffer: &Buffer, bytes: u64) -> Result<(), BackendError> {
        let native = raw(buffer)?;
        let mut guard = self.batch()?;
        self.encoder(&mut guard).clear_buffer(native, 0, Some(bytes));
        Ok(())
    }

    fn record_copy(&self, src: &Buffer, src_offset: u64, dst: &Buffer, dst_offset: u64, bytes: u64) -> Result<(), BackendError> {
        let (src, dst) = (raw(src)?, raw(dst)?);
        let mut guard = self.batch()?;
        self.encoder(&mut guard).copy_buffer_to_buffer(src, src_offset, dst, dst_offset, bytes);
        Ok(())
    }

    fn record_upload(&self, dst: &Buffer, data: &[u8], max_buffer_bytes: u64) -> Result<(), BackendError> {
        self.upload_at(dst, 0, data, max_buffer_bytes)
    }

    fn transfer_in(&self, stage: &Stage, dst: &Buffer, dst_offset: u64, bytes: u64, _after: &[Ticket]) -> Result<Ticket, BackendError> {
        // One queue, in order: `after` is satisfied by everything recorded so far. The only
        // overlap is host staging against compute: the copy shares the queue with the kernels.
        check_transfer(stage, StageDir::ToDevice, dst, dst_offset, bytes)?;
        self.upload_at(dst, dst_offset, &stage.as_slice()[..bytes as usize], self.info().limits.max_buffer_bytes)?;
        Ok(self.submitted_ticket())
    }

    fn barrier(&self, _after: &[Ticket]) -> Result<Ticket, BackendError> {
        Ok(self.submitted_ticket())
    }

    fn ticket_done(&self, t: &Ticket) -> Result<bool, BackendError> {
        let Some(WgpuTicket(index)) = t.downcast_ref() else {
            return if t.is_done_marker() { Ok(true) } else { Err(BackendError::Unsupported("a ticket of another backend".into())) };
        };
        match self.device().poll(wgpu::PollType::Wait { submission_index: Some(index.clone()), timeout: Some(Duration::ZERO) }) {
            Ok(_) => Ok(true),
            Err(wgpu::PollError::Timeout) => Ok(false),
            Err(e) => Err(BackendError::Poll(e.to_string())),
        }
    }

    fn ticket_wait(&self, t: &Ticket, spin: Duration) -> Result<(), BackendError> {
        let Some(WgpuTicket(index)) = t.downcast_ref() else {
            return if t.is_done_marker() { Ok(()) } else { Err(BackendError::Unsupported("a ticket of another backend".into())) };
        };
        let deadline = Instant::now() + spin;
        while Instant::now() < deadline {
            if self.ticket_done(t)? {
                return self.take_gpu_error();
            }
            std::hint::spin_loop();
        }
        self.device().poll(wgpu::PollType::Wait { submission_index: Some(index.clone()), timeout: None }).map_err(|e| BackendError::Poll(e.to_string()))?;
        self.take_gpu_error()
    }

    fn record_graph(&self, _graph: &dyn ExecGraph, _dispatches: u32) -> Result<u32, BackendError> {
        Err(BackendError::Unsupported("graph replay requires the CUDA backend".into()))
    }

    fn pending_dispatches(&self) -> u32 {
        self.dispatches_pending.load(Ordering::Relaxed)
    }

    fn gpu_busy(&self) -> bool {
        self.busy()
    }

    fn notify_released(&self, buffers: &[Buffer]) {
        if !buffers.is_empty() {
            lock(&self.released).extend(buffers.iter().cloned());
        }
    }

    fn flush_batch(&self) -> Result<(), BackendError> {
        let mut batch = self.batch()?;
        self.flush_locked(&mut batch);
        Ok(())
    }

    fn sync(&self, spin_budget: Duration) -> Result<Duration, BackendError> {
        self.flush_batch()?;
        let start = Instant::now();
        if !self.spin_until_idle(spin_budget) {
            self.device().poll(wgpu::PollType::wait_indefinitely()).map_err(|e| BackendError::Poll(e.to_string()))?;
        }
        let waited = start.elapsed();
        self.take_gpu_error()?;
        Ok(waited)
    }

    /// Copies the first `dst.len()` bytes of `src` into `dst`, after everything recorded so far;
    /// on success every byte of `dst` is written. Large reads go in `READ_CHUNK` pieces, each
    /// its own submission, so copying one piece out of its staging buffer overlaps the transfer
    /// of the next. With `spill` the pieces go through one staging buffer one after the other,
    /// for readbacks made while the device is full.
    fn read_into(&self, src: &Buffer, dst: &mut [MaybeUninit<u8>], spill: bool, spin_budget: Duration) -> Result<Duration, BackendError> {
        let src = raw(src)?;
        let bytes = dst.len() as u64;
        let mut longest = Duration::ZERO;

        if spill {
            let chunk = READ_CHUNK.min(bytes).max(1);
            let staging = self.readback_buffer(chunk)?;
            let native = own(&staging);
            let mut result = Ok(());
            for offset in (0..bytes).step_by(chunk as usize) {
                let size = chunk.min(bytes - offset);
                let step = (|| {
                    let submission = {
                        let mut guard = self.batch()?;
                        self.encoder(&mut guard).copy_buffer_to_buffer(src, offset, native, 0, size);
                        self.flush_locked(&mut guard)
                    };
                    let mapped = map_for_read(native.slice(..size));
                    let waited = self.wait_map(submission, &mapped, spin_budget)?;
                    longest = longest.max(waited);
                    let view = native.slice(..size).get_mapped_range().map_err(|e| BackendError::MapRange(e.to_string()))?;
                    copy_parallel(&view, &mut dst[offset as usize..(offset + size) as usize]);
                    Ok::<(), BackendError>(())
                })();
                native.unmap();
                if step.is_err() {
                    result = step;
                    break;
                }
            }
            self.recycle_readback(staging);
            return result.map(|()| longest);
        }

        let chunk = if bytes > 2 * READ_CHUNK { READ_CHUNK } else { bytes.max(1) };
        let mut staged = Vec::new();
        for offset in (0..bytes).step_by(chunk as usize) {
            let size = chunk.min(bytes - offset);
            match self.readback_buffer(size) {
                Ok(staging) => staged.push((offset, size, staging)),
                Err(err) => {
                    for (_, _, s) in staged {
                        self.recycle_readback(s);
                    }
                    return Err(err);
                }
            }
        }
        let mut maps = Vec::with_capacity(staged.len());
        match self.batch() {
            Ok(mut guard) => {
                for (offset, size, staging) in &staged {
                    self.encoder(&mut guard).copy_buffer_to_buffer(src, *offset, own(staging), 0, *size);
                    let submission = self.flush_locked(&mut guard);
                    maps.push((submission, map_for_read(own(staging).slice(..*size))));
                }
            }
            Err(err) => {
                for (_, _, s) in staged {
                    self.recycle_readback(s);
                }
                return Err(err);
            }
        }

        let mut staged_iter = staged.into_iter().zip(maps);
        while let Some(((offset, size, staging), (submission, mapped))) = staged_iter.next() {
            let native = own(&staging);
            let step = self.wait_map(submission, &mapped, spin_budget).and_then(|waited| {
                longest = longest.max(waited);
                let view = native.slice(..size).get_mapped_range().map_err(|e| BackendError::MapRange(e.to_string()))?;
                copy_parallel(&view, &mut dst[offset as usize..(offset + size) as usize]);
                Ok(())
            });
            native.unmap();
            if let Err(err) = step {
                self.recycle_readback(staging);
                for ((_, _, rest), _) in staged_iter {
                    own(&rest).unmap();
                    self.recycle_readback(rest);
                }
                return Err(err);
            }
            self.recycle_readback(staging);
        }
        Ok(longest)
    }

    /// Gives back every idle buffer: queues them for release, submits, waits for the device,
    /// then drops the bind groups that pinned the mapped ones.
    fn reclaim_idle(&self) -> Result<(), BackendError> {
        let mut idle = lock(&self.mappable).drain();
        idle.extend(lock(&self.uploads).drain());
        idle.extend(lock(&self.readbacks).drain());
        idle.extend(lock(&self.free_chunks).drain(..).map(|chunk| chunk.buffer));
        lock(&self.released).extend(idle);
        self.flush_locked(&mut *self.batch()?);
        self.device().poll(wgpu::PollType::wait_indefinitely()).map_err(|e| BackendError::Poll(e.to_string()))?;
        let mut late = lock(&self.mappable).drain();
        late.extend(lock(&self.uploads).drain());
        late.extend(lock(&self.free_chunks).drain(..).map(|chunk| chunk.buffer));
        if !late.is_empty() {
            let mut batch = self.batch()?;
            batch.pass = None;
            batch.bound = Bound::default();
            batch.bind_groups.forget(&late);
        }
        Ok(())
    }

    fn enable_profiling(&self) -> Result<(), BackendError> {
        if !self.timestamps {
            return Err(BackendError::Profiling("the GPU does not support timestamp queries (wgpu feature TIMESTAMP_QUERY)".into()));
        }
        let mut batch = self.batch()?;
        if batch.profiler.is_none() {
            batch.profiler = Some(Profiler::new());
        }
        Ok(())
    }

    fn finish_profiling(&self, splits: Vec<FusionSplit>) -> Result<Profile, BackendError> {
        let mut profiler = {
            let mut batch = self.batch()?;
            let Some(profiler) = batch.profiler.take() else {
                return Err(BackendError::Profiling("profiling is not enabled".into()));
            };
            self.flush_locked(&mut batch);
            profiler
        };
        self.device().poll(wgpu::PollType::wait_indefinitely()).map_err(|e| BackendError::Poll(e.to_string()))?;
        self.take_gpu_error()?;
        // Everything profiled has completed; only now are its timestamps reliably written.
        self.read_timestamps(&mut profiler)?;
        let mut profile = profiler.into_profile();
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
    use luma_core::kir::{BuiltinVar, KirType, ModuleBuilder, used_bindings};

    use super::*;

    #[test]
    fn size_classes_round_up_to_powers_of_two_and_clamp() {
        const MAX: u64 = 1 << 32;
        assert_eq!(size_class(1, MAX), Some(16));
        assert_eq!(size_class(17, MAX), Some(32));
        assert_eq!(size_class(12_000_012, MAX), Some(16_777_216));
        let max = (3 << 30) + 2;
        assert_eq!(size_class((1 << 30) + 4, max), Some(2 << 30));
        assert_eq!(size_class((2 << 30) + 4, max), Some(3 << 30));
        assert_eq!(size_class((3 << 30) + 4, max), None);
    }

    #[test]
    fn uniform_slots_are_aligned_and_at_least_a_word() {
        assert_eq!(slot_size(256, 16), 256);
        assert_eq!(slot_size(256, 257), 512);
        assert_eq!(slot_size(4, 1), 4);
        assert_eq!(slot_size(4, 0), 4);
    }

    #[test]
    fn usages_map_to_wgpu() {
        let usage = wgpu_usage(BufferUsage::STORAGE | BufferUsage::COPY_SRC | BufferUsage::MAP_WRITE);
        assert_eq!(usage, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::MAP_WRITE);
        assert_eq!(wgpu_usage(BufferUsage::NONE), wgpu::BufferUsages::empty());
    }

    /// Opens a device, or `None` where the machine has no adapter at all.
    fn open() -> Option<WgpuBackend> {
        match WgpuBackend::new(DeviceKind::Auto) {
            Ok(backend) => Some(backend),
            Err(BackendError::NoAdapter(_) | BackendError::NoDevice(_) | BackendError::RequestDevice(_)) => None,
            Err(other) => panic!("{other}"),
        }
    }

    #[test]
    fn a_kir_kernel_runs_and_reads_back() {
        let Some(backend) = open() else { return };
        let mut mb = ModuleBuilder::new("double", "main", [64, 1, 1]);
        let _unused = mb.add_storage_read("unused", KirType::U32);
        let out = mb.add_storage_read_write("out", KirType::U32);
        let mut b = mb.entry_builder();
        let gid = b.builtin(BuiltinVar::GlobalInvocationId);
        let i = b.field(gid, 0, KirType::U32);
        let n = b.grid_size(0);
        let oob = b.ge(i, n);
        b.if_then(oob, |b| b.ret(None));
        let two = b.lit_u32(2);
        b.store(out, i, i * two);
        b.finish();
        let module = mb.finish();
        let (declared, used) = used_bindings(&module);
        assert_eq!((declared, used.as_deref()), (2, Some(&[false, true][..])));

        let raw_kernel = backend.compile_kernel(&module, used.as_deref(), "double", KernelTrust::Trusted).expect("compiles");
        let infos = vec![module.bindings[1].info()];
        let kernel = CompiledKernel::new(Arc::from("double"), raw_kernel, infos, declared, used, [64, 1, 1]);

        let n = 100u32;
        let bytes = u64::from(n) * 4;
        let buffer = backend.create_buffer(bytes.next_power_of_two(), STORAGE_USAGE, false).expect("a buffer");
        let dummy = backend.create_buffer(16, STORAGE_USAGE, false).expect("a buffer");
        backend.record_clear(&buffer, bytes).unwrap();
        let pending = backend
            .record_launch(&kernel, Dispatch::Direct([2, 1, 1]), [n, 1, 1], &[RawArg::read(&dummy, 16), RawArg::write(&buffer, bytes)])
            .unwrap();
        assert_eq!(pending, 1);
        assert_eq!(backend.pending_dispatches(), 1);
        let mut out = vec![MaybeUninit::<u8>::uninit(); bytes as usize];
        backend.read_into(&buffer, &mut out, false, Duration::from_micros(500)).unwrap();
        let words: Vec<u32> = out.chunks(4).map(|c| u32::from_le_bytes([0, 1, 2, 3].map(|k| unsafe { c[k].assume_init() }))).collect();
        assert_eq!(words, (0..n).map(|i| i * 2).collect::<Vec<_>>());
        assert_eq!(backend.pending_dispatches(), 0);
        backend.sync(Duration::from_micros(500)).unwrap();
    }

    #[test]
    fn uploads_go_through_staging_and_reads_come_back() {
        let Some(backend) = open() else { return };
        let data: Vec<u8> = (0..4096u32).flat_map(|i| i.to_le_bytes()).collect();
        let max = backend.info().limits.max_buffer_bytes;
        let buffer = backend.create_buffer(data.len() as u64, STORAGE_USAGE, false).expect("a buffer");
        backend.record_upload(&buffer, &data, max).unwrap();
        for spill in [false, true] {
            let mut out = vec![MaybeUninit::<u8>::uninit(); data.len()];
            backend.read_into(&buffer, &mut out, spill, Duration::ZERO).unwrap();
            let back: Vec<u8> = out.iter().map(|b| unsafe { b.assume_init() }).collect();
            assert_eq!(back, data);
        }
    }

    #[test]
    fn graphs_are_a_cuda_capability() {
        let Some(backend) = open() else { return };
        assert!(backend.capture_graph(&[]).is_none());
        #[derive(Debug)]
        struct Fake;
        impl ExecGraph for Fake {}
        assert!(matches!(backend.record_graph(&Fake, 1), Err(BackendError::Unsupported(_))));
    }

    #[test]
    fn profiling_reports_each_launch_or_says_it_is_unsupported() {
        let Some(backend) = open() else { return };
        assert!(!backend.is_profiling().unwrap());
        match backend.enable_profiling() {
            Ok(()) => {
                assert!(backend.is_profiling().unwrap());
                let profile = backend.finish_profiling(Vec::new()).unwrap();
                assert_eq!(profile.launches(), 0);
                assert!(!backend.is_profiling().unwrap());
            }
            Err(BackendError::Profiling(_)) => assert!(!backend.info().features.contains(Features::TIMESTAMP_QUERY)),
            Err(other) => panic!("{other}"),
        }
        assert!(matches!(backend.finish_profiling(Vec::new()), Err(BackendError::Profiling(_))));
    }
}
