//! The batch of commands recorded since the last submission, and how it is submitted.

use std::hash::{Hash, Hasher};
use std::sync::{Arc, MutexGuard};

use wgpu::{BindGroupEntry, BindingResource, Buffer, BufferBinding};

use crate::error::{LumaError, Result};
use crate::gpu::hash::{FxHasher, FxMap};
use crate::kernel::KernelCore;
use crate::profile::Profiler;

use super::uniforms::{slot_size, Arena, Chunk, UNIFORM_PREFIX};
use super::{lock, Context};

/// Number of pending dispatches up to which [`Context::submit`] holds the batch back while the
/// GPU is busy. Each `queue.submit` costs the host 20-35 us and leaves a gap of 20-27 us on the
/// GPU, so a loop of one-kernel submissions is bound by that instead of by the kernels. Mandelbrot
/// loop, 256 submitted iterations, best of 25 (ms) at 1024x1024 / 4096x4096: hold 8 10.9 / 168.7,
/// 16 10.7 / 168.4, 32 10.6 / 168.2 (without holding: 15.3 / 175). 16 is within 2% of the best.
pub(super) const SUBMIT_HOLD: u32 = 16;

/// Whether `eval` leaves its request pending: the GPU has work to do anyway, the batch is small
/// and postponed arrays don't keep memory the budget is short of.
pub(super) fn postpone(busy: bool, dispatches: u32, pressure: bool) -> bool {
    busy && dispatches < SUBMIT_HOLD && !pressure
}

/// Number of dispatches after which the pending batch is submitted without waiting for a sync
/// point. Bounds how much work piles up in one command buffer.
pub(super) const BATCH_LIMIT: u32 = 256;

/// Bind groups cached per context before the cache is emptied (at the next flush).
const MAX_CACHED_BIND_GROUPS: usize = 2048;

/// Problem sizes whose grid uniform slot is remembered within a batch, so that launches with a
/// size seen before share its slot (and its bind group) instead of writing a new one.
const MAX_GRID_SLOTS: usize = 16;

/// One argument of a kernel launch, already validated: what to bind at `@group(1)` in order.
#[derive(Clone, Copy)]
pub(crate) enum RawArg<'a> {
    /// The first `bytes` bytes of a storage buffer.
    Storage { buffer: &'a Buffer, bytes: u64, writable: bool },
    /// A uniform value, copied at launch time.
    Uniform(&'a [u8]),
}

impl<'a> RawArg<'a> {
    pub(crate) fn read(buffer: &'a Buffer, bytes: u64) -> Self {
        RawArg::Storage { buffer, bytes, writable: false }
    }

    pub(crate) fn write(buffer: &'a Buffer, bytes: u64) -> Self {
        RawArg::Storage { buffer, bytes, writable: true }
    }

    /// What is bound: a buffer and how many of its bytes. Uniforms live in `uniforms`, the arena
    /// chunk this launch's uniform slots were allocated from.
    fn binding(&self, uniforms: &'a Buffer) -> (&'a Buffer, u64) {
        match *self {
            RawArg::Storage { buffer, bytes, .. } => (buffer, bytes),
            RawArg::Uniform(data) => (uniforms, data.len() as u64),
        }
    }
}

/// A bind group of `@group(1)` and what it was created for.
#[derive(Debug)]
struct CachedGroup {
    kernel: u64,
    /// What is bound, in binding order. Uniforms are identified by their arena chunk (the slot
    /// is a dynamic offset). Holding the buffers keeps them alive as long as the group is cached,
    /// so a buffer's identity cannot be reused by another one.
    entries: Box<[(Buffer, u64)]>,
    group: wgpu::BindGroup,
}

impl CachedGroup {
    fn matches<'a>(&self, kernel: u64, entries: impl ExactSizeIterator<Item = (&'a Buffer, u64)>) -> bool {
        self.kernel == kernel
            && self.entries.len() == entries.len()
            && self.entries.iter().zip(entries).all(|((a, a_bytes), (b, b_bytes))| a == b && *a_bytes == b_bytes)
    }
}

/// Bind groups of `@group(1)` by (kernel, bound buffers). Looked up with a hash computed from
/// borrowed handles, so a hit allocates and clones nothing.
#[derive(Debug, Default)]
pub(super) struct BindGroupCache {
    buckets: FxMap<u64, Vec<CachedGroup>>,
    len: usize,
}

impl BindGroupCache {
    /// Drops the groups binding any of `buffers`, so that the cache does not keep them alive.
    /// Invalidates indices into buckets: only call it while nothing is bound.
    pub(super) fn forget(&mut self, buffers: &[Buffer]) {
        self.buckets.retain(|_, bucket| {
            bucket.retain(|group| !group.entries.iter().any(|(b, _)| buffers.contains(b)));
            !bucket.is_empty()
        });
        self.len = self.buckets.values().map(Vec::len).sum();
    }
}

/// What the open compute pass has bound; launches skip what is already in place.
#[derive(Debug, Default, Clone, Copy)]
pub(super) struct Bound {
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
pub(super) struct Prefix {
    pub(super) staging: Buffer,
    pub(super) epoch: u64,
}

/// The commands recorded since the last submission.
#[derive(Debug)]
pub(super) struct Batch {
    /// The compute pass consecutive launches are recorded into (one pass per batch instead of
    /// one per launch). Commands that need the encoder itself end it first. Declared before the
    /// encoder so that it is dropped (ended) first.
    pub(super) pass: Option<wgpu::ComputePass<'static>>,
    pub(super) encoder: Option<wgpu::CommandEncoder>,
    /// Present once the batch has a dispatch and a staging buffer was available for [`Prefix`].
    pub(super) prefix: Option<Prefix>,
    pub(super) bound: Bound,
    pub(super) dispatches: u32,
    pub(super) arena: Arena,
    /// Slots of the grid uniform written to `arena.current` in this batch, by problem size.
    pub(super) grid_slots: Vec<([u32; 3], u32)>,
    pub(super) bind_groups: BindGroupCache,
    /// Dynamic offsets of the launch being recorded (kept to reuse its allocation).
    pub(super) offsets: Vec<u32>,
    /// Upload staging buffers copied from in this batch; mapped again once it is submitted.
    pub(super) uploads: Vec<Buffer>,
    /// Present while profiling is enabled.
    pub(super) profiler: Option<Profiler>,
}

/// Size of the `luma` grid uniform in bytes (`LumaGrid`).
const GRID_BYTES: u64 = 16;

fn dynamic_offset(offset: u64) -> Result<u32> {
    u32::try_from(offset).map_err(|_| LumaError::Gpu("uniform arena offset exceeds u32".into()))
}

/// A buffer bound twice to one launch, with at least one writable binding, is a wgpu validation
/// error (usage conflict). Arrays sharing a buffer through `reshape` make that reachable.
fn check_aliasing(args: &[RawArg<'_>]) -> Result<()> {
    for (second, arg) in args.iter().enumerate() {
        let RawArg::Storage { buffer, writable, .. } = arg else { continue };
        for (first, earlier) in args.iter().enumerate().take(second) {
            let RawArg::Storage { buffer: other, writable: other_writable, .. } = earlier else {
                continue;
            };
            if buffer == other && (*writable || *other_writable) {
                return Err(LumaError::KernelArgAlias { first, second });
            }
        }
    }
    Ok(())
}

impl Context {
    /// `true` while earlier submissions are still running on the GPU. A failed poll counts as
    /// idle, so that the submission goes ahead and the waiting paths report the error.
    fn gpu_busy(&self) -> bool {
        matches!(self.device().poll(wgpu::PollType::Poll), Ok(status) if !status.is_queue_empty())
    }

    /// Submits the pending batch without waiting for it, like [`Context::flush`] but without
    /// evaluating the lazy arrays that are still alive: only what was recorded so far goes to
    /// the GPU. For the fusion-limit cuts in `jit::create`.
    ///
    /// While the GPU is still busy with earlier submissions and fewer than [`SUBMIT_HOLD`]
    /// dispatches are pending, this does not submit at all: the GPU has work to do anyway, and
    /// every `queue.submit` costs the host 20-35 us and leaves a gap of about as much between
    /// two kernels. The held-back dispatches go out with the next call that submits,
    /// [`Context::flush`], anything that waits, or the batch limit.
    pub(crate) fn submit(&self) -> Result<()> {
        let mut batch = self.batch()?;
        if batch.dispatches < SUBMIT_HOLD && self.gpu_busy() {
            return Ok(());
        }
        self.flush_locked(&mut batch);
        Ok(())
    }

    /// Like [`Context::submit`], but first computes what `eval` requested. While the GPU is busy
    /// that is postponed too, so requested arrays the program drops in the meantime are never
    /// computed and the steps between two submissions fuse. Postponing stops once resident data
    /// reaches three quarters of the memory budget.
    pub(crate) fn submit_requested(&self) -> Result<()> {
        if postpone(self.gpu_busy(), self.batch()?.dispatches, self.pager().under_pressure()) {
            return Ok(());
        }
        self.jit().evaluate_requested();
        let mut batch = self.batch()?;
        self.flush_locked(&mut batch);
        Ok(())
    }

    /// Number of dispatches recorded but not yet submitted.
    #[cfg(test)]
    pub(crate) fn pending_dispatches(&self) -> u32 {
        self.batch().map(|b| b.dispatches).unwrap_or(0)
    }

    pub(super) fn take_gpu_error(&self) -> Result<()> {
        match lock(&self.inner.gpu_error).take() {
            Some(message) => Err(LumaError::Gpu(message)),
            None => Ok(()),
        }
    }

    pub(super) fn batch(&self) -> Result<MutexGuard<'_, Batch>> {
        self.inner.batch.lock().map_err(|_| LumaError::LockPoisoned)
    }

    /// Submits the pending batch (if there is one) and prepares the next. Never waits.
    pub(super) fn flush_locked(&self, batch: &mut Batch) -> Option<wgpu::SubmissionIndex> {
        // The open pass has to end before the encoder can be finished.
        batch.pass = None;
        batch.bound = Bound::default();
        batch.dispatches = 0;
        // The slots of the arena are reused by the next batch.
        batch.grid_slots.clear();
        let mut released = std::mem::take(&mut *lock(&self.inner.released));
        let prefix = batch.prefix.take();
        let uniforms = batch.arena.upload(self, prefix, &mut batch.uploads, &mut released);
        if batch.bind_groups.len > MAX_CACHED_BIND_GROUPS {
            // In-flight work keeps what it uses alive; this only drops our references.
            batch.bind_groups = BindGroupCache::default();
        } else if !released.is_empty() {
            batch.bind_groups.forget(&released);
        }
        // Everything recorded that uses these buffers is in the batch submitted below.
        let unmapped = std::mem::take(&mut *lock(&self.inner.unmapped));
        // The uniform copies go first: the batch's dispatches read what they write.
        let commands = batch.encoder.take().map(|encoder| encoder.finish());
        let submission = (uniforms.is_some() || commands.is_some())
            .then(|| self.queue().submit(uniforms.into_iter().chain(commands)));
        for staging in batch.uploads.drain(..) {
            let free = Arc::clone(&self.inner.uploads);
            let buffer = staging.clone();
            staging.slice(..).map_async(wgpu::MapMode::Write, move |result| {
                let mut free = lock(&free);
                free.in_flight = free.in_flight.saturating_sub(buffer.size());
                if result.is_ok() {
                    free.put(buffer);
                }
            });
        }
        for storage in unmapped {
            let free = Arc::clone(&self.inner.mappable);
            let released = Arc::clone(&self.inner.released);
            let buffer = storage.clone();
            storage.slice(..).map_async(wgpu::MapMode::Write, move |result| {
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

    fn new_encoder(&self) -> wgpu::CommandEncoder {
        self.device().create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("luma batch") })
    }

    /// Records copying the first [`UNIFORM_PREFIX`] bytes of the uniform chunk `chunk` (started in
    /// arena epoch `epoch`) out of a staging buffer that [`Arena::upload`] fills when the batch is
    /// submitted, so that the slots of a small batch need no command buffer of their own. Must
    /// be recorded before the batch's first dispatch. Without a staging buffer (the device is
    /// out of memory) everything goes through that separate buffer.
    fn copy_prefix(&self, encoder: &mut wgpu::CommandEncoder, prefix: &mut Option<Prefix>, chunk: &Buffer, epoch: u64) {
        if let Some(staging) = self.mapped_staging(UNIFORM_PREFIX) {
            lock(&self.inner.uploads).in_flight += staging.size();
            encoder.copy_buffer_to_buffer(&staging, 0, chunk, 0, UNIFORM_PREFIX);
            *prefix = Some(Prefix { staging, epoch });
        }
    }

    /// Records zeroing the first `bytes` bytes of `buffer`.
    pub(crate) fn clear(&self, buffer: &Buffer, bytes: u64) -> Result<()> {
        let mut guard = self.batch()?;
        self.encoder(&mut guard).clear_buffer(buffer, 0, Some(bytes));
        Ok(())
    }

    /// Records copying `bytes` bytes from `src` at `src_offset` to `dst` at `dst_offset`, for
    /// gathering the partial results of several segments into one buffer.
    pub(crate) fn copy(&self, src: &Buffer, src_offset: u64, dst: &Buffer, dst_offset: u64, bytes: u64) -> Result<()> {
        let mut guard = self.batch()?;
        self.encoder(&mut guard).copy_buffer_to_buffer(src, src_offset, dst, dst_offset, bytes);
        Ok(())
    }

    /// The batch's command encoder, for commands that are not dispatches. The open compute pass
    /// (which borrows the encoder) ends first, so the commands recorded here run after the
    /// launches recorded before and before those recorded after.
    pub(super) fn encoder<'a>(&self, batch: &'a mut Batch) -> &'a mut wgpu::CommandEncoder {
        batch.pass = None;
        batch.bound = Bound::default();
        batch.encoder.get_or_insert_with(|| self.new_encoder())
    }

    /// Records a dispatch of `kernel` and returns without waiting for it.
    ///
    /// `args` are bound to `@group(1)` in order; `size` is what `luma.size` is in the shader.
    /// Arguments must already be validated against the kernel's bindings.
    pub(crate) fn launch(
        &self,
        kernel: &KernelCore,
        workgroups: [u32; 3],
        size: [u32; 3],
        args: &[RawArg<'_>],
    ) -> Result<()> {
        check_aliasing(args)?;

        let mut guard = self.batch()?;
        self.record_launch(&mut guard, kernel, workgroups, size, args)?;
        if guard.dispatches >= BATCH_LIMIT {
            self.flush_locked(&mut guard);
        }
        Ok(())
    }

    /// The body of [`Context::launch`]: allocates the arena slots, finds or creates the bind
    /// group, and records the dispatch. After the first launch of a kernel with a given set of
    /// buffers this allocates nothing.
    fn record_launch(
        &self,
        batch: &mut Batch,
        kernel: &KernelCore,
        workgroups: [u32; 3],
        size: [u32; 3],
        args: &[RawArg<'_>],
    ) -> Result<()> {
        let inner = &*self.inner;
        let device = self.device();
        let align = inner.uniform_align;
        let Batch { encoder, prefix, pass, bound, dispatches, arena, grid_slots, bind_groups, offsets, profiler, .. } =
            batch;

        // Uniform values go into fresh arena slots, all of this launch's in one chunk. The grid
        // uniform shares the slot of an earlier launch of the same size.
        let uniform_bytes: u64 = args
            .iter()
            .map(|arg| match arg {
                RawArg::Uniform(data) => slot_size(align, data.len()),
                RawArg::Storage { .. } => 0,
            })
            .sum();
        let grid_bytes = slot_size(align, GRID_BYTES as usize);
        let mut grid_offset =
            grid_slots.iter().find(|(known, _)| *known == size).map(|&(_, offset)| offset);
        let needed = uniform_bytes + if grid_offset.is_some() { 0 } else { grid_bytes };
        if !arena.has_room(needed) {
            arena.roll(device, uniform_bytes + grid_bytes);
            // The remembered slots belong to the chunk that was left.
            grid_slots.clear();
            grid_offset = None;
        }
        let grid_offset = match grid_offset {
            Some(offset) => offset,
            None => {
                let offset = dynamic_offset(
                    arena.push(align, bytemuck::bytes_of(&[size[0], size[1], size[2], 0u32])),
                )?;
                if grid_slots.len() < MAX_GRID_SLOTS {
                    grid_slots.push((size, offset));
                }
                offset
            }
        };
        offsets.clear();
        for arg in args {
            if let RawArg::Uniform(data) = arg {
                offsets.push(dynamic_offset(arena.push(align, data))?);
            }
        }

        // Bind groups: the grid's per chunk, the arguments' per (kernel, buffers).
        let Chunk { buffer: uniforms, grid_group, .. } = &mut arena.current;
        let uniforms: &Buffer = uniforms;
        let grid_group = &*grid_group.get_or_insert_with(|| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &inner.grid_layout,
                entries: &[BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::Buffer(BufferBinding {
                        buffer: uniforms,
                        offset: 0,
                        size: std::num::NonZeroU64::new(GRID_BYTES),
                    }),
                }],
            })
        });

        let bindings = || args.iter().map(|arg| arg.binding(uniforms));
        let mut hasher = FxHasher::default();
        kernel.id.hash(&mut hasher);
        for (buffer, bytes) in bindings() {
            buffer.hash(&mut hasher);
            bytes.hash(&mut hasher);
        }
        let key = hasher.finish();
        let bucket = bind_groups.buckets.entry(key).or_default();
        let index = match bucket.iter().position(|cached| cached.matches(kernel.id, bindings())) {
            Some(index) => index,
            None => {
                let group_entries: Vec<BindGroupEntry<'_>> = bindings()
                    .enumerate()
                    .map(|(binding, (buffer, bytes))| BindGroupEntry {
                        binding: binding as u32,
                        resource: BindingResource::Buffer(BufferBinding {
                            buffer,
                            offset: 0,
                            size: std::num::NonZeroU64::new(bytes),
                        }),
                    })
                    .collect();
                let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout: &kernel.args_layout,
                    entries: &group_entries,
                });
                bucket.push(CachedGroup {
                    kernel: kernel.id,
                    entries: bindings().map(|(buffer, bytes)| (buffer.clone(), bytes)).collect(),
                    group,
                });
                bind_groups.len += 1;
                bucket.len().saturating_sub(1)
            }
        };
        let Some(args_group) = bucket.get(index).map(|cached| &cached.group) else {
            return Err(LumaError::Gpu("bind group cache entry vanished".into()));
        };

        // Record. Launches share one compute pass; a profiled launch gets a pass of its own that
        // carries the timestamp writes.
        let profiled = profiler.is_some();
        if profiled {
            *pass = None;
        }
        let encoder = encoder.get_or_insert_with(|| self.new_encoder());
        if *dispatches == 0 && prefix.is_none() {
            self.copy_prefix(encoder, prefix, uniforms, arena.epoch);
        }
        let pass_slot = pass;
        let pass = pass_slot.get_or_insert_with(|| {
            *bound = Bound::default();
            let timestamp_writes = profiler.as_mut().map(|p| p.writes(device, &kernel.label));
            encoder
                .begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes })
                .forget_lifetime()
        });
        if bound.kernel != Some(kernel.id) {
            pass.set_pipeline(&kernel.pipeline);
            bound.kernel = Some(kernel.id);
            bound.args = None;
        }
        let grid = Some((arena.epoch, grid_offset));
        if bound.grid != grid {
            pass.set_bind_group(0, grid_group, &[grid_offset]);
            bound.grid = grid;
        }
        // A group with uniforms is bound at new dynamic offsets by every launch.
        if !offsets.is_empty() || bound.args != Some((key, index)) {
            pass.set_bind_group(1, args_group, offsets.as_slice());
            bound.args = Some((key, index));
        }
        pass.dispatch_workgroups(workgroups[0], workgroups[1], workgroups[2]);
        if profiled {
            *pass_slot = None;
        }
        *dispatches += 1;
        Ok(())
    }
}
