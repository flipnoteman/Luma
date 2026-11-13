//! The arena uniform values of a batch are packed into.

use wgpu::Buffer;

use super::batch::Prefix;
use super::{Context, UNIFORM_USAGE};

/// Size of one uniform arena chunk. Uniform values of a batch are packed into it, one
/// `min_uniform_buffer_offset_alignment` slot each.
const ARENA_CHUNK: u64 = 256 * 1024;

/// Bytes at the start of the batch's first uniform chunk that are copied by the batch's own
/// command buffer, as its first command. The slots of a handful of launches fit: the batch then
/// needs no command buffer of its own for uniforms (a second encoder, its `finish` and a second
/// command buffer in the submission cost the host about 2.5 us and delay the GPU by about 2 us
/// per submission). Slots beyond it go through the separate command buffer.
pub(super) const UNIFORM_PREFIX: u64 = 4096;

/// One buffer of the uniform arena together with the bytes that will be written into it.
#[derive(Debug)]
pub(super) struct Chunk {
    pub(super) buffer: Buffer,
    /// The arena epoch this chunk was started in.
    epoch: u64,
    /// Host copy of the slots handed out so far (`data.len()` is the used size).
    data: Vec<u8>,
    /// Bind group of `@group(0)` for this chunk (the grid uniform at a dynamic offset).
    pub(super) grid_group: Option<wgpu::BindGroup>,
}

/// Uniform values of the pending batch. Every value gets its own slot in a chunk that stays
/// untouched until the batch executed, so each launch sees the value it was given even though
/// the data is copied into the chunk (from a mapped staging buffer, once per chunk when the
/// batch is submitted) before the batch's commands run: the copies are recorded into a command
/// buffer of their own that goes first in the same submission. Slots are only reused after the
/// batch holding them was submitted; the copy into a reused slot is ordered after that
/// submission on the queue.
#[derive(Debug)]
pub(super) struct Arena {
    /// The chunk slots are currently handed out from.
    pub(super) current: Chunk,
    /// Chunks of this batch that filled up; dropped when the batch is submitted.
    full: Vec<Chunk>,
    /// Counts the chunks that were started; identifies `current` in the bound-state tracking.
    pub(super) epoch: u64,
}

fn new_chunk(device: &wgpu::Device, size: u64, epoch: u64) -> Chunk {
    Chunk {
        epoch,
        buffer: device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("luma uniforms"),
            size,
            usage: UNIFORM_USAGE,
            mapped_at_creation: false,
        }),
        data: Vec::new(),
        grid_group: None,
    }
}

/// Bytes a uniform value of `len` bytes occupies in the arena.
pub(super) fn slot_size(align: u64, len: usize) -> u64 {
    (len as u64).next_multiple_of(wgpu::COPY_BUFFER_ALIGNMENT).max(4).next_multiple_of(align)
}

impl Arena {
    pub(super) fn new(device: &wgpu::Device) -> Self {
        Arena { current: new_chunk(device, ARENA_CHUNK, 0), full: Vec::new(), epoch: 0 }
    }

    /// `true` if `bytes` more bytes of slots fit into the current chunk.
    pub(super) fn has_room(&self, bytes: u64) -> bool {
        self.current.data.len() as u64 + bytes <= self.current.buffer.size()
    }

    /// Starts a new chunk with room for at least `bytes` bytes; the old one is kept until the
    /// batch is submitted.
    pub(super) fn roll(&mut self, device: &wgpu::Device, bytes: u64) {
        let fresh = new_chunk(device, bytes.max(ARENA_CHUNK), self.epoch + 1);
        self.full.push(std::mem::replace(&mut self.current, fresh));
        self.epoch += 1;
    }

    /// Reserves a fresh slot holding `data` in the current chunk and returns its offset; the
    /// caller has made room for it ([`Arena::has_room`]). The bytes reach the GPU in
    /// [`Arena::upload`].
    pub(super) fn push(&mut self, align: u64, data: &[u8]) -> u64 {
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
    /// batch's own, and so run after everything submitted earlier.
    ///
    /// The first [`UNIFORM_PREFIX`] bytes of the chunk the batch started in are not part of
    /// them: `prefix` is the staging buffer the batch's own commands copy them out of (see
    /// [`Context::copy_prefix`]); they are written into it here.
    ///
    /// The staging buffers are the reusable mapped ones of [`Context::upload`] (pushed to
    /// `staged`, which the caller maps again after the submission), not `Queue::write_buffer`:
    /// that one has wgpu create a staging buffer per call, which allocates and maps a fresh
    /// block of host-visible memory whenever nothing else lives in it, and frees it again at
    /// the next poll, on every submission.
    pub(super) fn upload(
        &mut self,
        ctx: &Context,
        prefix: Option<Prefix>,
        staged: &mut Vec<Buffer>,
        released: &mut Vec<Buffer>,
    ) -> Option<wgpu::CommandBuffer> {
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
                ctx.fill(&prefix.staging, &chunk.data[..from]);
                *written = true;
            }
            let rest = &chunk.data[from..];
            if rest.is_empty() {
                continue;
            }
            match ctx.stage(rest) {
                Some(staging) => {
                    encoder
                        .get_or_insert_with(|| {
                            ctx.device().create_command_encoder(&wgpu::CommandEncoderDescriptor {
                                label: Some("luma uniforms"),
                            })
                        })
                        .copy_buffer_to_buffer(&staging, 0, &chunk.buffer, from as u64, rest.len() as u64);
                    staged.push(staging);
                }
                // No staging buffer to be had (the device is out of memory): `write_buffer`
                // brings its own and is ordered the same way.
                None => ctx.queue().write_buffer(&chunk.buffer, from as u64, rest),
            }
        }
        if let Some((prefix, written)) = prefix {
            if !written {
                ctx.fill(&prefix.staging, &[]);
            }
            staged.push(prefix.staging);
        }
        released.extend(self.full.drain(..).map(|chunk| chunk.buffer));
        self.current.data.clear();
        encoder.map(wgpu::CommandEncoder::finish)
    }
}
