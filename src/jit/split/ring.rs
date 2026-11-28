//! The ring of a streamed stage: arrays that do not become resident. A streamed input is uploaded slab by slab into a ring of device
//! buffers by the transfer lane, a streamed output is downloaded slab by slab into host blocks (`State::Host` segments), and the
//! compute lane runs the stage's steps on the ring buffers in between, so upload, compute and download of neighbouring slabs overlap
//! where the backend has lanes (CUDA). Backends whose lanes are the defaults run the same loop synchronously.

use luma_core::hal::{size_class, HostBytes, Stage as HostStage, StageDir, Ticket};
use luma_core::Buffer;

use super::plan::Stage;
use super::run::{bind_all, launch_step};
use super::{Access, Open, Reach};
use crate::context::Context;
use crate::error::LumaError;
use crate::gpu::pager::Place;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Dir {
    /// Uploaded from host memory (or the file) into the ring.
    In,
    /// Downloaded from the ring into host blocks.
    Out,
}

/// An array a stage streams.
pub(super) struct Streamed {
    pub(super) array: u32,
    pub(super) dir: Dir,
    /// Per slab: whether it goes through the ring (an input whose segment is already on the device is bound by pin).
    pub(super) ring: Vec<bool>,
    /// Bytes of one ring buffer (a full segment, in the size class the device allocates).
    pub(super) buffer_bytes: u64,
}

/// What a stage streams, and with how many slabs in flight.
pub(super) struct Choice {
    pub(super) streams: Vec<Streamed>,
    pub(super) window: usize,
}

impl Choice {
    /// Device bytes of all ring buffers.
    pub(super) fn ring_bytes(&self) -> u64 {
        self.window as u64 * self.streams.iter().map(|s| s.buffer_bytes).sum::<u64>()
    }

    pub(super) fn streams_array(&self, array: u32) -> bool {
        self.streams.iter().any(|s| s.array == array)
    }
}

#[derive(Clone, Copy, Default)]
struct Use {
    read_far: bool,
    write: bool,
    in_place: bool,
}

/// Decides what `stage` streams, `None` when it runs the way kernel-major launches do (everything bound by pin, the pager making
/// room): when what it binds fits the budget, or nothing it binds can stream. Arrays are streamed by default and *retained* (made
/// resident, bound by pin) while they fit beside the ring, soonest next use first (D6). Two slabs in flight if they fit, else one.
pub(super) fn choose(ctx: &Context, open: &Open, stage: &Stage, next_stage: &dyn Fn(u32) -> u64) -> Option<Choice> {
    let budget = ctx.pager().budget()?;
    let space = stage.space(open);
    let max = ctx.max_buffer_bytes();
    let class = |bytes: u64| size_class(bytes, max).unwrap_or(bytes);
    let mut uses: Vec<Option<Use>> = vec![None; open.arrays.len()];
    for bind in open.steps[stage.steps.clone()].iter().flat_map(|step| step.binds.iter()) {
        let entry = uses[bind.array as usize].get_or_insert_with(Use::default);
        match bind.access {
            Access::Read => entry.read_far |= bind.reach != Reach::Point,
            Access::Write => entry.write = true,
            Access::ReadWrite => entry.in_place = true,
        }
    }
    let colours: u64 = stage.colours.iter().map(|&bytes| class(bytes)).sum();
    let mut whole = colours;
    // (next use, array, direction, ring slabs, ring buffer bytes, bytes of the whole array)
    let mut candidates: Vec<(u64, u32, Dir, Vec<bool>, u64, u64)> = Vec::new();
    for (index, usage) in uses.iter().enumerate() {
        let Some(usage) = usage else { continue };
        if stage.transients.iter().any(|t| t.array as usize == index) {
            continue;
        }
        let entry = &open.arrays[index];
        let bytes = entry.meta.layout.bytes(entry.meta.len);
        whole += bytes;
        if usage.read_far || usage.in_place || entry.meta.len != space.len || entry.meta.seg_len != space.seg_len {
            continue;
        }
        let places: Vec<Place> = entry.segments.iter().map(|segment| segment.place()).collect();
        let (dir, ring) = if usage.write {
            if !places.iter().all(|&place| place == Place::Empty) {
                continue;
            }
            (Dir::Out, vec![true; places.len()])
        } else {
            let ring: Vec<bool> = places.iter().map(|&place| matches!(place, Place::Host | Place::Away)).collect();
            if !ring.contains(&true) {
                continue;
            }
            (Dir::In, ring)
        };
        candidates.push((next_stage(index as u32), index as u32, dir, ring, class(entry.meta.layout.bytes(entry.meta.seg_len)), bytes));
    }
    if candidates.is_empty() || whole <= budget {
        return None;
    }
    candidates.sort_by_key(|&(next, array, ..)| (next, array));
    let mut chosen = None;
    for window in [2u64, 1] {
        if colours + window * candidates.iter().map(|c| c.4).sum::<u64>() <= budget {
            chosen = Some(window);
            break;
        }
    }
    let window = chosen?;
    let mut retained = vec![false; candidates.len()];
    let mut ring_sum: u64 = candidates.iter().map(|c| c.4).sum();
    let mut kept: u64 = 0;
    for (at, candidate) in candidates.iter().enumerate() {
        if colours + window * (ring_sum - candidate.4) + kept + candidate.5 <= budget {
            retained[at] = true;
            ring_sum -= candidate.4;
            kept += candidate.5;
        }
    }
    let streams: Vec<Streamed> = candidates
        .into_iter()
        .zip(retained)
        .filter(|&(_, retained)| !retained)
        .map(|((_, array, dir, ring, buffer_bytes, _), _)| Streamed { array, dir, ring, buffer_bytes })
        .collect();
    (!streams.is_empty()).then_some(Choice { streams, window: window as usize })
}

/// Waits for every ticket still outstanding when the loop ends, however it ends: the ring buffers and the host memory of the
/// transfers must not be freed while a transfer runs.
struct Outstanding<'a> {
    ctx: &'a Context,
    tickets: Vec<Ticket>,
}

impl Outstanding<'_> {
    fn finish(&mut self) -> Result<(), LumaError> {
        let mut result = Ok(());
        for ticket in self.tickets.drain(..) {
            if let Err(error) = self.ctx.ticket_wait(&ticket) {
                result = result.and(Err(error));
            }
        }
        result
    }
}

impl Drop for Outstanding<'_> {
    fn drop(&mut self) {
        let _ = self.finish();
    }
}

/// One streamed array's ring: `window` device buffers and the host memory of their transfers.
struct Ring {
    array: u32,
    dir: Dir,
    flags: Vec<bool>,
    bufs: Vec<Buffer>,
    hosts: Vec<HostStage>,
}

/// Rounds up to the 4 bytes transfers move in.
fn word(bytes: u64) -> u64 {
    bytes.next_multiple_of(4)
}

/// Runs the slabs of `stage` in `order` through the ring.
pub(super) fn run_slabs<'t>(ctx: &Context, open: &Open, stage: &Stage, choice: &Choice, order: &[usize], transient: &dyn Fn(u32) -> Option<&'t Buffer>) -> Result<(), LumaError> {
    let steps = &open.steps[stage.steps.clone()];
    let (window, n) = (choice.window, order.len());
    let mut rings: Vec<Ring> = Vec::with_capacity(choice.streams.len());
    for streamed in &choice.streams {
        let entry = &open.arrays[streamed.array as usize];
        let host_bytes = word(entry.meta.layout.bytes(entry.meta.seg_len));
        let dir = if streamed.dir == Dir::In { StageDir::ToDevice } else { StageDir::FromDevice };
        let mut ring = Ring { array: streamed.array, dir: streamed.dir, flags: streamed.ring.clone(), bufs: Vec::new(), hosts: Vec::new() };
        for _ in 0..window.min(n) {
            ring.bufs.push(ctx.lane_buffer(streamed.buffer_bytes)?);
            ring.hosts.push(ctx.stage_alloc(host_bytes, dir)?);
        }
        rings.push(ring);
    }
    let mut outstanding = Outstanding { ctx, tickets: Vec::new() };
    let mut up: Vec<Vec<Ticket>> = vec![Vec::new(); n];
    let mut down: Vec<Vec<Ticket>> = vec![Vec::new(); n];
    let mut done: Vec<Option<Ticket>> = vec![None; n];
    let (mut next_up, mut finalized) = (0, 0);

    // The host blocks of the slab at position `q`, once its download has completed.
    let finalize = |rings: &[Ring], down: &[Vec<Ticket>], q: usize| -> Result<(), LumaError> {
        let (slab, slot) = (order[q], q % window);
        for ring in rings.iter().filter(|ring| ring.dir == Dir::Out) {
            for ticket in &down[q] {
                ctx.ticket_wait(ticket)?;
            }
            let segment = &open.arrays[ring.array as usize].segments[slab];
            let bytes = segment.data_bytes() as usize;
            let mut block = HostBytes::zeroed(bytes);
            block.as_mut_slice().copy_from_slice(&ring.hosts[slot].as_slice()[..bytes]);
            segment.set_host(block)?;
        }
        Ok(())
    };

    for p in 0..n {
        let (slab, slot) = (order[p], p % window);
        // Uploads run ahead by `window - 1` slabs: a ring buffer is refilled once the compute of the slab it held has been submitted.
        while next_up < n && next_up < p + window {
            let (q, slot_q, slab_q) = (next_up, next_up % window, order[next_up]);
            for ring in rings.iter_mut().filter(|ring| ring.dir == Dir::In && ring.flags[slab_q]) {
                if q >= window {
                    // The host memory of the slot is free once the upload before it has completed.
                    for ticket in &up[q - window] {
                        ctx.ticket_wait(ticket)?;
                    }
                }
                let segment = &open.arrays[ring.array as usize].segments[slab_q];
                let bytes = word(segment.data_bytes());
                if !segment.stream_read(&mut ring.hosts[slot_q].as_mut_slice()[..bytes as usize])? {
                    return Err(LumaError::Gpu("a streamed input segment moved while its stage ran".into()));
                }
                let after: Vec<Ticket> = if q >= window { done[q - window].iter().cloned().collect() } else { Vec::new() };
                let ticket = ctx.transfer_in(&ring.hosts[slot_q], &ring.bufs[slot_q], bytes, &after)?;
                segment.note_streamed_in(segment.data_bytes());
                outstanding.tickets.push(ticket.clone());
                up[q].push(ticket);
            }
            next_up += 1;
        }
        // The slab computes after its uploads, and after the download that last read the ring slot it writes.
        let mut after = up[p].clone();
        if p >= window {
            after.extend(down[p - window].iter().cloned());
        }
        if !after.is_empty() {
            ctx.lane_barrier(&after)?;
        }
        {
            let fixed = |array: u32| {
                transient(array).or_else(|| rings.iter().find(|ring| ring.array == array && ring.flags[slab]).map(|ring| &ring.bufs[slot]))
            };
            let bound = bind_all(ctx, open, steps, &fixed);
            for (step, bound) in steps.iter().zip(&bound) {
                launch_step(ctx, step, bound, slab)?;
            }
        }
        let ticket = ctx.lane_barrier(&[])?;
        outstanding.tickets.push(ticket.clone());
        done[p] = Some(ticket);
        // Host memory of a slot is reused by the download `window` slabs later: its blocks are installed first.
        while finalized + window <= p {
            finalize(&rings, &down, finalized)?;
            finalized += 1;
        }
        for ring in rings.iter_mut().filter(|ring| ring.dir == Dir::Out) {
            let segment = &open.arrays[ring.array as usize].segments[slab];
            let bytes = word(segment.data_bytes());
            let after: Vec<Ticket> = done[p].iter().cloned().collect();
            let ticket = ctx.transfer_out(&ring.bufs[slot], &mut ring.hosts[slot], bytes, &after)?;
            outstanding.tickets.push(ticket.clone());
            down[p].push(ticket);
        }
    }
    outstanding.finish()?;
    while finalized < n {
        finalize(&rings, &down, finalized)?;
        finalized += 1;
    }
    Ok(())
}
