//! GPU timing of kernel launches with timestamp queries.
//!
//! While profiling is enabled every launch runs in a compute pass of its own whose start and end
//! the GPU timestamps, each launch into timestamps of its own (no query is written twice). They
//! are resolved only after all profiled work that wrote them has completed (Metal writes counter
//! samples late enough that a resolve in the same command buffer reads zeros or the samples of a
//! later batch): when profiling finishes, and before that whenever recording a launch would put
//! more than `LIVE_SETS` query sets live at once, which some drivers lose the device over. The
//! durations are aggregated per kernel as they are read.

use std::sync::{Arc, mpsc};
use std::time::Duration;

use luma_core::Profile;
use wgpu::BufferUsages;

/// Timestamps (a begin and an end per launch) of one query set; more launches get more sets.
const QUERIES: u32 = 512;

/// Most query sets alive at once; recording a launch that would need another first drains them.
/// Apple's Metal driver loses the device creating the 33rd set of 512 queries.
const LIVE_SETS: usize = 16;

/// Bytes of one resolved timestamp.
const TIMESTAMP_BYTES: u64 = 8;

/// The state of an enabled profile.
#[derive(Debug)]
pub(crate) struct Profiler {
    /// Query sets in use, each with the labels of the launches it timed; launch `i` of a set
    /// owns timestamps `2 i` and `2 i + 1`.
    sets: Vec<(wgpu::QuerySet, Vec<Arc<str>>)>,
    /// The set the next launch is recorded into; moved to `sets` when full.
    current: Option<(wgpu::QuerySet, Vec<Arc<str>>)>,
    /// The launches of the sets already resolved and released.
    timings: Profile,
}

impl Profiler {
    pub(crate) fn new() -> Self {
        Profiler { sets: Vec::new(), current: None, timings: Profile::default() }
    }

    /// Reserves the timestamps of a launch labelled `label` and returns the descriptor that makes
    /// the compute pass write them.
    pub(crate) fn writes(&mut self, device: &wgpu::Device, label: &Arc<str>) -> wgpu::ComputePassTimestampWrites<'_> {
        if self.current.as_ref().is_some_and(|(_, labels)| labels.len() * 2 >= QUERIES as usize) {
            self.sets.extend(self.current.take());
        }
        let (set, labels) = self.current.get_or_insert_with(|| {
            let set = device.create_query_set(&wgpu::QuerySetDescriptor { label: Some("luma timestamps"), ty: wgpu::QueryType::Timestamp, count: QUERIES });
            (set, Vec::new())
        });
        let begin = (labels.len() * 2) as u32;
        labels.push(Arc::clone(label));
        wgpu::ComputePassTimestampWrites { query_set: set, beginning_of_pass_write_index: Some(begin), end_of_pass_write_index: Some(begin + 1) }
    }

    /// Whether the next launch would open a query set while `LIVE_SETS` are already alive.
    pub(crate) fn at_capacity(&self) -> bool {
        let opens = self.current.as_ref().is_none_or(|(_, labels)| labels.len() * 2 >= QUERIES as usize);
        opens && self.sets.len() + usize::from(self.current.is_some()) >= LIVE_SETS
    }

    /// Adds the durations of resolved launches to the profile that [`Profiler::into_profile`]
    /// returns.
    pub(crate) fn add_samples(&mut self, samples: impl IntoIterator<Item = (Arc<str>, Duration)>) {
        self.timings.add_samples(samples);
    }

    /// The durations added so far.
    pub(crate) fn into_profile(self) -> Profile {
        self.timings
    }

    /// Records the copy of every timestamp of the query sets alive into one readback buffer and
    /// releases the sets; returns it with the labels of the launches, in timestamp order. `None`
    /// if no launch was timed.
    pub(crate) fn resolve(&mut self, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder) -> Option<(wgpu::Buffer, Vec<Arc<str>>)> {
        let sets: Vec<_> = self.sets.drain(..).chain(self.current.take()).collect();
        if sets.is_empty() {
            return None;
        }
        let total: u64 = sets.iter().map(|(_, labels)| labels.len() as u64 * 2).sum();
        let bytes = (total * TIMESTAMP_BYTES).max(TIMESTAMP_BYTES);
        let resolved = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("luma timestamps resolved"),
            size: bytes,
            usage: BufferUsages::QUERY_RESOLVE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("luma timestamps readback"),
            size: bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut offset = 0;
        let mut all = Vec::new();
        for (set, labels) in sets {
            let count = labels.len() as u32 * 2;
            encoder.resolve_query_set(&set, 0..count, &resolved, offset);
            offset += u64::from(count) * TIMESTAMP_BYTES;
            all.extend(labels);
        }
        encoder.copy_buffer_to_buffer(&resolved, 0, &staging, 0, bytes);
        Some((staging, all))
    }
}

/// Timestamp slots of the sampled launch timing (`Backend::time_next_launch`), kept apart from
/// the profiler's sets: timing never needs profiling mode.
const TIMER_SLOTS: usize = 8;

/// Where a timed launch is on its way from the batch to the caller.
#[derive(Debug)]
enum TimerState {
    Free,
    /// Recorded into the pending batch, which is not submitted yet.
    Recording(u64),
    /// In the submission `1`; its timestamps are only resolved once it completed (Metal writes
    /// the samples late, a resolve before that reads zeros).
    Submitted(u64, wgpu::SubmissionIndex),
    /// The resolved timestamps are being mapped for reading.
    Mapping(u64, mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>),
}

#[derive(Debug)]
struct TimerSlot {
    /// Two timestamps: the beginning and the end of the launch's pass.
    set: wgpu::QuerySet,
    resolved: wgpu::Buffer,
    staging: wgpu::Buffer,
    state: TimerState,
}

/// A ring of query slots, one per launch in flight; a launch armed while all are taken is not timed.
#[derive(Debug, Default)]
pub(crate) struct LaunchTimers {
    slots: Vec<TimerSlot>,
}

impl LaunchTimers {
    /// Takes a slot for the launch about to be recorded under `tag`; `None` if the ring is full.
    pub(crate) fn begin(&mut self, device: &wgpu::Device, tag: u64) -> Option<usize> {
        let index = match self.slots.iter().position(|slot| matches!(slot.state, TimerState::Free)) {
            Some(index) => index,
            None if self.slots.len() < TIMER_SLOTS => {
                let set = device.create_query_set(&wgpu::QuerySetDescriptor { label: Some("luma launch timing"), ty: wgpu::QueryType::Timestamp, count: 2 });
                let resolved = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("luma launch timing resolved"),
                    size: 2 * TIMESTAMP_BYTES,
                    usage: BufferUsages::QUERY_RESOLVE | BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                });
                let staging = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("luma launch timing readback"),
                    size: 2 * TIMESTAMP_BYTES,
                    usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                self.slots.push(TimerSlot { set, resolved, staging, state: TimerState::Free });
                self.slots.len() - 1
            }
            None => return None,
        };
        self.slots[index].state = TimerState::Recording(tag);
        Some(index)
    }

    /// The descriptor that makes the pass of the launch holding `slot` write its timestamps.
    pub(crate) fn writes(&self, slot: usize) -> wgpu::ComputePassTimestampWrites<'_> {
        wgpu::ComputePassTimestampWrites { query_set: &self.slots[slot].set, beginning_of_pass_write_index: Some(0), end_of_pass_write_index: Some(1) }
    }

    /// The batch holding every launch still `Recording` was submitted as `submission`.
    pub(crate) fn submitted(&mut self, submission: Option<&wgpu::SubmissionIndex>) {
        for slot in &mut self.slots {
            if let TimerState::Recording(tag) = slot.state {
                slot.state = match submission {
                    Some(index) => TimerState::Submitted(tag, index.clone()),
                    None => TimerState::Free,
                };
            }
        }
    }

    /// Moves the timings that are ready to `out`, never waiting: slots whose submission completed
    /// get their timestamps resolved and mapped, slots whose mapping completed are read and freed.
    /// A launch that is not done is reported by a later call.
    pub(crate) fn drain(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, out: &mut Vec<(u64, Duration)>) {
        if self.slots.iter().all(|slot| matches!(slot.state, TimerState::Free | TimerState::Recording(_))) {
            return;
        }
        // Hands the finished submissions' and mappings' callbacks over; never waits.
        if device.poll(wgpu::PollType::Poll).is_err() {
            return;
        }
        let period = f64::from(queue.get_timestamp_period());
        for slot in &mut self.slots {
            if let TimerState::Submitted(tag, index) = &slot.state {
                // A zero timeout is a poll: it only says whether the submission completed.
                if device.poll(wgpu::PollType::Wait { submission_index: Some(index.clone()), timeout: Some(Duration::ZERO) }).is_err() {
                    continue;
                }
                let tag = *tag;
                let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("luma launch timing") });
                encoder.resolve_query_set(&slot.set, 0..2, &slot.resolved, 0);
                encoder.copy_buffer_to_buffer(&slot.resolved, 0, &slot.staging, 0, 2 * TIMESTAMP_BYTES);
                queue.submit(Some(encoder.finish()));
                let (sender, receiver) = mpsc::channel();
                slot.staging.slice(..).map_async(wgpu::MapMode::Read, move |result| {
                    let _ = sender.send(result);
                });
                slot.state = TimerState::Mapping(tag, receiver);
            }
            if let TimerState::Mapping(tag, receiver) = &slot.state {
                let tag = *tag;
                match receiver.try_recv() {
                    Err(mpsc::TryRecvError::Empty) => {}
                    Ok(Ok(())) => {
                        let stamps: [u64; 2] = {
                            let view = slot.staging.slice(..).get_mapped_range();
                            match view {
                                Ok(view) => {
                                    let words: Vec<u64> = bytemuck::pod_collect_to_vec(&view);
                                    [words[0], words[1]]
                                }
                                Err(_) => [0, 0],
                            }
                        };
                        slot.staging.unmap();
                        // Equal stamps are samples the driver has not written; not a time.
                        if stamps[1] > stamps[0] {
                            out.push((tag, Duration::from_secs_f64((stamps[1] - stamps[0]) as f64 * period * 1e-9)));
                        }
                        slot.state = TimerState::Free;
                    }
                    Ok(Err(_)) | Err(mpsc::TryRecvError::Disconnected) => slot.state = TimerState::Free,
                }
            }
        }
    }
}
