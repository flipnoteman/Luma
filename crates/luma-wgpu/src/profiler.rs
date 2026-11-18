//! GPU timing of kernel launches with timestamp queries.
//!
//! While profiling is enabled every launch runs in a compute pass of its own whose start and end
//! the GPU timestamps, each launch into timestamps of its own (no query is written twice). They
//! are resolved only after all profiled work that wrote them has completed (Metal writes counter
//! samples late enough that a resolve in the same command buffer reads zeros or the samples of a
//! later batch): when profiling finishes, and before that whenever recording a launch would put
//! more than `LIVE_SETS` query sets live at once, which some drivers lose the device over. The
//! durations are aggregated per kernel as they are read.

use std::sync::Arc;
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
