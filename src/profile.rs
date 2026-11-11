//! GPU timing of kernel launches with timestamp queries.
//!
//! While profiling is enabled (see [`Context::profile`](crate::Context::profile)) every launch
//! runs in a compute pass of its own whose start and end the GPU timestamps, each launch into
//! timestamps of its own (no query is written twice). They are resolved only in
//! [`Context::finish_profiling`](crate::Context::finish_profiling), after all profiled work has
//! completed: Metal writes counter samples late enough that a resolve in the same command
//! buffer reads zeros or the samples of a later batch.

use std::fmt;
use std::sync::Arc;
use std::time::Duration;

use wgpu::{Buffer, BufferUsages};

/// Timestamps (a begin and an end per launch) of one query set; more launches get more sets.
const QUERIES: u32 = 512;

/// Bytes of one resolved timestamp.
pub(crate) const TIMESTAMP_BYTES: u64 = 8;

/// The GPU time one kernel took, summed over all of its launches in a profile.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KernelTiming {
    /// The label of the kernel: the one given to
    /// [`KernelBuilder::label`](crate::KernelBuilder::label), otherwise its entry point name.
    /// Built-in operations are labelled by operation and element type, for example
    /// `Fill f32`. Fused elementwise kernels (see the lazy arrays in [`Array`](crate::Array)) are
    /// labelled by the operations they contain, for example `Fused(Mul, Add, Sqrt) f32`, or
    /// `FusedReduce(Sum: Mul, Add) f32` for a reduction whose first pass computes the expression.
    pub label: String,
    /// How many launches were measured.
    pub launches: u32,
    /// Sum of the durations of all launches.
    pub total: Duration,
    /// The shortest single launch.
    pub min: Duration,
    /// The longest single launch.
    pub max: Duration,
}

impl KernelTiming {
    /// The average duration of a launch.
    pub fn mean(&self) -> Duration {
        self.total.checked_div(self.launches).unwrap_or_default()
    }
}

/// GPU durations per kernel, measured with timestamp queries; see
/// [`Context::profile`](crate::Context::profile).
///
/// A duration is the time from the start to the end of the compute pass of one launch, so it
/// includes the GPU's own scheduling of that pass but not the time between launches or any host
/// time.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Profile {
    /// One entry per kernel label, the one with the most total time first.
    pub kernels: Vec<KernelTiming>,
}

impl Profile {
    /// The entry of the kernel with this label.
    pub fn get(&self, label: &str) -> Option<&KernelTiming> {
        self.kernels.iter().find(|k| k.label == label)
    }

    /// Total GPU time of all measured launches.
    pub fn total(&self) -> Duration {
        self.kernels.iter().map(|k| k.total).sum()
    }

    /// Number of measured launches.
    pub fn launches(&self) -> u32 {
        self.kernels.iter().map(|k| k.launches).sum()
    }

    pub(crate) fn from_samples(samples: impl IntoIterator<Item = (Arc<str>, Duration)>) -> Self {
        let mut kernels: Vec<KernelTiming> = Vec::new();
        for (label, duration) in samples {
            match kernels.iter_mut().find(|k| k.label == *label) {
                Some(k) => {
                    k.launches += 1;
                    k.total += duration;
                    k.min = k.min.min(duration);
                    k.max = k.max.max(duration);
                }
                None => kernels.push(KernelTiming {
                    label: label.to_string(),
                    launches: 1,
                    total: duration,
                    min: duration,
                    max: duration,
                }),
            }
        }
        kernels.sort_by_key(|k| std::cmp::Reverse(k.total));
        Profile { kernels }
    }
}

impl fmt::Display for Profile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let width = self
            .kernels
            .iter()
            .map(|k| k.label.len())
            .max()
            .unwrap_or(0)
            .max(6);
        writeln!(
            f,
            "{:<width$} {:>9} {:>12} {:>12} {:>12} {:>12}",
            "kernel", "launches", "total us", "mean us", "min us", "max us"
        )?;
        let us = |d: Duration| d.as_secs_f64() * 1e6;
        for k in &self.kernels {
            writeln!(
                f,
                "{:<width$} {:>9} {:>12.2} {:>12.2} {:>12.2} {:>12.2}",
                k.label,
                k.launches,
                us(k.total),
                us(k.mean()),
                us(k.min),
                us(k.max)
            )?;
        }
        Ok(())
    }
}

/// The state of an enabled profile on a context.
#[derive(Debug)]
pub(crate) struct Profiler {
    /// Query sets in use, each with the labels of the launches it timed; launch `i` of a set
    /// owns timestamps `2 i` and `2 i + 1`.
    sets: Vec<(wgpu::QuerySet, Vec<Arc<str>>)>,
    /// The set the next launch is recorded into; moved to `sets` when full.
    current: Option<(wgpu::QuerySet, Vec<Arc<str>>)>,
}

impl Profiler {
    pub(crate) fn new() -> Self {
        Profiler { sets: Vec::new(), current: None }
    }

    /// Reserves the timestamps of a launch labelled `label` and returns the descriptor that makes
    /// the compute pass write them.
    pub(crate) fn writes(
        &mut self,
        device: &wgpu::Device,
        label: &Arc<str>,
    ) -> wgpu::ComputePassTimestampWrites<'_> {
        if self.current.as_ref().is_some_and(|(_, labels)| labels.len() * 2 >= QUERIES as usize) {
            self.sets.extend(self.current.take());
        }
        let (set, labels) = self.current.get_or_insert_with(|| {
            let set = device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("luma timestamps"),
                ty: wgpu::QueryType::Timestamp,
                count: QUERIES,
            });
            (set, Vec::new())
        });
        let begin = (labels.len() * 2) as u32;
        labels.push(Arc::clone(label));
        wgpu::ComputePassTimestampWrites {
            query_set: set,
            beginning_of_pass_write_index: Some(begin),
            end_of_pass_write_index: Some(begin + 1),
        }
    }

    /// Records the copy of every timestamp into one readback buffer; returns it with the labels
    /// of the launches, in timestamp order.
    pub(crate) fn resolve(
        self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
    ) -> (Buffer, Vec<Arc<str>>) {
        let sets: Vec<_> = self.sets.into_iter().chain(self.current).collect();
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
        (staging, all)
    }
}
