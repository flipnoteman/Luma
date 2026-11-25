//! The sink a strategy's `lower` launches into. In wave 2 it always launches; W3-U6 adds a sink that appends to a schedule, and the
//! match in every method then grows one arm.

use super::telemetry::SampleTag;
use crate::context::{Context, RawArg, Scratch};
use crate::error::Result;
use crate::kernel::KernelCore;
use luma_core::Buffer;

/// Most launches of one execution that are timed; a ninth disarms the sample and discards it.
pub(crate) const MAX_TIMED_LAUNCHES: u8 = 8;

/// Where a lowered strategy sends its launches.
#[derive(Clone)]
pub(crate) struct Emit<'a> {
    ctx: &'a Context,
    sink: Sink,
    /// The execution is one of the sampled ones (see `telemetry`): its launches are timed.
    sample: Option<SampleTag>,
    /// Launches made while sampled.
    launches: u8,
}

#[derive(Clone, Copy)]
enum Sink {
    /// Launch now.
    Now,
}

impl<'a> Emit<'a> {
    /// An emitter that launches and times nothing: what callers outside the registry (`speculate`, `until`, readbacks) use.
    #[inline(always)]
    pub(crate) fn now(ctx: &'a Context) -> Self {
        Emit { ctx, sink: Sink::Now, sample: None, launches: 0 }
    }

    /// An emitter whose launches are timed under `tag`.
    pub(crate) fn sampled(ctx: &'a Context, tag: SampleTag) -> Self {
        Emit { ctx, sink: Sink::Now, sample: Some(tag), launches: 0 }
    }

    #[inline(always)]
    pub(crate) fn ctx(&self) -> &'a Context {
        self.ctx
    }

    /// [`Context::launch`], timed when this execution is sampled and the launch is not captured into a recording.
    #[inline]
    pub(crate) fn launch(&mut self, core: &KernelCore, groups: [u32; 3], size: [u32; 3], args: &[RawArg<'_>]) -> Result<()> {
        let tag = match self.sample {
            None => None,
            Some(_) => self.next_tag(),
        };
        match self.sink {
            Sink::Now => self.ctx.launch_timed(core, groups, size, args, tag),
        }
    }

    /// The tag of the next timed launch, or `None` once more than [`MAX_TIMED_LAUNCHES`] were made (which discards the sample).
    #[cold]
    fn next_tag(&mut self) -> Option<u64> {
        let sample = self.sample?;
        if self.launches >= MAX_TIMED_LAUNCHES {
            self.ctx.jit().telemetry.discard(sample);
            self.sample = None;
            return None;
        }
        let tag = sample.part(self.launches);
        self.launches += 1;
        Some(tag)
    }

    pub(crate) fn copy(&mut self, src: &Buffer, src_offset: u64, dst: &Buffer, dst_offset: u64, bytes: u64) -> Result<()> {
        match self.sink {
            Sink::Now => self.ctx.copy(src, src_offset, dst, dst_offset, bytes),
        }
    }

    #[allow(dead_code)]
    pub(crate) fn clear(&mut self, buffer: &Buffer, bytes: u64) -> Result<()> {
        match self.sink {
            Sink::Now => self.ctx.clear(buffer, bytes),
        }
    }

    /// Scratch buffers that live as long as the returned value.
    pub(crate) fn scratch(&self) -> Scratch<'a> {
        Scratch::new(self.ctx)
    }

    /// Takes over the launches `other` (a copy of this emitter handed to code that owns one) made.
    pub(crate) fn absorb(&mut self, other: &Emit<'_>) {
        self.sample = other.sample;
        self.launches = other.launches;
    }

    /// What the execution made, for `Telemetry::finish`: its sample and how many launches were timed.
    pub(crate) fn into_sample(self) -> Option<(SampleTag, u8)> {
        self.sample.map(|tag| (tag, self.launches))
    }
}

impl std::ops::Deref for Emit<'_> {
    type Target = Context;

    #[inline(always)]
    fn deref(&self) -> &Context {
        self.ctx
    }
}
