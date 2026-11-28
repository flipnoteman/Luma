//! Borrowed views of the segments of an array: what a launch binds, whether the array is a [`Storage`] or only what a schedule
//! remembers of one (its segments).

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use crate::array::Storage;
use crate::context::Context;
use crate::error::Result;
use crate::gpu::pager::{Pin, Segment};
use crate::gpu_type::ElemLayout;

/// The shape of an array's segmentation: `len` elements of `layout` in segments of `seg_len` (the last one may be shorter).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct SegMeta {
    pub(crate) len: usize,
    pub(crate) layout: ElemLayout,
    pub(crate) seg_len: usize,
}

/// An array as launches see it: its shape and its segments, borrowed. [`Storage::as_seg`] and `ArrayRef::as_seg` make one; the eager
/// path and the executor of a schedule run the same code on it.
#[derive(Clone, Copy)]
pub(crate) struct SegRef<'a> {
    meta: SegMeta,
    segments: &'a [Arc<Segment>],
    ctx: &'a Context,
    /// The storage's flag that a segment was handed out for an eager command (pinning marks it; a schedule's arrays have none).
    materialized: Option<&'a AtomicBool>,
    /// Whether the one segment may be a window of another storage's buffer (a cheap check before the segment is locked).
    windowed: bool,
}

impl<'a> SegRef<'a> {
    pub(crate) fn new(ctx: &'a Context, meta: SegMeta, segments: &'a [Arc<Segment>], materialized: Option<&'a AtomicBool>, windowed: bool) -> Self {
        SegRef { meta, segments, ctx, materialized, windowed }
    }

    pub(crate) fn ctx(&self) -> &'a Context {
        self.ctx
    }

    /// Whether the one segment may be a window (see [`SegRef::window`]).
    pub(crate) fn windowed(&self) -> bool {
        self.windowed
    }

    pub(crate) fn meta(&self) -> SegMeta {
        self.meta
    }

    pub(crate) fn len(&self) -> usize {
        self.meta.len
    }

    pub(crate) fn layout(&self) -> ElemLayout {
        self.meta.layout
    }

    pub(crate) fn segments(&self) -> &'a [Arc<Segment>] {
        self.segments
    }

    /// Number of segments.
    pub(crate) fn segment_count(&self) -> usize {
        self.segments.len()
    }

    /// Bytes of data in the whole array.
    pub(crate) fn byte_len(&self) -> u64 {
        self.meta.layout.bytes(self.meta.len)
    }

    /// The elements segment `j` holds.
    pub(crate) fn range(&self, j: usize) -> std::ops::Range<usize> {
        let c = self.meta.seg_len;
        j * c..self.meta.len.min((j + 1) * c)
    }

    /// Bytes of data in segment `j`.
    pub(crate) fn segment_byte_len(&self, j: usize) -> u64 {
        self.meta.layout.bytes(self.range(j).len())
    }

    fn segment(&self, j: usize) -> &'a Arc<Segment> {
        if let Some(flag) = self.materialized {
            flag.store(true, Ordering::Relaxed);
        }
        &self.segments[j]
    }

    /// [`Storage::pin`].
    pub(crate) fn pin(&self, j: usize) -> Result<Pin> {
        self.segment(j).pin()
    }

    /// [`Storage::pin_read`].
    pub(crate) fn pin_read(&self, j: usize) -> Result<Pin> {
        self.segment(j).pin_read()
    }

    /// [`Storage::pin_overwrite`].
    pub(crate) fn pin_overwrite(&self, j: usize) -> Result<Pin> {
        self.segment(j).pin_overwrite()
    }

    /// [`Storage::window`]: the pack and the first element of this array in it, while its one segment only shows a range of it.
    pub(crate) fn window(&self) -> Option<(Arc<Storage>, usize)> {
        if !self.windowed || self.segments.len() != 1 {
            return None;
        }
        let (pack, offset) = self.segments[0].window()?;
        let at = usize::try_from(offset / pack.layout().stride() as u64).ok()?;
        Some((pack, at))
    }
}

/// What launches take as an array: a [`Storage`] (eager), or a [`SegRef`] of what a schedule holds.
pub(crate) trait AsSeg {
    fn as_seg(&self) -> SegRef<'_>;
}

impl AsSeg for SegRef<'_> {
    #[inline(always)]
    fn as_seg(&self) -> SegRef<'_> {
        *self
    }
}

impl AsSeg for Storage {
    #[inline(always)]
    fn as_seg(&self) -> SegRef<'_> {
        Storage::as_seg(self)
    }
}

impl AsSeg for Arc<Storage> {
    #[inline(always)]
    fn as_seg(&self) -> SegRef<'_> {
        Storage::as_seg(self)
    }
}

impl<T: AsSeg + ?Sized> AsSeg for &T {
    #[inline(always)]
    fn as_seg(&self) -> SegRef<'_> {
        (**self).as_seg()
    }
}
