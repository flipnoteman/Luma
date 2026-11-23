//! The batch of commands recorded since the last submission, and how it is submitted. The commands
//! themselves are the backend's: it keeps the open pass, the bind state and the uniform slots.

use luma_core::{Buffer, Dispatch, ExecGraph};

use crate::error::{LumaError, Result};
use crate::kernel::KernelCore;

use super::{lock, Context, RawArg};

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

/// A buffer bound twice to one launch, with at least one writable binding, is a validation
/// error on some backends (usage conflict) and a data race on the others. Arrays sharing a
/// buffer through `reshape` make that reachable.
fn check_aliasing(args: &[RawArg<'_>]) -> Result<()> {
    for (second, arg) in args.iter().enumerate() {
        let RawArg::Storage { buffer, offset, bytes, writable } = arg else { continue };
        for (first, earlier) in args.iter().enumerate().take(second) {
            let RawArg::Storage { buffer: other, offset: other_offset, bytes: other_bytes, writable: other_writable } = earlier else {
                continue;
            };
            // Disjoint ranges of one buffer do not alias.
            let overlap = *offset < other_offset + other_bytes && *other_offset < offset + bytes;
            if buffer == other && overlap && (*writable || *other_writable) {
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
        #[cfg(test)]
        if self.inner.assume_busy.load(std::sync::atomic::Ordering::Relaxed) {
            return true;
        }
        self.hal().gpu_busy()
    }

    /// Makes [`Context::submit`] and [`Context::submit_requested`] behave as if earlier
    /// submissions were still running, so that tests of postponed evaluation do not depend on
    /// how fast the GPU is.
    #[cfg(test)]
    pub(crate) fn set_assume_busy(&self, on: bool) {
        self.inner.assume_busy.store(on, std::sync::atomic::Ordering::Relaxed);
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
        // Looked at without anything held: a thread that holds back does not wait for the
        // ones that record or submit. One that submits in between only makes this flush a small batch.
        if self.hal().pending_dispatches() < SUBMIT_HOLD && self.gpu_busy() {
            return Ok(());
        }
        self.flush_batch()
    }

    /// Submits the pending batch now, busy GPU or not.
    pub(crate) fn submit_now(&self) -> Result<()> {
        self.flush_batch()
    }

    /// Like [`Context::submit`], but first computes what `eval` requested. While the GPU is busy
    /// that is postponed too, so requested arrays the program drops in the meantime are never
    /// computed and the steps between two submissions fuse. Postponing stops once resident data
    /// reaches three quarters of the memory budget.
    pub(crate) fn submit_requested(&self) -> Result<()> {
        if postpone(self.gpu_busy(), self.hal().pending_dispatches(), self.pager().under_pressure())
            && !self.jit().requested_stencil_chain()
        {
            return Ok(());
        }
        self.jit().evaluate_requested();
        self.flush_batch()
    }

    /// Number of dispatches recorded but not yet submitted.
    #[cfg(test)]
    pub(crate) fn pending_dispatches(&self) -> u32 {
        self.hal().pending_dispatches()
    }

    /// Tells the backend which buffers the pool let go of, so that it drops what it caches for
    /// them (the bind groups pinning them) with the next submission.
    pub(super) fn notify_released(&self) {
        let released = std::mem::take(&mut *lock(&self.inner.released));
        if !released.is_empty() {
            self.hal().notify_released(&released);
        }
        // Dropped only now: the buffers are freed once the backend let go of them.
        drop(released);
    }

    /// Submits the pending batch (if there is one) and prepares the next. Never waits.
    pub(super) fn flush_batch(&self) -> Result<()> {
        let released = std::mem::take(&mut *lock(&self.inner.released));
        if !released.is_empty() {
            self.hal().notify_released(&released);
        }
        let flushed = self.hal().flush_batch();
        // Everything recorded that uses these buffers is in the batch submitted above.
        drop(released);
        Ok(flushed?)
    }

    /// Records zeroing the first `bytes` bytes of `buffer`.
    pub(crate) fn clear(&self, buffer: &Buffer, bytes: u64) -> Result<()> {
        self.abort_recording()?;
        Ok(self.hal().record_clear(buffer, bytes)?)
    }

    /// Records copying `bytes` bytes from `src` at `src_offset` to `dst` at `dst_offset`, for
    /// gathering the partial results of several segments into one buffer.
    pub(crate) fn copy(&self, src: &Buffer, src_offset: u64, dst: &Buffer, dst_offset: u64, bytes: u64) -> Result<()> {
        self.abort_recording()?;
        Ok(self.hal().record_copy(src, src_offset, dst, dst_offset, bytes)?)
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
        kernel.check_count(args.len())?;
        kernel.check_kinds(args)?;
        if self.capture(kernel, workgroups, size, args) {
            return Ok(());
        }
        let pending = self.hal().record_launch(kernel, Dispatch::Direct(workgroups), size, args)?;
        if pending >= BATCH_LIMIT {
            self.flush_batch()?;
        }
        Ok(())
    }

    /// Like [`Context::launch`], but the workgroup counts are the three words of `args` at byte
    /// `offset`, read when the dispatch runs; `args` needs `BufferUsage::INDIRECT`.
    /// `max_workgroups` bounds those counts (for backends that size the grid up front).
    pub(crate) fn launch_indirect(
        &self,
        kernel: &KernelCore,
        args: &Buffer,
        offset: u64,
        size: [u32; 3],
        max_workgroups: [u32; 3],
        raw: &[RawArg<'_>],
    ) -> Result<()> {
        check_aliasing(raw)?;
        self.abort_recording()?;
        kernel.check_count(raw.len())?;
        kernel.check_kinds(raw)?;

        let dispatch = Dispatch::Indirect { buffer: args, offset, max_workgroups };
        let pending = self.hal().record_launch(kernel, dispatch, size, raw)?;
        if pending >= BATCH_LIMIT {
            self.flush_batch()?;
        }
        Ok(())
    }
}

impl Context {
    /// Whether GPU timing is enabled; profiled launches must run one by one, not as a graph.
    pub(crate) fn is_profiling(&self) -> bool {
        self.hal().is_profiling().unwrap_or(false)
    }

    /// Records a replay of a captured `graph` counting as `dispatches` dispatches.
    pub(crate) fn launch_graph(&self, graph: &dyn ExecGraph, dispatches: u32) -> Result<()> {
        self.abort_recording()?;
        let pending = self.hal().record_graph(graph, dispatches)?;
        if pending >= BATCH_LIMIT {
            self.flush_batch()?;
        }
        Ok(())
    }
}
