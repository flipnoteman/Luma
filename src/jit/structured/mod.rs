//! Structured values: the lazy value of an operation that is not an elementwise expression (L0 of the
//! unified planner IR). A structured value is made without recording anything; it is recorded into a
//! buffer when something needs it (see [`Structured::force`]), together with the others that are pending
//! where its family allows. Reductions (see [`crate::reduce`]) are the first family; later families plug
//! into [`Family`] instead of each keeping its own laziness.
//!
//! The family-neutral part lives here: the buffer once recorded, the nodes that stand for the value in
//! expressions (its leaves), and the bookkeeping that decides whom it may be recorded with (the thread
//! that made it, whether a recording was under way, whether it is being recorded). What a family adds
//! is in its payload ([`Comp`]).

use std::panic::Location;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex, OnceLock, Weak};

use super::{ElemTy, Node};
use crate::array::Storage;
use crate::builtin::{Finish, ReduceList};
use crate::context::Context;
use crate::error::LumaError;
use crate::gpu_type::ElemLayout;
use crate::reduce::{AxisGeom, PackRead};

/// A value that has not been recorded yet, one of a [`Family`]. Shared by the arrays that stand for it
/// (see `Repr::Structured`) and by the leaves that stand for it in expressions.
pub(crate) struct Structured {
    pub(crate) family: Family,
    /// The buffer once the value was recorded (or the error); one output in wave 1.
    result: OnceLock<std::result::Result<Arc<Storage>, Arc<LumaError>>>,
    /// The nodes that stand for the value in expressions (see [`Structured::leaf`]): they get the buffer when the value is
    /// recorded for any other reason, so that what reads them is a reader of that buffer (a write into it evaluates it first).
    leaves: Mutex<Vec<Weak<Node>>>,
    /// The thread that made the value: only that one's values are recorded together (see [`crate::jit`]'s `multireduce`).
    pub(crate) thread: std::thread::ThreadId,
    /// Whether it was made while this thread was recording a step, whose values are never taken along.
    pub(crate) in_recording: bool,
    /// Set while the value is being recorded.
    pub(crate) forcing: AtomicBool,
    /// What failures are reported as, and where the value was asked for.
    pub(crate) name: &'static str,
    pub(crate) location: &'static Location<'static>,
}

/// What a structured value is. W2-R0 renames `Reduce` to `Comprehension`.
pub(crate) enum Family {
    Reduce(Comp),
}

/// The payload of a value of the reduce family.
pub(crate) struct Comp {
    pub(crate) meta: Meta,
    pub(crate) run: RunState,
    pub(crate) origin: ReduceOrigin,
}

/// What predicates read without a lock.
pub(crate) struct Meta {
    pub(crate) ty: ElemTy,
    /// The length of the operand.
    pub(crate) len: usize,
    pub(crate) tag: OpTag,
    /// The lines of an axis reduction (see [`OpTag::ReduceAxis`]); `None` for a reduction of the whole array.
    pub(crate) geom: Option<AxisGeom>,
}

/// Which operation a value is.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum OpTag {
    /// A reduction of the whole array.
    Reduce,
    /// A reduction along lines, with one op: its buffer holds one value per line.
    ReduceAxis,
}

/// The strategy side bookkeeping of a value.
pub(crate) enum RunState {
    Reduce {
        wgsl: &'static str,
        /// Bytes of one element of the partial results (a `bool` reduces as a `u32` word).
        elem: u64,
        /// How the results are stored: the layout of the element type.
        layout: ElemLayout,
        /// Whether the operand only reads a buffer (the reduction of an array that is already evaluated), which no speculation of
        /// readbacks concerns.
        buffer: bool,
        /// The operations applied to the reduction's values (none for a plain reduction).
        finish: Finish,
        /// Set when the group this reduction was recorded in was not recorded in one kernel because that was not due or ready (see
        /// `Tuning::reduce_multi_after`): it is recorded alone, so that the others do not form other groups.
        solo: AtomicBool,
        /// Where the values are when they were recorded together with other reductions' (see [`PackRead`]), and the first one's index there.
        pack: OnceLock<(Arc<PackRead>, usize)>,
    },
}

/// Where a reduction came from: the operand and the ops. Temporary: W2-R0 moves them into the form.
pub(crate) struct ReduceOrigin {
    /// The node whose values the reduction reduces (the registered reader of the array it reduces, see `reduce::run_reduce`).
    pub(crate) operand: Arc<Node>,
    pub(crate) ops: ReduceList,
}

impl std::fmt::Debug for Structured {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.family {
            Family::Reduce(comp) => {
                let RunState::Reduce { finish, .. } = &comp.run;
                f.debug_struct("Structured").field("ops", &comp.origin.ops).field("steps", &finish.epilogue).field("forced", &self.result.get().is_some()).finish()
            }
        }
    }
}

impl Structured {
    #[inline(always)]
    pub(crate) fn new(family: Family, in_recording: bool, name: &'static str, location: &'static Location<'static>) -> Self {
        Structured {
            family,
            result: OnceLock::new(),
            leaves: Mutex::new(Vec::new()),
            thread: std::thread::current().id(),
            in_recording,
            forcing: AtomicBool::new(false),
            name,
            location,
        }
    }

    /// The payload of the reduce family, the only one for now.
    #[inline(always)]
    pub(crate) fn comp(&self) -> &Comp {
        match &self.family {
            Family::Reduce(comp) => comp,
        }
    }

    #[inline(always)]
    pub(crate) fn ty(&self) -> ElemTy {
        self.comp().meta.ty
    }

    /// The error of the operands' evaluation or of the recorded passes, if any.
    pub(crate) fn failure(&self) -> Option<&Arc<LumaError>> {
        self.comp().origin.operand.failure().or_else(|| self.result.get().and_then(|result| result.as_ref().err()))
    }

    /// The buffer of the value if it has been recorded (or the error), without recording it.
    pub(crate) fn result_if_recorded(&self) -> Option<&std::result::Result<Arc<Storage>, Arc<LumaError>>> {
        self.result.get()
    }

    /// Whether the value has not been recorded into a buffer yet.
    pub(crate) fn is_unforced(&self) -> bool {
        self.result.get().is_none()
    }

    /// A node standing for the value in expressions: evaluating it records the value.
    pub(crate) fn leaf(self: &Arc<Self>, ctx: &Context, len: usize) -> Arc<Node> {
        let leaf = super::deferred_value(ctx, len, self.ty(), Arc::clone(self), self.name, self.location);
        let mut leaves = crate::gpu::lock(&self.leaves);
        // Dead leaves (their expressions were dropped) are pruned as the list doubles.
        if leaves.len() >= 32 && leaves.len().is_power_of_two() {
            leaves.retain(|weak| weak.strong_count() > 0);
        }
        leaves.push(Arc::downgrade(&leaf));
        leaf
    }

    /// Records the value into a new buffer (once) and returns it. Nothing is read back.
    pub(crate) fn force(&self, ctx: &Context) -> &std::result::Result<Arc<Storage>, Arc<LumaError>> {
        let result = self.record_once(ctx);
        if let Ok(storage) = result {
            self.hand_over(storage);
        }
        result
    }

    /// [`Structured::force`] for the node that stands for the value: it takes the buffer itself.
    pub(crate) fn record_once(&self, ctx: &Context) -> &std::result::Result<Arc<Storage>, Arc<LumaError>> {
        self.record_with(|| match &self.family {
            Family::Reduce(_) => self.record_reduce(ctx),
        })
    }

    /// The buffer, recorded by `record` if there is none yet.
    pub(crate) fn record_with(&self, record: impl FnOnce() -> std::result::Result<Arc<Storage>, Arc<LumaError>>) -> &std::result::Result<Arc<Storage>, Arc<LumaError>> {
        self.result.get_or_init(record)
    }

    /// Takes `storage` as the result if there is none yet; whether it did.
    pub(crate) fn set_result(&self, storage: Arc<Storage>) -> bool {
        self.result.set(Ok(storage)).is_ok()
    }

    /// Takes the buffer `storage` recorded together with another value as the result, if there is none yet.
    pub(crate) fn adopt(&self, storage: Arc<Storage>) {
        if self.set_result(Arc::clone(&storage)) {
            self.hand_over(&storage);
        }
    }

    /// Gives the leaves that stand for the value in expressions its buffer.
    #[inline]
    fn hand_over(&self, storage: &Arc<Storage>) {
        for leaf in std::mem::take(&mut *crate::gpu::lock(&self.leaves)) {
            if let Some(leaf) = leaf.upgrade() {
                leaf.finish_shared(Arc::clone(storage));
            }
        }
    }

    /// The buffer of the value computed on the host, or the one already recorded; `None` if it cannot run on the host and has
    /// not been recorded.
    pub(crate) fn host_result(&self, ctx: &Context) -> Option<Arc<Storage>> {
        if self.result.get().is_none()
            && let Some(storage) = match &self.family {
                Family::Reduce(_) => crate::host::try_eval_reduced(self, ctx),
            }
        {
            let _ = self.set_result(storage);
        }
        self.result.get()?.as_ref().ok().cloned()
    }
}
