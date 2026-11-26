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

pub(crate) mod boundary;
pub(crate) mod contraction;
pub(crate) mod emit;
pub(crate) mod families;
pub(crate) mod opaque;
pub(crate) mod select;
pub(crate) mod strategy;
pub(crate) mod telemetry;

use super::comprehension::{Comprehension, OpTag};
use super::{ElemTy, Node};
use crate::array::Storage;
use crate::builtin::Finish;
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

/// What a structured value is. The comprehension family is every "multiply, then combine" operation that is lazy: reductions today,
/// the lazy library ops of W2-LAZY next.
pub(crate) enum Family {
    Comprehension(Comp),
}

/// The payload of a value of the comprehension family.
pub(crate) struct Comp {
    /// The form of the value (see [`Comprehension`]), what recognition, the rewrite engine and the cost function read, where
    /// predicates read `meta` without the lock. A reduction fills it the first time something asks (see `Structured::form`): its
    /// `meta`, `run` and builder say all of it, and a form built for every value cost +22% host instructions on `jit/reduce_build`
    /// (nothing reads it before the value is forced). The lazy products of W2-LAZY, whose form is their record, set it when they are
    /// made, and a forced value that holds nothing of its operands takes it. Boxed, so that a value that has no form pays a word.
    #[cfg_attr(not(test), expect(dead_code, reason = "read by `Structured::form`, which recognition (R1), the rewrite engine (R3) and `explain` call"))]
    pub(crate) form: Mutex<Option<Box<Comprehension>>>,
    /// What the hot predicates read without the lock.
    pub(crate) meta: Meta,
    pub(crate) run: RunState,
}

/// When a value launches: where it is made (`Now`: recorded when something forces it, as reductions are) or only when it is forced
/// (`AtForce`: recorded at its site, launched when used; W2-LAZY's lazy products).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Launch {
    Now,
    #[expect(dead_code, reason = "made by W2-LAZY's lazy sites")]
    AtForce,
}

/// What predicates read without a lock.
pub(crate) struct Meta {
    pub(crate) ty: ElemTy,
    /// The length of the operand.
    pub(crate) len: usize,
    /// Which operation the value is (and, for a reduction, its ops).
    pub(crate) tag: OpTag,
    /// The lines of an axis reduction (see [`OpTag::ReduceAxis`]); `None` for a reduction of the whole array.
    pub(crate) geom: Option<AxisGeom>,
    /// When the value launches.
    pub(crate) launch: Launch,
}

/// The strategy side bookkeeping of a value: what the form cannot say.
pub(crate) enum RunState {
    Reduce {
        /// The node whose values the reduction reduces (the registered reader of the array it reduces, see `reduce::run_reduce`); the
        /// form's input 0 is the same node.
        operand: Arc<Node>,
        wgsl: &'static str,
        /// Bytes of one element of the partial results (a `bool` reduces as a `u32` word).
        elem: u64,
        /// How the results are stored: the layout of the element type.
        layout: ElemLayout,
        /// Whether the operand only reads a buffer (the reduction of an array that is already evaluated), which no speculation of
        /// readbacks concerns.
        buffer: bool,
        /// The operations applied to the reduction's values (none for a plain reduction); mirrored as the form's epilogue.
        finish: Finish,
        /// Set when the group this reduction was recorded in was not recorded in one kernel because that was not due or ready (see
        /// `Tuning::reduce_multi_after`): it is recorded alone, so that the others do not form other groups.
        solo: AtomicBool,
        /// Where the values are when they were recorded together with other reductions' (see [`PackRead`]), and the first one's index there.
        pack: OnceLock<(Arc<PackRead>, usize)>,
    },
}

impl std::fmt::Debug for Structured {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.family {
            Family::Comprehension(comp) => {
                let RunState::Reduce { finish, .. } = &comp.run;
                f.debug_struct("Structured").field("tag", &comp.meta.tag).field("launch", &comp.meta.launch).field("steps", &finish.epilogue).field("forced", &self.result.get().is_some()).finish()
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

    /// The payload of the comprehension family, the only one for now.
    #[inline(always)]
    pub(crate) fn comp(&self) -> &Comp {
        match &self.family {
            Family::Comprehension(comp) => comp,
        }
    }

    #[inline(always)]
    pub(crate) fn ty(&self) -> ElemTy {
        self.comp().meta.ty
    }

    /// The error of the operands' evaluation or of the recorded passes, if any.
    pub(crate) fn failure(&self) -> Option<&Arc<LumaError>> {
        self.operand().failure().or_else(|| self.result.get().and_then(|result| result.as_ref().err()))
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
            Family::Comprehension(_) => self.record_reduce(ctx),
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
                Family::Comprehension(_) => crate::host::try_eval_reduced(self, ctx),
            }
        {
            let _ = self.set_result(storage);
        }
        self.result.get()?.as_ref().ok().cloned()
    }
}
