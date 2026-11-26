//! Lazy evaluation and fusion of elementwise operations.
//!
//! Elementwise operations (operators, math functions, comparisons, `select`, `powf`, `clamp`,
//! and shifts as shifted reads of an evaluated array) do not run when they are written. They add
//! a [`Node`] to a graph of pending expressions, and
//! the resulting array is *lazy*. Everything that can fail on shapes, contexts or poisoned
//! operands is still checked when the node is created, so errors and their source locations
//! behave as they would without laziness.
//!
//! A lazy array is evaluated when something needs its buffer: [`Array::to_vec`](crate::Array::to_vec), `eval`, `write`,
//! reshaping, passing it to a kernel, a shift or convolution, a compound assignment into a
//! buffer, or [`Context::sync`]. Evaluating a node walks the expression behind it, assigns
//! every distinct input buffer a binding and every scalar a slot of one uniform, and generates a
//! single WGSL kernel for the whole expression ([`codegen`]). Reductions fold the expression into
//! their first pass instead, so `(&a * &b).sum()` never materializes the product.
//!
//! Kernels are cached per context by the *structure* of the expression (operations, types,
//! how values are wired); scalar values are uniforms and not part of the key, so
//! `&x * 2.0` and `&x * 3.0` share one pipeline ([`program`]). The extents of the arrays are
//! uniforms too (see [`params`]): the same expression on a grid of another shape is the same
//! kernel, unless an extent is a power of two or the array has `2^23` elements or more, which
//! change the index math.
//!
//! # Where fusion stops
//!
//! * An expression holds at most [`JitState::max_nodes`] operations (default
//!   [`DEFAULT_MAX_NODES`] = 16384, at most `MAX_NODES_LIMIT` = 16384; `LUMA_FUSION_MAX_NODES`
//!   overrides the default and [`fusion_limit`] a scope), nests at most `MAX_DEPTH` = 512 levels (256 in builds with debug assertions)
//!   (the flattening walk is recursive, so this bounds its stack) and reads as many distinct input
//!   buffers as the device can bind (the evaluated arrays beyond that are first copied into one
//!   buffer that they are windows of, see `pack`, which a kernel binds once). A new node that would exceed one of them evaluates its
//!   biggest (for the depth: its deepest) operand first, which splits long chains into several
//!   kernels. Profiles list every such cut with its reason ([`Decision`]).
//!   The biggest operand is not evaluated alone if other live arrays *of the same length* share pending operations with
//!   it (they read nodes that its expression contains): those are computed in the same
//!   multi-output kernel, as far as the limits allow. Arrays that advance in lock step (`x`, `y`,
//!   `count` of an iteration) then split into a few multi-output kernels at the limit instead of
//!   one single-output kernel per operand, each recomputing what the others compute. Live arrays
//!   that share nothing stay lazy, so a single chain, or a chain beside unrelated arrays,
//!   splits as before. Only if the expression is still too big afterwards is the next biggest
//!   operand evaluated.
//! * A program that would keep more values live at once than `Tuning::max_live` (register
//!   pressure; `LUMA_FUSION_MAX_LIVE`) is cut when its single root is evaluated: the values
//!   crossing the point in its middle half where fewest are live are evaluated first, in one
//!   multi-output kernel, and the rest is flattened again (up to four times).
//! * Operations that are the same (same operation, length and operands, scalars by value) are one
//!   node: `create` looks them up in a table of weak references that is dropped whenever a buffer
//!   is written (`JitState::note_write`), and skips nodes computed inside another kernel. The
//!   lookup is skipped while a node operand has no consumer yet (an equal operation would be one),
//!   and the masks of region assignments are remembered by their box (`MaskKey`). A
//!   fused kernel computes a shared node once; a pending node that an expression outside the one
//!   being flattened also reads is evaluated first only if computing it once more costs more than
//!   writing and reading it back (`worth_storing`, the rule of circular shifts; the work of an
//!   expression is at most 8 f32 operations per distinct node, however often the expressions
//!   below it share one; one of a quarter of the node limit is always stored), otherwise each
//!   expression computes it. Of those, only the outermost are evaluated, together with their
//!   siblings (pending nodes of the same length made by the same operation at the same place in
//!   the program: the other fields of a record, see `program::with_siblings`), so that a residual
//!   that a step update and a convergence norm both read is one multi-output kernel, not its
//!   first field alone and then every shared intermediate of the others, level by level.
//!   Arrays evaluated together (`eval_all`, `Fields::cache`) that read one another at the element
//!   itself are one multi-output kernel too; one that another reads through a view is computed
//!   first, since the reader would recompute it at every index it reads.
//! * A view or gather (slice, pad, flip, transpose, `take`, broadcast) of a pending expression
//!   that fits one buffer reads it lazily, as a sub-program computed at the read index, whatever
//!   its size and however often it is read. The kernel computes every (expression, index) pair
//!   once: nested views are written out at every index they are read at, a view of a view is
//!   one view, and equal instructions are one (see [`inline`]). Then the cost model
//!   ([`CostModel`]) decides per program (`inline::judge`), choosing in turn the one kernel that
//!   recomputes every read (the baseline), the one kernel that computes a sub-program once per
//!   cell of a workgroup tile in workgroup memory (below), and the evaluation of sub-programs
//!   read at several indices first (one kernel writing them and recomputing what they share
//!   with the rest, one reading them back), each if it is estimated to take less than 0.9
//!   (staging) or 0.75 (evaluating first) times the time of the choice before it
//!   ([`SplitReason::Materialised`] for the last). What to evaluate first is searched for, not
//!   guessed: of the sub-programs the program reads most often, the group (the fields of one
//!   state go together, or the program is still too big) whose two kernels cost least, the
//!   first of which is judged the same way, so that the stage state of a time step that reads
//!   the stage before it through views is evaluated once, not recomputed by every cell that
//!   reads it. A program that fits no plan costs more than any that does, ordered by what
//!   writing it out would cost. A kernel that would grow past
//!   `Tuning::max_inlined` instructions (`LUMA_FUSION_MAX_INLINED`) is cut whatever the model
//!   says, at the group that search finds ([`SplitReason::Inlined`]; the live arrays that share
//!   work with it are evaluated in the same kernel, except in a cut inside the evaluation of
//!   another cut's group: those include everything built on top of it), and expressions over arrays
//!   split over several buffers are read
//!   from their evaluated buffers. Circular shifts keep their own rule: a small expression is
//!   recomputed at up to eight offsets as long as one more computation (by its operation
//!   weights) costs less than writing it and reading it back, or staged in workgroup memory;
//!   the reader beyond that evaluates it (with the live arrays that share work with it, see
//!   `JitState::evaluate_alongside`). [`Array::cache`](crate::Array::cache) evaluates on
//!   request.
//! * Sub-programs read through translations of 2-D (or 1-D) arrays (slices, shifted slices of
//!   padded arrays: the flux a cell and its neighbour both read) at several offsets can be
//!   staged: a store kernel runs one workgroup per tile of output cells, computes each staged
//!   sub-program once per cell of its region (the tile plus the offsets reached, and what the
//!   sub-programs staged after it need) into workgroup memory, and the reads come from there
//!   (see `staging`, `LUMA_STAGING=off` to switch it off). The tile shrinks until the regions
//!   fit `Tuning::stencil_tile_bytes`. A sub-program that is read once (or that only views another
//!   one) is computed inside the programs that read it, so what it reads is staged at the sum of
//!   the offsets on the way: a ghost-padded record read at four offsets by a flux that a cell reads
//!   once is staged all the same. Sub-programs read through anything else (edge handling,
//!   gathers, strides) are recomputed per read as above, and a program over arrays split over
//!   several buffers is never staged.
//! * A `select` whose mask depends on the coordinates only (the mask of an `assign` into a region
//!   of a lazy array) is written as `if`/`else` when the expressions only one of its operands
//!   reads are worth at least 8 f32 operations per element: a ghost expression is computed in
//!   the ghost cells, not at every cell that reads the padded array. The cost model counts those
//!   operations by the share of cells that take the branch (see `inline::branches`).
//! * When an expression is evaluated, a pending node inside it that another pending, live
//!   operation outside the expression also reads is evaluated on its own first, instead of
//!   being recomputed in both kernels. Sharing within one expression is free: a shared node is
//!   one instruction. (`let t = &a * &b; let u = &t + 1.0; let v = &t * 2.0;` evaluates `t` once
//!   when `u` or `v` is read; if `v` is dropped first, `t` is simply part of `u`'s kernel.)
//!
//! # Hazards
//!
//! A lazy array behaves as if it had been computed when it was written down: it sees the
//! contents its input buffers had *then*. Buffers are overwritten by `write`, compound
//! assignment and custom kernels taking `&mut`; before any of those, every pending node that
//! reads the buffer is evaluated ([`Readers`]). Nodes hold their input buffers, so a buffer is
//! not recycled while a lazy array still needs it. Dropping a lazy array that was never used
//! costs nothing.
//!
//! # Assigning into a pending array
//!
//! `assign` through a slice, flip or transpose of an array that is still an expression needs no
//! buffer: the arrays holding the expression are made to denote `select(in_region, rhs, old)`
//! from then on (see [`supersede`] and `Array::assign_lazy`), so a padded state with its ghost
//! cells assigned stays one expression that fuses into the kernels reading it. `old` is the
//! replaced node itself, which keeps its value for expressions recorded before the write. The
//! replacement is owned by the claims arrays hold on a node ([`Held`]), never by the node, as the
//! right-hand side typically reads the node it replaces.
//!
//! # Stacking
//!
//! [`crate::stack`] joins equal-shaped expressions along a new axis in one kernel of the tail
//! `Tail::Stack`: an invocation computes its element of every part and writes them next to each
//! other (through workgroup memory when the new axis is the fastest, so that the stores are
//! coalesced), see [`stack`]. It is recorded when called and returns an evaluated array; what it
//! cannot write (arrays split over several buffers) it joins lazily like `concat`.
//!
//! # Writing in place
//!
//! A store kernel whose output replaces an input that nothing else holds (a dying temporary or
//! a solely owned array) reuses that buffer instead of allocating one, as long as the input is
//! read only at the element being written (not through a shift). Compound assignment on an array
//! whose buffer is shared is such a kernel too, forced onto the shared buffer.
//!
//! # Loops
//!
//! `eval` calls that advance the same arrays through the same step are a loop: the arrays of
//! the newest call become *loop leaves* that stand for the step applied to the previous ones, so
//! the pending graph stays one step deep and the postponed steps run at once (see [`looped`]).

mod bits;
pub(crate) mod assign;
mod codegen;
mod coarsen;
pub(crate) mod comprehension;
pub(crate) mod cost;
mod inline;
mod kirgen;
mod launch;
pub(crate) mod lineage;
mod follow;
mod looped;
mod operator;
mod multireduce;
pub(crate) mod structured;
mod pack;
mod params;
mod plancache;

/// Whether an array of `len` elements has indices that need the exact division (see `params::DivRemSpec`).
pub(crate) fn is_wide(len: usize) -> bool {
    params::is_wide(len as u64)
}
mod program;
mod readers;
mod rerun;
mod replay;
mod simplify;
pub(crate) mod rules;
pub(crate) mod speculate;
pub(crate) mod stack;
mod staging;
mod stencil;
mod until;
pub(crate) mod transpose;
#[cfg(test)]
mod tests;
#[cfg(test)]
mod plancache_tests;
#[cfg(test)]
mod follow_tests;

use std::panic::Location;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicU8, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock, Weak};

use crate::array::{Failure, Storage};
use crate::builtin::{BinOp, Class, CmpOp, Part, UnOp};
use crate::context::Context;
use crate::error::{LumaError, Result};
use crate::gpu::built::BuildCache;
use crate::gpu::hash::{FxHasher, FxMap};
use crate::gpu::lock;
use crate::gpu_type::{ElemLayout, ScalarType};
use crate::kernel::{self, KernelCore, KernelKey};
use crate::profile::{Decision, DecisionSite, SplitReason, Verdict as DecisionVerdict};
use luma_core::kir::KernelModule;

use inline::{inline, Judgement, Lowered, Verdict};
use looped::LoopPlan;
use program::Instr;
pub(crate) use program::AtomicOp;

pub(crate) use launch::{axis_plan, axis_reduce, axis_reduce_planned, AxisPlan, check_finite, evaluate, evaluate_for_reuse, is_recomputed, evaluate_group, evaluate_into, force_group, matmul, matmul_with, radix_pass, reduce_first_pass, reduce_many, run_loop, solve, stored_along_k, ManyMode, MatmulKernel, MatmulSide, Member, PartSide};
pub(crate) use multireduce::unresolved_leaves;
pub(crate) use operator::{operator_source, Operator};
pub(crate) use readers::Readers;
pub(crate) use rerun::{stamp, Attempt, Mix, Site};
pub(crate) use replay::{force, force_stepped, force_traced, Forced, GraphBuilder, GraphCache, Plan, Stepper, Trace, PAYING_REPLAYS};
pub(crate) use stencil::plan as stencil_plan;
pub(crate) use until::{force_with_record, next_block, run_until, FoldGate, FIRST_BLOCK};
#[cfg(test)]
pub(crate) use until::{SEPARATE_GATE, SEPARATE_RECORD};

/// Length of the requested list at which entries of dropped or computed nodes are pruned.
const REQUESTED_PRUNE_AT: usize = 256;

/// The nodes of [`JitState::requested`], pruned of dropped and computed ones once the list
/// doubles.
#[derive(Debug, Default)]
struct RequestList {
    nodes: Vec<Weak<Node>>,
    prune_at: usize,
}

/// Most operations one fused kernel contains unless configured otherwise.
pub(crate) const DEFAULT_MAX_NODES: usize = 16384;
/// How many times too high the number of operations below an operation must be (it adds up what
/// its operands held when they were made, evaluated since or not) before `create` counts them
/// again.
const STALE_COUNT: usize = 10;
/// Upper bound of the configurable limit.
const MAX_NODES_LIMIT: usize = 16384;
/// Deepest nesting of one fused expression: the expression is walked recursively, so this bounds
/// the stack the walk needs whatever the node limit (an unoptimised build has frames several
/// times bigger, which a test thread's 2 MB does not hold at 512 levels).
const MAX_DEPTH: u16 = if cfg!(debug_assertions) { 256 } else { 512 };
/// Most input buffers of one fused kernel, whatever the device allows.
const MAX_INPUTS: usize = 256;
/// Environment variable overriding [`DEFAULT_MAX_NODES`] for new contexts. `1` makes every
/// operation its own kernel.
const MAX_NODES_ENV: &str = "LUMA_FUSION_MAX_NODES";
/// Environment variable lowering the most input buffers of a kernel below what the device
/// allows (to behave like a device with fewer bindings; it cannot raise the device's limit).
const MAX_INPUTS_ENV: &str = "LUMA_FUSION_MAX_INPUTS";
/// Set to `off` to disable speculation at readbacks (see [`speculate`]).
const SPECULATE_ENV: &str = "LUMA_SPECULATE";

thread_local! {
    /// The node limit [`fusion_limit`] set on this thread.
    static FUSION_LIMIT: std::cell::Cell<Option<usize>> = const { std::cell::Cell::new(None) };
}

/// Runs `f` with the fusion limit of lazy operations created on this thread set to `max_nodes`
/// operations per kernel (instead of the context's, 16384 unless `LUMA_FUSION_MAX_NODES`
/// changes it), restoring the previous limit afterwards. The value is clamped to `1..=16384`;
/// `1` makes every operation its own kernel. Only the limit at the time an operation is
/// *written* matters, so the expressions built inside `f` keep it when they are evaluated later.
///
/// ```
/// # fn main() -> luma::Result<()> {
/// let ctx = luma::Context::new()?;
/// let x = luma::Array::<f32>::from_slice_on(&ctx, &[1.0; 8], 8)?;
/// // Every operation is a kernel of its own.
/// let y = luma::fusion_limit(1, || ((&x + 1.0) * 2.0).sqrt());
/// assert_eq!(y.to_vec()?, vec![(4.0f32).sqrt(); 8]);
/// # Ok(()) }
/// ```
pub fn fusion_limit<R>(max_nodes: usize, f: impl FnOnce() -> R) -> R {
    struct Restore(Option<usize>);
    impl Drop for Restore {
        fn drop(&mut self) {
            FUSION_LIMIT.with(|limit| limit.set(self.0));
        }
    }
    let _restore = Restore(FUSION_LIMIT.with(|limit| limit.replace(Some(max_nodes))));
    f()
}

/// Element type of a lazy operation's operands and results.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum ElemTy {
    F32,
    U32,
    I32,
    /// A `bool`: a `u32` holding 0 or 1 in registers, one bit of a `u32` word in buffers (see
    /// `ElemLayout::PACKED`); the accessors `kirgen::Lower::load_in` / `kirgen::Lower::store` and the
    /// packed store of `kirgen::Build::packed` are where the layout is known.
    Bool,
    /// 8- and 16-bit integers: a `u32` (unsigned) or `i32` (signed) register holding the value
    /// (zero- or sign-extended, always in the range of the type), 4 or 2 elements to a `u32`
    /// word in buffers (`ElemLayout::BYTE`, `ElemLayout::HALF`). The arithmetic wraps: every
    /// operation that can leave the range narrows its result again (`kirgen::Lower::narrow`),
    /// and the accessors `kirgen::Lower::load_in` / `kirgen::Lower::store` and the word-wise
    /// store of `kirgen::Build::plain_words` are where the layout is known.
    U8,
    I8,
    U16,
    I16,
    /// Half-precision floats: [`half::f16`] and [`half::bf16`]. An `f32` register holding a value
    /// that is always exactly representable in the type (every operation computes in `f32` and
    /// rounds its result to the type again, to nearest even: `kirgen::Lower::round`), two
    /// elements to a `u32` word in buffers (`ElemLayout::HALF`, the bits of the host slice). The
    /// accessors `kirgen::Lower::load_in` / `kirgen::Lower::store` convert at the buffer.
    F16,
    BF16,
    /// 64-bit float (needs `luma_core::Features::SHADER_F64`).
    F64,
    /// Double-float: a `vec2<f32>` `(hi, lo)` with library arithmetic (see `builtin::df_library`).
    Df32,
    /// Complex single precision: a `vec2<f32>` `(re, im)` with the arithmetic of
    /// `builtin::complex_library`.
    C32,
}

impl ElemTy {
    /// The element type of the built-in arithmetic, if `T` is one. The one table that maps Rust
    /// types to `ElemTy`.
    pub(crate) fn try_of<T: 'static>() -> Option<ElemTy> {
        use std::any::TypeId;
        let id = TypeId::of::<T>();
        if id == TypeId::of::<f32>() {
            Some(ElemTy::F32)
        } else if id == TypeId::of::<u32>() {
            Some(ElemTy::U32)
        } else if id == TypeId::of::<i32>() {
            Some(ElemTy::I32)
        } else if id == TypeId::of::<bool>() {
            Some(ElemTy::Bool)
        } else if id == TypeId::of::<u8>() {
            Some(ElemTy::U8)
        } else if id == TypeId::of::<i8>() {
            Some(ElemTy::I8)
        } else if id == TypeId::of::<u16>() {
            Some(ElemTy::U16)
        } else if id == TypeId::of::<i16>() {
            Some(ElemTy::I16)
        } else if id == TypeId::of::<half::f16>() {
            Some(ElemTy::F16)
        } else if id == TypeId::of::<half::bf16>() {
            Some(ElemTy::BF16)
        } else if id == TypeId::of::<f64>() {
            Some(ElemTy::F64)
        } else if id == TypeId::of::<crate::gpu_type::Df32>() {
            Some(ElemTy::Df32)
        } else if id == TypeId::of::<crate::complex::C32>() {
            Some(ElemTy::C32)
        } else {
            None
        }
    }

    pub(crate) fn of<T: ScalarType>() -> std::result::Result<ElemTy, LumaError> {
        Self::try_of::<T>().ok_or_else(|| LumaError::InvalidShape(format!("Unsupported element type {}", T::wgsl_type())))
    }

    pub(crate) fn wgsl(self) -> &'static str {
        match self {
            ElemTy::F32 | ElemTy::F16 | ElemTy::BF16 => "f32",
            ElemTy::U32 | ElemTy::Bool | ElemTy::U8 | ElemTy::U16 => "u32",
            ElemTy::I32 | ElemTy::I8 | ElemTy::I16 => "i32",
            ElemTy::F64 => "f64",
            ElemTy::Df32 | ElemTy::C32 => "vec2<f32>",
        }
    }

    /// Whether this is a half-precision float (see [`ElemTy::F16`]).
    pub(crate) fn is_half(self) -> bool {
        matches!(self, ElemTy::F16 | ElemTy::BF16)
    }

    /// Whether this is an 8-bit or 16-bit integer.
    pub(crate) fn is_small_int(self) -> bool {
        matches!(self, ElemTy::U8 | ElemTy::I8 | ElemTy::U16 | ElemTy::I16)
    }

    /// Whether elements of this type are narrower than a word: the 8-bit and 16-bit integers
    /// and the half-precision floats, which share words in buffers. Only the integers narrow
    /// their results ([`ElemTy::is_small_int`]); a float half never goes through the sign
    /// extension of [`ElemTy::is_small_signed`].
    pub(crate) fn is_small(self) -> bool {
        self.is_small_int() || self.is_half()
    }

    /// The type the arithmetic on elements of this type is done in: `F32` for the halves.
    pub(crate) fn compute(self) -> ElemTy {
        if self.is_half() { ElemTy::F32 } else { self }
    }

    /// Whether elements of this type share words in buffers: the bits of a `bool`, the small integers.
    pub(crate) fn is_packed(self) -> bool {
        self == ElemTy::Bool || self.is_small()
    }

    /// Bits of a small integer (0 for any other type).
    pub(crate) fn small_bits(self) -> u32 {
        match self {
            ElemTy::U8 | ElemTy::I8 => 8,
            ElemTy::U16 | ElemTy::I16 | ElemTy::F16 | ElemTy::BF16 => 16,
            _ => 0,
        }
    }

    /// `log2` of the small integers to a word: 2 for bytes, 1 for halves.
    pub(crate) fn small_shift(self) -> u32 {
        match self.small_bits() {
            8 => 2,
            16 => 1,
            _ => 0,
        }
    }

    /// Whether this is a signed small integer.
    pub(crate) fn is_small_signed(self) -> bool {
        matches!(self, ElemTy::I8 | ElemTy::I16)
    }

    /// How elements of this type sit in a buffer; it sets the elements per segment of an array
    /// (see `ElemLayout::stride`) and the bytes of a buffer (`ElemLayout::bytes`).
    pub(crate) fn layout(self) -> ElemLayout {
        match self {
            ElemTy::F32 | ElemTy::U32 | ElemTy::I32 => ElemLayout::plain(4),
            ElemTy::Bool => ElemLayout::PACKED,
            ElemTy::U8 | ElemTy::I8 => ElemLayout::BYTE,
            ElemTy::U16 | ElemTy::I16 | ElemTy::F16 | ElemTy::BF16 => ElemLayout::HALF,
            ElemTy::F64 | ElemTy::Df32 | ElemTy::C32 => ElemLayout::plain(8),
        }
    }

    /// `layout().stride()`: the bytes that set the elements per segment.
    pub(crate) fn stride(self) -> u64 {
        self.layout().stride() as u64
    }

    /// Bytes of memory traffic per element, for the cost model: 1/8 for a bit.
    pub(crate) fn traffic(self) -> f64 {
        match self {
            ElemTy::Bool => 0.125,
            ElemTy::U8 | ElemTy::I8 => 1.0,
            ElemTy::U16 | ElemTy::I16 | ElemTy::F16 | ElemTy::BF16 => 2.0,
            other => other.layout().stride() as f64,
        }
    }

    /// How many `f32` uniform slots a scalar of this type takes (see `Flattener::operand`).
    pub(crate) fn slots(self) -> usize {
        match self {
            ElemTy::F32 | ElemTy::U32 | ElemTy::I32 | ElemTy::Bool | ElemTy::U8 | ElemTy::I8 | ElemTy::U16 | ElemTy::I16 | ElemTy::F16 | ElemTy::BF16 => 1,
            ElemTy::Df32 | ElemTy::C32 => 2,
            ElemTy::F64 => 3,
        }
    }

    /// The name users know the type by, for error messages.
    pub(crate) fn display(self) -> &'static str {
        match self {
            ElemTy::F32 => "f32",
            ElemTy::U32 => "u32",
            ElemTy::I32 => "i32",
            ElemTy::Bool => "bool",
            ElemTy::U8 => "u8",
            ElemTy::I8 => "i8",
            ElemTy::U16 => "u16",
            ElemTy::I16 => "i16",
            ElemTy::F16 => "f16",
            ElemTy::BF16 => "bf16",
            ElemTy::F64 => "f64",
            ElemTy::Df32 => "Df32",
            ElemTy::C32 => "C32",
        }
    }
}

/// What a node computes per element.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum OpKind {
    Unary(UnOp),
    Binary(BinOp),
    /// Comparison; the result is a [`ElemTy::Bool`].
    Compare(CmpOp),
    /// Whether the floating-point operand is finite, infinite or `NaN`; the result is a [`ElemTy::Bool`].
    Classify(Class),
    /// `pow(x, exponent)`; operands: x, exponent (a scalar).
    Powf,
    /// `clamp(x, lo, hi)`; operands: x, lo, hi (scalars).
    Clamp,
    /// A real quantity of a complex operand (see [`Part`]); the result is an [`ElemTy::F32`].
    Part(Part),
    /// `complex(re, im)`; operands: re, im (of type `F32`; the op's type); the result is a [`ElemTy::C32`].
    Make,
    /// `cond != 0 ? a : b`; operands: condition (a `bool`, or any `u32` word: non-zero is true), a, b.
    Select,
    /// Circular shift of a materialized array; operand 0 is its buffer (see [`ShiftKind`]).
    Shift(ShiftKind),
    /// Converts operand 0 (of the op's type) to the given type.
    Cast(ElemTy),
    /// A value computed from the element's index alone; no array operands (see [`Gen`]).
    Generate(Gen),
}

/// What a [`OpKind::Generate`] computes for element `i` (its index in the whole array).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum Gen {
    /// Threefry-2x32-20 of counter `(i, 0)` under the key in operands 0 and 1: the first word.
    Bits,
    /// The top 24 bits of [`Gen::Bits`] scaled into `[0, 1)`.
    Uniform,
    /// Box-Muller on both Threefry words.
    Normal,
    /// The flat element index `i` itself (`range` along the sole active axis).
    Index,
    /// `(i >> shift) & mask` for power-of-two `stride` and `size`; operands: shift, mask.
    RangePow2,
    /// `(i / stride) % size` for other strides and sizes. The numbers are in the key, so a
    /// kernel that keeps this kind (a staged one, see `params`) works on constants. `wide`: the
    /// array has `2^23` elements or more (see `params::DivRemSpec`).
    RangeConst { stride: u32, size: u32, wide: bool },
    /// [`Gen::RangeConst`] with its numbers in the uniform: the [words of the
    /// divisors](params::divisor_words) of `stride` and `size` start at those slots (see
    /// `params::RangeKind` for the ones that are used). `axis`: in a lowered program, the axis of
    /// the source of the view of the context the range is computed in that has this stride and
    /// extent, so that the range is the context's coordinate along it.
    ParamRange { kind: params::RangeKind, wide: bool, stride: u32, size: u32, axis: Option<u8> },
    /// `start + i * step`; operands: start, step.
    Linspace,
}

/// Which element a shift reads for element `i`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum ShiftKind {
    /// `(i - s) mod n` over the flat array; `s` (in `0..n`) is the scalar operand 1 and `n` (the
    /// array length) the scalar operand 2.
    Linear,
    /// Every dimension of an array of `dims` shifted circularly by `shifts` (each in
    /// `0..dim`). The numbers are in the key, so a kernel that keeps this kind (a staged one, see
    /// `params`) works on constants.
    Dims { dims: [u32; 4], shifts: [u32; 4] },
    /// A [`ViewMap`] of the source with `edge` handling coordinates outside it. Operand 1 is
    /// the fill value of [`Edge::Fill`].
    View { map: ViewMap, edge: Edge },
    /// Element `i` reads element `min(indices[i], len - 1)` of the flat source of `len`
    /// elements; operand 1 is the `u32` index array.
    Take { len: u32 },
    /// [`ShiftKind::View`] with its numbers in the uniform (see `params`); `id` names the view
    /// among those of the kernel.
    ParamView { id: u16, spec: params::ViewSpec },
    /// [`ShiftKind::Take`] with `len - 1` in uniform slot `max_slot`.
    ParamTake { max_slot: u32 },
    /// [`ShiftKind::Dims`] with its numbers in the uniform (see `params::DimsAxis`).
    ParamDims { axes: [params::DimsAxis; 4] },
}

/// One source axis of a view: source coordinate `start + step * o`, `o` = output coordinate along `from`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) struct AxisMap {
    pub(crate) from: u8,
    pub(crate) start: i32,
    pub(crate) step: i32,
}

/// Reads an array of dims `src` as an array of dims `out`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct ViewMap {
    pub(crate) src: [u32; 4],
    pub(crate) out: [u32; 4],
    pub(crate) axes: [AxisMap; 4],
}

impl std::hash::Hash for ViewMap {
    /// Nine words instead of the thirty-odd a field-by-field hash writes (every write waits for
    /// the one before it, and a view is made for nearly every other operation).
    #[inline]
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        let pair = |a: u32, b: u32| u64::from(a) | u64::from(b) << 32;
        state.write_u64(pair(self.src[0], self.src[1]));
        state.write_u64(pair(self.src[2], self.src[3]));
        state.write_u64(pair(self.out[0], self.out[1]));
        state.write_u64(pair(self.out[2], self.out[3]));
        for axis in &self.axes {
            state.write_u64(pair(axis.start as u32, axis.step as u32));
        }
        let from = self.axes.iter().fold(0u32, |word, axis| word << 8 | u32::from(axis.from));
        state.write_u32(from);
    }
}

/// What a view reads where the source coordinate leaves `0..src[axis]`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum Edge {
    /// Never out of range (checked when the view is built).
    None,
    /// The value of scalar operand 1.
    Fill,
    /// The nearest element.
    Clamp,
    /// Symmetric about the boundary face: `-1 -> 0`, `-2 -> 1`, `n -> n - 1`.
    Mirror,
    /// Circular.
    Wrap,
}

impl ViewMap {
    /// The view that reads an array of `dims` as itself.
    pub(crate) fn identity(dims: [u32; 4]) -> ViewMap {
        let mut axes = [AxisMap { from: 0, start: 0, step: 1 }; 4];
        for (a, axis) in axes.iter_mut().enumerate() {
            axis.from = a as u8;
        }
        ViewMap { src: dims, out: dims, axes }
    }

    /// Elements of the output.
    pub(crate) fn out_len(&self) -> usize {
        self.out.iter().map(|&d| d as usize).product()
    }

    /// The source coordinates `start` and `start + step * (out - 1)` of axis `a`, unclipped.
    pub(crate) fn ends(&self, a: usize) -> (i64, i64) {
        let axis = self.axes[a];
        let first = i64::from(axis.start);
        let last = first + i64::from(axis.step) * (i64::from(self.out[axis.from as usize]) - 1);
        (first, last)
    }

    /// Whether every coordinate read lies inside the source.
    pub(crate) fn in_range(&self) -> bool {
        (0..4).all(|a| {
            let (first, last) = self.ends(a);
            let n = i64::from(self.src[a]);
            (0..n).contains(&first) && (0..n).contains(&last)
        })
    }

    /// Whether distinct output elements read distinct source elements.
    pub(crate) fn is_injective(&self) -> bool {
        let mut seen = [false; 4];
        for axis in &self.axes {
            match seen.get_mut(axis.from as usize) {
                Some(slot) if !*slot && axis.step != 0 => *slot = true,
                _ => return false,
            }
        }
        self.in_range()
    }

    /// `outer` applied to the output of this view (`outer.src == self.out`).
    pub(crate) fn then(&self, outer: &ViewMap) -> ViewMap {
        let mut axes = self.axes;
        for axis in &mut axes {
            let b = axis.from as usize;
            let o = outer.axes[b];
            let (start, step) = (i64::from(axis.start), i64::from(axis.step));
            *axis = AxisMap {
                from: o.from,
                start: (start + step * i64::from(o.start)) as i32,
                step: (step * i64::from(o.step)) as i32,
            };
        }
        ViewMap { src: self.src, out: outer.out, axes }
    }

    /// Per source axis the inclusive range of coordinates read.
    pub(crate) fn src_box(&self, edge: Edge) -> [(u32, u32); 4] {
        self.src_box_in(edge, self.out.map(|d| (0, d.saturating_sub(1))))
    }

    /// [`ViewMap::src_box`] for the output coordinates in `region` (inclusive, per output axis) only.
    /// A superset of what is read where the edge folds coordinates back in.
    pub(crate) fn src_box_in(&self, edge: Edge, region: [(u32, u32); 4]) -> [(u32, u32); 4] {
        let mut out = [(0, 0); 4];
        for (a, slot) in out.iter_mut().enumerate() {
            let axis = self.axes[a];
            let (o_lo, o_hi) = region[axis.from as usize];
            let first = i64::from(axis.start) + i64::from(axis.step) * i64::from(o_lo);
            let last = i64::from(axis.start) + i64::from(axis.step) * i64::from(o_hi);
            let (lo, hi) = (first.min(last), first.max(last));
            let n = i64::from(self.src[a]);
            let top = n.max(1) - 1;
            let (lo, hi) = match edge {
                Edge::None => (lo, hi),
                Edge::Clamp | Edge::Fill => (lo.clamp(0, top), hi.clamp(0, top)),
                Edge::Mirror | Edge::Wrap if lo >= 0 && hi < n => (lo, hi),
                Edge::Mirror | Edge::Wrap => (0, top),
            };
            *slot = (lo.clamp(0, top) as u32, hi.clamp(0, top) as u32);
        }
        out
    }

    /// Whether every extent, offset and step fits the `i32` arithmetic of the generated index math.
    pub(crate) fn fits(&self) -> bool {
        let limit = i64::from(i32::MAX);
        let dims_ok = self.src.iter().chain(&self.out).all(|&d| i64::from(d) <= limit);
        dims_ok
            && self.axes.iter().all(|axis| {
                let span = i64::from(axis.step) * i64::from(self.out[axis.from as usize]);
                i64::from(axis.start).abs() <= limit && span.abs() <= limit && (i64::from(axis.start) + span).abs() <= limit
            })
    }
}

impl OpKind {
    pub(crate) fn arity(&self) -> usize {
        match self {
            OpKind::Unary(_) | OpKind::Cast(_) | OpKind::Classify(_) | OpKind::Part(_) | OpKind::Shift(ShiftKind::Dims { .. } | ShiftKind::ParamDims { .. } | ShiftKind::View { edge: Edge::None | Edge::Clamp | Edge::Mirror | Edge::Wrap, .. }) => 1,
            OpKind::Shift(ShiftKind::ParamView { spec, .. }) if spec.edge != Edge::Fill => 1,
            OpKind::Binary(_)
            | OpKind::Compare(_)
            | OpKind::Make
            | OpKind::Powf
            | OpKind::Shift(ShiftKind::View { edge: Edge::Fill, .. } | ShiftKind::ParamView { .. } | ShiftKind::Take { .. } | ShiftKind::ParamTake { .. }) => 2,
            OpKind::Shift(ShiftKind::Linear) | OpKind::Clamp | OpKind::Select => 3,
            OpKind::Generate(Gen::Index | Gen::RangeConst { .. } | Gen::ParamRange { .. }) => 0,
            OpKind::Generate(_) => 2,
        }
    }
}

/// An operation together with the element type of its operands.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub(crate) struct Op {
    /// A hash of `kind` and `ty`, made once: operations are hashed for every node, program
    /// and kernel lookup, and a view makes a long list of words to hash. (First, with the
    /// type, so that comparing and hashing operations reads one cache line.)
    fingerprint: u64,
    pub(crate) ty: ElemTy,
    pub(crate) kind: OpKind,
}

impl PartialEq for Op {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.fingerprint == other.fingerprint && self.ty == other.ty && self.kind == other.kind
    }
}

impl Eq for Op {}

impl std::hash::Hash for Op {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        state.write_u64(self.fingerprint);
    }
}

impl Op {
    pub(crate) fn new(kind: OpKind, ty: ElemTy) -> Self {
        use std::hash::{Hash, Hasher};
        let mut hasher = FxHasher::default();
        kind.hash(&mut hasher);
        ty.hash(&mut hasher);
        Op { kind, ty, fingerprint: hasher.finish() }
    }

    pub(crate) fn out_ty(&self) -> ElemTy {
        match self.kind {
            OpKind::Compare(_) | OpKind::Classify(_) => ElemTy::Bool,
            OpKind::Part(_) => ElemTy::F32,
            OpKind::Make => ElemTy::C32,
            OpKind::Cast(to) => to,
            _ => self.ty,
        }
    }

    /// Whether the value of this operation, computed in `f32`, is rounded to its half-precision
    /// element type (to nearest even) before anything reads it: every operation that can give a
    /// value the type does not hold. Not the ones whose result is one of their operands or an
    /// integer no larger than them (negation, absolute value, minimum, maximum, selection, the
    /// roundings to integers, ...), nor a cast that keeps the type.
    pub(crate) fn rounds_half(&self) -> bool {
        if !self.out_ty().is_half() {
            return false;
        }
        match self.kind {
            OpKind::Unary(op) => !matches!(
                op,
                UnOp::Neg | UnOp::Abs | UnOp::Id | UnOp::Floor | UnOp::Ceil | UnOp::Trunc | UnOp::Round | UnOp::Sign | UnOp::Conj
            ),
            OpKind::Binary(op) => !matches!(op, BinOp::Min | BinOp::Max | BinOp::Copysign),
            OpKind::Select | OpKind::Clamp | OpKind::Shift(_) => false,
            OpKind::Cast(to) => self.ty != to,
            _ => true,
        }
    }

    /// The element type operand `index` must have.
    pub(crate) fn operand_ty(&self, index: usize) -> ElemTy {
        match (&self.kind, index) {
            (OpKind::Shift(ShiftKind::View { .. } | ShiftKind::ParamView { .. }), 1) => self.ty,
            (OpKind::Select, 0) => ElemTy::Bool,
            (OpKind::Shift(ShiftKind::Linear), 1 | 2) | (OpKind::Shift(_), 1) => ElemTy::U32,
            (OpKind::Generate(Gen::Bits | Gen::Uniform | Gen::Normal | Gen::RangePow2), _) => ElemTy::U32,
            _ => self.ty,
        }
    }
}

impl Node {
    /// Whether the node is an unevaluated view: a pure index map of what it reads (a slice, flip,
    /// transpose, pad or non-circular shift), no arithmetic.
    pub(crate) fn is_pending_view(&self) -> bool {
        self.is_pending() && matches!(self.op.kind, OpKind::Shift(ShiftKind::View { .. }))
    }
}

/// A scalar operand: a uniform of the fused kernel, never part of its cache key. `bits` is the
/// value's bit pattern: the low 32 bits for a 4-byte type, `f64::to_bits` for `F64`, `hi` in the
/// low and `lo` in the high word for `Df32`.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Scalar {
    pub(crate) ty: ElemTy,
    pub(crate) bits: u64,
}

impl Scalar {
    pub(crate) fn f32(value: f32) -> Self {
        Scalar { ty: ElemTy::F32, bits: u64::from(value.to_bits()) }
    }

    pub(crate) fn u32(value: u32) -> Self {
        Scalar { ty: ElemTy::U32, bits: u64::from(value) }
    }

    pub(crate) fn of<T: ScalarType>(value: T) -> std::result::Result<Self, LumaError> {
        let ty = ElemTy::of::<T>()?;
        let bits = match ty.slots() {
            1 => {
                let word = bytemuck::pod_read_unaligned::<u32>(bytemuck::bytes_of(&value.into_repr()));
                match ty {
                    // The register of a half holds the value widened to `f32`.
                    ElemTy::F16 => u64::from(half::f16::from_bits(word as u16).to_f32().to_bits()),
                    ElemTy::BF16 => u64::from(half::bf16::from_bits(word as u16).to_f32().to_bits()),
                    _ => u64::from(word),
                }
            }
            _ => bytemuck::pod_read_unaligned::<u64>(bytemuck::bytes_of(&value.into_repr())),
        };
        // The register of a signed small integer holds the sign-extended value.
        let bits = if ty.is_small_signed() {
            let up = 32 - ty.small_bits();
            u64::from((((bits as u32) << up) as i32 >> up) as u32)
        } else {
            bits
        };
        Ok(Scalar { ty, bits })
    }

    /// The `f32` uniform slots (as bit patterns) this scalar occupies in a program: one for a
    /// 4-byte type, three for `F64` (`a + b + c` adds up to the value), two for `Df32` and `C32`.
    pub(crate) fn words(self) -> [u32; 3] {
        match self.ty {
            ElemTy::F32 | ElemTy::U32 | ElemTy::I32 | ElemTy::Bool | ElemTy::U8 | ElemTy::I8 | ElemTy::U16 | ElemTy::I16 | ElemTy::F16 | ElemTy::BF16 => [self.bits as u32, 0, 0],
            ElemTy::Df32 | ElemTy::C32 => [self.bits as u32, (self.bits >> 32) as u32, 0],
            ElemTy::F64 => {
                let v = f64::from_bits(self.bits);
                let a = v as f32;
                let r = v - f64::from(a);
                let b = r as f32;
                let c = (r - f64::from(b)) as f32;
                [a.to_bits(), b.to_bits(), c.to_bits()]
            }
        }
    }
}

/// An operand of a node.
#[derive(Clone, Debug)]
pub(crate) enum Src {
    /// A materialized array.
    Buffer(Arc<Storage>),
    /// A pending (or just evaluated) expression.
    Node(Arc<Node>),
    Scalar(Scalar),
}

impl Src {
    /// Filler for the operand slots an operation does not use.
    pub(crate) fn unused() -> Self {
        Src::Scalar(Scalar { ty: ElemTy::U32, bits: 0 })
    }
}

/// Up to three operands; slots beyond the operation's arity are [`Src::unused`].
pub(crate) type Args = [Src; 3];

type NodeResult = std::result::Result<Arc<Storage>, Arc<LumaError>>;

static NEXT_NODE_ID: AtomicU64 = AtomicU64::new(0);

/// The ids of the [`JitState`]s.
static NEXT_JIT_ID: AtomicU64 = AtomicU64::new(0);

/// One pending elementwise operation.
///
/// The fields the walks over an expression read (flattening, `create`, the sharing checks) come
/// first, so that a node costs few cache lines to visit; the big operation and what only
/// evaluating needs come after them.
#[repr(C)]
pub(crate) struct Node {
    /// Set once the node was evaluated: its buffer, or why that failed.
    result: OnceLock<NodeResult>,
    /// Set when the node was found to be a state of a repeating step (see [`looped`]): its value
    /// is output `j` of the plan, which runs all the steps at once, and the node released its
    /// own operands.
    looped: OnceLock<(Arc<LoopPlan>, u32)>,
    /// Unique per node.
    id: u64,
    /// Number of elements of the result (and of every array operand).
    len: usize,
    below_len: usize,
    /// The operations of the expression in f32-operation units (see [`Node::weight`]).
    weight: f64,
    /// What the node knows of the nodes and buffers below it, see [`Ids`]. A node made by trace
    /// following (see [`follow`]) has none until something asks for it ([`SPECIAL_LAZY`]).
    ids: IdsCell,
    /// Number of nodes this one is an operand of (a pre-filter for sharing checks).
    consumers: AtomicU32,
    /// How many array handles ([`Held`]) are on this node. A held node is never handed out again
    /// by the hash-consing table: a write through one array must not show in another.
    handles: AtomicU32,
    /// How many shifts, views and gathers read this node lazily, as a sub-program (see
    /// [`recompute_shifted`]).
    shifted_reads: AtomicU32,
    /// Levels of the expression below this node: 1 + the greatest of its operands' (0 for none),
    /// shifted operands included. Bounded by [`MAX_DEPTH`].
    depth: u16,
    /// Levels of shifted lazy expressions nested in the expression (0: none; see
    /// [`recompute_shifted`]).
    shift_depth: u8,
    /// What is unusual about the node, see [`SPECIAL_LOOPED`], [`SPECIAL_LAZY`], [`SPECIAL_UNREGISTERED`].
    /// Read first by every accessor of `ids`, in the place of the check of `looped` it had before.
    special: AtomicU8,
    /// Set once the node's value was computed inside another node's kernel (or a reduction).
    /// Flushing skips such nodes: they are usually the temporaries of a fused expression.
    fused: AtomicBool,
    /// Whether the node is in [`JitState::requested`].
    requested: AtomicBool,
    /// Set once a walk found no reduction that is not recorded (and no loop output) below this node: that stays so, as
    /// evaluating only removes them (see [`multireduce::unresolved_leaves`]).
    leaf_free: AtomicBool,
    /// The operands. Taken (and so released) when the node is evaluated.
    args: Mutex<Option<Args>>,
    op: Op,
    /// Pending nodes that read this node's result; handed to its storage when it is evaluated.
    readers: Readers,
    ctx: Context,
    /// Set when the node stands for a structured value that is recorded when the node is evaluated (see
    /// [`deferred_value`]).
    deferred: OnceLock<Arc<structured::Structured>>,
    /// What failures of this node are reported as.
    name: &'static str,
    location: &'static Location<'static>,
    /// The thread that created the node. The sweeps that evaluate "every lazy array" (`eval`,
    /// `flush`, `sync`, evaluating alongside) only take the nodes of the calling thread: another
    /// thread may be flattening the expression this very moment, or have a recording under way
    /// that must only capture its own launches. A node is still evaluated by whoever uses it.
    thread: std::thread::ThreadId,
    /// The trace the node was created in and its number there (0: none; see [`rerun`]).
    trace: u64,
}

/// The ids of the distinct nodes in the expression below a node (`below_len` of them) as a bit set
/// over `base..` (`words` words), or sorted `(word, bits)` pairs (bit `id & 63` of the entry for
/// word `id >> 6`) if they are spread too widely for that (`sparse`, `words` numbers); followed by
/// the ids of the distinct buffers the expression reads, sorted. Upper bounds once parts of the
/// expression have been evaluated (`create` counts the buffers exactly by walking the expression
/// whenever the bound is over the device's limit). One allocation for both.
struct Ids {
    ids: Box<[u64]>,
    /// The id that bit 0 of the first word stands for (a multiple of 64).
    base: u64,
    words: u32,
    sparse: bool,
}

impl From<Measure> for Ids {
    fn from(measured: Measure) -> Ids {
        Ids { ids: measured.ids, base: measured.base, words: measured.words, sparse: measured.sparse }
    }
}

/// The [`Ids`] of a node: written once more by [`follow`] for a node that was made without them,
/// before [`SPECIAL_LAZY`] is cleared (with release ordering) under the lock of that module. The
/// accessors of [`Node`] never read it while the flag is set.
struct IdsCell(std::cell::UnsafeCell<Ids>);

// SAFETY: see above: the cell is written only while `SPECIAL_LAZY` is set, by whoever holds the
// lock of `follow`, and read only after the flag was seen cleared.
unsafe impl Sync for IdsCell {}

impl IdsCell {
    fn new(ids: Ids) -> IdsCell {
        IdsCell(std::cell::UnsafeCell::new(ids))
    }
}

/// `Node::special`: the node is a state of a repeating step (`looped` is set).
const SPECIAL_LOOPED: u8 = 1;
/// `Node::special`: the ids were not made when the node was (see [`follow`]).
const SPECIAL_LAZY: u8 = 2;
/// `Node::special`: the node was made by [`follow`] and is not registered (with the readers of its
/// operands, the live list or the consumer counts) yet.
const SPECIAL_UNREGISTERED: u8 = 4;
/// `Node::special`: the node is on the readers of the buffers it reads (so that a write flushes
/// it) though it is not registered otherwise (see [`follow`]).
const SPECIAL_GUARDED: u8 = 8;

thread_local! {
    static THREAD: std::thread::ThreadId = std::thread::current().id();
}

/// The calling thread, without the reference count of `thread::current`.
fn current_thread() -> std::thread::ThreadId {
    THREAD.with(|thread| *thread)
}

/// An array's claim on a node (see [`Node::handles`]).
pub(crate) struct Held {
    node: Arc<Node>,
    link: OnceLock<Arc<Link>>,
}

/// What the claims on one node share: the node that replaced it as the value of the arrays
/// holding it, once an assignment did (see [`supersede`]). Kept by the claims, not by the node:
/// expressions recorded earlier keep reading the node and its old value, and the node that took
/// over may read them, so the graph owning the replacement would be a cycle.
#[derive(Default)]
struct Link {
    /// Held, so the node that took over is never handed out again to another array.
    next: OnceLock<Held>,
}

impl Held {
    pub(crate) fn new(node: Arc<Node>) -> Self {
        node.handles.fetch_add(1, Ordering::Relaxed);
        Held { node, link: OnceLock::new() }
    }

    fn link(&self) -> &Arc<Link> {
        self.link.get_or_init(|| Arc::new(Link::default()))
    }

    /// Another claim on the same node (and on whatever replaces it).
    pub(crate) fn share(&self) -> Held {
        let _ = self.end().link();
        let link = Arc::clone(self.link());
        self.node.handles.fetch_add(1, Ordering::Relaxed);
        Held { node: Arc::clone(&self.node), link: OnceLock::from(link) }
    }

    /// The claim on the node that holds the value of the arrays that were created holding this
    /// one: the end of the chain of [`supersede`]s (this claim if the node was never replaced).
    pub(crate) fn end(&self) -> &Held {
        let mut held = self;
        while let Some(next) = held.link.get().and_then(|link| link.next.get()) {
            held = next;
        }
        held
    }

    /// The node that holds the value of the arrays that were created holding this one.
    pub(crate) fn latest(&self) -> &Arc<Node> {
        &self.end().node
    }

    /// Whether another array holds a claim on the node too, or on a node that replaced it (see
    /// [`supersede`]): a view taken from the replacement writes into the value this array
    /// denotes, so rebinding the array to a new expression would not reach it. The chain holds
    /// one claim of its own on every replacement.
    pub(crate) fn is_shared(&self) -> bool {
        let mut held = self;
        loop {
            if held.node.handles.load(Ordering::Relaxed) > 1 {
                return true;
            }
            match held.link.get().and_then(|link| link.next.get()) {
                Some(next) => held = next,
                None => return false,
            }
        }
    }
}

impl Drop for Held {
    fn drop(&mut self) {
        self.node.handles.fetch_sub(1, Ordering::Relaxed);
    }
}

impl std::ops::Deref for Held {
    type Target = Arc<Node>;

    fn deref(&self) -> &Arc<Node> {
        &self.node
    }
}

impl std::fmt::Debug for Held {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.node.fmt(f)
    }
}

impl std::fmt::Debug for Node {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Node")
            .field("name", &self.name)
            .field("size", &self.size())
            .field("pending", &self.is_pending())
            .finish()
    }
}

impl Node {
    /// Operations in the expression behind this node, itself included.
    fn size(&self) -> usize {
        self.below_len() + 1
    }

    /// Distinct nodes in the expression below this one.
    fn below_len(&self) -> usize {
        if self.looped.get().is_some() { 0 } else { self.below_len }
    }

    /// What the node knows of what is below it, made first if the node was made without it (see
    /// [`follow`]).
    #[inline]
    fn ids(&self) -> &Ids {
        if self.special.load(Ordering::Acquire) & SPECIAL_LAZY != 0 {
            follow::make_ids(self);
        }
        self.ids_made()
    }

    /// [`Node::ids`] for a caller that found the flag cleared (or made them) a moment ago.
    #[inline(always)]
    fn ids_made(&self) -> &Ids {
        debug_assert!(self.special.load(Ordering::Relaxed) & SPECIAL_LAZY == 0, "the ids of a node were read before they were made");
        // SAFETY: the cell is written only while `SPECIAL_LAZY` is set (see `IdsCell`).
        unsafe { &*self.ids.0.get() }
    }

    /// The part of `ids` that holds the nodes below this one.
    #[inline]
    fn below_part(&self) -> &[u64] {
        if self.special.load(Ordering::Acquire) & (SPECIAL_LOOPED | SPECIAL_LAZY) != 0 {
            return self.below_part_special();
        }
        let ids = self.ids_made();
        ids.ids.get(..ids.words as usize).unwrap_or(&[])
    }

    #[cold]
    #[inline(never)]
    fn below_part_special(&self) -> &[u64] {
        let ids = self.ids();
        if self.special.load(Ordering::Acquire) & SPECIAL_LOOPED != 0 {
            return &[];
        }
        ids.ids.get(..ids.words as usize).unwrap_or(&[])
    }

    /// The first and last id that can be below this node (a superset), if there are any.
    fn below_span(&self) -> Option<(u64, u64)> {
        let part = self.below_part();
        let ids = self.ids_made();
        if ids.sparse {
            return Some((*part.first()? * 64, *part.get(part.len().checked_sub(2)?)? * 64 + 63));
        }
        (!part.is_empty()).then(|| (ids.base, ids.base + 64 * part.len() as u64 - 1))
    }

    /// Whether node `id` is below this one.
    pub(crate) fn below_contains(&self, id: u64) -> bool {
        let part = self.below_part();
        let ids = self.ids_made();
        if ids.sparse {
            let word = id >> 6;
            let pairs = part.as_chunks::<2>().0;
            let at = pairs.partition_point(|&[w, _]| w < word);
            return pairs.get(at).is_some_and(|&[w, bits]| w == word && bits >> (id & 63) & 1 == 1);
        }
        let Some(offset) = id.checked_sub(ids.base) else { return false };
        part.get((offset >> 6) as usize).is_some_and(|word| word >> (offset & 63) & 1 == 1)
    }

    /// Calls `f` with the ids of the distinct nodes below this one, ascending, until it says yes.
    fn below_any(&self, mut f: impl FnMut(u64) -> bool) -> bool {
        let part = self.below_part();
        let ids = self.ids_made();
        if ids.sparse {
            for &[word, mut bits] in part.as_chunks::<2>().0 {
                while bits != 0 {
                    if f(word * 64 + u64::from(bits.trailing_zeros())) {
                        return true;
                    }
                    bits &= bits - 1;
                }
            }
            return false;
        }
        for (k, &word) in part.iter().enumerate() {
            let (mut bits, first) = (word, ids.base + 64 * k as u64);
            while bits != 0 {
                if f(first + u64::from(bits.trailing_zeros())) {
                    return true;
                }
                bits &= bits - 1;
            }
        }
        false
    }

    /// Calls `f` with every word of 64 ids (its index, id / 64, and its bits) that has a node
    /// below this one, ascending.
    fn for_each_below_word(&self, mut f: impl FnMut(u64, u64)) {
        let part = self.below_part();
        let ids = self.ids_made();
        if ids.sparse {
            for &[word, bits] in part.as_chunks::<2>().0 {
                if bits != 0 {
                    f(word, bits);
                }
            }
            return;
        }
        for (k, &bits) in part.iter().enumerate() {
            if bits != 0 {
                f((ids.base >> 6) + k as u64, bits);
            }
        }
    }

    /// The ids of the distinct nodes below this one, sorted.
    fn below_vec(&self) -> Vec<u64> {
        let mut ids = Vec::with_capacity(self.below_len());
        self.below_any(|id| {
            ids.push(id);
            false
        });
        ids
    }

    /// Adds the nodes below this one to `set`, whose bit 0 stands for id `base`.
    fn mark_below(&self, set: &mut [u64], base: u64) {
        let part = self.below_part();
        let ids = self.ids_made();
        if ids.sparse {
            for &[word, bits] in part.as_chunks::<2>().0 {
                if let Some(slot) = set.get_mut((word - (base >> 6)) as usize) {
                    *slot |= bits;
                }
            }
            return;
        }
        // An empty set (a loop leaf) has no span: `base` was chosen without it.
        if part.is_empty() {
            return;
        }
        let at = ((ids.base - base) >> 6) as usize;
        bits::or_into(set.get_mut(at..).unwrap_or(&mut []), part);
    }

    /// Ids of the distinct buffers the expression reads, sorted.
    #[inline]
    fn inputs(&self) -> &[u64] {
        if self.special.load(Ordering::Acquire) & (SPECIAL_LOOPED | SPECIAL_LAZY) != 0 {
            return self.inputs_special();
        }
        let ids = self.ids_made();
        ids.ids.get(ids.words as usize..).unwrap_or(&[])
    }

    #[cold]
    #[inline(never)]
    fn inputs_special(&self) -> &[u64] {
        match self.looped.get() {
            Some((plan, _)) => &plan.input_ids,
            None => {
                let ids = self.ids();
                ids.ids.get(ids.words as usize..).unwrap_or(&[])
            }
        }
    }

    /// Levels of shifted lazy expressions in the expression (see [`recompute_shifted`]). The plan
    /// of a loop leaf evaluates its inputs, so it has none.
    fn shift_depth(&self) -> u8 {
        if self.looped.get().is_some() { 0 } else { self.shift_depth }
    }

    /// The operations of the expression in f32-operation units, an upper bound (see
    /// [`cost::op_weight`]) that counts a node an expression reads twice twice, up to
    /// [`MAX_AVERAGE_WEIGHT`] per distinct node. A loop leaf has none: its plan evaluates its inputs.
    pub(crate) fn weight(&self) -> f64 {
        if self.looped.get().is_some() { 0.0 } else { self.weight }
    }

    /// Levels of the expression below this node. The plan of a loop leaf evaluates its inputs,
    /// so it has none.
    fn depth(&self) -> u16 {
        if self.looped.get().is_some() { 0 } else { self.depth }
    }

    pub(crate) fn is_pending(&self) -> bool {
        self.result.get().is_none()
    }

    /// The buffer of a node that was evaluated successfully.
    pub(crate) fn evaluated(&self) -> Option<&Arc<Storage>> {
        match self.result.get() {
            Some(Ok(storage)) => Some(storage),
            _ => None,
        }
    }

    /// The error of a failed evaluation.
    pub(crate) fn failure(&self) -> Option<&Arc<LumaError>> {
        match self.result.get() {
            Some(Err(error)) => Some(error),
            _ => None,
        }
    }

    /// This node as an operand: its buffer if it was evaluated, itself otherwise.
    pub(crate) fn as_src(self: &Arc<Self>) -> std::result::Result<Src, Failure> {
        match self.result.get() {
            Some(Ok(storage)) => Ok(Src::Buffer(Arc::clone(storage))),
            Some(Err(error)) => Err(Failure::Inherited(Arc::clone(error))),
            None => Ok(Src::Node(Arc::clone(self))),
        }
    }

    /// Evaluates the node (once): records its fused kernel into the context's batch and returns
    /// the buffer it writes. Does not wait for the GPU.
    pub(crate) fn force(self: &Arc<Self>) -> &NodeResult {
        if self.result.get().is_none() {
            // A step that follows a tape ends here if this is all it computes (see [`follow`]).
            if !(self.ctx.jit().follow_on() && follow::escape(self)) {
                crate::host::try_eval_node(self);
            }
        }
        let mut ran = false;
        let result = self.result.get_or_init(|| {
            ran = true;
            self.compute()
        });
        if ran {
            self.settled();
        }
        result
    }

    /// Notes that the node stopped being pending (it got its result): see [`JitState::pending_nodes`].
    pub(crate) fn settled(&self) {
        self.ctx.jit().pending_nodes.fetch_sub(1, Ordering::Relaxed);
    }

    /// Unique per node.
    pub(crate) fn id(&self) -> u64 {
        self.id
    }

    /// The structured value this node stands for the value of, if it is such a node.
    pub(crate) fn deferred_value(&self) -> Option<&Arc<structured::Structured>> {
        self.deferred.get()
    }

    /// Whether the node is a state of a repeating step (see [`looped`]).
    pub(crate) fn is_looped(&self) -> bool {
        self.looped.get().is_some()
    }

    /// [`Node::finish_shared`] for a buffer computed on the host: the node's buffer, which is
    /// another one if a different thread evaluated the node meanwhile.
    pub(crate) fn finish_host(self: &Arc<Self>, storage: Arc<Storage>) -> Arc<Storage> {
        self.finish_shared(Arc::clone(&storage));
        self.evaluated().cloned().unwrap_or(storage)
    }

    fn compute(self: &Arc<Self>) -> NodeResult {
        let _computing = Computing::enter(self.id);
        let args = lock(&self.args).take();
        let outcome = match args {
            Some(args) => evaluate(self, args).map(Arc::new),
            None => match (self.looped.get(), self.deferred.get()) {
                (Some((plan, j)), _) => plan.output(&self.ctx, *j, self.name, self.location),
                (None, Some(value)) => value.record_once(&self.ctx).clone().map_err(Failure::Inherited),
                (None, None) => Err(Failure::New(LumaError::Gpu("lazy node evaluated recursively".into()))),
            },
        };
        match outcome {
            Ok(storage) => {
                if let Err(error) = check_finite(&self.ctx, &storage, self.op.out_ty()) {
                    return Err(Arc::new(LumaError::Op { op: self.name, location: self.location, source: Arc::new(error) }));
                }
                plancache::finished(self, &storage);
                storage.set_producer(self.id);
                storage.readers.absorb(&self.readers);
                Ok(storage)
            }
            // An operand failed earlier: its error already names the operation.
            Err(Failure::Inherited(error)) => Err(error),
            Err(Failure::New(error)) => Err(Arc::new(LumaError::Op {
                op: self.name,
                location: self.location,
                source: Arc::new(error),
            })),
        }
    }

    /// Records that the node's value, computed inside a kernel with several outputs, failed a
    /// check: the node's result is `error`, reported as an error of this operation.
    fn fail(&self, error: LumaError) {
        let failed = Arc::new(LumaError::Op { op: self.name, location: self.location, source: Arc::new(error) });
        if self.result.set(Err(failed)).is_ok() {
            self.settled();
            lock(&self.args).take();
        }
    }

    /// Stores `storage`, computed for this node inside a kernel with several outputs, as the
    /// node's result, unless another thread evaluated it meanwhile.
    fn finish(&self, storage: Storage) {
        self.finish_shared(Arc::new(storage));
    }

    /// [`Node::finish`] for a buffer that others hold too.
    pub(crate) fn finish_shared(&self, storage: Arc<Storage>) {
        if self.result.set(Ok(Arc::clone(&storage))).is_ok() {
            self.settled();
            plancache::finished(self, &storage);
            storage.set_producer(self.id);
            storage.readers.absorb(&self.readers);
            lock(&self.args).take();
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.len
    }

    pub(crate) fn op(&self) -> &Op {
        &self.op
    }

    pub(crate) fn context(&self) -> &Context {
        &self.ctx
    }

    pub(crate) fn depends_on(&self, storage_id: u64) -> bool {
        self.inputs().binary_search(&storage_id).is_ok()
    }

    /// Whether another pending node outside the group (the nodes `inside` tells by address) reads
    /// this node. Evaluating this node separately then beats recomputing it in every consumer's
    /// kernel.
    pub(crate) fn has_consumer_outside(&self, inside: impl Fn(*const Node) -> bool) -> bool {
        // Cheap filter: nodes with a single consumer cannot be shared.
        if self.consumers.load(Ordering::Relaxed) < 2 {
            return false;
        }
        // An expression that reads no buffer (the mask of a region, a coordinate) costs less to
        // recompute than to write and read again.
        if self.inputs().is_empty() {
            return false;
        }
        // Most readers are inside the group: tell by address and only upgrade the others.
        self.readers.any_pending(|weak| !inside(weak.as_ptr()))
    }

    /// A copy of the operands of a node that has not been evaluated.
    pub(crate) fn operands(&self) -> Option<Args> {
        lock(&self.args).clone()
    }

    /// Whether the node is pending and yields the elements of `storage` unchanged (a copy of it).
    pub(crate) fn is_copy_of(&self, storage: &Arc<Storage>) -> bool {
        self.is_pending()
            && self.looped.get().is_none()
            && self.len == storage.len()
            && matches!(self.op.kind, OpKind::Unary(UnOp::Id))
            && matches!(self.operands(), Some([Src::Buffer(source), ..]) if Arc::ptr_eq(&source, storage))
    }

    /// Whether the node has operands, which it does not once it is evaluated or being evaluated.
    pub(crate) fn has_operands(&self) -> bool {
        lock(&self.args).is_some()
    }
}

fn insert_sorted(ids: &mut Vec<u64>, id: u64) {
    if let Err(at) = ids.binary_search(&id) {
        ids.insert(at, id);
    }
}

/// A set of node ids as a bit set over the words of 64 ids the nodes it holds fall in: what
/// the nodes of a group of arrays below the node limit are counted in, and merged into, once per
/// candidate (a candidate's own set has the same shape, see [`Node::for_each_below_word`]).
#[derive(Default)]
struct IdSet {
    /// The index (id / 64) of `words[0]`.
    base: u64,
    words: Vec<u64>,
    len: usize,
}

impl IdSet {
    /// The set of the nodes below `node`, and `node` itself if `with_self`.
    fn of(node: &Node, with_self: bool) -> IdSet {
        let mut set = IdSet::default();
        set.add(node, with_self);
        set
    }

    fn len(&self) -> usize {
        self.len
    }

    fn word(&self, index: u64) -> u64 {
        index.checked_sub(self.base).and_then(|offset| self.words.get(usize::try_from(offset).ok()?)).copied().unwrap_or(0)
    }

    fn contains(&self, id: u64) -> bool {
        self.word(id >> 6) >> (id & 63) & 1 == 1
    }

    /// How many of the nodes below `node` are in the set, and how many are not.
    fn overlap(&self, node: &Node) -> (usize, usize) {
        let (mut shared, mut total) = (0usize, 0usize);
        node.for_each_below_word(|index, bits| {
            shared += (bits & self.word(index)).count_ones() as usize;
            total += bits.count_ones() as usize;
        });
        (shared, total - shared)
    }

    /// Adds the nodes below `node`, and `node` itself if `with_self`.
    fn add(&mut self, node: &Node, with_self: bool) {
        let span = node.below_span().map(|(first, last)| (first >> 6, last >> 6)).into_iter().chain(with_self.then_some((node.id >> 6, node.id >> 6)));
        let (mut lo, mut hi) = (u64::MAX, 0);
        for (first, last) in span {
            lo = lo.min(first);
            hi = hi.max(last);
        }
        if lo > hi {
            return;
        }
        if self.words.is_empty() {
            self.base = lo;
        } else if lo < self.base {
            let front = (self.base - lo) as usize;
            self.words.splice(0..0, std::iter::repeat_n(0, front));
            self.base = lo;
        }
        let needed = (hi - self.base + 1) as usize;
        if needed > self.words.len() {
            self.words.resize(needed, 0);
        }
        let (base, mut added) = (self.base, 0usize);
        let words = &mut self.words;
        node.for_each_below_word(|index, bits| {
            let slot = &mut words[(index - base) as usize];
            added += (bits & !*slot).count_ones() as usize;
            *slot |= bits;
        });
        if with_self && !self.contains(node.id) {
            self.words[((node.id >> 6) - base) as usize] |= 1 << (node.id & 63);
            added += 1;
        }
        self.len += added;
    }
}

/// How many elements of the sorted set `a` are in the sorted set `b`, and how many are not.
fn overlap(a: &[u64], b: &[u64]) -> (usize, usize) {
    let found = if a.len().saturating_mul(16) < b.len() {
        a.iter().filter(|id| b.binary_search(id).is_ok()).count()
    } else {
        // Two sorted lists of sizes alike: one pass over both.
        let (mut found, mut j) = (0, 0);
        for &id in a {
            while b.get(j).is_some_and(|&other| other < id) {
                j += 1;
            }
            match b.get(j) {
                Some(&other) if other == id => found += 1,
                Some(_) => {}
                None => break,
            }
        }
        found
    };
    (found, a.len() - found)
}

/// Merges the sorted `other` into the sorted `ids` (a set union), through `tmp`.
fn union_into(ids: &mut Vec<u64>, tmp: &mut Vec<u64>, other: &[u64]) {
    let (Some(&first), Some(&last)) = (other.first(), other.last()) else { return };
    // Operands are mostly nested or disjoint: nothing to interleave.
    if ids.last().is_none_or(|&top| top < first) {
        ids.extend_from_slice(other);
        return;
    }
    // Operands of one expression mostly read the same buffers, or a few more.
    if ids.as_slice() == other {
        return;
    }
    if let [id] = other {
        insert_sorted(ids, *id);
        return;
    }
    if ids.first().is_some_and(|&bottom| bottom > last) {
        tmp.clear();
        tmp.extend_from_slice(other);
        tmp.extend_from_slice(ids);
        std::mem::swap(ids, tmp);
        return;
    }
    tmp.clear();
    tmp.reserve(ids.len() + other.len());
    let (mut a, mut b) = (ids.as_slice(), other);
    while let (Some(&x), Some(&y)) = (a.first(), b.first()) {
        match x.cmp(&y) {
            std::cmp::Ordering::Less => {
                tmp.push(x);
                a = a.get(1..).unwrap_or(&[]);
            }
            std::cmp::Ordering::Greater => {
                tmp.push(y);
                b = b.get(1..).unwrap_or(&[]);
            }
            std::cmp::Ordering::Equal => {
                tmp.push(x);
                a = a.get(1..).unwrap_or(&[]);
                b = b.get(1..).unwrap_or(&[]);
            }
        }
    }
    tmp.extend_from_slice(a);
    tmp.extend_from_slice(b);
    std::mem::swap(ids, tmp);
}

/// Sets the bit of `id` in `set`, whose bit 0 stands for id `base`.
fn mark(set: &mut [u64], base: u64, id: u64) {
    let offset = id - base;
    if let Some(word) = set.get_mut((offset >> 6) as usize) {
        *word |= 1 << (offset & 63);
    }
}

/// Adds `id` to a set of ids kept as sorted `(word, bits)` pairs: bit `id & 63` of the entry for
/// word `id >> 6`.
fn add_id(set: &mut Vec<u64>, id: u64) {
    let (word, bit) = (id >> 6, 1u64 << (id & 63));
    // Ids are made in ascending order: the newest usually goes last.
    match set.len().checked_sub(2).and_then(|last| Some((last, *set.get(last)?))) {
        None => return set.extend([word, bit]),
        Some((_, top)) if top < word => return set.extend([word, bit]),
        Some((last, top)) if top == word => {
            if let Some(bits) = set.get_mut(last + 1) {
                *bits |= bit;
            }
            return;
        }
        Some(_) => {}
    }
    let at = set.as_chunks::<2>().0.partition_point(|&[w, _]| w < word);
    match set.get_mut(2 * at..2 * at + 2) {
        Some([w, bits]) if *w == word => *bits |= bit,
        _ => {
            set.splice(2 * at..2 * at, [word, bit]);
        }
    }
}

/// Merges the set `other` into `set` (both sorted `(word, bits)` pairs), through `tmp`.
fn merge_sets(set: &mut Vec<u64>, tmp: &mut Vec<u64>, other: &[u64]) {
    let (Some(&first), Some(&last)) = (other.first(), other.get(other.len().wrapping_sub(2))) else { return };
    // Operands are mostly nested or disjoint: nothing to interleave.
    match set.get(set.len().wrapping_sub(2)) {
        None => return set.extend_from_slice(other),
        Some(&top) if top < first => return set.extend_from_slice(other),
        Some(_) => {}
    }
    if set.first().is_some_and(|&bottom| bottom > last) {
        tmp.clear();
        tmp.extend_from_slice(other);
        tmp.extend_from_slice(set);
        std::mem::swap(set, tmp);
        return;
    }
    tmp.clear();
    tmp.reserve(set.len() + other.len());
    let (mut a, mut b) = (set.as_chunks::<2>().0, other.as_chunks::<2>().0);
    while let (Some(&[wx, bx]), Some(&[wy, by])) = (a.first(), b.first()) {
        match wx.cmp(&wy) {
            std::cmp::Ordering::Less => {
                tmp.extend([wx, bx]);
                a = a.get(1..).unwrap_or(&[]);
            }
            std::cmp::Ordering::Greater => {
                tmp.extend([wy, by]);
                b = b.get(1..).unwrap_or(&[]);
            }
            std::cmp::Ordering::Equal => {
                tmp.extend([wx, bx | by]);
                a = a.get(1..).unwrap_or(&[]);
                b = b.get(1..).unwrap_or(&[]);
            }
        }
    }
    tmp.extend(a.iter().flatten());
    tmp.extend(b.iter().flatten());
    std::mem::swap(set, tmp);
}

/// Working memory of [`measure`], kept per thread so that measuring allocates nothing but the
/// ids it returns.
#[derive(Default)]
struct Measured {
    inputs: Vec<u64>,
    below: Vec<u64>,
    decoded: Vec<u64>,
    tmp: Vec<u64>,
}

thread_local! {
    /// The roots of the expressions being flattened on this thread, outermost first.
    static FLATTENING: std::cell::RefCell<Vec<*const Node>> = const { std::cell::RefCell::new(Vec::new()) };
}

thread_local! {
    /// The ids of the nodes this thread is computing (see [`Node::force`]), outermost first.
    static COMPUTING: std::cell::RefCell<Vec<u64>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// Marks a node as being computed by this thread until it is dropped.
struct Computing;

impl Computing {
    fn enter(id: u64) -> Self {
        COMPUTING.with_borrow_mut(|computing| computing.push(id));
        Computing
    }
}

impl Drop for Computing {
    fn drop(&mut self) {
        COMPUTING.with_borrow_mut(|computing| computing.pop());
    }
}

/// Marks nodes as being flattened by this thread until it is dropped: they have been taken to
/// be evaluated already, so [`JitState::evaluate_alongside`] must not take them along (a group
/// containing the very expression that is too big to compile would be too big again).
pub(super) struct Flattening(usize);

impl Flattening {
    pub(super) fn enter(roots: impl Iterator<Item = *const Node>) -> Self {
        FLATTENING.with_borrow_mut(|flying| {
            let before = flying.len();
            flying.extend(roots);
            Flattening(before)
        })
    }

    /// Whether another expression was being flattened already when this one began: this
    /// flattening is part of evaluating a part of that one.
    pub(super) fn nested(&self) -> bool {
        self.0 > 0
    }
}

impl Drop for Flattening {
    fn drop(&mut self) {
        FLATTENING.with_borrow_mut(|flying| flying.truncate(self.0));
    }
}

thread_local! {
    static MEASURED: std::cell::RefCell<Measured> = const { std::cell::RefCell::new(Measured { inputs: Vec::new(), below: Vec::new(), decoded: Vec::new(), tmp: Vec::new() }) };
}

/// What an operation on some operands has below it: the counts, and the ids as a [`Node`] keeps
/// them.
struct Measure {
    /// Distinct nodes below.
    below: usize,
    /// Distinct buffers read.
    inputs: usize,
    ids: Box<[u64]>,
    base: u64,
    words: u32,
    sparse: bool,
}

/// The distinct nodes below and distinct buffers read by an operation on `args`.
fn measure(args: &[Src]) -> Measure {
    measure_nodes(args).unwrap_or_else(|| measure_any(args))
}

/// The number of ids in the union of the sorted lists `a` and `b`.
fn union_len(a: &[u64], b: &[u64]) -> usize {
    let (mut i, mut j, mut n) = (0, 0, 0);
    while let (Some(&x), Some(&y)) = (a.get(i), b.get(j)) {
        i += usize::from(x <= y);
        j += usize::from(y <= x);
        n += 1;
    }
    n + (a.len() - i) + (b.len() - j)
}

/// Writes the union of the sorted lists `a` and `b` to `out` (which holds exactly [`union_len`] ids).
fn union_write(a: &[u64], b: &[u64], out: &mut [u64]) {
    let (mut i, mut j, mut n) = (0, 0, 0);
    while let (Some(&x), Some(&y)) = (a.get(i), b.get(j)) {
        if let Some(slot) = out.get_mut(n) {
            *slot = x.min(y);
        }
        i += usize::from(x <= y);
        j += usize::from(y <= x);
        n += 1;
    }
    for (slot, &id) in out.iter_mut().skip(n).zip(a.get(i..).unwrap_or(&[]).iter().chain(b.get(j..).unwrap_or(&[]))) {
        *slot = id;
    }
}

/// [`measure`] for the operations nearly every expression is made of: one or two pending
/// operands (scalars beside them) whose sets are bit sets, read without a buffer operand. Merges
/// the buffers on the stack instead of through the thread's working memory. `None` if `args` are
/// anything else.
#[inline]
fn measure_nodes(args: &[Src]) -> Option<Measure> {
    let mut nodes: [Option<&Arc<Node>>; 2] = [None; 2];
    let mut count = 0;
    let (mut buffers, mut n_buffers) = ([0u64; 3], 0);
    for arg in args {
        match arg {
            Src::Scalar(_) => {}
            Src::Buffer(storage) => {
                *buffers.get_mut(n_buffers)? = storage.binding_id();
                n_buffers += 1;
            }
            Src::Node(node) => {
                if nodes.iter().take(count).flatten().any(|first| Arc::ptr_eq(first, node)) {
                    continue;
                }
                *nodes.get_mut(count)? = Some(node);
                count += 1;
            }
        }
    }
    if count == 0 {
        // Buffers and scalars only: the sorted distinct buffers are all there is.
        let found = buffers.get_mut(..n_buffers)?;
        found.sort_unstable();
        let mut distinct = 0;
        for at in 0..found.len() {
            if at == 0 || found[at] != found[at - 1] {
                found[distinct] = found[at];
                distinct += 1;
            }
        }
        let mut ids = bits::zeroed(distinct);
        ids.copy_from_slice(&found[..distinct]);
        return Some(Measure { below: 0, inputs: distinct, ids, base: 0, words: 0, sparse: true });
    }
    if n_buffers > 0 {
        return None;
    }
    let [Some(first), second] = nodes else { return None };
    if first.ids().sparse || second.is_some_and(|node| node.ids().sparse) {
        return None;
    }
    // The buffers: the operand's own, or the union of both operands'.
    let (a, b) = (first.inputs(), second.map_or(&[][..], |node| node.inputs()));
    let merged = !(a == b || b.is_empty() || a.is_empty());
    let inputs_len = if merged { union_len(a, b) } else { a.len().max(b.len()) };
    let (mut lo, mut hi, mut upper, mut used) = (u64::MAX, 0u64, 0usize, 0usize);
    for node in [Some(first), second].into_iter().flatten() {
        (lo, hi, upper) = (lo.min(node.id), hi.max(node.id), upper + 1 + node.below_len());
        used += 1 + node.below_part().len();
        if let Some((first, last)) = node.below_span() {
            (lo, hi) = (lo.min(first), hi.max(last));
        }
    }
    let width = ((hi >> 6) - (lo >> 6) + 1) as usize;
    if width > 2 * used + 8 {
        return None;
    }
    let base = lo & !63;
    let mut ids = bits::zeroed(width + inputs_len);
    let (dense, tail) = ids.split_at_mut(width);
    for node in [Some(first), second].into_iter().flatten() {
        mark(dense, base, node.id);
        node.mark_below(dense, base);
    }
    if merged {
        union_write(a, b, tail);
    } else {
        tail.copy_from_slice(if a.is_empty() { b } else { a });
    }
    let below = if second.is_none() { upper } else { bits::count_ones(dense) };
    Some(Measure { below, inputs: inputs_len, ids, base, words: width as u32, sparse: false })
}

/// [`measure`] for any operands.
fn measure_any(args: &[Src]) -> Measure {
    // Nodes made without their ids make them by measuring, which needs the working memory below.
    for arg in args {
        if let Src::Node(node) = arg {
            let _ = node.ids();
        }
    }
    MEASURED.with_borrow_mut(|m| {
        let Measured { inputs, below, decoded, tmp } = m;
        inputs.clear();
        let mut nodes: [Option<&Arc<Node>>; 3] = [None; 3];
        let mut count = 0;
        for arg in args {
            match arg {
                Src::Scalar(_) => {}
                Src::Buffer(storage) => insert_sorted(inputs, storage.binding_id()),
                Src::Node(node) => {
                    // `x * x`: the second operand adds nothing.
                    if nodes.iter().take(count).flatten().any(|first| Arc::ptr_eq(first, node)) {
                        continue;
                    }
                    if let Some(slot) = nodes.get_mut(count) {
                        *slot = Some(node);
                        count += 1;
                    }
                    union_into(inputs, tmp, node.inputs());
                }
            }
        }
        let operands = || nodes.iter().take(count).flatten();
        let (mut lo, mut hi, mut upper, mut used) = (u64::MAX, 0u64, 0usize, 0usize);
        for node in operands() {
            (lo, hi, upper) = (lo.min(node.id), hi.max(node.id), upper + 1 + node.below_len());
            used += 1 + if node.ids().sparse { node.below_part().len() / 2 } else { node.below_part().len() };
            if let Some((first, last)) = node.below_span() {
                (lo, hi) = (lo.min(first), hi.max(last));
            }
        }
        if count == 0 {
            return Measure { below: 0, inputs: inputs.len(), ids: concat_ids(&[], inputs), base: 0, words: 0, sparse: true };
        }
        // The ids of nodes made around the same time are close together: one bit each, which
        // merges with a handful of word operations. Sets with big gaps between their words are
        // merged as sorted lists of the words in use.
        let width = ((hi >> 6) - (lo >> 6) + 1) as usize;
        if width <= 2 * used + 8 {
            let base = lo & !63;
            // One allocation: the set (zeroed, then merged into) and the buffers behind it.
            let mut ids = bits::zeroed(width + inputs.len());
            let (dense, tail) = ids.split_at_mut(width);
            for node in operands() {
                mark(dense, base, node.id);
                node.mark_below(dense, base);
            }
            tail.copy_from_slice(inputs);
            // One operand: its set plus itself, which is not in it.
            let below = if count == 1 { upper } else { bits::count_ones(dense) };
            return Measure { below, inputs: inputs.len(), ids, base, words: width as u32, sparse: false };
        }
        below.clear();
        for node in operands() {
            let part = node.below_part();
            if node.ids().sparse || part.is_empty() {
                merge_sets(below, tmp, part);
            } else {
                // A bit set: its words that are not empty, as pairs.
                decoded.clear();
                for (k, &word) in part.iter().enumerate() {
                    if word != 0 {
                        decoded.extend([(node.ids().base >> 6) + k as u64, word]);
                    }
                }
                merge_sets(below, tmp, decoded);
            }
            add_id(below, node.id);
        }
        let distinct = below.as_chunks::<2>().0.iter().map(|&[_, bits]| bits.count_ones() as usize).sum();
        Measure { below: distinct, inputs: inputs.len(), ids: concat_ids(below, inputs), base: 0, words: below.len() as u32, sparse: true }
    })
}

impl Measure {
    /// Replaces the buffers by `inputs` (sorted), counted exactly.
    fn set_inputs(&mut self, inputs: &[u64]) {
        self.ids = concat_ids(self.ids.get(..self.words as usize).unwrap_or(&[]), inputs);
        self.inputs = inputs.len();
    }
}

/// The ids of a node: `below` (see [`Node::ids`]) followed by `inputs`, in one allocation.
fn concat_ids(below: &[u64], inputs: &[u64]) -> Box<[u64]> {
    let mut ids = Vec::with_capacity(below.len() + inputs.len());
    ids.extend_from_slice(below);
    ids.extend_from_slice(inputs);
    ids.into_boxed_slice()
}

/// Sorted ids of the distinct buffers the expressions `args` read, counted by walking them (so
/// not for every operation, see [`create`]): an evaluated node is its buffer, a node of `stop`
/// (which the caller is about to evaluate) is one buffer of its own, with the top bit of its id
/// set. Unlike [`Node::inputs`] this does not count what lies below a part of the expression
/// that was evaluated after the node was created.
fn count_inputs(args: &[Src], stop: &[Arc<Node>]) -> Vec<u64> {
    const STOPPED: u64 = 1 << 63;
    let mut ids = Vec::new();
    let mut seen: FxMap<u64, ()> = FxMap::default();
    let mut work: Vec<Src> = args.to_vec();
    while let Some(src) = work.pop() {
        let Src::Node(node) = src else {
            if let Src::Buffer(storage) = src {
                insert_sorted(&mut ids, storage.binding_id());
            }
            continue;
        };
        if seen.insert(node.id, ()).is_some() {
            continue;
        }
        if let Some(storage) = node.evaluated() {
            insert_sorted(&mut ids, storage.binding_id());
        } else if stop.iter().any(|stopped| Arc::ptr_eq(stopped, &node)) {
            insert_sorted(&mut ids, STOPPED | node.id);
        } else if let Some(operands) = node.operands() {
            work.extend(operands.into_iter().take(node.op.kind.arity()));
        } else {
            node.inputs().iter().for_each(|&id| insert_sorted(&mut ids, id));
        }
    }
    ids
}

/// Sorted ids of the pending operations the expressions `args` consist of, counted by walking
/// them: unlike [`Node::below_len`] this does not count what lies below a part of the expression
/// that was evaluated after the node was created.
fn count_pending(args: &[Src]) -> Vec<u64> {
    let mut ids = Vec::new();
    let mut seen: FxMap<u64, ()> = FxMap::default();
    let mut work: Vec<Arc<Node>> = args.iter().filter_map(|arg| if let Src::Node(node) = arg { Some(Arc::clone(node)) } else { None }).collect();
    while let Some(node) = work.pop() {
        if seen.insert(node.id, ()).is_some() || node.evaluated().is_some() {
            continue;
        }
        ids.push(node.id);
        if node.looped.get().is_some() {
            continue;
        }
        if let Some(operands) = node.operands() {
            work.extend(operands.into_iter().take(node.op.kind.arity()).filter_map(|operand| if let Src::Node(inner) = operand { Some(inner) } else { None }));
        }
    }
    ids.sort_unstable();
    ids
}

/// The greatest depth (see [`Node::depth`]) among the node operands `args` have, counting only
/// what is still pending.
fn pending_depth(args: &[Src]) -> Option<u16> {
    let mut memo: FxMap<u64, u16> = FxMap::default();
    let roots: Vec<Arc<Node>> = args.iter().filter_map(|arg| if let Src::Node(node) = arg { Some(Arc::clone(node)) } else { None }).collect();
    let mut stack: Vec<(Arc<Node>, bool)> = roots.iter().map(|node| (Arc::clone(node), false)).collect();
    while let Some((node, expanded)) = stack.pop() {
        if memo.contains_key(&node.id) {
            continue;
        }
        let operands = if node.evaluated().is_some() || node.looped.get().is_some() { None } else { node.operands() };
        let Some(operands) = operands else {
            memo.insert(node.id, node.depth());
            continue;
        };
        // The operands that are still pending (an iterator over the copy: no allocation per node).
        let inner = operands.into_iter().take(node.op.kind.arity()).filter_map(|operand| match operand {
            Src::Node(inner) if inner.evaluated().is_none() => Some(inner),
            _ => None,
        });
        if !expanded {
            stack.push((node, true));
            stack.extend(inner.filter(|operand| !memo.contains_key(&operand.id)).map(|operand| (operand, false)));
            continue;
        }
        let depth = inner.filter_map(|operand| memo.get(&operand.id)).map(|d| d.saturating_add(1)).max().unwrap_or(0);
        memo.insert(node.id, depth);
    }
    roots.iter().filter_map(|node| memo.get(&node.id).copied()).max()
}

/// The [`Measure`] of `old` with the nodes below replaced by the sorted `pending` ids (as sorted
/// words of 64 ids, see [`Node::ids`]).
fn measure_pending(pending: &[u64], old: &Measure) -> Measure {
    let mut pairs: Vec<u64> = Vec::new();
    for &id in pending {
        let n = pairs.len();
        if n >= 2 && pairs[n - 2] == id >> 6 {
            pairs[n - 1] |= 1 << (id & 63);
        } else {
            pairs.extend([id >> 6, 1 << (id & 63)]);
        }
    }
    let inputs = old.ids.get(old.words as usize..).unwrap_or(&[]);
    Measure { below: pending.len(), inputs: old.inputs, ids: concat_ids(&pairs, inputs), base: 0, words: pairs.len() as u32, sparse: true }
}

/// Replaces `args[i]` and every operand that is the same node by its evaluated buffer.
fn evaluate_operand(args: &mut [Src], i: usize) -> std::result::Result<(), Failure> {
    let node = match args.get(i) {
        Some(Src::Node(node)) => Arc::clone(node),
        _ => return Ok(()),
    };
    let storage = match node.force() {
        Ok(storage) => Arc::clone(storage),
        Err(error) => return Err(Failure::Inherited(Arc::clone(error))),
    };
    for arg in args.iter_mut() {
        if matches!(arg, Src::Node(n) if Arc::ptr_eq(n, &node)) {
            *arg = Src::Buffer(Arc::clone(&storage));
        }
    }
    Ok(())
}

/// Fails with [`LumaError::Unsupported`] if `op` does not exist for its element type: `f64` has
/// the math functions and `powf` only on the CUDA and CPU backends (and needs a device with 64-bit shader floats), `Df32`
/// only the operations of its library, `C32` the field operations, `exp`, `sqrt`, equality and the
/// real parts, and none of the three has generators; the bitwise operations only
/// exist for `u32` and `i32` (the rotations for `u32` alone), `u32` has no negation, and `i32` has
/// no float math. `bool` has no generators (their values would not be 0 or 1).
fn check_supported(ctx: &Context, op: &Op, name: &str) -> std::result::Result<(), LumaError> {
    let wide = matches!(op.ty, ElemTy::F64 | ElemTy::Df32 | ElemTy::C32);
    let exists = match op.kind {
        OpKind::Unary(unary) => unary.expr_for(op.ty).is_some(),
        OpKind::Binary(binary) => binary.expr_for(op.ty).is_some(),
        OpKind::Compare(cmp) => cmp.exists_for(op.ty),
        OpKind::Select | OpKind::Shift(_) | OpKind::Cast(_) | OpKind::Part(_) | OpKind::Make => true,
        OpKind::Clamp => op.ty != ElemTy::C32,
        OpKind::Classify(_) => !matches!(op.ty, ElemTy::U32 | ElemTy::I32 | ElemTy::Bool | ElemTy::C32),
        OpKind::Powf => matches!(op.ty, ElemTy::F32 | ElemTy::F64 | ElemTy::Df32 | ElemTy::F16 | ElemTy::BF16),
        // A `bool` holds only 0 or 1, which no ramp or counter keeps.
        OpKind::Generate(_) => !wide && op.ty != ElemTy::Bool,
    };
    if !exists {
        return Err(LumaError::Unsupported(format!("{name} is not available for {} arrays", op.ty.display())));
    }
    if (op.ty == ElemTy::F64 || op.out_ty() == ElemTy::F64) && !ctx.caps().contains(crate::caps::Caps::F64) {
        return Err(LumaError::Unsupported("f64 arrays need a GPU with 64-bit shader floats (Vulkan)".into()));
    }
    // WGSL has no `f64` math functions: they exist where the backend lowers them itself.
    let math = match op.kind {
        OpKind::Unary(unary) => unary.wide_math(),
        OpKind::Binary(binary) => binary.wide_math(),
        OpKind::Powf => true,
        _ => false,
    };
    if op.ty == ElemTy::F64 && math && !matches!(ctx.backend(), luma_core::BackendKind::Cuda | luma_core::BackendKind::Cpu) {
        return Err(LumaError::Unsupported(format!("{name} of f64 arrays needs the CUDA or the CPU backend")));
    }
    Ok(())
}

/// Turns the operands among the first `arity` that were evaluated since they were created into
/// plain buffers; an operand that failed makes the operation fail the same way.
#[inline]
fn convert_evaluated(args: &mut Args, arity: usize) -> std::result::Result<(), Failure> {
    for i in 0..arity {
        let evaluated = match args.get(i) {
            Some(Src::Node(node)) => match node.result.get() {
                Some(Ok(_)) => true,
                Some(Err(error)) => return Err(Failure::Inherited(Arc::clone(error))),
                None => false,
            },
            _ => false,
        };
        if evaluated {
            evaluate_operand(args, i)?;
        }
    }
    Ok(())
}

/// Adds a lazy operation `op` on `args` (checked by the caller) producing `len` elements.
///
/// This is where fusion is limited: operands that already have a consumer, and the biggest operand
/// while the expression is too large, are evaluated first (see the module documentation).
#[inline(always)]
pub(crate) fn create(
    ctx: &Context,
    len: usize,
    op: Op,
    args: Args,
    name: &'static str,
    location: &'static Location<'static>,
) -> std::result::Result<Arc<Node>, Failure> {
    // By reference: an `Op` is big, and passing it by value copies it for every callee.
    create_with(ctx, len, &op, args, name, location)
}

fn create_with(
    ctx: &Context,
    len: usize,
    first_op: &Op,
    mut args: Args,
    name: &'static str,
    location: &'static Location<'static>,
) -> std::result::Result<Arc<Node>, Failure> {
    let jit = ctx.jit();
    let mut arity = first_op.kind.arity();

    // Operands evaluated since they were created become plain buffers.
    convert_evaluated(&mut args, arity)?;

    // A tape of the step before may make this call, or the step is being recorded (see [`follow`]).
    let mut go = follow::Go::Off;
    if jit.follow_on() && follow::gate() {
        match follow::begin(ctx, len, first_op, &mut args, arity, name, location) {
            follow::Begun::Done(node) => return Ok(node),
            follow::Begun::Go(how) => go = how,
        }
    }

    // Algebraic simplification (see [`simplify`]); the operands a rewrite takes from a pending
    // operand may have been evaluated since it was created.
    let mut rewritten_op: Option<Op> = None;
    let mut rounds = 0;
    if jit.simplify_enabled() {
        while rounds < 4
            && let Some((rewritten, operands)) = simplify::rewrite(rewritten_op.as_ref().unwrap_or(first_op), &args, ctx.numerics_ref())
        {
            rewritten_op = Some(rewritten);
            args = operands;
            rounds += 1;
        }
        if rounds > 0 {
            arity = rewritten_op.as_ref().unwrap_or(first_op).kind.arity();
            convert_evaluated(&mut args, arity)?;
        }
    }
    let op: &Op = rewritten_op.as_ref().unwrap_or(first_op);
    check_supported(ctx, op, name).map_err(Failure::New)?;

    // Equal operations on identical operands are one node (see `ConsTable`).
    let cons_key = (!matches!(op.kind, OpKind::Unary(UnOp::Id))).then(|| cons_hash(op, len, &args));
    // An equal node would read every node operand too, which counts it as a consumer: an operand
    // nobody reads yet has no equal operation to find.
    let may_exist = args.iter().take(arity).all(|arg| !matches!(arg, Src::Node(node) if node.consumers.load(Ordering::Relaxed) == 0));
    if let Some(hash) = cons_key
        && may_exist
        && let Some(found) = jit.cons_lookup(hash, op, len, &args)
    {
        rerun::reused(&found);
        if !matches!(go, follow::Go::Off) {
            follow::finish_hit(go, rounds, &found);
        }
        return Ok(found);
    }

    let max_nodes = FUSION_LIMIT.with(std::cell::Cell::get).unwrap_or_else(|| jit.max_nodes()).clamp(1, MAX_NODES_LIMIT);
    let max_inputs = jit.max_inputs();
    // Most operations are within every limit: the cutting is in `settle`, which keeps this
    // function short (it runs for every operation).
    let first = measure(args.get(..arity).unwrap_or(&[]));
    let within = first.below < max_nodes
        && first.inputs <= max_inputs
        && !args.iter().take(arity).any(|arg| matches!(arg, Src::Node(node) if node.depth() >= MAX_DEPTH));
    let (measured, cut, exact_depth) = if within {
        (first, false, None)
    } else {
        drop(first);
        settle(ctx, op, &mut args, arity, name, location, max_nodes, max_inputs)?
    };
    if cut {
        // Without this, a long chain built without `eval` keeps the GPU idle until its end.
        let _ = ctx.submit();
    }
    // What the operands add up to: the levels of shifts and of nesting, and an upper bound of the
    // work of the expression (operands that are shared count once per use).
    let is_shift = matches!(op.kind, OpKind::Shift(_));
    let (mut shift_depth, mut deepest, mut operand_weight) = (0u8, None::<u16>, 0.0);
    for (i, arg) in args.iter().take(arity).enumerate() {
        if let Src::Node(operand) = arg {
            shift_depth = shift_depth.max(operand.shift_depth().saturating_add(u8::from(i == 0 && is_shift)));
            deepest = deepest.max(Some(operand.depth()));
            operand_weight += operand.weight();
        }
    }
    // Operands that share nodes count them once per use (`operand_weight` unfolds the expression into
    // a tree, which doubles with every level of a flux that two readers read, say): no operation
    // weighs more than `MAX_AVERAGE_WEIGHT` on average, so that is what an expression of that many
    // distinct nodes is bounded by.
    let weight = (cost::op_weight(op, &ctx.cost_model()) + operand_weight).min((measured.below + 1) as f64 * MAX_AVERAGE_WEIGHT);
    let depth = exact_depth.unwrap_or(deepest).map_or(0, |deepest| deepest.saturating_add(1));

    let trace = rerun::created(op, len, &args, arity);
    // The distinct operands the node registers with once it exists (see [`registered_of`]).
    let (registered, known) = registered_of(&args, arity);
    // What the table is told: a cut changed the operands, so the hash is made again.
    let table_hash = cons_key.map(|hash| if cut { cons_hash(op, len, &args) } else { hash });
    let node = new_node(ctx, len, op, args, name, location, trace, measured.below, Ids::from(measured), 0, shift_depth, depth, weight);
    let weak = Arc::downgrade(&node);
    register_node(jit, &weak, &registered[..known]);
    if let Some(hash) = table_hash {
        jit.cons_insert(hash, weak);
    }
    if !matches!(go, follow::Go::Off) {
        follow::finish_created(go, rounds, cut, &node);
    }
    Ok(node)
}

/// The distinct operands a new node registers with once it exists, and how many there are. The
/// node owns `args` from then on, which keeps them alive, so only their addresses are noted (an
/// `Arc` cloned for every operand costs an atomic operation to take and another to give back).
#[inline(always)]
fn registered_of(args: &Args, arity: usize) -> ([Registered; 3], usize) {
    let mut registered = [Registered::None; 3];
    let mut known = 0;
    for (i, arg) in args.iter().enumerate().take(arity) {
        let repeated = |other: &Src| match (arg, other) {
            (Src::Buffer(a), Src::Buffer(b)) => Arc::ptr_eq(a, b),
            (Src::Node(a), Src::Node(b)) => Arc::ptr_eq(a, b),
            _ => false,
        };
        if args.iter().take(i).any(repeated) {
            continue;
        }
        let entry = match arg {
            Src::Buffer(storage) => Registered::Buffer(Arc::as_ptr(storage)),
            Src::Node(operand) => Registered::Node(Arc::as_ptr(operand)),
            Src::Scalar(_) => continue,
        };
        if let Some(slot) = registered.get_mut(known) {
            *slot = entry;
            known += 1;
        }
    }
    (registered, known)
}

/// A node, built where it lives: the node is bigger than what a move of it copies cheaply.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn new_node(
    ctx: &Context,
    len: usize,
    op: &Op,
    args: Args,
    name: &'static str,
    location: &'static Location<'static>,
    trace: u64,
    below_len: usize,
    ids: Ids,
    special: u8,
    shift_depth: u8,
    depth: u16,
    weight: f64,
) -> Arc<Node> {
    let ptr = Arc::into_raw(Arc::<Node>::new_uninit()).cast::<Node>().cast_mut();
    // SAFETY: `ptr` is the one reference of a fresh allocation of a `Node`, whose fields are
    // all written (one by one: a struct literal would be built on the stack and copied) before
    // the `Arc` is made again.
    unsafe {
        std::ptr::addr_of_mut!((*ptr).trace).write(trace);
        std::ptr::addr_of_mut!((*ptr).thread).write(current_thread());
        std::ptr::addr_of_mut!((*ptr).ctx).write(ctx.clone());
        std::ptr::addr_of_mut!((*ptr).op).write(*op);
        std::ptr::addr_of_mut!((*ptr).len).write(len);
        std::ptr::addr_of_mut!((*ptr).args).write(Mutex::new(Some(args)));
        std::ptr::addr_of_mut!((*ptr).result).write(OnceLock::new());
        std::ptr::addr_of_mut!((*ptr).name).write(name);
        std::ptr::addr_of_mut!((*ptr).location).write(location);
        std::ptr::addr_of_mut!((*ptr).id).write(NEXT_NODE_ID.fetch_add(1, Ordering::Relaxed));
        std::ptr::addr_of_mut!((*ptr).below_len).write(below_len);
        std::ptr::addr_of_mut!((*ptr).ids).write(IdsCell::new(ids));
        std::ptr::addr_of_mut!((*ptr).special).write(AtomicU8::new(special));
        std::ptr::addr_of_mut!((*ptr).consumers).write(AtomicU32::new(0));
        std::ptr::addr_of_mut!((*ptr).fused).write(AtomicBool::new(false));
        std::ptr::addr_of_mut!((*ptr).shift_depth).write(shift_depth);
        std::ptr::addr_of_mut!((*ptr).depth).write(depth);
        std::ptr::addr_of_mut!((*ptr).weight).write(weight);
        std::ptr::addr_of_mut!((*ptr).shifted_reads).write(AtomicU32::new(0));
        std::ptr::addr_of_mut!((*ptr).readers).write(Readers::default());
        std::ptr::addr_of_mut!((*ptr).requested).write(AtomicBool::new(false));
        std::ptr::addr_of_mut!((*ptr).leaf_free).write(AtomicBool::new(false));
        std::ptr::addr_of_mut!((*ptr).looped).write(OnceLock::new());
        std::ptr::addr_of_mut!((*ptr).deferred).write(OnceLock::new());
        std::ptr::addr_of_mut!((*ptr).handles).write(AtomicU32::new(0));
        Arc::from_raw(ptr)
    }
}

/// Tells the operands (the readers of a buffer or of a pending node, the consumer count of a
/// node) and the context (live nodes, pending count) of a node that was just made.
#[inline(always)]
fn register_node(jit: &JitState, weak: &Weak<Node>, registered: &[Registered]) {
    for entry in registered {
        // SAFETY: the operand is held by the node's `args` (which only this thread has touched
        // since the node was made).
        unsafe {
            match *entry {
                Registered::Buffer(storage) => (*storage).readers.add_weak(weak.clone()),
                Registered::Node(operand) => {
                    (*operand).consumers.fetch_add(1, Ordering::Relaxed);
                    (*operand).readers.add_weak(weak.clone());
                }
                Registered::None => {}
            }
        }
    }
    jit.with_live(|live| live.add_weak(weak.clone()));
    jit.pending_nodes.fetch_add(1, Ordering::Relaxed);
}

/// An operand a new node registers with (see [`create_with`]).
#[derive(Clone, Copy)]
enum Registered {
    None,
    Buffer(*const Storage),
    Node(*const Node),
}

/// What [`create`] does about an operation whose operands make it too big, too deep or too wide
/// to be one kernel: measures it and evaluates operands until it fits. Returns what it measured,
/// whether it evaluated anything, and the depth counted exactly, if it had to be.
#[cold]
#[inline(never)]
#[allow(clippy::too_many_arguments)]
fn settle(
    ctx: &Context,
    op: &Op,
    args: &mut Args,
    arity: usize,
    name: &'static str,
    location: &'static Location<'static>,
    max_nodes: usize,
    max_inputs: usize,
) -> std::result::Result<(Measure, bool, Option<Option<u16>>), Failure> {
    let jit = ctx.jit();
    // What this evaluates is part of the call, not the end of a step (see [`follow::escape`]).
    let _settling = follow::Settling::enter();
    // The first time the expression is over a limit, its biggest operand is evaluated together
    // with the live arrays that share pending work with it (see the module documentation)
    // instead of alone.
    let mut batched = false;
    let mut packed = false;
    let mut cut = false;
    let mut noted = false;
    let mut exact_depth: Option<Option<u16>>;
    loop {
        let mut measured = measure(args.get(..arity).unwrap_or(&[]));
        let (mut below, mut inputs) = (measured.below, measured.inputs);
        if below >= max_nodes {
            // `measure` adds up what the operands held when they were created, which includes
            // what has been evaluated since: count the pending operations that are left. A count
            // that is a little too high is left alone (the cut it makes comes early, which keeps
            // arrays that advance together in the same kernels, see the tests); one that is
            // several times too high would cut every operation made after a cut.
            let pending = count_pending(args.get(..arity).unwrap_or(&[]));
            if pending.len() * STALE_COUNT <= below {
                measured = measure_pending(&pending, &measured);
                below = measured.below;
            }
        }
        if inputs > max_inputs {
            // `measure` adds up what the operands read when they were created: a part of the
            // expression evaluated since reads its one buffer now. Count what is read.
            let ids = count_inputs(args.get(..arity).unwrap_or(&[]), &[]);
            inputs = ids.len();
            measured.set_inputs(&ids);
            // Many buffers of one kind are copied into one that they then are windows of, which
            // takes one binding (see `pack`); only if that cannot be done is the expression cut.
            if inputs > max_inputs && !packed {
                packed = true;
                if pack::repack(ctx, *op, args.get(..arity).unwrap_or(&[]), inputs - max_inputs) {
                    continue;
                }
            }
        }
        let deepest = args.iter().take(arity).filter_map(|arg| match arg {
            Src::Node(node) => Some(node.depth()),
            _ => None,
        });
        let mut deepest_operand = deepest.max();
        exact_depth = None;
        if deepest_operand.unwrap_or(0) >= MAX_DEPTH {
            // Like the count of operations, the depth includes what was evaluated since.
            deepest_operand = pending_depth(args.get(..arity).unwrap_or(&[]));
            exact_depth = Some(deepest_operand);
        }
        let deep = deepest_operand.unwrap_or(0) >= MAX_DEPTH;
        let too_big = below >= max_nodes || inputs > max_inputs;
        if !too_big && !deep {
            return Ok((measured, cut, exact_depth));
        }
        if !noted {
            noted = true;
            let reason = if below >= max_nodes {
                SplitReason::Nodes { nodes: below + 1, limit: max_nodes }
            } else if inputs > max_inputs {
                SplitReason::Inputs { inputs, limit: max_inputs }
            } else {
                SplitReason::Depth { limit: usize::from(MAX_DEPTH) }
            };
            jit.note_cut(name, location, reason);
            if jit.follow_on() {
                follow::cut_reason(reason);
            }
        }
        if !too_big {
            // Only the nesting: evaluating the deepest operand cuts the chain.
            let deepest = args
                .iter()
                .take(arity)
                .enumerate()
                .filter_map(|(i, a)| match a {
                    Src::Node(n) => Some((i, n.depth())),
                    _ => None,
                })
                .max_by_key(|&(_, depth)| depth);
            if let Some((i, _)) = deepest {
                cut = true;
                // Like the cut on the node limit: with the live arrays that share its work.
                if let Some(Src::Node(seed)) = args.get(i) {
                    jit.evaluate_alongside(Arc::clone(seed));
                }
                evaluate_operand(&mut *args, i)?;
            }
            continue;
        }
        if !batched {
            batched = true;
            let seed = args
                .iter()
                .take(arity)
                .filter_map(|arg| match arg {
                    Src::Node(node) => Some(node),
                    _ => None,
                })
                .max_by_key(|node| node.size())
                .map(Arc::clone);
            if let Some(seed) = seed {
                cut = true;
                jit.evaluate_alongside(seed);
                for i in 0..arity {
                    if matches!(args.get(i), Some(Src::Node(node)) if node.result.get().is_some()) {
                        evaluate_operand(&mut *args, i)?;
                    }
                }
                continue;
            }
        }
        let biggest = args
            .iter()
            .take(arity)
            .enumerate()
            .filter_map(|(i, a)| match a {
                Src::Node(n) => Some((i, n.size())),
                _ => None,
            })
            .max_by_key(|&(_, size)| size);
        match biggest {
            Some((i, _)) => {
                cut = true;
                evaluate_operand(&mut *args, i)?;
            }
            // Only buffers and scalars left: at most three operands, within any sane limit.
            None => return Ok((measured, cut, exact_depth)),
        }
    }
}

/// A node of `len` elements of type `ty` that stands for the value of `value`: evaluating it
/// records the value. Operations on a reduction that cannot extend its last pass (any on a
/// `Df32` sum) read this node, so that building them evaluates nothing; the reduction is
/// recorded when something that needs the value is, after the arrays it shares work with. It has
/// no operands: expressions that read it find it evaluated first, like a loop leaf.
pub(crate) fn deferred_value(ctx: &Context, len: usize, ty: ElemTy, value: Arc<structured::Structured>, name: &'static str, location: &'static Location<'static>) -> Arc<Node> {
    follow::interrupt(ctx.jit());
    let measured = measure(&[]);
    let node = Arc::new(Node {
        trace: 0,
        thread: current_thread(),
        ctx: ctx.clone(),
        op: Op::new(OpKind::Unary(UnOp::Id), ty),
        len,
        args: Mutex::new(None),
        result: OnceLock::new(),
        name,
        location,
        id: NEXT_NODE_ID.fetch_add(1, Ordering::Relaxed),
        below_len: measured.below,
        ids: IdsCell::new(Ids::from(measured)),
        special: AtomicU8::new(0),
        consumers: AtomicU32::new(0),
        fused: AtomicBool::new(false),
        shift_depth: 0,
        depth: 0,
        weight: 0.0,
        shifted_reads: AtomicU32::new(0),
        readers: Readers::default(),
        requested: AtomicBool::new(false),
        leaf_free: AtomicBool::new(false),
        looped: OnceLock::new(),
        deferred: OnceLock::new(),
        handles: AtomicU32::new(0),
    });
    let _ = node.deferred.set(value);
    ctx.jit().with_live(|live| live.add_weak(Arc::downgrade(&node)));
    ctx.jit().pending_nodes.fetch_add(1, Ordering::Relaxed);
    node
}

/// Whether the value of the arrays holding `node` can be replaced by another expression with
/// [`supersede`]: `node` is a pending expression that has not been taken apart yet, and kernels
/// can read it as a sub-program (see [`recompute_viewed`]).
pub(crate) fn can_supersede(node: &Arc<Node>) -> bool {
    node.is_pending() && recompute_viewed(node) && lock(&node.args).is_some()
}

/// From now on the arrays that hold `old` (a claim on the latest node of their value, see
/// [`Held::end`]) denote `new`. Expressions recorded earlier that read the node as an operand
/// keep its value. `new` may read that node; the claims own `new`, not the node (see [`Link`]).
pub(crate) fn supersede(old: &Held, new: Arc<Node>) {
    let _ = old.end().link().next.set(Held::new(new));
}

/// A program structure with the judgement of it (see [`JitState::judgement`]).
#[derive(Debug)]
struct JudgementEntry {
    cap: usize,
    /// The length class (a power of two) the program was judged at.
    class: usize,
    instrs: Arc<Vec<Instr>>,
    subs: Arc<Vec<Vec<Instr>>>,
    input_tys: Vec<ElemTy>,
    outputs: Vec<u32>,
    tail: program::Tail,
    judgement: Arc<Judgement>,
    estimates: inline::Estimates,
}

/// Hash of what makes two lazy operations the same value: the operation, the length and the
/// operands (buffers and nodes by identity, scalars by value).
fn cons_hash(op: &Op, len: usize, args: &Args) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = FxHasher::default();
    op.hash(&mut hasher);
    hasher.write_usize(len);
    for arg in args.iter().take(op.kind.arity()) {
        match arg {
            Src::Buffer(storage) => hasher.write_u64(storage.id() << 2),
            Src::Node(node) => hasher.write_u64(node.id << 2 | 1),
            Src::Scalar(scalar) => {
                scalar.ty.hash(&mut hasher);
                hasher.write_u64(scalar.bits);
            }
        }
    }
    hasher.finish()
}

/// Whether `a` and `b` are the same operands of `op`: the same buffers and nodes, equal scalars.
fn cons_same(op: &Op, a: &Args, b: &Args) -> bool {
    a.iter().zip(b).take(op.kind.arity()).all(|pair| match pair {
        (Src::Buffer(a), Src::Buffer(b)) => Arc::ptr_eq(a, b),
        (Src::Node(a), Src::Node(b)) => Arc::ptr_eq(a, b),
        (Src::Scalar(a), Src::Scalar(b)) => a.ty == b.ty && a.bits == b.bits,
        _ => false,
    })
}

/// Pending nodes by what they compute, so that writing the same operation on the same operands
/// twice yields one node, which a fused kernel computes once. Valid for one `epoch` of
/// [`JitState::write_epoch`]: no buffer changed its contents since, so equal operations on
/// identical operands are equal values. Holds weak references only: a node nobody else holds
/// stays droppable, and [`assign`] counts the owners of the nodes it walks.
#[derive(Default)]
struct ConsTable {
    epoch: u64,
    /// The operations, by hash: sets of [`CONS_WAYS`] entries each (a power of two of them), made
    /// when the first entry is stored. No node is ever looked for by walking: a lookup or an
    /// insert reads the entries of one set (an entry of a dead node is replaced by the insert
    /// that finds it, and [`ConsTable::sweep`] drops the others in bulk). The table grows when a
    /// live entry would have to give way: an expression made of more operations than the table
    /// holds would not share its common parts any more, which costs far more than the table.
    sets: Vec<ConsSet>,
    /// The sets that hold an entry, so that forgetting them all does not visit the others.
    filled: Vec<u32>,
    /// Entries stored since the dead ones were last dropped.
    stored: usize,
    /// Entries stored at which the dead ones are dropped next (see [`ConsTable::sweep`]).
    sweep_at: usize,
    /// The masks of region assignments (see [`MaskKey`]).
    masks: FxMap<MaskKey, Weak<Node>>,
}

impl std::fmt::Debug for ConsTable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ConsTable({} sets, {} filled)", self.sets.len(), self.filled.len())
    }
}

/// Entries of a [`ConsSet`].
const CONS_WAYS: usize = 8;

/// Every entry of a set in use.
const CONS_FULL: u32 = (1 << CONS_WAYS) - 1;

/// Fewest entries stored between two drops of the dead ones (see [`ConsTable::sweep`]).
const CONS_SWEEP_MIN: usize = 16384;

/// Sets a [`ConsTable`] starts with (a power of two).
const CONS_SETS_MIN: usize = 128;

/// Most sets a [`ConsTable`] grows to: 1M operations. Past that live entries are replaced.
const CONS_SETS_MAX: usize = 1 << 17;

/// The entries of one hash value range: the node that computes an operation, if the entry is in
/// use (bit `i` of `used`), and the high half of the hash of the operation (the half that tells
/// the set, see [`ConsTable::set_of`]). A lookup reads the first cache line only (the tags and
/// `used`), compares the tags of all the entries without a branch per entry, and leaves a
/// collision of tags to the comparison of the operands.
#[derive(Default)]
#[repr(C, align(64))]
struct ConsSet {
    tags: [u32; CONS_WAYS],
    used: u32,
    nodes: [Option<Weak<Node>>; CONS_WAYS],
}

impl ConsSet {
    /// The entries (bit `i` for entry `i`) in use that hold the tag of `hash`.
    #[inline]
    fn matching(&self, hash: u64) -> u32 {
        let tag = (hash >> 32) as u32;
        self.used & self.tags.iter().enumerate().fold(0, |bits, (i, &entry)| bits | u32::from(entry == tag) << i)
    }

    /// Drops the entries of dead nodes; returns the entries that are still in use.
    fn drop_dead(&mut self) -> u32 {
        for way in 0..CONS_WAYS {
            if self.used >> way & 1 != 0 && self.nodes[way].as_ref().is_none_or(|weak| weak.strong_count() == 0) {
                self.nodes[way] = None;
                self.used &= !(1 << way);
            }
        }
        self.used
    }

    /// Puts `node` in entry `way`.
    #[inline]
    fn put(&mut self, way: usize, hash: u64, node: Weak<Node>) {
        self.tags[way] = (hash >> 32) as u32;
        self.nodes[way] = Some(node);
        self.used |= 1 << way;
    }
}

impl ConsTable {
    /// Forgets every entry: buffers changed since they were made (`epoch` is the new one).
    fn clear_for(&mut self, epoch: u64) {
        for at in self.filled.drain(..) {
            if let Some(set) = self.sets.get_mut(at as usize) {
                set.nodes.fill_with(|| None);
                set.used = 0;
            }
        }
        self.masks.clear();
        self.stored = 0;
        self.epoch = epoch;
    }

    /// Drops the entries of dead nodes. A `Weak` keeps the memory of its node (a few hundred
    /// bytes) allocated, so an entry left for the insert that replaces it keeps a whole dead
    /// expression's nodes out of the allocator: the operations made next get fresh memory, cold
    /// in every cache (a recorded SHA-3 of 4096 messages made 83 % more page faults and took
    /// 1.8x the host time), where memory freed at once is reused while it is still cached.
    /// Runs after as many inserts as entries survived the last time (at least
    /// [`CONS_SWEEP_MIN`]), which keeps what it costs per insert constant.
    fn sweep(&mut self) {
        let Self { sets, filled, .. } = self;
        let mut live = 0;
        let mut drop_dead = |set: &mut ConsSet| {
            let used = set.drop_dead();
            live += used.count_ones() as usize;
            used != 0
        };
        if filled.len() * 8 >= sets.len() {
            // Most sets hold something: reading them in order is what the memory is best at.
            filled.clear();
            filled.extend(sets.iter_mut().enumerate().filter_map(|(at, set)| drop_dead(set).then_some(at as u32)));
        } else {
            filled.retain(|&at| sets.get_mut(at as usize).is_some_and(&mut drop_dead));
        }
        self.stored = 0;
        self.sweep_at = live.max(CONS_SWEEP_MIN);
    }

    /// The set `hash` belongs to, of `sets` sets.
    #[inline]
    fn set_of(hash: u64, sets: usize) -> usize {
        // The high bits: the low ones of an `FxHasher` result are the weakest.
        ((hash >> 40) as usize) & (sets - 1)
    }

    /// The node recorded for `hash`, if it is still alive.
    #[inline]
    fn find(&self, hash: u64) -> Option<Arc<Node>> {
        let set = self.sets.get(Self::set_of(hash, self.sets.len().max(1)))?;
        let mut hits = set.matching(hash);
        while hits != 0 {
            let way = hits.trailing_zeros() as usize;
            hits &= hits - 1;
            if let Some(node) = set.nodes[way].as_ref().and_then(Weak::upgrade) {
                return Some(node);
            }
        }
        None
    }

    /// Records `node` for `hash`, in place of what the hash had, in a free entry, or in one whose
    /// node is dead; if every entry of the set is alive the table grows (or, at its limit, the
    /// hash picks the entry that gives way).
    #[inline(always)]
    fn store(&mut self, hash: u64, node: Weak<Node>) {
        if self.sets.is_empty() {
            self.sets.resize_with(CONS_SETS_MIN, ConsSet::default);
        }
        self.stored += 1;
        if self.stored >= self.sweep_at.max(CONS_SWEEP_MIN) {
            self.sweep();
        }
        loop {
            let count = self.sets.len();
            let at = Self::set_of(hash, count);
            let Some(set) = self.sets.get_mut(at) else { return };
            let used = set.used;
            let same = set.matching(hash);
            let way = if same != 0 {
                same.trailing_zeros() as usize
            } else if used != CONS_FULL {
                (!used).trailing_zeros() as usize
            } else if let Some(way) = set.nodes.iter().position(|node| node.as_ref().is_none_or(|weak| weak.strong_count() == 0)) {
                way
            } else if count < CONS_SETS_MAX {
                self.grow();
                continue;
            } else {
                (hash as usize) % CONS_WAYS
            };
            set.put(way, hash, node);
            if used == 0 {
                self.filled.push(at as u32);
            }
            return;
        }
    }

    /// Doubles the number of sets: the entries of set `i` go to `i` or to `i + sets`.
    fn grow(&mut self) {
        let old = self.sets.len();
        let mut sets: Vec<ConsSet> = Vec::new();
        sets.resize_with(old * 2, ConsSet::default);
        let mut filled = Vec::with_capacity(self.filled.len() * 2);
        for &at in &self.filled {
            let Some(from) = self.sets.get_mut(at as usize) else { continue };
            for way in 0..CONS_WAYS {
                let Some(node) = from.nodes[way].take() else { continue };
                let to = Self::set_of(u64::from(from.tags[way]) << 32, old * 2);
                let Some(set) = sets.get_mut(to) else { continue };
                if set.used == 0 {
                    filled.push(to as u32);
                }
                let slot = (!set.used).trailing_zeros() as usize;
                set.put(slot, u64::from(from.tags[way]) << 32, node);
            }
        }
        self.sets = sets;
        self.filled = filled;
    }
}

/// What the mask of a region assignment is a function of: the extents of the base and, per axis,
/// the inclusive range of base coordinates the region covers (see `Array::assign_lazy`).
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub(crate) struct MaskKey {
    dims: [u32; 4],
    pub(crate) ends: [(i64, i64); 4],
}

impl MaskKey {
    pub(crate) fn new(dims: [u32; 4], ends: [(i64, i64); 4]) -> MaskKey {
        MaskKey { dims, ends }
    }
}

/// What a thread keeps for each context it made operations in (see [`CONS`] and [`LIVE`]).
struct Slot<T> {
    /// The [`JitState::id`] of the context.
    jit: u64,
    /// Dead once the context is dropped, which frees the slot.
    alive: Weak<()>,
    value: T,
}

/// The [`Slot`]s of one thread, the latest context last.
struct Slots<T> {
    slots: Vec<Slot<T>>,
    /// Calls since slots of dropped contexts were last looked for.
    calls: u32,
}

impl<T> Slots<T> {
    const fn new() -> Self {
        Slots { slots: Vec::new(), calls: 0 }
    }
}

impl<T: Default> Slots<T> {
    /// How many calls (a power of two minus one) pass between looks for the slots of dropped
    /// contexts.
    const SWEEP_MASK: u32 = (1 << 14) - 1;

    /// The value of `jit`, made if the thread has none yet.
    #[inline]
    fn of(&mut self, jit: &JitState) -> &mut T {
        self.calls = self.calls.wrapping_add(1);
        // The context of the last call is the one of the next, nearly always.
        let last = self.slots.len().wrapping_sub(1);
        if self.calls & Self::SWEEP_MASK != 0 && self.slots.get(last).is_some_and(|slot| slot.jit == jit.id) {
            return &mut self.slots[last].value;
        }
        self.find(jit)
    }

    #[cold]
    #[inline(never)]
    fn find(&mut self, jit: &JitState) -> &mut T {
        self.slots.retain(|slot| slot.jit == jit.id || slot.alive.strong_count() > 0);
        let at = match self.slots.iter().rposition(|slot| slot.jit == jit.id) {
            Some(at) => at,
            None => {
                self.slots.push(Slot { jit: jit.id, alive: Arc::downgrade(&jit.alive), value: T::default() });
                self.slots.len() - 1
            }
        };
        // The latest context goes last, where `of` looks first.
        let last = self.slots.len() - 1;
        self.slots.swap(at, last);
        &mut self.slots[last].value
    }
}

thread_local! {
    /// Every node this thread created in a context and that is not yet known to be evaluated
    /// (oldest first), so that [`Context::sync`] can evaluate the live ones. The sweeps only take
    /// the nodes of the calling thread, so the lists need no lock.
    static LIVE: std::cell::RefCell<Slots<readers::ReaderList>> = const { std::cell::RefCell::new(Slots::new()) };
    /// The [`ConsTable`]s of this thread. A thread only ever shares the nodes it made itself, so
    /// the tables need no lock.
    static CONS: std::cell::RefCell<Slots<ConsTable>> = const { std::cell::RefCell::new(Slots::new()) };
}

/// Most levels of shifted lazy expressions a kernel stages in workgroup memory.
const MAX_STAGES: u8 = 3;

/// Whether a shift (of `dims`, by `offsets`) of `node` reads its expression lazily rather than
/// evaluated. A staged sub-program costs its expression about once per cell plus the halo,
/// while one called per shift costs it once per shift, so small shifts nest up to
/// [`MAX_STAGES`] levels; anything else recomputes a small expression one level only, or every
/// further stencil step would multiply the work by its number of shifts.
pub(crate) fn recompute_shifted(node: &Node, dims: [usize; 4], offsets: [isize; 4]) -> bool {
    // A lazy operand runs inside kernels of segmented programs only if it fits one segment, so
    // segmented programs have no sub-programs.
    if node.len() as u64 * node.op().out_ty().stride() > node.context().segment_bytes() {
        return false;
    }
    let small = |axis: usize| {
        let dim = dims[axis] as isize;
        let offset = offsets[axis].rem_euclid(dim.max(1));
        (if offset > dim / 2 { offset - dim } else { offset }).abs() <= 2
    };
    let stageable = dims[2] == 1
        && dims[3] == 1
        && (0..2).all(small)
        && node.shift_depth() < MAX_STAGES - 1
        && node.size() <= 64;
    stageable || claim_recompute(node)
}

/// The most an operation of an expression weighs on average, in f32 operations (see [`Node::weight`]):
/// a division or square root is 4, a transcendental function 8 to 16 and rare.
const MAX_AVERAGE_WEIGHT: f64 = 8.0;

/// Most lazy reads a node accepts however cheap its expression is.
const MAX_LAZY_READS: u32 = 8;

/// Whether a shift or view that reads `node` lazily (as a sub-program, once per read) may do so:
/// the expression is small enough, holds no shifted sub-expression, has been read lazily fewer
/// than [`MAX_LAZY_READS`] times so far, and one more computation of it costs less than writing
/// it and reading it back (by the cost model: the operations of the expression over all its
/// elements against two passes over its bytes). The reader that would be one too many takes
/// the evaluated buffer instead. Counts the read when it answers `true`.
fn claim_recompute(node: &Node) -> bool {
    if node.size() > node.context().tuning().max_recomputed || node.shift_depth() != 0 {
        return false;
    }
    let cost = node.context().cost_model();
    let len = node.len() as f64;
    let one_more = node.weight() * len / cost.f32_ops_per_ns;
    if one_more > cost::kernel_ns(&cost, len * node.op().out_ty().traffic() * 2.0, 0.0) {
        return false;
    }
    let mut reads = node.shifted_reads.load(Ordering::Relaxed);
    loop {
        if reads >= MAX_LAZY_READS {
            return false;
        }
        match node.shifted_reads.compare_exchange_weak(reads, reads + 1, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => return true,
            Err(now) => reads = now,
        }
    }
}

/// From this many nodes on, an expression that another one shares is always evaluated first (see
/// [`worth_storing`]): a quarter of the node limit.
const ALWAYS_STORED_NODES: usize = MAX_NODES_LIMIT / 4;

/// Whether the pending `node`, which an expression outside the one being flattened reads too,
/// costs more to compute once more than to write and read back (by the cost model, as in
/// [`claim_recompute`]): then it is evaluated before the expressions that share it, otherwise
/// each computes it again and no kernel is spent on it.
pub(crate) fn worth_storing(node: &Node) -> bool {
    worth_storing_at(node, node.weight())
}

/// [`worth_storing`] for an expression of `weight` operations: the weight the node would have if
/// the nodes below it that are stored too were read as buffers (see `Graph::to_store`).
pub(crate) fn worth_storing_at(node: &Node, weight: f64) -> bool {
    if node.looped.get().is_some() {
        return true;
    }
    // The code of an expression this big is not worth writing into a second kernel whatever the
    // time says: the model does not see what it costs to compile and to keep in registers.
    if node.size() >= ALWAYS_STORED_NODES {
        return true;
    }
    let cost = node.context().cost_model();
    let len = node.len() as f64;
    weight * len / cost.f32_ops_per_ns > cost::kernel_ns(&cost, len * node.op().out_ty().traffic() * 2.0, 0.0)
}

/// Whether a view or gather of the pending `node` reads its expression lazily (as a sub-program
/// computed at the read index) instead of evaluating it: whenever it fits one buffer, which is
/// what a kernel computing sub-programs needs. Arithmetic costs far less than the memory traffic
/// of a buffer that is written and read again, so the expression is computed in the kernel that
/// reads it, once per distinct index (see [`inline`]); the node limit of [`create`] and the
/// size check of `program::flatten_inner` bound the kernel.
pub(crate) fn recompute_viewed(node: &Node) -> bool {
    node.len() as u64 * node.op().out_ty().stride() <= node.context().segment_bytes() && !node.requested.load(Ordering::Relaxed)
}

/// The compiled kernels of one loop program, with the state layout each one was built for.
type LoopKernels = Vec<(Vec<Option<usize>>, Arc<KernelCore>)>;

/// The instructions of a program and of its sub-programs, as shared by equal programs.
type Structure = (Arc<Vec<Instr>>, Arc<Vec<Vec<Instr>>>);

/// Per-context state of the fusion machinery.

#[derive(Debug)]
pub(crate) struct JitState {
    /// Generated kernels by the structure of their program.
    kernels: BuildCache<program::FusedKey, Arc<KernelCore>>,
    /// Pre-parameterized keys mapped straight to their compiled kernel and appended uniform words,
    /// so repeating a step skips lowering and rewriting `key` on every launch.
    fast_kernels: Mutex<FxMap<u64, Vec<FastKernel>>>,
    /// Kernels of store programs with transposed operands (see [`transpose::plan_fused`]).
    fused: transpose::FusedCache,
    /// Time-blocked stencil kernels (see [`stencil`]) by their source.
    stencils: BuildCache<String, Arc<KernelCore>>,
    /// The kernels of `scan` (see [`JitState::scan_kernel`]) by what their source is made of, so
    /// that a call does not generate and hash the source of a kernel it has run before.
    scans: BuildCache<String, Arc<KernelCore>>,
    /// Kernels built as Kernel IR that are not keyed by a program of fused instructions
    /// (register loops, stencils; see [`JitState::module_kernel`]).
    modules: BuildCache<String, Arc<KernelCore>>,
    /// Register-loop kernels (see [`launch::run_loop_fused`]) by program and state layout, so
    /// that a loop run again does not generate and hash its source again.
    loops: Mutex<FxMap<program::FusedKey, LoopKernels>>,
    /// The structured values made and not yet recorded (see [`multireduce`]), oldest first.
    structured: Mutex<Vec<std::sync::Weak<structured::Structured>>>,
    /// How many times each structure of merged reductions was recorded (see `Tuning::reduce_multi_after`), by `fast_key_hash`.
    merged_seen: Mutex<FxMap<u64, u32>>,
    /// What the rewrite engine decided when the context opened, and the programs its normaliser remembers (see [`rules`]).
    pub(crate) rules: rules::RulesState,
    /// Strategy rankings by subject (see [`structured::select`]).
    pub(crate) selections: structured::select::SelectionCache,
    /// What the selection cache of the strategy registry needs of the context (see [`structured::select`]).
    pub(crate) selcache: structured::select::SelState,
    /// Sampled launch timings of strategies (see [`structured::telemetry`]).
    pub(crate) telemetry: structured::telemetry::Telemetry,
    /// Kernels of non-baseline strategies built so far, against the variant budget.
    pub(crate) variants: structured::select::Variants,
    /// Nodes `eval` asked for that are not computed yet; dropped ones are skipped.
    requested: Mutex<RequestList>,
    max_nodes: AtomicUsize,
    inputs_limit: AtomicUsize,
    #[cfg(test)]
    device_inputs: usize,
    /// Overrides [`Tuning::max_live`] when not 0 (tests).
    max_live: AtomicUsize,
    /// Overrides [`Tuning::max_inlined`] when not 0 (tests).
    max_inlined: AtomicUsize,
    /// Replaces the context's cost model when set (tests).
    #[cfg(test)]
    cost_override: Mutex<Option<cost::CostModel>>,
    /// Whether workgroup staging may be chosen (tests switch it off to compare).
    #[cfg(test)]
    staging: AtomicBool,
    /// The staging of the last kernel looked up (tests).
    #[cfg(test)]
    last_staging: Mutex<Option<staging::Staging>>,
    /// What judging a program (see [`inline::judge`]) found, by the structure of the program.
    judgements: Mutex<FxMap<u64, Vec<JudgementEntry>>>,
    /// The estimates of the last judgement made, as text (tests).
    #[cfg(test)]
    last_judgement: Mutex<Option<String>>,
    /// The programs seen so far by their shape hash, to make equal ones share their
    /// instructions (see [`JitState::intern`]).
    structures: Mutex<FxMap<u64, Vec<Structure>>>,
    /// The programs built from graphs seen so far, to build them again from graphs of the same
    /// shape without walking them (see [`program::Recipe`]).
    recipes: Mutex<program::RecipeTable>,
    /// How many programs were built from a recipe (tests).
    #[cfg(test)]
    recipe_hits: AtomicU64,
    /// Counts the writes into buffers, so that [`ConsTable`] never outlives a change of contents.
    write_epoch: AtomicU64,
    /// The planner's decisions made while profiling or after `record_decisions`, if on.
    decisions: Mutex<Option<Vec<Decision>>>,
    /// Instructions of the program of the last kernel looked up (tests).
    #[cfg(test)]
    last_program_len: AtomicUsize,
    /// Selects computed as `if`/`else` in the last kernel built (tests).
    #[cfg(test)]
    last_branches: AtomicUsize,
    /// Whether [`simplify`] runs (tests switch it off to compare against the unsimplified result).
    #[cfg(test)]
    simplify: AtomicBool,
    /// Instructions of the last lowered program a kernel was built from (tests).
    #[cfg(test)]
    last_inlined_len: AtomicUsize,
    /// Most element-instructions per launch of a register loop; 0: sized by the cost model (see
    /// `launch::steps_per_launch`). Only tests set it.
    loop_work: AtomicU64,
    stats: Stats,
    /// Tells this state from the ones of other contexts (the `eval` history of a thread is kept by the thread, see [`looped`]).
    id: u64,
    /// Only held here: the per-thread lists of this context ([`Slots`]) see by it when the
    /// context is gone.
    alive: Arc<()>,
    /// Follows reduction readbacks to find loops that can be speculated (see [`speculate`]).
    speculation: Mutex<speculate::Tracker>,
    /// The programs of register loops that ran before, by call site (see [`rerun`]).
    reruns: Mutex<rerun::Cache>,
    /// The plans of the evaluations `eval` made (see [`plancache`]).
    plans: Mutex<plancache::Table>,
    /// Whether evaluations use [`JitState::plans`] (`LUMA_PLAN_CACHE=off` switches it off).
    plan_cache: AtomicBool,
    /// Changes whenever a setting that decides how graphs are planned does: part of every key of
    /// [`JitState::plans`].
    plan_epoch: AtomicU64,
    plan_stats: plancache::Stats,
    /// What this context knows of trace following (see [`follow`]).
    follow: follow::Shared,
    /// Evaluations the plan cache still sits out (see [`plancache`]).
    plan_skip: AtomicU32,
    /// Whether [`JitState::decisions`] is recording (kept apart so that looking is a load).
    decisions_on: AtomicBool,
    /// Nodes of this context that are alive and pending (see [`Node::settled`]): the plan cache
    /// replays only if the graph it walked is all of them.
    pending_nodes: std::sync::atomic::AtomicI64,
}

/// Counters of the fusion machinery, read by tests.
#[derive(Debug, Default)]
struct Stats {
    /// Kernels generated and compiled.
    built: AtomicU64,
    /// Kernel lookups answered from the cache.
    hits: AtomicU64,
    /// Elementwise kernels launched.
    elementwise_launches: AtomicU64,
    /// Fused first passes of reductions launched.
    reduce_launches: AtomicU64,
    /// Passes of reductions after the first (over partial results, or over a buffer).
    reduce_passes: AtomicU64,
    /// Register loops run from a rerun instead of a flattened trace (see [`rerun`]).
    replays: AtomicU64,
    /// Of those, the ones whose launches had started before the step ran.
    speculated: AtomicU64,
    /// Of the elementwise launches, the tiled transposes (see [`transpose`]).
    transposes: AtomicU64,
    /// Of the elementwise launches, the ones that ran several elements per invocation (see [`coarsen`]).
    coarsened: AtomicU64,
}

/// A snapshot of the counters of a context's [`JitState`].
#[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct StatsSnapshot {
    pub(crate) built: u64,
    pub(crate) hits: u64,
    pub(crate) elementwise_launches: u64,
    pub(crate) reduce_launches: u64,
    pub(crate) reduce_passes: u64,
    pub(crate) replays: u64,
    pub(crate) speculated: u64,
    pub(crate) transposes: u64,
    pub(crate) coarsened: u64,
}

impl JitState {
    pub(crate) fn new(limits: &luma_core::DeviceLimits, numerics: &crate::numerics::Numerics) -> Self {
        let max_nodes = std::env::var(MAX_NODES_ENV)
            .ok()
            .and_then(|value| value.trim().parse::<usize>().ok())
            .unwrap_or(DEFAULT_MAX_NODES)
            .clamp(1, MAX_NODES_LIMIT);
        // One storage binding is the output, plus luma's grid uniform and Params uniform (counted
        // against the same per-stage buffer table on Metal).
        let device_inputs = (limits.max_storage_buffers as usize)
            .saturating_sub(3)
            .clamp(1, MAX_INPUTS);
        let max_inputs = std::env::var(MAX_INPUTS_ENV)
            .ok()
            .and_then(|value| value.trim().parse::<usize>().ok())
            .map_or(device_inputs, |limit| limit.clamp(1, device_inputs));
        JitState {
            kernels: BuildCache::default(),
            fast_kernels: Mutex::new(FxMap::default()),
            fused: transpose::FusedCache::default(),
            stencils: BuildCache::default(),
            scans: BuildCache::default(),
            modules: BuildCache::default(),
            loops: Mutex::new(FxMap::default()),
            structured: Mutex::new(Vec::new()),
            merged_seen: Mutex::new(FxMap::default()),
            selections: Mutex::new(FxMap::default()),
            rules: rules::RulesState::new(numerics),
            selcache: structured::select::SelState::new(),
            telemetry: structured::telemetry::Telemetry::default(),
            variants: structured::select::Variants::default(),
            requested: Mutex::new(RequestList::default()),
            max_nodes: AtomicUsize::new(max_nodes),
            inputs_limit: AtomicUsize::new(max_inputs),
            #[cfg(test)]
            device_inputs,
            max_live: AtomicUsize::new(0),
            max_inlined: AtomicUsize::new(0),
            #[cfg(test)]
            cost_override: Mutex::new(None),
            #[cfg(test)]
            staging: AtomicBool::new(true),
            #[cfg(test)]
            last_staging: Mutex::new(None),
            judgements: Mutex::new(FxMap::default()),
            #[cfg(test)]
            last_judgement: Mutex::new(None),
            structures: Mutex::new(FxMap::default()),
            recipes: Mutex::new(program::RecipeTable::default()),
            #[cfg(test)]
            recipe_hits: AtomicU64::new(0),
            write_epoch: AtomicU64::new(0),
            decisions: Mutex::new(env_records_decisions().then(Vec::new)),
            #[cfg(test)]
            last_program_len: AtomicUsize::new(0),
            #[cfg(test)]
            last_branches: AtomicUsize::new(0),
            #[cfg(test)]
            simplify: AtomicBool::new(true),
            #[cfg(test)]
            last_inlined_len: AtomicUsize::new(0),
            loop_work: AtomicU64::new(0),
            stats: Stats::default(),
            id: NEXT_JIT_ID.fetch_add(1, Ordering::Relaxed),
            alive: Arc::new(()),
            speculation: Mutex::new(speculate::Tracker::new(
                !std::env::var(SPECULATE_ENV).is_ok_and(|value| value.trim().eq_ignore_ascii_case("off")),
            )),
            reruns: Mutex::new(rerun::Cache::default()),
            plans: Mutex::new(plancache::Table::default()),
            plan_cache: AtomicBool::new(plancache::enabled_by(std::env::var(plancache::ENV).ok().as_deref())),
            plan_epoch: AtomicU64::new(0),
            plan_stats: plancache::Stats::default(),
            follow: follow::Shared::new(),
            plan_skip: AtomicU32::new(0),
            decisions_on: AtomicBool::new(env_records_decisions()),
            pending_nodes: std::sync::atomic::AtomicI64::new(0),
        }
    }

    pub(crate) fn max_nodes(&self) -> usize {
        self.max_nodes.load(Ordering::Relaxed)
    }

    /// Most values a fused kernel keeps live at once (see [`launch`]).
    pub(crate) fn max_live(&self, tuning: &crate::tuning::Tuning) -> usize {
        match self.max_live.load(Ordering::Relaxed) {
            0 => tuning.max_live,
            forced => forced,
        }
    }

    /// Most instructions of a lowered program (see [`inline`]).
    pub(crate) fn max_inlined(&self, tuning: &crate::tuning::Tuning) -> usize {
        match self.max_inlined.load(Ordering::Relaxed) {
            0 => tuning.max_inlined,
            forced => forced,
        }
    }

    /// Returns the buffers of precomputed speculation to the pool.
    pub(crate) fn drop_speculated(&self) {
        // A readback holds the lock while it computes (and may evict, which lands here): skip then.
        if let Ok(mut tracker) = self.speculation.try_lock() {
            tracker.clear_cache();
        }
    }

    /// Makes the instructions of `key` the ones every equal program shares, so that comparing
    /// keys from here on is a comparison of addresses.
    pub(crate) fn intern(&self, key: &mut program::FusedKey) {
        let mut structures = lock(&self.structures);
        if structures.len() >= 4096 {
            structures.clear();
        }
        let entries = structures.entry(key.shape).or_default();
        let same = |known: &(Arc<Vec<Instr>>, Arc<Vec<Vec<Instr>>>)| {
            (Arc::ptr_eq(&known.0, &key.instrs) || known.0 == key.instrs) && (Arc::ptr_eq(&known.1, &key.subs) || known.1 == key.subs)
        };
        match entries.iter().find(|known| same(known)) {
            Some((instrs, subs)) => {
                key.instrs = Arc::clone(instrs);
                key.subs = Arc::clone(subs);
            }
            None => entries.push((Arc::clone(&key.instrs), Arc::clone(&key.subs))),
        }
    }

    /// How to run `key` (which [`inlines`](program::FusedKey::inlines)) over about `len`
    /// elements, found once per program structure and length class (see [`inline::judge`]).
    pub(crate) fn judgement(&self, ctx: &Context, key: &program::FusedKey, len: usize, op: &'static str, location: &'static Location<'static>) -> Arc<Judgement> {
        use std::hash::{Hash, Hasher};
        let cap = self.max_inlined(ctx.tuning());
        let class = len.next_power_of_two();
        let mut hasher = FxHasher::default();
        (cap, key.shape, class).hash(&mut hasher);
        let hash = hasher.finish();
        let mut judgements = lock(&self.judgements);
        if judgements.len() >= 4096 {
            judgements.clear();
        }
        let entries = judgements.entry(hash).or_default();
        let same = |entry: &JudgementEntry| {
            entry.cap == cap
                && entry.class == class
                && (Arc::ptr_eq(&entry.instrs, &key.instrs) || entry.instrs == key.instrs)
                && (Arc::ptr_eq(&entry.subs, &key.subs) || entry.subs == key.subs)
                && entry.input_tys == key.input_tys
                && entry.outputs == key.outputs
                && entry.tail == key.tail
        };
        let mut cached = true;
        let (judgement, estimates) = match entries.iter().find(|entry| same(entry)) {
            Some(entry) => (Arc::clone(&entry.judgement), entry.estimates),
            None => {
                cached = false;
                let (judgement, estimates) = inline::judge(key, class, cap, &ctx.cost_model(), ctx.tuning(), ctx.tuning().staging && self.staging_enabled());
                let judgement = Arc::new(judgement);
                entries.push(JudgementEntry {
                    cap,
                    class,
                    instrs: Arc::clone(&key.instrs),
                    subs: Arc::clone(&key.subs),
                    input_tys: key.input_tys.clone(),
                    outputs: key.outputs.clone(),
                    tail: key.tail,
                    judgement: Arc::clone(&judgement),
                    estimates,
                });
                (judgement, estimates)
            }
        };
        drop(judgements);
        if self.recording_decisions() {
            self.note_judgement(&judgement, estimates, cached, op, location);
        }
        #[cfg(test)]
        {
            *lock(&self.last_judgement) = Some(format!("{judgement:?} {estimates:?}"));
        }
        judgement
    }

    /// Records what the inliner decided (and, when it ran the program in one kernel, whether the
    /// stencil staging was used).
    fn note_judgement(&self, judgement: &Judgement, estimates: inline::Estimates, cached: bool, op: &'static str, location: &'static Location<'static>) {
        let ns = |ns: f64| DecisionVerdict::Estimate { ns };
        let mut decision;
        match judgement {
            Judgement::Run { staging } => {
                decision = Decision::new(op, location, DecisionSite::Inline, "inline").cached(cached);
                if let Some(&(sub, t)) = estimates.best_sub.as_ref() {
                    decision = decision.candidate(format!("materialise sub={sub}"), ns(t));
                }
                self.note_decision(decision);
                let staged = match staging {
                    Some(staging) => Decision::new(op, location, DecisionSite::Staging, format!("staged tile={}x{} subs={}", staging.tile[0], staging.tile[1], staging.subs.len())),
                    None => Decision::new(op, location, DecisionSite::Staging, "direct"),
                };
                let mut staged = staged.cached(cached);
                if staging.is_none()
                    && let Some(t) = estimates.staged_ns
                {
                    staged = staged.candidate("staged", ns(t));
                }
                if staging.is_some() {
                    staged = staged.candidate("direct", ns(estimates.inline_ns));
                }
                self.note_decision(staged);
            }
            Judgement::Materialise { subs, .. } => {
                decision = Decision::new(op, location, DecisionSite::Inline, format!("materialise subs={}", subs.len())).cached(cached);
                decision = decision.candidate("inline", ns(estimates.inline_ns));
                if let Some(t) = estimates.staged_ns {
                    decision = decision.candidate("staged", ns(t));
                }
                self.note_decision(decision);
            }
        }
    }

    /// Counts a reduction pass other than the fused first one (see [`Stats`]).
    pub(crate) fn note_reduce_pass(&self) {
        self.stats.reduce_passes.fetch_add(1, Ordering::Relaxed);
    }

    /// Records that a buffer's contents are about to change (or did): operations written before
    /// are not the same values as identical ones written after.
    pub(crate) fn note_write(&self) {
        if self.follow_on() {
            follow::note_write(self);
        }
        self.write_epoch.fetch_add(1, Ordering::Relaxed);
    }

    /// The pending node that already computes the operation `op` on `args` (of `hash`, see
    /// [`cons_hash`]), if any.
    fn cons_lookup(&self, hash: u64, op: &Op, len: usize, args: &Args) -> Option<Arc<Node>> {
        let epoch = self.write_epoch.load(Ordering::Relaxed);
        // The upgraded node is checked outside the table: it may be the last owner of its node.
        let node = self.with_cons(|table| {
            if table.epoch != epoch {
                table.clear_for(epoch);
                return None;
            }
            table.find(hash)
        })?;
        // A node computed inside another kernel is a dead temporary of an earlier expression:
        // sharing it would tie this expression to that one (and to its steps in a loop).
        // (The table of a thread only holds the nodes that thread made: two threads sharing a
        // node can deadlock, one may be recording it, flattening it or waiting for what the other
        // evaluates.)
        let usable = node.is_pending()
            && node.handles.load(Ordering::Relaxed) == 0
            && !node.fused.load(Ordering::Relaxed)
            && match lock(&node.args).as_ref() {
                // The table holds hashes only: a collision must not share a different value.
                Some(theirs) => node.op == *op && node.len == len && cons_same(op, theirs, args),
                // A loop leaf has handed its operands to the plan.
                None => node.looped.get().is_some() && node.op == *op && node.len == len,
            };
        usable.then_some(node)
    }

    /// The pending mask node of `key` that no array holds, if one is still around: what the
    /// operations that build the mask would hand out again (see [`cons_lookup`]).
    pub(crate) fn mask_lookup(&self, key: &MaskKey) -> Option<Arc<Node>> {
        let epoch = self.write_epoch.load(Ordering::Relaxed);
        let node = self.with_cons(|table| {
            if table.epoch != epoch {
                table.clear_for(epoch);
                return None;
            }
            table.masks.get(key)?.upgrade()
        })?;
        let usable = node.is_pending() && node.handles.load(Ordering::Relaxed) == 0 && !node.fused.load(Ordering::Relaxed) && lock(&node.args).is_some();
        if !usable {
            return None;
        }
        rerun::reused(&node);
        Some(node)
    }

    /// Remembers `node` as the mask of `key`.
    pub(crate) fn mask_store(&self, key: MaskKey, node: &Arc<Node>) {
        let epoch = self.write_epoch.load(Ordering::Relaxed);
        self.with_cons(|table| {
            if table.epoch != epoch {
                table.clear_for(epoch);
            }
            if table.masks.len() >= 64 {
                table.masks.retain(|_, weak| weak.strong_count() > 0);
            }
            table.masks.insert(key, Arc::downgrade(node));
        });
    }

    fn cons_insert(&self, hash: u64, weak: Weak<Node>) {
        let epoch = self.write_epoch.load(Ordering::Relaxed);
        self.with_cons(|table| {
            if table.epoch != epoch {
                table.clear_for(epoch);
            }
            table.store(hash, weak);
        });
    }

    /// Runs `f` on the calling thread's [`ConsTable`] of this context.
    #[inline(always)]
    fn with_cons<R>(&self, f: impl FnOnce(&mut ConsTable) -> R) -> R {
        CONS.with_borrow_mut(|slots| f(slots.of(self)))
    }

    /// Runs `f` on the calling thread's list of live nodes of this context.
    #[inline(always)]
    fn with_live<R>(&self, f: impl FnOnce(&mut readers::ReaderList) -> R) -> R {
        LIVE.with_borrow_mut(|slots| f(slots.of(self)))
    }

    /// Starts recording decisions (see [`Decision`]); an earlier list is kept.
    pub(crate) fn start_decisions(&self) {
        lock(&self.decisions).get_or_insert_with(Vec::new);
        self.decisions_on.store(true, Ordering::Relaxed);
    }

    /// Whether decisions are recorded: call sites build a [`Decision`] only if so.
    #[inline]
    pub(crate) fn recording_decisions(&self) -> bool {
        self.decisions_on.load(Ordering::Relaxed)
    }

    /// Mutes (or unmutes) the recording without dropping the list: for a replay, whose launches
    /// would re-derive decisions the plan already holds.
    pub(crate) fn pause_decisions(&self, paused: bool) {
        self.decisions_on.store(!paused, Ordering::Relaxed);
    }

    /// Stops recording decisions and returns them.
    pub(crate) fn stop_decisions(&self) -> Vec<Decision> {
        self.decisions_on.store(false, Ordering::Relaxed);
        lock(&self.decisions).take().unwrap_or_default()
    }

    /// Keeps `decisions` (those of a profiled region) for the next [`JitState::drain_decisions`]
    /// without recording more.
    pub(crate) fn park_decisions(&self, decisions: Vec<Decision>) {
        *lock(&self.decisions) = Some(decisions);
    }

    /// Returns the decisions recorded so far and goes on recording (if it was on).
    pub(crate) fn drain_decisions(&self) -> Vec<Decision> {
        lock(&self.decisions).as_mut().map(std::mem::take).unwrap_or_default()
    }

    /// Records a decision, if anyone listens.
    pub(crate) fn note_decision(&self, decision: Decision) {
        if let Some(list) = lock(&self.decisions).as_mut() {
            list.push(decision);
        }
    }

    /// Records the decision `build` makes, if anyone listens; `build` gets the caller's location
    /// (for sites that have no operation of the user's to point at).
    #[track_caller]
    pub(crate) fn note_with(&self, build: impl FnOnce(&'static Location<'static>) -> Decision) {
        if self.recording_decisions() {
            self.note_decision(build(Location::caller()));
        }
    }

    /// Records that an expression was cut into several kernels, if anyone listens.
    pub(crate) fn note_cut(&self, op: &'static str, location: &'static Location<'static>, reason: SplitReason) {
        if self.recording_decisions() {
            self.note_decision(Decision::new(op, location, DecisionSite::Cut(reason), cut_label(reason)));
        }
    }

    #[cfg(test)]
    pub(crate) fn record_splits(&self) {
        self.start_decisions();
    }

    /// The `Cut` decisions recorded since `record_splits` (what `Profile::cuts` gives).
    #[cfg(test)]
    pub(crate) fn take_splits(&self) -> Vec<TestCut> {
        self.stop_decisions()
            .into_iter()
            .filter_map(|d| match d.site {
                DecisionSite::Cut(reason) => Some(TestCut { op: d.op, location: d.location, reason, decision: d }),
                _ => None,
            })
            .collect()
    }

    /// Overrides the most input buffers of a kernel (tests, to behave like a device with fewer bindings).
    #[cfg(test)]
    pub(crate) fn set_max_inputs(&self, max_inputs: usize) {
        self.inputs_limit
            .store(max_inputs.clamp(1, self.device_inputs), Ordering::Relaxed);
    }

    /// Most input buffers one fused kernel binds: what the device allows, capped by [`MAX_INPUTS`].
    pub(crate) fn max_inputs(&self) -> usize {
        self.inputs_limit.load(Ordering::Relaxed)
    }

    #[cfg(test)]
    pub(crate) fn set_cost_override(&self, model: cost::CostModel) {
        *lock(&self.cost_override) = Some(model);
        self.plan_epoch.fetch_add(1, Ordering::Relaxed);
        lock(&self.judgements).clear();
        lock(&self.fast_kernels).clear();
    }

    #[cfg(test)]
    pub(crate) fn cost_override(&self) -> Option<cost::CostModel> {
        *lock(&self.cost_override)
    }

    /// The estimates behind the last judgement made, as text (tests).
    #[cfg(test)]
    pub(crate) fn last_judgement(&self) -> Option<String> {
        lock(&self.last_judgement).clone()
    }

    /// Whether workgroup staging may be chosen (always, outside tests).
    fn staging_enabled(&self) -> bool {
        #[cfg(test)]
        return self.staging.load(Ordering::Relaxed);
        #[cfg(not(test))]
        true
    }

    /// Allows or forbids workgroup staging, and forgets the judgements made (tests).
    #[cfg(test)]
    pub(crate) fn set_staging(&self, on: bool) {
        self.staging.store(on, Ordering::Relaxed);
        self.plan_epoch.fetch_add(1, Ordering::Relaxed);
        lock(&self.judgements).clear();
        lock(&self.fast_kernels).clear();
    }

    /// The staging of the kernel of the last program looked up (tests).
    #[cfg(test)]
    pub(crate) fn last_staging(&self) -> Option<staging::Staging> {
        lock(&self.last_staging).clone()
    }

    #[cfg(test)]
    pub(crate) fn set_max_live(&self, max_live: usize) {
        self.max_live.store(max_live, Ordering::Relaxed);
    }

    #[cfg(test)]
    pub(crate) fn set_max_inlined(&self, max_inlined: usize) {
        self.max_inlined.store(max_inlined, Ordering::Relaxed);
    }

    /// Whether operations are simplified when created (always, outside tests).
    fn simplify_enabled(&self) -> bool {
        #[cfg(test)]
        return self.simplify.load(Ordering::Relaxed);
        #[cfg(not(test))]
        true
    }

    #[cfg(test)]
    pub(crate) fn set_simplify(&self, on: bool) {
        self.simplify.store(on, Ordering::Relaxed);
    }

    #[cfg(test)]
    pub(crate) fn last_inlined_len(&self) -> usize {
        self.last_inlined_len.load(Ordering::Relaxed)
    }

    #[cfg(test)]
    pub(crate) fn last_program_len(&self) -> usize {
        self.last_program_len.load(Ordering::Relaxed)
    }

    #[cfg(test)]
    pub(crate) fn recipe_hits(&self) -> u64 {
        self.recipe_hits.load(Ordering::Relaxed)
    }

    #[cfg(test)]
    pub(crate) fn last_branches(&self) -> usize {
        self.last_branches.load(Ordering::Relaxed)
    }

    #[cfg(test)]
    pub(crate) fn set_loop_work(&self, work: u64) {
        self.loop_work.store(work.max(1), Ordering::Relaxed);
    }

    #[cfg(test)]
    pub(crate) fn speculation_hits(&self) -> u64 {
        lock(&self.speculation).hits
    }

    #[cfg(test)]
    pub(crate) fn set_speculate(&self, on: bool) {
        lock(&self.speculation).enabled = on;
    }

    /// Switches the plan cache on or off (see [`plancache`]); dropping every plan.
    pub(crate) fn set_plan_cache(&self, on: bool) {
        self.plan_cache.store(on, Ordering::Relaxed);
        self.plan_skip.store(0, Ordering::Relaxed);
        // The tapes of trace following carry the plans they replay: they are stamped with the epoch.
        self.plan_epoch.fetch_add(1, Ordering::Relaxed);
        lock(&self.plans).clear();
    }

    /// The evaluation `eval` asked for was postponed (see [`follow`]).
    pub(crate) fn follow_postponed(&self) {
        if self.follow_on() {
            follow::postponed(self);
        }
    }

    /// Prints what trace following did so far (`LUMA_TRACE_FOLLOW=+stats`).
    pub(crate) fn follow_report(&self) {
        self.follow.report();
    }

    /// Whether trace following is on for this context (`LUMA_TRACE_FOLLOW`).
    #[inline(always)]
    fn follow_on(&self) -> bool {
        self.follow.on()
    }

    /// Switches trace following on or off for this context (tests).
    #[cfg(test)]
    pub(crate) fn set_follow(&self, on: bool) {
        self.follow.set(on);
    }

    /// What trace following did so far.
    #[cfg(test)]
    pub(crate) fn follow_stats(&self) -> follow::StatsSnapshot {
        self.follow.snapshot()
    }

    /// What the plan cache did so far.
    pub(crate) fn plan_stats(&self) -> plancache::StatsSnapshot {
        self.plan_stats.snapshot()
    }

    /// Plans held by the cache (tests).
    #[cfg(test)]
    pub(crate) fn plan_entries(&self) -> usize {
        lock(&self.plans).len()
    }

    /// The most launches one cached plan makes (tests).
    #[cfg(test)]
    pub(crate) fn longest_plan(&self) -> usize {
        lock(&self.plans).longest()
    }

    #[cfg(test)]
    pub(crate) fn requested_len(&self) -> usize {
        lock(&self.requested).nodes.len()
    }

    #[cfg(test)]
    pub(crate) fn set_max_nodes(&self, max_nodes: usize) {
        self.max_nodes.store(max_nodes.clamp(1, MAX_NODES_LIMIT), Ordering::Relaxed);
    }

    #[cfg(test)]
    pub(crate) fn stats(&self) -> StatsSnapshot {
        StatsSnapshot {
            built: self.stats.built.load(Ordering::Relaxed),
            hits: self.stats.hits.load(Ordering::Relaxed),
            elementwise_launches: self.stats.elementwise_launches.load(Ordering::Relaxed),
            reduce_launches: self.stats.reduce_launches.load(Ordering::Relaxed),
            reduce_passes: self.stats.reduce_passes.load(Ordering::Relaxed),
            replays: self.stats.replays.load(Ordering::Relaxed),
            speculated: self.stats.speculated.load(Ordering::Relaxed),
            transposes: self.stats.transposes.load(Ordering::Relaxed),
            coarsened: self.stats.coarsened.load(Ordering::Relaxed),
        }
    }

    /// Evaluates every lazy array that is still alive, newest first (an expression is looked at
    /// before the operands it fused, which usually died with it). Arrays of the same length are
    /// computed together, as many per kernel as the bindings allow: inputs they share are read
    /// once, and one of them that another uses is computed once.
    pub(crate) fn request(&self, node: &Arc<Node>) {
        if node.requested.swap(true, Ordering::Relaxed) {
            return;
        }
        let mut requested = lock(&self.requested);
        if requested.nodes.len() >= requested.prune_at.max(REQUESTED_PRUNE_AT) {
            requested.nodes.retain(|weak| weak.upgrade().is_some_and(|node| node.is_pending()));
            requested.prune_at = requested.nodes.len().saturating_mul(2);
        }
        requested.nodes.push(Arc::downgrade(node));
    }

    /// Whether one of the newest requested nodes reads other expressions through views and
    /// depends on a requested node that is still pending: a stencil step whose input is the
    /// postponed result of the step before. Postponing such steps chains them into one expression
    /// (the stencil recomputes its input at every index it reads), so they are computed as they
    /// are requested.
    pub(crate) fn requested_stencil_chain(&self) -> bool {
        const NEWEST: usize = 8;
        let requested = lock(&self.requested);
        let pending = |weak: &Weak<Node>| weak.upgrade().filter(|node| node.is_pending());
        requested.nodes.iter().rev().take(NEWEST).filter_map(pending).filter(|node| node.shift_depth() > 0).any(|node| {
            requested.nodes.iter().filter_map(pending).any(|other| !Arc::ptr_eq(&node, &other) && node.below_contains(other.id))
        })
    }

    /// Computes the requested nodes that are still alive and were created by this thread, those
    /// of one length together in multi-output kernels. What other threads requested stays
    /// requested: they are building and flattening those expressions right now, and a recording
    /// they may have under way must not be fed with launches of another thread (see
    /// [`Node::thread`]).
    pub(crate) fn evaluate_requested(&self) {
        let me = current_thread();
        let requested = std::mem::take(&mut lock(&self.requested).nodes);
        let mut others = Vec::new();
        let mut pending: Vec<Arc<Node>> = Vec::with_capacity(requested.len());
        for weak in requested {
            let Some(node) = weak.upgrade() else { continue };
            if node.thread != me {
                others.push(weak);
                continue;
            }
            node.requested.store(false, Ordering::Relaxed);
            if node.is_pending() {
                pending.push(node);
            }
        }
        if !others.is_empty() {
            lock(&self.requested).nodes.extend(others);
        }
        plancache::evaluate(self, pending, |pending| self.evaluate_pending(pending));
    }

    /// Computes `pending` (pending nodes of this thread), those of one length together.
    fn evaluate_pending(&self, mut pending: Vec<Arc<Node>>) {
        while let Some(first) = pending.first() {
            let len = first.len;
            let (same, rest): (Vec<_>, Vec<_>) = pending.into_iter().partition(|node| node.len == len);
            evaluate_group(&same);
            pending = rest;
        }
    }

    /// Computes the lazy arrays this thread created that are still alive (see
    /// [`JitState::evaluate_requested`] for the other threads'). The ones whose value was only
    /// computed inside the kernel of another array stay lazy: they are usually temporaries.
    pub(crate) fn evaluate_live(&self) {
        self.evaluate_swept(false);
    }

    /// [`JitState::evaluate_live`], and also the arrays of this thread that were only computed
    /// inside another array's kernel: afterwards no array this thread holds is lazy. A traced loop
    /// needs that (see `Trace::start`): a lazy array it reads would be evaluated by the recording.
    pub(crate) fn evaluate_held(&self) {
        self.evaluate_swept(true);
    }

    /// How many kernels this context has generated and compiled so far (cache hits are not counted).
    pub(crate) fn kernels_built(&self) -> u64 {
        self.stats.built.load(Ordering::Relaxed)
    }

    /// The sweep of [`JitState::evaluate_live`]; `held` also computes the arrays that were
    /// fused into the kernel of another one.
    fn evaluate_swept(&self, held: bool) {
        // The live list is what this reads: a step that follows a tape ends here.
        follow::interrupt(self);
        let me = current_thread();
        let nodes = self.with_live(|live| live.take_if(|node| node.thread == me));
        let mut pending: Vec<Arc<Node>> = nodes.into_iter().rev().filter_map(|weak| weak.upgrade()).collect();
        while let Some(first) = pending.first() {
            let len = first.len;
            let (same, mut rest): (Vec<_>, Vec<_>) = pending.into_iter().partition(|node| node.len == len);
            let mut group: Vec<Arc<Node>> = Vec::new();
            let (mut inputs, mut below) = (Vec::new(), IdSet::default());
            for node in same {
                // Computed inside a kernel of an earlier group (the value of an array is wanted
                // all the same when `held`).
                let kept = held && node.handles.load(Ordering::Relaxed) > 0;
                if !node.is_pending() || (node.fused.load(Ordering::Relaxed) && !kept) {
                    // Still wanted by a later `evaluate_held`.
                    if !held && node.is_pending() && node.handles.load(Ordering::Relaxed) > 0 {
                        self.with_live(|live| live.add_weak(Arc::downgrade(&node)));
                    }
                    continue;
                }
                // Part of an expression of the group: it is computed inside that kernel and only
                // evaluated on its own afterwards if it is still pending then.
                if below.contains(node.id) {
                    rest.push(node);
                    continue;
                }
                let mut tmp = Vec::new();
                let mut more_inputs = inputs.clone();
                union_into(&mut more_inputs, &mut tmp, &self.inputs_within(&node, &inputs, false, self.max_inputs()));
                let (_, new_below) = below.overlap(&node);
                let fits = more_inputs.len() + self.output_bindings(&group, &node) <= self.max_inputs() + 1
                    && below.len() + new_below + group.len() < MAX_NODES_LIMIT;
                if fits || group.is_empty() {
                    inputs = more_inputs;
                    below.add(&node, false);
                    group.push(node);
                } else {
                    rest.push(node);
                }
            }
            launch::evaluate_group(&group);
            pending = rest;
        }
    }

    /// The ids of the buffers the expression of `node` reads, sorted: those it was made with
    /// (see [`Node::inputs`]), unless they would not fit next to the `have` ones in `room`
    /// bindings. Evaluations since then have replaced what the expression read by buffers that
    /// may be fewer (or windows of one pack), so then they are counted again.
    fn inputs_within(&self, node: &Arc<Node>, have: &[u64], recounted: bool, room: usize) -> Vec<u64> {
        let made = node.inputs();
        if !recounted {
            let extra = made.iter().filter(|id| have.binary_search(id).is_err()).count();
            if have.len() + extra < room {
                return made.to_vec();
            }
        }
        match node.operands() {
            Some(args) => count_inputs(args.get(..node.op.kind.arity()).unwrap_or(&[]), &[]),
            None => made.to_vec(),
        }
    }

    /// The buffers a kernel binds for the outputs of `group` and `node`: one each, or one for all
    /// of them if they are written into a pack (see [`launch::run_packed`]).
    fn output_bindings(&self, group: &[Arc<Node>], node: &Arc<Node>) -> usize {
        let count = group.len() + 1;
        let ty = node.op.out_ty();
        let packed = group.iter().all(|member| member.op.out_ty() == ty && member.len == node.len) && launch::packable(&node.ctx, count, node.len, ty);
        if packed { 1 } else { count }
    }

    /// Evaluates `seed` (an operand of an expression that got too big) together with the live
    /// pending arrays of its length that read nodes of its expression, newest first, as many as
    /// one kernel takes: [`MAX_NODES_LIMIT`] operations (what [`Context::flush`] allows) and the
    /// bindings the device allows. Computing them together evaluates what they share once.
    pub(crate) fn evaluate_alongside(&self, seed: Arc<Node>) {
        let me = current_thread();
        let max_inputs = self.max_inputs();
        let mut recounted = false;
        let mut packed = false;
        let mut inputs = self.inputs_within(&seed, &[], false, max_inputs);
        let mut below = IdSet::of(&seed, true);
        let len = seed.len;
        let plain_seed = seed.shift_depth() == 0;
        let mut group = vec![seed];
        let flying = FLATTENING.with_borrow(Clone::clone);
        // A node that reads a node this thread is computing right now cannot be computed before
        // that one is: it would wait for it forever.
        let computing = COMPUTING.with_borrow(Clone::clone);
        let held_in_below = |node: &Node| {
            plain_seed
                && node.shift_depth() == 0
                && node.handles.load(Ordering::Relaxed) > 0
                && node.shifted_reads.load(Ordering::Relaxed) == 0
                && !node.is_pending_view()
        };
        let mut candidates = Vec::new();
        self.with_live(|live| live.for_each_newest(|weak| {
            let Some(node) = weak.upgrade() else { return };
            if node.len != len
                || node.thread != me
                || !node.is_pending()
                || node.fused.load(Ordering::Relaxed)
                || flying.contains(&Arc::as_ptr(&node))
                || computing.iter().any(|&id| node.below_contains(id))
                || group.iter().any(|m| m.id == node.id)
                || (below.contains(node.id) && !held_in_below(&node))
            {
                return;
            }
            candidates.push(node);
        }));
        let mut tmp = Vec::new();
        for (idx, node) in candidates.iter().enumerate() {
            if group.iter().any(|m| m.id == node.id) || (below.contains(node.id) && !held_in_below(node)) {
                continue;
            }
            let (shared_below, new_below) = if below.contains(node.id) { (1, 0) } else { below.overlap(node) };
            if shared_below == 0 || below.len() + new_below + group.len() >= MAX_NODES_LIMIT + 1024 {
                continue;
            }
            let mut node_inputs = self.inputs_within(node, &inputs, recounted, max_inputs);
            let (_, mut new_inputs) = overlap(&node_inputs, &inputs);
            let out_bindings = self.output_bindings(&group, node);
            if inputs.len() + new_inputs + out_bindings > max_inputs + 1 && !recounted {
                recounted = true;
                let srcs: Vec<Src> = group.iter().map(|m| Src::Node(Arc::clone(m))).collect();
                inputs = count_inputs(&srcs, &[]);
                node_inputs = self.inputs_within(node, &inputs, true, max_inputs);
                (_, new_inputs) = overlap(&node_inputs, &inputs);
            }
            if inputs.len() + new_inputs + out_bindings > max_inputs + 1 && !packed {
                packed = true;
                let all: Vec<Arc<Node>> = group.iter().chain(&candidates[idx..]).cloned().collect();
                if pack::repack_nodes(&group[0].ctx, &all, usize::MAX) {
                    let srcs: Vec<Src> = group.iter().map(|m| Src::Node(Arc::clone(m))).collect();
                    inputs = count_inputs(&srcs, &[]);
                    node_inputs = self.inputs_within(node, &inputs, true, max_inputs);
                    (_, new_inputs) = overlap(&node_inputs, &inputs);
                }
            }
            if inputs.len() + new_inputs + out_bindings <= max_inputs + 1 {
                union_into(&mut inputs, &mut tmp, &node_inputs);
                below.add(node, true);
                group.push(Arc::clone(node));
            }
        }
        drop(candidates);
        match group.as_slice() {
            // On its own, nothing but the node holds its operands, which lets its kernel
            // overwrite a dying operand's buffer.
            [node] => {
                let _ = node.force();
            }
            _ => launch::evaluate_group(&group),
        }
    }

    /// The kernel for `key`, generated and compiled on first use. `key` first gets the numbers
    /// of its views, gathers and scatter read from the uniform (see [`params`]), and the extents
    /// of its array if it stages values in workgroup memory ([`params::apply_tile`],
    /// [`params::apply_staging`]), which `scalars` then holds too: kernels that differ in
    /// those only are one kernel. Compiling takes no lock of the cache: threads that need other
    /// kernels carry on meanwhile (see [`BuildCache`]).
    pub(crate) fn kernel(&self, ctx: &Context, key: &mut program::FusedKey, scalars: &mut Vec<u32>) -> Result<Arc<KernelCore>> {
        #[cfg(test)]
        {
            self.last_program_len.store(key.instrs.len(), Ordering::Relaxed);
            *lock(&self.last_staging) = key.staging.clone();
        }
        let fast_hash = fast_key_hash(key);
        if let Some(hit) = lock(&self.fast_kernels).get(&fast_hash).and_then(|list| list.iter().find(|entry| entry.matches(key))) {
            scalars.extend_from_slice(&hit.extra_words);
            key.params = hit.params.clone();
            #[cfg(test)]
            if let Some((inlined_len, branches)) = hit.inlined_stats {
                self.last_inlined_len.store(inlined_len, Ordering::Relaxed);
                self.last_branches.store(branches, Ordering::Relaxed);
            }
            self.stats.hits.fetch_add(1, Ordering::Relaxed);
            return Ok(Arc::clone(&hit.core));
        }
        let orig_instrs = Arc::clone(&key.instrs);
        let orig_subs = Arc::clone(&key.subs);
        let orig_input_tys = key.input_tys.clone();
        let orig_scalar_tys = key.scalar_tys.clone();
        let orig_outputs = key.outputs.clone();
        let orig_tail = key.tail;
        let orig_in_place = key.in_place.clone();
        let orig_runs = key.runs;
        let orig_spans = key.spans.clone();
        let orig_staging = key.staging.clone();
        let orig_tile = key.tile.clone();
        let orig_guard = key.guard;
        let orig_coarsen = key.coarsen;
        let base_words = scalars.len();

        if key.staging.is_none() && key.tile.is_none() {
            self.parameterize(ctx, key, scalars);
        } else {
            params::apply_staging(ctx, key, scalars);
            params::apply_tile(key, scalars);
        }
        let key_ref = &*key;
        let (core, built) = self.kernels.get_or_build(key_ref, || self.build_kernel(ctx, key_ref))?;
        if !built {
            self.stats.hits.fetch_add(1, Ordering::Relaxed);
        }
        let extra_words: Box<[u32]> = scalars[base_words..].into();
        let params_marker = key.params.as_ref().map(|p| {
            Arc::new(params::Parameterized {
                lowered: None,
                slots: p.slots,
            })
        });
        #[cfg(test)]
        let inlined_stats = key
            .params
            .as_ref()
            .and_then(|p| match &p.lowered {
                Some(Lowered::Plain(inl)) => {
                    Some((inl.instrs.len(), inl.branch.iter().filter(|&&b| b).count()))
                }
                Some(Lowered::Staged(programs)) => Some((
                    programs.main.instrs.len(),
                    programs.bodies().map(|inl| inl.branch.iter().filter(|&&b| b).count()).sum(),
                )),
                None => None,
            });
        #[cfg(test)]
        if let Some((inlined_len, branches)) = inlined_stats {
            self.last_inlined_len.store(inlined_len, Ordering::Relaxed);
            self.last_branches.store(branches, Ordering::Relaxed);
        }
        let mut fast = lock(&self.fast_kernels);
        if fast.len() >= 4096 {
            fast.clear();
        }
        fast.entry(fast_hash).or_default().push(FastKernel {
            instrs: orig_instrs,
            subs: orig_subs,
            input_tys: orig_input_tys,
            scalar_tys: orig_scalar_tys,
            outputs: orig_outputs,
            tail: orig_tail,
            in_place: orig_in_place,
            runs: orig_runs,
            spans: orig_spans,
            staging: orig_staging,
            tile: orig_tile,
            guard: orig_guard,
            coarsen: orig_coarsen,
            extra_words,
            params: params_marker,
            core: Arc::clone(&core),
            #[cfg(test)]
            inlined_stats,
        });
        Ok(core)
    }

    /// Lowers `key` if it [inlines](program::FusedKey::inlines) and rewrites it by [`params::apply`].
    fn parameterize(&self, ctx: &Context, key: &mut program::FusedKey, scalars: &mut Vec<u32>) {
        let inlined = if key.inlines() {
            match inline(key, usize::MAX, &ctx.cost_model()) {
                Verdict::Fits(inlined) => Some(inlined),
                Verdict::TooBig { .. } => None,
            }
        } else {
            None
        };
        #[cfg(debug_assertions)]
        {
            let (inputs, slots) = (key.input_tys.len(), key.scalar_tys.len());
            let checked = program::validate(key, inputs, slots).and_then(|()| match &inlined {
                Some(inlined) => inline::validate_inlined(inlined, inputs, false),
                None => Ok(()),
            });
            if let Err(message) = checked {
                panic!("invalid fused program: {message}\n{key:?}");
            }
        }
        #[cfg(test)]
        if let Some(inlined) = &inlined {
            self.last_inlined_len.store(inlined.instrs.len(), Ordering::Relaxed);
            self.last_branches.store(inlined.branch.iter().filter(|&&branch| branch).count(), Ordering::Relaxed);
        }
        params::apply(key, inlined.as_ref(), scalars);
    }

    /// Generates and compiles the kernel of `key`.
    fn build_kernel(&self, ctx: &Context, key: &program::FusedKey) -> Result<Arc<KernelCore>> {
        let segment_len = u32::try_from(ctx.segment_bytes() / 4).unwrap_or(u32::MAX);
        // Whatever the judgement was, the kernel runs the program as it is: `flatten_inner` cut
        // the ones the model wants cut, any other that got here is lowered in full.
        let cost = ctx.cost_model();
        let lowered = match (&key.params, &key.staging) {
            (Some(params), _) => params.lowered.clone(),
            (None, Some(layout)) => Some(Lowered::Staged(
                inline::lower_staged(key, layout, usize::MAX, &cost).expect("a staged program lowers without a cap"),
            )),
            (None, None) => key.inlines().then(|| match inline(key, usize::MAX, &cost) {
                Verdict::Fits(inlined) => Some(Lowered::Plain(inlined)),
                Verdict::TooBig { .. } => None,
            }).flatten(),
        };
        #[cfg(debug_assertions)]
        if key.params.is_none() {
            let (inputs, scalars) = (key.input_tys.len(), key.scalar_tys.len());
            let checked = program::validate(key, inputs, scalars).and_then(|()| match &lowered {
                Some(Lowered::Plain(inlined)) => inline::validate_inlined(inlined, inputs, false),
                Some(Lowered::Staged(programs)) => programs.bodies().try_for_each(|inlined| inline::validate_inlined(inlined, inputs, true)),
                None => Ok(()),
            });
            if let Err(message) = checked {
                panic!("invalid fused program: {message}\n{key:?}");
            }
        }
        #[cfg(test)]
        if key
            .params
            .as_ref()
            .is_none_or(|params| matches!(params.lowered, Some(Lowered::Staged(_))))
        {
            match &lowered {
                Some(Lowered::Plain(inlined)) => self.last_inlined_len.store(inlined.instrs.len(), Ordering::Relaxed),
                Some(Lowered::Staged(programs)) => self.last_inlined_len.store(programs.main.instrs.len(), Ordering::Relaxed),
                None => {}
            }
            let count = |inlined: &inline::Inlined| inlined.branch.iter().filter(|&&branch| branch).count();
            let branches = match &lowered {
                Some(Lowered::Plain(inlined)) => count(inlined),
                Some(Lowered::Staged(programs)) => programs.bodies().map(count).sum::<usize>(),
                None => 0,
            };
            self.last_branches.store(branches, Ordering::Relaxed);
        }
        let built = codegen::build(key, ctx.tuning(), segment_len, lowered.as_ref())?;
        let label = key.label();
        let started = log::log_enabled!(log::Level::Debug).then(std::time::Instant::now);
        let (core, size) = match built {
            codegen::Built::Module(module) => {
                let statements = format!("{} statements of Kernel IR", module.entry_body.len());
                (kernel::build_module(ctx, module, Some(&label), luma_core::KernelTrust::Trusted)?, statements)
            }
            codegen::Built::Wgsl(source) => {
                let bytes = format!("{} bytes of WGSL", source.len());
                (kernel::build_core(ctx, &KernelKey::new(source, "main", None), Some(&label), luma_core::KernelTrust::Trusted)?, bytes)
            }
        };
        let core = Arc::new(core);
        if let Some(started) = started {
            log::debug!("luma: compiled {label} ({size}) in {:.1} ms", started.elapsed().as_secs_f64() * 1e3);
        }
        self.stats.built.fetch_add(1, Ordering::Relaxed);
        Ok(core)
    }

    /// The kernel that runs the store program `key` for several steps with its states
    /// (`states[k]`: the state input `k` reads) in registers, generated and compiled on first use.
    pub(crate) fn loop_kernel(&self, ctx: &Context, key: &program::FusedKey, states: &[Option<usize>]) -> Result<Arc<KernelCore>> {
        if let Some(core) = lock(&self.loops).get(key).and_then(|known| known.iter().find(|(layout, _)| layout == states)) {
            return Ok(Arc::clone(&core.1));
        }
        let label = format!("Loop{}", key.label().trim_start_matches("Fused"));
        let core = self.module_kernel(ctx, format!("loop {key:?} {states:?}"), &label, || kirgen::loop_module(key, states, ctx.tuning()))?;
        lock(&self.loops).entry(key.clone()).or_default().push((states.to_vec(), Arc::clone(&core)));
        Ok(core)
    }

    /// The stencil kernel with WGSL `source`, compiled on first use.
    pub(crate) fn stencil_kernel(&self, ctx: &Context, source: String, label: &str) -> Result<Arc<KernelCore>> {
        let (core, built) = self.stencils.get_or_build(&source, || {
            let started = log::log_enabled!(log::Level::Debug).then(std::time::Instant::now);
            let key = KernelKey::new(source.clone(), "main", None);
            let core = kernel::build_core(ctx, &key, Some(label), luma_core::KernelTrust::Trusted).map(Arc::new);
            if let Some(started) = started {
                log::debug!("luma: compiled {label} ({} bytes of WGSL) in {:.1} ms", source.len(), started.elapsed().as_secs_f64() * 1e3);
            }
            core
        })?;
        if built {
            self.stats.built.fetch_add(1, Ordering::Relaxed);
        }
        Ok(core)
    }

    /// The scan or reduction kernel whose WGSL `source` generates is a function of `key` alone:
    /// `source` runs, and the kernel compiles, on first use of the key.
    pub(crate) fn scan_kernel(&self, ctx: &Context, key: String, label: &str, source: impl FnOnce() -> String) -> Result<Arc<KernelCore>> {
        self.scans.get_or_build(&key, || self.stencil_kernel(ctx, source(), label)).map(|(core, _)| core)
    }

    /// The kernel built as Kernel IR by `build`, compiled on first use; `key` names what the
    /// module depends on, so that distinct modules have distinct keys.
    pub(crate) fn module_kernel(
        &self,
        ctx: &Context,
        key: String,
        label: &str,
        build: impl FnOnce() -> Result<KernelModule>,
    ) -> Result<Arc<KernelCore>> {
        let (core, built) = self.modules.get_or_build(&key, || {
            let started = log::log_enabled!(log::Level::Debug).then(std::time::Instant::now);
            let module = build()?;
            let statements = module.entry_body.len();
            let core = kernel::build_module(ctx, module, Some(label), luma_core::KernelTrust::Trusted).map(Arc::new);
            if let Some(started) = started {
                log::debug!("luma: compiled {label} ({statements} statements of Kernel IR) in {:.1} ms", started.elapsed().as_secs_f64() * 1e3);
            }
            core
        })?;
        if built {
            self.stats.built.fetch_add(1, Ordering::Relaxed);
        }
        Ok(core)
    }
}

/// Pre-parameterized lookup entry for [`JitState::fast_kernels`].
#[derive(Debug)]
struct FastKernel {
    instrs: Arc<Vec<Instr>>,
    subs: Arc<Vec<Vec<Instr>>>,
    input_tys: Vec<ElemTy>,
    scalar_tys: Vec<ElemTy>,
    outputs: Vec<u32>,
    tail: program::Tail,
    in_place: Vec<(u32, u32)>,
    runs: bool,
    spans: Vec<u8>,
    staging: Option<staging::Staging>,
    tile: Option<codegen::Tile>,
    guard: Option<u32>,
    coarsen: u8,
    extra_words: Box<[u32]>,
    params: Option<Arc<params::Parameterized>>,
    core: Arc<KernelCore>,
    #[cfg(test)]
    inlined_stats: Option<(usize, usize)>,
}

impl FastKernel {
    fn matches(&self, key: &program::FusedKey) -> bool {
        (Arc::ptr_eq(&self.instrs, &key.instrs) || self.instrs == key.instrs)
            && (Arc::ptr_eq(&self.subs, &key.subs) || self.subs == key.subs)
            && self.tail == key.tail
            && self.in_place == key.in_place
            && self.runs == key.runs
            && self.input_tys == key.input_tys
            && self.scalar_tys == key.scalar_tys
            && self.outputs == key.outputs
            && self.spans == key.spans
            && self.staging == key.staging
            && self.tile == key.tile
            && self.guard == key.guard
            && self.coarsen == key.coarsen
    }
}

/// A cut as the tests read it.
#[cfg(test)]
#[derive(Debug)]
#[allow(dead_code)]
pub(crate) struct TestCut {
    pub(crate) op: &'static str,
    pub(crate) location: &'static Location<'static>,
    pub(crate) reason: SplitReason,
    decision: Decision,
}

#[cfg(test)]
impl std::fmt::Display for TestCut {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.decision.fmt(f)
    }
}

/// `LUMA_RECORD_DECISIONS=1`: contexts start with decision recording on.
fn env_records_decisions() -> bool {
    std::env::var("LUMA_RECORD_DECISIONS").is_ok_and(|v| matches!(v.trim().to_ascii_lowercase().as_str(), "1" | "on" | "true"))
}

/// The stable label of a cut: its values are counts and limits, never addresses.
fn cut_label(reason: SplitReason) -> String {
    match reason {
        SplitReason::Nodes { nodes, limit } => format!("nodes {nodes}>{limit}"),
        SplitReason::Depth { limit } => format!("depth >{limit}"),
        SplitReason::Inputs { inputs, limit } => format!("inputs {inputs}>{limit}"),
        SplitReason::Live { live, limit } => format!("live {live}>{limit}"),
        SplitReason::Scalars { words, limit } => format!("scalars {words}>{limit}"),
        SplitReason::Inlined { instructions, limit } => format!("inlined {instructions}>{limit}"),
        SplitReason::Materialised { instances } => format!("materialised x{instances}"),
        SplitReason::Shared { weight } => format!("shared weight {weight}"),
        #[allow(unreachable_patterns)]
        _ => format!("{reason:?}"),
    }
}

fn fast_key_hash(key: &program::FusedKey) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = FxHasher::default();
    key.shape.hash(&mut hasher);
    key.tail.hash(&mut hasher);
    key.in_place.hash(&mut hasher);
    key.runs.hash(&mut hasher);
    key.spans.hash(&mut hasher);
    key.staging.hash(&mut hasher);
    key.tile.hash(&mut hasher);
    key.guard.hash(&mut hasher);
    key.coarsen.hash(&mut hasher);
    hasher.finish()
}
