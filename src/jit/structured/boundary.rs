#![allow(dead_code)] // the families take their types over one port at a time
//! What the eager-boundary families (sort, scan, scatter; search, compact and merge as light ports) tell the registry: the
//! attributes of a [`Boundary`] and the parameters a strategy's `fits` hands to the op's own body. Plain `Copy` data, built on the
//! stack by the op; nothing here evaluates anything or holds a node.

use std::borrow::Cow;
use std::marker::PhantomData;
use std::panic::Location;

use super::strategy::{note, Boundary, BoundaryAttrs, FamilyTag, Params, StrategyId};
use crate::context::Context;
use crate::profile::{Decision, DecisionSite};

impl<'a> Boundary<'a> {
    /// The descriptor of an op of `family` at `location`: `dims` is the shape that picks the plan.
    #[inline]
    pub(crate) fn new(family: FamilyTag, op: &'static str, location: &'static Location<'static>, dims: [u32; 4], attrs: BoundaryAttrs) -> Boundary<'a> {
        Boundary { family, op, location, dims, attrs, _borrow: PhantomData }
    }
}

/// `usize` as the `u32` of a [`Boundary`]'s `dims` (saturating: the exact size is in the attributes where a `fits` needs it).
#[inline]
pub(crate) fn dim(size: usize) -> u32 {
    u32::try_from(size).unwrap_or(u32::MAX)
}

// -- Sort --

/// Which selection a sort boundary stands for: the op chains several (the host path at its entry, the radix variant in
/// `radix_sort`, the axis and by-key compositions), and each strategy applies at one of them.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum SortStage {
    /// The entry of `sort`, `argsort`, `sort_with_indices`: the host path or on to the radix sort.
    Entry,
    /// `radix_sort`: which of its kernels.
    Radix,
    /// `sort_by_key_bits`: a payload through the radix passes or the permutation.
    ByKey,
    /// `sort_axis`: one kernel per short line or a sort of everything by line.
    Axis,
    /// `sort_by_key_axis`.
    ByKeyAxis,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum SortPayload {
    None,
    Index,
    Buffer,
}

/// What a sort boundary reads; `dims` is `[n, stride, lines, 0]` (the whole array: `[len, 1, 1, 0]`).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) struct SortAttrs {
    pub(crate) stage: SortStage,
    pub(crate) payload: SortPayload,
    /// The array holds an error (the host path would read it).
    pub(crate) errored: bool,
    /// The by-key values are 4-byte plain elements (they travel through the radix passes).
    pub(crate) value4: bool,
    /// The keys differ in their low `key_bits` bits.
    pub(crate) key_bits: u8,
    /// Elements in the sort (`len`).
    pub(crate) len: u64,
}

/// What a radix sort runs with: keys per invocation of a sweep tile, keys per tile, passes.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct SortParams {
    pub(crate) per_thread: u32,
    pub(crate) tile: u32,
    pub(crate) passes: u32,
}

impl SortParams {
    pub(crate) const NONE: SortParams = SortParams { per_thread: 0, tile: 0, passes: 0 };
}

// -- Scan --

/// What a scan boundary reads; `dims` is `[n, stride, lines, 0]`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) struct ScanAttrs {
    /// The operator groups however it likes (`u32`/`i32`, or an extremum): a look-back may take any earlier chunk's prefix.
    pub(crate) grouping_free: bool,
    pub(crate) by_key: bool,
    /// What the scan reads: an evaluated buffer, or a pending expression (cheap to compute again, or worth storing).
    pub(crate) operand: ScanOperand,
}

/// The operand of a scan, as far as the choice of its prologue needs to know.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum ScanOperand {
    Buffer,
    /// A pending expression that costs less to compute in each kernel that reads it than to write and read back
    /// (`jit::worth_storing` is false).
    Cheap,
    /// A pending expression that is worth storing: computed in the kernels only where one kernel reads it.
    Costly,
}

/// What a scan runs with: `vec` words per load of a warp pass, `nb` chunks (blocks) per line, `k` cells per block.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct ScanParams {
    pub(crate) vec: u32,
    pub(crate) nb: u32,
    pub(crate) k: u32,
    /// The kernels compute a pending operand themselves (the strategy's prologue): `prologue=fused` in the Decision.
    pub(crate) fused: bool,
}

impl ScanParams {
    pub(crate) const NONE: ScanParams = ScanParams { vec: 0, nb: 1, k: 0, fused: false };
}

// -- Scatter --

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum ScatterOp {
    None,
    Add,
    Min,
    Max,
}

impl ScatterOp {
    pub(crate) fn of(atomic: Option<crate::jit::AtomicOp>) -> ScatterOp {
        match atomic {
            None => ScatterOp::None,
            Some(crate::jit::AtomicOp::Add) => ScatterOp::Add,
            Some(crate::jit::AtomicOp::Min) => ScatterOp::Min,
            Some(crate::jit::AtomicOp::Max) => ScatterOp::Max,
        }
    }
}

/// What a scatter boundary reads; `dims` is `[base_len, count, 0, 0]` (saturated).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) struct ScatterAttrs {
    pub(crate) atomic: ScatterOp,
    pub(crate) base_len: u64,
    pub(crate) count: u64,
    pub(crate) base_is_view: bool,
    pub(crate) base_is_read_view: bool,
    /// The host-path site of `put_with` (false: the kernel site in `assign::scatter`).
    pub(crate) host_site: bool,
}

/// What a scatter runs with: the length of the target.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct ScatterParams {
    pub(crate) len: u32,
}

// -- Light ports: tag, site and `note` --

/// `FamilyTag::Compact`: the tile count and scatter of a compaction.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) struct CompactAttrs {
    pub(crate) padded: bool,
    pub(crate) words: bool,
}

/// `FamilyTag::Merge`: merge of two sorted arrays of `total` elements, or a factorize.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) struct MergeAttrs {
    pub(crate) total: u64,
}

/// Records the Decision of a light-port op (`FamilyTag::{Search, Compact, Merge}`: no table, no `fits`, no candidates): `label`
/// is the legacy `chosen` text. One relaxed load while nobody records.
#[inline]
pub(crate) fn note_light(ctx: &Context, family: FamilyTag, op: &'static str, location: &'static Location<'static>, label: impl Into<Cow<'static, str>>) {
    if !ctx.jit().recording_decisions() {
        return;
    }
    let site = match family {
        FamilyTag::Search => DecisionSite::SearchPlan,
        FamilyTag::Compact => DecisionSite::CompactPlan,
        FamilyTag::Merge => DecisionSite::MergePlan,
        _ => unreachable!("only the light ports note a label"),
    };
    ctx.jit().note_decision(Decision::new(op, location, site, label));
}

/// [`note`] for a body that chose `id` with `params` (here so that the families' files import one module).
#[inline]
pub(crate) fn note_chosen(ctx: &Context, boundary: &Boundary<'_>, id: StrategyId, params: &Params, declined: &[StrategyId]) {
    note(ctx, boundary, id, params, declined);
}
