//! The sort family: the host path at the entry of `sort`/`argsort`/`sort_with_indices`, the radix kernels in `radix_sort`, and the
//! compositions (axis sorts, by-key sorts). Each strategy applies at one [`SortStage`]; the op builds the boundary of the
//! selection it is at and runs the body of the id it gets. The `fits` functions are the inline conditions of `src/sort.rs` at
//! `f0e5723`.

use super::{boundary, entry, id, split_global, split_map, FitsInputs};
use crate::caps::Caps;
use crate::context::Context;
use crate::jit::structured::boundary::{SortAttrs, SortParams, SortPayload, SortStage};
use crate::jit::structured::strategy::{BoundaryAttrs, FamilyTag, Params, Residencies, Strategy, StrategyId, Subject};
use crate::sort::{sweep_per_thread, use_ballots, use_sweep, BALLOT_PER_THREAD, DIGIT_BITS, TILE, WORKGROUP};

impl StrategyId {
    pub(crate) const SORT_HOST: StrategyId = id(FamilyTag::Sort, 0);
    pub(crate) const SORT_RADIX_SWEEP: StrategyId = id(FamilyTag::Sort, 1);
    pub(crate) const SORT_RADIX_BALLOTS: StrategyId = id(FamilyTag::Sort, 2);
    pub(crate) const SORT_RADIX_COUNT_SCAN: StrategyId = id(FamilyTag::Sort, 3);
    pub(crate) const SORT_LINES: StrategyId = id(FamilyTag::Sort, 4);
    pub(crate) const SORT_LINES_RADIX: StrategyId = id(FamilyTag::Sort, 5);
    pub(crate) const SORT_BY_KEY_PAYLOAD: StrategyId = id(FamilyTag::Sort, 6);
    pub(crate) const SORT_BY_KEY_PERMUTATION: StrategyId = id(FamilyTag::Sort, 7);
    pub(crate) const SORT_BY_KEY_AXIS: StrategyId = id(FamilyTag::Sort, 8);
}

/// Table order is today's preference: the host path, then sweep before ballots before the count/scan/scatter passes.
pub(crate) static SORT_STRATEGIES: [Strategy; 9] = [
    entry(StrategyId::SORT_HOST, "host", Caps::HOST_PATH, FitsInputs::HOST, split_global, fits_host),
    entry(StrategyId::SORT_RADIX_SWEEP, "radix_sweep", Caps::LOOKBACK.with(Caps::SUBGROUP32), FitsInputs::NONE, split_global, fits_radix_sweep),
    entry(StrategyId::SORT_RADIX_BALLOTS, "radix_ballots", Caps::SUBGROUP32, FitsInputs::NONE, split_global, fits_radix_ballots),
    entry(StrategyId::SORT_RADIX_COUNT_SCAN, "radix_count_scan", Caps::empty(), FitsInputs::NONE, split_global, fits_radix_count_scan),
    entry(StrategyId::SORT_LINES, "lines", Caps::empty(), FitsInputs::NONE, split_map, fits_lines),
    entry(StrategyId::SORT_LINES_RADIX, "lines_radix", Caps::empty(), FitsInputs::NONE, split_global, fits_lines_radix),
    entry(StrategyId::SORT_BY_KEY_PAYLOAD, "by_key_payload", Caps::empty(), FitsInputs::HOST, split_global, fits_by_key_payload),
    entry(StrategyId::SORT_BY_KEY_PERMUTATION, "by_key_permutation", Caps::empty(), FitsInputs::NONE, split_global, fits_by_key_permutation),
    entry(StrategyId::SORT_BY_KEY_AXIS, "by_key_axis", Caps::empty(), FitsInputs::NONE, split_global, fits_by_key_axis),
];

fn attrs(subject: &Subject<'_>) -> SortAttrs {
    match boundary(subject).attrs {
        BoundaryAttrs::Sort(attrs) => attrs,
        _ => unreachable!("a sort boundary carries SortAttrs"),
    }
}

fn passes(attrs: &SortAttrs) -> u32 {
    u32::from(attrs.key_bits).clamp(1, 32).div_ceil(DIGIT_BITS)
}

/// The entry of a sort: `can_run_on_host() && len <= MAX_SORT_ELEMS && error().is_none()` (then `try_*` may decline).
fn fits_host(subject: &Subject<'_>, ctx: &Context, _: &Residencies) -> Option<Params> {
    let a = attrs(subject);
    (a.stage == SortStage::Entry && !a.errored && ctx.can_run_on_host() && a.len <= crate::host::MAX_SORT_ELEMS as u64).then_some(Params::Sort(SortParams::NONE))
}

/// `use_sweep`: `lookback_scan && use_ballots && sweep_bytes(16) <= workgroup_storage && n < 1 << 30`.
fn fits_radix_sweep(subject: &Subject<'_>, ctx: &Context, _: &Residencies) -> Option<Params> {
    let a = attrs(subject);
    if a.stage != SortStage::Radix || !use_sweep(ctx, a.len.min(u64::from(u32::MAX)) as u32) {
        return None;
    }
    let per_thread = sweep_per_thread(ctx.tuning(), a.payload != SortPayload::None);
    Some(Params::Sort(SortParams { per_thread, tile: WORKGROUP * per_thread, passes: passes(&a) }))
}

/// `use_ballots`: `subgroup32 && ballot_bytes() <= workgroup_storage`; tiles of 4096 keys.
fn fits_radix_ballots(subject: &Subject<'_>, ctx: &Context, _: &Residencies) -> Option<Params> {
    let a = attrs(subject);
    (a.stage == SortStage::Radix && use_ballots(ctx.tuning())).then(|| Params::Sort(SortParams { per_thread: 0, tile: WORKGROUP * BALLOT_PER_THREAD, passes: passes(&a) }))
}

/// The last of the radix paths: always.
fn fits_radix_count_scan(subject: &Subject<'_>, _: &Context, _: &Residencies) -> Option<Params> {
    let a = attrs(subject);
    (a.stage == SortStage::Radix).then(|| Params::Sort(SortParams { per_thread: 0, tile: TILE, passes: passes(&a) }))
}

/// An axis sort of lines of at most `segsort::max_line` cells (0: none).
fn fits_lines(subject: &Subject<'_>, ctx: &Context, _: &Residencies) -> Option<Params> {
    let b = boundary(subject);
    let a = attrs(subject);
    let max = crate::segsort::max_line(ctx);
    (a.stage == SortStage::Axis && max > 0 && b.dims[0] <= max).then_some(Params::Sort(SortParams::NONE))
}

fn fits_lines_radix(subject: &Subject<'_>, ctx: &Context, _: &Residencies) -> Option<Params> {
    let b = boundary(subject);
    (attrs(subject).stage == SortStage::Axis && b.dims[0] > crate::segsort::max_line(ctx)).then_some(Params::Sort(SortParams::NONE))
}

/// By key: 4-byte values that the host path does not take (`can_run_on_host() && len <= MAX_SORT_ELEMS`) travel through the radix
/// passes with their keys.
fn fits_by_key_payload(subject: &Subject<'_>, ctx: &Context, _: &Residencies) -> Option<Params> {
    let a = attrs(subject);
    (a.stage == SortStage::ByKey && a.value4 && !(ctx.can_run_on_host() && a.len <= crate::host::MAX_SORT_ELEMS as u64)).then_some(Params::Sort(SortParams::NONE))
}

/// By key otherwise: the permutation of an argsort (or sort with indices) and a gather.
fn fits_by_key_permutation(subject: &Subject<'_>, _: &Context, _: &Residencies) -> Option<Params> {
    (attrs(subject).stage == SortStage::ByKey).then_some(Params::Sort(SortParams::NONE))
}

/// `sort_by_key_axis` of several lines (one line is the flat by-key sort).
fn fits_by_key_axis(subject: &Subject<'_>, _: &Context, _: &Residencies) -> Option<Params> {
    let b = boundary(subject);
    (attrs(subject).stage == SortStage::ByKeyAxis && b.dims[2] != 1).then_some(Params::Sort(SortParams::NONE))
}
