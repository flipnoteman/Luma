//! The scan family: `Lines::scan` (every `cumsum`, `scan`, and the scans of the counts of a sort) and the by-key scan. The `fits`
//! functions are the inline conditions of `Lines::scan` and `run_scan_by_key`, in their order (table order is semantics).

use super::{boundary, entry, id, split_scan, with_prologue};
use crate::bykey::CHUNK as BY_KEY_CHUNK;
use crate::caps::Caps;
use crate::context::Context;
use crate::jit::structured::boundary::{ScanAttrs, ScanOperand, ScanParams};
use crate::jit::structured::strategy::{BoundaryAttrs, FamilyTag, Params, Residencies, Strategy, StrategyId, Subject};
use crate::scan::{strided_block, target_threads, LOOKBACK_CHUNK, SCAN_CHUNK, SHORT_LINE, WARP_MIN_LINES};

impl StrategyId {
    pub(crate) const SCAN_ROWS: StrategyId = id(FamilyTag::Scan, 0);
    pub(crate) const SCAN_WARP: StrategyId = id(FamilyTag::Scan, 1);
    pub(crate) const SCAN_LOOKBACK: StrategyId = id(FamilyTag::Scan, 2);
    pub(crate) const SCAN_CHUNK_SINGLE: StrategyId = id(FamilyTag::Scan, 3);
    pub(crate) const SCAN_CHUNK_THREE_PASS: StrategyId = id(FamilyTag::Scan, 4);
    pub(crate) const SCAN_SEQ: StrategyId = id(FamilyTag::Scan, 5);
    pub(crate) const SCAN_FOLD_WALK: StrategyId = id(FamilyTag::Scan, 6);
    pub(crate) const SCAN_BY_KEY_SINGLE: StrategyId = id(FamilyTag::Scan, 7);
    pub(crate) const SCAN_BY_KEY_CHUNKED: StrategyId = id(FamilyTag::Scan, 8);
}

pub(crate) static SCAN_STRATEGIES: [Strategy; 9] = [
    with_prologue(entry(StrategyId::SCAN_ROWS, "rows", Caps::empty(), split_scan, fits_rows)),
    with_prologue(entry(StrategyId::SCAN_WARP, "warp", Caps::SUBGROUP32, split_scan, fits_warp)),
    with_prologue(entry(StrategyId::SCAN_LOOKBACK, "lookback", Caps::LOOKBACK, split_scan, fits_lookback)),
    with_prologue(entry(StrategyId::SCAN_CHUNK_SINGLE, "chunk_single", Caps::empty(), split_scan, fits_chunk_single)),
    with_prologue(entry(StrategyId::SCAN_CHUNK_THREE_PASS, "chunk_three_pass", Caps::empty(), split_scan, fits_chunk_three_pass)),
    with_prologue(entry(StrategyId::SCAN_SEQ, "seq", Caps::empty(), split_scan, fits_seq)),
    with_prologue(entry(StrategyId::SCAN_FOLD_WALK, "fold_walk", Caps::empty(), split_scan, fits_fold_walk)),
    entry(StrategyId::SCAN_BY_KEY_SINGLE, "by_key_single", Caps::empty(), split_scan, fits_by_key_single),
    entry(StrategyId::SCAN_BY_KEY_CHUNKED, "by_key_chunked", Caps::empty(), split_scan, fits_by_key_chunked),
];

/// `(n, stride, lines)` and the attributes of a scan boundary.
fn shape(subject: &Subject<'_>) -> (u32, u32, u32, ScanAttrs) {
    let boundary = boundary(subject);
    match boundary.attrs {
        BoundaryAttrs::Scan(attrs) => (boundary.dims[0], boundary.dims[1], boundary.dims[2], attrs),
        _ => unreachable!("a scan boundary carries ScanAttrs"),
    }
}

/// A pending operand is computed in the kernel of a single-pass strategy whatever it costs (the alternative is a kernel that writes
/// it and the scan reads it back).
fn single(a: &ScanAttrs) -> bool {
    a.operand != ScanOperand::Buffer
}

/// ... and in the kernels of a strategy that reads it twice (fold, then scan) only where recomputing it is cheaper than storing it.
fn twice(a: &ScanAttrs) -> bool {
    a.operand == ScanOperand::Cheap
}

/// `stride == 1 && n <= 255`.
fn fits_rows(subject: &Subject<'_>, _: &Context, _: &Residencies) -> Option<Params> {
    let (n, stride, _, a) = shape(subject);
    (!a.by_key && stride == 1 && n <= SHORT_LINE).then_some(Params::Scan(ScanParams { fused: single(&a), ..ScanParams::NONE }))
}

/// `stride == 1 && lines >= scaled(128) && subgroup32`; `vec` is 4, 2 or 1 by what divides `n`.
fn fits_warp(subject: &Subject<'_>, ctx: &Context, _: &Residencies) -> Option<Params> {
    let (n, stride, lines, a) = shape(subject);
    if a.by_key || stride != 1 || lines < ctx.tuning().scaled(WARP_MIN_LINES) || !ctx.tuning().subgroup32 {
        return None;
    }
    Some(Params::Scan(ScanParams { vec: if n.is_multiple_of(4) { 4 } else if n.is_multiple_of(2) { 2 } else { 1 }, nb: 1, k: 0, fused: single(&a) }))
}

/// `stride == 1 && lookback_scan && n > 4096 && grouping_free`.
fn fits_lookback(subject: &Subject<'_>, ctx: &Context, _: &Residencies) -> Option<Params> {
    let (n, stride, _, a) = shape(subject);
    (!a.by_key && stride == 1 && ctx.tuning().lookback_scan && n > LOOKBACK_CHUNK && a.grouping_free).then(|| Params::Scan(ScanParams { vec: 0, nb: n.div_ceil(LOOKBACK_CHUNK), k: 0, fused: single(&a) }))
}

/// `stride == 1`, one chunk of 1024 cells.
fn fits_chunk_single(subject: &Subject<'_>, _: &Context, _: &Residencies) -> Option<Params> {
    let (n, stride, _, a) = shape(subject);
    (!a.by_key && stride == 1 && n.div_ceil(SCAN_CHUNK) == 1).then(|| Params::Scan(ScanParams { vec: 0, nb: 1, k: SCAN_CHUNK, fused: single(&a) }))
}

/// `stride == 1`, several chunks: fold, the scan of the totals (a nested selection), scan seeded by the prefix.
fn fits_chunk_three_pass(subject: &Subject<'_>, _: &Context, _: &Residencies) -> Option<Params> {
    let (n, stride, _, a) = shape(subject);
    let nb = n.div_ceil(SCAN_CHUNK);
    (!a.by_key && stride == 1 && nb > 1).then(|| Params::Scan(ScanParams { vec: 0, nb, k: SCAN_CHUNK, fused: twice(&a) }))
}

/// `stride > 1` and no blocks (`lines >= 65536` or `n <= 256`): one invocation walks a line.
fn fits_seq(subject: &Subject<'_>, ctx: &Context, _: &Residencies) -> Option<Params> {
    let (n, stride, lines, a) = shape(subject);
    (!a.by_key && stride > 1 && strided_block(n, lines, target_threads(ctx.tuning())).is_none()).then_some(Params::Scan(ScanParams { fused: single(&a), ..ScanParams::NONE }))
}

/// `stride > 1` otherwise: fold of blocks of `k` cells, the scan of the totals (a nested selection), walk.
fn fits_fold_walk(subject: &Subject<'_>, ctx: &Context, _: &Residencies) -> Option<Params> {
    let (n, stride, lines, a) = shape(subject);
    if a.by_key || stride <= 1 {
        return None;
    }
    let k = strided_block(n, lines, target_threads(ctx.tuning()))?;
    Some(Params::Scan(ScanParams { vec: 0, nb: n.div_ceil(k), k, fused: twice(&a) }))
}

/// By key: `nb = ceil(n / 4096) == 1`.
fn fits_by_key_single(subject: &Subject<'_>, _: &Context, _: &Residencies) -> Option<Params> {
    let (n, _, _, a) = shape(subject);
    (a.by_key && n.div_ceil(BY_KEY_CHUNK) == 1).then_some(Params::Scan(ScanParams { vec: 0, nb: 1, k: BY_KEY_CHUNK, fused: false }))
}

/// By key: `nb > 1`, which folds and carries before the scan kernel.
fn fits_by_key_chunked(subject: &Subject<'_>, _: &Context, _: &Residencies) -> Option<Params> {
    let (n, _, _, a) = shape(subject);
    let nb = n.div_ceil(BY_KEY_CHUNK);
    (a.by_key && nb > 1).then(|| Params::Scan(ScanParams { vec: 0, nb, k: BY_KEY_CHUNK, fused: false }))
}
