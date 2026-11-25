//! The sparse products of the `Opaque` family: the only Opaque entries with real branches. The row classes are read from the
//! matrix's memoised row plan, so selection reads them and never recomputes them; the first non-native product of a matrix scatters
//! (the matrix keeps a counter, so `fits` is told `first_non_native`), the next converts to the native format (not a strategy: the
//! recursion selects again). The other ops of the family are declarations (`structured::opaque`).

use super::{boundary, entry, id, split_global};
use crate::caps::Caps;
use crate::context::Context;
use crate::jit::structured::boundary::{OpaqueAttrs, OpaqueParams};
use crate::jit::structured::strategy::{BoundaryAttrs, FamilyTag, Params, Residencies, Strategy, StrategyId, Subject};

impl StrategyId {
    pub(crate) const OPAQUE_SPARSE_EMPTY: StrategyId = id(FamilyTag::Opaque, 0);
    pub(crate) const OPAQUE_SPARSE_SCATTER: StrategyId = id(FamilyTag::Opaque, 1);
    pub(crate) const OPAQUE_SPARSE_SPMV_MERGE: StrategyId = id(FamilyTag::Opaque, 2);
    pub(crate) const OPAQUE_SPARSE_ROWS: StrategyId = id(FamilyTag::Opaque, 3);
}

/// Table order is today's: the guard, the scatter of a first non-native product, then the native kernels.
pub(crate) static OPAQUE_STRATEGIES: [Strategy; 4] = [
    entry(StrategyId::OPAQUE_SPARSE_EMPTY, "sparse_empty", Caps::empty(), split_global, fits_empty),
    entry(StrategyId::OPAQUE_SPARSE_SCATTER, "sparse_scatter", Caps::empty(), split_global, fits_scatter),
    entry(StrategyId::OPAQUE_SPARSE_SPMV_MERGE, "sparse_spmv_merge", Caps::empty(), split_global, fits_merge),
    entry(StrategyId::OPAQUE_SPARSE_ROWS, "sparse_rows", Caps::empty(), split_global, fits_rows),
];

/// The guard, the scatter, and the native kernels: runs of the table, so a product scans only the entries of its own case.
pub(crate) const EMPTY_STRATEGIES: &[Strategy] = OPAQUE_STRATEGIES.split_at(1).0;
pub(crate) const SCATTER_STRATEGIES: &[Strategy] = OPAQUE_STRATEGIES.split_at(1).1.split_at(1).0;
pub(crate) const NATIVE_STRATEGIES: &[Strategy] = OPAQUE_STRATEGIES.split_at(2).1;

fn attrs(subject: &Subject<'_>) -> OpaqueAttrs {
    match boundary(subject).attrs {
        BoundaryAttrs::Opaque(attrs) => attrs,
        _ => unreachable!("an opaque boundary carries OpaqueAttrs"),
    }
}

/// `nnz == 0`: zeros.
fn fits_empty(subject: &Subject<'_>, _: &Context, _: &Residencies) -> Option<Params> {
    (attrs(subject).nnz == 0).then_some(Params::Opaque(OpaqueParams::NONE))
}

/// Not the native format, at the first such product: products scattered with atomics.
fn fits_scatter(subject: &Subject<'_>, _: &Context, _: &Residencies) -> Option<Params> {
    let a = attrs(subject);
    (a.nnz != 0 && !a.native && a.first_non_native).then_some(Params::Opaque(OpaqueParams::NONE))
}

/// Native, a vector, and rows longer than `SHORT_ROWS`: the tile kernel over `ceil(nnz / MERGE_TILE)` tiles.
fn fits_merge(subject: &Subject<'_>, _: &Context, _: &Residencies) -> Option<Params> {
    let a = attrs(subject);
    (a.nnz != 0 && a.native && a.columns == 1 && a.tiles).then(|| Params::Opaque(OpaqueParams { lanes_log: 0, tiles: crate::sparse::merge_tiles(a.nnz as usize) }))
}

/// Native, otherwise: the row kernels (`SPMV_ROWS` with `lanes_log` for a vector, `SPMM_ROWS` for columns), plus the long rows' pass.
fn fits_rows(subject: &Subject<'_>, _: &Context, _: &Residencies) -> Option<Params> {
    let a = attrs(subject);
    (a.nnz != 0 && a.native).then(|| Params::Opaque(OpaqueParams { lanes_log: if a.columns == 1 { crate::sparse::lanes_log(a.nnz as usize, a.outer as usize) } else { 0 }, tiles: 0 }))
}
