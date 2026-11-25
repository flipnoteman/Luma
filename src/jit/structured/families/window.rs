//! The window family: the windowed image kernels (`median`, `min`/`max`, `correlate`, Sobel, bilateral). `window::tiled` is the
//! plan of `image::plan`; the median selections are the two code paths of the kernel source (`MEDIAN_REGISTERS`); the separable
//! extremum is the entry of `minfilt`/`maxfilt` (`extremum`).

use super::{boundary, entry, id, split_halo};
use crate::caps::Caps;
use crate::context::Context;
use crate::image::first_fit;
use crate::image::window::{Op, MEDIAN_REGISTERS};
use crate::jit::structured::boundary::{WindowAttrs, WindowParams, WindowStage};
use crate::jit::structured::strategy::{BoundaryAttrs, FamilyTag, Params, Residencies, Strategy, StrategyId, Subject};

impl StrategyId {
    pub(crate) const WINDOW_SEPARABLE_EXTREMUM: StrategyId = id(FamilyTag::Window, 0);
    pub(crate) const WINDOW_MEDIAN_REGISTERS: StrategyId = id(FamilyTag::Window, 1);
    pub(crate) const WINDOW_MEDIAN_BITWISE: StrategyId = id(FamilyTag::Window, 2);
    pub(crate) const WINDOW_TILED: StrategyId = id(FamilyTag::Window, 3);
}

/// Table order: the separable extremum at the entry of `extremum` (its `if` is before `rank_filter`); at the plan, the median
/// selections then the tiled window every other op takes (and a forced `window:tiled` takes for a median: the same kernel).
pub(crate) static WINDOW_STRATEGIES: [Strategy; 4] = [
    entry(StrategyId::WINDOW_SEPARABLE_EXTREMUM, "separable_extremum", Caps::empty(), split_halo, fits_separable_extremum),
    entry(StrategyId::WINDOW_MEDIAN_REGISTERS, "median_registers", Caps::empty(), split_halo, fits_median_registers),
    entry(StrategyId::WINDOW_MEDIAN_BITWISE, "median_bitwise", Caps::empty(), split_halo, fits_median_bitwise),
    entry(StrategyId::WINDOW_TILED, "tiled", Caps::empty(), split_halo, fits_tiled),
];

/// The entries that apply at `WindowStage::Entry` and `Plan`: runs of the table, so an op scans only its own.
pub(crate) const ENTRY_STRATEGIES: &[Strategy] = WINDOW_STRATEGIES.split_at(1).0;
pub(crate) const PLAN_STRATEGIES: &[Strategy] = WINDOW_STRATEGIES.split_at(1).1;

#[inline(always)]
fn attrs<'a>(subject: &'a Subject<'_>) -> &'a WindowAttrs {
    match &boundary(subject).attrs {
        BoundaryAttrs::Window(attrs) => attrs,
        _ => unreachable!("a window boundary carries WindowAttrs"),
    }
}

fn cells(a: &WindowAttrs) -> usize {
    usize::from(a.k[0]) * usize::from(a.k[1]) * usize::from(a.k[2])
}

/// The tile candidates' first fit (`None`: the window does not fit workgroup memory).
fn plan(a: &WindowAttrs, ctx: &Context) -> Option<Params> {
    let limit = u64::from(ctx.tuning().workgroup_storage);
    first_fit(limit, a.op, a.k, a.stride, a.three_d, a.aux_cells, a.pixel, a.edge).map(|window| Params::Window(WindowParams { window: Some(window) }))
}

/// `k0 * k1 >= 25 && k0 > 1 && k1 > 1` of a minimum or maximum.
fn fits_separable_extremum(subject: &Subject<'_>, _: &Context, _: &Residencies) -> Option<Params> {
    let a = attrs(subject);
    let (k0, k1) = (usize::from(a.k[0]), usize::from(a.k[1]));
    (a.stage == WindowStage::Entry && matches!(a.op, Op::Min | Op::Max) && k0 * k1 >= 25 && k0 > 1 && k1 > 1).then_some(Params::Window(WindowParams { window: None }))
}

/// A median of at most `MEDIAN_REGISTERS` cells: the unrolled insertion selection.
fn fits_median_registers(subject: &Subject<'_>, ctx: &Context, _: &Residencies) -> Option<Params> {
    let a = attrs(subject);
    if a.stage != WindowStage::Plan || a.op != Op::Median || cells(a) > MEDIAN_REGISTERS {
        return None;
    }
    plan(a, ctx)
}

/// A median of more cells: bit by bit from workgroup memory.
fn fits_median_bitwise(subject: &Subject<'_>, ctx: &Context, _: &Residencies) -> Option<Params> {
    let a = attrs(subject);
    if a.stage != WindowStage::Plan || a.op != Op::Median || cells(a) <= MEDIAN_REGISTERS {
        return None;
    }
    plan(a, ctx)
}

/// Every window kernel, unless no candidate fits workgroup memory (`Unsupported`, as before).
fn fits_tiled(subject: &Subject<'_>, ctx: &Context, _: &Residencies) -> Option<Params> {
    let a = attrs(subject);
    if a.stage != WindowStage::Plan {
        return None;
    }
    plan(a, ctx)
}
