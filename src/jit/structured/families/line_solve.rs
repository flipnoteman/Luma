//! The line-solve family: `solve_tridiagonal`, `solve_block_tridiagonal` and their parts forms (`linalg::run`, `run_parts`). Table order
//! is the order of today's checks: cyclic reduction (`cyclic::plan`), the native Thomas solve, the Thomas kernel, and for parts the
//! composition through the stacked solver. The plans stay the functions they were (`cyclic::plan`, `thomas_plan`); `fits` calls them.

use super::{boundary, entry, id, split_map};
use crate::builtin::{thomas_plan, UNROLLED_BLOCK};
use crate::caps::Caps;
use crate::context::Context;
use crate::jit::structured::boundary::{LineSolveAttrs, LineSolveParams};
use crate::jit::structured::strategy::{BoundaryAttrs, FamilyTag, Params, Residencies, Strategy, StrategyId, Subject};
use crate::linalg::{thomas_parts_scratch, tuning_of};
use crate::tuning::{LineSolver, Tuning};

impl StrategyId {
    pub(crate) const LINE_SOLVE_CYCLIC: StrategyId = id(FamilyTag::LineSolve, 0);
    pub(crate) const LINE_SOLVE_NATIVE: StrategyId = id(FamilyTag::LineSolve, 1);
    pub(crate) const LINE_SOLVE_THOMAS: StrategyId = id(FamilyTag::LineSolve, 2);
    pub(crate) const LINE_SOLVE_STACKED_FROM_PARTS: StrategyId = id(FamilyTag::LineSolve, 3);
}

pub(crate) static LINE_SOLVE_STRATEGIES: [Strategy; 4] = [
    entry(StrategyId::LINE_SOLVE_CYCLIC, "cyclic", Caps::empty(), split_map, fits_cyclic),
    entry(StrategyId::LINE_SOLVE_NATIVE, "native", Caps::NATIVE_LINE_SOLVE, split_map, fits_native),
    entry(StrategyId::LINE_SOLVE_THOMAS, "thomas", Caps::empty(), split_map, fits_thomas),
    entry(StrategyId::LINE_SOLVE_STACKED_FROM_PARTS, "stacked_from_parts", Caps::empty(), split_map, fits_stacked_from_parts),
];

/// `(n, stride, lines, systems)` and the attributes of a line-solve boundary.
fn shape(subject: &Subject<'_>) -> ([u32; 4], LineSolveAttrs) {
    let boundary = boundary(subject);
    match boundary.attrs {
        BoundaryAttrs::LineSolve(attrs) => (boundary.dims, attrs),
        _ => unreachable!("a line-solve boundary carries LineSolveAttrs"),
    }
}

/// The Thomas plan and its scratch: a parts solve keeps the modified upper blocks of every system (`None` where they exceed a buffer),
/// a stacked one those of its cells.
fn thomas(a: &LineSolveAttrs, tuning: &Tuning, dims: [u32; 4]) -> Option<Params> {
    let plan = thomas_plan(tuning, u32::from(a.block), dims[2], dims[1], a.parts);
    let scratch_bytes = if a.parts {
        thomas_parts_scratch(a.len as usize, usize::from(a.block), dims[3] as usize, plan.tiled, a.max_scratch)?
    } else {
        a.len * u64::from(a.block) * u64::from(a.block) * 4
    };
    Some(Params::LineSolve(LineSolveParams {
        block: a.block,
        parts: a.parts,
        workgroup: u16::from(plan.workgroup),
        shift: 0,
        shared_rows: 0,
        tiled: plan.tiled,
        twisted: plan.twisted,
        scratch_bytes,
    }))
}

/// `cyclic::plan(..)` is `Some`; `LUMA_STRATEGY=line_solve:cyclic` forces it like `LUMA_TRIDIAGONAL=cyclic`.
fn fits_cyclic(subject: &Subject<'_>, ctx: &Context, _: &Residencies) -> Option<Params> {
    let (dims, a) = shape(subject);
    let tuning = tuning_of(ctx);
    let forced = ctx.strategy_override(FamilyTag::LineSolve) == Some(StrategyId::LINE_SOLVE_CYCLIC);
    let block = u32::from(a.block);
    let plan = if forced {
        crate::cyclic::plan(&Tuning { line_solver: LineSolver::Cyclic, ..*tuning }, block, dims[0], dims[2], dims[1], a.parts, a.fused, a.max_scratch)
    } else {
        crate::cyclic::plan(&tuning, block, dims[0], dims[2], dims[1], a.parts, a.fused, a.max_scratch)
    }?;
    Some(Params::LineSolve(LineSolveParams {
        block: a.block,
        parts: a.parts,
        workgroup: plan.workgroup,
        shift: plan.shift,
        shared_rows: plan.shared_rows,
        tiled: false,
        twisted: false,
        scratch_bytes: plan.scratch_bytes(block, dims[0], dims[2]),
    }))
}

/// The Thomas plan is the native one (`tuning.native_solve`), and, for parts, its scratch fits a buffer.
fn fits_native(subject: &Subject<'_>, ctx: &Context, _: &Residencies) -> Option<Params> {
    let (dims, a) = shape(subject);
    let tuning = tuning_of(ctx);
    if !tuning.native_solve {
        return None;
    }
    thomas(&a, &tuning, dims)
}

/// Not native; a parts solve needs `block <= UNROLLED_BLOCK` and a scratch that fits a buffer.
fn fits_thomas(subject: &Subject<'_>, ctx: &Context, _: &Residencies) -> Option<Params> {
    let (dims, a) = shape(subject);
    let tuning = tuning_of(ctx);
    if tuning.native_solve || (a.parts && u32::from(a.block) > UNROLLED_BLOCK) {
        return None;
    }
    thomas(&a, &tuning, dims)
}

/// Parts that neither of the above takes: the parts are stacked and the solve selects again on the stacked buffers.
fn fits_stacked_from_parts(subject: &Subject<'_>, _: &Context, _: &Residencies) -> Option<Params> {
    let (_, a) = shape(subject);
    a.parts.then_some(Params::LineSolve(LineSolveParams { block: a.block, parts: true, workgroup: 0, shift: 0, shared_rows: 0, tiled: false, twisted: false, scratch_bytes: 0 }))
}
