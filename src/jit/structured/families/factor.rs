//! The dense factorisation family (`dense/`): the leaves and the recursion of LU, Cholesky, QR, the triangular solves, SVD and eigh.
//! One table serves every stage: each strategy says in its `fits` which factorisation (`FactorKind`) it is a way of running, and an
//! op selects over the slice of its own stage ([`lu_stage`] and the others), so a call scans a handful of entries. The stages nest as
//! the code does (a `solve` selects an LU, whose recursion selects a leaf for every panel and a triangular solve for every update).
//! Every `fits` is today's condition, in today's order, over `(rows, cols, batch)`.

use std::panic::Location;

use super::{boundary, entry, id, split_global};
use crate::caps::Caps;
use crate::context::Context;
use crate::dense::svd::{self, Kind};
use crate::dense::{chol, lu, tri, trsv, tsqr, Dims};
use crate::jit::structured::boundary::{dim, FactorAttrs, FactorKind, FactorParams};
use crate::jit::structured::strategy::{Boundary, BoundaryAttrs, FamilyTag, Params, Residencies, Strategy, StrategyId, Subject};

impl StrategyId {
    pub(crate) const LU_CALU_SINGLE: StrategyId = id(FamilyTag::Factor, 0);
    pub(crate) const LU_CALU_TOURNAMENT: StrategyId = id(FamilyTag::Factor, 1);
    pub(crate) const LU_SMALL: StrategyId = id(FamilyTag::Factor, 2);
    pub(crate) const LU_PANEL: StrategyId = id(FamilyTag::Factor, 3);
    pub(crate) const LU_RECURSIVE: StrategyId = id(FamilyTag::Factor, 4);
    pub(crate) const CHOL_SMALL: StrategyId = id(FamilyTag::Factor, 5);
    pub(crate) const CHOL_BLOCKED: StrategyId = id(FamilyTag::Factor, 6);
    pub(crate) const QR_TSQR_PANEL: StrategyId = id(FamilyTag::Factor, 7);
    pub(crate) const QR_LEAF: StrategyId = id(FamilyTag::Factor, 8);
    pub(crate) const QR_RECURSIVE: StrategyId = id(FamilyTag::Factor, 9);
    pub(crate) const TRI_SUBSTITUTE: StrategyId = id(FamilyTag::Factor, 10);
    pub(crate) const TRI_TRSV: StrategyId = id(FamilyTag::Factor, 11);
    pub(crate) const TRI_BLOCKED: StrategyId = id(FamilyTag::Factor, 12);
    pub(crate) const SVD_SMALL: StrategyId = id(FamilyTag::Factor, 13);
    pub(crate) const SVD_JACOBI: StrategyId = id(FamilyTag::Factor, 14);
    pub(crate) const SVD_BLOCK_JACOBI: StrategyId = id(FamilyTag::Factor, 15);
    pub(crate) const SVD_PRECONDITIONED: StrategyId = id(FamilyTag::Factor, 16);
    pub(crate) const EIGH_SMALL: StrategyId = id(FamilyTag::Factor, 17);
    pub(crate) const EIGH_JACOBI: StrategyId = id(FamilyTag::Factor, 18);
    pub(crate) const EIGH_BLOCK_JACOBI: StrategyId = id(FamilyTag::Factor, 19);
    pub(crate) const LU_REFINED_F32: StrategyId = id(FamilyTag::Factor, 20);
    pub(crate) const CHOL_REFINED_F32: StrategyId = id(FamilyTag::Factor, 21);
}

pub(crate) static FACTOR_STRATEGIES: [Strategy; 22] = [
    entry(StrategyId::LU_CALU_SINGLE, "lu_calu_single", Caps::empty(), split_global, fits_lu_calu_single),
    entry(StrategyId::LU_CALU_TOURNAMENT, "lu_calu_tournament", Caps::empty(), split_global, fits_lu_calu_tournament),
    entry(StrategyId::LU_SMALL, "lu_small", Caps::empty(), split_global, fits_lu_small),
    entry(StrategyId::LU_PANEL, "lu_panel", Caps::empty(), split_global, fits_lu_panel),
    entry(StrategyId::LU_RECURSIVE, "lu_recursive", Caps::empty(), split_global, fits_lu_recursive),
    entry(StrategyId::CHOL_SMALL, "chol_small", Caps::empty(), split_global, fits_chol_small),
    entry(StrategyId::CHOL_BLOCKED, "chol_blocked", Caps::empty(), split_global, fits_chol_blocked),
    entry(StrategyId::QR_TSQR_PANEL, "qr_tsqr_panel", Caps::empty(), split_global, fits_qr_tsqr_panel),
    entry(StrategyId::QR_LEAF, "qr_leaf", Caps::empty(), split_global, fits_qr_leaf),
    entry(StrategyId::QR_RECURSIVE, "qr_recursive", Caps::empty(), split_global, fits_qr_recursive),
    entry(StrategyId::TRI_SUBSTITUTE, "tri_substitute", Caps::empty(), split_global, fits_tri_substitute),
    entry(StrategyId::TRI_TRSV, "tri_trsv", Caps::LOOKBACK, split_global, fits_tri_trsv),
    entry(StrategyId::TRI_BLOCKED, "tri_blocked", Caps::empty(), split_global, fits_tri_blocked),
    entry(StrategyId::SVD_SMALL, "svd_small", Caps::empty(), split_global, fits_svd_small),
    entry(StrategyId::SVD_JACOBI, "svd_jacobi", Caps::empty(), split_global, fits_svd_jacobi),
    entry(StrategyId::SVD_BLOCK_JACOBI, "svd_block_jacobi", Caps::empty(), split_global, fits_svd_block_jacobi),
    entry(StrategyId::SVD_PRECONDITIONED, "svd_preconditioned", Caps::empty(), split_global, fits_svd_preconditioned),
    entry(StrategyId::EIGH_SMALL, "eigh_small", Caps::empty(), split_global, fits_eigh_small),
    entry(StrategyId::EIGH_JACOBI, "eigh_jacobi", Caps::empty(), split_global, fits_eigh_jacobi),
    entry(StrategyId::EIGH_BLOCK_JACOBI, "eigh_block_jacobi", Caps::empty(), split_global, fits_eigh_block_jacobi),
    entry(StrategyId::LU_REFINED_F32, "lu_refined_f32", Caps::empty(), split_global, fits_lu_refined_f32),
    entry(StrategyId::CHOL_REFINED_F32, "chol_refined_f32", Caps::empty(), split_global, fits_chol_refined_f32),
];

/// The entries an LU selects over: the leaves in the order `leaf` tried them (tournament, workgroup memory, panel), then the recursion.
pub(crate) fn lu_stage() -> &'static [Strategy] {
    &FACTOR_STRATEGIES[0..5]
}

pub(crate) fn chol_stage() -> &'static [Strategy] {
    &FACTOR_STRATEGIES[5..7]
}

pub(crate) fn qr_stage() -> &'static [Strategy] {
    &FACTOR_STRATEGIES[7..10]
}

pub(crate) fn tri_stage() -> &'static [Strategy] {
    &FACTOR_STRATEGIES[10..13]
}

pub(crate) fn svd_stage() -> &'static [Strategy] {
    &FACTOR_STRATEGIES[13..17]
}

pub(crate) fn eigh_stage() -> &'static [Strategy] {
    &FACTOR_STRATEGIES[17..20]
}

/// The refined solves: only a call to `solve_refined` or `solve_spd_refined` asks for them (their own kinds), so no other candidate list
/// ever holds one.
pub(crate) fn refined_stage() -> &'static [Strategy] {
    &FACTOR_STRATEGIES[20..22]
}

/// The descriptor of a factorisation of the `d` matrices (`rhs` right-hand-side columns where there are any).
#[inline]
pub(crate) fn boundary_of(kind: FactorKind, op: &'static str, location: &'static Location<'static>, d: Dims, rhs: usize) -> Boundary<'static> {
    boundary_flags(kind, op, location, d, rhs, false, false, false)
}

/// [`boundary_of`] with the flags of the triangular solves (`inverse`) and of the SVD and eigh (`exact`, `vectors`).
#[inline]
#[allow(clippy::too_many_arguments)]
pub(crate) fn boundary_flags(kind: FactorKind, op: &'static str, location: &'static Location<'static>, d: Dims, rhs: usize, inverse: bool, exact: bool, vectors: bool) -> Boundary<'static> {
    Boundary::new(
        FamilyTag::Factor,
        op,
        location,
        [dim(d.rows), dim(d.cols), dim(d.batch), dim(rhs)],
        BoundaryAttrs::Factor(FactorAttrs { kind, inverse, exact, vectors }),
    )
}

/// `(rows, cols, batch, rhs)` and the attributes of a factorisation boundary.
fn shape(subject: &Subject<'_>) -> (usize, usize, usize, usize, FactorAttrs) {
    let boundary = boundary(subject);
    match boundary.attrs {
        BoundaryAttrs::Factor(attrs) => (boundary.dims[0] as usize, boundary.dims[1] as usize, boundary.dims[2] as usize, boundary.dims[3] as usize, attrs),
        _ => unreachable!("a factor boundary carries FactorAttrs"),
    }
}

fn some(params: FactorParams) -> Option<Params> {
    Some(Params::Factor(params))
}

// -- LU --

/// The tournament panel of `leaf`: one matrix of more than 64 rows whose columns fit a leaf and a tournament, in one chunk.
fn calu(subject: &Subject<'_>, ctx: &Context) -> Option<FactorParams> {
    let (rows, cols, batch, _, a) = shape(subject);
    if a.kind != FactorKind::Lu || batch != 1 || rows <= lu::SMALL || cols > lu::leaf_width(rows, batch) {
        return None;
    }
    lu::calu_plan(ctx, rows, cols)
}

fn fits_lu_calu_single(subject: &Subject<'_>, ctx: &Context, _: &Residencies) -> Option<Params> {
    calu(subject, ctx).filter(|plan| plan.chunks == 1).and_then(some)
}

fn fits_lu_calu_tournament(subject: &Subject<'_>, ctx: &Context, _: &Residencies) -> Option<Params> {
    calu(subject, ctx).filter(|plan| plan.chunks > 1).and_then(some)
}

/// `rows <= 64` and the columns fit a leaf.
fn fits_lu_small(subject: &Subject<'_>, _: &Context, _: &Residencies) -> Option<Params> {
    let (rows, cols, batch, _, a) = shape(subject);
    (a.kind == FactorKind::Lu && rows <= lu::SMALL && cols <= lu::leaf_width(rows, batch)).then_some(Params::Factor(FactorParams::NONE))
}

/// `rows > 64`, the columns fit a leaf (the tournament is tried first, by table order).
fn fits_lu_panel(subject: &Subject<'_>, ctx: &Context, _: &Residencies) -> Option<Params> {
    let (rows, cols, batch, _, a) = shape(subject);
    (a.kind == FactorKind::Lu && rows > lu::SMALL && cols <= lu::leaf_width(rows, batch)).then(|| Params::Factor(FactorParams { threads: lu::panel_threads(ctx, rows) as u32, ..FactorParams::NONE }))
}

/// More columns than a leaf takes.
fn fits_lu_recursive(subject: &Subject<'_>, _: &Context, _: &Residencies) -> Option<Params> {
    let (rows, cols, batch, _, a) = shape(subject);
    (a.kind == FactorKind::Lu && cols > lu::leaf_width(rows, batch)).then_some(Params::Factor(FactorParams::NONE))
}

// -- Cholesky --

/// `rows <= chol::leaf(ctx)`.
fn fits_chol_small(subject: &Subject<'_>, ctx: &Context, _: &Residencies) -> Option<Params> {
    let (rows, _, _, _, a) = shape(subject);
    let leaf = chol::leaf(ctx);
    (a.kind == FactorKind::Cholesky && rows <= leaf).then(|| Params::Factor(FactorParams { threads: leaf as u32, ..FactorParams::NONE }))
}

fn fits_chol_blocked(subject: &Subject<'_>, ctx: &Context, _: &Residencies) -> Option<Params> {
    let (rows, _, _, _, a) = shape(subject);
    (a.kind == FactorKind::Cholesky && rows > chol::leaf(ctx)).then_some(Params::Factor(FactorParams::NONE))
}

// -- QR --

/// `batch == 1 && cols <= 32 && tsqr::plan(..)` is `Some` (`LUMA_QR_TSQR=0` makes it `None`).
fn fits_qr_tsqr_panel(subject: &Subject<'_>, ctx: &Context, _: &Residencies) -> Option<Params> {
    let (rows, cols, batch, _, a) = shape(subject);
    if a.kind != FactorKind::Qr || batch != 1 || cols > crate::dense::qr::PANEL {
        return None;
    }
    let plan = tsqr::choose(ctx, rows, cols)?;
    some(FactorParams { groups: plan.groups as u32, bs: plan.bs as u32, k: plan.k as u32, ..FactorParams::NONE })
}

/// `cols <= (batch > 1 || rows <= 64 ? 32 : panel_width())`.
fn fits_qr_leaf(subject: &Subject<'_>, _: &Context, _: &Residencies) -> Option<Params> {
    let (rows, cols, batch, _, a) = shape(subject);
    let width = if batch > 1 || rows <= 64 { crate::dense::qr::PANEL } else { crate::dense::panel_width() };
    (a.kind == FactorKind::Qr && cols <= width).then_some(Params::Factor(FactorParams::NONE))
}

fn fits_qr_recursive(subject: &Subject<'_>, _: &Context, _: &Residencies) -> Option<Params> {
    let (_, _, _, _, a) = shape(subject);
    (a.kind == FactorKind::Qr).then_some(Params::Factor(FactorParams::NONE))
}

// -- Triangular solves --

fn fits_tri_substitute(subject: &Subject<'_>, _: &Context, _: &Residencies) -> Option<Params> {
    let (rows, _, _, _, a) = shape(subject);
    (a.kind == FactorKind::SolveTriangular && rows <= tri::BASE).then_some(Params::Factor(FactorParams::NONE))
}

/// `trsv::applies(ctx, n, k, batch)` (`LUMA_TRSV=0` makes it false).
fn fits_tri_trsv(subject: &Subject<'_>, ctx: &Context, _: &Residencies) -> Option<Params> {
    let (rows, _, batch, k, a) = shape(subject);
    if a.kind != FactorKind::SolveTriangular || !trsv::applies(ctx, rows, k, batch) {
        return None;
    }
    let (kc, inverse) = trsv::plan(rows, k, a.inverse);
    some(FactorParams { kc: kc as u32, inverse, ..FactorParams::NONE })
}

fn fits_tri_blocked(subject: &Subject<'_>, _: &Context, _: &Residencies) -> Option<Params> {
    let (_, _, _, _, a) = shape(subject);
    (a.kind == FactorKind::SolveTriangular).then_some(Params::Factor(FactorParams::NONE))
}

// -- SVD and eigh --

/// `dense::svd::plan(..)`: the workgroup-memory kernel takes the matrix (`rows >= cols` here).
fn small(subject: &Subject<'_>, ctx: &Context, kind: FactorKind) -> Option<Params> {
    let (rows, cols, _, _, a) = shape(subject);
    if a.kind != kind {
        return None;
    }
    let plan = svd::plan(ctx, if kind == FactorKind::Svd { Kind::Svd } else { Kind::Eigh }, a.vectors, rows, cols)?;
    some(FactorParams { svd: Some(plan), ..FactorParams::NONE })
}

fn fits_svd_small(subject: &Subject<'_>, ctx: &Context, _: &Residencies) -> Option<Params> {
    small(subject, ctx, FactorKind::Svd)
}

fn fits_eigh_small(subject: &Subject<'_>, ctx: &Context, _: &Residencies) -> Option<Params> {
    small(subject, ctx, FactorKind::Eigh)
}

/// A big matrix of `kind`: the workgroup-memory kernel does not take it; for an SVD, `preconditioned` is whether the QR route does.
fn big(subject: &Subject<'_>, ctx: &Context, kind: FactorKind) -> Option<(usize, bool)> {
    let (rows, cols, _, _, a) = shape(subject);
    if a.kind != kind || small(subject, ctx, kind).is_some() {
        return None;
    }
    let preconditioned = kind == FactorKind::Svd && a.exact && svd::precondition(cols);
    Some((cols, preconditioned)).filter(|_| rows >= cols)
}

fn fits_svd_jacobi(subject: &Subject<'_>, ctx: &Context, _: &Residencies) -> Option<Params> {
    big(subject, ctx, FactorKind::Svd).filter(|&(cols, pre)| !pre && cols < svd::BLOCK_MIN).map(|_| Params::Factor(FactorParams::NONE))
}

fn fits_svd_block_jacobi(subject: &Subject<'_>, ctx: &Context, _: &Residencies) -> Option<Params> {
    big(subject, ctx, FactorKind::Svd).filter(|&(cols, pre)| !pre && cols >= svd::BLOCK_MIN).map(|_| Params::Factor(FactorParams::NONE))
}

/// `tol == 0` and `min(rows, cols) >= 1024` (`LUMA_SVD_PRECOND`): QR first, then Jacobi on `Rᵀ`.
fn fits_svd_preconditioned(subject: &Subject<'_>, ctx: &Context, _: &Residencies) -> Option<Params> {
    big(subject, ctx, FactorKind::Svd).filter(|&(_, pre)| pre).map(|_| Params::Factor(FactorParams::NONE))
}

fn fits_eigh_jacobi(subject: &Subject<'_>, ctx: &Context, _: &Residencies) -> Option<Params> {
    big(subject, ctx, FactorKind::Eigh).filter(|&(cols, _)| cols < svd::BLOCK_MIN).map(|_| Params::Factor(FactorParams::NONE))
}

fn fits_eigh_block_jacobi(subject: &Subject<'_>, ctx: &Context, _: &Residencies) -> Option<Params> {
    big(subject, ctx, FactorKind::Eigh).filter(|&(cols, _)| cols >= svd::BLOCK_MIN).map(|_| Params::Factor(FactorParams::NONE))
}

// -- Refined solves --

/// A square system, outside a recording (the iteration reads back, which a recording cannot).
fn fits_lu_refined_f32(subject: &Subject<'_>, ctx: &Context, _: &Residencies) -> Option<Params> {
    let (rows, cols, _, _, a) = shape(subject);
    (a.kind == FactorKind::SolveRefined && rows == cols && !ctx.recording_open()).then_some(Params::Factor(FactorParams::NONE))
}

fn fits_chol_refined_f32(subject: &Subject<'_>, ctx: &Context, _: &Residencies) -> Option<Params> {
    let (rows, cols, _, _, a) = shape(subject);
    (a.kind == FactorKind::SolveSpdRefined && rows == cols && !ctx.recording_open()).then_some(Params::Factor(FactorParams::NONE))
}
