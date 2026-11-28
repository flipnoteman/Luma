//! The reduce family's strategies: one per branch of what `Structured::record_once` chose between with `if`s. Table order is
//! today's priority: host, merged, axis merged, bits, single pass, two passes, axis lines. Every one is a baseline (it exists at
//! f0e5723) and its own rank group; none has a cost in wave 2, so selection is the priority chain (checked by the Decision log).

use super::{AxisGeom, ElemTy, Structured};
use crate::array::Failure;
use crate::caps::Caps;
use crate::context::Context;
use crate::jit::structured::emit::Emit;
use crate::jit::structured::strategy::{ResClass, BitClass, FamilyTag, FitsInputs, Lowered, NumClass, Output, Params, PassMode, ReduceParams, Residencies, SplitClass, Strategy, StrategyId, Subject, Warm};

impl StrategyId {
    pub(crate) const REDUCE_HOST: StrategyId = StrategyId(FamilyTag::Reduce, 0);
    pub(crate) const REDUCE_MERGED: StrategyId = StrategyId(FamilyTag::Reduce, 1);
    pub(crate) const REDUCE_AXIS_MERGED: StrategyId = StrategyId(FamilyTag::Reduce, 2);
    pub(crate) const REDUCE_BITS: StrategyId = StrategyId(FamilyTag::Reduce, 3);
    pub(crate) const REDUCE_SINGLE_PASS: StrategyId = StrategyId(FamilyTag::Reduce, 4);
    pub(crate) const REDUCE_TWO_PASS: StrategyId = StrategyId(FamilyTag::Reduce, 5);
    pub(crate) const REDUCE_AXIS_LINES: StrategyId = StrategyId(FamilyTag::Reduce, 6);
    pub(crate) const REDUCE_CANONICAL_BLOCKS: StrategyId = StrategyId(FamilyTag::Reduce, 7);
}

/// The one strategy of a reduction over several segments in a context that splits: its first pass is canonical blocks, deferred into
/// the schedule (`jit::split`, D8). It is a table of its own, selected for `ResClass::Streamed`, so that the table above (and the
/// Decision log of every context that does not split) is what it was.
pub(crate) const REDUCE_STREAMED: [Strategy; 1] =
    [entry(StrategyId::REDUCE_CANONICAL_BLOCKS, "canonical-blocks", Caps::empty(), FitsInputs::NONE, 7, 0, fits_canonical, lower_canonical)];

/// The reductions' table, in preference order.
pub(crate) const REDUCE_STRATEGIES: [Strategy; 7] = [
    // The host's `f64` accumulation differs from the device's: its own bit class. `fits` of the host and merged paths read the
    // operand's life (forced, failed, pending) and the reductions registered beside it: `LIVE`, so the family is never cached.
    entry(StrategyId::REDUCE_HOST, "host", Caps::HOST_PATH, FitsInputs::HOST.with(FitsInputs::LIVE), 1, 0, fits_host, lower_host),
    entry(StrategyId::REDUCE_MERGED, "merged", Caps::empty(), FitsInputs::HOST.with(FitsInputs::LIVE), 2, 1, fits_merged, lower_merged),
    entry(StrategyId::REDUCE_AXIS_MERGED, "axis-merged", Caps::empty(), FitsInputs::HOST.with(FitsInputs::LIVE), 3, 2, fits_axis_merged, lower_axis_merged),
    entry(StrategyId::REDUCE_BITS, "bits", Caps::empty(), FitsInputs::NONE, 4, 3, fits_bits, lower_bits),
    // A first pass of several groups that finishes itself, or a second pass: different bits where the first pass has several groups.
    entry(StrategyId::REDUCE_SINGLE_PASS, "single-pass", Caps::empty(), FitsInputs::NONE, 5, 4, fits_single_pass, lower_single_pass),
    entry(StrategyId::REDUCE_TWO_PASS, "two-pass", Caps::empty(), FitsInputs::NONE, 6, 5, fits_two_pass, lower_two_pass),
    entry(StrategyId::REDUCE_AXIS_LINES, "axis", Caps::empty(), FitsInputs::NONE, 3, 6, fits_axis_lines, lower_axis_lines),
];

#[allow(clippy::too_many_arguments)]
const fn entry(
    id: StrategyId,
    name: &'static str,
    needs: Caps,
    reads: FitsInputs,
    bits: u16,
    group: u8,
    fits: fn(&Subject<'_>, &Context, &Residencies) -> Option<Params>,
    lower: fn(&Subject<'_>, &Params, &mut Emit<'_>) -> Result<Lowered, Failure>,
) -> Strategy {
    Strategy { id, name, needs, numerics: NumClass::Exact, bits: BitClass(bits), baseline: true, group, prologue: true, epilogue: false, partners: false, reads, fits, cost: None, split: split_reduce, warm: no_warm, lower: Some(lower) }
}

fn split_reduce(_: &Subject<'_>, _: &Params) -> SplitClass {
    SplitClass::Reduce
}

/// Baseline strategies are never warmed (they may block on their own compile, as before).
fn no_warm(_: &Subject<'_>, _: &Params, _: &Context) -> Result<Warm, Failure> {
    Ok(Warm { kernels: Vec::new() })
}

fn reduced<'a>(subject: &Subject<'a>) -> Option<&'a Structured> {
    match subject {
        Subject::Comp(reduced) => Some(reduced),
        Subject::Contraction(_) | Subject::Boundary(_) | Subject::Form(..) => None,
    }
}

fn params(reduced: &Structured, ctx: &Context, pass: PassMode) -> Params {
    Params::Reduce(ReduceParams { pass, groups: reduced.first_groups(ctx) })
}

fn one_group(pass: PassMode) -> Params {
    Params::Reduce(ReduceParams { pass, groups: 1 })
}

fn geom(reduced: &Structured) -> Option<AxisGeom> {
    reduced.geom()
}

fn done(storage: std::sync::Arc<crate::array::Storage>) -> Result<Lowered, Failure> {
    Ok(Lowered::Done(Output::Storage(storage)))
}

// -- host --

fn fits_host(subject: &Subject<'_>, ctx: &Context, _: &Residencies) -> Option<Params> {
    let reduced = reduced(subject)?;
    crate::host::reduced_fits(reduced, ctx).then(|| one_group(PassMode::Auto))
}

/// The host evaluation; `None` from it (it did not work out) is `Declined`.
fn lower_host(subject: &Subject<'_>, _: &Params, emit: &mut Emit<'_>) -> Result<Lowered, Failure> {
    let Some(reduced) = reduced(subject) else { return Ok(Lowered::Declined) };
    match crate::host::try_eval_reduced(reduced, emit.ctx()) {
        Some(storage) => done(storage),
        None => Ok(Lowered::Declined),
    }
}

// -- merged --

fn fits_merged(subject: &Subject<'_>, ctx: &Context, _: &Residencies) -> Option<Params> {
    let reduced = reduced(subject)?;
    (!reduced.is_axis() && reduced.mergeable(ctx) && ctx.jit().may_have_partners(reduced)).then(|| params(reduced, ctx, PassMode::Auto))
}

/// Forces the leaves the operand waits for (a side effect: hence here, never in `fits`) and records the group in one kernel. No
/// partners, nothing recorded, or a kernel that is not due or not ready (every member then runs alone, see `record_merged`): `Declined`.
fn lower_merged(subject: &Subject<'_>, _: &Params, emit: &mut Emit<'_>) -> Result<Lowered, Failure> {
    let Some(reduced) = reduced(subject) else { return Ok(Lowered::Declined) };
    let partners = reduced.take_partners(emit.ctx());
    if partners.is_empty() {
        return Ok(Lowered::Declined);
    }
    match reduced.record_merged(emit, &partners) {
        Some(storage) => done(storage),
        None => Ok(Lowered::Declined),
    }
}

// -- axis --

fn fits_axis_merged(subject: &Subject<'_>, ctx: &Context, _: &Residencies) -> Option<Params> {
    let reduced = reduced(subject)?;
    (reduced.is_axis() && reduced.axis_mergeable(ctx) && ctx.jit().may_have_partners(reduced)).then(|| one_group(PassMode::Auto))
}

fn lower_axis_merged(subject: &Subject<'_>, _: &Params, emit: &mut Emit<'_>) -> Result<Lowered, Failure> {
    let Some(reduced) = reduced(subject) else { return Ok(Lowered::Declined) };
    let Some(geom) = geom(reduced) else { return Ok(Lowered::Declined) };
    let partners = reduced.take_partners(emit.ctx());
    match reduced.record_axis_merged(emit, geom, &partners)? {
        Some(storage) => done(storage),
        None => Ok(Lowered::Declined),
    }
}

fn fits_axis_lines(subject: &Subject<'_>, _: &Context, _: &Residencies) -> Option<Params> {
    reduced(subject)?.is_axis().then(|| one_group(PassMode::Auto))
}

fn lower_axis_lines(subject: &Subject<'_>, _: &Params, emit: &mut Emit<'_>) -> Result<Lowered, Failure> {
    let Some(reduced) = reduced(subject) else { return Ok(Lowered::Declined) };
    let Some(geom) = geom(reduced) else { return Ok(Lowered::Declined) };
    done(reduced.record_axis_lines(emit, geom)?)
}

// -- canonical blocks --

fn fits_canonical(subject: &Subject<'_>, ctx: &Context, res: &Residencies) -> Option<Params> {
    let reduced = reduced(subject)?;
    (res.class == ResClass::Streamed && reduced.streamed(ctx)).then(|| params(reduced, ctx, PassMode::TwoPass))
}

fn lower_canonical(subject: &Subject<'_>, params: &Params, emit: &mut Emit<'_>) -> Result<Lowered, Failure> {
    let Some(reduced) = reduced(subject) else { return Ok(Lowered::Declined) };
    done(reduced.record_canonical(emit, StrategyId::REDUCE_CANONICAL_BLOCKS, *params)?)
}

// -- passes --

fn fits_bits(subject: &Subject<'_>, ctx: &Context, _: &Residencies) -> Option<Params> {
    let reduced = reduced(subject)?;
    // A pending operand still takes the fused first pass, the bit kernel runs where the operand is read from a buffer, as before:
    // the first pass follows the tuning (`Auto`).
    (!reduced.is_axis() && reduced.ty() == ElemTy::Bool).then(|| params(reduced, ctx, PassMode::Auto))
}

fn lower_bits(subject: &Subject<'_>, _: &Params, emit: &mut Emit<'_>) -> Result<Lowered, Failure> {
    let Some(reduced) = reduced(subject) else { return Ok(Lowered::Declined) };
    done(reduced.record_alone(emit, PassMode::Auto, true)?)
}

fn fits_single_pass(subject: &Subject<'_>, ctx: &Context, _: &Residencies) -> Option<Params> {
    let reduced = reduced(subject)?;
    (!reduced.is_axis() && ctx.tuning().single_pass_reduce && reduced.ty() != ElemTy::Bool).then(|| params(reduced, ctx, PassMode::SinglePass))
}

fn lower_single_pass(subject: &Subject<'_>, params: &Params, emit: &mut Emit<'_>) -> Result<Lowered, Failure> {
    let Some(reduced) = reduced(subject) else { return Ok(Lowered::Declined) };
    let Params::Reduce(params) = params else { unreachable!("a reduce strategy runs with reduce parameters") };
    done(reduced.record_alone(emit, params.pass, false)?)
}

fn fits_two_pass(subject: &Subject<'_>, ctx: &Context, _: &Residencies) -> Option<Params> {
    let reduced = reduced(subject)?;
    (!reduced.is_axis() && reduced.ty() != ElemTy::Bool).then(|| params(reduced, ctx, PassMode::TwoPass))
}

fn lower_two_pass(subject: &Subject<'_>, params: &Params, emit: &mut Emit<'_>) -> Result<Lowered, Failure> {
    let Some(reduced) = reduced(subject) else { return Ok(Lowered::Declined) };
    let Params::Reduce(params) = params else { unreachable!("a reduce strategy runs with reduce parameters") };
    done(reduced.record_alone(emit, params.pass, false)?)
}
