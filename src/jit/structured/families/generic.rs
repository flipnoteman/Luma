//! `FamilyTag::Generic`: the lowering of a form that no API tag and no recogniser took (W2-R0 §5.3-5.6). Three strategies, last in
//! the order of the families and never reached by an `Origin::Api` form (the 07 §9 guard): their `fits` returns `None` for it. The one exception is `OpTag::Einsum`, which has no strategy of its own.
//!
//! * `generic-contract` (G-A): the fused matmul tail with the body as the multiply-add, for a separable body and the roles of a
//!   contraction.
//! * `generic-reduce` (G-B1): the axis-reduce tail over a program of views, for any body (what `(bcast(a) * bcast(b)).sum_axis`
//!   runs today).
//! * `generic-unrolled` (G-B2): the same with the taps of a window unrolled, for rows that read two dims.
//!
//! One rank group: the cost decides between the contraction and the reduction tail where both fit. All three are baselines (a form
//! that reaches them has no other way to run, so the variant budget never refuses them).

use super::super::strategy::{BitClass, Estimate, FamilyTag, FitsInputs, GenericParams, Lowered, NumClass, Output, Params, Residencies, SplitClass, Strategy, StrategyId, Subject, Warm};
use super::super::emit::Emit;
use crate::array::Failure;
use crate::caps::Caps;
use crate::context::Context;
use crate::jit::comprehension::lower::{self, contract, Prepared};
use crate::jit::comprehension::{OpTag, Origin};

impl StrategyId {
    pub(crate) const GENERIC_CONTRACT: StrategyId = StrategyId(FamilyTag::Generic, 0);
    pub(crate) const GENERIC_REDUCE: StrategyId = StrategyId(FamilyTag::Generic, 1);
    pub(crate) const GENERIC_UNROLLED: StrategyId = StrategyId(FamilyTag::Generic, 2);
}

pub(crate) const GENERIC_STRATEGIES: [Strategy; 3] = [
    // The contraction tail sums `k` in the blocked order of the GEMM, the reduction tail in its own: different bits.
    strategy(StrategyId::GENERIC_CONTRACT, "generic-contract", 0, fits_contract, cost_contract, lower_contract),
    strategy(StrategyId::GENERIC_REDUCE, "generic-reduce", 1, fits_reduce, cost_reduce, lower_reduce),
    strategy(StrategyId::GENERIC_UNROLLED, "generic-unrolled", 1, fits_unrolled, cost_reduce, lower_reduce),
];

const fn strategy(id: StrategyId, name: &'static str, bits: u16, fits: super::super::strategy::FitsFn, cost: super::super::strategy::CostFn, lower: super::super::strategy::LowerFn) -> Strategy {
    Strategy {
        id,
        name,
        needs: Caps::empty(),
        numerics: NumClass::Exact,
        bits: BitClass(bits),
        baseline: true,
        group: 0,
        prologue: true,
        epilogue: false,
        partners: false,
        // `fits` reads the whole form (its space, accesses and body), which no key holds: never cached.
        reads: FitsInputs::LIVE,
        fits,
        cost: Some(cost),
        split: split_reduce,
        warm: no_warm,
        lower: Some(lower),
    }
}

fn split_reduce(_: &Subject<'_>, _: &Params) -> SplitClass {
    SplitClass::Reduce
}

fn no_warm(_: &Subject<'_>, _: &Params, _: &Context) -> Result<Warm, Failure> {
    Ok(Warm { kernels: Vec::new() })
}

/// The form of a subject that is not an API form.
fn form<'a>(subject: &Subject<'a>) -> Option<(&'a crate::jit::comprehension::Comprehension, &'static std::panic::Location<'static>)> {
    match subject {
        Subject::Form(form, location) if !matches!(&form.origin, Origin::Api(tag) if *tag != OpTag::Einsum) => Some((form, location)),
        _ => None,
    }
}

fn params(plan: &Prepared) -> Params {
    Params::Generic(GenericParams { tiles: u32::try_from(plan.tiles()).unwrap_or(u32::MAX), taps: u32::try_from(plan.taps()).unwrap_or(u32::MAX) })
}

fn fits_contract(subject: &Subject<'_>, _: &Context, _: &Residencies) -> Option<Params> {
    let (form, _) = form(subject)?;
    contract::prepare(form).map(|plan| Params::Generic(GenericParams { tiles: 1, taps: u32::try_from(plan.k).unwrap_or(u32::MAX) }))
}

fn fits_reduce(subject: &Subject<'_>, _: &Context, _: &Residencies) -> Option<Params> {
    let (form, _) = form(subject)?;
    lower::prepare(form).filter(|plan| !plan.unrolls()).map(|plan| params(&plan))
}

fn fits_unrolled(subject: &Subject<'_>, _: &Context, _: &Residencies) -> Option<Params> {
    let (form, _) = form(subject)?;
    lower::prepare(form).filter(Prepared::unrolls).map(|plan| params(&plan))
}

/// The points of the virtual space, a launch per tile.
fn points(subject: &Subject<'_>) -> f64 {
    match subject {
        Subject::Form(form, _) => form.space.iter().map(|dim| dim.extent as f64).product(),
        _ => 0.0,
    }
}

/// The tail reads every access at every point of the space.
fn cost_reduce(subject: &Subject<'_>, params: &Params, _: &Context, _: &Residencies) -> Estimate {
    let tiles = match params {
        Params::Generic(generic) => f64::from(generic.tiles),
        _ => 1.0,
    };
    let loads = match subject {
        Subject::Form(form, _) => form.inputs.len() as f64 + 1.0,
        _ => 1.0,
    };
    Estimate { ns: points(subject) * loads * 0.03 + tiles * 20_000.0, measured: false }
}

/// The GEMM tail does `2` flops per point at a high rate and reads each operand element once per tile.
fn cost_contract(subject: &Subject<'_>, _: &Params, _: &Context, _: &Residencies) -> Estimate {
    Estimate { ns: points(subject) * 0.0005 + 30_000.0, measured: false }
}

fn lower_reduce(subject: &Subject<'_>, _: &Params, emit: &mut Emit<'_>) -> Result<Lowered, Failure> {
    let Some((form, location)) = form(subject) else { return Ok(Lowered::Declined) };
    let Some(plan) = lower::prepare(form) else { return Ok(Lowered::Declined) };
    Ok(Lowered::Done(Output::Storage(plan.run(emit, location)?)))
}

fn lower_contract(subject: &Subject<'_>, _: &Params, emit: &mut Emit<'_>) -> Result<Lowered, Failure> {
    let Some((form, location)) = form(subject) else { return Ok(Lowered::Declined) };
    let Some(plan) = contract::prepare(form) else { return Ok(Lowered::Declined) };
    match plan.run(emit, location) {
        Ok(storage) => Ok(Lowered::Done(Output::Storage(storage))),
        // The operands do not fuse into one kernel: the reduction tail takes the form.
        Err(Failure::New(crate::error::LumaError::Unsupported(message))) if message == contract::NOT_FUSED => Ok(Lowered::Declined),
        Err(failure) => Err(failure),
    }
}
