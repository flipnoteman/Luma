//! The strategy registry: one static table of the ways an operation can run, and what `select` needs to know about each.
//!
//! A family of operations (reductions first, see [`FamilyTag`]) lists its strategies in preference order in a table of its own
//! (`reduce::REDUCE_STRATEGIES`); [`table`] finds the table of a family. A strategy is plain data (function pointers, no trait
//! objects): the capabilities and numerics it needs, a pure `fits` that turns the value into the parameters of the run (today's
//! plan and gate functions), an optional `cost`, and `lower`, which runs it. `select` (see [`super::select`]) filters the table,
//! ranks what is left and returns the strategy to run; `execute` lowers in rank order until one does not decline.

use std::borrow::Cow;
use std::marker::PhantomData;
use std::panic::Location;
use std::sync::Arc;

use super::boundary::{ConvAttrs, ConvParams, OpaqueAttrs, OpaqueParams, WindowAttrs, WindowParams, CompactAttrs, FactorAttrs, FactorParams, LineSolveAttrs, LineSolveParams, MergeAttrs, ScanAttrs, ScanParams, ScatterAttrs, ScatterParams, SortAttrs, SortParams};
use super::emit::Emit;
use super::contraction::{ContractionParams, ContractionView};
use super::Structured;
use crate::array::{Array, Failure, Storage};
use crate::caps::Caps;
use crate::context::Context;
use crate::kernel::KernelCore;
use crate::numerics::Reduced;
use crate::profile::{Decision, DecisionSite, Verdict};

/// The families that have strategies. Contraction: W2-GEMM part A; the eager boundary families: W2-U3a/b/c.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum FamilyTag {
    Reduce,
    Sort,
    Scan,
    Scatter,
    Search,
    Compact,
    Merge,
    Contraction,
    LineSolve,
    Factor,
    Spectral,
    Sample,
    Conv,
    Window,
    Opaque,
}

/// How many families have a table.
pub(crate) const FAMILY_COUNT: usize = 15;

/// Most entries one family's table may have (the ranking cache keeps them in a `u32` mask and an array).
pub(crate) const MAX_PER_FAMILY: usize = 12;

impl FamilyTag {
    #[inline(always)]
    pub(crate) const fn index(self) -> usize {
        match self {
            FamilyTag::Reduce => 0,
            FamilyTag::Sort => 1,
            FamilyTag::Scan => 2,
            FamilyTag::Scatter => 3,
            FamilyTag::Search => 4,
            FamilyTag::Compact => 5,
            FamilyTag::Merge => 6,
            FamilyTag::Contraction => 7,
            FamilyTag::LineSolve => 8,
            FamilyTag::Factor => 9,
            FamilyTag::Spectral => 10,
            FamilyTag::Sample => 11,
            FamilyTag::Conv => 12,
            FamilyTag::Window => 13,
            FamilyTag::Opaque => 14,
        }
    }

    pub(crate) const fn name(self) -> &'static str {
        match self {
            FamilyTag::Reduce => "reduce",
            FamilyTag::Sort => "sort",
            FamilyTag::Scan => "scan",
            FamilyTag::Scatter => "scatter",
            FamilyTag::Search => "search",
            FamilyTag::Compact => "compact",
            FamilyTag::Merge => "merge",
            FamilyTag::Contraction => "contraction",
            FamilyTag::LineSolve => "line_solve",
            FamilyTag::Factor => "factor",
            FamilyTag::Spectral => "spectral",
            FamilyTag::Sample => "sample",
            FamilyTag::Conv => "conv",
            FamilyTag::Window => "window",
            FamilyTag::Opaque => "opaque",
        }
    }

    pub(crate) fn parse(name: &str) -> Option<FamilyTag> {
        match name {
            "reduce" => Some(FamilyTag::Reduce),
            "sort" => Some(FamilyTag::Sort),
            "scan" => Some(FamilyTag::Scan),
            "scatter" => Some(FamilyTag::Scatter),
            "search" => Some(FamilyTag::Search),
            "compact" => Some(FamilyTag::Compact),
            "merge" => Some(FamilyTag::Merge),
            "contraction" => Some(FamilyTag::Contraction),
            "line_solve" => Some(FamilyTag::LineSolve),
            "factor" => Some(FamilyTag::Factor),
            "spectral" => Some(FamilyTag::Spectral),
            "sample" => Some(FamilyTag::Sample),
            "conv" => Some(FamilyTag::Conv),
            "window" => Some(FamilyTag::Window),
            "opaque" => Some(FamilyTag::Opaque),
            _ => None,
        }
    }
}

/// A strategy: its family and its place in the family's table. Part of every cache key and [`Decision`].
///
/// Each family module declares its own consts (`reduce.rs`: `REDUCE_HOST`, ...), so adding a strategy edits that family's file
/// and its table only.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) struct StrategyId(pub(crate) FamilyTag, pub(crate) u8);

impl StrategyId {
    pub(crate) fn family(self) -> FamilyTag {
        self.0
    }

    /// The name the table gives it (`host`, `single-pass`, ...), also its `LUMA_STRATEGY` spelling and its Decision label.
    pub(crate) fn name(self) -> &'static str {
        table(self.0).iter().find(|strategy| strategy.id == self).map_or("?", |strategy| strategy.name)
    }

    /// The strategy of `family` called `name` (`LUMA_STRATEGY`).
    pub(crate) fn parse(family: FamilyTag, name: &str) -> Option<StrategyId> {
        table(family).iter().find(|strategy| strategy.name == name).map(|strategy| strategy.id)
    }
}

/// What the strategy leaves unsplit or splits, for the schedule compiler (01 §2.3). Declared now, read by W3-U6; nothing in wave 2
/// consumes it.
#[allow(dead_code)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum SplitClass {
    Map,
    Halo { flat_radius: u32 },
    Reduce,
    Scan,
    Scatter,
    Global,
}

/// How the strategy's results relate to the family's reference strategy (06 D5).
#[allow(dead_code)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum NumClass {
    Exact,
    Reassociating,
    Reduced(Reduced),
}

/// Strategies with equal `BitClass` give bit-identical results for the same value on the same device and tuning. Measured costs may
/// reorder only within one class (see `telemetry`).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) struct BitClass(pub(crate) u16);

/// One way to run a family's operation. `prologue`, `epilogue` and `split` are declared for the schedule compiler and the epilogue
/// fusion (W3-U6, W3-U4); nothing in wave 2 reads them.
#[allow(dead_code)]
pub(crate) struct Strategy {
    pub(crate) id: StrategyId,
    /// The name in `LUMA_STRATEGY` and in Decision labels.
    pub(crate) name: &'static str,
    /// Capability only; sizes stay in `fits`.
    pub(crate) needs: Caps,
    pub(crate) numerics: NumClass,
    pub(crate) bits: BitClass,
    /// Exists at f0e5723: never refused by the variant budget or by "still compiling".
    pub(crate) baseline: bool,
    /// Rank group: costs are compared only inside a group (a run of entries with the same `group`), groups keep table order.
    pub(crate) group: u8,
    pub(crate) prologue: bool,
    /// Can take an absorbed epilogue (false for every wave-2 strategy).
    pub(crate) epilogue: bool,
    /// Today's plan and gate functions, pure: the parameters of the run, or `None` where the strategy does not apply.
    pub(crate) fits: FitsFn,
    /// `None`: table order decides.
    pub(crate) cost: Option<CostFn>,
    pub(crate) split: fn(&Subject<'_>, &Params) -> SplitClass,
    /// Builds (starts compiling) the kernels the strategy needs and launches nothing.
    pub(crate) warm: WarmFn,
    /// `None` for eager-boundary families: the op matches on the id.
    pub(crate) lower: Option<LowerFn>,
}

pub(crate) type FitsFn = fn(&Subject<'_>, &Context, &Residencies) -> Option<Params>;
pub(crate) type CostFn = fn(&Subject<'_>, &Params, &Context, &Residencies) -> Estimate;
pub(crate) type WarmFn = fn(&Subject<'_>, &Params, &Context) -> Result<Warm, Failure>;
pub(crate) type LowerFn = fn(&Subject<'_>, &Params, &mut Emit<'_>) -> Result<Lowered, Failure>;

/// The strategy table of `family`, in preference order.
pub(crate) fn table(family: FamilyTag) -> &'static [Strategy] {
    TABLES[family.index()]
}

/// One table per family; the shared edit of a new family is one line here.
///
/// Search, Compact, Merge, Spectral and Sample are light ports (tag, site and `note`): their tables stay empty, so `LUMA_STRATEGY` refuses them.
static TABLES: [&[Strategy]; FAMILY_COUNT] = [&crate::reduce::REDUCE_STRATEGIES, &super::families::sort::SORT_STRATEGIES, &super::families::scan::SCAN_STRATEGIES, &super::families::scatter::SCATTER_STRATEGIES, &[], &[], &[], &super::contraction::CONTRACTION_STRATEGIES, &super::families::line_solve::LINE_SOLVE_STRATEGIES, &super::families::factor::FACTOR_STRATEGIES, &[], &[], &super::families::conv::CONV_STRATEGIES, &super::families::window::WINDOW_STRATEGIES, &super::families::opaque::OPAQUE_STRATEGIES];

/// What a strategy is asked about: one variant per kind of value.
pub(crate) enum Subject<'a> {
    Reduce(&'a Structured),
    Contraction(&'a ContractionView<'a>),
    Boundary(&'a Boundary<'a>),
}

impl Subject<'_> {
    pub(crate) fn family(&self) -> FamilyTag {
        match self {
            Subject::Reduce(_) => FamilyTag::Reduce,
            Subject::Contraction(_) => FamilyTag::Contraction,
            Subject::Boundary(boundary) => boundary.family,
        }
    }

    /// The shape that picks the plan (the cache key's shape class and the telemetry cell's).
    pub(crate) fn dims(&self) -> [u32; 4] {
        match self {
            Subject::Reduce(reduced) => reduced.dims(),
            Subject::Contraction(view) => view.dims(),
            Subject::Boundary(boundary) => boundary.dims,
        }
    }

    /// The operation that asked and where it was written, for the Decision.
    pub(crate) fn site(&self) -> (&'static str, &'static Location<'static>) {
        match self {
            Subject::Reduce(reduced) => (reduced.name, reduced.location),
            Subject::Contraction(view) => ("matmul", view.location),
            Subject::Boundary(boundary) => (boundary.op, boundary.location),
        }
    }

    /// The Decision site of the family.
    pub(crate) fn decision_site(&self) -> DecisionSite {
        match self.family() {
            FamilyTag::Reduce => DecisionSite::ReducePlan,
            FamilyTag::Sort => DecisionSite::SortPlan,
            FamilyTag::Scan => DecisionSite::ScanPlan,
            FamilyTag::Scatter => DecisionSite::ScatterPlan,
            FamilyTag::Search => DecisionSite::SearchPlan,
            FamilyTag::Compact => DecisionSite::CompactPlan,
            FamilyTag::Merge => DecisionSite::MergePlan,
            FamilyTag::Contraction => DecisionSite::MatmulPlan,
            FamilyTag::LineSolve => DecisionSite::LineSolvePlan,
            FamilyTag::Factor => DecisionSite::FactorPlan,
            FamilyTag::Spectral => DecisionSite::SpectralPlan,
            FamilyTag::Sample => DecisionSite::SamplePlan,
            FamilyTag::Conv => DecisionSite::ConvPlan,
            FamilyTag::Window => DecisionSite::WindowPlan,
            FamilyTag::Opaque => DecisionSite::OpaquePlan,
        }
    }

    /// Whether the caller asked for reduced precision itself (`matmul_tf32`), whatever the context's numerics allow.
    pub(crate) fn requests_reduced(&self) -> bool {
        matches!(self, Subject::Contraction(view) if view.attrs.tf32 == super::contraction::Tf32Request::Tf32)
    }
}

/// The plain data a strategy's `fits` produces and its `lower` runs with: one variant per family.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Params {
    Reduce(ReduceParams),
    Sort(SortParams),
    Scan(ScanParams),
    Scatter(ScatterParams),
    Contraction(ContractionParams),
    LineSolve(LineSolveParams),
    Factor(FactorParams),
    Conv(ConvParams),
    Window(WindowParams),
    Opaque(OpaqueParams),
}

impl Params {
    /// The Decision text of the strategy `id` run with these parameters: no ids or pointers.
    pub(crate) fn label(&self, id: StrategyId, subject: &Subject<'_>, ctx: &Context) -> Cow<'static, str> {
        self.label_of(id.name(), subject, ctx)
    }

    /// [`Params::label`] of a strategy called `name`.
    pub(crate) fn label_of(&self, name: &'static str, subject: &Subject<'_>, ctx: &Context) -> Cow<'static, str> {
        match self {
            Params::Reduce(params) if params.groups > 1 => Cow::Owned(format!("{name} groups={}", params.groups)),
            Params::Reduce(_) => Cow::Borrowed(name),
            // The sort's three radix paths keep the labels the log has had since the sort recorded its choice.
            Params::Sort(_) => Cow::Borrowed(match name {
                "radix_sweep" => "sweep",
                "radix_ballots" => "radix ballot",
                "radix_count_scan" => "radix",
                other => other,
            }),
            Params::Scan(params) if params.fused => Cow::Owned(format!("{name} prologue=fused")),
            Params::Factor(params) if params.nb > 0 => Cow::Owned(format!("{name} nb={}", params.nb)),
            Params::Scan(_) | Params::Scatter(_) | Params::Factor(_) | Params::Conv(_) | Params::Window(_) => Cow::Borrowed(name),
            Params::Contraction(params) => Cow::Owned(super::contraction::label(params, subject, ctx)),
            // The line solver's log has said `cyclic ...` or `thomas ...` since it recorded its choice: the strategies that are
            // not cyclic reduction (the native and the stacked ones included) are all the Thomas algorithm there.
            Params::LineSolve(params) => Cow::Owned(params.legacy_label(name)),
            Params::Opaque(_) => Cow::Owned(format!("opaque::{name}")),
        }
    }
}

/// What a reduction runs with: how its passes finish, and how many workgroups its first pass has.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct ReduceParams {
    pub(crate) pass: PassMode,
    pub(crate) groups: u32,
}

/// Whether a reduction's first pass finishes the reduction itself.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum PassMode {
    /// Read `Tuning::single_pass_reduce`, as every caller that is not a reduction's force does.
    Auto,
    TwoPass,
    SinglePass,
}

impl PassMode {
    /// Whether the first pass of several workgroups folds the partial results itself.
    #[inline(always)]
    pub(crate) fn single(self, ctx: &Context) -> bool {
        match self {
            PassMode::Auto => ctx.tuning().single_pass_reduce,
            PassMode::TwoPass => false,
            PassMode::SinglePass => true,
        }
    }
}

/// A cost in nanoseconds and where it came from.
#[derive(Clone, Copy, PartialEq, Debug)]
pub(crate) struct Estimate {
    pub(crate) ns: f64,
    /// From sampled launch timings rather than a model.
    pub(crate) measured: bool,
}

/// The kernels a strategy needs, built (compiling) and not yet launched; ready when they all are.
pub(crate) struct Warm {
    pub(crate) kernels: Vec<Arc<KernelCore>>,
}

impl Warm {
    pub(crate) fn ready(&self) -> bool {
        self.kernels.iter().all(|core| core.raw.is_ready())
    }

    pub(crate) fn failed(&self) -> bool {
        self.kernels.iter().any(|core| core.raw.failed())
    }
}

/// What a strategy's `lower` did.
pub(crate) enum Lowered {
    Done(Output),
    /// It cannot run after all (the merged program does not fit one kernel, the host evaluation failed half way): `select`'s
    /// ranking goes on with the next candidate. Never an error.
    Declined,
}

/// What a lowered value is.
pub(crate) enum Output {
    Storage(Arc<Storage>),
    /// An array that is already a value of its own: `gemm-narrow-n`'s result is a lazy view of the product of a padded operand. Transitional
    /// (W2-GEMM part B retires the strategy and the variant).
    Array(Box<Array<f32>>),
}

impl Output {
    pub(crate) fn into_storage(self) -> Arc<Storage> {
        match self {
            Output::Storage(storage) => storage,
            Output::Array(_) => unreachable!("only a contraction strategy returns an array"),
        }
    }
}

/// Where the operands live; constant in wave 2 (W3-U6 fills it).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct Residencies {
    pub(crate) class: ResClass,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum ResClass {
    Device,
}

impl Residencies {
    pub(crate) const DEVICE: Residencies = Residencies { class: ResClass::Device };
}

// -- Eager-boundary families --

/// The stack descriptor an eager op selects over (sort, scan, scatter, ...: not lazy values yet): no `Structured` is allocated.
pub(crate) struct Boundary<'a> {
    pub(crate) family: FamilyTag,
    pub(crate) op: &'static str,
    pub(crate) location: &'static Location<'static>,
    /// The shape that picks the plan.
    pub(crate) dims: [u32; 4],
    /// Small and `Copy`, one variant per family, defined by the family's spec.
    pub(crate) attrs: BoundaryAttrs,
    pub(crate) _borrow: PhantomData<&'a ()>,
}

/// What a boundary family's `fits` reads besides the shape; one variant per family.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[allow(dead_code)]
pub(crate) enum BoundaryAttrs {
    None,
    Sort(SortAttrs),
    Scan(ScanAttrs),
    Scatter(ScatterAttrs),
    Conv(ConvAttrs),
    Window(WindowAttrs),
    Opaque(OpaqueAttrs),
    /// Light ports: nothing selects on these, they only tell the Decision (and the signature) what the op was.
    Compact(CompactAttrs),
    Merge(MergeAttrs),
    LineSolve(LineSolveAttrs),
    Factor(FactorAttrs),
}

/// The candidates of a boundary op in table order whose `needs` and numerics hold and whose `fits` returned `Some`, evaluated
/// lazily; the override of `LUMA_STRATEGY` makes the forced strategy the only one (when it fits). No cache: their conditions are
/// O(1) tests, cheaper than a lookup over at most six entries. The op loops over it, runs the body of today's function for the id
/// and, where a host path declines after `fits`, takes the next.
#[allow(dead_code)]
pub(crate) fn candidates<'a>(ctx: &'a Context, boundary: &'a Boundary<'a>) -> impl Iterator<Item = (StrategyId, Params)> + 'a {
    candidates_in(table(boundary.family), ctx, boundary)
}

pub(crate) fn candidates_in<'a>(entries: &'static [Strategy], ctx: &'a Context, boundary: &'a Boundary<'a>) -> impl Iterator<Item = (StrategyId, Params)> + 'a {
    let forced = ctx.strategy_override(boundary.family).and_then(|id| {
        let strategy = entries.iter().find(|strategy| strategy.id == id)?;
        super::select::admit(strategy, &Subject::Boundary(boundary), ctx, &Residencies::DEVICE).ok().map(|params| (id, params))
    });
    // A forced strategy that fits is the only candidate.
    let mut at = if forced.is_some() { entries.len() } else { 0 };
    let mut forced = forced;
    std::iter::from_fn(move || {
        if let Some(only) = forced.take() {
            return Some(only);
        }
        while at < entries.len() {
            let strategy = &entries[at];
            at += 1;
            if let Ok(params) = super::select::admit(strategy, &Subject::Boundary(boundary), ctx, &Residencies::DEVICE) {
                return Some((strategy.id, params));
            }
        }
        None
    })
}

/// The parameters of the strategy `id` for `boundary` if it applies (capabilities, numerics, `fits`): for the host paths a boundary op
/// tries first, which are the only strategy that applies at their site.
#[inline]
pub(crate) fn admit_one(ctx: &Context, boundary: &Boundary<'_>, id: StrategyId) -> Option<Params> {
    // A strategy's id is its place in its family's table.
    let strategy = &table(boundary.family)[usize::from(id.1)];
    debug_assert_eq!(strategy.id, id);
    super::select::admit(strategy, &Subject::Boundary(boundary), ctx, &Residencies::DEVICE).ok()
}

/// Whether the device has the capabilities strategy `id` needs: a boundary op tries a host path only where this holds, and builds
/// no descriptor otherwise (the capabilities are fixed when the context opens; `fits` still decides the rest).
#[inline]
pub(crate) fn has_caps_for(ctx: &Context, family: FamilyTag, id: StrategyId) -> bool {
    ctx.caps().contains(table(family)[usize::from(id.1)].needs)
}

/// The first of [`candidates`]: the strategy a boundary op without a host path to try runs, without the iterator (one scan of the table
/// in order, the forced strategy of `LUMA_STRATEGY` first where it fits).
#[inline]
pub(crate) fn first_candidate(ctx: &Context, boundary: &Boundary<'_>) -> Option<(StrategyId, Params)> {
    if ctx.strategy_override(boundary.family).is_some() {
        return candidates(ctx, boundary).next();
    }
    let subject = Subject::Boundary(boundary);
    table(boundary.family).iter().find_map(|strategy| super::select::admit(strategy, &subject, ctx, &Residencies::DEVICE).ok().map(|params| (strategy.id, params)))
}

/// [`first_candidate`] over the entries of one stage of a family whose table serves several (`Factor`: the LU leaves, the QR panels,
/// the triangular solves, ... and the stages of conv, window and the sparse products); the forced strategy of `LUMA_STRATEGY` counts where it is one of `entries`.
#[inline(always)]
pub(crate) fn first_candidate_in(entries: &'static [Strategy], ctx: &Context, boundary: &Boundary<'_>) -> Option<(StrategyId, Params)> {
    if ctx.strategy_override(boundary.family).is_some() {
        return candidates_in(entries, ctx, boundary).next();
    }
    let subject = Subject::Boundary(boundary);
    entries.iter().find_map(|strategy| super::select::admit(strategy, &subject, ctx, &Residencies::DEVICE).ok().map(|params| (strategy.id, params)))
}

/// Records the Decision of a boundary op that chose `chosen` (with `params`) after `declined` ones gave up, while decisions are
/// recorded (profiling or `Context::record_decisions`): one relaxed load otherwise.
#[allow(dead_code)]
#[inline(always)]
pub(crate) fn note(ctx: &Context, boundary: &Boundary<'_>, chosen: StrategyId, params: &Params, declined: &[StrategyId]) {
    if ctx.jit().recording_decisions() {
        note_recorded(ctx, boundary, table(boundary.family), chosen, params, declined);
    }
}

/// [`note`] listing as alternatives only the `entries` of the stage that chose.
#[inline(always)]
pub(crate) fn note_in(ctx: &Context, boundary: &Boundary<'_>, entries: &'static [Strategy], chosen: StrategyId, params: &Params, declined: &[StrategyId]) {
    if ctx.jit().recording_decisions() {
        note_recorded(ctx, boundary, entries, chosen, params, declined);
    }
}

#[cold]
#[inline(never)]
fn note_recorded(ctx: &Context, boundary: &Boundary<'_>, entries: &'static [Strategy], chosen: StrategyId, params: &Params, declined: &[StrategyId]) {
    let subject = Subject::Boundary(boundary);
    let mut decision = Decision::new(boundary.op, boundary.location, subject.decision_site(), params.label(chosen, &subject, ctx));
    for strategy in entries {
        if strategy.id == chosen {
            continue;
        }
        let verdict = if declined.contains(&strategy.id) {
            Verdict::Rejected("declined")
        } else {
            match super::select::admit(strategy, &subject, ctx, &Residencies::DEVICE) {
                Ok(_) => Verdict::Rejected("lower priority"),
                Err(reason) => Verdict::Rejected(reason.text()),
            }
        };
        decision = decision.candidate(strategy.name, verdict);
    }
    ctx.jit().note_decision(decision);
}
