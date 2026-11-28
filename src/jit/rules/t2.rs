//! Tier 2: the normaliser. When a structured value is forced it rewrites the planning [`Scope`] to a fixpoint, in a fixed order,
//! accepting a rewrite only if the [`Measure`] strictly decreases (at least 1/64 of the estimated time, or no slower with fewer
//! nodes; a rule that `dominates` is accepted on the node count alone). Bounded, deterministic, memoised, logged.
//!
//! **Termination.** `ns` is a non-negative integer. Each step that accepts on time shrinks it by a factor of at least 63/64, so
//! there are at most `64 ln(ns0)` of them; between two of those every step accepted on nodes strictly shrinks the node count
//! (at most [`MAX_SCOPE_NODES`]) without raising `ns`. A rewrite that raises `ns` or leaves both equal is never accepted, so an
//! A→B, B→A pair cannot loop. [`MAX_ITERS`] is the hard stop in case a cost function is inconsistent.
//!
//! **What T2 never does:** mutate a [`Structured`]; force or evaluate anything; cross threads; run inside a recording; change a
//! value that has a result. A rewrite is applied to a clone of the scope, and the value is later lowered from the clone's steps
//! ([`Planned`]); the value itself, which other holders may force, is untouched.

use std::sync::{Arc, Mutex};

use smallvec::SmallVec;

use super::measure::{Measure, Models, measure_with};
use super::registry::{RuleId, RuleInfo};
use super::scope::{CompId, MAX_SCOPE_COMPS, MAX_SCOPE_NODES, NoScope, Scope, ScopeRole, Site, SiteKind, SiteSig};
use crate::array::Storage;
use crate::context::Context;
use crate::error::LumaError;
use crate::gpu::hash::FxMap;
use crate::jit::comprehension::{Algebra, Comprehension, Origin};
use crate::jit::structured::Structured;
use crate::jit::structured::select;
use crate::jit::structured::strategy::Subject;
use crate::jit::Src;
use crate::numerics::Numerics;
use crate::profile::{Decision, DecisionSite, Verdict};

/// Rewrites one run of the normaliser may accept; the hard stop.
pub(crate) const MAX_ITERS: usize = 64;

/// A rule of tier 2: `apply` looks at one site of the scope and returns the rewrite it would make, or nothing.
pub(crate) struct T2Rule {
    pub(crate) info: &'static RuleInfo,
    pub(crate) sites: SiteKind,
    /// The rule applies to a scope of one comprehension and no producer (otherwise such a scope is not looked at).
    pub(crate) single: bool,
    pub(crate) apply: fn(&Scope, Site, &Numerics) -> Option<Rewrite>,
}

/// The rules of the library: chain order and sum of a product (W3-R4R5; rule 4 and rule 8 wait for W3-U4).
pub(crate) static T2_RULES: &[T2Rule] = &super::reform::REFORM_RULES;

/// What a rule changes in the scope.
#[derive(Clone, Default)]
pub(crate) struct Rewrite {
    /// Comprehensions that get a new form of the same type and output shape.
    pub(crate) replace: SmallVec<[(CompId, Comprehension); 2]>,
    /// New comprehensions, private to the scope; they get the ids [`Scope::next_id`] says, in order.
    pub(crate) add: SmallVec<[Comprehension; 2]>,
    /// Comprehensions this plan no longer forces (the value itself is untouched; other holders still can force it).
    pub(crate) drop: SmallVec<[CompId; 2]>,
    /// Accesses that read the output of a comprehension of the plan instead of what they read.
    pub(crate) feeds: SmallVec<[Feed; 2]>,
}

/// Access `access` of `consumer` reads the output of `producer`.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Feed {
    pub(crate) consumer: CompId,
    pub(crate) access: u8,
    pub(crate) producer: CompId,
}

/// One launch sequence of a plan: a form (`None`: the root's own value, which runs through the strategy of its tag) and the
/// accesses that read the outputs of earlier steps.
#[derive(Clone, Debug)]
pub(crate) struct PlannedStep {
    pub(crate) form: Option<Comprehension>,
    /// (access of the form, earlier step whose output it reads).
    pub(crate) feeds: SmallVec<[(u8, u8); 2]>,
}

/// What the normaliser made of a value: the steps that lower it, in order (the root last), and the program that made them.
#[derive(Clone, Debug)]
pub(crate) struct Planned {
    pub(crate) steps: SmallVec<[PlannedStep; 4]>,
    /// The rewrites applied, in order: the program, not the forms.
    pub(crate) applied: SmallVec<[(RuleId, SiteSig); 4]>,
}

/// What a run of the normaliser found.
pub(crate) struct Outcome {
    pub(crate) scope: Scope,
    pub(crate) applied: Vec<(RuleId, SiteSig)>,
    pub(crate) accepted: Vec<Accepted>,
    /// The hard stop was reached.
    pub(crate) capped: bool,
    /// The scope is over a bound: nothing was rewritten.
    pub(crate) too_large: bool,
}

/// One accepted rewrite, for the decision log.
pub(crate) struct Accepted {
    pub(crate) rule: &'static RuleInfo,
    pub(crate) before: Option<Measure>,
    pub(crate) after: Option<Measure>,
    /// Rules that matched at a site and were not taken, with why.
    pub(crate) rejected: Vec<(&'static str, &'static str)>,
    pub(crate) cached: bool,
}

/// The scopes the normaliser remembers rewrites for, per context.
pub(crate) type Memo = Mutex<FxMap<(u64, u64, u64), Arc<Vec<(RuleId, SiteSig)>>>>;

/// Most structures the memo keeps.
const MEMO_ENTRIES: usize = 1024;

/// What a context decides once, when it opens, about the rule engine.
#[derive(Debug)]
pub(crate) struct RulesState {
    /// Whether any rule of tier 1 for comprehensions may run (a rule is allowed by the numerics and the mode reaches `t1`): when
    /// false, which it always is while the table is empty, building a form costs one load and a branch.
    pub(crate) t1_active: bool,
    /// The same for tier 2 (mode `on`).
    pub(crate) t2_active: bool,
    /// Whether the scalar factors of a product's operands may move out of it (rule 12, `Reassociating`): the eager path's pre-filter
    /// reads this one flag, so a context that does not reassociate pays a load and a branch.
    pub(crate) scale_active: bool,
    pub(crate) memo: Memo,
}

impl RulesState {
    pub(crate) fn new(numerics: &Numerics) -> RulesState {
        RulesState::with(numerics, Algebra::current(), super::T1_RULES, T2_RULES)
    }

    /// The state for these rules and this mode (tests).
    pub(super) fn with(numerics: &Numerics, mode: Algebra, t1: &[super::t1::T1Rule], t2: &[T2Rule]) -> RulesState {
        RulesState {
            t1_active: mode >= Algebra::T1 && t1.iter().any(|rule| rule.info.allowed(numerics)),
            t2_active: mode >= Algebra::On && t2.iter().any(|rule| rule.info.allowed(numerics)),
            scale_active: mode >= Algebra::T1 && numerics.reassociate,
            memo: Mutex::new(FxMap::default()),
        }
    }
}

/// Runs the normaliser on `scope` with `rules` (sorted by gate rank, then order), the numerics `numerics` and the cost function
/// `cost`. `replay` is a recorded program: its rewrites are applied again without measuring, and a step that no longer applies
/// ends the replay (`None`: the program does not fit, run fresh).
pub(super) fn normalise(scope: Scope, rules: &[T2Rule], numerics: &Numerics, cost: &mut dyn FnMut(&Scope) -> Measure, replay: Option<&[(RuleId, SiteSig)]>) -> Option<Outcome> {
    if let Some(program) = replay {
        return replayed(scope, rules, numerics, program);
    }
    let mut outcome = Outcome { scope, applied: Vec::new(), accepted: Vec::new(), capped: false, too_large: false };
    if outcome.scope.comps.len() > MAX_SCOPE_COMPS || outcome.scope.nodes() > MAX_SCOPE_NODES {
        outcome.too_large = true;
        return Some(outcome);
    }
    let mut iters = 0;
    'iterate: loop {
        if iters == MAX_ITERS {
            debug_assert!(false, "the normaliser reached MAX_ITERS: a cost function that is not a function of the scope");
            outcome.capped = true;
            break;
        }
        // The estimate of the scope as it is, made when the first rule that needs one matches (a `dominates` rule needs none).
        let mut before: Option<Measure> = None;
        let single = outcome.scope.comps.len() == 1 && outcome.scope.edges.is_empty();
        let mut rejected: Vec<(&'static str, &'static str)> = Vec::new();
        for rule in rules {
            if !rule.info.allowed(numerics) || (single && !rule.single) {
                continue;
            }
            for site in outcome.scope.sites(rule.sites) {
                let Some(rewrite) = (rule.apply)(&outcome.scope, site, numerics) else { continue };
                let next = match outcome.scope.checked(&rewrite) {
                    Ok(next) => next,
                    Err(why) => {
                        rejected.push((rule.info.name, why));
                        continue;
                    }
                };
                let (taken, measured) = if rule.info.dominates {
                    // Cheaper by construction: the node count decides, with no estimate.
                    (next.nodes() < outcome.scope.nodes(), None)
                } else {
                    let was = *before.get_or_insert_with(|| cost(&outcome.scope));
                    let after = cost(&next);
                    (after.improves(was), Some((was, after)))
                };
                if !taken {
                    rejected.push((rule.info.name, "not faster"));
                    continue;
                }
                outcome.accepted.push(Accepted { rule: rule.info, before: measured.map(|m| m.0), after: measured.map(|m| m.1), rejected: std::mem::take(&mut rejected), cached: false });
                outcome.applied.push((rule.info.id, site));
                outcome.scope = next;
                iters += 1;
                continue 'iterate;
            }
        }
        break;
    }
    Some(outcome)
}

fn replayed(mut scope: Scope, rules: &[T2Rule], numerics: &Numerics, program: &[(RuleId, SiteSig)]) -> Option<Outcome> {
    let mut outcome = Outcome { scope: Scope::from_forms([], 0), applied: Vec::new(), accepted: Vec::new(), capped: false, too_large: false };
    for &(id, site) in program {
        let rule = rules.iter().find(|rule| rule.info.id == id)?;
        if !rule.info.allowed(numerics) {
            return None;
        }
        let rewrite = (rule.apply)(&scope, site, numerics)?;
        scope = scope.checked(&rewrite).ok()?;
        outcome.accepted.push(Accepted { rule: rule.info, before: None, after: None, rejected: Vec::new(), cached: true });
        outcome.applied.push((id, site));
    }
    outcome.scope = scope;
    Some(outcome)
}

impl Planned {
    /// The steps of the plan the scope `scope` is, from the root through the producers a rewrite redirected accesses to; `None`
    /// if no rewrite was applied.
    pub(super) fn of(scope: &Scope, applied: &[(RuleId, SiteSig)]) -> Option<Planned> {
        let &(last, _) = applied.last()?;
        let mut steps: SmallVec<[PlannedStep; 4]> = SmallVec::new();
        let mut order: SmallVec<[CompId; 4]> = SmallVec::new();
        place(scope, scope.root().id, last, &mut steps, &mut order);
        Some(Planned { steps, applied: applied.iter().copied().collect() })
    }
}

/// Appends the steps `id` needs (depth first over the planned edges), then its own, to `steps`; the index of its step.
fn place(scope: &Scope, id: CompId, rule: RuleId, steps: &mut SmallVec<[PlannedStep; 4]>, order: &mut SmallVec<[CompId; 4]>) -> u8 {
    if let Some(at) = order.iter().position(|placed| *placed == id) {
        return at as u8;
    }
    let comp = scope.comp(id).expect("an edge names a comprehension of the scope");
    let mut feeds: SmallVec<[(u8, u8); 2]> = SmallVec::new();
    for edge in scope.edges.iter().filter(|edge| edge.consumer == id && edge.planned) {
        feeds.push((edge.access, place(scope, edge.producer, rule, steps, order)));
    }
    // An unchanged root runs through the strategy of its tag; a fed one lowers its form, which then has no API tag.
    let form = if comp.role == ScopeRole::Root && feeds.is_empty() && matches!(comp.form.origin, Origin::Api(_)) {
        None
    } else {
        let mut form = comp.form.clone();
        if matches!(form.origin, Origin::Api(_)) {
            form.origin = Origin::Rewritten(rule);
        }
        Some(form)
    };
    steps.push(PlannedStep { form, feeds });
    order.push(id);
    (steps.len() - 1) as u8
}

/// Plans the forcing of `root`: builds its scope, runs the normaliser (or replays the program the memo holds for a scope of this
/// structure) and returns the plan if any rewrite was accepted. `None`: nothing to do, or nothing could be planned.
pub(crate) fn plan(ctx: &Context, root: &Structured) -> Option<Arc<Planned>> {
    if let Some(planned) = root.planned.get() {
        return planned.clone();
    }
    let planned = plan_with(ctx, root, T2_RULES, ctx.numerics_ref(), Algebra::current(), None);
    root.planned.get_or_init(|| planned).clone()
}

/// [`plan`] with these rules (tests).
pub(super) fn plan_with(ctx: &Context, root: &Structured, rules: &[T2Rule], numerics: &Numerics, mode: Algebra, cost: Option<&mut dyn FnMut(&Scope) -> Measure>) -> Option<Arc<Planned>> {
    let scope = match Scope::of(ctx, root) {
        Ok(scope) => scope,
        Err(NoScope::TooLarge) => {
            if ctx.jit().recording_decisions() {
                ctx.jit().note_decision(Decision::new(root.name, root.location, DecisionSite::Rewrite, "none").candidate("scope", Verdict::Rejected("scope too large")));
            }
            return None;
        }
        Err(NoScope::Unavailable) => return None,
    };
    debug_assert!(scope.comps.len() <= MAX_SCOPE_COMPS && scope.nodes() <= MAX_SCOPE_NODES);
    let models = Models::of(ctx);
    let key = (scope.structure_hash(), super::rules_hash(numerics, mode), cost_key(&models));
    let remembered = crate::gpu::lock(&ctx.jit().rules.memo).get(&key).cloned();
    let mut estimated = |scope: &Scope| measure_with(scope, &models);
    let cost: &mut dyn FnMut(&Scope) -> Measure = match cost {
        Some(cost) => cost,
        None => &mut estimated,
    };
    let mut outcome = remembered.as_ref().and_then(|program| normalise(scope.clone(), rules, numerics, cost, Some(program)));
    if outcome.is_none() {
        outcome = normalise(scope, rules, numerics, cost, None);
        if let Some(fresh) = &outcome {
            let mut memo = crate::gpu::lock(&ctx.jit().rules.memo);
            if memo.len() >= MEMO_ENTRIES {
                memo.clear();
            }
            memo.insert(key, Arc::new(fresh.applied.clone()));
        }
    }
    let outcome = outcome?;
    if outcome.too_large {
        if ctx.jit().recording_decisions() {
            ctx.jit().note_decision(Decision::new(root.name, root.location, DecisionSite::Rewrite, "none").candidate("scope", Verdict::Rejected("scope too large")));
        }
        return None;
    }
    if ctx.jit().recording_decisions() {
        for accepted in &outcome.accepted {
            let mut decision = Decision::new(root.name, root.location, DecisionSite::Rewrite, accepted.rule.name).cached(accepted.cached);
            if let (Some(before), Some(after)) = (accepted.before, accepted.after) {
                decision = decision.candidate("before", Verdict::Estimate { ns: before.nanoseconds() }).candidate("after", Verdict::Estimate { ns: after.nanoseconds() });
            }
            for &(name, why) in &accepted.rejected {
                decision = decision.candidate(name, Verdict::Rejected(why));
            }
            ctx.jit().note_decision(decision);
        }
    }
    Planned::of(&outcome.scope, &outcome.applied).map(Arc::new)
}

/// What of the cost model an estimate depends on: a recalibration that changes it must not replay a decision.
fn cost_key(models: &Models) -> u64 {
    use std::hash::Hasher;
    let mut h = crate::gpu::hash::FxHasher::default();
    let cost = &models.cost;
    for number in [cost.launch_us, cost.bandwidth_gbs, cost.f32_ops_per_ns, cost.df32_factor, cost.f64_factor.unwrap_or(-1.0)] {
        h.write_u64(number.to_bits());
    }
    if let Some(memory) = &models.memory {
        h.write_u64(memory.l2_bytes.unwrap_or(0));
        for number in [memory.l2_gbs, memory.gather_factor, memory.transpose_factor, memory.strided_factor[0], memory.strided_factor[1], memory.strided_factor[2]] {
            h.write_u64(number.to_bits());
        }
    }
    h.finish()
}

/// Lowers the steps of a plan in order and returns the root's buffer.
pub(crate) fn run(ctx: &Context, root: &Structured, planned: &Planned) -> Result<Arc<Storage>, Arc<LumaError>> {
    let mut outputs: SmallVec<[Arc<Storage>; 4]> = SmallVec::new();
    for step in &planned.steps {
        let output = match &step.form {
            None => match &root.comp().run {
                crate::jit::structured::RunState::Reduce { .. } => select::execute(ctx, &Subject::Comp(root)),
                // A lazy product keeps its operands until it is recorded, which is where its plan is made: an unchanged root has no step.
                crate::jit::structured::RunState::Contraction(_) => Err(crate::array::Failure::from(LumaError::Unsupported("a product runs through its own strategy".into()))),
            },
            Some(form) => {
                let mut form = form.clone();
                for &(access, from) in &step.feeds {
                    let storage = Arc::clone(&outputs[usize::from(from)]);
                    form.inputs[usize::from(access)].src = Src::Buffer(storage);
                }
                debug_assert!(form.validate().is_ok(), "a planned step is well formed: {:?}", form.validate());
                run_form(ctx, &form, root.location)
            }
        };
        let storage = output.map(crate::jit::structured::strategy::Output::into_storage).map_err(|failure| Arc::new(failure.into_error()))?;
        outputs.push(storage);
    }
    outputs.pop().ok_or_else(|| Arc::new(LumaError::Unsupported("an empty plan".into())))
}

/// One form of a plan: the strategies lower it without an epilogue that a rewrite moved into it (rule 8), which then runs over the
/// output as one elementwise pass (the backend-neutral way of W3-U4 §A: the same operations, on the same combined values).
fn run_form(ctx: &Context, form: &Comprehension, location: &'static std::panic::Location<'static>) -> Result<crate::jit::structured::strategy::Output, crate::array::Failure> {
    use crate::jit::comprehension::lower::output_dims;
    use crate::jit::comprehension::Epilogue;
    use crate::jit::structured::epilogue::EpiMap;
    use crate::jit::structured::strategy::Output;
    let Epilogue::Program(program) = &form.epilogue else { return select::execute(ctx, &Subject::Form(form, location)) };
    if !super::scope::moved_into(form, program) {
        return select::execute(ctx, &Subject::Form(form, location));
    }
    let mut plain = form.clone();
    plain.epilogue = Epilogue::None;
    let storage = select::execute(ctx, &Subject::Form(&plain, location))?.into_storage();
    let dims = output_dims(form);
    let shape = crate::shape::Shape::from(dims);
    let arrays: SmallVec<[crate::array::Array<f32>; 3]> = program
        .inputs
        .iter()
        .map(|input| {
            let own = crate::shape::Shape::from(match input.map {
                EpiMap::Point => dims,
                EpiMap::AlongCols => [dims[0], 1, 1, 1],
                EpiMap::AlongRows => [1, dims[1], 1, 1],
                EpiMap::Scalar => [1; 4],
            });
            match &input.src {
                Src::Node(node) => crate::array::Array::from_node(ctx, own, Arc::clone(node)),
                Src::Buffer(storage) => crate::array::Array::from_storage(ctx, own, Arc::clone(storage)),
                Src::Scalar(_) => unreachable!("an epilogue input is an array"),
            }
        })
        .collect();
    crate::jit::structured::lazy::apply_separately(ctx, program, &arrays, Src::Buffer(storage), shape, location).map(Output::Storage)
}
