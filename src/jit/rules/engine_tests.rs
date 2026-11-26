//! The engine (hook of tier 1 for comprehensions, scope, measure, normaliser of tier 2) over rules of its own: the production
//! tables are empty, so every rule here is a toy that moves a form along a made-up cost, and none is in `RULES`.

use std::panic::Location;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, LazyLock};

use luma_core::{BackendKind, DeviceKind};
use proptest::prelude::*;
use smallvec::{SmallVec, smallvec};

use super::measure::{Measure, measure};
use super::registry::{RuleId, RuleInfo, Tier, rules_hash};
use super::scope::{CompId, MAX_SCOPE_COMPS, MAX_SCOPE_NODES, NoScope, Scope, ScopeRole, Site, SiteKind};
use super::t1::{T1Rule, check_rewrite, run as run_t1, test_id};
use super::t2::{Feed, MAX_ITERS, Planned, Rewrite, RulesState, T2Rule, normalise, plan_with, run as run_planned};
use super::*;
use crate::array::Array;
use crate::builtin::{ReduceList, ReduceOp};
use crate::context::{Context, ContextOptions};
use crate::jit::comprehension::{Algebra, Comprehension, Compensation, Origin};
use crate::jit::ElemTy;
use crate::numerics::Numerics;
use crate::profile::{DecisionSite, Verdict};
use crate::reduce::AxisGeom;

fn cpu() -> Context {
    let options = ContextOptions { backend: BackendKind::Cpu, device: DeviceKind::Cpu, ..ContextOptions::from_env() }.calibrate(false);
    Context::with_tuning(options, |tuning| tuning.reduce_multi_after = 0).unwrap()
}

static CTX: LazyLock<Context> = LazyLock::new(cpu);

const fn info(n: u16, name: &'static str, class: Class, dominates: bool) -> RuleInfo {
    RuleInfo { id: RuleId(RuleId::TEST_BASE + n), name, class, gate: class, tier: Tier::T2, dominates, harness: "", order: n }
}

static HALVE_EXACT: RuleInfo = info(0, "halve_exact", Class::Exact, false);
static HALVE_EF: RuleInfo = info(1, "halve_ef", Class::ExactFinite, false);
static HALVE_RA: RuleInfo = info(2, "halve_ra", Class::Reassociating, false);
static HALVE_RF: RuleInfo = info(3, "halve_rf", Class::Reformulating, false);
static FLIP_A: RuleInfo = info(4, "flip_a", Class::Exact, false);
static FLIP_B: RuleInfo = info(5, "flip_b", Class::Exact, false);
static RAISE: RuleInfo = info(6, "raise", Class::Exact, false);
static DROP_PARTNER: RuleInfo = info(7, "drop_partner", Class::Exact, true);
static GENERIC: RuleInfo = info(8, "generic", Class::Exact, false);

/// The cost of a toy form: its `rank` (a made-up number of nanoseconds), 1000 where it has none.
fn rank(form: &Comprehension) -> u32 {
    form.props.rank.unwrap_or(1000)
}

fn halve(scope: &Scope, site: Site, _: &Numerics) -> Option<Rewrite> {
    let Site::Comp(id) = site else { return None };
    let form = &scope.comp(id)?.form;
    let r = rank(form);
    if r <= 10 {
        return None;
    }
    let mut form = form.clone();
    form.props.rank = Some(r / 2);
    Some(Rewrite { replace: smallvec![(id, form)], ..Rewrite::default() })
}

fn flip_a(scope: &Scope, site: Site, _: &Numerics) -> Option<Rewrite> {
    let Site::Comp(id) = site else { return None };
    let mut form = scope.comp(id)?.form.clone();
    if form.props.band.is_some() {
        return None;
    }
    form.props.band = Some((1, 1));
    Some(Rewrite { replace: smallvec![(id, form)], ..Rewrite::default() })
}

fn flip_b(scope: &Scope, site: Site, _: &Numerics) -> Option<Rewrite> {
    let Site::Comp(id) = site else { return None };
    let mut form = scope.comp(id)?.form.clone();
    form.props.band.take()?;
    Some(Rewrite { replace: smallvec![(id, form)], ..Rewrite::default() })
}

fn raise(scope: &Scope, site: Site, _: &Numerics) -> Option<Rewrite> {
    let Site::Comp(id) = site else { return None };
    let mut form = scope.comp(id)?.form.clone();
    let r = rank(&form);
    if r >= 100_000 {
        return None;
    }
    form.props.rank = Some(r * 2);
    Some(Rewrite { replace: smallvec![(id, form)], ..Rewrite::default() })
}

fn drop_partner(scope: &Scope, site: Site, _: &Numerics) -> Option<Rewrite> {
    let Site::Comp(id) = site else { return None };
    (scope.comp(id)?.role == ScopeRole::Partner).then(|| Rewrite { drop: smallvec![id], ..Rewrite::default() })
}

/// The root's API form becomes the same form as a generic one: the value is the same, the cost made up is lower.
fn generic(scope: &Scope, site: Site, _: &Numerics) -> Option<Rewrite> {
    let Site::Comp(id) = site else { return None };
    let comp = scope.comp(id)?;
    if comp.role != ScopeRole::Root || !matches!(comp.form.origin, Origin::Api(_)) {
        return None;
    }
    let mut form = comp.form.clone();
    form.origin = Origin::Rewritten(GENERIC.id);
    Some(Rewrite { replace: smallvec![(id, form)], ..Rewrite::default() })
}

const fn rule(info: &'static RuleInfo, apply: fn(&Scope, Site, &Numerics) -> Option<Rewrite>) -> T2Rule {
    T2Rule { info, sites: SiteKind::Comp, single: true, apply }
}

/// Every toy rule, sorted by (gate rank, order) as the engine wants.
fn toy_rules() -> Vec<T2Rule> {
    let mut all = vec![
        rule(&HALVE_EXACT, halve),
        rule(&HALVE_EF, halve),
        rule(&HALVE_RA, halve),
        rule(&HALVE_RF, halve),
        rule(&FLIP_A, flip_a),
        rule(&FLIP_B, flip_b),
        rule(&RAISE, raise),
        rule(&DROP_PARTNER, drop_partner),
    ];
    all.sort_by_key(|r| (r.info.gate.rank(), r.info.order));
    all
}

fn toy_cost(scope: &Scope) -> Measure {
    Measure { ns: scope.comps.iter().map(|comp| u64::from(rank(&comp.form)) * 1024).sum(), nodes: scope.nodes() }
}

/// A form of a whole reduction of a buffer of `len` elements.
fn sum_form(ctx: &Context, len: usize) -> Comprehension {
    let a = Array::from_slice_on(ctx, &vec![1.0f32; len], len).unwrap();
    Comprehension::reduce(a.src().ok().unwrap(), len, ReduceList::new(&[ReduceOp::Sum]), ElemTy::F32)
}

fn toy_scope(ctx: &Context, ranks: &[Option<u32>]) -> Scope {
    let forms = ranks.iter().enumerate().map(|(i, r)| {
        let mut form = sum_form(ctx, 8 + i);
        form.props.rank = *r;
        form
    });
    Scope::from_forms(forms, 0)
}

fn run(scope: Scope, rules: &[T2Rule], n: &Numerics) -> super::t2::Outcome {
    normalise(scope, rules, n, &mut |s| toy_cost(s), None).expect("a fresh run")
}

fn names(outcome: &super::t2::Outcome) -> Vec<&'static str> {
    outcome.accepted.iter().map(|a| a.rule.name).collect()
}

// -- the normaliser --

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// Any scope of up to 12 comprehensions, any subset of the toy rules (an A to B and B to A pair and a rule that raises the cost
    /// among them), a cost that is a function of the scope but not monotonic: the run ends before the hard stop, every accepted
    /// step improves the measure, and what is left is a fixpoint.
    #[test]
    fn the_normaliser_terminates_improving_to_a_fixpoint(
        ranks in prop::collection::vec(prop::option::of(1u32..3000), 1..13),
        picked in prop::collection::vec(any::<bool>(), 8),
        scramble in 0u32..4,
        links in prop::collection::vec((0usize..12, 0usize..12), 0..6),
    ) {
        let ctx = &*CTX;
        let rules: Vec<T2Rule> = toy_rules().into_iter().zip(picked).filter(|(_, on)| *on).map(|(r, _)| r).collect();
        let mut scope = toy_scope(ctx, &ranks);
        let count = ranks.len();
        for (a, b) in links {
            if a < b && b < count {
                scope.link(CompId(a as u32), 0, CompId(b as u32));
            }
        }
        // Not monotonic in the rank, so acceptance by the measure is what bounds the run.
        let cost = move |s: &Scope| {
            let ns = s.comps.iter().map(|c| u64::from((rank(&c.form).wrapping_mul(37).wrapping_add(scramble)) % 1000 + 1) * 1024).sum();
            Measure { ns, nodes: s.nodes() }
        };
        let numerics = Numerics::algebraic();
        let outcome = normalise(scope, &rules, &numerics, &mut { cost }, None).unwrap();
        prop_assert!(!outcome.capped && outcome.applied.len() <= MAX_ITERS);
        for step in &outcome.accepted {
            if let (Some(before), Some(after)) = (step.before, step.after) {
                prop_assert!(after.improves(before), "{}: {before:?} to {after:?}", step.rule.name);
            }
        }
        // A fixpoint: from where it ended no rule applies and is accepted.
        let again = normalise(outcome.scope.clone(), &rules, &numerics, &mut { cost }, None).unwrap();
        prop_assert!(again.applied.is_empty(), "not a fixpoint: {:?}", names(&again));
    }
}

#[test]
fn a_pair_that_undoes_itself_and_a_rule_that_raises_the_cost_are_never_taken() {
    let ctx = &*CTX;
    let rules = [rule(&FLIP_A, flip_a), rule(&FLIP_B, flip_b), rule(&RAISE, raise)];
    let outcome = run(toy_scope(ctx, &[Some(40)]), &rules, &Numerics::default());
    assert!(outcome.applied.is_empty() && !outcome.capped, "{:?}", names(&outcome));
}

#[test]
fn the_result_is_a_function_of_the_scope() {
    let ctx = &*CTX;
    let rules = toy_rules();
    let make = || toy_scope(ctx, &[Some(900), None, Some(70), Some(333)]);
    let first = run(make(), &rules, &Numerics::algebraic());
    assert!(!first.applied.is_empty());
    for _ in 0..20 {
        assert_eq!(run(make(), &rules, &Numerics::algebraic()).applied, first.applied);
    }
    std::thread::scope(|threads| {
        let handles: Vec<_> = (0..8).map(|_| threads.spawn(|| run(toy_scope(&CTX, &[Some(900), None, Some(70), Some(333)]), &toy_rules(), &Numerics::algebraic()).applied)).collect();
        for handle in handles {
            assert_eq!(handle.join().unwrap(), first.applied);
        }
    });
}

#[test]
fn a_rule_runs_only_where_its_class_is_allowed() {
    let ctx = &*CTX;
    for (info, flag) in [(&HALVE_EF, 1u8), (&HALVE_RA, 2), (&HALVE_RF, 4), (&HALVE_EXACT, 0)] {
        let rules = [rule(info, halve)];
        for bits in 0..8u8 {
            let n = Numerics { exact_finite: bits & 1 != 0, reassociate: bits & 2 != 0, reformulate: bits & 4 != 0, ..Numerics::strict() };
            let fired = !run(toy_scope(ctx, &[None]), &rules, &n).applied.is_empty();
            assert_eq!(fired, flag == 0 || bits & flag != 0, "{} under {n}", info.name);
        }
        assert!(!run(toy_scope(ctx, &[None]), &rules, &Numerics::algebraic()).applied.is_empty(), "{} under algebraic", info.name);
        assert_eq!(!run(toy_scope(ctx, &[None]), &rules, &Numerics::default()).applied.is_empty(), flag == 0, "{} under default", info.name);
    }
    // The exact rules go first.
    let outcome = run(toy_scope(ctx, &[None]), &toy_rules(), &Numerics::algebraic());
    assert_eq!(names(&outcome)[0], "halve_exact");
}

#[test]
fn a_scope_over_a_bound_is_left_alone() {
    let ctx = &*CTX;
    let rules = [rule(&DROP_PARTNER, drop_partner)];
    let wide = toy_scope(ctx, &vec![None; MAX_SCOPE_COMPS + 1]);
    let outcome = run(wide, &rules, &Numerics::algebraic());
    assert!(outcome.too_large && outcome.applied.is_empty());
    let deep = Scope::from_forms([sum_form(ctx, 8)], MAX_SCOPE_NODES);
    assert!(run(deep, &rules, &Numerics::algebraic()).too_large);
    let at_the_bound = toy_scope(ctx, &vec![None; MAX_SCOPE_COMPS]);
    assert!(!run(at_the_bound, &rules, &Numerics::algebraic()).too_large);
}

#[test]
fn a_rule_that_dominates_is_accepted_on_the_node_count_and_calls_no_cost() {
    let ctx = &*CTX;
    let calls = AtomicUsize::new(0);
    let rules = [rule(&DROP_PARTNER, drop_partner)];
    let outcome = normalise(toy_scope(ctx, &[None, None, None]), &rules, &Numerics::default(), &mut |s| {
        calls.fetch_add(1, Ordering::Relaxed);
        toy_cost(s)
    }, None)
    .unwrap();
    assert_eq!(outcome.scope.comps.len(), 1);
    assert_eq!(names(&outcome), ["drop_partner", "drop_partner"]);
    assert_eq!(calls.load(Ordering::Relaxed), 0, "no estimate for a rule that dominates");
    assert!(outcome.accepted.iter().all(|a| a.before.is_none() && a.after.is_none()));
}

#[test]
fn a_rewrite_that_is_not_valid_is_dropped_and_noted() {
    let ctx = &*CTX;
    // Changes the result type: not a rewrite of the same value.
    fn retype(scope: &Scope, site: Site, _: &Numerics) -> Option<Rewrite> {
        let Site::Comp(id) = site else { return None };
        let mut form = scope.comp(id)?.form.clone();
        (form.ty == ElemTy::F32).then(|| {
            form.ty = ElemTy::F64;
            Rewrite { replace: smallvec![(id, form)], ..Rewrite::default() }
        })
    }
    // Drops the root.
    fn drop_root(scope: &Scope, site: Site, _: &Numerics) -> Option<Rewrite> {
        let Site::Comp(id) = site else { return None };
        (scope.comp(id)?.role == ScopeRole::Root).then(|| Rewrite { drop: smallvec![id], ..Rewrite::default() })
    }
    static RETYPE: RuleInfo = info(20, "retype", Class::Exact, false);
    static DROP_ROOT: RuleInfo = info(21, "drop_root", Class::Exact, true);
    let rules = [rule(&RETYPE, retype), rule(&DROP_ROOT, drop_root), rule(&HALVE_EXACT, halve)];
    let outcome = run(toy_scope(ctx, &[None]), &rules, &Numerics::default());
    assert_eq!(outcome.scope.comps.len(), 1);
    let first = &outcome.accepted[0];
    assert_eq!(first.rule.name, "halve_exact");
    assert_eq!(first.rejected, [("retype", "invalid"), ("drop_root", "invalid")]);
}

// -- the scope and the plan of a real value --

fn data(ctx: &Context, len: usize) -> Array<f32> {
    Array::from_slice_on(ctx, &(0..len).map(|i| (i % 5) as f32).collect::<Vec<_>>(), len).unwrap()
}

/// A context that leaves axis reductions pending (a `max_axis` of a big array is a structured value that other expressions read through
/// a leaf).
fn deferring() -> Context {
    let options = ContextOptions { backend: BackendKind::Cpu, device: DeviceKind::Cpu, ..ContextOptions::from_env() }.calibrate(false);
    Context::with_tuning(options, |tuning| {
        tuning.reduce_multi_after = 0;
        tuning.axis_defer_cells = 0;
        tuning.axis_defer_weight = 0;
    })
    .unwrap()
}

fn volume(ctx: &Context) -> Array<f32> {
    Array::from_slice_on(ctx, &vec![1.0f32; 300 * 400 * 2], [300, 400, 2]).unwrap()
}

/// Lowers the root's own form as the toy `generic` rule rewrites it: a made-up cost where the rewritten form is cheaper.
fn generic_cost(scope: &Scope) -> Measure {
    let ns = scope.comps.iter().map(|c| if matches!(c.form.origin, Origin::Api(_)) { 1000u64 } else { 10 } * 1024).sum();
    Measure { ns, nodes: scope.nodes() }
}

#[test]
fn a_planned_root_gives_the_same_result_and_leaves_the_value_alone() {
    let ctx = cpu();
    let x = data(&ctx, 1000);
    let sum = (&x * 2.0f32).sum();
    let value = sum.structured().expect("a reduction of a lazy array is a structured value");
    let before = value.form(&ctx).unwrap();
    let rules = [rule(&GENERIC, generic)];
    let planned = plan_with(&ctx, value, &rules, &Numerics::default(), Algebra::On, Some(&mut generic_cost)).expect("the generic form is cheaper");
    assert_eq!(planned.applied.len(), 1);
    let [step] = &planned.steps[..] else { panic!("one step: {planned:?}") };
    assert!(matches!(step.form.as_ref().unwrap().origin, Origin::Rewritten(id) if id == GENERIC.id));
    let storage = run_planned(&ctx, value, &planned).unwrap();
    let got = ctx.read_storage::<f32>(&storage).unwrap();
    // The original is untouched: nothing was forced, and its form is what it was.
    assert!(value.is_unforced() && value.result_if_recorded().is_none());
    let after = value.form(&ctx).unwrap();
    assert_eq!((after.structure_hash(), &after.origin), (before.structure_hash(), &before.origin));
    let want: f32 = (0..1000).map(|i| ((i % 5) * 2) as f32).sum();
    assert_eq!(got, [want]);
    assert_eq!(sum.item().unwrap(), want);
}

#[test]
fn the_second_force_of_a_structure_replays_the_first_and_a_stale_program_runs_again() {
    let ctx = cpu();
    ctx.record_decisions(true);
    let rules = [rule(&GENERIC, generic)];
    let make = |len| {
        let x = data(&ctx, len);
        (&x * 2.0f32).sum()
    };
    let plan = |array: &Array<f32>, rules: &[T2Rule]| plan_with(&ctx, array.structured().unwrap(), rules, &Numerics::default(), Algebra::On, Some(&mut generic_cost));
    // One value at a time: the others that are pending would be partners and change the structure of the scope.
    let first = plan(&make(1000), &rules).unwrap();
    let second = plan(&make(1000), &rules).unwrap();
    assert_eq!(first.applied.len(), second.applied.len());
    let log: Vec<_> = ctx.take_decisions().into_iter().filter(|d| d.site == DecisionSite::Rewrite).collect();
    assert_eq!(log.len(), 2);
    assert_eq!((log[0].chosen.as_ref(), log[0].cached), ("generic", false));
    assert_eq!((log[1].chosen.as_ref(), log[1].cached), ("generic", true), "the memo replays it");
    assert!(log[0].candidates.iter().any(|(label, v)| label == "before" && matches!(v, Verdict::Estimate { .. })));
    // A rule that no longer applies where it was recorded: the replay fails and the structure is planned afresh.
    static LIVE: AtomicBool = AtomicBool::new(true);
    fn flaky(scope: &Scope, site: Site, n: &Numerics) -> Option<Rewrite> {
        if LIVE.load(Ordering::Relaxed) { generic(scope, site, n) } else { None }
    }
    let flaky_rules = [rule(&GENERIC, flaky)];
    assert!(plan(&make(1000), &flaky_rules).is_some());
    LIVE.store(false, Ordering::Relaxed);
    assert!(plan(&make(1000), &flaky_rules).is_none(), "the recorded rewrite no longer applies and nothing else does");
    let log: Vec<_> = ctx.take_decisions().into_iter().filter(|d| d.site == DecisionSite::Rewrite).collect();
    assert_eq!(log.len(), 1, "the failed replay logs nothing");
    assert!(log[0].cached, "the recorded program was replayed for the first of the two");
}

#[test]
fn a_scope_over_the_bound_records_that_it_is_too_large() {
    let ctx = deferring();
    ctx.record_decisions(true);
    let x = volume(&ctx);
    // One root over 40 deferred reductions: 41 comprehensions.
    let mut acc = &x.max_axis(1) * 0.5f32;
    for k in 1..(MAX_SCOPE_COMPS + 8) {
        acc = &acc + &(&(&x * (k as f32)).max_axis(1) * 0.5f32);
    }
    let root = acc.sum();
    let value = root.structured().unwrap();
    assert_eq!(Scope::of(&ctx, value).err(), Some(NoScope::TooLarge));
    let rules = toy_rules();
    assert!(plan_with(&ctx, value, &rules, &Numerics::algebraic(), Algebra::On, Some(&mut toy_cost)).is_none());
    let log = ctx.take_decisions();
    let log: Vec<_> = log.iter().filter(|d| d.site == DecisionSite::Rewrite).collect();
    assert_eq!(log.len(), 1);
    assert!(log[0].candidates.iter().any(|(_, v)| *v == Verdict::Rejected("scope too large")));
}

#[test]
fn a_producer_of_the_same_thread_is_in_the_scope_and_one_of_another_thread_or_a_recording_is_not() {
    let ctx = deferring();
    let x = volume(&ctx);
    let near = x.max_axis(1);
    let root = (&near * 0.5f32).sum();
    let scope = Scope::of(&ctx, root.structured().unwrap()).unwrap();
    assert_eq!(scope.comps.iter().map(|c| c.role).collect::<Vec<_>>(), [ScopeRole::Root, ScopeRole::Producer]);
    assert_eq!(scope.edges.len(), 1);
    assert_eq!((scope.edges[0].consumer, scope.edges[0].producer, scope.edges[0].access, scope.edges[0].direct), (CompId(0), CompId(1), 0, false));
    assert!(scope.sites(SiteKind::Edge).len() == 1 && scope.sites(SiteKind::Pair).len() == 1 && scope.sites(SiteKind::Comp).len() == 2);

    let far = std::thread::scope(|threads| threads.spawn(|| x.max_axis(1)).join().unwrap());
    assert!(far.structured().is_some_and(|v| v.is_unforced()));
    let root = (&far * 0.5f32).sum();
    let scope = Scope::of(&ctx, root.structured().unwrap()).unwrap();
    assert!(scope.comps.iter().all(|c| c.role != ScopeRole::Producer), "a value another thread made is not planned here");

    // A value made in a recording is never part of a scope, as a root or as a producer (axis reductions are not deferred in a
    // recording, so a whole reduction stands for it).
    assert!(ctx.begin_recording(Vec::new()));
    let inside = (&x * 2.0f32).sum();
    assert!(ctx.end_recording().is_some());
    let value = inside.structured().expect("a pending reduction");
    assert!(value.in_recording);
    assert_eq!(Scope::of(&ctx, value).err(), Some(NoScope::Unavailable));
    assert!(!super::scope::plannable(value), "nor as a producer");

    let alone = (&x * 2.0f32).sum();
    let single = Scope::of(&ctx, alone.structured().unwrap()).unwrap();
    assert_eq!(single.edges.len(), 0);
}

#[test]
fn applying_a_rewrite_changes_a_clone_and_a_feed_makes_a_step() {
    let ctx = cpu();
    let mut scope = toy_scope(&ctx, &[None, None]);
    scope.link(CompId(0), 0, CompId(1));
    // The new form reads what the scope already reads.
    let add = {
        let mut form = scope.comp(CompId(1)).unwrap().form.clone();
        form.origin = Origin::Rewritten(GENERIC.id);
        form
    };
    let fed = Rewrite {
        add: smallvec![add],
        feeds: smallvec![Feed { consumer: CompId(0), access: 0, producer: scope.next_id() }],
        ..Rewrite::default()
    };
    let next = scope.checked(&fed).unwrap();
    assert_eq!((scope.comps.len(), next.comps.len()), (2, 3), "the original scope is a clone away");
    assert!(next.edges.iter().any(|e| e.planned && e.producer == CompId(2)) && !next.edges.iter().any(|e| e.consumer == CompId(0) && !e.planned));
    let planned = Planned::of(&next, &[(GENERIC.id, Site::Comp(CompId(0)))]).unwrap();
    assert_eq!(planned.steps.len(), 2);
    assert_eq!(planned.steps[1].feeds.as_slice(), [(0u8, 0u8)]);
    assert!(planned.steps[0].form.is_some() && planned.steps[1].form.is_some());
    // A feed that makes a cycle is not valid.
    let cyclic = Rewrite { feeds: smallvec![Feed { consumer: CompId(1), access: 0, producer: CompId(0) }], ..Rewrite::default() };
    assert_eq!(scope.checked(&cyclic).err(), Some("invalid"));
}

// -- the cost --

#[test]
fn the_measure_orders_forms_by_work_and_traffic() {
    let ctx = cpu();
    let small = Scope::from_forms([sum_form(&ctx, 1 << 10)], 0);
    let big = Scope::from_forms([sum_form(&ctx, 1 << 24)], 0);
    let (a, b) = (measure(&small, &ctx), measure(&big, &ctx));
    assert!(a.ns < b.ns && a.nodes == b.nodes);
    let two = Scope::from_forms([sum_form(&ctx, 1 << 20), sum_form(&ctx, 1 << 20)], 0);
    let one = Scope::from_forms([sum_form(&ctx, 1 << 20)], 0);
    assert!(measure(&two, &ctx).ns > measure(&one, &ctx).ns && measure(&two, &ctx).nodes == 2);
    // An axis reduction's compulsory traffic is its source, once.
    let geom = AxisGeom { n: 64, stride: 1, lines: 64 };
    let x = Array::from_slice_on(&ctx, &vec![1.0f32; 64 * 64], [64, 64]).unwrap();
    let axis = Comprehension::reduce_axis(x.src().ok().unwrap(), geom, ReduceOp::Sum, ElemTy::F32);
    assert!(measure(&Scope::from_forms([axis], 0), &ctx).ns > 0);
    // Accepting: at least 1/64 faster, or no slower with fewer nodes.
    let m = |ns, nodes| Measure { ns, nodes };
    assert!(m(6400 - 100, 5).improves(m(6400, 5)));
    assert!(!m(6400 - 99, 5).improves(m(6400, 5)));
    assert!(m(6400, 4).improves(m(6400, 5)) && !m(6401, 4).improves(m(6400, 5)) && !m(6400, 5).improves(m(6400, 5)));
}

// -- the hook of tier 1 --

/// Makes a form that validates and that no lowering takes: a compensated sum along lines.
fn not_lowerable(form: &Comprehension, _: &Numerics) -> Option<Comprehension> {
    let mut form = form.clone();
    let combiner = form.combiner.as_mut()?;
    if combiner.compensation != Compensation::None {
        return None;
    }
    combiner.compensation = Compensation::Neumaier;
    Some(form)
}

fn tag_props(form: &Comprehension, _: &Numerics) -> Option<Comprehension> {
    if form.props.rank.is_some() {
        return None;
    }
    let mut form = form.clone();
    form.props.rank = Some(1);
    Some(form)
}

#[test]
fn a_form_that_cannot_be_lowered_is_dropped_with_a_rejected_candidate() {
    let ctx = cpu();
    ctx.record_decisions(true);
    static BAD: RuleInfo = info(30, "not_lowerable", Class::Exact, false);
    static GOOD: RuleInfo = info(31, "tag_props", Class::Exact, false);
    let geom = AxisGeom { n: 16, stride: 1, lines: 4 };
    let x = Array::from_slice_on(&ctx, &vec![1.0f32; 64], [16, 4]).unwrap();
    let form = Comprehension::reduce_axis(x.src().ok().unwrap(), geom, ReduceOp::Sum, ElemTy::F32);
    let rules = [T1Rule { info: &BAD, apply: not_lowerable }, T1Rule { info: &GOOD, apply: tag_props }];
    let out = run_t1(form.clone(), &ctx, &rules, &Numerics::default(), Location::caller());
    assert_eq!(out.props.rank, Some(1), "the next rule was taken");
    assert!(matches!(out.origin, Origin::Rewritten(id) if id == GOOD.id));
    let log = ctx.take_decisions();
    let log: Vec<_> = log.iter().filter(|d| d.site == DecisionSite::Rewrite).collect();
    // Round one takes `tag_props` past the rejected rule; round two finds nothing but the rejected rule again.
    assert_eq!(log.iter().map(|d| d.chosen.as_ref()).collect::<Vec<_>>(), ["tag_props", "none"]);
    assert!(log.iter().all(|d| d.candidates == [("not_lowerable".into(), Verdict::Rejected("not lowerable"))]));
    // Only the bad rule: the form is returned as it was.
    let only_bad = [T1Rule { info: &BAD, apply: not_lowerable }];
    let out = run_t1(form.clone(), &ctx, &only_bad, &Numerics::default(), Location::caller());
    assert_eq!(out.structure_hash(), form.structure_hash());
}

#[test]
fn a_rule_of_tier_one_runs_where_its_class_allows_and_at_most_four_rounds() {
    let ctx = cpu();
    static RA: RuleInfo = info(32, "tag_ra", Class::Reassociating, false);
    static COUNT: AtomicUsize = AtomicUsize::new(0);
    fn spin(form: &Comprehension, _: &Numerics) -> Option<Comprehension> {
        COUNT.fetch_add(1, Ordering::Relaxed);
        let mut form = form.clone();
        form.props.rank = Some(form.props.rank.unwrap_or(0) + 1);
        Some(form)
    }
    let rules = [T1Rule { info: &RA, apply: spin }];
    let form = sum_form(&ctx, 8);
    assert_eq!(run_t1(form.clone(), &ctx, &rules, &Numerics::default(), Location::caller()).props.rank, None);
    assert_eq!(COUNT.load(Ordering::Relaxed), 0);
    let n = Numerics { reassociate: true, ..Numerics::default() };
    assert_eq!(run_t1(form, &ctx, &rules, &n, Location::caller()).props.rank, Some(4), "four rounds, as `create` makes");
    assert_eq!(test_id(0).0, RuleId::TEST_BASE);
}

#[test]
#[should_panic(expected = "output shape")]
fn a_rewrite_that_changes_the_output_shape_trips_the_check() {
    let ctx = cpu();
    let a = Array::from_slice_on(&ctx, &vec![1.0f32; 64], [16, 4]).unwrap();
    let b = Array::from_slice_on(&ctx, &vec![1.0f32; 64], [8, 8]).unwrap();
    let old = Comprehension::reduce_axis(a.src().ok().unwrap(), AxisGeom { n: 16, stride: 1, lines: 4 }, ReduceOp::Sum, ElemTy::F32);
    let new = Comprehension::reduce_axis(b.src().ok().unwrap(), AxisGeom { n: 8, stride: 1, lines: 8 }, ReduceOp::Sum, ElemTy::F32);
    check_rewrite(&old, &new);
}

#[test]
#[should_panic(expected = "result type")]
fn a_rewrite_that_changes_the_type_trips_the_check() {
    let ctx = cpu();
    let old = sum_form(&ctx, 8);
    let mut new = old.clone();
    new.ty = ElemTy::F64;
    check_rewrite(&old, &new);
}

#[test]
#[should_panic(expected = "a source the original did not")]
fn a_rewrite_that_reads_a_new_source_trips_the_check() {
    let ctx = cpu();
    let old = sum_form(&ctx, 8);
    let new = sum_form(&ctx, 8);
    check_rewrite(&old, &new);
}

// -- activation --

#[test]
fn the_engine_is_active_only_with_a_rule_the_numerics_and_the_mode_allow() {
    let t1: &[T1Rule] = &[T1Rule { info: &HALVE_RA, apply: tag_props }];
    let t2 = [rule(&HALVE_RA, halve)];
    let state = |n: &Numerics, mode| {
        let s = RulesState::with(n, mode, t1, &t2);
        (s.t1_active, s.t2_active)
    };
    assert_eq!(state(&Numerics::default(), Algebra::On), (false, false));
    assert_eq!(state(&Numerics::algebraic(), Algebra::On), (true, true));
    assert_eq!(state(&Numerics::algebraic(), Algebra::T1), (true, false));
    assert_eq!(state(&Numerics::algebraic(), Algebra::Api), (false, false));
    assert_eq!(state(&Numerics { reassociate: true, ..Numerics::strict() }, Algebra::On), (true, true));
    assert_eq!(state(&Numerics { reformulate: true, ..Numerics::strict() }, Algebra::On), (false, false));
    assert_ne!(rules_hash(&Numerics::fast(), Algebra::On), rules_hash(&Numerics::fast(), Algebra::T1));
    let none = RulesState::with(&Numerics::fast(), Algebra::On, &[], &[]);
    assert!(!none.t1_active && !none.t2_active);
    let _: SmallVec<[u8; 1]> = SmallVec::new();
    let _: Option<Arc<Planned>> = None;
}
