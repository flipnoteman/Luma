//! The registry against the harness and the dispatcher: every rule of `RULES` has a harness entry of its class and fires, the
//! rules that fire under a set of class flags are exactly those whose gate the flags allow, the hash that stamps a trace or a plan
//! tells every set of flags and every mode apart, and the tables of the tiers the engine adds are empty in production.

use super::harness::{Harness, check_typed, typed_cases};
use super::registry::{RULES, Tier, trace};
use super::table::{Gate, Rule, rules};
use super::*;
use crate::context::{Context, ContextOptions};
use crate::jit::comprehension::Algebra;
use crate::numerics::Numerics;

/// The class flags of subset `bits` (exact finite, reassociate, reformulate), on an exact-`f32` context.
fn subset(bits: u8) -> Numerics {
    Numerics { exact_finite: bits & 1 != 0, reassociate: bits & 2 != 0, reformulate: bits & 4 != 0, ..Numerics::strict() }
}

fn open(numerics: Numerics) -> Context {
    Context::with_options(ContextOptions::from_env().calibrate(false).numerics(numerics)).unwrap()
}

#[test]
fn the_registry_is_sorted_and_its_ids_are_its_indices() {
    for (i, info) in RULES.iter().enumerate() {
        assert_eq!(usize::from(info.id.0), i, "{}", info.name);
        assert_eq!(info.id.info().map(|found| found.name), Some(info.name));
    }
    assert!(RULES.windows(2).all(|pair| pair[0].sort_key() < pair[1].sort_key()), "RULES is sorted by (tier, gate rank, order)");
    let mut names: Vec<_> = RULES.iter().map(|info| info.name).collect();
    names.sort_unstable();
    assert!(names.windows(2).all(|pair| pair[0] != pair[1]), "names are unique");
    assert!(RULES.iter().all(|info| info.tier == Tier::T1Elementwise), "no production rule of the other tiers");
}

#[test]
fn a_rule_runs_under_the_gate_of_its_class_except_the_four_that_keep_theirs() {
    use Class::{Exact, ExactFinite, Reformulating};
    for info in RULES {
        let kept = matches!(info.name, "div_one" | "add_neg_to_sub" | "add_neg_to_sub_left" | "sub_neg_to_add");
        assert_eq!(info.class == info.gate, !kept, "{}: class {:?}, gate {:?}", info.name, info.class, info.gate);
    }
    let named = |name| RULES.iter().find(|info| info.name == name).unwrap();
    // x / 1 is not x on a device that flushes subnormals: Reformulating, but it always ran.
    assert_eq!((named("div_one").class, named("div_one").gate), (Reformulating, Exact));
    // The three that are exact bit for bit, kept behind exact_finite until R4 promotes them.
    for name in ["add_neg_to_sub", "add_neg_to_sub_left", "sub_neg_to_add"] {
        assert_eq!((named(name).class, named(name).gate), (Exact, ExactFinite), "{name}");
    }
    // B6: not bit-equal on finite inputs (`-0 + 0`, `-x * 0`, `0 / 0`).
    for name in ["add_zero_float", "mul_zero_float", "div_same_float"] {
        assert_eq!(named(name).class, Reformulating, "{name}");
    }
}

#[test]
fn class_gating_follows_the_flags() {
    let all: Vec<Numerics> = (0..8).map(subset).collect();
    for class in [Class::Exact, Class::ExactFinite, Class::Reassociating, Class::Reformulating] {
        assert!(class.allowed(&Numerics::fast()) && class.allowed(&Numerics::algebraic()));
        for n in &all {
            let want = match class {
                Class::Exact => true,
                Class::ExactFinite => n.exact_finite,
                Class::Reassociating => n.reassociate,
                Class::Reformulating => n.reformulate,
            };
            assert_eq!(class.allowed(n), want);
        }
    }
    // default and strict run only the exact rules.
    for preset in [Numerics::default(), Numerics::strict()] {
        assert!(RULES.iter().all(|info| info.allowed(&preset) == (info.gate == Class::Exact)));
    }
    assert!(RULES.iter().all(|info| info.allowed(&Numerics::algebraic()) && info.allowed(&Numerics::fast())));
    let ranks: Vec<u8> = [Class::Exact, Class::ExactFinite, Class::Reassociating, Class::Reformulating].iter().map(|c| c.rank()).collect();
    assert_eq!(ranks, [0, 1, 2, 3]);
}

/// The rule a harness name stands for: a `Rule` of the table, a typed case (always `Exact`) or a unit test of `simplify.rs`.
enum Found {
    Rule(Class),
    Typed,
    Unit,
}

fn find(name: &str, table: &[Rule], typed: &[super::harness::Typed]) -> Option<Found> {
    if let Some(test) = name.strip_prefix("unit:") {
        return include_str!("../simplify.rs").contains(&format!("fn {test}()")).then_some(Found::Unit);
    }
    table.iter().find(|rule| rule.name == name).map(|rule| Found::Rule(rule.class)).or_else(|| typed.iter().find(|case| case.name == name && case.rule).map(|_| Found::Typed))
}

#[test]
fn every_registered_rule_has_a_harness_rule_of_its_class() {
    let (table, typed) = (rules(), typed_cases());
    for info in RULES {
        match find(info.harness, &table, &typed) {
            None => panic!("{}: no harness entry named {:?}", info.name, info.harness),
            Some(Found::Rule(class)) => assert_eq!(class, info.class, "{}: the harness rule {:?} has another class", info.name, info.harness),
            Some(Found::Typed) => assert_eq!(info.class, Class::Exact, "{}: a typed case is an exact rule", info.name),
            Some(Found::Unit) => {}
        }
    }
    // The other way round: a rule of the table that `simplify.rs` implements is named by a registered rule or fires one of its class.
    for rule in table.iter().filter(|rule| rule.gate != Gate::None) {
        assert!(RULES.iter().any(|info| info.class == rule.class), "{}: no registered rule has class {:?}", rule.name, rule.class);
    }
}

/// Runs the lhs of every rule of the table that `simplify.rs` implements, and every typed case that is a rule, in `ctx`, and
/// returns for each the rules of `RULES` that fired.
fn firings(h: &Harness, ctx: &Context) -> Vec<(&'static str, Vec<RuleId>)> {
    let mut out = Vec::new();
    for rule in rules().iter().filter(|rule| rule.gate != Gate::None) {
        trace::clear();
        h.run_lhs(ctx, rule);
        out.push((rule.name, trace::take().into_iter().map(|(id, _)| id).collect()));
    }
    for case in typed_cases().iter().filter(|case| case.rule) {
        trace::clear();
        let _ = (case.run)(ctx);
        out.push((case.name, trace::take().into_iter().map(|(id, _)| id).collect()));
    }
    out
}

/// Fires the two rules whose operand is a scalar a program cannot build from an `Array`, by the unit tests' calls.
fn scalar_rules(ctx: &Context, n: &Numerics) -> Vec<RuleId> {
    use crate::jit::{Args, BinOp, ElemTy, Op, OpKind, Scalar, Src, UnOp};
    let u = crate::array::Array::from_slice_on(ctx, &[1u32, 2, 3], 3).unwrap();
    let f = crate::array::Array::from_slice_on(ctx, &[1.0f32, 2.0, 3.0], 3).unwrap();
    let buffer = |storage: &std::sync::Arc<crate::array::Storage>| Src::Buffer(std::sync::Arc::clone(storage));
    let (ub, fb) = (buffer(u.storage().ok().unwrap()), buffer(f.storage().ok().unwrap()));
    trace::clear();
    let shift: Args = [ub.clone(), Src::Scalar(Scalar::u32(64)), Src::unused()];
    assert!(super::super::simplify::rewrite(&Op::new(OpKind::Binary(BinOp::Shl), ElemTy::U32), &shift, n).is_some());
    let select: Args = [Src::Scalar(Scalar::u32(1)), fb.clone(), fb];
    assert!(super::super::simplify::rewrite(&Op::new(OpKind::Select, ElemTy::F32), &select, n).is_some());
    let _ = UnOp::Id;
    trace::take().into_iter().map(|(id, _)| id).collect()
}

#[test]
fn every_elementwise_rule_fires_in_the_harness() {
    let h = Harness::new();
    let mut fired: Vec<RuleId> = scalar_rules(&h.strict, &Numerics::default());
    let table = rules();
    for (name, ids) in firings(&h, &h.fast) {
        assert!(!ids.is_empty(), "{name}: the lhs fires no rule");
        fired.extend(ids.iter().copied());
        // The registered rule this entry specifies is among the ones that fired.
        if let Some(info) = RULES.iter().find(|info| info.harness == name) {
            assert!(ids.contains(&info.id), "{name}: {} did not fire (fired {:?})", info.name, ids.iter().map(|id| id.info().unwrap().name).collect::<Vec<_>>());
        }
        let _ = &table;
    }
    for info in RULES {
        assert!(fired.contains(&info.id), "{} is hit by no harness lhs", info.name);
    }
}

#[test]
fn rules_fire_only_when_allowed() {
    let h = Harness::new();
    let (table, typed) = (rules(), typed_cases());
    for bits in 0..8u8 {
        let n = subset(bits);
        let ctx = open(n);
        let mut fired: Vec<RuleId> = scalar_rules(&ctx, &n);
        for (name, ids) in firings(&h, &ctx) {
            for id in &ids {
                let info = id.info().unwrap();
                assert!(info.allowed(&n), "{name}: {} (gate {:?}) fired under {n}", info.name, info.gate);
            }
            // A rule that has its own entry fires exactly when its gate is allowed.
            if let Some(info) = RULES.iter().find(|info| info.harness == name) {
                assert_eq!(ids.contains(&info.id), info.allowed(&n), "{name}: {} under {n}: fired {:?}", info.name, ids.iter().map(|id| id.info().unwrap().name).collect::<Vec<_>>());
            }
            fired.extend(ids);
        }
        // Under every subset the fired set is the rules of the allowed gates.
        for info in RULES {
            assert_eq!(fired.contains(&info.id), info.allowed(&n), "{} under {n}", info.name);
        }
        let _ = (&table, &typed);
    }
}

#[test]
fn every_rule_of_the_registry_passes_the_harness() {
    let h = Harness::new();
    let table = rules();
    for info in RULES {
        if let Some(rule) = table.iter().find(|rule| rule.name == info.harness) {
            super::harness::check(&h, rule);
        }
    }
    for case in typed_cases() {
        check_typed(&h.strict, &case);
    }
}

#[test]
fn the_rules_hash_tells_flags_and_modes_apart() {
    let modes = [Algebra::Off, Algebra::Api, Algebra::T1, Algebra::On];
    let mut seen = Vec::new();
    for bits in 0..8u8 {
        for mode in modes {
            let hash = rules_hash(&subset(bits), mode);
            assert!(!seen.contains(&hash), "{bits} {mode:?}");
            seen.push(hash);
            assert_eq!(hash, rules_hash(&subset(bits), mode));
        }
    }
    // Only the three flags and the mode: the reduced-precision bits are not rules.
    assert_eq!(rules_hash(&Numerics::default(), Algebra::On), rules_hash(&Numerics::strict(), Algebra::On));
}

#[test]
fn a_trace_made_under_other_rules_is_not_replayed() {
    let mut stamps = Vec::new();
    for bits in 0..8u8 {
        let ctx = open(subset(bits));
        let stamp = crate::jit::rerun::stamp(&ctx);
        assert!(!stamps.contains(&stamp), "{bits}");
        assert_eq!(stamp, crate::jit::rerun::stamp(&open(subset(bits))));
        stamps.push(stamp);
    }
}

#[test]
fn the_production_tables_are_empty_and_no_context_activates_the_engine() {
    assert!(T1_RULES.is_empty() && super::t2::T2_RULES.is_empty());
    for numerics in [Numerics::strict(), Numerics::default(), Numerics::algebraic(), Numerics::fast()] {
        let ctx = open(numerics);
        assert!(!ctx.jit().rules.t1_active && !ctx.jit().rules.t2_active, "{numerics}");
        // Forcing structured values plans nothing and builds no scope.
        trace::clear();
        let x = crate::array::Array::from_slice_on(&ctx, &(0..2000).map(|i| (i % 5) as f32).collect::<Vec<_>>(), 2000).unwrap();
        let sums: Vec<_> = (0..4).map(|k| (&x * (k as f32 + 1.0)).sum()).collect();
        let dependent = (&x * &sums[0]).sum();
        assert!(sums.iter().all(|s| s.item().unwrap() >= 0.0) && dependent.item().unwrap() >= 0.0);
        assert_eq!(trace::scopes_built(), 0);
    }
}
