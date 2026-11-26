//! Tier 1 for comprehensions: the rules that rewrite one form when it is made (07 §3.1), from nothing but the form.
//!
//! The table is empty in production in this part: rules 1, 3, 5 and 12 of 07 §3.2 are R4's. What is here is the hook and its
//! safety net: a rewrite is taken only if the new form is well formed and lowerable, keeps the result type and the output shape,
//! and reads nothing the old one did not (a debug assertion).

use std::panic::Location;

use super::registry::RuleInfo;
#[cfg(test)]
use super::registry::RuleId;
use super::scope::same_src;
use crate::context::Context;
use crate::jit::comprehension::lower::{lowerable, output_dims};
use crate::jit::comprehension::{Comprehension, DimKind, Origin};
use crate::jit::Src;
use crate::numerics::Numerics;
use crate::profile::{Decision, DecisionSite, Verdict};

/// A rule of tier 1 for comprehensions: a pure function of one form (its maps, body and combiner), no walk of the node graph.
/// `inline_sources` of R1 is the way to see through views.
pub(crate) struct T1Rule {
    pub(crate) info: &'static RuleInfo,
    pub(crate) apply: fn(&Comprehension, &Numerics) -> Option<Comprehension>,
}

/// The rules of the library. Empty: R4 adds rules 1, 3, 5 and 12.
pub(crate) static T1_RULES: &[T1Rule] = &[];

/// Rounds of the hook, as many as `create` makes.
const ROUNDS: usize = 4;

/// The form after the rules of [`T1_RULES`] that the context allows, at most [`ROUNDS`] rewrites. With no rule allowed (always, in
/// this part) this is one load and a branch.
#[inline]
#[track_caller]
pub(crate) fn t1(form: Comprehension, ctx: &Context) -> Comprehension {
    if !ctx.jit().rules.t1_active {
        return form;
    }
    run(form, ctx, T1_RULES, ctx.numerics_ref(), Location::caller())
}

/// [`t1`] over the rules `rules` (the engine's tests give their own).
pub(super) fn run(mut form: Comprehension, ctx: &Context, rules: &[T1Rule], numerics: &Numerics, location: &'static Location<'static>) -> Comprehension {
    for _ in 0..ROUNDS {
        let Some((rule, new)) = first(&form, rules, numerics, ctx, location) else { break };
        check_rewrite(&form, &new);
        form = new;
        form.origin = Origin::Rewritten(rule.info.id);
    }
    form
}

/// The first rule that makes a valid, lowerable form, in table order; the ones that made an invalid one are noted.
fn first<'a>(form: &Comprehension, rules: &'a [T1Rule], numerics: &Numerics, ctx: &Context, location: &'static Location<'static>) -> Option<(&'a T1Rule, Comprehension)> {
    let mut rejected: Vec<(&'static str, &'static str)> = Vec::new();
    let mut found = None;
    for rule in rules {
        if !rule.info.allowed(numerics) {
            continue;
        }
        let Some(mut new) = (rule.apply)(form, numerics) else { continue };
        new.origin = Origin::Rewritten(rule.info.id);
        if new.validate().is_err() {
            debug_assert!(false, "{}: a rewrite made an invalid form: {:?}", rule.info.name, new.validate());
            rejected.push((rule.info.name, "invalid"));
        } else if !lowerable(&new) {
            rejected.push((rule.info.name, "not lowerable"));
        } else {
            found = Some((rule, new));
            break;
        }
    }
    if ctx.jit().recording_decisions() && (found.is_some() || !rejected.is_empty()) {
        let chosen = found.as_ref().map_or("none", |(rule, _)| rule.info.name);
        let mut decision = Decision::new("comprehension", location, DecisionSite::Rewrite, chosen);
        for (name, why) in rejected {
            decision = decision.candidate(name, Verdict::Rejected(why));
        }
        ctx.jit().note_decision(decision);
    }
    found
}

/// A rewrite never changes the result type or the output shape, and never reads a source the original did not (so no new reader
/// has to be registered). Debug builds assert both.
pub(super) fn check_rewrite(old: &Comprehension, new: &Comprehension) {
    debug_assert_eq!(old.ty, new.ty, "a rewrite changed the result type");
    debug_assert_eq!(output_dims(old), output_dims(new), "a rewrite changed the output shape");
    #[cfg(debug_assertions)]
    {
        let sources = |form: &Comprehension| -> Vec<Src> {
            let mut all: Vec<Src> = form.inputs.iter().map(|access| access.src.clone()).collect();
            all.extend(form.inputs.iter().filter_map(|access| access.indirect.as_ref().map(|indirect| indirect.index.clone())));
            all.extend(form.space.iter().filter_map(|dim| match &dim.kind {
                DimKind::Segmented { offsets, .. } => Some((**offsets).clone()),
                _ => None,
            }));
            all
        };
        let had = sources(old);
        for src in sources(new) {
            debug_assert!(had.iter().any(|have| same_src(have, &src)), "a rewrite reads a source the original did not");
        }
    }
    #[cfg(not(debug_assertions))]
    let _ = (old, new);
}

/// The id a rule of the tests gets (see [`RuleId::TEST_BASE`]).
#[cfg(test)]
pub(crate) fn test_id(n: u16) -> RuleId {
    RuleId(RuleId::TEST_BASE + n)
}
