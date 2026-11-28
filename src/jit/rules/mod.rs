//! The rewrite rules: a registry of every rule the library runs and the engine that runs the structured ones.
//!
//! * [`registry`]: [`RULES`], one [`RuleInfo`] per rule (name, [`Class`], tier, the entry of the checking harness that specifies
//!   it) and the [`hit`] every firing of `simplify.rs` passes through; [`rules_hash`] is what a trace or a plan is stamped with.
//! * `t1`: the hook that rewrites a comprehension when its form is made (rules of tier 1 for comprehensions; none in production
//!   yet).
//! * `scope`, `measure`, `t2`: the normaliser that rewrites the planning scope of a structured value when it is forced, to a
//!   fixpoint, accepting a rewrite only if the estimated time (then the node count) strictly decreases (rules of tier 2; none in
//!   production yet).
//! * tests only: `table` (the rules as data, two expressions each, equal over the reals), `harness` (checks each by its class
//!   against exact-`f32` semantics: ℤ/p fingerprints, bit equality, error bounds) and the tests of the registry and the engine.

#![allow(dead_code, reason = "the registry is read by the tests and the tooling, and the tables of tiers 1 (comprehension) and 2 are empty until R4 and R5 add rules")]

mod chain;
mod index;
mod measure;
mod reform;
mod registry;
mod scope;
mod t1;
mod t2;
#[cfg(test)]
mod engine_tests;
#[cfg(test)]
mod harness;
#[cfg(test)]
mod index_tests;
#[cfg(test)]
mod table;
#[cfg(test)]
mod tests;

pub(crate) use index::{Product, with_operand_props, diag_of_product, needs_t1, product as product_hook};
pub(crate) use registry::{Rid, RuleId, hit, rules_hash};
pub(crate) use scope::MAX_SCOPE_COMPS;
pub(crate) use t1::{T1_RULES, t1};
pub(crate) use t2::{Planned, RulesState, plan, run as run_planned};

/// How far the two sides of a rule may differ.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Class {
    /// Bit-equal on every `f32` input (up to the payload of a NaN).
    Exact,
    /// Bit-equal on every finite input, signed zeros and subnormals included.
    ExactFinite,
    /// Equal on the reals; the rounding differs by an error bound.
    Reassociating,
    /// Equal on the reals except at the points `guard` names (a signed zero, a zero
    /// denominator, an infinity); the error bound holds on random finite inputs.
    Reformulating,
}
