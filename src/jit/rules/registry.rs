//! The registry of rewrite rules: every rule the engine runs has one [`RuleInfo`] in [`RULES`] (its name, class, tier and
//! the entry of the checking harness that specifies it), and every firing of a rule of `simplify.rs` is a [`hit`] of its
//! [`Rid`].
//!
//! The class of a rule is a property measured against exact-`f32` semantics: [`Class::Exact`] is bit-equal on every input,
//! [`Class::ExactFinite`] on every finite input including signed zeros, [`Class::Reassociating`] and
//! [`Class::Reformulating`] differ by an error bound (the latter also at the points its guard names). The harness checks
//! the class of every rule with the context pinned to `Numerics::strict()`.
//!
//! A rule runs when its **gate** is allowed by the context's [`Numerics`] ([`Class::allowed`]). The gate equals the class
//! except for four rules that keep the gate they had before classes existed, so that no configuration changes:
//! `div_one` (class `Reformulating`, because the division of a subnormal is flushed to zero where `x * 1` keeps it, gate
//! `Exact`: it always ran) and the three promote candidates `add_neg_to_sub`, `add_neg_to_sub_left` and `sub_neg_to_add`
//! (class `Exact` by the bit check, gate `ExactFinite`: they ran with fast math only; R4 moves them to the exact table,
//! which changes the programs of default-numerics contexts, never their bits).

use super::Class;
use crate::jit::comprehension::Algebra;
use crate::numerics::Numerics;

/// Which rule: an index in [`RULES`] for the rules of the library. The rules of the engine's own tests use ids from
/// [`RuleId::TEST_BASE`] up, which are never in [`RULES`].
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) struct RuleId(pub(crate) u16);

impl RuleId {
    /// The first id of the rules that only tests define.
    #[cfg(test)]
    pub(crate) const TEST_BASE: u16 = 0xF000;

    /// The registry entry of a library rule.
    pub(crate) fn info(self) -> Option<&'static RuleInfo> {
        RULES.get(usize::from(self.0))
    }
}

/// Where a rule applies.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum Tier {
    /// `simplify.rs`: one elementwise operation about to be created.
    T1Elementwise,
    /// A comprehension, when it is built.
    T1Comprehension,
    /// A scope of comprehensions, when a structured value is forced.
    T2,
}

impl Tier {
    fn rank(self) -> u8 {
        match self {
            Tier::T1Elementwise => 0,
            Tier::T1Comprehension => 1,
            Tier::T2 => 2,
        }
    }
}

/// What the engine and the tests know about a rule.
#[derive(Debug)]
pub(crate) struct RuleInfo {
    pub(crate) id: RuleId,
    /// Stable: decision records and the harness use it.
    pub(crate) name: &'static str,
    /// What the rule is, measured against exact-`f32` semantics.
    pub(crate) class: Class,
    /// The class whose flag lets the rule run (see the module documentation).
    pub(crate) gate: Class,
    pub(crate) tier: Tier,
    /// By construction never slower and never more nodes (structural: no cost needed to accept it).
    pub(crate) dominates: bool,
    /// What specifies the rule in the harness: the name of a `Rule` of its table, of a typed case, or `unit:<test>`, a unit
    /// test of `simplify.rs` for a rule whose operand is a scalar a program cannot build from an `Array`.
    pub(crate) harness: &'static str,
    /// Priority inside a tier and gate.
    pub(crate) order: u16,
}

macro_rules! registry {
    ($( $id:ident, $name:literal, $class:ident, $gate:ident, $dominates:literal, $harness:literal; )*) => {
        /// The rules of `simplify.rs`, by the name of their [`RuleInfo`]; the index in [`RULES`].
        #[derive(Clone, Copy, PartialEq, Eq, Debug)]
        #[repr(u16)]
        pub(crate) enum Rid { $( $id, )* }

        /// Every rule of the library, sorted by (tier, gate rank, order).
        pub(crate) static RULES: &[RuleInfo] = &[
            $( RuleInfo {
                id: RuleId(Rid::$id as u16),
                name: $name,
                class: Class::$class,
                gate: Class::$gate,
                tier: Tier::T1Elementwise,
                dominates: $dominates,
                harness: $harness,
                order: Rid::$id as u16,
            }, )*
        ];
    };
}

// Order is the order of the arms in `simplify.rs`: the tables are tried exact, exact-finite, reassociating, reformulating, and
// inside the exact-finite pass the three reformulating arms that always sat there (B6) keep their place.
registry! {
    // exact
    PartOfMake, "part_of_make", Exact, Exact, true, "complex real of make";
    MakeOfParts, "make_of_parts", Exact, Exact, true, "complex make of the parts";
    MulOne, "mul_one", Exact, Exact, true, "mul by one";
    MulNegOne, "mul_neg_one", Exact, Exact, true, "mul by minus one";
    MulZeroInt, "mul_zero_int", Exact, Exact, true, "u32 mul zero";
    DivOne, "div_one", Reformulating, Exact, true, "div by one";
    SubZero, "sub_zero", Exact, Exact, true, "sub plus zero";
    SubSameInt, "sub_same_int", Exact, Exact, true, "u32 sub self";
    AddNegZeroFloat, "add_neg_zero_float", Exact, Exact, true, "add minus zero";
    AddZeroInt, "add_zero_int", Exact, Exact, true, "u32 add zero";
    MinMaxSame, "minmax_same", Exact, Exact, true, "min same";
    RemOneUint, "rem_one_uint", Exact, Exact, true, "u32 rem one";
    BitOrZero, "bitor_zero", Exact, Exact, true, "u32 or zero";
    BitXorZero, "bitxor_zero", Exact, Exact, true, "u32 xor zero";
    BitXorSame, "bitxor_same", Exact, Exact, true, "u32 xor self";
    BitOpSame, "bitop_same", Exact, Exact, true, "u32 and self";
    BitAndZero, "bitand_zero", Exact, Exact, true, "u32 and zero";
    BitAndOnes, "bitand_ones", Exact, Exact, true, "u32 and all ones";
    ShiftMod32, "shift_mod32", Exact, Exact, true, "unit:simplify_binary_shift_by_a_multiple_of_32";
    ShiftZero, "shift_zero", Exact, Exact, true, "u32 shl zero";
    NegNeg, "neg_neg", Exact, Exact, true, "neg neg";
    ConjConj, "conj_conj", Exact, Exact, true, "complex conj conj";
    BitNotBitNot, "bitnot_bitnot", Exact, Exact, true, "u32 not not";
    AbsAbsNeg, "abs_abs_neg", Exact, Exact, true, "abs abs";
    RoundFamily, "round_family", Exact, Exact, true, "floor floor";
    ShlShl, "shl_shl", Exact, Exact, true, "u32 shl shl";
    ShrShr, "shr_shr", Exact, Exact, true, "u32 shr shr";
    RotlRotl, "rotl_rotl", Exact, Exact, true, "u32 rotl rotl";
    SignSign, "sign_sign", Exact, Exact, true, "sign sign";
    SelectScalarMask, "select_scalar_mask", Exact, Exact, true, "unit:simplify_scalar_select_picks_a_branch";
    SelectNotSwap, "select_not_swap", Exact, Exact, true, "select of a negated mask";
    SelectSame, "select_same", Exact, Exact, true, "select same";
    CmpSameInt, "cmp_same_int", Exact, Exact, true, "u32 le self";
    LtGtSame, "lt_gt_same", Exact, Exact, true, "lt same";
    CastThroughF64, "cast_through_f64", Exact, Exact, true, "cast through f64";
    // exact finite (the arms that were fast-only, in their old order)
    AddNegToSub, "add_neg_to_sub", Exact, ExactFinite, true, "add neg";
    AddNegToSubLeft, "add_neg_to_sub_left", Exact, ExactFinite, true, "neg add";
    SubNegToAdd, "sub_neg_to_add", Exact, ExactFinite, true, "sub neg";
    SubSameFloat, "sub_same_float", ExactFinite, ExactFinite, true, "sub same";
    CmpSameFloat, "cmp_same_float", ExactFinite, ExactFinite, true, "le same";
    // reassociating
    ReassociateMulMul, "reassociate_mul_mul", Reassociating, Reassociating, true, "mul chain";
    ReassociateAddAdd, "reassociate_add_add", Reassociating, Reassociating, true, "add chain";
    ReassociateAddSub, "reassociate_add_sub", Reassociating, Reassociating, true, "sub then add";
    ReassociateSubAdd, "reassociate_sub_add", Reassociating, Reassociating, true, "add then sub";
    ReassociateSubSub, "reassociate_sub_sub", Reassociating, Reassociating, true, "sub chain";
    DivConstToMul, "div_const_to_mul", Reassociating, Reassociating, true, "div by constant";
    PowfOne, "powf_one", Reassociating, Reassociating, true, "powf one";
    PowfTwo, "powf_two", Reassociating, Reassociating, true, "powf two";
    PowfNegOne, "powf_neg_one", Reassociating, Reassociating, true, "powf minus one";
    PowfHalf, "powf_half", Reassociating, Reassociating, true, "powf half";
    // reformulating
    AddZeroFloat, "add_zero_float", Reformulating, Reformulating, true, "add zero";
    MulZeroFloat, "mul_zero_float", Reformulating, Reformulating, true, "mul zero";
    DivSameFloat, "div_same_float", Reformulating, Reformulating, true, "div same";
    PowfZero, "powf_zero", Reformulating, Reformulating, true, "powf zero";
}

/// The rule `id` fired. Nothing in release: a rewrite is a branch and a copy of the operands, and every instruction here is
/// on the path of every operation that is created.
#[cfg(not(any(test, feature = "rule-trace")))]
#[inline(always)]
pub(crate) fn hit(_id: Rid, rewritten: Option<(crate::jit::Op, crate::jit::Args)>) -> Option<(crate::jit::Op, crate::jit::Args)> {
    rewritten
}

/// The rule `id` fired: counts it on this thread (what the tests read, see [`trace`]).
#[cfg(any(test, feature = "rule-trace"))]
#[inline(always)]
pub(crate) fn hit(id: Rid, rewritten: Option<(crate::jit::Op, crate::jit::Args)>) -> Option<(crate::jit::Op, crate::jit::Args)> {
    if rewritten.is_some() {
        trace::bump(id);
    }
    rewritten
}

/// Per-thread counts of the firings of the rules of `simplify.rs` (the `rule-trace` feature and tests).
#[cfg(any(test, feature = "rule-trace"))]
#[allow(dead_code)]
pub(crate) mod trace {
    use std::cell::RefCell;

    use super::{RULES, Rid, RuleId};

    thread_local! {
        static HITS: RefCell<Vec<u32>> = const { RefCell::new(Vec::new()) };
        static SCOPES: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
    }

    /// A planning scope was built on this thread.
    pub(crate) fn scope_built() {
        SCOPES.set(SCOPES.get() + 1);
    }

    /// The scopes built on this thread so far (the production tables are empty: none).
    pub(crate) fn scopes_built() -> u64 {
        SCOPES.get()
    }

    pub(super) fn bump(id: Rid) {
        HITS.with_borrow_mut(|hits| {
            if hits.len() < RULES.len() {
                hits.resize(RULES.len(), 0);
            }
            hits[id as usize] += 1;
        });
    }

    /// The rules that fired on this thread since the last call, with how often, and the counts back to zero.
    pub(crate) fn take() -> Vec<(RuleId, u32)> {
        HITS.with_borrow_mut(|hits| {
            let fired = hits.iter().enumerate().filter(|(_, count)| **count > 0).map(|(i, count)| (RuleId(i as u16), *count)).collect();
            hits.clear();
            fired
        })
    }

    /// Forgets the firings so far.
    pub(crate) fn clear() {
        HITS.with_borrow_mut(Vec::clear);
    }
}

impl Class {
    /// Whether a context with these numerics lets a rule of this class (gate) run.
    #[inline(always)]
    pub(crate) fn allowed(self, n: &Numerics) -> bool {
        match self {
            Class::Exact => true,
            Class::ExactFinite => n.exact_finite,
            Class::Reassociating => n.reassociate,
            Class::Reformulating => n.reformulate,
        }
    }

    /// Exact 0, ExactFinite 1, Reassociating 2, Reformulating 3: the order of tables and of ties.
    pub(crate) fn rank(self) -> u8 {
        match self {
            Class::Exact => 0,
            Class::ExactFinite => 1,
            Class::Reassociating => 2,
            Class::Reformulating => 3,
        }
    }
}

impl RuleInfo {
    /// The key [`RULES`] is sorted by.
    pub(crate) fn sort_key(&self) -> (u8, u8, u16) {
        (self.tier.rank(), self.gate.rank(), self.order)
    }

    /// Whether the numerics let this rule run.
    pub(crate) fn allowed(&self, n: &Numerics) -> bool {
        self.gate.allowed(n)
    }
}

/// What of the numerics and the algebra mode decides which rules run: the three class flags and the mode. A trace or a plan made
/// under other rules must not be replayed (see `rerun::stamp`).
#[inline]
pub(crate) fn rules_hash(n: &Numerics, mode: Algebra) -> u64 {
    u64::from(n.exact_finite) | u64::from(n.reassociate) << 1 | u64::from(n.reformulate) << 2 | (mode as u64) << 3
}
