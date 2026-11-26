//! The strategy tables of the eager-boundary families (W2-U3a: scatter, scan, sort). Each is plain data in the family's own file;
//! the op that selects loops over [`candidates`](super::strategy::candidates) and runs the body of today's function for the id it
//! gets (a `match` on a constant: the existing generic code stays monomorphic, so `lower` is `None`).

pub(crate) mod factor;
pub(crate) mod line_solve;
pub(crate) mod conv;
pub(crate) mod opaque;
pub(crate) mod scan;
pub(crate) mod scatter;
pub(crate) mod sort;
pub(crate) mod window;

use super::strategy::{BoundaryAttrs, BitClass, FamilyTag, FitsFn, NumClass, Params, SplitClass, Strategy, StrategyId, Subject, Warm};
use crate::array::Failure;
use crate::caps::Caps;
use crate::context::Context;
use super::strategy::Boundary;

/// The boundary a family's `fits` was asked about (these tables only ever see boundaries).
#[inline(always)]
pub(super) fn boundary<'a, 'b>(subject: &'b Subject<'a>) -> &'b Boundary<'a> {
    match subject {
        Subject::Boundary(boundary) => boundary,
        Subject::Comp(_) | Subject::Contraction(_) => unreachable!("a boundary family's table is only asked about boundaries"),
    }
}

/// An entry of a boundary family: exact, baseline, its own bit class and rank group, no cost (table order decides), no prologue
/// or epilogue; the op runs it by matching on `id`.
pub(super) const fn entry(id: StrategyId, name: &'static str, needs: Caps, split: fn(&Subject<'_>, &Params) -> SplitClass, fits: FitsFn) -> Strategy {
    Strategy { id, name, needs, numerics: NumClass::Exact, bits: BitClass(id.1 as u16), baseline: true, group: id.1, prologue: false, epilogue: false, fits, cost: None, split, warm: no_warm, lower: None }
}

/// `strategy` taking a pending operand into its first read (`Strategy::prologue`).
pub(super) const fn with_prologue(mut strategy: Strategy) -> Strategy {
    strategy.prologue = true;
    strategy
}

/// Baseline strategies are never warmed (they may block on their own compile, as before).
fn no_warm(_: &Subject<'_>, _: &Params, _: &Context) -> Result<Warm, Failure> {
    Ok(Warm { kernels: Vec::new() })
}

pub(super) fn split_global(_: &Subject<'_>, _: &Params) -> SplitClass {
    SplitClass::Global
}

/// A kernel that reads a halo of `k - 1` cells around its tile: conv and window strategies give the radius themselves.
pub(super) fn split_halo(subject: &Subject<'_>, _: &Params) -> SplitClass {
    let radius = match boundary(subject).attrs {
        BoundaryAttrs::Window(attrs) => u32::from(attrs.k[0].max(attrs.k[1]).max(attrs.k[2]).saturating_sub(1)),
        _ => 0,
    };
    SplitClass::Halo { flat_radius: radius }
}

pub(super) fn split_map(_: &Subject<'_>, _: &Params) -> SplitClass {
    SplitClass::Map
}

pub(super) fn split_scan(_: &Subject<'_>, _: &Params) -> SplitClass {
    SplitClass::Scan
}

pub(super) fn split_reduce(_: &Subject<'_>, _: &Params) -> SplitClass {
    SplitClass::Reduce
}

/// The family of `id` for the table's own consts.
pub(super) const fn id(family: FamilyTag, at: u8) -> StrategyId {
    StrategyId(family, at)
}

#[cfg(test)]
mod dense_tests;
#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_conv;
