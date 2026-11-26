//! The scatter family: `put`, `put_add`/`put_min`/`put_max` and everything built on them (`bincount`, `histogram`, `group_*`,
//! `factorize`, `unique`). Two selections: the host path in `put_with` and the kernel's tail in `jit::assign::scatter`
//! (`ScatterAttrs::host_site` tells which).

use super::{boundary, entry, id, split_global, split_reduce, FitsInputs};
use crate::caps::Caps;
use crate::context::Context;
use crate::jit::assign::{PRIVATE_BINS, PRIVATE_RATIO};
use crate::jit::structured::boundary::{ScatterAttrs, ScatterOp, ScatterParams};
use crate::jit::structured::strategy::{BoundaryAttrs, FamilyTag, Params, Residencies, Strategy, StrategyId, Subject};

impl StrategyId {
    pub(crate) const SCATTER_HOST: StrategyId = id(FamilyTag::Scatter, 0);
    pub(crate) const SCATTER_PLAIN: StrategyId = id(FamilyTag::Scatter, 1);
    pub(crate) const SCATTER_ATOMIC_PRIVATE: StrategyId = id(FamilyTag::Scatter, 2);
    pub(crate) const SCATTER_ATOMIC_GLOBAL: StrategyId = id(FamilyTag::Scatter, 3);
}

/// Table order is today's: the host path first, then the tail of the kernel.
pub(crate) static SCATTER_STRATEGIES: [Strategy; 4] = [
    entry(StrategyId::SCATTER_HOST, "host", Caps::HOST_PATH, FitsInputs::HOST, split_global, fits_host),
    entry(StrategyId::SCATTER_PLAIN, "plain", Caps::empty(), FitsInputs::NONE, split_global, fits_plain),
    entry(StrategyId::SCATTER_ATOMIC_PRIVATE, "atomic_private", Caps::empty(), FitsInputs::NONE, split_reduce, fits_atomic_private),
    entry(StrategyId::SCATTER_ATOMIC_GLOBAL, "atomic_global", Caps::empty(), FitsInputs::NONE, split_reduce, fits_atomic_global),
];

fn attrs(subject: &Subject<'_>) -> ScatterAttrs {
    match boundary(subject).attrs {
        BoundaryAttrs::Scatter(attrs) => attrs,
        _ => unreachable!("a scatter boundary carries ScatterAttrs"),
    }
}

fn params(attrs: &ScatterAttrs) -> Params {
    Params::Scatter(ScatterParams { len: u32::try_from(attrs.base_len).unwrap_or(u32::MAX) })
}

/// `put_with`: `atomic.is_none() && can_run_on_host() && !view && !read view && base <= 65536 && count <= 65536` (the host path may
/// still decline).
fn fits_host(subject: &Subject<'_>, ctx: &Context, _: &Residencies) -> Option<Params> {
    let a = attrs(subject);
    (a.host_site
        && a.atomic == ScatterOp::None
        && ctx.can_run_on_host()
        && !a.base_is_view
        && !a.base_is_read_view
        && a.base_len <= crate::host::MAX_HOST_ELEMS as u64
        && a.count <= crate::host::MAX_HOST_ELEMS as u64)
        .then(|| params(&a))
}

fn fits_plain(subject: &Subject<'_>, _: &Context, _: &Residencies) -> Option<Params> {
    let a = attrs(subject);
    (!a.host_site && a.atomic == ScatterOp::None).then(|| params(&a))
}

/// `base.len() <= PRIVATE_BINS && count / PRIVATE_RATIO >= base.len()`.
fn fits_atomic_private(subject: &Subject<'_>, _: &Context, _: &Residencies) -> Option<Params> {
    let a = attrs(subject);
    (!a.host_site && a.atomic != ScatterOp::None && a.base_len <= PRIVATE_BINS as u64 && a.count / PRIVATE_RATIO as u64 >= a.base_len).then(|| params(&a))
}

fn fits_atomic_global(subject: &Subject<'_>, _: &Context, _: &Residencies) -> Option<Params> {
    let a = attrs(subject);
    (!a.host_site && a.atomic != ScatterOp::None).then(|| params(&a))
}
