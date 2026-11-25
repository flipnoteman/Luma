//! `FamilyTag::Opaque`: the library ops that launch kernels and have no selection of their own are declared here, one entry per
//! public op family, so that the decision log and `LUMA_STRATEGY` see the whole library. A declaration is not a selection: its one
//! strategy is the op's current function and always applies, so there is nothing to scan; [`opaque!`] only records the
//! [`Decision`] while decisions are recorded (one relaxed atomic load otherwise). Every entry is `SplitClass::Global`.
//!
//! The labels are `opaque::<family>_<op>`, lowercase with underscores, stable from 0.6.0 (they are API for the decision log).
//! The sparse products have real branches and are strategies of the `Opaque` table (`families::opaque`).

use std::panic::Location;

use super::strategy::{FamilyTag, SplitClass, StrategyId};
use crate::context::Context;
use crate::profile::{Decision, DecisionSite};

/// One declared op family.
#[allow(dead_code)] // `id`, `eager_bounded` and `readbacks` are read by the tests and, after v0.6, by the indirect-dispatch step
pub(crate) struct OpaqueOp {
    /// `(Opaque, n)`: places after the sparse strategies of the table.
    pub(crate) id: StrategyId,
    /// The Decision label (`opaque::vision_fast`).
    pub(crate) label: &'static str,
    /// The size of the result depends on the data (06 D7): an eager boundary that reads a count back. The first customers of an
    /// indirect-dispatch step (after v0.6) are the entries with this flag.
    pub(crate) eager_bounded: bool,
    /// Device-to-host reads the op does (counts, sizes).
    pub(crate) readbacks: u8,
}

impl OpaqueOp {
    /// Every declared op is one global kernel sequence for the schedule compiler.
    #[allow(dead_code)]
    pub(crate) const fn split(&self) -> SplitClass {
        SplitClass::Global
    }
}

const fn declare(at: u8, label: &'static str, eager_bounded: bool, readbacks: u8) -> OpaqueOp {
    OpaqueOp { id: StrategyId(FamilyTag::Opaque, at), label, eager_bounded, readbacks }
}

/// Where the declared ops start in the family's ids: the sparse strategies take the first slots of the table.
const FIRST: u8 = 4;

pub(crate) const IMAGE_POINTWISE: OpaqueOp = declare(FIRST, "opaque::image_pointwise", false, 0);
pub(crate) const IMAGE_SAT: OpaqueOp = declare(FIRST + 1, "opaque::image_sat", false, 0);
pub(crate) const IMAGE_MOMENTS: OpaqueOp = declare(FIRST + 2, "opaque::image_moments", false, 0);
pub(crate) const IMAGE_UNWRAP: OpaqueOp = declare(FIRST + 3, "opaque::image_unwrap", false, 0);
pub(crate) const IMAGE_WRAP: OpaqueOp = declare(FIRST + 4, "opaque::image_wrap", false, 0);
pub(crate) const IMAGE_HIST_EQUAL: OpaqueOp = declare(FIRST + 5, "opaque::image_hist_equal", false, 0);
pub(crate) const SIGNAL_IIR: OpaqueOp = declare(FIRST + 6, "opaque::signal_iir", false, 0);
pub(crate) const SIGNAL_IIR_SOS: OpaqueOp = declare(FIRST + 7, "opaque::signal_iir_sos", false, 0);
pub(crate) const VISION_FAST: OpaqueOp = declare(FIRST + 8, "opaque::vision_fast", true, 1);
pub(crate) const VISION_HARRIS: OpaqueOp = declare(FIRST + 9, "opaque::vision_harris", true, 1);
pub(crate) const VISION_SUSAN: OpaqueOp = declare(FIRST + 10, "opaque::vision_susan", true, 1);
pub(crate) const VISION_DOG: OpaqueOp = declare(FIRST + 11, "opaque::vision_dog", false, 0);
pub(crate) const VISION_ORB: OpaqueOp = declare(FIRST + 12, "opaque::vision_orb", true, 1);
pub(crate) const VISION_SIFT: OpaqueOp = declare(FIRST + 13, "opaque::vision_sift", true, 1);
pub(crate) const VISION_GLOH: OpaqueOp = declare(FIRST + 14, "opaque::vision_gloh", true, 1);
pub(crate) const VISION_CANNY: OpaqueOp = declare(FIRST + 15, "opaque::vision_canny", false, 0);
pub(crate) const VISION_DIFFUSION: OpaqueOp = declare(FIRST + 16, "opaque::vision_diffusion", false, 0);
pub(crate) const VISION_MEAN_SHIFT: OpaqueOp = declare(FIRST + 17, "opaque::vision_mean_shift", false, 0);
pub(crate) const VISION_MATCH_TEMPLATE: OpaqueOp = declare(FIRST + 18, "opaque::vision_match_template", false, 0);
pub(crate) const VISION_NEAREST_NEIGHBOUR: OpaqueOp = declare(FIRST + 19, "opaque::vision_nearest_neighbour", false, 0);
pub(crate) const VISION_HAMMING: OpaqueOp = declare(FIRST + 20, "opaque::vision_hamming", false, 0);
pub(crate) const VISION_REGIONS: OpaqueOp = declare(FIRST + 21, "opaque::vision_regions", false, 0);
pub(crate) const VISION_CONFIDENCE_CC: OpaqueOp = declare(FIRST + 22, "opaque::vision_confidence_cc", false, 0);
pub(crate) const VISION_DECONV: OpaqueOp = declare(FIRST + 23, "opaque::vision_deconv", false, 0);
pub(crate) const VISION_HOMOGRAPHY: OpaqueOp = declare(FIRST + 24, "opaque::vision_homography", false, 0);
pub(crate) const SPARSE_CONVERT: OpaqueOp = declare(FIRST + 25, "opaque::sparse_convert", true, 0);
pub(crate) const SPARSE_ELEMENTWISE: OpaqueOp = declare(FIRST + 26, "opaque::sparse_elementwise", false, 0);
/// The row classes of a matrix, read back once at its first product (memoised per matrix).
pub(crate) const SPARSE_ROW_PLAN: OpaqueOp = declare(FIRST + 27, "opaque::sparse_row_plan", true, 2);

/// Every declared op, in declaration order.
#[allow(dead_code)]
pub(crate) static OPAQUE: [&OpaqueOp; 28] = [
    &IMAGE_POINTWISE,
    &IMAGE_SAT,
    &IMAGE_MOMENTS,
    &IMAGE_UNWRAP,
    &IMAGE_WRAP,
    &IMAGE_HIST_EQUAL,
    &SIGNAL_IIR,
    &SIGNAL_IIR_SOS,
    &VISION_FAST,
    &VISION_HARRIS,
    &VISION_SUSAN,
    &VISION_DOG,
    &VISION_ORB,
    &VISION_SIFT,
    &VISION_GLOH,
    &VISION_CANNY,
    &VISION_DIFFUSION,
    &VISION_MEAN_SHIFT,
    &VISION_MATCH_TEMPLATE,
    &VISION_NEAREST_NEIGHBOUR,
    &VISION_HAMMING,
    &VISION_REGIONS,
    &VISION_CONFIDENCE_CC,
    &VISION_DECONV,
    &VISION_HOMOGRAPHY,
    &SPARSE_CONVERT,
    &SPARSE_ELEMENTWISE,
    &SPARSE_ROW_PLAN,
];

/// Records the Decision of a declared op that is running: one relaxed load while nobody records.
#[inline(always)]
pub(crate) fn note_opaque(ctx: &Context, op: &OpaqueOp, name: &'static str, location: &'static Location<'static>) {
    if ctx.jit().recording_decisions() {
        record(ctx, op, name, location);
    }
}

#[cold]
#[inline(never)]
fn record(ctx: &Context, op: &OpaqueOp, name: &'static str, location: &'static Location<'static>) {
    ctx.jit().note_decision(Decision::new(name, location, DecisionSite::OpaquePlan, op.label));
}

/// One line at the top of the op's `run` function: `opaque!(ctx, VISION_FAST, "fast", location)`.
macro_rules! opaque {
    ($ctx:expr, $op:ident, $name:expr, $location:expr) => {
        $crate::jit::structured::opaque::note_opaque($ctx, &$crate::jit::structured::opaque::$op, $name, $location)
    };
}
pub(crate) use opaque;
