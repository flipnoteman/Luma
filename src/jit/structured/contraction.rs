//! The contraction family's strategies: the branches `matmul::run` chose between with `if`s, one strategy each, and the plan
//! functions they run with (moved here from `matmul.rs`). Table order is the priority chain: host, tensor cores in TF32 (the
//! aligned kernels, then the ragged and padded ones for any other shape), the backend's own `f32` tiles (likewise), the CPU's
//! native product, skinny, tiled. The aligned ones, the CPU's, skinny and tiled are baselines, each its own rank group; the ragged
//! and the padded kernel of one precision share a group and are ranked by cost (same bits, so either may stand in for the other).

use std::panic::Location;
use std::sync::Arc;

use super::emit::Emit;
use super::epilogue::{EpiMap, EpiProgram};
use super::gemm_hook;
use super::strategy::{BitClass, Estimate, FamilyTag, FitsInputs, Lowered, NumClass, Output, Params, Residencies, SplitClass, Strategy, StrategyId, Subject, Warm};
use crate::array::{Array, Failure, Storage};
use crate::builtin::Builtin;
use crate::caps::Caps;
use crate::context::{Context, RawArg, Scratch};
use crate::error::LumaError;
use crate::gpu_type::ElemLayout;
use crate::jit::comprehension::Props;
use crate::jit::structured::select;
use crate::jit::program::Ref;
use smallvec::SmallVec;
use crate::jit::{self, AxisMap, Edge, ElemTy, MatmulSide, OpKind, ShiftKind, Src};
use crate::matmul::{MatmulTile, Plan, SLICE};
use crate::numerics::Reduced;
use crate::ops::{fold_grid, Outcome};
use crate::shape::Shape;
use crate::tuning::Tuning;
use crate::view::PadMode;
use luma_core::{CostModel, GemmLanes};

// -- the value --

/// Which precision a product asks for: the context's numerics (`Default`, `matmul`), full `f32` (`matmul_f32`, batched products), or
/// TF32 whatever the numerics say (`matmul_tf32`).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum Tf32Request {
    Default,
    Exact,
    Tf32,
}

impl Tf32Request {
    /// The request as `OpTag::Matmul` says it: `None` for the context's numerics, else the precision asked for.
    pub(crate) fn request(self) -> Option<bool> {
        match self {
            Tf32Request::Default => None,
            Tf32Request::Exact => Some(false),
            Tf32Request::Tf32 => Some(true),
        }
    }
}

/// What a contraction's `fits` reads besides the operands.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) struct ContractionAttrs {
    pub(crate) tf32: Tf32Request,
    /// The operands are the same value (`xᵀ x`, `x xᵀ`): the product may compute half of its output and mirror it (`SyrkTiled`).
    pub(crate) half: bool,
}

/// The extents of a product: `lhs` is `[K, M]`, `rhs` `[N, K]`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct Geometry {
    pub(crate) m: u32,
    pub(crate) n: u32,
    pub(crate) k: u32,
}

/// A matrix product asked for, as the strategies see it (eager in this part: the operands may be pending expressions).
pub(crate) struct ContractionView<'a> {
    pub(crate) lhs: &'a Array<f32>,
    pub(crate) rhs: &'a Array<f32>,
    pub(crate) attrs: ContractionAttrs,
    pub(crate) location: &'static Location<'static>,
    /// What an absorbed consumer applies to each output element (W3-U4). A strategy whose `Strategy::epilogue` is set may take it: it
    /// then returns [`Output::Epilogued`]. Any other output is the plain product, and the caller applies the program as a separate
    /// elementwise pass ([`crate::jit::structured::lazy`]).
    #[allow(dead_code, reason = "W3-U4: the strategies that take an epilogue read it")]
    pub(crate) epilogue: Option<&'a EpiProgram>,
    /// Products that share `lhs` and may be launched with this one (W3-U4 A2); empty unless a strategy of the table takes partners. A
    /// strategy with `Strategy::partners` that launches them all returns [`Output::Group`]; any other output leaves them pending.
    #[allow(dead_code, reason = "W3-U4: the strategies that take partners read it")]
    pub(crate) partners: &'a [Partner],
}

/// A product that shares its left operand with the one being run: its right operand and its epilogue, as strategies read them.
#[derive(Clone)]
#[allow(dead_code, reason = "W3-U4: the strategies that take partners read it")]
pub(crate) struct Partner {
    pub(crate) rhs: Array<f32>,
    /// The program of its absorbed epilogue and the arrays of its inputs (`EpiProgram::inputs` order); all members' programs have the
    /// same instructions.
    pub(crate) epilogue: Option<PartnerEpilogue>,
}

/// A partner's epilogue program and the arrays of its inputs.
pub(crate) type PartnerEpilogue = (Arc<EpiProgram>, SmallVec<[Array<f32>; 3]>);

impl ContractionView<'_> {
    /// The extents, if the operands are of one context and their shapes are those of a product that fits `u32`.
    fn geometry(&self) -> Option<Geometry> {
        let (ld, rd) = (self.lhs.shape().dims(), self.rhs.shape().dims());
        if ld[0] != rd[1] || ld[2..].iter().chain(&rd[2..]).any(|&extent| extent != 1) || !self.lhs.context().same_as(self.rhs.context()) {
            return None;
        }
        Some(Geometry { m: u32::try_from(ld[1]).ok()?, n: u32::try_from(rd[0]).ok()?, k: u32::try_from(ld[0]).ok()? })
    }

    /// The shape that picks the plan (`[N, M, K]`), whatever the operands are.
    pub(crate) fn dims(&self) -> [u32; 4] {
        let (ld, rd) = (self.lhs.shape().dims(), self.rhs.shape().dims());
        let word = |extent: usize| u32::try_from(extent).unwrap_or(u32::MAX);
        [word(rd[0]), word(ld[1]), word(ld[0]), 1]
    }

    /// Whether the product runs in TF32 where the device has tensor cores: the request, else the context's numerics.
    fn wants_tf32(&self, ctx: &Context) -> bool {
        match self.attrs.tf32 {
            Tf32Request::Tf32 => true,
            Tf32Request::Exact => false,
            Tf32Request::Default => ctx.numerics().tf32_matmul(),
        }
    }

    /// The errors of a product in the order `run` reported them: a poisoned operand's own, a context mismatch, a shape mismatch,
    /// an extent above `u32`, an operand split over several buffers.
    pub(crate) fn check(&self) -> Result<Geometry, Failure> {
        let (lhs, rhs) = (self.lhs, self.rhs);
        for array in [lhs, rhs] {
            if let Some(error) = array.failed() {
                return Err(Failure::Inherited(Arc::clone(error)));
            }
        }
        let ctx = lhs.context();
        if !ctx.same_as(rhs.context()) {
            return Err(LumaError::ContextMismatch.into());
        }
        let (ld, rd) = (lhs.shape().dims(), rhs.shape().dims());
        if ld[0] != rd[1] || ld[2..].iter().chain(&rd[2..]).any(|&extent| extent != 1) {
            return Err(LumaError::ShapeMismatch { lhs: ld, rhs: rd }.into());
        }
        let (k, m, n) = (ld[0], ld[1], rd[0]);
        let too_large = |extent: usize| u32::try_from(extent).map_err(|_| Failure::from(LumaError::DimensionTooLarge { shape: [n, m, 1, 1] }));
        let geom = Geometry { m: too_large(m)?, n: too_large(n)?, k: too_large(k)? };
        for array in [lhs, rhs] {
            if array.pending().is_some_and(|node| node.len() as u64 * 4 > ctx.segment_bytes()) {
                return Err(LumaError::Chunked("a matrix product of an array split over several buffers").into());
            }
        }
        Ok(geom)
    }
}

/// The product of `lhs` and `rhs` through the registry (the body of `matmul`, `matmul_tf32`, `matmul_f32`): an evaluated array.
#[inline(always)]
pub(crate) fn run(lhs: &Array<f32>, rhs: &Array<f32>, attrs: ContractionAttrs, location: &'static Location<'static>) -> Outcome<Array<f32>> {
    let view = ContractionView { lhs, rhs, attrs, location, epilogue: None, partners: &[] };
    let geom = view.check()?;
    let ctx = lhs.context();
    let storage = select::execute(ctx, &Subject::Contraction(&view))?.into_storage();
    Ok(Array::from_storage(ctx, [geom.n as usize, geom.m as usize, 1, 1].into(), storage))
}

/// Whether the host strategy takes the product (a small one, on a backend whose memory the host reads). Pure.
pub(crate) fn host_fits(view: &ContractionView<'_>, ctx: &Context) -> bool {
    ctx.caps().contains(Caps::HOST_PATH) && fits_host(&Subject::Contraction(view), ctx, &Residencies::DEVICE).is_some()
}

/// What a launch of a product fails with when an operand or the result needs several buffers (the kernels read and write one).
pub(crate) fn chunked() -> Failure {
    LumaError::Chunked("this operation needs an array that fits one buffer").into()
}

/// Whether an operand that is evaluated already, or a result of `out_len` elements, needs more than one buffer: what a launch of
/// the product would fail on with [`chunked`], and what a lazy product reports when it is made.
pub(crate) fn split_over_buffers(ctx: &Context, operands: [&Array<f32>; 2], out_len: usize) -> bool {
    operands.iter().any(|array| array.recorded_storage().is_some_and(|storage| storage.segments() > 1)) || (out_len as u64).saturating_mul(4) > ctx.segment_bytes()
}

// -- what a strategy runs with --

/// What `fits` worked out and `lower` runs with: small, because it is copied through the registry's frames (a reduction forces
/// another through `execute`, and deep expressions recurse). The plan itself is computed once, by `lower` ([`plan_of`]).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct ContractionParams {
    /// The strategy these are for.
    pub(crate) by: StrategyId,
    pub(crate) geom: Geometry,
}

/// How an operand of a native product sits in its buffer: whether the buffer holds the transpose of the matrix, the leading dimension
/// (the distance in elements between the starts of its rows) and the element offset of its first element.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct Layout {
    pub(crate) transposed: bool,
    pub(crate) ld: u32,
    pub(crate) off: u32,
}

impl Layout {
    /// Whether 16-byte copies can read the operand: the leading dimension and the offset are multiples of 4.
    fn vector_aligned(self) -> bool {
        self.ld.is_multiple_of(4) && self.off.is_multiple_of(4)
    }
}

/// The extent of the first axis of `array`, which is its leading dimension when it is stored densely.
fn first_axis(array: &Array<f32>) -> u32 {
    u32::try_from(array.shape().dims()[0]).unwrap_or(u32::MAX)
}

/// How `array` is read as an operand of a native product, without evaluating anything: a window of a stored matrix in place, else a
/// dense buffer of the array's own shape. Pure.
fn layout(array: &Array<f32>) -> Layout {
    match matrix_window(array) {
        Some(window) => Layout { transposed: window.transposed, ld: window.ld, off: window.offset },
        None => Layout { transposed: false, ld: first_axis(array), off: 0 },
    }
}

/// Whether `by` is a padded strategy: it copies the operands that are not 16-byte aligned.
fn is_padded(by: StrategyId) -> bool {
    matches!(by, StrategyId::GEMM_TF32_PADDED | StrategyId::GEMM_F32_PADDED)
}

/// Whether `by` multiplies in `f32` (else TF32).
fn is_exact(by: StrategyId) -> bool {
    matches!(by, StrategyId::GEMM_F32 | StrategyId::GEMM_F32_RAGGED | StrategyId::GEMM_F32_PADDED)
}

/// The operands of the strategy `by` that are copied before the product: those the padded kernels cannot read in place.
fn copies(view: &ContractionView<'_>, by: StrategyId) -> [bool; 2] {
    [view.lhs, view.rhs].map(|array| is_padded(by) && !layout(array).vector_aligned())
}

/// The layouts the strategy `by` reads the operands with: the arrays' own, except the copies (dense, a leading dimension rounded
/// up to 4). Pure.
fn read_layouts(view: &ContractionView<'_>, by: StrategyId) -> [Layout; 2] {
    let copied = copies(view, by);
    let mut layouts = [layout(view.lhs), layout(view.rhs)];
    for ((layout, copied), array) in layouts.iter_mut().zip(copied).zip([view.lhs, view.rhs]) {
        if copied {
            *layout = Layout { transposed: false, ld: first_axis(array).next_multiple_of(4), off: 0 };
        }
    }
    layouts
}

/// `plan` for a kernel that copies its operands with `lanes`.
fn with_lanes(plan: Plan, lanes: GemmLanes) -> Plan {
    match plan.tile {
        MatmulTile::Gemm { rows, cols, lhs_t, rhs_t, exact } => Plan { tile: MatmulTile::GemmRagged { rows, cols, lhs_t, rhs_t, exact, lanes }, ..plan },
        _ => plan,
    }
}

/// The plan of the strategy `params.by` for its product, `transposed` saying which operands are read as the transpose of the buffer
/// they are views of (see [`matrix_window`]; it is the tile's, not the cut's): `None` where the strategy has none (the host),
/// the error of the plan function (a grid too large) otherwise. Pure.
fn plan_of(view: &ContractionView<'_>, ctx: &Context, params: &ContractionParams, transposed: [bool; 2]) -> Option<Result<Plan, LumaError>> {
    let Geometry { m, n, k } = params.geom;
    let (groups, bytes) = (ctx.max_workgroups_per_dimension(), ctx.segment_bytes());
    let finish = |plan: Result<Plan, LumaError>, lanes: Option<GemmLanes>| Some(match lanes {
        Some(lanes) => plan.map(|plan| with_lanes(plan, lanes)),
        None => plan,
    });
    let tensor = |lanes: Option<GemmLanes>| finish(gemm_plan(m, n, k, ctx.tuning(), GEMM_HALF_TILES_PER_SM, ctx.hal().gemm_tiles(false), transposed, false, groups, bytes), lanes);
    let f32_tiles = |lanes: Option<GemmLanes>| finish(gemm_plan(m, n, k, ctx.tuning(), GEMM_HALF_TILES_PER_SM, ctx.hal().gemm_tiles(true), transposed, true, groups, bytes), lanes);
    match params.by {
        StrategyId::GEMM_TF32 => tensor(None),
        StrategyId::GEMM_TF32_RAGGED => tensor(Some(GemmLanes::Scalar)),
        StrategyId::GEMM_TF32_PADDED => tensor(Some(GemmLanes::Vec4Tails)),
        StrategyId::GEMM_F32 => f32_tiles(None),
        StrategyId::GEMM_F32_RAGGED => f32_tiles(Some(GemmLanes::Scalar)),
        StrategyId::GEMM_F32_PADDED => f32_tiles(Some(GemmLanes::Vec4Tails)),
        StrategyId::CONTRACTION_NATIVE_CPU => Some(native_plan(m, n, k, ctx.tuning().native_gemm, transposed, groups, bytes)),
        StrategyId::CONTRACTION_SKINNY => skinny_plan(m, n, k, along_k(view.lhs, params.geom), groups, bytes).map(Ok),
        StrategyId::CONTRACTION_TILED => Some(tiled_plan(ctx.tuning(), m, n, k, groups, bytes)),
        by if by == StrategyId::SYRK_TILED => syrk_plan(view, ctx, params.geom, transposed),
        _ => None,
    }
}

/// Rule 4 (`shared_operand`, W3-R4R5): whether the product of the view's left operand with the right operands side by side (the view's
/// own and the `partners'`) sums every output over `K` as each product would alone. It holds when the plan of every member (`plan_of`,
/// pure: nothing is launched or compiled) and the plan of the merged product (`N` the sum of theirs) cut `K` the same way: the same
/// `kc` and no slicing of `K`. Then the merged product is `Exact`, the table entry of the rule; otherwise it is `Reassociating` and does
/// not run at the default numerics. `false` where a strategy has no plan for either.
#[allow(dead_code, reason = "W3-U4's multi-operand kernels ask it before a group is launched")]
pub(crate) fn merge_keeps_k_order(view: &ContractionView<'_>, ctx: &Context, by: StrategyId) -> bool {
    let Some(geom) = view.geometry() else { return false };
    let transposed = read_layouts(view, by).map(|layout| layout.transposed);
    let plan = |n: u32| match plan_of(view, ctx, &ContractionParams { by, geom: Geometry { n, ..geom } }, transposed) {
        Some(Ok(plan)) => Some(plan),
        _ => None,
    };
    let widths: SmallVec<[u32; 4]> = std::iter::once(geom.n).chain(view.partners.iter().map(|partner| u32::try_from(partner.rhs.shape().dims()[0]).unwrap_or(u32::MAX))).collect();
    let Some(merged) = widths.iter().try_fold(0u32, |all, &n| all.checked_add(n)).and_then(plan) else { return false };
    merged.slices == 1 && widths.iter().all(|&n| plan(n).is_some_and(|solo| solo.slices == 1 && solo.kc == merged.kc))
}

/// The stable label of the Decision of a strategy run with `params`.
pub(crate) fn label(params: &ContractionParams, subject: &Subject<'_>, ctx: &Context) -> String {
    match params.by {
        StrategyId::CONTRACTION_HOST => "host".to_string(),
        _ => match view(subject).and_then(|view| plan_of(view, ctx, params, [false; 2])) {
            Some(Ok(plan)) => plan_label(&plan),
            Some(Err(_)) => "grid too large".to_string(),
            None => params.by.name().to_string(),
        },
    }
}

fn params(by: StrategyId, geom: Geometry) -> Params {
    Params::Contraction(ContractionParams { by, geom })
}

fn view<'a>(subject: &Subject<'a>) -> Option<&'a ContractionView<'a>> {
    match subject {
        Subject::Contraction(view) => Some(view),
        _ => None,
    }
}

/// Whether the elements of the left operand along `K` are neighbours in memory (not a transposed view).
fn along_k(lhs: &Array<f32>, geom: Geometry) -> bool {
    lhs.pending().is_none_or(|node| jit::stored_along_k(&MatmulSide::Pending(Arc::clone(node)), [geom.k, geom.m]))
}

/// Whether the product runs on tensor cores where the device has them: the numerics ask for TF32 and the device can.
fn tf32_here(view: &ContractionView<'_>, ctx: &Context) -> bool {
    view.wants_tf32(ctx) && ctx.caps().contains(Caps::TF32)
}

/// The tensor-core tiles the product runs on in its aligned kernels, if it is one that does.
fn tensor_tiles<'c>(view: &ContractionView<'_>, ctx: &'c Context) -> &'c [(u32, u32)] {
    if tf32_here(view, ctx) && aligned_shape(view.lhs, view.rhs).is_some() { ctx.hal().gemm_tiles(false) } else { &[] }
}

/// Whether the operands' leading dimensions and offsets are multiples of 4: what the aligned kernels (and the 16-byte lanes) read.
fn aligned_operands(view: &ContractionView<'_>) -> bool {
    layout(view.lhs).vector_aligned() && layout(view.rhs).vector_aligned()
}

// -- the table --

impl StrategyId {
    pub(crate) const CONTRACTION_HOST: StrategyId = StrategyId(FamilyTag::Contraction, 0);
    pub(crate) const GEMM_TF32: StrategyId = StrategyId(FamilyTag::Contraction, 1);
    pub(crate) const GEMM_TF32_RAGGED: StrategyId = StrategyId(FamilyTag::Contraction, 2);
    pub(crate) const GEMM_TF32_PADDED: StrategyId = StrategyId(FamilyTag::Contraction, 3);
    pub(crate) const GEMM_F32: StrategyId = StrategyId(FamilyTag::Contraction, 4);
    pub(crate) const GEMM_F32_RAGGED: StrategyId = StrategyId(FamilyTag::Contraction, 5);
    pub(crate) const GEMM_F32_PADDED: StrategyId = StrategyId(FamilyTag::Contraction, 6);
    pub(crate) const CONTRACTION_NATIVE_CPU: StrategyId = StrategyId(FamilyTag::Contraction, 7);
    pub(crate) const CONTRACTION_SKINNY: StrategyId = StrategyId(FamilyTag::Contraction, 8);
    pub(crate) const CONTRACTION_TILED: StrategyId = StrategyId(FamilyTag::Contraction, 9);
    /// Not in [`CONTRACTION_STRATEGIES`]: the table of [`CONTRACTION_HALF`].
    pub(crate) const SYRK_TILED: StrategyId = StrategyId(FamilyTag::Contraction, 10);
}

/// The products' table, in preference order.
pub(crate) const CONTRACTION_STRATEGIES: [Strategy; 10] = [
    // The host's sums differ from the device's: its own bit class. The backend's own kernels and the skinny ones read the
    // operands' state (`FitsInputs::OPERANDS`); the others only the shape, the numerics and the device. The aligned, the ragged and
    // the padded kernels of one precision give the same bits for the same plan: one class each (3: TF32, 4: `f32`).
    entry(StrategyId::CONTRACTION_HOST, "host", Caps::HOST_PATH, FitsInputs::HOST, NumClass::Exact, 1, 0, fits_host, lower_host),
    entry(StrategyId::GEMM_TF32, "gemm-tf32", Caps::TF32, FitsInputs::OPERANDS, NumClass::Reduced(Reduced::TF32_MATMUL), 3, 1, fits_gemm_tf32, lower_gemm),
    cut(StrategyId::GEMM_TF32_RAGGED, "gemm-tf32-ragged", Caps::TF32, NumClass::Reduced(Reduced::TF32_MATMUL), 3, 2, fits_gemm_tf32_ragged),
    cut(StrategyId::GEMM_TF32_PADDED, "gemm-tf32-padded", Caps::TF32, NumClass::Reduced(Reduced::TF32_MATMUL), 3, 2, fits_gemm_tf32_padded),
    entry(StrategyId::GEMM_F32, "gemm-f32", Caps::GEMM_TILES, FitsInputs::OPERANDS, NumClass::Exact, 4, 3, fits_gemm_f32, lower_gemm),
    cut(StrategyId::GEMM_F32_RAGGED, "gemm-f32-ragged", Caps::GEMM_TILES, NumClass::Exact, 4, 4, fits_gemm_f32_ragged),
    cut(StrategyId::GEMM_F32_PADDED, "gemm-f32-padded", Caps::GEMM_TILES, NumClass::Exact, 4, 4, fits_gemm_f32_padded),
    entry(StrategyId::CONTRACTION_NATIVE_CPU, "native-cpu", Caps::NATIVE_GEMM, FitsInputs::NONE, NumClass::Exact, 5, 5, fits_native_cpu, lower_gemm),
    epilogued(entry(StrategyId::CONTRACTION_SKINNY, "skinny", Caps::empty(), FitsInputs::OPERANDS, NumClass::Exact, 6, 6, fits_skinny, lower_skinny)),
    epilogued(entry(StrategyId::CONTRACTION_TILED, "tiled", Caps::empty(), FitsInputs::NONE, NumClass::Exact, 7, 7, fits_tiled, lower_tiled)),
];

/// The table a Gram product (`ContractionAttrs::half`) is tried against before [`CONTRACTION_STRATEGIES`]: `select::execute` consults
/// it only for such a product, so a plain product pays one flag test (and never evaluates `fits_syrk`).
///
/// A Gram product (`ContractionAttrs::half`) computes the lower triangle of its tiles and mirrors it: the bits of the full
/// product's lower triangle, bit-symmetric (the upper triangle of the full product may differ on tensor cores, hence its own class).
/// On the tensor cores (the default numerics of a CUDA device) it is the TF32 kernel's lower triangle: it is admitted under any numerics
/// like the `f32` one, and takes the TF32 tiles where the product would run them (`wants_tf32`).
pub(crate) const CONTRACTION_HALF: [Strategy; 1] =
    [entry(StrategyId::SYRK_TILED, "syrk-tiled", Caps::empty(), FitsInputs::OPERANDS, NumClass::Exact, 8, 9, fits_syrk, lower_syrk)];

/// Whether the product is a Gram product that may run the half-work strategy.
#[inline(always)]
pub(crate) fn is_half(subject: &Subject<'_>) -> bool {
    matches!(subject, Subject::Contraction(view) if view.attrs.half)
}

/// Whether some strategy of the table launches partner products with the subject (`Strategy::partners`): the partner search runs only then.
pub(crate) const TAKES_PARTNERS: bool = any_partners(&CONTRACTION_STRATEGIES);

const fn any_partners(entries: &[Strategy]) -> bool {
    let mut at = 0;
    while at < entries.len() {
        if entries[at].partners {
            return true;
        }
        at += 1;
    }
    false
}

/// `strategy` that applies an absorbed epilogue at its stores (see [`epilogue_of`]).
const fn epilogued(mut strategy: Strategy) -> Strategy {
    strategy.epilogue = true;
    strategy
}

/// Whether the strategy `id` applies `ContractionView::epilogue` itself: the native tensor-core products (TF32), through the hook of
/// their store stage (`gemm_hook`). The exact (`f32`) kernel is not hooked: ptxas gives it 125 registers instead of 153 for any change
/// to its store stage, and it loses 1-6 % (spike D).
const fn takes_epilogue(id: StrategyId) -> bool {
    matches!(id, StrategyId::GEMM_TF32 | StrategyId::GEMM_TF32_RAGGED | StrategyId::GEMM_TF32_PADDED)
}

/// A baseline entry: it exists at f0e5723 and is never refused by the variant budget or by "still compiling"; no cost, the
/// first of the table that fits runs.
#[allow(clippy::too_many_arguments)]
const fn entry(
    id: StrategyId,
    name: &'static str,
    needs: Caps,
    reads: FitsInputs,
    numerics: NumClass,
    bits: u16,
    group: u8,
    fits: fn(&Subject<'_>, &Context, &Residencies) -> Option<Params>,
    lower: fn(&Subject<'_>, &Params, &mut Emit<'_>) -> Result<Lowered, Failure>,
) -> Strategy {
    Strategy { id, name, needs, numerics, bits: BitClass(bits), baseline: true, group, prologue: false, epilogue: takes_epilogue(id), partners: false, replayable: matches!(id.1, 1 | 4 | 7..=9), reads, fits, cost: None, split: split_global, warm: no_warm, lower: Some(lower) }
}

/// A ragged or a padded kernel: new in this part, so not a baseline (its kernel may be compiling, or over the variant budget: the
/// call then blocks on the compile, as nothing of its bit class can stand in), ranked by cost against the other of its group.
const fn cut(id: StrategyId, name: &'static str, needs: Caps, numerics: NumClass, bits: u16, group: u8, fits: fn(&Subject<'_>, &Context, &Residencies) -> Option<Params>) -> Strategy {
    Strategy {
        id,
        name,
        needs,
        numerics,
        bits: BitClass(bits),
        baseline: false,
        group,
        prologue: false,
        epilogue: takes_epilogue(id),
        partners: false,
        // The ragged kernels are plain launches over the evaluated operands; the padded ones make copies (new nodes the plan does not know).
        replayable: matches!(id.1, 2 | 5),
        reads: FitsInputs::OPERANDS,
        fits,
        cost: Some(cost_cut),
        split: split_global,
        warm: warm_cut,
        lower: Some(lower_gemm),
    }
}

fn split_global(_: &Subject<'_>, _: &Params) -> SplitClass {
    SplitClass::Global
}

/// Baseline strategies are never warmed (they may block on their own compile, as before).
fn no_warm(_: &Subject<'_>, _: &Params, _: &Context) -> Result<Warm, Failure> {
    Ok(Warm { kernels: Vec::new() })
}

fn done(storage: Arc<Storage>) -> Result<Lowered, Failure> {
    Ok(Lowered::Done(Output::Storage(storage)))
}

/// [`done`] for a launch that applied the epilogue (`applied`): its one output.
fn finish(storage: Arc<Storage>, applied: bool) -> Result<Lowered, Failure> {
    if applied { Ok(Lowered::Done(Output::Epilogued(smallvec::smallvec![storage]))) } else { done(storage) }
}

/// What every `lower` starts from: its view and its parameters, and a way to get the plan of its strategy once the operands are
/// known (`Err` for a grid too large, `None` where there is no plan to run: the strategy is declined).
fn lowering<'a>(subject: &Subject<'a>, params: &Params) -> Option<(&'a ContractionView<'a>, ContractionParams)> {
    let (Some(view), Params::Contraction(params)) = (view(subject), params) else { return None };
    Some((view, *params))
}

/// [`plan_of`] as a `Failure`.
fn plan_or_decline(view: &ContractionView<'_>, ctx: &Context, params: &ContractionParams, transposed: [bool; 2]) -> Result<Option<Plan>, Failure> {
    match plan_of(view, ctx, params, transposed) {
        Some(Ok(plan)) => Ok(Some(plan)),
        Some(Err(error)) => Err(error.into()),
        None => Ok(None),
    }
}

// -- host --

/// `host_product`'s checks: a small product whose operands are on the host (not `try_matmul` itself, which evaluates).
fn fits_host(subject: &Subject<'_>, ctx: &Context, _: &Residencies) -> Option<Params> {
    let view = view(subject)?;
    if !ctx.can_run_on_host() {
        return None;
    }
    let geom = view.geometry()?;
    let (m, n, k) = (u64::from(geom.m), u64::from(geom.n), u64::from(geom.k));
    let limit = crate::host::MAX_HOST_ELEMS as u64;
    if m * n * k > crate::host::MAX_MATMUL_FMAS || m * k > limit || k * n > limit || m * n > limit {
        return None;
    }
    Some(params(StrategyId::CONTRACTION_HOST, geom))
}

/// The product on the host; `None` from it (it did not work out) is `Declined`.
fn lower_host(subject: &Subject<'_>, params: &Params, emit: &mut Emit<'_>) -> Result<Lowered, Failure> {
    let (Some(view), Params::Contraction(params)) = (view(subject), params) else { return Ok(Lowered::Declined) };
    let geom = params.geom;
    match crate::host::try_matmul(emit.ctx(), view.lhs, view.rhs, geom.m as usize, geom.n as usize, geom.k as usize) {
        Some(out) => done(Arc::clone(out.storage()?)),
        None => Ok(Lowered::Declined),
    }
}

// -- the backend's own kernels --

/// The aligned tensor-core kernels: the shapes of [`aligned_shape`] with operands the 16-byte copies can read.
fn fits_gemm_tf32(subject: &Subject<'_>, ctx: &Context, _: &Residencies) -> Option<Params> {
    let view = view(subject)?;
    if tensor_tiles(view, ctx).is_empty() || !aligned_operands(view) {
        return None;
    }
    Some(params(StrategyId::GEMM_TF32, view.geometry()?))
}

/// The default product runs on the backend's own `f32` kernels when it reads its operands as they are stored and is not one of
/// the skinny shapes (see [`skinny_plan`]).
fn fits_gemm_f32(subject: &Subject<'_>, ctx: &Context, _: &Residencies) -> Option<Params> {
    let view = view(subject)?;
    if ctx.tuning().native_gemm != 0 || *OWN_F32_KERNELS_OFF {
        return None;
    }
    let geom = view.geometry()?;
    if aligned_shape(view.lhs, view.rhs).is_none() || !tensor_tiles(view, ctx).is_empty() {
        return None;
    }
    if !gemm_operand(ctx, view.lhs) || !gemm_operand(ctx, view.rhs) {
        return None;
    }
    if skinny_plan(geom.m, geom.n, geom.k, along_k(view.lhs, geom), ctx.max_workgroups_per_dimension(), ctx.segment_bytes()).is_some() || ctx.hal().gemm_tiles(true).is_empty() {
        return None;
    }
    if !aligned_operands(view) {
        return None;
    }
    Some(params(StrategyId::GEMM_F32, geom))
}

/// Smallest product (multiply-adds) that runs on a ragged or a padded kernel: below it the shader kernels' launch is as cheap.
const RAGGED_MIN_FMAS: u64 = 1 << 20;
/// Shortest K of a ragged or padded product: a short K is memory-bound and the shader kernels (and their skinny plans) do as well
/// (`2001 x 37 x 2001` was 1.21x slower on the ragged kernel, 4096 x 33 x 4096 4 % faster).
const RAGGED_MIN_K: usize = 64;
/// Largest `f32` product (multiply-adds) that runs on a ragged or padded kernel: the padded kernel's copy launch and its masked
/// copies lose what the 16-byte lanes gain on a big square (3001^3 1.05x, 4097^3 1.00x of the shader kernels, 1001^3 1.20x on an
/// RTX 3090), so those keep the shader kernels until the kernel is faster (follow-up). TF32 has no such cap.
const RAGGED_MAX_FMAS_F32: u64 = 4_000_000_000;
/// What the 4-byte lanes of a ragged kernel reach of the rate of the 16-byte ones (to be set from the bench rows of the gate).
const RAGGED_LANE_FACTOR: f64 = 0.6;
/// Measured rates of the backend's own kernels at 4096^3 on an RTX 3090, in flops per nanosecond, scaled by the device's
/// `f32_ops_per_ns` against the 3090's calibration.
const GEMM_F32_FLOPS_PER_NS: f64 = 22_900.0;
const GEMM_TF32_FLOPS_PER_NS: f64 = 27_900.0;

/// The products of any shape (and leading dimension and offset) that the backend's own kernels take: `exact` in `f32`, else TF32.
/// Not the skinny shapes (their shader kernels read the operands once), the products below [`GEMM_MIN_EXTENT`] on a side (a tile
/// would be mostly padding: the same bounds as the aligned kernels, see [`aligned_shape`]), nor those below [`RAGGED_MIN_FMAS`].
fn ragged_fits(subject: &Subject<'_>, ctx: &Context, by: StrategyId) -> Option<Params> {
    let view = view(subject)?;
    let exact = is_exact(by);
    if !ctx.hal().gemm_ragged() || ctx.tuning().native_gemm != 0 || *OWN_F32_KERNELS_OFF || tf32_here(view, ctx) == exact {
        return None;
    }
    let geom = view.geometry()?;
    let (m, n, k) = (geom.m as usize, geom.n as usize, geom.k as usize);
    let few_columns = if (SMALL_N_MIN_M..=SMALL_N_MAX_M).contains(&m) { SMALL_N_MIN } else { GEMM_MIN_EXTENT };
    let fmas = (m as u64) * (n as u64) * (k as u64);
    if m < GEMM_MIN_EXTENT || n < few_columns || k < RAGGED_MIN_K || fmas < RAGGED_MIN_FMAS || (exact && fmas > RAGGED_MAX_FMAS_F32) {
        return None;
    }
    if ctx.hal().gemm_tiles(exact).is_empty() || !gemm_operand(ctx, view.lhs) || !gemm_operand(ctx, view.rhs) {
        return None;
    }
    if skinny_plan(geom.m, geom.n, geom.k, along_k(view.lhs, geom), ctx.max_workgroups_per_dimension(), ctx.segment_bytes()).is_some() {
        return None;
    }
    Some(params(by, geom))
}

fn fits_gemm_tf32_ragged(subject: &Subject<'_>, ctx: &Context, _: &Residencies) -> Option<Params> {
    ragged_fits(subject, ctx, StrategyId::GEMM_TF32_RAGGED)
}

fn fits_gemm_tf32_padded(subject: &Subject<'_>, ctx: &Context, _: &Residencies) -> Option<Params> {
    ragged_fits(subject, ctx, StrategyId::GEMM_TF32_PADDED)
}

fn fits_gemm_f32_ragged(subject: &Subject<'_>, ctx: &Context, _: &Residencies) -> Option<Params> {
    ragged_fits(subject, ctx, StrategyId::GEMM_F32_RAGGED)
}

fn fits_gemm_f32_padded(subject: &Subject<'_>, ctx: &Context, _: &Residencies) -> Option<Params> {
    ragged_fits(subject, ctx, StrategyId::GEMM_F32_PADDED)
}

/// Nanoseconds a ragged or a padded kernel is expected to take: its launch, the copies the padded one makes (one kernel for both
/// operands, a read and a write of each), and the product at the rate of its tile (see [`f32_tile_rate`]) times the share of
/// that rate the lanes reach ([`RAGGED_LANE_FACTOR`] for the 4-byte ones).
fn cost_cut(subject: &Subject<'_>, params: &Params, ctx: &Context, _: &Residencies) -> Estimate {
    let (Some(view), Params::Contraction(params)) = (view(subject), params) else { return Estimate { ns: f64::INFINITY, measured: false } };
    let Geometry { m, n, k } = params.geom;
    let by = params.by;
    let model = ctx.cost_model();
    let layouts = read_layouts(view, by);
    let ns = plan_of(view, ctx, params, layouts.map(|layout| layout.transposed)).and_then(Result::ok).map_or(f64::INFINITY, |plan| {
        let (rows, cols) = plan.tile.edges();
        let tiles = f64::from(m.div_ceil(rows)) * f64::from(n.div_ceil(cols));
        let useful = f64::from(m) * f64::from(n) / (tiles * f64::from(rows * cols));
        let workgroups = plan.groups.iter().fold(1u32, |all, &g| all.saturating_mul(g));
        let base = if is_exact(by) { GEMM_F32_FLOPS_PER_NS } else { GEMM_TF32_FLOPS_PER_NS };
        let rate = base * f32_tile_rate((rows, cols), workgroups, ctx.tuning().compute_units, useful) * model.f32_ops_per_ns / CostModel::NVIDIA.f32_ops_per_ns;
        let flops = 2.0 * f64::from(m) * f64::from(n) * f64::from(k);
        let lanes = if is_padded(by) { 1.0 } else { RAGGED_LANE_FACTOR };
        let mut ns = model.launch_us * 1e3 + flops / (rate * lanes);
        let copied = copies(view, by);
        let copy_elems: u64 = [(copied[0], u64::from(m) * u64::from(k)), (copied[1], u64::from(n) * u64::from(k))].into_iter().filter(|&(copied, _)| copied).map(|(_, elems)| elems).sum();
        if copy_elems > 0 {
            ns += model.launch_us * 1e3 + (8 * copy_elems) as f64 / model.bandwidth_gbs;
        }
        ns
    });
    Estimate { ns, measured: false }
}

/// Starts compiling the kernel of a ragged or a padded product.
fn warm_cut(subject: &Subject<'_>, params: &Params, ctx: &Context) -> Result<Warm, Failure> {
    let Some((view, params)) = lowering(subject, params) else { return Ok(Warm { kernels: Vec::new() }) };
    let layouts = read_layouts(view, params.by);
    let Some(plan) = plan_or_decline(view, ctx, &params, layouts.map(|layout| layout.transposed))? else { return Ok(Warm { kernels: Vec::new() }) };
    let core = ctx.builtin(Builtin::Matmul { tile: plan.tile, aligned: plan.aligned, split: plan.split }, "f32")?;
    Ok(Warm { kernels: vec![core] })
}

/// The CPU's native product.
fn fits_native_cpu(subject: &Subject<'_>, ctx: &Context, _: &Residencies) -> Option<Params> {
    let view = view(subject)?;
    if ctx.tuning().native_gemm == 0 {
        return None;
    }
    Some(params(StrategyId::CONTRACTION_NATIVE_CPU, view.geometry()?))
}

/// A product the backend multiplies as native code (its own tiles, or the CPU's): the operands are evaluated first and read as
/// stored, windows of stored matrices in place (`native_sides`); the padded kernels first copy the operands their 16-byte copies
/// cannot read.
fn lower_gemm(subject: &Subject<'_>, params: &Params, emit: &mut Emit<'_>) -> Result<Lowered, Failure> {
    let ctx = emit.ctx();
    let Some((view, params)) = lowering(subject, params) else { return Ok(Lowered::Declined) };
    if is_padded(params.by) {
        return lower_padded(ctx, view, &params);
    }
    gemm(ctx, view, &params, view.lhs, view.rhs)
}

/// [`lower_gemm`] for a padded strategy: the operands its 16-byte copies cannot read are copied first.
#[inline(never)]
fn lower_padded(ctx: &Context, view: &ContractionView<'_>, params: &ContractionParams) -> Result<Lowered, Failure> {
    let copied = copies(view, params.by);
    let (lhs_copy, rhs_copy) = (copied[0].then(|| padded_copy(view.lhs)), copied[1].then(|| padded_copy(view.rhs)));
    let (lhs, rhs) = (lhs_copy.as_ref().unwrap_or(view.lhs), rhs_copy.as_ref().unwrap_or(view.rhs));
    gemm(ctx, view, params, lhs, rhs)
}

/// `TriangularK`: the word of the uniform that cuts the range of K of each tile row (column) to what a triangular operand leaves
/// non-zero: 1 a lower left operand, 2 an upper one, 3 a lower right operand, 4 an upper one, 0 the whole range. The terms left out
/// are products with exact zeros, so it is `exact_finite` (an infinity or a NaN in the zero region would have made the full range a
/// NaN); not for an operand with an offset (a window of a triangle is not one in its own coordinates), nor a half-work product.
#[inline(never)]
fn triangle_word(ctx: &Context, view: &ContractionView<'_>, layouts: [Layout; 2]) -> u32 {
    if !ctx.numerics().exact_finite || layouts.iter().any(|layout| layout.off != 0) {
        return 0;
    }
    let (lhs, rhs) = (view.lhs.props(), view.rhs.props());
    if lhs.has(Props::LOWER) {
        1
    } else if lhs.has(Props::UPPER) {
        2
    } else if rhs.has(Props::LOWER) {
        3
    } else if rhs.has(Props::UPPER) {
        4
    } else {
        0
    }
}

/// The product of `lhs` and `rhs` (the view's operands, or their padded copies) as a native launch: their sides and layouts, the plan
/// of the strategy, one launch.
fn gemm(ctx: &Context, view: &ContractionView<'_>, params: &ContractionParams, lhs: &Array<f32>, rhs: &Array<f32>) -> Result<Lowered, Failure> {
    let (sides, layouts) = native_sides(ctx, lhs, rhs)?;
    let Some(plan) = plan_or_decline(view, ctx, params, layouts.map(|layout| layout.transposed))? else { return Ok(Lowered::Declined) };
    let tri = if matches!(plan.tile, MatmulTile::Gemm { .. } | MatmulTile::GemmRagged { .. }) && (view.lhs.has_props() || view.rhs.has_props()) { triangle_word(ctx, view, layouts) } else { 0 };
    if let Some(program) = view.epilogue.filter(|_| takes_epilogue(params.by))
        && let Some(storage) = launch_hooked(ctx, &sides, &plan, params.geom, layouts, program)?
    {
        return Ok(Lowered::Done(Output::Epilogued(smallvec::smallvec![storage])));
    }
    done(launch(ctx, sides, plan, params.geom, Some((layouts, tri)), None)?)
}

/// `array` in a buffer of its own whose leading dimension is a multiple of 4: zero columns appended to its first axis (an expression,
/// evaluated first together with the other operand's), or a plain copy where the axis is one already and only the offset was not.
fn padded_copy(array: &Array<f32>) -> Array<f32> {
    let first = array.shape().dims()[0];
    let extra = first.next_multiple_of(4) - first;
    if extra > 0 { array.pad([(0, extra), (0, 0)], PadMode::Zero) } else { array.cache() }
}

// -- half-work products --

/// The strategies whose tiles a half-work product runs, in the order their `fits` are asked: the backend's own `f32` kernels, then
/// the CPU's native product, then the tensor-core kernels (`tf32`: only those), then the shader kernels of the product that no
/// other base takes.
const SYRK_BASES: [StrategyId; 8] = [
    StrategyId::GEMM_F32,
    StrategyId::GEMM_F32_RAGGED,
    StrategyId::GEMM_F32_PADDED,
    StrategyId::CONTRACTION_NATIVE_CPU,
    StrategyId::GEMM_TF32,
    StrategyId::GEMM_TF32_RAGGED,
    StrategyId::GEMM_TF32_PADDED,
    StrategyId::CONTRACTION_TILED,
];

/// The strategy whose tiles a half-work product runs: the first of [`SYRK_BASES`] that takes the product, for a Gram product
/// (`ContractionAttrs::half`) with a square output on a backend that has the half-work kernel.
#[inline(never)]
fn syrk_base(view: &ContractionView<'_>, ctx: &Context, tf32: bool) -> Option<StrategyId> {
    if !view.attrs.half {
        return None;
    }
    let geom = view.geometry()?;
    if geom.m != geom.n || geom.k == 0 || geom.m < 2 {
        return None;
    }
    let subject = Subject::Contraction(view);
    SYRK_BASES.into_iter().filter(|&by| (by == StrategyId::CONTRACTION_NATIVE_CPU || by == StrategyId::CONTRACTION_TILED || is_exact(by)) != tf32).find(|&by| {
        let fits = match by {
            // The shader kernels: the product the plain one runs on them (not a skinny one, not one the tensor cores take).
            StrategyId::CONTRACTION_TILED => {
                let skinny = skinny_plan(geom.m, geom.n, geom.k, along_k(view.lhs, geom), ctx.max_workgroups_per_dimension(), ctx.segment_bytes());
                return skinny.is_none() && syrk_base(view, ctx, true).is_none() && fits_tiled(&subject, ctx, &Residencies::DEVICE).is_some();
            }
            StrategyId::GEMM_F32 => fits_gemm_f32(&subject, ctx, &Residencies::DEVICE),
            StrategyId::GEMM_F32_RAGGED => fits_gemm_f32_ragged(&subject, ctx, &Residencies::DEVICE),
            StrategyId::GEMM_F32_PADDED => fits_gemm_f32_padded(&subject, ctx, &Residencies::DEVICE),
            StrategyId::GEMM_TF32 => fits_gemm_tf32(&subject, ctx, &Residencies::DEVICE),
            StrategyId::GEMM_TF32_RAGGED => fits_gemm_tf32_ragged(&subject, ctx, &Residencies::DEVICE),
            StrategyId::GEMM_TF32_PADDED => fits_gemm_tf32_padded(&subject, ctx, &Residencies::DEVICE),
            _ => fits_native_cpu(&subject, ctx, &Residencies::DEVICE),
        };
        fits.is_some() && (by == StrategyId::CONTRACTION_NATIVE_CPU || ctx.hal().gemm_half())
    })
}

/// The base of a half-work product and whether it runs on the tensor cores: a tensor-core base where the product would run on one,
/// else the first of the others.
#[inline(never)]
fn syrk_choice(view: &ContractionView<'_>, ctx: &Context) -> Option<(StrategyId, bool)> {
    syrk_base(view, ctx, true).map(|base| (base, true)).or_else(|| syrk_base(view, ctx, false).map(|base| (base, false)))
}

/// The plan of the half-work product: the plan of its base on square tiles, never split over K (the mirrored triangle would be a
/// sum of partial products), the lower triangle of its tiles as the grid. `None` where the base's plan splits K.
#[inline(never)]
fn syrk_plan(view: &ContractionView<'_>, ctx: &Context, geom: Geometry, transposed: [bool; 2]) -> Option<Result<Plan, LumaError>> {
    let (base, tf32) = syrk_choice(view, ctx)?;
    let Geometry { m, n, k } = geom;
    let (groups, bytes) = (ctx.max_workgroups_per_dimension(), ctx.segment_bytes());
    let plan = if base == StrategyId::CONTRACTION_NATIVE_CPU {
        native_plan(m, n, k, ctx.tuning().native_gemm, transposed, groups, bytes)
    } else if base == StrategyId::CONTRACTION_TILED {
        return tiled_half_plan(ctx.tuning(), m, k, groups);
    } else {
        let squares: Vec<(u32, u32)> = ctx.hal().gemm_tiles(!tf32).iter().copied().filter(|&(rows, cols)| rows == cols).collect();
        let lanes = match base {
            StrategyId::GEMM_F32_RAGGED | StrategyId::GEMM_TF32_RAGGED => GemmLanes::Scalar,
            StrategyId::GEMM_F32_PADDED | StrategyId::GEMM_TF32_PADDED => GemmLanes::Vec4Tails,
            _ => GemmLanes::Vec4,
        };
        gemm_plan(m, n, k, ctx.tuning(), GEMM_HALF_TILES_PER_SM, &squares, transposed, !tf32, groups, bytes).map(|plan| match plan.tile {
            MatmulTile::Gemm { rows, cols, lhs_t, rhs_t, exact } => Plan { tile: MatmulTile::GemmHalf { rows, cols, lhs_t, rhs_t, exact, lanes }, ..plan },
            _ => plan,
        })
    };
    let plan = match plan {
        Ok(plan) if plan.slices == 1 => plan,
        Ok(_) => return None,
        Err(error) => return Some(Err(error)),
    };
    let tile = match plan.tile {
        MatmulTile::Native { edge, lhs_t, rhs_t } => MatmulTile::NativeHalf { edge, lhs_t, rhs_t },
        half @ MatmulTile::GemmHalf { .. } => half,
        _ => return None,
    };
    let (rows, _) = tile.edges();
    let side = m.div_ceil(rows);
    let count = u64::from(side) * (u64::from(side) + 1) / 2;
    if count > u64::from(groups) {
        return None;
    }
    Some(Ok(Plan { tile, aligned: false, split: false, slices: 1, kc: k, groups: [count as u32, 1, 1] }))
}

/// Half tiles (the lower triangle of 128 x 128 tiles) below which the shader kernels' half-work product is not used: the full
/// product's tiles split a long K over the multiprocessors, which the mirrored triangle cannot, and a few big workgroups do not fill
/// them. Like [`SPLIT_TILES`] a count for the 82 multiprocessors of an RTX 3090, raised for bigger devices.
const TILED_HALF_MIN_TILES: u32 = 64;

/// The plan of a half-work product on the shader kernels: [`MatmulTile::T128Half`] with `M == N` of at least
/// [`TILED_HALF_MIN_TILES`] half tiles, `None` for a smaller one (or one whose grid is too large).
#[inline(never)]
fn tiled_half_plan(tuning: &Tuning, m: u32, k: u32, max_groups: u32) -> Option<Result<Plan, LumaError>> {
    let side = m.div_ceil(128);
    let count = u64::from(side) * (u64::from(side) + 1) / 2;
    if m < 128 || count < u64::from(tuning.scaled_up(TILED_HALF_MIN_TILES)) {
        return None;
    }
    let aligned = m.is_multiple_of(128) && k.is_multiple_of(SLICE);
    // The tiles are numbered along the folded grid, as the full ones are.
    let [x, y, _] = if count <= u64::from(max_groups) { [count as u32, 1, 1] } else { fold_grid(u32::try_from(count).ok()?, max_groups) };
    Some(Ok(Plan { tile: MatmulTile::T128Half, aligned, split: false, slices: 1, kc: k, groups: [x, y, 1] }))
}

/// A Gram product on the tiles of the strategy it would otherwise run on, the lower triangle of tiles only.
fn fits_syrk(subject: &Subject<'_>, ctx: &Context, res: &Residencies) -> Option<Params> {
    if !view(subject)?.attrs.half {
        return None;
    }
    fits_syrk_half(subject, ctx, res)
}

#[inline(never)]
fn fits_syrk_half(subject: &Subject<'_>, ctx: &Context, _: &Residencies) -> Option<Params> {
    let view = view(subject)?;
    let geom = view.geometry()?;
    let params = ContractionParams { by: StrategyId::SYRK_TILED, geom };
    syrk_plan(view, ctx, geom, [false; 2])?.ok()?;
    Some(Params::Contraction(params))
}

/// [`lower_gemm`] for a half-work product: the operands are copied first as its base strategy wants.
fn lower_syrk(subject: &Subject<'_>, params: &Params, emit: &mut Emit<'_>) -> Result<Lowered, Failure> {
    let ctx = emit.ctx();
    let Some((view, params)) = lowering(subject, params) else { return Ok(Lowered::Declined) };
    let Some((base, _)) = syrk_choice(view, ctx) else { return Ok(Lowered::Declined) };
    if base == StrategyId::CONTRACTION_TILED {
        let Some(plan) = plan_or_decline(view, ctx, &params, [false; 2])? else { return Ok(Lowered::Declined) };
        let sides = [side(ctx, view.lhs)?, side(view.rhs.context(), view.rhs)?];
        return done(launch(ctx, sides, plan, params.geom, None, None)?);
    }
    if is_padded(base) {
        let copied = copies(view, base);
        let (lhs_copy, rhs_copy) = (copied[0].then(|| padded_copy(view.lhs)), copied[1].then(|| padded_copy(view.rhs)));
        let (lhs, rhs) = (lhs_copy.as_ref().unwrap_or(view.lhs), rhs_copy.as_ref().unwrap_or(view.rhs));
        return gemm(ctx, view, &params, lhs, rhs);
    }
    gemm(ctx, view, &params, view.lhs, view.rhs)
}

// -- the shader kernels --

/// A small output over a long K, from global memory without staging. An operand that is a fused expression read a few times by the
/// tiles would be recomputed at each read: the staged tiles then do better, so the strategy does not fit.
fn fits_skinny(subject: &Subject<'_>, ctx: &Context, _: &Residencies) -> Option<Params> {
    let view = view(subject)?;
    let geom = view.geometry()?;
    let plan = skinny_plan(geom.m, geom.n, geom.k, along_k(view.lhs, geom), ctx.max_workgroups_per_dimension(), ctx.segment_bytes())?;
    if let MatmulTile::Skinny { rows, cols, .. } = plan.tile {
        let reads = [geom.n.div_ceil(u32::from(cols)), geom.m.div_ceil(u32::from(rows))];
        let costly = [view.lhs, view.rhs].into_iter().zip(reads).any(|(array, reads)| (SKINNY_FUSED_READS..SKINNY_EVALUATE_READS).contains(&reads) && array.pending().is_some_and(jit::is_recomputed));
        if costly {
            return None;
        }
    }
    Some(params(StrategyId::CONTRACTION_SKINNY, geom))
}

/// A skinny kernel computes a fused operand once per tile that reads it; past [`SKINNY_EVALUATE_READS`] reads storing the
/// expression first is cheaper (a side effect, so here and not in `fits`).
fn lower_skinny(subject: &Subject<'_>, params: &Params, emit: &mut Emit<'_>) -> Result<Lowered, Failure> {
    let ctx = emit.ctx();
    let Some((view, params)) = lowering(subject, params) else { return Ok(Lowered::Declined) };
    let Some(plan) = plan_or_decline(view, ctx, &params, [false; 2])? else { return Ok(Lowered::Declined) };
    let geom = params.geom;
    let mut sides = [side(ctx, view.lhs)?, side(view.rhs.context(), view.rhs)?];
    if let MatmulTile::Skinny { rows, cols, .. } = plan.tile {
        let reads = [geom.n.div_ceil(u32::from(cols)), geom.m.div_ceil(u32::from(rows))];
        let mut evaluated = false;
        for (array, reads) in [view.lhs, view.rhs].into_iter().zip(reads) {
            if reads >= SKINNY_EVALUATE_READS && let Some(node) = array.pending() {
                evaluated |= jit::evaluate_for_reuse(node)?;
            }
        }
        if evaluated {
            sides = [side(ctx, view.lhs)?, side(view.rhs.context(), view.rhs)?];
        }
    }
    let applied = epilogue_of(view, geom, &plan)?;
    finish(launch(ctx, sides, plan, geom, None, applied.as_ref())?, applied.is_some())
}

/// The staged tiles: always (the plan fails only when the grid is too large, which `lower` reports).
fn fits_tiled(subject: &Subject<'_>, _: &Context, _: &Residencies) -> Option<Params> {
    let view = view(subject)?;
    Some(params(StrategyId::CONTRACTION_TILED, view.geometry()?))
}

fn lower_tiled(subject: &Subject<'_>, params: &Params, emit: &mut Emit<'_>) -> Result<Lowered, Failure> {
    let ctx = emit.ctx();
    let Some((view, params)) = lowering(subject, params) else { return Ok(Lowered::Declined) };
    let Some(plan) = plan_or_decline(view, ctx, &params, [false; 2])? else { return Ok(Lowered::Declined) };
    let sides = [side(ctx, view.lhs)?, side(view.rhs.context(), view.rhs)?];
    let applied = epilogue_of(view, params.geom, &plan)?;
    finish(launch(ctx, sides, plan, params.geom, None, applied.as_ref())?, applied.is_some())
}

// -- epilogues --

/// An epilogue a launch applies at the stores of its kernel: the interned form that holds its structure (`ContractForm`), the storages
/// of its inputs in the order of the program, and the words of its scalars.
struct Applied {
    form: u16,
    inputs: Vec<Arc<Storage>>,
    scalars: Vec<u32>,
}

/// Whether the shader kernels can apply `program`: one stored value, and per element `f32` arithmetic of the operations the elementwise
/// kernels write inline (no helper functions, no bit operations), over `f32` inputs and scalars.
fn epilogue_fits(program: &EpiProgram) -> bool {
    use crate::jit::program::Instr;
    let f32_instr = |instr: &Instr| {
        let kind = match instr.op.kind {
            OpKind::Unary(op) => !op.needs_helper() && !op.is_bitwise(),
            OpKind::Binary(op) => !op.needs_helper() && !op.is_bitwise(),
            OpKind::Compare(_) | OpKind::Clamp | OpKind::Powf | OpKind::Select => true,
            _ => false,
        };
        let args_ok = instr.args.iter().enumerate().all(|(at, arg)| match arg {
            Ref::Unused => true,
            Ref::Sub(_) => false,
            _ => instr.op.operand_ty(at) == ElemTy::F32 || (at == 0 && matches!(instr.op.kind, OpKind::Select)),
        });
        kind && args_ok && instr.op.ty == ElemTy::F32
    };
    program.outputs.len() == 1
        && program.uses_acc()
        && program.validate(0).is_ok()
        && program.instrs.iter().all(f32_instr)
        && program.instrs.get(program.outputs[0] as usize).is_some_and(|instr| instr.op.out_ty() == ElemTy::F32)
        && program.inputs.iter().all(|input| input.ty == ElemTy::F32)
        && program.scalars.iter().all(|scalar| scalar.ty == ElemTy::F32)
}

/// The epilogue of `view` as `plan` can apply it: the plan writes its output once (K is not split into slices summed after), the program
/// is one the kernels take ([`epilogue_fits`]) and its inputs are evaluated arrays of the extent their map reads. `None` runs the product
/// plain, for the caller to apply the program as a separate pass.
fn epilogue_of(view: &ContractionView<'_>, geom: Geometry, plan: &Plan) -> Result<Option<Applied>, Failure> {
    let Some(program) = view.epilogue else { return Ok(None) };
    if plan.slices > 1 || !epilogue_fits(program) {
        return Ok(None);
    }
    let (m, n) = (u64::from(geom.m), u64::from(geom.n));
    let mut inputs = Vec::with_capacity(program.inputs.len());
    for input in &program.inputs {
        let storage = match &input.src {
            Src::Buffer(storage) => Arc::clone(storage),
            Src::Node(node) => match node.force() {
                Ok(storage) => Arc::clone(storage),
                Err(error) => return Err(Failure::Inherited(Arc::clone(error))),
            },
            Src::Scalar(_) => return Ok(None),
        };
        let needed = match input.map {
            EpiMap::Point => m * n,
            EpiMap::AlongRows => m,
            EpiMap::AlongCols => n,
            EpiMap::Scalar => 1,
        };
        if storage.segments() != 1 || (storage.len() as u64) < needed {
            return Ok(None);
        }
        inputs.push(storage);
    }
    let form = crate::jit::comprehension::lower::contract::intern_form(crate::jit::comprehension::lower::contract::ContractForm::with_epilogue(program.key()));
    let scalars = program.scalars.iter().map(|scalar| scalar.bits as u32).collect();
    Ok(Some(Applied { form, inputs, scalars }))
}

// -- launch --

/// The last element of a `rows x cols` matrix stored with `ld` columns from `off`, one past it: what the kernels index with 32 bits.
fn extent(off: u32, rows: u32, cols: u32, ld: u32) -> u64 {
    if rows == 0 || cols == 0 { 0 } else { u64::from(off) + u64::from(rows - 1) * u64::from(ld) + u64::from(cols) }
}

/// The product of `sides` with the epilogue `program` applied by the hooked native kernel, `None` where that is not possible and the
/// plain product runs (the caller applies the program as a separate pass, bit for bit the same): a split product (the epilogue
/// cannot run per slice), operands that are not evaluated, a kernel that is still compiling or that the backend cannot build, an
/// epilogue operand that is a window of another array, a program that is not one the hook computes.
fn launch_hooked(ctx: &Context, sides: &[MatmulSide; 2], plan: &Plan, geom: Geometry, layouts: [Layout; 2], program: &EpiProgram) -> Result<Option<Arc<Storage>>, Failure> {
    let (MatmulSide::Ready(a, a_bytes), MatmulSide::Ready(b, b_bytes)) = (&sides[0], &sides[1]) else { return Ok(None) };
    if plan.slices != 1 {
        return Ok(None);
    }
    let plain = Builtin::Matmul { tile: plan.tile, aligned: plan.aligned, split: plan.split };
    let Some(core) = gemm_hook::kernel(ctx, plain, program)? else { return Ok(None) };
    // Still compiling, or it did not (the table keeps the failure: the plain product and the pass stay): not waited for.
    if !core.raw.is_ready() {
        return Ok(None);
    }
    if core.raw.failed() {
        if let Err(error) = core.raw.resolve::<()>() {
            log::debug!("the epilogue hook of a product did not build: {error}");
            #[cfg(test)]
            {
                static ONCE: std::sync::Once = std::sync::Once::new();
                ONCE.call_once(|| eprintln!("the epilogue hook of a product did not build: {error}"));
            }
        }
        return Ok(None);
    }
    let mut operands: Vec<Arc<Storage>> = Vec::with_capacity(program.inputs.len());
    for input in &program.inputs {
        let storage = match &input.src {
            Src::Buffer(storage) => Arc::clone(storage),
            Src::Node(node) => Arc::clone(node.force().as_ref().map_err(|error| Failure::Inherited(Arc::clone(error)))?),
            Src::Scalar(_) => return Ok(None),
        };
        if storage.window().is_some() || storage.segments() != 1 {
            return Ok(None);
        }
        operands.push(storage);
    }
    let Geometry { m, n, k } = geom;
    let [lhs, rhs] = layouts;
    let (a_extent, b_extent) = (
        if lhs.transposed { extent(lhs.off, k, m, lhs.ld) } else { extent(lhs.off, m, k, lhs.ld) },
        if rhs.transposed { extent(rhs.off, n, k, rhs.ld) } else { extent(rhs.off, k, n, rhs.ld) },
    );
    if a_extent > u64::from(u32::MAX) || b_extent > u64::from(u32::MAX) {
        return Err(LumaError::DimensionTooLarge { shape: [n as usize, m as usize, 1, 1] }.into());
    }
    let len = Shape::from([n as usize, m as usize, 1, 1]).validate()?;
    let out_storage = Arc::new(Storage::new(ctx, len, ElemLayout::plain(4)));
    let out_pin = out_storage.pin_single()?;
    let out_bytes = out_storage.byte_len();
    let pins = operands.iter().map(|storage| storage.pin_read(0).map(|pin| (pin, storage.byte_len()))).collect::<Result<Vec<_>, _>>()?;
    let mut words: Vec<u32> = vec![m, n, k, plan.kc, lhs.ld, rhs.ld, lhs.off, rhs.off];
    words.extend(program.scalars.iter().map(|scalar| scalar.bits as u32));
    let mut args: Vec<RawArg<'_>> = Vec::with_capacity(5 + pins.len());
    args.push(RawArg::write(out_pin.buffer(), out_bytes));
    args.push(RawArg::read(a.buffer(), *a_bytes));
    args.push(RawArg::read(b.buffer(), *b_bytes));
    args.extend(pins.iter().map(|(pin, bytes)| RawArg::read(pin.buffer(), *bytes)));
    args.push(RawArg::Uniform(bytemuck::cast_slice(&words)));
    ctx.launch(&core, plan.groups, [n, m, 1], &args)?;
    drop(args);
    Ok(Some(out_storage))
}

/// Records the product of `sides` into a new buffer: one launch of the kernel of `plan`, and the sum of its slices if it was split.
/// `layouts` are the operands' leading dimensions and offsets for the backend's own kernels (which read eight words of uniform),
/// `None` for the shader kernels.
fn launch(ctx: &Context, sides: [MatmulSide; 2], plan: Plan, geom: Geometry, layouts: Option<([Layout; 2], u32)>, epilogue: Option<&Applied>) -> Result<Arc<Storage>, Failure> {
    let Geometry { m: m32, n: n32, k: k32 } = geom;
    let len = Shape::from([n32 as usize, m32 as usize, 1, 1]).validate()?;
    let words = match layouts {
        Some(([lhs, rhs], tri)) => {
            // The kernels index their operands with 32 bits.
            let (a_extent, b_extent) = (
                if lhs.transposed { extent(lhs.off, k32, m32, lhs.ld) } else { extent(lhs.off, m32, k32, lhs.ld) },
                if rhs.transposed { extent(rhs.off, n32, k32, rhs.ld) } else { extent(rhs.off, k32, n32, rhs.ld) },
            );
            if a_extent > u64::from(u32::MAX) || b_extent > u64::from(u32::MAX) {
                return Err(LumaError::DimensionTooLarge { shape: [n32 as usize, m32 as usize, 1, 1] }.into());
            }
            Some(([lhs.ld, rhs.ld, lhs.off, rhs.off], tri))
        }
        None => None,
    };
    let out_storage = Arc::new(Storage::new(ctx, len, ElemLayout::plain(4)));
    let out_pin = out_storage.pin_single()?;
    let out_bytes = out_storage.byte_len();
    let mut scratch = Scratch::new(ctx);
    let partial = if plan.slices > 1 { Some(scratch.take(u64::from(plan.slices) * out_bytes)?) } else { None };
    let (target, target_bytes) = match &partial {
        Some(buffer) => (buffer, u64::from(plan.slices) * out_bytes),
        None => (out_pin.buffer(), out_bytes),
    };
    let kernel = jit::MatmulKernel { tile: plan.tile, aligned: plan.aligned, split: plan.split, form: epilogue.map_or(0, |applied| applied.form) };
    let mut uniform = [m32, n32, k32, plan.kc, 0, 0, 0, 0, 0];
    let words = match words {
        Some((extra, tri)) => {
            uniform[4..8].copy_from_slice(&extra);
            // The kernels read the ninth word (`TriangularK`) as zero when it is not passed.
            uniform[8] = tri;
            &uniform[..8 + usize::from(tri != 0)]
        }
        None => &uniform[..4],
    };
    let out = RawArg::write(target, target_bytes);
    // The epilogue's inputs stay pinned while the kernel is recorded.
    let input_pins = match epilogue {
        Some(applied) => applied.inputs.iter().map(|input| input.pin_single_read().map(|pin| (pin, input.byte_len()))).collect::<Result<Vec<_>, _>>()?,
        None => Vec::new(),
    };
    match sides {
        sides if epilogue.is_some() => {
            let inputs: Vec<RawArg<'_>> = input_pins.iter().map(|(pin, bytes)| RawArg::read(pin.buffer(), *bytes)).collect();
            jit::matmul_epilogue(ctx, sides, kernel, words, &[], plan.groups, out, &inputs, epilogue.map_or(&[][..], |applied| &applied.scalars))?;
        }
        // Both operands evaluated: nothing to fuse, the kernel reads their buffers (what `jit::matmul` does for such sides).
        [MatmulSide::Ready(a, a_bytes), MatmulSide::Ready(b, b_bytes)] => {
            let core = ctx.builtin(Builtin::Matmul { tile: plan.tile, aligned: plan.aligned, split: plan.split }, "f32")?;
            ctx.launch(&core, plan.groups, [n32, m32, 1], &[out, RawArg::read(a.buffer(), a_bytes), RawArg::read(b.buffer(), b_bytes), RawArg::Uniform(bytemuck::cast_slice(words))])?;
        }
        sides => jit::matmul(ctx, sides, kernel, words, plan.groups, out)?,
    }
    if let Some(partial) = &partial {
        let sum = ctx.builtin(Builtin::MatmulSum, "f32")?;
        let slices = [plan.slices];
        let sum_groups = crate::ops::fold_workgroups(ctx, (m32 * n32).div_ceil(32))?;
        ctx.launch(
            &sum,
            sum_groups,
            [m32 * n32, 1, 1],
            &[RawArg::write(out_pin.buffer(), out_bytes), RawArg::read(partial, target_bytes), RawArg::Uniform(bytemuck::bytes_of(&slices))],
        )?;
    }
    drop(scratch);
    jit::check_finite(ctx, &out_storage, jit::ElemTy::F32)?;
    Ok(out_storage)
}

// -- plans --

/// Output tiles below which a long K is split (see [`split_of`]); like [`SPLIT_GROUPS`] a count
/// for the 82 multiprocessors of an RTX 3090, raised for bigger devices ([`Tuning::scaled_up`]).
const SPLIT_TILES: u32 = 128;
/// Shortest K that is split.
const SPLIT_MIN_K: u32 = 4096;
/// Workgroups a split product aims for. The 4 tiles of a 64x64 product over 2^20 take 3.0 ms
/// with 256 (3 workgroups per multiprocessor of an RTX 3090, which cannot hide the latency of
/// the slices), 2.5 with 512, 2.3 with 1024, 2.27 with 2048. Raised in proportion to the
/// multiprocessors of bigger devices ([`Tuning::scaled_up`]), never lowered: unchanged on the
/// 3090, and on an RTX 3080 Laptop (48) 1024 groups beat the proportional 599 by 2.7 %.
const SPLIT_GROUPS: u32 = 1024;
/// Smallest K a slice of a split product covers.
const SPLIT_MIN_SLICE: u32 = 1024;

/// Most outputs of a product that runs as a skinny kernel when its `K` is at least
/// [`SKINNY_LONG_K`]: a 96x96 Gram matrix over 65536 samples takes 0.29 ms on an RTX 3090 against
/// 0.38 for the staged tiles, a 128x128 one 0.35 against 0.22.
const SKINNY_MAX_OUTPUTS: u32 = 9216;
/// Most outputs for which a `K` of [`SKINNY_MIN_K`] is long enough (a 32x32 product is 2.5x
/// faster at 4096, 16 x 16 at 1M 2.4x; 64x64 over 16384 is 60% slower).
const SKINNY_SMALL_OUTPUTS: u32 = 512;
/// Shortest K of the products with up to [`SKINNY_MAX_OUTPUTS`] outputs.
const SKINNY_LONG_K: u32 = 65536;
/// Most outputs of one that reads its left operand along `K` (a matrix times a few vectors).
const SKINNY_MAX_OUTPUTS_ALONG_K: u32 = 1 << 22;
/// Shortest K of a skinny product.
const SKINNY_MIN_K: u32 = 2048;
/// Workgroups a skinny product aims for: one-hot 1M x 32 on an RTX 3090 takes 0.19 ms with 256 or
/// 1024, 0.18 with 512, 0.21 with 2048.
const SKINNY_GROUPS: u32 = 512;
/// Smallest K a workgroup of a skinny product covers: a 64x64 product over 65536 takes 0.14 ms
/// with 256 and 0.11 with 128 or 64 (at most 512 workgroups of 256 each otherwise).
const SKINNY_MIN_SLICE: u32 = 128;
/// Most accumulators an invocation of a skinny kernel holds.
const SKINNY_ACCUMULATORS: u32 = 80;
/// Most `f32` of workgroup memory a skinny kernel's reduction across groups uses (16 KiB, the
/// smallest limit of the backends).
const SKINNY_PART: u32 = 4096;
/// Tiles that read the same operand element from which a pending operand is evaluated first.
const SKINNY_EVALUATE_READS: u32 = 8;
/// Tiles that read the same element of a pending operand of several buffers from which the staged
/// tiles are preferred to a skinny kernel (below [`SKINNY_EVALUATE_READS`]).
const SKINNY_FUSED_READS: u32 = 4;

/// The slices a product of `tiles` output tiles over `k` is split into, and the elements of K
/// per slice (a multiple of the staged slice). Few tiles and a long K: about
/// [`SPLIT_GROUPS`] workgroups, each slice at least [`SPLIT_MIN_SLICE`] long. `None` if not split.
fn split_of(tuning: &Tuning, tiles: u32, k: u32) -> Option<(u32, u32)> {
    if tiles >= tuning.scaled_up(SPLIT_TILES) || k < SPLIT_MIN_K {
        return None;
    }
    let wanted = tuning.scaled_up(SPLIT_GROUPS).div_ceil(tiles).min(k / SPLIT_MIN_SLICE);
    if wanted < 2 {
        return None;
    }
    let kc = k.div_ceil(wanted).next_multiple_of(SLICE);
    let slices = k.div_ceil(kc);
    (slices >= 2).then_some((slices, kc))
}

/// The skinny kernel of an `m x n` output and the workgroups that cover its tiles: the
/// invocation tile (at most [`SKINNY_ACCUMULATORS`] outputs, up to 10 rows by 8 columns) that
/// costs the least per `k`, counting a fused multiply-add as 1 and a load as 2, over all the lanes
/// of the workgroups.
///
/// A workgroup is 64 invocations, or the next power of two of the tiles if there are more (up to
/// 256); with fewer tiles than that the surplus invocations form groups that take the `k` in turn
/// (a power of two of them, so that the groups reduce pairwise). `along_k` is for a left operand
/// stored along `K`: neighbouring invocations then take neighbouring `k` (32 or 64 groups of a
/// workgroup), and the output tiles are as many workgroups as they need; the right operand is
/// then read along `K` too if it has at most 8 columns, which is all of them.
fn skinny_tile(m: u32, n: u32, along_k: bool) -> Option<(MatmulTile, u32)> {
    let mut best: Option<((u32, u32), (MatmulTile, u32))> = None;
    let min_cols = if along_k { n } else { 1 };
    // Memory bound along `K`: a tile of at most 8 rows and 64 outputs, whose right operand's
    // columns weigh more (each is a load of its own, strided by the columns).
    let (max_rows, max_outputs) = if along_k { (8, 64) } else { (10, SKINNY_ACCUMULATORS) };
    for cols in min_cols..=n.min(8) {
        for rows in 1..=m.min(max_rows) {
            if rows * cols > max_outputs {
                continue;
            }
            let tiles = m.div_ceil(rows) * n.div_ceil(cols);
            let (width, groups) = if along_k {
                (64, if tiles >= 2 { 32 } else { 64 })
            } else {
                let width = tiles.next_power_of_two().max(64);
                if width > 256 {
                    continue;
                }
                (width, 1 << (width / tiles).ilog2())
            };
            let tile_groups = tiles.div_ceil(width / groups);
            if groups > 1 && rows * cols * width / 2 > SKINNY_PART {
                continue;
            }
            let loads = 2 * rows + if along_k && cols > 1 { 2 * cols * cols } else { 2 * cols };
            let cost = tile_groups * (width / groups) * (rows * cols + loads);
            let key = (cost, u32::MAX - rows * cols);
            if best.is_none_or(|(k, _)| key < k) {
                let tile = MatmulTile::Skinny { rows: rows as u8, cols: cols as u8, groups: groups as u8, width: width as u16, along_k };
                best = Some((key, (tile, tile_groups)));
            }
        }
    }
    best.map(|(_, tile)| tile)
}

/// Most rows of a skinny product of more than 8 columns: above, the staged tiles are faster (`Xᵀ R` of
/// 1M samples, 128 features and 20 classes on an RTX 3090: 0.78 ms tiled against 1.30 skinny; 128 x 32
/// 0.81 against 1.53; 64 x 20 1.15 tiled against 0.69 skinny, 64 x 64 2.26 against 1.53).
const SKINNY_MAX_ROWS: u32 = 96;

/// The plan of a small output over a long K (a Gram matrix, one-hot labels times features, a
/// gradient `Xᵀ r`, a matrix times a few vectors) as a skinny kernel: `slices` of K, each its
/// own workgroups, about [`SKINNY_GROUPS`] workgroups in all, no slice shorter than
/// [`SKINNY_MIN_SLICE`]. `along_k`: the left operand is stored along `K` (see [`skinny_tile`]).
/// `None` if the product is not of that kind.
fn skinny_plan(m: u32, n: u32, k: u32, along_k: bool, max_groups: u32, max_bytes: u64) -> Option<Plan> {
    let along_k = along_k && n <= 8;
    let outputs = u64::from(m) * u64::from(n);
    let few = if along_k {
        outputs <= u64::from(SKINNY_MAX_OUTPUTS_ALONG_K)
    } else {
        (outputs <= u64::from(SKINNY_SMALL_OUTPUTS) || (outputs <= u64::from(SKINNY_MAX_OUTPUTS) && k >= SKINNY_LONG_K)) && (m <= SKINNY_MAX_ROWS || n <= 8)
    };
    if m == 0 || n == 0 || k < SKINNY_MIN_K || !few {
        return None;
    }
    let (tile, tile_groups) = skinny_tile(m, n, along_k)?;
    let mut wanted = SKINNY_GROUPS.div_ceil(tile_groups).min(k / SKINNY_MIN_SLICE).clamp(1, max_groups);
    loop {
        let kc = k.div_ceil(wanted);
        let slices = k.div_ceil(kc);
        if wanted == 1 || u64::from(slices) * u64::from(m) * u64::from(n) * 4 <= max_bytes {
            if tile_groups > max_groups {
                return None;
            }
            return Some(Plan { tile, aligned: false, split: true, slices, kc, groups: [slices, tile_groups, 1] });
        }
        wanted /= 2;
    }
}

/// The plan of an `m x k` by `k x n` product. `along_k`: the left operand's elements along `K` are
/// neighbours in memory (not a transposed view). `max_groups` is the most workgroups per grid
/// dimension, `max_bytes` the most a buffer may hold (the split product's partial sums have to
/// fit one).
#[cfg(test)]
fn plan(tuning: &Tuning, m: u32, n: u32, k: u32, along_k: bool, max_groups: u32, max_bytes: u64) -> Result<Plan, LumaError> {
    if let Some(plan) = skinny_plan(m, n, k, along_k, max_groups, max_bytes) {
        return Ok(plan);
    }
    tiled_plan(tuning, m, n, k, max_groups, max_bytes)
}

/// [`plan`] with the staged tiles (never a skinny kernel).
fn tiled_plan(tuning: &Tuning, m: u32, n: u32, k: u32, max_groups: u32, max_bytes: u64) -> Result<Plan, LumaError> {
    let tile = match (m >= 128, n >= 128) {
        (true, true) => MatmulTile::T128,
        (true, false) => MatmulTile::T128x32,
        (false, true) => MatmulTile::T32x128,
        (false, false) => MatmulTile::T32,
    };
    let (row_edge, col_edge) = tile.edges();
    let aligned = m.is_multiple_of(row_edge) && n.is_multiple_of(col_edge) && k.is_multiple_of(SLICE);
    let (columns, rows) = (n.div_ceil(col_edge), m.div_ceil(row_edge));
    let tiles = columns.saturating_mul(rows);
    let (slices, kc) = split_of(tuning, tiles, k)
        .filter(|&(slices, _)| u64::from(slices) * u64::from(m) * u64::from(n) * 4 <= max_bytes)
        .unwrap_or((1, k));
    // Row-major tile numbers (see `body`): the grid is `[columns, rows]` if that fits, else the
    // tiles are folded over it, the groups past the last tile exiting at once.
    let [x, y, _] = if columns <= max_groups && rows <= max_groups { [columns, rows, 1] } else { fold_grid(tiles, max_groups) };
    let groups = [x, y, slices];
    if groups.iter().any(|&count| count > max_groups) {
        return Err(LumaError::KernelGridTooLarge { workgroups: groups, max: max_groups });
    }
    Ok(Plan { tile, aligned, split: slices > 1, slices, kc, groups })
}

/// Output tile edges a native product chooses from, largest first (see [`native_plan`]).
const NATIVE_EDGES: [u32; 3] = [128, 64, 32];
/// Output tiles per core a native product aims for: the cores take them as they free up.
const NATIVE_TILES_PER_CORE: u32 = 2;
/// Smallest K a slice of a split native product covers.
const NATIVE_SPLIT_SLICE: u32 = 1024;

/// The plan of an `m x k` by `k x n` product the backend multiplies as native code on `cores`
/// cores: the largest tile of [`NATIVE_EDGES`] that leaves [`NATIVE_TILES_PER_CORE`] tiles per
/// core, else the smallest; with fewer tiles than that and a long K, K is split over the third
/// grid dimension like [`split_of`] does (partial products summed by a second launch).
fn native_plan(m: u32, n: u32, k: u32, cores: u32, transposed: [bool; 2], max_groups: u32, max_bytes: u64) -> Result<Plan, LumaError> {
    let wanted = cores.max(1).saturating_mul(NATIVE_TILES_PER_CORE);
    let tiles_at = |edge: u32| m.div_ceil(edge).saturating_mul(n.div_ceil(edge));
    let edge = NATIVE_EDGES.into_iter().find(|&edge| tiles_at(edge) >= wanted).unwrap_or(NATIVE_EDGES[NATIVE_EDGES.len() - 1]);
    let (columns, rows) = (n.div_ceil(edge), m.div_ceil(edge));
    let tiles = columns.saturating_mul(rows);
    let split = (tiles < wanted && k >= 2 * NATIVE_SPLIT_SLICE).then(|| {
        let kc = k.div_ceil(wanted.div_ceil(tiles).min(k / NATIVE_SPLIT_SLICE)).next_multiple_of(SLICE);
        (k.div_ceil(kc), kc)
    });
    let (slices, kc) = split
        .filter(|&(slices, _)| slices >= 2 && u64::from(slices) * u64::from(m) * u64::from(n) * 4 <= max_bytes)
        .unwrap_or((1, k));
    let [x, y, _] = if columns <= max_groups && rows <= max_groups { [columns, rows, 1] } else { fold_grid(tiles, max_groups) };
    let groups = [x, y, slices];
    if groups.iter().any(|&count| count > max_groups) {
        return Err(LumaError::KernelGridTooLarge { workgroups: groups, max: max_groups });
    }
    Ok(Plan { tile: MatmulTile::Native { edge, lhs_t: transposed[0], rhs_t: transposed[1] }, aligned: false, split: slices > 1, slices, kc, groups })
}

/// `LUMA_OWN_GEMM=off` keeps the `f32` products on the shader kernels even where the backend has
/// its own (see [`run`]).
static OWN_F32_KERNELS_OFF: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| std::env::var("LUMA_OWN_GEMM").is_ok_and(|v| v.trim() == "off"));

/// Tiles per multiprocessor a tensor-core product aims for.
const GEMM_HALF_TILES_PER_SM: u32 = 4;
/// Smallest `M` and `K` of a product of the backend's own kernels (below it the tiles are mostly
/// padding), and of its `N`, except for a product of at most [`SMALL_N_MAX_M`] rows, whose few
/// columns (the classes of a classifier head) fill part of a 32-wide tile but still beat the
/// shader kernels, which launch two workgroups for them.
const GEMM_MIN_EXTENT: usize = 32;
/// Smallest `N` of a product of at most [`SMALL_N_MAX_M`] rows.
const SMALL_N_MIN: usize = 4;
/// Most rows of a product with `N` below [`GEMM_MIN_EXTENT`] that runs on the backend's own
/// kernels (more rows have enough tiles for the shader kernels' skinny and staged plans).
const SMALL_N_MAX_M: usize = 4096;
/// Fewest rows of such a product: with 64 rows the extra launches (the padded operand) cost more than the shader kernels
/// (`train/sgd_small`, `adam_small` of the `ml` suite, batch 64, 11 % slower on an RTX 3080 Laptop).
const SMALL_N_MIN_M: usize = 128;

/// The `M`, `N` and `K` of the product of `lhs` and `rhs` if the backend's aligned kernels take its
/// shapes: all multiples of 4 and at least [`GEMM_MIN_EXTENT`] (the extents are checked again by
/// [`run`]).
fn aligned_shape(lhs: &Array<f32>, rhs: &Array<f32>) -> Option<[u32; 3]> {
    let (ld, rd) = (lhs.shape().dims(), rhs.shape().dims());
    let ok = |extent: usize, least: usize| u32::try_from(extent).ok().filter(|_| extent >= least && extent.is_multiple_of(4));
    if ld[0] != rd[1] || ld[2..].iter().chain(&rd[2..]).any(|&extent| extent != 1) {
        return None;
    }
    let few_columns = if (SMALL_N_MIN_M..=SMALL_N_MAX_M).contains(&ld[1]) { SMALL_N_MIN } else { GEMM_MIN_EXTENT };
    Some([ok(ld[1], GEMM_MIN_EXTENT)?, ok(rd[0], few_columns)?, ok(ld[0], GEMM_MIN_EXTENT)?])
}

/// Whether the backend's own kernels can take `array` as an operand: they read buffers (or the
/// transpose of one), so an operand that is an expression is evaluated first, once, which beats
/// fusing it into the old shader kernels (`relu(z) W` of 256 x 512 x 512 takes 85 us fused and 3 + 20
/// evaluated first on an RTX 3090) and lets its other readers use the buffer. Arrays split over
/// several buffers are left to the shader kernels.
fn gemm_operand(ctx: &Context, array: &Array<f32>) -> bool {
    array.pending().is_none_or(|node| node.len() as u64 * 4 <= ctx.segment_bytes())
}

/// Output tiles below which a long K of an `f32` product of the backend's own kernels is split
/// (a count for the 82 multiprocessors of an RTX 3090, see [`Tuning::scaled_up`]), the workgroups
/// it aims for and the shortest slice. `32 x 4096 x 4096` (the weights of a small batch) takes
/// 0.16 ms split in 4 slices of 1024 by the shader kernels and 0.23 ms in 4 of 32 x 64 tiles;
/// `128 x 262144 x 128` 0.63 ms with 1024 workgroups, 1.4 with 124.
const F32_SPLIT_TILES: u32 = 128;
const F32_SPLIT_GROUPS: u32 = 1024;
/// Longest slice of K a split product of the backend's own kernels covers at least: an eighth of K
/// between [`GEMM_SPLIT_SLICE_MIN`] and `..._MAX` (measured on an RTX 3090 for `f32` / TF32: 256 x 512 x 12
/// 18.4 / 19.2 us unsplit, 9.2 in 4 slices of 128; 2048 x 2048 x 64 95 us unsplit, 70 in 8 slices of
/// 256, 83 in 16 of 128; 128 x 4096 x 4096 236 us in 8 slices of 512, 258 in 16 of 256; 64 x 8192 x 64
/// TF32 34 us unsplit, 10 in slices of 128; 256 x 784 x 512 28 / 28 us, 24 / 20 split).
const GEMM_SPLIT_SLICE_DIV: u32 = 8;
const GEMM_SPLIT_SLICE_MIN: u32 = 128;
const F32_SPLIT_SLICE_MAX: u32 = 512;
const GEMM_TF32_SPLIT_SLICE_MAX: u32 = 1024;
/// Shortest K that is split: 512 for a product of at most [`GEMM_SPLIT_FEW_TILES`] output tiles (a
/// few workgroups would idle most of the device and the fold of the slices is cheap against their
/// share of K), else 4096 (`f32`) / 2048 and slices of 1024 (TF32): a product of 32 to 128 tiles over
/// a K of 512 to 2048 is 5 to 40 % slower split (256 x 512 x 512 20 us unsplit, 24 split; 256 x 1024 x
/// 1024 TF32 35 us, 44 split).
const GEMM_SPLIT_MIN_K: u32 = 512;
const GEMM_SPLIT_FEW_TILES: u32 = 16;
const F32_SPLIT_MIN_K: u32 = 4096;

/// The slices an `f32` product of `tiles` output tiles over `k` is split into and their length.
fn f32_split(tuning: &Tuning, tiles: u32, k: u32, bytes: u64, max_bytes: u64) -> Option<(u32, u32)> {
    let min_k = if tiles <= tuning.scaled_up(GEMM_SPLIT_FEW_TILES) { GEMM_SPLIT_MIN_K } else { F32_SPLIT_MIN_K };
    if tiles >= tuning.scaled_up(F32_SPLIT_TILES) || k < min_k {
        return None;
    }
    let slice = (k / GEMM_SPLIT_SLICE_DIV).clamp(GEMM_SPLIT_SLICE_MIN, F32_SPLIT_SLICE_MAX);
    let wanted = tuning.scaled_up(F32_SPLIT_GROUPS).div_ceil(tiles).min(k / slice);
    if wanted < 2 {
        return None;
    }
    let kc = k.div_ceil(wanted).next_multiple_of(SLICE);
    let slices = k.div_ceil(kc);
    (slices >= 2 && u64::from(slices) * bytes <= max_bytes).then_some((slices, kc))
}

/// The rate of an `f32` kernel of `tile` (relative to the largest, at full occupancy) times the
/// share of the device `workgroups` of it fill and the share of its outputs that are not padding.
/// Measured on an RTX 3090 at 4096^3: 64 x 128 22.9 TFLOP/s, 128 x 64 and 64 x 64 21.8, 64 x 32 and
/// 32 x 64 16.5-17.3, 32 x 32 13.8 (an RTX 3080 Laptop: 9.3, 8.3-8.8, 6.0-6.7, 5.2); a workgroup is
/// 4 warps up to 128 x 64, else 2, and it takes about 8 resident warps per multiprocessor to keep
/// the multiply-add units busy (a 1024^3 product with 128 tiles of 64 x 128 runs at 18.5, with 256
/// of 64 x 64 at 16.8, with 256 of 32 x 32 at 12.5, with 64 of 64 x 128 at 6.5).
fn f32_tile_rate(tile: (u32, u32), workgroups: u32, units: u32, useful: f64) -> f64 {
    let area = tile.0 * tile.1;
    let (rate, warps) = match area {
        8192.. => (1.0, 4.0),
        4096.. => (0.95, 2.0),
        2048.. => (0.74, 2.0),
        _ => (0.6, 2.0),
    };
    rate * useful * (f64::from(workgroups) * warps / (f64::from(units.max(1)) * 8.0)).min(1.0)
}

/// The plan of an `m x k` by `k x n` product on the backend's own kernels (tensor cores, or with
/// `exact` the ordinary cores) with one of `tiles` (largest first). Tensor-core tiles: the largest
/// that gives `half_per_sm` half tiles per multiprocessor, else the smallest; few tiles over a
/// long K split K (see [`native_plan`]). `f32` tiles: the one with the best rate, K split as
/// [`f32_split`] says (see [`f32_tile_rate`]).
#[allow(clippy::too_many_arguments)]
fn gemm_plan(m: u32, n: u32, k: u32, tuning: &Tuning, half_per_sm: u32, tiles: &[(u32, u32)], transposed: [bool; 2], exact: bool, max_groups: u32, max_bytes: u64) -> Result<Plan, LumaError> {
    let units = tuning.compute_units;
    let wanted = units.max(1).saturating_mul(half_per_sm).div_ceil(2);
    let count = |&(rows, cols): &(u32, u32)| m.div_ceil(rows).saturating_mul(n.div_ceil(cols));
    let out_bytes = u64::from(m) * u64::from(n) * 4;
    let pick = || {
        if !exact {
            return tiles.iter().find(|tile| count(tile) >= wanted).or(tiles.last()).copied();
        }
        tiles
            .iter()
            .copied()
            .fold(None::<((u32, u32), f64)>, |best, tile| {
                let tile_count = count(&tile);
                let slices = f32_split(tuning, tile_count, k, out_bytes, max_bytes).map_or(1, |(slices, _)| slices);
                let useful = f64::from(m) * f64::from(n) / (f64::from(tile_count) * f64::from(tile.0 * tile.1));
                let rate = f32_tile_rate(tile, tile_count.saturating_mul(slices), units, useful);
                if best.is_none_or(|(_, top)| rate > top * 1.0001) { Some((tile, rate)) } else { best }
            })
            .map(|(tile, _)| tile)
    };
    let (rows, cols) = pick().unwrap_or((128, 128));
    let (columns, row_tiles) = (n.div_ceil(cols), m.div_ceil(rows));
    let tile_count = columns.saturating_mul(row_tiles);
    let split = if exact {
        f32_split(tuning, tile_count, k, out_bytes, max_bytes)
    } else {
        let few = tile_count <= tuning.scaled_up(GEMM_SPLIT_FEW_TILES);
        let slice = if few { (k / GEMM_SPLIT_SLICE_DIV).clamp(GEMM_SPLIT_SLICE_MIN, GEMM_TF32_SPLIT_SLICE_MAX) } else { NATIVE_SPLIT_SLICE };
        let min_k = if few { GEMM_SPLIT_MIN_K } else { 2 * NATIVE_SPLIT_SLICE };
        (tile_count < wanted && k >= min_k)
            .then(|| {
                let kc = k.div_ceil(wanted.div_ceil(tile_count).min(k / slice)).next_multiple_of(SLICE);
                (k.div_ceil(kc), kc)
            })
            .filter(|&(slices, _)| slices >= 2 && u64::from(slices) * out_bytes <= max_bytes)
    };
    let (slices, kc) = split.unwrap_or((1, k));
    let groups = [columns, row_tiles, slices];
    if groups.iter().any(|&count| count > max_groups) {
        return Err(LumaError::KernelGridTooLarge { workgroups: groups, max: max_groups });
    }
    Ok(Plan { tile: MatmulTile::Gemm { rows, cols, lhs_t: transposed[0], rhs_t: transposed[1], exact }, aligned: false, split: slices > 1, slices, kc, groups })
}

/// A matrix operand that is a window of a stored matrix: the source buffer (or the expression that computes it), where the matrix
/// starts in it (`offset`, in elements), the leading dimension of the source (`ld`) and whether the buffer holds the transpose.
pub(crate) struct MatrixWindow {
    pub(crate) src: Src,
    pub(crate) offset: u32,
    pub(crate) ld: u32,
    pub(crate) transposed: bool,
}

/// The window of a stored matrix that `array` is, if it is one: a view of a buffer (or of a pending expression) that keeps the two
/// axes, in order or swapped, each stepping by one, with a start on each, so that a product reads the buffer through a leading
/// dimension and an offset instead of the view being copied (a block of a matrix, its transpose, a block of a block: the maps of
/// views compose). A full swap with no offset is the special case this replaced. Pure: nothing is evaluated.
pub(crate) fn matrix_window(array: &Array<f32>) -> Option<MatrixWindow> {
    let node = array.pending()?;
    let op = node.op();
    let OpKind::Shift(ShiftKind::View { map, edge: Edge::None }) = op.kind else { return None };
    let unit = |from: u8| AxisMap { from, start: 0, step: 1 };
    // A reshape of a pending view is another handle on the same node (see `Array::reshape`): the node still has the shape it
    // was made with, so the window is the array's only if the shapes agree.
    if array.shape().dims().map(|extent| u32::try_from(extent).ok()) != map.out.map(Some) {
        return None;
    }
    if op.ty != ElemTy::F32 || map.axes[2] != unit(2) || map.axes[3] != unit(3) || map.src[2..] != [1, 1] || map.out[2..] != [1, 1] {
        return None;
    }
    // `axes[i]` is source axis `i`: it reads the output axis `from`.
    let [first, second] = [map.axes[0], map.axes[1]];
    let transposed = match (first.from, second.from) {
        (0, 1) => false,
        (1, 0) => true,
        _ => return None,
    };
    if first.step != 1 || second.step != 1 || first.start < 0 || second.start < 0 || node.failure().is_some() || node.is_looped() {
        return None;
    }
    let offset = u32::try_from(i64::from(first.start) + i64::from(map.src[0]) * i64::from(second.start)).ok()?;
    let [source, ..] = node.operands()?;
    matches!(source, Src::Buffer(_) | Src::Node(_)).then_some(MatrixWindow { src: source, offset, ld: map.src[0], transposed })
}

/// The operands of a product that is multiplied natively, with how each sits in its buffer: the native product reads buffers, so
/// the operands that are expressions (or the expressions a window reads) are evaluated first, together, sharing what they read; a
/// window of a stored matrix is read where it is, through its leading dimension and offset.
fn native_sides(ctx: &Context, lhs: &Array<f32>, rhs: &Array<f32>) -> Outcome<([MatmulSide; 2], [Layout; 2])> {
    // A poisoned operand reports its own error.
    let windows = [matrix_window(lhs), matrix_window(rhs)];
    for (array, window) in [lhs, rhs].into_iter().zip(&windows) {
        match (window.as_ref().map(|window| &window.src), array.pending()) {
            (Some(Src::Node(node)), _) if node.is_pending() => array.context().jit().request(node),
            (Some(_), _) => {}
            (None, Some(node)) => array.context().jit().request(node),
            (None, None) => drop(array.ready()?),
        }
    }
    if !ctx.same_as(rhs.context()) {
        return Err(LumaError::ContextMismatch.into());
    }
    ctx.jit().evaluate_requested();
    let [lhs_window, rhs_window] = windows;
    let (lhs_side, lhs_layout) = native_side(lhs, lhs_window)?;
    let (rhs_side, rhs_layout) = native_side(rhs, rhs_window)?;
    Ok(([lhs_side, rhs_side], [lhs_layout, rhs_layout]))
}

/// One operand of [`native_sides`], once what it reads is evaluated.
fn native_side(array: &Array<f32>, window: Option<MatrixWindow>) -> Outcome<(MatmulSide, Layout)> {
    let (storage, layout) = match window {
        Some(MatrixWindow { src: Src::Buffer(storage), offset, ld, transposed }) => (storage, Layout { transposed, ld, off: offset }),
        Some(MatrixWindow { src: Src::Node(node), offset, ld, transposed }) => {
            (Arc::clone(node.force().as_ref().map_err(|error| Failure::Inherited(Arc::clone(error)))?), Layout { transposed, ld, off: offset })
        }
        _ => {
            let (pin, bytes) = array.ready()?;
            return Ok((MatmulSide::Ready(pin, bytes), Layout { transposed: false, ld: first_axis(array), off: 0 }));
        }
    };
    Ok((MatmulSide::Ready(storage.pin_single_read()?, storage.byte_len()), layout))
}

/// The side of the product `array` stands for: its expression if it is pending (so it may be
/// fused), else its buffer.
fn side(ctx: &Context, array: &Array<f32>) -> Outcome<MatmulSide> {
    match array.pending() {
        Some(node) => {
            if node.len() as u64 * 4 > ctx.segment_bytes() {
                return Err(LumaError::Chunked("a matrix product of an array split over several buffers").into());
            }
            Ok(MatmulSide::Pending(Arc::clone(node)))
        }
        None => {
            let (pin, bytes) = array.ready()?;
            Ok(MatmulSide::Ready(pin, bytes))
        }
    }
}

/// The stable label of a plan, for the decision log.
fn plan_label(plan: &Plan) -> String {
    let tile = match plan.tile {
        MatmulTile::T128 => "tiled 128x128".to_string(),
        MatmulTile::T128Half => "syrk tiled 128x128".to_string(),
        MatmulTile::T128x32 => "tiled 128x32".to_string(),
        MatmulTile::T32x128 => "tiled 32x128".to_string(),
        MatmulTile::T32 => "tiled 32x32".to_string(),
        MatmulTile::Native { edge, .. } => format!("native {edge}"),
        MatmulTile::NativeHalf { edge, .. } => format!("syrk native {edge}"),
        MatmulTile::GemmHalf { rows, cols, exact, .. } => format!("syrk {rows}x{cols}{}", if exact { " exact" } else { "" }),
        MatmulTile::Gemm { rows, cols, exact, .. } => format!("gemm {rows}x{cols}{}", if exact { " exact" } else { "" }),
        MatmulTile::GemmRagged { rows, cols, exact, lanes, .. } => {
            format!("gemm {rows}x{cols}{} {}", if exact { " exact" } else { "" }, if lanes == GemmLanes::Scalar { "ragged" } else { "padded" })
        }
        MatmulTile::Skinny { rows, cols, groups, width, along_k } => format!("skinny {rows}x{cols} groups={groups} width={width}{}", if along_k { " along-k" } else { "" }),
    };
    format!("{tile} split={}{}", plan.slices, if plan.aligned { " aligned" } else { "" })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_long_product_of_few_tiles_is_split_in_slices_of_whole_k_steps() {
        let t = Tuning::for_tests();
        // The Gram matrix of 64 features over 2^20 samples: 4 tiles of 32x32.
        let (slices, kc) = split_of(&t, 4, 1 << 20).unwrap();
        assert_eq!((slices, kc), (256, 4096));
        for k in [4096u32, 5000, 100_000, 1 << 20] {
            for tiles in [1u32, 4, 70, 127] {
                let Some((slices, kc)) = split_of(&t, tiles, k) else { continue };
                assert!(slices >= 2 && kc % SLICE == 0, "{tiles} tiles, k {k}");
                assert_eq!(slices, k.div_ceil(kc), "no empty slice");
            }
        }
        assert_eq!(split_of(&t, 128, 1 << 20), None);
        assert_eq!(split_of(&t, 4, 4095), None);
        assert_eq!(split_of(&t, 4, 4096), Some((4, 1024)));
    }

    #[test]
    fn the_split_aims_for_a_device_full_of_workgroups() {
        // The reference device (82 multiprocessors) keeps the constants measured on it.
        let reference = Tuning { compute_units: 82, ..Tuning::for_tests() };
        assert_eq!(split_of(&reference, 4, 1 << 20), Some((256, 4096)));
        assert_eq!(split_of(&reference, 127, 1 << 20), Some((9, 116_512)));
        assert_eq!(split_of(&reference, 128, 1 << 20), None);
        // A smaller device keeps them (measured on an RTX 3080 Laptop, 48 multiprocessors).
        let laptop = Tuning { compute_units: 48, ..Tuning::for_tests() };
        assert_eq!(split_of(&laptop, 4, 1 << 20), Some((256, 4096)));
        assert_eq!(split_of(&laptop, 128, 1 << 20), None);
        // A bigger one more.
        let big = Tuning { compute_units: 128, ..Tuning::for_tests() };
        assert_eq!(split_of(&big, 4, 1 << 20).map(|s| s.0), Some(400));
        assert!(split_of(&big, 199, 1 << 20).is_some());
    }

    #[test]
    fn the_plan_picks_the_tile_alignment_and_grid() {
        let t = Tuning::for_tests();
        let p = plan(&t, 1024, 2048, 1024, false, 65535, u64::MAX).unwrap();
        assert_eq!((p.tile, p.aligned, p.slices, p.groups), (MatmulTile::T128, true, 1, [16, 8, 1]));
        let p = plan(&t, 1000, 1000, 1000, false, 65535, u64::MAX).unwrap();
        assert_eq!((p.tile, p.aligned, p.groups), (MatmulTile::T128, false, [8, 8, 1]));
        let p = plan(&t, 64, 64, 1024, false, 65535, u64::MAX).unwrap();
        assert_eq!((p.tile, p.aligned, p.slices, p.groups), (MatmulTile::T32, true, 1, [2, 2, 1]));
        // Tall products stay on the rectangular tile; the grid folds once row tiles exceed it.
        let p = plan(&t, 1 << 22, 8, 16, false, 65535, u64::MAX).unwrap();
        assert_eq!((p.tile, p.groups), (MatmulTile::T128x32, [1, 1 << 15, 1]));
        let p = plan(&t, 8, 1 << 22, 16, false, 65535, u64::MAX).unwrap();
        assert_eq!((p.tile, p.groups), (MatmulTile::T32x128, [1 << 15, 1, 1]));
        let p = plan(&t, 1 << 22, 1 << 6, 16, false, 65535, u64::MAX).unwrap();
        assert_eq!(p.tile, MatmulTile::T128x32);
        let p = plan(&t, 1 << 22, 8, 16, false, 1 << 12, u64::MAX).unwrap();
        assert!(p.groups[0] * p.groups[1] >= 1 << 15);
        assert!(plan(&t, 1 << 22, 64, 16, false, 65535, u64::MAX).is_ok());
    }

    #[test]
    fn a_small_output_over_a_long_k_is_a_skinny_product() {
        let skinny = |m, n, k, along_k, max_bytes| plan(&Tuning::for_tests(), m, n, k, along_k, 65535, max_bytes).unwrap();
        // One-hot labels times 32 features: 4 tiles of 10x8, 16 groups of them in a workgroup.
        let p = skinny(10, 32, 1_000_000, false, u64::MAX);
        assert_eq!(p.tile, MatmulTile::Skinny { rows: 10, cols: 8, groups: 16, width: 64, along_k: false });
        assert!(p.split && !p.aligned && p.groups[1] == 1 && p.groups[0] == p.slices && p.slices * p.kc >= 1_000_000);
        // A Gram matrix of 64 features: 64 tiles of 8x8, one group.
        let p = skinny(64, 64, 1 << 20, false, u64::MAX);
        assert_eq!(p.tile, MatmulTile::Skinny { rows: 8, cols: 8, groups: 1, width: 64, along_k: false });
        // A dot product: one tile, every invocation of the workgroup a group.
        let p = skinny(1, 1, 1 << 20, true, u64::MAX);
        assert_eq!(p.tile, MatmulTile::Skinny { rows: 1, cols: 1, groups: 64, width: 64, along_k: true });
        // Slices are whole `k` and none is empty.
        for (m, n, k) in [(10, 32, 100_003), (1, 1, 2048), (64, 64, 70_000), (33, 7, 1 << 20), (2000, 1, 70_001)] {
            for along_k in [false, true] {
                let p = skinny(m, n, k, along_k, u64::MAX);
                assert!(matches!(p.tile, MatmulTile::Skinny { .. }), "{m} {n} {k}");
                assert_eq!(p.slices, k.div_ceil(p.kc), "no empty slice");
                assert!(p.slices >= 1 && p.groups[0] == p.slices);
            }
        }
        // The partial sums must fit a buffer.
        assert_eq!(skinny(64, 64, 1 << 20, false, 1000).slices, 1);
        // A matrix times a vector reads the matrix along `K`: its row tiles are workgroups.
        let p = skinny(8192, 1, 8192, true, u64::MAX);
        assert!(matches!(p.tile, MatmulTile::Skinny { along_k: true, .. }));
        assert!(p.groups[1] > 1);
        // Not skinny: too short, or too many outputs.
        assert!(!matches!(skinny(10, 32, 1000, false, u64::MAX).tile, MatmulTile::Skinny { .. }));
        assert!(!matches!(skinny(256, 256, 1 << 20, false, u64::MAX).tile, MatmulTile::Skinny { .. }));
        assert!(!matches!(skinny(64, 64, 16384, false, u64::MAX).tile, MatmulTile::Skinny { .. }));
        assert!(!matches!(skinny(128, 128, 1 << 20, false, u64::MAX).tile, MatmulTile::Skinny { .. }));
        // More than 96 rows of more than 8 columns (a softmax gradient of 128 features and 20 classes) take the staged tiles.
        assert!(!matches!(skinny(128, 20, 1 << 20, false, u64::MAX).tile, MatmulTile::Skinny { .. }));
        assert!(matches!(skinny(64, 20, 1 << 20, false, u64::MAX).tile, MatmulTile::Skinny { .. }));
        assert!(matches!(skinny(128, 8, 1 << 20, false, u64::MAX).tile, MatmulTile::Skinny { .. }));
    }

    fn native(edge: u32) -> MatmulTile {
        MatmulTile::Native { edge, lhs_t: false, rhs_t: false }
    }

    #[test]
    fn a_native_plan_keeps_the_cores_busy() {
        let plan = |m, n, k, cores| native_plan(m, n, k, cores, [false; 2], 65535, u64::MAX).unwrap();
        // The largest tile that leaves two tiles per core.
        let p = plan(4096, 4096, 4096, 32);
        assert_eq!((p.tile, p.slices, p.groups), (native(128), 1, [32, 32, 1]));
        let p = plan(512, 512, 512, 32);
        assert_eq!((p.tile, p.groups), (native(64), [8, 8, 1]));
        let p = plan(100, 100, 100, 32);
        assert_eq!((p.tile, p.slices, p.groups), (native(32), 1, [4, 4, 1]));
        // Few tiles and a long K: sliced, every slice a multiple of 16 that covers K.
        let p = plan(64, 64, 4099, 32);
        assert_eq!((p.tile, p.groups[..2].to_vec()), (native(32), vec![2, 2]));
        assert!(p.slices >= 2 && p.kc % SLICE == 0 && p.slices * p.kc >= 4099 && (p.slices - 1) * p.kc < 4099, "{p:?}");
        assert_eq!(plan(64, 64, 1 << 20, 1).slices, 1);
        // The partial sums must fit a buffer.
        assert_eq!(native_plan(64, 64, 1 << 20, 32, [false; 2], 65535, 1000).unwrap().slices, 1);
        // The transposed operands are the tile's.
        let p = native_plan(512, 512, 512, 32, [true, false], 65535, u64::MAX).unwrap();
        assert_eq!(p.tile, MatmulTile::Native { edge: 64, lhs_t: true, rhs_t: false });
        // Tall products fold over the grid.
        let p = native_plan(1 << 22, 8, 16, 32, [false; 2], 1 << 12, u64::MAX).unwrap();
        assert!(p.groups[0] <= 1 << 12 && p.groups[1] <= 1 << 12 && p.groups[0] * p.groups[1] >= 1 << 15, "{p:?}");
    }

    #[test]
    fn a_tensor_plan_picks_the_largest_tile_that_fills_the_device() {
        const TILES: [(u32, u32); 4] = [(128, 128), (128, 64), (64, 128), (64, 64)];
        let plan = |m, n, k, transposed| gemm_plan(m, n, k, &Tuning::for_tests(), 4, &TILES, transposed, false, 65535, u64::MAX).unwrap();
        let tensor = |rows, cols| MatmulTile::Gemm { rows, cols, lhs_t: false, rhs_t: false, exact: false };
        // Two tiles per multiprocessor, the largest tile that gets them.
        let p = plan(4096, 4096, 4096, [false; 2]);
        assert_eq!((p.tile, p.slices, p.groups), (tensor(128, 128), 1, [32, 32, 1]));
        let p = plan(2048, 2048, 2048, [false; 2]);
        assert_eq!((p.tile, p.groups), (tensor(128, 128), [16, 16, 1]));
        let p = plan(1024, 1024, 1024, [false; 2]);
        assert_eq!((p.tile, p.groups), (tensor(64, 64), [16, 16, 1]));
        // Never fewer than the smallest tile: `[columns, rows, slices]`, with columns along `N`.
        let p = plan(256, 1024, 1024, [false; 2]);
        assert_eq!((p.tile, p.groups), (tensor(64, 64), [16, 4, 1]));
        // Few tiles and a long K: sliced in whole stages, every slice non-empty.
        let p = plan(64, 64, 1 << 20, [false; 2]);
        assert_eq!((p.tile, p.groups[..2].to_vec()), (tensor(64, 64), vec![1, 1]));
        assert!(p.slices >= 2 && p.kc % SLICE == 0 && p.slices * p.kc >= 1 << 20 && (p.slices - 1) * p.kc < 1 << 20, "{p:?}");
        // The partial sums must fit a buffer.
        assert_eq!(gemm_plan(64, 64, 1 << 20, &Tuning::for_tests(), 4, &TILES, [false; 2], false, 65535, 1000).unwrap().slices, 1);
        // The transposed operands are the tile's.
        let p = plan(4096, 4096, 4096, [true, false]);
        assert_eq!(p.tile, MatmulTile::Gemm { rows: 128, cols: 128, lhs_t: true, rhs_t: false, exact: false });
    }

    #[test]
    fn an_f32_gemm_plan_picks_the_tile_with_the_best_predicted_rate() {
        const TILES: [(u32, u32); 6] = [(64, 128), (128, 64), (64, 64), (64, 32), (32, 64), (32, 32)];
        let plan = |m, n, k, units: u32| gemm_plan(m, n, k, &Tuning { compute_units: units, ..Tuning::for_tests() }, GEMM_HALF_TILES_PER_SM, &TILES, [false; 2], true, 65535, u64::MAX).unwrap();
        let tile = |rows, cols| MatmulTile::Gemm { rows, cols, lhs_t: false, rhs_t: false, exact: true };
        assert_eq!(plan(4096, 4096, 4096, 82).tile, tile(64, 128));
        // 128 tiles of 64 x 128 fill 78 % of the device, 256 of 64 x 64 78 % at 95 % of the rate.
        let p = plan(1024, 1024, 1024, 82);
        assert_eq!((p.tile, p.groups), (tile(64, 128), [8, 16, 1]));
        // On 48 multiprocessors the same product gets the same tile; 256 x 1024 outputs take the smallest.
        assert_eq!(plan(1024, 1024, 1024, 48).tile, tile(64, 128));
        assert_eq!(plan(256, 1024, 1024, 82).tile, tile(32, 32));
        // Few output tiles over a K of 512 or more are split too (an attention `P V` of 2048 x 2048 x 64 takes 95 us unsplit, 70 in 8 slices of 256);
        // a short K over more tiles is not (256 x 1024 x 1024 has 256 tiles of 32 x 32).
        let p = plan(2048, 64, 2048, 82);
        assert_eq!((p.tile, p.slices), (tile(128, 64), 8));
        assert_eq!(plan(256, 1024, 1024, 82).slices, 1);
        assert_eq!(plan(256, 12, 512, 82).slices, 4);
        // Long K over few outputs splits to about a thousand workgroups, a few times the multiprocessors' worth.
        assert!(plan(64, 64, 1 << 20, 82).slices > 1);
        let p = plan(32, 4096, 4096, 82);
        assert!(p.slices >= 4 && p.groups.iter().product::<u32>() >= 256, "{p:?}");
        let p = plan(128, 128, 1 << 18, 82);
        assert!(p.groups.iter().product::<u32>() >= 900, "{p:?}");
    }

    // -- the table --

    use crate::context::ContextOptions;
    use crate::jit::structured::select::admit;
    use luma_core::BackendKind;

    fn cpu_with(cores: u32, forced: Option<&'static str>) -> Context {
        let options = ContextOptions::default().backend(BackendKind::Cpu).calibrate(false);
        let options = if let Some(forced) = forced { options.strategy(forced) } else { options };
        Context::with_tuning(options, |tuning| tuning.native_gemm = cores).unwrap()
    }

    /// The operands of an `m x k` by `k x n` product, stored or as views of the transposes of stored arrays.
    fn operands(ctx: &Context, (m, n, k): (usize, usize, usize), transposed: [bool; 2]) -> [Array<f32>; 2] {
        let lhs = if transposed[0] { Array::<f32>::zeros_on(ctx, [m, k]).unwrap().transpose() } else { Array::<f32>::zeros_on(ctx, [k, m]).unwrap() };
        let rhs = if transposed[1] { Array::<f32>::zeros_on(ctx, [k, n]).unwrap().transpose() } else { Array::<f32>::zeros_on(ctx, [n, k]).unwrap() };
        [lhs, rhs]
    }

    #[track_caller]
    fn selected(ctx: &Context, lhs: &Array<f32>, rhs: &Array<f32>, tf32: Tf32Request) -> StrategyId {
        let view = ContractionView { lhs, rhs, attrs: ContractionAttrs { tf32, half: false }, location: Location::caller(), epilogue: None, partners: &[] };
        select::select(ctx, &Subject::Contraction(&view), &Residencies::DEVICE).expect("the staged tiles always fit").id
    }

    /// `matmul::run`'s branches as they were, for the shapes a CPU context reaches: the CPU's native product when it has cores, else
    /// `plan` (a skinny kernel, or the staged tiles) with the demotion of a recomputed operand a skinny kernel would read often.
    /// (The host path, the narrow-`N` padding and the backend's own tiles need a GPU context: their conditions are not reachable here.)
    fn old_branch(ctx: &Context, lhs: &Array<f32>, rhs: &Array<f32>) -> StrategyId {
        let (ld, rd) = (lhs.shape().dims(), rhs.shape().dims());
        let (k, m, n) = (ld[0] as u32, ld[1] as u32, rd[0] as u32);
        if ctx.tuning().native_gemm > 0 {
            return StrategyId::CONTRACTION_NATIVE_CPU;
        }
        let along_k = lhs.pending().is_none_or(|node| jit::stored_along_k(&MatmulSide::Pending(Arc::clone(node)), [k, m]));
        let plan = plan(ctx.tuning(), m, n, k, along_k, ctx.max_workgroups_per_dimension(), ctx.segment_bytes()).unwrap();
        if let MatmulTile::Skinny { rows, cols, .. } = plan.tile {
            let reads = [n.div_ceil(u32::from(cols)), m.div_ceil(u32::from(rows))];
            let costly = [lhs, rhs].into_iter().zip(reads).any(|(array, reads)| (SKINNY_FUSED_READS..SKINNY_EVALUATE_READS).contains(&reads) && array.pending().is_some_and(jit::is_recomputed));
            return if costly { StrategyId::CONTRACTION_TILED } else { StrategyId::CONTRACTION_SKINNY };
        }
        StrategyId::CONTRACTION_TILED
    }

    #[test]
    fn the_table_is_well_formed() {
        let entries = super::super::strategy::table(FamilyTag::Contraction);
        assert!(entries.len() <= super::super::strategy::MAX_PER_FAMILY);
        for (at, strategy) in entries.iter().enumerate() {
            assert_eq!(usize::from(strategy.id.1), at, "{} sits at its id", strategy.name);
            assert_eq!(strategy.id.family(), FamilyTag::Contraction);
            assert!(strategy.lower.is_some());
            // The ragged and the padded kernels are the only ones ranked by cost, and the only ones that are not baselines.
            let cut = strategy.name.ends_with("-ragged") || strategy.name.ends_with("-padded");
            assert_eq!(strategy.baseline, !cut, "{}", strategy.name);
            assert_eq!(strategy.cost.is_some(), cut, "{}", strategy.name);
            assert_eq!(StrategyId::parse(FamilyTag::Contraction, strategy.name), Some(strategy.id));
            // Every entry is its own group, but a ragged and a padded kernel share one (and a bit class with the aligned kernel of their precision).
            let group = entries.iter().filter(|other| other.group == strategy.group).count();
            assert_eq!(group, if cut { 2 } else { 1 }, "{}", strategy.name);
        }
        let names: Vec<&str> = entries.iter().map(|strategy| strategy.name).collect();
        assert_eq!(names, ["host", "gemm-tf32", "gemm-tf32-ragged", "gemm-tf32-padded", "gemm-f32", "gemm-f32-ragged", "gemm-f32-padded", "native-cpu", "skinny", "tiled"]);
        let class = |name: &str| entries.iter().find(|strategy| strategy.name == name).unwrap().bits;
        assert_eq!(class("gemm-tf32"), class("gemm-tf32-ragged"));
        assert_eq!(class("gemm-tf32"), class("gemm-tf32-padded"));
        assert_eq!(class("gemm-f32"), class("gemm-f32-ragged"));
        assert_eq!(class("gemm-f32"), class("gemm-f32-padded"));
        assert_ne!(class("gemm-f32"), class("tiled"));
    }

    #[test]
    fn each_strategy_fits_exactly_where_the_old_branch_ran() {
        let shapes = [(10, 32, 1 << 20), (10, 32, 4096), (64, 64, 1 << 20), (1, 1, 2048), (256, 256, 256), (100, 100, 100), (33, 7, 5000), (4096, 4, 64), (8192, 1, 8192), (128, 20, 1 << 20), (3, 5, 7), (2000, 1, 70_001)];
        for cores in [0, 8] {
            let ctx = cpu_with(cores, None);
            for shape in shapes {
                for transposed in [[false; 2], [true, false], [false, true], [true; 2]] {
                    let [lhs, rhs] = operands(&ctx, shape, transposed);
                    for tf32 in [Tf32Request::Default, Tf32Request::Exact, Tf32Request::Tf32] {
                        let expected = old_branch(&ctx, &lhs, &rhs);
                        assert_eq!(selected(&ctx, &lhs, &rhs, tf32), expected, "cores {cores} shape {shape:?} transposed {transposed:?} {tf32:?}");
                    }
                }
            }
            // A CPU context has neither the host path nor the backend's own tiles: those strategies do not apply.
            let [lhs, rhs] = operands(&ctx, (64, 12, 64), [false; 2]);
            let view = ContractionView { lhs: &lhs, rhs: &rhs, attrs: ContractionAttrs { tf32: Tf32Request::Tf32, half: false }, location: Location::caller(), epilogue: None, partners: &[] };
            for id in [StrategyId::CONTRACTION_HOST, StrategyId::GEMM_TF32, StrategyId::GEMM_TF32_RAGGED, StrategyId::GEMM_TF32_PADDED, StrategyId::GEMM_F32, StrategyId::GEMM_F32_RAGGED, StrategyId::GEMM_F32_PADDED] {
                let strategy = &CONTRACTION_STRATEGIES[usize::from(id.1)];
                assert!(admit(strategy, &Subject::Contraction(&view), &ctx, &Residencies::DEVICE).is_err(), "{}", strategy.name);
            }
        }
    }

    #[test]
    fn a_skinny_plan_with_a_recomputed_operand_falls_to_tiled() {
        let ctx = cpu_with(0, None);
        // One-hot labels times features: 10 x 32 outputs over a long K are skinny tiles of 10 x 8, so the left operand is read 4 times.
        let (k, m, n) = (4096, 10, 32);
        let rhs = Array::<f32>::zeros_on(&ctx, [n, k]).unwrap();
        let stored = Array::<f32>::zeros_on(&ctx, [k, m]).unwrap();
        assert_eq!(selected(&ctx, &stored, &rhs, Tf32Request::Exact), StrategyId::CONTRACTION_SKINNY);
        // The same operand as an expression of two buffers is recomputed at every read.
        let other = Array::<f32>::zeros_on(&ctx, [k, m]).unwrap();
        let fused = &stored * &other;
        assert_eq!(selected(&ctx, &fused, &rhs, Tf32Request::Exact), StrategyId::CONTRACTION_TILED);
        assert_eq!(old_branch(&ctx, &fused, &rhs), StrategyId::CONTRACTION_TILED);
    }

    #[test]
    fn forcing_each_strategy_with_luma_strategy_matches_the_default_where_the_bit_class_is_equal() {
        let values = |len: usize, seed: u32| -> Vec<f32> { (0..len).map(|i| (((i as u32).wrapping_mul(2_654_435_761).wrapping_add(seed) >> 8) % 2001) as f32 / 1000.0 - 1.0).collect() };
        let run_on = |ctx: &Context, (m, n, k): (usize, usize, usize)| {
            let a = Array::from_slice_on(ctx, &values(k * m, 1), [k, m]).unwrap();
            let b = Array::from_slice_on(ctx, &values(n * k, 2), [n, k]).unwrap();
            a.matmul_f32(&b).to_vec().unwrap()
        };
        let default = cpu_with(8, None);
        for shape in [(10, 32, 4096), (37, 41, 29), (64, 64, 3000)] {
            let reference = run_on(&default, shape);
            for name in ["contraction:native-cpu", "contraction:skinny", "contraction:tiled"] {
                let forced = cpu_with(8, Some(name));
                let got = run_on(&forced, shape);
                if name == "contraction:native-cpu" {
                    assert_eq!(got, reference, "{name} {shape:?}: the default selection is this strategy: same bits");
                }
                let scale = reference.iter().fold(0.0f32, |top, v| top.max(v.abs())).max(1.0);
                assert!(got.iter().zip(&reference).all(|(a, b)| (a - b).abs() <= 1e-4 * scale), "{name} {shape:?}");
            }
        }
    }

    #[test]
    fn check_reports_errors_in_the_old_order() {
        let first = cpu_with(0, None);
        let second = cpu_with(0, None);
        let attrs = ContractionAttrs { tf32: Tf32Request::Default, half: false };
        let check = |lhs: &Array<f32>, rhs: &Array<f32>| ContractionView { lhs, rhs, attrs, location: Location::caller(), epilogue: None, partners: &[] }.check();
        let a = Array::<f32>::zeros_on(&first, [4, 3]).unwrap();
        let b = Array::<f32>::zeros_on(&first, [5, 4]).unwrap();
        assert!(matches!(check(&a, &b), Ok(Geometry { m: 3, n: 5, k: 4 })));
        // A shape mismatch, then a context mismatch before it, then a poisoned operand before both.
        let wrong = Array::<f32>::zeros_on(&first, [5, 6]).unwrap();
        assert!(matches!(check(&a, &wrong), Err(Failure::New(LumaError::ShapeMismatch { .. }))));
        let elsewhere = Array::<f32>::zeros_on(&second, [5, 6]).unwrap();
        assert!(matches!(check(&a, &elsewhere), Err(Failure::New(LumaError::ContextMismatch))));
        let poisoned = Array::<f32>::zeros_on(&second, [5, 6]).unwrap().matmul(&Array::zeros_on(&second, [9, 9]).unwrap());
        let Err(Failure::Inherited(error)) = check(&a, &poisoned) else { panic!("a poisoned operand reports its own error") };
        assert_eq!(error.to_string(), poisoned.error().unwrap().to_string());
        let Err(Failure::Inherited(_)) = check(&poisoned, &a) else { panic!("either operand") };
        // Extents past the third axis make no product.
        let cube = Array::<f32>::zeros_on(&first, [4, 4, 2]).unwrap();
        assert!(matches!(check(&cube, &a), Err(Failure::New(LumaError::ShapeMismatch { .. }))));
    }
}
