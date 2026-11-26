#![allow(dead_code)] // the families take their types over one port at a time
//! What the eager-boundary families (sort, scan, scatter; search, compact and merge as light ports) tell the registry: the
//! attributes of a [`Boundary`] and the parameters a strategy's `fits` hands to the op's own body. Plain `Copy` data, built on the
//! stack by the op; nothing here evaluates anything or holds a node.

use std::borrow::Cow;
use std::cell::Cell;
use std::marker::PhantomData;
use std::panic::Location;

use super::strategy::{note, Boundary, BoundaryAttrs, FamilyTag, Params, StrategyId};
use crate::context::Context;
use crate::profile::{Decision, DecisionSite};

impl<'a> Boundary<'a> {
    /// The descriptor of an op of `family` at `location`: `dims` is the shape that picks the plan.
    #[inline(always)]
    pub(crate) fn new(family: FamilyTag, op: &'static str, location: &'static Location<'static>, dims: [u32; 4], attrs: BoundaryAttrs) -> Boundary<'a> {
        let key = attrs.pack();
        Boundary { family, op, location, dims, attrs, key, from_cache: Cell::new(false), _borrow: PhantomData }
    }
}

/// `usize` as the `u32` of a [`Boundary`]'s `dims` (saturating: the exact size is in the attributes where a `fits` needs it).
#[inline]
pub(crate) fn dim(size: usize) -> u32 {
    u32::try_from(size).unwrap_or(u32::MAX)
}

// -- Sort --

/// Which selection a sort boundary stands for: the op chains several (the host path at its entry, the radix variant in
/// `radix_sort`, the axis and by-key compositions), and each strategy applies at one of them.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum SortStage {
    /// The entry of `sort`, `argsort`, `sort_with_indices`: the host path or on to the radix sort.
    Entry,
    /// `radix_sort`: which of its kernels.
    Radix,
    /// `sort_by_key_bits`: a payload through the radix passes or the permutation.
    ByKey,
    /// `sort_axis`: one kernel per short line or a sort of everything by line.
    Axis,
    /// `sort_by_key_axis`.
    ByKeyAxis,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum SortPayload {
    None,
    Index,
    Buffer,
}

/// What a sort boundary reads; `dims` is `[n, stride, lines, 0]` (the whole array: `[len, 1, 1, 0]`).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) struct SortAttrs {
    pub(crate) stage: SortStage,
    pub(crate) payload: SortPayload,
    /// The array holds an error (the host path would read it).
    pub(crate) errored: bool,
    /// The by-key values are 4-byte plain elements (they travel through the radix passes).
    pub(crate) value4: bool,
    /// The keys differ in their low `key_bits` bits.
    pub(crate) key_bits: u8,
    /// Elements in the sort (`len`).
    pub(crate) len: u64,
}

/// What a radix sort runs with: keys per invocation of a sweep tile, keys per tile, passes.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct SortParams {
    pub(crate) per_thread: u32,
    pub(crate) tile: u32,
    pub(crate) passes: u32,
}

impl SortParams {
    pub(crate) const NONE: SortParams = SortParams { per_thread: 0, tile: 0, passes: 0 };
}

// -- Scan --

/// What a scan boundary reads; `dims` is `[n, stride, lines, 0]`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) struct ScanAttrs {
    /// The operator groups however it likes (`u32`/`i32`, or an extremum): a look-back may take any earlier chunk's prefix.
    pub(crate) grouping_free: bool,
    pub(crate) by_key: bool,
    /// What the scan reads: an evaluated buffer, or a pending expression (cheap to compute again, or worth storing).
    pub(crate) operand: ScanOperand,
}

/// The operand of a scan, as far as the choice of its prologue needs to know.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum ScanOperand {
    Buffer,
    /// A pending expression that costs less to compute in each kernel that reads it than to write and read back
    /// (`jit::worth_storing` is false).
    Cheap,
    /// A pending expression that is worth storing: computed in the kernels only where one kernel reads it.
    Costly,
}

/// What a scan runs with: `vec` words per load of a warp pass, `nb` chunks (blocks) per line, `k` cells per block.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct ScanParams {
    pub(crate) vec: u32,
    pub(crate) nb: u32,
    pub(crate) k: u32,
    /// The kernels compute a pending operand themselves (the strategy's prologue): `prologue=fused` in the Decision.
    pub(crate) fused: bool,
}

impl ScanParams {
    pub(crate) const NONE: ScanParams = ScanParams { vec: 0, nb: 1, k: 0, fused: false };
}

// -- Scatter --

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum ScatterOp {
    None,
    Add,
    Min,
    Max,
}

impl ScatterOp {
    pub(crate) fn of(atomic: Option<crate::jit::AtomicOp>) -> ScatterOp {
        match atomic {
            None => ScatterOp::None,
            Some(crate::jit::AtomicOp::Add) => ScatterOp::Add,
            Some(crate::jit::AtomicOp::Min) => ScatterOp::Min,
            Some(crate::jit::AtomicOp::Max) => ScatterOp::Max,
        }
    }
}

/// What a scatter boundary reads; `dims` is `[base_len, count, 0, 0]` (saturated).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) struct ScatterAttrs {
    pub(crate) atomic: ScatterOp,
    pub(crate) base_len: u64,
    pub(crate) count: u64,
    pub(crate) base_is_view: bool,
    pub(crate) base_is_read_view: bool,
    /// The host-path site of `put_with` (false: the kernel site in `assign::scatter`).
    pub(crate) host_site: bool,
}

/// What a scatter runs with: the length of the target.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct ScatterParams {
    pub(crate) len: u32,
}

// -- Conv --

/// Which selection a conv boundary stands for: `conv1d`'s launch, `fft::convolve`, or the entry of `fir`/`fir_long`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum ConvStage {
    /// `launch_conv`: one direct convolution launch.
    Direct,
    /// `fft::convolve`: one FFT convolution of a single buffer.
    Fft,
    /// `fir`: the direct kernel for filters of up to 256 taps, else on to `Long`.
    Fir,
    /// `fir_long`: a filter longer than the direct kernel.
    FirLong,
}

/// What a conv boundary reads; `dims` is `[kernel_len, input_len, stride, batch]`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) struct ConvAttrs {
    pub(crate) stage: ConvStage,
    pub(crate) kernel_len: u32,
    pub(crate) input_len: u32,
    /// Outputs of the whole convolution (`Fft`) or of the launch (`Direct`).
    pub(crate) out_len: u32,
    pub(crate) stride: u32,
    /// Signals filtered together (`FirLong`), else 1.
    pub(crate) batch: u32,
    /// The device's FFT tile (`fft::tile_log`), 0 outside `Fft`.
    pub(crate) tile_log: u32,
}

/// What a convolution runs with: the tiled kernel's shape, the transform's sizes, the launches of a block-of-taps FIR.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct ConvParams {
    pub(crate) per_thread: u32,
    pub(crate) workgroup: u32,
    /// `log2` of the padded whole transform (`Fft`).
    pub(crate) log_n: u32,
    /// `log2` of the overlap-save block; 0 where the whole signal is one transform.
    pub(crate) log_b: u32,
    /// Launches of a block-of-taps FIR (`ceil(k / 256)`).
    pub(crate) blocks: u32,
}

impl ConvParams {
    pub(crate) const NONE: ConvParams = ConvParams { per_thread: 0, workgroup: 0, log_n: 0, log_b: 0, blocks: 0 };
}

// -- Window --

/// Which selection a window boundary stands for: the entry of `minfilt`/`maxfilt`, or the kernel's plan.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum WindowStage {
    Entry,
    Plan,
}

/// What a window boundary reads; `dims` is `[k0, k1, k2, stride0]`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) struct WindowAttrs {
    pub(crate) stage: WindowStage,
    pub(crate) op: crate::image::window::Op,
    pub(crate) k: [u16; 3],
    pub(crate) stride: [u8; 3],
    pub(crate) three_d: bool,
    /// The cells of the weights / mask binding.
    pub(crate) aux_cells: u64,
    pub(crate) pixel: crate::image::window::Pixel,
    pub(crate) edge: crate::image::window::Edge,
}

/// The kernel a window plan fits (`None` at the entry of an extremum: the passes plan their own).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct WindowParams {
    pub(crate) window: Option<crate::image::window::Window>,
}

// -- Opaque --

/// What a sparse product reads, as far as its selection goes.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) struct OpaqueAttrs {
    pub(crate) nnz: u64,
    /// Rows of the result (`outer`): the mean row length is `nnz / outer`.
    pub(crate) outer: u64,
    /// The compressed format the product sums over without atomics (CSR `A x`, CSC `A^T x`).
    pub(crate) native: bool,
    /// The first use of a non-native format (it scatters; the next one converts).
    pub(crate) first_non_native: bool,
    /// Columns of the dense operand (1: a vector).
    pub(crate) columns: u32,
    /// The matrix has rows longer than `SHORT_ROWS` (the merge tiles exist); read from the memoised row plan.
    pub(crate) tiles: bool,
    /// The matrix has rows longer than `LONG_ROW`.
    pub(crate) long: bool,
}

/// What a sparse product runs with: the lanes of the row kernel (`2^lanes_log` per row), the tiles of the merge kernel.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct OpaqueParams {
    pub(crate) lanes_log: u32,
    pub(crate) tiles: u32,
}

impl OpaqueParams {
    pub(crate) const NONE: OpaqueParams = OpaqueParams { lanes_log: 0, tiles: 0 };
}

// -- Light ports: tag, site and `note` --

/// `FamilyTag::Compact`: the tile count and scatter of a compaction.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) struct CompactAttrs {
    pub(crate) padded: bool,
    pub(crate) words: bool,
}

/// `FamilyTag::Merge`: merge of two sorted arrays of `total` elements, or a factorize.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) struct MergeAttrs {
    pub(crate) total: u64,
}

/// Records the Decision of a light-port op (`FamilyTag::{Search, Compact, Merge}`: no table, no `fits`, no candidates): `label`
/// is the legacy `chosen` text. One relaxed load while nobody records.
#[inline]
pub(crate) fn note_light(ctx: &Context, family: FamilyTag, op: &'static str, location: &'static Location<'static>, label: impl Into<Cow<'static, str>>) {
    if !ctx.jit().recording_decisions() {
        return;
    }
    let site = match family {
        FamilyTag::Search => DecisionSite::SearchPlan,
        FamilyTag::Compact => DecisionSite::CompactPlan,
        FamilyTag::Merge => DecisionSite::MergePlan,
        FamilyTag::Spectral => DecisionSite::SpectralPlan,
        FamilyTag::Sample => DecisionSite::SamplePlan,
        _ => unreachable!("only the light ports note a label"),
    };
    ctx.jit().note_decision(Decision::new(op, location, site, label));
}

/// [`note`] for a body that chose `id` with `params` (here so that the families' files import one module).
#[inline]
pub(crate) fn note_chosen(ctx: &Context, boundary: &Boundary<'_>, id: StrategyId, params: &Params, declined: &[StrategyId]) {
    note(ctx, boundary, id, params, declined);
}

// -- LineSolve --

/// What a line-solve boundary reads (`linalg::run`, `run_parts`); `dims` is `[n, stride, lines (systems folded), systems]`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) struct LineSolveAttrs {
    pub(crate) block: u8,
    /// Coefficients given as parts (computed in the kernel by `luma_cell`), else stacked buffers.
    pub(crate) parts: bool,
    /// Some part is a pending expression.
    pub(crate) fused: bool,
    pub(crate) starts: bool,
    /// Cells of one part (`n * lines`), which sizes the Thomas scratch of a parts solve.
    pub(crate) len: u64,
    /// The largest buffer the device binds (`ctx.max_buffer_bytes()`).
    pub(crate) max_scratch: u64,
}

/// What a line solve runs with: the cyclic plan or the Thomas plan, and the scratch it needs.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct LineSolveParams {
    pub(crate) block: u8,
    pub(crate) parts: bool,
    pub(crate) workgroup: u16,
    /// Cyclic reduction: rows folded away before the shared-memory levels, and rows of a line kept in workgroup memory.
    pub(crate) shift: u8,
    pub(crate) shared_rows: u32,
    /// Thomas: tiled scratch, two invocations per line.
    pub(crate) tiled: bool,
    pub(crate) twisted: bool,
    pub(crate) scratch_bytes: u64,
}

impl LineSolveParams {
    /// The text of the `LineSolvePlan` decisions since the line solver first recorded its choice: cyclic reduction, or the Thomas
    /// algorithm for every strategy that is not cyclic reduction.
    pub(crate) fn legacy_label(&self, name: &str) -> String {
        let parts = if self.parts { " parts" } else { "" };
        if name == "cyclic" {
            format!("cyclic{parts} block={} workgroup={} shift={}", self.block, self.workgroup, self.shift)
        } else {
            format!("thomas{parts} block={}", self.block)
        }
    }
}

// -- Factor --

/// What a factorisation boundary stands for; each strategy of the family applies at one or two of them.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum FactorKind {
    Lu,
    Cholesky,
    Qr,
    Svd,
    Eigh,
    SolveTriangular,
    SolveRefined,
    SolveSpdRefined,
    /// Compositions: they select nothing themselves (their stages are those of the factorisation and the triangular solves they call),
    /// but a captured launch list of one is keyed apart from its factorisation's.
    Solve,
    SolveSpd,
    Inverse,
    Lstsq,
}

/// What a factorisation boundary reads; `dims` is `[rows, cols, batch, rhs]`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) struct FactorAttrs {
    pub(crate) kind: FactorKind,
    /// Triangular solves: the diagonal blocks may be taken as explicit inverses (`LUMA_TRSV_INV=1`, or the solves of a factorisation).
    pub(crate) inverse: bool,
    /// SVD: no rank tolerance (a precondition by QR may apply).
    pub(crate) exact: bool,
    /// SVD and eigh: the vectors are computed too.
    pub(crate) vectors: bool,
}

/// What a factorisation strategy runs with: what the Decision label shows. The geometry (the tournament's chunks, the TSQR blocks, the
/// look-back kernel's columns) is recomputed by the op from the same pure functions `fits` used, so that a selection copies two bytes
/// where the small paths (one workgroup per matrix) are measured in a few thousand host instructions.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct FactorParams {
    /// The block of columns of the blocked Cholesky and triangular solves (`LUMA_CHOL_NB`); 0 for every other strategy.
    pub(crate) nb: u16,
}

impl FactorParams {
    pub(crate) const NONE: FactorParams = FactorParams { nb: 0 };
}

// -- The key the selection cache compares --

/// Every field of the attributes, in four words, so that two attributes are equal exactly where their words are: what the selection
/// cache's hit path compares instead of the structures (a `Boundary` is built where the variant is a constant, so the packing
/// folds to a few shifts at the op). A field that is added must be packed: `tests_selcache` flips each field in turn.
impl BoundaryAttrs {
    #[inline(always)]
    pub(crate) fn pack(&self) -> [u64; 4] {
        match self {
            BoundaryAttrs::None => [0; 4],
            BoundaryAttrs::Sort(a) => [a.len, a.stage as u64 | (a.payload as u64) << 8 | u64::from(a.errored) << 16 | u64::from(a.value4) << 17 | u64::from(a.key_bits) << 24, 0, 0],
            BoundaryAttrs::Scan(a) => [u64::from(a.grouping_free) | u64::from(a.by_key) << 1 | (a.operand as u64) << 2, 0, 0, 0],
            BoundaryAttrs::Scatter(a) => [a.base_len, a.count, a.atomic as u64 | u64::from(a.base_is_view) << 8 | u64::from(a.base_is_read_view) << 9 | u64::from(a.host_site) << 10, 0],
            BoundaryAttrs::Conv(a) => [
                u64::from(a.kernel_len) | u64::from(a.input_len) << 32,
                u64::from(a.out_len) | u64::from(a.stride) << 32,
                u64::from(a.batch) | u64::from(a.tile_log) << 32,
                a.stage as u64,
            ],
            BoundaryAttrs::Window(a) => [
                u64::from(a.k[0]) | u64::from(a.k[1]) << 16 | u64::from(a.k[2]) << 32 | u64::from(a.stride[0]) << 48 | u64::from(a.stride[1]) << 56,
                u64::from(a.stride[2]) | (a.op as u64) << 8 | u64::from(a.three_d) << 16 | (a.pixel as u64) << 24 | (a.edge as u64) << 32 | (a.stage as u64) << 40,
                a.aux_cells,
                0,
            ],
            BoundaryAttrs::Opaque(a) => [a.nnz, a.outer, u64::from(a.columns) | u64::from(a.native) << 32 | u64::from(a.first_non_native) << 33 | u64::from(a.tiles) << 34 | u64::from(a.long) << 35, 0],
            BoundaryAttrs::Compact(a) => [u64::from(a.padded) | u64::from(a.words) << 1, 0, 0, 0],
            BoundaryAttrs::Merge(a) => [a.total, 0, 0, 0],
            BoundaryAttrs::LineSolve(a) => [a.len, a.max_scratch, u64::from(a.block) | u64::from(a.parts) << 8 | u64::from(a.fused) << 9 | u64::from(a.starts) << 10, 0],
            BoundaryAttrs::Factor(a) => [a.kind as u64 | u64::from(a.inverse) << 8 | u64::from(a.exact) << 9 | u64::from(a.vectors) << 10, 0, 0, 0],
        }
    }
}
