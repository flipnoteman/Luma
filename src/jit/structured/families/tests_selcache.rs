//! The selection cache (see `select`): a second identical call is a hit that selects what the full `fits` selects; every input of
//! `fits` that is not the subject, changed in turn, is a miss that selects what the uncached path selects. Built on the CPU backend, so
//! it runs without a GPU (the host paths need one: the recording tests use the refined solves, whose `fits` read the recording too).

use std::panic::Location;

use luma_core::BackendKind;

use crate::context::{Context, ContextOptions};
use crate::image::window::{Edge, Op, Pixel};
use crate::jit::structured::boundary::{
    dim, ConvAttrs, ConvStage, FactorAttrs, FactorKind, LineSolveAttrs, OpaqueAttrs, ScanAttrs, ScanOperand, ScatterAttrs, ScatterOp, SortAttrs, SortPayload, SortStage, WindowAttrs,
    WindowStage,
};
use crate::jit::structured::families::factor::{boundary_of, refined_stage};
use crate::jit::structured::select::{cacheable, first_fit_live, last_was_hit, CacheMode};
use crate::jit::structured::strategy::{first_candidate, first_candidate_in, note, table, Boundary, BoundaryAttrs, FamilyTag, FitsInputs, Subject};
use crate::dense::Dims;

#[track_caller]
fn here() -> &'static Location<'static> {
    Location::caller()
}

fn cpu(forced: &'static str) -> Context {
    Context::with_tuning(ContextOptions::default().backend(BackendKind::Cpu).calibrate(false).strategy(forced), |tuning| {
        tuning.subgroup32 = true;
        tuning.lookback_scan = true;
    })
    .unwrap()
}

/// Whether the context caches at all (a run under `LUMA_SELCACHE=off` has nothing to check).
fn caching(ctx: &Context) -> bool {
    ctx.jit().selcache.mode() != CacheMode::Off
}

/// One call, checked against the full `fits`, with whether it was a hit.
#[track_caller]
fn ask(ctx: &Context, boundary: &Boundary<'_>) -> (bool, Option<(crate::jit::structured::strategy::StrategyId, crate::jit::structured::strategy::Params)>) {
    let found = first_candidate(ctx, boundary);
    let hit = last_was_hit();
    assert_eq!(found, first_fit_live(ctx, &Subject::Boundary(boundary), table(boundary.family)), "the answer is the full fits' ({})", boundary.op);
    (hit, found)
}

#[track_caller]
fn miss_then_hit(ctx: &Context, boundary: &Boundary<'_>) {
    let (first_hit, first) = ask(ctx, boundary);
    assert!(first.is_some(), "{}: something fits", boundary.op);
    assert!(!first_hit, "{}: the first call is a miss", boundary.op);
    let (second_hit, second) = ask(ctx, boundary);
    assert!(second_hit, "{}: the second identical call is a hit", boundary.op);
    assert_eq!(first, second, "{}", boundary.op);
    assert!(boundary.from_cache.get());
}

// -- one case per family --

fn sort(n: u32, key_bits: u8) -> Boundary<'static> {
    Boundary::new(
        FamilyTag::Sort,
        "sort",
        here(),
        [n, 1, 1, 0],
        BoundaryAttrs::Sort(SortAttrs { stage: SortStage::Radix, payload: SortPayload::None, errored: false, value4: false, key_bits, len: u64::from(n) }),
    )
}

fn scan(n: u32, operand: ScanOperand) -> Boundary<'static> {
    Boundary::new(FamilyTag::Scan, "cumsum", here(), [n, 1, 1, 0], BoundaryAttrs::Scan(ScanAttrs { grouping_free: true, by_key: false, operand }))
}

fn scatter(base: usize, count: usize, atomic: ScatterOp) -> Boundary<'static> {
    Boundary::new(
        FamilyTag::Scatter,
        "put_add",
        here(),
        [dim(base), dim(count), 0, 0],
        BoundaryAttrs::Scatter(ScatterAttrs { atomic, base_len: base as u64, count: count as u64, base_is_view: false, base_is_read_view: false, host_site: false }),
    )
}

fn line_solve(n: u32, parts: bool) -> Boundary<'static> {
    Boundary::new(
        FamilyTag::LineSolve,
        "tridiagonal",
        here(),
        [n, 1, 64, 1],
        BoundaryAttrs::LineSolve(LineSolveAttrs { block: 1, parts, fused: false, starts: false, len: u64::from(n) * 64, max_scratch: 1 << 28 }),
    )
}

fn factor(rows: usize) -> Boundary<'static> {
    boundary_of(FactorKind::Lu, "lu", here(), Dims { rows, cols: rows, batch: 1 }, 1)
}

fn conv(kernel: u32, stride: u32) -> Boundary<'static> {
    Boundary::new(
        FamilyTag::Conv,
        "conv1d",
        here(),
        [kernel, 1000, stride, 1],
        BoundaryAttrs::Conv(ConvAttrs { stage: ConvStage::Direct, kernel_len: kernel, input_len: 1000, out_len: 1000, stride, batch: 1, tile_log: 0 }),
    )
}

fn window(k: u16, op: Op) -> Boundary<'static> {
    let k = [k, k, 1];
    Boundary::new(
        FamilyTag::Window,
        "median",
        here(),
        [u32::from(k[0]), u32::from(k[1]), 1, 1],
        BoundaryAttrs::Window(WindowAttrs { stage: WindowStage::Plan, op, k, stride: [1; 3], three_d: false, aux_cells: 0, pixel: Pixel::F32, edge: Edge::Clamp }),
    )
}

fn sparse(nnz: u64, columns: u32) -> Boundary<'static> {
    Boundary::new(
        FamilyTag::Opaque,
        "Sparse::matmul",
        here(),
        [dim(nnz as usize), 100, columns, 0],
        BoundaryAttrs::Opaque(OpaqueAttrs { nnz, outer: 100, native: true, first_non_native: false, columns, tiles: false, long: false }),
    )
}

#[test]
fn every_boundary_family_hits_on_the_second_identical_call() {
    let ctx = cpu("");
    if !caching(&ctx) {
        return;
    }
    miss_then_hit(&ctx, &sort(1 << 20, 32));
    miss_then_hit(&ctx, &scan(5000, ScanOperand::Buffer));
    miss_then_hit(&ctx, &scatter(4096, 100_000, ScatterOp::Add));
    miss_then_hit(&ctx, &line_solve(512, false));
    miss_then_hit(&ctx, &factor(100));
    miss_then_hit(&ctx, &conv(9, 1));
    miss_then_hit(&ctx, &window(5, Op::Median));
    miss_then_hit(&ctx, &sparse(500, 1));
}

#[test]
fn a_changed_shape_or_attribute_is_a_miss_with_the_uncached_answer() {
    let ctx = cpu("");
    if !caching(&ctx) {
        return;
    }
    for (name, base, changed) in [
        ("sort length", sort(1 << 20, 32), sort((1 << 20) + 1, 32)),
        ("sort key bits", sort(1 << 20, 32), sort(1 << 20, 16)),
        ("scan length", scan(5000, ScanOperand::Buffer), scan(5001, ScanOperand::Buffer)),
        ("scan operand", scan(5000, ScanOperand::Buffer), scan(5000, ScanOperand::Cheap)),
        ("scatter bins", scatter(4096, 100_000, ScatterOp::Add), scatter(4097, 100_000, ScatterOp::Add)),
        ("scatter op", scatter(4096, 100_000, ScatterOp::Add), scatter(4096, 100_000, ScatterOp::None)),
        ("line solve parts", line_solve(512, false), line_solve(512, true)),
        ("factor rows", factor(100), factor(70)),
        ("conv stride", conv(9, 4), conv(9, 5)),
        ("window op", window(5, Op::Median), window(5, Op::Min)),
        ("sparse columns", sparse(500, 1), sparse(500, 2)),
    ] {
        let (_, _) = ask(&ctx, &base);
        let (hit, _) = ask(&ctx, &base);
        assert!(hit, "{name}: the base case hits");
        let (hit, _) = ask(&ctx, &changed);
        assert!(!hit, "{name}: a changed input is a miss");
    }
}

#[test]
fn a_global_input_that_is_not_in_the_key_bumps_the_generation() {
    let ctx = cpu("");
    if !caching(&ctx) {
        return;
    }
    let b = scan(5000, ScanOperand::Buffer);
    ask(&ctx, &b);
    assert!(ask(&ctx, &b).0);
    // What `enable_profiling`, `finish_profiling` and a recording that begins or ends do.
    ctx.jit().selcache.bump();
    assert!(!ask(&ctx, &b).0, "the first call after a bump is a miss");
    assert!(ask(&ctx, &b).0);
    ctx.set_assume_busy(true);
    assert!(!ask(&ctx, &b).0, "set_assume_busy bumps");
    // Another context has a generation of its own, so nothing it cached answers this one.
    let other = cpu("");
    assert!(!ask(&other, &b).0);
}

#[test]
fn a_recording_opened_between_two_identical_calls_is_never_answered_from_before() {
    let ctx = cpu("");
    if !caching(&ctx) {
        return;
    }
    // The refined solves fit while this thread has no recording open (`fits` reads `recording_open`; the host paths read the same).
    let pick = |ctx: &Context| first_candidate_in(refined_stage(), ctx, &boundary_of(FactorKind::SolveRefined, "solve_refined", here(), Dims { rows: 100, cols: 100, batch: 1 }, 1));
    let before = pick(&ctx);
    assert!(before.is_some());
    assert_eq!(pick(&ctx), before);
    assert!(last_was_hit());
    assert!(ctx.begin_recording(Vec::new()));
    assert_eq!(pick(&ctx), None, "a recording is open: the refined solves do not fit, whatever an earlier call found");
    assert!(!last_was_hit());
    // An aborted recording is still open.
    ctx.abort_recording_for("test").unwrap();
    assert_eq!(pick(&ctx), None);
    ctx.end_recording();
    let after = pick(&ctx);
    assert_eq!(after, before, "the recording ended: the same answer as before");
    // Another thread has no recording of its own: it is answered as without one.
    assert!(ctx.begin_recording(Vec::new()));
    let elsewhere = std::thread::scope(|scope| scope.spawn(|| pick(&ctx)).join().unwrap());
    assert_eq!(elsewhere, before);
    ctx.end_recording();
}

#[test]
fn a_forced_strategy_is_selected_and_cached() {
    let ctx = cpu("scan:chunk_three_pass");
    if !caching(&ctx) {
        return;
    }
    let b = scan(5000, ScanOperand::Buffer);
    let (hit, found) = ask(&ctx, &b);
    assert!(!hit);
    assert_eq!(found.map(|(id, _)| id), Some(crate::jit::structured::strategy::StrategyId::SCAN_CHUNK_THREE_PASS), "forced where it fits");
    let (hit, again) = ask(&ctx, &b);
    assert!(hit);
    assert_eq!(again, found);
    // Forced where it does not fit: the default selection.
    let b = scan(100_000, ScanOperand::Buffer);
    let forced_off = cpu("scan:seq");
    let (_, found) = ask(&forced_off, &b);
    assert_ne!(found.map(|(id, _)| id), Some(crate::jit::structured::strategy::StrategyId::SCAN_SEQ));
}

#[test]
fn the_decision_of_a_hit_says_cached_and_is_otherwise_the_same() {
    let ctx = cpu("");
    if !caching(&ctx) {
        return;
    }
    ctx.record_decisions(true);
    let b = scan(5000, ScanOperand::Buffer);
    for _ in 0..2 {
        let (id, params) = first_candidate(&ctx, &b).unwrap();
        note(&ctx, &b, id, &params, &[]);
    }
    let decisions = ctx.take_decisions();
    assert_eq!(decisions.len(), 2);
    assert!(!decisions[0].cached && decisions[1].cached);
    let (mut first, mut second) = (decisions[0].clone(), decisions[1].clone());
    first.cached = false;
    second.cached = false;
    assert_eq!(first, second, "the Decision differs in `cached` only");
}

// -- what each family declares --

#[test]
fn the_tables_declare_what_their_fits_read() {
    for family in [FamilyTag::Sort, FamilyTag::Scan, FamilyTag::Scatter, FamilyTag::LineSolve, FamilyTag::Factor, FamilyTag::Conv, FamilyTag::Window, FamilyTag::Opaque] {
        assert!(cacheable(family), "{}", family.name());
    }
    assert!(!cacheable(FamilyTag::Reduce), "a reduction's fits read its partners and its own life");
    assert!(!cacheable(FamilyTag::Contraction), "a product's live walk is cheaper than a lookup on the CPU");
    let reads = |family: FamilyTag| table(family).iter().fold(FitsInputs::NONE, |all, strategy| all.with(strategy.reads));
    assert!(reads(FamilyTag::Sort).contains(FitsInputs::HOST));
    assert!(reads(FamilyTag::Scatter).contains(FitsInputs::HOST));
    assert!(reads(FamilyTag::Factor).contains(FitsInputs::HOST));
    assert!(reads(FamilyTag::Contraction).contains(FitsInputs::HOST.with(FitsInputs::OPERANDS)));
    assert!(!reads(FamilyTag::Scan).contains(FitsInputs::HOST));
}

// -- the packed attributes --

/// Every field of every attribute structure moves a word of `BoundaryAttrs::pack`: the hit path compares only the words, so a field
/// that is not packed would be answered from another value's selection.
#[test]
fn each_attribute_field_changes_the_packed_key() {
    use crate::jit::structured::boundary::{CompactAttrs, MergeAttrs};
    fn differs(name: &str, base: BoundaryAttrs, changed: BoundaryAttrs) {
        assert_ne!(base.pack(), changed.pack(), "{name} is not in the packed key");
    }
    let sort = SortAttrs { stage: SortStage::Radix, payload: SortPayload::None, errored: false, value4: false, key_bits: 32, len: 1000 };
    let a = BoundaryAttrs::Sort(sort);
    differs("sort.stage", a, BoundaryAttrs::Sort(SortAttrs { stage: SortStage::Axis, ..sort }));
    differs("sort.payload", a, BoundaryAttrs::Sort(SortAttrs { payload: SortPayload::Buffer, ..sort }));
    differs("sort.errored", a, BoundaryAttrs::Sort(SortAttrs { errored: true, ..sort }));
    differs("sort.value4", a, BoundaryAttrs::Sort(SortAttrs { value4: true, ..sort }));
    differs("sort.key_bits", a, BoundaryAttrs::Sort(SortAttrs { key_bits: 16, ..sort }));
    differs("sort.len", a, BoundaryAttrs::Sort(SortAttrs { len: 1001, ..sort }));

    let scan = ScanAttrs { grouping_free: false, by_key: false, operand: ScanOperand::Buffer };
    let a = BoundaryAttrs::Scan(scan);
    differs("scan.grouping_free", a, BoundaryAttrs::Scan(ScanAttrs { grouping_free: true, ..scan }));
    differs("scan.by_key", a, BoundaryAttrs::Scan(ScanAttrs { by_key: true, ..scan }));
    differs("scan.operand", a, BoundaryAttrs::Scan(ScanAttrs { operand: ScanOperand::Cheap, ..scan }));

    let scatter = ScatterAttrs { atomic: ScatterOp::None, base_len: 5, count: 6, base_is_view: false, base_is_read_view: false, host_site: false };
    let a = BoundaryAttrs::Scatter(scatter);
    differs("scatter.atomic", a, BoundaryAttrs::Scatter(ScatterAttrs { atomic: ScatterOp::Max, ..scatter }));
    differs("scatter.base_len", a, BoundaryAttrs::Scatter(ScatterAttrs { base_len: 6, ..scatter }));
    differs("scatter.count", a, BoundaryAttrs::Scatter(ScatterAttrs { count: 7, ..scatter }));
    differs("scatter.base_is_view", a, BoundaryAttrs::Scatter(ScatterAttrs { base_is_view: true, ..scatter }));
    differs("scatter.base_is_read_view", a, BoundaryAttrs::Scatter(ScatterAttrs { base_is_read_view: true, ..scatter }));
    differs("scatter.host_site", a, BoundaryAttrs::Scatter(ScatterAttrs { host_site: true, ..scatter }));

    let conv = ConvAttrs { stage: ConvStage::Direct, kernel_len: 1, input_len: 2, out_len: 3, stride: 4, batch: 5, tile_log: 6 };
    let a = BoundaryAttrs::Conv(conv);
    differs("conv.stage", a, BoundaryAttrs::Conv(ConvAttrs { stage: ConvStage::Fir, ..conv }));
    differs("conv.kernel_len", a, BoundaryAttrs::Conv(ConvAttrs { kernel_len: 9, ..conv }));
    differs("conv.input_len", a, BoundaryAttrs::Conv(ConvAttrs { input_len: 9, ..conv }));
    differs("conv.out_len", a, BoundaryAttrs::Conv(ConvAttrs { out_len: 9, ..conv }));
    differs("conv.stride", a, BoundaryAttrs::Conv(ConvAttrs { stride: 9, ..conv }));
    differs("conv.batch", a, BoundaryAttrs::Conv(ConvAttrs { batch: 9, ..conv }));
    differs("conv.tile_log", a, BoundaryAttrs::Conv(ConvAttrs { tile_log: 9, ..conv }));

    let window = WindowAttrs { stage: WindowStage::Plan, op: Op::Median, k: [3, 4, 5], stride: [1, 2, 3], three_d: false, aux_cells: 7, pixel: Pixel::F32, edge: Edge::Clamp };
    let a = BoundaryAttrs::Window(window);
    differs("window.stage", a, BoundaryAttrs::Window(WindowAttrs { stage: WindowStage::Entry, ..window }));
    differs("window.op", a, BoundaryAttrs::Window(WindowAttrs { op: Op::Min, ..window }));
    for at in 0..3 {
        let mut k = window.k;
        k[at] += 1;
        differs("window.k", a, BoundaryAttrs::Window(WindowAttrs { k, ..window }));
        let mut stride = window.stride;
        stride[at] += 1;
        differs("window.stride", a, BoundaryAttrs::Window(WindowAttrs { stride, ..window }));
    }
    differs("window.three_d", a, BoundaryAttrs::Window(WindowAttrs { three_d: true, ..window }));
    differs("window.aux_cells", a, BoundaryAttrs::Window(WindowAttrs { aux_cells: 8, ..window }));
    differs("window.pixel", a, BoundaryAttrs::Window(WindowAttrs { pixel: Pixel::U32, ..window }));
    differs("window.edge", a, BoundaryAttrs::Window(WindowAttrs { edge: Edge::Wrap, ..window }));

    let opaque = OpaqueAttrs { nnz: 1, outer: 2, native: false, first_non_native: false, columns: 3, tiles: false, long: false };
    let a = BoundaryAttrs::Opaque(opaque);
    differs("opaque.nnz", a, BoundaryAttrs::Opaque(OpaqueAttrs { nnz: 9, ..opaque }));
    differs("opaque.outer", a, BoundaryAttrs::Opaque(OpaqueAttrs { outer: 9, ..opaque }));
    differs("opaque.native", a, BoundaryAttrs::Opaque(OpaqueAttrs { native: true, ..opaque }));
    differs("opaque.first_non_native", a, BoundaryAttrs::Opaque(OpaqueAttrs { first_non_native: true, ..opaque }));
    differs("opaque.columns", a, BoundaryAttrs::Opaque(OpaqueAttrs { columns: 9, ..opaque }));
    differs("opaque.tiles", a, BoundaryAttrs::Opaque(OpaqueAttrs { tiles: true, ..opaque }));
    differs("opaque.long", a, BoundaryAttrs::Opaque(OpaqueAttrs { long: true, ..opaque }));

    let line = LineSolveAttrs { block: 1, parts: false, fused: false, starts: false, len: 4, max_scratch: 5 };
    let a = BoundaryAttrs::LineSolve(line);
    differs("line.block", a, BoundaryAttrs::LineSolve(LineSolveAttrs { block: 2, ..line }));
    differs("line.parts", a, BoundaryAttrs::LineSolve(LineSolveAttrs { parts: true, ..line }));
    differs("line.fused", a, BoundaryAttrs::LineSolve(LineSolveAttrs { fused: true, ..line }));
    differs("line.starts", a, BoundaryAttrs::LineSolve(LineSolveAttrs { starts: true, ..line }));
    differs("line.len", a, BoundaryAttrs::LineSolve(LineSolveAttrs { len: 9, ..line }));
    differs("line.max_scratch", a, BoundaryAttrs::LineSolve(LineSolveAttrs { max_scratch: 9, ..line }));

    let factor = FactorAttrs { kind: FactorKind::Lu, inverse: false, exact: false, vectors: false };
    let a = BoundaryAttrs::Factor(factor);
    differs("factor.kind", a, BoundaryAttrs::Factor(FactorAttrs { kind: FactorKind::Qr, ..factor }));
    differs("factor.inverse", a, BoundaryAttrs::Factor(FactorAttrs { inverse: true, ..factor }));
    differs("factor.exact", a, BoundaryAttrs::Factor(FactorAttrs { exact: true, ..factor }));
    differs("factor.vectors", a, BoundaryAttrs::Factor(FactorAttrs { vectors: true, ..factor }));

    let a = BoundaryAttrs::Compact(CompactAttrs { padded: false, words: false });
    differs("compact.padded", a, BoundaryAttrs::Compact(CompactAttrs { padded: true, words: false }));
    differs("compact.words", a, BoundaryAttrs::Compact(CompactAttrs { padded: false, words: true }));
    differs("merge.total", BoundaryAttrs::Merge(MergeAttrs { total: 1 }), BoundaryAttrs::Merge(MergeAttrs { total: 2 }));
}
