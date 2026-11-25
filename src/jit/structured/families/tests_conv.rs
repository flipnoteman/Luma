//! The `fits` tables of the conv and window families and of the sparse products of the opaque family on both sides of every
//! threshold, and the labels of the ops that only log. Built on the CPU backend with hand-made `Tuning`s (no GPU needed).

use std::panic::Location;

use luma_core::BackendKind;

use crate::context::{Context, ContextOptions};
use crate::image::window::{Edge, Op, Pixel, MEDIAN_REGISTERS};
use crate::jit::structured::boundary::{dim, ConvAttrs, ConvStage, OpaqueAttrs, WindowAttrs, WindowParams, WindowStage};
use crate::jit::structured::opaque::OPAQUE;
use crate::jit::structured::strategy::{candidates, table, Boundary, BoundaryAttrs, FamilyTag, Params, StrategyId};
use crate::tuning::Tuning;

fn cpu_with(tune: impl FnOnce(&mut Tuning)) -> Context {
    Context::with_tuning(ContextOptions::default().backend(BackendKind::Cpu).calibrate(false).strategy(""), tune).unwrap()
}

#[track_caller]
fn here() -> &'static Location<'static> {
    Location::caller()
}

fn first(ctx: &Context, boundary: &Boundary<'_>) -> (StrategyId, Params) {
    candidates(ctx, boundary).next().expect("a strategy fits")
}

// -- Conv --

fn conv_boundary(stage: ConvStage, kernel_len: u32, out_len: u32, stride: u32, batch: u32, tile_log: u32) -> Boundary<'static> {
    Boundary::new(
        FamilyTag::Conv,
        "conv",
        here(),
        [kernel_len, out_len, stride, batch],
        BoundaryAttrs::Conv(ConvAttrs { stage, kernel_len, input_len: out_len, out_len, stride, batch, tile_log }),
    )
}

#[test]
fn conv_direct_fits_on_both_sides_of_the_stride_limit() {
    let ctx = cpu_with(|t| {
        t.conv_workgroup = 64;
        t.conv_per_thread = 16;
    });
    for stride in 1..=4 {
        let (id, params) = first(&ctx, &conv_boundary(ConvStage::Direct, 9, 1000, stride, 1, 0));
        assert_eq!(id, StrategyId::CONV_DIRECT_TILED);
        let Params::Conv(params) = params else { panic!("conv params") };
        assert_eq!((params.workgroup, params.per_thread), (64, 16));
    }
    assert_eq!(first(&ctx, &conv_boundary(ConvStage::Direct, 9, 1000, 5, 1, 0)).0, StrategyId::CONV_DIRECT);
}

#[test]
fn fir_fits_on_both_sides_of_every_threshold() {
    let ctx = cpu_with(|_| {});
    let pick = |k: u32, batch: u32| first(&ctx, &conv_boundary(ConvStage::Fir, k, 100_000, 1, batch, 0)).0;
    assert_eq!(pick(256, 1), StrategyId::CONV_FIR_DIRECT);
    assert_eq!(pick(256, 100), StrategyId::CONV_FIR_DIRECT);
    assert_eq!(pick(257, 1), StrategyId::CONV_FIR_BLOCKS);
    assert_eq!(pick(383, 1), StrategyId::CONV_FIR_BLOCKS);
    assert_eq!(pick(384, 1), StrategyId::CONV_FIR_FFT);
    assert_eq!(pick(384, 15), StrategyId::CONV_FIR_FFT);
    assert_eq!(pick(384, 16), StrategyId::CONV_FIR_BLOCKS);
    assert_eq!(pick(1000, 15), StrategyId::CONV_FIR_FFT);
    assert_eq!(pick(1000, 16), StrategyId::CONV_FIR_BLOCKS);
    let (_, params) = first(&ctx, &conv_boundary(ConvStage::Fir, 1000, 100_000, 1, 16, 0));
    assert!(matches!(params, Params::Conv(p) if p.blocks == 4));
}

/// The choice of one transform against overlap-save blocks flips where the two costs cross, as the production function says.
#[test]
fn the_overlap_save_rule_flips_where_the_costs_cross() {
    let ctx = cpu_with(|_| {});
    for (kernel_len, tile_log) in [(64u32, 12u32), (1000, 12), (3000, 11)] {
        let (mut singles, mut blocked) = (0, 0);
        let mut out_len = 2 * kernel_len;
        while out_len < 1 << 25 {
            let log_n = out_len.next_power_of_two().trailing_zeros();
            let expected = crate::fft::overlap_blocks(kernel_len, out_len, log_n, tile_log);
            let (id, params) = first(&ctx, &conv_boundary(ConvStage::Fft, kernel_len, out_len, 1, 1, tile_log));
            let Params::Conv(params) = params else { panic!("conv params") };
            match expected {
                Some((log_b, _)) => {
                    blocked += 1;
                    assert_eq!((id, params.log_b, params.log_n), (StrategyId::CONV_FFT_BLOCKED, log_b, log_n), "k={kernel_len} out={out_len}");
                }
                None => {
                    singles += 1;
                    assert_eq!((id, params.log_b, params.log_n), (StrategyId::CONV_FFT_SINGLE, 0, log_n), "k={kernel_len} out={out_len}");
                }
            }
            out_len += out_len / 7 + 1;
        }
        assert!(singles > 0 && blocked > 0, "k={kernel_len}: {singles} single, {blocked} blocked");
    }
}

// -- Window --

fn window_boundary(stage: WindowStage, op: Op, k: [u16; 3], three_d: bool) -> Boundary<'static> {
    let stride = [1; 3];
    Boundary::new(
        FamilyTag::Window,
        "window",
        here(),
        [u32::from(k[0]), u32::from(k[1]), u32::from(k[2]), 1],
        BoundaryAttrs::Window(WindowAttrs { stage, op, k, stride, three_d, aux_cells: 0, pixel: Pixel::F32, edge: Edge::Clamp }),
    )
}

#[test]
fn window_extremum_is_separable_from_25_cells() {
    let ctx = cpu_with(|_| {});
    let entry = |op, k0, k1| candidates(&ctx, &window_boundary(WindowStage::Entry, op, [k0, k1, 1], false)).next().map(|(id, _)| id);
    assert_eq!(entry(Op::Min, 5, 5), Some(StrategyId::WINDOW_SEPARABLE_EXTREMUM));
    assert_eq!(entry(Op::Max, 25, 1), None, "one axis is not separable");
    assert_eq!(entry(Op::Max, 1, 25), None);
    assert_eq!(entry(Op::Max, 3, 9), Some(StrategyId::WINDOW_SEPARABLE_EXTREMUM));
    assert_eq!(entry(Op::Min, 3, 7), None, "21 cells");
    assert_eq!(entry(Op::Min, 4, 6), None, "24 cells");
    assert_eq!(entry(Op::Min, 5, 5), Some(StrategyId::WINDOW_SEPARABLE_EXTREMUM), "25 cells");
    assert_eq!(entry(Op::Median, 7, 7), None, "only extrema are separable");
}

fn plan(ctx: &Context, op: Op, k: [u16; 3], three_d: bool) -> Option<(StrategyId, crate::image::window::Window)> {
    candidates(ctx, &window_boundary(WindowStage::Plan, op, k, three_d)).next().map(|(id, params)| match params {
        Params::Window(WindowParams { window: Some(w) }) => (id, w),
        other => panic!("a plan carries its kernel, got {other:?}"),
    })
}

#[test]
fn median_selection_changes_after_49_cells_and_the_tile_list_after_25() {
    let ctx = cpu_with(|t| t.workgroup_storage = 48 * 1024);
    let cells = |k0: u16, k1: u16| plan(&ctx, Op::Median, [k0, k1, 1], false).unwrap();
    assert_eq!(MEDIAN_REGISTERS, 49);
    assert_eq!(cells(7, 7).0, StrategyId::WINDOW_MEDIAN_REGISTERS);
    assert_eq!(cells(5, 10).0, StrategyId::WINDOW_MEDIAN_BITWISE);
    assert_eq!(cells(7, 7).1.ry, 1, "49 cells: the narrow tile list");
    // Up to 25 cells the list that starts with the four-outputs-per-invocation tile.
    assert_eq!((cells(5, 5).1.tile, cells(5, 5).1.ry), ([32, 32, 1], 4));
    assert_eq!((cells(5, 6).1.tile, cells(5, 6).1.ry), ([32, 8, 1], 1));
    // Other ops are tiled windows.
    assert_eq!(plan(&ctx, Op::Correlate, [3, 3, 1], false).unwrap().0, StrategyId::WINDOW_TILED);
    assert_eq!(plan(&ctx, Op::Min, [3, 3, 1], false).unwrap().0, StrategyId::WINDOW_TILED);
}

#[test]
fn a_window_that_does_not_fit_workgroup_memory_has_no_candidate() {
    let ctx = cpu_with(|t| t.workgroup_storage = 256);
    assert!(plan(&ctx, Op::Median, [9, 9, 1], false).is_none());
    assert!(plan(&ctx, Op::Correlate, [31, 31, 1], false).is_none());
}

#[test]
fn a_plan_is_the_same_whatever_the_stage_asks() {
    // `window:tiled` forced on a median runs the same kernel as the median selection.
    let ctx = cpu_with(|t| t.workgroup_storage = 48 * 1024);
    let (_, median) = plan(&ctx, Op::Median, [3, 3, 1], false).unwrap();
    let forced = Context::with_tuning(ContextOptions::default().backend(BackendKind::Cpu).calibrate(false).strategy("window:tiled"), |t| t.workgroup_storage = 48 * 1024).unwrap();
    assert_eq!(plan(&forced, Op::Median, [3, 3, 1], false), Some((StrategyId::WINDOW_TILED, median)));
}

// -- Opaque: the sparse products --

fn sparse_boundary(nnz: u64, native: bool, first_non_native: bool, columns: u32, tiles: bool, long: bool) -> Boundary<'static> {
    Boundary::new(
        FamilyTag::Opaque,
        "Sparse::matmul",
        here(),
        [dim(nnz as usize), 100, columns, 0],
        BoundaryAttrs::Opaque(OpaqueAttrs { nnz, outer: 100, native, first_non_native, columns, tiles, long }),
    )
}

#[test]
fn sparse_products_pick_by_format_columns_and_row_classes() {
    let ctx = cpu_with(|_| {});
    let pick = |nnz, native, first, columns, tiles| first_of(&ctx, &sparse_boundary(nnz, native, first, columns, tiles, false));
    assert_eq!(pick(0, true, false, 1, false), StrategyId::OPAQUE_SPARSE_EMPTY);
    assert_eq!(pick(500, false, true, 1, false), StrategyId::OPAQUE_SPARSE_SCATTER);
    assert_eq!(pick(500, false, true, 3, false), StrategyId::OPAQUE_SPARSE_SCATTER);
    // The second non-native product converts: nothing here fits (the op converts and selects again).
    assert!(candidates(&ctx, &sparse_boundary(500, false, false, 1, false, false)).next().is_none());
    assert_eq!(pick(500, true, false, 1, false), StrategyId::OPAQUE_SPARSE_ROWS);
    assert_eq!(pick(500, true, false, 1, true), StrategyId::OPAQUE_SPARSE_SPMV_MERGE);
    assert_eq!(pick(500, true, false, 2, true), StrategyId::OPAQUE_SPARSE_ROWS, "columns take the row kernels whatever the rows");
    let (_, params) = first(&ctx, &sparse_boundary(500, true, false, 1, true, false));
    assert!(matches!(params, Params::Opaque(p) if p.tiles == 1));
    let (_, params) = first(&ctx, &sparse_boundary(100 * 12, true, false, 1, false, false));
    assert!(matches!(params, Params::Opaque(p) if p.lanes_log == 2), "12 entries a row: 2^2 lanes");
}

fn first_of(ctx: &Context, boundary: &Boundary<'_>) -> StrategyId {
    first(ctx, boundary).0
}

// -- Opaque: the declarations --

#[test]
fn opaque_declarations_are_distinct_global_and_after_the_strategies() {
    let strategies = table(FamilyTag::Opaque).len() as u8;
    let mut ids: Vec<u8> = OPAQUE.iter().map(|op| op.id.1).collect();
    assert!(ids.iter().all(|&id| id >= strategies));
    ids.sort_unstable();
    ids.dedup();
    assert_eq!(ids.len(), OPAQUE.len());
    let mut labels: Vec<&str> = OPAQUE.iter().map(|op| op.label).collect();
    labels.sort_unstable();
    labels.dedup();
    assert_eq!(labels.len(), OPAQUE.len());
    assert!(OPAQUE.iter().all(|op| op.label.starts_with("opaque::") && op.label.bytes().all(|b| b.is_ascii_lowercase() || b == b'_' || b == b':')));
    assert!(OPAQUE.iter().all(|op| matches!(op.split(), crate::jit::structured::strategy::SplitClass::Global)));
}
