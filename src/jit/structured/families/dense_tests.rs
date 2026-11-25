//! The `fits` tables of the line-solve and factor families on both sides of every threshold, the legacy mirror of the inline conditions of
//! `f0e5723` over random shapes (kept while the ports go through their gate cycle, then deleted, as for sort, scan and scatter), the
//! env filters, forcing by name and nested selection. Built on the CPU backend with hand-made `Tuning`s, so it runs without a GPU.

use std::panic::Location;

use luma_core::BackendKind;

use crate::builtin::{thomas_plan, UNROLLED_BLOCK};
use crate::context::{Context, ContextOptions};
use crate::dense::lu::{calu_rows, leaf_width, CALU_W, SMALL};
use crate::dense::svd::{self, Kind};
use crate::dense::{chol, qr, trsv, tsqr, tri, Dims};
use crate::jit::structured::boundary::{dim, FactorKind, LineSolveAttrs};
use crate::jit::structured::families::factor::{boundary_flags, boundary_of, chol_stage, eigh_stage, lu_stage, qr_stage, svd_stage, tri_stage};
use crate::jit::structured::strategy::{candidates, first_candidate_in, Boundary, BoundaryAttrs, FamilyTag, StrategyId};
use crate::linalg::thomas_parts_scratch;
use crate::tuning::{LineSolver, Tuning, TuningProfile};

fn cpu_with(forced: &'static str, tune: impl FnOnce(&mut Tuning)) -> Context {
    Context::with_tuning(ContextOptions::default().backend(BackendKind::Cpu).calibrate(false).strategy(forced), tune).unwrap()
}

#[track_caller]
fn here() -> &'static Location<'static> {
    Location::caller()
}

/// A tiny deterministic generator (xorshift64*).
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }

    fn pick<T: Copy>(&mut self, items: &[T]) -> T {
        items[self.below(items.len() as u64) as usize]
    }
}

// -- LineSolve --

#[allow(clippy::too_many_arguments)]
fn line_boundary(block: u8, n: u32, stride: u32, lines: u32, systems: u32, len: u64, parts: bool, fused: bool, max_scratch: u64) -> Boundary<'static> {
    Boundary::new(
        FamilyTag::LineSolve,
        "tridiagonal",
        here(),
        [n, stride, lines, systems],
        BoundaryAttrs::LineSolve(LineSolveAttrs { block, parts, fused, starts: false, len, max_scratch }),
    )
}

fn first_line(ctx: &Context, boundary: &Boundary<'_>) -> StrategyId {
    candidates(ctx, boundary).next().expect("a line solve always fits").0
}

/// The order of today's checks (`linalg::run`, `run_parts`), written out.
#[allow(clippy::too_many_arguments)]
fn legacy_line(tuning: &Tuning, block: u32, n: u32, stride: u32, lines: u32, systems: u32, len: u64, parts: bool, fused: bool, max_buffer: u64) -> StrategyId {
    if crate::cyclic::plan(tuning, block, n, lines, stride, parts, fused, max_buffer).is_some() {
        return StrategyId::LINE_SOLVE_CYCLIC;
    }
    let thomas_kind = if tuning.native_solve { StrategyId::LINE_SOLVE_NATIVE } else { StrategyId::LINE_SOLVE_THOMAS };
    if !parts {
        return thomas_kind;
    }
    if (block <= UNROLLED_BLOCK || tuning.native_solve)
        && let thomas = thomas_plan(tuning, block, lines, stride, true)
        && thomas_parts_scratch(len as usize, block as usize, systems as usize, thomas.tiled, max_buffer).is_some()
    {
        thomas_kind
    } else {
        StrategyId::LINE_SOLVE_STACKED_FROM_PARTS
    }
}

#[test]
fn line_solve_fits_on_both_sides_of_every_threshold() {
    // Cyclic reduction against the Thomas algorithm: `lines * weight` around `line_cyclic_work`, for each (parts, stride == 1).
    let work = 1000u32;
    let ctx = cpu_with("", |t| {
        t.line_solver = LineSolver::Auto;
        t.native_solve = false;
        t.line_cyclic_work = work;
        t.line_bytes = 48 * 1024;
        t.profile = TuningProfile::Nvidia;
    });
    let pick = |block: u8, stride: u32, lines: u32, parts: bool| first_line(&ctx, &line_boundary(block, 64, stride, lines, 1, 64 * u64::from(lines), parts, false, u64::MAX));
    // Stacked scalars, contiguous: weight block^2 = 1; strided: 4 * block = 4.
    assert_eq!(pick(1, 1, work, false), StrategyId::LINE_SOLVE_CYCLIC);
    assert_eq!(pick(1, 1, work + 1, false), StrategyId::LINE_SOLVE_THOMAS);
    assert_eq!(pick(1, 8, work / 4, false), StrategyId::LINE_SOLVE_CYCLIC);
    assert_eq!(pick(1, 8, work / 4 + 1, false), StrategyId::LINE_SOLVE_THOMAS);
    // Parts along the contiguous axis: cyclic outright; strided: 2 * max(block, 2) = 4 with no shift.
    assert_eq!(pick(1, 1, 1_000_000, true), StrategyId::LINE_SOLVE_CYCLIC);
    assert_eq!(pick(1, 8, work / 4, true), StrategyId::LINE_SOLVE_CYCLIC);
    assert_eq!(pick(1, 8, work / 4 + 1, true), StrategyId::LINE_SOLVE_THOMAS);
    // Blocks past the unrolled size always take cyclic reduction as stacked blocks, and the stacked solver as parts that no kernel takes.
    assert_eq!(pick(UNROLLED_BLOCK as u8 + 1, 1, 1_000_000, false), StrategyId::LINE_SOLVE_CYCLIC);
    // `line_cyclic_work == 0` is Thomas.
    let never = cpu_with("", |t| {
        t.line_solver = LineSolver::Auto;
        t.line_cyclic_work = 0;
        t.native_solve = false;
    });
    assert_eq!(first_line(&never, &line_boundary(1, 64, 1, 1, 1, 64, false, false, u64::MAX)), StrategyId::LINE_SOLVE_THOMAS);

    // `LUMA_TRIDIAGONAL`: thomas and cyclic override the weights; native solves take no cyclic reduction on `auto`.
    let thomas = cpu_with("", |t| {
        t.line_solver = LineSolver::Thomas;
        t.native_solve = false;
    });
    assert_eq!(first_line(&thomas, &line_boundary(1, 64, 1, 1, 1, 64, false, false, u64::MAX)), StrategyId::LINE_SOLVE_THOMAS);
    let cyclic = cpu_with("", |t| {
        t.line_solver = LineSolver::Cyclic;
        t.line_cyclic_work = 0;
    });
    assert_eq!(first_line(&cyclic, &line_boundary(1, 64, 1, 100_000, 1, 64, false, false, u64::MAX)), StrategyId::LINE_SOLVE_CYCLIC);
    let native = cpu_with("", |t| {
        t.native_solve = true;
        t.line_solver = LineSolver::Auto;
    });
    assert_eq!(first_line(&native, &line_boundary(1, 64, 1, 1, 1, 64, false, false, u64::MAX)), StrategyId::LINE_SOLVE_NATIVE);
    let native_cyclic = cpu_with("", |t| {
        t.native_solve = true;
        t.line_solver = LineSolver::Cyclic;
    });
    assert_eq!(first_line(&native_cyclic, &line_boundary(1, 64, 1, 1, 1, 64, false, false, u64::MAX)), StrategyId::LINE_SOLVE_CYCLIC);

    // Parts: the Thomas kernel for blocks up to the unrolled size (with its scratch in one buffer), else the stacked solver.
    let plain = cpu_with("", |t| {
        t.line_solver = LineSolver::Thomas;
        t.native_solve = false;
    });
    let parts = |block: u8, max: u64| first_line(&plain, &line_boundary(block, 64, 1, 100, 1, 6400, true, false, max));
    assert_eq!(parts(UNROLLED_BLOCK as u8, u64::MAX), StrategyId::LINE_SOLVE_THOMAS);
    assert_eq!(parts(UNROLLED_BLOCK as u8 + 1, u64::MAX), StrategyId::LINE_SOLVE_STACKED_FROM_PARTS);
    let scratch = 6400 * u64::from(UNROLLED_BLOCK) * u64::from(UNROLLED_BLOCK) * 4;
    assert_eq!(parts(UNROLLED_BLOCK as u8, scratch), StrategyId::LINE_SOLVE_THOMAS);
    assert_eq!(parts(UNROLLED_BLOCK as u8, scratch - 1), StrategyId::LINE_SOLVE_STACKED_FROM_PARTS);
    // The native solve takes any block that fits a buffer.
    let native_parts = |block: u8, max: u64| first_line(&native, &line_boundary(block, 64, 1, 100, 1, 6400, true, false, max));
    assert_eq!(native_parts(8, u64::MAX), StrategyId::LINE_SOLVE_NATIVE);
    assert_eq!(native_parts(8, 6400 * 64 * 4 - 1), StrategyId::LINE_SOLVE_STACKED_FROM_PARTS);
}

#[test]
fn line_solve_is_forced_by_name_where_it_fits() {
    let line = |ctx: &Context| first_line(ctx, &line_boundary(1, 64, 1, 100_000, 1, 6_400_000, false, false, u64::MAX));
    // Many lines: Thomas on `auto`; `line_solve:cyclic` forces cyclic reduction like `LUMA_TRIDIAGONAL=cyclic`.
    let auto = cpu_with("", |t| {
        t.line_solver = LineSolver::Auto;
        t.line_cyclic_work = 1000;
        t.native_solve = false;
    });
    assert_eq!(line(&auto), StrategyId::LINE_SOLVE_THOMAS);
    let forced = cpu_with("line_solve:cyclic", |t| {
        t.line_solver = LineSolver::Auto;
        t.line_cyclic_work = 1000;
        t.native_solve = false;
    });
    assert_eq!(line(&forced), StrategyId::LINE_SOLVE_CYCLIC);
    // Thomas does not fit where the native solve is the plan: the default runs.
    let native = cpu_with("line_solve:thomas", |t| t.native_solve = true);
    assert_eq!(line(&native), StrategyId::LINE_SOLVE_NATIVE);
}

#[test]
fn line_solve_selection_mirrors_the_inline_conditions() {
    let mut rng = Rng(0x71e5);
    for _ in 0..10_000 {
        let solver = rng.pick(&[LineSolver::Auto, LineSolver::Thomas, LineSolver::Cyclic]);
        let (native, work, strided_from, line_bytes) = (rng.below(2) == 1, rng.pick(&[0u32, 100, 10_000, 1_000_000]), rng.pick(&[0u32, 16, 64]), rng.pick(&[1024u32, 16_384, 49_152]));
        let (profile, workgroup, units) = (rng.pick(&[TuningProfile::Cpu, TuningProfile::Nvidia]), rng.pick(&[32u32, 64, 128, 256]), rng.pick(&[1u32, 48, 82]));
        let ctx = cpu_with("", |t| {
            t.line_solver = solver;
            t.native_solve = native;
            t.line_cyclic_work = work;
            t.line_strided_from = strided_from;
            t.line_bytes = line_bytes;
            t.profile = profile;
            t.elementwise_workgroup = workgroup;
            t.compute_units = units;
        });
        let block = rng.pick(&[1u32, 2, 3, 4, 5, 6, 8]);
        let (n, stride, lines) = (1 + rng.below(600) as u32, rng.pick(&[1u32, 2, 8, 100]), 1 + rng.below(100_000) as u32);
        let (systems, parts, fused) = (1 + rng.below(3) as u32, rng.below(2) == 1, rng.below(2) == 1);
        let len = u64::from(n) * u64::from(lines);
        let max = rng.pick(&[u64::MAX, 1 << 20, 1 << 30]);
        let boundary = line_boundary(block as u8, n, stride, lines, if parts { systems } else { 1 }, len, parts, parts && fused, max);
        let want = legacy_line(ctx.tuning(), block, n, stride, lines, if parts { systems } else { 1 }, len, parts, parts && fused, max);
        assert_eq!(first_line(&ctx, &boundary), want, "{:?} native={native} work={work} block={block} n={n} stride={stride} lines={lines} parts={parts} fused={fused}", solver);
    }
}

// -- Factor --

fn dims(rows: usize, cols: usize, batch: usize) -> Dims {
    Dims { rows, cols, batch }
}

fn first_in(stage: &'static [crate::jit::structured::strategy::Strategy], ctx: &Context, boundary: &Boundary<'_>) -> StrategyId {
    first_candidate_in(stage, ctx, boundary).expect("a stage of the family fits").0
}

fn lu_pick(ctx: &Context, rows: usize, cols: usize, batch: usize) -> StrategyId {
    first_in(lu_stage(), ctx, &boundary_of(FactorKind::Lu, "lu", here(), dims(rows, cols, batch), 0))
}

/// What `leaf`, `factor` and `recurse` of `lu.rs` decided at `f0e5723`.
fn legacy_lu(ctx: &Context, rows: usize, cols: usize, batch: usize) -> StrategyId {
    if cols > leaf_width(rows, batch) {
        return StrategyId::LU_RECURSIVE;
    }
    if batch == 1 && rows > SMALL {
        let (cap, cap2) = calu_rows(ctx);
        if !(cols > CALU_W || cap < 64 || cap2 < 2 * cols || rows < cols) {
            let width = if rows <= cap { cap } else { cap.min(256) };
            let chunks = rows.div_ceil(width);
            let chunk_rows = rows.div_ceil(chunks);
            if rows - (chunks - 1) * chunk_rows >= cols {
                return if chunks == 1 { StrategyId::LU_CALU_SINGLE } else { StrategyId::LU_CALU_TOURNAMENT };
            }
        }
    }
    if rows <= SMALL { StrategyId::LU_SMALL } else { StrategyId::LU_PANEL }
}

#[test]
fn lu_fits_on_both_sides_of_every_threshold() {
    let ctx = cpu_with("", |_| {});
    // `rows` 64 / 65: the workgroup-memory leaf or a panel (the tournament for one matrix).
    assert_eq!(lu_pick(&ctx, 64, 32, 1), StrategyId::LU_SMALL);
    assert_eq!(lu_pick(&ctx, 65, 32, 1), StrategyId::LU_CALU_SINGLE);
    assert_eq!(lu_pick(&ctx, 65, 32, 2), StrategyId::LU_PANEL);
    // `cols` 32 / 33 around the panel width of a batch, 64 / 65 around the small leaf.
    assert_eq!(lu_pick(&ctx, 200, 32, 2), StrategyId::LU_PANEL);
    assert_eq!(lu_pick(&ctx, 200, 33, 2), StrategyId::LU_RECURSIVE);
    assert_eq!(lu_pick(&ctx, 64, 64, 2), StrategyId::LU_SMALL);
    assert_eq!(lu_pick(&ctx, 66, 64, 1), StrategyId::LU_RECURSIVE);
    // One chunk up to `calu_rows`, a tournament of chunks above.
    let (cap, _) = calu_rows(&ctx);
    assert!(cap >= 64);
    assert_eq!(lu_pick(&ctx, cap, 16, 1), StrategyId::LU_CALU_SINGLE);
    assert_eq!(lu_pick(&ctx, cap + 1, 16, 1), StrategyId::LU_CALU_TOURNAMENT);
    // A forced strategy that does not fit (a tournament of a matrix that fits one workgroup) is not taken.
    let forced = cpu_with("factor:lu_calu_tournament", |_| {});
    assert_eq!(lu_pick(&forced, 100, 16, 1), StrategyId::LU_CALU_SINGLE);
    let forced = cpu_with("factor:lu_panel", |_| {});
    assert_eq!(lu_pick(&forced, 100, 16, 1), StrategyId::LU_PANEL);
}

#[test]
fn factor_selection_mirrors_the_inline_conditions() {
    let ctx = cpu_with("", |_| {});
    let mut rng = Rng(0xf4c7);
    for _ in 0..10_000 {
        let (rows, batch) = (1 + rng.below(1500) as usize, 1 + rng.below(3) as usize);
        let cols = 1 + rng.below(rows as u64) as usize;
        assert_eq!(lu_pick(&ctx, rows, cols, batch), legacy_lu(&ctx, rows, cols, batch), "lu {rows}x{cols} batch {batch}");

        // Cholesky: `rows <= leaf`.
        let chol_pick = first_in(chol_stage(), &ctx, &boundary_of(FactorKind::Cholesky, "cholesky", here(), dims(rows, rows, batch), 0));
        assert_eq!(chol_pick, if rows <= chol::leaf(&ctx) { StrategyId::CHOL_SMALL } else { StrategyId::CHOL_BLOCKED }, "cholesky {rows}");

        // QR: the TSQR panel for one matrix of up to 32 columns that `choose` takes, else the leaf, else the recursion.
        let qr_pick = first_in(qr_stage(), &ctx, &boundary_of(FactorKind::Qr, "qr", here(), dims(rows, cols, batch), 0));
        let legacy_qr = if batch == 1 && cols <= 32 && tsqr::choose(&ctx, rows, cols).is_some() {
            StrategyId::QR_TSQR_PANEL
        } else if cols <= if batch > 1 || rows <= 64 { 32 } else { crate::dense::panel_width() } {
            StrategyId::QR_LEAF
        } else {
            StrategyId::QR_RECURSIVE
        };
        assert_eq!(qr_pick, legacy_qr, "qr {rows}x{cols} batch {batch}");

        // Triangular solves: substitution up to 64 rows, the look-back kernel, the blocked one.
        let k = 1 + rng.below(40) as usize;
        let inverse = rng.below(2) == 1;
        let tri_pick = first_in(tri_stage(), &ctx, &boundary_flags(FactorKind::SolveTriangular, "solve_triangular", here(), dims(rows, rows, batch), k, inverse, false, false));
        let legacy_tri = if rows <= tri::BASE {
            StrategyId::TRI_SUBSTITUTE
        } else if trsv::applies(&ctx, rows, k, batch) {
            StrategyId::TRI_TRSV
        } else {
            StrategyId::TRI_BLOCKED
        };
        assert_eq!(tri_pick, legacy_tri, "triangular {rows} x {k} batch {batch}");

        // SVD and eigh: the workgroup-memory kernel, the QR route, the Jacobi variants.
        let (rr, cc) = (rows.max(cols), rows.min(cols));
        let (vectors, exact) = (rng.below(2) == 1, rng.below(2) == 1);
        let svd_pick = first_in(svd_stage(), &ctx, &boundary_flags(FactorKind::Svd, "svd", here(), dims(rr, cc, batch), 0, false, exact, vectors));
        let legacy_svd = if svd::plan(&ctx, Kind::Svd, vectors, rr, cc).is_some() {
            StrategyId::SVD_SMALL
        } else if exact && svd::precondition(cc) {
            StrategyId::SVD_PRECONDITIONED
        } else if cc >= svd::BLOCK_MIN {
            StrategyId::SVD_BLOCK_JACOBI
        } else {
            StrategyId::SVD_JACOBI
        };
        assert_eq!(svd_pick, legacy_svd, "svd {rr}x{cc}");
        let eigh_pick = first_in(eigh_stage(), &ctx, &boundary_flags(FactorKind::Eigh, "eigh", here(), dims(rows, rows, batch), 0, false, true, vectors));
        let legacy_eigh = if svd::plan(&ctx, Kind::Eigh, vectors, rows, rows).is_some() {
            StrategyId::EIGH_SMALL
        } else if rows >= svd::BLOCK_MIN {
            StrategyId::EIGH_BLOCK_JACOBI
        } else {
            StrategyId::EIGH_JACOBI
        };
        assert_eq!(eigh_pick, legacy_eigh, "eigh {rows}");
    }
}

#[test]
fn cholesky_qr_and_triangular_solves_fit_on_both_sides_of_their_thresholds() {
    let ctx = cpu_with("", |t| t.lookback_scan = true);
    let leaf = chol::leaf(&ctx);
    let chol_pick = |rows| first_in(chol_stage(), &ctx, &boundary_of(FactorKind::Cholesky, "cholesky", here(), dims(rows, rows, 1), 0));
    assert_eq!(chol_pick(leaf), StrategyId::CHOL_SMALL);
    assert_eq!(chol_pick(leaf + 1), StrategyId::CHOL_BLOCKED);

    // QR: 32 / 33 columns around the TSQR panel, `gmax * 32` rows around its single level.
    let qr_pick = |rows, cols, batch| first_in(qr_stage(), &ctx, &boundary_of(FactorKind::Qr, "qr", here(), dims(rows, cols, batch), 0));
    assert!(tsqr::choose(&ctx, 1000, 32).is_some());
    assert_eq!(qr_pick(1000, 32, 1), StrategyId::QR_TSQR_PANEL);
    assert_eq!(qr_pick(1000, 33, 1), StrategyId::QR_RECURSIVE);
    assert_eq!(qr_pick(1000, 32, 2), StrategyId::QR_LEAF);
    assert_eq!(qr_pick(64, 32, 2), StrategyId::QR_LEAF);
    assert_eq!(qr_pick(64, 33, 2), StrategyId::QR_RECURSIVE);
    assert_eq!(qr_pick(65, 33, 2), StrategyId::QR_RECURSIVE);
    let gmax = (ctx.info().limits.max_workgroup_invocations.min(ctx.info().limits.max_workgroup_size[0]) as usize / 32).min(16);
    for rows in [gmax * 32, gmax * 32 + 1] {
        // A tall panel is cut in one level up to `gmax * 32` rows, in several above.
        let found = tsqr::choose(&ctx, rows, 8).expect("a tall panel is cut");
        assert_eq!(found.k == 1, rows <= gmax * 32, "{rows} rows");
        assert_eq!(qr_pick(rows, 8, 1), StrategyId::QR_TSQR_PANEL);
    }

    // Triangular solves: 64 / 65 rows, 32 / 33 columns of the look-back kernel, a batch.
    let tri_pick = |rows, k, batch, ctx: &Context| first_in(tri_stage(), ctx, &boundary_flags(FactorKind::SolveTriangular, "solve_triangular", here(), dims(rows, rows, batch), k, false, false, false));
    assert_eq!(tri_pick(64, 8, 1, &ctx), StrategyId::TRI_SUBSTITUTE);
    assert_eq!(tri_pick(65, 8, 1, &ctx), StrategyId::TRI_TRSV);
    assert_eq!(tri_pick(65, 32, 1, &ctx), StrategyId::TRI_TRSV);
    assert_eq!(tri_pick(65, 33, 1, &ctx), StrategyId::TRI_BLOCKED);
    assert_eq!(tri_pick(65, 8, 2, &ctx), StrategyId::TRI_BLOCKED);
    // Without the look-back, the blocked solve; the diagonal blocks are inverted from `LUMA_TRSV_INV_MIN` rows (768) when asked.
    let no_lookback = cpu_with("", |t| t.lookback_scan = false);
    assert_eq!(tri_pick(65, 8, 1, &no_lookback), StrategyId::TRI_BLOCKED);
    for (rows, inverted) in [(767usize, false), (768, true)] {
        assert_eq!(trsv::plan(rows, 1, true).1, inverted, "{rows} rows");
        assert!(!trsv::plan(rows, 1, false).1, "{rows} rows, not asked");
    }

    // SVD: 96 / 97 columns around the workgroup-memory kernel (given room), 127 / 128 around block Jacobi, 1023 / 1024 around the QR route.
    let svd_pick = |rows, cols, exact| first_in(svd_stage(), &ctx, &boundary_flags(FactorKind::Svd, "svd", here(), dims(rows, cols, 1), 0, false, exact, true));
    assert_eq!(svd_pick(96, 96, true), if svd::plan(&ctx, Kind::Svd, true, 96, 96).is_some() { StrategyId::SVD_SMALL } else { StrategyId::SVD_JACOBI });
    assert_eq!(svd_pick(97, 97, true), StrategyId::SVD_JACOBI);
    assert_eq!(svd_pick(127, 127, true), StrategyId::SVD_JACOBI);
    assert_eq!(svd_pick(128, 128, true), StrategyId::SVD_BLOCK_JACOBI);
    assert_eq!(svd_pick(1023, 1023, true), StrategyId::SVD_BLOCK_JACOBI);
    assert_eq!(svd_pick(1024, 1024, true), StrategyId::SVD_PRECONDITIONED);
    // A rank tolerance keeps the plain iteration.
    assert_eq!(svd_pick(1024, 1024, false), StrategyId::SVD_BLOCK_JACOBI);
    // Eigh is never preconditioned.
    let eigh_pick = |n| first_in(eigh_stage(), &ctx, &boundary_flags(FactorKind::Eigh, "eigh", here(), dims(n, n, 1), 0, false, true, true));
    assert_eq!(eigh_pick(127), StrategyId::EIGH_JACOBI);
    assert_eq!(eigh_pick(1024), StrategyId::EIGH_BLOCK_JACOBI);
}

/// The refined solves are asked for by their own kinds: no plain stage ever sees them, and a plain kind no refined stage.
#[test]
fn refined_strategies_are_reached_only_by_their_kinds() {
    use crate::jit::structured::families::factor::refined_stage;
    let ctx = cpu_with("", |_| {});
    for (kind, want) in [(FactorKind::SolveRefined, StrategyId::LU_REFINED_F32), (FactorKind::SolveSpdRefined, StrategyId::CHOL_REFINED_F32)] {
        let boundary = boundary_of(kind, "solve_refined", here(), dims(100, 100, 1), 1);
        assert_eq!(first_in(refined_stage(), &ctx, &boundary), want);
        // A non-square system fits neither.
        assert!(first_candidate_in(refined_stage(), &ctx, &boundary_of(kind, "solve_refined", here(), dims(100, 90, 1), 1)).is_none());
    }
    for kind in [FactorKind::Lu, FactorKind::Cholesky, FactorKind::Qr, FactorKind::SolveTriangular, FactorKind::Svd, FactorKind::Eigh] {
        let boundary = boundary_of(kind, "x", here(), dims(100, 100, 1), 1);
        assert!(first_candidate_in(refined_stage(), &ctx, &boundary).is_none(), "{kind:?}");
    }
    for stage in [lu_stage(), chol_stage(), qr_stage(), tri_stage(), svd_stage(), eigh_stage()] {
        for kind in [FactorKind::SolveRefined, FactorKind::SolveSpdRefined] {
            assert!(first_candidate_in(stage, &ctx, &boundary_of(kind, "x", here(), dims(100, 100, 1), 1)).is_none());
        }
    }
}

/// The names of `LUMA_STRATEGY` for the stages that do not fit stay quiet: the default runs and `Decision`s name it.
#[test]
fn factor_strategies_are_forced_by_name_and_log_their_stage() {
    use crate::array::Array;
    use crate::profile::DecisionSite;

    let make = |forced: &'static str| cpu_with(forced, |t| t.lookback_scan = true);
    let logged = |ctx: &Context, run: &dyn Fn()| -> Vec<String> {
        ctx.record_decisions(true);
        run();
        ctx.sync().unwrap();
        let decisions = ctx.take_decisions();
        ctx.record_decisions(false);
        decisions.iter().filter(|d| d.site == DecisionSite::FactorPlan).map(|d| d.chosen.to_string()).collect()
    };
    // `chol_blocked` on a 32 x 32 matrix does not fit (it needs rows > leaf): the small kernel runs.
    let ctx = make("factor:chol_blocked");
    let a = Array::<f32>::from_slice_on(&ctx, &(0..32 * 32).map(|i| if i % 33 == 0 { 40.0 } else { 0.01 }).collect::<Vec<_>>(), [32, 32]).unwrap();
    assert_eq!(logged(&ctx, &|| a.cholesky().eval().unwrap()), ["chol_small"]);
    // The nested stages of a solve: the LU's recursion, its panels, then the two triangular solves.
    let ctx = make("");
    let a = Array::<f32>::from_slice_on(&ctx, &(0..100 * 100).map(|i| if i % 101 == 0 { 50.0 } else { ((i * 37) % 11) as f32 * 0.1 }).collect::<Vec<_>>(), [100, 100]).unwrap();
    let b = Array::<f32>::from_slice_on(&ctx, &[1.0; 100], [1, 100]).unwrap();
    let labels = logged(&ctx, &|| a.solve(&b).eval().unwrap());
    assert_eq!(labels[0], "lu_recursive", "{labels:?}");
    assert!(labels.iter().any(|l| l == "lu_calu_single" || l == "lu_panel"), "{labels:?}");
    assert!(labels.iter().any(|l| l.starts_with("tri_")), "{labels:?}");
    // Forcing the panel kernel for the leaves of the same solve.
    let ctx = make("factor:lu_panel");
    let a = Array::<f32>::from_slice_on(&ctx, &(0..100 * 100).map(|i| if i % 101 == 0 { 50.0 } else { ((i * 37) % 11) as f32 * 0.1 }).collect::<Vec<_>>(), [100, 100]).unwrap();
    let labels = logged(&ctx, &|| a.lu().lu.eval().unwrap());
    assert!(labels.iter().any(|l| l == "lu_panel") && labels.iter().all(|l| l != "lu_calu_single"), "{labels:?}");
    let _ = dim(1);
}

/// `solve_refined` and the plain solve: two contexts of one process with different `TF32_LINALG` read their own policy (the trailing
/// updates' `product` takes it from the context, not from a process-global).
#[test]
fn tf32_linalg_is_read_from_the_context() {
    use crate::numerics::{Numerics, Reduced};
    let tf32 = Context::with_options(ContextOptions::default().backend(BackendKind::Cpu).calibrate(false).numerics(Numerics::default().with_reduced(Reduced::TF32_LINALG, true))).unwrap();
    let exact = Context::with_options(ContextOptions::default().backend(BackendKind::Cpu).calibrate(false)).unwrap();
    assert!(crate::dense::tf32_updates(&tf32));
    assert!(!crate::dense::tf32_updates(&exact));
    assert!(crate::dense::tf32_updates(&tf32) && !crate::dense::tf32_updates(&exact), "the second context must not see the first one's policy");
}

#[test]
fn the_defaults_that_are_not_filters_are_unchanged() {
    // `fits` of the lookup of an unknown stage is never called with a foreign kind: every table entry names its stage.
    let ctx = cpu_with("", |_| {});
    let boundary = boundary_of(FactorKind::Lu, "lu", here(), dims(10, 10, 1), 0);
    for strategy in &crate::jit::structured::strategy::table(FamilyTag::Factor)[5..] {
        assert!((strategy.fits)(&crate::jit::structured::strategy::Subject::Boundary(&boundary), &ctx, &crate::jit::structured::strategy::Residencies::DEVICE).is_none(), "{} fits an LU", strategy.name);
    }
    let _ = (qr::PANEL, dim(0));
}
