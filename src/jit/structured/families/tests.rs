//! The `fits` tables of the boundary families on both sides of every threshold, and the legacy mirror: the conditions as they were
//! written inline before the registry (`f0e5723`), checked against the registry's first candidate over a grid of random shapes
//! and tunings. Built on the CPU backend with hand-made `Tuning`s, so it runs without a GPU.

use std::panic::Location;

use luma_core::BackendKind;

use crate::context::{Context, ContextOptions};
use crate::jit::structured::boundary::{dim, ScanAttrs, ScanOperand, ScatterAttrs, ScatterOp, SortAttrs, SortParams, SortPayload, SortStage};
use crate::jit::structured::strategy::{candidates, Boundary, BoundaryAttrs, FamilyTag, Params, StrategyId};
use crate::sort::{ballot_bytes, sweep_bytes, sweep_per_thread, WORKGROUP};
use crate::scan::{strided_block, target_threads, LOOKBACK_CHUNK, SCAN_CHUNK, SHORT_LINE, WARP_MIN_LINES};
use crate::tuning::Tuning;

fn cpu_with(tune: impl FnOnce(&mut Tuning)) -> Context {
    Context::with_tuning(ContextOptions::default().backend(BackendKind::Cpu).calibrate(false).strategy(""), tune).unwrap()
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

    /// Log-uniform in `1..=max`, so the thresholds (255, 4096, 65536, ...) are all within reach.
    fn log(&mut self, max: u64) -> u64 {
        let bits = 64 - max.leading_zeros();
        let width = 1 + self.next() % u64::from(bits);
        (1 + self.next() % (1 << width)).min(max)
    }

    fn flag(&mut self) -> bool {
        self.next() & 1 == 1
    }
}

// -- Scatter --

fn scatter_boundary(atomic: ScatterOp, base_len: usize, count: usize, view: bool, read_view: bool, host_site: bool) -> Boundary<'static> {
    Boundary::new(
        FamilyTag::Scatter,
        "put",
        here(),
        [dim(base_len), dim(count), 0, 0],
        BoundaryAttrs::Scatter(ScatterAttrs { atomic, base_len: base_len as u64, count: count as u64, base_is_view: view, base_is_read_view: read_view, host_site }),
    )
}

fn first(ctx: &Context, boundary: &Boundary<'_>) -> StrategyId {
    candidates(ctx, boundary).next().expect("a strategy fits").0
}

#[test]
fn scatter_fits_on_both_sides_of_the_private_threshold() {
    let ctx = cpu_with(|_| {});
    let pick = |base: usize, count: usize| first(&ctx, &scatter_boundary(ScatterOp::Add, base, count, false, false, false));
    assert_eq!(pick(4096, 16 * 4096 - 1), StrategyId::SCATTER_ATOMIC_GLOBAL);
    assert_eq!(pick(4096, 16 * 4096), StrategyId::SCATTER_ATOMIC_PRIVATE);
    assert_eq!(pick(4097, 16 * 4097 + 100), StrategyId::SCATTER_ATOMIC_GLOBAL);
    assert_eq!(pick(1, 0), StrategyId::SCATTER_ATOMIC_GLOBAL);
    assert_eq!(pick(1, 16), StrategyId::SCATTER_ATOMIC_PRIVATE);
    for op in [ScatterOp::Min, ScatterOp::Max] {
        assert_eq!(first(&ctx, &scatter_boundary(op, 100, 1600, false, false, false)), StrategyId::SCATTER_ATOMIC_PRIVATE);
    }
    assert_eq!(first(&ctx, &scatter_boundary(ScatterOp::None, 100, 1600, false, false, false)), StrategyId::SCATTER_PLAIN);
}

/// The inline condition of `put_with` and `jit::assign::scatter` at `f0e5723`.
fn legacy_scatter(host_ok: bool, atomic: ScatterOp, base: usize, count: usize, view: bool, read_view: bool, host_site: bool) -> StrategyId {
    if host_site {
        if atomic == ScatterOp::None && host_ok && !view && !read_view && base <= 65_536 && count <= 65_536 {
            return StrategyId::SCATTER_HOST;
        }
        // Not a host site choice: the caller goes on to the kernel.
        return StrategyId::SCATTER_PLAIN;
    }
    match atomic {
        ScatterOp::None => StrategyId::SCATTER_PLAIN,
        _ if base <= 4096 && count / 16 >= base => StrategyId::SCATTER_ATOMIC_PRIVATE,
        _ => StrategyId::SCATTER_ATOMIC_GLOBAL,
    }
}

#[test]
fn scatter_matches_the_legacy_conditions() {
    let ctx = cpu_with(|_| {});
    let mut rng = Rng(0x5ca7_7e12);
    for _ in 0..10_000 {
        let atomic = [ScatterOp::None, ScatterOp::Add, ScatterOp::Min, ScatterOp::Max][(rng.next() % 4) as usize];
        let (base, count) = (rng.log(1 << 20) as usize, rng.log(1 << 24) as usize);
        let (view, read_view, host_site) = (rng.flag() && rng.flag(), rng.flag() && rng.flag(), rng.flag());
        let got = candidates(&ctx, &scatter_boundary(atomic, base, count, view, read_view, host_site)).next().map(|(id, _)| id);
        let want = legacy_scatter(ctx.can_run_on_host(), atomic, base, count, view, read_view, host_site);
        // `can_run_on_host()` is false on the CPU backend: the host site has no candidate and the caller goes on to the kernel.
        if host_site {
            assert_eq!(got, None, "{atomic:?} {base} {count}");
            assert_eq!(want, StrategyId::SCATTER_PLAIN);
        } else {
            assert_eq!(got, Some(want), "{atomic:?} {base} {count}");
        }
    }
}

// -- Scan --

fn scan_boundary(n: u32, stride: u32, lines: u32, grouping_free: bool, by_key: bool) -> Boundary<'static> {
    Boundary::new(FamilyTag::Scan, "scan", here(), [n, stride, lines, 0], BoundaryAttrs::Scan(ScanAttrs { grouping_free, by_key, operand: ScanOperand::Buffer }))
}

#[test]
fn scan_fits_on_both_sides_of_every_threshold() {
    let ctx = cpu_with(|t| {
        t.subgroup32 = true;
        t.lookback_scan = true;
        t.compute_units = crate::tuning::REFERENCE_COMPUTE_UNITS;
    });
    let pick = |n, stride, lines, free| first(&ctx, &scan_boundary(n, stride, lines, free, false));
    // rows up to 255 cells.
    assert_eq!(pick(255, 1, 1, true), StrategyId::SCAN_ROWS);
    assert_eq!(pick(256, 1, 1, true), StrategyId::SCAN_CHUNK_SINGLE);
    // warp from `scaled(128)` lines.
    let lines = ctx.tuning().scaled(WARP_MIN_LINES);
    assert_eq!(pick(1000, 1, lines - 1, true), StrategyId::SCAN_CHUNK_SINGLE);
    assert_eq!(pick(1000, 1, lines, true), StrategyId::SCAN_WARP);
    // lookback above 4096 cells, for operators that group freely; the two-pass chunks otherwise.
    assert_eq!(pick(LOOKBACK_CHUNK, 1, 1, true), StrategyId::SCAN_CHUNK_THREE_PASS);
    assert_eq!(pick(LOOKBACK_CHUNK + 1, 1, 1, true), StrategyId::SCAN_LOOKBACK);
    assert_eq!(pick(LOOKBACK_CHUNK + 1, 1, 1, false), StrategyId::SCAN_CHUNK_THREE_PASS);
    // one chunk of 1024 or several.
    assert_eq!(pick(SCAN_CHUNK, 1, 1, false), StrategyId::SCAN_CHUNK_SINGLE);
    assert_eq!(pick(SCAN_CHUNK + 1, 1, 1, false), StrategyId::SCAN_CHUNK_THREE_PASS);
    // strided: one invocation per line up to 256 cells or from 65536 lines, blocks between.
    assert_eq!(pick(256, 2, 10, true), StrategyId::SCAN_SEQ);
    assert_eq!(pick(257, 2, 10, true), StrategyId::SCAN_FOLD_WALK);
    assert_eq!(pick(1000, 2, 65_535, true), StrategyId::SCAN_FOLD_WALK);
    assert_eq!(pick(1000, 2, 65_536, true), StrategyId::SCAN_SEQ);
    // by key: one chunk of 4096 cells or several.
    let by_key = |n| first(&ctx, &scan_boundary(n, 1, 3, false, true));
    assert_eq!(by_key(4096), StrategyId::SCAN_BY_KEY_SINGLE);
    assert_eq!(by_key(4097), StrategyId::SCAN_BY_KEY_CHUNKED);
    // Without the capabilities the same shapes take the two-pass chunks.
    let plain = cpu_with(|t| {
        t.subgroup32 = false;
        t.lookback_scan = false;
    });
    assert_eq!(first(&plain, &scan_boundary(1000, 1, 100_000, true, false)), StrategyId::SCAN_CHUNK_SINGLE);
    assert_eq!(first(&plain, &scan_boundary(100_000, 1, 1, true, false)), StrategyId::SCAN_CHUNK_THREE_PASS);
}

/// `Lines::scan` at `f0e5723`, in its order.
fn legacy_scan(tuning: &Tuning, n: u32, stride: u32, lines: u32, grouping_free: bool) -> StrategyId {
    if stride == 1 {
        if n <= SHORT_LINE {
            return StrategyId::SCAN_ROWS;
        }
        if lines >= tuning.scaled(WARP_MIN_LINES) && tuning.subgroup32 {
            return StrategyId::SCAN_WARP;
        }
        if tuning.lookback_scan && n > LOOKBACK_CHUNK && grouping_free {
            return StrategyId::SCAN_LOOKBACK;
        }
        return if n.div_ceil(SCAN_CHUNK) == 1 { StrategyId::SCAN_CHUNK_SINGLE } else { StrategyId::SCAN_CHUNK_THREE_PASS };
    }
    match strided_block(n, lines, target_threads(tuning)) {
        None => StrategyId::SCAN_SEQ,
        Some(_) => StrategyId::SCAN_FOLD_WALK,
    }
}

#[test]
fn scan_matches_the_legacy_conditions() {
    let mut rng = Rng(0x00c0_ffee);
    let mut checked = 0;
    for variant in 0..8u32 {
        let ctx = cpu_with(|t| {
            t.subgroup32 = variant & 1 != 0;
            t.lookback_scan = variant & 2 != 0;
            t.compute_units = [82, 48, 14][(variant >> 2) as usize % 3] + variant;
        });
        for _ in 0..1250 {
            let n = rng.log(1 << 24) as u32;
            let stride = if rng.flag() { 1 } else { rng.log(64) as u32 };
            let lines = rng.log(1 << 18) as u32;
            let free = rng.flag();
            let got = first(&ctx, &scan_boundary(n, stride, lines, free, false));
            assert_eq!(got, legacy_scan(ctx.tuning(), n, stride, lines, free), "n={n} stride={stride} lines={lines} free={free} variant={variant}");
            checked += 1;
        }
    }
    assert_eq!(checked, 10_000);
}

// -- Sort --

fn sort_boundary(stage: SortStage, dims: [u32; 4], payload: SortPayload, value4: bool) -> Boundary<'static> {
    Boundary::new(
        FamilyTag::Sort,
        "sort",
        here(),
        dims,
        BoundaryAttrs::Sort(SortAttrs { stage, payload, errored: false, value4, key_bits: 32, len: u64::from(dims[0]) * u64::from(dims[2].max(1)) }),
    )
}

fn radix(ctx: &Context, n: u32, payload: SortPayload) -> StrategyId {
    first(ctx, &sort_boundary(SortStage::Radix, [n, 1, 1, 0], payload, false))
}

#[test]
fn sort_fits_on_both_sides_of_every_threshold() {
    let storage = sweep_bytes(16);
    let sweeping = cpu_with(|t| {
        t.subgroup32 = true;
        t.lookback_scan = true;
        t.workgroup_storage = storage;
    });
    assert_eq!(radix(&sweeping, 1 << 20, SortPayload::None), StrategyId::SORT_RADIX_SWEEP);
    assert_eq!(radix(&sweeping, (1 << 30) - 1, SortPayload::None), StrategyId::SORT_RADIX_SWEEP);
    assert_eq!(radix(&sweeping, 1 << 30, SortPayload::None), StrategyId::SORT_RADIX_BALLOTS);
    // The sweep's tile: 32 keys per invocation where they fit, fewer with a payload.
    let params = candidates(&sweeping, &sort_boundary(SortStage::Radix, [1 << 20, 1, 1, 0], SortPayload::Index, false)).next().unwrap().1;
    assert_eq!(params, Params::Sort(SortParams { per_thread: sweep_per_thread(sweeping.tuning(), true), tile: WORKGROUP * 16, passes: 4 }));
    // One byte of workgroup memory short of the sweep: ballots if they fit, else the count/scan/scatter passes.
    let short = cpu_with(|t| {
        t.subgroup32 = true;
        t.lookback_scan = true;
        t.workgroup_storage = storage - 1;
    });
    assert_eq!(radix(&short, 1 << 20, SortPayload::None), if ballot_bytes() < storage { StrategyId::SORT_RADIX_BALLOTS } else { StrategyId::SORT_RADIX_COUNT_SCAN });
    let ballots = cpu_with(|t| {
        t.subgroup32 = true;
        t.lookback_scan = false;
        t.workgroup_storage = ballot_bytes();
    });
    assert_eq!(radix(&ballots, 1 << 20, SortPayload::None), StrategyId::SORT_RADIX_BALLOTS);
    let tight = cpu_with(|t| {
        t.subgroup32 = true;
        t.workgroup_storage = ballot_bytes() - 1;
    });
    assert_eq!(radix(&tight, 1 << 20, SortPayload::None), StrategyId::SORT_RADIX_COUNT_SCAN);
    let plain = cpu_with(|t| t.subgroup32 = false);
    assert_eq!(radix(&plain, 1 << 20, SortPayload::None), StrategyId::SORT_RADIX_COUNT_SCAN);

    // Axis sorts: one kernel per line up to `max_line` (256 on the CPU backend), the sort by line above; none with too little
    // workgroup memory.
    let ctx = cpu_with(|t| t.workgroup_storage = 16384);
    let max = crate::segsort::max_line(&ctx);
    assert_eq!(max, 256);
    let axis = |n| first(&ctx, &sort_boundary(SortStage::Axis, [n, 4, 8, 0], SortPayload::None, false));
    assert_eq!(axis(max), StrategyId::SORT_LINES);
    assert_eq!(axis(max + 1), StrategyId::SORT_LINES_RADIX);
    let small = cpu_with(|t| t.workgroup_storage = 12_287);
    assert_eq!(first(&small, &sort_boundary(SortStage::Axis, [2, 1, 8, 0], SortPayload::None, false)), StrategyId::SORT_LINES_RADIX);

    // By key: a 4-byte payload travels with the keys, anything else through the permutation.
    let by_key = |value4| first(&ctx, &sort_boundary(SortStage::ByKey, [1000, 1, 1, 0], SortPayload::Buffer, value4));
    assert_eq!(by_key(true), StrategyId::SORT_BY_KEY_PAYLOAD);
    assert_eq!(by_key(false), StrategyId::SORT_BY_KEY_PERMUTATION);
    // By key along an axis: several lines, else the flat by-key sort (no candidate here).
    assert_eq!(first(&ctx, &sort_boundary(SortStage::ByKeyAxis, [10, 4, 4, 0], SortPayload::None, false)), StrategyId::SORT_BY_KEY_AXIS);
    assert!(candidates(&ctx, &sort_boundary(SortStage::ByKeyAxis, [10, 1, 1, 0], SortPayload::None, false)).next().is_none());
    // The entry of a sort has the host path only, and the CPU backend has none.
    assert!(candidates(&ctx, &sort_boundary(SortStage::Entry, [1000, 1, 1, 0], SortPayload::None, false)).next().is_none());
}

/// `radix_sort` at `f0e5723`.
fn legacy_radix(tuning: &Tuning, n: u32) -> StrategyId {
    let ballots = tuning.subgroup32 && ballot_bytes() <= tuning.workgroup_storage;
    if tuning.lookback_scan && ballots && sweep_bytes(16) <= tuning.workgroup_storage && n < 1 << 30 {
        StrategyId::SORT_RADIX_SWEEP
    } else if ballots {
        StrategyId::SORT_RADIX_BALLOTS
    } else {
        StrategyId::SORT_RADIX_COUNT_SCAN
    }
}

/// `run_sort_axis` at `f0e5723`.
fn legacy_axis(ctx: &Context, n: u32) -> StrategyId {
    if n <= crate::segsort::max_line(ctx) { StrategyId::SORT_LINES } else { StrategyId::SORT_LINES_RADIX }
}

#[test]
fn sort_matches_the_legacy_conditions() {
    let mut rng = Rng(0x0050_77ed);
    let storages = [0, 4096, 12_287, 12_288, 16_384, ballot_bytes() - 1, ballot_bytes(), sweep_bytes(16) - 1, sweep_bytes(16), sweep_bytes(32), 49_152, 65_536];
    let mut checked = 0;
    for variant in 0..8u32 {
        let ctx = cpu_with(|t| {
            t.subgroup32 = variant & 1 != 0;
            t.lookback_scan = variant & 2 != 0;
            t.workgroup_storage = storages[(variant as usize * 3 + 1) % storages.len()];
        });
        for round in 0..1250 {
            // Every 4th round re-draws the storage edge so each threshold is hit with each flag combination.
            let n = if round % 50 == 0 { (1u32 << 30) - 1 + (round / 50) % 3 } else { rng.log(u64::from(u32::MAX)) as u32 };
            assert_eq!(radix(&ctx, n, SortPayload::None), legacy_radix(ctx.tuning(), n), "n={n} variant={variant}");
            let line = rng.log(1 << 14) as u32;
            assert_eq!(first(&ctx, &sort_boundary(SortStage::Axis, [line, 1, 3, 0], SortPayload::None, false)), legacy_axis(&ctx, line), "line={line} variant={variant}");
            checked += 1;
        }
    }
    assert_eq!(checked, 10_000);
}

/// Forcing a strategy: the output is the same, the Decision says which ran; one that does not fit the call is not taken.
#[test]
fn strategies_of_sort_scan_and_scatter_are_forced_by_name() {
    use crate::array::Array;
    use crate::profile::DecisionSite;

    fn context(forced: &'static str) -> Context {
        Context::with_tuning(ContextOptions::default().backend(BackendKind::Cpu).calibrate(false).strategy(forced), |t| {
            t.subgroup32 = true;
            t.lookback_scan = true;
            t.workgroup_storage = 49_152;
        })
        .unwrap()
    }
    fn logged(ctx: &Context, site: DecisionSite, run: impl FnOnce()) -> Vec<String> {
        ctx.record_decisions(true);
        run();
        ctx.sync().unwrap();
        let decisions = ctx.take_decisions();
        ctx.record_decisions(false);
        decisions.iter().filter(|d| d.site == site).map(|d| d.chosen.to_string()).collect()
    }
    let mut rng = Rng(7);
    let keys: Vec<u32> = (0..(1 << 16)).map(|_| rng.next() as u32).collect();
    let mut want = keys.clone();
    want.sort_unstable();

    // Sort: the count/scan/scatter passes where the sweep fits.
    let free = context("");
    assert_eq!(radix(&free, keys.len() as u32, SortPayload::None), StrategyId::SORT_RADIX_SWEEP);
    let forced = context("sort:radix_count_scan");
    let x = Array::<u32>::from_slice_on(&forced, &keys, keys.len()).unwrap();
    let labels = logged(&forced, DecisionSite::SortPlan, || assert_eq!(x.sort().to_vec().unwrap(), want));
    assert_eq!(labels, ["radix"]);

    // Scan: the two-pass chunks where the look-back fits.
    let forced = context("scan:chunk_three_pass");
    let long: Vec<u32> = (0..5000).collect();
    let x = Array::<u32>::from_slice_on(&forced, &long, long.len()).unwrap();
    let labels = logged(&forced, DecisionSite::ScanPlan, || {
        let got = x.cumsum(0).to_vec().unwrap();
        assert_eq!(got[4999], (0..5000).sum::<u32>());
    });
    assert_eq!(labels[0], "chunk_three_pass");

    // A strategy that does not fit (the warp pass of strided lines) is not taken: the default runs.
    let forced = context("scan:warp");
    let x = Array::<u32>::from_slice_on(&forced, &vec![1; 600 * 7], [7, 600]).unwrap();
    let labels = logged(&forced, DecisionSite::ScanPlan, || assert_eq!(x.cumsum(1).to_vec().unwrap()[7 * 599], 600));
    assert_eq!(labels[0], "fold_walk");

    // Scatter: the global bins where the private ones fit.
    let forced = context("scatter:atomic_global");
    let counts = Array::<u32>::zeros_on(&forced, 16).unwrap();
    let mut counts = counts;
    let indices = Array::<u32>::from_slice_on(&forced, &(0..4096).map(|i| i % 16).collect::<Vec<_>>(), 4096).unwrap();
    let labels = logged(&forced, DecisionSite::ScatterPlan, || {
        counts.put_add(&indices, 1u32).unwrap();
        assert_eq!(counts.to_vec().unwrap(), vec![256; 16]);
    });
    assert_eq!(labels, ["atomic_global"]);
    let default = context("");
    let mut counts = Array::<u32>::zeros_on(&default, 16).unwrap();
    let indices = Array::<u32>::from_slice_on(&default, &(0..4096).map(|i| i % 16).collect::<Vec<_>>(), 4096).unwrap();
    let labels = logged(&default, DecisionSite::ScatterPlan, || counts.put_add(&indices, 1u32).unwrap());
    assert_eq!(labels, ["atomic_private"]);

    // The light ports have no strategies to force.
    for family in ["search", "compact", "merge"] {
        let text: &'static str = Box::leak(format!("{family}:anything").into_boxed_str());
        let error = Context::with_options(ContextOptions::default().backend(BackendKind::Cpu).calibrate(false).strategy(text)).map(drop).expect_err("the family is not ported");
        assert!(matches!(&error, crate::error::LumaError::InvalidOption(message) if message.contains("family not ported")), "{error}");
    }
}

/// A scan of the totals of a scan is a scan again: the log has the outer choice, then the nested ones, in that order.
#[test]
fn nested_scans_select_again() {
    use crate::array::Array;
    use crate::profile::DecisionSite;

    let ctx = cpu_with(|t| {
        t.lookback_scan = false;
        t.subgroup32 = false;
    });
    let x = Array::<u32>::full_on(&ctx, 10_000_000usize, 1).unwrap();
    ctx.record_decisions(true);
    let sums = x.cumsum(0);
    assert_eq!(sums.slice(0, 9_999_999..10_000_000).to_vec().unwrap(), vec![10_000_000]);
    ctx.sync().unwrap();
    let labels: Vec<String> = ctx.take_decisions().iter().filter(|d| d.site == DecisionSite::ScanPlan).map(|d| d.chosen.to_string()).collect();
    ctx.record_decisions(false);
    // 10M cells are 9766 chunks of 1024; the totals are 9766 cells, 10 chunks; those totals are 10 cells, one row.
    assert_eq!(labels, ["chunk_three_pass", "chunk_three_pass", "rows"]);
}

/// A host path that declines is recorded as declined next to the radix choice that ran.
#[test]
fn a_declined_candidate_is_listed() {
    use crate::profile::{DecisionSite, Verdict};

    let ctx = cpu_with(|_| {});
    let boundary = sort_boundary(SortStage::Radix, [1000, 1, 1, 0], SortPayload::None, false);
    let (id, params) = candidates(&ctx, &boundary).next().unwrap();
    ctx.record_decisions(true);
    crate::jit::structured::strategy::note(&ctx, &boundary, id, &params, &[StrategyId::SORT_HOST]);
    let decisions = ctx.take_decisions();
    ctx.record_decisions(false);
    assert_eq!(decisions.len(), 1);
    assert_eq!(decisions[0].site, DecisionSite::SortPlan);
    assert!(decisions[0].candidates.iter().any(|(label, verdict)| label == "host" && *verdict == Verdict::Rejected("declined")), "{:?}", decisions[0]);
}
