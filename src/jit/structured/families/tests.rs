//! The `fits` tables of the boundary families on both sides of every threshold, and the legacy mirror: the conditions as they were
//! written inline before the registry (`f0e5723`), checked against the registry's first candidate over a grid of random shapes
//! and tunings. Built on the CPU backend with hand-made `Tuning`s, so it runs without a GPU.

use std::panic::Location;

use luma_core::BackendKind;

use crate::context::{Context, ContextOptions};
use crate::jit::structured::boundary::{dim, ScanAttrs, ScatterAttrs, ScatterOp};
use crate::jit::structured::strategy::{candidates, Boundary, BoundaryAttrs, FamilyTag, StrategyId};
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
    Boundary::new(FamilyTag::Scan, "scan", here(), [n, stride, lines, 0], BoundaryAttrs::Scan(ScanAttrs { grouping_free, by_key }))
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
