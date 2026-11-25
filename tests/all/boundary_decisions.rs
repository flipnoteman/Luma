//! Sort, scan, scatter, search, compact and merge record a Decision at every selection site (`Context::record_decisions`): the
//! light ports (search, compact, merge: tag, site and label only) keep the labels below. Exact labels are asserted on the CPU
//! backend, which has no host paths and no look-back; elsewhere the site must be present with a label of the family.

use luma::*;

fn logged(ctx: &Context, f: impl FnOnce()) -> Vec<Decision> {
    ctx.record_decisions(true);
    f();
    ctx.sync().unwrap();
    let decisions = ctx.take_decisions();
    ctx.record_decisions(false);
    decisions
}

fn labels(decisions: &[Decision], site: DecisionSite) -> Vec<String> {
    decisions.iter().filter(|d| d.site == site).map(|d| d.chosen.to_string()).collect()
}

fn keys(n: usize) -> Vec<u32> {
    (0..n as u32).map(|i| i.wrapping_mul(2_654_435_761) >> 12).collect()
}

#[test]
fn every_boundary_family_logs_its_choice() {
    let ctx = Context::new().unwrap();
    let cpu = ctx.backend() == BackendKind::Cpu;
    // Above the host paths' limits, so the kernels run on every device.
    let n = 1 << 17;
    let x = Array::<u32>::from_slice_on(&ctx, &keys(n), n).unwrap();
    let mask = x.gt(1000u32);

    let sorted = logged(&ctx, || {
        x.sort().eval().unwrap();
    });
    let sort = labels(&sorted, DecisionSite::SortPlan);
    assert!(!sort.is_empty() && sort.iter().all(|label| ["sweep", "radix ballot", "radix"].contains(&label.as_str())), "{sort:?}");
    assert!(sorted.iter().filter(|d| d.site == DecisionSite::SortPlan).all(|d| d.op == "sort" && d.location.file().ends_with("sort.rs")), "{sorted:?}");
    if cpu {
        assert_eq!(sort, ["radix"]);
        // The scan of the counts of each pass is a scan decision (rows: 256 counts per tile column).
        assert!(!labels(&sorted, DecisionSite::ScanPlan).is_empty());
    }

    let filtered = logged(&ctx, || {
        let _ = x.filter(&mask).unwrap();
        let _ = x.filter_padded(&mask, 0);
        let _ = mask.nonzero().unwrap();
    });
    let compact = labels(&filtered, DecisionSite::CompactPlan);
    assert_eq!(compact, ["compact::count_scatter", "compact::count_scatter_padded", "compact::count_scatter"], "{filtered:?}");

    let table = Array::<u32>::from_slice_on(&ctx, &(0..4096).map(|i| i * 4).collect::<Vec<_>>(), 4096).unwrap();
    let searched = logged(&ctx, || {
        table.searchsorted(&x, Side::Left).eval().unwrap();
    });
    assert_eq!(labels(&searched, DecisionSite::SearchPlan), ["search::unrolled"]);

    let a = Array::<u32>::from_slice_on(&ctx, &(0..n as u32).map(|i| i * 2).collect::<Vec<_>>(), n).unwrap();
    let b = Array::<u32>::from_slice_on(&ctx, &(0..n as u32).map(|i| i * 3).collect::<Vec<_>>(), n).unwrap();
    let merged = logged(&ctx, || {
        let _ = a.set_union(&b, true).unwrap().to_vec().unwrap();
    });
    let merge = labels(&merged, DecisionSite::MergePlan);
    assert_eq!(merge.len(), 1, "{merged:?}");
    assert!(["merge::wide", "merge::three_kernels"].contains(&merge[0].as_str()), "{merge:?}");
    if cpu {
        assert_eq!(merge, ["merge::three_kernels"]);
    }

    let factorized = logged(&ctx, || {
        let _ = Array::<u32>::from_slice_on(&ctx, &keys(5000), 5000).unwrap().factorize();
    });
    let factor = labels(&factorized, DecisionSite::MergePlan);
    assert_eq!(factor.len(), 1, "{factorized:?}");
    assert!(["merge::factorize_host", "merge::factorize_sort"].contains(&factor[0].as_str()), "{factor:?}");

    let counts = logged(&ctx, || {
        let mut bins = Array::<u32>::zeros_on(&ctx, 64).unwrap();
        bins.put_add(&(&x % 64u32), 1u32).unwrap();
        let _ = x.cumsum(0).eval();
    });
    assert!(!labels(&counts, DecisionSite::ScatterPlan).is_empty(), "{counts:?}");
    let scan = labels(&counts, DecisionSite::ScanPlan);
    assert!(!scan.is_empty() && scan.iter().all(|label| ["rows", "warp", "lookback", "chunk_single", "chunk_three_pass", "seq", "fold_walk"].contains(&label.as_str())), "{scan:?}");
}

/// A strategy of a family that only logs is not selectable.
#[test]
fn light_ports_refuse_strategy_names() {
    for text in ["search:unrolled", "compact:count_scatter", "merge:wide"] {
        let error = Context::with_options(ContextOptions::from_env().calibrate(false).strategy(text)).err().expect("not ported");
        assert!(matches!(&error, LumaError::InvalidOption(message) if message.contains("family not ported")), "{error}");
    }
}
