//! The `datasci` suite: table and array-analytics workloads (sort, group-by, quantiles,
//! histograms, filters, joins, scans, rolling windows, column statistics, whole pipelines).
//!
//! Sizes: `big` is 10M rows on a GPU in a full run, 4M on the CPU backend and 1M with `--quick`;
//! `huge` (100M) only runs on a GPU in a full run. A case name carries its size only where a
//! family has several (`sort/f32_rand_1m`); the `size` column always has the exact count.

use std::fmt::Debug;

mod half;
mod stats;

use crate::harness::*;
use luma::*;

pub fn run(h: &mut Harness) {
    h.guarded("sort", sort_cases);
    h.guarded("sort", sort_extra_cases);
    h.guarded("argsort", argsort_cases);
    h.guarded("argsort", argsort_extra_cases);
    h.guarded("reduce", reduce_cases);
    h.guarded("reduce", reduce_extra_cases);
    h.guarded("reduce", stats::run);
    h.guarded("scan", scan_cases);
    h.guarded("scan", scan_extra_cases);
    h.guarded("group", group_cases);
    h.guarded("factorize", factorize_cases);
    h.guarded("quantile", quantile_cases);
    h.guarded("hist", hist_cases);
    h.guarded("topk", topk_cases);
    h.guarded("filter", filter_cases);
    h.guarded("search", search_cases);
    h.guarded("take", take_cases);
    h.guarded("table", table_cases);
    h.guarded("pivot", pivot_cases);
    h.guarded("pivot", pivot_extra_cases);
    h.guarded("rolling", rolling_cases);
    h.guarded("pipeline", pipeline_cases);
    h.guarded("small", small_cases);
    h.guarded("half", half::run);
    h.guarded("spill", spill_cases);
}

// -------------------------------------------------------------------------------------------------
// Sizes and small helpers
// -------------------------------------------------------------------------------------------------

/// The large size: 10M rows on a GPU, 4M on the CPU backend, 1M in quick runs.
fn big(h: &Harness) -> usize {
    if h.quick {
        1_000_000
    } else if h.ctx.is_cpu() {
        4_000_000
    } else {
        10_000_000
    }
}

/// 100M, only on a GPU in a full run.
fn huge(h: &Harness) -> Option<usize> {
    (!h.ctx.is_cpu() && !h.quick).then_some(100_000_000)
}

fn label(n: usize) -> String {
    if n >= 1_000_000 && n.is_multiple_of(1_000_000) {
        format!("{}m", n / 1_000_000)
    } else if n >= 1_000 && n.is_multiple_of(1_000) {
        format!("{}k", n / 1_000)
    } else {
        n.to_string()
    }
}

/// `sizes` plus `big` and (if any) `huge`, ascending and distinct.
fn ladder(h: &Harness, sizes: &[usize], with_huge: bool) -> Vec<usize> {
    let mut v = sizes.to_vec();
    v.push(big(h));
    if with_huge {
        v.extend(huge(h));
    }
    v.sort_unstable();
    v.dedup();
    v
}

fn up<C: HasCtx, T: GpuType>(c: &C, data: &[T]) -> Array<T> {
    Array::from_slice_on(c.ctx(), data, data.len()).unwrap()
}

fn exact<T: PartialEq + Debug>(name: &str, got: &[T], want: &[T]) {
    assert_eq!(got.len(), want.len(), "{name}: length");
    if let Some(i) = got.iter().zip(want).position(|(g, w)| g != w) {
        panic!("{name}: index {i}: got {:?}, want {:?}", got[i], want[i]);
    }
}

pub trait HasCtx {
    fn ctx(&self) -> &Context;
}
impl HasCtx for Harness {
    fn ctx(&self) -> &Context {
        &self.ctx
    }
}
impl HasCtx for Context {
    fn ctx(&self) -> &Context {
        self
    }
}

/// [`Harness::case`] after releasing the pooled idle buffers of the earlier cases, so that the
/// `mem MiB` column is this case's own footprint (its arrays plus what its calls allocate)
/// instead of the pool the whole run has grown.
fn case_trim(h: &mut Harness, name: &str, size: &str, work: Work, f: &mut dyn FnMut()) {
    if h.want(name) {
        h.ctx.trim().unwrap();
    }
    h.case(name, size, work, f);
}

// -------------------------------------------------------------------------------------------------
// sort / argsort
// -------------------------------------------------------------------------------------------------

/// The order-preserving `u32` image of a float (what the radix sort sorts).
fn key_f32(x: f32) -> u32 {
    let b = x.to_bits();
    b ^ (((b as i32) >> 31) as u32 | 0x8000_0000)
}

fn key_i32(x: i32) -> u32 {
    (x as u32) ^ 0x8000_0000
}

fn key_u32(x: u32) -> u32 {
    x
}

/// Order-independent fingerprint of a multiset of keys.
fn fingerprint(keys: &[u32]) -> (u64, u64) {
    let (mut s, mut m) = (0u64, 0u64);
    for &k in keys {
        s = s.wrapping_add(u64::from(k));
        let z = (u64::from(k) ^ 0x9E37_79B9_7F4A_7C15).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        m = m.wrapping_add(z ^ (z >> 29));
    }
    (s, m)
}

fn check_sorted(name: &str, got: &[u32], input: &[u32]) {
    assert_eq!(got.len(), input.len(), "{name}: length");
    if let Some(i) = got.windows(2).position(|w| w[0] > w[1]) {
        panic!("{name}: not sorted at {i}");
    }
    assert_eq!(fingerprint(got), fingerprint(input), "{name}: not a permutation of the input");
}

/// A permutation that orders the keys stably is unique, so these three checks are exact.
fn check_argsort(name: &str, idx: &[u32], keys: &[u32]) {
    let n = keys.len();
    assert_eq!(idx.len(), n, "{name}: length");
    let mut seen = vec![false; n];
    let mut prev: Option<(u32, u32)> = None;
    for (pos, &i) in idx.iter().enumerate() {
        assert!((i as usize) < n && !seen[i as usize], "{name}: position {pos}: index {i} out of range or repeated");
        seen[i as usize] = true;
        let k = keys[i as usize];
        if let Some((pk, pi)) = prev {
            assert!(pk < k || (pk == k && pi < i), "{name}: position {pos}: order violated (key {pk} idx {pi} before key {k} idx {i})");
        }
        prev = Some((k, i));
    }
}

fn sort_case<T: SortKey + Copy>(h: &mut Harness, name: &str, data: &[T], key: fn(T) -> u32) {
    let n = data.len();
    let x = up(h, data);
    let got = x.sort();
    got.eval().unwrap();
    let got_keys: Vec<u32> = got.to_vec().unwrap().into_iter().map(key).collect();
    let in_keys: Vec<u32> = data.iter().copied().map(key).collect();
    check_sorted(name, &got_keys, &in_keys);
    drop((got, got_keys, in_keys));
    case_trim(h, name, &n.to_string(), Work::Items(n as f64, "Mkey/s"), &mut || x.sort().eval().unwrap());
}

fn argsort_case<T: SortKey + Copy>(h: &mut Harness, name: &str, data: &[T], key: fn(T) -> u32) {
    let n = data.len();
    let x = up(h, data);
    let got = x.argsort();
    got.eval().unwrap();
    let in_keys: Vec<u32> = data.iter().copied().map(key).collect();
    check_argsort(name, &got.to_vec().unwrap(), &in_keys);
    drop((got, in_keys));
    case_trim(h, name, &n.to_string(), Work::Items(n as f64, "Mkey/s"), &mut || x.argsort().eval().unwrap());
}

fn rand_f32(n: usize, seed: u64) -> Vec<f32> {
    values(n, seed, -1000.0, 1000.0)
}

fn dups_f32(n: usize, seed: u64) -> Vec<f32> {
    ints(n, seed, 256).into_iter().map(|c| c as f32 * 0.5 - 40.0).collect()
}

fn sorted_f32(n: usize, seed: u64) -> Vec<f32> {
    let mut v = rand_f32(n, seed);
    v.sort_by(f32::total_cmp);
    v
}

fn sort_cases(h: &mut Harness) {
    for n in ladder(h, &[1_000, 100_000, 1_000_000], true) {
        let name = format!("sort/f32_rand_{}", label(n));
        if h.want(&name) {
            sort_case(h, &name, &rand_f32(n, 1), key_f32);
        }
    }
    let n = big(h);
    if h.want("sort/u32_rand") {
        sort_case(h, "sort/u32_rand", &ints(n, 2, u32::MAX), key_u32);
    }
    if h.want("sort/i32_rand") {
        let data: Vec<i32> = ints(n, 3, u32::MAX).into_iter().map(|v| v as i32).collect();
        sort_case(h, "sort/i32_rand", &data, key_i32);
    }
    if h.want("sort/f32_sorted") {
        sort_case(h, "sort/f32_sorted", &sorted_f32(n, 4), key_f32);
    }
    if h.want("sort/f32_dups") {
        sort_case(h, "sort/f32_dups", &dups_f32(n, 5), key_f32);
    }

    // Lines along an axis: 1024-element lines, contiguous (axis 0) and strided (axis 1).
    let total = if h.ctx.is_cpu() || h.quick { 1 << 20 } else { 1 << 22 };
    let (long, count) = (1024usize, total / 1024);
    for (axis, dims) in [(0usize, [long, count]), (1, [count, long])] {
        let name = format!("sort/axis{axis}");
        if !h.want(&name) {
            continue;
        }
        let hx = rand_f32(total, 6);
        let x = Array::from_slice_on(&h.ctx, &hx, dims).unwrap();
        let mut want = hx.clone();
        if axis == 0 {
            for line in want.chunks_mut(long) {
                line.sort_by(f32::total_cmp);
            }
        } else {
            for i in 0..count {
                let mut line: Vec<f32> = (0..long).map(|j| hx[i + count * j]).collect();
                line.sort_by(f32::total_cmp);
                for (j, v) in line.into_iter().enumerate() {
                    want[i + count * j] = v;
                }
            }
        }
        let got = x.sort_axis(axis);
        exact(&name, &got.to_vec().unwrap(), &want);
        drop(got);
        case_trim(h, &name, &format!("{}x{}", dims[0], dims[1]), Work::Items(total as f64, "Mkey/s"), &mut || x.sort_axis(axis).eval().unwrap());
    }
}

fn argsort_cases(h: &mut Harness) {
    for n in ladder(h, &[1_000, 100_000], false) {
        let name = format!("argsort/f32_rand_{}", label(n));
        if h.want(&name) {
            argsort_case(h, &name, &rand_f32(n, 11), key_f32);
        }
    }
    let n = 1_000_000;
    if h.want("argsort/f32_dups") {
        argsort_case(h, "argsort/f32_dups", &dups_f32(n, 12), key_f32);
    }
    if h.want("argsort/f32_sorted") {
        argsort_case(h, "argsort/f32_sorted", &sorted_f32(n, 13), key_f32);
    }
    if h.want("argsort/u32_rand") {
        argsort_case(h, "argsort/u32_rand", &ints(big(h), 14, u32::MAX), key_u32);
    }
}

// -------------------------------------------------------------------------------------------------
// reductions
// -------------------------------------------------------------------------------------------------

fn reduce_cases(h: &mut Harness) {
    for n in ladder(h, &[1_000, 100_000], true) {
        let name = format!("reduce/sum_f32_{}", label(n));
        if !h.want(&name) {
            continue;
        }
        let hx = values(n, 21, 0.0, 1.0);
        let x = up(h, &hx);
        let want: f64 = hx.iter().map(|&v| f64::from(v)).sum();
        check_scalar(&name, f64::from(x.sum().item().unwrap()), want, 1e-4);
        case_trim(h, &name, &n.to_string(), Work::Bytes(4.0 * n as f64), &mut || {
            x.sum().item().unwrap();
        });
    }

    let n = big(h);
    let sz = n.to_string();
    let bytes = 4.0 * n as f64;

    if h.want("reduce/sum_u32") {
        let hx = ints(n, 22, 1000);
        let x = up(h, &hx);
        let want = hx.iter().fold(0u32, |a, &v| a.wrapping_add(v));
        assert_eq!(x.sum().item().unwrap(), want, "reduce/sum_u32");
        case_trim(h, "reduce/sum_u32", &sz, Work::Bytes(bytes), &mut || {
            x.sum().item().unwrap();
        });
    }
    if h.want("reduce/max_f32") {
        let hx = values(n, 23, -5.0, 5.0);
        let x = up(h, &hx);
        let want = hx.iter().copied().fold(f32::MIN, f32::max);
        assert_eq!(x.max().item().unwrap(), want, "reduce/max_f32");
        case_trim(h, "reduce/max_f32", &sz, Work::Bytes(bytes), &mut || {
            x.max().item().unwrap();
        });
    }
    if h.want("reduce/dot_f32") {
        let (ha, hb) = (values(n, 24, 0.0, 1.0), values(n, 25, 0.0, 1.0));
        let (a, b) = (up(h, &ha), up(h, &hb));
        let want: f64 = ha.iter().zip(&hb).map(|(&p, &q)| f64::from(p) * f64::from(q)).sum();
        check_scalar("reduce/dot_f32", f64::from((&a * &b).sum().item().unwrap()), want, 1e-4);
        case_trim(h, "reduce/dot_f32", &sz, Work::Bytes(2.0 * bytes), &mut || {
            (&a * &b).sum().item().unwrap();
        });
    }
    if h.want("reduce/norm_f32") {
        let hx = values(n, 26, -1.0, 1.0);
        let x = up(h, &hx);
        let want = hx.iter().map(|&v| f64::from(v) * f64::from(v)).sum::<f64>().sqrt();
        check_scalar("reduce/norm_f32", f64::from(x.norm().item().unwrap()), want, 1e-4);
        case_trim(h, "reduce/norm_f32", &sz, Work::Bytes(bytes), &mut || {
            x.norm().item().unwrap();
        });
    }
    if h.want("reduce/stats4_f32") {
        let hx = values(n, 27, -1.0, 1.0);
        let x = up(h, &hx);
        let ops = [Reduction::Sum, Reduction::SumOfSquares, Reduction::Min, Reduction::Max];
        let got = x.reduce(ops).to_array::<4>().unwrap();
        let sum: f64 = hx.iter().map(|&v| f64::from(v)).sum();
        let ssq: f64 = hx.iter().map(|&v| f64::from(v) * f64::from(v)).sum();
        check_scalar("reduce/stats4 sum", f64::from(got[0]), sum, 1e-3);
        check_scalar("reduce/stats4 ssq", f64::from(got[1]), ssq, 1e-4);
        assert_eq!(got[2], hx.iter().copied().fold(f32::MAX, f32::min));
        assert_eq!(got[3], hx.iter().copied().fold(f32::MIN, f32::max));
        case_trim(h, "reduce/stats4_f32", &sz, Work::Bytes(bytes), &mut || {
            x.reduce(ops).to_array::<4>().unwrap();
        });
    }
    if h.want("reduce/count_gt") {
        let hx = values(n, 28, 0.0, 1.0);
        let x = up(h, &hx);
        let want = hx.iter().filter(|&&v| v > 0.3).count() as u32;
        assert_eq!(x.gt(0.3).count().item().unwrap(), want, "reduce/count_gt");
        case_trim(h, "reduce/count_gt", &sz, Work::Bytes(bytes), &mut || {
            x.gt(0.3).count().item().unwrap();
        });
    }
    if h.want("reduce/sum_where") {
        let hx = values(n, 29, 0.0, 1.0);
        let x = up(h, &hx);
        let want: f64 = hx.iter().filter(|&&v| v > 0.3).map(|&v| f64::from(v)).sum();
        check_scalar("reduce/sum_where", f64::from(x.sum_where(&x.gt(0.3)).item().unwrap()), want, 1e-4);
        case_trim(h, "reduce/sum_where", &sz, Work::Bytes(bytes), &mut || {
            x.sum_where(&x.gt(0.3)).item().unwrap();
        });
    }
    // Eight independent small reductions in one batch: launch and batching overhead.
    if h.want("reduce/batch8") {
        let m = 100_000;
        let data: Vec<Vec<f32>> = (0..8).map(|i| values(m, 30 + i, 0.0, 1.0)).collect();
        let arrays: Vec<Array<f32>> = data.iter().map(|d| up(h, d)).collect();
        let run = |arrays: &[Array<f32>]| {
            let sums: Vec<Array<f32>> = arrays.iter().map(|a| a.sum()).collect();
            let refs: Vec<&Array<f32>> = sums.iter().collect();
            eval_all(&refs).unwrap();
            sums
        };
        for (s, d) in run(&arrays).iter().zip(&data) {
            check_scalar("reduce/batch8", f64::from(s.item().unwrap()), d.iter().map(|&v| f64::from(v)).sum(), 1e-4);
        }
        case_trim(h, "reduce/batch8", &format!("8x{m}"), Work::Bytes(8.0 * 4.0 * m as f64), &mut || {
            run(&arrays);
        });
    }
}

// -------------------------------------------------------------------------------------------------
// scans and differences
// -------------------------------------------------------------------------------------------------

fn scan_cases(h: &mut Harness) {
    let n = big(h);
    for m in ladder(h, &[1_000, 1_000_000], false) {
        let name = format!("scan/cumsum_f32_{}", label(m));
        if !h.want(&name) {
            continue;
        }
        let hx = values(m, 41, 0.0, 1.0);
        let x = up(h, &hx);
        let mut acc = 0.0f64;
        let want: Vec<f32> = hx
            .iter()
            .map(|&v| {
                acc += f64::from(v);
                acc as f32
            })
            .collect();
        check(&name, &x.cumsum(0).to_vec().unwrap(), &want, 1e-4, 1e-3);
        case_trim(h, &name, &m.to_string(), Work::Bytes(8.0 * m as f64), &mut || x.cumsum(0).eval().unwrap());
    }
    if h.want("scan/cumsum_u32") {
        let hx = ints(n, 42, 16);
        let x = up(h, &hx);
        let mut acc = 0u32;
        let want: Vec<u32> = hx
            .iter()
            .map(|&v| {
                acc = acc.wrapping_add(v);
                acc
            })
            .collect();
        exact("scan/cumsum_u32", &x.cumsum(0).to_vec().unwrap(), &want);
        case_trim(h, "scan/cumsum_u32", &n.to_string(), Work::Bytes(8.0 * n as f64), &mut || x.cumsum(0).eval().unwrap());
    }
    // Lines of 1024 along each axis of a table.
    let total = if h.ctx.is_cpu() || h.quick { 1 << 20 } else { 1 << 22 };
    let count = total / 1024;
    for (axis, dims) in [(0usize, [1024, count]), (1, [count, 1024])] {
        let name = format!("scan/cumsum_axis{axis}");
        if !h.want(&name) {
            continue;
        }
        let hx = values(total, 44, 0.0, 1.0);
        let x = Array::from_slice_on(&h.ctx, &hx, dims).unwrap();
        let mut want = vec![0.0f32; total];
        if axis == 0 {
            for (src, dst) in hx.chunks(1024).zip(want.chunks_mut(1024)) {
                let mut acc = 0.0f64;
                for (s, d) in src.iter().zip(dst) {
                    acc += f64::from(*s);
                    *d = acc as f32;
                }
            }
        } else {
            for i in 0..count {
                let mut acc = 0.0f64;
                for j in 0..1024 {
                    acc += f64::from(hx[i + count * j]);
                    want[i + count * j] = acc as f32;
                }
            }
        }
        check(&name, &x.cumsum(axis).to_vec().unwrap(), &want, 1e-4, 1e-3);
        case_trim(h, &name, &format!("{}x{}", dims[0], dims[1]), Work::Bytes(8.0 * total as f64), &mut || x.cumsum(axis).eval().unwrap());
    }
    if h.want("scan/diff_f32") {
        let hx = values(n, 45, 0.0, 1.0);
        let x = up(h, &hx);
        let diff = |x: &Array<f32>| x.slice(0, 1..) - x.slice(0, ..n - 1);
        let want: Vec<f32> = hx.windows(2).map(|w| w[1] - w[0]).collect();
        check("scan/diff_f32", &diff(&x).to_vec().unwrap(), &want, 1e-6, 1e-7);
        case_trim(h, "scan/diff_f32", &n.to_string(), Work::Bytes(8.0 * n as f64), &mut || diff(&x).eval().unwrap());
    }
    if h.want("scan/cumprod_f32") {
        let m = 1_000_000;
        let hx = values(m, 43, 0.9999, 1.0001);
        let x = up(h, &hx);
        let mut acc = 1.0f64;
        let want: Vec<f32> = hx
            .iter()
            .map(|&v| {
                acc *= f64::from(v);
                acc as f32
            })
            .collect();
        // FINDING (luma, all backends): `scan` with `*` drifts linearly with the position, about
        // 6e-9 relative per element (6e-3 at 1e6 here), against 9e-6 for a sequential f32
        // product of the same data; the identical numbers on CPU, CUDA and wgpu point at the
        // algorithm, not at rounding. The check therefore only guards against gross errors and
        // prints the measured drift when it exceeds what an f32 product should have (1e-4).
        case_trim(h, "scan/cumprod_f32", &m.to_string(), Work::Bytes(8.0 * m as f64), &mut || x.scan(0, |a, b| a * b).eval().unwrap());
        let got = x.scan(0, |a, b| a * b).to_vec().unwrap();
        let worst = got.iter().zip(&want).map(|(g, w)| ((g - w) / w).abs()).fold(0.0f32, f32::max);
        if worst > 1e-4 {
            println!("FINDING: scan/cumprod_f32: worst relative error {worst:.2e} (sequential f32: < 1e-5)");
        }
        assert!(worst < 5e-2, "scan/cumprod_f32: worst relative error {worst:.2e}");
    }
}

// -------------------------------------------------------------------------------------------------
// group-by
// -------------------------------------------------------------------------------------------------

/// Host f64 per-group sum and sum of magnitudes.
fn host_group_sum(codes: &[u32], vals: &[f32], groups: usize) -> (Vec<f64>, Vec<f64>) {
    let (mut sum, mut mag) = (vec![0.0f64; groups], vec![0.0f64; groups]);
    for (&c, &v) in codes.iter().zip(vals) {
        if (c as usize) < groups {
            sum[c as usize] += f64::from(v);
            mag[c as usize] += f64::from(v).abs();
        }
    }
    (sum, mag)
}

/// `f32` group sums within `rel * sum|v|` of the f64 sums.
fn check_sums(name: &str, got: &[f32], sum: &[f64], mag: &[f64], rel: f64) {
    assert_eq!(got.len(), sum.len(), "{name}: length");
    for (g, ((&got, &want), &mag)) in got.iter().zip(sum).zip(mag).enumerate() {
        let tol = (rel * mag).max(1e-5);
        assert!((f64::from(got) - want).abs() <= tol, "{name}: group {g}: got {got}, want {want} (tol {tol})");
    }
}

fn group_cases(h: &mut Harness) {
    let n = big(h);
    let sz = n.to_string();
    let bytes = 8.0 * n as f64;
    let hv = values(n, 51, 0.0, 1.0);
    let vals = up(h, &hv);

    // Sum over few, medium and many groups; and all rows in one group (worst atomic contention).
    for (tag, groups, kind) in [("g16", 16usize, 0), ("g1k", 1000, 0), ("g1m", 1_000_000, 0), ("hot16", 16, 1), ("skew1k", 1000, 2)] {
        let name = format!("group/sum_{tag}");
        if !h.want(&name) {
            continue;
        }
        let hc: Vec<u32> = match kind {
            0 => ints(n, 52, groups as u32),
            1 => vec![0; n],
            _ => ints(n, 52, groups as u32).into_iter().map(|u| u * u / groups as u32).collect(),
        };
        let codes = up(h, &hc);
        let (sum, mag) = host_group_sum(&hc, &hv, groups);
        check_sums(&name, &vals.group_sum(&codes, groups).to_vec().unwrap(), &sum, &mag, 2e-4);
        case_trim(h, &name, &sz, Work::Bytes(bytes), &mut || vals.group_sum(&codes, groups).eval().unwrap());
    }

    // Codes in sorted order (a table sorted by its key): `group_sum` by code (one atomic add per row) against
    // `sum_segments` over the offsets of the groups.
    for (tag, groups) in [("g16", 16usize), ("g1k", 1000), ("g1m", 1_000_000)] {
        let (atomic, segments) = (format!("group/sum_sorted_{tag}"), format!("group/sum_sorted_seg_{tag}"));
        if !h.want(&atomic) && !h.want(&segments) {
            continue;
        }
        let mut hc = ints(n, 52, groups as u32);
        hc.sort_unstable();
        let mut offsets = vec![0u32; groups + 1];
        for &c in &hc {
            offsets[c as usize + 1] += 1;
        }
        for g in 0..groups {
            offsets[g + 1] += offsets[g];
        }
        let (sum, mag) = host_group_sum(&hc, &hv, groups);
        let (codes, offs) = (up(h, &hc), up(h, &offsets));
        if h.want(&atomic) {
            check_sums(&atomic, &vals.group_sum(&codes, groups).to_vec().unwrap(), &sum, &mag, 2e-4);
            case_trim(h, &atomic, &sz, Work::Bytes(bytes), &mut || vals.group_sum(&codes, groups).eval().unwrap());
        }
        if h.want(&segments) {
            check_sums(&segments, &vals.sum_segments(&offs).to_vec().unwrap(), &sum, &mag, 2e-4);
            case_trim(h, &segments, &sz, Work::Bytes(bytes), &mut || vals.sum_segments(&offs).eval().unwrap());
        }
    }

    let g = 1000usize;
    let hc = ints(n, 53, g as u32);
    let codes = up(h, &hc);
    if h.want("group/mean_g1k") {
        let (sum, _) = host_group_sum(&hc, &hv, g);
        let mut count = vec![0u32; g];
        hc.iter().for_each(|&c| count[c as usize] += 1);
        let got = vals.group_mean(&codes, g).to_vec().unwrap();
        let want: Vec<f32> = sum.iter().zip(&count).map(|(&s, &c)| (s / f64::from(c)) as f32).collect();
        check("group/mean_g1k", &got, &want, 1e-3, 1e-4);
        case_trim(h, "group/mean_g1k", &sz, Work::Bytes(2.0 * bytes), &mut || vals.group_mean(&codes, g).eval().unwrap());
    }
    if h.want("group/min_g1k") || h.want("group/max_g1k") {
        let mut lo = vec![f32::INFINITY; g];
        let mut hi = vec![f32::NEG_INFINITY; g];
        for (&c, &v) in hc.iter().zip(&hv) {
            lo[c as usize] = lo[c as usize].min(v);
            hi[c as usize] = hi[c as usize].max(v);
        }
        if h.want("group/min_g1k") {
            exact("group/min_g1k", &vals.group_min(&codes, g).to_vec().unwrap(), &lo);
            case_trim(h, "group/min_g1k", &sz, Work::Bytes(bytes), &mut || vals.group_min(&codes, g).eval().unwrap());
        }
        if h.want("group/max_g1k") {
            exact("group/max_g1k", &vals.group_max(&codes, g).to_vec().unwrap(), &hi);
            case_trim(h, "group/max_g1k", &sz, Work::Bytes(bytes), &mut || vals.group_max(&codes, g).eval().unwrap());
        }
    }
    if h.want("group/count_g1k") {
        let mut want = vec![0u32; g];
        hc.iter().for_each(|&c| want[c as usize] += 1);
        exact("group/count_g1k", &vals.group_count(&codes, g).to_vec().unwrap(), &want);
        case_trim(h, "group/count_g1k", &sz, Work::Bytes(4.0 * n as f64), &mut || vals.group_count(&codes, g).eval().unwrap());
    }
    // The usual aggregate table: sum, min, max, count in one go.
    if h.want("group/agg4_g1k") {
        let agg = || {
            let sum = vals.group_sum(&codes, g);
            let lo = vals.group_min(&codes, g);
            let hi = vals.group_max(&codes, g);
            let count = vals.group_count(&codes, g);
            eval_all(&[&sum, &lo, &hi]).unwrap();
            count.eval().unwrap();
            (sum, lo, hi, count)
        };
        let (sum, mag) = host_group_sum(&hc, &hv, g);
        let (s, ..) = agg();
        check_sums("group/agg4_g1k", &s.to_vec().unwrap(), &sum, &mag, 2e-4);
        case_trim(h, "group/agg4_g1k", &sz, Work::Bytes(4.0 * bytes), &mut || {
            agg();
        });
    }
    if h.want("group/sum_i32_g1k") {
        let hi_vals: Vec<i32> = ints(n, 54, 200).into_iter().map(|v| v as i32 - 100).collect();
        let iv = up(h, &hi_vals);
        let mut want = vec![0i32; g];
        for (&c, &v) in hc.iter().zip(&hi_vals) {
            want[c as usize] = want[c as usize].wrapping_add(v);
        }
        exact("group/sum_i32_g1k", &iv.group_sum(&codes, g).to_vec().unwrap(), &want);
        case_trim(h, "group/sum_i32_g1k", &sz, Work::Bytes(bytes), &mut || iv.group_sum(&codes, g).eval().unwrap());
    }
    // Two sparse labels per row: factorize both, combine into one dense code, aggregate.
    if h.want("group/multikey_sum_mean") {
        let (na, nb) = (100usize, 50usize);
        let ka: Vec<u32> = ints(n, 55, na as u32).into_iter().map(|c| 1000 + 7 * c).collect();
        let kb: Vec<u32> = ints(n, 56, nb as u32).into_iter().map(|c| 5 + 3 * c).collect();
        let (a, b) = (up(h, &ka), up(h, &kb));
        let agg = || {
            let (ua, ca) = a.factorize();
            let (ub, cb) = b.factorize();
            let groups = ua.len() * ub.len();
            let code = &ca * (ub.len() as u32) + &cb;
            let sum = vals.group_sum(&code, groups);
            let mean = vals.group_mean(&code, groups);
            eval_all(&[&sum, &mean]).unwrap();
            (sum, mean)
        };
        let ia: Vec<u32> = ka.iter().map(|&k| (k - 1000) / 7).collect();
        let ib: Vec<u32> = kb.iter().map(|&k| (k - 5) / 3).collect();
        let hcode: Vec<u32> = ia.iter().zip(&ib).map(|(&x, &y)| x * nb as u32 + y).collect();
        let (sum, mag) = host_group_sum(&hcode, &hv, na * nb);
        let (s, _) = agg();
        check_sums("group/multikey_sum_mean", &s.to_vec().unwrap(), &sum, &mag, 2e-4);
        case_trim(h, "group/multikey_sum_mean", &sz, Work::Bytes(4.0 * bytes), &mut || {
            agg();
        });
    }
}

// -------------------------------------------------------------------------------------------------
// factorize / unique
// -------------------------------------------------------------------------------------------------

fn factorize_cases(h: &mut Harness) {
    let n = big(h);
    for (name, rows, distinct) in [("factorize/u32_1m_d1k", 1_000_000usize, 1000u32), ("factorize/u32_big_d100k", n, 100_000)] {
        if !h.want(name) || (name.contains("big") && n == 1_000_000) {
            continue;
        }
        let hk: Vec<u32> = ints(rows, 61, distinct).into_iter().map(|c| 17 + 11 * c).collect();
        let x = up(h, &hk);
        let mut want = hk.clone();
        want.sort_unstable();
        want.dedup();
        let (u, c) = x.factorize();
        exact(name, &u.to_vec().unwrap(), &want);
        let codes = c.to_vec().unwrap();
        let uv = u.to_vec().unwrap();
        assert!(codes.iter().zip(&hk).all(|(&c, &k)| uv[c as usize] == k), "{name}: codes");
        drop((u, c));
        case_trim(h, name, &rows.to_string(), Work::Items(rows as f64, "Mrow/s"), &mut || {
            let (u, c) = x.factorize();
            eval_all(&[&u, &c]).unwrap();
        });
    }
    if h.want("factorize/unique_f32_1m") {
        let m = 1_000_000;
        let hx: Vec<f32> = ints(m, 62, 5000).into_iter().map(|c| c as f32 * 0.25).collect();
        let x = up(h, &hx);
        let mut want = hx.clone();
        want.sort_by(f32::total_cmp);
        want.dedup();
        exact("factorize/unique_f32_1m", &x.unique().to_vec().unwrap(), &want);
        case_trim(h, "factorize/unique_f32_1m", &m.to_string(), Work::Items(m as f64, "Mrow/s"), &mut || x.unique().eval().unwrap());
    }
    // Small, latency-bound: the readback of the distinct count dominates.
    if h.want("factorize/u32_10k_d100") {
        let m = 10_000;
        let hk: Vec<u32> = ints(m, 63, 100).into_iter().map(|c| 3 * c + 1).collect();
        let x = up(h, &hk);
        let mut want = hk.clone();
        want.sort_unstable();
        want.dedup();
        exact("factorize/u32_10k_d100", &x.unique().to_vec().unwrap(), &want);
        case_trim(h, "factorize/u32_10k_d100", &m.to_string(), Work::Items(m as f64, "Mrow/s"), &mut || x.unique().eval().unwrap());
    }
}

// -------------------------------------------------------------------------------------------------
// quantiles, histograms, top-k
// -------------------------------------------------------------------------------------------------

/// NumPy's `linear` quantile of ascending `sorted`, with luma's f32 interpolation.
fn host_quantile(sorted: &[f32], q: f32) -> f32 {
    let h = q * (sorted.len() - 1) as f32;
    let (lo, hi) = (h.floor() as usize, h.ceil() as usize);
    sorted[lo] + (h - lo as f32) * (sorted[hi] - sorted[lo])
}

fn quantile_cases(h: &mut Harness) {
    for n in ladder(h, &[1_000_000], false) {
        let name = format!("quantile/median_{}", label(n));
        if !h.want(&name) {
            continue;
        }
        let hx = values(n, 71, -3.0, 3.0);
        let x = up(h, &hx);
        let mut s = hx.clone();
        s.sort_by(f32::total_cmp);
        check_scalar(&name, f64::from(x.median().item().unwrap()), f64::from(host_quantile(&s, 0.5)), 1e-6);
        case_trim(h, &name, &n.to_string(), Work::Items(n as f64, "Mrow/s"), &mut || {
            x.median().item().unwrap();
        });
    }
    // The usual percentile table: five calls, five sorts.
    if h.want("quantile/five_1m") {
        let n = 1_000_000;
        let qs = [0.05f32, 0.25, 0.5, 0.75, 0.95];
        let hx = values(n, 72, -3.0, 3.0);
        let x = up(h, &hx);
        let mut s = hx.clone();
        s.sort_by(f32::total_cmp);
        let run = || {
            let r: Vec<Array<f32>> = qs.iter().map(|&q| x.quantile(q)).collect();
            let refs: Vec<&Array<f32>> = r.iter().collect();
            eval_all(&refs).unwrap();
            r
        };
        for (r, &q) in run().iter().zip(&qs) {
            check_scalar("quantile/five_1m", f64::from(r.item().unwrap()), f64::from(host_quantile(&s, q)), 1e-5);
        }
        case_trim(h, "quantile/five_1m", &n.to_string(), Work::Items(5.0 * n as f64, "Mrow/s"), &mut || {
            run();
        });
    }
    // The same table from one sort.
    if h.want("quantile/five_multi_1m") {
        let n = 1_000_000;
        let qs = [0.05f32, 0.25, 0.5, 0.75, 0.95];
        let hx = values(n, 72, -3.0, 3.0);
        let x = up(h, &hx);
        let mut s = hx.clone();
        s.sort_by(f32::total_cmp);
        let want: Vec<f32> = qs.iter().map(|&q| host_quantile(&s, q)).collect();
        check("quantile/five_multi_1m", &x.quantiles(&qs).to_vec().unwrap(), &want, 1e-5, 1e-6);
        case_trim(h, "quantile/five_multi_1m", &n.to_string(), Work::Items(5.0 * n as f64, "Mrow/s"), &mut || x.quantiles(&qs).eval().unwrap());
    }
    // Median of every column of a table: 1024 samples per column.
    if h.want("quantile/median_axis0") {
        let total = if h.ctx.is_cpu() || h.quick { 1 << 20 } else { 1 << 22 };
        let cols = total / 1024;
        let hx = values(total, 73, -3.0, 3.0);
        let x = Array::from_slice_on(&h.ctx, &hx, [1024, cols]).unwrap();
        let want: Vec<f32> = hx
            .chunks(1024)
            .map(|c| {
                let mut s = c.to_vec();
                s.sort_by(f32::total_cmp);
                host_quantile(&s, 0.5)
            })
            .collect();
        check("quantile/median_axis0", &x.median_axis(0).to_vec().unwrap(), &want, 1e-5, 1e-6);
        case_trim(h, "quantile/median_axis0", &format!("1024x{cols}"), Work::Items(total as f64, "Mrow/s"), &mut || x.median_axis(0).eval().unwrap());
    }
    // Describe-style statistics of every column of a 64-column table: three quantile calls.
    if h.want("quantile/cols3_x64") {
        let rows = table_rows(h);
        let qs = [0.25f32, 0.5, 0.75];
        let hx = values(rows * 64, 74, -3.0, 3.0);
        let x = Array::from_slice_on(&h.ctx, &hx, [rows, 64]).unwrap();
        let sorted = host_sort_axis(&hx, [rows, 64], 0);
        let run = || {
            let r: Vec<Array<f32>> = qs.iter().map(|&q| x.quantile_axis(0, q)).collect();
            let refs: Vec<&Array<f32>> = r.iter().collect();
            eval_all(&refs).unwrap();
            r
        };
        for (r, &q) in run().iter().zip(&qs) {
            let want: Vec<f32> = sorted.chunks(rows).map(|c| host_quantile(c, q)).collect();
            check("quantile/cols3_x64", &r.to_vec().unwrap(), &want, 1e-5, 1e-6);
        }
        case_trim(h, "quantile/cols3_x64", &format!("{rows}x64"), Work::Items((3 * rows * 64) as f64, "Mrow/s"), &mut || {
            run();
        });
    }
    if h.want("quantile/cols3_multi_x64") {
        let rows = table_rows(h);
        let qs = [0.25f32, 0.5, 0.75];
        let hx = values(rows * 64, 74, -3.0, 3.0);
        let x = Array::from_slice_on(&h.ctx, &hx, [rows, 64]).unwrap();
        let sorted = host_sort_axis(&hx, [rows, 64], 0);
        let mut want = Vec::new();
        for c in sorted.chunks(rows) {
            want.extend(qs.iter().map(|&q| host_quantile(c, q)));
        }
        check("quantile/cols3_multi_x64", &x.quantiles_axis(0, &qs).to_vec().unwrap(), &want, 1e-5, 1e-6);
        case_trim(h, "quantile/cols3_multi_x64", &format!("{rows}x64"), Work::Items((3 * rows * 64) as f64, "Mrow/s"), &mut || x.quantiles_axis(0, &qs).eval().unwrap());
    }
    // Seven quantiles of one long vector.
    if h.want("quantile/seven_16m") {
        let n = if h.ctx.is_cpu() || h.quick { 1 << 20 } else { 1 << 24 };
        let qs = [0.01f32, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99];
        let hx = values(n, 75, -3.0, 3.0);
        let x = up(h, &hx);
        let mut s = hx.clone();
        s.sort_by(f32::total_cmp);
        let run = || {
            let r: Vec<Array<f32>> = qs.iter().map(|&q| x.quantile(q)).collect();
            let refs: Vec<&Array<f32>> = r.iter().collect();
            eval_all(&refs).unwrap();
            r
        };
        for (r, &q) in run().iter().zip(&qs) {
            check_scalar("quantile/seven_16m", f64::from(r.item().unwrap()), f64::from(host_quantile(&s, q)), 1e-5);
        }
        case_trim(h, "quantile/seven_16m", &n.to_string(), Work::Items(7.0 * n as f64, "Mrow/s"), &mut || {
            run();
        });
    }
    if h.want("quantile/seven_multi_16m") {
        let n = if h.ctx.is_cpu() || h.quick { 1 << 20 } else { 1 << 24 };
        let qs = [0.01f32, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99];
        let hx = values(n, 75, -3.0, 3.0);
        let x = up(h, &hx);
        let mut s = hx.clone();
        s.sort_by(f32::total_cmp);
        let want: Vec<f32> = qs.iter().map(|&q| host_quantile(&s, q)).collect();
        check("quantile/seven_multi_16m", &x.quantiles(&qs).to_vec().unwrap(), &want, 1e-5, 1e-6);
        case_trim(h, "quantile/seven_multi_16m", &n.to_string(), Work::Items(7.0 * n as f64, "Mrow/s"), &mut || x.quantiles(&qs).eval().unwrap());
    }
}

fn host_hist(hx: &[f32], bins: usize, lo: f32, hi: f32) -> Vec<u32> {
    let scale = bins as f32 / (hi - lo);
    let mut want = vec![0u32; bins];
    for &v in hx {
        let t = (v - lo) * scale;
        let u = if t.is_nan() { 0 } else { t.clamp(0.0, 4294967040.0) as u32 };
        want[u.min(bins as u32 - 1) as usize] += 1;
    }
    want
}

fn hist_cases(h: &mut Harness) {
    let n = big(h);
    let sz = n.to_string();
    let mut hx = values(n, 81, -0.25, 1.25);
    for i in (0..n).step_by(100_003) {
        hx[i] = f32::NAN;
    }
    let x = up(h, &hx);
    for bins in [64usize, 256, 4096] {
        let name = format!("hist/f32_{bins}bins");
        if !h.want(&name) {
            continue;
        }
        exact(&name, &x.histogram(bins, 0.0, 1.0).to_vec().unwrap(), &host_hist(&hx, bins, 0.0, 1.0));
        case_trim(h, &name, &sz, Work::Bytes(4.0 * n as f64), &mut || x.histogram(bins, 0.0, 1.0).eval().unwrap());
    }
    // A peaked distribution: most values land in a few bins (atomic contention).
    if h.want("hist/f32_peaked_256bins") {
        let a = values(n, 82, 0.0, 1.0);
        let b = values(n, 83, 0.0, 1.0);
        let c = values(n, 84, 0.0, 1.0);
        let hp: Vec<f32> = (0..n).map(|i| (a[i] * b[i] * c[i]).powi(3)).collect();
        let xp = up(h, &hp);
        exact("hist/f32_peaked_256bins", &xp.histogram(256, 0.0, 1.0).to_vec().unwrap(), &host_hist(&hp, 256, 0.0, 1.0));
        case_trim(h, "hist/f32_peaked_256bins", &sz, Work::Bytes(4.0 * n as f64), &mut || xp.histogram(256, 0.0, 1.0).eval().unwrap());
    }
    for (name, bins) in [("hist/bincount_64", 64usize), ("hist/bincount_1m", 1_000_000)] {
        if !h.want(name) {
            continue;
        }
        let hc = ints(n, 85, bins as u32);
        let codes = up(h, &hc);
        let mut want = vec![0u32; bins];
        hc.iter().for_each(|&c| want[c as usize] += 1);
        exact(name, &bincount(&codes, bins).to_vec().unwrap(), &want);
        case_trim(h, name, &sz, Work::Bytes(4.0 * n as f64), &mut || bincount(&codes, bins).eval().unwrap());
    }
}

fn topk_cases(h: &mut Harness) {
    for n in ladder(h, &[1_000_000], false) {
        for (tag, k, largest) in [("top10", 10usize, true), ("bottom10", 10, false)] {
            let name = format!("topk/{tag}_{}", label(n));
            if !h.want(&name) || (!largest && n != 1_000_000) {
                continue;
            }
            let hx = values(n, 91, -1.0, 1.0);
            let x = up(h, &hx);
            let mut s = hx.clone();
            s.sort_by(f32::total_cmp);
            let want: Vec<f32> = if largest { s.iter().rev().take(k).copied().collect() } else { s.iter().take(k).copied().collect() };
            let (v, i) = if largest { x.topk(k) } else { x.bottomk(k) };
            let (gv, gi) = (v.to_vec().unwrap(), i.to_vec().unwrap());
            exact(&name, &gv, &want);
            assert!(gi.iter().zip(&gv).all(|(&i, &v)| hx[i as usize] == v), "{name}: indices");
            case_trim(h, &name, &n.to_string(), Work::Items(n as f64, "Mrow/s"), &mut || {
                let (v, i) = if largest { x.topk(k) } else { x.bottomk(k) };
                eval_all(&[&v]).unwrap();
                i.eval().unwrap();
            });
        }
    }
}

// -------------------------------------------------------------------------------------------------
// filter, search, gather
// -------------------------------------------------------------------------------------------------

fn filter_cases(h: &mut Harness) {
    let n = big(h);
    let sz = n.to_string();
    for (name, rows, cut) in [("filter/sel50_100k", 100_000usize, 0.5f32), ("filter/sel50", n, 0.5), ("filter/sel1", n, 0.99)] {
        if !h.want(name) {
            continue;
        }
        let hx = values(rows, 101, 0.0, 1.0);
        let x = up(h, &hx);
        let want: Vec<f32> = hx.iter().copied().filter(|&v| v > cut).collect();
        let got = x.filter(&x.gt(cut)).unwrap().unwrap();
        exact(name, &got.to_vec().unwrap(), &want);
        drop(got);
        let work = Work::Bytes(4.0 * (rows + want.len()) as f64);
        case_trim(h, name, &format!("{rows}->{}", want.len()), work, &mut || x.filter(&x.gt(cut)).unwrap().unwrap().eval().unwrap());
    }
    if h.want("filter/padded") {
        let hx = values(n, 102, 0.0, 1.0);
        let x = up(h, &hx);
        let want: Vec<f32> = hx.iter().copied().filter(|&v| v > 0.5).collect();
        let (k, c) = x.filter_padded(&x.gt(0.5), 0.0);
        assert_eq!(c.item().unwrap() as usize, want.len());
        exact("filter/padded", &k.to_vec().unwrap()[..want.len()], &want);
        drop((k, c));
        case_trim(h, "filter/padded", &sz, Work::Bytes(4.0 * (n + want.len()) as f64), &mut || {
            let (k, c) = x.filter_padded(&x.gt(0.5), 0.0);
            k.eval().unwrap();
            c.item().unwrap();
        });
    }
    if h.want("filter/nonzero_10pct") {
        let hx = values(n, 103, 0.0, 1.0);
        let x = up(h, &hx);
        let want: Vec<u32> = hx.iter().enumerate().filter(|&(_, &v)| v > 0.9).map(|(i, _)| i as u32).collect();
        let got = x.gt(0.9).nonzero().unwrap().unwrap();
        exact("filter/nonzero_10pct", &got.to_vec().unwrap(), &want);
        drop(got);
        case_trim(h, "filter/nonzero_10pct", &sz, Work::Bytes(4.0 * (n + want.len()) as f64), &mut || x.gt(0.9).nonzero().unwrap().unwrap().eval().unwrap());
    }
    // A pending product, selected by a mask over two other columns.
    if h.want("filter/expr_two_cols") {
        let (ha, hb) = (values(n, 104, 0.0, 1.0), values(n, 105, 0.0, 1.0));
        let (a, b) = (up(h, &ha), up(h, &hb));
        let want: Vec<f32> = ha.iter().zip(&hb).filter(|&(&p, &q)| p > 0.3 && q < 0.6).map(|(&p, &q)| p * q).collect();
        let run = || (&a * &b).filter(&(a.gt(0.3) & b.lt(0.6))).unwrap().unwrap();
        check("filter/expr_two_cols", &run().to_vec().unwrap(), &want, 1e-6, 0.0);
        case_trim(h, "filter/expr_two_cols", &format!("{n}->{}", want.len()), Work::Bytes(4.0 * (2 * n + want.len()) as f64), &mut || run().eval().unwrap());
    }
}

fn sorted_edges(len: usize, seed: u64) -> Vec<f32> {
    let mut e = values(len, seed, 0.0, 1.0);
    e.sort_by(f32::total_cmp);
    e
}

fn search_cases(h: &mut Harness) {
    let n = big(h);
    for (name, rows, len) in [("search/sorted_100k_in_1k", 100_000usize, 1_000usize), ("search/sorted_big_in_1k", n, 1_000), ("search/sorted_big_in_1m", n, 1_000_000)] {
        if !h.want(name) {
            continue;
        }
        let hx = values(rows, 111, -0.1, 1.1);
        let he = sorted_edges(len, 112);
        let (x, e) = (up(h, &hx), up(h, &he));
        let want: Vec<u32> = hx.iter().map(|&v| he.partition_point(|&p| p < v) as u32).collect();
        exact(name, &e.searchsorted(&x, Side::Left).to_vec().unwrap(), &want);
        case_trim(h, name, &format!("{rows} in {len}"), Work::Bytes(8.0 * rows as f64), &mut || e.searchsorted(&x, Side::Left).eval().unwrap());
    }
    // Uneven-bin histogram: digitize against 255 edges, then bincount.
    if h.want("search/digitize_hist") {
        let hx = values(n, 113, -0.1, 1.1);
        let he = sorted_edges(255, 114);
        let (x, e) = (up(h, &hx), up(h, &he));
        let mut want = vec![0u32; 256];
        hx.iter().for_each(|&v| want[he.partition_point(|&p| p <= v)] += 1);
        let run = || bincount(&x.digitize(&e, false), 256);
        exact("search/digitize_hist", &run().to_vec().unwrap(), &want);
        case_trim(h, "search/digitize_hist", &n.to_string(), Work::Bytes(4.0 * n as f64), &mut || run().eval().unwrap());
    }
    if h.want("search/isin_u32") {
        let rows = n;
        let hid = ints(rows, 115, 1_000_000);
        let hset = ints(10_000, 116, 1_000_000);
        let (ids, set) = (up(h, &hid), up(h, &hset));
        let members: std::collections::HashSet<u32> = hset.iter().copied().collect();
        let want: Vec<bool> = hid.iter().map(|v| members.contains(v)).collect();
        exact("search/isin_u32", &ids.isin(&set).to_vec().unwrap(), &want);
        case_trim(h, "search/isin_u32", &format!("{rows} in 10000"), Work::Bytes(5.0 * rows as f64), &mut || ids.isin(&set).eval().unwrap());
    }
    // Join-like lookup: for every left key find its row in a sorted unique key table (some
    // keys are missing) and fetch the payload.
    if h.want("search/join_lookup") {
        let (rows, table) = (n, 100_000usize);
        let hkeys: Vec<u32> = (0..table as u32).map(|i| 5 * i + 3).collect();
        let hpay = values(table, 117, 0.0, 100.0);
        let hleft: Vec<u32> = ints(rows, 118, table as u32).into_iter().zip(ints(rows, 119, 10)).map(|(i, miss)| 5 * i + 3 + u32::from(miss == 0)).collect();
        let (keys, pay, left) = (up(h, &hkeys), up(h, &hpay), up(h, &hleft));
        let run = || {
            let pos = keys.searchsorted(&left, Side::Left);
            let found = keys.take(&pos).eq(&left);
            select(&found, pay.take(&pos), -1.0f32)
        };
        let want: Vec<f32> = hleft
            .iter()
            .map(|&k| {
                let p = hkeys.partition_point(|&e| e < k);
                if p < table && hkeys[p] == k {
                    hpay[p]
                } else {
                    -1.0
                }
            })
            .collect();
        exact("search/join_lookup", &run().to_vec().unwrap(), &want);
        case_trim(h, "search/join_lookup", &format!("{rows} in {table}"), Work::Bytes(12.0 * rows as f64), &mut || run().eval().unwrap());
    }
}

fn take_cases(h: &mut Harness) {
    let n = big(h);
    for (name, rows, mode) in [("take/random_100k", 100_000usize, 0), ("take/random", n, 0), ("take/sorted_idx", n, 1)] {
        if !h.want(name) {
            continue;
        }
        let hx = values(rows, 121, 0.0, 1.0);
        let mut hi = ints(rows, 122, rows as u32);
        if mode == 1 {
            hi.sort_unstable();
        }
        let (x, idx) = (up(h, &hx), up(h, &hi));
        let want: Vec<f32> = hi.iter().map(|&i| hx[i as usize]).collect();
        exact(name, &x.take(&idx).to_vec().unwrap(), &want);
        case_trim(h, name, &rows.to_string(), Work::Bytes(12.0 * rows as f64), &mut || x.take(&idx).eval().unwrap());
    }
}

// -------------------------------------------------------------------------------------------------
// table statistics: columns of a [rows, features] table (axis 0 = rows)
// -------------------------------------------------------------------------------------------------

struct Cols {
    mean: Vec<f64>,
    std: Vec<f64>,
    min: Vec<f32>,
    max: Vec<f32>,
}

fn host_columns(hx: &[f32], rows: usize, feats: usize) -> Cols {
    let mut c = Cols { mean: vec![], std: vec![], min: vec![], max: vec![] };
    for f in 0..feats {
        let col = &hx[f * rows..(f + 1) * rows];
        let mean = col.iter().map(|&v| f64::from(v)).sum::<f64>() / rows as f64;
        let var = col.iter().map(|&v| (f64::from(v) - mean).powi(2)).sum::<f64>() / rows as f64;
        c.mean.push(mean);
        c.std.push(var.sqrt());
        c.min.push(col.iter().copied().fold(f32::MAX, f32::min));
        c.max.push(col.iter().copied().fold(f32::MIN, f32::max));
    }
    c
}

fn as_f32(v: &[f64]) -> Vec<f32> {
    v.iter().map(|&x| x as f32).collect()
}

/// Columns with different scales and offsets, so standardizing means something.
fn table(rows: usize, feats: usize, seed: u64) -> Vec<f32> {
    let mut hx = values(rows * feats, seed, 0.0, 1.0);
    for (f, col) in hx.chunks_mut(rows).enumerate() {
        let (scale, shift) = (1.0 + f as f32 * 0.5, f as f32 * 2.0 - 10.0);
        col.iter_mut().for_each(|v| *v = *v * scale + shift);
    }
    hx
}

fn table_cases(h: &mut Harness) {
    let small = h.ctx.is_cpu() || h.quick;
    let rows = if small { 100_000 } else { 1_000_000 };
    let feats = 32;
    let dims = format!("{rows}x{feats}");
    let elems = rows * feats;
    let hx = table(rows, feats, 131);
    let x = Array::from_slice_on(&h.ctx, &hx, [rows, feats]).unwrap();
    let want = host_columns(&hx, rows, feats);

    // mean, std, min, max as four separate lazy reductions (std is two-pass).
    if h.want("table/describe_4ops") {
        let run = || {
            let r = [x.mean_axis(0), x.std_axis(0), x.min_axis(0), x.max_axis(0)];
            eval_all(&[&r[0], &r[1], &r[2], &r[3]]).unwrap();
            r
        };
        let r = run();
        check("table/describe mean", &r[0].to_vec().unwrap(), &as_f32(&want.mean), 1e-5, 1e-5);
        check("table/describe std", &r[1].to_vec().unwrap(), &as_f32(&want.std), 1e-4, 1e-5);
        exact("table/describe min", &r[2].to_vec().unwrap(), &want.min);
        exact("table/describe max", &r[3].to_vec().unwrap(), &want.max);
        drop(r);
        case_trim(h, "table/describe_4ops", &dims, Work::Bytes(12.0 * elems as f64), &mut || {
            run();
        });
    }
    // The same in one pass: sum, sum of squares, min, max.
    if h.want("table/describe_1pass") {
        let run = || {
            let r = x.reduce_axis(0, [Reduction::Sum, Reduction::SumOfSquares, Reduction::Min, Reduction::Max]);
            eval_all(&[&r[0], &r[1], &r[2], &r[3]]).unwrap();
            r
        };
        let r = run();
        let mean: Vec<f32> = r[0].to_vec().unwrap().iter().map(|&s| s / rows as f32).collect();
        check("table/describe_1pass mean", &mean, &as_f32(&want.mean), 1e-4, 1e-4);
        let std: Vec<f32> = r[1].to_vec().unwrap().iter().zip(&mean).map(|(&q, &m)| (q / rows as f32 - m * m).max(0.0).sqrt()).collect();
        check("table/describe_1pass std", &std, &as_f32(&want.std), 2e-2, 1e-2);
        exact("table/describe_1pass min", &r[2].to_vec().unwrap(), &want.min);
        drop(r);
        case_trim(h, "table/describe_1pass", &dims, Work::Bytes(4.0 * elems as f64), &mut || {
            run();
        });
    }
    if h.want("table/standardize") {
        let run = || (&x - &x.mean_axis(0)) / &x.std_axis(0);
        let mut z = vec![0.0f32; elems];
        for f in 0..feats {
            for i in 0..rows {
                z[f * rows + i] = ((f64::from(hx[f * rows + i]) - want.mean[f]) / want.std[f]) as f32;
            }
        }
        check("table/standardize", &run().to_vec().unwrap(), &z, 1e-3, 1e-4);
        case_trim(h, "table/standardize", &dims, Work::Bytes(12.0 * elems as f64), &mut || run().eval().unwrap());
    }
    // Correlation matrix: standardized columns times their transpose.
    if h.want("table/corr") {
        let run = || {
            let z = (&x - &x.mean_axis(0)) / &x.std_axis(0);
            z.matmul(&z.transpose()) / rows as f32
        };
        let got = run().to_vec().unwrap();
        let mut dot = vec![0.0f64; feats * feats];
        for i in 0..feats {
            for j in i..feats {
                let (a, b) = (&hx[i * rows..(i + 1) * rows], &hx[j * rows..(j + 1) * rows]);
                let s: f64 = a.iter().zip(b).map(|(&p, &q)| f64::from(p) * f64::from(q)).sum();
                let c = (s / rows as f64 - want.mean[i] * want.mean[j]) / (want.std[i] * want.std[j]);
                dot[i * feats + j] = c;
                dot[j * feats + i] = c;
            }
        }
        check("table/corr", &got, &as_f32(&dot), 0.0, 5e-3);
        case_trim(h, "table/corr", &dims, Work::Flops(2.0 * (feats * feats * rows) as f64 + 12.0 * elems as f64), &mut || run().eval().unwrap());
    }
    drop(x);

    // A wide table: covariance of 256 features (sampled entries checked).
    if h.want("table/cov_wide") {
        let (rows, feats) = if small { (20_000usize, 256usize) } else { (100_000, 256) };
        let hx = table(rows, feats, 132);
        let x = Array::from_slice_on(&h.ctx, &hx, [rows, feats]).unwrap();
        let run = || {
            let c = &x - &x.mean_axis(0);
            c.matmul(&c.transpose()) / (rows as f32 - 1.0)
        };
        let got = run().to_vec().unwrap();
        let mean = |f: usize| hx[f * rows..(f + 1) * rows].iter().map(|&v| f64::from(v)).sum::<f64>() / rows as f64;
        let pairs = ints(120, 133, (feats * feats) as u32);
        for p in pairs {
            let (i, j) = (p as usize / feats, p as usize % feats);
            let (a, b) = (&hx[i * rows..(i + 1) * rows], &hx[j * rows..(j + 1) * rows]);
            let (ma, mb) = (mean(i), mean(j));
            let cov = a.iter().zip(b).map(|(&p, &q)| (f64::from(p) - ma) * (f64::from(q) - mb)).sum::<f64>() / (rows as f64 - 1.0);
            let g = f64::from(got[i + feats * j]);
            assert!((g - cov).abs() <= 2e-3 * cov.abs().max(1.0), "table/cov_wide: ({i},{j}): got {g}, want {cov}");
        }
        case_trim(h, "table/cov_wide", &format!("{rows}x{feats}"), Work::Flops(2.0 * (feats * feats * rows) as f64), &mut || run().eval().unwrap());
    }

    // NaN-aware statistics of a table with 5% missing cells.
    if h.want("table/describe_nan") {
        let (rows, feats) = if small { (100_000usize, 16usize) } else { (1_000_000, 16) };
        let mut hx = table(rows, feats, 134);
        for i in (0..hx.len()).step_by(20) {
            hx[i] = f32::NAN;
        }
        let x = Array::from_slice_on(&h.ctx, &hx, [rows, feats]).unwrap();
        let run = || {
            let r = [x.nanmean_axis(0), x.nanmin_axis(0), x.nanmax_axis(0)];
            eval_all(&[&r[0], &r[1], &r[2]]).unwrap();
            r
        };
        let r = run();
        let (mut mean, mut lo, mut hi) = (vec![], vec![], vec![]);
        for col in hx.chunks(rows) {
            let ok: Vec<f32> = col.iter().copied().filter(|v| !v.is_nan()).collect();
            mean.push((ok.iter().map(|&v| f64::from(v)).sum::<f64>() / ok.len() as f64) as f32);
            lo.push(ok.iter().copied().fold(f32::MAX, f32::min));
            hi.push(ok.iter().copied().fold(f32::MIN, f32::max));
        }
        check("table/describe_nan mean", &r[0].to_vec().unwrap(), &mean, 1e-4, 1e-4);
        exact("table/describe_nan min", &r[1].to_vec().unwrap(), &lo);
        exact("table/describe_nan max", &r[2].to_vec().unwrap(), &hi);
        drop(r);
        case_trim(h, "table/describe_nan", &format!("{rows}x{feats}"), Work::Bytes(4.0 * (rows * feats) as f64 * 2.0), &mut || {
            run();
        });
    }
}

// -------------------------------------------------------------------------------------------------
// pivot tables and one-hot encoding
// -------------------------------------------------------------------------------------------------

fn pivot_cases(h: &mut Harness) {
    let n = big(h);
    let sz = n.to_string();
    let ctx = h.ctx.clone();
    for (name, nr, nc) in [("pivot/sum_1000x100", 1000u32, 100u32), ("pivot/sum_24x7", 24, 7)] {
        if !h.want(name) {
            continue;
        }
        let (hr, hc) = (ints(n, 141, nr), ints(n, 142, nc));
        let hv = values(n, 143, 0.0, 10.0);
        let (r, c, v) = (up(h, &hr), up(h, &hc), up(h, &hv));
        let cells = (nr * nc) as usize;
        let run = || {
            let mut t = Array::<f32>::zeros_on(&ctx, cells).unwrap();
            t.put_add(&(&r * nc + &c), &v).unwrap();
            t
        };
        let flat: Vec<u32> = hr.iter().zip(&hc).map(|(&a, &b)| a * nc + b).collect();
        let (sum, mag) = host_group_sum(&flat, &hv, cells);
        check_sums(name, &run().to_vec().unwrap(), &sum, &mag, 2e-4);
        case_trim(h, name, &sz, Work::Bytes(12.0 * n as f64), &mut || run().eval().unwrap());
    }
    // Mean table: sums over counts, both scattered.
    if h.want("pivot/mean_100x10") {
        let (nr, nc) = (100u32, 10u32);
        let (hr, hc) = (ints(n, 144, nr), ints(n, 145, nc));
        let hv = values(n, 146, 0.0, 10.0);
        let (r, c, v) = (up(h, &hr), up(h, &hc), up(h, &hv));
        let cells = (nr * nc) as usize;
        let run = || {
            let idx = &r * nc + &c;
            let mut sum = Array::<f32>::zeros_on(&ctx, cells).unwrap();
            let mut count = Array::<u32>::zeros_on(&ctx, cells).unwrap();
            sum.put_add(&idx, &v).unwrap();
            count.put_add(&idx, 1u32).unwrap();
            let mean = &sum / &count.to_f32();
            mean.eval().unwrap();
            mean
        };
        let flat: Vec<u32> = hr.iter().zip(&hc).map(|(&a, &b)| a * nc + b).collect();
        let (sum, _) = host_group_sum(&flat, &hv, cells);
        let mut cnt = vec![0u32; cells];
        flat.iter().for_each(|&i| cnt[i as usize] += 1);
        let want: Vec<f32> = sum.iter().zip(&cnt).map(|(&s, &c)| (s / f64::from(c)) as f32).collect();
        check("pivot/mean_100x10", &run().to_vec().unwrap(), &want, 1e-3, 1e-4);
        case_trim(h, "pivot/mean_100x10", &sz, Work::Bytes(20.0 * n as f64), &mut || {
            run();
        });
    }
    // One-hot of class labels: [classes, rows] f32.
    for (name, rows, classes) in [("pivot/onehot_10", if n > 1_000_000 { 1_000_000usize } else { 500_000 }, 10usize), ("pivot/onehot_100", 100_000, 100)] {
        if !h.want(name) {
            continue;
        }
        let hl = ints(rows, 147, classes as u32);
        let labels = up(h, &hl);
        let cls = Array::<u32>::range_on(&h.ctx, [classes, rows], 0).unwrap();
        let run = || cls.eq(labels.reshape([1, rows])).to_f32();
        let mut want = vec![0.0f32; classes * rows];
        for (i, &l) in hl.iter().enumerate() {
            want[i * classes + l as usize] = 1.0;
        }
        exact(name, &run().to_vec().unwrap(), &want);
        case_trim(h, name, &format!("{rows}x{classes}"), Work::Bytes(4.0 * (rows * classes) as f64), &mut || run().eval().unwrap());
    }
}

// -------------------------------------------------------------------------------------------------
// rolling windows and differences
// -------------------------------------------------------------------------------------------------

/// Sum of `w` shifted slices along axis 0: element `i` is the sum of the window starting at `i`.
fn window_sum(x: &Array<f32>, w: usize) -> Array<f32> {
    let m = x.shape().dims()[0] - w + 1;
    let mut acc = x.slice(0, 0..m);
    for k in 1..w {
        acc = acc + x.slice(0, k..k + m);
    }
    acc
}

fn host_window_mean(hx: &[f32], w: usize) -> Vec<f32> {
    let mut acc: f64 = hx[..w].iter().map(|&v| f64::from(v)).sum();
    let mut out = vec![(acc / w as f64) as f32];
    for i in w..hx.len() {
        acc += f64::from(hx[i]) - f64::from(hx[i - w]);
        out.push((acc / w as f64) as f32);
    }
    out
}

fn rolling_cases(h: &mut Harness) {
    let n = big(h);
    for (name, rows, w) in [("rolling/mean_w16", n, 16usize), ("rolling/mean_w128", 1_000_000, 128)] {
        if !h.want(name) {
            continue;
        }
        let hx = values(rows, 151, 0.0, 1.0);
        let x = up(h, &hx);
        let run = || window_sum(&x, w) / w as f32;
        check(name, &run().to_vec().unwrap(), &host_window_mean(&hx, w), 1e-4, 1e-5);
        case_trim(h, name, &rows.to_string(), Work::Bytes(8.0 * rows as f64), &mut || run().eval().unwrap());
    }
    if h.want("rolling/std_w32") {
        let w = 32usize;
        let hx = values(n, 152, 0.0, 1.0);
        let x = up(h, &hx);
        let run = || {
            let sq = &x * &x;
            let mean = window_sum(&x, w) / w as f32;
            let msq = window_sum(&sq, w) / w as f32;
            (msq - &mean * &mean).maximum(0.0).sqrt()
        };
        let mean = host_window_mean(&hx, w);
        let sq: Vec<f32> = hx.iter().map(|&v| v * v).collect();
        let msq = host_window_mean(&sq, w);
        let want: Vec<f32> = mean.iter().zip(&msq).map(|(&m, &q)| (q - m * m).max(0.0).sqrt()).collect();
        check("rolling/std_w32", &run().to_vec().unwrap(), &want, 0.0, 5e-3);
        case_trim(h, "rolling/std_w32", &n.to_string(), Work::Bytes(8.0 * n as f64), &mut || run().eval().unwrap());
    }
    // Window means from the difference of a running sum (zero-mean data keeps it accurate).
    if h.want("rolling/mean_cumsum_w16") {
        let (rows, w) = (1_000_000usize, 16usize);
        let hx = values(rows, 153, -1.0, 1.0);
        let x = up(h, &hx);
        let run = || {
            let c = x.cumsum(0);
            (c.slice(0, w..rows) - c.slice(0, 0..rows - w)) / w as f32
        };
        let full = host_window_mean(&hx, w);
        check("rolling/mean_cumsum_w16", &run().to_vec().unwrap(), &full[1..], 0.0, 2e-3);
        case_trim(h, "rolling/mean_cumsum_w16", &rows.to_string(), Work::Bytes(16.0 * rows as f64), &mut || run().eval().unwrap());
    }
    // Per-column windows of a table: the window runs along axis 0.
    if h.want("rolling/mean_table_w16") {
        let (rows, feats, w) = (n / 32, 32usize, 16usize);
        let hx = values(rows * feats, 154, 0.0, 1.0);
        let x = Array::from_slice_on(&h.ctx, &hx, [rows, feats]).unwrap();
        let run = || window_sum(&x, w) / w as f32;
        let want: Vec<f32> = hx.chunks(rows).flat_map(|c| host_window_mean(c, w)).collect();
        check("rolling/mean_table_w16", &run().to_vec().unwrap(), &want, 1e-4, 1e-5);
        case_trim(h, "rolling/mean_table_w16", &format!("{rows}x{feats}"), Work::Bytes(8.0 * (rows * feats) as f64), &mut || run().eval().unwrap());
    }
    if h.want("rolling/diff2") {
        let hx = values(n, 155, 0.0, 1.0);
        let x = up(h, &hx);
        let run = || x.slice(0, 2..) - &(x.slice(0, 1..n - 1) * 2.0) + x.slice(0, ..n - 2);
        let want: Vec<f32> = hx.windows(3).map(|w| w[2] - 2.0 * w[1] + w[0]).collect();
        check("rolling/diff2", &run().to_vec().unwrap(), &want, 0.0, 1e-5);
        case_trim(h, "rolling/diff2", &n.to_string(), Work::Bytes(8.0 * n as f64), &mut || run().eval().unwrap());
    }
}

// -------------------------------------------------------------------------------------------------
// pipelines: the whole job per call
// -------------------------------------------------------------------------------------------------

fn pipeline_cases(h: &mut Harness) {
    let big = big(h);
    // ETL: load columns, filter, factorize the key, group-aggregate, sort the result, top rows.
    for n in ladder(h, &[1_000_000], false).into_iter().filter(|&n| n <= big) {
        let name = format!("pipeline/etl_{}", label(n));
        if !h.want(&name) {
            continue;
        }
        let ha = values(n, 161, 0.0, 100.0);
        let hq = values(n, 162, 0.0, 10.0);
        let hk: Vec<u32> = ints(n, 163, 1000).into_iter().map(|c| 5000 + 13 * c).collect();
        let top = 10usize;
        let ctx = h.ctx.clone();
        // Upload counts as part of the call: the table arrives as host columns.
        let run = |ctx: &Context| {
            let (amount, qty, key) = (up(ctx, &ha), up(ctx, &hq), up(ctx, &hk));
            let mask = amount.gt(10.0) & qty.gt(1.0);
            let idx = mask.nonzero().unwrap().unwrap();
            let (uniq, codes) = key.take(&idx).factorize();
            let groups = uniq.len();
            let a = amount.take(&idx);
            let revenue = &a * &qty.take(&idx);
            let total = revenue.group_sum(&codes, groups);
            let mean = revenue.group_mean(&codes, groups);
            let best = a.group_max(&codes, groups);
            let order = total.argsort_descending();
            let pick = order.slice(0, 0..top.min(groups));
            let cols = [total.take(&pick), mean.take(&pick), best.take(&pick)];
            let keys = uniq.take(&pick).to_vec().unwrap();
            let out: Vec<Vec<f32>> = cols.iter().map(|c| c.to_vec().unwrap()).collect();
            (keys, out)
        };
        // Host reference.
        let mut table: std::collections::HashMap<u32, (f64, u32, f32)> = std::collections::HashMap::new();
        for i in 0..n {
            if ha[i] > 10.0 && hq[i] > 1.0 {
                let e = table.entry(hk[i]).or_insert((0.0, 0, f32::MIN));
                e.0 += f64::from(ha[i] * hq[i]);
                e.1 += 1;
                e.2 = e.2.max(ha[i]);
            }
        }
        let mut ranked: Vec<(u32, (f64, u32, f32))> = table.into_iter().collect();
        ranked.sort_by(|a, b| b.1 .0.total_cmp(&a.1 .0));
        let (keys, out) = run(&ctx);
        assert_eq!(keys.len(), top, "{name}: rows");
        for r in 0..top {
            check_scalar(&name, f64::from(out[0][r]), ranked[r].1 .0, 1e-3);
            let (_, (tot, cnt, best)) = ranked.iter().find(|(k, _)| *k == keys[r]).copied().unwrap();
            check_scalar(&name, f64::from(out[0][r]), tot, 1e-3);
            check_scalar(&name, f64::from(out[1][r]), tot / f64::from(cnt), 1e-3);
            assert_eq!(out[2][r], best, "{name}: max of {}", keys[r]);
        }
        case_trim(h, &name, &n.to_string(), Work::Items(n as f64, "Mrow/s"), &mut || {
            run(&ctx);
        });
    }
    // EDA of one column: moments, extremes, five percentiles from one sort, a 64-bin histogram.
    for n in ladder(h, &[1_000_000], false).into_iter().filter(|&n| n <= big) {
        let name = format!("pipeline/eda_{}", label(n));
        if !h.want(&name) {
            continue;
        }
        let (a, b, c, d) = (values(n, 171, 0.0, 1.0), values(n, 172, 0.0, 1.0), values(n, 173, 0.0, 1.0), values(n, 174, 0.0, 1.0));
        let hx: Vec<f32> = (0..n).map(|i| (a[i] + b[i] + c[i] + d[i]) * 0.25).collect();
        drop((a, b, c, d));
        let x = up(h, &hx);
        let qs = [0.05f32, 0.25, 0.5, 0.75, 0.95];
        let pos: Vec<u32> = qs.iter().map(|&q| (q * (n - 1) as f32) as u32).collect();
        let pos_arr = up(h, &pos);
        let run = || {
            let [sum, ssq, lo, hi] = x.reduce([Reduction::Sum, Reduction::SumOfSquares, Reduction::Min, Reduction::Max]).to_array::<4>().unwrap();
            let picked = x.sort().take(&pos_arr).to_vec().unwrap();
            let hist = x.histogram(64, 0.0, 1.0).to_vec().unwrap();
            ([sum, ssq, lo, hi], picked, hist)
        };
        let (stats, picked, hist) = run();
        let mean = hx.iter().map(|&v| f64::from(v)).sum::<f64>() / n as f64;
        check_scalar(&name, f64::from(stats[0]), mean * n as f64, 1e-4);
        assert_eq!(stats[2], hx.iter().copied().fold(f32::MAX, f32::min));
        assert_eq!(stats[3], hx.iter().copied().fold(f32::MIN, f32::max));
        let mut s = hx.clone();
        s.sort_by(f32::total_cmp);
        let want_q: Vec<f32> = pos.iter().map(|&p| s[p as usize]).collect();
        exact(&name, &picked, &want_q);
        exact(&name, &hist, &host_hist(&hx, 64, 0.0, 1.0));
        case_trim(h, &name, &n.to_string(), Work::Items(n as f64, "Mrow/s"), &mut || {
            run();
        });
    }
    // Outlier removal: z-scores from device-resident statistics, filter |z| < 3, clean mean.
    {
        let name = "pipeline/outliers";
        if h.want(name) {
            let n = big;
            let mut hx = values(n, 181, 0.0, 1.0);
            for i in (0..n).step_by(97) {
                hx[i] = 6.0 + (i % 7) as f32;
            }
            let x = up(h, &hx);
            let run = || {
                let z = (&x - &x.mean_axis(0)) / &x.std_axis(0);
                let kept = x.filter(&z.abs().lt(3.0)).unwrap().unwrap();
                (kept.len(), kept.mean().item().unwrap())
            };
            let mean = hx.iter().map(|&v| f64::from(v)).sum::<f64>() / n as f64;
            let std = (hx.iter().map(|&v| (f64::from(v) - mean).powi(2)).sum::<f64>() / n as f64).sqrt();
            let clean: Vec<f64> = hx.iter().map(|&v| f64::from(v)).filter(|v| ((v - mean) / std).abs() < 3.0).collect();
            let (count, got_mean) = run();
            assert_eq!(count, clean.len(), "{name}: kept rows");
            check_scalar(name, f64::from(got_mean), clean.iter().sum::<f64>() / clean.len() as f64, 1e-4);
            case_trim(h, name, &n.to_string(), Work::Items(n as f64, "Mrow/s"), &mut || {
                run();
            });
        }
    }
}

// -------------------------------------------------------------------------------------------------
// more sorts, scans, reductions
// -------------------------------------------------------------------------------------------------

/// Rows of the 64-column tables of the sort and quantile cases: 1M on a GPU, 64k on the CPU
/// backend and in quick runs.
fn table_rows(h: &Harness) -> usize {
    if h.ctx.is_cpu() || h.quick { 1 << 16 } else { 1 << 20 }
}

/// `data` (shape `[d0, d1]`, axis 0 fastest) with every line along `axis` sorted.
fn host_sort_axis(data: &[f32], dims: [usize; 2], axis: usize) -> Vec<f32> {
    let mut out = data.to_vec();
    let (stride, n) = if axis == 0 { (1, dims[0]) } else { (dims[0], dims[1]) };
    let lines = data.len() / n;
    for l in 0..lines {
        let base = if axis == 0 { l * n } else { l };
        let mut line: Vec<f32> = (0..n).map(|k| data[base + k * stride]).collect();
        line.sort_by(f32::total_cmp);
        for (k, v) in line.into_iter().enumerate() {
            out[base + k * stride] = v;
        }
    }
    out
}

fn sort_extra_cases(h: &mut Harness) {
    let n = big(h);
    if h.want("sort/f32_desc") {
        let hx = rand_f32(n, 7);
        let x = up(h, &hx);
        let got = x.sort_descending();
        let mut got_keys: Vec<u32> = got.to_vec().unwrap().into_iter().map(key_f32).collect();
        got_keys.reverse();
        let in_keys: Vec<u32> = hx.iter().copied().map(key_f32).collect();
        check_sorted("sort/f32_desc", &got_keys, &in_keys);
        drop(got);
        case_trim(h, "sort/f32_desc", &n.to_string(), Work::Items(n as f64, "Mkey/s"), &mut || x.sort_descending().eval().unwrap());
    }
    // Sort a table by two columns: two stable argsorts and two gathers.
    if h.want("sort/lexsort2_u32") {
        let (ha, hb) = (ints(n, 8, 1000), ints(n, 9, u32::MAX));
        let (a, b) = (up(h, &ha), up(h, &hb));
        let run = || {
            let by_b = b.argsort();
            let by_a = a.take(&by_b).argsort();
            by_b.take(&by_a)
        };
        let mut want: Vec<u32> = (0..n as u32).collect();
        want.sort_by_key(|&i| (ha[i as usize], hb[i as usize], i));
        exact("sort/lexsort2_u32", &run().to_vec().unwrap(), &want);
        case_trim(h, "sort/lexsort2_u32", &n.to_string(), Work::Items(n as f64, "Mrow/s"), &mut || run().eval().unwrap());
    }
    // Many short lines.
    let total = if h.ctx.is_cpu() || h.quick { 1 << 20 } else { 1 << 22 };
    if h.want("sort/axis0_len16") {
        let dims = [16usize, total / 16];
        let hx = rand_f32(total, 10);
        let x = Array::from_slice_on(&h.ctx, &hx, dims).unwrap();
        let mut want = hx.clone();
        want.chunks_mut(16).for_each(|c| c.sort_by(f32::total_cmp));
        exact("sort/axis0_len16", &x.sort_axis(0).to_vec().unwrap(), &want);
        case_trim(h, "sort/axis0_len16", &format!("{}x{}", dims[0], dims[1]), Work::Items(total as f64, "Mkey/s"), &mut || x.sort_axis(0).eval().unwrap());
    }
    // A table of 64 columns: sort the columns (1M-element contiguous lines), the rows (64-element
    // strided lines, 1M of them), and a transposed table by its short contiguous lines.
    let rows = table_rows(h);
    for (name, dims, axis) in [("sort/axis0_x64", [rows, 64usize], 0usize), ("sort/axis1_x64", [rows, 64], 1), ("sort/axis0_len64", [64, rows], 0)] {
        if !h.want(name) {
            continue;
        }
        let hx = rand_f32(rows * 64, 15);
        let x = Array::from_slice_on(&h.ctx, &hx, dims).unwrap();
        exact(name, &x.sort_axis(axis).to_vec().unwrap(), &host_sort_axis(&hx, dims, axis));
        case_trim(h, name, &format!("{}x{}", dims[0], dims[1]), Work::Items((rows * 64) as f64, "Mkey/s"), &mut || x.sort_axis(axis).eval().unwrap());
    }
}

fn argsort_extra_cases(h: &mut Harness) {
    if let Some(m) = huge(h) {
        let name = format!("argsort/f32_rand_{}", label(m));
        if h.want(&name) {
            argsort_case(h, &name, &rand_f32(m, 16), key_f32);
        }
    }
    // The rank transform (Spearman correlation, percentile ranks): argsort of argsort.
    let n = big(h);
    if h.want("argsort/rank_f32") {
        let hx = values(n, 17, 0.0, 1.0);
        let x = up(h, &hx);
        let run = || x.argsort().argsort();
        let mut order: Vec<u32> = (0..n as u32).collect();
        order.sort_by(|&p, &q| hx[p as usize].total_cmp(&hx[q as usize]).then(p.cmp(&q)));
        let mut want = vec![0u32; n];
        for (rank, &i) in order.iter().enumerate() {
            want[i as usize] = rank as u32;
        }
        exact("argsort/rank_f32", &run().to_vec().unwrap(), &want);
        case_trim(h, "argsort/rank_f32", &n.to_string(), Work::Items(2.0 * n as f64, "Mkey/s"), &mut || run().eval().unwrap());
    }
}

fn reduce_extra_cases(h: &mut Harness) {
    let n = big(h);
    if h.want("reduce/argmax_f32") {
        let hx = values(n, 31, 0.0, 1.0);
        let x = up(h, &hx);
        let mut best = 0usize;
        for (i, &v) in hx.iter().enumerate() {
            if v > hx[best] {
                best = i;
            }
        }
        assert_eq!(x.argmax().item().unwrap() as usize, best, "reduce/argmax_f32");
        case_trim(h, "reduce/argmax_f32", &n.to_string(), Work::Bytes(4.0 * n as f64), &mut || {
            x.argmax().item().unwrap();
        });
    }
    // Only 32 long lines to reduce along axis 1 (strided): parallelism has to come from splitting lines.
    if h.want("reduce/sum_axis1_32xM") {
        let m = n / 32;
        let hx = values(32 * m, 32, 0.0, 1.0);
        let x = Array::from_slice_on(&h.ctx, &hx, [32, m]).unwrap();
        let want: Vec<f32> = (0..32).map(|i| (0..m).map(|j| f64::from(hx[i + 32 * j])).sum::<f64>() as f32).collect();
        check("reduce/sum_axis1_32xM", &x.sum_axis(1).to_vec().unwrap(), &want, 1e-4, 1e-3);
        case_trim(h, "reduce/sum_axis1_32xM", &format!("32x{m}"), Work::Bytes(4.0 * (32 * m) as f64), &mut || x.sum_axis(1).eval().unwrap());
    }
    // Millions of 4-element lines (per-pixel channel sums).
    if h.want("reduce/sum_axis0_4xM") {
        let m = n / 4;
        let hx = values(4 * m, 33, 0.0, 1.0);
        let x = Array::from_slice_on(&h.ctx, &hx, [4, m]).unwrap();
        let want: Vec<f32> = hx.chunks(4).map(|c| c.iter().sum()).collect();
        check("reduce/sum_axis0_4xM", &x.sum_axis(0).to_vec().unwrap(), &want, 1e-6, 1e-6);
        case_trim(h, "reduce/sum_axis0_4xM", &format!("4x{m}"), Work::Bytes(4.0 * (4 * m + m) as f64), &mut || x.sum_axis(0).eval().unwrap());
    }
}

fn scan_extra_cases(h: &mut Harness) {
    let n = big(h);
    // Running maximum: a scan with `max` (drawdown, cummax).
    if h.want("scan/cummax_f32") {
        let hx = values(n, 46, 0.0, 1.0);
        let x = up(h, &hx);
        let mut m = f32::MIN;
        let want: Vec<f32> = hx
            .iter()
            .map(|&v| {
                m = m.max(v);
                m
            })
            .collect();
        exact("scan/cummax_f32", &x.scan(0, |a, b| a.maximum(b)).to_vec().unwrap(), &want);
        case_trim(h, "scan/cummax_f32", &n.to_string(), Work::Bytes(8.0 * n as f64), &mut || x.scan(0, |a, b| a.maximum(b)).eval().unwrap());
    }
    // Many short lines: cumulative sum within groups of 16 rows.
    if h.want("scan/cumsum_len16") {
        let total = if h.ctx.is_cpu() || h.quick { 1 << 20 } else { 1 << 22 };
        let hx = values(total, 47, 0.0, 1.0);
        let x = Array::from_slice_on(&h.ctx, &hx, [16, total / 16]).unwrap();
        let mut want = vec![0.0f32; total];
        for (src, dst) in hx.chunks(16).zip(want.chunks_mut(16)) {
            let mut acc = 0.0f64;
            for (s, d) in src.iter().zip(dst) {
                acc += f64::from(*s);
                *d = acc as f32;
            }
        }
        check("scan/cumsum_len16", &x.cumsum(0).to_vec().unwrap(), &want, 1e-5, 1e-5);
        case_trim(h, "scan/cumsum_len16", &format!("16x{}", total / 16), Work::Bytes(8.0 * total as f64), &mut || x.cumsum(0).eval().unwrap());
    }
}

/// Embedding lookup as a product: one-hot labels (computed inside the product's loads) times a
/// `[dim, classes]` weight table.
fn pivot_extra_cases(h: &mut Harness) {
    if h.want("pivot/onehot_matmul") {
        let (rows, classes, dim) = (100_000usize, 100usize, 64usize);
        let hl = ints(rows, 148, classes as u32);
        let hw = values(dim * classes, 149, -1.0, 1.0);
        let labels = up(h, &hl);
        let w = Array::from_slice_on(&h.ctx, &hw, [dim, classes]).unwrap();
        let cls = Array::<u32>::range_on(&h.ctx, [classes, rows], 0).unwrap();
        let run = || cls.eq(labels.reshape([1, rows])).to_f32().matmul(&w);
        let mut want = vec![0.0f32; dim * rows];
        for (i, &l) in hl.iter().enumerate() {
            want[dim * i..dim * (i + 1)].copy_from_slice(&hw[dim * l as usize..dim * (l as usize + 1)]);
        }
        // The weights go through TF32 where the product runs on tensor cores (a one-hot gather loses 2^-11 of them).
        check("pivot/onehot_matmul", &run().to_vec().unwrap(), &want, 1e-3, 1e-3);
        case_trim(h, "pivot/onehot_matmul", &format!("{rows}x{classes}x{dim}"), Work::Flops(2.0 * (rows * classes * dim) as f64), &mut || run().eval().unwrap());
    }
}

// -------------------------------------------------------------------------------------------------
// small problems: latency-bound, one table row each
// -------------------------------------------------------------------------------------------------

fn host_corr(hx: &[f32], rows: usize, feats: usize) -> Vec<f64> {
    let cols = host_columns(hx, rows, feats);
    let mut out = vec![0.0; feats * feats];
    for i in 0..feats {
        for j in 0..feats {
            let (a, b) = (&hx[i * rows..(i + 1) * rows], &hx[j * rows..(j + 1) * rows]);
            let s: f64 = a.iter().zip(b).map(|(&p, &q)| f64::from(p) * f64::from(q)).sum();
            out[i * feats + j] = (s / rows as f64 - cols.mean[i] * cols.mean[j]) / (cols.std[i] * cols.std[j]);
        }
    }
    out
}

fn small_cases(h: &mut Harness) {
    let n = 10_000;
    let sz = n.to_string();
    if h.want("small/sort_10k") {
        let hx = rand_f32(n, 201);
        let x = up(h, &hx);
        let mut want = hx.clone();
        want.sort_by(f32::total_cmp);
        exact("small/sort_10k", &x.sort().to_vec().unwrap(), &want);
        case_trim(h, "small/sort_10k", &sz, Work::None, &mut || x.sort().eval().unwrap());
    }
    if h.want("small/group_sum_10k_g16") {
        let hv = values(n, 202, 0.0, 1.0);
        let hc = ints(n, 203, 16);
        let (v, c) = (up(h, &hv), up(h, &hc));
        let (sum, mag) = host_group_sum(&hc, &hv, 16);
        check_sums("small/group_sum_10k_g16", &v.group_sum(&c, 16).to_vec().unwrap(), &sum, &mag, 1e-5);
        case_trim(h, "small/group_sum_10k_g16", &sz, Work::None, &mut || v.group_sum(&c, 16).eval().unwrap());
    }
    if h.want("small/hist_10k") {
        let hx = values(n, 204, -0.25, 1.25);
        let x = up(h, &hx);
        exact("small/hist_10k", &x.histogram(256, 0.0, 1.0).to_vec().unwrap(), &host_hist(&hx, 256, 0.0, 1.0));
        case_trim(h, "small/hist_10k", &sz, Work::None, &mut || x.histogram(256, 0.0, 1.0).eval().unwrap());
    }
    if h.want("small/median_1k") {
        let hx = values(1000, 205, -3.0, 3.0);
        let x = up(h, &hx);
        let mut s = hx.clone();
        s.sort_by(f32::total_cmp);
        check_scalar("small/median_1k", f64::from(x.median().item().unwrap()), f64::from(host_quantile(&s, 0.5)), 1e-6);
        case_trim(h, "small/median_1k", "1000", Work::None, &mut || {
            x.median().item().unwrap();
        });
    }
    if h.want("small/topk10_10k") {
        let hx = values(n, 206, -1.0, 1.0);
        let x = up(h, &hx);
        let mut s = hx.clone();
        s.sort_by(|a, b| b.total_cmp(a));
        exact("small/topk10_10k", &x.topk(10).0.to_vec().unwrap(), &s[..10]);
        case_trim(h, "small/topk10_10k", &sz, Work::None, &mut || x.topk(10).0.eval().unwrap());
    }
    if h.want("small/filter_10k") {
        let hx = values(n, 207, 0.0, 1.0);
        let x = up(h, &hx);
        let want: Vec<f32> = hx.iter().copied().filter(|&v| v > 0.5).collect();
        exact("small/filter_10k", &x.filter(&x.gt(0.5)).unwrap().unwrap().to_vec().unwrap(), &want);
        case_trim(h, "small/filter_10k", &sz, Work::None, &mut || x.filter(&x.gt(0.5)).unwrap().unwrap().eval().unwrap());
    }
    if h.want("small/cumsum_10k") {
        let hx = values(n, 208, 0.0, 1.0);
        let x = up(h, &hx);
        let mut acc = 0.0f64;
        let want: Vec<f32> = hx
            .iter()
            .map(|&v| {
                acc += f64::from(v);
                acc as f32
            })
            .collect();
        check("small/cumsum_10k", &x.cumsum(0).to_vec().unwrap(), &want, 1e-5, 1e-4);
        case_trim(h, "small/cumsum_10k", &sz, Work::None, &mut || x.cumsum(0).eval().unwrap());
    }
    if h.want("small/searchsorted_10k_in_100") {
        let hx = values(n, 209, -0.1, 1.1);
        let he = sorted_edges(100, 210);
        let (x, e) = (up(h, &hx), up(h, &he));
        let want: Vec<u32> = hx.iter().map(|&v| he.partition_point(|&p| p < v) as u32).collect();
        exact("small/searchsorted_10k_in_100", &e.searchsorted(&x, Side::Left).to_vec().unwrap(), &want);
        case_trim(h, "small/searchsorted_10k_in_100", &sz, Work::None, &mut || e.searchsorted(&x, Side::Left).eval().unwrap());
    }
    // A 1000 x 16 table: describe and correlate.
    let (rows, feats) = (1000usize, 16usize);
    let hx = table(rows, feats, 211);
    let x = Array::from_slice_on(&h.ctx, &hx, [rows, feats]).unwrap();
    if h.want("small/describe_1kx16") {
        let want = host_columns(&hx, rows, feats);
        let run = || {
            let r = [x.mean_axis(0), x.std_axis(0), x.min_axis(0), x.max_axis(0)];
            eval_all(&[&r[0], &r[1], &r[2], &r[3]]).unwrap();
            r
        };
        let r = run();
        check("small/describe_1kx16 mean", &r[0].to_vec().unwrap(), &as_f32(&want.mean), 1e-5, 1e-5);
        check("small/describe_1kx16 std", &r[1].to_vec().unwrap(), &as_f32(&want.std), 1e-4, 1e-5);
        drop(r);
        case_trim(h, "small/describe_1kx16", "1000x16", Work::None, &mut || {
            run();
        });
    }
    if h.want("small/corr_1kx16") {
        let run = || {
            let z = (&x - &x.mean_axis(0)) / &x.std_axis(0);
            z.matmul(&z.transpose()) / rows as f32
        };
        check("small/corr_1kx16", &run().to_vec().unwrap(), &as_f32(&host_corr(&hx, rows, feats)), 0.0, 2e-3);
        case_trim(h, "small/corr_1kx16", "1000x16", Work::None, &mut || run().eval().unwrap());
    }
}

// -------------------------------------------------------------------------------------------------
// Paging
// -------------------------------------------------------------------------------------------------

/// Twelve columns of which a quarter fit the device budget, each read by a reduction per sweep:
/// every sweep pages the columns through the budget.
fn spill_cases(h: &mut Harness) {
    column_sweep(h);
    recompute_sweep(h);
}

fn column_sweep(h: &mut Harness) {
    if !h.want("spill/column_sweep") {
        return;
    }
    let n: usize = if h.quick { 1 << 20 } else { 1 << 22 };
    let columns = 12;
    let budget = (n * 4 * columns / 4) as u64;
    let ctx = Context::with_options(ContextOptions::from_env().memory_budget(budget)).expect("no device");
    let cols: Vec<Array<f32>> = (0..columns).map(|k| Array::from_slice_on(&ctx, &values(n, 300 + k as u64, -1.0, 1.0), n).unwrap()).collect();
    let sweep = || {
        let sums: Vec<f32> = cols.iter().map(|c| (c * 2.0).sum().item().unwrap()).collect();
        sums.iter().sum::<f32>()
    };
    let want = sweep();
    let size = format!("{columns}x{}", label(n));
    h.transfer_case("spill/column_sweep", &size, Work::Bytes((n * 4 * columns) as f64), &mut || {
        assert_eq!(sweep().to_bits(), want.to_bits());
    });
    println!("  spill/column_sweep: {} MiB spilled", ctx.spilled_bytes() >> 20);
}

/// Twelve cheap intermediates of two inputs that stay hot, `a * s + b`, held by the program: the
/// arrays together are one and a half times the device budget, so every sweep that reads them all
/// pages the intermediates. Each is a function of resident inputs: evicting one is free of
/// transfers when the pager recomputes it at the next read instead of reading it back.
fn recompute_sweep(h: &mut Harness) {
    if !h.want("spill/recompute_sweep") {
        return;
    }
    let n: usize = if h.quick { 1 << 20 } else { 1 << 22 };
    let steps = 12;
    let total = n * 4 * (steps + 2);
    let ctx = Context::with_options(ContextOptions::from_env().memory_budget((total * 2 / 3) as u64)).expect("no device");
    let a = Array::from_slice_on(&ctx, &values(n, 400, -1.0, 1.0), n).unwrap();
    let b = Array::from_slice_on(&ctx, &values(n, 401, -1.0, 1.0), n).unwrap();
    let held: Vec<Array<f32>> = (0..steps)
        .map(|k| {
            let t = &a * (k as f32 + 1.0) + &b;
            t.eval().unwrap();
            t
        })
        .collect();
    ctx.flush().unwrap();
    // The inputs are read at every step, so they stay resident (the least recently used segments
    // are the intermediates) while the intermediates page.
    let sweep = || {
        let sums: Vec<f32> = held
            .iter()
            .map(|t| {
                std::hint::black_box((&a + &b).sum().item().unwrap());
                t.sum().item().unwrap()
            })
            .collect();
        sums.iter().sum::<f32>()
    };
    let want = sweep();
    let size = format!("{steps}x{}", label(n));
    let before = ctx.paging_stats();
    h.transfer_case("spill/recompute_sweep", &size, Work::Bytes((n * 4 * steps) as f64), &mut || {
        assert_eq!(sweep().to_bits(), want.to_bits());
    });
    let after = ctx.paging_stats();
    println!(
        "  spill/recompute_sweep: {} MiB spilled, {} readback evictions, {} recompute evictions",
        ctx.spilled_bytes() >> 20,
        after.readback_evictions - before.readback_evictions,
        after.recompute_evictions - before.recompute_evictions
    );
}
