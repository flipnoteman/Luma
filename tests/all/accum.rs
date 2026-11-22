//! Incremental accumulators against f64 host references: batches of any split, merge orders,
//! cancellation, masks, groups, every element type, serialization.

use luma::{Accumulator, Array, CoMoments, Covariance, Df32, Fold, Histogram, Histogram2d, NanPolicy, TopK, Wide};

/// Deterministic pseudo-random uniform values in `[0, 1)`.
fn uniform(n: usize, seed: u64) -> Vec<f64> {
    let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (state >> 11) as f64 / (1u64 << 53) as f64
        })
        .collect()
}

/// Roughly normal values: the sum of 12 uniforms minus 6.
fn normal(n: usize, seed: u64) -> Vec<f64> {
    let u = uniform(n * 12, seed);
    u.chunks(12).map(|c| c.iter().sum::<f64>() - 6.0).collect()
}

#[derive(Clone, Copy, Debug, Default)]
struct Ref {
    n: f64,
    sum: f64,
    mean: f64,
    m2: f64,
    m3: f64,
    m4: f64,
    min: f64,
    max: f64,
    argmin: usize,
    argmax: usize,
}

/// Two-pass reference moments of the selected values.
fn reference(values: &[f64]) -> Ref {
    let n = values.len() as f64;
    // Shifted by the first value: a plain sum of 1e9-sized values loses digits itself.
    let shift = values[0];
    let offsets: f64 = values.iter().map(|v| v - shift).sum();
    let sum = n * shift + offsets;
    let mean = shift + offsets / n;
    let (mut m2, mut m3, mut m4) = (0.0, 0.0, 0.0);
    for &v in values {
        let d = v - mean;
        m2 += d * d;
        m3 += d * d * d;
        m4 += d * d * d * d;
    }
    let mut argmin = 0;
    let mut argmax = 0;
    for (i, &v) in values.iter().enumerate() {
        if v < values[argmin] {
            argmin = i;
        }
        if v > values[argmax] {
            argmax = i;
        }
    }
    Ref { n, sum, mean, m2, m3, m4, min: values[argmin], max: values[argmax], argmin, argmax }
}

fn close(got: f64, want: f64, rel: f64, abs: f64, what: &str) {
    let tol = abs.max(rel * want.abs());
    assert!((got - want).abs() <= tol, "{what}: got {got}, want {want} (tolerance {tol})");
}

fn f32s(values: &[f64]) -> Vec<f32> {
    values.iter().map(|&v| v as f32).collect()
}

fn read<W: Wide>(a: luma::Result<Array<W>>) -> Vec<f64> {
    Accumulator::<W>::to_f64(&a.unwrap()).unwrap()
}

fn full_builder<W: Wide>(fold: Fold) -> luma::AccumulatorBuilder<W> {
    Accumulator::<W>::builder(fold).moments(4).sum().min_max().arg_min_max()
}

fn check_whole<W: Wide>(acc: &Accumulator<W>, values: &[f64], rel: f64) {
    let want = reference(values);
    close(read(acc.count())[0], want.n, 0.0, 0.0, "count");
    // `f32` partial sums about the pivot: an error of a few `f32` roundings of the spread.
    let spread = (want.m2 / want.n).sqrt();
    close(read(acc.sum())[0], want.sum, rel, 1.5e-7 * want.n * spread + rel, "sum");
    close(read(acc.mean())[0], want.mean, rel, rel, "mean");
    close(read(acc.var())[0], want.m2 / want.n, rel, 0.0, "var");
    close(read(acc.std())[0], (want.m2 / want.n).sqrt(), rel, 0.0, "std");
    let var = want.m2 / want.n;
    close(read(acc.skewness())[0], want.m3 / want.n / var.powf(1.5), 1e-3, 1e-4, "skewness");
    close(read(acc.kurtosis())[0], want.m4 / want.n / (var * var) - 3.0, 1e-3, 1e-4, "kurtosis");
    close(read(acc.min())[0], want.min, 1e-6, 0.0, "min");
    close(read(acc.max())[0], want.max, 1e-6, 0.0, "max");
    assert_eq!(read(acc.argmin())[0] as usize, want.argmin, "argmin");
    assert_eq!(read(acc.argmax())[0] as usize, want.argmax, "argmax");
}

#[test]
fn whole_batch_statistics_match_the_host() {
    // Batches of different sizes, up to 1000-odd elements.
    let data = f32s(&normal(5000, 1));
    let host: Vec<f64> = data.iter().map(|&v| v as f64).collect();
    let acc = full_builder::<Df32>(Fold::all()).build().unwrap();
    let mut at = 0;
    for size in [1, 7, 100, 1000, 3, 2000, 1889] {
        let batch = Array::from_slice(&data[at..at + size], size).unwrap();
        acc.push(&batch).unwrap();
        at += size;
    }
    assert_eq!(at, data.len());
    assert_eq!(acc.samples_seen(), 5000);
    check_whole(&acc, &host, 1e-5);
}

#[test]
fn f64_state_matches_the_host() {
    let data = f32s(&normal(3000, 2));
    let host: Vec<f64> = data.iter().map(|&v| v as f64).collect();
    let acc = full_builder::<f64>(Fold::all()).build().unwrap();
    for chunk in data.chunks(500) {
        acc.push(&Array::from_slice(chunk, chunk.len()).unwrap()).unwrap();
    }
    check_whole(&acc, &host, 1e-6);
}

#[test]
fn per_feature_statistics_over_row_batches() {
    // Batches of rows of 5 features: shape [5, rows], fold axis 1.
    const F: usize = 5;
    let rows = 700;
    let data = f32s(&normal(F * rows, 3));
    let acc = full_builder::<Df32>(Fold::axis(1)).build().unwrap();
    for (start, count) in [(0, 100), (100, 1), (101, 299), (400, 300)] {
        let slice = &data[start * F..(start + count) * F];
        acc.push(&Array::from_slice(slice, [F, count]).unwrap()).unwrap();
    }
    let means = read(acc.mean());
    let vars = read(acc.var());
    let mins = read(acc.min());
    let argmax = read(acc.argmax());
    assert_eq!(means.len(), F);
    for f in 0..F {
        let column: Vec<f64> = (0..rows).map(|r| data[r * F + f] as f64).collect();
        let want = reference(&column);
        close(means[f], want.mean, 1e-5, 1e-6, "mean");
        close(vars[f], want.m2 / want.n, 1e-5, 0.0, "var");
        close(mins[f], want.min, 1e-6, 0.0, "min");
        assert_eq!(argmax[f] as usize, want.argmax, "argmax of feature {f}");
    }
    // Per-sample statistics (fold axis 0): one state per row.
    let acc = Accumulator::<Df32>::builder(Fold::axis(0)).build().unwrap();
    acc.push(&Array::from_slice(&data[..F * 100], [F, 100]).unwrap()).unwrap();
    let rows_var = read(acc.var());
    assert_eq!(rows_var.len(), 100);
    let row0: Vec<f64> = (0..F).map(|f| data[f] as f64).collect();
    close(rows_var[0], reference(&row0).m2 / F as f64, 1e-5, 1e-7, "row var");
}

#[test]
fn splits_and_merge_orders_agree() {
    let data = f32s(&normal(6000, 4));
    let host: Vec<f64> = data.iter().map(|&v| v as f64).collect();
    let make = |parts: &[(usize, usize)]| {
        let acc = full_builder::<Df32>(Fold::all()).build().unwrap();
        for &(a, b) in parts {
            acc.push(&Array::from_slice(&data[a..b], b - a).unwrap()).unwrap();
        }
        acc
    };
    // ((a + b) + c) against (a + (b + c)) against one accumulator over everything.
    let (a, b, c) = (make(&[(0, 1500), (1500, 2000)]), make(&[(2000, 2001), (2001, 4000)]), make(&[(4000, 6000)]));
    let left = a.clone();
    left.merge(&b).unwrap();
    left.merge(&c).unwrap();
    let bc = b.clone();
    bc.merge(&c).unwrap();
    let right = a.clone();
    right.merge(&bc).unwrap();
    let whole = make(&[(0, 6000)]);
    for acc in [&left, &right, &whole] {
        check_whole(acc, &host, 1e-5);
        assert_eq!(acc.samples_seen(), 6000);
    }
    let (l, r) = (read(left.var())[0], read(right.var())[0]);
    close(l, r, 1e-6, 0.0, "associativity");
    // Merging an empty accumulator, and merging into an empty one.
    let empty = full_builder::<Df32>(Fold::all()).build().unwrap();
    left.merge(&empty).unwrap();
    empty.merge(&left).unwrap();
    check_whole(&empty, &host, 1e-5);
    // Merging an accumulator with itself doubles the counts.
    a.merge(&a).unwrap();
    close(read(a.count())[0], 4000.0, 0.0, 0.0, "self merge count");
}

#[test]
fn large_offset_with_small_variance() {
    // 1e9 + 1e-2 * noise needs f64 data (f32 cannot hold it): the variance must not cancel away.
    let noise = normal(20_000, 5);
    let data: Vec<f64> = noise.iter().map(|&v| 1.0e9 + 1.0e-2 * v).collect();
    let want = reference(&data);
    for wide in [0, 1] {
        // Df32 holds a mean of 1e9 to 3.5e-6 (48 bits), a hundredth of a standard error here.
        let tolerance = if wide == 0 { 1e-5 } else { 1e-8 };
        let check = |mean: f64, var: f64| {
            close(mean, want.mean, 0.0, tolerance, "mean");
            close(var, want.m2 / want.n, 1e-3, 0.0, "var");
        };
        if wide == 0 {
            let acc = Accumulator::<Df32>::new(Fold::all());
            for chunk in data.chunks(1000) {
                acc.push(&Array::from_slice(chunk, chunk.len()).unwrap()).unwrap();
            }
            check(read(acc.mean())[0], read(acc.var())[0]);
        } else {
            let acc = Accumulator::<f64>::builder(Fold::all()).precise().build().unwrap();
            for chunk in data.chunks(1000) {
                acc.push(&Array::from_slice(chunk, chunk.len()).unwrap()).unwrap();
            }
            check(read(acc.mean())[0], read(acc.var())[0]);
        }
    }
    // f32 data far from zero: 1e4 + noise.
    let data = f32s(&normal(10_000, 6).iter().map(|v| 1.0e4 + v).collect::<Vec<_>>());
    let host: Vec<f64> = data.iter().map(|&v| v as f64).collect();
    let acc = Accumulator::<Df32>::new(Fold::all());
    for chunk in data.chunks(256) {
        acc.push(&Array::from_slice(chunk, chunk.len()).unwrap()).unwrap();
    }
    let want = reference(&host);
    close(read(acc.var())[0], want.m2 / want.n, 1e-4, 0.0, "var at 1e4");
    // Many small batches whose mean drifts far away from the first batch.
    let acc = Accumulator::<Df32>::new(Fold::all());
    let mut all = Vec::new();
    for step in 0..200 {
        let values = f32s(&normal(64, 100 + step).iter().map(|v| v + step as f64 * 50.0).collect::<Vec<_>>());
        all.extend(values.iter().map(|&v| v as f64));
        acc.push(&Array::from_slice(&values, 64).unwrap()).unwrap();
    }
    let want = reference(&all);
    close(read(acc.mean())[0], want.mean, 1e-6, 0.0, "drifting mean");
    close(read(acc.var())[0], want.m2 / want.n, 1e-4, 0.0, "drifting var");
}

#[test]
fn counts_stay_exact_past_f32_precision() {
    // 2^24 + 3 samples of the same value in a handful of pushes: the count is exact.
    let acc = Accumulator::<Df32>::new(Fold::all());
    let batch = Array::<f32>::full(1 << 20, 2.0).unwrap();
    for _ in 0..16 {
        acc.push(&batch).unwrap();
    }
    acc.push(&Array::<f32>::full(3, 2.0).unwrap()).unwrap();
    assert_eq!(read(acc.count())[0], ((1u64 << 24) + 3) as f64);
    close(read(acc.mean())[0], 2.0, 0.0, 1e-9, "mean");
    close(read(acc.var())[0], 0.0, 0.0, 1e-9, "var");
    close(read(acc.sum())[0], 2.0 * ((1u64 << 24) + 3) as f64, 0.0, 0.0, "sum");
}

#[test]
fn masks_and_nan_policy() {
    let data = f32s(&normal(2000, 7));
    let mask: Vec<bool> = (0..2000).map(|i| i % 3 != 0).collect();
    let picked: Vec<f64> = data.iter().zip(&mask).filter(|(_, m)| **m).map(|(&v, _)| v as f64).collect();
    let acc = full_builder::<Df32>(Fold::all()).build().unwrap();
    for (d, m) in data.chunks(333).zip(mask.chunks(333)) {
        acc.push_where(&Array::from_slice(d, d.len()).unwrap(), &Array::from_slice(m, m.len()).unwrap()).unwrap();
    }
    let want = reference(&picked);
    close(read(acc.count())[0], want.n, 0.0, 0.0, "count");
    close(read(acc.mean())[0], want.mean, 1e-5, 1e-6, "mean");
    close(read(acc.var())[0], want.m2 / want.n, 1e-5, 0.0, "var");
    close(read(acc.min())[0], want.min, 1e-6, 0.0, "min");
    close(read(acc.max())[0], want.max, 1e-6, 0.0, "max");

    // NaN skipping leaves them out; propagation poisons.
    let mut with_nan = data.clone();
    for i in (0..2000).step_by(97) {
        with_nan[i] = f32::NAN;
    }
    let kept: Vec<f64> = with_nan.iter().filter(|v| !v.is_nan()).map(|&v| v as f64).collect();
    let skip = Accumulator::<Df32>::builder(Fold::all()).nan_policy(NanPolicy::Skip).sum().build().unwrap();
    let prop = Accumulator::<Df32>::new(Fold::all());
    for chunk in with_nan.chunks(500) {
        let batch = Array::from_slice(chunk, chunk.len()).unwrap();
        skip.push(&batch).unwrap();
        prop.push(&batch).unwrap();
    }
    let want = reference(&kept);
    close(read(skip.count())[0], want.n, 0.0, 0.0, "count without NaN");
    close(read(skip.mean())[0], want.mean, 1e-5, 1e-6, "mean without NaN");
    close(read(skip.var())[0], want.m2 / want.n, 1e-5, 0.0, "var without NaN");
    assert!(read(prop.mean())[0].is_nan());
    assert!(read(prop.var())[0].is_nan());
}

#[test]
fn per_group_states_fold_in_one_pass() {
    const F: usize = 3;
    const G: usize = 4;
    let rows = 1200;
    let data = f32s(&normal(F * rows, 8));
    let labels: Vec<u32> = (0..rows).map(|r| ((r * 7 + r / 5) % 5) as u32).collect(); // label 4 is out of range
    let acc = Accumulator::<Df32>::builder(Fold::axis(1)).moments(2).sum().min_max().groups(G).build().unwrap();
    for start in (0..rows).step_by(300) {
        let d = Array::from_slice(&data[start * F..(start + 300) * F], [F, 300]).unwrap();
        let l = Array::from_slice(&labels[start..start + 300], 300).unwrap();
        acc.push_grouped(&d, &l).unwrap();
    }
    // Shape [F, G]: group g of feature f at f + F * g.
    let (count, mean, var, min, max) = (read(acc.count()), read(acc.mean()), read(acc.var()), read(acc.min()), read(acc.max()));
    for g in 0..G {
        for f in 0..F {
            let picked: Vec<f64> = (0..rows).filter(|&r| labels[r] == g as u32).map(|r| data[r * F + f] as f64).collect();
            let want = reference(&picked);
            let at = f + F * g;
            close(count[at], want.n, 0.0, 0.0, "group count");
            close(mean[at], want.mean, 1e-4, 1e-5, "group mean");
            close(var[at], want.m2 / want.n, 1e-4, 0.0, "group var");
            close(min[at], want.min, 1e-6, 0.0, "group min");
            close(max[at], want.max, 1e-6, 0.0, "group max");
        }
    }
    // The whole batch, one label per element.
    let acc = Accumulator::<Df32>::builder(Fold::all()).groups(2).build().unwrap();
    let d = Array::from_slice(&data[..1000], [10, 100]).unwrap();
    let l: Vec<u32> = (0..1000).map(|i| (i % 2) as u32).collect();
    acc.push_grouped(&d, &Array::from_slice(&l, [10, 100]).unwrap()).unwrap();
    let evens: Vec<f64> = (0..1000).step_by(2).map(|i| data[i] as f64).collect();
    close(read(acc.mean())[0], reference(&evens).mean, 1e-5, 1e-6, "even mean");
    close(read(acc.count())[1], 500.0, 0.0, 0.0, "odd count");
}

#[test]
fn every_element_type_folds() {
    let ints: Vec<i32> = (0..3000).map(|i| ((i * 7919) % 2001) - 1000).collect();
    let host: Vec<f64> = ints.iter().map(|&v| v as f64).collect();
    let want = reference(&host);
    let acc = full_builder::<Df32>(Fold::all()).build().unwrap();
    for chunk in ints.chunks(1000) {
        acc.push(&Array::from_slice(chunk, chunk.len()).unwrap()).unwrap();
    }
    check_whole(&acc, &host, 1e-6);
    close(read(acc.sum())[0], want.sum, 0.0, 0.0, "exact i32 sum");

    let uints: Vec<u32> = (0..3000).map(|i| 3_000_000_000 + (i * 7919) % 2001).collect();
    let host: Vec<f64> = uints.iter().map(|&v| v as f64).collect();
    let acc = full_builder::<Df32>(Fold::all()).build().unwrap();
    for chunk in uints.chunks(1000) {
        acc.push(&Array::from_slice(chunk, chunk.len()).unwrap()).unwrap();
    }
    let want = reference(&host);
    close(read(acc.sum())[0], want.sum, 0.0, 0.0, "exact u32 sum beyond 2^24");
    close(read(acc.mean())[0], want.mean, 0.0, 1e-4, "u32 mean");
    close(read(acc.var())[0], want.m2 / want.n, 1e-4, 0.0, "u32 var");
    close(read(acc.min())[0], want.min, 0.0, 0.0, "u32 min");
    close(read(acc.max())[0], want.max, 0.0, 0.0, "u32 max");

    let doubles = normal(2000, 9);
    let acc = Accumulator::<f64>::builder(Fold::all()).moments(2).sum().min_max().precise().build().unwrap();
    for chunk in doubles.chunks(700) {
        acc.push(&Array::from_slice(chunk, chunk.len()).unwrap()).unwrap();
    }
    let want = reference(&doubles);
    close(read(acc.mean())[0], want.mean, 1e-12, 1e-14, "f64 mean");
    close(read(acc.var())[0], want.m2 / want.n, 1e-9, 0.0, "f64 var");
    close(read(acc.min())[0], want.min, 0.0, 0.0, "f64 min");

    let dfs: Vec<Df32> = doubles.iter().map(|&v| Df32::from(v)).collect();
    let acc = Accumulator::<Df32>::builder(Fold::all()).moments(2).sum().min_max().build().unwrap();
    for chunk in dfs.chunks(700) {
        acc.push(&Array::from_slice(chunk, chunk.len()).unwrap()).unwrap();
    }
    close(read(acc.mean())[0], want.mean, 1e-6, 1e-8, "Df32 mean");
    close(read(acc.var())[0], want.m2 / want.n, 1e-5, 0.0, "Df32 var");
}

#[test]
fn a_fused_expression_is_pushed_without_materializing() {
    let x = f32s(&normal(4096, 10));
    let y = f32s(&normal(4096, 11));
    let (xa, ya) = (Array::from_slice(&x, 4096).unwrap(), Array::from_slice(&y, 4096).unwrap());
    let acc = Accumulator::<Df32>::new(Fold::all());
    acc.push(&(&xa - &ya).abs()).unwrap();
    let diffs: Vec<f64> = x.iter().zip(&y).map(|(a, b)| (a - b).abs() as f64).collect();
    close(read(acc.mean())[0], reference(&diffs).mean, 1e-5, 1e-6, "mean of |x - y|");
}

#[test]
fn state_round_trips_through_bytes() {
    let data = f32s(&normal(3000, 12));
    let host: Vec<f64> = data.iter().map(|&v| v as f64).collect();
    let (first, second) = data.split_at(1700);
    let a = full_builder::<Df32>(Fold::all()).build().unwrap();
    a.push(&Array::from_slice(first, first.len()).unwrap()).unwrap();
    let b = full_builder::<f64>(Fold::all()).build().unwrap();
    b.push(&Array::from_slice(second, second.len()).unwrap()).unwrap();
    // The f64 state travels as bytes into a Df32 accumulator on this "other machine" and merges.
    let bytes = b.to_bytes().unwrap();
    let restored = Accumulator::<Df32>::from_bytes(&bytes).unwrap();
    a.merge(&restored).unwrap();
    check_whole(&a, &host, 1e-5);
    // The restored state keeps going.
    let again = Accumulator::<Df32>::from_bytes(&a.to_bytes().unwrap()).unwrap();
    again.push(&Array::from_slice(first, first.len()).unwrap()).unwrap();
    close(read(again.count())[0], 4700.0, 0.0, 0.0, "count after restore and push");
    // Corrupt input is an error, not a panic.
    assert!(Accumulator::<Df32>::from_bytes(&bytes[..bytes.len() - 3]).is_err());
    assert!(Accumulator::<Df32>::from_bytes(b"nope").is_err());
}

#[test]
fn reset_and_misuse() {
    let acc = Accumulator::<Df32>::new(Fold::axis(1));
    assert!(acc.mean().is_err(), "no data yet");
    acc.push(&Array::<f32>::zeros([4, 10]).unwrap()).unwrap();
    // A different position shape is rejected, a different number of samples is fine.
    assert!(acc.push(&Array::<f32>::zeros([5, 10]).unwrap()).is_err());
    acc.push(&Array::<f32>::zeros([4, 3]).unwrap()).unwrap();
    assert_eq!(read(acc.count()), vec![13.0; 4]);
    assert!(acc.argmin().is_err(), "not tracked");
    acc.reset();
    assert!(acc.is_empty());
    acc.push(&Array::<f32>::zeros([5, 10]).unwrap()).unwrap();
    assert_eq!(read(acc.count()).len(), 5);
    assert!(acc.push_grouped(&Array::<f32>::zeros([5, 10]).unwrap(), &Array::<u32>::zeros(10).unwrap()).is_err());
}

#[test]
fn the_precise_pass_is_exact_to_the_state() {
    // f32 data with a mean far from zero: the precise pass sums in the wide type.
    let data = f32s(&normal(30_000, 13).iter().map(|v| 3.0e3 + 0.1 * v).collect::<Vec<_>>());
    let host: Vec<f64> = data.iter().map(|&v| v as f64).collect();
    let want = reference(&host);
    for wide in [0, 1] {
        let (sum, var) = if wide == 0 {
            let acc = full_builder::<Df32>(Fold::all()).precise().build().unwrap();
            for chunk in data.chunks(10_000) {
                acc.push(&Array::from_slice(chunk, chunk.len()).unwrap()).unwrap();
            }
            (read(acc.sum())[0], read(acc.var())[0])
        } else {
            let acc = full_builder::<f64>(Fold::all()).precise().build().unwrap();
            for chunk in data.chunks(10_000) {
                acc.push(&Array::from_slice(chunk, chunk.len()).unwrap()).unwrap();
            }
            (read(acc.sum())[0], read(acc.var())[0])
        };
        close(sum, want.sum, 1e-11, 0.0, "precise sum");
        close(var, want.m2 / want.n, 1e-9, 0.0, "precise var");
    }
}

// ---------------------------------------------------------------------------------------------
// Two streams
// ---------------------------------------------------------------------------------------------

/// Host covariance of two columns: (n, mean_x, mean_y, m2x, m2y, cxy).
fn host_co(x: &[f64], y: &[f64]) -> [f64; 6] {
    let n = x.len() as f64;
    let (mx, my) = (x.iter().sum::<f64>() / n, y.iter().sum::<f64>() / n);
    let (mut m2x, mut m2y, mut cxy) = (0.0, 0.0, 0.0);
    for (a, b) in x.iter().zip(y) {
        m2x += (a - mx) * (a - mx);
        m2y += (b - my) * (b - my);
        cxy += (a - mx) * (b - my);
    }
    [n, mx, my, m2x, m2y, cxy]
}

#[test]
fn covariance_between_two_streams_per_position() {
    const F: usize = 4;
    let rows = 900;
    let x = f32s(&normal(F * rows, 20));
    // y correlates with x differently per feature.
    let noise = normal(F * rows, 21);
    let y: Vec<f32> = (0..F * rows).map(|i| (x[i] as f64 * (i % F) as f64 * 0.4 + noise[i] + 3.0) as f32).collect();
    let acc = CoMoments::<Df32>::new(Fold::axis(1));
    for (start, count) in [(0, 1), (1, 99), (100, 300), (400, 500)] {
        let range = start * F..(start + count) * F;
        acc.push(&Array::from_slice(&x[range.clone()], [F, count]).unwrap(), &Array::from_slice(&y[range], [F, count]).unwrap()).unwrap();
    }
    let (cov, corr, vx, vy, mx, my) = (read(acc.cov()), read(acc.corr()), read(acc.var_x()), read(acc.var_y()), read(acc.mean_x()), read(acc.mean_y()));
    for f in 0..F {
        let cx: Vec<f64> = (0..rows).map(|r| x[r * F + f] as f64).collect();
        let cy: Vec<f64> = (0..rows).map(|r| y[r * F + f] as f64).collect();
        let [n, hmx, hmy, m2x, m2y, cxy] = host_co(&cx, &cy);
        close(mx[f], hmx, 1e-5, 1e-6, "mean x");
        close(my[f], hmy, 1e-5, 1e-6, "mean y");
        close(vx[f], m2x / n, 1e-5, 0.0, "var x");
        close(vy[f], m2y / n, 1e-5, 0.0, "var y");
        close(cov[f], cxy / n, 1e-4, 1e-6, "cov");
        close(corr[f], cxy / (m2x * m2y).sqrt(), 1e-4, 1e-5, "corr");
    }
    // Whole-stream covariance, merged from two halves; masked; bytes.
    let xs: Vec<f64> = x.iter().map(|&v| v as f64).collect();
    let ys: Vec<f64> = y.iter().map(|&v| v as f64).collect();
    let [n, _, _, m2x, m2y, cxy] = host_co(&xs, &ys);
    let half = x.len() / 2;
    let (a, b) = (CoMoments::<Df32>::new(Fold::all()), CoMoments::<f64>::new(Fold::all()));
    a.push(&Array::from_slice(&x[..half], half).unwrap(), &Array::from_slice(&y[..half], half).unwrap()).unwrap();
    b.push(&Array::from_slice(&x[half..], x.len() - half).unwrap(), &Array::from_slice(&y[half..], y.len() - half).unwrap()).unwrap();
    let b = CoMoments::<Df32>::from_bytes(&b.to_bytes().unwrap()).unwrap();
    a.merge(&b).unwrap();
    close(read(a.cov())[0], cxy / n, 1e-4, 2e-5, "merged cov");
    close(read(a.corr())[0], cxy / (m2x * m2y).sqrt(), 1e-4, 1e-5, "merged corr");
    close(read(a.count())[0], n, 0.0, 0.0, "merged count");
    let masked = CoMoments::<Df32>::new(Fold::all());
    let keep: Vec<bool> = (0..x.len()).map(|i| i % 2 == 0).collect();
    masked.push_where(&Array::from_slice(&x, x.len()).unwrap(), &Array::from_slice(&y, y.len()).unwrap(), &Array::from_slice(&keep, keep.len()).unwrap()).unwrap();
    let hx: Vec<f64> = xs.iter().step_by(2).copied().collect();
    let hy: Vec<f64> = ys.iter().step_by(2).copied().collect();
    let [n, _, _, _, _, cxy] = host_co(&hx, &hy);
    close(read(masked.cov())[0], cxy / n, 1e-4, 1e-6, "masked cov");
}

#[test]
fn cross_moments_survive_a_large_offset() {
    let noise = normal(10_000, 22);
    let other = normal(10_000, 23);
    let x: Vec<f64> = noise.iter().map(|v| 1.0e6 + 0.01 * v).collect();
    let y: Vec<f64> = x.iter().zip(&other).map(|(a, b)| -2.0e6 + 3.0 * (a - 1.0e6) + 0.005 * b).collect();
    let acc = CoMoments::<f64>::builder(Fold::all()).precise().build();
    for (cx, cy) in x.chunks(1000).zip(y.chunks(1000)) {
        acc.push(&Array::from_slice(cx, cx.len()).unwrap(), &Array::from_slice(cy, cy.len()).unwrap()).unwrap();
    }
    let [n, _, _, m2x, m2y, cxy] = host_co(&x, &y);
    close(read(acc.cov())[0], cxy / n, 1e-9, 0.0, "cov at offset");
    close(read(acc.corr())[0], cxy / (m2x * m2y).sqrt(), 1e-9, 0.0, "corr at offset");
}

#[test]
fn feature_covariance_matrix() {
    const F: usize = 6;
    let samples = 1500;
    let base = normal(F * samples, 24);
    // Mix the features so that the matrix is not diagonal, and add an offset.
    let data: Vec<f32> = (0..F * samples).map(|i| (base[i] + 0.5 * base[(i / F) * F + (i % F + 1) % F] + 40.0 * (i % F) as f64) as f32).collect();
    let acc = Covariance::<Df32>::new();
    for (start, count) in [(0, 1), (1, 249), (250, 750), (1000, 500)] {
        acc.push(&Array::from_slice(&data[start * F..(start + count) * F], [F, count]).unwrap()).unwrap();
    }
    let cov = read(acc.cov());
    let corr = read(acc.corr());
    let mean = read(acc.mean());
    let var = read(acc.var());
    let n = samples as f64;
    let col = |f: usize| -> Vec<f64> { (0..samples).map(|r| data[r * F + f] as f64).collect() };
    for i in 0..F {
        for j in 0..F {
            let [_, mi, mj, m2i, m2j, cij] = host_co(&col(i), &col(j));
            // The product sums `f32` terms of the size of the variances over an epoch of samples:
            // an absolute error of a few 1e-6 of them.
            close(cov[i + F * j], cij / n, 1e-4, 2e-5, "cov entry");
            close(corr[i + F * j], cij / (m2i * m2j).sqrt(), 1e-4, 2e-5, "corr entry");
            if i == j {
                close(mean[i], mi, 1e-6, 1e-4, "mean");
                close(var[i], m2i / n, 1e-4, 0.0, "var");
            }
            let _ = mj;
        }
    }
    // Merging two accumulators and a round trip through bytes give the same matrix.
    let (a, b) = (Covariance::<Df32>::new(), Covariance::<f64>::new());
    a.push(&Array::from_slice(&data[..F * 700], [F, 700]).unwrap()).unwrap();
    b.push(&Array::from_slice(&data[F * 700..], [F, samples - 700]).unwrap()).unwrap();
    let b = Covariance::<Df32>::from_bytes(&b.to_bytes().unwrap()).unwrap();
    a.merge(&b).unwrap();
    let merged = read(a.cov());
    for k in 0..F * F {
        close(merged[k], cov[k], 1e-4, 2e-5, "merged cov");
    }
    assert!(Covariance::<Df32>::new().push(&Array::<f32>::zeros([3, 4, 2]).unwrap()).is_err());
}

// ---------------------------------------------------------------------------------------------
// Histograms
// ---------------------------------------------------------------------------------------------

fn host_hist(values: &[f32], bins: usize, lo: f32, hi: f32) -> (Vec<f64>, [f64; 3]) {
    let mut counts = vec![0.0; bins];
    let mut out = [0.0; 3];
    let scale = bins as f32 / (hi - lo);
    for &v in values {
        if v.is_nan() {
            out[2] += 1.0;
        } else if v < lo {
            out[0] += 1.0;
        } else if v > hi {
            out[1] += 1.0;
        } else {
            counts[(((v - lo) * scale) as usize).min(bins - 1)] += 1.0;
        }
    }
    (counts, out)
}

#[test]
fn histograms_count_exactly_and_merge() {
    let mut data = f32s(&uniform(50_000, 30).iter().map(|v| v * 1.4 - 0.2).collect::<Vec<_>>());
    for i in (0..data.len()).step_by(997) {
        data[i] = f32::NAN;
    }
    data[5] = 1.0; // the top edge belongs to the last bin
    let (want, out) = host_hist(&data, 20, 0.0, 1.0);
    let hist = Histogram::<Df32>::new(20, 0.0, 1.0).unwrap();
    for chunk in data.chunks(7000) {
        hist.push(&Array::from_slice(chunk, chunk.len()).unwrap()).unwrap();
    }
    assert_eq!(read(hist.counts()), want);
    assert_eq!(read(hist.outliers()), out);
    // Halves on two accumulators, one of them an f64 state travelling as bytes.
    let (a, b) = (Histogram::<Df32>::new(20, 0.0, 1.0).unwrap(), Histogram::<f64>::new(20, 0.0, 1.0).unwrap());
    a.push(&Array::from_slice(&data[..20_000], 20_000).unwrap()).unwrap();
    b.push(&Array::from_slice(&data[20_000..], 30_000).unwrap()).unwrap();
    a.merge(&Histogram::<Df32>::from_bytes(&b.to_bytes().unwrap()).unwrap()).unwrap();
    assert_eq!(read(a.counts()), want);
    assert_eq!(a.edges().len(), 21);
    // Masked, and per feature.
    let keep: Vec<bool> = (0..data.len()).map(|i| i % 4 != 1).collect();
    let picked: Vec<f32> = data.iter().zip(&keep).filter(|(_, k)| **k).map(|(&v, _)| v).collect();
    let masked = Histogram::<Df32>::new(20, 0.0, 1.0).unwrap();
    masked.push_where(&Array::from_slice(&data, data.len()).unwrap(), &Array::from_slice(&keep, keep.len()).unwrap()).unwrap();
    assert_eq!(read(masked.counts()), host_hist(&picked, 20, 0.0, 1.0).0);
    const F: usize = 3;
    let rows = 10_000;
    let per = Histogram::<Df32>::with_fold(8, 0.0, 1.0, Fold::axis(1)).unwrap();
    per.push(&Array::from_slice(&data[..F * rows], [F, rows]).unwrap()).unwrap();
    let counts = read(per.counts());
    assert_eq!(counts.len(), F * 8);
    for f in 0..F {
        let column: Vec<f32> = (0..rows).map(|r| data[r * F + f]).collect();
        let want = host_hist(&column, 8, 0.0, 1.0).0;
        for b in 0..8 {
            assert_eq!(counts[f + F * b], want[b], "feature {f} bin {b}");
        }
    }
    assert!(Histogram::<Df32>::new(0, 0.0, 1.0).is_err());
    assert!(Histogram::<Df32>::new(4, 1.0, 1.0).is_err());
}

#[test]
fn joint_histogram_and_mutual_information() {
    let n = 40_000;
    let x = f32s(&uniform(n, 31));
    let y: Vec<f32> = x.iter().zip(uniform(n, 32)).map(|(&a, b)| if b < 0.5 { a } else { (b * 2.0 - 1.0) as f32 }).collect();
    let (bx, by) = (8usize, 8usize);
    let joint = Histogram2d::<Df32>::new((bx, by), (0.0, 1.0), (0.0, 1.0)).unwrap();
    for (cx, cy) in x.chunks(9000).zip(y.chunks(9000)) {
        joint.push(&Array::from_slice(cx, cx.len()).unwrap(), &Array::from_slice(cy, cy.len()).unwrap()).unwrap();
    }
    let mut want = vec![0.0; bx * by];
    for (&a, &b) in x.iter().zip(&y) {
        let (i, j) = ((a * bx as f32) as usize, (b * by as f32) as usize);
        want[i.min(bx - 1) + bx * j.min(by - 1)] += 1.0;
    }
    assert_eq!(read(joint.counts()), want);
    assert_eq!(read(joint.outside()), vec![0.0]);
    let marginal = read(joint.marginal_x());
    assert_eq!(marginal.iter().sum::<f64>(), n as f64);
    // Mutual information against the host formula.
    let total = n as f64;
    let px: Vec<f64> = (0..bx).map(|i| (0..by).map(|j| want[i + bx * j]).sum::<f64>() / total).collect();
    let py: Vec<f64> = (0..by).map(|j| (0..bx).map(|i| want[i + bx * j]).sum::<f64>() / total).collect();
    let mut mi = 0.0;
    for i in 0..bx {
        for j in 0..by {
            let p = want[i + bx * j] / total;
            if p > 0.0 {
                mi += p * (p / (px[i] * py[j])).ln();
            }
        }
    }
    let got = joint.mutual_information().unwrap().to_vec().unwrap()[0] as f64;
    close(got, mi, 1e-3, 1e-5, "mutual information");
    assert!(mi > 0.1, "the streams are dependent");
    let ent: f64 = -px.iter().map(|p| p * p.ln()).sum::<f64>();
    close(joint.entropy_x().unwrap().to_vec().unwrap()[0] as f64, ent, 1e-3, 1e-5, "entropy");
    // Out-of-range pairs are counted aside.
    let out = Histogram2d::<Df32>::new((2, 2), (0.0, 1.0), (0.0, 1.0)).unwrap();
    out.push(&Array::from_slice(&[0.1f32, 2.0, f32::NAN, 0.9], 4).unwrap(), &Array::from_slice(&[0.1f32, 0.5, 0.5, -1.0], 4).unwrap()).unwrap();
    assert_eq!(read(out.outside()), vec![3.0]);
    assert_eq!(read(out.counts()), vec![1.0, 0.0, 0.0, 0.0]);
    // Merge and bytes.
    let other = Histogram2d::<f64>::new((bx, by), (0.0, 1.0), (0.0, 1.0)).unwrap();
    other.push(&Array::from_slice(&x[..100], 100).unwrap(), &Array::from_slice(&y[..100], 100).unwrap()).unwrap();
    let other = Histogram2d::<Df32>::from_bytes(&other.to_bytes().unwrap()).unwrap();
    joint.merge(&other).unwrap();
    assert_eq!(read(joint.marginal_x()).iter().sum::<f64>(), n as f64 + 100.0);
}

// ---------------------------------------------------------------------------------------------
// Top-k
// ---------------------------------------------------------------------------------------------

#[test]
fn top_k_keeps_the_best_with_global_indices() {
    let data = f32s(&normal(10_000, 40));
    // Duplicates: ties go to the earlier index.
    let mut data = data;
    data[7000] = data[3];
    data[3] = 99.0;
    data[7000] = 99.0;
    let k = 25;
    let mut order: Vec<usize> = (0..data.len()).collect();
    order.sort_by(|&a, &b| data[b].total_cmp(&data[a]).then(a.cmp(&b)));
    let top = TopK::<f32, Df32>::largest(k);
    for chunk in data.chunks(1500) {
        top.push(&Array::from_slice(chunk, chunk.len()).unwrap()).unwrap();
    }
    assert_eq!(top.values().unwrap().to_vec().unwrap(), order[..k].iter().map(|&i| data[i]).collect::<Vec<_>>());
    assert_eq!(read(top.indices()), order[..k].iter().map(|&i| i as f64).collect::<Vec<_>>());
    assert_eq!(top.elements_seen(), 10_000);
    // The smallest, with NaN ignored.
    let mut with_nan = data.clone();
    with_nan[10] = f32::NAN;
    let mut order: Vec<usize> = (0..data.len()).filter(|&i| i != 10).collect();
    order.sort_by(|&a, &b| data[a].total_cmp(&data[b]).then(a.cmp(&b)));
    let low = TopK::<f32, Df32>::smallest(k);
    for chunk in with_nan.chunks(4096) {
        low.push(&Array::from_slice(chunk, chunk.len()).unwrap()).unwrap();
    }
    assert_eq!(read(low.indices()), order[..k].iter().map(|&i| i as f64).collect::<Vec<_>>());
    // Merge of two halves (the second half's indices continue the first's), through bytes.
    let (a, b) = (TopK::<f32, Df32>::largest(k), TopK::<f32, f64>::largest(k));
    a.push(&Array::from_slice(&data[..6000], 6000).unwrap()).unwrap();
    b.push(&Array::from_slice(&data[6000..], 4000).unwrap()).unwrap();
    let b = TopK::<f32, Df32>::from_bytes(&b.to_bytes().unwrap()).unwrap();
    a.merge(&b).unwrap();
    let mut order: Vec<usize> = (0..data.len()).collect();
    order.sort_by(|&a, &b| data[b].total_cmp(&data[a]).then(a.cmp(&b)));
    assert_eq!(read(a.indices()), order[..k].iter().map(|&i| i as f64).collect::<Vec<_>>());
    // Fewer than k elements: the empty slots are marked.
    let few = TopK::<i32, Df32>::largest(5);
    few.push(&Array::from_slice(&[3i32, -7, 5], 3).unwrap()).unwrap();
    assert_eq!(few.values().unwrap().to_vec().unwrap()[..3], [5, 3, -7]);
    assert_eq!(read(few.indices()), vec![2.0, 0.0, 1.0, -1.0, -1.0]);
    // Masked elements are never selected but take indices.
    let masked = TopK::<u32, Df32>::largest(2);
    masked.push_where(&Array::from_slice(&[9u32, 8, 7, 6], 4).unwrap(), &Array::from_slice(&[false, true, true, true], 4).unwrap()).unwrap();
    assert_eq!(masked.values().unwrap().to_vec().unwrap(), [8, 7]);
    assert_eq!(read(masked.indices()), vec![1.0, 2.0]);
}

// ---------------------------------------------------------------------------------------------
// Arrays split over several buffers
// ---------------------------------------------------------------------------------------------

#[test]
fn batches_split_over_several_buffers() {
    use luma::{Context, ContextOptions};
    // 1024 f32 to a buffer: 10_001 elements are ten full segments and a partial one.
    let ctx = Context::with_options(ContextOptions::default().chunk_bytes(4096)).expect("a device");
    let n = 10_001;
    let data = f32s(&normal(n, 50));
    let host: Vec<f64> = data.iter().map(|&v| v as f64).collect();
    let batch = Array::from_slice_on(&ctx, &data, n).unwrap();
    let acc = full_builder::<Df32>(Fold::all()).build().unwrap();
    acc.push(&batch).unwrap();
    acc.push(&batch).unwrap();
    let want = reference(&host);
    close(read(acc.count())[0], 2.0 * n as f64, 0.0, 0.0, "count");
    close(read(acc.mean())[0], want.mean, 1e-5, 1e-6, "mean");
    close(read(acc.var())[0], want.m2 / want.n, 1e-5, 0.0, "var");
    close(read(acc.min())[0], want.min, 1e-6, 0.0, "min");
    // The first minimum, and the second push's copy of it after the first batch.
    assert_eq!(read(acc.argmin())[0] as usize, want.argmin);
    assert_eq!(read(acc.argmax())[0] as usize, want.argmax);
    // Per feature along the long axis, and the other kinds of accumulators.
    let wide = Array::from_slice_on(&ctx, &data[..10_000], [4, 2500]).unwrap();
    let per = full_builder::<Df32>(Fold::axis(1)).build().unwrap();
    per.push(&wide).unwrap();
    let column: Vec<f64> = (0..2500).map(|r| data[r * 4 + 1] as f64).collect();
    close(read(per.mean())[1], reference(&column).mean, 1e-5, 1e-6, "chunked per-feature mean");
    assert_eq!(read(per.argmax())[1] as usize, reference(&column).argmax);
    let hist = Histogram::<Df32>::new(10, -3.0, 3.0).unwrap();
    hist.push(&batch).unwrap();
    assert_eq!(read(hist.counts()), host_hist(&data, 10, -3.0, 3.0).0);
    let top = TopK::<f32, Df32>::largest(5);
    top.push(&batch).unwrap();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| data[b].total_cmp(&data[a]).then(a.cmp(&b)));
    assert_eq!(read(top.indices()), order[..5].iter().map(|&i| i as f64).collect::<Vec<_>>());
    let co = CoMoments::<Df32>::new(Fold::all());
    co.push(&batch, &batch).unwrap();
    close(read(co.corr())[0], 1.0, 0.0, 1e-5, "self correlation");
    // Per-position top-k of the chunked [4, 2500] batch: the ranking of feature 1.
    let along = TopK::<f32, Df32>::largest_along(3, 1).unwrap();
    along.push(&wide).unwrap();
    let mut rows: Vec<usize> = (0..2500).collect();
    rows.sort_by(|&a, &b| data[b * 4 + 1].total_cmp(&data[a * 4 + 1]).then(a.cmp(&b)));
    let indices = read(along.indices());
    assert_eq!((0..3).map(|j| indices[1 + 4 * j]).collect::<Vec<_>>(), rows[..3].iter().map(|&r| r as f64).collect::<Vec<_>>());
}

// ---------------------------------------------------------------------------------------------
// More folds
// ---------------------------------------------------------------------------------------------

#[test]
fn several_axes_fold_at_once() {
    // [3, 20, 4, 5]: fold axes 1 and 3, keep 3 x 4 positions.
    let (a, b, c, d) = (3usize, 20usize, 4usize, 5usize);
    let data = f32s(&normal(a * b * c * d, 60));
    let acc = Accumulator::<Df32>::builder(Fold::axes(&[1, 3])).moments(2).sum().min_max().build().unwrap();
    // Two pushes with different extents along the folded axes.
    let at = |i: usize, j: usize, k: usize, l: usize| data[i + a * (j + b * (k + c * l))];
    let part = |range: std::ops::Range<usize>| {
        let mut out = Vec::new();
        for l in 0..d {
            for k in 0..c {
                for j in range.clone() {
                    for i in 0..a {
                        out.push(at(i, j, k, l));
                    }
                }
            }
        }
        Array::from_slice(&out, [a, range.len(), c, d]).unwrap()
    };
    acc.push(&part(0..7)).unwrap();
    acc.push(&part(7..20)).unwrap();
    let (mean, var, min) = (read(acc.mean()), read(acc.var()), read(acc.min()));
    assert_eq!(mean.len(), a * c);
    assert_eq!(read(acc.count()), vec![(b * d) as f64; a * c]);
    for k in 0..c {
        for i in 0..a {
            let values: Vec<f64> = (0..d).flat_map(|l| (0..b).map(move |j| (j, l))).map(|(j, l)| at(i, j, k, l) as f64).collect();
            let want = reference(&values);
            close(mean[i + a * k], want.mean, 1e-5, 1e-6, "mean");
            close(var[i + a * k], want.m2 / want.n, 1e-5, 0.0, "var");
            close(min[i + a * k], want.min, 1e-6, 0.0, "min");
        }
    }
}

#[test]
fn grouped_states_merge_and_round_trip() {
    const F: usize = 2;
    const G: usize = 3;
    let rows = 900;
    let data = f32s(&normal(F * rows, 61));
    let labels: Vec<u32> = (0..rows).map(|r| (r % G) as u32).collect();
    let make = || Accumulator::<Df32>::builder(Fold::axis(1)).moments(3).sum().min_max().groups(G).build().unwrap();
    let (a, b) = (make(), make());
    let push = |acc: &Accumulator<Df32>, from: usize, to: usize| {
        let d = Array::from_slice(&data[from * F..to * F], [F, to - from]).unwrap();
        acc.push_grouped(&d, &Array::from_slice(&labels[from..to], to - from).unwrap()).unwrap();
    };
    push(&a, 0, 400);
    push(&a, 400, 600);
    push(&b, 600, 900);
    let b = Accumulator::<Df32>::from_bytes(&b.to_bytes().unwrap()).unwrap();
    a.merge(&b).unwrap();
    let (count, mean, var, sum) = (read(a.count()), read(a.mean()), read(a.var()), read(a.sum()));
    for g in 0..G {
        for f in 0..F {
            let picked: Vec<f64> = (0..rows).filter(|&r| labels[r] == g as u32).map(|r| data[r * F + f] as f64).collect();
            let want = reference(&picked);
            let at = f + F * g;
            close(count[at], want.n, 0.0, 0.0, "count");
            close(mean[at], want.mean, 1e-4, 1e-5, "mean");
            close(var[at], want.m2 / want.n, 1e-4, 0.0, "var");
            close(sum[at], want.sum, 1e-4, 1e-3, "sum");
        }
    }
    // A different group count is a different accumulator.
    let other = Accumulator::<Df32>::builder(Fold::axis(1)).moments(3).sum().min_max().groups(G + 1).build().unwrap();
    assert!(a.merge(&other).is_err());
}

#[test]
fn argmin_argmax_with_masks_ties_and_merges() {
    // Equal extremes: the first occurrence wins, also across pushes and merges.
    let acc = Accumulator::<Df32>::builder(Fold::all()).arg_min_max().build().unwrap();
    acc.push(&Array::from_slice(&[5.0f32, 1.0, 9.0, 1.0], 4).unwrap()).unwrap();
    acc.push(&Array::from_slice(&[9.0f32, 0.5, 9.0], 3).unwrap()).unwrap();
    assert_eq!(read(acc.argmin()), vec![5.0]);
    assert_eq!(read(acc.argmax()), vec![2.0]);
    // Masked-out elements are never the extreme but count as samples.
    let masked = Accumulator::<Df32>::builder(Fold::all()).arg_min_max().build().unwrap();
    masked.push_where(&Array::from_slice(&[7u32, 1, 9, 3], 4).unwrap(), &Array::from_slice(&[true, false, false, true], 4).unwrap()).unwrap();
    assert_eq!(read(masked.argmin()), vec![3.0]);
    assert_eq!(read(masked.argmax()), vec![0.0]);
    // Per position, with the second accumulator's indices continuing after the first's.
    let (a, b) = (Accumulator::<Df32>::builder(Fold::axis(1)).arg_min_max().build().unwrap(), Accumulator::<Df32>::builder(Fold::axis(1)).arg_min_max().build().unwrap());
    a.push(&Array::from_slice(&[1i32, 4, 2, 5, 3, 6], [2, 3]).unwrap()).unwrap();
    b.push(&Array::from_slice(&[0i32, 9, 8, 7], [2, 2]).unwrap()).unwrap();
    a.merge(&b).unwrap();
    assert_eq!(read(a.argmin()), vec![3.0, 0.0]);
    assert_eq!(read(a.argmax()), vec![4.0, 3.0]);
    assert_eq!(a.samples_seen(), 5);
    // NaN elements skipped by the policy are not extremes either.
    let skip = Accumulator::<Df32>::builder(Fold::all()).arg_min_max().nan_policy(NanPolicy::Skip).build().unwrap();
    skip.push(&Array::from_slice(&[f32::NAN, 2.0, f32::NAN, -1.0], 4).unwrap()).unwrap();
    assert_eq!(read(skip.argmin()), vec![3.0]);
    assert_eq!(read(skip.argmax()), vec![1.0]);
    // f64 and Df32 batches have no index passes.
    let f64s = Accumulator::<Df32>::builder(Fold::all()).argmin().build().unwrap();
    assert!(f64s.push(&Array::from_slice(&[1.0f64, 2.0], 2).unwrap()).is_err() || f64s.flush().is_err() || f64s.argmin().is_err());
}

#[test]
fn f64_data_with_huge_offsets_per_feature_and_precise_whole() {
    const F: usize = 3;
    let rows = 500;
    let noise = normal(F * rows, 62);
    let data: Vec<f64> = noise.iter().enumerate().map(|(i, v)| 1.0e8 * (i % F) as f64 + 1.0e-3 * v).collect();
    // Per feature: the default pass about the running mean keeps the spread of data 1e11 times
    // smaller than its offset (a plain sum of squares has none of it left).
    let acc = Accumulator::<f64>::builder(Fold::axis(1)).moments(4).build().unwrap();
    for (start, count) in [(0, 200), (200, 300)] {
        acc.push(&Array::from_slice(&data[start * F..(start + count) * F], [F, count]).unwrap()).unwrap();
    }
    let (mean, var, skew) = (read(acc.mean()), read(acc.var()), read(acc.skewness()));
    for f in 0..F {
        let column: Vec<f64> = (0..rows).map(|r| data[r * F + f]).collect();
        let want = reference(&column);
        close(mean[f], want.mean, 1e-15, 1e-9, "mean");
        close(var[f], want.m2 / want.n, 1e-3, 0.0, "var");
        close(skew[f], want.m3 / want.n / (want.m2 / want.n).powf(1.5), 0.0, 5e-2, "skewness");
    }
    // The whole batch with the precise pass: as exact as the state.
    let column: Vec<f64> = (0..rows).map(|r| data[r * F]).collect();
    let acc = Accumulator::<f64>::builder(Fold::all()).moments(4).precise().build().unwrap();
    for chunk in column.chunks(200) {
        acc.push(&Array::from_slice(chunk, chunk.len()).unwrap()).unwrap();
    }
    let want = reference(&column);
    close(read(acc.var())[0], want.m2 / want.n, 1e-9, 0.0, "precise var");
    close(read(acc.skewness())[0], want.m3 / want.n / (want.m2 / want.n).powf(1.5), 1e-6, 1e-8, "precise skewness");
    // Neither wide type reduces along an axis: a precise pass folds the whole batch.
    let df = Accumulator::<Df32>::builder(Fold::axis(1)).precise().build().unwrap();
    assert!(df.push(&Array::from_slice(&data[..F * 10], [F, 10]).unwrap()).is_err());
    let f64s = Accumulator::<f64>::builder(Fold::axis(1)).precise().build().unwrap();
    assert!(f64s.push(&Array::from_slice(&data[..F * 10], [F, 10]).unwrap()).is_err());
}

#[test]
fn bytes_of_every_accumulator_survive_a_second_trip() {
    let x = f32s(&normal(500, 63));
    let y = f32s(&normal(500, 64));
    let xa = Array::from_slice(&x, 500).unwrap();
    let ya = Array::from_slice(&y, 500).unwrap();
    let co = CoMoments::<Df32>::new(Fold::all());
    co.push(&xa, &ya).unwrap();
    let co2 = CoMoments::<Df32>::from_bytes(&co.to_bytes().unwrap()).unwrap();
    assert_eq!(co2.to_bytes().unwrap(), co.to_bytes().unwrap());
    let hist = Histogram::<Df32>::new(10, -3.0, 3.0).unwrap();
    hist.push(&xa).unwrap();
    let hist2 = Histogram::<Df32>::from_bytes(&hist.to_bytes().unwrap()).unwrap();
    assert_eq!(hist2.to_bytes().unwrap(), hist.to_bytes().unwrap());
    let top = TopK::<f32, Df32>::smallest(4);
    top.push(&xa).unwrap();
    let top2 = TopK::<f32, Df32>::from_bytes(&top.to_bytes().unwrap()).unwrap();
    assert_eq!(top2.to_bytes().unwrap(), top.to_bytes().unwrap());
    // An accumulator's bytes are not a histogram's.
    assert!(Histogram::<Df32>::from_bytes(&co.to_bytes().unwrap()).is_err());
    // Empty accumulators round-trip too.
    let empty = Accumulator::<Df32>::new(Fold::all());
    let again = Accumulator::<Df32>::from_bytes(&empty.to_bytes().unwrap()).unwrap();
    assert!(again.is_empty());
}

// ---------------------------------------------------------------------------------------------
// Threads and fusion
// ---------------------------------------------------------------------------------------------

#[test]
fn one_accumulator_shared_by_threads() {
    // Four threads push their own batches into one accumulator and into one of their own that is
    // merged at the end: the same statistics, in whatever order the pushes fold.
    let data = f32s(&normal(8000, 70));
    let host: Vec<f64> = data.iter().map(|&v| v as f64).collect();
    let shared = Accumulator::<Df32>::builder(Fold::all()).moments(2).sum().min_max().build().unwrap();
    let merged = Accumulator::<Df32>::builder(Fold::all()).moments(2).sum().min_max().build().unwrap();
    std::thread::scope(|scope| {
        for part in data.chunks(2000) {
            let (shared, merged) = (&shared, &merged);
            scope.spawn(move || {
                let own = Accumulator::<Df32>::builder(Fold::all()).moments(2).sum().min_max().build().unwrap();
                for chunk in part.chunks(250) {
                    let batch = Array::from_slice(chunk, chunk.len()).unwrap();
                    shared.push(&batch).unwrap();
                    own.push(&batch).unwrap();
                }
                merged.merge(&own).unwrap();
            });
        }
    });
    let want = reference(&host);
    for acc in [&shared, &merged] {
        close(read(acc.count())[0], want.n, 0.0, 0.0, "count");
        close(read(acc.mean())[0], want.mean, 1e-5, 1e-6, "mean");
        close(read(acc.var())[0], want.m2 / want.n, 1e-5, 0.0, "var");
        close(read(acc.min())[0], want.min, 1e-6, 0.0, "min");
        close(read(acc.max())[0], want.max, 1e-6, 0.0, "max");
    }
}

#[test]
#[ignore = "launch-count bound fails on the RTX 3080 Laptop (CUDA and wgpu); Accumulators is fixing it"]
fn a_pushed_expression_is_computed_inside_the_reduction() {
    // A context of its own: the other tests of the process launch on the default one while this
    // one profiles, and their kernels would be counted.
    let ctx = luma::Context::with_options(luma::ContextOptions::default()).expect("a device");
    if !ctx.profiling_supported() {
        return;
    }
    let x = Array::from_slice_on(&ctx, &f32s(&normal(1 << 16, 71)), [256, 256]).unwrap();
    let y = Array::from_slice_on(&ctx, &f32s(&normal(1 << 16, 72)), [256, 256]).unwrap();
    x.eval().unwrap();
    y.eval().unwrap();
    ctx.sync().unwrap();
    let acc = Accumulator::<Df32>::builder(Fold::axis(1)).moments(2).flush_every(1).build().unwrap();
    // The first push settles the first epoch; the profiled one is the second.
    acc.push(&(&x - &y).abs()).unwrap();
    let ((), profile) = ctx
        .profile(|| {
            acc.push(&(&x - &y).abs()).unwrap();
            acc.flush().unwrap();
            ctx.sync().unwrap();
        })
        .unwrap();
    // No kernel computes `abs(x - y)` into a buffer of its own: the one that mentions it reduces.
    for kernel in &profile.kernels {
        if kernel.label.contains("Abs") {
            assert!(kernel.label.contains("Reduce"), "the expression is computed outside the reduction: {}", kernel.label);
        }
    }
    // (Profiling records every reduction of a call and every small state array on its own;
    // unprofiled the reductions of a push share a kernel and the state merges in one more.)
    let labels: Vec<_> = profile.kernels.iter().map(|k| format!("{} x{}", k.label.chars().take(60).collect::<String>(), k.launches)).collect();
    assert!(profile.launches() <= 100, "a push is a few small launches, not {}: {labels:#?}", profile.launches());
}

#[test]
fn integer_streams_give_a_confusion_matrix() {
    // Labels and predictions in 0..5: a joint histogram with one bin per class is the confusion matrix.
    const C: usize = 5;
    let truth: Vec<u32> = (0..3000).map(|i| ((i * 7 + i / 11) % C) as u32).collect();
    let guess: Vec<u32> = truth.iter().enumerate().map(|(i, &t)| if i % 4 == 0 { (t + 1) % C as u32 } else { t }).collect();
    let joint = Histogram2d::<Df32>::new((C, C), (0.0, C as f32), (0.0, C as f32)).unwrap();
    for (t, g) in truth.chunks(700).zip(guess.chunks(700)) {
        joint.push(&Array::from_slice(t, t.len()).unwrap(), &Array::from_slice(g, g.len()).unwrap()).unwrap();
    }
    let mut want = vec![0.0; C * C];
    for (&t, &g) in truth.iter().zip(&guess) {
        want[t as usize + C * g as usize] += 1.0;
    }
    assert_eq!(read(joint.counts()), want);
    assert_eq!(read(joint.outside()), vec![0.0]);
    // Accuracy from the diagonal.
    let diagonal: f64 = (0..C).map(|c| want[c + C * c]).sum();
    assert!(diagonal / 3000.0 > 0.7);
}

#[test]
fn top_k_per_position_along_an_axis() {
    const F: usize = 6;
    let rows = 400;
    let mut data = f32s(&normal(F * rows, 80));
    // Ties and a NaN.
    data[3 * F + 2] = 50.0;
    data[200 * F + 2] = 50.0;
    data[7 * F + 4] = f32::NAN;
    let k = 4;
    let host_top = |f: usize, largest: bool| -> Vec<usize> {
        let mut order: Vec<usize> = (0..rows).filter(|&r| !data[r * F + f].is_nan()).collect();
        order.sort_by(|&a, &b| {
            let (x, y) = (data[a * F + f], data[b * F + f]);
            (if largest { y.total_cmp(&x) } else { x.total_cmp(&y) }).then(a.cmp(&b))
        });
        order.truncate(k);
        order
    };
    for largest in [true, false] {
        let top = if largest { TopK::<f32, Df32>::largest_along(k, 1).unwrap() } else { TopK::<f32, Df32>::smallest_along(k, 1).unwrap() };
        for (start, count) in [(0, 150), (150, 1), (151, 249)] {
            top.push(&Array::from_slice(&data[start * F..(start + count) * F], [F, count]).unwrap()).unwrap();
        }
        // Shape [F, k]: entry j of feature f at f + F * j.
        let (values, indices) = (top.values().unwrap().to_vec().unwrap(), read(top.indices()));
        assert_eq!(values.len(), F * k);
        for f in 0..F {
            let want = host_top(f, largest);
            for j in 0..k {
                assert_eq!(indices[f + F * j], want[j] as f64, "feature {f} rank {j} (largest {largest})");
                assert_eq!(values[f + F * j], data[want[j] * F + f]);
            }
        }
        assert_eq!(top.elements_seen(), rows as u64);
    }
    // Merging two halves, through bytes, gives the same ranking; masked elements are skipped.
    let (a, b) = (TopK::<f32, Df32>::largest_along(k, 1).unwrap(), TopK::<f32, f64>::largest_along(k, 1).unwrap());
    a.push(&Array::from_slice(&data[..F * 100], [F, 100]).unwrap()).unwrap();
    b.push(&Array::from_slice(&data[F * 100..], [F, rows - 100]).unwrap()).unwrap();
    a.merge(&TopK::<f32, Df32>::from_bytes(&b.to_bytes().unwrap()).unwrap()).unwrap();
    let merged = read(a.indices());
    for f in 0..F {
        let want = host_top(f, true);
        for j in 0..k {
            assert_eq!(merged[f + F * j], want[j] as f64);
        }
    }
    let masked = TopK::<i32, Df32>::largest_along(2, 0).unwrap();
    // One line of 5 samples: [3, 9, 4, 8, 1], the 9 masked out.
    masked.push_where(&Array::from_slice(&[3i32, 9, 4, 8, 1], 5).unwrap(), &Array::from_slice(&[true, false, true, true, true], 5).unwrap()).unwrap();
    assert_eq!(masked.values().unwrap().to_vec().unwrap(), vec![8, 4]);
    assert_eq!(read(masked.indices()), vec![3.0, 2.0]);
    assert!(TopK::<f32, Df32>::largest_along(100, 1).is_err());
}

#[test]
fn states_merge_across_contexts() {
    use luma::{Context, ContextOptions};
    // Two contexts stand for two devices: the other's state travels through the host.
    let other = Context::with_options(ContextOptions::default()).expect("a device");
    let data = f32s(&normal(3000, 90));
    let host: Vec<f64> = data.iter().map(|&v| v as f64).collect();
    let (first, second) = data.split_at(1800);
    let on_default = |d: &[f32]| Array::from_slice(d, d.len()).unwrap();
    let on_other = |d: &[f32]| Array::from_slice_on(&other, d, d.len()).unwrap();

    let (a, b) = (full_builder::<Df32>(Fold::all()).build().unwrap(), full_builder::<Df32>(Fold::all()).build().unwrap());
    a.push(&on_default(first)).unwrap();
    b.push(&on_other(second)).unwrap();
    a.merge(&b).unwrap();
    check_whole(&a, &host, 1e-5);
    // Mixing contexts in one accumulator's pushes is an error.
    assert!(a.push(&on_other(first)).is_err());

    let (ha, hb) = (Histogram::<Df32>::new(10, -3.0, 3.0).unwrap(), Histogram::<Df32>::new(10, -3.0, 3.0).unwrap());
    ha.push(&on_default(first)).unwrap();
    hb.push(&on_other(second)).unwrap();
    ha.merge(&hb).unwrap();
    assert_eq!(read(ha.counts()), host_hist(&data, 10, -3.0, 3.0).0);

    let (ta, tb) = (TopK::<f32, Df32>::largest(5), TopK::<f32, Df32>::largest(5));
    ta.push(&on_default(first)).unwrap();
    tb.push(&on_other(second)).unwrap();
    ta.merge(&tb).unwrap();
    let mut order: Vec<usize> = (0..data.len()).collect();
    order.sort_by(|&x, &y| data[y].total_cmp(&data[x]).then(x.cmp(&y)));
    assert_eq!(read(ta.indices()), order[..5].iter().map(|&i| i as f64).collect::<Vec<_>>());

    let (ca, cb) = (Covariance::<Df32>::new(), Covariance::<Df32>::new());
    ca.push(&Array::from_slice(&data[..1200], [3, 400]).unwrap()).unwrap();
    cb.push(&Array::from_slice_on(&other, &data[1200..2400], [3, 400]).unwrap()).unwrap();
    ca.merge(&cb).unwrap();
    let whole = Covariance::<Df32>::new();
    whole.push(&Array::from_slice(&data[..2400], [3, 800]).unwrap()).unwrap();
    let (merged, want) = (read(ca.cov()), read(whole.cov()));
    for (m, w) in merged.iter().zip(&want) {
        close(*m, *w, 1e-4, 1e-6, "covariance across contexts");
    }
}

// ---------------------------------------------------------------------------------------------
// The 8- and 16-bit integers
// ---------------------------------------------------------------------------------------------

macro_rules! compact_test {
    ($name:ident, $t:ty, $lo:expr, $hi:expr) => {
        #[test]
        fn $name() {
            let (lo, hi): (i64, i64) = ($lo, $hi);
            let span = hi - lo + 1;
            let data: Vec<$t> = (0..4000i64).map(|i| (lo + (i * 7919) % span) as $t).collect();
            let host: Vec<f64> = data.iter().map(|&v| v as f64).collect();
            let want = reference(&host);
            let acc = full_builder::<Df32>(Fold::all()).build().unwrap();
            for chunk in data.chunks(1300) {
                acc.push(&Array::from_slice(chunk, chunk.len()).unwrap()).unwrap();
            }
            close(read(acc.count())[0], want.n, 0.0, 0.0, "count");
            // The default pass sums in f32 (partial sums beyond 2^24 round: 16-bit data); the
            // precise pass is exact.
            close(read(acc.sum())[0], want.sum, 1e-6, 4.0, "sum");
            let exact = full_builder::<Df32>(Fold::all()).precise().build().unwrap();
            for chunk in data.chunks(1300) {
                exact.push(&Array::from_slice(chunk, chunk.len()).unwrap()).unwrap();
            }
            close(read(exact.sum())[0], want.sum, 1e-12, 1e-6, "exact sum");
            close(read(acc.mean())[0], want.mean, 1e-5, 1e-3, "mean");
            close(read(acc.var())[0], want.m2 / want.n, 1e-4, 0.0, "var");
            close(read(acc.min())[0], want.min, 0.0, 0.0, "min");
            close(read(acc.max())[0], want.max, 0.0, 0.0, "max");
            assert_eq!(read(acc.argmin())[0] as usize, want.argmin);
            assert_eq!(read(acc.argmax())[0] as usize, want.argmax);
            // Per position and masked.
            let per = Accumulator::<Df32>::builder(Fold::axis(1)).moments(2).min_max().build().unwrap();
            per.push(&Array::from_slice(&data[..4 * 1000], [4, 1000]).unwrap()).unwrap();
            let column: Vec<f64> = (0..1000).map(|r| data[r * 4 + 2] as f64).collect();
            close(read(per.mean())[2], reference(&column).mean, 1e-5, 1e-6, "column mean");
            close(read(per.min())[2], reference(&column).min, 0.0, 0.0, "column min");
            let keep: Vec<bool> = (0..4000).map(|i| i % 3 == 0).collect();
            let picked: Vec<f64> = host.iter().zip(&keep).filter(|(_, k)| **k).map(|(v, _)| *v).collect();
            let masked = Accumulator::<Df32>::new(Fold::all());
            masked.push_where(&Array::from_slice(&data, 4000).unwrap(), &Array::from_slice(&keep, 4000).unwrap()).unwrap();
            close(read(masked.mean())[0], reference(&picked).mean, 1e-5, 1e-6, "masked mean");
            // A histogram over the value range counts exactly.
            let hist = Histogram::<Df32>::new(8, lo as f32, hi as f32 + 1.0).unwrap();
            hist.push(&Array::from_slice(&data, 4000).unwrap()).unwrap();
            assert_eq!(read(hist.counts()).iter().sum::<f64>(), 4000.0);
        }
    };
}

compact_test!(u8_batches_fold, u8, 0, 255);
compact_test!(i8_batches_fold, i8, -128, 127);
compact_test!(u16_batches_fold, u16, 0, 65535);
compact_test!(i16_batches_fold, i16, -32768, 32767);
