//! Incremental accumulators against f64 host references: batches of any split, merge orders,
//! cancellation, masks, groups, every element type, serialization.

use luma::{Accumulator, Array, Df32, Fold, NanPolicy, Wide};

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
