//! `scan_by_key`, the reductions by key and the set operations against host references.

use luma::{Array, Context};

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 32) as u32
    }
}

/// `n` keys in runs of 1 to `max_run` equal keys (consecutive runs differ), and small values
/// (exact in `f32` in any order).
fn runs(n: usize, max_run: u32, seed: u64) -> (Vec<u32>, Vec<i32>) {
    let mut rng = Rng(seed);
    let mut keys = Vec::with_capacity(n);
    let mut key = 0u32;
    while keys.len() < n {
        key += 1 + rng.next() % 3;
        let len = 1 + rng.next() % max_run;
        for _ in 0..len.min((n - keys.len()) as u32) {
            keys.push(key);
        }
    }
    let values = (0..n).map(|_| (rng.next() % 17) as i32 - 8).collect();
    (keys, values)
}

fn host_scan(keys: &[u32], values: &[i32], n: usize, op: impl Fn(i32, i32) -> i32, exclusive: Option<i32>) -> Vec<i32> {
    let mut out = vec![0; values.len()];
    for line in 0..values.len() / n {
        let base = line * n;
        let mut acc = 0;
        for j in 0..n {
            let i = base + j;
            let head = j == 0 || keys[i] != keys[i - 1];
            let next = if head { values[i] } else { op(acc, values[i]) };
            out[i] = match exclusive {
                Some(identity) => if head { identity } else { acc },
                None => next,
            };
            acc = next;
        }
    }
    out
}

#[test]
fn scan_by_key_matches_the_host_for_every_operator() {
    let ctx = Context::new().unwrap();
    for (n, max_run) in [(1usize, 1u32), (2, 3), (1000, 7), (4096, 5000), (4097, 3), (9000, 70), (70_000, 9000), (70_001, 1), (100_003, 100_000)] {
        let (keys, values) = runs(n, max_run, 7 + n as u64);
        let k = Array::from_slice_on(&ctx, &keys, n).unwrap();
        let ki: Vec<i32> = keys.iter().map(|&x| x as i32).collect();
        let kf: Vec<f32> = keys.iter().map(|&x| x as f32).collect();
        let (ki, kf) = (Array::from_slice_on(&ctx, &ki, n).unwrap(), Array::from_slice_on(&ctx, &kf, n).unwrap());
        let vi = Array::from_slice_on(&ctx, &values, n).unwrap();
        let vf: Vec<f32> = values.iter().map(|&x| x as f32).collect();
        let vf_arr = Array::from_slice_on(&ctx, &vf, n).unwrap();
        let sum = host_scan(&keys, &values, n, |a, b| a + b, None);
        assert_eq!(vi.scan_by_key(&k, 0, |a, b| a + b).to_vec().unwrap(), sum, "i32 sum, u32 keys, n={n}");
        assert_eq!(vi.cumsum_by_key(&ki, 0).to_vec().unwrap(), sum, "cumsum i32 keys, n={n}");
        assert_eq!(vf_arr.cumsum_by_key(&kf, 0).to_vec().unwrap(), sum.iter().map(|&x| x as f32).collect::<Vec<_>>(), "f32 sum, f32 keys, n={n}");
        assert_eq!(vi.scan_by_key(&k, 0, |a, b| a.minimum(b)).to_vec().unwrap(), host_scan(&keys, &values, n, i32::min, None), "min n={n}");
        assert_eq!(vi.scan_by_key(&k, 0, |a, b| a.maximum(b)).to_vec().unwrap(), host_scan(&keys, &values, n, i32::max, None), "max n={n}");
        let small: Vec<i32> = values.iter().map(|&x| 1 + x.rem_euclid(2)).collect();
        let vs = Array::from_slice_on(&ctx, &small, n).unwrap();
        let product = host_scan(&keys, &small, n, |a, b| a.wrapping_mul(b), None);
        assert_eq!(vs.scan_by_key(&k, 0, |a, b| a * b).to_vec().unwrap(), product, "product n={n}");
        assert_eq!(vi.scan_by_key_exclusive(&k, 0, 0, |a, b| a + b).to_vec().unwrap(), host_scan(&keys, &values, n, |a, b| a + b, Some(0)), "exclusive n={n}");
    }
}

#[test]
fn scan_by_key_along_every_axis() {
    let ctx = Context::new().unwrap();
    for (dims, axis) in [([37usize, 50, 1, 1], 1usize), ([50, 37, 1, 1], 0), ([6, 5, 4, 3], 2), ([6, 5, 4, 3], 3), ([6, 5, 4, 3], 1), ([3, 5000, 2, 1], 1), ([5000, 3, 1, 1], 0)] {
        let len: usize = dims.iter().product();
        let (keys, values) = runs(len, 4, 99 + len as u64);
        // Lines along `axis` through the array: line cells are `stride` apart.
        let stride: usize = dims[..axis].iter().product();
        let n = dims[axis];
        let mut want = vec![0; len];
        for line in 0..len / n {
            let base = line % stride + (line / stride) * stride * n;
            let mut acc = 0;
            for j in 0..n {
                let i = base + j * stride;
                let head = j == 0 || keys[i] != keys[i - stride];
                acc = if head { values[i] } else { acc + values[i] };
                want[i] = acc;
            }
        }
        let k = Array::from_slice_on(&ctx, &keys, dims).unwrap();
        let v = Array::from_slice_on(&ctx, &values, dims).unwrap();
        assert_eq!(v.cumsum_by_key(&k, axis).to_vec().unwrap(), want, "dims {dims:?} axis {axis}");
    }
}

/// Runs from the host: `(first key, run of values)`.
fn host_runs(keys: &[u32], values: &[i32]) -> Vec<(u32, Vec<i32>)> {
    let mut out: Vec<(u32, Vec<i32>)> = Vec::new();
    for (i, &k) in keys.iter().enumerate() {
        if i == 0 || keys[i - 1] != k {
            out.push((k, Vec::new()));
        }
        out.last_mut().unwrap().1.push(values[i]);
    }
    out
}

#[test]
fn reductions_by_key_match_the_host() {
    let ctx = Context::new().unwrap();
    for (n, max_run) in [(1usize, 1u32), (5, 2), (1000, 7), (9000, 70), (70_000, 9000), (70_001, 1)] {
        let (keys, values) = runs(n, max_run, 31 + n as u64);
        let k = Array::from_slice_on(&ctx, &keys, n).unwrap();
        let v = Array::from_slice_on(&ctx, &values, n).unwrap();
        let groups = host_runs(&keys, &values);
        let want_keys: Vec<u32> = groups.iter().map(|g| g.0).collect();
        let check = |name: &str, got: (Array<u32>, Array<i32>), f: &dyn Fn(&[i32]) -> i32| {
            assert_eq!(got.0.to_vec().unwrap(), want_keys, "{name} keys n={n}");
            assert_eq!(got.1.to_vec().unwrap(), groups.iter().map(|g| f(&g.1)).collect::<Vec<_>>(), "{name} n={n}");
        };
        check("sum", v.sum_by_key(&k).unwrap(), &|g| g.iter().sum());
        check("min", v.min_by_key(&k).unwrap(), &|g| *g.iter().min().unwrap());
        check("max", v.max_by_key(&k).unwrap(), &|g| *g.iter().max().unwrap());
        let small: Vec<i32> = values.iter().map(|&x| 1 + x.rem_euclid(2)).collect();
        let sg = host_runs(&keys, &small);
        let (pk, pv) = Array::from_slice_on(&ctx, &small, n).unwrap().product_by_key(&k).unwrap();
        assert_eq!(pk.to_vec().unwrap(), want_keys);
        assert_eq!(pv.to_vec().unwrap(), sg.iter().map(|g| g.1.iter().fold(1i32, |a, &b| a.wrapping_mul(b))).collect::<Vec<_>>(), "product n={n}");
        let (_, counts) = v.count_by_key(&k).unwrap();
        assert_eq!(counts.to_vec().unwrap(), groups.iter().map(|g| g.1.iter().filter(|&&x| x != 0).count() as u32).collect::<Vec<_>>(), "count n={n}");
        let (_, any) = v.any_true_by_key(&k).unwrap();
        assert_eq!(any.to_vec().unwrap(), groups.iter().map(|g| g.1.iter().any(|&x| x != 0)).collect::<Vec<_>>(), "any n={n}");
        let (_, all) = v.all_true_by_key(&k).unwrap();
        assert_eq!(all.to_vec().unwrap(), groups.iter().map(|g| g.1.iter().all(|&x| x != 0)).collect::<Vec<_>>(), "all n={n}");
        // Float values and keys.
        let kf: Vec<f32> = keys.iter().map(|&x| x as f32).collect();
        let vf: Vec<f32> = values.iter().map(|&x| x as f32).collect();
        let (fk, fv) = Array::from_slice_on(&ctx, &vf, n).unwrap().sum_by_key(&Array::from_slice_on(&ctx, &kf, n).unwrap()).unwrap();
        assert_eq!(fk.to_vec().unwrap(), want_keys.iter().map(|&x| x as f32).collect::<Vec<_>>());
        assert_eq!(fv.to_vec().unwrap(), groups.iter().map(|g| g.1.iter().sum::<i32>() as f32).collect::<Vec<_>>(), "f32 sum n={n}");
    }
}

#[test]
fn equal_keys_that_are_apart_are_different_runs() {
    let ctx = Context::new().unwrap();
    let k = Array::from_slice_on(&ctx, &[1u32, 1, 2, 1, 1, 1, 3], 7).unwrap();
    let v = Array::from_slice_on(&ctx, &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], 7).unwrap();
    let (keys, sums) = v.sum_by_key(&k).unwrap();
    assert_eq!(keys.to_vec().unwrap(), [1, 2, 1, 3]);
    assert_eq!(sums.to_vec().unwrap(), [3.0, 3.0, 15.0, 7.0]);
    assert!(v.sum_by_key(&Array::from_slice_on(&ctx, &[1u32, 2], 2).unwrap()).is_err());
}

fn host_unique(v: &[f32]) -> Vec<f32> {
    let mut sorted: Vec<f32> = v.to_vec();
    sorted.sort_by(f32::total_cmp);
    let mut out: Vec<f32> = Vec::new();
    for x in sorted {
        let last = out.last().copied();
        let same = last.is_some_and(|l| l == x || (l.is_nan() && x.is_nan()));
        if !same {
            out.push(x);
        }
    }
    out
}

#[test]
fn set_operations_match_the_host() {
    let ctx = Context::new().unwrap();
    for (na, nb) in [(1usize, 1usize), (5, 9), (1000, 800), (30_000, 40_000)] {
        let mut rng = Rng(5 + na as u64);
        let mut make = |n: usize, range: u32| -> Vec<f32> {
            (0..n).map(|i| if i % 97 == 5 { f32::NAN } else if i % 89 == 3 { -0.0 } else if i % 91 == 4 { 0.0 } else { (rng.next() % range) as f32 - (range / 2) as f32 }).collect()
        };
        let (a, b) = (make(na, 2000), make(nb, 3000));
        let (xa, xb) = (Array::from_slice_on(&ctx, &a, na).unwrap(), Array::from_slice_on(&ctx, &b, nb).unwrap());
        // Zeros are equal and a zero in a result is `+0.0`; all NaNs are one value.
        let bits = |v: &[f32]| v.iter().map(|x| if x.is_nan() { 0x7fc0_0000 } else if *x == 0.0 { 0 } else { x.to_bits() }).collect::<Vec<u32>>();
        let plus = |v: &[f32]| v.iter().map(|&x| if x == 0.0 { 0.0 } else { x }).collect::<Vec<f32>>();
        let zero_bits = |v: &[f32]| v.iter().filter(|x| **x == 0.0).map(|x| x.to_bits()).collect::<Vec<u32>>();
        // Unique.
        let got = xa.set_unique(false).unwrap().to_vec().unwrap();
        assert_eq!(bits(&got), bits(&host_unique(&a)), "unique n={na}");
        assert!(zero_bits(&got).iter().all(|&b| b == 0), "zeros of a unique are +0.0");
        let sorted = host_unique(&a);
        let xs = Array::from_slice_on(&ctx, &sorted, sorted.len()).unwrap();
        assert_eq!(bits(&xs.set_unique(true).unwrap().to_vec().unwrap()), bits(&sorted));
        // Union.
        let mut joined = a.to_vec();
        joined.extend_from_slice(&b);
        let got = xa.set_union(&xb, false).unwrap().to_vec().unwrap();
        assert_eq!(bits(&got), bits(&host_unique(&joined)), "union {na} {nb}");
        assert!(zero_bits(&got).iter().all(|&b| b == 0));
        let (ua, ub) = (host_unique(&a), host_unique(&b));
        let (xua, xub) = (Array::from_slice_on(&ctx, &ua, ua.len()).unwrap(), Array::from_slice_on(&ctx, &ub, ub.len()).unwrap());
        assert_eq!(bits(&xua.set_union(&xub, true).unwrap().to_vec().unwrap()), bits(&host_unique(&joined)));
        // Intersection.
        let both: Vec<f32> = ua.iter().copied().filter(|x| ub.iter().any(|y| y == x || (x.is_nan() && y.is_nan()))).collect();
        let got = xa.set_intersect(&xb, false).unwrap().map(|r| r.to_vec().unwrap()).unwrap_or_default();
        assert_eq!(bits(&got), bits(&plus(&both)), "intersect {na} {nb}");
        assert!(zero_bits(&got).iter().all(|&b| b == 0));
        let got = xua.set_intersect(&xub, true).unwrap().map(|r| r.to_vec().unwrap()).unwrap_or_default();
        assert_eq!(bits(&got), bits(&plus(&both)));
    }
    // Disjoint sets intersect in nothing; integer sets.
    let a = Array::from_slice_on(&ctx, &[1i32, 3, 5], 3).unwrap();
    let b = Array::from_slice_on(&ctx, &[2i32, 4, 6], 3).unwrap();
    assert!(a.set_intersect(&b, true).unwrap().is_none());
    assert_eq!(a.set_union(&b, true).unwrap().to_vec().unwrap(), [1, 2, 3, 4, 5, 6]);
    let c = Array::from_slice_on(&ctx, &[5u32, 1, 5, 1, 9], 5).unwrap();
    assert_eq!(c.set_unique(false).unwrap().to_vec().unwrap(), [1, 5, 9]);
}

/// Sorted distinct inputs of a few hundred thousand elements (hundreds of merge tiles): the
/// union and the intersection against host sets, for every key type.
#[test]
fn set_operations_on_large_sorted_inputs_match_host_sets() {
    let ctx = Context::new().unwrap();
    let mut rng = Rng(77);
    for (na, nb, range) in [(300_000usize, 250_000usize, 700_000u32), (5000, 400_000, 1_000_000), (100_000, 100_000, 100_000_000), (70_001, 70_000, 71_000)] {
        let mut a: Vec<u32> = (0..na).map(|_| rng.next() % range).collect();
        let mut b: Vec<u32> = (0..nb).map(|_| rng.next() % range).collect();
        a.sort_unstable();
        a.dedup();
        b.sort_unstable();
        b.dedup();
        let sb: std::collections::HashSet<u32> = b.iter().copied().collect();
        let both: Vec<u32> = a.iter().copied().filter(|x| sb.contains(x)).collect();
        let mut all = a.clone();
        all.extend_from_slice(&b);
        all.sort_unstable();
        all.dedup();
        let (xa, xb) = (Array::from_slice_on(&ctx, &a, a.len()).unwrap(), Array::from_slice_on(&ctx, &b, b.len()).unwrap());
        assert_eq!(xa.set_union(&xb, true).unwrap().to_vec().unwrap(), all, "u32 union {na} {nb}");
        assert_eq!(xa.set_intersect(&xb, true).unwrap().map(|r| r.to_vec().unwrap()).unwrap_or_default(), both, "u32 intersect {na} {nb}");
        // The same as i32 (negatives sort first).
        let shift = |v: &[u32]| v.iter().map(|&x| x as i32 - (range / 2) as i32).collect::<Vec<i32>>();
        let (ia, ib) = (shift(&a), shift(&b));
        let (xia, xib) = (Array::from_slice_on(&ctx, &ia, ia.len()).unwrap(), Array::from_slice_on(&ctx, &ib, ib.len()).unwrap());
        assert_eq!(xia.set_union(&xib, true).unwrap().to_vec().unwrap(), shift(&all), "i32 union {na} {nb}");
        assert_eq!(xia.set_intersect(&xib, true).unwrap().map(|r| r.to_vec().unwrap()).unwrap_or_default(), shift(&both));
        // Disjoint.
        let evens: Vec<u32> = a.iter().map(|x| x * 2).collect();
        let odds: Vec<u32> = b.iter().map(|x| x * 2 + 1).collect();
        let (xe, xo) = (Array::from_slice_on(&ctx, &evens, evens.len()).unwrap(), Array::from_slice_on(&ctx, &odds, odds.len()).unwrap());
        assert!(xe.set_intersect(&xo, true).unwrap().is_none());
        assert_eq!(xe.set_union(&xo, true).unwrap().len(), evens.len() + odds.len());
    }
}

/// Sorted inputs (duplicates allowed: the merge drops them) against `BTreeSet` references.
fn check_merge_u32(ctx: &Context, a: &[u32], b: &[u32], what: &str) {
    let sa: std::collections::BTreeSet<u32> = a.iter().copied().collect();
    let sb: std::collections::BTreeSet<u32> = b.iter().copied().collect();
    let union: Vec<u32> = sa.union(&sb).copied().collect();
    let both: Vec<u32> = sa.intersection(&sb).copied().collect();
    let (xa, xb) = (Array::from_slice_on(ctx, a, a.len()).unwrap(), Array::from_slice_on(ctx, b, b.len()).unwrap());
    assert_eq!(xa.set_union(&xb, true).unwrap().to_vec().unwrap(), union, "union {what}");
    let got = xa.set_intersect(&xb, true).unwrap();
    assert_eq!(got.map(|r| r.to_vec().unwrap()).unwrap_or_default(), both, "intersect {what}");
}

/// Tile boundaries of the merge (2048 merged items a tile), one-element sides, duplicates inside
/// `a` and `b` (also across tile boundaries), identical, nested and disjoint inputs.
#[test]
fn set_merge_tile_boundaries_duplicates_and_extremes() {
    let ctx = Context::new().unwrap();
    let mut rng = Rng(4242);
    let sorted = |rng: &mut Rng, n: usize, range: u32| -> Vec<u32> {
        let mut v: Vec<u32> = (0..n).map(|_| rng.next() % range).collect();
        v.sort_unstable();
        v
    };
    for (na, nb) in [(1usize, 1usize), (1, 2047), (2047, 1), (1, 2048), (2048, 1), (1024, 1024), (2048, 2048), (2049, 2047), (4095, 1), (1, 4097), (4097, 4095), (6000, 2), (3, 70_000)] {
        // Few distinct values: runs of duplicates that cross the tile boundaries.
        let (a, b) = (sorted(&mut rng, na, 1 + (na + nb) as u32 / 40), sorted(&mut rng, nb, 1 + (na + nb) as u32 / 40));
        check_merge_u32(&ctx, &a, &b, &format!("duplicates {na} {nb}"));
        let (a, b) = (sorted(&mut rng, na, 1 << 30), sorted(&mut rng, nb, 1 << 30));
        check_merge_u32(&ctx, &a, &b, &format!("sparse {na} {nb}"));
    }
    // One value everywhere; the extreme keys.
    check_merge_u32(&ctx, &vec![7; 5000], &vec![7; 3000], "one value");
    check_merge_u32(&ctx, &[0, 0, u32::MAX], &[0, u32::MAX, u32::MAX], "extremes");
    check_merge_u32(&ctx, &vec![u32::MAX; 4100], &vec![u32::MAX; 4100], "only the largest key");
    // Identical, nested, disjoint (a before b, b before a), interleaved one by one.
    let a = sorted(&mut rng, 30_000, 1 << 24);
    check_merge_u32(&ctx, &a, &a, "identical");
    let inner: Vec<u32> = a.iter().step_by(3).copied().collect();
    check_merge_u32(&ctx, &a, &inner, "nested");
    check_merge_u32(&ctx, &inner, &a, "nested the other way");
    let high: Vec<u32> = a.iter().map(|x| x + (1 << 25)).collect();
    check_merge_u32(&ctx, &a, &high, "a below b");
    check_merge_u32(&ctx, &high, &a, "b below a");
    let evens: Vec<u32> = (0..9000).map(|i| 2 * i).collect();
    let odds: Vec<u32> = (0..9000).map(|i| 2 * i + 1).collect();
    check_merge_u32(&ctx, &evens, &odds, "interleaved");
    check_merge_u32(&ctx, &evens, &evens[2000..4100], "a middle part");
}

/// Sorted floats with NaNs and both zeros, a few hundred tiles: all NaNs are one value, `-0.0`
/// and `+0.0` are equal and come out as `+0.0`.
#[test]
fn set_merge_floats_with_nans_and_zeros() {
    let ctx = Context::new().unwrap();
    let mut rng = Rng(99);
    let pool = |rng: &mut Rng, n: usize| -> Vec<f32> {
        let mut v: Vec<f32> = (0..n)
            .map(|_| match rng.next() % 40 {
                0 => f32::NAN,
                1 => -0.0,
                2 => 0.0,
                3 => f32::INFINITY,
                4 => f32::NEG_INFINITY,
                x => (x as f32 - 20.0) * 0.5,
            })
            .collect();
        v.sort_by(|x, y| match (x.is_nan(), y.is_nan()) {
            (true, true) => std::cmp::Ordering::Equal,
            (true, false) => std::cmp::Ordering::Greater,
            (false, true) => std::cmp::Ordering::Less,
            _ => x.partial_cmp(y).unwrap(),
        });
        v
    };
    let norm = |x: f32| if x.is_nan() { 0x7fc0_0000 } else if x == 0.0 { 0 } else { x.to_bits() };
    for (na, nb) in [(5usize, 9usize), (3000, 2500), (500_000, 400_000)] {
        let (a, b) = (pool(&mut rng, na), pool(&mut rng, nb));
        let sa: std::collections::BTreeSet<u32> = a.iter().map(|&x| norm(x)).collect();
        let sb: std::collections::BTreeSet<u32> = b.iter().map(|&x| norm(x)).collect();
        // The float order of the keys of a set: by value, NaN last (key order of the sort).
        let order = |bits: &u32| {
            let x = f32::from_bits(*bits);
            if x.is_nan() { (1, 0.0f32) } else { (0, x + 0.0) }
        };
        let sorted_by_value = |set: Vec<u32>| {
            let mut v = set;
            v.sort_by(|p, q| {
                let (p, q) = (order(p), order(q));
                p.0.cmp(&q.0).then(p.1.partial_cmp(&q.1).unwrap())
            });
            v
        };
        let union = sorted_by_value(sa.union(&sb).copied().collect());
        let both = sorted_by_value(sa.intersection(&sb).copied().collect());
        let (xa, xb) = (Array::from_slice_on(&ctx, &a, a.len()).unwrap(), Array::from_slice_on(&ctx, &b, b.len()).unwrap());
        let got: Vec<u32> = xa.set_union(&xb, true).unwrap().to_vec().unwrap().iter().map(|&x| norm(x)).collect();
        assert_eq!(got, union, "union {na} {nb}");
        let raw = xa.set_union(&xb, true).unwrap().to_vec().unwrap();
        assert!(raw.iter().all(|x| *x != 0.0 || x.to_bits() == 0), "a zero of the union is +0.0");
        let got: Vec<u32> = xa.set_intersect(&xb, true).unwrap().map(|r| r.to_vec().unwrap().iter().map(|&x| norm(x)).collect()).unwrap_or_default();
        assert_eq!(got, both, "intersect {na} {nb}");
    }
}
