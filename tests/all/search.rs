//! `searchsorted`, `digitize` and `isin` against host binary searches.

use luma::{bincount, Array, Context, LumaError, Side, SortKey};

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 32) as u32
    }
}

/// An element type with the host order the search promises: NaNs last and equal, `-0.0 == +0.0`.
trait Host: SortKey + Copy + std::fmt::Debug + PartialEq {
    fn random(rng: &mut Rng, i: usize) -> Self;
    /// Total-order comparison as the search sees it.
    fn lt(self, other: Self) -> bool;
    fn le(self, other: Self) -> bool {
        !other.lt(self)
    }
}

impl Host for u32 {
    fn random(rng: &mut Rng, _: usize) -> Self {
        rng.next() % 1000
    }
    fn lt(self, other: Self) -> bool {
        self < other
    }
}

impl Host for i32 {
    fn random(rng: &mut Rng, _: usize) -> Self {
        (rng.next() % 1000) as i32 - 500
    }
    fn lt(self, other: Self) -> bool {
        self < other
    }
}

impl Host for f32 {
    fn random(rng: &mut Rng, i: usize) -> Self {
        const SPECIAL: [f32; 7] = [f32::NAN, 0.0, -0.0, f32::INFINITY, f32::NEG_INFINITY, f32::MAX, f32::MIN_POSITIVE];
        if i % 11 == 3 {
            SPECIAL[rng.next() as usize % SPECIAL.len()]
        } else {
            ((rng.next() % 2001) as f32 - 1000.0) / 4.0
        }
    }
    fn lt(self, other: Self) -> bool {
        match (self.is_nan(), other.is_nan()) {
            (true, _) => false,
            (false, true) => true,
            (false, false) => self < other,
        }
    }
}

fn sorted_data<T: Host>(n: usize, seed: u64) -> Vec<T> {
    let mut rng = Rng(seed);
    let mut data: Vec<T> = (0..n).map(|i| T::random(&mut rng, i)).collect();
    data.sort_by(|a, b| if a.lt(*b) { std::cmp::Ordering::Less } else if b.lt(*a) { std::cmp::Ordering::Greater } else { std::cmp::Ordering::Equal });
    data
}

fn host_search<T: Host>(sorted: &[T], v: T, side: Side) -> u32 {
    match side {
        Side::Left => sorted.partition_point(|&e| e.lt(v)) as u32,
        Side::Right => sorted.partition_point(|&e| e.le(v)) as u32,
    }
}

fn check_search<T: Host>(ctx: &Context, sizes: &[usize], values: usize) {
    for &n in sizes {
        let sorted = sorted_data::<T>(n, 7 + n as u64);
        let mut rng = Rng(99 + n as u64);
        let probe: Vec<T> = (0..values).map(|i| T::random(&mut rng, i)).collect();
        let s = Array::from_slice_on(ctx, &sorted, n).unwrap();
        let v = Array::from_slice_on(ctx, &probe, [values / 3, 3]).unwrap();
        for side in [Side::Left, Side::Right] {
            let got = s.searchsorted(&v, side);
            assert_eq!(got.shape(), v.shape());
            let got = got.to_vec().unwrap();
            for (i, &x) in probe.iter().enumerate() {
                assert_eq!(got[i], host_search(&sorted, x, side), "n = {n} {side:?}: value {x:?} (index {i})");
            }
        }
    }
}

const SIZES: [usize; 11] = [1, 2, 3, 4, 5, 7, 8, 1000, 4095, 4096, 65537];

#[test]
fn searchsorted_matches_the_host_for_u32_i32_and_f32() {
    let ctx = Context::new().unwrap();
    check_search::<u32>(&ctx, &SIZES, 3000);
    check_search::<i32>(&ctx, &SIZES, 3000);
    check_search::<f32>(&ctx, &SIZES, 3000);
}

#[test]
fn searchsorted_finds_the_ends_ties_and_nans() {
    let ctx = Context::new().unwrap();
    let sorted = Array::from_slice_on(&ctx, &[1.0f32, 2.0, 2.0, 2.0, 5.0, f32::INFINITY, f32::NAN, f32::NAN], 8).unwrap();
    let v = Array::from_slice_on(&ctx, &[0.0f32, 1.0, 2.0, 3.0, 5.0, 9.0, f32::INFINITY, f32::NAN, -0.0, f32::NEG_INFINITY], 10).unwrap();
    assert_eq!(sorted.searchsorted(&v, Side::Left).to_vec().unwrap(), vec![0, 0, 1, 4, 4, 5, 5, 6, 0, 0]);
    assert_eq!(sorted.searchsorted(&v, Side::Right).to_vec().unwrap(), vec![0, 1, 4, 4, 5, 5, 6, 8, 0, 0]);
    // Zeros of both signs are one value.
    let zeros = Array::from_slice_on(&ctx, &[-1.0f32, -0.0, 0.0, 0.0, 1.0], 5).unwrap();
    let probe = Array::from_slice_on(&ctx, &[0.0f32, -0.0], 2).unwrap();
    assert_eq!(zeros.searchsorted(&probe, Side::Left).to_vec().unwrap(), vec![1, 1]);
    assert_eq!(zeros.searchsorted(&probe, Side::Right).to_vec().unwrap(), vec![4, 4]);
    // One element.
    let one = Array::from_slice_on(&ctx, &[7u32], 1).unwrap();
    let v = Array::from_slice_on(&ctx, &[6u32, 7, 8], 3).unwrap();
    assert_eq!(one.searchsorted(&v, Side::Left).to_vec().unwrap(), vec![0, 0, 1]);
    assert_eq!(one.searchsorted(&v, Side::Right).to_vec().unwrap(), vec![0, 1, 1]);
}

#[test]
fn a_large_sorted_array_is_searched() {
    let ctx = Context::new().unwrap();
    let n = 1usize << 22;
    let sorted = sorted_data::<u32>(n, 3).iter().enumerate().map(|(i, &v)| v + 2 * i as u32).collect::<Vec<_>>();
    let mut rng = Rng(5);
    let probe: Vec<u32> = (0..100_000).map(|_| rng.next() % (3 * n as u32)).collect();
    let s = Array::from_slice_on(&ctx, &sorted, n).unwrap();
    let v = Array::from_slice_on(&ctx, &probe, probe.len()).unwrap();
    let got = s.searchsorted(&v, Side::Right).to_vec().unwrap();
    for (i, &x) in probe.iter().enumerate() {
        assert_eq!(got[i], sorted.partition_point(|&e| e <= x) as u32, "value {x}");
    }
}

#[test]
fn searchsorted_fuses_the_values_and_reads_the_sorted_array_as_a_buffer() {
    let ctx = Context::new().unwrap();
    let edges = Array::from_slice_on(&ctx, &[0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], 8).unwrap();
    let x = Array::from_slice_on(&ctx, &[0.1f32, 1.5, 2.5, 9.0, -3.0, 6.9], 6).unwrap();
    edges.eval().unwrap();
    x.eval().unwrap();
    // `x * 2 - 1` is computed inside the search, which is the only kernel.
    let (got, profile) = ctx.profile(|| edges.searchsorted(&(&x * 2.0 - 1.0), Side::Right).to_vec().unwrap()).unwrap();
    assert_eq!(got, vec![0, 3, 5, 8, 0, 8]);
    if ctx.profiling_supported() {
        assert_eq!(profile.launches(), 1, "{profile:?}");
    }
    // A pending sorted array is evaluated first and then searched: two launches.
    let pending = Array::from_slice_on(&ctx, &[1u32, 2, 3, 4], 4).unwrap() * 10u32;
    let v = Array::from_slice_on(&ctx, &[15u32, 40, 41], 3).unwrap();
    assert_eq!(pending.searchsorted(&v, Side::Left).to_vec().unwrap(), vec![1, 3, 4]);
}

#[test]
fn digitize_follows_numpy() {
    let ctx = Context::new().unwrap();
    let edges = Array::from_slice_on(&ctx, &[0.0f32, 1.0, 2.5, 5.0], 4).unwrap();
    let x = Array::from_slice_on(&ctx, &[-1.0f32, 0.0, 0.5, 1.0, 2.4, 2.5, 4.9, 5.0, 100.0, f32::NAN], 10).unwrap();
    // np.digitize(x, bins, right=False) / right=True
    assert_eq!(x.digitize(&edges, false).to_vec().unwrap(), vec![0, 1, 1, 2, 2, 3, 3, 4, 4, 4]);
    assert_eq!(x.digitize(&edges, true).to_vec().unwrap(), vec![0, 0, 1, 1, 2, 2, 3, 3, 4, 4]);
    // Integers, and the digitized values feed a bincount: a histogram with uneven bins.
    let edges = Array::from_slice_on(&ctx, &[10i32, 20, 40], 3).unwrap();
    let ages = Array::from_slice_on(&ctx, &[5i32, 10, 15, 20, 39, 40, 99], 7).unwrap();
    let counts = bincount(&ages.digitize(&edges, false), 4).to_vec().unwrap();
    assert_eq!(counts, vec![1, 2, 2, 2]);
}

#[test]
fn isin_matches_a_host_set() {
    let ctx = Context::new().unwrap();
    let mut rng = Rng(11);
    let xs: Vec<u32> = (0..5000).map(|_| rng.next() % 2000).collect();
    let set: Vec<u32> = (0..300).map(|_| rng.next() % 2000).collect();
    let x = Array::from_slice_on(&ctx, &xs, [100, 50]).unwrap();
    let s = Array::from_slice_on(&ctx, &set, [20, 15]).unwrap();
    let got = x.isin(&s);
    assert_eq!(got.shape(), x.shape());
    let got = got.to_vec().unwrap();
    for (i, v) in xs.iter().enumerate() {
        assert_eq!(got[i], set.contains(v), "{v}");
    }
    // Floats: NaN is in no set, zeros of both signs are equal, infinities match.
    let f = Array::from_slice_on(&ctx, &[1.0f32, f32::NAN, 0.0, -0.0, f32::INFINITY, 3.5, -2.0], 7).unwrap();
    let set = Array::from_slice_on(&ctx, &[f32::NAN, -0.0, 3.5, f32::INFINITY, 1.0], 5).unwrap();
    assert_eq!(f.isin(&set).to_vec().unwrap(), vec![true, false, true, true, true, true, false]);
    // Lazy over the elements: `x + 1` is evaluated inside the search.
    let i = Array::from_slice_on(&ctx, &[1i32, 2, 3, 4], 4).unwrap();
    let set = Array::from_slice_on(&ctx, &[3i32, 5], 2).unwrap();
    assert_eq!((&i + 1i32).isin(&set).to_vec().unwrap(), vec![false, true, false, true]);
}

#[test]
fn errors_poison_the_result() {
    let ctx = Context::new().unwrap();
    let flat = Array::from_slice_on(&ctx, &[1u32, 2, 3, 4], 4).unwrap();
    let matrix = Array::from_slice_on(&ctx, &[1u32, 2, 3, 4], [2, 2]).unwrap();
    let v = Array::from_slice_on(&ctx, &[2u32], 1).unwrap();
    assert!(matches!(matrix.searchsorted(&v, Side::Left).to_vec(), Err(LumaError::Op { .. })));
    let other = Context::new().unwrap();
    let w = Array::from_slice_on(&other, &[2u32], 1).unwrap();
    assert!(flat.searchsorted(&w, Side::Left).to_vec().is_err());
    // A poisoned operand passes its error on.
    let bad = Array::<u32>::zeros_on(&ctx, 4).unwrap().reshape(5);
    assert!(bad.searchsorted(&v, Side::Left).to_vec().is_err());
    assert!(v.searchsorted(&bad, Side::Left).to_vec().is_err());
}
