//! `group_sum` / `group_count` / `group_mean` / `group_min` / `group_max` against a host groupby.

use luma::{Array, AtomicElem};

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
}

const N: usize = 1 << 20;

fn codes(rng: &mut Rng, groups: usize) -> Vec<u32> {
    // One in 16 codes is out of range and must be skipped.
    (0..N).map(|_| if rng.next().is_multiple_of(16) { (groups as u32).wrapping_add(rng.next() % 100) } else { rng.next() % groups as u32 }).collect()
}

fn group_lists<T: Copy>(codes: &[u32], vals: &[T], groups: usize) -> Vec<Vec<T>> {
    let mut lists = vec![Vec::new(); groups];
    for (&c, &v) in codes.iter().zip(vals) {
        if (c as usize) < groups {
            lists[c as usize].push(v);
        }
    }
    lists
}

fn check_int<T: AtomicElem + Copy + PartialEq + std::fmt::Debug>(
    vals: &[T],
    groups: usize,
    c: &[u32],
    add: impl Fn(T, T) -> T,
    zero: T,
    lo: impl Fn(T, T) -> T,
    hi: impl Fn(T, T) -> T,
) {
    let x = Array::from_slice(vals, N).unwrap();
    let codes = Array::from_slice(c, N).unwrap();
    let lists = group_lists(c, vals, groups);
    let sum: Vec<T> = lists.iter().map(|l| l.iter().fold(zero, |a, &b| add(a, b))).collect();
    let min: Vec<T> = lists.iter().map(|l| l.iter().fold(T::MIN_IDENTITY, |a, &b| lo(a, b))).collect();
    let max: Vec<T> = lists.iter().map(|l| l.iter().fold(T::MAX_IDENTITY, |a, &b| hi(a, b))).collect();
    let count: Vec<u32> = lists.iter().map(|l| l.len() as u32).collect();
    assert_eq!(x.group_sum(&codes, groups).to_vec().unwrap(), sum);
    assert_eq!(x.group_min(&codes, groups).to_vec().unwrap(), min);
    assert_eq!(x.group_max(&codes, groups).to_vec().unwrap(), max);
    assert_eq!(x.group_count(&codes, groups).to_vec().unwrap(), count);
}

#[test]
fn integer_groups() {
    let mut rng = Rng(1);
    for groups in [1usize, 16, 1024, 1 << 20] {
        let c = codes(&mut rng, groups);
        let u: Vec<u32> = (0..N).map(|_| rng.next()).collect();
        check_int(&u, groups, &c, u32::wrapping_add, 0, u32::min, u32::max);
        let i: Vec<i32> = (0..N).map(|_| rng.next() as i32 - (1 << 30)).collect();
        check_int(&i, groups, &c, i32::wrapping_add, 0, i32::min, i32::max);
    }
}

#[test]
fn float_groups() {
    let mut rng = Rng(2);
    for groups in [1usize, 16, 1024, 1 << 20] {
        let c = codes(&mut rng, groups);
        let v: Vec<f32> = (0..N).map(|_| (rng.next() % 2001) as f32 / 100.0 - 10.0).collect();
        let x = Array::from_slice(&v, N).unwrap();
        let codes = Array::from_slice(&c, N).unwrap();
        let lists = group_lists(&c, &v, groups);
        let sum = x.group_sum(&codes, groups).to_vec().unwrap();
        let mean = x.group_mean(&codes, groups).to_vec().unwrap();
        let min = x.group_min(&codes, groups).to_vec().unwrap();
        let max = x.group_max(&codes, groups).to_vec().unwrap();
        for (g, l) in lists.iter().enumerate() {
            let s: f64 = l.iter().map(|&v| f64::from(v)).sum();
            let mag: f64 = l.iter().map(|&v| f64::from(v).abs()).sum();
            let tol = 1e-5 * s.abs().max(mag);
            assert!((f64::from(sum[g]) - s).abs() <= tol, "sum of group {g}: {} vs {s}", sum[g]);
            if l.is_empty() {
                assert!(mean[g].is_nan());
                assert_eq!((min[g], max[g]), (f32::INFINITY, f32::NEG_INFINITY));
            } else {
                let m = s / l.len() as f64;
                assert!((f64::from(mean[g]) - m).abs() <= tol / l.len() as f64 + 1e-12, "mean of group {g}");
                assert_eq!(min[g], l.iter().copied().fold(f32::INFINITY, f32::min));
                assert_eq!(max[g], l.iter().copied().fold(f32::NEG_INFINITY, f32::max));
            }
        }
    }
}

#[test]
fn shape_mismatch_poisons() {
    let x = Array::<f32>::full(8, 1.0).unwrap();
    let c = Array::<u32>::zeros(4).unwrap();
    assert!(x.group_sum(&c, 2).to_vec().is_err());
    assert!(x.group_count(&c, 2).to_vec().is_err());
}

#[test]
fn group_sum_fuses_values() {
    let ctx = luma::Context::new().unwrap();
    let x = Array::<f32>::full_on(&ctx, 1 << 16, 1.0).unwrap();
    let c = Array::<u32>::zeros_on(&ctx, 1 << 16).unwrap();
    let (sum, profile) = ctx.profile(|| (&x * 2.0f32).group_sum(&c, 4).to_vec().unwrap()).unwrap();
    assert_eq!(sum[0], (1 << 17) as f32);
    assert!(profile.launches() <= 2, "{} launches", profile.launches());
}
