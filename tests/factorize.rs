//! `factorize` and `unique` against a host dedup of the sorted values.

use luma::{Array, SortKey};

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
}

trait Host: SortKey + Copy + std::fmt::Debug {
    fn make(r: u32, i: usize) -> Self;
    /// Equality as `unique` treats it.
    fn same(self, other: Self) -> bool;
    fn order(self, other: Self) -> std::cmp::Ordering;
}

impl Host for u32 {
    fn make(r: u32, _: usize) -> u32 {
        r
    }
    fn same(self, o: u32) -> bool {
        self == o
    }
    fn order(self, o: u32) -> std::cmp::Ordering {
        self.cmp(&o)
    }
}

impl Host for i32 {
    fn make(r: u32, _: usize) -> i32 {
        r as i32
    }
    fn same(self, o: i32) -> bool {
        self == o
    }
    fn order(self, o: i32) -> std::cmp::Ordering {
        self.cmp(&o)
    }
}

impl Host for f32 {
    fn make(r: u32, i: usize) -> f32 {
        match (i % 11 == 3).then_some(r % 5) {
            Some(0) => f32::NAN,
            Some(1) => 0.0,
            Some(2) => -0.0,
            Some(3) => f32::INFINITY,
            Some(_) => f32::NEG_INFINITY,
            None => (r as i32) as f32,
        }
    }
    fn same(self, o: f32) -> bool {
        self == o || (self.is_nan() && o.is_nan())
    }
    fn order(self, o: f32) -> std::cmp::Ordering {
        self.total_cmp(&o)
    }
}

fn check<T: Host>(n: usize, distinct: u32) {
    let mut rng = Rng(n as u64 + distinct as u64);
    let data: Vec<T> = (0..n).map(|i| T::make(rng.next() % distinct, i)).collect();
    let x = Array::from_slice(&data, n).unwrap();
    let (uniques, codes) = x.factorize();
    let (uniques, codes) = (uniques.to_vec().unwrap(), codes.to_vec().unwrap());
    // Host: sort, then drop neighbours that are the same (NaNs sort last under total_cmp of
    // positive NaNs; -0.0 sorts before 0.0 and is dropped as its equal).
    let mut sorted: Vec<T> = data.clone();
    sorted.sort_by(|a, b| a.order(*b));
    let mut want: Vec<T> = Vec::new();
    for v in sorted {
        if want.last().is_none_or(|&l| !l.same(v)) {
            want.push(v);
        }
    }
    assert_eq!(uniques.len(), want.len(), "n={n} distinct={distinct}");
    for (u, w) in uniques.iter().zip(&want) {
        assert!(u.same(*w), "{u:?} vs {w:?}");
    }
    for (i, &c) in codes.iter().enumerate() {
        assert!(uniques[c as usize].same(data[i]), "element {i}: {:?} vs {:?}", uniques[c as usize], data[i]);
    }
    assert_eq!(x.unique().to_vec().unwrap().len(), want.len());
}

fn sizes<T: Host>() {
    for (n, distinct) in [(1usize, 1u32), (2, 2), (1 << 20, 7), (1 << 20, 1 << 18), (1 << 20, u32::MAX)] {
        check::<T>(n, distinct);
    }
}

#[test]
fn factorize_u32() {
    sizes::<u32>();
}

#[test]
fn factorize_i32() {
    sizes::<i32>();
}

#[test]
fn factorize_f32() {
    sizes::<f32>();
}

#[test]
fn nan_and_zero_are_one_value() {
    let x = Array::from_slice(&[f32::NAN, 0.0, -0.0, 1.0, f32::NAN, 1.0], 6).unwrap();
    let (u, c) = x.factorize();
    let u = u.to_vec().unwrap();
    assert_eq!(u.len(), 3);
    assert_eq!(u[0], 0.0);
    assert_eq!(u[1], 1.0);
    assert!(u[2].is_nan());
    assert_eq!(c.to_vec().unwrap(), [2, 0, 0, 1, 2, 1]);
}

#[test]
fn the_unique_of_signed_zeros_is_the_first_in_sorted_order() {
    // Many copies, so that a race between the writes of a run would show.
    let data: Vec<f32> = (0..1 << 16).map(|i| if i % 3 == 0 { 0.0 } else if i % 3 == 1 { -0.0 } else { 1.0 }).collect();
    for _ in 0..4 {
        let x = Array::from_slice(&data, data.len()).unwrap();
        let bits: Vec<u32> = x.unique().to_vec().unwrap().iter().map(|v| v.to_bits()).collect();
        assert_eq!(bits, [(-0.0f32).to_bits(), 1.0f32.to_bits()]);
    }
    let (u, c) = Array::from_slice(&[0.0f32, -0.0, 1.0, -0.0], 4).unwrap().factorize();
    assert_eq!(u.to_vec().unwrap().iter().map(|v| v.to_bits()).collect::<Vec<_>>(), [(-0.0f32).to_bits(), 1.0f32.to_bits()]);
    assert_eq!(c.to_vec().unwrap(), [0, 0, 1, 0]);
}
