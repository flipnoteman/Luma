//! `Array<bool>`: the result of comparisons, its operators, promotions, reductions and casts.

use luma::*;

fn f32s(data: &[f32]) -> Array<f32> {
    Array::from_slice(data, data.len()).unwrap()
}

#[test]
fn comparisons_are_bools() {
    let x = f32s(&[1.0, -2.0, 0.0, 3.5, f32::NAN]);
    let m: Array<bool> = x.gt(0.0);
    assert_eq!(m.to_vec().unwrap(), [true, false, false, true, false]);
    assert_eq!(m.to_u32().to_vec().unwrap(), [1, 0, 0, 1, 0]);
    assert_eq!(m.to_i32().to_vec().unwrap(), [1, 0, 0, 1, 0]);
    assert_eq!(m.to_f32().to_vec().unwrap(), [1.0, 0.0, 0.0, 1.0, 0.0]);
    let classes: Array<bool> = x.is_nan();
    assert_eq!(classes.to_u32().to_vec().unwrap(), [0, 0, 0, 0, 1]);
}

#[test]
fn bools_round_trip_the_host() {
    let host = [true, false, true, true, false, false, true];
    let a = Array::from_slice(&host, host.len()).unwrap();
    assert_eq!(a.to_vec().unwrap(), host);
    let mut out = [false; 7];
    a.read_into(&mut out).unwrap();
    assert_eq!(out, host);
    let mut b = Array::<bool>::zeros(7).unwrap();
    assert_eq!(b.to_vec().unwrap(), [false; 7]);
    b.write(&host).unwrap();
    assert_eq!(b.to_vec().unwrap(), host);
    assert_eq!(Array::full(4, true).unwrap().to_vec().unwrap(), [true; 4]);
    assert_eq!(Array::full(4, false).unwrap().to_vec().unwrap(), [false; 4]);
    assert!(a.item().is_err());
    assert!(Array::from_slice(&[true], 1).unwrap().item().unwrap());
    assert_eq!(a.to_array::<7>().unwrap(), host);
}

#[test]
fn bools_combine_and_count() {
    let xs: Vec<f32> = (0..1000).map(|i| ((i * 37) % 101) as f32 - 50.0).collect();
    let ys: Vec<f32> = (0..1000).map(|i| ((i * 53) % 97) as f32 - 48.0).collect();
    let (x, y) = (f32s(&xs), f32s(&ys));
    let (m, m2) = (x.gt(0.0), y.lt(10.0));
    let want = xs.iter().zip(&ys).filter(|&(&a, &b)| a <= 0.0 && b < 10.0).count() as u32;
    assert_eq!((!&m & &m2).count().item().unwrap(), want);
    let want_or = xs.iter().zip(&ys).filter(|&(&a, &b)| a > 0.0 || b < 10.0).count() as u32;
    assert_eq!((&m | &m2).count().item().unwrap(), want_or);
    let want_xor = xs.iter().zip(&ys).filter(|&(&a, &b)| (a > 0.0) != (b < 10.0)).count() as u32;
    assert_eq!((&m ^ &m2).count().item().unwrap(), want_xor);
    let want_and = xs.iter().zip(&ys).filter(|&(&a, &b)| a > 0.0 && b < 10.0).count() as u32;
    let mut acc = m.clone();
    acc &= &m2;
    assert_eq!(acc.count().item().unwrap(), want_and);
    // `bool * bool` is the logical and; sums of booleans are counts.
    assert_eq!((&m * &m2).sum().item().unwrap(), want_and);
    assert_eq!((&m & true).count().item().unwrap(), m.count().item().unwrap());
    assert_eq!((&m | true).count().item().unwrap(), 1000);
    assert_eq!((&m ^ true).count().item().unwrap(), 1000 - m.count().item().unwrap());
    assert_eq!((true & &m2).count().item().unwrap(), m2.count().item().unwrap());
}

#[test]
fn any_and_all() {
    let x = f32s(&[1.0, 2.0, 3.0]);
    assert!(x.gt(2.5).any().item().unwrap());
    assert!(!x.gt(2.5).all().item().unwrap());
    assert!(x.gt(0.5).all().item().unwrap());
    assert!(!x.gt(5.0).any().item().unwrap());
    // Bools stay lazy: the reduction of a pending comparison fuses with it.
    assert!((&x * 2.0).gt(5.0).any().item().unwrap());
}

#[test]
fn select_takes_bool_and_u32_conditions() {
    let x = f32s(&[-1.0, 2.0, -3.0, 4.0]);
    assert_eq!(select(&x.gt(0.0), 1.0, 0.0).to_vec().unwrap(), [0.0, 1.0, 0.0, 1.0]);
    assert_eq!(select(&x.gt(0.0), &x, 0.0).to_vec().unwrap(), [0.0, 2.0, 0.0, 4.0]);
    assert_eq!(x.sum_where(&x.lt(0.0)).item().unwrap(), -4.0);
    let from_host = Array::from_slice(&[true, false, true, false], 4).unwrap();
    assert_eq!(select(&from_host, &x, 9.0).to_vec().unwrap(), [-1.0, 9.0, -3.0, 9.0]);
    assert_eq!(from_host.eq(true).to_u32().to_vec().unwrap(), [1, 0, 1, 0]);
    // 0.3 code: a `u32` condition, non-zero meaning true.
    let words = Array::from_slice(&[0u32, 7, 1, 0], 4).unwrap();
    assert_eq!(select(&words, &x, 9.0).to_vec().unwrap(), [9.0, 2.0, -3.0, 9.0]);
    assert_eq!(x.sum_where(&words).item().unwrap(), -1.0);
}

#[test]
fn bool_times_number_selects() {
    let x = f32s(&[-1.0, 2.0, f32::INFINITY, 4.0]);
    let m = Array::from_slice(&[true, false, false, true], 4).unwrap();
    let want = [-1.0, 0.0, 0.0, 4.0];
    assert_eq!((&m * &x).to_vec().unwrap(), want);
    assert_eq!((&x * &m).to_vec().unwrap(), want);
    assert_eq!((m.clone() * x.clone()).to_vec().unwrap(), want);
    assert_eq!((&m * 2.0).to_vec().unwrap(), [2.0, 0.0, 0.0, 2.0]);
    assert_eq!((2.0 * &m).to_vec().unwrap(), [2.0, 0.0, 0.0, 2.0]);
    assert_eq!((&m + &x).to_vec().unwrap(), [0.0, 2.0, f32::INFINITY, 5.0]);
    assert_eq!((&x - &m).to_vec().unwrap(), [-2.0, 2.0, f32::INFINITY, 3.0]);
    assert_eq!((&m - &x).to_vec().unwrap(), [2.0, -2.0, f32::NEG_INFINITY, -3.0]);
    assert_eq!((1.0 - &m).to_vec().unwrap(), [0.0, 1.0, 1.0, 0.0]);
    assert_eq!((&m + 1.0).to_vec().unwrap(), [2.0, 1.0, 1.0, 2.0]);
    assert_eq!((&m - 1.0).to_vec().unwrap(), [0.0, -1.0, -1.0, 0.0]);
    let mut acc = x.clone();
    acc += &m;
    assert_eq!(acc.to_vec().unwrap()[..2], [0.0, 2.0]);
    let mut acc = f32s(&[1.0, 2.0, 3.0, 4.0]);
    acc *= &m;
    assert_eq!(acc.to_vec().unwrap(), [1.0, 0.0, 0.0, 4.0]);
}

#[test]
fn promotions_cover_the_integer_types() {
    let m = Array::from_slice(&[true, false, true], 3).unwrap();
    let u = Array::from_slice(&[5u32, 6, 7], 3).unwrap();
    let i = Array::from_slice(&[-5i32, 6, -7], 3).unwrap();
    assert_eq!((&m * &u).to_vec().unwrap(), [5, 0, 7]);
    assert_eq!((&u + &m).to_vec().unwrap(), [6, 6, 8]);
    assert_eq!((&m - &u).to_vec().unwrap(), [1u32.wrapping_sub(5), 0u32.wrapping_sub(6), 1u32.wrapping_sub(7)]);
    assert_eq!((&m * &i).to_vec().unwrap(), [-5, 0, -7]);
    assert_eq!((&i - &m).to_vec().unwrap(), [-6, 6, -8]);
    assert_eq!((&m * 3u32).to_vec().unwrap(), [3, 0, 3]);
    assert_eq!((&m + 3i32).to_vec().unwrap(), [4, 3, 4]);
}

#[test]
fn promotions_broadcast() {
    let m = Array::from_slice(&[true, false], [1, 2]).unwrap();
    let x = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], [3, 2]).unwrap();
    let product = &m * &x;
    assert_eq!(product.shape().dims(), [3, 2, 1, 1]);
    assert_eq!(product.to_vec().unwrap(), [1.0, 2.0, 3.0, 0.0, 0.0, 0.0]);
    let row = Array::from_slice(&[true, false, true], [3, 1]).unwrap();
    assert_eq!((&x * &row).to_vec().unwrap(), [1.0, 0.0, 3.0, 4.0, 0.0, 6.0]);
    let wrong = Array::from_slice(&[true, false], 2).unwrap();
    assert!(matches!((&wrong * &x).error().map(LumaError::root), Some(LumaError::ShapeMismatch { .. })));
}

#[test]
fn sums_of_bools_are_counts() {
    let x = f32s(&[1.0, -2.0, 3.0, 4.0, -5.0, 6.0]);
    let m = x.gt(0.0);
    assert_eq!(m.sum().item().unwrap(), 4u32);
    assert_eq!(m.count().item().unwrap(), 4u32);
    let grid = Array::from_slice(&[1.0f32, -1.0, 2.0, 3.0, -4.0, -5.0], [3, 2]).unwrap().gt(0.0);
    assert_eq!(grid.sum_axis(0).to_vec().unwrap(), [2u32, 1]);
    assert_eq!(grid.sum_axis(1).to_vec().unwrap(), [2u32, 0, 1]);
    assert_eq!(m.cumsum(0).to_vec().unwrap(), [1u32, 1, 2, 3, 3, 4]);
    assert!(!m.min().item().unwrap());
    assert!(m.max().item().unwrap());
    assert_eq!(grid.max_axis(0).to_vec().unwrap(), [true, true]);
    assert_eq!(grid.min_axis(0).to_vec().unwrap(), [false, false]);
}

#[test]
fn bools_have_no_generators() {
    for range in [Array::<bool>::range(4, 0), Array::<bool>::range([4, 3], 1), Array::<bool>::range([8, 8], 0)] {
        assert!(matches!(range.map(|_| ()).unwrap_err().root(), LumaError::Unsupported(_)));
    }
}

#[test]
fn bool_views_gathers_and_scatters() {
    let m = Array::from_slice(&[true, false, false, true, true, false], [3, 2]).unwrap();
    assert_eq!(m.slice(0, 1..3).to_vec().unwrap(), [false, false, true, false]);
    assert_eq!(m.transpose().to_vec().unwrap(), [true, true, false, true, false, false]);
    let idx = Array::from_slice(&[5u32, 0, 3], 3).unwrap();
    assert_eq!(m.take(&idx).to_vec().unwrap(), [false, true, true]);
    let mut out = Array::<bool>::zeros(6).unwrap();
    out.put(&idx, Array::from_slice(&[true, true, true], 3).unwrap()).unwrap();
    assert_eq!(out.to_vec().unwrap(), [true, false, false, true, false, true]);
}
