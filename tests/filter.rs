//! `filter`, `nonzero`, their padded forms, `topk` and `bottomk` against host computations.

use luma::{Array, LumaError};

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
}

/// Sizes around the tile (8192 elements), the word (32) and a few tiles.
const SIZES: [usize; 11] = [1, 2, 31, 32, 33, 8191, 8192, 8193, 16384, 100_003, 1 << 20];

fn mask(rng: &mut Rng, n: usize, per_mille: u32) -> Vec<bool> {
    (0..n).map(|_| rng.next() % 1000 < per_mille).collect()
}

#[test]
fn filter_f32_matches_host() {
    for &n in &SIZES {
        for per_mille in [0, 10, 500, 990, 1000] {
            let mut rng = Rng(n as u64 * 31 + per_mille as u64);
            let vals: Vec<f32> = (0..n).map(|_| (rng.next() % 100_000) as f32 * 0.25).collect();
            let keep = mask(&mut rng, n, per_mille);
            let x = Array::from_slice(&vals, n).unwrap();
            let m = Array::from_slice(&keep, n).unwrap();
            let want = where_keep(&vals, &keep);
            match x.filter(&m).unwrap() {
                None => assert!(want.is_empty(), "n={n} p={per_mille}"),
                Some(got) => assert_eq!(got.to_vec().unwrap(), want, "n={n} p={per_mille}"),
            }
        }
    }
}

#[test]
fn filter_of_pending_expressions_and_u32_masks() {
    let n = 50_000usize;
    let vals: Vec<f32> = (0..n).map(|i| (i % 97) as f32 - 40.0).collect();
    let x = Array::from_slice(&vals, n).unwrap();
    // Pending operand and pending mask (a comparison).
    let y = &x * 2.0;
    let got = y.filter(&x.gt(0.0)).unwrap().unwrap().to_vec().unwrap();
    let want: Vec<f32> = vals.iter().filter(|&&v| v > 0.0).map(|&v| v * 2.0).collect();
    assert_eq!(got, want);
    // A u32 mask: non-zero selects.
    let words: Vec<u32> = (0..n as u32).map(|i| i % 5).collect();
    let w = Array::from_slice(&words, n).unwrap();
    let got = x.filter(&w).unwrap().unwrap().to_vec().unwrap();
    let want: Vec<f32> = vals.iter().zip(&words).filter(|&(_, &k)| k != 0).map(|(&v, _)| v).collect();
    assert_eq!(got, want);
}

#[test]
fn filter_other_element_types() {
    let n = 20_000usize;
    let mut rng = Rng(7);
    let keep = mask(&mut rng, n, 300);
    let m = Array::from_slice(&keep, n).unwrap();

    let u: Vec<u32> = (0..n).map(|_| rng.next()).collect();
    assert_eq!(Array::from_slice(&u, n).unwrap().filter(&m).unwrap().unwrap().to_vec().unwrap(), where_keep(&u, &keep));
    let i: Vec<i32> = (0..n).map(|_| rng.next() as i32).collect();
    assert_eq!(Array::from_slice(&i, n).unwrap().filter(&m).unwrap().unwrap().to_vec().unwrap(), where_keep(&i, &keep));
    let b = mask(&mut rng, n, 400);
    assert_eq!(Array::from_slice(&b, n).unwrap().filter(&m).unwrap().unwrap().to_vec().unwrap(), where_keep(&b, &keep));
    let d: Vec<f64> = (0..n).map(|_| rng.next() as f64 * 1.5).collect();
    if let Ok(x) = Array::from_slice(&d, n) {
        assert_eq!(x.filter(&m).unwrap().unwrap().to_vec().unwrap(), where_keep(&d, &keep));
    }
}

fn where_keep<T: Copy>(v: &[T], keep: &[bool]) -> Vec<T> {
    v.iter().zip(keep).filter(|&(_, &k)| k).map(|(&v, _)| v).collect()
}

#[test]
fn filter_multidimensional_is_flat() {
    let vals: Vec<u32> = (0..6 * 5 * 4).collect();
    let x = Array::from_slice(&vals, [6, 5, 4]).unwrap();
    let got = x.filter(&x.ge(100u32)).unwrap().unwrap();
    assert_eq!(got.shape().dims(), [20, 1, 1, 1]);
    assert_eq!(got.to_vec().unwrap(), (100..120).collect::<Vec<u32>>());
}

#[test]
fn filter_errors() {
    let x = Array::<f32>::from_slice(&[1.0, 2.0, 3.0], 3).unwrap();
    let m = Array::from_slice(&[true, false], 2).unwrap();
    assert!(matches!(x.filter(&m), Err(LumaError::ShapeMismatch { .. })));
    let bad = Array::<f32>::from_slice(&[1.0, 2.0], 2).unwrap().reshape(5);
    assert!(x.filter(&bad.gt(0.0)).is_err());
}

#[test]
fn filter_padded_matches_host() {
    for &n in &[1usize, 31, 8193, 100_003] {
        for per_mille in [0, 500, 1000] {
            let mut rng = Rng(n as u64 + 99 * per_mille as u64);
            let vals: Vec<f32> = (0..n).map(|_| (rng.next() % 1000) as f32).collect();
            let keep = mask(&mut rng, n, per_mille);
            let x = Array::from_slice(&vals, n).unwrap();
            let m = Array::from_slice(&keep, n).unwrap();
            let (padded, count) = x.filter_padded(&m, -1.0);
            let kept = where_keep(&vals, &keep);
            assert_eq!(count.item().unwrap() as usize, kept.len());
            let mut want = kept.clone();
            want.resize(n, -1.0);
            assert_eq!(padded.to_vec().unwrap(), want, "n={n} p={per_mille}");
        }
    }
    // bool values
    let keep = [true, false, true, true, false];
    let vals = [false, true, true, false, true];
    let (padded, count) = Array::from_slice(&vals, 5).unwrap().filter_padded(&Array::from_slice(&keep, 5).unwrap(), true);
    assert_eq!(padded.to_vec().unwrap(), vec![false, true, false, true, true]);
    assert_eq!(count.item().unwrap(), 3);
}

#[test]
fn nonzero_matches_host() {
    for &n in &SIZES {
        for per_mille in [0, 20, 500, 1000] {
            let mut rng = Rng(n as u64 * 17 + per_mille as u64);
            let keep = mask(&mut rng, n, per_mille);
            let m = Array::from_slice(&keep, n).unwrap();
            let want: Vec<u32> = (0..n as u32).filter(|&i| keep[i as usize]).collect();
            match m.nonzero().unwrap() {
                None => assert!(want.is_empty()),
                Some(got) => assert_eq!(got.to_vec().unwrap(), want, "n={n} p={per_mille}"),
            }
            let (padded, count) = m.nonzero_padded(u32::MAX);
            assert_eq!(count.item().unwrap() as usize, want.len());
            let mut padded_want = want.clone();
            padded_want.resize(n, u32::MAX);
            assert_eq!(padded.to_vec().unwrap(), padded_want);
        }
    }
    // u32 masks
    let words: Vec<u32> = (0..1000u32).map(|i| i % 3).collect();
    let got = Array::from_slice(&words, 1000).unwrap().nonzero().unwrap().unwrap().to_vec().unwrap();
    assert_eq!(got, (0..1000u32).filter(|i| i % 3 != 0).collect::<Vec<_>>());
    // Index math of an N-D array.
    let x = Array::from_slice(&(0..24u32).collect::<Vec<_>>(), [2, 3, 4]).unwrap();
    let idx = x.eq(17u32).nonzero().unwrap().unwrap().to_vec().unwrap();
    assert_eq!(idx, vec![17]);
    assert_eq!([idx[0] % 2, idx[0] / 2 % 3, idx[0] / 6 % 4], [1, 2, 2]);
}

#[test]
fn topk_and_bottomk_match_host() {
    let n = 100_000usize;
    let mut rng = Rng(5);
    let f: Vec<f32> = (0..n).map(|i| if i % 5001 == 7 { f32::NAN } else { (rng.next() % 5000) as f32 }).collect();
    let x = Array::from_slice(&f, n).unwrap();
    for k in [1usize, 10, 777, n] {
        let (v, i) = x.topk(k);
        let (v, i) = (v.to_vec().unwrap(), i.to_vec().unwrap());
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| f[b].total_cmp(&f[a]).then(a.cmp(&b)));
        let want_i: Vec<u32> = order[..k].iter().map(|&j| j as u32).collect();
        assert_eq!(i, want_i, "k={k}");
        for (got, &j) in v.iter().zip(&want_i) {
            assert!(got.to_bits() == f[j as usize].to_bits() || (got.is_nan() && f[j as usize].is_nan()));
        }
        let (v, i) = x.bottomk(k);
        let (v, i) = (v.to_vec().unwrap(), i.to_vec().unwrap());
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| f[a].total_cmp(&f[b]).then(a.cmp(&b)));
        let want_i: Vec<u32> = order[..k].iter().map(|&j| j as u32).collect();
        assert_eq!(i, want_i, "bottomk k={k}");
        assert_eq!(v.len(), k);
    }
    // i32 and u32 keys, and a 2-D array (flat indices).
    let g = Array::from_slice(&[3i32, -5, 9, 9, 0, -5], [3, 2]).unwrap();
    let (v, i) = g.topk(3);
    assert_eq!(v.to_vec().unwrap(), vec![9, 9, 3]);
    assert_eq!(i.to_vec().unwrap(), vec![2, 3, 0]);
    let (v, i) = Array::from_slice(&[4u32, 1, 7], 3).unwrap().bottomk(2);
    assert_eq!((v.to_vec().unwrap(), i.to_vec().unwrap()), (vec![1, 4], vec![1, 0]));
    // Bad k poisons.
    assert!(x.topk(0).0.to_vec().is_err());
    assert!(x.topk(n + 1).1.to_vec().is_err());
}
