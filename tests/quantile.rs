//! `quantile`, `median`, `quantile_axis` and `median_axis` against NumPy's linear method on host.

use luma::Array;

fn host_quantile(sorted: &[f32], q: f32) -> f32 {
    let h = f64::from(q) * (sorted.len() - 1) as f64;
    let (lo, hi) = (h.floor() as usize, h.ceil() as usize);
    let (a, b) = (f64::from(sorted[lo]), f64::from(sorted[hi]));
    (a + (h - h.floor()) * (b - a)) as f32
}

fn data(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 33) % 20001) as f32 / 10.0 - 1000.0
        })
        .collect()
}

fn close(got: f32, want: f32) {
    assert!((got - want).abs() <= 1e-6 * want.abs().max(1.0), "{got} vs {want}");
}

#[test]
fn flat_quantiles() {
    for n in [1usize, 2, 1001, 1 << 20] {
        let v = data(n, n as u64);
        let mut sorted = v.clone();
        sorted.sort_by(f32::total_cmp);
        let x = Array::from_slice(&v, n).unwrap();
        for q in [0.0f32, 0.25, 0.5, 1.0] {
            close(x.quantile(q).item().unwrap(), host_quantile(&sorted, q));
        }
        close(x.median().item().unwrap(), host_quantile(&sorted, 0.5));
    }
}

#[test]
fn axis_quantiles() {
    let (k, lines) = (37usize, 500usize);
    let v = data(k * lines, 9);
    let x = Array::from_slice(&v, [k, lines]).unwrap();
    for q in [0.0f32, 0.25, 0.5, 1.0] {
        let got = x.quantile_axis(0, q);
        assert_eq!(got.shape().dims(), [1, lines, 1, 1]);
        let got = got.to_vec().unwrap();
        for l in 0..lines {
            let mut line = v[l * k..(l + 1) * k].to_vec();
            line.sort_by(f32::total_cmp);
            close(got[l], host_quantile(&line, q));
        }
    }
    let got = x.median_axis(1).to_vec().unwrap();
    assert_eq!(got.len(), k);
    for i in 0..k {
        let mut line: Vec<f32> = (0..lines).map(|l| v[i + k * l]).collect();
        line.sort_by(f32::total_cmp);
        close(got[i], host_quantile(&line, 0.5));
    }
}

#[test]
fn invalid_q_poisons() {
    let x = Array::<f32>::full(4, 1.0).unwrap();
    assert!(x.quantile(1.5).item().is_err());
    assert!(x.quantile(f32::NAN).item().is_err());
    assert!(x.quantile_axis(7, 0.5).to_vec().is_err());
}

#[test]
fn nan_sorts_last() {
    let x = Array::from_slice(&[1.0, f32::NAN, 3.0, 2.0], 4).unwrap();
    close(x.quantile(0.0).item().unwrap(), 1.0);
    assert!(x.quantile(1.0).item().unwrap().is_nan());
}

/// The same values as `data` but with many ties and, if `nan`, a NaN in every seventh place.
fn tied(n: usize, seed: u64, nan: bool) -> Vec<f32> {
    data(n, seed).into_iter().enumerate().map(|(i, v)| if nan && i % 7 == 3 { f32::NAN } else { (v / 250.0).round() }).collect()
}

fn same(got: f32, want: f32, what: &str) {
    assert!(got == want || (got.is_nan() && want.is_nan()), "{what}: {got} vs {want}");
}

const QS: [f32; 8] = [0.0, 0.1, 0.25, 0.5, 0.5, 0.75, 0.999, 1.0];

#[test]
fn flat_quantiles_agree_with_one_call_each() {
    for n in [1usize, 2, 3, 4, 5, 1001, 40_000, 100_001] {
        for nan in [false, true] {
            let x = Array::from_slice(&tied(n, n as u64, nan), n).unwrap();
            let all = x.quantiles(&QS);
            assert_eq!(all.shape().dims(), [QS.len(), 1, 1, 1]);
            let all = all.to_vec().unwrap();
            for (j, &q) in QS.iter().enumerate() {
                same(all[j], x.quantile(q).item().unwrap(), &format!("n {n} nan {nan} q {q}"));
            }
            // A single quantile is the same call.
            same(x.quantiles(&[0.3]).item().unwrap(), x.quantile(0.3).item().unwrap(), "one");
        }
    }
}

#[test]
fn flat_quantiles_match_the_host() {
    let n = 1001;
    let v = data(n, 5);
    let mut sorted = v.clone();
    sorted.sort_by(f32::total_cmp);
    let x = Array::from_slice(&v, n).unwrap();
    let got = x.quantiles(&[0.0, 0.125, 0.333, 0.5, 0.9, 1.0]).to_vec().unwrap();
    for (g, q) in got.iter().zip([0.0f32, 0.125, 0.333, 0.5, 0.9, 1.0]) {
        close(*g, host_quantile(&sorted, q));
    }
}

#[test]
fn axis_quantiles_agree_with_one_call_each() {
    let shapes: [([usize; 4], usize); 8] = [
        ([37, 500, 1, 1], 0),
        ([37, 500, 1, 1], 1),
        ([5, 7, 11, 1], 0),
        ([5, 7, 11, 1], 1),
        ([5, 7, 11, 1], 2),
        ([4, 3, 2, 9], 3),
        ([3000, 6, 1, 1], 0),
        ([6, 3000, 1, 1], 1),
    ];
    for (k, &(dims, axis)) in shapes.iter().enumerate() {
        for nan in [false, true] {
            let len: usize = dims.iter().product();
            let x = Array::from_slice(&tied(len, 40 + k as u64, nan), dims).unwrap();
            let all = x.quantiles_axis(axis, &QS);
            let mut want_dims = dims;
            want_dims[axis] = QS.len();
            assert_eq!(all.shape().dims(), want_dims);
            for (j, &q) in QS.iter().enumerate() {
                let part = all.slice(axis, j..j + 1).to_vec().unwrap();
                let one = x.quantile_axis(axis, q).to_vec().unwrap();
                assert_eq!(part.len(), one.len());
                for (i, (a, b)) in part.iter().zip(&one).enumerate() {
                    same(*a, *b, &format!("dims {dims:?} axis {axis} nan {nan} q {q} line {i}"));
                }
            }
        }
    }
}

#[test]
fn axis_quantiles_match_the_host() {
    let (k, lines) = (37usize, 500usize);
    let v = data(k * lines, 9);
    let x = Array::from_slice(&v, [k, lines]).unwrap();
    let qs = [0.05f32, 0.5, 0.95];
    let got = x.quantiles_axis(0, &qs).to_vec().unwrap();
    for l in 0..lines {
        let mut line = v[l * k..(l + 1) * k].to_vec();
        line.sort_by(f32::total_cmp);
        for (j, &q) in qs.iter().enumerate() {
            close(got[l * qs.len() + j], host_quantile(&line, q));
        }
    }
    let got = x.quantiles_axis(1, &qs).to_vec().unwrap();
    for i in 0..k {
        let mut line: Vec<f32> = (0..lines).map(|l| v[i + k * l]).collect();
        line.sort_by(f32::total_cmp);
        for (j, &q) in qs.iter().enumerate() {
            close(got[i + k * j], host_quantile(&line, q));
        }
    }
}

#[test]
fn exact_positions_are_not_interpolated_through_infinities() {
    let x = Array::from_slice(&[f32::INFINITY, 1.0, f32::INFINITY, f32::INFINITY, 2.0], 5).unwrap();
    // Sorted: 1, 2, inf, inf, inf; 0.5 and 0.75 are exact positions (2 and 3).
    let got = x.quantiles(&[0.5, 0.75, 0.9, 0.25]).to_vec().unwrap();
    assert_eq!(got[0], f32::INFINITY);
    assert_eq!(got[1], f32::INFINITY);
    same(got[2], x.quantile(0.9).item().unwrap(), "0.9");
    assert_eq!(got[3], 2.0);
}

#[test]
fn bad_quantile_requests_poison() {
    let x = Array::<f32>::full([4, 3], 1.0).unwrap();
    assert!(x.quantiles(&[0.5, 1.5]).to_vec().is_err());
    assert!(x.quantiles(&[0.5, f32::NAN]).to_vec().is_err());
    assert!(x.quantiles(&[]).to_vec().is_err());
    assert!(x.quantiles_axis(1, &[0.1, -0.1]).to_vec().is_err());
    assert!(x.quantiles_axis(1, &[]).to_vec().is_err());
    assert!(x.quantiles_axis(4, &[0.5]).to_vec().is_err());
}
