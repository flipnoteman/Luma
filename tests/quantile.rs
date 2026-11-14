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
