//! `fir`, `iir` and `iir_sos` against sequential `f64` filters: filter lengths around the direct
//! kernel's and the FFT path's limits, signal lengths around the chunk sizes of the parallel
//! recursion, orders 1 to 8, batches, and errors.

use crate::common::values;
use luma::{Array, LumaError, IIR_MAX_ORDER};

fn fir_ref(x: &[f32], b: &[f32]) -> Vec<f64> {
    (0..x.len()).map(|i| b.iter().enumerate().filter(|(k, _)| *k <= i).map(|(k, &c)| f64::from(c) * f64::from(x[i - k])).sum()).collect()
}

fn iir_ref(x: &[f32], b: &[f32], a: &[f32]) -> Vec<f64> {
    let mut y = vec![0.0f64; x.len()];
    for i in 0..x.len() {
        let mut acc: f64 = b.iter().enumerate().filter(|(k, _)| *k <= i).map(|(k, &c)| f64::from(c) * f64::from(x[i - k])).sum();
        for (k, &c) in a.iter().enumerate().skip(1) {
            if i >= k {
                acc -= f64::from(c) * y[i - k];
            }
        }
        y[i] = acc / f64::from(a[0]);
    }
    y
}

/// `got` within `tol` of `want`, relative to the largest magnitude of `want` (filters add and
/// cancel, so a tolerance per element is too strict where the output is near 0).
fn close(got: &[f32], want: &[f64], tol: f64, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    let scale = want.iter().fold(1e-6f64, |m, w| m.max(w.abs()));
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!((f64::from(*g) - w).abs() <= tol * scale, "{what}: element {i}: got {g}, want {w} (scale {scale})");
    }
}

#[test]
fn fir_matches_the_reference_across_filter_lengths() {
    // 1 tap, short, around the direct kernel's 256-tap limit (blocks of taps), and the FFT path.
    for (n, k) in [(1, 1), (7, 3), (300, 17), (5000, 255), (5000, 256), (5000, 257), (3000, 700), (1500, 4096), (20000, 4100), (10, 40), (1, 300)] {
        let x = values(n, 1 + n as u64, -1.0, 1.0);
        let b = values(k, 2 + k as u64, -1.0, 1.0);
        let got = Array::from_slice(&x, n).unwrap().fir(&Array::from_slice(&b, k).unwrap());
        assert_eq!(got.shape().dims(), [n, 1, 1, 1]);
        close(&got.to_vec().unwrap(), &fir_ref(&x, &b), 2e-5, &format!("fir n={n} k={k}"));
    }
}

#[test]
fn fir_known_values() {
    let x = Array::from_slice(&[1.0f32, 4.0, 2.0, 8.0, 5.0, 7.0], 6).unwrap();
    let b = Array::from_slice(&[0.5f32, 0.25, 0.125], 3).unwrap();
    assert_eq!(x.fir(&b).to_vec().unwrap(), vec![0.5, 2.25, 2.125, 5.0, 4.75, 5.75]);
    // A filter longer than the signal.
    let long = Array::from_slice(&[1.0f32; 8], 8).unwrap();
    assert_eq!(x.fir(&long).to_vec().unwrap(), vec![1.0, 5.0, 7.0, 15.0, 20.0, 27.0]);
}

#[test]
fn fir_batches() {
    let (n, b) = (2500, 5);
    let x = values(n * b, 3, -1.0, 1.0);
    let xa = Array::from_slice(&x, [n, b]).unwrap();
    for k in [9, 300] {
        let taps = values(k * b, 4, -1.0, 1.0);
        // One filter for every signal.
        let got = xa.fir(&Array::from_slice(&taps[..k], k).unwrap()).to_vec().unwrap();
        let want: Vec<f64> = (0..b).flat_map(|j| fir_ref(&x[j * n..(j + 1) * n], &taps[..k])).collect();
        close(&got, &want, 2e-5, &format!("shared filter k={k}"));
        // One filter per signal.
        let got = xa.fir(&Array::from_slice(&taps, [k, b]).unwrap()).to_vec().unwrap();
        let want: Vec<f64> = (0..b).flat_map(|j| fir_ref(&x[j * n..(j + 1) * n], &taps[j * k..(j + 1) * k])).collect();
        close(&got, &want, 2e-5, &format!("filter per signal k={k}"));
        // One signal, one filter per batch entry.
        let one = Array::from_slice(&x[..n], n).unwrap();
        let got = one.fir(&Array::from_slice(&taps, [k, b]).unwrap());
        assert_eq!(got.shape().dims(), [n, b, 1, 1]);
        let want: Vec<f64> = (0..b).flat_map(|j| fir_ref(&x[..n], &taps[j * k..(j + 1) * k])).collect();
        close(&got.to_vec().unwrap(), &want, 2e-5, &format!("one signal k={k}"));
    }
    // A 3-D batch.
    let x3 = values(100 * 3 * 2, 5, -1.0, 1.0);
    let t = values(4, 6, -1.0, 1.0);
    let got = Array::from_slice(&x3, [100, 3, 2]).unwrap().fir(&Array::from_slice(&t, 4).unwrap()).to_vec().unwrap();
    let want: Vec<f64> = (0..6).flat_map(|j| fir_ref(&x3[j * 100..(j + 1) * 100], &t)).collect();
    close(&got, &want, 2e-5, "3-D batch");
}

#[test]
fn fir_long_filters_over_many_signals_use_blocks_of_taps() {
    // 20 signals (blocks of taps over the whole batch) and 2 signals (an FFT convolution each).
    for (batch, k) in [(20, 700), (20, 300), (2, 500), (3, 1000)] {
        let n = 1200;
        let x = values(n * batch, 20 + k as u64, -1.0, 1.0);
        let taps = values(k, 21, -1.0, 1.0);
        let got = Array::from_slice(&x, [n, batch]).unwrap().fir(&Array::from_slice(&taps, k).unwrap()).to_vec().unwrap();
        let want: Vec<f64> = (0..batch).flat_map(|j| fir_ref(&x[j * n..(j + 1) * n], &taps)).collect();
        close(&got, &want, 3e-5, &format!("batch {batch}, {k} taps"));
    }
}

#[test]
fn fir_errors() {
    let x = Array::from_slice(&values(50 * 4, 7, -1.0, 1.0), [50, 4]).unwrap();
    let mismatched = Array::from_slice(&values(5 * 3, 8, -1.0, 1.0), [5, 3]).unwrap();
    assert!(matches!(x.fir(&mismatched).to_vec(), Err(LumaError::Op { .. })));
    // A poisoned input poisons the result.
    let bad = Array::from_slice(&[1.0f32, 2.0], 2).unwrap() + &Array::from_slice(&[1.0f32, 2.0, 3.0], 3).unwrap();
    assert!(x.fir(&bad).to_vec().is_err());
    assert!(bad.fir(&Array::from_slice(&[1.0f32], 1).unwrap()).to_vec().is_err());
}

#[test]
fn iir_matches_the_reference() {
    // Signal lengths around a chunk (3072 samples), the chunks of one invocation of the carry kernel (64 chunks).
    const CHUNK: usize = 3072;
    let lengths = [1, 2, 5, 47, 48, 49, 3071, CHUNK, CHUNK + 1, 2 * CHUNK + 17, 64 * CHUNK, 64 * CHUNK + 1, 65 * CHUNK + 5, 150_000];
    let filters: [(&[f32], &[f32]); 4] = [
        (&[1.0], &[1.0, -0.5]),
        (&[0.2, 0.4, 0.2], &[1.0, -0.6, 0.2]),
        (&[0.5], &[1.0, -1.8, 0.9]),
        (&[0.1, 0.2, 0.3, 0.2, 0.1], &[1.0, -1.2, 0.9, -0.3, 0.1]),
    ];
    for &n in &lengths {
        let x = values(n, 10 + n as u64, -1.0, 1.0);
        let xa = Array::from_slice(&x, n).unwrap();
        for (b, a) in filters {
            let got = xa.iir(b, a);
            assert_eq!(got.shape().dims(), [n, 1, 1, 1]);
            close(&got.to_vec().unwrap(), &iir_ref(&x, b, a), 1e-4, &format!("iir n={n} b={b:?} a={a:?}"));
        }
    }
}

#[test]
fn iir_every_order() {
    let n = 20_000;
    let x = values(n, 11, -1.0, 1.0);
    let xa = Array::from_slice(&x, n).unwrap();
    for p in 1..=IIR_MAX_ORDER {
        // Poles inside the unit circle: the polynomial with roots 0.7 * e^(i k) -- built as a product of real factors.
        let mut a = vec![1.0f64];
        for k in 0..p {
            let r = 0.4 + 0.45 * (k as f64 / p as f64);
            let mut next = vec![0.0; a.len() + 1];
            for (i, v) in a.iter().enumerate() {
                next[i] += v;
                next[i + 1] -= r * v;
            }
            a = next;
        }
        let a: Vec<f32> = a.iter().map(|&v| v as f32).collect();
        let b = [0.3f32, 0.1];
        let got = xa.iir(&b, &a);
        close(&got.to_vec().unwrap(), &iir_ref(&x, &b, &a), 1e-3, &format!("order {p}"));
    }
}

#[test]
fn iir_poles_near_the_unit_circle() {
    // y[i] = x[i] + 0.999 y[i-1]: the chunk transitions are 0.999^3072; the error stays at f32 level.
    let n = 40_000;
    let x = values(n, 12, 0.0, 1.0);
    let got = Array::from_slice(&x, n).unwrap().iir(&[1.0], &[1.0, -0.999]).to_vec().unwrap();
    close(&got, &iir_ref(&x, &[1.0], &[1.0, -0.999]), 1e-4, "near the unit circle");
    // A resonator: a pole pair at radius 0.995.
    let (r, th) = (0.995f64, 0.1f64);
    let a = [1.0f32, (-2.0 * r * th.cos()) as f32, (r * r) as f32];
    let got = Array::from_slice(&x, n).unwrap().iir(&[1.0], &a).to_vec().unwrap();
    close(&got, &iir_ref(&x, &[1.0], &a), 3e-4, "resonator");
}

#[test]
fn iir_normalizes_by_a0_and_batches() {
    let (n, b) = (7000, 6);
    let x = values(n * b, 13, -1.0, 1.0);
    let (bc, ac) = ([0.4f32, 0.8, 0.4], [2.0f32, -0.5, 0.2]);
    let got = Array::from_slice(&x, [n, b]).unwrap().iir(&bc, &ac).to_vec().unwrap();
    let want: Vec<f64> = (0..b).flat_map(|j| iir_ref(&x[j * n..(j + 1) * n], &bc, &ac)).collect();
    close(&got, &want, 1e-4, "batched, a0 = 2");
    // A feedforward-only filter (a = [1]) is a FIR filter; more than 32 taps use the FIR kernel.
    let taps = values(40, 14, -1.0, 1.0);
    let xs = values(500, 15, -1.0, 1.0);
    let got = Array::from_slice(&xs, 500).unwrap().iir(&taps, &[1.0]).to_vec().unwrap();
    close(&got, &fir_ref(&xs, &taps), 2e-5, "a = [1]");
    // Impulse response of y[i] = x[i] + 0.5 y[i-1].
    let impulse = Array::from_slice(&[1.0f32, 0.0, 0.0, 0.0], 4).unwrap();
    assert_eq!(impulse.iir(&[1.0], &[1.0, -0.5]).to_vec().unwrap(), vec![1.0, 0.5, 0.25, 0.125]);
}

#[test]
fn iir_sos_is_the_cascade_of_its_sections() {
    let n = 30_000;
    let x = values(n, 16, -1.0, 1.0);
    let sections = [[0.2f32, 0.4, 0.2, 1.0, -0.5, 0.1], [1.0, 0.5, 0.0, 1.0, -0.7, 0.3], [0.5, 0.0, 0.0, 1.0, -0.2, 0.05]];
    let xa = Array::from_slice(&x, n).unwrap();
    let got = xa.iir_sos(&sections).to_vec().unwrap();
    let mut want: Vec<f64> = x.iter().map(|&v| f64::from(v)).collect();
    for s in sections {
        let input: Vec<f32> = want.iter().map(|&v| v as f32).collect();
        want = iir_ref(&input, &s[..3], &s[3..]);
    }
    close(&got, &want, 3e-4, "second-order sections");
    assert!(xa.iir_sos(&[]).to_vec().is_err());
}

#[test]
fn iir_errors() {
    let x = Array::from_slice(&[1.0f32, 2.0, 3.0], 3).unwrap();
    assert!(matches!(x.iir(&[1.0], &[0.0, 1.0]).to_vec(), Err(LumaError::Op { .. })), "a[0] = 0");
    assert!(x.iir(&[], &[1.0]).to_vec().is_err());
    assert!(x.iir(&[1.0], &[]).to_vec().is_err());
    assert!(x.iir(&[f32::NAN], &[1.0]).to_vec().is_err());
    assert!(x.iir(&[1.0], &[1.0, f32::INFINITY]).to_vec().is_err());
    assert!(x.iir(&[1.0], &[0.1; IIR_MAX_ORDER + 2]).to_vec().is_err(), "order too high");
    // NaN input propagates like arithmetic.
    let nan = Array::from_slice(&[1.0f32, f32::NAN, 1.0, 1.0], 4).unwrap();
    let got = nan.iir(&[1.0], &[1.0, -0.5]).to_vec().unwrap();
    assert_eq!(got[0], 1.0);
    assert!(got[1..].iter().all(|v| v.is_nan()));
}
