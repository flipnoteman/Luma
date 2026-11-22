//! `approx1` / `approx2`: every interpolation method against an `f64` reference written
//! from the formulas of the methods (not from the library's weights), with positions on, between
//! and outside the grid, NaN and infinite positions, batches, uniform grids and errors.

use crate::common::values;
use luma::{Array, Interp, LumaError};

const METHODS: [Interp; 6] = [Interp::Nearest, Interp::Lower, Interp::Linear, Interp::Cosine, Interp::Cubic, Interp::CubicSpline];

/// The interpolant of `method` at `x` of the samples `f(0..n)`, neighbours outside the grid
/// repeating the end samples.
fn axis(n: usize, x: f64, method: Interp, f: &dyn Fn(usize) -> f64) -> f64 {
    let at = |i: i64| f(i.clamp(0, n as i64 - 1) as usize);
    let i = x.floor() as i64;
    let t = x - x.floor();
    match method {
        Interp::Nearest => at((x + 0.5).floor() as i64),
        Interp::Lower => at(i),
        Interp::Linear => (1.0 - t) * at(i) + t * at(i + 1),
        Interp::Cosine => {
            let c = (1.0 - (t * std::f64::consts::PI).cos()) * 0.5;
            (1.0 - c) * at(i) + c * at(i + 1)
        }
        Interp::Cubic => {
            let (y0, y1, y2, y3) = (at(i - 1), at(i), at(i + 1), at(i + 2));
            let a0 = y3 - y2 - y0 + y1;
            let a1 = y0 - y1 - a0;
            let a2 = y2 - y0;
            a0 * t * t * t + a1 * t * t + a2 * t + y1
        }
        Interp::CubicSpline => {
            let (y0, y1, y2, y3) = (at(i - 1), at(i), at(i + 1), at(i + 2));
            let a0 = y1;
            let a1 = -0.5 * y0 + 0.5 * y2;
            let a2 = y0 - 2.5 * y1 + 2.0 * y2 - 0.5 * y3;
            let a3 = -0.5 * y0 + 1.5 * y1 - 1.5 * y2 + 0.5 * y3;
            a0 + a1 * t + a2 * t * t + a3 * t * t * t
        }
    }
}

fn inside(x: f32, n: usize) -> bool {
    x >= 0.0 && x <= (n - 1) as f32
}

fn close(got: &[f32], want: &[f64], tol: f64, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!((f64::from(*g) - w).abs() <= tol * (1.0 + w.abs()), "{what}: element {i}: got {g}, want {w}");
    }
}

#[test]
fn approx1_matches_the_reference_for_every_method() {
    let n = 40;
    let x = values(n, 1, -5.0, 5.0);
    let mut pos = values(500, 2, -3.0, n as f32 + 2.0);
    // Grid points, the ends, just outside, and a few special values.
    pos.extend([0.0, -0.0, 1.0, 5.5, (n - 1) as f32, (n - 1) as f32 - 0.25, -1e-6, (n - 1) as f32 + 1e-3, f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 1e30, -1e30]);
    let a = Array::from_slice(&x, n).unwrap();
    let p = Array::from_slice(&pos, pos.len()).unwrap();
    for method in METHODS {
        let got = a.approx1(&p, method, -9.0).to_vec().unwrap();
        let want: Vec<f64> = pos.iter().map(|&q| if inside(q, n) { axis(n, f64::from(q), method, &|i| f64::from(x[i])) } else { -9.0 }).collect();
        close(&got, &want, 2e-5, &format!("approx1 {method:?}"));
    }
}

#[test]
fn approx1_off_grid_value_is_used_for_nan_and_outside() {
    let a = Array::from_slice(&[1.0f32, 4.0, 2.0], 3).unwrap();
    let p = Array::from_slice(&[-0.1f32, 0.0, 2.0, 2.1, f32::NAN], 5).unwrap();
    let got = a.approx1(&p, Interp::Nearest, 7.5).to_vec().unwrap();
    assert_eq!(got, vec![7.5, 1.0, 2.0, 7.5, 7.5]);
}

#[test]
fn approx1_single_sample_and_two_samples() {
    let one = Array::from_slice(&[3.0f32], 1).unwrap();
    let p = Array::from_slice(&[0.0f32, 0.5, -0.5], 3).unwrap();
    for method in METHODS {
        assert_eq!(one.approx1(&p, method, 0.25).to_vec().unwrap(), vec![3.0, 0.25, 0.25], "{method:?}");
    }
    let two = Array::from_slice(&[1.0f32, 3.0], 2).unwrap();
    let p = Array::from_slice(&[0.5f32, 1.0], 2).unwrap();
    assert_eq!(two.approx1(&p, Interp::Linear, 0.0).to_vec().unwrap(), vec![2.0, 3.0]);
}

#[test]
fn approx1_batches_broadcast_and_follow_the_positions() {
    let (n, m, b) = (30, 25, 4);
    let x = values(n * b, 3, -2.0, 2.0);
    let pos = values(m * b, 4, -1.0, n as f32);
    let a = Array::from_slice(&x, [n, b]).unwrap();
    // One position vector shared by every batch.
    let shared = Array::from_slice(&pos[..m], m).unwrap();
    for method in [Interp::Linear, Interp::Cubic] {
        let got = a.approx1(&shared, method, 0.0).to_vec().unwrap();
        let mut want = vec![];
        for j in 0..b {
            for &q in &pos[..m] {
                want.push(if inside(q, n) { axis(n, f64::from(q), method, &|i| f64::from(x[j * n + i])) } else { 0.0 });
            }
        }
        close(&got, &want, 2e-5, "shared positions");
        // One position vector per batch.
        let per = Array::from_slice(&pos, [m, b]).unwrap();
        let got = a.approx1(&per, method, 0.0).to_vec().unwrap();
        let mut want = vec![];
        for j in 0..b {
            for &q in &pos[j * m..(j + 1) * m] {
                want.push(if inside(q, n) { axis(n, f64::from(q), method, &|i| f64::from(x[j * n + i])) } else { 0.0 });
            }
        }
        close(&got, &want, 2e-5, "batched positions");
    }
    // One signal, batched positions: the signal repeats.
    let one = Array::from_slice(&x[..n], n).unwrap();
    let per = Array::from_slice(&pos, [m, b]).unwrap();
    let got = one.approx1(&per, Interp::Linear, 0.0);
    assert_eq!(got.shape().dims(), [m, b, 1, 1]);
    let got = got.to_vec().unwrap();
    let want: Vec<f64> = pos.iter().map(|&q| if inside(q, n) { axis(n, f64::from(q), Interp::Linear, &|i| f64::from(x[i])) } else { 0.0 }).collect();
    close(&got, &want, 2e-5, "one signal");
    // Mismatching batches poison the result.
    let bad = Array::from_slice(&pos[..m * 3], [m, 3]).unwrap();
    assert!(matches!(a.approx1(&bad, Interp::Linear, 0.0).to_vec(), Err(e) if format!("{e}").contains("batch")), "mismatch");
}

#[test]
fn approx1_uniform_grid_and_other_dimensions() {
    let n = 30;
    let x = values(n, 5, -2.0, 2.0);
    let a = Array::from_slice(&x, n).unwrap();
    let pos = values(100, 6, -30.0, 30.0);
    let p = Array::from_slice(&pos, 100).unwrap();
    let (start, step) = (-10.0f32, 0.75f32);
    let got = a.approx1_uniform(&p, 0, start, step, Interp::Cubic, 1.5).to_vec().unwrap();
    let want: Vec<f64> = pos
        .iter()
        .map(|&q| {
            let g = (q - start) / step;
            if inside(g, n) { axis(n, f64::from(g), Interp::Cubic, &|i| f64::from(x[i])) } else { 1.5 }
        })
        .collect();
    close(&got, &want, 1e-4, "uniform");
    // Along dimension 1 of a [3, n] array.
    let m = Array::from_slice(&values(3 * n, 7, -1.0, 1.0), [3, n]).unwrap();
    let host = m.to_vec().unwrap();
    let p = Array::from_slice(&pos[..10], [1, 10]).unwrap();
    let got = m.approx1_uniform(&p, 1, 0.0, 1.0, Interp::Linear, 0.0);
    assert_eq!(got.shape().dims(), [3, 10, 1, 1]);
    let got = got.to_vec().unwrap();
    let mut want = vec![];
    for &q in &pos[..10] {
        for r in 0..3 {
            want.push(if inside(q, n) { axis(n, f64::from(q), Interp::Linear, &|j| f64::from(host[r + 3 * j])) } else { 0.0 });
        }
    }
    close(&got, &want, 2e-5, "dimension 1");
    // Invalid arguments.
    for bad in [a.approx1_uniform(&p, 4, 0.0, 1.0, Interp::Linear, 0.0), a.approx1_uniform(&p, 0, 0.0, 0.0, Interp::Linear, 0.0)] {
        assert!(matches!(bad.to_vec(), Err(LumaError::Op { .. })));
    }
}

#[test]
fn approx2_matches_the_reference_for_every_method() {
    let (w, h, c) = (13, 9, 3);
    let img = values(w * h * c, 8, -3.0, 3.0);
    let m = 300;
    let mut p0 = values(m, 9, -2.0, w as f32 + 1.0);
    let mut p1 = values(m, 10, -2.0, h as f32 + 1.0);
    // The corners, the edges, NaN and infinity in either coordinate.
    p0.extend([0.0, (w - 1) as f32, 0.0, (w - 1) as f32, f32::NAN, 3.0, f32::INFINITY, 2.5]);
    p1.extend([0.0, 0.0, (h - 1) as f32, (h - 1) as f32, 2.0, f32::NAN, 1.0, f32::NEG_INFINITY]);
    let n = p0.len();
    let a = Array::from_slice(&img, [w, h, c]).unwrap();
    let (a0, a1) = (Array::from_slice(&p0, n).unwrap(), Array::from_slice(&p1, n).unwrap());
    for method in METHODS {
        let got = a.approx2(&a0, &a1, method, -9.0);
        assert_eq!(got.shape().dims(), [n, 1, c, 1]);
        let got = got.to_vec().unwrap();
        let mut want = vec![];
        for ch in 0..c {
            for k in 0..n {
                want.push(if inside(p0[k], w) && inside(p1[k], h) {
                    axis(h, f64::from(p1[k]), method, &|j| axis(w, f64::from(p0[k]), method, &|i| f64::from(img[i + w * (j + h * ch)])))
                } else {
                    -9.0
                });
            }
        }
        close(&got, &want, 3e-5, &format!("approx2 {method:?}"));
    }
}

#[test]
fn approx2_with_a_grid_of_positions_and_a_uniform_grid() {
    let (w, h) = (11, 8);
    let img = values(w * h, 11, -3.0, 3.0);
    let a = Array::from_slice(&img, [w, h]).unwrap();
    let (q0, q1) = (6, 5);
    let p0 = values(q0 * q1, 12, 0.0, 12.0);
    let p1 = values(q0 * q1, 13, 0.0, 9.0);
    let (b0, b1) = (Array::from_slice(&p0, [q0, q1]).unwrap(), Array::from_slice(&p1, [q0, q1]).unwrap());
    let got = a.approx2(&b0, &b1, Interp::Cubic, 0.0);
    assert_eq!(got.shape().dims(), [q0, q1, 1, 1]);
    let got = got.to_vec().unwrap();
    let want: Vec<f64> = (0..q0 * q1)
        .map(|k| {
            if inside(p0[k], w) && inside(p1[k], h) {
                axis(h, f64::from(p1[k]), Interp::Cubic, &|j| axis(w, f64::from(p0[k]), Interp::Cubic, &|i| f64::from(img[i + w * j])))
            } else {
                0.0
            }
        })
        .collect();
    close(&got, &want, 3e-5, "grid of positions");

    // Grid points at start + step * i along dimensions 0 and 1.
    let got = a.approx2_uniform(&b0, &b1, [0, 1], [1.0, -2.0], [0.5, 2.0], Interp::Linear, 4.0).to_vec().unwrap();
    let want: Vec<f64> = (0..q0 * q1)
        .map(|k| {
            let (x, y) = ((p0[k] - 1.0) / 0.5, (p1[k] + 2.0) / 2.0);
            if inside(x, w) && inside(y, h) {
                axis(h, f64::from(y), Interp::Linear, &|j| axis(w, f64::from(x), Interp::Linear, &|i| f64::from(img[i + w * j])))
            } else {
                4.0
            }
        })
        .collect();
    close(&got, &want, 3e-5, "uniform 2-D");

    // Positions with different shapes, or equal dimensions: poisoned.
    let odd = Array::from_slice(&p1[..q0 * q1 - 1], q0 * q1 - 1).unwrap();
    assert!(a.approx2(&b0, &odd, Interp::Linear, 0.0).to_vec().is_err());
    assert!(a.approx2_uniform(&b0, &b1, [1, 1], [0.0; 2], [1.0; 2], Interp::Linear, 0.0).to_vec().is_err());
}

#[test]
fn approximation_reads_lazy_inputs_and_feeds_lazy_expressions() {
    // Positions computed from other arrays are evaluated first.
    let x = Array::from_slice(&[0.0f32, 10.0, 20.0, 30.0, 40.0], 5).unwrap();
    let pos = Array::<f32>::linspace(0.0, 4.0, 9).unwrap();
    let doubled = (x.approx1(&pos, Interp::Linear, 0.0) * 2.0).to_vec().unwrap();
    assert_eq!(doubled, vec![0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]);
}

#[test]
fn a_poisoned_input_poisons_the_result() {
    let good = Array::from_slice(&[1.0f32, 2.0], 2).unwrap();
    let bad = Array::from_slice(&[1.0f32, 2.0, 3.0], 3).unwrap() + &Array::from_slice(&[1.0f32, 2.0], 2).unwrap();
    assert!(good.approx1(&bad, Interp::Linear, 0.0).to_vec().is_err());
    assert!(bad.approx1(&good, Interp::Linear, 0.0).to_vec().is_err());
}

#[test]
fn a_batch_taller_than_the_grid_limit_runs_flat() {
    // 70000 signals: more rows than a grid dimension holds, so the kernel runs over a flat grid.
    let (n, b, m) = (8, 70_000, 5);
    let x = values(n * b, 14, -1.0, 1.0);
    let pos = [0.25f32, 1.5, 3.75, 6.0, 7.5];
    let got = Array::from_slice(&x, [n, b]).unwrap().approx1(&Array::from_slice(&pos, m).unwrap(), Interp::Linear, 9.0).to_vec().unwrap();
    assert_eq!(got.len(), m * b);
    for j in [0, 1, 4_321, 65_535, 65_536, b - 1] {
        for (k, &q) in pos.iter().enumerate() {
            let want = if inside(q, n) { axis(n, f64::from(q), Interp::Linear, &|i| f64::from(x[j * n + i])) } else { 9.0 };
            assert!((f64::from(got[j * m + k]) - want).abs() < 1e-5, "signal {j}, position {k}");
        }
    }
}

#[test]
fn sampling_reads_views_like_arrays() {
    let x = values(60, 42, -1.0, 1.0);
    let a = Array::from_slice(&x, 60).unwrap();
    let pos = Array::from_slice(&[0.5f32, 3.25, 10.0, 20.5, 39.0], 5).unwrap();
    let got = a.slice(0, 10..50).approx1(&pos, Interp::Linear, 0.0).to_vec().unwrap();
    let want: Vec<f64> = [0.5f32, 3.25, 10.0, 20.5, 39.0].iter().map(|&q| axis(40, f64::from(q), Interp::Linear, &|i| f64::from(x[10 + i]))).collect();
    close(&got, &want, 2e-5, "slice");
    // A view of the positions.
    let long = Array::from_slice(&[9.0f32, 9.0, 0.5, 3.25, 10.0], 5).unwrap();
    let got = a.approx1(&long.slice(0, 2..5), Interp::Linear, 0.0).to_vec().unwrap();
    let want: Vec<f64> = [0.5f32, 3.25, 10.0].iter().map(|&q| axis(60, f64::from(q), Interp::Linear, &|i| f64::from(x[i]))).collect();
    close(&got, &want, 2e-5, "position slice");
}
