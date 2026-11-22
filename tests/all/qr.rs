//! `qr`, `Qr::q` / `q_thin` / `r` / `apply_q` / `apply_qt` and `lstsq` against `f64` host
//! checks: tall, square and thin shapes around the panel and recursion thresholds, batches,
//! rank-deficient and ill-conditioned matrices, least squares.

use crate::common::values;
use crate::dense_host::*;
use luma::{Array, Context};

fn mat(ctx: &Context, data: &[f32], cols: usize, rows: usize, batch: usize) -> Array<f32> {
    Array::from_slice_on(ctx, data, [cols, rows, batch]).unwrap()
}

const SHAPES: [(usize, usize); 15] = [(1, 1), (2, 2), (3, 5), (8, 8), (31, 31), (32, 32), (33, 33), (40, 64), (64, 64), (100, 100), (50, 200), (129, 300), (257, 257), (40, 1000), (200, 200)];

/// `||Q R - A||_F / ||A||_F` and `||Qᵀ Q - I||_F` for the thin factors (`m x n`).
fn check_thin(q: &[f32], r: &[f32], a: &[f32], m: usize, n: usize) -> (f64, f64) {
    let (q, r, a) = (widen(q), widen(r), widen(a));
    for i in 0..n {
        for j in 0..i {
            assert_eq!(r[i * n + j], 0.0, "R below the diagonal at ({i}, {j})");
        }
    }
    let qr = mul(&q, &r, m, n, n);
    let qtq = mul(&transpose(&q, m, n), &q, n, m, n);
    let mut eye = vec![0.0; n * n];
    for i in 0..n {
        eye[i * n + i] = 1.0;
    }
    (diff(&qr, &a) / fro(&a).max(1e-30), diff(&qtq, &eye))
}

#[test]
fn qr_factors_every_shape() {
    let ctx = Context::new().unwrap();
    for &(n, m) in &SHAPES {
        if m < n {
            continue;
        }
        let a = values(n * m, 7 + (n * 31 + m) as u64, -1.0, 1.0);
        let f = mat(&ctx, &a, n, m, 1).qr();
        assert_eq!(f.qr.shape().dims(), [n, m, 1, 1]);
        assert_eq!(f.tau.shape().dims(), [n, 1, 1, 1]);
        assert_eq!(f.t.shape().dims(), [n, n, 1, 1]);
        let (q, r) = (f.q_thin(), f.r());
        assert_eq!(q.shape().dims(), [n, m, 1, 1]);
        assert_eq!(r.shape().dims(), [n, n, 1, 1]);
        let (err, orth) = check_thin(&q.to_vec().unwrap(), &r.to_vec().unwrap(), &a, m, n);
        let tol = 5e-6 * ((m * n) as f64).sqrt().max(1.0).sqrt() * 4.0;
        assert!(err < tol, "{m}x{n}: ||QR - A|| / ||A|| = {err}");
        assert!(orth < tol * 2.0, "{m}x{n}: ||QtQ - I|| = {orth}");
    }
}

#[test]
fn the_full_q_is_orthogonal_and_applies() {
    let ctx = Context::new().unwrap();
    for &(n, m) in &[(3usize, 7usize), (20, 20), (40, 100), (70, 150)] {
        let a = values(n * m, 3, -1.0, 1.0);
        let f = mat(&ctx, &a, n, m, 1).qr();
        let q = widen(&f.q().to_vec().unwrap());
        let qtq = mul(&transpose(&q, m, m), &q, m, m, m);
        let mut eye = vec![0.0; m * m];
        for i in 0..m {
            eye[i * m + i] = 1.0;
        }
        assert!(diff(&qtq, &eye) < 2e-5 * (m as f64).sqrt(), "{m}x{n}: full Q not orthogonal");
        // Qᵀ A = [R; 0].
        let qta = f.apply_qt(&mat(&ctx, &a, n, m, 1)).to_vec().unwrap();
        let r = f.r().to_vec().unwrap();
        for i in 0..m {
            for j in 0..n {
                let want = if i < n { r[i * n + j] } else { 0.0 };
                assert!((qta[i * n + j] - want).abs() < 2e-5 * (m as f32).sqrt(), "{m}x{n}: QtA at ({i}, {j}) = {}, want {want}", qta[i * n + j]);
            }
        }
        // Q (Qᵀ b) = b.
        let b = values(m * 2, 4, -1.0, 1.0);
        let back = f.apply_q(&f.apply_qt(&mat(&ctx, &b, 2, m, 1))).to_vec().unwrap();
        for (x, y) in back.iter().zip(&b) {
            assert!((x - y).abs() < 2e-5 * (m as f32).sqrt(), "{m}x{n}: Q Qt b");
        }
    }
}

#[test]
fn qr_batches() {
    let ctx = Context::new().unwrap();
    for &(n, m, batch) in &[(1usize, 1usize, 5usize), (4, 4, 100), (8, 12, 50), (16, 16, 64), (32, 40, 9), (33, 50, 3), (70, 70, 2)] {
        let mut data = Vec::new();
        for b in 0..batch {
            data.extend(values(n * m, 11 + (n * 13 + b) as u64, -1.0, 1.0));
        }
        let f = mat(&ctx, &data, n, m, batch).qr();
        assert_eq!(f.tau.shape().dims(), [n, 1, batch, 1]);
        let (q, r) = (f.q_thin().to_vec().unwrap(), f.r().to_vec().unwrap());
        for b in 0..batch {
            let (err, orth) = check_thin(&q[b * m * n..(b + 1) * m * n], &r[b * n * n..(b + 1) * n * n], &data[b * m * n..(b + 1) * m * n], m, n);
            assert!(err < 5e-5 && orth < 5e-5, "{m}x{n} matrix {b}: {err}, {orth}");
        }
    }
}

#[test]
fn rank_deficient_and_degenerate_inputs() {
    let ctx = Context::new().unwrap();
    // A zero column in the middle, a zero matrix, an already triangular matrix.
    for &(n, m) in &[(5usize, 9usize), (40, 40), (100, 130)] {
        let mut a = values(n * m, 1, -1.0, 1.0);
        for i in 0..m {
            a[i * n + n / 2] = 0.0;
        }
        let f = mat(&ctx, &a, n, m, 1).qr();
        let (q, r) = (f.q_thin().to_vec().unwrap(), f.r().to_vec().unwrap());
        let (err, orth) = check_thin(&q, &r, &a, m, n);
        assert!(err < 3e-5 && orth < 3e-5, "{m}x{n}: {err}, {orth}");
        assert!(r[(n / 2) * n + n / 2].abs() < 1e-5);
    }
    let z = mat(&ctx, &[0.0f32; 12], 3, 4, 1).qr();
    assert!(z.r().to_vec().unwrap().iter().all(|v| *v == 0.0));
    let n = 6;
    let mut tri = vec![0.0f32; n * n];
    for i in 0..n {
        for j in i..n {
            tri[i * n + j] = 1.0 + (i * 3 + j) as f32;
        }
    }
    let f = mat(&ctx, &tri, n, n, 1).qr();
    let (err, orth) = check_thin(&f.q_thin().to_vec().unwrap(), &f.r().to_vec().unwrap(), &tri, n, n);
    assert!(err < 1e-5 && orth < 1e-5);
    let nan = mat(&ctx, &[1.0f32, f32::NAN, 2.0, 3.0], 2, 2, 1).qr().r().to_vec().unwrap();
    assert!(nan.iter().any(|v| v.is_nan()));
}

#[test]
fn lstsq_solves_least_squares() {
    let ctx = Context::new().unwrap();
    for &(n, m, k) in &[(1usize, 3usize, 1usize), (3, 10, 2), (10, 10, 1), (32, 100, 3), (33, 200, 5), (70, 70, 2), (100, 400, 4), (150, 150, 1)] {
        let a = values(n * m, 20 + (n * 7 + m) as u64, -1.0, 1.0);
        let b = values(k * m, 21, -1.0, 1.0);
        let x = mat(&ctx, &a, n, m, 1).lstsq(&mat(&ctx, &b, k, m, 1));
        assert_eq!(x.shape().dims(), [k, n, 1, 1]);
        let x = widen(&x.to_vec().unwrap());
        // The normal equations in f64 as the reference.
        let ad = widen(&a);
        let at = transpose(&ad, m, n);
        let ata = mul(&at, &ad, n, m, n);
        let atb = mul(&at, &widen(&b), n, m, k);
        let want = solve(&ata, &atb, n, k);
        assert!(diff(&x, &want) / fro(&want) < 2e-3, "{m}x{n}: forward error {}", diff(&x, &want) / fro(&want));
        // The residual is orthogonal to the columns of A.
        let r: Vec<f64> = mul(&ad, &x, m, n, k).iter().zip(&widen(&b)).map(|(p, q)| p - q).collect();
        let atr = mul(&at, &r, n, m, k);
        assert!(fro(&atr) < 2e-5 * fro(&ad) * fro(&r) + 1e-5 * fro(&ad) * fro(&ad) * fro(&x), "{m}x{n}: Aᵀ r = {}", fro(&atr));
    }
}

#[test]
fn lstsq_ill_conditioned_stays_backward_stable() {
    let ctx = Context::new().unwrap();
    // Columns scaled over 4 decades: conditioning 1e4, where the normal equations would square it.
    let (n, m) = (40usize, 120usize);
    let mut a = values(n * m, 5, -1.0, 1.0);
    for i in 0..m {
        for j in 0..n {
            a[i * n + j] *= 10f32.powf(-4.0 * j as f32 / (n - 1) as f32);
        }
    }
    let b = values(m, 6, -1.0, 1.0);
    let x = widen(&mat(&ctx, &a, n, m, 1).lstsq(&mat(&ctx, &b, 1, m, 1)).to_vec().unwrap());
    let ad = widen(&a);
    let r: Vec<f64> = mul(&ad, &x, m, n, 1).iter().zip(&widen(&b)).map(|(p, q)| p - q).collect();
    let atr = mul(&transpose(&ad, m, n), &r, n, m, 1);
    assert!(fro(&atr) / (fro(&ad) * fro(&r)) < 1e-5);
}

#[test]
fn lstsq_batches_and_shape_errors() {
    let ctx = Context::new().unwrap();
    let (n, m, batch) = (4usize, 9usize, 20usize);
    let (mut a, mut b) = (Vec::new(), Vec::new());
    for i in 0..batch {
        a.extend(values(n * m, 50 + i as u64, -1.0, 1.0));
        b.extend(values(m, 80 + i as u64, -1.0, 1.0));
    }
    let x = mat(&ctx, &a, n, m, batch).lstsq(&mat(&ctx, &b, 1, m, batch)).to_vec().unwrap();
    for i in 0..batch {
        let ad = widen(&a[i * n * m..(i + 1) * n * m]);
        let xd = widen(&x[i * n..(i + 1) * n]);
        let r: Vec<f64> = mul(&ad, &xd, m, n, 1).iter().zip(&widen(&b[i * m..(i + 1) * m])).map(|(p, q)| p - q).collect();
        let atr = mul(&transpose(&ad, m, n), &r, n, m, 1);
        assert!(fro(&atr) < 3e-5 * fro(&ad) * fro(&r), "matrix {i}");
    }
    let wide = Array::<f32>::zeros_on(&ctx, [4, 3]).unwrap();
    assert!(wide.qr().qr.to_vec().is_err());
    let tall = Array::<f32>::zeros_on(&ctx, [3, 4]).unwrap();
    assert!(tall.lstsq(&Array::<f32>::zeros_on(&ctx, [1, 5]).unwrap()).to_vec().is_err());
}

#[test]
fn solve_handles_tall_and_wide_systems() {
    let ctx = Context::new().unwrap();
    // Tall: least squares, the same as `lstsq`.
    let (n, m) = (20usize, 50usize);
    let a = values(n * m, 1, -1.0, 1.0);
    let b = values(m * 2, 2, -1.0, 1.0);
    let (al, bl) = (mat(&ctx, &a, n, m, 1), mat(&ctx, &b, 2, m, 1));
    assert_eq!(al.solve(&bl).to_vec().unwrap(), al.lstsq(&bl).to_vec().unwrap());
    // Wide: the minimum norm solution of an underdetermined system: A x = b exactly and x in the
    // row space of A (x = Aᵀ y).
    for &(n, m) in &[(5usize, 3usize), (40, 20), (100, 60)] {
        let a = values(n * m, 3 + n as u64, -1.0, 1.0);
        let b = values(m * 2, 4, -1.0, 1.0);
        let x = widen(&mat(&ctx, &a, n, m, 1).solve(&mat(&ctx, &b, 2, m, 1)).to_vec().unwrap());
        let ad = widen(&a);
        let ax = mul(&ad, &x, m, n, 2);
        assert!(diff(&ax, &widen(&b)) < 3e-5 * (fro(&ad) * fro(&x)).max(1.0), "{m}x{n}: A x = b");
        // x = Aᵀ (A Aᵀ)⁻¹ b in f64.
        let aat = mul(&ad, &transpose(&ad, m, n), m, n, m);
        let y = solve(&aat, &widen(&b), m, 2);
        let want = mul(&transpose(&ad, m, n), &y, n, m, 2);
        assert!(diff(&x, &want) / fro(&want) < 2e-3, "{m}x{n}: minimum norm differs by {}", diff(&x, &want) / fro(&want));
    }
}

/// Tall panels are factored by blocks of rows and a second factorisation of their `R`s: shapes
/// around the block size, a last block of fewer rows than columns, many blocks, a panel of one
/// column, scaled columns.
#[test]
fn tall_matrices_are_cut_into_blocks() {
    let ctx = Context::new().unwrap();
    for &(n, m, scale) in &[(32usize, 600usize, false), (32, 520, false), (32, 513, false), (1, 2000, false), (17, 1100, false), (40, 1100, false), (70, 3000, false), (32, 4100, true), (32, 8192, false), (32, 9000, false), (33, 2048, true)] {
        let mut a = values(n * m, 91 + (n * 13 + m) as u64, -1.0, 1.0);
        if scale {
            for i in 0..m {
                for j in 0..n {
                    a[i * n + j] *= 10f32.powf(-3.0 * j as f32 / (n - 1) as f32);
                }
            }
        }
        let f = mat(&ctx, &a, n, m, 1).qr();
        let (err, orth) = check_thin(&f.q_thin().to_vec().unwrap(), &f.r().to_vec().unwrap(), &a, m, n);
        assert!(err < 3e-5 && orth < 3e-5 * (m as f64 / 100.0).sqrt().max(1.0), "{m}x{n}: ||QR - A|| / ||A|| = {err}, ||QtQ - I|| = {orth}");
        let tau = f.tau.to_vec().unwrap();
        assert!(tau.iter().all(|t| (0.0..=2.0 + 1e-5).contains(t)), "{m}x{n}: tau out of range");
        let b = values(2 * m, 5, -1.0, 1.0);
        let x = widen(&f.lstsq(&mat(&ctx, &b, 2, m, 1)).to_vec().unwrap());
        let ad = widen(&a);
        let r: Vec<f64> = mul(&ad, &x, m, n, 2).iter().zip(&widen(&b)).map(|(p, q)| p - q).collect();
        let atr = mul(&transpose(&ad, m, n), &r, n, m, 2);
        assert!(fro(&atr) < 5e-5 * fro(&ad) * fro(&r), "{m}x{n}: Aᵀ r = {}", fro(&atr));
    }
}
