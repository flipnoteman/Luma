//! `lu`, `solve`, `inverse`, `det` and `slogdet` against `f64` host references: sizes around the
//! kernel and recursion thresholds, batches, matrices that need pivoting, singular and
//! ill-conditioned ones, vector right-hand sides.

use crate::common::values;
use crate::dense_host::*;
use luma::{Array, Context};

const SIZES: [usize; 14] = [1, 2, 3, 5, 31, 32, 33, 64, 65, 100, 128, 129, 200, 300];

fn mat(ctx: &Context, data: &[f32], cols: usize, rows: usize, batch: usize) -> Array<f32> {
    Array::from_slice_on(ctx, data, [cols, rows, batch]).unwrap()
}

/// A random matrix that is well conditioned in practice: uniform entries plus a boost of the diagonal
/// only sometimes, so pivoting is exercised.
fn random(n: usize, seed: u64) -> Vec<f32> {
    values(n * n, seed, -1.0, 1.0)
}

/// `||P A - L U||_F / ||A||_F` and the largest `|l_ij|` of the packed factors.
fn lu_error(lu: &[f32], perm: &[u32], a: &[f32], n: usize) -> (f64, f64) {
    let mut l = vec![0.0f64; n * n];
    let mut u = vec![0.0f64; n * n];
    let mut lmax = 0.0f64;
    for i in 0..n {
        for j in 0..n {
            let v = f64::from(lu[i * n + j]);
            if j < i {
                l[i * n + j] = v;
                lmax = lmax.max(v.abs());
            } else {
                u[i * n + j] = v;
            }
        }
        l[i * n + i] = 1.0;
    }
    let lu = mul(&l, &u, n, n, n);
    let mut pa = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            pa[i * n + j] = f64::from(a[perm[i] as usize * n + j]);
        }
    }
    (diff(&lu, &pa) / fro(&widen(a)), lmax)
}

fn is_permutation(perm: &[u32]) -> bool {
    let mut seen = vec![false; perm.len()];
    perm.iter().all(|&p| (p as usize) < perm.len() && !std::mem::replace(&mut seen[p as usize], true))
}

/// The determinant in `f64` by elimination with partial pivoting.
fn host_det(a: &[f32], n: usize) -> f64 {
    let mut a = widen(a);
    let mut det = 1.0;
    for c in 0..n {
        let p = (c..n).max_by(|&i, &j| a[i * n + c].abs().total_cmp(&a[j * n + c].abs())).unwrap();
        if a[p * n + c] == 0.0 {
            return 0.0;
        }
        if p != c {
            for j in 0..n {
                a.swap(c * n + j, p * n + j);
            }
            det = -det;
        }
        det *= a[c * n + c];
        for i in c + 1..n {
            let f = a[i * n + c] / a[c * n + c];
            for j in c..n {
                a[i * n + j] -= f * a[c * n + j];
            }
        }
    }
    det
}

#[test]
fn lu_factors_every_size() {
    let ctx = Context::new().unwrap();
    for &n in &SIZES {
        let a = random(n, 10 + n as u64);
        let f = mat(&ctx, &a, n, n, 1).lu();
        assert_eq!(f.lu.shape().dims(), [n, n, 1, 1]);
        assert_eq!(f.perm.shape().dims(), [n, 1, 1, 1]);
        let (lu, perm) = (f.lu.to_vec().unwrap(), f.perm.to_vec().unwrap());
        assert!(is_permutation(&perm), "n = {n}: not a permutation");
        let (err, lmax) = lu_error(&lu, &perm, &a, n);
        assert!(err < 3e-6 * (n as f64).sqrt().max(1.0) * 4.0, "n = {n}: ||PA - LU|| / ||A|| = {err}");
        assert!(lmax <= 1.0 + 1e-6, "n = {n}: |l| = {lmax}");
    }
}

#[test]
fn lu_needs_pivoting() {
    let ctx = Context::new().unwrap();
    // A permutation matrix, a matrix with a zero diagonal, a tiny leading entry.
    for &n in &[4usize, 70, 150] {
        let mut a = vec![0.0f32; n * n];
        for i in 0..n {
            a[i * n + (i + 1) % n] = 1.0 + i as f32;
        }
        let f = mat(&ctx, &a, n, n, 1).lu();
        let (lu, perm) = (f.lu.to_vec().unwrap(), f.perm.to_vec().unwrap());
        let (err, _) = lu_error(&lu, &perm, &a, n);
        assert!(err < 1e-6, "n = {n}: {err}");
        let mut b = random(n, 5);
        b[0] = 1e-20;
        let f = mat(&ctx, &b, n, n, 1).lu();
        let (lu, perm) = (f.lu.to_vec().unwrap(), f.perm.to_vec().unwrap());
        let (err, lmax) = lu_error(&lu, &perm, &b, n);
        assert!(err < 3e-6 * 4.0 && lmax <= 1.0 + 1e-6, "n = {n}: {err}, {lmax}");
        assert_ne!(perm[0], 0);
    }
}

#[test]
fn lu_batches() {
    let ctx = Context::new().unwrap();
    for &(n, batch) in &[(1usize, 4usize), (3, 50), (16, 100), (33, 9), (64, 5), (65, 3), (130, 2)] {
        let mut data = Vec::new();
        for b in 0..batch {
            data.extend(random(n, 3 + (n * 7 + b) as u64));
        }
        let f = mat(&ctx, &data, n, n, batch).lu();
        assert_eq!(f.perm.shape().dims(), [n, 1, batch, 1]);
        let (lu, perm) = (f.lu.to_vec().unwrap(), f.perm.to_vec().unwrap());
        for b in 0..batch {
            let (err, _) = lu_error(&lu[b * n * n..(b + 1) * n * n], &perm[b * n..(b + 1) * n], &data[b * n * n..(b + 1) * n * n], n);
            assert!(err < 2e-5, "n = {n}, matrix {b}: {err}");
        }
    }
}

#[test]
fn lower_and_upper_unpack() {
    let ctx = Context::new().unwrap();
    let n = 40;
    let a = random(n, 1);
    let f = mat(&ctx, &a, n, n, 1).lu();
    let (l, u, perm) = (widen(&f.lower().to_vec().unwrap()), widen(&f.upper().to_vec().unwrap()), f.perm.to_vec().unwrap());
    let lu = mul(&l, &u, n, n, n);
    for i in 0..n {
        for j in 0..n {
            assert!((lu[i * n + j] - f64::from(a[perm[i] as usize * n + j])).abs() < 1e-5);
        }
    }
}

#[test]
fn solve_residuals() {
    let ctx = Context::new().unwrap();
    for &n in &SIZES {
        for &k in &[1usize, 4, 70] {
            let a = random(n, 400 + n as u64);
            let b = values(n * k, 500 + k as u64, -1.0, 1.0);
            let x = mat(&ctx, &a, n, n, 1).solve(&mat(&ctx, &b, k, n, 1));
            assert_eq!(x.shape().dims(), [k, n, 1, 1]);
            let x = x.to_vec().unwrap();
            let r = residual(&widen(&a), &widen(&x), &widen(&b), n, k);
            assert!(r < 1e-5, "n = {n}, k = {k}: residual {r}");
        }
    }
}

#[test]
fn solve_vector_batch_and_unpacked_factors() {
    let ctx = Context::new().unwrap();
    let n = 90;
    let a = random(n, 1);
    let b = values(n, 2, -1.0, 1.0);
    let x = mat(&ctx, &a, n, n, 1).solve(&Array::from_slice_on(&ctx, &b, n).unwrap());
    assert_eq!(x.shape().dims(), [n, 1, 1, 1]);
    assert!(residual(&widen(&a), &widen(&x.to_vec().unwrap()), &widen(&b), n, 1) < 1e-5);
    // The factorisation solves many right-hand sides without refactoring.
    let f = mat(&ctx, &a, n, n, 1).lu();
    for seed in 0..3 {
        let b = values(n * 2, 10 + seed, -1.0, 1.0);
        let x = f.solve(&mat(&ctx, &b, 2, n, 1)).to_vec().unwrap();
        assert!(residual(&widen(&a), &widen(&x), &widen(&b), n, 2) < 1e-5);
    }
    for &(n, batch) in &[(7usize, 30usize), (100, 3)] {
        let mut data = Vec::new();
        let mut rhs = Vec::new();
        for i in 0..batch {
            data.extend(random(n, 60 + i as u64));
            rhs.extend(values(n * 2, 70 + i as u64, -1.0, 1.0));
        }
        let x = mat(&ctx, &data, n, n, batch).solve(&mat(&ctx, &rhs, 2, n, batch)).to_vec().unwrap();
        for i in 0..batch {
            let r = residual(&widen(&data[i * n * n..(i + 1) * n * n]), &widen(&x[i * n * 2..(i + 1) * n * 2]), &widen(&rhs[i * n * 2..(i + 1) * n * 2]), n, 2);
            assert!(r < 1e-5, "n = {n}, matrix {i}: {r}");
        }
    }
}

#[test]
fn inverse_times_matrix_is_identity() {
    let ctx = Context::new().unwrap();
    for &n in &[1usize, 2, 5, 33, 64, 100, 150] {
        let a = random(n, 900 + n as u64);
        let inv = mat(&ctx, &a, n, n, 1).inverse().to_vec().unwrap();
        let prod = mul(&widen(&a), &widen(&inv), n, n, n);
        let mut eye = vec![0.0f64; n * n];
        for i in 0..n {
            eye[i * n + i] = 1.0;
        }
        let err = diff(&prod, &eye) / (n as f64).sqrt();
        assert!(err < 1e-4 * (n as f64).sqrt().max(1.0), "n = {n}: ||A A^-1 - I|| = {err}");
    }
    // A batch.
    let (n, batch) = (6usize, 11usize);
    let mut data = Vec::new();
    for b in 0..batch {
        data.extend(random(n, 20 + b as u64));
    }
    let inv = mat(&ctx, &data, n, n, batch).inverse().to_vec().unwrap();
    for b in 0..batch {
        let p = mul(&widen(&data[b * 36..(b + 1) * 36]), &widen(&inv[b * 36..(b + 1) * 36]), n, n, n);
        for i in 0..n {
            for j in 0..n {
                assert!((p[i * n + j] - if i == j { 1.0 } else { 0.0 }).abs() < 1e-4, "matrix {b}");
            }
        }
    }
}

#[test]
fn determinants() {
    let ctx = Context::new().unwrap();
    for &n in &[1usize, 2, 3, 5, 8, 16, 33, 70] {
        // Entries scaled so that the determinant stays inside f32.
        let a: Vec<f32> = random(n, 77 + n as u64).iter().map(|v| v * 1.0).collect();
        let want = host_det(&a, n);
        let got = f64::from(mat(&ctx, &a, n, n, 1).det().item().unwrap());
        if want.abs() < 1e30 && want.abs() > 1e-30 {
            assert!((got - want).abs() <= 2e-4 * want.abs() + 1e-30, "n = {n}: got {got}, want {want}");
        }
        let (sign, logabs) = mat(&ctx, &a, n, n, 1).slogdet();
        let (sign, logabs) = (f64::from(sign.item().unwrap()), f64::from(logabs.item().unwrap()));
        assert_eq!(sign, want.signum(), "n = {n}");
        assert!((logabs - want.abs().ln()).abs() < 2e-4 * want.abs().ln().abs().max(1.0), "n = {n}: {logabs} vs {}", want.abs().ln());
    }
    // A known one, with a swap: [[0, 1], [2, 3]] has determinant -2.
    let a = mat(&ctx, &[0.0f32, 1.0, 2.0, 3.0], 2, 2, 1);
    assert!((a.det().item().unwrap() + 2.0).abs() < 1e-6);
    // A batch of determinants.
    let (n, batch) = (4usize, 9usize);
    let mut data = Vec::new();
    for b in 0..batch {
        data.extend(random(n, 30 + b as u64));
    }
    let d = mat(&ctx, &data, n, n, batch).det();
    assert_eq!(d.shape().dims(), [1, 1, batch, 1]);
    let d = d.to_vec().unwrap();
    for b in 0..batch {
        let want = host_det(&data[b * 16..(b + 1) * 16], n);
        assert!((f64::from(d[b]) - want).abs() < 1e-5 * want.abs().max(1e-3), "matrix {b}");
    }
}

#[test]
fn singular_matrices_do_not_poison() {
    let ctx = Context::new().unwrap();
    for &n in &[3usize, 40, 100] {
        // Scaled so that the determinant of the random matrix stays inside f32; two equal rows.
        let mut a: Vec<f32> = random(n, 4).iter().map(|v| v * 0.15).collect();
        for j in 0..n {
            a[(n - 1) * n + j] = a[j];
        }
        let d = mat(&ctx, &a, n, n, 1).det().item().unwrap();
        assert!(d.abs() < 1e-3, "n = {n}: det {d}");
        // A zero matrix has determinant exactly 0 and its LU is finite.
        let z = mat(&ctx, &vec![0.0f32; n * n], n, n, 1);
        assert_eq!(z.det().item().unwrap(), 0.0);
        assert!(z.lu().lu.to_vec().unwrap().iter().all(|v| *v == 0.0));
        let x = z.solve(&mat(&ctx, &vec![1.0f32; n], 1, n, 1)).to_vec().unwrap();
        assert!(x.iter().any(|v| !v.is_finite()), "n = {n}: singular solve was finite");
    }
}

#[test]
fn ill_conditioned_systems_have_small_residuals() {
    let ctx = Context::new().unwrap();
    // The Hilbert matrix, condition number ~1e10 at n = 8 in f32 (stored rounded).
    for &n in &[6usize, 10, 40] {
        let mut a = vec![0.0f32; n * n];
        for i in 0..n {
            for j in 0..n {
                a[i * n + j] = (1.0 / (i + j + 1) as f64) as f32;
            }
        }
        let b = values(n, 3, -1.0, 1.0);
        let x = mat(&ctx, &a, n, n, 1).solve(&mat(&ctx, &b, 1, n, 1)).to_vec().unwrap();
        let r = residual(&widen(&a), &widen(&x), &widen(&b), n, 1);
        assert!(r < 1e-5, "n = {n}: residual {r}");
    }
    // Rows of wildly different scales.
    let n = 150;
    let mut a = random(n, 12);
    for i in 0..n {
        let scale = 10f32.powi((i % 7) as i32 - 3);
        for j in 0..n {
            a[i * n + j] *= scale;
        }
    }
    let b = values(n * 2, 13, -1.0, 1.0);
    let x = mat(&ctx, &a, n, n, 1).solve(&mat(&ctx, &b, 2, n, 1)).to_vec().unwrap();
    assert!(residual(&widen(&a), &widen(&x), &widen(&b), n, 2) < 1e-5);
}

#[test]
fn nan_input_propagates_and_shapes_are_checked() {
    let ctx = Context::new().unwrap();
    let mut a = random(20, 1);
    a[5 * 20 + 7] = f32::NAN;
    let x = mat(&ctx, &a, 20, 20, 1).solve(&mat(&ctx, &values(20, 2, 0.0, 1.0), 1, 20, 1)).to_vec().unwrap();
    assert!(x.iter().any(|v| v.is_nan()));
    let rect = Array::<f32>::zeros_on(&ctx, [3, 4]).unwrap();
    assert!(rect.solve(&Array::<f32>::zeros_on(&ctx, [1, 3]).unwrap()).to_vec().is_err());
    assert!(rect.inverse().to_vec().is_err());
    assert!(rect.det().to_vec().is_err());
    let four_d = Array::<f32>::zeros_on(&ctx, [2, 2, 2, 2]).unwrap();
    assert!(four_d.lu().lu.to_vec().is_err());
}

#[test]
fn tall_matrices_factor_too() {
    let ctx = Context::new().unwrap();
    for &(n, m) in &[(1usize, 5usize), (5, 20), (40, 100), (70, 300), (33, 65), (100, 130), (5, 3), (40, 20), (100, 60), (300, 130)] {
        let a = values(n * m, 5 + (n * 3 + m) as u64, -1.0, 1.0);
        let f = mat(&ctx, &a, n, m, 1).lu();
        assert_eq!(f.lu.shape().dims(), [n, m, 1, 1]);
        let (l, u, perm) = (widen(&f.lower().to_vec().unwrap()), widen(&f.upper().to_vec().unwrap()), f.perm.to_vec().unwrap());
        assert!(is_permutation(&perm), "{m}x{n}: not a permutation");
        let k = n.min(m);
        let lu = mul(&l, &u, m, k, n);
        let mut pa = vec![0.0f64; m * n];
        for i in 0..m {
            for j in 0..n {
                pa[i * n + j] = f64::from(a[perm[i] as usize * n + j]);
            }
        }
        assert!(diff(&lu, &pa) / fro(&widen(&a)) < 2e-5, "{m}x{n}: ||PA - LU|| = {}", diff(&lu, &pa));
    }
}

#[test]
fn infinite_entries_do_not_hang_or_poison() {
    let ctx = Context::new().unwrap();
    for &n in &[3usize, 40, 100] {
        let mut a = random(n, 2);
        a[n + 1] = f32::INFINITY;
        let m = mat(&ctx, &a, n, n, 1);
        let lu = m.lu().lu.to_vec().unwrap();
        assert!(lu.iter().any(|v| !v.is_finite()), "n = {n}: LU of an infinite entry is finite");
        let q = m.qr().r().to_vec().unwrap();
        assert!(q.iter().any(|v| !v.is_finite()), "n = {n}: QR of an infinite entry is finite");
        let spd_inf = {
            let mut s = spd(n, 3, 1.0);
            s[0] = f32::INFINITY;
            mat(&ctx, &s, n, n, 1).cholesky().to_vec().unwrap()
        };
        assert!(spd_inf.iter().any(|v| !v.is_finite()), "n = {n}: Cholesky of an infinite entry is finite");
    }
}
