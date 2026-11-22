//! `cholesky`, `cholesky_upper`, `solve_spd`, `cholesky_solve` and `solve_triangular` against `f64`
//! host references: sizes around the kernel and recursion thresholds, batches, ill-conditioned
//! matrices, matrices that are not positive definite, every triangle / transpose / unit variant,
//! vector right-hand sides.

use crate::common::values;
use crate::dense_host::*;
use luma::{Array, Context, Triangle};

const SIZES: [usize; 14] = [1, 2, 3, 5, 31, 32, 33, 64, 65, 127, 128, 129, 200, 333];



fn mat(ctx: &Context, data: &[f32], cols: usize, rows: usize, batch: usize) -> Array<f32> {
    Array::from_slice_on(ctx, data, [cols, rows, batch]).unwrap()
}

/// Relative error `||L Lᵀ - A||_F / ||A||_F` of a lower factor.
fn factor_error(l: &[f32], a: &[f32], n: usize) -> f64 {
    let l = widen(l);
    let a = widen(a);
    // The strict upper triangle must be exactly zero.
    for i in 0..n {
        for j in i + 1..n {
            assert_eq!(l[i * n + j], 0.0, "upper triangle ({i}, {j})");
        }
    }
    let llt = mul(&l, &transpose(&l, n, n), n, n, n);
    diff(&llt, &a) / fro(&a)
}

#[test]
fn cholesky_factors_and_matches_the_host() {
    let ctx = Context::new().unwrap();
    for &n in &SIZES {
        let a = spd(n, 100 + n as u64, 1.0);
        let l = mat(&ctx, &a, n, n, 1).cholesky().to_vec().unwrap();
        let err = factor_error(&l, &a, n);
        assert!(err < 2e-6 * (n as f64).sqrt().max(1.0), "n = {n}: ||LLt - A|| / ||A|| = {err}");
        let want = chol(&widen(&a), n).unwrap();
        let got = widen(&l);
        assert!(diff(&got, &want) / fro(&want) < 1e-5, "n = {n}: differs from the f64 factor by {}", diff(&got, &want) / fro(&want));
    }
}

#[test]
fn cholesky_batches() {
    let ctx = Context::new().unwrap();
    for &(n, batch) in &[(1usize, 5usize), (4, 7), (16, 100), (33, 9), (64, 4), (128, 3), (150, 3)] {
        let mut data = Vec::new();
        for b in 0..batch {
            data.extend(spd(n, 7 + (n * 13 + b) as u64, 0.5));
        }
        let l = mat(&ctx, &data, n, n, batch).cholesky();
        assert_eq!(l.shape().dims(), [n, n, batch, 1]);
        let l = l.to_vec().unwrap();
        for b in 0..batch {
            let err = factor_error(&l[b * n * n..(b + 1) * n * n], &data[b * n * n..(b + 1) * n * n], n);
            assert!(err < 3e-6, "n = {n}, matrix {b}: {err}");
        }
    }
}

#[test]
fn a_big_batch_folds_over_the_grid() {
    let ctx = Context::new().unwrap();
    let (n, batch) = (3usize, 70_000usize);
    let mut data = Vec::new();
    for b in 0..batch {
        let d = 1.0 + (b % 7) as f32;
        // [[d, 0, 0], [0, d, 0], [0, 0, d]] plus a coupling.
        data.extend([d + 1.0, 0.5, 0.0, 0.5, d + 1.0, 0.5, 0.0, 0.5, d + 1.0]);
    }
    let l = mat(&ctx, &data, n, n, batch).cholesky().to_vec().unwrap();
    for b in (0..batch).step_by(997).chain([batch - 1]) {
        let err = factor_error(&l[b * 9..(b + 1) * 9], &data[b * 9..(b + 1) * 9], n);
        assert!(err < 3e-6, "matrix {b}: {err}");
    }
}

#[test]
fn cholesky_upper_is_the_transpose() {
    let ctx = Context::new().unwrap();
    for &n in &[3usize, 40, 150] {
        let a = spd(n, 5, 1.0);
        let u = mat(&ctx, &a, n, n, 1).cholesky_upper().to_vec().unwrap();
        let l = widen(&u);
        let utu = mul(&transpose(&l, n, n), &l, n, n, n);
        assert!(diff(&utu, &widen(&a)) / fro(&widen(&a)) < 3e-6);
        for i in 0..n {
            for j in 0..i {
                assert_eq!(u[i * n + j], 0.0);
            }
        }
    }
}

#[test]
fn only_the_lower_triangle_is_read() {
    let ctx = Context::new().unwrap();
    for &n in &[10usize, 150] {
        let a = spd(n, 9, 1.0);
        let mut garbage = a.clone();
        for i in 0..n {
            for j in i + 1..n {
                garbage[i * n + j] = 1e6;
            }
        }
        let want = mat(&ctx, &a, n, n, 1).cholesky().to_vec().unwrap();
        let got = mat(&ctx, &garbage, n, n, 1).cholesky().to_vec().unwrap();
        assert_eq!(got, want, "n = {n}");
    }
}

#[test]
fn ill_conditioned_matrices_stay_backward_stable() {
    let ctx = Context::new().unwrap();
    for &(n, rho) in &[(20usize, 0.9f64), (100, 0.99), (200, 0.995)] {
        let a = kms(n, rho);
        let l = mat(&ctx, &a, n, n, 1).cholesky().to_vec().unwrap();
        let err = factor_error(&l, &a, n);
        assert!(err < 5e-6, "n = {n}, rho = {rho}: {err}");
        // Solve with a known solution; the residual is what is guaranteed, not the forward error.
        let b = values(n * 2, 3, -1.0, 1.0);
        let x = mat(&ctx, &a, n, n, 1).solve_spd(&mat(&ctx, &b, 2, n, 1)).to_vec().unwrap();
        let r = residual(&widen(&a), &widen(&x), &widen(&b), n, 2);
        assert!(r < 1e-5, "n = {n}, rho = {rho}: residual {r}");
    }
}

#[test]
fn a_matrix_that_is_not_positive_definite_gives_nan() {
    let ctx = Context::new().unwrap();
    for &n in &[2usize, 40, 200] {
        let mut a = spd(n, 11, 1.0);
        // A negative pivot half way.
        a[(n / 2) * n + n / 2] = -5.0;
        let l = mat(&ctx, &a, n, n, 1).cholesky().to_vec().unwrap();
        assert!((0..n).any(|i| l[i * n + i].is_nan()), "n = {n}: no NaN on the diagonal");
        let x = mat(&ctx, &a, n, n, 1).solve_spd(&mat(&ctx, &values(n, 1, 0.0, 1.0), 1, n, 1)).to_vec().unwrap();
        assert!(x.iter().any(|v| v.is_nan()), "n = {n}: solve gave no NaN");
    }
    // A zero matrix and a NaN entry.
    let l = mat(&ctx, &[0.0f32; 4], 2, 2, 1).cholesky().to_vec().unwrap();
    assert!(l[0].is_nan());
    let l = mat(&ctx, &[f32::NAN, 0.0, 0.0, 1.0], 2, 2, 1).cholesky().to_vec().unwrap();
    assert!(l[0].is_nan());
    // One bad matrix in a batch leaves the others alone.
    let good = spd(8, 1, 1.0);
    let mut data = good.clone();
    data.extend(vec![-1.0f32; 64]);
    data.extend(good.clone());
    let l = mat(&ctx, &data, 8, 8, 3).cholesky().to_vec().unwrap();
    assert!(l[64..128].iter().any(|v| v.is_nan()));
    assert!(factor_error(&l[..64], &good, 8) < 3e-6 && factor_error(&l[128..], &good, 8) < 3e-6);
}

#[test]
fn solve_spd_residuals() {
    let ctx = Context::new().unwrap();
    for &n in &SIZES {
        for &k in &[1usize, 5, 70] {
            let a = spd(n, 200 + n as u64, 1.0);
            let b = values(n * k, 300 + k as u64, -1.0, 1.0);
            let x = mat(&ctx, &a, n, n, 1).solve_spd(&mat(&ctx, &b, k, n, 1));
            assert_eq!(x.shape().dims(), [k, n, 1, 1]);
            let x = x.to_vec().unwrap();
            let r = residual(&widen(&a), &widen(&x), &widen(&b), n, k);
            assert!(r < 5e-6, "n = {n}, k = {k}: residual {r}");
            let want = solve(&widen(&a), &widen(&b), n, k);
            assert!(diff(&widen(&x), &want) / fro(&want) < 1e-3, "n = {n}, k = {k}: forward error {}", diff(&widen(&x), &want) / fro(&want));
        }
    }
}

#[test]
fn solve_spd_vector_and_batch() {
    let ctx = Context::new().unwrap();
    let n = 70;
    let a = spd(n, 1, 1.0);
    let b = values(n, 2, -1.0, 1.0);
    let x = mat(&ctx, &a, n, n, 1).solve_spd(&Array::from_slice_on(&ctx, &b, n).unwrap());
    assert_eq!(x.shape().dims(), [n, 1, 1, 1]);
    let x = x.to_vec().unwrap();
    assert!(residual(&widen(&a), &widen(&x), &widen(&b), n, 1) < 5e-6);

    for &(n, batch) in &[(6usize, 20usize), (90, 3)] {
        let mut data = Vec::new();
        let mut rhs = Vec::new();
        for i in 0..batch {
            data.extend(spd(n, 40 + i as u64, 1.0));
            rhs.extend(values(n * 3, 50 + i as u64, -1.0, 1.0));
        }
        let x = mat(&ctx, &data, n, n, batch).solve_spd(&mat(&ctx, &rhs, 3, n, batch)).to_vec().unwrap();
        for i in 0..batch {
            let r = residual(&widen(&data[i * n * n..(i + 1) * n * n]), &widen(&x[i * n * 3..(i + 1) * n * 3]), &widen(&rhs[i * n * 3..(i + 1) * n * 3]), n, 3);
            assert!(r < 5e-6, "n = {n}, matrix {i}: {r}");
        }
    }
}

#[test]
fn cholesky_solve_uses_the_factor() {
    let ctx = Context::new().unwrap();
    let n = 150;
    let a = spd(n, 3, 1.0);
    let b = values(n * 4, 4, -1.0, 1.0);
    let l = mat(&ctx, &a, n, n, 1).cholesky();
    let x = l.cholesky_solve(&mat(&ctx, &b, 4, n, 1)).to_vec().unwrap();
    assert!(residual(&widen(&a), &widen(&x), &widen(&b), n, 4) < 5e-6);
}

/// A triangular matrix with a clearly nonzero diagonal and the other triangle filled with `junk`.
fn triangular(n: usize, seed: u64, tri: Triangle, junk: f32) -> (Vec<f32>, Vec<f64>) {
    let r = values(n * n, seed, -1.0, 1.0);
    let mut stored = vec![0.0f32; n * n];
    let mut exact = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            let inside = match tri {
                Triangle::Lower => j <= i,
                Triangle::Upper => j >= i,
            };
            let v = if i == j { 1.5 + r[i * n + j].abs() } else { r[i * n + j] / (n as f32).sqrt() };
            stored[i * n + j] = if inside { v } else { junk };
            exact[i * n + j] = if inside { f64::from(v) } else { 0.0 };
        }
    }
    (stored, exact)
}

#[test]
fn solve_triangular_every_variant() {
    let ctx = Context::new().unwrap();
    for &n in &[1usize, 2, 17, 64, 65, 100, 130, 257] {
        for &k in &[1usize, 3, 66] {
            for tri in [Triangle::Lower, Triangle::Upper] {
                for transpose_a in [false, true] {
                    for unit in [false, true] {
                        let (stored, mut exact) = triangular(n, (n * 7 + k) as u64, tri, 1e3);
                        if unit {
                            for i in 0..n {
                                exact[i * n + i] = 1.0;
                            }
                        }
                        let b = values(n * k, 77 + n as u64, -1.0, 1.0);
                        let x = mat(&ctx, &stored, n, n, 1).solve_triangular(&mat(&ctx, &b, k, n, 1), tri, transpose_a, unit);
                        assert_eq!(x.shape().dims(), [k, n, 1, 1]);
                        let x = widen(&x.to_vec().unwrap());
                        let op = if transpose_a { transpose(&exact, n, n) } else { exact.clone() };
                        let r = residual(&op, &x, &widen(&b), n, k);
                        assert!(r < 2e-6, "n = {n}, k = {k}, {tri:?}, transpose {transpose_a}, unit {unit}: residual {r}");
                    }
                }
            }
        }
    }
}

/// A few right-hand sides of a triangle of many blocks of rows go through the one-launch solve of CUDA devices (columns in
/// groups of 8, rows in blocks of 64, the last one partial): every variant, with the junk half and without the diagonal.
#[test]
fn solve_triangular_few_columns_of_many_blocks() {
    let ctx = Context::new().unwrap();
    for &n in &[65usize, 130, 257, 600] {
        for &k in &[2usize, 8, 9, 16, 32, 33] {
            for (tri, transpose_a, unit) in [(Triangle::Lower, false, false), (Triangle::Upper, true, true), (Triangle::Upper, false, false), (Triangle::Lower, true, false)] {
                let (stored, mut exact) = triangular(n, (n * 11 + k) as u64, tri, 1e3);
                if unit {
                    for i in 0..n {
                        exact[i * n + i] = 1.0;
                    }
                }
                let b = values(n * k, 99 + n as u64, -1.0, 1.0);
                let x = mat(&ctx, &stored, n, n, 1).solve_triangular(&mat(&ctx, &b, k, n, 1), tri, transpose_a, unit);
                let x = widen(&x.to_vec().unwrap());
                let op = if transpose_a { transpose(&exact, n, n) } else { exact.clone() };
                let r = residual(&op, &x, &widen(&b), n, k);
                assert!(r < 2e-6, "n = {n}, k = {k}, {tri:?}, transpose {transpose_a}, unit {unit}: residual {r}");
            }
        }
    }
}

/// `solve_spd` of many blocks of rows with a few right-hand sides (the diagonal blocks taken as inverses on CUDA devices).
#[test]
fn solve_spd_few_columns_of_many_blocks() {
    let ctx = Context::new().unwrap();
    for &n in &[130usize, 257, 600] {
        for &k in &[1usize, 8, 32] {
            let a = spd(n, 700 + n as u64, 1.0);
            let b = values(n * k, 800 + k as u64, -1.0, 1.0);
            let x = mat(&ctx, &a, n, n, 1).solve_spd(&mat(&ctx, &b, k, n, 1)).to_vec().unwrap();
            let r = residual(&widen(&a), &widen(&x), &widen(&b), n, k);
            assert!(r < 5e-6, "n = {n}, k = {k}: residual {r}");
        }
    }
}

#[test]
fn solve_triangular_batches_and_vectors() {
    let ctx = Context::new().unwrap();
    let (n, batch) = (12usize, 9usize);
    let mut data = Vec::new();
    let mut exact = Vec::new();
    let mut rhs = Vec::new();
    for i in 0..batch {
        let (s, e) = triangular(n, i as u64, Triangle::Upper, f32::NAN);
        data.extend(s);
        exact.push(e);
        rhs.extend(values(n * 2, 9 + i as u64, -1.0, 1.0));
    }
    let x = mat(&ctx, &data, n, n, batch).solve_triangular(&mat(&ctx, &rhs, 2, n, batch), Triangle::Upper, true, false).to_vec().unwrap();
    for i in 0..batch {
        let op = transpose(&exact[i], n, n);
        let r = residual(&op, &widen(&x[i * n * 2..(i + 1) * n * 2]), &widen(&rhs[i * n * 2..(i + 1) * n * 2]), n, 2);
        assert!(r < 2e-6, "matrix {i}: {r}");
    }
    // A vector right-hand side keeps its shape.
    let n = 90;
    let (s, e) = triangular(n, 3, Triangle::Lower, 0.0);
    let b = values(n, 4, -1.0, 1.0);
    let x = mat(&ctx, &s, n, n, 1).solve_triangular(&Array::from_slice_on(&ctx, &b, n).unwrap(), Triangle::Lower, false, false);
    assert_eq!(x.shape().dims(), [n, 1, 1, 1]);
    assert!(residual(&e, &widen(&x.to_vec().unwrap()), &widen(&b), n, 1) < 2e-6);
}

#[test]
fn shape_errors_poison() {
    let ctx = Context::new().unwrap();
    let rect = Array::<f32>::zeros_on(&ctx, [3, 4]).unwrap();
    assert!(rect.cholesky().to_vec().is_err());
    let sq = mat(&ctx, &spd(4, 1, 1.0), 4, 4, 1);
    assert!(sq.solve_spd(&Array::<f32>::zeros_on(&ctx, [2, 5]).unwrap()).to_vec().is_err());
    assert!(sq.solve_triangular(&Array::<f32>::zeros_on(&ctx, [2, 3]).unwrap(), Triangle::Lower, false, false).to_vec().is_err());
    let four_d = Array::<f32>::zeros_on(&ctx, [2, 2, 2, 2]).unwrap();
    assert!(four_d.cholesky().to_vec().is_err());
}
