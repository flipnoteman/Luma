//! Dense linear algebra: Cholesky, triangular and SPD solves (single big matrices and batches of
//! small ones), verified by backward error against an `f64` host matrix-vector product with a
//! random vector (`O(n^2)`, so that n = 4096 verifies in milliseconds).

use super::{host, pick, upload};

fn widen(a: &[f32]) -> Vec<f64> {
    a.iter().map(|&x| f64::from(x)).collect()
}
use crate::harness::*;

/// A symmetric, strictly diagonally dominant (so positive definite) `n x n` matrix.
fn dd_spd(n: usize, seed: u64) -> Vec<f32> {
    let r = values(n * n, seed, -0.5, 0.5);
    let mut a = vec![0.0f32; n * n];
    for i in 0..n {
        for j in 0..=i {
            a[i * n + j] = r[i * n + j];
            a[j * n + i] = r[i * n + j];
        }
        a[i * n + i] = n as f32 * 0.5 + 1.0 + r[i * n + i].abs();
    }
    a
}

/// `M z` for the `n x n` row-major matrix `get(i, j)`.
fn matvec(get: &dyn Fn(usize, usize) -> f64, n: usize, z: &[f64]) -> Vec<f64> {
    (0..n).map(|i| (0..n).map(|j| get(i, j) * z[j]).sum()).collect()
}

fn rel_diff(a: &[f64], b: &[f64]) -> f64 {
    let d: f64 = a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f64>().sqrt();
    d / b.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-300)
}

/// `|| L Lᵀ z - A z || / || A z ||` for row-major `n x n` matrices.
fn chol_error(l: &[f32], a: &[f32], n: usize) -> f64 {
    let z: Vec<f64> = values(n, 5, -1.0, 1.0).iter().map(|&x| f64::from(x)).collect();
    let tri = |i: usize, j: usize| if j > i { 0.0 } else { f64::from(l[i * n + j]) };
    let lt_z = matvec(&|i, j| tri(j, i), n, &z);
    rel_diff(&matvec(&tri, n, &lt_z), &matvec(&|i, j| f64::from(a[i * n + j]), n, &z))
}

/// `I + 0.17 R / sqrt(n)`: a general matrix with spectrum in a disk of radius 0.1 around 1.
fn gen_mat(n: usize, seed: u64) -> Vec<f32> {
    let mut a = values(n * n, seed, -1.0, 1.0);
    let s = 0.1732 / (n as f32).sqrt();
    for v in &mut a {
        *v *= s;
    }
    for i in 0..n {
        a[i * n + i] += 1.0;
    }
    a
}

/// `|M x - b| / (|M|_F |x| + |b|)` for the `n x n` row-major `m` and one vector.
fn solve_error(m: &[f32], n: usize, x: &[f32], b: &[f32]) -> f64 {
    let xd: Vec<f64> = x.iter().map(|&v| f64::from(v)).collect();
    let mx = matvec(&|i, j| f64::from(m[i * n + j]), n, &xd);
    let bd: Vec<f64> = b.iter().map(|&v| f64::from(v)).collect();
    let norm = m.iter().map(|&v| f64::from(v) * f64::from(v)).sum::<f64>().sqrt();
    let d: f64 = mx.iter().zip(&bd).map(|(p, q)| (p - q) * (p - q)).sum::<f64>().sqrt();
    d / (norm * xd.iter().map(|v| v * v).sum::<f64>().sqrt() + bd.iter().map(|v| v * v).sum::<f64>().sqrt())
}

/// LU, general solve, inverse, determinant, QR and least squares.
fn general(h: &mut Harness) {
    let sizes: &[usize] = if h.ctx.is_cpu() { pick(h, &[128, 256], &[256, 512]) } else { pick(h, &[256, 1024], &[256, 1024, 4096]) };
    for &n in sizes {
        let a = gen_mat(n, 120 + n as u64);
        let al = upload(h, &a, [n, n]);
        let b = values(n, 121, -1.0, 1.0);
        let bl = upload(h, &b, [1, n]);
        let x = host(&al.solve(&bl));
        let r = solve_error(&a, n, &x, &b);
        assert!(r < 1e-5, "solve {n}: backward error {r}");
        let nf = n as f64;
        h.case("dense/lu", &format!("{n}x{n}"), Work::Flops(2.0 / 3.0 * nf * nf * nf), &mut || al.lu().lu.eval().unwrap());
        h.case("dense/solve", &format!("{n}x{n}, 1 rhs"), Work::Flops(2.0 / 3.0 * nf * nf * nf), &mut || al.solve(&bl).eval().unwrap());
        if n <= 1024 {
            let inv = host(&al.inverse());
            let z: Vec<f64> = values(n, 5, -1.0, 1.0).iter().map(|&v| f64::from(v)).collect();
            let y = matvec(&|i, j| f64::from(inv[i * n + j]), n, &z);
            let ay = matvec(&|i, j| f64::from(a[i * n + j]), n, &y);
            assert!(rel_diff(&ay, &z) < 1e-4, "inverse {n}: {}", rel_diff(&ay, &z));
            h.case("dense/inverse", &format!("{n}x{n}"), Work::Flops(2.0 * nf * nf * nf), &mut || al.inverse().eval().unwrap());
        }
        h.case("dense/det", &format!("{n}x{n}"), Work::None, &mut || {
            al.det().item().unwrap();
        });

        // QR of a tall matrix and of the square one, with the least-squares solve.
        let m = 2 * n;
        let t = values(n * m, 130 + n as u64, -1.0, 1.0);
        let tl = upload(h, &t, [n, m]);
        let f = tl.qr();
        let (q, rr) = (host(&f.q_thin()), host(&f.r()));
        let (qd, rd, td): (Vec<f64>, Vec<f64>, Vec<f64>) = (q.iter().map(|&v| f64::from(v)).collect(), rr.iter().map(|&v| f64::from(v)).collect(), t.iter().map(|&v| f64::from(v)).collect());
        let z: Vec<f64> = values(n, 6, -1.0, 1.0).iter().map(|&v| f64::from(v)).collect();
        let rz = matvec(&|i, j| rd[i * n + j], n, &z);
        let qrz: Vec<f64> = (0..m).map(|i| (0..n).map(|j| qd[i * n + j] * rz[j]).sum()).collect();
        let tz: Vec<f64> = (0..m).map(|i| (0..n).map(|j| td[i * n + j] * z[j]).sum()).collect();
        assert!(rel_diff(&qrz, &tz) < 1e-4, "qr {m}x{n}: {}", rel_diff(&qrz, &tz));
        let qtq: Vec<f64> = (0..n).map(|i| (0..m).map(|r| qd[r * n + i] * qd[r * n]).sum()).collect();
        assert!((qtq[0] - 1.0).abs() < 1e-4 && qtq[n - 1].abs() < 1e-4, "qr {m}x{n}: Q not orthonormal");
        let flops = 2.0 * (m * n * n) as f64 - 2.0 / 3.0 * (n * n * n) as f64;
        h.case("dense/qr", &format!("{m}x{n}"), Work::Flops(flops), &mut || {
            let f = tl.qr();
            f.qr.eval().unwrap();
            f.t.eval().unwrap();
        });
        let rhs = values(m, 131, -1.0, 1.0);
        let rl = upload(h, &rhs, [1, m]);
        let x = widen(&host(&tl.lstsq(&rl)));
        let res: Vec<f64> = (0..m).map(|i| (0..n).map(|j| td[i * n + j] * x[j]).sum::<f64>() - f64::from(rhs[i])).collect();
        let atr: Vec<f64> = (0..n).map(|j| (0..m).map(|i| td[i * n + j] * res[i]).sum()).collect();
        let nrm = |v: &[f64]| v.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!(nrm(&atr) < 1e-4 * nrm(&td) * nrm(&res), "lstsq {m}x{n}: |Aᵀ r| = {}", nrm(&atr));
        h.case("dense/lstsq", &format!("{m}x{n}, 1 rhs"), Work::Flops(flops + 4.0 * (m * n) as f64), &mut || tl.lstsq(&rl).eval().unwrap());
    }
    let batches: &[(usize, usize)] = pick(h, &[(8, 1024), (32, 256)], &[(8, 16384), (16, 4096), (32, 1024)]);
    for &(n, batch) in batches {
        let mut data = Vec::with_capacity(batch * n * n);
        let mut rhs = Vec::with_capacity(batch * n);
        for b in 0..batch {
            data.extend(gen_mat(n, 300 + b as u64));
            rhs.extend(values(n, 700 + b as u64, -1.0, 1.0));
        }
        let (al, rl) = (upload(h, &data, [n, n, batch]), upload(h, &rhs, [1, n, batch]));
        let x = host(&al.solve(&rl));
        for b in [0, batch - 1] {
            let r = solve_error(&data[b * n * n..(b + 1) * n * n], n, &x[b * n..(b + 1) * n], &rhs[b * n..(b + 1) * n]);
            assert!(r < 1e-5, "batched solve {n}x{batch}: matrix {b}: {r}");
        }
        let nf = n as f64 * n as f64 * n as f64 * batch as f64;
        let size = format!("{batch} x {n}x{n}");
        h.case("dense/lu_batched", &size, Work::Flops(2.0 / 3.0 * nf), &mut || al.lu().lu.eval().unwrap());
        h.case("dense/solve_batched", &size, Work::Flops(2.0 / 3.0 * nf), &mut || al.solve(&rl).eval().unwrap());
        h.case("dense/inverse_batched", &size, Work::Flops(2.0 * nf), &mut || al.inverse().eval().unwrap());
        h.case("dense/det_batched", &size, Work::None, &mut || al.det().eval().unwrap());
        h.case("dense/qr_batched", &size, Work::Flops(4.0 / 3.0 * nf), &mut || {
            let f = al.qr();
            f.qr.eval().unwrap();
            f.t.eval().unwrap();
        });
    }
}

/// Launches per factorisation (W2-U3b): what a call costs in kernel launches and in host time per launch, at every size from 64 to
/// 4096. `LUMA_BENCH_KERNELS=1` lists the launches per kernel label of each case (the class breakdown). The launches come from the
/// profiler, which the CPU backend does not have: there the cases only time the call.
fn launches(h: &mut Harness) {
    let sizes: &[usize] = if h.ctx.is_cpu() { pick(h, &[64, 128], &[64, 128, 256]) } else { pick(h, &[64, 256, 1024], &[64, 128, 256, 512, 1024, 2048, 4096]) };
    for &n in sizes {
        let a = dd_spd(n, 700 + n as u64);
        let al = upload(h, &a, [n, n]);
        let bl = upload(h, &values(n, 701, -1.0, 1.0), [1, n]);
        let tag = format!("n={n}");
        h.case("dense/launches_lu", &tag, Work::None, &mut || al.lu().lu.eval().unwrap());
        h.case("dense/launches_cholesky", &tag, Work::None, &mut || al.cholesky().eval().unwrap());
        h.case("dense/launches_qr", &tag, Work::None, &mut || {
            let f = al.qr();
            f.qr.eval().unwrap();
            f.t.eval().unwrap();
        });
        h.case("dense/launches_solve", &tag, Work::None, &mut || al.solve(&bl).eval().unwrap());
        h.case("dense/launches_solve_spd", &tag, Work::None, &mut || al.solve_spd(&bl).eval().unwrap());
        h.case("dense/launches_inverse", &tag, Work::None, &mut || al.inverse().eval().unwrap());
    }
}

/// `solve_refined` against `solve` of the same system (one right-hand side, `kappa` of a few), the ratio the spec gates at 1.3, and a
/// batch of small systems, which is launch bound and only reported.
fn refined(h: &mut Harness) {
    let sizes: &[usize] = if h.ctx.is_cpu() { pick(h, &[128], &[128, 256]) } else { pick(h, &[1024], &[1024, 2048, 4096]) };
    for &n in sizes {
        let a = gen_mat(n, 800 + n as u64);
        let al = upload(h, &a, [n, n]);
        let b = values(n, 801, -1.0, 1.0);
        let bl = upload(h, &b, [1, n]);
        let refined = al.solve_refined(&bl);
        assert!(refined.converged, "solve_refined {n}: not converged ({})", refined.achieved_rel_err);
        let x = host(&refined.x);
        let r = solve_error(&a, n, &x, &b);
        assert!(r < 1e-6, "solve_refined {n}: backward error {r}");
        let nf = n as f64;
        h.case("dense/solve_plain", &format!("{n}x{n}, 1 rhs"), Work::Flops(2.0 / 3.0 * nf * nf * nf), &mut || al.solve(&bl).eval().unwrap());
        h.case("dense/solve_refined", &format!("{n}x{n}, 1 rhs"), Work::Flops(2.0 / 3.0 * nf * nf * nf), &mut || {
            al.solve_refined(&bl).x.eval().unwrap();
        });
    }
    let (n, batch) = (64usize, 1000usize);
    let mut data = Vec::with_capacity(batch * n * n);
    for m in 0..batch {
        data.extend(gen_mat(n, 900 + m as u64));
    }
    let al = upload(h, &data, [n, n, batch]);
    let bl = upload(h, &values(n * batch, 902, -1.0, 1.0), [1, n, batch]);
    assert!(al.solve_refined(&bl).converged, "solve_refined batch");
    h.case("dense/solve_plain_batch", &format!("{n}x{n} x{batch}"), Work::None, &mut || al.solve(&bl).eval().unwrap());
    h.case("dense/solve_refined_batch", &format!("{n}x{n} x{batch}"), Work::None, &mut || {
        al.solve_refined(&bl).x.eval().unwrap();
    });
}

pub fn run(h: &mut Harness) {
    h.guarded("dense", general);
    h.guarded("dense", launches);
    h.guarded("dense", refined);
    h.guarded("dense", |h| {
        let sizes: &[usize] = if h.ctx.is_cpu() { pick(h, &[128, 256], &[256, 512, 1024]) } else { pick(h, &[256, 1024], &[256, 1024, 4096]) };
        for &n in sizes {
            let a = dd_spd(n, 90 + n as u64);
            let al = upload(h, &a, [n, n]);
            let l = host(&al.cholesky());
            let err = chol_error(&l, &a, n);
            assert!(err < 2e-4, "cholesky {n}: backward error {err}");
            let flops = (n as f64).powi(3) / 3.0;
            h.case("dense/cholesky", &format!("{n}x{n}"), Work::Flops(flops), &mut || al.cholesky().eval().unwrap());

            let k = 16.min(n);
            let b = values(n * k, 91, -1.0, 1.0);
            let bl = upload(h, &b, [k, n]);
            let x = host(&al.solve_spd(&bl));
            let (xv, bv): (Vec<f64>, Vec<f64>) = (x.iter().map(|&v| f64::from(v)).collect(), b.iter().map(|&v| f64::from(v)).collect());
            let ax: Vec<f64> = (0..n * k).map(|ij| (0..n).map(|p| f64::from(a[(ij / k) * n + p]) * xv[p * k + ij % k]).sum()).collect();
            let r = rel_diff(&ax, &bv);
            assert!(r < 1e-4, "solve_spd {n}: residual {r}");
            h.case("dense/solve_spd", &format!("{n}x{n}, {k} rhs"), Work::Flops(flops + 2.0 * (n * n * k) as f64), &mut || al.solve_spd(&bl).eval().unwrap());
        }
        let batches: &[(usize, usize)] = pick(h, &[(8, 1024), (32, 256)], &[(8, 16384), (16, 4096), (32, 1024), (64, 256)]);
        for &(n, batch) in batches {
            let mut data = Vec::with_capacity(batch * n * n);
            for b in 0..batch {
                data.extend(dd_spd(n, 5 + b as u64));
            }
            let al = upload(h, &data, [n, n, batch]);
            let l = host(&al.cholesky());
            for b in [0, batch / 2, batch - 1] {
                let err = chol_error(&l[b * n * n..(b + 1) * n * n], &data[b * n * n..(b + 1) * n * n], n);
                assert!(err < 2e-4, "batched cholesky {n}x{batch}: matrix {b}: {err}");
            }
            let flops = (n as f64).powi(3) / 3.0 * batch as f64;
            h.case("dense/cholesky_batched", &format!("{batch} x {n}x{n}"), Work::Flops(flops), &mut || al.cholesky().eval().unwrap());
        }
    });
}
