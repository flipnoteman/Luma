//! Dense linear algebra: luma's `cholesky` / `solve_spd` / `solve_triangular` / `lu` / `solve` /
//! `inverse` / `det` / `qr` / `svd` against ArrayFire's LAPACK functions (cuSOLVER on CUDA).
//!
//! Matrices are row-major on the host, `[cols, rows]` in both libraries, so ArrayFire sees the
//! transpose of the host matrix `A` (symmetric inputs are the same matrix). Results are verified
//! by residuals computed on the host (a matrix-vector product with random vectors, O(n^2)), for
//! each library on its own, not by comparing the factors.
//!
//! ArrayFire has no batched LAPACK: a batch is a loop of calls over arrays of their own.

use arrayfire as af;

use crate::afx;
use crate::bench::Bench;
use crate::cases::{A, L};
use crate::lumax;
use crate::util::*;

/// A symmetric, strictly diagonally dominant (so positive definite) `n x n` matrix.
pub fn dd_spd(n: usize, seed: u64) -> Vec<f32> {
    let r = uniform(n * n, seed, -0.5, 0.5);
    let mut a = vec![0.0f32; n * n];
    for i in 0..n {
        for j in 0..=i {
            let v = r[i * n + j];
            a[i * n + j] = v;
            a[j * n + i] = v;
        }
        a[i * n + i] = n as f32 * 0.5 + 1.0 + r[i * n + i].abs();
    }
    a
}

/// `y = M z` for a `rows x cols` matrix given as `get(i, j)`.
fn matvec(get: &dyn Fn(usize, usize) -> f64, rows: usize, cols: usize, z: &[f64]) -> Vec<f64> {
    (0..rows).map(|i| (0..cols).map(|j| get(i, j) * z[j]).sum()).collect()
}

fn rel(a: &[f64], b: &[f64]) -> f64 {
    let d: f64 = a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f64>().sqrt();
    d / b.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-300)
}

/// How a host copy of a matrix is laid out.
#[derive(Clone, Copy)]
pub enum Layout {
    /// luma: element `(i, j)` at `j + cols * i`.
    Row,
    /// ArrayFire: element `(i, j)` at `i + rows * j` (the host matrix `A` appears transposed).
    Col,
}

pub fn at(buf: &[f32], rows: usize, cols: usize, layout: Layout, i: usize, j: usize) -> f64 {
    f64::from(match layout {
        Layout::Row => buf[j + cols * i],
        Layout::Col => buf[i + rows * j],
    })
}

/// `|| L Lᵀ z - A z || / || A z ||` for the factor `l` holding the lower triangle (the other
/// triangle is ignored) of a matrix in `layout`; `a` is the symmetric host matrix.
fn chol_error(l: &[f32], a: &[f32], n: usize, layout: Layout, lower: bool) -> f64 {
    let z: Vec<f64> = uniform(n, 5, -1.0, 1.0).iter().map(|&x| f64::from(x)).collect();
    let tri = |i: usize, j: usize| -> f64 {
        // The factor as a mathematical lower triangular matrix (`lower` says which triangle of the buffer's
        // matrix holds it; an upper one is read transposed).
        let (r, c) = if lower { (i, j) } else { (j, i) };
        if c > r { 0.0 } else { at(l, n, n, layout, r, c) }
    };
    let lt_z = matvec(&|i, j| tri(j, i), n, n, &z);
    let llt_z = matvec(&tri, n, n, &lt_z);
    let az = matvec(&|i, j| f64::from(a[i * n + j]), n, n, &z);
    rel(&llt_z, &az)
}

fn sizes(b: &Bench) -> Vec<usize> {
    if b.quick { vec![256, 1024] } else { vec![256, 1024, 4096] }
}

/// `I + 0.17 R / sqrt(n)`: a general (nonsymmetric) matrix with spectrum in a disk of radius 0.1
/// around 1, so its determinant stays inside `f32` and it is well conditioned.
pub fn gen_mat(n: usize, seed: u64) -> Vec<f32> {
    let mut a = uniform(n * n, seed, -1.0, 1.0);
    let s = 0.1732 / (n as f32).sqrt();
    for v in &mut a {
        *v *= s;
    }
    for i in 0..n {
        a[i * n + i] += 1.0;
    }
    a
}

fn fro(a: &[f32]) -> f64 {
    a.iter().map(|&x| f64::from(x) * f64::from(x)).sum::<f64>().sqrt()
}

/// Backward error `|M x - b| / (|M|_F |x| + |b|)` of one solution vector `x` of `M x = b`, `M` as `get(i, j)`.
fn solve_error(get: &dyn Fn(usize, usize) -> f64, n: usize, norm: f64, x: &[f32], b: &[f32]) -> f64 {
    let xd: Vec<f64> = x.iter().map(|&v| f64::from(v)).collect();
    let mx = matvec(get, n, n, &xd);
    let bd: Vec<f64> = b.iter().map(|&v| f64::from(v)).collect();
    let d: f64 = mx.iter().zip(&bd).map(|(p, q)| (p - q) * (p - q)).sum::<f64>().sqrt();
    d / (norm * xd.iter().map(|v| v * v).sum::<f64>().sqrt() + bd.iter().map(|v| v * v).sum::<f64>().sqrt())
}

fn verdict2(label: &str, l: f64, a: f64, tol: f64) -> Verdict {
    let text = format!("{label}: luma {l:.1e}, AF {a:.1e}");
    if l < tol && a < tol { Verdict::yes(text) } else { Verdict::no(text) }
}

fn cholesky(b: &mut Bench) {
    let ctx = b.ctx.clone();
    for n in sizes(b) {
        let a = dd_spd(n, 70 + n as u64);
        let (al, aa) = (lumax::arr(&ctx, &a, [n, n]), afx::arr(&a, &[n as u64, n as u64]));
        let lf = |al: &L| al.cholesky();
        let lout = lumax::host(&lf(&al));
        let (aout, info) = af::cholesky(&aa, false);
        let aout = afx::host(&aout);
        let v = {
            let e_l = chol_error(&lout, &a, n, Layout::Row, true);
            let e_a = chol_error(&aout, &a, n, Layout::Col, true);
            let text = format!("|LLt z - A z|/|A z|: luma {e_l:.1e}, AF {e_a:.1e}");
            if e_l < 2e-4 && e_a < 2e-4 && info == 0 { Verdict::yes(text) } else { Verdict::no(text) }
        };
        b.run(
            "cholesky f32",
            &format!("{n}x{n}"),
            v,
            "AF cholesky (cuSOLVER potrf) out of place, lower",
            &mut || {
                lf(&al).eval().unwrap();
                ctx.sync().unwrap();
            },
            &mut || {
                let (r, _) = af::cholesky(&aa, false);
                r.eval();
                afx::sync();
            },
        );
    }
}

pub fn linalg(b: &mut Bench) {
    b.guarded("linalg", |b| {
        cholesky(b);
        triangular(b);
        spd_solve(b);
        lu_rows(b);
        general_solve(b);
        inverse_det(b);
        qr_rows(b);
        batched(b);
    });
}

fn triangular(b: &mut Bench) {
    let ctx = b.ctx.clone();
    for n in sizes(b) {
        // A lower triangular matrix with a dominant diagonal and 16 right-hand sides.
        let mut t = uniform(n * n, 80 + n as u64, -1.0, 1.0);
        for i in 0..n {
            for j in 0..n {
                t[i * n + j] = if j < i { t[i * n + j] / (n as f32).sqrt() } else if j == i { 2.0 } else { 0.0 };
            }
        }
        let k = 16usize;
        let rhs = uniform(n * k, 81, -1.0, 1.0);
        // luma: row-major [k, n]; ArrayFire solves T^T-as-seen = upper for the same buffer, so give it the
        // transposed buffer to solve the same lower system (right-hand sides column-major [n, k]).
        let tt: Vec<f32> = (0..n * n).map(|p| t[(p % n) * n + p / n]).collect();
        let rhs_cm: Vec<f32> = (0..n * k).map(|p| rhs[(p % n) * k + p / n]).collect();
        let (tl, rl) = (lumax::arr(&ctx, &t, [n, n]), lumax::arr(&ctx, &rhs, [k, n]));
        let (ta, ra) = (afx::arr(&tt, &[n as u64, n as u64]), afx::arr(&rhs_cm, &[n as u64, k as u64]));
        let lf = |tl: &L, rl: &L| tl.solve_triangular(rl, luma::Triangle::Lower, false, false);
        let af_solve = |ta: &A, ra: &A| af::solve(ta, ra, af::MatProp::LOWER);
        let lx = lumax::host(&lf(&tl, &rl));
        let ax = afx::host(&af_solve(&ta, &ra));
        let norm = fro(&t);
        let col = |buf: &[f32], c: usize, rm: bool| -> Vec<f32> { (0..n).map(|i| if rm { buf[i * k + c] } else { buf[i + n * c] }).collect() };
        let get = |i: usize, j: usize| f64::from(t[i * n + j]);
        let (mut el, mut ea) = (0.0f64, 0.0f64);
        for c in [0, k - 1] {
            el = el.max(solve_error(&get, n, norm, &col(&lx, c, true), &col(&rhs, c, true)));
            ea = ea.max(solve_error(&get, n, norm, &col(&ax, c, false), &col(&rhs_cm, c, false)));
        }
        b.run(
            "solve_triangular lower f32",
            &format!("{n}x{n}, {k} rhs"),
            verdict2("backward error", el, ea, 1e-5),
            "AF solve(MatProp::LOWER) (trsm)",
            &mut || {
                lf(&tl, &rl).eval().unwrap();
                ctx.sync().unwrap();
            },
            &mut || {
                af_solve(&ta, &ra).eval();
                afx::sync();
            },
        );
    }
}

fn spd_solve(b: &mut Bench) {
    let ctx = b.ctx.clone();
    for n in sizes(b) {
        let a = dd_spd(n, 70 + n as u64);
        let rhs = uniform(n, 82, -1.0, 1.0);
        let (al, aa) = (lumax::arr(&ctx, &a, [n, n]), afx::arr(&a, &[n as u64, n as u64]));
        let (rl, ra) = (lumax::arr(&ctx, &rhs, [1, n]), afx::arr(&rhs, &[n as u64]));
        let lf = |al: &L, rl: &L| al.solve_spd(rl);
        // cholesky + two triangular solves, the lower factor and its transpose.
        let af_solve = |aa: &A, ra: &A| {
            let (l, _) = af::cholesky(aa, false);
            let y = af::solve(&l, ra, af::MatProp::LOWER);
            af::solve(&af::transpose(&l, false), &y, af::MatProp::UPPER)
        };
        let lx = lumax::host(&lf(&al, &rl));
        let ax = afx::host(&af_solve(&aa, &ra));
        let get = |i: usize, j: usize| f64::from(a[i * n + j]);
        let norm = fro(&a);
        let (el, ea) = (solve_error(&get, n, norm, &lx, &rhs), solve_error(&get, n, norm, &ax, &rhs));
        b.run(
            "solve SPD f32",
            &format!("{n}x{n}, 1 rhs"),
            verdict2("backward error", el, ea, 1e-5),
            "luma solve_spd; AF cholesky + 2 x solve(LOWER/UPPER)",
            &mut || {
                lf(&al, &rl).eval().unwrap();
                ctx.sync().unwrap();
            },
            &mut || {
                af_solve(&aa, &ra).eval();
                afx::sync();
            },
        );
    }
}

fn lu_rows(b: &mut Bench) {
    let ctx = b.ctx.clone();
    for n in sizes(b) {
        for (label, a) in [("random", uniform(n * n, 60 + n as u64, -1.0, 1.0)), ("near identity", gen_mat(n, 61 + n as u64))] {
            if label == "random" && n == 256 {
                continue;
            }
            let (al, aa) = (lumax::arr(&ctx, &a, [n, n]), afx::arr(&a, &[n as u64, n as u64]));
            let f = al.lu();
            let (lu_l, perm) = (lumax::host(&f.lu), f.perm.to_vec().unwrap());
            let mut c = aa.copy();
            let piv = af::lu_inplace(&mut c, true);
            let (lu_a, piv) = (afx::host(&c), afx::host(&piv));
            // P A = L U checked on a vector: L (U z) against A[perm] z.
            let z: Vec<f64> = uniform(n, 9, -1.0, 1.0).iter().map(|&x| f64::from(x)).collect();
            let az = matvec(&|i, j| f64::from(a[i * n + j]), n, n, &z);
            let uz = matvec(&|i, j| if j >= i { at(&lu_l, n, n, Layout::Row, i, j) } else { 0.0 }, n, n, &z);
            let luz = matvec(&|i, j| if j < i { at(&lu_l, n, n, Layout::Row, i, j) } else if i == j { 1.0 } else { 0.0 }, n, n, &uz);
            let pa: Vec<f64> = (0..n).map(|i| az[perm[i] as usize]).collect();
            let el = rel(&luz, &pa);
            // ArrayFire factors the transposed matrix (column-major view of our buffer): A^T = P^T L U with LAPACK pivots.
            let mut at_z = matvec(&|i, j| f64::from(a[j * n + i]), n, n, &z);
            let uz = matvec(&|i, j| if j >= i { at(&lu_a, n, n, Layout::Col, i, j) } else { 0.0 }, n, n, &z);
            let luz = matvec(&|i, j| if j < i { at(&lu_a, n, n, Layout::Col, i, j) } else if i == j { 1.0 } else { 0.0 }, n, n, &uz);
            for (i, &p) in piv.iter().enumerate() {
                at_z.swap(i, (p - 1) as usize);
            }
            let ea = rel(&luz, &at_z);
            let v = verdict2("|LUz - PAz|/|PAz|", el, ea, 1e-4);
            b.run(
                "lu f32",
                &format!("{n}x{n} {label}"),
                v,
                "luma lu (packed); AF lu_inplace on a copy (cuSOLVER getrf)",
                &mut || {
                    al.lu().lu.eval().unwrap();
                    ctx.sync().unwrap();
                },
                &mut || {
                    let mut c = aa.copy();
                    let _p = af::lu_inplace(&mut c, true);
                    c.eval();
                    afx::sync();
                },
            );
        }
    }
}

fn general_solve(b: &mut Bench) {
    let ctx = b.ctx.clone();
    for n in sizes(b) {
        let a = gen_mat(n, 62 + n as u64);
        let rhs = uniform(n, 83, -1.0, 1.0);
        let at_buf: Vec<f32> = (0..n * n).map(|p| a[(p % n) * n + p / n]).collect();
        let (al, aa) = (lumax::arr(&ctx, &a, [n, n]), afx::arr(&at_buf, &[n as u64, n as u64]));
        let (rl, ra) = (lumax::arr(&ctx, &rhs, [1, n]), afx::arr(&rhs, &[n as u64]));
        let lf = |al: &L, rl: &L| al.solve(rl);
        let af_solve = |aa: &A, ra: &A| af::solve(aa, ra, af::MatProp::NONE);
        let (lx, ax) = (lumax::host(&lf(&al, &rl)), afx::host(&af_solve(&aa, &ra)));
        let get = |i: usize, j: usize| f64::from(a[i * n + j]);
        let norm = fro(&a);
        let v = verdict2("backward error", solve_error(&get, n, norm, &lx, &rhs), solve_error(&get, n, norm, &ax, &rhs), 1e-5);
        b.run(
            "solve general f32",
            &format!("{n}x{n}, 1 rhs"),
            v,
            "luma solve (LU); AF solve (getrf + getrs); AF given the transposed buffer so both solve the same system",
            &mut || {
                lf(&al, &rl).eval().unwrap();
                ctx.sync().unwrap();
            },
            &mut || {
                af_solve(&aa, &ra).eval();
                afx::sync();
            },
        );
    }
}

fn inverse_det(b: &mut Bench) {
    let ctx = b.ctx.clone();
    for n in sizes(b) {
        let a = gen_mat(n, 63 + n as u64);
        let (al, aa) = (lumax::arr(&ctx, &a, [n, n]), afx::arr(&a, &[n as u64, n as u64]));
        let z: Vec<f64> = uniform(n, 11, -1.0, 1.0).iter().map(|&x| f64::from(x)).collect();
        let norm = fro(&a);
        let lf = |al: &L| al.inverse();
        let af_inv = |aa: &A| af::inverse(aa, af::MatProp::NONE);
        let (li, ai) = (lumax::host(&lf(&al)), afx::host(&af_inv(&aa)));
        // A (X z) = z; AF's inverse of the transposed-view matrix is the transpose of ours.
        let check = |x: &[f32], layout: Layout, transposed: bool| {
            let y = matvec(&|i, j| at(x, n, n, layout, i, j), n, n, &z);
            let ay = matvec(&|i, j| if transposed { f64::from(a[j * n + i]) } else { f64::from(a[i * n + j]) }, n, n, &y);
            rel(&ay, &z) / (norm * y.iter().map(|v| v * v).sum::<f64>().sqrt() / z.iter().map(|v| v * v).sum::<f64>().sqrt())
        };
        let v = verdict2("|A X z - z| scaled", check(&li, Layout::Row, false), check(&ai, Layout::Col, true), 1e-5);
        b.run(
            "inverse f32",
            &format!("{n}x{n}"),
            v,
            "luma inverse (LU + 2 triangular solves with I); AF inverse (getrf + getri)",
            &mut || {
                lf(&al).eval().unwrap();
                ctx.sync().unwrap();
            },
            &mut || {
                af_inv(&aa).eval();
                afx::sync();
            },
        );
        let ld = f64::from(al.det().item().unwrap());
        let (ad, _) = af::det(&aa);
        let text = format!("luma {ld:.6e}, AF {ad:.6e}");
        let v = if (ld - ad).abs() <= 2e-3 * ad.abs() { Verdict::yes(text) } else { Verdict::no(text) };
        b.run(
            "det f32",
            &format!("{n}x{n}"),
            v,
            "result read back on both sides",
            &mut || {
                let _ = al.det().item().unwrap();
            },
            &mut || {
                let _ = af::det(&aa);
                afx::sync();
            },
        );
    }
}

/// QR of square and tall matrices, and the least-squares solve of the tall ones.
fn qr_rows(b: &mut Bench) {
    let ctx = b.ctx.clone();
    let mut shapes: Vec<(usize, usize)> = sizes(b).into_iter().map(|n| (n, n)).collect();
    shapes.push((256, 512));
    shapes.push((1024, 2048));
    for (n, m) in shapes {
        let a = uniform(n * m, 64 + (n + m) as u64, -1.0, 1.0);
        // ArrayFire gets the same m x n matrix in its column-major layout.
        let a_cm: Vec<f32> = (0..m * n).map(|p| a[(p % m) * n + p / m]).collect();
        let (al, aa) = (lumax::arr(&ctx, &a, [n, m]), afx::arr(&a_cm, &[m as u64, n as u64]));
        let z: Vec<f64> = uniform(n, 13, -1.0, 1.0).iter().map(|&x| f64::from(x)).collect();
        let az = matvec(&|i, j| f64::from(a[i * n + j]), m, n, &z);
        let f = al.qr();
        let (q, r) = (lumax::host(&f.q_thin()), lumax::host(&f.r()));
        let rz = matvec(&|i, j| f64::from(r[i * n + j]), n, n, &z);
        let el = rel(&matvec(&|i, j| f64::from(q[i * n + j]), m, n, &rz), &az);
        let (qa, ra, _tau) = af::qr(&aa);
        let (qa, ra) = (afx::host(&qa), afx::host(&ra));
        let rz = matvec(&|i, j| if i <= j { at(&ra, m, n, Layout::Col, i, j) } else { 0.0 }, m, n, &z);
        let ea = rel(&matvec(&|i, j| at(&qa, m, m, Layout::Col, i, j), m, m, &rz), &az);
        b.run(
            "qr f32",
            &format!("{m}x{n}"),
            verdict2("|Q R z - A z|/|A z|", el, ea, 1e-4),
            "luma qr (packed factor + T, Q not formed); AF qr_inplace (cuSOLVER geqrf) on a copy",
            &mut || {
                let f = al.qr();
                f.qr.eval().unwrap();
                f.t.eval().unwrap();
                ctx.sync().unwrap();
            },
            &mut || {
                let mut c = aa.copy();
                let _tau = af::qr_inplace(&mut c);
                c.eval();
                afx::sync();
            },
        );
        if m > n {
            let rhs = uniform(m, 85, -1.0, 1.0);
            let (rl, ra_) = (lumax::arr(&ctx, &rhs, [1, m]), afx::arr(&rhs, &[m as u64]));
            let lf = |al: &L, rl: &L| al.lstsq(rl);
            let af_ls = |aa: &A, ra: &A| af::solve(aa, ra, af::MatProp::NONE);
            let (lx, ax) = (lumax::host(&lf(&al, &rl)), afx::host(&af_ls(&aa, &ra_)));
            // A^T (A x - b) must vanish.
            let orth = |x: &[f32]| -> f64 {
                let xd: Vec<f64> = x.iter().map(|&v| f64::from(v)).collect();
                let res: Vec<f64> = matvec(&|i, j| f64::from(a[i * n + j]), m, n, &xd).iter().zip(&rhs).map(|(p, q)| p - f64::from(*q)).collect();
                let atr = matvec(&|i, j| f64::from(a[j * n + i]), n, m, &res);
                atr.iter().map(|v| v * v).sum::<f64>().sqrt() / (fro(&a) * res.iter().map(|v| v * v).sum::<f64>().sqrt())
            };
            b.run(
                "lstsq f32",
                &format!("{m}x{n}, 1 rhs"),
                verdict2("|At r|/(|A||r|)", orth(&lx), orth(&ax), 1e-4),
                "luma lstsq (QR); AF solve of a tall matrix (QR)",
                &mut || {
                    lf(&al, &rl).eval().unwrap();
                    ctx.sync().unwrap();
                },
                &mut || {
                    af_ls(&aa, &ra_).eval();
                    afx::sync();
                },
            );
        }
    }
}

/// A batch of small matrices: one luma launch against a loop of ArrayFire calls.
fn batched(b: &mut Bench) {
    let ctx = b.ctx.clone();
    let configs: &[(usize, usize)] = if b.quick { &[(16, 256)] } else { &[(8, 2048), (16, 1024), (32, 1024)] };
    for &(n, batch) in configs {
        let mut spd = Vec::new();
        let mut gen_all = Vec::new();
        for i in 0..batch {
            spd.extend(dd_spd(n, 500 + i as u64));
            gen_all.extend(gen_mat(n, 900 + i as u64));
        }
        let rhs = uniform(n * batch, 84, -1.0, 1.0);
        let slices = |data: &[f32]| -> Vec<A> { (0..batch).map(|i| afx::arr(&data[i * n * n..(i + 1) * n * n], &[n as u64, n as u64])).collect() };
        let (spd_l, gen_l) = (lumax::arr(&ctx, &spd, [n, n, batch]), lumax::arr(&ctx, &gen_all, [n, n, batch]));
        let rhs_l = lumax::arr(&ctx, &rhs, [1, n, batch]);
        let (spd_a, gen_a) = (slices(&spd), slices(&gen_all));
        let rhs_a: Vec<A> = (0..batch).map(|i| afx::arr(&rhs[i * n..(i + 1) * n], &[n as u64])).collect();
        let size = format!("{batch} x {n}x{n}");

        // Cholesky.
        let lout = lumax::host(&spd_l.cholesky());
        let mut worst = 0.0f64;
        for i in [0, batch / 2, batch - 1] {
            worst = worst.max(chol_error(&lout[i * n * n..(i + 1) * n * n], &spd[i * n * n..(i + 1) * n * n], n, Layout::Row, true));
        }
        let (a0, _) = af::cholesky(&spd_a[batch - 1], false);
        let ea = chol_error(&afx::host(&a0), &spd[(batch - 1) * n * n..], n, Layout::Col, true);
        b.run("batched cholesky f32", &size, verdict2("|LLt z - A z|/|A z|", worst, ea, 2e-4), "one launch vs a loop of AF cholesky calls (AF has no batched LAPACK)", &mut || {
            spd_l.cholesky().eval().unwrap();
            ctx.sync().unwrap();
        }, &mut || {
            for a in &spd_a {
                let (r, _) = af::cholesky(a, false);
                r.eval();
            }
            afx::sync();
        });

        // General solve.
        let lx = lumax::host(&gen_l.solve(&rhs_l));
        let mut worst = 0.0f64;
        for i in [0, batch - 1] {
            let m = &gen_all[i * n * n..(i + 1) * n * n];
            worst = worst.max(solve_error(&|r, c| f64::from(m[r * n + c]), n, fro(m), &lx[i * n..(i + 1) * n], &rhs[i * n..(i + 1) * n]));
        }
        let ax = afx::host(&af::solve(&af::transpose(&gen_a[0], false), &rhs_a[0], af::MatProp::NONE));
        let m = &gen_all[..n * n];
        let ea = solve_error(&|r, c| f64::from(m[r * n + c]), n, fro(m), &ax, &rhs[..n]);
        b.run("batched solve f32", &size, verdict2("backward error", worst, ea, 1e-5), "one launch (lu + 2 trsm) vs a loop of AF solve calls", &mut || {
            gen_l.solve(&rhs_l).eval().unwrap();
            ctx.sync().unwrap();
        }, &mut || {
            for (a, r) in gen_a.iter().zip(&rhs_a) {
                af::solve(a, r, af::MatProp::NONE).eval();
            }
            afx::sync();
        });

        // Batched LU, inverse, det.
        b.run_one("batched lu f32", &size, Some(&mut || {
            gen_l.lu().lu.eval().unwrap();
            ctx.sync().unwrap();
        }), Some(&mut || {
            for a in &gen_a {
                let mut c = a.copy();
                let _ = af::lu_inplace(&mut c, true);
                c.eval();
            }
            afx::sync();
        }), "one launch vs a loop of AF lu_inplace calls (separate timings, not interleaved)");
        b.run_one("batched qr f32", &size, Some(&mut || {
            let f = gen_l.qr();
            f.qr.eval().unwrap();
            f.t.eval().unwrap();
            ctx.sync().unwrap();
        }), Some(&mut || {
            for a in &gen_a {
                let mut c = a.copy();
                let _ = af::qr_inplace(&mut c);
                c.eval();
            }
            afx::sync();
        }), "one launch vs a loop of AF qr_inplace calls (separate timings)");
        b.run_one("batched inverse f32", &size, Some(&mut || {
            gen_l.inverse().eval().unwrap();
            ctx.sync().unwrap();
        }), Some(&mut || {
            for a in &gen_a {
                af::inverse(a, af::MatProp::NONE).eval();
            }
            afx::sync();
        }), "one launch (lu + 2 trsm with I) vs a loop of AF inverse calls (separate timings)");
        b.run_one("batched det f32", &size, Some(&mut || {
            gen_l.det().eval().unwrap();
            ctx.sync().unwrap();
        }), Some(&mut || {
            for a in &gen_a {
                let _ = af::det(a);
            }
        }), "one launch (result stays on the device) vs a loop of AF det calls (each returns to the host)");
    }
}
