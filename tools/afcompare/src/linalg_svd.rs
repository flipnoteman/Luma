//! `svd`, `singular_values`, `pinv` and `rank` against ArrayFire's `svd` / `pinverse` / `rank`
//! (cuSOLVER `gesvd` on CUDA). ArrayFire has no symmetric eigensolver and no batched LAPACK, so
//! there is no `eigh` row and a batch is a loop of ArrayFire calls over arrays of their own.
//!
//! Matrices are row-major on the host, `[cols, rows]` in both libraries, so ArrayFire sees the
//! transpose of the host matrix `A` and decomposes `Aᵀ`; each side is verified on its own by a
//! host matrix-vector residual against the matrix it was given (`O(n^2)`).

use std::time::Instant;

use arrayfire as af;

use crate::afx;
use crate::bench::Bench;
use crate::linalg::{at, Layout};
use crate::lumax;
use crate::util::*;

/// `y = M z` for a `rows x cols` matrix given as `get(i, j)`.
fn matvec(get: &dyn Fn(usize, usize) -> f64, rows: usize, cols: usize, z: &[f64]) -> Vec<f64> {
    (0..rows).map(|i| (0..cols).map(|j| get(i, j) * z[j]).sum()).collect()
}

fn rel(a: &[f64], b: &[f64]) -> f64 {
    let d: f64 = a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f64>().sqrt();
    d / b.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-300)
}

fn zvec(n: usize) -> Vec<f64> {
    uniform(n, 5, -1.0, 1.0).iter().map(|&x| f64::from(x)).collect()
}

/// Backward error `|M z - U S Vt z| / |M z|` and the loss of orthogonality `max(|Uᵀ U z - z|, |V Vᵀ z - z|) / |z|`
/// of an `n x n` decomposition in `layout` of the matrix `get`.
fn svd_error(get: &dyn Fn(usize, usize) -> f64, n: usize, u: &[f32], s: &[f32], vt: &[f32], layout: Layout) -> (f64, f64) {
    let z = zvec(n);
    let uf = |i: usize, j: usize| at(u, n, n, layout, i, j);
    let vf = |i: usize, j: usize| at(vt, n, n, layout, i, j);
    let sz: Vec<f64> = matvec(&vf, n, n, &z).iter().zip(s).map(|(x, &s)| x * f64::from(s)).collect();
    let back = rel(&matvec(&uf, n, n, &sz), &matvec(get, n, n, &z));
    let utuz = matvec(&|i, j| uf(j, i), n, n, &matvec(&uf, n, n, &z));
    let vvtz = matvec(&|i, j| vf(j, i), n, n, &matvec(&vf, n, n, &z));
    (back, rel(&utuz, &z).max(rel(&vvtz, &z)))
}

fn verdict(label: &str, l: (f64, f64), a: (f64, f64), tol: f64) -> Verdict {
    let text = format!("{label} backward / orthogonality: luma {:.1e} / {:.1e}, AF {:.1e} / {:.1e}", l.0, l.1, a.0, a.1);
    if l.0 < tol && a.0 < tol && l.1 < 10.0 * tol && a.1 < 10.0 * tol { Verdict::yes(text) } else { Verdict::no(text) }
}

/// The seconds one call of `f` takes.
fn seconds(f: &mut dyn FnMut()) -> f64 {
    let t = Instant::now();
    f();
    t.elapsed().as_secs_f64()
}

fn sizes(b: &Bench) -> Vec<usize> {
    if b.quick { vec![256, 1024] } else { vec![64, 128, 256, 1024, 4096] }
}

/// A random `n x n` matrix of known rank `r`: `B C` with uniform factors scaled to unit-order values.
fn low_rank(n: usize, r: usize, seed: u64) -> Vec<f32> {
    let bm = uniform(n * r, seed, -1.0, 1.0);
    let cm = uniform(r * n, seed + 1, -1.0, 1.0);
    let s = 1.0 / (r as f32).sqrt();
    let mut a = vec![0.0f32; n * n];
    for i in 0..n {
        for k in 0..r {
            let x = bm[i * r + k] * s;
            for j in 0..n {
                a[i * n + j] += x * cm[k * n + j];
            }
        }
    }
    a
}

pub fn svd_rows(b: &mut Bench) {
    b.guarded("linalg_svd", |b| {
        single(b);
        pinv_rank(b);
        batched(b);
    });
}

/// One big matrix: svd (with vectors) and the singular values alone.
fn single(b: &mut Bench) {
    let ctx = b.ctx.clone();
    for n in sizes(b) {
        let a = uniform(n * n, 140 + n as u64, -1.0, 1.0);
        let (al, aa) = (lumax::arr(&ctx, &a, [n, n]), afx::arr(&a, &[n as u64, n as u64]));
        // A call this slow is not timed: 4096 only when both sides finish in about a minute.
        if n >= 4096 {
            let tl = seconds(&mut || {
                let r = al.svd();
                luma::eval_all(&[&r.u, &r.s, &r.vt]).unwrap();
                ctx.sync().unwrap();
            });
            let ta = seconds(&mut || {
                let (u, s, vt) = af::svd(&aa);
                u.eval();
                s.eval();
                vt.eval();
                afx::sync();
            });
            if tl > 60.0 || ta > 60.0 {
                println!("  skipping svd {n}x{n}: luma {tl:.1} s, AF {ta:.1} s (more than a minute)");
                continue;
            }
        }
        let svd = al.svd();
        let (lu, ls, lvt) = (lumax::host(&svd.u), lumax::host(&svd.s), lumax::host(&svd.vt));
        let (au, as_, avt) = af::svd(&aa);
        let (au, as_, avt) = (afx::host(&au), afx::host(&as_), afx::host(&avt));
        let l_err = svd_error(&|i, j| f64::from(a[i * n + j]), n, &lu, &ls, &lvt, Layout::Row);
        // ArrayFire decomposes the transpose of the host matrix.
        let a_err = svd_error(&|i, j| f64::from(a[j * n + i]), n, &au, &as_, &avt, Layout::Col);
        let top = f64::from(ls[0]);
        let sv_diff = ls.iter().zip(&as_).map(|(x, y)| f64::from((x - y).abs())).fold(0.0, f64::max) / top;
        let v = verdict("svd", l_err, a_err, 1e-4).and(if sv_diff < 1e-4 { Verdict::yes(format!("values differ by {sv_diff:.1e} of the largest")) } else { Verdict::no(format!("values differ by {sv_diff:.1e}")) });
        b.run(
            "svd f32",
            &format!("{n}x{n}"),
            v,
            "luma one-sided Jacobi (u, s, vt); AF svd (cuSOLVER gesvd)",
            &mut || {
                let r = al.svd();
                luma::eval_all(&[&r.u, &r.s, &r.vt]).unwrap();
                ctx.sync().unwrap();
            },
            &mut || {
                let (u, s, vt) = af::svd(&aa);
                u.eval();
                s.eval();
                vt.eval();
                afx::sync();
            },
        );
        let lv = lumax::host(&al.singular_values());
        let dv = lv.iter().zip(&ls).map(|(x, y)| f64::from((x - y).abs())).fold(0.0, f64::max) / top;
        let v = if dv < 1e-5 { Verdict::yes(format!("matches svd().s to {dv:.1e}")) } else { Verdict::no(format!("differs from svd().s by {dv:.1e}")) };
        b.run(
            "singular values f32",
            &format!("{n}x{n}"),
            v,
            "luma singular_values (no V); AF has no values-only API, its full svd is timed",
            &mut || {
                al.singular_values().eval().unwrap();
                ctx.sync().unwrap();
            },
            &mut || {
                let (u, s, vt) = af::svd(&aa);
                u.eval();
                s.eval();
                vt.eval();
                afx::sync();
            },
        );
    }
}

/// `pinv` and `rank` of a full rank matrix and of one of known rank.
fn pinv_rank(b: &mut Bench) {
    let ctx = b.ctx.clone();
    let sizes: Vec<usize> = sizes(b).into_iter().filter(|&n| n <= 1024).collect();
    for n in sizes {
        let a = uniform(n * n, 150 + n as u64, -1.0, 1.0);
        let (al, aa) = (lumax::arr(&ctx, &a, [n, n]), afx::arr(&a, &[n as u64, n as u64]));
        // A pinv(A) z = A z for both, each on its own matrix (AF's is the transpose).
        let lp = lumax::host(&al.pinv_tol(1e-6));
        let ap = afx::host(&af::pinverse(&aa, 1e-6, af::MatProp::NONE));
        let z = zvec(n);
        let check = |get: &dyn Fn(usize, usize) -> f64, p: &[f32], layout: Layout| {
            let pz = matvec(&|i, j| at(p, n, n, layout, i, j), n, n, &matvec(get, n, n, &z));
            rel(&matvec(get, n, n, &pz), &matvec(get, n, n, &z))
        };
        let (el, ea) = (check(&|i, j| f64::from(a[i * n + j]), &lp, Layout::Row), check(&|i, j| f64::from(a[j * n + i]), &ap, Layout::Col));
        let text = format!("|A P A z - A z|/|A z|: luma {el:.1e}, AF {ea:.1e}");
        let v = if el < 5e-3 && ea < 5e-3 { Verdict::yes(text) } else { Verdict::no(text) };
        b.run(
            "pinv f32",
            &format!("{n}x{n}"),
            v,
            "luma pinv_tol(1e-6) (Jacobi svd + matmul_f32); AF pinverse(tol 1e-6)",
            &mut || {
                al.pinv_tol(1e-6).eval().unwrap();
                ctx.sync().unwrap();
            },
            &mut || {
                af::pinverse(&aa, 1e-6, af::MatProp::NONE).eval();
                afx::sync();
            },
        );
        let r = n / 2;
        let lr = low_rank(n, r, 160 + n as u64);
        let (rl, ra) = (lumax::arr(&ctx, &lr, [n, n]), afx::arr(&lr, &[n as u64, n as u64]));
        let (got_l, got_a) = (rl.rank().item().unwrap(), af::rank(&ra, 1e-2));
        let text = format!("rank {r}: luma {got_l}, AF {got_a}");
        let v = if got_l as usize == r && got_a as usize == r { Verdict::yes(text) } else { Verdict::no(text) };
        b.run(
            "rank f32",
            &format!("{n}x{n}, rank {r}"),
            v,
            "luma rank (singular values, count above the default relative tolerance); AF rank(tol 1e-2), far cheaper than AF's own svd; result read back on both sides",
            &mut || {
                let _ = rl.rank().item().unwrap();
            },
            &mut || {
                let _ = af::rank(&ra, 1e-2);
                afx::sync();
            },
        );
    }
}

/// A batch of small matrices: one luma launch against a loop of ArrayFire calls.
fn batched(b: &mut Bench) {
    let ctx = b.ctx.clone();
    let configs: &[(usize, usize)] = if b.quick { &[(16, 256)] } else { &[(8, 2048), (16, 1024), (32, 1024)] };
    for &(n, batch) in configs {
        let mut data = Vec::new();
        for i in 0..batch {
            data.extend(uniform(n * n, 700 + i as u64, -1.0, 1.0));
        }
        let al = lumax::arr(&ctx, &data, [n, n, batch]);
        let slices: Vec<af::Array<f32>> = (0..batch).map(|i| afx::arr(&data[i * n * n..(i + 1) * n * n], &[n as u64, n as u64])).collect();
        let size = format!("{batch} x {n}x{n}");
        let svd = al.svd();
        let (lu, ls, lvt) = (lumax::host(&svd.u), lumax::host(&svd.s), lumax::host(&svd.vt));
        let mut worst = (0.0f64, 0.0f64);
        for i in [0, batch / 2, batch - 1] {
            let m = &data[i * n * n..(i + 1) * n * n];
            let e = svd_error(&|r, c| f64::from(m[r * n + c]), n, &lu[i * n * n..(i + 1) * n * n], &ls[i * n..(i + 1) * n], &lvt[i * n * n..(i + 1) * n * n], Layout::Row);
            worst = (worst.0.max(e.0), worst.1.max(e.1));
        }
        let last = &data[(batch - 1) * n * n..];
        let (au, as_, avt) = af::svd(&slices[batch - 1]);
        let ea = svd_error(&|r, c| f64::from(last[c * n + r]), n, &afx::host(&au), &afx::host(&as_), &afx::host(&avt), Layout::Col);
        b.run(
            "batched svd f32",
            &size,
            verdict("svd", worst, ea, 1e-4),
            "one launch (one workgroup per matrix) vs a loop of AF svd calls (AF has no batched LAPACK)",
            &mut || {
                let r = al.svd();
                luma::eval_all(&[&r.u, &r.s, &r.vt]).unwrap();
                ctx.sync().unwrap();
            },
            &mut || {
                for a in &slices {
                    let (u, s, vt) = af::svd(a);
                    u.eval();
                    s.eval();
                    vt.eval();
                }
                afx::sync();
            },
        );
        b.run_one(
            "batched singular values f32",
            &size,
            Some(&mut || {
                al.singular_values().eval().unwrap();
                ctx.sync().unwrap();
            }),
            Some(&mut || {
                for a in &slices {
                    let (u, s, vt) = af::svd(a);
                    u.eval();
                    s.eval();
                    vt.eval();
                }
                afx::sync();
            }),
            "luma singular_values vs a loop of AF full svd calls (no values-only API); separate timings",
        );
    }
}
