//! Matrix properties (W3-R6): banded solves and banded products against what they replace. `props/band_solve_*` run
//! `Array::solve_banded` on a matrix made by `from_diagonals` and the line solver called directly on the same diagonals (the pair
//! shows the overhead of the property route, which should be none); `props/band_matmul_*` run the sum of shifts that `matmul` takes
//! under `exact_finite` against the dense product of the same matrix, evaluated. The half-work products: `props/syrk*` run `xᵀ x`
//! (the half-work strategy) against the product of an equal copy (`*_full`), on the `f32` kernels (strict numerics) and with the
//! default numerics (TF32 on tensor cores); `props/trimm*` run `lower(a) · b` under `exact_finite` (`TriangularK`) against the
//! product of the evaluated matrix.

use luma::*;

use super::{host, pick, upload};
use crate::harness::*;

/// A diagonally dominant band `(lo, hi)` of `n` rows as its diagonals (the offset convention of `from_diagonals`).
fn diagonals(h: &Harness, n: usize, lo: usize, hi: usize) -> Vec<(isize, Array<f32>)> {
    let mut out = vec![(0isize, upload(h, &values(n, 700, 4.0 * (lo + hi) as f32 + 2.0, 4.0 * (lo + hi) as f32 + 3.0), [n, 1]))];
    for k in 1..=lo {
        out.push((k as isize, upload(h, &values(n - k, 710 + k as u64, 0.5, 1.0), [n - k, 1])));
    }
    for k in 1..=hi {
        out.push((-(k as isize), upload(h, &values(n - k, 730 + k as u64, 0.5, 1.0), [n - k, 1])));
    }
    out
}

pub fn run(h: &mut Harness) {
    h.guarded("props", |h| {
        let cpu = h.ctx.is_cpu();
        let n: usize = if cpu { pick(h, 1 << 13, 1 << 15) } else { pick(h, 1 << 14, 1 << 15) }; // the band is an n x n expression: n < 2^16

        // Tridiagonal: `solve_banded` against the line solver on the same diagonals.
        let d = diagonals(h, n, 1, 1);
        let refs: Vec<(isize, &Array<f32>)> = d.iter().map(|(k, v)| (*k, v)).collect();
        let a = Array::from_diagonals(&refs);
        let pad_front = |v: &Array<f32>| v.pad([(1, 0)], PadMode::Zero).reshape([1, n]);
        let pad_back = |v: &Array<f32>| v.pad([(0, 1)], PadMode::Zero).reshape([1, n]);
        for rhs_cols in [1usize, 64] {
            let cols = if cpu { rhs_cols.min(8) } else { rhs_cols };
            let b = upload(h, &values(n * cols, 740, -1.0, 1.0), [cols, n]);
            let direct = || {
                let lo = pad_front(&d[1].1).broadcast_to([cols, n]);
                let up = pad_back(&d[2].1).broadcast_to([cols, n]);
                let diag = d[0].1.reshape([1, n]).broadcast_to([cols, n]);
                solve_tridiagonal(1, &lo, &diag, &up, &b)
            };
            let (x, y) = (host(&a.solve_banded(&b)), host(&direct()));
            assert!(x.iter().zip(&y).all(|(a, b)| a.to_bits() == b.to_bits()), "props/band_solve: solve_banded is the line solver");
            let work = Work::Bytes(4.0 * (2 * n * cols + 3 * n) as f64);
            h.case("props/band_solve_tridiagonal", &format!("n={n} rhs={cols}"), work, &mut || a.solve_banded(&b).eval().unwrap());
            h.case("props/band_solve_tridiagonal_direct", &format!("n={n} rhs={cols}"), work, &mut || direct().eval().unwrap());
        }

        // Bandwidth 4: the block-tridiagonal route (no dense reference at this size; checked against the dense solve at a small n).
        let (small, big) = (96usize, n / 4);
        let ds = diagonals(h, small, 4, 4);
        let rs: Vec<(isize, &Array<f32>)> = ds.iter().map(|(k, v)| (*k, v)).collect();
        let sa = Array::from_diagonals(&rs);
        let sb = upload(h, &values(small, 750, -1.0, 1.0), [1, small]);
        let (got, want) = (host(&sa.solve_banded(&sb)), host(&upload(h, &host(&sa), [small, small]).solve(&sb)));
        for (g, w) in got.iter().zip(&want) {
            assert!((g - w).abs() <= 1e-3 * (1.0 + w.abs()), "props/band_solve_block: {g} vs {w}");
        }
        let db = diagonals(h, big, 4, 4);
        let rb: Vec<(isize, &Array<f32>)> = db.iter().map(|(k, v)| (*k, v)).collect();
        let ba = Array::from_diagonals(&rb);
        let bb = upload(h, &values(big, 760, -1.0, 1.0), [1, big]);
        h.case("props/band_solve_block4", &format!("n={big} band=(4,4)"), Work::Bytes(4.0 * (2 * big + 9 * big) as f64), &mut || ba.solve_banded(&bb).eval().unwrap());

        // Banded products: the fused sum of shifts under exact_finite against the dense product of the evaluated matrix. The
        // harness context is swapped for one with the algebraic numerics (the profile follows `h.ctx`).
        let saved = std::mem::replace(&mut h.ctx, Context::with_options(ContextOptions::from_env().numerics(Numerics::algebraic())).unwrap());
        // Rows of the product, columns of the right-hand side, half-width of the band: the widths and sizes the cost model of the
        // route (`dense::route::banded_shifts_win`) is calibrated on.
        let shapes: &[(usize, usize, usize)] = if cpu { &[(1024, 64, 1), (1024, 64, 4), (1024, 64, 16), (512, 64, 4)] } else { &[(4096, 64, 1), (4096, 64, 4), (4096, 64, 16), (1024, 64, 4), (4096, 512, 4)] };
        for &(m, k, half) in shapes {
            let diags = diagonals(h, m, half, half);
            let r: Vec<(isize, &Array<f32>)> = diags.iter().map(|(o, v)| (*o, v)).collect();
            let band = Array::from_diagonals(&r);
            let x = upload(h, &values(m * k, 770, -1.0, 1.0), [k, m]);
            let dense_a = upload(h, &host(&band), [m, m]);
            let (got, want) = (host(&band.matmul(&x)), host(&dense_a.matmul_f32(&x)));
            for (g, w) in got.iter().zip(&want) {
                assert!((g - w).abs() <= 1e-3 * (1.0 + w.abs()), "props/band_matmul: {g} vs {w}");
            }
            let width = 2 * half + 1;
            let flops = 2.0 * (m * k * width) as f64;
            h.case(&format!("props/band_matmul_w{width}"), &format!("[{m},{m}] x [{k},{m}]"), Work::Flops(flops), &mut || band.matmul(&x).eval().unwrap());
            h.case(&format!("props/band_matmul_w{width}_dense"), &format!("[{m},{m}] x [{k},{m}]"), Work::Flops(flops), &mut || dense_a.matmul(&x).eval().unwrap());
        }
        h.ctx = saved;

        // Gram products: `xᵀ x` through the half-work strategy against the full product of an equal copy (so not recognised as a
        // Gram product), on the `f32` kernels (strict numerics) and with the default numerics (TF32 where the device has tensor cores).
        for (tag, numerics) in [("", Numerics::strict()), ("_tf32", Numerics::default())] {
            let saved = std::mem::replace(&mut h.ctx, Context::with_options(ContextOptions::from_env().numerics(numerics)).unwrap());
            let shapes: &[(usize, usize)] = if cpu { &[(512, 1024), (1024, 512)] } else { &[(1024, 4096), (4096, 1024)] };
            for &(cols, rows) in shapes {
                let data = values(rows * cols, 790, -1.0, 1.0);
                let x = upload(h, &data, [cols, rows]);
                let copy = upload(h, &data, [cols, rows]);
                let (gram, full) = (host(&x.matmul_tn(&x)), host(&x.matmul_tn(&copy)));
                for r in 0..cols {
                    for c in 0..cols {
                        assert_eq!(gram[r * cols + c].to_bits(), gram[c * cols + r].to_bits(), "props/syrk{tag}: bit-symmetric");
                        if c <= r {
                            // The full product of a device may split K (the half-work kernel does not): the same sums in another order.
                            let (g, f) = (gram[r * cols + c], full[r * cols + c]);
                            assert!(if cpu { g.to_bits() == f.to_bits() } else { (g - f).abs() <= 2e-3 * (1.0 + f.abs()) }, "props/syrk{tag}: the full product's lower triangle: {g} vs {f}");
                        }
                    }
                }
                let work = Work::Flops(2.0 * (cols * cols * rows) as f64);
                h.case(&format!("props/syrk{tag}"), &format!("x [{cols},{rows}]"), work, &mut || x.matmul_tn(&x).eval().unwrap());
                h.case(&format!("props/syrk{tag}_full"), &format!("x [{cols},{rows}]"), work, &mut || x.matmul_tn(&copy).eval().unwrap());
            }
            h.ctx = saved;
        }

        // `lower(a) · b` under `exact_finite` (`TriangularK`: each tile row stops where the zeros of the triangle begin) against the
        // product of the evaluated matrix (a copy carries no property, so it runs the whole range).
        for (tag, base) in [("", Numerics::strict()), ("_tf32", Numerics::default())] {
            let mut numerics = base;
            numerics.exact_finite = true;
            let saved = std::mem::replace(&mut h.ctx, Context::with_options(ContextOptions::from_env().numerics(numerics)).unwrap());
            let n: usize = if cpu { 1024 } else { pick(h, 2048, 4096) };
            let a = upload(h, &values(n * n, 800, -1.0, 1.0), [n, n]);
            let b = upload(h, &values(n * n, 810, -1.0, 1.0), [n, n]);
            let dense = a.upper(false).cache();
            let (cut, full) = (host(&a.upper(false).matmul(&b)), host(&dense.matmul(&b)));
            assert!(cut.iter().zip(&full).all(|(c, f)| c.to_bits() == f.to_bits()), "props/trimm{tag}: the cut range keeps the bits");
            let work = Work::Flops(2.0 * (n * n * n) as f64);
            h.case(&format!("props/trimm{tag}"), &format!("lower(a) [{n},{n}] x b"), work, &mut || a.upper(false).matmul(&b).eval().unwrap());
            h.case(&format!("props/trimm{tag}_full"), &format!("lower(a) [{n},{n}] x b"), work, &mut || dense.matmul(&b).eval().unwrap());
            h.ctx = saved;
        }
    });
}
