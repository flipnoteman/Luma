//! Matrix products: square sizes from launch-bound to compute-bound, a tall-skinny product, a
//! matrix-vector product (bandwidth-bound) and a product whose operand is a lazy transpose.

use super::{check64, host, pick, upload};
use crate::harness::*;

/// Row-major `M x K` times `K x N` (`bt` false) or times the transpose of the row-major `N x K`
/// matrix `b` (`bt` true), at the listed `(m, n)` entries only, in `f64`.
fn product_at(a: &[f32], b: &[f32], k: usize, n: usize, bt: bool, at: &[(usize, usize)]) -> Vec<f64> {
    at.iter()
        .map(|&(m, j)| {
            (0..k)
                .map(|l| {
                    let rhs = if bt { b[j * k + l] } else { b[l * n + j] };
                    f64::from(a[m * k + l]) * f64::from(rhs)
                })
                .sum()
        })
        .collect()
}

/// Verifies the `[n_cols, m_rows]` product `c` at every entry (small) or at a sample of them.
#[allow(clippy::too_many_arguments)]
fn verify(name: &str, c: &[f32], a: &[f32], b: &[f32], m: usize, k: usize, n: usize, bt: bool) {
    let entries: Vec<(usize, usize)> = if m * n <= 256 * 256 {
        (0..m).flat_map(|r| (0..n).map(move |j| (r, j))).collect()
    } else {
        let rows = ints(48, 31, m as u32);
        let cols = ints(48, 32, n as u32);
        let mut e: Vec<(usize, usize)> = rows.iter().zip(&cols).map(|(&r, &j)| (r as usize, j as usize)).collect();
        e.extend([(0, 0), (m - 1, n - 1), (0, n - 1), (m - 1, 0)]);
        e
    };
    let want = product_at(a, b, k, n, bt, &entries);
    let got: Vec<f32> = entries.iter().map(|&(r, j)| c[r * n + j]).collect();
    // Operands are uniform in [-1, 1): the products have standard deviation sqrt(k / 9).
    check64(name, &got, &want, 1e-3, 2e-4 * k as f64 + 1e-3);
}

pub fn run(h: &mut Harness) {
    h.guarded("matmul", |h| {
        let cpu = h.ctx.is_cpu();
        let sizes: &[usize] = match (cpu, h.quick) {
            (false, false) => &[128, 256, 512, 1024, 2048, 4096, 8192],
            (false, true) => &[128, 512, 2048],
            (true, false) => &[128, 256, 512, 1024],
            (true, true) => &[128, 512],
        };
        for &n in sizes {
            let (ad, bd) = (values(n * n, 41, -1.0, 1.0), values(n * n, 42, -1.0, 1.0));
            let (a, b) = (upload(h, &ad, [n, n]), upload(h, &bd, [n, n]));
            verify("matmul/square", &host(&a.matmul(&b)), &ad, &bd, n, n, n, false);
            h.case("matmul/square", &format!("{n}"), Work::Flops(2.0 * (n as f64).powi(3)), &mut || a.matmul(&b).eval().unwrap());
        }

        // A tall-skinny projection: memory-bound with a small K.
        {
            let m: usize = if cpu { pick(h, 1 << 14, 1 << 16) } else { pick(h, 1 << 16, 1 << 18) };
            let (k, n) = (64usize, 64usize);
            let (ad, bd) = (values(m * k, 43, -1.0, 1.0), values(k * n, 44, -1.0, 1.0));
            let (a, b) = (upload(h, &ad, [k, m]), upload(h, &bd, [n, k]));
            verify("matmul/skinny", &host(&a.matmul(&b)), &ad, &bd, m, k, n, false);
            h.case("matmul/skinny", &format!("{m}x{k}x{n}"), Work::Flops(2.0 * (m * k * n) as f64), &mut || a.matmul(&b).eval().unwrap());
        }

        // Matrix-vector: the vector is the 1-column right-hand side.
        let sizes: &[usize] = if cpu || h.quick { &[2048, 4096] } else { &[2048, 8192] };
        for &n in sizes {
            let (ad, xd) = (values(n * n, 45, -1.0, 1.0), values(n, 46, -1.0, 1.0));
            let (a, x) = (upload(h, &ad, [n, n]), upload(h, &xd, [1, n]));
            let got = host(&a.matmul(&x));
            let want: Vec<f64> = (0..n).map(|r| (0..n).map(|l| f64::from(ad[r * n + l]) * f64::from(xd[l])).sum()).collect();
            check64("matmul/matvec", &got, &want, 1e-3, 2e-4 * n as f64);
            h.case("matmul/matvec", &format!("{n}x{n}"), Work::Bytes(4.0 * (n * n + 2 * n) as f64), &mut || a.matmul(&x).eval().unwrap());
        }

        // A * B^T with the transpose a lazy operand read inside the product kernel.
        let sizes: &[usize] = if cpu { &[512] } else { &[1024, pick(h, 1024, 2048)] };
        let mut sizes = sizes.to_vec();
        sizes.dedup();
        for n in sizes {
            let (ad, bd) = (values(n * n, 47, -1.0, 1.0), values(n * n, 48, -1.0, 1.0));
            let (a, b) = (upload(h, &ad, [n, n]), upload(h, &bd, [n, n]));
            verify("matmul/transposed_rhs", &host(&a.matmul(&b.transpose())), &ad, &bd, n, n, n, true);
            h.case("matmul/transposed_rhs", &format!("{n}"), Work::Flops(2.0 * (n as f64).powi(3)), &mut || a.matmul(&b.transpose()).eval().unwrap());
        }
    });
}
