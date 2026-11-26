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
fn verify(name: &str, c: &[f32], a: &[f32], b: &[f32], m: usize, k: usize, n: usize, bt: bool, tf32: bool) {
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
    if tf32 {
        // Tensor cores round the operands to TF32: up to 2^-10 of the sum of |a b|, 0.25 k here.
        check64(name, &got, &want, 0.0, 3e-4 * k as f64 + 1e-3);
    } else {
        check64(name, &got, &want, 1e-3, 2e-4 * k as f64 + 1e-3);
    }
}

/// `f32` and TF32 products of the same shapes (`m x k` times `k x n`, `layout` 0 = plain, 1 = the
/// right operand a lazy transpose, 2 = the left one), timed alternately ABBA; prints the TF32
/// error against the `f64` product. Only on devices where `matmul_tf32` runs on tensor cores.
fn tf32_pair(h: &mut Harness, m: usize, k: usize, n: usize, layout: u8) {
    let name = ["nn", "nt", "tn"][layout as usize];
    let (ad, bd) = (values(m * k, 61, -1.0, 1.0), values(k * n, 62, -1.0, 1.0));
    let transpose = |d: &[f32], rows: usize, cols: usize| -> Vec<f32> { (0..rows * cols).map(|i| d[(i % rows) * cols + i / rows]).collect() };
    let (a, b, at, bt) = (upload(h, &ad, [k, m]), upload(h, &bd, [n, k]), upload(h, &transpose(&ad, m, k), [m, k]), upload(h, &transpose(&bd, k, n), [k, n]));
    type Product<'a> = Box<dyn Fn() -> luma::Array<f32> + 'a>;
    let (f32_run, tf32_run): (Product, Product) = match layout {
        0 => (Box::new(|| a.matmul_f32(&b)), Box::new(|| a.matmul_tf32(&b))),
        1 => (Box::new(|| a.matmul_f32(&bt.transpose())), Box::new(|| a.matmul_tf32(&bt.transpose()))),
        _ => (Box::new(|| at.transpose().matmul_f32(&b)), Box::new(|| at.transpose().matmul_tf32(&b))),
    };
    // Error of both against the f64 product at a sample of entries.
    let rows = ints(48, 71, m as u32);
    let cols = ints(48, 72, n as u32);
    let entries: Vec<(usize, usize)> = rows.iter().zip(&cols).map(|(&r, &j)| (r as usize, j as usize)).collect();
    let want = product_at(&ad, &bd, k, n, false, &entries);
    let err = |c: &[f32]| {
        let diffs: Vec<f64> = entries.iter().zip(&want).map(|(&(r, j), w)| f64::from(c[r * n + j]) - w).collect();
        let rms = (diffs.iter().map(|d| d * d).sum::<f64>() / diffs.len() as f64).sqrt();
        let max = diffs.iter().fold(0.0f64, |a, d| a.max(d.abs()));
        let scale = (want.iter().map(|w| w * w).sum::<f64>() / want.len() as f64).sqrt();
        (rms / scale, max / scale)
    };
    let (f32_err, tf32_err) = (err(&host(&f32_run())), err(&host(&tf32_run())));
    println!("  {name} {m}x{k}x{n}: error / rms(C): f32 rms {:.2e} max {:.2e}, tf32 rms {:.2e} max {:.2e}", f32_err.0, f32_err.1, tf32_err.0, tf32_err.1);
    assert!(tf32_err.1 < 5e-3, "tf32 {name} {m}x{k}x{n}: error {:.2e}", tf32_err.1);
    let flops = 2.0 * (m * k * n) as f64;
    let size = format!("{name} {m}x{k}x{n}");
    for tf32 in [false, true, true, false] {
        let run = if tf32 { &tf32_run } else { &f32_run };
        h.case(if tf32 { "matmul/tf32" } else { "matmul/f32" }, &size, Work::Flops(flops), &mut || run().eval().unwrap());
    }
}

/// One `f32` product (`layout` 0 = plain, 1 = the right operand a lazy transpose, 2 = the left),
/// verified against the host and timed as `matmul/shape`; run it with and without
/// `LUMA_OWN_GEMM=off` to compare kernels.
fn shape_case(h: &mut Harness, m: usize, k: usize, n: usize, layout: u8) {
    let name = ["nn", "nt", "tn"][layout as usize];
    let (ad, bd) = (values(m * k, 81, -1.0, 1.0), values(k * n, 82, -1.0, 1.0));
    let transpose = |d: &[f32], rows: usize, cols: usize| -> Vec<f32> { (0..rows * cols).map(|i| d[(i % rows) * cols + i / rows]).collect() };
    let (a, b, at, bt) = (upload(h, &ad, [k, m]), upload(h, &bd, [n, k]), upload(h, &transpose(&ad, m, k), [m, k]), upload(h, &transpose(&bd, k, n), [k, n]));
    let run = || match layout {
        0 => a.matmul_f32(&b),
        1 => a.matmul_f32(&bt.transpose()),
        _ => at.transpose().matmul_f32(&b),
    };
    verify("matmul/shape", &host(&run()), &ad, &bd, m, k, n, false, false);
    h.case("matmul/shape", &format!("{name} {m}x{k}x{n}"), Work::Flops(2.0 * (m * k * n) as f64), &mut || run().eval().unwrap());
}

/// `(m, k, n)` of the rows for the odd-shape kernels (`LUMA_BENCH_SHAPES=odd`, `LUMA_BENCH_TF32=odd`, also run after the full lists): the
/// odd squares with their aligned neighbours, rectangles of odd extents, and the products that keep the shader kernels (an extent below
/// 32 or a short K), which must not get slower.
const ODD_SHAPES: [(usize, usize, usize); 17] = [
    (1000, 1000, 1000),
    (1001, 1001, 1001),
    (1024, 1024, 1024),
    (3001, 3001, 3001),
    (4096, 4096, 4096),
    (4097, 4097, 4097),
    (1537, 513, 255),
    (4097, 1001, 129),
    (10, 4096, 4096),
    (4096, 4096, 10),
    (4096, 10, 4096),
    (4096, 31, 4096),
    (4096, 33, 4096),
    (2001, 37, 2001),
    (1024, 16, 1024),
    (257, 259, 261),
    (131, 67, 4001),
];

pub fn run(h: &mut Harness) {
    if !h.ctx.is_cpu() && std::env::var("LUMA_BENCH_SHAPES").as_deref() == Ok("odd") {
        h.guarded("matmul", |h| {
            for &(m, k, n) in &ODD_SHAPES {
                shape_case(h, m, k, n, 0);
            }
        });
        return;
    }
    if !h.ctx.is_cpu() && std::env::var("LUMA_BENCH_TF32").as_deref() == Ok("odd") {
        h.guarded("matmul", |h| {
            for &(m, k, n) in &ODD_SHAPES {
                tf32_pair(h, m, k, n, 0);
            }
        });
        return;
    }
    if !h.ctx.is_cpu() && std::env::var_os("LUMA_BENCH_SHAPES").is_some() {
        h.guarded("matmul", |h| {
            let small = std::env::var("LUMA_BENCH_SHAPES").as_deref() == Ok("small");
            let squares: &[usize] = if small { &[1024, 4096] } else { &[128, 256, 512, 1024, 2048, 4096, 8192] };
            for &n in squares {
                shape_case(h, n, n, n, 0);
            }
            let shapes: &[(usize, usize, usize)] = if small {
                &[(256, 1024, 1024), (2048, 512, 128), (2048, 2048, 64), (4096, 1024, 1024)]
            } else {
                &[(256, 1024, 1024), (4096, 1024, 1024), (2048, 512, 128), (256, 512, 8192), (32, 4096, 4096), (256, 16384, 64), (2048, 2048, 64), (2048, 64, 2048), (4096, 128, 4096), (4096, 4096, 128), (16384, 64, 64), (64, 1 << 20, 64), (128, 1 << 18, 128), (96, 96, 96), (192, 320, 160), (384, 384, 384), (768, 768, 768), (1536, 1536, 1536)]
            };
            for &(m, k, n) in shapes {
                shape_case(h, m, k, n, 0);
            }
            if !small {
                for &(m, k, n) in &ODD_SHAPES {
                    shape_case(h, m, k, n, 0);
                }
            }
            for layout in [1, 2] {
                shape_case(h, 1024, 1024, 1024, layout);
                shape_case(h, 2048, 512, 128, layout);
                if !small {
                    shape_case(h, 4096, 4096, 4096, layout);
                }
            }
        });
        return;
    }
    if !h.ctx.is_cpu() && std::env::var_os("LUMA_BENCH_TF32").is_some() {
        h.guarded("matmul", |h| {
            if std::env::var("LUMA_BENCH_TF32").as_deref() == Ok("small") {
                for (m, k, n, layout) in [(8192, 8192, 8192, 0), (4096, 4096, 4096, 0), (1024, 1024, 1024, 0), (4096, 1024, 1024, 0), (2048, 512, 128, 0), (256, 1024, 1024, 0), (2048, 2048, 64, 0), (4096, 4096, 4096, 1), (4096, 4096, 4096, 2), (64, 1 << 20, 64, 0)] {
                    tf32_pair(h, m, k, n, layout);
                }
                return;
            }
            for n in [512, 1024, 2048, 4096, 8192] {
                tf32_pair(h, n, n, n, 0);
            }
            for (m, k, n) in [(256, 1024, 1024), (4096, 1024, 1024), (4096, 4096, 4096), (256, 512, 8192), (32, 4096, 4096), (256, 16384, 64), (2048, 512, 128), (16384, 64, 64), (64, 1 << 20, 64), (128, 1 << 18, 128)] {
                tf32_pair(h, m, k, n, 0);
            }
            // Attention: Q K^T (K = head width) and P V (K = tokens).
            for (m, k, n) in [(2048, 64, 2048), (4096, 128, 4096), (2048, 2048, 64), (4096, 4096, 128)] {
                tf32_pair(h, m, k, n, 0);
            }
            for &(m, k, n) in &ODD_SHAPES {
                tf32_pair(h, m, k, n, 0);
            }
            for layout in [1, 2] {
                tf32_pair(h, 1024, 1024, 1024, layout);
                tf32_pair(h, 2048, 512, 128, layout);
                tf32_pair(h, 4096, 4096, 4096, layout);
            }
        });
        return;
    }
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
            verify("matmul/square", &host(&a.matmul(&b)), &ad, &bd, n, n, n, false, h.ctx.numerics().tf32_matmul());
            h.case("matmul/square", &format!("{n}"), Work::Flops(2.0 * (n as f64).powi(3)), &mut || a.matmul(&b).eval().unwrap());
        }

        // A tall-skinny projection: memory-bound with a small K.
        {
            let m: usize = if cpu { pick(h, 1 << 14, 1 << 16) } else { pick(h, 1 << 16, 1 << 18) };
            let (k, n) = (64usize, 64usize);
            let (ad, bd) = (values(m * k, 43, -1.0, 1.0), values(k * n, 44, -1.0, 1.0));
            let (a, b) = (upload(h, &ad, [k, m]), upload(h, &bd, [n, k]));
            verify("matmul/skinny", &host(&a.matmul(&b)), &ad, &bd, m, k, n, false, h.ctx.numerics().tf32_matmul());
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
            verify("matmul/transposed_rhs", &host(&a.matmul(&b.transpose())), &ad, &bd, n, n, n, true, h.ctx.numerics().tf32_matmul());
            h.case("matmul/transposed_rhs", &format!("{n}"), Work::Flops(2.0 * (n as f64).powi(3)), &mut || a.matmul(&b.transpose()).eval().unwrap());
        }
    });
}
