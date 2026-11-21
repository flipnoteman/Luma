//! Classic ML loops kept on the device: logistic regression by gradient descent (a chained loop
//! and `Array::iterate`), k-means iterations (distances by matmul, argmin, group means by a
//! one-hot matmul or `group_sum`), PCA (covariance by matmul of a lazily centered matrix, power
//! iteration).

use super::common::*;
use crate::harness::*;
use luma::*;

// ---------------------------------------------------------------------------------------------
// Logistic regression
// ---------------------------------------------------------------------------------------------

struct LogregData {
    xs: Vec<f32>,
    ys: Vec<f32>,
}

fn logreg_data(n: usize, d: usize) -> LogregData {
    let xs = values(n * d, 71, -1.0, 1.0);
    let truth = values(d, 72, -1.0, 1.0);
    let ys = xs.chunks(d).map(|r| f32::from(u8::from(r.iter().zip(&truth).map(|(a, b)| a * b).sum::<f32>() > 0.0))).collect();
    LogregData { xs, ys }
}

/// `steps` of full-batch gradient descent from zero weights on the host.
fn logreg_host(data: &LogregData, d: usize, steps: usize, lr: f64) -> Vec<f64> {
    let n = data.ys.len();
    let mut w = vec![0.0f64; d];
    for _ in 0..steps {
        let mut g = vec![0.0f64; d];
        for (row, &y) in data.xs.chunks(d).zip(&data.ys) {
            let z: f64 = row.iter().zip(&w).map(|(&x, w)| f64::from(x) * w).sum();
            let r = 1.0 / (1.0 + (-z).exp()) - f64::from(y);
            for (g, &x) in g.iter_mut().zip(row) {
                *g += r * f64::from(x);
            }
        }
        for (w, g) in w.iter_mut().zip(&g) {
            *w -= lr * g / n as f64;
        }
    }
    w
}

fn logreg(h: &mut Harness, n: usize, d: usize, iterate: bool) {
    let n = size(h, n / 16, n, n / 4);
    let steps = 20;
    let flops = 4.0 * (n * d) as f64 * steps as f64;
    if !fits(h, flops) {
        return;
    }
    let case = if iterate { "classic/logreg_iterate" } else { "classic/logreg" };
    let size = format!("N{n} D{d} x{steps}");
    h.guarded(&format!("{case} {size}"), |h| {
        let data = logreg_data(n, d);
        let lr = 2.0f32;
        let (x, y) = (arr(h, &data.xs, [d, n]), arr(h, &data.ys, [1, n]));
        let xt = x.transpose();
        let w0 = Array::<f32>::zeros_on(&h.ctx, [1, d]).unwrap();
        // One step: logits (matmul), sigmoid, residual, gradient (matmul with the transpose).
        let step = |w: &Array<f32>| {
            let z = matmul(&x, w);
            let p = 1.0 / (&(-&z).exp() + 1.0);
            let g = matmul(&xt, &(&p - &y));
            w - &(&g * (lr / n as f32))
        };
        let run = |steps: usize| {
            if iterate {
                w0.iterate(steps, step)
            } else {
                (0..steps).fold(w0.clone(), |w, _| step(&w))
            }
        };
        let want = logreg_host(&data, d, 5, f64::from(lr));
        let got = run(5).to_vec().unwrap();
        check64(case, &got, &want, 2e-3, 2e-4);
        h.case(case, &size, Work::Flops(flops), &mut || run(steps).eval().unwrap());
    });
}

// ---------------------------------------------------------------------------------------------
// Softmax regression
// ---------------------------------------------------------------------------------------------

/// Features (`n x d`, row-major) and class labels (`c` classes) of a softmax regression.
fn softmax_data(n: usize, d: usize, c: usize) -> (Vec<f32>, Vec<u32>) {
    (values(n * d, 81, -1.0, 1.0), ints(n, 82, c as u32))
}

/// `steps` of full-batch gradient descent from zero weights on the host: the `c x d` weights
/// (row-major `c x d`; the device weights are `[c, d]`, the transpose).
fn softmax_host(xs: &[f32], ys: &[u32], d: usize, c: usize, steps: usize, lr: f64) -> Vec<f64> {
    let n = ys.len();
    let mut w = vec![0.0f64; c * d];
    for _ in 0..steps {
        let mut g = vec![0.0f64; c * d];
        for (row, &y) in xs.chunks(d).zip(ys) {
            let z: Vec<f64> = (0..c).map(|k| row.iter().zip(&w[k * d..]).map(|(&x, w)| f64::from(x) * w).sum()).collect();
            let m = z.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let e: Vec<f64> = z.iter().map(|v| (v - m).exp()).collect();
            let total: f64 = e.iter().sum();
            for k in 0..c {
                let r = e[k] / total - f64::from(u8::from(k as u32 == y));
                for (g, &x) in g[k * d..][..d].iter_mut().zip(row) {
                    *g += r * f64::from(x);
                }
            }
        }
        for (w, g) in w.iter_mut().zip(&g) {
            *w -= lr * g / n as f64;
        }
    }
    w
}

/// The device loop: logits, a softmax along the classes, the one-hot residual and the gradient
/// (`Xᵀ R`), `steps` times; `x` is `[d, n]`, `onehot` `[c, n]`, the weights `[c, d]`.
fn softmax_fit(ctx: &Context, x: &Array<f32>, onehot: &Array<f32>, n: usize, steps: usize, lr: f32) -> Array<f32> {
    let [d, c] = [x.shape().dims()[0], onehot.shape().dims()[0]];
    let xt = x.transpose();
    let mut w = Array::<f32>::zeros_on(ctx, [c, d]).unwrap();
    for _ in 0..steps {
        let z = matmul(x, &w);
        let e = (&z - &z.max_axis(0)).exp();
        let p = &e / &e.sum_axis(0);
        w = &w - &(&matmul(&xt, &(&p - onehot)) * (lr / n as f32));
        w.eval().unwrap();
    }
    w
}

fn softmax(h: &mut Harness, n: usize, d: usize, c: usize, steps: usize) {
    let n = size(h, n / 16, n, n / 4);
    let flops = 4.0 * (n * d * c) as f64 * steps as f64;
    if !fits(h, flops) {
        return;
    }
    let case = "classic/softmax";
    let size = format!("N{n} D{d} C{c} x{steps}");
    h.guarded(&format!("{case} {size}"), |h| {
        let lr = 1.0f32;
        // Two steps on a small problem against the host.
        let (cn, cs) = (2048, 2);
        let (cx, cy) = softmax_data(cn, d, c);
        let x = arr(h, &cx, [d, cn]);
        let onehot = |ys: &[u32], n: usize| select(&range(h, [c, n], 0).eq(&arr_u32(h, ys, [1, n])), 1.0f32, 0.0f32);
        let got = softmax_fit(&h.ctx, &x, &onehot(&cy, cn), cn, cs, lr).to_vec().unwrap();
        let want = softmax_host(&cx, &cy, d, c, cs, f64::from(lr));
        let want: Vec<f64> = (0..c * d).map(|i| want[(i % c) * d + i / c]).collect();
        check64(case, &got, &want, 2e-3, 2e-4);
        let (xs, ys) = softmax_data(n, d, c);
        let (x, y) = (arr(h, &xs, [d, n]), onehot(&ys, n));
        let ctx = h.ctx.clone();
        h.case(case, &size, Work::Flops(flops), &mut || softmax_fit(&ctx, &x, &y, n, steps, lr).eval().unwrap());
    });
}

// ---------------------------------------------------------------------------------------------
// k-means
// ---------------------------------------------------------------------------------------------

struct KmeansData {
    points: Vec<f32>,
    centroids: Vec<f32>,
}

/// `k` well-separated blobs of `n` points in `d` dimensions, and centroids near the blob centers
/// (so the assignment does not depend on rounding).
fn kmeans_data(n: usize, d: usize, k: usize) -> KmeansData {
    let centers = values(k * d, 73, -5.0, 5.0);
    let noise = values(n * d, 74, -0.3, 0.3);
    let which = ints(n, 75, k as u32);
    let points = (0..n * d).map(|i| centers[which[i / d] as usize * d + i % d] + noise[i]).collect();
    let jitter = values(k * d, 76, -0.5, 0.5);
    KmeansData { points, centroids: centers.iter().zip(&jitter).map(|(c, j)| c + j).collect() }
}

/// One Lloyd iteration on the host: the new `k x d` centroids.
fn kmeans_host(data: &KmeansData, d: usize, k: usize) -> Vec<f64> {
    let mut sums = vec![0.0f64; k * d];
    let mut counts = vec![0usize; k];
    for p in data.points.chunks(d) {
        let dist = |c: usize| -> f64 { p.iter().zip(&data.centroids[c * d..][..d]).map(|(&a, &b)| (f64::from(a) - f64::from(b)).powi(2)).sum() };
        let best = (0..k).min_by(|&a, &b| dist(a).total_cmp(&dist(b))).unwrap();
        counts[best] += 1;
        for (s, &v) in sums[best * d..][..d].iter_mut().zip(p) {
            *s += f64::from(v);
        }
    }
    (0..k * d).map(|i| if counts[i / d] == 0 { f64::from(data.centroids[i]) } else { sums[i] / counts[i / d] as f64 }).collect()
}

/// `iters` Lloyd iterations on the device. Centroids are `[k, d]` so that they are the
/// right-hand operand of the distance product directly.
fn kmeans(h: &mut Harness, n: usize, d: usize, k: usize, group: bool) {
    let n = size(h, n / 16, n, n / 4);
    let iters = 5;
    let flops = (2.0 * (n * d * k) as f64 + 2.0 * (n * d) as f64) * iters as f64;
    if !fits(h, flops) {
        return;
    }
    let case = if group { "classic/kmeans_group_sum" } else { "classic/kmeans_matmul" };
    let size = format!("N{n} D{d} K{k} x{iters}");
    h.guarded(&format!("{case} {size}"), |h| {
        let ctx = h.ctx.clone();
        let data = kmeans_data(n, d, k);
        let x = arr(h, &data.points, [d, n]);
        let xt = x.transpose();
        // The centroid array as `[k, d]`: transpose the row-major `k x d` host matrix into `d x k`.
        let c0 = arr(h, &transpose(&data.centroids, k, d), [k, d]);
        let classes = Array::<u32>::range_on(&ctx, [k, n], 0).unwrap();
        let dim_offsets = Array::<u32>::range_on(&ctx, [d, n], 0).unwrap() * k as u32;
        let step = |c: &Array<f32>| -> Array<f32> {
            // |x - c|^2 = |x|^2 - 2 x.c + |c|^2; the first term does not change the argmin.
            let cross = matmul(&x, c);
            let cnorm = (c * c).sum_axis(1);
            let labels = (&cnorm - &(&cross * 2.0)).argmin_axis(0);
            if group {
                // Per-dimension group sums: group `dim * k + label` of every coordinate.
                let codes = &labels + &dim_offsets;
                let sums = x.group_sum(&codes, d * k).reshape([k, d]);
                let counts = bincount(&labels, k).to_f32().maximum(1.0);
                &sums / &counts
            } else {
                let onehot = select(&classes.eq(&labels), 1.0f32, 0.0f32);
                let sums = matmul(&xt, &onehot);
                &sums / &onehot.sum_axis(1).maximum(1.0)
            }
        };
        let run = |iters: usize| (0..iters).fold(c0.clone(), |c, _| step(&c));
        // One iteration against the host, centroids compared in the `[k, d]` -> row-major layout.
        let want = kmeans_host(&data, d, k);
        let got = transpose(&run(1).to_vec().unwrap(), d, k);
        check64(case, &got, &want, 1e-3, 1e-3);
        h.case(case, &size, Work::Flops(flops), &mut || run(iters).eval().unwrap());
    });
}

// ---------------------------------------------------------------------------------------------
// PCA
// ---------------------------------------------------------------------------------------------

/// Samples with a shared factor (a dominant direction) and per-feature scales.
fn pca_data(n: usize, d: usize) -> Vec<f32> {
    let noise = values(n * d, 77, -1.0, 1.0);
    let factor = values(n, 78, -3.0, 3.0);
    (0..n * d).map(|i| noise[i] * (1.0 + 3.0 / (1.0 + (i % d) as f32)) + 0.7 * factor[i / d] + 2.0).collect()
}

fn pca(h: &mut Harness, n: usize, d: usize) {
    let n = size(h, n / 8, n, n / 4);
    let power = 30;
    let flops = 2.0 * (n * d * d) as f64;
    if !fits(h, flops) {
        return;
    }
    let case = "classic/pca";
    let size = format!("N{n} D{d}");
    h.guarded(&format!("{case} {size}"), |h| {
        let xs = pca_data(n, d);
        let x = arr(h, &xs, [d, n]);
        let v0 = vec![1.0f32; d];
        let run = || {
            // Covariance of the centered data: the centering is a lazy operand of the product.
            let xc = &x - &x.mean_axis(1);
            let cov = &matmul(&xc.transpose(), &xc) / n as f32;
            let mut v = Array::<f32>::full_on(&x.context().clone(), [1, d], 1.0).unwrap();
            for _ in 0..power {
                let w = matmul(&cov, &v);
                v = &w / &w.norm();
            }
            let lambda = matmul(&cov, &v);
            (cov, v, lambda)
        };
        // Host covariance, centered in f64.
        let mean: Vec<f64> = (0..d).map(|c| xs.chunks(d).map(|r| f64::from(r[c])).sum::<f64>() / n as f64).collect();
        let centered: Vec<f32> = xs.chunks(d).flat_map(|r| r.iter().zip(&mean).map(|(&v, m)| (f64::from(v) - m) as f32)).collect();
        let cov: Vec<f32> = gemm(&transpose(&centered, n, d), &centered, d, n, d).iter().map(|v| v / n as f32).collect();
        let (got_cov, got_v, got_l) = {
            let (c, v, l) = run();
            (c.to_vec().unwrap(), v.to_vec().unwrap(), l.to_vec().unwrap())
        };
        let scale = max_abs(&cov);
        check("classic/pca covariance", &got_cov, &cov, 1e-2, 2e-3 * scale);
        // Power iteration on the host from the same start, and the Rayleigh quotient.
        let mut v = v0.iter().map(|&x| f64::from(x)).collect::<Vec<f64>>();
        for _ in 0..power {
            let w: Vec<f64> = (0..d).map(|i| (0..d).map(|j| f64::from(cov[i * d + j]) * v[j]).sum()).collect();
            let norm = w.iter().map(|x| x * x).sum::<f64>().sqrt();
            v = w.iter().map(|x| x / norm).collect();
        }
        let lambda: f64 = (0..d).map(|i| v[i] * (0..d).map(|j| f64::from(cov[i * d + j]) * v[j]).sum::<f64>()).sum();
        let got_lambda: f64 = got_v.iter().zip(&got_l).map(|(&a, &b)| f64::from(a) * f64::from(b)).sum();
        check_scalar("classic/pca eigenvalue", got_lambda, lambda, 2e-3);
        h.case(case, &size, Work::Flops(flops), &mut || {
            let (c, v, l) = run();
            eval_all(&[&c, &v, &l]).unwrap();
        });
    });
}

pub fn run(h: &mut Harness) {
    logreg(h, 262_144, 32, false);
    logreg(h, 1 << 20, 16, false);
    logreg(h, 262_144, 32, true);
    softmax(h, 200_000, 64, 10, 20);
    softmax(h, 1_000_000, 128, 20, 10);
    kmeans(h, 262_144, 16, 64, false);
    kmeans(h, 262_144, 16, 64, true);
    kmeans(h, 1 << 20, 8, 16, false);
    pca(h, 1 << 20, 32);
    pca(h, 262_144, 128);
    pca(h, 32768, 512);
}
