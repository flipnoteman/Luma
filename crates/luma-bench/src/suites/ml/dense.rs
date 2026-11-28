//! Dense layers: `act(x W + b)` over batch sizes and widths (square, tall-skinny, wide, long K),
//! and the three operand layouts of a matrix product (plain, transposed weights, transposed input).

use super::common::*;
use crate::harness::*;
use luma::*;

#[derive(Clone, Copy)]
enum Act {
    None,
    Relu,
    Gelu,
    Tanh,
}

impl Act {
    fn name(self) -> &'static str {
        match self {
            Act::None => "linear",
            Act::Relu => "relu",
            Act::Gelu => "gelu",
            Act::Tanh => "tanh",
        }
    }

    fn lazy(self, z: Array<f32>) -> Array<f32> {
        match self {
            Act::None => z,
            Act::Relu => z.maximum(0.0),
            Act::Tanh => z.tanh(),
            Act::Gelu => {
                let u = &z * (&z * &z * 0.044715 + 1.0) * 0.797_884_6;
                (u.tanh() + 1.0) * &z * 0.5
            }
        }
    }

    fn host(self, x: f64) -> f64 {
        match self {
            Act::None => x,
            Act::Relu => x.max(0.0),
            Act::Tanh => x.tanh(),
            Act::Gelu => gelu(x),
        }
    }
}

/// `act(x (m x k) * w (k x n) + b)`.
fn layer(h: &mut Harness, act: Act, m: usize, k: usize, n: usize) {
    let flops = 2.0 * m as f64 * k as f64 * n as f64;
    if !fits(h, flops) {
        return;
    }
    let case = format!("dense/{}", act.name());
    let size = format!("{m}x{k}x{n}");
    h.guarded(&format!("{case} {size}"), |h| {
        let xs = values(m * k, 1, -1.0, 1.0);
        let s = 1.0 / (k as f32).sqrt();
        let ws = values(k * n, 2, -s, s);
        let bs = values(n, 3, -0.5, 0.5);
        let (x, w, b) = (arr(h, &xs, [k, m]), arr(h, &ws, [n, k]), arr(h, &bs, n));
        let run = || act.lazy(&matmul(&x, &w) + &b);
        let rows = sample(m, 6);
        let want: Vec<f64> = gemm_rows(&xs, &ws, k, n, &rows).chunks(n).flat_map(|r| r.iter().zip(&bs).map(|(v, b)| act.host(v + f64::from(*b)))).collect();
        check64(&case, &gather_rows(&run().to_vec().unwrap(), n, &rows), &want, 2e-3, 2e-3);
        let bytes = 4.0 * (m * k + k * n + m * n) as f64;
        // Large products are compute-bound; the narrow ones are about moving the operands.
        let work = if flops / bytes > 40.0 { Work::Flops(flops) } else { Work::Bytes(bytes) };
        h.case(&case, &size, work, &mut || run().eval().unwrap());
    });
}

#[derive(Clone, Copy)]
enum Layout {
    /// `x W`.
    Nn,
    /// `x W^T`, weights stored `n x k`.
    Nt,
    /// `x^T W` with `x` stored `k x m`, as the weight gradient of a layer.
    Tn,
}

fn layout(h: &mut Harness, lay: Layout, m: usize, k: usize, n: usize) {
    let flops = 2.0 * m as f64 * k as f64 * n as f64;
    if !fits(h, flops) {
        return;
    }
    let case = format!("dense/matmul_{}", ["nn", "nt", "tn"][lay as usize]);
    let size = format!("{m}x{k}x{n}");
    h.guarded(&case, |h| {
        let xs = values(m * k, 11, -1.0, 1.0);
        let s = 1.0 / (k as f32).sqrt();
        let ws = values(k * n, 12, -s, s);
        let rows = sample(m, 6);
        let want = gemm_rows(&xs, &ws, k, n, &rows);
        let (x, w) = (arr(h, &xs, [k, m]), arr(h, &ws, [n, k]));
        // The operand that is stored transposed, and the view that undoes it.
        let wt = arr(h, &transpose(&ws, k, n), [k, n]);
        let xt = arr(h, &transpose(&xs, m, k), [m, k]);
        let run = || match lay {
            Layout::Nn => matmul(&x, &w),
            Layout::Nt => matmul(&x, &wt.transpose()),
            Layout::Tn => matmul(&xt.transpose(), &w),
        };
        check64(&case, &gather_rows(&run().to_vec().unwrap(), n, &rows), &want, 2e-3, 2e-3);
        h.case(&case, &size, Work::Flops(flops), &mut || run().eval().unwrap());
    });
}

/// The Gram matrix `x^T x` of a tall `x` (`rules/gram`, rule 11 of W3-R4R5: tagged symmetric, the half-work kernel is W3-R6's).
fn gram(h: &mut Harness, m: usize, n: usize) {
    let flops = 2.0 * m as f64 * n as f64 * n as f64;
    if !fits(h, flops) {
        return;
    }
    h.guarded("rules/gram", |h| {
        let xs = values(m * n, 21, -1.0, 1.0);
        let x = arr(h, &xs, [n, m]);
        h.case("rules/gram", &format!("{m}x{n}"), Work::Flops(flops), &mut || matmul_tn(&x, &x).eval().unwrap());
    });
}

/// Three products of one `x` (`x W1`, `x W2`, `x W3`, widths `n`, `n`, `n / 2`), each with a bias: what the partner fusion launches together.
fn shared_operand(h: &mut Harness, m: usize, k: usize, n: usize) {
    let flops = 2.0 * m as f64 * k as f64 * (2.5 * n as f64);
    if !fits(h, flops) {
        return;
    }
    let case = "fuse/shared_operand";
    let size = format!("{m}x{k}x{n}");
    h.guarded(case, |h| {
        let xs = values(m * k, 21, -1.0, 1.0);
        let s = 1.0 / (k as f32).sqrt();
        let widths = [n, n, n / 2];
        let ws: Vec<Vec<f32>> = widths.iter().enumerate().map(|(i, &w)| values(k * w, 22 + i as u64, -s, s)).collect();
        let bs: Vec<Vec<f32>> = widths.iter().enumerate().map(|(i, &w)| values(w, 25 + i as u64, -0.5, 0.5)).collect();
        let x = arr(h, &xs, [k, m]);
        let w: Vec<Array<f32>> = ws.iter().zip(widths).map(|(v, width)| arr(h, v, [width, k])).collect();
        let b: Vec<Array<f32>> = bs.iter().zip(widths).map(|(v, width)| arr(h, v, width)).collect();
        let run = || -> Vec<Array<f32>> { (0..3).map(|i| (&matmul(&x, &w[i]) + &b[i]).maximum(0.0)).collect() };
        let rows = sample(m, 6);
        let want: Vec<f64> = gemm_rows(&xs, &ws[0], k, n, &rows).chunks(n).flat_map(|r| r.iter().zip(&bs[0]).map(|(v, b)| (v + f64::from(*b)).max(0.0))).collect();
        check64(case, &gather_rows(&run()[0].to_vec().unwrap(), n, &rows), &want, 2e-3, 2e-3);
        let bytes = 4.0 * (m * k + k * 5 * n / 2 + m * 5 * n / 2) as f64;
        h.case(case, &size, Work::Bytes(bytes), &mut || {
            for out in run() {
                out.eval().unwrap();
            }
        });
    });
}

/// `relu(x W + b)` at 4096: the epilogue's share of a compute-bound product.
fn bias_relu_4096(h: &mut Harness) {
    let flops = 2.0 * 4096.0 * 4096.0 * 4096.0;
    if !fits(h, flops) {
        return;
    }
    let case = "fuse/bias_relu_4096";
    h.guarded(case, |h| {
        let d = 4096;
        let xs = values(d * d, 31, -1.0, 1.0);
        let s = 1.0 / (d as f32).sqrt();
        let ws = values(d * d, 32, -s, s);
        let bs = values(d, 33, -0.5, 0.5);
        let (x, w, b) = (arr(h, &xs, [d, d]), arr(h, &ws, [d, d]), arr(h, &bs, d));
        let run = || (&matmul(&x, &w) + &b).maximum(0.0);
        let rows = sample(d, 4);
        let want: Vec<f64> = gemm_rows(&xs, &ws, d, d, &rows).chunks(d).flat_map(|r| r.iter().zip(&bs).map(|(v, b)| (v + f64::from(*b)).max(0.0))).collect();
        check64(case, &gather_rows(&run().to_vec().unwrap(), d, &rows), &want, 2e-3, 2e-3);
        h.case(case, "4096x4096x4096", Work::Flops(flops), &mut || run().eval().unwrap());
    });
}

pub fn run(h: &mut Harness) {
    gram(h, 4096, 1024);
    shared_operand(h, 256, 512, 512);
    shared_operand(h, 512, 1024, 1024);
    bias_relu_4096(h);
    // Latency-bound (a launch or two) up to compute-bound (4096^3).
    layer(h, Act::Relu, 32, 256, 256);
    layer(h, Act::Relu, 256, 1024, 1024);
    layer(h, Act::Relu, 1024, 1024, 1024);
    layer(h, Act::Relu, 4096, 1024, 1024);
    layer(h, Act::Relu, 4096, 4096, 4096);
    // Tall and skinny (a stream of rows through a small layer), wide output, wide input with a
    // small batch (weight bandwidth), and a long K that the product splits.
    layer(h, Act::Relu, 16384, 64, 64);
    layer(h, Act::Relu, 256, 512, 8192);
    layer(h, Act::Relu, 32, 4096, 4096);
    layer(h, Act::Relu, 256, 16384, 64);
    // Other activations at a size where the epilogue's arithmetic shows.
    layer(h, Act::Gelu, 1024, 1024, 1024);
    layer(h, Act::Tanh, 1024, 1024, 1024);
    layer(h, Act::None, 1024, 1024, 1024);
    for lay in [Layout::Nn, Layout::Nt, Layout::Tn] {
        layout(h, lay, 1024, 1024, 1024);
        layout(h, lay, 2048, 512, 128);
    }
}
