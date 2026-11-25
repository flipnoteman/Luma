//! A three-layer MLP: inference, and whole training steps (forward, softmax cross-entropy,
//! backward through matmuls with transposes, SGD or Adam over the six parameter arrays).

use super::common::*;
use crate::harness::*;
use luma::*;

#[derive(Clone, Copy)]
struct Cfg {
    name: &'static str,
    batch: usize,
    /// Input width, two hidden widths, classes.
    d: [usize; 4],
}

impl Cfg {
    fn size(&self) -> String {
        format!("b{} {}-{}-{}-{}", self.batch, self.d[0], self.d[1], self.d[2], self.d[3])
    }

    /// Floating-point operations of a forward pass, and of a step (the backward pass has two
    /// products per layer, one fewer for the first).
    fn flops(&self) -> (f64, f64) {
        let layers: Vec<f64> = (0..3).map(|l| 2.0 * self.batch as f64 * self.d[l] as f64 * self.d[l + 1] as f64).collect();
        let fwd: f64 = layers.iter().sum();
        (fwd, 3.0 * fwd - layers[0])
    }

    /// Parameter shapes in luma order: `w1, b1, w2, b2, w3, b3`.
    fn shapes(&self) -> [Vec<usize>; 6] {
        let d = self.d;
        [vec![d[1], d[0]], vec![d[1]], vec![d[2], d[1]], vec![d[2]], vec![d[3], d[2]], vec![d[3]]]
    }
}

const CONFIGS: [Cfg; 4] = [
    Cfg { name: "small", batch: 64, d: [256, 256, 256, 10] },
    Cfg { name: "medium", batch: 256, d: [784, 512, 512, 10] },
    Cfg { name: "large", batch: 1024, d: [1024, 1024, 1024, 10] },
    Cfg { name: "wide", batch: 128, d: [2048, 2048, 2048, 100] },
];

/// Host parameters (`w1, b1, ...`), He-scaled weights and small biases.
fn host_params(cfg: &Cfg) -> Vec<Vec<f32>> {
    cfg.shapes()
        .iter()
        .enumerate()
        .map(|(i, shape)| {
            let n: usize = shape.iter().product();
            if i % 2 == 0 {
                let s = (2.0 / shape[1] as f32).sqrt();
                values(n, 100 + i as u64, -s, s)
            } else {
                values(n, 100 + i as u64, -0.1, 0.1)
            }
        })
        .collect()
}

struct Net {
    p: Vec<Array<f32>>,
    x: Array<f32>,
    labels: Array<u32>,
    classes: Array<u32>,
    batch: usize,
}

impl Net {
    fn new(h: &Harness, cfg: &Cfg, hp: &[Vec<f32>], x: &[f32], labels: &[u32]) -> Net {
        let p = cfg.shapes().iter().zip(hp).map(|(s, v)| Array::from_slice_on(&h.ctx, v, shape_of(s)).unwrap()).collect();
        Net {
            p,
            x: arr(h, x, [cfg.d[0], cfg.batch]),
            labels: arr_u32(h, labels, [1, cfg.batch]),
            classes: range(h, [cfg.d[3], cfg.batch], 0),
            batch: cfg.batch,
        }
    }

    /// Lazy logits (`[classes, batch]`) and the pre-activations of the hidden layers.
    fn forward(&self) -> (Array<f32>, Array<f32>, Array<f32>) {
        let p = &self.p;
        let z1 = &matmul(&self.x, &p[0]) + &p[1];
        let z2 = &matmul(&z1.maximum(0.0), &p[2]) + &p[3];
        let logits = &matmul(&z2.maximum(0.0), &p[4]) + &p[5];
        (z1, z2, logits)
    }

    /// The loss and the gradients `[dw1, db1, dw2, db2, dw3, db3]` of the cross-entropy.
    fn grads(&self) -> (Array<f32>, Vec<Array<f32>>) {
        let p = &self.p;
        let (z1, z2, logits) = self.forward();
        let (h1, h2) = (z1.maximum(0.0), z2.maximum(0.0));
        let m = logits.max_axis(0);
        let e = (&logits - &m).exp();
        let s = e.sum_axis(0);
        let y = select(&self.classes.eq(&self.labels), 1.0f32, 0.0f32);
        let inv = 1.0 / self.batch as f32;
        let loss = ((&m + &s.ln()).sum() - (&logits * &y).sum()) * inv;
        let dl = (&(&e / &s) - &y) * inv;
        let dw3 = matmul(&h2.transpose(), &dl);
        let db3 = dl.sum_axis(1);
        let dz2 = select(&z2.gt(0.0), matmul(&dl, &p[4].transpose()), 0.0f32);
        let dw2 = matmul(&h1.transpose(), &dz2);
        let db2 = dz2.sum_axis(1);
        let dz1 = select(&z1.gt(0.0), matmul(&dz2, &p[2].transpose()), 0.0f32);
        let dw1 = matmul(&self.x.transpose(), &dz1);
        let db1 = dz1.sum_axis(1);
        (loss, vec![dw1, db1, dw2, db2, dw3, db3])
    }

    /// The hidden layers' pre-activations as the device computes them, on the host.
    fn relu_masks(&self) -> (Vec<f32>, Vec<f32>) {
        let (z1, z2, _) = self.forward();
        (z1.to_vec().unwrap(), z2.to_vec().unwrap())
    }

    /// One SGD step; the new parameters and the loss are evaluated in one batch.
    fn sgd(&mut self, lr: f32) {
        let (loss, g) = self.grads();
        let new: Vec<Array<f32>> = self.p.iter().zip(&g).map(|(p, g)| p - &(g * lr)).collect();
        let all: Vec<&Array<f32>> = new.iter().chain([&loss]).collect();
        eval_all(&all).unwrap();
        self.p = new;
    }
}

/// Host forward and backward: the loss and the gradients, in the order of the luma ones.
/// `masks` are the pre-activations of the hidden layers as the device computed them: a pre-activation
/// within rounding of zero may fall on either side of the ReLU, and that one element would
/// otherwise move a gradient by more than the tolerance.
fn host_grads(cfg: &Cfg, p: &[Vec<f32>], x: &[f32], labels: &[u32], masks: &(Vec<f32>, Vec<f32>)) -> (f64, Vec<Vec<f32>>) {
    let (b, d) = (cfg.batch, cfg.d);
    let layer = |inp: &[f32], w: &[f32], bias: &[f32], k: usize, n: usize| {
        let mut z = gemm(inp, w, b, k, n);
        add_bias(&mut z, bias);
        z
    };
    let relu = |z: &[f32]| z.iter().map(|v| v.max(0.0)).collect::<Vec<f32>>();
    let z1 = layer(x, &p[0], &p[1], d[0], d[1]);
    let h1 = relu(&z1);
    let z2 = layer(&h1, &p[2], &p[3], d[1], d[2]);
    let h2 = relu(&z2);
    let logits = layer(&h2, &p[4], &p[5], d[2], d[3]);
    let mut loss = 0.0;
    let mut dl = vec![0.0f32; b * d[3]];
    for r in 0..b {
        let (probs, lse) = softmax_row(&logits[r * d[3]..][..d[3]]);
        loss += lse - f64::from(logits[r * d[3] + labels[r] as usize]);
        for c in 0..d[3] {
            let y = f64::from(u8::from(c == labels[r] as usize));
            dl[r * d[3] + c] = ((probs[c] - y) / b as f64) as f32;
        }
    }
    let colsum = |m: &[f32], n: usize| (0..n).map(|c| m.chunks(n).map(|r| f64::from(r[c])).sum::<f64>() as f32).collect::<Vec<f32>>();
    let mask = |g: Vec<f32>, z: &[f32]| g.iter().zip(z).map(|(g, z)| if *z > 0.0 { *g } else { 0.0 }).collect::<Vec<f32>>();
    let dw3 = gemm(&transpose(&h2, b, d[2]), &dl, d[2], b, d[3]);
    let dz2 = mask(gemm(&dl, &transpose(&p[4], d[2], d[3]), b, d[3], d[2]), &masks.1);
    let dw2 = gemm(&transpose(&h1, b, d[1]), &dz2, d[1], b, d[2]);
    let dz1 = mask(gemm(&dz2, &transpose(&p[2], d[1], d[2]), b, d[2], d[1]), &masks.0);
    let dw1 = gemm(&transpose(x, b, d[0]), &dz1, d[0], b, d[1]);
    let (db3, db2, db1) = (colsum(&dl, d[3]), colsum(&dz2, d[2]), colsum(&dz1, d[1]));
    (loss / b as f64, vec![dw1, db1, dw2, db2, dw3, db3])
}

fn data(cfg: &Cfg) -> (Vec<f32>, Vec<u32>, Vec<Vec<f32>>) {
    (values(cfg.batch * cfg.d[0], 7, -1.0, 1.0), ints(cfg.batch, 8, cfg.d[3] as u32), host_params(cfg))
}

fn sgd_case(h: &mut Harness, cfg: Cfg) {
    let (fwd, step) = cfg.flops();
    let _ = fwd;
    if !fits(h, step) {
        return;
    }
    let case = format!("train/sgd_{}", cfg.name);
    h.guarded(&case, |h| {
        let (x, labels, hp) = data(&cfg);
        let mut net = Net::new(h, &cfg, &hp, &x, &labels);
        let masks = net.relu_masks();
        let loss0 = net.grads().0.item().unwrap();
        // One step against the host, with a rate large enough to see the gradient in the weights.
        let lr = 0.5;
        net.sgd(lr);
        let (loss, g) = host_grads(&cfg, &hp, &x, &labels, &masks);
        check_scalar(&format!("{case}: loss"), f64::from(loss0), loss, 1e-4);
        for (i, (new, (old, g))) in net.p.iter().zip(hp.iter().zip(&g)).enumerate() {
            let want: Vec<f32> = old.iter().zip(g).map(|(o, g)| o - lr * g).collect();
            check(&format!("{case}: param {i}"), &new.to_vec().unwrap(), &want, 1e-3, 3e-3 * lr * max_abs(g) + 1e-6);
        }
        h.case(&case, &cfg.size(), Work::Flops(step), &mut || net.sgd(0.01));
    });
}

struct Adam {
    m: Vec<Array<f32>>,
    v: Vec<Array<f32>>,
    t: i32,
}

impl Adam {
    const LR: f32 = 1e-3;
    const B1: f32 = 0.9;
    const B2: f32 = 0.999;
    const EPS: f32 = 1e-6;

    /// One step: the gradients, then the fused update of the moments and the parameters, all
    /// evaluated in one batch.
    fn step(&mut self, net: &mut Net) {
        let (lr, b1, b2, eps) = (Self::LR, Self::B1, Self::B2, Self::EPS);
        self.t += 1;
        // The bias-corrected step size changes every call: a scalar, not a new kernel.
        let alpha = lr * (1.0 - b2.powi(self.t)).sqrt() / (1.0 - b1.powi(self.t));
        let (loss, g) = net.grads();
        let mut out: Vec<Array<f32>> = Vec::with_capacity(18);
        for i in 0..6 {
            let mn = &self.m[i] * b1 + &g[i] * (1.0 - b1);
            let vn = &self.v[i] * b2 + &(&g[i] * &g[i]) * (1.0 - b2);
            let pn = &net.p[i] - &(&mn / &(vn.sqrt() + eps)) * alpha;
            out.extend([pn, mn, vn]);
        }
        let all: Vec<&Array<f32>> = out.iter().chain([&loss]).collect();
        eval_all(&all).unwrap();
        for i in 0..6 {
            net.p[i] = out[3 * i].clone();
            self.m[i] = out[3 * i + 1].clone();
            self.v[i] = out[3 * i + 2].clone();
        }
    }
}

fn adam_case(h: &mut Harness, cfg: Cfg) {
    let (_, step) = cfg.flops();
    if !fits(h, step) {
        return;
    }
    let case = format!("train/adam_{}", cfg.name);
    h.guarded(&case, |h| {
        let (x, labels, hp) = data(&cfg);
        let mut net = Net::new(h, &cfg, &hp, &x, &labels);
        let zeros: Vec<Array<f32>> = cfg.shapes().iter().map(|s| Array::zeros_on(&h.ctx, shape_of(s)).unwrap()).collect();
        let masks = net.relu_masks();
        let mut adam = Adam { m: zeros.clone(), v: zeros, t: 0 };
        adam.step(&mut net);
        let (_, g) = host_grads(&cfg, &hp, &x, &labels, &masks);
        // m = (1 - b1) g is linear in the gradient and checked tightly; the parameters move by
        // about `lr` in the sign of g (they are checked with a few elements allowed off, those
        // whose gradient is at rounding level). The default product is TF32 where the context has it:
        // the gradient of the last layer (`h2^T dl`, whose terms cancel: the rows of `dl` sum to zero) then
        // carries an error of 2^-10 of the sum of |h2 dl|, which flips the sign of the entries near zero
        // and so moves their parameter by `2 lr` the other way. That is 0.27 % of `w3` here (7 of 2560,
        // 14 of 5120; none in f32 under `LUMA_NUMERICS=strict`, none on wgpu), so under TF32 the
        // allowance is 1 %.
        let frac = if h.ctx.numerics().tf32_matmul() { 1e-2 } else { 2e-3 };
        let (lr, b1, b2, eps) = (Adam::LR, Adam::B1, Adam::B2, Adam::EPS);
        for i in 0..6 {
            let want: Vec<f32> = g[i].iter().map(|g| (1.0 - b1) * g).collect();
            check(&format!("{case}: m{i}"), &adam.m[i].to_vec().unwrap(), &want, 1e-3, 5e-3 * (1.0 - b1) * max_abs(&g[i]) + 1e-9);
            let alpha = lr * (1.0 - b2).sqrt() / (1.0 - b1);
            let wanted: Vec<f32> = hp[i].iter().zip(&g[i]).map(|(p, g)| p - alpha * (1.0 - b1) * g / (((1.0 - b2) * g * g).sqrt() + eps)).collect();
            check_frac(&format!("{case}: param {i}"), &net.p[i].to_vec().unwrap(), &wanted, 1e-3, 2e-2 * lr, frac);
        }
        h.case(&case, &cfg.size(), Work::Flops(step), &mut || adam.step(&mut net));
    });
}

fn inference_case(h: &mut Harness, cfg: Cfg) {
    let (fwd, _) = cfg.flops();
    if !fits(h, fwd) {
        return;
    }
    let case = format!("train/inference_{}", cfg.name);
    h.guarded(&case, |h| {
        let (x, labels, hp) = data(&cfg);
        let net = Net::new(h, &cfg, &hp, &x, &labels);
        // Predicted classes and the largest probability of every row.
        let run = || {
            let (_, _, logits) = net.forward();
            let class = logits.argmax_axis(0);
            let conf = 1.0 / (&logits - &logits.max_axis(0)).exp().sum_axis(0);
            (class, conf)
        };
        let (class, conf) = run();
        let (class, conf) = (class.to_vec().unwrap(), conf.to_vec().unwrap());
        let d = cfg.d;
        let mut z = gemm(&x, &hp[0], cfg.batch, d[0], d[1]);
        add_bias(&mut z, &hp[1]);
        z.iter_mut().for_each(|v| *v = v.max(0.0));
        let mut z2 = gemm(&z, &hp[2], cfg.batch, d[1], d[2]);
        add_bias(&mut z2, &hp[3]);
        z2.iter_mut().for_each(|v| *v = v.max(0.0));
        let mut logits = gemm(&z2, &hp[4], cfg.batch, d[2], d[3]);
        add_bias(&mut logits, &hp[5]);
        let mut wrong = 0;
        for r in 0..cfg.batch {
            let row = &logits[r * d[3]..][..d[3]];
            let best = (0..d[3]).max_by(|&a, &b| row[a].total_cmp(&row[b])).unwrap();
            if best as u32 != class[r] {
                // Near ties may legitimately flip with the summation order.
                assert!((row[best] - row[class[r] as usize]).abs() < 1e-3, "{case}: row {r}: got class {}, want {best}", class[r]);
                wrong += 1;
            }
            let (probs, _) = softmax_row(row);
            let top = probs.iter().fold(0.0f64, |a, &b| a.max(b));
            assert!((f64::from(conf[r]) - top).abs() < 1e-3 * top + 1e-6, "{case}: row {r}: confidence {}, want {top}", conf[r]);
        }
        assert!(wrong <= 2, "{case}: {wrong} near-tie flips");
        h.case(&case, &cfg.size(), Work::Flops(fwd), &mut || {
            let (c, p) = run();
            eval_all(&[&p]).unwrap();
            c.eval().unwrap();
        });
    });
}

pub fn run(h: &mut Harness) {
    for cfg in CONFIGS {
        sgd_case(h, cfg);
    }
    for cfg in [CONFIGS[0], CONFIGS[1], CONFIGS[2]] {
        adam_case(h, cfg);
    }
    for cfg in [CONFIGS[1], CONFIGS[2]] {
        inference_case(h, cfg);
    }
}
