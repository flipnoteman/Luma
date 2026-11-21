//! Convolutions: luma's `conv1d` (direct and FFT), 2D convolution as im2col (shifted views
//! concatenated along the channel axis) times a matrix, a depthwise 3x3 (nine fused taps) and a
//! 5x5 single-channel stencil.

use super::common::*;
use crate::harness::*;
use luma::*;

fn len_of(h: &Harness) -> usize {
    size(h, 1 << 18, 1 << 20, 1 << 18)
}

/// Direct 1D convolution (cross-correlation), zero-padded by `k / 2`.
fn direct(h: &mut Harness, ksize: usize, stride: usize) {
    let len = len_of(h);
    let flops = 2.0 * len as f64 * ksize as f64 / stride as f64;
    if !fits(h, flops) {
        return;
    }
    let case = "conv/conv1d";
    let size = format!("L{len} k{ksize} s{stride}");
    h.guarded(&format!("{case} {size}"), |h| {
        let xs = values(len, 61, -1.0, 1.0);
        let s = 1.0 / (ksize as f32).sqrt();
        let ks = values(ksize, 62, -s, s);
        let (x, k) = (arr(h, &xs, len), arr(h, &ks, ksize));
        let pad = ksize / 2;
        let out_len = (len + 2 * pad - ksize) / stride + 1;
        let run = || x.conv1d(&k, stride, pad);
        let got = run().to_vec().unwrap();
        assert_eq!(got.len(), out_len, "{case}: output length");
        for i in sample(out_len, 64) {
            let want: f64 = (0..ksize).filter_map(|j| (i * stride + j >= pad && i * stride + j - pad < len).then(|| f64::from(xs[i * stride + j - pad]) * f64::from(ks[j]))).sum();
            check_scalar_abs(case, f64::from(got[i]), want, 1e-4);
        }
        h.case(case, &size, Work::Flops(flops), &mut || run().eval().unwrap());
    });
}

/// FFT convolution: the full linear convolution of length `L + k - 1`.
fn fft(h: &mut Harness, ksize: usize) {
    let len = len_of(h);
    let case = "conv/conv1d_fft";
    let size = format!("L{len} k{ksize}");
    h.guarded(&format!("{case} {size}"), |h| {
        let xs = values(len, 63, -1.0, 1.0);
        let s = 1.0 / (ksize as f32).sqrt();
        let ks = values(ksize, 64, -s, s);
        let (x, k) = (arr(h, &xs, len), arr(h, &ks, ksize));
        let run = || x.conv1d_fft(&k);
        let got = run().to_vec().unwrap();
        assert_eq!(got.len(), len + ksize - 1, "{case}: output length");
        for i in sample(got.len(), 64) {
            let want: f64 = (0..ksize).filter_map(|j| (i >= j && i - j < len).then(|| f64::from(xs[i - j]) * f64::from(ks[j]))).sum();
            check_scalar_abs(case, f64::from(got[i]), want, 2e-3);
        }
        h.case(case, &size, Work::Items(len as f64, "Msample/s"), &mut || run().eval().unwrap());
    });
}

fn check_scalar_abs(name: &str, got: f64, want: f64, abs: f64) {
    assert!((got - want).abs() <= abs, "{name}: got {got}, want {want} (abs {abs})");
}

const TAPS: [(isize, isize); 9] = [(-1, -1), (0, -1), (1, -1), (-1, 0), (0, 0), (1, 0), (-1, 1), (0, 1), (1, 1)];

/// 3x3 convolution, same padding, bias and ReLU over NHWC images `[c, w, h, n]`, as im2col: nine
/// shifted views concatenated along the channel axis, reshaped to `[9c, pixels]`, times
/// `[cout, 9c]`.
fn conv2d(h: &mut Harness, n: usize, c: usize, cout: usize, hw: usize) {
    let (n, hw) = if h.ctx.is_cpu() { ((n / 4).max(1), hw) } else { (n, hw) };
    let pixels = n * hw * hw;
    let flops = 2.0 * pixels as f64 * (9 * c * cout) as f64;
    if !fits(h, flops) {
        return;
    }
    let case = "conv/conv2d_im2col";
    let size = format!("N{n} {c}->{cout} {hw}x{hw}");
    h.guarded(&format!("{case} {size}"), |h| {
        let xs = values(pixels * c, 65, -1.0, 1.0);
        let s = 1.0 / ((9 * c) as f32).sqrt();
        let ws = values(9 * c * cout, 66, -s, s);
        let bs = values(cout, 67, -0.2, 0.2);
        let (x, w, b) = (arr(h, &xs, [c, hw, hw, n]), arr(h, &ws, [cout, 9 * c]), arr(h, &bs, cout));
        let run = || {
            let patches: Vec<Array<f32>> = TAPS.iter().map(|&(dx, dy)| x.shift_fill([0, -dx, -dy, 0], 0.0)).collect();
            let cols = concat_slice(0, &patches).reshape([9 * c, pixels]);
            (&matmul(&cols, &w) + &b).maximum(0.0)
        };
        let got = run().to_vec().unwrap();
        for p in sample(pixels, 24) {
            let (xx, yy, img) = ((p % hw) as isize, ((p / hw) % hw) as isize, p / (hw * hw));
            for co in 0..cout {
                let mut acc = f64::from(bs[co]);
                for (t, &(dx, dy)) in TAPS.iter().enumerate() {
                    let (sx, sy) = (xx + dx, yy + dy);
                    if sx < 0 || sy < 0 || sx >= hw as isize || sy >= hw as isize {
                        continue;
                    }
                    let base = ((img * hw + sy as usize) * hw + sx as usize) * c;
                    for ci in 0..c {
                        acc += f64::from(xs[base + ci]) * f64::from(ws[(t * c + ci) * cout + co]);
                    }
                }
                check_scalar_abs(case, f64::from(got[p * cout + co]), acc.max(0.0), 2e-3);
            }
        }
        h.case(case, &size, Work::Flops(flops), &mut || run().eval().unwrap());
    });
}

/// Depthwise 3x3 convolution of `[c, w, h]`: nine taps of per-channel weights in one expression.
fn depthwise(h: &mut Harness, c: usize, hw: usize) {
    let hw = if h.ctx.is_cpu() { hw / 2 } else { hw };
    let case = "conv/depthwise3x3";
    let size = format!("C{c} {hw}x{hw}");
    h.guarded(&format!("{case} {size}"), |h| {
        let xs = values(c * hw * hw, 68, -1.0, 1.0);
        let ws = values(9 * c, 69, -0.5, 0.5);
        let x = arr(h, &xs, [c, hw, hw]);
        let taps: Vec<Array<f32>> = (0..9).map(|t| arr(h, &ws[t * c..][..c], [c, 1, 1])).collect();
        let run = || {
            let mut acc = &taps[0] * &x.shift_fill([0, -TAPS[0].0, -TAPS[0].1], 0.0);
            for (t, &(dx, dy)) in TAPS.iter().enumerate().skip(1) {
                acc = acc + &taps[t] * &x.shift_fill([0, -dx, -dy], 0.0);
            }
            acc.maximum(0.0)
        };
        let got = run().to_vec().unwrap();
        for p in sample(hw * hw, 24) {
            let (xx, yy) = ((p % hw) as isize, (p / hw) as isize);
            for ch in sample(c, 8) {
                let mut acc = 0.0f64;
                for (t, &(dx, dy)) in TAPS.iter().enumerate() {
                    let (sx, sy) = (xx + dx, yy + dy);
                    if sx >= 0 && sy >= 0 && sx < hw as isize && sy < hw as isize {
                        acc += f64::from(xs[(sy as usize * hw + sx as usize) * c + ch]) * f64::from(ws[t * c + ch]);
                    }
                }
                check_scalar_abs(case, f64::from(got[p * c + ch]), acc.max(0.0), 1e-4);
            }
        }
        h.case(case, &size, Work::Bytes(8.0 * (c * hw * hw) as f64), &mut || run().eval().unwrap());
    });
}

/// A 5x5 Gaussian on one channel with clamped edges: 25 scalar-weighted taps.
fn stencil5(h: &mut Harness, side: usize) {
    let side = if h.ctx.is_cpu() { side / 2 } else { side };
    let case = "conv/stencil5x5";
    let size = format!("{side}x{side}");
    h.guarded(&format!("{case} {size}"), |h| {
        let xs = values(side * side, 70, 0.0, 1.0);
        let x = arr(h, &xs, [side, side]);
        let g = [1.0f32, 4.0, 6.0, 4.0, 1.0];
        let weight = |i: usize, j: usize| g[i] * g[j] / 256.0;
        let run = || {
            let mut acc: Option<Array<f32>> = None;
            for j in 0..5 {
                for i in 0..5 {
                    let tap = &x.shift_clamp([-(i as isize - 2), -(j as isize - 2)]) * weight(i, j);
                    acc = Some(match acc {
                        Some(a) => a + tap,
                        None => tap,
                    });
                }
            }
            acc.unwrap()
        };
        let got = run().to_vec().unwrap();
        let at = |x: isize, y: isize| xs[y.clamp(0, side as isize - 1) as usize * side + x.clamp(0, side as isize - 1) as usize];
        for p in sample(side * side, 64) {
            let (px, py) = ((p % side) as isize, (p / side) as isize);
            let want: f64 = (0..5).flat_map(|j| (0..5).map(move |i| (i, j))).map(|(i, j)| f64::from(at(px + i as isize - 2, py + j as isize - 2)) * f64::from(weight(i, j))).sum();
            check_scalar_abs(case, f64::from(got[p]), want, 1e-5);
        }
        h.case(case, &size, Work::Bytes(8.0 * (side * side) as f64), &mut || run().eval().unwrap());
    });
}

pub fn run(h: &mut Harness) {
    for k in [3, 15, 63] {
        direct(h, k, 1);
    }
    direct(h, 7, 2);
    direct(h, 1023, 1);
    for k in [63, 1023, 4095] {
        fft(h, k);
    }
    conv2d(h, 4, 3, 32, 224);
    conv2d(h, 8, 64, 64, 56);
    conv2d(h, 32, 128, 128, 14);
    depthwise(h, 64, 256);
    stencil5(h, 2048);
}
