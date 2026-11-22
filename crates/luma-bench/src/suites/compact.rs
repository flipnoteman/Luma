//! The `compact` suite: workloads on the 8-bit and 16-bit integer arrays (`u8`, `i16`) next to
//! the same work on `f32` / `u32` arrays, which read and write twice to four times the bytes:
//! host transfers, a cast fused into a reduction, per-column statistics of a trace matrix (mean,
//! variance and Welch's t-statistic over the traces, the shape of side-channel and sensor
//! analysis) and byte histograms.
//!
//! The matrix of `traces` rows of `samples` columns is one luma array of shape
//! `[samples, traces]` (axis 0 fastest), its columns reduced along axis 1.

use crate::harness::*;
use luma::*;

pub fn run(h: &mut Harness) {
    h.guarded("transfer", transfer_cases);
    h.guarded("cast", cast_cases);
    h.guarded("elementwise", elementwise_cases);
    h.guarded("views", view_cases);
    h.guarded("colstats", colstats_cases);
    h.guarded("hist", hist_cases);
}

/// The elements of the big arrays: 64M on a GPU in a full run, 8M on the CPU backend, 4M with `--quick`.
fn elems(h: &Harness) -> usize {
    if h.quick {
        4 << 20
    } else if h.ctx.is_cpu() {
        8 << 20
    } else {
        64 << 20
    }
}

/// The trace matrix: `(samples, traces)` holding about [`elems`] elements.
fn matrix(h: &Harness) -> (usize, usize) {
    let samples = 1024;
    (samples, elems(h) / samples)
}

fn lcg(n: usize, seed: u64) -> impl Iterator<Item = u64> {
    let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    (0..n).map(move |_| {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        state >> 24
    })
}

/// Samples of a noisy signal: 12-bit noise around a per-column level, so that the columns differ.
fn traces(samples: usize, count: usize, seed: u64) -> Vec<i16> {
    let noise: Vec<u64> = lcg(samples * count, seed).collect();
    (0..samples * count).map(|i| (((noise[i] % 4096) as i64) - 2048 + ((i % samples) as i64 * 7 % 600) - 300) as i16).collect()
}

fn close(name: &str, got: &[f32], want: &[f64], rel: f64) {
    assert_eq!(got.len(), want.len(), "{name}: length");
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        assert!((f64::from(g) - w).abs() <= rel * w.abs().max(1.0), "{name}: element {i}: got {g}, want {w}");
    }
}

fn transfer_cases(h: &mut Harness) {
    let n = elems(h);
    let sz = n.to_string();
    let hi: Vec<i16> = lcg(n, 1).map(|v| v as i16).collect();
    let hf: Vec<f32> = hi.iter().map(|&v| f32::from(v)).collect();
    let ctx = h.ctx.clone();
    if h.want("transfer/upload_i16") {
        assert_eq!(Array::from_slice_on(&ctx, &hi, n).unwrap().to_vec().unwrap(), hi);
        h.case("transfer/upload_i16", &sz, Work::Bytes(2.0 * n as f64), &mut || {
            Array::from_slice_on(&ctx, &hi, n).unwrap().eval().unwrap();
        });
    }
    if h.want("transfer/upload_f32") {
        h.case("transfer/upload_f32", &sz, Work::Bytes(4.0 * n as f64), &mut || {
            Array::from_slice_on(&ctx, &hf, n).unwrap().eval().unwrap();
        });
    }
    let (xi, xf) = (Array::from_slice_on(&ctx, &hi, n).unwrap(), Array::from_slice_on(&ctx, &hf, n).unwrap());
    let mut out_i = vec![0i16; n];
    let mut out_f = vec![0f32; n];
    h.case("transfer/download_i16", &sz, Work::Bytes(2.0 * n as f64), &mut || xi.read_into(&mut out_i).unwrap());
    h.case("transfer/download_f32", &sz, Work::Bytes(4.0 * n as f64), &mut || xf.read_into(&mut out_f).unwrap());
}

fn cast_cases(h: &mut Harness) {
    let n = elems(h);
    let sz = n.to_string();
    let hi: Vec<i16> = lcg(n, 2).map(|v| (v as i16) / 4).collect();
    let hf: Vec<f32> = hi.iter().map(|&v| f32::from(v)).collect();
    let (xi, xf) = (Array::from_slice_on(&h.ctx, &hi, n).unwrap(), Array::from_slice_on(&h.ctx, &hf, n).unwrap());
    let want: i64 = hi.iter().map(|&v| i64::from(v)).sum();
    // The cast is fused into the sum: the kernel reads 2 bytes per element.
    for (name, got) in [("cast/sum_i16_as_f32", f64::from(xi.to_f32().sum().item().unwrap())), ("cast/sum_i16", f64::from(xi.sum().item().unwrap()))] {
        assert!((got - want as f64).abs() <= 1e-3 * (want as f64).abs().max(1000.0), "{name}: {got} vs {want}");
    }
    case_trim(h, "cast/sum_i16", &sz, Work::Bytes(2.0 * n as f64), &mut || {
        xi.sum().item().unwrap();
    });
    case_trim(h, "cast/sum_i16_as_f32", &sz, Work::Bytes(2.0 * n as f64), &mut || {
        xi.to_f32().sum().item().unwrap();
    });
    case_trim(h, "cast/sum_f32", &sz, Work::Bytes(4.0 * n as f64), &mut || {
        xf.sum().item().unwrap();
    });
    // The widening cast stored: reads 2 bytes, writes 4.
    case_trim(h, "cast/i16_to_f32_store", &sz, Work::Bytes(6.0 * n as f64), &mut || xi.to_f32().eval().unwrap());
    // The narrowing cast stored: reads 4, writes 2.
    case_trim(h, "cast/f32_to_i16_store", &sz, Work::Bytes(6.0 * n as f64), &mut || xf.to_i16().eval().unwrap());
}

fn elementwise_cases(h: &mut Harness) {
    let n = elems(h);
    let sz = n.to_string();
    let hu: Vec<u8> = lcg(n, 3).map(|v| v as u8).collect();
    let hw: Vec<u32> = hu.iter().map(|&v| u32::from(v)).collect();
    let (xu, xw) = (Array::from_slice_on(&h.ctx, &hu, n).unwrap(), Array::from_slice_on(&h.ctx, &hw, n).unwrap());
    let want: Vec<u8> = hu.iter().map(|&v| (v.wrapping_mul(3).wrapping_add(1)) ^ 0x5a).collect();
    assert_eq!(((&xu * 3u8 + 1u8) ^ 0x5au8).to_vec().unwrap(), want);
    case_trim(h, "elementwise/u8_chain", &sz, Work::Bytes(2.0 * n as f64), &mut || ((&xu * 3u8 + 1u8) ^ 0x5au8).eval().unwrap());
    case_trim(h, "elementwise/u32_chain", &sz, Work::Bytes(8.0 * n as f64), &mut || ((&xw * 3u32 + 1u32) ^ 0x5au32).eval().unwrap());
    // Two inputs and a comparison select, the shape of a threshold or blend.
    let hv: Vec<u8> = lcg(n, 4).map(|v| v as u8).collect();
    let xv = Array::from_slice_on(&h.ctx, &hv, n).unwrap();
    let want: Vec<u8> = hu.iter().zip(&hv).map(|(&a, &b)| if a > b { a } else { b.wrapping_add(a) }).collect();
    assert_eq!(select(&xu.gt(&xv), &xu, &(&xv + &xu)).to_vec().unwrap(), want);
    case_trim(h, "elementwise/u8_select", &sz, Work::Bytes(3.0 * n as f64), &mut || select(&xu.gt(&xv), &xu, &(&xv + &xu)).eval().unwrap());
}

/// Welch's t-statistic of every column between the traces `group` selects (a `[1, traces]` array of
/// 0 and 1) and the others: sums and sums of squares of all traces and of the group's, in two
/// reductions over the traces.
fn ttest(x: &Array<f32>, group: &Array<f32>, n1: f32, n0: f32) -> Array<f32> {
    let [st, qt] = x.reduce_axis(1, [Reduction::Sum, Reduction::SumOfSquares]);
    let [s1, q1] = (x * group).reduce_axis(1, [Reduction::Sum, Reduction::SumOfSquares]);
    let (s0, q0) = (&st - &s1, &qt - &q1);
    let (m1, m0) = (&s1 / n1, &s0 / n0);
    let v1 = &(&q1 / n1) - &(&m1 * &m1);
    let v0 = &(&q0 / n0) - &(&m0 * &m0);
    (&m1 - &m0) / (&(&v1 / n1) + &(&v0 / n0)).sqrt()
}

fn view_cases(h: &mut Harness) {
    let (w, rows) = (2048usize, elems(h) / 2048);
    let n = w * rows;
    let sz = format!("{w}x{rows}");
    let hi = traces(w, rows, 9);
    let hf: Vec<f32> = hi.iter().map(|&v| f32::from(v)).collect();
    let (xi, xf) = (Array::from_slice_on(&h.ctx, &hi, [w, rows]).unwrap(), Array::from_slice_on(&h.ctx, &hf, [w, rows]).unwrap());
    // The transposed copy: element (c, r) of the result is (r, c) of the source.
    let t = xi.transpose().to_vec().unwrap();
    assert!((0..rows).step_by(97).all(|r| (0..w).step_by(61).all(|c| t[c * rows + r] == hi[r * w + c])), "transpose");
    case_trim(h, "views/transpose_i16", &sz, Work::Bytes(4.0 * n as f64), &mut || xi.transpose().cache().eval().unwrap());
    case_trim(h, "views/transpose_f32", &sz, Work::Bytes(8.0 * n as f64), &mut || xf.transpose().cache().eval().unwrap());
    // A difference of neighbouring rows (a shifted read of the same buffer) and a flip.
    let want: Vec<i16> = (0..n - w).map(|i| hi[i + w].wrapping_sub(hi[i])).collect();
    let d = (xi.slice(1, 1..rows) - xi.slice(1, ..rows - 1)).to_vec().unwrap();
    assert_eq!(d, want);
    case_trim(h, "views/row_difference_i16", &sz, Work::Bytes(6.0 * n as f64), &mut || (xi.slice(1, 1..rows) - xi.slice(1, ..rows - 1)).eval().unwrap());
    case_trim(h, "views/row_difference_f32", &sz, Work::Bytes(12.0 * n as f64), &mut || (xf.slice(1, 1..rows) - xf.slice(1, ..rows - 1)).eval().unwrap());
    // A gather: random elements.
    let idx: Vec<u32> = lcg(n, 10).map(|v| (v % n as u64) as u32).collect();
    let ix = Array::from_slice_on(&h.ctx, &idx, n).unwrap();
    let flat_i = xi.reshape([n]);
    let flat_f = xf.reshape([n]);
    let got = flat_i.take(&ix).to_vec().unwrap();
    assert!(got.iter().zip(&idx).step_by(101).all(|(&g, &i)| g == hi[i as usize]), "take");
    case_trim(h, "views/take_i16", &sz, Work::Bytes(6.0 * n as f64), &mut || flat_i.take(&ix).eval().unwrap());
    case_trim(h, "views/take_f32", &sz, Work::Bytes(8.0 * n as f64), &mut || flat_f.take(&ix).eval().unwrap());
}

fn colstats_cases(h: &mut Harness) {
    let (s, t) = matrix(h);
    let n = s * t;
    let sz = format!("{s}x{t}");
    let hi = traces(s, t, 5);
    let hf: Vec<f32> = hi.iter().map(|&v| f32::from(v)).collect();
    let flags: Vec<f32> = (0..t).map(|i| if lcg(1, i as u64 + 7).next().unwrap() % 2 == 0 { 1.0 } else { 0.0 }).collect();
    let n1 = flags.iter().sum::<f32>();
    let n0 = t as f32 - n1;
    let group = Array::from_slice_on(&h.ctx, &flags, [1, t]).unwrap();
    let xi = Array::from_slice_on(&h.ctx, &hi, [s, t]).unwrap();
    let xf = Array::from_slice_on(&h.ctx, &hf, [s, t]).unwrap();

    // The host reference, in f64.
    let (mut sum, mut sq) = (vec![0f64; s], vec![0f64; s]);
    let (mut s1, mut q1) = (vec![0f64; s], vec![0f64; s]);
    for (i, &v) in hi.iter().enumerate() {
        let (c, v) = (i % s, f64::from(v));
        sum[c] += v;
        sq[c] += v * v;
        if flags[i / s] > 0.5 {
            s1[c] += v;
            q1[c] += v * v;
        }
    }
    let mean: Vec<f64> = sum.iter().map(|&v| v / t as f64).collect();
    let var: Vec<f64> = (0..s).map(|c| sq[c] / t as f64 - mean[c] * mean[c]).collect();
    let tstat: Vec<f64> = (0..s)
        .map(|c| {
            let (m1, m0) = (s1[c] / f64::from(n1), (sum[c] - s1[c]) / f64::from(n0));
            let v1 = q1[c] / f64::from(n1) - m1 * m1;
            let v0 = (sq[c] - q1[c]) / f64::from(n0) - m0 * m0;
            (m1 - m0) / (v1 / f64::from(n1) + v0 / f64::from(n0)).sqrt()
        })
        .collect();

    close("colstats/mean_i16", &xi.mean_axis(1).to_vec().unwrap(), &mean, 1e-4);
    close("colstats/var_i16", &xi.var_axis(1).to_vec().unwrap(), &var, 1e-3);
    close("colstats/ttest_i16", &ttest(&xi.to_f32(), &group, n1, n0).to_vec().unwrap(), &tstat, 1e-2);

    let bytes_i = 2.0 * n as f64;
    let bytes_f = 4.0 * n as f64;
    case_trim(h, "colstats/mean_var_i16", &sz, Work::Bytes(bytes_i), &mut || {
        let (m, v) = (xi.mean_axis(1), xi.var_axis(1));
        eval_all(&[&m, &v]).unwrap();
    });
    case_trim(h, "colstats/mean_var_f32", &sz, Work::Bytes(bytes_f), &mut || {
        let (m, v) = (xf.mean_axis(1), xf.var_axis(1));
        eval_all(&[&m, &v]).unwrap();
    });
    case_trim(h, "colstats/mean_i16", &sz, Work::Bytes(bytes_i), &mut || xi.mean_axis(1).eval().unwrap());
    case_trim(h, "colstats/mean_f32", &sz, Work::Bytes(bytes_f), &mut || xf.mean_axis(1).eval().unwrap());
    case_trim(h, "colstats/var_i16", &sz, Work::Bytes(bytes_i), &mut || xi.var_axis(1).eval().unwrap());
    case_trim(h, "colstats/var_f32", &sz, Work::Bytes(bytes_f), &mut || xf.var_axis(1).eval().unwrap());
    case_trim(h, "colstats/sum_i16", &sz, Work::Bytes(bytes_i), &mut || xi.sum_axis(1).eval().unwrap());
    case_trim(h, "colstats/sum_f32", &sz, Work::Bytes(bytes_f), &mut || xf.sum_axis(1).eval().unwrap());
    case_trim(h, "colstats/ttest_i16", &sz, Work::Bytes(bytes_i), &mut || ttest(&xi.to_f32(), &group, n1, n0).eval().unwrap());
    case_trim(h, "colstats/ttest_f32", &sz, Work::Bytes(bytes_f), &mut || ttest(&xf, &group, n1, n0).eval().unwrap());
    // Along the contiguous axis: every row's mean.
    case_trim(h, "colstats/rowmean_i16", &sz, Work::Bytes(bytes_i), &mut || xi.mean_axis(0).eval().unwrap());
    case_trim(h, "colstats/rowmean_f32", &sz, Work::Bytes(bytes_f), &mut || xf.mean_axis(0).eval().unwrap());
}

fn hist_cases(h: &mut Harness) {
    let n = elems(h);
    let sz = n.to_string();
    let hu: Vec<u8> = lcg(n, 6).map(|v| (v as u8) & (v >> 8) as u8).collect();
    let hw: Vec<u32> = hu.iter().map(|&v| u32::from(v)).collect();
    let (xu, xw) = (Array::from_slice_on(&h.ctx, &hu, n).unwrap(), Array::from_slice_on(&h.ctx, &hw, n).unwrap());
    let mut want = vec![0u32; 256];
    hu.iter().for_each(|&v| want[v as usize] += 1);
    assert_eq!(xu.histogram().to_vec().unwrap(), want);
    assert_eq!(bincount(&xw, 256).to_vec().unwrap(), want);
    case_trim(h, "hist/u8_256", &sz, Work::Bytes(n as f64), &mut || xu.histogram().eval().unwrap());
    case_trim(h, "hist/u32_256", &sz, Work::Bytes(4.0 * n as f64), &mut || bincount(&xw, 256).eval().unwrap());
    let hs: Vec<u16> = lcg(n, 7).map(|v| (v as u16) & 0x3fff).collect();
    let xs = Array::from_slice_on(&h.ctx, &hs, n).unwrap();
    let mut want = vec![0u32; 16384];
    hs.iter().for_each(|&v| want[v as usize] += 1);
    assert_eq!(xs.bincount(16384).to_vec().unwrap(), want);
    case_trim(h, "hist/u16_16384", &sz, Work::Bytes(2.0 * n as f64), &mut || xs.bincount(16384).eval().unwrap());
}

/// [`Harness::case`] after releasing the pooled idle buffers of the earlier cases.
fn case_trim(h: &mut Harness, name: &str, size: &str, work: Work, f: &mut dyn FnMut()) {
    if h.want(name) {
        h.ctx.trim().unwrap();
    }
    h.case(name, size, work, f);
}
