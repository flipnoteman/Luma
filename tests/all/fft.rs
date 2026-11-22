//! Fourier transforms against an `f64` host reference.

use crate::common::values;
use luma::{Array, ConvMode, Context, FftNorm, C32};

type Z = (f64, f64);

/// The DFT of `x` in `f64`: mixed radix on the smallest prime factor, the naive sum for a prime.
fn dft(x: &[Z], inverse: bool) -> Vec<Z> {
    let n = x.len();
    if n <= 1 {
        return x.to_vec();
    }
    let p = (2..=n).find(|&p| n.is_multiple_of(p)).unwrap();
    let sign = if inverse { 1.0 } else { -1.0 };
    let w = |k: usize| {
        let a = sign * 2.0 * std::f64::consts::PI * (k % n) as f64 / n as f64;
        (a.cos(), a.sin())
    };
    if p == n {
        return (0..n)
            .map(|k| {
                let mut s = (0.0, 0.0);
                for (j, v) in x.iter().enumerate() {
                    let t = w(j * k);
                    s.0 += v.0 * t.0 - v.1 * t.1;
                    s.1 += v.0 * t.1 + v.1 * t.0;
                }
                s
            })
            .collect();
    }
    let m = n / p;
    // x[j] with j = p * a + r: transform every residue class, then combine.
    let subs: Vec<Vec<Z>> = (0..p).map(|r| dft(&(0..m).map(|a| x[p * a + r]).collect::<Vec<_>>(), inverse)).collect();
    (0..n)
        .map(|k| {
            let mut s = (0.0, 0.0);
            for (r, sub) in subs.iter().enumerate() {
                let t = w(r * k);
                let v = sub[k % m];
                s.0 += v.0 * t.0 - v.1 * t.1;
                s.1 += v.0 * t.1 + v.1 * t.0;
            }
            s
        })
        .collect()
}

fn signal(n: usize, seed: u64) -> Vec<C32> {
    let re = values(n, seed, -1.0, 1.0);
    let im = values(n, seed + 1000, -1.0, 1.0);
    re.iter().zip(&im).map(|(&re, &im)| C32::new(re, im)).collect()
}

fn to_z(x: &[C32]) -> Vec<Z> {
    x.iter().map(|c| (f64::from(c.re), f64::from(c.im))).collect()
}

/// Asserts the largest error is within `tol` of the rms of `want`.
fn check(got: &[C32], want: &[Z], tol: f64, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    let rms = (want.iter().map(|z| z.0 * z.0 + z.1 * z.1).sum::<f64>() / want.len().max(1) as f64).sqrt().max(1e-30);
    let mut worst = 0.0f64;
    let mut at = 0;
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let e = ((f64::from(g.re) - w.0).powi(2) + (f64::from(g.im) - w.1).powi(2)).sqrt();
        if e.is_nan() || e > worst {
            worst = e;
            at = i;
        }
    }
    assert!(worst <= tol * rms, "{what}: error {worst:e} at {at} against rms {rms:e} (got {:?}, want {:?})", got[at], want[at]);
}

const LENGTHS: &[usize] = &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 25, 27, 30, 31, 32, 49, 60, 64, 100, 121, 125, 128, 243, 256, 343, 500, 1000, 1024, 2000, 2048, 3125, 4096];

#[test]
fn fft_of_every_length_matches_the_reference() {
    let ctx = Context::new().unwrap();
    for &n in LENGTHS {
        let x = signal(n, n as u64);
        let a = Array::from_slice_on(&ctx, &x, n).unwrap();
        check(&a.fft().to_vec().unwrap(), &dft(&to_z(&x), false), 2e-6, &format!("fft {n}"));
        check(&a.ifft().to_vec().unwrap(), &dft(&to_z(&x), true).iter().map(|z| (z.0 / n as f64, z.1 / n as f64)).collect::<Vec<_>>(), 2e-6, &format!("ifft {n}"));
    }
}

#[test]
fn large_and_multi_pass_lengths() {
    let ctx = Context::new().unwrap();
    for &n in &[4097usize, 8192, 12288, 16384, 20000, 65536, 100000, 131072] {
        let x = signal(n, 7 + n as u64);
        let a = Array::from_slice_on(&ctx, &x, n).unwrap();
        check(&a.fft().to_vec().unwrap(), &dft(&to_z(&x), false), 4e-6, &format!("fft {n}"));
    }
}

#[test]
fn lengths_with_large_prime_factors_use_bluestein() {
    let ctx = Context::new().unwrap();
    for &n in &[37usize, 41, 97, 101, 127, 257, 509, 74, 999 * 2 + 1, 1009, 1000 + 7 * 41, 4099, 10007] {
        let x = signal(n, 31 + n as u64);
        let a = Array::from_slice_on(&ctx, &x, n).unwrap();
        check(&a.fft().to_vec().unwrap(), &dft(&to_z(&x), false), 1e-5, &format!("fft {n}"));
        let inv = dft(&to_z(&x), true).iter().map(|z| (z.0 / n as f64, z.1 / n as f64)).collect::<Vec<_>>();
        check(&a.ifft().to_vec().unwrap(), &inv, 1e-5, &format!("ifft {n}"));
    }
}

/// Transforms every line of `x` (dims, first fastest) along `axis` with the host reference.
fn along(x: &[Z], dims: [usize; 4], axis: usize, inverse: bool) -> Vec<Z> {
    let stride: usize = dims[..axis].iter().product();
    let n = dims[axis];
    let mut out = x.to_vec();
    for i in 0..x.len() {
        // The first element of every line.
        if !(i / stride).is_multiple_of(n) {
            continue;
        }
        let line: Vec<Z> = (0..n).map(|k| x[i + k * stride]).collect();
        for (k, v) in dft(&line, inverse).into_iter().enumerate() {
            out[i + k * stride] = v;
        }
    }
    out
}

#[test]
fn transforms_along_every_axis_of_a_batch() {
    let ctx = Context::new().unwrap();
    let shapes = [[12usize, 5, 3, 2], [8, 8, 8, 1], [1, 7, 5, 3], [37, 3, 2, 1], [3, 41, 2, 2], [4, 3, 90, 2], [2, 2, 3, 50], [30, 1, 1, 17]];
    for dims in shapes {
        let len: usize = dims.iter().product();
        let x = signal(len, len as u64);
        let a = Array::from_slice_on(&ctx, &x, dims).unwrap();
        for axis in 0..4 {
            let want = along(&to_z(&x), dims, axis, false);
            check(&a.fft_axis(axis, FftNorm::None).to_vec().unwrap(), &want, 1e-5, &format!("{dims:?} axis {axis}"));
            let want = along(&to_z(&x), dims, axis, true);
            let n = dims[axis] as f64;
            let want: Vec<Z> = want.iter().map(|z| (z.0 / n, z.1 / n)).collect();
            check(&a.ifft_axis(axis, FftNorm::Backward).to_vec().unwrap(), &want, 1e-5, &format!("{dims:?} inverse axis {axis}"));
        }
    }
}

#[test]
fn two_and_three_dimensional_transforms() {
    let ctx = Context::new().unwrap();
    for dims in [[16usize, 12, 3, 2], [30, 7, 5, 2], [64, 64, 1, 1], [10, 10, 10, 2]] {
        let len: usize = dims.iter().product();
        let x = signal(len, 5 + len as u64);
        let a = Array::from_slice_on(&ctx, &x, dims).unwrap();
        let w2 = along(&along(&to_z(&x), dims, 0, false), dims, 1, false);
        check(&a.fft2().to_vec().unwrap(), &w2, 1e-5, &format!("{dims:?} fft2"));
        let w3 = along(&w2, dims, 2, false);
        check(&a.fft3().to_vec().unwrap(), &w3, 1e-5, &format!("{dims:?} fft3"));
        let back = a.fft3().ifft3().to_vec().unwrap();
        check(&back, &to_z(&x), 1e-5, &format!("{dims:?} round trip"));
        let back = a.fft2().ifft2().to_vec().unwrap();
        check(&back, &to_z(&x), 1e-5, &format!("{dims:?} 2-D round trip"));
    }
}

#[test]
fn normalisation_options() {
    let ctx = Context::new().unwrap();
    let n = 60;
    let x = signal(n, 3);
    let a = Array::from_slice_on(&ctx, &x, n).unwrap();
    let raw = dft(&to_z(&x), false);
    let scaled = |f: f64| raw.iter().map(|z| (z.0 * f, z.1 * f)).collect::<Vec<_>>();
    check(&a.fft_axis(0, FftNorm::Backward).to_vec().unwrap(), &scaled(1.0), 1e-5, "backward");
    check(&a.fft_axis(0, FftNorm::Forward).to_vec().unwrap(), &scaled(1.0 / n as f64), 1e-5, "forward");
    check(&a.fft_axis(0, FftNorm::Ortho).to_vec().unwrap(), &scaled(1.0 / (n as f64).sqrt()), 1e-5, "ortho");
    check(&a.fft_axis(0, FftNorm::Scale(0.25)).to_vec().unwrap(), &scaled(0.25), 1e-5, "scale");
    let a2 = Array::from_slice_on(&ctx, &signal(6 * 10, 4), [6, 10]).unwrap();
    let want = along(&along(&to_z(&signal(60, 4)), [6, 10, 1, 1], 0, false), [6, 10, 1, 1], 1, false);
    let want: Vec<Z> = want.iter().map(|z| (z.0 * 0.5, z.1 * 0.5)).collect();
    check(&a2.fft_axes(&[0, 1], FftNorm::Scale(0.5)).to_vec().unwrap(), &want, 1e-5, "scale of two axes");
    let ortho = a2.fft_axes(&[0, 1], FftNorm::Ortho).ifft_axes(&[0, 1], FftNorm::Ortho).to_vec().unwrap();
    check(&ortho, &to_z(&signal(60, 4)), 1e-5, "ortho round trip");
}

#[test]
fn empty_and_single_point_arrays() {
    let ctx = Context::new().unwrap();
    let one = Array::from_slice_on(&ctx, &[C32::new(2.0, -3.0)], 1).unwrap();
    assert_eq!(one.fft().to_vec().unwrap(), vec![C32::new(2.0, -3.0)]);
    assert_eq!(one.ifft().to_vec().unwrap(), vec![C32::new(2.0, -3.0)]);
    let bad = one.fft_axis(4, FftNorm::Backward);
    assert!(bad.to_vec().is_err());
    assert!(one.fft_axes(&[], FftNorm::Backward).to_vec().is_err());
}

fn real_signal(n: usize, seed: u64) -> Vec<f32> {
    values(n, seed, -1.0, 1.0)
}

fn lift(x: &[f32]) -> Vec<Z> {
    x.iter().map(|&v| (f64::from(v), 0.0)).collect()
}

#[test]
fn real_input_fft_and_rfft() {
    let ctx = Context::new().unwrap();
    for &n in &[1usize, 2, 3, 4, 5, 6, 8, 9, 10, 12, 16, 30, 31, 37, 64, 74, 100, 127, 128, 250, 1000, 1024, 2050, 4096, 8192, 10007, 10010] {
        let x = real_signal(n, 11 + n as u64);
        let a = Array::from_slice_on(&ctx, &x, n).unwrap();
        let want = dft(&lift(&x), false);
        check(&a.fft().to_vec().unwrap(), &want, 1e-5, &format!("real fft {n}"));
        let half = a.rfft();
        assert_eq!(half.shape().dims()[0], n / 2 + 1);
        check(&half.to_vec().unwrap(), &want[..n / 2 + 1], 1e-5, &format!("rfft {n}"));
        let back = half.irfft(n).to_vec().unwrap();
        let got: Vec<Z> = lift(&back);
        check_real(&got, &x, 1e-5, &format!("irfft {n}"));
    }
}

fn check_real(got: &[Z], want: &[f32], tol: f64, what: &str) {
    let w: Vec<Z> = lift(want);
    let g: Vec<C32> = got.iter().map(|z| C32::new(z.0 as f32, z.1 as f32)).collect();
    check(&g, &w, tol, what);
}

#[test]
fn rfft_along_other_axes_and_in_2d_and_3d() {
    let ctx = Context::new().unwrap();
    for dims in [[10usize, 6, 3, 2], [7, 12, 4, 1], [16, 16, 16, 1], [9, 20, 2, 3]] {
        let len: usize = dims.iter().product();
        let x = real_signal(len, 9 + len as u64);
        let a = Array::from_slice_on(&ctx, &x, dims).unwrap();
        let full = to_z_real(&x);
        for axis in 0..4 {
            let want = along(&full, dims, axis, false);
            let got = a.rfft_axis(axis, FftNorm::None);
            let mut d = dims;
            d[axis] = dims[axis] / 2 + 1;
            assert_eq!(got.shape().dims(), d);
            let want = take_half(&want, dims, axis);
            check(&got.to_vec().unwrap(), &want, 1e-5, &format!("{dims:?} rfft axis {axis}"));
            let back = got.irfft_axes(&[axis], dims[axis], FftNorm::None).to_vec().unwrap();
            let scaled: Vec<f32> = x.iter().map(|v| v * dims[axis] as f32).collect();
            check_real(&lift(&back), &scaled, 1e-5, &format!("{dims:?} irfft axis {axis}"));
        }
        let w2 = along(&along(&full, dims, 0, false), dims, 1, false);
        let h2 = a.rfft2();
        check(&h2.to_vec().unwrap(), &take_half(&w2, dims, 0), 1e-5, &format!("{dims:?} rfft2"));
        check_real(&lift(&h2.irfft2(dims[0]).to_vec().unwrap()), &x, 1e-5, &format!("{dims:?} irfft2"));
        let w3 = along(&w2, dims, 2, false);
        let h3 = a.rfft3();
        check(&h3.to_vec().unwrap(), &take_half(&w3, dims, 0), 1e-5, &format!("{dims:?} rfft3"));
        check_real(&lift(&h3.irfft3(dims[0]).to_vec().unwrap()), &x, 1e-5, &format!("{dims:?} irfft3"));
    }
}

fn to_z_real(x: &[f32]) -> Vec<Z> {
    lift(x)
}

/// The first `dims[axis] / 2 + 1` entries along `axis`.
fn take_half(x: &[Z], dims: [usize; 4], axis: usize) -> Vec<Z> {
    let stride: usize = dims[..axis].iter().product();
    let keep = dims[axis] / 2 + 1;
    x.iter().enumerate().filter(|(i, _)| (i / stride) % dims[axis] < keep).map(|(_, z)| *z).collect()
}

/// The full linear convolution of the first `nd` axes of `x` (dims `sd`) with `k` (dims `kd`),
/// batched over the rest, in `f64`.
fn convolve_ref(x: &[f32], sd: [usize; 4], k: &[f32], kd: [usize; 4], nd: usize) -> (Vec<f64>, [usize; 4]) {
    let mut od = sd;
    for d in 0..nd {
        od[d] = sd[d] + kd[d] - 1;
    }
    let mut out = vec![0.0f64; od.iter().product()];
    let at = |dims: [usize; 4], c: [usize; 4]| c[0] + dims[0] * (c[1] + dims[1] * (c[2] + dims[2] * c[3]));
    for c3 in 0..sd[3] {
        for c2 in 0..sd[2] {
            for c1 in 0..sd[1] {
                for c0 in 0..sd[0] {
                    let v = f64::from(x[at(sd, [c0, c1, c2, c3])]);
                    for j2 in 0..kd[2] {
                        for j1 in 0..kd[1] {
                            for j0 in 0..kd[0] {
                                let o = at(od, [c0 + j0, c1 + j1, c2 + j2, c3]);
                                out[o] += v * f64::from(k[at(kd, [j0, j1, j2, 0])]);
                            }
                        }
                    }
                }
            }
        }
    }
    (out, od)
}

fn crop(full: &[f64], od: [usize; 4], sd: [usize; 4], kd: [usize; 4], nd: usize) -> Vec<f64> {
    let mut out = Vec::new();
    for c3 in 0..sd[3] {
        for c2 in 0..sd[2] {
            for c1 in 0..sd[1] {
                for c0 in 0..sd[0] {
                    let mut c = [c0, c1, c2, c3];
                    for d in 0..nd {
                        c[d] += kd[d] / 2;
                    }
                    out.push(full[c[0] + od[0] * (c[1] + od[1] * (c[2] + od[2] * c[3]))]);
                }
            }
        }
    }
    out
}

#[test]
fn fft_convolution_matches_direct_sums() {
    let ctx = Context::new().unwrap();
    let cases: &[(usize, [usize; 4], [usize; 4])] = &[
        (1, [100, 1, 1, 1], [7, 1, 1, 1]),
        (1, [33, 4, 2, 1], [8, 1, 1, 1]),
        (2, [20, 17, 1, 1], [5, 3, 1, 1]),
        (2, [16, 16, 3, 2], [4, 6, 1, 1]),
        (2, [31, 9, 2, 1], [1, 1, 1, 1]),
        (3, [10, 9, 8, 2], [3, 4, 2, 1]),
        (3, [12, 12, 12, 1], [5, 5, 5, 1]),
    ];
    for &(nd, sd, kd) in cases {
        let (sl, kl): (usize, usize) = (sd.iter().product(), kd.iter().product());
        let x = real_signal(sl, 21 + sl as u64);
        let k = real_signal(kl, 22 + kl as u64);
        let (a, b) = (Array::from_slice_on(&ctx, &x, sd).unwrap(), Array::from_slice_on(&ctx, &k, kd).unwrap());
        let (full, od) = convolve_ref(&x, sd, &k, kd, nd);
        let run = |mode| match nd {
            1 => a.fft_convolve1(&b, mode),
            2 => a.fft_convolve2(&b, mode),
            _ => a.fft_convolve3(&b, mode),
        };
        let got = run(ConvMode::Full);
        assert_eq!(got.shape().dims(), od, "full shape {nd}");
        check_real(&got.to_vec().unwrap().iter().map(|&v| (f64::from(v), 0.0)).collect::<Vec<_>>(), &full.iter().map(|&v| v as f32).collect::<Vec<_>>(), 1e-4, &format!("{nd}-D full {sd:?} * {kd:?}"));
        let got = run(ConvMode::Same);
        assert_eq!(got.shape().dims(), sd, "same shape {nd}");
        let want = crop(&full, od, sd, kd, nd);
        check_real(&got.to_vec().unwrap().iter().map(|&v| (f64::from(v), 0.0)).collect::<Vec<_>>(), &want.iter().map(|&v| v as f32).collect::<Vec<_>>(), 1e-4, &format!("{nd}-D same {sd:?} * {kd:?}"));
    }
    // A kernel with a batch dimension is rejected.
    let a = Array::from_slice_on(&ctx, &real_signal(16, 1), [4, 4]).unwrap();
    let b = Array::from_slice_on(&ctx, &real_signal(8, 2), [2, 2, 2]).unwrap();
    assert!(a.fft_convolve2(&b, ConvMode::Full).to_vec().is_err());
}

#[test]
fn large_real_transforms_unpack_in_the_last_pass() {
    let ctx = Context::new().unwrap();
    for &n in &[16384usize, 20000, 65536, 100000, 120000, 131072, 2 * 4099 * 3] {
        let x = real_signal(n, 17 + n as u64);
        let a = Array::from_slice_on(&ctx, &x, n).unwrap();
        let want = dft(&lift(&x), false);
        let half = a.rfft();
        check(&half.to_vec().unwrap(), &want[..n / 2 + 1], 1e-5, &format!("rfft {n}"));
        check_real(&lift(&half.irfft(n).to_vec().unwrap()), &x, 1e-5, &format!("irfft {n}"));
        // A batch of lines.
        let b = Array::from_slice_on(&ctx, &x.iter().chain(&x).copied().collect::<Vec<_>>(), [n, 2]).unwrap();
        let hb = b.rfft().to_vec().unwrap();
        check(&hb[n / 2 + 1..], &want[..n / 2 + 1], 1e-5, &format!("batched rfft {n}"));
    }
}

#[test]
fn non_finite_values_propagate_and_do_not_crash() {
    let ctx = Context::new().unwrap();
    for n in [8usize, 37, 1000] {
        let mut x = signal(n, 5);
        x[3] = C32::new(f32::NAN, 0.0);
        let got = Array::from_slice_on(&ctx, &x, n).unwrap().fft().to_vec().unwrap();
        assert!(got.iter().all(|c| c.re.is_nan() || c.im.is_nan()), "NaN reaches every frequency, n = {n}");
        let mut x = signal(n, 6);
        x[1] = C32::new(f32::INFINITY, 0.0);
        let got = Array::from_slice_on(&ctx, &x, n).unwrap().fft().to_vec().unwrap();
        assert!(got.iter().all(|c| !c.re.is_finite() || !c.im.is_finite()), "infinity reaches every frequency, n = {n}");
    }
}

#[test]
fn transforms_compose_with_complex_array_operations() {
    let ctx = Context::new().unwrap();
    let (n, k) = (96usize, 5usize);
    let x = real_signal(n, 8);
    let h = real_signal(k, 9);
    // The circular convolution by the convolution theorem, with lazy complex arithmetic between the transforms.
    let a = Array::from_slice_on(&ctx, &x, n).unwrap().to_complex();
    let mut padded = h.clone();
    padded.resize(n, 0.0);
    let b = Array::from_slice_on(&ctx, &padded, n).unwrap().to_complex();
    let got = (a.fft() * b.fft()).ifft().real().to_vec().unwrap();
    let want: Vec<f64> = (0..n).map(|i| (0..k).map(|j| f64::from(h[j]) * f64::from(x[(i + n - j) % n])).sum()).collect();
    for (g, w) in got.iter().zip(&want) {
        assert!((f64::from(*g) - w).abs() < 1e-4, "{g} against {w}");
    }
    // A lazily computed input.
    let y = (&a * C32::new(0.0, 2.0)).conj();
    let want = dft(&y.to_vec().unwrap().iter().map(|c| (f64::from(c.re), f64::from(c.im))).collect::<Vec<_>>(), false);
    check(&y.fft().to_vec().unwrap(), &want, 1e-5, "fft of a lazy expression");
}
