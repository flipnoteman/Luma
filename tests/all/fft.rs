//! Fourier transforms against an `f64` host reference.

use crate::common::values;
use luma::{Array, Context, FftNorm, C32};

type Z = (f64, f64);

/// The DFT of `x` in `f64`: mixed radix on the smallest prime factor, the naive sum for a prime.
fn dft(x: &[Z], inverse: bool) -> Vec<Z> {
    let n = x.len();
    if n <= 1 {
        return x.to_vec();
    }
    let p = (2..=n).find(|p| n % p == 0).unwrap();
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
        if !(e <= worst) {
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
        if (i / stride) % n != 0 {
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
}
