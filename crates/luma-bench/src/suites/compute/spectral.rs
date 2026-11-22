//! Fourier transforms: complex and real, 1-D to 3-D, power-of-two and awkward lengths.
//!
//! Every case checks a handful of output bins against the direct `f64` sum before it is timed.

use super::pick;
use crate::harness::*;
use luma::*;

/// The `f64` value of bin `k` (a multi-index in `dims` order, x fastest) of the transform of `x`.
fn bin(x: &[C32], dims: &[usize], k: &[usize], inverse: bool) -> (f64, f64) {
    let sign = if inverse { 1.0 } else { -1.0 };
    let (mut re, mut im) = (0.0, 0.0);
    for (i, v) in x.iter().enumerate() {
        // The phase of the multi-index of element `i`, kept as an exact fraction per axis.
        let (mut rest, mut phase) = (i, 0.0f64);
        for (&n, &kk) in dims.iter().zip(k) {
            let c = rest % n;
            rest /= n;
            phase += ((c as u64 * kk as u64) % n as u64) as f64 / n as f64;
        }
        let a = sign * 2.0 * std::f64::consts::PI * phase.fract();
        let (s, c) = a.sin_cos();
        re += f64::from(v.re) * c - f64::from(v.im) * s;
        im += f64::from(v.re) * s + f64::from(v.im) * c;
    }
    (re, im)
}

fn signal(n: usize, seed: u64) -> Vec<C32> {
    let (re, im) = (values(n, seed, -1.0, 1.0), values(n, seed + 1, -1.0, 1.0));
    re.iter().zip(&im).map(|(&re, &im)| C32::new(re, im)).collect()
}

/// Checks `bins` of `got` (an array of `dims`) against the direct sums over `x`.
fn verify(name: &str, got: &Array<C32>, x: &[C32], dims: &[usize], inverse: bool, bins: &[Vec<usize>]) {
    let host = got.to_vec().unwrap();
    let rms = (host.iter().map(|c| f64::from(c.re).powi(2) + f64::from(c.im).powi(2)).sum::<f64>() / host.len() as f64).sqrt();
    for k in bins {
        let at: usize = k.iter().zip(dims).scan(1usize, |stride, (&c, &n)| { let o = c * *stride; *stride *= n; Some(o) }).sum();
        let (re, im) = bin(x, dims, k, inverse);
        let (sr, si) = if inverse { (1.0 / x.len() as f64, 1.0 / x.len() as f64) } else { (1.0, 1.0) };
        let (re, im) = (re * sr, im * si);
        let e = ((f64::from(host[at].re) - re).powi(2) + (f64::from(host[at].im) - im).powi(2)).sqrt();
        let scale = if inverse { rms.max(1e-30) } else { rms };
        assert!(e <= 2e-5 * scale * (x.len() as f64).log2().max(1.0), "{name}: bin {k:?}: got {:?}, want ({re}, {im}), error {e:e} against rms {scale:e}", host[at]);
    }
}

pub fn run(h: &mut Harness) {
    h.guarded("spectral", |h| {
        let cpu = h.ctx.is_cpu();
        let c2c: Vec<usize> = if h.quick || cpu { vec![1 << 16, 1 << 20, 100_003] } else { vec![1 << 20, 1 << 24, 1_000_003, 100_000, 1 << 22] };
        for n in c2c {
            let x = signal(n, 41);
            let a = Array::from_slice_on(&h.ctx, &x, n).unwrap();
            a.eval().unwrap();
            let bins: Vec<Vec<usize>> = [0, 1, n / 3, n / 2, n - 1].iter().map(|&k| vec![k]).collect();
            let f = a.fft();
            verify("spectral/fft", &f, &x, &[n], false, &bins);
            h.case("spectral/fft", &format!("{n}"), Work::Items(n as f64, "Mpoints/s"), &mut || a.fft().eval().unwrap());
            let fx = f.to_vec().unwrap();
            let f = Array::from_slice_on(&h.ctx, &fx, n).unwrap();
            f.eval().unwrap();
            verify("spectral/ifft", &f.ifft(), &fx, &[n], true, &bins);
            h.case("spectral/ifft", &format!("{n}"), Work::Items(n as f64, "Mpoints/s"), &mut || f.ifft().eval().unwrap());
            // Real input.
            let r = values(n, 43, -1.0, 1.0);
            let ra = Array::from_slice_on(&h.ctx, &r, n).unwrap();
            ra.eval().unwrap();
            let rc: Vec<C32> = r.iter().map(|&v| C32::new(v, 0.0)).collect();
            let half = ra.rfft();
            assert_eq!(half.len(), n / 2 + 1);
            let half_bins: Vec<Vec<usize>> = bins.iter().map(|b| vec![b[0].min(n / 2)]).collect();
            verify("spectral/rfft", &half, &rc, &[n], false, &half_bins);
            h.case("spectral/rfft", &format!("{n}"), Work::Items(n as f64, "Mpoints/s"), &mut || ra.rfft().eval().unwrap());
        }
        let side = pick(h, 256usize, 2048);
        let x = signal(side * side, 45);
        let a = Array::from_slice_on(&h.ctx, &x, [side, side]).unwrap();
        a.eval().unwrap();
        let bins = vec![vec![0, 0], vec![1, 0], vec![0, 3], vec![side / 3, side / 2], vec![side - 1, side - 2]];
        verify("spectral/fft2", &a.fft2(), &x, &[side, side], false, &bins);
        h.case("spectral/fft2", &format!("{side}x{side}"), Work::Items((side * side) as f64, "Mpoints/s"), &mut || a.fft2().eval().unwrap());
        h.case("spectral/ifft2", &format!("{side}x{side}"), Work::Items((side * side) as f64, "Mpoints/s"), &mut || a.ifft2().eval().unwrap());
        let cube = pick(h, 64usize, 256);
        let x = signal(cube * cube * cube, 47);
        let a = Array::from_slice_on(&h.ctx, &x, [cube, cube, cube]).unwrap();
        a.eval().unwrap();
        let bins = vec![vec![0, 0, 0], vec![1, 2, 3], vec![cube - 1, cube / 2, 1]];
        verify("spectral/fft3", &a.fft3(), &x, &[cube, cube, cube], false, &bins);
        h.case("spectral/fft3", &format!("{cube}^3"), Work::Items((cube * cube * cube) as f64, "Mpoints/s"), &mut || a.fft3().eval().unwrap());
    });
}
