//! Deconvolution with a known point spread function (`Array<f32>::iterative_deconv`,
//! `Array<f32>::inverse_deconv`), ArrayFire's `iterativeDeconv` and `inverseDeconv`.
//!
//! Both work on the image padded to a size `N0 x N1` (`H + PH` and `W + PW` rounded up to a size
//! whose prime factors are all at most 7) with the nearest pixel, the point spread function
//! padded with zeros to the same size and rolled so that its centre `(PH / 2, PW / 2)` is the
//! origin: a circular convolution of the padded image is then the convolution of the image with
//! the replicated border. The transforms are the half-spectrum real FFTs of [`crate::spectral`].
//!
//! Landweber's iteration `x += a P^T (y - P x)` is linear and diagonal in the frequency domain
//! (`X += a conj(P) (Y - P X)` per frequency), so it needs one forward and one inverse
//! transform for any number of iterations. Richardson-Lucy needs two forward and two inverse
//! transforms per iteration.

use crate::array::Array;
use crate::ops::{select, Outcome};
use crate::vision::invalid;
use crate::view::PadMode;

/// The iterative deconvolution algorithms (ArrayFire's `af_iterative_deconv_algo`).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub enum IterativeDeconvAlgo {
    /// Landweber: `x_n = x_{n-1} + a P^T (y - P x_{n-1})` (ArrayFire's default).
    #[default]
    Landweber,
    /// Richardson-Lucy: `x_n = x_{n-1} (P^T (y / (P x_{n-1})))`.
    RichardsonLucy,
}

/// The inverse deconvolution algorithms (ArrayFire's `af_inverse_deconv_algo`).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub enum InverseDeconvAlgo {
    /// Tikhonov: `X = Y conj(P) / (|P|^2 + gamma)` per frequency (ArrayFire's default).
    #[default]
    Tikhonov,
}

/// The smallest size at least `n` whose prime factors are all at most 7 (ArrayFire's CUDA rule;
/// its CPU backend allows factors up to 13).
fn friendly(mut n: usize) -> usize {
    loop {
        let mut m = n;
        for p in [2, 3, 5, 7] {
            while m.is_multiple_of(p) {
                m /= p;
            }
        }
        if m == 1 {
            return n;
        }
        n += 1;
    }
}

/// The padded image, the transform of the centred point spread function, the padded size and the
/// offset of the image in it.
struct Padded {
    image: Array<f32>,
    psf: Array<crate::C32>,
    size: [usize; 2],
    offset: [usize; 2],
    extent: [usize; 2],
}

impl Padded {
    fn new(image: &Array<f32>, psf: &Array<f32>, op: &str) -> Outcome<Padded> {
        let (d, p) = (image.shape().dims(), psf.shape().dims());
        if d[2] != 1 || d[3] != 1 || p[2] != 1 || p[3] != 1 {
            return Err(invalid(format!("{op} takes an [H, W] image and an [PH, PW] point spread function, got shapes {d:?} and {p:?}")));
        }
        let size = [friendly(d[0] + p[0]), friendly(d[1] + p[1])];
        let offset = [(size[0] - d[0]) / 2, (size[1] - d[1]) / 2];
        let padded = image.pad([(offset[0], size[0] - d[0] - offset[0]), (offset[1], size[1] - d[1] - offset[1])], PadMode::Edge);
        let centred = psf.pad([(0, size[0] - p[0]), (0, size[1] - p[1])], PadMode::Zero).shift([-((p[0] / 2) as isize), -((p[1] / 2) as isize)]);
        Ok(Padded { image: padded, psf: centred.rfft2(), size, offset, extent: [d[0], d[1]] })
    }

    /// The part of a padded array that is the image.
    fn crop(&self, padded: &Array<f32>) -> Array<f32> {
        padded.slice(0, self.offset[0]..self.offset[0] + self.extent[0]).slice(1, self.offset[1]..self.offset[1] + self.extent[1])
    }
}

pub(super) fn iterative(image: &Array<f32>, psf: &Array<f32>, iterations: u32, relaxation: f32, algo: IterativeDeconvAlgo) -> Outcome<Array<f32>> {
    if iterations == 0 || relaxation.is_nan() || relaxation <= 0.0 {
        return Err(invalid(format!("iterative_deconv needs iterations > 0 and relaxation > 0, got {iterations} and {relaxation}")));
    }
    let p = Padded::new(image, psf, "iterative_deconv")?;
    let n0 = p.size[0];
    let estimate = match algo {
        IterativeDeconvAlgo::Landweber => {
            let y = p.image.rfft2();
            let drive = &y * &p.psf.conj();
            let gain = p.psf.norm_sqr().to_complex();
            let mut x = y;
            for _ in 0..iterations {
                x = &x + &((&drive - &(&x * &gain)) * relaxation);
            }
            x.irfft2(n0)
        }
        IterativeDeconvAlgo::RichardsonLucy => {
            let conj = p.psf.conj();
            let mut x = p.image.clone();
            for _ in 0..iterations {
                let blurred = (&x.rfft2() * &p.psf).irfft2(n0);
                // Where the blurred estimate is not positive the ratio is 0 (ArrayFire's is inf or NaN,
                // which spreads over the whole image through the transform).
                let ratio = select(&blurred.gt(0.0), &p.image / &blurred, 0.0);
                x = &x * &(&ratio.rfft2() * &conj).irfft2(n0);
            }
            x
        }
    };
    Ok(p.crop(&estimate))
}

pub(super) fn inverse(image: &Array<f32>, psf: &Array<f32>, gamma: f32, algo: InverseDeconvAlgo) -> Outcome<Array<f32>> {
    if gamma.is_nan() || gamma <= 0.0 {
        return Err(invalid(format!("inverse_deconv needs gamma > 0, got {gamma}")));
    }
    let p = Padded::new(image, psf, "inverse_deconv")?;
    let estimate = match algo {
        InverseDeconvAlgo::Tikhonov => {
            let gain = (1.0f32 / (p.psf.norm_sqr() + gamma)).to_complex();
            (&(&p.image.rfft2() * &p.psf.conj()) * &gain).irfft2(p.size[0])
        }
    };
    Ok(p.crop(&estimate))
}

