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
use crate::builtin::Builtin;
use crate::context::{RawArg, Scratch};
use crate::ops::{fold_workgroups, select, Outcome};
use crate::shape::Shape;
use crate::view::PadMode;
use crate::vision::{invalid, to_u32, Kernel as Any};
use crate::C32;

/// The kernels that make the half spectrum of the centred, zero padded point spread function
/// without transforming a whole padded array: `Cols` sums over the rows of the psf for every
/// frequency along axis 0, `Rows` over its columns for every frequency along axis 1.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum Kernel {
    Cols,
    Rows,
}

const GROUP: u32 = 256;
const PARAMS: &str = "struct Params { n0: u32, n1: u32, k0: u32, ph: u32, pw: u32, c0: u32, c1: u32, total: u32 }\n";

pub(crate) fn source(k: Kernel) -> String {
    let mut s = String::from(PARAMS);
    s.push_str(
        "fn tw(k: u32, d: i32, n: u32) -> vec2<f32> {
    let m = i32(n);
    var r = (i32(k % n) * (d % m)) % m;
    if (r < 0) { r += m; }
    let ang = -6.283185307179586 * f32(r) / f32(n);
    return vec2<f32>(cos(ang), sin(ang));
}
fn cmul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> { return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x); }
",
    );
    match k {
        Kernel::Cols => {
            s.push_str("@group(1) @binding(0) var<storage, read> psf: array<f32>;\n@group(1) @binding(1) var<storage, read_write> q: array<vec2<f32>>;\n@group(1) @binding(2) var<uniform> p: Params;\n");
            let _ = std::fmt::Write::write_fmt(
                &mut s,
                format_args!(
                    "@compute @workgroup_size({GROUP})
fn main(@builtin(local_invocation_index) t: u32, @builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {{
    let g = (wid.x + wid.y * nwg.x) * {GROUP}u + t;
    if (g >= p.total) {{ return; }}
    let k = g % p.k0;
    let b = g / p.k0;
    var acc = vec2<f32>(0.0);
    for (var a = 0u; a < p.ph; a++) {{
        acc += psf[a + p.ph * b] * tw(k, i32(a) - i32(p.c0), p.n0);
    }}
    q[g] = acc;
}}
"
                ),
            );
        }
        Kernel::Rows => {
            s.push_str("@group(1) @binding(0) var<storage, read> q: array<vec2<f32>>;\n@group(1) @binding(1) var<storage, read_write> out: array<vec2<f32>>;\n@group(1) @binding(2) var<uniform> p: Params;\n");
            let _ = std::fmt::Write::write_fmt(
                &mut s,
                format_args!(
                    "@compute @workgroup_size({GROUP})
fn main(@builtin(local_invocation_index) t: u32, @builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {{
    let g = (wid.x + wid.y * nwg.x) * {GROUP}u + t;
    if (g >= p.total) {{ return; }}
    let k = g % p.k0;
    let l = g / p.k0;
    var acc = vec2<f32>(0.0);
    for (var b = 0u; b < p.pw; b++) {{
        acc += cmul(q[k + p.k0 * b], tw(l, i32(b) - i32(p.c1), p.n1));
    }}
    out[g] = acc;
}}
"
                ),
            );
        }
    }
    s
}

/// The half spectrum (`n0 / 2 + 1` by `n1`) of `psf` centred at `(ph / 2, pw / 2)` and zero padded
/// to `n0 x n1`.
fn psf_spectrum(psf: &Array<f32>, n: [usize; 2]) -> Outcome<Array<C32>> {
    let ctx = psf.context();
    let d = psf.shape().dims();
    let k0 = n[0] / 2 + 1;
    let out = Array::<C32>::alloc(ctx, Shape::from([k0, n[1]]))?;
    let (src, src_bytes) = psf.ready()?;
    let (out_pin, out_bytes) = out.ready()?;
    let mut scratch = Scratch::new(ctx);
    let q_bytes = (k0 * d[1] * 8) as u64;
    let q = scratch.take(q_bytes)?;
    let run = |kernel: Kernel, total: usize, args: &[RawArg<'_>]| -> Outcome<()> {
        let params = [to_u32(n[0], "size")?, to_u32(n[1], "size")?, to_u32(k0, "size")?, d[0] as u32, d[1] as u32, (d[0] / 2) as u32, (d[1] / 2) as u32, to_u32(total, "size")?];
        let mut all = args.to_vec();
        all.push(RawArg::Uniform(bytemuck::bytes_of(&params)));
        let kernel_core = ctx.builtin(Builtin::Vision(Any::Regions(super::Kernel::Deconv(kernel))), "f32")?;
        let groups = fold_workgroups(ctx, (total as u32).div_ceil(GROUP))?;
        ctx.launch(&kernel_core, groups, [total as u32, 1, 1], &all)?;
        Ok(())
    };
    run(Kernel::Cols, k0 * d[1], &[RawArg::read(src.buffer(), src_bytes), RawArg::write(&q, q_bytes)])?;
    run(Kernel::Rows, k0 * n[1], &[RawArg::read(&q, q_bytes), RawArg::write(out_pin.buffer(), out_bytes)])?;
    Ok(out)
}

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
        psf.storage()?;
        let offset = [(size[0] - d[0]) / 2, (size[1] - d[1]) / 2];
        let padded = image.pad([(offset[0], size[0] - d[0] - offset[0]), (offset[1], size[1] - d[1] - offset[1])], PadMode::Edge);
        Ok(Padded { image: padded, psf: psf_spectrum(psf, size)?, size, offset, extent: [d[0], d[1]] })
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

