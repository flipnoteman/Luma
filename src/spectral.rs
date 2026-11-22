//! Fast Fourier transforms of any length along any axis: [`Array::fft`] and friends.
//!
//! A transform is a few passes of the kernels in [`crate::dft_kernel`], each applying the DFT of
//! a factor of the length to tiles in workgroup memory (see there for the algorithm). Lengths
//! whose prime factors are all at most [`MAX_RADIX`] are transformed directly, the others with
//! Bluestein's algorithm (a circular convolution of a power-of-two length, itself made of the
//! same passes). Real transforms of even length are half-size complex transforms with a
//! pass that unpacks (or packs) the spectrum.

use std::panic::Location;

use crate::array::Array;
use crate::builtin::Builtin;
use crate::complex::C32;
use crate::context::{Context, RawArg, Scratch};
use crate::dft_kernel::{DftPass, LoadKind, MAX_RADIX, MAX_STAGES, Params, StoreKind, Table, WORKGROUP};
use crate::error::LumaError;
use crate::ops::{fold_workgroups, Outcome};

/// How a transform is scaled.
///
/// The default of every method that does not take one is [`FftNorm::Backward`], the convention
/// of NumPy, FFTW and ArrayFire's `ifft`: forward transforms are not scaled and inverse
/// transforms are divided by the length.
#[derive(Clone, Copy, PartialEq, Debug, Default)]
pub enum FftNorm {
    /// The forward transform is unscaled, the inverse is divided by the number of points.
    #[default]
    Backward,
    /// Both directions are divided by the square root of the number of points: the transform is
    /// unitary.
    Ortho,
    /// The forward transform is divided by the number of points, the inverse is unscaled.
    Forward,
    /// Neither direction is scaled: an inverse transform of a forward one gives the signal times
    /// the number of points.
    None,
    /// Every transform is multiplied by this factor (once, however many axes it has).
    Scale(f32),
}

impl FftNorm {
    /// The factor of a transform of `n` points in the given direction.
    fn factor(self, n: usize, inverse: bool) -> f32 {
        let n = n as f64;
        (match self {
            FftNorm::Backward => {
                if inverse {
                    1.0 / n
                } else {
                    1.0
                }
            }
            FftNorm::Forward => {
                if inverse {
                    1.0
                } else {
                    1.0 / n
                }
            }
            FftNorm::Ortho => 1.0 / n.sqrt(),
            FftNorm::None => 1.0,
            FftNorm::Scale(s) => f64::from(s),
        }) as f32
    }
}

/// How an FFT convolution fits its output to the inputs.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum ConvMode {
    /// The output has the shape of the signal (the middle of the full convolution).
    #[default]
    Same,
    /// The full linear convolution: every dimension is `signal + kernel - 1` long.
    Full,
}

/// The elements of workgroup memory a pass may use as its tile (a power of two).
fn tile_elems(ctx: &Context) -> u32 {
    let bytes = u64::from(ctx.info().limits.max_workgroup_storage_bytes);
    [4096u32, 2048, 1024, 512, 256]
        .into_iter()
        .find(|&t| (u64::from(t) + u64::from(t / 16) + 2) * 8 <= bytes)
        .unwrap_or(256)
}

fn prime_factors(mut n: u32) -> Vec<u32> {
    let mut out = Vec::new();
    let mut p = 2;
    while p * p <= n {
        while n % p == 0 {
            out.push(p);
            n /= p;
        }
        p += 1;
    }
    if n > 1 {
        out.push(n);
    }
    out
}

/// Whether `n` is a product of primes the passes have a butterfly for.
fn is_smooth(n: u32) -> bool {
    prime_factors(n).iter().all(|&p| p <= MAX_RADIX)
}

/// The radices of a pass of `d` points: the power of two as radices of at most 16 (evenly), then
/// every other prime on its own; the largest first.
fn radices_of(d: u32) -> Vec<u8> {
    let factors = prime_factors(d);
    let twos = factors.iter().filter(|&&p| p == 2).count() as u32;
    let mut radices: Vec<u8> = factors.iter().filter(|&&p| p != 2).map(|&p| p as u8).collect();
    if twos > 0 {
        let parts = twos.div_ceil(4);
        for i in 0..parts {
            let bits = twos / parts + u32::from(i < twos % parts);
            radices.push(1u8 << bits);
        }
    }
    radices.sort_unstable_by(|a, b| b.cmp(a));
    radices
}

/// The sizes of the passes of an `n`-point transform: as few as the tile allows, balanced, the
/// largest first.
fn partition(n: u32, tile: u32) -> Vec<u32> {
    if n == 1 {
        return vec![1];
    }
    let mut factors = prime_factors(n);
    factors.sort_unstable_by(|a, b| b.cmp(a));
    let min_passes = ((f64::from(n).ln() / f64::from(tile).ln()).ceil() as usize).max(1);
    let mut best: Option<(f64, Vec<u32>)> = None;
    for passes in min_passes..=min_passes + 1 {
        let mut bins = vec![1u32; passes];
        let mut fits = true;
        for &f in &factors {
            let bin = bins.iter_mut().min_by_key(|b| **b).expect("at least one pass");
            *bin = bin.saturating_mul(f);
            fits &= *bin <= tile;
        }
        if !fits {
            continue;
        }
        bins.retain(|&b| b > 1);
        // Bytes moved: a pass of `d` points keeps `tile / d` columns, whose runs of 8 bytes each
        // should fill a 32-byte sector.
        let cost: f64 = bins.iter().map(|&d| 1.0 / (f64::from((tile / d).max(1)) * 8.0 / 32.0).min(1.0)).sum();
        if best.as_ref().is_none_or(|(c, _)| cost < *c) {
            bins.sort_unstable_by(|a, b| b.cmp(a));
            best = Some((cost, bins));
        }
    }
    best.map(|(_, bins)| bins).unwrap_or_else(|| vec![n])
}

/// The largest divisor of `n` that is at most `max`.
fn divisor_up_to(n: u32, max: u32) -> u32 {
    (1..=max.min(n)).rev().find(|d| n % d == 0).unwrap_or(1)
}

/// A batch of transforms along the middle axis of `[outer][n][inner]` (the first array
/// dimension is the fastest, so `inner` is the product of the dimensions before the axis).
pub(crate) struct Lines<'a> {
    pub(crate) outer: u32,
    pub(crate) n: u32,
    pub(crate) inner: u32,
    pub(crate) src: (&'a luma_core::Buffer, u64),
    pub(crate) dst: (&'a luma_core::Buffer, u64),
    pub(crate) inverse: bool,
    pub(crate) scale: f32,
    pub(crate) load: LoadKind,
    pub(crate) load_table: Option<(&'a luma_core::Buffer, u64, Table)>,
    /// The line length of the input if it is not `n` (for a Hermitian load, the half
    /// spectrum's length).
    pub(crate) n_in: Option<u32>,
    /// Loads of positions from here on are zero.
    pub(crate) n_lim: Option<u32>,
    pub(crate) n_full: u32,
    pub(crate) store: StoreKind,
    pub(crate) store_table: Option<(&'a luma_core::Buffer, u64, Table)>,
    /// The line length of the output if it is not `n`: positions beyond are not written.
    pub(crate) n_out: Option<u32>,
}

fn too_large(n: usize) -> LumaError {
    LumaError::DimensionTooLarge { shape: [n, 1, 1, 1] }
}

/// Records the passes of the transform `job`.
pub(crate) fn run_lines(ctx: &Context, scratch: &mut Scratch<'_>, job: &Lines<'_>) -> Result<(), LumaError> {
    let (outer, n, inner) = (job.outer, job.n, job.inner);
    let total = u64::from(outer) * u64::from(n) * u64::from(inner);
    if total > u64::from(u32::MAX) / 2 || n >= 1 << 28 {
        return Err(too_large(total as usize));
    }
    let tile = tile_elems(ctx);
    let sizes = partition(n, tile);
    let passes = sizes.len();
    let line_bytes = total * 8;
    // The buffers the intermediate results go to, alternating; the output itself is one of them
    // when it is a plain complex array.
    let reuse_out = job.store == StoreKind::Complex && job.n_out.is_none();
    let temp_a = if passes > 1 { Some(scratch.take(line_bytes)?) } else { None };
    let temp_b = if passes > 2 && !reuse_out { Some(scratch.take(line_bytes)?) } else { None };
    let mut mids: Vec<(&luma_core::Buffer, u64)> = Vec::new();
    for i in 0..passes.saturating_sub(1) {
        // Pass `i`'s result, counted back from the last pass.
        let back = passes - 2 - i;
        let buffer = if back % 2 == 0 {
            temp_a.as_ref().expect("temporary exists")
        } else if reuse_out {
            job.dst.0
        } else {
            temp_b.as_ref().expect("second temporary exists")
        };
        mids.push((buffer, line_bytes));
    }
    let mut low = inner;
    for (i, &d) in sizes.iter().enumerate() {
        let first = i == 0;
        let last = i + 1 == passes;
        let mid = n / (low / inner) / d;
        let j = mid * low;
        let rows = outer * j;
        let cmax = (tile / d).max(1);
        let cols = if j == 1 {
            cmax.min(outer)
        } else if low > 1 {
            divisor_up_to(low, cmax)
        } else {
            divisor_up_to(mid, cmax)
        };
        let mut radices = [0u8; MAX_STAGES];
        let list = radices_of(d);
        radices[..list.len()].copy_from_slice(&list);
        let in_os = if first && (job.n_in.is_some() || job.load == LoadKind::Hermitian) { job.n_in.unwrap_or(n) * inner } else { n * inner };
        let out_os = if last { job.n_out.unwrap_or(n) * inner } else { n * inner };
        let pass = DftPass {
            radices,
            stages: list.len() as u8,
            cols,
            in_fast: j > 1,
            out_fast: low > 1,
            twiddle: mid > 1,
            load: if first { job.load } else { LoadKind::Complex },
            load_table: if first { job.load_table.map_or(Table::None, |t| t.2) } else { Table::None },
            load_limit: first && job.n_lim.is_some(),
            store: if last { job.store } else { StoreKind::Complex },
            store_table: if last { job.store_table.map_or(Table::None, |t| t.2) } else { Table::None },
            store_limit: last && job.n_out.is_some(),
        };
        let out_rs = if low > 1 {
            1
        } else if mid > 1 {
            d
        } else {
            out_os
        };
        let params = Params {
            tiles: rows.div_ceil(cols),
            rows,
            j,
            low,
            mid,
            in_rs: if j == 1 { in_os } else { 1 },
            in_ds: j,
            in_os,
            out_rs,
            out_ds: if low > 1 { low } else { 1 },
            out_os,
            den: d.checked_mul(mid).ok_or_else(|| too_large(n as usize))?,
            inner,
            prod: low / inner,
            n_in: job.n_in.unwrap_or(n),
            n_lim: job.n_lim.unwrap_or(n),
            n_out: job.n_out.unwrap_or(n),
            n_full: job.n_full,
            scale: if last { job.scale } else { 1.0 },
            cj: if job.inverse { -1.0 } else { 1.0 },
            mrs: u32::from(low == 1 && mid > 1),
            pad: [0; 3],
        };
        let kernel = ctx.builtin(Builtin::Dft(pass), "f32")?;
        let src = if first { job.src } else { mids[i - 1] };
        let dst = if last { job.dst } else { mids[i] };
        let mut args = vec![RawArg::read(src.0, src.1), RawArg::write(dst.0, dst.1)];
        if let (true, Some((b, bytes, _))) = (first, job.load_table) {
            args.push(RawArg::read(b, bytes));
        }
        if let (true, Some((b, bytes, _))) = (last, job.store_table) {
            args.push(RawArg::read(b, bytes));
        }
        args.push(RawArg::Uniform(bytemuck::bytes_of(&params)));
        let groups = fold_workgroups(ctx, params.tiles)?;
        ctx.launch(&kernel, groups, [params.tiles.saturating_mul(WORKGROUP), 1, 1], &args)?;
        low *= d;
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// Small kernels: the real-transform packing and the chirps of Bluestein's algorithm.
// ---------------------------------------------------------------------------------------------

const TW_FUNCTION: &str = "
// e^(-2 pi i num / den) for num < den < 2^29: the angle is reduced to an octant with integers,
// so the polynomials see at most pi / 4.
fn tw(num: u32, den: u32) -> vec2<f32> {
    let a = 8u * num;
    let oct = a / den;
    let rem = a - oct * den;
    let odd = (oct & 1u) == 1u;
    let x = select(rem, den - rem, odd);
    let th = (f32(x) / f32(den)) * 0.7853981633974483;
    let t2 = th * th;
    let sn = th * (1.0 + t2 * (-0.16666667 + t2 * (0.008333334 + t2 * (-0.0001984127 + t2 * 0.0000027557319))));
    let cs = 1.0 + t2 * (-0.5 + t2 * (0.041666668 + t2 * (-0.0013888889 + t2 * (0.0000248015873 + t2 * -0.00000027557319))));
    let cp = select(cs, sn, odd);
    let sp = select(sn, cs, odd);
    let quad = oct >> 1u;
    var c = cp;
    var s = sp;
    if (quad == 1u) { c = -sp; s = cp; }
    if (quad == 2u) { c = -cp; s = -sp; }
    if (quad == 3u) { c = sp; s = -cp; }
    return vec2<f32>(c, -s);
}
fn cmul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}
";

/// The kernel of [`Builtin::RealStage`]: the step between a half-size complex transform and the
/// spectrum of a real signal (`forward`: unpack `h + 1` frequencies from `h` values; else pack
/// them, doubled, for the inverse). Bindings: `src`, `dst`, `Params { total, h, inner, scale }`.
pub(crate) fn real_stage_source(forward: bool) -> String {
    let body = if forward {
        // Z[k mod h] and conj(Z[(h - k) mod h]) give the even and odd samples' spectra E and O;
        // X[k] = E + W^k O.
        "let total = p.outer * (p.h + 1u) * p.inner;
    if (i >= total) { return; }
    let ii = i % p.inner;
    let k = (i / p.inner) % (p.h + 1u);
    let o = i / (p.inner * (p.h + 1u));
    let base = o * p.h * p.inner + ii;
    let zk = src[base + (k % p.h) * p.inner];
    let z2 = src[base + ((p.h - k) % p.h) * p.inner];
    let zc = vec2<f32>(z2.x, -z2.y);
    let e = (zk + zc) * 0.5;
    let d = zk - zc;
    let od = vec2<f32>(d.y, -d.x) * 0.5;
    dst[i] = (e + cmul(tw(k, 2u * p.h), od)) * p.scale;"
    } else {
        "let total = p.outer * p.h * p.inner;
    if (i >= total) { return; }
    let ii = i % p.inner;
    let k = (i / p.inner) % p.h;
    let o = i / (p.inner * p.h);
    let base = o * (p.h + 1u) * p.inner + ii;
    let xk = src[base + k * p.inner];
    let x2 = src[base + (p.h - k) * p.inner];
    let xc = vec2<f32>(x2.x, -x2.y);
    let e = xk + xc;
    let d = xk - xc;
    let w = tw(k, 2u * p.h);
    let od = cmul(d, vec2<f32>(w.x, -w.y));
    dst[i] = (e + vec2<f32>(-od.y, od.x)) * p.scale;"
    };
    format!(
        "struct Params {{ outer: u32, h: u32, inner: u32, scale: f32 }}
@group(1) @binding(0) var<storage, read> src: array<vec2<f32>>;
@group(1) @binding(1) var<storage, read_write> dst: array<vec2<f32>>;
@group(1) @binding(2) var<uniform> p: Params;
{TW_FUNCTION}
@compute @workgroup_size({WORKGROUP})
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {{
    let i = gid.x + gid.y * nwg.x * {WORKGROUP}u;
    {body}
}}
"
    )
}

/// The kernel of [`Builtin::Chirp`]: the chirp `w[i] = e^(sign pi i^2 / n)` for `i < n` and the
/// zero-padded, wrapped `conj(w)` of length `m` that Bluestein's algorithm convolves with.
/// Bindings: `chirp` (`n` values), `padded` (`m` values), `Params { n, m, sign }`.
pub(crate) fn chirp_source() -> String {
    format!(
        "struct Params {{ n: u32, m: u32, sign: f32, pad: u32 }}
@group(1) @binding(0) var<storage, read_write> chirp: array<vec2<f32>>;
@group(1) @binding(1) var<storage, read_write> padded: array<vec2<f32>>;
@group(1) @binding(2) var<uniform> p: Params;
{TW_FUNCTION}
// i^2 mod md without overflow: shift and add.
fn sqmod(i: u32, md: u32) -> u32 {{
    var acc = 0u;
    var a = i % md;
    var b = i;
    for (var k = 0u; k < 28u; k = k + 1u) {{
        if ((b & 1u) == 1u) {{ acc = acc + a; if (acc >= md) {{ acc = acc - md; }} }}
        a = a + a;
        if (a >= md) {{ a = a - md; }}
        b = b >> 1u;
    }}
    return acc;
}}
// e^(sign pi i^2 / n) = e^(-2 pi i (sign * -1) c / 2n) with c = i^2 mod 2n.
fn w_at(i: u32) -> vec2<f32> {{
    let t = tw(sqmod(i, 2u * p.n), 2u * p.n);
    return vec2<f32>(t.x, t.y * -p.sign);
}}
@compute @workgroup_size({WORKGROUP})
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {{
    let i = gid.x + gid.y * nwg.x * {WORKGROUP}u;
    if (i >= p.m) {{ return; }}
    var b = vec2<f32>(0.0, 0.0);
    if (i < p.n) {{
        let w = w_at(i);
        chirp[i] = w;
        b = vec2<f32>(w.x, -w.y);
    }} else if (i > p.m - p.n) {{
        let w = w_at(p.m - i);
        b = vec2<f32>(w.x, -w.y);
    }}
    padded[i] = b;
}}
"
    )
}

// ---------------------------------------------------------------------------------------------
// Transforms of whole arrays.
// ---------------------------------------------------------------------------------------------

/// The geometry of a transform along `axis` of an array of dimensions `dims`.
fn geometry(dims: [usize; 4], axis: usize) -> Result<(u32, u32, u32), LumaError> {
    let to = |v: usize| u32::try_from(v).map_err(|_| too_large(v));
    Ok((to(dims[axis + 1..].iter().product())?, to(dims[axis])?, to(dims[..axis].iter().product())?))
}

fn bytes_of(dims: [usize; 4]) -> u64 {
    dims.iter().product::<usize>() as u64 * 8
}

/// The smallest power of two of at least `n`.
fn pow2_at_least(n: u32) -> Result<u32, LumaError> {
    n.checked_next_power_of_two().filter(|&m| m < 1 << 28).ok_or_else(|| too_large(n as usize))
}

/// The input and output of a transform along one axis.
pub(crate) struct AxisIo<'a> {
    pub(crate) src: (&'a luma_core::Buffer, u64),
    pub(crate) dst: (&'a luma_core::Buffer, u64),
    pub(crate) load: LoadKind,
    pub(crate) store: StoreKind,
    /// `n_in` / `n_out` of [`Lines`]: line lengths of the input and output that are not `n`.
    pub(crate) n_in: Option<u32>,
    pub(crate) n_out: Option<u32>,
    pub(crate) n_full: u32,
    pub(crate) store_table: Option<(&'a luma_core::Buffer, u64, Table)>,
}

/// Records the transform of `n` points along the middle axis of `[outer][n][inner]`, by the
/// direct passes if the length factors into small primes and by Bluestein's algorithm otherwise.
pub(crate) fn run_axis(
    ctx: &Context,
    scratch: &mut Scratch<'_>,
    (outer, n, inner): (u32, u32, u32),
    inverse: bool,
    scale: f32,
    io: &AxisIo<'_>,
) -> Result<(), LumaError> {
    if is_smooth(n) {
        return run_lines(
            ctx,
            scratch,
            &Lines {
                outer,
                n,
                inner,
                src: io.src,
                dst: io.dst,
                inverse,
                scale,
                load: io.load,
                load_table: None,
                n_in: io.n_in,
                n_lim: None,
                n_full: io.n_full,
                store: io.store,
                store_table: io.store_table,
                n_out: io.n_out,
            },
        );
    }
    // Bluestein: X[k] = w[k] sum_j (x[j] w[j]) conj(w)[k - j], a circular convolution of
    // length m >= 2n - 1.
    if io.load == LoadKind::RealPair || io.store == StoreKind::RealPair || io.store_table.is_some() {
        return Err(LumaError::InvalidShape("internal: Bluestein with packed or tabled ends".into()));
    }
    let m = pow2_at_least(2 * n - 1)?;
    let chirp = scratch.take(u64::from(n) * 8)?;
    let padded = scratch.take(u64::from(m) * 8)?;
    {
        let params = [n, m, (if inverse { 1.0f32 } else { -1.0 }).to_bits(), 0];
        let kernel = ctx.builtin(Builtin::Chirp, "f32")?;
        let groups = fold_workgroups(ctx, m.div_ceil(WORKGROUP))?;
        ctx.launch(
            &kernel,
            groups,
            [m, 1, 1],
            &[RawArg::write(&chirp, u64::from(n) * 8), RawArg::write(&padded, u64::from(m) * 8), RawArg::Uniform(bytemuck::bytes_of(&params))],
        )?;
    }
    let size_bytes = |len: u32| u64::from(outer) * u64::from(len) * u64::from(inner) * 8;
    // The spectrum of the padded conj(chirp): one line, shared by every batch.
    let b_hat = scratch.take(u64::from(m) * 8)?;
    run_lines(
        ctx,
        scratch,
        &Lines {
            outer: 1,
            n: m,
            inner: 1,
            src: (&padded, u64::from(m) * 8),
            dst: (&b_hat, u64::from(m) * 8),
            inverse: false,
            scale: 1.0,
            load: LoadKind::Complex,
            load_table: None,
            n_in: None,
            n_lim: None,
            n_full: 0,
            store: StoreKind::Complex,
            store_table: None,
            n_out: None,
        },
    )?;
    // Spectrum of the weighted, zero-padded signal times the chirp's.
    let spectrum = scratch.take(size_bytes(m))?;
    run_lines(
        ctx,
        scratch,
        &Lines {
            outer,
            n: m,
            inner,
            src: io.src,
            dst: (&spectrum, size_bytes(m)),
            inverse: false,
            scale: 1.0,
            load: io.load,
            load_table: Some((&chirp, u64::from(n) * 8, Table::Line)),
            n_in: Some(io.n_in.unwrap_or(n)),
            n_lim: Some(n),
            n_full: io.n_full,
            store: StoreKind::Complex,
            store_table: Some((&b_hat, u64::from(m) * 8, Table::Line)),
            n_out: None,
        },
    )?;
    run_lines(
        ctx,
        scratch,
        &Lines {
            outer,
            n: m,
            inner,
            src: (&spectrum, size_bytes(m)),
            dst: io.dst,
            inverse: true,
            scale: scale / m as f32,
            load: LoadKind::Complex,
            load_table: None,
            n_in: None,
            n_lim: None,
            n_full: 0,
            store: io.store,
            store_table: Some((&chirp, u64::from(n) * 8, Table::Line)),
            n_out: Some(io.n_out.unwrap_or(n)),
        },
    )
}

fn check_axis(axis: usize) -> Result<(), LumaError> {
    if axis >= 4 {
        return Err(LumaError::InvalidShape(format!("FFT axis {axis} is not below 4")));
    }
    Ok(())
}

/// The complex transform of `input` along each of `axes` in turn (the first one reading real
/// values if `real_input`).
fn run_complex(input: &Array<C32>, axes: &[usize], inverse: bool, norm: FftNorm) -> Outcome<Array<C32>> {
    let ctx = input.context();
    let dims = input.shape().dims();
    for &a in axes {
        check_axis(a)?;
    }
    let output = Array::<C32>::alloc(ctx, input.shape())?;
    if input.len() == 0 {
        return Ok(output);
    }
    let (in_pin, in_bytes) = input.ready()?;
    let (out_pin, out_bytes) = output.ready()?;
    let mut scratch = Scratch::new(ctx);
    let mut current = (in_pin.buffer().clone(), in_bytes);
    for (i, &axis) in axes.iter().enumerate() {
        let last = i + 1 == axes.len();
        let target = if last { (out_pin.buffer().clone(), out_bytes) } else { (scratch.take(bytes_of(dims))?, bytes_of(dims)) };
        let factor = match norm {
            FftNorm::Scale(_) if i > 0 => 1.0,
            other => other.factor(dims[axis], inverse),
        };
        run_axis(
            ctx,
            &mut scratch,
            geometry(dims, axis)?,
            inverse,
            factor,
            &AxisIo { src: (&current.0, current.1), dst: (&target.0, target.1), load: LoadKind::Complex, store: StoreKind::Complex, n_in: None, n_out: None, n_full: 0, store_table: None },
        )?;
        current = target;
    }
    Ok(output)
}

impl Array<C32> {
    /// The discrete Fourier transform along axis 0, batched over the other axes:
    /// `X[k] = sum_n x[n] e^(-2 pi i n k / N)`. Any length is allowed (powers of two and products
    /// of small primes run as a few passes over the data; other lengths use Bluestein's
    /// algorithm, a few times slower). Errors poison the result like other operations.
    #[track_caller]
    pub fn fft(&self) -> Array<C32> {
        self.fft_axes(&[0], FftNorm::Backward)
    }

    /// The inverse transform along axis 0 ([`FftNorm::Backward`]: divided by `N`).
    #[track_caller]
    pub fn ifft(&self) -> Array<C32> {
        self.ifft_axes(&[0], FftNorm::Backward)
    }

    /// The transform along `axis` (0 to 3), batched over the other axes, with the given scaling.
    #[track_caller]
    pub fn fft_axis(&self, axis: usize, norm: FftNorm) -> Array<C32> {
        self.fft_axes(&[axis], norm)
    }

    /// The inverse transform along `axis`.
    #[track_caller]
    pub fn ifft_axis(&self, axis: usize, norm: FftNorm) -> Array<C32> {
        self.ifft_axes(&[axis], norm)
    }

    /// The two-dimensional transform over axes 0 and 1, batched over axes 2 and 3.
    #[track_caller]
    pub fn fft2(&self) -> Array<C32> {
        self.fft_axes(&[0, 1], FftNorm::Backward)
    }

    /// The two-dimensional inverse transform over axes 0 and 1.
    #[track_caller]
    pub fn ifft2(&self) -> Array<C32> {
        self.ifft_axes(&[0, 1], FftNorm::Backward)
    }

    /// The three-dimensional transform over axes 0, 1 and 2, batched over axis 3.
    #[track_caller]
    pub fn fft3(&self) -> Array<C32> {
        self.fft_axes(&[0, 1, 2], FftNorm::Backward)
    }

    /// The three-dimensional inverse transform over axes 0, 1 and 2.
    #[track_caller]
    pub fn ifft3(&self) -> Array<C32> {
        self.ifft_axes(&[0, 1, 2], FftNorm::Backward)
    }

    /// The transform over several axes (in the order given), scaled by `norm` (for
    /// [`FftNorm::Scale`], once in all).
    #[track_caller]
    pub fn fft_axes(&self, axes: &[usize], norm: FftNorm) -> Array<C32> {
        let location = Location::caller();
        let result = run_complex(self, axes, false, norm);
        Array::resolve(self.context(), self.shape(), result, "fft", location)
    }

    /// The inverse transform over several axes, scaled by `norm`.
    #[track_caller]
    pub fn ifft_axes(&self, axes: &[usize], norm: FftNorm) -> Array<C32> {
        let location = Location::caller();
        let result = run_complex(self, axes, true, norm);
        Array::resolve(self.context(), self.shape(), result, "ifft", location)
    }
}
