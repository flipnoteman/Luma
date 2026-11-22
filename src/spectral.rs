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
use crate::shape::Shape;
use crate::view::PadMode;

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
        while n.is_multiple_of(p) {
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
    (1..=max.min(n)).rev().find(|&d| n.is_multiple_of(d)).unwrap_or(1)
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
    /// The last pass unpacks the half spectrum of a real signal of `2n` points (see
    /// [`DftPass::post`]); the output has `n + 1` values per line. Needs `inner == 1` and a
    /// last pass of at most half a tile.
    pub(crate) post: bool,
}

fn too_large(n: usize) -> LumaError {
    LumaError::DimensionTooLarge { shape: [n, 1, 1, 1] }
}

/// Records the passes of the transform `job`.
pub(crate) fn run_lines(ctx: &Context, scratch: &mut Scratch<'_>, job: &Lines<'_>) -> Result<(), LumaError> {
    let (outer, n, inner) = (job.outer, job.n, job.inner);
    let total = u64::from(outer) * u64::from(n) * u64::from(inner);
    if total > u64::from(u32::MAX) / 2 || n >= 1 << 26 {
        return Err(too_large(total as usize));
    }
    let tile = tile_elems(ctx);
    let sizes = partition(n, tile);
    let passes = sizes.len();
    let line_bytes = total * 8;
    // The buffers the intermediate results go to, alternating; the output itself is one of them
    // when it is a plain complex array.
    let reuse_out = job.store == StoreKind::Complex && job.n_out.is_none() && !job.post;
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
        let post = last && job.post;
        let mid = n / (low / inner) / d;
        let j = mid * low;
        let rows = outer * j;
        let cmax = (tile / d).max(1);
        let cols = if post {
            (cmax / 2 * 2).max(2)
        } else if j == 1 {
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
        let out_os = if post { (n + 1) * inner } else if last { job.n_out.unwrap_or(n) * inner } else { n * inner };
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
            post,
            unit_inner: inner == 1,
        };
        let out_rs = if low > 1 {
            1
        } else if mid > 1 {
            d
        } else {
            out_os
        };
        let prodh = low / 2 + 1;
        let tpo = prodh.div_ceil((cols / 2).max(1));
        let mut params = Params {
            tiles: if post { outer * tpo } else { rows.div_ceil(cols) },
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
            den: if post { 2 * n } else { d.checked_mul(mid).ok_or_else(|| too_large(n as usize))? },
            inner,
            prod: low / inner,
            n_in: job.n_in.unwrap_or(n),
            n_lim: job.n_lim.unwrap_or(n),
            n_out: job.n_out.unwrap_or(n),
            n_full: job.n_full,
            scale: if last { job.scale } else { 1.0 },
            cj: if job.inverse { -1.0 } else { 1.0 },
            mrs: u32::from(low == 1 && mid > 1),
            tpo,
            prodh,
            inv8: 0.0,
        };
        params.inv8 = (8.0 / f64::from(params.den)) as f32;
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
/// them, doubled, for the inverse; with `table`, multiplying the spectrum by a table first).
/// An invocation handles the frequencies `k` and `h - k`, which share their intermediate
/// results. Bindings: `src`, `dst`, `tab` if any, `Params { outer, h, inner, scale }`.
pub(crate) fn real_stage_source(forward: bool, table: bool) -> String {
    let (read_a, read_b, binding) = if table {
        (
            "cmul(src[xb + k * p.inner], tab[ii + k * p.inner])",
            "cmul(src[xb + (p.h - k) * p.inner], tab[ii + (p.h - k) * p.inner])",
            "@group(1) @binding(2) var<storage, read> tab: array<vec2<f32>>;\n@group(1) @binding(3) var<uniform> p: Params;",
        )
    } else {
        ("src[xb + k * p.inner]", "src[xb + (p.h - k) * p.inner]", "@group(1) @binding(2) var<uniform> p: Params;")
    };
    let body = if forward {
        // E = (Z[k] + conj Z[h - k]) / 2, O = -i (Z[k] - conj Z[h - k]) / 2: X[k] = E + W^k O and
        // X[h - k] = conj(E - W^k O).
        "let zb = o * p.h * p.inner + ii;
    let zk = src[zb + (k % p.h) * p.inner];
    let z2 = src[zb + ((p.h - k) % p.h) * p.inner];
    let zc = vec2<f32>(z2.x, -z2.y);
    let e = (zk + zc) * 0.5;
    let d = zk - zc;
    let od = cmul(tw(k, 2u * p.h), vec2<f32>(d.y, -d.x) * 0.5);
    let ob = o * (p.h + 1u) * p.inner + ii;
    let a = e + od;
    let b = e - od;
    dst[ob + k * p.inner] = a * p.scale;
    if (2u * k != p.h) { dst[ob + (p.h - k) * p.inner] = vec2<f32>(b.x, -b.y) * p.scale; }"
            .to_string()
    } else {
        // E = X[k] + conj X[h - k], u = i (X[k] - conj X[h - k]) conj(W^k): Z[k] = E + u and
        // Z[h - k] = conj(E - u).
        format!(
            "let xb = o * (p.h + 1u) * p.inner + ii;
    let xk = {read_a};
    let x2 = {read_b};
    let xc = vec2<f32>(x2.x, -x2.y);
    let e = xk + xc;
    let d = xk - xc;
    let w = tw(k, 2u * p.h);
    let t = cmul(d, vec2<f32>(w.x, -w.y));
    let u = vec2<f32>(-t.y, t.x);
    let ob = o * p.h * p.inner + ii;
    let a = e + u;
    let b = e - u;
    dst[ob + k * p.inner] = a * p.scale;
    if (k > 0u && 2u * k != p.h) {{ dst[ob + (p.h - k) * p.inner] = vec2<f32>(b.x, -b.y) * p.scale; }}"
        )
    };
    format!(
        "struct Params {{ outer: u32, h: u32, inner: u32, scale: f32 }}
@group(1) @binding(0) var<storage, read> src: array<vec2<f32>>;
@group(1) @binding(1) var<storage, read_write> dst: array<vec2<f32>>;
{binding}
{TW_FUNCTION}
@compute @workgroup_size({WORKGROUP})
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {{
    let i = gid.x + gid.y * nwg.x * {WORKGROUP}u;
    let half = p.h / 2u + 1u;
    if (i >= p.outer * half * p.inner) {{ return; }}
    let ii = i % p.inner;
    let k = (i / p.inner) % half;
    let o = i / (p.inner * half);
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
    pub(crate) load_table: Option<(&'a luma_core::Buffer, u64, Table)>,
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
                load_table: io.load_table,
                n_in: io.n_in,
                n_lim: None,
                n_full: io.n_full,
                store: io.store,
                store_table: io.store_table,
                n_out: io.n_out,
                post: false,
            },
        );
    }
    // Bluestein: X[k] = w[k] sum_j (x[j] w[j]) conj(w)[k - j], a circular convolution of
    // length m >= 2n - 1.
    if io.load == LoadKind::RealPair || io.store == StoreKind::RealPair || io.store_table.is_some() || io.load_table.is_some() {
        return Err(LumaError::InvalidShape("internal: Bluestein with packed or tabled ends".into()));
    }
    let tables = chirp_tables(ctx, n, inverse)?;
    let (m, chirp, b_hat) = (tables.m, &tables.chirp, &tables.b_hat);
    let size_bytes = |len: u32| u64::from(outer) * u64::from(len) * u64::from(inner) * 8;
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
            load_table: Some((chirp, u64::from(n) * 8, Table::Line)),
            n_in: Some(io.n_in.unwrap_or(n)),
            n_lim: Some(n),
            n_full: io.n_full,
            store: StoreKind::Complex,
            store_table: Some((b_hat, u64::from(m) * 8, Table::Line)),
            n_out: None,
            post: false,
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
            store_table: Some((chirp, u64::from(n) * 8, Table::Line)),
            n_out: Some(io.n_out.unwrap_or(n)),
            post: false,
        },
    )
}

/// The tables of Bluestein's algorithm for one length and direction, kept by the context: the
/// chirp `w[j] = e^(+-i pi j^2 / n)` and the spectrum, of the padded length `m`, of `conj(w)`.
#[derive(Debug)]
pub(crate) struct ChirpTables {
    m: u32,
    chirp: luma_core::Buffer,
    b_hat: luma_core::Buffer,
}

/// The tables for `n` points (built and recorded on first use).
fn chirp_tables(ctx: &Context, n: u32, inverse: bool) -> Result<std::sync::Arc<ChirpTables>, LumaError> {
    if let Some(found) = ctx.chirps().lock().unwrap_or_else(std::sync::PoisonError::into_inner).get(&(n, inverse)) {
        return Ok(found.clone());
    }
    let m = pow2_at_least(2 * n - 1)?;
    let chirp = ctx.alloc_buffer(u64::from(n) * 8, crate::context::STORAGE_USAGE)?;
    let b_hat = ctx.alloc_buffer(u64::from(m) * 8, crate::context::STORAGE_USAGE)?;
    let mut scratch = Scratch::new(ctx);
    let padded = scratch.take(u64::from(m) * 8)?;
    let params = [n, m, (if inverse { 1.0f32 } else { -1.0 }).to_bits(), 0];
    let kernel = ctx.builtin(Builtin::Chirp, "f32")?;
    let groups = fold_workgroups(ctx, m.div_ceil(WORKGROUP))?;
    ctx.launch(
        &kernel,
        groups,
        [m, 1, 1],
        &[RawArg::write(&chirp, u64::from(n) * 8), RawArg::write(&padded, u64::from(m) * 8), RawArg::Uniform(bytemuck::bytes_of(&params))],
    )?;
    run_lines(
        ctx,
        &mut scratch,
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
            post: false,
        },
    )?;
    let tables = std::sync::Arc::new(ChirpTables { m, chirp, b_hat });
    let mut cache = ctx.chirps().lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    if cache.len() >= 16 {
        cache.clear();
    }
    cache.insert((n, inverse), tables.clone());
    Ok(tables)
}

fn check_axis(axis: usize) -> Result<(), LumaError> {
    if axis >= 4 {
        return Err(LumaError::InvalidShape(format!("FFT axis {axis} is not below 4")));
    }
    Ok(())
}

/// Launches the real-transform packing kernel over `total` outputs.
fn launch_real_stage(
    ctx: &Context,
    forward: bool,
    (outer, h, inner): (u32, u32, u32),
    scale: f32,
    src: (&luma_core::Buffer, u64),
    dst: (&luma_core::Buffer, u64),
    table: Option<(&luma_core::Buffer, u64)>,
) -> Result<(), LumaError> {
    let total = u64::from(outer) * u64::from(h / 2 + 1) * u64::from(inner);
    let total = u32::try_from(total).map_err(|_| too_large(total as usize))?;
    let params = [outer, h, inner, scale.to_bits()];
    let kernel = ctx.builtin(Builtin::RealStage(forward, table.is_some()), "f32")?;
    let groups = fold_workgroups(ctx, total.div_ceil(WORKGROUP))?;
    let mut args = vec![RawArg::read(src.0, src.1), RawArg::write(dst.0, dst.1)];
    if let Some((b, bytes)) = table {
        args.push(RawArg::read(b, bytes));
    }
    args.push(RawArg::Uniform(bytemuck::bytes_of(&params)));
    ctx.launch(&kernel, groups, [total, 1, 1], &args)
}

/// The transform of real values along `axis` of `dims` into the half spectrum (`n / 2 + 1`
/// values), scaled by `factor`.
fn real_forward(
    ctx: &Context,
    scratch: &mut Scratch<'_>,
    dims: [usize; 4],
    axis: usize,
    factor: f32,
    src: (&luma_core::Buffer, u64),
    dst: (&luma_core::Buffer, u64),
) -> Result<(), LumaError> {
    let (outer, n, inner) = geometry(dims, axis)?;
    if n % 2 == 0 && is_smooth(n / 2) {
        // The even and odd samples as one complex signal of half the length; the packing step
        // separates their spectra.
        let h = n / 2;
        let sizes = partition(h, tile_elems(ctx));
        if inner == 1 && sizes.len() >= 2 && sizes.last().is_some_and(|&d| 2 * d <= tile_elems(ctx)) {
            // The last pass unpacks the spectrum in workgroup memory.
            return run_lines(
                ctx,
                scratch,
                &Lines {
                    outer,
                    n: h,
                    inner,
                    src,
                    dst,
                    inverse: false,
                    scale: factor,
                    load: LoadKind::RealPair,
                    load_table: None,
                    n_in: None,
                    n_lim: None,
                    n_full: 0,
                    store: StoreKind::Complex,
                    store_table: None,
                    n_out: None,
                    post: true,
                },
            );
        }
        let bytes = u64::from(outer) * u64::from(h) * u64::from(inner) * 8;
        let z = scratch.take(bytes)?;
        run_lines(
            ctx,
            scratch,
            &Lines {
                outer,
                n: h,
                inner,
                src,
                dst: (&z, bytes),
                inverse: false,
                scale: 1.0,
                load: LoadKind::RealPair,
                load_table: None,
                n_in: None,
                n_lim: None,
                n_full: 0,
                store: StoreKind::Complex,
                store_table: None,
                n_out: None,
                post: false,
            },
        )?;
        return launch_real_stage(ctx, true, (outer, h, inner), factor, (&z, bytes), dst, None);
    }
    let io = AxisIo { src, dst, load: LoadKind::Real, store: StoreKind::Complex, n_in: None, n_out: Some(n / 2 + 1), n_full: 0, store_table: None, load_table: None };
    run_axis(ctx, scratch, (outer, n, inner), false, factor, &io)
}

/// The inverse of [`real_forward`]: `n` real values from the half spectrum along `axis` of
/// `dims` (the dimensions of the output), scaled by `factor`.
#[allow(clippy::too_many_arguments)]
fn real_inverse(
    ctx: &Context,
    scratch: &mut Scratch<'_>,
    dims: [usize; 4],
    axis: usize,
    factor: f32,
    src: (&luma_core::Buffer, u64),
    dst: (&luma_core::Buffer, u64),
    product: Option<(&luma_core::Buffer, u64)>,
) -> Result<(), LumaError> {
    let (outer, n, inner) = geometry(dims, axis)?;
    let k = n / 2 + 1;
    if product.is_some() && !(n % 2 == 0 && is_smooth(n / 2)) {
        return Err(LumaError::InvalidShape("internal: a spectrum product needs an even smooth length".into()));
    }
    if n % 2 == 0 && is_smooth(n / 2) {
        let h = n / 2;
        if inner == 1 {
            // The first pass packs the half spectrum as it loads it.
            return run_lines(
                ctx,
                scratch,
                &Lines {
                    outer,
                    n: h,
                    inner,
                    src,
                    dst,
                    inverse: true,
                    scale: factor,
                    load: LoadKind::HalfSpectrum,
                    load_table: product.map(|(b, bytes)| (b, bytes, Table::Slab)),
                    n_in: Some(h + 1),
                    n_lim: None,
                    n_full: 0,
                    store: StoreKind::RealPair,
                    store_table: None,
                    n_out: None,
                    post: false,
                },
            );
        }
        let bytes = u64::from(outer) * u64::from(h) * u64::from(inner) * 8;
        let z = scratch.take(bytes)?;
        launch_real_stage(ctx, false, (outer, h, inner), 1.0, src, (&z, bytes), product)?;
        return run_lines(
            ctx,
            scratch,
            &Lines {
                outer,
                n: h,
                inner,
                src: (&z, bytes),
                dst,
                inverse: true,
                scale: factor,
                load: LoadKind::Complex,
                load_table: None,
                n_in: None,
                n_lim: None,
                n_full: 0,
                store: StoreKind::RealPair,
                store_table: None,
                n_out: None,
                post: false,
            },
        );
    }
    let io = AxisIo { src, dst, load: LoadKind::Hermitian, store: StoreKind::Real, n_in: Some(k), n_out: None, n_full: n, store_table: None, load_table: None };
    run_axis(ctx, scratch, (outer, n, inner), true, factor, &io)
}

/// What the first axis of a transform reads.
#[derive(Clone, Copy, PartialEq)]
enum Input {
    Complex,
    Real,
}

/// The complex transform of the array in `src` (of `dims`) along each of `axes` in turn.
fn run_complex(
    ctx: &Context,
    dims: [usize; 4],
    src: (luma_core::Buffer, u64),
    input: Input,
    axes: &[usize],
    inverse: bool,
    norm: FftNorm,
) -> Outcome<Array<C32>> {
    for &a in axes {
        check_axis(a)?;
    }
    let output = Array::<C32>::alloc(ctx, Shape::from(dims))?;
    if dims.contains(&0) {
        return Ok(output);
    }
    let (out_pin, out_bytes) = output.ready()?;
    let mut scratch = Scratch::new(ctx);
    let mut current = src;
    for (i, &axis) in axes.iter().enumerate() {
        let last = i + 1 == axes.len();
        let target = if last { (out_pin.buffer().clone(), out_bytes) } else { (scratch.take(bytes_of(dims))?, bytes_of(dims)) };
        let factor = match norm {
            FftNorm::Scale(_) if i > 0 => 1.0,
            other => other.factor(dims[axis], inverse),
        };
        let load = if i == 0 && input == Input::Real { LoadKind::Real } else { LoadKind::Complex };
        run_axis(
            ctx,
            &mut scratch,
            geometry(dims, axis)?,
            inverse,
            factor,
            &AxisIo { src: (&current.0, current.1), dst: (&target.0, target.1), load, store: StoreKind::Complex, n_in: None, n_out: None, n_full: 0, store_table: None, load_table: None },
        )?;
        current = target;
    }
    Ok(output)
}

/// The half spectrum of real input: the first of `axes` is halved, the others are complex
/// transforms of the result.
fn run_real_forward(input: &Array<f32>, axes: &[usize], norm: FftNorm) -> Outcome<Array<C32>> {
    let ctx = input.context();
    let mut dims = input.shape().dims();
    for &a in axes {
        check_axis(a)?;
    }
    let Some(&first) = axes.first() else {
        return Err(LumaError::InvalidShape("rfft needs at least one axis".into()).into());
    };
    let n = dims[first];
    let mut half = dims;
    half[first] = n / 2 + 1;
    let output = Array::<C32>::alloc(ctx, Shape::from(half))?;
    if dims.contains(&0) {
        return Ok(output);
    }
    let (in_pin, in_bytes) = input.ready()?;
    let (out_pin, out_bytes) = output.ready()?;
    let mut scratch = Scratch::new(ctx);
    let half_bytes = bytes_of(half);
    let rest = &axes[1..];
    let target = if rest.is_empty() { (out_pin.buffer().clone(), out_bytes) } else { (scratch.take(half_bytes)?, half_bytes) };
    real_forward(ctx, &mut scratch, dims, first, norm.factor(n, false), (in_pin.buffer(), in_bytes), (&target.0, target.1))?;
    dims = half;
    let mut current = target;
    for (i, &axis) in rest.iter().enumerate() {
        let last = i + 1 == rest.len();
        let next = if last { (out_pin.buffer().clone(), out_bytes) } else { (scratch.take(half_bytes)?, half_bytes) };
        let factor = if matches!(norm, FftNorm::Scale(_)) { 1.0 } else { norm.factor(dims[axis], false) };
        run_axis(
            ctx,
            &mut scratch,
            geometry(dims, axis)?,
            false,
            factor,
            &AxisIo { src: (&current.0, current.1), dst: (&next.0, next.1), load: LoadKind::Complex, store: StoreKind::Complex, n_in: None, n_out: None, n_full: 0, store_table: None, load_table: None },
        )?;
        current = next;
    }
    Ok(output)
}

/// The real signal of `n` points along the first of `axes` from a half spectrum: the other
/// axes are inverse complex transforms first.
fn run_real_inverse(
    input: &Array<C32>,
    axes: &[usize],
    n: usize,
    norm: FftNorm,
    product: Option<&Array<C32>>,
) -> Outcome<Array<f32>> {
    let ctx = input.context();
    let dims = input.shape().dims();
    for &a in axes {
        check_axis(a)?;
    }
    let Some(&first) = axes.first() else {
        return Err(LumaError::InvalidShape("irfft needs at least one axis".into()).into());
    };
    if n == 0 || dims[first] != n / 2 + 1 {
        return Err(LumaError::InvalidShape(format!("irfft of {n} points needs {} frequencies along axis {first}, not {}", n / 2 + 1, dims[first])).into());
    }
    let mut out_dims = dims;
    out_dims[first] = n;
    let output = Array::<f32>::alloc(ctx, Shape::from(out_dims))?;
    if dims.contains(&0) {
        return Ok(output);
    }
    let (in_pin, in_bytes) = input.ready()?;
    let (out_pin, out_bytes) = output.ready()?;
    let table = product.map(|t| t.ready()).transpose()?;
    let table = table.as_ref().map(|(pin, bytes)| (pin.buffer(), *bytes));
    let mut scratch = Scratch::new(ctx);
    let rest = &axes[1..];
    let mut current = (in_pin.buffer().clone(), in_bytes);
    for (i, &axis) in rest.iter().enumerate() {
        let load_table = if i == 0 { table.map(|(b, bytes)| (b, bytes, Table::Slab)) } else { None };
        let next = (scratch.take(bytes_of(dims))?, bytes_of(dims));
        let factor = if matches!(norm, FftNorm::Scale(_)) { 1.0 } else { norm.factor(dims[axis], true) };
        run_axis(
            ctx,
            &mut scratch,
            geometry(dims, axis)?,
            true,
            factor,
            &AxisIo { src: (&current.0, current.1), dst: (&next.0, next.1), load: LoadKind::Complex, store: StoreKind::Complex, n_in: None, n_out: None, n_full: 0, store_table: None, load_table },
        )?;
        current = next;
    }
    let product = if rest.is_empty() { table } else { None };
    real_inverse(ctx, &mut scratch, out_dims, first, norm.factor(n, true), (&current.0, current.1), (out_pin.buffer(), out_bytes), product)?;
    Ok(output)
}

fn complex_input(input: &Array<C32>) -> Outcome<(luma_core::Buffer, u64)> {
    let (pin, bytes) = input.ready()?;
    Ok((pin.buffer().clone(), bytes))
}

impl Array<C32> {
    /// The discrete Fourier transform along axis 0, batched over the other axes:
    /// `X[k] = sum_n x[n] e^(-2 pi i n k / N)`, unscaled. Any length is allowed: powers of two
    /// and products of small primes run as a few passes over the data, other lengths use
    /// Bluestein's algorithm (several times slower). Errors poison the result like other
    /// operations.
    #[track_caller]
    pub fn fft(&self) -> Array<C32> {
        self.fft_axes(&[0], FftNorm::Backward)
    }

    /// The inverse transform along axis 0, divided by `N` (so `x.fft().ifft()` is `x`).
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

    /// The two-dimensional inverse transform over axes 0 and 1, divided by `N0 N1`.
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

    /// The transform over several axes (one after the other, in the order given), scaled by
    /// `norm` ([`FftNorm::Scale`] once in all).
    #[track_caller]
    pub fn fft_axes(&self, axes: &[usize], norm: FftNorm) -> Array<C32> {
        let location = Location::caller();
        let result = complex_input(self).and_then(|src| run_complex(self.context(), self.shape().dims(), src, Input::Complex, axes, false, norm));
        Array::resolve(self.context(), self.shape(), result, "fft", location)
    }

    /// The inverse transform over several axes, scaled by `norm`.
    #[track_caller]
    pub fn ifft_axes(&self, axes: &[usize], norm: FftNorm) -> Array<C32> {
        let location = Location::caller();
        let result = complex_input(self).and_then(|src| run_complex(self.context(), self.shape().dims(), src, Input::Complex, axes, true, norm));
        Array::resolve(self.context(), self.shape(), result, "ifft", location)
    }

    /// The real signal of `n` points along axis 0 whose half spectrum (`n / 2 + 1` values) this
    /// is, divided by `n` ([`FftNorm::Backward`]): the inverse of [`Array::<f32>::rfft`]. The
    /// imaginary parts of the zero and (for even `n`) Nyquist frequencies are ignored.
    #[track_caller]
    pub fn irfft(&self, n: usize) -> Array<f32> {
        self.irfft_axes(&[0], n, FftNorm::Backward)
    }

    /// The inverse of [`Array::<f32>::rfft2`]: complex inverse transform along axis 1, then the
    /// real one along axis 0, which has `n0` points.
    #[track_caller]
    pub fn irfft2(&self, n0: usize) -> Array<f32> {
        self.irfft_axes(&[0, 1], n0, FftNorm::Backward)
    }

    /// The inverse of [`Array::<f32>::rfft3`].
    #[track_caller]
    pub fn irfft3(&self, n0: usize) -> Array<f32> {
        self.irfft_axes(&[0, 1, 2], n0, FftNorm::Backward)
    }

    /// The inverse of [`Array::<f32>::rfft_axes`]: the first axis of `axes` is the real one and
    /// gets `n` points, the others are inverse complex transforms.
    #[track_caller]
    pub fn irfft_axes(&self, axes: &[usize], n: usize, norm: FftNorm) -> Array<f32> {
        let location = Location::caller();
        let mut dims = self.shape().dims();
        if let Some(&a) = axes.first().filter(|&&a| a < 4) {
            dims[a] = n;
        }
        let result = run_real_inverse(self, axes, n, norm, None);
        Array::resolve(self.context(), Shape::from(dims), result, "irfft", location)
    }
}

impl Array<f32> {
    /// The transform of a real signal along axis 0: all `N` frequencies (the upper half is the
    /// conjugate of the lower), see [`Array::<C32>::fft`]. [`Array::<f32>::rfft`] computes only the
    /// half that is not redundant, in about half the time.
    #[track_caller]
    pub fn fft(&self) -> Array<C32> {
        self.fft_axes(&[0], FftNorm::Backward)
    }

    /// The transform of a real array along `axis`.
    #[track_caller]
    pub fn fft_axis(&self, axis: usize, norm: FftNorm) -> Array<C32> {
        self.fft_axes(&[axis], norm)
    }

    /// The transform of a real array over several axes.
    #[track_caller]
    pub fn fft_axes(&self, axes: &[usize], norm: FftNorm) -> Array<C32> {
        let location = Location::caller();
        let result = self.ready().and_then(|(pin, bytes): (_, u64)| {
            run_complex(self.context(), self.shape().dims(), (pin.buffer().clone(), bytes), Input::Real, axes, false, norm)
        });
        Array::resolve(self.context(), self.shape(), result, "fft", location)
    }

    /// The transform of a real signal along axis 0, batched over the other axes: the first
    /// `N / 2 + 1` frequencies (the rest is their conjugate mirror), which is what
    /// ArrayFire's `fftR2C` returns. Even lengths run as a complex transform of half the length.
    #[track_caller]
    pub fn rfft(&self) -> Array<C32> {
        self.rfft_axes(&[0], FftNorm::Backward)
    }

    /// The half spectrum along `axis`.
    #[track_caller]
    pub fn rfft_axis(&self, axis: usize, norm: FftNorm) -> Array<C32> {
        self.rfft_axes(&[axis], norm)
    }

    /// The two-dimensional transform of real input: halved along axis 0, complete along axis 1.
    #[track_caller]
    pub fn rfft2(&self) -> Array<C32> {
        self.rfft_axes(&[0, 1], FftNorm::Backward)
    }

    /// The three-dimensional transform of real input: halved along axis 0.
    #[track_caller]
    pub fn rfft3(&self) -> Array<C32> {
        self.rfft_axes(&[0, 1, 2], FftNorm::Backward)
    }

    /// The transform of real input over several axes: the first of `axes` is halved (to
    /// `N / 2 + 1`), the others are complex transforms of the result.
    #[track_caller]
    pub fn rfft_axes(&self, axes: &[usize], norm: FftNorm) -> Array<C32> {
        let location = Location::caller();
        let mut dims = self.shape().dims();
        if let Some(&a) = axes.first().filter(|&&a| a < 4) {
            dims[a] = dims[a] / 2 + 1;
        }
        let result = run_real_forward(self, axes, norm);
        Array::resolve(self.context(), Shape::from(dims), result, "rfft", location)
    }
}

/// The smallest even length of at least `n` that is a product of 2, 3 and 5.
fn fft_size(n: usize) -> usize {
    let smooth = |mut m: usize| {
        for p in [2, 3, 5] {
            while m.is_multiple_of(p) {
                m /= p;
            }
        }
        m == 1
    };
    (n.max(2)..).find(|&m| m % 2 == 0 && smooth(m)).expect("a smooth number exists")
}

/// The convolution of the first `nd` axes of `signal` with `kernel`, batched over the others.
fn run_convolve(signal: &Array<f32>, kernel: &Array<f32>, nd: usize, mode: ConvMode) -> Outcome<Array<f32>> {
    if !signal.context().same_as(kernel.context()) {
        return Err(LumaError::ContextMismatch.into());
    }
    let (s, k) = (signal.shape().dims(), kernel.shape().dims());
    for d in 0..4 {
        let ok = if d < nd { s[d] > 0 && k[d] > 0 } else { k[d] == 1 };
        if !ok {
            return Err(LumaError::InvalidShape(format!(
                "fft_convolve{nd}: signal {s:?} and kernel {k:?} (the kernel must be non-empty along the first {nd} axes and 1 along the others)"
            ))
            .into());
        }
    }
    let mut padded = s;
    let mut widths_s = [(0usize, 0usize); 4];
    let mut widths_k = [(0usize, 0usize); 4];
    for d in 0..nd {
        padded[d] = fft_size(s[d] + k[d] - 1);
        widths_s[d] = (0, padded[d] - s[d]);
        widths_k[d] = (0, padded[d] - k[d]);
    }
    let pad = |a: &Array<f32>, w: [(usize, usize); 4]| a.pad(w, PadMode::Zero);
    let axes: Vec<usize> = (0..nd).collect();
    let spectrum = pad(signal, widths_s).rfft_axes(&axes, FftNorm::None);
    let kernel_spectrum = pad(kernel, widths_k).rfft_axes(&axes, FftNorm::None);
    // Inverse in the reverse order: the last forward axis first, where the product happens.
    let mut inverse_axes = vec![0];
    inverse_axes.extend((1..nd).rev());
    let full = run_real_inverse(&spectrum, &inverse_axes, padded[0], FftNorm::Backward, Some(&kernel_spectrum))?;
    let mut out = full;
    for d in 0..nd {
        let (start, len) = match mode {
            ConvMode::Full => (0, s[d] + k[d] - 1),
            ConvMode::Same => (k[d] / 2, s[d]),
        };
        if len != padded[d] || start != 0 {
            out = out.slice(d, start..start + len);
        }
    }
    Ok(out)
}

impl Array<f32> {
    /// The convolution of the signal with a kernel along axis 0 by FFTs (batched over the other
    /// axes; the kernel must be 1 along them): both are zero-padded to a fast length, multiplied
    /// in the frequency domain and transformed back. [`ConvMode::Same`] returns the signal's
    /// shape (the kernel centred as ArrayFire does), [`ConvMode::Full`] the whole
    /// `signal + kernel - 1` points.
    #[track_caller]
    pub fn fft_convolve1(&self, kernel: &Array<f32>, mode: ConvMode) -> Array<f32> {
        self.fft_convolve_nd(kernel, 1, mode)
    }

    /// The two-dimensional convolution over axes 0 and 1 by FFTs, batched over axes 2 and 3 (see
    /// [`Array::fft_convolve1`]).
    #[track_caller]
    pub fn fft_convolve2(&self, kernel: &Array<f32>, mode: ConvMode) -> Array<f32> {
        self.fft_convolve_nd(kernel, 2, mode)
    }

    /// The three-dimensional convolution over axes 0, 1 and 2 by FFTs, batched over axis 3 (see
    /// [`Array::fft_convolve1`]).
    #[track_caller]
    pub fn fft_convolve3(&self, kernel: &Array<f32>, mode: ConvMode) -> Array<f32> {
        self.fft_convolve_nd(kernel, 3, mode)
    }

    #[track_caller]
    fn fft_convolve_nd(&self, kernel: &Array<f32>, nd: usize, mode: ConvMode) -> Array<f32> {
        let location = Location::caller();
        let (s, k) = (self.shape().dims(), kernel.shape().dims());
        let mut dims = s;
        for d in 0..nd {
            dims[d] = match mode {
                ConvMode::Same => s[d],
                ConvMode::Full => s[d] + k[d].max(1) - 1,
            };
        }
        let result = run_convolve(self, kernel, nd, mode);
        Array::resolve(self.context(), Shape::from(dims), result, "fft_convolve", location)
    }
}
