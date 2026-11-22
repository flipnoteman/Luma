//! Homography estimation by RANSAC or least median of squares (`homography`).
//!
//! One workgroup per hypothesis: its first invocation draws four distinct point pairs (a hash of
//! the hypothesis number, so every run draws the same samples), solves the 8 x 8 system of the
//! direct linear transform on Hartley-normalised coordinates and writes the homography to
//! workgroup memory; then the whole workgroup scores it over all the points (RANSAC: the number of
//! points whose reprojection is within the threshold; least median of squares: the median of the
//! squared reprojection errors, found by bisection on the bits of the non-negative `f32` errors,
//! with the inliers counted at 2.5 robust standard deviations). A second kernel picks the best
//! hypothesis.

use std::panic::Location;

use super::{invalid, to_u32, Kernel as Any};
use crate::array::Array;
use crate::builtin::Builtin;
use crate::context::{RawArg, Scratch};
use crate::error::LumaError;
use crate::ops::{fold_workgroups, Outcome};
use crate::shape::Shape;

/// How [`homography`] scores a hypothesis.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum HomographyMethod {
    /// Random sample consensus: the best hypothesis is the one with the most inliers (points whose
    /// reprojection lies within the threshold of their destination).
    Ransac,
    /// Least median of squares: the best hypothesis is the one with the smallest median of the
    /// squared reprojection errors; the threshold is not used.
    Lmeds,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum Kernel {
    /// Scores one hypothesis per workgroup.
    Evaluate,
    /// Picks the best of the hypotheses.
    Best,
}

/// Invocations of the workgroup of the evaluation.
const GROUP: u32 = 64;
/// Floats of a hypothesis record: the nine entries of the homography and its key.
const RECORD: u32 = 12;

pub(crate) fn source(kernel: Kernel) -> String {
    match kernel {
        Kernel::Evaluate => evaluate_source(),
        Kernel::Best => best_source(),
    }
}

fn evaluate_source() -> String {
    format!(
        "struct Params {{ n: u32, iterations: u32, lmeds: u32, thr2: f32, seed: u32, a: u32, b: u32, c: u32 }}
@group(1) @binding(0) var<storage, read> xs: array<f32>;
@group(1) @binding(1) var<storage, read> ys: array<f32>;
@group(1) @binding(2) var<storage, read> xd: array<f32>;
@group(1) @binding(3) var<storage, read> yd: array<f32>;
@group(1) @binding(4) var<storage, read_write> hyp: array<f32>;
@group(1) @binding(5) var<uniform> p: Params;
var<workgroup> mat: array<f32, 72>;
var<workgroup> hh: array<f32, 9>;
var<workgroup> valid: u32;
var<workgroup> red: array<u32, {GROUP}>;

fn pcg(v: u32) -> u32 {{
    let s = v * 747796405u + 2891336453u;
    let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
    return (w >> 22u) ^ w;
}}

// The squared reprojection error of point i under hh (infinite where the point maps to infinity).
fn err2(i: u32) -> f32 {{
    let x = xs[i];
    let y = ys[i];
    let w = hh[6] * x + hh[7] * y + hh[8];
    let u = (hh[0] * x + hh[1] * y + hh[2]) / w;
    let v = (hh[3] * x + hh[4] * y + hh[5]) / w;
    let e = (u - xd[i]) * (u - xd[i]) + (v - yd[i]) * (v - yd[i]);
    return select(bitcast<f32>(0x7f800000u), e, e == e && abs(w) > 1e-20);
}}

fn solve_sample(it: u32) {{
    var idx: array<u32, 4>;
    var k = 0u;
    var tries = 0u;
    loop {{
        if (k == 4u || tries > 200u) {{ break; }}
        let r = pcg(p.seed ^ pcg(it * 256u + tries)) % p.n;
        tries += 1u;
        var fresh = true;
        for (var j = 0u; j < k; j++) {{ if (idx[j] == r) {{ fresh = false; }} }}
        if (fresh) {{ idx[k] = r; k += 1u; }}
    }}
    valid = 0u;
    if (k < 4u) {{ return; }}
    // Hartley normalisation of both point sets.
    var cs = vec2<f32>(0.0);
    var cd = vec2<f32>(0.0);
    for (var j = 0u; j < 4u; j++) {{
        cs += vec2<f32>(xs[idx[j]], ys[idx[j]]) * 0.25;
        cd += vec2<f32>(xd[idx[j]], yd[idx[j]]) * 0.25;
    }}
    var ds = 0.0;
    var dd = 0.0;
    for (var j = 0u; j < 4u; j++) {{
        ds += length(vec2<f32>(xs[idx[j]], ys[idx[j]]) - cs) * 0.25;
        dd += length(vec2<f32>(xd[idx[j]], yd[idx[j]]) - cd) * 0.25;
    }}
    if (ds < 1e-12 || dd < 1e-12) {{ return; }}
    let ss = 1.41421356 / ds;
    let sd = 1.41421356 / dd;
    for (var j = 0u; j < 4u; j++) {{
        let s = (vec2<f32>(xs[idx[j]], ys[idx[j]]) - cs) * ss;
        let d = (vec2<f32>(xd[idx[j]], yd[idx[j]]) - cd) * sd;
        let r0 = 2u * j * 9u;
        let r1 = r0 + 9u;
        mat[r0 + 0u] = s.x; mat[r0 + 1u] = s.y; mat[r0 + 2u] = 1.0;
        mat[r0 + 3u] = 0.0; mat[r0 + 4u] = 0.0; mat[r0 + 5u] = 0.0;
        mat[r0 + 6u] = -d.x * s.x; mat[r0 + 7u] = -d.x * s.y; mat[r0 + 8u] = d.x;
        mat[r1 + 0u] = 0.0; mat[r1 + 1u] = 0.0; mat[r1 + 2u] = 0.0;
        mat[r1 + 3u] = s.x; mat[r1 + 4u] = s.y; mat[r1 + 5u] = 1.0;
        mat[r1 + 6u] = -d.y * s.x; mat[r1 + 7u] = -d.y * s.y; mat[r1 + 8u] = d.y;
    }}
    // Gaussian elimination with partial pivoting of the 8 x 9 augmented system.
    for (var c = 0u; c < 8u; c++) {{
        var best = c;
        var bv = abs(mat[c * 9u + c]);
        for (var r = c + 1u; r < 8u; r++) {{
            let v = abs(mat[r * 9u + c]);
            if (v > bv) {{ bv = v; best = r; }}
        }}
        if (bv < 1e-6) {{ return; }}
        if (best != c) {{
            for (var j = 0u; j < 9u; j++) {{
                let tmp = mat[c * 9u + j];
                mat[c * 9u + j] = mat[best * 9u + j];
                mat[best * 9u + j] = tmp;
            }}
        }}
        let inv = 1.0 / mat[c * 9u + c];
        for (var r = c + 1u; r < 8u; r++) {{
            let f = mat[r * 9u + c] * inv;
            for (var j = c; j < 9u; j++) {{ mat[r * 9u + j] -= f * mat[c * 9u + j]; }}
        }}
    }}
    var hn: array<f32, 9>;
    hn[8] = 1.0;
    for (var c = 8u; c > 0u; c--) {{
        let r = c - 1u;
        var acc = mat[r * 9u + 8u];
        for (var j = c; j < 8u; j++) {{ acc -= mat[r * 9u + j] * hn[j]; }}
        hn[r] = acc / mat[r * 9u + r];
    }}
    // H = Td^-1 Hn Ts, Ts = [[ss, 0, -ss cs.x], [0, ss, -ss cs.y], [0, 0, 1]].
    var a: array<f32, 9>;
    for (var r = 0u; r < 3u; r++) {{
        a[r * 3u + 0u] = hn[r * 3u + 0u] * ss;
        a[r * 3u + 1u] = hn[r * 3u + 1u] * ss;
        a[r * 3u + 2u] = hn[r * 3u + 2u] - ss * (hn[r * 3u + 0u] * cs.x + hn[r * 3u + 1u] * cs.y);
    }}
    for (var c = 0u; c < 3u; c++) {{
        hh[c] = a[c] / sd + cd.x * a[6u + c];
        hh[3u + c] = a[3u + c] / sd + cd.y * a[6u + c];
        hh[6u + c] = a[6u + c];
    }}
    valid = 1u;
    for (var j = 0u; j < 9u; j++) {{ if (hh[j] != hh[j]) {{ valid = 0u; }} }}
}}

@compute @workgroup_size({GROUP})
fn main(@builtin(local_invocation_index) t: u32, @builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {{
    let it = wid.x + wid.y * nwg.x;
    if (it >= p.iterations) {{ return; }}
    if (t == 0u) {{ solve_sample(it); }}
    workgroupBarrier();
    let ok = workgroupUniformLoad(&valid);
    let rec = it * {RECORD}u;
    if (ok == 0u) {{
        if (t == 0u) {{
            for (var j = 0u; j < 9u; j++) {{ hyp[rec + j] = 0.0; }}
            hyp[rec + 9u] = bitcast<f32>(0x7f800000u);
            hyp[rec + 10u] = 0.0;
        }}
        return;
    }}
    if (p.lmeds == 0u) {{
        {count_thr}
        if (t == 0u) {{
            for (var j = 0u; j < 9u; j++) {{ hyp[rec + j] = hh[j]; }}
            hyp[rec + 9u] = -f32(inliers);
            hyp[rec + 10u] = bitcast<f32>(inliers);
        }}
        return;
    }}
    // The median: the smallest bit pattern b of a non-negative float with more than n / 2 errors
    // at most b.
    var lo = 0u;
    var hi = 0x7f800000u;
    let want = p.n / 2u + 1u;
    for (var step = 0u; step < 31u; step++) {{
        let mid = lo + (hi - lo) / 2u;
{count_mid}
        if (c >= want) {{ hi = mid; }} else {{ lo = mid + 1u; }}
    }}
    let med = bitcast<f32>(hi);
    let sigma = 1.4826 * (1.0 + 5.0 / max(f32(p.n) - 4.0, 1.0));
    let limit = 6.25 * sigma * sigma * med;
{count_lim}
    if (t == 0u) {{
        for (var j = 0u; j < 9u; j++) {{ hyp[rec + j] = hh[j]; }}
        hyp[rec + 9u] = med;
        hyp[rec + 10u] = bitcast<f32>(inliers);
    }}
}}
",
        count_thr = count_code("inliers", "p.thr2"),
        count_mid = count_code("c", "bitcast<f32>(mid)"),
        count_lim = count_code("inliers", "limit"),
    )
}

/// WGSL that counts the points whose squared error is at most `limit` over the workgroup into the
/// uniform `let` named `result`. Inline, not a function: the barriers must be in the entry point
/// (the CPU backend splits a workgroup into phases at the barriers of the entry point).
fn count_code(result: &str, limit: &str) -> String {
    format!(
        "    var cnt_{result} = 0u;
    for (var i = t; i < p.n; i += {GROUP}u) {{ if (err2(i) <= {limit}) {{ cnt_{result} += 1u; }} }}
    red[t] = cnt_{result};
    workgroupBarrier();
    for (var s = {half}u; s > 0u; s = s >> 1u) {{
        if (t < s) {{ red[t] = red[t] + red[t + s]; }}
        workgroupBarrier();
    }}
    let {result} = red[0];
    workgroupBarrier();
",
        half = GROUP / 2
    )
}

/// Arguments: the hypotheses, the homography (9 floats, row-major, scaled to `h22 = 1`), the
/// inlier count, uniform `{n, iterations, ...}`.
fn best_source() -> String {
    format!(
        "struct Params {{ n: u32, iterations: u32, lmeds: u32, thr2: f32, seed: u32, a: u32, b: u32, c: u32 }}
@group(1) @binding(0) var<storage, read> hyp: array<f32>;
@group(1) @binding(1) var<storage, read_write> out_h: array<f32>;
@group(1) @binding(2) var<storage, read_write> out_n: array<u32>;
@group(1) @binding(3) var<uniform> p: Params;
var<workgroup> key: array<f32, 256>;
var<workgroup> at: array<u32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) t: u32) {{
    var bk = bitcast<f32>(0x7f800000u);
    var bi = 0u;
    for (var i = t; i < p.iterations; i += 256u) {{
        let k = hyp[i * {RECORD}u + 9u];
        if (k < bk || (k == bk && i < bi)) {{ bk = k; bi = i; }}
    }}
    key[t] = bk;
    at[t] = bi;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {{
        if (t < s) {{
            let k = key[t + s];
            let i = at[t + s];
            if (k < key[t] || (k == key[t] && i < at[t])) {{ key[t] = k; at[t] = i; }}
        }}
        workgroupBarrier();
    }}
    if (t == 0u) {{
        let b = at[0] * {RECORD}u;
        let found = key[0] < bitcast<f32>(0x7f800000u);
        let scale = select(1.0, 1.0 / hyp[b + 8u], found && abs(hyp[b + 8u]) > 1e-20);
        for (var j = 0u; j < 9u; j++) {{ out_h[j] = select(0.0, hyp[b + j] * scale, found); }}
        out_n[0] = select(0u, bitcast<u32>(hyp[b + 10u]), found);
    }}
}}
"
    )
}

/// Estimates the homography that maps the source points to the destination points
/// (ArrayFire's `homography`): `iterations` hypotheses are each computed from four random point
/// pairs (the direct linear transform on normalised coordinates) and scored over all the points,
/// and the best one is returned together with its number of inliers.
///
/// The four arrays hold the `x` and `y` coordinates of `n >= 4` matching points. The result is the
/// homography `H` as a `[3, 3]` array whose nine values in memory order are the rows of the matrix
/// `H` of `(x', y', 1) ~ H (x, y, 1)`, scaled so that `H[2][2] = 1` (ArrayFire returns the same
/// matrix with an arbitrary scale), and the one-element inlier count (a lazy `u32` array: no
/// readback). A set of points without any valid hypothesis (all samples degenerate) gives zeros and
/// 0 inliers.
///
/// With [`HomographyMethod::Ransac`] a point is an inlier if its reprojection is within
/// `inlier_thr` (the L2 distance) of its destination and the hypothesis with the most inliers wins.
/// With [`HomographyMethod::Lmeds`] the winner has the smallest median of the squared
/// reprojection errors and its inliers are the points within 2.5 robust standard deviations
/// (`1.4826 (1 + 5 / (n - 4)) sqrt(median)`), whatever `inlier_thr` is.
///
/// The samples are drawn by a fixed hash of the hypothesis number, so a call is deterministic;
/// ArrayFire's samples differ, so the matrices agree only up to the noise that selects them. The
/// returned matrix is the one of the best sample, not refined over its inliers, as in ArrayFire.
///
/// ```
/// use luma::{homography, Array, HomographyMethod};
/// # fn main() -> Result<(), luma::LumaError> {
/// // A translation by (3, -2) of eight points.
/// let x: Vec<f32> = (0..8).map(|i| (i * 7 % 11) as f32).collect();
/// let y: Vec<f32> = (0..8).map(|i| (i * 5 % 13) as f32).collect();
/// let xd: Vec<f32> = x.iter().map(|v| v + 3.0).collect();
/// let yd: Vec<f32> = y.iter().map(|v| v - 2.0).collect();
/// let (h, inliers) = homography(
///     &Array::from_slice(&x, 8)?, &Array::from_slice(&y, 8)?,
///     &Array::from_slice(&xd, 8)?, &Array::from_slice(&yd, 8)?,
///     HomographyMethod::Ransac, 0.5, 50,
/// );
/// assert_eq!(inliers.item()?, 8);
/// let h = h.to_vec()?;
/// assert!((h[2] - 3.0).abs() < 1e-3 && (h[5] + 2.0).abs() < 1e-3);
/// # Ok(()) }
/// ```
#[track_caller]
pub fn homography(x_src: &Array<f32>, y_src: &Array<f32>, x_dst: &Array<f32>, y_dst: &Array<f32>, method: HomographyMethod, inlier_thr: f32, iterations: u32) -> (Array<f32>, Array<u32>) {
    let location = Location::caller();
    let ctx = x_src.context();
    match run(x_src, y_src, x_dst, y_dst, method, inlier_thr, iterations) {
        Ok(pair) => pair,
        Err(failure) => {
            let error = super::poison(failure, "homography", location);
            (Array::poisoned(ctx, Shape::new([3, 3, 1, 1]), error.clone()), Array::poisoned(ctx, Shape::from(1usize), error))
        }
    }
}

fn run(x_src: &Array<f32>, y_src: &Array<f32>, x_dst: &Array<f32>, y_dst: &Array<f32>, method: HomographyMethod, thr: f32, iterations: u32) -> Outcome<(Array<f32>, Array<u32>)> {
    let ctx = x_src.context();
    let n = x_src.len();
    for other in [y_src, x_dst, y_dst] {
        if other.len() != n {
            return Err(LumaError::LengthMismatch { expected: n, actual: other.len() }.into());
        }
        if !ctx.same_as(other.context()) {
            return Err(LumaError::ContextMismatch.into());
        }
    }
    if n < 4 {
        return Err(invalid(format!("homography needs at least 4 point pairs, got {n}")));
    }
    if iterations == 0 {
        return Err(invalid("homography needs at least one iteration"));
    }
    if thr.is_nan() || thr < 0.0 {
        return Err(invalid(format!("homography: inlier threshold {thr} is not a distance")));
    }
    let pins = [x_src.ready()?, y_src.ready()?, x_dst.ready()?, y_dst.ready()?];
    let h = Array::<f32>::alloc(ctx, Shape::new([3, 3, 1, 1]))?;
    let inliers = Array::<u32>::alloc(ctx, Shape::from(1usize))?;
    let (h_pin, n_pin) = (h.ready()?, inliers.ready()?);
    let mut scratch = Scratch::new(ctx);
    let hyp_bytes = u64::from(iterations) * u64::from(RECORD) * 4;
    let hyp = scratch.take(hyp_bytes)?;
    let params = [to_u32(n, "point count")?, iterations, u32::from(method == HomographyMethod::Lmeds), (thr * thr).to_bits(), 0x9e37_79b9, 0, 0, 0];
    let kernel = ctx.builtin(Builtin::Vision(Any::Homography(Kernel::Evaluate)), "f32")?;
    let args = [
        RawArg::read(pins[0].0.buffer(), pins[0].1),
        RawArg::read(pins[1].0.buffer(), pins[1].1),
        RawArg::read(pins[2].0.buffer(), pins[2].1),
        RawArg::read(pins[3].0.buffer(), pins[3].1),
        RawArg::write(&hyp, hyp_bytes),
        RawArg::Uniform(bytemuck::bytes_of(&params)),
    ];
    ctx.launch(&kernel, fold_workgroups(ctx, iterations)?, [iterations * GROUP, 1, 1], &args)?;
    let kernel = ctx.builtin(Builtin::Vision(Any::Homography(Kernel::Best)), "f32")?;
    let args = [
        RawArg::read(&hyp, hyp_bytes),
        RawArg::write(h_pin.0.buffer(), h_pin.1),
        RawArg::write(n_pin.0.buffer(), n_pin.1),
        RawArg::Uniform(bytemuck::bytes_of(&params)),
    ];
    ctx.launch(&kernel, [1, 1, 1], [256, 1, 1], &args)?;
    Ok((h, inliers))
}
