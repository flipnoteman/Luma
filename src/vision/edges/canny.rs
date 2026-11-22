//! The Canny edge detector (`Array::canny`).
//!
//! ArrayFire's pipeline, with its CUDA kernels as the reference:
//!
//! 1. smoothing with the separable 5-tap Gaussian `[0.11021, 0.23691, 0.30576, 0.23691,
//!    0.11021]` (zero outside the image), the 3 x 3 Sobel derivatives of the smoothed image (the
//!    border pixels mirrored without repeating the edge, `reflect101`) and the magnitude
//!    `sqrt(gx^2 + gy^2)` or, with `fast`, `|gx| + |gy|`;
//! 2. non-maximal suppression of the interior pixels (not the outermost ring): a pixel stays if
//!    its magnitude is strictly larger than both neighbours' along the gradient direction,
//!    linearly interpolated between the two pixels the direction falls between; every other
//!    pixel, and the ring, becomes 0;
//! 3. thresholds: `low` and `high` are ratios of the largest suppressed magnitude of the **whole
//!    array** (batch included); with Otsu the high threshold is the Otsu threshold of the
//!    histogram of the suppressed magnitudes of each plane (`floor(max)` bins over `0..=max`,
//!    zeros counted) and the low one is `low` times it;
//! 4. hysteresis: pixels at or above the high threshold are edges, so are the pixels at or above
//!    the low threshold that are 8-connected to an edge through such pixels.
//!
//! Launch plan (no synchronisation, nothing read back): (1) and (2) fused into one tiled kernel
//! that also takes the maximum; for Otsu a histogram kernel and one workgroup per plane that
//! picks the thresholds, for manual thresholds one that scales them; then the hysteresis as a
//! union-find over the candidate pixels (pixels at or above the low threshold, interior):
//! *label* finds the components inside every 32 x 8 tile in workgroup memory and records which
//! contain a strong pixel, *merge* links the components of neighbouring tiles (lock-free, larger
//! root under smaller), *mark* gives the strong flag to the final roots and *output* writes the
//! packed bits of the result (the 4 hysteresis kernels visit one invocation per 32 pixels, the
//! candidates are found through a bit mask, so their cost follows the number of edge pixels
//! rather than of pixels).

use std::fmt::Write as _;
use std::panic::Location;

use super::{geometry, plane_grid};
use crate::array::{Array, Failure};
use crate::builtin::Builtin;
use crate::context::{RawArg, Scratch};
use crate::ops::{fold_workgroups, Outcome};
use crate::vision::{invalid, tile_grid, Kernel as Any};

/// How `Array::canny` finds its high threshold (ArrayFire's `af_canny_threshold`).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum CannyThreshold {
    /// `low` and `high` are ratios of the largest suppressed gradient magnitude.
    Manual,
    /// The high threshold is Otsu's threshold of the magnitudes; `low` is the ratio of the high
    /// threshold the low one has; `high` is ignored.
    AutoOtsu,
}

const TX: u32 = 32;
const TY: u32 = 8;
const GROUP: u32 = 256;
/// Most bins of the Otsu histogram (ArrayFire uses `floor(max)`; images of 8 bits stay far below).
const BINS: u32 = 16384;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum Kernel {
    /// Smoothing, Sobel, magnitude and suppression.
    Suppress,
    /// The histogram of the suppressed magnitudes.
    Histogram,
    /// The thresholds of every plane (Otsu or scaled).
    Thresholds { otsu: bool },
    /// Components inside tiles.
    Label,
    /// Links the components of neighbouring tiles.
    Merge,
    /// Gives the roots of the components with a strong pixel the strong flag.
    Mark,
    /// Writes the result.
    Output,
}

pub(crate) fn source(k: Kernel) -> String {
    match k {
        Kernel::Suppress => suppress_source(),
        Kernel::Histogram => histogram_source(),
        Kernel::Thresholds { otsu } => thresholds_source(otsu),
        Kernel::Label => label_source(),
        Kernel::Merge => merge_source(),
        Kernel::Mark => mark_source(),
        Kernel::Output => output_source(),
    }
}

const STRUCT: &str = "struct Params { h: u32, w: u32, planes: u32, tiles_y: u32, total: u32, words: u32, fast: u32, bins: u32, low: f32, high: f32, a: u32, b: u32 }\n";

const G: [f32; 5] = [0.11021, 0.23691, 0.30576, 0.23691, 0.11021];

/// Arguments: image, suppressed magnitudes, maximum (one word), uniform.
fn suppress_source() -> String {
    let (rw, rh) = (TX + 8, TY + 8); // raw image: the tile and 4 around it
    let (hw, hh) = (TX + 4, TY + 8); // smoothed along axis 0: 2 around in axis 0, 4 in axis 1
    let (sw, sh) = (TX + 4, TY + 4); // smoothed: 2 around
    let (mw, mh) = (TX + 2, TY + 2); // magnitudes: 1 around
    let mut s = String::from(STRUCT);
    s.push_str("@group(1) @binding(0) var<storage, read> src: array<f32>;\n@group(1) @binding(1) var<storage, read_write> sup: array<f32>;\n@group(1) @binding(2) var<storage, read_write> gmax: array<atomic<u32>>;\n@group(1) @binding(3) var<uniform> p: Params;\n");
    let _ = write!(
        s,
        "var<workgroup> raw: array<f32, {raw}>;
var<workgroup> hs: array<f32, {hs}>;
var<workgroup> sm: array<f32, {sm}>;
var<workgroup> gxs: array<f32, {mg}>;
var<workgroup> gys: array<f32, {mg}>;
var<workgroup> mag: array<f32, {mg}>;
var<workgroup> wgmax: atomic<u32>;
const G0: f32 = {g0:?}; const G1: f32 = {g1:?}; const G2: f32 = {g2:?};
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) t: u32, @builtin(workgroup_id) wid: vec3<u32>) {{
    let ty = wid.y % p.tiles_y;
    let plane = wid.y / p.tiles_y;
    let h = i32(p.h);
    let w = i32(p.w);
    let base = plane * p.h * p.w;
    let tx0 = i32(wid.x) * {TX};
    let ty0 = i32(ty) * {TY};
    if (t == 0u) {{ atomicStore(&wgmax, 0u); }}
    for (var i = t; i < {raw}u; i += 256u) {{
        let x = tx0 - 4 + i32(i % {rw}u);
        let y = ty0 - 4 + i32(i / {rw}u);
        var v = 0.0;
        if (x >= 0 && x < h && y >= 0 && y < w) {{ v = src[base + u32(x) + p.h * u32(y)]; }}
        raw[i] = v;
    }}
    workgroupBarrier();
    for (var i = t; i < {hs}u; i += 256u) {{
        let c = i % {hw}u;
        let r = i / {hw}u;
        let o = r * {rw}u + c;
        hs[i] = fma(raw[o], G0, fma(raw[o + 1u], G1, fma(raw[o + 2u], G2, fma(raw[o + 3u], G1, raw[o + 4u] * G0))));
    }}
    workgroupBarrier();
    for (var i = t; i < {sm}u; i += 256u) {{
        let c = i % {sw}u;
        let r = i / {sw}u;
        let o = r * {hw}u + c;
        sm[i] = fma(hs[o], G0, fma(hs[o + {hw}u], G1, fma(hs[o + 2u * {hw}u], G2, fma(hs[o + 3u * {hw}u], G1, hs[o + 4u * {hw}u] * G0))));
    }}
    workgroupBarrier();
    for (var i = t; i < {mg}u; i += 256u) {{
        let c = i % {mw}u;
        let r = i / {mw}u;
        let x = tx0 - 1 + i32(c);
        let y = ty0 - 1 + i32(r);
        // The smoothed neighbours, mirrored at the image border (`reflect101`).
        let cc = c + 1u;
        let rr = r + 1u;
        let xl = select(cc - 1u, cc + 1u, x == 0);
        let xr = select(cc + 1u, cc - 1u, x == h - 1);
        let yu = select(rr - 1u, rr + 1u, y == 0);
        let yd = select(rr + 1u, rr - 1u, y == w - 1);
        let ll = sm[yu * {sw}u + xl]; let lc = sm[rr * {sw}u + xl]; let lr = sm[yd * {sw}u + xl];
        let cl = sm[yu * {sw}u + cc]; let cr = sm[yd * {sw}u + cc];
        let rl = sm[yu * {sw}u + xr]; let rc = sm[rr * {sw}u + xr]; let rd = sm[yd * {sw}u + xr];
        let dx = ((rl + rd) - (ll + lr)) + 2.0 * (rc - lc);
        let dy = ((lr + rd) - (ll + rl)) + 2.0 * (cr - cl);
        gxs[i] = dx;
        gys[i] = dy;
        if (p.fast != 0u) {{ mag[i] = abs(dx) + abs(dy); }} else {{ mag[i] = sqrt(dx * dx + dy * dy); }}
    }}
    workgroupBarrier();
    let lx = t & 31u;
    let ly = t >> 5u;
    let x = tx0 + i32(lx);
    let y = ty0 + i32(ly);
    if (x < h && y < w) {{
        var out = 0.0;
        if (x >= 1 && x < h - 1 && y >= 1 && y < w - 1) {{
            let i = (ly + 1u) * {mw}u + lx + 1u;
            let cmag = mag[i];
            let dx = gxs[i];
            let dy = gys[i];
            let se = mag[i + {mw}u + 1u];
            let nw = mag[i - {mw}u - 1u];
            let ea = mag[i + 1u];
            let we = mag[i - 1u];
            let ne = mag[i - {mw}u + 1u];
            let sw = mag[i + {mw}u - 1u];
            let no = mag[i - {mw}u];
            let so = mag[i + {mw}u];
            var a1 = 0.0; var a2 = 0.0; var b1 = 0.0; var b2 = 0.0; var alpha = 0.0;
            if (cmag > 0.0) {{
                if (dx >= 0.0) {{
                    if (dy >= 0.0) {{
                        let big = (dx - dy) >= 0.0;
                        a1 = select(so, ea, big); a2 = select(no, we, big); b1 = se; b2 = nw;
                        alpha = select(dx / dy, dy / dx, big);
                    }} else {{
                        let big = (dx + dy) >= 0.0;
                        a1 = select(no, ea, big); a2 = select(so, we, big); b1 = ne; b2 = sw;
                        alpha = select(dx / -dy, -dy / dx, big);
                    }}
                }} else {{
                    if (dy >= 0.0) {{
                        let big = (dx + dy) >= 0.0;
                        a1 = select(we, so, big); a2 = select(ea, no, big); b1 = sw; b2 = ne;
                        alpha = select(dy / -dx, -dx / dy, big);
                    }} else {{
                        let big = (-dx + dy) >= 0.0;
                        a1 = select(no, we, big); a2 = select(so, ea, big); b1 = nw; b2 = se;
                        alpha = select(dx / dy, dy / dx, big);
                    }}
                }}
                let mag1 = (1.0 - alpha) * a1 + alpha * b1;
                let mag2 = (1.0 - alpha) * a2 + alpha * b2;
                if (cmag > mag1 && cmag > mag2) {{ out = cmag; }}
            }}
        }}
        sup[base + u32(x) + p.h * u32(y)] = out;
        atomicMax(&wgmax, bitcast<u32>(out));
    }}
    workgroupBarrier();
    if (t == 0u) {{
        let m = atomicLoad(&wgmax);
        if (m != 0u) {{ atomicMax(&gmax[0], m); }}
    }}
}}
",
        raw = rw * rh,
        hs = hw * hh,
        sm = sw * sh,
        mg = mw * mh,
        g0 = G[0],
        g1 = G[1],
        g2 = G[2]
    );
    s
}

/// Arguments: suppressed magnitudes, maximum, histogram (`bins` words per plane), uniform.
fn histogram_source() -> String {
    let mut s = String::from(STRUCT);
    s.push_str("@group(1) @binding(0) var<storage, read> sup: array<f32>;\n@group(1) @binding(1) var<storage, read> gmax: array<u32>;\n@group(1) @binding(2) var<storage, read_write> hist: array<atomic<u32>>;\n@group(1) @binding(3) var<uniform> p: Params;\n");
    let _ = write!(
        s,
        "@compute @workgroup_size({GROUP})
fn main(@builtin(local_invocation_index) t: u32, @builtin(workgroup_id) wid: vec3<u32>) {{
    let mx = bitcast<f32>(gmax[0]);
    if (!(mx > 0.0)) {{ return; }}
    let nb = clamp(u32(mx), 1u, p.bins);
    let step = mx / f32(nb);
    let n = p.h * p.w;
    let base = wid.y * n;
    for (var k = 0u; k < 8u; k++) {{
        let i = (wid.x * 8u + k) * {GROUP}u + t;
        if (i < n) {{
            let v = sup[base + i];
            // Zeros are not counted (the Otsu kernel adds them back).
            if (v > 0.0) {{
                let b = u32(clamp(i32(v / step), 0, i32(nb) - 1));
                atomicAdd(&hist[wid.y * p.bins + b], 1u);
            }}
        }}
    }}
}}
"
    );
    s
}

/// Arguments: maximum, histogram, thresholds (`[low, high]` per plane), uniform. One workgroup
/// per plane.
fn thresholds_source(otsu: bool) -> String {
    let mut s = String::from(STRUCT);
    if !otsu {
        // Arguments: maximum, thresholds, uniform.
        s.push_str("@group(1) @binding(0) var<storage, read> gmax: array<u32>;\n@group(1) @binding(1) var<storage, read_write> thr: array<f32>;\n@group(1) @binding(2) var<uniform> p: Params;\n");
        s.push_str(
            "@compute @workgroup_size(1)
fn main(@builtin(workgroup_id) wid: vec3<u32>) {
    let mx = bitcast<f32>(gmax[0]);
    thr[wid.x * 2u] = p.low * mx;
    thr[wid.x * 2u + 1u] = p.high * mx;
}
",
        );
        return s;
    }
    s.push_str("@group(1) @binding(0) var<storage, read> gmax: array<u32>;\n@group(1) @binding(1) var<storage, read> hist: array<u32>;\n@group(1) @binding(2) var<storage, read_write> thr: array<f32>;\n@group(1) @binding(3) var<uniform> p: Params;\n");
    let _ = write!(
        s,
        "var<workgroup> cnt: array<u32, {GROUP}>;
var<workgroup> sum: array<f32, {GROUP}>;
var<workgroup> best_v: array<f32, {GROUP}>;
var<workgroup> best_i: array<u32, {GROUP}>;
@compute @workgroup_size({GROUP})
fn main(@builtin(local_invocation_index) t: u32, @builtin(workgroup_id) wid: vec3<u32>) {{
    let mx = bitcast<f32>(gmax[0]);
    let nb = clamp(u32(max(mx, 0.0)), 1u, p.bins);
    let step = mx / f32(nb);
    let n = p.h * p.w;
    let base = wid.x * p.bins;
    let per = (nb + {GROUP}u - 1u) / {GROUP}u;
    let lo = min(t * per, nb);
    let hi = min(lo + per, nb);
    // Counts and sums of bin * count of the bins of this thread (zero pixels are not in the
    // histogram: they are the pixels not counted, all in bin 0).
    var c = 0u;
    var sm = 0.0;
    for (var i = lo; i < hi; i++) {{
        let hc = hist[base + i];
        c += hc;
        sm += f32(i) * f32(hc);
    }}
    cnt[t] = c;
    sum[t] = sm;
    workgroupBarrier();
    for (var d = 1u; d < {GROUP}u; d = d << 1u) {{
        var ac = 0u;
        var as_ = 0.0;
        if (t >= d) {{ ac = cnt[t - d]; as_ = sum[t - d]; }}
        workgroupBarrier();
        cnt[t] = cnt[t] + ac;
        sum[t] = sum[t] + as_;
        workgroupBarrier();
    }}
    let zeros = n - cnt[{last}];
    let total_sum = sum[{last}];
    var n0 = zeros;
    var s0 = 0.0;
    if (t > 0u) {{ n0 += cnt[t - 1u]; s0 = sum[t - 1u]; }}
    var bv = -1.0;
    var bi = 0u;
    let nf = f32(n);
    for (var i = lo; i < hi; i++) {{
        let hc = hist[base + i];
        n0 += hc;
        s0 += f32(i) * f32(hc);
        if (i + 1u < nb) {{
            let n1 = n - n0;
            if (n0 > 0u && n1 > 0u) {{
                let w0 = f32(n0) / nf;
                let w1 = f32(n1) / nf;
                let mu0 = s0 / f32(n0);
                let mu1 = (total_sum - s0) / f32(n1);
                let sb = w0 * w1 * (mu0 - mu1) * (mu0 - mu1);
                if (sb > bv) {{ bv = sb; bi = i; }}
            }}
        }}
    }}
    best_v[t] = bv;
    best_i[t] = bi;
    workgroupBarrier();
    for (var d = {half}u; d > 0u; d = d >> 1u) {{
        if (t < d) {{
            let ov = best_v[t + d];
            let oi = best_i[t + d];
            if (ov > best_v[t] || (ov == best_v[t] && oi < best_i[t])) {{ best_v[t] = ov; best_i[t] = oi; }}
        }}
        workgroupBarrier();
    }}
    if (t == 0u) {{
        var high = f32(best_i[0]) * step;
        if (best_v[0] < 0.0) {{ high = 0.0; }}
        thr[wid.x * 2u] = p.low * high;
        thr[wid.x * 2u + 1u] = high;
    }}
}}
",
        last = GROUP - 1,
        half = GROUP / 2
    );
    s
}

/// The state of the pixel at `idx` of the suppressed image: 0 not a candidate, 1 weak, 2 strong.
fn state_fn() -> &'static str {
    "fn state_of(v: f32, t1: f32, t2: f32) -> u32 {
    if (v >= t2) { return 2u; }
    if (v >= t1) { return 1u; }
    return 0u;
}
"
}

/// Arguments: suppressed magnitudes, thresholds, parents, candidate bits, flag bits, uniform.
fn label_source() -> String {
    let mut s = String::from(STRUCT);
    s.push_str("@group(1) @binding(0) var<storage, read> sup: array<f32>;\n@group(1) @binding(1) var<storage, read> thr: array<f32>;\n@group(1) @binding(2) var<storage, read_write> parent: array<u32>;\n@group(1) @binding(3) var<storage, read_write> cand: array<atomic<u32>>;\n@group(1) @binding(4) var<storage, read_write> flags: array<atomic<u32>>;\n@group(1) @binding(5) var<uniform> p: Params;\n");
    s.push_str(state_fn());
    let _ = write!(
        s,
        "var<workgroup> lab: array<atomic<u32>, {GROUP}>;
var<workgroup> chg: atomic<u32>;
var<workgroup> anyc: atomic<u32>;
@compute @workgroup_size({GROUP})
fn main(@builtin(local_invocation_index) t: u32, @builtin(workgroup_id) wid: vec3<u32>) {{
    let ty = wid.y % p.tiles_y;
    let plane = wid.y / p.tiles_y;
    let lx = t & 31u;
    let ly = t >> 5u;
    let x = wid.x * {TX}u + lx;
    let y = ty * {TY}u + ly;
    let idx = plane * p.h * p.w + x + p.h * y;
    var st = 0u;
    if (x >= 1u && x + 1u < p.h && y >= 1u && y + 1u < p.w) {{
        st = state_of(sup[idx], thr[plane * 2u], thr[plane * 2u + 1u]);
    }}
    atomicStore(&lab[t], select(0xffffffffu, idx, st != 0u));
    if (t == 0u) {{ atomicStore(&anyc, 0u); }}
    workgroupBarrier();
    if (st != 0u) {{ atomicStore(&anyc, 1u); }}
    if (workgroupUniformLoad(&anyc) == 0u) {{ return; }}
    loop {{
        if (t == 0u) {{ atomicStore(&chg, 0u); }}
        workgroupBarrier();
        if (st != 0u) {{
            let cur = atomicLoad(&lab[t]);
            var m = cur;
            for (var dy = -1; dy <= 1; dy++) {{
                for (var dx = -1; dx <= 1; dx++) {{
                    let nx = i32(lx) + dx;
                    let ny = i32(ly) + dy;
                    if (nx >= 0 && nx < {TX} && ny >= 0 && ny < {TY}) {{ m = min(m, atomicLoad(&lab[u32(ny) * {TX}u + u32(nx)])); }}
                }}
            }}
            if (m < cur) {{ atomicStore(&lab[t], m); atomicStore(&chg, 1u); }}
        }}
        if (workgroupUniformLoad(&chg) == 0u) {{ break; }}
    }}
    if (st != 0u) {{
        let root = atomicLoad(&lab[t]);
        parent[idx] = root;
        atomicOr(&cand[idx >> 5u], 1u << (idx & 31u));
        if (st == 2u) {{ atomicOr(&flags[root >> 5u], 1u << (root & 31u)); }}
    }}
}}
"
    );
    s
}

/// Arguments: parents, candidate bits, uniform.
fn merge_source() -> String {
    let mut s = String::from(STRUCT);
    s.push_str("@group(1) @binding(0) var<storage, read_write> parent: array<atomic<u32>>;\n@group(1) @binding(1) var<storage, read> cand: array<u32>;\n@group(1) @binding(2) var<uniform> p: Params;\n");
    let _ = write!(
        s,
        "fn find(i: u32) -> u32 {{
    var r = i;
    loop {{
        let q = atomicLoad(&parent[r]);
        if (q == r) {{ break; }}
        r = q;
    }}
    return r;
}}
fn is_cand(i: u32) -> bool {{ return ((cand[i >> 5u] >> (i & 31u)) & 1u) != 0u; }}
fn link(a0: u32, b0: u32) {{
    var a = find(a0);
    var b = find(b0);
    loop {{
        if (a == b) {{ break; }}
        // The larger root goes under the smaller one.
        let hi = max(a, b);
        let lo = min(a, b);
        let r = atomicCompareExchangeWeak(&parent[hi], hi, lo);
        if (r.exchanged) {{ break; }}
        a = find(hi);
        b = find(lo);
    }}
}}
@compute @workgroup_size({GROUP})
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {{
    let wi = gid.x + gid.y * nwg.x * {GROUP}u;
    if (wi >= p.words) {{ return; }}
    var bits = cand[wi];
    let hw = p.h * p.w;
    while (bits != 0u) {{
        let b = firstTrailingBit(bits);
        bits = bits & (bits - 1u);
        let idx = wi * 32u + b;
        let plane = idx / hw;
        let loc = idx - plane * hw;
        let x = i32(loc % p.h);
        let y = i32(loc / p.h);
        // The four neighbours before the pixel; the ones inside the tile are linked already.
        for (var k = 0; k < 4; k++) {{
            var qx = x - 1;
            var qy = y;
            if (k == 1) {{ qy = y - 1; }}
            if (k == 2) {{ qx = x; qy = y - 1; }}
            if (k == 3) {{ qx = x + 1; qy = y - 1; }}
            if (qx < 0 || qy < 0 || qx >= i32(p.h)) {{ continue; }}
            if ((qx >> 5u) == (x >> 5u) && (qy >> 3u) == (y >> 3u)) {{ continue; }}
            let q = plane * hw + u32(qx) + p.h * u32(qy);
            if (is_cand(q)) {{ link(idx, q); }}
        }}
    }}
}}
"
    );
    s
}

/// Arguments: parents, flag bits, uniform.
fn mark_source() -> String {
    let mut s = String::from(STRUCT);
    s.push_str("@group(1) @binding(0) var<storage, read> parent: array<u32>;\n@group(1) @binding(1) var<storage, read_write> flags: array<atomic<u32>>;\n@group(1) @binding(2) var<uniform> p: Params;\n");
    let _ = write!(
        s,
        "@compute @workgroup_size({GROUP})
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {{
    let wi = gid.x + gid.y * nwg.x * {GROUP}u;
    if (wi >= p.words) {{ return; }}
    var bits = atomicLoad(&flags[wi]);
    while (bits != 0u) {{
        let b = firstTrailingBit(bits);
        bits = bits & (bits - 1u);
        let idx = wi * 32u + b;
        var r = idx;
        loop {{
            let q = parent[r];
            if (q == r) {{ break; }}
            r = q;
        }}
        if (r != idx) {{ atomicOr(&flags[r >> 5u], 1u << (r & 31u)); }}
    }}
}}
"
    );
    s
}

/// Arguments: parents, candidate bits, flag bits, result words, uniform.
fn output_source() -> String {
    let mut s = String::from(STRUCT);
    s.push_str("@group(1) @binding(0) var<storage, read> parent: array<u32>;\n@group(1) @binding(1) var<storage, read> cand: array<u32>;\n@group(1) @binding(2) var<storage, read> flags: array<u32>;\n@group(1) @binding(3) var<storage, read_write> out: array<u32>;\n@group(1) @binding(4) var<uniform> p: Params;\n");
    let _ = write!(
        s,
        "@compute @workgroup_size({GROUP})
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {{
    let wi = gid.x + gid.y * nwg.x * {GROUP}u;
    if (wi >= p.words) {{ return; }}
    var bits = cand[wi];
    var res = 0u;
    while (bits != 0u) {{
        let b = firstTrailingBit(bits);
        bits = bits & (bits - 1u);
        var r = wi * 32u + b;
        loop {{
            let q = parent[r];
            if (q == r) {{ break; }}
            r = q;
        }}
        if (((flags[r >> 5u] >> (r & 31u)) & 1u) != 0u) {{ res |= 1u << b; }}
    }}
    out[wi] = res;
}}
"
    );
    s
}

impl Array<f32> {
    /// The edges of `[H, W]` images (ArrayFire's `canny`; axes 2 and 3 are a batch) as a boolean
    /// array of the same shape.
    ///
    /// The image is smoothed with a 5-tap Gaussian, differentiated with the 3 x 3 Sobel
    /// operator, and the gradient magnitude (`sqrt(gx^2 + gy^2)`, or `|gx| + |gy|` with `fast`)
    /// is thinned: an interior pixel keeps its magnitude only if it is strictly larger than the
    /// two neighbours along the gradient (interpolated between the two pixels the direction
    /// falls between). Then the hysteresis threshold: pixels at or above the high threshold are
    /// edges, and so are the pixels at or above the low threshold that touch (8-connected, through
    /// such pixels) an edge. The outermost ring of pixels is never an edge.
    ///
    /// With [`CannyThreshold::Manual`] `low` and `high` are *ratios* of the largest thinned
    /// magnitude of the whole array (batch included): the thresholds are `low * max` and
    /// `high * max`. With [`CannyThreshold::AutoOtsu`] the high threshold is found by Otsu's
    /// method on the histogram of the thinned magnitudes of each plane (`floor(max)` bins from 0
    /// to the maximum of the whole array, at least 1 and at most 16384, the zeros of the thinning
    /// counted; the threshold is the lower edge of the first bin that maximises the
    /// between-class variance) and `low` is the ratio of that threshold the low one has; `high`
    /// is ignored. Because the bin width follows `floor(max)`, Otsu's result depends on the
    /// scale of the image (ArrayFire's behaviour: use images in `0..255`).
    ///
    /// Only `sobel_window == 3` exists: ArrayFire rejects every other value, so does this (the
    /// result is poisoned). Thresholds must be finite.
    ///
    /// Differences from ArrayFire: none intended. The CUDA backend of ArrayFire is the reference;
    /// its CPU backend marks the whole outer ring and mixes up the axes on images that are not
    /// square. Values that tie in exact arithmetic may fall on either side in `f32`, so pixels
    /// of a plateau (flat steps) can differ. A constant image makes ArrayFire fail; here
    /// it has edges near the border only (the smoothing pads with zeros) and none inside.
    ///
    /// Lazy: nothing is read back; 7 launches with Otsu thresholds, 6 with manual ones (the
    /// working buffers are an `H x W` plane of `f32` per plane, the same of parents and a few
    /// bit masks).
    ///
    /// ```
    /// use luma::{Array, CannyThreshold};
    /// # fn main() -> Result<(), luma::LumaError> {
    /// // A soft vertical edge near column 9 of a 20 x 20 image (a perfectly symmetric profile
    /// // would tie between two columns and have no strict maximum).
    /// let data: Vec<f32> = (0..400).map(|k| 100.0 * (1.0 + (((k / 20) as f32 - 9.2) / 1.5).tanh())).collect();
    /// let img = Array::from_slice(&data, [20, 20])?;
    /// let edges = img.canny(CannyThreshold::Manual, 0.2, 0.5, 3, false).to_vec()?;
    /// for i in 3..17 {
    ///     assert!(edges[i + 20 * 9], "row {i}");
    ///     assert!(!edges[i + 20 * 5]);
    /// }
    /// # Ok(()) }
    /// ```
    #[track_caller]
    pub fn canny(&self, threshold: CannyThreshold, low: f32, high: f32, sobel_window: u32, fast: bool) -> Array<bool> {
        let location = Location::caller();
        Array::resolve(self.context(), self.shape(), run(self, threshold, low, high, sobel_window, fast), "canny", location)
    }
}

fn run(input: &Array<f32>, threshold: CannyThreshold, low: f32, high: f32, window: u32, fast: bool) -> Outcome<Array<bool>> {
    if window != 3 {
        return Err(invalid(format!("canny: the Sobel window must be 3, got {window}")));
    }
    if !(low.is_finite() && high.is_finite()) {
        return Err(invalid(format!("canny: the thresholds must be finite, got {low} and {high}")));
    }
    let ctx = input.context();
    input.storage()?;
    let (h, w, planes) = geometry(input.shape())?;
    let total = h.checked_mul(w).and_then(|v| v.checked_mul(planes)).ok_or_else(|| Failure::from(crate::error::LumaError::DimensionTooLarge { shape: input.shape().dims() }))?;
    if h < 3 || w < 3 {
        // No interior pixel.
        return Ok(Array::<bool>::zeros_on(ctx, input.shape())?);
    }
    let words = total.div_ceil(32);
    let (tx, ty) = (h.div_ceil(TX), w.div_ceil(TY));
    let grid = plane_grid(ctx, tx, ty, planes)?;
    let otsu = threshold == CannyThreshold::AutoOtsu;
    let params: [u32; 12] = [h, w, planes, ty, total, words, u32::from(fast), BINS, low.to_bits(), high.to_bits(), 0, 0];
    let uniform = bytemuck::cast_slice(&params);

    let out = Array::<bool>::alloc(ctx, input.shape())?;
    let (src, src_bytes) = input.ready()?;
    let (out_pin, out_bytes) = out.ready()?;
    let mut scratch = Scratch::new(ctx);
    let sup_bytes = u64::from(total) * 4;
    let sup = scratch.take(sup_bytes)?;
    let parent = scratch.take(sup_bytes)?;
    let bits_bytes = u64::from(words) * 4;
    let cand = scratch.take(bits_bytes)?;
    let flags = scratch.take(bits_bytes)?;
    let gmax = scratch.take(16)?;
    let thr_bytes = u64::from(planes) * 8;
    let thr = scratch.take(thr_bytes)?;
    ctx.clear(&gmax, 16)?;
    ctx.clear(&cand, bits_bytes)?;
    ctx.clear(&flags, bits_bytes)?;
    let kernel = |k: Kernel| ctx.builtin(Builtin::Vision(Any::Edges(super::Kernel::Canny(k))), "f32");
    let words_grid = fold_workgroups(ctx, words.div_ceil(GROUP))?;
    let size = [h, w, planes];

    ctx.launch(&*kernel(Kernel::Suppress)?, grid, size, &[RawArg::read(src.buffer(), src_bytes), RawArg::write(&sup, sup_bytes), RawArg::write(&gmax, 16), RawArg::Uniform(uniform)])?;
    if otsu {
        let hist_bytes = u64::from(planes) * u64::from(BINS) * 4;
        let hist = scratch.take(hist_bytes)?;
        ctx.clear(&hist, hist_bytes)?;
        let n = h * w;
        let hist_grid = tile_grid(ctx, n.div_ceil(GROUP * 8), planes)?;
        ctx.launch(&*kernel(Kernel::Histogram)?, hist_grid, size, &[RawArg::read(&sup, sup_bytes), RawArg::read(&gmax, 16), RawArg::write(&hist, hist_bytes), RawArg::Uniform(uniform)])?;
        ctx.launch(&*kernel(Kernel::Thresholds { otsu: true })?, tile_grid(ctx, planes, 1)?, [GROUP, 1, 1], &[RawArg::read(&gmax, 16), RawArg::read(&hist, hist_bytes), RawArg::write(&thr, thr_bytes), RawArg::Uniform(uniform)])?;
    } else {
        // The histogram is not read.
        ctx.launch(&*kernel(Kernel::Thresholds { otsu: false })?, tile_grid(ctx, planes, 1)?, [1, 1, 1], &[RawArg::read(&gmax, 16), RawArg::write(&thr, thr_bytes), RawArg::Uniform(uniform)])?;
    }
    ctx.launch(
        &*kernel(Kernel::Label)?,
        grid,
        size,
        &[RawArg::read(&sup, sup_bytes), RawArg::read(&thr, thr_bytes), RawArg::write(&parent, sup_bytes), RawArg::write(&cand, bits_bytes), RawArg::write(&flags, bits_bytes), RawArg::Uniform(uniform)],
    )?;
    ctx.launch(&*kernel(Kernel::Merge)?, words_grid, [words, 1, 1], &[RawArg::write(&parent, sup_bytes), RawArg::read(&cand, bits_bytes), RawArg::Uniform(uniform)])?;
    ctx.launch(&*kernel(Kernel::Mark)?, words_grid, [words, 1, 1], &[RawArg::read(&parent, sup_bytes), RawArg::write(&flags, bits_bytes), RawArg::Uniform(uniform)])?;
    ctx.launch(
        &*kernel(Kernel::Output)?,
        words_grid,
        [words, 1, 1],
        &[RawArg::read(&parent, sup_bytes), RawArg::read(&cand, bits_bytes), RawArg::read(&flags, bits_bytes), RawArg::write(out_pin.buffer(), out_bytes), RawArg::Uniform(uniform)],
    )?;
    Ok(out)
}
