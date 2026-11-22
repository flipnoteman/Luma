//! SIFT and GLOH (`Array::sift`, `Array::gloh`): scale-space extrema of a Gaussian pyramid,
//! refined, oriented and described as in ArrayFire (its port of Rob Hess' SIFT).
//!
//! All of the pyramid and the detection is recorded without a readback. The pyramid of an
//! octave is one `[H, W, layers + 3]` array; the difference of Gaussians is never stored, the
//! kernels read two layers of the pyramid. Per octave:
//!
//! 1. **blur**: the layers are filtered by two tiled passes (along axis 0, along axis 1) between
//!    planes of the pyramid; the first layer of an octave is the bilinear half-size resize of the
//!    layer `n_layers` of the previous one;
//! 2. **extrema**: a thread per pixel compares the difference of Gaussians of the layers
//!    `1..=n_layers` with its 26 neighbours and appends the extrema to a list;
//! 3. **interpolation**: a thread per extremum refines it as ArrayFire does, rejects low
//!    contrast and edges, and inserts the final location into a hash table: a location that is
//!    already in it is a duplicate (ArrayFire sorts and compares neighbours instead);
//! 4. **orientation**: a workgroup per surviving point builds the 36-bin histogram of the
//!    gradient directions with integer atomics and appends one oriented point per peak.
//!
//! After all octaves the counts are read back (the only synchronization) and a workgroup per
//! oriented point computes its descriptor (4 x 4 x 8 for SIFT, 17 x 16 for GLOH) into the output.

use std::fmt::Write as _;

use super::{resize, Kernel as Descriptors};
use crate::array::{Array, Failure};
use crate::builtin::Builtin;
use crate::context::RawArg;
use crate::ops::{fold_workgroups, Outcome};
use crate::shape::Shape;
use crate::vision::{invalid, plane, tile_grid, to_u32, Kernel as Any, Keypoints};

/// Width of the border without extrema.
const BORDER: u32 = 5;
/// Bins of the orientation histogram.
const ORI_BINS: u32 = 36;
/// Workgroups of the orientation launch (each loops over its share of the points).
const ORI_GROUPS: u32 = 2048;

/// One kernel of SIFT.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum Kernel {
    /// A pass of the Gaussian blur along axis 0 or 1.
    Blur { axis: u8 },
    /// Extrema of the difference of Gaussians.
    Extrema,
    /// Refinement of the extrema and removal of duplicates.
    Interp,
    /// Orientation histogram and peaks.
    Orient,
    /// Descriptor of an oriented point.
    Describe { gloh: bool },
}

pub(crate) fn source(k: Kernel) -> String {
    match k {
        Kernel::Blur { axis } => blur_source(axis),
        Kernel::Extrema => extrema_source(),
        Kernel::Interp => interp_source(),
        Kernel::Orient => orient_source(),
        Kernel::Describe { gloh } => describe_source(gloh),
    }
}

/// C's `round` (half away from zero).
const RND: &str = "fn rnd(v: f32) -> f32 { return sign(v) * floor(abs(v) + 0.5); }\n";

/// One pass of the separable zero-padded filter between planes. Arguments: source, destination
/// (different buffers), taps, uniform `{h, w, src_off, dst_off, radius, len, 0, 0}`.
fn blur_source(axis: u8) -> String {
    let mut s = String::from(
        "struct Params { h: u32, w: u32, so: u32, dofs: u32, radius: u32, len: u32, a: u32, b: u32 }
@group(1) @binding(0) var<storage, read> src: array<f32>;
@group(1) @binding(1) var<storage, read_write> dst: array<f32>;
@group(1) @binding(2) var<storage, read> taps: array<f32>;
@group(1) @binding(3) var<uniform> p: Params;
var<workgroup> tp: array<f32, 32>;
",
    );
    if axis == 0 {
        s.push_str("var<workgroup> sh: array<f32, 512>;\n");
        s.push_str(
            "@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) t: u32) {
    let row0 = i32(wid.x) * 32;
    let col0 = i32(wid.y) * 8;
    let r = i32(p.radius);
    let h = i32(p.h);
    let w = i32(p.w);
    if (t < 32u) { tp[t] = select(0.0, taps[t], t < p.len); }
    let span = 32u + 2u * p.radius;
    for (var idx = t; idx < 512u; idx += 256u) {
        let lr = idx % 64u;
        let lc = idx / 64u;
        let gr = row0 - r + i32(lr);
        let gc = col0 + i32(lc);
        var v = 0.0;
        if (lr < span && gr >= 0 && gr < h && gc < w) { v = src[p.so + u32(gr) + p.h * u32(gc)]; }
        sh[idx] = v;
    }
    workgroupBarrier();
    let tx = t & 31u;
    let ty = t >> 5u;
    var acc = 0.0;
    for (var k = 0u; k < p.len; k++) { acc += tp[k] * sh[ty * 64u + tx + k]; }
    let row = row0 + i32(tx);
    let col = col0 + i32(ty);
    if (row < h && col < w) { dst[p.dofs + u32(row) + p.h * u32(col)] = acc; }
}
",
        );
    } else {
        s.push_str("var<workgroup> sh: array<f32, 1216>;\n");
        s.push_str(
            "@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) t: u32) {
    let row0 = i32(wid.x) * 32;
    let col0 = i32(wid.y) * 8;
    let r = i32(p.radius);
    let h = i32(p.h);
    let w = i32(p.w);
    if (t < 32u) { tp[t] = select(0.0, taps[t], t < p.len); }
    let total = 32u * (8u + 2u * p.radius);
    for (var idx = t; idx < total; idx += 256u) {
        let lr = idx % 32u;
        let lc = idx / 32u;
        let gr = row0 + i32(lr);
        let gc = col0 - r + i32(lc);
        var v = 0.0;
        if (gr < h && gc >= 0 && gc < w) { v = src[p.so + u32(gr) + p.h * u32(gc)]; }
        sh[idx] = v;
    }
    workgroupBarrier();
    let tx = t & 31u;
    let ty = t >> 5u;
    var acc = 0.0;
    for (var k = 0u; k < p.len; k++) { acc += tp[k] * sh[(ty + k) * 32u + tx]; }
    let row = row0 + i32(tx);
    let col = col0 + i32(ty);
    if (row < h && col < w) { dst[p.dofs + u32(row) + p.h * u32(col)] = acc; }
}
",
        );
    }
    s
}

/// The difference of Gaussians of layer `l` at (row, col) of an octave stack `g`.
const DOG_FN: &str = "fn dog(l: u32, r: i32, c: i32) -> f32 {
    let o = u32(c) * p.h + u32(r);
    return g[(l + 1u) * p.plane + o] - g[l * p.plane + o];
}
";

/// Extrema of the difference of Gaussians. A workgroup takes a tile of 32 rows by 8 columns and
/// walks up the layers: the pyramid layers (with a one-pixel halo) enter workgroup memory one at
/// a time, the differences of Gaussians of three consecutive layers are kept and the middle one
/// is searched. Arguments: pyramid of the octave, positions, layers, counters, uniform
/// `{h, w, n_layers, max_feat, threshold, plane, counter, 0}`.
fn extrema_source() -> String {
    format!(
        "struct Params {{ h: u32, w: u32, nl: u32, max_feat: u32, thr: f32, plane: u32, ctr: u32, c: u32 }}
@group(1) @binding(0) var<storage, read> g: array<f32>;
@group(1) @binding(1) var<storage, read_write> cand_pos: array<u32>;
@group(1) @binding(2) var<storage, read_write> cand_layer: array<u32>;
@group(1) @binding(3) var<storage, read_write> counters: array<atomic<u32>>;
@group(1) @binding(4) var<uniform> p: Params;
var<workgroup> gs: array<f32, 680>;
var<workgroup> dt: array<f32, 1020>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) t: u32) {{
    let row0 = i32({BORDER}u + wid.x * 32u);
    let col0 = i32({BORDER}u + wid.y * 8u);
    let h = i32(p.h);
    let w = i32(p.w);
    let tx = t & 31u;
    let ty = t >> 5u;
    let row = row0 + i32(tx);
    let col = col0 + i32(ty);
    let valid = row + {BORDER} < h && col + {BORDER} < w;
    // The tile of layer 0.
    for (var e = t; e < 340u; e += 256u) {{
        let gr = row0 - 1 + i32(e % 34u);
        let gc = col0 - 1 + i32(e / 34u);
        var v = 0.0;
        if (gr < h && gc < w) {{ v = g[u32(gc) * p.h + u32(gr)]; }}
        gs[e] = v;
    }}
    for (var l = 0u; l <= p.nl + 1u; l++) {{
        for (var e = t; e < 340u; e += 256u) {{
            let gr = row0 - 1 + i32(e % 34u);
            let gc = col0 - 1 + i32(e / 34u);
            var v = 0.0;
            if (gr < h && gc < w) {{ v = g[(l + 1u) * p.plane + u32(gc) * p.h + u32(gr)]; }}
            gs[((l + 1u) % 2u) * 340u + e] = v;
            dt[(l % 3u) * 340u + e] = v - gs[(l % 2u) * 340u + e];
        }}
        workgroupBarrier();
        if (l >= 2u && valid) {{
            let cl = l - 1u;
            let cb = (cl % 3u) * 340u;
            let c = ty * 34u + tx + 35u;
            let pv = dt[cb + c];
            if (abs(pv) > p.thr) {{
                var ok = true;
                for (var k = 0u; k < 27u; k++) {{
                    if (k == 13u) {{ continue; }}
                    let q = dt[((cl + k / 9u + 2u) % 3u) * 340u + c + (k / 3u) % 3u - 1u + ((k % 3u) * 34u) - 34u];
                    if (pv > 0.0) {{
                        if (!(pv > q)) {{ ok = false; break; }}
                    }} else {{
                        if (!(pv < q)) {{ ok = false; break; }}
                    }}
                }}
                if (ok) {{
                    let idx = atomicAdd(&counters[p.ctr], 1u);
                    if (idx < p.max_feat) {{
                        cand_pos[idx] = u32(row) + p.h * u32(col);
                        cand_layer[idx] = cl;
                    }}
                }}
            }}
        }}
        workgroupBarrier();
    }}
}}
"
    )
}

/// Refinement of the extrema. Arguments: pyramid, candidate positions, layers, counters, hash
/// table, `x`, `y`, layer, response and size of the survivors, uniform `{h, w, n_layers, max_feat,
/// octave, ext_counter, int_counter, hash_mask, contrast_thr, edge_thr, sigma, img_scale}`.
fn interp_source() -> String {
    format!(
        "struct Params {{ h: u32, w: u32, nl: u32, max_feat: u32, oct: u32, ctr_ext: u32, ctr_int: u32, mask: u32,
    contrast: f32, edge: f32, sigma: f32, scale: f32, plane: u32, a: u32, b: u32, c: u32 }}
@group(1) @binding(0) var<storage, read> g: array<f32>;
@group(1) @binding(1) var<storage, read> cand_pos: array<u32>;
@group(1) @binding(2) var<storage, read> cand_layer: array<u32>;
@group(1) @binding(3) var<storage, read_write> counters: array<atomic<u32>>;
@group(1) @binding(4) var<storage, read_write> table: array<atomic<u32>>;
@group(1) @binding(5) var<storage, read_write> out_x: array<f32>;
@group(1) @binding(6) var<storage, read_write> out_y: array<f32>;
@group(1) @binding(7) var<storage, read_write> out_l: array<u32>;
@group(1) @binding(8) var<storage, read_write> out_r: array<f32>;
@group(1) @binding(9) var<storage, read_write> out_z: array<f32>;
@group(1) @binding(10) var<uniform> p: Params;
{DOG_FN}
{RND}
var<private> lay: i32;
// The difference of Gaussians of the current layer (offset 0), the one below (-1) and above (1)
// at column c, row r.
fn cp(c: i32, r: i32) -> f32 {{ return dog(u32(lay), r, c); }}
fn pp(c: i32, r: i32) -> f32 {{ return dog(u32(lay - 1), r, c); }}
fn np(c: i32, r: i32) -> f32 {{ return dog(u32(lay + 1), r, c); }}
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_index) t: u32) {{
    let f = (wid.x + wid.y * nwg.x) * 256u + t;
    let n = min(atomicLoad(&counters[p.ctr_ext]), p.max_feat);
    if (f >= n) {{ return; }}
    let pos = cand_pos[f];
    var x = i32(pos / p.h);
    var y = i32(pos % p.h);
    lay = i32(cand_layer[f]);
    let first = p.scale * 0.5;
    let second = p.scale;
    let cross = p.scale * 0.25;
    var xl = 0.0;
    var xy = 0.0;
    var xx = 0.0;
    var converged = false;
    for (var step = 0; step < 5; step++) {{
        var dd = array<f32, 3>((cp(x + 1, y) - cp(x - 1, y)) * first, (cp(x, y + 1) - cp(x, y - 1)) * first, (np(x, y) - pp(x, y)) * first);
        let d2 = cp(x, y) * 2.0;
        let dxx = (cp(x + 1, y) + cp(x - 1, y) - d2) * second;
        let dyy = (cp(x, y + 1) + cp(x, y - 1) - d2) * second;
        let dss = (np(x, y) + pp(x, y) - d2) * second;
        let dxy = (cp(x + 1, y + 1) - cp(x - 1, y + 1) - cp(x + 1, y - 1) + cp(x - 1, y - 1)) * cross;
        let dxs = (np(x + 1, y) - np(x - 1, y) - pp(x + 1, y) + pp(x - 1, y)) * cross;
        let dys = (np(x, y + 1) - np(x - 1, y - 1) - pp(x, y - 1) + pp(x - 1, y - 1)) * cross;
        var a = array<f32, 9>(dxx, dxy, dxs, dxy, dyy, dys, dxs, dys, dss);
        for (var i = 0u; i < 2u; i++) {{
            for (var j = i + 1u; j < 3u; j++) {{
                let s = a[j * 3u + i] / a[i * 3u + i];
                for (var k = i; k < 3u; k++) {{ a[j * 3u + k] -= s * a[i * 3u + k]; }}
                dd[j] -= s * dd[i];
            }}
        }}
        var sol = array<f32, 3>(0.0, 0.0, 0.0);
        for (var i = 0u; i < 2u; i++) {{
            var sum = dd[i];
            for (var j = i + 1u; j < 3u; j++) {{ sum -= a[i * 3u + j] * sol[j]; }}
            sol[i] = sum / a[i * 3u + i];
        }}
        xl = -sol[2];
        xy = -sol[1];
        xx = -sol[0];
        if (xx != xx || xy != xy || xl != xl) {{ return; }}
        if (abs(xl) < 0.5 && abs(xy) < 0.5 && abs(xx) < 0.5) {{ converged = true; break; }}
        x += i32(rnd(xx));
        y += i32(rnd(xy));
        lay += i32(rnd(xl));
        if (lay < 1 || lay > i32(p.nl) || x < {BORDER} || x >= i32(p.w) - {BORDER} || y < {BORDER} || y >= i32(p.h) - {BORDER}) {{ return; }}
    }}
    if (!converged) {{ return; }}
    let dd0 = (cp(x + 1, y) - cp(x - 1, y)) * first;
    let dd1 = (cp(x, y + 1) - cp(x, y - 1)) * first;
    let dd2 = (np(x, y) - pp(x, y)) * first;
    let pv = dd0 * xx + dd1 * xy + dd2 * xl;
    let contr = cp(x, y) * p.scale + pv * 0.5;
    if (abs(contr) < p.contrast / f32(p.nl)) {{ return; }}
    let d2 = cp(x, y) * 2.0;
    let dxx = (cp(x + 1, y) + cp(x - 1, y) - d2) * second;
    let dyy = (cp(x, y + 1) + cp(x, y - 1) - d2) * second;
    let dxy = (cp(x + 1, y + 1) - cp(x - 1, y + 1) - cp(x + 1, y - 1) + cp(x - 1, y - 1)) * cross;
    let tr = dxx + dyy;
    let det = dxx * dyy - dxy * dxy;
    if (det <= 0.0 || tr * tr * p.edge >= (p.edge + 1.0) * (p.edge + 1.0) * det + 1.1920929e-7) {{ return; }}
    // A location already in the table is a duplicate.
    let key = (u32(lay) * p.h + u32(y)) * p.w + u32(x) + 1u;
    var hsh = key ^ (key >> 15u);
    hsh = hsh * 0x2c1b3c6du;
    hsh = hsh ^ (hsh >> 12u);
    hsh = hsh * 0x297a2d39u;
    hsh = hsh ^ (hsh >> 15u);
    var slot = hsh & p.mask;
    var winner = false;
    loop {{
        let r = atomicCompareExchangeWeak(&table[slot], 0u, key);
        if (r.exchanged) {{ winner = true; break; }}
        if (r.old_value == key) {{ break; }}
        if (r.old_value != 0u) {{ slot = (slot + 1u) & p.mask; }}
    }}
    if (!winner) {{ return; }}
    let idx = atomicAdd(&counters[p.ctr_int], 1u);
    if (idx < p.max_feat) {{
        let o = f32(1u << p.oct);
        out_x[idx] = (f32(x) + xx) * o;
        out_y[idx] = (f32(y) + xy) * o;
        out_l[idx] = u32(lay);
        out_r[idx] = abs(contr);
        out_z[idx] = p.sigma * pow(2.0, f32(p.oct) + (f32(lay) + xl) / f32(p.nl)) * 2.0;
    }}
}}
"
    )
}

/// Orientation histogram and peaks of the refined points. Arguments: pyramid, `x`, `y`, layer,
/// response and size of the points, counters, oriented `x`, `y`, layer, response, size and
/// orientation (degrees), uniform `{h, w, nl, max_feat, octave, int_counter, ori_counter,
/// max_oriented, fixed_point_scale, double, plane, 0}`.
fn orient_source() -> String {
    format!(
        "struct Params {{ h: u32, w: u32, nl: u32, max_feat: u32, oct: u32, ctr_int: u32, ctr_ori: u32, max_ori: u32, fix: f32, dbl: u32, plane: u32, c: u32 }}
@group(1) @binding(0) var<storage, read> g: array<f32>;
@group(1) @binding(1) var<storage, read> in_x: array<f32>;
@group(1) @binding(2) var<storage, read> in_y: array<f32>;
@group(1) @binding(3) var<storage, read> in_l: array<u32>;
@group(1) @binding(4) var<storage, read> in_r: array<f32>;
@group(1) @binding(5) var<storage, read> in_z: array<f32>;
@group(1) @binding(6) var<storage, read_write> counters: array<atomic<u32>>;
@group(1) @binding(7) var<storage, read_write> out_x: array<f32>;
@group(1) @binding(8) var<storage, read_write> out_y: array<f32>;
@group(1) @binding(9) var<storage, read_write> out_l: array<u32>;
@group(1) @binding(10) var<storage, read_write> out_r: array<f32>;
@group(1) @binding(11) var<storage, read_write> out_z: array<f32>;
@group(1) @binding(12) var<storage, read_write> out_o: array<f32>;
@group(1) @binding(13) var<uniform> p: Params;
{RND}
var<workgroup> hist: array<atomic<u32>, {ORI_BINS}>;
var<workgroup> wn: u32;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_index) t: u32) {{
    let pi = 3.14159265358979323846;
    let groups = nwg.x * nwg.y;
    var f = wid.x + wid.y * nwg.x;
    loop {{
        if (t == 0u) {{ wn = min(atomicLoad(&counters[p.ctr_int]), p.max_feat); }}
        workgroupBarrier();
        let n = workgroupUniformLoad(&wn);
        if (f >= n) {{ break; }}
        if (t < {ORI_BINS}u) {{ atomicStore(&hist[t], 0u); }}
        workgroupBarrier();
        let real_x = in_x[f];
        let real_y = in_y[f];
        let layer = in_l[f];
        let size = in_z[f];
        let octave = f32(1u << p.oct);
        let pt_x = i32(rnd(real_x / octave));
        let pt_y = i32(rnd(real_y / octave));
        let scl_octv = size * 0.5 / octave;
        let radius = i32(rnd(4.5 * scl_octv));
        let sigma = 1.5 * scl_octv;
        let len = radius * 2 + 1;
        let exp_denom = 2.0 * sigma * sigma;
        let base = layer * p.plane;
        let h = i32(p.h);
        let w = i32(p.w);
        for (var l = i32(t); l < len * len; l += 256) {{
            let i = l / len - radius;
            let j = l % len - radius;
            let y = pt_y + i;
            let x = pt_x + j;
            if (y < 1 || y >= h - 1 || x < 1 || x >= w - 1) {{ continue; }}
            let dx = g[base + u32(x + 1) * p.h + u32(y)] - g[base + u32(x - 1) * p.h + u32(y)];
            let dy = g[base + u32(x) * p.h + u32(y - 1)] - g[base + u32(x) * p.h + u32(y + 1)];
            let mag = sqrt(dx * dx + dy * dy);
            let ori = atan2(dy, dx);
            let wgt = exp(-f32(i * i + j * j) / exp_denom);
            var bin = i32(floor({ORI_BINS}.0 * (ori + pi) / (2.0 * pi) + 0.5));
            if (bin >= {ORI_BINS}) {{ bin = 0; }}
            atomicAdd(&hist[bin], u32(wgt * mag * p.fix + 0.5));
        }}
        workgroupBarrier();
        if (t == 0u) {{
            var hst: array<f32, {ORI_BINS}>;
            var tmp: array<f32, {ORI_BINS}>;
            for (var k = 0u; k < {ORI_BINS}u; k++) {{ hst[k] = f32(atomicLoad(&hist[k])); }}
            for (var sm = 0; sm < 2; sm++) {{
                for (var k = 0u; k < {ORI_BINS}u; k++) {{ tmp[k] = hst[k]; }}
                for (var k = 0u; k < {ORI_BINS}u; k++) {{
                    let prev = select(tmp[k - 1u], tmp[{ORI_BINS}u - 1u], k == 0u);
                    let next = select(tmp[k + 1u], tmp[0], k + 1u == {ORI_BINS}u);
                    hst[k] = 0.25 * prev + 0.5 * tmp[k] + 0.25 * next;
                }}
            }}
            var omax = hst[0];
            for (var k = 1u; k < {ORI_BINS}u; k++) {{ omax = max(omax, hst[k]); }}
            let mag_thr = omax * 0.8;
            for (var j = 0u; j < {ORI_BINS}u; j++) {{
                let l = select(j - 1u, {ORI_BINS}u - 1u, j == 0u);
                let r = (j + 1u) % {ORI_BINS}u;
                if (hst[j] > hst[l] && hst[j] > hst[r] && hst[j] >= mag_thr) {{
                    let idx = atomicAdd(&counters[p.ctr_ori], 1u);
                    if (idx < p.max_ori) {{
                        var bin = f32(j) + 0.5 * (hst[l] - hst[r]) / (hst[l] - 2.0 * hst[j] + hst[r]);
                        if (bin < 0.0) {{ bin = bin + {ORI_BINS}.0; }} else if (bin >= {ORI_BINS}.0) {{ bin = bin - {ORI_BINS}.0; }}
                        let ori = 360.0 - (360.0 / {ORI_BINS}.0) * bin;
                        var nx = real_x;
                        var ny = real_y;
                        var ns = size;
                        if (p.dbl != 0u) {{ nx *= 0.5; ny *= 0.5; ns *= 0.5; }}
                        out_x[idx] = nx;
                        out_y[idx] = ny;
                        out_l[idx] = layer;
                        out_r[idx] = in_r[f];
                        out_z[idx] = ns;
                        out_o[idx] = ori;
                    }}
                }}
            }}
        }}
        f += groups;
        workgroupBarrier();
    }}
}}
"
    )
}

/// Descriptor of an oriented point. Arguments: pyramid, oriented `x`, `y`, layer, response,
/// size, orientation (degrees), output `x`, `y`, score, orientation (radians), size,
/// descriptors, uniform `{h, w, count, offset, scale, fixed_point_scale, plane, 0}`.
fn describe_source(gloh: bool) -> String {
    let len = if gloh { 272 } else { 128 };
    let mut s = String::new();
    let _ = write!(
        s,
        "struct Params {{ h: u32, w: u32, n: u32, off: u32, scale: f32, fix: f32, plane: u32, c: u32 }}
@group(1) @binding(0) var<storage, read> g: array<f32>;
@group(1) @binding(1) var<storage, read> in_x: array<f32>;
@group(1) @binding(2) var<storage, read> in_y: array<f32>;
@group(1) @binding(3) var<storage, read> in_l: array<u32>;
@group(1) @binding(4) var<storage, read> in_r: array<f32>;
@group(1) @binding(5) var<storage, read> in_z: array<f32>;
@group(1) @binding(6) var<storage, read> in_o: array<f32>;
@group(1) @binding(7) var<storage, read_write> out_x: array<f32>;
@group(1) @binding(8) var<storage, read_write> out_y: array<f32>;
@group(1) @binding(9) var<storage, read_write> out_s: array<f32>;
@group(1) @binding(10) var<storage, read_write> out_o: array<f32>;
@group(1) @binding(11) var<storage, read_write> out_z: array<f32>;
@group(1) @binding(12) var<storage, read_write> out_d: array<f32>;
@group(1) @binding(13) var<uniform> p: Params;
{RND}
var<workgroup> dh: array<atomic<u32>, {len}>;
var<workgroup> red: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_index) t: u32) {{
    let pi = 3.14159265358979323846;
    let f = wid.x + wid.y * nwg.x;
    if (f >= p.n) {{ return; }}
    for (var k = t; k < {len}u; k += 256u) {{ atomicStore(&dh[k], 0u); }}
    workgroupBarrier();
    let layer = in_l[f];
    let ori_deg = in_o[f];
    var ori = (360.0 - ori_deg) * pi / 180.0;
    if (ori > pi) {{ ori = ori - pi * 2.0; }}
    let size = in_z[f];
    let fx = i32(rnd(in_x[f] * p.scale));
    let fy = i32(rnd(in_y[f] * p.scale));
    let base = layer * p.plane;
    let h = i32(p.h);
    let w = i32(p.w);
    let cos_t = cos(ori);
    let sin_t = sin(ori);
    let hist_width = 3.0 * size * p.scale * 0.5;
    let radius = i32(hist_width * sqrt(2.0) * 5.0 * 0.5 + 0.5);
    let len = radius * 2 + 1;
    for (var l = i32(t); l < len * len; l += 256) {{
        let i = l / len - radius;
        let j = l % len - radius;
        let y = fy + i;
        let x = fx + j;
        let fi = f32(i);
        let fj = f32(j);
"
    );
    if gloh {
        s.push_str(
            "        let x_rot = fj * cos_t - fi * sin_t;
        let y_rot = fj * sin_t + fi * cos_t;
        let r = sqrt(x_rot * x_rot + y_rot * y_rot) / f32(radius) * 15.0;
        var theta = atan2(y_rot, x_rot);
        while (theta < 0.0) { theta += pi * 2.0; }
        while (theta >= pi * 2.0) { theta -= pi * 2.0; }
        var tbin = theta * (8.0 / (pi * 2.0));
        var rbin = 0.0;
        if (r < 6.0) { rbin = r / 6.0; }
        else if (r < 11.0) { rbin = 1.0 + (r - 6.0) / 5.0; }
        else { rbin = min(2.0 + (r - 11.0) / 4.0, 3.0 - 1.1920929e-7); }
        if (r <= 15.0 && y > 0 && y < h - 1 && x > 0 && x < w - 1) {
            let dx = g[base + u32(x + 1) * p.h + u32(y)] - g[base + u32(x - 1) * p.h + u32(y)];
            let dy = g[base + u32(x) * p.h + u32(y - 1)] - g[base + u32(x) * p.h + u32(y + 1)];
            let grad_mag = sqrt(dx * dx + dy * dy);
            var grad_ori = atan2(dy, dx) - ori;
            while (grad_ori < 0.0) { grad_ori += pi * 2.0; }
            while (grad_ori >= pi * 2.0) { grad_ori -= pi * 2.0; }
            let wgt = exp(-r / (15.0 * 0.5));
            var obin = grad_ori * (16.0 / (pi * 2.0));
            let mag = grad_mag * wgt * p.fix;
            let t0 = i32(floor(tbin));
            let r0 = i32(floor(rbin));
            let o0 = i32(floor(obin));
            tbin -= f32(t0);
            rbin -= f32(r0);
            obin -= f32(o0);
            for (var rl = 0; rl <= 1; rl++) {
                var rb = r0 - rl;
                if (rbin > 0.5) { rb = r0 + rl; }
                let v_r = mag * select(rbin, 1.0 - rbin, rl == 0);
                if (rb >= 0 && rb <= 2) {
                    for (var tl = 0; tl <= 1; tl++) {
                        let tb = (t0 + tl) % 8;
                        let v_t = v_r * select(tbin, 1.0 - tbin, tl == 0);
                        for (var ol = 0; ol <= 1; ol++) {
                            let ob = (o0 + ol) % 16;
                            let v_o = v_t * select(obin, 1.0 - obin, ol == 0);
                            var idx = ob;
                            if (rb > 0) { idx = 16 + ((rb - 1) * 8 + tb) * 16 + ob; }
                            atomicAdd(&dh[idx], u32(v_o + 0.5));
                        }
                    }
                }
            }
        }
    }
",
        );
    } else {
        s.push_str(
            "        let x_rot = (fj * cos_t - fi * sin_t) / hist_width;
        let y_rot = (fj * sin_t + fi * cos_t) / hist_width;
        var xbin = x_rot + 2.0 - 0.5;
        var ybin = y_rot + 2.0 - 0.5;
        if (ybin > -1.0 && ybin < 4.0 && xbin > -1.0 && xbin < 4.0 && y > 0 && y < h - 1 && x > 0 && x < w - 1) {
            let dx = g[base + u32(x + 1) * p.h + u32(y)] - g[base + u32(x - 1) * p.h + u32(y)];
            let dy = g[base + u32(x) * p.h + u32(y - 1)] - g[base + u32(x) * p.h + u32(y + 1)];
            let grad_mag = sqrt(dx * dx + dy * dy);
            var grad_ori = atan2(dy, dx) - ori;
            while (grad_ori < 0.0) { grad_ori += pi * 2.0; }
            while (grad_ori >= pi * 2.0) { grad_ori -= pi * 2.0; }
            let wgt = exp(-(x_rot * x_rot + y_rot * y_rot) / 8.0);
            var obin = grad_ori * (8.0 / (pi * 2.0));
            let mag = grad_mag * wgt * p.fix;
            let x0 = i32(floor(xbin));
            let y0 = i32(floor(ybin));
            let o0 = i32(floor(obin));
            xbin -= f32(x0);
            ybin -= f32(y0);
            obin -= f32(o0);
            for (var yl = 0; yl <= 1; yl++) {
                let yb = y0 + yl;
                if (yb >= 0 && yb < 4) {
                    let v_y = mag * select(ybin, 1.0 - ybin, yl == 0);
                    for (var xl = 0; xl <= 1; xl++) {
                        let xb = x0 + xl;
                        if (xb >= 0 && xb < 4) {
                            let v_x = v_y * select(xbin, 1.0 - xbin, xl == 0);
                            for (var ol = 0; ol <= 1; ol++) {
                                let ob = (o0 + ol) % 8;
                                let v_o = v_x * select(obin, 1.0 - obin, ol == 0);
                                atomicAdd(&dh[(yb * 4 + xb) * 8 + ob], u32(v_o + 0.5));
                            }
                        }
                    }
                }
            }
        }
    }
",
        );
    }
    let two = len > 256;
    let _ = write!(
        s,
        "    workgroupBarrier();
    var v0 = 0.0;
    var v1 = 0.0;
    if (t < {len}u) {{ v0 = f32(atomicLoad(&dh[t])); }}
"
    );
    if two {
        let _ = writeln!(s, "    if (t + 256u < {len}u) {{ v1 = f32(atomicLoad(&dh[t + 256u])); }}");
    }
    // Two normalizations around the clipping, the reductions inline (barriers stay in main).
    for pass in 0..2 {
        let _ = write!(
            s,
            "    red[t] = v0 * v0 + v1 * v1;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {{
        if (t < s) {{ red[t] += red[t + s]; }}
        workgroupBarrier();
    }}
    let inv{pass} = 1.0 / sqrt(red[0]);
    workgroupBarrier();
    v0 = v0 * inv{pass};
    v1 = v1 * inv{pass};
"
        );
        if pass == 0 {
            s.push_str("    v0 = min(v0, 0.2);\n    v1 = min(v1, 0.2);\n");
        }
    }
    let _ = write!(
        s,
        "    let o = (p.off + f) * {len}u;
    if (t < {len}u) {{ out_d[o + t] = floor(min(255.0, v0 * 512.0) + 0.5); }}
"
    );
    if two {
        let _ = writeln!(s, "    if (t + 256u < {len}u) {{ out_d[o + t + 256u] = floor(min(255.0, v1 * 512.0) + 0.5); }}");
    }
    s.push_str(
        "    if (t == 0u) {
        let k = p.off + f;
        out_x[k] = in_x[f];
        out_y[k] = in_y[f];
        out_s[k] = in_r[f];
        out_o[k] = ori_deg * pi / 180.0;
        out_z[k] = size;
    }
}
",
    );
    s
}

/// A Gaussian filter as ArrayFire builds them: `min(round(6 sigma + 1) | 1, 31)` taps.
fn gauss_taps(sigma: f32) -> Vec<f32> {
    let len = ((sigma * 6.0 + 1.0).round() as usize | 1).min(31);
    let sigma = f64::from(sigma);
    let mut taps: Vec<f32> = (0..len)
        .map(|i| {
            let x = i as i32 - (len as i32 - 1) / 2;
            (1.0 / (2.0 * std::f64::consts::PI * sigma * sigma).sqrt() * (-f64::from(x * x) / (2.0 * sigma * sigma)).exp()) as f32
        })
        .collect();
    let sum: f32 = taps.iter().sum();
    for t in &mut taps {
        *t /= sum;
    }
    taps
}

/// One octave of the detection: its pyramid and the number of oriented points.
struct Octave {
    stack: Array<f32>,
    dims: [usize; 2],
    oriented: [Array<f32>; 2],
    oriented_layer: Array<u32>,
    oriented_resp: Array<f32>,
    oriented_size: Array<f32>,
    oriented_ori: Array<f32>,
    max_oriented: usize,
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn run(
    image: &Array<f32>,
    gloh: bool,
    n_layers: u32,
    contrast_thr: f32,
    edge_thr: f32,
    init_sigma: f32,
    double_input: bool,
    img_scale: f32,
    feature_ratio: f32,
) -> Outcome<Option<(Keypoints, Array<f32>)>> {
    let name = if gloh { "gloh" } else { "sift" };
    let ctx = image.context();
    let [h, w] = plane(image, name)?;
    if h < 15 || w < 15 {
        return Err(invalid(format!("{name} takes an image of at least 15 x 15, got {h} x {w}")));
    }
    if n_layers == 0 {
        return Err(invalid(format!("{name}: n_layers is 0")));
    }
    let valid = |v: f32, min: f32| v.is_finite() && v > min;
    if !valid(contrast_thr, 0.0) || !(edge_thr.is_finite() && edge_thr >= 1.0) || !valid(init_sigma, 0.5) || !valid(img_scale, 0.0) || !valid(feature_ratio, 0.0) {
        return Err(invalid(format!(
            "{name}: contrast_thr {contrast_thr} and intensity_scale {img_scale} and feature_ratio {feature_ratio} must be positive, edge_thr {edge_thr} at least 1, init_sigma {init_sigma} above 0.5"
        )));
    }
    image.storage()?;
    let nl = n_layers as usize;
    let layers = nl + 3;
    let mut min_dim = h.min(w);
    if double_input {
        min_dim *= 2;
    }
    let n_octaves = ((min_dim as f64).ln() / 2f64.ln()).floor() as i64 - 2;
    if n_octaves < 1 {
        return Ok(None);
    }
    let n_octaves = n_octaves as usize;
    let (ih, iw) = if double_input { (2 * h, 2 * w) } else { (h, w) };
    // Gaussian filters: the one of the initial image and one per layer.
    let init_s = if double_input { (init_sigma * init_sigma - 0.25 * 4.0).sqrt().max(0.1) } else { (init_sigma * init_sigma - 0.25).sqrt().max(0.1) };
    let k = 2f32.powf(1.0 / n_layers as f32);
    let mut sig = vec![init_sigma; layers];
    for (i, v) in sig.iter_mut().enumerate().skip(1) {
        let prev = k.powf((i - 1) as f32) * init_sigma;
        let total = prev * k;
        *v = (total * total - prev * prev).sqrt();
    }
    let mut tap_arrays: Vec<(Array<f32>, u32)> = Vec::new();
    for s in std::iter::once(init_s).chain(sig.iter().skip(1).copied()) {
        let taps = gauss_taps(s);
        let len = taps.len();
        tap_arrays.push((Array::from_slice_on(ctx, &taps, Shape::from(len))?, len as u32));
    }
    let launch = |kernel: Kernel| ctx.builtin(Builtin::Vision(Any::Descriptors(Descriptors::Sift(kernel))), "f32");
    // One pass of the blur from `src` (at `so`) to `dst` (at `dofs`) with the taps `tap`.
    let blur_pass = |axis: u8, src: &Array<f32>, so: usize, dst: &Array<f32>, dofs: usize, dims: [usize; 2], tap: usize| -> Outcome<()> {
        let (sp, sb) = src.ready()?;
        let (dp, db) = dst.ready()?;
        let (tp, tb) = tap_arrays[tap].0.ready()?;
        let len = tap_arrays[tap].1;
        let params = [to_u32(dims[0], "height")?, to_u32(dims[1], "width")?, to_u32(so, "offset")?, to_u32(dofs, "offset")?, len / 2, len, 0, 0];
        let kernel = launch(Kernel::Blur { axis })?;
        let args = [RawArg::read(sp.buffer(), sb), RawArg::write(dp.buffer(), db), RawArg::read(tp.buffer(), tb), RawArg::Uniform(bytemuck::bytes_of(&params))];
        ctx.launch(&kernel, tile_grid(ctx, params[0].div_ceil(32), params[1].div_ceil(8))?, [params[0], params[1], 1], &args)?;
        Ok(())
    };
    // Dimensions of the octaves and their stacks.
    let mut dims: Vec<[usize; 2]> = Vec::with_capacity(n_octaves);
    for o in 0..n_octaves {
        dims.push(if o == 0 { [ih, iw] } else { [dims[o - 1][0] / 2, dims[o - 1][1] / 2] });
    }
    let first_stack = Array::<f32>::alloc(ctx, Shape::from([dims[0][0], dims[0][1], layers]))?;
    let tmp = Array::<f32>::alloc(ctx, Shape::from(dims[0]))?;
    // The initial image: resize, then both passes of the blur into plane 0.
    if double_input {
        let big = Array::<f32>::alloc(ctx, Shape::from([ih, iw]))?;
        resize::launch(image, [h, w], &big, [ih, iw])?;
        blur_pass(0, &big, 0, &tmp, 0, [ih, iw], 0)?;
    } else {
        blur_pass(0, image, 0, &tmp, 0, [ih, iw], 0)?;
    }
    blur_pass(1, &tmp, 0, &first_stack, 0, [ih, iw], 0)?;
    // Pyramid.
    let mut stacks: Vec<Array<f32>> = vec![first_stack];
    for o in 0..n_octaves {
        let [oh, ow] = dims[o];
        if o > 0 {
            let stack = Array::<f32>::alloc(ctx, Shape::from([oh, ow, layers]))?;
            let prev = &stacks[o - 1];
            resize::launch_at(prev, nl * dims[o - 1][0] * dims[o - 1][1], dims[o - 1], &stack, 0, [oh, ow])?;
            stacks.push(stack);
        }
        let plane_len = oh * ow;
        for l in 1..layers {
            blur_pass(0, &stacks[o], (l - 1) * plane_len, &tmp, 0, [oh, ow], l)?;
            blur_pass(1, &tmp, 0, &stacks[o], l * plane_len, [oh, ow], l)?;
        }
    }
    // Detection.
    let counters = Array::<u32>::zeros_on(ctx, Shape::from(4 * n_octaves))?;
    let fix = (65536.0 * img_scale).clamp(1.0, 16_777_216.0);
    let mut octaves: Vec<Option<Octave>> = Vec::new();
    for o in 0..n_octaves {
        let [oh, ow] = dims[o];
        if oh < 2 * BORDER as usize + 1 || ow < 2 * BORDER as usize + 1 {
            octaves.push(None);
            continue;
        }
        let max_feat = ((oh * ow) as f32 * feature_ratio).ceil() as usize;
        let max_feat = max_feat.max(1);
        let plane_len = to_u32(oh * ow, "octave size")?;
        if (layers as u64 + 1) * u64::from(plane_len) >= u64::from(u32::MAX) {
            return Err(invalid("sift: the image is too large for the 32-bit locations of its extrema"));
        }
        let max_oriented = 3 * max_feat;
        let cand_pos = Array::<u32>::alloc(ctx, Shape::from(max_feat))?;
        let cand_layer = Array::<u32>::alloc(ctx, Shape::from(max_feat))?;
        let table_len = (2 * max_feat).next_power_of_two().max(64);
        let table = Array::<u32>::zeros_on(ctx, Shape::from(table_len))?;
        let int_x = Array::<f32>::alloc(ctx, Shape::from(max_feat))?;
        let int_y = Array::<f32>::alloc(ctx, Shape::from(max_feat))?;
        let int_l = Array::<u32>::alloc(ctx, Shape::from(max_feat))?;
        let int_r = Array::<f32>::alloc(ctx, Shape::from(max_feat))?;
        let int_z = Array::<f32>::alloc(ctx, Shape::from(max_feat))?;
        let or_x = Array::<f32>::alloc(ctx, Shape::from(max_oriented))?;
        let or_y = Array::<f32>::alloc(ctx, Shape::from(max_oriented))?;
        let or_l = Array::<u32>::alloc(ctx, Shape::from(max_oriented))?;
        let or_r = Array::<f32>::alloc(ctx, Shape::from(max_oriented))?;
        let or_z = Array::<f32>::alloc(ctx, Shape::from(max_oriented))?;
        let or_o = Array::<f32>::alloc(ctx, Shape::from(max_oriented))?;
        {
            let (g_pin, g_bytes) = stacks[o].ready()?;
            let (cp_pin, cp_bytes) = cand_pos.ready()?;
            let (cl_pin, cl_bytes) = cand_layer.ready()?;
            let (ct_pin, ct_bytes) = counters.ready()?;
            let (tb_pin, tb_bytes) = table.ready()?;
            let (ix_pin, ix_bytes) = int_x.ready()?;
            let (iy_pin, iy_bytes) = int_y.ready()?;
            let (il_pin, il_bytes) = int_l.ready()?;
            let (ir_pin, ir_bytes) = int_r.ready()?;
            let (iz_pin, iz_bytes) = int_z.ready()?;
            let (ox_pin, ox_bytes) = or_x.ready()?;
            let (oy_pin, oy_bytes) = or_y.ready()?;
            let (ol_pin, ol_bytes) = or_l.ready()?;
            let (or_pin, or_bytes) = or_r.ready()?;
            let (oz_pin, oz_bytes) = or_z.ready()?;
            let (oo_pin, oo_bytes) = or_o.ready()?;
            let (hh, ww, mf) = (to_u32(oh, "height")?, to_u32(ow, "width")?, to_u32(max_feat, "feature limit")?);
            // Extrema.
            let params = [hh, ww, n_layers, mf, (0.5 * contrast_thr / n_layers as f32).to_bits(), plane_len, 4 * o as u32, 0];
            let kernel = launch(Kernel::Extrema)?;
            let args = [
                RawArg::read(g_pin.buffer(), g_bytes),
                RawArg::write(cp_pin.buffer(), cp_bytes),
                RawArg::write(cl_pin.buffer(), cl_bytes),
                RawArg::write(ct_pin.buffer(), ct_bytes),
                RawArg::Uniform(bytemuck::bytes_of(&params)),
            ];
            ctx.launch(&kernel, tile_grid(ctx, (hh - 2 * BORDER).div_ceil(32), (ww - 2 * BORDER).div_ceil(8))?, [hh, ww, 1], &args)?;
            // Interpolation.
            let params = [
                hh,
                ww,
                n_layers,
                mf,
                o as u32,
                4 * o as u32,
                4 * o as u32 + 1,
                to_u32(table_len - 1, "table")?,
                contrast_thr.to_bits(),
                edge_thr.to_bits(),
                init_sigma.to_bits(),
                img_scale.to_bits(),
                plane_len,
                0,
                0,
                0,
            ];
            let kernel = launch(Kernel::Interp)?;
            let args = [
                RawArg::read(g_pin.buffer(), g_bytes),
                RawArg::read(cp_pin.buffer(), cp_bytes),
                RawArg::read(cl_pin.buffer(), cl_bytes),
                RawArg::write(ct_pin.buffer(), ct_bytes),
                RawArg::write(tb_pin.buffer(), tb_bytes),
                RawArg::write(ix_pin.buffer(), ix_bytes),
                RawArg::write(iy_pin.buffer(), iy_bytes),
                RawArg::write(il_pin.buffer(), il_bytes),
                RawArg::write(ir_pin.buffer(), ir_bytes),
                RawArg::write(iz_pin.buffer(), iz_bytes),
                RawArg::Uniform(bytemuck::bytes_of(&params)),
            ];
            ctx.launch(&kernel, fold_workgroups(ctx, mf.div_ceil(256))?, [mf, 1, 1], &args)?;
            // Orientation.
            let params = [hh, ww, n_layers, mf, o as u32, 4 * o as u32 + 1, 4 * o as u32 + 2, to_u32(max_oriented, "feature limit")?, fix.to_bits(), u32::from(double_input), plane_len, 0];
            let kernel = launch(Kernel::Orient)?;
            let args = [
                RawArg::read(g_pin.buffer(), g_bytes),
                RawArg::read(ix_pin.buffer(), ix_bytes),
                RawArg::read(iy_pin.buffer(), iy_bytes),
                RawArg::read(il_pin.buffer(), il_bytes),
                RawArg::read(ir_pin.buffer(), ir_bytes),
                RawArg::read(iz_pin.buffer(), iz_bytes),
                RawArg::write(ct_pin.buffer(), ct_bytes),
                RawArg::write(ox_pin.buffer(), ox_bytes),
                RawArg::write(oy_pin.buffer(), oy_bytes),
                RawArg::write(ol_pin.buffer(), ol_bytes),
                RawArg::write(or_pin.buffer(), or_bytes),
                RawArg::write(oz_pin.buffer(), oz_bytes),
                RawArg::write(oo_pin.buffer(), oo_bytes),
                RawArg::Uniform(bytemuck::bytes_of(&params)),
            ];
            let groups = mf.min(ORI_GROUPS);
            ctx.launch(&kernel, [groups, 1, 1], [groups * 256, 1, 1], &args)?;
        }
        octaves.push(Some(Octave { stack: stacks[o].clone(), dims: dims[o], oriented: [or_x, or_y], oriented_layer: or_l, oriented_resp: or_r, oriented_size: or_z, oriented_ori: or_o, max_oriented }));
    }
    // The only synchronization: the numbers of oriented points.
    let counts = counters.to_vec()?;
    let n_or: Vec<usize> = octaves.iter().enumerate().map(|(o, oc)| oc.as_ref().map_or(0, |oc| (counts[4 * o + 2] as usize).min(oc.max_oriented))).collect();
    let total: usize = n_or.iter().sum();
    if total == 0 {
        return Ok(None);
    }
    let desc_len = if gloh { 272 } else { 128 };
    let out_x = Array::<f32>::alloc(ctx, Shape::from(total))?;
    let out_y = Array::<f32>::alloc(ctx, Shape::from(total))?;
    let out_s = Array::<f32>::alloc(ctx, Shape::from(total))?;
    let out_o = Array::<f32>::alloc(ctx, Shape::from(total))?;
    let out_z = Array::<f32>::alloc(ctx, Shape::from(total))?;
    let out_d = Array::<f32>::alloc(ctx, Shape::from([desc_len, total]))?;
    {
        let (ox_pin, ox_bytes) = out_x.ready()?;
        let (oy_pin, oy_bytes) = out_y.ready()?;
        let (os_pin, os_bytes) = out_s.ready()?;
        let (oo_pin, oo_bytes) = out_o.ready()?;
        let (oz_pin, oz_bytes) = out_z.ready()?;
        let (od_pin, od_bytes) = out_d.ready()?;
        let mut offset = 0usize;
        for (o, oc) in octaves.iter().enumerate() {
            let Some(oc) = oc else { continue };
            let n = n_or[o];
            if n == 0 {
                continue;
            }
            let (g_pin, g_bytes) = oc.stack.ready()?;
            let (x_pin, x_bytes) = oc.oriented[0].ready()?;
            let (y_pin, y_bytes) = oc.oriented[1].ready()?;
            let (l_pin, l_bytes) = oc.oriented_layer.ready()?;
            let (r_pin, r_bytes) = oc.oriented_resp.ready()?;
            let (z_pin, z_bytes) = oc.oriented_size.ready()?;
            let (o_pin, o_bytes) = oc.oriented_ori.ready()?;
            let mut scale = 1.0 / (1u32 << o) as f32;
            if double_input {
                scale *= 2.0;
            }
            let [oh, ow] = oc.dims;
            let params = [to_u32(oh, "height")?, to_u32(ow, "width")?, to_u32(n, "count")?, to_u32(offset, "count")?, scale.to_bits(), fix.to_bits(), to_u32(oh * ow, "octave size")?, 0];
            let kernel = launch(Kernel::Describe { gloh })?;
            let args = [
                RawArg::read(g_pin.buffer(), g_bytes),
                RawArg::read(x_pin.buffer(), x_bytes),
                RawArg::read(y_pin.buffer(), y_bytes),
                RawArg::read(l_pin.buffer(), l_bytes),
                RawArg::read(r_pin.buffer(), r_bytes),
                RawArg::read(z_pin.buffer(), z_bytes),
                RawArg::read(o_pin.buffer(), o_bytes),
                RawArg::write(ox_pin.buffer(), ox_bytes),
                RawArg::write(oy_pin.buffer(), oy_bytes),
                RawArg::write(os_pin.buffer(), os_bytes),
                RawArg::write(oo_pin.buffer(), oo_bytes),
                RawArg::write(oz_pin.buffer(), oz_bytes),
                RawArg::write(od_pin.buffer(), od_bytes),
                RawArg::Uniform(bytemuck::bytes_of(&params)),
            ];
            let count = params[2];
            ctx.launch(&kernel, fold_workgroups(ctx, count)?, [count * 256, 1, 1], &args)?;
            offset += n;
        }
    }
    Ok(Some((Keypoints { x: out_x, y: out_y, score: out_s, orientation: out_o, size: out_z }, out_d)))
}

impl Array<f32> {
    /// SIFT keypoints and 128-element descriptors of an `[H, W]` image (ArrayFire's `sift`).
    ///
    /// The image (any range; `intensity_scale` is one over its maximum, `1/256` for 8-bit data)
    /// is optionally doubled in size (`double_input`, bilinear) and blurred to `init_sigma`, then
    /// a Gaussian pyramid of `floor(log2(min side)) - 2` octaves with `n_layers + 3` layers
    /// per octave and its difference of Gaussians are built. A pixel of layers `1..=n_layers` that
    /// is larger or smaller than its 26 neighbours (and above half the contrast threshold,
    /// `0.5 * contrast_thr / n_layers`, in the units of the image) is an extremum; it is
    /// refined (up to 5 steps, within 5 pixels of the border no extrema), rejected if its
    /// interpolated contrast `|D| * intensity_scale` is below `contrast_thr / n_layers` or if its
    /// principal curvature ratio exceeds `edge_thr`, and duplicates are removed. At most
    /// `ceil(feature_ratio * pixels of the octave)` extrema are kept per octave. Every surviving
    /// point gets one oriented point per peak (at least 80% of the highest) of a 36-bin
    /// gradient-orientation histogram and a descriptor of 4 x 4 orientation histograms of 8 bins,
    /// normalized, clipped at 0.2, scaled by 512 and rounded to integers in `0..=255` (stored as
    /// `f32`).
    ///
    /// Returns the [`Keypoints`] (`x` along axis 1 and `y` along axis 0 in the coordinates of
    /// the input image, sub-pixel; `score` the absolute interpolated contrast; `orientation`
    /// in **radians** in `[0, 2 pi)`; `size` the diameter `2 * init_sigma * 2^(octave + layer /
    /// n_layers)` in input pixels) and the descriptors as a `[128, N]` array (one column per
    /// point); `None` if there are no points. The points of an octave follow each other, finest
    /// octave first; within an octave their order is not specified.
    ///
    /// This reproduces ArrayFire's algorithm, including its quirks, so that the keypoints
    /// agree with ArrayFire's: its 3 x 3 solver for the sub-pixel offset never computes the
    /// offset along the scale (the layer, and so the size, of a point is the one it was found
    /// in), the scale-space cross derivative `d2D/dy ds` uses the pixel `(x - 1, y - 1)`, and
    /// the first threshold on the difference of Gaussians (above) is applied to the unscaled
    /// image. Differences: ArrayFire returns the orientation in degrees (luma converts to
    /// radians like [`Array::orb`]), sorts the points of an octave by position (luma removes
    /// duplicates with a hash table and leaves the order open), and its descriptor histograms
    /// are accumulated in floating point where luma uses integer atomics scaled by
    /// `65536 * intensity_scale` (against ArrayFire's CUDA backend the mean difference of the
    /// entries is below 1 of 255; a point whose position or radius sits on a rounding boundary can
    /// differ in a few entries by more; an image whose range is far from `1 / intensity_scale`
    /// can overflow the integers).
    ///
    /// Every octave is recorded without a readback; the number of points is read back once, so
    /// this **synchronizes** once. Fails for images smaller than 15 x 15 or with parameters
    /// outside ArrayFire's ranges (`n_layers > 0`, `contrast_thr > 0`, `edge_thr >= 1`,
    /// `init_sigma > 0.5`, `intensity_scale > 0`, `feature_ratio > 0`).
    ///
    /// ```
    /// use luma::Array;
    /// # fn main() -> Result<(), luma::LumaError> {
    /// // A Gaussian blob is a scale-space extremum.
    /// let mut data = vec![0.0f32; 96 * 96];
    /// for x in 0..96 { for y in 0..96 {
    ///     let (dx, dy) = (x as f32 - 48.0, y as f32 - 48.0);
    ///     data[y + 96 * x] = 200.0 * (-(dx * dx + dy * dy) / 50.0).exp();
    /// } }
    /// let img = Array::from_slice(&data, [96, 96])?;
    /// let (points, desc) = img.sift(3, 0.04, 10.0, 1.6, true, 1.0 / 256.0, 0.05)?.expect("a blob");
    /// assert_eq!(desc.shape().dims()[0], 128);
    /// assert_eq!(desc.shape().dims()[1], points.count());
    /// # Ok(()) }
    /// ```
    #[track_caller]
    #[allow(clippy::too_many_arguments)]
    pub fn sift(&self, n_layers: u32, contrast_thr: f32, edge_thr: f32, init_sigma: f32, double_input: bool, intensity_scale: f32, feature_ratio: f32) -> crate::error::Result<Option<(Keypoints, Array<f32>)>> {
        let _ = std::panic::Location::caller();
        run(self, false, n_layers, contrast_thr, edge_thr, init_sigma, double_input, intensity_scale, feature_ratio).map_err(Failure::into_error)
    }

    /// SIFT keypoints with GLOH descriptors (ArrayFire's `gloh`): the keypoints of
    /// [`Array::sift`] with the 272-element gradient location and orientation histogram of
    /// Mikolajczyk and Schmid: a circular patch of radius 15 in a log-polar grid of 3 radial
    /// (one inner disc, two rings of 8 angular sectors) bins with 16 orientation bins each,
    /// normalized like the SIFT descriptor. Returns a `[272, N]` descriptor array; everything else
    /// is as in [`Array::sift`].
    ///
    /// ```
    /// use luma::Array;
    /// # fn main() -> Result<(), luma::LumaError> {
    /// let mut data = vec![0.0f32; 96 * 96];
    /// for x in 0..96 { for y in 0..96 {
    ///     let (dx, dy) = (x as f32 - 48.0, y as f32 - 48.0);
    ///     data[y + 96 * x] = 200.0 * (-(dx * dx + dy * dy) / 50.0).exp();
    /// } }
    /// let img = Array::from_slice(&data, [96, 96])?;
    /// let (points, desc) = img.gloh(3, 0.04, 10.0, 1.6, true, 1.0 / 256.0, 0.05)?.expect("a blob");
    /// assert_eq!(desc.shape().dims()[0], 272);
    /// assert_eq!(desc.shape().dims()[1], points.count());
    /// # Ok(()) }
    /// ```
    #[track_caller]
    #[allow(clippy::too_many_arguments)]
    pub fn gloh(&self, n_layers: u32, contrast_thr: f32, edge_thr: f32, init_sigma: f32, double_input: bool, intensity_scale: f32, feature_ratio: f32) -> crate::error::Result<Option<(Keypoints, Array<f32>)>> {
        let _ = std::panic::Location::caller();
        run(self, true, n_layers, contrast_thr, edge_thr, init_sigma, double_input, intensity_scale, feature_ratio).map_err(Failure::into_error)
    }
}
