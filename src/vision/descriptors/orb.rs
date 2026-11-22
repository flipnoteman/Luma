//! ORB (`Array::orb`): FAST corners on an image pyramid, ranked by their Harris response, with the
//! intensity-centroid orientation and a rotated BRIEF descriptor.
//!
//! The whole call is recorded without a synchronization: the per-level candidate counts live on
//! the device, every launch is sized by a bound known on the host (the feature cap of the level)
//! and the groups past the real count exit at once. The only readback is the number of
//! features each level kept, at the end. Per level, five launches:
//!
//! 1. resize of the previous level (bilinear, ArrayFire's convention);
//! 2. FAST (arc 9) with non-maximal suppression over tiles of 32 x 8 pixels: a tile appends its
//!    corners to the candidate list with one atomic add (a single pass; ArrayFire counts, scans
//!    and scatters), computes the Harris response of each (7 x 7 block, `k = 0.04`, scaled by
//!    `1e-12`) and counts its key in a histogram;
//! 3. the cut of the ranking from the histogram (the best `lvl_best` candidates are the ones
//!    above it) and the compaction of the candidates at or above the cut;
//! 4. the exact ranking of the compacted candidates (ties by candidate index);
//! 5. one workgroup per kept corner: orientation (31 x 31 intensity centroid), then the 256 tests
//!    of the BRIEF pattern, one per invocation, on the (optionally Gaussian blurred) level image;
//!    the results go to the output arrays at the offset of the level.

use std::fmt::Write as _;

use super::orb_pattern::{PATCH, PATTERN};
use super::{resize, Kernel as Descriptors};
use crate::array::{Array, Failure};
use crate::builtin::Builtin;
use crate::context::RawArg;
use crate::image::{gaussian_kernel_1d_on, ConvSpec};
use crate::ops::{fold_workgroups, Outcome};
use crate::shape::Shape;
use crate::vision::{invalid, plane, tile_grid, to_u32, Kernel as Any, Keypoints};

/// Pixels of a FAST tile along axis 0 and axis 1.
const TX: u32 = 32;
const TY: u32 = 8;
/// The border the corners keep from the image: ArrayFire asks `ceil(31 * sqrt(2) / 2)` of FAST
/// and its non-maximal suppression drops two more pixels.
const EDGE: u32 = 22;
/// Bins of the histogram that finds the cut of the Harris ranking.
const BINS: u32 = 4096;
/// The fraction of the pixels of a level that may be corners (ArrayFire's CUDA backend).
const FAST_RATIO: f32 = 0.14;
/// Workgroups of the selection and ranking launches (they loop over the candidates).
const SELECT_GROUPS: u32 = 128;

/// One kernel of ORB.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum Kernel {
    /// FAST corners with their Harris responses, appended to a list.
    Fast,
    /// The cut of the Harris ranking and the compaction of the candidates above it.
    Select,
    /// Exact ranking of the compacted candidates.
    Rank,
    /// Orientation and descriptors; `blur` has a second (blurred) image for the tests.
    Describe { blur: bool },
}

pub(crate) fn source(k: Kernel) -> String {
    match k {
        Kernel::Fast => fast_source(),
        Kernel::Select => select_source(),
        Kernel::Rank => rank_source(),
        Kernel::Describe { blur } => describe_source(blur),
    }
}

/// The ring of the FAST detector, as offsets along axis 0 and 1, clockwise from the top.
const RING: [(i32, i32); 16] = [(3, 0), (3, 1), (2, 2), (1, 3), (0, 3), (-1, 3), (-2, 2), (-3, 1), (-3, 0), (-3, -1), (-2, -2), (-1, -3), (0, -3), (1, -3), (2, -2), (3, -1)];

/// The factor ArrayFire scales the Harris response with: `0.001^4` in single precision.
fn harris_scale() -> f32 {
    let r = 0.001f32;
    r * r * r * r
}

/// The key of a response: the bits of the float turned into an unsigned integer that orders like
/// the float.
const KEY_FN: &str = "fn key(h: f32) -> u32 {
    let b = bitcast<u32>(h);
    return select(b | 0x80000000u, ~b, (b >> 31u) != 0u);
}
";

/// FAST, arc 9, with the Harris response of the corners. The corners of a tile are appended to
/// the list with one atomic add per tile (the order of the list is not deterministic: ties of the ranking are broken arbitrarily) and counted
/// in the histogram of the keys of their responses. Arguments: image, positions, responses,
/// histogram, counters, uniform `{h, w, thr, edge, tiles_x, cap, level, 0}`.
fn fast_source() -> String {
    let (halo_in, halo_sc) = (4u32, 1u32);
    let (iw, ih) = (TX + 2 * halo_in, TY + 2 * halo_in);
    let (sw, sh) = (TX + 2 * halo_sc, TY + 2 * halo_sc);
    let mut s = String::new();
    s.push_str("struct Params { h: u32, w: u32, thr: f32, edge: u32, tiles_x: u32, cap: u32, level: u32, c: u32 }\n");
    s.push_str("@group(1) @binding(0) var<storage, read> src: array<f32>;\n");
    s.push_str("@group(1) @binding(1) var<storage, read_write> out_pos: array<u32>;\n");
    s.push_str("@group(1) @binding(2) var<storage, read_write> out_h: array<f32>;\n");
    s.push_str("@group(1) @binding(3) var<storage, read_write> hist: array<atomic<u32>>;\n");
    s.push_str("@group(1) @binding(4) var<storage, read_write> counters: array<atomic<u32>>;\n");
    s.push_str("@group(1) @binding(5) var<uniform> p: Params;\n");
    s.push_str(KEY_FN);
    let _ = writeln!(s, "var<workgroup> tile: array<f32, {}>;", iw * ih);
    let _ = writeln!(s, "var<workgroup> sc: array<f32, {}>;", sw * sh);
    s.push_str("var<workgroup> rank: array<u32, 256>;\nvar<workgroup> base: u32;\n");
    let off = |v: &str, d: i32| if d >= 0 { format!("{v} + {d}u") } else { format!("{v} - {}u", -d) };
    let at = |d0: i32, d1: i32| format!("tile[({}) * {iw}u + {}]", off("cy", d1), off("cx", d0));
    let _ = write!(
        s,
        "fn score_at(cx: u32, cy: u32) -> f32 {{
    let c = tile[cy * {iw}u + cx];
    let hi = c + p.thr;
    let lo = c - p.thr;
"
    );
    let card = [0usize, 4, 8, 12];
    for (n, &k) in card.iter().enumerate() {
        let (d0, d1) = RING[k];
        let _ = writeln!(s, "    let q{n} = {};", at(d0, d1));
    }
    s.push_str("    var nb = 0u; var nd = 0u;\n");
    for n in 0..4 {
        let _ = writeln!(s, "    nb += select(0u, 1u, q{n} > hi); nd += select(0u, 1u, q{n} < lo);");
    }
    s.push_str("    if (nb < 2u && nd < 2u) { return 0.0; }\n");
    for (j, &(d0, d1)) in RING.iter().enumerate() {
        if card.contains(&j) {
            let _ = writeln!(s, "    let r{j} = q{};", card.iter().position(|&c| c == j).unwrap());
        } else {
            let _ = writeln!(s, "    let r{j} = {};", at(d0, d1));
        }
    }
    s.push_str("    var bm = 0u; var dm = 0u; var sb = 0.0; var sd = 0.0;\n");
    for j in 0..16 {
        let _ = writeln!(
            s,
            "    if (r{j} > hi) {{ bm |= {}u; sb += r{j} - hi; }}\n    if (r{j} < lo) {{ dm |= {}u; sd += lo - r{j}; }}",
            1u32 << j,
            1u32 << j
        );
    }
    s.push_str("    var b2 = bm | (bm << 16u); var d2 = dm | (dm << 16u);\n    var tb = b2; var td = d2;\n");
    for i in 1..9 {
        let _ = writeln!(s, "    tb &= b2 >> {i}u; td &= d2 >> {i}u;");
    }
    s.push_str("    var out = 0.0;\n    if ((tb & 65535u) != 0u) { out = sb; }\n    if ((td & 65535u) != 0u) { out = max(out, sd); }\n    return out;\n}\n");
    // ArrayFire's Harris response of a corner: 7 x 7 block of central differences.
    let _ = write!(
        s,
        "fn harris(row: i32, col: i32) -> f32 {{
    var ixx = 0.0; var iyy = 0.0; var ixy = 0.0;
    let h = p.h;
    for (var k = 0u; k < 49u; k++) {{
        let i = i32(k / 7u) - 3;
        let j = i32(k % 7u) - 3;
        let ix = src[u32(row + j) + h * u32(col + i + 1)] - src[u32(row + j) + h * u32(col + i - 1)];
        let iy = src[u32(row + j + 1) + h * u32(col + i)] - src[u32(row + j - 1) + h * u32(col + i)];
        ixx += ix * ix; iyy += iy * iy; ixy += ix * iy;
    }}
    let tr = ixx + iyy;
    let det = ixx * iyy - ixy * ixy;
    return (det - 0.04 * (tr * tr)) * {scale:?};
}}
",
        scale = harris_scale()
    );
    let _ = write!(
        s,
        "@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) t: u32, @builtin(workgroup_id) wid: vec3<u32>) {{
    let o0 = i32(wid.x) * {TX} - {halo_in};
    let o1 = i32(wid.y) * {TY} - {halo_in};
    let h = i32(p.h);
    let w = i32(p.w);
    for (var i = t; i < {total}u; i += 256u) {{
        let cx = i % {iw}u;
        let cy = i / {iw}u;
        let a0 = o0 + i32(cx);
        let a1 = o1 + i32(cy);
        var v = 0.0;
        if (a0 >= 0 && a0 < h && a1 >= 0 && a1 < w) {{ v = src[u32(a0) + p.h * u32(a1)]; }}
        tile[i] = v;
    }}
    workgroupBarrier();
    let e = i32(p.edge);
    for (var i = t; i < {sc_total}u; i += 256u) {{
        let cx = i % {sw}u;
        let cy = i / {sw}u;
        let a0 = o0 + {halo_in} - {halo_sc} + i32(cx);
        let a1 = o1 + {halo_in} - {halo_sc} + i32(cy);
        var v = 0.0;
        if (a0 >= e && a0 < h - e && a1 >= e && a1 < w - e) {{ v = score_at(cx + {d}u, cy + {d}u); }}
        sc[i] = v;
    }}
    workgroupBarrier();
    let tx = t & 31u;
    let ty = t >> 5u;
    let row = i32(wid.x) * {TX} + i32(tx);
    let col = i32(wid.y) * {TY} + i32(ty);
    let r2 = e + 2;
    let s0 = sc[(ty + 1u) * {sw}u + tx + 1u];
    var keep = s0 > 0.0 && row >= r2 && row <= h - r2 && col >= r2 && col <= w - r2;
    if (keep) {{
        let b = ty * {sw}u + tx;
        keep = s0 > sc[b] && s0 > sc[b + 1u] && s0 > sc[b + 2u]
            && s0 > sc[b + {sw}u] && s0 > sc[b + {sw}u + 2u]
            && s0 > sc[b + 2u * {sw}u] && s0 > sc[b + 2u * {sw}u + 1u] && s0 > sc[b + 2u * {sw}u + 2u];
    }}
    rank[t] = select(0u, 1u, keep);
    workgroupBarrier();
    for (var d = 1u; d < 256u; d = d << 1u) {{
        var add = 0u;
        if (t >= d) {{ add = rank[t - d]; }}
        workgroupBarrier();
        rank[t] = rank[t] + add;
        workgroupBarrier();
    }}
    if (t == 255u && rank[255] != 0u) {{ base = atomicAdd(&counters[p.level * 4u], rank[255]); }}
    workgroupBarrier();
    if (keep) {{
        let k = base + rank[t] - 1u;
        if (k < p.cap) {{
            let hv = harris(row, col);
            out_pos[k] = u32(row) + p.h * u32(col);
            out_h[k] = hv;
            atomicAdd(&hist[p.level * {BINS}u + (key(hv) >> 20u)], 1u);
        }}
    }}
}}
",
        total = iw * ih,
        sc_total = sw * sh,
        d = halo_in - halo_sc
    );
    s
}

/// The cut of the ranking (the highest bin from which the bins above hold at least `best`
/// responses, 0 if there are fewer candidates) and the compaction of the candidates with a key
/// at or above it, by a grid-stride loop; the first invocation also stores the number of kept
/// features of the level. Arguments: positions, responses, counters, histogram, compacted
/// positions, responses and candidate indices, kept counts, uniform `{cap, best, level, 0}`.
fn select_source() -> String {
    format!(
        "struct Params {{ cap: u32, best: u32, level: u32, c: u32 }}
@group(1) @binding(0) var<storage, read> cand_pos: array<u32>;
@group(1) @binding(1) var<storage, read> cand_h: array<f32>;
@group(1) @binding(2) var<storage, read_write> counters: array<atomic<u32>>;
@group(1) @binding(3) var<storage, read> hist: array<u32>;
@group(1) @binding(4) var<storage, read_write> sel_pos: array<u32>;
@group(1) @binding(5) var<storage, read_write> sel_h: array<f32>;
@group(1) @binding(6) var<storage, read_write> sel_g: array<u32>;
@group(1) @binding(7) var<storage, read_write> kept: array<u32>;
@group(1) @binding(8) var<uniform> p: Params;
{KEY_FN}
var<workgroup> part: array<u32, 256>;
var<workgroup> cutv: u32;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_index) t: u32) {{
    let n = min(atomicLoad(&counters[p.level * 4u]), p.cap);
    let hb = p.level * {BINS}u;
    var sum = 0u;
    for (var k = 0u; k < 16u; k++) {{ sum += hist[hb + t * 16u + k]; }}
    part[t] = sum;
    workgroupBarrier();
    if (t == 0u) {{
        var cut = 0u;
        if (n > p.best) {{
            var acc = 0u;
            for (var c = 0u; c < 256u; c++) {{
                let ci = 255u - c;
                if (acc + part[ci] >= p.best) {{
                    for (var b = 0u; b < 16u; b++) {{
                        let bi = ci * 16u + 15u - b;
                        acc += hist[hb + bi];
                        if (acc >= p.best) {{ cut = bi; break; }}
                    }}
                    break;
                }}
                acc += part[ci];
            }}
        }}
        cutv = cut;
        if (wid.x == 0u && wid.y == 0u) {{ kept[p.level] = min(n, p.best); }}
    }}
    workgroupBarrier();
    let cut = workgroupUniformLoad(&cutv);
    let stride = nwg.x * nwg.y * 256u;
    for (var g = (wid.x + wid.y * nwg.x) * 256u + t; g < n; g += stride) {{
        let h = cand_h[g];
        if ((key(h) >> 20u) >= cut) {{
            let k = atomicAdd(&counters[p.level * 4u + 1u], 1u);
            sel_pos[k] = cand_pos[g];
            sel_h[k] = h;
            sel_g[k] = g;
        }}
    }}
}}
"
    )
}

/// Exact ranking of the compacted candidates: the first `best` by response, ties by candidate
/// index. Arguments: compacted positions, responses, indices, counters, kept positions and
/// responses, uniform `{cap, best, level, 0}`.
fn rank_source() -> String {
    format!(
        "struct Params {{ cap: u32, best: u32, level: u32, c: u32 }}
@group(1) @binding(0) var<storage, read> sel_pos: array<u32>;
@group(1) @binding(1) var<storage, read> sel_h: array<f32>;
@group(1) @binding(2) var<storage, read> sel_g: array<u32>;
@group(1) @binding(3) var<storage, read> counters: array<u32>;
@group(1) @binding(4) var<storage, read_write> kept_pos: array<u32>;
@group(1) @binding(5) var<storage, read_write> kept_h: array<f32>;
@group(1) @binding(6) var<uniform> p: Params;
{KEY_FN}
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_index) t: u32) {{
    let m = counters[p.level * 4u + 1u];
    let stride = nwg.x * nwg.y * 256u;
    for (var g = (wid.x + wid.y * nwg.x) * 256u + t; g < m; g += stride) {{
        let ke = key(sel_h[g]);
        let ge = sel_g[g];
        var rank = 0u;
        for (var f = 0u; f < m; f++) {{
            let kf = key(sel_h[f]);
            rank += select(0u, 1u, kf > ke || (kf == ke && sel_g[f] < ge));
        }}
        if (rank < p.best) {{
            kept_pos[rank] = sel_pos[g];
            kept_h[rank] = sel_h[g];
        }}
    }}
}}
"
    )
}

/// Orientation and descriptor of the kept corner `workgroup`. Arguments: level image, [blurred
/// image,] pattern, kept positions, kept responses, kept counts, `x`, `y`, `score`,
/// `orientation`, `size`, descriptors, uniform `{h, w, level, best, scale, 0, 0, 0}`.
fn describe_source(blur: bool) -> String {
    let mut s = String::from("struct Params { h: u32, w: u32, level: u32, best: u32, scl: f32, a: u32, b: u32, c: u32 }\n");
    let mut binding = 0;
    let mut bind = |s: &mut String, access: &str, name: &str, ty: &str| {
        let _ = writeln!(s, "@group(1) @binding({binding}) var<storage, {access}> {name}: array<{ty}>;");
        binding += 1;
    };
    bind(&mut s, "read", "img", "f32");
    if blur {
        bind(&mut s, "read", "bimg", "f32");
    }
    bind(&mut s, "read", "pat", "i32");
    bind(&mut s, "read", "kept_pos", "u32");
    bind(&mut s, "read", "kept_h", "f32");
    bind(&mut s, "read", "kept", "u32");
    bind(&mut s, "read_write", "out_x", "f32");
    bind(&mut s, "read_write", "out_y", "f32");
    bind(&mut s, "read_write", "out_s", "f32");
    bind(&mut s, "read_write", "out_o", "f32");
    bind(&mut s, "read_write", "out_z", "f32");
    bind(&mut s, "read_write", "out_d", "u32");
    let _ = writeln!(s, "@group(1) @binding({binding}) var<uniform> p: Params;");
    let tests = if blur { "bimg" } else { "img" };
    let r = PATCH / 2;
    let _ = write!(
        s,
        "var<workgroup> ra: array<f32, 256>;
var<workgroup> rb: array<f32, 256>;
var<workgroup> bits: array<u32, 256>;
var<workgroup> wcount: u32;
fn rnd(v: f32) -> f32 {{ return sign(v) * floor(abs(v) + 0.5); }}
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_index) t: u32) {{
    let g = wid.x + wid.y * nwg.x;
    if (t == 0u) {{ wcount = kept[p.level]; }}
    workgroupBarrier();
    if (g >= workgroupUniformLoad(&wcount)) {{ return; }}
    let pos = kept_pos[g];
    let row = i32(pos % p.h);
    let col = i32(pos / p.h);
    var m0 = 0.0;
    var m1 = 0.0;
    for (var k = t; k < {area}u; k += 256u) {{
        let i = i32(k / {PATCH}u) - {r};
        let j = i32(k % {PATCH}u) - {r};
        let v = img[u32(row + j) + p.h * u32(col + i)];
        m0 += f32(j) * v;
        m1 += f32(i) * v;
    }}
    ra[t] = m0;
    rb[t] = m1;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {{
        if (t < s) {{ ra[t] += ra[t + s]; rb[t] += rb[t + s]; }}
        workgroupBarrier();
    }}
    let ori = atan2(ra[0], rb[0]);
    let sn = sin(ori);
    let cs = cos(ori);
    let x1 = f32(pat[4u * t]);
    let y1 = f32(pat[4u * t + 1u]);
    let x2 = f32(pat[4u * t + 2u]);
    let y2 = f32(pat[4u * t + 3u]);
    let c1 = col + i32(rnd(x1 * cs - y1 * sn));
    let r1 = row + i32(rnd(x1 * sn + y1 * cs));
    let c2 = col + i32(rnd(x2 * cs - y2 * sn));
    let r2 = row + i32(rnd(x2 * sn + y2 * cs));
    let a = {tests}[u32(r1) + p.h * u32(c1)];
    let b = {tests}[u32(r2) + p.h * u32(c2)];
    bits[t] = select(0u, 1u, a < b);
    var off = 0u;
    for (var l = 0u; l < p.level; l++) {{ off += kept[l]; }}
    workgroupBarrier();
    if (t < 8u) {{
        var word = 0u;
        for (var k = 0u; k < 32u; k++) {{ word |= bits[t * 32u + k] << k; }}
        out_d[(off + g) * 8u + t] = word;
    }}
    if (t == 0u) {{
        let k = off + g;
        out_x[k] = floor(f32(col) * p.scl + 0.5);
        out_y[k] = floor(f32(row) * p.scl + 0.5);
        out_s[k] = kept_h[g];
        out_o[k] = ori;
        out_z[k] = {patch}.0 * p.scl;
    }}
}}
",
        area = PATCH * PATCH,
        patch = PATCH
    );
    s
}

/// The levels of the pyramid: scales, sizes and the number of features to keep per level, as
/// ArrayFire's CUDA backend computes them (single precision).
struct Levels {
    scale: Vec<f32>,
    dims: Vec<[usize; 2]>,
    best: Vec<u32>,
}

fn levels_of(h: usize, w: usize, max_feat: u32, scl: f32, levels: u32) -> Levels {
    let mut min_side = h.min(w) as u32;
    let mut count = 0u32;
    let mut sum = 0f32;
    for i in 0..levels {
        min_side = (min_side as f32 / scl) as u32;
        if min_side < PATCH as u32 || count == levels {
            break;
        }
        count += 1;
        sum += 1.0 / scl.powf(i as f32);
    }
    let n = count as usize;
    let scale: Vec<f32> = (0..n).map(|i| scl.powf(i as f32)).collect();
    let mut best = Vec::with_capacity(n);
    let mut used = 0u32;
    for s in scale.iter().take(n.saturating_sub(1)) {
        let b = ((max_feat as f32 / sum) / s).ceil() as u32;
        used = used.saturating_add(b);
        best.push(b);
    }
    if n > 0 {
        best.push(max_feat.saturating_sub(used));
    }
    let dims = scale.iter().enumerate().map(|(i, s)| if i == 0 { [h, w] } else { [((h as f32 / s).round() as usize).max(1), ((w as f32 / s).round() as usize).max(1)] }).collect();
    Levels { scale, dims, best }
}

pub(crate) fn run(image: &Array<f32>, fast_thr: f32, max_feat: u32, scl: f32, levels: u32, blur: bool) -> Outcome<Option<(Keypoints, Array<u32>)>> {
    let ctx = image.context();
    let [h, w] = plane(image, "orb")?;
    if h < 7 || w < 7 {
        return Err(invalid(format!("orb takes an image of at least 7 x 7, got {h} x {w}")));
    }
    if fast_thr.is_nan() || fast_thr <= 0.0 {
        return Err(invalid(format!("orb: fast_thr {fast_thr} is not positive")));
    }
    if max_feat == 0 {
        return Err(invalid("orb: max_feat is 0"));
    }
    if scl.is_nan() || scl <= 1.0 || scl.is_infinite() {
        return Err(invalid(format!("orb: scale_factor {scl} is not greater than 1")));
    }
    if levels == 0 {
        return Err(invalid("orb: levels is 0"));
    }
    image.storage()?;
    let lv = levels_of(h, w, max_feat, scl, levels);
    let n_levels = lv.scale.len();
    if n_levels == 0 {
        return Ok(None);
    }
    // The level images.
    let mut imgs: Vec<Array<f32>> = Vec::with_capacity(n_levels);
    imgs.push(image.clone());
    for i in 1..n_levels {
        let dst = Array::<f32>::alloc(ctx, Shape::from(lv.dims[i]))?;
        resize::launch(&imgs[i - 1], lv.dims[i - 1], &dst, lv.dims[i])?;
        imgs.push(dst);
    }
    let blurred: Vec<Option<Array<f32>>> = if blur {
        let taps = gaussian_kernel_1d_on(ctx, 9, 2.0)?;
        imgs.iter().map(|img| Some(img.conv2_sep(&taps, &taps, ConvSpec::new()))).collect()
    } else {
        vec![None; n_levels]
    };
    // Capacities.
    let caps: Vec<usize> = lv.dims.iter().map(|d| (((d[0] * d[1]) as f32 * FAST_RATIO).ceil() as usize).max(1)).collect();
    let keep: Vec<usize> = (0..n_levels).map(|i| caps[i].min(lv.best[i] as usize)).collect();
    let out_cap: usize = keep.iter().sum();
    if out_cap == 0 {
        return Ok(None);
    }
    let max_cap = *caps.iter().max().unwrap();
    let max_keep = *keep.iter().max().unwrap();
    let cand_pos = Array::<u32>::alloc(ctx, Shape::from(max_cap))?;
    let cand_h = Array::<f32>::alloc(ctx, Shape::from(max_cap))?;
    let sel_pos = Array::<u32>::alloc(ctx, Shape::from(max_cap))?;
    let sel_h = Array::<f32>::alloc(ctx, Shape::from(max_cap))?;
    let sel_g = Array::<u32>::alloc(ctx, Shape::from(max_cap))?;
    let kept_pos = Array::<u32>::alloc(ctx, Shape::from(max_keep))?;
    let kept_h = Array::<f32>::alloc(ctx, Shape::from(max_keep))?;
    let hist = Array::<u32>::zeros_on(ctx, Shape::from(n_levels * BINS as usize))?;
    let counters = Array::<u32>::zeros_on(ctx, Shape::from(4 * n_levels))?;
    let kept = Array::<u32>::zeros_on(ctx, Shape::from(n_levels))?;
    let pattern = Array::<i32>::from_slice_on(ctx, &PATTERN, Shape::from(PATTERN.len()))?;
    let out_x = Array::<f32>::alloc(ctx, Shape::from(out_cap))?;
    let out_y = Array::<f32>::alloc(ctx, Shape::from(out_cap))?;
    let out_s = Array::<f32>::alloc(ctx, Shape::from(out_cap))?;
    let out_o = Array::<f32>::alloc(ctx, Shape::from(out_cap))?;
    let out_z = Array::<f32>::alloc(ctx, Shape::from(out_cap))?;
    let out_d = Array::<u32>::alloc(ctx, Shape::from([8, out_cap]))?;
    {
        let (cp_pin, cp_bytes) = cand_pos.ready()?;
        let (ch_pin, ch_bytes) = cand_h.ready()?;
        let (sp_pin, sp_bytes) = sel_pos.ready()?;
        let (sh_pin, sh_bytes) = sel_h.ready()?;
        let (sg_pin, sg_bytes) = sel_g.ready()?;
        let (kp_pin, kp_bytes) = kept_pos.ready()?;
        let (kh_pin, kh_bytes) = kept_h.ready()?;
        let (hist_pin, hist_bytes) = hist.ready()?;
        let (ct_pin, ct_bytes) = counters.ready()?;
        let (kept_pin, kept_bytes) = kept.ready()?;
        let (pat_pin, pat_bytes) = pattern.ready()?;
        let (ox_pin, ox_bytes) = out_x.ready()?;
        let (oy_pin, oy_bytes) = out_y.ready()?;
        let (os_pin, os_bytes) = out_s.ready()?;
        let (oo_pin, oo_bytes) = out_o.ready()?;
        let (oz_pin, oz_bytes) = out_z.ready()?;
        let (od_pin, od_bytes) = out_d.ready()?;
        let launch = |kernel: Kernel| ctx.builtin(Builtin::Vision(Any::Descriptors(Descriptors::Orb(kernel))), "f32");
        for i in 0..n_levels {
            let [lh, lw] = lv.dims[i];
            if lh <= 2 * (EDGE as usize + 2) || lw <= 2 * (EDGE as usize + 2) {
                continue;
            }
            let (img_pin, img_bytes) = imgs[i].ready()?;
            let tiles_x = to_u32(lh.div_ceil(TX as usize), "tiles")?;
            let tiles_y = to_u32(lw.div_ceil(TY as usize), "tiles")?;
            let cap = to_u32(caps[i], "feature limit")?;
            let best = lv.best[i].min(cap);
            let grid = tile_grid(ctx, tiles_x, tiles_y)?;
            let fast_params = [to_u32(lh, "height")?, to_u32(lw, "width")?, fast_thr.to_bits(), EDGE, tiles_x, cap, i as u32, 0];
            let kernel = launch(Kernel::Fast)?;
            let args = [
                RawArg::read(img_pin.buffer(), img_bytes),
                RawArg::write(cp_pin.buffer(), cp_bytes),
                RawArg::write(ch_pin.buffer(), ch_bytes),
                RawArg::write(hist_pin.buffer(), hist_bytes),
                RawArg::write(ct_pin.buffer(), ct_bytes),
                RawArg::Uniform(bytemuck::bytes_of(&fast_params)),
            ];
            ctx.launch(&kernel, grid, [lh as u32, lw as u32, 1], &args)?;
            // The ranking.
            let p = [cap, best, i as u32, 0];
            let groups = cap.div_ceil(256).min(SELECT_GROUPS);
            let kernel = launch(Kernel::Select)?;
            let args = [
                RawArg::read(cp_pin.buffer(), cp_bytes),
                RawArg::read(ch_pin.buffer(), ch_bytes),
                RawArg::write(ct_pin.buffer(), ct_bytes),
                RawArg::read(hist_pin.buffer(), hist_bytes),
                RawArg::write(sp_pin.buffer(), sp_bytes),
                RawArg::write(sh_pin.buffer(), sh_bytes),
                RawArg::write(sg_pin.buffer(), sg_bytes),
                RawArg::write(kept_pin.buffer(), kept_bytes),
                RawArg::Uniform(bytemuck::bytes_of(&p)),
            ];
            ctx.launch(&kernel, [groups, 1, 1], [groups * 256, 1, 1], &args)?;
            let kernel = launch(Kernel::Rank)?;
            let args = [
                RawArg::read(sp_pin.buffer(), sp_bytes),
                RawArg::read(sh_pin.buffer(), sh_bytes),
                RawArg::read(sg_pin.buffer(), sg_bytes),
                RawArg::read(ct_pin.buffer(), ct_bytes),
                RawArg::write(kp_pin.buffer(), kp_bytes),
                RawArg::write(kh_pin.buffer(), kh_bytes),
                RawArg::Uniform(bytemuck::bytes_of(&p)),
            ];
            ctx.launch(&kernel, [groups, 1, 1], [groups * 256, 1, 1], &args)?;
            // Orientation and descriptors.
            let blurred_pin = blurred[i].as_ref().map(|b| b.ready()).transpose()?;
            let kernel = launch(Kernel::Describe { blur })?;
            let params = [to_u32(lh, "height")?, to_u32(lw, "width")?, i as u32, best, lv.scale[i].to_bits(), 0, 0, 0];
            let mut args = vec![RawArg::read(img_pin.buffer(), img_bytes)];
            if let Some((pin, bytes)) = &blurred_pin {
                args.push(RawArg::read(pin.buffer(), *bytes));
            }
            args.extend([
                RawArg::read(pat_pin.buffer(), pat_bytes),
                RawArg::read(kp_pin.buffer(), kp_bytes),
                RawArg::read(kh_pin.buffer(), kh_bytes),
                RawArg::read(kept_pin.buffer(), kept_bytes),
                RawArg::write(ox_pin.buffer(), ox_bytes),
                RawArg::write(oy_pin.buffer(), oy_bytes),
                RawArg::write(os_pin.buffer(), os_bytes),
                RawArg::write(oo_pin.buffer(), oo_bytes),
                RawArg::write(oz_pin.buffer(), oz_bytes),
                RawArg::write(od_pin.buffer(), od_bytes),
                RawArg::Uniform(bytemuck::bytes_of(&params)),
            ]);
            ctx.launch(&kernel, fold_workgroups(ctx, best)?, [best * 256, 1, 1], &args)?;
        }
    }
    let counts = kept.to_vec()?;
    let total: usize = counts.iter().map(|&c| c as usize).sum();
    if total == 0 {
        return Ok(None);
    }
    let points = Keypoints { x: out_x.slice(0, 0..total), y: out_y.slice(0, 0..total), score: out_s.slice(0, 0..total), orientation: out_o.slice(0, 0..total), size: out_z.slice(0, 0..total) };
    Ok(Some((points, out_d.slice(1, 0..total))))
}

impl Array<f32> {
    /// ORB features and descriptors of an `[H, W]` image (ArrayFire's `orb`): an image
    /// pyramid of `levels` bilinear resizes by `scale_factor`, FAST corners (arc 9, threshold
    /// `fast_thr`, non-maximal suppression) on every level, the strongest by Harris response
    /// kept per level (about `max_feat / scale^level` of them, `max_feat` in total), the
    /// orientation of the intensity centroid of the 31 x 31 patch and a 256-bit rotated BRIEF
    /// descriptor per feature.
    ///
    /// Returns the [`Keypoints`] (`x` along axis 1, `y` along axis 0, both in the coordinates of
    /// the input image; `score` is the Harris response scaled by `1e-12`, `orientation` in
    /// radians, `size` the patch size `31 * scale^level`) and the descriptors as a `[8, N]`
    /// array of `u32` (bit `k` of the 256 is word `k / 32`, bit `k % 32`); `None` if there
    /// are no features. The features of a level are sorted by decreasing Harris response, the
    /// levels follow each other, finest first. With `blur_img` the tests of the descriptor
    /// read the level image smoothed with a 9-tap Gaussian (sigma 2); the detection does not.
    ///
    /// Follows ArrayFire's CUDA backend: the number of levels stops where a level would be
    /// smaller than 31 pixels, the level images are resized from the previous level to
    /// `round(side / scale^level)`, FAST asks for features at least 22 (plus 2 for the
    /// suppression) pixels from the border and at most `0.14 * H * W` of them per level survive
    /// (after the suppression), the Harris
    /// response is the sum over a 7 x 7 block with `k = 0.04`, and the BRIEF pattern is OpenCV's
    /// `bit_pattern_31`.
    ///
    /// Differences from ArrayFire: ties of the Harris ranking and the cut at the cap of
    /// FAST candidates are decided in the order the tiles append their corners (arbitrary, as
    /// the order of ArrayFire's CUDA blocks is); ArrayFire blurs the input image in place when `blur_img` is set (luma does
    /// not touch its input); the last level takes `max_feat - sum` of the other levels' share,
    /// saturating at 0 where ArrayFire's unsigned arithmetic wraps. The ranking is exact but
    /// quadratic in the number of candidates at the cut, which stays small.
    ///
    /// The numbers of features per level are read back at the end of the call, so this
    /// **synchronizes** once; everything before it is recorded without a readback.
    ///
    /// ```
    /// use luma::Array;
    /// # fn main() -> Result<(), luma::LumaError> {
    /// // A bright square on a dark background has corners.
    /// let mut data = vec![0.0f32; 128 * 128];
    /// for x in 50..80 { for y in 50..80 { data[y + 128 * x] = 200.0; } }
    /// let img = Array::from_slice(&data, [128, 128])?;
    /// if let Some((points, desc)) = img.orb(20.0, 100, 1.5, 2, false)? {
    ///     assert_eq!(desc.shape().dims()[0], 8);
    ///     assert_eq!(desc.shape().dims()[1], points.count());
    /// }
    /// # Ok(()) }
    /// ```
    #[track_caller]
    pub fn orb(&self, fast_thr: f32, max_feat: u32, scale_factor: f32, levels: u32, blur_img: bool) -> crate::error::Result<Option<(Keypoints, Array<u32>)>> {
        let _ = std::panic::Location::caller();
        run(self, fast_thr, max_feat, scale_factor, levels, blur_img).map_err(Failure::into_error)
    }
}
