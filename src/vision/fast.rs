//! The FAST corner detector (`Array::fast`).
//!
//! A pixel is a corner if `arc_length` consecutive pixels of the 16 on the circle of radius 3
//! around it are all brighter than `p + threshold` or all darker than `p - threshold`. Its score
//! is the larger of the two sums `sum(r - p - threshold)` over the brighter ring pixels and
//! `sum(p - r - threshold)` over the darker ones (the ring pixels past the arc count too, as in
//! ArrayFire). With non-maximal suppression a corner stays if its score is strictly larger than
//! those of its eight neighbours.
//!
//! Two launches over tiles of 32 x 8 pixels do the detection and one tiny scan the compaction: the
//! first counts the corners of every tile, the scan turns the counts into offsets, the second
//! recomputes the scores and writes every tile's points at its offset (in the order of the tile's
//! pixels, axis 0 fastest). Nothing of the score image is stored; a tile without corners does no
//! more than the count. A tile stages its pixels, its ring and the pixels of the neighbours'
//! rings in workgroup memory, scores the tile and its one-pixel border once and then decides.

use std::fmt::Write as _;
use std::panic::Location;

use super::{invalid, plane, scan_counts, tile_grid, to_u32, Kernel as Any, Keypoints};
use crate::array::{Array, Failure};
use crate::builtin::Builtin;
use crate::context::RawArg;
use crate::error::Result;
use crate::ops::Outcome;
use crate::shape::Shape;

/// Pixels of a tile along axis 0 and axis 1.
const TX: u32 = 32;
const TY: u32 = 8;

/// One kernel: the arc length is a literal; the counting kernel only counts.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) struct Kernel {
    pub(crate) arc: u8,
    pub(crate) scatter: bool,
}

/// The ring of the detector, as offsets along axis 0 and 1, clockwise from the top.
const RING: [(i32, i32); 16] = [(3, 0), (3, 1), (2, 2), (1, 3), (0, 3), (-1, 3), (-2, 2), (-3, 1), (-3, 0), (-3, -1), (-2, -2), (-1, -3), (0, -3), (1, -3), (2, -2), (3, -1)];

/// The WGSL of a kernel.
pub(crate) fn source(k: Kernel) -> String {
    if k.scatter {
        scatter_source()
    } else {
        detect_source(k.arc)
    }
}

/// The detection kernel. Arguments: image, counts (one per tile), the score image (the score of
/// the corners that stay, 0 elsewhere), uniform.
fn detect_source(arc: u8) -> String {
    let (halo_in, halo_sc) = (4u32, 1u32);
    let (iw, ih) = (TX + 2 * halo_in, TY + 2 * halo_in);
    let (sw, sh) = (TX + 2 * halo_sc, TY + 2 * halo_sc);
    let mut s = String::new();
    s.push_str("struct Params { h: u32, w: u32, thr: f32, edge: u32, tiles_x: u32, cap: u32, nonmax: u32, c: u32 }\n");
    s.push_str("@group(1) @binding(0) var<storage, read> src: array<f32>;\n");
    s.push_str("@group(1) @binding(1) var<storage, read_write> counts: array<u32>;\n");
    s.push_str("@group(1) @binding(2) var<storage, read_write> img: array<f32>;\n");
    s.push_str("@group(1) @binding(3) var<uniform> p: Params;\n");
    let _ = writeln!(s, "var<workgroup> tile: array<f32, {}>;", iw * ih);
    let _ = writeln!(s, "var<workgroup> sc: array<f32, {}>;", sw * sh);
    s.push_str("var<workgroup> cnt: atomic<u32>;\n");
    // The score of the pixel at (cx, cy) of the staged tile.
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
    // Cardinal points: an arc of 9 or more contains at least two of the four.
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
    let _ = writeln!(s, "    var b2 = bm | (bm << 16u); var d2 = dm | (dm << 16u);");
    let _ = writeln!(s, "    var tb = b2; var td = d2;");
    for i in 1..arc {
        let _ = writeln!(s, "    tb &= b2 >> {i}u; td &= d2 >> {i}u;");
    }
    s.push_str("    var out = 0.0;\n    if ((tb & 65535u) != 0u) { out = sb; }\n    if ((td & 65535u) != 0u) { out = max(out, sd); }\n    return out;\n}\n");
    let _ = write!(
        s,
        "@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) t: u32, @builtin(workgroup_id) wid: vec3<u32>) {{
    let tile_id = wid.x + wid.y * p.tiles_x;
    let o0 = i32(wid.x) * {TX} - {halo_in};
    let o1 = i32(wid.y) * {TY} - {halo_in};
    let h = i32(p.h);
    let w = i32(p.w);
    if (t == 0u) {{ atomicStore(&cnt, 0u); }}
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
    let s0 = sc[(ty + 1u) * {sw}u + tx + 1u];
    var keep = s0 > 0.0;
    if (keep && p.nonmax != 0u) {{
        let base = ty * {sw}u + tx;
        keep = s0 > sc[base] && s0 > sc[base + 1u] && s0 > sc[base + 2u]
            && s0 > sc[base + {sw}u] && s0 > sc[base + {sw}u + 2u]
            && s0 > sc[base + 2u * {sw}u] && s0 > sc[base + 2u * {sw}u + 1u] && s0 > sc[base + 2u * {sw}u + 2u];
    }}
    if (keep) {{ atomicAdd(&cnt, 1u); }}
    let g0 = i32(wid.x) * {TX} + i32(tx);
    let g1 = i32(wid.y) * {TY} + i32(ty);
    if (g0 < h && g1 < w) {{ img[u32(g0) + p.h * u32(g1)] = select(0.0, s0, keep); }}
    workgroupBarrier();
    if (t == 0u) {{ counts[tile_id] = atomicLoad(&cnt); }}
}}
",
        total = iw * ih,
        sc_total = sw * sh,
        d = halo_in - halo_sc
    );
    s
}

/// The kernel writing the points. Arguments: the score image, the offsets (the scanned counts),
/// `x`, `y`, `score`, uniform.
fn scatter_source() -> String {
    format!(
        "struct Params {{ h: u32, w: u32, thr: f32, edge: u32, tiles_x: u32, cap: u32, nonmax: u32, c: u32 }}
@group(1) @binding(0) var<storage, read> img: array<f32>;
@group(1) @binding(1) var<storage, read> offs: array<u32>;
@group(1) @binding(2) var<storage, read_write> out_x: array<f32>;
@group(1) @binding(3) var<storage, read_write> out_y: array<f32>;
@group(1) @binding(4) var<storage, read_write> out_s: array<f32>;
@group(1) @binding(5) var<uniform> p: Params;
var<workgroup> rank: array<u32, 256>;
var<workgroup> tot: u32;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) t: u32, @builtin(workgroup_id) wid: vec3<u32>) {{
    let tile_id = wid.x + wid.y * p.tiles_x;
    if (t == 0u) {{ tot = offs[tile_id + 1u] - offs[tile_id]; }}
    workgroupBarrier();
    if (workgroupUniformLoad(&tot) == 0u) {{ return; }}
    let tx = t & 31u;
    let ty = t >> 5u;
    let g0 = wid.x * {TX}u + tx;
    let g1 = wid.y * {TY}u + ty;
    var s0 = 0.0;
    if (g0 < p.h && g1 < p.w) {{ s0 = img[g0 + p.h * g1]; }}
    let keep = s0 > 0.0;
    rank[t] = select(0u, 1u, keep);
    workgroupBarrier();
    for (var d = 1u; d < 256u; d = d << 1u) {{
        var add = 0u;
        if (t >= d) {{ add = rank[t - d]; }}
        workgroupBarrier();
        rank[t] = rank[t] + add;
        workgroupBarrier();
    }}
    if (keep) {{
        let k = offs[tile_id] + rank[t] - 1u;
        if (k < p.cap) {{
            out_x[k] = f32(g1);
            out_y[k] = f32(g0);
            out_s[k] = s0;
        }}
    }}
}}
"
    )
}

impl Array<f32> {
    /// The FAST corners of an `[H, W]` image (ArrayFire's `fast`): the pixels with `arc_length`
    /// (9 to 16) consecutive pixels of the circle of radius 3 around them that are all brighter
    /// than the pixel plus `threshold` or all darker than it minus `threshold`.
    ///
    /// The score of a corner is the larger of the sums of `|ring - pixel| - threshold` over the
    /// brighter and over the darker ring pixels. With `non_max` a corner has to score strictly
    /// more than its eight neighbours. `edge` (at least 3, the radius of the circle) is the
    /// width of the border of the image without corners. At most `ceil(feature_ratio * H * W)`
    /// points are returned, the ones of the first tiles of 32 x 8 pixels (tiles along axis 0
    /// first, in a tile axis 0 first); `None` if there are none. `orientation` is 0 and `size`
    /// 1.
    ///
    /// Differences from ArrayFire: the limit applies after the non-maximal suppression (ArrayFire's
    /// CPU backend limits the candidates before it), the order of the points is the one of the
    /// tiles, not row by row, and a point one pixel inside `edge` is not dropped.
    ///
    /// The count of points is read back, so this **synchronizes** once; everything before is
    /// recorded as three launches (count the points of every tile, scan the counts, write them).
    ///
    /// ```
    /// use luma::Array;
    /// # fn main() -> Result<(), luma::LumaError> {
    /// // A bright dot on a dark image is a corner.
    /// let mut data = vec![0.0f32; 16 * 16];
    /// data[8 + 16 * 8] = 100.0;
    /// let img = Array::from_slice(&data, [16, 16])?;
    /// let corners = img.fast(20.0, 9, true, 0.1, 3)?.expect("one corner");
    /// assert_eq!(corners.x.to_vec()?, [8.0]);
    /// assert_eq!(corners.y.to_vec()?, [8.0]);
    /// # Ok(()) }
    /// ```
    #[track_caller]
    pub fn fast(&self, threshold: f32, arc_length: u32, non_max: bool, feature_ratio: f32, edge: usize) -> Result<Option<Keypoints>> {
        let _ = Location::caller();
        run(self, threshold, arc_length, non_max, feature_ratio, edge).map_err(Failure::into_error)
    }
}

fn run(image: &Array<f32>, threshold: f32, arc: u32, non_max: bool, ratio: f32, edge: usize) -> Outcome<Option<Keypoints>> {
    let ctx = image.context();
    let [h, w] = plane(image, "fast")?;
    if !(9..=16).contains(&arc) {
        return Err(invalid(format!("fast: arc_length {arc} is not in 9..=16")));
    }
    if edge < 3 {
        return Err(invalid(format!("fast: edge {edge} is less than the radius 3 of the circle")));
    }
    if !(ratio > 0.0 && ratio <= 1.0) || threshold.is_nan() {
        return Err(invalid(format!("fast: feature_ratio {ratio} is not in (0, 1]")));
    }
    image.storage()?;
    if h <= 2 * edge || w <= 2 * edge {
        return Ok(None);
    }
    let cap = ((h * w) as f64 * f64::from(ratio)).ceil().max(1.0) as usize;
    let (tiles_x, tiles_y) = (to_u32(h.div_ceil(TX as usize), "tiles")?, to_u32(w.div_ceil(TY as usize), "tiles")?);
    let tiles = tiles_x * tiles_y;
    let grid = tile_grid(ctx, tiles_x, tiles_y)?;
    let counts = Array::<u32>::alloc(ctx, Shape::from(tiles as usize + 1))?;
    let xs = Array::<f32>::alloc(ctx, Shape::from(cap))?;
    let ys = Array::<f32>::alloc(ctx, Shape::from(cap))?;
    let scores = Array::<f32>::alloc(ctx, Shape::from(cap))?;
    let img = Array::<f32>::alloc(ctx, image.shape())?;
    let params = [to_u32(h, "height")?, to_u32(w, "width")?, threshold.to_bits(), to_u32(edge, "edge")?, tiles_x, to_u32(cap, "feature limit")?, u32::from(non_max), 0];
    {
        let (src, src_bytes) = image.ready()?;
        let counts_pin = counts.ready()?;
        let img_pin = img.ready()?;
        let kernel = ctx.builtin(Builtin::Vision(Any::Fast(Kernel { arc: arc as u8, scatter: false })), "f32")?;
        let args = [
            RawArg::read(src.buffer(), src_bytes),
            RawArg::write(counts_pin.0.buffer(), counts_pin.1),
            RawArg::write(img_pin.0.buffer(), img_pin.1),
            RawArg::Uniform(bytemuck::bytes_of(&params)),
        ];
        ctx.launch(&kernel, grid, [h as u32, w as u32, 1], &args)?;
        scan_counts(ctx, &counts_pin, tiles)?;
        let kernel = ctx.builtin(Builtin::Vision(Any::Fast(Kernel { arc: 0, scatter: true })), "f32")?;
        let (x_pin, y_pin, s_pin) = (xs.ready()?, ys.ready()?, scores.ready()?);
        let args = [
            RawArg::read(img_pin.0.buffer(), img_pin.1),
            RawArg::read(counts_pin.0.buffer(), counts_pin.1),
            RawArg::write(x_pin.0.buffer(), x_pin.1),
            RawArg::write(y_pin.0.buffer(), y_pin.1),
            RawArg::write(s_pin.0.buffer(), s_pin.1),
            RawArg::Uniform(bytemuck::bytes_of(&params)),
        ];
        ctx.launch(&kernel, grid, [h as u32, w as u32, 1], &args)?;
    }
    let total = counts.slice(0, tiles as usize..tiles as usize + 1).item()? as usize;
    let n = total.min(cap);
    if n == 0 {
        return Ok(None);
    }
    Ok(Some(Keypoints::plain(ctx, &xs, &ys, &scores, n)?))
}
