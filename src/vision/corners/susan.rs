//! The SUSAN corner detector (`Array::susan`).
//!
//! The USAN area of a pixel is `n = sum exp(-((c - v) / t)^6)` over the pixels `v` of the disc
//! `i^2 + j^2 < radius^2` around it (the pixel itself included), `c` being the centre's value and
//! `t` the intensity threshold. A pixel with `n < g` has the response `g - n`; a corner is a
//! pixel of positive response strictly larger than its eight neighbours. As for Harris one
//! kernel stages the image tile with its halo, computes the responses of the tile and its border
//! and lists the maxima of every tile.

use std::fmt::Write as _;
use std::panic::Location;

use super::detect::{self, Detector, Tile};
use super::Kernel as Corners;
use crate::array::{Array, Failure};
use crate::error::Result;
use crate::ops::Outcome;
use crate::vision::{invalid, plane, to_u32, Keypoints};

/// Largest radius (ArrayFire's limit).
const MAX_RADIUS: usize = 9;

/// One kernel: the radius and the tile are literals.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) struct Kernel {
    pub(crate) radius: u8,
    pub(crate) tile: Tile,
}

fn shared_bytes(radius: u32, tile: Tile) -> u64 {
    let (tx, ty) = (u32::from(tile.tx), u32::from(tile.ty));
    let words = (tx + 2 + 2 * radius) * (ty + 2 + 2 * radius) + (tx + 2) * (ty + 2) + tile.threads() + 1;
    u64::from(words) * 4
}

/// The WGSL of a kernel. Arguments: image, counts, slot indices, slot scores, uniform
/// `{h, w, tiles_x, lo, 1 / t, g, c, c}`.
pub(crate) fn source(k: Kernel) -> String {
    let r = u32::from(k.radius);
    let tile = k.tile;
    let (tx, ty, nth) = (u32::from(tile.tx), u32::from(tile.ty), tile.threads());
    let (ni0, ni1) = (tx + 2 + 2 * r, ty + 2 + 2 * r);
    let (nw0, nw1) = (tx + 2, ty + 2);
    let (ni, nw) = (ni0 * ni1, nw0 * nw1);
    let mut s = String::new();
    s.push_str("struct Params { h: u32, w: u32, tiles_x: u32, lo: u32, inv_t: f32, g: f32, c0: u32, c1: u32 }\n");
    s.push_str(&detect::declarations(tile));
    s.push_str("@group(1) @binding(4) var<uniform> p: Params;\n");
    let _ = writeln!(s, "var<workgroup> img: array<f32, {ni}>;");
    let _ = writeln!(s, "var<workgroup> rr: array<f32, {nw}>;");
    s.push_str("fn term(c: f32, v: f32) -> f32 {\n    let d = (c - v) * p.inv_t;\n    let d2 = d * d;\n    return exp(-(d2 * d2 * d2));\n}\n");
    let _ = write!(
        s,
        "@compute @workgroup_size({nth})
fn main(@builtin(local_invocation_index) t: u32, @builtin(workgroup_id) wid: vec3<u32>) {{
    let tile_id = wid.x + wid.y * p.tiles_x;
    let lo = i32(p.lo);
    let o0 = i32(wid.x) * {tx} + lo;
    let o1 = i32(wid.y) * {ty} + lo;
    let h = i32(p.h);
    let w = i32(p.w);
    if (t == 0u) {{ atomicStore(&cnt, 0u); }}
    for (var i = t; i < {ni}u; i += {nth}u) {{
        let a0 = o0 - {off} + i32(i % {ni0}u);
        let a1 = o1 - {off} + i32(i / {ni0}u);
        var v = 0.0;
        if (a0 >= 0 && a0 < h && a1 >= 0 && a1 < w) {{ v = src[u32(a0) + p.h * u32(a1)]; }}
        img[i] = v;
    }}
    workgroupBarrier();
    for (var i = t; i < {nw}u; i += {nth}u) {{
        let v0 = i % {nw0}u;
        let v1 = i / {nw0}u;
        let m = (v1 + {r}u) * {ni0}u + v0 + {r}u;
        let c = img[m];
        var n = 0.0;
",
        off = 1 + r
    );
    let ri = r as i32;
    for db in -ri..=ri {
        for da in -ri..=ri {
            if da * da + db * db < ri * ri {
                let off = db * ni0 as i32 + da;
                let at = if off >= 0 { format!("m + {off}u") } else { format!("m - {}u", -off) };
                let _ = writeln!(s, "        n += term(c, img[{at}]);");
            }
        }
    }
    let _ = write!(
        s,
        "        let q0 = o0 - 1 + i32(v0);
        let q1 = o1 - 1 + i32(v1);
        var r = 0.0;
        if (q0 < h - lo + 1 && q1 < w - lo + 1 && n < p.g) {{ r = p.g - n; }}
        rr[i] = r;
    }}
    workgroupBarrier();
"
    );
    s.push_str(&detect::epilogue(tile, nw0, ""));
    let _ = nw1;
    s
}

impl Array<f32> {
    /// The SUSAN corners of an `[H, W]` image (ArrayFire's `susan`): the pixels with a positive
    /// response `geom_thr - n` that is strictly larger than that of their eight neighbours,
    /// where `n = sum exp(-((c - v) / diff_thr)^6)` is the area of similar pixels in the disc of
    /// `radius` around the pixel (the pixels `v` with `i^2 + j^2 < radius^2` of offsets `(i, j)`,
    /// the pixel `c` itself included) and only pixels with `n < geom_thr` have a response.
    ///
    /// Pixels closer than `edge` to the border have no response, and a corner also needs its
    /// neighbours' responses, so corners lie at least `edge + 1` cells from the border. `radius`
    /// is at most 9 and at most `edge`; `diff_thr` and `geom_thr` are positive;
    /// `feature_ratio` is in `(0, 1]` and bounds the number of corners by
    /// `floor(feature_ratio * H * W)`: if there are more, the first ones in memory order (axis 0
    /// fastest) are kept, as ArrayFire's CPU backend does (its CUDA backend keeps an arbitrary
    /// subset). `score` is the response, `orientation` 0 and `size` 1.
    ///
    /// Differences from ArrayFire: `x` is the index along axis 1 (the column) and `y` the one
    /// along axis 0, like every detector of luma, while ArrayFire's `susan` returns them the
    /// other way round (its `x` is the row); corners not cut by the limit come in the order of
    /// the tiles of 32 x 8 pixels like ArrayFire's CUDA backend, not in memory order like its
    /// CPU backend; an `edge` larger than half the image gives `None` instead of an error; the
    /// input must be an `[H, W]` array of `f32` (ArrayFire takes the integer types too).
    ///
    /// The number of corners is read back, so this **synchronizes** once.
    ///
    /// ```
    /// use luma::Array;
    /// # fn main() -> Result<(), luma::LumaError> {
    /// // A bright square on a dark image: its four corners.
    /// let mut data = vec![0.0f32; 32 * 32];
    /// for j in 10..20 {
    ///     for i in 8..18 {
    ///         data[i + 32 * j] = 200.0;
    ///     }
    /// }
    /// let img = Array::from_slice(&data, [32, 32])?;
    /// let corners = img.susan(3, 32.0, 10.0, 0.5, 3)?.expect("corners");
    /// assert_eq!(corners.count(), 4);
    /// # Ok(()) }
    /// ```
    #[track_caller]
    pub fn susan(&self, radius: usize, diff_thr: f32, geom_thr: f32, feature_ratio: f32, edge: usize) -> Result<Option<Keypoints>> {
        let _ = Location::caller();
        run(self, radius, diff_thr, geom_thr, feature_ratio, edge).map_err(Failure::into_error)
    }
}

fn run(image: &Array<f32>, radius: usize, diff_thr: f32, geom_thr: f32, ratio: f32, edge: usize) -> Outcome<Option<Keypoints>> {
    let ctx = image.context();
    let [h, w] = plane(image, "susan")?;
    if radius > MAX_RADIUS {
        return Err(invalid(format!("susan: radius {radius} is more than {MAX_RADIUS}")));
    }
    if edge < radius {
        return Err(invalid(format!("susan: edge {edge} is less than the radius {radius}")));
    }
    if diff_thr.is_nan() || diff_thr <= 0.0 {
        return Err(invalid(format!("susan: diff_thr {diff_thr} must be positive")));
    }
    if geom_thr.is_nan() || geom_thr <= 0.0 {
        return Err(invalid(format!("susan: geom_thr {geom_thr} must be positive")));
    }
    if !(ratio > 0.0 && ratio <= 1.0) {
        return Err(invalid(format!("susan: feature_ratio {ratio} is not in (0, 1]")));
    }
    image.storage()?;
    let lo = edge.saturating_add(1);
    let limit = ((h * w) as f64 * f64::from(ratio)).floor() as usize;
    if h / 2 < lo || w / 2 < lo || h <= 2 * lo || w <= 2 * lo || limit == 0 {
        return Ok(None);
    }
    let tile = detect::plan_tile(ctx.tuning().workgroup_storage, |t| shared_bytes(radius as u32, t), "a SUSAN disc")?;
    let tiles_x = to_u32((h - 2 * lo).div_ceil(usize::from(tile.tx)), "tiles")?;
    let params = vec![to_u32(h, "height")?, to_u32(w, "width")?, tiles_x, to_u32(lo, "edge")?, (1.0 / diff_thr).to_bits(), geom_thr.to_bits(), 0, 0];
    let detector = Detector { image, kernel: Corners::Susan(Kernel { radius: radius as u8, tile }), params, tile, lo, swap: false, limit: Some(limit) };
    let Some(found) = detect::run(&detector)? else {
        return Ok(None);
    };
    let n = found.len();
    Ok(Some(Keypoints::plain(ctx, &found.x, &found.y, &found.score, n)?))
}
