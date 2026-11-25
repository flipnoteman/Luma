//! The Harris corner detector (`Array::harris`).
//!
//! The response of a pixel is `det(M) - k * trace(M)^2` of the structure tensor `M` of the
//! central differences `(in[i + 1] - in[i - 1]) / 2` along both axes, summed with a window: a
//! Gaussian (`block_size == 0`, its length derived from `sigma`) or a box (or a Gaussian of
//! `block_size` cells). The response is only defined on the pixels whose window lies inside the
//! image; a corner is a pixel of positive response that is strictly larger than its eight
//! neighbours. One kernel does everything up to the lists of corners of every tile of
//! 32 x 8 pixels: it stages the image, forms the two derivatives, runs the separable window over
//! the three products `Ixx`, `Iyy`, `Ixy` (the first pass of one product overlapping the second
//! pass of the previous one) and decides on the response tile, all in workgroup memory.

use std::fmt::Write as _;
use std::panic::Location;

use super::detect::{self, Detector, Tile};
use super::Kernel as Corners;
use crate::jit::structured::opaque::opaque;
use crate::array::{Array, Failure};
use crate::error::Result;
use crate::ops::Outcome;
use crate::vision::{invalid, plane, to_u32, Keypoints};

/// Longest window (accepted by ArrayFire: `block_size` up to 32).
const MAX_WINDOW: usize = 32;

/// One kernel: the window length and the tile are literals.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) struct Kernel {
    pub(crate) n: u8,
    pub(crate) tile: Tile,
}

/// Geometry of the tiles of the staged arrays, `n` the window length.
struct Layout {
    ni0: u32,
    ni1: u32,
    ng0: u32,
    ng1: u32,
    nt0: u32,
    nt1: u32,
    nw0: u32,
    nw1: u32,
}

fn layout(n: u32, tile: Tile) -> Layout {
    let (tx, ty) = (u32::from(tile.tx), u32::from(tile.ty));
    Layout { ni0: tx + n + 3, ni1: ty + n + 3, ng0: tx + n + 1, ng1: ty + n + 1, nt0: tx + 2, nt1: ty + n + 1, nw0: tx + 2, nw1: ty + 2 }
}

/// Bytes of workgroup memory of a kernel.
fn shared_bytes(n: u32, tile: Tile) -> u64 {
    let l = layout(n, tile);
    let words = l.ni0 * l.ni1 + 2 * l.ng0 * l.ng1 + 2 * l.nt0 * l.nt1 + 3 * l.nw0 * l.nw1 + tile.threads() + 1;
    u64::from(words) * 4
}

/// The window length for a Gaussian of standard deviation `sigma`: `floor(6 sigma)`, made odd.
fn window_of_sigma(sigma: f32) -> usize {
    let n = (6.0f32 * sigma).floor() as usize;
    if n.is_multiple_of(2) {
        n - 1
    } else {
        n
    }
}

/// The WGSL of a kernel. Arguments: image, counts, slot indices, slot scores, uniform
/// `{h, w, tiles_x, c, min_response, k, c, c, taps: 8 x vec4}` (see [`words`]).
pub(crate) fn source(k: Kernel) -> String {
    let n = u32::from(k.n);
    let tile = k.tile;
    let (tx, ty, nth) = (u32::from(tile.tx), u32::from(tile.ty), tile.threads());
    let c = n / 2;
    let lw = n - 1 - c;
    let border = n / 2 + 1;
    let lo = border + 1;
    let l = layout(n, tile);
    let (ni, ng, nt, nw) = (l.ni0 * l.ni1, l.ng0 * l.ng1, l.nt0 * l.nt1, l.nw0 * l.nw1);
    let mut s = String::new();
    s.push_str("struct Params { h: u32, w: u32, tiles_x: u32, c0: u32, minr: f32, k: f32, c1: u32, c2: u32, taps: array<vec4<f32>, 8> }\n");
    s.push_str(&detect::declarations(tile));
    s.push_str("@group(1) @binding(4) var<uniform> p: Params;\n");
    let _ = writeln!(s, "var<workgroup> img: array<f32, {ni}>;");
    let _ = writeln!(s, "var<workgroup> gx: array<f32, {ng}>;");
    let _ = writeln!(s, "var<workgroup> gy: array<f32, {ng}>;");
    let _ = writeln!(s, "var<workgroup> ta: array<f32, {nt}>;");
    let _ = writeln!(s, "var<workgroup> tb: array<f32, {nt}>;");
    let _ = writeln!(s, "var<workgroup> rr: array<f32, {nw}>;");
    let _ = writeln!(s, "var<workgroup> wyy: array<f32, {nw}>;");
    let _ = writeln!(s, "var<workgroup> wxy: array<f32, {nw}>;");
    let tap = |a: u32| format!("p.taps[{}u].{}", a / 4, ["x", "y", "z", "w"][(a % 4) as usize]);
    // The first pass of product `m` (0 xx, 1 yy, 2 xy) into `dst`.
    let pass1 = |m: u32, dst: &str| {
        let mut b = String::new();
        let _ = writeln!(
            b,
            "    for (var i = t; i < {nt}u; i += {nth}u) {{
        let base = (i / {nt0}u) * {ng0}u + (i % {nt0}u);
        var acc = 0.0;",
            nt0 = l.nt0,
            ng0 = l.ng0
        );
        for a in 0..n {
            let at = n - 1 - a;
            let term = match m {
                0 => format!("let u = gx[base + {at}u]; acc += {} * (u * u);", tap(a)),
                1 => format!("let u = gy[base + {at}u]; acc += {} * (u * u);", tap(a)),
                _ => format!("acc += {} * (gx[base + {at}u] * gy[base + {at}u]);", tap(a)),
            };
            let _ = writeln!(b, "        {{ {term} }}");
        }
        let _ = writeln!(b, "        {dst}[i] = acc;\n    }}");
        b
    };
    // The second pass of a product from `src` into `dst`.
    let pass2 = |src: &str, dst: &str| {
        let mut b = String::new();
        let _ = writeln!(b, "    for (var i = t; i < {nw}u; i += {nth}u) {{\n        var acc = 0.0;");
        for a in 0..n {
            let _ = writeln!(b, "        acc += {} * {src}[i + {}u];", tap(a), (n - 1 - a) * l.nt0);
        }
        let _ = writeln!(b, "        {dst}[i] = acc;\n    }}");
        b
    };
    let _ = write!(
        s,
        "@compute @workgroup_size({nth})
fn main(@builtin(local_invocation_index) t: u32, @builtin(workgroup_id) wid: vec3<u32>) {{
    let tile_id = wid.x + wid.y * p.tiles_x;
    let lo = {lo};
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
    for (var i = t; i < {ng}u; i += {nth}u) {{
        let m = (i / {ng0}u + 1u) * {ni0}u + (i % {ng0}u) + 1u;
        gx[i] = (img[m + {ni0}u] - img[m - {ni0}u]) * 0.5;
        gy[i] = (img[m + 1u] - img[m - 1u]) * 0.5;
    }}
    workgroupBarrier();
",
        off = 2 + lw,
        ni0 = l.ni0,
        ng0 = l.ng0
    );
    s.push_str(&pass1(0, "ta"));
    s.push_str("    workgroupBarrier();\n");
    s.push_str(&pass2("ta", "rr"));
    s.push_str(&pass1(1, "tb"));
    s.push_str("    workgroupBarrier();\n");
    s.push_str(&pass2("tb", "wyy"));
    s.push_str(&pass1(2, "ta"));
    s.push_str("    workgroupBarrier();\n");
    s.push_str(&pass2("ta", "wxy"));
    let _ = write!(
        s,
        "    for (var i = t; i < {nw}u; i += {nth}u) {{
        let q0 = o0 - 1 + i32(i % {nw0}u);
        let q1 = o1 - 1 + i32(i / {nw0}u);
        let a = rr[i];
        let d = wyy[i];
        let e = wxy[i];
        let tr = a + d;
        let r = a * d - e * e - p.k * (tr * tr);
        rr[i] = select(0.0, r, q0 < h - {border} && q1 < w - {border});
    }}
    workgroupBarrier();
",
        nw0 = l.nw0
    );
    s.push_str(&detect::epilogue(tile, l.nw0, " && s0 >= p.minr"));
    let _ = (l.ni1, l.ng1, l.nw1);
    s
}

/// The words of the uniform of a kernel: sizes, response threshold, `k` and the window taps.
fn words(h: usize, w: usize, tiles_x: u32, min_response: f32, k: f32, taps: &[f32]) -> std::result::Result<Vec<u32>, Failure> {
    let mut v = vec![to_u32(h, "height")?, to_u32(w, "width")?, tiles_x, 0, min_response.to_bits(), k.to_bits(), 0, 0];
    v.extend(taps.iter().map(|t| t.to_bits()));
    v.resize(8 + 32, 0);
    Ok(v)
}

impl Array<f32> {
    /// The Harris corners of an `[H, W]` image (ArrayFire's `harris`): the pixels whose Harris
    /// response `det(M) - k_thr * trace(M)^2` is positive and strictly larger than that of their
    /// eight neighbours. `M` is the structure tensor of the central differences `(in[i + 1] -
    /// in[i - 1]) / 2`, summed over a window around the pixel.
    ///
    /// The window is a Gaussian of `floor(6 * sigma)` cells (made odd by one less; `sigma` in
    /// `[0.5, 5]`) if `block_size` is 0, otherwise `block_size` cells (1 to 32) with the Gaussian
    /// weights of `sigma` if `sigma >= 0.5` and equal weights if it is smaller, as in ArrayFire
    /// (whose documentation only mentions the box and a `block_size` of 3 to 31). The window
    /// is convolved with the centre at cell `block_size / 2`, and a pixel is only a corner if
    /// its window and the windows of its neighbours lie inside the image (that is, at least
    /// `window / 2 + 2` cells from the border).
    ///
    /// With `max_corners > 0` at most that many corners are returned, the strongest ones, in
    /// descending order of response (ties in the order of the corners); `min_response` is not
    /// used. With `max_corners == 0` all corners with a response of at least `min_response`
    /// (which must then be positive) are returned. `k_thr` is at least 0.01. The `score` is the
    /// response, `orientation` is 0 and `size` 1.
    ///
    /// Differences from ArrayFire: the corners that are not ranked by response come in the order
    /// of the tiles of 32 x 8 pixels (axis 0 fastest), like ArrayFire's CUDA backend, not in
    /// column order like its CPU backend; an image too small for any corner gives `None`
    /// (ArrayFire raises an error for some sizes); the input must be an `[H, W]` array of `f32`
    /// (ArrayFire's `harris` takes `f32` and `f64`).
    ///
    /// The number of corners is read back, so this **synchronizes** once; everything before is
    /// recorded as a fused detection kernel, a scan of the per-tile counts and a gather (and, with
    /// `max_corners`, one kernel that selects and sorts the strongest).
    ///
    /// ```
    /// use luma::Array;
    /// # fn main() -> Result<(), luma::LumaError> {
    /// // A bright square on a dark image has four corners.
    /// let mut data = vec![0.0f32; 32 * 32];
    /// for j in 12..20 {
    ///     for i in 12..20 {
    ///         data[i + 32 * j] = 100.0;
    ///     }
    /// }
    /// let img = Array::from_slice(&data, [32, 32])?;
    /// let corners = img.harris(10, 0.0, 1.0, 0, 0.04)?.expect("corners");
    /// assert_eq!(corners.count(), 4);
    /// # Ok(()) }
    /// ```
    #[track_caller]
    pub fn harris(&self, max_corners: usize, min_response: f32, sigma: f32, block_size: usize, k_thr: f32) -> Result<Option<Keypoints>> {
        opaque!(self.context(), VISION_HARRIS, "harris", Location::caller());
        run(self, max_corners, min_response, sigma, block_size, k_thr).map_err(Failure::into_error)
    }
}

fn run(image: &Array<f32>, max_corners: usize, min_response: f32, sigma: f32, block_size: usize, k_thr: f32) -> Outcome<Option<Keypoints>> {
    let ctx = image.context();
    let [h, w] = plane(image, "harris")?;
    if max_corners == 0 && (min_response.is_nan() || min_response <= 0.0) {
        return Err(invalid(format!("harris: min_response {min_response} must be positive when max_corners is 0")));
    }
    if block_size > MAX_WINDOW {
        return Err(invalid(format!("harris: block_size {block_size} is more than {MAX_WINDOW}")));
    }
    if block_size == 0 && !(0.5..=5.0).contains(&sigma) {
        return Err(invalid(format!("harris: sigma {sigma} is not in [0.5, 5] (block_size is 0)")));
    }
    if k_thr.is_nan() || k_thr < 0.01 {
        return Err(invalid(format!("harris: k_thr {k_thr} is less than 0.01")));
    }
    if block_size > 0 && sigma.is_nan() {
        return Err(invalid("harris: sigma is NaN"));
    }
    image.storage()?;
    let n = if block_size == 0 { window_of_sigma(sigma) } else { block_size };
    let taps: Vec<f32> = if sigma >= 0.5 {
        // ArrayFire's `gaussian1D`: the centre is at `(n - 1) / 2` (integer division).
        let centre = ((n - 1) / 2) as f64;
        let line: Vec<f64> = (0..n).map(|i| (-(i as f64 - centre).powi(2) / (2.0 * f64::from(sigma) * f64::from(sigma))).exp()).collect();
        let total: f64 = line.iter().sum();
        line.iter().map(|v| (v / total) as f32).collect()
    } else {
        vec![1.0 / n as f32; n]
    };
    let lo = n / 2 + 2;
    if h <= 2 * lo || w <= 2 * lo {
        return Ok(None);
    }
    let limit = ctx.tuning().workgroup_storage;
    let tile = detect::plan_tile(limit, |t| shared_bytes(n as u32, t), "a Harris window")?;
    let tiles_x = to_u32((h - 2 * lo).div_ceil(usize::from(tile.tx)), "tiles")?;
    let min = if max_corners == 0 { min_response } else { 0.0 };
    let detector = Detector {
        image,
        kernel: Corners::Harris(Kernel { n: n as u8, tile }),
        params: words(h, w, tiles_x, min, k_thr, &taps)?,
        tile,
        lo,
        swap: false,
        limit: None,
    };
    let Some(found) = detect::run(&detector)? else {
        return Ok(None);
    };
    let count = found.len();
    if max_corners > 0 && count > max_corners {
        if let Some((x, y, score)) = super::select::strongest(&found.x, &found.y, &found.score, max_corners)? {
            return Ok(Some(Keypoints::plain(ctx, &x, &y, &score, max_corners)?));
        }
        let (score, order) = found.score.topk(max_corners);
        let (x, y) = (found.x.take(&order), found.y.take(&order));
        return Ok(Some(Keypoints::plain(ctx, &x, &y, &score, max_corners)?));
    }
    Ok(Some(Keypoints::plain(ctx, &found.x, &found.y, &found.score, count)?))
}
