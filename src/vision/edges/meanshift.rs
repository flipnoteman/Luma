//! Mean shift (`Array::mean_shift`).
//!
//! Every pixel walks: its window (the `2 r + 1` square around the current position, cut at the
//! image) is searched for the pixels whose colour is within the chromatic radius of the current
//! colour; their mean position and mean colour become the next position and colour. This is
//! ArrayFire's CUDA kernel, step for step, including its truncations to integers (see
//! [`Array::mean_shift`]).
//!
//! One invocation per pixel, tiles of 16 x 16 pixels. The pixels around the tile (as far as
//! `min(r, MAX_HALO)`, as far as workgroup memory allows) are staged in workgroup memory once;
//! the first step of every walk reads only from there. A walk that leaves the staged region
//! (later steps move the window) reads global memory, which the cache serves, so any number of
//! steps stays exact.

use std::fmt::Write as _;
use std::panic::Location;

use super::{geometry, plane_grid};
use crate::array::Array;
use crate::builtin::Builtin;
use crate::context::RawArg;
use crate::ops::Outcome;
use crate::vision::{invalid, Kernel as Any};

const TILE: u32 = 16;
/// Words of workgroup memory the staged window may take.
const TILE_WORDS: u32 = 4096;

/// One kernel: the channel count and the staged halo are literals.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) struct Kernel {
    channels: u8,
    halo: u8,
}

/// The staged halo of a window radius `r` for `channels` channels.
fn halo_for(radius: u32, channels: u32) -> u32 {
    // A colour pixel takes a `vec4`.
    let words = if channels == 3 { 4 } else { 1 };
    let mut halo = radius.min(32);
    while halo > 0 && words * (TILE + 2 * halo) * (TILE + 2 * halo) > TILE_WORDS {
        halo -= 1;
    }
    halo
}

/// The WGSL: arguments `src`, `dst`, uniform `{h, w, tiles_y, iters, radius, cvar, _, _}`.
///
/// A colour tile holds one `vec4` per pixel (the three channels and a pad), so that a tap is a
/// single 16-byte load; the window of a step is cut at the image once and, if it lies inside the
/// staged region, walked in workgroup memory without any further test.
pub(crate) fn source(k: Kernel) -> String {
    let ch = u32::from(k.channels);
    let color = ch == 3;
    let halo = u32::from(k.halo);
    let tw = TILE + 2 * halo;
    let sq = tw * tw;
    let mut s = String::new();
    s.push_str("struct Params { h: u32, w: u32, tiles_y: u32, iters: u32, radius: i32, cvar: f32, a: u32, b: u32 }\n");
    s.push_str("@group(1) @binding(0) var<storage, read> src: array<f32>;\n");
    s.push_str("@group(1) @binding(1) var<storage, read_write> dst: array<f32>;\n");
    s.push_str("@group(1) @binding(2) var<uniform> p: Params;\n");
    let (vt, zero) = if color { ("vec3<f32>", "vec3<f32>(0.0)") } else { ("f32", "0.0") };
    let tile_ty = if color { "vec4<f32>" } else { "f32" };
    let _ = writeln!(s, "var<workgroup> tile: array<{tile_ty}, {sq}>;");
    // The pixel at (ti, tj) from global memory.
    let load = |ti: &str, tj: &str| -> String {
        if color {
            format!("vec3<f32>(src[base + u32({ti}) + p.h * u32({tj})], src[base + p.h * p.w + u32({ti}) + p.h * u32({tj})], src[base + 2u * p.h * p.w + u32({ti}) + p.h * u32({tj})])")
        } else {
            format!("src[base + u32({ti}) + p.h * u32({tj})]")
        }
    };
    let norm = if color { "fma(d.z, d.z, fma(d.y, d.y, d.x * d.x))" } else { "d * d" };
    // The walk over the window (rows `tj0..=tj1`, columns `ti0..=ti1`), reading pixels with `fetch`.
    let walk = |fetch: &dyn Fn(&str, &str) -> String| -> String {
        format!(
            "            for (var tj = tj0; tj <= tj1; tj++) {{
                var hit = 0;
                for (var ti = ti0; ti <= ti1; ti++) {{
                    let v = {fetch};
                    let d = cc - v;
                    let norm = {norm};
                    if (norm <= p.cvar) {{
                        m += v;
                        sx += ti;
                        hit++;
                    }}
                }}
                count += u32(hit);
                sy += tj * hit;
            }}
",
            fetch = fetch("ti", "tj")
        )
    };
    let staged_fetch = |ti: &str, tj: &str| {
        if color {
            format!("tile[u32(({tj} - o1) * {tw} + {ti} - o0)].xyz")
        } else {
            format!("tile[u32(({tj} - o1) * {tw} + {ti} - o0)]")
        }
    };
    let global_fetch = |ti: &str, tj: &str| load(ti, tj);
    let _ = write!(
        s,
        "@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) t: u32, @builtin(workgroup_id) wid: vec3<u32>) {{
    let ty = wid.y % p.tiles_y;
    let plane = wid.y / p.tiles_y;
    let o0 = i32(wid.x) * {TILE} - {halo};
    let o1 = i32(ty) * {TILE} - {halo};
    let h = i32(p.h);
    let w = i32(p.w);
    let base = plane * {ch}u * p.h * p.w;
    for (var i = t; i < {sq}u; i += 256u) {{
        let a0 = o0 + i32(i % {tw}u);
        let a1 = o1 + i32(i / {tw}u);
        var v = {tile_zero};
        if (a0 >= 0 && a0 < h && a1 >= 0 && a1 < w) {{ v = {tile_load}; }}
        tile[i] = v;
    }}
    workgroupBarrier();
    let gx = i32(wid.x) * {TILE} + i32(t & 15u);
    let gy = i32(ty) * {TILE} + i32(t >> 4u);
    if (gx >= h || gy >= w) {{ return; }}
    var pi = gx;
    var pj = gy;
    var cc = {center};
    for (var it = 0u; it < p.iters; it++) {{
        let oi = pi;
        let oj = pj;
        var count = 0u;
        var sx = 0;
        var sy = 0;
        var m = {zero};
        let ti0 = max(pi - p.radius, 0);
        let ti1 = min(pi + p.radius, h - 1);
        let tj0 = max(pj - p.radius, 0);
        let tj1 = min(pj + p.radius, w - 1);
        if (ti0 >= o0 && ti1 < o0 + {tw} && tj0 >= o1 && tj1 < o1 + {tw}) {{
{staged}        }} else {{
{global}        }}
        if (count == 0u) {{ break; }}
        // 1 / count correctly rounded (a GPU's `/` may be a reciprocal approximation within 2 ulp, and
        // the truncation below is sensitive to the last bit): one Newton step with the exact residual.
        let r0 = 1.0 / f32(count);
        let fc = fma(fma(-f32(count), r0, 1.0), r0, r0);
        pi = i32(f32(sx) * fc);
        pj = i32(f32(sy) * fc);
        m = {truncate};
        let d = cc - m;
        let dist = {dist};
        let stop = (pj == oj && pi == oi) || (f32(abs(oj - pj) + abs(oi - pi)) + dist <= 1.0);
        cc = m;
        if (stop) {{ break; }}
    }}
{store}}}
",
        tile_zero = if color { "vec4<f32>(0.0)" } else { "0.0" },
        tile_load = if color { "vec4<f32>(".to_string() + &load("a0", "a1") + ", 0.0)" } else { load("a0", "a1") },
        center = load("gx", "gy"),
        staged = walk(&staged_fetch),
        global = walk(&global_fetch),
        truncate = if color { "vec3<f32>(vec3<i32>(m * fc))" } else { "f32(i32(m * fc))" },
        dist = if color { "fma(d.z, d.z, fma(d.y, d.y, d.x * d.x))" } else { "d * d" },
        store = if color {
            "    dst[base + u32(gx) + p.h * u32(gy)] = cc.x;\n    dst[base + p.h * p.w + u32(gx) + p.h * u32(gy)] = cc.y;\n    dst[base + 2u * p.h * p.w + u32(gx) + p.h * u32(gy)] = cc.z;\n"
        } else {
            "    dst[base + u32(gx) + p.h * u32(gy)] = cc;\n"
        }
    );
    let _ = vt;
    s
}

impl Array<f32> {
    /// Mean shift filtering of `[H, W]` images (ArrayFire's `meanShift`; axes 2 and 3 are a
    /// batch), or of `[H, W, 3]` colour images (`is_color`, axis 3 is a batch).
    ///
    /// Every pixel starts a walk. In each of the `iterations` steps the window of `2 r + 1`
    /// pixels a side (`r = max(floor(1.5 * spatial_sigma), 1)`, cut at the image) around the
    /// current position is searched for the pixels whose squared colour distance (summed over the
    /// channels) from the current colour is at most `chromatic_sigma^2`. The new position is the
    /// mean of their positions and the new colour the mean of their colours, both **truncated
    /// to integers** (ArrayFire's `__float2int_rz`). The walk stops early when the position did
    /// not move, or moved by at most one pixel in total with a colour change of at most 1. The
    /// result is the colour the walk ends with.
    ///
    /// Because of the truncation the result holds whole numbers: the filter is meant for images in
    /// the range `0..255`; an image in `0..1` ends up all zero, exactly as in ArrayFire.
    ///
    /// A colour image that is not `[H, W, 3]`, `iterations == 0` and negative or non-finite
    /// sigmas poison the result (ArrayFire returns the same errors).
    ///
    /// Differences from ArrayFire: none intended (both of its backends agree). The reciprocal of the pixel count is
    /// computed correctly rounded (the GPU's own division is an approximation), so the truncations
    /// agree with ArrayFire's.
    ///
    /// Lazy: the filter is recorded when the result is evaluated (one launch).
    ///
    /// ```
    /// use luma::Array;
    /// # fn main() -> Result<(), luma::LumaError> {
    /// // Two flat regions with a spike: the spike (far in colour) does not pull the regions together.
    /// let mut data = vec![10.0f32; 8 * 8];
    /// for j in 4..8 { for i in 0..8 { data[i + 8 * j] = 200.0; } }
    /// data[2 + 8 * 2] = 60.0;
    /// let img = Array::from_slice(&data, [8, 8])?;
    /// let out = img.mean_shift(1.0, 20.0, 2, false).to_vec()?;
    /// assert_eq!(out[0], 10.0);
    /// assert_eq!(out[7 + 8 * 7], 200.0);
    /// # Ok(()) }
    /// ```
    #[track_caller]
    pub fn mean_shift(&self, spatial_sigma: f32, chromatic_sigma: f32, iterations: u32, is_color: bool) -> Array<f32> {
        let location = Location::caller();
        Array::resolve(self.context(), self.shape(), run(self, spatial_sigma, chromatic_sigma, iterations, is_color), "mean_shift", location)
    }
}

fn run(input: &Array<f32>, spatial: f32, chromatic: f32, iterations: u32, color: bool) -> Outcome<Array<f32>> {
    if !(spatial.is_finite() && spatial >= 0.0 && chromatic.is_finite() && chromatic >= 0.0) {
        return Err(invalid(format!("mean_shift: the sigmas must be finite and not negative, got {spatial} and {chromatic}")));
    }
    if iterations == 0 {
        return Err(invalid("mean_shift: iterations must be at least 1"));
    }
    let dims = input.shape().dims();
    if color && dims[2] != 3 {
        return Err(invalid(format!("mean_shift: a colour image has 3 channels along axis 2, got shape {dims:?}")));
    }
    let ctx = input.context();
    input.storage()?;
    let (h, w, planes) = geometry(input.shape())?;
    // A colour image has its 3 channels along axis 2 and batches over axis 3.
    let (channels, planes) = if color { (3, dims[3] as u32) } else { (1, planes) };
    let radius = ((spatial * 1.5) as u32).max(1);
    let halo = halo_for(radius, channels);
    let (tx, ty) = (h.div_ceil(TILE), w.div_ceil(TILE));
    let grid = plane_grid(ctx, tx, ty, planes)?;
    let out = Array::<f32>::alloc(ctx, input.shape())?;
    let params = [h, w, ty, iterations, radius, (chromatic * chromatic).to_bits(), 0, 0];
    let (src, src_bytes) = input.ready()?;
    let (dst, dst_bytes) = out.ready()?;
    let kernel = ctx.builtin(Builtin::Vision(Any::Edges(super::Kernel::MeanShift(Kernel { channels: channels as u8, halo: halo as u8 }))), "f32")?;
    let args = [RawArg::read(src.buffer(), src_bytes), RawArg::write(dst.buffer(), dst_bytes), RawArg::Uniform(bytemuck::cast_slice(&params))];
    ctx.launch(&kernel, grid, [h, w, planes], &args)?;
    Ok(out)
}
