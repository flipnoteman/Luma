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
const TILE_WORDS: u32 = 3072;

/// One kernel: the channel count and the staged halo are literals.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) struct Kernel {
    channels: u8,
    halo: u8,
}

/// The staged halo of a window radius `r` for `channels` channels.
fn halo_for(radius: u32, channels: u32) -> u32 {
    let mut halo = radius.min(32);
    while halo > 0 && channels * (TILE + 2 * halo) * (TILE + 2 * halo) > TILE_WORDS {
        halo -= 1;
    }
    halo
}

/// The WGSL: arguments `src`, `dst`, uniform `{h, w, tiles_y, iters, radius, cvar, _, _}`.
pub(crate) fn source(k: Kernel) -> String {
    let ch = u32::from(k.channels);
    let halo = u32::from(k.halo);
    let tw = TILE + 2 * halo;
    let words = ch * tw * tw;
    let mut s = String::new();
    s.push_str("struct Params { h: u32, w: u32, tiles_y: u32, iters: u32, radius: i32, cvar: f32, a: u32, b: u32 }\n");
    s.push_str("@group(1) @binding(0) var<storage, read> src: array<f32>;\n");
    s.push_str("@group(1) @binding(1) var<storage, read_write> dst: array<f32>;\n");
    s.push_str("@group(1) @binding(2) var<uniform> p: Params;\n");
    let _ = writeln!(s, "var<workgroup> tile: array<f32, {words}>;");
    let _ = write!(
        s,
        "fn px(ch: u32, ti: i32, tj: i32, o0: i32, o1: i32, base: u32) -> f32 {{
    let lx = ti - o0;
    let ly = tj - o1;
    if (lx >= 0 && lx < {tw} && ly >= 0 && ly < {tw}) {{ return tile[ch * {sq}u + u32(ly) * {tw}u + u32(lx)]; }}
    return src[base + ch * p.h * p.w + u32(ti) + p.h * u32(tj)];
}}
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) t: u32, @builtin(workgroup_id) wid: vec3<u32>) {{
    let ty = wid.y % p.tiles_y;
    let plane = wid.y / p.tiles_y;
    let o0 = i32(wid.x) * {TILE} - {halo};
    let o1 = i32(ty) * {TILE} - {halo};
    let h = i32(p.h);
    let w = i32(p.w);
    let base = plane * {ch}u * p.h * p.w;
    for (var i = t; i < {words}u; i += 256u) {{
        let c = i / {sq}u;
        let r = i % {sq}u;
        let a0 = o0 + i32(r % {tw}u);
        let a1 = o1 + i32(r / {tw}u);
        var v = 0.0;
        if (a0 >= 0 && a0 < h && a1 >= 0 && a1 < w) {{ v = src[base + c * p.h * p.w + u32(a0) + p.h * u32(a1)]; }}
        tile[i] = v;
    }}
    workgroupBarrier();
    let gx = i32(wid.x) * {TILE} + i32(t & 15u);
    let gy = i32(ty) * {TILE} + i32(t >> 4u);
    if (gx >= h || gy >= w) {{ return; }}
    var pi = gx;
    var pj = gy;
",
        sq = tw * tw
    );
    for c in 0..ch {
        let _ = writeln!(s, "    var cc{c} = px({c}u, gx, gy, o0, o1, base);");
    }
    s.push_str("    for (var it = 0u; it < p.iters; it++) {\n        let oi = pi;\n        let oj = pj;\n        var count = 0u;\n        var sx = 0;\n        var sy = 0;\n");
    for c in 0..ch {
        let _ = writeln!(s, "        var m{c} = 0.0;");
    }
    s.push_str(
        "        for (var wj = -p.radius; wj <= p.radius; wj++) {
            let tj = pj + wj;
            if (tj < 0 || tj >= w) { continue; }
            var hit = 0;
            for (var wi = -p.radius; wi <= p.radius; wi++) {
                let ti = pi + wi;
                if (ti < 0 || ti >= h) { continue; }
                var norm = 0.0;
",
    );
    for c in 0..ch {
        let _ = writeln!(s, "                let v{c} = px({c}u, ti, tj, o0, o1, base);\n                norm += (cc{c} - v{c}) * (cc{c} - v{c});");
    }
    s.push_str("                if (norm <= p.cvar) {\n");
    for c in 0..ch {
        let _ = writeln!(s, "                    m{c} += v{c};");
    }
    s.push_str(
        "                    sx += ti;
                    hit++;
                }
            }
            count += u32(hit);
            sy += tj * hit;
        }
        if (count == 0u) { break; }
        let fc = 1.0 / f32(count);
        pi = i32(f32(sx) * fc);
        pj = i32(f32(sy) * fc);
        var dist = 0.0;
",
    );
    for c in 0..ch {
        let _ = writeln!(s, "        m{c} = f32(i32(m{c} * fc));\n        dist += (cc{c} - m{c}) * (cc{c} - m{c});");
    }
    s.push_str("        let stop = (pj == oj && pi == oi) || (f32(abs(oj - pj) + abs(oi - pi)) + dist <= 1.0);\n");
    for c in 0..ch {
        let _ = writeln!(s, "        cc{c} = m{c};");
    }
    s.push_str("        if (stop) { break; }\n    }\n");
    for c in 0..ch {
        let _ = writeln!(s, "    dst[base + {c}u * p.h * p.w + u32(gx) + p.h * u32(gy)] = cc{c};");
    }
    s.push_str("}\n");
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
    /// Differences from ArrayFire: none intended (the CUDA kernel is followed; ArrayFire's CPU
    /// backend differs in the stopping rule). Division by the pixel count is the GPU's single
    /// precision division, so a mean that is an integer up to rounding may truncate one lower
    /// than ArrayFire's.
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
