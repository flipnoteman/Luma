//! The bilinear resize of the image pyramids of ORB and SIFT.
//!
//! ArrayFire's convention (not the half-pixel one): output cell `i` samples the input at
//! `i * in / out` along each axis, the neighbours past the last cell are the last cell.

use super::Kernel as Descriptors;
use crate::array::Array;
use crate::builtin::Builtin;
use crate::context::RawArg;
use crate::ops::Outcome;
use crate::vision::{tile_grid, to_u32, Kernel as Any};

/// The WGSL: arguments source, destination, uniform `{ih, iw, oh, ow, sy, sx, 0, 0}`.
pub(crate) fn source() -> String {
    "struct Params { ih: u32, iw: u32, oh: u32, ow: u32, sy: f32, sx: f32, a: u32, b: u32 }
@group(1) @binding(0) var<storage, read> src: array<f32>;
@group(1) @binding(1) var<storage, read_write> dst: array<f32>;
@group(1) @binding(2) var<uniform> p: Params;
@compute @workgroup_size(16, 16)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let ox = wid.x * 16u + lid.x;
    let oy = wid.y * 16u + lid.y;
    if (ox >= p.oh || oy >= p.ow) { return; }
    let xf = f32(ox) * p.sy;
    let yf = f32(oy) * p.sx;
    var ix = i32(floor(xf));
    var iy = i32(floor(yf));
    let ih = i32(p.ih);
    let iw = i32(p.iw);
    if (ix >= ih) { ix = ih - 1; }
    if (iy >= iw) { iy = iw - 1; }
    let b = xf - f32(ix);
    let a = yf - f32(iy);
    let ix2 = select(ix, ix + 1, ix + 1 < ih);
    let iy2 = select(iy, iy + 1, iy + 1 < iw);
    let p1 = src[u32(ix) + p.ih * u32(iy)];
    let p2 = src[u32(ix) + p.ih * u32(iy2)];
    let p3 = src[u32(ix2) + p.ih * u32(iy)];
    let p4 = src[u32(ix2) + p.ih * u32(iy2)];
    let v = ((1.0 - a) * (1.0 - b)) * p1 + (a * (1.0 - b)) * p2 + ((1.0 - a) * b) * p3 + (a * b) * p4;
    dst[ox + p.oh * oy] = v;
}
"
    .to_string()
}

/// Records `dst` (`[oh, ow]`) as the bilinear resize of `src` (`[ih, iw]`).
pub(crate) fn launch(src: &Array<f32>, [ih, iw]: [usize; 2], dst: &Array<f32>, [oh, ow]: [usize; 2]) -> Outcome<()> {
    let ctx = src.context();
    let (src_pin, src_bytes) = src.ready()?;
    let (dst_pin, dst_bytes) = dst.ready()?;
    let params = [
        to_u32(ih, "height")?,
        to_u32(iw, "width")?,
        to_u32(oh, "height")?,
        to_u32(ow, "width")?,
        (ih as f32 / oh as f32).to_bits(),
        (iw as f32 / ow as f32).to_bits(),
        0,
        0,
    ];
    let kernel = ctx.builtin(Builtin::Vision(Any::Descriptors(Descriptors::Resize)), "f32")?;
    let args = [RawArg::read(src_pin.buffer(), src_bytes), RawArg::write(dst_pin.buffer(), dst_bytes), RawArg::Uniform(bytemuck::bytes_of(&params))];
    ctx.launch(&kernel, tile_grid(ctx, params[2].div_ceil(16), params[3].div_ceil(16))?, [params[2], params[3], 1], &args)?;
    Ok(())
}
