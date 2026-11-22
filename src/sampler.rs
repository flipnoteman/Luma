//! The sampling kernel behind [`Array::approx1`], [`Array::approx2`] and the image transforms of
//! [`crate::image`]: one WGSL kernel (run through [`crate::Kernel`], so on every backend) in
//! which an invocation computes one output element. It finds the element's coordinates in the
//! input grid (read from position arrays, scaled from the pixel index, or mapped by a matrix),
//! gathers the 1, 2 or 4 taps per axis of the [`Interp`] method and combines them, in registers.
//!
//! A first version composed this from lazy elementwise operations and gathers (`Array::take`).
//! The fusion planner materialized the weights and indices between a dozen kernels (9 launches
//! for a bilinear warp, 33 for a bicubic one), which made the warps 5 to 10 times slower than one
//! kernel that does all of it.

use std::fmt::Write as _;

use crate::array::Array;
use crate::context::Context;
use crate::error::LumaError;
use crate::interp::Interp;
use crate::kernel::{uniform, KernelArg};
use crate::ops::fold_workgroups;
use crate::shape::Shape;

const WORKGROUP: u32 = 256;

/// Where an output element's coordinates come from.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Source {
    /// Position arrays (`approx1`, `approx2`): `(pos - start) / step`; off-grid outside
    /// `0..=n-1`.
    Positions,
    /// The output pixel index divided by a scale (`resize`); no off-grid.
    Scaled,
    /// An affine or projective map of the output pixel index (`transform`); zero where the
    /// source is outside the image.
    Warp,
}

/// The uniform of the kernel; every member is a scalar, so the layout has no padding.
#[derive(crate::encase::ShaderType)]
struct Params {
    kind: u32,
    method: u32,
    ndims: u32,
    count: u32,
    od0: u32,
    od1: u32,
    od2: u32,
    od3: u32,
    ib0: u32,
    ib1: u32,
    ib2: u32,
    ib3: u32,
    ps0: u32,
    ps1: u32,
    ps2: u32,
    ps3: u32,
    gn0: u32,
    gn1: u32,
    gs0: u32,
    gs1: u32,
    start0: f32,
    start1: f32,
    step0: f32,
    step1: f32,
    scale0: f32,
    scale1: f32,
    off_grid: f32,
    persp: u32,
    m0: f32,
    m1: f32,
    m2: f32,
    m3: f32,
    m4: f32,
    m5: f32,
    m6: f32,
    m7: f32,
    m8: f32,
}

/// Elements per invocation (1, 2 or 4).
fn coarsen() -> usize {
    static C: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *C.get_or_init(|| std::env::var("LUMA_SAMPLER_COARSEN").ok().and_then(|v| v.parse().ok()).filter(|c| [1, 2, 4].contains(c)).unwrap_or(1))
}

/// One launch of the sampling kernel.
pub(crate) struct Sampling<'a> {
    pub src: &'a Array<f32>,
    /// Position arrays (the same array twice where the source has none or one is unused).
    pub pos: [&'a Array<f32>; 2],
    pub out: Shape,
    pub source: Source,
    pub method: Interp,
    /// The grid dimensions (one or two): their extents and strides in the input, and for
    /// `Positions` the `start` and `step` of the grid along them.
    pub grid: &'a [Grid],
    /// For `Positions`: the strides of the position arrays (0 where they have extent 1).
    pub pos_strides: [u32; 4],
    pub off_grid: f32,
    /// For `Scaled`: the output / input extent ratio of the two axes.
    pub scale: [f32; 2],
    /// For `Warp`: the row-major 3 x 3 map from output to input pixels and whether it divides.
    pub matrix: [f32; 9],
    pub perspective: bool,
}

/// A grid dimension of the input.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Grid {
    pub dim: usize,
    pub start: f32,
    pub step: f32,
}

fn strides(dims: [usize; 4]) -> [usize; 4] {
    [1, dims[0], dims[0] * dims[1], dims[0] * dims[1] * dims[2]]
}

/// The strides of an array along its dimensions for reads that repeat extent-1 dimensions.
pub(crate) fn broadcast_strides(dims: [usize; 4]) -> [u32; 4] {
    let s = strides(dims);
    std::array::from_fn(|k| if dims[k] > 1 { s[k] as u32 } else { 0 })
}

impl Sampling<'_> {
    /// Runs the kernel and returns the output array.
    pub(crate) fn run(&self, ctx: &Context) -> Result<Array<f32>, LumaError> {
        let in_dims = self.src.shape().dims();
        let to_u32 = |v: usize| u32::try_from(v).map_err(|_| LumaError::DimensionTooLarge { shape: self.out.dims() });
        let count = self.out.len();
        let od = self.out.dims();
        let in_s = strides(in_dims);
        // Batch strides: the input's strides along the dimensions that are not grid dimensions.
        let mut ib = broadcast_strides(in_dims);
        for g in self.grid {
            ib[g.dim] = 0;
        }
        let gd = |k: usize| self.grid.get(k).copied().unwrap_or(Grid { dim: 0, start: 0.0, step: 1.0 });
        let (g0, g1) = (gd(0), gd(1));
        let m = self.matrix;
        let params = Params {
            kind: self.source as u32,
            method: self.method as u32,
            ndims: self.grid.len() as u32,
            count: to_u32(count)?,
            od0: to_u32(od[0])?,
            od1: to_u32(od[1])?,
            od2: to_u32(od[2])?,
            od3: to_u32(od[3])?,
            ib0: ib[0],
            ib1: ib[1],
            ib2: ib[2],
            ib3: ib[3],
            ps0: self.pos_strides[0],
            ps1: self.pos_strides[1],
            ps2: self.pos_strides[2],
            ps3: self.pos_strides[3],
            gn0: to_u32(in_dims[g0.dim])?,
            gn1: to_u32(in_dims[g1.dim])?,
            gs0: to_u32(in_s[g0.dim])?,
            gs1: to_u32(in_s[g1.dim])?,
            start0: g0.start,
            start1: g1.start,
            step0: g0.step,
            step1: g1.step,
            scale0: self.scale[0],
            scale1: self.scale[1],
            off_grid: self.off_grid,
            persp: u32::from(self.perspective),
            m0: m[0],
            m1: m[1],
            m2: m[2],
            m3: m[3],
            m4: m[4],
            m5: m[5],
            m6: m[6],
            m7: m[7],
            m8: m[8],
        };
        // A grid shaped like the output saves the index arithmetic of the flat one where it fits.
        let max = ctx.max_workgroups_per_dimension() as usize;
        let (rows, batch) = (od[1], od[2] * od[3]);
        let c = coarsen();
        let per_group = WORKGROUP as usize * c;
        let shaped = od[0].div_ceil(per_group) <= max && rows <= max && batch <= max;
        let (kernel, grid) = if shaped {
            (ctx.cached_kernel("sampler", c, "main2d", || source(c))?, [to_u32(od[0].div_ceil(per_group))?, to_u32(rows)?, to_u32(batch)?])
        } else {
            (ctx.cached_kernel("sampler", c, "main", || source(c))?, fold_workgroups(ctx, to_u32(count.div_ceil(per_group))?)?)
        };
        let mut out = Array::<f32>::alloc(ctx, self.out)?;
        kernel.launch_raw(grid, [KernelArg::from(self.src), KernelArg::from(self.pos[0]), KernelArg::from(self.pos[1]), KernelArg::from(&mut out), uniform(&params)])?;
        Ok(out)
    }
}

/// The weights of the four taps `y0..y3` of the cubic methods at fraction `t`: `w0..w3`.
fn cubic_weights(w: &mut String, indent: &str, t: &str) {
    let _ = writeln!(
        w,
        "{indent}let {t}2 = {t} * {t};
{indent}let {t}3 = {t}2 * {t};
{indent}var {t}w0 = 0.0; var {t}w1 = 0.0; var {t}w2 = 0.0; var {t}w3 = 0.0;
{indent}if (p.method == 5u) {{
{indent}    {t}w0 = ({t}2 * 2.0 - {t}3 - {t}) * 0.5;
{indent}    {t}w1 = ({t}3 * 3.0 - {t}2 * 5.0 + 2.0) * 0.5;
{indent}    {t}w2 = ({t}2 * 4.0 - {t}3 * 3.0 + {t}) * 0.5;
{indent}    {t}w3 = ({t}3 - {t}2) * 0.5;
{indent}}} else {{
{indent}    {t}w0 = {t}2 * 2.0 - {t}3 - {t};
{indent}    {t}w1 = {t}3 - {t}2 * 2.0 + 1.0;
{indent}    {t}w2 = {t}2 + {t} - {t}3;
{indent}    {t}w3 = {t}3 - {t}2;
{indent}}}"
    );
}

/// The linear fraction of the two-tap methods: `t`, cosine smoothed for `Cosine`.
fn linear_fraction(w: &mut String, indent: &str, t: &str) {
    let _ = writeln!(
        w,
        "{indent}var {t}u = {t};
{indent}if (p.method == 3u) {{ {t}u = (1.0 - cos({t} * 3.1415927)) * 0.5; }}"
    );
}

/// The source of the sampling kernel.
fn source(coarsen: usize) -> String {
    let mut w = String::new();
    let _ = writeln!(w, "const C: u32 = {coarsen}u;");
    w.push_str(
        r#"
struct P {
    kind: u32, method: u32, ndims: u32, count: u32,
    od0: u32, od1: u32, od2: u32, od3: u32,
    ib0: u32, ib1: u32, ib2: u32, ib3: u32,
    ps0: u32, ps1: u32, ps2: u32, ps3: u32,
    gn0: u32, gn1: u32, gs0: u32, gs1: u32,
    start0: f32, start1: f32, step0: f32, step1: f32,
    scale0: f32, scale1: f32, off_grid: f32, persp: u32,
    m0: f32, m1: f32, m2: f32, m3: f32, m4: f32, m5: f32, m6: f32, m7: f32, m8: f32,
}
@group(1) @binding(0) var<storage, read> src: array<f32>;
@group(1) @binding(1) var<storage, read> pos0: array<f32>;
@group(1) @binding(2) var<storage, read> pos1: array<f32>;
@group(1) @binding(3) var<storage, read_write> out: array<f32>;
@group(1) @binding(4) var<uniform> p: P;

fn finite(x: f32) -> bool { return (bitcast<u32>(x) & 0x7f800000u) != 0x7f800000u; }
// The clamped grid index of tap `f + k`.
fn tap(f: f32, k: f32, hi: f32) -> u32 { return u32(clamp(f + k, 0.0, hi)); }

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) lid: u32) {
    // Element `i + k * stride` for k < C: the lanes of a warp stay on neighbouring elements and
    // the loads of the C elements are all issued before the first store.
    let stride = nwg.x * nwg.y * 256u;
    let first = (wid.x + wid.y * nwg.x) * 256u + lid;
    var v: array<f32, C>;
    for (var k = 0u; k < C; k += 1u) {
        let i = first + k * stride;
        if (i < p.count) {
            let c0 = i % p.od0;
            var r = i / p.od0;
            let c1 = r % p.od1;
            r = r / p.od1;
            let c2 = r % p.od2;
            let c3 = r / p.od2;
            v[k] = element(c0, c1, c2, c3);
        }
    }
    for (var k = 0u; k < C; k += 1u) {
        let i = first + k * stride;
        if (i < p.count) { out[i] = v[k]; }
    }
}

// The same with the grid shaped like the output: x along dimension 0, y along 1, z the batch.
@compute @workgroup_size(256)
fn main2d(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let c2 = gid.z % p.od2;
    let c3 = gid.z / p.od2;
    let stride = nwg.x * 256u;
    let row = p.od0 * (gid.y + p.od1 * gid.z);
    var v: array<f32, C>;
    for (var k = 0u; k < C; k += 1u) {
        let x = gid.x + k * stride;
        if (x < p.od0) { v[k] = element(x, gid.y, c2, c3); }
    }
    for (var k = 0u; k < C; k += 1u) {
        let x = gid.x + k * stride;
        if (x < p.od0) { out[row + x] = v[k]; }
    }
}

fn element(c0: u32, c1: u32, c2: u32, c3: u32) -> f32 {
    let base = c0 * p.ib0 + c1 * p.ib1 + c2 * p.ib2 + c3 * p.ib3;
    var x = 0.0;
    var y = 0.0;
    var ok = true;
    let two = p.ndims == 2u;
    let hx = f32(p.gn0) - 1.0;
    let hy = f32(p.gn1) - 1.0;
    if (p.kind == 0u) {
        let at = c0 * p.ps0 + c1 * p.ps1 + c2 * p.ps2 + c3 * p.ps3;
        let a = pos0[at];
        x = (a - p.start0) / p.step0;
        ok = finite(a) && x >= 0.0 && x <= hx;
        if (two) {
            let b = pos1[at];
            y = (b - p.start1) / p.step1;
            ok = ok && finite(b) && y >= 0.0 && y <= hy;
        }
    } else if (p.kind == 1u) {
        x = f32(c0) / p.scale0;
        y = f32(c1) / p.scale1;
    } else {
        let px = f32(c0);
        let py = f32(c1);
        x = (px * p.m0 + py * p.m1) + p.m2;
        y = (px * p.m3 + py * p.m4) + p.m5;
        if (p.persp != 0u) {
            let w = (px * p.m6 + py * p.m7) + p.m8;
            x = x / w;
            y = y / w;
        }
        let reach = select(0.0, 0.5, p.method == 0u);
        ok = finite(x) && finite(y) && x >= 0.0 && y >= 0.0 && x + reach < f32(p.gn0) && y + reach < f32(p.gn1);
    }
    var res = p.off_grid;
    if (ok) {
        let fx = floor(x);
        let tx = x - fx;
        let fy = floor(y);
        let ty = y - fy;
        var v = 0.0;
        if (!two) {
"#,
    );
    // One axis: 1, 2 or 4 taps along the grid dimension.
    w.push_str("            if (p.method <= 1u) {\n");
    w.push_str("                var f = fx;\n                if (p.method == 0u && tx >= 0.5) { f = fx + 1.0; }\n");
    w.push_str("                v = src[base + tap(f, 0.0, hx) * p.gs0];\n");
    w.push_str("            } else if (p.method <= 3u) {\n");
    linear_fraction(&mut w, "                ", "tx");
    w.push_str("                let i0 = tap(fx, 0.0, hx) * p.gs0;\n                var i1 = i0 + p.gs0;\n                if (fx + 1.0 > hx) { i1 = tap(fx, 1.0, hx) * p.gs0; }\n");
    w.push_str("                v = (1.0 - txu) * src[base + i0] + txu * src[base + i1];\n");
    w.push_str("            } else {\n");
    cubic_weights(&mut w, "                ", "tx");
    w.push_str("                var i0 = 0u; var i1 = 0u; var i2 = 0u; var i3 = 0u;\n");
    w.push_str("                if (fx >= 1.0 && fx + 2.0 <= hx) {\n                    i1 = u32(fx) * p.gs0;\n                    i0 = i1 - p.gs0;\n                    i2 = i1 + p.gs0;\n                    i3 = i2 + p.gs0;\n                } else {\n");
    w.push_str("                    i0 = tap(fx, -1.0, hx) * p.gs0;\n                    i1 = tap(fx, 0.0, hx) * p.gs0;\n                    i2 = tap(fx, 1.0, hx) * p.gs0;\n                    i3 = tap(fx, 2.0, hx) * p.gs0;\n                }\n");
    w.push_str("                v = txw0 * src[base + i0] + txw1 * src[base + i1] + txw2 * src[base + i2] + txw3 * src[base + i3];\n");
    w.push_str("            }\n        } else {\n");
    // Two axes.
    w.push_str("            if (p.method <= 1u) {\n");
    w.push_str("                var f = fx;\n                var g = fy;\n");
    w.push_str("                if (p.method == 0u && tx >= 0.5) { f = fx + 1.0; }\n                if (p.method == 0u && ty >= 0.5) { g = fy + 1.0; }\n");
    w.push_str("                v = src[base + tap(f, 0.0, hx) * p.gs0 + tap(g, 0.0, hy) * p.gs1];\n");
    w.push_str("            } else if (p.method <= 3u) {\n");
    linear_fraction(&mut w, "                ", "tx");
    linear_fraction(&mut w, "                ", "ty");
    w.push_str("                let x0 = tap(fx, 0.0, hx) * p.gs0;\n                var x1 = x0 + p.gs0;\n                if (fx + 1.0 > hx) { x1 = tap(fx, 1.0, hx) * p.gs0; }\n");
    w.push_str("                let r0 = base + tap(fy, 0.0, hy) * p.gs1;\n                var r1 = r0 + p.gs1;\n                if (fy + 1.0 > hy) { r1 = base + tap(fy, 1.0, hy) * p.gs1; }\n");
    w.push_str("                let a0 = (1.0 - txu) * src[r0 + x0] + txu * src[r0 + x1];\n                let a1 = (1.0 - txu) * src[r1 + x0] + txu * src[r1 + x1];\n");
    w.push_str("                v = (1.0 - tyu) * a0 + tyu * a1;\n");
    w.push_str("            } else {\n");
    cubic_weights(&mut w, "                ", "tx");
    cubic_weights(&mut w, "                ", "ty");
    w.push_str("                var x0 = 0u; var x1 = 0u; var x2 = 0u; var x3 = 0u;\n");
    w.push_str("                if (fx >= 1.0 && fx + 2.0 <= hx) {\n                    x1 = u32(fx) * p.gs0;\n                    x0 = x1 - p.gs0;\n                    x2 = x1 + p.gs0;\n                    x3 = x2 + p.gs0;\n                } else {\n");
    for (k, off) in ["-1.0", "0.0", "1.0", "2.0"].iter().enumerate() {
        let _ = writeln!(w, "                    x{k} = tap(fx, {off}, hx) * p.gs0;");
    }
    w.push_str("                }\n");
    w.push_str("                var r0 = 0u; var r1 = 0u; var r2 = 0u; var r3 = 0u;\n");
    w.push_str("                if (fy >= 1.0 && fy + 2.0 <= hy) {\n                    r1 = base + u32(fy) * p.gs1;\n                    r0 = r1 - p.gs1;\n                    r2 = r1 + p.gs1;\n                    r3 = r2 + p.gs1;\n                } else {\n");
    for (k, off) in ["-1.0", "0.0", "1.0", "2.0"].iter().enumerate() {
        let _ = writeln!(w, "                    r{k} = base + tap(fy, {off}, hy) * p.gs1;");
    }
    w.push_str("                }\n");
    for j in 0..4 {
        let _ = writeln!(
            w,
            "                let a{j} = txw0 * src[r{j} + x0] + txw1 * src[r{j} + x1] + txw2 * src[r{j} + x2] + txw3 * src[r{j} + x3];"
        );
    }
    w.push_str("                v = tyw0 * a0 + tyw1 * a1 + tyw2 * a2 + tyw3 * a3;\n");
    w.push_str("            }\n        }\n        res = v;\n    }\n    return res;\n}\n");
    w
}
