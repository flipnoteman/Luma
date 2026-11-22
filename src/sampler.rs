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
use crate::jit::cost::{read_path, ReadPath, ReadPattern, TextureRead};
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
    *C.get_or_init(|| std::env::var("LUMA_SAMPLER_COARSEN").ok().and_then(|v| v.parse().ok()).filter(|c| [1, 2, 4].contains(c)).unwrap_or(2))
}

/// The workgroup of the shaped kernel: `x` along dimension 0 and `y` rows. 32 x 8 (a warp along a row, eight rows) keeps the positions of neighbouring rows in the L1 together: scattered `approx2` on an RTX 3090 4K, bicubic 0.60 -> 0.48 ms, bilinear 0.31 -> 0.27 ms, against 256 x 1; smooth warps 1-8 % faster.
fn tile() -> [u32; 2] {
    static T: std::sync::OnceLock<[u32; 2]> = std::sync::OnceLock::new();
    *T.get_or_init(|| {
        std::env::var("LUMA_SAMPLER_TILE")
            .ok()
            .and_then(|v| v.split_once('x').and_then(|(x, y)| Some([x.parse().ok()?, y.parse().ok()?])))
            .filter(|t: &[u32; 2]| t[0] * t[1] == WORKGROUP)
            .unwrap_or([32, 8])
    })
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
        let mut params = Params {
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
        // A tile of eight rows idles seven of eight invocations of an output of one row (`approx1`:
        // 1M samples 18 -> 44 us on an RTX 3090): such outputs take a row of 256.
        let tile = if od[1] < tile()[1] as usize { [WORKGROUP, 1] } else { tile() };
        let per_group = WORKGROUP as usize * c;
        let per_tile = tile[0] as usize * c;
        let shaped = od[0].div_ceil(per_tile) <= max && rows.div_ceil(tile[1] as usize) <= max && batch <= max;
        let rows_per = self.rows_per();
        let rows_ok = self.source == Source::Scaled && od[0].div_ceil(WORKGROUP as usize) <= max && rows.div_ceil(rows_per) <= max && batch <= max;
        // The source as a texture, where it is one 2-D slice the device can see as one and the
        // read path pays (see `crate::jit::cost::read_path`).
        let read_mode = self.read_mode(ctx);
        let filter = if read_mode == Read::Sample { luma_core::TextureFilter::Linear } else { luma_core::TextureFilter::Nearest };
        let texture = match read_mode {
            Read::Buffer => None,
            _ => ctx.texture_2d(self.src, in_dims[g0.dim], in_dims[g1.dim], filter)?,
        };
        let read_mode = if texture.is_some() { read_mode } else { Read::Buffer };
        // Without a texture the hardware-filtered method is the exact bilinear one.
        let method = if self.method == Interp::LinearHardware { Interp::Linear } else { self.method };
        params.method = method as u32;
        #[cfg(test)]
        if texture.is_some() {
            tests::TEXTURED.set(tests::TEXTURED.get() + 1);
        }
        // The kernel is specialised for the method, the kind of source and the dimensions of the grid:
        // they are constants of its source, so each launch runs only the code it needs.
        let spec = Spec { method: params.method, kind: params.kind, ndims: params.ndims, persp: params.persp };
        let variant = spec.variant(c, read_mode, tile, rows_per);
        let (kernel, grid) = if rows_ok {
            (ctx.cached_kernel("sampler_rows", variant, "main_rows", || source(c, tile, true, read_mode, spec, rows_per))?, [to_u32(od[0].div_ceil(WORKGROUP as usize))?, to_u32(rows.div_ceil(rows_per))?, to_u32(batch)?])
        } else if shaped {
            (ctx.cached_kernel("sampler", variant, "main2d", || source(c, tile, false, read_mode, spec, rows_per))?, [to_u32(od[0].div_ceil(per_tile))?, to_u32(rows.div_ceil(tile[1] as usize))?, to_u32(batch)?])
        } else {
            (ctx.cached_kernel("sampler", variant, "main", || source(c, tile, false, read_mode, spec, rows_per))?, fold_workgroups(ctx, to_u32(count.div_ceil(per_group))?)?)
        };
        let mut out = Array::<f32>::alloc(ctx, self.out)?;
        let mut args: Vec<KernelArg<'_>> = Vec::with_capacity(6);
        match &texture {
            Some(texture) => {
                args.push(KernelArg::texture(texture));
                if matches!(read_mode, Read::Gather | Read::Sample) {
                    args.push(KernelArg::texture(texture));
                }
            }
            None => args.push(KernelArg::from(self.src)),
        }
        if !rows_ok {
            args.push(KernelArg::from(self.pos[0]));
            args.push(KernelArg::from(self.pos[1]));
        }
        args.push(KernelArg::from(&mut out));
        args.push(uniform(&params));
        kernel.launch_raw(grid, args)?;
        Ok(out)
    }
}

impl Sampling<'_> {
    /// Image rows per invocation of the `Scaled` kernel (`main_rows`). An invocation computes the
    /// taps of its column once for all its rows, which pays where the rows of the output share
    /// source rows (an upscale; a bicubic one the most), and costs parallelism where they do not:
    /// a downscale reads fresh rows for every output row, and fewer rows per invocation fill the
    /// device's waves better (RTX 3090 and 3080 Laptop, 4K -> 1080p: bicubic 0.060 -> 0.051 ms and
    /// 0.128 -> 0.116 ms with one row, bilinear 0.0526 -> 0.0509 ms with two; 1080p -> 4K bicubic
    /// 0.119 -> 0.114 ms with sixteen).
    fn rows_per(&self) -> usize {
        let cubic = matches!(self.method, Interp::Cubic | Interp::CubicSpline);
        if self.scale[1] < 1.0 {
            if cubic { 1 } else { 2 }
        } else if cubic {
            16
        } else {
            8
        }
    }

    /// How this launch reads its source: [`Read::Buffer`] unless the grid is one contiguous 2-D
    /// slice and reading it through the texture units pays.
    fn read_mode(&self, ctx: &Context) -> Read {
        let d = self.src.shape().dims();
        let two_d = self.grid.len() == 2 && self.grid[0].dim == 0 && self.grid[1].dim == 1 && d[2] * d[3] == 1;
        if !two_d {
            return Read::Buffer;
        }
        #[cfg(test)]
        if let Some(read) = tests::FORCE.get() {
            return read;
        }
        // The explicit opt-in: the texture unit's filter, wherever there is a texture to filter.
        if self.method == Interp::LinearHardware {
            return Read::Sample;
        }
        let taps = match self.method {
            Interp::Linear | Interp::Cosine => 4,
            Interp::Cubic | Interp::CubicSpline => 16,
            _ => return Read::Buffer,
        };
        let Some(caps) = ctx.hal().texture_caps() else { return Read::Buffer };
        let read = TextureRead {
            pattern: match self.source {
                Source::Positions => ReadPattern::Positions,
                Source::Scaled => ReadPattern::Scaled,
                Source::Warp => ReadPattern::Warp,
            },
            taps,
            outputs: self.out.len() as u64,
            src_bytes: self.src.byte_len(),
            zero_copy: caps.aliases,
        };
        match read_path(ctx, &read) {
            ReadPath::Texture => Read::Gather,
            ReadPath::Plain => Read::Buffer,
        }
    }
}

/// What a launch fixes of the kernel's behaviour: its method, kind of source, grid dimensions and
/// whether the map divides. The generic uniform holds them too, but a kernel built for them runs
/// none of the branches over them (approx2 bilinear 4K on an RTX 3090: a third of the instructions).
#[derive(Clone, Copy)]
struct Spec {
    method: u32,
    kind: u32,
    ndims: u32,
    persp: u32,
}

impl Spec {
    /// A number that tells the kernels of a launch apart (24 bits; the sizes are powers of two).
    fn variant(self, coarsen: usize, read: Read, tile: [u32; 2], rows: usize) -> usize {
        let bits = [
            (coarsen.ilog2(), 2),
            (read as u32, 2),
            (tile[0].ilog2(), 4),
            (tile[1].ilog2(), 4),
            (self.method, 3),
            (self.kind, 2),
            (self.ndims.saturating_sub(1), 1),
            (self.persp, 1),
            (rows as u32, 5),
        ];
        bits.iter().fold((0usize, 0u32), |(word, shift), &(v, width)| (word | (v as usize) << shift, shift + width)).0
    }
}

/// The source of the sampling kernel with `coarsen` elements per invocation in the generic
/// entry points.
fn source(coarsen: usize, tile: [u32; 2], rows_entry: bool, read: Read, spec: Spec, rows_per: usize) -> String {
    let mut w = String::new();
    let _ = writeln!(w, "const C: u32 = {coarsen}u;\nconst ROWS: u32 = {}u;\nconst TX: u32 = {}u;\nconst TY: u32 = {}u;", rows_per, tile[0], tile[1]);
    w.push_str(&bindings(rows_entry, read));
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

fn finite(x: f32) -> bool { return (bitcast<u32>(x) & 0x7f800000u) != 0x7f800000u; }
// The clamped grid index of tap `f + k`.
fn tap(f: f32, k: f32, hi: f32) -> u32 { return u32(clamp(f + k, 0.0, hi)); }

// The taps of one axis: element offsets (grid index times the stride) and weights; the taps a
// method does not use are 0.
struct Axis {
    o0: u32, o1: u32, o2: u32, o3: u32,
    w0: f32, w1: f32, w2: f32, w3: f32,
    f: f32,
}

fn make_axis(x: f32, h: f32, stride: u32) -> Axis {
    var a: Axis;
    let fx = floor(x);
    a.f = fx;
    let t = x - fx;
    if (p.method <= 1u) {
        var f = fx;
        if (p.method == 0u && t >= 0.5) { f = fx + 1.0; }
        a.o0 = tap(f, 0.0, h) * stride;
        a.w0 = 1.0;
    } else if (p.method <= 3u) {
        var u = t;
        if (p.method == 3u) { u = (1.0 - cos(t * 3.1415927)) * 0.5; }
        a.o0 = tap(fx, 0.0, h) * stride;
        if (fx + 1.0 <= h) { a.o1 = a.o0 + stride; } else { a.o1 = tap(fx, 1.0, h) * stride; }
        a.w0 = 1.0 - u;
        a.w1 = u;
    } else {
        if (fx >= 1.0 && fx + 2.0 <= h) {
            a.o1 = u32(fx) * stride;
            a.o0 = a.o1 - stride;
            a.o2 = a.o1 + stride;
            a.o3 = a.o2 + stride;
        } else {
            a.o0 = tap(fx, -1.0, h) * stride;
            a.o1 = tap(fx, 0.0, h) * stride;
            a.o2 = tap(fx, 1.0, h) * stride;
            a.o3 = tap(fx, 2.0, h) * stride;
        }
        let t2 = t * t;
        let t3 = t2 * t;
        if (p.method == 5u) {
            a.w0 = (t2 * 2.0 - t3 - t) * 0.5;
            a.w1 = (t3 * 3.0 - t2 * 5.0 + 2.0) * 0.5;
            a.w2 = (t2 * 4.0 - t3 * 3.0 + t) * 0.5;
            a.w3 = (t3 - t2) * 0.5;
        } else {
            a.w0 = t2 * 2.0 - t3 - t;
            a.w1 = t3 - t2 * 2.0 + 1.0;
            a.w2 = t2 + t - t3;
            a.w3 = t3 - t2;
        }
    }
    return a;
}

// @@ONE
fn combine1(base: u32, a: Axis) -> f32 {
    if (p.method <= 1u) { return src[base + a.o0]; }
    if (p.method <= 3u) { return a.w0 * src[base + a.o0] + a.w1 * src[base + a.o1]; }
    return a.w0 * src[base + a.o0] + a.w1 * src[base + a.o1] + a.w2 * src[base + a.o2] + a.w3 * src[base + a.o3];
}
// @@END
// @@TWO
fn combine2(base: u32, a: Axis, b: Axis) -> f32 {
    if (p.method <= 1u) { return src[base + a.o0 + b.o0]; }
    if (p.method <= 3u) {
        let r0 = base + b.o0;
        let r1 = base + b.o1;
        let s0 = a.w0 * src[r0 + a.o0] + a.w1 * src[r0 + a.o1];
        let s1 = a.w0 * src[r1 + a.o0] + a.w1 * src[r1 + a.o1];
        return b.w0 * s0 + b.w1 * s1;
    }
    let r0 = base + b.o0;
    let r1 = base + b.o1;
    let r2 = base + b.o2;
    let r3 = base + b.o3;
"#,
    );
    for j in 0..4 {
        let _ = writeln!(
            w,
            "    let s{j} = a.w0 * src[r{j} + a.o0] + a.w1 * src[r{j} + a.o1] + a.w2 * src[r{j} + a.o2] + a.w3 * src[r{j} + a.o3];"
        );
    }
    w.push_str(
        r#"    return b.w0 * s0 + b.w1 * s1 + b.w2 * s2 + b.w3 * s3;
}
// @@END
{TEXTURE}
// @@GENERIC
// The value of output element (c0, c1, c2, c3).
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
    if (!ok) { return p.off_grid; }
    let ax = make_axis(x, hx, p.gs0);
    if (!two) { return combine1(base, ax); }
    return combine2(base, ax, make_axis(y, hy, p.gs1));
}

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
@compute @workgroup_size(TX, TY)
fn main2d(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    if (gid.y >= p.od1) { return; }
    let c2 = gid.z % p.od2;
    let c3 = gid.z / p.od2;
    let stride = nwg.x * TX;
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

// @@END
// @@ROWS
// `Scaled` sources: the taps of a column are the same in every row, so an invocation computes
// them once for ROWS rows of its column.
@compute @workgroup_size(256)
fn main_rows(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= p.od0) { return; }
    let c2 = gid.z % p.od2;
    let c3 = gid.z / p.od2;
    let base = c2 * p.ib2 + c3 * p.ib3;
    let hy = f32(p.gn1) - 1.0;
    let ax = make_axis(f32(gid.x) / p.scale0, f32(p.gn0) - 1.0, p.gs0);
    let two = p.ndims == 2u;
    for (var r = 0u; r < ROWS; r += 1u) {
        let c1 = gid.y * ROWS + r;
        if (c1 >= p.od1) { break; }
        let ay = make_axis(f32(c1) / p.scale1, hy, p.gs1);
        out[gid.x + p.od0 * (c1 + p.od1 * gid.z)] = combine2(base, ax, ay);
    }
}
// @@END
"#,
    );
    let mut keep = vec![if rows_entry { "ROWS" } else { "GENERIC" }];
    if read == Read::Buffer {
        keep.extend(["ONE", "TWO"]);
    }
    let mut text = keep_sections(&w, &keep).replace("{TEXTURE}\n", &texture_reads(read));
    if read != Read::Buffer {
        // The texture is of one 2-D slice, so the taps of an axis are texel coordinates.
        text = text.replace("p.gs0", "1u").replace("p.gs1", "1u").replace("    if (!two) { return combine1(base, ax); }\n", "");
    }
    let consts = format!("const METHOD: u32 = {}u;\nconst KIND: u32 = {}u;\nconst NDIMS: u32 = {}u;\nconst PERSP: u32 = {}u;\n", spec.method, spec.kind, spec.ndims, spec.persp);
    consts + &text.replace("p.method", "METHOD").replace("p.kind", "KIND").replace("p.ndims", "NDIMS").replace("p.persp", "PERSP")
}

/// How the sampling kernel reads its source grid.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Read {
    /// A storage buffer, a load per tap.
    Buffer,
    /// A texture, a point fetch (`textureLoad`) per tap: the same taps as the buffer's.
    Load,
    /// A texture read by gathers (`textureGather`, one fetch for the four texels of a 2 x 2 block
    /// of taps); only for a source that is two-dimensional. Unfiltered, so the same values.
    Gather,
    /// A texture read by the texture unit's own bilinear filter, one fetch per output
    /// ([`Interp::LinearHardware`]): the weights are quantized by the hardware, so the values
    /// differ from [`Read::Buffer`]'s.
    Sample,
}

/// The bindings of `@group(1)` of the kernel.
fn bindings(rows_entry: bool, read: Read) -> String {
    let src = match read {
        Read::Buffer => "@group(1) @binding(0) var<storage, read> src: array<f32>;\n".to_owned(),
        Read::Load => "@group(1) @binding(0) var src: texture_2d<f32>;\n".to_owned(),
        Read::Gather | Read::Sample => "@group(1) @binding(0) var src: texture_2d<f32>;\n@group(1) @binding(1) var smp: sampler;\n".to_owned(),
    };
    let first = if matches!(read, Read::Gather | Read::Sample) { 2 } else { 1 };
    let mut out = src;
    let mut n = first;
    let mut add = |decl: &str| {
        let _ = writeln!(out, "@group(1) @binding({n}) {decl};");
        n += 1;
    };
    if !rows_entry {
        add("var<storage, read> pos0: array<f32>");
        add("var<storage, read> pos1: array<f32>");
    }
    add("var<storage, read_write> out: array<f32>");
    add("var<uniform> p: P");
    out
}

/// `combine2` of a texture read.
fn texture_reads(read: Read) -> String {
    if read == Read::Buffer {
        return String::new();
    }
    let mut w = String::from("fn texel(x: u32, y: u32) -> f32 { return textureLoad(src, vec2<i32>(i32(x), i32(y)), 0).x; }\n");
    if read == Read::Load {
        w.push_str(
            "fn combine2(base: u32, a: Axis, b: Axis) -> f32 {
    if (p.method <= 1u) { return texel(a.o0, b.o0); }
    if (p.method <= 3u) {
        let s0 = a.w0 * texel(a.o0, b.o0) + a.w1 * texel(a.o1, b.o0);
        let s1 = a.w0 * texel(a.o0, b.o1) + a.w1 * texel(a.o1, b.o1);
        return b.w0 * s0 + b.w1 * s1;
    }
",
        );
        for j in 0..4 {
            let _ = writeln!(w, "    let s{j} = a.w0 * texel(a.o0, b.o{j}) + a.w1 * texel(a.o1, b.o{j}) + a.w2 * texel(a.o2, b.o{j}) + a.w3 * texel(a.o3, b.o{j});");
        }
        w.push_str("    return b.w0 * s0 + b.w1 * s1 + b.w2 * s2 + b.w3 * s3;\n}\n");
    } else if read == Read::Sample {
        // The position is the grid coordinate of the taps' base plus the fraction a linear
        // method's second weight is; texel centres sit half a texel in.
        w.push_str(
            "fn combine2(base: u32, a: Axis, b: Axis) -> f32 {
    return textureSampleLevel(src, smp, vec2<f32>((a.f + a.w1 + 0.5) / f32(p.gn0), (b.f + b.w1 + 0.5) / f32(p.gn1)), 0.0).x;
}
",
        );
    } else if read == Read::Gather {
        // The gather at the texel centre (x + 1, y + 1) / size reads the block x..x+1, y..y+1
        // (WGSL order: (x, y+1), (x+1, y+1), (x+1, y), (x, y)), each clamped to the texture as
        // a tap is.
        w.push_str(
            "fn gat(x: f32, y: f32) -> vec4<f32> {
    return textureGather(0, src, smp, vec2<f32>((x + 1.0) / f32(p.gn0), (y + 1.0) / f32(p.gn1)));
}
fn combine2(base: u32, a: Axis, b: Axis) -> f32 {
    if (p.method <= 1u) { return texel(a.o0, b.o0); }
    if (p.method <= 3u) {
        let g = gat(a.f, b.f);
        let s0 = a.w0 * g.w + a.w1 * g.z;
        let s1 = a.w0 * g.x + a.w1 * g.y;
        return b.w0 * s0 + b.w1 * s1;
    }
    let g00 = gat(a.f - 1.0, b.f - 1.0);
    let g10 = gat(a.f + 1.0, b.f - 1.0);
    let g01 = gat(a.f - 1.0, b.f + 1.0);
    let g11 = gat(a.f + 1.0, b.f + 1.0);
    let s0 = a.w0 * g00.w + a.w1 * g00.z + a.w2 * g10.w + a.w3 * g10.z;
    let s1 = a.w0 * g00.x + a.w1 * g00.y + a.w2 * g10.x + a.w3 * g10.y;
    let s2 = a.w0 * g01.w + a.w1 * g01.z + a.w2 * g11.w + a.w3 * g11.z;
    let s3 = a.w0 * g01.x + a.w1 * g01.y + a.w2 * g11.x + a.w3 * g11.y;
    return b.w0 * s0 + b.w1 * s1 + b.w2 * s2 + b.w3 * s3;
}
",
        );
    }
    w
}

/// `text` without the sections (`// @@NAME` .. `// @@END`) other than `keep`.
fn keep_sections(text: &str, keep: &[&str]) -> String {
    let mut out = String::new();
    let mut section: Option<&str> = None;
    for line in text.lines() {
        if let Some(name) = line.strip_prefix("// @@") {
            section = if name == "END" { None } else { Some(name) };
            continue;
        }
        if section.is_none_or(|s| keep.contains(&s)) {
            out.push_str(line);
            out.push('\n');
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use super::Read;
    use crate::array::Array;
    use crate::interp::Interp;

    thread_local! {
        pub(super) static FORCE: Cell<Option<Read>> = const { Cell::new(None) };
        /// Launches that read their source through a texture.
        pub(super) static TEXTURED: Cell<u32> = const { Cell::new(0) };
    }

    fn values(n: usize, seed: u32, lo: f32, hi: f32) -> Vec<f32> {
        let mut x = seed.wrapping_mul(2654435761).wrapping_add(1);
        (0..n)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 17;
                x ^= x << 5;
                lo + (hi - lo) * (x >> 8) as f32 / (1u32 << 24) as f32
            })
            .collect()
    }

    /// The results of every way to read the source are those of the buffer, bit for bit.
    #[test]
    fn texture_reads_give_the_results_of_buffer_reads() {
        let ctx = crate::context::default_context().unwrap();
        if ctx.hal().texture_caps().is_none() {
            return;
        }
        let (w, h) = (128usize, 40usize);
        let img = Array::from_slice(&values(w * h, 3, -4.0, 4.0), [w, h]).unwrap();
        let m = 3000;
        let p0 = Array::from_slice(&values(m, 4, -2.0, w as f32 + 1.0), m).unwrap();
        let p1 = Array::from_slice(&values(m, 5, -2.0, h as f32 + 1.0), m).unwrap();
        for method in [Interp::Nearest, Interp::Lower, Interp::Linear, Interp::Cosine, Interp::Cubic, Interp::CubicSpline] {
            let run = || {
                [
                    img.approx2(&p0, &p1, method, -9.0).to_vec().unwrap(),
                    img.resize(200, 70, method).to_vec().unwrap(),
                    img.resize(50, 21, method).to_vec().unwrap(),
                    img.rotate(0.3, false, method).to_vec().unwrap(),
                    img.transform([0.9f32, 0.2, 3.0, -0.1, 1.1, 1.0], 120, 60, method, true).to_vec().unwrap(),
                ]
            };
            FORCE.set(Some(Read::Buffer));
            let base = run();
            for read in [Read::Load, Read::Gather] {
                FORCE.set(Some(read));
                let got = run();
                for (k, (g, b)) in got.iter().zip(&base).enumerate() {
                    assert_eq!(g.len(), b.len());
                    let bad = g.iter().zip(b).filter(|(x, y)| x.to_bits() != y.to_bits() && !(x.is_nan() && y.is_nan())).count();
                    assert_eq!(bad, 0, "{read:?} {method:?} case {k}: {bad} of {} elements differ", b.len());
                }
            }
            FORCE.set(None);
        }
        assert!(TEXTURED.get() > 0, "no launch read through a texture");
    }

    /// The hardware filter stays within its weight quantization (8 bits of fraction) of bilinear.
    #[test]
    fn hardware_filtering_is_within_its_quantization_of_bilinear() {
        let ctx = crate::context::default_context().unwrap();
        if ctx.hal().texture_caps().is_none() {
            return;
        }
        let (w, h) = (128usize, 40usize);
        let img = Array::from_slice(&values(w * h, 7, -4.0, 4.0), [w, h]).unwrap();
        let m = 3000;
        let p0 = Array::from_slice(&values(m, 8, -2.0, w as f32 + 1.0), m).unwrap();
        let p1 = Array::from_slice(&values(m, 9, -2.0, h as f32 + 1.0), m).unwrap();
        let cases = |method| {
            [
                img.approx2(&p0, &p1, method, -9.0).to_vec().unwrap(),
                img.resize(300, 70, method).to_vec().unwrap(),
                img.rotate(0.3, false, method).to_vec().unwrap(),
            ]
        };
        let exact = cases(Interp::Linear);
        TEXTURED.set(0);
        let hardware = cases(Interp::LinearHardware);
        assert!(TEXTURED.get() > 0, "the hardware filter did not run");
        let mut worst = 0.0f32;
        for (e, hw) in exact.iter().zip(&hardware) {
            assert_eq!(e.len(), hw.len());
            for (x, y) in e.iter().zip(hw) {
                worst = worst.max((x - y).abs());
            }
        }
        // Two axes, each weight off by at most 1/256 (8 bits of fraction) of a span of at most 8.
        assert!(worst <= 2.0 * 8.0 / 256.0 + 1e-3, "worst difference {worst}");
    }
}
