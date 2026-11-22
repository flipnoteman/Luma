//! Anisotropic diffusion (`Array::anisotropic_diffusion`).
//!
//! One iteration is ArrayFire's: with `g` the gradient of the current image (central differences,
//! one-sided at the border), `mean(g^2)` over **every element** of the array, the exponent
//! factor `mct = -1 / (2 K^2 mean(g^2))` and, for every pixel, the flux through its four
//! half-pixel faces,
//! `u' = u + dt * sum over faces of c(|grad u|^2 mct) * (u_neighbour - u)` with
//! `|grad u|^2` taken at the face from the two differences across it and along it. The image
//! outside is the nearest pixel.
//!
//! Three launches per iteration: the energy `sum g^2` per tile, one workgroup that sums the tiles
//! and writes `mct` (no host round trip: ArrayFire reads the sum back every iteration), and the
//! update, which reads one image and writes the other (the two ping-pong, the last one is the
//! result).

use std::fmt::Write as _;
use std::panic::Location;

use super::{geometry, plane_grid};
use crate::array::Array;
use crate::builtin::Builtin;
use crate::context::{RawArg, Scratch};
use crate::ops::Outcome;
use luma_core::Buffer;
use crate::vision::{invalid, Kernel as Any};

/// The flux (conductance) function of the Perona-Malik equation.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum FluxFunction {
    /// `1 / (1 + x)`, ArrayFire's `AF_FLUX_QUADRATIC`.
    Quadratic,
    /// `exp(x)`, ArrayFire's `AF_FLUX_EXPONENTIAL` (its default).
    Exponential,
}

/// The diffusion equation.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum DiffusionEq {
    /// Perona-Malik gradient diffusion (`AF_DIFFUSION_GRAD`, ArrayFire's default).
    Gradient,
    /// Modified curvature diffusion (`AF_DIFFUSION_MCDE`); always uses the exponential flux.
    Mcde,
}

const TX: u32 = 32;
const TY: u32 = 8;
const REDUCE_GROUP: u32 = 256;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum Kernel {
    /// Per-tile sums of the squared gradient.
    Energy,
    /// `mct` from the tile sums.
    Reduce,
    /// One diffusion step; `mcde` has no flux function of its own (it is `Exponential`).
    Update { flux: FluxFunction, mcde: bool },
}

pub(crate) fn source(k: Kernel) -> String {
    match k {
        Kernel::Energy => energy_source(),
        Kernel::Reduce => reduce_source(),
        Kernel::Update { flux, mcde } => update_source(flux, mcde),
    }
}

/// Stages the `(32 + 2) x (8 + 2)` pixels around the tile of workgroup `(wid.x, wid.y % tiles_y)`
/// (the nearest pixel outside the image) into `tile`.
fn stage() -> String {
    let (sw, sh) = (TX + 2, TY + 2);
    format!(
        "    let ty = wid.y % p.tiles_y;
    let plane = wid.y / p.tiles_y;
    let h = i32(p.h);
    let w = i32(p.w);
    let base = plane * p.h * p.w;
    let o0 = i32(wid.x) * {TX} - 1;
    let o1 = i32(ty) * {TY} - 1;
    for (var i = t; i < {total}u; i += 256u) {{
        let a0 = clamp(o0 + i32(i % {sw}u), 0, h - 1);
        let a1 = clamp(o1 + i32(i / {sw}u), 0, w - 1);
        tile[i] = src[base + u32(a0) + p.h * u32(a1)];
    }}
    workgroupBarrier();
    let lx = t & 31u;
    let ly = t >> 5u;
    let gx = i32(wid.x) * {TX} + i32(lx);
    let gy = i32(ty) * {TY} + i32(ly);
    let inside = gx < h && gy < w;
",
        total = sw * sh
    )
}

const STRUCT: &str = "struct Params { h: u32, w: u32, planes: u32, tiles_y: u32, dt: f32, n: f32, k2: f32, tiles: u32 }\n";

/// Arguments: image, partial sums, uniform.
fn energy_source() -> String {
    let (sw, sh) = (TX + 2, TY + 2);
    let mut s = String::from(STRUCT);
    s.push_str("@group(1) @binding(0) var<storage, read> src: array<f32>;\n@group(1) @binding(1) var<storage, read_write> sums: array<f32>;\n@group(1) @binding(2) var<uniform> p: Params;\n");
    let _ = writeln!(s, "var<workgroup> tile: array<f32, {}>;\nvar<workgroup> red: array<f32, 256>;", sw * sh);
    s.push_str("@compute @workgroup_size(256)\nfn main(@builtin(local_invocation_index) t: u32, @builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {\n");
    s.push_str(&stage());
    let _ = write!(
        s,
        "    let c = (ly + 1u) * {sw}u + lx + 1u;
    // The gradient along axis 0 and axis 1: half the central difference, the full one-sided at the edge.
    var e = 0.0;
    if (inside) {{
        let up = tile[c + 1u];
        let dn = tile[c - 1u];
        let rt = tile[c + {sw}u];
        let lt = tile[c - {sw}u];
        let f0 = select(0.5, 1.0, gx == 0 || gx >= h - 1);
        let f1 = select(0.5, 1.0, gy == 0 || gy >= w - 1);
        let g0 = f0 * (up - dn);
        let g1 = f1 * (rt - lt);
        e = g0 * g0 + g1 * g1;
    }}
    red[t] = e;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {{
        if (t < s) {{ red[t] = red[t] + red[t + s]; }}
        workgroupBarrier();
    }}
    if (t == 0u) {{ sums[wid.x + wid.y * nwg.x] = red[0]; }}
}}
"
    );
    s
}

/// Arguments: partial sums, mct (one word), uniform. `p.tiles` is the number of partial sums.
fn reduce_source() -> String {
    let mut s = String::from(STRUCT);
    s.push_str("@group(1) @binding(0) var<storage, read> sums: array<f32>;\n@group(1) @binding(1) var<storage, read_write> mct: array<f32>;\n@group(1) @binding(2) var<uniform> p: Params;\n");
    s.push_str("var<workgroup> red: array<f32, 256>;\n");
    let _ = write!(
        s,
        "@compute @workgroup_size({REDUCE_GROUP})
fn main(@builtin(local_invocation_index) t: u32) {{
    var acc = 0.0;
    for (var i = t; i < p.tiles; i += {REDUCE_GROUP}u) {{ acc += sums[i]; }}
    red[t] = acc;
    workgroupBarrier();
    for (var s = {half}u; s > 0u; s = s >> 1u) {{
        if (t < s) {{ red[t] = red[t] + red[t + s]; }}
        workgroupBarrier();
    }}
    // mct = -n / (2 K^2 sum); a constant image (sum 0) is a fixed point whatever the factor.
    if (t == 0u) {{
        let total = red[0];
        mct[0] = select(0.0, -p.n / (p.k2 * total), total > 0.0);
    }}
}}
",
        half = REDUCE_GROUP / 2
    );
    s
}

/// Arguments: image, result, mct, uniform.
fn update_source(flux: FluxFunction, mcde: bool) -> String {
    let (sw, sh) = (TX + 2, TY + 2);
    let mut s = String::from(STRUCT);
    s.push_str("@group(1) @binding(0) var<storage, read> src: array<f32>;\n@group(1) @binding(1) var<storage, read_write> dst: array<f32>;\n@group(1) @binding(2) var<storage, read> mct_buf: array<f32>;\n@group(1) @binding(3) var<uniform> p: Params;\n");
    let _ = writeln!(s, "var<workgroup> tile: array<f32, {}>;", sw * sh);
    let c = match flux {
        FluxFunction::Exponential => "exp(v)",
        FluxFunction::Quadratic => "1.0 / (1.0 + v)",
    };
    let _ = writeln!(s, "fn cond(v: f32) -> f32 {{ return {c}; }}");
    if mcde {
        s.push_str(
            "fn update(mct: f32, C: f32, S: f32, N: f32, W: f32, E: f32, SE: f32, SW: f32, NE: f32, NW: f32) -> f32 {
    let dx = (E - W) * 0.5;
    let dy = (S - N) * 0.5;
    var df = E - C;
    var db = C - W;
    let df0 = df;
    let db0 = db;
    var a = dy + 0.5 * (SE - NE);
    var b = dy + 0.5 * (SW - NW);
    var gmsqf = df * df + 0.25 * a * a;
    var gmsqb = db * db + 0.25 * b * b;
    var delta = (df / sqrt(1.0e-10 + gmsqf)) * exp(gmsqf * mct) - (db / sqrt(1.0e-10 + gmsqb)) * exp(gmsqb * mct);
    df = S - C;
    db = C - N;
    a = dx + 0.5 * (SE - SW);
    b = dx + 0.5 * (NE - NW);
    gmsqf = df * df + 0.25 * a * a;
    gmsqb = db * db + 0.25 * b * b;
    delta += (df / sqrt(1.0e-10 + gmsqf)) * exp(gmsqf * mct) - (db / sqrt(1.0e-10 + gmsqb)) * exp(gmsqb * mct);
    var prop = 0.0;
    if (delta > 0.0) {
        prop = min(db0, 0.0) * min(db0, 0.0) + max(df0, 0.0) * max(df0, 0.0) + min(db, 0.0) * min(db, 0.0) + max(df, 0.0) * max(df, 0.0);
    } else {
        prop = max(db0, 0.0) * max(db0, 0.0) + min(df0, 0.0) * min(df0, 0.0) + max(db, 0.0) * max(db, 0.0) + min(df, 0.0) * min(df, 0.0);
    }
    return sqrt(prop) * delta;
}
",
        );
    } else {
        s.push_str(
            "fn update(mct: f32, C: f32, S: f32, N: f32, W: f32, E: f32, SE: f32, SW: f32, NE: f32, NW: f32) -> f32 {
    let dx = (E - W) * 0.5;
    let dy = (S - N) * 0.5;
    var df = E - C;
    var db = C - W;
    var a = dy + 0.5 * (SE - NE);
    var b = dy + 0.5 * (SW - NW);
    var delta = cond((df * df + 0.25 * a * a) * mct) * df - cond((db * db + 0.25 * b * b) * mct) * db;
    df = S - C;
    db = C - N;
    a = dx + 0.5 * (SE - SW);
    b = dx + 0.5 * (NE - NW);
    delta += cond((df * df + 0.25 * a * a) * mct) * df - cond((db * db + 0.25 * b * b) * mct) * db;
    return delta;
}
",
        );
    }
    s.push_str("@compute @workgroup_size(256)\nfn main(@builtin(local_invocation_index) t: u32, @builtin(workgroup_id) wid: vec3<u32>) {\n");
    s.push_str(&stage());
    // Axis 0 of the image is the fastest axis of the tile: `S` is the next pixel along axis 0.
    let _ = write!(
        s,
        "    if (!inside) {{ return; }}
    let c = (ly + 1u) * {sw}u + lx + 1u;
    let ctr = tile[c];
    let d = update(mct_buf[0], ctr, tile[c + 1u], tile[c - 1u], tile[c - {sw}u], tile[c + {sw}u], tile[c + {sw}u + 1u], tile[c - {sw}u + 1u], tile[c + {sw}u - 1u], tile[c - {sw}u - 1u]);
    dst[base + u32(gx) + p.h * u32(gy)] = ctr + d * p.dt;
}}
"
    );
    s
}

impl Array<f32> {
    /// Anisotropic (edge-preserving) smoothing of `[H, W]` images (ArrayFire's
    /// `anisotropicDiffusion`; axes 2 and 3 are a batch): `iterations` steps of Perona-Malik
    /// gradient diffusion ([`DiffusionEq::Gradient`]) or of modified curvature diffusion
    /// ([`DiffusionEq::Mcde`], which always uses the exponential flux) with time step `timestep`.
    ///
    /// `conductance` (`K`) is relative: every iteration the exponent factor is
    /// `-1 / (2 K^2 mean(g^2))` with `g` the gradient of the current image (central differences,
    /// one-sided at the border) and the mean over every element of the array, batch included
    /// (ArrayFire does the same). The flux through the face between two pixels is
    /// `c(x) * (u_neighbour - u)` with `c = exp(x)` ([`FluxFunction::Exponential`]) or
    /// `c = 1 / (1 + x)` ([`FluxFunction::Quadratic`]) of `x = mct * |grad u|^2` evaluated at the
    /// face; outside the image the nearest pixel is used.
    ///
    /// ArrayFire's quadratic function has the sign of `x` inverted (`mct` is negative, so
    /// `c = 1 / (1 - |grad u|^2 / (2 K^2 mean g^2))`, which is above 1 and singular where the
    /// gradient is large): luma reproduces it to give the same results; prefer the exponential
    /// flux. Like ArrayFire there is no stability check of `timestep` (it stays stable for about
    /// `timestep <= 0.125`). A constant image is returned unchanged (ArrayFire's is NaN).
    ///
    /// Differences from ArrayFire: a step reads one image and writes another, so the result does
    /// not depend on the order of the pixels (ArrayFire's CPU backend updates in place, pixel by
    /// pixel). ArrayFire's CUDA kernel has a bug: on images with more than about 70 columns it
    /// updates only the first few columns along axis 1 (72 of 128, 80 of 130, 88 of 200, 96 of
    /// 300 in the sizes probed) and leaves the others as they were. luma updates every pixel. On the columns ArrayFire does update, one step agrees
    /// with it to `f32` rounding (about `1e-5`) on small images; on large ones its in-place update
    /// also races between tiles (a tile reads neighbours another tile has already written), so
    /// its result varies from run to run (differences of 0.3% to 43% of the step were measured).
    /// Poisons the result for `iterations == 0` or a conductance of 0 (as ArrayFire errors), and
    /// for non-finite arguments.
    ///
    /// Lazy; three launches per iteration, without a round trip to the host.
    ///
    /// ```
    /// use luma::{Array, DiffusionEq, FluxFunction};
    /// # fn main() -> Result<(), luma::LumaError> {
    /// // A spike on a flat image spreads out and loses height.
    /// let mut data = vec![0.0f32; 9 * 9];
    /// data[4 + 9 * 4] = 100.0;
    /// let img = Array::from_slice(&data, [9, 9])?;
    /// let out = img.anisotropic_diffusion(0.1, 10.0, 1, FluxFunction::Exponential, DiffusionEq::Gradient).to_vec()?;
    /// assert!(out[4 + 9 * 4] < 100.0 && out[3 + 9 * 4] > 0.0);
    /// # Ok(()) }
    /// ```
    #[track_caller]
    pub fn anisotropic_diffusion(&self, timestep: f32, conductance: f32, iterations: u32, flux: FluxFunction, equation: DiffusionEq) -> Array<f32> {
        let location = Location::caller();
        Array::resolve(self.context(), self.shape(), run(self, timestep, conductance, iterations, flux, equation), "anisotropic_diffusion", location)
    }
}

fn run(input: &Array<f32>, dt: f32, k: f32, iterations: u32, flux: FluxFunction, equation: DiffusionEq) -> Outcome<Array<f32>> {
    if iterations == 0 {
        return Err(invalid("anisotropic_diffusion: iterations must be at least 1"));
    }
    if !(dt.is_finite() && k.is_finite() && k != 0.0) {
        return Err(invalid(format!("anisotropic_diffusion: timestep {dt} and conductance {k} must be finite and the conductance not 0")));
    }
    let ctx = input.context();
    input.storage()?;
    let (h, w, planes) = geometry(input.shape())?;
    let (tx, ty) = (h.div_ceil(TX), w.div_ceil(TY));
    let grid = plane_grid(ctx, tx, ty, planes)?;
    let tiles = tx * ty * planes;
    let n = input.len() as f32;
    let params = [h, w, planes, ty, dt.to_bits(), n.to_bits(), (2.0 * k * k).to_bits(), tiles];
    let uniform = bytemuck::cast_slice(&params);
    let mcde = equation == DiffusionEq::Mcde;
    let flux = if mcde { FluxFunction::Exponential } else { flux };

    let out = Array::<f32>::alloc(ctx, input.shape())?;
    let (src0, src_bytes) = input.ready()?;
    let (out_pin, out_bytes) = out.ready()?;
    let mut scratch = Scratch::new(ctx);
    // The second image only exists for more than one iteration.
    let other = if iterations > 1 { Some(scratch.take(out_bytes)?) } else { None };
    let sums_bytes = u64::from(tiles) * 4;
    let sums = scratch.take(sums_bytes)?;
    let mct = scratch.take(16)?;
    let energy = ctx.builtin(Builtin::Vision(Any::Edges(super::Kernel::Diffusion(Kernel::Energy))), "f32")?;
    let reduce = ctx.builtin(Builtin::Vision(Any::Edges(super::Kernel::Diffusion(Kernel::Reduce))), "f32")?;
    let update = ctx.builtin(Builtin::Vision(Any::Edges(super::Kernel::Diffusion(Kernel::Update { flux, mcde }))), "f32")?;
    // Step `s` writes `out` when `iterations - 1 - s` is even, the other image otherwise; the
    // first step reads the input and every later one what the one before wrote.
    let target = |step: u32| -> (&Buffer, u64) {
        if (iterations - 1 - step) % 2 == 0 {
            (out_pin.buffer(), out_bytes)
        } else {
            (other.as_ref().expect("a second image"), out_bytes)
        }
    };
    for step in 0..iterations {
        let (read, read_bytes) = if step == 0 { (src0.buffer(), src_bytes) } else { target(step - 1) };
        let (write, write_bytes) = target(step);
        ctx.launch(&energy, grid, [h, w, planes], &[RawArg::read(read, read_bytes), RawArg::write(&sums, sums_bytes), RawArg::Uniform(uniform)])?;
        ctx.launch(&reduce, [1, 1, 1], [REDUCE_GROUP, 1, 1], &[RawArg::read(&sums, sums_bytes), RawArg::write(&mct, 16), RawArg::Uniform(uniform)])?;
        ctx.launch(&update, grid, [h, w, planes], &[RawArg::read(read, read_bytes), RawArg::write(write, write_bytes), RawArg::read(&mct, 16), RawArg::Uniform(uniform)])?;
    }
    Ok(out)
}
