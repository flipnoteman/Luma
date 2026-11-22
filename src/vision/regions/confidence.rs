//! Confidence connected components (`Array<f32>::confidence_cc`).
//!
//! ArrayFire's algorithm: the image is scaled linearly to `[1, 2]`; the mean and standard
//! deviation of the pixels in the `(2r + 1)^2` windows around the seeds give the interval
//! `[mean - m sd, mean + m sd]`, widened to contain the seeds' own values; the first segmentation
//! is the set of pixels with a value in the interval that are 8-connected to a seed through such
//! pixels. Every further iteration re-estimates mean and standard deviation from the pixels of the
//! current segmentation (the seeds' values still widen the interval) and segments again from the
//! seeds. The iterations stop when the segmentation is empty or when a variance is below 1e-6.
//!
//! Everything runs on the device without a read-back. A segmentation is the connected components
//! (union-find, the tiles and seams of [`label`](super::label)) of the pixels in the interval
//! that hold a seed: the pixels of the interval are labelled with the root of their component, the
//! seeds mark their roots, every pixel takes the mark of its root. An iteration is
//! `Sums` (count, sum and sum of squares of the pixels of the segmentation, per workgroup),
//! `Stats` (one workgroup: the interval), then the labelling; a control buffer carries the
//! interval and a `done` flag that turns the launches of the remaining iterations into no-ops.

use std::fmt::Write as _;

use crate::array::Array;
use crate::builtin::Builtin;
use crate::context::{Context, RawArg, Scratch};
use crate::ops::{fold_workgroups, Outcome};
use crate::vision::{invalid, Kernel as Any};

use super::label::{self, union_find, Geom, GROUP_SIZE, NONE_WORD, PARAMS};

/// Pixels a workgroup of [`Kernel::MinMax`] and [`Kernel::Sums`] covers (4 per invocation).
const CHUNK: u32 = label::CHUNK_PIXELS;
/// The variance below which the segmentation is final (ArrayFire's `epsilon`).
const EPSILON: &str = "1.0e-6";
/// Shift of the scaled pixel values in the sums (their mean is about 1.5): keeps the variance
/// free of cancellation.
const SHIFT: &str = "1.5";

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum Kernel {
    /// Minimum and maximum of the pixels of every chunk.
    MinMax,
    /// One workgroup: the scaling, the statistics of the seed windows, the first interval.
    Setup,
    /// Count, sum and sum of squares of the pixels of the segmentation, per chunk.
    Sums,
    /// One workgroup: the next interval from the sums.
    Stats,
    /// The seeds mark the roots of their components.
    Mark,
    /// Every pixel takes the mark of its root.
    Out,
}

/// `norm`, the pixel's value scaled to `[1, 2]` by the control words; needs a binding `ctrl`.
pub(super) fn norm_source() -> String {
    "fn jn(v: f32, mn: f32, inv: f32, flat: u32) -> f32 {
    if (flat != 0u) { return 1.0; }
    return (v - mn) * inv + 1.0;
}
fn norm(v: f32) -> f32 { return jn(v, bitcast<f32>(ctrl[2]), bitcast<f32>(ctrl[3]), ctrl[4]); }
"
    .to_string()
}

/// The control words: 0 and 1 the interval, 2 and 3 the minimum and the reciprocal of the range
/// of the image, 4 whether the image is constant, 6 whether the iterations are over after the
/// current one, 7 whether they are over, 8 and 9 the smallest and largest seed value.
const CTRL_BYTES: u64 = 16 * 4;

/// Reduces the shared arrays `names` (256 entries each) with `ops` into entry 0 (all invocations
/// must call it).
fn tree(s: &mut String, items: &[(&str, &str)]) {
    s.push_str("    workgroupBarrier();\n    for (var s = 128u; s > 0u; s = s >> 1u) {\n        if (t < s) {\n");
    for (name, op) in items {
        let _ = match *op {
            "+" => writeln!(s, "            {name}[t] = {name}[t] + {name}[t + s];"),
            f => writeln!(s, "            {name}[t] = {f}({name}[t], {name}[t + s]);"),
        };
    }
    s.push_str("        }\n        workgroupBarrier();\n    }\n");
}

pub(crate) fn source(k: Kernel) -> String {
    let mut s = String::from(PARAMS);
    let g = GROUP_SIZE;
    match k {
        Kernel::MinMax => {
            s.push_str("@group(1) @binding(0) var<storage, read> img: array<f32>;\n@group(1) @binding(1) var<storage, read_write> partial: array<u32>;\n@group(1) @binding(2) var<uniform> p: Params;\n");
            let _ = write!(
                s,
                "var<workgroup> lo: array<f32, {g}>;
var<workgroup> hi: array<f32, {g}>;
@compute @workgroup_size({g})
fn main(@builtin(local_invocation_index) t: u32, @builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {{
    let chunk = wid.x + wid.y * nwg.x;
    let base = chunk * {CHUNK}u + t * 4u;
    var mn = 3.0e38;
    var mx = -3.0e38;
    for (var j = 0u; j < 4u; j++) {{
        let q = base + j;
        if (q < p.n) {{ let v = img[q]; mn = min(mn, v); mx = max(mx, v); }}
    }}
    lo[t] = mn;
    hi[t] = mx;
"
            );
            tree(&mut s, &[("lo", "min"), ("hi", "max")]);
            s.push_str("    if (t == 0u) { partial[2u * chunk] = bitcast<u32>(lo[0]); partial[2u * chunk + 1u] = bitcast<u32>(hi[0]); }\n}\n");
        }
        Kernel::Setup => {
            s.push_str(
                "@group(1) @binding(0) var<storage, read> img: array<f32>;
@group(1) @binding(1) var<storage, read> partial: array<u32>;
@group(1) @binding(2) var<storage, read> sx: array<u32>;
@group(1) @binding(3) var<storage, read> sy: array<u32>;
@group(1) @binding(4) var<storage, read_write> ctrl: array<u32>;
@group(1) @binding(5) var<uniform> p: Params;
",
            );
            s.push_str(&norm_source());
            let _ = write!(
                s,
                "var<workgroup> a0: array<f32, {g}>;
var<workgroup> a1: array<f32, {g}>;
var<workgroup> a2: array<f32, {g}>;
var<workgroup> a3: array<f32, {g}>;
var<workgroup> a4: array<f32, {g}>;
var<workgroup> cnt: array<u32, {g}>;
@compute @workgroup_size({g})
fn main(@builtin(local_invocation_index) t: u32) {{
    var mn = 3.0e38;
    var mx = -3.0e38;
    for (var c = t; c < p.groups; c += {g}u) {{
        mn = min(mn, bitcast<f32>(partial[2u * c]));
        mx = max(mx, bitcast<f32>(partial[2u * c + 1u]));
    }}
    a0[t] = mn;
    a1[t] = mx;
"
            );
            tree(&mut s, &[("a0", "min"), ("a1", "max")]);
            let _ = write!(
                s,
                "    let imn = a0[0];
    let range = a1[0] - imn;
    let flat = select(0u, 1u, !(abs(range) >= 1.0e-6));
    var inv = 0.0;
    if (flat == 0u) {{ inv = 1.0 / range; }}
    workgroupBarrier();
    var s1 = 0.0;
    var s2 = 0.0;
    var n = 0u;
    var smn = 3.0e38;
    var smx = -3.0e38;
    for (var q = t; q < p.ns; q += {g}u) {{
        let x = sx[q];
        let y = sy[q];
        if (x < p.h && y < p.w) {{
            let v = jn(img[x + p.h * y], imn, inv, flat);
            smn = min(smn, v);
            smx = max(smx, v);
            var x0 = 0u;
            if (x >= p.radius) {{ x0 = x - p.radius; }}
            var y0 = 0u;
            if (y >= p.radius) {{ y0 = y - p.radius; }}
            let x1 = min(x + p.radius, p.h - 1u);
            let y1 = min(y + p.radius, p.w - 1u);
            for (var b = y0; b <= y1; b++) {{
                for (var a = x0; a <= x1; a++) {{
                    let d = jn(img[a + p.h * b], imn, inv, flat) - {SHIFT};
                    s1 += d;
                    s2 += d * d;
                    n++;
                }}
            }}
        }}
    }}
    a0[t] = s1;
    a1[t] = s2;
    a2[t] = smn;
    a3[t] = smx;
    cnt[t] = n;
"
            );
            tree(&mut s, &[("a0", "+"), ("a1", "+"), ("a2", "min"), ("a3", "max"), ("cnt", "+")]);
            let _ = write!(
                s,
                "    if (t == 0u) {{
        let m = bitcast<f32>(p.mult);
        var lo = 1.0;
        var hi = 0.0;
        var stop = 1u;
        if (cnt[0] != 0u) {{
            let nf = f32(cnt[0]);
            let mean = {SHIFT} + a0[0] / nf;
            var v = 0.0;
            if (cnt[0] > 1u) {{ v = (a1[0] - a0[0] * a0[0] / nf) / (nf - 1.0); }}
            let sd = sqrt(max(v, 0.0));
            lo = min(a2[0], mean - m * sd);
            hi = max(a3[0], mean + m * sd);
            stop = select(0u, 1u, abs(v) < {EPSILON});
        }}
        ctrl[0] = bitcast<u32>(lo);
        ctrl[1] = bitcast<u32>(hi);
        ctrl[2] = bitcast<u32>(imn);
        ctrl[3] = bitcast<u32>(inv);
        ctrl[4] = flat;
        ctrl[5] = 0u;
        ctrl[6] = stop;
        ctrl[7] = 0u;
        ctrl[8] = bitcast<u32>(a2[0]);
        ctrl[9] = bitcast<u32>(a3[0]);
    }}
}}
"
            );
        }
        Kernel::Sums => {
            s.push_str(
                "@group(1) @binding(0) var<storage, read> img: array<f32>;
@group(1) @binding(1) var<storage, read> seg: array<u32>;
@group(1) @binding(2) var<storage, read> ctrl: array<u32>;
@group(1) @binding(3) var<storage, read_write> partial: array<u32>;
@group(1) @binding(4) var<uniform> p: Params;
var<workgroup> stop: u32;
",
            );
            s.push_str(&norm_source());
            let _ = write!(
                s,
                "var<workgroup> a0: array<f32, {g}>;
var<workgroup> a1: array<f32, {g}>;
var<workgroup> cnt: array<u32, {g}>;
@compute @workgroup_size({g})
fn main(@builtin(local_invocation_index) t: u32, @builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {{
    if (t == 0u) {{ stop = ctrl[7] | ctrl[6]; }}
    workgroupBarrier();
    if (workgroupUniformLoad(&stop) != 0u) {{ return; }}
    let chunk = wid.x + wid.y * nwg.x;
    let base = chunk * {CHUNK}u + t * 4u;
    var s1 = 0.0;
    var s2 = 0.0;
    var n = 0u;
    for (var j = 0u; j < 4u; j++) {{
        let q = base + j;
        if (q < p.n && seg[q] != 0u) {{
            let d = norm(img[q]) - {SHIFT};
            s1 += d;
            s2 += d * d;
            n++;
        }}
    }}
    a0[t] = s1;
    a1[t] = s2;
    cnt[t] = n;
"
            );
            tree(&mut s, &[("a0", "+"), ("a1", "+"), ("cnt", "+")]);
            s.push_str("    if (t == 0u) { partial[3u * chunk] = cnt[0]; partial[3u * chunk + 1u] = bitcast<u32>(a0[0]); partial[3u * chunk + 2u] = bitcast<u32>(a1[0]); }\n}\n");
        }
        Kernel::Stats => {
            s.push_str(
                "@group(1) @binding(0) var<storage, read> partial: array<u32>;
@group(1) @binding(1) var<storage, read_write> ctrl: array<u32>;
@group(1) @binding(2) var<uniform> p: Params;
var<workgroup> stop: u32;
",
            );
            let _ = write!(
                s,
                "var<workgroup> a0: array<f32, {g}>;
var<workgroup> a1: array<f32, {g}>;
var<workgroup> cnt: array<u32, {g}>;
@compute @workgroup_size({g})
fn main(@builtin(local_invocation_index) t: u32) {{
    if (t == 0u) {{
        stop = ctrl[7] | ctrl[6];
        if (ctrl[6] != 0u) {{ ctrl[7] = 1u; }}
    }}
    workgroupBarrier();
    if (workgroupUniformLoad(&stop) != 0u) {{ return; }}
    var s1 = 0.0;
    var s2 = 0.0;
    var n = 0u;
    for (var c = t; c < p.groups; c += {g}u) {{
        n += partial[3u * c];
        s1 += bitcast<f32>(partial[3u * c + 1u]);
        s2 += bitcast<f32>(partial[3u * c + 2u]);
    }}
    a0[t] = s1;
    a1[t] = s2;
    cnt[t] = n;
"
            );
            tree(&mut s, &[("a0", "+"), ("a1", "+"), ("cnt", "+")]);
            let _ = write!(
                s,
                "    if (t == 0u) {{
        if (cnt[0] == 0u) {{
            ctrl[7] = 1u;
        }} else {{
            let m = bitcast<f32>(p.mult);
            let nf = f32(cnt[0]);
            let mean = {SHIFT} + a0[0] / nf;
            var v = 0.0;
            if (cnt[0] > 1u) {{ v = (a1[0] - a0[0] * a0[0] / nf) / (nf - 1.0); }}
            let sd = sqrt(max(v, 0.0));
            ctrl[0] = bitcast<u32>(min(bitcast<f32>(ctrl[8]), mean - m * sd));
            ctrl[1] = bitcast<u32>(max(bitcast<f32>(ctrl[9]), mean + m * sd));
            ctrl[6] = select(0u, 1u, abs(v) < {EPSILON});
        }}
    }}
}}
"
            );
        }
        Kernel::Mark => {
            s.push_str(
                "@group(1) @binding(0) var<storage, read_write> lab: array<atomic<u32>>;
@group(1) @binding(1) var<storage, read_write> seg: array<atomic<u32>>;
@group(1) @binding(2) var<storage, read> ctrl: array<u32>;
@group(1) @binding(3) var<storage, read> sx: array<u32>;
@group(1) @binding(4) var<storage, read> sy: array<u32>;
@group(1) @binding(5) var<uniform> p: Params;
",
            );
            union_find(&mut s, "lab", "g");
            let _ = write!(
                s,
                "@compute @workgroup_size({g})
fn main(@builtin(local_invocation_index) t: u32, @builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {{
    if (ctrl[7] != 0u) {{ return; }}
    let q = (wid.x + wid.y * nwg.x) * {g}u + t;
    if (q >= p.ns) {{ return; }}
    let x = sx[q];
    let y = sy[q];
    if (x < p.h && y < p.w) {{
        let l = atomicLoad(&lab[x + p.h * y]);
        if (l != {NONE_WORD}) {{ atomicStore(&seg[find_g(l)], 1u); }}
    }}
}}
"
            );
        }
        Kernel::Out => {
            s.push_str(
                "@group(1) @binding(0) var<storage, read_write> lab: array<atomic<u32>>;
@group(1) @binding(1) var<storage, read_write> seg: array<atomic<u32>>;
@group(1) @binding(2) var<storage, read> ctrl: array<u32>;
@group(1) @binding(3) var<uniform> p: Params;
",
            );
            union_find(&mut s, "lab", "g");
            let _ = write!(
                s,
                "@compute @workgroup_size({g})
fn main(@builtin(local_invocation_index) t: u32, @builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {{
    if (ctrl[7] != 0u) {{ return; }}
    let q = (wid.x + wid.y * nwg.x) * {g}u + t;
    if (q >= p.n) {{ return; }}
    let l = atomicLoad(&lab[q]);
    if (l == {NONE_WORD}) {{ return; }}
    let r = find_g(l);
    if (r != q && atomicLoad(&seg[r]) != 0u) {{ atomicStore(&seg[q], 1u); }}
}}
"
            );
        }
    }
    s
}

/// The segmentation (1 for the pixels of the components, 0 elsewhere) of `image`.
pub(super) fn run(image: &Array<f32>, seeds_x: &Array<u32>, seeds_y: &Array<u32>, radius: u32, multiplier: u32, iterations: u32) -> Outcome<Array<u32>> {
    let ctx = image.context();
    let geom = Geom::new("confidence_cc", image.shape())?;
    let seeds = seeds_x.len();
    if seeds == 0 || seeds != seeds_y.len() {
        return Err(invalid(format!("confidence_cc needs as many seed rows as columns and at least one, got {} and {}", seeds_x.len(), seeds_y.len())));
    }
    let seed_count = u32::try_from(seeds).map_err(|_| invalid("confidence_cc: too many seeds"))?;
    image.storage()?;
    let seg = Array::<u32>::alloc(ctx, image.shape())?;
    let (img, img_bytes) = image.ready()?;
    let (sx, sx_bytes) = seeds_x.ready()?;
    let (sy, sy_bytes) = seeds_y.ready()?;
    let (seg_pin, seg_bytes) = seg.ready()?;
    let mut scratch = Scratch::new(ctx);
    let lab_bytes = u64::from(geom.n) * 4;
    let lab = scratch.take(lab_bytes)?;
    let mm_bytes = u64::from(geom.chunks) * 8;
    let mm = scratch.take(mm_bytes)?;
    let sums_bytes = u64::from(geom.chunks) * 12;
    let sums = scratch.take(sums_bytes)?;
    let ctrl = scratch.take(CTRL_BYTES)?;
    let params = geom.params([seed_count, radius, f32_bits(multiplier), geom.chunks]);
    let uniform = RawArg::Uniform(bytemuck::bytes_of(&params));
    let groups = fold_workgroups(ctx, geom.chunks)?;
    let seed_groups = fold_workgroups(ctx, seed_count.div_ceil(GROUP_SIZE))?;
    let pixel_groups = fold_workgroups(ctx, geom.n.div_ceil(GROUP_SIZE))?;
    let launch = |ctx: &Context, k: Kernel, workgroups: [u32; 3], size: u32, args: &[RawArg<'_>]| -> Outcome<()> {
        let kernel = ctx.builtin(Builtin::Vision(Any::Regions(super::Kernel::Confidence(k))), "f32")?;
        ctx.launch(&kernel, workgroups, [size, 1, 1], args)?;
        Ok(())
    };
    launch(ctx, Kernel::MinMax, groups, geom.n, &[RawArg::read(img.buffer(), img_bytes), RawArg::write(&mm, mm_bytes), uniform])?;
    let args = [RawArg::read(img.buffer(), img_bytes), RawArg::read(&mm, mm_bytes), RawArg::read(sx.buffer(), sx_bytes), RawArg::read(sy.buffer(), sy_bytes), RawArg::write(&ctrl, CTRL_BYTES), uniform];
    launch(ctx, Kernel::Setup, [1, 1, 1], GROUP_SIZE, &args)?;
    // The segmentation of an interval: components of the pixels in it, marked by the seeds.
    let segment = |ctx: &Context| -> Outcome<()> {
        let local = [RawArg::read(img.buffer(), img_bytes), RawArg::read(&ctrl, CTRL_BYTES), RawArg::write(&lab, lab_bytes), RawArg::write(seg_pin.buffer(), seg_bytes)];
        geom.local(ctx, label::Kernel::Local { eight: true, range: true }, &local, &params)?;
        geom.seams(ctx, label::Kernel::Seams { eight: true, gated: true }, &[RawArg::write(&lab, lab_bytes), RawArg::read(&ctrl, CTRL_BYTES)], &params)?;
        let args = [RawArg::write(&lab, lab_bytes), RawArg::write(seg_pin.buffer(), seg_bytes), RawArg::read(&ctrl, CTRL_BYTES), RawArg::read(sx.buffer(), sx_bytes), RawArg::read(sy.buffer(), sy_bytes), uniform];
        launch(ctx, Kernel::Mark, seed_groups, seed_count, &args)?;
        let args = [RawArg::write(&lab, lab_bytes), RawArg::write(seg_pin.buffer(), seg_bytes), RawArg::read(&ctrl, CTRL_BYTES), uniform];
        launch(ctx, Kernel::Out, pixel_groups, geom.n, &args)
    };
    segment(ctx)?;
    for _ in 0..iterations {
        let args = [RawArg::read(img.buffer(), img_bytes), RawArg::read(seg_pin.buffer(), seg_bytes), RawArg::read(&ctrl, CTRL_BYTES), RawArg::write(&sums, sums_bytes), uniform];
        launch(ctx, Kernel::Sums, groups, geom.n, &args)?;
        launch(ctx, Kernel::Stats, [1, 1, 1], GROUP_SIZE, &[RawArg::read(&sums, sums_bytes), RawArg::write(&ctrl, CTRL_BYTES), uniform])?;
        segment(ctx)?;
    }
    Ok(seg)
}

/// The multiplier as the bits of an `f32` (the uniform words are plain `u32`s).
fn f32_bits(multiplier: u32) -> u32 {
    (multiplier as f32).to_bits()
}
