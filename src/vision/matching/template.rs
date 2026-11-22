//! Template matching: the disparity of a template with the window of the image that starts at
//! every pixel.
//!
//! A workgroup of 32 x 8 invocations computes a tile of outputs, `RX` per invocation along axis
//! 0 (a warp's lanes step 32 pixels apart so shared-memory reads are contiguous) and `RY` along
//! axis 1. The tile's window of the image (the tile plus the template's extent, zero beyond the
//! image) is staged in workgroup memory with the template. An invocation walks the template: for
//! one template row `a` it slides down the columns `b`, keeping the `RY` image values its outputs
//! need in registers (the loop is unrolled `RY` times, so values rotate by renaming, not by
//! moving): one shared-memory read of image per output column and one broadcast read of the
//! template per `RX * RY` accumulations. A template too large for the workgroup memory is
//! processed in slabs (sub-rectangles staged one after another). The metrics that subtract the
//! window's mean first sum the window (the same walk, without the template) before.

use std::fmt::Write as _;

use crate::array::Array;
use crate::builtin::Builtin;
use crate::context::RawArg;
use crate::error::LumaError;
use crate::ops::Outcome;

use super::super::{invalid, to_u32, Kernel as Any};
use super::MatchType;

/// Invocations along axis 0 and 1 of a workgroup.
const GX: u32 = 32;
const GY: u32 = 8;
/// Words of the uniform parameters.
const PARAM_WORDS: usize = 12;

/// One kernel: the metric and the geometry of its tile and workgroup memory (all literals).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) struct Kernel {
    pub(crate) metric: MatchType,
    /// Outputs per invocation along axis 0 and 1.
    pub(crate) rx: u8,
    pub(crate) ry: u8,
    /// Words of the staged image tile and of the staged template slab.
    pub(crate) tile_words: u32,
    pub(crate) tpl_words: u32,
}

impl Kernel {
    fn ox(&self) -> u32 {
        GX * u32::from(self.rx)
    }
    fn oy(&self) -> u32 {
        GY * u32::from(self.ry)
    }
}

/// The metrics that need the mean of the window before they can sum.
fn needs_mean(m: MatchType) -> bool {
    matches!(m, MatchType::Zsad | MatchType::Lsad | MatchType::Zssd | MatchType::Lssd | MatchType::Zncc)
}

/// Whether the template is replaced by the template minus its mean.
fn zero_mean_template(m: MatchType) -> bool {
    matches!(m, MatchType::Zsad | MatchType::Zssd | MatchType::Zncc)
}

fn normalized(m: MatchType) -> bool {
    matches!(m, MatchType::Ncc | MatchType::Zncc)
}

/// The accumulation of image value `v` against template value `t`, the `i`th output of the thread
/// (`s{i}` is its window mean, or the mean ratio for the locally scaled metrics).
fn accumulate(m: MatchType, i: &str, v: &str, t: &str) -> String {
    match m {
        MatchType::Sad => format!("ac{i} += abs({v} - {t});"),
        MatchType::Zsad => format!("ac{i} += abs({v} - s{i} - {t});"),
        MatchType::Lsad => format!("ac{i} += abs(fma(-s{i}, {t}, {v}));"),
        MatchType::Ssd => format!("{{ let d = {v} - {t}; ac{i} = fma(d, d, ac{i}); }}"),
        MatchType::Zssd => format!("{{ let d = {v} - s{i} - {t}; ac{i} = fma(d, d, ac{i}); }}"),
        MatchType::Lssd => format!("{{ let d = fma(-s{i}, {t}, {v}); ac{i} = fma(d, d, ac{i}); }}"),
        MatchType::Ncc => format!("ac{i} = fma({v}, {t}, ac{i}); sq{i} = fma({v}, {v}, sq{i});"),
        MatchType::Zncc => format!("{{ let d = {v} - s{i}; ac{i} = fma(d, {t}, ac{i}); sq{i} = fma(d, d, sq{i}); }}"),
    }
}

/// Arguments: image, template (already minus its mean for the zero-mean metrics), `info` (word 0
/// the mean of the template, word 1 its sum of squares), the output, the uniform.
pub(crate) fn source(k: Kernel) -> String {
    let (rx, ry) = (u32::from(k.rx), u32::from(k.ry));
    let (ox, oy) = (k.ox(), k.oy());
    let mean = needs_mean(k.metric);
    let norm = normalized(k.metric);
    let ids: Vec<(u32, u32)> = (0..rx).flat_map(|x| (0..ry).map(move |y| (x, y))).collect();
    let name = |x: u32, y: u32| format!("{x}_{y}");

    let mut s = String::new();
    s.push_str("struct P { h: u32, w: u32, th: u32, tw: u32, sa: u32, sb: u32, slabs: u32, ts: u32, nf: f32, a: u32, b: u32, c: u32 }\n");
    s.push_str("@group(1) @binding(0) var<storage, read> src: array<f32>;\n");
    s.push_str("@group(1) @binding(1) var<storage, read> tmpl: array<f32>;\n");
    s.push_str("@group(1) @binding(2) var<storage, read> info: array<f32>;\n");
    s.push_str("@group(1) @binding(3) var<storage, read_write> out: array<f32>;\n");
    s.push_str("@group(1) @binding(4) var<uniform> p: P;\n");
    let _ = writeln!(s, "var<workgroup> tile: array<f32, {}>;", k.tile_words);
    let _ = writeln!(s, "var<workgroup> tpl: array<f32, {}>;", k.tpl_words);
    // Stages the slab at (sa0, sb0) of the template, of ha x hb, and the image around the tile.
    let _ = write!(
        s,
        "fn load_slab(t: u32, x0: u32, y0: u32, base: u32, sa0: u32, sb0: u32, ha: u32, hb: u32) {{
    let cols = {ox}u + ha - 1u;
    let total = cols * ({oy}u + hb - 1u);
    for (var i = t; i < total; i += {inv}u) {{
        let cx = i % cols;
        let cy = i / cols;
        let gx = x0 + sa0 + cx;
        let gy = y0 + sb0 + cy;
        var v = 0.0;
        if (gx < p.h && gy < p.w) {{ v = src[base + gx + p.h * gy]; }}
        tile[cx + p.ts * cy] = v;
    }}
    for (var i = t; i < ha * hb; i += {inv}u) {{
        tpl[i] = tmpl[sa0 + i % ha + p.th * (sb0 + i / ha)];
    }}
}}
",
        inv = GX * GY
    );
    let _ = write!(
        s,
        "@compute @workgroup_size({GX}, {GY})
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {{
    let t = lid.y * {GX}u + lid.x;
    let x0 = wid.x * {ox}u;
    let y0 = wid.y * {oy}u;
    let base = wid.z * p.h * p.w;
    let yb = lid.y * {ry}u;
    let tm = info[0];
"
    );
    for &(x, y) in &ids {
        let n = name(x, y);
        let _ = writeln!(s, "    var ac{n} = 0.0;");
        if norm {
            let _ = writeln!(s, "    var sq{n} = 0.0;");
        }
        if mean {
            let _ = writeln!(s, "    var s{n} = 0.0;");
        }
    }
    s.push_str("    if (p.slabs == 1u) {\n        load_slab(t, x0, y0, base, 0u, 0u, p.th, p.tw);\n        workgroupBarrier();\n    }\n");
    // One pass over all the slabs: `sums` adds the window values into `s*`, otherwise the metric.
    let pass = |s: &mut String, sums: bool| {
        s.push_str("    for (var sa0 = 0u; sa0 < p.th; sa0 += p.sa) {\n        let ha = min(p.sa, p.th - sa0);\n        for (var sb0 = 0u; sb0 < p.tw; sb0 += p.sb) {\n            let hb = min(p.sb, p.tw - sb0);\n");
        s.push_str("            if (p.slabs != 1u) {\n                workgroupBarrier();\n                load_slab(t, x0, y0, base, sa0, sb0, ha, hb);\n                workgroupBarrier();\n            }\n");
        s.push_str("            for (var a = 0u; a < ha; a++) {\n                let rowa = lid.x + a + p.ts * yb;\n");
        // The first RY - 1 rows of the window.
        for x in 0..rx {
            for r in 0..ry - 1 {
                let _ = writeln!(s, "                var v{x}_{r} = tile[rowa + {}u + p.ts * {r}u];", GX * x);
            }
            let _ = writeln!(s, "                var v{x}_{} = 0.0;", ry - 1);
        }
        s.push_str("                for (var b = 0u; b < hb; b += {ry}u) {\n".replace("{ry}", &ry.to_string()).as_str());
        for j in 0..ry {
            let _ = writeln!(s, "                    if (b + {j}u < hb) {{");
            let slot_new = (j + ry - 1) % ry;
            for x in 0..rx {
                let _ = writeln!(s, "                        v{x}_{slot_new} = tile[rowa + {}u + p.ts * (b + {}u)];", GX * x, j + ry - 1);
            }
            if !sums {
                let _ = writeln!(s, "                        let tv = tpl[(b + {j}u) * ha + a];");
            }
            for &(x, y) in &ids {
                let slot = (j + y) % ry;
                let v = format!("v{x}_{slot}");
                let i = name(x, y);
                if sums {
                    let _ = writeln!(s, "                        s{i} += {v};");
                } else {
                    let _ = writeln!(s, "                        {}", accumulate(k.metric, &i, &v, "tv"));
                }
            }
            s.push_str("                    }\n");
        }
        s.push_str("                }\n            }\n        }\n    }\n");
    };
    if mean {
        pass(&mut s, true);
        for &(x, y) in &ids {
            let i = name(x, y);
            // The mean of the zero-padded window; the locally scaled metrics scale the template by its ratio to the template's mean.
            if matches!(k.metric, MatchType::Lsad | MatchType::Lssd) {
                let _ = writeln!(s, "    s{i} = s{i} / p.nf / tm;");
            } else {
                let _ = writeln!(s, "    s{i} = s{i} / p.nf;");
            }
        }
    }
    pass(&mut s, false);
    // The outputs.
    if norm {
        s.push_str("    let t2 = info[1];\n");
    }
    for &(x, y) in &ids {
        let i = name(x, y);
        let _ = writeln!(s, "    {{\n        let gx = x0 + lid.x + {}u;\n        let gy = y0 + yb + {y}u;\n        if (gx < p.h && gy < p.w) {{", GX * x);
        if norm {
            let _ = writeln!(s, "            let den = sqrt(sq{i} * t2);\n            out[base + gx + p.h * gy] = select(0.0, ac{i} / den, den > 0.0);");
        } else {
            let _ = writeln!(s, "            out[base + gx + p.h * gy] = ac{i};");
        }
        s.push_str("        }\n    }\n");
    }
    s.push_str("}\n");
    s
}

/// The tile geometry for a device with `bytes` of workgroup memory.
fn geometry(metric: MatchType, bytes: u32) -> Kernel {
    let words = bytes / 4;
    // tile + template slab (a quarter of the rest, at most 1536 words).
    if words >= 12288 {
        Kernel { metric, rx: 4, ry: 4, tile_words: 10240, tpl_words: 1536 }
    } else if words >= 8192 {
        Kernel { metric, rx: 4, ry: 4, tile_words: 6144, tpl_words: 1024 }
    } else {
        Kernel { metric, rx: 2, ry: 4, tile_words: 3072, tpl_words: 1024 }
    }
}

/// The slab of the template (`sa` rows by `sb` columns) that staged costs the fewest words in all.
fn slab(k: &Kernel, th: u32, tw: u32) -> Option<(u32, u32)> {
    let (ox, oy) = (k.ox(), k.oy());
    let mut best: Option<(u64, u32, u32)> = None;
    for sa in 1..=th.min(k.tpl_words) {
        let rows_fit = k.tile_words / (ox + sa - 1);
        if rows_fit < oy {
            break;
        }
        let sb = tw.min(k.tpl_words / sa).min(rows_fit - (oy - 1));
        if sb == 0 {
            continue;
        }
        let slabs = u64::from(th.div_ceil(sa)) * u64::from(tw.div_ceil(sb));
        let cost = slabs * u64::from(ox + sa - 1) * u64::from(oy + sb - 1);
        if best.is_none_or(|(c, ..)| cost < c) {
            best = Some((cost, sa, sb));
        }
    }
    best.map(|(_, sa, sb)| (sa, sb))
}

pub(crate) fn run(search: &Array<f32>, template: &Array<f32>, metric: MatchType) -> Outcome<Array<f32>> {
    let ctx = search.context();
    if !ctx.same_as(template.context()) {
        return Err(LumaError::ContextMismatch.into());
    }
    let td = template.shape().dims();
    if td[2] != 1 || td[3] != 1 {
        return Err(invalid(format!("match_template takes an [H, W] template, got shape {td:?}")));
    }
    let sd = search.shape().dims();
    let (h, w, th, tw) = (to_u32(sd[0], "height")?, to_u32(sd[1], "width")?, to_u32(td[0], "template height")?, to_u32(td[1], "template width")?);
    let planes = to_u32(sd[2] * sd[3], "batch")?;
    let kernel = geometry(metric, ctx.tuning().workgroup_storage);
    let (sa, sb) = slab(&kernel, th, tw).ok_or_else(|| LumaError::Unsupported(format!("a template of {th} x {tw} does not fit workgroup memory")))?;
    let (tiles_x, tiles_y) = (h.div_ceil(kernel.ox()), w.div_ceil(kernel.oy()));
    let max = ctx.max_workgroups_per_dimension();
    if tiles_x > max || tiles_y > max || planes > max {
        return Err(LumaError::KernelGridTooLarge { workgroups: [tiles_x, tiles_y, planes], max }.into());
    }
    search.storage()?;
    // The template as the kernel reads it, and `info`: its mean and sum of squares.
    let wants_info = needs_mean(metric) || normalized(metric);
    let (tpl, info) = if wants_info {
        let mean = template.mean();
        let tpl = if zero_mean_template(metric) { template - &mean } else { template.clone() };
        let squares = if normalized(metric) { (&tpl * &tpl).sum() } else { mean.clone() };
        let info = crate::view::concat(0, [&mean, &squares]);
        (tpl, info)
    } else {
        (template.clone(), template.clone())
    };
    let out = Array::<f32>::alloc(ctx, search.shape())?;
    let (src_pin, src_bytes) = search.ready()?;
    let (tpl_pin, tpl_bytes) = tpl.ready()?;
    let (info_pin, info_bytes) = info.ready()?;
    let (out_pin, out_bytes) = out.ready()?;
    let slabs = th.div_ceil(sa) * tw.div_ceil(sb);
    let mut params = [0u32; PARAM_WORDS];
    params[..9].copy_from_slice(&[h, w, th, tw, sa, sb, slabs, kernel.ox() + sa - 1, ((th * tw) as f32).to_bits()]);
    let kernel_id = ctx.builtin(Builtin::Vision(Any::Matching(super::Kernel::Template(kernel))), "f32")?;
    let args = [
        RawArg::read(src_pin.buffer(), src_bytes),
        RawArg::read(tpl_pin.buffer(), tpl_bytes),
        RawArg::read(info_pin.buffer(), info_bytes),
        RawArg::write(out_pin.buffer(), out_bytes),
        RawArg::Uniform(bytemuck::bytes_of(&params)),
    ];
    ctx.launch(&kernel_id, [tiles_x, tiles_y, planes], [h, w, planes], &args)?;
    Ok(out)
}
