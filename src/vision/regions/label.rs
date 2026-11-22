//! Connected-component labelling (`Array<bool>::regions`).
//!
//! Lock-free union-find over the pixels, in the order of the flat index (axis 0 fastest): a set's
//! root is always its smallest pixel, so the roots in ascending order are the components in
//! ascending order of their first pixel, which is the numbering ArrayFire's backends produce
//! (labels 1..=N). Six launches:
//!
//! 1. `Local`: a workgroup labels a tile of 32 x 8 pixels in workgroup memory (union of each
//!    foreground pixel with its backward neighbours in the tile, then `find`) and stores, for every
//!    pixel, the global index of the tile-local root (`NONE` for the background).
//! 2. `Seams`: one thread per pixel on a tile boundary unions the roots of the two sides in global
//!    memory (`atomicMin` hooks the larger root under the smaller one).
//! 3. `Flatten`: every pixel's entry becomes the final root; a workgroup of 256 threads takes 1024
//!    consecutive pixels, counts their roots, and gives every root its rank inside the chunk.
//! 4. the exclusive scan of the chunk counts ([`scan_counts`](super::super::scan_counts)).
//! 5. `Rename`: the entry of a pixel becomes `offset(chunk of its root) + rank(root) + 1`.

use std::fmt::Write as _;

use crate::array::Array;
use crate::builtin::Builtin;
use crate::context::{Context, RawArg, Scratch};
use crate::ops::{fold_workgroups, Outcome};
use crate::shape::Shape;
use crate::vision::{invalid, scan_counts, tile_grid, to_u32, Kernel as Any};

use super::Connectivity;

/// Tile of the local pass along axis 0 and axis 1 (256 invocations).
const TX: u32 = 32;
const TY: u32 = 8;
/// Pixels of a chunk of the flatten and rename passes: 4 per invocation.
const CHUNK: u32 = 1024;
const GROUP: u32 = 256;
const NONE: &str = "0xFFFFFFFFu";

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum Kernel {
    /// Local pass over the bits of a boolean image, or over the pixels of an `f32` image whose
    /// value (normalised by the control words of [`confidence`](super::confidence)) is in range.
    Local { eight: bool, range: bool },
    /// `gated` kernels do nothing once the control words of the confidence loop say it is done.
    Seams { eight: bool, gated: bool },
    Flatten,
    Rename,
}

/// The union-find functions over the atomic array `a` (`find` with path halving, `union` hooks a
/// larger root under a smaller one).
pub(super) fn union_find(s: &mut String, a: &str, suffix: &str) {
    let _ = write!(
        s,
        "fn find_{suffix}(x0: u32) -> u32 {{
    var x = x0;
    loop {{
        let p = atomicLoad(&{a}[x]);
        if (p == x) {{ break; }}
        let g = atomicLoad(&{a}[p]);
        if (g != p) {{ atomicMin(&{a}[x], g); }}
        x = g;
    }}
    return x;
}}
fn union_{suffix}(a0: u32, b0: u32) {{
    var a = find_{suffix}(a0);
    var b = find_{suffix}(b0);
    loop {{
        if (a == b) {{ return; }}
        let hi = max(a, b);
        let lo = min(a, b);
        let old = atomicMin(&{a}[hi], lo);
        if (old == hi) {{ return; }}
        a = find_{suffix}(old);
        b = find_{suffix}(lo);
    }}
}}
"
    );
}

/// The uniform of every kernel of this module (and of [`confidence`](super::confidence)); `h`
/// and `w` are the sizes along axis 0 and 1, `nv`/`nh` the seam pixels of [`Kernel::Seams`], `n`
/// the pixel count, `chunks` the 1024-pixel chunks, the rest belongs to the confidence loop.
pub(super) const PARAMS: &str = "struct Params { h: u32, w: u32, tiles_x: u32, tiles_y: u32, nv: u32, nh: u32, n: u32, chunks: u32, ns: u32, radius: u32, mult: u32, groups: u32 }\n";
pub(super) const PARAM_WORDS: usize = 12;
pub(super) const NONE_WORD: &str = NONE;
pub(super) const GROUP_SIZE: u32 = GROUP;
pub(super) const CHUNK_PIXELS: u32 = CHUNK;

pub(crate) fn source(k: Kernel) -> String {
    let mut s = String::from(PARAMS);
    match k {
        Kernel::Local { eight, range } => {
            // Bindings: the image (bits, or `f32` pixels with the control words), the labels (and
            // for the range variant the segmentation, cleared here), the uniform.
            let (tx_mask, tx_bits) = (TX - 1, TX.trailing_zeros());
            if range {
                s.push_str("@group(1) @binding(0) var<storage, read> src: array<f32>;\n@group(1) @binding(1) var<storage, read> ctrl: array<u32>;\n@group(1) @binding(2) var<storage, read_write> lab: array<u32>;\n@group(1) @binding(3) var<storage, read_write> seg: array<u32>;\n@group(1) @binding(4) var<uniform> p: Params;\nvar<workgroup> stop: u32;\n");
                s.push_str(&super::confidence::norm_source());
            } else {
                s.push_str("@group(1) @binding(0) var<storage, read> src: array<u32>;\n@group(1) @binding(1) var<storage, read_write> lab: array<u32>;\n@group(1) @binding(2) var<uniform> p: Params;\n");
            }
            let _ = writeln!(s, "var<workgroup> loc: array<atomic<u32>, {}>;\nvar<workgroup> fgm: array<u32, {}>;", TX * TY, TX * TY);
            union_find(&mut s, "loc", "l");
            let _ = write!(
                s,
                "@compute @workgroup_size({GROUP})
fn main(@builtin(local_invocation_index) t: u32, @builtin(workgroup_id) wid: vec3<u32>) {{
"
            );
            if range {
                s.push_str("    if (t == 0u) { stop = ctrl[7]; }\n    workgroupBarrier();\n    if (workgroupUniformLoad(&stop) != 0u) { return; }\n");
            }
            let _ = write!(
                s,
                "    let tx = t & {tx_mask}u;
    let ty = t >> {tx_bits}u;
    let i0 = wid.x * {TX}u + tx;
    let i1 = wid.y * {TY}u + ty;
    let inb = i0 < p.h && i1 < p.w;
    let idx = i0 + p.h * i1;
    var fg = 0u;
"
            );
            if range {
                s.push_str("    if (inb) { let v = norm(src[idx]); fg = select(0u, 1u, v >= bitcast<f32>(ctrl[0]) && v <= bitcast<f32>(ctrl[1])); }\n");
            } else {
                s.push_str("    if (inb) { fg = (src[idx >> 5u] >> (idx & 31u)) & 1u; }\n");
            }
            let _ = write!(
                s,
                "    fgm[t] = fg;
    atomicStore(&loc[t], t);
    workgroupBarrier();
    if (fg != 0u) {{
        if (tx > 0u && fgm[t - 1u] != 0u) {{ union_l(t, t - 1u); }}
        if (ty > 0u) {{
            if (fgm[t - {TX}u] != 0u) {{ union_l(t, t - {TX}u); }}
"
            );
            if eight {
                let _ = write!(
                    s,
                    "            if (tx > 0u && fgm[t - {a}u] != 0u) {{ union_l(t, t - {a}u); }}
            if (tx < {m}u && fgm[t - {b}u] != 0u) {{ union_l(t, t - {b}u); }}
",
                    a = TX + 1,
                    b = TX - 1,
                    m = TX - 1
                );
            }
            let _ = write!(
                s,
                "        }}
    }}
    workgroupBarrier();
    if (inb) {{
        if (fg != 0u) {{
            let r = find_l(t);
            lab[idx] = (wid.x * {TX}u + (r & {tx_mask}u)) + p.h * (wid.y * {TY}u + (r >> {tx_bits}u));
        }} else {{
            lab[idx] = {NONE};
        }}
"
            );
            if range {
                s.push_str("        seg[idx] = 0u;\n");
            }
            s.push_str("    }\n}\n");
        }
        Kernel::Seams { eight, gated } => {
            if gated {
                s.push_str("@group(1) @binding(0) var<storage, read_write> lab: array<atomic<u32>>;\n@group(1) @binding(1) var<storage, read> ctrl: array<u32>;\n@group(1) @binding(2) var<uniform> p: Params;\n");
            } else {
                s.push_str("@group(1) @binding(0) var<storage, read_write> lab: array<atomic<u32>>;\n@group(1) @binding(1) var<uniform> p: Params;\n");
            }
            union_find(&mut s, "lab", "g");
            let _ = write!(
                s,
                "fn seam(a: u32, b: u32) {{
    let x = atomicLoad(&lab[a]);
    let y = atomicLoad(&lab[b]);
    if (x != {NONE} && y != {NONE} && x != y) {{ union_g(x, y); }}
}}
@compute @workgroup_size({GROUP})
fn main(@builtin(local_invocation_index) t: u32, @builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {{
"
            );
            if gated {
                s.push_str("    if (ctrl[7] != 0u) { return; }\n");
            }
            let _ = write!(
                s,
                "    let g = (wid.x + wid.y * nwg.x) * {GROUP}u + t;
    if (g < p.nv) {{
        let i0 = (g / p.w + 1u) * {TX}u;
        let i1 = g % p.w;
        let a = i0 + p.h * i1;
        if (atomicLoad(&lab[a]) == {NONE}) {{ return; }}
        seam(a, a - 1u);
"
            );
            if eight {
                s.push_str("        if (i1 > 0u) { seam(a, a - 1u - p.h); }\n        if (i1 + 1u < p.w) { seam(a, a - 1u + p.h); }\n");
            }
            let _ = write!(
                s,
                "    }} else if (g < p.nv + p.nh) {{
        let g2 = g - p.nv;
        let i1 = (g2 / p.h + 1u) * {TY}u;
        let i0 = g2 % p.h;
        let a = i0 + p.h * i1;
        if (atomicLoad(&lab[a]) == {NONE}) {{ return; }}
        seam(a, a - p.h);
"
            );
            if eight {
                let _ = writeln!(
                    s,
                    "        if (i0 > 0u && (i0 & {m}u) != 0u) {{ seam(a, a - p.h - 1u); }}\n        if (i0 + 1u < p.h && ((i0 + 1u) & {m}u) != 0u) {{ seam(a, a - p.h + 1u); }}",
                    m = TX - 1
                );
            }
            s.push_str("    }\n}\n");
        }
        Kernel::Flatten => {
            s.push_str(
                "@group(1) @binding(0) var<storage, read_write> lab: array<atomic<u32>>;\n@group(1) @binding(1) var<storage, read_write> rank: array<u32>;\n@group(1) @binding(2) var<storage, read_write> counts: array<u32>;\n@group(1) @binding(3) var<uniform> p: Params;\n",
            );
            union_find(&mut s, "lab", "g");
            let _ = write!(
                s,
                "var<workgroup> sums: array<u32, {GROUP}>;
@compute @workgroup_size({GROUP})
fn main(@builtin(local_invocation_index) t: u32, @builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {{
    let chunk = wid.x + wid.y * nwg.x;
    let base = chunk * {CHUNK}u + t * 4u;
    var root = 0u;
    for (var j = 0u; j < 4u; j++) {{
        let q = base + j;
        if (q < p.n) {{
            let x = atomicLoad(&lab[q]);
            if (x != {NONE}) {{
                let r = find_g(x);
                if (r != x) {{ atomicStore(&lab[q], r); }}
                if (r == q) {{ root |= 1u << j; }}
            }}
        }}
    }}
    let c = countOneBits(root);
    sums[t] = c;
    workgroupBarrier();
    for (var d = 1u; d < {GROUP}u; d = d << 1u) {{
        var add = 0u;
        if (t >= d) {{ add = sums[t - d]; }}
        workgroupBarrier();
        sums[t] = sums[t] + add;
        workgroupBarrier();
    }}
    var r = sums[t] - c;
    for (var j = 0u; j < 4u; j++) {{
        if ((root & (1u << j)) != 0u) {{ rank[base + j] = r; r++; }}
    }}
    if (t == {last}u) {{ counts[chunk] = sums[t]; }}
}}
",
                last = GROUP - 1
            );
        }
        Kernel::Rename => {
            s.push_str(
                "@group(1) @binding(0) var<storage, read_write> lab: array<u32>;\n@group(1) @binding(1) var<storage, read> rank: array<u32>;\n@group(1) @binding(2) var<storage, read> offs: array<u32>;\n@group(1) @binding(3) var<uniform> p: Params;\n",
            );
            let _ = write!(
                s,
                "@compute @workgroup_size({GROUP})
fn main(@builtin(local_invocation_index) t: u32, @builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {{
    let chunk = wid.x + wid.y * nwg.x;
    let base = chunk * {CHUNK}u + t * 4u;
    for (var j = 0u; j < 4u; j++) {{
        let q = base + j;
        if (q < p.n) {{
            let x = lab[q];
            if (x == {NONE}) {{ lab[q] = 0u; }} else {{ lab[q] = offs[x / {CHUNK}u] + rank[x] + 1u; }}
        }}
    }}
}}
"
            );
        }
    }
    s
}

/// The sizes of the launches over an `[H, W]` image.
pub(super) struct Geom {
    pub(super) h: u32,
    pub(super) w: u32,
    pub(super) n: u32,
    pub(super) tiles_x: u32,
    pub(super) tiles_y: u32,
    pub(super) seam_threads: u32,
    pub(super) nv: u32,
    pub(super) nh: u32,
    pub(super) chunks: u32,
}

impl Geom {
    pub(super) fn new(op: &str, shape: Shape) -> Outcome<Geom> {
        let d = shape.dims();
        if d[2] != 1 || d[3] != 1 {
            return Err(invalid(format!("{op} takes an [H, W] image, got shape {d:?}")));
        }
        let (h, w) = (d[0], d[1]);
        let n = h.checked_mul(w).filter(|&n| n < u32::MAX as usize).ok_or_else(|| invalid(format!("{op}: {h} x {w} pixels do not fit 32-bit labels")))?;
        let (tiles_x, tiles_y) = (to_u32(h.div_ceil(TX as usize), "tiles")?, to_u32(w.div_ceil(TY as usize), "tiles")?);
        let nv = (tiles_x - 1) as u64 * w as u64;
        let nh = (tiles_y - 1) as u64 * h as u64;
        let seam_threads = u32::try_from(nv + nh).map_err(|_| invalid(format!("{op}: image too large")))?;
        Ok(Geom { h: h as u32, w: w as u32, n: n as u32, tiles_x, tiles_y, seam_threads, nv: nv as u32, nh: nh as u32, chunks: (n as u32).div_ceil(CHUNK) })
    }

    /// The uniform words; `rest` are the last four (seeds, radius, multiplier, groups).
    pub(super) fn params(&self, rest: [u32; 4]) -> [u32; PARAM_WORDS] {
        [self.h, self.w, self.tiles_x, self.tiles_y, self.nv, self.nh, self.n, self.chunks, rest[0], rest[1], rest[2], rest[3]]
    }

    /// Records the local pass (`args` without the uniform, which is appended).
    pub(super) fn local(&self, ctx: &Context, k: Kernel, args: &[RawArg<'_>], params: &[u32; PARAM_WORDS]) -> Outcome<()> {
        let grid = tile_grid(ctx, self.tiles_x, self.tiles_y)?;
        let mut all = args.to_vec();
        all.push(RawArg::Uniform(bytemuck::bytes_of(params)));
        let kernel = ctx.builtin(Builtin::Vision(Any::Regions(super::Kernel::Label(k))), "u32")?;
        ctx.launch(&kernel, grid, [self.h, self.w, 1], &all)?;
        Ok(())
    }

    /// Records the seam pass (nothing for an image of a single tile).
    pub(super) fn seams(&self, ctx: &Context, k: Kernel, args: &[RawArg<'_>], params: &[u32; PARAM_WORDS]) -> Outcome<()> {
        if self.seam_threads == 0 {
            return Ok(());
        }
        let groups = fold_workgroups(ctx, self.seam_threads.div_ceil(GROUP))?;
        let mut all = args.to_vec();
        all.push(RawArg::Uniform(bytemuck::bytes_of(params)));
        let kernel = ctx.builtin(Builtin::Vision(Any::Regions(super::Kernel::Label(k))), "u32")?;
        ctx.launch(&kernel, groups, [self.seam_threads, 1, 1], &all)?;
        Ok(())
    }
}

/// The label image of the binary `image` (a plain `[H, W]` array).
pub(super) fn run(image: &Array<bool>, connectivity: Connectivity) -> Outcome<Array<u32>> {
    let ctx = image.context();
    let geom = Geom::new("regions", image.shape())?;
    let eight = connectivity == Connectivity::Eight;
    image.storage()?;
    let out = Array::<u32>::alloc(ctx, image.shape())?;
    let counts = Array::<u32>::alloc(ctx, Shape::from(geom.chunks as usize + 1))?;
    let (src, src_bytes) = image.ready()?;
    let (lab, lab_bytes) = out.ready()?;
    let counts_pin = counts.ready()?;
    let mut scratch = Scratch::new(ctx);
    let rank_bytes = u64::from(geom.n) * 4;
    let rank = scratch.take(rank_bytes)?;
    let params = geom.params([0; 4]);
    let uniform = RawArg::Uniform(bytemuck::bytes_of(&params));
    let kernel = |k: Kernel| ctx.builtin(Builtin::Vision(Any::Regions(super::Kernel::Label(k))), "u32");

    geom.local(ctx, Kernel::Local { eight, range: false }, &[RawArg::read(src.buffer(), src_bytes), RawArg::write(lab.buffer(), lab_bytes)], &params)?;
    geom.seams(ctx, Kernel::Seams { eight, gated: false }, &[RawArg::write(lab.buffer(), lab_bytes)], &params)?;
    let groups = fold_workgroups(ctx, geom.chunks)?;
    let args = [RawArg::write(lab.buffer(), lab_bytes), RawArg::write(&rank, rank_bytes), RawArg::write(counts_pin.0.buffer(), counts_pin.1), uniform];
    let k = kernel(Kernel::Flatten)?;
    ctx.launch(&k, groups, [geom.n, 1, 1], &args)?;
    scan_counts(ctx, &counts_pin, geom.chunks)?;
    let args = [RawArg::write(lab.buffer(), lab_bytes), RawArg::read(&rank, rank_bytes), RawArg::read(counts_pin.0.buffer(), counts_pin.1), uniform];
    let k = kernel(Kernel::Rename)?;
    ctx.launch(&k, groups, [geom.n, 1, 1], &args)?;
    Ok(out)
}
