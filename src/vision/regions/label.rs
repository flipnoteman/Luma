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
use crate::context::{RawArg, Scratch};
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
    Local { eight: bool },
    Seams { eight: bool },
    Flatten,
    Rename,
}

/// The union-find functions over the atomic array `a` (`find` with path halving, `union` hooks a
/// larger root under a smaller one).
fn union_find(s: &mut String, a: &str, suffix: &str) {
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

const PARAMS: &str = "struct Params { h: u32, w: u32, tiles_x: u32, tiles_y: u32, nv: u32, nh: u32, n: u32, chunks: u32 }\n";

pub(crate) fn source(k: Kernel) -> String {
    let mut s = String::from(PARAMS);
    match k {
        Kernel::Local { eight } => {
            s.push_str("@group(1) @binding(0) var<storage, read> src: array<u32>;\n@group(1) @binding(1) var<storage, read_write> lab: array<u32>;\n@group(1) @binding(2) var<uniform> p: Params;\n");
            let _ = writeln!(s, "var<workgroup> loc: array<atomic<u32>, {}>;\nvar<workgroup> fgm: array<u32, {}>;", TX * TY, TX * TY);
            union_find(&mut s, "loc", "l");
            let _ = write!(
                s,
                "@compute @workgroup_size({GROUP})
fn main(@builtin(local_invocation_index) t: u32, @builtin(workgroup_id) wid: vec3<u32>) {{
    let tx = t & {tx_mask}u;
    let ty = t >> {tx_bits}u;
    let i0 = wid.x * {TX}u + tx;
    let i1 = wid.y * {TY}u + ty;
    let inb = i0 < p.h && i1 < p.w;
    let idx = i0 + p.h * i1;
    var fg = 0u;
    if (inb) {{ fg = (src[idx >> 5u] >> (idx & 31u)) & 1u; }}
    fgm[t] = fg;
    atomicStore(&loc[t], t);
    workgroupBarrier();
    if (fg != 0u) {{
        if (tx > 0u && fgm[t - 1u] != 0u) {{ union_l(t, t - 1u); }}
        if (ty > 0u) {{
            if (fgm[t - {TX}u] != 0u) {{ union_l(t, t - {TX}u); }}
",
                tx_mask = TX - 1,
                tx_bits = TX.trailing_zeros(),
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
    }}
}}
",
                tx_mask = TX - 1,
                tx_bits = TX.trailing_zeros(),
            );
        }
        Kernel::Seams { eight } => {
            s.push_str("@group(1) @binding(0) var<storage, read_write> lab: array<atomic<u32>>;\n@group(1) @binding(1) var<uniform> p: Params;\n");
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
    let g = (wid.x + wid.y * nwg.x) * {GROUP}u + t;
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

/// The label image of the binary `image` (a plain `[H, W]` array).
pub(super) fn run(image: &Array<bool>, connectivity: Connectivity) -> Outcome<Array<u32>> {
    let ctx = image.context();
    let d = image.shape().dims();
    if d[2] != 1 || d[3] != 1 {
        return Err(invalid(format!("regions takes an [H, W] image, got shape {d:?}")));
    }
    let (h, w) = (d[0], d[1]);
    let n = h.checked_mul(w).filter(|&n| n < u32::MAX as usize).ok_or_else(|| invalid(format!("regions: {h} x {w} pixels do not fit 32-bit labels")))?;
    let eight = connectivity == Connectivity::Eight;
    image.storage()?;
    let out = Array::<u32>::alloc(ctx, image.shape())?;
    let (tiles_x, tiles_y) = (to_u32(h.div_ceil(TX as usize), "tiles")?, to_u32(w.div_ceil(TY as usize), "tiles")?);
    let grid = tile_grid(ctx, tiles_x, tiles_y)?;
    let n32 = n as u32;
    let chunks = n32.div_ceil(CHUNK);
    let nv = (tiles_x - 1) as u64 * w as u64;
    let nh = (tiles_y - 1) as u64 * h as u64;
    let seam_threads = u32::try_from(nv + nh).map_err(|_| invalid("regions: image too large"))?;
    let params = [h as u32, w as u32, tiles_x, tiles_y, nv as u32, nh as u32, n32, chunks];
    let counts = Array::<u32>::alloc(ctx, Shape::from(chunks as usize + 1))?;
    let (src, src_bytes) = image.ready()?;
    let (lab, lab_bytes) = out.ready()?;
    let counts_pin = counts.ready()?;
    let mut scratch = Scratch::new(ctx);
    let rank_bytes = u64::from(n32) * 4;
    let rank = scratch.take(rank_bytes)?;
    let uniform = RawArg::Uniform(bytemuck::bytes_of(&params));
    let kernel = |k: Kernel| ctx.builtin(Builtin::Vision(Any::Regions(super::Kernel::Label(k))), "u32");

    let k = kernel(Kernel::Local { eight })?;
    ctx.launch(&k, grid, [h as u32, w as u32, 1], &[RawArg::read(src.buffer(), src_bytes), RawArg::write(lab.buffer(), lab_bytes), uniform])?;
    if seam_threads > 0 {
        let groups = fold_workgroups(ctx, seam_threads.div_ceil(GROUP))?;
        let k = kernel(Kernel::Seams { eight })?;
        ctx.launch(&k, groups, [seam_threads, 1, 1], &[RawArg::write(lab.buffer(), lab_bytes), uniform])?;
    }
    let groups = fold_workgroups(ctx, chunks)?;
    let args = [RawArg::write(lab.buffer(), lab_bytes), RawArg::write(&rank, rank_bytes), RawArg::write(counts_pin.0.buffer(), counts_pin.1), uniform];
    let k = kernel(Kernel::Flatten)?;
    ctx.launch(&k, groups, [n32, 1, 1], &args)?;
    scan_counts(ctx, &counts_pin, chunks)?;
    let args = [RawArg::write(lab.buffer(), lab_bytes), RawArg::read(&rank, rank_bytes), RawArg::read(counts_pin.0.buffer(), counts_pin.1), uniform];
    let k = kernel(Kernel::Rename)?;
    ctx.launch(&k, groups, [n32, 1, 1], &args)?;
    Ok(out)
}
