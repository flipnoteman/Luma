//! What the corner detectors share: the tile geometry, the end of a detection kernel (deciding
//! the strict local maxima of the response tile and listing them per tile), the gather kernel
//! that compacts the lists, and the host driver that records the three launches.
//!
//! A detection kernel's workgroup owns a tile of `tx x ty` decision pixels (axis 0 first, one
//! pixel per invocation), writes the number of corners of its tile to `counts[tile]` and their
//! local pixel indices and scores to the tile's slots (`slot` entries a tile: two strict maxima
//! are never neighbours, so a tile holds at most `ceil(tx / 2) * ceil(ty / 2)`). Every barrier
//! is in `main`.

use std::fmt::Write as _;

use super::Kernel as Corners;
use crate::array::{Array, Failure};
use crate::builtin::Builtin;
use crate::context::{RawArg, Scratch};
use crate::error::LumaError;
use crate::ops::Outcome;
use crate::shape::Shape;
use crate::vision::{scan_counts, tile_grid, to_u32, Kernel as Any};

/// Pixels of a tile along axis 0 and axis 1.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) struct Tile {
    pub(crate) tx: u8,
    pub(crate) ty: u8,
}

impl Tile {
    pub(crate) fn threads(self) -> u32 {
        u32::from(self.tx) * u32::from(self.ty)
    }

    /// Corners a tile can hold.
    pub(crate) fn slots(self) -> u32 {
        u32::from(self.tx).div_ceil(2) * u32::from(self.ty).div_ceil(2)
    }
}

/// The tiles to try, largest first.
const TILES: [(u8, u8); 12] = [(32, 8), (32, 4), (16, 8), (16, 4), (32, 2), (8, 8), (16, 2), (8, 4), (32, 1), (16, 1), (8, 2), (8, 1)];

/// The first tile whose kernel needs at most `limit` bytes of workgroup memory.
pub(crate) fn plan_tile(limit: u32, bytes: impl Fn(Tile) -> u64, what: &str) -> Result<Tile, Failure> {
    TILES
        .iter()
        .map(|&(tx, ty)| Tile { tx, ty })
        .find(|&tile| bytes(tile) <= u64::from(limit))
        .ok_or_else(|| LumaError::Unsupported(format!("{what} does not fit workgroup memory")).into())
}

/// The part of a detection kernel after the response tile `rr` (`(tx + 2) x (ty + 2)` cells with
/// row length `nw0`, zero outside the image's valid region) is complete: the strict local
/// maxima, the per-tile count and the slots. Expects in scope: `t`, `tile_id`, `o0`, `o1` (the
/// tile's first pixel), `h`, `w`, `lo` (all `i32`, the first decision pixel is `lo`), the
/// bindings `counts`, `slot_idx`, `slot_val`, and the module-scope `cnt`, `tot`, `rank`. `extra`
/// is an extra condition on the response `s0`.
pub(crate) fn epilogue(tile: Tile, nw0: u32, extra: &str) -> String {
    let (tx, nth, slots) = (tile.tx, tile.threads(), tile.slots());
    let mut s = String::new();
    let _ = write!(
        s,
        "    let tx = t % {tx}u;
    let ty = t / {tx}u;
    let c = (ty + 1u) * {nw0}u + tx + 1u;
    let s0 = rr[c];
    let base = ty * {nw0}u + tx;
    let keep = o0 + i32(tx) < h - lo && o1 + i32(ty) < w - lo && s0 > 0.0{extra}
        && s0 > rr[base] && s0 > rr[base + 1u] && s0 > rr[base + 2u]
        && s0 > rr[base + {nw0}u] && s0 > rr[base + {nw0}u + 2u]
        && s0 > rr[base + 2u * {nw0}u] && s0 > rr[base + 2u * {nw0}u + 1u] && s0 > rr[base + 2u * {nw0}u + 2u];
    if (keep) {{ atomicAdd(&cnt, 1u); }}
    workgroupBarrier();
    if (t == 0u) {{ tot = atomicLoad(&cnt); counts[tile_id] = tot; }}
    workgroupBarrier();
    if (workgroupUniformLoad(&tot) == 0u) {{ return; }}
    rank[t] = select(0u, 1u, keep);
    workgroupBarrier();
    for (var d = 1u; d < {nth}u; d = d << 1u) {{
        var add = 0u;
        if (t >= d) {{ add = rank[t - d]; }}
        workgroupBarrier();
        rank[t] = rank[t] + add;
        workgroupBarrier();
    }}
    if (keep) {{
        let k = tile_id * {slots}u + rank[t] - 1u;
        slot_idx[k] = t;
        slot_val[k] = s0;
    }}
}}
"
    );
    s
}

/// The shared module-scope declarations of a detection kernel (bindings 0 to 3 and the
/// counters); the detector adds its uniform as binding 4 and its workgroup arrays.
pub(crate) fn declarations(tile: Tile) -> String {
    let nth = tile.threads();
    format!(
        "@group(1) @binding(0) var<storage, read> src: array<f32>;
@group(1) @binding(1) var<storage, read_write> counts: array<u32>;
@group(1) @binding(2) var<storage, read_write> slot_idx: array<u32>;
@group(1) @binding(3) var<storage, read_write> slot_val: array<f32>;
var<workgroup> cnt: atomic<u32>;
var<workgroup> tot: u32;
var<workgroup> rank: array<u32, {nth}>;
"
    )
}

/// The gather kernel: `key` also writes the position in memory order (axis 0 fastest).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) struct Gather {
    pub(crate) key: bool,
}

/// Arguments: offsets (the scanned counts), slot indices, slot scores, `x`, `y`, `score`, then
/// the keys if `key`, and a uniform `{tiles_x, tx, slot, lo, cap, h, swap, ty}`.
pub(crate) fn gather_source(k: Gather) -> String {
    let mut s = String::new();
    s.push_str("struct Params { tiles_x: u32, tx: u32, slot: u32, lo: u32, cap: u32, h: u32, swap: u32, tile_y: u32 }\n");
    s.push_str("@group(1) @binding(0) var<storage, read> offs: array<u32>;\n");
    s.push_str("@group(1) @binding(1) var<storage, read> slot_idx: array<u32>;\n");
    s.push_str("@group(1) @binding(2) var<storage, read> slot_val: array<f32>;\n");
    s.push_str("@group(1) @binding(3) var<storage, read_write> out_x: array<f32>;\n");
    s.push_str("@group(1) @binding(4) var<storage, read_write> out_y: array<f32>;\n");
    s.push_str("@group(1) @binding(5) var<storage, read_write> out_s: array<f32>;\n");
    let uniform = if k.key {
        s.push_str("@group(1) @binding(6) var<storage, read_write> out_k: array<u32>;\n");
        7
    } else {
        6
    };
    let _ = writeln!(s, "@group(1) @binding({uniform}) var<uniform> p: Params;");
    s.push_str(
        "@compute @workgroup_size(64)
fn main(@builtin(local_invocation_index) t: u32, @builtin(workgroup_id) wid: vec3<u32>) {
    let tile_id = wid.x + wid.y * p.tiles_x;
    let first = offs[tile_id];
    if (t >= offs[tile_id + 1u] - first) { return; }
    let k = first + t;
    if (k >= p.cap) { return; }
    let at = tile_id * p.slot + t;
    let li = slot_idx[at];
    let ty = li / p.tx;
    let a0 = p.lo + wid.x * p.tx + (li - ty * p.tx);
    let a1 = p.lo + wid.y * p.tile_y + ty;
    out_s[k] = slot_val[at];
    out_x[k] = f32(select(a1, a0, p.swap != 0u));
    out_y[k] = f32(select(a0, a1, p.swap != 0u));
",
    );
    if k.key {
        s.push_str("    out_k[k] = a0 + p.h * a1;\n");
    }
    s.push_str("}\n");
    s
}

/// What a detector asks of [`run`].
pub(crate) struct Detector<'a> {
    pub(crate) image: &'a Array<f32>,
    /// The detection kernel.
    pub(crate) kernel: Corners,
    /// The words of its uniform.
    pub(crate) params: Vec<u32>,
    pub(crate) tile: Tile,
    /// The first and (from the end) last decision pixel along both axes.
    pub(crate) lo: usize,
    /// `x` is the index along axis 0 instead of axis 1.
    pub(crate) swap: bool,
    /// Keep only the first `limit` corners in memory order (axis 0 fastest).
    pub(crate) limit: Option<usize>,
}

/// The corners a detector found.
pub(crate) struct Found {
    pub(crate) x: Array<f32>,
    pub(crate) y: Array<f32>,
    pub(crate) score: Array<f32>,
}

impl Found {
    pub(crate) fn len(&self) -> usize {
        self.x.len()
    }
}

/// Records the detection, scan and gather launches of `d` and reads the corner count back (one
/// synchronization). `None` if there is no corner. The image must have decision pixels
/// (`H, W > 2 lo`).
pub(crate) fn run(d: &Detector<'_>) -> Outcome<Option<Found>> {
    let ctx = d.image.context();
    let dims = d.image.shape().dims();
    let (h, w) = (dims[0], dims[1]);
    let (v0, v1) = (h - 2 * d.lo, w - 2 * d.lo);
    let (tx, ty) = (u32::from(d.tile.tx), u32::from(d.tile.ty));
    let tiles_x = to_u32(v0.div_ceil(tx as usize), "tiles")?;
    let tiles_y = to_u32(v1.div_ceil(ty as usize), "tiles")?;
    let tiles = tiles_x.checked_mul(tiles_y).ok_or(LumaError::DimensionTooLarge { shape: dims })?;
    let grid = tile_grid(ctx, tiles_x, tiles_y)?;
    let slot = d.tile.slots();
    let slots = u64::from(tiles) * u64::from(slot);
    let counts = Array::<u32>::alloc(ctx, Shape::from(tiles as usize + 1))?;
    let mut scratch = Scratch::new(ctx);
    let slot_bytes = slots * 4;
    let slot_idx = scratch.take(slot_bytes)?;
    let slot_val = scratch.take(slot_bytes)?;
    let (src, src_bytes) = d.image.ready()?;
    {
        let counts_pin = counts.ready()?;
        let kernel = ctx.builtin(Builtin::Vision(Any::Corners(d.kernel)), "f32")?;
        let args = [
            RawArg::read(src.buffer(), src_bytes),
            RawArg::write(counts_pin.0.buffer(), counts_pin.1),
            RawArg::write(&slot_idx, slot_bytes),
            RawArg::write(&slot_val, slot_bytes),
            RawArg::Uniform(bytemuck::cast_slice(&d.params)),
        ];
        ctx.launch(&kernel, grid, [h as u32, w as u32, 1], &args)?;
        scan_counts(ctx, &counts_pin, tiles)?;
    }
    let total = counts.slice(0, tiles as usize..tiles as usize + 1).item()? as usize;
    if total == 0 {
        return Ok(None);
    }
    let cap = u32::try_from(total).map_err(|_| LumaError::DimensionTooLarge { shape: dims })?;
    let xs = Array::<f32>::alloc(ctx, Shape::from(total))?;
    let ys = Array::<f32>::alloc(ctx, Shape::from(total))?;
    let scores = Array::<f32>::alloc(ctx, Shape::from(total))?;
    let keys = match d.limit {
        Some(limit) if total > limit => Some(Array::<u32>::alloc(ctx, Shape::from(total))?),
        _ => None,
    };
    {
        let counts_pin = counts.ready()?;
        let kernel = ctx.builtin(Builtin::Vision(Any::Corners(Corners::Gather(Gather { key: keys.is_some() }))), "f32")?;
        let (x_pin, y_pin, s_pin) = (xs.ready()?, ys.ready()?, scores.ready()?);
        let k_pin = keys.as_ref().map(|k| k.ready()).transpose()?;
        let params = [tiles_x, tx, slot, d.lo as u32, cap, h as u32, u32::from(d.swap), ty];
        let mut args = vec![
            RawArg::read(counts_pin.0.buffer(), counts_pin.1),
            RawArg::read(&slot_idx, slot_bytes),
            RawArg::read(&slot_val, slot_bytes),
            RawArg::write(x_pin.0.buffer(), x_pin.1),
            RawArg::write(y_pin.0.buffer(), y_pin.1),
            RawArg::write(s_pin.0.buffer(), s_pin.1),
        ];
        if let Some(pin) = &k_pin {
            args.push(RawArg::write(pin.0.buffer(), pin.1));
        }
        args.push(RawArg::Uniform(bytemuck::bytes_of(&params)));
        ctx.launch(&kernel, grid, [h as u32, w as u32, 1], &args)?;
    }
    let found = Found { x: xs, y: ys, score: scores };
    let (Some(keys), Some(limit)) = (keys, d.limit) else {
        return Ok(Some(found));
    };
    // More corners than the limit: ArrayFire's CPU backend keeps the first ones in memory order.
    let first = keys.argsort().slice(0, 0..limit);
    Ok(Some(Found { x: found.x.take(&first), y: found.y.take(&first), score: found.score.take(&first) }))
}
