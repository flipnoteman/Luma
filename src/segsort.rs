//! Sorting many short lines along an axis in workgroup memory: one launch reads every element
//! once, sorts the lines of a tile with a bitonic network and writes every element once.
//!
//! A workgroup takes a tile of [`TILE_WORDS`] words: `W = TILE_WORDS / P` lines, each padded with
//! the largest key to the next power of two `P >= n` (so lines of up to [`MAX_LINE`] elements).
//! The lines of a tile are neighbours in memory whatever the axis:
//!
//! - along axis 0 (`stride == 1`) the lines are `n` consecutive elements, the tile `W` of them
//!   one after the other, loaded with consecutive invocations on consecutive elements of a line;
//! - along any other axis the line `s + o * stride` has its elements `stride` apart, and the
//!   tile takes `W` lines of consecutive `s`: the invocations of a warp read `W` consecutive
//!   words of one row of the `[stride]` axis, and the sorted lines go back to the same places, so
//!   no transpose is needed (the radix route of `sort_axis` has one).
//!
//! The keys are the words of the elements through the order-preserving [`KeyTransform`] of the
//! element type, so integers and floats (NaNs last) sort alike and the output inverts it.

use std::fmt::Write;

use crate::builtin::Builtin;
use crate::context::{Context, RawArg};
use crate::error::LumaError;
use crate::ops::fold_workgroups;
use crate::sort::KeyTransform;

/// Words of workgroup memory (before padding) a tile of lines is sorted in.
const TILE_WORDS: u32 = 2048;
/// Invocations of a workgroup.
const WORKGROUP: u32 = 256;
/// Longest line the kernel sorts.
const MAX_LINE: u32 = TILE_WORDS;
/// Longest line sorted by it on the CPU backend, where the `n log^2 n` comparisons of the network
/// cost more than the passes of the radix sort from a few hundred elements on (measured: even at
/// 256, 1M elements 24 ms against 25 ms; at 16 elements 7 ms against 25 ms; a GPU wins up to 2048
/// by 3 to 4 times, and is level at 4096).
const MAX_LINE_CPU: u32 = 256;

/// Bytes of workgroup memory the kernel declares at most: `TILE_WORDS / P` lines of `P + 1` words,
/// the most (3 words per line, 1024 lines) for `P = 2`.
const MAX_WORKGROUP_BYTES: u32 = TILE_WORDS / 2 * 3 * 4;

/// The longest line [`LineSort`] is used for on `ctx` (0: none, the device has too little workgroup memory).
pub(crate) fn max_line(ctx: &Context) -> u32 {
    if ctx.tuning().workgroup_storage < MAX_WORKGROUP_BYTES {
        0
    } else if ctx.is_cpu() {
        MAX_LINE_CPU
    } else {
        MAX_LINE
    }
}

/// The sort of the lines of an array, see the module documentation.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) struct LineSort {
    pub(crate) transform: KeyTransform,
    /// `log2` of the padded line length (1 to `log2(MAX_LINE)`).
    pub(crate) log2_pad: u8,
    /// Whether the lines are `stride` apart (every axis but 0) rather than consecutive.
    pub(crate) strided: bool,
}

/// What a launch of [`LineSort`] works on; the words of the uniform `Params`.
#[derive(Clone, Copy)]
pub(crate) struct LineJob {
    /// Elements per line.
    pub(crate) n: u32,
    /// Elements between the neighbours of a line (the extents of the axes below it; 1 for axis 0).
    pub(crate) stride: u32,
    /// Number of lines.
    pub(crate) lines: u32,
}

impl LineSort {
    /// The sort of lines of `n` elements (2 to `max_line`) along an axis with `stride`.
    pub(crate) fn new(transform: KeyTransform, n: u32, stride: u32) -> Self {
        debug_assert!((2..=MAX_LINE).contains(&n));
        LineSort { transform, log2_pad: n.next_power_of_two().trailing_zeros() as u8, strided: stride > 1 }
    }

    /// Lines per tile.
    fn per_tile(self) -> u32 {
        TILE_WORDS >> self.log2_pad
    }

    /// Words between the lines in workgroup memory: the padded length plus one, so that the
    /// invocations of a warp that load words of different lines (strided) hit different banks.
    fn pitch(self) -> u32 {
        (1 << self.log2_pad) + 1
    }

    /// Workgroups of a launch for `job`, and the tiles of a layer of `stride` (see the `tpo` word of the
    /// uniform).
    fn grid(self, job: LineJob) -> (u32, u32) {
        let per = self.per_tile();
        if self.strided {
            let per_outer = job.stride.div_ceil(per);
            (per_outer * (job.lines / job.stride), per_outer)
        } else {
            (job.lines.div_ceil(per), 0)
        }
    }

    /// The WGSL of the kernel. Arguments: out, src (both `u32` words), uniform `Params`
    /// `{n, stride, lines, tiles, tpo}`.
    pub(crate) fn source(self) -> String {
        let (pad, per, pitch) = (1u32 << self.log2_pad, self.per_tile(), self.pitch());
        let half = pad / 2;
        let slots = per * pad;
        let words = per * pitch;
        let workgroup = WORKGROUP;
        let mut src = String::new();
        let _ = write!(
            src,
            "struct Params {{ n: u32, stride: u32, lines: u32, tiles: u32, tpo: u32 }}
@group(1) @binding(0) var<storage, read_write> out: array<u32>;
@group(1) @binding(1) var<storage, read> src: array<u32>;
@group(1) @binding(2) var<uniform> p: Params;

var<workgroup> sm: array<u32, {words}>;

fn key_in(b: u32) -> u32 {{
    {forward}
}}
fn key_out(k: u32) -> u32 {{
    {inverse}
}}

@compute @workgroup_size({workgroup})
fn main(@builtin(local_invocation_index) t: u32, @builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {{
    let g = wid.x + wid.y * nwg.x;
    if (g >= p.tiles) {{ return; }}
",
            forward = self.transform.forward(),
            inverse = self.transform.inverse(),
        );
        // `base` is the offset of the tile's first element, `w`/`k` the line of the tile and the
        // position in it of the slot `j`, `lw` the lines of this tile that exist.
        if self.strided {
            let _ = write!(
                src,
                "    let o = g / p.tpo;
    let s0 = (g % p.tpo) * {per}u;
    let lw = min({per}u, p.stride - s0);
    let base = o * p.stride * p.n + s0;
    for (var j = t; j < {slots}u; j += {workgroup}u) {{
        let w = j % {per}u;
        let k = j / {per}u;
        var v = 0xffffffffu;
        if (w < lw && k < p.n) {{ v = key_in(src[base + k * p.stride + w]); }}
        sm[w * {pitch}u + k] = v;
    }}
"
            );
        } else {
            let _ = write!(
                src,
                "    let l0 = g * {per}u;
    let lw = min({per}u, p.lines - l0);
    let base = l0 * p.n;
    for (var j = t; j < {slots}u; j += {workgroup}u) {{
        let w = j / {pad}u;
        let k = j % {pad}u;
        var v = 0xffffffffu;
        if (w < lw && k < p.n) {{ v = key_in(src[base + w * p.n + k]); }}
        sm[w * {pitch}u + k] = v;
    }}
"
            );
        }
        let _ = write!(
            src,
            "    workgroupBarrier();
    for (var k = 2u; k <= {pad}u; k = k << 1u) {{
        for (var j = k >> 1u; j > 0u; j = j >> 1u) {{
            for (var q = t; q < {pairs}u; q += {workgroup}u) {{
                let w = q / {half}u;
                let r = q % {half}u;
                let i = ((r & ~(j - 1u)) << 1u) | (r & (j - 1u));
                let a = w * {pitch}u + i;
                let b = a + j;
                let x = sm[a];
                let y = sm[b];
                if ((x > y) == ((i & k) == 0u)) {{
                    sm[a] = y;
                    sm[b] = x;
                }}
            }}
            workgroupBarrier();
        }}
    }}
",
            pairs = per * half,
        );
        if self.strided {
            let _ = write!(
                src,
                "    for (var j = t; j < {slots}u; j += {workgroup}u) {{
        let w = j % {per}u;
        let k = j / {per}u;
        if (w < lw && k < p.n) {{ out[base + k * p.stride + w] = key_out(sm[w * {pitch}u + k]); }}
    }}
}}
"
            );
        } else {
            let _ = write!(
                src,
                "    for (var j = t; j < {slots}u; j += {workgroup}u) {{
        let w = j / {pad}u;
        let k = j % {pad}u;
        if (w < lw && k < p.n) {{ out[base + w * p.n + k] = key_out(sm[w * {pitch}u + k]); }}
    }}
}}
"
            );
        }
        src
    }

    /// Records the sort of the lines of `job` from `src` into `out` (`job.n * job.lines` words each).
    pub(crate) fn launch(self, ctx: &Context, job: LineJob, out: RawArg<'_>, src: RawArg<'_>) -> Result<(), LumaError> {
        let core = ctx.builtin(Builtin::SortLines(self), "u32")?;
        let (tiles, tpo) = self.grid(job);
        let params = [job.n, job.stride, job.lines, tiles, tpo];
        let size = job.n.checked_mul(job.lines).ok_or(LumaError::DimensionTooLarge { shape: [job.n as usize, job.lines as usize, 1, 1] })?;
        ctx.launch(&core, fold_workgroups(ctx, tiles)?, [size, 1, 1], &[out, src, RawArg::Uniform(bytemuck::bytes_of(&params))])
    }
}
