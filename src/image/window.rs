//! The windowed image kernels: every output cell is computed from a window of input cells
//! (convolution, rank and morphological filters, the bilateral filter, the Sobel operator and
//! the gradient), over up to three dimensions at once.
//!
//! A workgroup computes a *tile* of outputs. It first stages the input cells the tile's windows
//! cover (the tile plus a halo, bounds and border handling applied once, at staging) in
//! workgroup memory and then computes every output from that copy, so the input is read from
//! global memory about once whatever the window size. Dimension 0 is the fastest in memory and
//! is the one a warp walks along, so loads and stores are coalesced. Batches (the dimensions
//! past the window) are the last axis of the grid.
//!
//! Each `(op, window, border, tile)` combination is its own generated kernel: the window and
//! tile sizes are literals, so the compiler unrolls the loops over the taps.

use std::fmt::Write as _;

/// An element type of the window kernels.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Pixel {
    F32,
    I32,
    U32,
}

impl Pixel {
    pub(crate) fn name(self) -> &'static str {
        match self {
            Pixel::F32 => "f32",
            Pixel::I32 => "i32",
            Pixel::U32 => "u32",
        }
    }

    /// `bits` (a `u32` expression) as a value of this type.
    fn bits_as(self, bits: &str) -> String {
        match self {
            Pixel::U32 => bits.to_owned(),
            other => format!("bitcast<{}>({bits})", other.name()),
        }
    }

    fn zero(self) -> &'static str {
        match self {
            Pixel::F32 => "0.0",
            Pixel::I32 => "0",
            Pixel::U32 => "0u",
        }
    }
}

/// What the cells outside the array are, for the staging of a tile.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum Edge {
    /// Zeros.
    Zero,
    /// The nearest cell.
    Clamp,
    /// The array mirrored about its boundary face.
    Mirror,
    /// The array repeated periodically.
    Wrap,
    /// Not there: the filter's identity (the largest value for a minimum, ...), which the
    /// host puts in `p.fill`.
    Ignore,
}

/// What a window of cells is reduced to.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum Op {
    /// `sum(w * x)`; weights `f32` in binding 2.
    Correlate,
    /// The smallest / largest cell of the window.
    Min,
    Max,
    /// The smallest / largest cell where the mask (binding 2, the pixel type) is not 0.
    MaskedMin,
    MaskedMax,
    /// The cell of rank `n / 2` of the `n` of the window.
    Median,
    /// The bilateral filter: the spatial exponent factor `-1 / (2 s^2)` is `p.c0` and the range
    /// one `-1 / (2 c^2)` is `p.c1` (as `f32` bits). Only `f32`.
    Bilateral,
    /// The bilateral filter of a three-channel image: window `k x k x 3` over axis 2, the range
    /// distance joint over the channels, three planes written.
    BilateralColor,
    /// The Sobel derivatives along dimension 0 and 1 (two outputs), window `3 x 3`, `f32`.
    Sobel,
    /// The Sobel magnitude `sqrt(d0^2 + d1^2)`.
    SobelMagnitude,
    /// The Sobel magnitude `|d0| + |d1|`.
    SobelFast,
    /// The central-difference gradient along dimension 0 and 1 (two outputs), window `3 x 3`,
    /// clamped border, `f32`.
    Gradient,
}

impl Op {
    /// The number of output arrays.
    pub(crate) fn outputs(self) -> usize {
        match self {
            Op::Sobel | Op::Gradient => 2,
            _ => 1,
        }
    }

    /// Whether the kernel reads a weights / mask binding.
    pub(crate) fn has_aux(self) -> bool {
        matches!(self, Op::Correlate | Op::MaskedMin | Op::MaskedMax)
    }
}

/// One windowed kernel: the key of its cache entry.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) struct Window {
    pub(crate) op: Op,
    pub(crate) pixel: Pixel,
    /// The window extent along dimensions 0, 1, 2.
    pub(crate) k: [u16; 3],
    pub(crate) stride: [u8; 3],
    pub(crate) edge: Edge,
    /// Outputs per workgroup along dimensions 0, 1, 2.
    pub(crate) tile: [u8; 3],
    /// Outputs along dimension 1 each invocation computes (a divisor of `tile[1]`). A
    /// `Correlate` with `ry > 1` shares the rows its outputs cover (stride 1 along dimension 1
    /// only); the other operations compute their outputs one after the other.
    pub(crate) ry: u8,
    /// Whether the weights are copied to workgroup memory first.
    pub(crate) stage_aux: bool,
}

/// Words of the uniform block of every windowed kernel (`Params`).
pub(crate) const PARAM_WORDS: usize = 16;

/// Bytes of workgroup memory the kernel declares.
pub(crate) fn shared_bytes(w: &Window) -> u64 {
    let e = extents(w);
    let tile = u64::from(e[0]) * u64::from(e[1]) * u64::from(e[2]) * 4;
    let aux = if w.stage_aux { aux_len(w) as u64 * 4 } else { 0 };
    tile + aux
}

/// The extent of the staged input region along each dimension.
pub(crate) fn extents(w: &Window) -> [u32; 3] {
    let f = |d: usize| (u32::from(w.tile[d]) - 1) * u32::from(w.stride[d]) + u32::from(w.k[d]);
    [f(0), f(1), f(2)]
}

/// The invocations of a workgroup.
pub(crate) fn invocations(w: &Window) -> u32 {
    u32::from(w.tile[0]) * (u32::from(w.tile[1]) / u32::from(w.ry)) * u32::from(w.tile[2])
}

fn aux_len(w: &Window) -> usize {
    usize::from(w.k[0]) * usize::from(w.k[1]) * usize::from(w.k[2])
}

fn mapper(edge: Edge) -> &'static str {
    match edge {
        Edge::Zero | Edge::Ignore => "fn map(c: i32, n: u32) -> i32 { if (c < 0 || c >= i32(n)) { return -1; } return c; }",
        Edge::Clamp => "fn map(c: i32, n: u32) -> i32 { return clamp(c, 0, i32(n) - 1); }",
        Edge::Mirror => {
            "fn map(c: i32, n: u32) -> i32 { let m = i32(n); var r = c % (2 * m); if (r < 0) { r += 2 * m; } if (r >= m) { r = 2 * m - 1 - r; } return r; }"
        }
        Edge::Wrap => "fn map(c: i32, n: u32) -> i32 { let m = i32(n); var r = c % m; if (r < 0) { r += m; } return r; }",
    }
}

/// The key of an order-preserving map of a value to `u32` (and back), for the bitwise selection.
fn key_fns(pixel: Pixel) -> &'static str {
    match pixel {
        Pixel::F32 => {
            "fn key(v: f32) -> u32 { let b = bitcast<u32>(v); return select(b | 0x80000000u, ~b, (b >> 31u) != 0u); }
fn unkey(k: u32) -> f32 { return bitcast<f32>(select(k & 0x7fffffffu, ~k, (k >> 31u) == 0u)); }"
        }
        Pixel::I32 => {
            "fn key(v: i32) -> u32 { return bitcast<u32>(v) ^ 0x80000000u; }
fn unkey(k: u32) -> i32 { return bitcast<i32>(k ^ 0x80000000u); }"
        }
        Pixel::U32 => "fn key(v: u32) -> u32 { return v; }\nfn unkey(k: u32) -> u32 { return k; }",
    }
}

/// The most window cells whose median is selected with registers (an unrolled insertion
/// selection); larger windows are selected bit by bit from workgroup memory.
pub(crate) const MEDIAN_REGISTERS: usize = 49;

/// The most cells of a median window whose kernel is repeated for several outputs per invocation.
pub(crate) const MEDIAN_BLOCKED_CELLS: usize = 25;

/// The WGSL of `w`.
pub(crate) fn source(w: &Window) -> String {
    let ty = w.pixel.name();
    let [s0, s1, s2] = w.stride.map(u32::from);
    let [t0, t1, t2] = w.tile.map(u32::from);
    let ry = u32::from(w.ry);
    let [e0, e1, e2] = extents(w);
    let wg = invocations(w);
    let th = t1 / ry;
    let aux_ty = if w.op == Op::Correlate { "f32" } else { ty };
    let outs = w.op.outputs();
    let mut src = String::new();
    let _ = writeln!(
        src,
        "struct Params {{ n0: u32, n1: u32, n2: u32, m0: u32, m1: u32, m2: u32, p0: u32, p1: u32, p2: u32, t0: u32, t1: u32, t2: u32, nb: u32, fill: u32, c0: u32, c1: u32 }}"
    );
    let mut binding = 0;
    for name in ["out", "out1"].iter().take(outs) {
        let _ = writeln!(src, "@group(1) @binding({binding}) var<storage, read_write> {name}: array<{ty}>;");
        binding += 1;
    }
    let _ = writeln!(src, "@group(1) @binding({binding}) var<storage, read> src: array<{ty}>;");
    binding += 1;
    if w.op.has_aux() {
        let _ = writeln!(src, "@group(1) @binding({binding}) var<storage, read> aux: array<{aux_ty}>;");
        binding += 1;
    }
    let _ = writeln!(src, "@group(1) @binding({binding}) var<uniform> p: Params;");
    let _ = writeln!(src, "var<workgroup> tile: array<{ty}, {}>;", e0 * e1 * e2);
    let stage_aux = w.op.has_aux() && w.stage_aux;
    if stage_aux {
        let _ = writeln!(src, "var<workgroup> wsm: array<{aux_ty}, {}>;", aux_len(w));
    }
    let _ = writeln!(src, "{}", mapper(w.edge));
    if w.op == Op::Median && (aux_len(w) > MEDIAN_REGISTERS) {
        let _ = writeln!(src, "{}", key_fns(w.pixel));
    }
    let fill = w.pixel.bits_as("p.fill");
    let _ = write!(
        src,
        "
@compute @workgroup_size({wg})
fn main(@builtin(local_invocation_index) lid: u32, @builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {{
    let g = wid.x + wid.y * nwg.x;
    let per_batch = p.t0 * p.t1 * p.t2;
    if (g >= per_batch * p.nb) {{ return; }}
    let batch = g / per_batch;
    var r = g - batch * per_batch;
    let bx = r % p.t0;
    r = r / p.t0;
    let by = r % p.t1;
    let bz = r / p.t1;
    let o0 = bx * {t0}u;
    let o1 = by * {t1}u;
    let o2 = bz * {t2}u;
    let src_base = batch * (p.n0 * p.n1 * p.n2);
    let first0 = i32(o0 * {s0}u) - i32(p.p0);
    let first1 = i32(o1 * {s1}u) - i32(p.p1);
    let first2 = i32(o2 * {s2}u) - i32(p.p2);
"
    );
    if stage_aux {
        let _ = writeln!(src, "    for (var i = lid; i < {}u; i += {wg}u) {{ wsm[i] = aux[i]; }}", aux_len(w));
    }
    let _ = write!(
        src,
        "    let inside = first0 >= 0 && first1 >= 0 && first2 >= 0 && first0 + {e0} <= i32(p.n0) && first1 + {e1} <= i32(p.n1) && first2 + {e2} <= i32(p.n2);
    if (inside) {{
        let origin = src_base + u32(first0) + p.n0 * (u32(first1) + p.n1 * u32(first2));
        for (var i = lid; i < {total}u; i += {wg}u) {{
            let a = i % {e0}u;
            let rest = i / {e0}u;
            let b = rest % {e1}u;
            let c = rest / {e1}u;
            tile[i] = src[origin + a + p.n0 * (b + p.n1 * c)];
        }}
    }} else {{
        for (var i = lid; i < {total}u; i += {wg}u) {{
            let a = i % {e0}u;
            let rest = i / {e0}u;
            let b = rest % {e1}u;
            let c = rest / {e1}u;
            let m0 = map(first0 + i32(a), p.n0);
            let m1 = map(first1 + i32(b), p.n1);
            let m2 = map(first2 + i32(c), p.n2);
            var v = {fill};
            if (m0 >= 0 && m1 >= 0 && m2 >= 0) {{ v = src[src_base + u32(m0) + p.n0 * (u32(m1) + p.n1 * u32(m2))]; }}
            tile[i] = v;
        }}
    }}
    workgroupBarrier();
    let tx = lid % {t0}u;
    let tyb = (lid / {t0}u) % {th}u;
    let tz = lid / {plane}u;
    let u0 = o0 + tx;
    let u2 = o2 + tz;
",
        total = e0 * e1 * e2,
        plane = t0 * th,
    );
    if ry == 1 || w.op == Op::Correlate {
        src.push_str("    let ty = tyb;\n");
        src.push_str(&compute(w, ry));
    } else {
        // `ry` outputs one above the other per invocation, each its own block.
        for r in 0..ry {
            let _ = writeln!(src, "    {{\n    let ty = tyb * {ry}u + {r}u;");
            src.push_str(&compute(w, 1));
            src.push_str("    }\n");
        }
    }
    src.push_str("}\n");
    src
}

/// The index of the window cell `(x, y, z)` of the output `(tx, ty, tz)` in the staged tile.
fn cell(w: &Window, x: &str, y: &str, z: &str) -> String {
    let [e0, e1, _] = extents(w);
    format!("(({z}) * {e1}u + ({y})) * {e0}u + ({x})")
}

/// The loops over the window cells of one output `(tx, ty, tz)`, `body` run with the tile index
/// of the cell in `at`, the weight index in `wi` (and the constant positions as literals).
fn taps(w: &Window, body: &dyn Fn(&str, &str) -> String) -> String {
    let [k0, k1, k2] = w.k.map(u32::from);
    let [s0, s1, s2] = w.stride.map(u32::from);
    let mut out = String::new();
    let _ = write!(
        out,
        "    for (var kz = 0u; kz < {k2}u; kz++) {{
        for (var ky = 0u; ky < {k1}u; ky++) {{
            let row = {};
            for (var kx = 0u; kx < {k0}u; kx++) {{
                let at = row + kx;
                let wi = (kz * {k1}u + ky) * {k0}u + kx;
{}            }}
        }}
    }}
",
        cell(w, &format!("tx * {s0}u"), &format!("ty * {s1}u + ky"), &format!("tz * {s2}u + kz")),
        body("at", "wi")
    );
    out
}

fn store(name: &str, y: &str, value: &str) -> String {
    format!(
        "    if (u0 < p.m0 && {y} < p.m1 && u2 < p.m2) {{ {name}[batch * (p.m0 * p.m1 * p.m2) + u0 + p.m0 * ({y} + p.m1 * u2)] = {value}; }}\n"
    )
}

fn compute(w: &Window, ry: u32) -> String {
    let ty = w.pixel.name();
    let aux = if w.op.has_aux() && w.stage_aux { "wsm" } else { "aux" };
    let [k0, k1, _k2] = w.k.map(u32::from);
    let mut out = String::new();
    match w.op {
        Op::Correlate if ry > 1 => {
            // Register-blocked: `ry` outputs one above the other share the rows they cover.
            let [e0, e1, _] = extents(w);
            let [s0, _, s2] = w.stride.map(u32::from);
            let k2 = u32::from(w.k[2]);
            for i in 0..ry {
                let _ = writeln!(out, "    var acc{i} = 0.0;");
            }
            let _ = write!(
                out,
                "    for (var kz = 0u; kz < {k2}u; kz++) {{
        for (var kx = 0u; kx < {k0}u; kx++) {{
            let base = ((tz * {s2}u + kz) * {e1}u + ty * {ry}u) * {e0}u + tx * {s0}u + kx;
"
            );
            for j in 0..k1 + ry - 1 {
                let _ = writeln!(out, "            let v{j} = tile[base + {}u];", j * e0);
                for i in 0..ry {
                    if j >= i && j - i < k1 {
                        let ky = j - i;
                        let _ = writeln!(out, "            acc{i} = fma(v{j}, {aux}[(kz * {k1}u + {ky}u) * {k0}u + kx], acc{i});");
                    }
                }
            }
            out.push_str("        }\n    }\n");
            for i in 0..ry {
                let y = format!("o1 + ty * {ry}u + {i}u");
                out.push_str(&store("out", &y, &format!("acc{i}")));
            }
        }
        Op::Correlate => {
            let _ = writeln!(out, "    var acc = 0.0;");
            out.push_str(&taps(w, &|at, wi| format!("                acc = fma(tile[{at}], {aux}[{wi}], acc);\n")));
            out.push_str(&store("out", "o1 + ty", "acc"));
        }
        Op::Min | Op::Max => {
            let f = if w.op == Op::Min { "min" } else { "max" };
            let _ = writeln!(out, "    var acc = {};", w.pixel.bits_as("p.c0"));
            out.push_str(&taps(w, &|at, _| format!("                acc = {f}(acc, tile[{at}]);\n")));
            out.push_str(&store("out", "o1 + ty", "acc"));
        }
        Op::MaskedMin | Op::MaskedMax => {
            let f = if w.op == Op::MaskedMin { "min" } else { "max" };
            let zero = w.pixel.zero();
            let _ = writeln!(out, "    var acc = {};", w.pixel.bits_as("p.c0"));
            out.push_str(&taps(w, &|at, wi| {
                format!("                if ({aux}[{wi}] != {zero}) {{ acc = {f}(acc, tile[{at}]); }}\n")
            }));
            out.push_str(&store("out", "o1 + ty", "acc"));
        }
        Op::Median => out.push_str(&median(w, ty)),
        Op::Bilateral => {
            let centre = cell(w, &format!("tx + {}u", k0 / 2), &format!("ty + {}u", k1 / 2), &format!("tz + {}u", u32::from(w.k[2]) / 2));
            let _ = writeln!(out, "    let centre = tile[{centre}];");
            let _ = writeln!(out, "    let inv = bitcast<f32>(p.c1);\n    let sp = bitcast<f32>(p.c0);");
            let _ = writeln!(out, "    var acc = 0.0;\n    var norm = 0.0;");
            out.push_str(&taps(w, &|at, _| {
                format!(
                    "                let v = tile[{at}];
                let d = centre - v;
                let ex = f32(kx) - {}.0;
                let ey = f32(ky) - {}.0;
                let e = exp((ex * ex + ey * ey) * sp + d * d * inv);
                acc = fma(v, e, acc);
                norm += e;
",
                    k0 / 2,
                    k1 / 2
                )
            }));
            out.push_str(&store("out", "o1 + ty", "acc / norm"));
        }
        Op::BilateralColor => {
            let [e0, e1, _] = extents(w);
            let plane = e0 * e1;
            let centre = cell(w, &format!("tx + {}u", k0 / 2), &format!("ty + {}u", k1 / 2), "0u");
            let _ = writeln!(out, "    let cc = {centre};");
            let _ = writeln!(out, "    let c0 = tile[cc]; let c1 = tile[cc + {plane}u]; let c2 = tile[cc + {}u];", 2 * plane);
            let _ = writeln!(out, "    let inv = bitcast<f32>(p.c1);\n    let sp = bitcast<f32>(p.c0);");
            let _ = writeln!(out, "    var a0 = 0.0; var a1 = 0.0; var a2 = 0.0; var norm = 0.0;");
            let _ = write!(
                out,
                "    for (var ky = 0u; ky < {k1}u; ky++) {{
        for (var kx = 0u; kx < {k0}u; kx++) {{
            let at = (ty + ky) * {e0}u + tx + kx;
            let v0 = tile[at]; let v1 = tile[at + {plane}u]; let v2 = tile[at + {}u];
            let d0 = c0 - v0; let d1 = c1 - v1; let d2 = c2 - v2;
            let ex = f32(kx) - {}.0;
            let ey = f32(ky) - {}.0;
            let e = exp((ex * ex + ey * ey) * sp + (d0 * d0 + d1 * d1 + d2 * d2) * inv);
            a0 = fma(v0, e, a0); a1 = fma(v1, e, a1); a2 = fma(v2, e, a2);
            norm += e;
        }}
    }}
    let y = o1 + ty;
    if (u0 < p.m0 && y < p.m1) {{
        let o = batch * (p.m0 * p.m1 * 3u) + u0 + p.m0 * y;
        out[o] = a0 / norm;
        out[o + p.m0 * p.m1] = a1 / norm;
        out[o + 2u * p.m0 * p.m1] = a2 / norm;
    }}
",
                2 * plane,
                k0 / 2,
                k1 / 2
            );
        }
        Op::Sobel | Op::SobelMagnitude | Op::SobelFast => {
            let c = |x: u32, y: u32| format!("tile[{}]", cell(w, &format!("tx + {x}u"), &format!("ty + {y}u"), "tz"));
            let _ = writeln!(out, "    let a00 = {}; let a01 = {}; let a02 = {};", c(0, 0), c(0, 1), c(0, 2));
            let _ = writeln!(out, "    let a10 = {}; let a12 = {};", c(1, 0), c(1, 2));
            let _ = writeln!(out, "    let a20 = {}; let a21 = {}; let a22 = {};", c(2, 0), c(2, 1), c(2, 2));
            let _ = writeln!(out, "    let d0 = (a20 + 2.0 * a21 + a22) - (a00 + 2.0 * a01 + a02);");
            let _ = writeln!(out, "    let d1 = (a02 + 2.0 * a12 + a22) - (a00 + 2.0 * a10 + a20);");
            match w.op {
                Op::Sobel => {
                    out.push_str(&store("out", "o1 + ty", "d0"));
                    out.push_str(&store("out1", "o1 + ty", "d1"));
                }
                Op::SobelMagnitude => out.push_str(&store("out", "o1 + ty", "sqrt(d0 * d0 + d1 * d1)")),
                _ => out.push_str(&store("out", "o1 + ty", "abs(d0) + abs(d1)")),
            }
        }
        Op::Gradient => {
            let c = |x: u32, y: u32| format!("tile[{}]", cell(w, &format!("tx + {x}u"), &format!("ty + {y}u"), "tz"));
            let _ = writeln!(out, "    let last0 = u0 == 0u || u0 + 1u == p.m0;");
            let _ = writeln!(out, "    let last1 = (o1 + ty) == 0u || o1 + ty + 1u == p.m1;");
            let _ = writeln!(out, "    let d0 = ({} - {}) * select(0.5, 1.0, last0);", c(2, 1), c(0, 1));
            let _ = writeln!(out, "    let d1 = ({} - {}) * select(0.5, 1.0, last1);", c(1, 2), c(1, 0));
            out.push_str(&store("out", "o1 + ty", "d0"));
            out.push_str(&store("out1", "o1 + ty", "d1"));
        }
    }
    out
}

/// The median of the window: the cell of rank `n / 2`.
fn median(w: &Window, ty: &str) -> String {
    let n = aux_len(w);
    let [k0, k1, k2] = w.k.map(u32::from);
    let mut out = String::new();
    if n > MEDIAN_REGISTERS {
        // Bit by bit: the count of the cells with the prefix found so far and a 0 at the next bit.
        let rank = n / 2;
        let _ = writeln!(out, "    var prefix = 0u;\n    var rank = {rank}u;");
        let _ = writeln!(out, "    for (var bit = 31u; ; bit--) {{");
        let _ = writeln!(out, "        let high = ~((2u << bit) - 1u);\n        var count = 0u;");
        let body = taps(w, &|at, _| format!("                let kk = key(tile[{at}]);\n                if ((kk & high) == prefix && ((kk >> bit) & 1u) == 0u) {{ count++; }}\n"));
        for line in body.lines() {
            let _ = writeln!(out, "    {line}");
        }
        let _ = writeln!(out, "        if (rank >= count) {{ rank -= count; prefix |= 1u << bit; }}");
        let _ = writeln!(out, "        if (bit == 0u) {{ break; }}\n    }}");
        out.push_str(&store("out", "o1 + ty", "unkey(prefix)"));
        return out;
    }
    // The cells in registers.
    let mut names = Vec::with_capacity(n);
    for kz in 0..k2 {
        for ky in 0..k1 {
            for kx in 0..k0 {
                let name = format!("c{}", names.len());
                let at = cell(w, &format!("tx + {kx}u"), &format!("ty + {ky}u"), &format!("tz + {kz}u"));
                let _ = writeln!(out, "    var {name}: {ty} = tile[{at}];");
                names.push(name);
            }
        }
    }
    let swap = |out: &mut String, a: &str, b: &str| {
        let _ = writeln!(out, "    {{ let lo = min({a}, {b}); let hi = max({a}, {b}); {a} = lo; {b} = hi; }}");
    };
    let result = if n == 9 || n == 25 {
        let (net, at) = if n == 9 { (&MED9[..], 4) } else { (&MED25[..], 12) };
        for &(a, b) in net {
            swap(&mut out, &names[a], &names[b]);
        }
        names[at].clone()
    } else if n == 1 {
        names[0].clone()
    } else {
        // Insertion selection: keep the `m` smallest cells sorted in `c0..c{m-1}`; every later
        // cell is pushed through the chain, which drops the largest.
        let m = n / 2 + 1;
        for i in 1..m {
            for j in (1..=i).rev() {
                swap(&mut out, &names[j - 1], &names[j]);
            }
        }
        for i in m..n {
            let x = names[i].clone();
            for name in &names[..m] {
                swap(&mut out, name, &x);
            }
        }
        names[m - 1].clone()
    };
    out.push_str(&store("out", "o1 + ty", &result));
    out
}

/// Devillard's 19 exchange network for the median of 9 (the median ends in cell 4).
const MED9: [(usize, usize); 19] =
    [(1, 2), (4, 5), (7, 8), (0, 1), (3, 4), (6, 7), (1, 2), (4, 5), (7, 8), (0, 3), (5, 8), (4, 7), (3, 6), (1, 4), (2, 5), (4, 7), (4, 2), (6, 4), (4, 2)];

/// The 99 exchange network for the median of 25 (Devillard's `opt_med25`; the median ends in cell 12).
const MED25: [(usize, usize); 99] = [
    (0, 1), (3, 4), (2, 4), (2, 3), (6, 7), (5, 7), (5, 6), (9, 10), (8, 10), (8, 9), (12, 13), (11, 13), (11, 12), (15, 16), (14, 16), (14, 15), (18, 19), (17, 19), (17, 18), (21, 22),
    (20, 22), (20, 21), (23, 24), (2, 5), (3, 6), (0, 6), (0, 3), (4, 7), (1, 7), (1, 4), (11, 14), (8, 14), (8, 11), (12, 15), (9, 15), (9, 12), (13, 16), (10, 16), (10, 13), (20, 23),
    (17, 23), (17, 20), (21, 24), (18, 24), (18, 21), (19, 22), (8, 17), (9, 18), (0, 18), (0, 9), (10, 19), (1, 19), (1, 10), (11, 20), (2, 20), (2, 11), (12, 21), (3, 21), (3, 12),
    (13, 22), (4, 22), (4, 13), (14, 23), (5, 23), (5, 14), (15, 24), (6, 24), (6, 15), (7, 16), (7, 19), (13, 21), (15, 23), (7, 13), (7, 15), (1, 9), (3, 11), (5, 17), (11, 17),
    (9, 17), (4, 10), (6, 12), (7, 14), (4, 6), (4, 7), (12, 14), (10, 14), (6, 7), (10, 12), (6, 10), (6, 17), (12, 17), (7, 17), (7, 10), (12, 18), (7, 12), (10, 18), (12, 20),
    (10, 20), (10, 12),
];

#[cfg(test)]
mod tests {
    use super::*;

    /// By the 0-1 principle a network puts the median in cell `at` iff it does for every input of
    /// zeros and ones, where that cell is 1 exactly when at least `at + 1` inputs are.
    fn median_network_is_correct(net: &[(usize, usize)], n: usize, at: usize) {
        // 64 inputs at a time: bit `l` of word `c` is cell `c` of input `base + l`.
        let total = 1u64 << n;
        let mut base = 0u64;
        while base < total {
            let lanes = (total - base).min(64);
            let mut cells: Vec<u64> = (0..n)
                .map(|c| {
                    let mut word = 0u64;
                    for l in 0..lanes {
                        word |= (((base + l) >> c) & 1) << l;
                    }
                    word
                })
                .collect();
            let ones: Vec<u32> = (0..lanes).map(|l| ((base + l) as u32).count_ones()).collect();
            for &(a, b) in net {
                let (lo, hi) = (cells[a] & cells[b], cells[a] | cells[b]);
                cells[a] = lo;
                cells[b] = hi;
            }
            for l in 0..lanes {
                assert_eq!((cells[at] >> l) & 1 == 1, ones[l as usize] as usize > at, "input {}", base + l);
            }
            base += 64;
        }
    }

    #[test]
    fn the_median_networks_select_the_median() {
        median_network_is_correct(&MED9, 9, 4);
        median_network_is_correct(&MED25, 25, 12);
    }
}
