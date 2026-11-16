//! Cyclic-reduction tridiagonal solves: one workgroup per line.
//!
//! The block Thomas solve of [`crate::builtin`] runs one invocation per grid line, so a grid with
//! a few hundred long lines leaves a GPU almost idle and each invocation walks its line serially.
//! Here a workgroup solves a line together, by cyclic reduction in workgroup memory:
//!
//! 1. every row `(lo, diag, up | rhs)` is normalised to `x_r + L_r x_{r-1} + U_r x_{r+1} = f_r`
//!    (`L = diag^-1 lo`, `U = diag^-1 up`, `f = diag^-1 rhs`; blocks of up to 4 are inverted in
//!    closed form from their cofactors, bigger ones by Gaussian elimination with partial pivoting);
//! 2. forward levels `h = 1, 2, 4, ...`: the rows `r` that are multiples of `2h` eliminate their
//!    neighbours `r - h` and `r + h`, which leaves them coupled to the rows at distance `2h`. The
//!    neighbours are rows that are not modified at this level, so the update is in place and one
//!    barrier per level suffices. Row 0 ends with `x_0 = f_0`;
//! 3. backward levels, `h` halving: the rows `h (2q + 1)` follow from
//!    `x_r = f_r - L_r x_{r-h} - U_r x_{r+h}` with their own, never overwritten, `L`, `U` and `f`.
//!
//! No pivoting across rows, so like the Thomas solve this wants diagonally dominant systems.
//!
//! A row takes `2 B^2 + B` floats, held as `vec4`s (one access moves four floats) padded to an odd
//! count of them, so that the rows of a level, `2h` rows apart, fall on different banks as far as
//! they can. The rows live in workgroup memory up
//! to [`Tuning::line_bytes`]; if a line has more, only the rows that are a multiple of `2^shift`
//! do, and the others (touched in the first and last `shift` levels only) live in a global scratch
//! buffer, indexed compactly. See [`plan`] for when this beats the Thomas solve.

use std::fmt::Write;

use crate::solve::Io;
use crate::tuning::{LineSolver, Tuning};

/// How a workgroup solves one line.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Plan {
    /// Invocations per workgroup.
    pub(crate) workgroup: u16,
    /// Rows that are a multiple of `1 << shift` live in workgroup memory, the others in a global
    /// scratch buffer (`shift == 0`: no scratch).
    pub(crate) shift: u8,
    /// Rows in workgroup memory: `ceil(n / (1 << shift))`.
    pub(crate) shared_rows: u32,
}

impl Plan {
    /// Bytes of global scratch for `lines` lines of `n` rows of `block`-blocks; 0 without spill.
    pub(crate) fn scratch_bytes(&self, block: u32, n: u32, lines: u32) -> u64 {
        if self.shift == 0 {
            return 0;
        }
        u64::from(lines) * u64::from(n - self.shared_rows) * row_bytes(block)
    }
}

/// Floats of one row of `block`-blocks: `L`, `U` and `f`.
fn row_floats(block: u32) -> u32 {
    2 * block * block + block
}

/// Scalar lines (`block == 1`) keep their rows of three floats as floats: held as `vec4`s a third
/// of every row would be padding, and the three loads of a row cost less than the traffic of the
/// fourth lane (RTX 3090, 1024 lines of 1024 cells: 37 us against 43).
fn scalar_rows(block: u32) -> bool {
    block == 1
}

/// The elements of one row in workgroup memory (and the scratch), an odd number so that the rows
/// of a level, `2h` rows apart, fall on different banks as far as they can: floats for a scalar
/// line, else the `vec4`s of [`row_floats`].
fn row_elems(block: u32) -> u32 {
    if scalar_rows(block) { row_floats(block) | 1 } else { row_floats(block).div_ceil(4) | 1 }
}

/// Bytes of one row in workgroup memory.
fn row_bytes(block: u32) -> u64 {
    u64::from(row_elems(block)) * if scalar_rows(block) { 4 } else { 16 }
}

/// The WGSL type of the elements of [`row_elems`].
pub(crate) fn row_type(block: u32) -> &'static str {
    if scalar_rows(block) { "f32" } else { "vec4<f32>" }
}

/// The shift (log2 of the row stride) of the shared rows for a line of `n` rows, and the number
/// of shared rows. A line that fits the workgroup memory whole gets its rows rounded up to a
/// multiple of 32 (as long as that fits too): the array size is a constant of the source, so
/// lines of 106 and 122 rows share one kernel instead of compiling one each, at the price of at
/// most 31 unused rows of workgroup memory. A spilled line keeps its exact count, which the
/// scratch indexing bakes in.
fn layout(tuning: &Tuning, block: u32, n: u32) -> (u8, u32) {
    let row = row_bytes(block);
    let budget = u64::from(tuning.line_bytes);
    let mut shift = 0u8;
    while shift < 31 && u64::from(n.div_ceil(1 << shift)) * row > budget && n.div_ceil(1 << shift) > 1 {
        shift += 1;
    }
    let rows = n.div_ceil(1 << shift);
    let padded = rows.next_multiple_of(32);
    if shift == 0 && u64::from(padded) * row <= budget { (shift, padded) } else { (shift, rows) }
}

/// Whether the line solve of `lines` lines of `n` cells of `block`-blocks, `stride` cells apart
/// in memory, runs by cyclic reduction, and how. `None` means Thomas. Forced
/// (`LUMA_TRIDIAGONAL=cyclic`) it is taken whenever the device can run it; automatically only for
/// few lines (see [`Tuning::line_cyclic_work`]), where the one invocation per line of the Thomas
/// solve leaves the device idle. `parts` is the solve on parts (every block entry an array of its
/// own, see `linalg::run_parts`), whose memory access differs from stacked blocks. `max_scratch`
/// is the largest buffer the spill may take.
///
/// Stacked blocks, measured on an RTX 3090 (block 1, 2, 4, 8, `n` 32 to 4096, 4 to 16384 lines,
/// both axes): the workgroup per line beats Thomas by 2x to 50x up to a few hundred lines for
/// every block size and length, and keeps winning with more lines when the blocks are small
/// (scalar lines along the contiguous axis to 16384 lines), and loses (1.2x to 7x slower) once
/// there are thousands of lines of strided scalars or 1024 lines of 4x4 blocks, where one thread
/// per line already fills the device and its coalesced loads across lines beat a workgroup's
/// strided ones. The rule `lines * block^2 <= 8192` for contiguous lines and
/// `lines * 4 * block <= 8192` for strided ones keeps to the clear wins of both regimes (scalar
/// 2048 / 4096 lines, 4x4 blocks 512 / 512 lines, 8x8 blocks 128 / 256 lines).
///
/// Parts, measured on an RTX 3090 (block 1, 2, 4, 5, `n` 24 to 512, 256 to 16384 lines): along
/// the contiguous axis the Thomas threads read every part `n` cells apart, a sector per element,
/// and cyclic reduction wins at every size for blocks of 2 and up (5x5 blocks, 1680 lines of 187:
/// 330 us vs 3200 us; 4x4 blocks, 8192 lines of 187: 0.9 ms vs 15 ms) and for scalars of 70 cells
/// and up (16384 lines of 187: 110 us vs 760 us; 24 cells: within 15 %), so it is always taken.
/// Along a strided axis the Thomas loads coalesce across lines and the workgroup's do not: cyclic
/// wins to about 2048 lines of scalars or 2x2 blocks, 1024 of 4x4 and fewer of 5x5, and half
/// that when the line does not fit the workgroup memory (4x4 blocks, 1024 lines of 512: 2.5 ms
/// vs 1.0 ms), so the rule is `lines * 2 * max(block, 2) <= 8192`, the weight doubled for a line
/// that spills.
pub(crate) fn plan(tuning: &Tuning, block: u32, n: u32, lines: u32, stride: u32, parts: bool, max_scratch: u64) -> Option<Plan> {
    if tuning.line_solver == LineSolver::Thomas || n == 0 || lines == 0 {
        return None;
    }
    let (shift, shared_rows) = layout(tuning, block, n);
    let workgroup = workgroup_size(tuning, block, n, stride > 1);
    let plan = Plan { workgroup, shift, shared_rows };
    // A single row must fit the budget, and the scratch a buffer.
    if u64::from(shared_rows) * row_bytes(block) > u64::from(tuning.line_bytes) || plan.scratch_bytes(block, n, lines) > max_scratch {
        return None;
    }
    let scratch_elements = u64::from(lines) * u64::from(n.saturating_sub(shared_rows)) * u64::from(row_elems(block));
    if scratch_elements > u64::from(u32::MAX) {
        return None;
    }
    if tuning.line_solver == LineSolver::Auto {
        if tuning.line_cyclic_work == 0 {
            return None;
        }
        let weight = match (parts, stride == 1) {
            (true, true) => return Some(plan),
            (true, false) => 2 * block.max(2) * if shift > 0 { 2 } else { 1 },
            (false, true) => block * block,
            (false, false) => 4 * block,
        };
        if u64::from(lines) * u64::from(weight) > u64::from(tuning.line_cyclic_work) {
            return None;
        }
    }
    Some(plan)
}

/// Invocations per workgroup, from a warp of 32 up to a cap that falls with the block size, as
/// the registers of one row's solve grow with `block^2`. Scalar lines take one per two rows (the
/// first forward level; RTX 3090, 256 is best). Blocks of 2 to 4 take one per row, in warps: the
/// per-row solve (a closed-form inverse and its products) is a long chain of dependent
/// operations, so more threads hide its latency better than more rows per thread (4x4 blocks,
/// 144 rows: 32 threads 18 us, 64 15 us, 96 and up 13 us; 2048 rows of 2x2 blocks: 32 105 us,
/// 256 29 us); the cap is 256, or 128 for blocks of 3 and 4 along a strided axis, where the
/// uncoalesced reads of more rows at once stop paying (200 rows of 3x3 blocks, 400 lines: 128 47
/// us, 256 61 us). Blocks of 5, eliminated with many registers, take one per row up to 128 along
/// the contiguous axis, else one per 8 rows (the old rule, up to 64: 100 rows, 200 lines: 32
/// threads 42 us, 128 61 us). Bigger blocks take one per 8 rows up to 64.
fn workgroup_size(tuning: &Tuning, block: u32, n: u32, strided: bool) -> u16 {
    let rows = n.next_multiple_of(32);
    let wanted = match block {
        1 => n.div_ceil(2).next_power_of_two().clamp(32, 256),
        2 => rows.clamp(32, 256),
        3 | 4 => rows.clamp(32, if strided { 128 } else { 256 }),
        5 if !strided => rows.clamp(32, 128),
        _ => n.div_ceil(8).next_power_of_two().clamp(32, 64),
    };
    wanted.min(tuning.elementwise_workgroup.max(1)) as u16
}

/// What differs between the kernels of [`body`].
pub(crate) struct Config {
    pub(crate) block: u32,
    pub(crate) workgroup: u32,
    pub(crate) shift: u32,
    pub(crate) shared_rows: u32,
    /// Whether `starts[l]` is the first cell of line `l`.
    pub(crate) starts: bool,
    /// Independent systems on the same grid: workgroup `lg` solves line `lg % lines` of system
    /// `lg / lines`.
    pub(crate) systems: u32,
}

/// Bindings of the kernel of [`Io::buffers`], in order: `out`, the scratch `sm` (with spill),
/// `lo`, `diag`, `up`, `rhs`, `starts` (with `starts`), the uniform parameters `{n, stride,
/// lines}`.
pub(crate) fn source(block: u32, starts: bool, workgroup: u32, shift: u32, shared_rows: u32) -> String {
    let io = Io::buffers(block as usize);
    let mut binding = 0;
    let mut next = || {
        binding += 1;
        binding - 1
    };
    let mut src = String::new();
    let _ = writeln!(src, "struct Params {{ n: u32, stride: u32, lines: u32 }}");
    let _ = writeln!(src, "@group(1) @binding({}) var<storage, read_write> out: array<{}>;", next(), io.rhs_type());
    if shift > 0 {
        let _ = writeln!(src, "@group(1) @binding({}) var<storage, read_write> sm: array<{}>;", next(), row_type(block));
    }
    for name in ["lo", "diag", "up"] {
        let _ = writeln!(src, "@group(1) @binding({}) var<storage, read> {name}: array<{}>;", next(), io.coef_type());
    }
    let _ = writeln!(src, "@group(1) @binding({}) var<storage, read> rhs: array<{}>;", next(), io.rhs_type());
    if starts {
        let _ = writeln!(src, "@group(1) @binding({}) var<storage, read> starts: array<u32>;", next());
    }
    let _ = writeln!(src, "@group(1) @binding({}) var<uniform> p: Params;", next());
    src.push_str(&body(&Config { block, workgroup, shift, shared_rows, starts, systems: 1 }, &io));
    src
}

/// The expression of float `k` of a row (`L`, then `U`, then `f`) out of the solved columns
/// `{name}{i}_{j}`, zero beyond the row.
fn row_value(name: &str, b: usize, k: usize) -> String {
    let bb = b * b;
    if k < bb {
        format!("{name}{}_{}", k / b, k % b)
    } else if k < 2 * bb {
        format!("{name}{}_{}", (k - bb) / b, b + (k - bb) % b)
    } else if k < 2 * bb + b {
        format!("{name}{}_{}", k - 2 * bb, 2 * b)
    } else {
        "0.0".to_owned()
    }
}

/// Name of float `k` of a row held in the `vec4`s `{name}0..`.
fn fl(name: &str, k: usize) -> String {
    format!("{name}{}.{}", k / 4, ["x", "y", "z", "w"][k % 4])
}

/// The kernel after its bindings (`out`, `sm` with a spill, whatever the coefficients are read
/// from, `starts`, `p`): workgroup memory, the line addressing, and `main`.
pub(crate) fn body(cfg: &Config, io: &Io) -> String {
    let Config { block, workgroup, shift, shared_rows, starts, systems } = *cfg;
    let b = block as usize;
    let (bb, w) = (b * b, 2 * b + 1);
    let floats = row_floats(block) as usize;
    let dv = floats.div_ceil(4);
    let rv = row_elems(block) as usize;
    let row_type = row_type(block);
    let spill = shift > 0;
    let closed = b <= 4;
    let mut src = String::new();
    let _ = writeln!(
        src,
        "const WG: u32 = {workgroup}u;
const RV: u32 = {rv}u;
var<workgroup> sh: array<{row_type}, {}>;
fn line_base(l: u32) -> u32 {{ return (l % p.stride) + (l / p.stride) * p.stride * p.n; }}",
        shared_rows as usize * rv
    );
    // The rows: `ld4(r, e)` and `st4(r, e, v)` read and write the vec4 `e` of row `r`, wherever it
    // is (in shared memory, or in the scratch if `r` is not a multiple of `KS`); `ldf` and `stf`
    // the vec4s that hold `f`, the solution, which is the same for blocks of 2 and more. A scalar
    // line's row is three floats, read and written one at a time (`ld1`, `st1`, with a branch of
    // their own over the scratch: one branch per row measured 5% slower, 200 lines of 4096 cells),
    // and `ldf`, `stf` touch only its `f`, as the unused lanes of a function that returns a whole
    // row are loads that the compiler keeps.
    let scalar = scalar_rows(block);
    let ks = 1u32 << shift;
    if spill {
        let _ = writeln!(src, "const KS: u32 = {ks}u;\nvar<private> sbase: u32;");
    }
    // `array[{at}(r) + e]`, in shared memory or the scratch.
    let (in_shared, in_scratch) = (format!("(r >> {shift}u) * RV"), format!("sbase + (r - (r >> {shift}u) - 1u) * RV"));
    let mask = ks - 1;
    if scalar {
        if spill {
            let _ = writeln!(
                src,
                "fn ld1(r: u32, e: u32) -> f32 {{
    if ((r & {mask}u) == 0u) {{ return sh[{in_shared} + e]; }}
    return sm[{in_scratch} + e];
}}
fn st1(r: u32, e: u32, v: f32) {{
    if ((r & {mask}u) == 0u) {{ sh[{in_shared} + e] = v; return; }}
    sm[{in_scratch} + e] = v;
}}"
            );
        } else {
            src.push_str("fn ld1(r: u32, e: u32) -> f32 { return sh[r * RV + e]; }\nfn st1(r: u32, e: u32, v: f32) { sh[r * RV + e] = v; }\n");
        }
        src.push_str(
            "fn ld4(r: u32, e: u32) -> vec4<f32> { return vec4<f32>(ld1(r, 0u), ld1(r, 1u), ld1(r, 2u), 0.0); }
fn ldf(r: u32, e: u32) -> vec4<f32> { return vec4<f32>(0.0, 0.0, ld1(r, 2u), 0.0); }
fn st4(r: u32, e: u32, v: vec4<f32>) { st1(r, 0u, v.x); st1(r, 1u, v.y); st1(r, 2u, v.z); }
fn stf(r: u32, e: u32, v: vec4<f32>) { st1(r, 2u, v.z); }
",
        );
    } else {
        if spill {
            let _ = writeln!(
                src,
                "fn ld4(r: u32, e: u32) -> vec4<f32> {{
    if ((r & {mask}u) == 0u) {{ return sh[{in_shared} + e]; }}
    return sm[{in_scratch} + e];
}}
fn st4(r: u32, e: u32, v: vec4<f32>) {{
    if ((r & {mask}u) == 0u) {{ sh[{in_shared} + e] = v; return; }}
    sm[{in_scratch} + e] = v;
}}"
            );
        } else {
            src.push_str("fn ld4(r: u32, e: u32) -> vec4<f32> { return sh[r * RV + e]; }\nfn st4(r: u32, e: u32, v: vec4<f32>) { sh[r * RV + e] = v; }\n");
        }
        src.push_str("fn ldf(r: u32, e: u32) -> vec4<f32> { return ld4(r, e); }\nfn stf(r: u32, e: u32, v: vec4<f32>) { st4(r, e, v); }\n");
    }
    let s0 = if starts { "min(starts[l], p.n)" } else { "0u" };
    let sbase = if spill { format!("    sbase = lg * ((p.n - {shared_rows}u) * RV);\n") } else { String::new() };
    let (guard, sys, line) = if systems > 1 {
        (format!("p.lines * {systems}u"), "    let sys = lg / p.lines;\n", "lg % p.lines")
    } else {
        ("p.lines".to_owned(), "    let sys = 0u;\n", "lg")
    };
    let _ = write!(
        src,
        "@compute @workgroup_size({workgroup})
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_index) li: u32) {{
    let lg = wid.x + wid.y * nwg.x;
    if (lg >= {guard}) {{ return; }}
{sys}    let l = {line};
    let base = line_base(l);
    let s0 = {s0};
    for (var s = li; s < s0; s += WG) {{
        let q = base + s * p.stride;
"
    );
    let zeros: Vec<String> = vec!["0.0".to_owned(); b];
    src.push_str(&io.store("        ", "q", &zeros));
    src.push_str("    }\n    let m = p.n - s0;\n    if (m == 0u) { return; }\n");
    src.push_str(&sbase);
    // The barrier after a step that wrote rows of the scratch.
    let sync = |global: &str| -> String {
        if spill {
            format!("        if ({global}) {{ storageBarrier(); }}\n        workgroupBarrier();\n")
        } else {
            "        workgroupBarrier();\n".to_string()
        }
    };

    // Stores the solved columns `{name}{i}_{j}` (`L`, `U`, `f`) as the row `r`.
    let store_row = |ind: &str, name: &str| -> String {
        let mut out = String::new();
        for v in 0..dv {
            let lanes: Vec<String> = (4 * v..4 * v + 4).map(|k| row_value(name, b, k)).collect();
            let _ = writeln!(out, "{ind}st4(r, {v}u, vec4<f32>({}));", lanes.join(", "));
        }
        out
    };

    // Normalise every row.
    src.push_str("    for (var r = li; r < m; r += WG) {\n        let q = base + (s0 + r) * p.stride;\n");
    src.push_str(&io.load("        ", "q"));
    src.push_str(&io.load_lo("        ", "q"));
    for i in 0..b {
        for j in 0..b {
            let _ = writeln!(src, "        var m{i}_{j} = {};", io.diag("q", i * b + j));
            let _ = writeln!(src, "        var r{i}_{j} = select(0.0, {}, r > 0u);", io.lo("q", i * b + j));
            let _ = writeln!(src, "        var r{i}_{} = select(0.0, {}, r + 1u < m);", b + j, io.up("q", i * b + j));
        }
        let _ = writeln!(src, "        var r{i}_{} = {};", 2 * b, io.rhs("q", i));
    }
    let result = solve(&mut src, "        ", b, w, closed);
    src.push_str(&store_row("        ", result));
    src.push_str("    }\n");
    src.push_str(&sync("true").replace("        ", "    "));

    // Forward levels.
    src.push_str("    var h = 1u;\n    for (; h < m; ) {\n");
    src.push_str("        let step = 2u * h;\n        let count = (m - 1u) / step + 1u;\n        for (var j = li; j < count; j += WG) {\n            let r = j * step;\n");
    let ind = "            ";
    for v in 0..dv {
        let _ = writeln!(src, "{ind}let o{v} = ld4(r, {v}u);");
    }
    for i in 0..b {
        for j in 0..b {
            let _ = writeln!(src, "{ind}var m{i}_{j} = {};", if i == j { "1.0" } else { "0.0" });
            let _ = writeln!(src, "{ind}var r{i}_{j} = 0.0;\n{ind}var r{i}_{} = 0.0;", b + j);
        }
        let _ = writeln!(src, "{ind}var r{i}_{} = {};", 2 * b, fl("o", 2 * bb + i));
    }
    for (side, cond, neighbour, factor) in [("left", "r > 0u", "r - h", 0), ("right", "r + h < m", "r + h", bb)] {
        // `d -= factor * (opposite coupling of the neighbour)`, `(own side) = -factor * (same side of the neighbour)`.
        let (to_d, to_r_col) = if side == "left" { (bb, 0) } else { (0, b) };
        let same = if side == "left" { 0 } else { bb };
        let _ = writeln!(src, "{ind}if ({cond}) {{\n{ind}    let nb = {neighbour};");
        for v in 0..dv {
            let _ = writeln!(src, "{ind}    let n{v} = ld4(nb, {v}u);");
        }
        for i in 0..b {
            for j in 0..b {
                let sum_d = (0..b).map(|k| format!("{} * {}", fl("o", factor + i * b + k), fl("n", to_d + k * b + j))).collect::<Vec<_>>().join(" + ");
                let sum_r = (0..b).map(|k| format!("{} * {}", fl("o", factor + i * b + k), fl("n", same + k * b + j))).collect::<Vec<_>>().join(" + ");
                let _ = writeln!(src, "{ind}    m{i}_{j} -= {sum_d};\n{ind}    r{i}_{} = -({sum_r});", to_r_col + j);
            }
            let sum_f = (0..b).map(|k| format!("{} * {}", fl("o", factor + i * b + k), fl("n", 2 * bb + k))).collect::<Vec<_>>().join(" + ");
            let _ = writeln!(src, "{ind}    r{i}_{} -= {sum_f};", 2 * b);
        }
        let _ = writeln!(src, "{ind}}}");
    }
    let result = solve(&mut src, ind, b, w, closed);
    src.push_str(&store_row(ind, result));
    src.push_str("        }\n");
    src.push_str(&sync("2u * h < KS"));
    src.push_str("        if (h >= 0x80000000u) { break; }\n        h = h << 1u;\n    }\n");

    // Backward levels: only the vec4s holding `f` are written.
    let f_slots = (2 * bb / 4)..(floats.div_ceil(4));
    src.push_str("    while (h > 1u) {\n        h = h >> 1u;\n        let step = 2u * h;\n        let count = (m - 1u - h) / step + 1u;\n        for (var j = li; j < count; j += WG) {\n            let r = h + j * step;\n");
    for v in 0..dv {
        let _ = writeln!(src, "{ind}let o{v} = ld4(r, {v}u);");
    }
    for v in f_slots.clone() {
        let _ = writeln!(src, "{ind}let a{v} = ldf(r - h, {v}u);");
    }
    for i in 0..b {
        let sum = (0..b).map(|k| format!("{} * {}", fl("o", i * b + k), fl("a", 2 * bb + k))).collect::<Vec<_>>().join(" + ");
        let _ = writeln!(src, "{ind}var x{i} = {} - ({sum});", fl("o", 2 * bb + i));
    }
    let _ = writeln!(src, "{ind}if (r + h < m) {{");
    for v in f_slots.clone() {
        let _ = writeln!(src, "{ind}    let c{v} = ldf(r + h, {v}u);");
    }
    for i in 0..b {
        let sum = (0..b).map(|k| format!("{} * {}", fl("o", bb + i * b + k), fl("c", 2 * bb + k))).collect::<Vec<_>>().join(" + ");
        let _ = writeln!(src, "{ind}    x{i} -= {sum};");
    }
    let _ = writeln!(src, "{ind}}}");
    for v in f_slots {
        let lanes: Vec<String> = (4 * v..4 * v + 4)
            .map(|k| {
                if (2 * bb..2 * bb + b).contains(&k) {
                    format!("x{}", k - 2 * bb)
                } else if k < floats {
                    fl("o", k)
                } else {
                    "0.0".to_owned()
                }
            })
            .collect();
        let _ = writeln!(src, "{ind}stf(r, {v}u, vec4<f32>({}));", lanes.join(", "));
    }
    src.push_str("        }\n");
    src.push_str(&sync("h < KS"));
    src.push_str("    }\n");

    // The solution.
    src.push_str("    for (var r = li; r < m; r += WG) {\n        let q = base + (s0 + r) * p.stride;\n");
    for v in (2 * bb / 4)..floats.div_ceil(4) {
        let _ = writeln!(src, "        let e{v} = ldf(r, {v}u);");
    }
    let values: Vec<String> = (0..b).map(|i| fl("e", 2 * bb + i)).collect();
    src.push_str(&io.store("        ", "q", &values));
    src.push_str("    }\n}\n");
    src
}

/// Solves `m X = r` in place for the scalars `m{i}_{j}` (`b x b`) and `r{i}_{j}` (`b x w`) and
/// returns the prefix of the names that hold `X` afterwards (`r{i}_{j}` or `x{i}_{j}`). Blocks of
/// up to 4 (`closed`) are inverted from their cofactors and multiplied into `r`; bigger ones by
/// [`eliminate`].
fn solve(src: &mut String, ind: &str, b: usize, w: usize, closed: bool) -> &'static str {
    if !closed {
        eliminate(src, ind, b, w);
        return "r";
    }
    if b > 1 {
        equilibrate(src, ind, b);
    }
    inverse(src, ind, b);
    for i in 0..b {
        for j in 0..w {
            let sum = (0..b).map(|k| format!("iv{i}_{k} * r{k}_{j}")).collect::<Vec<_>>().join(" + ");
            let _ = writeln!(src, "{ind}let x{i}_{j} = {sum};");
        }
    }
    "x"
}

/// Scales the block `m` by `bt = 1 / max |m{i}_{j}|` (not a block of zeros), which keeps the
/// cofactors and the determinant of [`inverse`] in range for blocks of any uniform magnitude (a
/// 4x4 block of entries of `1e10` has a determinant of `1e40`); `inverse` folds `bt` back into the
/// inverse of the scaled block. A block of zeros stays so and singular. Rows of magnitudes that
/// differ by more than the range of `f32` over `b` still overflow, as the pivoted solves do not.
fn equilibrate(src: &mut String, ind: &str, b: usize) {
    let mut max = "abs(m0_0)".to_owned();
    for i in 0..b {
        for j in 0..b {
            if i + j > 0 {
                max = format!("max({max}, abs(m{i}_{j}))");
            }
        }
    }
    let _ = writeln!(src, "{ind}let bm = {max};\n{ind}let bt = select(1.0, 1.0 / bm, bm > 0.0);");
    for i in 0..b {
        for j in 0..b {
            let _ = writeln!(src, "{ind}m{i}_{j} *= bt;");
        }
    }
}

/// Declares `iv{i}_{j}`, the inverse of the `b x b` matrix `m{i}_{j}` (`b <= 4`), from its
/// cofactors (the 2x2 minors of the row pairs for `b = 4`). Nothing is pivoted, but no pivot is
/// divided by either: a matrix with a zero diagonal and a determinant that is not zero inverts
/// as well as any (the error grows with its condition number); a singular one gives `Inf`/`NaN`.
/// `m` must have been scaled by `bt` (see [`equilibrate`]; `b >= 2`), which `inverse` undoes.
/// About half the operations of the elimination of [`eliminate`], without its selects, and a
/// shorter chain of dependent ones.
fn inverse(src: &mut String, ind: &str, b: usize) {
    let mut line = |text: String| {
        let _ = writeln!(src, "{ind}{text}");
    };
    match b {
        1 => line("let iv0_0 = 1.0 / m0_0;".to_owned()),
        2 => {
            line("let kid = bt / (m0_0 * m1_1 - m0_1 * m1_0);".to_owned());
            for text in ["let iv0_0 = m1_1 * kid;", "let iv0_1 = -m0_1 * kid;", "let iv1_0 = -m1_0 * kid;", "let iv1_1 = m0_0 * kid;"] {
                line(text.to_owned());
            }
        }
        3 => {
            for text in ["let ka0 = m1_1 * m2_2 - m1_2 * m2_1;", "let ka1 = m1_2 * m2_0 - m1_0 * m2_2;", "let ka2 = m1_0 * m2_1 - m1_1 * m2_0;"] {
                line(text.to_owned());
            }
            line("let kid = bt / (m0_0 * ka0 + m0_1 * ka1 + m0_2 * ka2);".to_owned());
            for (i, row) in [
                ["ka0", "(m0_2 * m2_1 - m0_1 * m2_2)", "(m0_1 * m1_2 - m0_2 * m1_1)"],
                ["ka1", "(m0_0 * m2_2 - m0_2 * m2_0)", "(m0_2 * m1_0 - m0_0 * m1_2)"],
                ["ka2", "(m0_1 * m2_0 - m0_0 * m2_1)", "(m0_0 * m1_1 - m0_1 * m1_0)"],
            ]
            .iter()
            .enumerate()
            {
                for (j, entry) in row.iter().enumerate() {
                    line(format!("let iv{i}_{j} = {entry} * kid;"));
                }
            }
        }
        _ => {
            // Laplace expansion in the 2x2 minors of rows 0, 1 (`s`) and rows 2, 3 (`c`).
            for (name, (j, k)) in [("ks0", (0, 1)), ("ks1", (0, 2)), ("ks2", (0, 3)), ("ks3", (1, 2)), ("ks4", (1, 3)), ("ks5", (2, 3))] {
                line(format!("let {name} = m0_{j} * m1_{k} - m1_{j} * m0_{k};"));
            }
            for (name, (j, k)) in [("kc5", (2, 3)), ("kc4", (1, 3)), ("kc3", (1, 2)), ("kc2", (0, 3)), ("kc1", (0, 2)), ("kc0", (0, 1))] {
                line(format!("let {name} = m2_{j} * m3_{k} - m3_{j} * m2_{k};"));
            }
            line("let kid = bt / (ks0 * kc5 - ks1 * kc4 + ks2 * kc3 + ks3 * kc2 - ks4 * kc1 + ks5 * kc0);".to_owned());
            let rows = [
                ["m1_1 * kc5 - m1_2 * kc4 + m1_3 * kc3", "-m0_1 * kc5 + m0_2 * kc4 - m0_3 * kc3", "m3_1 * ks5 - m3_2 * ks4 + m3_3 * ks3", "-m2_1 * ks5 + m2_2 * ks4 - m2_3 * ks3"],
                ["-m1_0 * kc5 + m1_2 * kc2 - m1_3 * kc1", "m0_0 * kc5 - m0_2 * kc2 + m0_3 * kc1", "-m3_0 * ks5 + m3_2 * ks2 - m3_3 * ks1", "m2_0 * ks5 - m2_2 * ks2 + m2_3 * ks1"],
                ["m1_0 * kc4 - m1_1 * kc2 + m1_3 * kc0", "-m0_0 * kc4 + m0_1 * kc2 - m0_3 * kc0", "m3_0 * ks4 - m3_1 * ks2 + m3_3 * ks0", "-m2_0 * ks4 + m2_1 * ks2 - m2_3 * ks0"],
                ["-m1_0 * kc3 + m1_1 * kc1 - m1_2 * kc0", "m0_0 * kc3 - m0_1 * kc1 + m0_2 * kc0", "-m3_0 * ks3 + m3_1 * ks1 - m3_2 * ks0", "m2_0 * ks3 - m2_1 * ks1 + m2_2 * ks0"],
            ];
            for (i, row) in rows.iter().enumerate() {
                for (j, entry) in row.iter().enumerate() {
                    line(format!("let iv{i}_{j} = ({entry}) * kid;"));
                }
            }
        }
    }
}

/// Solves `m X = r` in place for the scalars `m{i}_{j}` (`b x b`) and `r{i}_{j}` (`b x w`):
/// Gaussian elimination with partial pivoting (the pivot row is found and exchanged with
/// `select`s, nothing branches; the first row of largest magnitude wins ties), then back
/// substitution. `r` holds `X` afterwards.
fn eliminate(src: &mut String, ind: &str, b: usize, w: usize) {
    for col in 0..b {
        if col + 1 < b {
            let _ = writeln!(src, "{ind}var best{col} = abs(m{col}_{col});\n{ind}var pv{col} = {col}u;");
            for i in col + 1..b {
                let _ = writeln!(
                    src,
                    "{ind}let a{col}_{i} = abs(m{i}_{col});\n{ind}let gt{col}_{i} = a{col}_{i} > best{col};\n{ind}best{col} = select(best{col}, a{col}_{i}, gt{col}_{i});\n{ind}pv{col} = select(pv{col}, {i}u, gt{col}_{i});"
                );
            }
            for i in col + 1..b {
                let _ = writeln!(src, "{ind}let sw{col}_{i} = pv{col} == {i}u;");
                let entries = (col..b).map(|j| format!("m{{}}_{j}")).chain((0..w).map(|j| format!("r{{}}_{j}")));
                for entry in entries {
                    let (top, other) = (entry.replace("{}", &col.to_string()), entry.replace("{}", &i.to_string()));
                    let _ = writeln!(
                        src,
                        "{ind}{{ let t = {top}; {top} = select({top}, {other}, sw{col}_{i}); {other} = select({other}, t, sw{col}_{i}); }}"
                    );
                }
            }
        }
        let _ = writeln!(src, "{ind}let v{col} = 1.0 / m{col}_{col};");
        for i in col + 1..b {
            let _ = writeln!(src, "{ind}let f{col}_{i} = m{i}_{col} * v{col};");
            for j in col + 1..b {
                let _ = writeln!(src, "{ind}m{i}_{j} -= f{col}_{i} * m{col}_{j};");
            }
            for j in 0..w {
                let _ = writeln!(src, "{ind}r{i}_{j} -= f{col}_{i} * r{col}_{j};");
            }
        }
    }
    for j in 0..w {
        for i in (0..b).rev() {
            let mut x = format!("r{i}_{j}");
            for k in i + 1..b {
                let _ = write!(x, " - m{i}_{k} * r{k}_{j}");
            }
            let _ = writeln!(src, "{ind}r{i}_{j} = ({x}) * v{i};");
        }
    }
}
