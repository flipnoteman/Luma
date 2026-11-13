//! Cyclic-reduction tridiagonal solves: one workgroup per line.
//!
//! The block Thomas solve of [`crate::builtin`] runs one invocation per grid line, so a grid with
//! a few hundred long lines leaves a GPU almost idle and each invocation walks its line serially.
//! Here a workgroup solves a line together, by cyclic reduction in workgroup memory:
//!
//! 1. every row `(lo, diag, up | rhs)` is normalised to `x_r + L_r x_{r-1} + U_r x_{r+1} = f_r`
//!    (`L = diag^-1 lo`, `U = diag^-1 up`, `f = diag^-1 rhs`; the blocks are inverted by Gaussian
//!    elimination with partial pivoting, like the pivots of the Thomas solve);
//! 2. forward levels `h = 1, 2, 4, ...`: the rows `r` that are multiples of `2h` eliminate their
//!    neighbours `r - h` and `r + h`, which leaves them coupled to the rows at distance `2h`. The
//!    neighbours are rows that are not modified at this level, so the update is in place and one
//!    barrier per level suffices. Row 0 ends with `x_0 = f_0`;
//! 3. backward levels, `h` halving: the rows `h (2q + 1)` follow from
//!    `x_r = f_r - L_r x_{r-h} - U_r x_{r+h}` with their own, never overwritten, `L`, `U` and `f`.
//!
//! No pivoting across rows, so like the Thomas solve this wants diagonally dominant systems.
//!
//! A row takes `2 B^2 + B` floats (rounded up to an odd count so that the rows of a level, `2h`
//! rows apart, fall on different banks as far as they can). The rows live in workgroup memory up
//! to [`Tuning::line_bytes`]; if a line has more, only the rows that are a multiple of `2^shift`
//! do, and the others (touched in the first and last `shift` levels only) live in a global scratch
//! buffer, indexed compactly. See [`plan`] for when this beats the Thomas solve.

use std::fmt::Write;

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
        u64::from(lines) * u64::from(n - self.shared_rows) * u64::from(row_floats(block)) * 4
    }
}

/// Floats of one row of `block`-blocks: `L`, `U` and `f`, rounded up to an odd number.
pub(crate) fn row_floats(block: u32) -> u32 {
    (2 * block * block + block) | 1
}

/// The shift (log2 of the row stride) of the shared rows for a line of `n` rows, and the number
/// of shared rows.
fn layout(tuning: &Tuning, block: u32, n: u32) -> (u8, u32) {
    let row = u64::from(row_floats(block)) * 4;
    let budget = u64::from(tuning.line_bytes);
    let mut shift = 0u8;
    while shift < 31 && u64::from(n.div_ceil(1 << shift)) * row > budget && n.div_ceil(1 << shift) > 1 {
        shift += 1;
    }
    (shift, n.div_ceil(1 << shift))
}

/// Whether the line solve of `lines` lines of `n` cells of `block`-blocks, `stride` cells apart
/// in memory, runs by cyclic reduction, and how. `None` means Thomas. Forced
/// (`LUMA_TRIDIAGONAL=cyclic`) it is taken whenever the device can run it; automatically only for
/// few lines (see [`Tuning::line_cyclic_work`]), where the one invocation per line of the Thomas
/// solve leaves the device idle. `max_scratch` is the largest buffer the spill may take.
///
/// Measured on an RTX 3090 (block 1, 2, 4, 8, `n` 32 to 4096, 4 to 16384 lines, both axes): the
/// workgroup per line beats Thomas by 2x to 50x up to a few hundred lines for every block size
/// and length, and keeps winning with more lines when the blocks are small (scalar lines along
/// the contiguous axis to 16384 lines), and loses (1.2x to 7x slower) once there are thousands of
/// lines of strided scalars or 1024 lines of 4x4 blocks, where one thread per line already fills
/// the device and its coalesced loads across lines beat a workgroup's strided ones. The rule
/// `lines * block^2 <= 8192` for contiguous lines and `lines * 4 * block <= 8192` for strided
/// ones keeps to the clear wins of both regimes (scalar 2048 / 4096 lines, 4x4 blocks 512 / 512
/// lines, 8x8 blocks 128 / 256 lines).
pub(crate) fn plan(tuning: &Tuning, block: u32, n: u32, lines: u32, stride: u32, max_scratch: u64) -> Option<Plan> {
    if tuning.line_solver == LineSolver::Thomas || n == 0 || lines == 0 {
        return None;
    }
    let (shift, shared_rows) = layout(tuning, block, n);
    let workgroup = workgroup_size(tuning, block, n);
    let plan = Plan { workgroup, shift, shared_rows };
    // A single row must fit the budget, and the scratch a buffer.
    if u64::from(shared_rows) * u64::from(row_floats(block)) * 4 > u64::from(tuning.line_bytes) || plan.scratch_bytes(block, n, lines) > max_scratch {
        return None;
    }
    let scratch_elements = u64::from(lines) * u64::from(n - shared_rows) * u64::from(row_floats(block));
    if scratch_elements > u64::from(u32::MAX) {
        return None;
    }
    if tuning.line_solver == LineSolver::Auto {
        let weight = if stride == 1 { block * block } else { 4 * block };
        if u64::from(lines) * u64::from(weight) > u64::from(tuning.line_cyclic_work) {
            return None;
        }
    }
    Some(plan)
}

/// Invocations per workgroup: one per two rows of the line (the first forward level), from a
/// warp of 32 up to a cap that falls with the block size, as the registers of one row's solve grow
/// with `block^2` (RTX 3090, `tri_probe`: 256 is best for scalar lines, 128 for blocks of 2 to 5,
/// 64 for bigger ones; each is up to 2x faster than the next size up for long lines).
fn workgroup_size(tuning: &Tuning, block: u32, n: u32) -> u16 {
    let cap = match block {
        1 => 256,
        2..=5 => 128,
        _ => 64,
    };
    // Blocks of 3 and more: the per-row solve is long and serial, so shorter lines want fewer
    // threads (64 for 100 to 400 rows).
    let rows_per_thread = if block >= 3 { 8 } else { 2 };
    let wanted = n.div_ceil(rows_per_thread).next_power_of_two().clamp(32, cap);
    wanted.min(tuning.elementwise_workgroup.max(1)) as u16
}

/// Bindings of the kernel, in order: `out`, the scratch `sm` (with spill), `lo`, `diag`, `up`,
/// `rhs`, `starts` (with `starts`), the uniform parameters `{n, stride, lines}`.
pub(crate) fn source(block: u32, starts: bool, workgroup: u32, shift: u32, shared_rows: u32) -> String {
    let b = block as usize;
    let (bb, w) = (b * b, 2 * b + 1);
    let row = row_floats(block) as usize;
    let spill = shift > 0;
    let mut binding = 0;
    let mut next = || {
        binding += 1;
        binding - 1
    };
    let mut src = String::new();
    let _ = writeln!(src, "struct Params {{ n: u32, stride: u32, lines: u32 }}");
    let _ = writeln!(src, "@group(1) @binding({}) var<storage, read_write> out: array<f32>;", next());
    if spill {
        let _ = writeln!(src, "@group(1) @binding({}) var<storage, read_write> sm: array<f32>;", next());
    }
    for name in ["lo", "diag", "up", "rhs"] {
        let _ = writeln!(src, "@group(1) @binding({}) var<storage, read> {name}: array<f32>;", next());
    }
    if starts {
        let _ = writeln!(src, "@group(1) @binding({}) var<storage, read> starts: array<u32>;", next());
    }
    let _ = writeln!(src, "@group(1) @binding({}) var<uniform> p: Params;", next());
    let _ = writeln!(
        src,
        "const WG: u32 = {workgroup}u;
const ROW: u32 = {row}u;
var<workgroup> sh: array<f32, {}>;
fn line_base(l: u32) -> u32 {{ return (l % p.stride) + (l / p.stride) * p.stride * p.n; }}",
        shared_rows as usize * row
    );
    if spill {
        let ks = 1u32 << shift;
        let _ = writeln!(
            src,
            "const KS: u32 = {ks}u;
var<private> sbase: u32;
fn ld(r: u32, e: u32) -> f32 {{
    if ((r & {mask}u) == 0u) {{ return sh[(r >> {shift}u) * ROW + e]; }}
    return sm[sbase + (r - (r >> {shift}u) - 1u) * ROW + e];
}}
fn st(r: u32, e: u32, v: f32) {{
    if ((r & {mask}u) == 0u) {{ sh[(r >> {shift}u) * ROW + e] = v; return; }}
    sm[sbase + (r - (r >> {shift}u) - 1u) * ROW + e] = v;
}}",
            mask = ks - 1
        );
    } else {
        src.push_str(
            "fn ld(r: u32, e: u32) -> f32 { return sh[r * ROW + e]; }
fn st(r: u32, e: u32, v: f32) { sh[r * ROW + e] = v; }
",
        );
    }
    let s0 = if starts { "min(starts[l], p.n)" } else { "0u" };
    let sbase = if spill { format!("    sbase = l * ((p.n - {shared_rows}u) * ROW);\n") } else { String::new() };
    let _ = write!(
        src,
        "@compute @workgroup_size({workgroup})
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_index) li: u32) {{
    let l = wid.x + wid.y * nwg.x;
    if (l >= p.lines) {{ return; }}
    let base = line_base(l);
    let s0 = {s0};
    for (var s = li; s < s0; s += WG) {{
        let oq = (base + s * p.stride) * {b}u;
"
    );
    for c in 0..b {
        let _ = writeln!(src, "        out[oq + {c}u] = 0.0;");
    }
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

    // Normalise every row.
    src.push_str("    for (var r = li; r < m; r += WG) {\n        let q = base + (s0 + r) * p.stride;\n");
    let _ = writeln!(src, "        let o = q * {bb}u;\n        let oq = q * {b}u;");
    for i in 0..b {
        for j in 0..b {
            let _ = writeln!(src, "        var m{i}_{j} = diag[o + {}u];", i * b + j);
            let _ = writeln!(src, "        var r{i}_{j} = select(0.0, lo[o + {}u], r > 0u);", i * b + j);
            let _ = writeln!(src, "        var r{i}_{} = select(0.0, up[o + {}u], r + 1u < m);", b + j, i * b + j);
        }
        let _ = writeln!(src, "        var r{i}_{} = rhs[oq + {i}u];", 2 * b);
    }
    solve(&mut src, "        ", b, w);
    store_row(&mut src, "        ", "r", b);
    src.push_str("    }\n");
    src.push_str(&sync("true").replace("        ", "    "));

    // Forward levels.
    src.push_str("    var h = 1u;\n    for (; h < m; h = h << 1u) {\n");
    src.push_str("        for (var r = 2u * h * li; r < m; r += 2u * h * WG) {\n");
    let ind = "            ";
    for i in 0..b {
        for j in 0..b {
            let _ = writeln!(src, "{ind}let lk{i}_{j} = ld(r, {}u);", i * b + j);
            let _ = writeln!(src, "{ind}let uk{i}_{j} = ld(r, {}u);", bb + i * b + j);
            let _ = writeln!(src, "{ind}var m{i}_{j} = {};", if i == j { "1.0" } else { "0.0" });
            let _ = writeln!(src, "{ind}var r{i}_{j} = 0.0;\n{ind}var r{i}_{} = 0.0;", b + j);
        }
        let _ = writeln!(src, "{ind}var r{i}_{} = ld(r, {}u);", 2 * b, 2 * bb + i);
    }
    for (side, cond, neighbour, factor) in [("left", "r > 0u", "r - h", "lk"), ("right", "r + h < m", "r + h", "uk")] {
        // `d -= factor * (opposite coupling of the neighbour)`, `(own side) = -factor * (same side of the neighbour)`.
        let (to_d, to_r_col) = if side == "left" { (bb, 0) } else { (0, b) };
        let (same, f_off) = if side == "left" { (0, 2 * bb) } else { (bb, 2 * bb) };
        let _ = writeln!(src, "{ind}if ({cond}) {{\n{ind}    let nb = {neighbour};");
        for i in 0..b {
            for j in 0..b {
                let sum_d = (0..b).map(|k| format!("{factor}{i}_{k} * ld(nb, {}u)", to_d + k * b + j)).collect::<Vec<_>>().join(" + ");
                let sum_r = (0..b).map(|k| format!("{factor}{i}_{k} * ld(nb, {}u)", same + k * b + j)).collect::<Vec<_>>().join(" + ");
                let _ = writeln!(src, "{ind}    m{i}_{j} -= {sum_d};\n{ind}    r{i}_{} = -({sum_r});", to_r_col + j);
            }
            let sum_f = (0..b).map(|k| format!("{factor}{i}_{k} * ld(nb, {}u)", f_off + k)).collect::<Vec<_>>().join(" + ");
            let _ = writeln!(src, "{ind}    r{i}_{} -= {sum_f};", 2 * b);
        }
        let _ = writeln!(src, "{ind}}}");
    }
    solve(&mut src, ind, b, w);
    store_row(&mut src, ind, "r", b);
    src.push_str("        }\n");
    src.push_str(&sync("2u * h < KS"));
    src.push_str("    }\n");

    // Backward levels.
    src.push_str("    while (h > 1u) {\n        h = h >> 1u;\n        for (var r = h + 2u * h * li; r < m; r += 2u * h * WG) {\n");
    for k in 0..b {
        let _ = writeln!(src, "{ind}let xl{k} = ld(r - h, {}u);", 2 * bb + k);
    }
    for i in 0..b {
        let sum = (0..b).map(|k| format!("ld(r, {}u) * xl{k}", i * b + k)).collect::<Vec<_>>().join(" + ");
        let _ = writeln!(src, "{ind}var x{i} = ld(r, {}u) - ({sum});", 2 * bb + i);
    }
    let _ = writeln!(src, "{ind}if (r + h < m) {{");
    for k in 0..b {
        let _ = writeln!(src, "{ind}    let xr{k} = ld(r + h, {}u);", 2 * bb + k);
    }
    for i in 0..b {
        let sum = (0..b).map(|k| format!("ld(r, {}u) * xr{k}", bb + i * b + k)).collect::<Vec<_>>().join(" + ");
        let _ = writeln!(src, "{ind}    x{i} -= {sum};");
    }
    let _ = writeln!(src, "{ind}}}");
    for i in 0..b {
        let _ = writeln!(src, "{ind}st(r, {}u, x{i});", 2 * bb + i);
    }
    src.push_str("        }\n");
    src.push_str(&sync("h < KS"));
    src.push_str("    }\n");

    // The solution.
    src.push_str("    for (var r = li; r < m; r += WG) {\n        let oq = (base + (s0 + r) * p.stride) * ");
    let _ = writeln!(src, "{b}u;");
    for i in 0..b {
        let _ = writeln!(src, "        out[oq + {i}u] = ld(r, {}u);", 2 * bb + i);
    }
    src.push_str("    }\n}\n");
    src
}

/// Stores the solved columns `r{i}_{j}` (`L`, `U`, `f`) as the row `r`.
fn store_row(src: &mut String, ind: &str, name: &str, b: usize) {
    let bb = b * b;
    for i in 0..b {
        for j in 0..b {
            let _ = writeln!(src, "{ind}st(r, {}u, {name}{i}_{j});", i * b + j);
            let _ = writeln!(src, "{ind}st(r, {}u, {name}{i}_{});", bb + i * b + j, b + j);
        }
        let _ = writeln!(src, "{ind}st(r, {}u, {name}{i}_{});", 2 * bb + i, 2 * b);
    }
}

/// Solves `m X = r` in place for the scalars `m{i}_{j}` (`b x b`) and `r{i}_{j}` (`b x w`):
/// Gaussian elimination with partial pivoting (the pivot row is found and exchanged with
/// `select`s, nothing branches; the first row of largest magnitude wins ties), then back
/// substitution. `r` holds `X` afterwards.
fn solve(src: &mut String, ind: &str, b: usize, w: usize) {
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
