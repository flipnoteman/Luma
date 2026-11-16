//! What the line-solver kernels (`builtin::tridiagonal_body`, `cyclic::body`) read and write, and
//! the solves on parts.
//!
//! A kernel solves `systems` independent block-tridiagonal systems of `block`-blocks on the same
//! grid, one per grid line and system. It reaches its data through an [`Io`]: either the
//! interleaved buffers of [`solve_block_tridiagonal`](crate::solve_block_tridiagonal)
//! ([`Io::buffers`]), or the *parts* of the `_parts` solvers ([`Io::parts`]), where every entry of
//! every block is an array of its own, computed in the kernel (by the function `luma_cell`, see
//! `jit::codegen`) or read from its buffer, and every component of the solution is written to a
//! buffer of its own.

use std::fmt::Write;

use crate::cyclic;

/// The kernel's access to the coefficients, right-hand sides and solution.
#[derive(Clone, Copy)]
pub(crate) struct Io {
    block: usize,
    systems: usize,
    parts: bool,
}

impl Io {
    /// The buffers `lo`, `diag`, `up` (`block^2` floats per cell, `array<vec4<f32>>` if that is a
    /// multiple of 4), `rhs` and `out` (`block` floats per cell, likewise); one system.
    pub(crate) fn buffers(block: usize) -> Io {
        Io { block, systems: 1, parts: false }
    }

    /// The parts of `systems` systems: coefficients from the `luma_cell(sys, q)` function (see
    /// [`Io::parts_per_system`] for its result), the solution in `out{system * block + component}`.
    pub(crate) fn parts(block: usize, systems: usize) -> Io {
        Io { block, systems, parts: true }
    }

    /// Parts of one system: `lo`, `diag`, `up` (`block^2` each, entry `(row, col)` at
    /// `row * block + col`) and `rhs` (`block`), in this order.
    pub(crate) fn parts_per_system(block: usize) -> usize {
        3 * block * block + block
    }

    fn vec_coef(&self) -> bool {
        !self.parts && (self.block * self.block).is_multiple_of(4)
    }

    fn vec_rhs(&self) -> bool {
        !self.parts && self.block.is_multiple_of(4)
    }

    /// The WGSL element type the buffer `lo`, `diag` or `up` is declared with.
    pub(crate) fn coef_type(&self) -> &'static str {
        if self.vec_coef() { "vec4<f32>" } else { "f32" }
    }

    /// The WGSL element type of `rhs` and `out`.
    pub(crate) fn rhs_type(&self) -> &'static str {
        if self.vec_rhs() { "vec4<f32>" } else { "f32" }
    }

    /// Statements that make the diagonal and upper blocks and the right-hand side of cell `q`
    /// available to the accessors below. The `vec4` buffers are read whole into locals: an access
    /// `diag[i].y` would load one float.
    pub(crate) fn load(&self, ind: &str, q: &str) -> String {
        if self.parts {
            return format!("{ind}let cell = luma_cell(sys, {q});\n");
        }
        let (b, bb) = (self.block, self.block * self.block);
        let mut src = String::new();
        if self.vec_coef() {
            for (name, local) in [("diag", "bd"), ("up", "bu")] {
                for v in 0..bb / 4 {
                    let _ = writeln!(src, "{ind}let {local}{v} = {name}[{q} * {}u + {v}u];", bb / 4);
                }
            }
        }
        if self.vec_rhs() {
            for v in 0..b / 4 {
                let _ = writeln!(src, "{ind}let br{v} = rhs[{q} * {}u + {v}u];", b / 4);
            }
        }
        src
    }

    /// Statements that make the lower block of cell `q` available (see [`Io::load`]); the parts
    /// come along with the others.
    pub(crate) fn load_lo(&self, ind: &str, q: &str) -> String {
        let bb = self.block * self.block;
        let mut src = String::new();
        if self.vec_coef() {
            for v in 0..bb / 4 {
                let _ = writeln!(src, "{ind}let bl{v} = lo[{q} * {}u + {v}u];", bb / 4);
            }
        }
        src
    }

    fn coef(&self, buffer: &str, local: &str, offset: usize, q: &str, k: usize) -> String {
        let bb = self.block * self.block;
        if self.parts {
            format!("cell[{}]", offset + k)
        } else if self.vec_coef() {
            format!("{local}{}.{}", k / 4, ["x", "y", "z", "w"][k % 4])
        } else {
            format!("{buffer}[{q} * {bb}u + {k}u]")
        }
    }

    /// The entry `k` (`row * block + col`) of the lower, diagonal or upper block of cell `q`
    /// (after [`Io::load_lo`], [`Io::load`] for the same `q`).
    pub(crate) fn lo(&self, q: &str, k: usize) -> String {
        self.coef("lo", "bl", 0, q, k)
    }

    pub(crate) fn diag(&self, q: &str, k: usize) -> String {
        self.coef("diag", "bd", self.block * self.block, q, k)
    }

    pub(crate) fn up(&self, q: &str, k: usize) -> String {
        self.coef("up", "bu", 2 * self.block * self.block, q, k)
    }

    /// Component `i` of the right-hand side of cell `q`.
    pub(crate) fn rhs(&self, q: &str, i: usize) -> String {
        let b = self.block;
        if self.parts {
            format!("cell[{}]", 3 * b * b + i)
        } else if self.vec_rhs() {
            format!("br{}.{}", i / 4, ["x", "y", "z", "w"][i % 4])
        } else {
            format!("rhs[{q} * {b}u + {i}u]")
        }
    }

    /// Statements writing the `block` components `values` of the solution of cell `q`.
    pub(crate) fn store(&self, ind: &str, q: &str, values: &[String]) -> String {
        let b = self.block;
        let mut src = String::new();
        if self.parts {
            for s in 0..self.systems {
                let guard = if self.systems > 1 { format!("if (sys == {s}u) ") } else { String::new() };
                let body: String = values.iter().enumerate().map(|(c, v)| format!("out{}[{q}] = {v}; ", s * b + c)).collect();
                let _ = writeln!(src, "{ind}{guard}{{ {body}}}");
            }
        } else if self.vec_rhs() {
            for (v, lanes) in values.chunks(4).enumerate() {
                let _ = writeln!(src, "{ind}out[{q} * {}u + {v}u] = vec4<f32>({});", b / 4, lanes.join(", "));
            }
        } else {
            for (c, v) in values.iter().enumerate() {
                let _ = writeln!(src, "{ind}out[{q} * {b}u + {c}u] = {v};");
            }
        }
        src
    }

    /// Statements declaring `{name}0..` as the components of the solution of cell `q` written
    /// earlier.
    pub(crate) fn load_out(&self, ind: &str, q: &str, name: &str) -> String {
        let b = self.block;
        let mut src = String::new();
        if self.parts {
            for c in 0..b {
                let _ = writeln!(src, "{ind}var {name}{c}: f32;");
            }
            for s in 0..self.systems {
                let loads: String = (0..b).map(|c| format!("{name}{c} = out{}[{q}]; ", s * b + c)).collect();
                let head = match (s, self.systems) {
                    (_, 1) => String::new(),
                    (0, _) => format!("if (sys == {s}u) "),
                    (s, n) if s + 1 == n => "else ".to_owned(),
                    (s, _) => format!("else if (sys == {s}u) "),
                };
                let _ = writeln!(src, "{ind}{head}{{ {loads}}}");
            }
        } else if self.vec_rhs() {
            for v in 0..b / 4 {
                let _ = writeln!(src, "{ind}let {name}v{v} = out[{q} * {}u + {v}u];", b / 4);
            }
            for c in 0..b {
                let _ = writeln!(src, "{ind}let {name}{c} = {name}v{}.{};", c / 4, ["x", "y", "z", "w"][c % 4]);
            }
        } else {
            for c in 0..b {
                let _ = writeln!(src, "{ind}let {name}{c} = out[{q} * {b}u + {c}u];");
            }
        }
        src
    }
}

/// How the lines of a system are solved.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum Algo {
    /// One invocation per line, `workgroup` per workgroup, the scratch `tiled` or not
    /// ([`Builtin::Tridiagonal`](crate::builtin::Builtin), [`thomas_plan`](crate::builtin::thomas_plan)).
    Thomas { workgroup: u8, tiled: bool },
    /// One workgroup per line ([`cyclic`]).
    Cyclic { workgroup: u16, shift: u8, shared_rows: u32 },
}

/// A solve of `systems` independent systems of `block`-blocks on one grid, on parts (see the
/// module documentation). Part `k` of system `s` is part `s * (3 block^2 + block) + k` of the
/// kernel, in the order of [`Io::parts_per_system`]; `fused` says for each part (bit `part % 64` of
/// word `part / 64`; 256 parts: 8x8 blocks have 200, 64 scalar systems 256) whether it is computed in the kernel (`luma_cell`) rather than read from the
/// buffer `pr{part}`.
///
/// Bindings, in order: the outputs `out{system * block + component}`, the scratch (`cp` for
/// [`Algo::Thomas`], `sm` for a [`Algo::Cyclic`] that spills), the inputs of the fused parts, the
/// buffers of the other parts, `starts` if `starts`, the uniform `{n, stride, lines}` and the
/// scalars of the fused parts.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) struct SolveSpec {
    pub(crate) algo: Algo,
    pub(crate) block: u8,
    pub(crate) systems: u8,
    pub(crate) starts: bool,
    pub(crate) fused: [u64; 4],
}

impl SolveSpec {
    fn per_system(&self) -> usize {
        Io::parts_per_system(usize::from(self.block))
    }

    /// Parts of the whole kernel.
    pub(crate) fn parts(&self) -> usize {
        self.per_system() * usize::from(self.systems)
    }

    pub(crate) fn is_fused(&self, part: usize) -> bool {
        self.fused[part / 64] >> (part % 64) & 1 == 1
    }

    pub(crate) fn fuse(&mut self, part: usize) {
        self.fused[part / 64] |= 1 << (part % 64);
    }

    /// Declares the outputs and the scratch, numbered from `*binding`.
    pub(crate) fn declare_writes(&self, src: &mut String, binding: &mut usize) {
        for j in 0..usize::from(self.systems) * usize::from(self.block) {
            let _ = writeln!(src, "@group(1) @binding({binding}) var<storage, read_write> out{j}: array<f32>;", binding = *binding);
            *binding += 1;
        }
        let scratch = match self.algo {
            Algo::Thomas { .. } => Some(("cp", "f32")),
            Algo::Cyclic { shift, .. } => (shift > 0).then_some(("sm", cyclic::row_type(u32::from(self.block)))),
        };
        if let Some((name, ty)) = scratch {
            let _ = writeln!(src, "@group(1) @binding({binding}) var<storage, read_write> {name}: array<{ty}>;", binding = *binding);
            *binding += 1;
        }
    }

    /// Declares the buffers of the parts that are not fused and `starts`.
    pub(crate) fn declare_reads(&self, src: &mut String, binding: &mut usize) {
        for part in (0..self.parts()).filter(|&part| !self.is_fused(part)) {
            let _ = writeln!(src, "@group(1) @binding({binding}) var<storage, read> pr{part}: array<f32>;", binding = *binding);
            *binding += 1;
        }
        if self.starts {
            let _ = writeln!(src, "@group(1) @binding({binding}) var<storage, read> starts: array<u32>;", binding = *binding);
            *binding += 1;
        }
    }

    /// `luma_cell(sys, i)`: the parts of cell `i` of system `sys`, from the functions
    /// `luma_cell{system}(i)`.
    pub(crate) fn cell_dispatch(&self) -> String {
        let (per, systems) = (self.per_system(), usize::from(self.systems));
        let mut src = format!("\nfn luma_cell(sys: u32, i: u32) -> array<f32, {per}> {{\n");
        for s in 0..systems - 1 {
            let _ = writeln!(src, "    if (sys == {s}u) {{ return luma_cell{s}(i); }}");
        }
        let _ = writeln!(src, "    return luma_cell{}(i);\n}}", systems - 1);
        src
    }

    /// The entry point, which reads the parts through `luma_cell` and writes the outputs.
    pub(crate) fn body(&self) -> String {
        let (block, systems) = (usize::from(self.block), u32::from(self.systems));
        let io = Io::parts(block, usize::from(self.systems));
        match self.algo {
            Algo::Cyclic { workgroup, shift, shared_rows } => cyclic::body(
                &cyclic::Config { block: u32::from(self.block), workgroup: u32::from(workgroup), shift: u32::from(shift), shared_rows, starts: self.starts, systems },
                &io,
            ),
            Algo::Thomas { workgroup, tiled } => crate::builtin::tridiagonal_body(block, self.starts, u32::from(workgroup), systems, &io, tiled),
        }
    }

    /// The whole kernel of [`Builtin::SolveParts`](crate::builtin::Builtin): every part is read
    /// from its buffer.
    pub(crate) fn source(&self) -> String {
        let mut src = String::from("struct Params { n: u32, stride: u32, lines: u32 }\n");
        let mut binding = 0;
        self.declare_writes(&mut src, &mut binding);
        self.declare_reads(&mut src, &mut binding);
        let _ = writeln!(src, "@group(1) @binding({binding}) var<uniform> p: Params;");
        let per = self.per_system();
        for s in 0..usize::from(self.systems) {
            let values: Vec<String> = (0..per).map(|k| format!("pr{}[i]", s * per + k)).collect();
            let _ = writeln!(src, "fn luma_cell{s}(i: u32) -> array<f32, {per}> {{ return array<f32, {per}>({}); }}", values.join(", "));
        }
        src.push_str(&self.cell_dispatch());
        src.push_str(&self.body());
        src
    }
}
