//! WGSL generation for fused programs.
//!
//! The source depends on the [`FusedKey`] only. Bindings, in order of the arguments: the outputs
//! (`out0`, `out1`, ...; a reduction has one, holding the partial results of every op), the
//! input buffers (`in0`, `in1`, ...; the one a key marks `in_place` is `out0` instead and gets no
//! binding), then, if the program has scalars (or is a reduction), one
//! uniform `p` holding them (`p.s0`, `p.s1`, ... after `p.n` for reductions).
//!
//! The segmented variant (`key.spans` not empty) runs once per segment of arrays spanning several
//! buffers and has shifts that leave their segment. `p` then also holds `base` (the flat index of
//! the segment's first element), `total` (the array length) and, per input read through shifts,
//! the ids of the segments bound as `in{k}_0 ..`; `ld{k}` loads a flat index through them. An
//! input is bound as `in{k}` there only if the program also loads it at `i`.
//!
//! The program loads the inputs at element `i` and runs the instructions as `let` bindings. A
//! store kernel writes its output instructions; a reduction wraps the program in `luma_eval(i)`
//! and feeds the last value into the tree reductions of the first pass.

use std::fmt::Write;

use crate::builtin::reduce_main;
use crate::tuning::{Tuning, STENCIL_RADIUS};

use super::program::{FusedKey, Instr, Ref, Tail, ELEMENT_BYTES};
use super::stencil::{signed, wrap_source, WORKGROUP};
use super::{ElemTy, Gen, OpKind, ShiftKind};

/// `template` with its operand identifiers `x` and `y` replaced by `a` and `b`.
fn substitute(template: &str, a: &str, b: &str) -> String {
    let mut out = String::with_capacity(template.len() + 16);
    let mut word = String::new();
    let flush = |word: &mut String, out: &mut String| {
        match word.as_str() {
            "x" => out.push_str(a),
            "y" => out.push_str(b),
            other => out.push_str(other),
        }
        word.clear();
    };
    for c in template.chars() {
        if c.is_ascii_alphanumeric() || c == '_' {
            word.push(c);
        } else {
            flush(&mut word, &mut out);
            out.push(c);
        }
    }
    flush(&mut word, &mut out);
    out
}

fn name(reference: Ref) -> String {
    match reference {
        Ref::Unused => "0u".to_owned(),
        Ref::Input(k) => format!("a{k}"),
        Ref::Scalar(k) => format!("p.s{k}"),
        Ref::Tmp(k) => format!("t{k}"),
        Ref::Sub(k) => format!("sub{k}"),
    }
}

/// The WGSL expression of `instr`. `index` names the element a shift is relative to and `total`
/// the array length; `segmented` makes shifts of inputs read through the accessors `ld{k}`
/// (which take a flat index over all segments) instead of indexing `in{k}` directly.
pub(super) fn expression(instr: &Instr, index: &str, total: &str, segmented: bool) -> String {
    let [a, b, c] = instr.args.map(name);
    match instr.op.kind {
        OpKind::Unary(op) => substitute(op.expr(), &a, ""),
        OpKind::Binary(op) => substitute(op.expr(), &a, &b),
        OpKind::Compare(op) => substitute(op.expr(), &a, &b),
        OpKind::Powf => format!("pow({a}, {b})"),
        OpKind::Clamp => format!("clamp({a}, {b}, {c})"),
        OpKind::Select => format!("select({c}, {b}, {a} != 0u)"),
        OpKind::Generate(kind) => generate(kind, instr.op.out_ty(), index, &a, &b),
        OpKind::Shift(kind) => {
            let at = shift_index(kind, &b, index, total);
            match instr.args[0] {
                Ref::Input(k) if segmented => format!("ld{k}({at})"),
                Ref::Input(k) => format!("in{k}[{at}]"),
                Ref::Sub(k) => format!("sub{k}({at})"),
                // A shift only reads an input buffer or a sub-program; anything else is not
                // valid shader code and surfaces as a shader compile error.
                other => format!("{}[{at}]", name(other)),
            }
        }
    }
}

fn generate(kind: Gen, ty: ElemTy, index: &str, a: &str, b: &str) -> String {
    match kind {
        Gen::Bits => format!("luma_threefry({index}, {a}, {b}).x"),
        Gen::Uniform => format!("luma_uniform(luma_threefry({index}, {a}, {b}).x)"),
        Gen::Normal => format!("luma_normal(luma_threefry({index}, {a}, {b}))"),
        Gen::Range { stride, size } => match ty {
            ElemTy::F32 => format!("f32(({index} / {stride}u) % {size}u)"),
            ElemTy::U32 => format!("(({index} / {stride}u) % {size}u)"),
        },
        Gen::Linspace => format!("({a} + f32({index}) * {b})"),
    }
}

/// Rotation amounts of Threefry-2x32 (Random123).
const THREEFRY_ROTATIONS: [u32; 8] = [13, 15, 26, 6, 17, 29, 16, 24];

/// The WGSL helpers of the random generators, if any instruction of `programs` uses them.
pub(super) fn generator_helpers<'a>(programs: impl IntoIterator<Item = &'a [Instr]>) -> String {
    let random = programs.into_iter().flatten().any(|instr| {
        matches!(instr.op.kind, OpKind::Generate(Gen::Bits | Gen::Uniform | Gen::Normal))
    });
    if !random {
        return String::new();
    }
    let keys = ["k0", "k1", "k2"];
    let mut src =
        String::from("fn luma_threefry(c: u32, k0: u32, k1: u32) -> vec2<u32> {\n    let k2 = 0x1BD11BDAu ^ k0 ^ k1;\n    var x0 = c + k0;\n    var x1 = k1;\n");
    for round in 0..20 {
        let r = THREEFRY_ROTATIONS[round % 8];
        let _ = writeln!(src, "    x0 = x0 + x1; x1 = ((x1 << {r}u) | (x1 >> {}u)) ^ x0;", 32 - r);
        if round % 4 == 3 {
            let s = round / 4 + 1;
            let _ = writeln!(src, "    x0 = x0 + {}; x1 = x1 + {} + {s}u;", keys[s % 3], keys[(s + 1) % 3]);
        }
    }
    src.push_str(
        "    return vec2<u32>(x0, x1);
}
fn luma_uniform(w: u32) -> f32 { return f32(w >> 8u) * 5.9604645e-8; }
fn luma_normal(w: vec2<u32>) -> f32 {
    let u1 = (f32(w.x >> 8u) + 1.0) * 5.9604645e-8;
    return sqrt(-2.0 * log(u1)) * cos(6.2831855 * f32(w.y >> 8u) * 5.9604645e-8);
}
",
    );
    src
}

/// Whether a program uses the element's index in the whole array, so a kernel run per segment
/// needs the segment's first index.
pub(super) fn generates(key: &FusedKey) -> bool {
    key.instrs.iter().chain(key.subs.iter().flatten()).any(|instr| matches!(instr.op.kind, OpKind::Generate(_)))
}

/// The index element `i` of a shift reads; `s` names the scalar shift of [`ShiftKind::Linear`],
/// `i` the element the shift is relative to and `total` the array length.
fn shift_index(kind: ShiftKind, s: &str, i: &str, total: &str) -> String {
    match kind {
        ShiftKind::Linear => format!("select({i} - {s}, {i} + ({total} - {s}), {i} < {s})"),
        ShiftKind::Dims { dims, shifts } => {
            let [d0, d1, d2, _] = dims;
            let strides = [1, d0, d0 * d1, d0 * d1 * d2];
            let mut terms = Vec::new();
            for axis in 0..4 {
                let (dim, shift, stride) = (dims[axis], shifts[axis], strides[axis]);
                if dim == 1 {
                    continue;
                }
                let coord = match (stride, axis) {
                    (1, _) => format!("({i} % {dim}u)"),
                    (_, 3) => format!("({i} / {stride}u)"),
                    _ => format!("(({i} / {stride}u) % {dim}u)"),
                };
                let wrapped = if shift == 0 {
                    coord
                } else {
                    format!("select({coord} - {shift}u, {coord} + {}u, {coord} < {shift}u)", dim - shift)
                };
                terms.push(if stride == 1 { wrapped } else { format!("{wrapped} * {stride}u") });
            }
            if terms.is_empty() {
                "0u".to_owned()
            } else {
                terms.join(" + ")
            }
        }
    }
}

/// Whether input `k` is loaded at the element being computed (as opposed to only through
/// shifts), so the kernel declares its buffer as `in{k}`. The launch code binds by the same rule.
pub(super) fn direct(instrs: &[Instr], k: u32) -> bool {
    instrs.iter().any(|instr| {
        let skip = usize::from(matches!(instr.op.kind, OpKind::Shift(_)));
        instr.args.iter().skip(skip).any(|arg| *arg == Ref::Input(k))
    })
}

/// The WGSL source of the kernel for `key`, with the workgroup shapes `tuning` gives.
/// `segment_len` (a power of two) is the elements per segment of the arrays the segmented
/// variant (`key.spans` not empty) runs on.
pub(crate) fn source(key: &FusedKey, tuning: &Tuning, segment_len: u32) -> String {
    let wg = tuning.elementwise_workgroup;
    let reduce = matches!(key.tail, Tail::Reduce(_));
    let segmented = !key.spans.is_empty();
    let slots = |k: usize| key.spans.get(k).copied().unwrap_or(0);
    let based = generates(key) && !segmented;
    let uniform = reduce || !key.scalar_tys.is_empty() || segmented || based;
    let mut src = String::new();

    if uniform {
        src.push_str("struct Params { ");
        if reduce {
            src.push_str("n: u32, ");
        }
        if segmented {
            src.push_str("base: u32, total: u32, ");
            for k in 0..key.input_tys.len() {
                for s in 0..slots(k) {
                    let _ = write!(src, "c{k}_{s}: u32, ");
                }
            }
        } else if based {
            src.push_str("base: u32, ");
        }
        for (k, ty) in key.scalar_tys.iter().enumerate() {
            let _ = write!(src, "s{k}: {}, ", ty.wgsl());
        }
        src.push_str("}\n");
    }
    let out_ty = |at: usize| key.instrs.get(at).map_or(ElemTy::F32, |instr| instr.op.out_ty()).wgsl();
    let outputs: Vec<usize> = match key.tail {
        Tail::Store => key.outputs.iter().map(|&at| at as usize).collect(),
        Tail::Reduce(_) => vec![key.instrs.len().saturating_sub(1)],
    };
    for (j, &at) in outputs.iter().enumerate() {
        // A reduction's single output is `out`, as `reduce_main` expects.
        let name = if reduce { "out".to_owned() } else { format!("out{j}") };
        let _ = writeln!(src, "@group(1) @binding({j}) var<storage, read_write> {name}: array<{}>;", out_ty(at));
    }
    let mut binding = outputs.len();
    for (k, ty) in key.input_tys.iter().enumerate() {
        if key.in_place == Some(k as u32) {
            continue;
        }
        if !segmented || direct(&key.instrs, k as u32) {
            let _ = writeln!(src, "@group(1) @binding({binding}) var<storage, read> in{k}: array<{}>;", ty.wgsl());
            binding += 1;
        }
        for s in 0..slots(k) {
            let _ = writeln!(src, "@group(1) @binding({binding}) var<storage, read> in{k}_{s}: array<{}>;", ty.wgsl());
            binding += 1;
        }
    }
    if uniform {
        let _ = writeln!(src, "@group(1) @binding({binding}) var<uniform> p: Params;");
    }
    if segmented {
        let (log, mask) = (segment_len.trailing_zeros(), segment_len.wrapping_sub(1));
        for (k, ty) in key.input_tys.iter().enumerate() {
            let m = slots(k);
            if m == 0 {
                continue;
            }
            let _ = writeln!(src, "fn ld{k}(g: u32) -> {} {{\n    let c = g >> {log}u; let o = g & {mask}u;", ty.wgsl());
            for s in 0..m - 1 {
                let _ = writeln!(src, "    if (c == p.c{k}_{s}) {{ return in{k}_{s}[o]; }}");
            }
            let _ = writeln!(src, "    return in{k}_{}[o];\n}}", m - 1);
        }
    }
    src.push_str(&generator_helpers(std::iter::once(key.instrs.as_slice()).chain(key.subs.iter().map(Vec::as_slice))));
    if key.tile.is_none() {
        src.push_str(&subs(key));
    }

    let (index, total) = match (segmented, based) {
        (true, _) => ("gi", "p.total"),
        (false, true) => ("gi", "luma.size.x"),
        (false, false) => ("i", "luma.size.x"),
    };
    let global = if segmented || based { "    let gi = p.base + i;\n" } else { "" };
    match key.tail {
        Tail::Store => {
            let stores: String = outputs.iter().enumerate().map(|(j, at)| format!("    out{j}[i] = t{at};\n")).collect();
            if let Some(tile) = &key.tile {
                src.push_str(&tiled_main(key, tile, based, &stores));
                return src;
            }
            let _ = write!(
                src,
                "
@compute @workgroup_size({wg})
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {{
    let i = gid.x + gid.y * nwg.x * {wg}u;
    if (i >= luma.size.x) {{ return; }}
{global}{}{stores}}}
",
                body(&key.instrs, &key.input_tys, key.in_place, index, total, segmented)
            );
        }
        Tail::Reduce(ops) => {
            let (ty, last) = (out_ty(outputs[0]), outputs[0]);
            let _ = write!(
                src,
                "
fn luma_eval(i: u32) -> {ty} {{
{global}{}    return t{last};
}}
{}",
                body(&key.instrs, &key.input_tys, None, index, total, segmented),
                reduce_main(ty, ops, false, "luma_eval", tuning)
            );
        }
    }
    src
}

/// A kernel running `p.steps` iterations of the store program `key` per element, with the
/// state in registers. `states[k]` is the state that input `k` reads (`None`: a loop-invariant
/// input). Bindings: one output and one initial value per state, the invariant inputs, `p`.
pub(super) fn loop_source(key: &FusedKey, states: &[Option<usize>], tuning: &Tuning) -> String {
    let wg = tuning.elementwise_workgroup;
    let count = key.outputs.len();
    let mut src = String::from("struct Params { steps: u32, ");
    for (k, ty) in key.scalar_tys.iter().enumerate() {
        let _ = write!(src, "s{k}: {}, ", ty.wgsl());
    }
    src.push_str("}\n");
    let mut binding = 0;
    for (role, access) in [("out", "read_write"), ("init", "read")] {
        for j in 0..count {
            let _ = writeln!(src, "@group(1) @binding({binding}) var<storage, {access}> {role}{j}: array<f32>;");
            binding += 1;
        }
    }
    for (k, ty) in key.input_tys.iter().enumerate() {
        if states.get(k).copied().flatten().is_none() {
            let _ = writeln!(src, "@group(1) @binding({binding}) var<storage, read> in{k}: array<{}>;", ty.wgsl());
            binding += 1;
        }
    }
    let _ = writeln!(src, "@group(1) @binding({binding}) var<uniform> p: Params;");
    src.push_str(&generator_helpers([key.instrs.as_slice()]));
    let _ = write!(
        src,
        "
@compute @workgroup_size({wg})
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {{
    let i = gid.x + gid.y * nwg.x * {wg}u;
    if (i >= luma.size.x) {{ return; }}
"
    );
    for j in 0..count {
        let _ = writeln!(src, "    var s{j}: f32 = init{j}[i];");
    }
    for (k, ty) in key.input_tys.iter().enumerate() {
        if states.get(k).copied().flatten().is_none() && direct(&key.instrs, k as u32) {
            let _ = writeln!(src, "    let a{k}: {} = in{k}[i];", ty.wgsl());
        }
    }
    src.push_str("    for (var step = 0u; step < p.steps; step = step + 1u) {\n");
    for (k, state) in states.iter().enumerate() {
        if let Some(j) = state {
            let _ = writeln!(src, "        let a{k}: f32 = s{j};");
        }
    }
    for (k, instr) in key.instrs.iter().enumerate() {
        let _ = writeln!(src, "        let t{k}: {} = {};", instr.op.out_ty().wgsl(), expression(instr, "i", "luma.size.x", false));
    }
    for (j, at) in key.outputs.iter().enumerate() {
        let _ = writeln!(src, "        s{j} = t{at};");
    }
    src.push_str("    }\n");
    for j in 0..count {
        let _ = writeln!(src, "    out{j}[i] = s{j};");
    }
    src.push_str("}\n");
    src
}

/// The loads of the inputs at element `i` and the instructions, as `let` bindings. The input
/// `in_place` (if any) is the output buffer `out0`, read before it is overwritten. `index`,
/// `total` and `segmented` are those of [`expression`].
fn body(
    instrs: &[Instr],
    input_tys: &[ElemTy],
    in_place: Option<u32>,
    index: &str,
    total: &str,
    segmented: bool,
) -> String {
    body_with(instrs, input_tys, in_place, index, total, segmented, &|_, _| None)
}

/// [`body`], where `staged(k, instr)` may replace the expression of instruction `k`.
fn body_with(
    instrs: &[Instr],
    input_tys: &[ElemTy],
    in_place: Option<u32>,
    index: &str,
    total: &str,
    segmented: bool,
    staged: &dyn Fn(usize, &Instr) -> Option<String>,
) -> String {
    let mut src = String::new();
    // Inputs read only through shifts are not loaded at `i`.
    for (k, ty) in input_tys.iter().enumerate() {
        if direct(instrs, k as u32) {
            let buffer = if in_place == Some(k as u32) { "out0".to_owned() } else { format!("in{k}") };
            let _ = writeln!(src, "    let a{k}: {} = {buffer}[i];", ty.wgsl());
        }
    }
    for (k, instr) in instrs.iter().enumerate() {
        let value = staged(k, instr).unwrap_or_else(|| expression(instr, index, total, segmented));
        let _ = writeln!(src, "    let t{k}: {} = {value};", instr.op.out_ty().wgsl());
    }
    src
}

/// The workgroup-memory staging of the sub-programs of a store kernel: each is computed once
/// per cell of its region (the tile plus a halo) and the shifts read it there. Part of the
/// [`FusedKey`] because the generated source bakes in the offsets, which for a
/// [`ShiftKind::Linear`] shift come from a scalar value.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub(crate) struct Tile {
    /// Width and height of the array (`h == 1` for 1D).
    pub(crate) dims: (u32, u32),
    /// Output cells per workgroup.
    pub(crate) tile: (u32, u32),
    /// Per sub-program, the halo (cells per side and axis) its region extends beyond the tile.
    pub(crate) need: Vec<(u32, u32)>,
    /// Per program (0 = main, `1 + k` = `subs[k]`) and instruction, the cell offset (x, y) a
    /// shift of a sub-program reads; `None` for every other instruction.
    pub(crate) offsets: Vec<Vec<Option<(i32, i32)>>>,
}

/// The staging of `key` on arrays of `len` elements, if its sub-programs are read through
/// small shifts only and their regions fit in workgroup memory. `None` keeps the sub-programs
/// as functions called per shift.
pub(super) fn plan_tile(key: &FusedKey, scalars: &[u32], len: usize, tuning: &Tuning) -> Option<Tile> {
    if key.tail != Tail::Store || key.subs.is_empty() || !key.spans.is_empty() {
        return None;
    }
    let programs: Vec<&[Instr]> = std::iter::once(key.instrs.as_slice()).chain(key.subs.iter().map(Vec::as_slice)).collect();
    let mut dims = None;
    for instr in programs.iter().copied().flatten() {
        if let OpKind::Shift(ShiftKind::Dims { dims: d, .. }) = instr.op.kind {
            if d[2] != 1 || d[3] != 1 || dims.is_some_and(|known| known != (d[0], d[1])) {
                return None;
            }
            dims = Some((d[0], d[1]));
        }
    }
    let (w, h) = dims.unwrap_or((u32::try_from(len).ok()?, 1));
    if u64::from(w) * u64::from(h) != len as u64 || w == 0 || h == 0 {
        return None;
    }

    let mut offsets = Vec::with_capacity(programs.len());
    for (p, instrs) in programs.iter().enumerate() {
        let mut row = Vec::with_capacity(instrs.len());
        for instr in *instrs {
            let skip = usize::from(matches!(instr.op.kind, OpKind::Shift(_)));
            if instr.args.iter().skip(skip).any(|arg| matches!(arg, Ref::Sub(_))) {
                return None;
            }
            let at = match (instr.op.kind, instr.args[0]) {
                // Sub-programs come before the programs reading them.
                (OpKind::Shift(_), Ref::Sub(j)) if p > 0 && j as usize + 1 >= p || j as usize >= key.subs.len() => return None,
                (OpKind::Shift(kind), Ref::Sub(_)) => {
                    let (dx, dy) = match kind {
                        ShiftKind::Dims { shifts, .. } => (signed(shifts[0], w), signed(shifts[1], h)),
                        ShiftKind::Linear => {
                            let Ref::Scalar(s) = instr.args[1] else { return None };
                            let amount = *scalars.get(s as usize)?;
                            if h == 1 {
                                (signed(amount, w), 0)
                            } else if amount % w == 0 {
                                (0, signed(amount / w, h))
                            } else {
                                return None;
                            }
                        }
                    };
                    // Element `c` of a shift is `u[c - shift]`.
                    let (ox, oy) = (-dx, -dy);
                    if ox.unsigned_abs() > STENCIL_RADIUS || oy.unsigned_abs() > STENCIL_RADIUS {
                        return None;
                    }
                    Some((ox, oy))
                }
                _ => None,
            };
            row.push(at);
        }
        offsets.push(row);
    }

    // Consumers first: the main program, then the sub-programs from the last (dependencies come first).
    let mut need = vec![(0u32, 0u32); key.subs.len()];
    for p in std::iter::once(0).chain((1..programs.len()).rev()) {
        let own = if p == 0 { (0, 0) } else { need[p - 1] };
        for (instr, at) in programs[p].iter().zip(&offsets[p]) {
            if let (Some((ox, oy)), Ref::Sub(j)) = (at, instr.args[0]) {
                let slot = &mut need[j as usize];
                *slot = (slot.0.max(own.0 + ox.unsigned_abs()), slot.1.max(own.1 + oy.unsigned_abs()));
            }
        }
    }

    let (mut tx, mut ty) = if h == 1 { (tuning.stencil_tile_1d, 1) } else { tuning.stencil_tile_2d };
    let bytes = |tx: u32, ty: u32| need.iter().map(|&(nx, ny)| (tx + 2 * nx) * (ty + 2 * ny) * ELEMENT_BYTES as u32).sum::<u32>();
    while bytes(tx, ty) > tuning.stencil_tile_bytes {
        if h == 1 || tx >= ty {
            tx /= 2;
        } else {
            ty /= 2;
        }
        if tx < 4 || (h > 1 && ty < 4) {
            return None;
        }
    }
    Some(Tile { dims: (w, h), tile: (tx, ty), need, offsets })
}

/// The kernel of `key` with its sub-programs staged as `tile` says, running one workgroup per
/// tile of the array. `based` and `stores` are those of the plain store kernel.
fn tiled_main(key: &FusedKey, tile: &Tile, based: bool, stores: &str) -> String {
    let ((w, h), (tx, ty)) = (tile.dims, tile.tile);
    let region = |j: usize| (tx + 2 * tile.need[j].0, ty + 2 * tile.need[j].1);
    let (halo_x, halo_y) = tile.need.iter().fold((0, 0), |(x, y), &(nx, ny)| (x.max(nx), y.max(ny)));
    let (index, global) = if based { ("gi", "        let gi = p.base + i;\n") } else { ("i", "") };
    // Cell (lx, ly) relative to the tile origin, as a global element.
    let element = if h == 1 {
        format!("wrap(i32(tile_x) + lx, {w}u)")
    } else {
        format!("wrap(i32(tile_x) + lx, {w}u) + wrap(i32(tile_y) + ly, {h}u) * {w}u")
    };
    let staged = |p: usize| {
        move |at: usize, instr: &Instr| -> Option<String> {
            let (ox, oy) = tile.offsets[p][at]?;
            let Ref::Sub(j) = instr.args[0] else { return None };
            let j = j as usize;
            let ((nx, ny), (lw, _)) = (tile.need[j], region(j));
            let row = if h == 1 { String::new() } else { format!(" + u32(ly + ({})) * {lw}u", oy + ny as i32) };
            Some(format!("w{j}[u32(lx + ({})){row}]", ox + nx as i32))
        }
    };

    let mut src = String::new();
    for (k, instrs) in key.subs.iter().enumerate() {
        let (lw, lh) = region(k);
        let ty = instrs.last().map_or(ElemTy::F32, |instr| instr.op.out_ty()).wgsl();
        let _ = writeln!(src, "var<workgroup> w{k}: array<{ty}, {}>;", lw * lh);
    }
    let _ = write!(
        src,
        "
{}
@compute @workgroup_size({WORKGROUP})
fn main(@builtin(local_invocation_index) lid: u32, @builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {{
    let g = wid.x + wid.y * nwg.x;
    let tile_x = (g % {tiles_x}u) * {tx}u;
    let tile_y = (g / {tiles_x}u) * {ty}u;
",
        wrap_source(w >= tx + 2 * halo_x && (h == 1 || h >= ty + 2 * halo_y)),
        tiles_x = w.div_ceil(tx),
    );
    for (k, instrs) in key.subs.iter().enumerate() {
        let ((nx, ny), (lw, lh)) = (tile.need[k], region(k));
        let last = instrs.len().saturating_sub(1);
        let _ = write!(
            src,
            "    for (var e = lid; e < {}u; e = e + {WORKGROUP}u) {{
        let lx = i32(e % {lw}u) - {nx};
        let ly = i32(e / {lw}u) - {ny};
        let i = {element};
{global}{}        w{k}[e] = t{last};
    }}
    workgroupBarrier();
",
            lw * lh,
            body_with(instrs, &key.input_tys, None, index, "luma.size.x", false, &staged(k + 1)),
        );
    }
    let _ = write!(
        src,
        "    for (var e = lid; e < {}u; e = e + {WORKGROUP}u) {{
        let lx = i32(e % {tx}u);
        let ly = i32(e / {tx}u);
        let x = tile_x + u32(lx);
        let y = tile_y + u32(ly);
        if (x < {w}u && y < {h}u) {{
            let i = x + y * {w}u;
{global}{}{stores}        }}
    }}
}}
",
        tx * ty,
        body_with(&key.instrs, &key.input_tys, key.in_place, index, "luma.size.x", false, &staged(0)),
    );
    src
}

/// One function `sub{k}(i)` per sub-program: the value of a shifted expression at element `i`.
/// Programs with sub-programs never run segmented.
fn subs(key: &FusedKey) -> String {
    let mut src = String::new();
    for (k, instrs) in key.subs.iter().enumerate() {
        let ty = instrs.last().map_or(ElemTy::F32, |instr| instr.op.out_ty()).wgsl();
        let last = instrs.len().saturating_sub(1);
        let _ = write!(
            src,
            "\nfn sub{k}(i: u32) -> {ty} {{\n{}    return t{last};\n}}\n",
            body(instrs, &key.input_tys, None, "i", "luma.size.x", false)
        );
    }
    src
}
