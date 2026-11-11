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
use crate::tuning::Tuning;

use super::program::{FusedKey, Instr, Ref, Tail};
use super::{ElemTy, OpKind, ShiftKind};

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
    let uniform = reduce || !key.scalar_tys.is_empty() || segmented;
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
    src.push_str(&subs(key));

    let (index, total) = if segmented { ("gi", "p.total") } else { ("i", "luma.size.x") };
    let global = if segmented { "    let gi = p.base + i;\n" } else { "" };
    match key.tail {
        Tail::Store => {
            let stores: String = outputs.iter().enumerate().map(|(j, at)| format!("    out{j}[i] = t{at};\n")).collect();
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
    let mut src = String::new();
    // Inputs read only through shifts are not loaded at `i`.
    for (k, ty) in input_tys.iter().enumerate() {
        if direct(instrs, k as u32) {
            let buffer = if in_place == Some(k as u32) { "out0".to_owned() } else { format!("in{k}") };
            let _ = writeln!(src, "    let a{k}: {} = {buffer}[i];", ty.wgsl());
        }
    }
    for (k, instr) in instrs.iter().enumerate() {
        let _ = writeln!(src, "    let t{k}: {} = {};", instr.op.out_ty().wgsl(), expression(instr, index, total, segmented));
    }
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
