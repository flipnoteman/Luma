//! WGSL generation for fused programs.
//!
//! The source depends on the [`FusedKey`] only. Bindings, in order of the arguments: the outputs
//! (`out0`, `out1`, ...; a reduction has one, holding the partial results of every op), the
//! input buffers (`in0`, `in1`, ...), then, if the program has scalars (or is a reduction), one
//! uniform `p` holding them (`p.s0`, `p.s1`, ... after `p.n` for reductions).
//!
//! The program loads the inputs at element `i` and runs the instructions as `let` bindings. A
//! store kernel writes its output instructions; a reduction wraps the program in `luma_eval(i)`
//! and feeds the last value into the tree reductions of the first pass.

use std::fmt::Write;

use crate::builtin::{reduce_main, ELEMENTWISE_WORKGROUP};

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
    }
}

pub(super) fn expression(instr: &Instr) -> String {
    let [a, b, c] = instr.args.map(name);
    match instr.op.kind {
        OpKind::Unary(op) => substitute(op.expr(), &a, ""),
        OpKind::Binary(op) => substitute(op.expr(), &a, &b),
        OpKind::Compare(op) => substitute(op.expr(), &a, &b),
        OpKind::Powf => format!("pow({a}, {b})"),
        OpKind::Clamp => format!("clamp({a}, {b}, {c})"),
        OpKind::Select => format!("select({c}, {b}, {a} != 0u)"),
        OpKind::Shift(kind) => {
            let Ref::Input(k) = instr.args[0] else {
                unreachable!("the operand of a shift is always an input buffer")
            };
            format!("in{k}[{}]", shift_index(kind, &b))
        }
    }
}

/// The index element `i` of a shift reads; `s` names the scalar shift of [`ShiftKind::Linear`].
fn shift_index(kind: ShiftKind, s: &str) -> String {
    match kind {
        ShiftKind::Linear => format!("select(i - {s}, i + (luma.size.x - {s}), i < {s})"),
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
                    (1, _) => format!("(i % {dim}u)"),
                    (_, 3) => format!("(i / {stride}u)"),
                    _ => format!("((i / {stride}u) % {dim}u)"),
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

/// The WGSL source of the kernel for `key`.
pub(crate) fn source(key: &FusedKey) -> String {
    let reduce = matches!(key.tail, Tail::Reduce(_));
    let mut src = String::new();

    if reduce || !key.scalar_tys.is_empty() {
        src.push_str("struct Params { ");
        if reduce {
            src.push_str("n: u32, ");
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
    for (k, ty) in key.input_tys.iter().enumerate() {
        let _ = writeln!(
            src,
            "@group(1) @binding({}) var<storage, read> in{k}: array<{}>;",
            k + outputs.len(),
            ty.wgsl()
        );
    }
    if reduce || !key.scalar_tys.is_empty() {
        let _ = writeln!(
            src,
            "@group(1) @binding({}) var<uniform> p: Params;",
            key.input_tys.len() + outputs.len()
        );
    }

    match key.tail {
        Tail::Store => {
            let wg = ELEMENTWISE_WORKGROUP;
            let stores: String = outputs.iter().enumerate().map(|(j, at)| format!("    out{j}[i] = t{at};\n")).collect();
            let _ = write!(
                src,
                "
@compute @workgroup_size({wg})
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {{
    let i = gid.x + gid.y * nwg.x * {wg}u;
    if (i >= luma.size.x) {{ return; }}
{}{stores}}}
",
                body(key)
            );
        }
        Tail::Reduce(ops) => {
            let (ty, last) = (out_ty(outputs[0]), outputs[0]);
            let _ = write!(
                src,
                "
fn luma_eval(i: u32) -> {ty} {{
{}    return t{last};
}}
{}",
                body(key),
                reduce_main(ty, ops, false, "luma_eval")
            );
        }
    }
    src
}

/// The loads of the inputs at element `i` and the instructions, as `let` bindings.
fn body(key: &FusedKey) -> String {
    let mut src = String::new();
    // Inputs read only through shifts are not loaded at `i`.
    let direct = |k: u32| {
        key.instrs.iter().any(|instr| {
            let skip = usize::from(matches!(instr.op.kind, OpKind::Shift(_)));
            instr.args.iter().skip(skip).any(|arg| *arg == Ref::Input(k))
        })
    };
    for (k, ty) in key.input_tys.iter().enumerate() {
        if direct(k as u32) {
            let _ = writeln!(src, "    let a{k}: {} = in{k}[i];", ty.wgsl());
        }
    }
    for (k, instr) in key.instrs.iter().enumerate() {
        let _ = writeln!(src, "    let t{k}: {} = {};", instr.op.out_ty().wgsl(), expression(instr));
    }
    src
}
