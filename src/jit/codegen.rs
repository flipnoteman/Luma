//! WGSL generation for fused programs.
//!
//! The source depends on the [`FusedKey`] only. Bindings, in order of the arguments: the output
//! (`out`), the input buffers (`in0`, `in1`, ...), then, if the program has scalars (or is a
//! reduction), one uniform `p` holding them (`p.s0`, `p.s1`, ... after `p.n` for reductions).
//!
//! The program becomes a function `luma_eval(i)` that loads the inputs at element `i`, runs the
//! instructions as `let` bindings and returns the last value. The kernel either stores it or
//! feeds it into the tree reduction of the first pass.

use std::fmt::Write;

use crate::builtin::{ReduceOp, ELEMENTWISE_WORKGROUP, REDUCE_BLOCK, REDUCE_WORKGROUP};

use super::program::{FusedKey, Instr, Ref, Tail};
use super::{ElemTy, OpKind};

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

fn expression(instr: &Instr) -> String {
    let [a, b, c] = instr.args.map(name);
    match instr.op.kind {
        OpKind::Unary(op) => substitute(op.expr(), &a, ""),
        OpKind::Binary(op) => substitute(op.expr(), &a, &b),
        OpKind::Compare(op) => substitute(op.expr(), &a, &b),
        OpKind::Powf => format!("pow({a}, {b})"),
        OpKind::Clamp => format!("clamp({a}, {b}, {c})"),
        OpKind::Select => format!("select({c}, {b}, {a} != 0u)"),
    }
}

/// The WGSL source of the kernel for `key`.
pub(crate) fn source(key: &FusedKey) -> String {
    let result_ty = key.instrs.last().map_or(ElemTy::F32, |instr| instr.op.out_ty()).wgsl();
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
    let _ = writeln!(src, "@group(1) @binding(0) var<storage, read_write> out: array<{result_ty}>;");
    for (k, ty) in key.input_tys.iter().enumerate() {
        let _ = writeln!(
            src,
            "@group(1) @binding({}) var<storage, read> in{k}: array<{}>;",
            k + 1,
            ty.wgsl()
        );
    }
    if reduce || !key.scalar_tys.is_empty() {
        let _ = writeln!(src, "@group(1) @binding({}) var<uniform> p: Params;", key.input_tys.len() + 1);
    }

    let _ = writeln!(src, "\nfn luma_eval(i: u32) -> {result_ty} {{");
    for (k, ty) in key.input_tys.iter().enumerate() {
        let _ = writeln!(src, "    let a{k}: {} = in{k}[i];", ty.wgsl());
    }
    for (k, instr) in key.instrs.iter().enumerate() {
        let _ = writeln!(src, "    let t{k}: {} = {};", instr.op.out_ty().wgsl(), expression(instr));
    }
    let last = key.instrs.len().saturating_sub(1);
    let _ = writeln!(src, "    return t{last};\n}}\n");

    match key.tail {
        Tail::Store => {
            let wg = ELEMENTWISE_WORKGROUP;
            let _ = write!(
                src,
                "@compute @workgroup_size({wg})
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {{
    let i = gid.x + gid.y * nwg.x * {wg}u;
    if (i >= luma.size.x) {{ return; }}
    out[i] = luma_eval(i);
}}
"
            );
        }
        Tail::Reduce(op) => reduce_main(&mut src, result_ty, op),
    }
    src
}

/// The first pass of a reduction, like `builtin::reduce` but loading through `luma_eval`.
fn reduce_main(src: &mut String, ty: &str, op: ReduceOp) {
    // Out-of-range loads: sums read zero, min/max re-read the last element (harmless for them).
    let (load, combine) = match op {
        ReduceOp::Sum => ("if (idx < p.n) { v = luma_eval(idx); }", "u + v"),
        ReduceOp::SumSquares => ("if (idx < p.n) { let e = luma_eval(idx); v = e * e; }", "u + v"),
        ReduceOp::Min => ("v = luma_eval(min(idx, p.n - 1u));", "min(u, v)"),
        ReduceOp::Max => ("v = luma_eval(min(idx, p.n - 1u));", "max(u, v)"),
    };
    let wg = REDUCE_WORKGROUP;
    let block = REDUCE_BLOCK;
    let half = wg / 2;
    let _ = write!(
        src,
        "var<workgroup> sh: array<{ty}, {wg}>;

fn combine(u: {ty}, v: {ty}) -> {ty} {{ return {combine}; }}

@compute @workgroup_size({wg})
fn main(
    @builtin(local_invocation_index) lid: u32,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {{
    let group = wid.x + wid.y * nwg.x;
    let base = group * {block}u + lid;
    var v0 = {ty}(0); var v1 = {ty}(0); var v2 = {ty}(0); var v3 = {ty}(0);
    {{ let idx = base; var v = {ty}(0); {load} v0 = v; }}
    {{ let idx = base + {wg}u; var v = {ty}(0); {load} v1 = v; }}
    {{ let idx = base + {wg}u * 2u; var v = {ty}(0); {load} v2 = v; }}
    {{ let idx = base + {wg}u * 3u; var v = {ty}(0); {load} v3 = v; }}
    sh[lid] = combine(combine(v0, v1), combine(v2, v3));
    workgroupBarrier();
    for (var s = {half}u; s > 0u; s = s >> 1u) {{
        if (lid < s) {{ sh[lid] = combine(sh[lid], sh[lid + s]); }}
        workgroupBarrier();
    }}
    if (lid == 0u && group * {block}u < p.n) {{ out[group] = sh[0]; }}
}}
"
    );
}
