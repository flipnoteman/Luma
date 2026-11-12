//! Elementwise expressions of two arguments as WGSL functions, the operator of a scan.

use std::fmt::Write;
use std::sync::Arc;

use crate::array::{Failure, Storage};
use crate::gpu::lock;

use super::codegen::{expression, NO_NAMES};
use super::program::{flatten, Tail};
use super::{ElemTy, Node, OpKind};

/// `fn op(x0: T, x1: T) -> T` and the scalars it reads.
pub(crate) struct Operator {
    /// The WGSL function `op`; its scalars are the uniform fields `p.s{k}` of `scalar_tys[k]`.
    pub source: String,
    pub scalar_tys: Vec<ElemTy>,
    /// The bit patterns of the scalars, in the order of `scalar_tys`.
    pub scalars: Vec<u32>,
}

/// `node = op(a, b)` as the WGSL function `fn op(x0: T, x1: T) -> T`, its scalars being the uniform
/// fields `p.s{k}`. `Ok(None)` if it is not an elementwise expression of `a` and `b` alone.
pub(crate) fn operator_source(
    node: &Arc<Node>,
    a: &Arc<Storage>,
    b: &Arc<Storage>,
) -> Result<Option<Operator>, Failure> {
    let Some(args) = lock(&node.args).take() else { return Ok(None) };
    let fused = flatten(&[(Arc::clone(node), args)], Tail::Store)?;
    let key = &fused.key;
    let Some(last) = key.instrs.len().checked_sub(1) else { return Ok(None) };
    let bad = !key.subs.is_empty()
        || key.outputs != [last as u32]
        || key.instrs.iter().any(|instr| matches!(instr.op.kind, OpKind::Shift(_) | OpKind::Generate(_)));
    if bad {
        return Ok(None);
    }
    // Which argument each input of the program is.
    let mut slots = Vec::with_capacity(fused.inputs.len());
    for input in &fused.inputs {
        if Arc::ptr_eq(input, a) {
            slots.push("x0");
        } else if Arc::ptr_eq(input, b) {
            slots.push("x1");
        } else {
            return Ok(None);
        }
    }
    let ty = key.instrs[last].op.out_ty().wgsl();
    let mut source = format!("fn op(x0: {ty}, x1: {ty}) -> {ty} {{\n");
    for (k, slot) in slots.iter().enumerate() {
        let _ = writeln!(source, "    let a{k}: {} = {slot};", key.input_tys[k].wgsl());
    }
    for (k, instr) in key.instrs.iter().enumerate() {
        let value = expression(instr, "0u", false, &NO_NAMES);
        let _ = writeln!(source, "    let t{k}: {} = {value};", instr.op.out_ty().wgsl());
    }
    let _ = writeln!(source, "    return t{last};\n}}");
    Ok(Some(Operator { source, scalar_tys: key.scalar_tys.clone(), scalars: fused.scalars.clone() }))
}
