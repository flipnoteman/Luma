//! Look-through: re-express an access whose source is a pending elementwise or view node as body instructions and composed maps
//! (W2-R1 §2). Works on a copy; the original form and the nodes it reads are untouched, and the arithmetic per point is the same
//! (class `Exact`).

use std::sync::Arc;

use super::map::{AffineMap, MapRef};
use super::{Access, Comprehension, MAX_INPUTS};
use crate::jit::program::{Instr, Ref};
use crate::jit::{Edge, Node, OpKind, Scalar, ShiftKind, Src};

/// What look-through may spend: nodes taken apart, and the limit on the inputs of the result.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Budget {
    pub(crate) nodes: usize,
    pub(crate) inputs: usize,
}

impl Default for Budget {
    /// Nodes 16, inputs [`MAX_INPUTS`].
    fn default() -> Budget {
        Budget { nodes: 16, inputs: MAX_INPUTS }
    }
}

/// A pending node that look-through may take apart, with its operands.
fn pending(src: &Src) -> Option<(Arc<Node>, [Src; 3])> {
    let Src::Node(node) = src else { return None };
    if !node.is_pending() || node.is_looped() || node.deferred_value().is_some() {
        return None;
    }
    Some((Arc::clone(node), node.operands()?))
}

fn same_scalar(a: &Scalar, b: &Scalar) -> bool {
    a.ty == b.ty && a.bits == b.bits
}

fn same_src(a: &Src, b: &Src) -> bool {
    match (a, b) {
        (Src::Node(a), Src::Node(b)) => Arc::ptr_eq(a, b),
        (Src::Buffer(a), Src::Buffer(b)) => Arc::ptr_eq(a, b),
        (Src::Scalar(a), Src::Scalar(b)) => same_scalar(a, b),
        _ => false,
    }
}

/// Whether two accesses read the same elements at every point.
fn same_access(a: &Access, b: &Access) -> bool {
    same_src(&a.src, &b.src)
        && a.shape == b.shape
        && a.ty == b.ty
        && *a.map == *b.map
        && a.edge == b.edge
        && match (&a.fill, &b.fill) {
            (None, None) => true,
            (Some(x), Some(y)) => same_scalar(x, y),
            _ => false,
        }
        && a.indirect.is_none()
        && b.indirect.is_none()
}

/// Maps the inputs of every instruction of `instrs` through `to`.
fn renumber(instrs: &mut [Instr], to: impl Fn(u32) -> Ref) {
    for instr in instrs {
        for arg in &mut instr.args {
            if let Ref::Input(j) = *arg {
                *arg = to(j);
            }
        }
    }
}

/// Takes access `j` apart if its source is a node the pass understands. `Some(true)` if the form changed, `Some(false)` if this
/// access stays as it is, `None` if a limit was hit.
fn step(form: &mut Comprehension, j: usize, budget: &mut Budget) -> Option<bool> {
    let Some((node, operands)) = pending(&form.inputs[j].src) else { return Some(false) };
    let access = &form.inputs[j];
    if access.indirect.is_some() {
        return Some(false);
    }
    match node.op.kind {
        OpKind::Shift(ShiftKind::View { map: view, edge }) => {
            // An outer edge cannot be moved through an inner one; the access reads the node as the node's own shape.
            if access.edge != Edge::None || access.shape != view.out {
                return Some(false);
            }
            let fill = match (edge, &operands[1]) {
                (Edge::Fill, Src::Scalar(value)) => Some(*value),
                (Edge::Fill, _) => return Some(false),
                _ => None,
            };
            if !matches!(operands[0], Src::Node(_) | Src::Buffer(_)) {
                return Some(false);
            }
            let Some(outer) = AffineMap::from_view(&view, [0, 1, 2, 3]).then(&access.map) else { return Some(false) };
            if budget.nodes == 0 {
                return None;
            }
            budget.nodes -= 1;
            let access = &mut form.inputs[j];
            access.src = operands[0].clone();
            access.shape = view.src;
            access.map = MapRef::owned(outer);
            access.edge = edge;
            access.fill = fill;
            Some(true)
        }
        kind @ (OpKind::Unary(_) | OpKind::Binary(_) | OpKind::Compare(_) | OpKind::Select | OpKind::Cast(_) | OpKind::Classify(_) | OpKind::Part(_) | OpKind::Make | OpKind::Clamp | OpKind::Powf) => {
            // A fill value is the value of the node, not an operand of it.
            if access.edge == Edge::Fill {
                return Some(false);
            }
            let arity = kind.arity();
            let operand_len = |src: &Src| match src {
                Src::Node(node) => node.len(),
                Src::Buffer(storage) => storage.len(),
                Src::Scalar(_) => 1,
            };
            if operands[..arity].iter().any(|src| !matches!(src, Src::Scalar(_)) && operand_len(src) != node.len()) {
                return Some(false);
            }
            let arrays = operands[..arity].iter().filter(|src| !matches!(src, Src::Scalar(_))).count();
            if form.inputs.len() - 1 + arrays > budget.inputs {
                return None;
            }
            if budget.nodes == 0 {
                return None;
            }
            budget.nodes -= 1;
            let template = form.inputs.remove(j);
            // The accesses that stay keep their order; the node's array operands follow them.
            let mut args = [Ref::Unused; 3];
            let mut scalars = form.body.scalars.clone();
            for (i, src) in operands[..arity].iter().enumerate() {
                args[i] = match src {
                    Src::Scalar(value) => {
                        scalars.push(*value);
                        Ref::Scalar(scalars.len() as u32 - 1)
                    }
                    other => {
                        form.inputs.push(Access { src: other.clone(), shape: template.shape, ty: node.op.operand_ty(i), map: template.map.clone(), edge: template.edge, fill: None, indirect: None, props: super::Props::NONE });
                        Ref::Input(form.inputs.len() as u32 - 1)
                    }
                };
            }
            // The value the access was is instruction 0; the others move down one.
            let mut instrs: Vec<Instr> = Vec::with_capacity(form.body.instrs.len() + 1);
            instrs.push(Instr { op: node.op, args });
            instrs.extend(form.body.instrs.iter().cloned().map(|mut instr| {
                for arg in &mut instr.args {
                    if let Ref::Tmp(t) = *arg {
                        *arg = Ref::Tmp(t + 1);
                    }
                }
                instr
            }));
            // Old input `i` is now `i - 1` past the removed one; the node's own operands were appended after the remaining ones, so
            // they already have their final numbers (their instruction sits first and is not renumbered).
            renumber(&mut instrs[1..], |i| match i.cmp(&(j as u32)) {
                std::cmp::Ordering::Less => Ref::Input(i),
                std::cmp::Ordering::Equal => Ref::Tmp(0),
                std::cmp::Ordering::Greater => Ref::Input(i - 1),
            });
            form.body.instrs = Arc::new(instrs);
            form.body.scalars = scalars;
            for out in &mut form.body.outs {
                *out += 1;
            }
            Some(true)
        }
        _ => Some(false),
    }
}

/// Merges accesses that read the same elements (CSE of loads); `true` if some were merged.
fn merge(form: &mut Comprehension) -> bool {
    let mut merged = false;
    let mut j = 0;
    while j < form.inputs.len() {
        match (0..j).find(|&i| same_access(&form.inputs[i], &form.inputs[j])) {
            Some(i) => {
                form.inputs.remove(j);
                let mut instrs: Vec<Instr> = form.body.instrs.to_vec();
                renumber(&mut instrs, |k| match k.cmp(&(j as u32)) {
                    std::cmp::Ordering::Less => Ref::Input(k),
                    std::cmp::Ordering::Equal => Ref::Input(i as u32),
                    std::cmp::Ordering::Greater => Ref::Input(k - 1),
                });
                form.body.instrs = Arc::new(instrs);
                merged = true;
            }
            None => j += 1,
        }
    }
    merged
}

/// A copy of `c` in which every access whose source is a pending view or elementwise node reads that node's sources instead (the
/// node's operation moved into the body, its view composed into the map). Nodes that are looped, deferred or not understood
/// (generators, circular shifts, takes) stay as opaque inputs. Returns `None` if the budget of nodes or inputs runs out; a form
/// nothing could be done to comes back unchanged.
pub(crate) fn inline_sources(c: &Comprehension, budget: &mut Budget) -> Option<Comprehension> {
    let mut form = c.clone();
    loop {
        let mut changed = false;
        let mut j = 0;
        while j < form.inputs.len() {
            if step(&mut form, j, budget)? {
                changed = true;
                // The access at `j` is now another one (or the instruction took its place): look at the same index again.
                continue;
            }
            j += 1;
        }
        changed |= merge(&mut form);
        if !changed {
            break;
        }
    }
    debug_assert!(form.validate().is_ok(), "look-through keeps the form valid");
    Some(form)
}
