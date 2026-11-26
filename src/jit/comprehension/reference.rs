//! A host interpreter of a form in `f64`: slow and obviously correct, the oracle of the tests of the recogniser, the look-through
//! and the generic lowering (07 §4.2).

use super::{CombineOp, CombinePart, Comprehension, DimKind};
use crate::builtin::{BinOp, CmpOp, UnOp};
use crate::jit::program::{Instr, Ref};
use crate::jit::{Edge, OpKind};

impl Comprehension {
    /// The result of the form over `inputs` (the flat source of access `j` in `inputs[j]`): one value per point of the parallel
    /// dims, output axis 0 fastest. Single-part combiners only; the elementwise instructions the tests use.
    pub(crate) fn eval_reference(&self, inputs: &[&[f64]]) -> Vec<f64> {
        let extents: Vec<u64> = self.space.iter().map(|dim| dim.extent).collect();
        assert!(self.space.iter().all(|dim| matches!(dim.kind, DimKind::Parallel | DimKind::Reduce)), "no segmented dims in the reference");
        let part = match self.combiner.as_ref().map(|combiner| &combiner.op) {
            Some(CombineOp::Single(part)) => Some(*part),
            None => None,
            Some(CombineOp::Tuple(_)) => panic!("no tuple combiners in the reference"),
        };
        let out_axes: Vec<usize> = self.output.dim_of_axis.iter().flatten().map(|&d| usize::from(d)).collect();
        let out_len: u64 = out_axes.iter().map(|&d| extents[d]).product();
        let reduce: Vec<usize> = (0..extents.len()).filter(|&d| self.space[d].is_reduce()).collect();
        let reduce_len: u64 = reduce.iter().map(|&d| extents[d]).product();
        let mut out = Vec::with_capacity(out_len as usize);
        let mut point = vec![0i64; extents.len()];
        for o in 0..out_len {
            let mut rest = o;
            for &d in &out_axes {
                point[d] = (rest % extents[d]) as i64;
                rest /= extents[d];
            }
            let mut acc: Option<f64> = None;
            for r in 0..reduce_len.max(1) {
                let mut rest = r;
                for &d in &reduce {
                    point[d] = (rest % extents[d]) as i64;
                    rest /= extents[d];
                }
                let value = self.body_at(&point, inputs);
                acc = Some(match (acc, part) {
                    (None, _) => value,
                    (Some(a), Some(CombinePart::Add)) => a + value,
                    (Some(a), Some(CombinePart::Mul)) => a * value,
                    (Some(a), Some(CombinePart::Min)) => a.min(value),
                    (Some(a), Some(CombinePart::Max)) => a.max(value),
                    (Some(a), other) => panic!("no {other:?} in the reference ({a})"),
                });
            }
            out.push(acc.unwrap());
        }
        out
    }

    /// The first out of the body at `point`.
    fn body_at(&self, point: &[i64], inputs: &[&[f64]]) -> f64 {
        let loads: Vec<f64> = self.inputs.iter().enumerate().map(|(j, access)| load(access, point, inputs[j])).collect();
        let mut temps: Vec<f64> = Vec::with_capacity(self.body.instrs.len());
        for instr in self.body.instrs.iter() {
            let arg = |r: Ref| match r {
                Ref::Input(j) => loads[j as usize],
                Ref::Scalar(k) => f64::from(f32::from_bits(self.body.scalars[k as usize].bits as u32)),
                Ref::Tmp(t) => temps[t as usize],
                Ref::Unused => 0.0,
                other => panic!("no {other:?} in the reference"),
            };
            let value = apply(instr, [arg(instr.args[0]), arg(instr.args[1]), arg(instr.args[2])]);
            temps.push(value);
        }
        temps[self.body.outs[0] as usize]
    }
}

fn load(access: &super::Access, point: &[i64], data: &[f64]) -> f64 {
    let mut flat = 0i64;
    let mut stride = 1i64;
    for (a, row) in access.map.rows.iter().enumerate() {
        let mut coordinate = row.offset;
        for (&c, &p) in row.coeff.iter().zip(point) {
            coordinate += i64::from(c) * p;
        }
        let extent = i64::from(access.shape[a]);
        coordinate = match access.edge {
            Edge::None => coordinate,
            Edge::Fill if !(0..extent).contains(&coordinate) => return f64::from(f32::from_bits(access.fill.unwrap().bits as u32)),
            Edge::Fill => coordinate,
            Edge::Clamp => coordinate.clamp(0, extent - 1),
            Edge::Mirror => {
                let period = 2 * extent;
                let c = coordinate.rem_euclid(period);
                if c < extent { c } else { period - 1 - c }
            }
            Edge::Wrap => coordinate.rem_euclid(extent),
        };
        flat += coordinate * stride;
        stride *= extent;
    }
    data[flat as usize]
}

fn apply(instr: &Instr, [x, y, z]: [f64; 3]) -> f64 {
    let flag = |b: bool| if b { 1.0 } else { 0.0 };
    match instr.op.kind {
        OpKind::Unary(UnOp::Id) | OpKind::Cast(_) => x,
        OpKind::Unary(UnOp::Neg) => -x,
        OpKind::Unary(UnOp::Abs) => x.abs(),
        OpKind::Unary(UnOp::Sqrt) => x.sqrt(),
        OpKind::Binary(BinOp::Add) => x + y,
        OpKind::Binary(BinOp::Sub) => x - y,
        OpKind::Binary(BinOp::Mul) => x * y,
        OpKind::Binary(BinOp::Div) => x / y,
        OpKind::Binary(BinOp::Min) => x.min(y),
        OpKind::Binary(BinOp::Max) => x.max(y),
        OpKind::Compare(CmpOp::Lt) => flag(x < y),
        OpKind::Compare(CmpOp::Le) => flag(x <= y),
        OpKind::Compare(CmpOp::Gt) => flag(x > y),
        OpKind::Compare(CmpOp::Ge) => flag(x >= y),
        OpKind::Compare(CmpOp::Eq) => flag(x == y),
        OpKind::Compare(CmpOp::Ne) => flag(x != y),
        OpKind::Select => if x != 0.0 { y } else { z },
        other => panic!("the reference does not interpret {other:?}"),
    }
}
