//! The measure a rewrite of the normaliser must strictly decrease: the estimated nanoseconds of the scope, then its node count.
//!
//! `est` of one comprehension is a compulsory-traffic roofline: the flops of its body, combiner and the prologues of its
//! accesses over the points of its space, against the bytes each access reads once (the elements some row of its map reaches)
//! and the output writes, scaled by the memory model's factors. It does not know the strategy that will run: for rules whose
//! benefit is algorithmic (n³ → n²) that is enough, for constant-factor ones the rule carries `dominates` or its own estimate.

use super::scope::Scope;
use crate::context::Context;
use crate::jit::comprehension::{Access, CombinePart, Comprehension, Dim, Epilogue};
use crate::jit::cost::{CostModel, MemoryModel, kernel_ns_on, op_weight};
use crate::jit::Src;

/// What a scope costs.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct Measure {
    /// Estimated nanoseconds × 1024, rounded.
    pub(crate) ns: u64,
    pub(crate) nodes: u32,
}

impl Measure {
    /// Whether a rewrite from `before` to `self` is worth taking: at least 1/64 faster, or no slower with fewer nodes.
    pub(crate) fn improves(self, before: Measure) -> bool {
        self.ns + (before.ns >> 6) <= before.ns || (self.ns <= before.ns && self.nodes < before.nodes)
    }

    /// The estimated nanoseconds.
    pub(crate) fn nanoseconds(self) -> f64 {
        self.ns as f64 / 1024.0
    }
}

/// The scale of [`Measure::ns`].
const FIXED: f64 = 1024.0;

/// The cost and memory models an estimate reads.
pub(crate) struct Models {
    pub(crate) cost: CostModel,
    pub(crate) memory: Option<MemoryModel>,
}

impl Models {
    pub(crate) fn of(ctx: &Context) -> Models {
        Models { cost: ctx.cost_model(), memory: ctx.memory_model() }
    }
}

/// The measure of `scope` in `ctx`.
pub(crate) fn measure(scope: &Scope, ctx: &Context) -> Measure {
    measure_with(scope, &Models::of(ctx))
}

/// The measure of `scope` under `models`.
pub(crate) fn measure_with(scope: &Scope, models: &Models) -> Measure {
    let ns: f64 = scope.comps.iter().map(|c| estimate(&c.form, models)).sum();
    Measure { ns: (ns * FIXED).round() as u64, nodes: scope.nodes() }
}

fn points(space: &[Dim]) -> f64 {
    space.iter().map(|dim| dim.extent as f64).product()
}

/// Per element of the combined values, in f32-operation units.
fn combiner_weight(form: &Comprehension) -> f64 {
    form.combiner.as_ref().map_or(0.0, |combiner| {
        combiner
            .op
            .parts()
            .iter()
            .map(|part| match part {
                CombinePart::Add | CombinePart::Mul | CombinePart::Min | CombinePart::Max | CombinePart::Or | CombinePart::And => 1.0,
                CombinePart::ArgMax | CombinePart::ArgMin => 2.0,
            })
            .sum()
    })
}

/// The work of the elementwise expression below an access (its prologue), per read.
fn prologue_weight(access: &Access) -> f64 {
    match &access.src {
        Src::Node(node) if node.is_pending() && node.deferred_value().is_none() => node.weight(),
        _ => 0.0,
    }
}

/// How many elements of the source the access reads: the product of the extents of the dims some row of its map uses, at most
/// the source's length (a window row reads fewer elements than its dims' product, which the length clamps).
fn unique(form: &Comprehension, access: &Access) -> f64 {
    let mut used = [false; crate::jit::comprehension::MAX_DIMS];
    for row in &access.map.rows {
        for (d, &c) in row.coeff.iter().enumerate() {
            if c != 0 {
                used[d] = true;
            }
        }
    }
    let reached: f64 = form.space.iter().enumerate().filter(|(d, _)| used[*d]).map(|(_, dim)| dim.extent as f64).product();
    reached.min(access.shape.iter().map(|&d| f64::from(d)).product())
}

/// Time per byte of reading `access`, relative to coalesced access: the memory model's factor for its pattern.
fn factor(form: &Comprehension, access: &Access, memory: Option<&MemoryModel>) -> f64 {
    let Some(memory) = memory else { return 1.0 };
    if access.indirect.is_some() {
        return memory.gather_factor;
    }
    // The fastest dim of the space (listed first) should be on the fastest axis (row 0) of the source.
    let on = |row: usize| access.map.rows[row].coeff[0] != 0;
    if form.space.is_empty() || on(0) || !(1..4).any(on) {
        return 1.0;
    }
    let axis = (1..4).find(|&row| on(row)).unwrap_or(1);
    // A full swap: the second dim of the space is on the fastest axis.
    if axis == 1 && form.space.len() > 1 && access.map.rows[0].coeff[1] != 0 {
        return memory.transpose_factor;
    }
    let stride: u64 = access.shape[..axis].iter().map(|&d| u64::from(d)).product();
    let bucket = if stride <= 2 { 0 } else if stride <= 8 { 1 } else { 2 };
    memory.strided_factor[bucket]
}

/// The estimated nanoseconds of one comprehension.
pub(crate) fn estimate(form: &Comprehension, models: &Models) -> f64 {
    let cost = &models.cost;
    let space = points(&form.space);
    let body: f64 = form.body.instrs.iter().map(|instr| op_weight(&instr.op, cost)).sum();
    let prologues: f64 = form.inputs.iter().map(prologue_weight).sum();
    let out: f64 = form.output.dim_of_axis.iter().flatten().map(|&d| form.space[usize::from(d)].extent as f64).product();
    let epilogue = match &form.epilogue {
        Epilogue::None => 0.0,
        Epilogue::Steps(finish) => out * finish.epilogue.steps().count() as f64,
        Epilogue::Program(program) => out * program.instrs.iter().map(|instr| op_weight(&instr.op, cost)).sum::<f64>(),
    };
    let flops = space * (body + combiner_weight(form) + prologues) + epilogue;
    // Bytes at the bandwidth of DRAM: a read that stays in L2 is charged the bytes it would cost there.
    let mut bytes = out * form.ty.layout().bytes(1) as f64;
    for access in &form.inputs {
        let elem = access.ty.layout().bytes(1) as f64;
        let read = unique(form, access) * elem;
        let bandwidth = match &models.memory {
            Some(memory) if memory.l2_bytes.is_some_and(|l2| read <= l2 as f64 / 2.0) && memory.l2_gbs > 0.0 => memory.l2_gbs,
            _ => cost.bandwidth_gbs,
        };
        bytes += read * factor(form, access, models.memory.as_ref()) * cost.bandwidth_gbs / bandwidth;
    }
    let invocations = form.space.iter().filter(|dim| !dim.is_reduce()).map(|dim| dim.extent as f64).product::<f64>();
    kernel_ns_on(cost, bytes, flops, invocations)
}
