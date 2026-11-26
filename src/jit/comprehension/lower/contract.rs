//! G-A: the fused matmul tail for any separable body (W2-R0 §5.3).

use std::panic::Location;
use std::sync::Arc;

use crate::array::Failure;
use crate::jit::comprehension::Comprehension;
use crate::jit::structured::emit::Emit;
use crate::jit::MatmulSide;

use std::collections::HashMap;

use smallvec::SmallVec;

use super::{combine_of, coalesce, view_node, Origin6};
use crate::array::Storage;
use crate::builtin::{BinOp, UnOp};
use crate::context::RawArg;
use crate::jit::comprehension::roles::{infer_contraction_roles, separate, Roles};
use crate::jit::comprehension::{CombineOp, CombinePart, Compensation, DimKind, Epilogue, MAX_DIMS};
use crate::jit::program::{Instr, Ref};
use crate::jit::{self, ElemTy, Node, Op, OpKind, Src};
use crate::matmul::MatmulTile;
use crate::context::Context;
use crate::error::LumaError;

/// The message of the failure `lower` turns into a declined strategy: the operands could not be fused into the kernel.
pub(crate) const NOT_FUSED: &str = "a generic contraction needs its operands fused into the kernel";

/// A form that G-A runs: the roles, the operand programs and what the kernel computes.
pub(crate) struct Prepared {
    form: Comprehension,
    /// Per side (lhs, rhs): the factors whose product is the operand's value (one input for a form of two inputs).
    factors: [SmallVec<[Ref; 2]>; 2],
    /// The dims of the axes of the operand of each side: lhs `[k.., m.., b..]`, rhs `[n.., k.., b..]` (fastest first).
    axes: [SmallVec<[usize; 4]>; 2],
    pub(crate) m: u64,
    pub(crate) n: u64,
    pub(crate) k: u64,
    batch: u64,
    mac: Mac,
    /// The instructions the operand expressions are made of (the form's body).
    instrs: std::sync::Arc<Vec<Instr>>,
}

/// Whether `instr` computes elementwise from its operands (what a multiply-add may be made of).
fn elementwise(instr: &Instr) -> bool {
    matches!(instr.op.kind, OpKind::Unary(_) | OpKind::Binary(_) | OpKind::Compare(_) | OpKind::Select | OpKind::Cast(_) | OpKind::Clamp | OpKind::Classify(_) | OpKind::Powf | OpKind::Part(_) | OpKind::Make)
}

/// The instruction `r` names through `Unary(Id)`, as a `Ref`.
fn skip_id(instrs: &[Instr], mut r: Ref) -> Ref {
    while let Ref::Tmp(t) = r {
        match instrs.get(t as usize) {
            Some(Instr { op, args }) if matches!(op.kind, OpKind::Unary(UnOp::Id)) => r = args[0],
            _ => break,
        }
    }
    r
}

/// Plans the contraction tail for `form`, or `None` if it has none: two sides that read disjoint inputs, the roles of a contraction
/// (`m`, `n`, `k` and a batch), a body that is a function of one load of each side, `f32`, and an output in the order `n`, `m`, batch.
pub(crate) fn prepare(form: &Comprehension) -> Option<Prepared> {
    if form.ty != ElemTy::F32 || !matches!(form.epilogue, Epilogue::None) || form.output.order != crate::jit::comprehension::OutMap::IDENTITY || form.validate().is_err() {
        return None;
    }
    if form.space.iter().any(|dim| matches!(dim.kind, DimKind::Segmented { .. })) || form.inputs.iter().any(|a| a.indirect.is_some() || matches!(a.src, Src::Scalar(_))) || form.inputs.len() < 2 {
        return None;
    }
    let combiner = form.combiner.as_ref()?;
    let CombineOp::Single(part) = combiner.op else { return None };
    combine_of(part)?;
    if combiner.compensation != Compensation::None || form.body.outs.len() != 1 || !form.body.scalars.is_empty() || form.body.instrs.iter().any(|instr| !elementwise(instr)) {
        return None;
    }
    let form = coalesce(form);
    let roles: Roles = infer_contraction_roles(&form)?;
    if !roles.free.is_empty() {
        return None;
    }
    let separation = (form.inputs.len() > 2).then(|| separate(&form)).flatten();
    if form.inputs.len() > 2 && separation.is_none() {
        return None;
    }
    // The sides: `roles.lhs` owns `m`. The output is `[n.., m.., batch..]` (fastest first), or `[m.., n.., batch..]` with the sides exchanged.
    let sides: [SmallVec<[u8; 4]>; 2] = match &separation {
        Some(separation) => separation.sides.clone(),
        None => [SmallVec::from_slice(&[0]), SmallVec::from_slice(&[1])],
    };
    let factors: [SmallVec<[Ref; 2]>; 2] = match &separation {
        Some(separation) => separation.factors.clone(),
        None => [SmallVec::from_slice(&[Ref::Input(0)]), SmallVec::from_slice(&[Ref::Input(1)])],
    };
    let order: SmallVec<[usize; 4]> = form.output.dim_of_axis.iter().flatten().map(|&d| usize::from(d)).filter(|&d| form.space[d].extent > 1).collect();
    let rank = |d: usize| -> u8 {
        if roles.n.contains(d) { 0 } else if roles.m.contains(d) { 1 } else { 2 }
    };
    let ranks: SmallVec<[u8; 4]> = order.iter().map(|&d| rank(d)).collect();
    let standard = ranks.windows(2).all(|w| w[0] <= w[1]);
    // `m` before `n` before the batch: the ranks of `n` and `m` exchanged are in order.
    let exchanged = {
        let swapped: SmallVec<[u8; 4]> = ranks.iter().map(|&r| match r { 0 => 1, 1 => 0, other => other }).collect();
        swapped.windows(2).all(|w| w[0] <= w[1])
    };
    let swap = if standard { false } else if exchanged { true } else { return None };
    let (lhs_side, rhs_side) = if swap { (usize::from(roles.rhs), usize::from(roles.lhs)) } else { (usize::from(roles.lhs), usize::from(roles.rhs)) };
    let (m_set, n_set) = if swap { (roles.n, roles.m) } else { (roles.m, roles.n) };
    let in_order = |set: crate::jit::comprehension::roles::DimMask| -> SmallVec<[usize; 4]> { order.iter().copied().filter(|&d| set.contains(d)).collect() };
    let (m_dims, n_dims, b_dims) = (in_order(m_set), in_order(n_set), in_order(roles.batch));
    let k_dims: SmallVec<[usize; 4]> = roles.k.dims().collect();
    let extent = |dims: &[usize]| dims.iter().map(|&d| form.space[d].extent).product::<u64>();
    let (m, n, k, batch) = (extent(&m_dims), extent(&n_dims), extent(&k_dims), extent(&b_dims));
    let limit = u64::from(u32::MAX);
    if [m, n, k, batch, m * k * batch, k * n * batch, m * n * batch].iter().any(|&x| x > limit) {
        return None;
    }
    let lhs_axes: SmallVec<[usize; 4]> = k_dims.iter().chain(&m_dims).chain(&b_dims).copied().collect();
    let rhs_axes: SmallVec<[usize; 4]> = n_dims.iter().chain(&k_dims).chain(&b_dims).copied().collect();
    if lhs_axes.len() > 4 || rhs_axes.len() > 4 {
        return None;
    }
    // Every input must read only the dims of the operand it is on.
    for (side, axes) in [(lhs_side, &lhs_axes), (rhs_side, &rhs_axes)] {
        for &j in &sides[side] {
            let access = &form.inputs[usize::from(j)];
            if (0..form.space.len()).any(|d| form.space[d].extent > 1 && crate::jit::comprehension::roles::uses(access, d) && !axes.contains(&d)) {
                return None;
            }
        }
    }
    let [lhs_factors, rhs_factors] = if lhs_side == 0 { [factors[0].clone(), factors[1].clone()] } else { [factors[1].clone(), factors[0].clone()] };
    // What the kernel computes per pair of loads.
    let instrs = std::sync::Arc::clone(&form.body.instrs);
    let mac = if separation.is_some() {
        if part == CombinePart::Add {
            Mac::Fma
        } else {
            let mul = Instr { op: Op::new(OpKind::Binary(BinOp::Mul), ElemTy::F32), args: [Ref::Input(0), Ref::Input(1), Ref::Unused] };
            Mac::Body { instrs: std::sync::Arc::new(vec![mul]), out: 0, combine: part }
        }
    } else {
        let (l, r) = (sides[lhs_side][0], sides[rhs_side][0]);
        let out = skip_id(&instrs, Ref::Tmp(form.body.outs[0]));
        let product = matches!(out, Ref::Tmp(t) if matches!(instrs[t as usize].op.kind, OpKind::Binary(BinOp::Mul)) && {
            let args = &instrs[t as usize].args;
            (args[0] == Ref::Input(u32::from(l)) && args[1] == Ref::Input(u32::from(r))) || (args[0] == Ref::Input(u32::from(r)) && args[1] == Ref::Input(u32::from(l)))
        });
        if product && part == CombinePart::Add {
            Mac::Fma
        } else {
            // The loads of the left operand are input 0 of the multiply-add, those of the right one input 1.
            let remap = |input: u32| if input == u32::from(l) { 0 } else { 1 };
            let mapped: Vec<Instr> = instrs
                .iter()
                .map(|instr| {
                    let mut instr = instr.clone();
                    for arg in &mut instr.args {
                        if let Ref::Input(j) = *arg {
                            *arg = Ref::Input(remap(j));
                        }
                    }
                    instr
                })
                .collect();
            Mac::Body { instrs: std::sync::Arc::new(mapped), out: form.body.outs[0], combine: part }
        }
    };
    Some(Prepared { form, factors: [lhs_factors, rhs_factors], axes: [lhs_axes, rhs_axes], m, n, k, batch, mac, instrs })
}

/// The points of the operand over `axes`.
fn extent_len(form: &Comprehension, axes: &[usize]) -> usize {
    axes.iter().map(|&d| form.space[d].extent as usize).product()
}

/// The nodes of one side's operand program: the views of its inputs and the instructions over them.
struct Operand<'a> {
    plan: &'a Prepared,
    ctx: &'a Context,
    axes: &'a [usize],
    extent: SmallVec<[u64; 4]>,
    len: usize,
    views: HashMap<u32, Arc<Node>>,
    temps: HashMap<u32, Arc<Node>>,
    location: &'static Location<'static>,
}

impl Operand<'_> {
    /// The value `r` names, as an operand of a node.
    fn value(&mut self, r: Ref) -> Result<Src, Failure> {
        match r {
            Ref::Input(j) => {
                if let Some(view) = self.views.get(&j) {
                    return Ok(Src::Node(Arc::clone(view)));
                }
                let view = view_node(self.ctx, &self.plan.form, j as usize, self.axes, &Origin6([0; MAX_DIMS]), &self.extent, self.location)?;
                self.views.insert(j, Arc::clone(&view));
                Ok(Src::Node(view))
            }
            Ref::Scalar(k) => Ok(Src::Scalar(self.plan.form.body.scalars[k as usize])),
            Ref::Tmp(t) => {
                if let Some(node) = self.temps.get(&t) {
                    return Ok(Src::Node(Arc::clone(node)));
                }
                let instr = &self.plan.instrs[t as usize];
                let mut args = [Src::unused(), Src::unused(), Src::unused()];
                for (slot, &arg) in args.iter_mut().zip(&instr.args) {
                    *slot = self.value(arg)?;
                }
                let node = jit::create(self.ctx, self.len, instr.op, args, "comprehension", self.location)?;
                self.temps.insert(t, Arc::clone(&node));
                Ok(Src::Node(node))
            }
            Ref::Unused | Ref::Sub(_) => Ok(Src::unused()),
        }
    }
}

impl Prepared {
    /// The program of a side: the product of its factors over the views of its inputs, one node of `[k.., m.., b..]` points.
    fn operand(&self, ctx: &Context, side: usize, location: &'static Location<'static>) -> Result<Arc<Node>, Failure> {
        let axes = &self.axes[side];
        let extent: SmallVec<[u64; 4]> = axes.iter().map(|&d| self.form.space[d].extent).collect();
        let mut program = Operand { plan: self, ctx, axes, extent, len: extent_len(&self.form, axes), views: HashMap::new(), temps: HashMap::new(), location };
        let mut product: Option<Arc<Node>> = None;
        for &factor in &self.factors[side] {
            let node = match program.value(factor)? {
                Src::Node(node) => node,
                _ => return Err(Failure::from(LumaError::Unsupported("an operand of a contraction is a constant".into()))),
            };
            product = Some(match product {
                None => node,
                Some(acc) => jit::create(ctx, program.len, Op::new(OpKind::Binary(BinOp::Mul), ElemTy::F32), [Src::Node(acc), Src::Node(node), Src::unused()], "comprehension", location)?,
            });
        }
        Ok(product.expect("a side has a factor"))
    }

    /// Runs the product: one launch of the `T32` kernel of the form over the two operand programs.
    pub(crate) fn run(&self, emit: &mut Emit<'_>, location: &'static Location<'static>) -> Result<Arc<Storage>, Failure> {
        let ctx = emit.ctx();
        let (m, n, k, batch) = (self.m as u32, self.n as u32, self.k as u32, self.batch as u32);
        let sides = [MatmulSide::Pending(self.operand(ctx, 0, location)?), MatmulSide::Pending(self.operand(ctx, 1, location)?)];
        let tile = MatmulTile::T32;
        let aligned = m.is_multiple_of(32) && n.is_multiple_of(32) && k.is_multiple_of(crate::matmul::SLICE);
        let (columns, rows) = (n.div_ceil(32), m.div_ceil(32));
        let max_groups = ctx.max_workgroups_per_dimension();
        let [x, y, _] = if columns <= max_groups && rows <= max_groups { [columns, rows, 1] } else { crate::ops::fold_grid(columns.saturating_mul(rows), max_groups) };
        if batch > max_groups || x > max_groups || y > max_groups {
            return Err(LumaError::KernelGridTooLarge { workgroups: [x, y, batch], max: max_groups }.into());
        }
        let len = (self.m * self.n * self.batch) as usize;
        let storage = Arc::new(Storage::new(ctx, len, ElemTy::F32.layout()));
        let pin = storage.pin_single()?;
        let id = intern_form(ContractForm { mac: self.mac.clone(), batch: self.batch > 1 });
        let kernel = jit::MatmulKernel { tile, aligned, split: false, form: id };
        let extra: &[u32] = if self.batch > 1 { &[batch, m * k, k * n, m * n] } else { &[] };
        jit::matmul_with(ctx, sides, kernel, &[m, n, k, k], extra, [x, y, batch.max(1)], RawArg::write(pin.buffer(), storage.byte_len()))?;
        drop(pin);
        jit::check_finite(ctx, &storage, ElemTy::F32)?;
        Ok(storage)
    }
}

/// What the product kernel does with a pair of loaded values (W2-R0 §5.3). The default is `acc = fma(a, b, acc)`; a generic
/// contraction (a body other than `x0 * x1`, a combiner other than `Add`, a batch) names its own, interned so that [`Tail::Matmul`](crate::jit::program::Tail) stays
/// `Copy`: the kernel of the product is the T32 tile with `luma_mac` as the multiply-add.
#[derive(Clone, PartialEq, Eq, Debug)]
pub(crate) struct ContractForm {
    pub(crate) mac: Mac,
    /// A batch of products in the third grid dimension: the uniform has `nb, sa, sb, sc` after `kc` (the batch count and the strides
    /// of the batch in `lhs`, `rhs` and the output, in elements).
    pub(crate) batch: bool,
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub(crate) enum Mac {
    /// `acc = fma(a, b, acc)`, the accumulator starting at 0: today's kernel.
    Fma,
    /// `acc = combine(acc, luma_mac(a, b))`, `luma_mac(a0, a1)` computed by the instructions of `instrs` (inputs 0 and 1 are the
    /// loads of the left and right operand), the accumulator starting at the identity of `combine`.
    Body { instrs: std::sync::Arc<Vec<crate::jit::program::Instr>>, out: u32, combine: crate::jit::comprehension::CombinePart },
}

impl ContractForm {
    pub(crate) const DEFAULT: ContractForm = ContractForm { mac: Mac::Fma, batch: false };

    /// Whether this is today's kernel.
    pub(crate) fn is_default(&self) -> bool {
        *self == ContractForm::DEFAULT
    }

    /// The accumulator update of `acc` with the loads `a` and `b`.
    pub(crate) fn update(&self, acc: &str, a: &str, b: &str) -> String {
        use crate::jit::comprehension::CombinePart;
        match &self.mac {
            Mac::Fma => format!("{acc} = fma({a}, {b}, {acc});"),
            Mac::Body { combine, .. } => {
                let mac = format!("luma_mac({a}, {b})");
                match combine {
                    CombinePart::Add => format!("{acc} = {acc} + {mac};"),
                    CombinePart::Mul => format!("{acc} = {acc} * {mac};"),
                    CombinePart::Min => format!("{acc} = min({acc}, {mac});"),
                    CombinePart::Max => format!("{acc} = max({acc}, {mac});"),
                    other => unreachable!("a contraction does not combine with {other:?}"),
                }
            }
        }
    }

    /// The WGSL literal the accumulator starts at.
    pub(crate) fn identity(&self) -> &'static str {
        use crate::jit::comprehension::CombinePart;
        match &self.mac {
            Mac::Fma => "0.0",
            Mac::Body { combine, .. } => match combine {
                CombinePart::Add => "0.0",
                CombinePart::Mul => "1.0",
                CombinePart::Min => "bitcast<f32>(0x7f800000u)",
                CombinePart::Max => "bitcast<f32>(0xff800000u)",
                other => unreachable!("a contraction does not combine with {other:?}"),
            },
        }
    }

    /// The function `luma_mac`, written by the printer of the generated kernels.
    pub(crate) fn mac_function(&self) -> String {
        match &self.mac {
            Mac::Fma => String::new(),
            Mac::Body { instrs, out, .. } => crate::jit::codegen::scalar_function("luma_mac", instrs, *out as usize, 2),
        }
    }
}

/// The interned forms of the generic contractions; id 0 is [`ContractForm::DEFAULT`].
static FORMS: std::sync::Mutex<Vec<std::sync::Arc<ContractForm>>> = std::sync::Mutex::new(Vec::new());

/// The id of `form` (0 for the default): equal forms have one id for the life of the process, so that a [`Tail`](crate::jit::program::Tail)
/// naming it is `Copy + Hash` and the kernels of equal forms are one.
pub(crate) fn intern_form(form: ContractForm) -> u16 {
    if form.is_default() {
        return 0;
    }
    let mut forms = FORMS.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    if let Some(at) = forms.iter().position(|known| **known == form) {
        return at as u16 + 1;
    }
    forms.push(std::sync::Arc::new(form));
    u16::try_from(forms.len()).expect("fewer than 65536 generic contraction forms")
}

/// The form with id `id`.
pub(crate) fn form(id: u16) -> std::sync::Arc<ContractForm> {
    match id {
        0 => std::sync::Arc::new(ContractForm::DEFAULT),
        _ => std::sync::Arc::clone(&FORMS.lock().unwrap_or_else(std::sync::PoisonError::into_inner)[usize::from(id) - 1]),
    }
}
