//! The epilogue program (W3-U4 §A): a small elementwise program a structured value applies to each output as it stores it. It is
//! what `Epilogue::Program` of a [`Comprehension`](crate::jit::comprehension::Comprehension) holds, and what the backends' hooks
//! (`luma_store` in the shader kernels, the CPU tile function, the CUDA `luma_epi` call) are generated from.
//!
//! The program is backend neutral: [`Instr`]s over `f32` where [`Ref::Acc`] is the combined value (for a contraction, the sum after
//! the whole K loop), [`Ref::Input`]`(j)` is the extra operand `j` of [`EpiProgram::inputs`] read at the output element through its
//! [`EpiMap`], and [`Ref::Scalar`]`(k)` is a uniform scalar. A program that does not use `Ref::Acc` keeps the meaning
//! `Ref::Input(j)` = combined part `j` that the window forms (bilateral, Sobel) gave their epilogues in W2-R0, with no inputs.
//!
//! What the backends see of a program is its [`EpiKey`] (instruction structure and input kinds, not values), so that two programs that
//! differ only in a bias buffer or a scale share their kernel.

#![allow(dead_code, reason = "W3-U4 part A defines the types; absorption, partners and the backends' hooks use them next")]

use std::hash::{Hash, Hasher};
use std::sync::Arc;

use smallvec::SmallVec;

use crate::jit::program::{Instr, Ref};
use crate::jit::{ElemTy, Op, OpKind, Scalar, Src};

/// The most instructions an epilogue holds.
pub(crate) const MAX_EPILOGUE_INSTRS: usize = 16;
/// The most extra operands an epilogue reads.
pub(crate) const MAX_EPI_INPUTS: usize = 3;
/// The most uniform scalars of an epilogue.
pub(crate) const MAX_EPI_SCALARS: usize = 8;

/// The program applied to the combined values of a structured value, see the module documentation.
#[derive(Clone, Debug)]
pub(crate) struct EpiProgram {
    /// The pointwise program; `Ref::Acc` is the combined value, `Ref::Input(j)` epilogue input `j`, `Ref::Scalar(k)` scalar `k`.
    pub(crate) instrs: Arc<Vec<Instr>>,
    /// The extra operands.
    pub(crate) inputs: SmallVec<[EpiInput; MAX_EPI_INPUTS]>,
    /// The values of the uniform scalars (`Ref::Scalar(k)` is `scalars[k]`); they are uniform words of the launch, not part of the key.
    pub(crate) scalars: SmallVec<[Scalar; 2]>,
    /// The instruction indices that are stored; more than one only for the partner outputs of one base value (`z = x·W + b` and
    /// `a = relu(z)` store two instructions of one program).
    pub(crate) outputs: SmallVec<[u32; 2]>,
}

/// An extra operand of an epilogue.
#[derive(Clone, Debug)]
pub(crate) struct EpiInput {
    pub(crate) src: Src,
    pub(crate) map: EpiMap,
    pub(crate) ty: ElemTy,
}

/// Which output element an epilogue input is read at.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum EpiMap {
    /// The element at the output's own index (the same shape as the output).
    Point,
    /// Depends on the output row only (axis 1 of the luma shape): `x[row]`.
    AlongRows,
    /// Depends on the output column only (axis 0): a bias `[N]` seen as `[N, M]`. For an axis reduction, the line index.
    AlongCols,
    /// One value for every element.
    Scalar,
}

/// The structure of an epilogue: what a generated kernel depends on. Equal keys, one kernel.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub(crate) struct EpiKey {
    pub(crate) instrs: Arc<Vec<Instr>>,
    pub(crate) inputs: SmallVec<[(EpiMap, ElemTy); MAX_EPI_INPUTS]>,
    /// The types of the scalars, in order.
    pub(crate) scalars: SmallVec<[ElemTy; 2]>,
    pub(crate) outputs: SmallVec<[u32; 2]>,
}

impl EpiProgram {
    /// A program over combined parts with no inputs of its own: the epilogue of a window form (`Ref::Input(j)` is part `j`), whose
    /// last instruction is its value.
    pub(crate) fn of_parts(instrs: Vec<Instr>) -> EpiProgram {
        let last = instrs.len().saturating_sub(1) as u32;
        EpiProgram { instrs: Arc::new(instrs), inputs: SmallVec::new(), scalars: SmallVec::new(), outputs: SmallVec::from_elem(last, 1) }
    }

    /// Whether the program reads the combined value as `Ref::Acc` (and `Ref::Input(j)` is then an epilogue input, not a part).
    pub(crate) fn uses_acc(&self) -> bool {
        self.instrs.iter().any(|instr| instr.args.contains(&Ref::Acc))
    }

    /// The structure the kernels are keyed on.
    pub(crate) fn key(&self) -> EpiKey {
        EpiKey {
            instrs: Arc::clone(&self.instrs),
            inputs: self.inputs.iter().map(|input| (input.map, input.ty)).collect(),
            scalars: self.scalars.iter().map(|scalar| scalar.ty).collect(),
            outputs: self.outputs.clone(),
        }
    }

    /// Hashes the structure (what [`EpiProgram::key`] holds), not the buffers or scalar values.
    pub(crate) fn hash_structure<H: Hasher>(&self, h: &mut H) {
        self.instrs.hash(h);
        for input in &self.inputs {
            input.map.hash(h);
            input.ty.hash(h);
        }
        for scalar in &self.scalars {
            scalar.ty.hash(h);
        }
        self.outputs.hash(h);
    }

    /// Whether two programs have the same structure: partners may differ only in their inputs' buffers and scalar values.
    pub(crate) fn same_structure(&self, other: &EpiProgram) -> bool {
        self.key() == other.key()
    }

    /// Checks the program against `parts` combined parts (the parts a program without `Ref::Acc` reads as `Ref::Input`).
    pub(crate) fn validate(&self, parts: usize) -> Result<(), &'static str> {
        if self.instrs.is_empty() || self.instrs.len() > MAX_EPILOGUE_INSTRS {
            return Err("an epilogue has one to sixteen instructions");
        }
        if self.inputs.len() > MAX_EPI_INPUTS || self.scalars.len() > MAX_EPI_SCALARS {
            return Err("an epilogue has too many inputs or scalars");
        }
        if self.outputs.is_empty() || self.outputs.iter().any(|&out| out as usize >= self.instrs.len()) {
            return Err("an epilogue output is not an instruction");
        }
        let acc = self.uses_acc();
        for (at, instr) in self.instrs.iter().enumerate() {
            for arg in instr.args {
                match arg {
                    Ref::Input(j) if acc && j as usize >= self.inputs.len() => return Err("an epilogue input is not an input"),
                    Ref::Input(j) if !acc && j as usize >= parts => return Err("an epilogue input is not a combined part"),
                    Ref::Scalar(k) if k as usize >= self.scalars.len() => return Err("an epilogue scalar is not a scalar"),
                    Ref::Tmp(t) if t as usize >= at => return Err("an epilogue temporary is not an earlier instruction"),
                    Ref::Sub(_) => return Err("an epilogue has no sub-programs"),
                    _ => {}
                }
            }
        }
        Ok(())
    }
}

impl EpiProgram {
    /// One step over the combined value: `op` applied to `Ref::Acc`, with the scalar `scalar` (`op(acc, s)`, or `op(s, acc)` if
    /// `scalar_left`) or the epilogue input `input` (always on the right: `op(acc, input)`).
    pub(crate) fn step(kind: OpKind, operand: StepOperand) -> EpiProgram {
        let mut scalars = SmallVec::new();
        let mut inputs = SmallVec::new();
        let args = match operand {
            StepOperand::None => [Ref::Acc, Ref::Unused, Ref::Unused],
            StepOperand::Scalar { value, scalar_left } => {
                scalars.push(value);
                if scalar_left { [Ref::Scalar(0), Ref::Acc, Ref::Unused] } else { [Ref::Acc, Ref::Scalar(0), Ref::Unused] }
            }
            StepOperand::Input(input) => {
                inputs.push(input);
                [Ref::Acc, Ref::Input(0), Ref::Unused]
            }
        };
        EpiProgram { instrs: Arc::new(vec![Instr { op: Op::new(kind, ElemTy::F32), args }]), inputs, scalars, outputs: SmallVec::from_elem(0, 1) }
    }

    /// This program followed by `delta`, whose `Ref::Acc` is this program's value: `None` if the result has more than
    /// [`MAX_EPILOGUE_INSTRS`] instructions, [`MAX_EPI_INPUTS`] inputs or [`MAX_EPI_SCALARS`] scalars. Only for programs of one output.
    pub(crate) fn then(&self, delta: &EpiProgram) -> Option<EpiProgram> {
        let (n, inputs, scalars) = (self.instrs.len(), self.inputs.len() as u32, self.scalars.len() as u32);
        if n + delta.instrs.len() > MAX_EPILOGUE_INSTRS || self.inputs.len() + delta.inputs.len() > MAX_EPI_INPUTS || self.scalars.len() + delta.scalars.len() > MAX_EPI_SCALARS || self.outputs.len() != 1 {
            return None;
        }
        let value = Ref::Tmp(self.outputs[0]);
        let mut instrs: Vec<Instr> = self.instrs.as_ref().clone();
        for instr in delta.instrs.iter() {
            let shift = |arg: Ref| match arg {
                Ref::Acc => value,
                Ref::Tmp(t) => Ref::Tmp(t + n as u32),
                Ref::Input(j) => Ref::Input(j + inputs),
                Ref::Scalar(k) => Ref::Scalar(k + scalars),
                other => other,
            };
            instrs.push(Instr { op: instr.op, args: instr.args.map(shift) });
        }
        let last = instrs.len() as u32 - 1;
        Some(EpiProgram {
            instrs: Arc::new(instrs),
            inputs: self.inputs.iter().chain(&delta.inputs).cloned().collect(),
            scalars: self.scalars.iter().chain(&delta.scalars).copied().collect(),
            outputs: SmallVec::from_elem(last, 1),
        })
    }
}

/// The second operand of an [`EpiProgram::step`].
pub(crate) enum StepOperand {
    None,
    Scalar { value: Scalar, scalar_left: bool },
    Input(EpiInput),
}

// -- Axis reductions (W3-U4 D1) --

/// Whether `instr` may be part of the epilogue of an axis reduction: an `f32` unary or binary operation that the generated WGSL
/// writes as an expression (no helper function, no integers, no complex numbers).
pub(crate) fn axis_instr_allowed(instr: &Instr) -> bool {
    use crate::builtin::UnOp;
    use crate::jit::OpKind;
    instr.op.out_ty() == ElemTy::F32
        && match instr.op.kind {
            OpKind::Unary(op) => !op.needs_helper() && !op.is_bitwise() && op != UnOp::Conj,
            OpKind::Binary(op) => !op.needs_helper() && !op.is_bitwise(),
            _ => false,
        }
}

/// The structure of the epilogues of the ops of one reducing pass, by handle: a pass is a kernel cache key and `Copy`, so it names
/// its epilogues by a small id (the pattern of `builtin::ReduceList::merged`). Entry `j` is the epilogue of op `j`; only the last
/// pass of an axis reduction has one.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) struct AxisEpiId(u32);

#[derive(Default)]
struct AxisEpis {
    specs: Vec<Arc<Vec<Option<EpiKey>>>>,
    ids: crate::gpu::hash::FxMap<Vec<Option<EpiKey>>, u32>,
}

static AXIS_EPIS: std::sync::LazyLock<std::sync::Mutex<AxisEpis>> = std::sync::LazyLock::new(Default::default);

impl AxisEpiId {
    /// The handle of `ops`, the epilogue of every op of a pass (`None` for an op that has none).
    pub(crate) fn intern(ops: Vec<Option<EpiKey>>) -> AxisEpiId {
        let mut table = crate::gpu::lock(&AXIS_EPIS);
        if let Some(&id) = table.ids.get(&ops) {
            return AxisEpiId(id);
        }
        let id = table.specs.len() as u32;
        table.specs.push(Arc::new(ops.clone()));
        table.ids.insert(ops, id);
        AxisEpiId(id)
    }

    fn spec(self) -> Arc<Vec<Option<EpiKey>>> {
        Arc::clone(&crate::gpu::lock(&AXIS_EPIS).specs[self.0 as usize])
    }

    /// Whether op `j` has an epilogue.
    pub(crate) fn has(self, j: usize) -> bool {
        self.spec().get(j).is_some_and(Option::is_some)
    }

    /// The extra read bindings of the pass.
    pub(crate) fn input_count(self) -> usize {
        self.spec().iter().flatten().map(|key| key.inputs.len()).sum()
    }

    /// The fields of the `Params` uniform that hold the scalars, `es{j}_{k}: f32,`, in the order of the ops and their scalars.
    pub(crate) fn fields(self) -> String {
        use std::fmt::Write;
        let mut out = String::new();
        for (j, key) in self.spec().iter().enumerate() {
            for k in 0..key.as_ref().map_or(0, |key| key.scalars.len()) {
                let _ = write!(out, "es{j}_{k}: f32, ");
            }
        }
        out
    }

    /// The declarations of the extra inputs `epi{j}_{k}`, from binding `*binding` on.
    pub(crate) fn bindings(self, binding: &mut usize) -> String {
        use std::fmt::Write;
        let mut out = String::new();
        for (j, key) in self.spec().iter().enumerate() {
            for k in 0..key.as_ref().map_or(0, |key| key.inputs.len()) {
                let _ = writeln!(out, "@group(1) @binding({}) var<storage, read> epi{j}_{k}: array<f32>;", *binding);
                *binding += 1;
            }
        }
        out
    }

    /// `fn luma_epi{j}(at: u32, acc: f32) -> f32` for every op with an epilogue: the program on the value `acc` of line `at`.
    pub(crate) fn functions(self) -> String {
        use crate::jit::OpKind;
        use std::fmt::Write;
        let mut out = String::new();
        for (j, key) in self.spec().iter().enumerate() {
            let Some(key) = key else { continue };
            let _ = writeln!(out, "fn luma_epi{j}(at: u32, acc: f32) -> f32 {{");
            let name = |arg: Ref| -> String {
                match arg {
                    Ref::Acc => "acc".to_owned(),
                    Ref::Input(k) => format!("epi{j}_{k}[at]"),
                    Ref::Scalar(k) => format!("p.es{j}_{k}"),
                    Ref::Tmp(t) => format!("t{t}"),
                    Ref::Unused | Ref::Sub(_) => unreachable!("an axis epilogue reads the value, inputs, scalars and temporaries"),
                }
            };
            for (n, instr) in key.instrs.iter().enumerate() {
                let expr = match instr.op.kind {
                    OpKind::Unary(op) => crate::builtin::substitute(op.expr(), &name(instr.args[0]), ""),
                    OpKind::Binary(op) => crate::builtin::substitute(op.expr(), &name(instr.args[0]), &name(instr.args[1])),
                    _ => unreachable!("an axis epilogue has unary and binary operations"),
                };
                let _ = writeln!(out, "    let t{n}: f32 = {expr};");
            }
            let last = key.outputs.first().copied().unwrap_or(key.instrs.len() as u32 - 1);
            let _ = writeln!(out, "    return t{last};\n}}");
        }
        out
    }
}

// -- Whole reductions (W3-U4 D2) --

/// The most steps (instructions) a reduction of the whole array absorbs: the limit of the `Finish` this program form replaced.
pub(crate) const MAX_REDUCE_STEPS: usize = 4;

/// The epilogue of a whole reduction in a kernel's cache key: a program over `Ref::Acc` of unary and binary `f32` operations whose
/// binary operands are scalars (the words of the scalars are uniforms of the last pass, not part of the key), by handle, so that a
/// [`ReduceList`](crate::builtin::ReduceList) stays a small `Copy` value. Interned by structure (see [`EpiKey`]); the default is none.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub(crate) struct EpiId(Option<std::num::NonZeroU32>);

#[derive(Default)]
struct WholeEpis {
    keys: Vec<Arc<EpiKey>>,
    ids: crate::gpu::hash::FxMap<EpiKey, u32>,
}

static WHOLE_EPIS: std::sync::LazyLock<std::sync::Mutex<WholeEpis>> = std::sync::LazyLock::new(Default::default);

impl EpiId {
    /// The handle of `program`, none for no program.
    pub(crate) fn of(program: Option<&EpiProgram>) -> EpiId {
        let Some(program) = program else { return EpiId(None) };
        let key = program.key();
        let mut table = crate::gpu::lock(&WHOLE_EPIS);
        if let Some(&id) = table.ids.get(&key) {
            return EpiId(std::num::NonZeroU32::new(id));
        }
        table.keys.push(Arc::new(key.clone()));
        let id = table.keys.len() as u32;
        table.ids.insert(key, id);
        EpiId(std::num::NonZeroU32::new(id))
    }

    pub(crate) fn is_none(self) -> bool {
        self.0.is_none()
    }

    fn key(self) -> Option<Arc<EpiKey>> {
        let id = self.0?.get() as usize;
        Some(Arc::clone(&crate::gpu::lock(&WHOLE_EPIS).keys[id - 1]))
    }

    /// The fields of the `Params` uniform that hold the scalars: `e{tag}{k}: f32,` for scalar `k`.
    pub(crate) fn fields(self, tag: &str) -> String {
        use std::fmt::Write;
        let mut out = String::new();
        for k in 0..self.key().map_or(0, |key| key.scalars.len()) {
            let _ = write!(out, "e{tag}{k}: f32, ");
        }
        out
    }

    /// WGSL statements applying the program to the `f32` expression `value` (`let {prefix}{n}` is the result of instruction `n`) and
    /// the name of the final result (`value` if there is no program); scalar `k` is named `field(k)`.
    pub(crate) fn statements(self, value: &str, prefix: &str, field: impl Fn(usize) -> String) -> (String, String) {
        use std::fmt::Write;
        let Some(key) = self.key() else { return (String::new(), value.to_owned()) };
        let name = |arg: Ref| -> String {
            match arg {
                Ref::Acc => value.to_owned(),
                Ref::Tmp(t) => format!("{prefix}{t}"),
                Ref::Scalar(k) => field(k as usize),
                Ref::Input(_) | Ref::Unused | Ref::Sub(_) => unreachable!("a whole reduction's epilogue reads the value, scalars and temporaries"),
            }
        };
        let mut out = String::new();
        for (n, instr) in key.instrs.iter().enumerate() {
            let expr = match instr.op.kind {
                OpKind::Unary(op) => crate::builtin::substitute(op.expr(), &name(instr.args[0]), ""),
                OpKind::Binary(op) => crate::builtin::substitute(op.expr(), &name(instr.args[0]), &name(instr.args[1])),
                _ => unreachable!("a whole reduction's epilogue has unary and binary operations"),
            };
            let _ = writeln!(out, "        let {prefix}{n}: f32 = {expr};");
        }
        let last = key.outputs.first().copied().unwrap_or(key.instrs.len() as u32 - 1);
        (out, format!("{prefix}{last}"))
    }
}

impl EpiProgram {
    /// The words of the scalars (`f32` bits), in order: what a last pass is given after its fixed uniform fields.
    pub(crate) fn words(&self) -> impl Iterator<Item = u32> + '_ {
        self.scalars.iter().map(|scalar| scalar.words()[0])
    }
}
