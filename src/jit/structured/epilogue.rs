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
use crate::jit::{ElemTy, Scalar, Src};

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
