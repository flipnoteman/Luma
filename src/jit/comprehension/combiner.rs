//! How the values of a reduce dim combine.

use smallvec::SmallVec;

use crate::builtin::ReduceOp;
use crate::jit::{ElemTy, Scalar};

/// What combines two partial values.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum CombinePart {
    Add,
    Mul,
    Min,
    Max,
    Or,
    And,
    /// The index (along the reduce dim) of the first largest value; a NaN is the largest.
    ArgMax,
    /// The index of the first smallest value; a NaN is the smallest.
    ArgMin,
}

/// One part per body output (not recursive: a tuple holding combine ops would be an infinitely sized type).
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub(crate) enum CombineOp {
    Single(CombinePart),
    Tuple(SmallVec<[CombinePart; 4]>),
}

impl CombineOp {
    /// The parts, one per body output.
    pub(crate) fn parts(&self) -> &[CombinePart] {
        match self {
            CombineOp::Single(part) => std::slice::from_ref(part),
            CombineOp::Tuple(parts) => parts,
        }
    }
}

/// How a combiner makes up for the rounding of a float sum (lowered through the double-float library, see `Compensation` of W2-R0 5.6).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum Compensation {
    None,
    Neumaier,
    Dot2,
}

#[derive(Clone, Debug)]
pub(crate) struct Combiner {
    pub(crate) op: CombineOp,
    /// The value that combines to the other one, per part (of the type of the accesses).
    pub(crate) identity: SmallVec<[Scalar; 4]>,
    pub(crate) commutative: bool,
    /// Whether any order of combining gives the same bits (see [`reorder_exact`]).
    pub(crate) reorder_exact: bool,
    pub(crate) compensation: Compensation,
}

impl Combiner {
    /// The combiner of the ops of a reduction over values of type `ty`: one part per op.
    pub(crate) fn of_ops(ops: &[ReduceOp], ty: ElemTy) -> Combiner {
        let part = |op: &ReduceOp| part_of(*op);
        let op = match ops {
            [only] => CombineOp::Single(part(only)),
            _ => CombineOp::Tuple(ops.iter().map(part).collect()),
        };
        Combiner::of(op, ty)
    }

    /// The combiner of `op` over values of type `ty`, with the identities and properties of its parts.
    pub(crate) fn of(op: CombineOp, ty: ElemTy) -> Combiner {
        let parts = op.parts();
        Combiner {
            identity: parts.iter().map(|&part| identity(part, ty)).collect(),
            commutative: true,
            reorder_exact: parts.iter().all(|&part| reorder_exact(part, ty)),
            compensation: Compensation::None,
            op,
        }
    }
}

/// The part a reduction op combines with (a sum of squares adds the squares, which its body makes).
pub(crate) fn part_of(op: ReduceOp) -> CombinePart {
    match op {
        ReduceOp::Sum | ReduceOp::SumSquares => CombinePart::Add,
        ReduceOp::Min => CombinePart::Min,
        ReduceOp::Max => CombinePart::Max,
        ReduceOp::Prod => CombinePart::Mul,
        ReduceOp::ArgMax => CombinePart::ArgMax,
        ReduceOp::ArgMin => CombinePart::ArgMin,
    }
}

/// Whether combining in any order gives the same bits: `Min`/`Max`, `Or`/`And`, integer `Add`/`Mul` and the arg parts (ties go
/// to the lowest index) do; float `Add`/`Mul` (rounding) and the double-float sums do not. **[INFERENCE]** float `min`/`max` agree
/// in any order except for the sign of a zero result.
pub(crate) fn reorder_exact(part: CombinePart, ty: ElemTy) -> bool {
    match part {
        CombinePart::Add | CombinePart::Mul => !is_float(ty),
        _ => true,
    }
}

fn is_float(ty: ElemTy) -> bool {
    matches!(ty, ElemTy::F32 | ElemTy::F16 | ElemTy::BF16 | ElemTy::F64 | ElemTy::Df32 | ElemTy::C32)
}

/// The identity of `part` for values of type `ty`, as the register value the scalars of a program hold (a half widened to `f32`,
/// a small signed integer sign-extended, a pair of `f32`s as `hi | lo << 32`). `ArgMax`/`ArgMin` have the identity of the
/// `Max`/`Min` of the values they compare.
pub(crate) fn identity(part: CombinePart, ty: ElemTy) -> Scalar {
    let bits = match part {
        CombinePart::Add | CombinePart::Or => 0,
        CombinePart::Mul => one(ty),
        CombinePart::Min | CombinePart::ArgMin => highest(ty),
        CombinePart::Max | CombinePart::ArgMax => lowest(ty),
        CombinePart::And => all_ones(ty),
    };
    Scalar { ty, bits }
}

/// A pair of `f32` words as the bits of a two-word scalar.
fn pair(hi: f32, lo: f32) -> u64 {
    u64::from(hi.to_bits()) | u64::from(lo.to_bits()) << 32
}

fn one(ty: ElemTy) -> u64 {
    match ty {
        ElemTy::F32 | ElemTy::F16 | ElemTy::BF16 => u64::from(1f32.to_bits()),
        ElemTy::F64 => 1f64.to_bits(),
        // The product of the components of a `C32` reduces componentwise (see `ReduceElem::combine`).
        ElemTy::C32 => pair(1.0, 1.0),
        ElemTy::Df32 => pair(1.0, 0.0),
        _ => 1,
    }
}

fn highest(ty: ElemTy) -> u64 {
    match ty {
        ElemTy::F32 | ElemTy::F16 | ElemTy::BF16 => u64::from(f32::INFINITY.to_bits()),
        ElemTy::F64 => f64::INFINITY.to_bits(),
        ElemTy::C32 => pair(f32::INFINITY, f32::INFINITY),
        ElemTy::Df32 => pair(f32::INFINITY, 0.0),
        ElemTy::U32 => u64::from(u32::MAX),
        ElemTy::I32 => u64::from(i32::MAX as u32),
        ElemTy::Bool => 1,
        ElemTy::U8 => u64::from(u8::MAX),
        ElemTy::I8 => u64::from(i8::MAX as u32),
        ElemTy::U16 => u64::from(u16::MAX),
        ElemTy::I16 => u64::from(i16::MAX as u32),
    }
}

fn lowest(ty: ElemTy) -> u64 {
    match ty {
        ElemTy::F32 | ElemTy::F16 | ElemTy::BF16 => u64::from(f32::NEG_INFINITY.to_bits()),
        ElemTy::F64 => f64::NEG_INFINITY.to_bits(),
        ElemTy::C32 => pair(f32::NEG_INFINITY, f32::NEG_INFINITY),
        ElemTy::Df32 => pair(f32::NEG_INFINITY, 0.0),
        ElemTy::I32 => u64::from(i32::MIN as u32),
        ElemTy::I8 => u64::from(i32::from(i8::MIN) as u32),
        ElemTy::I16 => u64::from(i32::from(i16::MIN) as u32),
        ElemTy::U32 | ElemTy::Bool | ElemTy::U8 | ElemTy::U16 => 0,
    }
}

fn all_ones(ty: ElemTy) -> u64 {
    match ty {
        ElemTy::Bool => 1,
        ElemTy::U8 => u64::from(u8::MAX),
        ElemTy::U16 => u64::from(u16::MAX),
        // Signed registers are sign-extended: all ones is -1.
        _ => u64::from(u32::MAX),
    }
}
