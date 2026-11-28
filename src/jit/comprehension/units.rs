//! The matrix-unit table (W3-R7): each unit is a fixed-size comprehension `C[m,n] += A[m,k]·B[k,n]` (its extents, operand and
//! accumulator types, the lanes that cooperate) plus the numerics class that permits it and the rates the Contraction strategies cost
//! it with. The strategies `GemmF32` / `GemmTf32` (and their ragged and padded siblings) read their rates and caps from these rows
//! instead of literals; the tiles a unit runs on are the backend's (`Backend::gemm_tiles(unit)`).

use crate::caps::Caps;
use crate::jit::ElemTy;
use crate::numerics::Reduced;
use std::panic::Location;

use crate::context::Context;
use luma_core::{Decision, DecisionSite, GemmSpec, GemmUnit, Verdict};

/// The identity of a row of [`UNITS`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum UnitId {
    /// TF32 tensor cores (CUDA `mma.m16n8k8`).
    CudaTf32,
    /// The exact `f32` kernels on the ordinary cores: the reference unit.
    Fma,
    /// `f16` tensor cores (CUDA `mma.m16n8k16`), `f32` accumulation.
    CudaF16,
    /// `bf16` tensor cores (CUDA `mma.m16n8k16`), `f32` accumulation.
    CudaBf16,
    /// `f16` cooperative matrices (`coop_mat16x16<f16>` through wgpu on Vulkan), `f32` accumulation.
    CoopF16,
}

/// One matrix unit.
#[derive(Debug)]
pub(crate) struct UnitDesc {
    pub(crate) id: UnitId,
    /// The unit as a [`luma_core::GemmSpec`] names it.
    pub(crate) gemm: GemmUnit,
    /// The unit's own `(m, n, k)` extents.
    pub(crate) shape: [u32; 3],
    /// Lanes that cooperate.
    pub(crate) subgroup: u32,
    /// Operand and accumulator types.
    pub(crate) a: ElemTy,
    pub(crate) b: ElemTy,
    pub(crate) acc: ElemTy,
    /// What the device must have.
    pub(crate) needs: Caps,
    /// The numerics bit that permits the unit.
    pub(crate) reduced: Reduced,
    /// Measured rate of the backend's own kernel at 4096^3 on an RTX 3090, flops per nanosecond, scaled by the device's
    /// `f32_ops_per_ns` against the 3090's calibration.
    pub(crate) rate: f64,
}

/// The units, in preference order.
pub(crate) static UNITS: [UnitDesc; 5] = [
    UnitDesc {
        id: UnitId::CudaTf32,
        gemm: GemmUnit::Tf32,
        shape: [16, 8, 8],
        subgroup: 32,
        a: ElemTy::F32,
        b: ElemTy::F32,
        acc: ElemTy::F32,
        needs: Caps::TF32,
        reduced: Reduced::TF32_MATMUL,
        rate: 27_900.0,
    },
    UnitDesc {
        id: UnitId::Fma,
        gemm: GemmUnit::Fma,
        shape: [1, 1, 1],
        subgroup: 1,
        a: ElemTy::F32,
        b: ElemTy::F32,
        acc: ElemTy::F32,
        needs: Caps::GEMM_TILES,
        reduced: Reduced::NONE,
        rate: 22_900.0,
    },
    // The half-precision units multiply exactly and accumulate in `f32` (`MATRIX_ACCUMULATE`: no operand is rounded below its stored
    // precision). Their rates are measured at 4096^3 on an RTX 3090 (see the CHANGELOG of W3-R7); no ragged cap: the kernels' edges are
    // in-kernel.
    UnitDesc {
        id: UnitId::CudaF16,
        gemm: GemmUnit::F16,
        shape: [16, 8, 16],
        subgroup: 32,
        a: ElemTy::F16,
        b: ElemTy::F16,
        acc: ElemTy::F32,
        needs: Caps::F16_MMA,
        reduced: Reduced::MATRIX_ACCUMULATE,
        rate: 55_000.0,
    },
    UnitDesc {
        id: UnitId::CudaBf16,
        gemm: GemmUnit::Bf16,
        shape: [16, 8, 16],
        subgroup: 32,
        a: ElemTy::BF16,
        b: ElemTy::BF16,
        acc: ElemTy::F32,
        needs: Caps::BF16_MMA,
        reduced: Reduced::MATRIX_ACCUMULATE,
        rate: 55_000.0,
    },
    // The cooperative-matrix unit (wgpu): the same operand and accumulation semantics as `CudaF16`; its rate is measured at 4096^3 on an
    // RTX 3090 (see the CHANGELOG of W3-R7), its edges are in the kernel.
    UnitDesc {
        id: UnitId::CoopF16,
        gemm: GemmUnit::CoopF16,
        shape: [16, 16, 16],
        subgroup: 32,
        a: ElemTy::F16,
        b: ElemTy::F16,
        acc: ElemTy::F32,
        needs: Caps::COOP_F16,
        reduced: Reduced::MATRIX_ACCUMULATE,
        rate: 20_000.0,
    },
];

/// The unit of a `MatmulTile::Gemm*`'s `exact` flag (the flag stays in the tile: its `Debug` text is a kernel label that decisions
/// and the plan cache carry).
pub(crate) const fn gemm_unit(exact: bool) -> GemmUnit {
    if exact { GemmUnit::Fma } else { GemmUnit::Tf32 }
}

impl UnitDesc {
    /// The row of `unit`.
    #[inline]
    pub(crate) fn of(unit: GemmUnit) -> &'static UnitDesc {
        match unit {
            GemmUnit::Tf32 => &UNITS[0],
            GemmUnit::F16 => &UNITS[2],
            GemmUnit::Bf16 => &UNITS[3],
            GemmUnit::CoopF16 => &UNITS[4],
            _ => &UNITS[1],
        }
    }
}

/// The label of a unit variant in the decision log: `<unit> <rows>x<cols> <edge>`.
pub(crate) fn variant_label(spec: &GemmSpec) -> String {
    format!("{:?} {}x{} {:?}", spec.unit, spec.rows, spec.cols, spec.lanes)
}

impl Context {
    /// Whether the unit variant of `spec` may be selected: its reference verification (W3-R7 §5, `luma_core::hal::verify`) passed. The
    /// first call of a variant on a device with no stored verdict runs the verification and waits for it (about a kernel launch, and
    /// stored on disk afterwards); the same call never returns another answer later. A failure disables the variant and is recorded as
    /// `Rejected("verification failed")`; the first use of a verified variant records the wait. The baseline units (`Fma`, `Tf32`) are
    /// not subject to verification: callers do not ask for them.
    pub(crate) fn unit_verified(&self, spec: &GemmSpec, op: &'static str, location: &'static Location<'static>) -> bool {
        let (verdict, waited) = self.unit_verifier().verdict(self.hal(), spec);
        if waited.is_some() {
            let label = variant_label(spec);
            let decision = if verdict.ok {
                Decision::new(op, location, DecisionSite::Unit, format!("{label} verified"))
            } else {
                Decision::new(op, location, DecisionSite::Unit, label.clone()).candidate(label, Verdict::Rejected("verification failed"))
            };
            self.jit().note_decision(decision);
        }
        verdict.ok
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn each_gemm_unit_has_its_row() {
        for unit in [GemmUnit::Fma, GemmUnit::Tf32, GemmUnit::F16, GemmUnit::Bf16, GemmUnit::CoopF16] {
            assert_eq!(UnitDesc::of(unit).gemm, unit);
        }
        assert_eq!(UNITS.iter().map(|u| u.id).collect::<Vec<_>>(), [UnitId::CudaTf32, UnitId::Fma, UnitId::CudaF16, UnitId::CudaBf16, UnitId::CoopF16]);
    }

    /// The reference verification through a context: a verified variant is accepted on its first call (with a decision recording the
    /// wait) and silently afterwards; a variant whose kernel is broken (a test hook alters its output) is disabled, with a
    /// `Rejected("verification failed")` decision, and stays so.
    #[test]
    fn unit_variants_are_verified_once_and_a_broken_one_is_disabled() {
        let Ok(ctx) = Context::new() else { return };
        if !ctx.caps().contains(Caps::TF32) {
            eprintln!("SKIPPED: no tensor-core units on this device");
            return;
        }
        let tiles = ctx.hal().gemm_tiles(GemmUnit::Tf32).to_vec();
        let (rows, cols) = tiles[tiles.len() - 1];
        let good = GemmSpec::new(rows, cols, false, true, GemmUnit::Tf32);
        let broken = GemmSpec::new(rows, cols, true, false, GemmUnit::Tf32);
        ctx.record_decisions(true);
        let here = Location::caller();
        assert!(ctx.unit_verified(&good, "matmul", here));
        let first = ctx.take_decisions();
        let unit: Vec<_> = first.iter().filter(|d| d.site == DecisionSite::Unit).collect();
        assert_eq!(unit.len(), 1, "the wait is recorded: {first:?}");
        assert!(unit[0].chosen.ends_with("verified"), "{:?}", unit[0]);
        assert!(ctx.unit_verified(&good, "matmul", here));
        assert!(ctx.take_decisions().iter().all(|d| d.site != DecisionSite::Unit), "a known verdict records nothing");

        // The flipped fragment index of a broken kernel: two columns of every row swapped.
        ctx.unit_verifier().set_tamper(Some(std::sync::Arc::new(|spec: &GemmSpec, out: &mut [f32]| {
            if spec.lhs_transposed {
                let n = out.len();
                out.swap(n / 3, n / 3 + 1);
                out[n / 2] += 1.0e9;
            }
        })));
        assert!(!ctx.unit_verified(&broken, "matmul", here));
        let log = ctx.take_decisions();
        let rejected = log.iter().find(|d| d.site == DecisionSite::Unit).expect("the failure is recorded");
        assert_eq!(rejected.candidates.len(), 1);
        assert_eq!(rejected.candidates[0].1, Verdict::Rejected("verification failed"));
        ctx.unit_verifier().set_tamper(None);
        assert!(!ctx.unit_verified(&broken, "matmul", here), "a Bad verdict stays for the life of the context");
        assert!(ctx.unit_verified(&good, "matmul", here));
    }
}
