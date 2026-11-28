//! The matrix-unit table (W3-R7): each unit is a fixed-size comprehension `C[m,n] += A[m,k]·B[k,n]` (its extents, operand and
//! accumulator types, the lanes that cooperate) plus the numerics class that permits it and the rates the Contraction strategies cost
//! it with. The strategies `GemmF32` / `GemmTf32` (and their ragged and padded siblings) read their rates and caps from these rows
//! instead of literals; the tiles a unit runs on are the backend's (`Backend::gemm_tiles(unit)`).

use crate::caps::Caps;
use crate::jit::ElemTy;
use crate::numerics::Reduced;
use luma_core::GemmUnit;

/// The identity of a row of [`UNITS`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum UnitId {
    /// TF32 tensor cores (CUDA `mma.m16n8k8`).
    CudaTf32,
    /// The exact `f32` kernels on the ordinary cores: the reference unit.
    Fma,
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
    /// Largest product (multiply-adds) that runs on a ragged or padded kernel of the unit. The padded `f32` kernel's copy launch and
    /// its masked copies lose what the 16-byte lanes gain on a big square (3001^3 1.05x, 4097^3 1.00x of the shader kernels, 1001^3
    /// 1.20x on an RTX 3090), so those keep the shader kernels until the kernel is faster (follow-up). TF32 has no such cap.
    pub(crate) ragged_max_fmas: u64,
}

/// The units, in preference order.
pub(crate) static UNITS: [UnitDesc; 2] = [
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
        ragged_max_fmas: u64::MAX,
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
        ragged_max_fmas: 4_000_000_000,
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
            _ => &UNITS[1],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn each_gemm_unit_has_its_row() {
        for unit in [GemmUnit::Fma, GemmUnit::Tf32] {
            assert_eq!(UnitDesc::of(unit).gemm, unit);
        }
        assert_eq!(UNITS.iter().map(|u| u.id).collect::<Vec<_>>(), [UnitId::CudaTf32, UnitId::Fma]);
    }
}
