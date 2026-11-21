//! Native matrix products: the tiled GEMM of `luma`'s `Builtin::Matmul`, run by `matrixmultiply`
//! (blocked, packed, AVX-512/AVX2+FMA/NEON microkernels picked at run time) instead of compiled
//! shader code.
//!
//! The launch is the shader's (see [`MatmulSpec`] for the bindings, the uniform and the numbering
//! of the tiles): a workgroup multiplies one output tile over its slice of K on one core, so the
//! pool spreads the tiles of a product over every core and recorded steps, graphs and profiles
//! treat it like any kernel. Every element of `C` is a sum over K in the order the blocking gives,
//! not the shader's; the results agree up to the rounding of the sums.

use luma_core::{BackendError, MatmulSpec, NativeKernel};

use crate::abi::{WorkgroupCtx, WorkgroupFn};

/// A matrix product compiled to a function.
#[derive(Debug)]
pub struct NativeMatmul {
    pub(crate) spec: MatmulSpec,
    pub(crate) func: WorkgroupFn,
    /// What a workgroup costs, learned by the pool (see [`crate::exec::CostHint`]).
    pub(crate) cost: crate::exec::CostHint,
}

impl NativeKernel for NativeMatmul {}

/// Bindings of a launch: `c`, `a`, `b` and the uniform.
const BINDINGS: usize = 4;

impl NativeMatmul {
    /// The product of `spec`, `None` if its tile is empty.
    pub fn new(spec: &MatmulSpec) -> Option<Self> {
        (spec.edge > 0).then(|| NativeMatmul { spec: *spec, func: group, cost: crate::exec::CostHint::default() })
    }

    /// Checks that the buffers of a launch hold every element the product touches (it trusts
    /// them): `bytes` are the sizes of the bindings, `words` the uniform `{m, n, k, kc}` and
    /// `workgroups` the grid.
    pub(crate) fn validate(&self, bytes: &[u64], words: [u32; 4], workgroups: [u32; 3]) -> Result<(), BackendError> {
        let bad = |what: String| Err(BackendError::Unsupported(format!("native matrix product: {what}")));
        if bytes.len() != BINDINGS {
            return bad(format!("{} bindings, expected {BINDINGS}", bytes.len()));
        }
        let [m, n, k, kc] = words.map(u64::from);
        if m == 0 || n == 0 || workgroups.contains(&0) {
            return Ok(());
        }
        let slices = u64::from(workgroups[2]);
        if k > 0 && (kc == 0 || k.div_ceil(kc) > slices) {
            return bad(format!("{slices} slices of {kc} do not cover k = {k}"));
        }
        let edge = u64::from(self.spec.edge);
        let tiles = n.div_ceil(edge) * m.div_ceil(edge);
        if u64::from(workgroups[0]) * u64::from(workgroups[1]) < tiles {
            return bad(format!("{} workgroups for {tiles} tiles", u64::from(workgroups[0]) * u64::from(workgroups[1])));
        }
        let need = |at: usize, floats: u64, name: &str| -> Result<(), BackendError> {
            if bytes[at] < floats * 4 {
                return bad(format!("{name} (binding {at}) holds {} bytes, {} needed", bytes[at], floats * 4));
            }
            Ok(())
        };
        need(0, slices * m * n, "c")?;
        need(1, m * k, "a")?;
        need(2, k * n, "b")
    }
}

/// One workgroup: the output tile `x + y * num_workgroups.x` over the slice of K of layer `z`.
unsafe extern "C" fn group(ctx: *const WorkgroupCtx) {
    // SAFETY: the dispatcher passes a valid context whose `native` is the kernel it dispatched
    // and whose bindings the launch validated ([`NativeMatmul::validate`]).
    unsafe {
        let ctx = &*ctx;
        let kernel = &*ctx.native.cast::<NativeMatmul>();
        let [c, a, b, uniform] = [0, 1, 2, 3].map(|at| *ctx.bindings.add(at));
        let [m, n, k, kc] = uniform.cast::<[u32; 4]>().read().map(|word| word as usize);
        let edge = kernel.spec.edge as usize;
        let [x, y, z, _] = ctx.workgroup_id;
        let tile = y as usize * ctx.num_workgroups[0] as usize + x as usize;
        let columns = n.div_ceil(edge);
        if m == 0 || n == 0 || tile >= columns * m.div_ceil(edge) {
            return;
        }
        let (row, col) = (tile / columns * edge, tile % columns * edge);
        let (rows, cols) = (edge.min(m - row), edge.min(n - col));
        let begin = z as usize * kc;
        let depth = if begin < k { kc.min(k - begin) } else { 0 };
        let c = c.cast::<f32>().add(z as usize * m * n + row * n + col);
        if depth == 0 {
            for r in 0..rows {
                std::ptr::write_bytes(c.add(r * n), 0, cols);
            }
            return;
        }
        // Element `(r, q)` of a row-major matrix of `ld` columns is at `r * ld + q`, of its
        // transpose (what the buffer holds if transposed) at `q * ld + r`.
        let (lhs_strides, rhs_strides) = (
            if kernel.spec.lhs_transposed { (1, m as isize) } else { (k as isize, 1) },
            if kernel.spec.rhs_transposed { (1, k as isize) } else { (n as isize, 1) },
        );
        let a = a.cast::<f32>().offset(row as isize * lhs_strides.0 + begin as isize * lhs_strides.1);
        let b = b.cast::<f32>().offset(begin as isize * rhs_strides.0 + col as isize * rhs_strides.1);
        matrixmultiply::sgemm(rows, depth, cols, 1.0, a, lhs_strides.0, lhs_strides.1, b, rhs_strides.0, rhs_strides.1, 0.0, c, n as isize, 1);
    }
}
