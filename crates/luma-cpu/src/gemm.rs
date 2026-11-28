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
    /// The epilogue applied to each tile after its sum (W3-U4 B3).
    pub(crate) epilogue: Option<TileEpilogue>,
}

/// The epilogue of a product, compiled over one tile row (see `Backend::compile_matmul_epilogue`).
#[derive(Debug)]
pub(crate) struct TileEpilogue {
    kernel: crate::codegen::CompiledCpuKernel,
    /// The read-only operands (bindings `3..3 + inputs` of the launch).
    inputs: usize,
    /// The scalars (words `8..8 + scalars` of the launch's uniform).
    scalars: usize,
}

/// Most operands and scalars of a tile epilogue (its bindings and uniform words are on the stack).
const MAX_EPILOGUE_INPUTS: usize = 12;
const MAX_EPILOGUE_SCALARS: usize = 11;
/// Words of the uniform of the tile kernel: `rows, cols, row, col, n`, three zero words, the scalars.
const TILE_WORDS: usize = 8 + MAX_EPILOGUE_SCALARS;

impl NativeKernel for NativeMatmul {}

/// Bindings of a launch: `c`, `a`, `b` and the uniform.
const BINDINGS: usize = 4;
/// Words of the uniform of a product: `m, n, k, kc, lda, ldb, a_off, b_off`.
pub(crate) const UNIFORM_WORDS: usize = 8;

impl NativeMatmul {
    /// The product of `spec`, `None` if its tile is empty.
    pub fn new(spec: &MatmulSpec) -> Option<Self> {
        (spec.edge > 0).then(|| NativeMatmul { spec: *spec, func: if spec.half { group_half } else { group }, cost: crate::exec::CostHint::default(), epilogue: None })
    }

    /// The product of `spec` applying `tile` (compiled, bindings `e0..e(E-1)`, `c`, `p`) to each output tile; `None` if the product is
    /// not one that has an epilogue (an empty tile, a half-work product) or the epilogue has more operands than a tile takes.
    pub(crate) fn with_epilogue(spec: &MatmulSpec, tile: crate::codegen::CompiledCpuKernel, bindings: usize, scalars: usize) -> Option<Self> {
        let inputs = bindings.checked_sub(2)?;
        if spec.half || spec.edge > tile.plan.workgroup_size[0] || tile.plan.workgroup_size[1..] != [1, 1] || inputs > MAX_EPILOGUE_INPUTS || scalars > MAX_EPILOGUE_SCALARS || tile.plan.shared_bytes != 0 || tile.plan.scratch_bytes != 0 {
            return None;
        }
        let mut product = Self::new(spec)?;
        product.epilogue = Some(TileEpilogue { kernel: tile, inputs, scalars });
        Some(product)
    }

    /// Checks that the buffers of a launch hold every element the product touches (it trusts
    /// them): `bytes` are the sizes of the bindings, `words` the uniform `{m, n, k, kc, lda, ldb,
    /// a_off, b_off}` and `workgroups` the grid.
    pub(crate) fn validate(&self, bytes: &[u64], words: [u32; UNIFORM_WORDS], workgroups: [u32; 3]) -> Result<(), BackendError> {
        let bad = |what: String| Err(BackendError::Unsupported(format!("native matrix product: {what}")));
        let bindings = BINDINGS + self.epilogue.as_ref().map_or(0, |epilogue| epilogue.inputs);
        if bytes.len() != bindings {
            return bad(format!("{} bindings, expected {bindings}", bytes.len()));
        }
        if let Some(epilogue) = &self.epilogue
            && bytes[bindings - 1] < 4 * (UNIFORM_WORDS + epilogue.scalars) as u64
        {
            return bad(format!("a uniform of {} bytes for {} scalars", bytes[bindings - 1], epilogue.scalars));
        }
        let [m, n, k, kc, lda, ldb, a_off, b_off] = words.map(u64::from);
        if m == 0 || n == 0 || workgroups.contains(&0) {
            return Ok(());
        }
        let slices = u64::from(workgroups[2]);
        if k > 0 && (kc == 0 || k.div_ceil(kc) > slices) {
            return bad(format!("{slices} slices of {kc} do not cover k = {k}"));
        }
        let edge = u64::from(self.spec.edge);
        let mut tiles = n.div_ceil(edge) * m.div_ceil(edge);
        if self.spec.half {
            if m != n || slices != 1 {
                return bad(format!("a half-work product needs a square output and one slice, got {m} x {n} in {slices} slices"));
            }
            let side = m.div_ceil(edge);
            tiles = side * (side + 1) / 2;
        }
        if u64::from(workgroups[0]) * u64::from(workgroups[1]) < tiles {
            return bad(format!("{} workgroups for {tiles} tiles", u64::from(workgroups[0]) * u64::from(workgroups[1])));
        }
        let need = |at: usize, floats: u64, name: &str| -> Result<(), BackendError> {
            if bytes[at] < floats * 4 {
                return bad(format!("{name} (binding {at}) holds {} bytes, {} needed", bytes[at], floats * 4));
            }
            Ok(())
        };
        // The elements a matrix of `rows x cols` stored with `ld` columns from `off` touches: up to the last element of its last row.
        let extent = |off: u64, rows: u64, cols: u64, ld: u64| if rows == 0 || cols == 0 { 0 } else { off + (rows - 1) * ld + cols };
        need(0, slices * m * n, "c")?;
        if k > 0 {
            let (a_rows, a_cols) = if self.spec.lhs_transposed { (k, m) } else { (m, k) };
            let (b_rows, b_cols) = if self.spec.rhs_transposed { (n, k) } else { (k, n) };
            need(1, extent(a_off, a_rows, a_cols, lda), "a")?;
            need(2, extent(b_off, b_rows, b_cols, ldb), "b")?;
        }
        Ok(())
    }
}

impl TileEpilogue {
    /// Applies the epilogue to the tile (`row`, `col`, `rows`, `cols` of an output `n` wide) that `ctx`'s workgroup just summed, by
    /// running its compiled kernel as a workgroup of its own over the launch's buffers.
    ///
    /// # Safety
    /// `ctx` is the context of a launch of the product that owns `self` (validated: [`NativeMatmul::validate`]) and `uniform` its
    /// uniform.
    #[inline(never)]
    unsafe fn apply(&self, ctx: &WorkgroupCtx, (row, col, rows, cols, n): (usize, usize, usize, usize, usize), uniform: *const u8) {
        // SAFETY: the launch holds `3 + inputs` bindings and the uniform after them, `binding_bytes` their sizes, and the uniform holds
        // `8 + scalars` words.
        unsafe {
            let mut words = [0u32; TILE_WORDS];
            words[..5].copy_from_slice(&[rows as u32, cols as u32, row as u32, col as u32, n as u32]);
            std::ptr::copy_nonoverlapping(uniform.cast::<u32>().add(UNIFORM_WORDS), words.as_mut_ptr().add(8), self.scalars);
            let mut bindings = [std::ptr::null_mut::<u8>(); MAX_EPILOGUE_INPUTS + 2];
            let mut bytes = [0u64; MAX_EPILOGUE_INPUTS + 2];
            for k in 0..self.inputs {
                bindings[k] = *ctx.bindings.add(BINDINGS - 1 + k);
                bytes[k] = *ctx.binding_bytes.add(BINDINGS - 1 + k);
            }
            bindings[self.inputs] = *ctx.bindings;
            bytes[self.inputs] = *ctx.binding_bytes;
            bindings[self.inputs + 1] = words.as_mut_ptr().cast();
            bytes[self.inputs + 1] = 4 * words.len() as u64;
            let mut lengths = [0u32; MAX_EPILOGUE_INPUTS + 2];
            for (slot, &(binding, stride)) in lengths.iter_mut().zip(&self.kernel.plan.array_lengths) {
                *slot = u32::try_from(bytes[binding as usize] / u64::from(stride.max(1))).unwrap_or(u32::MAX);
            }
            let mut tile = WorkgroupCtx {
                grid_size: [cols as u32 * rows as u32, 1, 1, 0],
                num_workgroups: [rows as u32, 1, 1, 0],
                workgroup_id: [0, 0, 0, 1],
                bindings: bindings.as_ptr(),
                binding_bytes: bytes.as_ptr(),
                lengths: lengths.as_ptr(),
                shared: std::ptr::null_mut(),
                scratch: std::ptr::null_mut(),
                math: ctx.math,
                native: std::ptr::null(),
            };
            // One workgroup per row of the tile.
            let func = self.kernel.exec.as_fn();
            for r in 0..rows as u32 {
                tile.workgroup_id[0] = r;
                func(&tile);
            }
        }
    }
}

/// One workgroup: the output tile `x + y * num_workgroups.x` over the slice of K of layer `z`.
unsafe extern "C" fn group(ctx: *const WorkgroupCtx) {
    // SAFETY: as `workgroup`.
    unsafe { workgroup::<false>(ctx) }
}

/// One workgroup of a half-work product ([`MatmulSpec::half`]): [`group`] with the lower triangle of tiles as its grid, mirrored.
unsafe extern "C" fn group_half(ctx: *const WorkgroupCtx) {
    // SAFETY: as `workgroup`.
    unsafe { workgroup::<true>(ctx) }
}

/// The body of [`group`] and [`group_half`]: `HALF` is a constant, so the ordinary product carries nothing of the half-work one.
#[inline(always)]
unsafe fn workgroup<const HALF: bool>(ctx: *const WorkgroupCtx) {
    // SAFETY: the dispatcher passes a valid context whose `native` is the kernel it dispatched
    // and whose bindings the launch validated ([`NativeMatmul::validate`]).
    unsafe {
        let ctx = &*ctx;
        let kernel = &*ctx.native.cast::<NativeMatmul>();
        // The epilogue's operands sit between `b` and the uniform.
        let uniform_at = BINDINGS - 1 + kernel.epilogue.as_ref().map_or(0, |epilogue| epilogue.inputs);
        let [c, a, b, uniform] = [0, 1, 2, uniform_at].map(|at| *ctx.bindings.add(at));
        let [m, n, k, kc, lda, ldb, a_off, b_off] = uniform.cast::<[u32; UNIFORM_WORDS]>().read().map(|word| word as usize);
        let edge = kernel.spec.edge as usize;
        let [x, y, z, _] = ctx.workgroup_id;
        let tile = y as usize * ctx.num_workgroups[0] as usize + x as usize;
        let columns = n.div_ceil(edge);
        if m == 0 || n == 0 || tile >= if HALF { columns * (columns + 1) / 2 } else { columns * m.div_ceil(edge) } {
            return;
        }
        let (row, col) = if HALF {
            let (ty, tx) = luma_core::triangle_tile(tile as u32);
            (ty as usize * edge, tx as usize * edge)
        } else {
            (tile / columns * edge, tile % columns * edge)
        };
        let (rows, cols) = (edge.min(m - row), edge.min(n - col));
        let begin = z as usize * kc;
        let depth = if begin < k { kc.min(k - begin) } else { 0 };
        let c = c.cast::<f32>().add(z as usize * m * n + row * n + col);
        if depth == 0 {
            for r in 0..rows {
                std::ptr::write_bytes(c.add(r * n), 0, cols);
            }
            if let Some(epilogue) = &kernel.epilogue {
                epilogue.apply(ctx, (row, col, rows, cols, n), uniform);
            }
            return;
        }
        // Element `(r, q)` of a row-major matrix of `ld` columns is at `r * ld + q`, of its
        // transpose (what the buffer holds if transposed) at `q * ld + r`; the matrix starts at
        // element `off` of its buffer.
        let (lhs_strides, rhs_strides) = (
            if kernel.spec.lhs_transposed { (1, lda as isize) } else { (lda as isize, 1) },
            if kernel.spec.rhs_transposed { (1, ldb as isize) } else { (ldb as isize, 1) },
        );
        let a = a.cast::<f32>().add(a_off).offset(row as isize * lhs_strides.0 + begin as isize * lhs_strides.1);
        let b = b.cast::<f32>().add(b_off).offset(begin as isize * rhs_strides.0 + col as isize * rhs_strides.1);
        matrixmultiply::sgemm(rows, depth, cols, 1.0, a, lhs_strides.0, lhs_strides.1, b, rhs_strides.0, rhs_strides.1, 0.0, c, n as isize, 1);
        if let Some(epilogue) = &kernel.epilogue {
            epilogue.apply(ctx, (row, col, rows, cols, n), uniform);
        }
        if HALF {
            // The tile is the full kernel's, so the lower triangle of the output is the full product's. Its transpose goes to
            // the tile above the diagonal (written by no other workgroup); on a diagonal tile the upper half is overwritten by
            // the mirror of the lower, so the whole output is bit-symmetric.
            if row == col {
                for r in 0..rows {
                    for q in r + 1..cols {
                        *c.add(r * n + q) = *c.add(q * n + r);
                    }
                }
            } else {
                let mirror = c.offset((col * n + row) as isize - (row * n + col) as isize);
                for r in 0..rows {
                    for q in 0..cols {
                        *mirror.add(q * n + r) = *c.add(r * n + q);
                    }
                }
            }
        }
    }
}

/// A batched matrix product compiled to a function (see `Backend::compile_batched_matmul`).
#[derive(Debug)]
pub struct NativeBatchedMatmul {
    pub(crate) func: WorkgroupFn,
    /// What a workgroup costs, learned by the pool (see [`crate::exec::CostHint`]).
    pub(crate) cost: crate::exec::CostHint,
}

impl NativeKernel for NativeBatchedMatmul {}

/// The words of the uniform of a batched launch: `m, n, k, matrices, a_stride, b_stride`, two
/// more being padding.
const BATCHED_WORDS: usize = 8;

impl Default for NativeBatchedMatmul {
    fn default() -> Self {
        NativeBatchedMatmul { func: batched_group, cost: crate::exec::CostHint::default() }
    }
}

impl NativeBatchedMatmul {
    /// Checks that the buffers of a launch hold every element the products touch (it trusts them):
    /// `bytes` are the sizes of the bindings, `words` the uniform and `workgroups` the grid.
    pub(crate) fn validate(&self, bytes: &[u64], words: [u32; BATCHED_WORDS], workgroups: [u32; 3]) -> Result<(), BackendError> {
        let bad = |what: String| Err(BackendError::Unsupported(format!("native batched matrix product: {what}")));
        if bytes.len() != BINDINGS {
            return bad(format!("{} bindings, expected {BINDINGS}", bytes.len()));
        }
        let [m, n, k, matrices, a_stride, b_stride, ..] = words.map(u64::from);
        if m == 0 || n == 0 || matrices == 0 || workgroups.contains(&0) {
            return Ok(());
        }
        if u64::from(workgroups[0]) * u64::from(workgroups[1]) * u64::from(workgroups[2]) < matrices {
            return bad(format!("{} workgroups for {matrices} matrices", u64::from(workgroups[0]) * u64::from(workgroups[1])));
        }
        let need = |at: usize, floats: u64, name: &str| -> Result<(), BackendError> {
            if bytes[at] < floats * 4 {
                return bad(format!("{name} (binding {at}) holds {} bytes, {} needed", bytes[at], floats * 4));
            }
            Ok(())
        };
        need(0, matrices * m * n, "c")?;
        need(1, (matrices - 1) * a_stride + m * k, "a")?;
        need(2, (matrices - 1) * b_stride + k * n, "b")
    }
}

/// One workgroup: the matrix `x + y * num_workgroups.x`, multiplied as a whole.
unsafe extern "C" fn batched_group(ctx: *const WorkgroupCtx) {
    // SAFETY: the dispatcher passes a valid context whose bindings the launch validated
    // ([`NativeBatchedMatmul::validate`]).
    unsafe {
        let ctx = &*ctx;
        let [c, a, b, uniform] = [0, 1, 2, 3].map(|at| *ctx.bindings.add(at));
        let [m, n, k, matrices, a_stride, b_stride, ..] = uniform.cast::<[u32; BATCHED_WORDS]>().read().map(|word| word as usize);
        let [x, y, ..] = ctx.workgroup_id;
        let index = y as usize * ctx.num_workgroups[0] as usize + x as usize;
        if index >= matrices {
            return;
        }
        let c = c.cast::<f32>().add(index * m * n);
        if k == 0 {
            std::ptr::write_bytes(c, 0, m * n);
            return;
        }
        let a = a.cast::<f32>().add(index * a_stride);
        let b = b.cast::<f32>().add(index * b_stride);
        matrixmultiply::sgemm(m, k, n, 1.0, a, k as isize, 1, b, n as isize, 1, 0.0, c, n as isize, 1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn product(lhs_transposed: bool, rhs_transposed: bool) -> NativeMatmul {
        NativeMatmul::new(&MatmulSpec { edge: 32, lhs_transposed, rhs_transposed, half: false }).unwrap()
    }

    /// The sizes of the bindings `c, a, b` in floats, then the uniform's.
    fn launch(product: &NativeMatmul, floats: [u64; 3], words: [u32; UNIFORM_WORDS]) -> Result<(), BackendError> {
        product.validate(&[floats[0] * 4, floats[1] * 4, floats[2] * 4, 32], words, [1, 1, 1])
    }

    #[test]
    fn ld_and_offset_bounds_are_checked_on_the_cpu() {
        // A 4 x 5 left window (rows of `k = 5`) at offset 3 of a buffer with rows of 8: its last element is `3 + 3 * 8 + 5 = 32`, one past.
        let plain = product(false, false);
        let words = |lda, ldb, a_off, b_off| [4, 6, 5, 5, lda, ldb, a_off, b_off];
        // The right matrix `5 x 6` with rows of 6, whole.
        assert!(launch(&plain, [24, 32, 30], words(8, 6, 3, 0)).is_ok());
        assert!(launch(&plain, [24, 31, 30], words(8, 6, 3, 0)).is_err(), "one float short of the last row of `a`");
        assert!(launch(&plain, [24, 32, 30], words(8, 6, 3, 1)).is_err(), "`b` shifted past its end");
        assert!(launch(&plain, [24, 32, 35], words(8, 7, 3, 1)).is_ok());
        // Transposed operands are `k` rows of `lda` (`m` columns) and `n` rows of `ldb` (`k` columns).
        let transposed = product(true, true);
        assert!(launch(&transposed, [24, 3 + 4 * 9 + 4, 2 + 5 * 7 + 5], words(9, 7, 3, 2)).is_ok());
        assert!(launch(&transposed, [24, 3 + 4 * 9 + 3, 2 + 5 * 7 + 5], words(9, 7, 3, 2)).is_err());
        // Offsets and leading dimensions near the top of `u32` do not wrap.
        assert!(launch(&plain, [24, 1 << 20, 1 << 20], words(u32::MAX, 6, u32::MAX, 0)).is_err());
        // An empty product touches nothing.
        assert!(launch(&plain, [0, 0, 0], [0, 6, 5, 5, 1 << 31, 1 << 31, 1 << 31, 1 << 31]).is_ok());
        // Bindings are `c, a, b` and the uniform.
        assert!(plain.validate(&[96, 128, 120], [4, 6, 5, 5, 8, 6, 3, 0], [1, 1, 1]).is_err());
    }
}
