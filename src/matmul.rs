//! Matrix product: [`Array::matmul`] and [`matmul()`].
//!
//! Luma's axis 0 varies fastest, so the shapes are those of NumPy's `a @ b` reversed: `lhs` is
//! `[K, M]`, `rhs` is `[N, K]` and the result `[N, M]`, `C[n + N m] = Σ_k A[k + K m] B[n + N k]`,
//! the row-major `C = A B` of `M x K` and `K x N` matrices.
//!
//! One kernel per product, in four sizes of output tile: 128x128 (16x16 invocations of 8x8
//! outputs each), 128x32 and 32x128 when only `N` or only `M` is below 128, or 32x32 (2x2 each)
//! when both are. All stage 16-wide slices of
//! K of the two operands in workgroup memory, see [`body`]. A product with few output tiles and a
//! long K would leave most of the GPU idle: it is split over K, the slices written to a scratch
//! buffer by the third grid dimension and summed by a second launch (`split_of`).
//!
//! Products with a small output and a long K (a Gram matrix, one-hot labels times features, a
//! gradient `Xᵀ r`, a matrix times a few vectors) have mostly padding in those tiles. They run as
//! skinny kernels instead (`skinny_plan`): each invocation owns up to 10x8 outputs in registers
//! and loads straight from global memory, without staging or barriers in the loop, and the
//! invocations the small output does not need take other `k`; a second launch sums the slices of
//! K as above.
//!
//! On a backend with its own gemm kernels (CUDA, see the strategies of `jit::structured::contraction`) the shapes they take (`M`, `K` and `N`
//! multiples of 4 and at least 32, or a few columns: the aligned kernels; any shape above 32 on a side and a million multiply-adds:
//! the ragged and padded ones, which cut the tiles at the edges) are multiplied by them,
//! reading the operands as stored (a transposed view is read in place) and evaluating an operand
//! that is an expression first, once (`gemm_operand`). Otherwise, with the shader kernels, an operand that is a pending expression is not evaluated first: its program is flattened
//! into the kernel (`jit::program::Tail::Matmul`), which calls it as `luma_lhs(i)` /
//! `luma_rhs(i)` where a buffer kernel indexes `a` / `b`. A transposed operand is such an
//! expression (a view), and the kernel then divides its slices so that neighbouring invocations
//! load neighbours in memory ([`MatmulOperand::Transposed`]). Programs that read sub-programs
//! through circular shifts, or arrays split over several buffers, are evaluated instead. The
//! product itself launches when it is called, and returns an evaluated array, unless its call site is a lazy one (`LUMA_LAZY`, see
//! `jit::structured::sites`): it is then a value that launches when it is used (`jit::structured::lazy`).

use std::panic::Location;

use crate::array::Array;
use luma_core::GemmLanes;
use crate::jit::comprehension::lower::contract::ContractForm;
use crate::jit::comprehension::OpTag;
use crate::jit::structured::contraction::{ContractionAttrs, Tf32Request};
use crate::jit::structured::lazy;

/// Output tile of the kernel.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum MatmulTile {
    /// 128x128, each of the 16x16 invocations computes 8x8.
    T128,
    /// 128 rows by 32 columns, each of the 16x16 invocations computes 8x2.
    T128x32,
    /// 32 rows by 128 columns, each of the 16x16 invocations computes 2x8.
    T32x128,
    /// 32x32, each of the 16x16 invocations computes 2x2.
    T32,
    /// A square tile of the given `edge` that a backend with native products multiplies as native
    /// code (see [`native_plan`]), the buffer of an operand holding the matrix transposed if
    /// `lhs_t` / `rhs_t`; its shader is that of `T32` (the bindings), never run.
    Native { edge: u32, lhs_t: bool, rhs_t: bool },
    /// A tile of `rows` x `cols` a backend multiplies in its own kernels, on tensor cores in TF32
    /// or with `exact` in `f32` on the ordinary cores (see [`gemm_plan`]), the buffer of an operand holding the matrix transposed if `lhs_t` /
    /// `rhs_t`; its shader is that of `T32` (the bindings), never run.
    Gemm { rows: u32, cols: u32, lhs_t: bool, rhs_t: bool, exact: bool },
    /// [`MatmulTile::Gemm`] for a product of any extents, leading dimensions and offsets, copying the operands with `lanes` (never
    /// `Vec4`: those are `Gemm`); see [`GemmLanes`].
    GemmRagged { rows: u32, cols: u32, lhs_t: bool, rhs_t: bool, exact: bool, lanes: GemmLanes },
    /// A small output (see [`skinny_tile`]): each invocation of a workgroup of `width` computes
    /// `rows` x `cols` outputs straight from global memory, and the `groups` invocations that own
    /// the same outputs take the `k` in turn. With `along_k` neighbouring invocations take
    /// neighbouring `k` (the left operand is stored along `K`), else neighbouring output tiles.
    Skinny { rows: u8, cols: u8, groups: u8, width: u16, along_k: bool },
}

impl MatmulTile {
    /// The rows and columns of the tile.
    pub(crate) fn edges(self) -> (u32, u32) {
        match self {
            MatmulTile::T128 => (128, 128),
            MatmulTile::T128x32 => (128, 32),
            MatmulTile::T32x128 => (32, 128),
            MatmulTile::T32 => (32, 32),
            MatmulTile::Native { edge, .. } => (edge, edge),
            MatmulTile::Gemm { rows, cols, .. } | MatmulTile::GemmRagged { rows, cols, .. } => (rows, cols),
            MatmulTile::Skinny { .. } => unreachable!("a skinny product has no output tiles of fixed edges"),
        }
    }
}

/// K elements of each operand a workgroup stages at a time.
pub(crate) const SLICE: u32 = 16;


impl Array<f32> {
    /// The matrix product with `rhs`. `self` has shape `[K, M]`, `rhs` `[N, K]` and the result
    /// `[N, M]`: element `(n, m)` is `Σ_k self[k, m] * rhs[n, k]`, which is the row-major
    /// product of an `M x K` and a `K x N` matrix (axis 0, the fastest, is the column index).
    /// One-dimensional operands count as having extent 1 on axis 1.
    ///
    /// **Precision.** On the CUDA backend with tensor cores (compute capability 8.0 and up) this
    /// is [`Array::matmul_tf32`] unless the context turned that off
    /// ([`Numerics::strict`](crate::Numerics::strict) in [`ContextOptions::numerics`](crate::ContextOptions::numerics), `LUMA_NUMERICS=strict`):
    /// the operands are rounded to TF32 (a 10-bit mantissa) and the sums are `f32`, an error of
    /// about 2e-4 of the RMS of the product (at most 1.5e-3 of the sum of the absolute terms of
    /// an entry). Use [`Array::matmul_f32`] where full `f32` precision matters; every other
    /// backend, and every product the tensor cores do not take, is the `f32` product.
    ///
    /// Operands that are lazy expressions are fused into the kernel's loads instead of being
    /// evaluated first (the tensor-core product evaluates them first). The product is recorded when this is called and the result is an
    /// evaluated array: what consumes it runs as its own kernels.
    ///
    /// Products with few output tiles and a long `K` are split over `K` and summed in a second
    /// kernel; the sum order is fixed, so results are reproducible.
    ///
    /// # Errors
    ///
    /// The result is poisoned with [`LumaError::ShapeMismatch`](crate::LumaError::ShapeMismatch) if `self.shape()[0] !=
    /// rhs.shape()[1]` or an operand has more than two axes of extent above 1,
    /// [`LumaError::ContextMismatch`](crate::LumaError::ContextMismatch) if the operands live on different contexts and
    /// [`LumaError::Chunked`](crate::LumaError::Chunked) if an operand or the result needs more than one buffer.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use luma::Array;
    ///
    /// // [[1, 2], [3, 4]] @ [[5, 6], [7, 8]], rows listed first.
    /// let a = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], [2, 2])?;
    /// let b = Array::from_slice(&[5.0f32, 6.0, 7.0, 8.0], [2, 2])?;
    /// assert_eq!(a.matmul(&b).to_vec()?, [19.0, 22.0, 43.0, 50.0]);
    /// # Ok::<(), luma::LumaError>(())
    /// ```
    #[track_caller]
    pub fn matmul(&self, rhs: &Array<f32>) -> Array<f32> {
        matmul(self, rhs)
    }

    /// [`Array::matmul`] on the tensor cores of the device, whatever the context's
    /// [`numerics`](crate::ContextOptions::numerics) say: the operands are rounded to TF32
    /// (a 10-bit mantissa, the range of `f32`) and the products accumulated in `f32`, so the
    /// result differs from the `f32` product by up to 1.5e-3 of the sum of the absolute terms of
    /// an entry (about 2e-4 of the RMS of the product for operands of random sign), the
    /// precision of PyTorch's `allow_tf32`. Infinities and NaNs go through as in the `f32`
    /// product, and an operand within 3.4e-4 of the largest `f32` becomes an infinity.
    ///
    /// Only the CUDA backend on a device with tensor cores of compute capability 8.0 or higher
    /// runs it, and only for products whose `M`, `N` and `K` are multiples of 4 and at least 32
    /// (operands that are transposed views are read in place, other expressions evaluated
    /// first, as for any native product); every other product, and every other backend, computes
    /// the plain `f32` product. Like `matmul`, the result is reproducible (the sum order is
    /// fixed).
    ///
    /// # Errors
    ///
    /// As [`Array::matmul`].
    #[track_caller]
    pub fn matmul_tf32(&self, rhs: &Array<f32>) -> Array<f32> {
        matmul_tf32(self, rhs)
    }

    /// [`Array::matmul`] in full `f32` precision on every backend, whatever the context's
    /// [`numerics`](crate::ContextOptions::numerics) say: each entry is a sum of `f32`
    /// products accumulated with fused multiply-adds in a fixed order. Slower than the tensor
    /// cores of `matmul` where those apply (about 1.7x on large products on an RTX 3090).
    ///
    /// # Errors
    ///
    /// As [`Array::matmul`].
    #[track_caller]
    pub fn matmul_f32(&self, rhs: &Array<f32>) -> Array<f32> {
        matmul_f32(self, rhs)
    }

    /// `self` times the transpose of `rhs` ([`Array::matmul`] of `self` and `rhs.transpose()`, which
    /// is `A Bᵀ` of the row-major matrices): `self` has shape `[K, M]`, `rhs` `[K, N]` (the row-major
    /// `N x K` matrix) and the result `[N, M]`. `rhs` is read in place by the backend's own
    /// kernels (no transposed copy) when it is a stored array, and evaluated first, once, when it
    /// is an expression. Precision as [`Array::matmul`].
    ///
    /// # Errors
    ///
    /// As [`Array::matmul`] for the shapes `self` `[K, M]` and `rhs.transpose()` `[N, K]`.
    #[track_caller]
    pub fn matmul_nt(&self, rhs: &Array<f32>) -> Array<f32> {
        matmul_nt(self, rhs)
    }

    /// The transpose of `self` times `rhs` (`Aᵀ B` of the row-major matrices): `self` has shape
    /// `[M, K]` (the row-major `K x M` matrix), `rhs` `[N, K]` and the result `[N, M]`. Operands are
    /// read in place as for [`Array::matmul_nt`]; precision as [`Array::matmul`].
    ///
    /// # Errors
    ///
    /// As [`Array::matmul`] for the shapes `self.transpose()` `[K, M]` and `rhs` `[N, K]`.
    #[track_caller]
    pub fn matmul_tn(&self, rhs: &Array<f32>) -> Array<f32> {
        matmul_tn(self, rhs)
    }

    /// The transpose of `self` times the transpose of `rhs` (`Aᵀ Bᵀ` of the row-major matrices):
    /// `self` has shape `[M, K]`, `rhs` `[K, N]` and the result `[N, M]`. Operands are read in
    /// place as for [`Array::matmul_nt`]; precision as [`Array::matmul`].
    ///
    /// # Errors
    ///
    /// As [`Array::matmul`] for the shapes `self.transpose()` `[K, M]` and `rhs.transpose()` `[N, K]`.
    #[track_caller]
    pub fn matmul_tt(&self, rhs: &Array<f32>) -> Array<f32> {
        matmul_tt(self, rhs)
    }

    /// The dot product of the elements of `self` and `rhs`, which have the same shape, as a lazy
    /// one-element array (`(self * rhs).sum()`: one fused reduction that reads both once).
    ///
    /// # Errors
    ///
    /// The result is poisoned with [`LumaError::ShapeMismatch`](crate::LumaError::ShapeMismatch) if the shapes differ.
    #[track_caller]
    pub fn dot(&self, rhs: &Array<f32>) -> Array<f32> {
        dot(self, rhs)
    }
}

/// [`Array::matmul_nt`] as a function.
#[track_caller]
pub fn matmul_nt(lhs: &Array<f32>, rhs: &Array<f32>) -> Array<f32> {
    product(lhs, &rhs.transpose(), None)
}

/// [`Array::matmul_tn`] as a function.
#[track_caller]
pub fn matmul_tn(lhs: &Array<f32>, rhs: &Array<f32>) -> Array<f32> {
    product(&lhs.transpose(), rhs, None)
}

/// [`Array::matmul_tt`] as a function.
#[track_caller]
pub fn matmul_tt(lhs: &Array<f32>, rhs: &Array<f32>) -> Array<f32> {
    product(&lhs.transpose(), &rhs.transpose(), None)
}

/// [`Array::dot`] as a function.
#[track_caller]
pub fn dot(lhs: &Array<f32>, rhs: &Array<f32>) -> Array<f32> {
    (lhs * rhs).reduce_tagged(OpTag::Dot, "sum", Location::caller())
}

/// [`Array::matmul`] as a function: the product of `lhs` (`[K, M]`) and `rhs` (`[N, K]`) as an
/// array of shape `[N, M]`.
#[track_caller]
pub fn matmul(lhs: &Array<f32>, rhs: &Array<f32>) -> Array<f32> {
    product(lhs, rhs, None)
}

/// [`Array::matmul_tf32`] as a function.
#[track_caller]
pub fn matmul_tf32(lhs: &Array<f32>, rhs: &Array<f32>) -> Array<f32> {
    product(lhs, rhs, Some(true))
}

/// [`Array::matmul_f32`] as a function.
#[track_caller]
pub fn matmul_f32(lhs: &Array<f32>, rhs: &Array<f32>) -> Array<f32> {
    product(lhs, rhs, Some(false))
}

#[track_caller]
fn product(lhs: &Array<f32>, rhs: &Array<f32>, tf32: Option<bool>) -> Array<f32> {
    let location = Location::caller();
    let tf32 = match tf32 {
        None => Tf32Request::Default,
        Some(true) => Tf32Request::Tf32,
        Some(false) => Tf32Request::Exact,
    };
    let shape = [rhs.shape().dims()[0], lhs.shape().dims()[1], 1, 1];
    let shape = if shape.contains(&0) { lhs.shape() } else { shape.into() };
    let product = lazy::product(lhs, rhs, ContractionAttrs { tf32 }, shape, location);
    if lhs.has_props() || rhs.has_props() { product.with_props(lhs.props().product(rhs.props())) } else { product }
}

/// How a product is cut into workgroups.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct Plan {
    pub(crate) tile: MatmulTile,
    pub(crate) aligned: bool,
    /// Whether the kernel writes the slice of its workgroup at `z * M * N` (see [`body`]);
    /// `slices > 1` unless the kernel always does.
    pub(crate) split: bool,
    /// Slices of K, 1 if the product is not split.
    pub(crate) slices: u32,
    /// Elements of K per slice (K if not split).
    pub(crate) kc: u32,
    /// `[column tiles, row tiles, slices]`; a skinny kernel has `[slices, 1, 1]`.
    pub(crate) groups: [u32; 3],
}

/// The kernel of [`Builtin::Matmul`]: the operands are the buffers `a` and `b`.
pub(crate) fn builtin_source(tile: MatmulTile, aligned: bool, split: bool) -> String {
    format!(
        "struct Params {{ m: u32, n: u32, k: u32, kc: u32 }}
@group(1) @binding(0) var<storage, read_write> c: array<f32>;
@group(1) @binding(1) var<storage, read> a: array<f32>;
@group(1) @binding(2) var<storage, read> b: array<f32>;
@group(1) @binding(3) var<uniform> p: Params;
{}",
        body(tile, aligned, split, MatmulOperand::Buffer, MatmulOperand::Buffer, &ContractForm::DEFAULT)
    )
}

/// How the kernel reads one operand.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum MatmulOperand {
    /// From the buffer `a` / `b`, in the layout of the matrix.
    Buffer,
    /// Through `luma_lhs(i)` / `luma_rhs(i)`, the matrix being stored in its layout (the elements
    /// of a row of `lhs`, a column of `rhs` next to each other).
    Fused,
    /// Through the function, the matrix being stored transposed: the neighbours of `lhs` along
    /// `M`, of `rhs` along `K`, are next to each other (a transposed view, see
    /// `jit::launch::matmul`). Same results as `Fused`; the invocations of a workgroup divide the
    /// staged slice so that those that load neighbouring elements are neighbours.
    Transposed,
}

impl MatmulOperand {
    /// Whether the operand is a function of the element index.
    pub(crate) fn fused(self) -> bool {
        self != MatmulOperand::Buffer
    }
}

/// Which way neighbouring invocations of a workgroup step through a staged slice of an operand:
/// the slice has `rows` (of `M` or `N`) and 16 `k`, slot `e` of its 512 being `(rows, k)`.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Lanes {
    /// `k = e % 16`, row group `e / 16`: the next lane is the next `k`.
    AlongK,
    /// `k = e / 32`, row group `e % 32`: the next lane is the next row group.
    AlongRows,
}

impl Lanes {
    /// The `k` of the slice slot `e` is, and its row group (one row for `T32`, four for `T128`).
    fn split(self) -> (&'static str, &'static str) {
        match self {
            Lanes::AlongK => ("e % 16u", "e / 16u"),
            Lanes::AlongRows => ("e / 32u", "e % 32u"),
        }
    }

    /// Where slot `e` is in workgroup memory: k-major, `k * 32 + group`.
    fn stored(self, e: &str) -> String {
        match self {
            Lanes::AlongK => format!("k_major({e})"),
            Lanes::AlongRows => e.to_owned(),
        }
    }
}

/// The GEMM kernel after its declarations: expects the uniform `p` (`m`, `n`, `k`, `kc` first),
/// the output `c` (row-major `M x N`; with `split`, slice `z` is written at `z * M * N`) and the
/// operands as the buffers `a` (`M x K`) and `b` (`K x N`), or for a fused one as the function
/// `luma_lhs(i)` / `luma_rhs(i)` of the flat element index. The one source both
/// [`Builtin::Matmul`] and the fused kernels of `Tail::Matmul` come from.
///
/// `MatmulTile::T128` is `SGEMM` of `benches/common/sgemm.rs`: each workgroup computes a 128x128
/// tile of `C`, each of its 16x16 invocations 8x8 of it, from 16-wide slices of `A` and `B` staged
/// in workgroup memory as vec4s of four rows (columns), so that one 16-byte read feeds four
/// accumulators. An invocation owns two 4-wide groups of rows and of columns, 64 apart, so that
/// workgroup memory reads of neighbouring invocations are contiguous. The slice goes from global
/// to workgroup memory without detour through registers (the 64 accumulators leave none to
/// spare), and the loop over it is unrolled by two. `MatmulTile::T32` is the same scheme with
/// scalar staging and 2x2 outputs per invocation.
///
/// Without `aligned` (`M` and `N` multiples of the tile, `K` of [`SLICE`]) every load is
/// `select(0.0, load, in_bounds)` of an index clamped into the operand and every store is
/// guarded. With `split` the workgroup of grid layer `z` covers `K` from `z * kc` to
/// `min(K, (z + 1) * kc)`.
pub(crate) fn body(tile: MatmulTile, aligned: bool, split: bool, lhs: MatmulOperand, rhs: MatmulOperand, form: &ContractForm) -> String {
    let loads = Loads {
        lhs: Load { fused: lhs.fused(), lanes: if lhs == MatmulOperand::Transposed { Lanes::AlongRows } else { Lanes::AlongK } },
        rhs: Load { fused: rhs.fused(), lanes: if rhs == MatmulOperand::Transposed { Lanes::AlongK } else { Lanes::AlongRows } },
        batch: form.batch,
    };
    debug_assert!(form.is_default() || matches!(tile, MatmulTile::T32), "a generic contraction runs on the T32 tile");
    match tile {
        MatmulTile::T128 => t128(aligned, split, &loads),
        MatmulTile::T128x32 => t128x32(aligned, split, &loads),
        MatmulTile::T32x128 => t32x128(aligned, split, &loads),
        MatmulTile::T32 | MatmulTile::Native { .. } | MatmulTile::Gemm { .. } | MatmulTile::GemmRagged { .. } => t32(aligned, split, &loads, form),
        MatmulTile::Skinny { rows, cols, groups, width, along_k } => skinny(u32::from(rows), u32::from(cols), u32::from(groups), u32::from(width), along_k, &loads),
    }
}

/// One operand of the kernel: whether it is a function, and how its slices are divided.
#[derive(Clone, Copy)]
struct Load {
    fused: bool,
    lanes: Lanes,
}

struct Loads {
    lhs: Load,
    rhs: Load,
    /// The matrices of a batch: the loads add `zi * p.sa` (`p.sb`) to the flat index, `zi` being the grid layer of the workgroup.
    batch: bool,
}

impl Loads {
    /// The expression loading element `index` (of the flat matrix) of the left operand.
    fn lhs(&self, index: &str) -> String {
        let index = if self.batch { format!("zi * p.sa + {index}") } else { index.to_owned() };
        if self.lhs.fused { format!("luma_lhs({index})") } else { format!("a[{index}]") }
    }

    fn rhs(&self, index: &str) -> String {
        let index = if self.batch { format!("zi * p.sb + {index}") } else { index.to_owned() };
        if self.rhs.fused { format!("luma_rhs({index})") } else { format!("b[{index}]") }
    }

    /// The statements staging slots `t` and `t + 256` of both operands.
    fn stage(&self, ke: &str) -> String {
        let at = |e: &str| format!("at[{}] = load_a(row0, k0, {e}{ke});", self.lhs.lanes.stored(e));
        let bt = |e: &str| format!("bt[{}] = load_b(col0, k0, {e}{ke});", self.rhs.lanes.stored(e));
        format!("{}\n        {}\n        {}\n        {}", at("t"), at("t + 256u"), bt("t"), bt("t + 256u"))
    }
}

/// The statements that start a workgroup of a kernel: the range of K it covers (`k_begin`,
/// `k_end`, as WGSL expressions) and the offset of its output.
struct Range {
    declare: &'static str,
    k_begin: &'static str,
    k_end: &'static str,
    out: &'static str,
}

fn range(aligned: bool, split: bool) -> Range {
    match (split, aligned) {
        (true, _) => Range {
            declare: "    let kb = wid.z * p.kc;\n    let ke = min(p.k, kb + p.kc);\n    let zo = wid.z * p.m * p.n;\n",
            k_begin: "kb",
            k_end: "ke",
            out: "zo + ",
        },
        (false, false) => Range { declare: "    let ke = p.k;\n", k_begin: "0u", k_end: "ke", out: "" },
        (false, true) => Range { declare: "", k_begin: "0u", k_end: "p.k", out: "" },
    }
}

/// The load of the 4-row (column) group of slot `e` of a slice of `T128`: `fn load_a(row0, k0, e)`
/// for the left operand (`rows` is `p.m`, the matrix `M x K`), `load_b` for the right one
/// (`p.n`, `K x N`).
fn t128_load(left: bool, aligned: bool, lanes: Lanes, load: &dyn Fn(&str) -> String) -> String {
    let (k, group) = lanes.split();
    let (name, base, rows) = if left { ("load_a", "row0", "p.m") } else { ("load_b", "col0", "p.n") };
    let values: Vec<String> = if aligned {
        // Row `r + j` of A starts `p.k` further on; column `c + j` of B is the next element.
        let step = |j: u32| match (left, j) {
            (_, 0) => String::new(),
            (true, 1) => " + p.k".to_owned(),
            (true, j) => format!(" + {j}u * p.k"),
            (false, j) => format!(" + {j}u"),
        };
        (0..4).map(|j| load(&format!("i{}", step(j)))).collect()
    } else {
        (0..4)
            .map(|j| {
                let index = if left {
                    format!("min(r + {j}u, p.m - 1u) * p.k + kk")
                } else {
                    format!("row + min(r + {j}u, p.n - 1u)")
                };
                format!("select(0.0, {}, ok && r + {j}u < {rows})", load(&index))
            })
            .collect()
    };
    let body = if aligned {
        let i = if left { "r * p.k + k" } else { "k * p.n + r" };
        format!("    let r = {base} + ({group}) * 4u;\n    let k = k0 + {k};\n    let i = {i};\n    return vec4<f32>({});", values.join(", "))
    } else {
        let row = if left { "" } else { "    let row = min(k, ke - 1u) * p.n;\n" };
        format!(
            "    let r = {base} + ({group}) * 4u;\n    let k = k0 + {k};\n    let kk = min(k, ke - 1u);\n{row}    let ok = k < ke;\n    return vec4<f32>({});",
            values.join(", ")
        )
    };
    let args = if aligned { "" } else { ", ke: u32" };
    format!("fn {name}({base}: u32, k0: u32, e: u32{args}) -> vec4<f32> {{\n{body}\n}}")
}

fn t128(aligned: bool, split: bool, loads: &Loads) -> String {
    const LANES: [&str; 4] = ["x", "y", "z", "w"];
    let load_a = t128_load(true, aligned, loads.lhs.lanes, &|index| loads.lhs(index));
    let load_b = t128_load(false, aligned, loads.rhs.lanes, &|index| loads.rhs(index));
    let Range { declare, k_begin, k_end, out } = range(aligned, split);
    let end = if aligned { "" } else { ", ke" };
    // Accumulator `c{2 i + h}` holds row `i` (of 8), columns half `h`. The loops are unrolled
    // here so the accumulators stay in registers.
    let mut decl = String::new();
    let mut update = String::new();
    let mut store = String::new();
    for i in 0..8 {
        for h in 0..2 {
            decl += &format!("var c{} = vec4<f32>(0.0);\n    ", i * 2 + h);
            let (row, col) = ((i / 4) * 64 + i % 4, h * 64);
            let writes = if aligned {
                format!(
                    "let i = {out}r * p.n + col;
        c[i] = v.x; c[i + 1u] = v.y; c[i + 2u] = v.z; c[i + 3u] = v.w;"
                )
            } else {
                let lanes: Vec<String> = (0..4)
                    .map(|j| format!("if (col + {j}u < p.n) {{ c[i{}] = v.{}; }}", if j == 0 { String::new() } else { format!(" + {j}u") }, LANES[j]))
                    .collect();
                format!("if (r < p.m) {{\n            let i = {out}r * p.n + col;\n            {}\n        }}", lanes.join("\n            "))
            };
            store += &format!(
                "{{
        let r = row0 + {row} + lid.y * 4u;
        let col = col0 + {col} + lid.x * 4u;
        let v = c{};
        {writes}
    }}\n    ",
                i * 2 + h
            );
        }
    }
    // Two consecutive `k` per trip of the loop over the slice.
    for u in 0..2 {
        update += &format!(
            "{{
                let o = (kk + {u}u) * 32u;
                let a0 = at[o + lid.y];
                let a1 = at[o + lid.y + 16u];
                let b0 = bt[o + lid.x];
                let b1 = bt[o + lid.x + 16u];
                "
        );
        for i in 0..8 {
            let a_row = format!("a{}.{}", i / 4, LANES[i % 4]);
            for h in 0..2 {
                let acc = format!("c{}", i * 2 + h);
                update += &format!("{acc} = fma(vec4<f32>({a_row}), b{h}, {acc});\n                ");
            }
        }
        update += "}\n            ";
    }
    let stage = loads.stage(end);
    format!(
        "var<workgroup> at: array<vec4<f32>, 512>;
var<workgroup> bt: array<vec4<f32>, 512>;

// Slot `e` of a slice stored k-major: k `e % 16`, row groups `e / 16`.
fn k_major(e: u32) -> u32 {{ return (e % 16u) * 32u + e / 16u; }}
{load_a}
{load_b}

@compute @workgroup_size(16, 16)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {{
    // The output tiles are numbered along the folded grid (see `plan`).
    let columns = (p.n + 127u) / 128u;
    let tile = wid.x + wid.y * nwg.x;
    if (tile >= columns * ((p.m + 127u) / 128u)) {{ return; }}
    let row0 = (tile / columns) * 128u;
    let col0 = (tile % columns) * 128u;
    let t = lid.y * 16u + lid.x;
{declare}    {decl}
    for (var k0 = {k_begin}; k0 < {k_end}; k0 = k0 + 16u) {{
        {stage}
        workgroupBarrier();
        for (var kk = 0u; kk < 16u; kk = kk + 2u) {{
            {update}}}
        workgroupBarrier();
    }}
    {store}}}
"
    )
}

/// `fn load_a` / `fn load_b` of `T32`: slot `e` of a slice holds one row (column) of one `k`.
fn t32_load(left: bool, aligned: bool, lanes: Lanes, load: &dyn Fn(&str) -> String) -> String {
    let (k, group) = lanes.split();
    let (name, base) = if left { ("load_a", "row0") } else { ("load_b", "col0") };
    if aligned {
        let index = if left { "r * p.k + k" } else { "k * p.n + r" };
        format!(
            "fn {name}({base}: u32, k0: u32, e: u32) -> f32 {{
    let r = {base} + {group};
    let k = k0 + {k};
    return {};
}}",
            load(index)
        )
    } else {
        let (index, limit) = if left { ("min(r, p.m - 1u) * p.k + min(k, ke - 1u)", "p.m") } else { ("min(k, ke - 1u) * p.n + min(r, p.n - 1u)", "p.n") };
        format!(
            "fn {name}({base}: u32, k0: u32, e: u32, ke: u32) -> f32 {{
    let r = {base} + {group};
    let k = k0 + {k};
    return select(0.0, {}, r < {limit} && k < ke);
}}",
            load(index)
        )
    }
}

fn t32(aligned: bool, split: bool, loads: &Loads, form: &ContractForm) -> String {
    let load_a = t32_load(true, aligned, loads.lhs.lanes, &|index| loads.lhs(index));
    let load_b = t32_load(false, aligned, loads.rhs.lanes, &|index| loads.rhs(index));
    let Range { declare, k_begin, k_end, out } = range(aligned, split);
    let end = if aligned { "" } else { ", ke" };
    // A batch writes its layer at `zi * p.sc`.
    let out = if loads.batch { "zi * p.sc + " } else { out };
    // Rows `lid.y` and `lid.y + 16`, columns `lid.x` and `lid.x + 16` of the tile.
    let store = |acc: &str, row: &str, col: &str| {
        if aligned {
            format!("{{ let r = row0 + {row}; let col = col0 + {col}; c[{out}r * p.n + col] = {acc}; }}\n    ")
        } else {
            format!("{{ let r = row0 + {row}; let col = col0 + {col}; if (r < p.m && col < p.n) {{ c[{out}r * p.n + col] = {acc}; }} }}\n    ")
        }
    };
    // A product that is not a sum of multiplications must not combine the zeros that pad the last slice of K.
    let steps = if form.is_default() || aligned { "16u".to_owned() } else { format!("min(16u, {k_end} - k0)") };
    let init = form.identity();
    let (u00, u01, u10, u11) = (form.update("c00", "a0", "b0"), form.update("c01", "a0", "b1"), form.update("c10", "a1", "b0"), form.update("c11", "a1", "b1"));
    let private = if loads.batch { "var<private> zi: u32 = 0u;\n" } else { "" };
    let set_zi = if loads.batch { "    zi = wid.z;\n" } else { "" };
    let mac = form.mac_function();
    let stores = [("c00", "lid.y", "lid.x"), ("c01", "lid.y", "lid.x + 16u"), ("c10", "lid.y + 16u", "lid.x"), ("c11", "lid.y + 16u", "lid.x + 16u")]
        .iter()
        .map(|&(acc, row, col)| store(acc, row, col))
        .collect::<String>();
    let stage = loads.stage(end);
    format!(
        "{private}var<workgroup> at: array<f32, 512>;
var<workgroup> bt: array<f32, 512>;
{mac}
// Slot `e` of a slice stored k-major: k `e % 16`, row `e / 16`.
fn k_major(e: u32) -> u32 {{ return (e % 16u) * 32u + e / 16u; }}
{load_a}
{load_b}

@compute @workgroup_size(16, 16)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {{
    // The output tiles are numbered along the folded grid (see `plan`).
    let columns = (p.n + 31u) / 32u;
    let tile = wid.x + wid.y * nwg.x;
    if (tile >= columns * ((p.m + 31u) / 32u)) {{ return; }}
{set_zi}    let row0 = (tile / columns) * 32u;
    let col0 = (tile % columns) * 32u;
    let t = lid.y * 16u + lid.x;
{declare}    var c00 = {init};
    var c01 = {init};
    var c10 = {init};
    var c11 = {init};
    for (var k0 = {k_begin}; k0 < {k_end}; k0 = k0 + 16u) {{
        {stage}
        workgroupBarrier();
        for (var kk = 0u; kk < {steps}; kk = kk + 1u) {{
            let o = kk * 32u;
            let a0 = at[o + lid.y];
            let a1 = at[o + lid.y + 16u];
            let b0 = bt[o + lid.x];
            let b1 = bt[o + lid.x + 16u];
            {u00}
            {u01}
            {u10}
            {u11}
        }}
        workgroupBarrier();
    }}
    {stores}}}
"
    )
}

/// The statements storing accumulator `acc` (a `vec4`) at row `row`, columns from `col` of a tile.
fn store_vec4(aligned: bool, out: &str, row: &str, col: &str, acc: &str) -> String {
    const LANES: [&str; 4] = ["x", "y", "z", "w"];
    let writes = if aligned {
        format!("let i = {out}r * p.n + col;\n        c[i] = v.x; c[i + 1u] = v.y; c[i + 2u] = v.z; c[i + 3u] = v.w;")
    } else {
        let lanes: Vec<String> = (0..4)
            .map(|j| format!("if (col + {j}u < p.n) {{ c[i{}] = v.{}; }}", if j == 0 { String::new() } else { format!(" + {j}u") }, LANES[j]))
            .collect();
        format!("if (r < p.m) {{\n            let i = {out}r * p.n + col;\n            {}\n        }}", lanes.join("\n            "))
    };
    format!("{{\n        let r = row0 + {row};\n        let col = col0 + {col};\n        let v = {acc};\n        {writes}\n    }}\n    ")
}

/// `MatmulTile::T128x32`: 128 rows by 32 columns, the rows staged as in `T128` and the columns
/// as in `T32`; each invocation owns 8 rows (two groups of 4, 64 apart) of columns `lid.x` and
/// `lid.x + 16`.
fn t128x32(aligned: bool, split: bool, loads: &Loads) -> String {
    const LANES: [&str; 4] = ["x", "y", "z", "w"];
    let load_a = t128_load(true, aligned, loads.lhs.lanes, &|index| loads.lhs(index));
    let load_b = t32_load(false, aligned, loads.rhs.lanes, &|index| loads.rhs(index));
    let Range { declare, k_begin, k_end, out } = range(aligned, split);
    let end = if aligned { "" } else { ", ke" };
    let mut decl = String::new();
    let mut update = String::new();
    let mut store = String::new();
    for i in 0..8 {
        for h in 0..2 {
            decl += &format!("var c{i}_{h} = 0.0;\n    ");
            let (row, col) = (format!("{} + lid.y * 4u", (i / 4) * 64 + i % 4), format!("{} + lid.x", h * 16));
            let guard = if aligned { String::new() } else { " if (r < p.m && col < p.n)".to_owned() };
            let write = format!("c[{out}r * p.n + col] = c{i}_{h};");
            store += &if aligned {
                format!("{{ let r = row0 + {row}; let col = col0 + {col}; {write} }}\n    ")
            } else {
                format!("{{ let r = row0 + {row}; let col = col0 + {col};{guard} {{ {write} }} }}\n    ")
            };
        }
    }
    for u in 0..2 {
        update += &format!(
            "{{
                let o = (kk + {u}u) * 32u;
                let a0 = at[o + lid.y];
                let a1 = at[o + lid.y + 16u];
                let b0 = bt[o + lid.x];
                let b1 = bt[o + lid.x + 16u];
                "
        );
        for i in 0..8 {
            let a_row = format!("a{}.{}", i / 4, LANES[i % 4]);
            for h in 0..2 {
                update += &format!("c{i}_{h} = fma({a_row}, b{h}, c{i}_{h});\n                ");
            }
        }
        update += "}\n            ";
    }
    let stage = loads.stage(end);
    format!(
        "var<workgroup> at: array<vec4<f32>, 512>;
var<workgroup> bt: array<f32, 512>;

// Slot `e` of a slice stored k-major: k `e % 16`, row (group) `e / 16`.
fn k_major(e: u32) -> u32 {{ return (e % 16u) * 32u + e / 16u; }}
{load_a}
{load_b}

@compute @workgroup_size(16, 16)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {{
    // The output tiles are numbered along the folded grid (see `plan`).
    let columns = (p.n + 31u) / 32u;
    let tile = wid.x + wid.y * nwg.x;
    if (tile >= columns * ((p.m + 127u) / 128u)) {{ return; }}
    let row0 = (tile / columns) * 128u;
    let col0 = (tile % columns) * 32u;
    let t = lid.y * 16u + lid.x;
{declare}    {decl}
    for (var k0 = {k_begin}; k0 < {k_end}; k0 = k0 + 16u) {{
        {stage}
        workgroupBarrier();
        for (var kk = 0u; kk < 16u; kk = kk + 2u) {{
            {update}}}
        workgroupBarrier();
    }}
    {store}}}
"
    )
}

/// `MatmulTile::T32x128`: 32 rows by 128 columns, the rows staged as in `T32` and the columns
/// as in `T128`; each invocation owns rows `lid.y` and `lid.y + 16`, two groups of 4 columns.
fn t32x128(aligned: bool, split: bool, loads: &Loads) -> String {
    let load_a = t32_load(true, aligned, loads.lhs.lanes, &|index| loads.lhs(index));
    let load_b = t128_load(false, aligned, loads.rhs.lanes, &|index| loads.rhs(index));
    let Range { declare, k_begin, k_end, out } = range(aligned, split);
    let end = if aligned { "" } else { ", ke" };
    let mut decl = String::new();
    let mut update = String::new();
    let mut store = String::new();
    for i in 0..2 {
        for h in 0..2 {
            decl += &format!("var c{} = vec4<f32>(0.0);\n    ", i * 2 + h);
            store += &store_vec4(aligned, out, &format!("{} + lid.y", i * 16), &format!("{} + lid.x * 4u", h * 64), &format!("c{}", i * 2 + h));
        }
    }
    for u in 0..2 {
        update += &format!(
            "{{
                let o = (kk + {u}u) * 32u;
                let a0 = at[o + lid.y];
                let a1 = at[o + lid.y + 16u];
                let b0 = bt[o + lid.x];
                let b1 = bt[o + lid.x + 16u];
                "
        );
        for i in 0..2 {
            for h in 0..2 {
                let acc = format!("c{}", i * 2 + h);
                update += &format!("{acc} = fma(vec4<f32>(a{i}), b{h}, {acc});\n                ");
            }
        }
        update += "}\n            ";
    }
    let stage = loads.stage(end);
    format!(
        "var<workgroup> at: array<f32, 512>;
var<workgroup> bt: array<vec4<f32>, 512>;

// Slot `e` of a slice stored k-major: k `e % 16`, row (group) `e / 16`.
fn k_major(e: u32) -> u32 {{ return (e % 16u) * 32u + e / 16u; }}
{load_a}
{load_b}

@compute @workgroup_size(16, 16)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {{
    // The output tiles are numbered along the folded grid (see `plan`).
    let columns = (p.n + 127u) / 128u;
    let tile = wid.x + wid.y * nwg.x;
    if (tile >= columns * ((p.m + 31u) / 32u)) {{ return; }}
    let row0 = (tile / columns) * 32u;
    let col0 = (tile % columns) * 128u;
    let t = lid.y * 16u + lid.x;
{declare}    {decl}
    for (var k0 = {k_begin}; k0 < {k_end}; k0 = k0 + 16u) {{
        {stage}
        workgroupBarrier();
        for (var kk = 0u; kk < 16u; kk = kk + 2u) {{
            {update}}}
        workgroupBarrier();
    }}
    {store}}}
"
    )
}

/// The skinny kernel (`MatmulTile::Skinny`) for products with a small output and a long `K`:
/// each of the `width` invocations of a workgroup owns `rows` x `cols` outputs in registers and
/// loads both operands straight from global memory, with neither staging nor barriers in the loop.
/// The output tiles number `columns * row tiles`; a workgroup covers `width / groups` of them
/// (grid layer `y` the tiles from `y * width / groups`) and the `groups` invocations that own the
/// same tile take the `k` of the workgroup's slice in turn (`k = kb + group + groups * i`), then
/// add their accumulators pairwise through workgroup memory, so that the sums have a fixed order.
/// The workgroup of grid layer `x` covers `K` from `x * kc` to `min(K, (x + 1) * kc)` and writes its
/// tiles at `x * M * N`. With `along_k` the invocations of a warp are neighbouring groups, else
/// neighbouring tiles.
///
/// Rows and columns past the output read the last row (column) and are not stored: the loop
/// carries no bounds checks.
fn skinny(rows: u32, cols: u32, groups: u32, width: u32, along_k: bool, loads: &Loads) -> String {
    let per_group = width / groups;
    let mut decl = String::new();
    let mut bases = String::new();
    let mut write = String::new();
    let mut read = String::new();
    let mut store = String::new();
    // Tile `(m0, n0)` owns the rows `m0..m0 + rows` and the columns `n0..n0 + cols`; consecutive
    // loads of a row of the left operand stored along `M` (or of the right operand) are
    // neighbours, which the compiler fuses into wide loads.
    for i in 0..rows {
        bases += &format!("    let ra{i} = min(m0 + {i}u, p.m - 1u) * p.k;\n");
    }
    for j in 0..cols {
        bases += &format!("    let cb{j} = min(n0 + {j}u, p.n - 1u);\n");
    }
    for i in 0..rows {
        for j in 0..cols {
            let e = i * cols + j;
            decl += &format!("    var c{i}_{j} = 0.0;\n");
            write += &format!("            part[o + {e}u * q] = c{i}_{j};\n");
            read += &format!("            c{i}_{j} += part[o + {e}u * q];\n");
            store += &format!("    if (g == 0u && live && m0 + {i}u < p.m && n0 + {j}u < p.n) {{ c[ob + {i}u * p.n + {j}u] = c{i}_{j}; }}\n");
        }
    }
    // One `k` of this invocation: all the loads first, then the multiply-adds.
    let mut step = String::from("        let kn = k * p.n;\n");
    for i in 0..rows {
        step += &format!("        let a{i} = {};\n", loads.lhs(&format!("ra{i} + k")));
    }
    for j in 0..cols {
        step += &format!("        let b{j} = {};\n", loads.rhs(&format!("kn + cb{j}")));
    }
    for i in 0..rows {
        for j in 0..cols {
            step += &format!("        c{i}_{j} = fma(a{i}, b{j}, c{i}_{j});\n");
        }
    }
    // Workgroup memory of the reduction, slot `e` of the accumulators of group `g` (of those that
    // write, `g - s`) and tile `tl`: neighbouring invocations write neighbouring words.
    let (layout, group) = if along_k {
        ("tl * s + (g - s)".to_owned(), "tl * s + g".to_owned())
    } else {
        (format!("(g - s) * {per_group}u + tl"), format!("g * {per_group}u + tl"))
    };
    let (part, reduce) = if groups > 1 {
        (
            format!("var<workgroup> part: array<f32, {}>;\n", rows * cols * width / 2),
            format!(
                "    for (var s = {}u; s > 0u; s = s / 2u) {{
        let q = s * {per_group}u;
        if (g >= s && g < 2u * s) {{
            let o = {layout};
{write}        }}
        workgroupBarrier();
        if (g < s) {{
            let o = {group};
{read}        }}
        workgroupBarrier();
    }}
",
                groups / 2
            ),
        )
    } else {
        (String::new(), String::new())
    };
    let lanes = if along_k { format!("let g = tid % {groups}u;\n    let tl = tid / {groups}u;") } else { format!("let g = tid / {per_group}u;\n    let tl = tid % {per_group}u;") };
    format!(
        "{part}
@compute @workgroup_size({width})
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) tid: u32) {{
    let columns = (p.n + {}u) / {cols}u;
    let tiles = columns * ((p.m + {}u) / {rows}u);
    {lanes}
    let tile = wid.y * {per_group}u + tl;
    let live = tile < tiles;
    let m0 = (tile / columns) * {rows}u;
    let n0 = (tile % columns) * {cols}u;
    let kb = wid.x * p.kc;
    let ke = min(p.k, kb + p.kc);
    let ob = wid.x * p.m * p.n + m0 * p.n + n0;
{bases}{decl}    for (var k = select(ke, kb + g, live); k < ke; k += {groups}u) {{
{step}    }}
{reduce}{store}}}
",
        cols - 1,
        rows - 1
    )
}
