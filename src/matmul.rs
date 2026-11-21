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
//! buffer by the third grid dimension and summed by a second launch ([`split_of`]).
//!
//! Products with a small output and a long K (a Gram matrix, one-hot labels times features, a
//! gradient `Xᵀ r`, a matrix times a few vectors) have mostly padding in those tiles. They run as
//! skinny kernels instead ([`skinny_plan`]): each invocation owns up to 10x8 outputs in registers
//! and loads straight from global memory, without staging or barriers in the loop, and the
//! invocations the small output does not need take other `k`; a second launch sums the slices of
//! K as above.
//!
//! An operand that is a pending expression is not evaluated first: its program is flattened
//! into the kernel (`jit::program::Tail::Matmul`), which calls it as `luma_lhs(i)` /
//! `luma_rhs(i)` where a buffer kernel indexes `a` / `b`. A transposed operand is such an
//! expression (a view), and the kernel then divides its slices so that neighbouring invocations
//! load neighbours in memory ([`MatmulOperand::Transposed`]). Programs that read sub-programs
//! through circular shifts, or arrays split over several buffers, are evaluated instead. The
//! product itself is not lazy: it is recorded when called and returns an evaluated array, so what
//! consumes it runs as its own kernel.

use std::panic::Location;
use std::sync::Arc;

use crate::array::{Array, Failure};
use crate::builtin::Builtin;
use crate::context::{Context, RawArg, Scratch};
use crate::error::LumaError;
use crate::jit::{self, AxisMap, Edge, ElemTy, MatmulSide, OpKind, ShiftKind, Src};
use crate::ops::{fold_grid, Outcome};
use crate::tuning::Tuning;

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
            MatmulTile::Skinny { .. } => unreachable!("a skinny product has no output tiles of fixed edges"),
        }
    }
}

/// K elements of each operand a workgroup stages at a time.
const SLICE: u32 = 16;

/// Output tiles below which a long K is split (see [`split_of`]); like [`SPLIT_GROUPS`] a count
/// for the 82 multiprocessors of an RTX 3090, raised for bigger devices ([`Tuning::scaled_up`]).
const SPLIT_TILES: u32 = 128;
/// Shortest K that is split.
const SPLIT_MIN_K: u32 = 4096;
/// Workgroups a split product aims for. The 4 tiles of a 64x64 product over 2^20 take 3.0 ms
/// with 256 (3 workgroups per multiprocessor of an RTX 3090, which cannot hide the latency of
/// the slices), 2.5 with 512, 2.3 with 1024, 2.27 with 2048. Raised in proportion to the
/// multiprocessors of bigger devices ([`Tuning::scaled_up`]), never lowered: unchanged on the
/// 3090, and on an RTX 3080 Laptop (48) 1024 groups beat the proportional 599 by 2.7 %.
const SPLIT_GROUPS: u32 = 1024;
/// Smallest K a slice of a split product covers.
const SPLIT_MIN_SLICE: u32 = 1024;

/// Most outputs of a product that runs as a skinny kernel when its `K` is at least
/// [`SKINNY_LONG_K`]: a 96x96 Gram matrix over 65536 samples takes 0.29 ms on an RTX 3090 against
/// 0.38 for the staged tiles, a 128x128 one 0.35 against 0.22.
const SKINNY_MAX_OUTPUTS: u32 = 9216;
/// Most outputs for which a `K` of [`SKINNY_MIN_K`] is long enough (a 32x32 product is 2.5x
/// faster at 4096, 16 x 16 at 1M 2.4x; 64x64 over 16384 is 60% slower).
const SKINNY_SMALL_OUTPUTS: u32 = 512;
/// Shortest K of the products with up to [`SKINNY_MAX_OUTPUTS`] outputs.
const SKINNY_LONG_K: u32 = 65536;
/// Most outputs of one that reads its left operand along `K` (a matrix times a few vectors).
const SKINNY_MAX_OUTPUTS_ALONG_K: u32 = 1 << 22;
/// Shortest K of a skinny product.
const SKINNY_MIN_K: u32 = 2048;
/// Workgroups a skinny product aims for: one-hot 1M x 32 on an RTX 3090 takes 0.19 ms with 256 or
/// 1024, 0.18 with 512, 0.21 with 2048.
const SKINNY_GROUPS: u32 = 512;
/// Smallest K a workgroup of a skinny product covers: a 64x64 product over 65536 takes 0.14 ms
/// with 256 and 0.11 with 128 or 64 (at most 512 workgroups of 256 each otherwise).
const SKINNY_MIN_SLICE: u32 = 128;
/// Most accumulators an invocation of a skinny kernel holds.
const SKINNY_ACCUMULATORS: u32 = 80;
/// Most `f32` of workgroup memory a skinny kernel's reduction across groups uses (16 KiB, the
/// smallest limit of the backends).
const SKINNY_PART: u32 = 4096;
/// Tiles that read the same operand element from which a pending operand is evaluated first.
const SKINNY_EVALUATE_READS: u32 = 8;
/// Tiles that read the same element of a pending operand of several buffers from which the staged
/// tiles are preferred to a skinny kernel (below [`SKINNY_EVALUATE_READS`]).
const SKINNY_FUSED_READS: u32 = 4;

impl Array<f32> {
    /// The matrix product with `rhs`. `self` has shape `[K, M]`, `rhs` `[N, K]` and the result
    /// `[N, M]`: element `(n, m)` is `Σ_k self[k, m] * rhs[n, k]`, which is the row-major
    /// product of an `M x K` and a `K x N` matrix (axis 0, the fastest, is the column index).
    /// One-dimensional operands count as having extent 1 on axis 1.
    ///
    /// Operands that are lazy expressions are fused into the kernel's loads instead of being
    /// evaluated first. The product is recorded when this is called and the result is an
    /// evaluated array: what consumes it runs as its own kernels.
    ///
    /// Products with few output tiles and a long `K` are split over `K` and summed in a second
    /// kernel; the sum order is fixed, so results are reproducible.
    ///
    /// # Errors
    ///
    /// The result is poisoned with [`LumaError::ShapeMismatch`] if `self.shape()[0] !=
    /// rhs.shape()[1]` or an operand has more than two axes of extent above 1,
    /// [`LumaError::ContextMismatch`] if the operands live on different contexts and
    /// [`LumaError::Chunked`] if an operand or the result needs more than one buffer.
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
}

/// [`Array::matmul`] as a function: the product of `lhs` (`[K, M]`) and `rhs` (`[N, K]`) as an
/// array of shape `[N, M]`.
#[track_caller]
pub fn matmul(lhs: &Array<f32>, rhs: &Array<f32>) -> Array<f32> {
    let location = Location::caller();
    let result = run(lhs, rhs);
    let shape = [rhs.shape().dims()[0], lhs.shape().dims()[1], 1, 1];
    let shape = if shape.contains(&0) { lhs.shape() } else { shape.into() };
    Array::resolve(lhs.context(), shape, result, "matmul", location)
}

/// How a product is cut into workgroups.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct Plan {
    tile: MatmulTile,
    aligned: bool,
    /// Whether the kernel writes the slice of its workgroup at `z * M * N` (see [`body`]);
    /// `slices > 1` unless the kernel always does.
    split: bool,
    /// Slices of K, 1 if the product is not split.
    slices: u32,
    /// Elements of K per slice (K if not split).
    kc: u32,
    /// `[column tiles, row tiles, slices]`; a skinny kernel has `[slices, 1, 1]`.
    groups: [u32; 3],
}

/// The slices a product of `tiles` output tiles over `k` is split into, and the elements of K
/// per slice (a multiple of the staged slice). Few tiles and a long K: about
/// [`SPLIT_GROUPS`] workgroups, each slice at least [`SPLIT_MIN_SLICE`] long. `None` if not split.
fn split_of(tuning: &Tuning, tiles: u32, k: u32) -> Option<(u32, u32)> {
    if tiles >= tuning.scaled_up(SPLIT_TILES) || k < SPLIT_MIN_K {
        return None;
    }
    let wanted = tuning.scaled_up(SPLIT_GROUPS).div_ceil(tiles).min(k / SPLIT_MIN_SLICE);
    if wanted < 2 {
        return None;
    }
    let kc = k.div_ceil(wanted).next_multiple_of(SLICE);
    let slices = k.div_ceil(kc);
    (slices >= 2).then_some((slices, kc))
}

/// The skinny kernel of an `m x n` output and the workgroups that cover its tiles: the
/// invocation tile (at most [`SKINNY_ACCUMULATORS`] outputs, up to 10 rows by 8 columns) that
/// costs the least per `k`, counting a fused multiply-add as 1 and a load as 2, over all the lanes
/// of the workgroups.
///
/// A workgroup is 64 invocations, or the next power of two of the tiles if there are more (up to
/// 256); with fewer tiles than that the surplus invocations form groups that take the `k` in turn
/// (a power of two of them, so that the groups reduce pairwise). `along_k` is for a left operand
/// stored along `K`: neighbouring invocations then take neighbouring `k` (32 or 64 groups of a
/// workgroup), and the output tiles are as many workgroups as they need; the right operand is
/// then read along `K` too if it has at most 8 columns, which is all of them.
fn skinny_tile(m: u32, n: u32, along_k: bool) -> Option<(MatmulTile, u32)> {
    let mut best: Option<((u32, u32), (MatmulTile, u32))> = None;
    let min_cols = if along_k { n } else { 1 };
    // Memory bound along `K`: a tile of at most 8 rows and 64 outputs, whose right operand's
    // columns weigh more (each is a load of its own, strided by the columns).
    let (max_rows, max_outputs) = if along_k { (8, 64) } else { (10, SKINNY_ACCUMULATORS) };
    for cols in min_cols..=n.min(8) {
        for rows in 1..=m.min(max_rows) {
            if rows * cols > max_outputs {
                continue;
            }
            let tiles = m.div_ceil(rows) * n.div_ceil(cols);
            let (width, groups) = if along_k {
                (64, if tiles >= 2 { 32 } else { 64 })
            } else {
                let width = tiles.next_power_of_two().max(64);
                if width > 256 {
                    continue;
                }
                (width, 1 << (width / tiles).ilog2())
            };
            let tile_groups = tiles.div_ceil(width / groups);
            if groups > 1 && rows * cols * width / 2 > SKINNY_PART {
                continue;
            }
            let loads = 2 * rows + if along_k && cols > 1 { 2 * cols * cols } else { 2 * cols };
            let cost = tile_groups * (width / groups) * (rows * cols + loads);
            let key = (cost, u32::MAX - rows * cols);
            if best.is_none_or(|(k, _)| key < k) {
                let tile = MatmulTile::Skinny { rows: rows as u8, cols: cols as u8, groups: groups as u8, width: width as u16, along_k };
                best = Some((key, (tile, tile_groups)));
            }
        }
    }
    best.map(|(_, tile)| tile)
}

/// The plan of a small output over a long K (a Gram matrix, one-hot labels times features, a
/// gradient `Xᵀ r`, a matrix times a few vectors) as a skinny kernel: `slices` of K, each its
/// own workgroups, about [`SKINNY_GROUPS`] workgroups in all, no slice shorter than
/// [`SKINNY_MIN_SLICE`]. `along_k`: the left operand is stored along `K` (see [`skinny_tile`]).
/// `None` if the product is not of that kind.
fn skinny_plan(m: u32, n: u32, k: u32, along_k: bool, max_groups: u32, max_bytes: u64) -> Option<Plan> {
    let along_k = along_k && n <= 8;
    let outputs = u64::from(m) * u64::from(n);
    let few = if along_k {
        outputs <= u64::from(SKINNY_MAX_OUTPUTS_ALONG_K)
    } else {
        outputs <= u64::from(SKINNY_SMALL_OUTPUTS) || (outputs <= u64::from(SKINNY_MAX_OUTPUTS) && k >= SKINNY_LONG_K)
    };
    if m == 0 || n == 0 || k < SKINNY_MIN_K || !few {
        return None;
    }
    let (tile, tile_groups) = skinny_tile(m, n, along_k)?;
    let mut wanted = SKINNY_GROUPS.div_ceil(tile_groups).min(k / SKINNY_MIN_SLICE).clamp(1, max_groups);
    loop {
        let kc = k.div_ceil(wanted);
        let slices = k.div_ceil(kc);
        if wanted == 1 || u64::from(slices) * u64::from(m) * u64::from(n) * 4 <= max_bytes {
            if tile_groups > max_groups {
                return None;
            }
            return Some(Plan { tile, aligned: false, split: true, slices, kc, groups: [slices, tile_groups, 1] });
        }
        wanted /= 2;
    }
}

/// The plan of an `m x k` by `k x n` product. `along_k`: the left operand's elements along `K` are
/// neighbours in memory (not a transposed view). `max_groups` is the most workgroups per grid
/// dimension, `max_bytes` the most a buffer may hold (the split product's partial sums have to
/// fit one).
fn plan(tuning: &Tuning, m: u32, n: u32, k: u32, along_k: bool, max_groups: u32, max_bytes: u64) -> Result<Plan, LumaError> {
    if let Some(plan) = skinny_plan(m, n, k, along_k, max_groups, max_bytes) {
        return Ok(plan);
    }
    tiled_plan(tuning, m, n, k, max_groups, max_bytes)
}

/// [`plan`] with the staged tiles (never a skinny kernel).
fn tiled_plan(tuning: &Tuning, m: u32, n: u32, k: u32, max_groups: u32, max_bytes: u64) -> Result<Plan, LumaError> {
    let tile = match (m >= 128, n >= 128) {
        (true, true) => MatmulTile::T128,
        (true, false) => MatmulTile::T128x32,
        (false, true) => MatmulTile::T32x128,
        (false, false) => MatmulTile::T32,
    };
    let (row_edge, col_edge) = tile.edges();
    let aligned = m.is_multiple_of(row_edge) && n.is_multiple_of(col_edge) && k.is_multiple_of(SLICE);
    let (columns, rows) = (n.div_ceil(col_edge), m.div_ceil(row_edge));
    let tiles = columns.saturating_mul(rows);
    let (slices, kc) = split_of(tuning, tiles, k)
        .filter(|&(slices, _)| u64::from(slices) * u64::from(m) * u64::from(n) * 4 <= max_bytes)
        .unwrap_or((1, k));
    // Row-major tile numbers (see `body`): the grid is `[columns, rows]` if that fits, else the
    // tiles are folded over it, the groups past the last tile exiting at once.
    let [x, y, _] = if columns <= max_groups && rows <= max_groups { [columns, rows, 1] } else { fold_grid(tiles, max_groups) };
    let groups = [x, y, slices];
    if groups.iter().any(|&count| count > max_groups) {
        return Err(LumaError::KernelGridTooLarge { workgroups: groups, max: max_groups });
    }
    Ok(Plan { tile, aligned, split: slices > 1, slices, kc, groups })
}

/// Output tile edges a native product chooses from, largest first (see [`native_plan`]).
const NATIVE_EDGES: [u32; 3] = [128, 64, 32];
/// Output tiles per core a native product aims for: the cores take them as they free up.
const NATIVE_TILES_PER_CORE: u32 = 2;
/// Smallest K a slice of a split native product covers.
const NATIVE_SPLIT_SLICE: u32 = 1024;

/// The plan of an `m x k` by `k x n` product the backend multiplies as native code on `cores`
/// cores: the largest tile of [`NATIVE_EDGES`] that leaves [`NATIVE_TILES_PER_CORE`] tiles per
/// core, else the smallest; with fewer tiles than that and a long K, K is split over the third
/// grid dimension like [`split_of`] does (partial products summed by a second launch).
fn native_plan(m: u32, n: u32, k: u32, cores: u32, transposed: [bool; 2], max_groups: u32, max_bytes: u64) -> Result<Plan, LumaError> {
    let wanted = cores.max(1).saturating_mul(NATIVE_TILES_PER_CORE);
    let tiles_at = |edge: u32| m.div_ceil(edge).saturating_mul(n.div_ceil(edge));
    let edge = NATIVE_EDGES.into_iter().find(|&edge| tiles_at(edge) >= wanted).unwrap_or(NATIVE_EDGES[NATIVE_EDGES.len() - 1]);
    let (columns, rows) = (n.div_ceil(edge), m.div_ceil(edge));
    let tiles = columns.saturating_mul(rows);
    let split = (tiles < wanted && k >= 2 * NATIVE_SPLIT_SLICE).then(|| {
        let kc = k.div_ceil(wanted.div_ceil(tiles).min(k / NATIVE_SPLIT_SLICE)).next_multiple_of(SLICE);
        (k.div_ceil(kc), kc)
    });
    let (slices, kc) = split
        .filter(|&(slices, _)| slices >= 2 && u64::from(slices) * u64::from(m) * u64::from(n) * 4 <= max_bytes)
        .unwrap_or((1, k));
    let [x, y, _] = if columns <= max_groups && rows <= max_groups { [columns, rows, 1] } else { fold_grid(tiles, max_groups) };
    let groups = [x, y, slices];
    if groups.iter().any(|&count| count > max_groups) {
        return Err(LumaError::KernelGridTooLarge { workgroups: groups, max: max_groups });
    }
    Ok(Plan { tile: MatmulTile::Native { edge, lhs_t: transposed[0], rhs_t: transposed[1] }, aligned: false, split: slices > 1, slices, kc, groups })
}

/// The operand of a native product that is a transposed view (a swap of axes 0 and 1, see
/// `host::transposed_source`): what it is the transpose of, so that the product reads that
/// buffer through its strides instead of the view being copied.
fn transposed_source(array: &Array<f32>) -> Option<Src> {
    let node = array.pending()?;
    let op = node.op();
    let OpKind::Shift(ShiftKind::View { map, edge: Edge::None }) = op.kind else { return None };
    let unit = |from: u8| AxisMap { from, start: 0, step: 1 };
    let swaps = op.ty == ElemTy::F32
        && map.axes[0] == unit(1)
        && map.axes[1] == unit(0)
        && map.axes[2] == unit(2)
        && map.axes[3] == unit(3)
        && map.src[0] == map.out[1]
        && map.src[1] == map.out[0]
        && map.src[2..] == [1, 1]
        && map.out[2..] == [1, 1];
    if !swaps || node.failure().is_some() || node.is_looped() {
        return None;
    }
    let [source, ..] = node.operands()?;
    matches!(source, Src::Buffer(_) | Src::Node(_)).then_some(source)
}

/// The operands of a product that is multiplied natively, with whether each is read transposed:
/// the native product reads buffers, so the operands that are expressions (or the expressions a
/// transposed view reads) are evaluated first, together, sharing what they read.
fn native_sides(ctx: &Context, lhs: &Array<f32>, rhs: &Array<f32>) -> Outcome<([MatmulSide; 2], [bool; 2])> {
    // A poisoned operand reports its own error.
    let mut sources = [None, None];
    for (array, source) in [lhs, rhs].into_iter().zip(&mut sources) {
        *source = transposed_source(array);
        match (&*source, array.pending()) {
            (Some(Src::Node(node)), _) if node.is_pending() => array.context().jit().request(node),
            (Some(_), _) => {}
            (None, Some(node)) => array.context().jit().request(node),
            (None, None) => drop(array.ready()?),
        }
    }
    if !ctx.same_as(rhs.context()) {
        return Err(LumaError::ContextMismatch.into());
    }
    ctx.jit().evaluate_requested();
    let mut sides = Vec::with_capacity(2);
    for (array, source) in [lhs, rhs].into_iter().zip(sources) {
        let (storage, transposed) = match source {
            Some(Src::Buffer(storage)) => (storage, true),
            Some(Src::Node(node)) => (Arc::clone(node.force().as_ref().map_err(|error| Failure::Inherited(Arc::clone(error)))?), true),
            _ => {
                let (pin, bytes) = array.ready()?;
                sides.push((MatmulSide::Ready(pin, bytes), false));
                continue;
            }
        };
        sides.push((MatmulSide::Ready(storage.pin_single()?, storage.byte_len()), transposed));
    }
    let [(lhs_side, lhs_t), (rhs_side, rhs_t)]: [_; 2] = sides.try_into().ok().ok_or_else(|| Failure::from(LumaError::Gpu("matrix product operands".into())))?;
    Ok(([lhs_side, rhs_side], [lhs_t, rhs_t]))
}

/// The side of the product `array` stands for: its expression if it is pending (so it may be
/// fused), else its buffer.
fn side(ctx: &Context, array: &Array<f32>) -> Outcome<MatmulSide> {
    match array.pending() {
        Some(node) => {
            if node.len() as u64 * 4 > ctx.segment_bytes() {
                return Err(LumaError::Chunked("a matrix product of an array split over several buffers").into());
            }
            Ok(MatmulSide::Pending(Arc::clone(node)))
        }
        None => {
            let (pin, bytes) = array.ready()?;
            Ok(MatmulSide::Ready(pin, bytes))
        }
    }
}

/// The product on the host if it is small and its operands are there (before anything pins an
/// operand to the device).
fn host_product(ctx: &Context, lhs: &Array<f32>, rhs: &Array<f32>) -> Option<Array<f32>> {
    let (ld, rd) = (lhs.shape().dims(), rhs.shape().dims());
    if !ctx.can_run_on_host() || !ctx.same_as(rhs.context()) || ld[0] != rd[1] || ld[2..].iter().chain(&rd[2..]).any(|&extent| extent != 1) {
        return None;
    }
    let (k, m, n) = (ld[0], ld[1], rd[0]);
    let limit = crate::host::MAX_HOST_ELEMS as u64;
    if (m as u64) * (n as u64) * (k as u64) > crate::host::MAX_MATMUL_FMAS || (m as u64) * (k as u64) > limit || (k as u64) * (n as u64) > limit || (m as u64) * (n as u64) > limit {
        return None;
    }
    crate::host::try_matmul(ctx, lhs, rhs, m, n, k)
}

fn run(lhs: &Array<f32>, rhs: &Array<f32>) -> Outcome<Array<f32>> {
    let ctx = lhs.context();
    if let Some(out) = host_product(ctx, lhs, rhs) {
        return Ok(out);
    }
    // Operands first, so that a poisoned one reports its own error.
    let cores = ctx.tuning().native_gemm;
    let (sides, transposed) = if cores > 0 { native_sides(ctx, lhs, rhs)? } else { ([side(ctx, lhs)?, side(rhs.context(), rhs)?], [false; 2]) };
    if !ctx.same_as(rhs.context()) {
        return Err(LumaError::ContextMismatch.into());
    }
    let (ld, rd) = (lhs.shape().dims(), rhs.shape().dims());
    if ld[0] != rd[1] || ld[2..].iter().chain(&rd[2..]).any(|&extent| extent != 1) {
        return Err(LumaError::ShapeMismatch { lhs: ld, rhs: rd }.into());
    }
    let (k, m, n) = (ld[0], ld[1], rd[0]);
    let too_large = |extent: usize| u32::try_from(extent).map_err(|_| Failure::from(LumaError::DimensionTooLarge { shape: [n, m, 1, 1] }));
    let (m32, n32, k32) = (too_large(m)?, too_large(n)?, too_large(k)?);
    let mut plan = if cores > 0 {
        native_plan(m32, n32, k32, cores, transposed, ctx.max_workgroups_per_dimension(), ctx.segment_bytes())?
    } else {
        let along_k = jit::stored_along_k(&sides[0], [k32, m32]);
        plan(ctx.tuning(), m32, n32, k32, along_k, ctx.max_workgroups_per_dimension(), ctx.segment_bytes())?
    };
    // A skinny kernel computes a fused operand once per tile that reads it. A few reads of an
    // expression of several buffers already cost more than the staged tiles, which compute each
    // element twice (a correlation matrix, `(x - mean) / std` against itself: 1.2 ms against 1.7);
    // past `SKINNY_EVALUATE_READS` storing the expression first is cheaper.
    let mut sides = sides;
    if let MatmulTile::Skinny { rows, cols, .. } = plan.tile {
        let reads = [n32.div_ceil(u32::from(cols)), m32.div_ceil(u32::from(rows))];
        let costly = [lhs, rhs].into_iter().zip(reads).any(|(array, reads)| (SKINNY_FUSED_READS..SKINNY_EVALUATE_READS).contains(&reads) && array.pending().is_some_and(jit::is_recomputed));
        if costly {
            plan = tiled_plan(ctx.tuning(), m32, n32, k32, ctx.max_workgroups_per_dimension(), ctx.segment_bytes())?;
        }
    }
    if let MatmulTile::Skinny { rows, cols, .. } = plan.tile {
        let reads = [n32.div_ceil(u32::from(cols)), m32.div_ceil(u32::from(rows))];
        let mut evaluated = false;
        for (array, reads) in [lhs, rhs].into_iter().zip(reads) {
            if reads >= SKINNY_EVALUATE_READS && let Some(node) = array.pending() {
                evaluated |= jit::evaluate_for_reuse(node)?;
            }
        }
        if evaluated {
            sides = [side(ctx, lhs)?, side(rhs.context(), rhs)?];
        }
    }

    let out = Array::<f32>::alloc(ctx, [n, m, 1, 1].into())?;
    let out_storage = Arc::clone(out.storage()?);
    let out_pin = out_storage.pin_single()?;
    let out_bytes = out_storage.byte_len();
    let mut scratch = Scratch::new(ctx);
    let partial = if plan.slices > 1 { Some(scratch.take(u64::from(plan.slices) * out_bytes)?) } else { None };
    let (target, target_bytes) = match &partial {
        Some(buffer) => (buffer, u64::from(plan.slices) * out_bytes),
        None => (out_pin.buffer(), out_bytes),
    };
    let kernel = jit::MatmulKernel { tile: plan.tile, aligned: plan.aligned, split: plan.split };
    jit::matmul(ctx, sides, kernel, [m32, n32, k32, plan.kc], plan.groups, RawArg::write(target, target_bytes))?;
    if let Some(partial) = &partial {
        let sum = ctx.builtin(Builtin::MatmulSum, "f32")?;
        let slices = [plan.slices];
        let sum_groups = crate::ops::fold_workgroups(ctx, (m32 * n32).div_ceil(32))?;
        ctx.launch(
            &sum,
            sum_groups,
            [m32 * n32, 1, 1],
            &[RawArg::write(out_pin.buffer(), out_bytes), RawArg::read(partial, target_bytes), RawArg::Uniform(bytemuck::bytes_of(&slices))],
        )?;
    }
    drop(scratch);
    jit::check_finite(ctx, &out_storage, jit::ElemTy::F32)?;
    Ok(out)
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
        body(tile, aligned, split, MatmulOperand::Buffer, MatmulOperand::Buffer)
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
pub(crate) fn body(tile: MatmulTile, aligned: bool, split: bool, lhs: MatmulOperand, rhs: MatmulOperand) -> String {
    let loads = Loads {
        lhs: Load { fused: lhs.fused(), lanes: if lhs == MatmulOperand::Transposed { Lanes::AlongRows } else { Lanes::AlongK } },
        rhs: Load { fused: rhs.fused(), lanes: if rhs == MatmulOperand::Transposed { Lanes::AlongK } else { Lanes::AlongRows } },
    };
    match tile {
        MatmulTile::T128 => t128(aligned, split, &loads),
        MatmulTile::T128x32 => t128x32(aligned, split, &loads),
        MatmulTile::T32x128 => t32x128(aligned, split, &loads),
        MatmulTile::T32 | MatmulTile::Native { .. } => t32(aligned, split, &loads),
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
}

impl Loads {
    /// The expression loading element `index` (of the flat matrix) of the left operand.
    fn lhs(&self, index: &str) -> String {
        if self.lhs.fused { format!("luma_lhs({index})") } else { format!("a[{index}]") }
    }

    fn rhs(&self, index: &str) -> String {
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

fn t32(aligned: bool, split: bool, loads: &Loads) -> String {
    let load_a = t32_load(true, aligned, loads.lhs.lanes, &|index| loads.lhs(index));
    let load_b = t32_load(false, aligned, loads.rhs.lanes, &|index| loads.rhs(index));
    let Range { declare, k_begin, k_end, out } = range(aligned, split);
    let end = if aligned { "" } else { ", ke" };
    // Rows `lid.y` and `lid.y + 16`, columns `lid.x` and `lid.x + 16` of the tile.
    let store = |acc: &str, row: &str, col: &str| {
        if aligned {
            format!("{{ let r = row0 + {row}; let col = col0 + {col}; c[{out}r * p.n + col] = {acc}; }}\n    ")
        } else {
            format!("{{ let r = row0 + {row}; let col = col0 + {col}; if (r < p.m && col < p.n) {{ c[{out}r * p.n + col] = {acc}; }} }}\n    ")
        }
    };
    let stores = [("c00", "lid.y", "lid.x"), ("c01", "lid.y", "lid.x + 16u"), ("c10", "lid.y + 16u", "lid.x"), ("c11", "lid.y + 16u", "lid.x + 16u")]
        .iter()
        .map(|&(acc, row, col)| store(acc, row, col))
        .collect::<String>();
    let stage = loads.stage(end);
    format!(
        "var<workgroup> at: array<f32, 512>;
var<workgroup> bt: array<f32, 512>;

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
    let row0 = (tile / columns) * 32u;
    let col0 = (tile % columns) * 32u;
    let t = lid.y * 16u + lid.x;
{declare}    var c00 = 0.0;
    var c01 = 0.0;
    var c10 = 0.0;
    var c11 = 0.0;
    for (var k0 = {k_begin}; k0 < {k_end}; k0 = k0 + 16u) {{
        {stage}
        workgroupBarrier();
        for (var kk = 0u; kk < 16u; kk = kk + 1u) {{
            let o = kk * 32u;
            let a0 = at[o + lid.y];
            let a1 = at[o + lid.y + 16u];
            let b0 = bt[o + lid.x];
            let b1 = bt[o + lid.x + 16u];
            c00 = fma(a0, b0, c00);
            c01 = fma(a0, b1, c01);
            c10 = fma(a1, b0, c10);
            c11 = fma(a1, b1, c11);
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_long_product_of_few_tiles_is_split_in_slices_of_whole_k_steps() {
        let t = Tuning::for_tests();
        // The Gram matrix of 64 features over 2^20 samples: 4 tiles of 32x32.
        let (slices, kc) = split_of(&t, 4, 1 << 20).unwrap();
        assert_eq!((slices, kc), (256, 4096));
        for k in [4096u32, 5000, 100_000, 1 << 20] {
            for tiles in [1u32, 4, 70, 127] {
                let Some((slices, kc)) = split_of(&t, tiles, k) else { continue };
                assert!(slices >= 2 && kc % SLICE == 0, "{tiles} tiles, k {k}");
                assert_eq!(slices, k.div_ceil(kc), "no empty slice");
            }
        }
        assert_eq!(split_of(&t, 128, 1 << 20), None);
        assert_eq!(split_of(&t, 4, 4095), None);
        assert_eq!(split_of(&t, 4, 4096), Some((4, 1024)));
    }

    #[test]
    fn the_split_aims_for_a_device_full_of_workgroups() {
        // The reference device (82 multiprocessors) keeps the constants measured on it.
        let reference = Tuning { compute_units: 82, ..Tuning::for_tests() };
        assert_eq!(split_of(&reference, 4, 1 << 20), Some((256, 4096)));
        assert_eq!(split_of(&reference, 127, 1 << 20), Some((9, 116_512)));
        assert_eq!(split_of(&reference, 128, 1 << 20), None);
        // A smaller device keeps them (measured on an RTX 3080 Laptop, 48 multiprocessors).
        let laptop = Tuning { compute_units: 48, ..Tuning::for_tests() };
        assert_eq!(split_of(&laptop, 4, 1 << 20), Some((256, 4096)));
        assert_eq!(split_of(&laptop, 128, 1 << 20), None);
        // A bigger one more.
        let big = Tuning { compute_units: 128, ..Tuning::for_tests() };
        assert_eq!(split_of(&big, 4, 1 << 20).map(|s| s.0), Some(400));
        assert!(split_of(&big, 199, 1 << 20).is_some());
    }

    #[test]
    fn the_plan_picks_the_tile_alignment_and_grid() {
        let t = Tuning::for_tests();
        let p = plan(&t, 1024, 2048, 1024, false, 65535, u64::MAX).unwrap();
        assert_eq!((p.tile, p.aligned, p.slices, p.groups), (MatmulTile::T128, true, 1, [16, 8, 1]));
        let p = plan(&t, 1000, 1000, 1000, false, 65535, u64::MAX).unwrap();
        assert_eq!((p.tile, p.aligned, p.groups), (MatmulTile::T128, false, [8, 8, 1]));
        let p = plan(&t, 64, 64, 1024, false, 65535, u64::MAX).unwrap();
        assert_eq!((p.tile, p.aligned, p.slices, p.groups), (MatmulTile::T32, true, 1, [2, 2, 1]));
        // Tall products stay on the rectangular tile; the grid folds once row tiles exceed it.
        let p = plan(&t, 1 << 22, 8, 16, false, 65535, u64::MAX).unwrap();
        assert_eq!((p.tile, p.groups), (MatmulTile::T128x32, [1, 1 << 15, 1]));
        let p = plan(&t, 8, 1 << 22, 16, false, 65535, u64::MAX).unwrap();
        assert_eq!((p.tile, p.groups), (MatmulTile::T32x128, [1 << 15, 1, 1]));
        let p = plan(&t, 1 << 22, 1 << 6, 16, false, 65535, u64::MAX).unwrap();
        assert_eq!(p.tile, MatmulTile::T128x32);
        let p = plan(&t, 1 << 22, 8, 16, false, 1 << 12, u64::MAX).unwrap();
        assert!(p.groups[0] * p.groups[1] >= 1 << 15);
        assert!(plan(&t, 1 << 22, 64, 16, false, 65535, u64::MAX).is_ok());
    }

    #[test]
    fn a_small_output_over_a_long_k_is_a_skinny_product() {
        let skinny = |m, n, k, along_k, max_bytes| plan(&Tuning::for_tests(), m, n, k, along_k, 65535, max_bytes).unwrap();
        // One-hot labels times 32 features: 4 tiles of 10x8, 16 groups of them in a workgroup.
        let p = skinny(10, 32, 1_000_000, false, u64::MAX);
        assert_eq!(p.tile, MatmulTile::Skinny { rows: 10, cols: 8, groups: 16, width: 64, along_k: false });
        assert!(p.split && !p.aligned && p.groups[1] == 1 && p.groups[0] == p.slices && p.slices * p.kc >= 1_000_000);
        // A Gram matrix of 64 features: 64 tiles of 8x8, one group.
        let p = skinny(64, 64, 1 << 20, false, u64::MAX);
        assert_eq!(p.tile, MatmulTile::Skinny { rows: 8, cols: 8, groups: 1, width: 64, along_k: false });
        // A dot product: one tile, every invocation of the workgroup a group.
        let p = skinny(1, 1, 1 << 20, true, u64::MAX);
        assert_eq!(p.tile, MatmulTile::Skinny { rows: 1, cols: 1, groups: 64, width: 64, along_k: true });
        // Slices are whole `k` and none is empty.
        for (m, n, k) in [(10, 32, 100_003), (1, 1, 2048), (64, 64, 70_000), (33, 7, 1 << 20), (2000, 1, 70_001)] {
            for along_k in [false, true] {
                let p = skinny(m, n, k, along_k, u64::MAX);
                assert!(matches!(p.tile, MatmulTile::Skinny { .. }), "{m} {n} {k}");
                assert_eq!(p.slices, k.div_ceil(p.kc), "no empty slice");
                assert!(p.slices >= 1 && p.groups[0] == p.slices);
            }
        }
        // The partial sums must fit a buffer.
        assert_eq!(skinny(64, 64, 1 << 20, false, 1000).slices, 1);
        // A matrix times a vector reads the matrix along `K`: its row tiles are workgroups.
        let p = skinny(8192, 1, 8192, true, u64::MAX);
        assert!(matches!(p.tile, MatmulTile::Skinny { along_k: true, .. }));
        assert!(p.groups[1] > 1);
        // Not skinny: too short, or too many outputs.
        assert!(!matches!(skinny(10, 32, 1000, false, u64::MAX).tile, MatmulTile::Skinny { .. }));
        assert!(!matches!(skinny(256, 256, 1 << 20, false, u64::MAX).tile, MatmulTile::Skinny { .. }));
        assert!(!matches!(skinny(64, 64, 16384, false, u64::MAX).tile, MatmulTile::Skinny { .. }));
        assert!(!matches!(skinny(128, 128, 1 << 20, false, u64::MAX).tile, MatmulTile::Skinny { .. }));
    }

    fn native(edge: u32) -> MatmulTile {
        MatmulTile::Native { edge, lhs_t: false, rhs_t: false }
    }

    #[test]
    fn a_native_plan_keeps_the_cores_busy() {
        let plan = |m, n, k, cores| native_plan(m, n, k, cores, [false; 2], 65535, u64::MAX).unwrap();
        // The largest tile that leaves two tiles per core.
        let p = plan(4096, 4096, 4096, 32);
        assert_eq!((p.tile, p.slices, p.groups), (native(128), 1, [32, 32, 1]));
        let p = plan(512, 512, 512, 32);
        assert_eq!((p.tile, p.groups), (native(64), [8, 8, 1]));
        let p = plan(100, 100, 100, 32);
        assert_eq!((p.tile, p.slices, p.groups), (native(32), 1, [4, 4, 1]));
        // Few tiles and a long K: sliced, every slice a multiple of 16 that covers K.
        let p = plan(64, 64, 4099, 32);
        assert_eq!((p.tile, p.groups[..2].to_vec()), (native(32), vec![2, 2]));
        assert!(p.slices >= 2 && p.kc % SLICE == 0 && p.slices * p.kc >= 4099 && (p.slices - 1) * p.kc < 4099, "{p:?}");
        assert_eq!(plan(64, 64, 1 << 20, 1).slices, 1);
        // The partial sums must fit a buffer.
        assert_eq!(native_plan(64, 64, 1 << 20, 32, [false; 2], 65535, 1000).unwrap().slices, 1);
        // The transposed operands are the tile's.
        let p = native_plan(512, 512, 512, 32, [true, false], 65535, u64::MAX).unwrap();
        assert_eq!(p.tile, MatmulTile::Native { edge: 64, lhs_t: true, rhs_t: false });
        // Tall products fold over the grid.
        let p = native_plan(1 << 22, 8, 16, 32, [false; 2], 1 << 12, u64::MAX).unwrap();
        assert!(p.groups[0] <= 1 << 12 && p.groups[1] <= 1 << 12 && p.groups[0] * p.groups[1] >= 1 << 15, "{p:?}");
    }
}
