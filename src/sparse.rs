//! Sparse matrices: [`Sparse`] in CSR, CSC and COO storage, with conversions, products with dense
//! matrices and vectors, and elementwise operations with dense arrays and scalars.
//!
//! A matrix of `rows x cols` is stored as arrays of `u32` indices and `f32` values (one per
//! stored entry) on the device. Dense matrices follow the rest of luma: a dense `rows x cols`
//! matrix is an [`Array`] of shape `[cols, rows]` (axis 0, the fastest, is the column index), so
//! the dense form of a sparse matrix, [`Sparse::to_dense`], has that shape and a dense right-hand
//! side of `c` columns has shape `[c, cols]`.
//!
//! * **CSR** (compressed sparse row): `row_ptr` of `rows + 1` offsets (entry `k` of row `r`
//!   lies in `row_ptr[r]..row_ptr[r + 1]`), the column and the value of every entry in row
//!   order.
//! * **CSC** (compressed sparse column): `col_ptr` of `cols + 1` offsets, the row and the value
//!   of every entry in column order.
//! * **COO** (coordinate): the row, the column and the value of every entry, in any order.
//!
//! Duplicate entries (the same row and column stored twice, which COO allows and conversions
//! keep) add up: they count as one entry holding their sum in [`Sparse::to_dense`], in products
//! and in elementwise operations with dense operands, while [`Sparse::nnz`] counts them
//! separately. Indices must be in range (column below `cols`, row below `rows`); out-of-range
//! indices give unspecified results, never reads or writes outside the arrays.
//!
//! luma has no arrays without elements, so a matrix without stored entries has no index or
//! value arrays: [`Sparse::values`] and the other accessors return `None` for it
//! ([`Sparse::empty`]; [`Sparse::from_dense`] of an all-zero matrix gives one).
//!
//! # Products
//!
//! [`Sparse::matmul`] computes `A x` and [`Sparse::matmul_t`] `A^T x` for a vector or a dense
//! matrix `x`. A product whose output runs along the compressed axis (CSR `A x`, CSC `A^T x`) runs
//! on the row kernels: for a vector with rows of at most 64 entries a thread, or a few lanes
//! combining in workgroup memory, per row; for a vector with longer rows a kernel that takes
//! tiles of consecutive entries whatever the row lengths are (the balance of a merge-path
//! product, with a carry pass for the rows that cross tiles); for `c` dense columns one invocation per output element walks its row (the `c`
//! invocations of a row read `c` adjacent elements of `x`, so the gather is coalesced) and rows of
//! more than 256 entries are cut in pieces of 2048 that a workgroup each sums, the pieces added in
//! order. None of them uses atomics: the sums are the same on every run. The first product of
//! any other combination scatters the products with an atomic add by output index
//! ([`Array::group_sum`]), whose rounding depends on the order in which the atomics land; the
//! second converts the matrix once to the form that runs the first way and keeps that copy (see
//! [`Sparse::matmul`]). The first product of a compressed matrix reads its row lengths back (one
//! synchronization, kept with the matrix).
//!
//! # Errors
//!
//! As everywhere in luma, a failure poisons the result: a bad constructor argument gives a
//! [`Sparse`] whose [`Sparse::error`] is set and which poisons everything computed from it; a
//! dense result is a poisoned array.

use std::fmt;
use std::ops::{Div, Mul, Neg};
use std::panic::Location;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, OnceLock};

use crate::array::Array;
use crate::context::{default_context, Context};
use crate::error::{LumaError, Result};
use crate::scatter::AtomicElem;
use crate::search::Side;
use crate::shape::Shape;
use crate::view::concat;

/// Rows longer than this are not walked by the row kernels (their lanes would idle behind the one
/// that holds the longest row): a workgroup of its own reduces each, see [`LONG_ROWS`].
const LONG_ROW: u32 = 256;
/// Entries of a long row one workgroup of [`LONG_ROWS`] sums (a long row is cut into pieces of this
/// size, whose partial sums [`LONG_SUMS`] adds up in order).
const CHUNK: u32 = 2048;
/// Invocations of a workgroup of every row kernel.
const WORKGROUP: u32 = 128;
/// Workgroups along x of the grids of the row kernels (the dispatch limit of every backend).
const GRID_X: u32 = 65_535;

#[derive(encase::ShaderType)]
struct RowParams {
    /// `log2` of the lanes per row of the SpMV kernel; the columns `c` of the SpMM kernels.
    shape: u32,
    /// Output elements (SpMM) or rows (SpMV); long rows: how many.
    total: u32,
    cap: u32,
    /// Pieces per long row.
    chunks: u32,
}

/// `y[r]` of the rows of at most `cap` entries, `V = 2^shape` lanes per row: lane `l` sums the
/// entries `l, l + V, ...` of its row (the `V` lanes read consecutive entries), the lanes of a row
/// combine in workgroup memory. Long rows are left to [`LONG_ROWS`].
const SPMV_ROWS: &str = r#"
struct Params { shape: u32, total: u32, cap: u32, chunks: u32 }
@group(1) @binding(0) var<storage, read> ptr: array<u32>;
@group(1) @binding(1) var<storage, read> minor: array<u32>;
@group(1) @binding(2) var<storage, read> val: array<f32>;
@group(1) @binding(3) var<storage, read> x: array<f32>;
@group(1) @binding(4) var<storage, read_write> y: array<f32>;
@group(1) @binding(5) var<uniform> p: Params;
var<workgroup> sh: array<f32, 128>;

@compute @workgroup_size(128)
fn main(@builtin(local_invocation_index) li: u32, @builtin(workgroup_id) wg: vec3<u32>) {
    let lanes = 1u << p.shape;
    let row = ((wg.y * 65535u + wg.x) * 128u + li) >> p.shape;
    let lane = li & (lanes - 1u);
    var acc = 0.0;
    var mine = false;
    if (row < p.total) {
        let b = ptr[row];
        let e = ptr[row + 1u];
        if (e - b <= p.cap) {
            mine = true;
            var k = b + lane;
            loop {
                if (k + lanes >= e) { break; }
                let k2 = k + lanes;
                let c0 = minor[k];
                let c1 = minor[k2];
                let x0 = x[c0];
                let x1 = x[c1];
                acc = acc + val[k] * x0;
                acc = acc + val[k2] * x1;
                k = k2 + lanes;
            }
            loop {
                if (k >= e) { break; }
                acc = acc + val[k] * x[minor[k]];
                k += lanes;
            }
        }
    }
    sh[li] = acc;
    workgroupBarrier();
    var s = lanes >> 1u;
    loop {
        if (s == 0u) { break; }
        if (lane < s) { sh[li] = sh[li] + sh[li + s]; }
        workgroupBarrier();
        s = s >> 1u;
    }
    if (mine && lane == 0u) { y[row] = sh[li]; }
}
"#;

/// Invocations of a workgroup of the tile kernel.
const MERGE_WG: u32 = 128;
/// Entries an invocation of the tile kernel sums: one `vec4` of `minor` and of `val`.
const MERGE_PER: u32 = 4;
/// Entries of a tile.
const MERGE_TILE: u32 = MERGE_WG * MERGE_PER;

/// The tile kernel of a vector product over a matrix with rows longer than [`SHORT_ROWS`]: a
/// workgroup takes a tile of [`MERGE_TILE`] consecutive entries, whatever the lengths of the rows
/// it cuts. The balance by entries is that of the merge-path product of Merrill and Garland,
/// "Merge-based parallel sparse matrix-vector multiplication" (SC'16), of which this is a
/// simplification: the tiles are cut at multiples of the tile size and the rows that start in a
/// tile come from `part` (the first row that starts at or after the first entry of every tile,
/// searched once per matrix), where the merge path cuts the sum of rows and entries by a binary
/// search per block; empty rows are written by the invocation that finds them, and the part of a
/// row that crosses a tile is completed by [`merge_carry_source`].
///
/// Every invocation loads its own run of four consecutive entries as one `vec4` of `minor` and of
/// `val` (the same buffers bound a second time as `vec4`s: the run is one aligned 16 byte load
/// that a warp covers in a few sectors, where a coalesced scalar load would have to be
/// transposed through workgroup memory, which measured 8 to 10 % slower on an RTX 3090 and an
/// RTX 3080 Laptop), multiplies them by `x`, and sums the run cutting it at the rows that start in
/// it. The row starts of the tile are scattered by one invocation per row to `hrow`, one word
/// per entry slot (the row plus one at the first entry of a row). The part of a row that
/// continues into the following invocations is added by the invocation that holds its start,
/// which walks over their sums in order (a row longer than a few tiles is the only case where the
/// walk is long); what a tile contributes to the rows that cross its ends goes to `sc` (the part
/// before the first row start; the part of the last row and its number plus one). Every row is
/// written by exactly one invocation and every sum has a fixed order, so the result is the same
/// on every run.
///
/// The kernel is trusted (no bounds checks): whatever `ptr`, `minor` and `part` hold, every
/// access is clamped or guarded to be in bounds (a wrong row pointer or column gives an
/// unspecified result).
fn merge_source() -> String {
    let (w, tile) = (MERGE_WG, MERGE_TILE);
    let padded = tile + tile / MERGE_PER;
    format!(
        "struct Params {{ n: u32, rows: u32, nb: u32, nx: u32 }}
@group(1) @binding(0) var<storage, read> ptr: array<u32>;
@group(1) @binding(1) var<storage, read> minor: array<u32>;
@group(1) @binding(2) var<storage, read> val: array<f32>;
@group(1) @binding(3) var<storage, read> x: array<f32>;
@group(1) @binding(4) var<storage, read> part: array<u32>;
@group(1) @binding(5) var<storage, read_write> y: array<f32>;
@group(1) @binding(6) var<storage, read_write> sc: array<u32>;
@group(1) @binding(7) var<uniform> p: Params;
@group(1) @binding(8) var<storage, read> minor4: array<vec4<u32>>;
@group(1) @binding(9) var<storage, read> val4: array<vec4<f32>>;
fn pad(j: u32) -> u32 {{ return j + j / 4u; }}
var<workgroup> hrow: array<u32, {padded}>;
var<workgroup> sv: array<f32, {w}>;
var<workgroup> sf: array<u32, {w}>;
@compute @workgroup_size({w})
fn main(@builtin(local_invocation_index) t: u32, @builtin(workgroup_id) wg: vec3<u32>) {{
    let tile = wg.y * 65535u + wg.x;
    if (tile >= p.nb) {{ return; }}
    let e0 = tile * {tile}u;
    let e1 = min(e0 + {tile}u, p.n);
    let last = tile + 1u == p.nb;
    let c0 = e0 + t * 4u;
    var k = vec4<u32>(0u);
    var v = vec4<f32>(0.0);
    if (c0 + 4u <= e1) {{
        k = minor4[c0 / 4u];
        v = val4[c0 / 4u];
    }} else {{
        if (c0 < e1) {{ k.x = minor[c0]; v.x = val[c0]; }}
        if (c0 + 1u < e1) {{ k.y = minor[c0 + 1u]; v.y = val[c0 + 1u]; }}
        if (c0 + 2u < e1) {{ k.z = minor[c0 + 2u]; v.z = val[c0 + 2u]; }}
        if (c0 + 3u < e1) {{ k.w = minor[c0 + 3u]; v.w = val[c0 + 3u]; }}
    }}
    let g = vec4<f32>(
        select(0.0, v.x * x[min(k.x, p.nx)], c0 < e1),
        select(0.0, v.y * x[min(k.y, p.nx)], c0 + 1u < e1),
        select(0.0, v.z * x[min(k.z, p.nx)], c0 + 2u < e1),
        select(0.0, v.w * x[min(k.w, p.nx)], c0 + 3u < e1));
    let r0 = part[tile];
    let r1 = min(select(part[tile + 1u], p.rows, last), p.rows);
    for (var j = 0u; j < 4u; j += 1u) {{ hrow[pad(t + j * {w}u)] = 0u; }}
    workgroupBarrier();
    for (var r = r0 + t; r < r1; r += {w}u) {{
        let s = ptr[r];
        if (ptr[r + 1u] > s) {{
            if (s >= e0 && s < e1) {{ hrow[pad(s - e0)] = r + 1u; }}
        }} else {{
            y[r] = 0.0;
        }}
    }}
    workgroupBarrier();
    var run = 0.0;
    var runrow = 0u;
    var has = false;
    var lead = 0.0;
    for (var i = 0u; i < 4u; i += 1u) {{
        let hr = hrow[pad(t * 4u + i)];
        if (hr != 0u) {{
            if (has) {{ y[runrow - 1u] = run; }} else {{ lead = run; }}
            has = true;
            run = g[i];
            runrow = hr;
        }} else {{
            run = run + g[i];
        }}
    }}
    if (!has) {{ lead = run; }}
    sv[t] = lead;
    sf[t] = select(0u, 1u, has);
    workgroupBarrier();
    if (t == 0u) {{
        var acc = 0.0;
        var u = 0u;
        loop {{
            if (u >= {w}u) {{ break; }}
            acc = acc + sv[u];
            if (sf[u] != 0u) {{ break; }}
            u += 1u;
        }}
        sc[tile * 3u] = bitcast<u32>(acc);
        if (u >= {w}u) {{ sc[tile * 3u + 2u] = 0u; }}
    }}
    if (has) {{
        var total = run;
        var u = t + 1u;
        loop {{
            if (u >= {w}u) {{ break; }}
            total = total + sv[u];
            if (sf[u] != 0u) {{ break; }}
            u += 1u;
        }}
        let row = runrow - 1u;
        if (u >= {w}u && ptr[row + 1u] > e1) {{
            sc[tile * 3u + 1u] = bitcast<u32>(total);
            sc[tile * 3u + 2u] = runrow;
        }} else {{
            y[row] = total;
            if (u >= {w}u) {{ sc[tile * 3u + 2u] = 0u; }}
        }}
    }}
}}
"
    )
}

/// The rows that continue past their tile of [`merge_source`]: one invocation per tile whose last
/// row does, adding the carries of the tiles the row covers, in order, to the tile's own part.
fn merge_carry_source() -> String {
    let tile = MERGE_TILE;
    format!(
        "struct Params {{ n: u32, rows: u32, nb: u32, nx: u32 }}
@group(1) @binding(0) var<storage, read_write> y: array<f32>;
@group(1) @binding(1) var<storage, read> sc: array<u32>;
@group(1) @binding(2) var<storage, read> ptr: array<u32>;
@group(1) @binding(3) var<uniform> p: Params;
@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {{
    let k = gid.x + gid.y * nwg.x * 128u;
    if (k >= p.nb) {{ return; }}
    let crossing = sc[k * 3u + 2u];
    if (crossing == 0u) {{ return; }}
    let row = crossing - 1u;
    let hi = min((ptr[row + 1u] - 1u) / {tile}u + 1u, p.nb);
    var a0 = 0.0;
    var a1 = 0.0;
    var a2 = 0.0;
    var a3 = 0.0;
    var j = k + 1u;
    for (; j + 3u < hi; j += 4u) {{
        a0 = a0 + bitcast<f32>(sc[j * 3u]);
        a1 = a1 + bitcast<f32>(sc[(j + 1u) * 3u]);
        a2 = a2 + bitcast<f32>(sc[(j + 2u) * 3u]);
        a3 = a3 + bitcast<f32>(sc[(j + 3u) * 3u]);
    }}
    for (; j < hi; j += 1u) {{
        a0 = a0 + bitcast<f32>(sc[j * 3u]);
    }}
    y[row] = bitcast<f32>(sc[k * 3u + 1u]) + ((a0 + a1) + (a2 + a3));
}}
"
    )
}

/// `Y[j, r] = sum_k val[k] * X[j, minor[k]]` over the entries `k` of row `r` (at most `cap` of
/// them), one invocation per output element, entries in order, four loads in flight.
const SPMM_ROWS: &str = r#"
struct Params { shape: u32, total: u32, cap: u32, chunks: u32 }
@group(1) @binding(0) var<storage, read> ptr: array<u32>;
@group(1) @binding(1) var<storage, read> minor: array<u32>;
@group(1) @binding(2) var<storage, read> val: array<f32>;
@group(1) @binding(3) var<storage, read> x: array<f32>;
@group(1) @binding(4) var<storage, read_write> y: array<f32>;
@group(1) @binding(5) var<uniform> p: Params;

@compute @workgroup_size(128)
fn main(@builtin(local_invocation_index) li: u32, @builtin(workgroup_id) wg: vec3<u32>) {
    let t = (wg.y * 65535u + wg.x) * 128u + li;
    if (t >= p.total) { return; }
    let c = p.shape;
    let r = t / c;
    let j = t - r * c;
    var k = ptr[r];
    let e = ptr[r + 1u];
    if (e - k > p.cap) { return; }
    var acc = 0.0;
    loop {
        if (k + 4u > e) { break; }
        let c0 = minor[k];
        let c1 = minor[k + 1u];
        let c2 = minor[k + 2u];
        let c3 = minor[k + 3u];
        let x0 = x[c0 * c + j];
        let x1 = x[c1 * c + j];
        let x2 = x[c2 * c + j];
        let x3 = x[c3 * c + j];
        acc = acc + val[k] * x0;
        acc = acc + val[k + 1u] * x1;
        acc = acc + val[k + 2u] * x2;
        acc = acc + val[k + 3u] * x3;
        k += 4u;
    }
    loop {
        if (k >= e) { break; }
        acc = acc + val[k] * x[minor[k] * c + j];
        k += 1u;
    }
    y[t] = acc;
}
"#;

/// The pieces of the rows longer than `cap`: workgroup `g` is piece `g / c` of the long rows (found
/// by a binary search in `piece_ptr`, the offsets of the pieces of every long row) for the column
/// `g % c`: the 128 invocations stride over the piece (`chunks` entries at most) and combine in
/// workgroup memory into `part[g]`. `shape` is the columns, `total` the long rows; exactly
/// pieces x columns workgroups run.
const LONG_ROWS: &str = r#"
struct Params { shape: u32, total: u32, cap: u32, chunks: u32 }
@group(1) @binding(0) var<storage, read> ptr: array<u32>;
@group(1) @binding(1) var<storage, read> minor: array<u32>;
@group(1) @binding(2) var<storage, read> val: array<f32>;
@group(1) @binding(3) var<storage, read> x: array<f32>;
@group(1) @binding(4) var<storage, read_write> part: array<f32>;
@group(1) @binding(5) var<storage, read> rows: array<u32>;
@group(1) @binding(6) var<storage, read> piece_ptr: array<u32>;
@group(1) @binding(7) var<uniform> p: Params;
var<workgroup> sh: array<f32, 128>;

@compute @workgroup_size(128)
fn main(@builtin(local_invocation_index) li: u32, @builtin(workgroup_id) wg: vec3<u32>) {
    let g = wg.y * 65535u + wg.x;
    let c = p.shape;
    let piece = g / c;
    let j = g - piece * c;
    // The last long row whose first piece is at or before this one.
    var lo = 0u;
    var hi = p.total;
    loop {
        if (hi - lo <= 1u) { break; }
        let mid = (lo + hi) / 2u;
        if (piece_ptr[mid] <= piece) { lo = mid; } else { hi = mid; }
    }
    let row = rows[lo];
    let b = ptr[row] + (piece - piece_ptr[lo]) * p.chunks;
    let e = min(ptr[row + 1u], b + p.chunks);
    var acc = 0.0;
    var k = b + li;
    loop {
        if (k >= e) { break; }
        acc = acc + val[k] * x[minor[k] * c + j];
        k += 128u;
    }
    sh[li] = acc;
    workgroupBarrier();
    var s = 64u;
    loop {
        if (s == 0u) { break; }
        if (li < s) { sh[li] = sh[li] + sh[li + s]; }
        workgroupBarrier();
        s = s >> 1u;
    }
    if (li == 0u) { part[g] = sh[0]; }
}
"#;

/// The sums of the pieces of every (long row, column), in order, into `y`: `total` is rows x columns.
const LONG_SUMS: &str = r#"
struct Params { shape: u32, total: u32, cap: u32, chunks: u32 }
@group(1) @binding(0) var<storage, read> part: array<f32>;
@group(1) @binding(1) var<storage, read> rows: array<u32>;
@group(1) @binding(2) var<storage, read> piece_ptr: array<u32>;
@group(1) @binding(3) var<storage, read_write> y: array<f32>;
@group(1) @binding(4) var<uniform> p: Params;

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let t = gid.x;
    if (t >= p.total) { return; }
    let c = p.shape;
    let i = t / c;
    let j = t - i * c;
    var acc = 0.0;
    for (var q = piece_ptr[i]; q < piece_ptr[i + 1u]; q += 1u) {
        acc = acc + part[q * c + j];
    }
    y[rows[i] * c + j] = acc;
}
"#;

/// How the entries of a [`Sparse`] matrix are stored.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SparseFormat {
    /// Compressed sparse row: a row pointer, and the column and value of every entry in row order.
    Csr,
    /// Compressed sparse column: a column pointer, and the row and value of every entry in column
    /// order.
    Csc,
    /// Coordinate list: the row, column and value of every entry, in any order.
    Coo,
}

/// Element types of [`Sparse`]: `f32`.
pub trait SparseElem: AtomicElem {}

impl SparseElem for f32 {}

/// A sparse `rows x cols` matrix of `f32` in CSR, CSC or COO storage. See the guide chapter
/// "Sparse matrices" for the layouts, the conventions and the errors.
///
/// ```no_run
/// use luma::{Array, Sparse, SparseFormat};
///
/// // [[1, 0, 2], [0, 0, 3]] as CSR, times [1, 10, 100].
/// let a = Sparse::from_csr(
///     2,
///     3,
///     &Array::from_slice(&[0u32, 2, 3], 3)?,
///     &Array::from_slice(&[0u32, 2, 2], 3)?,
///     &Array::from_slice(&[1.0f32, 2.0, 3.0], 3)?,
/// );
/// let x = Array::from_slice(&[1.0f32, 10.0, 100.0], 3)?;
/// assert_eq!(a.matmul(&x).to_vec()?, [201.0, 300.0]);
/// assert_eq!(a.to_format(SparseFormat::Coo).nnz(), 3);
/// # Ok::<(), luma::LumaError>(())
/// ```
pub struct Sparse<T: SparseElem = f32> {
    ctx: Context,
    rows: usize,
    cols: usize,
    format: SparseFormat,
    nnz: usize,
    /// CSR row pointer or CSC column pointer.
    ptr: Option<Array<u32>>,
    /// The row of every entry (CSC and COO).
    row: Option<Array<u32>>,
    /// The column of every entry (CSR and COO).
    col: Option<Array<u32>>,
    val: Option<Array<T>>,
    /// The expanded pointer: the row of every entry of a CSR matrix, the column of every entry
    /// of a CSC matrix.
    expanded: OnceLock<Array<u32>>,
    /// The CSR and CSC forms of this matrix, built by the second product that would otherwise
    /// scatter with atomics (see [`Sparse::matmul`]).
    as_csr: OnceLock<Box<Sparse<T>>>,
    as_csc: OnceLock<Box<Sparse<T>>>,
    /// Products run so far that were not compressed along their output.
    scattered: AtomicU32,
    /// The rows longer than [`LONG_ROW`] of a compressed matrix, found at its first product.
    long: OnceLock<std::result::Result<RowPlan, Arc<LumaError>>>,
    error: Option<Arc<LumaError>>,
}

/// What the row kernels need to know about a compressed matrix, read back once.
#[derive(Clone)]
struct RowPlan {
    /// The rows longer than [`LONG_ROW`] (`None` if there are none).
    long: Option<LongRows>,
    /// For a matrix with rows longer than [`SHORT_ROWS`]: the first row that starts at or after the
    /// first entry of every tile of [`merge_source`] (`tiles + 1` of them, the last unused).
    tiles: Option<Array<u32>>,
}

/// The rows longer than [`LONG_ROW`], each cut in pieces of [`CHUNK`] entries.
#[derive(Clone)]
struct LongRows {
    /// The rows, ascending.
    rows: Array<u32>,
    /// `rows.len() + 1` offsets: the pieces of row `i` are `piece_ptr[i]..piece_ptr[i + 1]`.
    piece_ptr: Array<u32>,
    /// All the pieces.
    pieces: usize,
}

/// Matrices whose rows are all at most this long take the row kernel for a vector product (a
/// thread or a few lanes per row, no scratch pass); others the tile kernel ([`merge_source`]), which
/// splits the entries evenly whatever the row lengths are.
const SHORT_ROWS: u32 = 64;

/// The arrays of a [`Sparse`] matrix as [`Sparse::info`] returns them (ArrayFire's
/// `sparseGetInfo`).
#[derive(Clone, Debug)]
pub struct SparseInfo<T: SparseElem = f32> {
    /// The storage format.
    pub format: SparseFormat,
    /// Number of rows.
    pub rows: usize,
    /// Number of columns.
    pub cols: usize,
    /// Number of stored entries.
    pub nnz: usize,
    /// The values of the entries; `None` if there are none.
    pub values: Option<Array<T>>,
    /// CSR: the row pointer (`rows + 1` offsets); CSC and COO: the row of every entry (`None`
    /// without entries).
    pub row_idx: Option<Array<u32>>,
    /// CSC: the column pointer (`cols + 1` offsets); CSR and COO: the column of every entry
    /// (`None` without entries).
    pub col_idx: Option<Array<u32>>,
}

impl<T: SparseElem> Clone for Sparse<T> {
    fn clone(&self) -> Self {
        let expanded = OnceLock::new();
        if let Some(e) = self.expanded.get() {
            let _ = expanded.set(e.clone());
        }
        Sparse {
            ctx: self.ctx.clone(),
            rows: self.rows,
            cols: self.cols,
            format: self.format,
            nnz: self.nnz,
            ptr: self.ptr.clone(),
            row: self.row.clone(),
            col: self.col.clone(),
            val: self.val.clone(),
            expanded,
            as_csr: OnceLock::new(),
            as_csc: OnceLock::new(),
            scattered: AtomicU32::new(0),
            long: OnceLock::new(),
            error: self.error.clone(),
        }
    }
}

impl<T: SparseElem> fmt::Debug for Sparse<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Sparse")
            .field("rows", &self.rows)
            .field("cols", &self.cols)
            .field("format", &self.format)
            .field("nnz", &self.nnz)
            .field("error", &self.error)
            .finish()
    }
}

/// The failure `error` of `op` called at `location`, as the error a result is poisoned with.
fn failure(op: &'static str, location: &'static Location<'static>, error: LumaError) -> Arc<LumaError> {
    Arc::new(LumaError::Op { op, location, source: Arc::new(error) })
}

/// `Some(error)` unless `rows` and `cols` are non-zero and fit in `u32` with everything the
/// kernels index (`rows * cols` need not fit: only dense forms need it).
fn check_dims(rows: usize, cols: usize) -> Option<LumaError> {
    if rows == 0 || cols == 0 {
        return Some(LumaError::InvalidShape(format!("a sparse matrix of {rows} x {cols}: dimensions must be non-zero")));
    }
    if u32::try_from(rows).is_err() || u32::try_from(cols).is_err() || u32::try_from(rows + 1).is_err() || u32::try_from(cols + 1).is_err() {
        return Some(LumaError::DimensionTooLarge { shape: [cols, rows, 1, 1] });
    }
    None
}

/// `array` as one-dimensional, evaluated or recorded for good: what the sparse arrays are stored as.
fn flat<T: crate::gpu_type::ScalarType>(array: &Array<T>) -> Array<T> {
    let n = array.len();
    if array.shape().dims() == [n, 1, 1, 1] { array.cache() } else { array.reshape(n).cache() }
}

/// The offsets of `n` groups from the group of every entry in ascending order (`sorted`): `n + 1`
/// offsets, the first 0, the offset of group `g` being the number of entries of lower groups.
fn pointer(sorted: &Array<u32>, n: usize) -> Array<u32> {
    match Array::<u32>::range_on(sorted.context(), n + 1, 0) {
        Ok(groups) => sorted.searchsorted(&groups, Side::Left).cache(),
        Err(error) => Array::poisoned(sorted.context(), Shape::from(n + 1), Arc::new(error)),
    }
}

/// `(pointer, other, values)` of the entries regrouped by `key` (a stable sort, so the entries of a
/// group keep their order), for `n` groups: two radix sorts of the key (below `n`, so of
/// `ceil(log2 n)` bits) that carry `other` and `val` along, then the offsets from the sorted keys.
fn regroup(key: &Array<u32>, other: &Array<u32>, val: &Array<f32>, n: usize) -> (Array<u32>, Array<u32>, Array<f32>) {
    let bits = usize::BITS - n.saturating_sub(1).leading_zeros();
    let (sorted, other) = key.sort_by_key_bits(other, bits.max(1));
    let (_, val) = key.sort_by_key_bits(val, bits.max(1));
    (pointer(&sorted.cache(), n), other.cache(), val.cache())
}

impl Sparse<f32> {
    // -- Construction --

    fn assemble(
        ctx: &Context,
        (rows, cols): (usize, usize),
        format: SparseFormat,
        ptr: Option<Array<u32>>,
        row: Option<Array<u32>>,
        col: Option<Array<u32>>,
        val: Array<f32>,
    ) -> Sparse {
        Sparse { ctx: ctx.clone(), rows, cols, format, nnz: val.len(), ptr, row, col, val: Some(val), expanded: OnceLock::new(), as_csr: OnceLock::new(), as_csc: OnceLock::new(), scattered: AtomicU32::new(0), long: OnceLock::new(), error: None }
    }

    fn poisoned(ctx: &Context, rows: usize, cols: usize, format: SparseFormat, error: Arc<LumaError>) -> Sparse {
        Sparse { ctx: ctx.clone(), rows, cols, format, nnz: 0, ptr: None, row: None, col: None, val: None, expanded: OnceLock::new(), as_csr: OnceLock::new(), as_csc: OnceLock::new(), scattered: AtomicU32::new(0), long: OnceLock::new(), error: Some(error) }
    }

    /// A matrix of `rows x cols` without stored entries, in `format`, on the default context.
    ///
    /// # Errors
    ///
    /// A zero dimension, or one that does not fit in `u32`, is an error.
    pub fn empty(rows: usize, cols: usize, format: SparseFormat) -> Result<Sparse> {
        if let Some(e) = check_dims(rows, cols) {
            return Err(e);
        }
        Ok(Self::empty_on(&default_context()?, rows, cols, format))
    }

    /// [`Sparse::empty`] on an explicit context. An invalid size gives a poisoned matrix
    /// ([`Sparse::error`]).
    #[track_caller]
    pub fn empty_on(ctx: &Context, rows: usize, cols: usize, format: SparseFormat) -> Sparse {
        let loc = Location::caller();
        if let Some(e) = check_dims(rows, cols) {
            return Self::poisoned(ctx, rows, cols, format, failure("Sparse::empty", loc, e));
        }
        let n = match format {
            SparseFormat::Csr => rows,
            SparseFormat::Csc => cols,
            SparseFormat::Coo => 0,
        };
        let ptr = if format == SparseFormat::Coo {
            None
        } else {
            match Array::<u32>::zeros_on(ctx, n + 1) {
                Ok(p) => Some(p),
                Err(e) => return Self::poisoned(ctx, rows, cols, format, failure("Sparse::empty", loc, e)),
            }
        };
        Sparse { ctx: ctx.clone(), rows, cols, format, nnz: 0, ptr, row: None, col: None, val: None, expanded: OnceLock::new(), as_csr: OnceLock::new(), as_csc: OnceLock::new(), scattered: AtomicU32::new(0), long: OnceLock::new(), error: None }
    }

    /// The non-zero entries of a dense matrix of shape `[cols, rows]` (`rows x cols`, one
    /// dimension of extent 1 for a vector; more than two axes are an error) in `format`. Entries
    /// equal to zero (`-0.0` too) are dropped, NaN is kept. CSR and COO list the entries in row
    /// order, CSC in column order, both ascending inside a row (column).
    ///
    /// The number of entries depends on the data, so this reads a count back and
    /// **synchronizes** once, as [`Array::filter`] does. Errors, including a poisoned `dense`,
    /// poison the matrix.
    #[track_caller]
    pub fn from_dense(dense: &Array<f32>, format: SparseFormat) -> Sparse {
        let loc = Location::caller();
        let ctx = dense.context();
        let d = dense.shape().dims();
        let (cols, rows) = (d[0], d[1]);
        if d[2] != 1 || d[3] != 1 {
            let e = LumaError::InvalidShape(format!("sparse of an array of shape {}: it must be a matrix", dense.shape()));
            return Self::poisoned(ctx, rows, cols, format, failure("sparse", loc, e));
        }
        match Self::dense_entries(dense, format, rows, cols) {
            Ok(s) => s,
            Err(e) => Self::poisoned(ctx, rows, cols, format, failure("sparse", loc, e)),
        }
    }

    fn dense_entries(dense: &Array<f32>, format: SparseFormat, rows: usize, cols: usize) -> Result<Sparse> {
        let ctx = dense.context();
        if let Some(e) = check_dims(rows, cols) {
            return Err(e);
        }
        // Compressed order: the major axis is what the pointer indexes, the minor one what is
        // stored per entry. The dense matrix is read along that order (a transpose for CSC).
        let (major, minor, source) = match format {
            SparseFormat::Csc => (cols, rows, dense.transpose()),
            _ => (rows, cols, dense.clone()),
        };
        // NaN is an entry (a NaN comparison is unspecified on some backends, so it is tested apart).
        let mask = source.ne(0.0f32) | source.is_nan();
        let Some(idx) = mask.nonzero()? else {
            return Ok(Self::empty_on(ctx, rows, cols, format));
        };
        let idx = idx.cache();
        let val = source.take(&idx).cache();
        let (id_major, id_minor) = ((&idx / minor as u32).cache(), (&idx % minor as u32).cache());
        let ptr = || -> Array<u32> {
            let counts = mask.sum_axis(0).reshape(major);
            match Array::<u32>::zeros_on(ctx, 1) {
                Ok(zero) => concat(0, [&zero, &counts.cumsum(0)]).cache(),
                Err(e) => Array::poisoned(ctx, Shape::from(major + 1), Arc::new(e)),
            }
        };
        Ok(match format {
            SparseFormat::Csr => Self::assemble(ctx, (rows, cols), format, Some(ptr()), None, Some(id_minor), val),
            SparseFormat::Csc => Self::assemble(ctx, (rows, cols), format, Some(ptr()), Some(id_minor), None, val),
            SparseFormat::Coo => Self::assemble(ctx, (rows, cols), format, None, Some(id_major), Some(id_minor), val),
        })
    }

    /// A matrix from `(row, column, value)` triplets in `format`: entry `k` is `values[k]` at
    /// `(row_idx[k], col_idx[k])`. The three arrays must have the same length (else the result is
    /// poisoned with [`LumaError::LengthMismatch`]); the order is free and duplicates are kept
    /// (they add up, see the guide chapter "Sparse matrices"). Into CSR (CSC) the entries are
    /// sorted by row (column) with a stable sort: entries of one row (column) keep their input
    /// order, which is not sorted by column (row) unless the input was.
    #[track_caller]
    pub fn from_triplets(rows: usize, cols: usize, row_idx: &Array<u32>, col_idx: &Array<u32>, values: &Array<f32>, format: SparseFormat) -> Sparse {
        let loc = Location::caller();
        let ctx = values.context();
        let n = values.len();
        let bad = if let Some(e) = check_dims(rows, cols) {
            Some(e)
        } else if row_idx.len() != n {
            Some(LumaError::LengthMismatch { expected: n, actual: row_idx.len() })
        } else if col_idx.len() != n {
            Some(LumaError::LengthMismatch { expected: n, actual: col_idx.len() })
        } else if !ctx.same_as(row_idx.context()) || !ctx.same_as(col_idx.context()) {
            Some(LumaError::ContextMismatch)
        } else {
            None
        };
        if let Some(e) = bad {
            return Self::poisoned(ctx, rows, cols, format, failure("Sparse::from_triplets", loc, e));
        }
        let (r, c, v) = (flat(row_idx), flat(col_idx), flat(values));
        match format {
            SparseFormat::Coo => Self::assemble(ctx, (rows, cols), format, None, Some(r), Some(c), v),
            SparseFormat::Csr => {
                let (ptr, col, val) = regroup(&r, &c, &v, rows);
                Self::assemble(ctx, (rows, cols), format, Some(ptr), None, Some(col), val)
            }
            SparseFormat::Csc => {
                let (ptr, row, val) = regroup(&c, &r, &v, cols);
                Self::assemble(ctx, (rows, cols), format, Some(ptr), Some(row), None, val)
            }
        }
    }

    /// A COO matrix from the arrays of its entries: [`Sparse::from_triplets`] with
    /// [`SparseFormat::Coo`].
    #[track_caller]
    pub fn from_coo(rows: usize, cols: usize, row_idx: &Array<u32>, col_idx: &Array<u32>, values: &Array<f32>) -> Sparse {
        Self::from_triplets(rows, cols, row_idx, col_idx, values, SparseFormat::Coo)
    }

    /// A CSR matrix from its arrays: `row_ptr` has `rows + 1` offsets (non-decreasing, the first 0
    /// and the last the number of entries, which this cannot check without reading them back),
    /// `col_idx` and `values` one element per entry. Wrong lengths poison the result with
    /// [`LumaError::LengthMismatch`]. The arrays are used as they are (no copy of evaluated ones).
    #[track_caller]
    pub fn from_csr(rows: usize, cols: usize, row_ptr: &Array<u32>, col_idx: &Array<u32>, values: &Array<f32>) -> Sparse {
        let loc = Location::caller();
        match Self::check_compressed(rows, cols, row_ptr, rows, col_idx, values) {
            Some(e) => Self::poisoned(values.context(), rows, cols, SparseFormat::Csr, failure("Sparse::from_csr", loc, e)),
            None => Self::assemble(values.context(), (rows, cols), SparseFormat::Csr, Some(flat(row_ptr)), None, Some(flat(col_idx)), flat(values)),
        }
    }

    /// A CSC matrix from its arrays: `col_ptr` has `cols + 1` offsets, `row_idx` and `values` one
    /// element per entry; otherwise as [`Sparse::from_csr`].
    #[track_caller]
    pub fn from_csc(rows: usize, cols: usize, col_ptr: &Array<u32>, row_idx: &Array<u32>, values: &Array<f32>) -> Sparse {
        let loc = Location::caller();
        match Self::check_compressed(rows, cols, col_ptr, cols, row_idx, values) {
            Some(e) => Self::poisoned(values.context(), rows, cols, SparseFormat::Csc, failure("Sparse::from_csc", loc, e)),
            None => Self::assemble(values.context(), (rows, cols), SparseFormat::Csc, Some(flat(col_ptr)), Some(flat(row_idx)), None, flat(values)),
        }
    }

    fn check_compressed(rows: usize, cols: usize, ptr: &Array<u32>, major: usize, idx: &Array<u32>, values: &Array<f32>) -> Option<LumaError> {
        if let Some(e) = check_dims(rows, cols) {
            return Some(e);
        }
        if ptr.len() != major + 1 {
            return Some(LumaError::LengthMismatch { expected: major + 1, actual: ptr.len() });
        }
        if idx.len() != values.len() {
            return Some(LumaError::LengthMismatch { expected: values.len(), actual: idx.len() });
        }
        let ctx = values.context();
        (!ctx.same_as(ptr.context()) || !ctx.same_as(idx.context())).then_some(LumaError::ContextMismatch)
    }

    // -- Introspection --

    /// Number of rows.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Number of columns.
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// The number of stored entries (duplicates count separately).
    pub fn nnz(&self) -> usize {
        self.nnz
    }

    /// The storage format.
    pub fn format(&self) -> SparseFormat {
        self.format
    }

    /// The [`Context`] the arrays live on.
    pub fn context(&self) -> &Context {
        &self.ctx
    }

    /// The error this matrix is poisoned with, if any (it names the operation that failed and
    /// where it was called from).
    pub fn error(&self) -> Option<&LumaError> {
        self.error.as_deref()
    }

    /// The values of the stored entries (`nnz` elements, in storage order); `None` if there are
    /// none. Sharing the matrix's buffer: no copy.
    pub fn values(&self) -> Option<Array<f32>> {
        self.val.clone()
    }

    /// ArrayFire's row array: CSR, the row pointer of `rows + 1` offsets; CSC and COO, the row
    /// of every entry (`None` without entries). See [`SparseInfo::row_idx`].
    pub fn row_idx(&self) -> Option<Array<u32>> {
        match self.format {
            SparseFormat::Csr => self.ptr.clone(),
            _ => self.row.clone(),
        }
    }

    /// ArrayFire's column array: CSC, the column pointer of `cols + 1` offsets; CSR and COO, the
    /// column of every entry (`None` without entries). See [`SparseInfo::col_idx`].
    pub fn col_idx(&self) -> Option<Array<u32>> {
        match self.format {
            SparseFormat::Csc => self.ptr.clone(),
            _ => self.col.clone(),
        }
    }

    /// The format, size and arrays of the matrix in one value (ArrayFire's `sparseGetInfo`).
    pub fn info(&self) -> SparseInfo<f32> {
        SparseInfo {
            format: self.format,
            rows: self.rows,
            cols: self.cols,
            nnz: self.nnz,
            values: self.values(),
            row_idx: self.row_idx(),
            col_idx: self.col_idx(),
        }
    }

    /// The shape of the dense form, `[cols, rows]`.
    fn dense_shape(&self) -> Shape {
        Shape::new([self.cols, self.rows, 1, 1])
    }

    /// A poisoned dense array of `shape` if this matrix is poisoned.
    fn dense_poison(&self, shape: Shape) -> Option<Array<f32>> {
        self.error.as_ref().map(|e| Array::poisoned(&self.ctx, shape, Arc::clone(e)))
    }

    /// The row of every entry of a CSR matrix, the column of every entry of a CSC matrix: the
    /// pointer expanded by a binary search per entry, computed once and kept. Only for a matrix
    /// with entries.
    fn expanded(&self) -> &Array<u32> {
        self.expanded.get_or_init(|| {
            let (ptr, nnz) = (self.ptr.as_ref().expect("compressed matrix"), self.nnz);
            let major = ptr.len() - 1;
            let ids = match Array::<u32>::range_on(&self.ctx, nnz, 0) {
                Ok(ids) => ids,
                Err(e) => return Array::poisoned(&self.ctx, Shape::from(nnz), Arc::new(e)),
            };
            ptr.slice(0, 1..=major).searchsorted(&ids, Side::Right).cache()
        })
    }

    /// The row and the column of every entry. Only for a matrix with entries.
    fn coords(&self) -> (Array<u32>, Array<u32>) {
        match self.format {
            SparseFormat::Csr => (self.expanded().clone(), self.col.clone().expect("entries")),
            SparseFormat::Csc => (self.row.clone().expect("entries"), self.expanded().clone()),
            SparseFormat::Coo => (self.row.clone().expect("entries"), self.col.clone().expect("entries")),
        }
    }

    /// The flat index of every entry in the dense form (`row * cols + col`).
    fn linear(&self) -> Array<u32> {
        let (r, c) = self.coords();
        &r * self.cols as u32 + &c
    }

    // -- Conversions --

    /// The same matrix in `format`. The same format is a clone (no work, buffers shared); other
    /// conversions are a stable sort of the entries by their new major index (CSR to CSC and back
    /// keep the entries of a row, a column, in order; COO to a compressed format as in
    /// [`Sparse::from_triplets`]) and take the time of one sort of `u32` keys with a gather of
    /// the indices and values. Conversions to COO move no data beyond expanding the pointer. A
    /// poisoned matrix converts to a poisoned matrix.
    pub fn to_format(&self, format: SparseFormat) -> Sparse {
        if self.error.is_some() || format == self.format {
            return self.clone();
        }
        if self.nnz == 0 {
            return Self::empty_on(&self.ctx, self.rows, self.cols, format);
        }
        let (ctx, rows, cols) = (&self.ctx, self.rows, self.cols);
        let val = self.val.as_ref().expect("entries");
        let (r, c) = self.coords();
        match format {
            SparseFormat::Coo => Self::assemble(ctx, (rows, cols), format, None, Some(r), Some(c), val.clone()),
            SparseFormat::Csr => {
                let (ptr, col, val) = regroup(&r, &c, val, rows);
                Self::assemble(ctx, (rows, cols), format, Some(ptr), None, Some(col), val)
            }
            SparseFormat::Csc => {
                let (ptr, row, val) = regroup(&c, &r, val, cols);
                Self::assemble(ctx, (rows, cols), format, Some(ptr), Some(row), None, val)
            }
        }
    }

    /// The dense matrix, of shape `[cols, rows]`: zeros with the entries added in (duplicates
    /// add up). Needs `rows * cols` to fit in `u32` like any array; a matrix that does not
    /// gives a poisoned array.
    #[track_caller]
    pub fn to_dense(&self) -> Array<f32> {
        let loc = Location::caller();
        let shape = self.dense_shape();
        if let Some(p) = self.dense_poison(shape) {
            return p;
        }
        let mut out = match Array::<f32>::zeros_on(&self.ctx, shape) {
            Ok(out) => out,
            Err(e) => return Array::poisoned(&self.ctx, shape, failure("Sparse::to_dense", loc, e)),
        };
        if self.nnz > 0 {
            let val = self.val.as_ref().expect("entries");
            if let Err(e) = out.put_add(&self.linear(), val) {
                out.poison(Arc::new(e));
            }
        }
        out
    }

    /// The transpose, a `cols x rows` matrix. Free for the compressed formats (a CSR matrix is
    /// the CSC form of its transpose: the arrays are shared and only the format changes) and for
    /// COO (rows and columns swap).
    pub fn transpose(&self) -> Sparse {
        let expanded = OnceLock::new();
        if let Some(e) = self.expanded.get() {
            let _ = expanded.set(e.clone());
        }
        let (format, row, col) = match self.format {
            SparseFormat::Csr => (SparseFormat::Csc, self.col.clone(), None),
            SparseFormat::Csc => (SparseFormat::Csr, None, self.row.clone()),
            SparseFormat::Coo => (SparseFormat::Coo, self.col.clone(), self.row.clone()),
        };
        Sparse {
            ctx: self.ctx.clone(),
            rows: self.cols,
            cols: self.rows,
            format,
            nnz: self.nnz,
            ptr: self.ptr.clone(),
            row,
            col,
            val: self.val.clone(),
            expanded,
            as_csr: OnceLock::new(),
            as_csc: OnceLock::new(),
            scattered: AtomicU32::new(0),
            long: OnceLock::new(),
            error: self.error.clone(),
        }
    }

    // -- Products --

    /// The matrix product `A x` of this `rows x cols` matrix with a dense `x`: a vector (shape
    /// `[cols]`, result `[rows]`) or a matrix of `c` columns (shape `[c, cols]`, result
    /// `[c, rows]`); see [`Sparse::matmul_t`] for `A^T x`. A shape that is neither poisons the
    /// result with [`LumaError::ShapeMismatch`]. `x` may be a pending expression (it is gathered
    /// from inside the kernel). See the guide chapter "Sparse matrices" for how each format runs.
    ///
    /// A product whose output is not the compressed axis of the format (CSC `A x`, CSR `A^T x`,
    /// COO either way) scatters its products with atomic adds the first time. A second one
    /// converts the matrix to the compressed form its output wants (a CSR copy for `A x`, a CSC
    /// copy for `A^T x`, kept with the matrix until it is dropped, at the cost of one more copy
    /// of the entries) and runs from that: faster on rows with many entries and reproducible
    /// to the last bit.
    #[track_caller]
    pub fn matmul(&self, x: &Array<f32>) -> Array<f32> {
        self.product(x, false, "Sparse::matmul", Location::caller())
    }

    /// The product `A^T x` with the transpose of this matrix (no transposed copy is made):
    /// `x` is a vector of `rows` elements or a matrix of shape `[c, rows]`, the result has `cols`
    /// rows. Otherwise as [`Sparse::matmul`].
    #[track_caller]
    pub fn matmul_t(&self, x: &Array<f32>) -> Array<f32> {
        self.product(x, true, "Sparse::matmul_t", Location::caller())
    }

    fn product(&self, x: &Array<f32>, transposed: bool, op: &'static str, loc: &'static Location<'static>) -> Array<f32> {
        let (inner, outer) = if transposed { (self.rows, self.cols) } else { (self.cols, self.rows) };
        let xd = x.shape().dims();
        // A vector of `inner` elements, or `[c, inner]` columns.
        let (c, vector) = if xd[1..] == [1, 1, 1] && xd[0] == inner {
            (1, true)
        } else if xd[2..] == [1, 1] && xd[1] == inner {
            (xd[0], false)
        } else {
            let lhs = self.dense_shape().dims();
            let e = LumaError::ShapeMismatch { lhs, rhs: xd };
            return Array::poisoned(&self.ctx, Shape::from(outer), failure(op, loc, e));
        };
        let out_shape = if vector { Shape::from(outer) } else { Shape::new([c, outer, 1, 1]) };
        if let Some(p) = self.dense_poison(out_shape) {
            return p;
        }
        if !self.ctx.same_as(x.context()) {
            return Array::poisoned(&self.ctx, out_shape, failure(op, loc, LumaError::ContextMismatch));
        }
        if self.nnz == 0 {
            return Array::zeros_on(&self.ctx, out_shape).unwrap_or_else(|e| Array::poisoned(&self.ctx, out_shape, failure(op, loc, e)));
        }
        // The sum over a compressed pointer needs no atomics: CSR `A x`, CSC `A^T x`.
        let native = matches!((self.format, transposed), (SparseFormat::Csr, false) | (SparseFormat::Csc, true));
        if !native && self.scattered.fetch_add(1, Ordering::Relaxed) >= 1 {
            let form = if transposed { self.compressed(SparseFormat::Csc) } else { self.compressed(SparseFormat::Csr) };
            return form.product(x, transposed, op, loc);
        }
        let val = self.val.as_ref().expect("entries");
        if native {
            let minor = if self.format == SparseFormat::Csr { self.col.as_ref() } else { self.row.as_ref() };
            let (minor, ptr) = (minor.expect("entries"), self.ptr.as_ref().expect("compressed"));
            let plan = match self.row_plan(ptr) {
                Ok(plan) => plan,
                Err(e) => return Array::poisoned(&self.ctx, out_shape, failure(op, loc, e)),
            };
            if let (1, Some(tiles)) = (c, &plan.tiles) {
                return match self.merge_product((ptr, minor, val), tiles, &x.reshape(inner), outer) {
                    Ok(y) => y.reshape(out_shape),
                    Err(e) => Array::poisoned(&self.ctx, out_shape, failure(op, loc, e)),
                };
            }
            return match self.rows_kernels(plan, (ptr, minor, val), x, c, outer) {
                Ok(y) => y.reshape(out_shape),
                Err(e) => Array::poisoned(&self.ctx, out_shape, failure(op, loc, e)),
            };
        }
        let (r, cc) = self.coords();
        let (gather, scatter) = if transposed { (r, cc) } else { (cc, r) };
        if c == 1 {
            return (val * x.reshape(inner).take(&gather)).group_sum(&scatter, outer).reshape(out_shape);
        }
        // `c` columns: the `c` values of an entry are adjacent in `x` and in the result, so an
        // entry's products are one `[c]` run of the product array, scattered with atomics.
        let nnz = self.nnz;
        let j = match Array::<u32>::range_on(&self.ctx, [c, nnz], 0) {
            Ok(j) => j,
            Err(e) => return Array::poisoned(&self.ctx, out_shape, failure(op, loc, e)),
        };
        let wide = |ids: &Array<u32>| ids.reshape([1, nnz]).broadcast_to([c, nnz]) * c as u32 + &j;
        let prod = val.reshape([1, nnz]).broadcast_to([c, nnz]) * x.take(&wide(&gather));
        prod.group_sum(&wide(&scatter), outer * c).reshape(out_shape)
    }

    /// This matrix in a compressed `format` (CSR or CSC), converted once and kept.
    fn compressed(&self, format: SparseFormat) -> &Sparse {
        if self.format == format {
            return self;
        }
        let slot = if format == SparseFormat::Csr { &self.as_csr } else { &self.as_csc };
        slot.get_or_init(|| Box::new(self.to_format(format)))
    }

    /// The row lengths of a compressed matrix (see [`RowPlan`]), read back at the first product.
    fn row_plan(&self, ptr: &Array<u32>) -> Result<RowPlan> {
        let plan = self.long.get_or_init(|| {
            let n = ptr.len() - 1;
            let len = ptr.slice(0, 1..=n) - ptr.slice(0, 0..n);
            let max_len = len.max().item().map_err(Arc::new)?;
            let mut long = None;
            if max_len > LONG_ROW
                && let Some(rows) = len.gt(LONG_ROW).nonzero().map_err(Arc::new)?
            {
                let ends = ((len.take(&rows) + (CHUNK - 1)) / CHUNK).cumsum(0).cache();
                let zero = Array::<u32>::zeros_on(&self.ctx, 1).map_err(Arc::new)?;
                let piece_ptr = concat(0, [&zero, &ends]).cache();
                let pieces = ends.slice(0, rows.len() - 1..rows.len()).item().map_err(Arc::new)? as usize;
                long = Some(LongRows { rows, piece_ptr, pieces });
            }
            let mut tiles = None;
            if max_len > SHORT_ROWS {
                let nb = self.nnz.div_ceil(MERGE_TILE as usize);
                let first = (Array::<u32>::range_on(&self.ctx, nb + 1, 0).map_err(Arc::new)? * MERGE_TILE).minimum(self.nnz as u32);
                tiles = Some(ptr.searchsorted(&first, Side::Left).cache());
            }
            Ok(RowPlan { long, tiles })
        });
        match plan {
            Ok(plan) => Ok(plan.clone()),
            Err(e) => Err(LumaError::clone(e)),
        }
    }

    /// `y = A x` for a compressed matrix by the tile kernel ([`merge_source`]) and its carry pass.
    fn merge_product(&self, (ptr, minor, val): (&Array<u32>, &Array<u32>, &Array<f32>), tiles: &Array<u32>, x: &Array<f32>, outer: usize) -> Result<Array<f32>> {
        use crate::array::Failure;
        use crate::context::{RawArg, Scratch};
        // The same buffers are bound twice, as scalars and as `vec4`s, which the typed kernel
        // arguments of `launch_raw` do not allow.
        let plain = |f: Failure| match f {
            Failure::New(e) => e,
            Failure::Inherited(e) => LumaError::clone(&e),
        };
        let ctx = &self.ctx;
        let nb = self.nnz.div_ceil(MERGE_TILE as usize);
        let y = Array::<f32>::alloc(ctx, Shape::from(outer))?;
        let y_storage = Arc::clone(y.storage().map_err(plain)?);
        let y_pin = y_storage.pin_single()?;
        let (ptr_pin, ptr_bytes) = ptr.ready().map_err(plain)?;
        let (minor_pin, minor_bytes) = minor.ready().map_err(plain)?;
        let (val_pin, val_bytes) = val.ready().map_err(plain)?;
        let (x_pin, x_bytes) = x.ready().map_err(plain)?;
        let (tiles_pin, tiles_bytes) = tiles.ready().map_err(plain)?;
        let mut scratch = Scratch::new(ctx);
        let sc_bytes = nb as u64 * 12;
        let sc = scratch.take(sc_bytes)?;
        let nbu = u32::try_from(nb).map_err(|_| LumaError::DimensionTooLarge { shape: [nb, 1, 1, 1] })?;
        // `Params` of the kernels: the entries, the rows, the tiles, the last index of `x`.
        let params = [self.nnz as u32, outer as u32, nbu, (x.len() - 1) as u32];
        let uniform = bytemuck::cast_slice::<u32, u8>(&params);
        let core = ctx.jit().scan_kernel(ctx, "sparse_spmv_tiles".into(), "sparse_spmv_tiles", merge_source)?;
        let args = [
            RawArg::read(ptr_pin.buffer(), ptr_bytes),
            RawArg::read(minor_pin.buffer(), minor_bytes),
            RawArg::read(val_pin.buffer(), val_bytes),
            RawArg::read(x_pin.buffer(), x_bytes),
            RawArg::read(tiles_pin.buffer(), tiles_bytes),
            RawArg::write(y_pin.buffer(), y.byte_len()),
            RawArg::write(&sc, sc_bytes),
            RawArg::Uniform(uniform),
            RawArg::read(minor_pin.buffer(), minor_bytes),
            RawArg::read(val_pin.buffer(), val_bytes),
        ];
        ctx.launch(&core, crate::ops::fold_workgroups(ctx, nbu)?, [nbu, 1, 1], &args)?;
        if nb > 1 {
            let core = ctx.jit().scan_kernel(ctx, "sparse_spmv_carry".into(), "sparse_spmv_carry", merge_carry_source)?;
            let args = [RawArg::write(y_pin.buffer(), y.byte_len()), RawArg::read(&sc, sc_bytes), RawArg::read(ptr_pin.buffer(), ptr_bytes), RawArg::Uniform(uniform)];
            ctx.launch(&core, crate::ops::fold_workgroups(ctx, nbu.div_ceil(128))?, [nbu, 1, 1], &args)?;
        }
        drop(scratch);
        Ok(y)
    }

    /// `Y = A X` for a compressed matrix (`ptr`, the `minor` index of every entry, `val`) and a
    /// vector or a dense `X` of shape `[c, inner]`, from the row kernels: rows of at most
    /// [`LONG_ROW`] entries by `2^k` lanes each (a vector: `2^k` the power of two nearest the
    /// mean row length, at most 32) or by one invocation per output element (columns: the `c`
    /// invocations of a row read the same entry and `c` adjacent elements of `X`, so the gather is
    /// coalesced), longer rows by a workgroup per row (and column). Sums are in entry order per
    /// lane, so the results are the same on every run.
    fn rows_kernels(&self, plan: RowPlan, (ptr, minor, val): (&Array<u32>, &Array<u32>, &Array<f32>), x: &Array<f32>, c: usize, outer: usize) -> Result<Array<f32>> {
        let shape = if c == 1 { Shape::from(outer) } else { Shape::new([c, outer, 1, 1]) };
        shape.validate()?;
        let total = u32::try_from(shape.len()).map_err(|_| LumaError::DimensionTooLarge { shape: shape.dims() })?;
        let long = plan.long.clone();
        let mut y = Array::<f32>::alloc(&self.ctx, shape)?;
        let grid = |groups: usize| -> Result<[u32; 3]> {
            let groups = u32::try_from(groups).map_err(|_| LumaError::DimensionTooLarge { shape: [groups, 1, 1, 1] })?;
            Ok([groups.min(GRID_X), groups.div_ceil(GRID_X), 1])
        };
        if c == 1 {
            let avg = self.nnz as f64 / outer as f64;
            // Two to four entries per lane: a barrier costs more than an idle lane, a long serial
            // walk more than a few barriers (measured on an RTX 3090: 5 entries per row best with
            // one or two lanes, 16 with eight).
            let lanes_log = ((avg / 3.0).max(1.0).log2().ceil() as u32).min(4);
            let kernel = self.ctx.kernel(SPMV_ROWS).label("sparse_spmv_rows").build()?;
            let params = RowParams { shape: lanes_log, total: outer as u32, cap: LONG_ROW, chunks: 1 };
            let groups = (outer << lanes_log).div_ceil(WORKGROUP as usize);
            kernel.launch_raw(grid(groups)?, crate::args![ptr, minor, val, x, &mut y, crate::kernel::uniform(&params)])?;
        } else {
            let kernel = self.ctx.kernel(SPMM_ROWS).label("sparse_spmm_rows").build()?;
            let params = RowParams { shape: c as u32, total, cap: LONG_ROW, chunks: 1 };
            let groups = (total as usize).div_ceil(WORKGROUP as usize);
            kernel.launch_raw(grid(groups)?, crate::args![ptr, minor, val, x, &mut y, crate::kernel::uniform(&params)])?;
        }
        if let Some(long) = long {
            let (m, workgroups) = (long.rows.len(), long.pieces * c);
            let mut part = Array::<f32>::alloc(&self.ctx, Shape::from(workgroups))?;
            let kernel = self.ctx.kernel(LONG_ROWS).label("sparse_long_rows").build()?;
            let params = RowParams { shape: c as u32, total: m as u32, cap: LONG_ROW, chunks: CHUNK };
            kernel.launch_raw(grid(workgroups)?, crate::args![ptr, minor, val, x, &mut part, &long.rows, &long.piece_ptr, crate::kernel::uniform(&params)])?;
            let sums = self.ctx.kernel(LONG_SUMS).label("sparse_long_sums").build()?;
            let params = RowParams { shape: c as u32, total: (m * c) as u32, cap: LONG_ROW, chunks: CHUNK };
            sums.launch([(m * c) as u32], crate::args![&part, &long.rows, &long.piece_ptr, &mut y, crate::kernel::uniform(&params)])?;
        }
        Ok(y)
    }

    // -- Elementwise --

    /// The error of an operation with `dense` unless this matrix is valid and `dense` has the
    /// shape `[cols, rows]` of the dense form.
    fn dense_operand_error(&self, dense: &Array<f32>, op: &'static str, loc: &'static Location<'static>) -> Option<Arc<LumaError>> {
        if let Some(e) = &self.error {
            return Some(Arc::clone(e));
        }
        let shape = self.dense_shape();
        if dense.shape() != shape {
            let e = LumaError::ShapeMismatch { lhs: shape.dims(), rhs: dense.shape().dims() };
            return Some(failure(op, loc, e));
        }
        (!self.ctx.same_as(dense.context())).then(|| failure(op, loc, LumaError::ContextMismatch))
    }

    /// This matrix with `f` applied to its values: same pattern, new values array.
    fn map_values(&self, f: impl FnOnce(&Array<f32>) -> Array<f32>) -> Sparse {
        let mut s = self.clone();
        if let Some(v) = &self.val {
            s.val = Some(f(v).cache());
        }
        s
    }

    /// The sparse matrix with the entries multiplied by the dense matrix at the same positions,
    /// `A .* D` (`dense` has shape `[cols, rows]`): the pattern is that of this matrix (a zero
    /// in `dense` gives a stored zero; an infinite or NaN `dense` where this matrix has no
    /// entry does not matter, as the implied `0 * inf` is not computed). Duplicate entries
    /// multiply separately, which adds up to the product of their sum.
    #[track_caller]
    pub fn mul_dense(&self, dense: &Array<f32>) -> Sparse {
        self.elementwise_dense(dense, "Sparse::mul_dense", Location::caller(), |v, d| v * d)
    }

    /// The sparse matrix with the entries divided by the dense matrix at the same positions,
    /// `A ./ D`; like [`Sparse::mul_dense`].
    #[track_caller]
    pub fn div_dense(&self, dense: &Array<f32>) -> Sparse {
        self.elementwise_dense(dense, "Sparse::div_dense", Location::caller(), |v, d| v / d)
    }

    fn elementwise_dense(&self, dense: &Array<f32>, op: &'static str, loc: &'static Location<'static>, f: impl FnOnce(&Array<f32>, Array<f32>) -> Array<f32>) -> Sparse {
        if let Some(e) = self.dense_operand_error(dense, op, loc) {
            return Self::poisoned(&self.ctx, self.rows, self.cols, self.format, e);
        }
        if self.nnz == 0 {
            return self.clone();
        }
        let at = dense.take(&self.linear());
        self.map_values(|v| f(v, at))
    }

    /// `A * s`: the values multiplied by a scalar, same pattern.
    pub fn mul_scalar(&self, s: f32) -> Sparse {
        self.map_values(|v| v * s)
    }

    /// `A / s`: the values divided by a scalar, same pattern.
    pub fn div_scalar(&self, s: f32) -> Sparse {
        self.map_values(|v| v / s)
    }

    /// `-A`.
    pub fn neg(&self) -> Sparse {
        self.map_values(|v| -v)
    }

    /// `A + D` as a dense matrix (`dense` has shape `[cols, rows]`): a copy of `dense` with the
    /// entries added in (duplicates add up). `f32` sums of several entries into one position
    /// (duplicates) are accumulated atomically in no fixed order.
    #[track_caller]
    pub fn add_dense(&self, dense: &Array<f32>) -> Array<f32> {
        self.accumulate(dense, false, "Sparse::add_dense", Location::caller())
    }

    /// `A - D` as a dense matrix; see [`Sparse::add_dense`].
    #[track_caller]
    pub fn sub_dense(&self, dense: &Array<f32>) -> Array<f32> {
        self.accumulate(dense, true, "Sparse::sub_dense", Location::caller())
    }

    fn accumulate(&self, dense: &Array<f32>, negate_dense: bool, op: &'static str, loc: &'static Location<'static>) -> Array<f32> {
        if let Some(e) = self.dense_operand_error(dense, op, loc) {
            return Array::poisoned(&self.ctx, self.dense_shape(), e);
        }
        let mut out = if negate_dense { (-dense).cache() } else { dense.copy().cache() };
        if self.nnz > 0 {
            let val = self.val.as_ref().expect("entries");
            if let Err(e) = out.put_add(&self.linear(), val) {
                out.poison(Arc::new(e));
            }
        }
        out
    }

    /// `D - A` as a dense matrix; see [`Sparse::add_dense`].
    #[track_caller]
    pub fn rsub_dense(&self, dense: &Array<f32>) -> Array<f32> {
        let (op, loc) = ("Sparse::rsub_dense", Location::caller());
        if let Some(e) = self.dense_operand_error(dense, op, loc) {
            return Array::poisoned(&self.ctx, self.dense_shape(), e);
        }
        let mut out = dense.copy().cache();
        if self.nnz > 0 {
            let val = self.val.as_ref().expect("entries");
            if let Err(e) = out.put_add(&self.linear(), &-val) {
                out.poison(Arc::new(e));
            }
        }
        out
    }

    /// `A + s` as a dense matrix: every element, stored or not, plus the scalar.
    #[track_caller]
    pub fn add_scalar(&self, s: f32) -> Array<f32> {
        self.to_dense() + s
    }

    /// `A - s` as a dense matrix; see [`Sparse::add_scalar`].
    #[track_caller]
    pub fn sub_scalar(&self, s: f32) -> Array<f32> {
        self.to_dense() - s
    }
}

impl Mul<f32> for &Sparse<f32> {
    type Output = Sparse<f32>;

    /// [`Sparse::mul_scalar`].
    fn mul(self, s: f32) -> Sparse<f32> {
        self.mul_scalar(s)
    }
}

impl Div<f32> for &Sparse<f32> {
    type Output = Sparse<f32>;

    /// [`Sparse::div_scalar`].
    fn div(self, s: f32) -> Sparse<f32> {
        self.div_scalar(s)
    }
}

impl Neg for &Sparse<f32> {
    type Output = Sparse<f32>;

    /// `-A`, as [`Sparse::neg`].
    fn neg(self) -> Sparse<f32> {
        Sparse::neg(self)
    }
}

/// The sparse matrix of a dense one in `format`: [`Sparse::from_dense`] (ArrayFire's `sparse`).
#[track_caller]
pub fn sparse(dense: &Array<f32>, format: SparseFormat) -> Sparse<f32> {
    Sparse::from_dense(dense, format)
}
