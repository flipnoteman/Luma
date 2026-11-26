//! Batched matrix product: [`Array::matmul_batched`] and [`matmul_batched()`].
//!
//! The shapes are those of [`Array::matmul`] with the batch along axes 2 and 3: `lhs` is
//! `[K, M, B]`, `rhs` `[N, K, B]` and the result `[N, M, B]`, `C[b][n + N m] = Σ_k A[b][k + K m]
//! B[b][n + N k]`: `B` row-major `M x K` times `K x N` products, stored one after the other. An
//! operand whose batch axes are 1 is shared by all the products (a weight matrix applied to a
//! batch of inputs).
//!
//! The whole batch is one launch of a kernel in which a workgroup computes the output tile of
//! one product, or of several products when the tile is small, so that a launch of thousands of
//! 16x16 to 64x64 products is not thousands of launches. The tile is `4 x 4` outputs per
//! invocation of at most 64x64 outputs (an edge of 16, 32, 48 or 64, see [`edge`]): products of
//! up to 128 on a side take at most 4 tiles. Slices of 16 `k` of both operands are staged in
//! workgroup memory like the tiles of [`Array::matmul`] (see `matmul::body`), with the same
//! vec4 layout (four rows of `A`, four columns of `B` per vec4, so that one 16-byte read feeds
//! four accumulators). A single product, big products (the per-product kernels of
//! [`Array::matmul`] are better at 128x128 and beyond, with their split over K) are computed by
//! [`Array::matmul`], one launch each.
//!
//! On a backend with native matrix products (the CPU) a workgroup is a whole product, multiplied
//! by the native GEMM; the products are spread over the cores unless there are fewer products
//! than cores and they are big enough for the tiles of [`Array::matmul`] to do better.
//!
//! The operands are evaluated first (the kernel reads buffers). The launch is made when the product is called, like
//! [`Array::matmul`]'s, unless the call site is a lazy one (`LUMA_LAZY`, see `jit::structured::sites`): the product is then a value
//! that launches when it is used (`jit::structured::lazy`). A batch that is a loop of products is always made at the call, as a view
//! of one product per matrix, each of which is its own call site.

use std::panic::Location;
use std::sync::Arc;

use crate::array::{Array, Failure, Storage};
use crate::builtin::Builtin;
use crate::context::{Context, RawArg};
use crate::error::LumaError;
use crate::jit;
use crate::jit::structured::lazy;
use crate::ops::Outcome;

/// K elements of each operand a workgroup stages at a time.
const SLICE: u32 = 16;
/// Invocations a workgroup aims for when the tile is small: tiles of products that are a few
/// outputs per invocation are computed several to a workgroup.
const WORKGROUP: u32 = 64;
/// Longest side of a product the tiled kernel computes (see the module documentation).
const MAX_EDGE: u32 = 128;
/// Longest K of a product the tiled kernel computes: a long K of a few products is better split
/// over workgroups, which [`Array::matmul`] does.
const MAX_K: u32 = 1024;
/// Fewest FMAs of a matrix product the native GEMM spreads over the cores' tiles ([`Array::matmul`])
/// instead of giving each product to one core, when there are fewer products than cores.
const NATIVE_SPLIT_FMAS: u64 = 1 << 21;

/// The kernel of [`Builtin::Bmm`]: the output tile of one product is `tm x tn`, `4 x 4` outputs
/// per invocation; `aligned` if `M`, `N` and `K` are multiples of the tile and the slice, which
/// removes the guards of the loads and stores.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) struct Bmm {
    pub(crate) tm: u8,
    pub(crate) tn: u8,
    pub(crate) aligned: bool,
}

impl Bmm {
    /// Invocations that compute one tile.
    fn tile_threads(self) -> u32 {
        u32::from(self.tm) / 4 * (u32::from(self.tn) / 4)
    }

    /// Tiles a workgroup computes at once.
    fn tiles_per_group(self) -> u32 {
        WORKGROUP.div_ceil(self.tile_threads())
    }
}

/// The edge of the output tile of a side of `extent` outputs.
fn edge(extent: u32) -> u32 {
    match extent {
        0..=16 => 16,
        17..=32 => 32,
        33..=48 => 48,
        _ => 64,
    }
}

/// The kernel of an `m x k` by `k x n` product and the workgroups of `batch` of them.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct Launch {
    kernel: Bmm,
    groups: [u32; 3],
}

/// How the batch is computed.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Route {
    /// One launch.
    Batched(Launch),
    /// A product per matrix, by [`Array::matmul`].
    Looped,
}

/// The route of `batch` products of `m x k` by `k x n`. `cores` is the cores of a backend with
/// native products (0 if it has none), `max_groups` the most workgroups of a grid dimension.
fn route(m: u32, n: u32, k: u32, batch: u32, cores: u32, max_groups: u32) -> Result<Route, LumaError> {
    if batch == 1 {
        return Ok(Route::Looped);
    }
    if cores > 0 {
        let fmas = u64::from(m) * u64::from(n) * u64::from(k);
        if batch < cores && fmas > NATIVE_SPLIT_FMAS {
            return Ok(Route::Looped);
        }
        let [x, y, _] = crate::ops::fold_grid(batch, max_groups);
        if y > max_groups {
            return Err(LumaError::KernelGridTooLarge { workgroups: [x, y, 1], max: max_groups });
        }
        // The shader is not compiled for a native product: any tile names the kernel.
        return Ok(Route::Batched(Launch { kernel: Bmm { tm: 16, tn: 16, aligned: false }, groups: [x, y, 1] }));
    }
    if m > MAX_EDGE || n > MAX_EDGE || k > MAX_K {
        return Ok(Route::Looped);
    }
    let (tm, tn) = (edge(m), edge(n));
    let aligned = m.is_multiple_of(tm) && n.is_multiple_of(tn) && k.is_multiple_of(SLICE);
    let kernel = Bmm { tm: tm as u8, tn: tn as u8, aligned };
    let tiles = u64::from(m.div_ceil(tm)) * u64::from(n.div_ceil(tn)) * u64::from(batch);
    let groups = u32::try_from(tiles.div_ceil(u64::from(kernel.tiles_per_group()))).map_err(|_| LumaError::KernelGridTooLarge { workgroups: [u32::MAX, 1, 1], max: max_groups })?;
    let [x, y, _] = crate::ops::fold_grid(groups, max_groups);
    if y > max_groups {
        return Err(LumaError::KernelGridTooLarge { workgroups: [x, y, 1], max: max_groups });
    }
    Ok(Route::Batched(Launch { kernel, groups: [x, y, 1] }))
}

/// The sizes of a batched product.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Geometry {
    pub(crate) k: usize,
    pub(crate) m: usize,
    pub(crate) n: usize,
    /// Batch extents (axes 2 and 3) of the result.
    batch_dims: [usize; 2],
    /// Whether the operands have the batch (else one matrix is shared by every product).
    pub(crate) lhs_batched: bool,
    pub(crate) rhs_batched: bool,
}

impl Geometry {
    fn new(lhs: &Array<f32>, rhs: &Array<f32>) -> Result<Geometry, LumaError> {
        let (ld, rd) = (lhs.shape().dims(), rhs.shape().dims());
        let (lb, rb) = ([ld[2], ld[3]], [rd[2], rd[3]]);
        let unit = [1, 1];
        let (batch_dims, lhs_batched, rhs_batched) = match (lb == rb, lb == unit, rb == unit) {
            (true, ..) => (lb, lb != unit, rb != unit),
            (false, true, _) => (rb, false, true),
            (false, _, true) => (lb, true, false),
            _ => return Err(LumaError::ShapeMismatch { lhs: ld, rhs: rd }),
        };
        if ld[0] != rd[1] {
            return Err(LumaError::ShapeMismatch { lhs: ld, rhs: rd });
        }
        Ok(Geometry { k: ld[0], m: ld[1], n: rd[0], batch_dims, lhs_batched, rhs_batched })
    }

    pub(crate) fn batch(&self) -> usize {
        self.batch_dims[0] * self.batch_dims[1]
    }

    pub(crate) fn out_shape(&self) -> [usize; 4] {
        [self.n, self.m, self.batch_dims[0], self.batch_dims[1]]
    }
}

impl Array<f32> {
    /// The products of the matrices of `self` and of `rhs`, a batch along axes 2 and 3. `self` has
    /// shape `[K, M, B]`, `rhs` `[N, K, B]` and the result `[N, M, B]`: matrix `b` of the result
    /// is the [`matmul`](Array::matmul) of matrix `b` of each, that is, as row-major matrices,
    /// `A[b] (M x K) times B[b] (K x N)`. An operand with a batch of 1 (`[N, K]` or `[N, K, 1]`) is
    /// shared by all the products, like the weights of a layer applied to a batch of inputs; both
    /// batches are `[B]` or `[B2, B3]` (axes 2 and 3) and otherwise must be equal.
    ///
    /// The whole batch is one launch when the products are up to 128 on a side (and `K` up to
    /// 1024); bigger ones are one [`matmul_f32`](Array::matmul_f32) each: batched products are always in full `f32`
    /// precision, whatever the context's [`numerics`](crate::ContextOptions::numerics) say. The operands are evaluated
    /// first, and the product is recorded when this is called, like [`Array::matmul`]'s.
    ///
    /// # Errors
    ///
    /// The result is poisoned with [`LumaError::ShapeMismatch`] if `self.shape()[0] !=
    /// rhs.shape()[1]` or the batches differ (neither is 1), [`LumaError::ContextMismatch`] if
    /// the operands live on different contexts and [`LumaError::Chunked`] if an operand or the
    /// result needs more than one buffer.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use luma::Array;
    ///
    /// // Two products, [[1, 2], [3, 4]] @ [[1, 0], [0, 1]] and [[5, 6], [7, 8]] @ [[0, 1], [1, 0]].
    /// let a = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], [2, 2, 2])?;
    /// let b = Array::from_slice(&[1.0f32, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0], [2, 2, 2])?;
    /// assert_eq!(a.matmul_batched(&b).to_vec()?, [1.0, 2.0, 3.0, 4.0, 6.0, 5.0, 8.0, 7.0]);
    /// # Ok::<(), luma::LumaError>(())
    /// ```
    #[track_caller]
    pub fn matmul_batched(&self, rhs: &Array<f32>) -> Array<f32> {
        matmul_batched(self, rhs)
    }
}

/// [`Array::matmul_batched`] as a function.
#[track_caller]
pub fn matmul_batched(lhs: &Array<f32>, rhs: &Array<f32>) -> Array<f32> {
    let location = Location::caller();
    let geometry = match Geometry::new(lhs, rhs) {
        Ok(geometry) => geometry,
        Err(error) => return Array::resolve(lhs.context(), lhs.shape(), Err(error.into()), "matmul_batched", location),
    };
    let shape = geometry.out_shape();
    let shape = if shape.contains(&0) { lhs.shape() } else { shape.into() };
    if let Some(lazy) = lazy::batched(lhs, rhs, &geometry, shape, location) {
        return lazy;
    }
    match run(lhs, rhs, &geometry) {
        Ok(Some(out)) => out,
        Ok(None) => looped(lhs, rhs, &geometry),
        Err(failure) => Array::resolve(lhs.context(), shape, Err(failure), "matmul_batched", location),
    }
}

/// The product as one [`Array::matmul_f32`] per matrix of the batch, stacked along axis 2.
#[track_caller]
fn looped(lhs: &Array<f32>, rhs: &Array<f32>, g: &Geometry) -> Array<f32> {
    let batch = g.batch();
    if batch == 1 {
        return lhs.matmul_f32(rhs);
    }
    let a = lhs.reshape([g.k, g.m, if g.lhs_batched { batch } else { 1 }]);
    let b = rhs.reshape([g.n, g.k, if g.rhs_batched { batch } else { 1 }]);
    let pick = |x: &Array<f32>, batched: bool, i: usize| if batched { x.slice(2, i..=i) } else { x.clone() };
    let products: Vec<Array<f32>> = (0..batch).map(|i| pick(&a, g.lhs_batched, i).matmul_f32(&pick(&b, g.rhs_batched, i))).collect();
    crate::concat_slice(2, &products).reshape(g.out_shape())
}

/// How the batch is launched, `None` if [`route`] sends it to [`looped`]: pure (it reads the sizes, the backend's constants and
/// whether an operand is too big for a buffer), so a lazy product can decide it when it is made.
pub(crate) fn plan(lhs: &Array<f32>, rhs: &Array<f32>, g: &Geometry) -> Outcome<Option<Launch>> {
    let ctx = lhs.context();
    // Operands first, so that a poisoned one reports its own error.
    let too_large = |extent: usize| u32::try_from(extent).map_err(|_| Failure::from(LumaError::DimensionTooLarge { shape: g.out_shape() }));
    let (m, n, k, batch) = (too_large(g.m)?, too_large(g.n)?, too_large(g.k)?, too_large(g.batch())?);
    let (cores, max_groups) = (ctx.tuning().native_gemm, ctx.max_workgroups_per_dimension());
    let Route::Batched(launch) = route(m, n, k, batch, cores, max_groups)? else { return Ok(None) };
    for array in [lhs, rhs] {
        if array.pending().is_some_and(|node| node.len() as u64 * 4 > ctx.segment_bytes()) {
            return Ok(None);
        }
    }
    Ok(Some(launch))
}

/// The batched product as one launch, `None` if [`plan`] sends it to [`looped`].
fn run(lhs: &Array<f32>, rhs: &Array<f32>, g: &Geometry) -> Outcome<Option<Array<f32>>> {
    let ctx = lhs.context();
    let Some(launch) = plan(lhs, rhs, g)? else { return Ok(None) };
    let storage = execute(ctx, lhs, rhs, g, launch)?;
    Ok(Some(Array::from_storage(ctx, g.out_shape().into(), storage)))
}

/// Launches the batched product of `lhs` and `rhs` as `launch` says into a new buffer.
pub(crate) fn execute(ctx: &Context, lhs: &Array<f32>, rhs: &Array<f32>, g: &Geometry, launch: Launch) -> Outcome<Arc<Storage>> {
    let (m, n, k, batch) = (g.m as u32, g.n as u32, g.k as u32, g.batch() as u32);
    let (a, a_bytes) = lhs.ready()?;
    let (b, b_bytes) = rhs.ready()?;
    if !ctx.same_as(rhs.context()) {
        return Err(LumaError::ContextMismatch.into());
    }
    let out = Array::<f32>::alloc(ctx, g.out_shape().into())?;
    let out_storage = Arc::clone(out.storage()?);
    let out_pin = out_storage.pin_single()?;
    let out_bytes = out_storage.byte_len();
    let stride = |batched: bool, rows: u32, cols: u32| if batched { rows * cols } else { 0 };
    let words: [u32; 8] = [m, n, k, batch, stride(g.lhs_batched, m, k), stride(g.rhs_batched, k, n), 0, 0];
    let core = ctx.builtin(Builtin::Bmm(launch.kernel), "f32")?;
    ctx.launch(
        &core,
        launch.groups,
        [n, m, batch],
        &[RawArg::write(out_pin.buffer(), out_bytes), RawArg::read(a.buffer(), a_bytes), RawArg::read(b.buffer(), b_bytes), RawArg::Uniform(bytemuck::cast_slice(&words))],
    )?;
    jit::check_finite(ctx, &out_storage, jit::ElemTy::F32)?;
    Ok(out_storage)
}

/// The kernel of [`Builtin::Bmm`]: arguments out `c`, the buffers `a` and `b` and the uniform
/// `p` (`m`, `n`, `k`, `matrices`, the elements between the matrices of `a` and of `b`, 0 for a
/// shared one, and two words of padding).
///
/// A workgroup is `tiles_per_group` tiles of `tm x tn` outputs, a tile `tm / 4 x tn / 4`
/// invocations of 4x4 outputs; tile `t` of a product, then product by product, is numbered along
/// the folded grid. Slices of 16 `k` are staged as in `matmul::t128`: a vec4 of four rows of `A`
/// (columns of `B`) at one `k`, `A` k-major with a pad of one vec4 per `k` so that the staging
/// stores of neighbouring `k` do not meet in a bank. Tiles past the last product repeat the last
/// one (the loads stay in bounds, the barriers uniform) and do not store.
pub(crate) fn source(kernel: Bmm) -> String {
    let (tm, tn) = (u32::from(kernel.tm), u32::from(kernel.tn));
    let (row_groups, col_groups) = (tm / 4, tn / 4);
    let threads = kernel.tile_threads();
    let per_group = kernel.tiles_per_group();
    let aligned = kernel.aligned;
    let a_pitch = row_groups + 1;
    let a_slots = row_groups * SLICE;
    let b_slots = col_groups * SLICE;

    // Staging of slot `e` of the slices of `A` and `B`: four loads each, in range or zero.
    let stage = |slots: u32, body: &dyn Fn(&str) -> String| -> String {
        let mut out = String::new();
        for i in 0..slots.div_ceil(threads) {
            let guard = (i + 1) * threads > slots;
            let inner = body(&format!("lt + {}u", i * threads));
            out += &if guard { format!("    if (lt + {}u < {slots}u) {{\n{inner}    }}\n", i * threads) } else { format!("    {{\n{inner}    }}\n") };
        }
        out
    };
    let stage_a = stage(a_slots, &|e| {
        let loads: Vec<String> = (0..4)
            .map(|j| {
                if aligned {
                    format!("a[a_base + (r + {j}u) * p.k + k]")
                } else {
                    format!("select(0.0, a[a_base + min(r + {j}u, p.m - 1u) * p.k + kc], ok && r + {j}u < p.m)")
                }
            })
            .collect();
        let guards = if aligned { "" } else { "        let kc = min(k, p.k - 1u);\n        let ok = k < p.k;\n" };
        format!(
            "        let e = {e};\n        let k = k0 + e % {SLICE}u;\n        let r = row0 + (e / {SLICE}u) * 4u;\n{guards}        at[a_off + (e % {SLICE}u) * {a_pitch}u + e / {SLICE}u] = vec4<f32>({});\n",
            loads.join(", ")
        )
    });
    let stage_b = stage(b_slots, &|e| {
        let loads: Vec<String> = (0..4)
            .map(|j| {
                if aligned {
                    format!("b[row + c + {j}u]")
                } else {
                    format!("select(0.0, b[row + min(c + {j}u, p.n - 1u)], ok && c + {j}u < p.n)")
                }
            })
            .collect();
        let row = if aligned { "        let row = b_base + k * p.n;\n" } else { "        let row = b_base + min(k, p.k - 1u) * p.n;\n        let ok = k < p.k;\n" };
        format!(
            "        let e = {e};\n        let k = k0 + e / {col_groups}u;\n        let c = col0 + (e % {col_groups}u) * 4u;\n{row}        bt[b_off + (e / {col_groups}u) * {col_groups}u + e % {col_groups}u] = vec4<f32>({});\n",
            loads.join(", ")
        )
    });

    let mut decl = String::new();
    let mut update = String::new();
    let mut store = String::new();
    const LANES: [&str; 4] = ["x", "y", "z", "w"];
    for (i, row_lane) in LANES.iter().enumerate() {
        for (j, col_lane) in LANES.iter().enumerate() {
            decl += &format!("    var c{i}{j} = 0.0;\n");
            update += &format!("            c{i}{j} = fma(av.{row_lane}, bv.{col_lane}, c{i}{j});\n");
        }
        let writes: String = (0..4)
            .map(|j| {
                if aligned {
                    format!("            c[at_c + {j}u] = c{i}{j};\n")
                } else {
                    format!("            if (col + {j}u < p.n) {{ c[at_c + {j}u] = c{i}{j}; }}\n")
                }
            })
            .collect();
        let guard = if aligned { String::new() } else { format!("        if (row0 + ty * 4u + {i}u < p.m) {{\n") };
        let close = if aligned { "" } else { "        }\n" };
        store += &format!(
            "{guard}        {{\n            let at_c = c_base + (row0 + ty * 4u + {i}u) * p.n + col;\n{writes}        }}\n{close}"
        );
    }

    format!(
        "struct Params {{ m: u32, n: u32, k: u32, matrices: u32, a_stride: u32, b_stride: u32, pad0: u32, pad1: u32 }}
@group(1) @binding(0) var<storage, read_write> c: array<f32>;
@group(1) @binding(1) var<storage, read> a: array<f32>;
@group(1) @binding(2) var<storage, read> b: array<f32>;
@group(1) @binding(3) var<uniform> p: Params;

var<workgroup> at: array<vec4<f32>, {at_len}>;
var<workgroup> bt: array<vec4<f32>, {bt_len}>;

@compute @workgroup_size({workgroup})
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_index) t: u32) {{
    let slot = t / {threads}u;
    let lt = t % {threads}u;
    let tiles_n = (p.n + {tn_m1}u) / {tn}u;
    let per_matrix = ((p.m + {tm_m1}u) / {tm}u) * tiles_n;
    let first = wid.x + wid.y * nwg.x;
    let wanted = first * {per_group}u + slot;
    let total = p.matrices * per_matrix;
    let valid = wanted < total;
    let job = min(wanted, total - 1u);
    let tile = job % per_matrix;
    let matrix = job / per_matrix;
    let row0 = (tile / tiles_n) * {tm}u;
    let col0 = (tile % tiles_n) * {tn}u;
    let a_base = matrix * p.a_stride;
    let b_base = matrix * p.b_stride;
    let c_base = matrix * p.m * p.n;
    let ty = lt / {col_groups}u;
    let tx = lt % {col_groups}u;
    let a_off = slot * {a_len}u;
    let b_off = slot * {b_len}u;
{decl}    for (var k0 = 0u; k0 < p.k; k0 = k0 + {SLICE}u) {{
{stage_a}{stage_b}        workgroupBarrier();
        for (var kk = 0u; kk < {SLICE}u; kk = kk + 1u) {{
            let av = at[a_off + kk * {a_pitch}u + ty];
            let bv = bt[b_off + kk * {col_groups}u + tx];
{update}        }}
        workgroupBarrier();
    }}
    if (valid) {{
        let col = col0 + tx * 4u;
{store}    }}
}}
",
        at_len = per_group * SLICE * a_pitch,
        bt_len = per_group * SLICE * col_groups,
        workgroup = per_group * threads,
        tn_m1 = tn - 1,
        tm_m1 = tm - 1,
        a_len = SLICE * a_pitch,
        b_len = SLICE * col_groups,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiles_follow_the_extent() {
        assert_eq!([1, 16, 17, 32, 33, 48, 49, 64, 128].map(edge), [16, 16, 32, 32, 48, 48, 64, 64, 64]);
    }

    #[test]
    fn small_tiles_share_a_workgroup() {
        let kernel = |tm, tn| Bmm { tm, tn, aligned: false };
        assert_eq!([kernel(16, 16), kernel(32, 16), kernel(32, 32), kernel(64, 32), kernel(64, 64)].map(Bmm::tiles_per_group), [4, 2, 1, 1, 1]);
        assert_eq!(kernel(16, 48).tiles_per_group() * kernel(16, 48).tile_threads(), 96);
    }

    #[test]
    fn the_route_depends_on_the_size_and_the_backend() {
        let launch = |m, n, k, batch, cores| match route(m, n, k, batch, cores, 65535).unwrap() {
            Route::Batched(launch) => Some(launch),
            Route::Looped => None,
        };
        // One product, big products and a long K are matrix products of their own.
        assert_eq!(launch(32, 32, 32, 1, 0), None);
        assert_eq!(launch(129, 32, 32, 8, 0), None);
        assert_eq!(launch(32, 32, 1025, 8, 0), None);
        // 4096 products of 32x32x32: whole 32x32 tiles, 4096 workgroups.
        let l = launch(32, 32, 32, 4096, 0).unwrap();
        assert_eq!((l.kernel, l.groups), (Bmm { tm: 32, tn: 32, aligned: true }, [4096, 1, 1]));
        // 16x16 tiles four to a workgroup; K of 20 is not a whole slice.
        let l = launch(16, 16, 20, 4096, 0).unwrap();
        assert_eq!((l.kernel, l.groups), (Bmm { tm: 16, tn: 16, aligned: false }, [1024, 1, 1]));
        // A 100x100 product is 2x2 tiles of 64x64.
        assert_eq!(launch(100, 100, 64, 10, 0).unwrap().groups, [40, 1, 1]);
        // The grid folds past 65535 workgroups.
        let l = launch(32, 32, 32, 100_000, 0).unwrap();
        assert!(l.groups[1] > 1 && l.groups[0] <= 65535 && u64::from(l.groups[0]) * u64::from(l.groups[1]) >= 100_000);
        // Native products: a workgroup per product, unless few big products leave cores idle.
        assert_eq!(launch(128, 128, 128, 6, 8).map(|l| l.groups), Some([6, 1, 1]));
        assert_eq!(launch(512, 512, 512, 6, 8), None);
        assert_eq!(launch(512, 512, 512, 8, 8).map(|l| l.groups), Some([8, 1, 1]));
    }
}
