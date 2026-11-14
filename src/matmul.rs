//! Matrix product: [`Array::matmul`] and [`matmul()`].
//!
//! Luma's axis 0 varies fastest, so the shapes are those of NumPy's `a @ b` reversed: `lhs` is
//! `[K, M]`, `rhs` is `[N, K]` and the result `[N, M]`, `C[n + N m] = Σ_k A[k + K m] B[n + N k]`,
//! the row-major `C = A B` of `M x K` and `K x N` matrices.
//!
//! One kernel per product, in two sizes of output tile: 128x128 (16x16 invocations of 8x8
//! outputs each), or 32x32 (2x2 each) when `M` or `N` is below 128. Both stage 16-wide slices of
//! K of the two operands in workgroup memory, see [`body`]. A product with few output tiles and a
//! long K would leave most of the GPU idle: it is split over K, the slices written to a scratch
//! buffer by the third grid dimension and summed by a second launch ([`split_of`]).
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
use crate::jit::{self, MatmulSide};
use crate::ops::{launch_1d, Outcome};

/// Output tile of the kernel.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum MatmulTile {
    /// 128x128, each of the 16x16 invocations computes 8x8.
    T128,
    /// 32x32, each of the 16x16 invocations computes 2x2.
    T32,
}

impl MatmulTile {
    /// The edge of the tile.
    pub(crate) fn edge(self) -> u32 {
        match self {
            MatmulTile::T128 => 128,
            MatmulTile::T32 => 32,
        }
    }
}

/// K elements of each operand a workgroup stages at a time.
const SLICE: u32 = 16;

/// Output tiles below which a long K is split (see [`split_of`]).
const SPLIT_TILES: u32 = 128;
/// Shortest K that is split.
const SPLIT_MIN_K: u32 = 4096;
/// Workgroups a split product aims for. The 4 tiles of a 64x64 product over 2^20 take 3.0 ms
/// with 256 (3 workgroups per multiprocessor of an RTX 3090, which cannot hide the latency of
/// the slices), 2.5 with 512, 2.3 with 1024, 2.27 with 2048.
const SPLIT_GROUPS: u32 = 1024;
/// Smallest K a slice of a split product covers.
const SPLIT_MIN_SLICE: u32 = 1024;

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
    /// Slices of K, 1 if the product is not split.
    slices: u32,
    /// Elements of K per slice (K if not split).
    kc: u32,
    /// `[column tiles, row tiles, slices]`.
    groups: [u32; 3],
}

/// The slices a product of `tiles` output tiles over `k` is split into, and the elements of K
/// per slice (a multiple of the staged slice). Few tiles and a long K: about
/// [`SPLIT_GROUPS`] workgroups, each slice at least [`SPLIT_MIN_SLICE`] long. `None` if not split.
fn split_of(tiles: u32, k: u32) -> Option<(u32, u32)> {
    if tiles >= SPLIT_TILES || k < SPLIT_MIN_K {
        return None;
    }
    let wanted = SPLIT_GROUPS.div_ceil(tiles).min(k / SPLIT_MIN_SLICE);
    if wanted < 2 {
        return None;
    }
    let kc = k.div_ceil(wanted).next_multiple_of(SLICE);
    let slices = k.div_ceil(kc);
    (slices >= 2).then_some((slices, kc))
}

/// The plan of an `m x k` by `k x n` product. `max_groups` is the most workgroups per grid
/// dimension, `max_bytes` the most a buffer may hold (the split product's partial sums have to
/// fit one).
fn plan(m: u32, n: u32, k: u32, max_groups: u32, max_bytes: u64) -> Result<Plan, LumaError> {
    let tile = if m >= 128 && n >= 128 { MatmulTile::T128 } else { MatmulTile::T32 };
    let edge = tile.edge();
    let aligned = m.is_multiple_of(edge) && n.is_multiple_of(edge) && k.is_multiple_of(SLICE);
    let (columns, rows) = (n.div_ceil(edge), m.div_ceil(edge));
    let tiles = columns.saturating_mul(rows);
    let (slices, kc) = split_of(tiles, k)
        .filter(|&(slices, _)| u64::from(slices) * u64::from(m) * u64::from(n) * 4 <= max_bytes)
        .unwrap_or((1, k));
    let groups = [columns, rows, slices];
    if groups.iter().any(|&count| count > max_groups) {
        return Err(LumaError::KernelGridTooLarge { workgroups: groups, max: max_groups });
    }
    Ok(Plan { tile, aligned, slices, kc, groups })
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

fn run(lhs: &Array<f32>, rhs: &Array<f32>) -> Outcome<Array<f32>> {
    let ctx = lhs.context();
    // Operands first, so that a poisoned one reports its own error.
    let sides = [side(ctx, lhs)?, side(rhs.context(), rhs)?];
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
    let plan = plan(m32, n32, k32, ctx.max_workgroups_per_dimension(), ctx.segment_bytes())?;

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
    let kernel = jit::MatmulKernel { tile: plan.tile, aligned: plan.aligned, split: plan.slices > 1 };
    jit::matmul(ctx, sides, kernel, [m32, n32, k32, plan.kc], plan.groups, RawArg::write(target, target_bytes))?;
    if let Some(partial) = &partial {
        let sum = ctx.builtin(Builtin::MatmulSum, "f32")?;
        let slices = [plan.slices];
        launch_1d(
            ctx,
            &sum,
            m32 * n32,
            &[RawArg::write(out_pin.buffer(), out_bytes), RawArg::read(partial, target_bytes), RawArg::Uniform(bytemuck::bytes_of(&slices))],
        )?;
    }
    drop(scratch);
    jit::check_finite(ctx, &out_storage, jit::ElemTy::F32)?;
    Ok(out)
}

/// The kernel summing the partial products of a split product (see [`Builtin::MatmulSum`]).
pub(crate) fn sum_source(workgroup: u32) -> String {
    format!(
        "struct Params {{ s: u32 }}
@group(1) @binding(0) var<storage, read_write> out: array<f32>;
@group(1) @binding(1) var<storage, read> part: array<f32>;
@group(1) @binding(2) var<uniform> p: Params;
@compute @workgroup_size({workgroup})
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {{
    let i = gid.x + gid.y * nwg.x * {workgroup}u;
    if (i >= luma.size.x) {{ return; }}
    var acc = 0.0;
    for (var z = 0u; z < p.s; z = z + 1u) {{ acc = acc + part[z * luma.size.x + i]; }}
    out[i] = acc;
}}
"
    )
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
        MatmulTile::T32 => t32(aligned, split, &loads),
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
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {{
    let row0 = wid.y * 128u;
    let col0 = wid.x * 128u;
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
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {{
    let row0 = wid.y * 32u;
    let col0 = wid.x * 32u;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_long_product_of_few_tiles_is_split_in_slices_of_whole_k_steps() {
        // The Gram matrix of 64 features over 2^20 samples: 4 tiles of 32x32.
        let (slices, kc) = split_of(4, 1 << 20).unwrap();
        assert_eq!((slices, kc), (256, 4096));
        for k in [4096u32, 5000, 100_000, 1 << 20] {
            for tiles in [1u32, 4, 70, 127] {
                let Some((slices, kc)) = split_of(tiles, k) else { continue };
                assert!(slices >= 2 && kc % SLICE == 0, "{tiles} tiles, k {k}");
                assert_eq!(slices, k.div_ceil(kc), "no empty slice");
            }
        }
        assert_eq!(split_of(128, 1 << 20), None);
        assert_eq!(split_of(4, 4095), None);
        assert_eq!(split_of(4, 4096), Some((4, 1024)));
    }

    #[test]
    fn the_plan_picks_the_tile_alignment_and_grid() {
        let p = plan(1024, 2048, 1024, 65535, u64::MAX).unwrap();
        assert_eq!((p.tile, p.aligned, p.slices, p.groups), (MatmulTile::T128, true, 1, [16, 8, 1]));
        let p = plan(1000, 1000, 1000, 65535, u64::MAX).unwrap();
        assert_eq!((p.tile, p.aligned, p.groups), (MatmulTile::T128, false, [8, 8, 1]));
        let p = plan(64, 64, 1 << 20, 65535, u64::MAX).unwrap();
        assert_eq!((p.tile, p.aligned, p.slices, p.groups), (MatmulTile::T32, true, 256, [2, 2, 256]));
        // The partial sums must fit a buffer.
        assert_eq!(plan(64, 64, 1 << 20, 65535, 1000).unwrap().slices, 1);
        assert!(plan(1 << 22, 64, 16, 65535, u64::MAX).is_err());
    }
}
