//! Hand-written WGSL reference kernels for `bench_datasci`, run through luma's custom-kernel API.
//! Each one is written independently of the library implementation it is compared with.

use luma::*;

/// The tiled SGEMM of the benchmarks (`benches/common/sgemm.rs`), non-cooperative, no bias.
#[allow(dead_code)]
#[path = "../../benches/common/sgemm.rs"]
mod sgemm;

/// Row-major `C[M][N] = A[M][K] B[K][N]`, `M`, `N` multiples of 128 and `K` of 16.
#[allow(dead_code)]
pub struct HandSgemm(Kernel);

#[allow(dead_code)]
impl HandSgemm {
    pub fn new(ctx: &Context) -> Result<HandSgemm> {
        let builder = ctx.kernel(&sgemm::sgemm_source(false, false)).label("hand_sgemm");
        // `LUMA_HAND_UNCHECKED=1` builds the kernel without bounds checks (A/B of their cost).
        // SAFETY: the kernel only indexes inside the operands `run` sizes for it.
        let builder = if std::env::var_os("LUMA_HAND_UNCHECKED").is_some() { unsafe { builder.unchecked_bounds() } } else { builder };
        builder.build().map(HandSgemm)
    }

    /// Records `c = a b`; `c` must hold `m * n` elements.
    pub fn run(&self, a: &Array<f32>, b: &Array<f32>, c: &mut Array<f32>, m: usize, n: usize, k: usize) {
        let p = sgemm::SgemmParams { m: m as u32, n: n as u32, k: k as u32, relu: 0 };
        let tile = sgemm::TILE;
        self.0.launch_raw([(n / tile) as u32, (m / tile) as u32, 1], args![a, b, c, uniform(&p)]).unwrap();
    }
}

#[derive(encase::ShaderType)]
struct SaxpyParams {
    alpha: f32,
    n: u32,
}

/// `out = alpha * x + y`, one element per invocation on a 2D grid (the x extent is limited).
pub struct HandSaxpy(Kernel);

impl HandSaxpy {
    pub fn new(ctx: &Context) -> Result<HandSaxpy> {
        const SRC: &str = r#"
struct P { alpha: f32, n: u32 }
@group(1) @binding(0) var<storage, read> x: array<f32>;
@group(1) @binding(1) var<storage, read> y: array<f32>;
@group(1) @binding(2) var<storage, read_write> out: array<f32>;
@group(1) @binding(3) var<uniform> p: P;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
    let i = gid.x + gid.y * nw.x * 256u;
    if (i >= p.n) { return; }
    out[i] = p.alpha * x[i] + y[i];
}
"#;
        ctx.kernel(SRC).label("hand_saxpy").build().map(HandSaxpy)
    }

    pub fn run(&self, alpha: f32, x: &Array<f32>, y: &Array<f32>, out: &mut Array<f32>, n: usize) {
        let groups = n.div_ceil(256);
        let gx = groups.min(1 << 15);
        let gy = groups.div_ceil(gx);
        let p = SaxpyParams { alpha, n: n as u32 };
        self.0.launch_raw([gx as u32, gy as u32, 1], args![x, y, out, uniform(&p)]).unwrap();
    }
}

#[derive(encase::ShaderType)]
struct GramParams {
    f: u32,
    k: u32,
    kc: u32,
    slices: u32,
}

/// `C = Zᵀ Z` for `Z[k][f] = (X[f + F*k] - c[f]) * s[f]`, `X` of luma shape `[F, K]`, `F` a
/// multiple of 32: a row-major GEMM `A = Zᵀ` (F×K), `B = Z` (K×F) with the standardisation done
/// inline in the global loads. A 32×32 output tile per workgroup (16×16 invocations, 2×2 outputs
/// each) over 16-wide K slices staged in workgroup memory; the K range is split into `S` chunks
/// (grid z) written to a scratch of `S * F * F` partials that a second kernel sums into `C`.
/// The scratch is allocated on the first `run` and reused while `F` and `K` are unchanged.
pub struct HandGram {
    ctx: Context,
    gemm: Kernel,
    sum: Kernel,
    scratch: std::cell::RefCell<Option<(usize, Array<f32>)>>,
}

const GRAM_GEMM: &str = r#"
struct P { f: u32, k: u32, kc: u32, slices: u32 }
@group(1) @binding(0) var<storage, read> x: array<f32>;
@group(1) @binding(1) var<storage, read> mean: array<f32>;
@group(1) @binding(2) var<storage, read> scale: array<f32>;
@group(1) @binding(3) var<storage, read_write> partial: array<f32>;
@group(1) @binding(4) var<uniform> p: P;

var<workgroup> sa: array<f32, 512>; // [16][32]: Z[k0 + kk][m0 + m]
var<workgroup> sb: array<f32, 512>; // [16][32]: Z[k0 + kk][n0 + n]

fn z(row: u32, feature: u32) -> f32 {
    return (x[feature + p.f * row] - mean[feature]) * scale[feature];
}

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) tid: u32, @builtin(workgroup_id) wg: vec3<u32>) {
    let m0 = wg.y * 32u;
    let n0 = wg.x * 32u;
    let kbeg = wg.z * p.kc;
    let kend = min(kbeg + p.kc, p.k);
    let lm = tid & 31u;   // staged column: consecutive invocations read consecutive features
    let lk = tid >> 5u;   // staged row, plus 8 for the second load
    let tx = tid & 15u;
    let ty = tid >> 4u;
    var acc: array<f32, 4>;
    for (var k0 = kbeg; k0 < kend; k0 += 16u) {
        for (var h = 0u; h < 2u; h++) {
            let kk = lk + h * 8u;
            let k = k0 + kk;
            let ok = k < kend;
            let r = min(k, p.k - 1u);
            sa[kk * 32u + lm] = select(0.0, z(r, m0 + lm), ok);
            sb[kk * 32u + lm] = select(0.0, z(r, n0 + lm), ok);
        }
        workgroupBarrier();
        for (var kk = 0u; kk < 16u; kk++) {
            let a0 = sa[kk * 32u + ty];
            let a1 = sa[kk * 32u + ty + 16u];
            let b0 = sb[kk * 32u + tx];
            let b1 = sb[kk * 32u + tx + 16u];
            acc[0] += a0 * b0;
            acc[1] += a0 * b1;
            acc[2] += a1 * b0;
            acc[3] += a1 * b1;
        }
        workgroupBarrier();
    }
    let base = wg.z * p.f * p.f;
    for (var i = 0u; i < 2u; i++) {
        for (var j = 0u; j < 2u; j++) {
            partial[base + (m0 + ty + 16u * i) * p.f + n0 + tx + 16u * j] = acc[i * 2u + j];
        }
    }
}
"#;

const GRAM_SUM: &str = r#"
struct P { f: u32, k: u32, kc: u32, slices: u32 }
@group(1) @binding(0) var<storage, read> partial: array<f32>;
@group(1) @binding(1) var<storage, read_write> out: array<f32>;
@group(1) @binding(2) var<uniform> p: P;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let n = p.f * p.f;
    if (i >= n) { return; }
    var total = 0.0;
    for (var s = 0u; s < p.slices; s++) {
        total += partial[s * n + i];
    }
    out[i] = total;
}
"#;

impl HandGram {
    pub fn new(ctx: &Context) -> Result<HandGram> {
        Ok(HandGram {
            ctx: ctx.clone(),
            gemm: ctx.kernel(GRAM_GEMM).label("hand_gram").build()?,
            sum: ctx.kernel(GRAM_SUM).label("hand_gram_sum").build()?,
            scratch: Default::default(),
        })
    }

    /// Records `out = Zᵀ Z` (`out` holds `f * f` elements, row-major).
    pub fn run(&self, x: &Array<f32>, c: &Array<f32>, s: &Array<f32>, out: &mut Array<f32>, f: u32, k: u32) {
        assert!(f.is_multiple_of(32) && k > 0, "HandGram needs F a multiple of 32 and K > 0");
        let tiles = (f / 32) * (f / 32);
        let split = (1024u32.div_ceil(tiles)).min(k / 1024).max(1);
        let kc = k.div_ceil(split).next_multiple_of(16);
        let slices = k.div_ceil(kc);
        let need = (slices * f * f) as usize;
        let mut cache = self.scratch.borrow_mut();
        if cache.as_ref().is_none_or(|(len, _)| *len != need) {
            *cache = Some((need, Array::<f32>::zeros_on(&self.ctx, need).unwrap()));
        }
        let partial = &mut cache.as_mut().unwrap().1;
        let p = GramParams { f, k, kc, slices };
        self.gemm.launch_raw([f / 32, f / 32, slices], args![x, c, s, &mut *partial, uniform(&p)]).unwrap();
        let groups = (f * f).div_ceil(256);
        self.sum.launch_raw([groups, 1, 1], args![&*partial, out, uniform(&p)]).unwrap();
    }
}

/// Launch grid of `groups` workgroups of 256 on two dimensions (the x extent is limited); a kernel
/// recovers the element with `gid.x + gid.y * nw.x * 256`.
fn grid_2d(groups: usize) -> [u32; 3] {
    let gx = groups.min(1 << 15);
    [gx as u32, groups.div_ceil(gx) as u32, 1]
}

#[derive(encase::ShaderType)]
struct CountParam {
    n: u32,
}

/// Zeroes the first `n` words of an array of any 4-byte element type.
pub struct HandZero(Kernel);

impl HandZero {
    pub fn new(ctx: &Context) -> Result<HandZero> {
        const SRC: &str = r#"
struct P { n: u32 }
@group(1) @binding(0) var<storage, read_write> out: array<u32>;
@group(1) @binding(1) var<uniform> p: P;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
    let i = gid.x + gid.y * nw.x * 256u;
    if (i < p.n) { out[i] = 0u; }
}
"#;
        ctx.kernel(SRC).label("hand_zero").build().map(HandZero)
    }

    pub fn run<T: ScalarType>(&self, out: &mut Array<T>, n: usize) {
        self.0.launch_raw(grid_2d(n.div_ceil(256)), args![out, uniform(&CountParam { n: n as u32 })]).unwrap();
    }
}

#[derive(encase::ShaderType)]
struct AxisParams {
    cols: u32,
    rows: u32,
    rb: u32,
    blocks: u32,
    inv: f32,
}

/// Rows per block of the strided-axis passes: about `2^20` invocations (one per column and block),
/// at least 16 rows each.
fn axis_rows_per_block(cols: usize, rows: usize) -> usize {
    (cols * rows).div_ceil(1 << 20).max(16)
}

/// Combine kernel: `out[k][c] = inv * sum_b partial[k][b][c]`, one workgroup per 8 columns of each
/// of the `k` partial sets (32 lanes stride the blocks, then a tree reduction).
const AXIS_COMBINE: &str = r#"
struct P { cols: u32, rows: u32, rb: u32, blocks: u32, inv: f32 }
@group(1) @binding(0) var<storage, read> partial: array<f32>;
@group(1) @binding(1) var<storage, read_write> out: array<f32>;
@group(1) @binding(2) var<uniform> p: P;

var<workgroup> sh: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) tid: u32, @builtin(workgroup_id) wg: vec3<u32>) {
    let per = p.cols / 8u;
    let k = wg.x / per;
    let c = (wg.x % per) * 8u + (tid & 7u);
    var s = 0.0;
    for (var b = tid >> 3u; b < p.blocks; b += 32u) {
        s += partial[(k * p.blocks + b) * p.cols + c];
    }
    sh[tid] = s;
    workgroupBarrier();
    for (var h = 128u; h >= 8u; h >>= 1u) {
        if (tid < h) { sh[tid] += sh[tid + h]; }
        workgroupBarrier();
    }
    if (tid < 8u) { out[k * p.cols + c] = sh[tid] * p.inv; }
}
"#;

/// Header and block walk shared by the strided-axis pass kernels: invocation `g` owns column
/// `g % cols` and rows `[rb * (g / cols), rb * (g / cols + 1))`, so neighbouring invocations read
/// neighbouring columns of a row.
const AXIS_PASS_HEAD: &str = r#"
struct P { cols: u32, rows: u32, rb: u32, blocks: u32, inv: f32 }
@group(1) @binding(0) var<storage, read> x: array<f32>;
"#;

fn axis_pass(extra_bindings: &str, setup: &str, body: &str, store: &str) -> String {
    format!(
        "{AXIS_PASS_HEAD}{extra_bindings}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let g = gid.x;
    if (g >= p.cols * p.blocks) {{ return; }}
    let c = g % p.cols;
    let blk = g / p.cols;
    let r0 = blk * p.rb;
    let r1 = min(r0 + p.rb, p.rows);
    {setup}
    for (var r = r0; r < r1; r++) {{
        let v = x[c + p.cols * r];
        {body}
    }}
    {store}
}}
"
    )
}

/// Reusable partial-sum scratch, reallocated when the size changes.
struct Scratch {
    ctx: Context,
    buf: std::cell::RefCell<Option<(usize, Array<f32>)>>,
}

impl Scratch {
    fn new(ctx: &Context) -> Scratch {
        Scratch { ctx: ctx.clone(), buf: Default::default() }
    }

    fn get(&self, len: usize) -> std::cell::RefMut<'_, Array<f32>> {
        let mut cache = self.buf.borrow_mut();
        if cache.as_ref().is_none_or(|(n, _)| *n != len) {
            *cache = Some((len, Array::<f32>::zeros_on(&self.ctx, len).unwrap()));
        }
        std::cell::RefMut::map(cache, |c| &mut c.as_mut().unwrap().1)
    }
}

fn axis_geometry(cols: usize, rows: usize, inv: f32) -> (AxisParams, u32, usize) {
    assert!(cols.is_multiple_of(8) && rows > 0, "axis references need cols a multiple of 8 and rows > 0");
    let rb = axis_rows_per_block(cols, rows);
    let blocks = rows.div_ceil(rb);
    let groups = (cols * blocks).div_ceil(256) as u32;
    (AxisParams { cols: cols as u32, rows: rows as u32, rb: rb as u32, blocks: blocks as u32, inv }, groups, blocks)
}

/// Per column `c` of `X` (luma shape `[cols, rows]`): `sum_r X w[c]` and `sum_r (X w[c])^2`.
/// One pass, each invocation owning a column and a block of rows and keeping both sums in
/// registers, writes partials `[2][blocks][cols]`; a combine kernel sums them into
/// `out = [sums (cols), sums of squares (cols)]`.
pub struct HandAxisSumSq {
    pass: Kernel,
    combine: Kernel,
    scratch: Scratch,
}

impl HandAxisSumSq {
    pub fn new(ctx: &Context) -> Result<HandAxisSumSq> {
        let src = axis_pass(
            "@group(1) @binding(1) var<storage, read> w: array<f32>;
@group(1) @binding(2) var<storage, read_write> partial: array<f32>;
@group(1) @binding(3) var<uniform> p: P;",
            "var s = 0.0; var q = 0.0; let wc = w[c];",
            "let y = v * wc; s += y; q += y * y;",
            "partial[blk * p.cols + c] = s; partial[(p.blocks + blk) * p.cols + c] = q;",
        );
        Ok(HandAxisSumSq {
            pass: ctx.kernel(&src).label("hand_axis_sum_sq").build()?,
            combine: ctx.kernel(AXIS_COMBINE).label("hand_axis_sum_sq_combine").build()?,
            scratch: Scratch::new(ctx),
        })
    }

    pub fn run(&self, x: &Array<f32>, w: &Array<f32>, out: &mut Array<f32>, cols: usize, rows: usize) {
        let (p, groups, blocks) = axis_geometry(cols, rows, 1.0);
        let mut partial = self.scratch.get(2 * blocks * cols);
        self.pass.launch_raw([groups, 1, 1], args![x, w, &mut *partial, uniform(&p)]).unwrap();
        self.combine.launch_raw([(2 * cols / 8) as u32, 1, 1], args![&*partial, out, uniform(&p)]).unwrap();
    }
}

/// Population variance per column of `X` (luma shape `[cols, rows]`), two-pass: per-column sums
/// (the decomposition of [`HandAxisSumSq`]) combined into the mean, then per-column sums of
/// `(x - mean)^2` combined and divided by `rows`. Four kernels.
pub struct HandAxisVar {
    sum: Kernel,
    dev: Kernel,
    combine: Kernel,
    scratch: Scratch,
    mean: Scratch,
}

impl HandAxisVar {
    pub fn new(ctx: &Context) -> Result<HandAxisVar> {
        let sum = axis_pass(
            "@group(1) @binding(1) var<storage, read_write> partial: array<f32>;
@group(1) @binding(2) var<uniform> p: P;",
            "var s = 0.0;",
            "s += v;",
            "partial[blk * p.cols + c] = s;",
        );
        let dev = axis_pass(
            "@group(1) @binding(1) var<storage, read> mean: array<f32>;
@group(1) @binding(2) var<storage, read_write> partial: array<f32>;
@group(1) @binding(3) var<uniform> p: P;",
            "var s = 0.0; let m = mean[c];",
            "let d = v - m; s += d * d;",
            "partial[blk * p.cols + c] = s;",
        );
        Ok(HandAxisVar {
            sum: ctx.kernel(&sum).label("hand_axis_var_sum").build()?,
            dev: ctx.kernel(&dev).label("hand_axis_var_dev").build()?,
            combine: ctx.kernel(AXIS_COMBINE).label("hand_axis_var_combine").build()?,
            scratch: Scratch::new(ctx),
            mean: Scratch::new(ctx),
        })
    }

    /// Records `out[c] = var_r X[c, r]`; `out` holds `cols` elements.
    pub fn run(&self, x: &Array<f32>, out: &mut Array<f32>, cols: usize, rows: usize) {
        let (p, groups, blocks) = axis_geometry(cols, rows, 1.0 / rows as f32);
        let mut partial = self.scratch.get(blocks * cols);
        let mut mean = self.mean.get(cols);
        let combine = [(cols / 8) as u32, 1, 1];
        self.sum.launch_raw([groups, 1, 1], args![x, &mut *partial, uniform(&p)]).unwrap();
        self.combine.launch_raw(combine, args![&*partial, &mut *mean, uniform(&p)]).unwrap();
        self.dev.launch_raw([groups, 1, 1], args![x, &*mean, &mut *partial, uniform(&p)]).unwrap();
        self.combine.launch_raw(combine, args![&*partial, out, uniform(&p)]).unwrap();
    }
}

#[derive(encase::ShaderType)]
struct BinParams {
    n: u32,
    lo: f32,
    scale: f32,
}

/// Workgroups of the privatised kernels: `count / (4 * bins)` clamped to `[1, cap]`, `cap` 1024
/// for native integer atomics and 256 for the `f32` CAS loop, whose merge into global memory
/// serialises on contended words (G=1024 groups: 2.6 ms at 1024 workgroups, 0.51 ms at 256).
fn private_workgroups(count: usize, bins: usize, cas: bool) -> u32 {
    (count / (4 * bins)).clamp(1, if cas { 256 } else { 1024 }) as u32
}

/// Histogram of `bucket(x) = min(u32(clamp((x - lo) * s, 0, 4294967040)) [NaN -> 0], bins - 1)`
/// over an `f32` array, privatised per workgroup: the workgroup zeroes `var<workgroup>
/// array<atomic<u32>, BINS>`, grid-strides over its share of the input counting with workgroup
/// atomics, then merges each non-zero bin into the global counts with one `atomicAdd`.
/// `run` adds into `out` (zero it first, see [`HandZero`]).
pub struct HandHist {
    kernel: Kernel,
    bins: usize,
}

impl HandHist {
    pub fn new(ctx: &Context, bins: usize) -> Result<HandHist> {
        let src = format!(
            r#"
const BINS = {bins}u;
struct P {{ n: u32, lo: f32, scale: f32 }}
@group(1) @binding(0) var<storage, read> x: array<f32>;
@group(1) @binding(1) var<storage, read_write> out: array<atomic<u32>>;
@group(1) @binding(2) var<uniform> p: P;

var<workgroup> acc: array<atomic<u32>, BINS>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) tid: u32, @builtin(workgroup_id) wg: vec3<u32>,
        @builtin(num_workgroups) nw: vec3<u32>) {{
    for (var b = tid; b < BINS; b += 256u) {{ atomicStore(&acc[b], 0u); }}
    workgroupBarrier();
    let stride = nw.x * 256u;
    for (var i = wg.x * 256u + tid; i < p.n; i += stride) {{
        let t = (x[i] - p.lo) * p.scale;
        let u = select(0u, u32(clamp(t, 0.0, 4294967040.0)), t == t);
        atomicAdd(&acc[min(u, BINS - 1u)], 1u);
    }}
    workgroupBarrier();
    for (var b = tid; b < BINS; b += 256u) {{
        let v = atomicLoad(&acc[b]);
        if (v != 0u) {{ atomicAdd(&out[b], v); }}
    }}
}}
"#
        );
        Ok(HandHist { kernel: ctx.kernel(&src).label(&format!("hand_hist{bins}")).build()?, bins })
    }

    /// `out` holds `bins` `u32` counts.
    pub fn run(&self, x: &Array<f32>, out: &mut Array<u32>, n: usize, lo: f32, scale: f32) {
        let p = BinParams { n: n as u32, lo, scale };
        self.kernel.launch_raw([private_workgroups(n, self.bins, false), 1, 1], args![x, out, uniform(&p)]).unwrap();
    }
}

/// `out[idx[i]] += val[i]` for `f32` bins, one pair per invocation, with a global compare-and-swap
/// loop on `atomic<u32>` bit patterns that carries the observed value across retries; indices
/// `>= len` are skipped. `run` adds into `out` (zero it first).
pub struct HandScatterAddF32(Kernel);

#[derive(encase::ShaderType)]
struct ScatterParams {
    n: u32,
    len: u32,
}

impl HandScatterAddF32 {
    pub fn new(ctx: &Context) -> Result<HandScatterAddF32> {
        const SRC: &str = r#"
struct P { n: u32, len: u32 }
@group(1) @binding(0) var<storage, read> idx: array<u32>;
@group(1) @binding(1) var<storage, read> val: array<f32>;
@group(1) @binding(2) var<storage, read_write> out: array<atomic<u32>>;
@group(1) @binding(3) var<uniform> p: P;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
    let i = gid.x + gid.y * nw.x * 256u;
    if (i >= p.n) { return; }
    let j = idx[i];
    if (j >= p.len) { return; }
    let v = val[i];
    var old = atomicLoad(&out[j]);
    loop {
        let r = atomicCompareExchangeWeak(&out[j], old, bitcast<u32>(bitcast<f32>(old) + v));
        if (r.exchanged) { break; }
        old = r.old_value;
    }
}
"#;
        ctx.kernel(SRC).label("hand_scatter_add_f32").build().map(HandScatterAddF32)
    }

    pub fn run(&self, idx: &Array<u32>, val: &Array<f32>, out: &mut Array<f32>, n: usize, len: usize) {
        let p = ScatterParams { n: n as u32, len: len as u32 };
        self.0.launch_raw(grid_2d(n.div_ceil(256)), args![idx, val, out, uniform(&p)]).unwrap();
    }
}

/// `f32` group sum `out[codes[i]] += vals[i]` (codes `>= groups` skipped) in one kernel: with
/// `groups <= 4096`, privatised per workgroup (workgroup CAS-loop accumulation over a grid-strided
/// share, then a global CAS-loop merge of each non-zero bin); otherwise the global CAS loop, one
/// row per invocation. `run` adds into `out` (zero it first).
pub struct HandGroupSum {
    kernel: Kernel,
    groups: usize,
}

const GROUP_CAS: &str = r#"
fn cas_add_global(j: u32, v: f32) {
    var old = atomicLoad(&out[j]);
    loop {
        let r = atomicCompareExchangeWeak(&out[j], old, bitcast<u32>(bitcast<f32>(old) + v));
        if (r.exchanged) { break; }
        old = r.old_value;
    }
}
"#;

impl HandGroupSum {
    pub fn new(ctx: &Context, groups: usize) -> Result<HandGroupSum> {
        let head = r#"
struct P { n: u32, groups: u32 }
@group(1) @binding(0) var<storage, read> codes: array<u32>;
@group(1) @binding(1) var<storage, read> vals: array<f32>;
@group(1) @binding(2) var<storage, read_write> out: array<atomic<u32>>;
@group(1) @binding(3) var<uniform> p: P;
"#;
        let src = if groups <= 4096 {
            format!(
                r#"{head}{GROUP_CAS}
const GROUPS = {groups}u;
var<workgroup> acc: array<atomic<u32>, GROUPS>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) tid: u32, @builtin(workgroup_id) wg: vec3<u32>,
        @builtin(num_workgroups) nw: vec3<u32>) {{
    for (var b = tid; b < GROUPS; b += 256u) {{ atomicStore(&acc[b], 0u); }}
    workgroupBarrier();
    let stride = nw.x * 256u;
    for (var i = wg.x * 256u + tid; i < p.n; i += stride) {{
        let j = codes[i];
        if (j >= GROUPS) {{ continue; }}
        let v = vals[i];
        var old = atomicLoad(&acc[j]);
        loop {{
            let r = atomicCompareExchangeWeak(&acc[j], old, bitcast<u32>(bitcast<f32>(old) + v));
            if (r.exchanged) {{ break; }}
            old = r.old_value;
        }}
    }}
    workgroupBarrier();
    for (var b = tid; b < GROUPS; b += 256u) {{
        let s = atomicLoad(&acc[b]);
        if (s != 0u) {{ cas_add_global(b, bitcast<f32>(s)); }}
    }}
}}
"#
            )
        } else {
            format!(
                r#"{head}{GROUP_CAS}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {{
    let i = gid.x + gid.y * nw.x * 256u;
    if (i >= p.n) {{ return; }}
    let j = codes[i];
    if (j >= p.groups) {{ return; }}
    cas_add_global(j, vals[i]);
}}
"#
            )
        };
        Ok(HandGroupSum { kernel: ctx.kernel(&src).label(&format!("hand_group_sum{groups}")).build()?, groups })
    }

    pub fn run(&self, codes: &Array<u32>, vals: &Array<f32>, out: &mut Array<f32>, n: usize) {
        let p = ScatterParams { n: n as u32, len: self.groups as u32 };
        let grid = if self.groups <= 4096 { [private_workgroups(n, self.groups, true), 1, 1] } else { grid_2d(n.div_ceil(256)) };
        self.kernel.launch_raw(grid, args![codes, vals, out, uniform(&p)]).unwrap();
    }
}

#[derive(encase::ShaderType)]
struct CompactParams {
    len: u32,
    nwords: u32,
    tiles: u32,
}

/// Stream compaction of an `f32` array by a bit-packed mask (`Array<bool>`): three kernels.
/// Tiles of 8192 elements (a workgroup of 256 takes one mask word each): (1) per-tile popcounts,
/// (2) one workgroup turns the `tiles <= 4096` counts into exclusive offsets, (3) each tile scans
/// its 256 word counts in workgroup memory, then 256 invocations walk the tile 256 elements at a
/// time so that loads and stores are consecutive, writing every selected element to
/// `offset + rank`. The scratch for the counts is allocated on the first `run` and reused while
/// the size is unchanged.
pub struct HandCompact {
    count: Kernel,
    offsets: Kernel,
    scatter: Kernel,
    scratch: std::cell::RefCell<Option<(usize, Array<u32>)>>,
}

impl HandCompact {
    pub fn new(ctx: &Context) -> Result<HandCompact> {
        const COUNT: &str = r#"
struct P { len: u32, nwords: u32, tiles: u32 }
@group(1) @binding(0) var<storage, read_write> tile_count: array<u32>;
@group(1) @binding(1) var<storage, read> bits: array<u32>;
@group(1) @binding(2) var<uniform> p: P;
var<workgroup> part: array<u32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) tid: u32, @builtin(workgroup_id) wg: vec3<u32>,
        @builtin(num_workgroups) nw: vec3<u32>) {
    let tile = wg.x + wg.y * nw.x;
    if (tile >= p.tiles) { return; }
    let word = tile * 256u + tid;
    part[tid] = select(0u, countOneBits(bits[word]), word < p.nwords);
    workgroupBarrier();
    var width = 128u;
    loop {
        if (tid < width) { part[tid] += part[tid + width]; }
        workgroupBarrier();
        if (width == 1u) { break; }
        width = width >> 1u;
    }
    if (tid == 0u) { tile_count[tile] = part[0]; }
}
"#;
        const OFFSETS: &str = r#"
struct P { len: u32, nwords: u32, tiles: u32 }
@group(1) @binding(0) var<storage, read_write> tile_count: array<u32>;
@group(1) @binding(1) var<uniform> p: P;
var<workgroup> sums: array<u32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) tid: u32) {
    // 16 consecutive tiles per invocation.
    var run = 0u;
    for (var j = 0u; j < 16u; j++) {
        let t = tid * 16u + j;
        if (t < p.tiles) { run += tile_count[t]; }
    }
    sums[tid] = run;
    workgroupBarrier();
    for (var d = 1u; d < 256u; d = d << 1u) {
        var v = sums[tid];
        if (tid >= d) { v += sums[tid - d]; }
        workgroupBarrier();
        sums[tid] = v;
        workgroupBarrier();
    }
    var base = sums[tid] - run;
    for (var j = 0u; j < 16u; j++) {
        let t = tid * 16u + j;
        if (t < p.tiles) {
            let c = tile_count[t];
            tile_count[t] = base;
            base += c;
        }
    }
}
"#;
        const SCATTER: &str = r#"
struct P { len: u32, nwords: u32, tiles: u32 }
@group(1) @binding(0) var<storage, read_write> out: array<f32>;
@group(1) @binding(1) var<storage, read> x: array<f32>;
@group(1) @binding(2) var<storage, read> bits: array<u32>;
@group(1) @binding(3) var<storage, read> offset: array<u32>;
@group(1) @binding(4) var<uniform> p: P;
var<workgroup> words: array<u32, 256>;
var<workgroup> scan: array<u32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) tid: u32, @builtin(workgroup_id) wg: vec3<u32>,
        @builtin(num_workgroups) nw: vec3<u32>) {
    let tile = wg.x + wg.y * nw.x;
    if (tile >= p.tiles) { return; }
    let word = tile * 256u + tid;
    let w = select(0u, bits[word], word < p.nwords);
    words[tid] = w;
    let own = countOneBits(w);
    scan[tid] = own;
    workgroupBarrier();
    for (var d = 1u; d < 256u; d = d << 1u) {
        var v = scan[tid];
        if (tid >= d) { v += scan[tid - d]; }
        workgroupBarrier();
        scan[tid] = v;
        workgroupBarrier();
    }
    // Rank of the first element of word `k` of the tile: offset + scan[k] - count(k).
    let origin = offset[tile];
    let lane = tid & 31u;
    let lower = (1u << lane) - 1u;
    for (var step = 0u; step < 32u; step++) {
        let k = step * 8u + (tid >> 5u);
        let wk = words[k];
        if (((wk >> lane) & 1u) != 0u) {
            let before = scan[k] - countOneBits(wk);
            out[origin + before + countOneBits(wk & lower)] = x[tile * 8192u + step * 256u + tid];
        }
    }
}
"#;
        Ok(HandCompact {
            count: ctx.kernel(COUNT).label("hand_compact_count").build()?,
            offsets: ctx.kernel(OFFSETS).label("hand_compact_offsets").build()?,
            scatter: ctx.kernel(SCATTER).label("hand_compact_scatter").build()?,
            scratch: std::cell::RefCell::new(None),
        })
    }

    /// Records `out[..count] = x[mask]`; `n <= 4096 * 8192` elements.
    pub fn run(&self, ctx: &Context, x: &Array<f32>, mask: &Array<bool>, out: &mut Array<f32>, n: usize) {
        let tiles = n.div_ceil(8192);
        assert!(tiles <= 4096, "HandCompact: at most 4096 tiles");
        let mut scratch = self.scratch.borrow_mut();
        if scratch.as_ref().is_none_or(|(t, _)| *t != tiles) {
            *scratch = Some((tiles, Array::<u32>::zeros_on(ctx, tiles).unwrap()));
        }
        let counts = &mut scratch.as_mut().unwrap().1;
        let p = CompactParams { len: n as u32, nwords: n.div_ceil(32) as u32, tiles: tiles as u32 };
        let grid = grid_2d(tiles);
        self.count.launch_raw(grid, args![&mut *counts, mask, uniform(&p)]).unwrap();
        self.offsets.launch_raw([1, 1, 1], args![&mut *counts, uniform(&p)]).unwrap();
        self.scatter.launch_raw(grid, args![out, x, mask, &*counts, uniform(&p)]).unwrap();
    }
}

/// The order-preserving `u32` key of an `f32` for the arg-reduction references: larger floats get
/// larger keys, `-0.0` and `0.0` share one, and every NaN is the largest key (an `argmax` returns
/// the first NaN, as NumPy does). Ties are broken by the smaller index by the callers.
const ARG_KEY: &str = r#"
fn key(x: f32) -> u32 {
    let b = select(bitcast<u32>(x), 0u, x == 0.0);
    if ((b & 0x7fffffffu) > 0x7f800000u) { return 0xffffffffu; }
    return select(b | 0x80000000u, ~b, (b & 0x80000000u) != 0u);
}
"#;

#[derive(encase::ShaderType)]
struct ArgmaxParams {
    n: u32,
}

/// Workgroups of [`HandArgmax`]'s first kernel (the best of 256..8192 measured on an RTX 3090).
const ARGMAX_GROUPS: u32 = 4096;

/// `argmax` of a whole `f32` array: a grid-stride loop per invocation keeping the best (key,
/// index), a shared-memory tree per workgroup, then one workgroup combining the `ARGMAX_GROUPS`
/// partial pairs. `out` receives the index of the first largest element (the first NaN if any).
pub struct HandArgmax {
    pass: Kernel,
    combine: Kernel,
    partial: Array<u32>,
}

impl HandArgmax {
    pub fn new(ctx: &Context) -> Result<HandArgmax> {
        let groups = ARGMAX_GROUPS;
        let pass = format!(
            r#"{ARG_KEY}
struct P {{ n: u32 }}
@group(1) @binding(0) var<storage, read> x: array<f32>;
@group(1) @binding(1) var<storage, read_write> partial: array<u32>;
@group(1) @binding(2) var<uniform> p: P;

var<workgroup> shk: array<u32, 256>;
var<workgroup> shi: array<u32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) tid: u32, @builtin(workgroup_id) wg: vec3<u32>,
        @builtin(num_workgroups) nw: vec3<u32>) {{
    let stride = nw.x * 256u;
    var bk = 0u;
    var bi = 0u;
    for (var i = wg.x * 256u + tid; i < p.n; i += stride) {{
        let k = key(x[i]);
        if (k > bk) {{ bk = k; bi = i; }}
    }}
    shk[tid] = bk;
    shi[tid] = bi;
    workgroupBarrier();
    for (var h = 128u; h > 0u; h >>= 1u) {{
        if (tid < h) {{
            let k = shk[tid + h];
            let i = shi[tid + h];
            if (k > shk[tid] || (k == shk[tid] && i < shi[tid])) {{ shk[tid] = k; shi[tid] = i; }}
        }}
        workgroupBarrier();
    }}
    if (tid == 0u) {{
        partial[wg.x] = shk[0];
        partial[nw.x + wg.x] = shi[0];
    }}
}}
"#
        );
        let combine = format!(
            r#"struct P {{ n: u32 }}
@group(1) @binding(0) var<storage, read> partial: array<u32>;
@group(1) @binding(1) var<storage, read_write> out: array<u32>;
@group(1) @binding(2) var<uniform> p: P;

var<workgroup> shk: array<u32, 256>;
var<workgroup> shi: array<u32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) tid: u32) {{
    var bk = 0u;
    var bi = 0u;
    for (var g = tid; g < {groups}u; g += 256u) {{
        let k = partial[g];
        let i = partial[{groups}u + g];
        if (k > bk || (k == bk && i < bi)) {{ bk = k; bi = i; }}
    }}
    shk[tid] = bk;
    shi[tid] = bi;
    workgroupBarrier();
    for (var h = 128u; h > 0u; h >>= 1u) {{
        if (tid < h) {{
            let k = shk[tid + h];
            let i = shi[tid + h];
            if (k > shk[tid] || (k == shk[tid] && i < shi[tid])) {{ shk[tid] = k; shi[tid] = i; }}
        }}
        workgroupBarrier();
    }}
    if (tid == 0u && p.n != 0u) {{ out[0] = shi[0]; }}
}}
"#
        );
        Ok(HandArgmax {
            pass: ctx.kernel(&pass).label("hand_argmax").build()?,
            combine: ctx.kernel(&combine).label("hand_argmax_combine").build()?,
            partial: Array::<u32>::zeros_on(ctx, 2 * groups as usize).unwrap(),
        })
    }

    /// Records `out[0] = argmax(x[..n])`.
    pub fn run(&mut self, x: &Array<f32>, out: &mut Array<u32>, n: usize) {
        let p = ArgmaxParams { n: n as u32 };
        self.pass.launch_raw([ARGMAX_GROUPS, 1, 1], args![x, &mut self.partial, uniform(&p)]).unwrap();
        self.combine.launch_raw([1, 1, 1], args![&self.partial, out, uniform(&p)]).unwrap();
    }
}

/// Combine kernel of [`HandArgmaxAxis`]: `out[c]` is the index of the best (key, index) pair of
/// the column's `blocks` partial pairs, which are contiguous (`[cols][blocks]`); one workgroup
/// per column, its 256 invocations stride the blocks, then a tree.
const ARG_AXIS_COMBINE: &str = r#"
struct P { cols: u32, rows: u32, rb: u32, blocks: u32, inv: f32 }
@group(1) @binding(0) var<storage, read> pk: array<u32>;
@group(1) @binding(1) var<storage, read> pi: array<u32>;
@group(1) @binding(2) var<storage, read_write> out: array<u32>;
@group(1) @binding(3) var<uniform> p: P;

var<workgroup> shk: array<u32, 256>;
var<workgroup> shi: array<u32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) tid: u32, @builtin(workgroup_id) wg: vec3<u32>) {
    let c = wg.x;
    var bk = 0u;
    var bi = 0u;
    for (var b = tid; b < p.blocks; b += 256u) {
        let k = pk[c * p.blocks + b];
        let i = pi[c * p.blocks + b];
        if (k > bk || (k == bk && i < bi)) { bk = k; bi = i; }
    }
    shk[tid] = bk;
    shi[tid] = bi;
    workgroupBarrier();
    for (var h = 128u; h > 0u; h >>= 1u) {
        if (tid < h) {
            let k = shk[tid + h];
            let i = shi[tid + h];
            if (k > shk[tid] || (k == shk[tid] && i < shi[tid])) { shk[tid] = k; shi[tid] = i; }
        }
        workgroupBarrier();
    }
    if (tid == 0u) { out[c] = shi[0]; }
}
"#;

/// Per column `c` of `X` (luma shape `[cols, rows]`), the row of the first largest element (the
/// first NaN if any): the strided-axis decomposition of [`HandAxisSumSq`] (an invocation per
/// column and block of rows, neighbours on neighbouring columns) keeping the best (key, row) in
/// registers, partial pairs written column-major (`[cols][blocks]`), and a combine kernel.
pub struct HandArgmaxAxis {
    pass: Kernel,
    combine: Kernel,
    keys: ScratchU32,
    rows: ScratchU32,
}

/// [`Scratch`] of `u32`s.
struct ScratchU32 {
    ctx: Context,
    buf: std::cell::RefCell<Option<(usize, Array<u32>)>>,
}

impl ScratchU32 {
    fn new(ctx: &Context) -> ScratchU32 {
        ScratchU32 { ctx: ctx.clone(), buf: Default::default() }
    }

    fn get(&self, len: usize) -> std::cell::RefMut<'_, Array<u32>> {
        let mut cache = self.buf.borrow_mut();
        if cache.as_ref().is_none_or(|(n, _)| *n != len) {
            *cache = Some((len, Array::<u32>::zeros_on(&self.ctx, len).unwrap()));
        }
        std::cell::RefMut::map(cache, |c| &mut c.as_mut().unwrap().1)
    }
}

impl HandArgmaxAxis {
    pub fn new(ctx: &Context) -> Result<HandArgmaxAxis> {
        let pass = axis_pass(
            &format!(
                "@group(1) @binding(1) var<storage, read_write> pk: array<u32>;
@group(1) @binding(2) var<storage, read_write> pi: array<u32>;
@group(1) @binding(3) var<uniform> p: P;{ARG_KEY}"
            ),
            "var bk = 0u; var bi = r0;",
            "let k = key(v); if (k > bk) { bk = k; bi = r; }",
            "pk[c * p.blocks + blk] = bk; pi[c * p.blocks + blk] = bi;",
        );
        Ok(HandArgmaxAxis {
            pass: ctx.kernel(&pass).label("hand_argmax_axis").build()?,
            combine: ctx.kernel(ARG_AXIS_COMBINE).label("hand_argmax_axis_combine").build()?,
            keys: ScratchU32::new(ctx),
            rows: ScratchU32::new(ctx),
        })
    }

    /// Records `out[c] = argmax_r X[c, r]`; `out` holds `cols` elements.
    pub fn run(&self, x: &Array<f32>, out: &mut Array<u32>, cols: usize, rows: usize) {
        let (p, groups, blocks) = axis_geometry(cols, rows, 1.0);
        let (mut pk, mut pi) = (self.keys.get(blocks * cols), self.rows.get(blocks * cols));
        self.pass.launch_raw([groups, 1, 1], args![x, &mut *pk, &mut *pi, uniform(&p)]).unwrap();
        self.combine.launch_raw([cols as u32, 1, 1], args![&*pk, &*pi, out, uniform(&p)]).unwrap();
    }
}

/// `out[i] = |{e in edges : e < v[i]}|` (`Left`) or `|{e : e <= v[i]}|` (`Right`) for `f32`
/// values `v` and sorted `f32` edges (NaN last, equal to each other, above everything else): a
/// binary search per invocation, one value each, over power-of-two steps in a loop (measured
/// 2% faster than the `lo`/`hi` form), with the NaN-aware order tested by the bits of the probe.
pub struct HandSearchSorted(Kernel);

#[derive(encase::ShaderType)]
struct SearchParams {
    n: u32,
    len: u32,
}

impl HandSearchSorted {
    pub fn new(ctx: &Context, right: bool) -> Result<HandSearchSorted> {
        let before = if right { "vn || (!isnan(e) && e <= v)" } else { "!isnan(e) && (vn || e < v)" };
        let src = format!(
            r#"
struct P {{ n: u32, len: u32 }}
@group(1) @binding(0) var<storage, read> edges: array<f32>;
@group(1) @binding(1) var<storage, read> vals: array<f32>;
@group(1) @binding(2) var<storage, read_write> out: array<u32>;
@group(1) @binding(3) var<uniform> p: P;

fn isnan(x: f32) -> bool {{ return (bitcast<u32>(x) & 0x7fffffffu) > 0x7f800000u; }}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {{
    let i = gid.x + gid.y * nw.x * 256u;
    if (i >= p.n) {{ return; }}
    let v = vals[i];
    let vn = isnan(v);
    var lo = 0u;
    var step = 1u << (31u - countLeadingZeros(p.len));
    loop {{
        let idx = lo + step - 1u;
        if (idx < p.len) {{
            let e = edges[idx];
            if ({before}) {{ lo += step; }}
        }}
        if (step == 1u) {{ break; }}
        step >>= 1u;
    }}
    out[i] = lo;
}}
"#
        );
        ctx.kernel(&src).label(if right { "hand_searchsorted_right" } else { "hand_searchsorted_left" }).build().map(HandSearchSorted)
    }

    pub fn run(&self, edges: &Array<f32>, vals: &Array<f32>, out: &mut Array<u32>, n: usize, len: usize) {
        let p = SearchParams { n: n as u32, len: len as u32 };
        self.0.launch_raw(grid_2d(n.div_ceil(256)), args![edges, vals, out, uniform(&p)]).unwrap();
    }
}
