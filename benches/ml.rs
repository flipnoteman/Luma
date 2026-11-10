//! Machine-learning workloads, Luma against the same work on every CPU core.
//!
//! `cargo bench --bench ml` (add `-- <filter>` for one group). Build with
//! `RUSTFLAGS="-C target-cpu=native"` so the CPU side gets the host's vector units. The CPU
//! baselines are parallel over all cores; matrix products use Apple's Accelerate
//! (`cblas_sgemm`) and everything else is written so that LLVM vectorizes the inner loops. Every
//! group first checks that both sides compute the same result.

use std::hint::black_box;
use std::time::Duration;

use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use luma::*;

fn threads() -> usize {
    std::thread::available_parallelism().map_or(1, |n| n.get())
}

/// Runs `f(first_row, rows)` over contiguous runs of `row_len`-element rows of `out`, one run
/// per thread.
fn par_rows<T: Send>(out: &mut [T], row_len: usize, f: impl Fn(usize, &mut [T]) + Sync) {
    let rows = out.len() / row_len;
    let per = rows.div_ceil(threads()).max(1);
    std::thread::scope(|s| {
        for (i, chunk) in out.chunks_mut(per * row_len).enumerate() {
            let f = &f;
            s.spawn(move || f(i * per, chunk));
        }
    });
}

/// Deterministic values in `lo..hi`.
fn values(n: usize, seed: u64, lo: f32, hi: f32) -> Vec<f32> {
    let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            lo + ((state >> 40) as f32 / (1u64 << 24) as f32) * (hi - lo)
        })
        .collect()
}

/// Largest difference relative to the largest magnitude of `want`.
fn check(name: &str, got: &[f32], want: &[f32], tolerance: f32) {
    assert_eq!(got.len(), want.len(), "{name}: length");
    let scale = want.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1e-30);
    let worst = got.iter().zip(want).fold(0.0f32, |m, (g, w)| m.max((g - w).abs()));
    assert!(worst / scale <= tolerance, "{name}: GPU and CPU differ by {worst} (scale {scale})");
}

// -- Accelerate --

#[cfg(target_os = "macos")]
#[link(name = "Accelerate", kind = "framework")]
extern "C" {
    #[allow(clippy::too_many_arguments)]
    fn cblas_sgemm(
        order: i32,
        trans_a: i32,
        trans_b: i32,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        a: *const f32,
        lda: i32,
        b: *const f32,
        ldb: i32,
        beta: f32,
        c: *mut f32,
        ldc: i32,
    );
}

/// Row-major `C (m x n) = A (m x k) B`, with `B` either `k x n` or, when `b_transposed`, `n x k`.
#[cfg(target_os = "macos")]
fn sgemm_cpu(b_transposed: bool, m: usize, n: usize, k: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    const ROW_MAJOR: i32 = 101;
    const NO_TRANS: i32 = 111;
    const TRANS: i32 = 112;
    assert!(a.len() == m * k && b.len() == k * n && c.len() == m * n);
    let (trans_b, ldb) = if b_transposed { (TRANS, k) } else { (NO_TRANS, n) };
    // SAFETY: the lengths asserted above match the dimensions and leading dimensions passed.
    unsafe {
        cblas_sgemm(
            ROW_MAJOR,
            NO_TRANS,
            trans_b,
            m as i32,
            n as i32,
            k as i32,
            1.0,
            a.as_ptr(),
            k as i32,
            b.as_ptr(),
            ldb as i32,
            0.0,
            c.as_mut_ptr(),
            n as i32,
        )
    };
}

/// `sgemm_cpu` with the rows of `C` split over all cores, one Accelerate call per thread. A
/// single call does not always spread over every core (a transposed `B` with a small `k`
/// barely does).
#[cfg(target_os = "macos")]
fn sgemm_par_cpu(b_transposed: bool, n: usize, k: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    par_rows(c, n, |first, rows| {
        let m = rows.len() / n;
        sgemm_cpu(b_transposed, m, n, k, &a[first * k..][..m * k], b, rows);
    });
}

// -- Tiled SGEMM shared by the MLP and the similarity matrix --

#[path = "common/sgemm.rs"]
mod sgemm;
use sgemm::{sgemm_source, SgemmParams, TILE};

// -- k-means: one assignment + centroid update iteration --

const KM_N: usize = 4 << 20;
const KM_K: usize = 64;
/// Workgroups of the fused kernel; each handles `KM_N / KM_GROUPS` points, 256 at a time.
const KM_GROUPS: usize = 1024;

/// Assigns each point to its nearest centroid and accumulates per-cluster sums and counts in
/// the same pass. A workgroup labels a tile of 256 points (one per invocation, centroids in
/// workgroup memory), then every invocation owns one cluster and a quarter of the tile and
/// scans the labels, so the sums need no atomics. 64 clusters are hard-coded.
const KMEANS_ASSIGN: &str = r#"
struct Params { tiles: u32 }
@group(1) @binding(0) var<storage, read> points: array<vec4<f32>>;
@group(1) @binding(1) var<storage, read> centroids: array<vec4<f32>>;
@group(1) @binding(2) var<storage, read_write> part_sum: array<vec4<f32>>;
@group(1) @binding(3) var<storage, read_write> part_cnt: array<u32>;
@group(1) @binding(4) var<uniform> p: Params;
var<workgroup> cent: array<vec4<f32>, 64>;
var<workgroup> tile_pt: array<vec4<f32>, 256>;
var<workgroup> tile_label: array<u32, 256>;
var<workgroup> lane_sum: array<vec4<f32>, 256>;
var<workgroup> lane_cnt: array<u32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) t: u32) {
    if t < 64u { cent[t] = centroids[t]; }
    workgroupBarrier();
    let cluster = t % 64u;
    let lane = t / 64u;
    var s = vec4<f32>(0.0);
    var n = 0u;
    let base = wid.x * p.tiles * 256u;
    for (var tile = 0u; tile < p.tiles; tile = tile + 1u) {
        let pt = points[base + tile * 256u + t];
        var best = 3.0e38;
        var label = 0u;
        for (var k = 0u; k < 64u; k = k + 1u) {
            let d = pt - cent[k];
            let dist = dot(d, d);
            if dist < best { best = dist; label = k; }
        }
        tile_pt[t] = pt;
        tile_label[t] = label;
        workgroupBarrier();
        for (var j = lane; j < 256u; j = j + 4u) {
            let hit = tile_label[j] == cluster;
            s = s + select(vec4<f32>(0.0), tile_pt[j], hit);
            n = n + select(0u, 1u, hit);
        }
        workgroupBarrier();
    }
    lane_sum[t] = s;
    lane_cnt[t] = n;
    workgroupBarrier();
    if t < 64u {
        let o = wid.x * 64u + t;
        part_sum[o] = lane_sum[t] + lane_sum[t + 64u] + lane_sum[t + 128u] + lane_sum[t + 192u];
        part_cnt[o] = lane_cnt[t] + lane_cnt[t + 64u] + lane_cnt[t + 128u] + lane_cnt[t + 192u];
    }
}
"#;

/// One workgroup per cluster adds that cluster's partial sums and divides; an empty cluster
/// keeps its old centroid.
const KMEANS_REDUCE: &str = r#"
struct Params { groups: u32, _a: u32, _b: u32, _c: u32 }
@group(1) @binding(0) var<storage, read> part_sum: array<vec4<f32>>;
@group(1) @binding(1) var<storage, read> part_cnt: array<u32>;
@group(1) @binding(2) var<storage, read> prev: array<vec4<f32>>;
@group(1) @binding(3) var<storage, read_write> next: array<vec4<f32>>;
@group(1) @binding(4) var<storage, read_write> counts: array<u32>;
@group(1) @binding(5) var<uniform> p: Params;
var<workgroup> rs: array<vec4<f32>, 256>;
var<workgroup> rn: array<u32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) t: u32) {
    let c = wid.x;
    var s = vec4<f32>(0.0);
    var n = 0u;
    for (var g = t; g < p.groups; g = g + 256u) {
        s = s + part_sum[g * 64u + c];
        n = n + part_cnt[g * 64u + c];
    }
    rs[t] = s;
    rn[t] = n;
    workgroupBarrier();
    for (var stride = 128u; stride > 0u; stride = stride >> 1u) {
        if t < stride {
            rs[t] = rs[t] + rs[t + stride];
            rn[t] = rn[t] + rn[t + stride];
        }
        workgroupBarrier();
    }
    if t == 0u {
        counts[c] = rn[0];
        if rn[0] > 0u { next[c] = rs[0] / f32(rn[0]); } else { next[c] = prev[c]; }
    }
}
"#;

#[derive(encase::ShaderType)]
struct KmeansAssignParams {
    tiles: u32,
}

#[derive(encase::ShaderType)]
struct KmeansReduceParams {
    groups: u32,
    a: u32,
    b: u32,
    c: u32,
}

struct KmeansGpu {
    assign: Kernel,
    reduce: Kernel,
    part_sum: Array<[f32; 4]>,
    part_cnt: Array<u32>,
    centroids: Array<[f32; 4]>,
    counts: Array<u32>,
}

impl KmeansGpu {
    fn new() -> Self {
        KmeansGpu {
            assign: Kernel::new(KMEANS_ASSIGN).unwrap(),
            reduce: Kernel::new(KMEANS_REDUCE).unwrap(),
            part_sum: Array::zeros(KM_GROUPS * KM_K).unwrap(),
            part_cnt: Array::zeros(KM_GROUPS * KM_K).unwrap(),
            centroids: Array::zeros(KM_K).unwrap(),
            counts: Array::zeros(KM_K).unwrap(),
        }
    }

    /// Records one iteration; the updated centroids and cluster sizes land in `self`.
    fn iterate(&mut self, points: &Array<[f32; 4]>, centroids: &Array<[f32; 4]>) {
        let tiles = (KM_N / 256 / KM_GROUPS) as u32;
        self.assign
            .launch_raw(
                [KM_GROUPS as u32, 1, 1],
                args![points, centroids, &mut self.part_sum, &mut self.part_cnt, uniform(&KmeansAssignParams { tiles })],
            )
            .unwrap();
        let params = KmeansReduceParams { groups: KM_GROUPS as u32, a: 0, b: 0, c: 0 };
        self.reduce
            .launch_raw(
                [KM_K as u32, 1, 1],
                args![&self.part_sum, &self.part_cnt, centroids, &mut self.centroids, &mut self.counts, uniform(&params)],
            )
            .unwrap();
    }
}

/// Centroid coordinates in structure-of-arrays form, 8 clusters per vector.
type Soa = [[[f32; 8]; KM_K / 8]; 4];

/// Index of the nearest centroid (lowest index on ties, like the kernel): 8 independent
/// running minima that LLVM turns into vector compares and selects.
#[inline(always)]
fn nearest(p: [f32; 4], cx: &Soa) -> usize {
    let mut best = [f32::INFINITY; 8];
    let mut index = [0u32; 8];
    for block in 0..KM_K / 8 {
        for l in 0..8 {
            let dx = p[0] - cx[0][block][l];
            let dy = p[1] - cx[1][block][l];
            let dz = p[2] - cx[2][block][l];
            let dw = p[3] - cx[3][block][l];
            let dist = dx * dx + dy * dy + dz * dz + dw * dw;
            if dist < best[l] {
                best[l] = dist;
                index[l] = (block * 8 + l) as u32;
            }
        }
    }
    let (mut b, mut bi) = (best[0], index[0]);
    for l in 1..8 {
        if best[l] < b || (best[l] == b && index[l] < bi) {
            b = best[l];
            bi = index[l];
        }
    }
    bi as usize
}

/// Each thread assigns a contiguous run of points and keeps its own sums; the partials are
/// merged at the end.
fn kmeans_cpu(points: &[[f32; 4]], centroids: &[[f32; 4]], next: &mut [[f32; 4]], counts: &mut [u32]) {
    let mut cx: Soa = [[[0.0; 8]; KM_K / 8]; 4];
    for (k, c) in centroids.iter().enumerate() {
        for d in 0..4 {
            cx[d][k / 8][k % 8] = c[d];
        }
    }
    let per = points.len().div_ceil(threads());
    let partials: Vec<(Vec<[f32; 4]>, Vec<u32>)> = std::thread::scope(|s| {
        let cx = &cx;
        let handles: Vec<_> = points
            .chunks(per)
            .map(|chunk| {
                s.spawn(move || {
                    let mut sums = vec![[0.0f32; 4]; KM_K];
                    let mut counts = vec![0u32; KM_K];
                    for &p in chunk {
                        let c = nearest(p, cx);
                        let sum = &mut sums[c];
                        for d in 0..4 {
                            sum[d] += p[d];
                        }
                        counts[c] += 1;
                    }
                    (sums, counts)
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    for k in 0..KM_K {
        let mut sum = [0.0f32; 4];
        let mut n = 0u32;
        for (sums, counts) in &partials {
            for d in 0..4 {
                sum[d] += sums[k][d];
            }
            n += counts[k];
        }
        counts[k] = n;
        next[k] = if n > 0 { sum.map(|v| v / n as f32) } else { centroids[k] };
    }
}

/// Noisy blobs around 64 random centres, so clusters overlap somewhat as in real data.
fn kmeans_data() -> (Vec<[f32; 4]>, Vec<[f32; 4]>) {
    let centers = values(KM_K * 4, 1, 0.0, 10.0);
    let ids = values(KM_N, 2, 0.0, KM_K as f32);
    let noise = values(KM_N * 4, 3, -1.0, 1.0);
    let points: Vec<[f32; 4]> = (0..KM_N)
        .map(|i| {
            let c = (ids[i] as usize).min(KM_K - 1) * 4;
            std::array::from_fn(|d| centers[c + d] + noise[i * 4 + d])
        })
        .collect();
    let centroids = (0..KM_K).map(|k| points[k * (KM_N / KM_K)]).collect();
    (points, centroids)
}

fn kmeans(c: &mut Criterion) {
    let (points, centroids) = kmeans_data();
    let gp = Array::<[f32; 4]>::from_slice(&points, KM_N).unwrap();
    let gc = Array::<[f32; 4]>::from_slice(&centroids, KM_K).unwrap();
    let mut gpu = KmeansGpu::new();
    let ctx = default_context().unwrap();
    let run = |gpu: &mut KmeansGpu| {
        gpu.iterate(&gp, &gc);
        ctx.sync().unwrap();
    };
    run(&mut gpu);

    let mut next = vec![[0.0f32; 4]; KM_K];
    let mut counts = vec![0u32; KM_K];
    kmeans_cpu(&points, &centroids, &mut next, &mut counts);
    let flat = |v: &[[f32; 4]]| v.iter().flatten().copied().collect::<Vec<f32>>();
    check("kmeans centroids", &flat(&gpu.centroids.to_vec().unwrap()), &flat(&next), 1e-3);
    // A point on a numerical tie may land in a neighbouring cluster on one side only.
    let moved: u32 = gpu.counts.to_vec().unwrap().iter().zip(&counts).map(|(g, w)| g.abs_diff(*w)).sum();
    assert!(moved <= 64, "kmeans: {moved} points assigned differently");
    assert_eq!(counts.iter().sum::<u32>() as usize, KM_N);

    let mut group = c.benchmark_group("kmeans_iteration_4M_points_k64");
    group.throughput(Throughput::Elements(KM_N as u64));
    group.bench_function("luma", |b| b.iter(|| run(&mut gpu)));
    group.bench_function("cpu", |b| b.iter(|| kmeans_cpu(&points, &centroids, black_box(&mut next), &mut counts)));
    group.finish();
}

// -- MLP forward pass --

const BATCH: usize = 4096;
const WIDTH: usize = 1024;
const CLASSES: usize = 10;
/// The last layer's output width rounded up to the GEMM tile; padded columns are never read.
const CLASSES_PADDED: usize = TILE;

/// One row per invocation: `probs = softmax(logits)` over `cols` of the `stride`-wide rows.
const SOFTMAX: &str = r#"
struct Params { rows: u32, cols: u32, stride: u32 }
@group(1) @binding(0) var<storage, read> logits: array<f32>;
@group(1) @binding(1) var<storage, read_write> probs: array<f32>;
@group(1) @binding(2) var<uniform> p: Params;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let r = gid.x;
    if r >= p.rows { return; }
    let src = r * p.stride;
    var m = -3.0e38;
    for (var j = 0u; j < p.cols; j = j + 1u) { m = max(m, logits[src + j]); }
    var sum = 0.0;
    for (var j = 0u; j < p.cols; j = j + 1u) { sum = sum + exp(logits[src + j] - m); }
    for (var j = 0u; j < p.cols; j = j + 1u) {
        probs[r * p.cols + j] = exp(logits[src + j] - m) / sum;
    }
}
"#;

#[derive(encase::ShaderType)]
struct SoftmaxParams {
    rows: u32,
    cols: u32,
    stride: u32,
}

/// Row-major weights (`inputs x outputs`) and biases of the three layers, unpadded.
struct MlpHost {
    w: [Vec<f32>; 3],
    b: [Vec<f32>; 3],
}

impl MlpHost {
    fn new() -> Self {
        // Scaled so activations and logits stay O(1) through the network.
        MlpHost {
            w: [
                values(WIDTH * WIDTH, 10, -0.06, 0.06),
                values(WIDTH * WIDTH, 11, -0.06, 0.06),
                values(WIDTH * CLASSES, 12, -0.3, 0.3),
            ],
            b: [values(WIDTH, 13, -0.1, 0.1), values(WIDTH, 14, -0.1, 0.1), values(CLASSES, 15, -0.5, 0.5)],
        }
    }
}

struct MlpGpu {
    gemm: Kernel,
    softmax: Kernel,
    w: [Array<f32>; 3],
    b: [Array<f32>; 3],
    h1: Array<f32>,
    h2: Array<f32>,
    logits: Array<f32>,
    probs: Array<f32>,
}

impl MlpGpu {
    fn new(host: &MlpHost) -> Self {
        let pad = |w: &[f32], rows: usize, cols: usize| {
            let mut out = vec![0.0f32; rows * CLASSES_PADDED];
            for r in 0..rows {
                out[r * CLASSES_PADDED..][..cols].copy_from_slice(&w[r * cols..][..cols]);
            }
            out
        };
        let mut bias3 = host.b[2].clone();
        bias3.resize(CLASSES_PADDED, 0.0);
        let w3 = pad(&host.w[2], WIDTH, CLASSES);
        MlpGpu {
            gemm: Kernel::new(&sgemm_source(false, true)).unwrap(),
            softmax: Kernel::new(SOFTMAX).unwrap(),
            w: [
                Array::from_slice(&host.w[0], [WIDTH, WIDTH]).unwrap(),
                Array::from_slice(&host.w[1], [WIDTH, WIDTH]).unwrap(),
                Array::from_slice(&w3, [CLASSES_PADDED, WIDTH]).unwrap(),
            ],
            b: [
                Array::from_slice(&host.b[0], WIDTH).unwrap(),
                Array::from_slice(&host.b[1], WIDTH).unwrap(),
                Array::from_slice(&bias3, CLASSES_PADDED).unwrap(),
            ],
            h1: Array::zeros([WIDTH, BATCH]).unwrap(),
            h2: Array::zeros([WIDTH, BATCH]).unwrap(),
            logits: Array::zeros([CLASSES_PADDED, BATCH]).unwrap(),
            probs: Array::zeros([CLASSES, BATCH]).unwrap(),
        }
    }

    /// Records the forward pass of the batch `x`; the class probabilities land in `self.probs`.
    fn forward(&mut self, x: &Array<f32>) {
        let hidden = SgemmParams { m: BATCH as u32, n: WIDTH as u32, k: WIDTH as u32, relu: 1 };
        let last = SgemmParams { m: BATCH as u32, n: CLASSES_PADDED as u32, k: WIDTH as u32, relu: 0 };
        let (tiles_m, tiles_hidden) = ((BATCH / TILE) as u32, (WIDTH / TILE) as u32);
        let g = &self.gemm;
        g.launch_raw([tiles_hidden, tiles_m, 1], args![x, &self.w[0], &mut self.h1, uniform(&hidden), &self.b[0]]).unwrap();
        g.launch_raw([tiles_hidden, tiles_m, 1], args![&self.h1, &self.w[1], &mut self.h2, uniform(&hidden), &self.b[1]])
            .unwrap();
        g.launch_raw([1, tiles_m, 1], args![&self.h2, &self.w[2], &mut self.logits, uniform(&last), &self.b[2]]).unwrap();
        let sm = SoftmaxParams { rows: BATCH as u32, cols: CLASSES as u32, stride: CLASSES_PADDED as u32 };
        self.softmax.launch([BATCH as u32], args![&self.logits, &mut self.probs, uniform(&sm)]).unwrap();
    }
}

/// `h = max(h + bias, 0)` on every row.
fn bias_relu_cpu(h: &mut [f32], bias: &[f32]) {
    par_rows(h, bias.len(), |_, rows| {
        for row in rows.chunks_exact_mut(bias.len()) {
            for (v, b) in row.iter_mut().zip(bias) {
                *v = (*v + *b).max(0.0);
            }
        }
    });
}

/// Row-wise softmax of `logits + bias`, 10 classes per row.
fn softmax_cpu(logits: &[f32], bias: &[f32], probs: &mut [f32]) {
    par_rows(probs, CLASSES, |first, rows| {
        for (i, out) in rows.chunks_exact_mut(CLASSES).enumerate() {
            let row = &logits[(first + i) * CLASSES..][..CLASSES];
            let mut z = [0.0f32; CLASSES];
            for ((z, l), b) in z.iter_mut().zip(row).zip(bias) {
                *z = l + b;
            }
            let max = z.iter().fold(f32::NEG_INFINITY, |m, v| m.max(*v));
            let mut sum = 0.0;
            for (o, z) in out.iter_mut().zip(z) {
                *o = (z - max).exp();
                sum += *o;
            }
            for o in out.iter_mut() {
                *o /= sum;
            }
        }
    });
}

#[cfg(target_os = "macos")]
struct MlpCpuBuffers {
    h1: Vec<f32>,
    h2: Vec<f32>,
    logits: Vec<f32>,
    probs: Vec<f32>,
}

#[cfg(target_os = "macos")]
impl MlpCpuBuffers {
    fn new() -> Self {
        MlpCpuBuffers {
            h1: vec![0.0; BATCH * WIDTH],
            h2: vec![0.0; BATCH * WIDTH],
            logits: vec![0.0; BATCH * CLASSES],
            probs: vec![0.0; BATCH * CLASSES],
        }
    }
}

#[cfg(target_os = "macos")]
fn mlp_cpu(x: &[f32], host: &MlpHost, buf: &mut MlpCpuBuffers) {
    sgemm_par_cpu(false, WIDTH, WIDTH, x, &host.w[0], &mut buf.h1);
    bias_relu_cpu(&mut buf.h1, &host.b[0]);
    sgemm_par_cpu(false, WIDTH, WIDTH, &buf.h1, &host.w[1], &mut buf.h2);
    bias_relu_cpu(&mut buf.h2, &host.b[1]);
    sgemm_par_cpu(false, CLASSES, WIDTH, &buf.h2, &host.w[2], &mut buf.logits);
    softmax_cpu(&buf.logits, &host.b[2], &mut buf.probs);
}

fn mlp(c: &mut Criterion) {
    let host = MlpHost::new();
    let x = values(BATCH * WIDTH, 16, -1.0, 1.0);
    let gx = Array::from_slice(&x, [WIDTH, BATCH]).unwrap();
    let mut gpu = MlpGpu::new(&host);
    let ctx = default_context().unwrap();
    gpu.forward(&gx);
    ctx.sync().unwrap();

    #[cfg(target_os = "macos")]
    let mut cpu = MlpCpuBuffers::new();
    #[cfg(target_os = "macos")]
    {
        mlp_cpu(&x, &host, &mut cpu);
        let got = gpu.probs.to_vec().unwrap();
        check("mlp probabilities", &got, &cpu.probs, 1e-3);
        for row in got.chunks_exact(CLASSES) {
            assert!((row.iter().sum::<f32>() - 1.0).abs() < 1e-4, "mlp: probabilities do not sum to 1");
        }
    }

    let mut group = c.benchmark_group("mlp_forward_batch4096_1024x1024x1024x10");
    group.throughput(Throughput::Elements(BATCH as u64));
    group.bench_function("luma", |b| {
        b.iter(|| {
            gpu.forward(&gx);
            ctx.sync().unwrap();
        })
    });
    group.bench_function("luma_with_transfers", |b| {
        b.iter(|| {
            let gx = Array::from_slice(&x, [WIDTH, BATCH]).unwrap();
            gpu.forward(&gx);
            black_box(gpu.probs.to_vec().unwrap());
        })
    });
    #[cfg(target_os = "macos")]
    group.bench_function("cpu", |b| b.iter(|| mlp_cpu(&x, &host, black_box(&mut cpu))));
    group.finish();
}

// -- Cosine similarity matrix --

const EMB_N: usize = 8192;
const EMB_D: usize = 128;

/// One row per invocation: `out = e / |e|`.
const NORMALIZE: &str = r#"
struct Params { rows: u32, dim: u32 }
@group(1) @binding(0) var<storage, read> e: array<f32>;
@group(1) @binding(1) var<storage, read_write> out: array<f32>;
@group(1) @binding(2) var<uniform> p: Params;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let r = gid.x;
    if r >= p.rows { return; }
    var sum = 0.0;
    for (var j = 0u; j < p.dim; j = j + 1u) { let v = e[r * p.dim + j]; sum = sum + v * v; }
    let inv = 1.0 / sqrt(sum);
    for (var j = 0u; j < p.dim; j = j + 1u) { out[r * p.dim + j] = e[r * p.dim + j] * inv; }
}
"#;

#[derive(encase::ShaderType)]
struct NormalizeParams {
    rows: u32,
    dim: u32,
}

#[cfg(target_os = "macos")]
fn normalize_cpu(e: &[f32], out: &mut [f32]) {
    par_rows(out, EMB_D, |first, rows| {
        for (i, row) in rows.chunks_exact_mut(EMB_D).enumerate() {
            let src = &e[(first + i) * EMB_D..][..EMB_D];
            let mut lanes = [0.0f32; 8];
            for chunk in src.chunks_exact(8) {
                for (l, v) in lanes.iter_mut().zip(chunk) {
                    *l += v * v;
                }
            }
            let inv = 1.0 / lanes.iter().sum::<f32>().sqrt();
            for (o, v) in row.iter_mut().zip(src) {
                *o = v * inv;
            }
        }
    });
}

fn cosine_similarity(c: &mut Criterion) {
    let e = values(EMB_N * EMB_D, 20, -1.0, 1.0);
    let ge = Array::from_slice(&e, [EMB_D, EMB_N]).unwrap();
    let mut unit = Array::<f32>::zeros([EMB_D, EMB_N]).unwrap();
    let mut sim = Array::<f32>::zeros([EMB_N, EMB_N]).unwrap();
    let normalize = Kernel::new(NORMALIZE).unwrap();
    let gemm = Kernel::new(&sgemm_source(true, false)).unwrap();
    let ctx = default_context().unwrap();
    let norm = NormalizeParams { rows: EMB_N as u32, dim: EMB_D as u32 };
    let dims = SgemmParams { m: EMB_N as u32, n: EMB_N as u32, k: EMB_D as u32, relu: 0 };
    let run = |unit: &mut Array<f32>, sim: &mut Array<f32>| {
        normalize.launch([EMB_N as u32], args![&ge, &mut *unit, uniform(&norm)]).unwrap();
        gemm.launch_raw([(EMB_N / TILE) as u32, (EMB_N / TILE) as u32, 1], args![&*unit, &*unit, &mut *sim, uniform(&dims)])
            .unwrap();
        ctx.sync().unwrap();
    };
    run(&mut unit, &mut sim);

    #[cfg(target_os = "macos")]
    let mut cpu_unit = vec![0.0f32; EMB_N * EMB_D];
    #[cfg(target_os = "macos")]
    let mut cpu_sim = vec![0.0f32; EMB_N * EMB_N];
    #[cfg(target_os = "macos")]
    let run_cpu = |unit: &mut [f32], sim: &mut [f32]| {
        normalize_cpu(&e, unit);
        sgemm_par_cpu(true, EMB_N, EMB_D, unit, unit, sim);
    };
    #[cfg(target_os = "macos")]
    {
        run_cpu(&mut cpu_unit, &mut cpu_sim);
        check("cosine similarity", &sim.to_vec().unwrap(), &cpu_sim, 1e-5);
    }

    let mut group = c.benchmark_group("cosine_similarity_8192x8192_dim128");
    group.throughput(Throughput::Elements((EMB_N * EMB_N) as u64));
    group.bench_function("luma", |b| b.iter(|| run(&mut unit, &mut sim)));
    #[cfg(target_os = "macos")]
    group.bench_function("cpu", |b| b.iter(|| run_cpu(&mut cpu_unit, black_box(&mut cpu_sim))));
    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(10)
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(3));
    targets = kmeans, mlp, cosine_similarity
}
criterion_main!(benches);
