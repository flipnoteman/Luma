//! Benchmarks of the line solvers against a hand-written cyclic-reduction kernel.
//!
//! `cargo run --release --example bench_solve -- [--only <substring>]`
//!
//! `block4/...` rows solve a 4x4 block-tridiagonal system along every line of a grid with
//! [`solve_block_tridiagonal`] and with `HAND`, the parallel cyclic reduction kernel the bowshock
//! implicit solver had before it moved to luma (closed-form 4x4 inverse as `mat4x4<f32>` algebra,
//! `vec4` rows, one workgroup per line, one thread per row up to 256). Both are verified against
//! a host `f64` block Thomas solve, then timed interleaved: GPU time of the one solver launch
//! (`ctx.profile`, best of 7 after a warm-up of at least 300 ms, as the GPU lowers its clocks when
//! idle). `parts/...` rows time the block and scalar solves on parts against the stacked forms.
//!
//! One row per case: `case shape lines x rows luma_us hand_us ratio`.

use luma::*;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::time::{Duration, Instant};

type A = Array<f32>;
/// An array of `vec4<f32>`.
type V = Array<[f32; 4]>;

fn vec4s(ctx: &Context, host: &[f32]) -> V {
    let quads: &[[f32; 4]] = bytemuck::cast_slice(host);
    Array::from_slice_on(ctx, quads, quads.len()).unwrap()
}

const RUNS: usize = 7;
const WARMUP: Duration = Duration::from_millis(300);

// =================================================================================================
// Hand-written reference
// =================================================================================================

#[derive(encase::ShaderType)]
struct HandParams {
    n: u32,
    stride: u32,
    lines: u32,
}

/// Workgroup memory the rows of one line may take (what luma's solver is given).
const LINE_BYTES: usize = 16 * 1024;

/// Threads per line of a solve of length `n`: one per row up to 256, a multiple of the warp size.
fn hand_threads(n: usize) -> u32 {
    let rows = n.div_ceil(256);
    (n.div_ceil(rows).div_ceil(32) * 32) as u32
}

/// Power of two `ks`: the rows that are a multiple of `ks` live in workgroup memory (9 `vec4` per
/// row), the others in the global scratch.
fn hand_stride(n: usize) -> usize {
    let mut ks = 1;
    while n.div_ceil(ks) * 144 > LINE_BYTES {
        ks *= 2;
    }
    ks
}

const HAND_LIB: &str = r#"
alias M4 = mat4x4<f32>;
struct P { n: u32, stride: u32, lines: u32 }
@group(1) @binding(0) var<storage, read_write> out: array<vec4<f32>>;
//@SM
@group(1) @binding(BLO) var<storage, read> blo: array<vec4<f32>>;
@group(1) @binding(BD) var<storage, read> bd: array<vec4<f32>>;
@group(1) @binding(BUP) var<storage, read> bup: array<vec4<f32>>;
@group(1) @binding(BRHS) var<storage, read> rhs: array<vec4<f32>>;
@group(1) @binding(BP) var<uniform> p: P;

//@CONSTS

var<workgroup> shs: array<vec4<f32>, 9u * NSH>;
var<private> row_base: u32;

fn m_zero() -> M4 { return M4(vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0)); }
fn m_eye() -> M4 {
    return M4(vec4<f32>(1.0, 0.0, 0.0, 0.0), vec4<f32>(0.0, 1.0, 0.0, 0.0), vec4<f32>(0.0, 0.0, 1.0, 0.0), vec4<f32>(0.0, 0.0, 0.0, 1.0));
}
fn m_rows(a: vec4<f32>, b: vec4<f32>, c: vec4<f32>, d: vec4<f32>) -> M4 { return transpose(M4(a, b, c, d)); }

// Inverse by the Laplace expansion in 2x2 minors (no pivoting).
fn inv4(m: M4) -> M4 {
    let m00 = m[0][0]; let m01 = m[1][0]; let m02 = m[2][0]; let m03 = m[3][0];
    let m10 = m[0][1]; let m11 = m[1][1]; let m12 = m[2][1]; let m13 = m[3][1];
    let m20 = m[0][2]; let m21 = m[1][2]; let m22 = m[2][2]; let m23 = m[3][2];
    let m30 = m[0][3]; let m31 = m[1][3]; let m32 = m[2][3]; let m33 = m[3][3];
    let s0 = m00 * m11 - m10 * m01;
    let s1 = m00 * m12 - m10 * m02;
    let s2 = m00 * m13 - m10 * m03;
    let s3 = m01 * m12 - m11 * m02;
    let s4 = m01 * m13 - m11 * m03;
    let s5 = m02 * m13 - m12 * m03;
    let c5 = m22 * m33 - m32 * m23;
    let c4 = m21 * m33 - m31 * m23;
    let c3 = m21 * m32 - m31 * m22;
    let c2 = m20 * m33 - m30 * m23;
    let c1 = m20 * m32 - m30 * m22;
    let c0 = m20 * m31 - m30 * m21;
    let id = 1.0 / (s0 * c5 - s1 * c4 + s2 * c3 + s3 * c2 - s4 * c1 + s5 * c0);
    return m_rows(
        vec4<f32>(m11 * c5 - m12 * c4 + m13 * c3, -m01 * c5 + m02 * c4 - m03 * c3, m31 * s5 - m32 * s4 + m33 * s3, -m21 * s5 + m22 * s4 - m23 * s3) * id,
        vec4<f32>(-m10 * c5 + m12 * c2 - m13 * c1, m00 * c5 - m02 * c2 + m03 * c1, -m30 * s5 + m32 * s2 - m33 * s1, m20 * s5 - m22 * s2 + m23 * s1) * id,
        vec4<f32>(m10 * c4 - m11 * c2 + m13 * c0, -m00 * c4 + m01 * c2 - m03 * c0, m30 * s4 - m31 * s2 + m33 * s0, -m20 * s4 + m21 * s2 - m23 * s0) * id,
        vec4<f32>(-m10 * c3 + m11 * c1 - m12 * c0, m00 * c3 - m01 * c1 + m02 * c0, -m30 * s3 + m31 * s1 - m32 * s0, m20 * s3 - m21 * s1 + m22 * s0) * id);
}

// The block of cell `q` of a luma `[16, ...]` array: row `r` is `a[4 q + r]`.
fn ld_blk(a: u32, q: u32) -> M4 {
    var t: M4;
    if (a == 0u) { t = M4(blo[4u * q], blo[4u * q + 1u], blo[4u * q + 2u], blo[4u * q + 3u]); }
    else if (a == 1u) { t = M4(bd[4u * q], bd[4u * q + 1u], bd[4u * q + 2u], bd[4u * q + 3u]); }
    else { t = M4(bup[4u * q], bup[4u * q + 1u], bup[4u * q + 2u], bup[4u * q + 3u]); }
    return transpose(t);
}

fn in_shared(r: u32) -> bool { return (r & (KS - 1u)) == 0u; }
fn ld_v(e: u32, r: u32) -> vec4<f32> {
    if (in_shared(r)) { return shs[e * NSH + r / KS]; }
    //@LDV
}
fn st_v(e: u32, r: u32, v: vec4<f32>) {
    if (in_shared(r)) { shs[e * NSH + r / KS] = v; return; }
    //@STV
}
fn ld_m(e: u32, r: u32) -> M4 {
    if (in_shared(r)) {
        let o = e * NSH + r / KS;
        return M4(shs[o], shs[o + NSH], shs[o + 2u * NSH], shs[o + 3u * NSH]);
    }
    //@LDM
}
fn st_m(e: u32, r: u32, m: M4) {
    if (in_shared(r)) {
        let o = e * NSH + r / KS;
        shs[o] = m[0];
        shs[o + NSH] = m[1];
        shs[o + 2u * NSH] = m[2];
        shs[o + 3u * NSH] = m[3];
        return;
    }
    //@STM
}

fn sync(global_rows: bool) {
    if (global_rows) { storageBarrier(); }
    workgroupBarrier();
}

@compute @workgroup_size(WG)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let l = wid.x;
    let m = p.n;
    let base = (l % p.stride) + (l / p.stride) * p.stride * p.n;
    row_base = l * m;
    for (var r: u32 = li; r < m; r += WG) {
        let q = base + r * p.stride;
        let di = inv4(ld_blk(1u, q));
        var lm = m_zero();
        var um = m_zero();
        if (r > 0u) { lm = di * ld_blk(0u, q); }
        if (r + 1u < m) { um = di * ld_blk(2u, q); }
        st_m(0u, r, lm);
        st_m(4u, r, um);
        st_v(8u, r, di * rhs[q]);
    }
    sync(KS > 1u);
    var h = 1u;
    for (; h < m; h = h << 1u) {
        for (var r: u32 = 2u * h * li; r < m; r += 2u * h * WG) {
            let lk = ld_m(0u, r);
            let uk = ld_m(4u, r);
            var fk = ld_v(8u, r);
            var d = m_eye();
            var ln = m_zero();
            var un = m_zero();
            if (r > 0u) {
                d = d - lk * ld_m(4u, r - h);
                ln = (lk * ld_m(0u, r - h)) * -1.0;
                fk = fk - lk * ld_v(8u, r - h);
            }
            if (r + h < m) {
                d = d - uk * ld_m(0u, r + h);
                un = (uk * ld_m(4u, r + h)) * -1.0;
                fk = fk - uk * ld_v(8u, r + h);
            }
            let di = inv4(d);
            st_m(0u, r, di * ln);
            st_m(4u, r, di * un);
            st_v(8u, r, di * fk);
        }
        sync(2u * h < KS);
    }
    while (h > 1u) {
        h = h >> 1u;
        for (var r: u32 = h + 2u * h * li; r < m; r += 2u * h * WG) {
            var xk = ld_v(8u, r) - ld_m(0u, r) * ld_v(8u, r - h);
            if (r + h < m) { xk = xk - ld_m(4u, r) * ld_v(8u, r + h); }
            st_v(8u, r, xk);
        }
        sync(h < KS);
    }
    for (var r: u32 = li; r < m; r += WG) {
        out[base + r * p.stride] = ld_v(8u, r);
    }
}
"#;

/// The kernel for lines of `n` rows; `scratch` says whether it spills rows to global memory.
fn hand_source(n: usize) -> String {
    let ks = hand_stride(n);
    let spill = ks > 1;
    let (blo, bd, bup, brhs, bp) = if spill { (2, 3, 4, 5, 6) } else { (1, 2, 3, 4, 5) };
    let mut src = HAND_LIB
        .replace("BLO", &blo.to_string())
        .replace("BUP", &bup.to_string())
        .replace("BRHS", &brhs.to_string())
        .replace("BP", &bp.to_string())
        .replace("BD", &bd.to_string())
        .replace("//@CONSTS", &format!("const WG: u32 = {}u;\nconst KS: u32 = {ks}u;\nconst NSH: u32 = {}u;", hand_threads(n), n.div_ceil(ks)));
    if spill {
        src = src
            .replace("//@SM", "@group(1) @binding(1) var<storage, read_write> sm: array<vec4<f32>>;")
            .replace("//@LDV", "return sm[e * p.lines * p.n + row_base + r];")
            .replace("//@STV", "sm[e * p.lines * p.n + row_base + r] = v;")
            .replace(
                "//@LDM",
                "let c = p.lines * p.n;\n    let o = e * c + row_base + r;\n    return M4(sm[o], sm[o + c], sm[o + 2u * c], sm[o + 3u * c]);",
            )
            .replace(
                "//@STM",
                "let c = p.lines * p.n;\n    let o = e * c + row_base + r;\n    sm[o] = m[0];\n    sm[o + c] = m[1];\n    sm[o + 2u * c] = m[2];\n    sm[o + 3u * c] = m[3];",
            );
    } else {
        src = src
            .replace("//@SM", "")
            .replace("//@LDV", "return vec4<f32>(0.0);")
            .replace("//@STV", "")
            .replace("//@LDM", "return m_zero();")
            .replace("//@STM", "");
    }
    src
}

/// The hand-written solve of `[16, g0, g1]` blocks (`rhs` `[4, g0, g1]`) along `axis` (1 or 2).
struct Hand {
    kernel: Kernel,
    scratch: Option<V>,
    n: usize,
    stride: usize,
    lines: usize,
}

impl Hand {
    fn new(ctx: &Context, dims: [usize; 2], axis: usize) -> Hand {
        let n = dims[axis - 1];
        let stride = if axis == 1 { 1 } else { dims[0] };
        let lines = dims[0] * dims[1] / n;
        let kernel = ctx.kernel(&hand_source(n)).label("hand_line").build().unwrap();
        let scratch = (hand_stride(n) > 1).then(|| vec4s(ctx, &vec![0.0; 4 * 9 * lines * n]));
        Hand { kernel, scratch, n, stride, lines }
    }

    fn run(&mut self, out: &mut V, lo: &V, diag: &V, up: &V, rhs: &V) {
        let p = HandParams { n: self.n as u32, stride: self.stride as u32, lines: self.lines as u32 };
        let groups = [self.lines as u32, 1, 1];
        match &mut self.scratch {
            Some(sm) => self.kernel.launch_raw(groups, args![out, sm, lo, diag, up, rhs, uniform(&p)]).unwrap(),
            None => self.kernel.launch_raw(groups, args![out, lo, diag, up, rhs, uniform(&p)]).unwrap(),
        }
    }
}

// =================================================================================================
// Harness
// =================================================================================================

fn values(n: usize, seed: u64, lo: f32, hi: f32) -> Vec<f32> {
    let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let unit = ((state >> 40) as f32) / (1u64 << 24) as f32;
            lo + unit * (hi - lo)
        })
        .collect()
}

fn check(name: &str, got: &[f32], want: &[f32], rel: f32, abs: f32) {
    assert_eq!(got.len(), want.len(), "{name}: length");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let tol = abs.max(rel * w.abs());
        assert!((g - w).abs() <= tol, "{name}: index {i}: got {g}, want {w} (tol {tol})");
    }
}

/// GPU time of the kernels `f` records whose label contains `label`, in microseconds.
fn gpu_us(ctx: &Context, f: &mut dyn FnMut(), label: &str) -> f64 {
    let ((), p) = ctx
        .profile(|| {
            f();
            ctx.sync().unwrap();
        })
        .unwrap();
    p.kernels.iter().filter(|k| k.label.contains(label)).map(|k| k.total.as_secs_f64() * 1e6).sum()
}

/// Best GPU times (microseconds) of `a` and `b` (kernels with the given labels), interleaved,
/// after a warm-up that keeps the GPU busy for [`WARMUP`].
fn interleaved(ctx: &Context, a: &mut dyn FnMut(), la: &str, b: &mut dyn FnMut(), lb: &str) -> (f64, f64) {
    let start = Instant::now();
    while start.elapsed() < WARMUP {
        a();
        ctx.sync().unwrap();
        b();
        ctx.sync().unwrap();
    }
    let (mut ba, mut bb) = (f64::INFINITY, f64::INFINITY);
    for _ in 0..RUNS {
        ba = ba.min(gpu_us(ctx, a, la));
        bb = bb.min(gpu_us(ctx, b, lb));
    }
    (ba, bb)
}

/// Host block-tridiagonal solve of every line of `grid` along `axis` (block Thomas in `f64`).
#[allow(clippy::too_many_arguments)]
fn host_tri(b: usize, grid: [usize; 3], axis: usize, lo: &[f32], diag: &[f32], up: &[f32], rhs: &[f32]) -> Vec<f32> {
    let cells: usize = grid.iter().product();
    let n = grid[axis];
    let stride: usize = grid[..axis].iter().product();
    let mut x = vec![0.0f32; cells * b];
    let block = |a: &[f32], q: usize| -> Vec<f64> { (0..b * b).map(|k| f64::from(a[q * b * b + k])).collect() };
    for l in 0..cells / n {
        let base = (l % stride) + (l / stride) * stride * n;
        let mut c: Vec<Vec<f64>> = Vec::with_capacity(n);
        let mut d: Vec<Vec<f64>> = Vec::with_capacity(n);
        for s in 0..n {
            let q = base + s * stride;
            let mut m = block(diag, q);
            let mut r: Vec<f64> = (0..b).map(|i| f64::from(rhs[q * b + i])).collect();
            if s > 0 {
                let lq = block(lo, q);
                for i in 0..b {
                    for k in 0..b {
                        for j in 0..b {
                            m[i * b + j] -= lq[i * b + k] * c[s - 1][k * b + j];
                        }
                        r[i] -= lq[i * b + k] * d[s - 1][k];
                    }
                }
            }
            let uq = block(up, q);
            let mut cs = vec![0.0f64; b * b];
            for j in 0..b {
                let column = dense_solve(m.clone(), (0..b).map(|i| uq[i * b + j]).collect(), b);
                for i in 0..b {
                    cs[i * b + j] = column[i];
                }
            }
            d.push(dense_solve(m, r, b));
            c.push(cs);
        }
        let mut next = vec![0.0f64; b];
        for s in (0..n).rev() {
            let q = base + s * stride;
            let cur: Vec<f64> = (0..b).map(|i| d[s][i] - (0..b).map(|k| c[s][i * b + k] * next[k]).sum::<f64>()).collect();
            for i in 0..b {
                x[q * b + i] = cur[i] as f32;
            }
            next = cur;
        }
    }
    x
}

/// Dense solve of `m x = r` (`n x n`, row-major) with partial pivoting in f64.
fn dense_solve(mut m: Vec<f64>, mut r: Vec<f64>, n: usize) -> Vec<f64> {
    for col in 0..n {
        let pivot = (col..n).max_by(|&i, &j| m[i * n + col].abs().total_cmp(&m[j * n + col].abs())).unwrap();
        for j in 0..n {
            m.swap(col * n + j, pivot * n + j);
        }
        r.swap(col, pivot);
        for i in col + 1..n {
            let f = m[i * n + col] / m[col * n + col];
            for j in col..n {
                m[i * n + j] -= f * m[col * n + j];
            }
            r[i] -= f * r[col];
        }
    }
    for i in (0..n).rev() {
        let mut x = r[i];
        for j in i + 1..n {
            x -= m[i * n + j] * r[j];
        }
        r[i] = x / m[i * n + i];
    }
    r
}

/// Blocks of an implicit step: `I`-dominated diagonal blocks and small couplings.
fn block_system(cells: usize, b: usize, seed: u64) -> [Vec<f32>; 4] {
    let bb = b * b;
    let mut diag = values(cells * bb, seed + 1, -1.0, 1.0);
    for (k, v) in diag.iter_mut().enumerate() {
        if (k % bb) / b == k % b {
            *v += 2.0 * b as f32 + 2.0;
        }
    }
    [values(cells * bb, seed, -1.0, 1.0), diag, values(cells * bb, seed + 2, -1.0, 1.0), values(cells * b, seed + 3, -1.0, 1.0)]
}

fn block4(ctx: &Context, only: &Option<String>) {
    // `[g0, g1]` grids: axis 1 is `g1` lines of `g0` rows (contiguous), axis 2 `g0` lines of `g1` rows.
    for (g0, g1) in [(144usize, 70usize), (288, 140), (156, 122)] {
        for axis in [1usize, 2] {
            let (n, lines) = if axis == 1 { (g0, g1) } else { (g1, g0) };
            let name = format!("block4/{lines}x{n}");
            if only.as_deref().is_some_and(|f| !name.contains(f)) {
                continue;
            }
            let result = catch_unwind(AssertUnwindSafe(|| {
                let cells = g0 * g1;
                let host = block_system(cells, 4, 31);
                let [lo, diag, up, rhs]: [A; 4] = std::array::from_fn(|k| {
                    let c = if k == 3 { 4 } else { 16 };
                    Array::from_slice_on(ctx, &host[k], [c, g0, g1]).unwrap()
                });
                let want = host_tri(4, [g0, g1, 1], axis - 1, &host[0], &host[1], &host[2], &host[3]);
                let mut hand = Hand::new(ctx, [g0, g1], axis);
                let [vlo, vdiag, vup] = [0, 1, 2].map(|k| vec4s(ctx, &host[k]));
                let vrhs = vec4s(ctx, &host[3]);
                let mut out = vec4s(ctx, &vec![0.0; cells * 4]);
                hand.run(&mut out, &vlo, &vdiag, &vup, &vrhs);
                let got: Vec<f32> = out.to_vec().unwrap().concat();
                if std::env::var_os("BENCH_NOCHECK").is_none() {
                    check(&format!("{name} hand"), &got, &want, 1e-3, 1e-3);
                    check(&format!("{name} luma"), &solve_block_tridiagonal(4, axis, &lo, &diag, &up, &rhs).to_vec().unwrap(), &want, 1e-3, 1e-3);
                }
                let mut luma = || solve_block_tridiagonal(4, axis, &lo, &diag, &up, &rhs).eval().unwrap();
                let mut hand_run = || hand.run(&mut out, &vlo, &vdiag, &vup, &vrhs);
                let (l, h) = interleaved(ctx, &mut luma, "Tridiagonal", &mut hand_run, "hand_line");
                println!("{name:<24} {g0}x{g1} axis{axis} {l:>9.2} {h:>9.2} {:>7.3}", l / h);
            }));
            if result.is_err() {
                println!("SKIPPED: {name}");
            }
        }
    }
}

/// Total GPU time of everything `f` records, in microseconds.
fn total_us(ctx: &Context, f: &mut dyn FnMut()) -> (f64, u32) {
    let ((), p) = ctx
        .profile(|| {
            f();
            ctx.sync().unwrap();
        })
        .unwrap();
    (p.total().as_secs_f64() * 1e6, p.launches())
}

/// Best (time, launches) of `a` and `b` interleaved, after the warm-up.
fn interleaved_total(ctx: &Context, a: &mut dyn FnMut(), b: &mut dyn FnMut()) -> ((f64, u32), (f64, u32)) {
    let start = Instant::now();
    while start.elapsed() < WARMUP {
        a();
        ctx.sync().unwrap();
        b();
        ctx.sync().unwrap();
    }
    let (mut ba, mut bb) = ((f64::INFINITY, 0), (f64::INFINITY, 0));
    for _ in 0..RUNS {
        let ra = total_us(ctx, a);
        if ra.0 < ba.0 {
            ba = ra;
        }
        let rb = total_us(ctx, b);
        if rb.0 < bb.0 {
            bb = rb;
        }
    }
    (ba, bb)
}

/// The solves on parts against the stacked forms, on the shapes of a block-implicit and a
/// two-equation turbulence step: GPU time and launches of everything each records (stacking,
/// solving, unstacking). Block 4 on `[ni, nj]` grids: the 16 + 16 + 16 + 4 parts evaluated (as
/// the cached blocks of an implicit step are) or expressions of two fields; scalar: two systems.
fn parts(ctx: &Context, only: &Option<String>) {
    for (ni, nj) in [(144usize, 70usize), (156, 122), (288, 140)] {
        for lazy in [false, true] {
            for dir in 0..2usize {
                let name = format!("parts/block4_{}_{ni}x{nj}_dir{dir}", if lazy { "lazy" } else { "cached" });
                if only.as_deref().is_some_and(|f| !name.contains(f)) {
                    continue;
                }
                let result = catch_unwind(AssertUnwindSafe(|| {
                    let cells = ni * nj;
                    let host = block_system(cells, 4, 61);
                    // Entry `e` of the blocks and components of the right-hand side as `[ni, nj]` fields.
                    let field = |which: usize, width: usize, e: usize| -> A {
                        let entry: Vec<f32> = (0..cells).map(|q| host[which][q * width + e]).collect();
                        Array::from_slice_on(ctx, &entry, [ni, nj]).unwrap()
                    };
                    let base: Vec<A> = (0..16).flat_map(|e| [field(0, 16, e), field(1, 16, e), field(2, 16, e)]).collect();
                    let rhs_base: Vec<A> = (0..4).map(|e| field(3, 4, e)).collect();
                    let scale = field(3, 4, 0);
                    let mk = |k: usize| -> A { if lazy { &(&base[k] * 1.0) + &(&scale * 0.0) } else { base[k].cache() } };
                    let part = |which: usize, e: usize| mk(3 * e + which);
                    let lo: Vec<A> = (0..16).map(|e| part(0, e)).collect();
                    let diag: Vec<A> = (0..16).map(|e| part(1, e)).collect();
                    let up: Vec<A> = (0..16).map(|e| part(2, e)).collect();
                    let rhs: Vec<A> = rhs_base.iter().map(|r| if lazy { r * 1.0 } else { r.cache() }).collect();
                    for a in lo.iter().chain(&diag).chain(&up).chain(&rhs) {
                        if !lazy {
                            a.eval().unwrap();
                        }
                    }
                    let (lor, diagr, upr, rhsr): (Vec<&A>, Vec<&A>, Vec<&A>, Vec<&A>) = (lo.iter().collect(), diag.iter().collect(), up.iter().collect(), rhs.iter().collect());
                    let mut stacked = || {
                        let (l, d, u, x) = (stack_slice(0, &lo), stack_slice(0, &diag), stack_slice(0, &up), stack_slice(0, &rhs));
                        let sol = solve_block_tridiagonal(4, dir + 1, &l, &d, &u, &x);
                        let comp: Vec<A> = (0..4).map(|c| sol.slice(0, c..c + 1).reshape([ni, nj])).collect();
                        eval_all(&comp.iter().collect::<Vec<_>>()).unwrap();
                    };
                    let mut on_parts = || {
                        let x = solve_block_tridiagonal_parts(
                            dir,
                            std::array::from_fn::<_, 16, _>(|k| lor[k]),
                            std::array::from_fn::<_, 16, _>(|k| diagr[k]),
                            std::array::from_fn::<_, 16, _>(|k| upr[k]),
                            std::array::from_fn::<_, 4, _>(|k| rhsr[k]),
                        );
                        eval_all(&x.iter().collect::<Vec<_>>()).unwrap();
                    };
                    let want = host_tri(4, [ni, nj, 1], dir, &host[0], &host[1], &host[2], &host[3]);
                    let x = solve_block_tridiagonal_parts(
                        dir,
                        std::array::from_fn::<_, 16, _>(|k| lor[k]),
                        std::array::from_fn::<_, 16, _>(|k| diagr[k]),
                        std::array::from_fn::<_, 16, _>(|k| upr[k]),
                        std::array::from_fn::<_, 4, _>(|k| rhsr[k]),
                    );
                    if std::env::var_os("BENCH_NOCHECK").is_none() {
                        let comps: Vec<Vec<f32>> = x.iter().map(|c| c.to_vec().unwrap()).collect();
                        let got: Vec<f32> = (0..cells * 4).map(|k| comps[k % 4][k / 4]).collect();
                        check(&name, &got, &want, 1e-3, 1e-3);
                    }
                    let ((s, sl), (p, pl)) = interleaved_total(ctx, &mut stacked, &mut on_parts);
                    if std::env::var_os("BENCH_KERNELS").is_some() {
                        let ((), prof) = ctx.profile(|| {
                            on_parts();
                            ctx.sync().unwrap();
                        })
                        .unwrap();
                        for k in &prof.kernels {
                            println!("    {:<60} {:>8.2} us", k.label.chars().take(60).collect::<String>(), k.total.as_secs_f64() * 1e6);
                        }
                    }
                    println!("{name:<34} stacked {s:>8.2} us {sl:>3} launches   parts {p:>8.2} us {pl:>3} launches   ratio {:.3}", p / s);
                }));
                if result.is_err() {
                    println!("SKIPPED: {name}");
                }
            }
        }
    }
    for (ni, nj) in [(156usize, 122usize), (312, 122)] {
        for dir in 0..2usize {
            let name = format!("parts/scalar2_{ni}x{nj}_axis{dir}");
            if only.as_deref().is_some_and(|f| !name.contains(f)) {
                continue;
            }
            let result = catch_unwind(AssertUnwindSafe(|| {
                let cells = ni * nj;
                let sys: Vec<[Vec<f32>; 4]> = (0..2).map(|k| block_system(cells, 1, 71 + 9 * k)).collect();
                let arr = |s: usize, which: usize| -> A { Array::from_slice_on(ctx, &sys[s][which], [ni, nj]).unwrap() };
                let a: Vec<[A; 4]> = (0..2).map(|s| std::array::from_fn(|w| arr(s, w))).collect();
                a.iter().flatten().for_each(|x| x.eval().unwrap());
                let mut separate = || {
                    let x = solve_tridiagonal(dir, &a[0][0], &a[0][1], &a[0][2], &a[0][3]);
                    let y = solve_tridiagonal(dir, &a[1][0], &a[1][1], &a[1][2], &a[1][3]);
                    eval_all(&[&x, &y]).unwrap();
                };
                let mut batched = || {
                    let [x, y] = solve_tridiagonal_parts(dir, [(&a[0][0], &a[0][1], &a[0][2], &a[0][3]), (&a[1][0], &a[1][1], &a[1][2], &a[1][3])]);
                    eval_all(&[&x, &y]).unwrap();
                };
                if std::env::var_os("BENCH_NOCHECK").is_none() {
                    let [x, y] = solve_tridiagonal_parts(dir, [(&a[0][0], &a[0][1], &a[0][2], &a[0][3]), (&a[1][0], &a[1][1], &a[1][2], &a[1][3])]);
                    for (s, x) in [x, y].iter().enumerate() {
                        let want = host_tri(1, [ni, nj, 1], dir, &sys[s][0], &sys[s][1], &sys[s][2], &sys[s][3]);
                        check(&name, &x.to_vec().unwrap(), &want, 1e-3, 1e-3);
                    }
                }
                let ((s, sl), (p, pl)) = interleaved_total(ctx, &mut separate, &mut batched);
                println!("{name:<34} separate {s:>7.2} us {sl:>3} launches   parts {p:>8.2} us {pl:>3} launches   ratio {:.3}", p / s);
            }));
            if result.is_err() {
                println!("SKIPPED: {name}");
            }
        }
    }
}

/// Luma only: the block sizes and grids of `bench_pde --only tri` (and a few more), to compare
/// builds of the library: one row per case, best GPU time of the solver launch in microseconds.
fn sweep(ctx: &Context, only: &Option<String>) {
    let mut cases: Vec<(usize, usize, usize, usize)> = Vec::new();
    for b in [1usize, 2, 3, 4, 5, 6, 8] {
        for (g0, g1) in [(100usize, 50usize), (200, 100), (400, 200), (144, 70), (156, 122)] {
            for axis in [1usize, 2] {
                cases.push((b, g0, g1, axis));
            }
        }
    }
    for axis in [1usize, 2] {
        cases.extend([(4, 1024, 100, axis), (4, 1024, 1024, axis), (1, 4096, 200, axis), (1, 1024, 1024, axis), (8, 256, 64, axis), (2, 2048, 32, axis)]);
    }
    for (b, g0, g1, axis) in cases {
        let name = format!("sweep/b{b}_{g0}x{g1}_axis{axis}");
        if only.as_deref().is_some_and(|f| !name.contains(f)) {
            continue;
        }
        let result = catch_unwind(AssertUnwindSafe(|| {
            let cells = g0 * g1;
            let host = block_system(cells, b, 41);
            let [lo, diag, up, rhs]: [A; 4] = std::array::from_fn(|k| {
                let c = if k == 3 { b } else { b * b };
                if b == 1 { Array::from_slice_on(ctx, &host[k], [g0, g1]).unwrap() } else { Array::from_slice_on(ctx, &host[k], [c, g0, g1]).unwrap() }
            });
            let run = || if b == 1 { solve_tridiagonal(axis - 1, &lo, &diag, &up, &rhs) } else { solve_block_tridiagonal(b, axis, &lo, &diag, &up, &rhs) };
            let want = if std::env::var_os("BENCH_NOCHECK").is_none() { host_tri(b, [g0, g1, 1], axis - 1, &host[0], &host[1], &host[2], &host[3]) } else { Vec::new() };
            if std::env::var_os("BENCH_NOCHECK").is_none() {
                check(&name, &run().to_vec().unwrap(), &want, 1e-3, 1e-3);
            }
            let mut f = || run().eval().unwrap();
            let start = Instant::now();
            while start.elapsed() < WARMUP {
                f();
                ctx.sync().unwrap();
            }
            let us = (0..RUNS).map(|_| gpu_us(ctx, &mut f, "Tridiagonal")).fold(f64::INFINITY, f64::min);
            println!("{name:<32} {us:>9.2}");
        }));
        if result.is_err() {
            println!("SKIPPED: {name}");
        }
    }
}

fn main() {
    let mut only = None;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--only" => only = args.next(),
            "--bench" => {}
            other => panic!("unknown argument {other}; usage: bench_solve [--only <substring>]"),
        }
    }
    let ctx = default_context().unwrap();
    println!("bench_solve, profiling supported: {}", ctx.profiling_supported());
    println!("{:<24} {:>7} {:>9} {:>9} {:>7}", "case", "grid", "luma us", "hand us", "ratio");
    block4(&ctx, &only);
    parts(&ctx, &only);
    if only.as_deref().is_some_and(|f| "sweep/".contains(f) || f.starts_with("sweep")) || only.is_none() {
        println!("\n{:<32} {:>9}", "sweep case", "luma us");
        sweep(&ctx, &only);
    }
}
