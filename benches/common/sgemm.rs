//! Tiled SGEMM kernel shared by the benchmarks (`#[path]`-included).

/// Output tile edge of the GEMM kernel.
pub const TILE: usize = 128;

/// Row-major `C = A B` (or `A Bᵀ`); each workgroup computes a 128x128 tile of `C`, each of its
/// 16x16 invocations 8x8 of it, from 16-wide slices of `A` and `B` staged in workgroup memory as
/// vec4s of four rows (columns), so that one 16-byte read feeds four accumulators. An invocation
/// owns two 4-wide groups of rows and of columns, 64 apart, so that workgroup memory reads of
/// neighbouring invocations are contiguous. The slice goes from global to workgroup memory
/// without detour through registers (the 64 accumulators leave none to spare: holding the next
/// slice there measured 20% slower on an M1 Max), and the loop over it is unrolled by two, which
/// halves the loop overhead without hoisting so many reads that registers spill. Dimensions must
/// be multiples of `TILE` (M, N) and 16 (K). The `@..@` markers are filled in by `sgemm_source`.
const SGEMM: &str = r#"
struct Params { m: u32, n: u32, k: u32, relu: u32 }
@group(1) @binding(0) var<storage, read> a: array<f32>;
@group(1) @binding(1) var<storage, read> b: array<f32>;
@group(1) @binding(2) var<storage, read_write> c: array<f32>;
@group(1) @binding(3) var<uniform> p: Params;
@BIAS@
var<workgroup> at: array<vec4<f32>, 512>;
var<workgroup> bt: array<vec4<f32>, 512>;

// Slot `e` of a slice stored k-major: k `e % 16`, rows `4 (e / 16) ..`.
fn rows4(m: u32, k0: u32, e: u32) -> u32 { return (m + (e / 16u) * 4u) * p.k + k0 + e % 16u; }
fn k_major(e: u32) -> u32 { return (e % 16u) * 32u + e / 16u; }
fn load_a(row0: u32, k0: u32, e: u32) -> vec4<f32> {
    let i = rows4(row0, k0, e);
    return vec4<f32>(a[i], a[i + p.k], a[i + 2u * p.k], a[i + 3u * p.k]);
}
@B_LOAD@

@compute @workgroup_size(16, 16)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let row0 = wid.y * 128u;
    let col0 = wid.x * 128u;
    let t = lid.y * 16u + lid.x;
    @ACC_DECL@
    for (var k0 = 0u; k0 < p.k; k0 = k0 + 16u) {
        at[k_major(t)] = load_a(row0, k0, t);
        at[k_major(t + 256u)] = load_a(row0, k0, t + 256u);
        bt[b_slot(t)] = load_b(col0, k0, t);
        bt[b_slot(t + 256u)] = load_b(col0, k0, t + 256u);
        workgroupBarrier();
        for (var kk = 0u; kk < 16u; kk = kk + 2u) {
            @ACC_UPDATE@
        }
        workgroupBarrier();
    }
    @STORE@
}
"#;

#[derive(encase::ShaderType)]
pub struct SgemmParams {
    pub m: u32,
    pub n: u32,
    pub k: u32,
    /// Whether the biased kernel applies ReLU after the bias.
    pub relu: u32,
}

/// `B` is `k x n`, or `n x k` with `b_transposed`. Arguments `a, b, c, params`, plus `bias` (one
/// per column, added before the optional ReLU) when `biased`.
pub fn sgemm_source(b_transposed: bool, biased: bool) -> String {
    let b_load = if b_transposed {
        // Rows of B are columns of C: the same layout as A.
        "fn b_slot(e: u32) -> u32 { return k_major(e); }
fn load_b(col0: u32, k0: u32, e: u32) -> vec4<f32> {
    let i = rows4(col0, k0, e);
    return vec4<f32>(b[i], b[i + p.k], b[i + 2u * p.k], b[i + 3u * p.k]);
}"
    } else {
        // Slot `e`: k `e / 32`, columns `4 (e % 32) ..`, contiguous in a row of B.
        "fn b_slot(e: u32) -> u32 { return e; }
fn load_b(col0: u32, k0: u32, e: u32) -> vec4<f32> {
    let i = (k0 + e / 32u) * p.n + col0 + (e % 32u) * 4u;
    return vec4<f32>(b[i], b[i + 1u], b[i + 2u], b[i + 3u]);
}"
    };
    let (bias, epilogue) = if biased {
        (
            "@group(1) @binding(4) var<storage, read> bias: array<f32>;",
            "v = v + vec4<f32>(bias[col], bias[col + 1u], bias[col + 2u], bias[col + 3u]);
        if p.relu != 0u { v = max(v, vec4<f32>(0.0)); }",
        )
    } else {
        ("", "")
    };
    // Accumulator `c{2 i + h}` holds row `i` (of 8), columns half `h`. The loops are unrolled
    // here so the accumulators stay in registers.
    let mut decl = String::new();
    let mut update = String::new();
    let mut store = String::new();
    for i in 0..8 {
        for h in 0..2 {
            decl += &format!("var c{} = vec4<f32>(0.0);\n    ", i * 2 + h);
            store += &format!(
                "{{
        let r = row0 + {} + lid.y * 4u;
        let col = col0 + {} + lid.x * 4u;
        var v = c{};
        {epilogue}
        let i = r * p.n + col;
        c[i] = v.x; c[i + 1u] = v.y; c[i + 2u] = v.z; c[i + 3u] = v.w;
    }}\n    ",
                (i / 4) * 64 + i % 4,
                h * 64,
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
            let a_row = format!("a{}.{}", i / 4, ["x", "y", "z", "w"][i % 4]);
            for h in 0..2 {
                let acc = format!("c{}", i * 2 + h);
                update += &format!("{acc} = fma(vec4<f32>({a_row}), b{h}, {acc});\n                ");
            }
        }
        update += "}\n            ";
    }
    SGEMM
        .replace("@BIAS@", bias)
        .replace("@B_LOAD@", b_load)
        .replace("@ACC_DECL@", &decl)
        .replace("@ACC_UPDATE@", &update)
        .replace("@STORE@", &store)
}

/// The cooperative-matrix GEMM a GPU runs.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum CoopKind {
    /// f16 operands, f32 accumulation on 16x16 fragments (tensor cores); 128x128 tiles.
    F16,
    /// Exact f32 on 8x8 fragments (simdgroup matrices); 64x64 tiles.
    F32,
}

impl CoopKind {
    pub fn for_context(ctx: &luma::Context) -> Option<CoopKind> {
        if !ctx.features().contains(wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX) {
            return None;
        }
        match ctx.tuning_profile() {
            luma::TuningProfile::Nvidia => Some(CoopKind::F16),
            luma::TuningProfile::Apple => Some(CoopKind::F32),
            _ => None,
        }
    }

    /// Output tile edge; M and N must be multiples of it and K of 32.
    pub fn tile(self) -> usize {
        match self {
            CoopKind::F16 => 128,
            CoopKind::F32 => 64,
        }
    }

    /// The check tolerance for a result compared with an f32 reference: the f16 variant rounds
    /// its operands to 11 bits, so it gets `f16_tolerance`.
    pub fn tolerance(self, f32_tolerance: f32, f16_tolerance: f32) -> f32 {
        match self {
            CoopKind::F16 => f16_tolerance,
            CoopKind::F32 => f32_tolerance,
        }
    }

    /// Same arguments and semantics as `sgemm_source`.
    pub fn source(self, b_transposed: bool, biased: bool) -> String {
        let (bias, add_bias) = if biased {
            (
                "@group(1) @binding(4) var<storage, read> bias: array<f32>;",
                "v = v + bias[col];\n        if p.relu != 0u { v = max(v, 0.0); }",
            )
        } else {
            ("", "")
        };
        match self {
            CoopKind::F16 => f16_source(b_transposed, bias, add_bias),
            CoopKind::F32 => f32_source(b_transposed, bias, add_bias, biased),
        }
    }
}

/// One workgroup of 4 subgroups computes a 128x128 tile, each subgroup 64x64 as 4x4 fragments.
/// Slices of 32 of K are converted to f16 into double-buffered workgroup memory (rows padded by 8
/// against bank conflicts) while the previous slice is consumed; the next slice sits in
/// registers meanwhile. Fragments go out through workgroup memory because naga's SPIR-V output
/// cannot store a fragment with a runtime stride.
const F16: &str = r#"
enable f16;
enable wgpu_cooperative_matrix;
struct Params { m: u32, n: u32, k: u32, relu: u32 }
@group(1) @binding(0) var<storage, read> a: array<f32>;
@group(1) @binding(1) var<storage, read> b: array<f32>;
@group(1) @binding(2) var<storage, read_write> c: array<f32>;
@group(1) @binding(3) var<uniform> p: Params;
@BIAS@
var<workgroup> sa: array<f16, 10240>;
var<workgroup> sb: array<f16, @SB_SIZE@>;
var<workgroup> stage: array<f32, 1024>;

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) t: u32) {
    let row0 = wid.y * 128u;
    let col0 = wid.x * 128u;
    let sg = t / 32u;
    let lane = t % 32u;
    let wm = (sg / 2u) * 64u;
    let wn = (sg % 2u) * 64u;
    @ACC_DECL@
    @PREFETCH_DECL@
    @FETCH_FIRST@
    @COMMIT_FIRST@
    workgroupBarrier();
    for (var k0 = 0u; k0 < p.k; k0 = k0 + 32u) {
        let cur = (k0 / 32u) % 2u;
        let more = k0 + 32u < p.k;
        if more {
            @FETCH_NEXT@
        }
        let abase = cur * 5120u;
        let bbase = cur * @B_HALF@u;
        @FRAGMENTS@
        if more {
            @COMMIT_NEXT@
        }
        workgroupBarrier();
    }
    @STORE@
}
"#;

/// Statements that load (`fetch`) the slice of A and B starting at `k` into registers, or store
/// those registers, as f16, into workgroup buffer `buf`. Each thread holds 32 values of each.
fn stage(b_transposed: bool, fetch: bool, k: &str, buf: &str) -> String {
    let mut out = String::new();
    for i in 0..32 {
        let (idx, slot) = (
            format!("a[(row0 + t / 32u + {}u) * p.k + {k} + t % 32u]", 4 * i),
            format!("sa[{buf} * 5120u + (t / 32u + {}u) * 40u + t % 32u]", 4 * i),
        );
        out += &if fetch { format!("pa{i} = {idx};\n") } else { format!("{slot} = f16(pa{i});\n") };
    }
    for i in 0..32 {
        let (idx, slot) = if b_transposed {
            // Rows of B are columns of C: staged like A.
            (
                format!("b[(col0 + t / 32u + {}u) * p.k + {k} + t % 32u]", 4 * i),
                format!("sb[{buf} * 5120u + (t / 32u + {}u) * 40u + t % 32u]", 4 * i),
            )
        } else {
            // `t / 128u` is always 0; the plain `t` form measured 1.7x slower on the 3090.
            (
                format!("b[({k} + t / 128u + {i}u) * p.n + col0 + t % 128u]"),
                format!("sb[{buf} * 4352u + (t / 128u + {i}u) * 136u + t % 128u]"),
            )
        };
        out += &if fetch { format!("pb{i} = {idx};\n") } else { format!("{slot} = f16(pb{i});\n") };
    }
    out
}

fn f16_source(b_transposed: bool, bias: &str, add_bias: &str) -> String {
    let b_half = if b_transposed { 5120 } else { 4352 };
    let prefetch: String = (0..32).map(|i| format!("var pa{i} = 0.0;\nvar pb{i} = 0.0;\n")).collect();
    let mut decl = String::new();
    let mut frags = String::new();
    let mut store = String::new();
    for i in 0..4 {
        for j in 0..4 {
            decl += &format!("var c{i}_{j} = coopLoad<coop_mat16x16<f32, C>>(&stage[0], 16u);\n    ");
            store += &format!(
                "coopStoreT(c{i}_{j}, &stage[sg * 256u], 16u);
    workgroupBarrier();
    for (var e = lane; e < 256u; e = e + 32u) {{
        let col = col0 + wn + {}u + e % 16u;
        var v = stage[sg * 256u + e];
        {add_bias}
        c[(row0 + wm + {}u + e / 16u) * p.n + col] = v;
    }}
    workgroupBarrier();
    ",
                16 * j,
                16 * i
            );
        }
    }
    // B fragments first, then each A fragment right before its products: fewer live registers.
    for kk in [0, 16] {
        frags += "{\n        ";
        for j in 0..4 {
            frags += &if b_transposed {
                format!(
                    "let b{j} = coopLoad<coop_mat16x16<f16, B>>(&sb[bbase + (wn + {}u) * 40u + {kk}u], 40u);\n        ",
                    16 * j
                )
            } else {
                format!(
                    "let b{j} = coopLoadT<coop_mat16x16<f16, B>>(&sb[bbase + {}u + wn + {}u], 136u);\n        ",
                    kk * 136,
                    16 * j
                )
            };
        }
        for i in 0..4 {
            frags += &format!(
                "let a{i} = coopLoadT<coop_mat16x16<f16, A>>(&sa[abase + (wm + {}u) * 40u + {kk}u], 40u);\n        ",
                16 * i
            );
            for j in 0..4 {
                frags += &format!("c{i}_{j} = coopMultiplyAdd(a{i}, b{j}, c{i}_{j});\n        ");
            }
        }
        frags += "}\n        ";
    }
    F16.replace("@BIAS@", bias)
        .replace("@PREFETCH_DECL@", &prefetch)
        .replace("@FETCH_FIRST@", &stage(b_transposed, true, "0u", ""))
        .replace("@COMMIT_FIRST@", &stage(b_transposed, false, "", "0u"))
        .replace("@FETCH_NEXT@", &stage(b_transposed, true, "(k0 + 32u)", ""))
        .replace("@COMMIT_NEXT@", &stage(b_transposed, false, "", "(1u - cur)"))
        .replace("@B_HALF@", &b_half.to_string())
        .replace("@SB_SIZE@", &(2 * b_half).to_string())
        .replace("@ACC_DECL@", &decl)
        .replace("@FRAGMENTS@", &frags)
        .replace("@STORE@", &store)
}

/// One workgroup of 2 subgroups computes a 64x64 tile, each subgroup 64x32 as 8x4 fragments
/// loaded straight from global memory, which Metal serves from cache at the f32 ALU peak. The
/// direct store is Metal only (see `F16`).
const F32: &str = r#"
enable wgpu_cooperative_matrix;
struct Params { m: u32, n: u32, k: u32, relu: u32 }
@group(1) @binding(0) var<storage, read> a: array<f32>;
@group(1) @binding(1) var<storage, read> b: array<f32>;
@group(1) @binding(2) var<storage, read_write> c: array<f32>;
@group(1) @binding(3) var<uniform> p: Params;
@BIAS@
var<workgroup> stage: array<f32, 128>;

fn store_tile(m: coop_mat8x8<f32, C>, sg: u32, lane: u32, row: u32, col0: u32) {
    @STORE_TILE@
}

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) t: u32) {
    let sg = t / 32u;
    let row0 = wid.y * 64u;
    let col0 = wid.x * 64u + sg * 32u;
    @ACC_DECL@
    for (var k0 = 0u; k0 < p.k; k0 = k0 + 8u) {
        @FRAGMENTS@
    }
    @STORE@
}
"#;

fn f32_source(b_transposed: bool, bias: &str, add_bias: &str, biased: bool) -> String {
    let store_tile = if biased {
        format!("coopStoreT(m, &stage[sg * 64u], 8u);
    workgroupBarrier();
    for (var e = lane; e < 64u; e = e + 32u) {{
        let col = col0 + e % 8u;
        var v = stage[sg * 64u + e];
        {add_bias}
        c[(row + e / 8u) * p.n + col] = v;
    }}
    workgroupBarrier();")
    } else {
        "coopStoreT(m, &c[row * p.n + col0], p.n);".to_string()
    };
    let mut decl = String::new();
    let mut frags = String::new();
    let mut store = String::new();
    for i in 0..8 {
        for j in 0..4 {
            decl += &format!("var c{i}_{j} = coopLoad<coop_mat8x8<f32, C>>(&stage[0], 8u);\n    ");
            store += &format!("store_tile(c{i}_{j}, sg, t % 32u, row0 + {}u, col0 + {}u);\n    ", 8 * i, 8 * j);
        }
        frags += &format!("let a{i} = coopLoadT<coop_mat8x8<f32, A>>(&a[(row0 + {}u) * p.k + k0], p.k);\n        ", 8 * i);
    }
    for j in 0..4 {
        frags += &if b_transposed {
            format!("let b{j} = coopLoad<coop_mat8x8<f32, B>>(&b[(col0 + {}u) * p.k + k0], p.k);\n        ", 8 * j)
        } else {
            format!("let b{j} = coopLoadT<coop_mat8x8<f32, B>>(&b[k0 * p.n + col0 + {}u], p.n);\n        ", 8 * j)
        };
    }
    for i in 0..8 {
        for j in 0..4 {
            frags += &format!("c{i}_{j} = coopMultiplyAdd(a{i}, b{j}, c{i}_{j});\n        ");
        }
    }
    F32.replace("@BIAS@", bias)
        .replace("@STORE_TILE@", &store_tile)
        .replace("@ACC_DECL@", &decl)
        .replace("@FRAGMENTS@", &frags)
        .replace("@STORE@", &store)
}
