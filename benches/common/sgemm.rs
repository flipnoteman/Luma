//! Tiled SGEMM kernel shared by the benchmarks (`#[path]`-included).

/// Output tile edge of the GEMM kernel.
pub const TILE: usize = 128;

/// Row-major `C = A B` (or `A Bᵀ`); each workgroup computes a 128x128 tile of `C`, each of its
/// 16x16 invocations 8x8 of it, from 16-wide slices of `A` and `B` staged in workgroup memory as
/// vec4s of four rows (columns), so that one 16-byte read feeds four accumulators. An invocation
/// owns two 4-wide groups of rows and of columns, 64 apart, so that workgroup memory reads of
/// neighbouring invocations are contiguous; it loads the next slice into registers while the
/// current one is consumed. Dimensions must be multiples of `TILE` (M, N) and 16 (K). The
/// `@..@` markers are filled in by `sgemm_source`.
const SGEMM: &str = r#"
struct Params { m: u32, n: u32, k: u32, relu: u32 }
@group(1) @binding(0) var<storage, read> a: array<f32>;
@group(1) @binding(1) var<storage, read> b: array<f32>;
@group(1) @binding(2) var<storage, read_write> c: array<f32>;
@group(1) @binding(3) var<uniform> p: Params;
@BIAS@
var<workgroup> at: array<vec4<f32>, 512>;
var<workgroup> bt: array<vec4<f32>, 512>;

// Slot `e` of a slice stored k-major: k `e % 16` of rows `4 (e / 16) ..`.
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
    var pa0 = load_a(row0, 0u, t);
    var pa1 = load_a(row0, 0u, t + 256u);
    var pb0 = load_b(col0, 0u, t);
    var pb1 = load_b(col0, 0u, t + 256u);
    for (var k0 = 0u; k0 < p.k; k0 = k0 + 16u) {
        at[k_major(t)] = pa0;
        at[k_major(t + 256u)] = pa1;
        bt[b_slot(t)] = pb0;
        bt[b_slot(t + 256u)] = pb1;
        workgroupBarrier();
        if (k0 + 16u < p.k) {
            pa0 = load_a(row0, k0 + 16u, t);
            pa1 = load_a(row0, k0 + 16u, t + 256u);
            pb0 = load_b(col0, k0 + 16u, t);
            pb1 = load_b(col0, k0 + 16u, t + 256u);
        }
        for (var kk = 0u; kk < 16u; kk = kk + 1u) {
            let a0 = at[kk * 32u + lid.y];
            let a1 = at[kk * 32u + lid.y + 16u];
            let b0 = bt[kk * 32u + lid.x];
            let b1 = bt[kk * 32u + lid.x + 16u];
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
        let a_row = format!("a{}.{}", i / 4, ["x", "y", "z", "w"][i % 4]);
        for h in 0..2 {
            let acc = format!("c{}", i * 2 + h);
            decl += &format!("var {acc} = vec4<f32>(0.0);\n    ");
            update += &format!("{acc} = fma(vec4<f32>({a_row}), b{h}, {acc});\n            ");
            store += &format!(
                "{{
        let r = row0 + {} + lid.y * 4u;
        let col = col0 + {} + lid.x * 4u;
        var v = {acc};
        {epilogue}
        let i = r * p.n + col;
        c[i] = v.x; c[i + 1u] = v.y; c[i + 2u] = v.z; c[i + 3u] = v.w;
    }}\n    ",
                (i / 4) * 64 + i % 4,
                h * 64
            );
        }
    }
    SGEMM
        .replace("@BIAS@", bias)
        .replace("@B_LOAD@", b_load)
        .replace("@ACC_DECL@", &decl)
        .replace("@ACC_UPDATE@", &update)
        .replace("@STORE@", &store)
}
