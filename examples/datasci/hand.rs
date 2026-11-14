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
        ctx.kernel(&sgemm::sgemm_source(false, false)).label("hand_sgemm").build().map(HandSgemm)
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
