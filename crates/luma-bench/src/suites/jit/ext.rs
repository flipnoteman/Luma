//! The public extension API (W2-EXT): a fused chain around a custom elementwise function against the same arithmetic written
//! with built-ins, and the host cost of a custom op's `apply` against a `Kernel::launch` of the same kernel.

use super::common::*;
use crate::harness::*;
use luma::ext::{Bind, OutShape, Strategy};
use luma::*;

const SCALE: &str = r#"
@group(1) @binding(0) var<storage, read> x: array<f32>;
@group(1) @binding(1) var<storage, read_write> y: array<f32>;
@group(1) @binding(2) var<uniform> k: f32;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= luma.size.x) { return; }
    y[gid.x] = k * x[gid.x];
}
"#;

const CHAIN: usize = 64;
const N: usize = 1024;

pub fn run(h: &mut Harness) {
    h.guarded("ext", |h| {
        let ctx = h.ctx.clone();

        // A fused chain around a custom function against the same arithmetic of built-ins.
        let n = size(h, 1 << 16, 1 << 18, 1 << 22);
        let xh = values(n, 11, 0.5, 1.5);
        let x = A::from_slice_on(&ctx, &xh, n).unwrap();
        let g = ctx.elementwise_fn::<f32>("bench_gelu", "fn bench_gelu(x: f32) -> f32 { return x * x * 0.5 + x * 0.25 - 0.125; }").unwrap();
        let custom = |x: &A| {
            let mut y = x * 1.0001;
            for _ in 0..4 {
                y = g.call1(&y) * 0.5 + 0.5;
            }
            y
        };
        let builtin = |x: &A| {
            let mut y = x * 1.0001;
            for _ in 0..4 {
                y = (&y * &y * 0.5 + &y * 0.25 - 0.125) * 0.5 + 0.5;
            }
            y
        };
        check("ext/fn_chain", &custom(&x).to_vec().unwrap(), &builtin(&x).to_vec().unwrap(), 1e-5, 1e-5);
        let bytes = (n * 8) as f64;
        h.case("ext/fn_chain", &format!("n={n}"), Work::Bytes(bytes), &mut || force(&custom(&x)));
        h.case("ext/fn_chain_builtin", &format!("n={n}"), Work::Bytes(bytes), &mut || force(&builtin(&x)));

        // The host cost of a custom op's `apply` against `Kernel::launch` of the same kernel.
        let xs = A::from_slice_on(&ctx, &values(N, 12, 0.5, 1.5), N).unwrap();
        let kernel = ctx.kernel(SCALE).build().unwrap();
        let op = ctx
            .custom_op("bench_scale")
            .input::<f32>()
            .output::<f32>(OutShape::Same(0))
            .params(4)
            .strategy(Strategy::new("plain", move |lower| {
                let kernel = lower.kernel(SCALE).build()?;
                let params = lower.params().to_vec();
                lower.launch(&kernel, [N as u32, 1, 1], &[Bind::Input(0), Bind::Output(0), Bind::Uniform(params)])
            }))
            .build()
            .unwrap();
        let params = luma::ext::params_bytes(&1.0001f32).unwrap();
        let apply = |x: &A| {
            let mut y = x.clone();
            for _ in 0..CHAIN {
                y = op.apply1::<f32>(&[&y], &params).unwrap();
            }
            y
        };
        let launch = |x: &A| {
            let mut y = x.clone();
            for _ in 0..CHAIN {
                let mut out = A::zeros_on(&ctx, N).unwrap();
                kernel.launch([N as u32], args![&y, &mut out, uniform(&1.0001f32)]).unwrap();
                y = out;
            }
            y
        };
        check("ext/op_apply", &apply(&xs).to_vec().unwrap(), &launch(&xs).to_vec().unwrap(), 1e-5, 1e-5);
        h.case("ext/op_apply", &format!("n={N}"), per_sec(CHAIN, "apply/s"), &mut || force(&apply(&xs)));
        h.case("ext/kernel_launch", &format!("n={N}"), per_sec(CHAIN, "launch/s"), &mut || force(&launch(&xs)));
    });
}
