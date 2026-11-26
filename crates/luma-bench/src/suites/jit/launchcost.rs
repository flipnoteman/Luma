//! Host cost per launch of dependent chains of tiny launches: elementwise ops, reductions,
//! matmuls and a scalar readback each time. The arrays are small enough for the GPU to be idle
//! most of the time, so the time of a case divided by its launch count is the host's cost of one
//! launch (build the lazy expression, plan, look the kernel up, pack the arguments, enqueue).

use super::common::*;
use crate::harness::*;
use luma::*;

const N: usize = 1024;
const CHAIN: usize = 64;

pub fn run(h: &mut Harness) {
    h.guarded("launchcost", |h| {
        let ctx = h.ctx.clone();
        let xh = values(N, 5, 0.5, 1.5);
        let x = A::from_slice_on(&ctx, &xh, N).unwrap();

        // Dependent elementwise launches, one kernel with a new scalar every time.
        let ew = |x: &A| {
            let mut y = x.clone();
            for i in 0..CHAIN {
                y = &y * 1.0001 + (i as f32) * 1e-6;
                y.eval().unwrap();
            }
            y
        };
        let mut want = xh.clone();
        for i in 0..CHAIN {
            for v in &mut want {
                *v = *v * 1.0001 + (i as f32) * 1e-6;
            }
        }
        check("launchcost/ew", &ew(&x).to_vec().unwrap(), &want, 1e-4, 1e-4);
        h.case("launchcost/ew_chain64", &format!("n={N}"), per_sec(CHAIN, "launch/s"), &mut || force(&ew(&x)));

        // Independent reductions, each evaluated on its own.
        let red = |x: &A| {
            let outs: Vec<A> = (0..CHAIN).map(|i| (x * (1.0 + i as f32 * 1e-3)).sum()).collect();
            for o in &outs {
                o.eval().unwrap();
            }
            outs
        };
        let got = red(&x)[CHAIN - 1].to_vec().unwrap()[0];
        let want = xh.iter().map(|v| f64::from(*v) * (1.0 + (CHAIN - 1) as f64 * 1e-3)).sum::<f64>() as f32;
        check("launchcost/sum", &[got], &[want], 1e-4, 1e-3);
        h.case("launchcost/sum64", &format!("n={N}"), per_sec(CHAIN, "launch/s"), &mut || {
            red(&x);
            ctx.sync().unwrap();
        });

        // Chained small matmuls.
        let d = 32;
        let wh = values(d * d, 6, 0.0, 2.0 / d as f32);
        let ah = values(d * d, 7, 0.5, 1.5);
        let w = A::from_slice_on(&ctx, &wh, [d, d]).unwrap();
        let a = A::from_slice_on(&ctx, &ah, [d, d]).unwrap();
        let mm = |a: &A| {
            let mut y = a.clone();
            for _ in 0..CHAIN / 2 {
                y = matmul(&y, &w);
                y.eval().unwrap();
            }
            y
        };
        {
            let mut want = to_f64(&ah);
            for _ in 0..CHAIN / 2 {
                let mut next = vec![0.0f64; d * d];
                for i in 0..d {
                    for k in 0..d {
                        for j in 0..d {
                            next[i * d + j] += want[i * d + k] * f64::from(wh[k * d + j]);
                        }
                    }
                }
                want = next;
            }
            // The case measures the default path, which is TF32 where the context has it (decision V2):
            // 32 chained TF32 products compound a per-product error of about 2^-11 roughly linearly
            // (1.0e-3 observed on CUDA), so the tolerance follows the backend instead of the case
            // switching to `matmul_f32`, which would change what it measures.
            let rel = if ctx.numerics().tf32_matmul() { 1e-2 } else { 1e-3 };
            check("launchcost/matmul", &mm(&a).to_vec().unwrap(), &to_f32(&want), rel, 1e-4);
        }
        h.case("launchcost/matmul32", &format!("{d}x{d}"), per_sec(CHAIN / 2, "launch/s"), &mut || force(&mm(&a)));

        // A scalar read back after every launch: submit, wait, copy.
        let rb = |x: &A| {
            let mut acc = 0.0f32;
            for i in 0..16 {
                acc += (x * (1.0 + i as f32 * 1e-3)).sum().item().unwrap();
            }
            acc
        };
        std::hint::black_box(rb(&x));
        h.case("launchcost/readback16", &format!("n={N}"), per_sec(16, "readback/s"), &mut || {
            std::hint::black_box(rb(&x));
        });

        // Dependent launches of a custom kernel that declares its reach (`Kernel::launch` with the check of the declaration).
        {
            const SCALE: &str = "@group(1) @binding(0) var<storage, read> x: array<f32>;
@group(1) @binding(1) var<storage, read_write> y: array<f32>;
@group(1) @binding(2) var<uniform> k: f32;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) { if (gid.x < luma.size.x) { y[gid.x] = k * x[gid.x]; } }";
            let kernel = ctx.kernel(SCALE).elementwise().build().unwrap();
            let custom = |x: &A| {
                let mut y = x.clone();
                for _ in 0..CHAIN {
                    let mut out = A::zeros_on(&ctx, N).unwrap();
                    kernel.launch([N as u32], args![&y, &mut out, uniform(&1.0001f32)]).unwrap();
                    y = out;
                }
                y
            };
            let want: Vec<f32> = xh.iter().map(|v| (f64::from(*v) * 1.0001f64.powi(CHAIN as i32)) as f32).collect();
            check("launchcost/custom", &custom(&x).to_vec().unwrap(), &want, 1e-4, 1e-4);
            h.case("launchcost/custom64", &format!("n={N}"), per_sec(CHAIN, "launch/s"), &mut || force(&custom(&x)));
        }
    });
}
