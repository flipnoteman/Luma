//! Compile cost: kernels the context has never built (fresh structures every call) against the
//! same expressions with a warm in-process cache, and what does *not* need a new kernel (other
//! scalar values, other lengths).

use super::common::*;
use crate::harness::*;
use luma::*;

pub fn run(h: &mut Harness) {
    h.guarded("compile", |h| {
        let ctx = h.ctx.clone();
        let n = 4096;
        let k = size(h, 4, 6, 8);
        // `compile/shape_variants` takes one more element per kernel: room for ~64k kernels.
        let big = 65536;
        let xh = values(big, 1, 0.5, 1.5);
        let yh = values(big, 2, 0.5, 1.5);
        let x = A::from_slice_on(&ctx, &xh[..n], n).unwrap();
        let y = A::from_slice_on(&ctx, &yh[..n], n).unwrap();

        // Verification: a recipe on the device against the host.
        let steps = fresh_recipe(12);
        check("compile/chain12", &apply(&steps, &x, &y).to_vec().unwrap(), &apply_host(&steps, &xh[..n], &yh[..n]), 2e-3, 2e-3);

        let size_k = format!("{k}x12op n={n}");
        h.case("compile/cold_chain12", &size_k, per_sec(k, "kernel/s"), &mut || {
            for _ in 0..k {
                force(&apply(&fresh_recipe(12), &x, &y));
            }
        });

        let fixed: Vec<Vec<Step>> = (0..k).map(|i| recipe(100 + i as u64, 12)).collect();
        h.case("compile/warm_chain12", &size_k, per_sec(k, "kernel/s"), &mut || {
            for steps in &fixed {
                force(&apply(steps, &x, &y));
            }
        });

        // The same structures with other scalar values: a kernel is built once.
        let mut round = 0u64;
        let scalar_base = {
            let base = recipe(7, 12);
            let mut rng = Rng(99);
            let other: Vec<Step> = base.iter().map(|s| s.rescaled(&mut rng)).collect();
            check("compile/scalars", &apply(&other, &x, &y).to_vec().unwrap(), &apply_host(&other, &xh[..n], &yh[..n]), 2e-3, 2e-3);
            base
        };
        h.case("compile/scalar_variants", &size_k, per_sec(k, "kernel/s"), &mut || {
            round += 1;
            let mut rng = Rng(round.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
            for _ in 0..k {
                let steps: Vec<Step> = scalar_base.iter().map(|s| s.rescaled(&mut rng)).collect();
                force(&apply(&steps, &x, &y));
            }
        });

        // The same structure at lengths never seen: a kernel that bakes the length in is built again.
        let mut next_len = 1024usize;
        let shape_steps = recipe(11, 12);
        h.case("compile/shape_variants", &size_k, per_sec(k, "kernel/s"), &mut || {
            for _ in 0..k {
                // One element more per kernel: ~64k lengths never seen before; past the host buffer the lengths
                // come round again (kernels then come from the cache, which only the warm-up calls can reach).
                next_len = if next_len + 1 > big { 1025 } else { next_len + 1 };
                let (xa, ya) = (A::from_slice_on(&ctx, &xh[..next_len], next_len).unwrap(), A::from_slice_on(&ctx, &yh[..next_len], next_len).unwrap());
                force(&apply(&shape_steps, &xa, &ya));
            }
        });

        // Views inside a fused kernel: pad, four slices and a fresh elementwise program.
        let (w, hh) = (64, 64);
        let xs = A::from_slice_on(&ctx, &xh[..w * hh], [w, hh]).unwrap();
        let ys = A::from_slice_on(&ctx, &yh[..w * hh], [w, hh]).unwrap();
        let stencil = |steps: &[Step]| {
            let e = apply(steps, &xs, &ys);
            &e + lap_edge(&e, w, hh) * 0.1
        };
        {
            let steps = fresh_recipe(8);
            let e = to_f64(&apply_host(&steps, &xh[..w * hh], &yh[..w * hh]));
            let lap = lap_edge_host(&e, w, hh);
            let want: Vec<f32> = e.iter().zip(&lap).map(|(e, l)| (e + 0.1 * l) as f32).collect();
            check("compile/stencil", &stencil(&steps).to_vec().unwrap(), &want, 2e-3, 2e-3);
        }
        let size_s = format!("{}x8op+pad {w}x{hh}", k.min(6));
        let ks = k.min(6);
        h.case("compile/cold_stencil", &size_s, per_sec(ks, "kernel/s"), &mut || {
            for _ in 0..ks {
                force(&stencil(&fresh_recipe(8)));
            }
        });
        let fixed_s: Vec<Vec<Step>> = (0..ks).map(|i| recipe(200 + i as u64, 8)).collect();
        h.case("compile/warm_stencil", &size_s, per_sec(ks, "kernel/s"), &mut || {
            for steps in &fixed_s {
                force(&stencil(steps));
            }
        });

        // Reductions are kernels of their own (and finish in one or two launches).
        {
            let steps = fresh_recipe(12);
            let want: f64 = to_f64(&apply_host(&steps, &xh[..n], &yh[..n])).iter().sum();
            check_scalar("compile/reduce", f64::from(apply(&steps, &x, &y).sum().item().unwrap()), want, 1e-3);
        }
        h.case("compile/cold_reduce", &format!("{ks}x12op sum n={n}"), per_sec(ks, "kernel/s"), &mut || {
            for _ in 0..ks {
                force(&apply(&fresh_recipe(12), &x, &y).sum());
            }
        });

        // One multi-output kernel of 8 outputs: compile cost grows with the program, not the launch.
        let outs = 8;
        {
            let progs: Vec<Vec<Step>> = (0..outs).map(|_| fresh_recipe(8)).collect();
            let arrays: Vec<A> = progs.iter().map(|p| apply(p, &x, &y)).collect();
            eval_all(&arrays.iter().collect::<Vec<_>>()).unwrap();
            for (p, a) in progs.iter().zip(&arrays) {
                check("compile/multiout", &a.to_vec().unwrap(), &apply_host(p, &xh[..n], &yh[..n]), 2e-3, 2e-3);
            }
        }
        let programs = size(h, 1, 2, 2);
        h.case("compile/cold_multiout", &format!("{programs}x{outs}outx8op n={n}"), per_sec(programs, "kernel/s"), &mut || {
            for _ in 0..programs {
                let arrays: Vec<A> = (0..outs).map(|_| apply(&fresh_recipe(8), &x, &y)).collect();
                eval_all(&arrays.iter().collect::<Vec<_>>()).unwrap();
                ctx.sync().unwrap();
            }
        });

        // One bigger kernel: compile time against program size.
        let kb = size(h, 1, 2, 2);
        {
            let steps = fresh_recipe(64);
            check("compile/chain64", &apply(&steps, &x, &y).to_vec().unwrap(), &apply_host(&steps, &xh[..n], &yh[..n]), 2e-3, 2e-3);
        }
        h.case("compile/cold_chain64", &format!("{kb}x64op n={n}"), per_sec(kb, "kernel/s"), &mut || {
            for _ in 0..kb {
                force(&apply(&fresh_recipe(64), &x, &y));
            }
        });

        // Df32 (double-float) arithmetic expands every operation into many: bigger programs to build.
        let kd = k.min(4);
        {
            let steps = fresh_df_recipe(8);
            check("compile/df32", &apply_df(&steps, &x, &y).to_vec().unwrap(), &apply_host(&steps, &xh[..n], &yh[..n]), 2e-3, 2e-3);
        }
        h.case("compile/cold_df32", &format!("{kd}x8op df32 n={n}"), per_sec(kd, "kernel/s"), &mut || {
            for _ in 0..kd {
                force(&apply_df(&fresh_df_recipe(8), &x, &y));
            }
        });

        // A context opened in this process (what a new run of an application pays): the kernels
        // of the cases above come from the on-disk cache, not from the compiler. `newctx_empty`
        // is the opening plus one tiny kernel, `newctx_kernels` adds the `k` kernels of
        // `warm_chain12`.
        let opts = ContextOptions::from_env();
        {
            let c = Context::with_options(opts).unwrap();
            let (xa, ya) = (A::from_slice_on(&c, &xh[..n], n).unwrap(), A::from_slice_on(&c, &yh[..n], n).unwrap());
            check("compile/newctx", &apply(&fixed[0], &xa, &ya).to_vec().unwrap(), &apply_host(&fixed[0], &xh[..n], &yh[..n]), 2e-3, 2e-3);
        }
        h.transfer_case("compile/newctx_empty", "open + 1 op", per_sec(1, "context/s"), &mut || {
            let c = Context::with_options(opts).unwrap();
            force(&(&A::from_slice_on(&c, &xh[..n], n).unwrap() * 1.5 + 0.5));
        });
        h.transfer_case("compile/newctx_kernels", &format!("open + {size_k}"), per_sec(1, "context/s"), &mut || {
            let c = Context::with_options(opts).unwrap();
            let (xa, ya) = (A::from_slice_on(&c, &xh[..n], n).unwrap(), A::from_slice_on(&c, &yh[..n], n).unwrap());
            for steps in &fixed {
                force(&apply(steps, &xa, &ya));
            }
        });
    });
}
