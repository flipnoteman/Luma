//! Launch latency of tiny arrays: one op, a fused 10-op chain, 100 independent ops. Compute is
//! nil, so these time the host: building the lazy expression, looking its kernel up, recording
//! the launch, submitting and waiting.

use super::common::*;
use crate::harness::*;

pub fn run(h: &mut Harness) {
    h.guarded("launch", |h| {
        let ctx = h.ctx.clone();
        let chain = recipe(5, 10);
        for n in [1usize, 4096] {
            let xh = values(n, 3, 0.5, 1.5);
            let x = A::from_slice_on(&ctx, &xh, n).unwrap();

            let want: Vec<f32> = xh.iter().map(|v| v * 1.5 + 0.25).collect();
            check("launch/op1", &(&x * 1.5 + 0.25).to_vec().unwrap(), &want, 1e-5, 1e-5);
            let mut c = 0.0f32;
            h.case(&format!("launch/op1_n{n}"), &format!("n={n}"), per_sec(1, "launch/s"), &mut || {
                c += 1e-3;
                force(&(&x * 1.5 + c));
            });

            if n > 1 {
                check("launch/chain10", &apply(&chain, &x, &x).to_vec().unwrap(), &apply_host(&chain, &xh, &xh), 2e-3, 2e-3);
                h.case(&format!("launch/chain10_n{n}"), &format!("10op n={n}"), per_sec(1, "launch/s"), &mut || {
                    force(&apply(&chain, &x, &x));
                });
            }
            let count = 100;
            // 100 launches that cannot share a kernel: the inputs have 100 different lengths.
            let inputs: Vec<(Vec<f32>, A)> = (0..count)
                .map(|i| {
                    let m = n + i;
                    let host = values(m, 40 + m as u64, 0.5, 1.5);
                    let a = A::from_slice_on(&ctx, &host, m).unwrap();
                    (host, a)
                })
                .collect();
            let separate = |inputs: &[(Vec<f32>, A)]| -> Vec<A> { inputs.iter().enumerate().map(|(i, (_, a))| a * (1.0 + i as f32) + 0.5).collect() };
            {
                let outs = separate(&inputs);
                for o in &outs {
                    o.eval().unwrap();
                }
                for i in [0, count - 1] {
                    let want: Vec<f32> = inputs[i].0.iter().map(|v| v * (1.0 + i as f32) + 0.5).collect();
                    check("launch/indep100", &outs[i].to_vec().unwrap(), &want, 1e-5, 1e-5);
                }
            }
            h.case(&format!("launch/indep{count}_n{n}"), &format!("{count}x2op n={n}.."), per_sec(count, "launch/s"), &mut || {
                let outs = separate(&inputs);
                for o in &outs {
                    o.eval().unwrap();
                }
                ctx.sync().unwrap();
            });

            // The same with 100 arrays of one length that share an input: evaluations postponed while
            // the device is busy are fused into multi-output kernels.
            {
                let outs: Vec<A> = (0..count).map(|i| &x * (1.0 + i as f32) + 0.5).collect();
                for o in &outs {
                    o.eval().unwrap();
                }
                for i in [0, count - 1] {
                    let want: Vec<f32> = xh.iter().map(|v| v * (1.0 + i as f32) + 0.5).collect();
                    check("launch/batch100", &outs[i].to_vec().unwrap(), &want, 1e-5, 1e-5);
                }
            }
            h.case(&format!("launch/batch{count}_n{n}"), &format!("{count}x2op n={n}"), per_sec(count, "op/s"), &mut || {
                let outs: Vec<A> = (0..count).map(|i| &x * (1.0 + i as f32) + 0.5).collect();
                for o in &outs {
                    o.eval().unwrap();
                }
                ctx.sync().unwrap();
            });
        }
    });
}
