//! Threads on one shared context against threads with a context each (and one thread alone):
//! every thread runs an `iterate_until_all` loop of a stencil step (24 elementwise operations,
//! then a padded five-point Laplacian), so the loops trace, compile and replay concurrently.

use std::sync::Barrier;

use super::common::*;
use crate::harness::*;
use luma::*;

type One = (A,);

fn step(u: &A, side: usize) -> A {
    let e = heavy(u);
    u + lap_edge(&e, side, side) * 0.05
}

fn work(ctx: &Context, uh: &[f32], side: usize, steps: usize) -> Vec<f32> {
    let u0 = A::from_slice_on(ctx, uh, [side, side]).unwrap();
    let run = iterate_until_all((u0,), Until::new(steps).check_every(16).record("max", |s: &One| s.0.max()), |s, _| (step(&s.0, side),)).unwrap();
    run.state.0.to_vec().unwrap()
}

/// Every context's thread runs `work` once, started together; each result is compared with `want`.
fn round(ctxs: &[Context], uh: &[f32], side: usize, steps: usize, want: &[f32]) {
    let barrier = Barrier::new(ctxs.len());
    std::thread::scope(|scope| {
        let threads: Vec<_> = ctxs
            .iter()
            .map(|ctx| {
                let barrier = &barrier;
                scope.spawn(move || {
                    barrier.wait();
                    work(ctx, uh, side, steps)
                })
            })
            .collect();
        for t in threads {
            let got = t.join().expect("a thread panicked");
            check("threads", &got, want, 1e-4, 1e-4);
        }
    });
}

pub fn run(h: &mut Harness) {
    h.guarded("threads", |h| {
        let threads = 4;
        let side = size(h, 64, 64, 64);
        let steps = size(h, 32, 32, 64);
        let uh = values(side * side, 18, 0.2, 1.0);

        // The host reference of the loop.
        let mut want: Vec<f32> = uh.clone();
        for _ in 0..steps {
            let e: Vec<f64> = want.iter().map(|&v| heavy_host(f64::from(v))).collect();
            let lap = lap_edge_host(&e, side, side);
            want = want.iter().zip(&lap).map(|(&u, l)| (f64::from(u) + 0.05 * l) as f32).collect();
        }

        let alone = vec![h.ctx.clone()];
        let shared: Vec<Context> = (0..threads).map(|_| h.ctx.clone()).collect();
        let private: Vec<Context> = (0..threads).map(|_| Context::new().unwrap()).collect();
        for (name, ctxs) in [("alone", &alone), ("shared", &shared), ("private", &private)] {
            check("threads", &work(&ctxs[0], &uh, side, steps), &want, 1e-3, 1e-3);
            let t = ctxs.len();
            let case = if t == 1 { "threads/alone".to_string() } else { format!("threads/{name}{t}") };
            let size = format!("{t} thr {side}x{side} x{steps}");
            let mut f = || round(ctxs, &uh, side, steps, &want);
            if name == "private" {
                // The kernels run on contexts of their own, which the harness profile cannot see.
                h.transfer_case(&case, &size, per_sec(t * steps, "step/s"), &mut f);
            } else {
                h.case(&case, &size, per_sec(t * steps, "step/s"), &mut f);
            }
        }
    });
}
