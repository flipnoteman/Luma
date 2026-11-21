//! Time loops of small steps: the same step run eagerly (evaluate every step), as an
//! `iterate_all` register loop or replay, and as `iterate_until_all` / `compile_until_all` loops.

use super::common::*;
use crate::harness::*;
use luma::*;

type One = (A,);

fn elem_step(x: &A) -> A {
    x * 0.999 + 0.001
}

fn stencil_step(x: &A) -> A {
    (x.shift(1) + x.shift(-1)) * 0.25 + x * 0.5
}

/// `v <- 0.999 v + 0.001` repeated: the fixed point 1 is approached geometrically.
fn elem_host(x: &[f32], steps: usize) -> Vec<f32> {
    let decay = 0.999f64.powi(steps as i32);
    x.iter().map(|&v| (1.0 + (f64::from(v) - 1.0) * decay) as f32).collect()
}

fn stencil_host(x: &[f32], steps: usize) -> Vec<f32> {
    let n = x.len();
    let mut u = to_f64(x);
    let mut next = vec![0.0; n];
    for _ in 0..steps {
        next[0] = (u[n - 1] + u[1]) * 0.25 + u[0] * 0.5;
        next[n - 1] = (u[n - 2] + u[0]) * 0.25 + u[n - 1] * 0.5;
        for i in 1..n - 1 {
            next[i] = (u[i - 1] + u[i + 1]) * 0.25 + u[i] * 0.5;
        }
        std::mem::swap(&mut u, &mut next);
    }
    to_f32(&u)
}

/// A host loop: evaluate the state after every step (and read its maximum every 16).
fn eval_loop(x0: &A, steps: usize, step: fn(&A) -> A, read_every: usize) -> A {
    let mut u = x0.clone();
    let mut seen = 0.0;
    for t in 0..steps {
        u = step(&u);
        u.eval().unwrap();
        if read_every > 0 && t % read_every == read_every - 1 {
            seen += u.max().item().unwrap();
        }
    }
    std::hint::black_box(seen);
    u
}

fn until_loop(x0: &A, steps: usize, step: fn(&A) -> A) -> (A, usize) {
    let run = iterate_until_all(
        (x0.clone(),),
        Until::new(steps).record("max", |s: &One| s.0.max()).stop_when(|h| h["max"].last().gt(1e30)),
        |s, _| (step(&s.0),),
    )
    .unwrap();
    let recorded = run.history("max").map_or(0, <[f32]>::len);
    (run.state.0, recorded)
}

pub fn run(h: &mut Harness) {
    h.guarded("loop", |h| {
        let ctx = h.ctx.clone();
        let (n, steps) = (4096, size(h, 200, 1000, 1000));
        let xh = values(n, 4, 0.5, 1.5);
        let x0 = A::from_slice_on(&ctx, &xh, n).unwrap();
        let tag = format!("n={n} x{steps}");

        // Verification (a short loop against the host).
        let short = 24;
        let (want_e, want_s) = (elem_host(&xh, short), stencil_host(&xh, short));
        check("loop/eval_elem", &eval_loop(&x0, short, elem_step, 0).to_vec().unwrap(), &want_e, 1e-4, 1e-4);
        let [u] = iterate_all([&x0], short, |s| [elem_step(&s[0])]);
        check("loop/iterate_elem", &u.to_vec().unwrap(), &want_e, 1e-4, 1e-4);
        let [u] = iterate_all([&x0], short, |s| [stencil_step(&s[0])]);
        check("loop/iterate_stencil", &u.to_vec().unwrap(), &want_s, 1e-4, 1e-4);
        let (u, recorded) = until_loop(&x0, short, stencil_step);
        assert_eq!(recorded, short, "loop/until: history length");
        check("loop/until_stencil", &u.to_vec().unwrap(), &want_s, 1e-4, 1e-4);

        h.case("loop/eval_elem", &tag, per_sec(steps, "step/s"), &mut || {
            force(&eval_loop(&x0, steps, elem_step, 0));
        });
        h.case("loop/iterate_elem", &tag, per_sec(steps, "step/s"), &mut || {
            let [u] = iterate_all([&x0], steps, |s| [elem_step(&s[0])]);
            force(&u);
        });
        h.case("loop/eval_stencil", &tag, per_sec(steps, "step/s"), &mut || {
            force(&eval_loop(&x0, steps, stencil_step, 0));
        });
        h.case("loop/eval_stencil_read16", &tag, per_sec(steps, "step/s"), &mut || {
            force(&eval_loop(&x0, steps, stencil_step, 16));
        });
        h.case("loop/iterate_stencil", &tag, per_sec(steps, "step/s"), &mut || {
            let [u] = iterate_all([&x0], steps, |s| [stencil_step(&s[0])]);
            force(&u);
        });
        h.case("loop/until_stencil", &tag, per_sec(steps, "step/s"), &mut || {
            force(&until_loop(&x0, steps, stencil_step).0);
        });
        let mut compiled: Option<CompiledUntil<One>> = None;
        {
            let mut c = compile_until_all(
                &(x0.clone(),),
                Until::new(24).record("max", |s: &One| s.0.max()).stop_when(|h| h["max"].last().gt(1e30)),
                |s, _| (stencil_step(&s.0),),
            )
            .unwrap();
            let run = c.run((x0.clone(),)).unwrap();
            check("loop/compiled_stencil", &run.state.0.to_vec().unwrap(), &want_s, 1e-4, 1e-4);
        }
        h.case("loop/compiled_stencil", &tag, per_sec(steps, "step/s"), &mut || {
            let c = compiled.get_or_insert_with(|| {
                compile_until_all(
                    &(x0.clone(),),
                    Until::new(steps).record("max", |s: &One| s.0.max()).stop_when(|h| h["max"].last().gt(1e30)),
                    |s, _| (stencil_step(&s.0),),
                )
                .unwrap()
            });
            force(&c.run((x0.clone(),)).unwrap().state.0);
        });

        // A big state: bandwidth-bound steps (one read and one write of the state when fused).
        let nb = size(h, 1 << 20, 1 << 20, 1 << 22);
        let sb = size(h, 50, 50, 100);
        let xbh = values(nb, 5, 0.5, 1.5);
        let xb = A::from_slice_on(&ctx, &xbh, nb).unwrap();
        // The eval loop may be time-blocked and `iterate_all` replayed: both at the full step count.
        let want_big = stencil_host(&xbh, sb);
        check("loop/eval_stencil_big", &eval_loop(&xb, sb, stencil_step, 0).to_vec().unwrap(), &want_big, 1e-4, 1e-4);
        let [u] = iterate_all([&xb], sb, |s| [stencil_step(&s[0])]);
        check("loop/iterate_stencil_big", &u.to_vec().unwrap(), &want_big, 1e-4, 1e-4);
        let bytes = 8.0 * nb as f64 * sb as f64;
        let tag = format!("n={nb} x{sb}");
        h.case("loop/eval_stencil_big", &tag, Work::Bytes(bytes), &mut || {
            force(&eval_loop(&xb, sb, stencil_step, 0));
        });
        h.case("loop/iterate_stencil_big", &tag, Work::Bytes(bytes), &mut || {
            let [u] = iterate_all([&xb], sb, |s| [stencil_step(&s[0])]);
            force(&u);
        });

        // The elementwise loop as one register-resident kernel over a big state.
        let se = size(h, 200, 200, 1000);
        {
            let [u] = iterate_all([&xb], se, |s| [elem_step(&s[0])]);
            check("loop/iterate_elem_big", &u.to_vec().unwrap(), &elem_host(&xbh, se), 1e-4, 1e-4);
        }
        h.case("loop/iterate_elem_big", &format!("n={nb} x{se}"), Work::Cells((nb * se) as f64), &mut || {
            let [u] = iterate_all([&xb], se, |s| [elem_step(&s[0])]);
            force(&u);
        });
    });
}
