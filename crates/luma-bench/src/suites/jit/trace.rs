//! Recording and replay: the trace cost of a 30-kernel step (`iterate_all`), the replay cost per
//! step, `iterate_until_all` with records and a stop rule, `compile_until_all`, chunked runs (the
//! pattern of a solver that looks at the state between chunks) and loops whose kernels are new.

use super::common::*;
use crate::harness::*;
use luma::*;

type One = (A,);

const STAGES: usize = 30;

/// `k` diffusion stages, each a kernel of its own (the stage is cached): `s += w * laplacian(s)`
/// with circular neighbours and a weight per stage.
fn stages(u: &A, k: usize) -> A {
    let mut s = u.clone();
    for j in 0..k {
        let w = 0.05 + 0.002 * j as f32;
        s = (&s + lap_wrap(&s) * w).cache();
    }
    s
}

fn stages_host(u: &[f32], w: usize, h: usize, k: usize) -> Vec<f32> {
    let mut s = to_f64(u);
    for j in 0..k {
        let wt = f64::from(0.05 + 0.002 * j as f32);
        let lap = lap_wrap_host(&s, w, h);
        for (s, l) in s.iter_mut().zip(&lap) {
            *s += wt * l;
        }
    }
    to_f32(&s)
}

/// Stages whose kernels are new: `s = tanh(s + 0.02 * laplacian(g_j(s)))` with a recipe `g_j` each.
fn fresh_stages(u: &A, recipes: &[Vec<Step>]) -> A {
    let mut s = u.clone();
    for steps in recipes {
        let g = apply(steps, &s, &s);
        s = (&s + lap_wrap(&g) * 0.02).tanh().cache();
    }
    s
}

fn fresh_stages_host(u: &[f32], w: usize, h: usize, recipes: &[Vec<Step>]) -> Vec<f32> {
    let mut s = u.to_vec();
    for steps in recipes {
        let g = to_f64(&apply_host(steps, &s, &s));
        let lap = lap_wrap_host(&g, w, h);
        s = s.iter().zip(&lap).map(|(&s, l)| (f64::from(s) + 0.02 * l).tanh() as f32).collect();
    }
    s
}

fn until_rec(state: One, steps: usize, step: &dyn Fn(&A) -> A, records: usize) -> Run<One> {
    let mut until = Until::new(steps).record("max", |s: &One| s.0.max());
    if records > 1 {
        until = until.record("norm", |s: &One| s.0.norm());
    }
    iterate_until_all(state, until.stop_when(|h| h["max"].last().gt(1e30)), |s, _| (step(&s.0),)).unwrap()
}

pub fn run(h: &mut Harness) {
    h.guarded("trace", |h| {
        let ctx = h.ctx.clone();
        let side = 64;
        let uh = values(side * side, 11, 0.0, 1.0);
        let u = A::from_slice_on(&ctx, &uh, [side, side]).unwrap();
        let stage_step = |a: &A| stages(a, STAGES);

        // Verification: a few steps of the traced loop against the host, history included.
        {
            let k = 3;
            let [r] = iterate_all([&u], k, |s| [stage_step(&s[0])]);
            let mut want = uh.clone();
            let mut maxima = Vec::new();
            for _ in 0..k {
                want = stages_host(&want, side, side, STAGES);
                maxima.push(want.iter().copied().fold(f32::MIN, f32::max));
            }
            check("trace/step30", &r.to_vec().unwrap(), &want, 1e-3, 1e-4);
            for records in [1, 2] {
                let run = until_rec((u.clone(),), k, &stage_step, records);
                check("trace/until30", &run.state.0.to_vec().unwrap(), &want, 1e-3, 1e-4);
                check("trace/until30 history", run.history("max").unwrap(), &maxima, 1e-3, 1e-4);
            }
        }

        let tag = |steps: usize| format!("{STAGES}k {side}x{side} x{steps}");
        h.case("trace/step30_x1", &tag(1), per_sec(1, "trace/s"), &mut || {
            let [r] = iterate_all([&u], 1, |s| [stage_step(&s[0])]);
            force(&r);
        });
        let steps = size(h, 50, 200, 200);
        h.case("trace/step30_replay", &tag(steps), per_sec(steps * STAGES, "launch/s"), &mut || {
            let [r] = iterate_all([&u], steps, |s| [stage_step(&s[0])]);
            force(&r);
        });

        // 16 independent fields, a stage each: the launches of a step share no buffer, so a graph
        // may run them side by side.
        const WIDE: usize = 16;
        let fields: Vec<A> = (0..WIDE).map(|i| A::from_slice_on(&ctx, &values(side * side, 20 + i as u64, 0.0, 1.0), [side, side]).unwrap()).collect();
        let wide_init: [&A; WIDE] = std::array::from_fn(|i| &fields[i]);
        let wide_step = |s: &[A; WIDE]| -> [A; WIDE] { std::array::from_fn(|i| stages(&s[i], 1)) };
        {
            let got = iterate_all(wide_init, 2, wide_step);
            for i in [0, WIDE - 1] {
                let want = stages_host(&stages_host(&values(side * side, 20 + i as u64, 0.0, 1.0), side, side, 1), side, side, 1);
                check("trace/wide16", &got[i].to_vec().unwrap(), &want, 1e-3, 1e-4);
            }
        }
        h.case("trace/wide16_replay", &format!("{WIDE}k {side}x{side} x{steps}"), per_sec(steps * WIDE, "launch/s"), &mut || {
            let r = iterate_all(wide_init, steps, wide_step);
            force(&r[0]);
        });

        // One evolving field and six that the step hands back unchanged: they take no launch.
        const PASS: usize = 7;
        let pass_init: [&A; PASS] = std::array::from_fn(|i| &fields[i]);
        let pass_step = |s: &[A; PASS]| -> [A; PASS] { std::array::from_fn(|i| if i == 0 { stage_step(&s[0]) } else { &s[i] * 1.0 }) };
        {
            let got = iterate_all(pass_init, 2, pass_step);
            let want = stages_host(&stages_host(&values(side * side, 20, 0.0, 1.0), side, side, STAGES), side, side, STAGES);
            check("trace/pass6", &got[0].to_vec().unwrap(), &want, 1e-3, 1e-4);
            check("trace/pass6 unchanged", &got[PASS - 1].to_vec().unwrap(), &values(side * side, 20 + (PASS - 1) as u64, 0.0, 1.0), 0.0, 0.0);
        }
        h.case("trace/step30_pass6_replay", &format!("{STAGES}k+6 {side}x{side} x{steps}"), per_sec(steps * STAGES, "launch/s"), &mut || {
            let r = iterate_all(pass_init, steps, pass_step);
            force(&r[0]);
        });

        // A bigger state, where the launches are no longer free.
        let big = size(h, 256, 256, 512);
        let bh = values(big * big, 12, 0.0, 1.0);
        let ub = A::from_slice_on(&ctx, &bh, [big, big]).unwrap();
        let bsteps = size(h, 10, 10, 20);
        h.case("trace/step30_replay_big", &format!("{STAGES}k {big}x{big} x{bsteps}"), Work::Cells((big * big * STAGES * bsteps) as f64), &mut || {
            let [r] = iterate_all([&ub], bsteps, |s| [stage_step(&s[0])]);
            force(&r);
        });

        // iterate_until_all: one record (traced with the step) and two (a launch of their own).
        for records in [1, 2] {
            h.case(&format!("trace/until30_rec{records}"), &tag(steps), per_sec(steps * STAGES, "launch/s"), &mut || {
                force(&until_rec((u.clone(),), steps, &stage_step, records).state.0);
            });
        }
        h.case("trace/until30_big", &format!("{STAGES}k {big}x{big} x{bsteps}"), Work::Cells((big * big * STAGES * bsteps) as f64), &mut || {
            force(&until_rec((ub.clone(),), bsteps, &stage_step, 1).state.0);
        });

        // A compiled loop: traced on the first call, then only launches.
        let compile = |steps: usize| {
            compile_until_all(
                &(u.clone(),),
                Until::new(steps).record("max", |s: &One| s.0.max()).stop_when(|h| h["max"].last().gt(1e30)),
                |s, _| (stage_step(&s.0),),
            )
            .unwrap()
        };
        {
            let mut c = compile(3);
            let r = c.run((u.clone(),)).unwrap();
            check("trace/compiled30", &r.state.0.to_vec().unwrap(), &stages_host(&stages_host(&stages_host(&uh, side, side, STAGES), side, side, STAGES), side, side, STAGES), 1e-3, 1e-4);
        }
        let mut compiled: Option<CompiledUntil<One>> = None;
        h.case("trace/compiled30", &tag(steps), per_sec(steps * STAGES, "launch/s"), &mut || {
            let c = compiled.get_or_insert_with(|| compile(steps));
            force(&c.run((u.clone(),)).unwrap().state.0);
        });

        // Chunks: `chunks` runs of `chunk` steps with the state read between them.
        let (chunks, chunk) = (size(h, 5, 5, 10), 10);
        h.case("trace/chunked_until", &format!("{STAGES}k {side}x{side} {chunks}x{chunk}"), per_sec(chunks * chunk * STAGES, "launch/s"), &mut || {
            let mut state = (u.clone(),);
            let mut peak = 0.0f32;
            for _ in 0..chunks {
                let run = until_rec(state, chunk, &stage_step, 1);
                peak = peak.max(run.history("max").and_then(|m| m.last().copied()).unwrap_or(0.0));
                state = run.state;
            }
            std::hint::black_box(peak);
            force(&state.0);
        });
        let mut chunked: Option<CompiledUntil<One>> = None;
        h.case("trace/chunked_compiled", &format!("{STAGES}k {side}x{side} {chunks}x{chunk}"), per_sec(chunks * chunk * STAGES, "launch/s"), &mut || {
            let c = chunked.get_or_insert_with(|| compile(chunk));
            let mut state = (u.clone(),);
            let mut peak = 0.0f32;
            for _ in 0..chunks {
                let run = c.run_steps(state, chunk).unwrap();
                peak = peak.max(run.history("max").and_then(|m| m.last().copied()).unwrap_or(0.0));
                state = run.state;
            }
            std::hint::black_box(peak);
            force(&state.0);
        });

        // Loops whose kernels the context has never built: trace + compile + first launches. A recipe has 10 kinds of
        // step, so `len` steps are `10^len` kernel structures: with 3 (1000) the calls of one run and of earlier runs
        // (compiled kernels are cached on disk) soon met structures built before, and the case timed the state of the
        // disk cache (a new emitter revision, which empties it, "regressed" it 1.5-2x) rather than the compile.
        const FRESH_LEN: usize = 6;
        let fresh_n = size(h, 4, 5, 8);
        {
            let recipes: Vec<Vec<Step>> = (0..2).map(|_| fresh_recipe(FRESH_LEN)).collect();
            let run = until_rec((u.clone(),), 3, &|a| fresh_stages(a, &recipes), 1);
            let mut want = uh.clone();
            for _ in 0..3 {
                want = fresh_stages_host(&want, side, side, &recipes);
            }
            check("trace/fresh_until", &run.state.0.to_vec().unwrap(), &want, 1e-3, 1e-4);
        }
        h.case("trace/fresh_until", &format!("{fresh_n}k new {side}x{side} x20"), per_sec(fresh_n, "kernel/s"), &mut || {
            let recipes: Vec<Vec<Step>> = (0..fresh_n).map(|_| fresh_recipe(FRESH_LEN)).collect();
            force(&until_rec((u.clone(),), 20, &|a| fresh_stages(a, &recipes), 1).state.0);
        });
    });
}
