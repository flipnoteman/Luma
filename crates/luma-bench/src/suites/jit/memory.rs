//! Memory of a recorded 3D-like step: five fields on an `n^3` grid, a step of ten kernels (a
//! cached Laplacian per field, then a coupled update) run eagerly, as `iterate_all`, as
//! `iterate_until_all` and as a compiled loop. The footprint of each mode is measured on a
//! context of its own (`memory_stats` after the run, while its results and loops are alive), and
//! put in the case's size column; a second group checks that nothing stays resident once
//! everything is dropped.

use super::common::*;
use crate::harness::*;
use luma::*;

type S = [A; 5];

fn lap3(f: &A, n: usize) -> A {
    let p = f.pad([(1, 1); 3], PadMode::Edge);
    let at = |i: usize, j: usize, k: usize| p.slice(0, i..i + n).slice(1, j..j + n).slice(2, k..k + n);
    at(0, 1, 1) + at(2, 1, 1) + at(1, 0, 1) + at(1, 2, 1) + at(1, 1, 0) + at(1, 1, 2) - at(1, 1, 1) * 6.0
}

fn step3d(s: &S, n: usize) -> S {
    let lap: Vec<A> = s.iter().map(|f| lap3(f, n).cache()).collect();
    std::array::from_fn(|i| (&s[i] + &lap[i] * 0.05 + (&s[(i + 1) % 5] - &s[i]) * 0.01).cache())
}

fn step3d_host(s: &[Vec<f64>], n: usize) -> Vec<Vec<f64>> {
    let idx = |i: isize, j: isize, k: isize| {
        let c = |v: isize| v.clamp(0, n as isize - 1) as usize;
        c(i) + n * (c(j) + n * c(k))
    };
    let lap: Vec<Vec<f64>> = s
        .iter()
        .map(|f| {
            let mut out = vec![0.0; n * n * n];
            for k in 0..n as isize {
                for j in 0..n as isize {
                    for i in 0..n as isize {
                        out[idx(i, j, k)] = f[idx(i - 1, j, k)] + f[idx(i + 1, j, k)] + f[idx(i, j - 1, k)] + f[idx(i, j + 1, k)] + f[idx(i, j, k - 1)] + f[idx(i, j, k + 1)] - 6.0 * f[idx(i, j, k)];
                    }
                }
            }
            out
        })
        .collect();
    (0..5).map(|i| (0..n * n * n).map(|x| s[i][x] + 0.05 * lap[i][x] + (s[(i + 1) % 5][x] - s[i][x]) * 0.01).collect()).collect()
}

#[derive(Clone, Copy)]
enum Mode {
    Eager,
    Iterate,
    Until,
    Compiled,
}

impl Mode {
    const ALL: [(Mode, &'static str); 4] = [(Mode::Eager, "eager"), (Mode::Iterate, "iterate_all"), (Mode::Until, "until"), (Mode::Compiled, "compiled")];
}

fn until_of(steps: usize) -> Until<'static, S> {
    Until::new(steps).record("max", |s: &S| s[0].max())
}

/// One run of `steps` steps in `mode`; `cache` keeps the compiled loop between calls.
fn run_once(mode: Mode, state: &S, n: usize, steps: usize, cache: &mut Option<CompiledUntil<S>>) -> S {
    match mode {
        Mode::Eager => {
            let mut cur = state.clone();
            for _ in 0..steps {
                cur = step3d(&cur, n);
                eval_all(&cur.each_ref()).unwrap();
            }
            cur
        }
        Mode::Iterate => iterate_all(state.each_ref(), steps, |s| step3d(s, n)),
        Mode::Until => iterate_until_all(state.clone(), until_of(steps), |s, _| step3d(s, n)).unwrap().state,
        Mode::Compiled => {
            let c = cache.get_or_insert_with(|| compile_until_all(state, until_of(steps), |s, _| step3d(s, n)).unwrap());
            c.run(state.clone()).unwrap().state
        }
    }
}

fn make_state(ctx: &Context, n: usize) -> (S, Vec<Vec<f32>>) {
    let hosts: Vec<Vec<f32>> = (0..5).map(|k| values(n * n * n, 30 + k, 0.5, 1.5)).collect();
    let arrays: Vec<A> = hosts.iter().map(|v| A::from_slice_on(ctx, v, [n, n, n]).unwrap()).collect();
    let Ok(state) = <S>::try_from(arrays) else { unreachable!() };
    (state, hosts)
}

fn mib(bytes: u64) -> f64 {
    bytes as f64 / (1 << 20) as f64
}

pub fn run(h: &mut Harness) {
    let n = size(h, 64, 64, 128);
    let steps = size(h, 10, 10, 20);
    h.guarded("memory", |h| {
        let ctx = h.ctx.clone();

        // Verification at a small size: every mode against the host.
        {
            let m = 16;
            let (state, hosts) = make_state(&ctx, m);
            let mut want: Vec<Vec<f64>> = hosts.iter().map(|v| to_f64(v)).collect();
            for _ in 0..3 {
                want = step3d_host(&want, m);
            }
            for (mode, name) in Mode::ALL {
                let mut cache = None;
                let got = run_once(mode, &state, m, 3, &mut cache);
                for (g, w) in got.iter().zip(&want) {
                    check(&format!("memory/{name}"), &g.to_vec().unwrap(), &to_f32(w), 1e-3, 1e-4);
                }
            }
        }

        let cells = n * n * n;
        let state_mib = mib(5 * 4 * cells as u64);
        for (mode, name) in Mode::ALL {
            // Footprint on a context of its own, so that nothing else is in its pool.
            let tag = {
                let fresh = Context::new().unwrap();
                let (state, _) = make_state(&fresh, n);
                fresh.sync().unwrap();
                let mut cache = None;
                let out = run_once(mode, &state, n, steps, &mut cache);
                fresh.sync().unwrap();
                let st = fresh.memory_stats();
                let tag = format!("{n}^3x5 st {:.0}M held {:.0}M+{:.0}M pool", state_mib, mib(st.resident_bytes), mib(st.pooled_bytes));
                drop((out, cache, state));
                tag
            };
            let _ = h.ctx.trim();
            let (state, _) = make_state(&ctx, n);
            let mut cache = None;
            h.case(&format!("memory/step3d_{name}"), &format!("{tag} x{steps}"), Work::Cells((cells * steps) as f64), &mut || {
                let out = run_once(mode, &state, n, steps, &mut cache);
                eval_all(&out.each_ref()).unwrap();
            });
        }
    });

    // Nothing may stay resident after the loops and their results are dropped.
    h.guarded("memory_leak", |h| {
        let _ = h;
        let fresh = Context::new().unwrap();
        let m = 32;
        let base = {
            fresh.sync().unwrap();
            let _ = fresh.trim();
            fresh.memory_stats().resident_bytes
        };
        for (mode, name) in Mode::ALL {
            for round in 0..3 {
                {
                    let (state, _) = make_state(&fresh, m);
                    let mut cache = None;
                    let out = run_once(mode, &state, m, 4, &mut cache);
                    eval_all(&out.each_ref()).unwrap();
                }
                fresh.sync().unwrap();
                fresh.trim().unwrap();
                let left = fresh.memory_stats().resident_bytes.saturating_sub(base);
                assert!(left < 64 << 10, "memory_leak/{name}: {left} bytes stay resident after round {round} (all arrays and loops dropped)");
            }
        }
        println!("memory_leak: no resident bytes left after 3 rounds of each mode");
    });
}
