//! Host cost per op of a synthetic step of 10, 100 or 1000 elementwise operations, built, evaluated
//! and dropped over and over (a `*_per_step` loop without a time-stepping scheme around it). The
//! arrays are small, so the GPU is idle most of the time and the time of a call is the host's:
//! building the lazy expression, planning (or replaying a plan), packing arguments, enqueueing.
//! Every step has fresh scalar values, as a time loop has, so plan-cache hits rebind them. A call
//! runs `1000 / k` steps of `k` operations, so every size does about 1000 operations per call and
//! the rates compare (op/s). `chain` is one dependent chain; `wide` is a sum of many short
//! independent chains of two inputs. The `_planoff` cases run with the plan cache off.
//!
//! `tools/hosttime` measures these cases in instructions; the experiments they come from are in
//! `experiments/host.md`.

use std::sync::atomic::{AtomicU32, Ordering};

use super::common::*;
use crate::harness::*;

const N: usize = 4096;
/// Operations per call.
const OPS: usize = 1000;

/// A salt that differs on every step of the process, so that scalar values never repeat.
fn next_salt() -> u32 {
    static SALT: AtomicU32 = AtomicU32::new(1);
    SALT.fetch_add(1, Ordering::Relaxed)
}

/// The scalar of operation `i` of a step with salt `salt`.
fn scalar(i: usize, salt: u32) -> f32 {
    1.0 + i as f32 * 1e-5 + salt as f32 * 1e-6
}

/// `k` dependent elementwise operations; the scalars are distinct per operation and per salt.
fn chain(x: &A, k: usize, salt: u32) -> A {
    let mut y = x.clone();
    for i in 0..k {
        let c = scalar(i, salt);
        y = match i % 4 {
            0 => &y * c,
            1 => &y + c * 1e-3,
            2 => y.abs(),
            _ => &y - c * 1e-3,
        };
    }
    y
}

fn chain_host(x: &[f32], k: usize, salt: u32) -> Vec<f32> {
    x.iter()
        .map(|&v| {
            let mut y = v;
            for i in 0..k {
                let c = scalar(i, salt);
                y = match i % 4 {
                    0 => y * c,
                    1 => y + c * 1e-3,
                    2 => y.abs(),
                    _ => y - c * 1e-3,
                };
            }
            y
        })
        .collect()
}

/// A wide DAG of about `k` operations: the sum of `k / 5` short chains of `x` and `z`.
fn wide(x: &A, z: &A, k: usize, salt: u32) -> A {
    let mut acc: Option<A> = None;
    for i in 0..(k / 5).max(1) {
        let c = scalar(i, salt);
        let base = if i % 2 == 0 { x } else { z };
        let t = &(&(base * c) + c * 1e-3).abs() * c;
        acc = Some(match acc {
            None => t,
            Some(a) => a + t,
        });
    }
    acc.unwrap()
}

fn wide_host(x: &[f32], z: &[f32], k: usize, salt: u32) -> Vec<f32> {
    (0..x.len())
        .map(|j| {
            let mut acc: Option<f32> = None;
            for i in 0..(k / 5).max(1) {
                let c = scalar(i, salt);
                let base = if i % 2 == 0 { x[j] } else { z[j] };
                let t = (base * c + c * 1e-3).abs() * c;
                acc = Some(acc.map_or(t, |a| a + t));
            }
            acc.unwrap()
        })
        .collect()
}

pub fn run(h: &mut Harness) {
    h.guarded("hoststep", |h| {
        let ctx = h.ctx.clone();
        let xh = values(N, 11, 0.5, 1.5);
        let zh = values(N, 12, 0.5, 1.5);
        let x = A::from_slice_on(&ctx, &xh, N).unwrap();
        let z = A::from_slice_on(&ctx, &zh, N).unwrap();
        let tag = format!("n={N}");
        // With `LUMA_PLAN_CACHE=off` every case is a `_planoff` case already.
        let plan_cache_on = !std::env::var("LUMA_PLAN_CACHE").is_ok_and(|v| matches!(v.trim().to_ascii_lowercase().as_str(), "off" | "0" | "false" | "no"));

        for k in [10usize, 100, 1000] {
            let steps = OPS / k;
            let got = chain(&x, k, 7).to_vec().unwrap();
            check(&format!("hoststep/chain{k}"), &got, &chain_host(&xh, k, 7), 1e-3, 1e-4);
            let got = wide(&x, &z, k, 7).to_vec().unwrap();
            check(&format!("hoststep/wide{k}"), &got, &wide_host(&xh, &zh, k, 7), 1e-3, 1e-4);

            // Every step is evaluated (and the harness syncs after the call); the last one is read.
            let run_chain = || {
                for _ in 0..steps - 1 {
                    chain(&x, k, next_salt()).eval().unwrap();
                }
                force(&chain(&x, k, next_salt()));
            };
            let run_wide = || {
                for _ in 0..steps - 1 {
                    wide(&x, &z, k, next_salt()).eval().unwrap();
                }
                force(&wide(&x, &z, k, next_salt()));
            };
            h.case(&format!("hoststep/chain{k}"), &tag, per_sec(steps * k, "op/s"), &mut || run_chain());
            h.case(&format!("hoststep/wide{k}"), &tag, per_sec(steps * k, "op/s"), &mut || run_wide());
            if plan_cache_on {
                ctx.set_plan_cache(false);
                h.case(&format!("hoststep/chain{k}_planoff"), &tag, per_sec(steps * k, "op/s"), &mut || run_chain());
                ctx.set_plan_cache(true);
            }
        }
    });
}
