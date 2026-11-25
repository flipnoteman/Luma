#![allow(dead_code)] // W3-U5 turns a `FactorGraph` into a replayable plan; wave 2 defines the type, the key and the capture and measures
//! A factorisation as a launch list: what a plan cache would key and replay, and what the launches-per-factorisation measurement reads.
//!
//! `capture` records the launches of one call (`lu`, `cholesky`, `solve`, ...) with the context's recording, the way a traced step is
//! recorded (`jit::replay`), and keeps the list with the numbers that explain the call's host time. **Nothing is replayed here** and
//! the context keeps no cache: replay is W3-U5, which is the first consumer. The key says what a replay would have to match: the
//! kind, the shape, the batch, the strategy and its parameters, the device.
//!
//! A recording holds plain launches only. A `clear` or a `copy` of a buffer, a readback, an upload or a sync aborts it
//! ([`Context::abort_recording_for`]), so a factorisation that issues one is not capturable: `capture` returns the reason, which is
//! what U5 needs to know about it. `svd` and `eigh` of a matrix that does not fit workgroup memory read back (the sweeps are recorded in
//! chunks and the convergence flag is read), and the refined solves read back by design: `capture` refuses all of these before it starts.

use std::hash::{Hash, Hasher};
use std::sync::Arc;

use std::panic::Location;

use super::svd::{self, Kind};
use super::{panel_width, tri, Dims};
use crate::array::Array;
use crate::context::{CmdArg, Recording};
use crate::context::Context;
use crate::error::{LumaError, Result};
use crate::gpu::hash::FxHasher;
use crate::jit::structured::boundary::FactorKind;
use crate::jit::structured::families::factor::{boundary_flags, chol_stage, eigh_stage, lu_stage, qr_stage, refined_stage, svd_stage, tri_stage};
use crate::jit::structured::strategy::{first_candidate_in, StrategyId};

/// What a replay of a factorisation must match.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) struct FactorKey {
    pub kind: FactorKind,
    pub rows: u32,
    pub cols: u32,
    pub batch: u32,
    /// Right-hand-side columns of a solve.
    pub rhs: u32,
    /// The strategy that ran the top of the call (`lu_recursive`, `chol_blocked`, ...) and the sizes of its parameters.
    pub strategy: StrategyId,
    pub nb: u16,
    pub panel: u16,
    pub tf32: bool,
    /// A hash of the device and what it can do (`Caps`, name, backend, compute units): the `Tuning` and the capabilities enter the
    /// key through it.
    pub device: u64,
}

impl FactorKey {
    /// The key of a `kind` call on the `d` matrices (`rhs` right-hand-side columns for the solves) on `ctx`: the strategy is the one the
    /// top of the call selects (its stage's first candidate), `nb` the block width of the blocked Cholesky and triangular solves
    /// (`LUMA_CHOL_NB`) and `panel` the width of the LU and QR panels (`LUMA_LINALG_PANEL`), 0 where the kind has none.
    pub(crate) fn for_call(ctx: &Context, kind: FactorKind, d: Dims, rhs: usize) -> FactorKey {
        let stage = match kind {
            FactorKind::Lu | FactorKind::Solve | FactorKind::Inverse => lu_stage(),
            FactorKind::Cholesky | FactorKind::SolveSpd => chol_stage(),
            FactorKind::Qr | FactorKind::Lstsq => qr_stage(),
            FactorKind::Svd => svd_stage(),
            FactorKind::Eigh => eigh_stage(),
            FactorKind::SolveTriangular => tri_stage(),
            FactorKind::SolveRefined | FactorKind::SolveSpdRefined => refined_stage(),
        };
        let stage_kind = match kind {
            FactorKind::Solve | FactorKind::Inverse => FactorKind::Lu,
            FactorKind::SolveSpd => FactorKind::Cholesky,
            FactorKind::Lstsq => FactorKind::Qr,
            other => other,
        };
        let boundary = boundary_flags(stage_kind, "factor_graph", Location::caller(), d, rhs, false, true, true);
        let strategy = first_candidate_in(stage, ctx, &boundary).map_or(StrategyId::LU_RECURSIVE, |(id, _)| id);
        let blocked = matches!(strategy, StrategyId::CHOL_BLOCKED | StrategyId::TRI_BLOCKED) || matches!(kind, FactorKind::Solve | FactorKind::SolveSpd | FactorKind::Inverse | FactorKind::SolveTriangular);
        let dim = |v: usize| u32::try_from(v).unwrap_or(u32::MAX);
        FactorKey {
            kind,
            rows: dim(d.rows),
            cols: dim(d.cols),
            batch: dim(d.batch),
            rhs: dim(rhs),
            strategy,
            nb: if blocked { tri::block_size(ctx) as u16 } else { 0 },
            panel: if matches!(stage_kind, FactorKind::Lu | FactorKind::Qr) { panel_width() as u16 } else { 0 },
            tf32: ctx.numerics().tf32_linalg(),
            device: device_hash(ctx),
        }
    }
}

/// A hash of the device and its capabilities.
fn device_hash(ctx: &Context) -> u64 {
    let mut hasher = FxHasher::default();
    ctx.caps().bits().hash(&mut hasher);
    ctx.device_name().hash(&mut hasher);
    format!("{:?}", ctx.backend()).hash(&mut hasher);
    ctx.info().compute_units.hash(&mut hasher);
    env!("CARGO_PKG_VERSION").hash(&mut hasher);
    hasher.finish()
}

/// The recorded launch list of one factorisation call.
pub(crate) struct FactorGraph {
    pub key: FactorKey,
    pub recording: Recording,
    /// Kernel launches in the list.
    pub launches: u32,
    /// Launches per kernel label, in first-use order.
    pub by_kernel: Vec<(Arc<str>, u32)>,
    /// The sum of the bound storage bytes of every launch: what a replay moves.
    pub arg_bytes: u64,
}

impl FactorGraph {
    /// Records `run()` (a factorisation call on `inputs`) and forces its outputs. `Err` if the kind reads back, if another recording
    /// is open on this thread, or if the call does something a recording cannot hold (the error names it).
    pub(crate) fn capture(ctx: &Context, key: FactorKey, inputs: &[&Array<f32>], run: impl FnOnce() -> Vec<Array<f32>>) -> Result<FactorGraph> {
        refuse_readbacks(ctx, &key)?;
        if ctx.recording_open() {
            return Err(LumaError::Unsupported("a factorisation cannot be captured inside another recording".into()));
        }
        // The operands exist before the recording starts: computing them is not part of the factorisation.
        for input in inputs {
            input.eval()?;
        }
        ctx.sync()?;
        if !ctx.begin_recording(Vec::new()) {
            return Err(LumaError::Unsupported("a factorisation cannot be captured inside another recording".into()));
        }
        let outputs = run();
        let forced = outputs.iter().try_for_each(Array::request);
        ctx.jit().evaluate_requested();
        let reason = ctx.abort_reason();
        let recording = ctx.end_recording();
        forced?;
        match recording {
            Some(recording) => Ok(FactorGraph::of(key, recording)),
            None => Err(LumaError::Unsupported(format!(
                "a factorisation that does a {} cannot be recorded",
                reason.unwrap_or("non-launch operation")
            ))),
        }
    }

    fn of(key: FactorKey, recording: Recording) -> FactorGraph {
        let mut by_kernel: Vec<(Arc<str>, u32)> = Vec::new();
        let mut arg_bytes = 0u64;
        for cmd in &recording.cmds {
            match by_kernel.iter_mut().find(|(label, _)| *label == cmd.kernel.label) {
                Some((_, count)) => *count += 1,
                None => by_kernel.push((Arc::clone(&cmd.kernel.label), 1)),
            }
            arg_bytes += cmd.args.iter().map(|arg| if let CmdArg::Storage { bytes, .. } = arg { *bytes } else { 0 }).sum::<u64>();
        }
        FactorGraph { key, launches: recording.cmds.len() as u32, by_kernel, arg_bytes, recording }
    }
}

/// Refuses what reads back before a recording starts, not mid-way: the SVD and eigh of a matrix the workgroup-memory kernel does not
/// take (their sweeps read the convergence flag), and the refined solves.
fn refuse_readbacks(ctx: &Context, key: &FactorKey) -> Result<()> {
    let refused = |what: &str| Err(LumaError::Unsupported(format!("{what} reads back and cannot be captured")));
    match key.kind {
        FactorKind::SolveRefined | FactorKind::SolveSpdRefined => refused("an adaptive refined solve"),
        FactorKind::Svd | FactorKind::Eigh => {
            let (rr, cc) = (key.rows.max(key.cols) as usize, key.rows.min(key.cols) as usize);
            let kind = if key.kind == FactorKind::Svd { Kind::Svd } else { Kind::Eigh };
            if svd::plan(ctx, kind, true, rr, cc).is_none() {
                return refused("a decomposition of a matrix that does not fit workgroup memory");
            }
            Ok(())
        }
        _ => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::ContextOptions;
    use crate::BackendKind;

    fn cpu() -> Context {
        Context::with_options(ContextOptions::default().backend(BackendKind::Cpu).calibrate(false)).unwrap()
    }

    /// A diagonally dominant symmetric matrix: SPD, and it needs no pivoting.
    fn spd(ctx: &Context, n: usize) -> Array<f32> {
        Array::from_slice_on(ctx, &(0..n * n).map(|i| if i % (n + 1) == 0 { n as f32 } else { ((i * 31) % 7) as f32 * 0.01 }).collect::<Vec<_>>(), [n, n]).unwrap()
    }

    fn square(n: usize) -> Dims {
        Dims { rows: n, cols: n, batch: 1 }
    }

    /// What a call computes.
    type Call<'a> = Box<dyn Fn() -> Vec<Array<f32>> + 'a>;

    /// The calls of a factorisation test: name, kind, and what it computes.
    fn calls<'a>(a: &'a Array<f32>, b: &'a Array<f32>) -> Vec<(&'static str, FactorKind, Call<'a>)> {
        vec![
            ("lu", FactorKind::Lu, Box::new(|| vec![a.lu().lu])),
            ("cholesky", FactorKind::Cholesky, Box::new(|| vec![a.cholesky()])),
            ("qr", FactorKind::Qr, Box::new(|| {
                let f = a.qr();
                vec![f.qr, f.t]
            })),
            ("solve", FactorKind::Solve, Box::new(|| vec![a.solve(b)])),
            ("solve_spd", FactorKind::SolveSpd, Box::new(|| vec![a.solve_spd(b)])),
            ("inverse", FactorKind::Inverse, Box::new(|| vec![a.inverse()])),
        ]
    }

    /// What the profiler counts for the same call is what the graph holds. (The CPU backend has no launch timer, so there the graph's
    /// own count is checked to be non-zero and the profile comparison is the GPU gate's.) A factorisation that clears a buffer (the
    /// blocked Cholesky starts from a zeroed output) is not capturable, and says why.
    #[test]
    fn factor_graph_matches_profile_launches() {
        let ctx = Context::new().unwrap();
        let sizes: &[usize] = if ctx.info().cpu { &[16, 100] } else { &[16, 100, 700] };
        for &n in sizes {
            let a = spd(&ctx, n);
            let b = Array::<f32>::from_slice_on(&ctx, &vec![1.0; n], [1, n]).unwrap();
            a.eval().unwrap();
            b.eval().unwrap();
            for (name, kind, call) in calls(&a, &b) {
                let key = FactorKey::for_call(&ctx, kind, square(n), 1);
                let captured = FactorGraph::capture(&ctx, key, &[&a, &b], &call);
                // The blocked Cholesky starts from a zeroed output, and the look-back triangular solve of a GPU zeroes its ticket buffer.
                let must_clear = matches!(kind, FactorKind::Cholesky | FactorKind::SolveSpd) && n > super::super::chol::leaf(&ctx);
                match captured {
                    Ok(graph) => {
                        assert!(!must_clear, "{name} n={n}: a blocked Cholesky clears its output, which a recording cannot hold");
                        assert!(graph.launches > 0 && graph.launches as usize == graph.recording.cmds.len(), "{name} n={n}");
                        assert_eq!(graph.by_kernel.iter().map(|(_, count)| count).sum::<u32>(), graph.launches);
                        assert!(graph.arg_bytes > 0);
                        if ctx.profiling_supported() {
                            let (_, profile) = ctx.profile(|| call().iter().for_each(|x| x.eval().unwrap())).unwrap();
                            assert_eq!(profile.launches(), graph.launches, "{name} n={n}");
                        }
                        // The recording left nothing behind: the next one starts.
                        assert!(ctx.begin_recording(Vec::new()) && ctx.end_recording().is_some());
                    }
                    Err(error) => {
                        assert!(error.to_string().contains("clear"), "{name} n={n}: {error}");
                        assert!(ctx.begin_recording(Vec::new()) && ctx.end_recording().is_some(), "an aborted capture ends cleanly");
                    }
                }
            }
        }
    }

    #[test]
    fn factor_graph_is_stable() {
        let ctx = cpu();
        let a = spd(&ctx, 100);
        let b = Array::<f32>::from_slice_on(&ctx, &[1.0; 100], [1, 100]).unwrap();
        for (name, kind, call) in calls(&a, &b).into_iter().filter(|(name, ..)| matches!(*name, "lu" | "qr" | "solve" | "inverse")) {
            let key = FactorKey::for_call(&ctx, kind, square(100), 1);
            let first = FactorGraph::capture(&ctx, key, &[&a, &b], &call).unwrap();
            let second = FactorGraph::capture(&ctx, key, &[&a, &b], &call).unwrap();
            assert_eq!(first.by_kernel, second.by_kernel, "{name}");
            assert_eq!((first.launches, first.arg_bytes), (second.launches, second.arg_bytes), "{name}");
        }
    }

    #[test]
    fn factor_graph_refuses_readbacks() {
        let ctx = cpu();
        // 200 x 200 does not fit the workgroup-memory SVD: its sweeps read the convergence flag back.
        let a = Array::<f32>::from_slice_on(&ctx, &(0..200 * 200).map(|i| ((i * 37) % 11) as f32 * 0.1 + if i % 201 == 0 { 5.0 } else { 0.0 }).collect::<Vec<_>>(), [200, 200]).unwrap();
        a.eval().unwrap();
        for kind in [FactorKind::Svd, FactorKind::Eigh, FactorKind::SolveRefined, FactorKind::SolveSpdRefined] {
            let key = FactorKey::for_call(&ctx, kind, square(200), 1);
            let refused = FactorGraph::capture(&ctx, key, &[&a], || panic!("a refused capture must not run the call")).map(drop).unwrap_err();
            assert!(matches!(refused, LumaError::Unsupported(ref what) if what.contains("reads back")), "{kind:?}: {refused}");
            assert!(ctx.begin_recording(Vec::new()) && ctx.end_recording().is_some(), "{kind:?} left a recording open");
        }
        // A small decomposition (one workgroup, no readback) is captured.
        let small = Array::<f32>::from_slice_on(&ctx, &(0..16 * 16).map(|i| ((i * 37) % 11) as f32 + if i % 17 == 0 { 5.0 } else { 0.0 }).collect::<Vec<_>>(), [16, 16]).unwrap();
        small.eval().unwrap();
        let key = FactorKey::for_call(&ctx, FactorKind::Svd, square(16), 0);
        let graph = FactorGraph::capture(&ctx, key, &[&small], || vec![small.svd().s]).unwrap();
        assert_eq!(graph.launches, 1);
        // Inside another recording a capture does not start.
        assert!(ctx.begin_recording(Vec::new()));
        let key = FactorKey::for_call(&ctx, FactorKind::Lu, square(16), 0);
        assert!(FactorGraph::capture(&ctx, key, &[&small], || vec![small.lu().lu]).is_err());
        assert!(ctx.end_recording().is_some());
    }

    /// The key says which strategy and which sizes ran: another block width (`LUMA_CHOL_NB`) is another key. The environment is set before
    /// a child process starts (see `the_environment_gives_other_keys`).
    #[test]
    #[ignore = "run by the_key_follows_the_environment in a child process with its own environment"]
    fn key_child() {
        let ctx = cpu();
        let key = FactorKey::for_call(&ctx, FactorKind::Cholesky, square(300), 0);
        println!("child-key={key:?}");
    }

    #[test]
    fn the_key_follows_the_environment() {
        let run = |nb: &str| {
            let output = std::process::Command::new(std::env::current_exe().unwrap())
                .args(["dense::graph::tests::key_child", "--exact", "--ignored", "--nocapture", "--test-threads=1"])
                .env("LUMA_CHOL_NB", nb)
                .env("LUMA_DEVICE", "cpu")
                .env("LUMA_CALIBRATE", "0")
                .output()
                .unwrap();
            assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
            String::from_utf8_lossy(&output.stdout).lines().find_map(|line| line.split_once("child-key=").map(|(_, key)| key.to_owned())).expect("the child prints its key")
        };
        let (narrow, wide) = (run("32"), run("64"));
        assert!(narrow.contains("nb: 32") && wide.contains("nb: 64"), "{narrow} / {wide}");
        assert!(narrow.contains("strategy: StrategyId(Factor, 6)") && narrow != wide, "{narrow} / {wide}");
        // The kind and the shape are part of it as well.
        let ctx = cpu();
        let key = |kind, n| FactorKey::for_call(&ctx, kind, square(n), 1);
        assert_ne!(key(FactorKind::Lu, 100), key(FactorKind::Solve, 100));
        assert_ne!(key(FactorKind::Lu, 100), key(FactorKind::Lu, 101));
        assert_eq!(key(FactorKind::Lu, 100), key(FactorKind::Lu, 100));
    }
}
