//! The decision log (`Context::record_decisions`): every place the planner chooses between
//! alternatives leaves a `Decision`. The tests are backend tolerant: they assert that the site is
//! present and its label non-empty, and exact labels only on the CPU backend.

use luma::*;

fn ctx() -> Context {
    Context::new().unwrap()
}

/// Runs `f` with decisions recorded and returns what was logged.
fn logged(ctx: &Context, f: impl FnOnce()) -> Vec<Decision> {
    ctx.record_decisions(true);
    f();
    ctx.sync().unwrap();
    let decisions = ctx.take_decisions();
    ctx.record_decisions(false);
    decisions
}

fn at(decisions: &[Decision], site: DecisionSite) -> Vec<&Decision> {
    decisions.iter().filter(|d| d.site == site).collect()
}

fn ramp(n: usize) -> Vec<f32> {
    (0..n).map(|i| ((i * 7919) % 1000) as f32 * 1e-3).collect()
}

#[test]
fn a_matmul_logs_its_plan() {
    let ctx = ctx();
    let n = 4096;
    let a = Array::<f32>::full_on(&ctx, [n, n], 1.0).unwrap();
    let b = Array::<f32>::full_on(&ctx, [n, n], 0.5).unwrap();
    let decisions = logged(&ctx, || {
        let c = a.matmul(&b);
        c.eval().unwrap();
    });
    let plans = at(&decisions, DecisionSite::MatmulPlan);
    assert!(!plans.is_empty() && plans.iter().all(|d| !d.chosen.is_empty()), "{decisions:?}");
    assert!(plans.iter().all(|d| d.op == "matmul" && d.location.file().ends_with("decisions.rs")), "{plans:?}");
    if ctx.backend() == BackendKind::Cpu {
        assert_eq!(plans[0].chosen, "native 128 split=1", "{plans:?}");
    }
}

#[test]
fn a_fused_transpose_logs_its_choice() {
    let ctx = ctx();
    let n = 1024;
    let a = Array::<f32>::from_slice_on(&ctx, &ramp(n * n), [n, n]).unwrap();
    let decisions = logged(&ctx, || {
        let t = a.transpose() * 2.0;
        t.eval().unwrap();
    });
    let choices = at(&decisions, DecisionSite::TransposeFusion);
    assert!(!choices.is_empty() && choices.iter().all(|d| !d.chosen.is_empty()), "{decisions:?}");
    if ctx.backend() == BackendKind::Cpu {
        assert_eq!(choices[0].chosen, "generic");
    }
}

#[test]
fn a_long_elementwise_chain_logs_coarsening_and_cuts() {
    let ctx = ctx();
    let n = 1 << 22;
    let x = Array::<f32>::from_slice_on(&ctx, &ramp(n), n).unwrap();
    let decisions = logged(&ctx, || {
        let y = fusion_limit(2, || (((&x + 1.0) * 2.0 - 3.0) / 4.0).sqrt());
        y.eval().unwrap();
    });
    let coarsen = at(&decisions, DecisionSite::Coarsen);
    assert!(!coarsen.is_empty() && coarsen.iter().all(|d| d.chosen.starts_with("coarsen ")), "{decisions:?}");
    assert!(decisions.iter().any(|d| matches!(d.site, DecisionSite::Cut(SplitReason::Nodes { .. }))), "{decisions:?}");
}

#[test]
fn a_stencil_logs_the_inliner_and_staging() {
    let ctx = ctx();
    let (w, h) = (512, 512);
    let u = Array::<f32>::from_slice_on(&ctx, &ramp(w * h), [w, h]).unwrap();
    let decisions = logged(&ctx, || {
        let p = u.pad([(1, 1), (1, 1)], PadMode::Edge);
        let s = p.slice(0, 0..w).slice(1, 1..1 + h) + p.slice(0, 2..2 + w).slice(1, 1..1 + h) + p.slice(0, 1..1 + w).slice(1, 0..h) + p.slice(0, 1..1 + w).slice(1, 2..2 + h);
        let r = &u + 0.1 * (s - 4.0 * &u);
        r.eval().unwrap();
    });
    let inline = at(&decisions, DecisionSite::Inline);
    assert!(!inline.is_empty() && inline.iter().all(|d| !d.chosen.is_empty()), "{decisions:?}");
    let staging = at(&decisions, DecisionSite::Staging);
    assert!(!staging.is_empty() && staging.iter().all(|d| !d.chosen.is_empty()), "{decisions:?}");
}

#[test]
fn a_sort_logs_its_plan() {
    let ctx = ctx();
    let n = 1 << 16;
    let x = Array::<f32>::from_slice_on(&ctx, &ramp(n), n).unwrap();
    let decisions = logged(&ctx, || {
        let s = x.sort();
        s.eval().unwrap();
    });
    let plans = at(&decisions, DecisionSite::SortPlan);
    assert!(!plans.is_empty() && plans.iter().all(|d| !d.chosen.is_empty()), "{decisions:?}");
}

#[test]
fn a_tridiagonal_solve_logs_the_line_solver() {
    let ctx = ctx();
    let (n, lines) = (256, 64);
    let ones = |v: f32| Array::<f32>::full_on(&ctx, [n, lines], v).unwrap();
    let (lo, up, rhs) = (ones(-1.0), ones(-1.0), ones(1.0));
    let diag = ones(4.0);
    let decisions = logged(&ctx, || {
        let x = solve_tridiagonal(0, &lo, &diag, &up, &rhs);
        x.eval().unwrap();
    });
    let plans = at(&decisions, DecisionSite::LineSolvePlan);
    assert!(!plans.is_empty() && plans.iter().all(|d| !d.chosen.is_empty()), "{decisions:?}");
}

#[test]
fn a_qr_panel_logs_its_choice() {
    let ctx = ctx();
    let (m, w) = (512, 16);
    let a = Array::<f32>::from_slice_on(&ctx, &ramp(m * w), [w, m]).unwrap();
    let decisions = logged(&ctx, || {
        let f = a.qr();
        let (q, r) = (f.q_thin(), f.r());
        q.eval().unwrap();
        r.eval().unwrap();
    });
    let panels = at(&decisions, DecisionSite::QrPanel);
    assert!(!panels.is_empty() && panels.iter().all(|d| !d.chosen.is_empty()), "{decisions:?}");
}

#[test]
fn profiling_returns_the_decisions_and_cuts() {
    let ctx = ctx();
    if !ctx.profiling_supported() {
        return;
    }
    let x = Array::<f32>::full_on(&ctx, 1024, 1.0).unwrap();
    let (_, profile) = ctx
        .profile(|| {
            let y = fusion_limit(2, || (((&x + 1.0) * 2.0 - 3.0) / 4.0).sqrt());
            y.to_vec().unwrap()
        })
        .unwrap();
    assert!(profile.cuts().any(|(_, _, reason)| matches!(reason, SplitReason::Nodes { .. })));
    let decisions = ctx.take_decisions();
    assert!(decisions.len() >= profile.cuts().count());
}
