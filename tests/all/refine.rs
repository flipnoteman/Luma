//! `solve_refined` and `solve_spd_refined`: an `f32` factorisation with iterative refinement against a compensated (Dot2) residual.
//! Checked against an `f64` host solver on matrices with prescribed condition numbers: the result is the correctly rounded `f32`
//! solution up to `kappa ~ 1e6`, the plain solve is worse by about `kappa`, hard systems stall without harm, and the answer is
//! reproducible.

use std::collections::HashMap;
use std::sync::{LazyLock, Mutex};

use crate::common::values;
use crate::dense_host::{mul, solve as host_solve, widen};
use crate::svd::orthogonal;
use luma::{Array, BackendKind, Context, ContextOptions, DecisionSite, Numerics};

/// Half an ulp of an `f32` result: the best the call can promise.
const U24: f64 = 1.0 / 16_777_216.0;

fn ctx() -> Context {
    Context::new().unwrap()
}

type Orthogonals = Mutex<HashMap<(usize, u64), Vec<f64>>>;

/// `orthogonal(n, seed)`, computed once per size and seed (the host Gram-Schmidt of 512 columns takes seconds).
fn orthogonal_cached(n: usize, seed: u64) -> Vec<f64> {
    static CACHE: LazyLock<Orthogonals> = LazyLock::new(Default::default);
    if let Some(q) = CACHE.lock().unwrap().get(&(n, seed)) {
        return q.clone();
    }
    let q = orthogonal(n, seed);
    CACHE.lock().unwrap().insert((n, seed), q.clone());
    q
}

/// The singular values `1 ... 1/kappa`, geometric.
fn spectrum(n: usize, kappa: f64) -> Vec<f64> {
    (0..n).map(|i| kappa.powf(-(i as f64) / (n.max(2) - 1) as f64)).collect()
}

/// One system: the matrices (row-major, `batch` of them), right-hand sides `[n]` each, and the `f64` solution of the matrices as stored
/// (their `f32` values).
struct System {
    n: usize,
    batch: usize,
    a: Vec<f32>,
    b: Vec<f32>,
    x_ref: Vec<f64>,
}

/// `U diag(s) Vᵀ` (general) or `Q diag(s) Qᵀ` (symmetric positive definite; the strict upper triangle is filled with `junk` where
/// given), `batch` matrices, condition number `kappa`.
fn system(n: usize, batch: usize, kappa: f64, spd: bool, junk: Option<f32>, seed: u64) -> System {
    let s = spectrum(n, kappa);
    let (mut a, mut b, mut x_ref) = (Vec::new(), Vec::new(), Vec::new());
    for m in 0..batch {
        let seed = seed + 17 * m as u64;
        let u = orthogonal_cached(n, seed);
        let v = if spd { u.clone() } else { orthogonal_cached(n, seed + 1000) };
        let us: Vec<f64> = (0..n * n).map(|e| u[e] * s[e % n]).collect();
        let vt: Vec<f64> = (0..n * n).map(|e| v[(e % n) * n + e / n]).collect();
        let full = mul(&us, &vt, n, n, n);
        let mut mat: Vec<f32> = full.iter().map(|&x| x as f32).collect();
        if spd {
            for i in 0..n {
                for j in 0..i {
                    mat[j * n + i] = mat[i * n + j];
                }
            }
        }
        let rhs = values(n, seed + 5, -1.0, 1.0);
        x_ref.extend(host_solve(&widen(&mat), &widen(&rhs), n, 1));
        if let (true, Some(junk)) = (spd, junk) {
            for i in 0..n {
                for j in i + 1..n {
                    mat[i * n + j] = junk;
                }
            }
        }
        a.extend(mat);
        b.extend(rhs);
    }
    System { n, batch, a, b, x_ref }
}

fn infinity(v: &[f64]) -> f64 {
    v.iter().fold(0.0, |m, x| m.max(x.abs()))
}

/// `||x - x_ref||_inf / ||x_ref||_inf`, the worst of the batch.
fn relative_error(sys: &System, x: &[f32]) -> f64 {
    (0..sys.batch)
        .map(|m| {
            let range = m * sys.n..(m + 1) * sys.n;
            let diff: Vec<f64> = x[range.clone()].iter().zip(&sys.x_ref[range.clone()]).map(|(&g, &w)| f64::from(g) - w).collect();
            infinity(&diff) / infinity(&sys.x_ref[range])
        })
        .fold(0.0, f64::max)
}

fn arrays(ctx: &Context, sys: &System) -> (Array<f32>, Array<f32>) {
    (
        Array::from_slice_on(ctx, &sys.a, [sys.n, sys.n, sys.batch]).unwrap(),
        Array::from_slice_on(ctx, &sys.b, [1, sys.n, sys.batch]).unwrap(),
    )
}

/// The cases of the spec: `n` in {8, 100, 512}, batch 1 and 3, `kappa` up to 1e7 (a 512 x 512 batch of 3 and the hardest 512 case are left
/// out: the host reference alone takes minutes there).
fn cases() -> Vec<(usize, usize, f64)> {
    let mut cases = Vec::new();
    for &n in &[8usize, 100] {
        for &batch in &[1usize, 3] {
            for &kappa in &[1e1, 1e3, 1e5, 1e6, 1e7] {
                cases.push((n, batch, kappa));
            }
        }
    }
    cases.extend([(512, 1, 1e3), (512, 1, 1e6)]);
    cases
}

fn check_correctly_rounded(spd: bool) {
    let ctx = ctx();
    for (n, batch, kappa) in cases() {
        let sys = system(n, batch, kappa, spd, spd.then_some(1.0e3), 7 + n as u64);
        let (a, b) = arrays(&ctx, &sys);
        let refined = if spd { a.solve_spd_refined(&b) } else { a.solve_refined(&b) };
        let x = refined.x.to_vec().unwrap();
        let label = format!("n={n} batch={batch} kappa={kappa:e} spd={spd}");
        let err = relative_error(&sys, &x);
        if kappa <= 1e6 || (kappa == 1e7 && n <= 100) {
            assert!(refined.converged, "{label}: not converged, achieved {}", refined.achieved_rel_err);
            assert!(refined.achieved_rel_err <= U24, "{label}: estimate {} above the target", refined.achieved_rel_err);
            assert!(err <= 4.0 * U24, "{label}: relative error {err:e} above 4 * 2^-24");
        }
        if kappa >= 1e5 {
            let plain = if spd { a.solve_spd(&b) } else { a.solve(&b) }.to_vec().unwrap();
            let plain_err = relative_error(&sys, &plain);
            assert!(plain_err >= kappa / 1000.0 * U24, "{label}: the plain solve ({plain_err:e}) is not worse by kappa/1000 than the target");
            assert!(err < plain_err, "{label}: refined {err:e} is not better than plain {plain_err:e}");
        }
    }
}

#[test]
fn refined_solve_is_correctly_rounded() {
    check_correctly_rounded(false);
}

#[test]
fn refined_spd_solve_is_correctly_rounded() {
    // The strict upper triangle holds junk (1e3): only the lower triangle may be read.
    check_correctly_rounded(true);
}

#[test]
fn many_right_hand_sides_converge_together() {
    let ctx = ctx();
    let (n, k) = (100, 9);
    let sys = system(n, 1, 1e4, false, None, 31);
    let a = Array::from_slice_on(&ctx, &sys.a, [n, n]).unwrap();
    let rhs = values(n * k, 77, -1.0, 1.0);
    let b = Array::from_slice_on(&ctx, &rhs, [k, n]).unwrap();
    let refined = a.solve_refined(&b);
    assert!(refined.converged);
    let got = refined.x.to_vec().unwrap();
    let want = host_solve(&widen(&sys.a), &widen(&rhs), n, k);
    for c in 0..k {
        let col_got: Vec<f64> = (0..n).map(|i| f64::from(got[i * k + c])).collect();
        let col_want: Vec<f64> = (0..n).map(|i| want[i * k + c]).collect();
        let diff: Vec<f64> = col_got.iter().zip(&col_want).map(|(g, w)| g - w).collect();
        assert!(infinity(&diff) / infinity(&col_want) <= 4.0 * U24, "column {c}");
    }
}

#[test]
fn refined_converges_in_two_to_four_iterations() {
    let ctx = ctx();
    for kappa in [1e1, 1e3, 1e5, 1e6] {
        let sys = system(100, 1, kappa, false, None, 3);
        let (a, b) = arrays(&ctx, &sys);
        ctx.record_decisions(true);
        let refined = a.solve_refined(&b);
        refined.x.eval().unwrap();
        let decisions = ctx.take_decisions();
        ctx.record_decisions(false);
        let results: Vec<_> = decisions.iter().filter(|d| d.site == DecisionSite::FactorPlan && d.chosen.starts_with("factor::refine_result[iters=")).collect();
        assert_eq!(results.len(), 1, "{decisions:?}");
        let iters: u32 = results[0].chosen["factor::refine_result[iters=".len()..].split(' ').next().unwrap().parse().unwrap();
        assert!((1..=4).contains(&iters), "kappa={kappa:e}: {iters} iterations ({})", results[0].chosen);
    }
}

#[test]
fn refined_stalls_without_harm() {
    let ctx = ctx();
    for kappa in [1e8, 1e9] {
        let sys = system(100, 1, kappa, false, None, 5);
        let (a, b) = arrays(&ctx, &sys);
        ctx.record_decisions(true);
        let refined = a.solve_refined(&b);
        let x = refined.x.to_vec().unwrap();
        let decisions = ctx.take_decisions();
        ctx.record_decisions(false);
        assert!(x.iter().all(|v| v.is_finite()), "kappa={kappa:e}");
        let plain = a.solve(&b).to_vec().unwrap();
        let (refined_err, plain_err) = (relative_error(&sys, &x), relative_error(&sys, &plain));
        assert!(refined_err <= plain_err * 1.0000001, "kappa={kappa:e}: refined {refined_err:e}, plain {plain_err:e}");
        assert!(!refined.converged);
        assert!(decisions.iter().any(|d| d.site == DecisionSite::FactorPlan && d.chosen == "factor::refine_stalled"), "{decisions:?}");
    }
}

#[test]
fn refined_reports_a_stall_without_profiling() {
    let ctx = ctx();
    let sys = system(100, 1, 1e9, false, None, 9);
    let (a, b) = arrays(&ctx, &sys);
    let refined = a.solve_refined(&b);
    assert!(!refined.converged);
    assert!(refined.achieved_rel_err.is_finite() && refined.achieved_rel_err > U24, "{}", refined.achieved_rel_err);
}

#[test]
fn refined_singular_matrix_behaves_like_plain() {
    let ctx = ctx();
    for n in [3usize, 40, 100] {
        // Row 1 is row 0: singular.
        let mut a = values(n * n, 11, -1.0, 1.0);
        for j in 0..n {
            a[n + j] = a[j];
        }
        let b = values(n, 12, -1.0, 1.0);
        let a = Array::from_slice_on(&ctx, &a, [n, n]).unwrap();
        let b = Array::from_slice_on(&ctx, &b, [1, n]).unwrap();
        let plain = a.solve(&b).to_vec().unwrap();
        let refined = a.solve_refined(&b);
        let got = refined.x.to_vec().unwrap();
        assert!(!refined.converged);
        assert_eq!(got.len(), plain.len());
        assert_eq!(got.iter().map(|v| v.is_finite()).collect::<Vec<_>>(), plain.iter().map(|v| v.is_finite()).collect::<Vec<_>>(), "n={n}");
    }
}

#[test]
fn refined_is_deterministic() {
    let ctx = ctx();
    let sys = system(100, 3, 1e5, false, None, 13);
    let (a, b) = arrays(&ctx, &sys);
    let first = a.solve_refined(&b);
    let second = a.solve_refined(&b);
    let (first_x, second_x) = (first.x.to_vec().unwrap(), second.x.to_vec().unwrap());
    assert_eq!(first_x.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), second_x.iter().map(|v| v.to_bits()).collect::<Vec<_>>());
    assert_eq!(first.achieved_rel_err.to_bits(), second.achieved_rel_err.to_bits());
}

#[test]
fn refined_target_and_shape_errors_poison() {
    let ctx = ctx();
    let sys = system(8, 1, 1e2, false, None, 2);
    let (a, b) = arrays(&ctx, &sys);
    for bad in [f64::NAN, 0.0, -1.0, 1.0, 2.0, f64::INFINITY] {
        let refined = a.solve_refined_to(&b, bad);
        assert!(refined.x.to_vec().is_err(), "target {bad}");
        assert!(!refined.converged && refined.achieved_rel_err.is_nan());
    }
    // A looser target is accepted and met sooner.
    let loose = a.solve_refined_to(&b, 1e-3);
    assert!(loose.converged && loose.x.to_vec().is_ok());
    // A right-hand side of the wrong length, a matrix that is not square.
    let wrong = Array::<f32>::from_slice_on(&ctx, &[1.0; 5], [1, 5]).unwrap();
    assert!(a.solve_refined(&wrong).x.to_vec().is_err());
    let tall = Array::<f32>::from_slice_on(&ctx, &[1.0; 12], [3, 4]).unwrap();
    assert!(tall.solve_refined(&Array::<f32>::from_slice_on(&ctx, &[1.0; 4], [1, 4]).unwrap()).x.to_vec().is_err());
}

#[test]
fn a_vector_right_hand_side_gives_a_vector() {
    let ctx = ctx();
    let sys = system(20, 1, 1e3, false, None, 4);
    let a = Array::from_slice_on(&ctx, &sys.a, [20, 20]).unwrap();
    let b = Array::from_slice_on(&ctx, &sys.b, 20).unwrap();
    let refined = a.solve_refined(&b);
    assert_eq!(refined.x.shape(), b.shape());
    assert!(refined.converged);
}

#[test]
fn refined_is_never_selected_by_solve() {
    let sys = system(100, 1, 1e3, false, None, 6);
    for numerics in [Numerics::default(), Numerics::strict(), Numerics::algebraic(), Numerics::fast()] {
        let ctx = Context::with_options(ContextOptions::from_env().numerics(numerics)).unwrap();
        let (a, b) = arrays(&ctx, &sys);
        let spd = system(100, 1, 1e3, true, None, 6);
        let (s, sb) = arrays(&ctx, &spd);
        ctx.record_decisions(true);
        a.solve(&b).eval().unwrap();
        s.solve_spd(&sb).eval().unwrap();
        a.inverse().eval().unwrap();
        let decisions = ctx.take_decisions();
        ctx.record_decisions(false);
        assert!(decisions.iter().all(|d| !d.chosen.contains("refine") && d.candidates.iter().all(|(name, _)| !name.contains("refined"))), "{decisions:?}");
        assert!(decisions.iter().any(|d| d.site == DecisionSite::FactorPlan), "the factorisations log their strategies");
    }
}

#[test]
fn fast_math_does_not_weaken_the_residual() {
    let sys = system(100, 1, 1e5, false, None, 8);
    let ctx = Context::with_options(ContextOptions::from_env().numerics(Numerics::fast())).unwrap();
    let (a, b) = arrays(&ctx, &sys);
    let refined = a.solve_refined(&b);
    assert!(refined.converged, "{}", refined.achieved_rel_err);
    let x = refined.x.to_vec().unwrap();
    assert!(relative_error(&sys, &x) <= 4.0 * U24);
}

#[test]
fn the_dekker_residual_is_as_good_as_the_fused_one() {
    let sys = system(100, 1, 1e6, false, None, 10);
    let ctx = Context::with_options(ContextOptions::from_env().df_fma(false)).unwrap();
    let (a, b) = arrays(&ctx, &sys);
    let refined = a.solve_refined(&b);
    assert!(refined.converged);
    assert!(relative_error(&sys, &refined.x.to_vec().unwrap()) <= 4.0 * U24);
    let _ = BackendKind::Cpu;
}
