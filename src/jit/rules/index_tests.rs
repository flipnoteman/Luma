//! The rules of W3-R4R5 on products: what the application of each does to the program (the decision log) and to the bits. The
//! harness (`harness.rs`) holds their classes; these tests hold the places they fire.

use super::index::{diag_of_product_form, needs_t1};
use super::t1::t1;
use crate::array::Array;
use crate::context::{Context, ContextOptions};
use crate::jit::comprehension::{Algebra, Origin, lower};
use crate::jit::structured::contraction::{ContractionAttrs, ContractionView, Tf32Request};
use crate::jit::structured::sites::LazyMode;
use crate::numerics::Numerics;
use crate::profile::{Decision, DecisionSite};

type A = Array<f32>;

fn open(mode: LazyMode, numerics: Numerics) -> Context {
    Context::with_tuning(ContextOptions::from_env().calibrate(false).numerics(numerics), |tuning| tuning.lazy = mode).unwrap()
}

/// Values with signed zeros and small integers, so products and sums are exact in every precision.
fn values(len: usize, seed: usize) -> Vec<f32> {
    (0..len)
        .map(|i| match (i + seed) % 7 {
            0 => 0.0,
            1 => -0.0,
            r => (r as f32) - 3.5 + (i % 3) as f32,
        })
        .collect()
}

fn bits(array: &A) -> Vec<u32> {
    array.to_vec().unwrap().into_iter().map(f32::to_bits).collect()
}

fn rewrites(ctx: &Context) -> Vec<String> {
    ctx.take_decisions().into_iter().filter(|decision: &Decision| decision.site == DecisionSite::Rewrite).map(|decision| decision.chosen.to_string()).collect()
}

fn plans(ctx: &Context) -> usize {
    ctx.take_decisions().into_iter().filter(|decision| decision.site == DecisionSite::MatmulPlan).count()
}

fn view<'a>(lhs: &'a A, rhs: &'a A, tf32: Tf32Request) -> ContractionView<'a> {
    ContractionView { lhs, rhs, attrs: ContractionAttrs { tf32, half: false }, location: std::panic::Location::caller(), epilogue: None, partners: &[] }
}

#[test]
fn a_product_of_one_term_is_an_elementwise_product_plus_zero() {
    if Algebra::current() < Algebra::On {
        return; // LUMA_ALGEBRA below `on`: no rule of the engine runs
    }
    let ctx = open(LazyMode::Never, Numerics::default());
    let (ha, hb) = (values(6, 0), values(4, 3));
    let a = A::from_slice_on(&ctx, &ha, [1, 6]).unwrap();
    let b = A::from_slice_on(&ctx, &hb, [4, 1]).unwrap();
    ctx.record_decisions(true);
    let c = a.matmul(&b);
    let want: Vec<u32> = (0..6).flat_map(|m| (0..4).map(move |n| (m, n))).map(|(m, n)| (ha[m] * hb[n] + 0.0).to_bits()).collect();
    assert_eq!(bits(&c), want, "-0 products are +0, as the kernels' sums made them");
    assert_eq!(c.shape().dims(), [4, 6, 1, 1]);
    assert!(ctx.take_decisions().iter().any(|d| d.site == DecisionSite::Rewrite && d.chosen == "contract_k1"));
    // The result is what the unrewritten product gives, bit for bit.
    let plain = open(LazyMode::Never, Numerics::default());
    plain.jit().set_simplify(true);
    let (pa, pb) = (A::from_slice_on(&plain, &ha, [1, 6]).unwrap(), A::from_slice_on(&plain, &hb, [4, 1]).unwrap());
    assert_eq!(bits(&pa.matmul_f32(&pb)), want);
}

#[test]
fn an_explicit_tf32_product_of_one_term_keeps_its_kernel() {
    if Algebra::current() < Algebra::On {
        return; // LUMA_ALGEBRA below `on`: no rule of the engine runs
    }
    let ctx = open(LazyMode::Never, Numerics::default());
    let a = A::from_slice_on(&ctx, &values(6, 0), [1, 6]).unwrap();
    let b = A::from_slice_on(&ctx, &values(4, 3), [4, 1]).unwrap();
    ctx.record_decisions(true);
    let _ = a.matmul_tf32(&b).to_vec().unwrap();
    assert!(!rewrites(&ctx).iter().any(|rule| rule == "contract_k1"));
}

#[test]
fn the_pre_filter_is_false_for_an_ordinary_product() {
    if Algebra::current() < Algebra::On {
        return; // LUMA_ALGEBRA below `on`: no rule of the engine runs
    }
    let ctx = open(LazyMode::Never, Numerics::default());
    let a = A::from_slice_on(&ctx, &values(12 * 8, 0), [12, 8]).unwrap();
    let b = A::from_slice_on(&ctx, &values(5 * 12, 1), [5, 12]).unwrap();
    assert!(!needs_t1(&ctx, &a, &b));
    let square = A::from_slice_on(&ctx, &values(64, 2), [8, 8]).unwrap();
    let other = A::from_slice_on(&ctx, &values(64, 3), [8, 8]).unwrap();
    assert!(!needs_t1(&ctx, &square, &other), "a square product of two sources is not a candidate");
    assert!(needs_t1(&ctx, &A::identity_on(&ctx, [8, 8]).unwrap(), &square), "an operand with a property is");
}

#[test]
fn a_pending_view_is_composed_into_the_map_and_keeps_the_sum_order() {
    if Algebra::current() < Algebra::On {
        return; // LUMA_ALGEBRA below `on`: no rule of the engine runs
    }
    let ctx = open(LazyMode::Never, Numerics::default());
    let (ha, hb) = (values(5 * 4, 1), values(3 * 4, 2));
    let a = A::from_slice_on(&ctx, &ha, [5, 4]).unwrap();
    // `aᵀ` is [4, 5]: `M = 5`, `K = 4`; `b` must be [N, 4].
    let b = A::from_slice_on(&ctx, &hb, [3, 4]).unwrap();
    let at = a.transpose();
    let form = view(&at, &b, Tf32Request::Default).form().unwrap();
    let moved = t1(form.clone(), &ctx);
    assert_eq!(moved.origin, Origin::Rewritten(super::RuleId(super::Rid::ContractView as u16)));
    assert!(moved.inputs.iter().all(|access| !matches!(&access.src, crate::jit::Src::Node(node) if node.is_pending_view())), "no view is left");
    assert!(lower::lowers(&moved));
    let direct = lower::build(&ctx, &moved);
    let want = at.matmul_f32(&b);
    assert_eq!(bits(&direct), bits(&want));
    let _ = (ha, hb);
}

#[test]
fn the_diagonal_of_a_lazy_product_is_n_squared_work_and_equals_the_product_then_diag() {
    if Algebra::current() < Algebra::On {
        return; // LUMA_ALGEBRA below `on`: no rule of the engine runs
    }
    let ctx = open(LazyMode::Always, Numerics::default());
    let n = 9;
    let (ha, hb) = (values(n * 6, 1), values(n * 6, 2));
    let a = A::from_slice_on(&ctx, &ha, [6, n]).unwrap();
    let b = A::from_slice_on(&ctx, &hb, [n, 6]).unwrap();
    let want = {
        let plain = open(LazyMode::Never, Numerics::strict());
        let (pa, pb) = (A::from_slice_on(&plain, &ha, [6, n]).unwrap(), A::from_slice_on(&plain, &hb, [n, 6]).unwrap());
        bits(&pa.matmul_f32(&pb).diag(0))
    };
    ctx.record_decisions(true);
    let product = a.matmul_f32(&b);
    assert!(product.structured().is_some_and(|value| value.is_unforced()));
    let diagonal = product.diag(0);
    assert_eq!(diagonal.shape().dims(), [n, 1, 1, 1]);
    assert_eq!(bits(&diagonal), want);
    let log = ctx.take_decisions();
    assert!(log.iter().any(|d| d.site == DecisionSite::Rewrite && d.chosen == "diag_of_product"));
    assert!(!log.iter().any(|d| d.site == DecisionSite::MatmulPlan), "no product of n x n x k was planned");
    assert!(product.structured().is_some_and(|value| value.is_unforced()), "the product itself was not forced");
}

#[test]
fn the_diagonal_form_merges_the_two_parallel_dims() {
    if Algebra::current() < Algebra::On {
        return; // LUMA_ALGEBRA below `on`: no rule of the engine runs
    }
    let ctx = open(LazyMode::Never, Numerics::default());
    let a = A::from_slice_on(&ctx, &values(24, 1), [4, 6]).unwrap();
    let b = A::from_slice_on(&ctx, &values(24, 2), [6, 4]).unwrap();
    let form = view(&a, &b, Tf32Request::Default).form().unwrap();
    let diag = diag_of_product_form(&form).unwrap();
    assert_eq!(diag.space.len(), 2);
    assert_eq!(super::super::comprehension::lower::output_dims(&diag), [6, 1, 1, 1]);
    // Only a square product has a diagonal of this kind.
    let wide = A::from_slice_on(&ctx, &values(20, 2), [5, 4]).unwrap();
    let rect = view(&a, &wide, Tf32Request::Default).form().unwrap();
    assert!(diag_of_product_form(&rect).is_none());
}

/// Small integers without a signed zero (an `a·diag(v)` through the sum from `+0.0` has no `-0` either).
fn finite(len: usize, seed: usize) -> Vec<f32> {
    (0..len).map(|i| ((i * 5 + seed * 3) % 11) as f32 - 5.0 + 0.5).collect()
}

#[test]
fn the_form_of_a_gram_product_is_tagged_symmetric_and_the_product_is_bit_symmetric() {
    if Algebra::current() < Algebra::On {
        return; // LUMA_ALGEBRA below `on`: no rule of the engine runs
    }
    use crate::jit::comprehension::Props;
    let ctx = open(LazyMode::Always, Numerics::default());
    let (k, n) = (7, 6);
    let x = A::from_slice_on(&ctx, &finite(k * n, 1), [n, k]).unwrap();
    // `xᵀ·x` is [n, n]: lhs `xᵀ` is [k, n]... in luma's order `matmul_tn(x, x)`.
    let at = x.transpose();
    let form = t1(view(&at, &x, Tf32Request::Default).form().unwrap(), &ctx);
    assert!(form.props.has(Props::SYMMETRIC), "{:?}", form.props);
    let c = x.matmul_tn(&x).to_vec().unwrap();
    for i in 0..n {
        for j in 0..n {
            assert_eq!(c[i * n + j].to_bits(), c[j * n + i].to_bits());
        }
    }
    // `x·xᵀ` is symmetric too, and two different sources are not.
    let xt = x.transpose();
    assert!(t1(view(&x, &xt, Tf32Request::Default).form().unwrap(), &ctx).props.has(Props::SYMMETRIC));
    let y = A::from_slice_on(&ctx, &finite(k * n, 2), [n, k]).unwrap();
    assert!(!t1(view(&at, &y, Tf32Request::Default).form().unwrap(), &ctx).props.has(Props::SYMMETRIC));
}

#[test]
fn a_product_with_the_identity_is_the_operand_under_algebraic_and_a_product_otherwise() {
    if Algebra::current() < Algebra::On {
        return; // LUMA_ALGEBRA below `on`: no rule of the engine runs
    }
    let n = 6;
    let data = finite(n * 4, 1);
    for (numerics, rewritten) in [(Numerics::algebraic(), true), (Numerics::default(), false)] {
        let ctx = open(LazyMode::Never, numerics);
        let a = A::from_slice_on(&ctx, &data, [n, 4]).unwrap();
        let eye = A::identity_on(&ctx, [n, n]).unwrap();
        ctx.record_decisions(true);
        let right = a.matmul(&eye);
        assert_eq!(right.shape().dims(), [n, 4, 1, 1]);
        let want: Vec<u32> = data.iter().map(|v| v.to_bits()).collect();
        assert_eq!(bits(&right), want);
        let log = ctx.take_decisions();
        let fired = log.iter().filter(|d| d.site == DecisionSite::Rewrite && d.chosen == "contract_identity").count();
        let planned = log.iter().filter(|d| d.site == DecisionSite::MatmulPlan).count();
        // `identity · a` is routed as a permutation by W3-R6 whatever the numerics: it is the operand with no rule and no plan.
        let left = A::identity_on(&ctx, [4, 4]).unwrap().matmul(&a);
        assert_eq!(bits(&left), want);
        assert_eq!((fired, planned), if rewritten { (1, 0) } else { (0, 1) }, "{numerics}");
    }
}

#[test]
fn a_product_with_a_diagonal_matrix_is_a_broadcast_product_under_algebraic() {
    if Algebra::current() < Algebra::On {
        return; // LUMA_ALGEBRA below `on`: no rule of the engine runs
    }
    let (k, m) = (6, 4);
    let (da, dv) = (finite(k * m, 1), finite(k, 4));
    for (numerics, rewritten) in [(Numerics::algebraic(), true), (Numerics::default(), false)] {
        let ctx = open(LazyMode::Never, numerics);
        let a = A::from_slice_on(&ctx, &da, [k, m]).unwrap();
        let v = A::from_slice_on(&ctx, &dv, [k, 1]).unwrap();
        ctx.record_decisions(true);
        let scaled = a.matmul(&v.diag_matrix(0));
        let want: Vec<u32> = (0..m).flat_map(|row| (0..k).map(move |col| (row, col))).map(|(row, col)| (da[col + k * row] * dv[col]).to_bits()).collect();
        assert_eq!(bits(&scaled), want, "{numerics}");
        // `diag(v)·a`: the rows of `a` scaled by `v` (a is [N, K] with K = m here).
        let w = A::from_slice_on(&ctx, &finite(m, 2), [m, 1]).unwrap();
        let b = A::from_slice_on(&ctx, &finite(5 * m, 3), [5, m]).unwrap();
        let rows = w.diag_matrix(0).matmul(&b);
        let hb = finite(5 * m, 3);
        let hw = finite(m, 2);
        let want: Vec<u32> = (0..m).flat_map(|row| (0..5).map(move |col| (row, col))).map(|(row, col)| (hw[row] * hb[col + 5 * row]).to_bits()).collect();
        assert_eq!(bits(&rows), want, "{numerics}");
        let log = ctx.take_decisions();
        let fired = log.iter().filter(|d| d.site == DecisionSite::Rewrite && d.chosen == "contract_diag").count();
        let planned = log.iter().filter(|d| d.site == DecisionSite::MatmulPlan).count();
        // `diag(v)·b` may be routed by W3-R6 instead of rewritten: the rule fires for `a·diag(v)` at least, and a product is planned only
        // where it did not.
        assert!(if rewritten { fired >= 1 && planned == 0 } else { fired == 0 && planned >= 1 }, "{numerics}: {fired} rewrites, {planned} plans");
    }
}

// -- tier 2 --

/// `[K, M] · [N, K]` on the host in `f64`: row `m` of the result at `[n + N m]`.
fn host_product(a: &[f32], b: &[f32], k: usize, m: usize, n: usize) -> Vec<f64> {
    (0..m).flat_map(|i| (0..n).map(move |j| (i, j))).map(|(i, j)| (0..k).map(|l| f64::from(a[l + k * i]) * f64::from(b[j + n * l])).sum()).collect()
}

fn close(got: &[f32], want: &[f64], tol: f64) {
    assert_eq!(got.len(), want.len());
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!((f64::from(*g) - w).abs() <= tol * (w.abs() + 1.0), "element {i}: {g} against {w}");
    }
}

fn rows(ctx: &Context) -> (Vec<DecisionSite>, Vec<String>) {
    let log = ctx.take_decisions();
    (log.iter().map(|d| d.site).collect(), log.into_iter().filter(|d| d.site == DecisionSite::Rewrite).map(|d| d.chosen.to_string()).collect())
}

#[test]
fn the_sum_of_a_product_is_two_sums_and_a_dot_under_algebraic() {
    if Algebra::current() < Algebra::On {
        return; // LUMA_ALGEBRA below `on`: no rule of the engine runs
    }
    let (k, m, n) = (256, 256, 256);
    let (ha, hb) = (finite(k * m, 1), finite(n * k, 2));
    let want: f64 = host_product(&ha, &hb, k, m, n).iter().sum();
    for (numerics, rewritten) in [(Numerics::algebraic(), true), (Numerics::default(), false)] {
        let ctx = open(LazyMode::Never, numerics);
        let a = A::from_slice_on(&ctx, &ha, [k, m]).unwrap();
        let b = A::from_slice_on(&ctx, &hb, [n, k]).unwrap();
        ctx.record_decisions(true);
        let total = a.matmul(&b).sum();
        close(&[total.item().unwrap()], &[want], 1e-5);
        let (sites, fired) = rows(&ctx);
        assert_eq!(fired.iter().any(|rule| rule == "sum_of_product"), rewritten, "{numerics}: {fired:?}");
        assert_eq!(sites.contains(&DecisionSite::MatmulPlan), !rewritten, "{numerics}: the n^3 product runs only if the rule did not fire");
    }
}

#[test]
fn the_sum_along_an_axis_of_a_product_is_a_vector_product_under_algebraic() {
    if Algebra::current() < Algebra::On {
        return; // LUMA_ALGEBRA below `on`: no rule of the engine runs
    }
    let (k, m, n) = (512, 448, 480);
    let (ha, hb) = (finite(k * m, 3), finite(n * k, 4));
    let full = host_product(&ha, &hb, k, m, n);
    let ctx = open(LazyMode::Never, Numerics::algebraic());
    let a = A::from_slice_on(&ctx, &ha, [k, m]).unwrap();
    let b = A::from_slice_on(&ctx, &hb, [n, k]).unwrap();
    ctx.record_decisions(true);
    // Axis 0 adds over `n`: one value per row `m`.
    let rows0: Vec<f64> = (0..m).map(|i| (0..n).map(|j| full[j + n * i]).sum()).collect();
    close(&a.matmul(&b).sum_axis(0).to_vec().unwrap(), &rows0, 1e-5);
    // Axis 1 adds over `m`: one value per column `n`.
    let cols: Vec<f64> = (0..n).map(|j| (0..m).map(|i| full[j + n * i]).sum()).collect();
    close(&a.matmul(&b).sum_axis(1).to_vec().unwrap(), &cols, 1e-5);
    let (sites, fired) = rows(&ctx);
    assert_eq!(fired.iter().filter(|rule| *rule == "sum_of_product").count(), 2, "{fired:?}");
    assert!(!sites.contains(&DecisionSite::MatmulPlan));
}

#[test]
fn a_chain_with_a_vector_is_multiplied_from_the_vector_under_algebraic() {
    if Algebra::current() < Algebra::On {
        return; // LUMA_ALGEBRA below `on`: no rule of the engine runs
    }
    let n = 384;
    let (ha, hb, hv) = (finite(n * n, 1), finite(n * n, 2), finite(n, 3));
    // (a·b)·v on the host in f64: v is [1, n] (the column), the product [1, n].
    let ab = host_product(&ha, &hb, n, n, n);
    let want: Vec<f64> = (0..n).map(|i| (0..n).map(|l| ab[l + n * i] * f64::from(hv[l])).sum()).collect();
    for (numerics, rewritten) in [(Numerics::algebraic(), true), (Numerics::default(), false)] {
        let ctx = open(LazyMode::Always, numerics);
        let a = A::from_slice_on(&ctx, &ha, [n, n]).unwrap();
        let b = A::from_slice_on(&ctx, &hb, [n, n]).unwrap();
        let v = A::from_slice_on(&ctx, &hv, [1, n]).unwrap();
        ctx.record_decisions(true);
        let got = a.matmul_f32(&b).matmul_f32(&v);
        assert_eq!(got.shape().dims(), [1, n, 1, 1]);
        close(&got.to_vec().unwrap(), &want, 1e-4);
        let (_, fired) = rows(&ctx);
        assert_eq!(fired.iter().any(|rule| rule == "chain_order"), rewritten, "{numerics}: {fired:?}");
    }
}

#[test]
fn the_rules_of_products_fire_exactly_under_the_flags_that_allow_them() {
    if Algebra::current() < Algebra::On {
        return; // LUMA_ALGEBRA below `on`: no rule of the engine runs
    }
    let n = 6;
    let data = finite(n * 4, 1);
    for bits in 0..8u8 {
        let numerics = Numerics { exact_finite: bits & 1 != 0, reassociate: bits & 2 != 0, reformulate: bits & 4 != 0, ..Numerics::strict() };
        let ctx = open(LazyMode::Never, numerics);
        let a = A::from_slice_on(&ctx, &data, [n, 4]).unwrap();
        let one = A::from_slice_on(&ctx, &values(4, 1), [1, 4]).unwrap();
        let row = A::from_slice_on(&ctx, &values(n, 2), [n, 1]).unwrap();
        ctx.record_decisions(true);
        let _ = a.matmul(&A::identity_on(&ctx, [n, n]).unwrap()).to_vec().unwrap();
        let _ = a.matmul(&row.diag_matrix(0)).to_vec().unwrap();
        let _ = one.matmul(&row).to_vec().unwrap();
        let (_, fired) = rows(&ctx);
        let fired = |rule: &str| fired.iter().any(|name| name == rule);
        assert!(fired("contract_k1"), "{numerics}: exact, always");
        assert_eq!(fired("contract_identity"), numerics.reformulate, "{numerics}");
        assert_eq!(fired("contract_diag"), numerics.reformulate, "{numerics}");
    }
}

#[test]
fn a_scalar_factor_of_an_operand_is_the_epilogue_of_the_product_under_reassociate() {
    if Algebra::current() < Algebra::On {
        return; // LUMA_ALGEBRA below `on`: no rule of the engine runs
    }
    let (k, m, n) = (96, 64, 80);
    let (ha, hb) = (finite(k * m, 5), finite(n * k, 6));
    let full = host_product(&ha, &hb, k, m, n);
    let scaled: Vec<f64> = full.iter().map(|v| v * 2.5 * -0.75).collect();
    for (numerics, rewritten) in [(Numerics::algebraic(), true), (Numerics::default(), false), (Numerics { reassociate: true, ..Numerics::strict() }, true)] {
        let ctx = open(LazyMode::Never, numerics);
        let a = A::from_slice_on(&ctx, &ha, [k, m]).unwrap();
        let b = A::from_slice_on(&ctx, &hb, [n, k]).unwrap();
        ctx.record_decisions(true);
        // The factor of each operand: both move out of the product.
        let c = (&a * 2.5).matmul_f32(&(&b * -0.75));
        close(&c.to_vec().unwrap(), &scaled, 1e-4);
        let log = ctx.take_decisions();
        let fired = log.iter().filter(|d| d.site == DecisionSite::Rewrite && d.chosen == "scale_through").count();
        let absorbed = log.iter().filter(|d| d.site == DecisionSite::Epilogue && d.chosen.to_string().starts_with("absorbed")).count();
        assert_eq!(fired > 0, rewritten, "{numerics}: {log:?}");
        assert_eq!(absorbed > 0, rewritten, "{numerics}: the factors are steps of the product's epilogue");
    }
}

#[test]
fn the_scale_pre_filter_is_only_asked_when_the_numerics_reassociate() {
    if Algebra::current() < Algebra::On {
        return; // LUMA_ALGEBRA below `on`: no rule of the engine runs
    }
    for (numerics, wants) in [(Numerics::default(), false), (Numerics::algebraic(), true)] {
        let ctx = open(LazyMode::Never, numerics);
        let a = A::from_slice_on(&ctx, &values(12 * 8, 0), [12, 8]).unwrap();
        let b = A::from_slice_on(&ctx, &values(5 * 12, 1), [5, 12]).unwrap();
        assert!(!needs_t1(&ctx, &a, &b), "{numerics}");
        assert_eq!(needs_t1(&ctx, &(&a * 3.0), &b), wants, "{numerics}: a scaled operand");
        assert_eq!(needs_t1(&ctx, &a, &(&b * 3.0)), wants, "{numerics}: a scaled operand");
    }
}

#[test]
fn a_factor_that_does_not_depend_on_the_summed_index_leaves_the_sum_under_reformulate() {
    if Algebra::current() < Algebra::On {
        return; // LUMA_ALGEBRA below `on`: no rule of the engine runs
    }
    // `out(i) = Σ_j s(i)·x(j, i)` with `s` an expensive function of a vector: s is read once per output after the rule.
    let (cols, rows) = (1024, 2048);
    let (hx, hv) = (finite(cols * rows, 3), finite(rows, 4));
    let want: Vec<f64> = (0..rows).map(|i| { let s = f64::from(hv[i]).exp().sqrt().sin(); (0..cols).map(|j| f64::from(hx[j + cols * i]) * s).sum() }).collect();
    for (numerics, rewritten) in [(Numerics::algebraic(), true), (Numerics::default(), false), (Numerics { reformulate: true, ..Numerics::strict() }, true), (Numerics { reassociate: true, ..Numerics::strict() }, false)] {
        let ctx = open(LazyMode::Never, numerics);
        let x = A::from_slice_on(&ctx, &hx, [cols, rows]).unwrap();
        let v = A::from_slice_on(&ctx, &hv, [1, rows]).unwrap();
        ctx.record_decisions(true);
        let s = v.exp().sqrt().sin();
        let got = (&x * &s).sum_axis(0);
        assert_eq!(got.shape().dims(), [1, rows, 1, 1]);
        close(&got.to_vec().unwrap(), &want, 1e-4);
        let (_, fired) = rows_of(&ctx);
        assert_eq!(fired.iter().any(|rule| rule == "factor_invariant"), rewritten, "{numerics}: {fired:?}");
    }
}

fn rows_of(ctx: &Context) -> (Vec<DecisionSite>, Vec<String>) {
    rows(ctx)
}
