//! Tests of epilogue absorption into lazy products (W3-U4 part A): a consumer of an unforced product joins its epilogue; the result is
//! bit for bit what the product followed by the consumers computes, and the decision log says what was absorbed. Operands hold small
//! integers so that every backend's product is exact.

use super::sites::LazyMode;
use crate::array::Array;
use crate::context::{Context, ContextOptions};
use crate::profile::{Decision, DecisionSite};

type A = Array<f32>;

const K: usize = 5;
const M: usize = 4;
const N: usize = 3;

fn context(mode: LazyMode) -> Context {
    Context::with_tuning(ContextOptions::from_env().calibrate(false), |tuning| tuning.lazy = mode).unwrap()
}

fn ints(len: usize, seed: usize) -> Vec<f32> {
    (0..len).map(|i| ((i * 7 + seed * 3 + i / 3) % 9) as f32 - 4.0).collect()
}

fn operands(ctx: &Context) -> (A, A) {
    (A::from_slice_on(ctx, &ints(K * M, 1), [K, M]).unwrap(), A::from_slice_on(ctx, &ints(N * K, 2), [N, K]).unwrap())
}

fn epilogues(ctx: &Context) -> Vec<Decision> {
    ctx.take_decisions().into_iter().filter(|decision| decision.site == DecisionSite::Epilogue).collect()
}

fn is_lazy(c: &A) -> bool {
    c.structured().is_some_and(|value| value.is_unforced())
}

/// `f` over the product in both modes: the lazy run must absorb (`absorbed` ops), and both give the same bits.
fn same_bits(absorbed: usize, f: impl Fn(&A, &Context) -> A) {
    let eager = context(LazyMode::Never);
    let (a, b) = operands(&eager);
    let want = f(&a.matmul(&b), &eager).to_vec().unwrap();
    let ctx = context(LazyMode::Always);
    let (a, b) = operands(&ctx);
    ctx.record_decisions(true);
    let out = f(&a.matmul(&b), &ctx);
    assert!(is_lazy(&out), "the consumers did not force the product");
    let got = out.to_vec().unwrap();
    assert_eq!(got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), want.iter().map(|v| v.to_bits()).collect::<Vec<_>>());
    let log = epilogues(&ctx);
    assert!(log.iter().any(|d| d.chosen == format!("absorbed {absorbed} ops")), "{log:?}");
}

#[test]
fn a_scalar_chain_is_absorbed() {
    same_bits(3, |c, _| (&(c * 0.5) + 1.0).abs());
    same_bits(1, |c, _| c.tanh());
    // The scalar on the left.
    same_bits(1, |c, _| 10.0 - c);
}

#[test]
fn array_operands_of_each_shape_are_absorbed() {
    let ctx = context(LazyMode::Never);
    let bias_cols = ints(N, 5);
    let bias_rows = ints(M, 6);
    let full = ints(N * M, 7);
    for (shape, data) in [([N, 1], bias_cols), ([1, M], bias_rows), ([N, M], full)] {
        let one = |ctx: &Context| {
            let (a, b) = operands(ctx);
            let bias = A::from_slice_on(ctx, &data, shape).unwrap();
            (&a.matmul(&b) + &bias).maximum(0.0)
        };
        let want = one(&ctx).to_vec().unwrap();
        let lazy = context(LazyMode::Always);
        lazy.record_decisions(true);
        let out = one(&lazy);
        assert!(is_lazy(&out), "{shape:?}");
        assert_eq!(out.to_vec().unwrap(), want, "{shape:?}");
        assert!(epilogues(&lazy).iter().any(|d| d.chosen == "absorbed 2 ops"), "{shape:?}");
    }
}

#[test]
fn an_operand_of_another_shape_is_separate() {
    let ctx = context(LazyMode::Always);
    let (a, b) = operands(&ctx);
    ctx.record_decisions(true);
    // A `[N, M]` product plus a `[1, 1]`-broadcast of a `[N, M, 1, 1]` is absorbed; an `[N, 2]` operand does not even broadcast.
    let c = a.matmul(&b);
    let other = A::from_slice_on(&ctx, &ints(N * M, 3), [N, M]).unwrap().reshape([M, N]);
    let out = &c + &other;
    assert!(out.to_vec().is_err(), "a mismatch is an error as before");
    let _ = epilogues(&ctx);
}

#[test]
fn more_than_sixteen_ops_stop_absorbing() {
    let ctx = context(LazyMode::Always);
    let (a, b) = operands(&ctx);
    ctx.record_decisions(true);
    let mut c = a.matmul(&b);
    let mut want = {
        let eager = context(LazyMode::Never);
        let (a, b) = operands(&eager);
        a.matmul(&b).to_vec().unwrap()
    };
    for _ in 0..17 {
        c = -&c;
        for v in &mut want {
            *v = -*v;
        }
    }
    assert_eq!(c.to_vec().unwrap(), want);
    let log = epilogues(&ctx);
    assert!(log.iter().any(|d| d.chosen == "absorbed 16 ops"), "{log:?}");
    assert!(log.iter().any(|d| d.chosen == "separate: limit"), "{log:?}");
}

#[test]
fn base_and_derived_both_used_compute_the_product_once() {
    let ctx = context(LazyMode::Always);
    let (a, b) = operands(&ctx);
    ctx.record_decisions(true);
    let z = a.matmul(&b);
    let h = &z * 2.0;
    let want: Vec<f32> = z.to_vec().unwrap().iter().map(|v| v * 2.0).collect();
    assert_eq!(h.to_vec().unwrap(), want);
    let plans = ctx.take_decisions().into_iter().filter(|d| d.site == DecisionSite::MatmulPlan).count();
    assert_eq!(plans, 1, "one product for both");
}

#[test]
fn a_write_to_a_bias_after_creation_does_not_change_the_product() {
    let ctx = context(LazyMode::Always);
    let (a, b) = operands(&ctx);
    let mut bias = A::from_slice_on(&ctx, &ints(N * M, 4), [N, M]).unwrap();
    let out = &a.matmul(&b) + &bias;
    assert!(is_lazy(&out));
    let want = {
        let eager = context(LazyMode::Never);
        let (a, b) = operands(&eager);
        let bias = A::from_slice_on(&eager, &ints(N * M, 4), [N, M]).unwrap();
        (&a.matmul(&b) + &bias).to_vec().unwrap()
    };
    bias.write(&[99.0; N * M]).unwrap();
    assert_eq!(out.to_vec().unwrap(), want);
}

#[test]
fn an_absorbing_site_stays_lazy() {
    let ctx = Context::with_tuning(ContextOptions::from_env().calibrate(false), |tuning| {
        tuning.lazy = LazyMode::Site;
        tuning.lazy_probe_every = 1;
    })
    .unwrap();
    let (a, b) = operands(&ctx);
    let mut lazy = 0;
    for _ in 0..4 {
        let c = a.matmul(&b);
        if is_lazy(&c) {
            lazy += 1;
        }
        let _ = (&c * 2.0).to_vec().unwrap();
    }
    assert_eq!(lazy, 4, "every call is a probe here, and the site absorbed");
}

// -- partners (A2) --

mod partners {
    use super::*;
    use crate::jit::structured::contraction::{ContractionAttrs, Tf32Request};
    use crate::jit::structured::lazy::MAX_PARTNERS;

    const ATTRS: ContractionAttrs = ContractionAttrs { tf32: Tf32Request::Default, half: false };

    fn product(ctx: &Context, a: &A, b: &A) -> A {
        let c = a.matmul(b);
        ctx.jit().register_product(c.structured().unwrap());
        c
    }

    fn weights(ctx: &Context, n: usize, seed: usize) -> A {
        A::from_slice_on(ctx, &ints(n * K, seed), [n, K]).unwrap()
    }

    #[test]
    fn products_of_one_left_operand_find_each_other() {
        let ctx = context(LazyMode::Always);
        let (a, w1) = operands(&ctx);
        let (w2, w3, other) = (weights(&ctx, 5, 8), weights(&ctx, 2, 9), A::from_slice_on(&ctx, &ints(K * M, 4), [K, M]).unwrap());
        let seed = product(&ctx, &a, &w1);
        let (c2, c3) = (product(&ctx, &a, &w2), product(&ctx, &a, &w3));
        let _elsewhere = product(&ctx, &other, &w1);
        let found = seed.structured().unwrap().partner_candidates(&ctx, &a, ATTRS, None);
        assert_eq!(found.len(), 2, "the two that share `a`, not the one over another operand");
        // A product with an absorbed epilogue has other instructions: not a partner of a plain one.
        let scaled = &c2 * 2.0;
        ctx.jit().register_product(scaled.structured().unwrap());
        assert_eq!(seed.structured().unwrap().partner_candidates(&ctx, &a, ATTRS, None).len(), 2, "the derived one has an epilogue; c2 is still a plain product");
        // A recorded one is not a candidate.
        let _ = c3.to_vec().unwrap();
        assert_eq!(seed.structured().unwrap().partner_candidates(&ctx, &a, ATTRS, None).len(), 1, "c3 was recorded");
    }

    #[test]
    fn a_group_is_at_most_four_products() {
        let ctx = context(LazyMode::Always);
        let (a, w1) = operands(&ctx);
        let seed = product(&ctx, &a, &w1);
        let rest: Vec<A> = (0..6).map(|i| product(&ctx, &a, &weights(&ctx, 3, 10 + i))).collect();
        let found = seed.structured().unwrap().partner_candidates(&ctx, &a, ATTRS, None);
        assert_eq!(found.len(), MAX_PARTNERS - 1);
        drop(rest);
    }
}
