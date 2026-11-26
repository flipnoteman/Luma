//! Tests of the lazy product (W2-LAZY): what a product launched when it is used does that the eager one did at the call. Most run in
//! both modes (`Never` and `Always`) and say the same thing about the results; what only the lazy one can show (it records nothing
//! until it is used, it survives a write into an operand) asks for `Always`. Operands hold small integers, so every backend's product
//! (the tensor cores' TF32 included) is exact and the results compare bit for bit.

use std::panic::Location;

use super::sites::{LazyMode, SiteKey};
use crate::array::Array;
use crate::context::{Context, ContextOptions};
use crate::error::LumaError;
use crate::profile::{Decision, DecisionSite};

type A = Array<f32>;

/// `lhs` is `[K, M]`, `rhs` `[N, K]`.
const K: usize = 5;
const M: usize = 4;
const N: usize = 3;

fn options() -> ContextOptions {
    ContextOptions::from_env().calibrate(false)
}

fn context(mode: LazyMode) -> Context {
    Context::with_tuning(options(), |tuning| tuning.lazy = mode).unwrap()
}

/// Small integers, so that products and their sums are exact in every precision.
fn ints(len: usize, seed: usize) -> Vec<f32> {
    (0..len).map(|i| ((i * 7 + seed * 3 + i / 3) % 9) as f32 - 4.0).collect()
}

/// The product `C[n + N m] = Σ_k A[k + K m] B[n + N k]` on the host.
fn host(a: &[f32], b: &[f32], k: usize, m: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; n * m];
    for i in 0..m {
        for j in 0..n {
            c[j + n * i] = (0..k).map(|l| a[l + k * i] * b[j + n * l]).sum();
        }
    }
    c
}

struct Operands {
    a: A,
    b: A,
    ha: Vec<f32>,
    hb: Vec<f32>,
}

fn operands(ctx: &Context) -> Operands {
    let (ha, hb) = (ints(K * M, 1), ints(N * K, 2));
    Operands { a: A::from_slice_on(ctx, &ha, [K, M]).unwrap(), b: A::from_slice_on(ctx, &hb, [N, K]).unwrap(), ha, hb }
}

/// Whether `c` is a product that was made and not recorded yet.
fn is_lazy(c: &A) -> bool {
    c.structured().is_some_and(|value| value.is_unforced())
}

/// The plans of matrix products recorded since the last call.
fn plans(ctx: &Context) -> Vec<Decision> {
    ctx.take_decisions().into_iter().filter(|decision| decision.site == DecisionSite::MatmulPlan).collect()
}

const BOTH: [LazyMode; 2] = [LazyMode::Never, LazyMode::Always];

#[test]
fn a_lazy_product_records_nothing_until_it_is_used() {
    let ctx = context(LazyMode::Always);
    let Operands { a, b, ha, hb } = operands(&ctx);
    ctx.record_decisions(true);
    let doubled = &a * 2.0;
    let before = ctx.jit().stats();
    let c = doubled.matmul(&b);
    assert!(is_lazy(&c), "the product is a value that has not been recorded");
    assert!(doubled.pending().is_some(), "its operand was not evaluated for it");
    assert_eq!(c.shape().dims(), [N, M, 1, 1]);
    assert!(plans(&ctx).is_empty(), "no strategy was selected at the call");
    assert_eq!(ctx.jit().stats().elementwise_launches, before.elementwise_launches);
    let want = host(&ha.iter().map(|v| v * 2.0).collect::<Vec<_>>(), &hb, K, M, N);
    assert_eq!(c.to_vec().unwrap(), want);
    assert!(!is_lazy(&c), "using it recorded it");
    assert_eq!(plans(&ctx).len(), 1, "one selection, at the force");
    // Once recorded it is a buffer like any other, and using it again records nothing.
    assert_eq!(c.to_vec().unwrap(), want);
    assert!(plans(&ctx).is_empty());
    // The default mode launches at the call.
    let eager = context(LazyMode::Never);
    let Operands { a, b, .. } = operands(&eager);
    eager.record_decisions(true);
    let c = a.matmul(&b);
    assert!(c.structured().is_none());
    assert_eq!(plans(&eager).len(), 1, "the selection is at the call");
    assert_eq!(c.to_vec().unwrap(), host(&ha, &hb, K, M, N));
}

#[test]
fn a_dropped_product_is_never_computed_and_releases_its_operands() {
    let ctx = context(LazyMode::Always);
    let Operands { a, b, .. } = operands(&ctx);
    ctx.record_decisions(true);
    let c = (&a * 2.0).matmul(&b);
    assert!(is_lazy(&c));
    drop(c);
    ctx.sync().unwrap();
    assert!(plans(&ctx).is_empty(), "nothing asked for the product");
    // The operands are not held by anything the product left behind.
    assert!(a.recorded_storage().is_some_and(|storage| std::sync::Arc::strong_count(storage) == 1));
}

#[test]
fn a_forced_product_no_longer_holds_its_operands() {
    let ctx = context(LazyMode::Always);
    let Operands { a, b, ha, hb } = operands(&ctx);
    let c = a.matmul(&b);
    assert!(a.recorded_storage().is_some_and(|storage| std::sync::Arc::strong_count(storage) > 1), "an unforced product pins its operands");
    assert_eq!(c.to_vec().unwrap(), host(&ha, &hb, K, M, N));
    assert!(a.recorded_storage().is_some_and(|storage| std::sync::Arc::strong_count(storage) == 1), "a forced one releases them");
}

#[test]
fn a_write_after_a_pending_product_does_not_change_it() {
    for mode in BOTH {
        let ctx = context(mode);
        let lazy = mode == LazyMode::Always;
        let new = vec![100.0f32; K * M];
        let new_b = vec![-7.0f32; N * K];
        // The operand is a valid buffer.
        let Operands { mut a, b, ha, hb } = operands(&ctx);
        let c = a.matmul(&b);
        assert_eq!(is_lazy(&c), lazy);
        a.write(&new).unwrap();
        assert_eq!(c.to_vec().unwrap(), host(&ha, &hb, K, M, N), "{mode:?}: lhs written");
        // The right operand.
        let Operands { a, mut b, ha, hb } = operands(&ctx);
        let c = a.matmul(&b);
        b.write(&new_b).unwrap();
        assert_eq!(c.to_vec().unwrap(), host(&ha, &hb, K, M, N), "{mode:?}: rhs written");
        // Both.
        let Operands { mut a, mut b, ha, hb } = operands(&ctx);
        let c = a.matmul(&b);
        a.write(&new).unwrap();
        b.write(&new_b).unwrap();
        assert_eq!(c.to_vec().unwrap(), host(&ha, &hb, K, M, N), "{mode:?}: both written");
        // The operand is a pending expression, written after the product was made (reviewer B2): the write evaluates the expression and
        // then what reads its buffer, which has to include the product.
        let Operands { a: x, b, ha, hb } = operands(&ctx);
        let mut doubled = &x * 2.0;
        assert!(doubled.pending().is_some());
        let c = doubled.matmul(&b);
        assert_eq!(is_lazy(&c), lazy);
        doubled.write(&new).unwrap();
        let old: Vec<f32> = ha.iter().map(|v| v * 2.0).collect();
        assert_eq!(c.to_vec().unwrap(), host(&old, &hb, K, M, N), "{mode:?}: pending lhs written");
        // The same for the right operand.
        let Operands { a, b: y, ha, hb } = operands(&ctx);
        let mut negated = -&y;
        assert!(negated.pending().is_some());
        let c = a.matmul(&negated);
        negated.write(&new_b).unwrap();
        let old: Vec<f32> = hb.iter().map(|v| -v).collect();
        assert_eq!(c.to_vec().unwrap(), host(&ha, &old, K, M, N), "{mode:?}: pending rhs written");
        // Both pending, both written.
        let Operands { a: x, b: y, ha, hb } = operands(&ctx);
        let (mut p, mut q) = (&x + 1.0, &y - 1.0);
        let c = p.matmul(&q);
        p.write(&new).unwrap();
        q.write(&new_b).unwrap();
        let (old_a, old_b): (Vec<f32>, Vec<f32>) = (ha.iter().map(|v| v + 1.0).collect(), hb.iter().map(|v| v - 1.0).collect());
        assert_eq!(c.to_vec().unwrap(), host(&old_a, &old_b, K, M, N), "{mode:?}: both pending operands written");
        // What a pending operand reads is written: the expression is evaluated with the old values (the write flushes its readers), and
        // the product reads that.
        let Operands { a: mut x, b, ha, hb } = operands(&ctx);
        let doubled = &x * 2.0;
        let c = doubled.matmul(&b);
        x.write(&new).unwrap();
        let old: Vec<f32> = ha.iter().map(|v| v * 2.0).collect();
        assert_eq!(c.to_vec().unwrap(), host(&old, &hb, K, M, N), "{mode:?}: the root of the pending operand written");
    }
}

#[test]
fn a_write_into_the_operand_of_a_chain_does_not_change_the_products_downstream() {
    for mode in BOTH {
        let ctx = context(mode);
        let Operands { a, b, ha, hb } = operands(&ctx);
        // `c1` is `[N, M]`, `c2 = c1 * d` with `d` `[3, 3]`.
        let hd = ints(N * N, 5);
        let d = A::from_slice_on(&ctx, &hd, [N, N]).unwrap();
        let mut c1 = a.matmul(&b);
        let c2 = c1.matmul(&d);
        c1.write(&[1000.0; N * M]).unwrap();
        let first = host(&ha, &hb, K, M, N);
        assert_eq!(c2.to_vec().unwrap(), host(&first, &hd, N, M, N), "{mode:?}: the middle of a chain written");
    }
}

#[test]
fn a_write_into_the_operand_of_a_reduction_valued_product_does_not_change_it() {
    for mode in BOTH {
        let ctx = context(mode);
        // `[1]` times `[1]`: a sum is a one-element array, and a product of one-element arrays is legal.
        let x = A::from_slice_on(&ctx, &[1.0, 2.0, 3.0, 4.0], 4).unwrap();
        let mut sum = x.sum();
        let t = A::from_slice_on(&ctx, &[3.0], 1).unwrap();
        let c = sum.matmul(&t);
        sum.write(&[100.0]).unwrap();
        assert_eq!(c.to_vec().unwrap(), [30.0], "{mode:?}");
    }
}

#[test]
fn a_product_of_a_lazy_product_chains() {
    for mode in BOTH {
        let ctx = context(mode);
        let Operands { a, b, ha, hb } = operands(&ctx);
        let hd = ints(N * N, 5);
        let d = A::from_slice_on(&ctx, &hd, [N, N]).unwrap();
        ctx.record_decisions(true);
        let c = a.matmul(&b).matmul(&d);
        if mode == LazyMode::Always {
            assert!(is_lazy(&c));
            assert!(plans(&ctx).is_empty(), "neither product was recorded");
        } else {
            assert_eq!(plans(&ctx).len(), 2, "both were recorded at their calls");
        }
        assert_eq!(c.to_vec().unwrap(), host(&host(&ha, &hb, K, M, N), &hd, N, M, N), "{mode:?}");
        if mode == LazyMode::Always {
            assert_eq!(plans(&ctx).len(), 2, "the same two selections, at the force");
        }
    }
}

#[test]
fn a_lazy_product_used_twice_is_computed_once() {
    let ctx = context(LazyMode::Always);
    let Operands { a, b, ha, hb } = operands(&ctx);
    ctx.record_decisions(true);
    let c = a.matmul(&b);
    let again = c.clone();
    let sum = &c + &again;
    let product = host(&ha, &hb, K, M, N);
    assert_eq!(sum.to_vec().unwrap(), product.iter().map(|v| v * 2.0).collect::<Vec<_>>());
    assert_eq!(plans(&ctx).len(), 1, "the clones share one product");
    // Two expressions reading it, evaluated together.
    let c = a.matmul(&b);
    let (twice, thrice) = (&c * 2.0, &c * 3.0);
    assert_eq!(twice.to_vec().unwrap(), product.iter().map(|v| v * 2.0).collect::<Vec<_>>());
    assert_eq!(thrice.to_vec().unwrap(), product.iter().map(|v| v * 3.0).collect::<Vec<_>>());
    assert_eq!(plans(&ctx).len(), 1, "two readers of one product");
    // A reshape of it is the same product.
    let c = a.matmul(&b);
    assert!(is_lazy(&c.reshape([M, N])));
    assert_eq!(c.reshape([M, N]).to_vec().unwrap(), product);
    assert_eq!(plans(&ctx).len(), 1);
}

#[test]
fn every_variant_of_the_product_is_lazy_and_agrees_with_the_eager_one() {
    let (lazy, eager) = (context(LazyMode::Always), context(LazyMode::Never));
    let results = |ctx: &Context| {
        let (ha, hb) = (ints(K * M, 1), ints(N * K, 2));
        let a = A::from_slice_on(ctx, &ha, [K, M]).unwrap();
        let b = A::from_slice_on(ctx, &hb, [N, K]).unwrap();
        // The transposed forms take views of the other layout.
        let (at, bt) = (A::from_slice_on(ctx, &ha, [M, K]).unwrap(), A::from_slice_on(ctx, &hb, [K, N]).unwrap());
        let products = [a.matmul(&b), a.matmul_f32(&b), a.matmul_tf32(&b), a.matmul_nt(&bt), at.matmul_tn(&b), at.matmul_tt(&bt), (&a * 1.0).matmul(&(&b + 0.0))];
        products.iter().map(|c| (is_lazy(c), c.to_vec().unwrap())).collect::<Vec<_>>()
    };
    let (lazy, eager) = (results(&lazy), results(&eager));
    assert!(lazy.iter().all(|(is_lazy, _)| *is_lazy), "{:?}", lazy.iter().map(|(is_lazy, _)| *is_lazy).collect::<Vec<_>>());
    assert!(eager.iter().all(|(is_lazy, _)| !*is_lazy));
    for ((_, got), (_, want)) in lazy.iter().zip(&eager) {
        assert_eq!(got, want);
    }
}

/// A product whose failure the call can see without using it.
#[test]
fn creation_time_errors_are_the_eager_ones() {
    let root = |c: &A| c.error().map(|error| error.root().to_string());
    // The shape errors and the context mismatch of `tests/all/matmul.rs`.
    for (lazy, eager) in [(context(LazyMode::Always), context(LazyMode::Never))] {
        for ctx in [&lazy, &eager] {
            let (a, b) = (A::zeros_on(ctx, [4, 3]).unwrap(), A::zeros_on(ctx, [5, 6]).unwrap());
            let c = a.matmul(&b);
            assert!(matches!(c.error().map(LumaError::root), Some(LumaError::ShapeMismatch { lhs: [4, 3, 1, 1], rhs: [5, 6, 1, 1] })), "{:?}", c.error());
            assert!(c.structured().is_none(), "the error is not a pending value");
            let cube = A::zeros_on(ctx, [4, 4, 2]).unwrap();
            let ok = A::zeros_on(ctx, [4, 4]).unwrap();
            for c in [cube.matmul(&ok), ok.matmul(&cube)] {
                assert!(matches!(c.error().map(LumaError::root), Some(LumaError::ShapeMismatch { .. })), "{:?}", c.error());
            }
            let bad = A::zeros_on(ctx, [5, 6]).unwrap().matmul(&A::zeros_on(ctx, [9, 9]).unwrap());
            let c = A::zeros_on(ctx, [4, 3]).unwrap().matmul(&bad);
            assert_eq!(c.error().map(ToString::to_string), bad.error().map(ToString::to_string), "a poisoned operand's error passes on unchanged");
        }
        let second = context(LazyMode::Always);
        let c = A::zeros_on(&lazy, [4, 3]).unwrap().matmul(&A::zeros_on(&second, [5, 4]).unwrap());
        assert!(matches!(c.error().map(LumaError::root), Some(LumaError::ContextMismatch)), "{:?}", c.error());
        // The batched product reports the same, before it is used.
        for ctx in [&lazy, &eager] {
            let (a, b) = (A::zeros_on(ctx, [4, 3, 2]).unwrap(), A::zeros_on(ctx, [5, 6, 2]).unwrap());
            assert!(matches!(a.matmul_batched(&b).error().map(LumaError::root), Some(LumaError::ShapeMismatch { .. })));
            let (a, b) = (A::zeros_on(ctx, [4, 3, 2]).unwrap(), A::zeros_on(ctx, [5, 4, 3]).unwrap());
            assert!(matches!(a.matmul_batched(&b).error().map(LumaError::root), Some(LumaError::ShapeMismatch { .. })), "the batches differ");
        }
        let c = A::zeros_on(&lazy, [4, 3, 2]).unwrap().matmul_batched(&A::zeros_on(&second, [5, 4, 2]).unwrap());
        assert!(matches!(c.error().map(LumaError::root), Some(LumaError::ContextMismatch)), "{:?}", c.error());
        // A poisoned operand of a batch passes its own error on (its shape is `[9, 6]`, the one a failed `matmul` of `[5, 6]` by `[9, 9]` keeps).
        for ctx in [&lazy, &eager] {
            let bad = A::zeros_on(ctx, [5, 6]).unwrap().matmul(&A::zeros_on(ctx, [9, 9]).unwrap());
            let c = A::zeros_on(ctx, [6, 3, 2]).unwrap().matmul_batched(&bad);
            assert!(root(&bad).is_some());
            assert_eq!(c.error().map(ToString::to_string), bad.error().map(ToString::to_string), "a poisoned operand of a batch");
        }
    }
}

#[test]
fn products_of_arrays_split_over_several_buffers_are_poisoned_at_the_call() {
    // Buffers of 4096 bytes: 1024 `f32`.
    let chunked = |mode| {
        let ctx = Context::with_tuning(options().chunk_bytes(4096), |tuning| tuning.lazy = mode).unwrap();
        let array = move |e: usize, f: usize| A::zeros_on(&ctx, [e, f]).unwrap();
        let chunked = |c: &A| match c.error().map(LumaError::root) {
            Some(LumaError::Chunked(_)) => Some(true),
            Some(_) => Some(false),
            None => None,
        };
        [
            chunked(&array(100, 100).matmul(&array(4, 100))),
            chunked(&array(100, 4).matmul(&array(100, 100))),
            chunked(&(&array(100, 100) * 2.0).matmul(&array(4, 100))),
            chunked(&array(4, 40).matmul(&array(40, 4))),
            chunked(&array(8, 16).matmul(&array(16, 8))),
            chunked(&(&array(8, 16) * 2.0).matmul(&array(16, 8))),
            chunked(&array(8, 16).matmul(&array(16, 9))),
        ]
    };
    let want = [Some(true), Some(true), Some(true), Some(true), None, None, Some(false)];
    assert_eq!(chunked(LazyMode::Never), want);
    assert_eq!(chunked(LazyMode::Always), want, "a product of arrays that need several buffers is poisoned before it is used");
}

#[test]
fn a_resource_error_surfaces_at_force_with_the_call_site() {
    // At most 2 workgroups per grid dimension: the tiles of a product this size have no plan.
    let ctx = Context::with_workgroup_limit(options(), 2, |tuning| tuning.lazy = LazyMode::Always).unwrap();
    let (k, m, n) = (256, 1024, 1024);
    let (a, b) = (A::zeros_on(&ctx, [k, m]).unwrap(), A::zeros_on(&ctx, [n, k]).unwrap());
    let line = line!() + 1;
    let c = a.matmul(&b);
    assert!(is_lazy(&c));
    assert!(c.error().is_none(), "a resource error is not known before the product is recorded");
    let error = c.to_vec().unwrap_err();
    assert!(matches!(error.root(), LumaError::KernelGridTooLarge { .. }), "{error}");
    let location = error.location().expect("the failure names the call");
    assert_eq!((location.file().ends_with("lazy_tests.rs"), location.line()), (true, line), "{error}");
    assert!(c.error().is_some(), "and the result is poisoned from then on");
    // The same product in the default mode fails at the call, with the same error.
    let eager = Context::with_workgroup_limit(options(), 2, |tuning| tuning.lazy = LazyMode::Never).unwrap();
    let (a, b) = (A::zeros_on(&eager, [k, m]).unwrap(), A::zeros_on(&eager, [n, k]).unwrap());
    let c = a.matmul(&b);
    assert!(matches!(c.error().map(LumaError::root), Some(LumaError::KernelGridTooLarge { .. })), "{:?}", c.error());
}

#[test]
fn a_failing_operand_surfaces_when_the_product_is_recorded() {
    let ctx = Context::with_tuning(options().check_finite(true), |tuning| tuning.lazy = LazyMode::Always).unwrap();
    let Operands { b, .. } = operands(&ctx);
    // `ln` of a negative number is NaN: with the finite check on, evaluating the operand fails.
    let bad = A::from_slice_on(&ctx, &[-1.0; K * M], [K, M]).unwrap().ln();
    let c = bad.matmul(&b);
    assert!(is_lazy(&c));
    assert!(c.to_vec().is_err());
    assert!(c.error().is_some());
}

#[test]
fn a_product_made_in_a_recording_is_eager() {
    let results = |mode| {
        let ctx = context(mode);
        let w = A::from_slice_on(&ctx, &ints(16, 3).iter().map(|v| v * 0.05).collect::<Vec<_>>(), [4, 4]).unwrap();
        let x = A::from_slice_on(&ctx, &[1.0; 16], [4, 4]).unwrap();
        let recorded = std::cell::Cell::new(0);
        let out = x.iterate_until(30, 1e-6, |x| {
            let c = x.matmul(&w);
            if ctx.recording_here() {
                recorded.set(recorded.get() + 1);
                assert!(c.structured().is_none(), "a launch recorded for a loop body is part of the loop");
            }
            &c * 0.5
        });
        (out.to_vec().unwrap(), recorded.get())
    };
    let (lazy, eager) = (results(LazyMode::Always), results(LazyMode::Never));
    assert_eq!(lazy.0, eager.0);
}

#[test]
fn a_product_before_a_loop_is_forced_when_the_loop_starts() {
    let results = |mode| {
        let ctx = context(mode);
        let Operands { a, b, .. } = operands(&ctx);
        let c = a.matmul(&b);
        let x = A::from_slice_on(&ctx, &[1.0; N * M], [N, M]).unwrap();
        let out = x.iterate_until(20, 1e-6, |x| {
            if ctx.recording_here() {
                assert!(!is_lazy(&c), "a product made before the loop is recorded before the body is");
            }
            &(x * 0.5) + &(&c * 0.01)
        });
        (out.to_vec().unwrap(), c.to_vec().unwrap())
    };
    assert_eq!(results(LazyMode::Always), results(LazyMode::Never));
}

#[test]
fn a_lazy_product_can_be_forced_from_another_thread() {
    let ctx = context(LazyMode::Always);
    let Operands { a, b, ha, hb } = operands(&ctx);
    let c = a.matmul(&b);
    assert!(is_lazy(&c));
    let got = std::thread::scope(|scope| scope.spawn(|| c.to_vec().unwrap()).join().unwrap());
    assert_eq!(got, host(&ha, &hb, K, M, N));
    // Made on another thread, used on this one.
    let c = std::thread::scope(|scope| scope.spawn(|| a.matmul(&b)).join().unwrap());
    assert_eq!(c.to_vec().unwrap(), host(&ha, &hb, K, M, N));
}

#[test]
fn flushing_and_syncing_start_the_lazy_products_of_the_thread() {
    let ctx = context(LazyMode::Always);
    let Operands { a, b, .. } = operands(&ctx);
    let (c1, c2) = (a.matmul(&b), (&a * 2.0).matmul(&b));
    assert!(is_lazy(&c1) && is_lazy(&c2));
    ctx.flush().unwrap();
    assert!(!is_lazy(&c1) && !is_lazy(&c2), "`flush` records what the thread made");
    let c3 = a.matmul(&b);
    ctx.sync().unwrap();
    assert!(!is_lazy(&c3), "`sync` too");
}

#[test]
fn ctx_profile_counts_the_launches_of_a_lazy_product_it_never_forced() {
    let ctx = context(LazyMode::Always);
    if !ctx.profiling_supported() {
        return;
    }
    let (k, m, n) = (128, 128, 128);
    let a = A::from_slice_on(&ctx, &ints(k * m, 1), [k, m]).unwrap();
    let b = A::from_slice_on(&ctx, &ints(n * k, 2), [n, k]).unwrap();
    a.eval().unwrap();
    b.eval().unwrap();
    ctx.flush().unwrap();
    let (c, profile) = ctx.profile(|| a.matmul(&b)).unwrap();
    assert!(profile.launches() >= 1, "{profile}");
    assert!(!is_lazy(&c));
    let eager = context(LazyMode::Never);
    let a = A::from_slice_on(&eager, &ints(k * m, 1), [k, m]).unwrap();
    let b = A::from_slice_on(&eager, &ints(n * k, 2), [n, k]).unwrap();
    a.eval().unwrap();
    b.eval().unwrap();
    eager.flush().unwrap();
    let (_, want) = eager.profile(|| a.matmul(&b)).unwrap();
    assert_eq!(profile.launches(), want.launches(), "the same launches, counted in the region that made the product");
}

#[test]
fn a_lazy_product_in_a_host_evaluated_expression() {
    let ctx = context(LazyMode::Always);
    if !ctx.can_run_on_host() {
        return;
    }
    let Operands { a, b, ha, hb } = operands(&ctx);
    ctx.record_decisions(true);
    let c = a.matmul(&b);
    let scaled = &c * 2.0;
    assert!(is_lazy(&c));
    let launches = ctx.jit().stats().elementwise_launches;
    let want: Vec<f32> = host(&ha, &hb, K, M, N).iter().map(|v| v * 2.0).collect();
    assert_eq!(scaled.to_vec().unwrap(), want);
    assert_eq!(ctx.jit().stats().elementwise_launches, launches, "no launch: the host computed the expression");
    let chosen = plans(&ctx);
    assert_eq!(chosen.len(), 1, "{chosen:?}");
    assert_eq!(chosen[0].chosen, "host");
}

/// `[K, M, B]` by `[N, K, B]` (a side without the batch is shared) as `B` products on the host.
fn host_batched(a: &[f32], b: &[f32], (k, m, n, batch): (usize, usize, usize, usize), (lhs_batched, rhs_batched): (bool, bool)) -> Vec<f32> {
    (0..batch)
        .flat_map(|i| {
            let a = if lhs_batched { &a[i * k * m..(i + 1) * k * m] } else { a };
            let b = if rhs_batched { &b[i * n * k..(i + 1) * n * k] } else { b };
            host(a, b, k, m, n)
        })
        .collect()
}

#[test]
fn a_lazy_batched_product_matches_the_eager_one() {
    let (lazy, eager) = (context(LazyMode::Always), context(LazyMode::Never));
    let (k, m, n, batch) = (K, M, N, 6);
    for (lhs_batched, rhs_batched) in [(true, true), (true, false), (false, true)] {
        let (ha, hb) = (ints(k * m * if lhs_batched { batch } else { 1 }, 1), ints(n * k * if rhs_batched { batch } else { 1 }, 2));
        let want = host_batched(&ha, &hb, (k, m, n, batch), (lhs_batched, rhs_batched));
        let run = |ctx: &Context| {
            let a = A::from_slice_on(ctx, &ha, [k, m, if lhs_batched { batch } else { 1 }]).unwrap();
            let b = A::from_slice_on(ctx, &hb, [n, k, if rhs_batched { batch } else { 1 }]).unwrap();
            ctx.record_decisions(true);
            let c = a.matmul_batched(&b);
            let lazy_at_call = is_lazy(&c);
            assert_eq!(c.shape().dims(), [n, m, batch, 1]);
            (lazy_at_call, c.to_vec().unwrap())
        };
        let (was_lazy, got) = run(&lazy);
        assert!(was_lazy, "a batch of one launch is a lazy value ({lhs_batched}, {rhs_batched})");
        assert_eq!(got, want, "lazy ({lhs_batched}, {rhs_batched})");
        let (was_lazy, got) = run(&eager);
        assert!(!was_lazy);
        assert_eq!(got, want, "eager ({lhs_batched}, {rhs_batched})");
    }
    // Batches along both axes, and a write into an operand after the call.
    let (ha, hb) = (ints(k * m * 6, 1), ints(n * k * 6, 2));
    let mut a = A::from_slice_on(&lazy, &ha, [k, m, 3, 2]).unwrap();
    let b = A::from_slice_on(&lazy, &hb, [n, k, 3, 2]).unwrap();
    let c = a.matmul_batched(&b);
    assert!(is_lazy(&c));
    a.write(&vec![50.0; k * m * 6]).unwrap();
    assert_eq!(c.shape().dims(), [n, m, 3, 2]);
    assert_eq!(c.to_vec().unwrap(), host_batched(&ha, &hb, (k, m, n, 6), (true, true)));
}

#[test]
fn a_looped_batch_is_decided_at_creation() {
    // 129 rows: past the edge the batched kernel takes (a backend with native products also loops when there are few products and they
    // are big, so the size is chosen for the shader backends and the check is what the batch turned out to be).
    let (k, m, n, batch) = (16, 130, 16, 2);
    let lazy = context(LazyMode::Always);
    let (ha, hb) = (ints(k * m * batch, 1), ints(n * k * batch, 2));
    let a = A::from_slice_on(&lazy, &ha, [k, m, batch]).unwrap();
    let b = A::from_slice_on(&lazy, &hb, [n, k, batch]).unwrap();
    lazy.record_decisions(true);
    let c = a.matmul_batched(&b);
    assert!(plans(&lazy).is_empty(), "nothing is recorded at the call, whichever way the batch goes");
    match c.structured() {
        // A backend that took it in one launch.
        Some(_) => assert!(is_lazy(&c)),
        // The loop: a lazy product per matrix, joined by a view.
        None => assert!(c.error().is_none()),
    }
    assert_eq!(c.to_vec().unwrap(), host_batched(&ha, &hb, (k, m, n, batch), (true, true)));
    let chosen = plans(&lazy).len();
    assert!(chosen == 0 || chosen == batch, "one selection per matrix of a loop, none for a launch of the batch: {chosen}");
}

#[test]
fn in_site_mode_a_site_is_lazy_once_it_has_something_to_attach_to() {
    #[track_caller]
    fn product_here(a: &A, b: &A) -> (A, &'static Location<'static>) {
        (a.matmul(b), Location::caller())
    }
    let ctx = Context::with_tuning(options(), |tuning| tuning.lazy = LazyMode::Site).unwrap();
    let Operands { a, b, ha, hb } = operands(&ctx);
    let mut was_lazy = Vec::new();
    for round in 0..20 {
        let (c, location) = product_here(&a, &b);
        was_lazy.push(is_lazy(&c));
        if round == 1 {
            // What W3-U4 will write when an epilogue fuses at this site.
            ctx.jit().sites.set_attached(SiteKey::of(location), 1, 0);
        }
        assert_eq!(c.to_vec().unwrap(), host(&ha, &hb, K, M, N));
    }
    assert_eq!(&was_lazy[..2], [false, false], "a site that has not absorbed anything launches at the call");
    assert!(was_lazy[2..18].iter().all(|&lazy| lazy), "and one that has is lazy: {was_lazy:?}");
    assert!(!was_lazy[18], "until it was forced alone sixteen times: {was_lazy:?}");
}

#[test]
fn in_site_mode_the_probe_makes_the_first_call_and_every_nth_lazy() {
    let ctx = Context::with_tuning(options(), |tuning| {
        tuning.lazy = LazyMode::Site;
        tuning.lazy_probe_every = 4;
    })
    .unwrap();
    let Operands { a, b, .. } = operands(&ctx);
    let mut was_lazy = Vec::new();
    for _ in 0..9 {
        // One call site, nine calls.
        let c = a.matmul(&b);
        was_lazy.push(is_lazy(&c));
        c.eval().unwrap();
    }
    assert_eq!(was_lazy, [true, false, false, false, true, false, false, false, true]);
}

#[test]
fn a_narrow_n_product_is_never_lazy() {
    // 10 classes over 256 rows and 64 features: the padded strategy of the backend's own kernels takes it where there are any.
    let ctx = context(LazyMode::Always);
    let (k, m, n) = (64, 256, 10);
    let a = A::from_slice_on(&ctx, &ints(k * m, 1), [k, m]).unwrap();
    let b = A::from_slice_on(&ctx, &ints(n * k, 2), [n, k]).unwrap();
    let c = a.matmul(&b);
    let narrow = super::contraction::narrow_n_fits(&super::contraction::ContractionView { lhs: &a, rhs: &b, attrs: super::contraction::ContractionAttrs { tf32: super::contraction::Tf32Request::Default }, location: Location::caller() }, &ctx);
    assert_eq!(is_lazy(&c), !narrow, "lazy unless the padded strategy takes it");
    assert_eq!(c.to_vec().unwrap(), host(&ints(k * m, 1), &ints(n * k, 2), k, m, n));
}

#[test]
fn a_long_chain_of_products_neither_recurses_without_bound_nor_changes_the_result() {
    let ctx = context(LazyMode::Always);
    let ha = ints(9, 4);
    let a = A::from_slice_on(&ctx, &ha, [3, 3]).unwrap();
    let identity = A::from_slice_on(&ctx, &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], [3, 3]).unwrap();
    // Unused products pile up without being recorded, and dropping the end of the chain drops them one after another.
    let mut c = a.clone();
    for _ in 0..5000 {
        c = c.matmul(&identity);
    }
    assert_eq!(c.to_vec().unwrap(), ha);
    let mut dropped = a.clone();
    for _ in 0..5000 {
        dropped = dropped.matmul(&identity);
    }
    drop(dropped);
    // Below the limit the whole chain is one lazy value.
    let mut short = a.clone();
    for _ in 0..10 {
        short = short.matmul(&identity);
    }
    assert!(is_lazy(&short));
    assert_eq!(short.to_vec().unwrap(), ha);
}

#[test]
fn an_expression_waiting_to_be_evaluated_that_reads_the_product_does_not_deadlock_its_recording() {
    // `y` is requested but postponed (the GPU is busy); recording the product it reads evaluates the requested expressions (the
    // backends' own products do, to compute their operands together), which reaches the product that is being recorded.
    for mode in BOTH {
        let ctx = context(mode);
        let Operands { a, b, ha, hb } = operands(&ctx);
        ctx.set_assume_busy(true);
        let doubled = &a * 2.0;
        let c = doubled.matmul(&b);
        let y = &c * 3.0;
        y.eval().unwrap();
        let product = host(&ha.iter().map(|v| v * 2.0).collect::<Vec<_>>(), &hb, K, M, N);
        assert_eq!(c.to_vec().unwrap(), product, "{mode:?}");
        assert_eq!(y.to_vec().unwrap(), product.iter().map(|v| v * 3.0).collect::<Vec<_>>(), "{mode:?}");
        ctx.set_assume_busy(false);
    }
}
