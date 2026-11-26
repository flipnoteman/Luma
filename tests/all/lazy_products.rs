//! Matrix products at call sites that are lazy (`LUMA_LAZY=always`, W2-LAZY) and eager ones (the default): what a user can see of the
//! difference is nothing. These tests say it through the public API and hold in every mode, so the suite runs them both ways (the
//! internal tests of `jit::structured::lazy_tests` choose the mode of their own contexts and look at the values behind the arrays).
//!
//! Operands hold small integers, so every backend's product (the tensor cores' TF32 included) is exact and the results compare bit for bit.

use luma::*;

type A = Array<f32>;

const K: usize = 5;
const M: usize = 4;
const N: usize = 3;

fn ints(len: usize, seed: usize) -> Vec<f32> {
    (0..len).map(|i| ((i * 7 + seed * 3 + i / 3) % 9) as f32 - 4.0).collect()
}

/// `C[n + N m] = Σ_k A[k + K m] B[n + N k]`.
fn host(a: &[f32], b: &[f32], k: usize, m: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; n * m];
    for i in 0..m {
        for j in 0..n {
            c[j + n * i] = (0..k).map(|l| a[l + k * i] * b[j + n * l]).sum();
        }
    }
    c
}

fn ctx() -> Context {
    Context::with_options(ContextOptions::from_env()).unwrap()
}

struct Case {
    ctx: Context,
    ha: Vec<f32>,
    hb: Vec<f32>,
}

impl Case {
    fn new() -> Case {
        Case { ctx: ctx(), ha: ints(K * M, 1), hb: ints(N * K, 2) }
    }

    fn a(&self) -> A {
        A::from_slice_on(&self.ctx, &self.ha, [K, M]).unwrap()
    }

    fn b(&self) -> A {
        A::from_slice_on(&self.ctx, &self.hb, [N, K]).unwrap()
    }

    fn product(&self) -> Vec<f32> {
        host(&self.ha, &self.hb, K, M, N)
    }
}

#[test]
fn a_product_is_the_same_whenever_it_is_launched() {
    let case = Case::new();
    let (a, b) = (case.a(), case.b());
    let c = a.matmul(&b);
    assert_eq!(c.shape().dims(), [N, M, 1, 1]);
    assert_eq!(c.to_vec().unwrap(), case.product());
    // Used once more it is the same data, whichever launch made it.
    assert_eq!(c.to_vec().unwrap(), case.product());
    assert_eq!(a.matmul_f32(&b).to_vec().unwrap(), case.product());
    assert_eq!(a.matmul_tf32(&b).to_vec().unwrap(), case.product());
}

#[test]
fn a_write_after_a_product_does_not_change_it() {
    // An operand that is a buffer.
    let case = Case::new();
    let (mut a, b) = (case.a(), case.b());
    let c = a.matmul(&b);
    a.write(&[100.0; K * M]).unwrap();
    assert_eq!(c.to_vec().unwrap(), case.product(), "lhs written");
    let (a, mut b) = (case.a(), case.b());
    let c = a.matmul(&b);
    b.write(&[-7.0; N * K]).unwrap();
    assert_eq!(c.to_vec().unwrap(), case.product(), "rhs written");
    // An operand that is a pending expression, written after the product was made.
    let (x, b) = (case.a(), case.b());
    let mut doubled = &x * 2.0;
    let c = doubled.matmul(&b);
    doubled.write(&[100.0; K * M]).unwrap();
    let old: Vec<f32> = case.ha.iter().map(|v| v * 2.0).collect();
    assert_eq!(c.to_vec().unwrap(), host(&old, &case.hb, K, M, N), "pending lhs written");
    // What the pending operand reads is written.
    let (mut x, b) = (case.a(), case.b());
    let doubled = &x * 2.0;
    let c = doubled.matmul(&b);
    x.write(&[100.0; K * M]).unwrap();
    assert_eq!(c.to_vec().unwrap(), host(&old, &case.hb, K, M, N), "the root of the pending operand written");
    // Both operands pending and both written.
    let (x, y) = (case.a(), case.b());
    let (mut p, mut q) = (&x + 1.0, &y - 1.0);
    let c = p.matmul(&q);
    p.write(&[100.0; K * M]).unwrap();
    q.write(&[-7.0; N * K]).unwrap();
    let (old_a, old_b): (Vec<f32>, Vec<f32>) = (case.ha.iter().map(|v| v + 1.0).collect(), case.hb.iter().map(|v| v - 1.0).collect());
    assert_eq!(c.to_vec().unwrap(), host(&old_a, &old_b, K, M, N), "both pending operands written");
}

#[test]
fn products_chain_and_clones_share_one() {
    let case = Case::new();
    let (a, b) = (case.a(), case.b());
    let hd = ints(N * N, 5);
    let d = A::from_slice_on(&case.ctx, &hd, [N, N]).unwrap();
    let chained = a.matmul(&b).matmul(&d);
    assert_eq!(chained.to_vec().unwrap(), host(&case.product(), &hd, N, M, N));
    let c = a.matmul(&b);
    let sum = &c + &c.clone();
    assert_eq!(sum.to_vec().unwrap(), case.product().iter().map(|v| v * 2.0).collect::<Vec<_>>());
    // A write into the middle of a chain does not reach what was made from it.
    let mut c1 = a.matmul(&b);
    let c2 = c1.matmul(&d);
    c1.write(&[1000.0; N * M]).unwrap();
    assert_eq!(c2.to_vec().unwrap(), host(&case.product(), &hd, N, M, N));
}

#[test]
fn a_product_of_expressions_and_views_matches_the_host() {
    let case = Case::new();
    let (a, b) = (case.a(), case.b());
    let lazy = (&a * 2.0).matmul(&(&b + 1.0));
    let (da, db): (Vec<f32>, Vec<f32>) = (case.ha.iter().map(|v| v * 2.0).collect(), case.hb.iter().map(|v| v + 1.0).collect());
    assert_eq!(lazy.to_vec().unwrap(), host(&da, &db, K, M, N));
    // The same operands stored the other way round: `at` is `[M, K]`, `bt` is `[K, N]`.
    let (hat, hbt): (Vec<f32>, Vec<f32>) = ((0..K * M).map(|i| case.ha[(i % M) * K + i / M]).collect(), (0..N * K).map(|i| case.hb[(i % K) * N + i / K]).collect());
    let (at, bt) = (A::from_slice_on(&case.ctx, &hat, [M, K]).unwrap(), A::from_slice_on(&case.ctx, &hbt, [K, N]).unwrap());
    assert_eq!(a.matmul_nt(&bt).to_vec().unwrap(), case.product(), "nt");
    assert_eq!(at.matmul_tn(&b).to_vec().unwrap(), case.product(), "tn");
    assert_eq!(at.matmul_tt(&bt).to_vec().unwrap(), case.product(), "tt");
    // An elementwise step reads the product through the expression it is part of.
    let relu = a.matmul(&b).maximum(0.0);
    assert_eq!(relu.to_vec().unwrap(), case.product().iter().map(|v| v.max(0.0)).collect::<Vec<_>>());
}

#[test]
fn shape_context_and_poison_errors_are_known_at_the_call() {
    let ctx = ctx();
    let (a, b) = (A::zeros_on(&ctx, [4, 3]).unwrap(), A::zeros_on(&ctx, [5, 6]).unwrap());
    let c = a.matmul(&b);
    assert!(matches!(c.error().map(LumaError::root), Some(LumaError::ShapeMismatch { lhs: [4, 3, 1, 1], rhs: [5, 6, 1, 1] })), "{:?}", c.error());
    let bad = A::zeros_on(&ctx, [5, 6]).unwrap().matmul(&A::zeros_on(&ctx, [9, 9]).unwrap());
    let c = a.matmul(&bad);
    assert_eq!(c.error().map(ToString::to_string), bad.error().map(ToString::to_string));
    let second = Context::with_options(ContextOptions::from_env()).unwrap();
    let c = a.matmul(&A::zeros_on(&second, [5, 4]).unwrap());
    assert!(matches!(c.error().map(LumaError::root), Some(LumaError::ContextMismatch)), "{:?}", c.error());
    let cube = A::zeros_on(&ctx, [4, 3, 2]).unwrap();
    assert!(matches!(cube.matmul_batched(&A::zeros_on(&ctx, [5, 6, 2]).unwrap()).error().map(LumaError::root), Some(LumaError::ShapeMismatch { .. })));
}

#[test]
fn arrays_that_need_several_buffers_are_known_at_the_call() {
    let ctx = Context::with_options(ContextOptions::from_env().chunk_bytes(4096)).unwrap();
    let array = |e: usize, f: usize| A::zeros_on(&ctx, [e, f]).unwrap();
    let chunked = |c: &A| matches!(c.error().map(LumaError::root), Some(LumaError::Chunked(_)));
    assert!(chunked(&array(100, 100).matmul(&array(4, 100))));
    assert!(chunked(&array(100, 4).matmul(&array(100, 100))));
    assert!(chunked(&(&array(100, 100) * 2.0).matmul(&array(4, 100))));
    assert!(chunked(&array(4, 40).matmul(&array(40, 4))));
    assert!(!chunked(&array(8, 16).matmul(&array(16, 8))));
}

#[test]
fn a_product_can_be_made_on_one_thread_and_used_on_another() {
    let case = Case::new();
    let (a, b) = (case.a(), case.b());
    let c = a.matmul(&b);
    let got = std::thread::scope(|scope| scope.spawn(|| c.to_vec().unwrap()).join().unwrap());
    assert_eq!(got, case.product());
    let c = std::thread::scope(|scope| scope.spawn(|| a.matmul(&b)).join().unwrap());
    assert_eq!(c.to_vec().unwrap(), case.product());
}

#[test]
fn evaluating_and_flushing_record_the_products_made_before() {
    let case = Case::new();
    let (a, b) = (case.a(), case.b());
    let c1 = a.matmul(&b);
    let c2 = (&a * 2.0).matmul(&b);
    let c3 = a.matmul(&b);
    c1.eval().unwrap();
    case.ctx.flush().unwrap();
    case.ctx.sync().unwrap();
    assert_eq!(c1.to_vec().unwrap(), case.product());
    let doubled: Vec<f32> = case.ha.iter().map(|v| v * 2.0).collect();
    assert_eq!(c2.to_vec().unwrap(), host(&doubled, &case.hb, K, M, N));
    assert_eq!(c3.to_vec().unwrap(), case.product());
    eval_all(&[&c1, &c2, &c3]).unwrap();
}

#[test]
fn a_profiled_region_counts_the_launches_of_a_product_it_never_used() {
    let ctx = ctx();
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
    c.check().unwrap();
}

#[test]
fn a_batched_product_matches_the_host() {
    let case = Case::new();
    let batch = 6;
    for (lhs_batched, rhs_batched) in [(true, true), (true, false), (false, true)] {
        let ha = ints(K * M * if lhs_batched { batch } else { 1 }, 1);
        let hb = ints(N * K * if rhs_batched { batch } else { 1 }, 2);
        let a = A::from_slice_on(&case.ctx, &ha, [K, M, if lhs_batched { batch } else { 1 }]).unwrap();
        let b = A::from_slice_on(&case.ctx, &hb, [N, K, if rhs_batched { batch } else { 1 }]).unwrap();
        let want: Vec<f32> = (0..batch)
            .flat_map(|i| {
                let a = if lhs_batched { &ha[i * K * M..(i + 1) * K * M] } else { &ha[..] };
                let b = if rhs_batched { &hb[i * N * K..(i + 1) * N * K] } else { &hb[..] };
                host(a, b, K, M, N)
            })
            .collect();
        assert_eq!(a.matmul_batched(&b).to_vec().unwrap(), want, "({lhs_batched}, {rhs_batched})");
    }
    // A batch that is a loop of products (the edge is past what the batched kernel takes).
    let (k, m, n) = (16, 130, 16);
    let (ha, hb) = (ints(k * m * 2, 1), ints(n * k * 2, 2));
    let a = A::from_slice_on(&case.ctx, &ha, [k, m, 2]).unwrap();
    let b = A::from_slice_on(&case.ctx, &hb, [n, k, 2]).unwrap();
    let want: Vec<f32> = (0..2).flat_map(|i| host(&ha[i * k * m..(i + 1) * k * m], &hb[i * n * k..(i + 1) * n * k], k, m, n)).collect();
    assert_eq!(a.matmul_batched(&b).to_vec().unwrap(), want);
}
