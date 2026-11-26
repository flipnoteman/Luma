//! Custom ops with strategies (`luma::ext`).

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use luma::ext::{ArgInfo, Bind, NumericsClass, OutShape, Strategy};
use luma::*;

const SCALE: &str = r#"
struct P { k: f32 }
@group(1) @binding(0) var<storage, read> x: array<f32>;
@group(1) @binding(1) var<storage, read_write> y: array<f32>;
@group(1) @binding(2) var<uniform> p: P;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= luma.size.x) { return; }
    y[gid.x] = p.k * x[gid.x];
}
"#;

#[derive(encase::ShaderType)]
struct Scale {
    k: f32,
}

fn ctx() -> Context {
    default_context().expect("Could not create Context.")
}

fn array(ctx: &Context, n: usize) -> Array<f32> {
    Array::from_slice_on(ctx, &(0..n).map(|i| i as f32).collect::<Vec<_>>(), n).unwrap()
}

/// A strategy that scales its input by the `k` of its params.
fn scaler(name: &'static str) -> Strategy {
    Strategy::new(name, |lower| {
        let kernel = lower.kernel(SCALE).build()?;
        let n = lower.inputs()[0].shape.len() as u32;
        let params = lower.params().to_vec();
        lower.launch(&kernel, [n, 1, 1], &[Bind::Input(0), Bind::Output(0), Bind::Uniform(params)])
    })
}

fn scale_op(ctx: &Context, name: &str, strategies: Vec<Strategy>) -> Result<ext::Op> {
    let mut builder = ctx.custom_op(name).input::<f32>().output::<f32>(OutShape::Same(0)).params(4);
    for strategy in strategies {
        builder = builder.strategy(strategy);
    }
    builder.build()
}

fn k(value: f32) -> Vec<u8> {
    ext::params_bytes(&Scale { k: value }).unwrap()
}

fn doubled(n: usize) -> Vec<f32> {
    (0..n).map(|i| 2.0 * i as f32).collect()
}

#[test]
fn a_custom_op_runs_its_only_strategy_and_returns_a_valid_array() {
    let ctx = ctx();
    let op = scale_op(&ctx, "t_only", vec![scaler("plain")]).unwrap();
    assert_eq!(op.name(), "t_only");
    let x = array(&ctx, 1000);
    let y = op.apply1::<f32>(&[&x], &k(2.0)).unwrap();
    assert_eq!(y.shape(), x.shape());
    assert_eq!(y.to_vec().unwrap(), doubled(1000));
    // `apply` returns the outputs type-erased.
    let outputs = op.apply(&[&x], &k(3.0)).unwrap();
    assert_eq!(outputs.len(), 1);
    assert_eq!(outputs[0].elem(), ElemKind::F32);
    let z = outputs.into_iter().next().unwrap().take::<f32>().unwrap();
    assert_eq!(z.to_vec().unwrap()[10], 30.0);
}

#[test]
fn apply_checks_context_count_and_element_kinds() {
    let ctx = ctx();
    let op = scale_op(&ctx, "t_checks", vec![scaler("plain")]).unwrap();
    let x = array(&ctx, 8);
    assert!(matches!(op.apply(&[], &k(1.0)), Err(LumaError::InvalidShape(_))));
    assert!(matches!(op.apply(&[&x, &x], &k(1.0)), Err(LumaError::InvalidShape(_))));
    assert!(matches!(op.apply(&[&x], &[0u8; 3]), Err(LumaError::InvalidShape(_))));
    let wrong = Array::<u32>::zeros_on(&ctx, 8).unwrap();
    assert!(matches!(op.apply(&[&wrong], &k(1.0)), Err(LumaError::InvalidShape(_))));
    let other = Context::with_options(ContextOptions::from_env()).unwrap();
    let foreign = array(&other, 8);
    assert!(matches!(op.apply(&[&foreign], &k(1.0)), Err(LumaError::ContextMismatch)));
    // An inherited error comes back as it is.
    let poisoned = x.reshape([3, 3]);
    assert!(poisoned.error().is_some());
    assert!(op.apply(&[&poisoned], &k(1.0)).is_err());
}

#[test]
fn a_lazy_input_is_forced_first() {
    let ctx = ctx();
    let op = scale_op(&ctx, "t_lazy", vec![scaler("plain")]).unwrap();
    let x = array(&ctx, 500);
    let lazy = &x * 2.0 + 1.0;
    let y = op.apply1::<f32>(&[&lazy], &k(2.0)).unwrap();
    let want: Vec<f32> = (0..500).map(|i| 2.0 * (2.0 * i as f32 + 1.0)).collect();
    assert_eq!(y.to_vec().unwrap(), want);
}

#[test]
fn a_write_after_apply_does_not_change_the_result() {
    let ctx = ctx();
    let op = scale_op(&ctx, "t_order", vec![scaler("plain")]).unwrap();
    let mut x = array(&ctx, 256);
    let y = op.apply1::<f32>(&[&x], &k(2.0)).unwrap();
    x.slice_mut(0, ..).assign(-1.0).unwrap();
    assert_eq!(y.to_vec().unwrap(), doubled(256));
    assert_eq!(x.to_vec().unwrap(), vec![-1.0; 256]);
}

fn counting(name: &'static str, calls: &Arc<AtomicUsize>) -> Strategy {
    let calls = Arc::clone(calls);
    Strategy::new(name, move |lower| {
        calls.fetch_add(1, Ordering::SeqCst);
        let kernel = lower.kernel(SCALE).build()?;
        let params = lower.params().to_vec();
        lower.launch(&kernel, [1u32, 1, 1], &[Bind::Input(0), Bind::Output(0), Bind::Uniform(params)])
    })
}

/// Which strategy ran: it writes its own `k` factor.
fn tagged(name: &'static str, factor: f32) -> Strategy {
    Strategy::new(name, move |lower| {
        let kernel = lower.kernel(SCALE).build()?;
        let params = ext::params_bytes(&Scale { k: factor })?;
        let n = lower.inputs()[0].shape.len() as u32;
        lower.launch(&kernel, [n, 1, 1], &[Bind::Input(0), Bind::Output(0), Bind::Uniform(params)])
    })
}

fn ran(op: &ext::Op, ctx: &Context) -> f32 {
    let x = Array::from_slice_on(ctx, &[1.0f32], 1).unwrap();
    op.apply1::<f32>(&[&x], &k(0.0)).unwrap().to_vec().unwrap()[0]
}

#[test]
fn needs_and_fits_filter_the_candidates() {
    let ctx = ctx();
    let op = scale_op(
        &ctx,
        "t_filter",
        vec![
            tagged("needs_f64", 1.0).needs(Features::SHADER_F64.union(Features::EXPERIMENTAL_COOPERATIVE_MATRIX).union(Features::SUBGROUP).union(Features::SHADER_F16)),
            tagged("never", 2.0).fits(|_| false),
            tagged("reassociates", 3.0).numerics(NumericsClass::Reassociating),
            tagged("small", 4.0).fits(|fit| fit.inputs()[0].shape.len() == 1),
            tagged("anything", 5.0),
        ],
    )
    .unwrap();
    let features = ctx.features();
    let all = Features::SHADER_F64.union(Features::EXPERIMENTAL_COOPERATIVE_MATRIX).union(Features::SUBGROUP).union(Features::SHADER_F16);
    let first = if features.contains(all) { 1.0 } else { 4.0 };
    assert_eq!(ran(&op, &ctx), first);
    // `small` fits one element only, `never` and `reassociates` never apply: `anything`, or the first where the device has every feature.
    let big = Array::from_slice_on(&ctx, &[1.0f32, 1.0], 2).unwrap();
    let want = if features.contains(all) { 1.0 } else { 5.0 };
    assert_eq!(op.apply1::<f32>(&[&big], &k(0.0)).unwrap().to_vec().unwrap(), vec![want, want]);
}

#[test]
fn the_cheapest_fitting_strategy_wins_and_cost_is_not_called_for_a_single_candidate() {
    let ctx = ctx();
    let op = scale_op(
        &ctx,
        "t_cost",
        vec![tagged("slow", 1.0).cost(|_| 900.0), tagged("fast", 2.0).cost(|_| 100.0), tagged("middle", 3.0).cost(|_| 500.0)],
    )
    .unwrap();
    assert_eq!(ran(&op, &ctx), 2.0);

    let panics = scale_op(&ctx, "t_cost_single", vec![tagged("only", 7.0).cost(|_| panic!("cost must not be called")), tagged("nope", 1.0).fits(|_| false).cost(|_| 1.0)]).unwrap();
    assert_eq!(ran(&panics, &ctx), 7.0);
}

#[test]
fn registration_order_decides_without_costs() {
    let ctx = ctx();
    let op = scale_op(&ctx, "t_order_of", vec![tagged("b", 1.0), tagged("a", 2.0).cost(|_| 1.0), tagged("c", 3.0)]).unwrap();
    // `a` has a cost but `b` has none, so no costs are compared.
    assert_eq!(ran(&op, &ctx), 1.0);
}

#[test]
fn no_fitting_strategy_is_no_strategy_with_reasons() {
    let ctx = ctx();
    let op = scale_op(&ctx, "t_none", vec![tagged("a", 1.0).fits(|_| false), tagged("b", 2.0).numerics(NumericsClass::Reassociating)]).unwrap();
    let x = array(&ctx, 4);
    let Err(LumaError::NoStrategy { op: name, reasons }) = op.apply(&[&x], &k(0.0)) else { panic!("expected NoStrategy") };
    assert_eq!(name, "t_none");
    assert_eq!(reasons, vec![("a".to_owned(), "does not fit".to_owned()), ("b".to_owned(), "numerics policy".to_owned())]);
}

#[test]
fn a_strategy_that_declines_before_launching_falls_through() {
    let ctx = ctx();
    let declined = Arc::new(AtomicUsize::new(0));
    let counter = Arc::clone(&declined);
    let decliner = Strategy::new("decliner", move |_| {
        counter.fetch_add(1, Ordering::SeqCst);
        Err(LumaError::Declined)
    });
    let op = scale_op(&ctx, "t_decline", vec![decliner, tagged("fallback", 6.0)]).unwrap();
    assert_eq!(ran(&op, &ctx), 6.0);
    assert_eq!(declined.load(Ordering::SeqCst), 1);

    // Declining after a launch is an error.
    let late = Strategy::new("late", |lower| {
        let kernel = lower.kernel(SCALE).build()?;
        lower.launch(&kernel, [1u32, 1, 1], &[Bind::Input(0), Bind::Output(0), Bind::Uniform(lower.params().to_vec())])?;
        Err(LumaError::Declined)
    });
    let op = scale_op(&ctx, "t_decline_late", vec![late, tagged("fallback", 6.0)]).unwrap();
    let x = array(&ctx, 1);
    assert!(matches!(op.apply(&[&x], &k(1.0)), Err(LumaError::Extension { .. })));

    // Everything declines.
    let op = scale_op(&ctx, "t_decline_all", vec![Strategy::new("d", |_| Err(LumaError::Declined))]).unwrap();
    assert!(matches!(op.apply(&[&x], &k(1.0)), Err(LumaError::NoStrategy { .. })));
}

#[test]
fn luma_strategy_forces_an_ext_strategy() {
    let ctx = Context::with_options(ContextOptions::from_env().strategy("ext:t_forced/second,ext:t_unknown/nothing,ext:t_nofit/never")).unwrap();
    let op = scale_op(&ctx, "t_forced", vec![tagged("first", 1.0), tagged("second", 2.0)]).unwrap();
    assert_eq!(ran(&op, &ctx), 2.0);
    // A forced strategy that does not exist or does not fit gives the default selection.
    let op = scale_op(&ctx, "t_unknown", vec![tagged("first", 1.0), tagged("second", 2.0)]).unwrap();
    assert_eq!(ran(&op, &ctx), 1.0);
    let op = scale_op(&ctx, "t_nofit", vec![tagged("first", 1.0), tagged("never", 2.0).fits(|_| false)]).unwrap();
    assert_eq!(ran(&op, &ctx), 1.0);
    // The other contexts are not forced.
    let plain = Context::with_options(ContextOptions::from_env().strategy("reduce:two-pass")).unwrap();
    let op = scale_op(&plain, "t_forced", vec![tagged("first", 1.0), tagged("second", 2.0)]).unwrap();
    assert_eq!(ran(&op, &plain), 1.0);
}

#[test]
fn profile_decisions_list_the_custom_choice() {
    // Its own context: the decision log of the shared one holds the other tests' ops too.
    let ctx = Context::with_options(ContextOptions::from_env()).unwrap();
    let op = scale_op(
        &ctx,
        "t_log",
        vec![tagged("slow", 1.0).cost(|_| 900.0), tagged("fast", 2.0).cost(|_| 100.0), tagged("never", 3.0).fits(|_| false)],
    )
    .unwrap();
    ctx.record_decisions(true);
    let x = array(&ctx, 4);
    op.apply1::<f32>(&[&x], &k(0.0)).unwrap().eval().unwrap();
    let decisions: Vec<Decision> = ctx.take_decisions().into_iter().filter(|d| d.site == DecisionSite::Extension).collect();
    ctx.record_decisions(false);
    assert_eq!(decisions.len(), 1, "{decisions:?}");
    let d = &decisions[0];
    assert_eq!((d.op, d.chosen.as_ref()), ("t_log", "fast"));
    let find = |name: &str| d.candidates.iter().find(|(n, _)| n == name).map(|(_, v)| v.clone());
    assert_eq!(find("slow"), Some(Verdict::Estimate { ns: 900.0 }));
    assert_eq!(find("never"), Some(Verdict::Rejected("does not fit")));
    assert!(!d.cached);
}

#[test]
fn a_panicking_lower_is_an_extension_error_and_poisons_the_outputs() {
    let ctx = ctx();
    let op = scale_op(&ctx, "t_panic", vec![Strategy::new("boom", |_| panic!("kaboom"))]).unwrap();
    let x = array(&ctx, 4);
    match op.apply(&[&x], &k(1.0)) {
        Err(LumaError::Extension { op, strategy, message }) => assert_eq!((op.as_str(), strategy.as_str(), message.as_str()), ("t_panic", "boom", "kaboom")),
        other => panic!("{other:?}"),
    }
    // The context works, and the input is untouched.
    assert_eq!(x.to_vec().unwrap()[3], 3.0);
    let failing = Strategy::new("fails", |_| Err(LumaError::InvalidShape("no".into())));
    let op = scale_op(&ctx, "t_fail", vec![failing]).unwrap();
    assert!(matches!(op.apply(&[&x], &k(1.0)), Err(LumaError::Extension { message, .. }) if message == "no"));
}

#[test]
fn a_failing_launch_validates_binds_like_kernel_launch() {
    let ctx = ctx();
    let seen: Arc<Mutex<Vec<LumaError>>> = Arc::default();
    let sink = Arc::clone(&seen);
    let strategy = Strategy::new("bad", move |lower| {
        let kernel = lower.kernel(SCALE).build()?;
        let params = lower.params().to_vec();
        // The uniform binding gets a storage array.
        let kind = lower.launch(&kernel, [4u32, 1, 1], &[Bind::Input(0), Bind::Output(0), Bind::Input(0)]).unwrap_err();
        // The array binding is `array<f32>`, the shader below has 8-byte elements.
        let wide = lower
            .kernel(
                r#"
@group(1) @binding(0) var<storage, read> x: array<vec2<f32>>;
@group(1) @binding(1) var<storage, read_write> y: array<f32>;
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) { y[gid.x] = x[gid.x].x; }
"#,
            )
            .build()?;
        let size = lower.launch(&wide, [4u32, 1, 1], &[Bind::Input(0), Bind::Output(0)]).unwrap_err();
        // A bind that names an input the op does not have.
        let index = lower.launch(&kernel, [4u32, 1, 1], &[Bind::Input(5), Bind::Output(0), Bind::Uniform(params)]).unwrap_err();
        sink.lock().unwrap().extend([kind, size, index]);
        lower.launch(&kernel, [4u32, 1, 1], &[Bind::Input(0), Bind::Output(0), Bind::Uniform(lower.params().to_vec())])
    });
    let op = scale_op(&ctx, "t_binds", vec![strategy]).unwrap();
    let x = array(&ctx, 4);
    assert_eq!(op.apply1::<f32>(&[&x], &k(1.0)).unwrap().to_vec().unwrap(), vec![0.0, 1.0, 2.0, 3.0]);
    let seen = seen.lock().unwrap();
    assert!(matches!(seen[0], LumaError::KernelArgKind { index: 2, .. }), "{:?}", seen[0]);
    assert!(matches!(seen[1], LumaError::KernelArgElementSize { index: 0, .. }), "{:?}", seen[1]);
    assert!(matches!(seen[2], LumaError::InvalidShape(_)), "{:?}", seen[2]);
}

#[test]
fn re_registering_an_identical_op_returns_the_same_handle() {
    let ctx = ctx();
    let a = scale_op(&ctx, "t_same", vec![scaler("plain")]).unwrap();
    let b = scale_op(&ctx, "t_same", vec![scaler("plain")]).unwrap();
    let x = array(&ctx, 4);
    assert_eq!(a.apply1::<f32>(&[&x], &k(2.0)).unwrap().to_vec().unwrap(), b.apply1::<f32>(&[&x], &k(2.0)).unwrap().to_vec().unwrap());
    assert_eq!(a.name(), b.name());
}

#[test]
fn registering_a_different_op_under_a_name_is_an_error() {
    let ctx = ctx();
    scale_op(&ctx, "t_diff", vec![scaler("plain")]).unwrap();
    let err = scale_op(&ctx, "t_diff", vec![scaler("other")]).unwrap_err();
    assert!(matches!(&err, LumaError::InvalidShape(m) if m == "op `t_diff` is already registered differently"), "{err}");
    let err = ctx.custom_op("t_diff").input::<u32>().output::<f32>(OutShape::Same(0)).strategy(scaler("plain")).build().unwrap_err();
    assert!(matches!(err, LumaError::InvalidShape(_)));
    // Declarations that cannot work.
    assert!(ctx.custom_op("t_bad").input::<f32>().strategy(scaler("p")).build().is_err());
    assert!(ctx.custom_op("t_bad").input::<f32>().output::<f32>(OutShape::Same(1)).strategy(scaler("p")).build().is_err());
    assert!(ctx.custom_op("t_bad").input::<f32>().output::<f32>(OutShape::Same(0)).build().is_err());
}

#[test]
fn out_shapes_follow_the_inputs_and_the_params() {
    let ctx = ctx();
    let sum_rows = Strategy::new("rows", |lower| {
        // Writes `n` copies of the first input element.
        let kernel = lower
            .kernel(
                r#"
@group(1) @binding(0) var<storage, read> x: array<f32>;
@group(1) @binding(1) var<storage, read_write> y: array<f32>;
@compute @workgroup_size(16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) { if (gid.x < luma.size.x) { y[gid.x] = x[0]; } }
"#,
            )
            .build()?;
        let n = lower.outputs()[0].shape.len() as u32;
        lower.launch(&kernel, [n, 1, 1], &[Bind::Input(0), Bind::Output(0)])
    });
    let op = ctx
        .custom_op("t_shape")
        .input::<f32>()
        .output::<f32>(OutShape::with(|_: &[ArgInfo], params: &[u8]| Ok(Shape::from(params[0] as usize))))
        .output::<u32>(OutShape::Fixed(Shape::from(2)))
        .params(1)
        .strategy(sum_rows)
        .build()
        .unwrap();
    let x = Array::from_slice_on(&ctx, &[5.0f32, 6.0], 2).unwrap();
    let mut out = op.apply(&[&x], &[7]).unwrap();
    assert_eq!(out.len(), 2);
    let second = out.pop().unwrap();
    assert_eq!((second.elem(), second.shape()), (ElemKind::U32, Shape::from(2)));
    assert_eq!(out.pop().unwrap().take::<f32>().unwrap().to_vec().unwrap(), vec![5.0; 7]);
    assert!(op.apply1::<f32>(&[&x], &[7]).is_err());
}

#[test]
fn scratch_is_charged_to_the_budget_and_released() {
    // Its own context: the memory of the shared one moves with every other test.
    let ctx = Context::with_options(ContextOptions::from_env()).unwrap();
    let before = ctx.memory_stats().resident_bytes;
    let seen = Arc::new(AtomicUsize::new(0));
    let (probe, seen_in) = (ctx.clone(), Arc::clone(&seen));
    const BYTES: u64 = 8 << 20;
    let strategy = Strategy::new("scratchy", move |lower| {
        let scratch = lower.scratch(BYTES)?;
        let fill = lower
            .kernel(
                r#"
@group(1) @binding(0) var<storage, read_write> s: array<u32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) { if (gid.x < luma.size.x) { s[gid.x] = gid.x; } }
"#,
            )
            .build()?;
        lower.launch(&fill, [1024u32, 1, 1], &[Bind::Scratch(scratch)])?;
        seen_in.store((probe.memory_stats().resident_bytes - before) as usize, Ordering::SeqCst);
        let kernel = lower.kernel(SCALE).build()?;
        lower.launch(&kernel, [4u32, 1, 1], &[Bind::Input(0), Bind::Output(0), Bind::Uniform(lower.params().to_vec())])
    });
    let op = scale_op(&ctx, "t_scratch", vec![strategy]).unwrap();
    let x = array(&ctx, 4);
    let y = op.apply1::<f32>(&[&x], &k(2.0)).unwrap();
    assert!(seen.load(Ordering::SeqCst) as u64 >= BYTES, "scratch was not charged: {}", seen.load(Ordering::SeqCst));
    assert_eq!(y.to_vec().unwrap(), vec![0.0, 2.0, 4.0, 6.0]);
    ctx.sync().unwrap();
    drop((x, y));
    let after = ctx.memory_stats().resident_bytes;
    assert!(after.saturating_sub(before) < BYTES, "scratch was not released: {before} -> {after}");
}

#[test]
fn a_custom_op_in_an_iterate_all_step_runs_step_by_step() {
    let ctx = ctx();
    let op = scale_op(&ctx, "t_iter", vec![scaler("plain")]).unwrap();
    let x = Array::from_slice_on(&ctx, &[1.0f32, 2.0, 3.0], 3).unwrap();
    let [y] = iterate_all([&x], 3, |[s]| [op.apply1::<f32>(&[s], &k(2.0)).unwrap() + 1.0]);
    // s -> 2 s + 1, three times.
    let want: Vec<f32> = [1.0f32, 2.0, 3.0].iter().map(|&v| 2.0 * (2.0 * (2.0 * v + 1.0) + 1.0) + 1.0).collect();
    assert_eq!(y.to_vec().unwrap(), want);
}

#[test]
fn a_custom_op_on_every_backend() {
    for open in [Context::cpu, Context::cuda, Context::wgpu, Context::metal] {
        let Ok(ctx) = open() else { continue };
        let op = scale_op(&ctx, "t_backend", vec![scaler("plain")]).unwrap();
        let x = array(&ctx, 300);
        let y = op.apply1::<f32>(&[&x], &k(2.0)).unwrap();
        assert_eq!(y.to_vec().unwrap(), doubled(300), "{:?}", ctx.backend());
    }
}

#[test]
fn counting_strategies_are_run_once_per_apply() {
    let ctx = ctx();
    let calls = Arc::new(AtomicUsize::new(0));
    let op = scale_op(&ctx, "t_count", vec![counting("only", &calls)]).unwrap();
    let x = array(&ctx, 4);
    op.apply(&[&x], &k(1.0)).unwrap();
    op.apply(&[&x], &k(1.0)).unwrap();
    assert_eq!(calls.load(Ordering::SeqCst), 2);
}
