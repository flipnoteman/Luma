//! The strategy registry (`LUMA_STRATEGY`, `Context::record_decisions`): forcing a reduction strategy changes which kernels run,
//! never what a strategy of the same bit class computes, and the decision log names the strategy that ran.

use crate::common::values;
use luma::*;

/// The strategies of the reductions and their bit class (strategies with equal classes give equal bits on one device).
const REDUCE: [(&str, u8); 7] = [("host", 1), ("merged", 2), ("axis-merged", 3), ("bits", 4), ("single-pass", 5), ("two-pass", 6), ("axis", 3)];

fn open(forced: Option<&'static str>) -> Context {
    let options = ContextOptions::from_env().calibrate(false);
    Context::with_options(match forced {
        Some(forced) => options.strategy(forced),
        None => options.strategy(""),
    })
    .unwrap_or_else(|error| panic!("{forced:?}: {error}"))
}

fn class_of(label: &str) -> u8 {
    let name = label.split(' ').next().unwrap();
    REDUCE.iter().find(|(known, _)| *known == name).unwrap_or_else(|| panic!("{label:?} is not a reduce strategy")).1
}

/// The value of a lazy reduction, recorded by an operation that uses it on the device (which is where the registry selects:
/// `item()` of a reduction nothing else uses is answered by the immediate readback).
fn f32_of(ctx: &Context, value: &Array<f32>) -> f32 {
    (&Array::<f32>::full_on(ctx, [1], 1.0).unwrap() * value).item().unwrap()
}

fn u32_of(ctx: &Context, value: &Array<u32>) -> u32 {
    (&Array::<u32>::full_on(ctx, [1], 1).unwrap() * value).item().unwrap()
}

fn bool_of(ctx: &Context, value: &Array<bool>) -> bool {
    f32_of(ctx, &value.to_f32()) != 0.0
}

/// The strategy names a battery of reductions ran with, and their values as bits.
struct Run {
    labels: Vec<String>,
    bits: Vec<u32>,
}

/// Forces each reduction of the battery on its own (so a forced strategy sees one value at a time), recording which strategy ran.
fn battery(ctx: &Context) -> Run {
    let n = 200_003;
    let host = values(n, 21, -2.0, 2.0);
    let x = Array::from_slice_on(ctx, &host, n).unwrap();
    let words: Vec<u32> = host.iter().map(|v| (v * 1000.0) as i32 as u32).collect();
    let u = Array::<u32>::from_slice_on(ctx, &words, n).unwrap();
    let mut bits = Vec::new();
    ctx.record_decisions(true);
    let mut force = |value: Array<f32>| bits.push(f32_of(ctx, &value).to_bits());
    // A pending float sum (reassociates between a one- and a two-pass reduction), an evaluated one, extrema.
    force((&x * 1.5 + 0.25).sum());
    force(x.sum());
    force((&x * 0.5).min());
    force((&x * 0.5).max());
    force((&x * &x).sum());
    let mut int_bits = Vec::new();
    // Integer and boolean reductions are exact whichever strategy runs.
    int_bits.push(u32_of(ctx, &(&u + 3u32).sum()));
    int_bits.push(u32::from(bool_of(ctx, &x.gt(0.5).any())));
    int_bits.push(u32::from(bool_of(ctx, &x.gt(-5.0).all())));
    int_bits.push(u32_of(ctx, &x.gt(0.0).count()));
    ctx.sync().unwrap();
    let decisions = ctx.take_decisions();
    ctx.record_decisions(false);
    let labels = decisions.iter().filter(|d| d.site == DecisionSite::ReducePlan).map(|d| d.chosen.to_string()).collect();
    bits.extend(int_bits);
    Run { labels, bits }
}

#[test]
fn forcing_each_reduce_strategy_matches_the_default_bit_for_bit_where_the_class_is_equal() {
    let default = battery(&open(None));
    assert!(!default.labels.is_empty(), "the log names the strategies of the default run");
    // The floating-point values the default run's strategy made: what a strategy of the same class has to reproduce.
    let float_class = class_of(&default.labels[0]);
    for (name, class) in REDUCE {
        let forced: &'static str = Box::leak(format!("reduce:{name}").into_boxed_str());
        let ctx = open(Some(forced));
        let run = battery(&ctx);
        assert_eq!(run.bits.len(), default.bits.len());
        // Extrema and integer reductions are exact whichever strategy runs; so are the booleans.
        for at in [2, 3, 5, 6, 7, 8] {
            assert_eq!(run.bits[at], default.bits[at], "{name}: value {at}");
        }
        // The float sums: the bits of the default strategy's class, else the same values within the rounding of another order.
        for at in [0, 1, 4] {
            if run.labels.first().map(|label| class_of(label)) == Some(float_class) && class == float_class {
                assert_eq!(run.bits[at], default.bits[at], "{name}: value {at}: same class, same bits");
            } else {
                let (got, want) = (f32::from_bits(run.bits[at]), f32::from_bits(default.bits[at]));
                assert!((got - want).abs() <= 1e-3 * want.abs().max(1.0), "{name}: value {at}: {got} against {want}");
            }
        }
    }
}

#[test]
fn profile_decisions_name_the_reduce_strategy() {
    let ctx = open(None);
    let n = 1 << 20;
    let x = Array::from_slice_on(&ctx, &values(n, 5, 0.5, 1.5), n).unwrap();
    ctx.record_decisions(true);
    f32_of(&ctx, &(&x * 2.0).sum());
    let small = Array::from_slice_on(&ctx, &values(64, 6, 0.5, 1.5), 64).unwrap();
    f32_of(&ctx, &small.sum());
    ctx.sync().unwrap();
    let decisions = ctx.take_decisions();
    let plans: Vec<&Decision> = decisions.iter().filter(|d| d.site == DecisionSite::ReducePlan).collect();
    assert!(plans.len() >= 2 && plans.iter().all(|d| !d.chosen.is_empty() && d.op == "sum"), "{decisions:?}");
    assert!(plans.iter().all(|d| d.location.file().ends_with("strategies.rs")), "{plans:?}");
    for plan in &plans {
        // Labels are the strategy name, plus the first pass's groups where there are several: no ids, no pointers.
        let name = plan.chosen.split(' ').next().unwrap();
        assert!(REDUCE.iter().any(|(known, _)| *known == name), "{}", plan.chosen);
        assert!(plan.candidates.iter().all(|(label, _)| REDUCE.iter().any(|(known, _)| known == label)), "{plan:?}");
        // A strategy that is not chosen says why not.
        assert!(plan.candidates.iter().all(|(_, verdict)| matches!(verdict, Verdict::Rejected(why) if !why.is_empty()) || matches!(verdict, Verdict::Estimate { .. })), "{plan:?}");
    }
    if ctx.backend() == BackendKind::Cpu {
        // `Caps::HOST_PATH` is a GPU on unified memory: the host strategy is out on the CPU backend, by capability, not by declining.
        let large = plans.iter().find(|d| d.chosen.starts_with("single-pass") || d.chosen.starts_with("two-pass")).expect("a device strategy ran");
        assert!(large.chosen.starts_with("two-pass groups="), "{large:?}");
        assert!(large.candidates.iter().all(|(_, verdict)| *verdict != Verdict::Rejected("declined")), "{large:?}");
        assert!(large.candidates.contains(&("host".into(), Verdict::Rejected("caps"))), "{large:?}");
    }
}

#[test]
fn a_strategy_that_does_not_fit_leaves_the_default_selection_and_says_so() {
    // `bits` fits only a packed boolean value: forced on a float sum it is rejected and the default strategy runs.
    let ctx = open(Some("reduce:bits"));
    let x = Array::from_slice_on(&ctx, &values(4096, 3, 0.5, 1.5), 4096).unwrap();
    ctx.record_decisions(true);
    f32_of(&ctx, &(&x + 1.0).sum());
    let decisions = ctx.take_decisions();
    let plan = decisions.iter().find(|d| d.site == DecisionSite::ReducePlan).expect("a reduce decision");
    assert!(!plan.chosen.starts_with("bits"), "{plan:?}");
    assert!(plan.candidates.contains(&("bits".into(), Verdict::Rejected("forced strategy does not fit"))), "{plan:?}");
}

#[test]
fn an_unknown_strategy_in_the_options_fails_the_context() {
    for text in ["reduce:fast", "scan:two-pass", "reduce"] {
        let error = Context::with_options(ContextOptions::from_env().calibrate(false).strategy(text)).err().expect("fails");
        assert!(matches!(error, LumaError::InvalidOption(_)), "{text}: {error}");
    }
}
