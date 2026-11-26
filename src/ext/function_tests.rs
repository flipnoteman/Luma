//! Custom elementwise functions: what the planner does with a call.

use crate::*;

fn ctx() -> Context {
    Context::new().unwrap()
}

fn f32s(ctx: &Context, data: &[f32]) -> Array<f32> {
    Array::from_slice_on(ctx, data, data.len()).unwrap()
}

#[test]
fn a_chain_around_a_custom_function_is_one_launch() {
    let ctx = ctx();
    let f = ctx.elementwise_fn::<f32>("fu_f", "fn fu_f(x: f32) -> f32 { return x * x - 1.0; }").unwrap();
    let xs: Vec<f32> = (0..5000).map(|i| i as f32 * 0.001 - 2.0).collect();
    let (x, y) = (f32s(&ctx, &xs), f32s(&ctx, &xs));
    let before = ctx.jit().stats();
    let z = f.call1(&x * 2.0) + &y;
    assert_eq!(ctx.jit().stats().elementwise_launches, before.elementwise_launches, "recording launches nothing");
    let got = z.to_vec().unwrap();
    assert_eq!(ctx.jit().stats().elementwise_launches - before.elementwise_launches, 1);
    let want: Vec<f32> = xs.iter().map(|&v| (v * 2.0) * (v * 2.0) - 1.0 + v).collect();
    for (g, w) in got.iter().zip(&want) {
        assert!((g - w).abs() <= 1e-5 * w.abs().max(1.0), "{g} vs {w}");
    }
}

#[test]
fn calls_hash_apart_in_the_kernel_cache() {
    let ctx = ctx();
    let f = ctx.elementwise_fn::<f32>("fh_f", "fn fh_f(x: f32) -> f32 { return x + 1.0; }").unwrap();
    let g = ctx.elementwise_fn::<f32>("fh_g", "fn fh_g(x: f32) -> f32 { return x + 2.0; }").unwrap();
    let x = f32s(&ctx, &[1.0, 2.0, 3.0]);
    let built = |array: &Array<f32>| {
        let before = ctx.jit().stats();
        let values = array.to_vec().unwrap();
        let after = ctx.jit().stats();
        (values, after.built - before.built, after.hits - before.hits)
    };
    let (vf, built_f, _) = built(&f.call1(&x));
    let (vg, built_g, _) = built(&g.call1(&x));
    assert_eq!((vf, vg), (vec![2.0, 3.0, 4.0], vec![3.0, 4.0, 5.0]));
    assert_eq!((built_f, built_g), (1, 1), "two functions are two kernels");
    // The same function again is a hit, not a build.
    let (again, built_again, hits) = built(&f.call1(&x));
    assert_eq!((again, built_again, hits), (vec![2.0, 3.0, 4.0], 0, 1));
}

#[test]
fn a_function_with_a_loop_and_a_branch_runs() {
    let ctx = ctx();
    let collatz = ctx
        .elementwise_fn::<u32>(
            "fl_steps",
            "fn fl_steps(n: u32) -> u32 {
    var v = n;
    var steps = 0u;
    loop {
        if (v <= 1u) { break; }
        if ((v & 1u) == 0u) { v = v >> 1u; } else { v = 3u * v + 1u; }
        steps = steps + 1u;
    }
    return steps;
}",
        )
        .unwrap();
    let x = Array::from_slice_on(&ctx, &[1u32, 2, 3, 6, 7, 27], 6).unwrap();
    assert_eq!(collatz.call1(&x).to_vec().unwrap(), vec![0, 1, 7, 8, 16, 111]);
}
