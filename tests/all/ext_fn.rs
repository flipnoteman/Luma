//! Custom elementwise functions (`Context::elementwise_fn`).

use crate::common;

use common::{assert_close, values};
use luma::*;

fn ctx() -> Context {
    default_context().expect("Could not create Context.")
}

fn f32s(ctx: &Context, data: &[f32]) -> Array<f32> {
    Array::from_slice_on(ctx, data, data.len()).unwrap()
}

const SMOOTH: &str = "fn t_smooth(a: f32, b: f32, x: f32) -> f32 {
    let t = clamp((x - a) / (b - a), 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
}";

#[test]
fn a_smoothstep_function_matches_the_builtin_expression() {
    let ctx = ctx();
    let smooth = ctx.elementwise_fn::<f32>("t_smooth", SMOOTH).unwrap();
    let xs = values(5000, 3, -1.0, 2.0);
    let x = f32s(&ctx, &xs);
    let got = smooth.call3(0.0f32, 1.0f32, &x).to_vec().unwrap();
    let want: Vec<f32> = xs
        .iter()
        .map(|&v| {
            let t = v.clamp(0.0, 1.0);
            t * t * (3.0 - 2.0 * t)
        })
        .collect();
    assert_close(&got, &want, 1e-5, 1e-6);
    // The same on arrays for the edges, broadcast.
    let a = f32s(&ctx, &[0.0]);
    let again = smooth.call3(&a, 1.0f32, &x).to_vec().unwrap();
    assert_eq!(again, got);
}

const XORSHIFT: &str = "fn t_xorshift(x: u32) -> u32 {
    var v = x;
    v ^= v << 13u;
    v ^= v >> 17u;
    v ^= v << 5u;
    return v;
}";

fn xorshift(mut v: u32) -> u32 {
    v ^= v << 13;
    v ^= v >> 17;
    v ^= v << 5;
    v
}

#[test]
fn an_integer_hash_function_is_bit_exact_on_every_backend() {
    let data: Vec<u32> = (0..3000u32).map(|i| i.wrapping_mul(2654435761).wrapping_add(12345)).collect();
    let want: Vec<u32> = data.iter().map(|&v| xorshift(v)).collect();
    let mut ran = 0;
    for open in [Context::cpu, Context::cuda, Context::wgpu, Context::metal] {
        let Ok(ctx) = open() else { continue };
        let hash = ctx.elementwise_fn::<u32>("t_xorshift", XORSHIFT).unwrap();
        let x = Array::from_slice_on(&ctx, &data, data.len()).unwrap();
        assert_eq!(hash.call1(&x).to_vec().unwrap(), want, "{:?}", ctx.backend());
        // Fused with its neighbours.
        let chained = (hash.call1(&(&x ^ 1u32)) >> 3u32).to_vec().unwrap();
        let want_chained: Vec<u32> = data.iter().map(|&v| xorshift(v ^ 1) >> 3).collect();
        assert_eq!(chained, want_chained, "{:?}", ctx.backend());
        ran += 1;
    }
    assert!(ran >= 1);
}

#[test]
fn an_i32_function_with_a_scalar_operand() {
    let ctx = ctx();
    let axpy = ctx.elementwise_fn::<i32>("t_axpy", "fn t_axpy(a: i32, x: i32, b: i32) -> i32 { return a * x + b; }").unwrap();
    let data: Vec<i32> = (-50..50).collect();
    let x = Array::from_slice_on(&ctx, &data, data.len()).unwrap();
    let got = axpy.call3(3, &x, -7).to_vec().unwrap();
    assert_eq!(got, data.iter().map(|&v| 3 * v - 7).collect::<Vec<_>>());
    // Needs an array operand.
    let err = axpy.call3(1, 2, 3).to_vec().unwrap_err();
    assert!(matches!(err.root(), LumaError::InvalidShape(_)), "{err}");
}

#[test]
fn two_different_custom_functions_in_one_kernel_do_not_collide() {
    let ctx = ctx();
    let f = ctx
        .elementwise_fn::<f32>("t_fa", "const t_fa_k: f32 = 3.0;\nfn t_fa_helper(x: f32) -> f32 { return x * t_fa_k; }\nfn t_fa(x: f32) -> f32 { return t_fa_helper(x) + 1.0; }")
        .unwrap();
    let g = ctx
        .elementwise_fn::<f32>("t_fb", "const t_fb_k: f32 = 5.0;\nfn t_fb_helper(x: f32) -> f32 { return x * t_fb_k; }\nfn t_fb(x: f32) -> f32 { return t_fb_helper(x) - 1.0; }")
        .unwrap();
    let xs = values(2000, 8, -2.0, 2.0);
    let x = f32s(&ctx, &xs);
    let got = (f.call1(&x) + g.call1(&x)).to_vec().unwrap();
    let want: Vec<f32> = xs.iter().map(|&v| (v * 3.0 + 1.0) + (v * 5.0 - 1.0)).collect();
    assert_close(&got, &want, 1e-5, 1e-5);
}

#[test]
fn a_custom_function_in_a_reduction_prologue() {
    let ctx = ctx();
    let sq = ctx.elementwise_fn::<f32>("t_sq", "fn t_sq(x: f32) -> f32 { return x * x + 1.0; }").unwrap();
    let xs = values(10_000, 4, -1.0, 1.0);
    let x = f32s(&ctx, &xs);
    let got = sq.call1(&x * 0.5).sum().item().unwrap();
    let want: f64 = xs.iter().map(|&v| f64::from(v * 0.5) * f64::from(v * 0.5) + 1.0).sum();
    assert!((f64::from(got) - want).abs() < 1e-2 * want.abs(), "{got} vs {want}");
}

fn refused(ctx: &Context, name: &str, wgsl: &str) -> String {
    match ctx.elementwise_fn::<f32>(name, wgsl) {
        Err(LumaError::ShaderCompile(message)) => message,
        Err(other) => panic!("{other:?}"),
        Ok(_) => panic!("`{name}` was accepted"),
    }
}

#[test]
fn a_function_that_reads_a_global_is_refused() {
    let ctx = ctx();
    refused(&ctx, "t_g1", "var<private> t_g1_state: f32 = 1.0;\nfn t_g1(x: f32) -> f32 { return x + t_g1_state; }");
    refused(&ctx, "t_g2", "@group(1) @binding(0) var<storage, read> t_g2_data: array<f32>;\nfn t_g2(x: f32) -> f32 { return x + t_g2_data[0]; }");
    refused(&ctx, "t_g3", "struct T_g3_u { k: f32 }\n@group(1) @binding(0) var<uniform> t_g3_u: T_g3_u;\nfn t_g3(x: f32) -> f32 { return x * t_g3_u.k; }");
    refused(&ctx, "t_g4", "var<workgroup> t_g4_w: f32;\nfn t_g4(x: f32) -> f32 { return x + t_g4_w; }");
}

#[test]
fn arrays_pointers_atomics_and_barriers_are_refused() {
    let ctx = ctx();
    let m = refused(&ctx, "t_arr", "fn t_arr(x: f32) -> f32 { var a = array<f32, 2>(x, x); return a[1]; }");
    assert!(m.contains("array"), "{m}");
    let m = refused(&ctx, "t_ptr", "fn t_ptr_inner(p: ptr<function, f32>) -> f32 { return *p; }\nfn t_ptr(x: f32) -> f32 { var v = x; return t_ptr_inner(&v); }");
    assert!(m.contains("pointer"), "{m}");
    refused(&ctx, "t_atomic", "var<workgroup> t_atomic_a: atomic<u32>;\nfn t_atomic(x: f32) -> f32 { atomicAdd(&t_atomic_a, 1u); return x; }");
    refused(&ctx, "t_barrier", "fn t_barrier(x: f32) -> f32 { workgroupBarrier(); return x; }");
    refused(&ctx, "t_deriv", "fn t_deriv(x: f32) -> f32 { return dpdx(x); }");
}

#[test]
fn mixed_parameter_types_are_refused() {
    let ctx = ctx();
    let m = refused(&ctx, "t_mix", "fn t_mix(x: f32, n: u32) -> f32 { return x * f32(n); }");
    assert!(m.contains("f32"), "{m}");
    refused(&ctx, "t_ret", "fn t_ret(x: f32) -> u32 { return u32(x); }");
    refused(&ctx, "t_vec", "fn t_vec(x: vec2<f32>) -> f32 { return x.x; }");
    refused(&ctx, "t_none", "fn t_none() -> f32 { return 1.0; }");
    refused(&ctx, "t_four", "fn t_four(a: f32, b: f32, c: f32, d: f32) -> f32 { return a + b + c + d; }");
    refused(&ctx, "t_missing", "fn other(x: f32) -> f32 { return x; }");
    // A u32 function asked for as f32.
    assert!(ctx.elementwise_fn::<u32>("t_u", "fn t_u(x: f32) -> f32 { return x; }").is_err());
}

#[test]
fn helpers_must_carry_the_function_name_prefix() {
    let ctx = ctx();
    let m = refused(&ctx, "t_pre", "fn helper(x: f32) -> f32 { return x; }\nfn t_pre(x: f32) -> f32 { return helper(x); }");
    assert!(m.contains("helper"), "{m}");
    let m = refused(&ctx, "t_pre2", "const k: f32 = 2.0;\nfn t_pre2(x: f32) -> f32 { return x * k; }");
    assert!(m.contains("`k`"), "{m}");
    ctx.elementwise_fn::<f32>("t_pre3", "const t_pre3_k: f32 = 2.0;\nfn t_pre3_h(x: f32) -> f32 { return x * t_pre3_k; }\nfn t_pre3(x: f32) -> f32 { return t_pre3_h(x); }").unwrap();
    let m = refused(&ctx, "luma_mine", "fn luma_mine(x: f32) -> f32 { return x; }");
    assert!(m.contains("library"), "{m}");
}

#[test]
fn a_second_source_under_one_name_is_an_error() {
    let ctx = ctx();
    let first = ctx.elementwise_fn::<f32>("t_twice", "fn t_twice(x: f32) -> f32 { return x + 1.0; }").unwrap();
    let again = ctx.elementwise_fn::<f32>("t_twice", "fn t_twice(x: f32) -> f32 { return x + 1.0; }").unwrap();
    let x = f32s(&ctx, &[1.0, 2.0]);
    assert_eq!(first.call1(&x).to_vec().unwrap(), again.call1(&x).to_vec().unwrap());
    let m = refused(&ctx, "t_twice", "fn t_twice(x: f32) -> f32 { return x + 2.0; }");
    assert!(m.contains("already registered"), "{m}");
}

#[test]
fn operand_count_and_context_are_checked() {
    let ctx = ctx();
    let f = ctx.elementwise_fn::<f32>("t_ops", "fn t_ops(a: f32, b: f32) -> f32 { return a - b; }").unwrap();
    let x = f32s(&ctx, &[1.0, 2.0, 3.0]);
    assert_eq!(f.call2(&x, 1.0f32).to_vec().unwrap(), vec![0.0, 1.0, 2.0]);
    assert_eq!(f.call2(10.0f32, &x).to_vec().unwrap(), vec![9.0, 8.0, 7.0]);
    assert!(f.call1(&x).to_vec().is_err());
    let other = Context::with_options(ContextOptions::from_env()).unwrap();
    let foreign = f32s(&other, &[1.0, 2.0, 3.0]);
    assert!(f.call2(&foreign, &x).to_vec().is_err());
    let short = f32s(&ctx, &[1.0, 2.0]);
    assert!(f.call2(&short, &x).to_vec().is_err());
    // The default-context form.
    let g = elementwise_fn::<f32>("t_default", "fn t_default(x: f32) -> f32 { return -x; }").unwrap();
    assert_eq!(g.call1(Array::from_slice(&[1.0f32], 1).unwrap()).to_vec().unwrap(), vec![-1.0]);
    assert_eq!(g.name(), "t_default");
}
