//! Backend selection and cross-backend differential tests.
//!
//! Every backend this host can open (wgpu everywhere it has an adapter, Metal on Apple GPUs, CUDA
//! on NVIDIA GPUs) runs the same programs. Each result is checked against a deterministic host
//! computation, and the backends are compared with one another (the first available of wgpu,
//! Metal, CUDA is the baseline). A backend is skipped only when opening it fails with
//! `NoAdapter` / `NoDevice`, and the skip is printed: any other error, a shader that does not
//! compile or a GPU error is a failure.

mod common;

use std::fmt::Debug;
use std::process::Command;
use std::sync::atomic::{AtomicBool, Ordering};

use common::values;
use luma::*;

// -- Opening backends --

/// Wgpu first: it is the baseline the native backends are compared with.
const ALL: [BackendKind; 3] = [BackendKind::Wgpu, BackendKind::Metal, BackendKind::Cuda];

fn label(kind: BackendKind) -> &'static str {
    match kind {
        BackendKind::Wgpu => "wgpu",
        BackendKind::Metal => "metal",
        BackendKind::Cuda => "cuda",
        _ => "auto",
    }
}

static ANNOUNCED: [AtomicBool; 3] = [AtomicBool::new(false), AtomicBool::new(false), AtomicBool::new(false)];

/// A context of `kind`, or `None` (announced, once per process) if the host has no such device.
fn open(kind: BackendKind) -> Option<Context> {
    let result = Context::with_options(ContextOptions::default().backend(kind).calibrate(false));
    let slot = ALL.iter().position(|&k| k == kind).expect("a concrete backend kind");
    let first = !ANNOUNCED[slot].swap(true, Ordering::Relaxed);
    match result {
        Ok(ctx) => {
            assert_eq!(ctx.backend(), kind, "a context of {} reports another backend", label(kind));
            if first {
                println!("[backends] {}: available", label(kind));
            }
            Some(ctx)
        }
        Err(error @ (LumaError::NoAdapter(_) | LumaError::NoDevice(_))) => {
            if first {
                println!("[backends] {}: SKIPPED, not available on this host ({error})", label(kind));
            }
            None
        }
        Err(error) => panic!("opening the {} backend failed: {error}", label(kind)),
    }
}

/// A fresh context of every backend the host has.
fn backends() -> Vec<(BackendKind, Context)> {
    ALL.iter().filter_map(|&kind| open(kind).map(|ctx| (kind, ctx))).collect()
}

fn on<T: GpuType>(ctx: &Context, data: &[T]) -> Array<T> {
    Array::from_slice_on(ctx, data, data.len()).unwrap()
}

// -- Comparisons --

fn first_diff<T: PartialEq + Debug>(a: &[T], b: &[T]) -> Result<(), String> {
    if a.len() != b.len() {
        return Err(format!("length {} vs {}", a.len(), b.len()));
    }
    match a.iter().zip(b).position(|(x, y)| x != y) {
        None => Ok(()),
        Some(i) => Err(format!("index {i}: {:?} vs {:?}", a[i], b[i])),
    }
}

/// Elementwise within `abs` or `rel` (of `b`).
fn first_far(a: &[f32], b: &[f32], rel: f32, abs: f32) -> Result<(), String> {
    if a.len() != b.len() {
        return Err(format!("length {} vs {}", a.len(), b.len()));
    }
    let far = |x: f32, y: f32| !(x == y || (x - y).abs() <= abs.max(rel * y.abs()));
    match a.iter().zip(b).position(|(&x, &y)| far(x, y)) {
        None => Ok(()),
        Some(i) => Err(format!("index {i}: {} vs {} (rel {rel}, abs {abs})", a[i], b[i])),
    }
}

fn must(who: &str, what: &str, outcome: Result<(), String>) {
    if let Err(message) = outcome {
        panic!("{who}: {what}: {message}");
    }
}

/// Every result after the first agrees with the first (the baseline backend's).
fn agree<R>(what: &str, results: &[(BackendKind, R)], same: impl Fn(&R, &R) -> Result<(), String>) {
    let Some((base_kind, base)) = results.first() else { return };
    for (kind, result) in &results[1..] {
        if let Err(message) = same(base, result) {
            panic!("{what}: {} differs from {}: {message}", label(*kind), label(*base_kind));
        }
    }
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
}

// -- Backend selection --

#[test]
fn a_gpu_backend_is_available() {
    let available: Vec<&str> = backends().iter().map(|(kind, _)| label(*kind)).collect();
    let skipped: Vec<&str> = ALL.iter().map(|&k| label(k)).filter(|name| !available.contains(name)).collect();
    println!("[backends] available: {available:?}, skipped: {skipped:?}");
    assert!(!available.is_empty(), "no backend could be opened on this host");
}

#[test]
fn constructors_open_their_backend_or_report_it_unavailable() {
    type Open = fn() -> Result<Context>;
    for (kind, make) in [(BackendKind::Wgpu, Context::wgpu as Open), (BackendKind::Metal, Context::metal as Open), (BackendKind::Cuda, Context::cuda as Open)] {
        let from_options = Context::with_options(ContextOptions::from_env().backend(kind).calibrate(false));
        match make() {
            Ok(ctx) => {
                assert_eq!(ctx.backend(), kind, "Context::{}()", label(kind));
                assert!(from_options.is_ok(), "{}: the constructor opened a device, the options did not", label(kind));
            }
            Err(LumaError::NoAdapter(_) | LumaError::NoDevice(_)) => {
                assert!(from_options.is_err(), "{}: the constructor found no device, the options did", label(kind));
            }
            Err(error) => panic!("Context::{}() failed: {error}", label(kind)),
        }
    }
}

#[test]
fn a_cpu_device_exists_only_through_wgpu() {
    for kind in [BackendKind::Metal, BackendKind::Cuda] {
        let options = ContextOptions::default().backend(kind).device(DeviceKind::Cpu).calibrate(false);
        match Context::with_options(options) {
            Err(LumaError::NoDevice(message)) => {
                assert!(message.contains("wgpu") || message.contains("built without"), "{}: {message}", label(kind));
            }
            other => panic!("{} on a CPU device must be refused with NoDevice, got {other:?}", label(kind)),
        }
    }
    match Context::with_options(ContextOptions::default().backend(BackendKind::Wgpu).device(DeviceKind::Cpu).calibrate(false)) {
        Ok(ctx) => {
            assert!(ctx.is_cpu());
            assert_eq!(ctx.backend(), BackendKind::Wgpu);
            let sum = on(&ctx, &[1.0f32, 2.0, 3.0, 4.0]).sum().item().unwrap();
            assert_eq!(sum, 10.0);
            println!("[backends] wgpu CPU device: available");
        }
        Err(LumaError::NoAdapter(_) | LumaError::NoDevice(_)) => println!("[backends] wgpu CPU device: SKIPPED, no software adapter on this host"),
        Err(error) => panic!("opening the wgpu CPU device failed: {error}"),
    }
}

#[test]
fn auto_takes_the_first_available_of_cuda_metal_wgpu() {
    let expected = [BackendKind::Cuda, BackendKind::Metal, BackendKind::Wgpu].into_iter().find(|&kind| open(kind).is_some());
    let result = Context::with_options(ContextOptions::default().backend(BackendKind::Auto).calibrate(false));
    match (result, expected) {
        (Ok(ctx), Some(kind)) => assert_eq!(ctx.backend(), kind, "auto must prefer cuda, then metal, then wgpu"),
        (Err(LumaError::NoAdapter(_) | LumaError::NoDevice(_)), None) => {}
        (other, expected) => panic!("auto gave {other:?}, expected {expected:?}"),
    }
}

const CHILD: &str = "environment_child_probe";

/// Runs in a child process of `the_environment_selects_the_backend`, which sets the environment
/// before the process starts (changing a process's environment while tests run is not safe).
#[test]
#[ignore = "run by the_environment_selects_the_backend in a child process with its own environment"]
fn environment_child_probe() {
    let expect = std::env::var("LUMA_BACKENDS_EXPECT").expect("LUMA_BACKENDS_EXPECT is set by the parent test");
    match (Context::new(), expect.as_str()) {
        (Ok(ctx), "wgpu-cpu") => assert!(ctx.is_cpu() && ctx.backend() == BackendKind::Wgpu, "{:?} cpu={}", ctx.backend(), ctx.is_cpu()),
        (Ok(ctx), name) => assert_eq!(label(ctx.backend()), name),
        (Err(LumaError::NoAdapter(_) | LumaError::NoDevice(_)), "unavailable") => {}
        (other, name) => panic!("expected {name}, got {other:?}"),
    }
    println!("child: the environment resolved to {expect}");
}

#[test]
fn the_environment_selects_the_backend() {
    let named = |kind: BackendKind| if open(kind).is_some() { label(kind) } else { "unavailable" };
    let auto = [BackendKind::Cuda, BackendKind::Metal, BackendKind::Wgpu].into_iter().find(|&kind| open(kind).is_some()).map_or("unavailable", label);
    let wgpu_cpu = Context::with_options(ContextOptions::default().backend(BackendKind::Wgpu).device(DeviceKind::Cpu).calibrate(false)).is_ok();
    let cases: Vec<(Vec<(&str, &str)>, &str)> = vec![
        (vec![("LUMA_BACKEND", "wgpu")], named(BackendKind::Wgpu)),
        (vec![("LUMA_BACKEND", "metal")], named(BackendKind::Metal)),
        (vec![("LUMA_BACKEND", "cuda")], named(BackendKind::Cuda)),
        (vec![("LUMA_BACKEND", "auto")], auto),
        // LUMA_DEVICE names a backend when LUMA_BACKEND is not set ...
        (vec![("LUMA_DEVICE", "wgpu")], named(BackendKind::Wgpu)),
        (vec![("LUMA_DEVICE", "metal")], named(BackendKind::Metal)),
        // ... and LUMA_BACKEND wins when both are.
        (vec![("LUMA_BACKEND", "metal"), ("LUMA_DEVICE", "wgpu")], named(BackendKind::Metal)),
        // A CPU device only exists through wgpu.
        (vec![("LUMA_BACKEND", "metal"), ("LUMA_DEVICE", "cpu")], "unavailable"),
        (vec![("LUMA_BACKEND", "wgpu"), ("LUMA_DEVICE", "cpu")], if wgpu_cpu { "wgpu-cpu" } else { "unavailable" }),
    ];
    let exe = std::env::current_exe().expect("the path of this test binary");
    for (vars, expect) in cases {
        let mut command = Command::new(&exe);
        command.args([CHILD, "--exact", "--ignored", "--nocapture", "--test-threads=1"]);
        for var in ["LUMA_BACKEND", "LUMA_DEVICE", "LUMA_CALIBRATE"] {
            command.env_remove(var);
        }
        command.env("LUMA_CALIBRATE", "0").env("LUMA_BACKENDS_EXPECT", expect);
        for (name, value) in &vars {
            command.env(name, value);
        }
        let output = command.output().expect("the child process starts");
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(output.status.success(), "{vars:?} (expected {expect}) failed:\n{stdout}\n{stderr}");
        println!("[backends] {vars:?} -> {expect}");
    }
}

// -- Elementwise, tanh, select --

struct Elementwise {
    tanh: Vec<f32>,
    fused_tanh: Vec<f32>,
    select_f32: Vec<f32>,
    select_u32: Vec<u32>,
    select_i32: Vec<i32>,
}

#[test]
fn elementwise_tanh_and_scalar_select_agree_with_the_host() {
    let n = 4099;
    let mut xs = values(n, 11, -20.0, 20.0);
    // The saturation boundary of the clamped tanh, tiny arguments and huge ones.
    let special = [0.0, 15.0, -15.0, 14.99, -14.99, 15.01, -15.01, 100.0, -100.0, 1e-6, -1e-6, 88.0];
    xs[..special.len()].copy_from_slice(&special);
    let us: Vec<u32> = (0..n as u32).map(|i| i.wrapping_mul(2654435761) >> 22).collect();
    let is: Vec<i32> = (0..n as i32).map(|i| i - 2000).collect();

    let want_tanh: Vec<f32> = xs.iter().map(|v| v.tanh()).collect();
    let want_fused: Vec<f32> = xs.iter().map(|v| (v * 0.5 + 1.0).tanh() * 2.0).collect();
    let want_select_f32: Vec<f32> = xs.iter().map(|&v| if v > 0.0 { v * 2.0 } else { v * -0.5 }).collect();
    let want_select_u32: Vec<u32> = us.iter().map(|&u| if u > 500 { u / 7 } else { u * 3 }).collect();
    let want_select_i32: Vec<i32> = is.iter().map(|&i| if i < 0 { i * -2 } else { i + 5 }).collect();

    let mut results = Vec::new();
    for (kind, ctx) in backends() {
        let (x, u, i) = (on(&ctx, &xs), on(&ctx, &us), on(&ctx, &is));
        let doubled = &x * 2.0;
        let halved = &x * -0.5;
        let result = Elementwise {
            tanh: x.tanh().to_vec().unwrap(),
            fused_tanh: ((&x * 0.5 + 1.0).tanh() * 2.0).to_vec().unwrap(),
            select_f32: select(&x.gt(0.0), &doubled, &halved).to_vec().unwrap(),
            select_u32: select(&u.gt(500u32), &(&u / 7u32), &(&u * 3u32)).to_vec().unwrap(),
            select_i32: select(&i.lt(0), &(&i * -2), &(&i + 5)).to_vec().unwrap(),
        };
        let who = label(kind);
        must(who, "tanh", first_far(&result.tanh, &want_tanh, 2e-5, 2e-6));
        must(who, "fused tanh", first_far(&result.fused_tanh, &want_fused, 2e-5, 1e-5));
        // Beyond |x| = 15 the result is exactly +-1 whatever the device's tanh does.
        for (k, (&x, &t)) in xs.iter().zip(&result.tanh).enumerate() {
            if x.abs() > 15.0 {
                assert_eq!(t, x.signum(), "{who}: tanh({x}) at {k}");
            }
        }
        must(who, "select f32", first_diff(&result.select_f32, &want_select_f32));
        must(who, "select u32", first_diff(&result.select_u32, &want_select_u32));
        must(who, "select i32", first_diff(&result.select_i32, &want_select_i32));
        results.push((kind, result));
    }
    agree("elementwise", &results, |a, b| {
        first_far(&a.tanh, &b.tanh, 2e-5, 2e-6)?;
        first_far(&a.fused_tanh, &b.fused_tanh, 2e-5, 1e-5)?;
        first_diff(&a.select_f32, &b.select_f32)?;
        first_diff(&a.select_u32, &b.select_u32)?;
        first_diff(&a.select_i32, &b.select_i32)
    });
}

#[test]
fn select_on_vec4_values_in_a_custom_kernel() {
    const SRC: &str = r#"
    @group(1) @binding(0) var<storage, read> a: array<vec4<f32>>;
    @group(1) @binding(1) var<storage, read> b: array<vec4<f32>>;
    @group(1) @binding(2) var<storage, read_write> by_scalar: array<vec4<f32>>;
    @group(1) @binding(3) var<storage, read_write> by_lane: array<vec4<f32>>;
    @compute @workgroup_size(32)
    fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
        let i = gid.x;
        if (i >= luma.size.x) { return; }
        let x = a[i];
        let y = b[i];
        by_scalar[i] = select(y, x, x.x + x.w > y.x);
        by_lane[i] = select(y, x, x > y);
    }"#;
    let n = 100usize;
    let flat = values(n * 4, 21, -1.0, 1.0);
    let a: Vec<[f32; 4]> = flat.chunks(4).map(|c| [c[0], c[1], c[2], c[3]]).collect();
    // Some lanes tie with `a` (the comparison is strict), the others are independent.
    let b: Vec<[f32; 4]> = values(n * 4, 22, -1.0, 1.0)
        .chunks(4)
        .enumerate()
        .map(|(i, c)| [c[0], if i % 3 == 0 { a[i][1] } else { c[1] }, c[2], if i % 5 == 0 { a[i][3] } else { c[3] }])
        .collect();
    let want_scalar: Vec<[f32; 4]> = (0..n).map(|i| if a[i][0] + a[i][3] > b[i][0] { a[i] } else { b[i] }).collect();
    let want_lane: Vec<[f32; 4]> = (0..n).map(|i| std::array::from_fn(|c| if a[i][c] > b[i][c] { a[i][c] } else { b[i][c] })).collect();
    for (kind, ctx) in backends() {
        let (ga, gb) = (on(&ctx, &a), on(&ctx, &b));
        let mut by_scalar = Array::<[f32; 4]>::zeros_on(&ctx, n).unwrap();
        let mut by_lane = Array::<[f32; 4]>::zeros_on(&ctx, n).unwrap();
        let kernel = ctx.kernel(SRC).label("backends_select_vec4").build().unwrap();
        kernel.launch([n as u32], args![&ga, &gb, &mut by_scalar, &mut by_lane]).unwrap();
        must(label(kind), "scalar condition", first_diff(&by_scalar.to_vec().unwrap(), &want_scalar));
        must(label(kind), "lane mask", first_diff(&by_lane.to_vec().unwrap(), &want_lane));
    }
}

// -- Wide indices --

/// An array of more than `2^23` elements divides its flat indices by the extents with the
/// high word of a product instead of an `f32` quotient: `range` along the second axis is
/// `flat / d`, along the first `flat % d`, and a transposed read divides again.
fn wide_division(kind: BackendKind, ctx: &Context, dims: [usize; 2]) {
    let (d, w) = (dims[0], dims[1]);
    let len = d * w;
    assert!(len > 1 << 23, "{dims:?} is not a wide array");
    let who = label(kind);
    let q = Array::<u32>::range_on(ctx, dims, 1).unwrap();
    let r = Array::<u32>::range_on(ctx, dims, 0).unwrap();
    let joined = &q * (d as u32) + &r;
    let transposed = q.transpose();
    let got_q = q.to_vec().unwrap();
    if let Some(k) = got_q.iter().enumerate().position(|(k, &v)| v as usize != k / d) {
        panic!("{who} {dims:?}: quotient of {k}: got {}, want {}", got_q[k], k / d);
    }
    let got_r = r.to_vec().unwrap();
    if let Some(k) = got_r.iter().enumerate().position(|(k, &v)| v as usize != k % d) {
        panic!("{who} {dims:?}: remainder of {k}: got {}, want {}", got_r[k], k % d);
    }
    let got_joined = joined.to_vec().unwrap();
    if let Some(k) = got_joined.iter().enumerate().position(|(k, &v)| v as usize != k) {
        panic!("{who} {dims:?}: quotient * {d} + remainder of {k} is {}", got_joined[k]);
    }
    // Element (j, i) of the transpose is `q[i, j] = j`, at flat index j + w * i.
    let got_t = transposed.to_vec().unwrap();
    if let Some(k) = got_t.iter().enumerate().position(|(k, &v)| v as usize != k % w) {
        panic!("{who} {dims:?}: transposed element {k}: got {}, want {}", got_t[k], k % w);
    }
}

#[test]
fn wide_index_division_past_2_pow_23_is_exact() {
    for (kind, ctx) in backends() {
        // Just above the 2^23 boundary, with the extents 3 and 5 (not powers of two).
        wide_division(kind, &ctx, [3, 2_796_203]);
        wide_division(kind, &ctx, [5, 1_677_722]);
    }
}

// -- Df32 --

fn df_values(n: usize, seed: u64, lo: f32, hi: f32) -> Vec<Df32> {
    values(n, seed, lo, hi).iter().enumerate().map(|(i, &v)| Df32::from(f64::from(v) + (i as f64 + 1.0) * 1.0e-9)).collect()
}

fn near(got: f64, want: f64, rel: f64) -> Result<(), String> {
    if (got - want).abs() <= rel * want.abs() + 1e-30 { Ok(()) } else { Err(format!("got {got:e}, want {want:e} (relative {rel:e})")) }
}

#[test]
fn df32_residuals_and_arithmetic_do_not_depend_on_the_backend() {
    let n = 257;
    let xs = df_values(n, 41, -3.0, 3.0);
    let ys = df_values(n, 42, -3.0, 3.0);
    let ps = df_values(n, 43, 0.25, 4.0);
    let qs = df_values(n, 44, 0.25, 4.0);
    let mut results = Vec::new();
    for (kind, ctx) in backends() {
        let who = label(kind);
        // 1 + 1e-10 does not fit an f32; its double-float keeps the 1e-10 in the low word.
        let one = Array::from_slice_on(&ctx, &[Df32 { hi: 1.0, lo: 0.0 }; 4], 4).unwrap();
        let tiny = Array::from_slice_on(&ctx, &[Df32 { hi: 1e-10, lo: 0.0 }; 4], 4).unwrap();
        let sum = &one + &tiny;
        let residual = (&sum - sum.to_f32().to_df32()).to_f32().to_vec().unwrap();
        assert_eq!(residual, vec![1e-10f32; 4], "{who}: (x - x.to_f32().to_df32()).to_f32()");
        assert_eq!(sum.to_f32().to_vec().unwrap(), vec![1.0f32; 4], "{who}: the rounded sum");

        let (x, y, p, q) = (on(&ctx, &xs), on(&ctx, &ys), on(&ctx, &ps), on(&ctx, &qs));
        let low = (&x - x.to_f32().to_df32()).to_f32().to_vec().unwrap();
        let want_low: Vec<f32> = xs.iter().map(|d| d.lo).collect();
        assert_eq!(low, want_low, "{who}: the low word of a double-float survives fusion");

        let added = (&x + &y).to_vec().unwrap();
        let multiplied = (&x * &y).to_vec().unwrap();
        let exps = x.exp().to_vec().unwrap();
        let logs = (&p * &q).ln().to_vec().unwrap();
        for k in 0..n {
            let (a, b) = (xs[k].to_f64(), ys[k].to_f64());
            let want_add = a + b;
            if (added[k].to_f64() - want_add).abs() > 1e-12 * (a.abs() + b.abs()) + 1e-30 {
                panic!("{who}: add at {k}: got {:?}, want {want_add:e}", added[k]);
            }
            must(who, &format!("mul at {k}"), near(multiplied[k].to_f64(), a * b, 1e-12));
            must(who, &format!("exp at {k}"), near(exps[k].to_f64(), a.exp(), 1e-11));
            let (pp, qq) = (ps[k].to_f64(), qs[k].to_f64());
            must(who, &format!("ln at {k}"), near(logs[k].to_f64(), (pp * qq).ln(), 1e-11));
        }
        results.push((kind, (added, multiplied, exps, logs)));
    }
    // The accuracy is a precision contract (about 48 bits), not a canonical representation: the
    // backends may differ in the low word, so the results are compared by value.
    agree("df32", &results, |a, b| {
        for (k, (u, v)) in a.0.iter().zip(&b.0).enumerate() {
            let scale = xs[k].to_f64().abs() + ys[k].to_f64().abs();
            if (u.to_f64() - v.to_f64()).abs() > 2e-12 * scale + 1e-30 {
                return Err(format!("add at {k}: {u:?} vs {v:?}"));
            }
        }
        for (name, x, y) in [("mul", &a.1, &b.1), ("exp", &a.2, &b.2), ("ln", &a.3, &b.3)] {
            for (k, (u, v)) in x.iter().zip(y).enumerate() {
                near(u.to_f64(), v.to_f64(), 2e-11).map_err(|e| format!("{name} at {k}: {e}"))?;
            }
        }
        Ok(())
    });
}

// -- Stencils and views --

fn wrap(i: usize, offset: isize, n: usize) -> usize {
    (i as isize + offset).rem_euclid(n as isize) as usize
}

#[test]
fn staged_2d_stencils_match_the_host() {
    let (w, h) = (70usize, 50usize);
    let a = values(w * h, 51, -1.0, 1.0);
    let b = values(w * h, 52, -1.0, 1.0);
    let e: Vec<f32> = a.iter().zip(&b).map(|(x, y)| x * y + 0.5).collect();
    let at = |x: usize, y: usize| e[x + w * y];
    let mut want_cross = vec![0.0f32; w * h];
    let mut want_far = vec![0.0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            want_cross[x + w * y] = at(wrap(x, -1, w), y) + at(wrap(x, 1, w), y) + at(x, wrap(y, -1, h)) + at(x, wrap(y, 1, h)) - at(x, y) * 4.0;
            // `shift([2, -1])` reads the element at (x - 2, y + 1).
            want_far[x + w * y] = (at(wrap(x, -2, w), wrap(y, 1, h)) + at(wrap(x, 1, w), wrap(y, -2, h))) * 0.5 - at(x, y) * 0.25;
        }
    }
    let mut results = Vec::new();
    for (kind, ctx) in backends() {
        let (ga, gb) = (Array::from_slice_on(&ctx, &a, [w, h]).unwrap(), Array::from_slice_on(&ctx, &b, [w, h]).unwrap());
        // The lazy expression is read at shifted cells: the kernel recomputes or stages it.
        let lazy = &ga * &gb + 0.5;
        let cross = lazy.shift([1, 0]) + lazy.shift([-1, 0]) + lazy.shift([0, 1]) + lazy.shift([0, -1]) - &lazy * 4.0;
        let far = (lazy.shift([2, -1]) + lazy.shift([-1, 2])) * 0.5 - &lazy * 0.25;
        let (cross, far) = (cross.to_vec().unwrap(), far.to_vec().unwrap());
        must(label(kind), "cross stencil", first_far(&cross, &want_cross, 1e-5, 1e-5));
        must(label(kind), "far stencil", first_far(&far, &want_far, 1e-5, 1e-5));
        // The same stencil over an evaluated array (reads the buffer instead of recomputing).
        let cached = (&ga * &gb + 0.5).cache();
        let cross_cached = (cached.shift([1, 0]) + cached.shift([-1, 0]) + cached.shift([0, 1]) + cached.shift([0, -1]) - &cached * 4.0).to_vec().unwrap();
        must(label(kind), "cached cross stencil", first_far(&cross_cached, &want_cross, 1e-5, 1e-5));
        results.push((kind, (cross, far)));
    }
    agree("stencils", &results, |x, y| {
        first_far(&x.0, &y.0, 1e-5, 1e-5)?;
        first_far(&x.1, &y.1, 1e-5, 1e-5)
    });
}

// -- Packed booleans --

const WORDS: &str = r#"
@group(1) @binding(0) var<storage, read> src: array<u32>;
@group(1) @binding(1) var<storage, read_write> dst: array<u32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= luma.size.x) { return; }
    dst[gid.x] = src[gid.x];
}
"#;

fn bits_of(x: &[f32]) -> Vec<bool> {
    x.iter().map(|&v| v > 0.0).collect()
}

fn pack(bits: &[bool]) -> Vec<u32> {
    bits.chunks(32).map(|chunk| chunk.iter().enumerate().fold(0u32, |word, (i, &b)| word | u32::from(b) << i)).collect()
}

/// `m` holds exactly `want`, as the host and a custom kernel (which sees the packed words) read it.
fn check_bits(who: &str, ctx: &Context, m: &Array<bool>, want: &[bool], what: &str) {
    assert_eq!(m.len(), want.len(), "{who} {what}: length");
    assert_eq!(m.to_vec().unwrap(), want, "{who} {what}: to_vec");
    let count = m.len().div_ceil(32);
    let mut out = Array::<u32>::zeros_on(ctx, count).unwrap();
    ctx.kernel(WORDS).build().unwrap().launch([count as u32], args![m, &mut out]).unwrap();
    assert_eq!(out.to_vec().unwrap(), pack(want), "{who} {what}: words");
    let set = want.iter().filter(|&&b| b).count();
    assert_eq!(m.count().item().unwrap() as usize, set, "{who} {what}: count");
}

fn region_of<'a>(base: &'a mut Array<bool>, xs: &std::ops::Range<usize>, ys: &std::ops::Range<usize>, flip_rows: bool) -> ViewMut<'a, Array<bool>> {
    let view = base.slice_mut(0, xs.clone()).slice(1, ys.clone());
    if flip_rows { view.flip(1) } else { view }
}

#[test]
fn packed_bool_region_writes_keep_the_bits_around_them() {
    for (kind, ctx) in backends() {
        let who = label(kind);
        // Rows at every alignment of the 32-bit words: runs of whole rows, rows shorter than a
        // word, rows that start inside a word, flipped rows.
        for (w, h, xs, ys) in [(100usize, 7usize, 3..99, 0..7), (96, 9, 1..96, 0..9), (130, 70, 1..129, 1..69), (200, 5, 31..95, 0..5), (40, 40, 3..37, 2..30), (96, 9, 0..96, 2..8)] {
            for flip_rows in [false, true] {
                let tag = format!("{w}x{h} columns {xs:?} rows {ys:?} flipped={flip_rows}");
                let want = bits_of(&values(w * h, (w * 31 + h) as u64, -1.0, 1.0));
                let (rw, rh) = (xs.len(), ys.len());
                let source = values(rw * rh, 5 + (w + h) as u64, -1.0, 1.0);
                let new = bits_of(&source);
                let mut model = want.clone();
                for y in 0..rh {
                    let row = if flip_rows { ys.start + rh - 1 - y } else { ys.start + y };
                    for x in 0..rw {
                        model[xs.start + x + w * row] = new[x + rw * y];
                    }
                }
                let mut base = Array::from_slice_on(&ctx, &want, [w, h]).unwrap();
                let rhs = Array::from_slice_on(&ctx, &new, [rw, rh]).unwrap();
                region_of(&mut base, &xs, &ys, flip_rows).assign(&rhs).unwrap();
                check_bits(who, &ctx, &base.reshape(w * h), &model, &format!("{tag}: assign of an array"));
                // The same, from a comparison that is computed inside the write.
                let mut again = Array::from_slice_on(&ctx, &want, [w, h]).unwrap();
                let pending = Array::from_slice_on(&ctx, &source, [rw, rh]).unwrap().gt(0.0);
                region_of(&mut again, &xs, &ys, flip_rows).assign(&pending).unwrap();
                check_bits(who, &ctx, &again.reshape(w * h), &model, &format!("{tag}: assign of a comparison"));
                // Constants: every word of a run is all ones or all zeros.
                for value in [true, false] {
                    let mut constant = Array::from_slice_on(&ctx, &want, [w, h]).unwrap();
                    region_of(&mut constant, &xs, &ys, flip_rows).assign(Array::full_on(&ctx, [rw, rh], value).unwrap()).unwrap();
                    let mut expect = want.clone();
                    for y in ys.clone() {
                        expect[xs.start + w * y..xs.end + w * y].fill(value);
                    }
                    check_bits(who, &ctx, &constant.reshape(w * h), &expect, &format!("{tag}: assign of {value}"));
                }
            }
        }
    }
}

// -- Scatter-add --

#[test]
fn scatter_add_of_f32_u32_and_i32_is_exact() {
    let n = 1 << 16;
    for (kind, ctx) in backends() {
        let who = label(kind);
        for bins in [64usize, 4096] {
            let mut rng = Rng(7 + bins as u64);
            // One index in 16 is out of range and is skipped.
            let codes: Vec<u32> = (0..n).map(|_| if rng.next().is_multiple_of(16) { bins as u32 + rng.next() % 1000 } else { rng.next() % bins as u32 }).collect();
            // Multiples of 1/16: every partial sum is exact in f32, whatever the order of the atomics.
            let fs: Vec<f32> = (0..n).map(|_| (rng.next() % 257) as f32 / 16.0 - 8.0).collect();
            let us: Vec<u32> = (0..n).map(|_| rng.next() % 1000).collect();
            let is: Vec<i32> = (0..n).map(|_| (rng.next() % 2001) as i32 - 1000).collect();
            let (mut want_f, mut want_u, mut want_i) = (vec![0.0f32; bins], vec![0u32; bins], vec![0i32; bins]);
            for k in 0..n {
                let c = codes[k] as usize;
                if c < bins {
                    want_f[c] += fs[k];
                    want_u[c] += us[k];
                    want_i[c] += is[k];
                }
            }
            let indices = on(&ctx, &codes);
            let mut target_f = Array::<f32>::zeros_on(&ctx, bins).unwrap();
            target_f.put_add(&indices, on(&ctx, &fs)).unwrap();
            let mut target_u = Array::<u32>::zeros_on(&ctx, bins).unwrap();
            target_u.put_add(&indices, on(&ctx, &us)).unwrap();
            let mut target_i = Array::<i32>::zeros_on(&ctx, bins).unwrap();
            target_i.put_add(&indices, on(&ctx, &is)).unwrap();
            must(who, &format!("f32 put_add, {bins} bins"), first_diff(&target_f.to_vec().unwrap(), &want_f));
            must(who, &format!("u32 put_add, {bins} bins"), first_diff(&target_u.to_vec().unwrap(), &want_u));
            must(who, &format!("i32 put_add, {bins} bins"), first_diff(&target_i.to_vec().unwrap(), &want_i));
            // Counting with the constant 1 is a bincount.
            let mut counts = Array::<u32>::zeros_on(&ctx, bins).unwrap();
            counts.put_add(&indices, 1u32).unwrap();
            let mut want_counts = vec![0u32; bins];
            codes.iter().filter(|&&c| (c as usize) < bins).for_each(|&c| want_counts[c as usize] += 1);
            must(who, &format!("put_add of 1, {bins} bins"), first_diff(&counts.to_vec().unwrap(), &want_counts));
        }
    }
}

// -- Reductions, scans, arg reductions --

/// The cell indices of every line along `axis` of an array of `dims` (axis 0 varies fastest),
/// in the order of the result of a reduction along `axis`.
fn host_lines(dims: [usize; 3], axis: usize) -> Vec<Vec<usize>> {
    let stride: usize = dims[..axis].iter().product();
    let lines = dims.iter().product::<usize>() / dims[axis];
    (0..lines).map(|l| (0..dims[axis]).map(|k| (l % stride) + (l / stride) * stride * dims[axis] + k * stride).collect()).collect()
}

fn host_sum<T: Copy + std::ops::Add<Output = T>>(x: &[T], lines: &[Vec<usize>]) -> Vec<T> {
    lines.iter().map(|cells| cells[1..].iter().fold(x[cells[0]], |sum, &c| sum + x[c])).collect()
}

fn host_cumsum<T: Copy + std::ops::Add<Output = T>>(x: &[T], lines: &[Vec<usize>]) -> Vec<T> {
    let mut out = x.to_vec();
    for cells in lines {
        for k in 1..cells.len() {
            out[cells[k]] = out[cells[k - 1]] + x[cells[k]];
        }
    }
    out
}

fn host_extreme<T: Copy + PartialOrd>(x: &[T], lines: &[Vec<usize>], max: bool) -> Vec<T> {
    lines.iter().map(|cells| cells[1..].iter().fold(x[cells[0]], |best, &c| if (max && x[c] > best) || (!max && x[c] < best) { x[c] } else { best })).collect()
}

/// The first of the extreme elements of every line (NumPy's `argmax`).
fn host_arg<T: Copy + PartialOrd>(x: &[T], lines: &[Vec<usize>], max: bool) -> Vec<u32> {
    lines
        .iter()
        .map(|cells| {
            let mut best = 0;
            for k in 1..cells.len() {
                if (max && x[cells[k]] > x[cells[best]]) || (!max && x[cells[k]] < x[cells[best]]) {
                    best = k;
                }
            }
            best as u32
        })
        .collect()
}

#[test]
fn reductions_scans_and_arg_reductions_are_exact() {
    let dims = [97usize, 29, 3];
    let len: usize = dims.iter().product();
    let mut rng = Rng(77);
    // Small integers: every sum is exact in f32, so all backends must agree bit for bit.
    let fs: Vec<f32> = (0..len).map(|_| (rng.next() % 17) as f32 - 8.0).collect();
    let us: Vec<u32> = (0..len).map(|_| rng.next() % 100).collect();
    let is: Vec<i32> = (0..len).map(|_| (rng.next() % 41) as i32 - 20).collect();
    let big: Vec<f32> = (0..300_001).map(|_| (rng.next() % 7) as f32 - 3.0).collect();
    let all: Vec<Vec<usize>> = vec![(0..len).collect()];
    let all_big: Vec<Vec<usize>> = vec![(0..big.len()).collect()];

    for (kind, ctx) in backends() {
        let who = label(kind);
        let x = Array::from_slice_on(&ctx, &fs, dims).unwrap();
        let [sum, min, max, squares] = x.reduce([Reduction::Sum, Reduction::Min, Reduction::Max, Reduction::SumOfSquares]).to_array::<4>().unwrap();
        assert_eq!(sum, host_sum(&fs, &all)[0], "{who}: sum");
        assert_eq!(min, host_extreme(&fs, &all, false)[0], "{who}: min");
        assert_eq!(max, host_extreme(&fs, &all, true)[0], "{who}: max");
        assert_eq!(squares, fs.iter().map(|v| v * v).sum::<f32>(), "{who}: sum of squares");
        assert_eq!(x.sum().item().unwrap(), sum, "{who}: sum()");
        assert_eq!(x.argmax().item().unwrap(), host_arg(&fs, &all, true)[0], "{who}: argmax");
        assert_eq!(x.argmin().item().unwrap(), host_arg(&fs, &all, false)[0], "{who}: argmin");

        let xu = Array::from_slice_on(&ctx, &us, dims).unwrap();
        let xi = Array::from_slice_on(&ctx, &is, dims).unwrap();
        for axis in 0..3 {
            let lines = host_lines(dims, axis);
            let what = |name: &str| format!("{name} along axis {axis}");
            must(who, &what("sum_axis"), first_diff(&x.sum_axis(axis).to_vec().unwrap(), &host_sum(&fs, &lines)));
            must(who, &what("cumsum"), first_diff(&x.cumsum(axis).to_vec().unwrap(), &host_cumsum(&fs, &lines)));
            let [mn, mx] = x.reduce_axis(axis, [Reduction::Min, Reduction::Max]);
            must(who, &what("min of reduce_axis"), first_diff(&mn.to_vec().unwrap(), &host_extreme(&fs, &lines, false)));
            must(who, &what("max of reduce_axis"), first_diff(&mx.to_vec().unwrap(), &host_extreme(&fs, &lines, true)));
            must(who, &what("argmax_axis"), first_diff(&x.argmax_axis(axis).to_vec().unwrap(), &host_arg(&fs, &lines, true)));
            must(who, &what("argmin_axis"), first_diff(&x.argmin_axis(axis).to_vec().unwrap(), &host_arg(&fs, &lines, false)));
            must(who, &what("u32 cumsum"), first_diff(&xu.cumsum(axis).to_vec().unwrap(), &host_cumsum(&us, &lines)));
            must(who, &what("u32 sum_axis"), first_diff(&xu.sum_axis(axis).to_vec().unwrap(), &host_sum(&us, &lines)));
            must(who, &what("i32 argmax_axis"), first_diff(&xi.argmax_axis(axis).to_vec().unwrap(), &host_arg(&is, &lines, true)));
            must(who, &what("i32 argmin_axis"), first_diff(&xi.argmin_axis(axis).to_vec().unwrap(), &host_arg(&is, &lines, false)));
        }

        // One long line: several passes, with ties everywhere.
        let b = on(&ctx, &big);
        assert_eq!(b.sum().item().unwrap(), host_sum(&big, &all_big)[0], "{who}: long sum");
        assert_eq!(b.max().item().unwrap(), host_extreme(&big, &all_big, true)[0], "{who}: long max");
        assert_eq!(b.argmax().item().unwrap(), host_arg(&big, &all_big, true)[0], "{who}: long argmax");
        assert_eq!(b.argmin().item().unwrap(), host_arg(&big, &all_big, false)[0], "{who}: long argmin");
        must(who, "long cumsum", first_diff(&b.cumsum(0).to_vec().unwrap(), &host_cumsum(&big, &all_big)));
    }
}

// -- Sorting --

/// Sorts `data` ascending and descending, values and indices, as the stable host sort by `key`
/// does; `bits` compares values bit for bit (`-0.0` is not `+0.0`).
fn check_sorts<T: SortKey>(who: &str, ctx: &Context, data: &[T], key: impl Fn(T) -> u32, bits: impl Fn(T) -> u32) {
    let n = data.len();
    let x = on(ctx, data);
    let mut ascending: Vec<usize> = (0..n).collect();
    ascending.sort_by_key(|&i| key(data[i]));
    let mut descending: Vec<usize> = (0..n).collect();
    descending.sort_by_key(|&i| std::cmp::Reverse(key(data[i])));
    let as_bits = |v: Vec<T>| -> Vec<u32> { v.into_iter().map(&bits).collect() };
    let by = |order: &[usize]| -> Vec<u32> { order.iter().map(|&i| bits(data[i])).collect() };
    let indices = |order: &[usize]| -> Vec<u32> { order.iter().map(|&i| i as u32).collect() };
    must(who, &format!("sort of {n}"), first_diff(&as_bits(x.sort().to_vec().unwrap()), &by(&ascending)));
    must(who, &format!("sort_descending of {n}"), first_diff(&as_bits(x.sort_descending().to_vec().unwrap()), &by(&descending)));
    must(who, &format!("argsort of {n}"), first_diff(&x.argsort().to_vec().unwrap(), &indices(&ascending)));
    must(who, &format!("argsort_descending of {n}"), first_diff(&x.argsort_descending().to_vec().unwrap(), &indices(&descending)));
}

#[test]
fn sort_and_argsort_are_stable_and_agree_with_the_host() {
    const SPECIAL: [f32; 6] = [0.0, -0.0, f32::INFINITY, f32::NEG_INFINITY, f32::MAX, f32::MIN_POSITIVE];
    for (kind, ctx) in backends() {
        let who = label(kind);
        // Lengths around the 32-lane subgroups, with many equal keys (stability matters).
        for n in [1usize, 31, 32, 33, 1000, 70_001] {
            let mut rng = Rng(n as u64);
            let us: Vec<u32> = (0..n).map(|_| rng.next() % 1000).collect();
            let is: Vec<i32> = (0..n).map(|_| (rng.next() % 2001) as i32 - 1000).collect();
            let fs: Vec<f32> = (0..n)
                .map(|i| if i % 13 == 5 { SPECIAL[rng.next() as usize % SPECIAL.len()] } else { ((rng.next() % 2001) as f32 - 1000.0) * 2f32.powi((rng.next() % 9) as i32 - 4) })
                .collect();
            check_sorts(who, &ctx, &us, |v| v, |v| v);
            check_sorts(who, &ctx, &is, |v| v as u32 ^ 0x8000_0000, |v| v as u32);
            check_sorts(who, &ctx, &fs, |v| if v.is_sign_negative() { !v.to_bits() } else { v.to_bits() | 0x8000_0000 }, f32::to_bits);
        }
        // Every line along an axis on its own.
        let (w, h) = (37usize, 5usize);
        let mut rng = Rng(5);
        let data: Vec<u32> = (0..w * h).map(|_| rng.next() % 50).collect();
        let sorted = Array::from_slice_on(&ctx, &data, [w, h]).unwrap().sort_axis(0).to_vec().unwrap();
        let want: Vec<u32> = data.chunks(w).flat_map(|line| {
            let mut line = line.to_vec();
            line.sort();
            line
        }).collect();
        must(who, "sort_axis", first_diff(&sorted, &want));
    }
}

// -- Convolutions --

fn conv_host(input: &[f32], kernel: &[f32], stride: usize, padding: usize) -> Vec<f32> {
    let len = (input.len() + 2 * padding - kernel.len()) / stride + 1;
    (0..len)
        .map(|j| {
            (0..kernel.len())
                .map(|t| {
                    let at = (j * stride + t) as isize - padding as isize;
                    if (0..input.len() as isize).contains(&at) { f64::from(input[at as usize]) * f64::from(kernel[t]) } else { 0.0 }
                })
                .sum::<f64>() as f32
        })
        .collect()
}

/// The full linear convolution (the kernel is flipped), of length `n + k - 1`.
fn full_conv_host(input: &[f32], kernel: &[f32]) -> Vec<f32> {
    (0..input.len() + kernel.len() - 1)
        .map(|j| (0..kernel.len()).filter(|&t| j >= t && j - t < input.len()).map(|t| f64::from(input[j - t]) * f64::from(kernel[t])).sum::<f64>() as f32)
        .collect()
}

#[test]
fn direct_and_fft_convolutions_match_the_host() {
    for (kind, ctx) in backends() {
        let who = label(kind);
        // (input length, kernel length, stride, padding): short and long kernels, strides, padding.
        for (n, k, stride, padding) in [(1000usize, 7usize, 1usize, 3usize), (777, 33, 2, 0), (1024, 129, 1, 64), (500, 5, 3, 2), (64, 64, 1, 0)] {
            let input = values(n, 90 + n as u64, -1.0, 1.0);
            let kernel = values(k, 91 + k as u64, -1.0, 1.0);
            let got = on(&ctx, &input).conv1d(&on(&ctx, &kernel), stride, padding).to_vec().unwrap();
            must(who, &format!("conv1d n={n} k={k} stride={stride} pad={padding}"), first_far(&got, &conv_host(&input, &kernel, stride, padding), 1e-4, 1e-4));
        }
        for (n, k) in [(1000usize, 33usize), (3000, 257), (100, 100), (1, 1)] {
            let input = values(n, 95 + n as u64, -1.0, 1.0);
            let kernel = values(k, 96 + k as u64, -1.0, 1.0);
            let result = on(&ctx, &input).conv1d_fft(&on(&ctx, &kernel));
            let got = result.to_vec().unwrap();
            must(who, &format!("conv1d_fft n={n} k={k}"), first_far(&got, &full_conv_host(&input, &kernel), 1e-3, 5e-3));
        }
        // Both orientations in one: a symmetric kernel gives the direct convolution inside the full one.
        let input = values(300, 97, -1.0, 1.0);
        let kernel = [0.1f32, 0.2, 0.4, 0.2, 0.1];
        let spatial = on(&ctx, &input).conv1d(&on(&ctx, &kernel), 1, 0).to_vec().unwrap();
        let full = on(&ctx, &input).conv1d_fft(&on(&ctx, &kernel)).to_vec().unwrap();
        must(who, "symmetric kernel: direct vs fft", first_far(&spatial, &full[4..4 + spatial.len()], 1e-3, 2e-3));
    }
}

// -- Matrix products --

/// `a_t` is `m x k` stored with `m` fastest (`a_t[m + M * k]`), `b` `n x k` (`b[n + N * k]`); the
/// product `out[n + N * m] = sum_k a_t[m + M * k] * b[n + N * k]`.
fn matmul_host(a_t: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n * m];
    for i in 0..m {
        for j in 0..n {
            out[j + n * i] = (0..k).map(|t| f64::from(a_t[i + m * t]) * f64::from(b[j + n * t])).sum::<f64>() as f32;
        }
    }
    out
}

#[test]
fn matmul_with_a_fused_transpose_matches_the_host() {
    for (kind, ctx) in backends() {
        let who = label(kind);
        // Ragged tiles, aligned tiles, and a long K that is split over workgroups.
        for (m, k, n) in [(17usize, 33usize, 19usize), (64, 64, 64), (8, 2048, 8), (130, 70, 50)] {
            let a_t = values(m * k, 100 + m as u64, -1.0, 1.0);
            let b = values(n * k, 101 + n as u64, -1.0, 1.0);
            let abs = 2e-5 * (k as f32).sqrt();
            let lhs = Array::from_slice_on(&ctx, &a_t, [m, k]).unwrap();
            let rhs = Array::from_slice_on(&ctx, &b, [n, k]).unwrap();
            // The transpose is read by the kernel's loads, not evaluated first.
            let got = lhs.transpose().matmul(&rhs).to_vec().unwrap();
            must(who, &format!("{m}x{k}x{n}"), first_far(&got, &matmul_host(&a_t, &b, m, k, n), 1e-4, abs));
            // Lazy operands are fused into the loads too.
            let scaled = (&lhs * 0.5).transpose();
            let lazy = scaled.matmul(&(&rhs * 1.5)).to_vec().unwrap();
            let half: Vec<f32> = a_t.iter().map(|v| v * 0.5).collect();
            let wide: Vec<f32> = b.iter().map(|v| v * 1.5).collect();
            must(who, &format!("{m}x{k}x{n} lazy"), first_far(&lazy, &matmul_host(&half, &wide, m, k, n), 1e-4, abs));
        }
    }
}

// -- Block tridiagonal solves on many bindings --

fn dense_solve(mut m: Vec<f64>, mut r: Vec<f64>, n: usize) -> Vec<f64> {
    for col in 0..n {
        let pivot = (col..n).max_by(|&i, &j| m[i * n + col].abs().total_cmp(&m[j * n + col].abs())).unwrap();
        for j in 0..n {
            m.swap(col * n + j, pivot * n + j);
        }
        r.swap(col, pivot);
        for i in col + 1..n {
            let f = m[i * n + col] / m[col * n + col];
            for j in col..n {
                m[i * n + j] -= f * m[col * n + j];
            }
            r[i] -= f * r[col];
        }
    }
    for i in (0..n).rev() {
        let mut x = r[i];
        for j in i + 1..n {
            x -= m[i * n + j] * r[j];
        }
        r[i] = x / m[i * n + i];
    }
    r
}

/// The solution components `out[c][cell]` of `lines` block systems of `n` cells along axis 0
/// (cell `s` of line `l` is `l * n + s`), blocks of `b x b`; `lo`, `diag`, `up` hold `b * b`
/// values per cell and `rhs` `b`.
fn host_block_solve(b: usize, n: usize, lines: usize, lo: &[f32], diag: &[f32], up: &[f32], rhs: &[f32]) -> Vec<Vec<f32>> {
    let bb = b * b;
    let size = n * b;
    let mut out = vec![vec![0.0f32; n * lines]; b];
    for l in 0..lines {
        let mut matrix = vec![0.0f64; size * size];
        let mut r = vec![0.0f64; size];
        for s in 0..n {
            let q = l * n + s;
            for i in 0..b {
                r[s * b + i] = f64::from(rhs[q * b + i]);
                for j in 0..b {
                    matrix[(s * b + i) * size + s * b + j] = f64::from(diag[q * bb + i * b + j]);
                    if s > 0 {
                        matrix[(s * b + i) * size + (s - 1) * b + j] = f64::from(lo[q * bb + i * b + j]);
                    }
                    if s + 1 < n {
                        matrix[(s * b + i) * size + (s + 1) * b + j] = f64::from(up[q * bb + i * b + j]);
                    }
                }
            }
        }
        let solution = dense_solve(matrix, r, size);
        for s in 0..n {
            for i in 0..b {
                out[i][l * n + s] = solution[s * b + i] as f32;
            }
        }
    }
    out
}

/// One 4 x 4 block system whose 52 coefficient parts are each the sum of two distinct storage
/// arrays: 104 input buffers in one fused kernel, as many as the public API can bind for a
/// single system (the solve of several block systems in one launch is internal). On a backend
/// that binds that many buffers (Metal argument buffers, CUDA pointers) the solve is one launch;
/// under WebGPU's binding cap it is split or stacked into several.
#[test]
fn a_block_solve_over_104_input_bindings_is_one_launch_where_the_backend_binds_them() {
    let (b, n, lines) = (4usize, 24usize, 3usize);
    let cells = n * lines;
    let widths = [16usize, 16, 16, 4];
    let mut first: Vec<Vec<f32>> = Vec::new();
    let mut second: Vec<Vec<f32>> = Vec::new();
    let mut host: Vec<Vec<f32>> = widths.iter().map(|&w| vec![0.0f32; cells * w]).collect();
    for k in 0..52usize {
        let (which, e) = match k {
            0..16 => (0, k),
            16..32 => (1, k - 16),
            32..48 => (2, k - 32),
            _ => (3, k - 48),
        };
        let mut a = values(cells, 300 + k as u64, -1.0, 1.0);
        let c = values(cells, 400 + k as u64, -1.0, 1.0);
        // Diagonal dominance: the diagonal of every diagonal block is about 8.
        if which == 1 && e / b == e % b {
            a.iter_mut().for_each(|v| *v += 8.0);
        }
        for q in 0..cells {
            host[which][q * widths[which] + e] = a[q] + c[q];
        }
        first.push(a);
        second.push(c);
    }
    let want = host_block_solve(b, n, lines, &host[0], &host[1], &host[2], &host[3]);

    let mut results = Vec::new();
    for (kind, ctx) in backends() {
        let who = label(kind);
        let a: Vec<Array<f32>> = first.iter().map(|d| Array::from_slice_on(&ctx, d, [n, lines]).unwrap()).collect();
        let d: Vec<Array<f32>> = second.iter().map(|v| Array::from_slice_on(&ctx, v, [n, lines]).unwrap()).collect();
        // Lazy sums: the solve kernel computes them, reading both buffers of every part.
        let parts: Vec<Array<f32>> = a.iter().zip(&d).map(|(x, y)| x + y).collect();
        let refs: Vec<&Array<f32>> = parts.iter().collect();
        let solve = || -> Vec<Vec<f32>> {
            let solution = solve_block_tridiagonal_parts(
                0,
                std::array::from_fn::<_, 16, _>(|k| refs[k]),
                std::array::from_fn::<_, 16, _>(|k| refs[16 + k]),
                std::array::from_fn::<_, 16, _>(|k| refs[32 + k]),
                std::array::from_fn::<_, 4, _>(|k| refs[48 + k]),
            );
            solution.iter().map(|x| x.to_vec().unwrap()).collect()
        };
        let (solution, launches) = if ctx.profiling_supported() {
            let (solution, profile) = ctx.profile(solve).unwrap();
            (solution, Some(profile.launches()))
        } else {
            (solve(), None)
        };
        println!("[backends] {who}: block solve over 104 parts took {launches:?} launches");
        if kind != BackendKind::Wgpu {
            assert_eq!(launches, Some(1), "{who}: the 104 buffers of one block solve must be bound by a single fused kernel");
        }
        for (c, component) in solution.iter().enumerate() {
            must(who, &format!("component {c} against the host"), first_far(component, &want[c], 1e-4, 1e-5));
        }
        results.push((kind, solution));
    }
    agree("block solve", &results, |x, y| {
        for (c, (u, v)) in x.iter().zip(y).enumerate() {
            first_far(u, v, 1e-5, 1e-5).map_err(|e| format!("component {c}: {e}"))?;
        }
        Ok(())
    });
}

// -- Loops --

fn shift_step(state: &[Array<f32>; 2]) -> [Array<f32>; 2] {
    let [u, v] = state;
    [(u.shift(1) + u.shift(-1)) * 0.25 + u * 0.5 + v * 0.01, v * 0.5 + u * 0.5]
}

fn shift_host(u: &[f32], v: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let n = u.len();
    ((0..n).map(|i| (u[(i + n - 1) % n] + u[(i + 1) % n]) * 0.25 + u[i] * 0.5 + v[i] * 0.01).collect(), (0..n).map(|i| v[i] * 0.5 + u[i] * 0.5).collect())
}

/// A step that writes a ghost cell and caches a staged expression.
fn padded_step(u: &Array<f32>) -> Array<f32> {
    let n = u.len();
    let mut w = u.pad([(1, 1)], PadMode::Edge);
    w.slice_mut(0, ..1).assign(0.5).unwrap();
    let c = (w.slice(0, 1..n + 1) * 0.5 + w.slice(0, ..n) * 0.25 + w.slice(0, 2..n + 2) * 0.25).cache();
    c * 0.99 + 0.001
}

fn padded_host(u: &[f32]) -> Vec<f32> {
    let n = u.len();
    let w: Vec<f32> = std::iter::once(0.5).chain(u.iter().copied()).chain(std::iter::once(u[n - 1])).collect();
    (0..n).map(|i| (0.5 * w[i + 1] + 0.25 * w[i] + 0.25 * w[i + 2]) * 0.99 + 0.001).collect()
}

#[test]
fn iterate_all_of_steps_that_are_not_pointwise_matches_the_host_for_every_step_count() {
    let n = 256;
    let (u0, v0) = (values(n, 61, 0.0, 1.0), values(n, 62, 0.0, 1.0));
    // `host[k]` is the state after `k` steps.
    let mut host = vec![(u0.clone(), v0.clone())];
    for _ in 0..9 {
        let next = shift_host(&host.last().unwrap().0, &host.last().unwrap().1);
        host.push(next);
    }
    let mut padded = vec![u0.clone()];
    for _ in 0..5 {
        let next = padded_host(padded.last().unwrap());
        padded.push(next);
    }
    for (kind, ctx) in backends() {
        let who = label(kind);
        let (u, v) = (on(&ctx, &u0), on(&ctx, &v0));
        // Even and odd step counts, none and one; repeating a run replays the program recorded by the first.
        for steps in [0usize, 1, 2, 3, 4, 5, 8, 9] {
            for round in 0..3 {
                let [ru, rv] = iterate_all([&u, &v], steps, shift_step);
                must(who, &format!("{steps} steps, run {round}: u"), first_far(&ru.to_vec().unwrap(), &host[steps].0, 1e-5, 1e-6));
                must(who, &format!("{steps} steps, run {round}: v"), first_far(&rv.to_vec().unwrap(), &host[steps].1, 1e-5, 1e-6));
            }
        }
        must(who, "the initial u is untouched", first_diff(&u.to_vec().unwrap(), &u0));
        must(who, "the initial v is untouched", first_diff(&v.to_vec().unwrap(), &v0));
        // The returned arrays feed the next run, and the first ones are dropped in between.
        let [a, b] = iterate_all([&u, &v], 3, shift_step);
        let [c, d] = iterate_all([&a, &b], 4, shift_step);
        drop((a, b));
        let [e, f] = iterate_all([&c, &d], 2, shift_step);
        must(who, "3 + 4 steps: u", first_far(&c.to_vec().unwrap(), &host[7].0, 1e-5, 1e-6));
        must(who, "3 + 4 steps: v", first_far(&d.to_vec().unwrap(), &host[7].1, 1e-5, 1e-6));
        must(who, "3 + 4 + 2 steps: u", first_far(&e.to_vec().unwrap(), &host[9].0, 1e-5, 1e-6));
        must(who, "3 + 4 + 2 steps: v", first_far(&f.to_vec().unwrap(), &host[9].1, 1e-5, 1e-6));

        let x = on(&ctx, &u0);
        for (steps, want_padded) in padded.iter().enumerate() {
            let [y] = iterate_all([&x], steps, |[s]| [padded_step(s)]);
            must(who, &format!("padded step x{steps}"), first_far(&y.to_vec().unwrap(), want_padded, 1e-5, 1e-6));
        }
    }
}

type One = (Array<f32>,);

fn ones(ctx: &Context, n: usize, value: f32) -> One {
    (Array::full_on(ctx, n, value).unwrap(),)
}

/// What a loop reports: steps, why it stopped, the recorded history.
type Outcome = (usize, StopReason, Vec<f32>);

fn outcome(run: &Run<One>) -> Outcome {
    (run.steps, run.reason, run.history("max").unwrap().to_vec())
}

#[test]
fn iterate_until_all_stops_at_the_same_step_for_the_same_reason() {
    let halving: Vec<f32> = (1..=10).map(|k| 0.5f32.powi(k)).collect();
    let mut results = Vec::new();
    for (kind, ctx) in backends() {
        let who = label(kind);
        // The state halves every step (a shifted average of a constant: exact in f32); the second
        // rule holds first, at step 10.
        let until = |max_steps: usize| {
            Until::new(max_steps)
                .record("max", |s: &One| s.0.max())
                .stop_when(|h| h["max"].last().le(1e-9))
                .stop_when(|h| h["max"].last().le(1e-3))
        };
        let shifted = iterate_until_all(ones(&ctx, 100, 1.0), until(100), |s, _| ((s.0.shift(1) + s.0.shift(-1)) * 0.25,)).unwrap();
        let pointwise = iterate_until_all(ones(&ctx, 100, 1.0), until(100), |s, _| (&s.0 * 0.5,)).unwrap();
        for (what, run) in [("shifted", &shifted), ("pointwise", &pointwise)] {
            assert_eq!((run.steps, run.reason), (10, StopReason::Predicate(1)), "{who} {what}");
            must(who, what, first_diff(run.history("max").unwrap(), &halving));
            must(who, what, first_diff(&run.state.0.to_vec().unwrap(), &vec![0.5f32.powi(10); 100]));
        }
        // The step limit.
        let limited = iterate_until_all(ones(&ctx, 100, 1.0), until(4), |s, _| (&s.0 * 0.5,)).unwrap();
        assert_eq!((limited.steps, limited.reason), (4, StopReason::MaxSteps), "{who}");
        must(who, "limited history", first_diff(limited.history("max").unwrap(), &[0.5, 0.25, 0.125, 0.0625]));
        // The rule is only looked at every 8 steps, yet the loop reports the step at which it
        // first held (30: 2^-30 <= 1e-9) and the history of every step up to it.
        let blocks = iterate_until_all(
            ones(&ctx, 100, 1.0),
            Until::new(100).record("max", |s: &One| s.0.max()).stop_when(|h| h["max"].last().le(1e-9)).check_every(8),
            |s, _| (&s.0 * 0.5,),
        )
        .unwrap();
        assert_eq!((blocks.steps, blocks.reason), (30, StopReason::Predicate(0)), "{who}");
        // Counting is exact in f32 across the blocks of 16, 32, 64 and 128 steps.
        let counting = iterate_until_all(
            (Array::<f32>::zeros_on(&ctx, 10).unwrap(),),
            Until::new(1000).record("max", |s: &One| s.0.max()).stop_when(|h| h["max"].last().ge(300.0)).check_every(7),
            |s, _| (&s.0 + 1.0,),
        )
        .unwrap();
        assert_eq!((counting.steps, counting.reason), (300, StopReason::Predicate(0)), "{who}");
        let want_counts: Vec<f32> = (1..=300).map(|k| k as f32).collect();
        must(who, "counting history", first_diff(counting.history("max").unwrap(), &want_counts));
        assert_eq!(counting.state.0.to_vec().unwrap(), vec![300.0f32; 10], "{who}");

        // A compiled loop replays on any number of initial states.
        let init = ones(&ctx, 64, 1.0);
        let mut compiled = compile_until_all(&init, until(100), |s, _| ((s.0.shift(1) + s.0.shift(-1)) * 0.25,)).unwrap();
        let first = compiled.run(ones(&ctx, 64, 1.0)).unwrap();
        let second = compiled.run(ones(&ctx, 64, 8.0)).unwrap();
        let again = compiled.run(first.state.clone()).unwrap();
        assert_eq!((first.steps, first.reason), (10, StopReason::Predicate(1)), "{who}: first compiled run");
        // 8 * 2^-13 = 2^-10 <= 1e-3 at step 13.
        assert_eq!((second.steps, second.reason), (13, StopReason::Predicate(1)), "{who}: second compiled run");
        // Starting from 2^-10 the loop stops after 1 step (2^-11 <= 1e-3 already holds).
        assert_eq!((again.steps, again.reason), (1, StopReason::Predicate(1)), "{who}: run from a returned state");
        assert_eq!(second.state.0.to_vec().unwrap(), vec![0.5f32.powi(10); 64], "{who}");
        results.push((kind, (outcome(&shifted), outcome(&pointwise), outcome(&limited), outcome(&blocks), outcome(&counting))));
    }
    agree("iterate_until_all", &results, |a, b| {
        if a == b { Ok(()) } else { Err(format!("{a:?} vs {b:?}")) }
    });
}

// -- Profiling --

#[test]
fn profiles_count_the_launches_of_every_kernel_by_label() {
    const SCALE: &str = r#"
    @group(1) @binding(0) var<storage, read> x: array<f32>;
    @group(1) @binding(1) var<storage, read_write> y: array<f32>;
    @compute @workgroup_size(64)
    fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
        if (gid.x >= luma.size.x) { return; }
        y[gid.x] = x[gid.x] * 2.0;
    }"#;
    let n = 4096usize;
    let xs = values(n, 71, -1.0, 1.0);
    let want_scaled: Vec<f32> = xs.iter().map(|v| v * 2.0).collect();
    let want_affine: Vec<f32> = xs.iter().map(|v| v * 3.0 + 1.0).collect();
    for (kind, ctx) in backends() {
        let who = label(kind);
        let x = on(&ctx, &xs);
        let mut y = Array::<f32>::zeros_on(&ctx, n).unwrap();
        let kernel = ctx.kernel(SCALE).label("backends_scale").build().unwrap();
        if !ctx.profiling_supported() {
            assert!(matches!(ctx.enable_profiling(), Err(LumaError::Profiling(_))), "{who}");
            assert!(matches!(ctx.profile(|| ()), Err(LumaError::Profiling(_))), "{who}");
            println!("[backends] {who}: GPU timing is not supported, only the error was checked");
            continue;
        }
        let (affine, profile) = ctx
            .profile(|| {
                for _ in 0..3 {
                    kernel.launch([n as u32], args![&x, &mut y]).unwrap();
                }
                (&x * 3.0 + 1.0).to_vec().unwrap()
            })
            .unwrap();
        must(who, "the fused elementwise result", first_far(&affine, &want_affine, 1e-5, 1e-6));
        must(who, "the custom kernel result", first_diff(&y.to_vec().unwrap(), &want_scaled));
        let custom = profile.get("backends_scale").unwrap_or_else(|| panic!("{who}: no entry for the custom kernel in\n{profile}"));
        assert_eq!(custom.launches, 3, "{who}: {profile}");
        assert!(profile.kernels.len() >= 2, "{who}: the fused kernel has an entry of its own:\n{profile}");
        // A profile covers only the launches since it was enabled.
        let ((), second) = ctx.profile(|| kernel.launch([n as u32], args![&x, &mut y]).unwrap()).unwrap();
        assert_eq!(second.launches(), 1, "{who}: {second}");
        assert_eq!(second.get("backends_scale").map(|k| k.launches), Some(1), "{who}: {second}");
    }
}

// -- Custom kernels --

/// A 3-component vector for encase (16-byte aligned, 12 bytes wide in WGSL).
#[derive(Clone, Copy)]
struct Vec3([f32; 3]);

impl AsRef<[f32; 3]> for Vec3 {
    fn as_ref(&self) -> &[f32; 3] {
        &self.0
    }
}

impl AsMut<[f32; 3]> for Vec3 {
    fn as_mut(&mut self) -> &mut [f32; 3] {
        &mut self.0
    }
}

impl From<[f32; 3]> for Vec3 {
    fn from(v: [f32; 3]) -> Self {
        Vec3(v)
    }
}

encase::impl_vector!(3, Vec3, f32; using AsRef AsMut From);

/// In WGSL `v` sits at offset 16 and `b` at 28 (inside the padding of the `vec3`); a `repr(C)`
/// struct would put `v` at 4.
#[derive(encase::ShaderType)]
struct Padded {
    a: f32,
    v: Vec3,
    b: f32,
}

#[test]
fn a_custom_kernel_reads_a_uniform_with_vec3_padding() {
    const SRC: &str = r#"
    struct Padded { a: f32, v: vec3<f32>, b: f32 }
    @group(1) @binding(0) var<uniform> p: Padded;
    @group(1) @binding(1) var<storage, read_write> out: array<f32>;
    @compute @workgroup_size(1)
    fn main() {
        out[0] = p.a; out[1] = p.v.x; out[2] = p.v.y; out[3] = p.v.z; out[4] = p.b;
    }"#;
    for (kind, ctx) in backends() {
        let mut out = Array::<f32>::zeros_on(&ctx, 5).unwrap();
        let value = Padded { a: 1.0, v: Vec3([2.0, 3.0, 4.0]), b: 5.0 };
        ctx.kernel(SRC).build().unwrap().launch([1], args![uniform(&value), &mut out]).unwrap();
        assert_eq!(out.to_vec().unwrap(), vec![1.0, 2.0, 3.0, 4.0, 5.0], "{}", label(kind));
    }
}

#[derive(encase::ShaderType)]
struct AccessParams {
    n: u32,
}

/// Indices that leave the arrays at run time: reads give zero (or, under WebGPU, any element
/// of the buffer), writes and atomics must not change anything else.
#[test]
fn dynamic_out_of_bounds_loads_stores_and_atomics_stay_inside_the_buffers() {
    const SRC: &str = r#"
    struct Params { n: u32 }
    @group(1) @binding(0) var<storage, read> idx: array<u32>;
    @group(1) @binding(1) var<storage, read> src: array<f32>;
    @group(1) @binding(2) var<storage, read_write> loaded: array<f32>;
    @group(1) @binding(3) var<storage, read_write> dst: array<f32>;
    @group(1) @binding(4) var<storage, read_write> counts: array<atomic<u32>>;
    @group(1) @binding(5) var<uniform> p: Params;
    @compute @workgroup_size(32)
    fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
        let i = gid.x;
        if (i >= p.n) { return; }
        let k = idx[i];
        loaded[i] = src[k];
        dst[k] = 7.0;
        atomicAdd(&counts[k], 1u);
    }"#;
    let (n, src_len, dst_len, counts_len) = (64usize, 16usize, 16usize, 12usize);
    // Indices 0..19 cycle (16 and above leave `src` and `dst`, 12 and above leave `counts`), and
    // every 17th is enormous.
    let idx: Vec<u32> = (0..n).map(|i| if i % 17 == 16 { 0xFFFF_FFF0 } else { (i * 5 % 20) as u32 }).collect();
    let src: Vec<f32> = (0..src_len).map(|i| (i + 1) as f32 * 3.0).collect();
    let mut want_dst = vec![-1.0f32; dst_len];
    let mut want_counts = vec![0u32; counts_len];
    for &k in &idx {
        if (k as usize) < dst_len {
            want_dst[k as usize] = 7.0;
        }
        if (k as usize) < counts_len {
            want_counts[k as usize] += 1;
        }
    }
    for (kind, ctx) in backends() {
        let who = label(kind);
        // WebGPU may clamp an out-of-range access to the last element instead of skipping it.
        let strict = kind != BackendKind::Wgpu;
        let (gidx, gsrc) = (on(&ctx, &idx), on(&ctx, &src));
        let mut loaded = Array::<f32>::full_on(&ctx, n, -5.0).unwrap();
        let mut dst = Array::<f32>::full_on(&ctx, dst_len, -1.0).unwrap();
        let mut counts = Array::<u32>::zeros_on(&ctx, counts_len).unwrap();
        let kernel = ctx.kernel(SRC).label("backends_oob").build().unwrap();
        kernel.launch([n as u32], args![&gidx, &gsrc, &mut loaded, &mut dst, &mut counts, uniform(&AccessParams { n: n as u32 })]).unwrap();

        let loaded = loaded.to_vec().unwrap();
        for (i, &k) in idx.iter().enumerate() {
            if (k as usize) < src_len {
                assert_eq!(loaded[i], src[k as usize], "{who}: in-range load {i}");
            } else if strict {
                assert_eq!(loaded[i], 0.0, "{who}: out-of-range load {i} must read zero");
            } else {
                assert!(loaded[i] == 0.0 || src.contains(&loaded[i]), "{who}: out-of-range load {i} read {}, not zero or an element", loaded[i]);
            }
        }
        let dst = dst.to_vec().unwrap();
        let counts = counts.to_vec().unwrap();
        if strict {
            must(who, "stores", first_diff(&dst, &want_dst));
            must(who, "atomics", first_diff(&counts, &want_counts));
        } else {
            must(who, "stores before the last element", first_diff(&dst[..dst_len - 1], &want_dst[..dst_len - 1]));
            assert!(dst[dst_len - 1] == want_dst[dst_len - 1] || dst[dst_len - 1] == 7.0, "{who}: last element {}", dst[dst_len - 1]);
            must(who, "atomics before the last element", first_diff(&counts[..counts_len - 1], &want_counts[..counts_len - 1]));
            assert!(counts[counts_len - 1] >= want_counts[counts_len - 1], "{who}: last counter {}", counts[counts_len - 1]);
        }
    }
}
