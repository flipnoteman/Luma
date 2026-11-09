//! GPU timing with timestamp queries: a smoke test where the adapter supports them, and the
//! graceful error where it does not.

use luma::*;
use std::time::Duration;

const SPIN: &str = r#"
struct Params { rounds: u32 }
@group(1) @binding(0) var<storage, read_write> a: array<f32>;
@group(1) @binding(1) var<uniform> p: Params;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= luma.size.x) { return; }
    var v = a[gid.x];
    for (var i = 0u; i < p.rounds; i = i + 1u) { v = v * 1.0000001 + 0.5; }
    a[gid.x] = v;
}
"#;

#[derive(encase::ShaderType)]
struct Params {
    rounds: u32,
}

#[test]
fn profile_reports_gpu_time_per_kernel_label() {
    let ctx = Context::new().unwrap();
    let spin = ctx.kernel(SPIN).label("spin").build().unwrap();
    let n = 1 << 16;
    let mut a = Array::<f32>::zeros_on(&ctx, n).unwrap();

    if !ctx.profiling_supported() {
        assert!(matches!(
            ctx.enable_profiling(),
            Err(LumaError::Profiling(_))
        ));
        assert!(matches!(ctx.profile(|| ()), Err(LumaError::Profiling(_))));
        assert!(matches!(
            ctx.finish_profiling(),
            Err(LumaError::Profiling(_))
        ));
        // Everything else keeps working.
        spin.launch([n as u32], args![&mut a, uniform(&Params { rounds: 1 })])
            .unwrap();
        assert_eq!(a.to_vec().unwrap()[0], 0.5);
        return;
    }

    // Not enabled: nothing to finish.
    assert!(matches!(
        ctx.finish_profiling(),
        Err(LumaError::Profiling(_))
    ));

    let b = Array::<f32>::full_on(&ctx, n, 2.0).unwrap();
    let (c, profile) = ctx
        .profile(|| {
            for _ in 0..10 {
                spin.launch(
                    [n as u32],
                    args![&mut a, uniform(&Params { rounds: 20_000 })],
                )
                .unwrap();
            }
            &b * 3.0
        })
        .unwrap();
    println!("{profile}");

    let spin_time = profile
        .get("spin")
        .expect("the custom kernel is listed by its label");
    assert_eq!(spin_time.launches, 10);
    assert!(spin_time.total > Duration::ZERO, "{spin_time:?}");
    assert!(spin_time.min <= spin_time.mean() && spin_time.mean() <= spin_time.max);
    assert!(spin_time.total >= spin_time.max);
    // Built-in operations are measured too, by operation and element type.
    let mul = profile
        .kernels
        .iter()
        .find(|k| k.label.contains("Mul") && k.label.contains("f32"))
        .expect("the built-in multiplication is listed");
    assert_eq!(mul.launches, 1);
    assert_eq!(profile.launches(), 11);
    // Sorted by total time, the heavy kernel first.
    assert_eq!(profile.kernels[0].label, "spin");
    assert!(profile.total() >= spin_time.total);
    assert!(profile.to_string().contains("spin"));

    // The measured work really ran.
    assert_eq!(c.to_vec().unwrap(), vec![6.0; n]);
    assert!(a.to_vec().unwrap().iter().all(|&v| v > 1000.0));

    // Profiling is off again: finishing fails, ordinary launches work and are not measured.
    assert!(matches!(
        ctx.finish_profiling(),
        Err(LumaError::Profiling(_))
    ));
    let again = &b + 1.0;
    assert_eq!(again.to_vec().unwrap(), vec![3.0; n]);
}

#[test]
fn profile_covers_more_launches_than_fit_in_one_batch() {
    let ctx = Context::new().unwrap();
    if !ctx.profiling_supported() {
        return;
    }
    let mut x = Array::<f32>::zeros_on(&ctx, 1000).unwrap();
    let launches = 1500;
    let ((), profile) = ctx
        .profile(|| {
            for _ in 0..launches {
                x += 1.0;
            }
        })
        .unwrap();
    assert_eq!(profile.launches(), launches);
    assert_eq!(profile.kernels.len(), 1);
    assert_eq!(x.to_vec().unwrap(), vec![launches as f32; 1000]);
}

#[test]
fn enable_and_finish_profiling_bracket_manual_sections() {
    let ctx = Context::new().unwrap();
    if !ctx.profiling_supported() {
        return;
    }
    let mut x = Array::<f32>::zeros_on(&ctx, 100).unwrap();
    x += 1.0; // recorded before: not measured
    ctx.enable_profiling().unwrap();
    ctx.enable_profiling().unwrap(); // idempotent
    x *= 3.0;
    x += 1.0;
    let profile = ctx.finish_profiling().unwrap();
    assert_eq!(profile.launches(), 2);
    assert_eq!(x.to_vec().unwrap(), vec![4.0; 100]);
}

#[test]
fn profile_labels_fused_kernels_by_their_operations() {
    let ctx = Context::new().unwrap();
    if !ctx.profiling_supported() {
        return;
    }
    let n = 4096;
    let x = Array::<f32>::full_on(&ctx, n, 4.0).unwrap();
    let y = Array::<f32>::full_on(&ctx, n, 5.0).unwrap();
    // `z` is still lazy when the closure returns; finishing the profile records its kernel.
    let ((z, total), profile) = ctx
        .profile(|| {
            let z = (&x * 2.0 + &y).sqrt();
            let total = (&x * &y).sum().unwrap();
            (z, total)
        })
        .unwrap();
    println!("{profile}");
    assert_eq!(total, 20.0 * n as f32);
    let fused = profile
        .kernels
        .iter()
        .find(|k| k.label == "Fused(Mul, Add, Sqrt) f32")
        .expect("the fused elementwise kernel is listed by its operations");
    assert_eq!(fused.launches, 1);
    assert!(profile.kernels.iter().any(|k| k.label.starts_with("FusedReduce(Sum: Mul)")));
    assert!(z.to_vec().unwrap().iter().all(|&v| (v - 13.0f32.sqrt()).abs() < 1e-5));
}

#[test]
fn every_launch_of_a_profile_gets_its_own_timing() {
    // One batch per launch (each waited for), so timestamps of different submissions must
    // neither be lost nor mixed up.
    let ctx = Context::new().unwrap();
    if !ctx.profiling_supported() {
        return;
    }
    let spin = ctx.kernel(SPIN).label("spin").build().unwrap();
    let n = 1 << 16;
    let mut a = Array::<f32>::zeros_on(&ctx, n).unwrap();
    let ((), profile) = ctx
        .profile(|| {
            for _ in 0..6 {
                spin.launch([n as u32], args![&mut a, uniform(&Params { rounds: 2000 })]).unwrap();
                ctx.sync().unwrap();
            }
        })
        .unwrap();
    let spin_time = profile.get("spin").unwrap();
    assert_eq!(spin_time.launches, 6);
    // A timestamp resolved before the GPU wrote it reads as zero.
    assert!(!spin_time.min.is_zero(), "{profile}");
}
