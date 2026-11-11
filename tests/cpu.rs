//! The software (CPU) device: Mesa llvmpipe through Vulkan. Tests return early when no software
//! driver is installed.

use luma::{Array, Context, LumaError, TuningProfile};

/// A CPU context, or `None` when the machine has no software driver (the test then has nothing
/// to check).
fn cpu() -> Option<Context> {
    match Context::cpu() {
        Ok(ctx) => Some(ctx),
        Err(LumaError::NoAdapter(_)) => None,
        Err(e) => panic!("opening the CPU device failed: {e}"),
    }
}

#[test]
fn cpu_context_is_selected_and_tuned() {
    let Some(ctx) = cpu() else { return };
    assert!(ctx.is_cpu());
    assert_eq!(ctx.tuning_profile(), TuningProfile::Cpu);
    let data: Vec<f32> = (0..1000).map(|i| i as f32 * 0.5).collect();
    let x = Array::from_slice_on(&ctx, &data, 1000).unwrap();
    let got = (&x * 2.0 + 1.0).to_vec().unwrap();
    let want: Vec<f32> = data.iter().map(|v| v * 2.0 + 1.0).collect();
    assert_eq!(got, want);
}

#[test]
fn fft_convolution_is_exact_without_lockstep_invocations() {
    let Some(ctx) = cpu() else { return };
    for (input_len, kernel_len) in [(3_000usize, 50usize), (70_000, 300)] {
        let input_data: Vec<f32> = (0..input_len).map(|i| ((i * 37 % 101) as f32 - 50.0) / 50.0).collect();
        let kernel_data: Vec<f32> = (0..kernel_len).map(|k| ((k * 13 % 29) as f32 - 14.0) / 14.0).collect();
        let input = Array::from_slice_on(&ctx, &input_data, input_len).unwrap();
        let kernel = Array::from_slice_on(&ctx, &kernel_data, kernel_len).unwrap();
        let got = input.conv1d_fft(&kernel).to_vec().unwrap();
        assert_eq!(got.len(), input_len + kernel_len - 1);
        let mut worst = 0.0f64;
        let ends = 2048;
        for (o, g) in got.iter().enumerate().filter(|(o, _)| o % 7 == 0 || *o < ends || o + ends > got.len()) {
            let lo = o.saturating_sub(input_len - 1);
            let want: f64 = (lo..kernel_len.min(o + 1))
                .map(|k| f64::from(input_data[o - k]) * f64::from(kernel_data[k]))
                .sum();
            worst = worst.max((f64::from(*g) - want).abs());
        }
        assert!(worst < 1e-3 * (kernel_len as f64).sqrt(), "{input_len}/{kernel_len}: max error {worst}");
    }
}

#[test]
fn gpu_and_cpu_contexts_coexist() {
    let Some(cpu_ctx) = cpu() else { return };
    let other = Context::new().unwrap();
    let data: Vec<f32> = (0..5000).map(|i| (i % 17) as f32 * 0.25).collect();
    let on = |ctx: &Context| {
        let x = Array::from_slice_on(ctx, &data, data.len()).unwrap();
        (&x * &x - 3.0).to_vec().unwrap()
    };
    let (a, b) = (on(&other), on(&cpu_ctx));
    assert_eq!(a.len(), b.len());
    for (i, (x, y)) in a.iter().zip(&b).enumerate() {
        assert!((x - y).abs() <= 1e-6, "at {i}: {x} vs {y}");
    }

    let x = Array::from_slice_on(&other, &data, data.len()).unwrap();
    let y = Array::from_slice_on(&cpu_ctx, &data, data.len()).unwrap();
    let err = (&x + &y).to_vec().unwrap_err();
    assert!(matches!(err.root(), LumaError::ContextMismatch), "{err}");
}
