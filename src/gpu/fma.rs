//! Whether a device's `fma` is a fused multiply-add: the `Df32` library takes the exact error of a
//! product from it (see `builtin::df_library`), which is only exact when `fma(a, b, -p)` rounds
//! once. Some compilers and software rasterizers lower `fma` to a multiplication and an
//! addition; there the library must split the factors instead.

use wgpu::util::DeviceExt;

/// `out[i] = fma(a[i], b[i], -p)` with `p` the rounded product made opaque to the compiler by the
/// bits of `zero[i]` (all zero): a fused `fma` returns the rounding error of the product, a
/// multiplication and an addition return 0.
const SHADER: &str = "
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read> zero: array<u32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    let i = g.x;
    let p = bitcast<f32>(bitcast<u32>(a[i] * b[i]) | zero[i]);
    out[i] = fma(a[i], b[i], -p);
}
";

/// Factor pairs of 24 significant bits each, whose products need more than 24 (so that the
/// rounding error of the product is not zero): a fixed pseudo-random sequence.
fn pairs() -> Vec<(f32, f32)> {
    let mut state = 0x2545_f491_4f6c_dd1d_u64;
    let mut next = move || {
        state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        // 24 random bits in [1, 2): the mantissa of an f32.
        f32::from_bits(0x3f80_0000 | ((state >> 40) as u32 & 0x007f_ffff))
    };
    (0..64).map(|_| (next(), next())).collect()
}

/// Runs the exactness test on `device`: `Some(true)` when every `fma(a, b, -p)` returned exactly
/// the rounding error of `a * b`. `None` when the test could not run (buffer, poll or readback
/// failure), so the caller does not keep a transient failure.
pub(crate) fn fused(device: &wgpu::Device, queue: &wgpu::Queue) -> Option<bool> {
    let pairs = pairs();
    let expected: Vec<f32> = pairs.iter().map(|&(a, b)| (f64::from(a) * f64::from(b) - f64::from(a * b)) as f32).collect();
    if expected.iter().all(|&e| e == 0.0) {
        return Some(false);
    }
    let bytes = (pairs.len() * 4) as u64;
    let a: Vec<f32> = pairs.iter().map(|p| p.0).collect();
    let b: Vec<f32> = pairs.iter().map(|p| p.1).collect();
    let zero = vec![0u32; pairs.len()];
    let upload = |data: &[u8]| device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("luma fma"), contents: data, usage: wgpu::BufferUsages::STORAGE });
    let (buf_a, buf_b, buf_zero) = (upload(bytemuck::cast_slice(&a)), upload(bytemuck::cast_slice(&b)), upload(bytemuck::cast_slice(&zero)));
    let out = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("luma fma out"),
        size: bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let read = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("luma fma read"),
        size: bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("luma fma"), source: wgpu::ShaderSource::Wgsl(SHADER.into()) });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("luma fma"),
        layout: None,
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    fn entry(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
        wgpu::BindGroupEntry { binding, resource: buffer.as_entire_binding() }
    }
    let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("luma fma"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[entry(0, &buf_a), entry(1, &buf_b), entry(2, &buf_zero), entry(3, &out)],
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("luma fma") });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("luma fma"), timestamp_writes: None });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(pairs.len() as u32, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&out, 0, &read, 0, bytes);
    queue.submit([encoder.finish()]);
    let (sender, receiver) = std::sync::mpsc::channel();
    read.slice(..).map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    if device.poll(wgpu::PollType::wait_indefinitely()).is_err() || !matches!(receiver.recv(), Ok(Ok(()))) {
        return None;
    }
    let Ok(view) = read.slice(..).get_mapped_range() else { return None };
    let got: Vec<f32> = bytemuck::cast_slice(&view).to_vec();
    drop(view);
    read.unmap();
    Some(got.iter().zip(&expected).all(|(g, e)| g.to_bits() == e.to_bits()))
}

#[cfg(test)]
mod tests {
    use crate::{Context, ContextOptions};

    #[test]
    fn the_test_is_repeatable_and_the_context_uses_its_answer() {
        let ctx = Context::with_options(ContextOptions::from_env().calibrate(false)).unwrap();
        let first = super::fused(ctx.device(), ctx.queue());
        assert_eq!(first, super::fused(ctx.device(), ctx.queue()));
        if std::env::var_os("LUMA_DF_FMA").is_none() {
            // The stored answer of an earlier process may be used instead of a new test, but both are one device.
            assert_eq!(Some(ctx.tuning().df_fma), first);
        }
        let forced = Context::with_options(ContextOptions::from_env().calibrate(false).df_fma(false)).unwrap();
        assert!(!forced.tuning().df_fma);
    }
}
