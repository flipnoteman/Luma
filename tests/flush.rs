//! `Context::flush` hands the batch to the GPU without waiting for it.
//!
//! This test measures wall-clock time, so it lives in a binary of its own: other tests sharing
//! the GPU at the same time can stall submissions for milliseconds.

use luma::*;

#[test]
fn flush_returns_while_the_gpu_is_still_running() {
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
    }"#;
    #[derive(encase::ShaderType)]
    struct Params {
        rounds: u32,
    }

    let ctx = Context::new().unwrap();
    let spin = ctx.kernel(SPIN).build().unwrap();
    let n = 1u32 << 16;
    let mut a = Array::<f32>::zeros_on(&ctx, n as usize).unwrap();

    // Find a round count that takes the GPU a few tens of milliseconds, however fast it is.
    let mut rounds = 1u32 << 10;
    loop {
        let start = std::time::Instant::now();
        spin.launch([n], args![&mut a, uniform(&Params { rounds })])
            .unwrap();
        ctx.sync().unwrap();
        if start.elapsed() > std::time::Duration::from_millis(30) || rounds >= 1 << 28 {
            break;
        }
        rounds *= 2;
    }

    spin.launch([n], args![&mut a, uniform(&Params { rounds })])
        .unwrap();
    let start = std::time::Instant::now();
    ctx.flush().unwrap();
    let flush_time = start.elapsed();
    let start = std::time::Instant::now();
    ctx.sync().unwrap();
    let sync_time = start.elapsed();
    // The GPU ran while flush returned: the wait that is left is far longer than the flush.
    assert!(
        flush_time * 4 < sync_time,
        "flush took {flush_time:?}, the sync after it {sync_time:?}"
    );
}

#[test]
fn flush_evaluates_live_lazy_arrays() {
    let ctx = Context::new().unwrap();
    let x = Array::<f32>::full_on(&ctx, 64, 3.0).unwrap();
    let y = &x * 2.0 + 1.0; // lazy: nothing recorded yet
    ctx.flush().unwrap();
    assert_eq!(y.to_vec().unwrap(), vec![7.0; 64]);
}
