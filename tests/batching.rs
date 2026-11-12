// wgpu's handle types nest deeply; proving `Context: Send` for the threads below exceeds the
// default limit on recent nightlies (future-compat lint `recursion_depth_exceeding_limit`).
#![recursion_limit = "256"]

//! Operations are recorded into one pending batch and only submitted at synchronization points
//! (or after a few hundred dispatches). These tests try to break the ordering guarantees that
//! batching must keep.
//!
//! Each test uses its own `Context` (device): the pool reuse the hazards depend on must not be
//! disturbed by tests running in parallel.

mod common;

use common::{assert_close, values};
use luma::*;

fn f32s_on(ctx: &Context, data: &[f32]) -> Array<f32> {
    Array::from_slice_on(ctx, data, data.len()).unwrap()
}

// (a) Host data written into a reused pooled buffer while an earlier operation still reads the
// old contents.

#[test]
fn from_slice_into_a_reused_buffer_does_not_disturb_pending_readers() {
    let ctx = Context::new().unwrap();
    let old: Vec<f32> = (0..1000).map(|i| i as f32).collect();
    let new: Vec<f32> = (0..1000).map(|i| -(i as f32) - 1.0).collect();

    let a = f32s_on(&ctx, &old);
    let doubled = &a * 2.0; // pending: reads a's buffer
    let shifted = a.shift(3); // pending as well
    let sum = &a + &a;
    drop(a); // buffer goes back to the pool while three operations still need it

    // Takes the very buffer of `a` (same size class), uploading new contents into it.
    let b = f32s_on(&ctx, &new);
    let negated = -&b;

    assert_eq!(b.to_vec().unwrap(), new);
    assert_eq!(negated.to_vec().unwrap(), old.iter().map(|x| x + 1.0).collect::<Vec<_>>());
    assert_eq!(doubled.to_vec().unwrap(), old.iter().map(|x| x * 2.0).collect::<Vec<_>>());
    assert_eq!(sum.to_vec().unwrap(), old.iter().map(|x| x * 2.0).collect::<Vec<_>>());
    let n = old.len();
    let want_shifted: Vec<f32> = (0..n).map(|i| old[(i + n - 3) % n]).collect();
    assert_eq!(shifted.to_vec().unwrap(), want_shifted);
}

#[test]
fn write_into_an_array_that_pending_operations_read() {
    let ctx = Context::new().unwrap();
    let mut x = f32s_on(&ctx, &[1.0; 500]);

    let mut outputs = Vec::new();
    for i in 0..40 {
        let data = vec![i as f32; 500];
        x.write(&data).unwrap();
        // Recorded before the next write: must see this write's data, not the next one's.
        outputs.push(&x * 10.0 + 1.0);
    }
    x.write(&[99.0; 500]).unwrap();
    for (i, out) in outputs.iter().enumerate() {
        assert_eq!(out.to_vec().unwrap(), vec![i as f32 * 10.0 + 1.0; 500], "output {i}");
    }
    assert_eq!(x.to_vec().unwrap(), vec![99.0; 500]);
}

#[test]
fn zeros_and_full_into_reused_buffers() {
    let ctx = Context::new().unwrap();
    let a = f32s_on(&ctx, &[7.0; 300]);
    let kept = &a + 1.0;
    drop(a);
    let z = Array::<f32>::zeros_on(&ctx, 300).unwrap(); // reuses a's buffer
    let f = Array::full_on(&ctx, 300, 5.0f32).unwrap();
    drop(z);
    let z2 = Array::<f32>::zeros_on(&ctx, 300).unwrap();
    assert_eq!(kept.to_vec().unwrap(), vec![8.0; 300]);
    assert_eq!(f.to_vec().unwrap(), vec![5.0; 300]);
    assert_eq!(z2.to_vec().unwrap(), vec![0.0; 300]);
}

// (b) Dropping arrays and allocating new ones while their users are still pending.

#[test]
fn drop_and_reallocate_while_operations_are_pending() {
    let ctx = Context::new().unwrap();
    let data = values(777, 31, 0.0, 1.0);
    let x = f32s_on(&ctx, &data);

    // Each iteration drops its intermediates immediately, so the next iteration's allocations
    // take the buffers the previous (not yet executed) operations are still using.
    let mut expected = data.clone();
    let mut acc = x;
    for i in 0..60 {
        let t = &acc * 1.5;
        let u = &t + (i as f32);
        drop(t);
        let v = u.sqrt();
        drop(u);
        drop(acc);
        let fresh = f32s_on(&ctx, &vec![(i + 1) as f32; 777]); // takes recycled buffers too
        acc = &v + &fresh;
        for e in &mut expected {
            *e = (*e * 1.5 + i as f32).sqrt() + (i + 1) as f32;
        }
    }
    assert_close(&acc.to_vec().unwrap(), &expected, 1e-4, 1e-4);
}

#[test]
fn scratch_buffers_of_multi_pass_operations_are_reused_while_pending() {
    let ctx = Context::new().unwrap();
    let input_data = values(300, 32, -1.0, 1.0);
    let kernel_data = values(40, 33, -1.0, 1.0);
    let input = f32s_on(&ctx, &input_data);
    let kernel = f32s_on(&ctx, &kernel_data);

    // Several FFT convolutions and reductions recorded back to back share scratch buffers.
    let convs: Vec<Array<f32>> = (0..5).map(|_| input.conv1d_fft(&kernel)).collect();
    let shifted: Vec<Array<f32>> = (0..5).map(|i| input.shift(i as isize)).collect();
    let spatial = input.conv1d(&kernel, 1, 0);

    let mut want = vec![0.0f32; 339];
    for (i, x) in input_data.iter().enumerate() {
        for (k, w) in kernel_data.iter().enumerate() {
            want[i + k] += x * w;
        }
    }
    for conv in &convs {
        assert_close(&conv.to_vec().unwrap(), &want, 1e-3, 1e-3);
    }
    for (i, s) in shifted.iter().enumerate() {
        let n = input_data.len();
        let want: Vec<f32> = (0..n).map(|j| input_data[(j + n - i) % n]).collect();
        assert_eq!(s.to_vec().unwrap(), want);
    }
    assert_eq!(spatial.len(), 261);
}

// (c) More operations than the batch limit.

#[test]
fn chains_longer_than_the_dispatch_limit_flush_correctly() {
    let ctx = Context::new().unwrap();
    let n = 2000;
    let mut x = Array::<f32>::zeros_on(&ctx, 16).unwrap();
    let mut expected = 0.0f32;
    for i in 0..n {
        // Every dispatch has its own scalar: if uniforms of one batch were shared or
        // overwritten, the sum would be off.
        let s = (i % 17) as f32;
        x += s;
        expected += s;
    }
    assert_eq!(x.to_vec().unwrap(), vec![expected; 16]);

    // Same with expression temporaries (allocation + drop on every step).
    let mut y = f32s_on(&ctx, &[1.0; 64]);
    for i in 0..1000 {
        y = &y + (i % 5) as f32 - (i % 3) as f32 * 0.5;
    }
    let want: f32 = (0..1000).fold(1.0f32, |acc, i| acc + (i % 5) as f32 - (i % 3) as f32 * 0.5);
    assert_close(&y.to_vec().unwrap(), &vec![want; 64], 1e-5, 1e-3);
}

#[test]
fn many_independent_results_collected_after_the_batch_overflowed() {
    let ctx = Context::new().unwrap();
    let x = f32s_on(&ctx, &[2.0; 100]);
    let results: Vec<Array<f32>> = (0..700).map(|i| &x * (i as f32)).collect();
    for i in [0usize, 1, 255, 256, 257, 511, 512, 699] {
        assert_eq!(results[i].to_vec().unwrap(), vec![2.0 * i as f32; 100], "result {i}");
    }
}

#[test]
fn reductions_inside_long_chains() {
    let ctx = Context::new().unwrap();
    let x = f32s_on(&ctx, &[1.0; 1000]);
    let mut total = 0.0f32;
    for i in 0..300 {
        total += (&x * (i as f32)).sum().item().unwrap();
    }
    let want: f32 = (0..300).map(|i| 1000.0 * i as f32).sum();
    assert_eq!(total, want);
}

// Uniform values must stay distinct per launch, also when the arena of the batch spills over
// into further chunks.

#[derive(Clone, Copy)]
struct V4([u32; 4]);

impl AsRef<[u32; 4]> for V4 {
    fn as_ref(&self) -> &[u32; 4] {
        &self.0
    }
}
impl AsMut<[u32; 4]> for V4 {
    fn as_mut(&mut self) -> &mut [u32; 4] {
        &mut self.0
    }
}
impl From<[u32; 4]> for V4 {
    fn from(v: [u32; 4]) -> Self {
        V4(v)
    }
}
encase::impl_vector!(4, V4, u32; using AsRef AsMut From);

#[derive(encase::ShaderType)]
struct Big {
    slot: u32,
    value: u32,
    // 16 KiB of padding so that each launch needs a large slot of the uniform arena.
    padding: [V4; 1024],
}

#[test]
fn custom_kernel_uniforms_stay_distinct_across_arena_chunks() {
    const SRC: &str = r#"
    struct Big { slot: u32, value: u32, padding: array<vec4<u32>, 1024> }
    @group(1) @binding(0) var<storage, read_write> out: array<u32>;
    @group(1) @binding(1) var<uniform> p: Big;
    @compute @workgroup_size(1)
    fn main() { out[p.slot] = p.value + p.padding[1023].w; }"#;
    let ctx = Context::new().unwrap();
    let kernel = ctx.kernel(SRC).build().unwrap();
    let launches = 200u32; // 200 * ~16.6 KiB: several 256 KiB arena chunks, below the batch limit
    let mut out = Array::<u32>::zeros_on(&ctx, launches as usize).unwrap();
    for k in 0..launches {
        let big = Big { slot: k, value: k * 3 + 1, padding: [V4([0, 0, 0, k]); 1024] };
        kernel.launch([1], args![&mut out, uniform(&big)]).unwrap();
    }
    let want: Vec<u32> = (0..launches).map(|k| k * 3 + 1 + k).collect();
    assert_eq!(out.to_vec().unwrap(), want);
}

// Custom kernel launches and operators share one batch and keep their relative order.

#[test]
fn kernel_launches_interleaved_with_operators() {
    const SAXPY: &str = r#"
    struct Params { alpha: f32, n: u32 }
    @group(1) @binding(0) var<storage, read> x: array<f32>;
    @group(1) @binding(1) var<storage, read_write> y: array<f32>;
    @group(1) @binding(2) var<uniform> p: Params;
    @compute @workgroup_size(64)
    fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
        if (gid.x >= p.n) { return; }
        y[gid.x] = p.alpha * x[gid.x] + y[gid.x];
    }"#;
    #[derive(encase::ShaderType)]
    struct Params {
        alpha: f32,
        n: u32,
    }

    let ctx = Context::new().unwrap();
    let kernel = ctx.kernel(SAXPY).build().unwrap();
    let n = 1000usize;
    let xs = values(n, 40, -1.0, 1.0);
    let ys = values(n, 41, -1.0, 1.0);
    let x = f32s_on(&ctx, &xs);
    let mut y = f32s_on(&ctx, &ys);

    let mut expected = ys.clone();
    // 150 rounds of (operator, kernel, operator): 450 dispatches, crossing the batch limit, with a
    // different uniform value for every launch.
    for round in 0..150 {
        let alpha = 0.01 * (round % 7) as f32 - 0.02;
        y *= 0.99; // operator
        for e in &mut expected {
            *e *= 0.99;
        }
        kernel
            .launch([n as u32], args![&x, &mut y, uniform(&Params { alpha, n: n as u32 })])
            .unwrap();
        for (e, xv) in expected.iter_mut().zip(&xs) {
            *e += alpha * xv;
        }
        y += 0.001; // operator reading the kernel's result
        for e in &mut expected {
            *e += 0.001;
        }
    }
    assert_close(&y.to_vec().unwrap(), &expected, 1e-3, 1e-3);
}

// Dropping the last handle of an array whose buffer a launch of the unsubmitted batch uses, then
// uploading again into the same size class: with mappable (zero-copy) storage the recycled
// buffer must only be mapped after that batch was submitted.
#[test]
fn dropped_uploads_used_by_unsubmitted_launches_are_remapped_only_after_submit() {
    const COPY: &str = r#"
    @group(1) @binding(0) var<storage, read> x: array<f32>;
    @group(1) @binding(1) var<storage, read_write> y: array<f32>;
    @compute @workgroup_size(64)
    fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
        if (gid.x >= arrayLength(&y)) { return; }
        y[gid.x] = x[gid.x] + 1.0;
    }"#;

    let ctx = Context::new().unwrap();
    let kernel = ctx.kernel(COPY).build().unwrap();
    let n = 1000usize;

    let mut outputs = Vec::new();
    let mut wanted = Vec::new();
    for round in 0..20 {
        let data = vec![round as f32; n];
        let x = f32s_on(&ctx, &data);
        let mut y = f32s_on(&ctx, &vec![0.0; n]);
        kernel.launch([n as u32], args![&x, &mut y]).unwrap();
        drop(x); // recycled while the launch is recorded but not submitted
        outputs.push(y);
        wanted.push(vec![round as f32 + 1.0; n]);
    }
    for (y, want) in outputs.iter().zip(&wanted) {
        assert_eq!(&y.to_vec().unwrap(), want);
    }
}

#[test]
fn kernel_output_feeds_operators_in_the_same_batch() {
    const SRC: &str = r#"
    @group(1) @binding(0) var<storage, read_write> data: array<u32>;
    @compute @workgroup_size(32)
    fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
        if (gid.x < luma.size.x) { data[gid.x] = data[gid.x] * 3u; }
    }"#;
    let ctx = Context::new().unwrap();
    let kernel = ctx.kernel(SRC).build().unwrap();
    let mut a = Array::from_slice_on(&ctx, &[1u32, 2, 3, 4], 4).unwrap();
    a += 1; // 2 3 4 5
    kernel.launch([4], args![&mut a]).unwrap(); // 6 9 12 15
    let b = &a - 1; // 5 8 11 14
    kernel.launch([4], args![&mut a]).unwrap(); // 18 27 36 45
    let c = &a + &b; // 23 35 47 59
    assert_eq!(c.to_vec().unwrap(), vec![23, 35, 47, 59]);
    assert_eq!(b.to_vec().unwrap(), vec![5, 8, 11, 14]);
}

#[test]
fn sync_waits_and_eval_returns_ok() {
    let ctx = Context::new().unwrap();
    let mut x = Array::<u32>::zeros_on(&ctx, 10).unwrap();
    for _ in 0..10 {
        x += 1;
    }
    ctx.sync().unwrap();
    x.eval().unwrap();
    assert_eq!(x.to_vec().unwrap(), vec![10; 10]);
}

#[test]
fn threads_share_a_context() {
    let ctx = Context::new().unwrap();
    std::thread::scope(|scope| {
        let handles: Vec<_> = (0..4)
            .map(|t| {
                let ctx = ctx.clone();
                scope.spawn(move || {
                    let mut x = Array::<f32>::zeros_on(&ctx, 256).unwrap();
                    for i in 0..300 {
                        x += (t * 1000 + i) as f32;
                    }
                    let want: f32 = (0..300).map(|i| (t * 1000 + i) as f32).sum();
                    assert_eq!(x.to_vec().unwrap(), vec![want; 256]);
                })
            })
            .collect();
        for handle in handles {
            handle.join().unwrap();
        }
    });
}

// -- Non-blocking flush --

#[test]
fn flush_submits_without_waiting_and_recording_continues_in_order() {
    let ctx = Context::new().unwrap();
    flush_empty_batch_is_fine(&ctx);

    // Different values per operation, flushed every few operations: slots of the uniform arena
    // are reused by later batches while earlier ones may still be running.
    let n = 1 << 20;
    let mut x = Array::<u32>::zeros_on(&ctx, n).unwrap();
    let mut expected = 0u32;
    for round in 0..60u32 {
        for k in 0..5u32 {
            let v = round * 5 + k + 1;
            x += v;
            expected += v;
        }
        ctx.flush().unwrap();
        // A write between flushes is ordered after the flushed work and before later work.
        if round % 20 == 19 {
            x.write(&vec![expected; n]).unwrap();
        }
    }
    assert_eq!(x.to_vec().unwrap(), vec![expected; n]);
}

fn flush_empty_batch_is_fine(ctx: &Context) {
    ctx.flush().unwrap();
    ctx.flush().unwrap();
    ctx.sync().unwrap();
}

#[test]
fn default_context_flush() {
    let x = Array::from_slice(&[1.0f32, 2.0, 3.0], 3).unwrap();
    let y = &x * 2.0;
    luma::flush().unwrap();
    assert_eq!(y.to_vec().unwrap(), vec![2.0, 4.0, 6.0]);
}

#[test]
fn dropping_a_context_with_unsubmitted_launches_is_clean() {
    let ctx = Context::new().unwrap();
    let mut x = Array::<f32>::zeros_on(&ctx, 64).unwrap();
    for _ in 0..10 {
        x += 1.0;
    }
    // Still pending, with the recording pass open: dropping everything must neither panic nor
    // poison anything for the next context.
    drop(x);
    drop(ctx);
    let ctx = Context::new().unwrap();
    assert_eq!(Array::<u32>::full_on(&ctx, 4, 9).unwrap().to_vec().unwrap(), vec![9; 4]);
}
