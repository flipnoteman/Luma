//! The transfer lane of the HAL (`Backend::stage_alloc`, `transfer_in`, `transfer_out`,
//! `barrier`, tickets) on the backend `LUMA_BACKEND` names: round trips through a kernel,
//! dependencies between the lanes, and (CUDA, ignored under nextest) overlap with a kernel.

use std::time::{Duration, Instant};

use luma_core::{Backend, StageDir};

use crate::common::lanes::{USAGE, affine_host, affine_kernel, launch, open};

fn words(n: usize, salt: u32) -> Vec<u32> {
    (0..n as u32).map(|i| i.wrapping_mul(2_654_435_761) ^ salt).collect()
}

fn write_words(backend: &dyn Backend, stage: &mut luma_core::Stage, data: &[u32]) {
    backend.stage_ptr(stage)[..data.len() * 4].copy_from_slice(bytemuck::cast_slice(data));
}

fn read_words(stage: &luma_core::Stage, n: usize) -> Vec<u32> {
    bytemuck::pod_collect_to_vec(&stage.as_slice()[..n * 4])
}

#[test]
fn a_round_trip_through_a_kernel_returns_the_bytes() {
    let Some(backend) = open() else { return };
    let n = 100_003usize;
    let bytes = n as u64 * 4;
    let data = words(n, 7);
    let (a, b) = (backend.create_buffer_unordered(bytes, USAGE).unwrap(), backend.create_buffer_unordered(bytes, USAGE).unwrap());
    let mut up = backend.stage_alloc(bytes, StageDir::ToDevice).unwrap();
    write_words(&*backend, &mut up, &data);
    let mut down = backend.stage_alloc(bytes, StageDir::FromDevice).unwrap();

    let uploaded = backend.transfer_in(&up, &a, 0, bytes, &[]).unwrap();
    let ready = backend.barrier(std::slice::from_ref(&uploaded)).unwrap();
    launch(&*backend, &affine_kernel(&*backend, 1), &a, &b, n as u32);
    let computed = backend.barrier(&[ready]).unwrap();
    let downloaded = backend.transfer_out(&b, 0, &mut down, bytes, std::slice::from_ref(&computed)).unwrap();

    backend.ticket_wait(&downloaded, Duration::from_micros(100)).unwrap();
    for ticket in [&uploaded, &computed, &downloaded] {
        assert!(backend.ticket_done(ticket).unwrap());
        backend.ticket_wait(ticket, Duration::ZERO).unwrap();
    }
    let want: Vec<u32> = data.iter().map(|&x| affine_host(x, 1)).collect();
    assert_eq!(read_words(&down, n), want);
}

#[test]
fn a_transfer_waits_for_the_kernel_it_depends_on() {
    let Some(backend) = open() else { return };
    let n = 1 << 18;
    let bytes = n as u64 * 4;
    let data = words(n, 11);
    let (a, b) = (backend.create_buffer_unordered(bytes, USAGE).unwrap(), backend.create_buffer_unordered(bytes, USAGE).unwrap());
    let mut up = backend.stage_alloc(bytes, StageDir::ToDevice).unwrap();
    write_words(&*backend, &mut up, &data);
    let mut down = backend.stage_alloc(bytes, StageDir::FromDevice).unwrap();
    // Zero `b`, so that reading it before the kernel is done shows.
    let zeros = backend.stage_alloc(bytes, StageDir::ToDevice).unwrap();
    backend.transfer_in(&zeros, &b, 0, bytes, &[]).unwrap();
    let uploaded = backend.transfer_in(&up, &a, 0, bytes, &[]).unwrap();
    // The kernel follows the upload.
    backend.barrier(&[uploaded]).unwrap();
    // Long enough that an unordered readback would run first.
    let iters = if std::env::var("LUMA_BACKEND").as_deref() == Ok("cpu") { 200 } else { 20_000 };
    launch(&*backend, &affine_kernel(&*backend, iters), &a, &b, n as u32);
    let computed = backend.barrier(&[]).unwrap();
    let downloaded = backend.transfer_out(&b, 0, &mut down, bytes, &[computed]).unwrap();
    backend.ticket_wait(&downloaded, Duration::ZERO).unwrap();
    let got = read_words(&down, n);
    for i in (0..n).step_by(4093) {
        assert_eq!(got[i], affine_host(data[i], iters), "word {i}");
    }
    assert_eq!(got[n - 1], affine_host(data[n - 1], iters));
}

#[test]
fn transfers_honor_offsets() {
    let Some(backend) = open() else { return };
    let (total, offset, n) = (4096u64, 1024u64, 512usize);
    let buffer = backend.create_buffer_unordered(total * 4, USAGE).unwrap();
    let data = words(n, 3);
    let mut up = backend.stage_alloc(n as u64 * 4, StageDir::ToDevice).unwrap();
    write_words(&*backend, &mut up, &data);
    let mut down = backend.stage_alloc(n as u64 * 4, StageDir::FromDevice).unwrap();
    let uploaded = backend.transfer_in(&up, &buffer, offset, n as u64 * 4, &[]).unwrap();
    let barrier = backend.barrier(&[uploaded]).unwrap();
    let downloaded = backend.transfer_out(&buffer, offset, &mut down, n as u64 * 4, &[barrier]).unwrap();
    backend.ticket_wait(&downloaded, Duration::ZERO).unwrap();
    assert_eq!(read_words(&down, n), data);
}

#[test]
fn misuse_is_an_error() {
    let Some(backend) = open() else { return };
    let buffer = backend.create_buffer_unordered(64, USAGE).unwrap();
    let down = backend.stage_alloc(64, StageDir::FromDevice).unwrap();
    assert!(backend.transfer_in(&down, &buffer, 0, 64, &[]).is_err(), "a stage for reading used to write");
    let up = backend.stage_alloc(64, StageDir::ToDevice).unwrap();
    assert!(backend.transfer_in(&up, &buffer, 16, 64, &[]).is_err(), "past the end of the buffer");
    assert!(backend.transfer_in(&up, &buffer, 0, 128, &[]).is_err(), "more than the stage holds");
    assert_eq!(backend.stage_alloc(0, StageDir::ToDevice).map(|s| s.len()).ok(), Some(0));
}

/// Wall time of `f`.
fn time(f: impl FnOnce()) -> Duration {
    let start = Instant::now();
    f();
    start.elapsed()
}

/// Uploading 256 MiB on the transfer lane beside a kernel of about the same duration takes under
/// 0.8 of the two run one after the other. Run by hand (it needs the device to itself and
/// `CUDA_DEVICE_MAX_CONNECTIONS` at its default, which nextest's profile lowers to 1):
/// `LUMA_BACKEND=cuda cargo test --release --test all transfer_lanes::cuda_upload_overlaps_a_kernel -- --ignored`.
#[test]
#[ignore = "needs the device to itself and the default CUDA_DEVICE_MAX_CONNECTIONS; run through gpuq"]
fn cuda_upload_overlaps_a_kernel() {
    let Some(backend) = open() else { return };
    if std::env::var("LUMA_BACKEND").as_deref() != Ok("cuda") {
        return;
    }
    let slab = 256u64 << 20;
    let n = 1u32 << 22;
    let (dst, a, b) = (
        backend.create_buffer_unordered(slab, USAGE).unwrap(),
        backend.create_buffer_unordered(u64::from(n) * 4, USAGE).unwrap(),
        backend.create_buffer_unordered(u64::from(n) * 4, USAGE).unwrap(),
    );
    let mut stage = backend.stage_alloc(slab, StageDir::ToDevice).unwrap();
    backend.stage_ptr(&mut stage).fill(1);
    let upload = |backend: &dyn Backend| {
        let t = backend.transfer_in(&stage, &dst, 0, slab, &[]).unwrap();
        backend.ticket_wait(&t, Duration::ZERO).unwrap();
    };
    let run = |backend: &dyn Backend, kernel: &luma_core::CompiledKernel| {
        launch(backend, kernel, &a, &b, n);
        backend.sync(Duration::ZERO).unwrap();
    };
    upload(&*backend);
    let up = (0..3).map(|_| time(|| upload(&*backend))).min().unwrap();
    // A kernel as long as the upload: the iteration count is scaled until the two match (the
    // launch's fixed cost makes one step overshoot).
    let mut iters = 1000u32;
    let mut kernel = affine_kernel(&*backend, iters);
    for _ in 0..4 {
        run(&*backend, &kernel);
        let alone = (0..3).map(|_| time(|| run(&*backend, &kernel))).min().unwrap();
        iters = ((f64::from(iters) * up.as_secs_f64() / alone.as_secs_f64()) as u32).max(1);
        kernel = affine_kernel(&*backend, iters);
    }
    run(&*backend, &kernel);
    let alone = (0..3).map(|_| time(|| run(&*backend, &kernel))).min().unwrap();
    let both = (0..3)
        .map(|_| {
            time(|| {
                let t = backend.transfer_in(&stage, &dst, 0, slab, &[]).unwrap();
                launch(&*backend, &kernel, &a, &b, n);
                backend.sync(Duration::ZERO).unwrap();
                backend.ticket_wait(&t, Duration::ZERO).unwrap();
            })
        })
        .min()
        .unwrap();
    let serial = up + alone;
    eprintln!("upload {up:?}, kernel {alone:?}, together {both:?}, serial sum {serial:?}");
    assert!(both.as_secs_f64() < 0.8 * serial.as_secs_f64(), "no overlap: {both:?} against {serial:?}");
}
