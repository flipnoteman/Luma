//! `file/` cases: arrays that live in a file (`Array::open`), paged through a device budget a
//! quarter of the file's size. Wave 2 reads each segment synchronously, once per kernel that needs
//! it (no schedules, no overlap of I/O with compute), so the numbers to look at are the achieved
//! GB/s next to a sequential `read` of the same file in the same run (printed under the case).
//! No threshold: the cold-read target is the acceptance number of the streaming spec.
//!
//! The file is written to `std::env::temp_dir()` (set `TMPDIR` to measure a real disk: on a
//! RAM-backed `/tmp` the "cold" cases only measure the copy out of the page cache).
//!
//! Sizes: 4 GiB with a 1 GiB budget on a GPU in a full run, 1 GiB with 256 MiB on the CPU backend, 256 MiB with 64 MiB with `--quick`.

use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use luma_io::{Advice, Extent, IoFile};

use crate::harness::*;
use luma::*;

/// Elements of the file.
fn elems(h: &Harness) -> usize {
    if h.quick {
        1 << 26
    } else if h.ctx.is_cpu() {
        1 << 28
    } else {
        1 << 30
    }
}

/// Removes the file when dropped.
struct Scratch(PathBuf);

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

fn scratch(name: &str) -> Scratch {
    Scratch(std::env::temp_dir().join(format!("luma-bench-{}-{name}.f32", std::process::id())))
}

/// Writes `n` values (a repeating ramp, not zeros: a sparse file would not read from the disk).
fn write_file(path: &Path, n: usize) {
    let piece: Vec<f32> = (0..1 << 20).map(|i| (i % 1021) as f32 * 0.125).collect();
    let mut out = std::io::BufWriter::with_capacity(1 << 22, std::fs::File::create(path).expect("create the bench file"));
    for _ in 0..n.div_ceil(piece.len()) {
        out.write_all(bytemuck_bytes(&piece[..piece.len().min(n)])).expect("write the bench file");
    }
    out.flush().unwrap();
    std::fs::OpenOptions::new().write(true).open(path).unwrap().set_len(n as u64 * 4).unwrap();
}

fn bytemuck_bytes(values: &[f32]) -> &[u8] {
    // SAFETY: `f32` has no padding and every byte is a valid `u8`.
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast(), std::mem::size_of_val(values)) }
}

/// Tells the page cache to forget the file.
fn drop_cache(path: &Path) {
    if let Ok(file) = IoFile::open(path, false) {
        file.advise(Extent { offset: 0, len: 0 }, Advice::DontNeed);
    }
}

/// GB/s of one sequential `read` of the whole file in 4 MiB pieces, after the cache was dropped (`cold`) or not.
fn baseline(path: &Path, cold: bool) -> f64 {
    if cold {
        drop_cache(path);
    }
    let mut file = std::fs::File::open(path).unwrap();
    let mut buf = vec![0u8; 4 << 20];
    let (mut total, started) = (0u64, Instant::now());
    loop {
        let n = file.read(&mut buf).unwrap();
        if n == 0 {
            break;
        }
        total += n as u64;
    }
    total as f64 / started.elapsed().as_secs_f64() / 1e9
}

/// GB/s of writing `bytes` bytes sequentially to a new file in 4 MiB pieces, synced at the end: the write ceiling of `file/axpy_to_file`.
fn write_baseline(path: &Path, bytes: u64) -> f64 {
    let piece = vec![0x5au8; 4 << 20];
    let started = Instant::now();
    let mut file = std::fs::File::create(path).expect("create the write baseline file");
    let mut left = bytes;
    while left > 0 {
        let n = left.min(piece.len() as u64) as usize;
        file.write_all(&piece[..n]).unwrap();
        left -= n as u64;
    }
    file.sync_data().unwrap();
    let rate = bytes as f64 / started.elapsed().as_secs_f64() / 1e9;
    drop(file);
    let _ = std::fs::remove_file(path);
    rate
}

/// GB/s of `passes` passes over `bytes` bytes that `f` makes.
fn rate(bytes: u64, passes: u64, f: &mut dyn FnMut()) -> f64 {
    let started = Instant::now();
    f();
    (bytes * passes) as f64 / started.elapsed().as_secs_f64() / 1e9
}

pub(super) fn run(h: &mut Harness) {
    // A decision log (gate 4) compares planner decisions: these cases add none and write gigabytes.
    if std::env::var_os("LUMA_BENCH_DECISIONS").is_some() {
        return;
    }
    if !["file/sum_cold", "file/sum_warm", "file/axpy_to_file", "file/chain3_reads_3x", "file/h2d_pinned"].iter().any(|case| h.want(case)) {
        return;
    }
    let n = elems(h);
    let bytes = n as u64 * 4;
    let budget = bytes / 4;
    let input = scratch("in");
    write_file(&input.0, n);
    let ctx = Context::with_options(ContextOptions::from_env().memory_budget(budget)).expect("no device");
    let x = Array::<f32>::open_on(&ctx, &input.0, FileOptions::raw().shape(n)).unwrap();
    let size = format!("{}MiB/{}MiB", bytes >> 20, budget >> 20);
    let ramp: f64 = (0..1usize << 20).map(|i| f64::from((i % 1021) as f32 * 0.125)).sum::<f64>() * (n >> 20) as f64;

    let check_sum = |sum: f32| {
        // The ramp is repeated, so the sum is known; f32 accumulation in trees stays within a relative 1e-3.
        assert!((f64::from(sum) - ramp).abs() <= ramp * 1e-3, "sum {sum} vs {ramp}");
    };

    h.transfer_case("file/sum_cold", &size, Work::Bytes(bytes as f64), &mut || {
        drop_cache(&input.0);
        check_sum(x.sum().item().unwrap());
    });
    let cold = rate(bytes, 1, &mut || {
        drop_cache(&input.0);
        x.sum().item().unwrap();
    });
    println!("  file/sum_cold: {cold:.2} GB/s, sequential read {:.2} GB/s ({:.0}%)", baseline(&input.0, true), 100.0 * cold / baseline(&input.0, true));

    h.transfer_case("file/sum_warm", &size, Work::Bytes(bytes as f64), &mut || {
        check_sum(x.sum().item().unwrap());
    });
    let warm = rate(bytes, 1, &mut || {
        x.sum().item().unwrap();
    });
    println!("  file/sum_warm: {warm:.2} GB/s, sequential read (cached) {:.2} GB/s", baseline(&input.0, false));

    // y = 2x + 1, stored into a file of its own: read a segment, write a segment.
    let output = scratch("out");
    if h.want("file/axpy_to_file") {
        let mut y = Array::<f32>::create_on(&ctx, &output.0, n, FileOptions::raw().sidecar(false)).unwrap();
        h.transfer_case("file/axpy_to_file", &size, Work::Bytes(2.0 * bytes as f64), &mut || {
            (&x * 2.0f32 + 1.0f32).store(&mut y).unwrap();
            y.flush().unwrap();
        });
        let axpy = rate(2 * bytes, 1, &mut || {
            (&x * 2.0f32 + 1.0f32).store(&mut y).unwrap();
            y.flush().unwrap();
        });
        let before = ctx.paging_stats();
        println!("  file/axpy_to_file: {axpy:.2} GB/s read + write ({} MiB written so far)", before.io_written_bytes >> 20);
        let (read, write) = (baseline(&input.0, false), write_baseline(&scratch("seqw").0, bytes));
        println!("  file/axpy_to_file ceiling: sequential read (cached) {read:.2} GB/s, sequential write {write:.2} GB/s; both in turn: {:.2} GB/s", 2.0 / (1.0 / read + 1.0 / write));
        // The last element of the last segment made it to the file.
        let tail = y.slice(0, n - 4..n).to_vec().unwrap();
        assert_eq!(tail, (n - 4..n).map(|i| 2.0 * ((i % (1 << 20)) % 1021) as f32 * 0.125 + 1.0).collect::<Vec<_>>());
        drop(y);
    }

    // The ceiling of the cold and warm rows: uploading host memory (no file, no kernel) to the device, the same number of bytes in 256 MiB pieces.
    if h.want("file/h2d_pinned") {
        let piece = n.min(1 << 26);
        let host: Vec<f32> = (0..piece).map(|i| (i % 1021) as f32 * 0.125).collect();
        let pieces = n / piece;
        h.transfer_case("file/h2d_pinned", &size, Work::Bytes((pieces * piece * 4) as f64), &mut || {
            for _ in 0..pieces {
                Array::from_slice_on(&ctx, &host, piece).unwrap().eval().unwrap();
            }
        });
    }

    // Three passes that each need the result of the one before: the file is read three times.
    let chain = || {
        let top = x.max().item().unwrap();
        let spread = (&x - top).abs().sum().item().unwrap();
        let square = ((&x - top) * (&x - top)).sum().item().unwrap();
        std::hint::black_box((spread, square));
    };
    let reads = ctx.paging_stats().io_read_bytes;
    h.transfer_case("file/chain3_reads_3x", &size, Work::Bytes(3.0 * bytes as f64), &mut || chain());
    let read = ctx.paging_stats().io_read_bytes - reads;
    let chain3 = rate(bytes, 3, &mut || chain());
    println!("  file/chain3_reads_3x: {chain3:.2} GB/s over three passes; {} MiB read in {} runs of {} MiB", read >> 20, read / bytes, bytes >> 20);
}
