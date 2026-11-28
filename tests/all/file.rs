//! File-backed arrays: open, create, store, write_to, flush; raw, sidecar and `.npy`; wide-lite.

use std::io::ErrorKind;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use luma::{Array, Context, ContextOptions, FileOptions, LumaError, C32, bf16, f16};
use luma_io::{FailPlan, FailingIo, Fault, PoolIo, TileIo};

/// A temporary file path that removes the file and its companions when dropped.
struct Temp(PathBuf);

impl Temp {
    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for Temp {
    fn drop(&mut self) {
        let mut sidecar = self.0.clone().into_os_string();
        sidecar.push(".luma");
        let mut marker = self.0.clone().into_os_string();
        marker.push(".luma-writing");
        for path in [self.0.clone(), PathBuf::from(sidecar), PathBuf::from(marker)] {
            let _ = std::fs::remove_file(path);
        }
    }
}

fn temp_file(name: &str) -> Temp {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let unique = NEXT.fetch_add(1, Ordering::Relaxed);
    Temp(std::env::temp_dir().join(format!("luma-file-test-{}-{unique}-{name}", std::process::id())))
}

/// A context with 4 KiB segments and a 64 KiB budget (host budget 64 KiB for file data on the CPU backend).
fn small() -> Context {
    Context::with_options(ContextOptions::from_env().calibrate(false).chunk_bytes(1 << 12).memory_budget(1 << 16).host_budget(Some(1 << 16))).unwrap()
}

/// A context without a memory budget (the CPU backend then bounds file data by the host budget).
fn unbudgeted() -> Context {
    Context::with_options(ContextOptions::from_env().calibrate(false).chunk_bytes(1 << 12).host_budget(Some(1 << 16))).unwrap()
}

fn values(len: usize, salt: usize) -> Vec<f32> {
    (0..len).map(|i| ((i * 7 + salt * 131) % 1021) as f32).collect()
}

fn bytes_of<T: bytemuck::Pod>(data: &[T]) -> Vec<u8> {
    bytemuck::cast_slice(data).to_vec()
}

fn root(error: &LumaError) -> &LumaError {
    error.root()
}

fn io_kind(error: &LumaError) -> Option<ErrorKind> {
    match root(error) {
        LumaError::Io { source, .. } => Some(source.kind()),
        _ => None,
    }
}

fn raw_shape(len: usize) -> FileOptions {
    FileOptions::raw().shape(len)
}

fn failing(plan: FailPlan) -> Arc<dyn TileIo> {
    Arc::new(FailingIo::new(PoolIo::new(2), plan))
}

// -- Formats --

fn round_trip<T: luma::GpuType + bytemuck::Pod + PartialEq + std::fmt::Debug>(ctx: &Context, data: Vec<T>) {
    let file = temp_file("round-trip");
    std::fs::write(file.path(), bytes_of(&data)).unwrap();
    let array = Array::<T>::open_on(ctx, file.path(), raw_shape(data.len())).unwrap();
    assert_eq!(array.to_vec().unwrap(), data, "{}", std::any::type_name::<T>());
    // Through a kernel too: reading from the device, not the file.
    let again = Array::<T>::open_on(ctx, file.path(), raw_shape(data.len())).unwrap();
    drop(array);
    assert_eq!(again.copy().to_vec().unwrap(), data, "{} copy", std::any::type_name::<T>());
}

#[test]
fn raw_round_trips_every_element_type() {
    let ctx = small();
    let n = 5003;
    round_trip::<f32>(&ctx, values(n, 1));
    round_trip::<u32>(&ctx, (0..n as u32).map(|i| i.wrapping_mul(2654435761)).collect());
    round_trip::<i32>(&ctx, (0..n as i32).map(|i| i.wrapping_mul(-40503)).collect());
    round_trip::<f64>(&ctx, (0..n).map(|i| i as f64 * 0.25).collect());
    round_trip::<u8>(&ctx, (0..n).map(|i| (i % 251) as u8).collect());
    round_trip::<i8>(&ctx, (0..n).map(|i| (i % 251) as i8).collect());
    round_trip::<u16>(&ctx, (0..n).map(|i| (i * 7) as u16).collect());
    round_trip::<i16>(&ctx, (0..n).map(|i| (i as i16).wrapping_mul(-13)).collect());
    round_trip::<f16>(&ctx, (0..n).map(|i| f16::from_f32((i % 100) as f32)).collect());
    round_trip::<bf16>(&ctx, (0..n).map(|i| bf16::from_f32((i % 100) as f32)).collect());
    round_trip::<C32>(&ctx, (0..n).map(|i| C32::new(i as f32, -(i as f32))).collect());
}

#[test]
fn open_checks_region_length_and_dtype() {
    let ctx = small();
    let file = temp_file("region");
    std::fs::write(file.path(), bytes_of(&values(100, 1))).unwrap();
    // Longer than the file.
    let error = Array::<f32>::open_on(&ctx, file.path(), raw_shape(101)).unwrap_err();
    assert_eq!(io_kind(&error), Some(ErrorKind::UnexpectedEof), "{error}");
    // A region at an offset that runs past the end.
    let error = Array::<f32>::open_on(&ctx, file.path(), raw_shape(100).offset(4)).unwrap_err();
    assert_eq!(io_kind(&error), Some(ErrorKind::UnexpectedEof), "{error}");
    // A region inside the file.
    let part = Array::<f32>::open_on(&ctx, file.path(), raw_shape(50).offset(40)).unwrap();
    assert_eq!(part.to_vec().unwrap(), values(100, 1)[10..60].to_vec());
    // Raw without a shape.
    let error = Array::<f32>::open_on(&ctx, file.path(), FileOptions::raw()).unwrap_err();
    assert!(matches!(error, LumaError::FileFormat { .. }), "{error}");
    // Types without a file layout.
    let error = Array::<bool>::open_on(&ctx, file.path(), raw_shape(100)).unwrap_err();
    assert!(matches!(error, LumaError::Unsupported(_)), "{error}");
    let error = Array::<luma::Df32>::open_on(&ctx, file.path(), raw_shape(100)).unwrap_err();
    assert!(matches!(error, LumaError::Unsupported(_)), "{error}");
    // A dtype that does not match the sidecar or the header.
    let created = Array::<u32>::create_on(&ctx, temp_file("dtype").path(), 16, FileOptions::raw()).unwrap();
    drop(created);
    let npy = temp_file("dtype-npy");
    Array::<f32>::create_on(&ctx, npy.path(), [4, 4], FileOptions::npy().recover()).unwrap().close().unwrap();
    let error = Array::<u32>::open_on(&ctx, npy.path(), FileOptions::npy()).unwrap_err();
    match &error {
        LumaError::FileFormat { message, .. } => assert!(message.contains("<f4") && message.contains("u32"), "{message}"),
        other => panic!("{other}"),
    }
}

fn data_file(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data").join(name)
}

#[test]
fn npy_round_trip_and_golden_header() {
    let ctx = small();
    // NumPy-written: a C-order (3, 4) array is [4, 3] in luma, no data is reordered.
    let c = Array::<f32>::open_on(&ctx, data_file("npy_v1_c.npy"), FileOptions::npy()).unwrap();
    assert_eq!(c.shape().dims(), [4, 3, 1, 1]);
    assert_eq!(c.to_vec().unwrap(), (0..12).map(|i| i as f32).collect::<Vec<_>>());
    // Fortran order keeps its dimensions; the file holds the elements column by column.
    let f = Array::<f32>::open_on(&ctx, data_file("npy_v1_f.npy"), FileOptions::npy()).unwrap();
    assert_eq!(f.shape().dims(), [3, 4, 1, 1]);
    assert_eq!(f.to_vec().unwrap(), vec![0.0, 4.0, 8.0, 1.0, 5.0, 9.0, 2.0, 6.0, 10.0, 3.0, 7.0, 11.0]);
    // Version 2, three dimensions, 16-bit integers.
    let v2 = Array::<i16>::open_on(&ctx, data_file("npy_v2_c.npy"), FileOptions::npy()).unwrap();
    assert_eq!(v2.shape().dims(), [4, 3, 2, 1]);
    assert_eq!(v2.to_vec().unwrap(), (0..24).collect::<Vec<i16>>());
    // A file created by luma has the header NumPy writes for a Fortran-order array of the same shape.
    let made = temp_file("golden.npy");
    let mut array = Array::<f32>::create_on(&ctx, made.path(), [3, 4], FileOptions::npy()).unwrap();
    array.write(&f.to_vec().unwrap()).unwrap();
    array.flush().unwrap();
    drop(array);
    let golden = std::fs::read(data_file("npy_v1_f.npy")).unwrap();
    assert_eq!(std::fs::read(made.path()).unwrap(), golden);
    // And it opens again, now complete.
    let again = Array::<f32>::open_on(&ctx, made.path(), FileOptions::npy()).unwrap();
    assert_eq!(again.to_vec().unwrap(), f.to_vec().unwrap());
}

#[test]
fn a_sidecar_gives_shape_and_dtype() {
    let ctx = small();
    let file = temp_file("sidecar");
    let data = values(6 * 7 * 5, 3);
    let mut array = Array::<f32>::create_on(&ctx, file.path(), [6, 7, 5], FileOptions::raw()).unwrap();
    array.write(&data).unwrap();
    array.close().unwrap();
    let sidecar = std::fs::read_to_string(format!("{}.luma", file.path().display())).unwrap();
    assert!(sidecar.contains("dtype = f32") && sidecar.contains("state = complete"), "{sidecar}");
    let opened = Array::<f32>::open_on(&ctx, file.path(), FileOptions::raw()).unwrap();
    assert_eq!(opened.shape().dims(), [6, 7, 5, 1]);
    assert_eq!(opened.to_vec().unwrap(), data);
    // The wrong element type is refused, and so is a shape that disagrees.
    drop(opened);
    let error = Array::<u32>::open_on(&ctx, file.path(), FileOptions::raw()).unwrap_err();
    assert!(matches!(error, LumaError::FileFormat { .. }), "{error}");
    let error = Array::<f32>::open_on(&ctx, file.path(), FileOptions::raw().shape(210)).unwrap_err();
    assert!(matches!(error, LumaError::FileFormat { .. }), "{error}");
}

#[test]
fn a_file_being_written_is_refused_until_recover() {
    let ctx = small();
    for (npy, name) in [(false, "writing-raw"), (true, "writing-npy")] {
        let file = temp_file(name);
        let opts = || if npy { FileOptions::npy() } else { FileOptions::raw() };
        let array = Array::<f32>::create_on(&ctx, file.path(), 64, opts()).unwrap();
        // The writer "dies": the file is closed without a flush (no data written, so nothing was completed).
        drop(array);
        let error = Array::<f32>::open_on(&ctx, file.path(), opts()).unwrap_err();
        assert!(matches!(&error, LumaError::FileFormat { message, .. } if message.contains("recover")), "{error}");
        let recovered = Array::<f32>::open_on(&ctx, file.path(), opts().recover()).unwrap();
        assert_eq!(recovered.to_vec().unwrap(), vec![0.0; 64]);
    }
}

// -- Paging --

fn big_file(name: &str, len: usize) -> (Temp, Vec<f32>) {
    let file = temp_file(name);
    let data = values(len, 5);
    std::fs::write(file.path(), bytes_of(&data)).unwrap();
    (file, data)
}

#[test]
fn a_file_larger_than_the_budget_sums_like_the_same_data_in_memory() {
    let ctx = small();
    let len = 1 << 18;
    let (file, data) = big_file("sum", len);
    let from_file = Array::<f32>::open_on(&ctx, file.path(), raw_shape(len)).unwrap();
    let before = ctx.paging_stats();
    let sum = (&from_file * 2.0f32 + 1.0f32).sum().item().unwrap();
    let read = ctx.paging_stats().io_read_bytes - before.io_read_bytes;
    let in_memory = Array::from_slice_on(&ctx, &data, len).unwrap();
    let expected = (&in_memory * 2.0f32 + 1.0f32).sum().item().unwrap();
    assert_eq!(sum.to_bits(), expected.to_bits(), "{sum} vs {expected}");
    assert_eq!(read, 4 * len as u64, "one fused chain reads the file once");
}

#[test]
fn evicting_a_clean_file_segment_reads_nothing_back() {
    let ctx = small();
    let len = 1 << 18;
    let (file, _) = big_file("clean", len);
    let array = Array::<f32>::open_on(&ctx, file.path(), raw_shape(len)).unwrap();
    array.sum().item().unwrap();
    array.max().item().unwrap();
    let stats = ctx.paging_stats();
    assert_eq!(stats.readback_evictions, 0, "{stats:?}");
    // With split on, the sum streams through the ring and no segment is made resident, so nothing is evicted.
    if ctx.stream_stats().streamed == 0 {
        assert!(stats.home_drops > 0, "{stats:?}");
    }
    assert_eq!(stats.io_written_bytes, 0);
    assert_eq!(ctx.spilled_bytes(), 0);
}

#[test]
fn to_vec_of_a_file_array_reads_without_uploading() {
    let ctx = small();
    let len = 10_000;
    let (file, data) = big_file("to-vec", len);
    let array = Array::<f32>::open_on(&ctx, file.path(), raw_shape(len)).unwrap();
    assert_eq!(array.to_vec().unwrap(), data);
    let stats = ctx.paging_stats();
    assert_eq!(stats.io_read_bytes, 4 * len as u64);
    assert_eq!(stats.file_resident_bytes, 0, "no segment came to the device");
    assert_eq!(ctx.memory_stats().resident_bytes, 0);
    let mut out = vec![0.0f32; len];
    array.read_into(&mut out).unwrap();
    assert_eq!(out, data);
}

#[test]
fn a_chain_of_two_kernels_reads_the_file_twice() {
    // Wave 2 has no schedules: a pass that needs the result of another one reads the file again.
    let ctx = small();
    let len = 1 << 16;
    let (file, data) = big_file("twice", len);
    let array = Array::<f32>::open_on(&ctx, file.path(), raw_shape(len)).unwrap();
    let mean = array.sum() / len as f32;
    let centered = &array - &mean;
    let var = (&centered * &centered).sum().item().unwrap();
    let m = data.iter().map(|&v| f64::from(v)).sum::<f64>() / len as f64;
    let expected = data.iter().map(|&v| (f64::from(v) - m).powi(2)).sum::<f64>();
    assert!((f64::from(var) - expected).abs() <= expected * 1e-3, "{var} vs {expected}");
    assert_eq!(ctx.paging_stats().io_read_bytes, 2 * 4 * len as u64);
}

#[test]
fn the_cpu_host_budget_bounds_file_data() {
    let ctx = unbudgeted();
    if !ctx.is_cpu() {
        return;
    }
    let len = 1 << 18;
    let (file, _) = big_file("host-budget", len);
    let array = Array::<f32>::open_on(&ctx, file.path(), raw_shape(len)).unwrap();
    array.sum().item().unwrap();
    let stats = ctx.paging_stats();
    assert!(stats.file_resident_bytes <= 1 << 17, "{stats:?}");
    assert!(stats.home_drops > 0, "{stats:?}");
}

// -- Writes --

#[test]
fn store_into_a_created_file_matches_the_in_memory_result() {
    let ctx = small();
    let len = 1 << 16;
    let x = values(len, 1);
    let y = values(len, 2);
    let a = Array::from_slice_on(&ctx, &x, len).unwrap();
    let b = Array::from_slice_on(&ctx, &y, len).unwrap();
    let file = temp_file("store");
    let mut target = Array::<f32>::create_on(&ctx, file.path(), len, FileOptions::raw()).unwrap();
    (&a * 2.0f32 + &b).store(&mut target).unwrap();
    target.flush().unwrap();
    let expected: Vec<f32> = x.iter().zip(&y).map(|(x, y)| x * 2.0 + y).collect();
    assert_eq!(target.to_vec().unwrap(), expected);
    drop(target);
    let reopened = Array::<f32>::open_on(&ctx, file.path(), FileOptions::raw()).unwrap();
    assert_eq!(reopened.to_vec().unwrap(), expected);
    // Storing an array that has a buffer already, and a file into a file.
    let copy_file = temp_file("store-copy");
    reopened.write_to(copy_file.path(), FileOptions::raw()).unwrap();
    assert_eq!(std::fs::read(copy_file.path()).unwrap(), bytes_of(&expected));
}

#[test]
fn flush_makes_the_data_durable_and_reopen_reads_it() {
    let ctx = small();
    let len = 1 << 14;
    let data = values(len, 4);
    let file = temp_file("flush");
    let mut array = Array::<f32>::create_on(&ctx, file.path(), len, FileOptions::raw()).unwrap();
    array.write(&data).unwrap();
    array.flush().unwrap();
    // The file already holds the data while the array lives.
    assert_eq!(std::fs::read(file.path()).unwrap(), bytes_of(&data));
    let sidecar = std::fs::read_to_string(format!("{}.luma", file.path().display())).unwrap();
    assert!(sidecar.contains("state = complete"), "{sidecar}");
    array.close().unwrap();
    let again = Array::<f32>::open_on(&ctx, file.path(), FileOptions::raw()).unwrap();
    assert_eq!(again.to_vec().unwrap(), data);

    // An .npy gets its marker removed.
    let npy = temp_file("flush.npy");
    let mut array = Array::<f32>::create_on(&ctx, npy.path(), len, FileOptions::npy()).unwrap();
    let marker = format!("{}.luma-writing", npy.path().display());
    assert!(Path::new(&marker).exists());
    array.write(&data).unwrap();
    array.flush().unwrap();
    assert!(!Path::new(&marker).exists());
}

#[test]
fn a_dirty_segment_is_written_back_on_eviction_and_on_drop() {
    let ctx = small();
    let len = 1 << 16;
    let data = values(len, 6);
    let file = temp_file("dirty");
    let mut array = Array::<f32>::create_on(&ctx, file.path(), len, FileOptions::raw()).unwrap();
    array.write(&data).unwrap();
    // 256 KiB written through a 64 KiB budget: the early segments left dirty.
    let stats = ctx.paging_stats();
    assert!(stats.home_writebacks > 0 && stats.io_written_bytes > 0, "{stats:?}");
    // The rest goes at drop.
    drop(array);
    assert_eq!(std::fs::read(file.path()).unwrap(), bytes_of(&data));
}

#[test]
fn assign_and_compound_assignment_work_on_a_writable_file_array() {
    let ctx = small();
    let len = 1000;
    let data = values(len, 7);
    let file = temp_file("assign");
    std::fs::write(file.path(), bytes_of(&data)).unwrap();
    let mut array = Array::<f32>::open_on(&ctx, file.path(), raw_shape(len).writable()).unwrap();
    array += 1.0f32;
    array.slice_mut(0, ..10).assign(-5.0f32).unwrap();
    let mut expected: Vec<f32> = data.iter().map(|v| v + 1.0).collect();
    expected[..10].fill(-5.0);
    assert_eq!(array.to_vec().unwrap(), expected);
    array.close().unwrap();
    assert_eq!(std::fs::read(file.path()).unwrap(), bytes_of(&expected));
}

#[test]
fn write_to_is_atomic() {
    let ctx = small();
    let len = 1 << 14;
    let data = values(len, 8);
    let source = Array::from_slice_on(&ctx, &data, len).unwrap();
    let dir = std::env::temp_dir();
    for (npy, name) in [(true, "atomic.npy"), (false, "atomic.f32")] {
        let target = temp_file(name);
        let opts = |plan: Option<FailPlan>| {
            let opts = if npy { FileOptions::npy() } else { FileOptions::raw() };
            match plan {
                Some(plan) => opts.engine(failing(plan)),
                None => opts,
            }
        };
        // Header (npy) and four 16 KiB... segments are the writes; fail the last of them.
        let header = usize::from(npy);
        let segments = 4 * len / 4096;
        let plan = FailPlan { write: Some(((header + segments - 1) as u64, Fault::Fail(ErrorKind::Other))), ..FailPlan::default() };
        let error = source.write_to(target.path(), opts(Some(plan))).unwrap_err();
        assert!(matches!(root(&error), LumaError::Io { .. }), "{error}");
        assert!(!target.path().exists(), "{name}: the target appeared");
        let leftovers: Vec<_> = std::fs::read_dir(&dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .filter(|n| n.starts_with(&format!("luma-file-test-{}-", std::process::id())) && n.contains(name) && n.ends_with(".tmp"))
            .collect();
        assert!(leftovers.is_empty(), "{leftovers:?}");

        // An existing file stays as it was.
        std::fs::write(target.path(), b"old").unwrap();
        let error = source.write_to(target.path(), opts(Some(plan)).overwrite()).unwrap_err();
        assert!(matches!(root(&error), LumaError::Io { .. }), "{error}");
        assert_eq!(std::fs::read(target.path()).unwrap(), b"old");
        // Without overwrite an existing file is an error too.
        let error = source.write_to(target.path(), opts(None)).unwrap_err();
        assert_eq!(io_kind(&error), Some(ErrorKind::AlreadyExists));
        // And it works.
        source.write_to(target.path(), opts(None).overwrite()).unwrap();
        let back = Array::<f32>::open_on(&ctx, target.path(), if npy { FileOptions::npy() } else { FileOptions::raw() }).unwrap();
        assert_eq!(back.to_vec().unwrap(), data, "{name}");
    }
}

// -- I5 and read-only --

const SCALE: &str = r#"
struct Params { n: u32 }
@group(1) @binding(0) var<storage, read_write> y: array<f32>;
@group(1) @binding(1) var<uniform> p: Params;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= p.n) { return; }
    y[gid.x] = y[gid.x] * 2.0;
}
"#;

#[derive(encase::ShaderType)]
struct ScaleParams {
    n: u32,
}

#[test]
fn a_read_only_file_is_never_modified() {
    let ctx = small();
    let len = 1000;
    let (file, data) = big_file("read-only", len);
    let before = std::fs::read(file.path()).unwrap();
    {
        // The temporary whose operand array was dropped: it must not take the file's buffer over.
        let array = Array::<f32>::open_on(&ctx, file.path(), raw_shape(len)).unwrap();
        let temporary = &array * 2.0f32;
        drop(array);
        assert_eq!(temporary.to_vec().unwrap(), data.iter().map(|v| v * 2.0).collect::<Vec<_>>());
    }
    let mut array = Array::<f32>::open_on(&ctx, file.path(), raw_shape(len)).unwrap();
    let expect_read_only = |result: luma::Result<()>| match result {
        Err(error) => assert!(matches!(root(&error), LumaError::ReadOnly(_)), "{error}"),
        Ok(()) => panic!("a write to a read-only file array succeeded"),
    };
    expect_read_only(array.write(&data));
    expect_read_only(array.slice_mut(0, ..4).assign(1.0f32));
    let mut compound = array.clone();
    compound += 1.0f32;
    expect_read_only(compound.check());
    let kernel = ctx.kernel(SCALE).build().unwrap();
    expect_read_only(kernel.launch([len as u32], luma::args![&mut array, luma::uniform(&ScaleParams { n: len as u32 })]));
    expect_read_only(array.store_check());
    drop((array, compound));
    assert_eq!(std::fs::read(file.path()).unwrap(), before);
}

trait StoreCheck {
    fn store_check(&mut self) -> luma::Result<()>;
}

impl StoreCheck for Array<f32> {
    fn store_check(&mut self) -> luma::Result<()> {
        let source = Array::<f32>::zeros_on(self.context(), self.shape())?;
        source.store(self)
    }
}

#[test]
fn a_pin_for_write_on_a_read_only_store_detaches_the_home() {
    let ctx = small();
    let len = 1000;
    let (file, data) = big_file("detach", len);
    let before = std::fs::read(file.path()).unwrap();
    let array = Array::<f32>::open_on(&ctx, file.path(), raw_shape(len)).unwrap();
    let drops = ctx.paging_stats().home_drops;
    // An operation that pins its operand as a writer (the kernel launch API's `&mut`-free pin of `sort`).
    let sorted = array.sort();
    sorted.to_vec().unwrap();
    assert!(ctx.paging_stats().home_drops > drops, "no write pin reached the file array");
    // Evict everything: the data survives (copy on write), the file is untouched.
    let filler: Vec<Array<f32>> = (0..40).map(|k| Array::from_slice_on(&ctx, &values(4096, k), 4096).unwrap()).collect();
    for filler in &filler {
        filler.sum().item().unwrap();
    }
    assert_eq!(array.to_vec().unwrap(), data);
    drop(array);
    assert_eq!(std::fs::read(file.path()).unwrap(), before);
}

#[test]
fn a_file_array_is_never_taken_over_in_place() {
    let ctx = small();
    let len = 1000;
    let file = temp_file("in-place");
    let data = values(len, 9);
    std::fs::write(file.path(), bytes_of(&data)).unwrap();
    let array = Array::<f32>::open_on(&ctx, file.path(), raw_shape(len).writable()).unwrap();
    let temporary = &array * 2.0f32 + 1.0f32;
    drop(array);
    assert_eq!(temporary.to_vec().unwrap(), data.iter().map(|v| v * 2.0 + 1.0).collect::<Vec<_>>());
    drop(temporary);
    assert_eq!(std::fs::read(file.path()).unwrap(), bytes_of(&data));
}

// -- Locks and the registry --

#[test]
fn a_second_open_of_a_writable_extent_is_refused() {
    let ctx = small();
    let file = temp_file("busy");
    std::fs::write(file.path(), bytes_of(&values(200, 1))).unwrap();
    let first = Array::<f32>::open_on(&ctx, file.path(), raw_shape(100).writable()).unwrap();
    // The same extent, or part of it, is busy: for a reader too.
    let error = Array::<f32>::open_on(&ctx, file.path(), raw_shape(100)).unwrap_err();
    assert_eq!(io_kind(&error), Some(ErrorKind::ResourceBusy), "{error}");
    let error = Array::<f32>::open_on(&ctx, file.path(), raw_shape(50).offset(200).writable()).unwrap_err();
    assert_eq!(io_kind(&error), Some(ErrorKind::ResourceBusy), "{error}");
    // A disjoint extent of the same file does not overlap.
    let second = Array::<f32>::open_on(&ctx, file.path(), raw_shape(100).offset(400).writable()).unwrap();
    assert_eq!(second.to_vec().unwrap(), values(200, 1)[100..].to_vec());
    drop((first, second));
}

#[test]
fn identical_reopens_share_the_storage() {
    let ctx = Context::with_options(ContextOptions::from_env().calibrate(false)).unwrap();
    let len = 10_000;
    let (file, data) = big_file("share", len);
    let a = Array::<f32>::open_on(&ctx, file.path(), raw_shape(len)).unwrap();
    let b = Array::<f32>::open_on(&ctx, file.path(), raw_shape(len)).unwrap();
    assert_eq!(a.sum().item().unwrap(), b.sum().item().unwrap());
    // One storage: the second array's sum found the segments resident.
    assert_eq!(ctx.paging_stats().io_read_bytes, 4 * len as u64);
    assert_eq!(b.to_vec().unwrap(), data);
}

#[test]
fn the_lock_is_held_while_the_array_lives() {
    let ctx = small();
    let file = temp_file("lock");
    let array = Array::<f32>::create_on(&ctx, file.path(), 100, FileOptions::raw()).unwrap();
    let other = std::fs::File::open(file.path()).unwrap();
    assert!(matches!(other.try_lock_shared(), Err(std::fs::TryLockError::WouldBlock)));
    drop(array);
    other.try_lock_shared().unwrap();
    // A read-only array holds a shared lock: readers coexist, a writer does not.
    let reader = Array::<f32>::open_on(&ctx, file.path(), raw_shape(100).recover()).unwrap();
    other.try_lock_shared().unwrap();
    drop(other);
    let writer = std::fs::OpenOptions::new().read(true).write(true).open(file.path()).unwrap();
    assert!(matches!(writer.try_lock(), Err(std::fs::TryLockError::WouldBlock)));
    drop(reader);
    writer.try_lock().unwrap();
}

#[test]
fn lock_false_skips_the_lock() {
    let ctx = small();
    let file = temp_file("no-lock");
    let array = Array::<f32>::create_on(&ctx, file.path(), 100, FileOptions::raw().lock(false)).unwrap();
    let other = std::fs::File::open(file.path()).unwrap();
    other.try_lock_shared().unwrap();
    drop(array);
}

// -- Errors --

#[test]
fn an_io_error_surfaces_at_the_next_waiting_call_and_poisons_the_array() {
    let ctx = small();
    let len = 1 << 14;
    let (file, _) = big_file("io-error", len);
    for fault in [Fault::Fail(ErrorKind::Other), Fault::Short { keep: 100 }] {
        let engine = failing(FailPlan { read: Some((1, fault)), ..FailPlan::default() });
        let array = Array::<f32>::open_on(&ctx, file.path(), raw_shape(len).engine(engine)).unwrap();
        let doubled = &array * 2.0f32;
        let error = doubled.to_vec().unwrap_err();
        assert!(matches!(root(&error), LumaError::Io { op: "read", .. }), "{error}");
        // The result stays poisoned with the same error.
        assert!(doubled.check().is_err());
        assert!(matches!(root(&(&doubled + 1.0f32).to_vec().unwrap_err()), LumaError::Io { .. }));
        drop((array, doubled));
    }
}

#[test]
fn enospc_fails_create() {
    // A file of a petabyte: the file system refuses to size it, or to reserve its space.
    let ctx = small();
    let file = temp_file("enospc");
    let error = Array::<f32>::create_on(&ctx, file.path(), [1 << 24, 1 << 24], FileOptions::raw()).unwrap_err();
    assert!(matches!(error, LumaError::Io { .. }), "{error}");
    assert!(!file.path().exists(), "a failed create leaves nothing behind");
}

#[test]
fn a_failed_write_back_keeps_the_segment_dirty() {
    let ctx = small();
    let len = 1 << 12;
    let data = values(len, 10);
    let file = temp_file("failed-write");
    let engine = failing(FailPlan { write: Some((0, Fault::Fail(ErrorKind::Other))), ..FailPlan::default() });
    let mut array = Array::<f32>::create_on(&ctx, file.path(), len, FileOptions::raw().sidecar(false).engine(engine)).unwrap();
    array.write(&data).unwrap();
    let error = array.flush().unwrap_err();
    assert!(matches!(root(&error), LumaError::Io { op: "write", .. }), "{error}");
    // Nothing was lost: the array still reads its data, and the next flush writes the rest.
    assert_eq!(array.to_vec().unwrap(), data);
    array.flush().unwrap();
    drop(array);
    assert_eq!(std::fs::read(file.path()).unwrap(), bytes_of(&data));
}

// -- Interaction --

#[test]
fn p1b_host_copies_and_file_homes_do_not_double_count_host_bytes() {
    let ctx = small();
    let len = 1 << 16;
    let (file, _) = big_file("host-bytes", len);
    let array = Array::<f32>::open_on(&ctx, file.path(), raw_shape(len)).unwrap();
    array.sum().item().unwrap();
    array.max().item().unwrap();
    let stats = ctx.paging_stats();
    assert_eq!(stats.host_copy_bytes, 0, "the file is the clean copy: no host copy is kept");
    assert_eq!(ctx.spilled_bytes(), 0);
}

#[test]
fn lineage_eviction_is_refused_for_an_away_input() {
    let ctx = small();
    let len = 1000;
    let (file, data) = big_file("lineage", len);
    let array = Array::<f32>::open_on(&ctx, file.path(), raw_shape(len)).unwrap();
    let doubled = &array * 2.0f32;
    doubled.eval().unwrap();
    // Push everything off the device: the file array is away, so nothing is evicted by recompute over it.
    let filler: Vec<Array<f32>> = (0..40).map(|k| Array::from_slice_on(&ctx, &values(4096, k), 4096).unwrap()).collect();
    for filler in &filler {
        filler.sum().item().unwrap();
    }
    assert_eq!(doubled.to_vec().unwrap(), data.iter().map(|v| v * 2.0).collect::<Vec<_>>());
    assert_eq!(ctx.paging_stats().recompute_evictions, 0);
}

// -- Wide-lite --

/// A sparse file of `2^32 + 1` bytes with a 1 at both ends, opened as `[6700417, 641]` `u8`.
#[cfg(unix)]
fn wide_file() -> (Temp, Array<u8>) {
    let file = temp_file("wide");
    let len = (1u64 << 32) + 1;
    {
        use std::os::unix::fs::FileExt;
        let handle = std::fs::File::create(file.path()).unwrap();
        handle.set_len(len).unwrap();
        handle.write_at(&[1], 0).unwrap();
        handle.write_at(&[1], len - 1).unwrap();
    }
    let ctx = Context::with_options(ContextOptions::from_env().calibrate(false).chunk_bytes(1 << 30).host_budget(Some(1 << 30))).unwrap();
    let array = Array::<u8>::open_on(&ctx, file.path(), FileOptions::raw().shape([6_700_417, 641])).unwrap();
    (file, array)
}

#[test]
#[cfg(unix)]
fn a_wide_u8_file_sums_ends_and_all() {
    let (_file, array) = wide_file();
    assert_eq!(array.len(), (1usize << 32) + 1);
    assert_eq!(array.sum().item().unwrap(), 2);
    assert_eq!(array.max().item().unwrap(), 1);
    assert_eq!(array.min().item().unwrap(), 0);
    let nonzero = array.to_u32().gt(0u32);
    assert!(nonzero.any().item().unwrap());
    assert!(!nonzero.all().item().unwrap());
    // The user divides by the count in f64.
    let mean = f64::from(array.sum().item().unwrap()) / array.len() as f64;
    assert!(mean > 0.0 && mean < 1e-9);
}

#[test]
#[cfg(unix)]
fn wide_arrays_refuse_shifts_generators_views_axis_reductions_and_sorts() {
    let (_file, array) = wide_file();
    let refused = |what: &str, result: luma::Result<()>| match result {
        Err(error) => assert!(matches!(root(&error), LumaError::DimensionTooLarge { .. } | LumaError::Chunked(_)), "{what}: {error}"),
        Ok(()) => panic!("{what} gave a value on a wide array"),
    };
    refused("shift", array.shift(1).eval());
    refused("slice", array.slice(0, ..10).eval());
    refused("flip", array.flip(0).eval());
    refused("reshape", array.reshape(array.len()).eval());
    refused("sum_axis", array.sum_axis(0).eval());
    refused("sort", array.sort().eval());
    refused("argmax", array.argmax().eval());
    refused("count", array.to_u32().gt(0u32).count().eval());
}

#[test]
#[cfg(unix)]
fn wide_arrays_refuse_mean_var_and_std() {
    let (_file, array) = wide_file();
    let wide = array.to_f32();
    for (what, result) in [
        ("mean", wide.mean().eval()),
        ("var", wide.var(luma::Bias::Population).eval()),
        ("std", wide.std(luma::Bias::Population).eval()),
        ("u8 mean", array.mean().eval()),
    ] {
        match result {
            Err(error) => assert!(matches!(root(&error), LumaError::DimensionTooLarge { .. }), "{what}: {error}"),
            Ok(()) => panic!("{what} gave a value on a wide array"),
        }
    }
}

#[test]
#[cfg(unix)]
#[ignore = "16 GiB of disk and several minutes; run with LUMA_TEST_WIDE=1 -- --ignored"]
fn a_wide_f32_file_of_16_gib_sums() {
    if std::env::var_os("LUMA_TEST_WIDE").is_none() {
        return;
    }
    let file = temp_file("wide-f32");
    let len = 4u64 << 30;
    {
        use std::os::unix::fs::FileExt;
        let handle = std::fs::File::create(file.path()).unwrap();
        handle.set_len(len * 4).unwrap();
        handle.write_at(&1.0f32.to_le_bytes(), 0).unwrap();
        handle.write_at(&2.0f32.to_le_bytes(), len * 4 - 4).unwrap();
    }
    let array = Array::<f32>::open(file.path(), FileOptions::raw().shape([1 << 16, 1 << 16])).unwrap();
    assert_eq!(array.sum().item().unwrap(), 3.0);
}
