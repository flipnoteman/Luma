//! The on-disk cache: what compiling pipelines and measuring the device cost, kept between
//! processes.
//!
//! One directory ([`directory`]: `LUMA_CACHE_DIR`, else `$XDG_CACHE_HOME/luma`, else
//! `~/.cache/luma`; `LUMA_CACHE=off` disables it) holds, per adapter and driver (see
//! [`adapter_key`]), the data of wgpu's `PipelineCache` (the driver's compiled pipelines, which
//! turns a seconds-long first call of a large kernel into milliseconds) and the calibrated
//! [`CostModel`]. Every file is written to a temporary name and renamed over the old one, so
//! concurrent processes never see a half-written file (the last writer wins); every file carries
//! a checksum, and one that does not match is ignored as if it were absent. Nothing here ever
//! fails an operation: unreadable or unwritable cache files only cost the speed-up.

use std::ffi::OsString;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use log::debug;

use super::lock;
use crate::jit::cost::CostModel;

/// Layout version of the files; bumping it makes older files unreadable (and so replaced).
const FORMAT: u32 = 1;
/// Magic of the pipeline cache file.
const PIPELINES: [u8; 8] = *b"LUMAPIPE";
/// Magic of the calibration file (2: measured with the faster `Df32` library, so older files
/// are not read).
const COST: [u8; 8] = *b"LUMACOS2";
/// Magic of the file with the result of the `fma` test.
const FMA: [u8; 8] = *b"LUMAFMA ";
/// The largest pipeline cache that is kept, in bytes. NVIDIA's driver (615.71.09) works through a
/// loaded cache on compiler threads of its own, and with a cache of 116 MB (the kernels of many
/// programs and versions) those threads were still busy when the process ended and corrupted the
/// heap or crashed in about one run in five; caches of a few MB never did. A larger cache is
/// dropped (not loaded, not written) and starts again from empty.
const MAX_PIPELINES: usize = 32 << 20;
/// Magic, format, payload length, payload checksum.
const HEADER: usize = 8 + 4 + 8 + 8;

/// FNV-1a: a stable 64-bit hash, for checksums and file names.
fn fnv1a(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |hash, &byte| (hash ^ u64::from(byte)).wrapping_mul(0x0100_0000_01b3))
}

/// Whether the environment value `text` switches a feature off (`0`, `off`, `false`).
fn is_off(text: &OsString) -> bool {
    text.to_str().is_some_and(|text| ["0", "off", "false"].contains(&text.to_ascii_lowercase().as_str()))
}

/// The cache directory for the environment `var` describes, `None` when caching is off or no
/// directory can be found.
fn resolve(var: impl Fn(&str) -> Option<OsString>) -> Option<PathBuf> {
    if var("LUMA_CACHE").is_some_and(|text| is_off(&text)) {
        return None;
    }
    let set = |name: &str| var(name).filter(|value| !value.is_empty());
    if let Some(dir) = set("LUMA_CACHE_DIR") {
        return Some(PathBuf::from(dir));
    }
    if let Some(base) = set("XDG_CACHE_HOME").map(PathBuf::from).filter(|base| base.is_absolute()) {
        return Some(base.join("luma"));
    }
    if cfg!(windows)
        && let Some(base) = set("LOCALAPPDATA")
    {
        return Some(PathBuf::from(base).join("luma"));
    }
    set("HOME").or_else(|| set("USERPROFILE")).map(|home| PathBuf::from(home).join(".cache").join("luma"))
}

/// The cache directory of this process, from the environment.
///
/// Unit tests of the crate use a directory under the system's temporary directory unless
/// `LUMA_CACHE_DIR` or `LUMA_CACHE` is set, so that they neither read nor fill the user's cache.
pub(crate) fn directory() -> Option<PathBuf> {
    #[cfg(test)]
    if std::env::var_os("LUMA_CACHE_DIR").is_none() && std::env::var_os("LUMA_CACHE").is_none() {
        return Some(std::env::temp_dir().join("luma-test-cache"));
    }
    resolve(|name| std::env::var_os(name))
}

/// Names the adapter and driver the cached data belong to: backend, vendor and device ids, and a
/// hash of the adapter's name, driver and driver version, so that a driver update starts a new
/// file instead of rewriting the old one back and forth.
fn adapter_key(info: &wgpu::AdapterInfo) -> String {
    let identity = format!("{}\n{}\n{}", info.name, info.driver, info.driver_info);
    format!("{:?}-{:04x}-{:04x}-{:016x}", info.backend, info.vendor, info.device, fnv1a(identity.as_bytes())).to_ascii_lowercase()
}

/// The payload of the file at `path` if it is complete, of this format and has the checksum its
/// header states.
fn read_blob(path: &Path, magic: [u8; 8]) -> Option<Vec<u8>> {
    let bytes = std::fs::read(path).ok()?;
    let header = bytes.get(..HEADER)?;
    let payload = &bytes[HEADER..];
    let format = u32::from_le_bytes(header[8..12].try_into().ok()?);
    let length = u64::from_le_bytes(header[12..20].try_into().ok()?);
    let checksum = u64::from_le_bytes(header[20..28].try_into().ok()?);
    let valid = header[..8] == magic && format == FORMAT && length == payload.len() as u64 && checksum == fnv1a(payload);
    valid.then(|| payload.to_vec())
}

/// Distinguishes the temporary files of threads of one process.
static TEMPORARY: AtomicU64 = AtomicU64::new(0);

/// Writes `payload` to `path` atomically: a temporary file in the same directory, renamed over
/// `path`.
fn write_blob(path: &Path, magic: [u8; 8], payload: &[u8]) -> std::io::Result<()> {
    let dir = path.parent().unwrap_or(Path::new("."));
    std::fs::create_dir_all(dir)?;
    let temporary = dir.join(format!(
        ".{}.{}.{}.tmp",
        path.file_name().and_then(|name| name.to_str()).unwrap_or("luma"),
        std::process::id(),
        TEMPORARY.fetch_add(1, Ordering::Relaxed)
    ));
    let written = (|| {
        let mut file = std::fs::File::create(&temporary)?;
        let mut head = Vec::with_capacity(HEADER);
        head.extend_from_slice(&magic);
        head.extend_from_slice(&FORMAT.to_le_bytes());
        head.extend_from_slice(&(payload.len() as u64).to_le_bytes());
        head.extend_from_slice(&fnv1a(payload).to_le_bytes());
        file.write_all(&head)?;
        file.write_all(payload)?;
        file.flush()?;
        std::fs::rename(&temporary, path)
    })();
    if written.is_err() {
        let _ = std::fs::remove_file(&temporary);
    }
    written
}

/// What was last written.
struct Saved {
    /// Hash of the pipeline cache data on disk (or loaded from it), 0 if none.
    hash: u64,
    /// Whether pipelines were built since.
    pending: bool,
}

/// A device's view of the cache directory.
pub(crate) struct DiskCache {
    /// The directory, `None` when caching is off.
    dir: Option<PathBuf>,
    /// [`adapter_key`] of the device.
    key: String,
    /// wgpu's pipeline cache of the device, `None` without `PIPELINE_CACHE` or without a directory.
    pipelines: Option<wgpu::PipelineCache>,
    saved: Mutex<Saved>,
}

impl std::fmt::Debug for DiskCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DiskCache").field("dir", &self.dir).field("key", &self.key).field("pipelines", &self.pipelines.is_some()).finish()
    }
}

/// The caches the exit handler writes (see [`flush_at_exit`]).
#[cfg(target_os = "linux")]
static OPEN: Mutex<Vec<Arc<DiskCache>>> = Mutex::new(Vec::new());

impl DiskCache {
    /// Opens the cache of `device` (described by `info`) in `dir`, loading the stored pipeline
    /// cache if there is one. Without a directory, or on a device without
    /// [`wgpu::Features::PIPELINE_CACHE`], there is no pipeline cache: pipelines compile as usual.
    pub(crate) fn open(dir: Option<PathBuf>, device: &wgpu::Device, info: &wgpu::AdapterInfo) -> Arc<DiskCache> {
        let key = adapter_key(info);
        let mut hash = 0;
        let pipelines = dir.as_ref().filter(|_| device.features().contains(wgpu::Features::PIPELINE_CACHE)).map(|dir| {
            let stored = read_blob(&dir.join(format!("pipelines-{key}.bin")), PIPELINES).filter(|stored| stored.len() <= MAX_PIPELINES);
            hash = stored.as_deref().map_or(0, fnv1a);
            // SAFETY: the data is empty or was returned by `PipelineCache::get_data` (the
            // checksum guards against a damaged file), and wgpu and the driver check that it
            // belongs to this adapter and driver; with `fallback` an unusable blob only yields
            // an empty cache.
            unsafe {
                device.create_pipeline_cache(&wgpu::PipelineCacheDescriptor {
                    label: Some("luma"),
                    data: stored.as_deref(),
                    fallback: true,
                })
            }
        });
        Arc::new(DiskCache {
            dir,
            key,
            pipelines,
            saved: Mutex::new(Saved { hash, pending: false }),
        })
    }

    /// The pipeline cache to hand to pipeline creation.
    pub(crate) fn pipelines(&self) -> Option<&wgpu::PipelineCache> {
        self.pipelines.as_ref()
    }

    /// Notes that a pipeline was just created with [`pipelines`](Self::pipelines). The cache is
    /// written when the context's device is released and at process exit
    /// ([`flush`](Self::flush)), never while pipelines are being built: reading the driver's cache
    /// data and writing megabytes of it slowed the first launches of a run by tens of
    /// milliseconds even from a background thread.
    pub(crate) fn built(&self) {
        if self.pipelines.is_some() {
            lock(&self.saved).pending = true;
        }
    }

    /// Writes the pipeline cache out, on the calling thread, if pipelines were built since it was
    /// last written.
    pub(crate) fn flush(&self) {
        let pending = std::mem::take(&mut lock(&self.saved).pending);
        if pending {
            self.write_out();
        }
    }

    /// Writes the data of the pipeline cache if it changed since the last write. Holds the lock
    /// on [`Saved`] only to compare and record the hash, so that builds never wait for the disk.
    fn write_out(&self) {
        let (Some(dir), Some(pipelines)) = (&self.dir, &self.pipelines) else { return };
        let Some(data) = pipelines.get_data() else { return };
        let path = dir.join(format!("pipelines-{}.bin", self.key));
        if data.len() > MAX_PIPELINES {
            // Too big to load again (see `MAX_PIPELINES`): the next process starts from empty.
            let _ = std::fs::remove_file(&path);
            return;
        }
        let hash = fnv1a(&data);
        if hash == lock(&self.saved).hash {
            return;
        }
        match write_blob(&path, PIPELINES, &data) {
            Ok(()) => lock(&self.saved).hash = hash,
            Err(error) => debug!("luma: writing the pipeline cache failed: {error}"),
        }
    }

    /// The cost model stored for this device and driver, if there is a valid one.
    pub(crate) fn load_cost(&self) -> Option<CostModel> {
        let dir = self.dir.as_ref()?;
        let bytes = read_blob(&dir.join(format!("cost-{}.bin", self.key)), COST)?;
        let numbers: Vec<f64> = bytes.as_chunks::<8>().0.iter().map(|chunk| f64::from_le_bytes(*chunk)).collect();
        let [launch_us, bandwidth_gbs, f32_ops_per_ns, df32_factor, f64_factor] = numbers[..] else { return None };
        let f64_factor = (!f64_factor.is_nan()).then_some(f64_factor);
        let model = CostModel { launch_us, bandwidth_gbs, f32_ops_per_ns, df32_factor, f64_factor };
        let sane = |x: f64| x.is_finite() && x > 0.0;
        (sane(launch_us) && sane(bandwidth_gbs) && sane(f32_ops_per_ns) && sane(df32_factor) && f64_factor.is_none_or(sane)).then_some(model)
    }

    /// Stores the cost model measured for this device and driver.
    pub(crate) fn store_cost(&self, model: &CostModel) {
        let Some(dir) = &self.dir else { return };
        let numbers = [model.launch_us, model.bandwidth_gbs, model.f32_ops_per_ns, model.df32_factor, model.f64_factor.unwrap_or(f64::NAN)];
        let bytes: Vec<u8> = numbers.iter().flat_map(|x| x.to_le_bytes()).collect();
        if let Err(error) = write_blob(&dir.join(format!("cost-{}.bin", self.key)), COST, &bytes) {
            debug!("luma: writing the calibration failed: {error}");
        }
    }

    /// Whether this device and driver were found to have a fused `fma` (see `gpu::fma`), if
    /// that was stored.
    pub(crate) fn load_fma(&self) -> Option<bool> {
        let dir = self.dir.as_ref()?;
        match read_blob(&dir.join(format!("fma-{}.bin", self.key)), FMA)?[..] {
            [0] => Some(false),
            [1] => Some(true),
            _ => None,
        }
    }

    /// Stores whether this device and driver have a fused `fma`.
    pub(crate) fn store_fma(&self, fused: bool) {
        let Some(dir) = &self.dir else { return };
        if let Err(error) = write_blob(&dir.join(format!("fma-{}.bin", self.key)), FMA, &[u8::from(fused)]) {
            debug!("luma: writing the fma test result failed: {error}");
        }
    }
}

/// Writes every cache registered with [`flush_at_exit`] that has unwritten pipelines. Called
/// from the exit handler.
#[cfg(target_os = "linux")]
pub(crate) fn flush_all() {
    // `try_lock`: this runs at exit, when a thread still running may hold the list.
    if let Ok(open) = OPEN.try_lock() {
        for cache in open.iter() {
            cache.flush();
        }
    }
}

/// Has the exit handler write `cache` too. Not for devices of instances with validation layers
/// (see `exit::track`): at exit the layer no longer finds them and aborts on the first call.
#[cfg(target_os = "linux")]
pub(crate) fn flush_at_exit(cache: &Arc<DiskCache>) {
    if cache.pipelines.is_some() {
        lock(&OPEN).push(Arc::clone(cache));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A fresh empty directory under the system's temporary directory.
    fn scratch(name: &str) -> PathBuf {
        static COUNT: AtomicU64 = AtomicU64::new(0);
        let dir = std::env::temp_dir().join(format!("luma-cache-unit-{name}-{}-{}", std::process::id(), COUNT.fetch_add(1, Ordering::Relaxed)));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    fn env<'a>(pairs: &'a [(&'a str, &'a str)]) -> impl Fn(&str) -> Option<OsString> + 'a {
        move |name| pairs.iter().find(|(key, _)| *key == name).map(|(_, value)| OsString::from(value))
    }

    #[test]
    fn the_directory_follows_the_environment() {
        assert_eq!(resolve(env(&[("HOME", "/home/u")])), Some(PathBuf::from("/home/u/.cache/luma")));
        assert_eq!(resolve(env(&[("HOME", "/home/u"), ("XDG_CACHE_HOME", "/x")])), Some(PathBuf::from("/x/luma")));
        assert_eq!(resolve(env(&[("HOME", "/home/u"), ("XDG_CACHE_HOME", "relative")])), Some(PathBuf::from("/home/u/.cache/luma")));
        assert_eq!(resolve(env(&[("HOME", "/h"), ("XDG_CACHE_HOME", "/x"), ("LUMA_CACHE_DIR", "/d")])), Some(PathBuf::from("/d")));
        assert_eq!(resolve(env(&[("HOME", "/h"), ("LUMA_CACHE_DIR", "/d"), ("LUMA_CACHE", "off")])), None);
        assert_eq!(resolve(env(&[("HOME", "/h"), ("LUMA_CACHE", "0")])), None);
        assert_eq!(resolve(env(&[("HOME", "/h"), ("LUMA_CACHE", "FALSE")])), None);
        assert_eq!(resolve(env(&[("HOME", "/h"), ("LUMA_CACHE", "on")])), Some(PathBuf::from("/h/.cache/luma")));
        assert_eq!(resolve(env(&[])), None);
    }

    #[test]
    fn blobs_round_trip_and_damage_is_ignored() {
        let dir = scratch("blob");
        let path = dir.join("a.bin");
        assert_eq!(read_blob(&path, COST), None);
        write_blob(&path, COST, b"hello").unwrap();
        assert_eq!(read_blob(&path, COST).as_deref(), Some(&b"hello"[..]));
        assert_eq!(read_blob(&path, PIPELINES), None, "another magic");
        write_blob(&path, COST, b"second").unwrap();
        assert_eq!(read_blob(&path, COST).as_deref(), Some(&b"second"[..]), "replaced");
        let good = std::fs::read(&path).unwrap();
        for damaged in [good[..good.len() - 1].to_vec(), good[..HEADER - 1].to_vec(), Vec::new(), {
            let mut bytes = good.clone();
            *bytes.last_mut().unwrap() ^= 1;
            bytes
        }] {
            std::fs::write(&path, damaged).unwrap();
            assert_eq!(read_blob(&path, COST), None);
        }
        let leftovers = std::fs::read_dir(&dir).unwrap().filter(|entry| entry.as_ref().unwrap().file_name().to_string_lossy().ends_with(".tmp")).count();
        assert_eq!(leftovers, 0, "no temporary files remain");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn concurrent_writers_leave_a_valid_file() {
        let dir = scratch("race");
        let path = dir.join("race.bin");
        std::thread::scope(|scope| {
            for worker in 0..8u8 {
                let path = &path;
                scope.spawn(move || {
                    for round in 0..50u8 {
                        write_blob(path, PIPELINES, &vec![worker ^ round; 4096 + usize::from(round)]).unwrap();
                        if let Some(payload) = read_blob(path, PIPELINES) {
                            assert!(payload.windows(2).all(|pair| pair[0] == pair[1]), "a torn write was visible");
                        }
                    }
                });
            }
        });
        assert!(read_blob(&path, PIPELINES).is_some());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_cost_model_is_stored_per_device() {
        let ctx = crate::Context::new().unwrap();
        let (device, info) = (ctx.device(), ctx.adapter_info());
        let dir = scratch("cost");
        let disk = DiskCache::open(Some(dir.clone()), device, info);
        assert_eq!(disk.load_cost(), None);
        let model = CostModel { launch_us: 3.5, bandwidth_gbs: 123.0, f32_ops_per_ns: 456.0, df32_factor: 7.0, f64_factor: None };
        disk.store_cost(&model);
        assert_eq!(disk.load_cost(), Some(model));
        let with_f64 = CostModel { f64_factor: Some(9.0), ..model };
        disk.store_cost(&with_f64);
        assert_eq!(DiskCache::open(Some(dir.clone()), device, info).load_cost(), Some(with_f64), "a second open reads it");
        assert_eq!(DiskCache::open(None, device, info).load_cost(), None, "no directory, no cache");
        let file = dir.join(format!("cost-{}.bin", adapter_key(info)));
        std::fs::write(&file, b"garbage that is not a cost file at all").unwrap();
        assert_eq!(disk.load_cost(), None, "a corrupt file is ignored");
        write_blob(&file, COST, &[0u8; 40]).unwrap();
        assert_eq!(disk.load_cost(), None, "zeros are not a cost model");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_calibrating_context_leaves_a_clean_result_in_the_cache_or_uses_the_profile() {
        let ctx = crate::Context::with_options(crate::ContextOptions::from_env().calibrate(true)).unwrap();
        let disk = ctx.disk_cache();
        if disk.dir.is_some() {
            let used = *ctx.device_cost().get().unwrap();
            match disk.load_cost() {
                Some(stored) => assert!(!crate::jit::cost::contended(&stored, &CostModel::fallback(ctx.tuning().profile)), "only clean results are stored"),
                None => assert_eq!(used, CostModel::fallback(ctx.tuning().profile), "a contended measurement is neither used nor stored"),
            }
        }
    }

    #[test]
    fn pipelines_are_stored_and_survive_corruption() {
        let ctx = crate::Context::new().unwrap();
        let (device, info) = (ctx.device(), ctx.adapter_info());
        let dir = scratch("pipelines");
        let disk = DiskCache::open(Some(dir.clone()), device, info);
        if disk.pipelines().is_none() {
            // No PIPELINE_CACHE on this adapter: nothing to store, and nothing may be written.
            disk.built();
            disk.flush();
            assert!(!dir.exists());
            return;
        }
        let file = dir.join(format!("pipelines-{}.bin", adapter_key(info)));
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl("@group(0) @binding(0) var<storage, read_write> a: array<f32>; @compute @workgroup_size(1) fn main(@builtin(global_invocation_id) g: vec3<u32>) { a[g.x] = a[g.x] * 2.0; }".into()),
        });
        let _pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: disk.pipelines(),
        });
        disk.built();
        disk.flush();
        let stored = read_blob(&file, PIPELINES).expect("the pipeline cache was written");
        assert!(!stored.is_empty());
        assert!(DiskCache::open(Some(dir.clone()), device, info).pipelines().is_some(), "a stored cache loads");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(&file, b"LUMAPIPE damaged").unwrap();
        let reopened = DiskCache::open(Some(dir.clone()), device, info);
        assert!(reopened.pipelines().is_some(), "a corrupt file falls back to an empty cache");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
