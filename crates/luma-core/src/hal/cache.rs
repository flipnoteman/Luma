//! The on-disk cache: compiled kernels and measured device costs, kept between processes.
//!
//! One directory ([`directory`]: `LUMA_CACHE_DIR`, else `$XDG_CACHE_HOME/luma`, else
//! `~/.cache/luma`; `LUMA_CACHE=off` disables it) holds, per device and driver key, opaque blobs
//! (wgpu pipeline caches, CUDA PTX), the calibrated [`CostModel`] and small notes. Every file is
//! written to a temporary name and renamed over the old one, so concurrent processes never see a
//! half-written file (the last writer wins); every file carries a checksum, and one that does not
//! match is ignored as if it were absent. Nothing here ever fails an operation: unreadable or
//! unwritable cache files only cost the speed-up.

use std::ffi::OsString;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Duration;

use log::{debug, info};

use super::CostModel;

/// Layout version of the files; bumping it makes older files unreadable (and so replaced).
const FORMAT: u32 = 1;
/// Magic of the pipeline cache file.
pub const PIPELINES_MAGIC: [u8; 8] = *b"LUMAPIPE";
/// Magic of compiled PTX blobs.
pub const PTX_MAGIC: [u8; 8] = *b"LUMAPTX1";
/// Magic of compiled CPU machine-code blobs.
pub const CPU_MAGIC: [u8; 8] = *b"LUMACPU1";
/// Magic of the calibration file (2: measured with the faster `Df32` library, so older files
/// are not read).
const COST: [u8; 8] = *b"LUMACOS2";
/// Magic of the file with the result of the `fma` test.
const FMA: [u8; 8] = *b"LUMAFMA ";
/// Magic of the file noting that a calibration looked contended.
const CONTENDED: [u8; 8] = *b"LUMACONT";
/// How long a contended calibration stops other processes from measuring again: a machine whose
/// measurements always look contended would otherwise measure twice on every device it opens.
const CONTENDED_RETRY: Duration = Duration::from_secs(24 * 60 * 60);
/// The largest pipeline cache that is kept, in bytes. A larger one is dropped (not loaded, not
/// written) and starts again from empty: NVIDIA's driver crashed at exit about one run in five
/// with a 116 MB cache still being worked through on its compiler threads.
pub const MAX_PIPELINES: usize = 32 << 20;
/// Magic, format, payload length, payload checksum.
const HEADER: usize = 8 + 4 + 8 + 8;

/// FNV-1a: a stable 64-bit hash, for checksums and file names.
pub fn fnv1a(bytes: &[u8]) -> u64 {
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

/// The directory tests use instead of the user's cache.
pub fn test_directory() -> PathBuf {
    std::env::temp_dir().join("luma-test-cache")
}

/// The cache directory of this process, from the environment.
///
/// Unit tests, and processes with `LUMA_TEST_CACHE=1`, use [`test_directory`] unless
/// `LUMA_CACHE_DIR` or `LUMA_CACHE` is set, so that they neither read nor fill the user's cache.
pub fn directory() -> Option<PathBuf> {
    let unset = std::env::var_os("LUMA_CACHE_DIR").is_none() && std::env::var_os("LUMA_CACHE").is_none();
    if unset && (cfg!(test) || std::env::var_os("LUMA_TEST_CACHE").is_some_and(|v| v == "1")) {
        return Some(test_directory());
    }
    resolve(|name| std::env::var_os(name))
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

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|e| e.into_inner())
}

/// A device's view of the cache directory.
pub struct DiskCache {
    /// The directory, `None` when caching is off.
    dir: Option<PathBuf>,
    /// Names the device and driver the cached data belong to.
    key: String,
    /// Hash of the pipeline blob last loaded or written, 0 if none.
    pipelines_hash: Mutex<u64>,
}

impl std::fmt::Debug for DiskCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DiskCache").field("dir", &self.dir).field("key", &self.key).finish()
    }
}

impl DiskCache {
    /// Opens the cache of the device named by `key` in `dir` (`None`: caching is off).
    pub fn open(dir: Option<PathBuf>, key: String) -> Arc<DiskCache> {
        Arc::new(DiskCache { dir, key, pipelines_hash: Mutex::new(0) })
    }

    /// The directory, `None` when caching is off.
    pub fn dir(&self) -> Option<&Path> {
        self.dir.as_deref()
    }

    /// The device key the files are named by.
    pub fn key(&self) -> &str {
        &self.key
    }

    /// The payload stored as `{stem}.bin` if valid and of this `magic`.
    pub fn load_blob(&self, stem: &str, magic: [u8; 8]) -> Option<Vec<u8>> {
        read_blob(&self.dir.as_ref()?.join(format!("{stem}.bin")), magic)
    }

    /// Atomically stores `payload` as `{stem}.bin`; whether it was written.
    pub fn store_blob(&self, stem: &str, magic: [u8; 8], payload: &[u8]) -> bool {
        let Some(dir) = &self.dir else { return false };
        match write_blob(&dir.join(format!("{stem}.bin")), magic, payload) {
            Ok(()) => true,
            Err(error) => {
                debug!("luma: writing cache file {stem} failed: {error}");
                false
            }
        }
    }

    /// Deletes `{stem}.bin`.
    pub fn remove_blob(&self, stem: &str) {
        if let Some(dir) = &self.dir {
            let _ = std::fs::remove_file(dir.join(format!("{stem}.bin")));
        }
    }

    fn pipelines_stem(&self) -> String {
        format!("pipelines-{}", self.key)
    }

    /// The stored driver pipeline cache, if valid and not over [`MAX_PIPELINES`].
    pub fn load_pipelines(&self) -> Option<Vec<u8>> {
        let blob = self.load_blob(&self.pipelines_stem(), PIPELINES_MAGIC).filter(|b| b.len() <= MAX_PIPELINES)?;
        *lock(&self.pipelines_hash) = fnv1a(&blob);
        Some(blob)
    }

    /// Stores the driver pipeline cache if it changed since it was loaded or last stored; one
    /// over [`MAX_PIPELINES`] is dropped instead.
    pub fn store_pipelines(&self, data: &[u8]) {
        if self.dir.is_none() {
            return;
        }
        let stem = self.pipelines_stem();
        if data.len() > MAX_PIPELINES {
            info!(
                "luma: the pipeline cache of {} MB is over the {} MB kept, and is dropped; the next process compiles its kernels again",
                data.len() >> 20,
                MAX_PIPELINES >> 20
            );
            self.remove_blob(&stem);
            return;
        }
        let hash = fnv1a(data);
        if hash == *lock(&self.pipelines_hash) {
            return;
        }
        if self.store_blob(&stem, PIPELINES_MAGIC, data) {
            *lock(&self.pipelines_hash) = hash;
        }
    }

    /// The cost model stored for this device and driver, if there is a valid one.
    pub fn load_cost(&self) -> Option<CostModel> {
        let bytes = self.load_blob(&format!("cost-{}", self.key), COST)?;
        let numbers: Vec<f64> = bytes.as_chunks::<8>().0.iter().map(|chunk| f64::from_le_bytes(*chunk)).collect();
        let [launch_us, bandwidth_gbs, f32_ops_per_ns, df32_factor, f64_factor] = numbers[..] else { return None };
        let f64_factor = (!f64_factor.is_nan()).then_some(f64_factor);
        let model = CostModel { launch_us, bandwidth_gbs, f32_ops_per_ns, df32_factor, f64_factor };
        let sane = |x: f64| x.is_finite() && x > 0.0;
        (sane(launch_us) && sane(bandwidth_gbs) && sane(f32_ops_per_ns) && sane(df32_factor) && f64_factor.is_none_or(sane)).then_some(model)
    }

    /// Stores the cost model measured for this device and driver.
    pub fn store_cost(&self, model: &CostModel) {
        let numbers = [model.launch_us, model.bandwidth_gbs, model.f32_ops_per_ns, model.df32_factor, model.f64_factor.unwrap_or(f64::NAN)];
        let bytes: Vec<u8> = numbers.iter().flat_map(|x| x.to_le_bytes()).collect();
        self.store_blob(&format!("cost-{}", self.key), COST, &bytes);
    }

    /// Whether a calibration of this device and driver looked contended less than a day ago, so
    /// that it is not measured again yet.
    pub fn recently_contended(&self) -> bool {
        let Some(dir) = &self.dir else { return false };
        let path = dir.join(format!("contended-{}.bin", self.key));
        read_blob(&path, CONTENDED).is_some()
            && std::fs::metadata(&path).and_then(|meta| meta.modified()).ok().and_then(|at| at.elapsed().ok()).is_some_and(|age| age < CONTENDED_RETRY)
    }

    /// Notes that a calibration of this device and driver looked contended.
    pub fn note_contended(&self) {
        self.store_blob(&format!("contended-{}", self.key), CONTENDED, &[]);
    }

    /// Whether this device and driver were found to have a fused `fma`, if that was stored.
    pub fn load_fma(&self) -> Option<bool> {
        match self.load_blob(&format!("fma-{}", self.key), FMA)?[..] {
            [0] => Some(false),
            [1] => Some(true),
            _ => None,
        }
    }

    /// Stores whether this device and driver have a fused `fma`.
    pub fn store_fma(&self, fused: bool) {
        self.store_blob(&format!("fma-{}", self.key), FMA, &[u8::from(fused)]);
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
        assert_eq!(read_blob(&path, PIPELINES_MAGIC), None, "another magic");
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
                        write_blob(path, PIPELINES_MAGIC, &vec![worker ^ round; 4096 + usize::from(round)]).unwrap();
                        if let Some(payload) = read_blob(path, PIPELINES_MAGIC) {
                            assert!(payload.windows(2).all(|pair| pair[0] == pair[1]), "a torn write was visible");
                        }
                    }
                });
            }
        });
        assert!(read_blob(&path, PIPELINES_MAGIC).is_some());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_contended_calibration_is_not_measured_again_for_a_day() {
        let dir = scratch("contended");
        let disk = DiskCache::open(Some(dir.clone()), "test-key".into());
        assert!(!disk.recently_contended());
        disk.note_contended();
        assert!(disk.recently_contended());
        assert!(DiskCache::open(Some(dir.clone()), "test-key".into()).recently_contended(), "another process sees it");
        let file = dir.join("contended-test-key.bin");
        let old = std::time::SystemTime::now() - CONTENDED_RETRY - Duration::from_secs(60);
        std::fs::File::options().write(true).open(&file).unwrap().set_modified(old).unwrap();
        assert!(!disk.recently_contended(), "a day later it is measured again");
        assert!(!DiskCache::open(None, "test-key".into()).recently_contended(), "no directory, no note");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_cost_model_is_stored_per_device() {
        let dir = scratch("cost");
        let disk = DiskCache::open(Some(dir.clone()), "test-key".into());
        assert_eq!(disk.load_cost(), None);
        let model = CostModel { launch_us: 3.5, bandwidth_gbs: 123.0, f32_ops_per_ns: 456.0, df32_factor: 7.0, f64_factor: None };
        disk.store_cost(&model);
        assert_eq!(disk.load_cost(), Some(model));
        let with_f64 = CostModel { f64_factor: Some(9.0), ..model };
        disk.store_cost(&with_f64);
        assert_eq!(DiskCache::open(Some(dir.clone()), "test-key".into()).load_cost(), Some(with_f64), "a second open reads it");
        assert_eq!(DiskCache::open(None, "test-key".into()).load_cost(), None, "no directory, no cache");
        assert_eq!(DiskCache::open(Some(dir.clone()), "other".into()).load_cost(), None, "another device");
        let file = dir.join("cost-test-key.bin");
        std::fs::write(&file, b"garbage that is not a cost file at all").unwrap();
        assert_eq!(disk.load_cost(), None, "a corrupt file is ignored");
        write_blob(&file, COST, &[0u8; 40]).unwrap();
        assert_eq!(disk.load_cost(), None, "zeros are not a cost model");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn pipelines_are_stored_and_oversize_is_dropped() {
        let dir = scratch("pipelines");
        let disk = DiskCache::open(Some(dir.clone()), "test-key".into());
        assert_eq!(disk.load_pipelines(), None);
        disk.store_pipelines(b"driver data");
        assert_eq!(DiskCache::open(Some(dir.clone()), "test-key".into()).load_pipelines().as_deref(), Some(&b"driver data"[..]));
        std::fs::write(dir.join("pipelines-test-key.bin"), b"LUMAPIPE damaged").unwrap();
        assert_eq!(disk.load_pipelines(), None, "a corrupt file is ignored");
        disk.store_pipelines(&vec![0u8; MAX_PIPELINES + 1]);
        assert!(!dir.join("pipelines-test-key.bin").exists(), "oversize cache removed");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn ptx_blobs_and_fma_round_trip() {
        let dir = scratch("ptx");
        let disk = DiskCache::open(Some(dir.clone()), "k".into());
        assert_eq!(disk.load_blob("ptx-0000000000000001", PTX_MAGIC), None);
        assert!(disk.store_blob("ptx-0000000000000001", PTX_MAGIC, b".version 8.0"));
        assert_eq!(disk.load_blob("ptx-0000000000000001", PTX_MAGIC).as_deref(), Some(&b".version 8.0"[..]));
        assert_eq!(disk.load_blob("ptx-0000000000000001", PIPELINES_MAGIC), None, "another magic");
        disk.remove_blob("ptx-0000000000000001");
        assert_eq!(disk.load_blob("ptx-0000000000000001", PTX_MAGIC), None);
        assert_eq!(disk.load_fma(), None);
        disk.store_fma(true);
        assert_eq!(disk.load_fma(), Some(true));
        disk.store_fma(false);
        assert_eq!(disk.load_fma(), Some(false));
        assert!(!DiskCache::open(None, "k".into()).store_blob("x", PTX_MAGIC, b"y"));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
