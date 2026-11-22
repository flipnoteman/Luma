//! The process-wide pool of opened wgpu devices, and the driver's pipeline cache of each.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, OnceLock, PoisonError};
use std::time::{Duration, Instant};

use log::debug;
use luma_core::hal::cache;
use luma_core::{BackendError, CostModel, DeviceKind, DiskCache};
use wgpu::{Device, InstanceDescriptor, MemoryHints, PowerPreference, Queue};

/// Locks `mutex`, ignoring poisoning: every mutex of the crate guards data that stays consistent
/// if a holder panicked.
pub(crate) fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

/// The driver's compiled pipelines of a device, kept between processes through its
/// [`DiskCache`]. Without a cache directory, or on a device without
/// [`wgpu::Features::PIPELINE_CACHE`], there is none: pipelines compile as usual.
///
/// The stored cache is up to [`cache::MAX_PIPELINES`] of data to read, check and hand to the
/// driver, tens of milliseconds that opening a device must not wait for: a thread loads it while
/// the rest of the open goes on, and the first pipeline built (or the first write) waits for it.
#[derive(Debug)]
pub(crate) struct PipelineStore {
    /// The loading thread, until its result is taken.
    loading: Mutex<Option<std::thread::JoinHandle<Option<wgpu::PipelineCache>>>>,
    /// The loaded cache, set once by the first [`cache`](Self::cache).
    cache: OnceLock<Option<wgpu::PipelineCache>>,
    disk: Arc<DiskCache>,
    /// Whether pipelines were built since the cache was last written.
    pending: AtomicBool,
    /// When the cache was opened or last written, for [`checkpoint`](Self::checkpoint).
    last_write: Mutex<Instant>,
}

/// How long a [`PipelineStore`] goes without a [`checkpoint`](PipelineStore::checkpoint) write
/// after it was opened or written: reading the driver's cache back costs a few milliseconds per
/// megabyte, which a program that makes and drops contexts in a loop must not pay each time.
const CHECKPOINT_EVERY: Duration = Duration::from_secs(5);

impl PipelineStore {
    /// Opens the store of `device`, starting to load the stored cache if there is one.
    fn open(disk: Arc<DiskCache>, device: &Device) -> Arc<PipelineStore> {
        let wanted = disk.dir().is_some() && device.features().contains(wgpu::Features::PIPELINE_CACHE);
        let loading = wanted
            .then(|| {
                let (disk, device) = (Arc::clone(&disk), device.clone());
                std::thread::Builder::new().name("luma-pipeline-cache".into()).spawn(move || Self::load(&disk, &device)).ok()
            })
            .flatten();
        let cache = OnceLock::new();
        if !wanted {
            let _ = cache.set(None);
        }
        Arc::new(PipelineStore { loading: Mutex::new(loading), cache, disk, pending: AtomicBool::new(false), last_write: Mutex::new(Instant::now()) })
    }

    /// Reads the stored cache and creates the driver's cache from it (an empty one if there is
    /// none, or it is unusable).
    fn load(disk: &DiskCache, device: &Device) -> Option<wgpu::PipelineCache> {
        let stored = disk.load_pipelines();
        // SAFETY: the data is empty or was returned by `PipelineCache::get_data` (the
        // checksum of the disk cache guards against a damaged file), and wgpu and the driver
        // check that it belongs to this adapter and driver; with `fallback` an unusable blob
        // only yields an empty cache.
        Some(crate::exit::driver_call(|| unsafe {
            device.create_pipeline_cache(&wgpu::PipelineCacheDescriptor {
                label: Some("luma"),
                data: stored.as_ref().map(cache::PipelineBlob::data),
                fallback: true,
            })
        }))
    }

    /// The pipeline cache to hand to pipeline creation; waits until it is loaded.
    pub(crate) fn cache(&self) -> Option<&wgpu::PipelineCache> {
        self.cache.get_or_init(|| lock(&self.loading).take().and_then(|thread| thread.join().ok()).flatten()).as_ref()
    }

    /// Whether there is a cache that [`flush`](Self::flush) writes.
    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    pub(crate) fn is_active(&self) -> bool {
        self.cache.get().is_none_or(Option::is_some)
    }

    /// Notes that a pipeline was just created with [`cache`](Self::cache). The cache is written
    /// when a device is released (at most every [`CHECKPOINT_EVERY`]) and at process exit, never
    /// while pipelines are being built: reading the driver's cache data and writing megabytes of
    /// it slowed the first launches of a run by tens of milliseconds even from a background
    /// thread.
    pub(crate) fn built(&self) {
        self.pending.store(true, Ordering::Release);
    }

    /// Writes the pipeline cache out, on the calling thread, if pipelines were built since it was
    /// last written. Also waits for a load still running, so that no thread of ours is inside the
    /// driver when the process exits.
    pub(crate) fn flush(&self) {
        let cache = self.cache();
        if self.pending.swap(false, Ordering::AcqRel)
            && let Some(data) = cache.and_then(wgpu::PipelineCache::get_data)
        {
            self.disk.store_pipelines(&data);
            *lock(&self.last_write) = Instant::now();
        }
    }

    /// [`flush`](Self::flush) for a device going idle, unless the cache was opened or written less
    /// than [`CHECKPOINT_EVERY`] ago. A context made and dropped in a loop would otherwise copy
    /// the whole driver cache out each time to find that it did not change; what a skipped
    /// checkpoint leaves unwritten is written by the next one or at exit.
    pub(crate) fn checkpoint(&self) {
        if lock(&self.last_write).elapsed() >= CHECKPOINT_EVERY {
            self.flush();
        }
    }
}

/// An opened device with its queue and what the adapter is. Clones are handles to the same
/// device.
#[derive(Clone, Debug)]
pub(crate) struct Gpu {
    pub(crate) device: Device,
    pub(crate) queue: Queue,
    pub(crate) adapter: wgpu::AdapterInfo,
    /// Set by the device-lost callback; a lost device is never pooled again.
    lost: Arc<AtomicBool>,
    /// Total bytes of the adapter's device-local memory heaps, if the backend can tell (Vulkan
    /// only): the basis of the default memory budget.
    pub(crate) memory: Option<u64>,
    /// Streaming multiprocessors or compute units, if Vulkan can tell (`VK_NV_shader_sm_builtins`,
    /// `VK_AMD_shader_core_properties`).
    pub(crate) compute_units: Option<u32>,
    /// The device's measured cost model, set by the first context that calibrates it, so that a
    /// pooled device is calibrated once per process.
    pub(crate) cost: Arc<OnceLock<CostModel>>,
    /// The on-disk cache of calibration and notes.
    pub(crate) disk: Arc<DiskCache>,
    /// The driver's pipeline cache.
    pub(crate) store: Arc<PipelineStore>,
}

/// Names the adapter and driver the cached data belong to: backend, vendor and device ids, and a
/// hash of the adapter's name, driver and driver version, so that a driver update starts a new
/// file instead of rewriting the old one back and forth.
fn adapter_key(info: &wgpu::AdapterInfo) -> String {
    let identity = format!("{}\n{}\n{}", info.name, info.driver, info.driver_info);
    format!("{:?}-{:04x}-{:04x}-{:016x}", info.backend, info.vendor, info.device, cache::fnv1a(identity.as_bytes())).to_ascii_lowercase()
}

/// Total bytes of the device-local heaps of `adapter`: how much array data a device can hold at
/// most. wgpu has no portable call for it, so this asks Vulkan directly.
#[cfg(any(target_os = "linux", target_os = "android", windows))]
fn device_local_bytes(adapter: &wgpu::Adapter) -> Option<u64> {
    // SAFETY: only reads properties of the physical device; nothing is destroyed.
    let hal = unsafe { adapter.as_hal::<wgpu::hal::api::Vulkan>() }?;
    let props = unsafe { hal.shared_instance().raw_instance().get_physical_device_memory_properties(hal.raw_physical_device()) };
    // VK_MEMORY_HEAP_DEVICE_LOCAL_BIT
    const DEVICE_LOCAL: u32 = 1;
    Some(props.memory_heaps.iter().take(props.memory_heap_count as usize).filter(|heap| heap.flags.as_raw() & DEVICE_LOCAL != 0).map(|heap| heap.size).sum())
}

/// Multiprocessors of `adapter`: asks Vulkan through the vendor extensions that report them
/// (NVIDIA SMs, AMD compute units); `None` for other vendors.
#[cfg(any(target_os = "linux", target_os = "android", windows))]
fn compute_unit_count(adapter: &wgpu::Adapter) -> Option<u32> {
    use ash::vk;
    // SAFETY: only reads properties of the physical device; nothing is destroyed.
    let hal = unsafe { adapter.as_hal::<wgpu::hal::api::Vulkan>() }?;
    let instance = hal.shared_instance().raw_instance();
    let physical = hal.raw_physical_device();
    let extensions = unsafe { instance.enumerate_device_extension_properties(physical) }.ok()?;
    let has = |name: &std::ffi::CStr| extensions.iter().any(|e| e.extension_name_as_c_str().is_ok_and(|n| n == name));
    if has(vk::NV_SHADER_SM_BUILTINS_NAME) {
        let mut sm = vk::PhysicalDeviceShaderSMBuiltinsPropertiesNV::default();
        let mut props = vk::PhysicalDeviceProperties2::default().push_next(&mut sm);
        unsafe { instance.get_physical_device_properties2(physical, &mut props) };
        return Some(sm.shader_sm_count).filter(|&n| n > 0);
    }
    if has(vk::AMD_SHADER_CORE_PROPERTIES_NAME) {
        let mut core = vk::PhysicalDeviceShaderCorePropertiesAMD::default();
        let mut props = vk::PhysicalDeviceProperties2::default().push_next(&mut core);
        unsafe { instance.get_physical_device_properties2(physical, &mut props) };
        let units = core.shader_engine_count * core.shader_arrays_per_engine_count * core.compute_units_per_shader_array;
        return Some(units).filter(|&n| n > 0);
    }
    None
}

#[cfg(not(any(target_os = "linux", target_os = "android", windows)))]
fn compute_unit_count(_adapter: &wgpu::Adapter) -> Option<u32> {
    None
}

/// Other platforms cannot read the heap sizes: no automatic memory budget.
#[cfg(not(any(target_os = "linux", target_os = "android", windows)))]
fn device_local_bytes(_adapter: &wgpu::Adapter) -> Option<u64> {
    None
}

/// The process's wgpu instance and the adapter picked on it, once there is one.
static ADAPTER: Mutex<Option<(wgpu::Instance, wgpu::Adapter)>> = Mutex::new(None);

/// Drops the cached instance and adapter, so that the next device opens on new ones. Done after a
/// device was lost: the loss may have left the instance and its adapter unusable, and a later
/// context must not inherit that. Devices already open keep what they need alive on their own.
fn forget_adapter() {
    let old = lock(&ADAPTER).take();
    drop(old);
}

/// Picks an adapter of the wanted kind and opens a [Device] / [Queue] pair on it; with `fresh`, on
/// an instance and adapter opened anew. Only [`acquire`] calls this.
async fn request_device(fresh: bool) -> Result<Gpu, BackendError> {
    let mut instance_desc = InstanceDescriptor::new_without_display_handle();
    // Buffer creation reports out-of-memory once usage would pass 90% of the driver's live
    // budget (Vulkan with VK_EXT_memory_budget, DX12) instead of failing later in a command.
    instance_desc.memory_budget_thresholds.for_resource_creation = Some(90);
    instance_desc.backends = wgpu::Backends::PRIMARY;
    // Indirect dispatches only ever read workgroup counts that a gate kernel of this crate
    // copied from a template of valid counts or zeroed, so wgpu's check of them, an extra
    // validation dispatch in front of every indirect one, protects nothing.
    instance_desc.flags.remove(wgpu::InstanceFlags::VALIDATION_INDIRECT_CALL);
    // No `VK_EXT_debug_utils` object names (on by default in debug builds). The Vulkan loader
    // (1.4.341) finds the device of `vkSetDebugUtilsObjectNameEXT` by walking the device lists of
    // all instances, while `vkCreateDevice` links a new device into such a list without the
    // loader's lock: naming an object while another thread opens a device segfaults the process.
    // Validation layers still report errors, through their own default output.
    instance_desc.flags.remove(wgpu::InstanceFlags::DEBUG);
    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    let validated = instance_desc.flags.contains(wgpu::InstanceFlags::VALIDATION);
    // The instance and the adapter are opened once per process: the Vulkan loader and the driver
    // take about 70 ms for them, and every further device of the process (contexts alive at the
    // same time each have one) reuses them. They are dropped after a device was lost (see
    // [`forget_adapter`]) and when `fresh` says so, so that nothing a loss left in them reaches a
    // later device. Only `acquire` calls this, under `OPENING`.
    if fresh {
        forget_adapter();
    }
    let cached = lock(&ADAPTER).as_ref().map(|(_, adapter)| adapter.clone());
    let adapter = match cached {
        Some(adapter) => adapter,
        None => {
            let instance = wgpu::Instance::new(instance_desc);
            let adapter = instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: PowerPreference::HighPerformance,
                    ..Default::default()
                })
                .await
                .map_err(|e| BackendError::NoAdapter(e.to_string()))?;
            *lock(&ADAPTER) = Some((instance, adapter.clone()));
            adapter
        }
    };
    // Software rasterizers (llvmpipe, SwiftShader, WARP) are never used: the cpu backend runs
    // on the CPU natively and faster.
    if adapter.get_info().device_type == wgpu::DeviceType::Cpu {
        return Err(BackendError::NoAdapter(format!("only a software adapter ({}) is available; luma does not use software adapters", adapter.get_info().name)));
    }
    debug!("Adapter(s) = {:?}", adapter.get_info());

    // Request the adapter's full limits so large arrays aren't capped by the
    // conservative WebGPU defaults (e.g. 128 MiB storage bindings).
    // Timestamp queries are optional: they only serve profiling.
    let mut required_features = adapter.features() & wgpu::Features::TIMESTAMP_QUERY;
    // Shader features custom kernels may use; requested whenever the adapter has them.
    required_features |= adapter.features()
        & (wgpu::Features::SUBGROUP | wgpu::Features::SHADER_F16 | wgpu::Features::SHADER_F64 | wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX);
    // Hardware `f32` atomic adds for the scatter and group kernels (a compare-and-swap loop on
    // the bits is the fallback, and collapses when many invocations add to few cells).
    required_features |= adapter.features() & wgpu::Features::SHADER_FLOAT32_ATOMIC;
    // Only on unified memory: on a discrete GPU mappable storage buffers live in slow host memory.
    if adapter.get_info().device_type == wgpu::DeviceType::IntegratedGpu {
        required_features |= adapter.features() & wgpu::Features::MAPPABLE_PRIMARY_BUFFERS;
    }
    // The driver's compiled pipelines are kept between processes.
    let cache_dir = cache::directory();
    if cache_dir.is_some() {
        required_features |= adapter.features() & wgpu::Features::PIPELINE_CACHE;
    }

    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("Device 1"),
            required_features,
            required_limits: adapter.limits(),
            memory_hints: MemoryHints::MemoryUsage,
            // SAFETY: the only experimental feature requested is cooperative matrix, which only
            // kernels that opt in with `enable wgpu_cooperative_matrix;` use.
            experimental_features: unsafe { wgpu::ExperimentalFeatures::enabled() },
            ..Default::default()
        })
        .await
        .map_err(|e| BackendError::RequestDevice(e.to_string()))?;
    let lost = Arc::new(AtomicBool::new(false));
    let flag = Arc::clone(&lost);
    device.set_device_lost_callback(move |reason, message| {
        debug!("device lost ({reason:?}): {message}");
        flag.store(true, Ordering::Release);
    });
    let memory = device_local_bytes(&adapter);
    let compute_units = compute_unit_count(&adapter);
    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    crate::exit::track(&device, validated);
    let info = adapter.get_info();
    let disk = DiskCache::open(cache_dir, adapter_key(&info));
    let store = PipelineStore::open(Arc::clone(&disk), &device);
    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    if validated {
        // Nothing waits for the load at exit for these devices: finish it here.
        store.cache();
    } else {
        crate::exit::flush_at_exit(&store);
    }
    Ok(Gpu { device, queue, adapter: info, lost, memory, compute_units, cost: Arc::new(OnceLock::new()), disk, store })
}

/// The process-wide pool of opened GPU devices, and who holds the others.
///
/// Destroying a Vulkan device while other threads are inside the driver deadlocks the NVIDIA
/// driver (615.71): the thread in `vkDestroyDevice` holds the driver's global mutex and the
/// loader lock and joins a driver worker thread that never exits. Devices are therefore never
/// destroyed: [`release`] parks the device of a finished backend in the idle list and
/// [`acquire`] hands it to the next one. The pool never shrinks; it holds as many devices
/// as backends were alive at once.
struct Pool {
    /// Devices no backend uses; none of them is lost.
    idle: Vec<Gpu>,
    /// Devices handed out by [`acquire`] that [`release`] has not got back yet.
    leased: usize,
    /// How many times [`release`] ran: a waiting [`acquire`] sees that a device came back.
    returns: u64,
}

static POOL: Mutex<Pool> = Mutex::new(Pool { idle: Vec::new(), leased: 0, returns: 0 });

/// Signalled by [`release`], for an [`acquire`] that waits for a device to come back.
static RETURNED: Condvar = Condvar::new();

/// Serializes [`acquire`]. A burst of threads opening backends must not each create a device
/// while the devices of finished ones are about to come back, and the driver caps how many
/// devices can be alive at once: 63 or 64 per process on NVIDIA's Linux driver (not per instance:
/// a second instance gets none once the first has used them up), after which `vkCreateDevice` fails
/// with `VK_ERROR_INITIALIZATION_FAILED`, which wgpu reports as
/// `RequestDevice("Parent device is lost")`, though no device is lost.
static OPENING: Mutex<()> = Mutex::new(());

/// How long [`acquire`] waits for a device to come back when none can be created.
const DEVICE_WAIT: Duration = Duration::from_secs(60);

/// Hands out an idle pooled device of the wanted kind, or opens a new one when none is idle.
///
/// Holds `OPENING` for the whole call so that concurrent callers queue up and reuse the devices
/// that [`release`] returns in the meantime instead of all creating their own.
///
/// A device that cannot be created while others are out is not an error yet: the driver may be
/// at its limit of devices, so the call waits (up to [`DEVICE_WAIT`]) for a backend to give its
/// device back and takes that one. With no device out, the failure is the GPU's: a lost device
/// may have left the instance and adapter unusable, so the device is created once more on new ones.
///
/// Devices found lost are discarded, never handed out, and make the next device open on a new
/// instance and adapter (see [`forget_adapter`]).
pub(crate) fn acquire(kind: DeviceKind) -> Result<Gpu, BackendError> {
    if kind == DeviceKind::Cpu {
        return Err(BackendError::NoDevice("software adapters are not used; the cpu backend runs on the CPU"));
    }
    let give_up = Instant::now() + DEVICE_WAIT;
    let _opening = lock(&OPENING);
    let mut fresh = false;
    loop {
        let (idle, returns) = take_idle();
        if let Some(gpu) = idle {
            return Ok(gpu);
        }
        let error = match crate::exit::driver_call(|| pollster::block_on(request_device(fresh))) {
            Ok(gpu) => {
                lock(&POOL).leased += 1;
                return Ok(gpu);
            }
            Err(error) => error,
        };
        if lock(&POOL).leased > 0 {
            if !wait_for_return(returns, give_up) {
                return Err(error);
            }
        } else if fresh || !matches!(error, BackendError::RequestDevice(_)) {
            return Err(error);
        } else {
            fresh = true;
        }
    }
}

/// Takes an idle device that is not lost, if there is one, and counts it as leased; also says how
/// many returns [`release`] has made so far. Devices found lost are discarded, and make the next
/// device open on a new instance and adapter.
fn take_idle() -> (Option<Gpu>, u64) {
    let mut lost = Vec::new();
    let (taken, returns) = {
        let mut pool = lock(&POOL);
        let taken = loop {
            match pool.idle.pop() {
                Some(gpu) if gpu.lost.load(Ordering::Acquire) => lost.push(gpu),
                found => break found,
            }
        };
        if taken.is_some() {
            pool.leased += 1;
        }
        (taken, pool.returns)
    };
    if !lost.is_empty() {
        forget_adapter();
    }
    drop(lost);
    (taken, returns)
}

/// Waits until [`release`] has run since it had made `seen` returns. `false` once `give_up` has
/// passed before that.
fn wait_for_return(seen: u64, give_up: Instant) -> bool {
    let mut pool = lock(&POOL);
    while pool.returns == seen {
        let Some(left) = give_up.checked_duration_since(Instant::now()) else { return false };
        pool = RETURNED.wait_timeout(pool, left).unwrap_or_else(PoisonError::into_inner).0;
    }
    true
}

/// Returns the device of a finished backend to the pool.
///
/// Waits for the device to go idle first, so that the backend's submitted work, its map
/// callbacks and the deferred destruction of its resources are finished before the next owner
/// gets the device (destroying the device used to wait for exactly that). A lost device is
/// dropped instead of pooled, as it must not be handed out again, and the instance and adapter
/// it came from are dropped too (see [`forget_adapter`]). Does not take `OPENING`: a thread
/// returning a device must never wait for one being created.
pub(crate) fn release(gpu: Gpu) {
    let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
    gpu.store.checkpoint();
    let lost = gpu.lost.load(Ordering::Acquire);
    if lost {
        forget_adapter();
    }
    {
        let mut pool = lock(&POOL);
        pool.leased -= 1;
        pool.returns += 1;
        if !lost {
            pool.idle.push(gpu);
        }
    }
    RETURNED.notify_all();
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::AtomicUsize;

    use super::*;

    /// A device from the pool, or `None` where the machine has no adapter.
    fn open() -> Option<Gpu> {
        match acquire(DeviceKind::Auto) {
            Ok(gpu) => Some(gpu),
            Err(BackendError::NoAdapter(_) | BackendError::NoDevice(_)) => None,
            Err(other) => panic!("{other}"),
        }
    }

    /// Whether `gpu` runs work: a buffer written and read back.
    fn works(gpu: &Gpu) -> bool {
        let buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 16,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        gpu.queue.write_buffer(&buffer, 0, &(1..=16u8).collect::<Vec<_>>());
        gpu.queue.submit(std::iter::empty::<wgpu::CommandBuffer>());
        let (sender, receiver) = std::sync::mpsc::channel();
        buffer.slice(..).map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result.is_ok());
        });
        gpu.device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        receiver.recv().unwrap() && buffer.slice(..).get_mapped_range().unwrap().iter().copied().eq(1..=16u8)
    }

    /// A device lost while a backend used it is dropped when the backend is, never pooled, and
    /// the next device opens on a new instance and adapter. Holding `OPENING` keeps the other
    /// tests of the process from opening one in between.
    #[test]
    fn a_lost_device_is_not_pooled_and_the_instance_is_opened_anew() {
        let Some(gpu) = open() else { return };
        let lost = Arc::clone(&gpu.lost);
        gpu.device.destroy();
        {
            let _opening = lock(&OPENING);
            release(gpu);
            assert!(lost.load(Ordering::Acquire), "the loss was noticed");
            assert!(lock(&ADAPTER).is_none(), "the instance and adapter of a lost device are dropped");
            assert!(lock(&POOL).idle.iter().all(|idle| !Arc::ptr_eq(&idle.lost, &lost)), "a lost device was pooled");
        }
        let next = open().expect("a device opens after a loss");
        assert!(!Arc::ptr_eq(&next.lost, &lost));
        assert!(works(&next));
        release(next);
    }

    /// Devices that were lost while idle (a GPU reset takes all of them) are never handed out.
    #[test]
    fn idle_devices_that_were_lost_are_discarded() {
        let Some(first) = open() else { return };
        let second = open().expect("a second device opens");
        let lost = [Arc::clone(&first.lost), Arc::clone(&second.lost)];
        let devices = [first.device.clone(), second.device.clone()];
        {
            let _opening = lock(&OPENING);
            release(first);
            release(second);
            for device in &devices {
                device.destroy();
                device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
            }
            assert!(lost.iter().all(|flag| flag.load(Ordering::Acquire)), "the loss was noticed");
        }
        let next = open().expect("a device opens after the idle ones were lost");
        assert!(!lost.iter().any(|flag| Arc::ptr_eq(flag, &next.lost)), "a lost device was handed out");
        assert!(works(&next));
        release(next);
        assert!(lock(&POOL).idle.iter().all(|idle| !lost.iter().any(|flag| Arc::ptr_eq(flag, &idle.lost))), "a lost device stayed in the pool");
    }

    /// The driver gives a process only so many devices (63 or 64 on NVIDIA's Linux driver, where
    /// the next `vkCreateDevice` fails with an error wgpu words "Parent device is lost"): contexts
    /// beyond that wait for one to come back instead of failing. More threads than the limit hold
    /// a device each until the ones that found no device left are waiting, then all let go.
    #[test]
    fn more_devices_than_the_driver_gives_wait_for_one_to_come_back() {
        const THREADS: usize = 72;
        let Some(probe) = open() else { return };
        release(probe);
        let held = AtomicUsize::new(0);
        let go = (Mutex::new(false), Condvar::new());
        let finished = std::thread::scope(|scope| {
            let workers: Vec<_> = (0..THREADS)
                .map(|_| {
                    scope.spawn(|| {
                        let gpu = acquire(DeviceKind::Auto)?;
                        held.fetch_add(1, Ordering::SeqCst);
                        drop(go.1.wait_while(lock(&go.0), |go| !*go).unwrap_or_else(PoisonError::into_inner));
                        release(gpu);
                        Ok::<(), BackendError>(())
                    })
                })
                .collect();
            // Done once all hold one, or the count stopped growing: the rest wait for a device.
            let mut last = (0, Instant::now());
            loop {
                let now = held.load(Ordering::SeqCst);
                if now != last.0 {
                    last = (now, Instant::now());
                }
                if now == THREADS || (now > 0 && last.1.elapsed() > Duration::from_secs(3)) {
                    break;
                }
                std::thread::sleep(Duration::from_millis(50));
            }
            // Visible with `--nocapture`: how many devices the driver gave before the rest waited.
            eprintln!("{} of {THREADS} threads hold a device; the others wait for one", held.load(Ordering::SeqCst));
            *lock(&go.0) = true;
            go.1.notify_all();
            workers.into_iter().map(|worker| worker.join().unwrap()).collect::<Result<Vec<_>, _>>()
        });
        finished.expect("every thread got a device");
        assert_eq!(held.load(Ordering::SeqCst), THREADS);
    }
}
