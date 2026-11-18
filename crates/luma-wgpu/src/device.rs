//! The process-wide pool of opened wgpu devices, and the driver's pipeline cache of each.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock, PoisonError};

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
#[derive(Debug)]
pub(crate) struct PipelineStore {
    cache: Option<wgpu::PipelineCache>,
    disk: Arc<DiskCache>,
    /// Whether pipelines were built since the cache was last written.
    pending: AtomicBool,
}

impl PipelineStore {
    /// Opens the store of `device`, loading the stored cache if there is one.
    fn open(disk: Arc<DiskCache>, device: &Device) -> Arc<PipelineStore> {
        let wanted = disk.dir().is_some() && device.features().contains(wgpu::Features::PIPELINE_CACHE);
        let cache = wanted.then(|| {
            let stored = disk.load_pipelines();
            // SAFETY: the data is empty or was returned by `PipelineCache::get_data` (the
            // checksum of the disk cache guards against a damaged file), and wgpu and the driver
            // check that it belongs to this adapter and driver; with `fallback` an unusable blob
            // only yields an empty cache.
            unsafe {
                device.create_pipeline_cache(&wgpu::PipelineCacheDescriptor { label: Some("luma"), data: stored.as_deref(), fallback: true })
            }
        });
        Arc::new(PipelineStore { cache, disk, pending: AtomicBool::new(false) })
    }

    /// The pipeline cache to hand to pipeline creation.
    pub(crate) fn cache(&self) -> Option<&wgpu::PipelineCache> {
        self.cache.as_ref()
    }

    /// Whether there is a cache that [`flush`](Self::flush) writes.
    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    pub(crate) fn is_active(&self) -> bool {
        self.cache.is_some()
    }

    /// Notes that a pipeline was just created with [`cache`](Self::cache). The cache is written
    /// when the device is released and at process exit, never while pipelines are being built:
    /// reading the driver's cache data and writing megabytes of it slowed the first launches of a
    /// run by tens of milliseconds even from a background thread.
    pub(crate) fn built(&self) {
        if self.cache.is_some() {
            self.pending.store(true, Ordering::Release);
        }
    }

    /// Writes the pipeline cache out, on the calling thread, if pipelines were built since it was
    /// last written.
    pub(crate) fn flush(&self) {
        if self.pending.swap(false, Ordering::AcqRel)
            && let Some(data) = self.cache.as_ref().and_then(wgpu::PipelineCache::get_data)
        {
            self.disk.store_pipelines(&data);
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

/// Other platforms cannot read the heap sizes: no automatic memory budget.
#[cfg(not(any(target_os = "linux", target_os = "android", windows)))]
fn device_local_bytes(_adapter: &wgpu::Adapter) -> Option<u64> {
    None
}

/// Picks an adapter of the wanted kind and opens a [Device] / [Queue] pair on it. Only
/// [`acquire`] calls this.
async fn request_device() -> Result<Gpu, BackendError> {
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
    let instance = wgpu::Instance::new(instance_desc);

    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: PowerPreference::HighPerformance,
            ..Default::default()
        })
        .await
        .map_err(|e| BackendError::NoAdapter(e.to_string()))?;
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
    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    crate::exit::track(&device, validated);
    let info = adapter.get_info();
    let disk = DiskCache::open(cache_dir, adapter_key(&info));
    let store = PipelineStore::open(Arc::clone(&disk), &device);
    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    if !validated {
        crate::exit::flush_at_exit(&store);
    }
    Ok(Gpu { device, queue, adapter: info, lost, memory, cost: Arc::new(OnceLock::new()), disk, store })
}

/// Process-wide pool of opened GPU devices.
///
/// Destroying a Vulkan device while other threads are inside the driver deadlocks the NVIDIA
/// driver (615.71): the thread in `vkDestroyDevice` holds the driver's global mutex and the
/// loader lock and joins a driver worker thread that never exits. Devices are therefore never
/// destroyed: [`release`] parks the device of a finished backend in an idle list and
/// [`acquire`] hands it to the next one. The pool never shrinks; it holds as many devices
/// as backends were alive at once.
static IDLE: Mutex<Vec<Gpu>> = Mutex::new(Vec::new());

/// Serializes [`acquire`]. A burst of threads opening backends must not each create a device
/// while the devices of finished ones are about to come back, and the driver caps how many
/// devices can be alive at once (around 60 on NVIDIA; `RequestDevice(Device(Lost))` beyond).
static OPENING: Mutex<()> = Mutex::new(());

/// Hands out an idle pooled device of the wanted kind, or opens a new one when none is idle.
///
/// Holds `OPENING` for the whole call so that concurrent callers queue up and reuse the devices
/// that [`release`] returns in the meantime instead of all creating their own.
pub(crate) fn acquire(kind: DeviceKind) -> Result<Gpu, BackendError> {
    if kind == DeviceKind::Cpu {
        return Err(BackendError::NoDevice("software adapters are not used; the cpu backend runs on the CPU"));
    }
    let _opening = lock(&OPENING);
    if let Some(gpu) = lock(&IDLE).pop() {
        return Ok(gpu);
    }
    crate::exit::driver_call(|| pollster::block_on(request_device()))
}

/// Returns the device of a finished backend to the pool.
///
/// Waits for the device to go idle first, so that the backend's submitted work, its map
/// callbacks and the deferred destruction of its resources are finished before the next owner
/// gets the device (destroying the device used to wait for exactly that). A lost device is
/// dropped instead of pooled, as it must not be handed out again. Does not take `OPENING`: a
/// thread returning a device must never wait for one being created.
pub(crate) fn release(gpu: Gpu) {
    let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
    gpu.store.flush();
    if gpu.lost.load(Ordering::Acquire) {
        return;
    }
    lock(&IDLE).push(gpu);
}
