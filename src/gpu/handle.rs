use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use log::debug;
use wgpu::{Device, InstanceDescriptor, MemoryHints, PowerPreference, Queue};

use crate::context::DeviceKind;
use crate::error::{LumaError, Result};
use crate::gpu::cache::{self, DiskCache};
use crate::gpu::lock;
use crate::jit::cost::CostModel;

/// An opened device with its queue and what the adapter is. Clones are handles to the same
/// device.
#[derive(Clone, Debug)]
pub(crate) struct Gpu {
    pub(crate) device: Device,
    pub(crate) queue: Queue,
    pub(crate) adapter: wgpu::AdapterInfo,
    /// Set by the device-lost callback; a lost device is never pooled again.
    lost: Arc<AtomicBool>,
    /// Whether the adapter is a software (CPU) rasterizer such as Mesa llvmpipe.
    pub(crate) cpu: bool,
    /// Total bytes of the adapter's device-local memory heaps, if the backend can tell (Vulkan
    /// only): the basis of the default memory budget.
    pub(crate) memory: Option<u64>,
    /// The device's measured cost model, set by the first context that calibrates it, so that a
    /// pooled device is calibrated once per process.
    pub(crate) cost: Arc<OnceLock<CostModel>>,
    /// The on-disk cache of pipelines and of the calibration (see [`super::cache`]).
    pub(crate) disk: Arc<DiskCache>,
}

/// Total bytes of the device-local heaps of `adapter`: how much array data a device can hold at
/// most. wgpu has no portable call for it, so this asks Vulkan directly.
#[cfg(any(target_os = "linux", target_os = "android", windows))]
fn device_local_bytes(adapter: &wgpu::Adapter) -> Option<u64> {
    // SAFETY: only reads properties of the physical device; nothing is destroyed.
    let hal = unsafe { adapter.as_hal::<wgpu::hal::api::Vulkan>() }?;
    let props = unsafe {
        hal.shared_instance().raw_instance().get_physical_device_memory_properties(hal.raw_physical_device())
    };
    // VK_MEMORY_HEAP_DEVICE_LOCAL_BIT
    const DEVICE_LOCAL: u32 = 1;
    Some(
        props
            .memory_heaps
            .iter()
            .take(props.memory_heap_count as usize)
            .filter(|heap| heap.flags.as_raw() & DEVICE_LOCAL != 0)
            .map(|heap| heap.size)
            .sum(),
    )
}

/// Other platforms cannot read the heap sizes: no automatic memory budget.
#[cfg(not(any(target_os = "linux", target_os = "android", windows)))]
fn device_local_bytes(_adapter: &wgpu::Adapter) -> Option<u64> {
    None
}

/// Picks an adapter of the wanted kind and opens a [Device] / [Queue] pair on it. Only
/// [`acquire`] calls this.
async fn request_device(kind: DeviceKind) -> Result<Gpu> {
    let mut instance_desc = InstanceDescriptor::new_without_display_handle();
    // Buffer creation reports out-of-memory once usage would pass 90% of the driver's live
    // budget (Vulkan with VK_EXT_memory_budget, DX12) instead of failing later in a command.
    instance_desc.memory_budget_thresholds.for_resource_creation = Some(90);
    instance_desc.backends = wgpu::Backends::PRIMARY;
    // Indirect dispatches only ever read workgroup counts that a gate kernel of this crate
    // copied from a template of valid counts or zeroed (see `jit::until`), so wgpu's check of
    // them, an extra validation dispatch in front of every indirect one, protects nothing.
    instance_desc.flags.remove(wgpu::InstanceFlags::VALIDATION_INDIRECT_CALL);
    #[cfg(target_os = "linux")]
    let validated = instance_desc.flags.contains(wgpu::InstanceFlags::VALIDATION);
    let instance = wgpu::Instance::new(instance_desc);

    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: PowerPreference::HighPerformance,
            force_fallback_adapter: kind == DeviceKind::Cpu,
            ..Default::default()
        })
        .await
        .map_err(LumaError::NoAdapter)?;
    let cpu = adapter.get_info().device_type == wgpu::DeviceType::Cpu;
    if kind == DeviceKind::Gpu && cpu {
        return Err(LumaError::NoDevice("only a software (CPU) adapter is available"));
    }
    debug!("Adapter(s) = {:?}", adapter.get_info());

    // Request the adapter's full limits so large arrays aren't capped by the
    // conservative WebGPU defaults (e.g. 128 MiB storage bindings).
    // Timestamp queries are optional: they only serve `Context::profile`.
    let mut required_features = adapter.features() & wgpu::Features::TIMESTAMP_QUERY;
    // Shader features custom kernels may use; requested whenever the adapter has them.
    required_features |= adapter.features()
        & (wgpu::Features::SUBGROUP
            | wgpu::Features::SHADER_F16
            | wgpu::Features::SHADER_F64
            | wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX);
    // Only on unified memory: on a discrete GPU mappable storage buffers live in slow host memory.
    if adapter.get_info().device_type == wgpu::DeviceType::IntegratedGpu {
        required_features |= adapter.features() & wgpu::Features::MAPPABLE_PRIMARY_BUFFERS;
    }
    // The driver's compiled pipelines are kept between processes (see `cache`).
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
        .map_err(LumaError::RequestDevice)?;
    let lost = Arc::new(AtomicBool::new(false));
    let flag = Arc::clone(&lost);
    device.set_device_lost_callback(move |reason, message| {
        debug!("device lost ({reason:?}): {message}");
        flag.store(true, Ordering::Release);
    });
    let memory = device_local_bytes(&adapter);
    #[cfg(target_os = "linux")]
    super::exit::track(&device, validated);
    let info = adapter.get_info();
    let disk = DiskCache::open(cache_dir, &device, &info);
    #[cfg(target_os = "linux")]
    if !validated {
        cache::flush_at_exit(&disk);
    }
    Ok(Gpu { device, queue, adapter: info, lost, cpu, memory, cost: Arc::new(OnceLock::new()), disk })
}

/// Process-wide pool of opened GPU devices.
///
/// Destroying a Vulkan device while other threads are inside the driver deadlocks the NVIDIA
/// driver (615.71): the thread in `vkDestroyDevice` holds the driver's global mutex and the
/// loader lock and joins a driver worker thread that never exits. Devices are therefore never
/// destroyed: [`release`] parks the device of a finished context in an idle list and
/// [`acquire`] hands it to the next context. The pool never shrinks; it holds as many devices
/// as contexts were alive at once.
static IDLE: Mutex<Vec<Gpu>> = Mutex::new(Vec::new());

/// Serializes [`acquire`]. A burst of threads opening contexts must not each create a device
/// while the devices of finished contexts are about to come back, and the driver caps how many
/// devices can be alive at once (around 60 on NVIDIA; `RequestDevice(Device(Lost))` beyond).
static OPENING: Mutex<()> = Mutex::new(());

/// What [`DeviceKind::Auto`] resolved to the first time: `true` when it ended up on a CPU
/// device. Fixed for the process so that every `Auto` context uses the same kind of device.
static AUTO_IS_CPU: OnceLock<bool> = OnceLock::new();

/// Hands out an idle pooled device of the wanted kind, or opens a new one when none is idle.
///
/// Holds `OPENING` for the whole call so that concurrent callers queue up and reuse the devices
/// that [`release`] returns in the meantime instead of all creating their own.
pub(crate) fn acquire(kind: DeviceKind) -> Result<Gpu> {
    let _opening = lock(&OPENING);
    let mut want = match kind {
        DeviceKind::Gpu => Some(false),
        DeviceKind::Cpu => Some(true),
        DeviceKind::Auto => AUTO_IS_CPU.get().copied(),
    };
    if want.is_none() && lock(&IDLE).iter().any(|gpu| !gpu.cpu) {
        want = Some(false);
        let _ = AUTO_IS_CPU.set(false);
    }
    if let Some(cpu) = want {
        let mut idle = lock(&IDLE);
        if let Some(at) = idle.iter().rposition(|gpu| gpu.cpu == cpu) {
            return Ok(idle.remove(at));
        }
    }
    let gpu = pollster::block_on(request_device(kind))?;
    if kind == DeviceKind::Auto && *AUTO_IS_CPU.get_or_init(|| gpu.cpu) {
        log::info!("luma: no GPU found, running on the CPU through {}", gpu.adapter.name);
    }
    Ok(gpu)
}

/// Returns the device of a finished context to the pool.
///
/// Waits for the device to go idle first, so that the context's submitted work, its map
/// callbacks and the deferred destruction of its resources are finished before the next owner
/// gets the device (destroying the device used to wait for exactly that). A lost device is
/// dropped instead of pooled, as it must not be handed out again. Does not take `OPENING`: a
/// thread returning a device must never wait for one being created.
pub(crate) fn release(gpu: Gpu) {
    let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
    gpu.disk.flush();
    if gpu.lost.load(Ordering::Acquire) {
        return;
    }
    lock(&IDLE).push(gpu);
}
