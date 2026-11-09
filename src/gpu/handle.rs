use log::debug;
use wgpu::{Device, InstanceDescriptor, MemoryHints, PowerPreference, Queue};

/// GpuHandle
/// This will hold our [Device] and [Queue] for later executions
#[derive(Debug)]
pub struct GpuHandle {
    pub device: Box<Device>,
    pub queue: Box<Queue>,
}

impl GpuHandle {
    pub fn new(device: Device, queue: Queue) -> Self {
        GpuHandle {
            device: Box::new(device),
            queue: Box::new(queue),
        }
    }
}

pub async fn get_adapter_info() -> Result<GpuHandle, String> {
    let mut instance_desc = InstanceDescriptor::new_without_display_handle();
    instance_desc.backends = wgpu::Backends::PRIMARY;
    let instance = wgpu::Instance::new(instance_desc);

    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: PowerPreference::HighPerformance,
            ..Default::default()
        })
        .await
        .map_err(|e| format!("Found no adapters: {e}"))?;

    debug!("Adapter(s) = {:?}", adapter.get_info());

    // Request the adapter's full limits so large arrays aren't capped by the
    // conservative WebGPU defaults (e.g. 128 MiB storage bindings).
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("Device 1"),
            required_limits: adapter.limits(),
            memory_hints: MemoryHints::MemoryUsage,
            ..Default::default()
        })
        .await
        .map_err(|e| format!("Error requesting device: {e}"))?;

    Ok(GpuHandle::new(device, queue))
}
