use log::debug;
use wgpu::{Device, InstanceDescriptor, MemoryHints, PowerPreference, Queue};

use crate::error::{LumaError, Result};

/// Picks an adapter and opens a [Device] / [Queue] pair on it; also returns what the adapter is.
pub(crate) async fn request_device() -> Result<(Device, Queue, wgpu::AdapterInfo)> {
    let mut instance_desc = InstanceDescriptor::new_without_display_handle();
    instance_desc.backends = wgpu::Backends::PRIMARY;
    let instance = wgpu::Instance::new(instance_desc);

    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: PowerPreference::HighPerformance,
            ..Default::default()
        })
        .await
        .map_err(LumaError::NoAdapter)?;

    debug!("Adapter(s) = {:?}", adapter.get_info());

    // Request the adapter's full limits so large arrays aren't capped by the
    // conservative WebGPU defaults (e.g. 128 MiB storage bindings).
    // Timestamp queries are optional: they only serve `Context::profile`.
    let mut required_features = adapter.features() & wgpu::Features::TIMESTAMP_QUERY;
    // Only on unified memory: on a discrete GPU mappable storage buffers live in slow host memory.
    if adapter.get_info().device_type == wgpu::DeviceType::IntegratedGpu {
        required_features |= adapter.features() & wgpu::Features::MAPPABLE_PRIMARY_BUFFERS;
    }

    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("Device 1"),
            required_features,
            required_limits: adapter.limits(),
            memory_hints: MemoryHints::MemoryUsage,
            ..Default::default()
        })
        .await
        .map_err(LumaError::RequestDevice)?;
    Ok((device, queue, adapter.get_info()))
}
