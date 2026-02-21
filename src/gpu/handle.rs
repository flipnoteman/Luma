use log::debug;
use wgpu::{Device, Features, InstanceDescriptor, InstanceFlags, MemoryHints, PowerPreference, Queue};

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
    let instance = wgpu::Instance::new(&InstanceDescriptor {
        backends: wgpu::Backends::PRIMARY,
        backend_options: wgpu::BackendOptions {
            gl: wgpu::GlBackendOptions {
                gles_minor_version: Default::default(),
            },
            dx12: wgpu::Dx12BackendOptions {
                shader_compiler: Default::default(),
            },
        },
        flags: InstanceFlags::empty(),
    });

    let Some(adapter) = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        })
        .await
    else {
        return Err("Found no adapters.".into());
    };

    debug!("Adapter(s) = {:?}", adapter.get_info());

    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: Some("Device 1"),
                required_features: Features::empty(),
                required_limits: Default::default(),
                memory_hints: MemoryHints::MemoryUsage,
            },
            None,
        )
        .await
        .expect("Error requesting device.");

    Ok(GpuHandle::new(device, queue))
}
