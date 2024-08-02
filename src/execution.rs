use std::borrow::Cow;
use std::collections::HashMap;
use std::error::Error;
use std::ops::Index;
use bytemuck;
use wgpu::{Device, Features, InstanceDescriptor, InstanceFlags, MemoryHints, PowerPreference, Queue, RequestDeviceError, ShaderModule};
use flume;
use wgpu::util::DeviceExt;
use log::{info, debug};

pub type ShaderResources = HashMap<String, ShaderModule>;

#[derive(Debug)]
pub struct GpuHandle {
    pub device: Device,
    pub queue: Queue,
}

impl GpuHandle {
    pub fn new(device: Device, queue: Queue) -> Self {
        GpuHandle {
            device,
            queue,
        }
    }
}

pub struct Executor {
    pub adapter: Option<GpuHandle>,
    pub shaders: Option<ShaderResources>,
}

impl Default for Executor {
    fn default() -> Self {
        Executor {
            adapter: None,
            shaders: None,
        }
    }
}

// Public impl
impl Executor {
    // Create a new ```Executor``` with populated adapter and operations fields.
    pub async fn new(shader_path_directory: &str) -> Result<Self, String> {
        let mut ex = Executor::default();
        let adapter = Executor::get_adapter_info().await?;
        let shaders = Executor::add_shader_modules_from_directory(&adapter.device, shader_path_directory).await;

        ex.shaders = shaders;
        ex.adapter = Some(adapter);
        Ok(ex)
    }

    // Prints Executor fields for debugging. Must have log_level set to debug
    pub fn info(&self) {
        log::debug!("{:?}", self.shaders);
        log::debug!("{:?}", self.adapter);
    }


}

// Private impl
impl Executor {

    /// Get device description. Should return the highest performance device on a system. Should only be called once unless you need to request another adapter.
    async fn get_adapter_info() -> Result<GpuHandle, String>{

        // Creates adapters and surfaces using the information in the ```InstanceDescriptor```
        let instance = wgpu::Instance::new(
            InstanceDescriptor {
                backends: wgpu::Backends::PRIMARY,
                flags: InstanceFlags::empty(), // Instance flags for debugging.
                dx12_shader_compiler: Default::default(), // Select which DX12 compiler to use.
                gles_minor_version: Default::default(), // Select which minor version of Open GL to use.
            }
        );

        // Gives us a handle to all gpu compute adapters with the given ```RequestAdapterOptions```
        let Some(adapter) = instance.request_adapter(
            &wgpu::RequestAdapterOptions {
                power_preference: PowerPreference::HighPerformance, // HighPerformance will tell it to return adapters that offer higher performance, like GPUs.
                force_fallback_adapter: false, // If true, will force WGPU to use an adapter that is supported by all hardware.
                compatible_surface: None, // If given a surface (like a window / display) it will return adapters that can present to that surface.
            }
        ).await else {
            return Err("Found no adapters.".parse().unwrap())
        };

        log::debug!("Adapter(s) = {:?}", adapter.get_info());

        let (device, queue) = adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("Device 1"), // Debug label
                required_features: Features::empty(), // Define a list of features that the device must implement.
                required_limits: Default::default(), // Defines a list of limits of certain types of resources that we can create.
                memory_hints: MemoryHints::MemoryUsage, // Defines memory allocation hints for our device.
            },
            None // Typically a path used for tracing api calls.
        ).await.expect("Error requesting device.");

        Ok(GpuHandle {
            device,
            queue
        })
    }

    /// Returns a list of ShaderModules after being given a list of shader paths
    async fn add_shader_modules<'a>(
        device: &Device,
        shader_paths: &[String]
    ) -> Option<ShaderResources> {
        let mut shader_module_hm = HashMap::new();

        // iterate paths in shader_paths and create shader modules
        for path in shader_paths {
            let cs_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(path),
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(&*std::fs::read_to_string(path).unwrap())),
            });
            shader_module_hm.insert(path.to_owned(), cs_module);
        }
        Some(shader_module_hm)
    }

    async fn add_shader_modules_from_directory<'a>(
        device: &Device,
        shaders_directory: &str
    ) -> Option<ShaderResources> {
        let mut shader_module_hm = HashMap::new();

        let shader_paths = match std::fs::read_dir(shaders_directory){
            Ok(s) => {s}
            Err(_) => {return None}
        }.map(|path| {
            path.unwrap().path().into_os_string().into_string().unwrap()
        }).collect::<Vec<String>>();

        // Iterate paths and create shader modules out of them
        for path in shader_paths.iter() {
            let file_name = path
                .strip_prefix(shaders_directory).unwrap()
                .strip_prefix("\\").unwrap()
                .strip_suffix(".wgsl").unwrap();

            let shader: Cow<str> = Cow::from(std::fs::read_to_string(path).expect(format!("Could not read file contents from: {}", path).as_str()));
            let cs_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(file_name),
                source: wgpu::ShaderSource::Wgsl(shader),
            });
            shader_module_hm.insert(file_name.to_owned(), cs_module);
        }

        Some(shader_module_hm)
    }
}
