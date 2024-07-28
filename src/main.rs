use std::borrow::Cow;
use std::collections::HashMap;
use std::error::Error;
use std::ops::Index;
use bytemuck;
use wgpu::{Device, Features, InstanceDescriptor, InstanceFlags, MemoryHints, PowerPreference, Queue, RequestDeviceError, ShaderModule};
use flume;
use wgpu::util::DeviceExt;
type ShaderResources = HashMap<String, ShaderModule>;

struct GpuHandle {
    device: Device,
    queue: Queue,
}


impl GpuHandle {
    fn new(device: Device, queue: Queue) -> Self {
        GpuHandle {
            device,
            queue,
        }
    }
}

pub struct Executor {
    adapter: Option<GpuHandle>,
    shaders: Option<ShaderResources>,
}

impl Default for Executor {
    fn default() -> Self {
        Executor {
            adapter: None,
            shaders: None,
        }
    }
}

impl Executor {
    async fn new(shader_path_directory: &str) -> Result<Self, String> {
        let mut ex = Executor::default();
        let adapter = Executor::get_adapter_info().await?;
        let shaders = Executor::add_shader_modules(&adapter.device, &["./shaders/shader.wgsl".parse().unwrap()]).await;

        ex.shaders = shaders;
        ex.adapter = Some(adapter);
        Ok(ex)
    }

    /// Get device description. Should return the highest performance device on a system. Should only be called once unless you need to request another adapter.
    async fn get_adapter_info() -> Result<GpuHandle, String>{

        // Creates adapters and surfaces using the information in the ```InstanceDescriptor```
        let instance = wgpu::Instance::new(
            InstanceDescriptor {
                backends: wgpu::Backends::PRIMARY,
                flags: InstanceFlags::DEBUG, // Instance flags for debugging.
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

        log::info!("Adapter(s) = {:?}", adapter.get_info());

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

        for path in shader_paths.iter() {
            let cs_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(path),
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(path)),
            });
            shader_module_hm.insert(path.to_owned(), cs_module);
        }

        Some(shader_module_hm)
    }
}



#[tokio::main]
async fn main() {


    let numbers = vec![1u32, 2, 3, 4];

    let mut ex = Box::new(Executor::new("./shaders").await.unwrap());
    let adapter = &ex.adapter.unwrap();
    let device = &adapter.device;
    let queue = &adapter.queue;
    let shaders = ex.shaders;

    // Instantiates buffer with data (`numbers`).
    // Usage allowing the buffer to be:
    //   A storage buffer (can be bound within a bind group and thus available to a shader).
    //   The destination of a copy.
    //   The source of a copy.
    let storage_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Storage Buffer"),
        contents: bytemuck::cast_slice(&numbers),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
    });

    // Instantiates buffer without data.
    // `usage` of buffer specifies how it can be used:
    //   `BufferUsages::MAP_READ` allows it to be read (outside the shader).
    //   `BufferUsages::COPY_DST` allows it to be the destination of the copy.
    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: storage_buffer.size(),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    println!("{:?}", shaders);

    // A bind group defines how buffers are accessed by shaders.
    // It is to WebGPU what a descriptor set is to Vulkan.
    // `binding` here refers to the `binding` of a buffer in the shader (`layout(set = 0, binding = 0) buffer`).

    // A pipeline specifies the operation of a shader

    // Instantiates the pipeline.
    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: None,
        module: shaders.unwrap().index("./shaders/shader.wgsl"),
        entry_point: "main",
        compilation_options: Default::default(),
        cache: None,
    });

    // Instantiates the bind group, once again specifying the binding of buffers.
    let bind_group_layout = compute_pipeline.get_bind_group_layout(0);
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bind_group_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: storage_buffer.as_entire_binding(),
        }],
    });

    // A command encoder executes one or many pipelines.
    // It is to WebGPU what a command buffer is to Vulkan.
    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        cpass.set_pipeline(&compute_pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.insert_debug_marker("compute collatz iterations");
        cpass.dispatch_workgroups(numbers.len() as u32, 1, 1); // Number of cells to run, the (x,y,z) size of item being processed
    }
    // Sets adds copy operation to command encoder.
    // Will copy data from storage buffer on GPU to staging buffer on CPU.
    encoder.copy_buffer_to_buffer(&storage_buffer, 0, &staging_buffer, 0, staging_buffer.size());

    // Submits command encoder for processing
    queue.submit(Some(encoder.finish()));

    // Note that we're not calling `.await` here.
    let buffer_slice = staging_buffer.slice(..);
    // Sets the buffer up for mapping, sending over the result of the mapping back to us when it is finished.
    let (sender, receiver) = flume::bounded(1);
    buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

    // Poll the device in a blocking manner so that our future resolves.
    // In an actual application, `device.poll(...)` should
    // be called in an event loop or on another thread.
    device.poll(wgpu::Maintain::wait()).panic_on_timeout();

    // Awaits until `buffer_future` can be read from
    if let Ok(Ok(())) = receiver.recv_async().await {
        // Gets contents of buffer
        let data = buffer_slice.get_mapped_range();
        // Since contents are got in bytes, this converts these bytes back to u32
        let result: Vec<u32> = bytemuck::cast_slice(&data).to_vec();

        // With the current interface, we have to make sure all mapped views are
        // dropped before we unmap the buffer.
        drop(data);
        staging_buffer.unmap(); // Unmaps buffer from memory
        // If you are familiar with C++ these 2 lines can be thought of similarly to:
        //   delete myPointer;
        //   myPointer = NULL;
        // It effectively frees the memory

        println!("Result = {:?}", result);
    } else {
        panic!("failed to run compute on gpu!");
    }
}
