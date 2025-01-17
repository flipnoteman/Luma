#![allow(dead_code)]
use bytemuck;
use bytemuck::Pod;
use flume;
use log::debug;
use std::borrow::Cow;
use std::collections::HashMap;
use std::ops::Index;
use std::sync::{Arc, RwLock};
use wgpu::util::DeviceExt;
use wgpu::{Buffer, Device, Features, InstanceDescriptor, InstanceFlags, MemoryHints, PowerPreference, Queue, ShaderModule};

pub type ShaderResources = HashMap<String, ShaderModule>;

fn decode_operation<'a>(op: Operation) -> &'a str {
    match op {
        Operation::DOUBLE => "double",
        Operation::ADD => "add",
        Operation::SUBTRACT => "subtract",
        Operation::MULTIPLY => "multiply",
        Operation::DIVIDE => "divide",
    }
}

/// GpuHandle
/// This will hold our [Device] and [Queue] for later executions
#[derive(Debug)]
pub struct GpuHandle {
    pub device: Box<Device>,
    pub queue: Box<Queue>,
}

#[derive(Debug)]
pub struct Buffers {
    storage_buffer: Buffer,
    staging_buffer: Buffer,
    dimensions_buffer: Buffer,
}

/// Operations to be performed on the given data.
pub enum Operation {
    DOUBLE, // Still a test operation
    ADD,
    SUBTRACT,
    MULTIPLY,
    DIVIDE,
}

impl GpuHandle {
    pub fn new(device: Device, queue: Queue) -> Self {
        GpuHandle {
            device: Box::new(device),
            queue: Box::new(queue),
        }
    }
}

/// Executor object. Holds [ShaderResources], [GpuHandle] and [Buffer]s for dynamically executing commands on the GPU
/// Shouldn't be called by the user. A static [Executor] must exist for the [Array] to execute operations.
#[derive(Debug)]
pub struct Executor {
    pub adapter: Option<Box<GpuHandle>>,
    pub shaders: Option<Box<ShaderResources>>,
    buffers: Arc<RwLock<HashMap<String, Buffers>>>,
}

impl Default for Executor {
    fn default() -> Self {
        Executor {
            adapter: None,
            shaders: None,
            buffers: Arc::new(RwLock::new(HashMap::new())), // RwLock locks the value so that there can only be one writer at a time. Also, can be used for interior mutability.
        }
    }
}

// Public impl
impl Executor {
    // Create a new ```Executor``` with populated adapter and operations fields.
    pub async fn new(shader_path_directory: &str) -> Result<Self, String> {
        let mut ex = Executor::default();
        let adapter = Executor::get_adapter_info().await?;
        // TODO: Switch this to add shader modules only when you stage the associated function
        let shaders =
            Executor::add_shader_modules_from_directory(&adapter.device, shader_path_directory)
                .await;

        if let Some(shaders) = shaders {
            ex.shaders = Some(Box::new(shaders))
        } else {
            ex.shaders = None
        }
        ex.adapter = Some(Box::new(adapter));

        Ok(ex)
    }

    // Prints Executor fields for debugging. Must have log_level set to debug
    pub fn info(&self) {
        debug!("{:?}", self.shaders);
        debug!("{:?}", self.adapter);
    }

    pub fn drop(&self, id: &String) {
        self.buffers.write().unwrap().remove(id);
    }

    /// Sets up storage and staging (input, output) buffers and adds them to the executor
    pub async fn setup_buffers<T>(&self, dimensions: &[usize; 4], data: &[T], id: String) -> Result<(), String>
    where
        T: Pod,
    {
        let Some(ref adapter) = self.adapter else {
            return Err("No operations loaded".parse().unwrap());
        };
        // Instantiates buffer with data (`numbers`).
        // Usage allowing the buffer to be:
        //   A storage buffer (can be bound within a bind group and thus available to a shader).
        //   The destination of a copy.
        //   The source of a copy.
        let storage_buffer = adapter
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Storage Buffer"),
                contents: bytemuck::cast_slice::<T, u8>(&data),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
            });

        // Instantiates buffer without data.
        // `usage` of buffer specifies how it can be used:
        //   `BufferUsages::MAP_READ` allows it to be read (outside the shader).
        //   `BufferUsages::COPY_DST` allows it to be the destination of the copy.
        let staging_buffer = adapter.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size: storage_buffer.size(),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let dimensions_buffer = adapter.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Dimensions Buffer"),
            contents:  bytemuck::cast_slice::<usize, u8>(dimensions),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
        });

        self.buffers.write().unwrap().insert(
            id.clone(),
            Buffers {
                storage_buffer,
                staging_buffer,
                dimensions_buffer
            }
        );

        Ok(())
    }

    /// Test function.
    /// Doubles the array input
    pub async fn execute_op(&self, id: &String, operation: Operation) -> Result<Vec<u32>, String> {
        // Instantiate our Executor
        let Some(adapter) = self.adapter.as_ref() else {
            return Err("Not operations loaded".parse().unwrap());
        };
        let device = &adapter.device;
        let queue = &adapter.queue;
        let Some(shaders) = self.shaders.as_ref() else {
            return Err("Not operations loaded".parse().unwrap());
        };

        // Get our buffers from our data
        let buffers = self.buffers.read().unwrap();
        let buffer = buffers.get(id).unwrap();
        let staging_buffer = &buffer.staging_buffer;
        let storage_buffer = &buffer.storage_buffer;
        let dimensions_buffer = &buffer.dimensions_buffer;

        // A bind group defines how buffers are accessed by operations.
        // It is to WebGPU what a descriptor set is to Vulkan.
        // `binding` here refers to the `binding` of a buffer in the shader (`layout(set = 0, binding = 0) buffer`).
        // Instantiates the bind group, once again specifying the binding of buffers.
        // let bind_group_layout = compute_pipeline.get_bind_group_layout(0);
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bind_group_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage {
                            read_only: false,
                        },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage {
                            read_only: true,
                        },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }
            ]
        });

        // Now we need to create our bind groups with our buffers.
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: storage_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: dimensions_buffer.as_entire_binding(),
                }
            ],
        });

        // We need to define the layout of our pipeline (shader in this case) we're using as well.
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        // A pipeline specifies the operation of a shader
        // Instantiates the pipeline.
        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Compute Pipeline"),
            layout: Some(&pipeline_layout),
            module: shaders.index(decode_operation(operation)),
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
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
            cpass.insert_debug_marker("");
            cpass.dispatch_workgroups(storage_buffer.size() as u32, 1, 1); // Number of cells to run, the (x,y,z) size of item being processed
        }
        // Sets adds copy operation to command encoder.
        // Will copy data from storage buffer on GPU to staging buffer on CPU.
        encoder.copy_buffer_to_buffer(
            storage_buffer,
            0,
            staging_buffer,
            0,
            staging_buffer.size(),
        );

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

            Ok(result)
        } else {
            Err("failed to run compute on gpu!".into())
        }
    }
}

// Private impl
impl Executor {
    /// Get device description. Should return the highest performance device on a system. Should only be called once unless you need to request another adapter.
    async fn get_adapter_info() -> Result<GpuHandle, String> {
        // Creates adapters and surfaces using the information in the ```InstanceDescriptor```
        let instance = wgpu::Instance::new(&InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            backend_options: wgpu::BackendOptions {
                gl: wgpu::GlBackendOptions {
                    gles_minor_version: Default::default(), // Select which minor version of Open GL to use.
                },
                dx12: wgpu::Dx12BackendOptions {
                    shader_compiler: Default::default(),
                }
            },
            flags: InstanceFlags::empty(), // Instance flags for debugging.
        });

        // Gives us a handle to all gpu compute adapters with the given ```RequestAdapterOptions```
        let Some(adapter) = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: PowerPreference::HighPerformance, // HighPerformance will tell it to return adapters that offer higher performance, like GPUs.
                force_fallback_adapter: false, // If true, will force WGPU to use an adapter that is supported by all hardware.
                compatible_surface: None, // If given a surface (like a window / display) it will return adapters that can present to that surface.
            })
            .await
        else {
            return Err("Found no adapters.".parse().unwrap());
        };

        debug!("Adapter(s) = {:?}", adapter.get_info());

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("Device 1"),                // Debug label
                    required_features: Features::empty(), // Define a list of features that the device must implement.
                    required_limits: Default::default(), // Defines a list of limits of certain types of resources that we can create.
                    memory_hints: MemoryHints::MemoryUsage, // Defines memory allocation hints for our device.
                },
                None, // Typically a path used for tracing api calls.
            )
            .await
            .expect("Error requesting device.");

        Ok(GpuHandle::new(device, queue))
    }

    /// Returns a list of [ShaderModule] after being given a list of shader paths
    async fn add_shader_modules<'a>(
        device: &Device,
        shader_paths: &[String],
    ) -> Option<ShaderResources> {
        let mut shader_module_hm = HashMap::new();

        // iterate paths in shader_paths and create shader modules
        for path in shader_paths {
            let cs_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(path),
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(
                    &*std::fs::read_to_string(path).unwrap(),
                )),
            });
            shader_module_hm.insert(path.to_owned(), cs_module);
        }
        Some(shader_module_hm)
    }

    /// Returns a list of [ShaderModule]s from a given directory
    async fn add_shader_modules_from_directory<'a>(
        device: &Device,
        shaders_directory: &str,
    ) -> Option<ShaderResources> {
        let mut shader_module_hm = HashMap::new();

        let shader_paths = match std::fs::read_dir(shaders_directory) {
            Ok(s) => s,
            Err(_) => return None,
        }
        .map(|path| path.unwrap().path().into_os_string().into_string().unwrap())
        .collect::<Vec<String>>();

        // Iterate paths and create shader modules out of them
        for path in shader_paths.iter() {
            let file_name = path
                .strip_prefix(shaders_directory)
                .unwrap()
                .strip_prefix("\\")
                .unwrap()
                .strip_suffix(".wgsl")
                .unwrap();

            let shader: Cow<str> = Cow::from(
                std::fs::read_to_string(path)
                    .expect(format!("Could not read file contents from: {}", path).as_str()),
            );
            let cs_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(file_name),
                source: wgpu::ShaderSource::Wgsl(shader),
            });
            shader_module_hm.insert(file_name.to_owned(), cs_module);
        }

        Some(shader_module_hm)
    }
}
