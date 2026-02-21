use wgpu::util::DeviceExt;

use crate::executor::{Buffers, Executor};
use crate::operations::{shader_name, Operation};

impl Executor {
    /// Execute a unary operation on the GPU. Returns a new buffer id with the result.
    /// The original buffer is untouched.
    pub async fn execute_unary_gpu(&self, src_id: &str, operation: &Operation) -> Result<String, String> {
        let adapter = self.adapter.as_ref().ok_or("No adapter loaded")?;
        let shaders = self.shaders.as_ref().ok_or("No shaders loaded")?;
        let device = &adapter.device;
        let queue = &adapter.queue;

        // Read source buffer info
        let buffers = self.buffers.read().unwrap();
        let src = buffers.get(src_id).ok_or("Source buffer not found")?;
        let src_size = src.storage_buffer.size();
        let element_size = src.element_size;
        let type_suffix = src.type_suffix.clone();
        let dims_size = src.dimensions_buffer.size();

        // Create new output storage buffer
        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Unary Output Storage Buffer"),
            size: src_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Create new dimensions buffer (copy of source dims)
        let dims_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Unary Output Dimensions Buffer"),
            size: dims_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Copy source data into new output buffer and dims
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.copy_buffer_to_buffer(&src.storage_buffer, 0, &output_buffer, 0, src_size);
        encoder.copy_buffer_to_buffer(&src.dimensions_buffer, 0, &dims_buffer, 0, dims_size);
        queue.submit(Some(encoder.finish()));

        // Now run the compute shader on the new output buffer
        let name = shader_name(operation, &type_suffix);
        let shader = shaders.get(&name).ok_or(format!("Shader '{}' not found", name))?;

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Unary Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Unary Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: dims_buffer.as_entire_binding(),
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Unary Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Unary Compute Pipeline"),
            layout: Some(&pipeline_layout),
            module: shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let element_count = (src_size / element_size as u64) as u32;
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            cpass.set_pipeline(&compute_pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.dispatch_workgroups(element_count, 1, 1);
        }
        queue.submit(Some(encoder.finish()));

        // Store result under a new UUID
        let new_id = uuid::Uuid::new_v4().to_string();
        // Drop the read lock before acquiring write lock
        drop(buffers);

        self.buffers.write().unwrap().insert(
            new_id.clone(),
            Buffers {
                storage_buffer: output_buffer,
                staging_buffer: None,
                dimensions_buffer: dims_buffer,
                element_size,
                type_suffix,
            },
        );

        Ok(new_id)
    }

    /// Execute a binary operation on the GPU. Returns a new buffer id with the result.
    /// Both source buffers are untouched.
    pub async fn execute_binary_gpu(&self, lhs_id: &str, rhs_id: &str, operation: &Operation) -> Result<String, String> {
        let adapter = self.adapter.as_ref().ok_or("No adapter loaded")?;
        let shaders = self.shaders.as_ref().ok_or("No shaders loaded")?;
        let device = &adapter.device;
        let queue = &adapter.queue;

        let buffers = self.buffers.read().unwrap();
        let lhs = buffers.get(lhs_id).ok_or("LHS buffer not found")?;
        let rhs = buffers.get(rhs_id).ok_or("RHS buffer not found")?;

        let lhs_size = lhs.storage_buffer.size();
        let element_size = lhs.element_size;
        let type_suffix = lhs.type_suffix.clone();
        let dims_size = lhs.dimensions_buffer.size();

        // Create new output buffer (same size as lhs), copy lhs data into it
        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Binary Output Storage Buffer"),
            size: lhs_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let dims_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Binary Output Dimensions Buffer"),
            size: dims_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.copy_buffer_to_buffer(&lhs.storage_buffer, 0, &output_buffer, 0, lhs_size);
        encoder.copy_buffer_to_buffer(&lhs.dimensions_buffer, 0, &dims_buffer, 0, dims_size);
        queue.submit(Some(encoder.finish()));

        // Set up the 3-binding layout for binary ops: output (rw), rhs (r), dims (r)
        let name = shader_name(operation, &type_suffix);
        let shader = shaders.get(&name).ok_or(format!("Shader '{}' not found", name))?;

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Binary Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Binary Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: rhs.storage_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: dims_buffer.as_entire_binding(),
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Binary Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Binary Compute Pipeline"),
            layout: Some(&pipeline_layout),
            module: shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let element_count = (lhs_size / element_size as u64) as u32;
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            cpass.set_pipeline(&compute_pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.dispatch_workgroups(element_count, 1, 1);
        }
        queue.submit(Some(encoder.finish()));

        let new_id = uuid::Uuid::new_v4().to_string();
        drop(buffers);

        self.buffers.write().unwrap().insert(
            new_id.clone(),
            Buffers {
                storage_buffer: output_buffer,
                staging_buffer: None,
                dimensions_buffer: dims_buffer,
                element_size,
                type_suffix,
            },
        );

        Ok(new_id)
    }

    /// Execute a 1D convolution on the GPU (f32 only).
    /// Returns a new buffer id with the convolution result.
    pub async fn execute_conv1d_gpu(
        &self,
        input_id: &str,
        kernel_id: &str,
        stride: u32,
        padding: u32,
    ) -> Result<(String, usize), String> {
        let adapter = self.adapter.as_ref().ok_or("No adapter loaded")?;
        let shaders = self.shaders.as_ref().ok_or("No shaders loaded")?;
        let device = &adapter.device;
        let queue = &adapter.queue;

        let buffers = self.buffers.read().unwrap();
        let input_buf = buffers.get(input_id).ok_or("Input buffer not found")?;
        let kernel_buf = buffers.get(kernel_id).ok_or("Kernel buffer not found")?;

        let input_len = (input_buf.storage_buffer.size() / input_buf.element_size as u64) as u32;
        let kernel_len = (kernel_buf.storage_buffer.size() / kernel_buf.element_size as u64) as u32;

        if input_len + 2 * padding < kernel_len {
            return Err("Kernel is larger than padded input".into());
        }

        let output_len = ((input_len + 2 * padding - kernel_len) / stride + 1) as usize;
        if output_len == 0 {
            return Err("Output length is zero".into());
        }

        let shader_name = shader_name(&Operation::CONV1D, "_f32");
        let shader = shaders.get(&shader_name).ok_or(format!("Shader '{}' not found", shader_name))?;

        // Create output buffer
        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Conv1D Output Buffer"),
            size: (output_len * 4) as u64, // f32 = 4 bytes
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Create params buffer: [kernel_len, stride, padding]
        let params: [u32; 3] = [kernel_len, stride, padding];
        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Conv1D Params Buffer"),
            contents: bytemuck::cast_slice(&params),
            usage: wgpu::BufferUsages::STORAGE,
        });

        // Create dimensions buffer for the output
        let dims: [usize; 4] = [output_len, 1, 1, 1];
        let dims_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Conv1D Output Dimensions Buffer"),
            contents: bytemuck::cast_slice(&dims),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });

        // 4-binding layout: output(rw), input(r), kernel(r), params(r)
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Conv1D Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Conv1D Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: input_buf.storage_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: kernel_buf.storage_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Conv1D Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Conv1D Compute Pipeline"),
            layout: Some(&pipeline_layout),
            module: shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            cpass.set_pipeline(&compute_pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.dispatch_workgroups(output_len as u32, 1, 1);
        }
        queue.submit(Some(encoder.finish()));

        let new_id = uuid::Uuid::new_v4().to_string();
        drop(buffers);

        self.buffers.write().unwrap().insert(
            new_id.clone(),
            Buffers {
                storage_buffer: output_buffer,
                staging_buffer: None,
                dimensions_buffer: dims_buffer,
                element_size: 4, // f32
                type_suffix: "_f32".to_string(),
            },
        );

        Ok((new_id, output_len))
    }
}
