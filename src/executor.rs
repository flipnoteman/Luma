#![allow(dead_code)]
use bytemuck::Pod;
use log::debug;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use wgpu::util::DeviceExt;
use wgpu::Buffer;

use crate::gpu::handle::{self, GpuHandle};
use crate::operations::{self, ShaderResources};

/// Converts host-side dimensions to the `array<u32, 4>` layout the shaders expect.
fn dims_to_u32(dims: &[usize; 4]) -> Result<[u32; 4], String> {
    let mut out = [0u32; 4];
    for (o, d) in out.iter_mut().zip(dims) {
        *o = u32::try_from(*d).map_err(|_| format!("Dimension {d} exceeds u32::MAX"))?;
    }
    Ok(out)
}

#[derive(Debug)]
pub struct Buffers {
    pub(crate) storage_buffer: Buffer,
    pub(crate) staging_buffer: Option<Buffer>,
    pub(crate) dimensions_buffer: Buffer,
    pub(crate) element_size: usize,
    pub(crate) type_suffix: String,
}

/// Executor object. Holds [ShaderResources], [GpuHandle] and [Buffer]s for dynamically executing commands on the GPU.
/// Shouldn't be called by the user. A static [Executor] must exist for the [Array] to execute operations.
#[derive(Debug)]
pub struct Executor {
    pub adapter: Option<Box<GpuHandle>>,
    pub shaders: Option<Box<ShaderResources>>,
    pub(crate) buffers: Arc<RwLock<HashMap<String, Buffers>>>,
}

impl Default for Executor {
    fn default() -> Self {
        Executor {
            adapter: None,
            shaders: None,
            buffers: Arc::new(RwLock::new(HashMap::new())),
        }
    }
}

impl Executor {
    pub async fn new() -> Result<Self, String> {
        let mut ex = Executor::default();
        let adapter = handle::get_adapter_info().await?;
        ex.shaders = Some(Box::new(operations::create_shader_modules(&adapter.device)));
        ex.adapter = Some(Box::new(adapter));

        Ok(ex)
    }

    pub fn info(&self) {
        debug!("{:?}", self.shaders);
        debug!("{:?}", self.adapter);
    }

    pub fn drop(&self, id: &String) {
        self.buffers.write().unwrap().remove(id);
    }

    /// Sets up a storage buffer with data and stores it under the given id.
    /// Staging buffer is lazy — only created on readback.
    pub async fn setup_buffers<T>(&self, dimensions: &[usize; 4], data: &[T], id: String, type_suffix: String) -> Result<(), String>
    where
        T: Pod,
    {
        let Some(ref adapter) = self.adapter else {
            return Err("No adapter loaded".into());
        };

        let storage_buffer = adapter
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Storage Buffer"),
                contents: bytemuck::cast_slice::<T, u8>(data),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
            });

        let dimensions_buffer = adapter.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Dimensions Buffer"),
            contents: bytemuck::cast_slice::<u32, u8>(&dims_to_u32(dimensions)?),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });

        self.buffers.write().unwrap().insert(
            id,
            Buffers {
                storage_buffer,
                staging_buffer: None,
                dimensions_buffer,
                element_size: std::mem::size_of::<T>(),
                type_suffix,
            },
        );

        Ok(())
    }

    /// Read data back from the GPU to CPU. Creates a staging buffer lazily if needed.
    pub async fn readback(&self, id: &str) -> Result<Vec<u8>, String> {
        let adapter = self.adapter.as_ref().ok_or("No adapter loaded")?;
        let device = &adapter.device;
        let queue = &adapter.queue;

        // Ensure staging buffer exists
        {
            let mut buffers = self.buffers.write().unwrap();
            let buf = buffers.get_mut(id).ok_or("Buffer not found")?;
            if buf.staging_buffer.is_none() {
                let staging = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("Staging Buffer"),
                    size: buf.storage_buffer.size(),
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                buf.staging_buffer = Some(staging);
            }
        }

        let buffers = self.buffers.read().unwrap();
        let buf = buffers.get(id).ok_or("Buffer not found")?;
        let storage_buffer = &buf.storage_buffer;
        let staging_buffer = buf.staging_buffer.as_ref().unwrap();

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.copy_buffer_to_buffer(storage_buffer, 0, staging_buffer, 0, storage_buffer.size());
        queue.submit(Some(encoder.finish()));

        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = flume::bounded(1);
        buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

        device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| format!("GPU poll failed: {e}"))?;

        if let Ok(Ok(())) = receiver.recv_async().await {
            let data = buffer_slice
                .get_mapped_range()
                .map_err(|e| format!("Failed to map staging buffer: {e}"))?;
            let result: Vec<u8> = data.to_vec();
            drop(data);
            staging_buffer.unmap();
            Ok(result)
        } else {
            Err("Failed to read data back from GPU".into())
        }
    }

    /// Get the element size for a given buffer id.
    pub fn element_size(&self, id: &str) -> Result<usize, String> {
        let buffers = self.buffers.read().unwrap();
        let buf = buffers.get(id).ok_or("Buffer not found")?;
        Ok(buf.element_size)
    }

    /// Get the type suffix for a given buffer id.
    pub fn type_suffix(&self, id: &str) -> Result<String, String> {
        let buffers = self.buffers.read().unwrap();
        let buf = buffers.get(id).ok_or("Buffer not found")?;
        Ok(buf.type_suffix.clone())
    }
}
