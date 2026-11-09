use std::borrow::Cow;
use std::collections::HashMap;
use wgpu::{Device, ShaderModule};

pub type ShaderResources = HashMap<String, ShaderModule>;

/// Access mode of a storage-buffer binding in a compute kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Access {
    ReadWrite,
    ReadOnly,
}

use Access::{ReadOnly as RO, ReadWrite as RW};

/// Operations to be performed on the given data.
///
/// Every operation maps to one compute kernel per element type and has a fixed binding layout,
/// see [`Operation::bindings`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[allow(non_camel_case_types)]
pub enum Operation {
    DOUBLE,
    ADD,
    SUBTRACT,
    MULTIPLY,
    DIVIDE,
    CONV1D,
    FFT_BIT_REVERSE,
    FFT_BUTTERFLY,
    FFT_COMPLEX_MUL,
    IFFT_BUTTERFLY,
    FFT_EXTRACT_REAL,
    FFT_COMPLEX_BIT_REVERSE,
}

impl Operation {
    pub fn name(self) -> &'static str {
        match self {
            Operation::DOUBLE => "double",
            Operation::ADD => "add",
            Operation::SUBTRACT => "subtract",
            Operation::MULTIPLY => "multiply",
            Operation::DIVIDE => "divide",
            Operation::CONV1D => "conv1d",
            Operation::FFT_BIT_REVERSE => "fft_bit_reverse",
            Operation::FFT_BUTTERFLY => "fft_butterfly",
            Operation::FFT_COMPLEX_MUL => "fft_complex_mul",
            Operation::IFFT_BUTTERFLY => "ifft_butterfly",
            Operation::FFT_EXTRACT_REAL => "fft_extract_real",
            Operation::FFT_COMPLEX_BIT_REVERSE => "fft_complex_bit_reverse",
        }
    }

    /// Storage-buffer bindings of the kernel in `@group(0)`, in binding order.
    ///
    /// Must match the `@binding(n)` declarations in `operations/*.wgsl`.
    pub fn bindings(self) -> &'static [Access] {
        match self {
            // data (in/out), dims
            Operation::DOUBLE => &[RW, RO],
            // output (in/out), rhs, dims, rhs dims
            Operation::ADD | Operation::SUBTRACT | Operation::MULTIPLY | Operation::DIVIDE => {
                &[RW, RO, RO, RO]
            }
            // output, input, kernel, params
            Operation::CONV1D => &[RW, RO, RO, RO],
            // output, input, params
            Operation::FFT_BIT_REVERSE
            | Operation::FFT_EXTRACT_REAL
            | Operation::FFT_COMPLEX_BIT_REVERSE => &[RW, RO, RO],
            // data, params
            Operation::FFT_BUTTERFLY | Operation::IFFT_BUTTERFLY => &[RW, RO],
            // result, a, b, params
            Operation::FFT_COMPLEX_MUL => &[RW, RO, RO, RO],
        }
    }
}

/// Name of the embedded shader implementing `op` for the element type with `type_suffix`.
pub fn shader_name(op: Operation, type_suffix: &str) -> String {
    format!("{}{}", op.name(), type_suffix)
}

/// Shader sources embedded at compile time, keyed by file stem.
const SHADER_SOURCES: &[(&str, &str)] = &[
    ("add", include_str!("../operations/add.wgsl")),
    ("add_f32", include_str!("../operations/add_f32.wgsl")),
    ("conv1d_f32", include_str!("../operations/conv1d_f32.wgsl")),
    ("divide", include_str!("../operations/divide.wgsl")),
    ("divide_f32", include_str!("../operations/divide_f32.wgsl")),
    ("double", include_str!("../operations/double.wgsl")),
    ("double_f32", include_str!("../operations/double_f32.wgsl")),
    ("fft_bit_reverse_f32", include_str!("../operations/fft_bit_reverse_f32.wgsl")),
    ("fft_butterfly_f32", include_str!("../operations/fft_butterfly_f32.wgsl")),
    ("fft_complex_bit_reverse_f32", include_str!("../operations/fft_complex_bit_reverse_f32.wgsl")),
    ("fft_complex_mul_f32", include_str!("../operations/fft_complex_mul_f32.wgsl")),
    ("fft_extract_real_f32", include_str!("../operations/fft_extract_real_f32.wgsl")),
    ("ifft_butterfly_f32", include_str!("../operations/ifft_butterfly_f32.wgsl")),
    ("multiply", include_str!("../operations/multiply.wgsl")),
    ("multiply_f32", include_str!("../operations/multiply_f32.wgsl")),
    ("subtract", include_str!("../operations/subtract.wgsl")),
    ("subtract_f32", include_str!("../operations/subtract_f32.wgsl")),
];

/// Compiles all embedded shaders into [ShaderModule]s keyed by name.
pub fn create_shader_modules(device: &Device) -> ShaderResources {
    SHADER_SOURCES
        .iter()
        .map(|(name, source)| {
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(name),
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(source)),
            });
            ((*name).to_owned(), module)
        })
        .collect()
}
