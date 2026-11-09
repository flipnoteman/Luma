use std::error::Error;
use std::fmt;

/// Result alias used throughout the crate; the error type defaults to [`LumaError`].
pub type Result<T, E = LumaError> = std::result::Result<T, E>;

/// Everything that can go wrong when using luma.
#[derive(Debug)]
pub enum LumaError {
    /// No suitable GPU adapter was found.
    NoAdapter(wgpu::RequestAdapterError),
    /// The adapter refused to create a device.
    RequestDevice(wgpu::RequestDeviceError),
    /// The operands of a binary operation have different shapes.
    ShapeMismatch { lhs: [usize; 4], rhs: [usize; 4] },
    /// The number of host elements does not match the number of elements of the array.
    LengthMismatch { expected: usize, actual: usize },
    /// A shape or operation argument is invalid (zero-sized dimension, kernel too large, ...).
    InvalidShape(String),
    /// A dimension or the total element count does not fit in `u32`, which the GPU kernels use.
    DimensionTooLarge { shape: [usize; 4] },
    /// A GPU buffer of this size exceeds the device's buffer or storage-binding limit.
    BufferTooLarge { requested: u64, max: u64 },
    /// The compute kernel for an operation is not embedded in the crate.
    ShaderNotFound(String),
    /// The arrays involved were created by different [`Context`](crate::Context)s.
    ContextMismatch,
    /// Waiting for the GPU to finish failed.
    Poll(wgpu::PollError),
    /// Mapping the readback buffer failed.
    Map(wgpu::BufferAsyncError),
    /// Accessing the mapped readback buffer failed.
    MapRange(wgpu::MapRangeError),
}

impl fmt::Display for LumaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LumaError::NoAdapter(e) => write!(f, "Found no adapters: {e}"),
            LumaError::RequestDevice(e) => write!(f, "Error requesting device: {e}"),
            LumaError::ShapeMismatch { lhs, rhs } => {
                write!(f, "Dimension mismatch: {lhs:?} vs {rhs:?}")
            }
            LumaError::LengthMismatch { expected, actual } => {
                write!(f, "Length mismatch: array holds {expected} elements, got {actual}")
            }
            LumaError::InvalidShape(msg) => f.write_str(msg),
            LumaError::DimensionTooLarge { shape } => write!(
                f,
                "Shape {shape:?} exceeds u32::MAX in a dimension or in its total element count"
            ),
            LumaError::BufferTooLarge { requested, max } => {
                write!(f, "Buffer of {requested} bytes exceeds the device limit of {max} bytes")
            }
            LumaError::ShaderNotFound(name) => write!(f, "Shader '{name}' not found"),
            LumaError::ContextMismatch => {
                f.write_str("Arrays belong to different luma contexts")
            }
            LumaError::Poll(e) => write!(f, "GPU poll failed: {e}"),
            LumaError::Map(e) => write!(f, "Failed to read data back from GPU: {e}"),
            LumaError::MapRange(e) => write!(f, "Failed to map staging buffer: {e}"),
        }
    }
}

impl Error for LumaError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            LumaError::NoAdapter(e) => Some(e),
            LumaError::RequestDevice(e) => Some(e),
            LumaError::Poll(e) => Some(e),
            LumaError::Map(e) => Some(e),
            LumaError::MapRange(e) => Some(e),
            _ => None,
        }
    }
}
