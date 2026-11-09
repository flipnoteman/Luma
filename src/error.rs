use std::error::Error;
use std::fmt;
use std::panic::Location;
use std::sync::Arc;

/// Result alias used throughout the crate; the error type defaults to [`LumaError`].
pub type Result<T, E = LumaError> = std::result::Result<T, E>;

/// Everything that can go wrong when using luma.
///
/// Errors are [`Clone`] because a poisoned [`Array`](crate::Array) hands its error out every time
/// it is asked for it.
#[derive(Debug, Clone)]
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
    /// A custom kernel's WGSL failed to parse or validate; holds naga's rendered message.
    ShaderCompile(String),
    /// A custom kernel declares resources luma cannot bind (group 0, groups above 1, gaps in the
    /// binding numbers, textures, ...).
    KernelBinding(String),
    /// The requested entry point does not exist or is not a compute entry point.
    KernelEntryPoint(String),
    /// The workgroup size is invalid for the device or disagrees with the shader.
    KernelWorkgroup(String),
    /// A kernel was dispatched with the wrong number of arguments.
    KernelArgCount { expected: usize, actual: usize },
    /// Argument `index` is a storage buffer where a uniform is expected or vice versa.
    KernelArgKind { index: usize, expected: crate::BindingKind, actual: crate::BindingKind },
    /// A shared `&Array` was passed to a `read_write` storage binding; pass `&mut Array`.
    KernelArgAccess { index: usize },
    /// The array's element size differs from the stride of the shader's `array<T>`.
    KernelArgElementSize { index: usize, expected: u32, actual: u32 },
    /// The uniform value could not be serialized with encase.
    KernelArgUniform { index: usize, message: String },
    /// The uniform value is smaller than the shader's uniform type.
    KernelArgUniformSize { index: usize, expected: u64, actual: u64 },
    /// One buffer is bound twice to a launch and at least one of the bindings is writable. This
    /// happens when arrays that share a buffer (see [`Array::reshape`](crate::Array::reshape))
    /// are passed together, or an in-place operation is given a view of its own array.
    KernelArgAlias { first: usize, second: usize },
    /// A launch needs more workgroups in a dimension than the device supports.
    KernelGridTooLarge { workgroups: [u32; 3], max: u32 },
    /// The arrays involved were created by different [`Context`](crate::Context)s.
    ContextMismatch,
    /// A buffer allocation of this many bytes failed because the device is out of memory.
    OutOfMemory { requested: u64 },
    /// The GPU reported an error (validation, internal, ...) while running luma's commands.
    /// It is reported at the next synchronization point.
    Gpu(String),
    /// A lock inside the context was poisoned by a panic on another thread.
    LockPoisoned,
    /// GPU timing is not available (the adapter has no timestamp queries) or not enabled.
    Profiling(String),
    /// An operation on arrays failed. Produced when an operator or math method poisons its result:
    /// records which operation failed and where in the user's code it was called.
    /// [`LumaError::root`] gives the underlying error.
    Op {
        op: &'static str,
        location: &'static Location<'static>,
        source: Arc<LumaError>,
    },
    /// Waiting for the GPU to finish failed.
    Poll(Arc<wgpu::PollError>),
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
            LumaError::ShaderCompile(msg) => write!(f, "Shader failed to compile:\n{msg}"),
            LumaError::KernelBinding(msg) => write!(f, "Unsupported kernel binding: {msg}"),
            LumaError::KernelEntryPoint(msg) => f.write_str(msg),
            LumaError::KernelWorkgroup(msg) => write!(f, "Invalid workgroup size: {msg}"),
            LumaError::KernelArgCount { expected, actual } => {
                write!(f, "Kernel takes {expected} arguments, got {actual}")
            }
            LumaError::KernelArgKind { index, expected, actual } => write!(
                f,
                "Kernel argument {index}: binding is {expected}, but a {actual} was passed"
            ),
            LumaError::KernelArgAccess { index } => write!(
                f,
                "Kernel argument {index}: binding is read_write, pass the array as `&mut`"
            ),
            LumaError::KernelArgElementSize { index, expected, actual } => write!(
                f,
                "Kernel argument {index}: shader array elements are {expected} bytes, \
                 array elements are {actual} bytes"
            ),
            LumaError::KernelArgUniform { index, message } => {
                write!(f, "Kernel argument {index}: cannot encode uniform: {message}")
            }
            LumaError::KernelArgUniformSize { index, expected, actual } => write!(
                f,
                "Kernel argument {index}: uniform is {actual} bytes, shader needs {expected}"
            ),
            LumaError::KernelArgAlias { first, second } => write!(
                f,
                "Kernel arguments {first} and {second} share one buffer and at least one is \
                 writable (arrays created by `reshape` share their buffer)"
            ),
            LumaError::KernelGridTooLarge { workgroups, max } => write!(
                f,
                "Dispatch of {workgroups:?} workgroups exceeds the device limit of {max} per dimension"
            ),
            LumaError::ContextMismatch => {
                f.write_str("Arrays belong to different luma contexts")
            }
            LumaError::OutOfMemory { requested } => {
                write!(f, "Out of GPU memory allocating {requested} bytes")
            }
            LumaError::Gpu(msg) => write!(f, "GPU error: {msg}"),
            LumaError::LockPoisoned => f.write_str("A luma context lock was poisoned"),
            LumaError::Profiling(msg) => write!(f, "GPU profiling: {msg}"),
            LumaError::Op { op, location, source } => write!(
                f,
                "`{op}` failed at {}:{}:{}: {source}",
                location.file(),
                location.line(),
                location.column()
            ),
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
            LumaError::Poll(e) => Some(&**e),
            LumaError::Op { source, .. } => Some(&**source),
            LumaError::Map(e) => Some(e),
            LumaError::MapRange(e) => Some(e),
            _ => None,
        }
    }
}

impl LumaError {
    /// The error underneath any [`LumaError::Op`] wrappers: what actually went wrong.
    pub fn root(&self) -> &LumaError {
        let mut error = self;
        while let LumaError::Op { source, .. } = error {
            error = source;
        }
        error
    }

    /// Where in the user's code the failed operation was called, if this error comes from a
    /// poisoned array.
    pub fn location(&self) -> Option<&'static Location<'static>> {
        match self {
            LumaError::Op { location, .. } => Some(location),
            _ => None,
        }
    }
}
