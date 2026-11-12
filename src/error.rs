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
#[non_exhaustive]
pub enum LumaError {
    /// No suitable GPU adapter was found.
    NoAdapter(wgpu::RequestAdapterError),
    /// The requested kind of device is not available.
    NoDevice(&'static str),
    /// The operation cannot run over an array that is split into several buffers.
    Chunked(&'static str),
    /// The adapter refused to create a device.
    RequestDevice(wgpu::RequestDeviceError),
    /// The operands of a binary operation have different shapes.
    ShapeMismatch {
        /// Shape of the left operand.
        lhs: [usize; 4],
        /// Shape of the right operand.
        rhs: [usize; 4],
    },
    /// The number of host elements does not match the number of elements of the array.
    LengthMismatch {
        /// Number of elements the array holds.
        expected: usize,
        /// Number of elements that were supplied.
        actual: usize,
    },
    /// A shape or operation argument is invalid (zero-sized dimension, kernel too large, ...).
    InvalidShape(String),
    /// A dimension or the total element count does not fit in `u32`, which the GPU kernels use.
    DimensionTooLarge {
        /// The offending shape.
        shape: [usize; 4],
    },
    /// A GPU buffer of this size exceeds the device's buffer or storage-binding limit.
    BufferTooLarge {
        /// Size of the requested buffer in bytes.
        requested: u64,
        /// The device's limit in bytes.
        max: u64,
    },
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
    KernelArgCount {
        /// Number of bindings the kernel declares.
        expected: usize,
        /// Number of arguments that were passed.
        actual: usize,
    },
    /// Argument `index` is a storage buffer where a uniform is expected or vice versa.
    KernelArgKind {
        /// Index of the offending argument.
        index: usize,
        /// The kind of binding the shader declares.
        expected: crate::BindingKind,
        /// The kind of value that was passed.
        actual: crate::BindingKind,
    },
    /// A shared `&Array` was passed to a `read_write` storage binding; pass `&mut Array`.
    KernelArgAccess {
        /// Index of the offending argument.
        index: usize,
    },
    /// The array's element size differs from the stride of the shader's `array<T>`.
    KernelArgElementSize {
        /// Index of the offending argument.
        index: usize,
        /// Byte stride of the shader's array element type.
        expected: u32,
        /// Byte size of the array's elements.
        actual: u32,
    },
    /// The uniform value could not be serialized with encase.
    KernelArgUniform {
        /// Index of the offending argument.
        index: usize,
        /// The encoder's error message.
        message: String,
    },
    /// The uniform value is smaller than the shader's uniform type.
    KernelArgUniformSize {
        /// Index of the offending argument.
        index: usize,
        /// Byte size of the shader's uniform type.
        expected: u64,
        /// Byte size of the value that was passed.
        actual: u64,
    },
    /// One buffer is bound twice to a launch and at least one of the bindings is writable. This
    /// happens when arrays that share a buffer (see [`Array::reshape`](crate::Array::reshape))
    /// are passed together to a custom kernel.
    KernelArgAlias {
        /// Index of the first argument bound to the shared buffer.
        first: usize,
        /// Index of the second argument bound to the shared buffer.
        second: usize,
    },
    /// A launch needs more workgroups in a dimension than the device supports.
    KernelGridTooLarge {
        /// The workgroup counts the launch needs per dimension.
        workgroups: [u32; 3],
        /// The device's limit of workgroups per dimension.
        max: u32,
    },
    /// The arrays involved were created by different [`Context`](crate::Context)s.
    ContextMismatch,
    /// A buffer allocation of this many bytes failed because the device is out of memory.
    OutOfMemory {
        /// Size of the failed allocation in bytes.
        requested: u64,
    },
    /// The GPU reported an error (validation, internal, ...) while running luma's commands.
    /// It is reported at the next synchronization point.
    Gpu(String),
    /// A lock inside the context was poisoned by a panic on another thread.
    LockPoisoned,
    /// The array cannot be written into: it is the state handed to a traced iteration step.
    ReadOnly(&'static str),
    /// The operation does not exist for this element type or on this device (`f64` and
    /// `Df32` arrays have a reduced set of operations; `f64` needs 64-bit shader floats).
    Unsupported(String),
    /// The finite check (see `ContextOptions::check_finite`) found a NaN or an infinity in the
    /// result of an operation.
    NonFinite,
    /// GPU timing is not available (the adapter has no timestamp queries) or not enabled.
    Profiling(String),
    /// An operation on arrays failed. Produced when an operator or math method poisons its result:
    /// records which operation failed and where in the user's code it was called.
    /// [`LumaError::root`] gives the underlying error.
    Op {
        /// Name of the operation that failed (the operator or method).
        op: &'static str,
        /// Where in the user's code the operation was called.
        location: &'static Location<'static>,
        /// The error that made the operation fail.
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
            LumaError::NoDevice(e) => write!(f, "No suitable device: {e}"),
            LumaError::Chunked(e) => write!(f, "Array spans several buffers: {e}"),
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
            LumaError::ReadOnly(what) => write!(f, "cannot write into {what}"),
            LumaError::Unsupported(what) => f.write_str(what),
            LumaError::NonFinite => f.write_str("the result contains NaN or Inf"),
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
