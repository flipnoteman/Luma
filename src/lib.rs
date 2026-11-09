//! GPU-driven linear algebra/math library.
//!
//! Everything starts at a [`Context`], an explicit handle to a GPU device. It creates [`Array`]s,
//! which own their GPU buffer and run operations on the GPU. Operations are submitted without
//! waiting for the GPU; [`Array::read`] is the one call that blocks until results are ready.
//!
//! # Example
//! ```no_run
//! let ctx = luma::Context::new()?;
//!
//! let a = ctx.from_slice(&[3, 1, 1, 1], &[1u32, 6, 5])?;
//! let b = ctx.from_slice(&[3, 1, 1, 1], &[4u32, 12, 10])?;
//!
//! let sum = a.add(&b)?.double()?;
//! assert_eq!(sum.read()?, vec![10, 36, 30]);
//! # Ok::<(), luma::LumaError>(())
//! ```

mod array;
mod context;
mod error;
mod gpu;
mod gpu_type;
mod kernel;
mod operations;

pub use array::Array;
pub use context::Context;
pub use error::{LumaError, Result};
pub use gpu_type::{GpuType, ScalarType};
pub use kernel::{
    uniform, BindingInfo, BindingKind, Kernel, KernelArg, KernelBuilder, ProblemSize,
};
