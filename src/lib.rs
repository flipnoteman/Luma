//! GPU-driven array library: ArrayFire-style math on the GPU in normal Rust.
//!
//! An [`Array`] lives on the GPU. Arithmetic is written with operators and methods, runs on the
//! GPU, and nothing waits until the host needs a result:
//!
//! ```no_run
//! use luma::Array;
//!
//! let x = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], 4)?;
//! let y = Array::from_slice(&[10.0f32, 20.0, 30.0, 40.0], 4)?;
//!
//! let z = (&x * 2.0 + &y).sqrt();           // recorded, not yet run
//! println!("{:?}", z.to_vec()?);            // runs the batch and waits
//! println!("{}", (&x * &y).sum()?);         // reductions return host values
//! # Ok::<(), luma::LumaError>(())
//! ```
//!
//! * **Default device**: arrays and kernels are created on a lazily opened global
//!   [`default_context`]; every constructor has an `_on(&ctx, ..)` form for an explicit
//!   [`Context`].
//! * **Errors do not panic**: operators and math methods return poisoned arrays on failure; the
//!   error (with the source location of the failed operation) comes out as `Err` at
//!   [`Array::to_vec`], [`Array::eval`], a reduction, ... See [`Array`].
//! * **Batching**: operations are recorded and submitted together, see [`Context`].
//! * **Custom kernels**: [`Kernel`] runs your own WGSL on arrays, in the same batch.

// wgpu's handle types nest deeply; proving `Context: Sync` for the global default context
// exceeds the default limit on recent nightlies (future-compat lint
// `recursion_depth_exceeding_limit`).
#![recursion_limit = "256"]

mod array;
mod builtin;
mod context;
mod conv;
mod error;
mod gpu;
mod gpu_type;
mod kernel;
mod ops;
mod shape;

pub use array::Array;
pub use context::{default_context, Context};
pub use error::{LumaError, Result};
pub use gpu_type::{GpuType, ScalarType};
pub use kernel::{
    uniform, BindingInfo, BindingKind, Kernel, KernelArg, KernelBuilder, ProblemSize,
};
pub use ops::{select, Operand, Rhs};
pub use shape::{Offsets, Shape};
