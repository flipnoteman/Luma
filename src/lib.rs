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
//! * **CPU fallback**: without a GPU the context opens a software Vulkan driver (Mesa
//!   llvmpipe) instead; `LUMA_DEVICE=cpu` or [`Context::cpu`] force it. See [`DeviceKind`].
//! * **Arrays larger than a buffer**: an array that exceeds the device's buffer limit is split
//!   over several buffers and the built-in operations run over them in pieces. Array data
//!   beyond the device's memory budget moves to host memory and back on demand
//!   ([`ContextOptions`], [`Context::spilled_bytes`]). Custom kernels need arrays that fit one
//!   buffer ([`LumaError::Chunked`]).
//! * **Errors do not panic**: operators and math methods return poisoned arrays on failure; the
//!   error (with the source location of the failed operation) comes out as `Err` at
//!   [`Array::to_vec`], [`Array::eval`], a reduction, ... See [`Array`].
//! * **Batching**: operations are recorded and submitted together, see [`Context`];
//!   [`Context::flush`] hands the batch to the GPU without waiting for it.
//! * **Custom kernels**: [`Kernel`] runs your own WGSL on arrays, in the same batch.
//! * **GPU timing**: [`Context::profile`] measures the GPU time per kernel where the adapter
//!   supports timestamp queries.

// wgpu's handle types nest deeply; proving `Context: Sync` for the global default context
// exceeds the default limit on recent nightlies (future-compat lint
// `recursion_depth_exceeding_limit`).
#![recursion_limit = "256"]

mod array;
mod builtin;
mod context;
mod conv;
mod error;
mod fft;
mod gpu;
mod gpu_type;
mod jit;
mod kernel;
mod ops;
mod profile;
mod shape;
mod tuning;

pub use array::Array;
pub use context::{default_context, flush, Context, ContextOptions, DeviceKind};
pub use error::{LumaError, Result};
pub use gpu_type::{GpuType, ScalarType};
pub use kernel::{
    uniform, BindingInfo, BindingKind, Kernel, KernelArg, KernelBuilder, ProblemSize,
};
pub use ops::{eval_all, select, Operand, Reduction, Rhs};
pub use profile::{KernelTiming, Profile};
pub use shape::{Offsets, Shape};
pub use tuning::TuningProfile;

/// The `wgpu` crate luma is built on. wgpu types appear in the public API ([`LumaError`]
/// variants, [`Context::features`]); naming them through this re-export guarantees the same
/// version.
pub use wgpu;
