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
//! println!("{}", (&x * &y).sum().item()?);  // reductions are lazy arrays; `item` reads the value
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
//!   [`Array::to_vec`], [`Array::item`], [`Array::eval`], ... See [`Array`].
//! * **Batching**: operations are recorded and submitted together, see [`Context`];
//!   [`Context::flush`] hands the batch to the GPU without waiting for it.
//! * **Custom kernels**: [`Kernel`] runs your own WGSL on arrays, in the same batch.
//! * **GPU timing**: [`Context::profile`] measures the GPU time per kernel where the adapter
//!   supports timestamp queries.

// wgpu's handle types nest deeply; proving `Context: Sync` for the global default context
// exceeds the default limit on recent nightlies (future-compat lint
// `recursion_depth_exceeding_limit`).
#![recursion_limit = "256"]

// The derive emits `::luma::` paths, also inside this crate.
extern crate self as luma;

mod array;
mod builtin;
mod context;
mod conv;
mod cyclic;
mod error;
mod fields;
#[cfg(test)]
mod fuzz;
mod fft;
mod generate;
mod gpu;
mod gpu_type;
mod iterate;
mod jit;
mod kernel;
mod linalg;
mod ops;
mod precision;
mod profile;
mod reduce;
mod scan;
mod scalar;
mod shape;
mod tuning;
mod until;
mod view;

pub use array::Array;
pub use context::{default_context, flush, Context, ContextOptions, DeviceKind};
pub use error::{LumaError, Result};
pub use fields::Fields;
pub use gpu_type::{Df32, GpuType, ScalarType};
pub use kernel::{uniform, view, BindingInfo, BindingKind, Kernel, KernelArg, KernelBuilder, ProblemSize, UniformValue};
pub use iterate::iterate_all;
pub use until::{iterate_until_all, Cond, History, Run, Series, Stat, StopReason, Until};
pub use linalg::{solve_block_tridiagonal, solve_block_tridiagonal_from, solve_tridiagonal, solve_tridiagonal_from};
pub use jit::cost::CostModel;
pub use jit::fusion_limit;
pub use ops::{eval_all, select, Operand};
pub use reduce::Reduction;
pub use scalar::Scalar;
pub use profile::{FusionSplit, KernelTiming, Profile, SplitReason};
pub use shape::{Offsets, Shape};
pub use tuning::TuningProfile;
pub use luma_derive::Fields;
pub use view::{concat, concat_slice, stack, stack_slice, PadMode, ViewMut};

/// Plumbing of `#[derive(Fields)]`; not part of the public API.
#[doc(hidden)]
pub mod __private {
    pub use crate::fields::FieldOp;
    pub use crate::ops::Rhs;
}

/// The `wgpu` crate luma is built on. wgpu types appear in the public API ([`LumaError`]
/// variants, [`Context::features`]); naming them through this re-export guarantees the same
/// version.
pub use wgpu;

/// The `encase` crate [`uniform`] values are written with; deriving `encase::ShaderType` on a
/// struct makes it a [`UniformValue`]. Naming it through this re-export guarantees the same
/// version.
pub use encase;
