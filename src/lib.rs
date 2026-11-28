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
//! * **Backends**: with [`BackendKind::Auto`] a context opens CUDA, then Metal, then wgpu (Vulkan,
//!   Metal, DX12); [`ContextOptions::backend`], the `Context::cuda` / `metal` / `wgpu` / `cpu`
//!   constructors and `LUMA_BACKEND` choose one, and [`Context::backend`] says which opened.
//! * **CPU fallback**: without a GPU the context opens the native `cpu` backend (SIMD code on
//!   the host's threads); `LUMA_DEVICE=cpu` or [`Context::cpu`] force it. Software Vulkan
//!   adapters such as llvmpipe are never used. [`Context::device`] says which device runs. See [`DeviceKind`].
//! * **Arrays larger than a buffer**: an array that exceeds the device's buffer limit is split
//!   over several buffers and the built-in operations run over them in pieces. Array data
//!   beyond the device's memory budget moves to host memory and back on demand
//!   ([`ContextOptions`], [`Context::spilled_bytes`]). Custom kernels need arrays that fit one
//!   buffer ([`LumaError::Chunked`]).
//! * **Arrays larger than a buffer, or than memory**: [`Array::open`] and [`Array::create`] make an
//!   array whose data is a file (raw, with a sidecar, or `.npy`): parts are read when a kernel
//!   needs them and dropped or written back when the device needs the room ([`FileOptions`],
//!   [`Array::store`], [`Array::write_to`], [`Array::flush`]); see the guide chapter "File-backed arrays".
//! * **Errors do not panic**: operators and math methods return poisoned arrays on failure; the
//!   error (with the source location of the failed operation) comes out as `Err` at
//!   [`Array::to_vec`], [`Array::item`], [`Array::eval`], ... See [`Array`].
//! * **Batching**: operations are recorded and submitted together, see [`Context`];
//!   [`Context::flush`] hands the batch to the GPU without waiting for it.
//! * **Custom kernels**: [`Kernel`] runs your own WGSL on arrays, in the same batch.
//! * **GPU timing**: [`Context::profile`] measures the GPU time per kernel where the adapter
//!   supports timestamp queries.
//! * **Incremental accumulation**: [`Accumulator`], [`CoMoments`], [`Covariance`], [`Histogram`],
//!   [`Histogram2d`] and [`TopK`] fold batches that arrive over time into mergeable states on the
//!   GPU; see the guide chapter "Incremental accumulation".
//! * **Linear algebra and signals**: [`C32`] arrays with FFTs of any length over any axes,
//!   [`Sparse`] matrices (CSR, CSC, COO) with sparse-dense products, dense `f32` factorizations and
//!   solves ([`Lu`], [`Qr`], [`Svd`], [`Eigh`], Cholesky, least squares), interpolation and warps
//!   ([`Interp`]), and FIR / IIR filters.
//! * **Images and vision**: convolutions and rank filters ([`ConvSpec`], [`Border`]), colour spaces
//!   ([`ColorSpace`]), and feature detection and matching (`fast`, `harris`, `orb`, `sift`, `canny`,
//!   `match_template`, connected components; points come back as [`Keypoints`]).
//! * **Element types**: `f32`, `u32`, `i32`, `bool`, compact integers (`u8`, `i8`, `u16`, `i16`),
//!   two-byte floats (`f16`, `bf16`: storage plus `f32` compute), [`Df32`] (double-float) and `f64`
//!   where the device has it.
//! * **Matrix products**: [`Array::matmul`] runs on tensor cores in TF32 on CUDA devices of compute
//!   capability 8.0 and up (results differ from the exact product by TF32 rounding);
//!   [`ContextOptions::numerics`] with [`Numerics::strict`] or `LUMA_NUMERICS=strict` turns that off for a context and
//!   [`Array::matmul_f32`] is the exact product of one call.

// wgpu's handle types nest deeply; proving `Context: Sync` for the global default context
// exceeds the default limit on recent nightlies (future-compat lint
// `recursion_depth_exceeding_limit`).
#![recursion_limit = "256"]

/// Compiles and runs the code blocks of the user guide (`guide/src/*.md`) as doctests.
#[cfg(doctest)]
mod guide {
    #[doc = include_str!("../guide/src/introduction.md")]
    pub mod introduction {}
    #[doc = include_str!("../guide/src/arrays.md")]
    pub mod arrays {}
    #[doc = include_str!("../guide/src/lazy.md")]
    pub mod lazy {}
    #[doc = include_str!("../guide/src/views.md")]
    pub mod views {}
    #[doc = include_str!("../guide/src/reductions.md")]
    pub mod reductions {}
    #[doc = include_str!("../guide/src/data.md")]
    pub mod data {}
    #[doc = include_str!("../guide/src/fft.md")]
    pub mod fft {}
    #[doc = include_str!("../guide/src/sparse.md")]
    pub mod sparse {}
    #[doc = include_str!("../guide/src/linear-algebra.md")]
    pub mod linear_algebra {}
    #[doc = include_str!("../guide/src/signal.md")]
    pub mod signal {}
    #[doc = include_str!("../guide/src/records.md")]
    pub mod records {}
    #[doc = include_str!("../guide/src/loops.md")]
    pub mod loops {}
    #[doc = include_str!("../guide/src/precision.md")]
    pub mod precision {}
    #[doc = include_str!("../guide/src/complex.md")]
    pub mod complex {}
    #[doc = include_str!("../guide/src/integers.md")]
    pub mod integers {}
    #[doc = include_str!("../guide/src/compact.md")]
    pub mod compact {}
    #[doc = include_str!("../guide/src/half.md")]
    pub mod half_floats {}
    #[doc = include_str!("../guide/src/custom-kernels.md")]
    pub mod custom_kernels {}
    #[doc = include_str!("../guide/src/performance.md")]
    pub mod performance {}
    #[doc = include_str!("../guide/src/examples.md")]
    pub mod examples {}
    #[doc = include_str!("../guide/src/images.md")]
    pub mod images {}
    #[doc = include_str!("../guide/src/vision.md")]
    pub mod vision {}
    #[doc = include_str!("../guide/src/vision-corners.md")]
    pub mod vision_corners {}
    #[doc = include_str!("../guide/src/vision-matching.md")]
    pub mod vision_matching {}
    #[doc = include_str!("../guide/src/vision-edges.md")]
    pub mod vision_edges {}
    #[doc = include_str!("../guide/src/vision-regions.md")]
    pub mod vision_regions {}
    #[doc = include_str!("../guide/src/vision-descriptors.md")]
    pub mod vision_descriptors {}
    #[doc = include_str!("../guide/src/accumulators.md")]
    pub mod accumulators {}
    #[doc = include_str!("../guide/src/files.md")]
    pub mod files {}
    #[doc = include_str!("../guide/src/environment.md")]
    pub mod environment {}
    #[doc = include_str!("../guide/src/faq.md")]
    pub mod faq {}
    #[doc = include_str!("../guide/src/migration.md")]
    pub mod migration {}
}

// The derive emits `::luma::` paths, also inside this crate.
extern crate self as luma;

mod accum;
mod arg;
mod array;
mod builtin;
mod caps;
mod bykey;
mod context;
mod conv;
mod cyclic;
mod dense;
mod solve;
mod erased;
pub mod ext;
pub use ext::{elementwise_fn, ElemFnType, ElementwiseFn};
mod error;
mod fields;
mod fields_macro;
#[cfg(test)]
mod fuzz;
mod fft;
mod factorize;
mod file;
mod filter;
mod generate;
mod gpu;
mod complex;
mod dft_kernel;
mod einsum;
mod gpu_type;
mod group;
mod halves;
mod host;
mod image;
mod interp;
mod iterate;
mod jit;
mod kernel;
mod linalg;
mod bmm;
mod matmul;
mod nan;
mod numerics;
mod ops;
mod precision;
mod profile;
mod quantile;
mod reduce;
mod scan;
mod scatter;
mod sampler;
mod search;
mod signal;
mod segments;
mod segsort;
mod small;
mod setmerge;
mod setops;
mod scalar;
mod shape;
mod sort;
mod props;
mod structure;
mod stats;
mod spectral;
mod sparse;
mod texture;
// The replay engine behind compiled callables: internal until the library drives it by itself.
#[allow(dead_code)]
mod trace;
#[cfg(test)]
mod trace_tests;
mod tuning;
mod until;
mod view;
mod vision;

pub use accum::{AccumCore, AccumInput, Accumulator, AccumulatorBuilder, CoMoments, CoMomentsBuilder, Covariance, Fold, Histogram, Histogram2d, NanPolicy, TopK, Wide};
pub use array::Array;
pub use context::{default_context, flush, Context, ContextOptions, DeviceSummary};
pub use gpu::pager::{MemoryStats, PagingStats};
#[doc(hidden)]
pub use jit::split::{SplitMode, StreamStats};
pub use numerics::{Numerics, Reduced};
#[allow(deprecated)]
pub use luma_core::{BackendKind, BindingInfo, BindingKind, CostModel, Decision, DecisionSite, DeviceKind, MemoryModel, Features, FusionSplit, KernelTiming, Profile, SplitReason, TuningProfile, Verdict};
pub use luma_core::hal::HostBytes;
pub use error::{LumaError, Result};
pub use fields::Fields;
pub use file::FileOptions;
pub use complex::C32;
pub use spectral::{ConvMode, FftNorm};
pub use gpu_type::{Df32, GpuType, Numeric, ScalarType};
pub use half::{bf16, f16};
pub use kernel::{uniform, view, Kernel, KernelArg, KernelBuilder, ProblemSize, Reach, UniformValue};
pub use erased::{AnyArray, Dyn, ElemKind};
pub use iterate::iterate_all;
pub use image::{transform_coordinates, transform_coordinates_on, TransformMatrix};
pub use interp::Interp;
pub use signal::IIR_MAX_ORDER;
pub use until::{compile_until_all, iterate_until_all, CompiledUntil, Cond, History, Run, Series, Stat, StopReason, Until};
pub use linalg::{
    solve_block_tridiagonal, solve_block_tridiagonal_from, solve_block_tridiagonal_parts, solve_block_tridiagonal_parts_from, solve_tridiagonal, solve_tridiagonal_from,
    solve_tridiagonal_parts, solve_tridiagonal_parts_from,
};
pub use bmm::matmul_batched;
pub use einsum::einsum;
pub use dense::lu::Lu;
pub use dense::eigh::Eigh;
pub use dense::svd::Svd;
pub use dense::qr::Qr;
pub use dense::refine::Refined;
pub use dense::Triangle;
pub use matmul::{dot, matmul, matmul_f32, matmul_nt, matmul_tf32, matmul_tn, matmul_tt};
pub use jit::fusion_limit;
pub use ops::{eval_all, select, Condition, Operand};
pub use reduce::Reduction;
pub use scalar::Scalar;
pub use scatter::{bincount, AtomicElem};
pub use search::Side;
pub use shape::{Offsets, Shape};
pub use sort::SortKey;
pub use stats::{Bias, Norm};
pub use sparse::{sparse, Sparse, SparseElem, SparseFormat, SparseInfo};
pub use view::{concat, concat_slice, stack, stack_slice, PadMode, ViewMut};
pub use vision::*;
pub use image::{gaussian_kernel, gaussian_kernel_1d, gaussian_kernel_1d_on, gaussian_kernel_on, Border, ColorSpace, ConvSpec, ImagePixel, Padding, YccStd};

/// Plumbing of `fields!`; not part of the public API.
#[doc(hidden)]
pub mod __private {
    pub use crate::fields::FieldOp;
    pub use crate::ops::Rhs;
}

/// The `wgpu` crate luma's wgpu backend is built on (with the `wgpu` feature); naming it through
/// this re-export guarantees the same version.
#[cfg(feature = "wgpu")]
pub use luma_wgpu::wgpu;

/// The `encase` crate [`uniform`] values are written with; deriving `encase::ShaderType` on a
/// struct makes it a [`UniformValue`]. Naming it through this re-export guarantees the same
/// version.
pub use encase;
