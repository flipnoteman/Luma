//! Native SIMD CPU backend (`luma-cpu`).

pub mod abi;
pub mod backend;
pub mod codegen;
pub mod exec;
pub mod fission;
pub mod gemm;
pub mod solve;
#[cfg(all(target_arch = "x86_64", unix))]
pub mod vecmath;

pub use backend::{CpuBackend, open};
