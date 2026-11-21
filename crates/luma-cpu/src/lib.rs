//! Native SIMD CPU backend (`luma-cpu`).

pub mod abi;
pub mod backend;
pub mod codegen;
pub mod exec;
pub mod fission;
pub mod solve;

pub use backend::{CpuBackend, open};
