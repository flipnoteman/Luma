//! The CUDA backend of luma: CUDA C++ code generation for the kernel IR (compiled with NVRTC)
//! and a [`Backend`] that runs on one ordered stream and replays CUDA graphs.

pub mod backend;
pub mod codegen;

pub use backend::CudaBackend;

use luma_core::{Backend, BackendError};

/// Opens the first CUDA device (taking the process-wide shared one when another context holds
/// it) and returns the backend running on it. `Err(NoAdapter)` without an NVIDIA driver, NVRTC
/// or device.
pub fn open() -> Result<Box<dyn Backend>, BackendError> {
    CudaBackend::new().map(|backend| Box::new(backend) as Box<dyn Backend>)
}
