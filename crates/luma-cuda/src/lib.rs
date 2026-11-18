//! The CUDA backend of luma: NVVM IR code generation for the kernel IR (compiled with the NVIDIA
//! driver's NVVM, linking NVIDIA's libdevice)
//! and a [`Backend`] that runs on one ordered stream and replays CUDA graphs.

pub mod backend;
pub mod codegen;
mod nvvm;

pub use backend::CudaBackend;

use luma_core::{Backend, BackendError};

/// Opens the first CUDA device (taking the process-wide shared one when another context holds
/// it) and returns the backend running on it. `Err(NoAdapter)` without an NVIDIA driver, a usable
/// driver NVVM or device.
pub fn open() -> Result<Box<dyn Backend>, BackendError> {
    CudaBackend::new().map(|backend| Box::new(backend) as Box<dyn Backend>)
}
