//! The wgpu backend of luma: a device pool, WGSL code generation for the kernel IR and a
//! [`Backend`] that batches work into single compute passes.

#![recursion_limit = "256"]

pub use wgpu;

pub mod backend;
pub mod codegen;
mod coop;
mod device;
#[cfg(all(target_os = "linux", target_env = "gnu"))]
mod exit;
#[cfg(not(all(target_os = "linux", target_env = "gnu")))]
mod exit {
    pub(crate) fn driver_call<R>(call: impl FnOnce() -> R) -> R {
        call()
    }
}
mod profiler;

pub use backend::WgpuBackend;

use luma_core::{Backend, BackendError, DeviceKind};

/// Opens a wgpu device of `kind` (taking an idle one from the process-wide pool when there is
/// one) and returns the backend running on it.
pub fn open(kind: DeviceKind) -> Result<Box<dyn Backend>, BackendError> {
    WgpuBackend::new(kind).map(|backend| Box::new(backend) as Box<dyn Backend>)
}
