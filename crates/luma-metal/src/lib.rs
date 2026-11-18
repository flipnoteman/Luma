//! The Metal backend of luma: MSL code generation for the kernel IR and a
//! [`Backend`](luma_core::Backend) that batches work into command buffers. The runtime only
//! exists on Apple platforms; elsewhere [`open`] reports that no adapter is available.

pub mod backend;
pub mod codegen;

// `MTLCreateSystemDefaultDevice` needs the WindowServer/GPU connection that CoreGraphics sets up;
// command line binaries do not link it otherwise.
#[cfg(target_vendor = "apple")]
#[link(name = "CoreGraphics", kind = "framework")]
unsafe extern "C" {}

/// Opens the system's default Metal device (taking an idle one from the process-wide pool when
/// there is one) and returns the backend running on it.
pub fn open() -> Result<Box<dyn luma_core::Backend>, luma_core::BackendError> {
    backend::open()
}
