pub mod hal;
pub mod kir;

#[allow(deprecated)]
pub use hal::{
    Backend, BackendError, BackendKind, BindingInfo, BindingKind, Buffer, BufferUsage, TextureCaps, TextureDesc, TextureFilter, TextureFormat,
    COPY_BUFFER_ALIGNMENT, CompiledKernel, CostModel, MemoryModel, TransferModel, DeviceInfo, DeviceKind, DeviceLimits,
    Decision, DecisionSite, DiskCache, Dispatch, ExecGraph, Features, FreeUploads, FusionSplit, GraphLaunch, KernelTiming, MemoryNow, KernelTrust,
    LineSolveSpec, MatmulSpec, GemmSpec, triangle_tile, GemmLanes, NativeBuffer, NativeKernel, Profile, RawArg, RawKernel, NativeStage, Stage, StageDir, Ticket, SplitReason, TuningProfile, Verdict,
};
pub use naga;
