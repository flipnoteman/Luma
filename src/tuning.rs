//! Kernel parameters per GPU family.
//!
//! Workgroup sizes and tile shapes that suit one GPU can cost another a lot (Apple GPUs, for
//! example, lose occupancy sharply above about 8 KiB of workgroup memory per workgroup). A
//! [`Tuning`] is chosen once per context from the adapter the device was opened on; the kernel
//! generators read it instead of fixed constants. `LUMA_TUNING=apple`, `LUMA_TUNING=nvidia` or
//! `LUMA_TUNING=generic` in the environment overrides the choice for new contexts.

/// Environment variable forcing a profile.
const TUNING_ENV: &str = "LUMA_TUNING";

/// The GPU family a [`Tuning`] was measured on.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TuningProfile {
    /// Apple GPUs (Metal), measured on an M5 Pro.
    Apple,
    /// NVIDIA GPUs, measured on an RTX 3090 (Vulkan).
    Nvidia,
    /// Every other GPU: common defaults, not measured yet.
    Generic,
}

/// Kernel lengths below these use the first, then the second entry of
/// [`Tuning::conv_per_thread`]; longer kernels the third.
const CONV_TAP_CLASSES: [usize; 2] = [128, 512];

/// Kernel parameters of a context.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct Tuning {
    pub(crate) profile: TuningProfile,
    /// Invocations per workgroup of elementwise kernels (fused, fills, in-place operators).
    pub(crate) elementwise_workgroup: u32,
    /// Tiled 1D convolution: invocations per workgroup, and outputs per invocation at stride 1
    /// by kernel length (see [`CONV_TAP_CLASSES`]): short kernels gain from more invocations,
    /// long ones from reusing each loaded input more often.
    pub(crate) conv_workgroup: u32,
    pub(crate) conv_per_thread: [u32; 3],
    /// Time-blocked stencils: workgroup memory for the two tile buffers, and the output tile of
    /// 2D and 1D arrays.
    pub(crate) stencil_tile_bytes: u32,
    pub(crate) stencil_tile_2d: (u32, u32),
    pub(crate) stencil_tile_1d: u32,
    /// Elements each reduction invocation loads (and combines pairwise) before the workgroup
    /// reduces them.
    pub(crate) reduce_per_thread: u32,
}

impl Tuning {
    const APPLE: Tuning = Tuning {
        profile: TuningProfile::Apple,
        elementwise_workgroup: 64,
        conv_workgroup: 256,
        conv_per_thread: [16, 16, 16],
        stencil_tile_bytes: 8 * 1024,
        stencil_tile_2d: (32, 16),
        // A wider 1D tile leaves no room in 8 KiB for the halo of a second step.
        stencil_tile_1d: 512,
        reduce_per_thread: 4,
    };

    const NVIDIA: Tuning = Tuning {
        profile: TuningProfile::Nvidia,
        elementwise_workgroup: 256,
        conv_workgroup: 256,
        conv_per_thread: [8, 16, 32],
        stencil_tile_bytes: 32 * 1024,
        stencil_tile_2d: (64, 32),
        stencil_tile_1d: 2048,
        reduce_per_thread: 8,
    };

    const GENERIC: Tuning = Tuning {
        profile: TuningProfile::Generic,
        elementwise_workgroup: 256,
        conv_workgroup: 256,
        conv_per_thread: [16, 16, 16],
        stencil_tile_bytes: 16 * 1024,
        stencil_tile_2d: (32, 16),
        stencil_tile_1d: 1024,
        reduce_per_thread: 4,
    };

    /// The parameters for the GPU `info` describes, unless the environment forces a profile.
    pub(crate) fn for_adapter(info: &wgpu::AdapterInfo, limits: &wgpu::Limits) -> Tuning {
        let forced = std::env::var(TUNING_ENV).ok();
        let profile = match forced.as_deref().map(str::trim) {
            Some("apple") => TuningProfile::Apple,
            Some("nvidia") => TuningProfile::Nvidia,
            Some("generic") => TuningProfile::Generic,
            _ if is_apple(info) => TuningProfile::Apple,
            _ if info.vendor == NVIDIA_VENDOR => TuningProfile::Nvidia,
            _ => TuningProfile::Generic,
        };
        let tuning = match profile {
            TuningProfile::Apple => Tuning::APPLE,
            TuningProfile::Nvidia => Tuning::NVIDIA,
            TuningProfile::Generic => Tuning::GENERIC,
        };
        tuning.within(limits)
    }

    /// Shrinks what the device cannot run.
    fn within(mut self, limits: &wgpu::Limits) -> Tuning {
        let invocations = limits.max_compute_invocations_per_workgroup.min(limits.max_compute_workgroup_size_x);
        self.elementwise_workgroup = self.elementwise_workgroup.min(invocations);
        self.conv_workgroup = self.conv_workgroup.min(invocations);
        for per_thread in &mut self.conv_per_thread {
            while *per_thread > 4 && conv_workgroup_bytes(self.conv_workgroup, *per_thread) > limits.max_compute_workgroup_storage_size {
                *per_thread /= 2;
            }
        }
        self.stencil_tile_bytes = self.stencil_tile_bytes.min(limits.max_compute_workgroup_storage_size);
        self
    }

    /// Outputs per invocation of the tiled convolution at stride 1, for a kernel of `taps` taps.
    pub(crate) fn conv_per_thread(&self, taps: usize) -> u32 {
        let class = CONV_TAP_CLASSES.iter().take_while(|&&limit| taps >= limit).count();
        self.conv_per_thread[class]
    }

    /// Elements one reduction workgroup consumes.
    pub(crate) fn reduce_block(&self) -> u32 {
        crate::builtin::REDUCE_WORKGROUP * self.reduce_per_thread
    }
}

/// Workgroup memory of the tiled convolution at stride 1: the input window (with one skew word
/// per 32) and the weights of a chunk (see `builtin::conv1d_tiled`).
fn conv_workgroup_bytes(workgroup: u32, per_thread: u32) -> u32 {
    let span = workgroup * per_thread + workgroup;
    (span + span / 32 + 1 + workgroup) * 4
}

/// Apple's PCI vendor id, which wgpu reports for Apple GPUs on Metal.
const APPLE_VENDOR: u32 = 0x106B;
const NVIDIA_VENDOR: u32 = 0x10DE;

fn is_apple(info: &wgpu::AdapterInfo) -> bool {
    info.backend == wgpu::Backend::Metal && (info.vendor == APPLE_VENDOR || info.name.starts_with("Apple"))
}

#[cfg(test)]
mod tests {
    use super::{conv_workgroup_bytes, Tuning};

    #[test]
    fn tunings_fit_the_smallest_webgpu_limits() {
        let limits = wgpu::Limits::downlevel_defaults();
        for tuning in [Tuning::APPLE, Tuning::NVIDIA, Tuning::GENERIC] {
            let t = tuning.within(&limits);
            for per_thread in t.conv_per_thread {
                assert!(conv_workgroup_bytes(t.conv_workgroup, per_thread) <= limits.max_compute_workgroup_storage_size, "{t:?}");
            }
            assert!(t.stencil_tile_bytes <= limits.max_compute_workgroup_storage_size);
            assert!(t.elementwise_workgroup <= limits.max_compute_invocations_per_workgroup);
        }
    }
}
