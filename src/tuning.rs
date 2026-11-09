//! Kernel parameters per GPU family.
//!
//! Workgroup sizes and tile shapes that suit one GPU can cost another a lot (Apple GPUs, for
//! example, lose occupancy sharply above about 8 KiB of workgroup memory per workgroup). A
//! [`Tuning`] is chosen once per context from the adapter the device was opened on; the kernel
//! generators read it instead of fixed constants. `LUMA_TUNING=apple` or `LUMA_TUNING=generic`
//! in the environment overrides the choice for new contexts.

/// Environment variable forcing a profile.
const TUNING_ENV: &str = "LUMA_TUNING";

/// The GPU family a [`Tuning`] was measured on.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TuningProfile {
    /// Apple GPUs (Metal), measured on an M5 Pro.
    Apple,
    /// Every other GPU: common defaults, not measured yet.
    Generic,
}

/// Kernel parameters of a context.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct Tuning {
    pub(crate) profile: TuningProfile,
    /// Invocations per workgroup of elementwise kernels (fused, fills, in-place operators).
    pub(crate) elementwise_workgroup: u32,
    /// Tiled 1D convolution: invocations per workgroup, outputs per invocation at stride 1.
    pub(crate) conv_workgroup: u32,
    pub(crate) conv_per_thread: u32,
    /// Time-blocked stencils: workgroup memory for the two tile buffers, and the output tile of
    /// 2D and 1D arrays.
    pub(crate) stencil_tile_bytes: u32,
    pub(crate) stencil_tile_2d: (u32, u32),
    pub(crate) stencil_tile_1d: u32,
}

impl Tuning {
    const APPLE: Tuning = Tuning {
        profile: TuningProfile::Apple,
        elementwise_workgroup: 64,
        conv_workgroup: 256,
        conv_per_thread: 16,
        stencil_tile_bytes: 8 * 1024,
        stencil_tile_2d: (32, 16),
        stencil_tile_1d: 1024,
    };

    const GENERIC: Tuning = Tuning {
        profile: TuningProfile::Generic,
        elementwise_workgroup: 256,
        conv_workgroup: 256,
        conv_per_thread: 16,
        stencil_tile_bytes: 16 * 1024,
        stencil_tile_2d: (32, 16),
        stencil_tile_1d: 1024,
    };

    /// The parameters for the GPU `info` describes, unless the environment forces a profile.
    pub(crate) fn for_adapter(info: &wgpu::AdapterInfo, limits: &wgpu::Limits) -> Tuning {
        let forced = std::env::var(TUNING_ENV).ok();
        let profile = match forced.as_deref().map(str::trim) {
            Some("apple") => TuningProfile::Apple,
            Some("generic") => TuningProfile::Generic,
            _ if is_apple(info) => TuningProfile::Apple,
            _ => TuningProfile::Generic,
        };
        let tuning = match profile {
            TuningProfile::Apple => Tuning::APPLE,
            TuningProfile::Generic => Tuning::GENERIC,
        };
        tuning.within(limits)
    }

    /// Shrinks what the device cannot run.
    fn within(mut self, limits: &wgpu::Limits) -> Tuning {
        let invocations = limits.max_compute_invocations_per_workgroup.min(limits.max_compute_workgroup_size_x);
        self.elementwise_workgroup = self.elementwise_workgroup.min(invocations);
        self.conv_workgroup = self.conv_workgroup.min(invocations);
        while self.conv_per_thread > 4 && self.conv_workgroup_bytes() > limits.max_compute_workgroup_storage_size {
            self.conv_per_thread /= 2;
        }
        self.stencil_tile_bytes = self.stencil_tile_bytes.min(limits.max_compute_workgroup_storage_size);
        self
    }

    /// Workgroup memory of the tiled convolution at stride 1: the input window (with one skew
    /// word per 32) and the weights of a chunk (see `builtin::conv1d_tiled`).
    fn conv_workgroup_bytes(&self) -> u32 {
        let span = self.conv_workgroup * self.conv_per_thread + self.conv_workgroup;
        (span + span / 32 + 1 + self.conv_workgroup) * 4
    }
}

/// Apple's PCI vendor id, which wgpu reports for Apple GPUs on Metal.
const APPLE_VENDOR: u32 = 0x106B;

fn is_apple(info: &wgpu::AdapterInfo) -> bool {
    info.backend == wgpu::Backend::Metal && (info.vendor == APPLE_VENDOR || info.name.starts_with("Apple"))
}

#[cfg(test)]
mod tests {
    use super::Tuning;

    #[test]
    fn tunings_fit_the_smallest_webgpu_limits() {
        let limits = wgpu::Limits::downlevel_defaults();
        for tuning in [Tuning::APPLE, Tuning::GENERIC] {
            let t = tuning.within(&limits);
            assert!(t.conv_workgroup_bytes() <= limits.max_compute_workgroup_storage_size, "{t:?}");
            assert!(t.stencil_tile_bytes <= limits.max_compute_workgroup_storage_size);
            assert!(t.elementwise_workgroup <= limits.max_compute_invocations_per_workgroup);
        }
    }
}
