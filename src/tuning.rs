//! Kernel parameters per GPU family.
//!
//! Workgroup sizes and tile shapes that suit one GPU can cost another a lot (Apple GPUs, for
//! example, lose occupancy sharply above about 8 KiB of workgroup memory per workgroup). A
//! [`Tuning`] is chosen once per context from the adapter the device was opened on; the kernel
//! generator reads it instead of fixed constants. `LUMA_TUNING=apple`, `LUMA_TUNING=nvidia`,
//! `LUMA_TUNING=cpu` or `LUMA_TUNING=generic` in the environment overrides the choice for new
//! contexts.

/// Environment variable forcing a profile.
const TUNING_ENV: &str = "LUMA_TUNING";
/// `LUMA_SUBGROUPS=off` keeps the built-in kernels off subgroup operations.
const SUBGROUPS_ENV: &str = "LUMA_SUBGROUPS";
/// `LUMA_FUSION_MAX_LIVE=n` overrides [`Tuning::max_live`] (clamped to `8..=4096`).
const MAX_LIVE_ENV: &str = "LUMA_FUSION_MAX_LIVE";
/// `LUMA_FUSION_MAX_INLINED=n` overrides [`Tuning::max_inlined`] (clamped to `16..=16384`).
const MAX_INLINED_ENV: &str = "LUMA_FUSION_MAX_INLINED";
/// `LUMA_STAGING=off` keeps sub-programs read through views out of workgroup memory (see `jit::staging`).
const STAGING_ENV: &str = "LUMA_STAGING";

/// The device family a tuning profile was measured on.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum TuningProfile {
    /// Apple GPUs (Metal), measured on an M5 Pro.
    Apple,
    /// NVIDIA GPUs, measured on an RTX 3090 (Vulkan).
    Nvidia,
    /// Every other GPU: common defaults, not measured yet.
    Generic,
    /// Software rasterizers on the CPU (Mesa llvmpipe), measured on a Ryzen 9 7950X.
    Cpu,
}

/// Kernel parameters of a context.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct Tuning {
    pub(crate) profile: TuningProfile,
    /// Invocations per workgroup of elementwise kernels (fused and fills).
    pub(crate) elementwise_workgroup: u32,
    /// Tiled 1D convolution: invocations per workgroup, and outputs per invocation at stride 1.
    pub(crate) conv_workgroup: u32,
    pub(crate) conv_per_thread: u32,
    /// Time-blocked stencils: workgroup memory for the two tile buffers, and the output tile of
    /// 2D and 1D arrays.
    pub(crate) stencil_tile_bytes: u32,
    pub(crate) stencil_tile_2d: (u32, u32),
    pub(crate) stencil_tile_1d: u32,
    /// Elements each reduction invocation loads (and combines pairwise) per block.
    pub(crate) reduce_per_thread: u32,
    /// Most workgroups of a reduction pass; they stride over arrays larger than that many blocks.
    pub(crate) reduce_max_groups: u32,
    /// Whether reductions combine within subgroups (the device has `Features::SUBGROUP`).
    pub(crate) subgroups: bool,
    /// Most operations of a lazy expression a shift recomputes at the shifted element instead of evaluating it first.
    pub(crate) max_recomputed: usize,
    /// Most values a fused kernel keeps live at once before it is split in two (register pressure).
    pub(crate) max_live: usize,
    /// Most instructions of a kernel that reads sub-programs through views once they are written
    /// out at every index they are read at (see `jit::inline`); a bigger one is cut by evaluating
    /// one of its sub-programs first.
    pub(crate) max_inlined: usize,
    /// Whether sub-programs read through translation views at several offsets may be computed once
    /// per workgroup tile in workgroup memory (see `jit::staging`); `LUMA_STAGING=off` clears it.
    pub(crate) staging: bool,
}

impl Tuning {
    const APPLE: Tuning = Tuning {
        profile: TuningProfile::Apple,
        elementwise_workgroup: 64,
        conv_workgroup: 128,
        conv_per_thread: 32,
        stencil_tile_bytes: 8 * 1024,
        stencil_tile_2d: (32, 16),
        // A wider 1D tile leaves no room in 8 KiB for the halo of a second step.
        stencil_tile_1d: 512,
        reduce_per_thread: 4,
        reduce_max_groups: 1024,
        subgroups: false,
        max_recomputed: 24,
        max_live: 64,
        max_inlined: 1024,
        staging: true,
    };

    const NVIDIA: Tuning = Tuning {
        profile: TuningProfile::Nvidia,
        elementwise_workgroup: 256,
        conv_workgroup: 256,
        conv_per_thread: 32,
        stencil_tile_bytes: 32 * 1024,
        stencil_tile_2d: (64, 32),
        stencil_tile_1d: 2048,
        reduce_per_thread: 8,
        reduce_max_groups: 2048,
        subgroups: false,
        max_recomputed: 32,
        max_live: 96,
        max_inlined: 1536,
        staging: true,
    };

    const GENERIC: Tuning = Tuning {
        profile: TuningProfile::Generic,
        elementwise_workgroup: 256,
        conv_workgroup: 256,
        conv_per_thread: 16,
        stencil_tile_bytes: 16 * 1024,
        stencil_tile_2d: (32, 16),
        stencil_tile_1d: 1024,
        reduce_per_thread: 4,
        reduce_max_groups: 1024,
        subgroups: false,
        max_recomputed: 24,
        max_live: 96,
        max_inlined: 1024,
        staging: true,
    };

    /// Software rasterizers: the generic parameters with the smaller elementwise workgroups
    /// that measured faster on llvmpipe.
    const CPU: Tuning = Tuning { profile: TuningProfile::Cpu, elementwise_workgroup: 64, max_live: 256, max_inlined: 192, ..Tuning::GENERIC };

    /// The parameters for the GPU `info` describes, unless the environment forces a profile.
    pub(crate) fn for_adapter(info: &wgpu::AdapterInfo, limits: &wgpu::Limits, features: wgpu::Features) -> Tuning {
        let forced = std::env::var(TUNING_ENV).ok();
        let profile = match forced.as_deref().map(str::trim) {
            Some("apple") => TuningProfile::Apple,
            Some("nvidia") => TuningProfile::Nvidia,
            Some("generic") => TuningProfile::Generic,
            Some("cpu") => TuningProfile::Cpu,
            _ if info.device_type == wgpu::DeviceType::Cpu => TuningProfile::Cpu,
            _ if is_apple(info) => TuningProfile::Apple,
            _ if info.vendor == NVIDIA_VENDOR => TuningProfile::Nvidia,
            _ => TuningProfile::Generic,
        };
        let tuning = match profile {
            TuningProfile::Apple => Tuning::APPLE,
            TuningProfile::Nvidia => Tuning::NVIDIA,
            TuningProfile::Generic => Tuning::GENERIC,
            TuningProfile::Cpu => Tuning::CPU,
        };
        let mut tuning = tuning.within(limits);
        if let Some(live) = std::env::var(MAX_LIVE_ENV).ok().and_then(|value| value.trim().parse::<usize>().ok()) {
            tuning.max_live = live.clamp(8, 4096);
        }
        if let Some(inlined) = std::env::var(MAX_INLINED_ENV).ok().and_then(|value| value.trim().parse::<usize>().ok()) {
            tuning.max_inlined = inlined.clamp(16, 16384);
        }
        tuning.subgroups = features.contains(wgpu::Features::SUBGROUP) && std::env::var(SUBGROUPS_ENV).as_deref() != Ok("off");
        tuning.staging = std::env::var(STAGING_ENV).as_deref().map(str::trim) != Ok("off");
        tuning
    }

    /// The generic parameters, for tests that need a `Tuning` without a device.
    #[cfg(test)]
    pub(crate) fn for_tests() -> Tuning {
        Tuning::GENERIC
    }

    /// Shrinks what the device cannot run.
    fn within(mut self, limits: &wgpu::Limits) -> Tuning {
        let invocations = limits.max_compute_invocations_per_workgroup.min(limits.max_compute_workgroup_size_x);
        self.elementwise_workgroup = self.elementwise_workgroup.min(invocations);
        self.conv_workgroup = self.conv_workgroup.min(invocations);
        while self.conv_per_thread > 4 && conv_workgroup_bytes(self.conv_workgroup, self.conv_per_thread) > limits.max_compute_workgroup_storage_size {
            self.conv_per_thread /= 2;
        }
        self.stencil_tile_bytes = self.stencil_tile_bytes.min(limits.max_compute_workgroup_storage_size);
        // `stencil::layout` always runs at least one step, whatever the budget.
        while self.stencil_tile_1d > 32 && stencil_bytes((self.stencil_tile_1d, 1), (STENCIL_RADIUS, 0)) > self.stencil_tile_bytes {
            self.stencil_tile_1d /= 2;
        }
        let radius = (STENCIL_RADIUS, STENCIL_RADIUS);
        while self.stencil_tile_2d.0.max(self.stencil_tile_2d.1) > 4 && stencil_bytes(self.stencil_tile_2d, radius) > self.stencil_tile_bytes {
            let (x, y) = &mut self.stencil_tile_2d;
            if *x >= *y { *x /= 2 } else { *y /= 2 }
        }
        self
    }

    /// Elements one reduction workgroup consumes at a time.
    pub(crate) fn reduce_block(&self) -> u32 {
        crate::builtin::REDUCE_WORKGROUP * self.reduce_per_thread
    }

    /// Workgroups of a reduction pass over `n` elements.
    pub(crate) fn reduce_groups(&self, n: u32) -> u32 {
        n.div_ceil(self.reduce_block()).min(self.reduce_max_groups)
    }
}

/// Workgroup memory of the tiled convolution at stride 1: the input window (with one skew word
/// per 32) and the weights of a chunk (see `builtin::conv1d_tiled`).
fn conv_workgroup_bytes(workgroup: u32, per_thread: u32) -> u32 {
    let span = workgroup * per_thread + workgroup;
    (span + span / 32 + 1 + workgroup) * 4
}

/// Largest stencil radius (per axis) time-blocked; wider stencils iterate one kernel per step.
pub(crate) const STENCIL_RADIUS: u32 = 4;

/// Workgroup memory of the two buffers of a stencil tile of `tile` outputs with a halo of
/// `halo` cells per side and axis.
pub(crate) fn stencil_bytes(tile: (u32, u32), halo: (u32, u32)) -> u32 {
    (tile.0 + 2 * halo.0) * (tile.1 + 2 * halo.1) * 8
}

/// Apple's PCI vendor id, which wgpu reports for Apple GPUs on Metal.
const APPLE_VENDOR: u32 = 0x106B;
const NVIDIA_VENDOR: u32 = 0x10DE;

fn is_apple(info: &wgpu::AdapterInfo) -> bool {
    info.backend == wgpu::Backend::Metal && (info.vendor == APPLE_VENDOR || info.name.starts_with("Apple"))
}

#[cfg(test)]
mod tests {
    use super::{conv_workgroup_bytes, stencil_bytes, Tuning, STENCIL_RADIUS};

    #[test]
    fn tunings_fit_the_smallest_webgpu_limits() {
        let limits = wgpu::Limits::downlevel_defaults();
        for tuning in [Tuning::APPLE, Tuning::NVIDIA, Tuning::GENERIC, Tuning::CPU] {
            let t = tuning.within(&limits);
            assert!(conv_workgroup_bytes(t.conv_workgroup, t.conv_per_thread) <= limits.max_compute_workgroup_storage_size, "{t:?}");
            assert!(t.stencil_tile_bytes <= limits.max_compute_workgroup_storage_size);
            assert!(stencil_bytes((t.stencil_tile_1d, 1), (STENCIL_RADIUS, 0)) <= t.stencil_tile_bytes, "{t:?}");
            assert!(stencil_bytes(t.stencil_tile_2d, (STENCIL_RADIUS, STENCIL_RADIUS)) <= t.stencil_tile_bytes, "{t:?}");
            assert!(t.elementwise_workgroup <= limits.max_compute_invocations_per_workgroup);
        }
    }
}
