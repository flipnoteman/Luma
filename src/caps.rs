//! What the device can do, as one bit set computed when a context opens.
//!
//! Capability tests read [`Caps`] instead of repeating the questions of the HAL: `Features` bits,
//! the tuning profile, the backend's opt-ins and the memory model. [`Tuning`](crate::tuning::Tuning)
//! flags stay as they are: they are tuning, not capability.

use luma_core::{Backend, DeviceInfo, Features, TuningProfile};

use crate::tuning::Tuning;

/// A set of device capabilities.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Caps(u32);

#[allow(dead_code)]
impl Caps {
    /// Subgroup operations.
    pub(crate) const SUBGROUP: Self = Self(1 << 0);
    /// Subgroups of exactly 32 lanes.
    pub(crate) const SUBGROUP32: Self = Self(1 << 1);
    /// 16-bit floats in shaders.
    pub(crate) const F16: Self = Self(1 << 2);
    /// 64-bit floats in shaders.
    pub(crate) const F64: Self = Self(1 << 3);
    /// Cooperative matrix operations.
    pub(crate) const COOP_MATRIX: Self = Self(1 << 4);
    /// Texture reads of storage data.
    pub(crate) const TEXTURE: Self = Self(1 << 5);
    /// Texture reads pay off on this device (measured: the NVIDIA profile).
    pub(crate) const TEXTURE_PROFITABLE: Self = Self(1 << 6);
    /// A native matrix product.
    pub(crate) const NATIVE_GEMM: Self = Self(1 << 7);
    /// A native line solve.
    pub(crate) const NATIVE_LINE_SOLVE: Self = Self(1 << 8);
    /// A native batched matrix product (offered with the native matrix product).
    pub(crate) const NATIVE_BMM: Self = Self(1 << 9);
    /// Single-pass scans with decoupled look-back.
    pub(crate) const LOOKBACK: Self = Self(1 << 10);
    /// Tensor-core products in TF32 (the backend has TF32 `gemm` tiles).
    pub(crate) const TF32: Self = Self(1 << 11);
    /// Host and device share memory.
    pub(crate) const UNIFIED_MEMORY: Self = Self(1 << 12);
    /// The static part of the host-path condition: a GPU backend on unified memory, where host
    /// execution of small work beats a launch. Not `Context::can_run_on_host`, which is dynamic.
    pub(crate) const HOST_PATH: Self = Self(1 << 13);
    /// The backend can time single launches without profiling mode (`Backend::can_time_launches`).
    pub(crate) const TELEMETRY: Self = Self(1 << 14);

    /// Whether every capability of `other` is set.
    pub(crate) const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }

    /// `self` and the capabilities of `other`.
    pub(crate) const fn with(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    /// No capability.
    pub(crate) const fn empty() -> Self {
        Self(0)
    }

    /// `self` without the capabilities of `other`.
    pub(crate) const fn without(self, other: Self) -> Self {
        Self(self.0 & !other.0)
    }

    /// Computes the capabilities of a device.
    pub(crate) fn of(info: &DeviceInfo, backend: &dyn Backend, tuning: &Tuning) -> Caps {
        let mut caps = Caps(0);
        let mut set = |on: bool, bit: Caps| {
            if on {
                caps.0 |= bit.0;
            }
        };
        let f = info.features;
        let texture = backend.texture_caps().is_some();
        set(f.contains(Features::SUBGROUP), Caps::SUBGROUP);
        set(tuning.subgroup32, Caps::SUBGROUP32);
        set(f.contains(Features::SHADER_F16), Caps::F16);
        set(f.contains(Features::SHADER_F64), Caps::F64);
        set(f.contains(Features::EXPERIMENTAL_COOPERATIVE_MATRIX), Caps::COOP_MATRIX);
        set(texture, Caps::TEXTURE);
        set(texture && tuning.profile == TuningProfile::Nvidia, Caps::TEXTURE_PROFITABLE);
        set(backend.native_matmul().is_some(), Caps::NATIVE_GEMM);
        set(backend.native_line_solve(), Caps::NATIVE_LINE_SOLVE);
        set(backend.native_matmul().is_some(), Caps::NATIVE_BMM);
        set(tuning.lookback_scan, Caps::LOOKBACK);
        set(!backend.gemm_tiles(false).is_empty(), Caps::TF32);
        set(info.zero_copy, Caps::UNIFIED_MEMORY);
        set(!info.cpu && info.zero_copy, Caps::HOST_PATH);
        set(backend.can_time_launches(), Caps::TELEMETRY);
        caps
    }
}
