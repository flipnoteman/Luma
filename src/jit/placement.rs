//! The price of running a stage on the host (W3-HOSTCPU §3). Placement itself belongs to W3-U6's
//! residency decision: it calls [`price`] with the stage's bytes and rates and records the answer
//! in its `Schedule` decision (`Pricing::candidates`). Nothing here chooses, caches or logs.

use std::borrow::Cow;

use luma_core::hal::{CostModel, Verdict};

use super::hostexec::CPU_REDUCE_GBS;

/// Where one input of a stage lives.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum InputRes {
    /// In a device buffer: never a host input (no download is added to run on the CPU).
    Device,
    /// On the device with a valid host copy: the host can read it, the device does not need an upload.
    Both,
    /// In host memory (`State::Host`).
    Host,
    /// In a file (`State::Away`): read straight into the host buffer, or into a staging buffer before the upload.
    Away,
    /// Scalars or generators: no bytes.
    Fresh,
}

/// What a stage reads and writes, and how hard it is to compute.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Work {
    /// Bytes read per [`InputRes`] (`Device`, `Both`, `Host`, `Away`).
    pub(crate) in_bytes: [u64; 4],
    pub(crate) out_bytes: u64,
    pub(crate) len: u64,
    /// f32 operations per element.
    pub(crate) weight: f64,
    /// A reduction: the CPU backend runs it at [`CPU_REDUCE_GBS`], a quarter of its elementwise rate.
    pub(crate) reduce: bool,
}

impl Work {
    /// Whether any input is not on the device alone: the precomputed per-step flag of the V5 host-time gate.
    pub(crate) fn any_off_device(&self) -> bool {
        self.in_bytes[1] + self.in_bytes[2] + self.in_bytes[3] > 0
    }

    /// Bytes the host cannot read without a download: a stage with any is never a host stage.
    pub(crate) fn device_only_bytes(&self) -> u64 {
        self.in_bytes[0]
    }
}

/// Rates of the link and of file reads, GB/s.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Rates {
    pub(crate) h2d_gbs: f64,
    pub(crate) d2h_gbs: f64,
    /// Share of the CPU's DRAM bandwidth left while a transfer runs (`MemoryModel.transfer.overlap_derate`).
    pub(crate) overlap_derate: f64,
    /// File read rate for `Away` inputs (W2-P5a's calibration).
    pub(crate) file_gbs: f64,
}

impl Rates {
    /// The priors until W3-U6's `MemoryModel.transfer` is present, for host data that is ordinary (pageable) memory, which is what
    /// `HostBytes` is: measured on the 3090 over PCIe by `jit/hostexec/upload` (9.5 to 11.9 GB/s up) and `_dev_rb` (3.7 to 6 GB/s down).
    /// Pinned memory goes 14 to 25 GB/s (`luma-cuda/src/backend.rs`).
    pub(crate) const PCIE: Rates = Rates { h2d_gbs: 10.0, d2h_gbs: 5.0, overlap_derate: 1.0, file_gbs: 3.0 };
    /// The spec's example rates (pinned transfers).
    #[cfg(test)]
    pub(crate) const PCIE_PINNED: Rates = Rates { h2d_gbs: 20.0, d2h_gbs: 20.0, overlap_derate: 1.0, file_gbs: 3.0 };
    /// Unified memory: nothing crosses a link, so the host never beats the device on bytes.
    pub(crate) const UNIFIED: Rates = Rates { h2d_gbs: f64::INFINITY, d2h_gbs: f64::INFINITY, overlap_derate: 1.0, file_gbs: 3.0 };
}

/// The prior that the result is read back on the host (named so the A/B can move it).
pub(crate) const PHI: f64 = 0.5;
/// `Device` unless `Host` is cheaper by more than this share (no flipping between near-equal placements).
pub(crate) const HYSTERESIS: f64 = 0.10;

/// What the aux CPU context says it costs: its model, and a learned estimate of the kernel's time when it ran before.
#[derive(Clone, Copy, Debug)]
pub(crate) struct HostEstimate {
    pub(crate) cpu: CostModel,
    /// `Backend::learned_ns` of the program, nanoseconds, if known.
    pub(crate) learned_ns: Option<f64>,
}

/// Device and host cost of one stage, and the verdict.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Pricing {
    pub(crate) device_ns: f64,
    pub(crate) host_ns: f64,
    /// `None`: the host may run it; `Some(reason)`: it may not (the `Rejected` label of the decision row).
    pub(crate) refused: Option<&'static str>,
}

impl Pricing {
    /// Whether the host stage wins by more than [`HYSTERESIS`].
    pub(crate) fn host_wins(&self) -> bool {
        self.refused.is_none() && self.host_ns < self.device_ns * (1.0 - HYSTERESIS)
    }

    /// The two candidate rows of U6's `Schedule` decision: `("device", Estimate)` and `("host", Estimate | Rejected)`.
    pub(crate) fn candidates(&self) -> [(Cow<'static, str>, Verdict); 2] {
        [
            ("device".into(), Verdict::Estimate { ns: self.device_ns }),
            ("host".into(), match self.refused {
                Some(reason) => Verdict::Rejected(reason),
                None => Verdict::Estimate { ns: self.host_ns },
            }),
        ]
    }
}

/// Prices stage `w` on the device (`dev` is the device's cost model, `kernel_ns` the device's estimate of the
/// kernel) and on the host. `refused` carries the gates the caller already knows (numerics, host budget, compile
/// readiness, residency); a stage reading bytes only the device has is refused here (`"residency"`).
pub(crate) fn price(dev: &CostModel, kernel_ns: f64, rates: &Rates, w: &Work, host: &HostEstimate, refused: Option<&'static str>) -> Pricing {
    let gb = |bytes: u64, gbs: f64| bytes as f64 / gbs; // bytes / (GB/s) = ns
    let up = w.in_bytes[2] + w.in_bytes[3];
    // Device: upload what is not there (`Away` pays its file read first), launch, kernel, and the result is read back with probability phi.
    let device_ns = gb(up, rates.h2d_gbs) + gb(w.in_bytes[3], rates.file_gbs) + dev.launch_us * 1e3 + kernel_ns + PHI * gb(w.out_bytes, rates.d2h_gbs);
    // Host: launch plus the larger of the memory time and the arithmetic time (or the learned time), the result goes to the device with probability 1 - phi.
    let touched = (w.in_bytes[1] + w.in_bytes[2] + w.in_bytes[3] + w.out_bytes) as f64;
    let cpu = &host.cpu;
    let gbs = if w.reduce { cpu.bandwidth_gbs.min(CPU_REDUCE_GBS) } else { cpu.bandwidth_gbs };
    let kernel = host.learned_ns.unwrap_or_else(|| (touched / gbs).max(w.weight * w.len as f64 / cpu.f32_ops_per_ns));
    let host_ns = cpu.launch_us * 1e3 + kernel / rates.overlap_derate.max(f64::MIN_POSITIVE) + (1.0 - PHI) * gb(w.out_bytes, rates.h2d_gbs) + gb(w.in_bytes[3], rates.file_gbs);
    let refused = refused.or((w.device_only_bytes() > 0).then_some("residency"));
    Pricing { device_ns, host_ns, refused }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GIB: u64 = 1 << 30;

    fn cpu() -> CostModel {
        CostModel::CPU
    }

    fn work(in_host: u64, out: u64, weight: f64) -> Work {
        Work { in_bytes: [0, 0, in_host, 0], out_bytes: out, len: in_host / 4, weight, reduce: out == 4 }
    }

    fn est() -> HostEstimate {
        HostEstimate { cpu: cpu(), learned_ns: None }
    }

    /// Roughly the 3090: 740 GB/s, launch 2.9 us, kernel at bandwidth.
    fn gpu(bytes: u64) -> (CostModel, f64) {
        (CostModel::NVIDIA, bytes as f64 / 740.0)
    }

    #[test]
    fn a_light_map_over_a_gib_of_host_data_is_cheaper_on_the_host() {
        // The spec's example at pinned rates and phi = 1: device 54 ms up + 3 ms + 54 ms down = 111 ms, host 69 ms.
        let (dev, kernel) = gpu(2 * GIB);
        let p = price(&dev, kernel, &Rates::PCIE_PINNED, &work(GIB, GIB, 2.0), &est(), None);
        assert!((p.device_ns / 1e6 - (53.7 + 2.9 + 0.5 * 53.7)).abs() < 5.0, "{p:?}");
        assert!((p.host_ns / 1e6 - (69.3 + 0.5 * 53.7)).abs() < 5.0, "{p:?}");
        // At the rates measured for pageable memory the host wins by more.
        let q = price(&dev, kernel, &Rates::PCIE, &work(GIB, GIB, 2.0), &est(), None);
        assert!(q.host_wins() && q.host_ns < 0.7 * q.device_ns, "{q:?}");
    }

    #[test]
    fn a_sum_of_a_gib_stays_on_the_device_because_the_cpu_reduction_is_slow() {
        // Measured: the CPU backend reduces at about 8 GB/s, so the host takes 134 ms against 107 ms of upload.
        let (dev, kernel) = gpu(GIB);
        let p = price(&dev, kernel, &Rates::PCIE, &work(GIB, 4, 2.0), &est(), None);
        assert!((p.device_ns / 1e6 - 109.0).abs() < 8.0, "{p:?}");
        assert!((p.host_ns / 1e6 - 134.0).abs() < 8.0, "{p:?}");
        assert!(!p.host_wins());
        // At the spec's pinned rates a reduction would have won, which the measurement refutes.
        let ideal = price(&dev, kernel, &Rates::PCIE_PINNED, &Work { reduce: false, ..work(GIB, 4, 2.0) }, &est(), None);
        assert!(ideal.host_wins());
    }

    #[test]
    fn a_heavy_program_stays_on_the_device() {
        let (dev, kernel) = gpu(2 * GIB);
        let p = price(&dev, kernel, &Rates::PCIE, &work(GIB, GIB, 200.0), &est(), None);
        assert!(!p.host_wins(), "{p:?}");
    }

    #[test]
    fn the_host_must_win_by_more_than_the_hysteresis() {
        let p = Pricing { device_ns: 100.0, host_ns: 95.0, refused: None };
        assert!(!p.host_wins());
        let p = Pricing { device_ns: 100.0, host_ns: 85.0, refused: None };
        assert!(p.host_wins());
    }

    #[test]
    fn phi_moves_the_answer() {
        // With a result that is always read back on the host (phi = 1 for the device), the device pays the download in full.
        let (dev, kernel) = gpu(2 * GIB);
        let w = work(GIB, GIB, 2.0);
        let p = price(&dev, kernel, &Rates::PCIE, &w, &est(), None);
        let device_down_half = 0.5 * GIB as f64 / Rates::PCIE.d2h_gbs;
        assert!(p.device_ns > device_down_half && p.host_ns > 0.5 * GIB as f64 / Rates::PCIE.h2d_gbs);
    }

    #[test]
    fn nothing_crosses_a_unified_memory_link_so_a_large_step_stays_on_the_device() {
        let (dev, kernel) = gpu(2 * GIB);
        let p = price(&dev, kernel, &Rates::UNIFIED, &work(GIB, GIB, 2.0), &est(), None);
        assert!(!p.host_wins(), "{p:?}");
    }

    #[test]
    fn device_only_bytes_refuse_the_host() {
        let (dev, kernel) = gpu(GIB);
        let w = Work { in_bytes: [GIB, 0, 0, 0], out_bytes: 4, len: GIB / 4, weight: 2.0, reduce: true };
        let p = price(&dev, kernel, &Rates::PCIE, &w, &est(), None);
        assert_eq!(p.refused, Some("residency"));
        assert!(!p.host_wins());
        assert_eq!(p.candidates()[1].1, Verdict::Rejected("residency"));
    }

    #[test]
    fn a_learned_time_replaces_the_model() {
        let (dev, kernel) = gpu(GIB);
        let w = work(GIB, 4, 2.0);
        let slow = price(&dev, kernel, &Rates::PCIE, &w, &HostEstimate { cpu: cpu(), learned_ns: Some(1e9) }, None);
        assert!(!slow.host_wins());
        assert!(slow.host_ns > 1e9);
    }

    #[test]
    fn an_input_off_the_device_sets_the_flag() {
        assert!(work(4, 4, 1.0).any_off_device());
        assert!(!Work { in_bytes: [4, 0, 0, 0], out_bytes: 4, len: 1, weight: 1.0, reduce: false }.any_off_device());
    }
}
