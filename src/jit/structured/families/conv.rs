//! The conv family: `conv1d` (one direct launch), `conv1d_fft` (`fft::convolve`) and `fir` (the direct kernel, blocks of taps, the
//! FFT). Each strategy applies at one [`ConvStage`]; the `fits` functions are the inline conditions of `src/conv.rs`,
//! `src/fft.rs::convolve` and `src/signal.rs` at `f0e5723`.
//!
//! The overlap-save rule of `fft::convolve` (whole transform against blocks) is a cost comparison of its own, in units of passes; it
//! stays a function ([`crate::fft::overlap_blocks`]) called from `fits`, as the spec says, and its result travels in the
//! parameters so that the lowering does not compute it twice.

use super::{boundary, entry, id, split_global};
use crate::builtin::CONV_MAX_STRIDE;
use crate::caps::Caps;
use crate::context::Context;
use crate::jit::structured::boundary::{ConvAttrs, ConvParams, ConvStage};
use crate::jit::structured::strategy::{BoundaryAttrs, FamilyTag, Params, Residencies, SplitClass, Strategy, StrategyId, Subject};
use crate::signal::{FIR_DIRECT_MAX, FIR_FFT_MAX_BATCH, FIR_FFT_MIN};

impl StrategyId {
    pub(crate) const CONV_DIRECT_TILED: StrategyId = id(FamilyTag::Conv, 0);
    pub(crate) const CONV_DIRECT: StrategyId = id(FamilyTag::Conv, 1);
    pub(crate) const CONV_FFT_SINGLE: StrategyId = id(FamilyTag::Conv, 2);
    pub(crate) const CONV_FFT_BLOCKED: StrategyId = id(FamilyTag::Conv, 3);
    pub(crate) const CONV_FIR_DIRECT: StrategyId = id(FamilyTag::Conv, 4);
    pub(crate) const CONV_FIR_BLOCKS: StrategyId = id(FamilyTag::Conv, 5);
    pub(crate) const CONV_FIR_FFT: StrategyId = id(FamilyTag::Conv, 6);
}

pub(crate) static CONV_STRATEGIES: [Strategy; 7] = [
    entry(StrategyId::CONV_DIRECT_TILED, "direct_tiled", Caps::empty(), split_halo, fits_direct_tiled),
    entry(StrategyId::CONV_DIRECT, "direct", Caps::empty(), split_halo, fits_direct),
    entry(StrategyId::CONV_FFT_SINGLE, "fft_single", Caps::empty(), split_global, fits_fft_single),
    entry(StrategyId::CONV_FFT_BLOCKED, "fft_blocked", Caps::empty(), split_halo, fits_fft_blocked),
    entry(StrategyId::CONV_FIR_DIRECT, "fir_direct", Caps::empty(), split_halo, fits_fir_direct),
    entry(StrategyId::CONV_FIR_BLOCKS, "fir_blocks", Caps::empty(), split_halo, fits_fir_blocks),
    entry(StrategyId::CONV_FIR_FFT, "fir_fft", Caps::empty(), split_halo, fits_fir_fft),
];

/// The entries that apply at `ConvStage::Direct`, `Fft` and `Fir`: runs of the table, so an op scans only its own.
pub(crate) const DIRECT_STRATEGIES: &[Strategy] = CONV_STRATEGIES.split_at(2).0;
pub(crate) const FFT_STRATEGIES: &[Strategy] = CONV_STRATEGIES.split_at(2).1.split_at(2).0;
pub(crate) const FIR_STRATEGIES: &[Strategy] = CONV_STRATEGIES.split_at(4).1;

fn attrs(subject: &Subject<'_>) -> ConvAttrs {
    match boundary(subject).attrs {
        BoundaryAttrs::Conv(attrs) => attrs,
        _ => unreachable!("a conv boundary carries ConvAttrs"),
    }
}

/// The halo of a direct kernel or a block: `kernel_len - 1` cells.
fn split_halo(subject: &Subject<'_>, _: &Params) -> SplitClass {
    SplitClass::Halo { flat_radius: attrs(subject).kernel_len.saturating_sub(1) }
}

/// `stride <= CONV_MAX_STRIDE`: the tiled kernel, `conv_workgroup` invocations of `conv_per_thread` outputs.
fn fits_direct_tiled(subject: &Subject<'_>, ctx: &Context, _: &Residencies) -> Option<Params> {
    let a = attrs(subject);
    (a.stage == ConvStage::Direct && a.stride <= CONV_MAX_STRIDE).then(|| {
        let tuning = ctx.tuning();
        Params::Conv(ConvParams { per_thread: tuning.conv_per_thread, workgroup: tuning.conv_workgroup, ..ConvParams::NONE })
    })
}

/// `stride > CONV_MAX_STRIDE`: one invocation per output.
fn fits_direct(subject: &Subject<'_>, _: &Context, _: &Residencies) -> Option<Params> {
    let a = attrs(subject);
    (a.stage == ConvStage::Direct && a.stride > CONV_MAX_STRIDE).then_some(Params::Conv(ConvParams::NONE))
}

/// The padded transform's size and the overlap-save block (`None`: one transform of the whole signal).
fn overlap(a: &ConvAttrs) -> (u32, Option<u32>) {
    let log_n = a.out_len.checked_next_power_of_two().map_or(32, u32::trailing_zeros);
    (log_n, crate::fft::overlap_blocks(a.kernel_len, a.out_len, log_n, a.tile_log).map(|(log_b, _)| log_b))
}

/// `fft::convolve` where the overlap-save rule picks no blocks (`blocks.filter(cost < full_cost)` is `None`).
fn fits_fft_single(subject: &Subject<'_>, _: &Context, _: &Residencies) -> Option<Params> {
    let a = attrs(subject);
    if a.stage != ConvStage::Fft {
        return None;
    }
    let (log_n, blocks) = overlap(&a);
    blocks.is_none().then_some(Params::Conv(ConvParams { log_n, ..ConvParams::NONE }))
}

/// `fft::convolve` where the rule picks a block size.
fn fits_fft_blocked(subject: &Subject<'_>, _: &Context, _: &Residencies) -> Option<Params> {
    let a = attrs(subject);
    if a.stage != ConvStage::Fft {
        return None;
    }
    let (log_n, blocks) = overlap(&a);
    blocks.map(|log_b| Params::Conv(ConvParams { log_n, log_b, ..ConvParams::NONE }))
}

/// `fir`: `k <= FIR_DIRECT_MAX`.
fn fits_fir_direct(subject: &Subject<'_>, _: &Context, _: &Residencies) -> Option<Params> {
    let a = attrs(subject);
    (a.stage == ConvStage::Fir && a.kernel_len as usize <= FIR_DIRECT_MAX).then_some(Params::Conv(ConvParams::NONE))
}

/// `fir_long`: `k > FIR_DIRECT_MAX && (k < FIR_FFT_MIN || batch >= FIR_FFT_MAX_BATCH)`.
fn fits_fir_blocks(subject: &Subject<'_>, _: &Context, _: &Residencies) -> Option<Params> {
    let a = attrs(subject);
    let k = a.kernel_len as usize;
    (a.stage == ConvStage::Fir && k > FIR_DIRECT_MAX && (k < FIR_FFT_MIN || a.batch as usize >= FIR_FFT_MAX_BATCH))
        .then(|| Params::Conv(ConvParams { blocks: k.div_ceil(FIR_DIRECT_MAX) as u32, ..ConvParams::NONE }))
}

/// `fir_long`: the rest (`k >= FIR_FFT_MIN` and a batch of fewer than `FIR_FFT_MAX_BATCH`).
fn fits_fir_fft(subject: &Subject<'_>, _: &Context, _: &Residencies) -> Option<Params> {
    let a = attrs(subject);
    let k = a.kernel_len as usize;
    (a.stage == ConvStage::Fir && k > FIR_DIRECT_MAX && k >= FIR_FFT_MIN && (a.batch as usize) < FIR_FFT_MAX_BATCH).then_some(Params::Conv(ConvParams::NONE))
}
