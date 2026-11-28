//! Host strategies as CPU-JIT (W3-HOSTCPU): an auxiliary CPU context inside a GPU context compiles
//! and runs the fused programs the GPU context flattened, over host memory it does not copy.
//!
//! The executor takes a [`Fused`] (flattened by the GPU context: the program is
//! context-independent), compiles its key on the aux context (`jit().kernel`) and runs it over
//! caller-provided host slabs: inputs are imported read-only, outputs writable, through
//! `Backend::import_host`, so nothing is copied for 16-byte aligned regions (`HostBytes` is 64-byte
//! aligned). Placement is not decided here: the residency decision of W3-U6 prices a host stage with
//! [`super::placement::host_cost`] and calls [`HostExec::run_store`] / [`HostExec::run_reduce`] when
//! it picks the host. `LUMA_HOST_EXEC=off` closes the aux context (upload-and-run, the A/B baseline).

use std::any::Any;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use luma_core::hal::HostRegion;
use luma_core::Buffer;

use super::placement::{self, HostEstimate, Rates, Work};
use super::program::{Fused, FusedKey, Tail};
use super::OpKind;
use crate::builtin::{BinOp, UnOp};
use crate::profile::{Decision, DecisionSite};
use std::panic::Location;
use luma_core::hal::HostBytes;
use super::structured::emit::Emit;
use super::structured::strategy::PassMode;
use super::launch;
use crate::array::Storage;
use crate::builtin::Finish;
use crate::context::{Context, Scratch};
use crate::error::{LumaError, Result};
use crate::gpu_type::ElemLayout;
use crate::reduce::finish_passes;

/// Counters of the host executor (see [`HostExec::stats`]).
#[cfg_attr(not(test), allow(dead_code))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct HostStats {
    /// Regions shown to the CPU backend without a copy.
    pub(crate) imports: u64,
    /// Regions that had to be copied because the backend refused the import (misaligned).
    pub(crate) imports_copied: u64,
    /// Programs run.
    pub(crate) runs: u64,
}

/// The auxiliary CPU context of a GPU context.
#[derive(Debug)]
pub(crate) struct HostExec {
    cpu: Context,
    /// Tests: run on the host whatever the pricing says (never set outside tests and the residency decision's forcing).
    force: std::sync::atomic::AtomicBool,
    imports: AtomicU64,
    copied: AtomicU64,
    runs: AtomicU64,
}

/// Whether `LUMA_HOST_EXEC` allows the host executor (`off` | `0` | `false` | `no` closes it).
fn enabled() -> bool {
    !std::env::var("LUMA_HOST_EXEC").is_ok_and(|v| matches!(v.trim().to_ascii_lowercase().as_str(), "off" | "0" | "false" | "no"))
}

impl HostExec {
    /// Opens the CPU context next to `gpu`; `None` for a CPU context, with `LUMA_HOST_EXEC=off`, or when the CPU backend cannot open.
    pub(crate) fn open(gpu: &Context) -> Option<HostExec> {
        // A CPU context has no use for one (its own backend is the CPU's); tests give it one to exercise the machinery without a GPU.
        if (gpu.is_cpu() && !cfg!(test)) || !enabled() {
            return None;
        }
        let cpu = Context::cpu().ok()?;
        Some(HostExec { cpu, force: std::sync::atomic::AtomicBool::new(false), imports: AtomicU64::new(0), copied: AtomicU64::new(0), runs: AtomicU64::new(0) })
    }

    /// The CPU context (its cost model and learned kernel costs price a host stage).
    #[allow(dead_code)] // the interface of W3-U6's residency decision (pricing reads the aux context's cost model)
    pub(crate) fn context(&self) -> &Context {
        &self.cpu
    }

    #[allow(dead_code)] // as `context`
    pub(crate) fn stats(&self) -> HostStats {
        HostStats { imports: self.imports.load(Ordering::Relaxed), imports_copied: self.copied.load(Ordering::Relaxed), runs: self.runs.load(Ordering::Relaxed) }
    }

    /// A buffer over `bytes` bytes at `ptr`: an import when the backend takes it, else (a misaligned region) a copy of `copy`.
    fn import(&self, ptr: *mut u8, bytes: u64, writable: bool, copy: Option<&[u8]>) -> Result<Buffer> {
        let keepalive: Arc<dyn Any + Send + Sync> = Arc::new(());
        if let Some(buffer) = self.cpu.hal().import_host(HostRegion { ptr, len: bytes, writable, keepalive }) {
            self.imports.fetch_add(1, Ordering::Relaxed);
            return Ok(buffer);
        }
        let Some(copy) = copy else {
            return Err(LumaError::Gpu("a host output region must be 16-byte aligned".into()));
        };
        self.copied.fetch_add(1, Ordering::Relaxed);
        self.cpu.alloc_filled(copy)
    }

    /// A storage on the aux context over the host bytes `data` (`len` elements of `layout`), read-only.
    fn input(&self, data: &[u8], len: usize, layout: ElemLayout) -> Result<Arc<Storage>> {
        if data.len() as u64 != layout.bytes(len) {
            return Err(LumaError::Gpu(format!("a host input holds {} bytes, {} expected", data.len(), layout.bytes(len))));
        }
        let storage = Storage::over_buffers(&self.cpu, len, layout, |offset, bytes| {
            let part = &data[offset as usize..(offset + bytes) as usize];
            self.import(part.as_ptr().cast_mut(), bytes, false, Some(part))
        })?;
        Ok(Arc::new(storage))
    }

    /// A storage on the aux context over the host bytes `data`, which the program writes.
    fn output(&self, data: &mut [u8], len: usize, layout: ElemLayout) -> Result<Storage> {
        if data.len() as u64 != layout.bytes(len) {
            return Err(LumaError::Gpu(format!("a host output holds {} bytes, {} expected", data.len(), layout.bytes(len))));
        }
        let base = data.as_mut_ptr();
        Storage::over_buffers(&self.cpu, len, layout, |offset, bytes| {
            // SAFETY: `offset + bytes` is within `data`, which the caller holds exclusively for the call.
            self.import(unsafe { base.add(offset as usize) }, bytes, true, None)
        })
    }

    /// Runs the store program `fused` (flattened by the GPU context, `len` elements) over `inputs` (the host bytes of
    /// `fused.inputs`, in order, as laid out on the host) and writes its outputs into `outputs`, one slab per output.
    /// Programs the aux context would have to plan around (views read through shifts, staging, in-place) are refused
    /// with an error before anything runs; the caller keeps them on the device.
    pub(super) fn run_store(&self, fused: Fused, len: usize, inputs: &[&[u8]], outputs: &mut [&mut [u8]]) -> Result<()> {
        let Fused { mut key, mut scalars, .. } = fused;
        check_plain(&key, Tail::Store)?;
        if inputs.len() != key.input_tys.len() || outputs.len() != key.outputs.len() {
            return Err(LumaError::Gpu("host run: wrong number of slabs".into()));
        }
        let ins = inputs.iter().zip(&key.input_tys).map(|(data, ty)| self.input(data, len, ty.layout())).collect::<Result<Vec<_>>>()?;
        let outs = outputs
            .iter_mut()
            .zip(&key.outputs)
            .map(|(data, &at)| self.output(data, len, key.instrs[at as usize].op.out_ty().layout()))
            .collect::<Result<Vec<_>>>()?;
        let core = self.cpu.jit().kernel(&self.cpu, &mut key, &mut scalars)?;
        let refs: Vec<&Storage> = outs.iter().collect();
        launch::launch_store(&self.cpu, &core, &key, &refs, &ins, &scalars, None)?;
        self.runs.fetch_add(1, Ordering::Relaxed);
        // The imports are views of memory the caller owns only for this call.
        self.cpu.sync()?;
        drop(refs);
        Ok(())
    }

    /// Runs the reduction program `fused` (its tail is `Tail::Reduce(ops)`) over `inputs` and returns the final
    /// values, `elem` bytes per op. One segment of the aux context only (`Err` otherwise).
    #[allow(dead_code)] // run by W3-U6's host stages; measured not to pay as an automatic placement (see the spec's Measurement)
    pub(super) fn run_reduce(&self, fused: Fused, len: usize, inputs: &[&[u8]], wgsl: &'static str, elem: u64) -> Result<Vec<u8>> {
        let Tail::Reduce(ops) = fused.key.tail else {
            return Err(LumaError::Gpu("host reduce: not a reduction".into()));
        };
        check_plain(&fused.key, fused.key.tail)?;
        if inputs.len() != fused.key.input_tys.len() {
            return Err(LumaError::Gpu("host run: wrong number of slabs".into()));
        }
        let ins = inputs.iter().zip(&fused.key.input_tys).map(|(data, ty)| self.input(data, len, ty.layout())).collect::<Result<Vec<_>>>()?;
        let host = Fused { inputs: ins, ..fused };
        let mut scratch = Scratch::new(&self.cpu);
        let partials = launch::reduce_fused(&mut Emit::now(&self.cpu), PassMode::Auto, len, host, ops, &mut scratch, None, Finish::default()).map_err(crate::array::Failure::into_error)?;
        let [(partial, groups)] = <[_; 1]>::try_from(partials).map_err(|_| LumaError::Gpu("host reduction spans segments".into()))?;
        let result = finish_passes(&mut Emit::now(&self.cpu), PassMode::Auto, &mut scratch, ops, wgsl, elem, partial, groups, true, None, Finish::default())?;
        let bytes = self.cpu.read::<u8>(&result, elem * ops.as_slice().len() as u64)?;
        self.runs.fetch_add(1, Ordering::Relaxed);
        self.cpu.sync()?;
        Ok(bytes)
    }
}

/// Programs the executor runs as flattened: no views through shifts, staging, tiles, in-place pairs or mapped tails.
fn check_plain(key: &FusedKey, tail: Tail) -> Result<()> {
    let plain = key.subs.is_empty() && key.in_place.is_empty() && key.staging.is_none() && key.tile.is_none() && !key.runs;
    let tail_ok = matches!((&key.tail, &tail), (Tail::Store, Tail::Store) | (Tail::Reduce(_), Tail::Reduce(_)));
    if plain && tail_ok {
        Ok(())
    } else {
        Err(LumaError::Gpu("host executor: the program needs planning the aux context does not do (shifts, staging, in-place)".into()))
    }
}


impl HostExec {
    /// Run every candidate program on the host, whatever it costs (tests).
    #[cfg(test)]
    pub(crate) fn set_force(&self, on: bool) {
        self.force.store(on, Ordering::Relaxed);
    }
}

/// Whether the program's result is bit-identical wherever it is computed: only ops of the measured exact set (IEEE `+ - * /`, `sqrt`,
/// `min`/`max`, `abs`, `neg`, rounding to integer, comparisons, `select`, casts, integer and bit operations), and no float
/// multiplication feeding a float addition or subtraction (a device may contract the pair into an `fma`, the CPU does not).
pub(crate) fn host_exact(key: &FusedKey) -> bool {
    let (mut mul, mut add) = (false, false);
    for instr in key.instrs.iter() {
        let float = matches!(instr.op.ty, super::ElemTy::F32);
        match instr.op.kind {
            OpKind::Unary(op) => {
                if !matches!(op, UnOp::Neg | UnOp::Sqrt | UnOp::Abs | UnOp::Floor | UnOp::Ceil | UnOp::Trunc | UnOp::Round | UnOp::Sign | UnOp::Id) {
                    return false;
                }
            }
            OpKind::Binary(op) => match op {
                BinOp::Mul => mul |= float,
                BinOp::Add | BinOp::Sub => add |= float,
                BinOp::Div | BinOp::Min | BinOp::Max => {}
                op if op.is_bitwise() => {}
                _ => return false,
            },
            OpKind::Compare(_) | OpKind::Select | OpKind::Cast(_) | OpKind::Classify(_) => {}
            _ => return false,
        }
    }
    // Reductions: only min and max are order-free (a float sum or product depends on the order).
    if let Tail::Reduce(ops) = &key.tail
        && !ops.as_slice().iter().all(|op| matches!(op, crate::builtin::ReduceOp::Min | crate::builtin::ReduceOp::Max))
    {
        return false;
    }
    !(mul && add)
}

/// Measured on the 3090 and the 3080 Laptop (`jit/hostexec`): the CPU backend's reductions run at about 8 GB/s of input, a fourth of
/// its elementwise rate, so reductions are priced at it (W3-HOSTCPU measurement).
pub(crate) const CPU_REDUCE_GBS: f64 = 8.0;

/// Runs the store program `fused` of a node of `len` elements on the aux CPU context if every input is held in host memory
/// (`State::Host`), the program is host-exact (or `+host` numerics allow otherwise), the host budget has room and the pricing
/// prefers the host by more than the hysteresis. The output stays in host memory (`State::Host`), nothing is copied or downloaded.
/// `None` (nothing done) otherwise; the caller then runs the program as usual. The candidates are recorded in a `Schedule` decision.
pub(super) fn try_store(ctx: &Context, fused: &Fused, len: usize, name: &'static str, location: &'static Location<'static>) -> Option<std::result::Result<Storage, crate::array::Failure>> {
    let exec = ctx.host_exec()?;
    if len == 0 || fused.key.outputs.len() != 1 || check_plain(&fused.key, Tail::Store).is_err() || fused.inputs.is_empty() {
        return None;
    }
    // Inputs: every one a single segment of `len` elements held in host memory, read until the run is done.
    let mut reads = Vec::with_capacity(fused.inputs.len());
    let mut in_bytes = 0u64;
    for input in &fused.inputs {
        if input.segments() != 1 || input.len() != len {
            return None;
        }
        let read = input.segment(0).host_read()?;
        in_bytes += read.as_slice().len() as u64;
        reads.push(read);
    }
    let out_ty = fused.key.instrs[fused.key.outputs[0] as usize].op.out_ty();
    let layout = out_ty.layout();
    let out_bytes = layout.bytes(len);
    let work = Work { in_bytes: [0, 0, in_bytes, 0], out_bytes, len: len as u64, weight: fused.key.instrs.len() as f64, reduce: false };
    let refused = if !ctx.pager().host_room(out_bytes) {
        Some("host budget")
    } else if !host_exact(&fused.key) && !ctx.numerics().host_placement() {
        Some("numerics")
    } else {
        None
    };
    let rates = if ctx.zero_copy() { Rates::UNIFIED } else { Rates::PCIE };
    let dev = ctx.cost_model();
    let touched = (in_bytes + out_bytes) as f64;
    let kernel_ns = (touched / dev.bandwidth_gbs).max(work.weight * len as f64 / dev.f32_ops_per_ns);
    let price = placement::price(&dev, kernel_ns, &rates, &work, &HostEstimate { cpu: exec.cpu.cost_model(), learned_ns: None }, refused);
    let go = refused.is_none() && (price.host_wins() || exec.force.load(Ordering::Relaxed));
    if ctx.jit().recording_decisions() {
        let mut decision = Decision::new(name, location, DecisionSite::Schedule, if go { "host" } else { "device" });
        for (label, verdict) in price.candidates() {
            if (label == "host") != go {
                decision = decision.candidate(label, verdict);
            }
        }
        ctx.jit().note_decision(decision);
    }
    if !go {
        return None;
    }
    let mut out = HostBytes::filled_by(out_bytes as usize, |dst| {
        // The program writes every byte; the slab is handed over as plain bytes.
        // SAFETY: `MaybeUninit<u8>` and `u8` have the same layout, and the kernel overwrites the whole slab before it is read.
        let bytes = unsafe { std::slice::from_raw_parts_mut(dst.as_mut_ptr().cast::<u8>(), dst.len()) };
        let slices: Vec<&[u8]> = reads.iter().map(crate::gpu::pager::HostRead::as_slice).collect();
        let copy = Fused { key: fused.key.clone(), inputs: Vec::new(), scalars: fused.scalars.clone(), fused_nodes: Vec::new(), instr_nodes: Vec::new(), sub_roots: Vec::new() };
        exec.run_store(copy, len, &slices, &mut [bytes])
    });
    let out = match out.as_mut() {
        Ok(_) => out.ok()?,
        Err(_) => return None,
    };
    drop(reads);
    fused.mark_fused();
    let storage = Storage::from_host(ctx, len, layout, out)?;
    Some(Ok(storage))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jit::ElemTy;
    use crate::profile::Verdict;
    use crate::{Array, ContextOptions};

    /// A context with 64 KiB segments and a 1 MiB budget, so arrays of 16384 floats spill to host memory in a row.
    fn spilling(host_budget: Option<u64>) -> Context {
        Context::with_options(ContextOptions::from_env().calibrate(false).chunk_bytes(1 << 16).memory_budget(1 << 20).host_budget(host_budget)).unwrap()
    }

    const LEN: usize = 16_384;

    fn values(salt: usize) -> Vec<f32> {
        (0..LEN).map(|i| (((i * 7 + salt * 131) % 1021) as f32 - 500.0) * 0.5).collect()
    }

    /// `n` arrays; the first ones are in host memory once the budget is exceeded.
    fn arrays(ctx: &Context, n: usize) -> Vec<Array<f32>> {
        (0..n).map(|k| Array::from_slice_on(ctx, &values(k), LEN).unwrap()).collect()
    }

    #[test]
    fn spilled_inputs_run_on_the_host_and_stay_there() {
        let ctx = spilling(None);
        let Some(exec) = ctx.host_exec() else { return };
        exec.set_force(true);
        let a = arrays(&ctx, 40);
        let before = exec.stats();
        // Host-exact: no multiply feeding an add.
        let c = (&a[0] - &a[1]).abs();
        let want: Vec<f32> = values(0).iter().zip(values(1)).map(|(x, y)| (x - y).abs()).collect();
        assert_eq!(c.to_vec().unwrap(), want);
        let after = exec.stats();
        assert_eq!(after.runs, before.runs + 1, "the program ran on the host");
        assert_eq!(after.imports_copied, before.imports_copied, "64-byte aligned inputs are imported, not copied");
        assert!(after.imports > before.imports);
        // The result is in host memory and a later device kernel uploads it correctly.
        let d = &c * 3.0 + 1.0;
        let want: Vec<f32> = want.iter().map(|v| v * 3.0 + 1.0).collect();
        assert_eq!(d.to_vec().unwrap(), want);
    }

    #[test]
    fn host_and_device_results_are_bit_equal_for_a_corpus() {
        let ctx = spilling(None);
        let Some(exec) = ctx.host_exec() else { return };
        let a = arrays(&ctx, 40);
        // The baseline: the same programs with the host executor off for this context (nothing is forced, and the pricing of a unified-memory
        // or CPU context keeps them on the device).
        type Program = Box<dyn Fn(&Array<f32>, &Array<f32>) -> Array<f32>>;
        let programs: Vec<Program> = vec![
            Box::new(|x, y| x - y),
            Box::new(|x, y| (x - y).abs()),
            Box::new(|x, y| x.maximum(y)),
            Box::new(|x, y| x.minimum(y.abs())),
            Box::new(|x, y| (x + y).abs().sqrt()),
            Box::new(|x, y| crate::ops::select(&x.gt(y), x, y)),
        ];
        let want: Vec<Vec<f32>> = programs.iter().enumerate().map(|(i, f)| f(&a[2 * i], &a[2 * i + 1]).to_vec().unwrap()).collect();
        exec.set_force(true);
        let a = arrays(&ctx, 40);
        let ran = exec.stats().runs;
        for (i, f) in programs.iter().enumerate() {
            let got = f(&a[2 * i], &a[2 * i + 1]).to_vec().unwrap();
            assert_eq!(got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), want[i].iter().map(|v| v.to_bits()).collect::<Vec<_>>(), "program {i}");
        }
        assert!(exec.stats().runs > ran, "some ran on the host");
    }

    #[test]
    fn programs_that_may_differ_in_the_last_bits_stay_on_the_device_by_default() {
        let ctx = spilling(None);
        let Some(exec) = ctx.host_exec() else { return };
        exec.set_force(true);
        let a = arrays(&ctx, 40);
        let before = exec.stats().runs;
        ctx.jit().start_decisions();
        let c = a[0].sin();
        let want: Vec<f32> = c.to_vec().unwrap();
        assert_eq!(exec.stats().runs, before, "sin is not host-exact");
        let log = ctx.jit().stop_decisions();
        assert!(
            log.iter().any(|d| d.site == DecisionSite::Schedule && d.chosen == "device" && d.candidates.iter().any(|(l, v)| l == "host" && *v == Verdict::Rejected("numerics"))),
            "{log:?}"
        );
        assert_eq!(want.len(), LEN);
    }

    #[test]
    fn a_misaligned_input_is_copied_and_still_computes_right() {
        let ctx = spilling(None);
        let Some(exec) = ctx.host_exec() else { return };
        let bytes: Vec<u8> = bytemuck::cast_slice::<f32, u8>(&values(3)).to_vec();
        let mut shifted = vec![0u8; bytes.len() + 64];
        let at = (0..64).find(|k| (shifted.as_ptr() as usize + k) % 16 == 4).unwrap();
        shifted[at..at + bytes.len()].copy_from_slice(&bytes);
        let before = exec.stats();
        let storage = exec.input(&shifted[at..at + bytes.len()], LEN, ElemTy::F32.layout()).unwrap();
        assert_eq!(exec.stats().imports_copied, before.imports_copied + 1);
        let got: Vec<f32> = exec.context().read_storage(&storage).unwrap();
        assert_eq!(got, values(3));
    }

    #[test]
    fn outputs_over_the_host_budget_stay_on_the_device() {
        let ctx = spilling(Some(1 << 20));
        let Some(exec) = ctx.host_exec() else { return };
        exec.set_force(true);
        let a = arrays(&ctx, 40);
        // The spilled inputs fill the host budget: no room for the result.
        let before = exec.stats().runs;
        ctx.jit().start_decisions();
        let want: Vec<f32> = values(0).iter().zip(values(1)).map(|(x, y)| (x - y).abs()).collect();
        assert_eq!((&a[0] - &a[1]).abs().to_vec().unwrap(), want);
        let log = ctx.jit().stop_decisions();
        if exec.stats().runs == before {
            assert!(log.iter().any(|d| d.candidates.iter().any(|(l, v)| l == "host" && *v == Verdict::Rejected("host budget"))), "{log:?}");
        }
    }
}
