//! Lineage: what a store kernel computed a segment from, so that a segment pushed off the device
//! can be recomputed at device bandwidth instead of being read back over the transfer link.
//!
//! A segment gets a [`Lineage`] when [`super::launch::run_store`] makes it: the kernel, its
//! scalars and the inputs it read. While the segment is resident the lineage holds its inputs
//! weakly, so it extends no lifetime. When the pager evicts the segment by lineage (see
//! `Context::evict_one`) the inputs become strong, and the next pin runs the kernel again into a
//! new buffer (see `Segment::pin`). The recomputation is bit-identical: it is the same kernel on
//! the same input values.
//!
//! What keeps the input values the lineage describes (the rules of `gpu::pager`):
//! - the input segments list the segments whose lineage reads them (their *dependents*);
//! - a may-write pin of an input first recomputes the dependents that are evicted by lineage,
//!   then drops the lineage of **every** dependent (resident ones too: their weak inputs would
//!   still upgrade and describe the old values);
//! - the lineage stores the write generation of each input ([`Readers::generation`]) and is not
//!   used to evict once it changed: a second guard, for a write that did not go through a pin.
//!
//! [`Readers::generation`]: super::readers::Readers::generation

use std::sync::{Arc, Weak};

use luma_core::Buffer;

use crate::array::Storage;
use crate::context::{Context, RawArg};
use crate::error::LumaError;
use crate::kernel::KernelCore;
use crate::ops::{fold_workgroups, len_u32};

use super::cost::{kernel_ns, op_weight};
use super::launch::{params, pin_inputs, InPin};
use super::program::{FusedKey, Tail};

/// Bytes per nanosecond (12 GB/s) at which a readback or an upload is assumed to move data. A
/// conservative constant (PCIe 3.0 x16 reaches 12-13 GB/s pinned, 25 GB/s on 4.0): it stays
/// until the transfer lanes are calibrated (see P2b and U6).
pub(crate) const TRANSFER_BYTES_PER_NS: f64 = 12.0;

/// Bytes below which no lineage is kept: recomputing beats a transfer only when the transfer
/// (2 x bytes / [`TRANSFER_BYTES_PER_NS`]) outweighs a kernel launch, which is at least 2 us on
/// any device the cost model knows, so an output under 12 KiB never pays. Keeping lineage costs
/// host time on every store of a context under a budget, which a launch-bound loop of small
/// arrays would feel.
const MIN_BYTES: u64 = 16 << 10;

/// How to recompute one segment: the store kernel that made it and what it read.
pub(crate) struct Lineage {
    core: Arc<KernelCore>,
    key: Arc<FusedKey>,
    scalars: Box<[u32]>,
    /// Elements of the output.
    len: usize,
    /// The inputs while the segment is evicted by lineage; empty while it is resident.
    inputs: Box<[Arc<Storage>]>,
    /// The inputs, held while the segment is resident.
    weak_inputs: Box<[Weak<Storage>]>,
    /// [`Readers::generation`](super::readers::Readers::generation) of each input at record time.
    input_generations: Box<[u64]>,
}

impl std::fmt::Debug for Lineage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Lineage").field("len", &self.len).field("inputs", &self.weak_inputs.len()).field("held", &!self.inputs.is_empty()).finish_non_exhaustive()
    }
}

/// What the eviction choice needs to know about a lineage, copied out so that no lock is held
/// while the inputs are looked at.
pub(crate) struct Snapshot {
    key: Arc<FusedKey>,
    len: usize,
    weak_inputs: Box<[Weak<Storage>]>,
    input_generations: Box<[u64]>,
}

/// Everything one recomputation needs, copied out of a [`Lineage`] so that it runs without its lock.
pub(crate) struct Recompute {
    core: Arc<KernelCore>,
    key: Arc<FusedKey>,
    scalars: Box<[u32]>,
    len: usize,
    inputs: Box<[Arc<Storage>]>,
}

/// The inputs of a recomputation, pinned on the device until it was launched.
pub(crate) struct PinnedInputs(Vec<InPin>);

/// Whether the store that `run_store` is about to finish into `output` from `inputs` can be
/// recomputed later: a plain one-output store of single-segment arrays, outside any recording,
/// on a context that evicts under a budget (without one a segment only moves when the device is
/// full, and the cost of keeping lineage for every kernel is not paid).
pub(super) fn wanted(ctx: &Context, key: &FusedKey, output: &Storage, inputs: &[Arc<Storage>]) -> bool {
    ctx.pager().evicts_under_budget()
        && matches!(key.tail, Tail::Store)
        && key.outputs.len() == 1
        && key.in_place.is_empty()
        && output.segments() == 1
        && (output.byte_len() >= MIN_BYTES || ctx.pager().lineage_forced())
        && inputs.iter().all(|input| input.segments() == 1 && !input.is_window())
        && !ctx.recording_here()
}

impl Lineage {
    /// The lineage of a launch of `core` (built from `key`) with `scalars` over `inputs`.
    pub(super) fn new(core: Arc<KernelCore>, key: FusedKey, scalars: Vec<u32>, len: usize, inputs: &[Arc<Storage>]) -> Lineage {
        Lineage {
            core,
            key: Arc::new(key),
            scalars: scalars.into_boxed_slice(),
            len,
            inputs: Box::new([]),
            weak_inputs: inputs.iter().map(Arc::downgrade).collect(),
            input_generations: inputs.iter().map(|input| input.readers.generation()).collect(),
        }
    }

    /// What the eviction choice looks at.
    pub(crate) fn snapshot(&self) -> Snapshot {
        Snapshot {
            key: Arc::clone(&self.key),
            len: self.len,
            weak_inputs: self.weak_inputs.clone(),
            input_generations: self.input_generations.clone(),
        }
    }

    /// Holds `inputs` strongly (the segment is evicted by lineage); returns their bytes.
    pub(crate) fn hold(&mut self, inputs: Box<[Arc<Storage>]>) -> u64 {
        let bytes = inputs.iter().map(|input| input.byte_len()).sum();
        self.inputs = inputs;
        bytes
    }

    /// Lets go of the strong inputs (the segment is resident again); returns them with their
    /// bytes, for the caller to drop outside its locks.
    pub(crate) fn release(&mut self) -> (u64, Box<[Arc<Storage>]>) {
        let inputs = std::mem::take(&mut self.inputs);
        (inputs.iter().map(|input| input.byte_len()).sum(), inputs)
    }

    /// Bytes of the strong inputs.
    pub(crate) fn held_bytes(&self) -> u64 {
        self.inputs.iter().map(|input| input.byte_len()).sum()
    }

    /// The recomputation, if the inputs are held (the segment is evicted by lineage).
    pub(crate) fn recompute(&self) -> Option<Recompute> {
        if self.inputs.len() != self.weak_inputs.len() {
            return None;
        }
        Some(Recompute {
            core: Arc::clone(&self.core),
            key: Arc::clone(&self.key),
            scalars: self.scalars.clone(),
            len: self.len,
            inputs: self.inputs.clone(),
        })
    }
}

impl Snapshot {
    /// The inputs, strongly, if evicting a segment of `bytes` bytes by lineage is allowed and
    /// pays: every input is alive, was not written since the lineage was made, and is resident
    /// on the device (so that none is itself recomputed or read back, which also keeps the
    /// recursion of a pin one level deep); and the kernel costs less than reading the segment
    /// back and uploading it again (only uploading it again if `clean`: a valid host copy is kept).
    ///
    /// Takes the input segments' state locks one at a time; the caller holds none.
    pub(crate) fn evictable_inputs(&self, ctx: &Context, bytes: u64, clean: bool) -> Option<Box<[Arc<Storage>]>> {
        let inputs = self.weak_inputs.iter().map(Weak::upgrade).collect::<Option<Box<[_]>>>()?;
        if inputs.iter().zip(&self.input_generations).any(|(input, &generation)| input.readers.generation() != generation) {
            return None;
        }
        if !inputs.iter().all(|input| input.only_segment().on_device()) {
            return None;
        }
        if ctx.pager().lineage_forced() {
            return Some(inputs);
        }
        let cost = ctx.cost_model();
        let weight: f64 = self.key.instrs.iter().chain(self.key.subs.iter().flatten()).map(|instr| op_weight(&instr.op, &cost)).sum();
        let traffic = bytes as f64 + inputs.iter().map(|input| input.byte_len() as f64).sum::<f64>();
        let recompute_ns = kernel_ns(&cost, traffic, weight * self.len as f64);
        let transfer_ns = if clean { 1.0 } else { 2.0 } * bytes as f64 / TRANSFER_BYTES_PER_NS;
        (recompute_ns < transfer_ns).then_some(inputs)
    }
}

impl Recompute {
    /// Pins the inputs on the device, bringing back the ones that left it.
    pub(crate) fn pin_inputs(&self) -> Result<PinnedInputs, LumaError> {
        pin_inputs(&self.key, &self.inputs, None, 0, len_u32(self.len)?).map(PinnedInputs)
    }

    /// Launches the kernel into `out`, a buffer of at least `bytes` bytes. The launch is not
    /// counted as an elementwise launch of the program: it is the pager's.
    pub(crate) fn launch(&self, ctx: &Context, out: &Buffer, bytes: u64, inputs: &PinnedInputs) -> Result<(), LumaError> {
        let key = &*self.key;
        let n = len_u32(self.len)?;
        let words = params(key, None, None, 0, 0, None, &self.scalars)?;
        let mut raw: Vec<RawArg<'_>> = Vec::with_capacity(inputs.0.len() + 2);
        raw.push(RawArg::write(out, bytes));
        raw.extend(inputs.0.iter().map(InPin::raw));
        if !words.is_empty() {
            raw.push(RawArg::Uniform(bytemuck::cast_slice(&words)));
        }
        let groups = match (&key.tile, &key.staging) {
            (Some(tile), _) => tile.tiles(),
            (None, Some(staging)) => staging.tiles(),
            (None, None) => n.div_ceil(self.core.workgroup[0] * u32::from(key.coarsen.max(1)) * key.pack_factor()),
        };
        ctx.launch(&self.core, fold_workgroups(ctx, groups)?, [n, 1, 1], &raw)
    }
}
