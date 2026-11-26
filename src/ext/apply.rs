//! `Op::apply`: validation, selection, the decision record, and the `Lower` a strategy launches through.

use std::borrow::Cow;
use std::panic::{catch_unwind, AssertUnwindSafe, Location};
use std::sync::Arc;

use super::{numerics_allowed, warn_once, ArgInfo, OpDef};
use crate::array::Storage;
use crate::context::Context;
use crate::erased::{AnyArray, Dyn, Erased};
use crate::error::{LumaError, Result};
use crate::gpu_type::ElemLayout;
use crate::jit::structured::emit::Emit;
use crate::kernel::{check_reach, Kernel, KernelArg, KernelBuilder, ProblemSize, Reach};
use crate::numerics::Numerics;
use crate::profile::{Decision, DecisionSite, Verdict};
use luma_core::{CostModel, Features};

/// What a strategy sees when it is asked whether it applies and what it costs.
#[non_exhaustive]
pub struct Fit<'a> {
    inputs: &'a [ArgInfo],
    outputs: &'a [ArgInfo],
    params: &'a [u8],
    features: Features,
    cost_model: CostModel,
    numerics: Numerics,
}

impl Fit<'_> {
    /// The input operands.
    pub fn inputs(&self) -> &[ArgInfo] {
        self.inputs
    }

    /// The outputs the op will produce.
    pub fn outputs(&self) -> &[ArgInfo] {
        self.outputs
    }

    /// The uniform block of the call.
    pub fn params(&self) -> &[u8] {
        self.params
    }

    /// The device's features.
    pub fn features(&self) -> Features {
        self.features
    }

    /// The device's cost model.
    pub fn cost_model(&self) -> &CostModel {
        &self.cost_model
    }

    /// The numerics policy of the context.
    pub fn numerics(&self) -> Numerics {
        self.numerics
    }
}

impl std::fmt::Debug for Fit<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Fit").field("inputs", &self.inputs).field("outputs", &self.outputs).finish_non_exhaustive()
    }
}

/// A scratch buffer of a [`Lower`], bound with [`Bind::Scratch`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ScratchId(u32);

/// What one binding of a launch of a strategy is.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum Bind {
    /// Input `i`, read-only.
    Input(usize),
    /// Output `i`, writable.
    Output(usize),
    /// A scratch buffer from [`Lower::scratch`], writable.
    Scratch(ScratchId),
    /// A uniform of these bytes (a copy), bound as the next uniform of the kernel.
    Uniform(Vec<u8>),
}

/// What a strategy's `lower` launches through: the operands of the call and a way to launch
/// kernels over them. It holds the buffers and their pins for the duration of the call.
#[non_exhaustive]
pub struct Lower<'a> {
    ctx: &'a Context,
    inputs: &'a [ArgInfo],
    outputs: &'a [ArgInfo],
    params: &'a [u8],
    in_storage: &'a [Arc<Storage>],
    out_storage: &'a [Arc<Storage>],
    scratch: Vec<Arc<Storage>>,
    reach: Reach,
    emit: Emit<'a>,
    launched: u32,
}

impl Lower<'_> {
    /// The input operands.
    pub fn inputs(&self) -> &[ArgInfo] {
        self.inputs
    }

    /// The outputs.
    pub fn outputs(&self) -> &[ArgInfo] {
        self.outputs
    }

    /// The uniform block of the call.
    pub fn params(&self) -> &[u8] {
        self.params
    }

    /// Starts building a kernel on the op's context; `Checked` unless the author calls the unsafe
    /// [`KernelBuilder::unchecked_bounds`].
    pub fn kernel(&self, wgsl: &str) -> KernelBuilder<'_> {
        self.ctx.kernel(wgsl)
    }

    /// Device scratch of at least `bytes` bytes, charged to the memory budget and released after
    /// the op; its contents are unspecified.
    pub fn scratch(&mut self, bytes: u64) -> Result<ScratchId> {
        let words = usize::try_from(bytes.div_ceil(4).max(1)).map_err(|_| LumaError::BufferTooLarge { requested: bytes, max: u64::MAX })?;
        let storage = Arc::new(Storage::new(self.ctx, words, ElemLayout::plain(4)));
        if storage.segments() > 1 {
            return Err(LumaError::Chunked("scratch of a custom op must fit one buffer"));
        }
        self.scratch.push(storage);
        Ok(ScratchId(u32::try_from(self.scratch.len() - 1).map_err(|_| LumaError::InvalidShape("too many scratch buffers".into()))?))
    }

    /// Launches `kernel` over `size` with `binds` as its arguments, in order. Validated as
    /// [`Kernel::launch`] validates, and against the strategy's declared reach. The launch is
    /// recorded with the rest of the batch.
    pub fn launch(&mut self, kernel: &Kernel, size: impl ProblemSize, binds: &[Bind]) -> Result<()> {
        let size = size.into_size();
        let mut args: Vec<KernelArg<'_>> = Vec::with_capacity(binds.len());
        for bind in binds {
            let arg = match bind {
                Bind::Input(i) => KernelArg::from_storage(self.ctx, self.in_storage.get(*i).ok_or_else(|| bad_bind("input", *i, self.in_storage.len()))?, false)?,
                Bind::Output(i) => KernelArg::from_storage(self.ctx, self.out_storage.get(*i).ok_or_else(|| bad_bind("output", *i, self.out_storage.len()))?, true)?,
                Bind::Scratch(ScratchId(i)) => {
                    let i = *i as usize;
                    KernelArg::from_storage(self.ctx, self.scratch.get(i).ok_or_else(|| bad_bind("scratch", i, self.scratch.len()))?, true)?
                }
                Bind::Uniform(bytes) => KernelArg::from_uniform_bytes(bytes.clone()),
            };
            args.push(arg);
        }
        if self.reach != Reach::Whole {
            check_reach(self.reach, size, &args, false)?;
        }
        kernel.launch_into(size, &args, &mut self.emit)?;
        self.launched += 1;
        Ok(())
    }
}

impl std::fmt::Debug for Lower<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Lower").field("inputs", &self.inputs).field("outputs", &self.outputs).finish_non_exhaustive()
    }
}

fn bad_bind(what: &str, index: usize, count: usize) -> LumaError {
    LumaError::InvalidShape(format!("bind of {what} {index}, but the op has {count}"))
}

/// Where a strategy stands in a selection.
#[derive(Clone, Copy)]
enum Status {
    Rejected(&'static str),
    /// Fits; with its estimated cost once it was asked for.
    Fits(Option<f64>),
}

fn panic_text(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(text) = payload.downcast_ref::<&str>() {
        (*text).to_owned()
    } else if let Some(text) = payload.downcast_ref::<String>() {
        text.clone()
    } else {
        "panicked".to_owned()
    }
}

pub(super) fn apply(ctx: &Context, def: &Arc<OpDef>, inputs: &[&dyn AnyArray], params: &[u8], location: &'static Location<'static>) -> Result<Vec<Dyn>> {
    let name = def.name;
    // 1. The operands.
    if inputs.len() != def.inputs.len() {
        return Err(LumaError::InvalidShape(format!("op `{name}` takes {} inputs, got {}", def.inputs.len(), inputs.len())));
    }
    if params.len() != def.params {
        return Err(LumaError::InvalidShape(format!("op `{name}` takes {} bytes of params, got {}", def.params, params.len())));
    }
    for (i, input) in inputs.iter().enumerate() {
        if !input.context().same_as(ctx) {
            return Err(LumaError::ContextMismatch);
        }
        if input.elem() != def.inputs[i] {
            return Err(LumaError::InvalidShape(format!("op `{name}` input {i} is {}, got {}", def.inputs[i], input.elem())));
        }
    }
    let mut in_infos = Vec::with_capacity(inputs.len());
    let mut in_storage = Vec::with_capacity(inputs.len());
    for input in inputs {
        let storage = input.storage_ref().map_err(|error| LumaError::clone(&error))?;
        let storage = Arc::clone(storage.0);
        if storage.segments() > 1 {
            return Err(LumaError::Chunked("custom ops need arrays that fit one buffer"));
        }
        in_infos.push(ArgInfo::new(input.elem(), input.shape()));
        in_storage.push(storage);
    }

    // 2. The outputs.
    let mut out_infos = Vec::with_capacity(def.outputs.len());
    let mut out_storage = Vec::with_capacity(def.outputs.len());
    for (kind, rule) in &def.outputs {
        let shape = match rule {
            super::OutShape::Same(i) => in_infos[*i].shape,
            super::OutShape::Fixed(shape) => *shape,
            super::OutShape::With(f) => f(&in_infos, params)?,
        };
        let len = shape.validate()?;
        let storage = Arc::new(Storage::new(ctx, len, kind.layout()));
        if storage.segments() > 1 {
            return Err(LumaError::Chunked("custom ops need arrays that fit one buffer"));
        }
        out_infos.push(ArgInfo::new(*kind, shape));
        out_storage.push(storage);
    }

    // 3. Selection.
    let fit = Fit { inputs: &in_infos, outputs: &out_infos, params, features: ctx.features(), cost_model: super::cost_model(ctx), numerics: ctx.numerics() };
    let mut status: Vec<Status> = def
        .strategies
        .iter()
        .map(|strategy| {
            if !fit.features.contains(strategy.needs) {
                Status::Rejected("needs a device feature")
            } else if !numerics_allowed(strategy.numerics, &fit.numerics) {
                Status::Rejected("numerics policy")
            } else if strategy.fits.as_ref().is_some_and(|fits| !fits(&fit)) {
                Status::Rejected("does not fit")
            } else {
                Status::Fits(None)
            }
        })
        .collect();
    let mut order: Vec<usize> = (0..status.len()).filter(|&k| matches!(status[k], Status::Fits(_))).collect();
    let mut forced = false;
    if let Some(wanted) = ctx.ext_registry().forced(name) {
        match def.strategies.iter().position(|strategy| strategy.name == wanted) {
            Some(k) if matches!(status[k], Status::Fits(_)) => {
                for &other in &order {
                    if other != k {
                        status[other] = Status::Rejected("not the forced strategy");
                    }
                }
                order = vec![k];
                forced = true;
            }
            _ => warn_once(format!("luma: LUMA_STRATEGY forces ext:{name}/{wanted}, which does not fit this call; the default selection runs")),
        }
    }
    if order.len() > 1 && !forced && order.iter().all(|&k| def.strategies[k].cost.is_some()) {
        for &k in &order {
            if let Some(cost) = &def.strategies[k].cost {
                status[k] = Status::Fits(Some(cost(&fit)));
            }
        }
        let cost_of = |k: usize| if let Status::Fits(Some(ns)) = status[k] { ns } else { f64::INFINITY };
        order.sort_by(|&a, &b| cost_of(a).total_cmp(&cost_of(b)));
    }

    // 4. Run the first that does not decline.
    let mut chosen = None;
    for &k in &order {
        let strategy = &def.strategies[k];
        let mut lower = Lower {
            ctx,
            inputs: &in_infos,
            outputs: &out_infos,
            params,
            in_storage: &in_storage,
            out_storage: &out_storage,
            scratch: Vec::new(),
            reach: strategy.reach,
            emit: Emit::now(ctx),
            launched: 0,
        };
        let result = catch_unwind(AssertUnwindSafe(|| (strategy.lower)(&mut lower)));
        let fail = |message: String| LumaError::Extension { op: name.to_owned(), strategy: strategy.name.to_owned(), message };
        match result {
            Ok(Ok(())) => {
                chosen = Some(k);
                break;
            }
            Ok(Err(LumaError::Declined)) if lower.launched == 0 => status[k] = Status::Rejected("declined"),
            Ok(Err(LumaError::Declined)) => return Err(fail("declined after it launched".to_owned())),
            Ok(Err(error)) => return Err(fail(error.to_string())),
            Err(payload) => return Err(fail(panic_text(&*payload))),
        }
    }
    let Some(chosen) = chosen else {
        let reasons = def
            .strategies
            .iter()
            .zip(&status)
            .map(|(strategy, status)| {
                let why = match status {
                    Status::Rejected(why) => (*why).to_owned(),
                    Status::Fits(_) => "declined".to_owned(),
                };
                (strategy.name.to_owned(), why)
            })
            .collect();
        return Err(LumaError::NoStrategy { op: name.to_owned(), reasons });
    };

    // 5. The record.
    if ctx.jit().recording_decisions() {
        let mut decision = Decision::new(name, location, DecisionSite::Extension, Cow::Borrowed(def.strategies[chosen].name));
        for (k, strategy) in def.strategies.iter().enumerate() {
            if k == chosen {
                continue;
            }
            let verdict = match status[k] {
                Status::Rejected(why) => Verdict::Rejected(why),
                Status::Fits(Some(ns)) => Verdict::Estimate { ns },
                Status::Fits(None) => Verdict::Rejected("registration order"),
            };
            decision = decision.candidate(Cow::Borrowed(strategy.name), verdict);
        }
        ctx.jit().note_decision(decision);
    }

    Ok(out_storage.into_iter().zip(&out_infos).map(|(storage, info)| Dyn::new(Erased::from_storage(info.elem, ctx, info.shape, storage))).collect())
}
