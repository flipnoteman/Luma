//! Strategy costs that learn from measurement: one execution of a strategy in `Tuning::telemetry_every` is timed on the device, and
//! the median of the last samples of a (strategy, shape class, numerics) cell replaces a static estimate where `select` ranks
//! candidates of one `BitClass` (see `select`). Per context; cheap enough to stay on: an execution that is not sampled pays one
//! relaxed counter increment.

use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::Instant;

use super::emit::Emit;
use super::strategy::{FamilyTag, StrategyId, Subject, FAMILY_COUNT, MAX_PER_FAMILY};
use crate::caps::Caps;
use crate::context::Context;
use crate::gpu::hash::FxMap;
use crate::gpu::lock;
use crate::numerics::Numerics;

/// Executions of a strategy that are never sampled: the first ones include cold caches and the kernel compile.
const WARM_UP: u32 = 3;

/// Samples a cell keeps.
const RING: usize = 16;

/// Sampled executions whose timings may be outstanding at once; beyond that the oldest are forgotten (a backend that dropped a
/// tag never reports it).
const MAX_INFLIGHT: usize = 64;

/// The shape of a subject, coarsely: what cached rankings and measured costs are keyed by.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) struct ShapeClass {
    pub(crate) log2: [u8; 4],
    /// Bit `d`: extent `d` is a multiple of 4.
    pub(crate) aligned: u8,
    /// Bit `d`: extent `d` is below 32.
    pub(crate) small: u8,
}

impl ShapeClass {
    pub(crate) fn of(dims: [u32; 4]) -> ShapeClass {
        let mut class = ShapeClass { log2: [0; 4], aligned: 0, small: 0 };
        for (d, &extent) in dims.iter().enumerate() {
            class.log2[d] = extent.checked_ilog2().unwrap_or(0) as u8;
            class.aligned |= u8::from(extent % 4 == 0) << d;
            class.small |= u8::from(extent < 32) << d;
        }
        class
    }
}

/// The timings of one execution: its sequence number and, in the low byte, which launch of it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct SampleTag(u64);

impl SampleTag {
    #[inline]
    pub(crate) fn part(self, launch: u8) -> u64 {
        self.0 | u64::from(launch)
    }

    fn seq(tag: u64) -> u64 {
        tag >> 8
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) struct CellKey {
    pub(crate) strategy: StrategyId,
    pub(crate) shape: ShapeClass,
    pub(crate) numerics: Numerics,
}

/// The last samples of a cell: nanoseconds per execution of the strategy.
struct Cell {
    ring: [f32; RING],
    filled: u8,
    next: u8,
}

impl Cell {
    fn push(&mut self, ns: f32) -> u8 {
        self.ring[usize::from(self.next)] = ns;
        self.next = (self.next + 1) % RING as u8;
        self.filled = (self.filled + 1).min(RING as u8);
        self.filled
    }

    fn median(&self) -> f64 {
        let mut sorted = [0f32; RING];
        let samples = &mut sorted[..usize::from(self.filled)];
        samples.copy_from_slice(&self.ring[..usize::from(self.filled)]);
        samples.sort_unstable_by(f32::total_cmp);
        f64::from(samples[samples.len() / 2])
    }
}

/// The launches of one sampled execution, summed as their timings arrive.
struct Pending {
    cell: CellKey,
    ns: f64,
    parts: u8,
    /// How many launches the execution made, once it is over.
    expected: Option<u8>,
}

/// What [`Telemetry::begin`] hands the execution.
pub(crate) struct Begun<'a> {
    pub(crate) emit: Emit<'a>,
    /// Set for a sampled strategy that makes no launch: timed on the host.
    timer: Option<(CellKey, Instant)>,
}

pub(crate) struct Telemetry {
    /// Executions begun, by strategy: the sampling clock.
    counters: [[AtomicU32; MAX_PER_FAMILY]; FAMILY_COUNT],
    /// Executions, by family: the exploration clock.
    explores: [AtomicU32; FAMILY_COUNT],
    seq: AtomicU64,
    /// Bumped when a cell gets its `min_samples`th sample: invalidates the cached rankings.
    epoch: AtomicU64,
    /// Sampled executions that have not been completed or discarded.
    pending: AtomicU32,
    tick: AtomicU32,
    cells: Mutex<FxMap<CellKey, Cell>>,
    inflight: Mutex<FxMap<u64, Pending>>,
}

impl std::fmt::Debug for Telemetry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Telemetry").field("epoch", &self.epoch()).finish_non_exhaustive()
    }
}

impl Default for Telemetry {
    fn default() -> Self {
        Telemetry {
            counters: std::array::from_fn(|_| std::array::from_fn(|_| AtomicU32::new(0))),
            explores: std::array::from_fn(|_| AtomicU32::new(0)),
            seq: AtomicU64::new(0),
            epoch: AtomicU64::new(0),
            pending: AtomicU32::new(0),
            tick: AtomicU32::new(0),
            cells: Mutex::new(FxMap::default()),
            inflight: Mutex::new(FxMap::default()),
        }
    }
}

/// Whether the `n`th execution (counting from 1) is sampled: not one of the first, then one in `every`.
pub(crate) fn sample_due(n: u32, every: u32) -> bool {
    every != 0 && n > WARM_UP && n % every == 1 % every
}

impl Telemetry {
    #[inline]
    pub(crate) fn epoch(&self) -> u64 {
        self.epoch.load(Ordering::Relaxed)
    }

    /// Starts an execution of `id` on `subject`: the emitter it launches into, timed if this is one of the sampled executions.
    #[inline]
    pub(crate) fn begin<'a>(&self, ctx: &'a Context, subject: &Subject<'_>, id: StrategyId, host: bool) -> Begun<'a> {
        let every = ctx.tuning().telemetry_every;
        if every == 0 {
            return Begun { emit: Emit::now(ctx), timer: None };
        }
        let n = self.counters[id.family().index()][usize::from(id.1)].fetch_add(1, Ordering::Relaxed).wrapping_add(1);
        if !sample_due(n, every) {
            return Begun { emit: Emit::now(ctx), timer: None };
        }
        self.begin_sampled(ctx, subject, id, host)
    }

    #[cold]
    fn begin_sampled<'a>(&self, ctx: &'a Context, subject: &Subject<'_>, id: StrategyId, host: bool) -> Begun<'a> {
        let cell = CellKey { strategy: id, shape: ShapeClass::of(subject.dims()), numerics: ctx.numerics() };
        if host {
            return Begun { emit: Emit::now(ctx), timer: Some((cell, Instant::now())) };
        }
        // A launch captured into a recording is not run, and a backend that cannot time launches is never asked.
        if !ctx.caps().contains(Caps::TELEMETRY) || ctx.recording_here() {
            return Begun { emit: Emit::now(ctx), timer: None };
        }
        let seq = self.seq.fetch_add(1, Ordering::Relaxed) + 1;
        let mut inflight = lock(&self.inflight);
        if inflight.len() >= MAX_INFLIGHT {
            inflight.clear();
            self.pending.store(0, Ordering::Relaxed);
        }
        inflight.insert(seq, Pending { cell, ns: 0.0, parts: 0, expected: None });
        self.pending.fetch_add(1, Ordering::Relaxed);
        Begun { emit: Emit::sampled(ctx, SampleTag(seq << 8)), timer: None }
    }

    /// Ends an execution: `done` is whether it produced its output. A host strategy's time goes into its cell now; a launching
    /// one's is completed by [`Telemetry::absorb`] once every launch has reported.
    pub(crate) fn finish(&self, ctx: &Context, begun: Begun<'_>, done: bool) {
        if let Some((cell, start)) = begun.timer {
            if done {
                self.complete(ctx.tuning().telemetry_min, cell, start.elapsed().as_nanos() as f64);
            }
            return;
        }
        let Some((tag, launches)) = begun.emit.into_sample() else { return };
        let seq = SampleTag::seq(tag.0);
        let mut inflight = lock(&self.inflight);
        let Some(entry) = inflight.get_mut(&seq) else { return };
        if !done || launches == 0 {
            inflight.remove(&seq);
            self.pending.fetch_sub(1, Ordering::Relaxed);
            return;
        }
        entry.expected = Some(launches);
        if entry.parts >= launches {
            let entry = inflight.remove(&seq).expect("present");
            drop(inflight);
            self.pending.fetch_sub(1, Ordering::Relaxed);
            self.complete(ctx.tuning().telemetry_min, entry.cell, entry.ns);
        }
    }

    /// Forgets a sample that outgrew the launches an execution may time.
    pub(crate) fn discard(&self, tag: SampleTag) {
        if lock(&self.inflight).remove(&SampleTag::seq(tag.0)).is_some() {
            self.pending.fetch_sub(1, Ordering::Relaxed);
        }
    }

    /// Adds the timing of one launch of a sampled execution; completes the execution once all of its launches are in.
    pub(crate) fn absorb(&self, min_samples: u32, tag: u64, ns: f64) {
        let seq = SampleTag::seq(tag);
        let mut inflight = lock(&self.inflight);
        let Some(entry) = inflight.get_mut(&seq) else { return };
        entry.ns += ns;
        entry.parts += 1;
        if entry.expected.is_some_and(|expected| entry.parts >= expected) {
            let entry = inflight.remove(&seq).expect("present");
            drop(inflight);
            self.pending.fetch_sub(1, Ordering::Relaxed);
            self.complete(min_samples, entry.cell, entry.ns);
        }
    }

    pub(crate) fn complete(&self, min_samples: u32, cell: CellKey, ns: f64) {
        let filled = lock(&self.cells).entry(cell).or_insert(Cell { ring: [0.0; RING], filled: 0, next: 0 }).push(ns as f32);
        if u32::from(filled) == min_samples {
            self.epoch.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Takes the timings the backend has finished (never waits).
    pub(crate) fn drain(&self, ctx: &Context) {
        let mut done = Vec::new();
        ctx.hal().drain_launch_times(&mut done);
        let min = ctx.tuning().telemetry_min;
        for (tag, time) in done {
            self.absorb(min, tag, time.as_secs_f64() * 1e9);
        }
    }

    /// [`Telemetry::drain`] while samples are outstanding, every 16th call (from `Context::launch`).
    #[inline]
    pub(crate) fn drain_some(&self, ctx: &Context) {
        if self.pending.load(Ordering::Relaxed) != 0 && self.tick.fetch_add(1, Ordering::Relaxed).is_multiple_of(16) {
            self.drain(ctx);
        }
    }

    /// [`Telemetry::drain`] while samples are outstanding (from `flush_batch` and `sync`).
    #[inline]
    pub(crate) fn drain_pending(&self, ctx: &Context) {
        if self.pending.load(Ordering::Relaxed) != 0 {
            self.drain(ctx);
        }
    }

    /// The median time of the strategy on this shape class, once it has `min_samples` samples.
    pub(crate) fn measured(&self, cell: &CellKey, min_samples: u32) -> Option<f64> {
        let cells = lock(&self.cells);
        let cell = cells.get(cell).filter(|cell| u32::from(cell.filled) >= min_samples.max(1))?;
        Some(cell.median())
    }

    /// How many samples the cell has.
    pub(crate) fn samples(&self, cell: &CellKey) -> u32 {
        lock(&self.cells).get(cell).map_or(0, |cell| u32::from(cell.filled))
    }

    /// Whether this is an execution that runs a rival instead of the chosen strategy (`Tuning::explore_every`): the counter of the
    /// family's executions, so that one in `every` explores.
    pub(crate) fn explore_due(&self, family: FamilyTag, every: u32) -> bool {
        every != 0 && self.explores[family.index()].fetch_add(1, Ordering::Relaxed).wrapping_add(1).is_multiple_of(every)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtin::Builtin;
    use crate::context::{ContextOptions, RawArg};
    use crate::jit::structured::strategy::{Boundary, BoundaryAttrs};
    use crate::numerics::Numerics;
    use luma_core::BackendKind;

    fn cpu(every: u32) -> Context {
        Context::with_tuning(ContextOptions::default().backend(BackendKind::Cpu).calibrate(false), |tuning| tuning.telemetry_every = every).unwrap()
    }

    fn boundary() -> Boundary<'static> {
        #[track_caller]
        fn here() -> &'static std::panic::Location<'static> {
            std::panic::Location::caller()
        }
        Boundary { family: FamilyTag::Reduce, op: "test", location: here(), dims: [4096, 1, 1, 1], attrs: BoundaryAttrs::None, _borrow: std::marker::PhantomData }
    }

    /// The sampled execution of `id`: begins it until the clock says so (the first executions are warm-up).
    fn sampled<'a>(ctx: &'a Context, subject: &Subject<'_>, id: StrategyId) -> Begun<'a> {
        loop {
            let begun = ctx.jit().telemetry.begin(ctx, subject, id, false);
            if begun.emit.clone().into_sample().is_some() {
                return begun;
            }
        }
    }

    /// One launch of a built-in kernel (a fill of 64 words) through `emit`.
    fn launch_fill(emit: &mut Emit<'_>) {
        let ctx = emit.ctx();
        let core = ctx.builtin(Builtin::Fill, "u32").unwrap();
        let buffer = ctx.alloc_buffer(256, crate::context::STORAGE_USAGE).unwrap();
        emit.launch(&core, [1, 1, 1], [64, 1, 1], &[RawArg::write(&buffer, 256), RawArg::Uniform(bytemuck::bytes_of(&7u32))]).unwrap();
        ctx.recycle(buffer);
    }

    #[test]
    fn a_sampled_execution_of_several_launches_is_timed_as_their_sum_on_the_cpu_backend() {
        let ctx = cpu(1);
        let b = boundary();
        let subject = Subject::Boundary(&b);
        let id = StrategyId(FamilyTag::Reduce, 3);
        let mut begun = sampled(&ctx, &subject, id);
        for _ in 0..3 {
            launch_fill(&mut begun.emit);
        }
        ctx.jit().telemetry.finish(&ctx, begun, true);
        ctx.sync().unwrap();
        let cell = CellKey { strategy: id, shape: ShapeClass::of(subject.dims()), numerics: ctx.numerics() };
        assert_eq!(ctx.jit().telemetry.samples(&cell), 1, "one execution, one sample, however many launches");
        assert_eq!(ctx.jit().telemetry.pending.load(Ordering::Relaxed), 0);
    }

    /// The context of the backend the tests run on (`LUMA_BACKEND`), sampling every execution.
    fn default_backend() -> Context {
        Context::with_tuning(ContextOptions::from_env().calibrate(false), |tuning| tuning.telemetry_every = 1).unwrap()
    }

    #[test]
    fn an_armed_launch_reports_a_time() {
        let ctx = default_backend();
        if !ctx.caps().contains(Caps::TELEMETRY) {
            println!("SKIPPED: {} cannot time launches", ctx.device_name());
            return;
        }
        let b = boundary();
        let subject = Subject::Boundary(&b);
        let id = StrategyId(FamilyTag::Reduce, 2);
        let mut begun = sampled(&ctx, &subject, id);
        launch_fill(&mut begun.emit);
        launch_fill(&mut begun.emit);
        ctx.jit().telemetry.finish(&ctx, begun, true);
        // The timings arrive once the work has run; wgpu reports them from a later poll.
        let cell = CellKey { strategy: id, shape: ShapeClass::of(subject.dims()), numerics: ctx.numerics() };
        for _ in 0..400 {
            ctx.sync().unwrap();
            if ctx.jit().telemetry.samples(&cell) == 1 {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        assert_eq!(ctx.jit().telemetry.samples(&cell), 1, "{}: the sampled execution of two launches was never timed", ctx.device_name());
        assert!(ctx.jit().telemetry.measured(&cell, 1).is_some_and(|ns| ns > 0.0));
    }

    /// Host time of a launch that is armed for timing against one that is not: `cargo test --release --lib
    /// host_cost_of_an_armed_launch -- --ignored --nocapture`, on the backend `LUMA_BACKEND` names.
    #[test]
    #[ignore = "a measurement, not a check"]
    fn host_cost_of_an_armed_launch() {
        let ctx = default_backend();
        if !ctx.caps().contains(Caps::TELEMETRY) {
            println!("SKIPPED: {} cannot time launches", ctx.device_name());
            return;
        }
        let core = ctx.builtin(Builtin::Fill, "u32").unwrap();
        let buffer = ctx.alloc_buffer(256, crate::context::STORAGE_USAGE).unwrap();
        let args = [RawArg::write(&buffer, 256), RawArg::Uniform(bytemuck::bytes_of(&7u32))];
        let rounds = 400;
        let mut measure = |armed: bool| {
            let mut best = f64::MAX;
            for _ in 0..7 {
                ctx.sync().unwrap();
                let start = std::time::Instant::now();
                for _ in 0..rounds {
                    let mut emit = if armed { Emit::sampled(&ctx, SampleTag(1 << 8)) } else { Emit::now(&ctx) };
                    for _ in 0..8 {
                        emit.launch(&core, [1, 1, 1], [64, 1, 1], &args).unwrap();
                    }
                    ctx.flush().unwrap();
                }
                best = best.min(start.elapsed().as_secs_f64() * 1e9 / f64::from(rounds * 8));
                ctx.sync().unwrap();
            }
            best
        };
        let (plain, armed) = (measure(false), measure(true));
        println!("{}: {plain:.0} ns per launch, {armed:.0} ns armed: +{:.0} ns ({:.2} us once in 32 launches)", ctx.device_name(), armed - plain, (armed - plain) / 32.0 / 1e3);
        ctx.recycle(buffer);
    }

    #[test]
    fn more_than_eight_launches_discard_the_sample() {
        let ctx = cpu(1);
        let b = boundary();
        let subject = Subject::Boundary(&b);
        let id = StrategyId(FamilyTag::Reduce, 4);
        let mut begun = sampled(&ctx, &subject, id);
        for _ in 0..9 {
            launch_fill(&mut begun.emit);
        }
        assert_eq!(ctx.jit().telemetry.pending.load(Ordering::Relaxed), 0, "the ninth launch discards the sample");
        ctx.jit().telemetry.finish(&ctx, begun, true);
        ctx.sync().unwrap();
        let cell = CellKey { strategy: id, shape: ShapeClass::of(subject.dims()), numerics: ctx.numerics() };
        assert_eq!(ctx.jit().telemetry.samples(&cell), 0);
    }

    #[test]
    fn telemetry_off_is_inert() {
        let ctx = cpu(0);
        let b = boundary();
        let subject = Subject::Boundary(&b);
        let id = StrategyId(FamilyTag::Reduce, 5);
        for _ in 0..200 {
            let mut begun = ctx.jit().telemetry.begin(&ctx, &subject, id, false);
            launch_fill(&mut begun.emit);
            ctx.jit().telemetry.finish(&ctx, begun, true);
        }
        ctx.sync().unwrap();
        assert_eq!(ctx.jit().telemetry.counters[0][5].load(Ordering::Relaxed), 0, "not even the clock runs");
        assert_eq!(ctx.jit().telemetry.pending.load(Ordering::Relaxed), 0);
        assert!(lock(&ctx.jit().telemetry.cells).is_empty());
        assert_eq!(ctx.jit().telemetry.epoch(), 0);
    }

    fn key(id: u8, dims: [u32; 4]) -> CellKey {
        CellKey { strategy: StrategyId(FamilyTag::Reduce, id), shape: ShapeClass::of(dims), numerics: Numerics::default() }
    }

    #[test]
    fn one_launch_in_n_is_sampled_after_the_warm_up() {
        let every = 32;
        let sampled: Vec<u32> = (1..=200).filter(|&n| sample_due(n, every)).collect();
        assert_eq!(sampled, vec![33, 65, 97, 129, 161, 193]);
        assert!((1..=3).all(|n| !sample_due(n, 1)), "the first executions are never sampled");
        assert!((4..=20).all(|n| sample_due(n, 1)), "every = 1 samples every execution after the warm-up");
        assert!((1..=100).all(|n| !sample_due(n, 0)), "0 is off");
    }

    #[test]
    fn a_median_needs_min_samples() {
        let telemetry = Telemetry::default();
        let cell = key(1, [1 << 20, 1, 1, 1]);
        for ns in [100.0, 300.0, 200.0, 5000.0, 150.0, 250.0, 180.0] {
            telemetry.complete(8, cell, ns);
        }
        assert_eq!(telemetry.measured(&cell, 8), None, "seven samples are not enough");
        assert_eq!(telemetry.epoch(), 0);
        telemetry.complete(8, cell, 220.0);
        assert_eq!(telemetry.epoch(), 1, "the eighth sample invalidates the rankings");
        // 100 150 180 200 220 250 300 5000: the median element is the 5th (index 4) of the sorted samples.
        assert_eq!(telemetry.measured(&cell, 8), Some(220.0));
        assert_eq!(telemetry.measured(&key(1, [1 << 10, 1, 1, 1]), 8), None, "another shape class has its own cell");
    }

    #[test]
    fn an_execution_of_several_launches_is_timed_as_their_sum() {
        let telemetry = Telemetry::default();
        let cell = key(2, [4096, 1, 1, 1]);
        // Two launches of the execution with sequence number 7 report before it is over, one after.
        telemetry.inflight.lock().unwrap().insert(7, Pending { cell, ns: 0.0, parts: 0, expected: None });
        telemetry.pending.store(1, Ordering::Relaxed);
        telemetry.absorb(1, 7 << 8, 100.0);
        telemetry.absorb(1, (7 << 8) | 1, 250.0);
        assert_eq!(telemetry.samples(&cell), 0, "the number of launches is not known yet");
        telemetry.inflight.lock().unwrap().get_mut(&7).unwrap().expected = Some(3);
        telemetry.absorb(1, (7 << 8) | 2, 50.0);
        assert_eq!(telemetry.samples(&cell), 1);
        assert_eq!(telemetry.measured(&cell, 1), Some(400.0));
        assert_eq!(telemetry.pending.load(Ordering::Relaxed), 0);
        // A timing of an execution that was discarded is ignored.
        telemetry.absorb(1, 99 << 8, 5.0);
        assert_eq!(telemetry.samples(&cell), 1);
    }

    #[test]
    fn the_ring_keeps_the_last_sixteen_samples() {
        let telemetry = Telemetry::default();
        let cell = key(1, [64, 1, 1, 1]);
        for _ in 0..16 {
            telemetry.complete(8, cell, 1000.0);
        }
        for _ in 0..16 {
            telemetry.complete(8, cell, 10.0);
        }
        assert_eq!(telemetry.measured(&cell, 8), Some(10.0));
    }

    #[test]
    fn the_shape_class_is_coarse() {
        let a = ShapeClass::of([1000, 1, 1, 1]);
        let b = ShapeClass::of([1023, 1, 1, 1]);
        assert_eq!(a.log2, b.log2);
        assert_ne!(a, b, "1000 is a multiple of 4, 1023 is not");
        assert_eq!(ShapeClass::of([31, 32, 0, 1]).small, 0b1101);
    }
}
