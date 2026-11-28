//! `select`: which strategy of a family runs a subject, and `execute`, which runs it.
//!
//! 1. **Override.** `LUMA_STRATEGY=<family>:<id>[,..]` names the one strategy of a family that runs where it fits.
//! 2. **Filter**, in table order: capabilities, numerics, `fits`. The variant gate (below) applies to the strategy that ranks first.
//! 3. **One candidate left:** returned without calling `cost`.
//! 4. **Rank** a block of entries that share a `group` by cost when every member has one, else keep table order. A measured cost
//!    (see `telemetry`) reorders the members of one `BitClass` among the places that class holds, never across classes.
//! 5. The ranking is cached per context (`JitState::selections`); `fits` always runs (it makes the exact parameters).
//! 6. **Background compile and budget.** A non-baseline strategy is not taken while a kernel of its `warm` is compiling or its
//!    family is over `Tuning::variant_budget`: the call falls back **only to a candidate of the same `BitClass`**; with none, it
//!    blocks on the compile (the same call must not return different bits early and late in a process).

use std::cell::Cell;
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::Mutex;

use super::strategy::{table, BitClass, Boundary, FamilyTag, FitsInputs, NumClass, Output, Params, PassMode, ReduceParams, Residencies, ResClass, Strategy, StrategyId, Subject, FAMILY_COUNT, MAX_PER_FAMILY};
use super::telemetry::{CellKey, ShapeClass};
use crate::array::Failure;
use crate::caps::Caps;
use crate::context::Context;
use crate::error::LumaError;
use crate::gpu::hash::{FxHasher, FxMap};
use crate::gpu::lock;
use crate::numerics::Numerics;
use crate::profile::{Decision, Verdict};

/// Rankings kept per context before the cache is cleared whole.
const SELECTIONS_LIMIT: usize = 4096;

// -- Override --

/// `LUMA_STRATEGY`: the strategy forced per family, parsed once when the context opens.
#[derive(Clone, Copy, Default, PartialEq, Eq, Debug)]
pub(crate) struct StrategyOverride([Option<StrategyId>; FAMILY_COUNT]);

impl StrategyOverride {
    /// `family:id[,family:id...]`; an unknown family or id is an error.
    pub(crate) fn parse(text: &str) -> Result<StrategyOverride, LumaError> {
        let mut forced = StrategyOverride::default();
        for item in text.split(',').map(str::trim).filter(|item| !item.is_empty()) {
            let (family, name) = item.split_once(':').ok_or_else(|| LumaError::InvalidOption(format!("LUMA_STRATEGY: {item:?} is not <family>:<strategy>")))?;
            // `ext:<op>/<strategy>` names a custom op, which the context's registry holds (see `crate::ext`).
            if family.trim() == "ext" {
                continue;
            }
            let family = FamilyTag::parse(family.trim()).ok_or_else(|| LumaError::InvalidOption(format!("LUMA_STRATEGY: unknown family {:?} (known: reduce, sort, scan, scatter, contraction, line_solve, factor, generic)", family.trim())))?;
            if table(family).is_empty() {
                return Err(LumaError::InvalidOption(format!("LUMA_STRATEGY: {}: family not ported (it records its Decision only)", family.name())));
            }
            let id = StrategyId::parse(family, name.trim()).ok_or_else(|| {
                let known: Vec<&str> = table(family).iter().map(|strategy| strategy.name).collect();
                LumaError::InvalidOption(format!("LUMA_STRATEGY: unknown {} strategy {:?} (known: {})", family.name(), name.trim(), known.join(", ")))
            })?;
            forced.0[family.index()] = Some(id);
        }
        Ok(forced)
    }

    #[inline]
    pub(crate) fn of(&self, family: FamilyTag) -> Option<StrategyId> {
        self.0[family.index()]
    }
}

/// Warns once per strategy that a forced strategy did not fit.
fn warn_forced(id: StrategyId) {
    static WARNED: Mutex<Vec<StrategyId>> = Mutex::new(Vec::new());
    let mut warned = lock(&WARNED);
    if !warned.contains(&id) {
        warned.push(id);
        log::warn!("luma: LUMA_STRATEGY forces {}:{}, which does not fit this operation; the default selection runs", id.family().name(), id.name());
    }
}

// -- Admission --

/// Why a strategy was left out of a selection (the texts of `Verdict::Rejected`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Skip {
    None,
    Caps,
    Numerics,
    Shape,
    Compiling,
    Budget,
    Declined,
    ForcedNoFit,
    NotForced,
}

impl Skip {
    pub(crate) fn text(self) -> &'static str {
        match self {
            Skip::None => "",
            Skip::Caps => "caps",
            Skip::Numerics => "numerics",
            Skip::Shape => "shape",
            Skip::Compiling => "compiling",
            Skip::Budget => "variant budget",
            Skip::Declined => "declined",
            Skip::ForcedNoFit => "forced strategy does not fit",
            Skip::NotForced => "not the forced strategy",
        }
    }
}

fn numerics_allowed(class: NumClass, policy: &Numerics) -> bool {
    match class {
        NumClass::Exact => true,
        NumClass::Reassociating => policy.reassociate,
        NumClass::Reduced(features) => policy.reduced.contains(features),
    }
}

/// The parameters of `strategy` for `subject` if it applies: capabilities, numerics and `fits`.
#[inline]
pub(crate) fn admit(strategy: &Strategy, subject: &Subject<'_>, ctx: &Context, res: &Residencies) -> Result<Params, Skip> {
    if !ctx.caps().contains(strategy.needs) {
        return Err(Skip::Caps);
    }
    if strategy.numerics != NumClass::Exact && !numerics_allowed(strategy.numerics, &ctx.numerics()) && !subject.requests_reduced() {
        return Err(Skip::Numerics);
    }
    (strategy.fits)(subject, ctx, res).ok_or(Skip::Shape)
}

// -- The ranking cache --

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) struct SelKey {
    family: FamilyTag,
    /// The family signature: Reduce: op list, element type, packed or not, axis flag.
    sig: u64,
    shape: ShapeClass,
    res: ResClass,
    forced: u32,
}

/// What the cache keeps: the cost order of the fitting entries, valid while the same entries fit and no measurement changed.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Ranked {
    fit_mask: u32,
    order: [u8; MAX_PER_FAMILY],
    epoch: u64,
}

pub(crate) type SelectionCache = Mutex<FxMap<SelKey, Ranked>>;

fn signature(subject: &Subject<'_>) -> u64 {
    let mut hasher = FxHasher::default();
    match subject {
        Subject::Comp(reduced) => reduced.signature(&mut hasher),
        Subject::Contraction(view) => view.attrs.hash(&mut hasher),
        Subject::Form(form, _) => form.structure_hash().hash(&mut hasher),
        Subject::Boundary(boundary) => {
            boundary.op.hash(&mut hasher);
            boundary.attrs.hash(&mut hasher);
        }
    }
    hasher.finish()
}

// -- The selection cache --
//
// A call whose inputs to `fits` are the ones of an earlier call reuses that call's answer: the first strategy of the entries that
// fits, with its parameters. Everything a `fits` reads is one of
//
// * the subject: its shape and attributes (in the key); a contraction also reads the state of its operands (`FitsInputs::OPERANDS`),
//   which no key holds today: products are not cached (`CACHED`);
// * what is fixed when the context opens (capabilities, limits, tuning, numerics, budgets, `LUMA_STRATEGY`, the environment): not in
//   the key, but every key names the context it was made on (the generation below), so a context that forced another strategy
//   never shares an answer;
// * this thread's recording (`Context::can_run_on_host`, `recording_open`): while any thread has a recording open on the context its
//   selections do not use the cache (`SelState::gate`), so an answer from before a recording opened is never used inside it;
// * whether the device is profiling (`can_run_on_host`): bumps the generation (`enable_profiling`, `finish_profiling`), as does
//   `set_assume_busy` and a recording beginning or ending;
// * the life of other values (`FitsInputs::LIVE`: the reductions' partners): the family is not cached.
//
// Each strategy's `reads` says which of these its `fits` has; the union over the family's table is what `cacheable` and the tests
// check. The cache is a per-thread table of 64 slots, direct mapped, overwritten on collision: no lock, and the context's gate and
// generation are the only shared reads. It serves tables that are priority chains (no cost, every entry a baseline), where the
// first strategy that fits is the selection: a table that ranks by cost or gates variants selects through `select_in`, whose
// ranking has a cache of its own (`SelectionCache`).

/// `LUMA_SELCACHE`: `on` (default), `off`, or `verify`, which runs the full `fits` on every hit and panics on a different answer
/// (debug builds do that on 1 hit in 16 under `on`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum CacheMode {
    Off,
    On,
    Verify,
}

impl CacheMode {
    fn from_env() -> CacheMode {
        match std::env::var("LUMA_SELCACHE").as_deref() {
            Ok("off") => CacheMode::Off,
            Ok("verify") => CacheMode::Verify,
            Ok("on") | Err(_) => CacheMode::On,
            Ok(other) => {
                log::warn!("luma: LUMA_SELCACHE={other:?} is not off, on or verify; on");
                CacheMode::On
            }
        }
    }
}

/// Generations are told apart across contexts: each starts in a range of its own (the low 32 bits count its bumps).
static NEXT_GENERATION: AtomicU64 = AtomicU64::new(1 << 32);

/// Set in [`SelState::gate`] when the mode is not plain `on`: every lookup takes the slow path, which does what the mode says.
const GATE_MODE: u32 = 1 << 31;

/// What a context keeps of the selection cache.
#[derive(Debug)]
pub(crate) struct SelState {
    /// The generation that a global input of `fits` changing bumps (see the module comment).
    generation: AtomicU64,
    /// 0 where a lookup may use the cache: the mode is `on` and no thread has a recording open on the context. Else the number of
    /// recordings open, with [`GATE_MODE`] where the mode is `off` or `verify`.
    gate: AtomicU32,
    mode: CacheMode,
}

impl SelState {
    pub(crate) fn new() -> SelState {
        let mode = CacheMode::from_env();
        SelState {
            generation: AtomicU64::new(NEXT_GENERATION.fetch_add(1 << 32, Ordering::Relaxed)),
            gate: AtomicU32::new(if mode == CacheMode::On { 0 } else { GATE_MODE }),
            mode,
        }
    }

    /// An input of `fits` that the key does not hold changed: everything cached so far is stale.
    pub(crate) fn bump(&self) {
        self.generation.fetch_add(1, Ordering::Release);
    }

    /// A thread began a recording on the context.
    pub(crate) fn recording_opened(&self) {
        self.gate.fetch_add(1, Ordering::Relaxed);
        self.bump();
    }

    /// A thread's recording on the context ended.
    pub(crate) fn recording_closed(&self) {
        self.gate.fetch_sub(1, Ordering::Relaxed);
        self.bump();
    }

    /// Whether any thread has a recording open on the context: false means no thread has an entry in the recordings table for it.
    #[inline(always)]
    pub(crate) fn recordings_open(&self) -> bool {
        self.gate.load(Ordering::Relaxed) & !GATE_MODE != 0
    }

    /// How this context was told to cache (tests).
    #[cfg(test)]
    pub(crate) fn mode(&self) -> CacheMode {
        self.mode
    }
}

/// What each family's `fits` read, over its whole table.
static READS: [FitsInputs; FAMILY_COUNT] = [
    FitsInputs::union(&crate::reduce::REDUCE_STRATEGIES),
    FitsInputs::union(&super::families::sort::SORT_STRATEGIES),
    FitsInputs::union(&super::families::scan::SCAN_STRATEGIES),
    FitsInputs::union(&super::families::scatter::SCATTER_STRATEGIES),
    FitsInputs::NONE,
    FitsInputs::NONE,
    FitsInputs::NONE,
    FitsInputs::union(&super::contraction::CONTRACTION_STRATEGIES),
    FitsInputs::union(&super::families::line_solve::LINE_SOLVE_STRATEGIES),
    FitsInputs::union(&super::families::factor::FACTOR_STRATEGIES),
    FitsInputs::NONE,
    FitsInputs::NONE,
    FitsInputs::union(&super::families::conv::CONV_STRATEGIES),
    FitsInputs::union(&super::families::window::WINDOW_STRATEGIES),
    FitsInputs::union(&super::families::opaque::OPAQUE_STRATEGIES),
    FitsInputs::union(&super::families::generic::GENERIC_STRATEGIES),
];

/// Whether `family`'s selections are cached: its table is a priority chain whose `fits` are all keyed (none reads
/// [`FitsInputs::LIVE`]) and a lookup costs less than the live walk. Measured with `tools/hosttime` (CPU, `luma_self`, per call):
/// every boundary family gains from the cache (`cumsum_small` 5433 to 5330, `medfilt_small` 5492 to 5246); a product's live walk is
/// short on the CPU (one `fits` of the native product) and the cache cost more than it saved (`launchcost/matmul32` +0.84%), so
/// products are not cached. Reductions read `LIVE`.
static CACHED: [bool; FAMILY_COUNT] = [false, true, true, true, false, false, false, false, true, true, false, false, true, true, true, false];

#[inline(always)]
pub(crate) fn cacheable(family: FamilyTag) -> bool {
    CACHED[family.index()] && !READS[family.index()].contains(FitsInputs::LIVE)
}

const SLOTS: usize = 64;

/// One slot of the per-thread table: the key (`generation` and the words of [`Words`]) and what it answered.
struct Slot {
    /// The context and its generation; 0 for an empty slot (no generation is 0).
    generation: Cell<u64>,
    stage: Cell<u64>,
    d01: Cell<u64>,
    d23: Cell<u64>,
    k0: Cell<u64>,
    k1: Cell<u64>,
    k2: Cell<u64>,
    k3: Cell<u64>,
    found: Cell<(StrategyId, Params)>,
}

impl Slot {
    const fn empty() -> Slot {
        Slot {
            generation: Cell::new(0),
            stage: Cell::new(0),
            d01: Cell::new(0),
            d23: Cell::new(0),
            k0: Cell::new(0),
            k1: Cell::new(0),
            k2: Cell::new(0),
            k3: Cell::new(0),
            found: Cell::new((StrategyId(FamilyTag::Reduce, 0), Params::Reduce(ReduceParams { pass: PassMode::Auto, groups: 0 }))),
        }
    }
}

thread_local! {
    static PICKS: [Slot; SLOTS] = const { [const { Slot::empty() }; SLOTS] };
    /// Hits so far on this thread, for the sampled verification of debug builds.
    static HITS: Cell<u32> = const { Cell::new(0) };
    /// Whether the last [`first_fit`] of this thread was answered from the cache (tests).
    #[cfg(test)]
    static LAST_HIT: Cell<bool> = const { Cell::new(false) };
}

/// What a lookup is keyed on besides the generation, in seven words.
#[derive(Clone, Copy)]
struct Words {
    /// Where the entries asked about start (a stage of a family's table: this also says the family) and how many (above bit 48).
    stage: u64,
    /// The shape: a contraction's fourth extent, which is 1, holds the precision asked for and the state of its operands.
    d01: u64,
    d23: u64,
    /// The attributes of a boundary (`BoundaryAttrs::pack`).
    k: [u64; 4],
}

impl Words {
    #[inline(always)]
    fn new(entries: &[Strategy], dims: [u32; 4], k: [u64; 4]) -> Words {
        Words { stage: entries.as_ptr() as u64 | (entries.len() as u64) << 48, d01: u64::from(dims[0]) | u64::from(dims[1]) << 32, d23: u64::from(dims[2]) | u64::from(dims[3]) << 32, k }
    }

    #[inline(always)]
    fn slot(&self) -> usize {
        ((self.stage ^ self.d01 ^ self.d23.rotate_left(32)).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 58) as usize
    }
}

/// This thread's slots, without going through a closure that the compiler would leave out of line.
#[inline(always)]
fn slots() -> &'static [Slot; SLOTS] {
    // SAFETY: `PICKS` is const-initialised and has no destructor, so its storage lives as long as the thread and is never torn down
    // before it ends. The reference stays on the calling thread (`Slot` holds `Cell`s, so it is not `Sync`), and every access to a
    // slot is a `Cell::get` or `set` of one word: nothing borrows into a cell, so reentrancy cannot alias.
    PICKS.with(|slots| unsafe { &*std::ptr::from_ref(slots) })
}

/// The answer the slot of the key holds, if it holds that key's.
#[inline(always)]
fn probe(generation: u64, words: &Words) -> Option<(StrategyId, Params)> {
    let slot = &slots()[words.slot()];
    let differs = (slot.generation.get() ^ generation)
        | (slot.stage.get() ^ words.stage)
        | (slot.d01.get() ^ words.d01)
        | (slot.d23.get() ^ words.d23)
        | (slot.k0.get() ^ words.k[0])
        | (slot.k1.get() ^ words.k[1])
        | (slot.k2.get() ^ words.k[2])
        | (slot.k3.get() ^ words.k[3]);
    if differs == 0 { Some(slot.found.get()) } else { None }
}

/// Whether the last selection of this thread came from the cache (tests).
#[cfg(test)]
pub(crate) fn last_was_hit() -> bool {
    LAST_HIT.with(Cell::get)
}

/// The first strategy of `entries` (a table or a stage of one) that applies to `boundary`, with its parameters; the forced strategy of `LUMA_STRATEGY` first where it fits. The hit path, linear and short and inlined into the op (where the stage, the family's attributes and
/// often the shape are constants); every other case (a miss, a mode, a recording open) is the slow one, out of line.
#[inline(always)]
pub(crate) fn first_fit_boundary(ctx: &Context, boundary: &Boundary<'_>, entries: &'static [Strategy]) -> Option<(StrategyId, Params)> {
    let state = &ctx.jit().selcache;
    if state.gate.load(Ordering::Relaxed) == 0 {
        let words = Words::new(entries, boundary.dims, boundary.key);
        if let Some(found) = probe(state.generation.load(Ordering::Acquire), &words) {
            boundary.from_cache.set(true);
            #[cfg(test)]
            LAST_HIT.with(|hit| hit.set(true));
            if cfg!(debug_assertions) && sampled() {
                verify(ctx, &Subject::Boundary(boundary), entries, found);
            }
            return Some(found);
        }
    }
    boundary.from_cache.set(false);
    first_fit_slow(ctx, &Subject::Boundary(boundary), entries)
}

/// Everything but the hit: the full `fits` where the cache is not to be used (the mode is `off`, a recording is open, the family is
/// not cacheable), the check of a hit under `verify`, and the full `fits` of a miss, kept for the next call that asks the same.
#[inline(never)]
fn first_fit_slow(ctx: &Context, subject: &Subject<'_>, entries: &'static [Strategy]) -> Option<(StrategyId, Params)> {
    #[cfg(test)]
    LAST_HIT.with(|hit| hit.set(false));
    let state = &ctx.jit().selcache;
    #[cfg(test)]
    if crate::linalg::tuning_forced() {
        return first_fit_live(ctx, subject, entries);
    }
    let gate = state.gate.load(Ordering::Relaxed);
    if state.mode == CacheMode::Off || gate & !GATE_MODE != 0 || !cacheable(subject.family()) {
        return first_fit_live(ctx, subject, entries);
    }
    let generation = state.generation.load(Ordering::Acquire);
    let words = match subject {
        Subject::Boundary(boundary) => Words::new(entries, boundary.dims, boundary.key),
        Subject::Contraction(_) | Subject::Comp(_) | Subject::Form(..) => return first_fit_live(ctx, subject, entries),
    };
    if state.mode == CacheMode::Verify
        && let Some(found) = probe(generation, &words)
    {
        verify(ctx, subject, entries, found);
        #[cfg(test)]
        LAST_HIT.with(|hit| hit.set(true));
        if let Subject::Boundary(boundary) = subject {
            boundary.from_cache.set(true);
        }
        return Some(found);
    }
    let live = first_fit_live(ctx, subject, entries);
    if let Some(found) = live {
        let slot = &slots()[words.slot()];
        slot.generation.set(generation);
        slot.stage.set(words.stage);
        slot.d01.set(words.d01);
        slot.d23.set(words.d23);
        slot.k0.set(words.k[0]);
        slot.k1.set(words.k[1]);
        slot.k2.set(words.k[2]);
        slot.k3.set(words.k[3]);
        slot.found.set(found);
    }
    live
}

/// [`first_fit`] without the cache: the entries in order, each one's capabilities, numerics and `fits`.
pub(crate) fn first_fit_live(ctx: &Context, subject: &Subject<'_>, entries: &'static [Strategy]) -> Option<(StrategyId, Params)> {
    let res = &Residencies::DEVICE;
    if let Some(forced) = ctx.strategy_override(subject.family())
        && let Some(strategy) = entries.iter().find(|strategy| strategy.id == forced)
        && let Ok(params) = admit(strategy, subject, ctx, res)
    {
        return Some((forced, params));
    }
    // A boundary's table has no `lower` (the op runs the strategy it is given); a chain's entries without one are skipped.
    let lowered = !matches!(subject, Subject::Boundary(_));
    entries.iter().filter(|strategy| !lowered || strategy.lower.is_some()).find_map(|strategy| admit(strategy, subject, ctx, res).ok().map(|params| (strategy.id, params)))
}

/// Whether this hit runs the full `fits` too (debug builds: 1 hit in 16).
#[cold]
#[inline(never)]
fn sampled() -> bool {
    HITS.with(|hits| {
        let n = hits.get().wrapping_add(1);
        hits.set(n);
        n % 16 == 0
    })
}

#[cold]
#[inline(never)]
fn verify(ctx: &Context, subject: &Subject<'_>, entries: &'static [Strategy], cached: (StrategyId, Params)) {
    let live = first_fit_live(ctx, subject, entries);
    assert!(
        live == Some(cached),
        "LUMA_SELCACHE: a {} selection answered from the cache ({cached:?}) differs from the full `fits` ({live:?}) at {}: an input of `fits` is not in the key",
        subject.family().name(),
        subject.site().0,
    );
}

// -- Variants --

/// The kernels non-baseline strategies built, per family, against `Tuning::variant_budget`.
#[derive(Default, Debug)]
pub(crate) struct Variants {
    per_family: [AtomicU32; FAMILY_COUNT],
    seen: Mutex<FxMap<(FamilyTag, u64), ()>>,
    /// Strategies whose kernel failed to compile: not taken again in this context.
    failed: Mutex<FxMap<StrategyId, ()>>,
}

impl Variants {
    /// Counts the kernels `ids` of `family` that were not seen; `false` (counting nothing) if that would pass `budget`.
    fn admit(&self, family: FamilyTag, ids: &[u64], budget: u32) -> bool {
        let mut seen = lock(&self.seen);
        let new = ids.iter().filter(|&&id| !seen.contains_key(&(family, id))).count() as u32;
        if new == 0 {
            return true;
        }
        if self.per_family[family.index()].load(Ordering::Relaxed) + new > budget {
            return false;
        }
        for &id in ids {
            seen.insert((family, id), ());
        }
        self.per_family[family.index()].fetch_add(new, Ordering::Relaxed);
        true
    }

    /// Counts the kernels whatever the budget says (a strategy that was taken anyway because nothing of its bit class could stand in).
    fn count(&self, family: FamilyTag, ids: &[u64]) {
        let mut seen = lock(&self.seen);
        for &id in ids {
            if seen.insert((family, id), ()).is_none() {
                self.per_family[family.index()].fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    /// Kernels counted for `family`.
    #[cfg(test)]
    pub(crate) fn built(&self, family: FamilyTag) -> u32 {
        self.per_family[family.index()].load(Ordering::Relaxed)
    }

    pub(crate) fn mark_failed(&self, id: StrategyId) {
        lock(&self.failed).insert(id, ());
    }

    fn has_failed(&self, id: StrategyId) -> bool {
        let failed = lock(&self.failed);
        !failed.is_empty() && failed.contains_key(&id)
    }
}

/// Whether the non-baseline `strategy` may not be taken now: its kernels are still compiling, or the family is over its budget.
/// The kernel ids come with it so the caller can count them when it takes the strategy anyway.
fn gate(strategy: &Strategy, subject: &Subject<'_>, params: &Params, ctx: &Context) -> (Skip, Vec<u64>) {
    let jit = ctx.jit();
    if jit.variants.has_failed(strategy.id) {
        return (Skip::Declined, Vec::new());
    }
    let Ok(warm) = (strategy.warm)(subject, params, ctx) else { return (Skip::Declined, Vec::new()) };
    if warm.failed() {
        jit.variants.mark_failed(strategy.id);
        return (Skip::Declined, Vec::new());
    }
    let ids: Vec<u64> = warm.kernels.iter().map(|core| core.raw.id()).collect();
    if !jit.variants.admit(strategy.id.family(), &ids, ctx.tuning().variant_budget) {
        return (Skip::Budget, ids);
    }
    if !warm.ready() {
        return (Skip::Compiling, ids);
    }
    (Skip::None, ids)
}

// -- Selection --

/// A short list of candidates that stays on the stack.
#[derive(Clone, Copy)]
pub(crate) struct Rest {
    items: [Option<(StrategyId, Params)>; MAX_PER_FAMILY],
    len: u8,
}

impl Rest {
    const fn new() -> Rest {
        Rest { items: [None; MAX_PER_FAMILY], len: 0 }
    }

    fn push(&mut self, item: (StrategyId, Params)) {
        self.items[usize::from(self.len)] = Some(item);
        self.len += 1;
    }

    pub(crate) fn iter(&self) -> impl Iterator<Item = (StrategyId, Params)> + '_ {
        self.items[..usize::from(self.len)].iter().flatten().copied()
    }
}

/// What `select` decided, with what the Decision needs.
#[derive(Clone, Copy)]
pub(crate) struct Selection {
    pub(crate) id: StrategyId,
    pub(crate) params: Params,
    /// The other candidates in rank order, for `Lowered::Declined`.
    pub(crate) rest: Rest,
    /// Why each table entry was left out, by table index.
    skipped: [Skip; MAX_PER_FAMILY],
    /// The ranking of the entries that fit, by table index (tests).
    #[cfg(test)]
    order: [u8; MAX_PER_FAMILY],
    #[cfg(test)]
    fitted: u8,
}

/// The strategy of the table of `subject`'s family that runs it, with the candidates after it.
#[allow(dead_code)]
pub(crate) fn select(ctx: &Context, subject: &Subject<'_>, res: &Residencies) -> Option<Selection> {
    select_in(table(subject.family()), ctx, subject, res, ctx.strategy_override(subject.family()))
}

/// [`select`] over `entries`, with `forced` as the override (tests give their own table).
pub(crate) fn select_in(entries: &'static [Strategy], ctx: &Context, subject: &Subject<'_>, res: &Residencies, forced: Option<StrategyId>) -> Option<Selection> {
    let count = entries.len();
    debug_assert!(count <= MAX_PER_FAMILY, "a family's table has at most {MAX_PER_FAMILY} entries");
    let mut skipped = [Skip::None; MAX_PER_FAMILY];
    let mut params: [Option<Params>; MAX_PER_FAMILY] = [None; MAX_PER_FAMILY];
    let mut fit_mask = 0u32;
    let mut fitted = 0usize;
    let mut forced_at = None;

    // 1. The override: the one candidate where it fits.
    if let Some(id) = forced
        && let Some(at) = entries.iter().position(|strategy| strategy.id == id)
    {
        match admit(&entries[at], subject, ctx, res) {
            Ok(found) => {
                params[at] = Some(found);
                fit_mask = 1 << at;
                fitted = 1;
                skipped = [Skip::NotForced; MAX_PER_FAMILY];
                skipped[at] = Skip::None;
            }
            Err(_) => {
                warn_forced(id);
                skipped[at] = Skip::ForcedNoFit;
                forced_at = Some(at);
            }
        }
    }
    // 2. Filter.
    if fitted == 0 {
        for (at, strategy) in entries.iter().enumerate() {
            if forced_at == Some(at) {
                continue;
            }
            match admit(strategy, subject, ctx, res) {
                Ok(found) => {
                    params[at] = Some(found);
                    fit_mask |= 1 << at;
                    fitted += 1;
                }
                Err(why) => skipped[at] = why,
            }
        }
    }
    if fitted == 0 {
        return None;
    }

    // 3-5. One candidate, else the ranking.
    let mut order = [0u8; MAX_PER_FAMILY];
    if fitted == 1 {
        order[0] = fit_mask.trailing_zeros() as u8;
    } else {
        rank(entries, ctx, subject, res, &params, fit_mask, forced, &mut order);
    }

    // 6. The first candidate may not be one a call can wait for (see the module documentation).
    let mut at = usize::from(order[0]);
    let top = &entries[at];
    if !top.baseline {
        let found = params[at].expect("fits");
        let (why, ids) = gate(top, subject, &found, ctx);
        if why != Skip::None {
            let stand_in = order[1..fitted].iter().map(|&c| usize::from(c)).find(|&other| {
                let strategy = &entries[other];
                strategy.bits == top.bits && (strategy.baseline || gate(strategy, subject, &params[other].expect("fits"), ctx).0 == Skip::None)
            });
            match stand_in {
                Some(other) => {
                    skipped[at] = why;
                    at = other;
                }
                // Nothing of its class can stand in: the call blocks on the compile (and a refused kernel is built and counted).
                None if why == Skip::Budget => ctx.jit().variants.count(top.id.family(), &ids),
                None => {}
            }
        }
    }

    let mut rest = Rest::new();
    for &other in &order[..fitted] {
        let other = usize::from(other);
        if other != at && skipped[other] == Skip::None {
            rest.push((entries[other].id, params[other].expect("fits")));
        }
    }
    Some(Selection {
        id: entries[at].id,
        params: params[at].expect("fits"),
        rest,
        skipped,
        #[cfg(test)]
        order,
        #[cfg(test)]
        fitted: fitted as u8,
    })
}

/// Ranks the entries of `fit_mask` into `order`: blocks of one `group` by cost, groups in table order.
#[allow(clippy::too_many_arguments)]
fn rank(entries: &[Strategy], ctx: &Context, subject: &Subject<'_>, res: &Residencies, params: &[Option<Params>; MAX_PER_FAMILY], fit_mask: u32, forced: Option<StrategyId>, order: &mut [u8; MAX_PER_FAMILY]) {
    let fitting: Vec<usize> = (0..entries.len()).filter(|&at| fit_mask >> at & 1 == 1).collect();
    // Nothing to learn from costs unless some block of at least two entries has them all: table order, no cache.
    let costed = fitting.chunk_by(|&a, &b| entries[a].group == entries[b].group).any(|block| block.len() > 1 && block.iter().all(|&at| entries[at].cost.is_some()));
    if !costed {
        for (slot, &at) in fitting.iter().enumerate() {
            order[slot] = at as u8;
        }
        return;
    }
    let telemetry = &ctx.jit().telemetry;
    let key = SelKey {
        family: subject.family(),
        sig: signature(subject),
        shape: ShapeClass::of(subject.dims()),
        res: res.class,
        forced: forced.map_or(0, |id| u32::from(id.1) + 1),
    };
    let epoch = telemetry.epoch();
    if let Some(ranked) = lock(&ctx.jit().selections).get(&key)
        && ranked.fit_mask == fit_mask
        && ranked.epoch == epoch
    {
        *order = ranked.order;
        return;
    }
    let min = ctx.tuning().telemetry_min;
    let numerics = ctx.numerics();
    let shape = key.shape;
    let mut slot = 0;
    let mut start = 0;
    while start < fitting.len() {
        let group = entries[fitting[start]].group;
        let end = start + fitting[start..].iter().take_while(|&&at| entries[at].group == group).count();
        let block = &fitting[start..end];
        let mut ranked: Vec<usize> = block.to_vec();
        if block.len() > 1 && block.iter().all(|&at| entries[at].cost.is_some()) {
            let statics: Vec<f64> = block.iter().map(|&at| (entries[at].cost.expect("checked"))(subject, &params[at].expect("fits"), ctx, res).ns).collect();
            let mut by_cost: Vec<usize> = (0..block.len()).collect();
            by_cost.sort_by(|&a, &b| statics[a].total_cmp(&statics[b]));
            ranked = by_cost.iter().map(|&i| block[i]).collect();
            // Measured costs reorder the members of one bit class among the places the class holds.
            let classes: Vec<BitClass> = {
                let mut classes: Vec<BitClass> = Vec::new();
                for &at in &ranked {
                    if !classes.contains(&entries[at].bits) {
                        classes.push(entries[at].bits);
                    }
                }
                classes
            };
            for class in classes {
                let places: Vec<usize> = (0..ranked.len()).filter(|&p| entries[ranked[p]].bits == class).collect();
                let members: Vec<(usize, f64)> = places
                    .iter()
                    .filter_map(|&p| {
                        let at = ranked[p];
                        telemetry.measured(&CellKey { strategy: entries[at].id, shape, numerics }, min).map(|ns| (at, ns))
                    })
                    .collect();
                if members.len() == places.len() && members.len() > 1 {
                    let mut sorted = members;
                    sorted.sort_by(|a, b| a.1.total_cmp(&b.1));
                    for (&place, (at, _)) in places.iter().zip(sorted) {
                        ranked[place] = at;
                    }
                }
            }
        }
        for at in ranked {
            order[slot] = at as u8;
            slot += 1;
        }
        start = end;
    }
    let mut cache = lock(&ctx.jit().selections);
    if cache.len() >= SELECTIONS_LIMIT {
        cache.clear();
    }
    cache.insert(key, Ranked { fit_mask, order: *order, epoch });
}

// -- Execution --

/// Selects a strategy for `subject` and lowers it, taking the next candidate while a `lower` declines; records the Decision.
pub(crate) fn execute(ctx: &Context, subject: &Subject<'_>) -> Result<Output, Failure> {
    let family = subject.family();
    // A table that is a priority chain apart from kernels only a device with the backend's own tiles has (`needs` them, so their `fits`
    // never run elsewhere) is a plain chain on every other device.
    let plain = PLAIN[family.index()] || PLAIN_WITHOUT_TILES[family.index()] && !ctx.caps().contains(Caps::GEMM_TILES);
    let forced = ctx.strategy_override(family);
    if super::contraction::is_half(subject)
        && let Some(done) = execute_half(ctx, subject, forced)
    {
        return done;
    }
    // Where the operand spans several segments and the context splits, the one strategy for it (see `crate::jit::split`).
    if ctx.split_mode() != crate::jit::split::SplitMode::Off
        && let Some(done) = execute_streamed(ctx, subject)
    {
        return done;
    }
    execute_with(table(family), plain, ctx, subject, forced)
}

/// A reduction over an operand of several segments on a splitting context: the streamed strategy. `None` for every other subject.
#[cold]
#[inline(never)]
fn execute_streamed(ctx: &Context, subject: &Subject<'_>) -> Option<Result<Output, Failure>> {
    let Subject::Comp(reduced) = subject else { return None };
    reduced.streamed(ctx).then(|| execute_with_res(&crate::reduce::REDUCE_STREAMED, true, ctx, subject, None, &Residencies::STREAMED))
}

/// A Gram product (`ContractionAttrs::half`): the half-work strategy, which is in a table of its own so that a plain product never
/// asks its `fits`. `None` where it does not fit (or another strategy is forced): the product takes the main table.
#[cold]
#[inline(never)]
fn execute_half(ctx: &Context, subject: &Subject<'_>, forced: Option<StrategyId>) -> Option<Result<Output, Failure>> {
    let entries = &super::contraction::CONTRACTION_HALF;
    if forced.is_some_and(|forced| forced != entries[0].id) || admit(&entries[0], subject, ctx, &Residencies::DEVICE).is_err() {
        return None;
    }
    Some(execute_ranked(entries, ctx, subject, forced, &Residencies::DEVICE))
}

/// [`plain_without_tiles`] of every family's table, worked out once.
static PLAIN_WITHOUT_TILES: [bool; FAMILY_COUNT] = [false, false, false, false, false, false, false, plain_without_tiles(&super::contraction::CONTRACTION_STRATEGIES), false, false, false, false, false, false, false, false];

/// Whether the entries that are not plain (not a baseline, or ranked by cost) all need the capabilities of the backend's own tiles,
/// and the others are.
const fn plain_without_tiles(entries: &[Strategy]) -> bool {
    let tiles = Caps::TF32.with(Caps::GEMM_TILES).bits();
    let mut at = 0;
    while at < entries.len() {
        let strategy = &entries[at];
        if (strategy.cost.is_some() || !strategy.baseline) && strategy.needs.bits() & tiles == 0 {
            return false;
        }
        at += 1;
    }
    true
}

/// [`plain`] of every family's table, worked out once.
static PLAIN: [bool; FAMILY_COUNT] = [plain(&crate::reduce::REDUCE_STRATEGIES), plain(&super::families::sort::SORT_STRATEGIES), plain(&super::families::scan::SCAN_STRATEGIES), plain(&super::families::scatter::SCATTER_STRATEGIES), true, true, true, plain(&super::contraction::CONTRACTION_STRATEGIES), plain(&super::families::line_solve::LINE_SOLVE_STRATEGIES), plain(&super::families::factor::FACTOR_STRATEGIES), true, true, plain(&super::families::conv::CONV_STRATEGIES), plain(&super::families::window::WINDOW_STRATEGIES), plain(&super::families::opaque::OPAQUE_STRATEGIES), plain(&super::families::generic::GENERIC_STRATEGIES)];

/// Whether every entry is a baseline without a cost: selection is the priority chain.
const fn plain(entries: &[Strategy]) -> bool {
    let mut at = 0;
    while at < entries.len() {
        if entries[at].cost.is_some() || !entries[at].baseline {
            return false;
        }
        at += 1;
    }
    true
}

/// [`execute_in`] for a table that is a priority chain (see [`plain`]), when no one listens to the Decision: the entries are
/// tried in table order, each one's `fits` evaluated when it is its turn, which is what the `if` chains it replaces did. Nothing
/// is ranked or cached (a product's live `fits` walk costs no more than a lookup: see [`CACHED`]).
fn execute_plain(entries: &[Strategy], ctx: &Context, subject: &Subject<'_>, res: &Residencies) -> Result<Output, Failure> {
    use super::strategy::Lowered;
    let telemetry = &ctx.jit().telemetry;
    for strategy in entries {
        let Some(lower) = strategy.lower else { continue };
        let Ok(params) = admit(strategy, subject, ctx, res) else { continue };
        let mut begun = telemetry.begin(ctx, subject, strategy.id, strategy.needs.contains(Caps::HOST_PATH));
        let result = lower(subject, &params, &mut begun.emit);
        telemetry.finish(ctx, begun, matches!(result, Ok(Lowered::Done(_))));
        if let Lowered::Done(output) = result? {
            return Ok(output);
        }
    }
    Err(LumaError::Unsupported(format!("no {} strategy ran {}", subject.family().name(), subject.site().0)).into())
}

#[cfg(test)]
pub(crate) fn execute_in(entries: &'static [Strategy], ctx: &Context, subject: &Subject<'_>, forced: Option<StrategyId>) -> Result<Output, Failure> {
    execute_with(entries, plain(entries), ctx, subject, forced)
}

#[inline(always)]
fn execute_with(entries: &'static [Strategy], plain: bool, ctx: &Context, subject: &Subject<'_>, forced: Option<StrategyId>) -> Result<Output, Failure> {
    execute_with_res(entries, plain, ctx, subject, forced, &Residencies::DEVICE)
}

#[inline(always)]
fn execute_with_res(entries: &'static [Strategy], plain: bool, ctx: &Context, subject: &Subject<'_>, forced: Option<StrategyId>, res: &Residencies) -> Result<Output, Failure> {
    if forced.is_none() && plain && !ctx.jit().recording_decisions() && ctx.tuning().explore_every == 0 {
        return execute_plain(entries, ctx, subject, res);
    }
    execute_ranked(entries, ctx, subject, forced, res)
}

/// [`execute_in`] through `select_in`. Kept out of line: reductions recurse through it (forcing a leaf forces another reduction),
/// so the frame of the plain path must not carry this one's locals.
#[inline(never)]
fn execute_ranked(entries: &'static [Strategy], ctx: &Context, subject: &Subject<'_>, forced: Option<StrategyId>, res: &Residencies) -> Result<Output, Failure> {
    use super::strategy::Lowered;
    let telemetry = &ctx.jit().telemetry;
    let mut forced = forced;
    let mut declined = [None; MAX_PER_FAMILY];
    let mut declined_len = 0;
    'select: loop {
        let Some(mut selection) = select_in(entries, ctx, subject, res, forced) else {
            return Err(LumaError::Unsupported(format!("no {} strategy fits {}", subject.family().name(), subject.site().0)).into());
        };
        let mut first = (selection.id, selection.params);
        let explore = ctx.tuning().explore_every;
        if explore != 0
            && telemetry.explore_due(subject.family(), explore)
            && let Some(rival) = rival(entries, ctx, subject, &selection)
        {
            selection.rest.push(first);
            first = rival;
        }
        for (id, params) in std::iter::once(first).chain(selection.rest.iter().filter(|&(id, _)| id != first.0)) {
            if declined[..declined_len].contains(&Some(id)) {
                continue;
            }
            let Some(strategy) = entries.iter().find(|strategy| strategy.id == id) else { continue };
            let Some(lower) = strategy.lower else { continue };
            // Only the first candidate was gated by `select`: a later one may still be compiling.
            if !strategy.baseline && id != selection.id && gate(strategy, subject, &params, ctx).0 != Skip::None {
                continue;
            }
            let mut begun = telemetry.begin(ctx, subject, id, strategy.needs.contains(Caps::HOST_PATH));
            let result = lower(subject, &params, &mut begun.emit);
            telemetry.finish(ctx, begun, matches!(result, Ok(Lowered::Done(_))));
            match result? {
                Lowered::Done(output) => {
                    if ctx.jit().recording_decisions() {
                        record(ctx, subject, entries, &selection, id, &params, &declined[..declined_len]);
                    }
                    return Ok(output);
                }
                Lowered::Declined => {
                    declined[declined_len] = Some(id);
                    declined_len += 1;
                    // A non-baseline strategy that declined because its kernel failed to compile is not tried again in this context.
                    if !strategy.baseline && (strategy.warm)(subject, &params, ctx).is_ok_and(|warm| warm.failed()) {
                        ctx.jit().variants.mark_failed(id);
                    }
                    if forced == Some(id) {
                        // The override declined: the default selection takes over.
                        forced = None;
                        continue 'select;
                    }
                }
            }
        }
        return Err(LumaError::Unsupported(format!("every {} strategy declined {}", subject.family().name(), subject.site().0)).into());
    }
}

/// A rival of the same bit class as the selected strategy that has fewer than `Tuning::telemetry_min` samples (`LUMA_EXPLORE`).
fn rival(entries: &[Strategy], ctx: &Context, subject: &Subject<'_>, selection: &Selection) -> Option<(StrategyId, Params)> {
    let min = ctx.tuning().telemetry_min;
    let shape = ShapeClass::of(subject.dims());
    let numerics = ctx.numerics();
    let bits = entries.iter().find(|strategy| strategy.id == selection.id)?.bits;
    selection.rest.iter().find(|&(id, _)| {
        entries.iter().find(|strategy| strategy.id == id).is_some_and(|strategy| strategy.baseline && strategy.bits == bits)
            && ctx.jit().telemetry.samples(&CellKey { strategy: id, shape, numerics }) < min
    })
}

/// The Decision of an execution: the strategy that ran, and why each other entry did not.
fn record(ctx: &Context, subject: &Subject<'_>, entries: &[Strategy], selection: &Selection, ran: StrategyId, params: &Params, declined: &[Option<StrategyId>]) {
    let (op, location) = subject.site();
    let ran_name = entries.iter().find(|strategy| strategy.id == ran).map_or("?", |strategy| strategy.name);
    let mut decision = Decision::new(op, location, subject.decision_site(), params.label_of(ran_name, subject, ctx));
    let res = &Residencies::DEVICE;
    for (at, strategy) in entries.iter().enumerate() {
        if strategy.id == ran {
            continue;
        }
        let verdict = if declined.contains(&Some(strategy.id)) {
            Verdict::Rejected("declined")
        } else if selection.skipped[at] != Skip::None {
            Verdict::Rejected(selection.skipped[at].text())
        } else if let (Some(cost), Ok(found)) = (strategy.cost, admit(strategy, subject, ctx, res)) {
            Verdict::Estimate { ns: cost(subject, &found, ctx, res).ns }
        } else {
            Verdict::Rejected("lower priority")
        };
        decision = decision.candidate(strategy.name, verdict);
    }
    ctx.jit().note_decision(decision);
}

/// Entries of the table that fit `subject` and the order they rank in (tests).
#[cfg(test)]
pub(crate) fn ranking(selection: &Selection) -> Vec<usize> {
    selection.order[..usize::from(selection.fitted)].iter().map(|&at| usize::from(at)).collect()
}

#[cfg(test)]
mod tests {
    use std::panic::Location;
    use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
    use std::sync::{Arc, OnceLock};
    use std::time::Duration;

    use luma_core::{BackendError, BackendKind, CompiledKernel, NativeKernel, RawKernel, Verdict};

    use super::*;
    use crate::array::Storage;
    use crate::context::ContextOptions;
    use crate::gpu_type::ElemLayout;
    use crate::jit::structured::emit::Emit;
    use crate::jit::structured::strategy::{Boundary, BoundaryAttrs, CostFn, Estimate, FitsFn, Lowered, LowerFn, ReduceParams, SplitClass, Warm, WarmFn};
    use crate::kernel::KernelCore;
    use crate::profile::DecisionSite;

    type Fits = FitsFn;
    type Cost = CostFn;
    type Lower = LowerFn;

    fn cpu() -> Context {
        Context::with_options(ContextOptions::default().backend(BackendKind::Cpu).calibrate(false)).unwrap()
    }

    fn cpu_with(tune: impl FnOnce(&mut crate::tuning::Tuning)) -> Context {
        Context::with_tuning(ContextOptions::default().backend(BackendKind::Cpu).calibrate(false), tune).unwrap()
    }

    #[track_caller]
    fn here() -> &'static Location<'static> {
        Location::caller()
    }

    fn boundary() -> Boundary<'static> {
        Boundary::new(FamilyTag::Reduce, "test", here(), [1 << 16, 1, 1, 1], BoundaryAttrs::None)
    }

    fn params() -> Params {
        Params::Reduce(ReduceParams { pass: super::super::strategy::PassMode::Auto, groups: 1 })
    }

    fn fits_always(_: &Subject<'_>, _: &Context, _: &Residencies) -> Option<Params> {
        Some(params())
    }

    fn fits_never(_: &Subject<'_>, _: &Context, _: &Residencies) -> Option<Params> {
        None
    }

    fn no_warm(_: &Subject<'_>, _: &Params, _: &Context) -> Result<Warm, Failure> {
        Ok(Warm { kernels: Vec::new() })
    }

    fn split(_: &Subject<'_>, _: &Params) -> SplitClass {
        SplitClass::Reduce
    }

    fn lower_done(_: &Subject<'_>, _: &Params, emit: &mut Emit<'_>) -> Result<Lowered, Failure> {
        Ok(Lowered::Done(Output::Storage(Arc::new(Storage::new(emit.ctx(), 1, ElemLayout::plain(4))))))
    }

    fn lower_declined(_: &Subject<'_>, _: &Params, _: &mut Emit<'_>) -> Result<Lowered, Failure> {
        Ok(Lowered::Declined)
    }

    #[allow(clippy::too_many_arguments)]
    const fn strat(id: u8, name: &'static str, bits: u16, baseline: bool, fits: Fits, cost: Option<Cost>, warm: WarmFn, lower: Lower) -> Strategy {
        Strategy {
            id: StrategyId(FamilyTag::Reduce, id),
            name,
            needs: Caps::empty(),
            numerics: NumClass::Exact,
            bits: BitClass(bits),
            baseline,
            group: 0,
            prologue: false,
            reads: FitsInputs::NONE,
            epilogue: false,
        partners: false,
            fits,
            cost,
            split,
            warm,
            lower: Some(lower),
        }
    }

    fn select_of(table: &'static [Strategy], ctx: &Context, forced: Option<StrategyId>) -> Selection {
        let b = boundary();
        select_in(table, ctx, &Subject::Boundary(&b), &Residencies::DEVICE, forced).expect("a strategy fits")
    }

    fn id(at: u8) -> StrategyId {
        StrategyId(FamilyTag::Reduce, at)
    }

    // -- the table of the reductions --

    #[test]
    fn the_reduce_table_is_well_formed() {
        let entries = table(FamilyTag::Reduce);
        assert!(entries.len() <= MAX_PER_FAMILY);
        for (at, strategy) in entries.iter().enumerate() {
            assert_eq!(usize::from(strategy.id.1), at, "{} sits at its id", strategy.name);
            assert_eq!(strategy.id.family(), FamilyTag::Reduce);
            assert!(strategy.baseline && strategy.cost.is_none() && strategy.lower.is_some());
            assert_eq!(StrategyId::parse(FamilyTag::Reduce, strategy.name), Some(strategy.id));
            assert_eq!(strategy.id.name(), strategy.name);
            assert_eq!(entries.iter().filter(|other| other.group == strategy.group).count(), 1, "every Reduce entry is its own group");
        }
        let names: Vec<&str> = entries.iter().map(|strategy| strategy.name).collect();
        assert_eq!(names, ["host", "merged", "axis-merged", "bits", "single-pass", "two-pass", "axis"]);
    }

    // -- select --

    fn cost_panics(_: &Subject<'_>, _: &Params, _: &Context, _: &Residencies) -> Estimate {
        panic!("cost was called")
    }

    static ONE_FITS: [Strategy; 2] = [
        strat(0, "a", 1, true, fits_always, Some(cost_panics), no_warm, lower_done),
        strat(1, "b", 1, true, fits_never, Some(cost_panics), no_warm, lower_done),
    ];

    #[test]
    fn a_single_fitting_strategy_is_returned_without_calling_cost() {
        let ctx = cpu();
        let selection = select_of(&ONE_FITS, &ctx, None);
        assert_eq!(selection.id, id(0));
        assert_eq!(selection.rest.iter().count(), 0);
    }

    static NO_COST: [Strategy; 3] = [
        strat(0, "a", 1, true, fits_always, None, no_warm, lower_done),
        strat(1, "b", 2, true, fits_always, None, no_warm, lower_done),
        strat(2, "c", 3, true, fits_always, None, no_warm, lower_done),
    ];

    #[test]
    fn candidates_with_no_cost_keep_table_order() {
        let ctx = cpu();
        let selection = select_of(&NO_COST, &ctx, None);
        assert_eq!(ranking(&selection), vec![0, 1, 2]);
        assert_eq!(selection.rest.iter().map(|(rest, _)| rest.1).collect::<Vec<_>>(), vec![1, 2]);
    }

    #[test]
    fn an_override_names_the_one_strategy() {
        let ctx = cpu();
        let selection = select_of(&NO_COST, &ctx, Some(id(1)));
        assert_eq!(selection.id, id(1));
        assert_eq!(selection.rest.iter().count(), 0, "the forced strategy is the only candidate");
        // The text of LUMA_STRATEGY names one strategy of a family.
        let forced = StrategyOverride::parse("reduce:two-pass").unwrap();
        assert_eq!(forced.of(FamilyTag::Reduce), Some(StrategyId::REDUCE_TWO_PASS));
        assert_eq!(StrategyOverride::parse(" reduce : host , ").unwrap().of(FamilyTag::Reduce), Some(StrategyId::REDUCE_HOST));
        assert_eq!(StrategyOverride::parse("").unwrap(), StrategyOverride::default());
    }

    static FORCED_NO_FIT: [Strategy; 2] = [
        strat(0, "a", 1, true, fits_always, None, no_warm, lower_done),
        strat(1, "b", 1, true, fits_never, None, no_warm, lower_done),
    ];

    #[test]
    fn an_override_that_does_not_fit_falls_back_and_is_recorded() {
        let ctx = cpu();
        ctx.record_decisions(true);
        let b = boundary();
        execute_in(&FORCED_NO_FIT, &ctx, &Subject::Boundary(&b), Some(id(1))).ok().expect("the default selection runs");
        let decisions = ctx.take_decisions();
        let [decision] = decisions.as_slice() else { panic!("one decision: {decisions:?}") };
        assert_eq!(decision.site, DecisionSite::ReducePlan);
        assert_eq!(decision.chosen, "a");
        assert_eq!(decision.candidates, vec![("b".into(), Verdict::Rejected("forced strategy does not fit"))]);
    }

    #[test]
    fn an_unknown_override_fails_context_creation() {
        let open = |text: &'static str| Context::with_options(ContextOptions::default().backend(BackendKind::Cpu).calibrate(false).strategy(text));
        for text in ["reduce:nope", "sort:merge", "reduce", "reduce:two-pass,scan:x"] {
            let error = open(text).err().unwrap_or_else(|| panic!("{text:?} opened"));
            assert!(matches!(error, LumaError::InvalidOption(_)), "{text:?}: {error:?}");
        }
        assert!(open("reduce:two-pass").is_ok());
    }

    static FLAG_C: AtomicBool = AtomicBool::new(false);
    static COST_CALLS: AtomicU32 = AtomicU32::new(0);

    fn fits_flag_c(_: &Subject<'_>, _: &Context, _: &Residencies) -> Option<Params> {
        FLAG_C.load(Ordering::Relaxed).then(params)
    }

    fn cost_a(_: &Subject<'_>, _: &Params, _: &Context, _: &Residencies) -> Estimate {
        COST_CALLS.fetch_add(1, Ordering::Relaxed);
        Estimate { ns: 300.0, measured: false }
    }

    fn cost_b(_: &Subject<'_>, _: &Params, _: &Context, _: &Residencies) -> Estimate {
        COST_CALLS.fetch_add(1, Ordering::Relaxed);
        Estimate { ns: 100.0, measured: false }
    }

    fn cost_c(_: &Subject<'_>, _: &Params, _: &Context, _: &Residencies) -> Estimate {
        COST_CALLS.fetch_add(1, Ordering::Relaxed);
        Estimate { ns: 200.0, measured: false }
    }

    static RANKED: [Strategy; 3] = [
        strat(0, "a", 1, true, fits_always, Some(cost_a), no_warm, lower_done),
        strat(1, "b", 1, true, fits_always, Some(cost_b), no_warm, lower_done),
        strat(2, "c", 1, true, fits_flag_c, Some(cost_c), no_warm, lower_done),
    ];

    #[test]
    fn a_changed_fit_set_recomputes_the_ranking() {
        let ctx = cpu();
        let first = select_of(&RANKED, &ctx, None);
        assert_eq!(ranking(&first), vec![1, 0], "by ascending cost");
        assert_eq!(first.id, id(1));
        assert_eq!(COST_CALLS.load(Ordering::Relaxed), 2);
        // The same entries fit: the cached ranking is used, `cost` is not called again.
        select_of(&RANKED, &ctx, None);
        assert_eq!(COST_CALLS.load(Ordering::Relaxed), 2);
        // Another entry fits now: ranked again.
        FLAG_C.store(true, Ordering::Relaxed);
        let third = select_of(&RANKED, &ctx, None);
        assert_eq!(ranking(&third), vec![1, 2, 0]);
        assert_eq!(COST_CALLS.load(Ordering::Relaxed), 5);
    }

    // -- measured costs --

    static FLAG_ONE_MEASURED: AtomicBool = AtomicBool::new(false);

    fn cost_100(_: &Subject<'_>, _: &Params, _: &Context, _: &Residencies) -> Estimate {
        Estimate { ns: 100.0, measured: false }
    }

    fn cost_150(_: &Subject<'_>, _: &Params, _: &Context, _: &Residencies) -> Estimate {
        Estimate { ns: 150.0, measured: false }
    }

    fn cost_200(_: &Subject<'_>, _: &Params, _: &Context, _: &Residencies) -> Estimate {
        Estimate { ns: 200.0, measured: false }
    }

    static CLASSES: [Strategy; 3] = [
        strat(0, "a", 1, true, fits_always, Some(cost_100), no_warm, lower_done),
        strat(1, "b", 2, true, fits_always, Some(cost_150), no_warm, lower_done),
        strat(2, "c", 1, true, fits_always, Some(cost_200), no_warm, lower_done),
    ];

    fn feed(ctx: &Context, strategy: u8, ns: f64) {
        let b = boundary();
        let cell = CellKey { strategy: id(strategy), shape: ShapeClass::of(Subject::Boundary(&b).dims()), numerics: ctx.numerics() };
        for _ in 0..ctx.tuning().telemetry_min {
            ctx.jit().telemetry.complete(ctx.tuning().telemetry_min, cell, ns);
        }
    }

    #[test]
    fn measured_costs_reorder_only_within_a_bit_class() {
        let _ = &FLAG_ONE_MEASURED;
        let ctx = cpu();
        assert_eq!(ranking(&select_of(&CLASSES, &ctx, None)), vec![0, 1, 2], "static costs");
        // `a` measured slow, `c` fast: both of class 1, so they trade places; `b` (class 2) keeps its place however it measures.
        feed(&ctx, 0, 5000.0);
        assert_eq!(ranking(&select_of(&CLASSES, &ctx, None)), vec![0, 1, 2], "one member of the class measured: static costs for the set");
        feed(&ctx, 2, 50.0);
        assert_eq!(ranking(&select_of(&CLASSES, &ctx, None)), vec![2, 1, 0], "the class is measured: its members reorder in their places");
        feed(&ctx, 1, 1.0);
        assert_eq!(ranking(&select_of(&CLASSES, &ctx, None)), vec![2, 1, 0], "a class of one never moves");
    }

    // -- background compile and budget --

    #[derive(Debug)]
    struct Dummy;
    impl NativeKernel for Dummy {}

    fn compiled(raw: RawKernel) -> Arc<KernelCore> {
        Arc::new(CompiledKernel { id: 0, label: "test".into(), raw, bindings: Vec::new(), declared: 0, used: None, workgroup: [1, 1, 1] })
    }

    static LATCH: AtomicBool = AtomicBool::new(false);

    /// A kernel whose compile waits on [`LATCH`].
    fn held() -> &'static Arc<KernelCore> {
        static KERNEL: OnceLock<Arc<KernelCore>> = OnceLock::new();
        KERNEL.get_or_init(|| {
            compiled(RawKernel::spawn(|| {
                for _ in 0..4000 {
                    if LATCH.load(Ordering::Acquire) {
                        return Ok(Dummy);
                    }
                    std::thread::sleep(Duration::from_millis(5));
                }
                Err(BackendError::Gpu("the latch was never released".into()))
            }))
        })
    }

    fn warm_held(_: &Subject<'_>, _: &Params, _: &Context) -> Result<Warm, Failure> {
        Ok(Warm { kernels: vec![Arc::clone(held())] })
    }

    static RAN: AtomicU32 = AtomicU32::new(0);

    fn lower_a(subject: &Subject<'_>, params: &Params, emit: &mut Emit<'_>) -> Result<Lowered, Failure> {
        // The strategy needs its kernel: this is where a launch waits for the compile.
        held().raw.resolve::<Dummy>().map_err(|error| LumaError::Gpu(error.to_string()))?;
        RAN.store(1, Ordering::Relaxed);
        lower_done(subject, params, emit)
    }

    fn lower_b(subject: &Subject<'_>, params: &Params, emit: &mut Emit<'_>) -> Result<Lowered, Failure> {
        RAN.store(2, Ordering::Relaxed);
        lower_done(subject, params, emit)
    }

    static SAME_CLASS: [Strategy; 2] = [
        strat(0, "new", 1, false, fits_always, Some(cost_100), warm_held, lower_a),
        strat(1, "old", 1, true, fits_always, Some(cost_200), no_warm, lower_b),
    ];

    #[test]
    fn a_non_baseline_strategy_is_skipped_until_its_kernel_is_ready() {
        let ctx = cpu();
        let first = select_of(&SAME_CLASS, &ctx, None);
        assert_eq!(first.id, id(1), "the compile is held: the baseline of the same class runs");
        assert_eq!(first.skipped[0], Skip::Compiling);
        assert_eq!(first.rest.iter().count(), 0, "a strategy that is compiling is not a fallback");
        // The ranking is not memoised: the call that finds the kernel ready takes the new strategy.
        LATCH.store(true, Ordering::Release);
        let start = std::time::Instant::now();
        while !held().raw.is_ready() {
            assert!(start.elapsed() < Duration::from_secs(20), "the kernel never compiled");
            std::thread::sleep(Duration::from_millis(2));
        }
        assert_eq!(select_of(&SAME_CLASS, &ctx, None).id, id(0));
    }

    static LATCH_2: AtomicBool = AtomicBool::new(false);

    fn held_2() -> &'static Arc<KernelCore> {
        static KERNEL: OnceLock<Arc<KernelCore>> = OnceLock::new();
        KERNEL.get_or_init(|| {
            compiled(RawKernel::spawn(|| {
                for _ in 0..4000 {
                    if LATCH_2.load(Ordering::Acquire) {
                        return Ok(Dummy);
                    }
                    std::thread::sleep(Duration::from_millis(5));
                }
                Err(BackendError::Gpu("the latch was never released".into()))
            }))
        })
    }

    fn warm_held_2(_: &Subject<'_>, _: &Params, _: &Context) -> Result<Warm, Failure> {
        Ok(Warm { kernels: vec![Arc::clone(held_2())] })
    }

    static RAN_2: AtomicU32 = AtomicU32::new(0);

    fn lower_new_2(subject: &Subject<'_>, params: &Params, emit: &mut Emit<'_>) -> Result<Lowered, Failure> {
        held_2().raw.resolve::<Dummy>().map_err(|error| LumaError::Gpu(error.to_string()))?;
        RAN_2.store(1, Ordering::Relaxed);
        lower_done(subject, params, emit)
    }

    fn lower_old_2(subject: &Subject<'_>, params: &Params, emit: &mut Emit<'_>) -> Result<Lowered, Failure> {
        RAN_2.store(2, Ordering::Relaxed);
        lower_done(subject, params, emit)
    }

    static OTHER_CLASS: [Strategy; 2] = [
        strat(0, "new", 1, false, fits_always, Some(cost_100), warm_held_2, lower_new_2),
        strat(1, "old", 2, true, fits_always, Some(cost_200), no_warm, lower_old_2),
    ];

    #[test]
    fn a_skipped_candidate_falls_back_only_to_the_same_bit_class() {
        let ctx = cpu();
        let b = boundary();
        let subject = Subject::Boundary(&b);
        // Cold: the compile is held. The baseline is of another class (other bits), so the call blocks on the compile.
        let release = std::thread::spawn(|| {
            std::thread::sleep(Duration::from_millis(150));
            LATCH_2.store(true, Ordering::Release);
        });
        let selection = select_in(&OTHER_CLASS, &ctx, &subject, &Residencies::DEVICE, None).unwrap();
        assert_eq!(selection.id, id(0), "no candidate of the class can stand in");
        let started = std::time::Instant::now();
        execute_in(&OTHER_CLASS, &ctx, &subject, None).ok().expect("runs");
        release.join().unwrap();
        assert_eq!(RAN_2.load(Ordering::Relaxed), 1, "the new strategy ran, after waiting for its kernel");
        let cold = RAN_2.swap(0, Ordering::Relaxed);
        // Warm: the same strategy, the same answer.
        execute_in(&OTHER_CLASS, &ctx, &subject, None).ok().expect("runs");
        assert_eq!(RAN_2.load(Ordering::Relaxed), cold, "identical cold and warm");
        assert!(started.elapsed() < Duration::from_secs(30));
    }

    fn ready_kernel(slot: &'static OnceLock<Arc<KernelCore>>) -> Arc<KernelCore> {
        Arc::clone(slot.get_or_init(|| compiled(RawKernel::new(Dummy))))
    }

    static K1: OnceLock<Arc<KernelCore>> = OnceLock::new();
    static K2: OnceLock<Arc<KernelCore>> = OnceLock::new();
    static FIT_A1: AtomicBool = AtomicBool::new(false);
    static FIT_A2: AtomicBool = AtomicBool::new(false);
    static FIT_OLD: AtomicBool = AtomicBool::new(false);

    fn fits_a1(_: &Subject<'_>, _: &Context, _: &Residencies) -> Option<Params> {
        FIT_A1.load(Ordering::Relaxed).then(params)
    }

    fn fits_a2(_: &Subject<'_>, _: &Context, _: &Residencies) -> Option<Params> {
        FIT_A2.load(Ordering::Relaxed).then(params)
    }

    fn fits_old(_: &Subject<'_>, _: &Context, _: &Residencies) -> Option<Params> {
        FIT_OLD.load(Ordering::Relaxed).then(params)
    }

    fn warm_k1(_: &Subject<'_>, _: &Params, _: &Context) -> Result<Warm, Failure> {
        Ok(Warm { kernels: vec![ready_kernel(&K1)] })
    }

    fn warm_k2(_: &Subject<'_>, _: &Params, _: &Context) -> Result<Warm, Failure> {
        Ok(Warm { kernels: vec![ready_kernel(&K2)] })
    }

    static BUDGETED: [Strategy; 3] = [
        strat(0, "a1", 1, false, fits_a1, Some(cost_100), warm_k1, lower_done),
        strat(1, "a2", 1, false, fits_a2, Some(cost_100), warm_k2, lower_done),
        strat(2, "old", 1, true, fits_old, Some(cost_200), no_warm, lower_done),
    ];

    #[test]
    fn the_variant_budget_refuses_new_kernels_but_not_baseline_ones() {
        let ctx = cpu_with(|tuning| tuning.variant_budget = 1);
        let family = FamilyTag::Reduce;
        FIT_OLD.store(true, Ordering::Relaxed);
        // The first kernel is within the budget.
        FIT_A1.store(true, Ordering::Relaxed);
        assert_eq!(select_of(&BUDGETED, &ctx, None).id, id(0));
        assert_eq!(ctx.jit().variants.built(family), 1);
        // A second kernel is over it: the baseline of the same class runs, which is never refused.
        FIT_A1.store(false, Ordering::Relaxed);
        FIT_A2.store(true, Ordering::Relaxed);
        let over = select_of(&BUDGETED, &ctx, None);
        assert_eq!(over.id, id(2));
        assert_eq!(over.skipped[1], Skip::Budget);
        assert_eq!(ctx.jit().variants.built(family), 1, "a refused kernel is not counted");
        // The first kernel was counted: it is still allowed.
        FIT_A1.store(true, Ordering::Relaxed);
        FIT_A2.store(false, Ordering::Relaxed);
        assert_eq!(select_of(&BUDGETED, &ctx, None).id, id(0));
        // Nothing of the class can stand in: the kernel is built anyway, and counted.
        FIT_A1.store(false, Ordering::Relaxed);
        FIT_A2.store(true, Ordering::Relaxed);
        FIT_OLD.store(false, Ordering::Relaxed);
        assert_eq!(select_of(&BUDGETED, &ctx, None).id, id(1));
        assert_eq!(ctx.jit().variants.built(family), 2);
    }

    static DECLINES: [Strategy; 3] = [
        strat(0, "a", 1, true, fits_always, None, no_warm, lower_declined),
        strat(1, "b", 2, true, fits_always, None, no_warm, lower_declined),
        strat(2, "c", 3, true, fits_always, None, no_warm, lower_done),
    ];

    #[test]
    fn a_declined_strategy_hands_over_to_the_next_candidate_and_is_recorded() {
        let ctx = cpu();
        let b = boundary();
        let subject = Subject::Boundary(&b);
        // Without a listener, and with one: the same walk down the table.
        execute_in(&DECLINES, &ctx, &subject, None).ok().expect("c runs");
        ctx.record_decisions(true);
        execute_in(&DECLINES, &ctx, &subject, None).ok().expect("c runs");
        let decisions = ctx.take_decisions();
        let [decision] = decisions.as_slice() else { panic!("{decisions:?}") };
        assert_eq!(decision.chosen, "c");
        assert_eq!(decision.candidates, vec![("a".into(), Verdict::Rejected("declined")), ("b".into(), Verdict::Rejected("declined"))]);
        // An override that declines hands over to the default selection.
        execute_in(&DECLINES, &ctx, &subject, Some(id(0))).ok().expect("c runs");
    }

    #[test]
    fn boundary_candidates_come_in_table_order_and_honour_the_override() {
        let ctx = cpu();
        let b = boundary();
        let all: Vec<u8> = super::super::strategy::candidates_in(&FORCED_NO_FIT, &ctx, &b).map(|(found, _)| found.1).collect();
        assert_eq!(all, vec![0]);
        let all: Vec<u8> = super::super::strategy::candidates_in(&NO_COST, &ctx, &b).map(|(found, _)| found.1).collect();
        assert_eq!(all, vec![0, 1, 2]);
        let forcing = Context::with_options(ContextOptions::default().backend(BackendKind::Cpu).calibrate(false).strategy("reduce:axis")).unwrap();
        // `axis` is a strategy of the real table that does not fit a boundary: the ordinary candidates come.
        let none: Vec<u8> = super::super::strategy::candidates_in(&NO_COST, &forcing, &b).map(|(found, _)| found.1).collect();
        assert_eq!(none, vec![0, 1, 2]);
    }
}
