//! Per-call-site laziness of library ops (W2-LAZY §5): whether a call of `matmul` (and, later, of the other families) starts its
//! launch at the call or when the value is used.
//!
//! A library op launches at its call unless its call site has *absorbed* an epilogue or taken a *partner* before (W3-U4, R4): a
//! GPU-bound product whose result is used late must start on the GPU while the host keeps recording, and one whose launch moves
//! with its consumer anyway gains nothing from starting early. Nothing in wave 2 absorbs or partners, so the default mode
//! ([`LazyMode::Never`]) sends every call down the eager path; `LUMA_LAZY=always` makes every call lazy (the whole test suite runs
//! both ways), and `LUMA_LAZY=site` reads the table below.
//!
//! The table is one word per slot, read and written without a lock: a race only costs a wrong answer (a launch a little early or
//! late), never a wrong result.

use std::panic::Location;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::tuning::Tuning;

/// `LUMA_LAZY`: when a library op launches. Removed before 0.6.0 with the other switches (`Site` is then hard-wired).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum LazyMode {
    /// Every call launches at the call (the wave-2 default).
    Never,
    /// A call is lazy where its site absorbed or partnered before, and for the probes of `Tuning::lazy_probe_every`.
    Site,
    /// Every call is lazy: the dev switch the suites run under.
    Always,
}

impl LazyMode {
    /// The mode `LUMA_LAZY` names.
    pub(crate) fn parse(text: &str) -> Option<LazyMode> {
        match text.trim() {
            "never" => Some(LazyMode::Never),
            "site" => Some(LazyMode::Site),
            "always" => Some(LazyMode::Always),
            _ => None,
        }
    }
}

/// What a call does.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Laziness {
    /// Launch at the call.
    Eager,
    /// Make a value that launches when it is used.
    Lazy,
}

/// Where a call's slot is, for [`Sites::note_forced_alone`]. [`SiteSlot::NONE`] for a call that has none (every mode but `Site`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct SiteSlot(u16);

impl SiteSlot {
    pub(crate) const NONE: SiteSlot = SiteSlot(u16::MAX);
}

/// Slots of the table; a power of two.
const SLOTS: usize = 1024;
/// A lazy site that was forced with nothing absorbed this many times goes back to eager.
const ALONE_LIMIT: u8 = 16;
/// The most calls between two probes (`calls` is a byte).
pub(crate) const MAX_PROBE_EVERY: u32 = 256;

/// One site's state, packed in a word: `tag:u32 | absorbed:u8 | partnered:u8 | alone:u8 | calls:u8` (low byte last).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct State(u64);

impl State {
    const fn fresh(tag: u32) -> State {
        State((tag as u64) << 32)
    }

    const fn tag(self) -> u32 {
        (self.0 >> 32) as u32
    }

    const fn byte(self, at: u32) -> u8 {
        (self.0 >> at) as u8
    }

    const fn with(self, at: u32, value: u8) -> State {
        State(self.0 & !(0xff << at) | (value as u64) << at)
    }

    const CALLS: u32 = 0;
    const ALONE: u32 = 8;
    const PARTNERED: u32 = 16;
    const ABSORBED: u32 = 24;

    const fn calls(self) -> u8 {
        self.byte(Self::CALLS)
    }

    const fn alone(self) -> u8 {
        self.byte(Self::ALONE)
    }

    /// Whether the site fused an epilogue or took a partner before: a lazy value has something to attach to there.
    const fn attached(self) -> bool {
        self.byte(Self::PARTNERED) as u16 + self.byte(Self::ABSORBED) as u16 > 0
    }
}

/// A call site hashed: its slot and the tag that tells two sites of one slot apart.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct SiteKey {
    slot: u16,
    tag: u32,
}

impl SiteKey {
    /// The key of a call site: `(file, line, column)` mixed, the slot from the high bits and the tag from the low ones.
    pub(crate) fn of(location: &'static Location<'static>) -> SiteKey {
        let file = location.file().as_ptr() as usize as u64;
        let place = (u64::from(location.line()) << 12) ^ u64::from(location.column());
        let mix = (file >> 3).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ place.wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
        let mix = (mix ^ (mix >> 29)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        SiteKey { slot: (mix >> 54) as u16, tag: mix as u32 }
    }
}

/// The per-site table of one context.
pub(crate) struct Sites {
    slots: [AtomicU64; SLOTS],
}

impl std::fmt::Debug for Sites {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Sites").field("slots", &SLOTS).finish_non_exhaustive()
    }
}

impl Default for Sites {
    fn default() -> Self {
        Sites { slots: std::array::from_fn(|_| AtomicU64::new(0)) }
    }
}

impl Sites {
    /// Whether the call at `location` launches now or when its value is used, and the slot to tell [`Sites::note_forced_alone`] about.
    ///
    /// A recording is always eager: a launch made for a loop body is part of the loop, and a lazy one would be recorded wherever it is
    /// first used. `narrow_n` (the call is one `GemmNarrowN` takes, which returns a lazy view and has no lazy form) is eager in every
    /// mode, `Always` included; it is only asked for in the modes that could be lazy. `Never` is eager and `Always` lazy whatever the
    /// slot says; `Site` is lazy where the slot absorbed or partnered, and for the probe of every `tuning.lazy_probe_every`-th call
    /// (counted before it is incremented, so the first call of a site probes).
    #[inline]
    pub(crate) fn decide(&self, location: &'static Location<'static>, tuning: &Tuning, recording: bool, narrow_n: impl FnOnce() -> bool) -> (Laziness, SiteSlot) {
        if recording || tuning.lazy == LazyMode::Never || narrow_n() {
            return (Laziness::Eager, SiteSlot::NONE);
        }
        match tuning.lazy {
            LazyMode::Always => (Laziness::Lazy, SiteSlot::NONE),
            _ => self.decide_site(SiteKey::of(location), tuning.lazy_probe_every),
        }
    }

    /// [`Sites::decide`] of `Site` mode for the slot `key` names.
    pub(crate) fn decide_site(&self, key: SiteKey, probe_every: u32) -> (Laziness, SiteSlot) {
        let cell = &self.slots[usize::from(key.slot)];
        let seen = State(cell.load(Ordering::Relaxed));
        // A slot of another site is reset: collisions cost speed, not correctness.
        let state = if seen.tag() == key.tag { seen } else { State::fresh(key.tag) };
        let probe = probe_every != 0 && u32::from(state.calls()) % probe_every == 0;
        let lazy = state.attached() || probe;
        cell.store(state.with(State::CALLS, state.calls().wrapping_add(1)).0, Ordering::Relaxed);
        (if lazy { Laziness::Lazy } else { Laziness::Eager }, SiteSlot(key.slot))
    }

    /// A lazy value of `slot` was forced with nothing absorbed into it: at [`ALONE_LIMIT`] of them the site goes back to eager.
    pub(crate) fn note_forced_alone(&self, slot: SiteSlot) {
        let Some(cell) = self.slots.get(usize::from(slot.0)) else { return };
        let state = State(cell.load(Ordering::Relaxed));
        let alone = state.alone().saturating_add(1);
        let next = if alone >= ALONE_LIMIT { state.with(State::ALONE, 0).with(State::ABSORBED, 0).with(State::PARTNERED, 0) } else { state.with(State::ALONE, alone) };
        cell.store(next.0, Ordering::Relaxed);
    }

    /// Marks the slot of `key` as one that absorbed `absorbed` epilogues and took `partnered` partners (what W3-U4 and R4 write).
    #[cfg(test)]
    pub(crate) fn set_attached(&self, key: SiteKey, absorbed: u8, partnered: u8) {
        let cell = &self.slots[usize::from(key.slot)];
        let seen = State(cell.load(Ordering::Relaxed));
        let state = if seen.tag() == key.tag { seen } else { State::fresh(key.tag) };
        cell.store(state.with(State::ABSORBED, absorbed).with(State::PARTNERED, partnered).0, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[track_caller]
    fn here() -> &'static Location<'static> {
        Location::caller()
    }

    fn tuned(mode: LazyMode, probe_every: u32) -> Tuning {
        Tuning { lazy: mode, lazy_probe_every: probe_every, ..Tuning::for_tests() }
    }

    const KEY: SiteKey = SiteKey { slot: 7, tag: 0xABCD };

    #[test]
    fn an_unseen_site_is_eager() {
        let sites = Sites::default();
        let (laziness, slot) = sites.decide(here(), &tuned(LazyMode::Site, 0), false, || false);
        assert_eq!(laziness, Laziness::Eager);
        assert_ne!(slot, SiteSlot::NONE);
    }

    #[test]
    fn a_site_with_a_partner_or_absorption_is_lazy() {
        let sites = Sites::default();
        let site = tuned(LazyMode::Site, 0);
        for (absorbed, partnered) in [(1, 0), (0, 1), (3, 2)] {
            sites.set_attached(KEY, absorbed, partnered);
            assert_eq!(sites.decide_site(KEY, site.lazy_probe_every).0, Laziness::Lazy, "absorbed {absorbed}, partnered {partnered}");
        }
        sites.set_attached(KEY, 0, 0);
        assert_eq!(sites.decide_site(KEY, 0).0, Laziness::Eager);
    }

    #[test]
    fn a_lazy_site_forced_alone_sixteen_times_returns_to_eager() {
        let sites = Sites::default();
        sites.set_attached(KEY, 1, 0);
        for round in 0..ALONE_LIMIT {
            let (laziness, slot) = sites.decide_site(KEY, 0);
            assert_eq!(laziness, Laziness::Lazy, "call {round}");
            sites.note_forced_alone(slot);
        }
        assert_eq!(sites.decide_site(KEY, 0).0, Laziness::Eager);
        // Reset for good: nothing is left to absorb again until a writer says so.
        assert_eq!(sites.decide_site(KEY, 0).0, Laziness::Eager);
    }

    #[test]
    fn a_site_forced_alone_fewer_times_stays_lazy() {
        let sites = Sites::default();
        sites.set_attached(KEY, 1, 0);
        for _ in 0..ALONE_LIMIT - 1 {
            let (_, slot) = sites.decide_site(KEY, 0);
            sites.note_forced_alone(slot);
        }
        assert_eq!(sites.decide_site(KEY, 0).0, Laziness::Lazy);
    }

    #[test]
    fn a_colliding_location_resets_the_slot() {
        let sites = Sites::default();
        sites.set_attached(KEY, 1, 1);
        assert_eq!(sites.decide_site(KEY, 0).0, Laziness::Lazy);
        // Another site of the same slot starts from nothing ...
        let other = SiteKey { slot: KEY.slot, tag: KEY.tag ^ 1 };
        assert_eq!(sites.decide_site(other, 0).0, Laziness::Eager);
        // ... and the first one has lost what it learned.
        assert_eq!(sites.decide_site(KEY, 0).0, Laziness::Eager);
    }

    #[test]
    fn never_and_always_override_the_slot() {
        let sites = Sites::default();
        let location = here();
        sites.set_attached(SiteKey::of(location), 1, 1);
        assert_eq!(sites.decide(location, &tuned(LazyMode::Never, 1), false, || false).0, Laziness::Eager);
        let fresh = Sites::default();
        assert_eq!(fresh.decide(location, &tuned(LazyMode::Always, 0), false, || false).0, Laziness::Lazy);
        // `Site` reads the slot.
        assert_eq!(sites.decide(location, &tuned(LazyMode::Site, 0), false, || false).0, Laziness::Lazy);
    }

    #[test]
    fn a_recording_is_always_eager() {
        let sites = Sites::default();
        let location = here();
        sites.set_attached(SiteKey::of(location), 1, 1);
        for mode in [LazyMode::Never, LazyMode::Site, LazyMode::Always] {
            assert_eq!(sites.decide(location, &tuned(mode, 1), true, || false).0, Laziness::Eager, "{mode:?}");
        }
    }

    #[test]
    fn a_narrow_n_product_is_eager_even_under_always() {
        let sites = Sites::default();
        let location = here();
        sites.set_attached(SiteKey::of(location), 1, 1);
        for mode in [LazyMode::Never, LazyMode::Site, LazyMode::Always] {
            assert_eq!(sites.decide(location, &tuned(mode, 1), false, || true).0, Laziness::Eager, "{mode:?}");
        }
        // The question is not asked where the answer cannot matter.
        let asked = std::cell::Cell::new(false);
        let _ = sites.decide(location, &tuned(LazyMode::Never, 0), false, || {
            asked.set(true);
            true
        });
        assert!(!asked.get());
    }

    /// The calls (counted from 1) of `calls` that are lazy at a fresh site with probes every `probe_every`.
    fn lazy_calls(probe_every: u32, calls: usize) -> Vec<usize> {
        let sites = Sites::default();
        (1..=calls).filter(|_| sites.decide_site(KEY, probe_every).0 == Laziness::Lazy).collect()
    }

    #[test]
    fn the_probe_fires_on_the_first_call_and_every_n_calls() {
        assert_eq!(lazy_calls(16, 40), [1, 17, 33]);
        assert_eq!(lazy_calls(0, 40), Vec::<usize>::new());
        assert_eq!(lazy_calls(1, 40).len(), 40);
    }

    #[test]
    fn the_call_counter_wraps_at_the_probe_period_of_256() {
        assert_eq!(lazy_calls(256, 600), [1, 257, 513]);
    }

    #[test]
    fn locations_of_different_lines_have_different_keys() {
        let (a, b) = (SiteKey::of(here()), SiteKey::of(here()));
        assert_ne!(a, b);
        let again = |location: &'static Location<'static>| SiteKey::of(location);
        let location = here();
        assert_eq!(again(location), again(location));
    }
}
