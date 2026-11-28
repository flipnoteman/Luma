//! The slab length `L` of a stage: how many elements one pass of the stage's steps covers.
//!
//! A slab lies in one segment of every array the stage binds (so `L` is a power of two that divides every segment length), starts at a
//! multiple of the device's offset alignment (so a slab binds by offset into its segment), and is sized by what the stage keeps live
//! per element and by what a launch costs. On a GPU the cost is flat above the L2 knee, so the choice is the cap [`SLAB_CAP_BYTES`]
//! per array or the segment, whichever is smaller; where the knee is measured (CPU, M1) the optimum is the largest slab whose live set
//! stays in half the last-level cache.

use std::borrow::Cow;

use luma_core::{CostModel, MemoryModel, Verdict};

/// The largest slab, per element of the widest array (`SLAB_CAP_BYTES / max stride` elements): a launch costs about as much as moving
/// 2 MB on the 3090, so bigger slabs gain nothing and only grow what a stage keeps live.
pub(super) const SLAB_CAP_BYTES: u64 = 256 << 20;

/// The fewest elements of a slab: a workgroup row.
const MIN_ELEMS: usize = 32;

/// What the planner knows of a stage when it picks `L`.
pub(super) struct Facts {
    /// Elements of the iteration space.
    pub(super) len: usize,
    /// Steps of the stage (a launch each per slab).
    pub(super) steps: usize,
    /// The shortest segment of the arrays bound by slab (`usize::MAX` if none).
    pub(super) seg_min: usize,
    /// The widest element, bytes (`ElemLayout::stride`).
    pub(super) max_stride: u64,
    /// The elements one offset alignment covers for the least dense array: slabs start on a multiple of it.
    pub(super) align_elems: usize,
    /// Bytes per element of the transients (one slab-sized buffer per colour).
    pub(super) transient_stride: u64,
    /// Flat reach of the widest strip beyond the body of a ghost read (0 without ghosts).
    pub(super) halo: usize,
    /// Bytes per element of the inputs read through a ghost.
    pub(super) ghost_stride: u64,
    /// The canonical block of a float reduction (1 without).
    pub(super) block: usize,
    /// A step that runs as one launch over the whole space (tiled, staged): the stage is one slab.
    pub(super) one_slab: bool,
    /// Every array the stage binds by slab has the segment length of the stage's space (no mixed element sizes).
    pub(super) uniform: bool,
}

/// What the planner reads from the context.
pub(super) struct Env {
    pub(super) cost: CostModel,
    pub(super) memory: Option<MemoryModel>,
    pub(super) budget: Option<u64>,
    pub(super) max_buffer_bytes: u64,
    /// `ContextOptions::slab_bytes` (0: none).
    pub(super) pinned_bytes: u64,
    /// `Context::set_slab_len`.
    pub(super) pinned_len: Option<usize>,
}

/// The chosen length and what the decision record says of it.
pub(super) struct Chosen {
    pub(super) len: usize,
    pub(super) chosen: String,
    pub(super) candidates: Vec<(Cow<'static, str>, Verdict)>,
}

fn pow2_floor(n: usize) -> usize {
    if n == 0 { 0 } else { 1 << n.ilog2() }
}

/// Which bound limits `L` from above, for the decision record.
fn upper_bound(f: &Facts, env: &Env, mem: usize) -> (usize, &'static str) {
    let stride = f.max_stride.max(1);
    let by_buffer = usize::try_from(env.max_buffer_bytes / stride).unwrap_or(usize::MAX);
    let by_cap = usize::try_from(SLAB_CAP_BYTES / stride).unwrap_or(usize::MAX);
    [(mem, "L_mem"), (f.seg_min, "seg_len"), (by_buffer, "max_buffer"), (by_cap, "cap"), (1 << 31, "u32")]
        .into_iter()
        .min_by_key(|&(bound, _)| bound)
        .map_or((1, "none"), |(bound, why)| (pow2_floor(bound), why))
}

/// `T(L)`, nanoseconds, of a stage run in slabs of `l` (see the module documentation).
fn estimate(f: &Facts, env: &Env, l: usize) -> f64 {
    let slabs = f.len.div_ceil(l) as f64;
    let launches = slabs * f.steps as f64 * env.cost.launch_us * 1e3;
    // What the stage reads and writes inside its slab: transients, once written, once read.
    let live = f.transient_stride as f64 * l as f64;
    let l2 = env.memory.and_then(|memory| memory.l2_bytes.map(|bytes| (bytes, memory)));
    let bw_slab = match l2 {
        Some((bytes, memory)) if live <= bytes as f64 / if memory.l2_exact { 2.0 } else { 4.0 } => memory.l2_gbs,
        _ => env.cost.bandwidth_gbs,
    };
    let inside = 2.0 * f.transient_stride as f64 * f.len as f64 / bw_slab.max(1e-9);
    let ghost = slabs * (2 * f.halo).min(l) as f64 * f.ghost_stride as f64 / env.cost.bandwidth_gbs.max(1e-9);
    launches + inside + ghost
}

/// Whether a stage over arrays whose shortest bound segment is `seg` elements and whose widest element is `max_stride` bytes will run
/// slabs shorter than `seg`: the kernel is planned (ghost or slots, see `Span`) before the stage is formed, so this is the part of
/// [`choose`] that does not depend on the other steps of the stage. A pin, the cap on a slab and the buffer limit shorten a slab; the
/// cost model only does so for stages whose transients fit the L2, which the planner does not predict here (such a stage runs the
/// kernel it was planned with, which is correct either way).
pub(crate) fn below_segment(ctx: &crate::context::Context, seg: usize, max_stride: u64) -> bool {
    let stride = max_stride.max(1);
    if let Some(l) = ctx.pinned_slab_len() {
        return pow2_floor(l.max(1)) < seg;
    }
    let bytes = ctx.slab_bytes();
    if bytes > 0 {
        return pow2_floor(usize::try_from(bytes / stride).unwrap_or(usize::MAX).max(1)) < seg;
    }
    let by_buffer = usize::try_from(ctx.max_buffer_bytes() / stride).unwrap_or(usize::MAX);
    let by_cap = usize::try_from(SLAB_CAP_BYTES / stride).unwrap_or(usize::MAX);
    pow2_floor(by_buffer.min(by_cap)) < seg
}

pub(super) fn choose(f: &Facts, env: &Env) -> Chosen {
    if f.one_slab || f.seg_min == usize::MAX {
        return Chosen { len: f.len.max(1), chosen: format!("L={} one launch", f.len), candidates: Vec::new() };
    }
    let mem = match env.budget {
        // The stage's own segments take the other half of the budget (no ring yet: window 1).
        Some(budget) if f.transient_stride > 0 => usize::try_from(budget / 2 / f.transient_stride).unwrap_or(usize::MAX),
        _ => usize::MAX,
    };
    let (max, bound) = upper_bound(f, env, mem);
    let floor = |n: usize| n.max(MIN_ELEMS).max(f.block).max(f.align_elems).next_power_of_two();
    let min = floor(4 * f.halo);
    let hard = floor(2 * f.halo);
    let ghost = |l: usize| if f.halo == 0 { 0.0 } else { (2 * f.halo).min(l) as f64 / l as f64 * 100.0 };
    let describe = |l: usize, note: &str| format!("L=2^{} slab_elems={l} ghost={:.1}%{note}", l.ilog2(), ghost(l));
    if let Some(l) = env.pinned_len {
        let l = pow2_floor(l.clamp(1, f.seg_min.max(1)));
        return Chosen { len: l, chosen: describe(l.max(1), " pinned"), candidates: Vec::new() };
    }
    if env.pinned_bytes > 0 {
        let wanted = pow2_floor(usize::try_from(env.pinned_bytes / f.max_stride.max(1)).unwrap_or(usize::MAX)).max(1);
        let l = wanted.max(hard.min(max)).min(max.max(1));
        return Chosen { len: l, chosen: describe(l, " pinned"), candidates: Vec::new() };
    }
    if max < min {
        // Either the segments are shorter than the slab the stage would like, or the ghosts would cost too much: the largest slab
        // there is. Below `hard` the ghost is bigger than the body, which the record says.
        let l = max.max(1);
        let note = if l < hard { " below the ghost bound" } else { " ghost over 50%" };
        return Chosen { len: l, chosen: describe(l, note), candidates: vec![(Cow::Borrowed(bound), Verdict::Rejected("L_min"))] };
    }
    let mut best: Option<(usize, f64)> = None;
    let mut all: Vec<(usize, f64)> = Vec::new();
    let mut l = min;
    while l <= max {
        let ns = estimate(f, env, l);
        all.push((l, ns));
        // Ties go to the larger slab.
        if best.is_none_or(|(_, b)| ns <= b) {
            best = Some((l, ns));
        }
        l *= 2;
    }
    let (chosen, _) = best.unwrap_or((max, 0.0));
    let mut candidates: Vec<(Cow<'static, str>, Verdict)> = all
        .iter()
        .filter(|&&(l, _)| l != chosen && (l == chosen / 2 || l == chosen * 2))
        .map(|&(l, ns)| (Cow::Owned(format!("L=2^{}", l.ilog2())), Verdict::Estimate { ns }))
        .collect();
    if chosen * 2 > max {
        candidates.push((Cow::Owned(format!("L=2^{}", (chosen * 2).ilog2())), Verdict::Rejected(bound)));
    }
    Chosen { len: chosen, chosen: describe(chosen, ""), candidates }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn facts() -> Facts {
        Facts { len: 1 << 24, steps: 4, seg_min: 1 << 22, max_stride: 4, align_elems: 64, transient_stride: 8, halo: 0, ghost_stride: 0, block: 1, one_slab: false, uniform: true }
    }

    fn env(memory: Option<MemoryModel>) -> Env {
        Env { cost: CostModel::TEST, memory, budget: None, max_buffer_bytes: 1 << 30, pinned_bytes: 0, pinned_len: None }
    }

    fn minimiser(f: &Facts, e: &Env) -> usize {
        let (max, _) = upper_bound(f, e, usize::MAX);
        let mut best = (0, f64::INFINITY);
        let mut l = f.align_elems.max(MIN_ELEMS).next_power_of_two();
        while l <= max {
            let ns = estimate(f, e, l);
            if ns <= best.1 {
                best = (l, ns);
            }
            l *= 2;
        }
        best.0
    }

    #[test]
    fn slab_choice_follows_the_model() {
        // A flat memory: launches decide, so the largest slab wins (ties go to the larger one too).
        let f = facts();
        assert_eq!(choose(&f, &env(None)).len, 1 << 22);
        assert_eq!(choose(&f, &env(None)).len, minimiser(&f, &env(None)));
        // A hand-made L2: the live set of a slab fits half of 1 MiB up to 2^15 elements of 8 bytes per element; the saving beats the
        // launches until the slab is smaller than that.
        let memory = MemoryModel::new(Some(1 << 20), true, 4000.0, None, [1.0; 3], 1.0, 1.0, 100.0);
        let e = env(Some(memory));
        let l = choose(&f, &e).len;
        assert_eq!(l, minimiser(&f, &e));
        assert!(l.is_power_of_two() && l >= 64 && l <= f.seg_min);
        // The bounds: segments, the buffer limit, the cap.
        let narrow = Facts { seg_min: 1 << 10, ..facts() };
        assert_eq!(choose(&narrow, &env(None)).len, 1 << 10);
        let wide = Facts { max_stride: 16, seg_min: 1 << 30, ..facts() };
        assert_eq!(choose(&wide, &env(None)).len, (SLAB_CAP_BYTES / 16) as usize);
    }

    #[test]
    fn pins_clamp_to_what_the_stage_allows() {
        let f = Facts { halo: 100, ..facts() };
        let hard = (2 * 100usize).next_power_of_two();
        let mut e = env(None);
        e.pinned_bytes = 4 * 8;
        assert_eq!(choose(&f, &e).len, hard, "clamped up to L_hard");
        e.pinned_bytes = 4 << 40;
        assert_eq!(choose(&f, &e).len, f.seg_min, "clamped down to the segment");
        e.pinned_bytes = 4096;
        assert_eq!(choose(&f, &e).len, 1024);
        e.pinned_len = Some(16);
        assert_eq!(choose(&f, &e).len, 16, "the test hook bypasses L_hard");
    }
}
