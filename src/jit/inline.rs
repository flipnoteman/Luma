//! Lowering of a program that reads sub-programs through views and gathers into code that
//! computes every (expression, index) pair once.
//!
//! A [`FusedKey`] keeps the expression behind a view as a sub-program `sub{k}(i)` that is called
//! at the index of every read. Expressions read through nested views are then recomputed once
//! per path that reaches them: a flux read at two faces by a residual that two slices read
//! computes its operations four times. [`inline`] instead unfolds the sub-programs into one list
//! of instructions in which every instruction has a *context*, the index it is computed at, and
//! two instructions that are equal (same operation, operands, context) are one. Contexts are
//! canonical: a slice of a slice is one view, so the same cell reached on different paths is the
//! same context and its expression is computed once.
//!
//! The lowering is a pure function of the structure of the key (no nodes, no buffers), so it runs
//! only when a kernel is built or judged (see `JitState::judgement`). [`judge`] weighs the
//! lowered program against storing a sub-program that is read at several indices with the
//! device's [`CostModel`]: the estimated time of one kernel that recomputes it at every read
//! against two kernels, the first writing it once, the second reading it back. The instruction
//! cap only bounds the size of the generated code.
//!
//! Where several cells of a workgroup read the same expression through translation views (see
//! `jit::staging`), the lowered program has a third form: the expression is computed once per
//! cell of the workgroup's region into workgroup memory, and the reads of it stay in the
//! lowered programs as `Shift` instructions of `Ref::Sub` operands that the generated code
//! turns into workgroup loads ([`lower_staged`]). [`judge`] weighs that against the other two.

use crate::gpu::hash::FxMap;
use crate::profile::SplitReason;
use crate::tuning::Tuning;

use super::cost::{kernel_ns, op_weight, CostModel};
use super::program::{FusedKey, Instr, Ref, Tail};
use super::staging::{self, Staging};
use super::{Edge, Gen, Op, OpKind, ShiftKind, ViewMap};

/// How the index of a context derives from its parent's: the read index of one `Shift`.
#[derive(Debug)]
pub(crate) struct Ctx {
    pub(crate) parent: u32,
    pub(crate) kind: ShiftKind,
    /// Operands 1 and 2 of the shift, lowered (`Linear`: amount and total scalars; `Take`: the
    /// index value, read at the parent context; unused otherwise).
    pub(crate) args: [Ref; 2],
}

/// One instruction of the lowered program; instruction `n` defines `t{n}`.
#[derive(Debug)]
pub(crate) struct Flat {
    pub(crate) op: Op,
    /// `Ref::Tmp(n)` is lowered instruction `n`; `Ref::Input(k)` is input `k` read at the index
    /// of context `at` (operand 0 of a `Shift`: at the index of context `child`). `Ref::Sub(s)`
    /// is operand 0 of the reads of a sub-program that is staged in workgroup memory only.
    pub(crate) args: [Ref; 3],
    /// The context whose index this instruction is computed at (0: the element itself).
    pub(crate) at: u32,
    /// `Shift` instructions only: the context operand 0 is read at.
    pub(crate) child: u32,
}

/// Where an instruction of a lowered program is computed.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum Region {
    /// For every cell.
    Top,
    /// Only where the mask of the branching `Select` (the lowered instruction) holds.
    Taken(u32),
    /// Only where the mask of the branching `Select` does not hold.
    Skipped(u32),
}

/// A key written out at every index its sub-programs are read at.
#[derive(Debug)]
pub(crate) struct Inlined {
    /// `ctxs[0]` is a placeholder for the element itself.
    pub(crate) ctxs: Vec<Ctx>,
    pub(crate) instrs: Vec<Flat>,
    /// For every instruction of the main program `key.instrs`, the lowered instruction holding its value.
    pub(crate) main: Vec<u32>,
    /// Per sub-program, at how many contexts it was lowered.
    pub(crate) count: Vec<u32>,
    /// Per lowered instruction, where it is computed (see [`branches`]).
    pub(crate) region: Vec<Region>,
    /// Per lowered instruction, whether it is a `Select` computed as `if`/`else`: operand 1 only
    /// where the mask holds, operand 2 only where it does not.
    pub(crate) branch: Vec<bool>,
    /// The operations of the lowered program in f32-operation units (see [`op_weight`]) per
    /// element, those of a branch weighted by the share of elements that compute it.
    pub(crate) ops: f64,
}

/// A key lowered for a workgroup-staged kernel: the main program and, in the order of
/// [`Staging::subs`], one program per staged sub-program, each lowered at its own cell.
#[derive(Debug)]
pub(crate) struct StagedPrograms {
    pub(crate) main: Inlined,
    pub(crate) subs: Vec<Inlined>,
}

/// How a kernel's source is generated from its key.
#[derive(Debug)]
pub(crate) enum Lowered {
    /// Sub-programs written out at every index they are read at.
    Plain(Inlined),
    /// Some sub-programs staged in workgroup memory (the key's `staging`).
    Staged(StagedPrograms),
}

/// What [`inline`] found.
#[derive(Debug)]
pub(crate) enum Verdict {
    Fits(Inlined),
    /// The lowered program has more instructions than allowed; evaluating sub-program `sub` first
    /// (reading its buffer instead of recomputing it) is what saves the most.
    TooBig { sub: u32 },
}

struct Lowering<'a> {
    subs: &'a [Vec<Instr>],
    /// Per sub-program, whether it is staged in workgroup memory: its reads are not unfolded.
    staged: &'a [bool],
    cap: usize,
    ctxs: Vec<Ctx>,
    context_ids: FxMap<(u32, ShiftKind, [Ref; 2]), u32>,
    instrs: Vec<Flat>,
    numbered: FxMap<(Op, [Ref; 3], u32, u32), u32>,
    /// The instruction holding the value of sub-program `s` at context `c`.
    instances: FxMap<(u32, u32), u32>,
    /// Per sub-program, at how many contexts it was lowered.
    count: Vec<u32>,
    /// A sub-program read from its evaluated buffer instead of being unfolded (for costing what
    /// evaluating it first leaves of the program): its reads are one load each.
    cut: Option<u32>,
    /// Set once more than `cap` instructions exist: nothing is lowered any further.
    over: bool,
}

impl<'a> Lowering<'a> {
    fn new(subs: &'a [Vec<Instr>], staged: &'a [bool], cap: usize) -> Lowering<'a> {
        let placeholder = Ctx { parent: 0, kind: ShiftKind::Dims { dims: [1; 4], shifts: [0; 4] }, args: [Ref::Unused; 2] };
        Lowering {
            subs,
            staged,
            cap,
            ctxs: vec![placeholder],
            context_ids: FxMap::default(),
            instrs: Vec::new(),
            numbered: FxMap::default(),
            instances: FxMap::default(),
            count: vec![0; subs.len()],
            cut: None,
            over: false,
        }
    }

    /// The lowered program once `main` (the values of the program's instructions) is known and
    /// `roots` (lowered instructions the code after it reads) are chosen. A read of a staged
    /// sub-program counts as 2 operations: an index and a workgroup load.
    fn finish(self, main: Vec<u32>, roots: Vec<u32>, cost: &CostModel) -> Inlined {
        let staged_reads = self.staged.iter().any(|&staged| staged);
        let found = branches(&self.instrs, &self.ctxs, &roots, cost, staged_reads);
        Inlined {
            ctxs: self.ctxs,
            instrs: self.instrs,
            main,
            count: self.count,
            region: found.region,
            branch: found.branch,
            ops: found.ops,
        }
    }
}

/// The weight of a lowered instruction in f32-operation units; with `staged_reads`, a read of a
/// sub-program (staged in workgroup memory) is an index and a workgroup load.
fn flat_weight(flat: &Flat, cost: &CostModel, staged_reads: bool) -> f64 {
    if staged_reads && matches!(flat.args[0], Ref::Sub(_)) { 2.0 } else { op_weight(flat.op, cost) }
}

/// What the least a branch must save, in f32-operation units per element, for a `Select` to be
/// computed as `if`/`else`.
const BRANCH_GAIN: f64 = 8.0;

/// The result of [`branches`].
struct Branches {
    region: Vec<Region>,
    branch: Vec<bool>,
    ops: f64,
}

/// The share of the elements of a program that a `Select` with the mask `mask` takes its
/// operand 1 at, as far as the program shows: the part of the destination a clamped view of a
/// smaller array covers (a region assigned into an array: the right-hand side read at the
/// coordinates of the whole array), else a half.
fn taken_share(flats: &[Flat], ctxs: &[Ctx], select: &Flat) -> f64 {
    let cells = |dims: [u32; 4]| dims.iter().map(|&d| f64::from(d)).product::<f64>();
    if let Ref::Tmp(value) = select.args[1]
        && let Some(taken) = flats.get(value as usize)
        && let Some(Ctx { kind: ShiftKind::View { map, .. }, .. }) = ctxs.get(taken.at as usize)
    {
        return (cells(map.src) / cells(map.out)).clamp(1.0 / 4096.0, 1.0);
    }
    0.5
}

/// Finds the selects of a lowered program that are better computed as `if`/`else` than as a
/// `select` of two values computed for every element, and what runs where.
///
/// A select qualifies if its mask depends on the coordinates only (generated ranges and the
/// scalars they are compared with: the elements that take a branch are neighbours, so a
/// warp mostly takes one) and the instructions only one of its branches reads (and that no
/// branch outside it, no context, no stored value reads) are worth at least [`BRANCH_GAIN`]
/// operations per element. Those instructions are computed inside the branch. The weighted
/// operation count of the program counts them by the share of elements that take it.
fn branches(flats: &[Flat], ctxs: &[Ctx], roots: &[u32], cost: &CostModel, staged_reads: bool) -> Branches {
    let n = flats.len();
    let weight: Vec<f64> = flats.iter().map(|flat| flat_weight(flat, cost, staged_reads)).collect();
    let mut users: Vec<Vec<(u32, usize)>> = vec![Vec::new(); n];
    let mut pinned = vec![false; n];
    for (at, flat) in flats.iter().enumerate() {
        for (slot, arg) in flat.args.iter().enumerate() {
            if let Ref::Tmp(t) = *arg
                && let Some(list) = users.get_mut(t as usize)
            {
                list.push((at as u32, slot));
            }
        }
    }
    for ctx in ctxs {
        for arg in ctx.args {
            if let Ref::Tmp(t) = arg
                && let Some(flag) = pinned.get_mut(t as usize)
            {
                *flag = true;
            }
        }
    }
    for &root in roots {
        if let Some(flag) = pinned.get_mut(root as usize) {
            *flag = true;
        }
    }
    // Values that depend on the coordinates and scalars only.
    let mut coordinates = vec![false; n];
    for (at, flat) in flats.iter().enumerate() {
        coordinates[at] = match flat.op.kind {
            OpKind::Generate(Gen::Range { .. }) => true,
            OpKind::Unary(_) | OpKind::Binary(_) | OpKind::Compare(_) | OpKind::Cast(_) | OpKind::Clamp | OpKind::Select => {
                flat.args.iter().take(flat.op.kind.arity()).all(|arg| match *arg {
                    Ref::Tmp(t) => coordinates.get(t as usize).copied().unwrap_or(false),
                    Ref::Scalar(_) | Ref::Unused => true,
                    _ => false,
                })
            }
            _ => false,
        };
    }
    let candidate: Vec<bool> = flats
        .iter()
        .map(|flat| {
            matches!(flat.op.kind, OpKind::Select)
                && matches!(flat.args[0], Ref::Tmp(mask) if coordinates.get(mask as usize).copied().unwrap_or(false))
                && (matches!(flat.args[1], Ref::Tmp(_)) || matches!(flat.args[2], Ref::Tmp(_)))
        })
        .collect();
    let share: Vec<f64> = flats.iter().map(|flat| if matches!(flat.op.kind, OpKind::Select) { taken_share(flats, ctxs, flat) } else { 1.0 }).collect();

    let mut branch = candidate.clone();
    loop {
        // The region of every instruction: the innermost one that holds all its uses.
        let mut region = vec![Region::Top; n];
        let parent = |region: &[Region], r: Region| match r {
            Region::Top => Region::Top,
            Region::Taken(s) | Region::Skipped(s) => region[s as usize],
        };
        let depth = |region: &[Region], mut r: Region| {
            let mut d = 0;
            while r != Region::Top {
                r = parent(region, r);
                d += 1;
            }
            d
        };
        for x in (0..n).rev() {
            if pinned[x] || users[x].is_empty() {
                continue;
            }
            let mut joint: Option<Region> = None;
            for &(user, slot) in &users[x] {
                let u = user as usize;
                let used_in = match (branch[u], slot) {
                    (true, 1) => Region::Taken(user),
                    (true, 2) => Region::Skipped(user),
                    _ => region[u],
                };
                joint = Some(match joint {
                    None => used_in,
                    Some(mut a) => {
                        let mut b = used_in;
                        let (mut da, mut db) = (depth(&region, a), depth(&region, b));
                        while a != b {
                            if da >= db {
                                a = parent(&region, a);
                                da -= 1;
                            } else {
                                b = parent(&region, b);
                                db -= 1;
                            }
                        }
                        a
                    }
                });
            }
            region[x] = joint.unwrap_or(Region::Top);
        }
        // The share of the elements that compute each region.
        let mut chance = vec![1.0f64; n];
        let region_chance = |chance: &[f64], r: Region| match r {
            Region::Top => 1.0,
            Region::Taken(s) => chance[s as usize] * share[s as usize],
            Region::Skipped(s) => chance[s as usize] * (1.0 - share[s as usize]),
        };
        // Selects come after their branches: an enclosing select has the higher index.
        for x in (0..n).rev() {
            chance[x] = region_chance(&chance, region[x]);
        }
        // What each branch saves: the work of the instructions it computes, run by every
        // element that reaches the select instead of the share that takes the branch.
        let mut saved = vec![0.0f64; n];
        for x in 0..n {
            let mut r = region[x];
            while r != Region::Top {
                let (s, here) = match r {
                    Region::Taken(s) => (s as usize, share[s as usize]),
                    Region::Skipped(s) => (s as usize, 1.0 - share[s as usize]),
                    Region::Top => break,
                };
                if here > 0.0 {
                    saved[s] += weight[x] * chance[x] * (1.0 - here) / here;
                }
                r = parent(&region, r);
            }
        }
        let useless: Vec<usize> = (0..n).filter(|&s| branch[s] && saved[s] < BRANCH_GAIN).collect();
        if useless.is_empty() {
            let ops = (0..n).map(|x| weight[x] * chance[x]).sum();
            return Branches { region, branch, ops };
        }
        for s in useless {
            branch[s] = false;
        }
    }
}

impl Lowering<'_> {
    /// Appends `flat`, or finds the equal instruction that exists already.
    fn push(&mut self, flat: Flat) -> u32 {
        let key = (flat.op, flat.args, flat.at, flat.child);
        if let Some(&found) = self.numbered.get(&key) {
            return found;
        }
        let at = u32::try_from(self.instrs.len()).unwrap_or(u32::MAX);
        self.instrs.push(flat);
        self.numbered.insert(key, at);
        self.over |= self.instrs.len() > self.cap;
        at
    }

    /// The context of a shift of kind `kind` (operands `args`) read at context `parent`. A view
    /// of a view without edge handling is the one view of both, and a shift by nothing is its
    /// parent.
    fn context(&mut self, mut parent: u32, mut kind: ShiftKind, args: [Ref; 2]) -> u32 {
        match kind {
            ShiftKind::View { mut map, edge: Edge::None } => {
                while parent != 0
                    && let Some(&Ctx { parent: grand, kind: ShiftKind::View { map: outer, edge: Edge::None }, .. }) = self.ctxs.get(parent as usize)
                    && outer.src == map.out
                {
                    let composed = map.then(&outer);
                    if !composed.fits() {
                        break;
                    }
                    (map, parent) = (composed, grand);
                }
                if map == ViewMap::identity(map.src) {
                    return parent;
                }
                kind = ShiftKind::View { map, edge: Edge::None };
            }
            ShiftKind::Dims { shifts: [0, 0, 0, 0], .. } => return parent,
            _ => {}
        }
        let key = (parent, kind, args);
        if let Some(&found) = self.context_ids.get(&key) {
            return found;
        }
        let id = u32::try_from(self.ctxs.len()).unwrap_or(u32::MAX);
        self.ctxs.push(Ctx { parent, kind, args });
        self.context_ids.insert(key, id);
        id
    }

    /// The value of sub-program `s` at context `ctx`, lowered once per pair.
    fn instance(&mut self, s: u32, ctx: u32) -> u32 {
        if let Some(&found) = self.instances.get(&(s, ctx)) {
            return found;
        }
        let subs = self.subs;
        let Some(program) = subs.get(s as usize) else { return 0 };
        if let Some(count) = self.count.get_mut(s as usize) {
            *count += 1;
        }
        let value = self.lower(program, ctx).last().copied().unwrap_or(0);
        self.instances.insert((s, ctx), value);
        value
    }

    /// Lowers `program` computed at context `ctx`: for every local `t{k}`, the lowered
    /// instruction holding its value.
    fn lower(&mut self, program: &[Instr], ctx: u32) -> Vec<u32> {
        let mut local: Vec<u32> = Vec::with_capacity(program.len());
        for instr in program {
            if self.over {
                local.push(0);
                continue;
            }
            let rewrite = |reference: Ref, local: &[u32]| match reference {
                Ref::Tmp(j) => Ref::Tmp(local.get(j as usize).copied().unwrap_or(0)),
                other => other,
            };
            let args = instr.args.map(|reference| rewrite(reference, &local));
            let value = match instr.op.kind {
                OpKind::Shift(kind) => {
                    let [source, b, c] = [instr.args[0], args[1], args[2]];
                    if let Ref::Sub(s) = source
                        && self.staged.get(s as usize).copied().unwrap_or(false)
                    {
                        // Read from workgroup memory at the cell the context stands for (a
                        // translation of the cell of the program, see `staging`): equal reads
                        // are one instruction.
                        let flat = Flat { op: instr.op, args: [source, b, Ref::Unused], at: ctx, child: 0 };
                        let value = self.push(flat);
                        local.push(value);
                        continue;
                    }
                    let context_args = match kind {
                        ShiftKind::Linear => [b, c],
                        ShiftKind::Take { .. } => [b, Ref::Unused],
                        ShiftKind::View { .. } | ShiftKind::Dims { .. } => [Ref::Unused; 2],
                    };
                    let child = self.context(ctx, kind, context_args);
                    match source {
                        Ref::Sub(s) if self.cut == Some(s) => self.push(Flat { op: instr.op, args: [source, b, Ref::Unused], at: ctx, child }),
                        Ref::Sub(s) => {
                            let value = self.instance(s, child);
                            if matches!(kind, ShiftKind::View { edge: Edge::Fill, .. }) {
                                self.push(Flat { op: instr.op, args: [Ref::Tmp(value), b, Ref::Unused], at: ctx, child })
                            } else {
                                value
                            }
                        }
                        other => self.push(Flat { op: instr.op, args: [rewrite(other, &local), b, c], at: ctx, child }),
                    }
                }
                _ => self.push(Flat { op: instr.op, args, at: ctx, child: 0 }),
            };
            local.push(value);
        }
        local
    }
}

/// The lowered instructions the code around the main program of `key` reads: what it stores,
/// scatters or reduces. `main` holds the lowered instruction of every program instruction.
fn key_roots(key: &FusedKey, main: &[u32]) -> Vec<u32> {
    let value = |at: u32| main.get(at as usize).copied();
    match key.tail {
        Tail::Store | Tail::StoreMapped(_) => key.outputs.iter().filter_map(|&at| value(at)).collect(),
        Tail::Scatter { .. } => key.outputs.iter().take(2).filter_map(|&at| value(at)).collect(),
        Tail::Reduce(_) => main.last().copied().into_iter().collect(),
    }
}

/// What evaluating sub-program `s` of `key` on its own costs per element: its operations with
/// the sub-programs it reads written out (see [`op_weight`]), and the bytes of the inputs it
/// loads. `None` past `cap` instructions.
fn alone(key: &FusedKey, s: usize, cap: usize, cost: &CostModel) -> Option<(f64, f64)> {
    let mut lowering = Lowering::new(&key.subs, &[], cap);
    let main = lowering.lower(key.subs.get(s)?, 0);
    if lowering.over {
        return None;
    }
    let roots = main.last().copied().into_iter().collect();
    let lowered = lowering.finish(main, roots, cost);
    let reads = lowered.instrs.iter().flat_map(|flat| flat.args).chain(lowered.ctxs.iter().flat_map(|ctx| ctx.args));
    Some((lowered.ops, input_bytes(key, reads)))
}

/// The operations per element (see [`op_weight`]) of `key` once sub-program `s` is read from its
/// evaluated buffer: what is left of the lowered program, with the sub-programs that `s` shares
/// with the rest still computed. `None` past `cap` instructions.
fn without(key: &FusedKey, s: u32, cap: usize, cost: &CostModel) -> Option<f64> {
    let mut lowering = Lowering::new(&key.subs, &[], cap);
    lowering.cut = Some(s);
    let main = lowering.lower(&key.instrs, 0);
    if lowering.over {
        return None;
    }
    let roots = key_roots(key, &main);
    Some(lowering.finish(main, roots, cost).ops)
}

/// Lowers `key` (main program at context 0), giving up once more than `cap` instructions exist.
pub(crate) fn inline(key: &FusedKey, cap: usize, cost: &CostModel) -> Verdict {
    let mut lowering = Lowering::new(&key.subs, &[], cap);
    let main = lowering.lower(&key.instrs, 0);
    if lowering.over {
        // Saves the most: the sub-program whose extra instances cost the most instructions;
        // without any, the longest.
        let sub = (0..key.subs.len())
            .max_by_key(|&s| {
                let length = key.subs[s].len() as u64;
                (u64::from(lowering.count[s].saturating_sub(1)) * length, length)
            })
            .unwrap_or(0);
        return Verdict::TooBig { sub: u32::try_from(sub).unwrap_or(u32::MAX) };
    }
    let roots = key_roots(key, &main);
    Verdict::Fits(lowering.finish(main, roots, cost))
}

/// Lowers `program` at its own cell (context 0), with the sub-programs `staged` left as reads
/// from workgroup memory. `roots` picks the lowered instructions the code after it reads from the
/// lowered instruction of each of the program's. `None` if more than `cap` instructions result.
fn lower_program(
    subs: &[Vec<Instr>],
    program: &[Instr],
    staged: &[bool],
    cap: usize,
    cost: &CostModel,
    roots: impl FnOnce(&[u32]) -> Vec<u32>,
) -> Option<Inlined> {
    let mut lowering = Lowering::new(subs, staged, cap);
    let main = lowering.lower(program, 0);
    if lowering.over {
        return None;
    }
    let roots = roots(&main);
    Some(lowering.finish(main, roots, cost))
}

/// Lowers the main program of `key` and every sub-program `staging` stages, each at its own
/// cell, with the reads of staged sub-programs left for workgroup memory. `None` if any
/// program has more than `cap` instructions.
pub(crate) fn lower_staged(key: &FusedKey, staging: &Staging, cap: usize, cost: &CostModel) -> Option<StagedPrograms> {
    let mut staged = vec![false; key.subs.len()];
    for entry in &staging.subs {
        *staged.get_mut(entry.sub as usize)? = true;
    }
    let main = lower_program(&key.subs, &key.instrs, &staged, cap, cost, |main| key_roots(key, main))?;
    let subs = staging
        .subs
        .iter()
        .map(|entry| {
            lower_program(&key.subs, key.subs.get(entry.sub as usize)?, &staged, cap, cost, |main| main.last().copied().into_iter().collect())
        })
        .collect::<Option<Vec<_>>>()?;
    Some(StagedPrograms { main, subs })
}

/// What [`judge`] decided.
#[derive(Debug)]
pub(crate) enum Judgement {
    /// Run the program as one kernel: lowered by [`inline`], or by [`lower_staged`] if `staging`
    /// is set (it becomes the `staging` of the key).
    Run { staging: Option<Staging> },
    /// Evaluate sub-program `sub` first (the kernel then reads its buffer) and flatten again.
    Materialise { sub: u32, reason: SplitReason },
}

/// The estimated times [`judge`] compared, in nanoseconds.
#[derive(Clone, Copy, Debug)]
#[cfg_attr(not(test), allow(dead_code))] // only tests read them (as the text of the judgement)
pub(crate) struct Estimates {
    /// One kernel running the lowered program (infinite if it exceeds the cap).
    pub(crate) inline_ns: f64,
    /// The best way of evaluating one sub-program first, with its two kernels' time.
    pub(crate) best_sub: Option<(u32, f64)>,
    /// One kernel staging the sub-programs read at several offsets in workgroup memory.
    pub(crate) staged_ns: Option<f64>,
}

/// Bytes per element of the distinct inputs among the operands `refs` read.
fn input_bytes(key: &FusedKey, refs: impl Iterator<Item = Ref>) -> f64 {
    let mut seen = vec![false; key.input_tys.len()];
    let mut bytes = 0.0;
    for reference in refs {
        if let Ref::Input(k) = reference
            && let Some(slot) = seen.get_mut(k as usize)
            && !*slot
        {
            *slot = true;
            bytes += key.input_tys[k as usize].bytes() as f64;
        }
    }
    bytes
}

/// Bytes per element the program writes: every output of a store, the value of a scatter,
/// nothing for a reduction (its partial sums are negligible).
fn output_bytes(key: &FusedKey) -> f64 {
    let bytes = |index: &u32| key.instrs.get(*index as usize).map_or(0.0, |instr| instr.op.out_ty().bytes() as f64);
    match key.tail {
        Tail::Store | Tail::StoreMapped(_) => key.outputs.iter().map(bytes).sum(),
        Tail::Scatter { .. } => key.outputs.first().map_or(0.0, bytes),
        Tail::Reduce(_) => 0.0,
    }
}

/// The number of elements of sub-program `s`, from the first shift that reads it (`len`, the
/// elements of the kernel, for a flat circular shift).
fn sub_len(key: &FusedKey, s: u32, len: usize) -> f64 {
    for instr in key.instrs.iter().chain(key.subs.iter().flatten()) {
        if let OpKind::Shift(kind) = instr.op.kind
            && instr.args[0] == Ref::Sub(s)
        {
            return match kind {
                ShiftKind::View { map, .. } => map.src.iter().map(|&d| f64::from(d)).product(),
                ShiftKind::Take { len } => f64::from(len),
                ShiftKind::Dims { dims, .. } => dims.iter().map(|&d| f64::from(d)).product(),
                ShiftKind::Linear => len as f64,
            };
        }
    }
    len as f64
}

/// The workgroup staging of `key` and its lowered programs, if it has one within `cap`
/// instructions per program.
fn stage(key: &FusedKey, cap: usize, cost: &CostModel, tuning: &Tuning) -> Option<(Staging, StagedPrograms)> {
    let found = staging::candidates(key)?;
    let bytes = |s: u32| key.subs.get(s as usize).and_then(|program| program.last()).map_or(4, |instr| instr.op.out_ty().bytes() as u64);
    let staging = staging::plan(&found, tuning, bytes)?;
    let programs = lower_staged(key, &staging, cap, cost)?;
    Some((staging, programs))
}

/// The estimated time of one kernel running `programs` as `staging` lays them out over
/// `elements` output cells: the main program at every cell, a staged sub-program at every cell of
/// its region (and a store and a load of the value, counted as 4 operations).
fn staged_time(key: &FusedKey, staging: &Staging, programs: &StagedPrograms, elements: f64, cost: &CostModel) -> f64 {
    let tile = f64::from(staging.tile[0]) * f64::from(staging.tile[1]);
    let all = std::iter::once(&programs.main).chain(&programs.subs);
    let reads = all.flat_map(|lowered| lowered.instrs.iter().flat_map(|flat| flat.args).chain(lowered.ctxs.iter().flat_map(|ctx| ctx.args)));
    let io_bytes = elements * (input_bytes(key, reads) + output_bytes(key));
    let staged_ops: f64 = staging.subs.iter().zip(&programs.subs).map(|(entry, lowered)| f64::from(entry.cells()) / tile * (lowered.ops + 4.0)).sum();
    kernel_ns(cost, io_bytes, (programs.main.ops + staged_ops) * elements)
}

/// How much cheaper a staged kernel must be estimated than the one it replaces to be chosen.
const STAGE_MARGIN: f64 = 0.9;

/// How much cheaper evaluating a sub-program first must be estimated than the kernel it replaces.
/// Stricter than [`STAGE_MARGIN`]: the program is flattened again on the host, which the estimate
/// leaves out, and the operation counts err most on long programs bound by memory latency (the
/// MUSCL step at 2048² ran slower with a sub-program its estimate put at 11% cheaper).
const MATERIALISE_MARGIN: f64 = 0.75;

/// Decides how to run `key` (which [`inlines`](FusedKey::inlines)) over `len` elements: as one
/// kernel that computes every sub-program at each index it is read at, as one kernel that
/// stages the sub-programs read through translations in workgroup memory (if `staging_on`), or
/// with the sub-program whose repeated computation costs the most evaluated first (see the
/// module documentation). Staging must be estimated below [`STAGE_MARGIN`] times the time of one
/// kernel, and evaluating a sub-program first below [`MATERIALISE_MARGIN`] times the time of the
/// choice before it. Lowered programs over `cap` instructions are always cut, unless staging
/// makes them fit.
pub(crate) fn judge(key: &FusedKey, len: usize, cap: usize, cost: &CostModel, tuning: &Tuning, staging_on: bool) -> (Judgement, Estimates) {
    let elements = len as f64;
    let staged = if staging_on { stage(key, cap, cost, tuning) } else { None };
    let staged_ns = staged.as_ref().map(|(staging, programs)| staged_time(key, staging, programs, elements, cost));
    let inlined = match inline(key, cap, cost) {
        Verdict::Fits(inlined) => inlined,
        Verdict::TooBig { sub } => {
            let estimates = Estimates { inline_ns: f64::INFINITY, best_sub: None, staged_ns };
            if let Some((staging, _)) = staged {
                return (Judgement::Run { staging: Some(staging) }, estimates);
            }
            let reason = SplitReason::Inlined { instructions: cap.saturating_add(1), limit: cap };
            return (Judgement::Materialise { sub, reason }, estimates);
        }
    };
    let out_bytes = output_bytes(key);
    let reads = inlined.instrs.iter().flat_map(|flat| flat.args).chain(inlined.ctxs.iter().flat_map(|ctx| ctx.args));
    let io_bytes = elements * (input_bytes(key, reads) + out_bytes);
    let inline_ns = kernel_ns(cost, io_bytes, inlined.ops * elements);

    let mut best: Option<(u32, f64)> = None;
    for (s, program) in key.subs.iter().enumerate() {
        let instances = inlined.count.get(s).copied().unwrap_or(0);
        if instances < 2 {
            continue;
        }
        let s32 = u32::try_from(s).unwrap_or(u32::MAX);
        // Evaluated on its own, the sub-program computes the sub-programs it reads too; what it
        // shares with the rest of the program is still computed there.
        let Some((once, in_s)) = alone(key, s, cap, cost) else { continue };
        let Some(rest) = without(key, s32, cap, cost) else { continue };
        let len_s = sub_len(key, s32, len);
        let out_s = program.last().map_or(0.0, |instr| instr.op.out_ty().bytes() as f64);
        let t = kernel_ns(cost, len_s * (in_s + out_s), once * len_s) + kernel_ns(cost, io_bytes + elements * out_s, rest * elements);
        if best.is_none_or(|(_, t_best)| t < t_best) {
            best = Some((s32, t));
        }
    }
    let estimates = Estimates { inline_ns, best_sub: best, staged_ns };
    let mut current = inline_ns;
    let mut choice = None;
    if let (Some((staging, _)), Some(t)) = (staged, staged_ns)
        && t < STAGE_MARGIN * current
    {
        current = t;
        choice = Some(staging);
    }
    match best {
        Some((sub, t)) if t < MATERIALISE_MARGIN * current => {
            let instances = inlined.count[sub as usize] as usize;
            (Judgement::Materialise { sub, reason: SplitReason::Materialised { instances } }, estimates)
        }
        _ => (Judgement::Run { staging: choice }, estimates),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtin::{BinOp, CmpOp, UnOp};
    use crate::jit::program::Tail;
    use crate::jit::{ElemTy, Gen};

    fn instr(kind: OpKind, args: [Ref; 3]) -> Instr {
        Instr { op: Op::new(kind, ElemTy::F32), args }
    }

    fn key(instrs: Vec<Instr>, subs: Vec<Vec<Instr>>) -> FusedKey {
        let outputs = vec![instrs.len() as u32 - 1];
        FusedKey {
            shape: crate::jit::program::shape_of(&instrs, &subs),
            instrs: std::sync::Arc::new(instrs),
            input_tys: vec![ElemTy::F32],
            scalar_tys: vec![ElemTy::F32, ElemTy::F32],
            outputs,
            subs: std::sync::Arc::new(subs),
            tail: Tail::Store,
            in_place: Vec::new(),
            spans: Vec::new(),
            tile: None,
            staging: None,
            guard: None,
        }
    }

    /// The slice `start..start + out` of a 1-D array of `n` elements.
    fn slice(n: u32, start: i32, out: u32) -> ShiftKind {
        let mut map = ViewMap::identity([n, 1, 1, 1]);
        map.out[0] = out;
        map.axes[0].start = start;
        ShiftKind::View { map, edge: Edge::None }
    }

    fn shift(kind: ShiftKind, sub: u32) -> Instr {
        instr(OpKind::Shift(kind), [Ref::Sub(sub), Ref::Unused, Ref::Unused])
    }

    /// `e = sqrt(x * s0 + s1)`: three instructions.
    fn expression() -> Vec<Instr> {
        vec![
            instr(OpKind::Binary(BinOp::Mul), [Ref::Input(0), Ref::Scalar(0), Ref::Unused]),
            instr(OpKind::Binary(BinOp::Add), [Ref::Tmp(0), Ref::Scalar(1), Ref::Unused]),
            instr(OpKind::Unary(UnOp::Sqrt), [Ref::Tmp(1), Ref::Unused, Ref::Unused]),
        ]
    }

    fn fits(verdict: Verdict) -> Inlined {
        match verdict {
            Verdict::Fits(inlined) => inlined,
            Verdict::TooBig { sub } => panic!("too big: sub {sub}"),
        }
    }

    #[test]
    fn lowered_programs_pass_validation_and_broken_ones_do_not() {
        let main = vec![shift(slice(9, 1, 8), 0), shift(slice(9, 0, 8), 0), instr(OpKind::Binary(BinOp::Sub), [Ref::Tmp(0), Ref::Tmp(1), Ref::Unused])];
        let k = key(main, vec![expression()]);
        assert_eq!(crate::jit::program::validate(&k, 1, 2), Ok(()));
        let mut inlined = fits(inline(&k, 1000, &CostModel::TEST));
        assert_eq!(validate_inlined(&inlined, 1, false), Ok(()));
        assert!(validate_inlined(&inlined, 0, false).unwrap_err().contains("input 0"));
        inlined.ctxs[1].parent = 1;
        assert!(validate_inlined(&inlined, 1, false).unwrap_err().contains("parent"));
    }

    #[test]
    fn one_level_of_views_computes_the_expression_once_per_index() {
        // `e.slice(1..) - e.slice(..n - 1)` with n = 9.
        let main = vec![
            shift(slice(9, 1, 8), 0),
            shift(slice(9, 0, 8), 0),
            instr(OpKind::Binary(BinOp::Sub), [Ref::Tmp(0), Ref::Tmp(1), Ref::Unused]),
        ];
        let inlined = fits(inline(&key(main, vec![expression()]), 1000, &CostModel::TEST));
        // Two indices of `e` (three instructions each) and the difference; the shifts are aliases.
        assert_eq!(inlined.instrs.len(), 7);
        assert_eq!(inlined.ctxs.len(), 3);
        assert_eq!(inlined.main, vec![2, 5, 6]);
        let last = &inlined.instrs[6];
        assert_eq!((last.at, last.args[0], last.args[1]), (0, Ref::Tmp(2), Ref::Tmp(5)));
    }

    #[test]
    fn slices_of_slices_reach_the_same_index_as_one_context() {
        // f = e.slice(1..) - e.slice(..n - 1); g = f.slice(1..) * f.slice(..n - 2) + f.slice(1..).
        let f = vec![
            shift(slice(9, 1, 8), 0),
            shift(slice(9, 0, 8), 0),
            instr(OpKind::Binary(BinOp::Sub), [Ref::Tmp(0), Ref::Tmp(1), Ref::Unused]),
        ];
        let g = vec![
            shift(slice(8, 1, 7), 1),
            shift(slice(8, 0, 7), 1),
            instr(OpKind::Binary(BinOp::Mul), [Ref::Tmp(0), Ref::Tmp(1), Ref::Unused]),
            instr(OpKind::Binary(BinOp::Add), [Ref::Tmp(2), Ref::Tmp(0), Ref::Unused]),
        ];
        let inlined = fits(inline(&key(g, vec![expression(), f]), 1000, &CostModel::TEST));
        // `e` at the starts 0, 1 and 2 (1 is reached twice): 3 x 3; `f` at two indices; `*` and `+`.
        assert_eq!(inlined.instrs.len(), 13);
        // Contexts: the two slices of `f`, and the three composed slices of `e`.
        assert_eq!(inlined.ctxs.len(), 1 + 2 + 3);
        let starts: Vec<i32> = inlined
            .ctxs
            .iter()
            .skip(1)
            .filter_map(|ctx| match ctx.kind {
                ShiftKind::View { map, .. } if map.src[0] == 9 => Some(map.axes[0].start),
                _ => None,
            })
            .collect();
        assert_eq!(starts, vec![2, 1, 0]);
    }

    #[test]
    fn a_view_of_everything_is_the_element_itself() {
        let identity = ShiftKind::View { map: ViewMap::identity([9, 1, 1, 1]), edge: Edge::None };
        let main = vec![shift(identity, 0), instr(OpKind::Binary(BinOp::Add), [Ref::Tmp(0), Ref::Input(0), Ref::Unused])];
        let inlined = fits(inline(&key(main, vec![expression()]), 1000, &CostModel::TEST));
        assert_eq!(inlined.ctxs.len(), 1);
        assert!(inlined.instrs.iter().all(|flat| flat.at == 0));
    }

    #[test]
    fn a_fill_view_keeps_its_select_and_never_merges_with_a_plain_one() {
        let mut map = ViewMap::identity([9, 1, 1, 1]);
        map.axes[0].start = -1;
        let main = vec![
            instr(OpKind::Shift(ShiftKind::View { map, edge: Edge::Fill }), [Ref::Sub(0), Ref::Scalar(0), Ref::Unused]),
            shift(ShiftKind::View { map, edge: Edge::Clamp }, 0),
        ];
        let inlined = fits(inline(&key(main, vec![expression()]), 1000, &CostModel::TEST));
        // The fill is one instruction on top of the expression at its context; clamp is an alias
        // of the expression at another context.
        assert_eq!(inlined.instrs.len(), 3 + 1 + 3);
        assert_eq!(inlined.ctxs.len(), 3);
    }

    #[test]
    fn equal_instructions_of_different_sub_programs_are_one() {
        // Two sub-programs that both contain `e` (the same instructions): read at one index.
        let a = vec![expression()[0].clone(), expression()[1].clone(), instr(OpKind::Unary(UnOp::Abs), [Ref::Tmp(1), Ref::Unused, Ref::Unused])];
        let b = vec![expression()[0].clone(), expression()[1].clone(), instr(OpKind::Unary(UnOp::Neg), [Ref::Tmp(1), Ref::Unused, Ref::Unused])];
        let main = vec![
            shift(slice(9, 1, 8), 0),
            shift(slice(9, 1, 8), 1),
            instr(OpKind::Binary(BinOp::Add), [Ref::Tmp(0), Ref::Tmp(1), Ref::Unused]),
        ];
        let inlined = fits(inline(&key(main, vec![a, b]), 1000, &CostModel::TEST));
        // Mul, Add shared; Abs, Neg; the sum.
        assert_eq!(inlined.instrs.len(), 5);
    }

    #[test]
    fn too_big_names_the_sub_program_with_the_most_repeated_instructions() {
        // `short` is read at 2 indices, `long` at 4: evaluating `long` saves the most.
        let long = expression();
        let short = vec![expression()[0].clone()];
        let main = vec![
            shift(slice(9, 1, 8), 0),
            shift(slice(9, 0, 8), 0),
            shift(slice(9, 2, 7), 1),
            shift(slice(9, 0, 7), 1),
            shift(slice(9, 3, 6), 0),
            shift(slice(9, 4, 5), 0),
            instr(OpKind::Binary(BinOp::Add), [Ref::Tmp(0), Ref::Tmp(1), Ref::Unused]),
        ];
        let verdict = inline(&key(main.clone(), vec![long, short]), 8, &CostModel::TEST);
        assert!(matches!(verdict, Verdict::TooBig { sub: 0 }), "{verdict:?}");
        // Within the cap it fits.
        let again = inline(&key(main, vec![expression(), vec![expression()[0].clone()]]), 100, &CostModel::TEST);
        assert!(matches!(again, Verdict::Fits(_)));
    }

    /// `e.slice(1..) - e.slice(..n - 1)` over a long array.
    fn difference_key() -> FusedKey {
        let main = vec![
            shift(slice(1 << 20, 1, (1 << 20) - 1), 0),
            shift(slice(1 << 20, 0, (1 << 20) - 1), 0),
            instr(OpKind::Binary(BinOp::Sub), [Ref::Tmp(0), Ref::Tmp(1), Ref::Unused]),
        ];
        key(main, vec![expression()])
    }

    /// [`difference_key`] of an expression of 20 operations.
    fn long_difference_key() -> FusedKey {
        let mut long = expression();
        for k in 2..19 {
            long.push(instr(OpKind::Unary(UnOp::Sqrt), [Ref::Tmp(k), Ref::Unused, Ref::Unused]));
        }
        FusedKey { subs: std::sync::Arc::new(vec![long]), ..difference_key() }
    }

    #[test]
    fn costly_launches_keep_the_recomputation_and_costly_arithmetic_stores_the_expression() {
        let key = long_difference_key();
        let len = 1 << 20;
        let tuning = Tuning::for_tests();
        let (cheap_launch, _) = judge(&key, len, 1000, &CostModel { launch_us: 1e6, ..CostModel::TEST }, &tuning, true);
        assert!(matches!(cheap_launch, Judgement::Run { .. }), "{cheap_launch:?}");
        let (cheap_memory, estimates) = judge(&key, len, 1000, &CostModel { f32_ops_per_ns: 1e-6, ..CostModel::TEST }, &tuning, false);
        let Judgement::Materialise { sub: 0, reason: SplitReason::Materialised { instances: 2 } } = cheap_memory else {
            panic!("{cheap_memory:?} {estimates:?}");
        };
        assert!(estimates.best_sub.is_some_and(|(_, t)| t < estimates.inline_ns), "{estimates:?}");
        // A short expression costs less to recompute than its two reads through views.
        let (short, estimates) = judge(&difference_key(), len, 1000, &CostModel { f32_ops_per_ns: 1e-6, ..CostModel::TEST }, &tuning, false);
        assert!(matches!(short, Judgement::Run { staging: None }), "{short:?} {estimates:?}");
    }

    #[test]
    fn a_program_over_the_cap_is_cut_whatever_the_model_says() {
        let slow = CostModel { launch_us: 1e6, ..CostModel::TEST };
        let (judgement, estimates) = judge(&difference_key(), 1 << 20, 4, &slow, &Tuning::for_tests(), false);
        assert!(matches!(judgement, Judgement::Materialise { sub: 0, reason: SplitReason::Inlined { limit: 4, .. } }), "{judgement:?}");
        assert!(estimates.inline_ns.is_infinite());
    }

    #[test]
    fn a_program_over_the_cap_is_staged_when_staging_makes_it_fit() {
        let slow = CostModel { launch_us: 1e6, ..CostModel::TEST };
        let (judgement, estimates) = judge(&difference_key(), 1 << 20, 4, &slow, &Tuning::for_tests(), true);
        assert!(matches!(judgement, Judgement::Run { staging: Some(_) }), "{judgement:?} {estimates:?}");
        assert!(estimates.staged_ns.is_some() && estimates.inline_ns.is_infinite(), "{estimates:?}");
        let (judgement, _) = judge(&difference_key(), 1 << 20, 2, &slow, &Tuning::for_tests(), true);
        assert!(matches!(judgement, Judgement::Materialise { reason: SplitReason::Inlined { limit: 2, .. }, .. }), "{judgement:?}");
    }

    /// `select(coordinate >= s0, e read through a clamped view of a 4-cell array, input)` over 64
    /// cells, `e` being `expression()` followed by `extra` square roots (or the mask read from
    /// the input, which makes it depend on data).
    fn region_key(extra: u32, data_mask: bool) -> FusedKey {
        let mut long = expression();
        for k in 0..extra {
            long.push(instr(OpKind::Unary(UnOp::Sqrt), [Ref::Tmp(2 + k), Ref::Unused, Ref::Unused]));
        }
        let mut map = ViewMap::identity([4, 1, 1, 1]);
        map.out[0] = 64;
        let (first, mask) = if data_mask {
            (Instr { op: Op::new(OpKind::Binary(BinOp::Add), ElemTy::U32), args: [Ref::Input(1), Ref::Input(1), Ref::Unused] }, Ref::Tmp(0))
        } else {
            (Instr { op: Op::new(OpKind::Generate(Gen::Range { stride: 1, size: 64 }), ElemTy::U32), args: [Ref::Unused; 3] }, Ref::Tmp(0))
        };
        let main = vec![
            first,
            Instr { op: Op::new(OpKind::Compare(CmpOp::Ge), ElemTy::U32), args: [mask, Ref::Scalar(0), Ref::Unused] },
            instr(OpKind::Shift(ShiftKind::View { map, edge: Edge::Clamp }), [Ref::Sub(0), Ref::Unused, Ref::Unused]),
            instr(OpKind::Select, [Ref::Tmp(1), Ref::Tmp(2), Ref::Input(0)]),
        ];
        let mut key = key(main, vec![long]);
        key.input_tys = vec![ElemTy::F32, ElemTy::U32];
        key
    }

    #[test]
    fn a_select_on_coordinates_computes_the_operand_only_where_it_is_taken() {
        let cost = CostModel::TEST;
        let inlined = fits(inline(&region_key(20, false), 1000, &cost));
        // The expression (3 + 20 instructions, no other reader) is computed where the mask holds.
        let select = inlined.instrs.len() - 1;
        assert_eq!(select, 25);
        assert!(inlined.branch[select], "{:?}", inlined.region);
        assert!(inlined.region[2..25].iter().all(|&region| region == Region::Taken(select as u32)), "{:?}", inlined.region);
        assert!([0, 1, select].iter().all(|&n| inlined.region[n] == Region::Top), "{:?}", inlined.region);
        // 4 of the 64 cells take it: the operations are the program's four plus 1/16 of the expression's.
        let all: f64 = inlined.instrs.iter().map(|flat| op_weight(flat.op, &cost)).sum();
        let expression: f64 = inlined.instrs.iter().zip(&inlined.region).filter(|&(_, &region)| region != Region::Top).map(|(flat, _)| op_weight(flat.op, &cost)).sum();
        assert!((inlined.ops - (all - expression + expression / 16.0)).abs() < 1e-9, "{} of {all}, expression {expression}", inlined.ops);
        assert_eq!(validate_inlined(&inlined, 1, false), Ok(()));
    }

    #[test]
    fn a_cheap_branch_or_a_mask_that_reads_data_stays_a_select() {
        let cost = CostModel::TEST;
        let cheap = fits(inline(&region_key(0, false), 1000, &cost));
        assert!(cheap.branch.iter().all(|&branch| !branch), "{:?}", cheap.region);
        assert!(cheap.region.iter().all(|&region| region == Region::Top));
        let data = fits(inline(&region_key(20, true), 1000, &cost));
        assert!(data.branch.iter().all(|&branch| !branch));
    }
}

/// Checks the structure the code generator relies on in a lowered program of `key`'s kernel
/// with `inputs` inputs: contexts descend from lower ones, temporaries refer backwards, contexts
/// exist, and sub-program reads appear only as shifts of staged lowerings.
#[cfg(any(debug_assertions, test))]
pub(super) fn validate_inlined(inlined: &Inlined, inputs: usize, staged: bool) -> Result<(), String> {
    for (id, ctx) in inlined.ctxs.iter().enumerate().skip(1) {
        if ctx.parent as usize >= id {
            return Err(format!("context {id} has parent {}", ctx.parent));
        }
    }
    let n = inlined.instrs.len();
    for (at, flat) in inlined.instrs.iter().enumerate() {
        let here = format!("lowered instruction {at} ({:?})", flat.op.kind);
        if flat.at as usize >= inlined.ctxs.len() || flat.child as usize >= inlined.ctxs.len() {
            return Err(format!("{here}: context {} / {} of {}", flat.at, flat.child, inlined.ctxs.len()));
        }
        for (slot, &arg) in flat.args.iter().enumerate() {
            match arg {
                Ref::Tmp(t) if t as usize >= at => return Err(format!("{here}: operand {slot} reads t{t}, not an earlier instruction")),
                Ref::Input(k) if k as usize >= inputs => return Err(format!("{here}: operand {slot} reads input {k} of {inputs}")),
                Ref::Sub(_) if !(staged && slot == 0 && matches!(flat.op.kind, OpKind::Shift(_))) => {
                    return Err(format!("{here}: operand {slot} is a sub-program"));
                }
                _ => {}
            }
        }
    }
    for ctx in inlined.ctxs.iter().skip(1) {
        for arg in ctx.args {
            if let Ref::Tmp(t) = arg
                && t as usize >= n
            {
                return Err(format!("a context reads t{t} of {n}"));
            }
        }
    }
    if let Some(&at) = inlined.main.iter().find(|&&at| at as usize >= n) {
        return Err(format!("main entry {at} of {n} lowered instructions"));
    }
    Ok(())
}
