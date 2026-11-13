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

use std::sync::Arc;

use crate::gpu::hash::FxMap;
use crate::profile::SplitReason;
use crate::tuning::Tuning;

use super::cost::{kernel_ns_on, op_weight, CostModel};
use super::program::{shape_of, FusedKey, Instr, Ref, Tail};
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
    /// Per lowered instruction, where it is computed (see [`branches`]).
    pub(crate) region: Vec<Region>,
    /// Per lowered instruction, whether it is a `Select` computed as `if`/`else`: operand 1 only
    /// where the mask holds, operand 2 only where it does not.
    pub(crate) branch: Vec<bool>,
    /// The operations of the lowered program in f32-operation units (see [`op_weight`]) per
    /// element, those of a branch weighted by the share of elements that compute it.
    pub(crate) ops: f64,
}

/// Staged sub-programs of one level over the same region, lowered together: they run in one loop
/// and an instruction that several of them compute (the wave speeds the four fields of a flux
/// share) is one instruction.
#[derive(Debug)]
pub(crate) struct StagedGroup {
    /// Indices into [`Staging::subs`] of the sub-programs of the group.
    pub(crate) members: Vec<usize>,
    /// Per member, the lowered instruction holding its value.
    pub(crate) values: Vec<u32>,
    pub(crate) body: Inlined,
}

/// A key lowered for a workgroup-staged kernel: the main program and the staged sub-programs
/// in groups (in the order of [`Staging::subs`]), each lowered at its own cell.
#[derive(Debug)]
pub(crate) struct StagedPrograms {
    pub(crate) main: Inlined,
    pub(crate) groups: Vec<StagedGroup>,
}

impl StagedPrograms {
    /// The main program and the bodies of the groups.
    pub(crate) fn bodies(&self) -> impl Iterator<Item = &Inlined> {
        std::iter::once(&self.main).chain(self.groups.iter().map(|group| &group.body))
    }
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

/// Lowers the sub-programs of `members` (entries of `staging`) together at their own cell, with the
/// reads of staged sub-programs left for workgroup memory. `None` past `cap` instructions.
fn lower_group(key: &FusedKey, staging: &Staging, staged: &[bool], members: &[usize], cap: usize, cost: &CostModel) -> Option<StagedGroup> {
    let mut lowering = Lowering::new(&key.subs, staged, cap);
    let mut values = Vec::with_capacity(members.len());
    for &member in members {
        let program = key.subs.get(staging.subs.get(member)?.sub as usize)?;
        values.push(lowering.lower(program, 0).last().copied().unwrap_or(0));
    }
    if lowering.over {
        return None;
    }
    let body = lowering.finish(Vec::new(), values.clone(), cost);
    Some(StagedGroup { members: members.to_vec(), values, body })
}

/// Lowers the main program of `key` and every sub-program `staging` stages, each at its own
/// cell, with the reads of staged sub-programs left for workgroup memory. The sub-programs of
/// one level over the same region are lowered together (see [`StagedGroup`]), apart if together
/// they have more than `cap` instructions. `None` if any program has more than `cap`.
pub(crate) fn lower_staged(key: &FusedKey, staging: &Staging, cap: usize, cost: &CostModel) -> Option<StagedPrograms> {
    let mut staged = vec![false; key.subs.len()];
    for entry in &staging.subs {
        *staged.get_mut(entry.sub as usize)? = true;
    }
    let main = lower_program(&key.subs, &key.instrs, &staged, cap, cost, |main| key_roots(key, main))?;
    let mut groups = Vec::new();
    for level in staging.levels.windows(2) {
        let mut by_region: Vec<Vec<usize>> = Vec::new();
        for member in level[0] as usize..level[1] as usize {
            let own = staging.subs.get(member)?;
            match by_region.iter_mut().find(|found| staging.subs.get(found[0]).is_some_and(|first| (first.dims, first.lo, first.hi) == (own.dims, own.lo, own.hi))) {
                Some(found) => found.push(member),
                None => by_region.push(vec![member]),
            }
        }
        for members in by_region {
            match lower_group(key, staging, &staged, &members, cap, cost) {
                Some(group) => groups.push(group),
                None => {
                    for &member in &members {
                        groups.push(lower_group(key, staging, &staged, &[member], cap, cost)?);
                    }
                }
            }
        }
    }
    Some(StagedPrograms { main, groups })
}

/// What [`judge`] decided.
#[derive(Debug)]
pub(crate) enum Judgement {
    /// Run the program as one kernel: lowered by [`inline`], or by [`lower_staged`] if `staging`
    /// is set (it becomes the `staging` of the key).
    Run { staging: Option<Staging> },
    /// Evaluate the sub-programs `subs` first, together (the kernel then reads their buffers),
    /// and flatten again.
    Materialise { subs: Vec<u32>, reason: SplitReason },
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
/// its region rounded up to whole rounds of the workgroup (and a store and a load of the value,
/// counted as 4 operations).
fn staged_time(key: &FusedKey, staging: &Staging, programs: &StagedPrograms, elements: f64, cost: &CostModel) -> f64 {
    let tile = f64::from(staging.tile[0]) * f64::from(staging.tile[1]);
    let all = programs.bodies();
    let reads = all.flat_map(|lowered| lowered.instrs.iter().flat_map(|flat| flat.args).chain(lowered.ctxs.iter().flat_map(|ctx| ctx.args)));
    let io_bytes = elements * (input_bytes(key, reads) + output_bytes(key));
    let staged_ops: f64 = programs
        .groups
        .iter()
        .filter_map(|group| {
            // The workgroup computes the region in rounds of one cell per invocation, and a
            // round costs the same however few invocations have a cell left in it: a region of
            // 272 cells over 256 invocations is two rounds, not 1.06.
            let cells = staging.subs.get(*group.members.first()?)?.cells();
            let workgroup = f64::from(super::stencil::WORKGROUP);
            let computed = (f64::from(cells) / workgroup).ceil() * workgroup;
            Some(computed / tile * (group.body.ops + 4.0 * group.members.len() as f64))
        })
        .sum();
    let invocations = f64::from(staging.tiles()) * f64::from(super::stencil::WORKGROUP);
    kernel_ns_on(cost, io_bytes, (programs.main.ops + staged_ops) * elements, invocations)
}

/// How much cheaper a staged kernel must be estimated than the one it replaces to be chosen.
const STAGE_MARGIN: f64 = 0.9;

/// How much cheaper evaluating a sub-program first must be estimated than the kernel it replaces.
/// Stricter than [`STAGE_MARGIN`]: the program is flattened again on the host, which the estimate
/// leaves out, and the operation counts err most on long programs bound by memory latency (the
/// MUSCL step at 2048² ran slower with a sub-program its estimate put at 11% cheaper).
const MATERIALISE_MARGIN: f64 = 0.75;

/// How [`judge`] weighs the ways of running a program: the instruction cap, the device's model,
/// the tuning of workgroup staging and whether staging is allowed at all.
#[derive(Clone, Copy)]
struct Env<'a> {
    cap: usize,
    cost: &'a CostModel,
    tuning: &'a Tuning,
    staging_on: bool,
}

/// The ways of running a key as one kernel and what they cost over `elements` elements.
struct Plans {
    /// The program written out at every index its sub-programs are read at; past the
    /// cap, the sub-program that saves the most instructions when evaluated first.
    inlined: Result<Inlined, u32>,
    /// One kernel running the lowered program (infinite if it exceeds the cap).
    inline_ns: f64,
    /// The workgroup staging of the sub-programs read through views and the programs it runs.
    staged: Option<(Staging, StagedPrograms)>,
    /// One kernel staging them, with its time.
    staged_ns: Option<f64>,
    /// What the program that does not fit costs (see [`Plans::run_ns`]).
    unfit_ns: f64,
}

/// What a program that does not fit one kernel costs on top of its estimate: more than any
/// program that does.
const UNFIT_NS: f64 = 1e15;

/// Whether evaluating sub-programs first (two kernels, estimated `t`) is worth it over the
/// program as one kernel (estimated `run`): [`MATERIALISE_MARGIN`] times cheaper, or cheaper
/// at all if the program does not fit.
fn pays(t: f64, run: f64) -> bool {
    if run >= UNFIT_NS { t < run } else { t < MATERIALISE_MARGIN * run }
}

/// The estimated time of `key` over `len` elements if it were written out in full: its
/// operations and those of every sub-program it reads, counted once per distinct read (see
/// [`recomputation`]), without the sharing of equal instructions the lowering finds, and the
/// bytes of every input it reads.
fn written_out_ns(key: &FusedKey, len: usize, cost: &CostModel) -> f64 {
    let (_, alone) = recomputation(key, cost);
    let mut reads: Vec<(u32, ShiftKind)> = Vec::new();
    let mut ops = 0.0;
    for instr in key.instrs.iter() {
        ops += op_weight(instr.op, cost);
        if let (OpKind::Shift(kind), Ref::Sub(j)) = (instr.op.kind, instr.args[0])
            && !reads.contains(&(j, kind))
        {
            reads.push((j, kind));
            ops += alone.get(j as usize).copied().unwrap_or(0.0);
        }
    }
    let bytes = input_bytes(key, key.instrs.iter().chain(key.subs.iter().flatten()).flat_map(|instr| instr.args)) + output_bytes(key);
    kernel_ns_on(cost, len as f64 * bytes, ops * len as f64, len as f64)
}

impl Plans {
    /// Finds the plans of `key` over `len` elements.
    fn of(key: &FusedKey, len: usize, env: Env) -> Plans {
        let elements = len as f64;
        let staged = if env.staging_on { stage(key, env.cap, env.cost, env.tuning) } else { None };
        let staged_ns = staged.as_ref().map(|(staging, programs)| staged_time(key, staging, programs, elements, env.cost));
        match inline(key, env.cap, env.cost) {
            Verdict::Fits(inlined) => {
                let reads = inlined.instrs.iter().flat_map(|flat| flat.args).chain(inlined.ctxs.iter().flat_map(|ctx| ctx.args));
                let io_bytes = elements * (input_bytes(key, reads) + output_bytes(key));
                let inline_ns = kernel_ns_on(env.cost, io_bytes, inlined.ops * elements, elements);
                Plans { inlined: Ok(inlined), inline_ns, staged, staged_ns, unfit_ns: 0.0 }
            }
            Verdict::TooBig { sub } => Plans { inlined: Err(sub), inline_ns: f64::INFINITY, staged, staged_ns, unfit_ns: UNFIT_NS + written_out_ns(key, len, env.cost) },
        }
    }

    /// Whether the program runs as one kernel: written out within the cap, or staged.
    fn fits(&self) -> bool {
        self.inlined.is_ok() || self.staged.is_some()
    }

    /// Whether the program runs staged: it must be estimated [`STAGE_MARGIN`] times cheaper than
    /// one kernel without staging, unless that does not fit.
    fn stages(&self) -> bool {
        match self.staged_ns {
            Some(t) => t < STAGE_MARGIN * self.inline_ns,
            None => false,
        }
    }

    /// The estimated time of the program as one kernel, the way it will run. A program that
    /// does not fit costs [`UNFIT_NS`] more than any that does, and among those that do not
    /// the estimate of what writing it out would cost orders them: the search then finds the
    /// sub-programs to evaluate first even where no single one makes the rest fit.
    fn run_ns(&self) -> f64 {
        match self.staged_ns {
            Some(t) if self.stages() => t,
            _ if self.inlined.is_ok() => self.inline_ns,
            _ => self.unfit_ns,
        }
    }
}

/// The sub-programs `main` reaches (those it reads through shifts, and those they read),
/// renumbered in their order, and for each the number it had in `subs`; the reads of the
/// sub-programs of `replaced` (pairs of sub-program and input) become reads of that input.
fn reachable(main: &[Instr], subs: &[Vec<Instr>], replaced: &[(u32, u32)]) -> (Vec<Instr>, Vec<Vec<Instr>>, Vec<u32>) {
    let kept = |reference: Ref| match reference {
        Ref::Sub(j) if !replaced.iter().any(|&(sub, _)| sub == j) => Some(j as usize),
        _ => None,
    };
    let mut used = vec![false; subs.len()];
    for instr in main {
        if let Some(slot) = kept(instr.args[0]).and_then(|j| used.get_mut(j)) {
            *slot = true;
        }
    }
    for (s, program) in subs.iter().enumerate().rev() {
        if used[s] {
            for instr in program {
                if let Some(slot) = kept(instr.args[0]).and_then(|j| used.get_mut(j)) {
                    *slot = true;
                }
            }
        }
    }
    let mut renumbered = vec![0u32; subs.len()];
    let mut originals = Vec::new();
    for (s, &used) in used.iter().enumerate() {
        if used {
            renumbered[s] = u32::try_from(originals.len()).unwrap_or(u32::MAX);
            originals.push(u32::try_from(s).unwrap_or(u32::MAX));
        }
    }
    let rewrite = |program: &[Instr]| -> Vec<Instr> {
        program
            .iter()
            .map(|instr| {
                let mut instr = instr.clone();
                if let Ref::Sub(j) = instr.args[0] {
                    instr.args[0] = match replaced.iter().find(|&&(sub, _)| sub == j) {
                        Some(&(_, input)) => Ref::Input(input),
                        None => Ref::Sub(renumbered.get(j as usize).copied().unwrap_or(0)),
                    };
                }
                instr
            })
            .collect()
    };
    let main = rewrite(main);
    let subs = subs.iter().zip(&used).filter(|&(_, &used)| used).map(|(program, _)| rewrite(program)).collect();
    (main, subs, originals)
}

/// `key` with the program replaced by `main` over `subs` (see [`reachable`]) and the tail,
/// outputs and inputs given.
fn rebuilt(key: &FusedKey, main: Vec<Instr>, subs: Vec<Vec<Instr>>, outputs: Vec<u32>, tail: Tail, input_tys: Vec<super::ElemTy>) -> FusedKey {
    FusedKey {
        shape: shape_of(&main, &subs),
        instrs: Arc::new(main),
        input_tys,
        scalar_tys: key.scalar_tys.clone(),
        outputs,
        subs: Arc::new(subs),
        tail,
        in_place: Vec::new(),
        spans: Vec::new(),
        tile: None,
        staging: None,
        guard: key.guard,
    }
}

/// The sub-programs `group` of `key` as one program storing all their values (equal
/// instructions among them are one).
fn group_key(key: &FusedKey, group: &[u32]) -> Option<FusedKey> {
    let mut main: Vec<Instr> = Vec::new();
    let mut outputs = Vec::new();
    for &s in group {
        let base = u32::try_from(main.len()).ok()?;
        for instr in key.subs.get(s as usize)? {
            let mut instr = instr.clone();
            for arg in &mut instr.args {
                if let Ref::Tmp(t) = arg {
                    *t += base;
                }
            }
            main.push(instr);
        }
        outputs.push(u32::try_from(main.len().checked_sub(1)?).ok()?);
    }
    let (main, subs, _) = reachable(&main, &key.subs, &[]);
    Some(rebuilt(key, main, subs, outputs, Tail::Store, key.input_tys.clone()))
}

/// `key` once the sub-programs `group` are evaluated: its reads of them are reads of buffers.
/// Also the number each of its sub-programs has in `key`.
fn rest_key(key: &FusedKey, group: &[u32]) -> Option<(FusedKey, Vec<u32>)> {
    let first = u32::try_from(key.input_tys.len()).ok()?;
    let mut input_tys = key.input_tys.clone();
    let mut replaced = Vec::with_capacity(group.len());
    for (k, &s) in group.iter().enumerate() {
        input_tys.push(key.subs.get(s as usize)?.last()?.op.out_ty());
        replaced.push((s, first + u32::try_from(k).ok()?));
    }
    let (main, subs, originals) = reachable(&key.instrs, &key.subs, &replaced);
    Some((rebuilt(key, main, subs, key.outputs.clone(), key.tail, input_tys), originals))
}

/// What the sub-programs of `key` cost when written out, by a count that ignores that equal
/// instructions are one: per sub-program, the number of times the main program reaches it, and
/// the operations (see [`op_weight`]) of its program with every sub-program it reads written out.
fn recomputation(key: &FusedKey, cost: &CostModel) -> (Vec<f64>, Vec<f64>) {
    let n = key.subs.len();
    let reads = |program: &[Instr]| -> Vec<(u32, ShiftKind)> {
        let mut found: Vec<(u32, ShiftKind)> = Vec::new();
        for instr in program {
            if let (OpKind::Shift(kind), Ref::Sub(j)) = (instr.op.kind, instr.args[0])
                && !found.contains(&(j, kind))
            {
                found.push((j, kind));
            }
        }
        found
    };
    let weight = |program: &[Instr]| program.iter().map(|instr| op_weight(instr.op, cost)).sum::<f64>();
    let mut alone = vec![0.0f64; n];
    for s in 0..n {
        let own = &key.subs[s];
        alone[s] = weight(own) + reads(own).iter().map(|&(j, _)| alone.get(j as usize).copied().unwrap_or(0.0)).sum::<f64>();
    }
    let mut times = vec![0.0f64; n];
    for (j, _) in reads(&key.instrs) {
        if let Some(t) = times.get_mut(j as usize) {
            *t += 1.0;
        }
    }
    for s in (0..n).rev() {
        let t = times[s];
        for (j, _) in reads(&key.subs[s]) {
            if let Some(slot) = times.get_mut(j as usize) {
                *slot += t;
            }
        }
    }
    (times, alone)
}

/// The `limit` sub-programs worth costing as the one to evaluate first: those the main program
/// reaches more than once, the ones whose recomputation (see [`recomputation`]) wastes the most
/// first.
fn worth_evaluating(key: &FusedKey, cost: &CostModel, limit: usize) -> Vec<u32> {
    let (times, alone) = recomputation(key, cost);
    let mut found: Vec<(u32, f64)> = (0..key.subs.len()).filter(|&s| times[s] >= 2.0).map(|s| (s as u32, (times[s] - 1.0) * alone[s])).collect();
    found.sort_by(|a, b| b.1.total_cmp(&a.1));
    found.truncate(limit);
    found.into_iter().map(|(s, _)| s).collect()
}

/// How many sub-programs are costed as the one to evaluate first at every level of the search.
const CANDIDATES: usize = 6;

/// By what factor at most the cost of writing out two sub-programs that are evaluated together
/// differ (the fields of one state differ little; another stage of a time step by orders of
/// magnitude).
const SIBLING: f64 = 8.0;

/// How many sub-programs are evaluated together at most.
const GROUP: usize = 8;

/// How many levels deep the search goes: evaluating a sub-program first is itself a program
/// that is judged the same way.
const SEARCH_DEPTH: u32 = 4;

/// The search of [`judge`]: what to evaluate first, and what the program costs once it is.
struct Search<'a> {
    env: Env<'a>,
    /// The best time found (and the levels it was searched with) per program structure and length.
    memo: FxMap<(u64, usize, usize), (u32, f64)>,
}

impl Search<'_> {
    /// The sub-programs to evaluate together with `seed` (over `len` elements): the ones of
    /// the same length that cost about as much to write out and most waste recomputation,
    /// until what is left of the program fits one kernel. Sub-programs that read the same
    /// expression (the fields of a state) cost next to nothing to evaluate together and each
    /// leaves the program too big otherwise; those of another stage do not belong to them.
    fn group(&self, key: &FusedKey, seed: u32, len: usize) -> Vec<u32> {
        let wide = sub_len(key, seed, len);
        let (_, alone) = recomputation(key, self.env.cost);
        let like = |s: u32| alone.get(s as usize).zip(alone.get(seed as usize)).is_some_and(|(&a, &b)| a >= b / SIBLING && a <= b * SIBLING);
        let mut group = vec![seed];
        let Some((mut rest, mut originals)) = rest_key(key, &group) else { return group };
        while group.len() < GROUP {
            let Some(next) = worth_evaluating(&rest, self.env.cost, usize::MAX).into_iter().find(|&j| sub_len(&rest, j, len) == wide && like(originals[j as usize])) else { break };
            group.push(originals[next as usize]);
            let Some((smaller, kept)) = rest_key(&rest, &[next]) else { break };
            originals = kept.iter().map(|&k| originals[k as usize]).collect();
            rest = smaller;
        }
        group
    }

    /// The best group of sub-programs of `key` to evaluate first (over `len` elements, `plans`
    /// being how it runs as one kernel), with the estimated time of both kernels, each run as
    /// it will be; `None` if no sub-program is worth it or the search is out of levels.
    fn evaluate_first(&mut self, key: &FusedKey, len: usize, plans: &Plans, left: u32) -> Option<(Vec<u32>, f64)> {
        let run = plans.run_ns();
        // Two launches alone cost more than the margin allows.
        if left == 0 || 2.0 * self.env.cost.launch_us * 1e3 >= MATERIALISE_MARGIN * run {
            return None;
        }
        let mut best: Option<(Vec<u32>, f64)> = None;
        let mut tried: Vec<Vec<u32>> = Vec::new();
        // The one that wastes the most of each length: the padded state, the fluxes of each
        // direction and the like are different groups, and the first of them alone would crowd
        // the others out.
        let mut lengths: Vec<f64> = Vec::new();
        let seeds = worth_evaluating(key, self.env.cost, usize::MAX).into_iter().filter(|&s| {
            let of = sub_len(key, s, len);
            let new = !lengths.contains(&of);
            if new {
                lengths.push(of);
            }
            new
        });
        for seed in seeds.take(CANDIDATES).collect::<Vec<_>>() {
            let mut group = self.group(key, seed, len);
            group.sort_unstable();
            if tried.contains(&group) {
                continue;
            }
            let (Some(own), Some((rest, _))) = (group_key(key, &group), rest_key(key, &group)) else { continue };
            let len_s = sub_len(key, group[0], len) as usize;
            // What is left may evaluate one more group first (the fluxes of the other direction),
            // which is as far as that is searched.
            let t_rest = if left == SEARCH_DEPTH { self.best(&rest, len, 1) } else { Plans::of(&rest, len, self.env).run_ns() };
            let t = self.best(&own, len_s, left - 1) + t_rest;
            if t.is_finite() && best.as_ref().is_none_or(|&(_, t_best)| t < t_best) {
                best = Some((group.clone(), t));
            }
            tried.push(group);
        }
        best
    }

    /// The estimated time of `key` over `len` elements run the way [`judge`] would choose.
    fn best(&mut self, key: &FusedKey, len: usize, left: u32) -> f64 {
        let slot = (key.shape, len, key.input_tys.len());
        if let Some(&(searched, t)) = self.memo.get(&slot)
            && searched >= left
        {
            return t;
        }
        let plans = Plans::of(key, len, self.env);
        let run = plans.run_ns();
        let t = match self.evaluate_first(key, len, &plans, left) {
            Some((_, t)) if pays(t, run) => t,
            _ => run,
        };
        self.memo.insert(slot, (left, t));
        t
    }
}

/// Decides how to run `key` (which [`inlines`](FusedKey::inlines)) over `len` elements: as one
/// kernel that computes every sub-program at each index it is read at, as one kernel that
/// stages the sub-programs read through translations in workgroup memory (if `staging_on`), or
/// with some sub-programs evaluated first (see the module documentation). What to evaluate
/// first is searched for: of the sub-programs the program reads at the most indices, the group
/// that makes both kernels cost least, each judged the same way (the sub-programs of a
/// sub-program are evaluated first where that pays). Staging must be estimated below
/// [`STAGE_MARGIN`] times the time of one kernel, and evaluating sub-programs first below
/// [`MATERIALISE_MARGIN`] times the time of the choice before it. Lowered programs over `cap`
/// instructions are always cut, unless staging makes them fit.
pub(crate) fn judge(key: &FusedKey, len: usize, cap: usize, cost: &CostModel, tuning: &Tuning, staging_on: bool) -> (Judgement, Estimates) {
    let env = Env { cap, cost, tuning, staging_on };
    let plans = Plans::of(key, len, env);
    let run = plans.run_ns();
    let mut search = Search { env, memo: FxMap::default() };
    let first = search.evaluate_first(key, len, &plans, SEARCH_DEPTH);
    let estimates = Estimates { inline_ns: plans.inline_ns, best_sub: first.as_ref().map(|(group, t)| (group[0], *t)), staged_ns: plans.staged_ns };
    let over_cap = SplitReason::Inlined { instructions: cap.saturating_add(1), limit: cap };
    match first {
        Some((subs, t)) if pays(t, run) => {
            let reason = if plans.fits() {
                let (times, _) = recomputation(key, cost);
                SplitReason::Materialised { instances: times[subs[0] as usize] as usize }
            } else {
                over_cap
            };
            (Judgement::Materialise { subs, reason }, estimates)
        }
        _ if plans.fits() => {
            let staging = if plans.stages() { plans.staged.map(|(staging, _)| staging) } else { None };
            (Judgement::Run { staging }, estimates)
        }
        _ => (Judgement::Materialise { subs: vec![plans.inlined.err().unwrap_or(0)], reason: over_cap }, estimates),
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
        let Judgement::Materialise { subs, reason: SplitReason::Materialised { instances: 2 } } = &cheap_memory else { panic!("{cheap_memory:?} {estimates:?}") };
        assert_eq!(subs, &[0], "{cheap_memory:?}");
        assert!(estimates.best_sub.is_some_and(|(_, t)| t < estimates.inline_ns), "{estimates:?}");
        // A short expression costs less to recompute than its two reads through views.
        let (short, estimates) = judge(&difference_key(), len, 1000, &CostModel { f32_ops_per_ns: 1e-6, ..CostModel::TEST }, &tuning, false);
        assert!(matches!(short, Judgement::Run { staging: None }), "{short:?} {estimates:?}");
    }

    #[test]
    fn a_program_over_the_cap_is_cut_whatever_the_model_says() {
        let slow = CostModel { launch_us: 1e6, ..CostModel::TEST };
        let (judgement, estimates) = judge(&difference_key(), 1 << 20, 4, &slow, &Tuning::for_tests(), false);
        assert!(matches!(judgement, Judgement::Materialise { reason: SplitReason::Inlined { limit: 4, .. }, .. }), "{judgement:?}");
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

    #[test]
    fn staged_sub_programs_over_one_region_share_their_instructions() {
        use crate::jit::staging::StagedSub;
        // Two sub-programs over the same region that share their first two instructions (like the
        // fields of a flux), the main program reads each at two offsets.
        let first = expression();
        let second = vec![first[0].clone(), first[1].clone(), instr(OpKind::Unary(UnOp::Abs), [Ref::Tmp(1), Ref::Unused, Ref::Unused])];
        let main = vec![
            shift(slice(9, 1, 8), 0),
            shift(slice(9, 0, 8), 0),
            shift(slice(9, 1, 8), 1),
            shift(slice(9, 0, 8), 1),
            instr(OpKind::Binary(BinOp::Add), [Ref::Tmp(0), Ref::Tmp(1), Ref::Unused]),
            instr(OpKind::Binary(BinOp::Add), [Ref::Tmp(2), Ref::Tmp(3), Ref::Unused]),
            instr(OpKind::Binary(BinOp::Mul), [Ref::Tmp(4), Ref::Tmp(5), Ref::Unused]),
        ];
        let key = key(main, vec![first, second]);
        let region = |sub| StagedSub { sub, dims: [9, 1], lo: [0, 0], hi: [16, 0] };
        let staging = Staging { dims: [8, 1], tile: [16, 1], subs: vec![region(0), region(1)], levels: vec![0, 2] };
        let programs = lower_staged(&key, &staging, 1000, &CostModel::TEST).expect("lowers");
        assert_eq!(programs.groups.len(), 1);
        let group = &programs.groups[0];
        assert_eq!(group.members, vec![0, 1]);
        // Mul, Add, Sqrt, Abs: the first two are one instruction.
        assert_eq!(group.body.instrs.len(), 4);
        assert_eq!(group.values, vec![2, 3]);
        assert_eq!(validate_inlined(&group.body, 1, true), Ok(()));
    }

    /// `s0 = e(x)`, `s1 = sqrt(s0[..8])`, an unused `s2`, and a main program reading `s1` at two offsets.
    fn chain_key() -> FusedKey {
        let reads_s0 = vec![shift(slice(9, 0, 8), 0), instr(OpKind::Unary(UnOp::Sqrt), [Ref::Tmp(0), Ref::Unused, Ref::Unused])];
        let main = vec![
            shift(slice(8, 1, 7), 1),
            shift(slice(8, 0, 7), 1),
            instr(OpKind::Binary(BinOp::Add), [Ref::Tmp(0), Ref::Tmp(1), Ref::Unused]),
        ];
        key(main, vec![expression(), reads_s0, expression()])
    }

    #[test]
    fn what_is_left_of_a_program_reads_the_evaluated_sub_program_from_a_buffer() {
        let key = chain_key();
        // `s1` evaluated: its reads are reads of a new input, and nothing else is reached.
        let (rest, originals) = rest_key(&key, &[1]).expect("s1 has a value");
        assert!(rest.subs.is_empty() && originals.is_empty(), "{rest:?}");
        assert_eq!(rest.input_tys.len(), 2);
        assert!(rest.instrs.iter().take(2).all(|instr| instr.args[0] == Ref::Input(1)), "{rest:?}");
        // `s0` evaluated: `s1` is left, reading it; the unused `s2` is gone and `s1` is renumbered.
        let (rest, originals) = rest_key(&key, &[0]).expect("s0 has a value");
        assert_eq!(originals, vec![1]);
        assert_eq!(rest.subs.len(), 1);
        assert_eq!(rest.subs[0][0].args[0], Ref::Input(1));
        assert_eq!(rest.instrs[0].args[0], Ref::Sub(0));
    }

    #[test]
    fn sub_programs_evaluated_together_are_one_program_storing_each() {
        let key = chain_key();
        let together = group_key(&key, &[0, 1]).expect("both have values");
        // `s0`'s three instructions, then `s1`'s two (after them), each stored; `s1` reads `s0`.
        assert_eq!(together.instrs.len(), 5);
        assert_eq!(together.outputs, vec![2, 4]);
        assert_eq!(together.subs.len(), 1);
        assert!(matches!(together.instrs[3].args[0], Ref::Sub(0)));
        let alone = group_key(&key, &[1]).expect("s1 has a value");
        assert_eq!((alone.instrs.len(), alone.subs.len(), alone.outputs.clone()), (2, 1, vec![1]));
    }

    #[test]
    fn siblings_that_leave_too_much_alone_are_evaluated_together() {
        // Two expressions read at two offsets each; over a cap of 10 instructions, evaluating one
        // leaves 11, both leave 7.
        let first = expression();
        let second = vec![
            instr(OpKind::Binary(BinOp::Mul), [Ref::Input(0), Ref::Scalar(1), Ref::Unused]),
            instr(OpKind::Binary(BinOp::Add), [Ref::Tmp(0), Ref::Scalar(0), Ref::Unused]),
            instr(OpKind::Unary(UnOp::Sqrt), [Ref::Tmp(1), Ref::Unused, Ref::Unused]),
        ];
        let main = vec![
            shift(slice(9, 1, 8), 0),
            shift(slice(9, 0, 8), 0),
            shift(slice(9, 1, 8), 1),
            shift(slice(9, 0, 8), 1),
            instr(OpKind::Binary(BinOp::Add), [Ref::Tmp(0), Ref::Tmp(1), Ref::Unused]),
            instr(OpKind::Binary(BinOp::Add), [Ref::Tmp(2), Ref::Tmp(3), Ref::Unused]),
            instr(OpKind::Binary(BinOp::Mul), [Ref::Tmp(4), Ref::Tmp(5), Ref::Unused]),
        ];
        let key = key(main, vec![first, second]);
        let (judgement, estimates) = judge(&key, 1 << 20, 10, &CostModel::TEST, &Tuning::for_tests(), false);
        let Judgement::Materialise { subs, reason: SplitReason::Inlined { limit: 10, .. } } = &judgement else { panic!("{judgement:?} {estimates:?}") };
        assert_eq!(subs, &[0, 1], "{estimates:?}");
        // With room for the whole program nothing is cut.
        let (judgement, estimates) = judge(&key, 1 << 20, 100, &CostModel::TEST, &Tuning::for_tests(), false);
        assert!(matches!(judgement, Judgement::Run { staging: None }), "{judgement:?} {estimates:?}");
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
