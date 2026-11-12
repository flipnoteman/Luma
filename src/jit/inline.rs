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
use super::{Edge, Op, OpKind, ShiftKind, ViewMap};

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
    /// The operations of the lowered program in f32-operation units (see [`op_weight`]), per element.
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

    /// The lowered program once `main` (the values of the program's instructions) is known. A
    /// read of a staged sub-program counts as 2 operations: an index and a workgroup load.
    fn finish(self, main: Vec<u32>, cost: &CostModel) -> Inlined {
        let weight = |flat: &Flat| if matches!(flat.args[0], Ref::Sub(_)) { 2.0 } else { op_weight(flat.op, cost) };
        let ops = self.instrs.iter().map(weight).sum();
        Inlined { ctxs: self.ctxs, instrs: self.instrs, main, count: self.count, ops }
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
                        // Read from workgroup memory at the cell itself: equal reads are one instruction.
                        debug_assert_eq!(ctx, 0, "a staged sub-program read away from the cell of its reader");
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
    Verdict::Fits(lowering.finish(main, cost))
}

/// Lowers `program` at its own cell (context 0), with the sub-programs `staged` left as reads
/// from workgroup memory. `None` if more than `cap` instructions result.
fn lower_program(subs: &[Vec<Instr>], program: &[Instr], staged: &[bool], cap: usize, cost: &CostModel) -> Option<Inlined> {
    let mut lowering = Lowering::new(subs, staged, cap);
    let main = lowering.lower(program, 0);
    (!lowering.over).then(|| lowering.finish(main, cost))
}

/// Lowers the main program of `key` and every sub-program `staging` stages, each at its own
/// cell, with the reads of staged sub-programs left for workgroup memory. `None` if any
/// program has more than `cap` instructions.
pub(crate) fn lower_staged(key: &FusedKey, staging: &Staging, cap: usize, cost: &CostModel) -> Option<StagedPrograms> {
    let mut staged = vec![false; key.subs.len()];
    for entry in &staging.subs {
        *staged.get_mut(entry.sub as usize)? = true;
    }
    let main = lower_program(&key.subs, &key.instrs, &staged, cap, cost)?;
    let subs = staging
        .subs
        .iter()
        .map(|entry| lower_program(&key.subs, key.subs.get(entry.sub as usize)?, &staged, cap, cost))
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

/// Decides how to run `key` (which [`inlines`](FusedKey::inlines)) over `len` elements: as one
/// kernel that computes every sub-program at each index it is read at, as one kernel that
/// stages the sub-programs read through translations in workgroup memory (if `staging_on`), or
/// with the sub-program whose repeated computation costs the most evaluated first (see the
/// module documentation). Each choice must be estimated below 0.9 times the time of the one
/// before it, in that order. Lowered programs over `cap` instructions are always cut, unless
/// staging makes them fit.
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
        let once: f64 = program.iter().map(|instr| op_weight(instr.op, cost)).sum();
        let len_s = sub_len(key, s32, len);
        let in_s = input_bytes(key, program.iter().flat_map(|instr| instr.args));
        let out_s = program.last().map_or(0.0, |instr| instr.op.out_ty().bytes() as f64);
        let rest = (inlined.ops - f64::from(instances) * once).max(0.0);
        let t = kernel_ns(cost, len_s * (in_s + out_s), once * len_s) + kernel_ns(cost, io_bytes + elements * out_s, rest * elements);
        if best.is_none_or(|(_, t_best)| t < t_best) {
            best = Some((s32, t));
        }
    }
    let estimates = Estimates { inline_ns, best_sub: best, staged_ns };
    let mut current = inline_ns;
    let mut choice = None;
    if let (Some((staging, _)), Some(t)) = (staged, staged_ns)
        && t < 0.9 * current
    {
        current = t;
        choice = Some(staging);
    }
    match best {
        Some((sub, t)) if t < 0.9 * current => {
            let instances = inlined.count[sub as usize] as usize;
            (Judgement::Materialise { sub, reason: SplitReason::Materialised { instances } }, estimates)
        }
        _ => (Judgement::Run { staging: choice }, estimates),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtin::{BinOp, UnOp};
    use crate::jit::program::Tail;
    use crate::jit::ElemTy;

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

    #[test]
    fn costly_launches_keep_the_recomputation_and_costly_arithmetic_stores_the_expression() {
        let key = difference_key();
        let len = 1 << 20;
        let tuning = Tuning::for_tests();
        let (cheap_launch, _) = judge(&key, len, 1000, &CostModel { launch_us: 1e6, ..CostModel::TEST }, &tuning, true);
        assert!(matches!(cheap_launch, Judgement::Run { .. }), "{cheap_launch:?}");
        let (cheap_memory, estimates) = judge(&key, len, 1000, &CostModel { f32_ops_per_ns: 1e-6, ..CostModel::TEST }, &tuning, true);
        let Judgement::Materialise { sub: 0, reason: SplitReason::Materialised { instances: 2 } } = cheap_memory else {
            panic!("{cheap_memory:?} {estimates:?}");
        };
        assert!(estimates.best_sub.is_some_and(|(_, t)| t < estimates.inline_ns), "{estimates:?}");
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
}
