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
//! only when a kernel is built or judged (see `JitState::inline_verdict`).

use crate::gpu::hash::FxMap;

use super::program::{FusedKey, Instr, Ref};
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
    /// of context `at` (operand 0 of a `Shift`: at the index of context `child`); never `Ref::Sub`.
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
pub(crate) fn inline(key: &FusedKey, cap: usize) -> Verdict {
    let placeholder = Ctx { parent: 0, kind: ShiftKind::Dims { dims: [1; 4], shifts: [0; 4] }, args: [Ref::Unused; 2] };
    let mut lowering = Lowering {
        subs: &key.subs,
        cap,
        ctxs: vec![placeholder],
        context_ids: FxMap::default(),
        instrs: Vec::new(),
        numbered: FxMap::default(),
        instances: FxMap::default(),
        count: vec![0; key.subs.len()],
        over: false,
    };
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
    Verdict::Fits(Inlined { ctxs: lowering.ctxs, instrs: lowering.instrs, main })
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
        let inlined = fits(inline(&key(main, vec![expression()]), 1000));
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
        let inlined = fits(inline(&key(g, vec![expression(), f]), 1000));
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
        let inlined = fits(inline(&key(main, vec![expression()]), 1000));
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
        let inlined = fits(inline(&key(main, vec![expression()]), 1000));
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
        let inlined = fits(inline(&key(main, vec![a, b]), 1000));
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
        let verdict = inline(&key(main.clone(), vec![long, short]), 8);
        assert!(matches!(verdict, Verdict::TooBig { sub: 0 }), "{verdict:?}");
        // Within the cap it fits.
        let again = inline(&key(main, vec![expression(), vec![expression()[0].clone()]]), 100);
        assert!(matches!(again, Verdict::Fits(_)));
    }
}
