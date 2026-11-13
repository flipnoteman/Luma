//! Moving the extents, offsets and strides of views, gathers and scatters out of the source of a
//! kernel and into slots of its uniform `p`.
//!
//! A [`FusedKey`] carries the numbers of a `slice` or `pad` in its views, so a loop over the
//! offsets of one slice would compile one kernel per offset. [`apply`] rewrites such a key (and
//! its lowering, if it has one) to the variants that read the numbers from the uniform: of a view
//! only what shapes the code stays in the key (see [`ViewSpec`]). Workgroup-staged and tiled
//! kernels bake their offsets into the source and are not rewritten.

use std::fmt::Write;
use std::sync::Arc;

use super::codegen::EDGE_HELPERS;
use super::inline::{Ctx, Flat, Inlined};
use super::program::{shape_of, FusedKey, Instr, Tail};
use super::{Edge, ElemTy, Op, OpKind, ShiftKind, ViewMap};

/// How the step of a view axis enters the index.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum StepSpec {
    Zero,
    PosOne,
    NegOne,
    /// Any other step, read from the uniform.
    Runtime,
}

/// How the output index is divided or reduced modulo an extent: a shift or mask from the uniform
/// for a power of two, a literal constant otherwise (so the shader compiler emits `mulhi` and
/// shares it across views instead of a software integer division).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum DivRemSpec {
    None,
    Pow2,
    Const(u32),
}

/// What of one source axis of a view the generated code depends on; the rest are uniform words.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) struct AxisSpec {
    pub(crate) from: u8,
    /// The source has more than one element along the axis (a unit axis adds nothing to the index).
    pub(crate) active: bool,
    /// The output has one element along `from`: the coordinate is its start.
    pub(crate) out_unit: bool,
    pub(crate) out_div: DivRemSpec,
    pub(crate) out_rem: DivRemSpec,
    pub(crate) src_fastest: bool,
    pub(crate) step: StepSpec,
    /// The edge handling applies: the coordinate may leave the source.
    pub(crate) checked: bool,
    /// Source extent baked in for [`Edge::Wrap`] so its `%` is by a constant.
    pub(crate) wrap_size: u32,
}

/// A view whose numbers are in the uniform from slot `base_slot` on.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) struct ViewSpec {
    pub(crate) edge: Edge,
    pub(crate) base_slot: u32,
    pub(crate) axes: [AxisSpec; 4],
}
/// What a key rewritten by [`apply`] adds to what its other fields say.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub(crate) struct Parameterized {
    /// The lowering of the key (written with the same variants), if it is lowered.
    pub(crate) inlined: Option<Inlined>,
    /// Whether the uniform has slots the rewrite added.
    pub(crate) slots: bool,
}

const UNUSED_AXIS: AxisSpec = AxisSpec {
    from: 0,
    active: false,
    out_unit: true,
    out_div: DivRemSpec::None,
    out_rem: DivRemSpec::None,
    src_fastest: true,
    step: StepSpec::Zero,
    checked: false,
    wrap_size: 0,
};

/// The slots added so far and the views and gathers they belong to.
struct Builder {
    tys: Vec<ElemTy>,
    words: Vec<u32>,
    views: Vec<(ViewMap, Edge, u16, ViewSpec)>,
    takes: Vec<(u32, u32)>,
}

impl Builder {
    fn new(tys: &[ElemTy], words: &[u32]) -> Builder {
        Builder { tys: tys.to_vec(), words: words.to_vec(), views: Vec::new(), takes: Vec::new() }
    }

    fn word(&mut self, value: u32) -> u32 {
        let slot = self.tys.len() as u32;
        self.tys.push(ElemTy::U32);
        self.words.push(value);
        slot
    }

    fn view(&mut self, map: ViewMap, edge: Edge) -> Option<(u16, ViewSpec)> {
        if let Some(&(_, _, id, spec)) = self.views.iter().find(|&&(known, kind, ..)| known == map && kind == edge) {
            return Some((id, spec));
        }
        let id = u16::try_from(self.views.len()).ok()?;
        let mut out_stride = [1u64; 4];
        let mut src_stride = [1u64; 4];
        for d in 1..4 {
            out_stride[d] = out_stride[d - 1] * u64::from(map.out[d - 1]);
            src_stride[d] = src_stride[d - 1] * u64::from(map.src[d - 1]);
        }
        let base_slot = self.tys.len() as u32;
        let mut axes = [UNUSED_AXIS; 4];
        for a in 0..4 {
            let axis = map.axes[a];
            let from = axis.from as usize;
            let (first, last) = map.ends(a);
            let size = i64::from(map.src[a]);
            let inside = (0..size).contains(&first) && (0..size).contains(&last);
            let checked = edge != Edge::None && !inside;
            let stride_out = out_stride[from] as u32;
            let dim_out = map.out[from];
            let top_out = (from + 1..4).all(|d| map.out[d] == 1);
            let out_div = if stride_out <= 1 {
                DivRemSpec::None
            } else if stride_out.is_power_of_two() {
                DivRemSpec::Pow2
            } else {
                DivRemSpec::Const(stride_out)
            };
            let out_rem = if top_out || dim_out <= 1 {
                DivRemSpec::None
            } else if dim_out.is_power_of_two() {
                DivRemSpec::Pow2
            } else {
                DivRemSpec::Const(dim_out)
            };
            let spec = AxisSpec {
                from: axis.from,
                active: map.src[a] > 1,
                out_unit: dim_out == 1,
                out_div,
                out_rem,
                src_fastest: src_stride[a] == 1,
                step: match axis.step {
                    0 => StepSpec::Zero,
                    1 => StepSpec::PosOne,
                    -1 => StepSpec::NegOne,
                    _ => StepSpec::Runtime,
                },
                checked,
                wrap_size: if checked && edge == Edge::Wrap { map.src[a] } else { 0 },
            };
            if !(spec.active || (edge == Edge::Fill && spec.checked)) {
                continue;
            }
            axes[a] = spec;
            if !spec.out_unit && spec.step != StepSpec::Zero {
                if spec.out_div == DivRemSpec::Pow2 {
                    self.word(stride_out.trailing_zeros());
                }
                if spec.out_rem == DivRemSpec::Pow2 {
                    self.word(dim_out - 1);
                }
                if spec.step == StepSpec::Runtime {
                    self.word(axis.step as u32);
                }
            }
            self.word(axis.start as u32);
            if spec.checked && spec.wrap_size == 0 {
                self.word(map.src[a]);
            }
            if spec.active && !spec.src_fastest {
                self.word(src_stride[a] as u32);
            }
        }
        let spec = ViewSpec { edge, base_slot, axes };
        self.views.push((map, edge, id, spec));
        Some((id, spec))
    }

    fn take(&mut self, len: u32) -> u32 {
        if let Some(&(_, slot)) = self.takes.iter().find(|&&(known, _)| known == len) {
            return slot;
        }
        let slot = self.word(len.saturating_sub(1));
        self.takes.push((len, slot));
        slot
    }

    fn kind(&mut self, kind: ShiftKind) -> Option<ShiftKind> {
        Some(match kind {
            ShiftKind::View { map, edge } => {
                let (id, spec) = self.view(map, edge)?;
                ShiftKind::ParamView { id, spec }
            }
            ShiftKind::Take { len } => ShiftKind::ParamTake { max_slot: self.take(len) },
            other => other,
        })
    }

    fn op(&mut self, op: Op) -> Option<Op> {
        match op.kind {
            OpKind::Shift(kind) => Some(Op::new(OpKind::Shift(self.kind(kind)?), op.ty)),
            _ => Some(op),
        }
    }

    fn program(&mut self, instrs: &[Instr]) -> Option<Vec<Instr>> {
        instrs.iter().map(|instr| Some(Instr { op: self.op(instr.op)?, args: instr.args })).collect()
    }

    fn tail(&mut self, tail: Tail) -> Option<Tail> {
        Some(match tail {
            Tail::StoreMapped(map) => {
                let (id, spec) = self.view(map, Edge::None)?;
                Tail::ParamStoreMapped { id, spec }
            }
            Tail::Scatter { len } => Tail::ParamScatter { len_slot: self.word(len) },
            other => other,
        })
    }

    fn inlined(&mut self, inlined: &Inlined) -> Option<Inlined> {
        let ctxs = inlined
            .ctxs
            .iter()
            .map(|ctx| Some(Ctx { parent: ctx.parent, kind: self.kind(ctx.kind)?, args: ctx.args }))
            .collect::<Option<Vec<_>>>()?;
        let instrs = inlined
            .instrs
            .iter()
            .map(|flat| Some(Flat { op: self.op(flat.op)?, args: flat.args, at: flat.at, child: flat.child }))
            .collect::<Option<Vec<_>>>()?;
        Some(Inlined {
            ctxs,
            instrs,
            main: inlined.main.clone(),
            region: inlined.region.clone(),
            branch: inlined.branch.clone(),
            ops: inlined.ops,
        })
    }
}

/// Whether `key` has a view, gather or scatter to move into the uniform.
fn has_numbers(key: &FusedKey) -> bool {
    matches!(key.tail, Tail::StoreMapped(_) | Tail::Scatter { .. })
        || key.instrs.iter().chain(key.subs.iter().flatten()).any(|instr| matches!(instr.op.kind, OpKind::Shift(ShiftKind::View { .. } | ShiftKind::Take { .. })))
}

/// Rewrites `key` (and `lowered`, its lowering if it has one) to read its views, gathers and
/// scatter from the uniform, appending their numbers to the slots `scalars` fills.
pub(super) fn apply(key: &mut FusedKey, lowered: Option<&Inlined>, scalars: &mut Vec<u32>) {
    if !has_numbers(key) {
        return;
    }
    let mut builder = Builder::new(&key.scalar_tys, scalars);
    let rewritten = (|| {
        let (instrs, subs, inlined);
        let tail = builder.tail(key.tail)?;
        match lowered {
            None => {
                instrs = builder.program(&key.instrs)?;
                subs = key.subs.iter().map(|sub| builder.program(sub)).collect::<Option<Vec<_>>>()?;
                inlined = None;
            }
            Some(lowered) => {
                // The generated code reads the lowering only: the views of the programs it came
                // from get no slots, so that the key says nothing about their numbers.
                let mut unused = Builder::new(&[], &[]);
                instrs = unused.program(&key.instrs)?;
                subs = key.subs.iter().map(|sub| unused.program(sub)).collect::<Option<Vec<_>>>()?;
                inlined = Some(builder.inlined(lowered)?);
            }
        }
        Some((instrs, subs, tail, inlined))
    })();
    let Some((instrs, subs, tail, inlined)) = rewritten else { return };
    let slots = builder.tys.len() > key.scalar_tys.len();
    key.shape = shape_of(&instrs, &subs);
    key.instrs = Arc::new(instrs);
    key.subs = Arc::new(subs);
    key.tail = tail;
    key.scalar_tys = builder.tys;
    *scalars = builder.words;
    key.params = Some(Arc::new(Parameterized { inlined, slots }));
}

/// The distinct views a rewritten key's generated code calls, by id.
pub(super) fn views_used(key: &FusedKey, inlined: Option<&Inlined>) -> Vec<(u16, ViewSpec)> {
    let mut found: Vec<(u16, ViewSpec)> = Vec::new();
    let mut note = |kind: ShiftKind| {
        if let ShiftKind::ParamView { id, spec } = kind
            && !found.iter().any(|&(known, _)| known == id)
        {
            found.push((id, spec));
        }
    };
    match inlined {
        Some(inlined) => {
            inlined.ctxs.iter().for_each(|ctx| note(ctx.kind));
            inlined.instrs.iter().for_each(|flat| {
                if let OpKind::Shift(kind) = flat.op.kind {
                    note(kind);
                }
            });
        }
        None => {
            for instr in key.instrs.iter().chain(key.subs.iter().flatten()) {
                if let OpKind::Shift(kind) = instr.op.kind {
                    note(kind);
                }
            }
        }
    }
    if let Tail::ParamStoreMapped { id, spec } = key.tail {
        note(ShiftKind::ParamView { id, spec });
    }
    found
}

/// The WGSL functions of `views`: `pview{id}(i)` is the flat source index the view reads for output
/// element `i` and, for [`Edge::Fill`], `pview{id}_in(i)` whether it lies inside the source.
pub(super) fn view_helpers(views: &[(u16, ViewSpec)]) -> String {
    if views.is_empty() {
        return String::new();
    }
    let mut src = String::from(EDGE_HELPERS);
    for &(id, spec) in views {
        let mut slot = spec.base_slot;
        let mut take = || {
            slot += 1;
            slot - 1
        };
        let mut raws: [Option<String>; 4] = [None, None, None, None];
        let mut sizes: [Option<String>; 4] = [None, None, None, None];
        let mut terms = Vec::new();
        for a in 0..4 {
            let axis = spec.axes[a];
            if !(axis.active || (spec.edge == Edge::Fill && axis.checked)) {
                continue;
            }
            let mut coord_expr = None;
            if !axis.out_unit && axis.step != StepSpec::Zero {
                let quotient = match axis.out_div {
                    DivRemSpec::None => "i".to_owned(),
                    DivRemSpec::Pow2 => format!("(i >> p.s{})", take()),
                    DivRemSpec::Const(d) => format!("(i / {d}u)"),
                };
                let index_along = match axis.out_rem {
                    DivRemSpec::None => quotient,
                    DivRemSpec::Pow2 => format!("({quotient} & p.s{})", take()),
                    DivRemSpec::Const(m) => format!("({quotient} % {m}u)"),
                };
                let scaled = match axis.step {
                    StepSpec::PosOne => format!("i32({index_along})"),
                    StepSpec::NegOne => format!("(-i32({index_along}))"),
                    _ => format!("(i32({index_along}) * i32(p.s{}))", take()),
                };
                coord_expr = Some(scaled);
            }
            let start = format!("i32(p.s{})", take());
            let raw = match coord_expr {
                None => start,
                Some(scaled) => format!("({scaled} + {start})"),
            };
            let size = if !axis.checked {
                None
            } else if axis.wrap_size != 0 {
                Some(format!("{}i", axis.wrap_size))
            } else {
                Some(format!("i32(p.s{})", take()))
            };
            let stride = (axis.active && !axis.src_fastest).then(|| format!("p.s{}", take()));
            if axis.active {
                let coord = match (&size, spec.edge) {
                    (Some(size), Edge::Clamp | Edge::Fill) => format!("luma_clamp_i({raw}, {size})"),
                    (Some(size), Edge::Mirror) => format!("luma_mirror_i({raw}, {size})"),
                    (Some(size), Edge::Wrap) => format!("luma_wrap_i({raw}, {size})"),
                    _ => raw.clone(),
                };
                terms.push(match stride {
                    Some(stride) => format!("u32({coord}) * {stride}"),
                    None => format!("u32({coord})"),
                });
            }
            raws[a] = Some(raw);
            sizes[a] = size;
        }
        let sum = if terms.is_empty() { "0u".to_owned() } else { terms.join(" + ") };
        let _ = writeln!(src, "fn pview{id}(i: u32) -> u32 {{ return {sum}; }}");
        if spec.edge == Edge::Fill {
            let tests: Vec<String> = (0..4)
                .filter_map(|a| match (&raws[a], &sizes[a]) {
                    (Some(raw), Some(size)) => Some(format!("{raw} >= 0 && {raw} < {size}")),
                    _ => None,
                })
                .collect();
            let all = if tests.is_empty() { "true".to_owned() } else { tests.join(" && ") };
            let _ = writeln!(src, "fn pview{id}_in(i: u32) -> bool {{ return {all}; }}");
        }
    }
    src
}
