//! Moving the extents, offsets and strides of views, gathers, scatters, circular shifts and
//! coordinate generators out of the source of a kernel and into slots of its uniform `p`.
//!
//! A [`FusedKey`] carries the numbers of a `slice` or `pad` in its views, so a loop over the
//! offsets of one slice would compile one kernel per offset, and the extents of the arrays in
//! them and in the `shift`s and `range`s, so the same expression on another grid shape would
//! compile again (about 2 s for the solver of a CFD program, with a cold driver cache). [`apply`]
//! rewrites such a key (and its lowering, if it has one) to the variants that read the numbers
//! from the uniform: of a view only what shapes the code stays in the key (see [`ViewSpec`]).
//! [`apply_tile`] does the same for the extents of a tiled kernel.
//!
//! Dividing an index by an extent (the coordinates of a flat index) is what the numbers cost at
//! run time, against the multiplication by a constant of the compiler. An extent that is a power
//! of two is a shift or a mask from the uniform (a class of its own in the key); any other has the
//! [words of a divisor](divisor_words): below `2^23` elements the quotient is an `f32` product
//! and a correction, above it the high word of a product. Two views that divide by the same
//! number share the words, so that the shader compiler computes the quotient once: the first
//! version, with the words of every view of its own, was 70 % slower on an arithmetic-bound
//! kernel (`euler/per_step`, 2000 x 2000 cells), the one with the quotient shared costs 3 to 5 %
//! (see `codegen::DIV_HELPERS`).
//!
//! Workgroup-staged kernels (`FusedKey::staging`) bake their layout and views into the source
//! and are not rewritten.

use std::fmt::Write;
use std::sync::Arc;

use super::codegen::{udiv, urem, TileSlots, EDGE_HELPERS};
use super::inline::{Ctx, Flat, Inlined};
use super::program::{shape_of, FusedKey, Instr, Tail};
use super::{Edge, ElemTy, Gen, Op, OpKind, ShiftKind, ViewMap};

/// How the step of a view axis enters the index.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum StepSpec {
    Zero,
    PosOne,
    NegOne,
    /// Any other step, read from the uniform.
    Runtime,
}

/// How an index is divided or reduced modulo an extent: a shift or mask from the uniform for a
/// power of two, otherwise the [words of the divisor](divisor_words) from the uniform.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum DivRemSpec {
    None,
    Pow2,
    /// `wide`: the index may be `2^23` or more (an array of that many elements), which the
    /// quick division of `luma_udiv` does not handle (see `codegen::DIV_HELPERS`).
    Runtime { wide: bool },
}

/// The arrays of this many elements or more divide indices that `luma_udiv` cannot handle.
const WIDE: u64 = 1 << 23;

/// Whether an array of `len` elements is one whose indices `luma_udiv` cannot divide.
pub(super) fn is_wide(len: u64) -> bool {
    len >= WIDE
}

/// The words `d`, `m`, `s` and `r` of the division by `d >= 2` (see `luma_udiv` in the kernels):
///
/// - For `n >= 2^23`, `n / d` is `(t + ((n - t) >> 1)) >> s` with `t` the high 32 bits of
///   `m * n` (Granlund and Montgomery, "Division by invariant integers using multiplication",
///   figure 4.1), exact for every `n`: with `l = ceil(log2 d)`, `s = l - 1` and
///   `m = floor(2^32 (2^l - d) / d) + 1`.
/// - Below `2^23` the quotient comes from `r`, the bits of the smallest `f32`
///   that is at least `1 / d`: `n` and the quotient `q` are exact in an `f32`, so `n * r` is at
///   least `n / d` whatever the rounding, and exceeds it by less than
///   `n / d * 1.5 * 2^-23 <= 0.5` (for `d >= 3`; a power of two has an exact `r`): the product
///   rounded to an integer is `q` or `q + 1`, and `q + 1` shows in `n - (q + 1) * d < 0`.
pub(crate) fn divisor_words(d: u32) -> [u32; 4] {
    debug_assert!(d >= 2, "the division by {d}");
    let l = 32 - (d - 1).leading_zeros();
    let m = (((1u64 << l) - u64::from(d)) << 32) / u64::from(d) + 1;
    [d, m as u32, l - 1, reciprocal_up(d).to_bits()]
}

/// The smallest `f32` that is at least `1 / d`.
fn reciprocal_up(d: u32) -> f32 {
    let nearest = (1.0 / f64::from(d)) as f32;
    // `nearest * d >= 1` in exact arithmetic: the product of a 24-bit mantissa and a 32-bit
    // number is exact in 128 bits.
    let reaches = |r: f32| {
        let bits = r.to_bits();
        let (mantissa, exponent) = (u128::from(bits & 0x7F_FFFF | 0x80_0000), (bits >> 23) as i32 - 150);
        // r = mantissa * 2^exponent, with exponent < 0 for every 1 / d, d >= 2.
        mantissa * u128::from(d) >= 1u128 << (-exponent)
    };
    // The nearest `f32` is within half an ulp of `1 / d`: if it reaches it, the one before it
    // does not.
    if reaches(nearest) { nearest } else { f32::from_bits(nearest.to_bits() + 1) }
}

/// What a [`Gen::ParamRange`] computes of the element index `i`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum RangeKind {
    /// Always 0 (an axis of one element).
    Zero,
    /// `i % size`.
    Mod,
    /// `(i / stride) % size`.
    DivMod,
}

/// What of one source axis of a view the generated code depends on; the rest are uniform words.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) struct AxisSpec {
    /// The source has more than one element along the axis (a unit axis adds nothing to the index).
    pub(crate) active: bool,
    /// The output has one element along the axis the coordinate comes from: it is the start.
    pub(crate) out_unit: bool,
    pub(crate) out_div: DivRemSpec,
    pub(crate) out_rem: DivRemSpec,
    pub(crate) src_fastest: bool,
    pub(crate) step: StepSpec,
    /// The edge handling applies: the coordinate may leave the source (its extent is a word).
    pub(crate) checked: bool,
    /// [`Edge::Wrap`] over a power of two: the word of the extent is its mask (the coordinate
    /// is reduced by a bit-wise and).
    pub(crate) wrap_mask: bool,
    /// Where the numbers of the axis are in the uniform.
    pub(crate) slots: AxisSlots,
}

/// The uniform slots of the numbers of one view axis: those of `out_div` and `out_rem` (a shift
/// or mask word, or the [words of a divisor](divisor_words)), the runtime `step`, the `start`, the
/// source extent (`size`, if the edge handling applies) and the source `stride` (if it is not 1).
/// Slots that the axis has no number for are 0. Which slots are equal is part of the key: the
/// views of a kernel that divide by the same number share the slots, so that the shader
/// compiler can compute the quotient once.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub(crate) struct AxisSlots {
    pub(crate) div: u32,
    pub(crate) rem: u32,
    pub(crate) step: u32,
    pub(crate) start: u32,
    pub(crate) size: u32,
    pub(crate) stride: u32,
}

/// What of one axis of a [`ShiftKind::Dims`] shift the generated code depends on; the stride, the
/// extent and the shift are uniform words (in this order, those that are used).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) struct DimsAxis {
    /// The axis has more than one element: a unit axis adds nothing to the index.
    pub(crate) used: bool,
    /// How the index is divided by the stride: not at all if it is 1 (the first axis, or every
    /// one before it is a unit axis); the word is the shift of a power of two, else the stride.
    pub(crate) stride: DivRemSpec,
    /// How the coordinate is reduced modulo the extent: not at all for the last used axis (the
    /// element index is below the element count); the word is the mask of a power of two, else
    /// the extent. Without a remainder a shifted axis still reads the extent.
    pub(crate) rem: DivRemSpec,
    /// The shift is not 0.
    pub(crate) shifted: bool,
    /// The slots of the stride (a shift word or the words of a divisor), of the extent (a mask
    /// word, the words of a divisor, or the extent itself without a remainder) and of the shift;
    /// 0 where the axis has none.
    pub(crate) slots: [u32; 3],
}

/// A view whose numbers are in the uniform, at the slots of its axes.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) struct ViewSpec {
    pub(crate) edge: Edge,
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
    active: false,
    out_unit: true,
    out_div: DivRemSpec::None,
    out_rem: DivRemSpec::None,
    src_fastest: true,
    step: StepSpec::Zero,
    checked: false,
    wrap_mask: false,
    slots: AxisSlots { div: 0, rem: 0, step: 0, start: 0, size: 0, stride: 0 },
};

/// The slots added so far and the views and gathers they belong to.
struct Builder {
    tys: Vec<ElemTy>,
    words: Vec<u32>,
    views: Vec<(ViewMap, Edge, u16, ViewSpec)>,
    takes: Vec<(u32, u32)>,
    dims: Vec<([u32; 4], [u32; 4], ShiftKind)>,
    /// The divisors whose words are in the uniform, and where.
    divisors: Vec<(u32, u32)>,
    /// The other words that expressions share, and where (see [`Builder::shared`]).
    shared: Vec<(u32, u32)>,
}

impl Builder {
    fn new(tys: &[ElemTy], words: &[u32]) -> Builder {
        Builder { tys: tys.to_vec(), words: words.to_vec(), views: Vec::new(), takes: Vec::new(), dims: Vec::new(), divisors: Vec::new(), shared: Vec::new() }
    }

    fn word(&mut self, value: u32) -> u32 {
        let slot = self.tys.len() as u32;
        self.tys.push(ElemTy::U32);
        self.words.push(value);
        slot
    }

    /// The slot of a word of `value` that views and shifts of the kernel with the same one share
    /// (a shift or a mask): the same slot makes the same expressions for the shader compiler to
    /// compute once.
    fn shared(&mut self, value: u32) -> u32 {
        if let Some(&(_, slot)) = self.shared.iter().find(|&&(known, _)| known == value) {
            return slot;
        }
        let slot = self.word(value);
        self.shared.push((value, slot));
        slot
    }

    /// The words of the division by `d` (see [`divisor_words`]).
    fn divisor(&mut self, d: u32) -> u32 {
        if let Some(&(_, slot)) = self.divisors.iter().find(|&&(known, _)| known == d) {
            return slot;
        }
        let slot = self.tys.len() as u32;
        for word in divisor_words(d) {
            self.word(word);
        }
        self.divisors.push((d, slot));
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
        let mut axes = [UNUSED_AXIS; 4];
        let wide = is_wide(map.out.iter().map(|&d| u64::from(d)).product());
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
                DivRemSpec::Runtime { wide }
            };
            let out_rem = if top_out || dim_out <= 1 {
                DivRemSpec::None
            } else if dim_out.is_power_of_two() {
                DivRemSpec::Pow2
            } else {
                DivRemSpec::Runtime { wide }
            };
            let spec = AxisSpec {
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
                wrap_mask: checked && edge == Edge::Wrap && map.src[a].is_power_of_two(),
                slots: AxisSlots::default(),
            };
            if !(spec.active || (edge == Edge::Fill && spec.checked)) {
                continue;
            }
            let mut slots = AxisSlots::default();
            if !spec.out_unit && spec.step != StepSpec::Zero {
                slots.div = match spec.out_div {
                    DivRemSpec::None => 0,
                    DivRemSpec::Pow2 => self.shared(stride_out.trailing_zeros()),
                    DivRemSpec::Runtime { .. } => self.divisor(stride_out),
                };
                slots.rem = match spec.out_rem {
                    DivRemSpec::None => 0,
                    DivRemSpec::Pow2 => self.shared(dim_out - 1),
                    DivRemSpec::Runtime { .. } => self.divisor(dim_out),
                };
                if spec.step == StepSpec::Runtime {
                    slots.step = self.word(axis.step as u32);
                }
            }
            slots.start = self.word(axis.start as u32);
            if spec.checked {
                slots.size = self.word(if spec.wrap_mask { map.src[a] - 1 } else { map.src[a] });
            }
            if spec.active && !spec.src_fastest {
                slots.stride = self.word(src_stride[a] as u32);
            }
            axes[a] = AxisSpec { slots, ..spec };
        }
        let spec = ViewSpec { edge, axes };
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

    /// The shift of an array of `dims` by `shifts` circularly along every axis, its numbers in the
    /// uniform.
    fn dims(&mut self, dims: [u32; 4], shifts: [u32; 4]) -> ShiftKind {
        if let Some(&(.., kind)) = self.dims.iter().find(|&&(known, by, _)| known == dims && by == shifts) {
            return kind;
        }
        let mut axes = [DimsAxis { used: false, stride: DivRemSpec::None, rem: DivRemSpec::None, shifted: false, slots: [0; 3] }; 4];
        let mut stride = 1u32;
        let last = (0..4).rev().find(|&a| dims[a] > 1);
        let wide = is_wide(dims.iter().map(|&d| u64::from(d)).product());
        let class = |d: u32| if d.is_power_of_two() { DivRemSpec::Pow2 } else { DivRemSpec::Runtime { wide } };
        for a in 0..4 {
            if dims[a] > 1 {
                let mut axis = DimsAxis {
                    used: true,
                    stride: if stride == 1 { DivRemSpec::None } else { class(stride) },
                    rem: if last.is_some_and(|last| a < last) { class(dims[a]) } else { DivRemSpec::None },
                    shifted: shifts[a] != 0,
                    slots: [0; 3],
                };
                axis.slots[0] = match axis.stride {
                    DivRemSpec::None => 0,
                    DivRemSpec::Pow2 => self.shared(stride.trailing_zeros()),
                    DivRemSpec::Runtime { .. } => self.divisor(stride),
                };
                axis.slots[1] = match (axis.rem, axis.shifted) {
                    (DivRemSpec::None, false) => 0,
                    (DivRemSpec::Pow2, _) => self.shared(dims[a] - 1),
                    (DivRemSpec::Runtime { .. }, _) => self.divisor(dims[a]),
                    (DivRemSpec::None, true) => self.word(dims[a]),
                };
                if axis.shifted {
                    axis.slots[2] = self.word(shifts[a]);
                }
                axes[a] = axis;
            }
            stride = stride.wrapping_mul(dims[a]);
        }
        let kind = ShiftKind::ParamDims { axes };
        self.dims.push((dims, shifts, kind));
        kind
    }

    fn kind(&mut self, kind: ShiftKind) -> Option<ShiftKind> {
        Some(match kind {
            ShiftKind::Dims { dims, shifts } => self.dims(dims, shifts),
            ShiftKind::View { map, edge } => {
                let (id, spec) = self.view(map, edge)?;
                ShiftKind::ParamView { id, spec }
            }
            ShiftKind::Take { len } => ShiftKind::ParamTake { max_slot: self.take(len) },
            other => other,
        })
    }

    /// The generator that reads its numbers from the uniform instead of keeping them in the key.
    fn range(&mut self, stride: u32, size: u32, wide: bool) -> Gen {
        match (stride, size) {
            (_, 0 | 1) => Gen::ParamRange { kind: RangeKind::Zero, wide, stride: 0, size: 0 },
            (0 | 1, _) => Gen::ParamRange { kind: RangeKind::Mod, wide, stride: 0, size: self.divisor(size) },
            _ => {
                let stride = self.divisor(stride);
                Gen::ParamRange { kind: RangeKind::DivMod, wide, stride, size: self.divisor(size) }
            }
        }
    }

    fn op(&mut self, op: Op) -> Option<Op> {
        match op.kind {
            OpKind::Shift(kind) => Some(Op::new(OpKind::Shift(self.kind(kind)?), op.ty)),
            OpKind::Generate(Gen::RangeConst { stride, size, wide }) => Some(Op::new(OpKind::Generate(self.range(stride, size, wide)), op.ty)),
            _ => Some(op),
        }
    }

    /// The op of a lowered instruction. Its view is the context's (`child`), which the generated
    /// code indexes through, except for the test whether a [`Edge::Fill`] read lies inside the
    /// source; any other view an instruction names is not read, and when the context is the
    /// composition of views (or none) it is not even the view the code applies. It gets no slots
    /// then, and the key no numbers: a uniform no code reads has no binding.
    fn flat_op(&mut self, op: Op) -> Option<Op> {
        match op.kind {
            OpKind::Shift(ShiftKind::View { edge, .. }) if edge != Edge::Fill => Some(Op::new(OpKind::Shift(ShiftKind::Linear), op.ty)),
            OpKind::Shift(ShiftKind::Dims { .. }) => Some(Op::new(OpKind::Shift(ShiftKind::Linear), op.ty)),
            _ => self.op(op),
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
            Tail::ScatterAdd { len, op, private: false, .. } => Tail::ScatterAdd { len: 0, op, private: false, len_slot: Some(self.word(len)) },
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
            .map(|flat| Some(Flat { op: self.flat_op(flat.op)?, args: flat.args, at: flat.at, child: flat.child }))
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
    matches!(key.tail, Tail::StoreMapped(_) | Tail::Scatter { .. } | Tail::ScatterAdd { private: false, .. })
        || key.instrs.iter().chain(key.subs.iter().flatten()).any(|instr| {
            matches!(instr.op.kind, OpKind::Shift(ShiftKind::View { .. } | ShiftKind::Take { .. } | ShiftKind::Dims { .. }) | OpKind::Generate(Gen::RangeConst { .. }))
        })
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

/// Rewrites the tiling of `key` (see [`Tile`]) to read the extents of its array from the uniform,
/// appending them to the slots `scalars` fills.
pub(super) fn apply_tile(key: &mut FusedKey, scalars: &mut Vec<u32>) {
    let Some(tile) = &mut key.tile else { return };
    if tile.slots.is_some() {
        return;
    }
    let mut builder = Builder::new(&key.scalar_tys, scalars);
    let (w, h) = tile.dims;
    let base = builder.word(w);
    builder.word(h);
    builder.word(w.div_ceil(tile.tile.0));
    tile.slots = Some(TileSlots { base, one_d: h == 1, small: tile.fits() });
    // The shifts read through the tile are the offsets of the tile; the extents they carry are
    // not read.
    let canonical = |instrs: &[Instr], offsets: &[Option<(i32, i32)>]| -> Vec<Instr> {
        instrs
            .iter()
            .zip(offsets)
            .map(|(instr, offset)| match (instr.op.kind, offset) {
                (OpKind::Shift(ShiftKind::Dims { .. }), Some(_)) => Instr { op: Op::new(OpKind::Shift(ShiftKind::Dims { dims: [1; 4], shifts: [0; 4] }), instr.op.ty), args: instr.args },
                _ => instr.clone(),
            })
            .collect()
    };
    let instrs = canonical(&key.instrs, &tile.offsets[0]);
    let subs: Vec<Vec<Instr>> = key.subs.iter().zip(&tile.offsets[1..]).map(|(sub, offsets)| canonical(sub, offsets)).collect();
    key.shape = shape_of(&instrs, &subs);
    key.instrs = Arc::new(instrs);
    key.subs = Arc::new(subs);
    key.scalar_tys = builder.tys;
    *scalars = builder.words;
    key.params = Some(Arc::new(Parameterized { inlined: None, slots: true }));
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
        let mut raws: [Option<String>; 4] = [None, None, None, None];
        let mut sizes: [Option<String>; 4] = [None, None, None, None];
        let mut terms = Vec::new();
        for a in 0..4 {
            let axis = spec.axes[a];
            if !(axis.active || (spec.edge == Edge::Fill && axis.checked)) {
                continue;
            }
            let slots = axis.slots;
            let mut coord_expr = None;
            if !axis.out_unit && axis.step != StepSpec::Zero {
                let quotient = match axis.out_div {
                    DivRemSpec::None => "i".to_owned(),
                    DivRemSpec::Pow2 => format!("(i >> p.s{})", slots.div),
                    DivRemSpec::Runtime { wide } => udiv("i", slots.div, wide),
                };
                let index_along = match axis.out_rem {
                    DivRemSpec::None => quotient,
                    DivRemSpec::Pow2 => format!("({quotient} & p.s{})", slots.rem),
                    DivRemSpec::Runtime { wide } => urem(&quotient, slots.rem, wide),
                };
                let scaled = match axis.step {
                    StepSpec::PosOne => format!("i32({index_along})"),
                    StepSpec::NegOne => format!("(-i32({index_along}))"),
                    _ => format!("(i32({index_along}) * i32(p.s{}))", slots.step),
                };
                coord_expr = Some(scaled);
            }
            let start = format!("i32(p.s{})", slots.start);
            let raw = match coord_expr {
                None => start,
                Some(scaled) => format!("({scaled} + {start})"),
            };
            let size = axis.checked.then(|| format!("i32(p.s{})", slots.size));
            let stride = (axis.active && !axis.src_fastest).then(|| format!("p.s{}", slots.stride));
            if axis.active {
                let coord = match (&size, spec.edge) {
                    (Some(size), Edge::Clamp | Edge::Fill) => format!("luma_clamp_i({raw}, {size})"),
                    (Some(size), Edge::Mirror) => format!("luma_mirror_i({raw}, {size})"),
                    (Some(size), Edge::Wrap) if axis.wrap_mask => format!("luma_wrap_mask({raw}, u32({size}))"),
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

#[cfg(test)]
mod tests {
    use super::divisor_words;

    /// `luma_mulhi` of the generated kernels, in the same 32-bit wrapping operations.
    fn mulhi(a: u32, b: u32) -> u32 {
        let (a0, a1, b0, b1) = (a & 0xFFFF, a >> 16, b & 0xFFFF, b >> 16);
        let (m1, m2) = (a1.wrapping_mul(b0), a0.wrapping_mul(b1));
        let carry = ((a0.wrapping_mul(b0) >> 16) + (m1 & 0xFFFF) + (m2 & 0xFFFF)) >> 16;
        a1.wrapping_mul(b1).wrapping_add(m1 >> 16).wrapping_add(m2 >> 16).wrapping_add(carry)
    }

    /// `luma_udiv` of the generated kernels: for `n < 2^23`. `fused` is whether the compiler
    /// computes the sum of the product with one rounding.
    fn udiv_small(n: u32, [d, _, _, r]: [u32; 4], fused: bool) -> u32 {
        let f = f32::from_bits(n | 0x4B00_0000) - 8_388_608.0;
        let r = f32::from_bits(r);
        let sum = if fused { f.mul_add(r, 8_388_608.0) } else { f * r + 8_388_608.0 };
        let q = sum.to_bits() & 0x7F_FFFF;
        q - u32::from((n.wrapping_sub(q.wrapping_mul(d)) as i32) < 0)
    }

    /// `luma_udiv_wide` of the generated kernels: for every `n`.
    fn udiv(n: u32, [_, m, s, _]: [u32; 4]) -> u32 {
        let t = mulhi(n, m);
        t.wrapping_add(n.wrapping_sub(t) >> 1) >> s
    }

    #[test]
    fn the_high_word_of_a_product_is_exact() {
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 16) as u32
        };
        let edges = [0, 1, 0xFFFF, 0x1_0000, 0x7FFF_FFFF, 0x8000_0000, 0xFFFF_FFFE, u32::MAX];
        for &a in &edges {
            for &b in &edges {
                assert_eq!(mulhi(a, b), ((u64::from(a) * u64::from(b)) >> 32) as u32, "{a} * {b}");
            }
        }
        for _ in 0..200_000 {
            let (a, b) = (next(), next());
            assert_eq!(mulhi(a, b), ((u64::from(a) * u64::from(b)) >> 32) as u32, "{a} * {b}");
        }
    }

    #[test]
    fn the_division_by_a_divisor_in_the_uniform_is_exact_for_every_dividend() {
        let mut state = 0xD1B5_4A32_D192_ED03u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 16) as u32
        };
        // Every divisor below 3000, the powers of two and their neighbours, and random ones of
        // every size; for each the dividends at both ends, around multiples of the divisor
        // (where an error of one shows) and random ones.
        let mut divisors: Vec<u32> = (2..3000).collect();
        for bit in 1..32 {
            divisors.extend([(1u32 << bit).saturating_sub(1).max(2), 1 << bit, (1u32 << bit).saturating_add(1)]);
        }
        divisors.extend((0..2000).map(|_| (next() >> (next() % 31)).max(2)));
        divisors.extend([u32::MAX, u32::MAX - 1, 0x8000_0001, 0x7FFF_FFFF]);
        for d in divisors {
            let words = divisor_words(d);
            let check = |n: u32| assert_eq!(udiv(n, words), n / d, "{n} / {d}");
            for n in [0, 1, d - 1, d, d.saturating_add(1), u32::MAX, u32::MAX - 1, 0x8000_0000, 0x7FFF_FFFF] {
                check(n);
            }
            for _ in 0..200 {
                check(next());
                let multiple = (next() / d).saturating_mul(d);
                check(multiple);
                check(multiple.wrapping_sub(1));
                check(multiple.saturating_add(d - 1));
            }
            let top = u32::MAX / d * d;
            for n in [top.wrapping_sub(1), top, top.saturating_add(d - 1)] {
                check(n);
            }
        }
    }

    #[test]
    fn the_quotient_below_two_to_the_23_is_exact() {
        for d in [2u32, 3, 4, 7, 66, 130] {
            let words = divisor_words(d);
            for n in 0..0x80_0000u32 {
                assert_eq!(udiv_small(n, words, false), n / d, "{n} / {d}");
                assert_eq!(udiv_small(n, words, true), n / d, "{n} / {d} fused");
            }
        }
        // The largest quotients are where the product is furthest above the quotient.
        for d in (2..5000u32).chain([65_535, 65_536, 65_537, 1 << 20, (1 << 20) + 1, 0x7FF_FFFF, 0xFF_FFFF, 0x100_0000, 0x100_0001]) {
            let words = divisor_words(d);
            for n in (0x80_0000u32.saturating_sub(3 * d)..0x80_0000).chain(0..(3 * d).min(0x80_0000)) {
                assert_eq!(udiv_small(n, words, false), n / d, "{n} / {d}");
                assert_eq!(udiv_small(n, words, true), n / d, "{n} / {d} fused");
            }
        }
    }
}
