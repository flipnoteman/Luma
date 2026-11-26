//! The comprehension form: one description of "for every point of a space, read some accesses, compute a body, combine along the
//! reduce dims" (07 §2.1). Reductions, `dot`, the `matmul` family, convolutions, windowed filters and segmented reductions all have
//! it as their meaning; the form carries the operation's origin tag, so the specialisation that runs is the existing one called with the
//! same arguments (the API guarantee of 07 §4.1).
//!
//! In this part the form is plain data: [`build`] makes it (reductions build one per value, see `crate::reduce`; the eager
//! families have pure builders for tests, `explain` and recognition), [`Comprehension::validate`] checks it and
//! [`Comprehension::structure_hash`] keys memos and caches on its structure. Recognition, rewrite rules and the generic lowering
//! read it later; until they do, most of the module is not used by the library itself, which the `expect` below says.
//!
//! Spaces are listed fastest axis first. A body reads the value of access `j` at the point as `Ref::Input(j)` and the scalars of the
//! body as `Ref::Scalar(k)`; an instruction `Generate(Gen::RangeConst { stride, size, .. })` in a body is the coordinate of the dim
//! whose `stride` (the product of the extents of the faster dims) and `size` it names.

#![cfg_attr(not(test), expect(dead_code, reason = "R0a defines the form; its readers are recognition (R1), the generic lowering (R0b), the rules (R3) and the lazy library ops"))]
#![cfg_attr(test, allow(dead_code))]

use std::hash::{Hash, Hasher};
use std::sync::Arc;

use smallvec::SmallVec;

use crate::builtin::{Finish, ReduceList, ReduceOp};
use crate::gpu::hash::FxHasher;
use crate::image::window::Op as WindowOp;
use crate::image::ConvSpec;
use crate::jit::program::{Instr, Ref};
use crate::jit::{Edge, ElemTy, Scalar, Src};

pub(crate) mod build;
mod combiner;
pub(crate) mod inline;
mod map;
#[cfg(test)]
mod recognition_tests;
#[cfg(test)]
mod reference;
pub(crate) mod roles;
#[cfg(test)]
mod tests;

pub(crate) use combiner::{CombineOp, CombinePart, Combiner};
pub(crate) use map::{AffineMap, MapRef};
pub(crate) use roles::classify;

/// 07 open question 8, resolved: six dims, four inputs, four source rows (luma arrays have four axes).
pub(crate) const MAX_DIMS: usize = 6;
pub(crate) const MAX_INPUTS: usize = 4;
pub(crate) const MAX_ROWS: usize = 4;

/// The `LUMA_ALGEBRA` switch (removed before the release): `off` makes the builders skip the form, so the operations run as they
/// did before the form existed; `api` builds the forms and lets only `Origin::Api` lower; `on` (the default) is everything the
/// later specs add (recognition, rules; W2-R3 inserts `t1` between `api` and `on`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Algebra {
    Off,
    Api,
    On,
}

impl Algebra {
    /// What `LUMA_ALGEBRA` says; anything else is the default.
    pub(crate) fn parse(text: &str) -> Option<Algebra> {
        match text.trim().to_ascii_lowercase().as_str() {
            "off" | "0" | "false" => Some(Algebra::Off),
            "api" => Some(Algebra::Api),
            "on" | "1" | "true" => Some(Algebra::On),
            _ => None,
        }
    }

    /// The switch of this process.
    #[inline(always)]
    pub(crate) fn current() -> Algebra {
        static MODE: std::sync::LazyLock<Algebra> = std::sync::LazyLock::new(|| std::env::var("LUMA_ALGEBRA").ok().and_then(|text| Algebra::parse(&text)).unwrap_or(Algebra::On));
        *MODE
    }

    /// Whether the operations build their forms.
    #[inline(always)]
    pub(crate) fn builds_forms() -> bool {
        Algebra::current() != Algebra::Off
    }
}

/// What a form is a form of; the shape of a form together with its origin says which strategy specialises it.
#[derive(Clone, Debug)]
pub(crate) struct Comprehension {
    pub(crate) space: SmallVec<[Dim; 4]>,
    pub(crate) inputs: SmallVec<[Access; 2]>,
    pub(crate) body: Body,
    /// `Some` iff some dim is `Reduce` or `Segmented`.
    pub(crate) combiner: Option<Combiner>,
    pub(crate) epilogue: Epilogue,
    pub(crate) output: OutMap,
    pub(crate) props: Props,
    pub(crate) origin: Origin,
    /// The element type of the result.
    pub(crate) ty: ElemTy,
}

#[derive(Clone, Debug)]
pub(crate) struct Dim {
    pub(crate) extent: u64,
    pub(crate) kind: DimKind,
}

#[derive(Clone, Debug)]
pub(crate) enum DimKind {
    Parallel,
    Reduce,
    /// A reduce dim whose range depends on parallel dim `selector`: for coordinate `c` of the selector the dim runs over
    /// `offsets[c]..offsets[c + 1]` (CSR row pointers, group ids). `extent` is the upper bound (`offsets[last]`, read back or known
    /// from the operand length).
    Segmented { offsets: Box<Src>, selector: u8 },
}

impl Dim {
    pub(crate) const fn parallel(extent: u64) -> Dim {
        Dim { extent, kind: DimKind::Parallel }
    }

    pub(crate) const fn reduce(extent: u64) -> Dim {
        Dim { extent, kind: DimKind::Reduce }
    }

    pub(crate) fn is_reduce(&self) -> bool {
        !matches!(self.kind, DimKind::Parallel)
    }
}

/// A read: the point of the space to the element of `src` it reads.
#[derive(Clone, Debug)]
pub(crate) struct Access {
    /// A node (a prologue: it stays an expression) or a buffer.
    pub(crate) src: Src,
    /// The shape `map` addresses: the source is read row-major, flat, and the product is the source's length (a node is evaluated by
    /// flat index, so reading `[len, 1, 1, 1]` of a four-axis array is exact).
    pub(crate) shape: [u32; 4],
    pub(crate) ty: ElemTy,
    pub(crate) map: MapRef,
    /// What a coordinate outside the shape reads (`Edge::None`: never happens, checked at build).
    pub(crate) edge: Edge,
    /// The value of `Edge::Fill`.
    pub(crate) fill: Option<Scalar>,
    pub(crate) indirect: Option<Box<Indirect>>,
}

/// Source row `row` is `index[index_map(point)] + rows[row](point)`; the other rows are as in the access's `map`. `x[col[j]]` is one
/// row whose affine part is 0. The edge applies to the final coordinate (`Clamp` is `ShiftKind::Take`).
#[derive(Clone, Debug)]
pub(crate) struct Indirect {
    pub(crate) index: Src,
    pub(crate) index_shape: [u32; 4],
    pub(crate) index_map: AffineMap,
    pub(crate) row: u8,
}

/// The values computed at a point. `Ref::Input(j)` is the value of access `j`, `Ref::Scalar(k)` is `scalars[k]`; `outs` lists the
/// instructions whose values are the outputs, one per part of a tuple combiner.
#[derive(Clone, Debug)]
pub(crate) struct Body {
    pub(crate) instrs: Arc<Vec<Instr>>,
    pub(crate) scalars: SmallVec<[Scalar; 2]>,
    pub(crate) outs: SmallVec<[u32; 4]>,
}

/// What is applied to the combined values.
#[derive(Clone, Debug)]
pub(crate) enum Epilogue {
    None,
    /// Today's reduce epilogue (`reduce::Finish`), so that `fuse_step` keeps working on a comprehension value. W3-U4 adds `Program`.
    Steps(Finish),
    /// A program over the combined values: `Ref::Input(j)` is combined part `j`, its last instruction the result (the bilateral
    /// division of the two sums, the Sobel magnitude). W3-U4 absorbs consumers into it.
    Program(Arc<Vec<Instr>>),
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum LayoutChoice {
    Default,
    /// The layout the consumer asks for (W2-R1 `prepare`, wave 3).
    Consumer,
}

#[derive(Clone, Debug)]
pub(crate) struct OutMap {
    /// Output axis to the parallel dim it follows (`None`: extent 1).
    pub(crate) dim_of_axis: [Option<u8>; 4],
    /// The physical order of the axes, fastest first; the identity everywhere in this part (a strategy that cannot honour another
    /// order rejects the form).
    pub(crate) order: [u8; 4],
    pub(crate) chosen: LayoutChoice,
}

impl OutMap {
    pub(crate) const IDENTITY: [u8; 4] = [0, 1, 2, 3];

    pub(crate) const fn new(dim_of_axis: [Option<u8>; 4]) -> OutMap {
        OutMap { dim_of_axis, order: OutMap::IDENTITY, chosen: LayoutChoice::Default }
    }
}

/// Matrix properties of the output (07 §5): facts, so that `General` is no flags. Only a type in this part: every form has
/// [`Props::NONE`] until R6 fills it.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) struct Props {
    pub(crate) flags: u16,
    /// Lower and upper bandwidth.
    pub(crate) band: Option<(u32, u32)>,
    /// Block diagonal with square blocks of this edge.
    pub(crate) block: Option<u32>,
    pub(crate) rank: Option<u32>,
}

impl Props {
    pub(crate) const IDENTITY: u16 = 1 << 0;
    pub(crate) const DIAGONAL: u16 = 1 << 1;
    pub(crate) const PERMUTATION: u16 = 1 << 2;
    pub(crate) const LOWER: u16 = 1 << 3;
    pub(crate) const UPPER: u16 = 1 << 4;
    pub(crate) const SYMMETRIC: u16 = 1 << 5;
    pub(crate) const NONE: Props = Props { flags: 0, band: None, block: None, rank: None };

    /// What holds for the values of either of two alternatives (the facts both have; the weaker bound of each number).
    pub(crate) fn meet(self, other: Props) -> Props {
        Props {
            flags: self.flags & other.flags,
            band: self.band.zip(other.band).map(|(a, b)| (a.0.max(b.0), a.1.max(b.1))),
            block: self.block.zip(other.block).and_then(|(a, b)| lcm(a, b)),
            rank: self.rank.zip(other.rank).map(|(a, b)| a.max(b)),
        }
    }

    /// What holds for a value that both describe (the facts of either; the stronger bound of each number).
    pub(crate) fn join(self, other: Props) -> Props {
        let stronger = |a: Option<u32>, b: Option<u32>| match (a, b) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (a, b) => a.or(b),
        };
        Props {
            flags: self.flags | other.flags,
            band: match (self.band, other.band) {
                (Some(a), Some(b)) => Some((a.0.min(b.0), a.1.min(b.1))),
                (a, b) => a.or(b),
            },
            block: match (self.block, other.block) {
                (Some(a), Some(b)) => Some(gcd(a, b)),
                (a, b) => a.or(b),
            },
            rank: stronger(self.rank, other.rank),
        }
    }
}

fn gcd(a: u32, b: u32) -> u32 {
    if b == 0 { a } else { gcd(b, a % b) }
}

fn lcm(a: u32, b: u32) -> Option<u32> {
    (a / gcd(a, b).max(1)).checked_mul(b)
}

/// Where a form came from. `Api(tag)`: made for an API call, so the strategy of the tag runs (today's code, the same arguments).
/// `Recognised(form)`: found in a hand-written pattern (W2-R1); `Rewritten(rule)`: made by a rewrite rule (W2-R3). Neither has an
/// API strategy: only the generic lowering takes them when the recogniser declines.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub(crate) enum Origin {
    Api(OpTag),
    Recognised(Form),
    Rewritten(RuleId),
}

/// Which rewrite rule made a form. The rule table is W2-R3's; the generic lowering only needs to tell the origin from an API tag.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) struct RuleId(pub(crate) u16);

/// Which API operation a value or form is. Not `Copy`: the convolution's spec is boxed, so that the tag every reduction carries is
/// the size of its op list and not of the largest spec.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub(crate) enum OpTag {
    /// A reduction of the whole array, with up to four ops.
    Reduce(ReduceList),
    /// A reduction along lines, with one op: its buffer holds one value per line.
    ReduceAxis(ReduceList),
    Arg { max: bool, axis: bool },
    Segments(ReduceOp),
    /// `(lhs * rhs).sum()`: a whole reduction of the product node.
    Dot,
    /// `matmul` (`tf32: None`: the context's numerics), `matmul_f32` (`Some(false)`), `matmul_tf32` (`Some(true)`).
    Matmul { tf32: Option<bool> },
    MatmulBatched,
    Conv1d,
    /// `conv2` (`false`) or `conv3` (`true`).
    Conv(Box<ConvSpec>, bool),
    ConvSep,
    Window(WindowOp),
}

impl OpTag {
    /// Whether this is a reduction of the whole array: a `Reduce`, or a `Dot`, which is one over the product node.
    #[inline(always)]
    pub(crate) fn is_whole_reduce(&self) -> bool {
        matches!(self, OpTag::Reduce(_) | OpTag::Dot)
    }

    /// The ops of a reduction (a `dot` is a sum); `None` for the other operations.
    pub(crate) fn reduce_list(&self) -> Option<ReduceList> {
        match self {
            OpTag::Reduce(ops) | OpTag::ReduceAxis(ops) => Some(*ops),
            OpTag::Dot => Some(ReduceList::new(&[ReduceOp::Sum])),
            _ => None,
        }
    }

    /// How many values the operation makes per output: the ops of a reduction, else one.
    #[inline(always)]
    pub(crate) fn op_count(&self) -> usize {
        match self {
            OpTag::Reduce(ops) | OpTag::ReduceAxis(ops) => ops.as_slice().len(),
            _ => 1,
        }
    }
}

/// The class of a form (W2-R1 §1): what [`classify`] says it is, which sets the strategy of a form without a tag.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum Form {
    /// `sum_k lhs * rhs` with parallel dims on each side.
    Gemm,
    /// A contraction with parallel dims on one side only.
    Gemv,
    /// A contraction with no parallel dim of its own on either side (batched or not).
    Dot,
    /// A product of two operands with disjoint parallel dims and nothing reduced.
    Outer,
    /// A reduction of identity-mapped accesses.
    Reduce,
    /// A row of some access reads two or more dims: a convolution or a filter.
    Window,
    Elementwise,
    /// Anything else: only the generic lowering takes it.
    Generic,
}

impl OpTag {
    /// The classes an API form of this tag may have. A degenerate extent (a dim of extent 1 has no role) moves a form to a simpler
    /// class: a product with `K = 1` is an outer product, a reduction of one element is elementwise, a `dot` is a reduction of one
    /// access (its form has no second operand), a convolution with one output or one tap has no window left.
    pub(crate) fn expected_forms(&self) -> &'static [Form] {
        match self {
            OpTag::Reduce(_) | OpTag::ReduceAxis(_) | OpTag::Arg { .. } | OpTag::Segments(_) => &[Form::Reduce, Form::Elementwise],
            OpTag::Dot => &[Form::Dot, Form::Reduce, Form::Elementwise],
            OpTag::Matmul { .. } | OpTag::MatmulBatched => &[Form::Gemm, Form::Gemv, Form::Dot, Form::Outer, Form::Elementwise],
            OpTag::Conv1d | OpTag::Conv(..) | OpTag::ConvSep | OpTag::Window(_) => &[Form::Window, Form::Elementwise, Form::Outer, Form::Dot, Form::Gemv, Form::Reduce],
        }
    }
}

/// Why a form is not well formed.
#[derive(Clone, PartialEq, Eq, Debug)]
pub(crate) enum FormError {
    /// More dims, inputs or rows than the limits allow.
    TooMany(&'static str),
    /// A combiner iff a reduce dim.
    Combiner,
    /// The body's parts, outs or refs do not match the accesses and the combiner.
    Body(&'static str),
    /// The output axes do not follow exactly the parallel dims.
    Output,
    /// A segmented dim whose selector is not a parallel dim.
    Selector,
    /// A map row has a coefficient on a dim the space does not have (`input`, `row`).
    MapDim { input: usize, row: usize },
    /// An `Edge::None` access reads outside its shape somewhere in the space (`input`, `row`).
    OutOfRange { input: usize, row: usize },
    /// A `Fill` edge without a fill value, or a fill value without a `Fill` edge.
    Fill(usize),
    /// A shape whose product is not the source's length.
    ShapeLength(usize),
    /// An indirect access whose row is not a source row.
    Indirect(usize),
    /// A coefficient or extent that does not fit the form's integers.
    Coefficient,
    /// An API form whose class is not one its tag expects (`OpTag::expected_forms`): the builder and the op disagree.
    Class(Form),
    /// A window operation that is not a form of the kind asked for, or whose geometry the builder does not cover.
    Window(WindowOp),
}

impl Comprehension {
    /// Whether some dim reduces.
    pub(crate) fn has_reduce(&self) -> bool {
        self.space.iter().any(Dim::is_reduce)
    }

    /// The extents of the dims.
    fn extents(&self) -> SmallVec<[u64; 6]> {
        self.space.iter().map(|dim| dim.extent).collect()
    }

    /// Checks the form: a combiner iff a reduce dim, every `Ref::Input(j)` is an access, the output axes follow exactly the parallel
    /// dims, a segmented selector is a parallel dim, map rows have coefficients on existing dims only, an `Edge::None` map stays in
    /// range at the extremes of the space. Runs in debug builds and tests always; in release at build only when the origin is not
    /// `Api` (an API form is made by the builders, which the tests cover).
    pub(crate) fn validate(&self) -> Result<(), FormError> {
        if self.space.len() > MAX_DIMS {
            return Err(FormError::TooMany("dims"));
        }
        if self.inputs.len() > MAX_INPUTS {
            return Err(FormError::TooMany("inputs"));
        }
        if self.combiner.is_some() != self.has_reduce() {
            return Err(FormError::Combiner);
        }
        let parts = self.combiner.as_ref().map_or(1, |combiner| combiner.op.parts().len());
        if let Some(combiner) = &self.combiner
            && combiner.identity.len() != parts
        {
            return Err(FormError::Combiner);
        }
        if self.body.outs.len() != parts {
            return Err(FormError::Body("one out per part of the combiner"));
        }
        if self.body.outs.iter().any(|&out| out as usize >= self.body.instrs.len()) {
            return Err(FormError::Body("an out is not an instruction"));
        }
        for instr in self.body.instrs.iter() {
            for arg in instr.args {
                match arg {
                    Ref::Input(j) if j as usize >= self.inputs.len() => return Err(FormError::Body("an input is not an access")),
                    Ref::Scalar(k) if k as usize >= self.body.scalars.len() => return Err(FormError::Body("a scalar is not a body scalar")),
                    Ref::Tmp(t) if t as usize >= self.body.instrs.len() => return Err(FormError::Body("a temporary is not an instruction")),
                    _ => {}
                }
            }
        }
        if let Epilogue::Program(instrs) = &self.epilogue {
            for instr in instrs.iter() {
                for arg in instr.args {
                    match arg {
                        Ref::Input(j) if j as usize >= parts => return Err(FormError::Body("an epilogue input is not a combined part")),
                        Ref::Tmp(t) if t as usize >= instrs.len() => return Err(FormError::Body("an epilogue temporary is not an instruction")),
                        _ => {}
                    }
                }
            }
        }
        let mut followed = [false; MAX_DIMS];
        for dim in self.output.dim_of_axis.iter().flatten() {
            match followed.get_mut(usize::from(*dim)) {
                Some(slot) if !*slot && self.space.get(usize::from(*dim)).is_some_and(|dim| !dim.is_reduce()) => *slot = true,
                _ => return Err(FormError::Output),
            }
        }
        if self.space.iter().enumerate().any(|(d, dim)| !dim.is_reduce() && !followed[d]) {
            return Err(FormError::Output);
        }
        for dim in &self.space {
            if let DimKind::Segmented { selector, .. } = &dim.kind
                && !self.space.get(usize::from(*selector)).is_some_and(|dim| !dim.is_reduce())
            {
                return Err(FormError::Selector);
            }
        }
        let extents = self.extents();
        for (j, access) in self.inputs.iter().enumerate() {
            let len = match &access.src {
                Src::Node(node) => Some(node.len() as u64),
                Src::Scalar(_) => Some(1),
                Src::Buffer(storage) => Some(storage.len() as u64),
            };
            if len.is_some_and(|len| len != access.shape.iter().map(|&d| u64::from(d)).product::<u64>()) {
                return Err(FormError::ShapeLength(j));
            }
            for (a, row) in access.map.rows.iter().enumerate() {
                if row.coeff.iter().skip(self.space.len()).any(|&c| c != 0) {
                    return Err(FormError::MapDim { input: j, row: a });
                }
                if access.edge == Edge::None
                    && access.indirect.as_ref().is_none_or(|indirect| usize::from(indirect.row) != a)
                    && let Some((lo, hi)) = access.map.row_range(a, &extents)
                    && (lo < 0 || hi >= i128::from(access.shape[a]))
                {
                    return Err(FormError::OutOfRange { input: j, row: a });
                }
            }
            if (access.edge == Edge::Fill) != access.fill.is_some() {
                return Err(FormError::Fill(j));
            }
            if access.indirect.as_ref().is_some_and(|indirect| usize::from(indirect.row) >= MAX_ROWS) {
                return Err(FormError::Indirect(j));
            }
        }
        // The guarantee of 07 §4.1: an API form is of the class its tag says.
        if let Origin::Api(tag) = &self.origin {
            let class = classify(self);
            if !tag.expected_forms().contains(&class) {
                return Err(FormError::Class(class));
            }
        }
        Ok(())
    }

    /// The structure of the form: extents, kinds, maps, edges, body, combiner, epilogue shape, origin tag and types; **not** the
    /// identities of sources or the values of scalars. The key of the T2 memo (W2-R3) and part of the selection-cache key.
    pub(crate) fn structure_hash(&self) -> u64 {
        let mut h = FxHasher::default();
        h.write_u8(self.space.len() as u8);
        for dim in &self.space {
            dim.extent.hash(&mut h);
            match &dim.kind {
                DimKind::Parallel => h.write_u8(0),
                DimKind::Reduce => h.write_u8(1),
                DimKind::Segmented { selector, .. } => {
                    h.write_u8(2);
                    h.write_u8(*selector);
                }
            }
        }
        h.write_u8(self.inputs.len() as u8);
        for access in &self.inputs {
            access.shape.hash(&mut h);
            access.ty.hash(&mut h);
            (*access.map).hash(&mut h);
            access.edge.hash(&mut h);
            access.fill.is_some().hash(&mut h);
            match &access.indirect {
                None => h.write_u8(0),
                Some(indirect) => {
                    h.write_u8(1);
                    indirect.index_shape.hash(&mut h);
                    indirect.index_map.hash(&mut h);
                    indirect.row.hash(&mut h);
                }
            }
        }
        self.body.instrs.hash(&mut h);
        self.body.scalars.len().hash(&mut h);
        self.body.outs.hash(&mut h);
        match &self.combiner {
            None => h.write_u8(0),
            Some(combiner) => {
                h.write_u8(1);
                combiner.op.hash(&mut h);
                combiner.commutative.hash(&mut h);
                combiner.reorder_exact.hash(&mut h);
                combiner.compensation.hash(&mut h);
            }
        }
        match &self.epilogue {
            Epilogue::None => h.write_u8(0),
            Epilogue::Steps(finish) => {
                h.write_u8(1);
                finish.epilogue.hash(&mut h);
            }
            Epilogue::Program(instrs) => {
                h.write_u8(2);
                instrs.hash(&mut h);
            }
        }
        self.output.dim_of_axis.hash(&mut h);
        self.output.order.hash(&mut h);
        self.output.chosen.hash(&mut h);
        self.props.hash(&mut h);
        self.origin.hash(&mut h);
        self.ty.hash(&mut h);
        h.finish()
    }
}
