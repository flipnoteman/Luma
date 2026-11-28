//! The builders: the form of each API operation, as pure functions of the operation's arguments. The reductions call theirs when they
//! make a value (`crate::reduce`); the others are called where something reads the form (tests, `explain`, recognition), because
//! on the eager path nothing could read it before the value is evaluated.

use std::sync::{Arc, OnceLock};

use smallvec::{smallvec, SmallVec};

use super::map::{AffineMap, AffineRow, MapRef};
use super::{Access, Body, CombineOp, CombinePart, Combiner, Comprehension, Dim, DimKind, Epilogue, FormError, OpTag, Origin, OutMap, Props, MAX_DIMS};
use crate::builtin::{BinOp, CmpOp, Finish, ReduceList, ReduceOp, UnOp};
use crate::image::window::Op as WindowOp;
use crate::image::{Border, ConvSpec};
use crate::jit::program::{Instr, Ref};
use crate::jit::structured::contraction::{ContractionAttrs, Geometry, Tf32Request};
use crate::jit::{Edge, ElemTy, Gen, Op, OpKind, Scalar, Src};
use crate::reduce::AxisGeom;

/// The most element types (the variants of [`ElemTy`]): the interned bodies are one per type.
const TYPES: usize = 13;

/// One cell per element type, filled the first time the type is asked for (the initialiser depends on the type, so these are
/// `OnceLock`s indexed by it, not one `LazyLock` each).
type Table = [OnceLock<Arc<Vec<Instr>>>; TYPES];

/// The bodies of the API operations: interned (one atomic increment to take one, not an allocation), one per element type, since
/// an instruction carries its type.
pub(crate) struct Bodies;

impl Bodies {
    /// `x0`.
    pub(crate) fn x0(ty: ElemTy) -> Arc<Vec<Instr>> {
        static TABLE: Table = [const { OnceLock::new() }; TYPES];
        interned(&TABLE, ty, || vec![unary(UnOp::Id, ty)])
    }

    /// `x0 * x0`.
    pub(crate) fn squared(ty: ElemTy) -> Arc<Vec<Instr>> {
        static TABLE: Table = [const { OnceLock::new() }; TYPES];
        interned(&TABLE, ty, || vec![binary(BinOp::Mul, ty, Ref::Input(0), Ref::Input(0))])
    }

    /// `x0`, `x0 * x0` (a reduction of several ops that has a sum of squares among them).
    pub(crate) fn x0_and_squared(ty: ElemTy) -> Arc<Vec<Instr>> {
        static TABLE: Table = [const { OnceLock::new() }; TYPES];
        interned(&TABLE, ty, || vec![unary(UnOp::Id, ty), binary(BinOp::Mul, ty, Ref::Input(0), Ref::Input(0))])
    }

    /// `x0 * x1`.
    pub(crate) fn x0_times_x1(ty: ElemTy) -> Arc<Vec<Instr>> {
        static TABLE: Table = [const { OnceLock::new() }; TYPES];
        interned(&TABLE, ty, || vec![binary(BinOp::Mul, ty, Ref::Input(0), Ref::Input(1))])
    }

    /// `x0 * x1 * x2` (a separable convolution).
    pub(crate) fn x0_times_x1_times_x2(ty: ElemTy) -> Arc<Vec<Instr>> {
        static TABLE: Table = [const { OnceLock::new() }; TYPES];
        interned(&TABLE, ty, || vec![binary(BinOp::Mul, ty, Ref::Input(0), Ref::Input(1)), binary(BinOp::Mul, ty, Ref::Tmp(0), Ref::Input(2))])
    }
}

#[inline]
fn interned(table: &'static Table, ty: ElemTy, make: impl FnOnce() -> Vec<Instr>) -> Arc<Vec<Instr>> {
    Arc::clone(table[ty as usize].get_or_init(|| Arc::new(make())))
}

fn unary(op: UnOp, ty: ElemTy) -> Instr {
    Instr { op: Op::new(OpKind::Unary(op), ty), args: [Ref::Input(0), Ref::Unused, Ref::Unused] }
}

fn binary(op: BinOp, ty: ElemTy, a: Ref, b: Ref) -> Instr {
    Instr { op: Op::new(OpKind::Binary(op), ty), args: [a, b, Ref::Unused] }
}

// The maps every API reduction and product shares.
/// `[len]`: row 0 follows dim 0.
static WHOLE: AffineMap = AffineMap::following([0]);
/// `[stride, n, outer]`: rows follow dims `s`, `k`, `o`.
static AXIS: AffineMap = AffineMap::following([0, 1, 2]);
/// A segmented reduction reads `x[j]`: row 0 follows dim 1 (`j`).
static SEGMENT: AffineMap = AffineMap::following([1]);
/// `lhs [K, M]` of a product over `(n, m, k)`.
static LHS: AffineMap = AffineMap::following([2, 1]);
/// `rhs [N, K]` of a product over `(n, m, k)`.
static RHS: AffineMap = AffineMap::following([0, 2]);

fn word(extent: usize) -> u32 {
    u32::try_from(extent).unwrap_or(u32::MAX)
}

fn access(src: Src, shape: [u32; 4], ty: ElemTy, map: MapRef) -> Access {
    Access { src, shape, ty, map, edge: Edge::None, fill: None, indirect: None, props: Props::NONE }
}

fn comprehension(space: SmallVec<[Dim; 4]>, inputs: SmallVec<[Access; 2]>, body: Body, combiner: Option<Combiner>, output: OutMap, origin: Origin, ty: ElemTy) -> Comprehension {
    Comprehension { space, inputs, body, combiner, epilogue: Epilogue::None, output, props: Props::NONE, origin, ty }
}

/// The body of a reduction of `ops` over values of type `ty`: a sum of squares adds the squares of the values, the other ops the
/// values; one out per op.
fn reduce_body(ops: &[ReduceOp], ty: ElemTy) -> Body {
    let squares = ops.iter().filter(|&&op| op == ReduceOp::SumSquares).count();
    let (instrs, outs) = if squares == 0 {
        (Bodies::x0(ty), ops.iter().map(|_| 0).collect())
    } else if squares == ops.len() {
        (Bodies::squared(ty), ops.iter().map(|_| 0).collect())
    } else {
        (Bodies::x0_and_squared(ty), ops.iter().map(|&op| u32::from(op == ReduceOp::SumSquares)).collect())
    };
    Body { instrs, scalars: SmallVec::new(), outs }
}

impl Comprehension {
    /// `x.sum()`, `min`, `max`, `product`, `norm`, `reduce([..])`, `sum_where`: a reduce dim over the `len` elements of `src` (the
    /// registered reader node of the operand, see `reduce::run_reduce`), one out per op; the result has `ops.len()` values.
    #[inline]
    pub(crate) fn reduce(src: Src, len: usize, ops: ReduceList, ty: ElemTy) -> Comprehension {
        Comprehension::whole(src, len, ops.as_slice(), ty, OpTag::Reduce(ops))
    }

    /// `a.dot(b)`: `(a * b).sum()`, a reduce dim over the product node `src`, one access.
    pub(crate) fn dot(src: Src, len: usize, ty: ElemTy) -> Comprehension {
        Comprehension::whole(src, len, &[ReduceOp::Sum], ty, OpTag::Dot)
    }

    #[inline]
    fn whole(src: Src, len: usize, ops: &[ReduceOp], ty: ElemTy, tag: OpTag) -> Comprehension {
        comprehension(
            smallvec![Dim::reduce(len as u64)],
            smallvec![access(src, [word(len), 1, 1, 1], ty, MapRef::Static(&WHOLE))],
            reduce_body(ops, ty),
            Some(Combiner::of_ops(ops, ty)),
            OutMap::new([None; 4]),
            Origin::Api(tag),
            ty,
        )
    }

    /// `sum_axis`, `min_axis`, `max_axis`, `reduce_axis` along the lines `geom`: `s` (`stride`) and `o` (`outer`) parallel, `k` (`n`)
    /// reduce; line `l = s + stride * o`, the `AxisGeom` line.
    pub(crate) fn reduce_axis(src: Src, geom: AxisGeom, op: ReduceOp, ty: ElemTy) -> Comprehension {
        let (space, shape) = axis_space(geom);
        let ops = [op];
        comprehension(
            space,
            smallvec![access(src, shape, ty, MapRef::Static(&AXIS))],
            reduce_body(&ops, ty),
            Some(Combiner::of_ops(&ops, ty)),
            OutMap::new([Some(0), Some(2), None, None]),
            Origin::Api(OpTag::ReduceAxis(ReduceList::new(&ops))),
            ty,
        )
    }

    /// `argmax`/`argmin` of the whole array (`geom: None`) or along lines: the value is the body, the index along the reduce dim the
    /// result (a `u32`); ties go to the lowest index.
    pub(crate) fn arg(src: Src, len: usize, geom: Option<AxisGeom>, max: bool, ty: ElemTy) -> Comprehension {
        let part = if max { CombinePart::ArgMax } else { CombinePart::ArgMin };
        let combiner = Combiner::of(CombineOp::Single(part), ty);
        let body = Body { instrs: Bodies::x0(ty), scalars: SmallVec::new(), outs: smallvec![0] };
        let origin = Origin::Api(OpTag::Arg { max, axis: geom.is_some() });
        match geom {
            None => comprehension(smallvec![Dim::reduce(len as u64)], smallvec![access(src, [word(len), 1, 1, 1], ty, MapRef::Static(&WHOLE))], body, Some(combiner), OutMap::new([None; 4]), origin, ElemTy::U32),
            Some(geom) => {
                let (space, shape) = axis_space(geom);
                comprehension(space, smallvec![access(src, shape, ty, MapRef::Static(&AXIS))], body, Some(combiner), OutMap::new([Some(0), Some(2), None, None]), origin, ElemTy::U32)
            }
        }
    }

    /// `sum_segments`, `min_segments`, `max_segments`: `rows` parallel, and a segmented reduce dim over the `len` elements of `src`
    /// that runs for row `r` over `offsets[r]..offsets[r + 1]`.
    pub(crate) fn segments(op: ReduceOp, src: Src, len: usize, offsets: Src, rows: usize, ty: ElemTy) -> Comprehension {
        let ops = [op];
        comprehension(
            smallvec![Dim::parallel(rows as u64), Dim { extent: len as u64, kind: DimKind::Segmented { offsets: Box::new(offsets), selector: 0 } }],
            smallvec![access(src, [word(len), 1, 1, 1], ty, MapRef::Static(&SEGMENT))],
            reduce_body(&ops, ty),
            Some(Combiner::of_ops(&ops, ty)),
            OutMap::new([Some(0), None, None, None]),
            Origin::Api(OpTag::Segments(op)),
            ty,
        )
    }

    /// `matmul(lhs [K, M], rhs [N, K])`, and `matmul_nt/_tn/_tt` (the operand is the array as handed over: `rhs.transpose()` is a
    /// view node, with the same maps): `n` and `m` parallel, `k` reduce, `x0 * x1` added; the result is `[N, M]`.
    pub(crate) fn contract(lhs: Src, rhs: Src, geom: Geometry, attrs: ContractionAttrs) -> Comprehension {
        let ty = ElemTy::F32;
        let (m, n, k) = (geom.m, geom.n, geom.k);
        let tf32 = match attrs.tf32 {
            Tf32Request::Default => None,
            Tf32Request::Exact => Some(false),
            Tf32Request::Tf32 => Some(true),
        };
        comprehension(
            smallvec![Dim::parallel(u64::from(n)), Dim::parallel(u64::from(m)), Dim::reduce(u64::from(k))],
            smallvec![access(lhs, [k, m, 1, 1], ty, MapRef::Static(&LHS)), access(rhs, [n, k, 1, 1], ty, MapRef::Static(&RHS))],
            product_body(ty),
            Some(Combiner::of(CombineOp::Single(CombinePart::Add), ty)),
            OutMap::new([Some(0), Some(1), None, None]),
            Origin::Api(OpTag::Matmul { tf32 }),
            ty,
        )
    }

    /// `matmul_batched` (`[K, M, B]` by `[N, K, B]`, the batch axes 2 and 3 coalesced into one dim `b`): `n`, `m`, `b` parallel, `k`
    /// reduce. An operand that is not batched (`lhs_batched == false`) is shared: its batch row has coefficient 0.
    pub(crate) fn contract_batched(lhs: Src, rhs: Src, [k, m, n, batch]: [u32; 4], lhs_batched: bool, rhs_batched: bool) -> Comprehension {
        let ty = ElemTy::F32;
        // Dims: n, m, b, k.
        let operand = |first: usize, second: usize, batched: bool| {
            let mut map = AffineMap::ZERO;
            map.rows[0] = AffineRow::dim(first);
            map.rows[1] = AffineRow::dim(second);
            if batched {
                map.rows[2] = AffineRow::dim(2);
            }
            MapRef::owned(map)
        };
        let (lb, rb) = (if lhs_batched { batch } else { 1 }, if rhs_batched { batch } else { 1 });
        comprehension(
            smallvec![Dim::parallel(u64::from(n)), Dim::parallel(u64::from(m)), Dim::parallel(u64::from(batch)), Dim::reduce(u64::from(k))],
            smallvec![access(lhs, [k, m, lb, 1], ty, operand(3, 1, lhs_batched)), access(rhs, [n, k, rb, 1], ty, operand(0, 3, rhs_batched))],
            product_body(ty),
            Some(Combiner::of(CombineOp::Single(CombinePart::Add), ty)),
            OutMap::new([Some(0), Some(1), Some(2), None]),
            Origin::Api(OpTag::MatmulBatched),
            ty,
        )
    }

    /// `conv1d(x, w, stride, padding)`: `o` parallel over the `out_len` outputs, `t` reduce over the `taps`; `x` reads
    /// `stride * o + t - padding` (two coefficients) and zero outside, `w` reads `t`.
    pub(crate) fn conv1d(x: Src, w: Src, in_len: usize, taps: usize, out_len: usize, stride: usize, padding: usize) -> Result<Comprehension, FormError> {
        let ty = ElemTy::F32;
        let mut row = AffineRow::ZERO;
        row.coeff[0] = coefficient(stride)?;
        row.coeff[1] = 1;
        row.offset = -i64::try_from(padding).map_err(|_| FormError::Coefficient)?;
        let mut map = AffineMap::ZERO;
        map.rows[0] = row;
        let mut x = access(x, [word(in_len), 1, 1, 1], ty, MapRef::owned(map));
        x.edge = Edge::Fill;
        x.fill = Some(Scalar::f32(0.0));
        Ok(comprehension(
            smallvec![Dim::parallel(out_len as u64), Dim::reduce(taps as u64)],
            smallvec![x, access(w, [word(taps), 1, 1, 1], ty, MapRef::Static(&SEGMENT))],
            product_body(ty),
            Some(Combiner::of(CombineOp::Single(CombinePart::Add), ty)),
            OutMap::new([Some(0), None, None, None]),
            Origin::Api(OpTag::Conv1d),
            ty,
        ))
    }
}

fn product_body(ty: ElemTy) -> Body {
    Body { instrs: Bodies::x0_times_x1(ty), scalars: SmallVec::new(), outs: smallvec![0] }
}

fn coefficient(value: usize) -> Result<i32, FormError> {
    i32::try_from(value).map_err(|_| FormError::Coefficient)
}

/// The space and the shape of the access of an axis reduction: `s: P(stride)`, `k: R(n)`, `o: P(outer)`.
fn axis_space(geom: AxisGeom) -> (SmallVec<[Dim; 4]>, [u32; 4]) {
    let outer = geom.lines / geom.stride.max(1);
    (smallvec![Dim::parallel(u64::from(geom.stride)), Dim::reduce(u64::from(geom.n)), Dim::parallel(u64::from(outer))], [geom.stride, geom.n, outer, 1])
}

// -- windows --

/// The geometry of a windowed launch (what `image::Spec` and `image::geometry` resolve to): the source's shape, and along axes 0, 1,
/// 2 the window extent, the stride, the padding before and the output extent. Axis 2 is part of the window when `three_d`; otherwise
/// axes 2 and 3 are one batch, as is axis 3 of a 3-d window.
#[derive(Clone, Copy, Debug)]
pub(crate) struct WindowGeom {
    pub(crate) src: [u32; 4],
    pub(crate) k: [u32; 3],
    pub(crate) stride: [u32; 3],
    pub(crate) before: [u32; 3],
    pub(crate) out: [u32; 3],
    pub(crate) three_d: bool,
}

/// Where the dims of a window's space are: `o[a]` and `t[a]` along window axis `a`, the batch if it has an extent above 1.
struct Layout {
    space: SmallVec<[Dim; 4]>,
    o: [usize; 3],
    t: [usize; 3],
    batch: Option<usize>,
    axes: usize,
}

impl WindowGeom {
    fn batch(&self) -> u64 {
        if self.three_d { u64::from(self.src[3]) } else { u64::from(self.src[2]) * u64::from(self.src[3]) }
    }

    /// Output dims `o0, o1, (o2)`, the batch if above 1, then the taps `t0, t1, (t2)`.
    fn layout(&self) -> Result<Layout, FormError> {
        let axes = if self.three_d { 3 } else { 2 };
        let mut space: SmallVec<[Dim; 4]> = SmallVec::new();
        let mut o = [usize::MAX; 3];
        let mut t = [usize::MAX; 3];
        for (a, slot) in o.iter_mut().enumerate().take(axes) {
            *slot = space.len();
            space.push(Dim::parallel(u64::from(self.out[a])));
        }
        let batch = (self.batch() > 1).then(|| {
            space.push(Dim::parallel(self.batch()));
            space.len() - 1
        });
        for (a, slot) in t.iter_mut().enumerate().take(axes) {
            *slot = space.len();
            space.push(Dim::reduce(u64::from(self.k[a])));
        }
        if space.len() > MAX_DIMS {
            return Err(FormError::TooMany("dims"));
        }
        Ok(Layout { space, o, t, batch, axes })
    }

    /// The access of the source `x` by the window: `stride * o + t - before` along the window axes, the batch dim along the others.
    fn access(&self, layout: &Layout, x: Src, ty: ElemTy, edge: Edge, fill: Option<Scalar>) -> Result<Access, FormError> {
        let mut map = AffineMap::ZERO;
        for a in 0..layout.axes {
            map.rows[a].coeff[layout.o[a]] = coefficient(self.stride[a] as usize)?;
            map.rows[a].coeff[layout.t[a]] = 1;
            map.rows[a].offset = -i64::from(self.before[a]);
        }
        let shape = if self.three_d {
            if let Some(b) = layout.batch {
                map.rows[3] = AffineRow::dim(b);
            }
            self.src
        } else {
            if let Some(b) = layout.batch {
                map.rows[2] = AffineRow::dim(b);
            }
            [self.src[0], self.src[1], u32::try_from(self.batch()).map_err(|_| FormError::Coefficient)?, 1]
        };
        let mut access = access(x, shape, ty, MapRef::owned(map));
        access.edge = edge;
        access.fill = fill;
        Ok(access)
    }

    /// The access of the output cell `o` (the centre of the window): the window's access with the taps dropped.
    fn centre(&self, layout: &Layout, x: Src, ty: ElemTy) -> Result<Access, FormError> {
        let mut access = self.access(layout, x, ty, Edge::None, None)?;
        let MapRef::Owned(map) = &mut access.map else { unreachable!("a window access owns its map") };
        for a in 0..layout.axes {
            map.rows[a].coeff[layout.t[a]] = 0;
            map.rows[a].offset = 0;
        }
        Ok(access)
    }

    /// The access of a tap array (weights, a mask) of the window's extent: row `a` follows `t[a]`.
    fn taps(&self, layout: &Layout, w: Src, ty: ElemTy) -> Access {
        let mut map = AffineMap::ZERO;
        for a in 0..layout.axes {
            map.rows[a] = AffineRow::dim(layout.t[a]);
        }
        let shape = [self.k[0], self.k[1], if self.three_d { self.k[2] } else { 1 }, 1];
        access(w, shape, ty, MapRef::owned(map))
    }

    fn out_map(&self, layout: &Layout) -> OutMap {
        let dim = |d: usize| Some(d as u8);
        let mut axes = [None; 4];
        axes[0] = dim(layout.o[0]);
        axes[1] = dim(layout.o[1]);
        if self.three_d {
            axes[2] = dim(layout.o[2]);
            axes[3] = layout.batch.and_then(dim);
        } else {
            axes[2] = layout.batch.and_then(dim);
        }
        OutMap::new(axes)
    }
}

/// What a border reads past the array, for a reduction `part` over values of type `ty`: `Zero` reads 0, `Edge` the nearest cell,
/// `Ignore` the identity of the reduction (the cells that are not there do not take part).
fn edge_of(border: Border, part: CombinePart, ty: ElemTy) -> (Edge, Option<Scalar>) {
    match border {
        Border::Zero => (Edge::Fill, Some(Scalar { ty, bits: 0 })),
        Border::Edge => (Edge::Clamp, None),
        Border::Mirror => (Edge::Mirror, None),
        Border::Wrap => (Edge::Wrap, None),
        Border::Ignore => (Edge::Fill, Some(super::combiner::identity(part, ty))),
    }
}

/// A small assembler of bodies: instructions appended in order, values named by the `Ref` they return.
struct Asm {
    instrs: Vec<Instr>,
    scalars: SmallVec<[Scalar; 2]>,
}

impl Asm {
    fn new() -> Asm {
        Asm { instrs: Vec::new(), scalars: SmallVec::new() }
    }

    fn push(&mut self, kind: OpKind, ty: ElemTy, args: [Ref; 3]) -> Ref {
        self.instrs.push(Instr { op: Op::new(kind, ty), args });
        Ref::Tmp(self.instrs.len() as u32 - 1)
    }

    fn scalar(&mut self, value: Scalar) -> Ref {
        self.scalars.push(value);
        Ref::Scalar(self.scalars.len() as u32 - 1)
    }

    fn binary(&mut self, op: BinOp, ty: ElemTy, a: Ref, b: Ref) -> Ref {
        self.push(OpKind::Binary(op), ty, [a, b, Ref::Unused])
    }

    fn unary(&mut self, op: UnOp, ty: ElemTy, a: Ref) -> Ref {
        self.push(OpKind::Unary(op), ty, [a, Ref::Unused, Ref::Unused])
    }

    /// The coordinate of dim `d` of `space` as an `f32`: the index of the point divided and wrapped by the extents of the dims.
    fn coordinate(&mut self, space: &[Dim], d: usize) -> Ref {
        let stride: u64 = space[..d].iter().map(|dim| dim.extent).product();
        let (stride, size) = (stride as u32, space[d].extent as u32);
        let index = self.push(OpKind::Generate(Gen::RangeConst { stride, size, wide: false }), ElemTy::U32, [Ref::Unused; 3]);
        self.push(OpKind::Cast(ElemTy::F32), ElemTy::U32, [index, Ref::Unused, Ref::Unused])
    }

    fn finish(self, outs: SmallVec<[u32; 4]>) -> Body {
        Body { instrs: Arc::new(self.instrs), scalars: self.scalars, outs }
    }
}

/// The values of two parts as the inputs of the epilogue program `f`.
fn program(f: impl FnOnce(&mut Asm) -> Ref) -> Epilogue {
    let mut asm = Asm::new();
    let out = f(&mut asm);
    debug_assert_eq!(out, Ref::Tmp(asm.instrs.len() as u32 - 1), "the program's last instruction is its value");
    Epilogue::Program(Arc::new(asm.instrs))
}

impl Comprehension {
    fn window_form(geom: &WindowGeom, layout: Layout, inputs: SmallVec<[Access; 2]>, body: Body, parts: &[CombinePart], ty: ElemTy, tag: OpTag) -> Comprehension {
        let combiner = Combiner::of(if let [part] = parts { CombineOp::Single(*part) } else { CombineOp::Tuple(parts.iter().copied().collect()) }, ty);
        let out = geom.out_map(&layout);
        comprehension(layout.space, inputs, body, Some(combiner), out, Origin::Api(tag), ty)
    }

    /// `conv2` (`three_d == false`) and `conv3`: correlation of `x` with the kernel `w` under the border of `spec`: `x0 * x1` added.
    pub(crate) fn conv(spec: ConvSpec, geom: &WindowGeom, border: Border, x: Src, w: Src) -> Result<Comprehension, FormError> {
        let ty = ElemTy::F32;
        let layout = geom.layout()?;
        let (edge, fill) = edge_of(border, CombinePart::Add, ty);
        let inputs = smallvec![geom.access(&layout, x, ty, edge, fill)?, geom.taps(&layout, w, ty)];
        Ok(Comprehension::window_form(geom, layout, inputs, product_body(ty), &[CombinePart::Add], ty, OpTag::Conv(Box::new(spec), geom.three_d)))
    }

    /// `conv2_sep`: correlation with the outer product of two 1-d kernels, `x0 * x1 * x2` added; `w0` follows tap dim 0, `w1` tap dim 1.
    pub(crate) fn conv_sep(geom: &WindowGeom, border: Border, x: Src, w0: Src, w1: Src) -> Result<Comprehension, FormError> {
        let ty = ElemTy::F32;
        let layout = geom.layout()?;
        let (edge, fill) = edge_of(border, CombinePart::Add, ty);
        let line = |src: Src, dim: usize, extent: u32| access(src, [extent, 1, 1, 1], ty, MapRef::owned(AffineMap { rows: [AffineRow::dim(dim), AffineRow::ZERO, AffineRow::ZERO, AffineRow::ZERO] }));
        let inputs = smallvec![geom.access(&layout, x, ty, edge, fill)?, line(w0, layout.t[0], geom.k[0]), line(w1, layout.t[1], geom.k[1])];
        let body = Body { instrs: Bodies::x0_times_x1_times_x2(ty), scalars: SmallVec::new(), outs: smallvec![1] };
        Ok(Comprehension::window_form(geom, layout, inputs, body, &[CombinePart::Add], ty, OpTag::ConvSep))
    }

    /// `minfilt`/`maxfilt` (`WindowOp::Min`/`Max`) and `dilate`/`erode` (`MaskedMin`/`MaskedMax`, with the structuring element `mask`
    /// along the taps): the extremum of the window; a masked tap whose mask is 0 reads the identity of the extremum
    /// (`select(m != 0, x0, +-inf)`).
    pub(crate) fn window_extremum(op: WindowOp, geom: &WindowGeom, border: Border, x: Src, mask: Option<Src>, ty: ElemTy) -> Result<Comprehension, FormError> {
        let (part, masked) = match op {
            WindowOp::Min => (CombinePart::Min, false),
            WindowOp::Max => (CombinePart::Max, false),
            WindowOp::MaskedMin => (CombinePart::Min, true),
            WindowOp::MaskedMax => (CombinePart::Max, true),
            _ => return Err(FormError::Window(op)),
        };
        if masked != mask.is_some() {
            return Err(FormError::Window(op));
        }
        let layout = geom.layout()?;
        let (edge, fill) = edge_of(border, part, ty);
        let mut inputs: SmallVec<[Access; 2]> = smallvec![geom.access(&layout, x, ty, edge, fill)?];
        let body = match mask {
            None => Body { instrs: Bodies::x0(ty), scalars: SmallVec::new(), outs: smallvec![0] },
            Some(mask) => {
                inputs.push(geom.taps(&layout, mask, ty));
                let mut asm = Asm::new();
                let zero = asm.scalar(Scalar { ty, bits: 0 });
                let identity = asm.scalar(super::combiner::identity(part, ty));
                let present = asm.push(OpKind::Compare(CmpOp::Ne), ty, [Ref::Input(1), zero, Ref::Unused]);
                asm.push(OpKind::Select, ty, [present, Ref::Input(0), identity]);
                asm.finish(smallvec![1])
            }
        };
        Ok(Comprehension::window_form(geom, layout, inputs, body, &[part], ty, OpTag::Window(op)))
    }

    /// The Sobel filters over a `3 x 3` window of `f32`: `Sobel` is the two derivatives (two comprehensions that share the access, in
    /// the order dimension 0, dimension 1), `SobelMagnitude` and `SobelFast` are one comprehension that adds both derivatives and
    /// combines them in its epilogue (`sqrt(a * a + b * b)`, `|a| + |b|`). The weight of tap `(t0, t1)` for dimension 0 is
    /// `(t0 - 1) * (2 - |t1 - 1|)`, for dimension 1 the same with the taps swapped.
    pub(crate) fn sobel(op: WindowOp, geom: &WindowGeom, border: Border, x: Src) -> Result<SmallVec<[Comprehension; 2]>, FormError> {
        if !matches!(op, WindowOp::Sobel | WindowOp::SobelMagnitude | WindowOp::SobelFast) {
            return Err(FormError::Window(op));
        }
        let weight = |asm: &mut Asm, layout: &Layout, along: usize| {
            let one = asm.scalar(Scalar::f32(1.0));
            let two = asm.scalar(Scalar::f32(2.0));
            let (across, smooth) = if along == 0 { (layout.t[0], layout.t[1]) } else { (layout.t[1], layout.t[0]) };
            let t = asm.coordinate(&layout.space, across);
            let d = asm.binary(BinOp::Sub, ElemTy::F32, t, one);
            let s = asm.coordinate(&layout.space, smooth);
            let s = asm.binary(BinOp::Sub, ElemTy::F32, s, one);
            let s = asm.unary(UnOp::Abs, ElemTy::F32, s);
            let s = asm.binary(BinOp::Sub, ElemTy::F32, two, s);
            asm.binary(BinOp::Mul, ElemTy::F32, d, s)
        };
        let combine = |asm: &mut Asm, along: usize, layout: &Layout| {
            let w = weight(asm, layout, along);
            asm.binary(BinOp::Mul, ElemTy::F32, Ref::Input(0), w)
        };
        derivatives(op, geom, border, x, combine, |asm| match op {
            WindowOp::SobelMagnitude => {
                let (a, b) = (asm.binary(BinOp::Mul, ElemTy::F32, Ref::Input(0), Ref::Input(0)), asm.binary(BinOp::Mul, ElemTy::F32, Ref::Input(1), Ref::Input(1)));
                let sum = asm.binary(BinOp::Add, ElemTy::F32, a, b);
                asm.unary(UnOp::Sqrt, ElemTy::F32, sum)
            }
            _ => {
                let (a, b) = (asm.unary(UnOp::Abs, ElemTy::F32, Ref::Input(0)), asm.unary(UnOp::Abs, ElemTy::F32, Ref::Input(1)));
                asm.binary(BinOp::Add, ElemTy::F32, a, b)
            }
        })
    }

    /// The central-difference gradient (`WindowOp::Gradient`) over a `3 x 3` window of `f32`, clamped: two comprehensions that share
    /// the access, dimension 0 then dimension 1. The weight of tap `(t0, t1)` for dimension 0 is `(t0 - 1) / 2 * (1 - |t1 - 1|)`.
    pub(crate) fn gradient(geom: &WindowGeom, x: Src) -> Result<SmallVec<[Comprehension; 2]>, FormError> {
        let combine = |asm: &mut Asm, along: usize, layout: &Layout| {
            let one = asm.scalar(Scalar::f32(1.0));
            let half = asm.scalar(Scalar::f32(0.5));
            let (across, other) = if along == 0 { (layout.t[0], layout.t[1]) } else { (layout.t[1], layout.t[0]) };
            let t = asm.coordinate(&layout.space, across);
            let d = asm.binary(BinOp::Sub, ElemTy::F32, t, one);
            let d = asm.binary(BinOp::Mul, ElemTy::F32, d, half);
            let s = asm.coordinate(&layout.space, other);
            let s = asm.binary(BinOp::Sub, ElemTy::F32, s, one);
            let s = asm.unary(UnOp::Abs, ElemTy::F32, s);
            let s = asm.binary(BinOp::Sub, ElemTy::F32, one, s);
            let w = asm.binary(BinOp::Mul, ElemTy::F32, d, s);
            asm.binary(BinOp::Mul, ElemTy::F32, Ref::Input(0), w)
        };
        derivatives(WindowOp::Gradient, geom, Border::Edge, x, combine, |_| unreachable!("the gradient has two outputs"))
    }

    /// The bilateral filter (`WindowOp::Bilateral`) over a `k x k` window of `f32`: the centre `x0` and the window `x1`;
    /// `w = exp(c1 * (x0 - x1)^2 + c0 * (dx^2 + dy^2))` (`c0` the spatial and `c1` the range exponent factor, `dx`, `dy` the tap's
    /// offset from the centre), the parts are `(w * x1, w)` added and the epilogue divides the first by the second. (The colour
    /// variant is not a form: its range distance reduces over the channels inside the weight.)
    pub(crate) fn bilateral(geom: &WindowGeom, border: Border, x: Src, c0: f32, c1: f32) -> Result<Comprehension, FormError> {
        let ty = ElemTy::F32;
        if geom.three_d {
            return Err(FormError::Window(WindowOp::Bilateral));
        }
        let layout = geom.layout()?;
        let (edge, fill) = edge_of(border, CombinePart::Add, ty);
        let mut asm = Asm::new();
        let (c0, c1) = (asm.scalar(Scalar::f32(c0)), asm.scalar(Scalar::f32(c1)));
        let offset = |asm: &mut Asm, axis: usize| {
            let centre = asm.scalar(Scalar::f32(geom.before[axis] as f32));
            let t = asm.coordinate(&layout.space, layout.t[axis]);
            let d = asm.binary(BinOp::Sub, ty, t, centre);
            asm.binary(BinOp::Mul, ty, d, d)
        };
        let (dx2, dy2) = (offset(&mut asm, 0), offset(&mut asm, 1));
        let spatial = asm.binary(BinOp::Add, ty, dx2, dy2);
        let spatial = asm.binary(BinOp::Mul, ty, c0, spatial);
        let d = asm.binary(BinOp::Sub, ty, Ref::Input(0), Ref::Input(1));
        let range = asm.binary(BinOp::Mul, ty, d, d);
        let range = asm.binary(BinOp::Mul, ty, c1, range);
        let exponent = asm.binary(BinOp::Add, ty, spatial, range);
        let w = asm.unary(UnOp::Exp, ty, exponent);
        let wx = asm.binary(BinOp::Mul, ty, w, Ref::Input(1));
        let body = asm.finish(smallvec![out(wx), out(w)]);
        let inputs = smallvec![geom.centre(&layout, x.clone(), ty)?, geom.access(&layout, x, ty, edge, fill)?];
        let mut form = Comprehension::window_form(geom, layout, inputs, body, &[CombinePart::Add, CombinePart::Add], ty, OpTag::Window(WindowOp::Bilateral));
        form.epilogue = program(|asm| asm.binary(BinOp::Div, ty, Ref::Input(0), Ref::Input(1)));
        Ok(form)
    }

    /// The form of a reduction with a finished value: `self` with the steps of `finish` as its epilogue (what `fuse_step` makes).
    pub(crate) fn with_steps(mut self, finish: Finish) -> Comprehension {
        self.epilogue = if finish.epilogue.is_empty() { Epilogue::None } else { Epilogue::Steps(finish) };
        self
    }
}

/// The index of the instruction a value is.
fn out(value: Ref) -> u32 {
    match value {
        Ref::Tmp(k) => k,
        other => unreachable!("an out is an instruction, not {other:?}"),
    }
}

/// The comprehension of each of the two derivatives of a `3 x 3` window (`one`), or (for the magnitudes) the one comprehension that
/// has both as parts and `finish` as its epilogue. `term` builds the value one tap adds to derivative `along`.
fn derivatives(
    op: WindowOp,
    geom: &WindowGeom,
    border: Border,
    x: Src,
    term: impl Fn(&mut Asm, usize, &Layout) -> Ref,
    finish: impl FnOnce(&mut Asm) -> Ref,
) -> Result<SmallVec<[Comprehension; 2]>, FormError> {
    let ty = ElemTy::F32;
    if geom.three_d || geom.k[..2] != [3, 3] {
        return Err(FormError::Window(op));
    }
    let (edge, fill) = edge_of(border, CombinePart::Add, ty);
    let two_outputs = matches!(op, WindowOp::Sobel | WindowOp::Gradient);
    if two_outputs {
        let mut forms = SmallVec::new();
        for along in 0..2 {
            let layout = geom.layout()?;
            let mut asm = Asm::new();
            let value = term(&mut asm, along, &layout);
            let body = asm.finish(smallvec![out(value)]);
            let inputs = smallvec![geom.access(&layout, x.clone(), ty, edge, fill)?];
            forms.push(Comprehension::window_form(geom, layout, inputs, body, &[CombinePart::Add], ty, OpTag::Window(op)));
        }
        return Ok(forms);
    }
    let layout = geom.layout()?;
    let mut asm = Asm::new();
    let (a, b) = (term(&mut asm, 0, &layout), term(&mut asm, 1, &layout));
    let body = asm.finish(smallvec![out(a), out(b)]);
    let inputs = smallvec![geom.access(&layout, x, ty, edge, fill)?];
    let mut form = Comprehension::window_form(geom, layout, inputs, body, &[CombinePart::Add, CombinePart::Add], ty, OpTag::Window(op));
    form.epilogue = program(finish);
    Ok(smallvec![form])
}
