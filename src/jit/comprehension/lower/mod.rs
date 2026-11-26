//! The generic lowering of a form (W2-R0 §5.3-5.6): what runs a `Recognised` or `Rewritten` form that no recogniser specialised.
//!
//! * **G-B** ([`reduce`], this file): the existing axis-reduce tail over a *transient program of the tile*. For every tile of the
//!   virtual space each access is one view node over its source (the `ViewMap` of the tile), the body's instructions are
//!   elementwise nodes over those views, and the reduce dim is one axis of the node, so [`Array::run_axis_reduce`] runs it exactly
//!   as it runs a hand-written `(bcast(a) * bcast(b)).sum_axis(k)`. Rows that read two dims (`o + t`, a window) cannot be views:
//!   when their reduce dims have at most [`UNROLL_MAX`] taps those dims are unrolled, one program per tap, combined in tap order.
//! * **G-A** ([`contract`]): the fused matmul tail for any separable body (`Mac::Body`).
//! * **Tiling**: the space is `u64`; every node of a tile has at most [`tile_budget`] points, so every index in a kernel fits `u32`.
//!   The tiles that cover a slab of the output are partial results combined in tile order.
//!
//! The transient nodes are real nodes (`jit::create`), made when the strategy runs and dropped before it returns: they register as
//! readers on their sources only while the one synchronous lowering lives, and the accounting of `pending_nodes` is symmetric.
//! (The phantom nodes of W2-R0 5.7 were not needed: the only forms that reach this code are made by recognisers and rules, never
//! by an API call, so the extra bookkeeping per node is paid by the fallback only.)

pub(crate) mod contract;
#[cfg(test)]
mod tests;

use std::panic::Location;
use std::sync::Arc;

use smallvec::SmallVec;

use super::map::{AffineMap, AffineRow};
use super::{Access, CombineOp, CombinePart, Compensation, Comprehension, DimKind, Epilogue, MAX_DIMS, Origin};
use crate::array::{Array, Failure, Storage};
use crate::builtin::{BinOp, ReduceList, ReduceOp};
use crate::context::Context;
use crate::error::LumaError;
use crate::gpu_type::Df32;
use crate::jit::program::Ref;
use crate::jit::structured::emit::Emit;
use crate::jit::{self, ElemTy, Node, Op, OpKind, ShiftKind, Src};
use crate::shape::Shape;

/// Taps a window row may be unrolled over (G-B2).
pub(crate) const UNROLL_MAX: u64 = 64;
/// Launches one lowering may make (tiles times partial results); beyond it the form has no lowering.
pub(crate) const MAX_TILES: u64 = 4096;

/// Points one node of a tile may have: every index in the kernel fits `u32`. `LUMA_TILE_BUDGET` lowers it (a test hook, so that the
/// tiler and the chunked reduction run at test size).
pub(crate) fn tile_budget() -> u64 {
    #[cfg(test)]
    if let Some(budget) = TEST_BUDGET.with(std::cell::Cell::get) {
        return budget;
    }
    static BUDGET: std::sync::LazyLock<u64> = std::sync::LazyLock::new(|| {
        let cap = u64::from(u32::MAX);
        std::env::var("LUMA_TILE_BUDGET").ok().and_then(|text| text.trim().parse::<u64>().ok()).map_or(cap, |budget| budget.clamp(1, cap))
    });
    *BUDGET
}

#[cfg(test)]
thread_local! {
    /// The budget of the tests of this thread.
    pub(crate) static TEST_BUDGET: std::cell::Cell<Option<u64>> = const { std::cell::Cell::new(None) };
}

/// The shape class of a form, for the cost of selection and the telemetry: the first four extents, the greatest first.
pub(crate) fn dims(form: &Comprehension) -> [u32; 4] {
    let mut extents: SmallVec<[u64; 6]> = form.space.iter().map(|dim| dim.extent).collect();
    extents.sort_unstable_by(|a, b| b.cmp(a));
    let mut out = [1u32; 4];
    for (slot, extent) in out.iter_mut().zip(extents) {
        *slot = u32::try_from(extent).unwrap_or(u32::MAX);
    }
    out
}

/// The part of a combiner as the binary op that combines two partial values, and the reduction that folds a line of them.
pub(super) fn combine_of(part: CombinePart) -> Option<(BinOp, ReduceOp)> {
    match part {
        CombinePart::Add => Some((BinOp::Add, ReduceOp::Sum)),
        CombinePart::Mul => Some((BinOp::Mul, ReduceOp::Prod)),
        CombinePart::Min => Some((BinOp::Min, ReduceOp::Min)),
        CombinePart::Max => Some((BinOp::Max, ReduceOp::Max)),
        CombinePart::Or | CombinePart::And | CombinePart::ArgMax | CombinePart::ArgMin => None,
    }
}

/// How the dims of a (coalesced) form sit in the nodes of a tile.
#[derive(Clone, Debug)]
pub(crate) struct Prepared {
    /// The form with its dims coalesced.
    pub(crate) form: Comprehension,
    /// Reduce dims unrolled over their taps (G-B2): the dims of rows that read two dims.
    taps: SmallVec<[usize; 2]>,
    /// The dims on the axes of a node, fastest first: the parallel dims in the order of the output, then the reduce dim.
    axes: SmallVec<[usize; 4]>,
    /// Whether the last axis is the reduce dim.
    reduces: bool,
    /// Parallel dims that are not on an axis: a host loop of extent 1 per tile, slowest last.
    outer: SmallVec<[usize; 3]>,
    /// Reduce dims that are not on the axis: a host loop whose partial results combine.
    folded: SmallVec<[usize; 2]>,
    part: Option<CombinePart>,
}

impl Prepared {
    /// The number of taps unrolled into one program.
    pub(crate) fn taps(&self) -> u64 {
        self.taps.iter().map(|&d| self.form.space[d].extent).product()
    }

    /// Whether some row of some access reads two dims (G-B2 applies).
    pub(crate) fn unrolls(&self) -> bool {
        !self.taps.is_empty()
    }

    /// The tiles the lowering launches (not counting the taps, which share a program).
    pub(crate) fn tiles(&self) -> u64 {
        let plan = self.tiling();
        plan.slabs.saturating_mul(plan.chunks).saturating_mul(plan.loops)
    }
}

/// How the space is cut: slabs of the output, chunks of the reduce axis, loops over the folded dims.
struct Tiling {
    /// Extent per axis in a tile.
    extent: SmallVec<[u64; 4]>,
    /// Slabs of the parallel axes (the output is `slabs` contiguous pieces).
    slabs: u64,
    /// Chunks of the reduce axis.
    chunks: u64,
    /// Product of the extents of the folded dims.
    loops: u64,
}

impl Prepared {
    /// Cuts the node's axes so that no tile has more than [`tile_budget`] points. The reduce axis is kept whole where it can be (a
    /// chunk of it is a partial result to combine); the parallel axes are cut from the slowest, so that a tile writes a contiguous
    /// slab of the output.
    fn tiling(&self) -> Tiling {
        let budget = tile_budget();
        let extents: SmallVec<[u64; 4]> = self.axes.iter().map(|&d| self.form.space[d].extent).collect();
        let mut extent = extents.clone();
        let mut remaining = budget;
        let loops: u64 = self.folded.iter().map(|&d| self.form.space[d].extent).product();
        // The reduce axis first (kept whole while it fits), then the parallel axes fastest to slowest.
        let mut order: SmallVec<[usize; 4]> = SmallVec::new();
        if self.reduces {
            order.push(extents.len() - 1);
        }
        order.extend(0..extents.len() - usize::from(self.reduces));
        let mut cut = false;
        for at in order {
            let whole = extents[at];
            if cut {
                extent[at] = 1;
            } else if whole <= remaining {
                remaining /= whole.max(1);
            } else {
                extent[at] = remaining.max(1);
                remaining = 1;
                cut = true;
            }
        }
        let chunks = if self.reduces { extents[extents.len() - 1].div_ceil(extent[extents.len() - 1]) } else { 1 };
        let parallel = extents.len() - usize::from(self.reduces);
        let slabs: u64 = (0..parallel).map(|a| extents[a].div_ceil(extent[a])).product::<u64>() * self.outer.iter().map(|&d| self.form.space[d].extent).product::<u64>();
        Tiling { extent, slabs, chunks, loops }
    }
}

/// Merges dims that follow each other in every access (and, for parallel dims, in the output): the coordinate `c_slow * s +
/// c_fast * f` of every row is `c_fast * (f + e_fast * s)` for the merged dim. Maps keep their rows, so every edge is exact.
pub(super) fn coalesce(form: &Comprehension) -> Comprehension {
    let mut form = form.clone();
    loop {
        let live: SmallVec<[usize; 6]> = (0..form.space.len()).filter(|&d| form.space[d].extent > 1).collect();
        let mut merged = None;
        'pairs: for &slow in &live {
            for &fast in &live {
                if slow == fast || form.space[slow].is_reduce() != form.space[fast].is_reduce() || matches!(form.space[slow].kind, DimKind::Segmented { .. }) {
                    continue;
                }
                let extent = form.space[fast].extent;
                if form.space[slow].extent.saturating_mul(extent) > u64::from(u32::MAX) {
                    continue;
                }
                let composes = form.inputs.iter().all(|a| {
                    a.map.rows.iter().all(|row| {
                        let (cs, cf) = (i64::from(row.coeff[slow]), i64::from(row.coeff[fast]));
                        cs == cf * extent as i64
                    })
                });
                // A parallel pair must be neighbours in the output order, the faster one first.
                let adjacent = form.space[fast].is_reduce() || {
                    let axes: SmallVec<[usize; 4]> = form.output.dim_of_axis.iter().flatten().map(|&d| usize::from(d)).filter(|&d| form.space[d].extent > 1).collect();
                    axes.iter().position(|&d| d == fast).zip(axes.iter().position(|&d| d == slow)).is_some_and(|(f, s)| s == f + 1)
                };
                if composes && adjacent {
                    merged = Some((slow, fast));
                    break 'pairs;
                }
            }
        }
        let Some((slow, fast)) = merged else { return form };
        // The merged dim is `fast`; `slow` has extent 1 and no coefficient left.
        form.space[fast].extent *= form.space[slow].extent;
        form.space[slow].extent = 1;
        for access in &mut form.inputs {
            let mut map = *access.map;
            for row in &mut map.rows {
                row.coeff[slow] = 0;
            }
            access.map = super::MapRef::owned(map);
        }
        for axis in &mut form.output.dim_of_axis {
            if *axis == Some(slow as u8) {
                *axis = None;
            }
        }
    }
}

/// Plans the lowering of `form`, or `None` when it has none: `lowerable` is exactly this.
pub(crate) fn prepare(form: &Comprehension) -> Option<Prepared> {
    if form.ty != ElemTy::F32 || !matches!(form.epilogue, Epilogue::None) || form.output.order != super::OutMap::IDENTITY || form.validate().is_err() {
        return None;
    }
    if form.space.iter().any(|dim| matches!(dim.kind, DimKind::Segmented { .. })) || form.inputs.iter().any(|a| a.indirect.is_some() || matches!(a.src, Src::Scalar(_))) {
        return None;
    }
    let part = match form.combiner.as_ref().map(|combiner| &combiner.op) {
        None => None,
        Some(CombineOp::Single(part)) if combine_of(*part).is_some() => Some(*part),
        Some(_) => return None,
    };
    if form.body.outs.len() != 1 || form.inputs.iter().any(|a| a.ty == ElemTy::F64 || a.ty.layout() != ElemTy::F32.layout() && !a.ty.is_packed()) {
        return None;
    }
    let compensated = form.combiner.as_ref().is_some_and(|combiner| combiner.compensation != Compensation::None);
    let form = coalesce(form);
    let live: SmallVec<[usize; 6]> = (0..form.space.len()).filter(|&d| form.space[d].extent > 1).collect();
    // The reduce dims of rows that read two dims are unrolled; every other row must read at most one dim of the node's axes.
    let multi = |row: &AffineRow, among: &[usize]| among.iter().filter(|&&d| row.coeff[d] != 0).count() >= 2;
    let mut taps: SmallVec<[usize; 2]> = SmallVec::new();
    for &d in live.iter().filter(|&&d| form.space[d].is_reduce()) {
        if form.inputs.iter().any(|a| a.map.rows.iter().any(|row| multi(row, &live) && row.coeff[d] != 0)) {
            taps.push(d);
        }
    }
    if taps.iter().map(|&d| form.space[d].extent).product::<u64>() > UNROLL_MAX {
        return None;
    }
    let rest: SmallVec<[usize; 6]> = live.iter().copied().filter(|d| !taps.contains(d)).collect();
    // A row that still reads two dims once the taps are fixed is no view (an anti-diagonal of two parallel dims).
    if form.inputs.iter().any(|a| a.map.rows.iter().any(|row| multi(row, &rest))) {
        return None;
    }
    let parallel: SmallVec<[usize; 4]> = form.output.dim_of_axis.iter().flatten().map(|&d| usize::from(d)).filter(|d| rest.contains(d)).collect();
    let mut reduce: SmallVec<[usize; 2]> = rest.iter().copied().filter(|&d| form.space[d].is_reduce()).collect();
    // The greatest reduce dim is the axis; the others are looped and their partial results combined.
    reduce.sort_by_key(|&d| std::cmp::Reverse(form.space[d].extent));
    let reduces = !reduce.is_empty();
    let inline_parallel = parallel.len().min(if reduces { 3 } else { 4 });
    let mut axes: SmallVec<[usize; 4]> = parallel[..inline_parallel].iter().copied().collect();
    let outer: SmallVec<[usize; 3]> = parallel[inline_parallel..].iter().copied().collect();
    let folded: SmallVec<[usize; 2]> = reduce.iter().skip(1).copied().collect();
    if let Some(&axis) = reduce.first() {
        axes.push(axis);
    }
    // Compensated sums are whole reductions through the double-float library.
    if compensated && (!parallel.is_empty() || !folded.is_empty() || !taps.is_empty() || part != Some(CombinePart::Add)) {
        return None;
    }
    let plan = Prepared { form, taps, axes, reduces, outer, folded, part };
    (plan.tiles() <= if compensated { 1 } else { MAX_TILES }).then_some(plan)
}

/// Whether the generic strategies can run `form` (`build` refuses one that cannot).
pub(crate) fn lowerable(form: &Comprehension) -> bool {
    prepare(form).is_some()
}

/// The values the dims of a tile start at.
#[derive(Clone, Copy)]
pub(super) struct Origin6(pub(super) [i64; MAX_DIMS]);

/// The node of access `j` of `form` over a tile: a view of the source whose axis `o` follows dim `axes[o]` for `extent[o]` points,
/// the dims off the axes held at their value in `origin` (so are the dims on them, plus their coordinate in the tile). The access's
/// edge and fill are the view's. The rows must read at most one dim of the axes each.
pub(super) fn view_node(ctx: &Context, form: &Comprehension, j: usize, axes_of: &[usize], origin: &Origin6, extent: &[u64], location: &'static Location<'static>) -> Result<Arc<Node>, Failure> {
    let access: &Access = &form.inputs[j];
    let mut map = AffineMap::ZERO;
    for (row, source) in map.rows.iter_mut().zip(&access.map.rows) {
        let mut offset = source.offset;
        for (d, &c) in source.coeff.iter().enumerate().take(form.space.len()) {
            offset += i64::from(c) * origin.0[d];
            if axes_of.contains(&d) {
                row.coeff[d] = c;
            }
        }
        row.offset = offset;
    }
    let mut axes = [None; 4];
    let mut out = [1u32; 4];
    for (o, &d) in axes_of.iter().enumerate() {
        axes[o] = Some(d as u8);
        out[o] = u32::try_from(extent[o]).map_err(|_| Failure::from(LumaError::InvalidShape("a tile of a form has more points than the budget".into())))?;
    }
    let view = map.to_view(axes, access.shape, out).ok_or_else(|| Failure::from(LumaError::Unsupported("a form's access is not a view of its source".into())))?;
    let len: usize = out.iter().map(|&e| e as usize).product();
    let fill = access.fill.map_or_else(Src::unused, Src::Scalar);
    let op = Op::new(OpKind::Shift(ShiftKind::View { map: view, edge: access.edge }), access.ty);
    jit::create(ctx, len, op, [access.src.clone(), fill, Src::unused()], "comprehension", location)
}

impl Prepared {
    /// The node of access `j` over the tile at `origin` with the extents `extent` per axis: the view of the source.
    fn view(&self, ctx: &Context, j: usize, origin: &Origin6, extent: &[u64], location: &'static Location<'static>) -> Result<Arc<Node>, Failure> {
        view_node(ctx, &self.form, j, &self.axes, origin, extent, location)
    }

    /// The body of the form over the tile as elementwise nodes: the node of the out.
    fn body(&self, ctx: &Context, origin: &Origin6, extent: &[u64], location: &'static Location<'static>) -> Result<Arc<Node>, Failure> {
        let (_, temps) = self.nodes(ctx, origin, extent, location)?;
        Ok(Arc::clone(&temps[self.form.body.outs[0] as usize]))
    }

    /// The views of the accesses and the nodes of the instructions of the body over the tile.
    fn nodes(&self, ctx: &Context, origin: &Origin6, extent: &[u64], location: &'static Location<'static>) -> Result<(Vec<Arc<Node>>, Vec<Arc<Node>>), Failure> {
        let views = (0..self.form.inputs.len()).map(|j| self.view(ctx, j, origin, extent, location)).collect::<Result<Vec<_>, _>>()?;
        let len: usize = extent.iter().map(|&e| e as usize).product();
        let mut temps: Vec<Arc<Node>> = Vec::with_capacity(self.form.body.instrs.len());
        for instr in self.form.body.instrs.iter() {
            let arg = |r: Ref| match r {
                Ref::Input(j) => Src::Node(Arc::clone(&views[j as usize])),
                Ref::Scalar(k) => Src::Scalar(self.form.body.scalars[k as usize]),
                Ref::Tmp(t) => Src::Node(Arc::clone(&temps[t as usize])),
                Ref::Unused | Ref::Sub(_) => Src::unused(),
            };
            let args = [arg(instr.args[0]), arg(instr.args[1]), arg(instr.args[2])];
            temps.push(jit::create(ctx, len, instr.op, args, "comprehension", location)?);
        }
        Ok((views, temps))
    }

    /// The program of one tile with the taps fixed to every combination, combined in tap order.
    fn tile_node(&self, ctx: &Context, origin: &Origin6, extent: &[u64], location: &'static Location<'static>) -> Result<Arc<Node>, Failure> {
        let total = self.taps();
        let mut combined: Option<Arc<Node>> = None;
        for tap in 0..total.max(1) {
            let mut at = *origin;
            let mut rest = tap;
            for &d in &self.taps {
                let e = self.form.space[d].extent;
                at.0[d] = (rest % e) as i64;
                rest /= e;
            }
            let node = self.body(ctx, &at, extent, location)?;
            combined = Some(match (combined, self.part.and_then(combine_of)) {
                (Some(acc), Some((op, _))) => {
                    let args = [Src::Node(acc), Src::Node(node), Src::unused()];
                    jit::create(ctx, extent.iter().map(|&e| e as usize).product(), Op::new(OpKind::Binary(op), ElemTy::F32), args, "comprehension", location)?
                }
                (_, _) => node,
            });
        }
        Ok(combined.expect("a form has a tap"))
    }

    /// The result of one tile: the reduction of its node along the reduce axis (or the node itself when nothing reduces), one value
    /// per output point of the slab, forced into a storage.
    fn tile(&self, ctx: &Context, origin: &Origin6, extent: &[u64], location: &'static Location<'static>) -> Result<Arc<Storage>, Failure> {
        let root = self.tile_node(ctx, origin, extent, location)?;
        let mut dims = [1usize; 4];
        for (slot, &e) in dims.iter_mut().zip(extent) {
            *slot = e as usize;
        }
        let array = Array::<f32>::from_node(ctx, Shape::from(dims), root);
        if !self.reduces {
            return Ok(Arc::clone(array.storage()?));
        }
        let (_, op) = self.part.and_then(combine_of).expect("a reducing form has a combiner");
        let outs = array.run_axis_reduce(extent.len() - 1, ReduceList::new(&[op]))?;
        let out = outs.into_iter().next().expect("one op, one result");
        Ok(Arc::clone(out.storage()?))
    }

    /// Runs the lowering: every slab of the output is the combination, in order, of the partial results of its chunks and loops.
    pub(crate) fn run(&self, emit: &mut Emit<'_>, location: &'static Location<'static>) -> Result<Arc<Storage>, Failure> {
        let ctx = emit.ctx();
        if self.form.combiner.as_ref().is_some_and(|combiner| combiner.compensation != Compensation::None) {
            return self.run_compensated(ctx, location);
        }
        let plan = self.tiling();
        let len: u64 = self.form.output.dim_of_axis.iter().flatten().map(|&d| self.form.space[usize::from(d)].extent).product();
        if plan.slabs * plan.chunks * plan.loops == 1 {
            return self.tile(ctx, &Origin6([0; MAX_DIMS]), &plan.extent, location);
        }
        let out = Arc::new(Storage::new(ctx, len as usize, ElemTy::F32.layout()));
        let out_pin = out.pin_single()?;
        let parallel = plan.extent.len() - usize::from(self.reduces);
        // The output axes are the parallel axes then the outer dims, fastest first: slab `s` is `slab` points of the output at `s * slab`.
        let slab: u64 = plan.extent[..parallel].iter().product();
        let mut slab_at = 0u64;
        let mut origin = Origin6([0; MAX_DIMS]);
        // Walk the slabs: the parallel axes (cut) then the outer dims, as an odometer over their grids.
        let grids: SmallVec<[(usize, u64, u64); 6]> = (0..parallel)
            .map(|a| (self.axes[a], plan.extent[a], self.form.space[self.axes[a]].extent.div_ceil(plan.extent[a])))
            .chain(self.outer.iter().map(|&d| (d, 1, self.form.space[d].extent)))
            .collect();
        let slabs: u64 = grids.iter().map(|g| g.2).product();
        for s in 0..slabs {
            let mut rest = s;
            for &(d, step, count) in &grids {
                origin.0[d] = ((rest % count) * step) as i64;
                rest /= count;
            }
            // A slab at the edge of a cut axis is shorter.
            let extent: SmallVec<[u64; 4]> = plan
                .extent
                .iter()
                .enumerate()
                .map(|(a, &e)| if a < parallel { e.min(self.form.space[self.axes[a]].extent - origin.0[self.axes[a]] as u64) } else { e })
                .collect();
            let points: u64 = extent[..parallel].iter().product();
            let partial = self.slab(ctx, &mut origin, &extent, &plan, location)?;
            let pin = partial.pin_single()?;
            emit.copy(pin.buffer(), 0, out_pin.buffer(), slab_at * 4, points * 4)?;
            slab_at += points;
            debug_assert!(points <= slab);
        }
        debug_assert_eq!(slab_at, len);
        Ok(out)
    }

    /// One slab: the combination, in chunk order, of the partial results of its chunks of the reduce axis and its loops.
    fn slab(&self, ctx: &Context, origin: &mut Origin6, extent: &[u64], plan: &Tiling, location: &'static Location<'static>) -> Result<Arc<Storage>, Failure> {
        if !self.reduces || plan.chunks * plan.loops == 1 {
            return self.tile(ctx, origin, extent, location);
        }
        let axis = extent.len() - 1;
        let reduce_dim = self.axes[axis];
        let whole = self.form.space[reduce_dim].extent;
        let combine = self.part.and_then(combine_of).map(|(op, _)| op).expect("a reducing form has a combiner");
        let slab_len: usize = extent[..axis].iter().map(|&e| e as usize).product();
        let mut acc: Option<Arc<Node>> = None;
        let mut loop_at = 0u64;
        while loop_at < plan.loops {
            let mut rest = loop_at;
            for &d in &self.folded {
                let e = self.form.space[d].extent;
                origin.0[d] = (rest % e) as i64;
                rest /= e;
            }
            for chunk in 0..plan.chunks {
                origin.0[reduce_dim] = (chunk * plan.extent[axis]) as i64;
                let mut piece: SmallVec<[u64; 4]> = extent.iter().copied().collect();
                piece[axis] = plan.extent[axis].min(whole - chunk * plan.extent[axis]);
                let storage = self.tile(ctx, origin, &piece, location)?;
                let node = jit::create(ctx, slab_len, Op::new(OpKind::Unary(crate::builtin::UnOp::Id), ElemTy::F32), [Src::Buffer(storage), Src::unused(), Src::unused()], "comprehension", location)?;
                acc = Some(match acc {
                    None => node,
                    Some(previous) => jit::create(ctx, slab_len, Op::new(OpKind::Binary(combine), ElemTy::F32), [Src::Node(previous), Src::Node(node), Src::unused()], "comprehension", location)?,
                });
            }
            loop_at += 1;
        }
        let node = acc.expect("a chunk");
        match node.force() {
            Ok(storage) => Ok(Arc::clone(storage)),
            Err(error) => Err(Failure::Inherited(Arc::clone(error))),
        }
    }

    /// A compensated whole reduction: the values as `Df32` (`Dot2`: both factors cast and multiplied with `df_mul`, so the product is
    /// exact), summed by the double-float library (`df_add` is a two-sum), rounded back (W2-R0 5.6). About 44 bits at the cost of a
    /// double-float operation per element.
    fn run_compensated(&self, ctx: &Context, location: &'static Location<'static>) -> Result<Arc<Storage>, Failure> {
        let plan = self.tiling();
        let origin = Origin6([0; MAX_DIMS]);
        let (views, temps) = self.nodes(ctx, &origin, &plan.extent, location)?;
        let len: usize = plan.extent.iter().map(|&e| e as usize).product();
        let node_of = |r: Ref| match r {
            Ref::Input(j) => Some(Src::Node(Arc::clone(&views[j as usize]))),
            Ref::Tmp(t) => Some(Src::Node(Arc::clone(&temps[t as usize]))),
            _ => None,
        };
        let to_df32 = |src: Src, from: ElemTy| jit::create(ctx, len, Op::new(OpKind::Cast(ElemTy::Df32), from), [src, Src::unused(), Src::unused()], "comprehension", location);
        let out = &self.form.body.instrs[self.form.body.outs[0] as usize];
        let values = match self.form.combiner.as_ref().map(|combiner| combiner.compensation) {
            Some(Compensation::Dot2) => {
                let (a, b) = match (out.op.kind, node_of(out.args[0]), node_of(out.args[1])) {
                    (OpKind::Binary(BinOp::Mul), Some(a), Some(b)) => (a, b),
                    _ => return Err(Failure::from(LumaError::Unsupported("a Dot2 form's body is a product of two loads".into()))),
                };
                let (a, b) = (to_df32(a, ElemTy::F32)?, to_df32(b, ElemTy::F32)?);
                jit::create(ctx, len, Op::new(OpKind::Binary(BinOp::Mul), ElemTy::Df32), [Src::Node(a), Src::Node(b), Src::unused()], "comprehension", location)?
            }
            _ => to_df32(Src::Node(Arc::clone(&temps[self.form.body.outs[0] as usize])), ElemTy::F32)?,
        };
        let array = Array::<Df32>::from_node(ctx, Shape::from([len, 1, 1, 1]), values);
        let sum = array.sum();
        let rounded = jit::create(ctx, 1, Op::new(OpKind::Cast(ElemTy::F32), ElemTy::Df32), [sum.src()?, Src::unused(), Src::unused()], "comprehension", location)?;
        match rounded.force() {
            Ok(storage) => Ok(Arc::clone(storage)),
            Err(error) => Err(Failure::Inherited(Arc::clone(error))),
        }
    }
}

/// Builds the array of a form: the result of the generic lowering, or an array that failed with `InvalidShape` for a form that has
/// none. Test and recogniser entry point; the strategies of the registry call [`Prepared::run`] themselves.
#[track_caller]
pub(crate) fn build(ctx: &Context, form: &Comprehension) -> Array<f32> {
    let location = Location::caller();
    let shape = Shape::from(output_dims(form));
    let result = match &form.origin {
        Origin::Api(_) => Err(Failure::from(LumaError::Unsupported("an API form runs through its own strategy".into()))),
        _ if !lowerable(form) && !contract::prepare(form).is_some() => Err(Failure::from(LumaError::InvalidShape("the form has no lowering".into()))),
        _ => crate::jit::structured::select::execute(ctx, &crate::jit::structured::strategy::Subject::Form(form, location)).map(|output| Array::from_storage(ctx, shape, output.into_storage())),
    };
    Array::resolve(ctx, shape, result, "comprehension", location)
}

/// The extents of the output axes of a form.
pub(crate) fn output_dims(form: &Comprehension) -> [usize; 4] {
    let mut dims = [1usize; 4];
    for (slot, axis) in dims.iter_mut().zip(form.output.dim_of_axis) {
        if let Some(d) = axis {
            *slot = form.space[usize::from(d)].extent as usize;
        }
    }
    dims
}
