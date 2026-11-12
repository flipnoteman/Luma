//! Region writes ([`crate::Array::assign`]) and scatters ([`crate::Array::put`]): one fused
//! kernel computes the new values and stores them straight into the buffer they replace.
//!
//! The expression being stored may read the buffer it writes (the ghost cells of a boundary
//! condition are computed from interior cells of the same array). Every pending operation that
//! reads the buffer was recorded against its old contents and must keep seeing them, so before
//! the write each of them is either
//!
//! * evaluated now (into a buffer of its own, ahead of the write in the batch), or
//! * proven not to be affected: it reads cells the write does not touch, or the element the same
//!   invocation of the store kernel overwrites, after it was read ([`Hazards::keeps`]).
//!
//! A kept operation still fuses into the store kernel when the right-hand side uses it, which is
//! what makes `w.slice(0, ..2).assign(w.slice(0, 2..4).flip(0) * -1.0)` one kernel.
//!
//! This module is for targets that have a buffer. When the target is a pending expression and the
//! view steps by one cell, `Array::assign_lazy` replaces the expression instead and nothing is
//! written or launched (see the module documentation of [`crate::jit`]).

use std::collections::HashMap;
use std::panic::Location;
use std::sync::Arc;

use crate::array::{Failure, Storage};
use crate::builtin::UnOp;
use crate::context::Context;
use crate::error::LumaError;

use super::program::{flatten, Tail};
use super::{launch, lock, Args, Edge, ElemTy, Node, Op, OpKind, ShiftKind, Src, ViewMap};

/// One region of an array that `assign` overwrites.
pub(crate) struct Region {
    /// The array written into: one buffer, not an alias.
    pub(crate) base: Arc<Storage>,
    /// Element `o` of the right-hand side goes to the element of `base` this map reads for `o`.
    /// Injective.
    pub(crate) map: ViewMap,
    /// The new values: `map.out_len()` elements.
    pub(crate) rhs: Src,
    /// Element type of `base`.
    pub(crate) ty: ElemTy,
    /// The pending view node of the array `assign` was called on. It is not evaluated because of
    /// the write: read afterwards it shows the assigned data.
    pub(crate) receiver: Option<Arc<Node>>,
}

fn is_identity(map: &ViewMap) -> bool {
    *map == ViewMap::identity(map.src)
}

/// Whether two boxes of inclusive coordinate ranges share no cell.
fn disjoint(a: &[(u32, u32); 4], b: &[(u32, u32); 4]) -> bool {
    a.iter().zip(b).any(|(a, b)| a.1 < b.0 || b.1 < a.0)
}

/// A copy of the buffer `src` (`len` elements of `ty`), made by its own kernel ahead of the write.
fn copy_of(
    ctx: &Context,
    base: &Arc<Storage>,
    len: usize,
    ty: ElemTy,
    name: &'static str,
    location: &'static Location<'static>,
) -> Result<Src, Failure> {
    let args = [Src::Buffer(Arc::clone(base)), Src::unused(), Src::unused()];
    let node = super::create(ctx, len, Op::new(OpKind::Unary(UnOp::Id), ty), args, name, location)?;
    match node.force() {
        Ok(storage) => Ok(Src::Buffer(Arc::clone(storage))),
        Err(error) => Err(Failure::Inherited(Arc::clone(error))),
    }
}

/// How a pending node reads a buffer through its operand slots.
struct Reads {
    /// Operand 0 is the buffer.
    first: bool,
    /// Another operand is.
    others: bool,
}

/// The operand slots of `node` that hold `base` (as a buffer, or as a node evaluated to it);
/// `None` if the node has no operands (it is being evaluated).
fn reads(node: &Node, base: &Arc<Storage>) -> Option<Reads> {
    let guard = lock(&node.args);
    let args = guard.as_ref()?;
    let holds = |src: &Src| match src {
        Src::Buffer(storage) => Arc::ptr_eq(storage, base),
        Src::Node(operand) => operand.evaluated().is_some_and(|storage| Arc::ptr_eq(storage, base)),
        Src::Scalar(_) => false,
    };
    let arity = node.op().kind.arity();
    Some(Reads { first: arity > 0 && holds(&args[0]), others: args.iter().take(arity).skip(1).any(holds) })
}

/// A pending node below the stores.
struct Entry {
    node: Arc<Node>,
    /// Operand slots of walked nodes (and stores) holding the node.
    slots: usize,
    /// Bit `j` is set if the node is below store `j`.
    owners: u64,
    /// Read through operand 0 of a shift, at an element other than the one being computed.
    under_shift: bool,
    /// Nothing outside the walked nodes holds the node, directly or through its consumers.
    private: bool,
}

/// The pending nodes below the stores through every operand slot (operand 0 of shifts included).
struct Walk {
    entries: Vec<Entry>,
    index: HashMap<*const Node, usize>,
}

impl Walk {
    fn new(stores: &[Arc<Node>]) -> Walk {
        let mut entries: Vec<Entry> = Vec::new();
        let mut index = HashMap::new();
        let mut edges: Vec<(usize, usize, bool)> = Vec::new();
        let roots = stores.len();
        for (j, store) in stores.iter().enumerate() {
            let owners = if j < 64 { 1u64 << j } else { u64::MAX };
            entries.push(Entry { node: Arc::clone(store), slots: 0, owners, under_shift: false, private: true });
        }
        let mut next = 0;
        while next < entries.len() {
            let parent = next;
            next += 1;
            let node = Arc::clone(&entries[parent].node);
            let shift = matches!(node.op().kind, OpKind::Shift(_));
            let arity = node.op().kind.arity();
            let guard = lock(&node.args);
            let Some(args) = guard.as_ref() else { continue };
            for (i, arg) in args.iter().enumerate().take(arity) {
                let Src::Node(child) = arg else { continue };
                if !child.is_pending() {
                    continue;
                }
                let ptr = Arc::as_ptr(child);
                let at = match index.get(&ptr) {
                    Some(&at) => at,
                    None => {
                        entries.push(Entry { node: Arc::clone(child), slots: 0, owners: 0, under_shift: false, private: false });
                        index.insert(ptr, entries.len() - 1);
                        entries.len() - 1
                    }
                };
                entries[at].slots += 1;
                edges.push((parent, at, shift && i == 0));
            }
        }
        // The temporary clone above is gone: a node nobody else holds is referenced by the entry
        // and by the operand slots that were counted.
        for entry in entries.iter_mut().skip(roots) {
            entry.private = Arc::strong_count(&entry.node) == entry.slots + 1;
        }
        loop {
            let mut changed = false;
            for &(parent, child, shifted) in &edges {
                let (owners, under, private) = {
                    let p = &entries[parent];
                    (p.owners, p.under_shift, p.private)
                };
                let c = &mut entries[child];
                if c.owners | owners != c.owners {
                    c.owners |= owners;
                    changed = true;
                }
                if (under || shifted) && !c.under_shift {
                    c.under_shift = true;
                    changed = true;
                }
                if !private && c.private {
                    c.private = false;
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
        for (j, entry) in entries.iter_mut().enumerate().take(roots) {
            index.insert(Arc::as_ptr(&entry.node), j);
        }
        Walk { entries, index }
    }

    fn get(&self, node: &Node) -> Option<&Entry> {
        self.index.get(&(node as *const Node)).map(|&at| &self.entries[at])
    }
}

/// Everything the decision of what to evaluate before a write needs.
struct Hazards<'a> {
    regions: &'a [Region],
    stores: &'a [Arc<Node>],
    walk: &'a Walk,
    /// Every region has the same map, so all stores run in one kernel.
    same: bool,
}

impl Hazards<'_> {
    /// Evaluates the pending readers of every base that the write could change the value of (see
    /// the module documentation). With `conservative`, also the ones that are only kept because
    /// they would be fused into a kernel with all the stores (they are evaluated before a store
    /// kernel for each region instead).
    fn flush(&self, conservative: bool) {
        for (j, region) in self.regions.iter().enumerate() {
            let write = region.map.src_box(Edge::None);
            region.base.readers.flush_where(|reader| self.keeps(reader, j, &write, conservative));
        }
        // The receivers show the assigned data afterwards; what was recorded on top of them
        // before keeps the old values.
        for receiver in self.regions.iter().filter_map(|region| region.receiver.as_ref()) {
            if self.walk.get(receiver).is_none() {
                receiver.readers.flush();
            }
        }
    }

    /// Whether the pending reader `m` of the base of region `j` may stay pending while region `j`
    /// (the cells `write`) is overwritten.
    fn keeps(&self, m: &Node, j: usize, write: &[(u32, u32); 4], conservative: bool) -> bool {
        let ptr = m as *const Node;
        let region = &self.regions[j];
        // The stores themselves.
        if self.stores.iter().any(|store| Arc::as_ptr(store) == ptr) {
            return true;
        }
        let walked = self.walk.get(m);
        // A receiver the right-hand side does not read.
        if walked.is_none() && self.regions.iter().any(|r| r.receiver.as_ref().is_some_and(|rcv| Arc::as_ptr(rcv) == ptr)) {
            return true;
        }
        let Some(reads) = reads(m, &region.base) else { return false };
        // A view that never reads a written cell.
        if let OpKind::Shift(ShiftKind::View { map, edge }) = m.op().kind
            && reads.first
            && !reads.others
            && map.src == region.map.src
            && disjoint(&map.src_box(edge), write)
        {
            return true;
        }
        // An operation private to the right-hand side that reads the element its own invocation
        // overwrites, before writing it.
        let direct = !(matches!(m.op().kind, OpKind::Shift(_)) && reads.first);
        !conservative
            && direct
            && is_identity(&region.map)
            && m.len() == region.base.len()
            && walked.is_some_and(|entry| {
                entry.private && !entry.under_shift && (self.same || entry.owners == 1u64.checked_shl(j as u32).unwrap_or(0))
            })
    }

    /// Flattens and launches the stores: one kernel for all regions of one map, else one per
    /// region.
    fn launch(&self, ctx: &Context) -> Result<(), Failure> {
        let count = self.regions.len();
        let mut split = !self.same;
        let (mut conservative, mut forced) = (false, false);
        loop {
            let groups: Vec<Vec<usize>> =
                if split { (0..count).map(|j| vec![j]).collect() } else { vec![(0..count).collect()] };
            let mut programs = Vec::with_capacity(groups.len());
            let mut fits = true;
            for group in &groups {
                let mut roots: Vec<(Arc<Node>, Args)> = Vec::with_capacity(group.len());
                for &j in group {
                    let Some(args) = self.stores[j].operands() else {
                        return Err(LumaError::Gpu("assignment evaluated twice".into()).into());
                    };
                    roots.push((Arc::clone(&self.stores[j]), args));
                }
                let first = &self.regions[group[0]];
                let fused = flatten(&roots, Tail::StoreMapped(first.map))?;
                let bases: Vec<Arc<Storage>> = group.iter().map(|&j| Arc::clone(&self.regions[j].base)).collect();
                if launch::mapped_bindings(&fused.inputs, &bases) > ctx.jit().max_inputs() + 1 {
                    fits = false;
                    break;
                }
                programs.push((fused, bases, first.map.out_len()));
            }
            if fits {
                for (fused, bases, n) in programs {
                    launch::run_mapped(ctx, fused, &bases, n)?;
                }
                return Ok(());
            }
            drop(programs);
            if !conservative {
                // The kernels of different regions no longer run together: nothing may be left
                // to read a base after another store's kernel overwrote it.
                conservative = true;
                split = true;
                self.flush(true);
            } else if !forced {
                forced = true;
                for store in self.stores {
                    evaluate_rhs(store)?;
                }
            } else {
                return Err(LumaError::Chunked("an assignment that reads this many buffers at once").into());
            }
        }
    }
}

/// Evaluates the expression `store` stores, so that it reads one buffer.
fn evaluate_rhs(store: &Arc<Node>) -> Result<(), Failure> {
    let operand = match lock(&store.args).as_ref().map(|args| args[0].clone()) {
        Some(Src::Node(node)) => node,
        _ => return Ok(()),
    };
    match operand.force() {
        Ok(_) => Ok(()),
        Err(error) => Err(Failure::Inherited(Arc::clone(error))),
    }
}

/// Releases a store node that ran: its operands go, and it counts as evaluated to the buffer it wrote.
fn retire(store: &Arc<Node>, base: &Arc<Storage>) {
    lock(&store.args).take();
    let _ = store.result.set(Ok(Arc::clone(base)));
}

/// Writes every region: `regions[j].rhs` into `regions[j].base` through `regions[j].map`, in one
/// kernel when the maps agree. Pending operations that read a base keep the values they were
/// recorded against (see the module documentation). The caller has dropped its own handles on
/// the right-hand sides, which is what lets them fuse into the kernel.
pub(crate) fn assign(
    ctx: &Context,
    mut regions: Vec<Region>,
    name: &'static str,
    location: &'static Location<'static>,
) -> Result<(), Failure> {
    if regions.is_empty() {
        return Ok(());
    }
    for (j, region) in regions.iter().enumerate() {
        if region.base.segments() != 1 {
            return Err(LumaError::Chunked("assigning into an array split over several buffers").into());
        }
        if region.base.is_alias() {
            return Err(LumaError::ReadOnly("the state passed to an iteration step").into());
        }
        if !region.map.is_injective() {
            return Err(LumaError::InvalidShape("assign needs a slice, flip or transpose view".into()).into());
        }
        if regions[..j].iter().any(|other| Arc::ptr_eq(&other.base, &region.base)) {
            return Err(LumaError::InvalidShape("assign of overlapping records".into()).into());
        }
    }
    let same = regions.iter().all(|region| region.map == regions[0].map);

    // A store that reads a base through a map that moves elements would race with the writes
    // (and one that reads a base another kernel writes would see new data): copy it first.
    for j in 0..regions.len() {
        let Src::Buffer(read) = &regions[j].rhs else { continue };
        let conflict = regions.iter().any(|q| Arc::ptr_eq(&q.base, read) && (!is_identity(&q.map) || !same));
        if conflict {
            let read = Arc::clone(read);
            let (len, ty) = (regions[j].map.out_len(), regions[j].ty);
            regions[j].rhs = copy_of(ctx, &read, len, ty, name, location)?;
        }
    }

    let mut stores: Vec<Arc<Node>> = Vec::with_capacity(regions.len());
    for region in &mut regions {
        let rhs = std::mem::replace(&mut region.rhs, Src::unused());
        let args = [rhs, Src::unused(), Src::unused()];
        let store = super::create(ctx, region.map.out_len(), Op::new(OpKind::Unary(UnOp::Id), region.ty), args, name, location)?;
        // Only this function may evaluate a store (into its base): marked as fused, it is never
        // picked up as a live array by the multi-output evaluations that the flushes below run.
        store.fused.store(true, std::sync::atomic::Ordering::Relaxed);
        stores.push(store);
    }

    ctx.jit().note_write();
    let result = {
        let walk = Walk::new(&stores);
        let hazards = Hazards { regions: &regions, stores: &stores, walk: &walk, same };
        hazards.flush(false);
        hazards.launch(ctx)
    };
    for (store, region) in stores.iter().zip(&regions) {
        retire(store, &region.base);
    }
    result?;
    // The written bases are the results of this operation.
    for region in &regions {
        super::check_finite(ctx, &region.base, region.ty)?;
    }
    Ok(())
}

/// `base[indices[i]] = values[i]` for `count` elements, skipping indices outside `base`, in one
/// kernel. Duplicate indices: one of them wins.
#[allow(clippy::too_many_arguments)]
pub(crate) fn scatter(
    ctx: &Context,
    base: &Arc<Storage>,
    indices: Src,
    values: Src,
    ty: ElemTy,
    count: usize,
    name: &'static str,
    location: &'static Location<'static>,
) -> Result<(), Failure> {
    if base.segments() != 1 {
        return Err(LumaError::Chunked("assigning into an array split over several buffers").into());
    }
    if base.is_alias() {
        return Err(LumaError::ReadOnly("the state passed to an iteration step").into());
    }
    // Whatever was recorded against the old contents runs before the write.
    base.readers.flush();
    ctx.jit().note_write();
    // A store reading the buffer it scatters into would race with it.
    let own = |src: Src, ty: ElemTy| match src {
        Src::Buffer(read) if Arc::ptr_eq(&read, base) => copy_of(ctx, &read, count, ty, name, location),
        other => Ok(other),
    };
    let values = own(values, ty)?;
    let indices = own(indices, ElemTy::U32)?;
    let unused = || Src::unused();
    let value = super::create(ctx, count, Op::new(OpKind::Unary(UnOp::Id), ty), [values, unused(), unused()], name, location)?;
    let index = super::create(ctx, count, Op::new(OpKind::Unary(UnOp::Id), ElemTy::U32), [indices, unused(), unused()], name, location)?;
    let stores = [value, index];
    // As in `assign`: only this function evaluates its stores.
    for store in &stores {
        store.fused.store(true, std::sync::atomic::Ordering::Relaxed);
    }
    let tail = Tail::Scatter { len: u32::try_from(base.len()).map_err(|_| LumaError::DimensionTooLarge { shape: [base.len(), 1, 1, 1] })? };
    let bases = [Arc::clone(base)];

    let run = || -> Result<(), Failure> {
        let mut forced = false;
        loop {
            let mut roots: Vec<(Arc<Node>, Args)> = Vec::with_capacity(2);
            for store in &stores {
                let Some(args) = store.operands() else {
                    return Err(LumaError::Gpu("scatter evaluated twice".into()).into());
                };
                roots.push((Arc::clone(store), args));
            }
            let fused = flatten(&roots, tail)?;
            if launch::mapped_bindings(&fused.inputs, &bases) <= ctx.jit().max_inputs() + 1 {
                return launch::run_mapped(ctx, fused, &bases, count);
            }
            if forced {
                return Err(LumaError::Chunked("a scatter that reads this many buffers at once").into());
            }
            forced = true;
            for store in &stores {
                evaluate_rhs(store)?;
            }
        }
    };
    let result = run();
    for store in &stores {
        retire(store, base);
    }
    result?;
    super::check_finite(ctx, base, ty)?;
    Ok(())
}
