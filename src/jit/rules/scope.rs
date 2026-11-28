//! The planning scope of a forced structured value (06 §3.6): the comprehensions the normaliser may rewrite together, and what
//! connects them.
//!
//! [`Scope::of`] starts from the value being forced (the *root*) and follows each access that reads a pending elementwise
//! expression down to the structured values the expression waits for (the walk of `multireduce::unresolved_leaves`, bounded,
//! same thread, refusing looped nodes and nodes another thread evaluates). A leaf whose value is an unforced value of the same
//! thread and not made inside a recording becomes a *producer*; values that could be recorded together with the root (R0's
//! `partner_of`) are *partners*. The scope is a clone: rewriting never touches a [`Structured`].

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::sync::atomic::Ordering;

use smallvec::SmallVec;

use super::t2::Rewrite;
use crate::context::Context;
use crate::jit::comprehension::lower::{lowers, output_dims};
use crate::jit::comprehension::{Comprehension, DimKind, Origin};
use crate::jit::structured::Structured;
use crate::jit::{Node, Scalar, Src};

/// Most comprehensions a scope may have; over it the engine does nothing.
pub(crate) const MAX_SCOPE_COMPS: usize = 32;
/// Most nodes (comprehensions and distinct elementwise nodes) a scope may have.
pub(crate) const MAX_SCOPE_NODES: u32 = 256;
/// Most nodes one walk looks at (the budget of `multireduce`).
const WALK_BUDGET: usize = 4096;

/// A comprehension of a scope, by creation sequence: the order of the sites, so the result of the engine is a function of the
/// scope alone.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub(crate) struct CompId(pub(crate) u32);

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum ScopeRole {
    /// The value being forced.
    Root,
    /// A value the root (or another producer) reads through a leaf.
    Producer,
    /// A value that could be recorded together with the root.
    Partner,
    /// Made by a rewrite; private to the scope.
    Added,
}

#[derive(Clone)]
pub(crate) struct SComp {
    pub(crate) id: CompId,
    pub(crate) form: Comprehension,
    /// The value the comprehension is the form of; `None` for the root (the caller holds it) and for an added comprehension.
    pub(crate) value: Option<Arc<Structured>>,
    pub(crate) role: ScopeRole,
}

/// The value of comprehension `producer` is what access `access` of `consumer` reads.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) struct Edge {
    pub(crate) consumer: CompId,
    pub(crate) access: u8,
    pub(crate) producer: CompId,
    /// The access reads the leaf of the producer itself, with no elementwise expression between.
    pub(crate) direct: bool,
    /// Made by a rewrite: the access reads the output of the planned step of `producer`, not the leaf of its value.
    pub(crate) planned: bool,
}

/// Where a rule applies.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum SiteKind {
    Comp,
    Edge,
    Pair,
}

/// One place a rule may apply; also what the memo records, so that a replay finds the same place in a scope of the same
/// structure.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum Site {
    Comp(CompId),
    Edge { consumer: CompId, access: u8, producer: CompId },
    Pair(CompId, CompId),
}

/// What identifies a place.
pub(crate) type SiteSig = Site;

/// Why no scope was made.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum NoScope {
    /// The value cannot be planned: no form, another thread, a recording, a node that cannot be walked.
    Unavailable,
    /// Over [`MAX_SCOPE_COMPS`] or [`MAX_SCOPE_NODES`].
    TooLarge,
}

#[derive(Clone)]
pub(crate) struct Scope {
    pub(crate) comps: SmallVec<[SComp; 8]>,
    pub(crate) edges: SmallVec<[Edge; 4]>,
    /// Distinct elementwise nodes below the accesses: the prologues, summarised by their weights in the measure.
    elementwise: u32,
    next: u32,
}

impl Scope {
    /// A scope of these forms, the first the root.
    pub(crate) fn from_forms(forms: impl IntoIterator<Item = Comprehension>, elementwise: u32) -> Scope {
        let mut scope = Scope { comps: SmallVec::new(), edges: SmallVec::new(), elementwise, next: 0 };
        for form in forms {
            let role = if scope.comps.is_empty() { ScopeRole::Root } else { ScopeRole::Partner };
            scope.push(form, None, role);
        }
        scope
    }

    fn push(&mut self, form: Comprehension, value: Option<Arc<Structured>>, role: ScopeRole) -> CompId {
        let id = CompId(self.next);
        self.next += 1;
        self.comps.push(SComp { id, form, value, role });
        id
    }

    /// Adds an edge (tests and rules that build a scope by hand).
    pub(crate) fn link(&mut self, consumer: CompId, access: u8, producer: CompId) {
        self.edges.push(Edge { consumer, access, producer, direct: true, planned: false });
    }

    /// The id the next comprehension added gets.
    pub(crate) fn next_id(&self) -> CompId {
        CompId(self.next)
    }

    pub(crate) fn comp(&self, id: CompId) -> Option<&SComp> {
        self.comps.iter().find(|comp| comp.id == id)
    }

    pub(crate) fn root(&self) -> &SComp {
        self.comps.iter().find(|comp| comp.role == ScopeRole::Root).expect("a scope has a root")
    }

    /// Comprehensions plus distinct elementwise nodes.
    pub(crate) fn nodes(&self) -> u32 {
        self.comps.len() as u32 + self.elementwise
    }

    /// The scope of the value `root`, which is being forced.
    pub(crate) fn of(ctx: &Context, root: &Structured) -> Result<Scope, NoScope> {
        #[cfg(any(test, feature = "rule-trace"))]
        super::registry::trace::scope_built();
        if root.in_recording || ctx.recording_here() || root.thread != std::thread::current().id() {
            return Err(NoScope::Unavailable);
        }
        let form = root.form(ctx).ok_or(NoScope::Unavailable)?;
        let mut scope = Scope { comps: SmallVec::new(), edges: SmallVec::new(), elementwise: 0, next: 0 };
        let root_id = scope.push(form, None, ScopeRole::Root);
        let mut ids: HashMap<*const Structured, CompId> = HashMap::new();
        ids.insert(root, root_id);
        let mut counted: HashSet<*const Node> = HashSet::new();
        let mut at = 0;
        while at < scope.comps.len() {
            let consumer = scope.comps[at].id;
            let reads: SmallVec<[(u8, Arc<Node>); 4]> = scope.comps[at]
                .form
                .inputs
                .iter()
                .enumerate()
                .filter_map(|(j, access)| match &access.src {
                    Src::Node(node) => Some((j as u8, Arc::clone(node))),
                    _ => None,
                })
                .collect();
            for (access, node) in reads {
                let direct = node.deferred_value().is_some();
                let leaves = walk(&node, &mut counted, &mut scope.elementwise)?;
                if scope.nodes() > MAX_SCOPE_NODES {
                    return Err(NoScope::TooLarge);
                }
                for leaf in leaves {
                    let Some(value) = leaf.deferred_value() else { continue };
                    let key = Arc::as_ptr(value);
                    let producer = match ids.get(&key) {
                        Some(&id) => id,
                        None => {
                            if !plannable(value) {
                                continue;
                            }
                            let Some(form) = value.form(ctx) else { continue };
                            if scope.comps.len() >= MAX_SCOPE_COMPS {
                                return Err(NoScope::TooLarge);
                            }
                            let id = scope.push(form, Some(Arc::clone(value)), ScopeRole::Producer);
                            ids.insert(key, id);
                            id
                        }
                    };
                    scope.edges.push(Edge { consumer, access, producer, direct, planned: false });
                }
            }
            at += 1;
        }
        // The values that could be recorded together with the root, whose forms the planned forms are compared with.
        if root.mergeable(ctx) || root.axis_mergeable(ctx) {
            for partner in ctx.jit().scope_partners(root) {
                if ids.contains_key(&Arc::as_ptr(&partner)) {
                    continue;
                }
                if scope.comps.len() >= MAX_SCOPE_COMPS {
                    return Err(NoScope::TooLarge);
                }
                if let Some(form) = partner.form(ctx) {
                    let id = scope.push(form, Some(Arc::clone(&partner)), ScopeRole::Partner);
                    ids.insert(Arc::as_ptr(&partner), id);
                }
            }
        }
        if scope.nodes() > MAX_SCOPE_NODES {
            return Err(NoScope::TooLarge);
        }
        Ok(scope)
    }

    /// The places of one kind, in ascending order of their comprehensions' ids (and of the access).
    pub(crate) fn sites(&self, kind: SiteKind) -> SmallVec<[Site; 8]> {
        match kind {
            SiteKind::Comp => self.comps.iter().map(|comp| Site::Comp(comp.id)).collect(),
            SiteKind::Edge => {
                let mut edges: SmallVec<[Edge; 4]> = self.edges.clone();
                edges.sort_by_key(|edge| (edge.consumer, edge.access, edge.producer));
                edges.iter().map(|edge| Site::Edge { consumer: edge.consumer, access: edge.access, producer: edge.producer }).collect()
            }
            SiteKind::Pair => {
                let mut pairs = SmallVec::new();
                for (i, a) in self.comps.iter().enumerate() {
                    for b in &self.comps[i + 1..] {
                        pairs.push(Site::Pair(a.id, b.id));
                    }
                }
                pairs
            }
        }
    }

    /// The scope after `rewrite`, unchecked (see [`Scope::checked`]).
    pub(crate) fn applied(&self, rewrite: &Rewrite) -> Scope {
        let mut next = self.clone();
        for (id, form) in &rewrite.replace {
            if let Some(comp) = next.comps.iter_mut().find(|comp| comp.id == *id) {
                comp.form = form.clone();
            }
        }
        for form in &rewrite.add {
            next.push(form.clone(), None, ScopeRole::Added);
        }
        for id in &rewrite.drop {
            next.comps.retain(|comp| comp.id != *id);
            next.edges.retain(|edge| edge.consumer != *id && edge.producer != *id);
        }
        for feed in &rewrite.feeds {
            next.edges.retain(|edge| !(edge.consumer == feed.consumer && edge.access == feed.access));
            next.edges.push(Edge { consumer: feed.consumer, access: feed.access, producer: feed.producer, direct: true, planned: true });
        }
        next
    }

    /// [`Scope::applied`], or why the rewrite is not valid: the root survives, every comprehension that changed is well formed
    /// and lowerable, a replaced one keeps its type and output shape, and every feed names comprehensions of the scope.
    pub(crate) fn checked(&self, rewrite: &Rewrite) -> Result<Scope, &'static str> {
        if rewrite.drop.iter().any(|id| self.comp(*id).is_some_and(|comp| comp.role == ScopeRole::Root)) {
            return Err("invalid");
        }
        let next = self.applied(rewrite);
        for (id, form) in &rewrite.replace {
            let before = self.comp(*id).ok_or("invalid")?;
            if form.ty != before.form.ty || output_dims(form) != output_dims(&before.form) {
                return Err("invalid");
            }
        }
        let changed = rewrite.replace.iter().map(|(_, form)| form).chain(&rewrite.add);
        for form in changed {
            if form.validate().is_err() {
                return Err("invalid");
            }
            if !lowers(form) {
                return Err("not lowerable");
            }
        }
        for feed in &rewrite.feeds {
            let consumer = next.comp(feed.consumer).ok_or("invalid")?;
            let producer = next.comp(feed.producer).ok_or("invalid")?;
            // A producer that runs as a step of the plan has a form of its own, not the tag of an API operation.
            if usize::from(feed.access) >= consumer.form.inputs.len() || matches!(producer.form.origin, Origin::Api(_)) {
                return Err("invalid");
            }
        }
        if !next.acyclic() {
            return Err("invalid");
        }
        #[cfg(debug_assertions)]
        self.check_rewrite(rewrite);
        Ok(next)
    }

    /// Whether following the edges from consumer to producer never comes back.
    fn acyclic(&self) -> bool {
        // Repeatedly remove comprehensions no remaining edge reads; a cycle is what is left.
        let mut live: SmallVec<[CompId; 8]> = self.comps.iter().map(|comp| comp.id).collect();
        loop {
            let before = live.len();
            let snapshot = live.clone();
            live.retain(|id| self.edges.iter().any(|edge| edge.producer == *id && snapshot.contains(&edge.consumer)));
            if live.len() == before {
                return live.is_empty();
            }
        }
    }

    /// Debug assertion: a rewrite reads no source the scope did not read, except through a feed.
    #[cfg(debug_assertions)]
    fn check_rewrite(&self, rewrite: &Rewrite) {
        let known = |src: &Src| self.comps.iter().any(|comp| sources(&comp.form).any(|have| same_src(have, src)));
        for (id, form) in &rewrite.replace {
            for src in sources(form) {
                let fed = rewrite.feeds.iter().any(|feed| feed.consumer == *id);
                debug_assert!(known(src) || fed, "a rewrite of {id:?} reads a source the scope does not");
            }
        }
        for (i, form) in rewrite.add.iter().enumerate() {
            // An added comprehension has the next free id; one that a feed names reads its placeholder from the producer.
            let id = CompId(self.next + i as u32);
            let fed = rewrite.feeds.iter().any(|feed| feed.consumer == id);
            for src in sources(form) {
                debug_assert!(known(src) || fed, "an added comprehension reads a source the scope does not");
            }
        }
    }

    /// What identifies the scope's structure (the key of the memo): the comprehensions' structures, roles and links, not the
    /// identity of their sources.
    pub(crate) fn structure_hash(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut h = crate::gpu::hash::FxHasher::default();
        for comp in &self.comps {
            comp.id.hash(&mut h);
            comp.role.hash(&mut h);
            comp.form.structure_hash().hash(&mut h);
        }
        for edge in &self.edges {
            edge.hash(&mut h);
        }
        self.elementwise.hash(&mut h);
        h.finish()
    }
}

/// Whether the value of a leaf may be a comprehension of the scope: unforced, of this thread, not made in a recording or being
/// recorded, and not failed.
pub(super) fn plannable(value: &Structured) -> bool {
    value.is_unforced()
        && value.thread == std::thread::current().id()
        && !value.in_recording
        && !value.forcing.load(Ordering::Relaxed)
        && value.failure().is_none()
}

/// The structured values the expression behind `root` waits for, and the elementwise nodes it has (counted once over the
/// whole scope in `elementwise`). `Unavailable` where `unresolved_leaves` has none: a loop's output, a node another thread
/// evaluates, an expression over the budget.
fn walk(root: &Arc<Node>, counted: &mut HashSet<*const Node>, elementwise: &mut u32) -> Result<Vec<Arc<Node>>, NoScope> {
    let mut stack = vec![Arc::clone(root)];
    let mut local: HashSet<*const Node> = HashSet::new();
    let mut leaves = Vec::new();
    while let Some(node) = stack.pop() {
        if node.result.get().is_some() {
            continue;
        }
        if node.looped.get().is_some() {
            return Err(NoScope::Unavailable);
        }
        if !local.insert(Arc::as_ptr(&node)) {
            continue;
        }
        if local.len() > WALK_BUDGET {
            return Err(NoScope::Unavailable);
        }
        match node.operands() {
            Some(args) => {
                if counted.insert(Arc::as_ptr(&node)) {
                    *elementwise += 1;
                    if *elementwise > MAX_SCOPE_NODES {
                        return Err(NoScope::TooLarge);
                    }
                }
                // What an earlier walk found free of leaves is not walked again.
                if node.leaf_free.load(Ordering::Relaxed) {
                    continue;
                }
                for arg in args.iter().take(node.op().kind.arity()) {
                    if let Src::Node(operand) = arg {
                        stack.push(Arc::clone(operand));
                    }
                }
            }
            None if node.deferred.get().is_some() => leaves.push(node),
            None => return Err(NoScope::Unavailable),
        }
    }
    Ok(leaves)
}

/// Every source a form reads: accesses, index arrays and segment offsets.
fn sources(form: &Comprehension) -> impl Iterator<Item = &Src> {
    form.inputs
        .iter()
        .flat_map(|access| std::iter::once(&access.src).chain(access.indirect.iter().map(|indirect| &indirect.index)))
        .chain(form.space.iter().filter_map(|dim| match &dim.kind {
            DimKind::Segmented { offsets, .. } => Some(&**offsets),
            _ => None,
        }))
}

fn same_scalar(a: &Scalar, b: &Scalar) -> bool {
    a.ty == b.ty && a.bits == b.bits
}

/// The same node, buffer or constant.
pub(crate) fn same_src(a: &Src, b: &Src) -> bool {
    match (a, b) {
        (Src::Node(a), Src::Node(b)) => Arc::ptr_eq(a, b),
        (Src::Buffer(a), Src::Buffer(b)) => Arc::ptr_eq(a, b),
        (Src::Scalar(a), Src::Scalar(b)) => same_scalar(a, b),
        _ => false,
    }
}
