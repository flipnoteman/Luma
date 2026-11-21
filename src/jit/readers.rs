//! Self-pruning lists of pending nodes: the readers of a buffer, and a context's live nodes.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, Weak};

use super::{lock, Node};

/// Pending lazy nodes that read one buffer (or the result of another pending node), or every
/// live node of a context.
///
/// Whoever is about to overwrite the buffer calls [`Readers::flush`] first: that evaluates those
/// nodes, recording their reads ahead of the write.
#[derive(Debug, Default)]
pub(crate) struct Readers {
    list: Mutex<ReaderList>,
    /// Counts the flushes: it changes whenever the buffer may be about to be overwritten.
    generation: AtomicU64,
}

/// How many references a [`Weaks`] holds in itself.
const INLINE: usize = 3;

/// Weak references to nodes, oldest first. Most nodes are read by a few operations, so the first
/// references live in the list itself: no allocation until a fourth one arrives.
#[derive(Debug, Default)]
pub(super) struct Weaks {
    /// The oldest references, from the front.
    inline: [Option<Weak<Node>>; INLINE],
    /// The others; there are some only if `inline` is full.
    rest: Vec<Weak<Node>>,
}

impl Weaks {
    fn push(&mut self, node: Weak<Node>) {
        match self.inline.iter_mut().find(|slot| slot.is_none()) {
            Some(slot) => *slot = Some(node),
            None => self.rest.push(node),
        }
    }

    fn len(&self) -> usize {
        self.inline.iter().flatten().count() + self.rest.len()
    }

    fn is_empty(&self) -> bool {
        self.inline.first().is_none_or(Option::is_none)
    }

    fn iter(&self) -> impl DoubleEndedIterator<Item = &Weak<Node>> {
        self.inline.iter().flatten().chain(&self.rest)
    }

    fn retain(&mut self, mut keep: impl FnMut(&Weak<Node>) -> bool) {
        // In order, so that what is kept stays oldest first.
        let mut kept = 0;
        for at in 0..INLINE {
            let Some(weak) = self.inline.get_mut(at).and_then(Option::take) else { continue };
            if keep(&weak) {
                if let Some(slot) = self.inline.get_mut(kept) {
                    *slot = Some(weak);
                }
                kept += 1;
            }
        }
        self.rest.retain_mut(|weak| keep(weak));
        // The oldest of the others take the places that were freed.
        let moved = (INLINE - kept).min(self.rest.len());
        for (slot, weak) in self.inline.iter_mut().skip(kept).zip(self.rest.drain(..moved)) {
            *slot = Some(weak);
        }
    }
}

impl IntoIterator for Weaks {
    type Item = Weak<Node>;
    type IntoIter = std::iter::Chain<std::iter::Flatten<std::array::IntoIter<Option<Weak<Node>>, INLINE>>, std::vec::IntoIter<Weak<Node>>>;

    fn into_iter(self) -> Self::IntoIter {
        self.inline.into_iter().flatten().chain(self.rest)
    }
}

impl Extend<Weak<Node>> for Weaks {
    fn extend<I: IntoIterator<Item = Weak<Node>>>(&mut self, nodes: I) {
        nodes.into_iter().for_each(|node| self.push(node));
    }
}

#[derive(Debug, Default)]
pub(super) struct ReaderList {
    nodes: Weaks,
    /// Length at which dead and evaluated entries are dropped next.
    prune_at: usize,
}

impl ReaderList {
    pub(super) fn add_weak(&mut self, weak: Weak<Node>) {
        self.nodes.push(weak);
        if self.nodes.len() > self.prune_at.max(PRUNE_MIN) {
            self.nodes.retain(|weak| weak.upgrade().is_some_and(|node| node.is_pending()));
            self.prune_at = self.nodes.len().saturating_mul(2);
        }
    }

    /// Takes the live entries `take` accepts, oldest first; the others stay registered. Dead
    /// entries are dropped.
    pub(super) fn take_if(&mut self, take: impl Fn(&Node) -> bool) -> Vec<Weak<Node>> {
        self.prune_at = 0;
        let (mut taken, mut kept) = (Vec::new(), Weaks::default());
        for weak in std::mem::take(&mut self.nodes) {
            let Some(node) = weak.upgrade() else { continue };
            if take(&node) {
                taken.push(weak);
            } else {
                kept.push(weak);
            }
        }
        self.nodes = kept;
        taken
    }

    /// Calls `f` with every registered node, newest first, without copying the list. `f` must
    /// not register nodes.
    pub(super) fn for_each_newest(&self, f: impl FnMut(&Weak<Node>)) {
        self.nodes.iter().rev().for_each(f);
    }
}

/// Smallest list length worth pruning.
const PRUNE_MIN: usize = 32;

impl Readers {
    pub(crate) fn add(&self, node: &Arc<Node>) {
        self.add_weak(Arc::downgrade(node));
    }

    /// [`Readers::add`] for a caller that made the weak reference already (making one is an
    /// atomic compare-and-swap; cloning one is cheaper).
    pub(super) fn add_weak(&self, weak: Weak<Node>) {
        lock(&self.list).add_weak(weak);
    }

    pub(super) fn take(&self) -> Weaks {
        let mut list = lock(&self.list);
        list.prune_at = 0;
        std::mem::take(&mut list.nodes)
    }

    /// Takes over the readers of `other`: a node that was evaluated passes its readers on to the
    /// buffer it produced.
    pub(crate) fn absorb(&self, other: &Readers) {
        let moved = other.take();
        if moved.is_empty() {
            return;
        }
        lock(&self.list).nodes.extend(moved);
    }

    /// Whether a reader that `consider` accepts is still pending. Only accepted readers are
    /// upgraded.
    pub(super) fn any_pending(&self, mut consider: impl FnMut(&Weak<Node>) -> bool) -> bool {
        lock(&self.list)
            .nodes
            .iter()
            .any(|weak| consider(weak) && weak.upgrade().is_some_and(|reader| reader.is_pending()))
    }

    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        lock(&self.list).nodes.len()
    }

    pub(crate) fn generation(&self) -> u64 {
        self.generation.load(Ordering::Relaxed)
    }

    /// Evaluates every pending node registered here.
    pub(crate) fn flush(&self) {
        self.flush_where(|_| false);
    }

    /// Evaluates every pending node registered here that `keep` rejects. The kept ones stay
    /// registered (and pending).
    pub(crate) fn flush_where(&self, keep: impl Fn(&Node) -> bool) {
        self.generation.fetch_add(1, Ordering::Relaxed);
        let mut to_force = Vec::new();
        {
            let mut list = lock(&self.list);
            list.nodes.retain(|weak| {
                let Some(node) = weak.upgrade() else { return false };
                if !node.is_pending() {
                    return false;
                }
                if keep(&node) {
                    true
                } else {
                    to_force.push(node);
                    false
                }
            });
            list.prune_at = list.nodes.len().saturating_mul(2);
        }
        for node in to_force {
            // A failure is stored in the node and reported when its array is used.
            let _ = node.force();
        }
    }
}
