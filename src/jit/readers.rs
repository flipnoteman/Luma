//! Self-pruning lists of pending nodes: the readers of a buffer, and a context's live nodes.

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
}

/// Weak references to nodes. Most nodes are read by a single operation, so the first reference
/// lives in the list itself: no allocation until a second one arrives.
#[derive(Debug, Default)]
pub(super) struct Weaks {
    first: Option<Weak<Node>>,
    rest: Vec<Weak<Node>>,
}

impl Weaks {
    fn push(&mut self, node: Weak<Node>) {
        if self.first.is_none() {
            self.first = Some(node);
        } else {
            self.rest.push(node);
        }
    }

    fn len(&self) -> usize {
        usize::from(self.first.is_some()) + self.rest.len()
    }

    fn iter(&self) -> impl DoubleEndedIterator<Item = &Weak<Node>> {
        self.first.iter().chain(&self.rest)
    }

    fn retain(&mut self, mut keep: impl FnMut(&Weak<Node>) -> bool) {
        if self.first.as_ref().is_some_and(|weak| !keep(weak)) {
            self.first = None;
        }
        self.rest.retain(|weak| keep(weak));
        // `first` leads the list: refill it so that a single reader never needs the vector.
        if self.first.is_none() && !self.rest.is_empty() {
            self.first = Some(self.rest.remove(0));
        }
    }
}

impl IntoIterator for Weaks {
    type Item = Weak<Node>;
    type IntoIter = std::iter::Chain<std::option::IntoIter<Weak<Node>>, std::vec::IntoIter<Weak<Node>>>;

    fn into_iter(self) -> Self::IntoIter {
        self.first.into_iter().chain(self.rest)
    }
}

impl Extend<Weak<Node>> for Weaks {
    fn extend<I: IntoIterator<Item = Weak<Node>>>(&mut self, nodes: I) {
        nodes.into_iter().for_each(|node| self.push(node));
    }
}

#[derive(Debug, Default)]
struct ReaderList {
    nodes: Weaks,
    /// Length at which dead and evaluated entries are dropped next.
    prune_at: usize,
}

/// Smallest list length worth pruning.
const PRUNE_MIN: usize = 32;

impl Readers {
    pub(crate) fn add(&self, node: &Arc<Node>) {
        let mut list = lock(&self.list);
        list.nodes.push(Arc::downgrade(node));
        if list.nodes.len() > list.prune_at.max(PRUNE_MIN) {
            list.nodes.retain(|weak| weak.upgrade().is_some_and(|node| node.is_pending()));
            list.prune_at = list.nodes.len().saturating_mul(2);
        }
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
        if moved.first.is_none() {
            return;
        }
        lock(&self.list).nodes.extend(moved);
    }

    pub(super) fn newest_first(&self) -> Vec<Weak<Node>> {
        lock(&self.list).nodes.iter().rev().cloned().collect()
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

    /// Evaluates every pending node registered here.
    pub(crate) fn flush(&self) {
        for weak in self.take() {
            if let Some(node) = weak.upgrade() {
                // A failure is stored in the node and reported when its array is used.
                let _ = node.force();
            }
        }
    }
}
