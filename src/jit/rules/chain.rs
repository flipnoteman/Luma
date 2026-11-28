//! The order of a chain of matrix products (rule 9, `chain_order`): the textbook dynamic program over `m·k·n`, with the cost of a
//! link measured the way the rewrite engine measures a kernel (flops at the device's rate, bytes at its bandwidth, plus a launch), so
//! that a chain of small matrices keeps the order it was written in (a launch costs more than the flops it saves).
//!
//! [`best`] is the function `einsum`'s path search may call for a chain of two-operand products: it needs nothing but the extents.

use crate::jit::cost::CostModel;

/// What a link of a chain costs on a device.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct LinkCosts {
    pub(crate) flops_per_ns: f64,
    pub(crate) bytes_per_ns: f64,
    pub(crate) launch_ns: f64,
}

impl LinkCosts {
    /// The costs of the context's cost model.
    pub(crate) fn of(model: &CostModel) -> LinkCosts {
        LinkCosts { flops_per_ns: model.f32_ops_per_ns.max(1e-9), bytes_per_ns: model.bandwidth_gbs.max(1e-9), launch_ns: model.launch_us * 1000.0 }
    }

    /// The time of `[m, k] · [k, n]`: `2mkn` flops, the three matrices once, one launch.
    pub(crate) fn link(&self, m: u64, k: u64, n: u64) -> f64 {
        let (m, k, n) = (m as f64, k as f64, n as f64);
        2.0 * m * k * n / self.flops_per_ns + 4.0 * (m * k + k * n + m * n) / self.bytes_per_ns + self.launch_ns
    }
}

/// How a chain of matrices is parenthesised.
#[derive(Clone, PartialEq, Eq, Debug)]
pub(crate) enum Tree {
    /// Matrix `i` of the chain.
    Leaf(usize),
    /// The product of two sub-chains, in this order.
    Join(Box<Tree>, Box<Tree>),
}

impl Tree {
    /// The left-to-right association `((a·b)·c)·d` of `n` matrices.
    pub(crate) fn left_to_right(n: usize) -> Tree {
        (1..n).fold(Tree::Leaf(0), |acc, i| Tree::Join(Box::new(acc), Box::new(Tree::Leaf(i))))
    }

    /// The matrices `first..=last` this tree covers.
    fn span(&self) -> (usize, usize) {
        match self {
            Tree::Leaf(i) => (*i, *i),
            Tree::Join(l, r) => (l.span().0, r.span().1),
        }
    }

    /// The time of this order for matrices whose extents are `dims` (matrix `i` is `dims[i] x dims[i + 1]`).
    pub(crate) fn cost(&self, dims: &[u64], costs: &LinkCosts) -> f64 {
        match self {
            Tree::Leaf(_) => 0.0,
            Tree::Join(l, r) => {
                let ((first, mid), last) = ((l.span().0, l.span().1 + 1), r.span().1 + 1);
                l.cost(dims, costs) + r.cost(dims, costs) + costs.link(dims[first], dims[mid], dims[last])
            }
        }
    }
}

/// The cheapest order of the chain whose matrix `i` is `dims[i] x dims[i + 1]` (`dims.len() == matrices + 1`), and its time. Ties
/// keep the latest split, so a chain whose orders cost the same stays as written (left to right). `O(n³)` in the number of matrices.
pub(crate) fn best(dims: &[u64], costs: &LinkCosts) -> (Tree, f64) {
    let n = dims.len().saturating_sub(1);
    assert!(n >= 1, "a chain has a matrix");
    // cost[i][j]: the cheapest order of matrices i..=j; split[i][j]: where it cuts.
    let mut cost = vec![vec![0.0f64; n]; n];
    let mut split = vec![vec![0usize; n]; n];
    for len in 2..=n {
        for i in 0..=n - len {
            let j = i + len - 1;
            let mut cheapest = f64::INFINITY;
            for cut in (i..j).rev() {
                let total = cost[i][cut] + cost[cut + 1][j] + costs.link(dims[i], dims[cut + 1], dims[j + 1]);
                if total < cheapest {
                    cheapest = total;
                    split[i][j] = cut;
                }
            }
            cost[i][j] = cheapest;
        }
    }
    fn build(split: &[Vec<usize>], i: usize, j: usize) -> Tree {
        if i == j {
            return Tree::Leaf(i);
        }
        let cut = split[i][j];
        Tree::Join(Box::new(build(split, i, cut)), Box::new(build(split, cut + 1, j)))
    }
    (build(&split, 0, n - 1), cost[0][n - 1])
}

#[cfg(test)]
mod tests {
    use super::*;

    const COSTS: LinkCosts = LinkCosts { flops_per_ns: 10_000.0, bytes_per_ns: 900.0, launch_ns: 5_000.0 };

    #[test]
    fn a_matrix_times_a_matrix_times_a_vector_multiplies_the_vector_first() {
        // (a·b)·v with a, b of 4096 x 4096 and v of 4096 x 1: a·(b·v) is two matrix-vector products.
        let dims = [4096, 4096, 4096, 1];
        let (tree, ns) = best(&dims, &COSTS);
        assert_eq!(tree, Tree::Join(Box::new(Tree::Leaf(0)), Box::new(Tree::Join(Box::new(Tree::Leaf(1)), Box::new(Tree::Leaf(2))))));
        assert!(ns * 20.0 < Tree::left_to_right(3).cost(&dims, &COSTS), "{ns}");
    }

    #[test]
    fn small_matrices_keep_the_order_they_were_written_in() {
        // At 64 the launch is what costs: both orders cost the same and the one written wins.
        let dims = [64, 64, 64, 64];
        assert_eq!(best(&dims, &COSTS).0, Tree::left_to_right(3));
    }

    #[test]
    fn the_textbook_chain_has_its_known_optimum() {
        // 10x30, 30x5, 5x60: (A·B)·C = 1500 + 3000 multiplications, A·(B·C) = 9000 + 18000.
        let flops_only = LinkCosts { flops_per_ns: 1.0, bytes_per_ns: 1e18, launch_ns: 0.0 };
        let (tree, ns) = best(&[10, 30, 5, 60], &flops_only);
        assert_eq!(tree, Tree::left_to_right(3));
        assert!((ns - 2.0 * 4500.0).abs() < 1e-6, "{ns}");
        // Six matrices with a known optimum of the CLRS example: 15125 multiplications.
        let (_, ns) = best(&[30, 35, 15, 5, 10, 20, 25], &flops_only);
        assert!((ns - 2.0 * 15125.0).abs() < 1e-6, "{ns}");
    }

    #[test]
    fn one_matrix_is_a_leaf() {
        assert_eq!(best(&[3, 4], &COSTS), (Tree::Leaf(0), 0.0));
    }
}
