//! The contraction order of an `einsum` under `numerics.reassociate` (rule 9, W3-R4R5 §4): every way of pairing up to 4 operands,
//! a subset DP up to 8, greedy by the smallest intermediate (opt_einsum's rule) above.
//!
//! A path is a list of steps over a running list of terms: step `(i, j)` removes terms `i` and `j` (`i` is the left operand) and
//! appends their product. A candidate pair costs its flops plus its bytes at the device's bandwidth plus a launch, the measure
//! of the rewrite engine's T2 rules.

use crate::gpu::hash::FxMap;
use crate::numerics::Numerics;

pub(super) type Path = Vec<(usize, usize)>;

/// What a pair costs on the device.
#[derive(Clone, Copy, Debug)]
pub(super) struct Costs {
    pub(super) flops_per_ns: f64,
    pub(super) bytes_per_ns: f64,
    pub(super) launch_ns: f64,
}

/// The labels of each operand and of the output as bit sets, and the extent of each label.
pub(super) struct Problem<'a> {
    pub(super) inputs: &'a [u64],
    pub(super) output: u64,
    pub(super) extent: &'a [u32; 52],
}

impl Problem<'_> {
    fn size(&self, labels: u64) -> f64 {
        (0..52).filter(|&bit| labels >> bit & 1 == 1).map(|bit| f64::from(self.extent[bit])).product()
    }

    /// The cost of the product of terms with the labels `a` and `b`, and the labels of the result when `keep` is what is needed
    /// afterwards; `None` when the result has more than 4 axes.
    fn pair(&self, a: u64, b: u64, keep: u64, costs: &Costs) -> Option<(f64, u64)> {
        let result = (a | b) & keep;
        if result.count_ones() > 4 {
            return None;
        }
        let flops = 2.0 * self.size(a | b);
        let bytes = 4.0 * (self.size(a) + self.size(b) + self.size(result));
        Some((flops / costs.flops_per_ns + bytes / costs.bytes_per_ns + costs.launch_ns, result))
    }

    /// The labels operands outside `terms` and the output still need.
    fn needed_besides(&self, terms: &[u64], skip: &[usize]) -> u64 {
        terms.iter().enumerate().filter(|(t, _)| !skip.contains(t)).fold(self.output, |mask, (_, &labels)| mask | labels)
    }
}

/// The default order: the running result against the next operand.
pub(super) fn left_to_right(n: usize) -> Path {
    (0..n.saturating_sub(1)).map(|step| if step == 0 { (0, 1) } else { (n - step - 1, 0) }).collect()
}

/// The cost of `path`, `None` if a step makes a result of more than 4 axes.
pub(super) fn cost(problem: &Problem<'_>, path: &[(usize, usize)], costs: &Costs) -> Option<f64> {
    let mut terms = problem.inputs.to_vec();
    let mut total = 0.0;
    for &(i, j) in path {
        let keep = problem.needed_besides(&terms, &[i, j]);
        let (step, result) = problem.pair(terms[i], terms[j], keep, costs)?;
        total += step;
        let (hi, lo) = (i.max(j), i.min(j));
        terms.remove(hi);
        terms.remove(lo);
        terms.push(result);
    }
    Some(total)
}

/// The cheapest order of the operands (exhaustive up to 4, subset DP up to 8, greedy above); the default order when no order
/// avoids a result of more than 4 axes (the contraction then reports it).
pub(super) fn search(problem: &Problem<'_>, costs: &Costs) -> Path {
    let n = problem.inputs.len();
    let found = match n {
        0..=2 => Some(left_to_right(n)),
        3..=4 => exhaustive(problem, costs),
        5..=8 => subsets(problem, costs),
        _ => greedy(problem, costs),
    };
    found.unwrap_or_else(|| left_to_right(n))
}

/// Every order of pairing, depth first.
fn exhaustive(problem: &Problem<'_>, costs: &Costs) -> Option<Path> {
    fn go(problem: &Problem<'_>, costs: &Costs, terms: &[u64], path: &mut Path, spent: f64, best: &mut Option<(f64, Path)>) {
        if terms.len() == 1 {
            if best.as_ref().is_none_or(|(cost, _)| spent < *cost) {
                *best = Some((spent, path.clone()));
            }
            return;
        }
        for i in 0..terms.len() {
            for j in i + 1..terms.len() {
                let keep = problem.needed_besides(terms, &[i, j]);
                let Some((step, result)) = problem.pair(terms[i], terms[j], keep, costs) else { continue };
                let mut rest: Vec<u64> = terms.iter().enumerate().filter(|&(t, _)| t != i && t != j).map(|(_, &labels)| labels).collect();
                rest.push(result);
                path.push((i, j));
                go(problem, costs, &rest, path, spent + step, best);
                path.pop();
            }
        }
    }
    let mut best = None;
    go(problem, costs, problem.inputs, &mut Vec::new(), 0.0, &mut best);
    best.map(|(_, path)| path)
}

/// The best tree over every subset of the operands (`3^n` splits).
fn subsets(problem: &Problem<'_>, costs: &Costs) -> Option<Path> {
    let n = problem.inputs.len();
    let full = (1usize << n) - 1;
    // What a subset yields: the labels of its operands that the rest and the output still need.
    let yields: Vec<u64> = (0..=full)
        .map(|set| {
            let (inside, outside) = (0..n).fold((0, problem.output), |(inside, outside), t| if set >> t & 1 == 1 { (inside | problem.inputs[t], outside) } else { (inside, outside | problem.inputs[t]) });
            inside & outside
        })
        .collect();
    let mut best: Vec<Option<(f64, usize)>> = vec![None; full + 1];
    for t in 0..n {
        best[1 << t] = Some((0.0, 0));
    }
    for set in 1..=full {
        if set.is_power_of_two() {
            continue;
        }
        // Splits with the lowest operand in the left half, so each is seen once.
        let low = set & set.wrapping_neg();
        let mut part = (set - 1) & set;
        while part > 0 {
            if part & low != 0 {
                let other = set ^ part;
                if let (Some((c1, _)), Some((c2, _))) = (best[part], best[other])
                    && let Some((step, _)) = problem.pair(yields[part], yields[other], yields[set], costs)
                {
                    let total = c1 + c2 + step;
                    if best[set].is_none_or(|(cost, _)| total < cost) {
                        best[set] = Some((total, part));
                    }
                }
            }
            part = (part - 1) & set;
        }
    }
    best[full]?;
    // The tree as steps over the running list.
    fn emit(best: &[Option<(f64, usize)>], set: usize, list: &mut Vec<usize>, path: &mut Path) {
        if set.is_power_of_two() {
            return;
        }
        let part = best[set].expect("a split").1;
        let other = set ^ part;
        emit(best, part, list, path);
        emit(best, other, list, path);
        let at = |list: &Vec<usize>, wanted: usize| list.iter().position(|&s| s == wanted).expect("a built subset");
        let (i, j) = (at(list, part), at(list, other));
        path.push((i, j));
        list.retain(|&s| s != part && s != other);
        list.push(set);
    }
    let mut list: Vec<usize> = (0..n).map(|t| 1 << t).collect();
    let mut path = Vec::new();
    emit(&best, full, &mut list, &mut path);
    Some(path)
}

/// The pair with the smallest result, repeatedly (ties: the cheaper pair, then the lowest positions).
fn greedy(problem: &Problem<'_>, costs: &Costs) -> Option<Path> {
    let mut terms = problem.inputs.to_vec();
    let mut path = Path::new();
    while terms.len() > 1 {
        let mut best: Option<((f64, f64), (usize, usize), u64)> = None;
        for i in 0..terms.len() {
            for j in i + 1..terms.len() {
                let keep = problem.needed_besides(&terms, &[i, j]);
                let Some((step, result)) = problem.pair(terms[i], terms[j], keep, costs) else { continue };
                let key = (problem.size(result), step);
                if best.is_none_or(|(k, ..)| key < k) {
                    best = Some((key, (i, j), result));
                }
            }
        }
        let (_, (i, j), result) = best?;
        terms.remove(j);
        terms.remove(i);
        terms.push(result);
        path.push((i, j));
    }
    Some(path)
}

/// What an `einsum` call is remembered by: its spec, the shapes of its operands and the numerics (which decide whether it is
/// searched at all).
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub(crate) struct Key {
    pub(super) spec: String,
    pub(super) shapes: Vec<[usize; 4]>,
    pub(super) numerics: Numerics,
}

/// The paths found so far (in `JitState`, next to the selection cache), so a loop pays the search once.
pub(crate) type Memo = std::sync::Mutex<FxMap<Key, std::sync::Arc<Path>>>;

#[cfg(test)]
mod tests {
    use super::*;

    const COSTS: Costs = Costs { flops_per_ns: 1000.0, bytes_per_ns: 500.0, launch_ns: 5000.0 };

    fn mask(labels: &str) -> u64 {
        labels.bytes().fold(0, |mask, label| mask | super::super::bit(label))
    }

    fn extents(sizes: &[(char, u32)]) -> [u32; 52] {
        let mut extent = [1; 52];
        for &(label, size) in sizes {
            extent[super::super::index(label as u8)] = size;
        }
        extent
    }

    #[test]
    fn a_skewed_chain_contracts_the_thin_pair_first() {
        // (ij · jk) · kl costs i·j·k + i·k·l, ij · (jk · kl) costs j·k·l + i·j·l: the second is far cheaper.
        let extent = extents(&[('i', 1000), ('j', 10), ('k', 1000), ('l', 10)]);
        let problem = Problem { inputs: &[mask("ij"), mask("jk"), mask("kl")], output: mask("il"), extent: &extent };
        let path = search(&problem, &COSTS);
        assert_eq!(path[0], (1, 2), "{path:?}");
        assert!(cost(&problem, &path, &COSTS).unwrap() < cost(&problem, &left_to_right(3), &COSTS).unwrap());
        // Swapped sizes: the default order is the best one.
        let extent = extents(&[('i', 10), ('j', 1000), ('k', 10), ('l', 1000)]);
        let problem = Problem { inputs: &[mask("ij"), mask("jk"), mask("kl")], output: mask("il"), extent: &extent };
        assert_eq!(search(&problem, &COSTS)[0], (0, 1));
    }

    #[test]
    fn the_subset_search_agrees_with_the_exhaustive_one() {
        let sizes = [10u32, 200, 3, 50, 7, 90];
        for seed in 0..40usize {
            let extent = extents(&"abcdef".chars().enumerate().map(|(t, label)| (label, sizes[(t * 7 + seed * 3) % 6])).collect::<Vec<_>>());
            // A chain with a branch: ab, bc, cd, de, plus a vector on f joined to a.
            let inputs = [mask("ab"), mask("bc"), mask("cd"), mask("de")];
            let problem = Problem { inputs: &inputs, output: mask("ae"), extent: &extent };
            let (a, b) = (exhaustive(&problem, &COSTS).unwrap(), subsets(&problem, &COSTS).unwrap());
            let (ca, cb) = (cost(&problem, &a, &COSTS).unwrap(), cost(&problem, &b, &COSTS).unwrap());
            assert!((ca - cb).abs() <= 1e-6 * ca, "seed {seed}: {a:?} {ca} against {b:?} {cb}");
        }
    }

    #[test]
    fn nine_operands_take_the_greedy_search_in_a_few_milliseconds() {
        let labels = "abcdefghij";
        let inputs: Vec<u64> = (0..9).map(|t| mask(&labels[t..t + 2])).collect();
        let extent = extents(&labels.chars().enumerate().map(|(t, label)| (label, 4 + (t as u32 * 37) % 29)).collect::<Vec<_>>());
        let problem = Problem { inputs: &inputs, output: mask("aj"), extent: &extent };
        let start = std::time::Instant::now();
        let path = search(&problem, &COSTS);
        assert!(start.elapsed() < std::time::Duration::from_millis(5), "{:?}", start.elapsed());
        assert_eq!(path.len(), 8);
        assert!(cost(&problem, &path, &COSTS).is_some());
    }
}
