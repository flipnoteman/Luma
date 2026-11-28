//! `einsum`: contractions written as subscripts (W3-R4R5 §4).
//!
//! Subscripts list the axes of an operand slowest first, as the matrices are written (`"ij,jk->ik"` is `matmul` of an `[J, I]`
//! array and a `[K, J]` array, the `[K, I]` result's first axis being `k`). The operands are contracted pairwise, left to right
//! (or in the order [`path`] finds under `numerics.reassociate`). One pair takes the first of three lowerings that applies:
//!
//! 1. `matmul` / `matmul_batched`, for one contracted label, at most one `M` and one `N` label and at most two batch labels;
//! 2. the generic lowering of a comprehension with an `Origin::Api(Einsum)` form, for any other pair (Hadamard and outer products,
//!    several free or contracted labels, three or more batch labels) that the generic strategies can run;
//! 3. copy-then-matmul: the labels of each group merged into one axis (a reshape, which evaluates what is not a view), one
//!    `matmul_batched`, and a reshape back. It always exists, so no valid spec is refused.
//!
//! The sum of one pair runs over its contracted labels in the order of the spec string, the first label the slowest axis of the
//! flattened `K`: that is the exactness contract (07 §4). The operands of a pair are first reduced: a label that is in no other
//! operand and not in the output is summed out of the operand, a label repeated inside an operand takes the diagonal.

use std::panic::Location;

use crate::array::{Array, Failure};
use crate::context::Context;
use crate::error::LumaError;
use crate::jit::comprehension::build::Bodies;
use crate::jit::comprehension::{lower, Access, AffineMap, Body, CombineOp, CombinePart, Combiner, Comprehension, Dim, Epilogue, MapRef, OpTag, Origin, OutMap, Props, MAX_DIMS};
use crate::jit::{Edge, ElemTy};
use crate::shape::Shape;
use smallvec::{smallvec, SmallVec};

#[cfg(test)]
mod tests;

// Which tier each pair of this thread's `einsum` calls took, and whether tier 2 is switched off (`lowerable == false`).
#[cfg(test)]
thread_local! {
    static TIERS: std::cell::RefCell<Vec<u8>> = const { std::cell::RefCell::new(Vec::new()) };
    static NO_GENERIC: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

fn note_tier(tier: u8) {
    #[cfg(test)]
    TIERS.with(|tiers| tiers.borrow_mut().push(tier));
    let _ = tier;
}

/// The label of each axis of an operand, fastest axis first; `None` is an axis of extent 1 that no label names.
type Slots = [Option<u8>; 4];

/// The `a-zA-Z` labels, one bit each.
fn bit(label: u8) -> u64 {
    1u64 << index(label)
}

fn index(label: u8) -> usize {
    match label {
        b'a'..=b'z' => usize::from(label - b'a'),
        _ => 26 + usize::from(label - b'A'),
    }
}

/// The labels of a spec: one list per operand and the output, slowest axis first.
#[derive(Debug, PartialEq, Eq)]
struct Spec {
    inputs: Vec<Vec<u8>>,
    output: Vec<u8>,
}

fn malformed(message: impl Into<String>) -> LumaError {
    LumaError::InvalidOption(format!("einsum: {}", message.into()))
}

/// Parses `spec` for `operands` operands (NumPy's syntax without `...`).
fn parse(spec: &str, operands: usize) -> Result<Spec, LumaError> {
    let text: String = spec.chars().filter(|c| !c.is_whitespace()).collect();
    if text.contains('.') {
        return Err(malformed("`...` is not supported"));
    }
    let (lhs, rhs) = match text.split_once("->") {
        Some((lhs, rhs)) => (lhs, Some(rhs)),
        None => (text.as_str(), None),
    };
    let labels = |part: &str| -> Result<Vec<u8>, LumaError> {
        part.bytes().map(|b| if b.is_ascii_alphabetic() { Ok(b) } else { Err(malformed(format!("`{}` is not a label in `{spec}`", char::from(b)))) }).collect()
    };
    let inputs = lhs.split(',').map(labels).collect::<Result<Vec<_>, _>>()?;
    if inputs.len() != operands {
        return Err(malformed(format!("`{spec}` names {} operands, {operands} given", inputs.len())));
    }
    let mut counts = [0u32; 52];
    for &label in inputs.iter().flatten() {
        counts[index(label)] += 1;
    }
    let output = match rhs {
        Some(rhs) => {
            let output = labels(rhs)?;
            for (i, &label) in output.iter().enumerate() {
                if counts[index(label)] == 0 {
                    return Err(malformed(format!("output label `{}` is in no operand", char::from(label))));
                }
                if output[..i].contains(&label) {
                    return Err(malformed(format!("output label `{}` is repeated", char::from(label))));
                }
            }
            output
        }
        None => {
            let mut once: Vec<u8> = inputs.iter().flatten().copied().filter(|&label| counts[index(label)] == 1).collect();
            once.sort_unstable();
            once
        }
    };
    if let Some(rank) = inputs.iter().map(Vec::len).chain([output.len()]).find(|&rank| rank > 4) {
        return Err(LumaError::InvalidShape(format!("einsum: an operand or the result of `{spec}` has {rank} axes, luma arrays have 4")));
    }
    Ok(Spec { inputs, output })
}

/// What every pair of one `einsum` call reads: the extent of each label and where each is first written in the spec.
struct Labels {
    extent: [u32; 52],
    first: [u32; 52],
}

impl Labels {
    fn extent(&self, label: u8) -> usize {
        self.extent[index(label)] as usize
    }

    fn product(&self, labels: &[u8]) -> usize {
        labels.iter().map(|&label| self.extent(label)).product()
    }
}

/// An operand being contracted: `array`'s axis `a` carries label `slots[a]`.
struct Term {
    slots: Slots,
    array: Array<f32>,
}

/// The labels of `slots`, fastest first.
fn labels_of(slots: &Slots) -> impl Iterator<Item = u8> + '_ {
    slots.iter().flatten().copied()
}

/// The first slots hold `labels` (fastest first), the rest none.
fn pack(labels: &[u8]) -> Slots {
    let mut slots = [None; 4];
    for (slot, &label) in slots.iter_mut().zip(labels) {
        *slot = Some(label);
    }
    slots
}

impl Term {
    fn mask(&self) -> u64 {
        labels_of(&self.slots).fold(0, |mask, label| mask | bit(label))
    }

    /// The array with the labels on the axes `want` says: a lazy permutation, no copy (the axes `want` leaves free are the
    /// extent-1 ones, in order).
    fn permuted(&self, want: &Slots, location: &'static Location<'static>) -> Array<f32> {
        let mut free = (0..4).filter(|&a| self.slots[a].is_none());
        let mut perm = [0usize; 4];
        for (axis, label) in perm.iter_mut().zip(want) {
            *axis = match label {
                Some(label) => self.slots.iter().position(|&s| s == Some(*label)).expect("every wanted label is on the operand"),
                None => free.next().expect("the operand has as many unnamed axes as the layout"),
            };
        }
        if perm == [0, 1, 2, 3] {
            return self.array.clone();
        }
        self.array.permute_at(perm, false, "einsum", location)
    }

    /// This term with its labels on the axes `want` says.
    fn arranged(self, want: &Slots, location: &'static Location<'static>) -> Term {
        if self.slots == *want {
            return self;
        }
        Term { array: self.permuted(want, location), slots: *want }
    }
}

/// `term` with the diagonal taken for a label it repeats, and the labels no one needs (`needed`) summed out.
#[track_caller]
fn prepare(mut term: Term, needed: u64, location: &'static Location<'static>) -> Term {
    while let Some((p, q)) = (0..4).find_map(|p| {
        let label = term.slots[p]?;
        (p + 1..4).find(|&q| term.slots[q] == Some(label)).map(|q| (p, q))
    }) {
        let rest: Vec<usize> = (0..4).filter(|&a| a != p && a != q).collect();
        let swapped = term.array.permute_at([p, q, rest[0], rest[1]], false, "einsum", location);
        let diagonal = swapped.diag_at(0, location);
        let [_, _, b2, b3] = diagonal.shape().dims();
        let array = if b2 == 1 && b3 == 1 { diagonal } else { diagonal.permute_at([0, 2, 3, 1], false, "einsum", location) };
        term = Term { slots: [term.slots[p], term.slots[rest[0]], term.slots[rest[1]], None], array };
    }
    let unneeded: Vec<usize> = (0..4).filter(|&a| term.slots[a].is_some_and(|label| needed & bit(label) == 0)).collect();
    if unneeded.is_empty() {
        return term;
    }
    if unneeded.len() == labels_of(&term.slots).count() {
        return Term { slots: [None; 4], array: term.array.sum() };
    }
    for axis in unneeded {
        term.array = term.array.sum_axis(axis);
        term.slots[axis] = None;
    }
    term
}

/// The product of `a` and `b` summed over the labels they share that `keep` (the output and the operands still to come) does
/// not need, as a term; `want` is the layout of the result if the caller has one.
#[track_caller]
fn pair(labels: &Labels, a: Term, b: Term, keep: u64, want: Option<Slots>, location: &'static Location<'static>) -> Result<Term, Failure> {
    let a = prepare(a, keep | b.mask(), location);
    let b = prepare(b, keep | a.mask(), location);
    let (ma, mb) = (a.mask(), b.mask());
    if ((ma | mb) & keep).count_ones() > 4 {
        return Err(LumaError::InvalidShape("einsum: an intermediate result has more than 4 axes".into()).into());
    }
    let of = |term: &Term, mask: u64| -> Vec<u8> { labels_of(&term.slots).filter(|&label| mask & bit(label) != 0).collect() };
    let batch = of(&a, ma & mb & keep);
    let m = of(&a, ma & !mb);
    let n = of(&b, mb & !ma);
    // Fastest first: the first label of the spec string is the slowest axis of the flattened `K`.
    let mut k = of(&a, ma & mb & !keep);
    k.sort_unstable_by_key(|&label| std::cmp::Reverse(labels.first[index(label)]));
    if k.len() == 1 && m.len() <= 1 && n.len() <= 1 && batch.len() <= 2 {
        note_tier(1);
        let result = matmul_pair(&a, &b, k[0], m.first().copied(), n.first().copied(), &batch, location);
        let slots = want.unwrap_or(result.slots);
        return Ok(result.arranged(&slots, location));
    }
    if let Some(result) = generic_pair(labels, &a, &b, &k, keep, want)? {
        note_tier(2);
        return Ok(result);
    }
    note_tier(3);
    let result = copy_then_matmul(labels, &a, &b, &k, &m, &n, &batch, location);
    let slots = want.unwrap_or(result.slots);
    Ok(result.arranged(&slots, location))
}

/// Tier 2: the pair as a comprehension form with `Origin::Api(Einsum)`, run by the generic lowering. `None` when the form has no
/// lowering (more than 6 dims, or what the generic strategies cannot run).
#[track_caller]
fn generic_pair(labels: &Labels, a: &Term, b: &Term, k: &[u8], keep: u64, want: Option<Slots>) -> Result<Option<Term>, Failure> {
    #[cfg(test)]
    if NO_GENERIC.with(std::cell::Cell::get) {
        return Ok(None);
    }
    // The result's axes, fastest first: the caller's layout, else A's labels then B's.
    let result: Vec<u8> = match want {
        Some(want) => labels_of(&want).collect(),
        None => {
            let kept = |label: &u8| keep & bit(*label) != 0;
            let from_a: Vec<u8> = labels_of(&a.slots).filter(kept).collect();
            let from_b: Vec<u8> = labels_of(&b.slots).filter(kept).filter(|label| !from_a.contains(label)).collect();
            [from_a, from_b].concat()
        }
    };
    if result.len() + k.len() > MAX_DIMS {
        return Ok(None);
    }
    // Dims: the result's axes in order, then the contracted labels, the last label of the spec string the fastest.
    let dims: Vec<u8> = result.iter().chain(k).copied().collect();
    let access = |term: &Term| -> Result<Access, Failure> {
        let mut map = AffineMap::ZERO;
        for (row, slot) in map.rows.iter_mut().zip(&term.slots) {
            if let Some(label) = slot {
                let dim = dims.iter().position(|d| d == label).expect("every label of an operand is a dim");
                row.coeff[dim] = 1;
            }
        }
        let shape = term.array.shape().dims().map(|d| u32::try_from(d).unwrap_or(u32::MAX));
        Ok(Access { src: term.array.src()?, shape, ty: ElemTy::F32, map: MapRef::owned(map), edge: Edge::None, fill: None, indirect: None, props: Props::NONE })
    };
    let mut axes = [None; 4];
    for (d, axis) in axes.iter_mut().zip(0..result.len()) {
        *d = Some(axis as u8);
    }
    let form = Comprehension {
        space: dims.iter().enumerate().map(|(d, &label)| if d < result.len() { Dim::parallel(labels.extent(label) as u64) } else { Dim::reduce(labels.extent(label) as u64) }).collect(),
        inputs: SmallVec::from_buf([access(a)?, access(b)?]),
        body: Body { instrs: Bodies::x0_times_x1(ElemTy::F32), scalars: SmallVec::new(), outs: smallvec![0] },
        combiner: (!k.is_empty()).then(|| Combiner::of(CombineOp::Single(CombinePart::Add), ElemTy::F32)),
        epilogue: Epilogue::None,
        output: OutMap::new(axes),
        props: Props::NONE,
        origin: Origin::Api(OpTag::Einsum),
        ty: ElemTy::F32,
    };
    if form.validate().is_err() || !lower::lowers(&form) {
        return Ok(None);
    }
    Ok(Some(Term { slots: pack(&result), array: lower::build(a.array.context(), &form) }))
}

/// Tier 1: the pair is one `matmul` or `matmul_batched` of views.
#[track_caller]
fn matmul_pair(a: &Term, b: &Term, k: u8, m: Option<u8>, n: Option<u8>, batch: &[u8], location: &'static Location<'static>) -> Term {
    let (b0, b1) = (batch.first().copied(), batch.get(1).copied());
    let lhs = a.permuted(&[Some(k), m, b0, b1], location);
    let rhs = b.permuted(&[n, Some(k), b0, b1], location);
    let array = if batch.is_empty() { lhs.matmul(&rhs) } else { lhs.matmul_batched(&rhs) };
    Term { slots: [n, m, b0, b1], array }
}

/// Tier 3: each group of labels merged into one axis (a reshape of a view, which evaluates what is not one), one product, and a
/// reshape back.
#[allow(clippy::too_many_arguments)]
#[track_caller]
fn copy_then_matmul(labels: &Labels, a: &Term, b: &Term, k: &[u8], m: &[u8], n: &[u8], batch: &[u8], location: &'static Location<'static>) -> Term {
    let [kc, mc, nc, bc] = [k, m, n, batch].map(|group| labels.product(group));
    let lhs = a.permuted(&pack(&[k, m, batch].concat()), location).reshape([kc, mc, bc, 1]);
    let rhs = b.permuted(&pack(&[n, k, batch].concat()), location).reshape([nc, kc, bc, 1]);
    let product = if batch.is_empty() { lhs.matmul(&rhs) } else { lhs.matmul_batched(&rhs) };
    let slots = pack(&[n, m, batch].concat());
    let dims = slots.map(|slot| slot.map_or(1, |label| labels.extent(label)));
    Term { slots, array: product.reshape(dims) }
}

/// `spec` over `operands`, as an array.
#[track_caller]
fn run(ctx: &Context, spec: &str, operands: &[&Array<f32>], location: &'static Location<'static>) -> Result<Array<f32>, Failure> {
    let Spec { inputs, output } = parse(spec, operands.len())?;
    let mut labels = Labels { extent: [0; 52], first: [u32::MAX; 52] };
    let mut order = 0;
    for (n, (operand, subscripts)) in operands.iter().zip(&inputs).enumerate() {
        if !ctx.same_as(operand.context()) {
            return Err(LumaError::ContextMismatch.into());
        }
        let dims = operand.shape().dims();
        if dims[subscripts.len()..].iter().any(|&d| d != 1) {
            return Err(LumaError::InvalidShape(format!("einsum: operand {n} has shape {}, its subscripts name {} axes", operand.shape(), subscripts.len())).into());
        }
        for (a, &label) in subscripts.iter().rev().enumerate() {
            let extent = u32::try_from(dims[a]).map_err(|_| LumaError::DimensionTooLarge { shape: dims })?;
            let at = index(label);
            if labels.extent[at] != 0 && labels.extent[at] != extent {
                return Err(LumaError::InvalidShape(format!("einsum: label `{}` is {} in one operand and {extent} in another", char::from(label), labels.extent[at])).into());
            }
            labels.extent[at] = extent;
        }
        for &label in subscripts {
            let at = index(label);
            if labels.first[at] == u32::MAX {
                labels.first[at] = order;
                order += 1;
            }
        }
    }
    let mut terms: Vec<Term> = operands
        .iter()
        .zip(&inputs)
        .map(|(operand, subscripts)| {
            let reversed: Vec<u8> = subscripts.iter().rev().copied().collect();
            Term { slots: pack(&reversed), array: (*operand).clone() }
        })
        .collect();
    let reversed: Vec<u8> = output.iter().rev().copied().collect();
    let wanted = pack(&reversed);
    let wanted_mask = output.iter().fold(0, |mask, &label| mask | bit(label));
    if terms.len() == 1 {
        let term = prepare(terms.pop().expect("one operand"), wanted_mask, location);
        return Ok(term.arranged(&wanted, location).array);
    }
    for (i, j) in left_to_right(terms.len()) {
        let (a, b) = if i > j {
            let a = terms.remove(i);
            (a, terms.remove(j))
        } else {
            let b = terms.remove(j);
            (terms.remove(i), b)
        };
        let keep = terms.iter().fold(wanted_mask, |mask, term| mask | term.mask());
        let last = terms.is_empty();
        terms.push(pair(&labels, a, b, keep, last.then_some(wanted), location)?);
    }
    Ok(terms.pop().expect("one result").array)
}

/// The pairs of the default order over `n` operands: positions in the running list, the result of a step appended to it.
fn left_to_right(n: usize) -> Vec<(usize, usize)> {
    (0..n.saturating_sub(1)).map(|step| if step == 0 { (0, 1) } else { (n - step - 1, 0) }).collect()
}

/// Contracts `operands` as `spec` says (`"ij,jk->ik"` is a matrix product; see the [module](self) for the syntax).
///
/// Subscripts list the axes of an operand slowest first, as matrices are written: an `[J, I]` array is `"ij"` (`I` rows of `J`
/// elements), and a `[K, I]` result `"ik"`. The output is explicit (`->`) or, without it, the labels that appear once, in
/// alphabetical order. A label repeated inside one operand takes the diagonal (`"ii->i"`), a label that is not in the output is
/// summed. `...` is not supported, and operands and result have at most 4 axes.
///
/// More than two operands are contracted pairwise from the left; under [`Numerics::reassociate`](crate::Numerics::reassociate)
/// the cheapest order is searched instead and remembered per spec and shapes. A pair that is not a matrix product still runs
/// (as a generic comprehension, or through a copy and a `matmul`), so no valid spec is refused.
///
/// # Errors
///
/// The result is poisoned with [`LumaError::InvalidOption`] for a malformed spec (unknown label, repeated output label,
/// `...`, wrong operand count), [`LumaError::InvalidShape`] for an operand or intermediate with more than 4 axes or a
/// label of two extents, and [`LumaError::ContextMismatch`] if the operands live on different contexts.
///
/// ```no_run
/// use luma::{einsum, Array};
///
/// let a = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], [2, 2])?;
/// let b = Array::from_slice(&[5.0f32, 6.0, 7.0, 8.0], [2, 2])?;
/// assert_eq!(einsum("ij,jk->ik", &[&a, &b]).to_vec()?, a.matmul(&b).to_vec()?);
/// assert_eq!(einsum("ii", &[&a]).to_vec()?, [5.0]);
/// # Ok::<(), luma::LumaError>(())
/// ```
#[track_caller]
pub fn einsum(spec: &str, operands: &[&Array<f32>]) -> Array<f32> {
    let location = Location::caller();
    let ctx = match operands.first() {
        Some(first) => first.context().clone(),
        None => crate::default_context().expect("einsum needs a context"),
    };
    let result = run(&ctx, spec, operands, location);
    Array::resolve(&ctx, Shape::new([1, 1, 1, 1]), result, "einsum", location)
}

impl Array<f32> {
    /// The sum of the diagonal of each matrix (axes 0 and 1; axes 2 and 3 are batches): the result has shape `[1, 1, b2, b3]`.
    ///
    /// Fails (poisoned array) as [`Array::diag`] does for an empty diagonal, which only a poisoned `self` has.
    ///
    /// ```no_run
    /// # use luma::Array;
    /// let a = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], [2, 2])?;
    /// assert_eq!(a.trace().to_vec()?, [5.0]);
    /// # Ok::<(), luma::LumaError>(())
    /// ```
    #[track_caller]
    pub fn trace(&self) -> Array<f32> {
        let diagonal = self.diag_at(0, Location::caller());
        let [_, _, b2, b3] = diagonal.shape().dims();
        if b2 == 1 && b3 == 1 { diagonal.sum() } else { diagonal.sum_axis(0) }
    }
}
