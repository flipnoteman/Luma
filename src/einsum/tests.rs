//! `einsum` against a host f64 reference (small integers: every sum is exact in `f32`), and against the operations it lowers to.

use crate::{einsum, Array, Context, ContextOptions};

fn cpu() -> Context {
    Context::with_options(ContextOptions::from_env().calibrate(false)).unwrap()
}

/// The luma dims of an operand written `subscripts` with the extents `sizes`.
fn dims(subscripts: &str, sizes: &[(char, usize)]) -> [usize; 4] {
    let mut dims = [1usize; 4];
    for (a, label) in subscripts.chars().rev().enumerate() {
        dims[a] = sizes.iter().find(|(l, _)| *l == label).expect("a size for every label").1;
    }
    dims
}

fn values(len: usize, seed: usize) -> Vec<f32> {
    (0..len).map(|i| ((i * 7 + seed * 5 + i / 13) % 9) as f32 - 3.0).collect()
}

fn operand(ctx: &Context, subscripts: &str, sizes: &[(char, usize)], seed: usize) -> (Array<f32>, Vec<f32>) {
    let d = dims(subscripts, sizes);
    let data = values(d.iter().product(), seed);
    (Array::from_slice_on(ctx, &data, d).unwrap(), data)
}

/// The sum over every assignment of the labels, in f64.
fn reference(spec: &str, sizes: &[(char, usize)], data: &[Vec<f32>]) -> Vec<f64> {
    let (lhs, rhs) = spec.split_once("->").unwrap_or((spec, ""));
    let inputs: Vec<&str> = lhs.split(',').collect();
    let output: String = if spec.contains("->") {
        rhs.to_string()
    } else {
        let mut once: Vec<char> = lhs.chars().filter(|&c| c != ',' && lhs.matches(c).count() == 1).collect();
        once.sort_unstable();
        once.into_iter().collect()
    };
    let mut all: Vec<char> = lhs.chars().filter(|&c| c != ',').collect();
    all.sort_unstable();
    all.dedup();
    let extent = |label: char| sizes.iter().find(|(l, _)| *l == label).unwrap().1;
    let flat = |subscripts: &str, at: &dyn Fn(char) -> usize| -> usize {
        let (mut index, mut stride) = (0, 1);
        for label in subscripts.chars().rev() {
            index += at(label) * stride;
            stride *= extent(label);
        }
        index
    };
    let mut out = vec![0f64; output.chars().map(extent).product()];
    let total: usize = all.iter().map(|&l| extent(l)).product();
    for point in 0..total {
        let mut rest = point;
        let mut at = std::collections::HashMap::new();
        for &label in &all {
            at.insert(label, rest % extent(label));
            rest /= extent(label);
        }
        let get = |label: char| at[&label];
        let product: f64 = inputs.iter().zip(data).map(|(s, d)| f64::from(d[flat(s, &get)])).product();
        out[flat(&output, &get)] += product;
    }
    out
}

fn check(spec: &str, sizes: &[(char, usize)]) {
    let ctx = cpu();
    let inputs: Vec<&str> = spec.split("->").next().unwrap().split(',').collect();
    let made: Vec<(Array<f32>, Vec<f32>)> = inputs.iter().enumerate().map(|(i, s)| operand(&ctx, s, sizes, i + 1)).collect();
    let arrays: Vec<&Array<f32>> = made.iter().map(|m| &m.0).collect();
    let data: Vec<Vec<f32>> = made.iter().map(|m| m.1.clone()).collect();
    let got: Vec<f64> = einsum(spec, &arrays).to_vec().unwrap_or_else(|e| panic!("{spec}: {e}")).into_iter().map(f64::from).collect();
    assert_eq!(got, reference(spec, sizes, &data), "{spec}");
}

#[test]
fn a_matrix_product_is_matmul_bitwise() {
    let ctx = cpu();
    let sizes = [('i', 5), ('j', 7), ('k', 3)];
    let (a, _) = operand(&ctx, "ij", &sizes, 1);
    let (b, _) = operand(&ctx, "jk", &sizes, 2);
    assert_eq!(einsum("ij,jk->ik", &[&a, &b]).to_vec().unwrap(), a.matmul(&b).to_vec().unwrap());
    // The transposed output is a view of it.
    assert_eq!(einsum("ij,jk->ki", &[&a, &b]).to_vec().unwrap(), a.matmul(&b).transpose().to_vec().unwrap());
}

#[test]
fn a_batched_product_is_matmul_batched_bitwise() {
    let ctx = cpu();
    let sizes = [('b', 4), ('i', 5), ('j', 7), ('k', 3)];
    let (a, _) = operand(&ctx, "bij", &sizes, 1);
    let (b, _) = operand(&ctx, "bjk", &sizes, 2);
    assert_eq!(einsum("bij,bjk->bik", &[&a, &b]).to_vec().unwrap(), a.matmul_batched(&b).to_vec().unwrap());
}

#[test]
fn single_operand_specs_are_the_operations_they_name() {
    let ctx = cpu();
    let sizes = [('i', 6), ('j', 4)];
    let (sq, _) = operand(&ctx, "ii", &[('i', 6)], 1);
    assert_eq!(einsum("ii->i", &[&sq]).to_vec().unwrap(), sq.diag(0).to_vec().unwrap());
    assert_eq!(einsum("ii", &[&sq]).to_vec().unwrap(), sq.trace().to_vec().unwrap());
    let (m, _) = operand(&ctx, "ij", &sizes, 2);
    assert_eq!(einsum("ij->", &[&m]).to_vec().unwrap(), m.sum().to_vec().unwrap());
    assert_eq!(einsum("ij->ji", &[&m]).to_vec().unwrap(), m.transpose().to_vec().unwrap());
    assert_eq!(einsum("ij->j", &[&m]).to_vec().unwrap(), m.sum_axis(1).to_vec().unwrap());
    check("ii", &[('i', 5)]);
    check("iij->j", &[('i', 3), ('j', 4)]);
    check("bii->b", &[('b', 3), ('i', 4)]);
}

#[test]
fn trace_sums_each_diagonal() {
    let ctx = cpu();
    let a = Array::<f32>::from_slice_on(&ctx, &values(3 * 3 * 2, 1), [3, 3, 2]).unwrap();
    let expected: Vec<f32> = (0..2).map(|b| (0..3).map(|i| a.to_vec().unwrap()[b * 9 + i * 3 + i]).sum()).collect();
    assert_eq!(a.trace().to_vec().unwrap(), expected);
    assert_eq!(a.trace().shape().dims(), [1, 1, 2, 1]);
}

#[test]
fn pairs_that_are_not_matrix_products_do_not_refuse() {
    let s = [('a', 2), ('b', 3), ('c', 2), ('d', 3), ('i', 4), ('j', 5), ('k', 3)];
    check("ij,ij->ij", &s);
    check("ij,ij->", &s);
    check("i,j->ij", &s);
    check("i,j->ji", &s);
    check("abc,abd->cd", &s);
    // Three batch axes (the spec's `abcij,abcjk->abcik` has five axes, more than a luma array).
    check("abcj,abcj->abc", &[('a', 2), ('b', 3), ('c', 2), ('j', 4)]);
    check("abcj,abcj->cab", &[('a', 2), ('b', 3), ('c', 2), ('j', 4)]);
    check("ij,jk,kl->il", &[('i', 3), ('j', 4), ('k', 2), ('l', 5)]);
    check("ij,kj->ik", &s);
    check("ijk,jkl->il", &[('i', 3), ('j', 4), ('k', 2), ('l', 5)]);
    check("bij,bjk->ik", &[('b', 3), ('i', 3), ('j', 4), ('k', 2)]);
    check("ij,jk", &s);
    check("i,i", &s);
    check("ij,j->i", &s);
    check("j,jk->k", &s);
}

#[test]
fn three_and_four_operand_chains_are_pairwise_matmuls() {
    let ctx = cpu();
    let sizes = [('i', 5), ('j', 7), ('k', 3), ('l', 4), ('m', 6)];
    let made: Vec<Array<f32>> = ["ij", "jk", "kl", "lm"].iter().enumerate().map(|(n, s)| operand(&ctx, s, &sizes, n + 1).0).collect();
    let by_hand = made[0].matmul(&made[1]).matmul(&made[2]);
    assert_eq!(einsum("ij,jk,kl->il", &[&made[0], &made[1], &made[2]]).to_vec().unwrap(), by_hand.to_vec().unwrap());
    let four = by_hand.matmul(&made[3]);
    assert_eq!(einsum("ij,jk,kl,lm->im", &[&made[0], &made[1], &made[2], &made[3]]).to_vec().unwrap(), four.to_vec().unwrap());
}

#[test]
fn malformed_specs_poison() {
    let ctx = cpu();
    let a = Array::<f32>::from_slice_on(&ctx, &values(6, 1), [2, 3]).unwrap();
    let b = Array::<f32>::from_slice_on(&ctx, &values(6, 2), [3, 2]).unwrap();
    for (spec, operands) in [
        ("ij,jk->iz", vec![&a, &b]),
        ("i1,jk->ik", vec![&a, &b]),
        ("ij,jk->ii", vec![&a, &b]),
        ("ij,jk,kl->il", vec![&a, &b]),
        ("ij->ij", vec![&a, &b]),
        ("...,jk->ik", vec![&a, &b]),
        ("ij,jk->ik", vec![&a, &a]),
        ("ijk,jk->ik", vec![&a, &b]),
        ("ijklm->i", vec![&a]),
    ] {
        assert!(einsum(spec, &operands).to_vec().is_err(), "{spec} should poison");
    }
}
