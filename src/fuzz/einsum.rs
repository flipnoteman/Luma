//! `einsum` programs (W3-R4R5): random specs over up to four operands with repeated labels, summed-out labels, implicit and
//! permuted outputs, run under default and `reassociate` numerics, with the generic lowering on and off. The exact domain
//! (small integers) is bit-compared against a host f64 sum; chains of random floats are bit-compared against the pairwise
//! `matmul`s they mean at default numerics and bounded against f64 when the order is searched.

use proptest::prelude::*;

use super::drive;
use crate::einsum::tests::{operand, reference};
use crate::{einsum, Array, Context, ContextOptions, Numerics};

#[derive(Clone, Debug)]
struct Case {
    spec: String,
    sizes: Vec<(char, usize)>,
    reassociate: bool,
    no_generic: bool,
}

/// Up to four operands of up to three labels from `a..e`, an output that is implicit, or a shuffled subset of the labels.
fn cases() -> impl Strategy<Value = Case> {
    (prop::collection::vec(prop::collection::vec(0u8..5, 1..=3), 1..=4), prop::collection::vec(1usize..=4, 5), any::<u64>(), any::<bool>(), any::<bool>()).prop_map(|(operands, extents, seed, reassociate, no_generic)| {
        let label = |l: u8| char::from(b'a' + l);
        let inputs: Vec<String> = operands.iter().map(|labels| labels.iter().map(|&l| label(l)).collect()).collect();
        let mut present: Vec<char> = inputs.iter().flat_map(|s| s.chars()).collect();
        present.sort_unstable();
        present.dedup();
        let sizes: Vec<(char, usize)> = present.iter().map(|&c| (c, extents[usize::from(c as u8 - b'a')])).collect();
        let mut spec = inputs.join(",");
        if seed % 4 != 0 {
            let mut rng = seed;
            let mut pool = present.clone();
            let mut output = String::new();
            while !pool.is_empty() && output.len() < 4 && rng % 5 != 0 {
                output.push(pool.remove((rng >> 8) as usize % pool.len()));
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            }
            spec = format!("{spec}->{output}");
        }
        Case { spec, sizes, reassociate, no_generic }
    })
}

thread_local! {
    static CONTEXTS: [Context; 2] = [
        Context::with_options(ContextOptions::from_env().calibrate(false)).unwrap(),
        Context::with_options(ContextOptions::from_env().calibrate(false).numerics(Numerics::algebraic())).unwrap(),
    ];
}

fn run(case: Case) -> Result<(), String> {
    CONTEXTS.with(|contexts| {
        let ctx = &contexts[usize::from(case.reassociate)];
        let inputs: Vec<&str> = case.spec.split("->").next().unwrap_or("").split(',').collect();
        let made: Vec<(Array<f32>, Vec<f32>)> = inputs.iter().enumerate().map(|(i, s)| operand(ctx, s, &case.sizes, i + 1)).collect();
        let arrays: Vec<&Array<f32>> = made.iter().map(|m| &m.0).collect();
        let data: Vec<Vec<f32>> = made.iter().map(|m| m.1.clone()).collect();
        crate::einsum::tests::set_no_generic(case.no_generic);
        let got = einsum(&case.spec, &arrays).to_vec();
        crate::einsum::tests::set_no_generic(false);
        match got {
            Ok(got) => {
                let expected = reference(&case.spec, &case.sizes, &data);
                let got: Vec<f64> = got.into_iter().map(f64::from).collect();
                if got == expected { Ok(()) } else { Err(format!("{}: {got:?} against {expected:?}", case.spec)) }
            }
            // The one error a valid spec may have: an intermediate or the implicit output with more than four axes.
            Err(error) if format!("{error:?}").contains("more than 4 axes") || format!("{error:?}").contains("luma arrays have 4") => Ok(()),
            Err(error) => Err(format!("{}: {error}", case.spec)),
        }
    })
}

#[test]
fn einsum_programs() {
    drive("tests/proptest-regressions/fuzz_einsum.txt", cases(), |case| format!("{case:?}"), run);
}

/// A chain of `n` matrices (the extents `dims`, `n + 1` of them) of random floats.
fn chain() -> impl Strategy<Value = (Vec<usize>, u64, bool)> {
    (prop::collection::vec(1usize..=33, 3..=5), any::<u64>(), any::<bool>())
}

#[test]
fn einsum_chains_are_the_pairwise_matmuls() {
    drive("tests/proptest-regressions/fuzz_einsum.txt", chain(), |case| format!("{case:?}"), |(dims, seed, reassociate)| {
        CONTEXTS.with(|contexts| {
            let ctx = &contexts[usize::from(reassociate)];
            let n = dims.len() - 1;
            let random = |len: usize, k: usize| -> Vec<f32> {
                let mut state = seed ^ (k as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15);
                (0..len)
                    .map(|_| {
                        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                        ((state >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
                    })
                    .collect()
            };
            // Matrix `t` is `dims[t]` rows of `dims[t + 1]` columns: luma shape `[dims[t + 1], dims[t]]`.
            let matrices: Vec<Array<f32>> = (0..n).map(|t| Array::from_slice_on(ctx, &random(dims[t] * dims[t + 1], t), [dims[t + 1], dims[t]]).unwrap()).collect();
            let labels: Vec<char> = (0..=n).map(|t| char::from(b'a' + t as u8)).collect();
            let inputs: Vec<String> = (0..n).map(|t| format!("{}{}", labels[t], labels[t + 1])).collect();
            let spec = format!("{}->{}{}", inputs.join(","), labels[0], labels[n]);
            let refs: Vec<&Array<f32>> = matrices.iter().collect();
            let got = einsum(&spec, &refs).to_vec().map_err(|e| format!("{spec}: {e}"))?;
            if !reassociate {
                let by_hand = matrices[1..].iter().fold(matrices[0].clone(), |acc, m| acc.matmul(m)).to_vec().map_err(|e| e.to_string())?;
                return if got == by_hand { Ok(()) } else { Err(format!("{spec} {dims:?}: not the pairwise matmuls")) };
            }
            // Any order: within the reassociating bound of the f64 chain (TF32 is 1.5e-3 of the sum of the absolute terms).
            let host = |abs: bool| -> Vec<f64> {
                let mut acc: Vec<f64> = matrices[0].to_vec().unwrap().into_iter().map(|v| f64::from(if abs { v.abs() } else { v })).collect();
                let mut cols = dims[1];
                for (t, m) in matrices.iter().enumerate().skip(1) {
                    let next = m.to_vec().unwrap();
                    let mut out = vec![0f64; dims[0] * dims[t + 1]];
                    for r in 0..dims[0] {
                        for c in 0..dims[t + 1] {
                            out[r * dims[t + 1] + c] = (0..cols).map(|k| acc[r * cols + k] * f64::from(if abs { next[k * dims[t + 1] + c].abs() } else { next[k * dims[t + 1] + c] })).sum();
                        }
                    }
                    acc = out;
                    cols = dims[t + 1];
                }
                acc
            };
            let (exact, bound) = (host(false), host(true));
            for (i, (&g, (&e, &b))) in got.iter().zip(exact.iter().zip(&bound)).enumerate() {
                if (f64::from(g) - e).abs() > 2e-3 * b + 1e-6 {
                    return Err(format!("{spec} {dims:?}: element {i} is {g}, f64 says {e}"));
                }
            }
            Ok(())
        })
    });
}
