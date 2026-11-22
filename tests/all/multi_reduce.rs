//! Reductions that are pending together are recorded in one kernel (see `jit::multireduce`): every value
//! is the bits the same reduction gives when it is recorded alone.

use crate::common;

use common::values;
use luma::{Array, Context, Reduction};

fn data(ctx: &Context, n: usize, seed: u64) -> (Vec<f32>, Array<f32>) {
    let host = values(n, seed, -50.0, 50.0);
    let array = Array::from_slice_on(ctx, &host, n).unwrap();
    (host, array)
}

/// The sizes: one element, less than a block, odd, a block boundary, several workgroups.
const SIZES: [usize; 8] = [1, 7, 255, 1000, 2048, 5001, 200_003, 1 << 20];

/// A mask of `x` that keeps about `1/k` of the elements, a different one for every `k`.
fn mask(x: &Array<f32>, k: usize) -> Array<bool> {
    let step = 17.0 / (k as f32 + 1.0);
    (x * (1.0 + 0.1 * k as f32)).gt(-25.0 + step)
}

#[test]
fn masked_sums_forced_together_equal_the_ones_recorded_alone() {
    let ctx = Context::new().unwrap();
    for n in SIZES {
        let (_, x) = data(&ctx, n, 3);
        // One at a time: nothing else is pending.
        let alone: Vec<f32> = (0..12).map(|k| x.sum_where(&mask(&x, k)).item().unwrap()).collect();
        let together: Vec<Array<f32>> = (0..12).map(|k| x.sum_where(&mask(&x, k))).collect();
        // Read in a different order than they were made.
        for k in (0..12).rev() {
            assert_eq!(together[k].item().unwrap().to_bits(), alone[k].to_bits(), "n = {n}, mask {k}");
        }
    }
}

#[test]
fn describe_columns_mean_variance_min_max() {
    let ctx = Context::new().unwrap();
    for n in [3usize, 1000, 123_457] {
        let columns: Vec<Array<f32>> = (0..8).map(|c| data(&ctx, n, 10 + c).1).collect();
        let stats = |c: &Array<f32>| {
            let mean = c.mean();
            let centered = c - &mean;
            let var = (&centered * &centered).mean();
            (mean, var, c.min(), c.max())
        };
        let alone: Vec<[f32; 4]> = columns
            .iter()
            .map(|c| {
                let (mean, var, min, max) = stats(c);
                [mean.item().unwrap(), var.item().unwrap(), min.item().unwrap(), max.item().unwrap()]
            })
            .collect();
        let together: Vec<_> = columns.iter().map(stats).collect();
        for (c, (mean, var, min, max)) in together.iter().enumerate() {
            // The variance depends on the mean, which is recorded first.
            let got = [var.item().unwrap(), mean.item().unwrap(), min.item().unwrap(), max.item().unwrap()];
            let want = [alone[c][1], alone[c][0], alone[c][2], alone[c][3]];
            for (k, (g, w)) in got.iter().zip(&want).enumerate() {
                assert_eq!(g.to_bits(), w.to_bits(), "n = {n}, column {c}, statistic {k}");
            }
        }
    }
}

#[test]
fn nan_and_infinities_propagate_as_they_do_alone() {
    let ctx = Context::new().unwrap();
    let n = 10_007;
    let mut host = values(n, 5, -1.0, 1.0);
    host[3] = f32::NAN;
    host[5000] = f32::INFINITY;
    host[9000] = f32::NEG_INFINITY;
    let x = Array::from_slice_on(&ctx, &host, n).unwrap();
    let y = Array::from_slice_on(&ctx, &values(n, 6, -1.0, 1.0), n).unwrap();
    let make = || (x.sum(), x.min(), x.max(), (&x * &y).sum(), y.sum(), (&y * 2.0).max(), (&y + 1.0).min());
    let alone = {
        let (a, b, c, d, e, f, g) = make();
        [a.item().unwrap(), b.item().unwrap(), c.item().unwrap(), d.item().unwrap(), e.item().unwrap(), f.item().unwrap(), g.item().unwrap()]
    };
    let (a, b, c, d, e, f, g) = make();
    let got = [g.item().unwrap(), a.item().unwrap(), b.item().unwrap(), c.item().unwrap(), d.item().unwrap(), e.item().unwrap(), f.item().unwrap()];
    let want = [alone[6], alone[0], alone[1], alone[2], alone[3], alone[4], alone[5]];
    for (k, (g, w)) in got.iter().zip(&want).enumerate() {
        assert_eq!(g.to_bits(), w.to_bits(), "reduction {k}: {g} vs {w}");
    }
    assert!(alone[0].is_nan());
}

#[test]
fn types_mix_without_mixing_kernels() {
    let ctx = Context::new().unwrap();
    let n = 70_001;
    let host = values(n, 9, -100.0, 100.0);
    let x = Array::from_slice_on(&ctx, &host, n).unwrap();
    let ints = x.to_i32();
    let words = (&x + 100.0).to_u32();
    let make = || (x.sum(), ints.sum(), ints.max(), words.sum(), words.min(), x.norm(), x.mean(), ints.min());
    let alone = {
        let (a, b, c, d, e, f, g, h) = make();
        (a.item().unwrap(), b.item().unwrap(), c.item().unwrap(), d.item().unwrap(), e.item().unwrap(), f.item().unwrap(), g.item().unwrap(), h.item().unwrap())
    };
    let (a, b, c, d, e, f, g, h) = make();
    // Interleave the types in the order they are read.
    assert_eq!(e.item().unwrap(), alone.4);
    assert_eq!(b.item().unwrap(), alone.1);
    assert_eq!(a.item().unwrap().to_bits(), alone.0.to_bits());
    assert_eq!(d.item().unwrap(), alone.3);
    assert_eq!(g.item().unwrap().to_bits(), alone.6.to_bits());
    assert_eq!(c.item().unwrap(), alone.2);
    assert_eq!(f.item().unwrap().to_bits(), alone.5.to_bits());
    assert_eq!(h.item().unwrap(), alone.7);
    let want_sum: i64 = host.iter().map(|&v| (v as i32) as i64).sum();
    assert_eq!(i64::from(alone.1), want_sum);
}

#[test]
fn several_ops_per_reduction_and_scaled_results() {
    let ctx = Context::new().unwrap();
    for n in [5usize, 3000, 300_000] {
        let (host, x) = data(&ctx, n, 21);
        let make = |scale: f32| {
            let both = x.reduce([Reduction::Min, Reduction::Max]);
            let moments = x.reduce([Reduction::Sum, Reduction::SumOfSquares, Reduction::Min, Reduction::Max]);
            (both, moments, (x.sum() * scale + 1.0), x.norm() / scale, (&x * 2.0).sum() / (n as f32))
        };
        let alone = {
            let (a, b, c, d, e) = make(0.5);
            (a.to_vec().unwrap(), b.to_vec().unwrap(), c.item().unwrap(), d.item().unwrap(), e.item().unwrap())
        };
        let (a, b, c, d, e) = make(0.5);
        assert_eq!(e.item().unwrap().to_bits(), alone.4.to_bits(), "n = {n}");
        assert_eq!(c.item().unwrap().to_bits(), alone.2.to_bits(), "n = {n}");
        assert_eq!(d.item().unwrap().to_bits(), alone.3.to_bits(), "n = {n}");
        assert_eq!(b.to_vec().unwrap().iter().map(|v| v.to_bits()).collect::<Vec<_>>(), alone.1.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), "n = {n}");
        assert_eq!(a.to_vec().unwrap().iter().map(|v| v.to_bits()).collect::<Vec<_>>(), alone.0.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), "n = {n}");
        let min = host.iter().copied().fold(f32::INFINITY, f32::min);
        assert_eq!(a.to_vec().unwrap()[0], min);
    }
}

#[test]
fn eval_all_and_expressions_over_pending_reductions() {
    let ctx = Context::new().unwrap();
    let n = 100_000;
    let (_, x) = data(&ctx, n, 31);
    let (_, y) = data(&ctx, n, 32);
    let want_dot = (&x * &y).sum().item().unwrap();
    let want_xx = (&x * &x).sum().item().unwrap();
    let want_yy = (&y * &y).sum().item().unwrap();

    let (xy, xx, yy) = ((&x * &y).sum(), (&x * &x).sum(), (&y * &y).sum());
    luma::eval_all(&[&xy, &xx, &yy]).unwrap();
    assert_eq!(xy.item().unwrap().to_bits(), want_dot.to_bits());
    assert_eq!(xx.item().unwrap().to_bits(), want_xx.to_bits());
    assert_eq!(yy.item().unwrap().to_bits(), want_yy.to_bits());

    // The cosine similarity is one expression over three pending reductions.
    let (xy, xx, yy) = ((&x * &y).sum(), (&x * &x).sum(), (&y * &y).sum());
    let cosine = &xy / &(&xx.sqrt() * &yy.sqrt());
    let got = cosine.item().unwrap();
    let want = want_dot / (want_xx.sqrt() * want_yy.sqrt());
    assert!((got - want).abs() <= 1e-5 * want.abs().max(1.0), "{got} vs {want}");
}

#[test]
fn a_write_after_a_pending_reduction_does_not_change_it() {
    let ctx = Context::new().unwrap();
    let n = 4096;
    let (host, mut x) = data(&ctx, n, 41);
    let before = x.sum();
    let others: Vec<Array<f32>> = (0..4).map(|k| (&x * (k as f32 + 1.0)).sum()).collect();
    x.write(&vec![1.0f32; n]).unwrap();
    let want: f64 = host.iter().map(|&v| f64::from(v)).sum();
    assert!((f64::from(before.item().unwrap()) - want).abs() < 1e-1);
    for (k, other) in others.iter().enumerate() {
        let want = want * (k as f64 + 1.0);
        assert!((f64::from(other.item().unwrap()) - want).abs() < 1e-1 * (k as f64 + 1.0), "scale {}", k + 1);
    }
    assert_eq!(x.sum().item().unwrap(), n as f32);
}

#[test]
fn more_reductions_than_one_kernel_takes() {
    let ctx = Context::new().unwrap();
    let n = 50_000;
    let (_, x) = data(&ctx, n, 51);
    let make = || (0..40).map(|k| x.sum_where(&mask(&x, k))).collect::<Vec<_>>();
    let alone: Vec<u32> = make().iter().map(|r| r.item().unwrap().to_bits()).collect();
    let together = make();
    for (k, r) in together.iter().enumerate().rev() {
        assert_eq!(r.item().unwrap().to_bits(), alone[k], "reduction {k}");
    }
}

#[test]
fn reductions_made_by_other_threads_and_dropped_ones_are_left_alone() {
    let ctx = Context::new().unwrap();
    let (_, x) = data(&ctx, 20_000, 61);
    let dropped: Vec<Array<f32>> = (0..5).map(|k| (&x + k as f32).sum()).collect();
    drop(dropped);
    let keep = x.sum();
    let other = {
        let x = x.clone();
        std::thread::spawn(move || (&x * 3.0).sum())
    }
    .join()
    .unwrap();
    let alone = x.sum().item().unwrap();
    assert_eq!(keep.item().unwrap().to_bits(), alone.to_bits());
    assert!((other.item().unwrap() - 3.0 * alone).abs() <= 1e-2 * alone.abs().max(1.0));
}

#[test]
fn reductions_in_traced_steps_and_pending_ones_beside_them() {
    let ctx = Context::new().unwrap();
    let n = 3000;
    let (host, x) = data(&ctx, n, 81);
    let (_, s0) = data(&ctx, n, 82);
    // Pending outside the loop: the steps' reductions never take them along, and they are the values they have alone.
    let outside: Vec<Array<f32>> = (0..4).map(|k| (&x * (k as f32 + 1.0)).sum()).collect();
    let total = x.sum();
    let looped = luma::iterate_all([&s0], 5, |s| {
        let norm = (&s[0] * &s[0]).sum();
        let mean = s[0].mean();
        [&s[0] * 0.5 + &norm * 1e-6 + &mean * 1e-3]
    });
    let mut want = s0.to_vec().unwrap();
    for _ in 0..5 {
        let norm: f32 = want.iter().map(|v| v * v).sum();
        let mean = want.iter().sum::<f32>() / n as f32;
        want = want.iter().map(|v| v * 0.5 + norm * 1e-6 + mean * 1e-3).collect();
    }
    let got = looped[0].to_vec().unwrap();
    for (g, w) in got.iter().zip(&want) {
        assert!((g - w).abs() <= 1e-3 * w.abs().max(1.0), "{g} vs {w}");
    }
    let sum: f64 = host.iter().map(|&v| f64::from(v)).sum();
    assert!((f64::from(total.item().unwrap()) - sum).abs() < 1e-1);
    for (k, r) in outside.iter().enumerate() {
        assert!((f64::from(r.item().unwrap()) - sum * (k as f64 + 1.0)).abs() < 1e-1 * (k as f64 + 1.0));
    }
}

#[test]
fn empty_arrays_and_one_element() {
    let ctx = Context::new().unwrap();
    // An empty array is an error or a neutral value, the same alone and together with others.
    let empty = Array::<f32>::zeros_on(&ctx, 0);
    if let Ok(empty) = empty {
        let alone = (empty.sum().item(), empty.max().item());
        let together = (empty.sum(), empty.max(), (&empty + 1.0).sum());
        assert_eq!(format!("{:?}", together.0.item()), format!("{:?}", alone.0));
        assert_eq!(format!("{:?}", together.1.item()), format!("{:?}", alone.1));
        let _ = together.2.item();
    }
    let one = Array::from_slice_on(&ctx, &[3.5f32], 1).unwrap();
    let (a, b, c, d) = (one.sum(), one.min(), one.max(), (&one * 2.0).sum());
    assert_eq!([c.item().unwrap(), a.item().unwrap(), b.item().unwrap(), d.item().unwrap()], [3.5, 3.5, 3.5, 7.0]);
}
