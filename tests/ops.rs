mod common;

use common::{assert_close, values};
use luma::*;

fn f32s(data: &[f32]) -> Array<f32> {
    Array::from_slice(data, data.len()).unwrap()
}

fn u32s(data: &[u32]) -> Array<u32> {
    Array::from_slice(data, data.len()).unwrap()
}

// -- Operators vs the CPU --

#[test]
fn f32_expressions_match_cpu() {
    let n = 10_007; // not a multiple of the workgroup size
    let xs = values(n, 1, -4.0, 4.0);
    let ys = values(n, 2, 0.5, 3.0); // away from zero: used as divisor
    let a = f32s(&xs);
    let b = f32s(&ys);

    // Array op array, all four operators, with owned and borrowed operands mixed.
    let got = ((&a + &b) * &a - &b / &a.abs().maximum(0.1)).to_vec().unwrap();
    let want: Vec<f32> = xs
        .iter()
        .zip(&ys)
        .map(|(x, y)| (x + y) * x - y / x.abs().max(0.1))
        .collect();
    assert_close(&got, &want, 1e-5, 1e-5);

    // Scalar on both sides.
    let got = (2.0 - &a).to_vec().unwrap();
    assert_close(&got, &xs.iter().map(|x| 2.0 - x).collect::<Vec<_>>(), 1e-6, 1e-6);
    let got = (1.0 / &b).to_vec().unwrap();
    assert_close(&got, &ys.iter().map(|y| 1.0 / y).collect::<Vec<_>>(), 1e-5, 1e-6);
    let got = (3.0 * &a + 0.5).to_vec().unwrap();
    assert_close(&got, &xs.iter().map(|x| 3.0 * x + 0.5).collect::<Vec<_>>(), 1e-5, 1e-5);
    let got = (&a / 4.0 - 1.0).to_vec().unwrap();
    assert_close(&got, &xs.iter().map(|x| x / 4.0 - 1.0).collect::<Vec<_>>(), 1e-5, 1e-6);
    let got = (10.0 + f32s(&xs)).to_vec().unwrap(); // owned scalar-lhs form
    assert_close(&got, &xs.iter().map(|x| 10.0 + x).collect::<Vec<_>>(), 1e-6, 1e-5);

    // Owned/borrowed combinations produce the same thing.
    let both_owned = (f32s(&xs) + f32s(&ys)).to_vec().unwrap();
    let lhs_owned = (f32s(&xs) + &b).to_vec().unwrap();
    let rhs_owned = (&a + f32s(&ys)).to_vec().unwrap();
    assert_eq!(both_owned, lhs_owned);
    assert_eq!(both_owned, rhs_owned);
}

#[test]
fn f32_negation() {
    let xs = values(1000, 3, -5.0, 5.0);
    let a = f32s(&xs);
    let want: Vec<f32> = xs.iter().map(|x| -x).collect();
    assert_eq!((-&a).to_vec().unwrap(), want);
    assert_eq!((-a).to_vec().unwrap(), want);
}

#[test]
fn compound_assignment_matches_cpu() {
    let n = 5_001;
    let xs = values(n, 4, -2.0, 2.0);
    let ys = values(n, 5, 0.5, 2.0);
    let mut x = f32s(&xs);
    let y = f32s(&ys);

    x += &y;
    x -= 0.25;
    x *= &y;
    x /= 2.0;
    x += 1.5;
    x -= &y;
    x *= 3.0;
    x /= &y;

    let want: Vec<f32> = xs
        .iter()
        .zip(&ys)
        .map(|(x, y)| ((((((x + y) - 0.25) * y) / 2.0 + 1.5) - y) * 3.0) / y)
        .collect();
    assert_close(&x.to_vec().unwrap(), &want, 1e-4, 1e-4);
    // The right-hand side is untouched.
    assert_eq!(y.to_vec().unwrap(), ys);
}

#[test]
fn u32_operators_and_compound_assignment() {
    let xs: Vec<u32> = (1..=2000).collect();
    let ys: Vec<u32> = (0..2000).map(|i| i % 7 + 1).collect();
    let a = u32s(&xs);
    let b = u32s(&ys);

    let got = ((&a + &b) * 3 - &b / 2u32 + 100 / &b).to_vec().unwrap();
    let want: Vec<u32> = xs
        .iter()
        .zip(&ys)
        .map(|(x, y)| (x + y) * 3 - y / 2 + 100 / y)
        .collect();
    assert_eq!(got, want);

    assert_eq!((1_000_000 - &a).to_vec().unwrap(), xs.iter().map(|x| 1_000_000 - x).collect::<Vec<_>>());

    let mut c = u32s(&xs);
    c += &b;
    c *= 2;
    c -= 1;
    c /= &b;
    c += 7;
    let want: Vec<u32> = xs
        .iter()
        .zip(&ys)
        .map(|(x, y)| ((x + y) * 2 - 1) / y + 7)
        .collect();
    assert_eq!(c.to_vec().unwrap(), want);
}

// -- Math functions vs the CPU --

#[test]
fn math_functions_match_cpu() {
    let n = 4_099;
    let signed = values(n, 6, -3.0, 3.0);
    let positive = values(n, 7, 0.1, 10.0);
    let small = values(n, 8, -1.2, 1.2);
    let exps = values(n, 9, -5.0, 5.0);

    let check = |input: &[f32], got: Array<f32>, f: fn(f32) -> f32, rel: f32, abs: f32| {
        let want: Vec<f32> = input.iter().map(|&x| f(x)).collect();
        assert_close(&got.to_vec().unwrap(), &want, rel, abs);
    };
    check(&positive, f32s(&positive).sqrt(), f32::sqrt, 1e-5, 0.0);
    check(&exps, f32s(&exps).exp(), f32::exp, 1e-4, 0.0);
    check(&positive, f32s(&positive).ln(), f32::ln, 1e-4, 1e-5);
    check(&signed, f32s(&signed).abs(), f32::abs, 0.0, 0.0);
    check(&signed, f32s(&signed).sin(), f32::sin, 1e-4, 1e-5);
    check(&signed, f32s(&signed).cos(), f32::cos, 1e-4, 1e-5);
    check(&small, f32s(&small).tan(), f32::tan, 1e-4, 1e-5);

    let got = f32s(&positive).powf(2.5).to_vec().unwrap();
    let want: Vec<f32> = positive.iter().map(|x| x.powf(2.5)).collect();
    assert_close(&got, &want, 1e-4, 0.0);

    let got = f32s(&signed).clamp(-1.0, 0.5).to_vec().unwrap();
    let want: Vec<f32> = signed.iter().map(|x| x.clamp(-1.0, 0.5)).collect();
    assert_eq!(got, want);

    let other = values(n, 10, -3.0, 3.0);
    let got = f32s(&signed).minimum(f32s(&other)).to_vec().unwrap();
    let want: Vec<f32> = signed.iter().zip(&other).map(|(a, b)| a.min(*b)).collect();
    assert_eq!(got, want);
    let got = f32s(&signed).maximum(f32s(&other)).to_vec().unwrap();
    let want: Vec<f32> = signed.iter().zip(&other).map(|(a, b)| a.max(*b)).collect();
    assert_eq!(got, want);
    assert_eq!(
        f32s(&signed).maximum(0.0).to_vec().unwrap(),
        signed.iter().map(|x| x.max(0.0)).collect::<Vec<_>>()
    );

    // A chain of math functions stays on the GPU.
    let got = f32s(&positive).sqrt().ln().exp().to_vec().unwrap();
    let want: Vec<f32> = positive.iter().map(|x| x.sqrt().ln().exp()).collect();
    assert_close(&got, &want, 1e-4, 1e-5);
}

#[test]
fn u32_min_max() {
    let a = u32s(&[5, 1, 9, 4]);
    let b = u32s(&[3, 3, 3, 3]);
    assert_eq!(a.minimum(&b).to_vec().unwrap(), vec![3, 1, 3, 3]);
    assert_eq!(a.maximum(&b).to_vec().unwrap(), vec![5, 3, 9, 4]);
    assert_eq!(a.maximum(4u32).to_vec().unwrap(), vec![5, 4, 9, 4]);
}

// -- Comparisons and select --

#[test]
fn comparisons_produce_masks() {
    let xs = [1.0f32, 2.0, 3.0, 4.0, 5.0, f32::from_bits(0x4040_0000)]; // last one is 3.0
    let a = f32s(&xs);
    let b = f32s(&[5.0, 2.0, 1.0, 4.0, 0.0, 3.0]);

    let mask = |m: Array<u32>| m.to_vec().unwrap();
    assert_eq!(mask(a.lt(&b)), vec![1, 0, 0, 0, 0, 0]);
    assert_eq!(mask(a.le(&b)), vec![1, 1, 0, 1, 0, 1]);
    assert_eq!(mask(a.gt(&b)), vec![0, 0, 1, 0, 1, 0]);
    assert_eq!(mask(a.ge(&b)), vec![0, 1, 1, 1, 1, 1]);
    assert_eq!(mask(a.eq(&b)), vec![0, 1, 0, 1, 0, 1]);
    assert_eq!(mask(a.ne(&b)), vec![1, 0, 1, 0, 1, 0]);

    // Scalar right-hand sides.
    assert_eq!(mask(a.gt(2.5)), vec![0, 0, 1, 1, 1, 1]);
    assert_eq!(mask(a.eq(3.0)), vec![0, 0, 1, 0, 0, 1]);

    let u = u32s(&[1, 5, 9]);
    assert_eq!(mask(u.lt(5u32)), vec![1, 0, 0]);
    assert_eq!(mask(u.ge(u32s(&[2, 5, 8]))), vec![0, 1, 1]);
}

#[test]
fn select_picks_per_element() {
    let n = 3_000;
    let xs = values(n, 11, -1.0, 1.0);
    let x = f32s(&xs);

    // relu via select
    let relu = select(&x.gt(0.0), &x, Array::zeros(n).unwrap());
    let want: Vec<f32> = xs.iter().map(|v| v.max(0.0)).collect();
    assert_eq!(relu.to_vec().unwrap(), want);

    // Mask arithmetic: x.lt(0) as u32 mask selects between two arrays.
    let a = Array::full(n, 7u32).unwrap();
    let b = Array::full(n, 9u32).unwrap();
    let picked = select(&x.lt(0.0), &a, &b).to_vec().unwrap();
    let want: Vec<u32> = xs.iter().map(|v| if *v < 0.0 { 7 } else { 9 }).collect();
    assert_eq!(picked, want);

    // Shape mismatches poison the result.
    let short = Array::full(5, 1.0f32).unwrap();
    let err = select(&x.gt(0.0), &x, &short).to_vec().unwrap_err();
    assert!(matches!(err.root(), LumaError::ShapeMismatch { .. }), "{err}");
}

// -- Creation --

#[test]
fn full_fills_every_element() {
    let a = Array::full([5, 4], 2.5f32).unwrap();
    assert_eq!(a.shape(), [5, 4, 1, 1]);
    assert_eq!(a.to_vec().unwrap(), vec![2.5; 20]);
    let b = Array::full(100_001, 9u32).unwrap();
    assert_eq!(b.to_vec().unwrap(), vec![9u32; 100_001]);
}

// -- Shift --

/// `af::shift` on the host: element at `c` moves to `c + shift` in every dimension, circularly.
fn shift_cpu<T: Copy>(data: &[T], dims: [usize; 4], shift: [isize; 4]) -> Vec<T> {
    let mut out = data.to_vec();
    for (i, slot) in out.iter_mut().enumerate() {
        let coords = [
            i % dims[0],
            (i / dims[0]) % dims[1],
            (i / (dims[0] * dims[1])) % dims[2],
            i / (dims[0] * dims[1] * dims[2]),
        ];
        let src: Vec<usize> = (0..4)
            .map(|d| (coords[d] as isize - shift[d]).rem_euclid(dims[d] as isize) as usize)
            .collect();
        *slot = data[src[0] + dims[0] * (src[1] + dims[1] * (src[2] + dims[2] * src[3]))];
    }
    out
}

#[test]
fn shift_1d() {
    let a = f32s(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    assert_eq!(a.shift(1).to_vec().unwrap(), vec![5.0, 1.0, 2.0, 3.0, 4.0]);
    assert_eq!(a.shift(-1).to_vec().unwrap(), vec![2.0, 3.0, 4.0, 5.0, 1.0]);
    assert_eq!(a.shift(0).to_vec().unwrap(), vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    assert_eq!(a.shift(7).to_vec().unwrap(), a.shift(2).to_vec().unwrap());
    assert_eq!(a.shift(-12).to_vec().unwrap(), a.shift(-2).to_vec().unwrap());
    assert_eq!(a.shift([2]).to_vec().unwrap(), vec![4.0, 5.0, 1.0, 2.0, 3.0]);
    assert_eq!(u32s(&[1, 2, 3]).shift(1isize).to_vec().unwrap(), vec![3, 1, 2]);
}

#[test]
fn shift_2d_and_3d_match_cpu() {
    let (nx, ny) = (4usize, 3usize);
    let data: Vec<f32> = (0..nx * ny).map(|i| i as f32).collect();
    let a = Array::from_slice(&data, [nx, ny]).unwrap();
    for shift in [[1isize, 0], [0, 1], [-1, 1], [1, -1], [5, 2], [-7, -4]] {
        let got = a.shift(shift).to_vec().unwrap();
        let want = shift_cpu(&data, [nx, ny, 1, 1], [shift[0], shift[1], 0, 0]);
        assert_eq!(got, want, "shift {shift:?}");
    }

    let dims = [3usize, 4, 2];
    let data: Vec<u32> = (0..24).collect();
    let a = Array::from_slice(&data, dims).unwrap();
    let got = a.shift([1isize, -1, 1]).to_vec().unwrap();
    assert_eq!(got, shift_cpu(&data, [3, 4, 2, 1], [1, -1, 1, 0]));

    // Large non-square array.
    let (nx, ny) = (301usize, 257usize);
    let data = values(nx * ny, 12, 0.0, 1.0);
    let a = Array::from_slice(&data, [nx, ny]).unwrap();
    let got = a.shift([17isize, -100]).to_vec().unwrap();
    assert_eq!(got, shift_cpu(&data, [nx, ny, 1, 1], [17, -100, 0, 0]));
}

#[test]
fn shift_builds_a_stencil() {
    // 1D three-point average with periodic boundaries, written with shifts.
    let xs = values(1000, 13, 0.0, 1.0);
    let x = f32s(&xs);
    let smoothed = ((x.shift(1) + &x + x.shift(-1)) / 3.0).to_vec().unwrap();
    let n = xs.len();
    let want: Vec<f32> = (0..n)
        .map(|i| (xs[(i + n - 1) % n] + xs[i] + xs[(i + 1) % n]) / 3.0)
        .collect();
    assert_close(&smoothed, &want, 1e-5, 1e-6);
}

// -- Reductions --

#[test]
fn reductions_on_small_and_odd_sizes() {
    for n in [1usize, 2, 3, 63, 64, 65, 255, 256, 257, 1023, 1024, 1025, 4097, 100_003] {
        let xs = values(n, n as u64, -10.0, 10.0);
        let a = f32s(&xs);
        let want_sum: f64 = xs.iter().map(|&x| f64::from(x)).sum();
        let got = f64::from(a.sum().item().unwrap());
        let scale: f64 = xs.iter().map(|&x| f64::from(x).abs()).sum();
        assert!((got - want_sum).abs() <= 1e-6 * scale + 1e-6, "sum n={n}: got {got}, want {want_sum}");

        assert_eq!(a.min().item().unwrap(), xs.iter().copied().fold(f32::INFINITY, f32::min), "min n={n}");
        assert_eq!(a.max().item().unwrap(), xs.iter().copied().fold(f32::NEG_INFINITY, f32::max), "max n={n}");

        let mean = f64::from(a.mean().item().unwrap());
        assert!((mean - want_sum / n as f64).abs() <= 1e-6 * scale / n as f64 + 1e-6, "mean n={n}");

        let want_norm = xs.iter().map(|&x| f64::from(x) * f64::from(x)).sum::<f64>().sqrt();
        let norm = f64::from(a.norm().item().unwrap());
        assert!((norm - want_norm).abs() <= 1e-5 * want_norm + 1e-6, "norm n={n}: {norm} vs {want_norm}");
    }
}

#[test]
fn u32_reductions() {
    for n in [1usize, 7, 1025, 5000] {
        let xs: Vec<u32> = (0..n as u32).map(|i| i.wrapping_mul(2654435761u32) % 1000 + 3).collect();
        let a = u32s(&xs);
        assert_eq!(a.sum().item().unwrap(), xs.iter().fold(0u32, |s, x| s.wrapping_add(*x)), "n={n}");
        assert_eq!(a.min().item().unwrap(), *xs.iter().min().unwrap(), "n={n}");
        assert_eq!(a.max().item().unwrap(), *xs.iter().max().unwrap(), "n={n}");
    }
}

#[test]
fn u32_reductions_beyond_one_block_per_workgroup() {
    // Past the workgroup cap of a pass, so workgroups stride over several blocks each.
    let n = 6_000_011usize;
    let mut xs: Vec<u32> = (0..n as u32).map(|i| i.wrapping_mul(2654435761u32) % 1000 + 3).collect();
    xs[n - 2] = 1;
    xs[n / 2 + 1] = 5000;
    let a = u32s(&xs);
    assert_eq!(a.sum().item().unwrap(), xs.iter().fold(0u32, |s, x| s.wrapping_add(*x)));
    assert_eq!(a.min().item().unwrap(), 1);
    assert_eq!(a.max().item().unwrap(), 5000);
}

#[test]
fn reductions_ignore_stale_pool_contents() {
    // Fresh context so the pooled buffers of the first reduction are the ones the second reuses.
    let ctx = Context::new().unwrap();
    let big = Array::from_slice_on(&ctx, &vec![1000.0f32; 50_000], 50_000).unwrap();
    assert_eq!(big.sum().item().unwrap(), 50_000_000.0);
    drop(big);
    let small = Array::from_slice_on(&ctx, &[1.0f32, 2.0, 3.0], 3).unwrap();
    assert_eq!(small.sum().item().unwrap(), 6.0);
    assert_eq!(small.min().item().unwrap(), 1.0);
    assert_eq!(small.max().item().unwrap(), 3.0);
}

#[test]
fn f32_sum_of_ten_million_elements_is_accurate() {
    let n = 10_000_000usize;
    let xs = values(n, 99, 0.0, 1.0);
    let want: f64 = xs.iter().map(|&x| f64::from(x)).sum();
    let a = f32s(&xs);

    let got = f64::from(a.sum().item().unwrap());
    let relative = ((got - want) / want).abs();
    assert!(relative < 2e-6, "sum of {n} elements: got {got}, want {want}, relative error {relative}");

    // A naive sequential f32 sum on the CPU is a lot worse than that; the tree reduction is not.
    let naive: f32 = xs.iter().sum();
    let naive_error = ((f64::from(naive) - want) / want).abs();
    assert!(relative <= naive_error.max(2e-6), "tree {relative} vs naive {naive_error}");

    let mean = f64::from(a.mean().item().unwrap());
    assert!(((mean - want / n as f64) / (want / n as f64)).abs() < 2e-6);
    let want_norm = xs.iter().map(|&x| f64::from(x) * f64::from(x)).sum::<f64>().sqrt();
    let norm = f64::from(a.norm().item().unwrap());
    assert!(((norm - want_norm) / want_norm).abs() < 2e-6, "norm: {norm} vs {want_norm}");
    assert!(a.min().item().unwrap() >= 0.0 && a.max().item().unwrap() < 1.0);
}

#[test]
fn reduction_of_an_expression() {
    // Dot product through the operators: sum(a * b).
    let n = 100_000;
    let xs = values(n, 20, -1.0, 1.0);
    let ys = values(n, 21, -1.0, 1.0);
    let dot = (&f32s(&xs) * &f32s(&ys)).sum().item().unwrap();
    let want: f64 = xs.iter().zip(&ys).map(|(x, y)| f64::from(*x) * f64::from(*y)).sum();
    assert!((f64::from(dot) - want).abs() < 1e-3, "{dot} vs {want}");
}

// -- reshape --

#[test]
fn reshape_shares_the_buffer_and_keeps_data() {
    let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
    let a = Array::from_slice(&data, 12).unwrap();
    let b = a.reshape([3, 4]);
    let c = b.reshape([2, 2, 3]);
    assert_eq!(b.shape(), [3, 4, 1, 1]);
    assert_eq!(c.shape(), [2, 2, 3, 1]);
    assert_eq!(b.to_vec().unwrap(), data);
    assert_eq!(c.to_vec().unwrap(), data);

    // Shared storage: writing through one array shows in the others.
    let mut a = a;
    a.write(&[1.0; 12]).unwrap();
    assert_eq!(b.to_vec().unwrap(), vec![1.0; 12]);
    assert_eq!(c.to_vec().unwrap(), vec![1.0; 12]);

    // ... and in-place operations too.
    let mut d = c;
    d += 2.0;
    assert_eq!(a.to_vec().unwrap(), vec![3.0; 12]);
    assert_eq!(b.to_vec().unwrap(), vec![3.0; 12]);

    // Math on a reshaped array sees its new shape.
    let flat = Array::from_slice(&data, 12).unwrap();
    let grid = flat.reshape([3, 4]);
    assert_eq!((&grid + &grid).shape(), [3, 4, 1, 1]);
    assert_eq!(grid.shift([1isize, 0]).to_vec().unwrap(), shift_cpu(&data, [3, 4, 1, 1], [1, 0, 0, 0]));
}

#[test]
fn u32_division_by_zero_gives_the_dividend() {
    // WGSL defines `x / 0u` as `x`; fused and in-place kernels must keep that.
    let a = u32s(&[7, 8, 9, 10]);
    let b = u32s(&[0, 2, 0, 5]);
    assert_eq!((&a / &b).to_vec().unwrap(), vec![7, 4, 9, 2]);
    assert_eq!((&a / 0u32).to_vec().unwrap(), vec![7, 8, 9, 10]);
    let mut c = u32s(&[7, 8, 9, 10]);
    c /= &b;
    assert_eq!(c.to_vec().unwrap(), vec![7, 4, 9, 2]);
}

// -- Broadcasting along size-1 axes --

fn col_and_row() -> (Array<f32>, Array<f32>) {
    let col = Array::from_slice(&[1.0f32, 2.0, 3.0], [3, 1]).unwrap();
    let row = Array::from_slice(&[10.0f32, 20.0], [1, 2]).unwrap();
    (col, row)
}

#[test]
fn size_one_axes_broadcast_in_operators() {
    let (col, row) = col_and_row();
    let sum = &col + &row;
    assert_eq!(sum.shape().dims(), [3, 2, 1, 1]);
    assert_eq!(sum.to_vec().unwrap(), [11.0, 12.0, 13.0, 21.0, 22.0, 23.0]);
    assert_eq!((&row - &col).to_vec().unwrap(), [9.0, 8.0, 7.0, 19.0, 18.0, 17.0]);
    assert_eq!((&col * &row).to_vec().unwrap(), [10.0, 20.0, 30.0, 20.0, 40.0, 60.0]);

    // A one-element array broadcasts to every shape, on either side.
    let one = Array::from_slice(&[2.0f32], 1).unwrap();
    assert_eq!((&one * &col).to_vec().unwrap(), [2.0, 4.0, 6.0]);
    let both = &sum + &one;
    assert_eq!(both.shape().dims(), [3, 2, 1, 1]);
    assert_eq!(both.to_vec().unwrap(), [13.0, 14.0, 15.0, 23.0, 24.0, 25.0]);

    // Other extents still have to agree.
    let bad = &sum + &Array::<f32>::zeros([2, 3]).unwrap();
    let err = bad.to_vec().unwrap_err();
    assert!(matches!(err.root(), LumaError::ShapeMismatch { lhs: [3, 2, 1, 1], rhs: [2, 3, 1, 1] }), "{err}");
}

#[test]
fn comparisons_and_select_broadcast() {
    let (col, _) = col_and_row();
    let thr = Array::from_slice(&[1.5f32, 2.5], [1, 2]).unwrap();
    let mask = col.gt(&thr);
    assert_eq!(mask.shape().dims(), [3, 2, 1, 1]);
    assert_eq!(mask.to_vec().unwrap(), [0, 1, 1, 0, 0, 1]);

    // An `[n, 1]` branch under an `[n, m]` mask is read at every column.
    let picked = select(&mask, &col, 0.0);
    assert_eq!(picked.to_vec().unwrap(), [0.0, 2.0, 3.0, 0.0, 0.0, 3.0]);
    let picked = select(&mask, 1.0f32, &thr);
    assert_eq!(picked.to_vec().unwrap(), [1.5, 1.0, 1.0, 2.5, 2.5, 1.0]);

    // The mask's shape is the result's: a branch cannot grow it.
    let wide = Array::<f32>::zeros([3, 2]).unwrap();
    let err = select(&col.gt(1.0), &wide, 0.0).to_vec().unwrap_err();
    assert!(matches!(err.root(), LumaError::ShapeMismatch { lhs: [3, 1, 1, 1], rhs: [3, 2, 1, 1] }), "{err}");
}

#[test]
fn compound_assignment_broadcasts_the_right_hand_side_only() {
    let (col, row) = col_and_row();
    let mut m = Array::<f32>::zeros([3, 2]).unwrap();
    m += &row;
    assert_eq!(m.to_vec().unwrap(), [10.0, 10.0, 10.0, 20.0, 20.0, 20.0]);
    m += &col;
    assert_eq!(m.to_vec().unwrap(), [11.0, 12.0, 13.0, 21.0, 22.0, 23.0]);

    // The target never grows.
    let mut c = col.copy();
    c += &row;
    let err = c.to_vec().unwrap_err();
    assert!(matches!(err.root(), LumaError::ShapeMismatch { .. }), "{err}");
}

#[test]
fn broadcast_to_repeats_size_one_axes() {
    let (col, row) = col_and_row();
    let b = col.broadcast_to([3, 2]);
    assert_eq!(b.shape().dims(), [3, 2, 1, 1]);
    assert_eq!(b.to_vec().unwrap(), [1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
    assert_eq!(row.broadcast_to([3, 2]).to_vec().unwrap(), [10.0, 10.0, 10.0, 20.0, 20.0, 20.0]);
    assert_eq!(col.broadcast_to([3, 1]).to_vec().unwrap(), [1.0, 2.0, 3.0]);

    // It is an ordinary lazy array: it fuses and is not a view that writes through.
    let outer = col.broadcast_to([3, 2]) * &row.broadcast_to([3, 2]) + 1.0;
    assert_eq!(outer.to_vec().unwrap(), [11.0, 21.0, 31.0, 21.0, 41.0, 61.0]);

    let err = col.broadcast_to([4, 2]).to_vec().unwrap_err();
    assert!(matches!(err.root(), LumaError::ShapeMismatch { lhs: [3, 1, 1, 1], rhs: [4, 2, 1, 1] }), "{err}");
}
