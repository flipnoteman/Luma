//! Scans and reductions along an axis.

mod common;

use common::{assert_close, values};
use luma::{Array, Context, LumaError};

#[test]
fn cumsum_along_each_axis() {
    let ctx = Context::new().unwrap();
    let (w, h) = (50usize, 9usize);
    let a = values(w * h, 1, -1.0, 1.0);
    let x = Array::from_slice_on(&ctx, &a, [w, h]).unwrap();

    let mut want = a.clone();
    for y in 0..h {
        for i in 1..w {
            want[i + w * y] += want[i - 1 + w * y];
        }
    }
    assert_close(&x.cumsum(0).to_vec().unwrap(), &want, 1e-4, 1e-5);

    let mut want = a.clone();
    for y in 1..h {
        for i in 0..w {
            want[i + w * y] += want[i + w * (y - 1)];
        }
    }
    assert_close(&x.cumsum(1).to_vec().unwrap(), &want, 1e-4, 1e-5);
}

#[test]
fn scan_with_a_custom_operator() {
    let ctx = Context::new().unwrap();
    let a = values(50 * 9, 2, -1.0, 1.0);
    let x = Array::from_slice_on(&ctx, &a, [50, 9]).unwrap();
    let got = x.scan(0, |x, y| x.maximum(y)).to_vec().unwrap();
    let mut want = a.clone();
    for y in 0..9 {
        for i in 1..50 {
            want[i + 50 * y] = want[i + 50 * y].max(want[i - 1 + 50 * y]);
        }
    }
    assert_eq!(got, want);

    // Scalars are uniforms: `x + y + 0.25` is associative (`(x + y + c) + z + c == x + (y + z + c) + c`).
    let got = x.slice(1, ..1).scan(0, |x, y| &(x + y) + 0.25).to_vec().unwrap();
    let mut want = a[..50].to_vec();
    for i in 1..50 {
        want[i] += want[i - 1] + 0.25;
    }
    assert_close(&got, &want, 1e-4, 1e-5);
}

#[test]
fn long_lines_scan_in_blocks() {
    let ctx = Context::new().unwrap();
    for n in [5000usize, 1_100_000] {
        let x = Array::<f32>::full_on(&ctx, n, 1.0).unwrap();
        let got = x.cumsum(0).to_vec().unwrap();
        assert!(got.iter().enumerate().all(|(i, &v)| v == (i + 1) as f32), "n = {n}");
    }
    let x = Array::<u32>::full_on(&ctx, 3000, 2).unwrap();
    let got = x.cumsum(0).to_vec().unwrap();
    assert!(got.iter().enumerate().all(|(i, &v)| v == 2 * (i as u32 + 1)));

    // Several long lines: the lines run along axis 1 with stride 3 and along axis 0.
    let a = values(3 * 2500, 3, 0.0, 1.0);
    let x = Array::from_slice_on(&ctx, &a, [3, 2500]).unwrap();
    let got = x.cumsum(1).to_vec().unwrap();
    let mut want = a.clone();
    for y in 1..2500 {
        for i in 0..3 {
            want[i + 3 * y] += want[i + 3 * (y - 1)];
        }
    }
    assert_close(&got, &want, 1e-4, 1e-5);
}

#[test]
fn reductions_along_an_axis() {
    let ctx = Context::new().unwrap();
    let (w, h) = (6usize, 2000usize);
    let a = values(w * h, 4, -1.0, 1.0);
    let x = Array::from_slice_on(&ctx, &a, [w, h]).unwrap();

    let s = x.sum_axis(1);
    assert_eq!(s.shape().dims(), [6, 1, 1, 1]);
    let want: Vec<f32> = (0..w).map(|i| (0..h).map(|y| f64::from(a[i + w * y])).sum::<f64>() as f32).collect();
    assert_close(&s.to_vec().unwrap(), &want, 1e-3, 1e-3);

    let m = x.min_axis(0);
    assert_eq!(m.shape().dims(), [1, 2000, 1, 1]);
    let want: Vec<f32> = (0..h).map(|y| (0..w).map(|i| a[i + w * y]).fold(f32::INFINITY, f32::min)).collect();
    assert_eq!(m.to_vec().unwrap(), want);

    let m = x.max_axis(0);
    let want: Vec<f32> = (0..h).map(|y| (0..w).map(|i| a[i + w * y]).fold(f32::NEG_INFINITY, f32::max)).collect();
    assert_eq!(m.to_vec().unwrap(), want);

    // Lazy input and a reduction that uses a lazy result.
    let lazy = (&x * 2.0).sum_axis(0);
    let want: Vec<f32> = (0..h).map(|y| (0..w).map(|i| a[i + w * y] * 2.0).sum()).collect();
    assert_close(&lazy.to_vec().unwrap(), &want, 1e-4, 1e-5);
}

#[test]
fn operators_must_be_elementwise() {
    let ctx = Context::new().unwrap();
    let x = Array::from_slice_on(&ctx, &values(16, 5, 0.0, 1.0), 16).unwrap();
    let err = x.scan(0, |x, y| &x.shift(1) + y).to_vec().unwrap_err();
    assert!(matches!(err.root(), LumaError::InvalidShape(_)), "{err}");
    let err = x.scan(0, |_, _| Array::<f32>::full_on(&ctx, 1, 0.0).unwrap()).to_vec().unwrap_err();
    assert!(matches!(err.root(), LumaError::InvalidShape(_)), "{err}");
    let err = x.cumsum(4).to_vec().unwrap_err();
    assert!(matches!(err.root(), LumaError::InvalidShape(_)), "{err}");
}

/// `u32` data in `0..1000` from a linear congruential generator.
fn counts(n: usize, seed: u32) -> Vec<u32> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 16) % 1000
        })
        .collect()
}

/// Folds every line of a `dims`-shaped array along `axis`: `running` keeps all prefixes, otherwise only the total.
fn host_lines(dims: [usize; 4], axis: usize, data: &[u32], running: bool, f: impl Fn(u32, u32) -> u32) -> Vec<u32> {
    let stride: usize = dims[..axis].iter().product();
    let n = dims[axis];
    let outer: usize = dims[axis + 1..].iter().product();
    let mut out = if running { data.to_vec() } else { vec![0; stride * outer] };
    for o in 0..outer {
        for i in 0..stride {
            let at = |s: usize| i + stride * (s + n * o);
            let mut acc = data[at(0)];
            if running {
                out[at(0)] = acc;
            }
            for s in 1..n {
                acc = f(acc, data[at(s)]);
                if running {
                    out[at(s)] = acc;
                }
            }
            if !running {
                out[i + stride * o] = acc;
            }
        }
    }
    out
}

fn shape4(dims: &[usize]) -> [usize; 4] {
    let mut d = [1; 4];
    d[..dims.len()].copy_from_slice(dims);
    d
}

/// Shapes whose lines end around the boundaries of the scan kernels: short lines, one chunk of a
/// workgroup, several chunks, several levels of totals; strided lines walked whole or in blocks.
const LINE_SHAPES: &[(&[usize], usize)] = &[
    (&[31], 0), (&[32], 0), (&[33], 0), (&[127], 0), (&[128], 0), (&[129], 0),
    (&[1023], 0), (&[1024], 0), (&[1025], 0), (&[2047, 3], 0), (&[2048, 3], 0), (&[2049, 3], 0),
    (&[1_048_575], 0), (&[1_048_576], 0), (&[1_048_577], 0),
    (&[33, 5], 0), (&[1025, 4], 0), (&[1, 1025], 1),
    (&[3, 255], 1), (&[3, 256], 1), (&[3, 257], 1), (&[2, 271], 1), (&[2, 272], 1), (&[2, 273], 1),
    (&[3, 5000], 1), (&[40, 70_000], 1), (&[3, 300, 2], 1), (&[2, 3, 1100], 2), (&[5, 4, 300, 2], 2),
];

#[test]
fn scans_are_exact_around_block_boundaries() {
    let ctx = Context::new().unwrap();
    for &(dims, axis) in LINE_SHAPES {
        let d = shape4(dims);
        let len: usize = dims.iter().product();
        let data = counts(len, len as u32);
        let x = Array::from_slice_on(&ctx, &data, d).unwrap();
        let what = format!("{dims:?} axis {axis}");
        let want = host_lines(d, axis, &data, true, u32::wrapping_add);
        assert_eq!(x.cumsum(axis).to_vec().unwrap(), want, "cumsum {what}");
        // The operators are not commutative: every block must combine its prefix on the left.
        let want = host_lines(d, axis, &data, true, |_, y| y);
        assert_eq!(x.scan(axis, |_, y| y * 1u32).to_vec().unwrap(), want, "last {what}");
        let want = host_lines(d, axis, &data, true, |x, _| x);
        assert_eq!(x.scan(axis, |x, _| x * 1u32).to_vec().unwrap(), want, "first {what}");
    }
}

#[test]
fn reductions_are_exact_around_block_boundaries() {
    let ctx = Context::new().unwrap();
    let mut shapes: Vec<(&[usize], usize)> = LINE_SHAPES.to_vec();
    shapes.extend_from_slice(&[
        (&[16383], 0), (&[16384], 0), (&[16385], 0), (&[32769, 2], 0), (&[16385, 3], 0), (&[40000, 2, 2], 0),
        (&[2, 16385], 1), (&[300, 3, 300], 1), (&[4, 4, 17000], 2),
    ]);
    for (dims, axis) in shapes {
        let d = shape4(dims);
        let len: usize = dims.iter().product();
        let data = counts(len, len as u32 + 1);
        let x = Array::from_slice_on(&ctx, &data, d).unwrap();
        let what = format!("{dims:?} axis {axis}");
        let sum = x.sum_axis(axis);
        let mut out = d;
        out[axis] = 1;
        assert_eq!(sum.shape().dims(), out, "{what}");
        assert_eq!(sum.to_vec().unwrap(), host_lines(d, axis, &data, false, u32::wrapping_add), "sum {what}");
        assert_eq!(x.min_axis(axis).to_vec().unwrap(), host_lines(d, axis, &data, false, u32::min), "min {what}");
        assert_eq!(x.max_axis(axis).to_vec().unwrap(), host_lines(d, axis, &data, false, u32::max), "max {what}");
    }
}
