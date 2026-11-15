//! `argmax`, `argmin`, `argmax_axis` and `argmin_axis`, against host computations.

mod common;

use common::values;
use luma::{Array, Context, LumaError};

/// The index NumPy's `argmax` (`want_max`) or `argmin` returns for `line`: the first NaN if any,
/// else the first of the extreme elements.
fn host_arg<T: Copy + PartialOrd>(line: impl Iterator<Item = T>, want_max: bool) -> u32 {
    let mut best: Option<(T, u32)> = None;
    for (i, v) in line.enumerate() {
        let nan = v.partial_cmp(&v).is_none();
        best = match best {
            None => Some((v, i as u32)),
            Some((b, _)) if b.partial_cmp(&b).is_none() => best,
            Some(_) if nan => Some((v, i as u32)),
            Some((b, _)) if (want_max && v > b) || (!want_max && v < b) => Some((v, i as u32)),
            other => other,
        };
    }
    best.unwrap().1
}

/// `(n, stride, lines)` of `axis` of an array of `dims`, and the cell `k` of line `l`.
fn line_cell(dims: &[usize], axis: usize, l: usize, k: usize) -> usize {
    let stride: usize = dims[..axis].iter().product();
    (l % stride) + (l / stride) * stride * dims[axis] + k * stride
}

fn host_axis<T: Copy + PartialOrd>(data: &[T], dims: &[usize], axis: usize, want_max: bool) -> Vec<u32> {
    let lines = data.len() / dims[axis];
    (0..lines).map(|l| host_arg((0..dims[axis]).map(|k| data[line_cell(dims, axis, l, k)]), want_max)).collect()
}

/// `dims` as the four extents of a shape.
fn pad(dims: &[usize]) -> [usize; 4] {
    std::array::from_fn(|i| dims.get(i).copied().unwrap_or(1))
}

/// Values with many ties: a few distinct levels.
fn levels(n: usize, seed: u64, count: f32) -> Vec<f32> {
    values(n, seed, 0.0, count).iter().map(|v| v.floor()).collect()
}

#[test]
fn whole_array_f32_in_every_pass_shape() {
    let ctx = Context::new().unwrap();
    // Short lines (one pass over rows), one chunk, several chunks with the totals reduced by rows
    // and by chunks.
    for n in [1usize, 2, 7, 255, 256, 1000, 16384, 16385, 100_000, 5_000_000] {
        let host = values(n, n as u64, -1.0, 1.0);
        let x = Array::from_slice_on(&ctx, &host, n).unwrap();
        assert_eq!(x.argmax().item().unwrap(), host_arg(host.iter().copied(), true), "argmax n = {n}");
        assert_eq!(x.argmin().item().unwrap(), host_arg(host.iter().copied(), false), "argmin n = {n}");
    }
}

#[test]
fn ties_go_to_the_lowest_index() {
    let ctx = Context::new().unwrap();
    for n in [3usize, 300, 40_000, 1_000_000] {
        let host = levels(n, 7 + n as u64, 3.0);
        let x = Array::from_slice_on(&ctx, &host, n).unwrap();
        assert_eq!(x.argmax().item().unwrap(), host_arg(host.iter().copied(), true), "argmax n = {n}");
        assert_eq!(x.argmin().item().unwrap(), host_arg(host.iter().copied(), false), "argmin n = {n}");
        // `-0.0` and `0.0` are equal.
        let zeros = Array::from_slice_on(&ctx, &[0.0f32, -0.0, 0.0, -0.0], 4).unwrap();
        assert_eq!(zeros.argmax().item().unwrap(), 0);
        assert_eq!(zeros.argmin().item().unwrap(), 0);
    }
}

#[test]
fn nan_is_the_first_nan_for_both() {
    let ctx = Context::new().unwrap();
    for n in [5usize, 1000, 70_000] {
        let mut host = values(n, 11, -1.0, 1.0);
        host[n / 2 + 1] = f32::NAN;
        host[n - 1] = f32::NAN;
        let x = Array::from_slice_on(&ctx, &host, n).unwrap();
        assert_eq!(x.argmax().item().unwrap(), (n / 2 + 1) as u32, "argmax n = {n}");
        assert_eq!(x.argmin().item().unwrap(), (n / 2 + 1) as u32, "argmin n = {n}");
        let m = x.argmax_axis(0);
        assert_eq!(m.item().unwrap(), (n / 2 + 1) as u32);
    }
    // Infinities are numbers.
    let x = Array::from_slice_on(&ctx, &[1.0f32, f32::INFINITY, f32::INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY], 5).unwrap();
    assert_eq!(x.argmax().item().unwrap(), 1);
    assert_eq!(x.argmin().item().unwrap(), 3);
}

#[test]
fn integers() {
    let ctx = Context::new().unwrap();
    for n in [4usize, 1000, 300_000] {
        let host_u: Vec<u32> = values(n, 3, 0.0, 50.0).iter().map(|&v| v as u32).collect();
        let x = Array::from_slice_on(&ctx, &host_u, n).unwrap();
        assert_eq!(x.argmax().item().unwrap(), host_arg(host_u.iter().copied(), true));
        assert_eq!(x.argmin().item().unwrap(), host_arg(host_u.iter().copied(), false));
        // The top bit set: a `u32` above `i32::MAX` is the largest.
        let mut big = host_u.clone();
        big[n - 2] = u32::MAX - 1;
        big[1] = u32::MAX - 1;
        let x = Array::from_slice_on(&ctx, &big, n).unwrap();
        assert_eq!(x.argmax().item().unwrap(), 1);
        let host_i: Vec<i32> = values(n, 4, -50.0, 50.0).iter().map(|&v| v as i32).collect();
        let x = Array::from_slice_on(&ctx, &host_i, n).unwrap();
        assert_eq!(x.argmax().item().unwrap(), host_arg(host_i.iter().copied(), true));
        assert_eq!(x.argmin().item().unwrap(), host_arg(host_i.iter().copied(), false));
        let mut extremes = host_i.clone();
        extremes[n - 1] = i32::MIN;
        extremes[2] = i32::MIN;
        let x = Array::from_slice_on(&ctx, &extremes, n).unwrap();
        assert_eq!(x.argmin().item().unwrap(), 2);
    }
}

#[test]
fn along_every_axis_and_pass_shape() {
    let ctx = Context::new().unwrap();
    let shapes: [&[usize]; 13] = [
        &[1000, 7],      // contiguous lines of one chunk
        &[50_000, 3],    // chunks whose totals are short rows
        &[5_000_000, 2], // chunks whose totals are chunks
        &[100, 3],       // short rows
        &[7, 3000],      // strided lines in blocks, the totals short rows
        &[64, 4096],     // strided lines in blocks, the totals one chunk
        &[3, 100],       // short strided lines
        &[4, 5, 6],
        &[5, 400, 7],
        &[1, 9],
        &[9, 1],
        &[2, 70_000],    // strided, many blocks per line
        &[2, 400_000],   // strided, more blocks than a chunk takes: the totals are strided too
    ];
    for dims in shapes {
        let len: usize = dims.iter().product();
        for (name, host) in [("random", values(len, len as u64, -1.0, 1.0)), ("ties", levels(len, 5, 4.0))] {
            let x = Array::from_slice_on(&ctx, &host, pad(dims)).unwrap();
            for axis in 0..dims.len() {
                let max = x.argmax_axis(axis);
                let min = x.argmin_axis(axis);
                let mut out = dims.to_vec();
                out[axis] = 1;
                assert_eq!(max.shape().dims()[..dims.len()], out[..], "{dims:?} axis {axis}");
                assert_eq!(max.to_vec().unwrap(), host_axis(&host, dims, axis, true), "argmax {name} {dims:?} axis {axis}");
                assert_eq!(min.to_vec().unwrap(), host_axis(&host, dims, axis, false), "argmin {name} {dims:?} axis {axis}");
            }
        }
    }
}

#[test]
fn nan_along_an_axis() {
    let ctx = Context::new().unwrap();
    let dims = [64usize, 3000];
    let mut host = values(64 * 3000, 21, -1.0, 1.0);
    for (i, c) in [(5usize, 3usize), (2999, 8), (1500, 8), (0, 9)] {
        host[c + 64 * i] = f32::NAN;
    }
    let x = Array::from_slice_on(&ctx, &host, dims).unwrap();
    for axis in [0usize, 1] {
        assert_eq!(x.argmax_axis(axis).to_vec().unwrap(), host_axis(&host, &dims, axis, true), "axis {axis}");
        assert_eq!(x.argmin_axis(axis).to_vec().unwrap(), host_axis(&host, &dims, axis, false), "axis {axis}");
    }
    assert_eq!(x.argmax_axis(1).to_vec().unwrap()[8], 1500);
}

#[test]
fn pending_operands_are_fused() {
    let ctx = Context::new().unwrap();
    let host = values(64 * 5000, 31, -1.0, 1.0);
    let x = Array::from_slice_on(&ctx, &host, [64usize, 5000]).unwrap();
    x.eval().unwrap();
    ctx.flush().unwrap();
    // `-2 x + 1` reverses the order: the largest element becomes the smallest.
    let want_max = host_axis(&host, &[64, 5000], 1, false);
    let want_flat = host_arg(host.iter().copied(), false);
    let lazy = &(&x * -2.0) + 1.0;
    let (got, profile) = ctx.profile(|| lazy.argmax_axis(1)).unwrap();
    assert_eq!(got.to_vec().unwrap(), want_max);
    if ctx.profiling_supported() {
        assert!(profile.launches() <= 2, "{profile}");
    }
    let (got, profile) = ctx.profile(|| lazy.argmax()).unwrap();
    assert_eq!(got.item().unwrap(), want_flat);
    if ctx.profiling_supported() {
        assert!(profile.launches() <= 2, "{profile}");
    }
    // A comparison or a cast in the expression.
    let y = x.gt(0.0).to_f32();
    let flags: Vec<f32> = host.iter().map(|&v| f32::from(v > 0.0)).collect();
    assert_eq!(y.argmax_axis(1).to_vec().unwrap(), host_axis(&flags, &[64, 5000], 1, true));
}

#[test]
fn a_view_is_read_through() {
    let ctx = Context::new().unwrap();
    let host = values(40 * 30, 41, -1.0, 1.0);
    let x = Array::from_slice_on(&ctx, &host, [40usize, 30]).unwrap();
    let t = x.slice(1, 3..20);
    let want: Vec<f32> = (3..20).flat_map(|r| host[40 * r..40 * (r + 1)].to_vec()).collect();
    assert_eq!(t.argmax().item().unwrap(), host_arg(want.iter().copied(), true));
    assert_eq!(t.argmin_axis(1).to_vec().unwrap(), host_axis(&want, &[40, 17], 1, false));
}

#[test]
fn unsupported_types_and_bad_axes_poison_the_result() {
    let ctx = Context::new().unwrap();
    let x = Array::<f32>::zeros_on(&ctx, [4, 4]).unwrap();
    assert!(matches!(x.argmax_axis(4).error().map(LumaError::root), Some(LumaError::InvalidShape(_))));
    let d = x.to_df32();
    assert!(matches!(d.argmax().error().map(LumaError::root), Some(LumaError::Unsupported(_))), "{:?}", d.argmax().error());
    // A poisoned operand passes its error on.
    let bad = x.matmul(&Array::<f32>::zeros_on(&ctx, [5, 6]).unwrap());
    assert!(bad.argmax().error().is_some());
}

#[test]
fn the_index_feeds_the_gpu_without_a_readback() {
    let ctx = Context::new().unwrap();
    let host = values(100_000, 51, -1.0, 1.0);
    let x = Array::from_slice_on(&ctx, &host, 100_000).unwrap();
    let top = x.take(&x.argmax());
    let bottom = x.take(&x.argmin());
    assert_eq!(top.item().unwrap(), host.iter().copied().fold(f32::MIN, f32::max));
    assert_eq!(bottom.item().unwrap(), host.iter().copied().fold(f32::MAX, f32::min));
}
