//! Arrays larger than one buffer: split into segments of `chunk_bytes`. Convolutions run piecewise
//! over such arrays too; they are compared with the same call on an unchunked context.

use luma::{Array, Context, ContextOptions, LumaError};

/// Elements of the test arrays: ten full segments of 1024 `f32` plus a partial one.
const N: usize = 10_001;

/// A context whose buffers hold 4096 bytes: 1024 `f32` per segment.
fn chunked() -> Context {
    Context::with_options(ContextOptions::default().chunk_bytes(4096))
        .expect("a device is available")
}

fn ramp() -> Vec<f32> {
    (0..N).map(|i| i as f32 * 0.25 - 100.0).collect()
}

#[test]
fn upload_and_to_vec_round_trip() {
    let ctx = chunked();
    let data = ramp();
    let x = Array::from_slice_on(&ctx, &data, N).unwrap();
    assert_eq!(x.to_vec().unwrap(), data);
}

#[test]
fn read_into_fills_every_segment() {
    let ctx = chunked();
    let data = ramp();
    let x = Array::from_slice_on(&ctx, &data, N).unwrap();
    let mut out = vec![0.0f32; N];
    x.read_into(&mut out).unwrap();
    assert_eq!(out, data);
    let mut short = vec![0.0f32; N - 1];
    assert!(matches!(
        x.read_into(&mut short),
        Err(LumaError::LengthMismatch { expected: N, actual }) if actual == N - 1
    ));
}

#[test]
fn write_replaces_every_segment() {
    let ctx = chunked();
    let mut x = Array::from_slice_on(&ctx, &ramp(), N).unwrap();
    let fresh: Vec<f32> = (0..N).map(|i| (i % 97) as f32).collect();
    x.write(&fresh).unwrap();
    assert_eq!(x.to_vec().unwrap(), fresh);
}

#[test]
fn zeros_clears_every_segment() {
    let ctx = chunked();
    // Dirty the pool so that reused buffers hold stale data.
    drop(Array::from_slice_on(&ctx, &vec![7.0f32; N], N).unwrap());
    let x = Array::<f32>::zeros_on(&ctx, N).unwrap();
    assert_eq!(x.to_vec().unwrap(), vec![0.0f32; N]);
}

#[test]
fn u32_array_round_trip() {
    let ctx = chunked();
    let data: Vec<u32> = (0..N as u32).map(|i| i.wrapping_mul(2_654_435_761)).collect();
    let x = Array::from_slice_on(&ctx, &data, N).unwrap();
    assert_eq!(x.to_vec().unwrap(), data);
}

#[test]
fn vector_array_round_trip() {
    let ctx = chunked();
    // 16-byte elements: 256 per segment, 40 segments.
    let data: Vec<[f32; 4]> = (0..N).map(|i| [i as f32, -(i as f32), 0.5, i as f32 * 2.0]).collect();
    let x = Array::from_slice_on(&ctx, &data, N).unwrap();
    assert_eq!(x.to_vec().unwrap(), data);
}

#[test]
fn chunk_bytes_must_be_a_power_of_two_of_at_least_1024() {
    for bytes in [1000, 512] {
        let options = ContextOptions::default().chunk_bytes(bytes);
        assert!(
            matches!(Context::with_options(options), Err(LumaError::InvalidShape(_))),
            "chunk_bytes {bytes} must be rejected"
        );
    }
}

use luma::{eval_all, select, Reduction};

fn close(got: f32, want: f32, tol: f32) {
    assert!((got - want).abs() <= tol * want.abs().max(1.0), "got {got}, want {want}");
}

fn ys() -> Vec<f32> {
    (0..N).map(|i| ((i * 7) % 31) as f32 * 0.5 - 3.0).collect()
}

#[test]
fn full_fills_every_segment() {
    let ctx = chunked();
    drop(Array::from_slice_on(&ctx, &vec![9.0f32; N], N).unwrap());
    assert_eq!(Array::<f32>::full_on(&ctx, N, 1.5).unwrap().to_vec().unwrap(), vec![1.5f32; N]);
    assert_eq!(Array::<u32>::full_on(&ctx, N, 7).unwrap().to_vec().unwrap(), vec![7u32; N]);
}

#[test]
fn fused_chain_with_select_and_scalars() {
    let ctx = chunked();
    let (xs, ys) = (ramp(), ys());
    let x = Array::from_slice_on(&ctx, &xs, N).unwrap();
    let y = Array::from_slice_on(&ctx, &ys, N).unwrap();
    let out = select(&x.gt(0.25), &(&x * 2.0 + &y), &(&y * &y + 1.0)).sqrt() + 0.5;
    let want: Vec<f32> = xs
        .iter()
        .zip(&ys)
        .map(|(&x, &y)| (if x > 0.25 { x * 2.0 + y } else { y * y + 1.0 }).sqrt() + 0.5)
        .collect();
    let got = out.to_vec().unwrap();
    for (g, w) in got.iter().zip(&want) {
        if w.is_nan() {
            assert!(g.is_nan());
        } else {
            close(*g, *w, 1e-5);
        }
    }
}

#[test]
fn compound_assignment_chain_on_a_solely_owned_array() {
    let ctx = chunked();
    let (xs, ys) = (ramp(), ys());
    let mut z = Array::from_slice_on(&ctx, &xs, N).unwrap();
    let y = Array::from_slice_on(&ctx, &ys, N).unwrap();
    z += &y;
    z *= 0.5;
    z -= 0.25;
    let want: Vec<f32> = xs.iter().zip(&ys).map(|(x, y)| (x + y) * 0.5 - 0.25).collect();
    let got = z.to_vec().unwrap();
    for (g, w) in got.iter().zip(&want) {
        close(*g, *w, 1e-6);
    }
}

#[test]
fn compound_assignment_through_a_reshape_view() {
    let ctx = chunked();
    let x = Array::from_slice_on(&ctx, &ramp(), N).unwrap();
    let mut view = x.reshape([N]);
    view += 10.0;
    let want: Vec<f32> = ramp().iter().map(|v| v + 10.0).collect();
    assert_eq!(x.to_vec().unwrap(), want);
    assert_eq!(view.to_vec().unwrap(), want);
}

#[test]
fn reductions_over_materialized_and_lazy_arrays() {
    let ctx = chunked();
    let xs = ramp();
    let x = Array::from_slice_on(&ctx, &xs, N).unwrap();
    let sum: f64 = xs.iter().map(|&v| f64::from(v)).sum();
    let squares: f64 = xs.iter().map(|&v| f64::from(v) * f64::from(v)).sum();
    let lo = xs.iter().copied().fold(f32::INFINITY, f32::min);
    let hi = xs.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let lazy = &x * 1.0;
    for array in [&x, &lazy] {
        close(array.sum().item().unwrap(), sum as f32, 1e-4);
        assert_eq!(array.min().item().unwrap(), lo);
        assert_eq!(array.max().item().unwrap(), hi);
        close(array.norm().item().unwrap(), squares.sqrt() as f32, 1e-4);
        let [s, mn, mx, sq] =
            array.reduce([Reduction::Sum, Reduction::Min, Reduction::Max, Reduction::SumOfSquares]).to_array().unwrap();
        close(s, sum as f32, 1e-4);
        assert_eq!((mn, mx), (lo, hi));
        close(sq, squares as f32, 1e-4);
    }
}

#[test]
fn u32_sum_wraps_around() {
    let ctx = chunked();
    let data: Vec<u32> = (0..N as u32).map(|i| u32::MAX / 3 + i).collect();
    let want = data.iter().fold(0u32, |a, &v| a.wrapping_add(v));
    let x = Array::from_slice_on(&ctx, &data, N).unwrap();
    assert_eq!(x.reduce([Reduction::Sum]).item().unwrap(), want);
    let lazy = &x + 0u32;
    assert_eq!(lazy.reduce([Reduction::Sum]).item().unwrap(), want);
}

#[test]
fn vector_sum_spans_segments() {
    let ctx = chunked();
    let data: Vec<[f32; 4]> = (0..N).map(|i| [(i % 10) as f32, 1.0, -0.5, (i % 3) as f32]).collect();
    let x = Array::from_slice_on(&ctx, &data, N).unwrap();
    let got = x.sum().unwrap();
    for c in 0..4 {
        let want: f64 = data.iter().map(|v| f64::from(v[c])).sum();
        close(got[c], want as f32, 1e-5);
    }
}

#[test]
fn eval_all_computes_chunked_arrays_together() {
    let ctx = chunked();
    let (xs, ys) = (ramp(), ys());
    let x = Array::from_slice_on(&ctx, &xs, N).unwrap();
    let y = Array::from_slice_on(&ctx, &ys, N).unwrap();
    let a = &x * 2.0 + 1.0;
    let b = &a - &y;
    eval_all(&[&a, &b]).unwrap();
    let want_a: Vec<f32> = xs.iter().map(|v| v * 2.0 + 1.0).collect();
    let want_b: Vec<f32> = want_a.iter().zip(&ys).map(|(a, y)| a - y).collect();
    assert_eq!(a.to_vec().unwrap(), want_a);
    assert_eq!(b.to_vec().unwrap(), want_b);
}

/// Element `i` of the circular shift of `data` (first dimension fastest) by `offsets`: the
/// element at coordinates `c - offset` in every dimension.
fn shift_host(data: &[f32], dims: &[usize], offsets: &[isize]) -> Vec<f32> {
    let mut out = vec![0.0f32; data.len()];
    for (i, slot) in out.iter_mut().enumerate() {
        let (mut rest, mut stride, mut from) = (i, 1usize, 0usize);
        for (&dim, &offset) in dims.iter().zip(offsets) {
            let coord = (rest % dim) as isize;
            rest /= dim;
            from += (coord - offset).rem_euclid(dim as isize) as usize * stride;
            stride *= dim;
        }
        *slot = data[from];
    }
    out
}

#[test]
fn linear_shifts_cross_segments() {
    let ctx = chunked();
    let xs = ramp();
    let u = Array::from_slice_on(&ctx, &xs, N).unwrap();
    for s in [1isize, -1, 1500, -4097] {
        assert_eq!(u.shift(s).to_vec().unwrap(), shift_host(&xs, &[N], &[s]), "shift {s}");
    }
    let lap = u.shift(1) + u.shift(-1) - &u * 2.0;
    let want: Vec<f32> = shift_host(&xs, &[N], &[1])
        .iter()
        .zip(shift_host(&xs, &[N], &[-1]))
        .zip(&xs)
        .map(|((a, b), x)| a + b - x * 2.0)
        .collect();
    for (g, w) in lap.to_vec().unwrap().iter().zip(&want) {
        close(*g, *w, 1e-5);
    }
}

#[test]
fn multi_dimensional_shifts_cross_segments() {
    let ctx = chunked();
    let dims = [101usize, 99];
    let n = dims[0] * dims[1];
    let xs: Vec<f32> = (0..n).map(|i| ((i * 13) % 251) as f32 * 0.5).collect();
    let u = Array::from_slice_on(&ctx, &xs, dims).unwrap();
    for offsets in [[1isize, 0], [0, -1], [3, 5]] {
        assert_eq!(u.shift(offsets).to_vec().unwrap(), shift_host(&xs, &dims, &offsets), "shift {offsets:?}");
    }
    let stencil = u.shift([1, 0]) + u.shift([-1, 0]) + u.shift([0, 1]) + u.shift([0, -1]) - &u * 4.0;
    let parts = [[1isize, 0], [-1, 0], [0, 1], [0, -1]].map(|o| shift_host(&xs, &dims, &o));
    let want: Vec<f32> = (0..n)
        .map(|i| parts[0][i] + parts[1][i] + parts[2][i] + parts[3][i] - xs[i] * 4.0)
        .collect();
    for (g, w) in stencil.to_vec().unwrap().iter().zip(&want) {
        close(*g, *w, 1e-5);
    }
}

#[test]
fn shifted_expression_inside_a_reduction() {
    let ctx = chunked();
    let xs = ramp();
    let u = Array::from_slice_on(&ctx, &xs, N).unwrap();
    let lazy = u.shift(1) + &u;
    let shifted = shift_host(&xs, &[N], &[1]);
    let want: f64 = shifted.iter().zip(&xs).map(|(a, b)| f64::from(*a) + f64::from(*b)).sum();
    close(lazy.sum().item().unwrap(), want as f32, 1e-4);
}

#[test]
fn shift_of_a_lazy_expression_is_evaluated_first() {
    let ctx = chunked();
    let (xs, ys) = (ramp(), ys());
    let a = Array::from_slice_on(&ctx, &xs, N).unwrap();
    let b = Array::from_slice_on(&ctx, &ys, N).unwrap();
    let product: Vec<f32> = xs.iter().zip(&ys).map(|(x, y)| x * y).collect();
    let got = (&a * &b).shift(1).to_vec().unwrap();
    assert_eq!(got, shift_host(&product, &[N], &[1]));
}

#[test]
fn iterate_on_chunked_arrays_matches_the_unchunked_result() {
    let dims = [101usize, 99];
    let n = dims[0] * dims[1];
    let xs: Vec<f32> = (0..n).map(|i| ((i * 31) % 17) as f32).collect();
    let heat = |u: &Array<f32>| {
        let lap = u.shift([1, 0]) + u.shift([-1, 0]) + u.shift([0, 1]) + u.shift([0, -1]) - u * 4.0;
        u + lap * 0.2
    };
    let chunked_ctx = chunked();
    let got = Array::from_slice_on(&chunked_ctx, &xs, dims).unwrap().iterate(7, heat).to_vec().unwrap();
    let plain = Context::new().unwrap();
    let want = Array::from_slice_on(&plain, &xs, dims).unwrap().iterate(7, heat).to_vec().unwrap();
    for (g, w) in got.iter().zip(&want) {
        close(*g, *w, 1e-5);
    }
}

// -- Convolutions in pieces (65536-byte segments) --

/// A context whose buffers hold 16384 `f32` each.
fn chunked_64k() -> Context {
    Context::with_options(ContextOptions::default().chunk_bytes(65536)).unwrap()
}

fn input_data(len: usize) -> Vec<f32> {
    (0..len).map(|i| ((i * 37 % 101) as f32 - 50.0) * 0.01).collect()
}

fn kernel_data(len: usize) -> Vec<f32> {
    (0..len).map(|k| ((k * 13 % 29) as f32 - 14.0) * 0.05).collect()
}

/// `conv1d` on the chunked context and on the default one must agree.
fn check_conv1d(chunked: &Context, plain: &Context, input_len: usize, kernel_len: usize, stride: usize, padding: usize) {
    let (x, k) = (input_data(input_len), kernel_data(kernel_len));
    let want = Array::from_slice_on(plain, &x, input_len)
        .unwrap()
        .conv1d(&Array::from_slice_on(plain, &k, kernel_len).unwrap(), stride, padding)
        .to_vec()
        .unwrap();
    let got = Array::from_slice_on(chunked, &x, input_len)
        .unwrap()
        .conv1d(&Array::from_slice_on(chunked, &k, kernel_len).unwrap(), stride, padding)
        .to_vec()
        .unwrap();
    let case = format!("{input_len}/{kernel_len} stride {stride} padding {padding}");
    assert_eq!(got.len(), want.len(), "{case}");
    for (o, (g, w)) in got.iter().zip(&want).enumerate() {
        assert!((g - w).abs() < 1e-3, "{case} at {o}: got {g}, want {w}");
    }
}

#[test]
fn conv1d_over_several_buffers_matches_one_buffer() {
    let (chunked, plain) = (chunked_64k(), Context::new().unwrap());
    for stride in 1..=5 {
        for padding in [0, 32] {
            check_conv1d(&chunked, &plain, 100_000, 64, stride, padding);
        }
    }
}

#[test]
fn conv1d_cases_of_the_tile_test_over_several_buffers() {
    let (chunked, plain) = (chunked_64k(), Context::new().unwrap());
    let cases = [
        (100_000usize, 300usize, 0usize),
        (90_001, 517, 7),
        (50_000, 1, 0),
        (41_113, 16, 2),
        (40_000, 260, 130),
    ];
    // Strides up to one beyond the tiled kernels.
    for stride in 1..=5 {
        for (input_len, kernel_len, padding) in cases {
            check_conv1d(&chunked, &plain, input_len, kernel_len, stride, padding);
        }
    }
    check_conv1d(&chunked, &plain, 100_000, 64, 9, 5);
}

#[test]
fn conv1d_with_pieces_inside_the_padding() {
    let (chunked, plain) = (chunked_64k(), Context::new().unwrap());
    check_conv1d(&chunked, &plain, 40_000, 9, 1, 20_000);
    check_conv1d(&chunked, &plain, 40_000, 9, 3, 30_000);
}

#[test]
fn conv1d_fft_over_several_buffers_matches_one_buffer() {
    let (chunked, plain) = (chunked_64k(), Context::new().unwrap());
    for (input_len, kernel_len) in [(100_000usize, 300usize), (70_000, 1000)] {
        let (x, k) = (input_data(input_len), kernel_data(kernel_len));
        let want = Array::from_slice_on(&plain, &x, input_len)
            .unwrap()
            .conv1d_fft(&Array::from_slice_on(&plain, &k, kernel_len).unwrap())
            .to_vec()
            .unwrap();
        let got = Array::from_slice_on(&chunked, &x, input_len)
            .unwrap()
            .conv1d_fft(&Array::from_slice_on(&chunked, &k, kernel_len).unwrap())
            .to_vec()
            .unwrap();
        assert_eq!(got.len(), want.len());
        let scale = want.iter().fold(1.0f32, |m, v| m.max(v.abs()));
        for (o, (g, w)) in got.iter().zip(&want).enumerate() {
            assert!((g - w).abs() < 1e-3 * scale, "{input_len}/{kernel_len} at {o}: got {g}, want {w}");
        }
    }
}

#[test]
fn kernels_too_long_for_the_pieces_are_refused() {
    let ctx = chunked_64k();
    let input = Array::from_slice_on(&ctx, &input_data(100_000), 100_000).unwrap();
    let long = Array::from_slice_on(&ctx, &kernel_data(9000), 9000).unwrap();
    let err = input.conv1d(&long, 1, 0).to_vec().unwrap_err();
    assert!(matches!(err.root(), LumaError::Chunked(_)), "got: {err}");
    let long = Array::from_slice_on(&ctx, &kernel_data(2000), 2000).unwrap();
    let err = input.conv1d_fft(&long).to_vec().unwrap_err();
    assert!(matches!(err.root(), LumaError::Chunked(_)), "got: {err}");
}
