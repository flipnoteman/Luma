//! `sort`, `argsort`, their descending forms and `sort_axis` against Rust's stable sorts.

use luma::{Array, Context, SortKey};

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 32) as u32
    }
}

/// A key type with the order the sort promises, as a host `u32` key.
trait Host: SortKey + Copy + std::fmt::Debug {
    fn random(rng: &mut Rng, i: usize) -> Self;
    /// Compares as the sort orders: `-0.0 < +0.0`, NaNs last (their sign is not kept).
    fn key(self) -> u32;
}

impl Host for u32 {
    fn random(rng: &mut Rng, _: usize) -> Self {
        rng.next()
    }
    fn key(self) -> u32 {
        self
    }
}

impl Host for i32 {
    fn random(rng: &mut Rng, _: usize) -> Self {
        rng.next() as i32
    }
    fn key(self) -> u32 {
        self as u32 ^ 0x8000_0000
    }
}

impl Host for f32 {
    fn random(rng: &mut Rng, i: usize) -> Self {
        const SPECIAL: [f32; 8] = [f32::NAN, -f32::NAN, 0.0, -0.0, f32::INFINITY, f32::NEG_INFINITY, f32::MAX, f32::MIN_POSITIVE];
        if i % 13 == 5 {
            SPECIAL[rng.next() as usize % SPECIAL.len()]
        } else {
            // Wide exponent range, both signs, plenty of equal values.
            let m = (rng.next() % 2001) as f32 - 1000.0;
            m * 2f32.powi((rng.next() % 9) as i32 - 4)
        }
    }
    fn key(self) -> u32 {
        let b = if self.is_nan() { 0x7fc0_0000 } else { self.to_bits() };
        if b & 0x8000_0000 != 0 { !b } else { b | 0x8000_0000 }
    }
}

fn data<T: Host>(n: usize, seed: u64) -> Vec<T> {
    let mut rng = Rng(seed);
    (0..n).map(|i| T::random(&mut rng, i)).collect()
}

fn keys<T: Host>(v: &[T]) -> Vec<u32> {
    v.iter().map(|x| x.key()).collect()
}

/// The stable order of `host`: ascending, or ascending by the complemented key.
fn host_order<T: Host>(host: &[T], descending: bool) -> Vec<u32> {
    let mut order: Vec<u32> = (0..host.len() as u32).collect();
    if descending {
        order.sort_by_key(|&i| !host[i as usize].key());
    } else {
        order.sort_by_key(|&i| host[i as usize].key());
    }
    order
}

fn check_all<T: Host>(ctx: &Context, host: &[T]) {
    let n = host.len();
    let x = Array::from_slice_on(ctx, host, n).unwrap();
    for descending in [false, true] {
        let order = host_order(host, descending);
        let sorted: Vec<T> = order.iter().map(|&i| host[i as usize]).collect();
        let (values, indices) = if descending { (x.sort_descending(), x.argsort_descending()) } else { (x.sort(), x.argsort()) };
        let got = values.to_vec().unwrap();
        assert_eq!(keys(&got), keys(&sorted), "sort n={n} descending={descending}");
        assert_eq!(indices.to_vec().unwrap(), order, "argsort n={n} descending={descending}");
    }
}

#[test]
fn sorts_match_the_host_for_every_type_and_size() {
    let ctx = Context::new().unwrap();
    for n in [1usize, 2, 255, 256, 2047, 2048, 2049, 4095, 4096, 4097, 5000, 1 << 22] {
        check_all::<u32>(&ctx, &data(n, 1));
        check_all::<i32>(&ctx, &data(n, 2));
        check_all::<f32>(&ctx, &data(n, 3));
    }
}

#[test]
fn many_duplicates_keep_their_input_order() {
    let ctx = Context::new().unwrap();
    for n in [100_000usize, 4096, 4097] {
        let host: Vec<u32> = data::<u32>(n, 4).iter().map(|x| x % 7).collect();
        check_all(&ctx, &host);
        let same = vec![9u32; n];
        check_all(&ctx, &same);
        let signed: Vec<i32> = host.iter().map(|&x| x as i32 - 3).collect();
        check_all(&ctx, &signed);
    }
}

#[test]
fn results_keep_the_shape() {
    let ctx = Context::new().unwrap();
    let host: Vec<f32> = data(24, 5);
    let x = Array::from_slice_on(&ctx, &host, [4, 3, 2]).unwrap();
    assert_eq!(x.sort().shape().dims(), [4, 3, 2, 1]);
    assert_eq!(x.argsort().shape().dims(), [4, 3, 2, 1]);
    let order = host_order(&host, false);
    assert_eq!(x.argsort().to_vec().unwrap(), order);
}

/// Sorts, argsorts and sorts descending `lazy` (an expression, possibly pending) and compares
/// with the host sorts of `host`, its values.
fn check_lazy<T: Host>(what: &str, lazy: &Array<T>, host: &[T]) {
    for descending in [false, true] {
        let order = host_order(host, descending);
        let sorted: Vec<T> = order.iter().map(|&i| host[i as usize]).collect();
        let (values, indices) = if descending { (lazy.sort_descending(), lazy.argsort_descending()) } else { (lazy.sort(), lazy.argsort()) };
        assert_eq!(keys(&values.to_vec().unwrap()), keys(&sorted), "{what}: sort descending={descending}");
        assert_eq!(indices.to_vec().unwrap(), order, "{what}: argsort descending={descending}");
    }
}

#[test]
fn pending_keys_are_sorted_without_evaluating_them() {
    let ctx = Context::new().unwrap();
    for n in [2usize, 2049, 10_000] {
        let host: Vec<f32> = data(n, 6);
        let x = Array::from_slice_on(&ctx, &host, n).unwrap();
        let doubled: Vec<f32> = host.iter().map(|v| v * 2.0 - 1.0).collect();
        check_lazy("f32", &(&x * 2.0 - 1.0), &doubled);

        let host: Vec<i32> = data(n, 7);
        let x = Array::from_slice_on(&ctx, &host, n).unwrap();
        let mapped: Vec<i32> = host.iter().map(|v| v.wrapping_mul(3).wrapping_add(1)).collect();
        check_lazy("i32", &(&x * 3i32 + 1i32), &mapped);

        let host: Vec<u32> = data(n, 8);
        let x = Array::from_slice_on(&ctx, &host, n).unwrap();
        let mapped: Vec<u32> = host.iter().map(|v| v.wrapping_mul(7).wrapping_add(3) % 1000).collect();
        check_lazy("u32", &((&x * 7u32 + 3u32) % 1000u32), &mapped);
    }
    // Many equal keys: stability through the fused first pass.
    let host: Vec<u32> = data::<u32>(100_000, 9).iter().map(|x| x % 5).collect();
    let x = Array::from_slice_on(&ctx, &host, host.len()).unwrap();
    let mapped: Vec<u32> = host.iter().map(|v| v + 1).collect();
    check_lazy("duplicates", &(&x + 1u32), &mapped);
}

#[test]
fn pending_keys_through_views_and_broadcasts() {
    let ctx = Context::new().unwrap();
    let host: Vec<f32> = data(1000 * 10, 10);
    let x = Array::from_slice_on(&ctx, &host, [1000, 10]).unwrap();
    // `transposed[i + 10 * j] = host[j + 1000 * i]`.
    let transposed: Vec<f32> = (0..host.len()).map(|f| host[(f / 10) + 1000 * (f % 10)]).collect();
    check_lazy("transpose", &(&x.transpose() * 2.0), &transposed.iter().map(|v| v * 2.0).collect::<Vec<_>>());

    let column: Vec<f32> = data(100, 11);
    let a = Array::from_slice_on(&ctx, &column, [100, 1]).unwrap();
    let wide: Vec<f32> = (0..100 * 50).map(|f| column[f % 100] + (f / 100) as f32).collect();
    let steps = Array::<f32>::range_on(&ctx, 50, 0).unwrap().reshape([1, 50]);
    let lazy = &a.broadcast_to([100, 50]) + &steps.broadcast_to([100, 50]);
    check_lazy("broadcast", &lazy, &wide);

    // A circular shift of an expression, with or without fusing it into the first pass.
    let n = 5000;
    let host: Vec<f32> = data(n, 12);
    let x = Array::from_slice_on(&ctx, &host, n).unwrap();
    let shifted: Vec<f32> = (0..n).map(|i| host[(i + n - 3) % n] * 2.0 + host[i]).collect();
    check_lazy("shift", &((&x * 2.0).shift(3) + &x), &shifted);
}

#[test]
fn pending_keys_are_not_evaluated_separately() {
    let ctx = Context::new().unwrap();
    if !ctx.profiling_supported() {
        return;
    }
    let n = 100_000;
    let host: Vec<f32> = data::<f32>(n, 13).into_iter().filter(|x| x.is_finite()).collect();
    let n = host.len();
    let x = Array::from_slice_on(&ctx, &host, n).unwrap();
    x.eval().unwrap();
    ctx.flush().unwrap();
    let (sorted, profile) = ctx.profile(|| (&x * 2.0).sort()).unwrap();
    let (fused, others): (Vec<_>, Vec<_>) = profile.kernels.iter().partition(|k| k.label.starts_with("FusedRadix"));
    assert_eq!(fused.iter().map(|k| k.launches).sum::<u32>(), 2, "{profile}");
    assert!(others.iter().all(|k| !k.label.starts_with("Fused")), "{profile}");
    let doubled: Vec<f32> = host.iter().map(|v| v * 2.0).collect();
    let order = host_order(&doubled, false);
    let want: Vec<f32> = order.iter().map(|&i| doubled[i as usize]).collect();
    assert_eq!(keys(&sorted.to_vec().unwrap()), keys(&want));
    // A payload sort: both first-pass kernels compute the keys.
    let (indices, profile) = ctx.profile(|| (&x * 2.0).argsort()).unwrap();
    assert!(profile.kernels.iter().all(|k| !k.label.starts_with("Fused(")), "{profile}");
    assert_eq!(indices.to_vec().unwrap(), order);
}

/// Sorts every line of `host` (shape `dims`) along `axis`.
fn host_sort_axis<T: Host>(host: &[T], dims: [usize; 4], axis: usize) -> Vec<T> {
    let n = dims[axis];
    let stride: usize = dims[..axis].iter().product();
    let mut out = host.to_vec();
    for line in 0..host.len() / n {
        let base = line % stride + (line / stride) * stride * n;
        let mut cells: Vec<T> = (0..n).map(|k| host[base + k * stride]).collect();
        cells.sort_by_key(|x| x.key());
        for (k, v) in cells.into_iter().enumerate() {
            out[base + k * stride] = v;
        }
    }
    out
}

fn check_axis<T: Host>(ctx: &Context, dims: &[usize], axis: usize, seed: u64) {
    let len: usize = dims.iter().product();
    let host: Vec<T> = data(len, seed);
    let mut full = [1usize; 4];
    full[..dims.len()].copy_from_slice(dims);
    let x = Array::from_slice_on(ctx, &host, full).unwrap();
    let want = host_sort_axis(&host, full, axis);
    let got = x.sort_axis(axis).to_vec().unwrap();
    assert_eq!(keys(&got), keys(&want), "dims {dims:?} axis {axis}");
}

#[test]
fn sort_axis_matches_the_host_per_line_sort() {
    let ctx = Context::new().unwrap();
    check_axis::<f32>(&ctx, &[37, 1000], 1, 7);
    check_axis::<f32>(&ctx, &[1000, 37], 0, 8);
    check_axis::<u32>(&ctx, &[1000, 37], 0, 9);
    check_axis::<i32>(&ctx, &[37, 1000], 1, 10);
    check_axis::<f32>(&ctx, &[5, 7, 11], 1, 11);
    check_axis::<u32>(&ctx, &[5, 7, 11, 3], 2, 12);
    check_axis::<f32>(&ctx, &[5, 7, 11], 0, 13);
    check_axis::<f32>(&ctx, &[2, 70000], 1, 14);
    check_axis::<u32>(&ctx, &[1, 4097], 1, 15);
    // One line along axis 0 and a single cell along axis 1.
    check_axis::<f32>(&ctx, &[3000], 0, 16);
    check_axis::<f32>(&ctx, &[3000, 1], 1, 17);
}

#[test]
fn bad_axis_poisons() {
    let ctx = Context::new().unwrap();
    let x = Array::from_slice_on(&ctx, &[3.0f32, 1.0, 2.0], 3).unwrap();
    assert!(x.sort_axis(4).to_vec().is_err());
}
