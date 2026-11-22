//! `sort`, `argsort`, their descending forms and `sort_axis` against a stable host sort (an LSD radix sort on the keys).

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

/// The stable order of `host`: ascending, or ascending by the complemented key. A least-significant-digit radix sort
/// on three 11-bit digits of the key is stable, and several times faster than a comparison sort of the indices on the
/// 4M-element cases (which dominated the time of these tests).
fn host_order<T: Host>(host: &[T], descending: bool) -> Vec<u32> {
    let key = |i: usize| if descending { !host[i].key() } else { host[i].key() };
    let mut from: Vec<(u32, u32)> = (0..host.len()).map(|i| (key(i), i as u32)).collect();
    let mut to = from.clone();
    for shift in [0u32, 11, 22] {
        let mut start = [0usize; 2049];
        for &(k, _) in &from {
            start[((k >> shift) & 2047) as usize + 1] += 1;
        }
        for digit in 0..2048 {
            start[digit + 1] += start[digit];
        }
        for &entry in &from {
            let slot = &mut start[((entry.0 >> shift) & 2047) as usize];
            to[*slot] = entry;
            *slot += 1;
        }
        std::mem::swap(&mut from, &mut to);
    }
    from.into_iter().map(|(_, i)| i).collect()
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

/// Lines shorter than a tile of the line-sort kernel (padded to a power of two, several per
/// workgroup), longer ones (the radix route and its transpose), along every axis, with ragged
/// ends: lines that do not fill the last tile and extents on both sides of a power of two.
#[test]
fn sort_axis_matches_the_host_at_ragged_tiles() {
    let ctx = Context::new().unwrap();
    let cases: &[(&[usize], usize)] = &[
        (&[2, 1001], 0),
        (&[1001, 2], 1),
        (&[3, 5000], 0),
        (&[31, 100], 0),
        (&[33, 77], 0),
        (&[64, 100], 0),
        (&[65, 33], 0),
        (&[1000, 7], 0),
        (&[1024, 5], 0),
        (&[1025, 3], 0),
        (&[2048, 4], 0),
        (&[2049, 3], 0),
        (&[4097, 2], 0),
        (&[100, 33], 1),
        (&[70, 64], 1),
        (&[100, 1025], 1),
        (&[1000, 1025], 1),
        (&[9, 2048], 1),
        (&[9, 2049], 1),
        (&[5, 64, 7], 1),
        (&[33, 65, 9], 1),
        (&[7, 2048, 3], 1),
        (&[3, 2500, 5], 1),
        (&[3, 5, 17, 2], 2),
        (&[3, 5, 3000], 2),
        (&[40, 3, 2, 70], 3),
        (&[40, 3, 2, 70], 0),
        (&[40, 3, 2, 70], 1),
        (&[40, 3, 2, 70], 2),
        (&[1500, 3000], 1),
    ];
    for (k, &(dims, axis)) in cases.iter().enumerate() {
        check_axis::<f32>(&ctx, dims, axis, 100 + k as u64);
        check_axis::<u32>(&ctx, dims, axis, 200 + k as u64);
        check_axis::<i32>(&ctx, dims, axis, 300 + k as u64);
    }
}

/// The result of `sort_axis` is an array like any other: it feeds further operations, and a line
/// of a sorted array sorts to itself.
#[test]
fn sorted_axis_results_compose() {
    let ctx = Context::new().unwrap();
    for dims in [[50usize, 40], [40, 3000]] {
        let host: Vec<f32> = data(dims[0] * dims[1], 7).into_iter().map(|v: f32| if v.is_nan() { 1.0 } else { v }).collect();
        let x = Array::from_slice_on(&ctx, &host, dims).unwrap();
        for axis in 0..2 {
            let once = x.sort_axis(axis);
            let twice = once.sort_axis(axis);
            assert_eq!(keys(&once.to_vec().unwrap()), keys(&twice.to_vec().unwrap()));
            let shifted = (&once + 1.0).to_vec().unwrap();
            let want: Vec<f32> = host_sort_axis(&host, [dims[0], dims[1], 1, 1], axis).iter().map(|v| v + 1.0).collect();
            assert_eq!(keys(&shifted), keys(&want));
        }
    }
}

#[test]
fn bad_axis_poisons() {
    let ctx = Context::new().unwrap();
    let x = Array::from_slice_on(&ctx, &[3.0f32, 1.0, 2.0], 3).unwrap();
    assert!(x.sort_axis(4).to_vec().is_err());
}

// sort_by_key

/// A type of values the tests carry: made from a number, compared exactly.
trait Val: luma::ScalarType + PartialEq {
    fn of(i: u32) -> Self;
}
impl Val for u32 {
    fn of(i: u32) -> Self {
        i
    }
}
impl Val for i32 {
    fn of(i: u32) -> Self {
        i as i32 - 100_000
    }
}
impl Val for f32 {
    fn of(i: u32) -> Self {
        i as f32 * 0.5
    }
}
impl Val for bool {
    fn of(i: u32) -> Self {
        i.is_multiple_of(3)
    }
}
impl Val for f64 {
    fn of(i: u32) -> Self {
        f64::from(i) * 0.25
    }
}

fn check_by_key<T: Host, V: Val + std::fmt::Debug>(ctx: &Context, host: &[T], what: &str) {
    let n = host.len();
    let vals: Vec<V> = (0..n as u32).map(V::of).collect();
    let (Ok(x), Ok(v)) = (Array::from_slice_on(ctx, host, n), Array::from_slice_on(ctx, &vals, n)) else {
        return;
    };
    for descending in [false, true] {
        let order = host_order(host, descending);
        let want_keys: Vec<T> = order.iter().map(|&i| host[i as usize]).collect();
        let want_vals: Vec<V> = order.iter().map(|&i| V::of(i)).collect();
        let (k, w) = if descending { x.sort_by_key_descending(&v) } else { x.sort_by_key(&v) };
        let got_keys = k.to_vec().unwrap();
        assert_eq!(keys(&got_keys), keys(&want_keys), "{what} keys, n={n} descending={descending}");
        assert_eq!(w.to_vec().unwrap(), want_vals, "{what} values, n={n} descending={descending}");
    }
}

fn check_by_key_types<T: Host>(ctx: &Context, host: &[T]) {
    let name = std::any::type_name::<T>();
    check_by_key::<T, u32>(ctx, host, &format!("{name}/u32"));
    check_by_key::<T, i32>(ctx, host, &format!("{name}/i32"));
    check_by_key::<T, f32>(ctx, host, &format!("{name}/f32"));
    check_by_key::<T, bool>(ctx, host, &format!("{name}/bool"));
    check_by_key::<T, f64>(ctx, host, &format!("{name}/f64"));
}

/// Keys of every type with every value type, ascending and descending, at odd sizes (tile
/// boundaries of the radix kernels among them): the values follow their keys, equal keys keep
/// their input order (the values are the input positions, so the host order checks stability),
/// NaN keys sort last (first when descending).
#[test]
fn sort_by_key_matches_the_host_for_every_type_and_size() {
    let ctx = Context::new().unwrap();
    for n in [2usize, 3, 255, 256, 2047, 2049, 4095, 4096, 4097, 8193, 5000, 100_001] {
        check_by_key_types::<u32>(&ctx, &data(n, 21));
        check_by_key_types::<i32>(&ctx, &data(n, 22));
        check_by_key_types::<f32>(&ctx, &data(n, 23));
    }
}

#[test]
fn sort_by_key_with_many_equal_keys_is_stable() {
    let ctx = Context::new().unwrap();
    for n in [100_000usize, 4096, 4097, 70_000] {
        let few: Vec<u32> = data::<u32>(n, 24).iter().map(|x| x % 5).collect();
        check_by_key_types(&ctx, &few);
        check_by_key_types(&ctx, &vec![7u32; n]);
        let signed: Vec<i32> = few.iter().map(|&x| x as i32 - 2).collect();
        check_by_key_types(&ctx, &signed);
        // Only the low byte differs: the upper passes move every key to the place it has.
        let low: Vec<u32> = data::<u32>(n, 25).iter().map(|x| x & 0xff).collect();
        check_by_key_types(&ctx, &low);
        // Only the high byte differs.
        let high: Vec<u32> = data::<u32>(n, 26).iter().map(|x| x & 0xff00_0000).collect();
        check_by_key_types(&ctx, &high);
        // NaNs and signed zeros among few distinct floats.
        let floats: Vec<f32> = few.iter().enumerate().map(|(i, &x)| [f32::NAN, -0.0, 0.0, -1.5, 2.0][x as usize] * if i % 2 == 0 { 1.0 } else { -1.0 }).collect();
        check_by_key_types(&ctx, &floats);
    }
}

#[test]
fn sort_by_key_keys_may_be_pending() {
    let ctx = Context::new().unwrap();
    let host: Vec<f32> = data::<f32>(70_001, 27).into_iter().map(|v| if v.is_nan() { 3.0 } else { v }).collect();
    let x = Array::from_slice_on(&ctx, &host, host.len()).unwrap();
    let vals: Vec<u32> = (0..host.len() as u32).collect();
    let v = Array::from_slice_on(&ctx, &vals, host.len()).unwrap();
    let lazy = &x * 2.0 - 1.0;
    let eager: Vec<f32> = host.iter().map(|h| h * 2.0 - 1.0).collect();
    let (k, w) = lazy.sort_by_key(&(&v + 1));
    let order = host_order(&eager, false);
    assert_eq!(keys(&k.to_vec().unwrap()), keys(&order.iter().map(|&i| eager[i as usize]).collect::<Vec<_>>()));
    assert_eq!(w.to_vec().unwrap(), order.iter().map(|&i| i + 1).collect::<Vec<_>>());
}

fn check_by_key_axis<T: Host>(ctx: &Context, dims: &[usize], axis: usize, seed: u64) {
    let len: usize = dims.iter().product();
    let host: Vec<T> = data(len, seed);
    let mut full = [1usize; 4];
    full[..dims.len()].copy_from_slice(dims);
    let vals: Vec<u32> = (0..len as u32).collect();
    let x = Array::from_slice_on(ctx, &host, full).unwrap();
    let v = Array::from_slice_on(ctx, &vals, full).unwrap();
    for descending in [false, true] {
        let (n, stride) = (full[axis], full[..axis].iter().product::<usize>());
        let mut want_keys = host.clone();
        let mut want_vals = vals.clone();
        for line in 0..len / n {
            let base = line % stride + (line / stride) * stride * n;
            let cells: Vec<T> = (0..n).map(|k| host[base + k * stride]).collect();
            let order = host_order(&cells, descending);
            for (k, &o) in order.iter().enumerate() {
                want_keys[base + k * stride] = cells[o as usize];
                want_vals[base + k * stride] = vals[base + o as usize * stride];
            }
        }
        let (k, w) = if descending { x.sort_by_key_axis_descending(axis, &v) } else { x.sort_by_key_axis(axis, &v) };
        assert_eq!(keys(&k.to_vec().unwrap()), keys(&want_keys), "keys dims {dims:?} axis {axis} descending {descending}");
        assert_eq!(w.to_vec().unwrap(), want_vals, "values dims {dims:?} axis {axis} descending {descending}");
        assert_eq!(k.shape().dims(), full);
    }
}

#[test]
fn sort_by_key_axis_matches_the_host_per_line_sort() {
    let ctx = Context::new().unwrap();
    let cases: &[(&[usize], usize)] = &[
        (&[37, 1000], 1),
        (&[1000, 37], 0),
        (&[5, 7, 11], 1),
        (&[5, 7, 11, 3], 2),
        (&[2, 7000], 1),
        (&[1, 4097], 1),
        (&[3000], 0),
        (&[3000, 1], 1),
        (&[33, 65, 9], 1),
        (&[3, 5, 3000], 2),
        (&[40, 3, 2, 70], 3),
        (&[40, 3, 2, 70], 0),
    ];
    for (k, &(dims, axis)) in cases.iter().enumerate() {
        check_by_key_axis::<f32>(&ctx, dims, axis, 400 + k as u64);
        check_by_key_axis::<u32>(&ctx, dims, axis, 500 + k as u64);
        check_by_key_axis::<i32>(&ctx, dims, axis, 600 + k as u64);
    }
}

#[test]
fn sort_by_key_errors_poison_both_results() {
    let ctx = Context::new().unwrap();
    let x = Array::from_slice_on(&ctx, &[3.0f32, 1.0, 2.0], 3).unwrap();
    let v = Array::from_slice_on(&ctx, &[1u32, 2], 2).unwrap();
    let (k, w) = x.sort_by_key(&v);
    assert!(k.to_vec().is_err() && w.to_vec().is_err());
    let (k, w) = x.sort_by_key_axis(4, &x);
    assert!(k.to_vec().is_err() && w.to_vec().is_err());
}

/// Many duplicates, NaNs and both zeros over several tiles of the sweep kernels.
#[test]
fn sorts_floats_with_many_duplicates_nans_and_zeros() {
    let ctx = Context::new().unwrap();
    let mut rng = Rng(5 + 30_000);
    for n in [8191usize, 8192, 8193, 16_384, 30_000, 100_001] {
        let host: Vec<f32> = (0..n)
            .map(|i| if i % 97 == 5 { f32::NAN } else if i % 89 == 3 { -0.0 } else if i % 91 == 4 { 0.0 } else { (rng.next() % 2000) as f32 - 1000.0 })
            .collect();
        check_all::<f32>(&ctx, &host);
    }
}
