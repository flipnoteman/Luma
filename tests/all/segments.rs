//! `sum_segments` / `min_segments` / `max_segments` against a host reduction over the same offsets:
//! empty segments (leading, trailing, runs of them), one huge segment, power-law lengths, lengths
//! and boundaries off the tile size, a fused operand, and every element type.

use luma::{Array, AtomicElem};

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
}

/// Offsets of segments of the given lengths.
fn offsets(lengths: &[usize]) -> Vec<u32> {
    let mut at = 0u32;
    let mut out = vec![0u32];
    for &l in lengths {
        at += l as u32;
        out.push(at);
    }
    out
}

fn upload_u32(v: &[u32]) -> Array<u32> {
    Array::from_slice(v, v.len()).unwrap()
}

/// Small multiples of 1/16, so that every sum is exact in `f32` in any order.
fn quantized(rng: &mut Rng, n: usize) -> Vec<f32> {
    (0..n).map(|_| (rng.next() % 129) as f32 / 16.0 - 4.0).collect()
}

fn host<T: Copy, A: Copy>(vals: &[T], offs: &[u32], init: A, f: impl Fn(A, T) -> A) -> Vec<A> {
    offs.windows(2).map(|w| vals[w[0] as usize..w[1] as usize].iter().fold(init, |a, &b| f(a, b))).collect()
}

fn check_f32(vals: &[f32], lengths: &[usize]) {
    let offs = offsets(lengths);
    assert_eq!(*offs.last().unwrap() as usize, vals.len());
    let x = Array::from_slice(vals, vals.len()).unwrap();
    let o = upload_u32(&offs);
    assert_eq!(x.sum_segments(&o).to_vec().unwrap(), host(vals, &offs, 0.0, |a, b| a + b), "sum {:.10?}", lengths);
    assert_eq!(x.min_segments(&o).to_vec().unwrap(), host(vals, &offs, f32::INFINITY, f32::min), "min");
    assert_eq!(x.max_segments(&o).to_vec().unwrap(), host(vals, &offs, f32::NEG_INFINITY, f32::max), "max");
}

fn random_lengths(rng: &mut Rng, rows: usize, max: u32, empty_one_in: u32) -> Vec<usize> {
    (0..rows).map(|_| if rng.next().is_multiple_of(empty_one_in) { 0 } else { (rng.next() % max) as usize }).collect()
}

#[test]
fn single_tile_and_small() {
    let mut rng = Rng(1);
    for n in [1usize, 2, 7, 127, 128, 129, 1023, 1024] {
        let vals = quantized(&mut rng, n);
        check_f32(&vals, &[n]);
        check_f32(&vals, &[0, n, 0]);
        let half = n / 2;
        check_f32(&vals, &[0, 0, half, 0, n - half, 0]);
        let ones = vec![1usize; n];
        check_f32(&vals, &ones);
    }
}

#[test]
fn empty_segments_everywhere() {
    let mut rng = Rng(2);
    let vals = quantized(&mut rng, 5000);
    // Leading and trailing runs of empty segments, and many between.
    let mut lengths = vec![0usize; 3000];
    lengths.extend(random_lengths(&mut rng, 300, 40, 3));
    let total: usize = lengths.iter().sum();
    lengths.push(5000 - total);
    lengths.extend(vec![0usize; 2500]);
    check_f32(&vals, &lengths);
    // Everything in the last or the first segment.
    check_f32(&vals, &[0, 0, 0, 5000]);
    check_f32(&vals, &[5000, 0, 0, 0]);
    // Many empty segments inside one tile.
    let mut lengths = vec![1000usize];
    lengths.extend(vec![0usize; 5000]);
    lengths.push(4000);
    check_f32(&vals, &lengths);
}

#[test]
fn one_huge_segment() {
    let mut rng = Rng(3);
    for n in [1025usize, 4096, 300_000, 2_000_003] {
        let vals = quantized(&mut rng, n);
        check_f32(&vals, &[n]);
        check_f32(&vals, &[0, n, 0, 0]);
        check_f32(&vals, &[3, n - 5, 2]);
    }
}

#[test]
fn boundaries_at_tile_edges() {
    let mut rng = Rng(4);
    let n = 8 * 1024 + 5;
    let vals = quantized(&mut rng, n);
    for edge in [1023usize, 1024, 1025, 2047, 2048, 2049] {
        check_f32(&vals, &[edge, n - edge]);
        check_f32(&vals, &[edge, 0, 0, n - edge]);
        check_f32(&vals, &[edge - 1, 1, n - edge]);
    }
    // Segments of exactly a tile, of a tile and one, of a tile less one.
    for l in [1023usize, 1024, 1025] {
        let rows = n / l;
        let mut lengths = vec![l; rows];
        lengths.push(n - rows * l);
        check_f32(&vals, &lengths);
    }
}

#[test]
fn power_law_lengths() {
    let mut rng = Rng(5);
    // Lengths of 1, 2, 4, ... with fewer segments the longer: a few very long ones among many short ones.
    let mut lengths = Vec::new();
    let mut total = 0usize;
    for shift in 0..17 {
        for _ in 0..(1usize << (16 - shift)).min(4000) {
            let l = (1usize << shift) + (rng.next() % (1 << shift)) as usize;
            lengths.push(l);
            total += l;
        }
    }
    for i in (1..lengths.len()).rev() {
        lengths.swap(i, rng.next() as usize % (i + 1));
    }
    let vals = quantized(&mut rng, total);
    check_f32(&vals, &lengths);
}

#[test]
fn random_segment_sizes() {
    let mut rng = Rng(6);
    for (rows, max, empty) in [(1usize, 9000u32, 5u32), (50, 700, 4), (4000, 30, 6), (30_000, 20, 3), (1000, 3000, 7)] {
        let lengths = random_lengths(&mut rng, rows, max, empty);
        let total: usize = lengths.iter().sum();
        if total == 0 {
            continue;
        }
        let vals = quantized(&mut rng, total);
        check_f32(&vals, &lengths);
    }
}

#[test]
fn all_but_one_segment_empty() {
    let vals = [3.5f32];
    check_f32(&vals, &[0, 0, 1, 0, 0]);
    // The extrema of an empty segment are the identities.
    let x = Array::from_slice(&vals, 1).unwrap();
    let o = upload_u32(&[0, 0, 1, 1]);
    assert_eq!(x.min_segments(&o).to_vec().unwrap(), vec![f32::INFINITY, 3.5, f32::INFINITY]);
    assert_eq!(x.max_segments(&o).to_vec().unwrap(), vec![f32::NEG_INFINITY, 3.5, f32::NEG_INFINITY]);
    assert_eq!(x.sum_segments(&o).to_vec().unwrap(), vec![0.0, 3.5, 0.0]);
}

fn check_int<T: AtomicElem + Copy + PartialEq + std::fmt::Debug>(vals: &[T], lengths: &[usize], add: impl Fn(T, T) -> T, zero: T, lo: impl Fn(T, T) -> T, hi: impl Fn(T, T) -> T) {
    let offs = offsets(lengths);
    let x = Array::from_slice(vals, vals.len()).unwrap();
    let o = upload_u32(&offs);
    assert_eq!(x.sum_segments(&o).to_vec().unwrap(), host(vals, &offs, zero, &add));
    assert_eq!(x.min_segments(&o).to_vec().unwrap(), host(vals, &offs, T::MIN_IDENTITY, &lo));
    assert_eq!(x.max_segments(&o).to_vec().unwrap(), host(vals, &offs, T::MAX_IDENTITY, &hi));
}

#[test]
fn integer_segments() {
    let mut rng = Rng(7);
    let lengths = random_lengths(&mut rng, 3000, 900, 4);
    let total: usize = lengths.iter().sum();
    let u: Vec<u32> = (0..total).map(|_| rng.next()).collect();
    check_int(&u, &lengths, u32::wrapping_add, 0, u32::min, u32::max);
    let i: Vec<i32> = (0..total).map(|_| rng.next() as i32 - (1 << 30)).collect();
    check_int(&i, &lengths, i32::wrapping_add, 0, i32::min, i32::max);
}

#[test]
fn fused_operand_is_one_expression() {
    let mut rng = Rng(8);
    // A CSR matrix-vector product: values times the gathered vector, summed per row.
    let rows = 20_000usize;
    let mut lengths = random_lengths(&mut rng, rows, 30, 5);
    lengths[7] = 40_000;
    lengths[rows - 1] = 3000;
    let nnz: usize = lengths.iter().sum();
    let cols = 5000usize;
    let col: Vec<u32> = (0..nnz).map(|_| rng.next() % cols as u32).collect();
    let val = quantized(&mut rng, nnz);
    let xh = quantized(&mut rng, cols);
    let ptr = offsets(&lengths);
    let products: Vec<f32> = (0..nnz).map(|e| val[e] * xh[col[e] as usize]).collect();
    let want = host(&products, &ptr, 0.0, |a, b| a + b);
    let want_max = host(&products, &ptr, f32::NEG_INFINITY, f32::max);
    let (val, col, ptr, x) = (Array::from_slice(&val, nnz).unwrap(), upload_u32(&col), upload_u32(&ptr), Array::from_slice(&xh, cols).unwrap());
    assert_eq!((&val * x.take(&col)).sum_segments(&ptr).to_vec().unwrap(), want);
    // From a buffer instead of a pending expression.
    let p = &val * x.take(&col);
    p.eval().unwrap();
    assert_eq!(p.sum_segments(&ptr).to_vec().unwrap(), want);
    assert_eq!((&val * x.take(&col)).max_segments(&ptr).to_vec().unwrap(), want_max);
}

#[test]
fn results_are_reproducible() {
    let mut rng = Rng(9);
    // Arbitrary `f32` values: the sums round, but every run rounds the same way.
    let lengths = random_lengths(&mut rng, 2000, 700, 6);
    let total: usize = lengths.iter().sum();
    let vals: Vec<f32> = (0..total).map(|_| (rng.next() as f32 / u32::MAX as f32 - 0.5) * 100.0).collect();
    let offs = offsets(&lengths);
    let x = Array::from_slice(&vals, total).unwrap();
    let o = upload_u32(&offs);
    let first = x.sum_segments(&o).to_vec().unwrap();
    for _ in 0..3 {
        assert_eq!(x.sum_segments(&o).to_vec().unwrap(), first);
    }
    let want = host(&vals, &offs, 0.0f64, |a, b| a + f64::from(b));
    for (s, (g, w)) in first.iter().zip(&want).enumerate() {
        assert!((f64::from(*g) - w).abs() <= 1e-4 * (1.0 + w.abs()) + 1e-3, "segment {s}: {g} vs {w}");
    }
}

#[test]
fn too_few_offsets_poison() {
    let x = Array::from_slice(&[1.0f32, 2.0], 2).unwrap();
    let o = upload_u32(&[0]);
    assert!(x.sum_segments(&o).to_vec().is_err());
}

#[test]
fn inside_an_iterate_all_step() {
    let mut rng = Rng(10);
    // Power iterations on a sparse matrix, the product recorded once and replayed.
    let n = 3000usize;
    let lengths = random_lengths(&mut rng, n, 12, 5);
    let nnz: usize = lengths.iter().sum();
    let col: Vec<u32> = (0..nnz).map(|_| rng.next() % n as u32).collect();
    let val: Vec<f32> = (0..nnz).map(|_| (rng.next() % 8) as f32 / 16.0).collect();
    let ptr = offsets(&lengths);
    let x0: Vec<f32> = (0..n).map(|_| (rng.next() % 4) as f32).collect();
    let mut want = x0.clone();
    for _ in 0..4 {
        let products: Vec<f32> = (0..nnz).map(|e| val[e] * want[col[e] as usize]).collect();
        want = host(&products, &ptr, 0.0, |a, b| a + b).into_iter().map(|v| v * 0.5).collect();
    }
    let (val, col, ptr) = (Array::from_slice(&val, nnz).unwrap(), upload_u32(&col), upload_u32(&ptr));
    let x = Array::from_slice(&x0, n).unwrap();
    let [y] = luma::iterate_all([&x], 4, |[r]| [(&val * r.take(&col)).sum_segments(&ptr) * 0.5f32]);
    assert_eq!(y.to_vec().unwrap(), want);
}
