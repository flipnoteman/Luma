//! Image and filter operations (`conv2`, `conv3`, `conv2_sep`, rank and morphological filters,
//! `sobel`, `grad`, `bilateral`, `sat`, `moments`, `hist_equal`, colour conversions, `unwrap` /
//! `wrap`) against naive host computations.

use crate::common::values;
use luma::{gaussian_kernel_1d_on, gaussian_kernel_on, Array, Border, ColorSpace, Context, ConvSpec, LumaError, Padding, YccStd};

fn ctx() -> Context {
    Context::new().unwrap()
}

// ---------------------------------------------------------------------------------------------
// Host references
// ---------------------------------------------------------------------------------------------

const MODES: [Border; 4] = [Border::Zero, Border::Edge, Border::Mirror, Border::Wrap];

/// The cell of an axis of `n` that `c` stands for, `None` for a missing one.
fn remap(c: i64, n: i64, border: Border) -> Option<i64> {
    match border {
        Border::Zero | Border::Ignore => (0..n).contains(&c).then_some(c),
        Border::Edge => Some(c.clamp(0, n - 1)),
        Border::Mirror => {
            let mut r = c.rem_euclid(2 * n);
            if r >= n {
                r = 2 * n - 1 - r;
            }
            Some(r)
        }
        Border::Wrap => Some(c.rem_euclid(n)),
    }
}

/// An `[n0, n1, n2]` volume (axis 0 fastest) with a batch.
#[derive(Clone)]
struct Vol<T> {
    n: [usize; 3],
    batch: usize,
    data: Vec<T>,
}

impl<T: Copy> Vol<T> {
    fn plane(&self) -> usize {
        self.n.iter().product()
    }
    fn at(&self, b: usize, x: [i64; 3], border: Border) -> Option<T> {
        let mut i = [0usize; 3];
        for d in 0..3 {
            i[d] = remap(x[d], self.n[d] as i64, border)? as usize;
        }
        Some(self.data[b * self.plane() + i[0] + self.n[0] * (i[1] + self.n[1] * i[2])])
    }
}

fn out_len(n: usize, before: usize, after: usize, k: usize, s: usize) -> usize {
    (n + before + after - k) / s + 1
}

/// The cross-correlation `out[o] = sum_j in[o * stride + j - before] * w[j]`.
fn conv_ref(v: &Vol<f32>, w: &[f32], k: [usize; 3], stride: [usize; 3], before: [usize; 3], after: [usize; 3], border: Border) -> Vol<f64> {
    let m = [0, 1, 2].map(|d| out_len(v.n[d], before[d], after[d], k[d], stride[d]));
    let mut out = Vec::with_capacity(m.iter().product::<usize>() * v.batch);
    for b in 0..v.batch {
        for o2 in 0..m[2] {
            for o1 in 0..m[1] {
                for o0 in 0..m[0] {
                    let mut acc = 0.0f64;
                    for j2 in 0..k[2] {
                        for j1 in 0..k[1] {
                            for j0 in 0..k[0] {
                                let x = [
                                    (o0 * stride[0] + j0) as i64 - before[0] as i64,
                                    (o1 * stride[1] + j1) as i64 - before[1] as i64,
                                    (o2 * stride[2] + j2) as i64 - before[2] as i64,
                                ];
                                if let Some(val) = v.at(b, x, border) {
                                    acc += f64::from(val) * f64::from(w[j0 + k[0] * (j1 + k[1] * j2)]);
                                }
                            }
                        }
                    }
                    out.push(acc);
                }
            }
        }
    }
    Vol { n: m, batch: v.batch, data: out }
}

fn close(got: &[f32], want: &[f64], rel: f64, abs: f64, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        let tol = abs + rel * w.abs();
        assert!((f64::from(g) - w).abs() <= tol, "{what}: index {i}: got {g}, want {w} (tol {tol})");
    }
}

fn vol_f32(seed: u64, n: [usize; 3], batch: usize) -> (Vol<f32>, Vec<f32>) {
    let len = n.iter().product::<usize>() * batch;
    let data = values(len, seed, -1.0, 1.0);
    (Vol { n, batch, data: data.clone() }, data)
}

fn array_of<T: luma::ScalarType>(ctx: &Context, data: &[T], n: [usize; 3], batch: usize) -> Array<T> {
    let shape = if n[2] == 1 && batch == 1 {
        vec![n[0], n[1]]
    } else if n[2] == 1 {
        vec![n[0], n[1], batch]
    } else {
        vec![n[0], n[1], n[2], batch]
    };
    let shape: [usize; 4] = [shape[0], shape[1], *shape.get(2).unwrap_or(&1), *shape.get(3).unwrap_or(&1)];
    Array::from_slice_on(ctx, data, shape).unwrap()
}

/// `(before, after)` of the padding kinds on kernel `k`.
fn pads(padding: Padding, k: [usize; 3], axes: usize) -> ([usize; 3], [usize; 3]) {
    let (mut b, mut a) = ([0; 3], [0; 3]);
    for d in 0..axes {
        match padding {
            Padding::Valid => {}
            Padding::Same => {
                b[d] = (k[d] - 1) / 2;
                a[d] = k[d] - 1 - b[d];
            }
            Padding::Full => {
                b[d] = k[d] - 1;
                a[d] = k[d] - 1;
            }
            Padding::Explicit { before, after } => {
                b[d] = before[d];
                a[d] = after[d];
            }
        }
    }
    (b, a)
}

// ---------------------------------------------------------------------------------------------
// Convolution
// ---------------------------------------------------------------------------------------------

#[test]
fn conv2_matches_the_host_for_every_border_padding_and_stride() {
    let ctx = ctx();
    let sizes = [[1usize, 1], [2, 3], [7, 5], [31, 33], [33, 65], [64, 40]];
    let kernels = [[1usize, 1], [3, 3], [4, 4], [5, 7], [2, 1], [1, 6], [15, 15]];
    let paddings = [Padding::Same, Padding::Valid, Padding::Full, Padding::Explicit { before: [2, 0, 0], after: [1, 3, 0] }];
    let mut case = 0u64;
    for &[h, w] in &sizes {
        for &k in &kernels {
            for padding in paddings {
                for border in MODES {
                    for stride in [[1usize, 1], [2, 1], [1, 3], [2, 2]] {
                        // Cheap subsample of the product: every case of the small sizes, a quarter of the rest.
                        case += 1;
                        if h * w > 100 && !case.is_multiple_of(4) {
                            continue;
                        }
                        let k3 = [k[0], k[1], 1];
                        let (before, after) = pads(padding, k3, 2);
                        if h + before[0] + after[0] < k[0] || w + before[1] + after[1] < k[1] {
                            continue;
                        }
                        let batch = if case.is_multiple_of(7) { 3 } else { 1 };
                        let (v, data) = vol_f32(case, [h, w, 1], batch);
                        let kd = values(k[0] * k[1], case + 1000, -1.0, 1.0);
                        let x = array_of(&ctx, &data, [h, w, 1], batch);
                        let kern = Array::from_slice_on(&ctx, &kd, [k[0], k[1]]).unwrap();
                        let spec = ConvSpec::new().padding(padding).stride(stride).border(border);
                        let got = x.conv2(&kern, spec).to_vec().unwrap();
                        let want = conv_ref(&v, &kd, k3, [stride[0], stride[1], 1], before, after, border);
                        close(&got, &want.data, 1e-4, 1e-4, &format!("conv2 {h}x{w} k {k:?} {padding:?} {border:?} stride {stride:?} batch {batch}"));
                    }
                }
            }
        }
    }
}

#[test]
fn conv2_with_large_kernels() {
    let ctx = ctx();
    for k in [17usize, 31] {
        let (v, data) = vol_f32(k as u64, [70, 90, 1], 1);
        let kd = values(k * k, 3, -0.1, 0.1);
        let x = array_of(&ctx, &data, [70, 90, 1], 1);
        let kern = Array::from_slice_on(&ctx, &kd, [k, k]).unwrap();
        let got = x.conv2(&kern, ConvSpec::new().border(Border::Mirror)).to_vec().unwrap();
        let (before, after) = pads(Padding::Same, [k, k, 1], 2);
        let want = conv_ref(&v, &kd, [k, k, 1], [1; 3], before, after, Border::Mirror);
        close(&got, &want.data, 1e-4, 1e-4, &format!("conv2 {k}x{k}"));
        let ints = small_ints(k as u64, 70 * 90);
        let vi = Vol { n: [70, 90, 1], batch: 1, data: ints.clone() };
        let xi = array_of(&ctx, &ints, [70, 90, 1], 1);
        let max = |v: Vec<i32>| *v.iter().max().unwrap();
        assert_eq!(xi.maxfilt([k, k], Border::Ignore).to_vec().unwrap(), window_ref(&vi, [k, k, 1], 0, Border::Ignore, None, &max), "maxfilt {k}x{k}");
    }
}

#[test]
fn conv2_of_pending_inputs_and_batches() {
    let ctx = ctx();
    let (h, w) = (40usize, 70usize);
    let (v, data) = vol_f32(5, [h, w, 1], 6);
    let x = Array::from_slice_on(&ctx, &data, [h, w, 2, 3]).unwrap();
    let kd = values(25, 6, -1.0, 1.0);
    let kern = Array::from_slice_on(&ctx, &kd, [5, 5]).unwrap() * 2.0;
    // The operand is a pending expression; the result feeds another operation.
    let y = (&x * 0.5 + 1.0).conv2(&kern, ConvSpec::new()).sqrt_abs();
    let scaled: Vec<f32> = data.iter().map(|v| v * 0.5 + 1.0).collect();
    let v = Vol { data: scaled, ..v };
    let k2: Vec<f32> = kd.iter().map(|v| v * 2.0).collect();
    let want = conv_ref(&v, &k2, [5, 5, 1], [1; 3], [2, 2, 0], [2, 2, 0], Border::Zero);
    assert_eq!(y.shape().dims(), [h, w, 2, 3]);
    let want: Vec<f64> = want.data.iter().map(|v| v.abs().sqrt()).collect();
    close(&y.to_vec().unwrap(), &want, 1e-3, 1e-3, "pending conv2");
}

trait SqrtAbs {
    fn sqrt_abs(&self) -> Self;
}
impl SqrtAbs for Array<f32> {
    fn sqrt_abs(&self) -> Self {
        self.abs().sqrt()
    }
}

#[test]
fn conv3_matches_the_host() {
    let ctx = ctx();
    let cases = [
        ([5usize, 6, 7], [3usize, 3, 3], Padding::Same, Border::Zero, [1usize, 1, 1]),
        ([9, 4, 3], [2, 3, 2], Padding::Valid, Border::Edge, [1, 1, 1]),
        ([8, 8, 8], [3, 1, 5], Padding::Full, Border::Mirror, [2, 1, 1]),
        ([33, 5, 4], [7, 3, 3], Padding::Same, Border::Wrap, [1, 2, 2]),
        ([4, 4, 4], [1, 1, 1], Padding::Same, Border::Zero, [1, 1, 1]),
    ];
    for (i, &(n, k, padding, border, stride)) in cases.iter().enumerate() {
        for batch in [1usize, 2] {
            let (v, data) = vol_f32(i as u64 + 10, n, batch);
            let kd = values(k[0] * k[1] * k[2], i as u64 + 50, -1.0, 1.0);
            let x = Array::from_slice_on(&ctx, &data, [n[0], n[1], n[2], batch]).unwrap();
            let kern = Array::from_slice_on(&ctx, &kd, k).unwrap();
            let got = x.conv3(&kern, ConvSpec::new().padding(padding).stride(stride).border(border)).to_vec().unwrap();
            let (before, after) = pads(padding, k, 3);
            let want = conv_ref(&v, &kd, k, stride, before, after, border);
            close(&got, &want.data, 1e-4, 1e-4, &format!("conv3 {n:?} k {k:?} {padding:?} {border:?} stride {stride:?}"));
        }
    }
}

#[test]
fn separable_convolution_equals_the_outer_product_kernel() {
    let ctx = ctx();
    for (h, w, k0, k1) in [(50usize, 61usize, 5usize, 3usize), (9, 9, 4, 6), (130, 7, 11, 1), (1, 1, 1, 1)] {
        for border in MODES {
            let (v, data) = vol_f32(h as u64, [h, w, 1], 2);
            let a = values(k0, 1, -1.0, 1.0);
            let b = values(k1, 2, -1.0, 1.0);
            let x = Array::from_slice_on(&ctx, &data, [h, w, 2]).unwrap();
            let (ka, kb) = (Array::from_slice_on(&ctx, &a, k0).unwrap(), Array::from_slice_on(&ctx, &b, k1).unwrap());
            let got = x.conv2_sep(&ka, &kb, ConvSpec::new().border(border)).to_vec().unwrap();
            let mut full = vec![0.0f32; k0 * k1];
            for j1 in 0..k1 {
                for j0 in 0..k0 {
                    full[j0 + k0 * j1] = a[j0] * b[j1];
                }
            }
            let (before, after) = pads(Padding::Same, [k0, k1, 1], 2);
            let want = conv_ref(&v, &full, [k0, k1, 1], [1; 3], before, after, border);
            close(&got, &want.data, 1e-4, 1e-4, &format!("conv2_sep {h}x{w} {k0}x{k1} {border:?}"));
        }
    }
    // Strides and valid padding apply per axis.
    let (v, data) = vol_f32(3, [41, 29, 1], 1);
    let a = values(5, 4, -1.0, 1.0);
    let b = values(3, 5, -1.0, 1.0);
    let x = Array::from_slice_on(&ctx, &data, [41, 29]).unwrap();
    let (ka, kb) = (Array::from_slice_on(&ctx, &a, 5).unwrap(), Array::from_slice_on(&ctx, &b, 3).unwrap());
    let got = x.conv2_sep(&ka, &kb, ConvSpec::new().padding(Padding::Valid).stride([2, 3])).to_vec().unwrap();
    let mut full = vec![0.0f32; 15];
    for j1 in 0..3 {
        for j0 in 0..5 {
            full[j0 + 5 * j1] = a[j0] * b[j1];
        }
    }
    let want = conv_ref(&v, &full, [5, 3, 1], [2, 3, 1], [0; 3], [0; 3], Border::Zero);
    close(&got, &want.data, 1e-4, 1e-4, "conv2_sep valid stride");
}

#[test]
fn gaussian_kernels() {
    let ctx = ctx();
    for (r, c, sr, sc) in [(5usize, 5usize, 0.0f32, 0.0f32), (3, 5, 1.0, 2.0), (4, 4, 1.0, 1.0), (1, 7, 0.0, 1.5), (9, 2, 3.0, 0.0)] {
        let g = gaussian_kernel_on(&ctx, r, c, sr, sc).unwrap();
        assert_eq!(g.shape().dims(), [r, c, 1, 1]);
        let got = g.to_vec().unwrap();
        assert!((got.iter().map(|&v| f64::from(v)).sum::<f64>() - 1.0).abs() < 1e-5);
        let s = |sig: f32, n: usize| if sig > 0.0 { f64::from(sig) } else { 0.25 * n as f64 };
        let (sr, sc) = (s(sr, r), s(sc, c));
        let mut want = Vec::new();
        let mut total = 0.0;
        for j in 0..c {
            for i in 0..r {
                let (x, y) = (i as f64 - (r as f64 - 1.0) / 2.0, j as f64 - (c as f64 - 1.0) / 2.0);
                let v = (-(x * x) / (2.0 * sr * sr) - (y * y) / (2.0 * sc * sc)).exp();
                total += v;
                want.push(v);
            }
        }
        let want: Vec<f64> = want.iter().map(|v| v / total).collect();
        close(&got, &want, 1e-5, 1e-7, "gaussian_kernel");
    }
    let g1 = gaussian_kernel_1d_on(&ctx, 9, 2.0).unwrap().to_vec().unwrap();
    assert!((g1.iter().sum::<f32>() - 1.0).abs() < 1e-6);
    assert!(g1[4] > g1[3] && g1[3] > g1[2] && (g1[0] - g1[8]).abs() < 1e-9);
    // A Gaussian blur of a constant image is the constant away from the border.
    let img = Array::from_slice_on(&ctx, &vec![3.0f32; 20 * 20], [20, 20]).unwrap();
    let k = gaussian_kernel_on(&ctx, 5, 5, 1.0, 1.0).unwrap();
    let blur = img.conv2(&k, ConvSpec::new().border(Border::Edge)).to_vec().unwrap();
    for v in blur {
        assert!((v - 3.0).abs() < 1e-5);
    }
}

// ---------------------------------------------------------------------------------------------
// Rank and morphological filters
// ---------------------------------------------------------------------------------------------

/// Host rank filter: the window's `pick` of the sorted cells (`pick(len)` the index), cells
/// outside the array by `border`.
fn window_ref<T: Copy + PartialOrd + std::fmt::Debug>(v: &Vol<T>, k: [usize; 3], zero: T, border: Border, mask: Option<&[T]>, reduce: &dyn Fn(Vec<T>) -> T) -> Vec<T> {
    let before = k.map(|k| k / 2);
    let mut out = Vec::new();
    for b in 0..v.batch {
        for o2 in 0..v.n[2] {
            for o1 in 0..v.n[1] {
                for o0 in 0..v.n[0] {
                    let mut cells = Vec::new();
                    for j2 in 0..k[2] {
                        for j1 in 0..k[1] {
                            for j0 in 0..k[0] {
                                if let Some(m) = mask
                                    && m[j0 + k[0] * (j1 + k[1] * j2)] == zero
                                {
                                    continue;
                                }
                                let x = [o0 as i64 + j0 as i64 - before[0] as i64, o1 as i64 + j1 as i64 - before[1] as i64, o2 as i64 + j2 as i64 - before[2] as i64];
                                match v.at(b, x, border) {
                                    Some(c) => cells.push(c),
                                    None if border == Border::Ignore => {}
                                    None => cells.push(zero),
                                }
                            }
                        }
                    }
                    out.push(reduce(cells));
                }
            }
        }
    }
    out
}

fn sorted<T: PartialOrd + Copy>(mut v: Vec<T>) -> Vec<T> {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v
}

fn median_of<T: PartialOrd + Copy>(v: Vec<T>) -> T {
    sorted(v.clone())[v.len() / 2]
}

/// Random cells with many duplicates (small integers) as every supported type.
fn small_ints(seed: u64, n: usize) -> Vec<i32> {
    values(n, seed, -20.0, 20.0).iter().map(|v| v.round() as i32).collect()
}

#[test]
fn medfilt_matches_the_host_for_every_type_border_and_window() {
    let ctx = ctx();
    let cases = [
        ([9usize, 11usize], [3usize, 3usize]),
        ([8, 7], [5, 5]),
        ([20, 13], [3, 5]),
        ([12, 12], [7, 7]),
        ([13, 11], [9, 9]),
        ([10, 9], [4, 4]),
        ([6, 5], [1, 1]),
        ([40, 40], [5, 1]),
        ([3, 3], [5, 5]),
        ([1, 1], [3, 3]),
    ];
    for (i, &([h, w], k)) in cases.iter().enumerate() {
        for border in MODES {
            let ints = small_ints(i as u64, h * w * 2);
            let k3 = [k[0], k[1], 1];
            // i32
            let vi = Vol { n: [h, w, 1], batch: 2, data: ints.clone() };
            let x = array_of(&ctx, &ints, [h, w, 1], 2);
            assert_eq!(x.medfilt2(k, border).to_vec().unwrap(), window_ref(&vi, k3, 0, border, None, &median_of), "i32 {h}x{w} {k:?} {border:?}");
            // u32
            let us: Vec<u32> = ints.iter().map(|&v| (v + 20) as u32).collect();
            let vu = Vol { n: [h, w, 1], batch: 2, data: us.clone() };
            let x = array_of(&ctx, &us, [h, w, 1], 2);
            assert_eq!(x.medfilt2(k, border).to_vec().unwrap(), window_ref(&vu, k3, 0, border, None, &median_of), "u32 {h}x{w} {k:?} {border:?}");
            // f32, with infinities and signed zeros mixed in
            let mut fs: Vec<f32> = values(h * w * 2, i as u64 + 7, -3.0, 3.0);
            if fs.len() > 4 {
                fs[1] = f32::INFINITY;
                fs[3] = f32::NEG_INFINITY;
            }
            let vf = Vol { n: [h, w, 1], batch: 2, data: fs.clone() };
            let x = array_of(&ctx, &fs, [h, w, 1], 2);
            assert_eq!(x.medfilt2(k, border).to_vec().unwrap(), window_ref(&vf, k3, 0.0, border, None, &median_of), "f32 {h}x{w} {k:?} {border:?}");
        }
    }
}

#[test]
fn medfilt1_filters_along_axis_zero() {
    let ctx = ctx();
    for len in [1usize, 3, 5, 8, 15] {
        for border in MODES {
            let data = values(70 * 4, len as u64, -1.0, 1.0);
            let v = Vol { n: [70, 4, 1], batch: 1, data: data.clone() };
            let x = Array::from_slice_on(&ctx, &data, [70, 4]).unwrap();
            let got = x.medfilt1(len, border).to_vec().unwrap();
            assert_eq!(got, window_ref(&v, [len, 1, 1], 0.0, border, None, &median_of), "medfilt1 {len} {border:?}");
        }
    }
}

#[test]
fn min_and_max_filters_match_the_host() {
    let ctx = ctx();
    let cases = [([9usize, 11usize], [3usize, 3usize]), ([30, 17], [5, 5]), ([20, 21], [3, 9]), ([40, 33], [7, 4]), ([5, 4], [9, 9]), ([64, 64], [1, 5])];
    for (i, &([h, w], k)) in cases.iter().enumerate() {
        for border in [Border::Ignore, Border::Zero, Border::Edge, Border::Mirror, Border::Wrap] {
            let k3 = [k[0], k[1], 1];
            let ints = small_ints(i as u64 + 100, h * w * 2);
            let vi = Vol { n: [h, w, 1], batch: 2, data: ints.clone() };
            let x = array_of(&ctx, &ints, [h, w, 1], 2);
            let min = |v: Vec<i32>| *v.iter().min().unwrap();
            let max = |v: Vec<i32>| *v.iter().max().unwrap();
            assert_eq!(x.minfilt(k, border).to_vec().unwrap(), window_ref(&vi, k3, 0, border, None, &min), "min i32 {h}x{w} {k:?} {border:?}");
            assert_eq!(x.maxfilt(k, border).to_vec().unwrap(), window_ref(&vi, k3, 0, border, None, &max), "max i32 {h}x{w} {k:?} {border:?}");
            let us: Vec<u32> = ints.iter().map(|&v| (v + 20) as u32).collect();
            let vu = Vol { n: [h, w, 1], batch: 2, data: us.clone() };
            let x = array_of(&ctx, &us, [h, w, 1], 2);
            let min = |v: Vec<u32>| *v.iter().min().unwrap();
            let max = |v: Vec<u32>| *v.iter().max().unwrap();
            assert_eq!(x.minfilt(k, border).to_vec().unwrap(), window_ref(&vu, k3, 0, border, None, &min), "min u32 {h}x{w} {k:?} {border:?}");
            assert_eq!(x.maxfilt(k, border).to_vec().unwrap(), window_ref(&vu, k3, 0, border, None, &max), "max u32 {h}x{w} {k:?} {border:?}");
            let fs = values(h * w * 2, i as u64 + 3, -5.0, 5.0);
            let vf = Vol { n: [h, w, 1], batch: 2, data: fs.clone() };
            let x = array_of(&ctx, &fs, [h, w, 1], 2);
            let min = |v: Vec<f32>| v.iter().copied().fold(f32::INFINITY, f32::min);
            let max = |v: Vec<f32>| v.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            assert_eq!(x.minfilt(k, border).to_vec().unwrap(), window_ref(&vf, k3, 0.0, border, None, &min), "min f32 {h}x{w} {k:?} {border:?}");
            assert_eq!(x.maxfilt(k, border).to_vec().unwrap(), window_ref(&vf, k3, 0.0, border, None, &max), "max f32 {h}x{w} {k:?} {border:?}");
        }
    }
}

#[test]
fn min_and_max_filters_of_three_dimensions() {
    let ctx = ctx();
    let (n, k) = ([9usize, 7, 6], [3usize, 3, 2]);
    let ints = small_ints(9, n[0] * n[1] * n[2] * 2);
    let v = Vol { n, batch: 2, data: ints.clone() };
    let x = array_of(&ctx, &ints, n, 2);
    for border in [Border::Ignore, Border::Edge, Border::Wrap] {
        let min = |v: Vec<i32>| *v.iter().min().unwrap();
        let max = |v: Vec<i32>| *v.iter().max().unwrap();
        assert_eq!(x.minfilt3(k, border).to_vec().unwrap(), window_ref(&v, k, 0, border, None, &min), "minfilt3 {border:?}");
        assert_eq!(x.maxfilt3(k, border).to_vec().unwrap(), window_ref(&v, k, 0, border, None, &max), "maxfilt3 {border:?}");
    }
}

#[test]
fn dilate_and_erode_match_the_host_for_arbitrary_masks() {
    let ctx = ctx();
    let masks: [([usize; 2], Vec<i32>); 4] = [
        ([3, 3], vec![0, 1, 0, 1, 1, 1, 0, 1, 0]),
        ([3, 3], vec![1, 1, 1, 0, 1, 0, 0, 0, 0]),
        ([5, 3], vec![1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1]),
        ([2, 4], vec![1, 0, 0, 1, 1, 1, 0, 0]),
    ];
    for (i, (md, mask)) in masks.iter().enumerate() {
        for &[h, w] in &[[1usize, 1usize], [7, 9], [32, 33], [50, 20]] {
            let ints = small_ints(i as u64 + 200, h * w * 3);
            let v = Vol { n: [h, w, 1], batch: 3, data: ints.clone() };
            let x = array_of(&ctx, &ints, [h, w, 1], 3);
            let m = Array::from_slice_on(&ctx, mask, *md).unwrap();
            let k3 = [md[0], md[1], 1];
            let min = |c: Vec<i32>| c.iter().copied().min().unwrap_or(i32::MAX);
            let max = |c: Vec<i32>| c.iter().copied().max().unwrap_or(i32::MIN);
            assert_eq!(x.dilate(&m).to_vec().unwrap(), window_ref(&v, k3, 0, Border::Ignore, Some(mask), &max), "dilate i32 {h}x{w} mask {md:?}");
            assert_eq!(x.erode(&m).to_vec().unwrap(), window_ref(&v, k3, 0, Border::Ignore, Some(mask), &min), "erode i32 {h}x{w} mask {md:?}");
            for border in [Border::Zero, Border::Mirror] {
                assert_eq!(x.dilate_with(&m, border).to_vec().unwrap(), window_ref(&v, k3, 0, border, Some(mask), &max), "dilate_with {border:?}");
                assert_eq!(x.erode_with(&m, border).to_vec().unwrap(), window_ref(&v, k3, 0, border, Some(mask), &min), "erode_with {border:?}");
            }
            // f32 masks of 1.0 / 0.0 on f32 images
            let fs = values(h * w * 3, i as u64, -1.0, 1.0);
            let vf = Vol { n: [h, w, 1], batch: 3, data: fs.clone() };
            let xf = array_of(&ctx, &fs, [h, w, 1], 3);
            let mf: Vec<f32> = mask.iter().map(|&v| v as f32).collect();
            let mfa = Array::from_slice_on(&ctx, &mf, *md).unwrap();
            let minf = |c: Vec<f32>| c.iter().copied().fold(f32::INFINITY, f32::min);
            let maxf = |c: Vec<f32>| c.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            assert_eq!(xf.dilate(&mfa).to_vec().unwrap(), window_ref(&vf, k3, 0.0, Border::Ignore, Some(&mf), &maxf), "dilate f32");
            assert_eq!(xf.erode(&mfa).to_vec().unwrap(), window_ref(&vf, k3, 0.0, Border::Ignore, Some(&mf), &minf), "erode f32");
        }
    }
}

#[test]
fn dilate3_and_erode3_match_the_host() {
    let ctx = ctx();
    let md = [3usize, 3, 3];
    let mask: Vec<i32> = (0..27).map(|i| i32::from(i % 3 == 0 || i == 13)).collect();
    for n in [[4usize, 4, 4], [9, 7, 5], [33, 3, 2]] {
        let ints = small_ints(n[0] as u64, n[0] * n[1] * n[2] * 2);
        let v = Vol { n, batch: 2, data: ints.clone() };
        let x = array_of(&ctx, &ints, n, 2);
        let m = Array::from_slice_on(&ctx, &mask, md).unwrap();
        let min = |c: Vec<i32>| c.iter().copied().min().unwrap_or(i32::MAX);
        let max = |c: Vec<i32>| c.iter().copied().max().unwrap_or(i32::MIN);
        assert_eq!(x.dilate3(&m).to_vec().unwrap(), window_ref(&v, md, 0, Border::Ignore, Some(&mask), &max), "dilate3 {n:?}");
        assert_eq!(x.erode3(&m).to_vec().unwrap(), window_ref(&v, md, 0, Border::Ignore, Some(&mask), &min), "erode3 {n:?}");
    }
}

// ---------------------------------------------------------------------------------------------
// Derivatives
// ---------------------------------------------------------------------------------------------

#[test]
fn sobel_and_gradient_match_the_host() {
    let ctx = ctx();
    for (h, w) in [(1usize, 1usize), (2, 3), (9, 8), (33, 65)] {
        let (v, data) = vol_f32(h as u64 + 3, [h, w, 1], 2);
        let x = array_of(&ctx, &data, [h, w, 1], 2);
        for border in [Border::Edge, Border::Zero, Border::Mirror, Border::Wrap] {
            let (dx, dy) = x.sobel_with(border);
            let a = [1.0f32, 0.0, -1.0];
            let _ = a;
            // d0 = (taps (-1, 0, 1) along 0) x (taps (1, 2, 1) along 1)
            let k0: Vec<f32> = vec![-1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -1.0, 0.0, 1.0];
            let mut k1 = vec![0.0f32; 9];
            for j1 in 0..3 {
                for j0 in 0..3 {
                    k1[j0 + 3 * j1] = k0[j1 + 3 * j0];
                }
            }
            let want0 = conv_ref(&v, &k0, [3, 3, 1], [1; 3], [1, 1, 0], [1, 1, 0], border);
            let want1 = conv_ref(&v, &k1, [3, 3, 1], [1; 3], [1, 1, 0], [1, 1, 0], border);
            close(&dx.to_vec().unwrap(), &want0.data, 1e-5, 1e-5, &format!("sobel d0 {h}x{w} {border:?}"));
            close(&dy.to_vec().unwrap(), &want1.data, 1e-5, 1e-5, &format!("sobel d1 {h}x{w} {border:?}"));
        }
        let (dx, dy) = x.sobel();
        let mag = x.sobel_magnitude().to_vec().unwrap();
        let fast = x.sobel_magnitude_fast().to_vec().unwrap();
        let (dx, dy) = (dx.to_vec().unwrap(), dy.to_vec().unwrap());
        let want: Vec<f64> = dx.iter().zip(&dy).map(|(&a, &b)| (f64::from(a).powi(2) + f64::from(b).powi(2)).sqrt()).collect();
        close(&mag, &want, 1e-5, 1e-5, "sobel_magnitude");
        let want: Vec<f64> = dx.iter().zip(&dy).map(|(&a, &b)| f64::from(a.abs() + b.abs())).collect();
        close(&fast, &want, 1e-5, 1e-5, "sobel_magnitude_fast");

        let (g0, g1) = x.grad();
        let (g0, g1) = (g0.to_vec().unwrap(), g1.to_vec().unwrap());
        let mut want0 = Vec::new();
        let mut want1 = Vec::new();
        for b in 0..2 {
            for j in 0..w {
                for i in 0..h {
                    let at = |i: usize, j: usize| f64::from(data[b * h * w + i + h * j]);
                    want0.push(if h == 1 {
                        0.0
                    } else if i == 0 {
                        at(1, j) - at(0, j)
                    } else if i == h - 1 {
                        at(h - 1, j) - at(h - 2, j)
                    } else {
                        (at(i + 1, j) - at(i - 1, j)) / 2.0
                    });
                    want1.push(if w == 1 {
                        0.0
                    } else if j == 0 {
                        at(i, 1) - at(i, 0)
                    } else if j == w - 1 {
                        at(i, w - 1) - at(i, w - 2)
                    } else {
                        (at(i, j + 1) - at(i, j - 1)) / 2.0
                    });
                }
            }
        }
        close(&g0, &want0, 1e-5, 1e-5, &format!("grad d0 {h}x{w}"));
        close(&g1, &want1, 1e-5, 1e-5, &format!("grad d1 {h}x{w}"));
    }
}

// ---------------------------------------------------------------------------------------------
// Bilateral
// ---------------------------------------------------------------------------------------------

#[test]
fn bilateral_matches_the_host() {
    let ctx = ctx();
    for (h, w, ss, cs) in [(8usize, 8usize, 2.0f32, 20.0f32), (17, 33, 1.0, 0.5), (40, 9, 3.0, 2.0), (1, 5, 1.0, 1.0)] {
        let data: Vec<f32> = values(h * w * 2, h as u64, 0.0, 100.0);
        let x = array_of(&ctx, &data, [h, w, 1], 2);
        let got = x.bilateral(ss, cs).to_vec().unwrap();
        let r = ((ss * 1.5) as i64).max(1);
        let mut want = Vec::new();
        for b in 0..2 {
            for j in 0..w as i64 {
                for i in 0..h as i64 {
                    let px = |i: i64, j: i64| f64::from(data[b * h * w + i.clamp(0, h as i64 - 1) as usize + h * j.clamp(0, w as i64 - 1) as usize]);
                    let c = px(i, j);
                    let (mut acc, mut norm) = (0.0, 0.0);
                    for dj in -r..=r {
                        for di in -r..=r {
                            let v = px(i + di, j + dj);
                            let e = (-((di * di + dj * dj) as f64) / (2.0 * f64::from(ss).powi(2)) - (v - c).powi(2) / (2.0 * f64::from(cs).powi(2))).exp();
                            acc += v * e;
                            norm += e;
                        }
                    }
                    want.push(acc / norm);
                }
            }
        }
        close(&got, &want, 1e-3, 1e-3, &format!("bilateral {h}x{w} {ss} {cs}"));
    }
    // Colour: the range distance is joint over the channels.
    let (h, w) = (12usize, 15usize);
    let data: Vec<f32> = values(h * w * 3 * 2, 77, 0.0, 100.0);
    let x = Array::from_slice_on(&ctx, &data, [h, w, 3, 2]).unwrap();
    let (ss, cs) = (1.5f32, 25.0f32);
    let got = x.bilateral_color(ss, cs).to_vec().unwrap();
    let r = ((ss * 1.5) as i64).max(1);
    let mut want = vec![0.0f64; data.len()];
    for b in 0..2 {
        for j in 0..w as i64 {
            for i in 0..h as i64 {
                let px = |i: i64, j: i64, c: usize| f64::from(data[(b * 3 + c) * h * w + i.clamp(0, h as i64 - 1) as usize + h * j.clamp(0, w as i64 - 1) as usize]);
                let centre = [px(i, j, 0), px(i, j, 1), px(i, j, 2)];
                let mut acc = [0.0; 3];
                let mut norm = 0.0;
                for dj in -r..=r {
                    for di in -r..=r {
                        let v = [px(i + di, j + dj, 0), px(i + di, j + dj, 1), px(i + di, j + dj, 2)];
                        let d2: f64 = (0..3).map(|c| (v[c] - centre[c]).powi(2)).sum();
                        let e = (-((di * di + dj * dj) as f64) / (2.0 * f64::from(ss).powi(2)) - d2 / (2.0 * f64::from(cs).powi(2))).exp();
                        for c in 0..3 {
                            acc[c] += v[c] * e;
                        }
                        norm += e;
                    }
                }
                for c in 0..3 {
                    want[(b * 3 + c) * h * w + i as usize + h * j as usize] = acc[c] / norm;
                }
            }
        }
    }
    close(&got, &want, 1e-3, 1e-3, "bilateral_color");
    assert!(matches!(x.slice(2, 0..2).bilateral_color(1.0, 1.0).to_vec(), Err(LumaError::Op { .. })));
}

// ---------------------------------------------------------------------------------------------
// Summed-area table, moments, histogram equalisation
// ---------------------------------------------------------------------------------------------

#[test]
fn sat_is_the_double_prefix_sum() {
    let ctx = ctx();
    let (h, w) = (37usize, 29usize);
    let ints: Vec<i32> = small_ints(1, h * w * 2);
    let x = array_of(&ctx, &ints, [h, w, 1], 2);
    let got = x.sat().to_vec().unwrap();
    let mut want = vec![0i32; ints.len()];
    for b in 0..2 {
        for j in 0..w {
            for i in 0..h {
                let mut s = 0;
                for jj in 0..=j {
                    for ii in 0..=i {
                        s += ints[b * h * w + ii + h * jj];
                    }
                }
                want[b * h * w + i + h * j] = s;
            }
        }
    }
    assert_eq!(got, want);
    let us: Vec<u32> = ints.iter().map(|&v| (v + 20) as u32).collect();
    let got = array_of(&ctx, &us, [h, w, 1], 2).sat().to_vec().unwrap();
    for b in 0..2 {
        assert_eq!(got[b * h * w + h * w - 1], us[b * h * w..(b + 1) * h * w].iter().sum::<u32>());
    }
}

#[test]
fn sat_at_ragged_sizes_and_chunk_boundaries() {
    let ctx = ctx();
    // Lines past one segment (1024 cells) of axis 0, past one chunk (64 cells) of axis 1, and 1-wide.
    for (h, w, planes) in [(1usize, 1usize, 1usize), (1, 200, 2), (5, 64, 1), (5, 65, 3), (1025, 3, 1), (2100, 130, 1), (300, 129, 2), (1024, 128, 1), (1023, 191, 1)] {
        let ints = small_ints((h * w) as u64, h * w * planes);
        let x = Array::from_slice_on(&ctx, &ints, [h, w, planes]).unwrap();
        let got = x.sat().to_vec().unwrap();
        let mut want = vec![0i64; ints.len()];
        for p in 0..planes {
            let at = |i: usize, j: usize| p * h * w + i + h * j;
            for j in 0..w {
                for i in 0..h {
                    let mut s = i64::from(ints[at(i, j)]);
                    if i > 0 {
                        s += want[at(i - 1, j)];
                    }
                    if j > 0 {
                        s += want[at(i, j - 1)];
                    }
                    if i > 0 && j > 0 {
                        s -= want[at(i - 1, j - 1)];
                    }
                    want[at(i, j)] = s;
                }
            }
        }
        let want: Vec<i32> = want.iter().map(|&v| v as i32).collect();
        assert_eq!(got, want, "i32 {h}x{w}x{planes}");
        let fs = values(h * w * planes, h as u64, -1.0, 1.0);
        let xf = Array::from_slice_on(&ctx, &fs, [h, w, planes]).unwrap();
        let sat = xf.sat().to_vec().unwrap();
        for p in 0..planes {
            let mut total = 0.0f64;
            for k in 0..h * w {
                total += f64::from(fs[p * h * w + k]);
            }
            let last = f64::from(sat[p * h * w + h * w - 1]);
            assert!((last - total).abs() < 1e-3 * (1.0 + (h * w) as f64).sqrt(), "f32 {h}x{w}: {last} vs {total}");
        }
    }
}

#[test]
fn moments_match_the_host() {
    let ctx = ctx();
    let (h, w) = (23usize, 17usize);
    let data: Vec<f32> = values(h * w * 3, 4, 0.0, 1.0);
    let x = Array::from_slice_on(&ctx, &data, [h, w, 3]).unwrap();
    let m = x.moments();
    assert_eq!(m.shape().dims(), [4, 1, 3, 1]);
    let got = m.to_vec().unwrap();
    for b in 0..3 {
        let (mut m00, mut m01, mut m10, mut m11) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
        for j in 0..w {
            for i in 0..h {
                let v = f64::from(data[b * h * w + i + h * j]);
                m00 += v;
                m01 += v * i as f64;
                m10 += v * j as f64;
                m11 += v * (i * j) as f64;
            }
        }
        close(&got[b * 4..b * 4 + 4], &[m00, m01, m10, m11], 1e-4, 1e-3, "moments");
    }
}

#[test]
fn hist_equal_matches_the_host() {
    let ctx = ctx();
    let (h, w) = (31usize, 20usize);
    let bins = 16usize;
    let us: Vec<u32> = values(h * w, 5, 0.0, bins as f32 - 0.01).iter().enumerate().map(|(i, &v)| if i % 5 == 0 { 3 } else { v as u32 }).collect();
    let x = Array::from_slice_on(&ctx, &us, [h, w]).unwrap();
    let hist = x.to_f32().histogram(bins, 0.0, bins as f32);
    let hv = hist.to_vec().unwrap();
    let mut cdf = Vec::new();
    let mut s = 0.0f64;
    for &c in &hv {
        s += f64::from(c);
        cdf.push(s);
    }
    let (lo, hi) = (cdf.iter().copied().fold(f64::INFINITY, f64::min), cdf.iter().copied().fold(f64::NEG_INFINITY, f64::max));
    let norm: Vec<f64> = cdf.iter().map(|c| (c - lo) * (bins as f64 - 1.0) / (hi - lo)).collect();
    let got = x.hist_equal(&hist).to_vec().unwrap();
    for (i, (&g, &u)) in got.iter().zip(&us).enumerate() {
        let want = norm[u as usize] as f32 as u32;
        assert!((i64::from(g) - i64::from(want)).abs() <= 1, "u32 hist_equal at {i}: {g} vs {want}");
    }
    let got = x.to_f32().hist_equal(&hist).to_vec().unwrap();
    for (i, (&g, &u)) in got.iter().zip(&us).enumerate() {
        let want = norm[u as usize];
        assert!((f64::from(g) - want).abs() < 1e-2 + 1e-4 * want.abs(), "f32 hist_equal at {i}: {g} vs {want}");
    }
    let got = x.to_i32().hist_equal(&hist).to_vec().unwrap();
    for (&g, &u) in got.iter().zip(&us) {
        assert!((i64::from(g) - norm[u as usize] as i64).abs() <= 1);
    }
}

// ---------------------------------------------------------------------------------------------
// Colour
// ---------------------------------------------------------------------------------------------

fn rgb_to_hsv(r: f64, g: f64, b: f64) -> [f64; 3] {
    let hi = r.max(g).max(b);
    let lo = r.min(g).min(b);
    let d = hi - lo;
    let mut h = 0.0;
    if d > 0.0 {
        h = if hi == r {
            let h = (g - b) / d;
            if h < 0.0 { h + 6.0 } else { h }
        } else if hi == g {
            (b - r) / d + 2.0
        } else {
            (r - g) / d + 4.0
        };
    }
    [h / 6.0, if hi > 0.0 { d / hi } else { 0.0 }, hi]
}

fn ycc_ref(std: YccStd, r: f64, g: f64, b: f64) -> [f64; 3] {
    let (kr, kb) = match std {
        YccStd::Bt601 => (0.299, 0.114),
        YccStd::Bt709 => (0.2126, 0.0722),
        YccStd::Bt2020 => (0.2627, 0.0593),
    };
    let kg = 1.0 - kr - kb;
    let y = kr * r + kg * g + kb * b;
    [16.0 + 219.0 * y, 128.0 + 224.0 * (b - y) / (2.0 * (1.0 - kb)), 128.0 + 224.0 * (r - y) / (2.0 * (1.0 - kr))]
}

#[test]
fn colour_conversions_match_the_host() {
    let ctx = ctx();
    let (h, w, n) = (13usize, 11usize, 2usize);
    let p = h * w;
    let mut data = values(p * 3 * n, 8, 0.0, 1.0);
    // Greys, primaries and black pixels exercise the branches.
    for (i, &(r, g, b)) in [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (0.5, 0.5, 0.5), (0.1, 0.5, 0.5)].iter().enumerate() {
        data[i] = r;
        data[p + i] = g;
        data[2 * p + i] = b;
    }
    let x = Array::from_slice_on(&ctx, &data, [h, w, 3, n]).unwrap();
    let px = |img: usize, i: usize| [f64::from(data[img * 3 * p + i]), f64::from(data[img * 3 * p + p + i]), f64::from(data[img * 3 * p + 2 * p + i])];

    let gray = x.rgb2gray();
    assert_eq!(gray.shape().dims(), [h, w, 1, n]);
    let got = gray.to_vec().unwrap();
    let want: Vec<f64> = (0..n * p).map(|i| {
        let c = px(i / p, i % p);
        0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
    }).collect();
    close(&got, &want, 1e-5, 1e-6, "rgb2gray");
    let got = x.rgb2gray_with(1.0, 2.0, 3.0).to_vec().unwrap();
    let want: Vec<f64> = (0..n * p).map(|i| {
        let c = px(i / p, i % p);
        c[0] + 2.0 * c[1] + 3.0 * c[2]
    }).collect();
    close(&got, &want, 1e-5, 1e-6, "rgb2gray_with");

    let three = gray.gray2rgb(1.0, 0.5, 2.0);
    assert_eq!(three.shape().dims(), [h, w, 3, n]);
    let got = three.to_vec().unwrap();
    for img in 0..n {
        for i in 0..p {
            let g = f64::from(gray.to_vec().unwrap()[img * p + i]);
            for (c, f) in [1.0, 0.5, 2.0].iter().enumerate() {
                assert!((f64::from(got[img * 3 * p + c * p + i]) - g * f).abs() < 1e-6);
            }
        }
    }

    let hsv = x.rgb2hsv().to_vec().unwrap();
    for img in 0..n {
        for i in 0..p {
            let want = rgb_to_hsv(px(img, i)[0], px(img, i)[1], px(img, i)[2]);
            for c in 0..3 {
                let g = f64::from(hsv[img * 3 * p + c * p + i]);
                assert!((g - want[c]).abs() < 1e-5, "rgb2hsv img {img} px {i} ch {c}: {g} vs {}", want[c]);
            }
        }
    }
    let back = x.rgb2hsv().hsv2rgb().to_vec().unwrap();
    close(&back, &data.iter().map(|&v| f64::from(v)).collect::<Vec<_>>(), 1e-4, 1e-5, "hsv round trip");

    for std in [YccStd::Bt601, YccStd::Bt709, YccStd::Bt2020] {
        let ycc = x.rgb2ycbcr(std).to_vec().unwrap();
        for img in 0..n {
            for i in 0..p {
                let want = ycc_ref(std, px(img, i)[0], px(img, i)[1], px(img, i)[2]);
                for c in 0..3 {
                    let g = f64::from(ycc[img * 3 * p + c * p + i]);
                    assert!((g - want[c]).abs() < 1e-3, "{std:?} img {img} px {i} ch {c}: {g} vs {}", want[c]);
                }
            }
        }
        let back = x.rgb2ycbcr(std).ycbcr2rgb(std).to_vec().unwrap();
        close(&back, &data.iter().map(|&v| f64::from(v)).collect::<Vec<_>>(), 1e-4, 1e-4, &format!("{std:?} round trip"));
    }

    // color_space chains the conversions through RGB.
    let via = x.color_space(ColorSpace::Rgb, ColorSpace::Hsv).to_vec().unwrap();
    assert_eq!(via, x.rgb2hsv().to_vec().unwrap());
    let grey_from_hsv = x.rgb2hsv().color_space(ColorSpace::Hsv, ColorSpace::Gray).to_vec().unwrap();
    close(&grey_from_hsv, &want_gray(&data, p, n), 1e-4, 1e-5, "hsv to gray");
    let id = x.color_space(ColorSpace::YCbCr(YccStd::Bt709), ColorSpace::YCbCr(YccStd::Bt709)).to_vec().unwrap();
    assert_eq!(id.len(), data.len());

    // Wrong channel count poisons.
    assert!(matches!(gray.rgb2hsv().to_vec(), Err(LumaError::Op { .. })));
    assert!(matches!(x.gray2rgb(1.0, 1.0, 1.0).to_vec(), Err(LumaError::Op { .. })));
}

fn want_gray(data: &[f32], p: usize, n: usize) -> Vec<f64> {
    (0..n * p)
        .map(|i| {
            let (img, k) = (i / p, i % p);
            let c = |c: usize| f64::from(data[img * 3 * p + c * p + k]);
            0.2126 * c(0) + 0.7152 * c(1) + 0.0722 * c(2)
        })
        .collect()
}

// ---------------------------------------------------------------------------------------------
// unwrap / wrap
// ---------------------------------------------------------------------------------------------

#[test]
fn unwrap_and_wrap_match_the_host() {
    let ctx = ctx();
    let cases = [
        ([5usize, 6usize], [2usize, 3usize], [1usize, 2usize], [0usize, 1usize]),
        ([5, 6], [2, 3], [2, 1], [1, 0]),
        ([9, 9], [3, 3], [1, 1], [1, 1]),
        ([1, 1], [1, 1], [1, 1], [0, 0]),
        ([16, 7], [4, 2], [4, 2], [0, 0]),
        ([10, 12], [5, 4], [3, 3], [2, 2]),
    ];
    for (n, win, stride, pad) in cases {
        let data = values(n[0] * n[1] * 2 * 2, n[0] as u64, 0.0, 10.0);
        let x = Array::from_slice_on(&ctx, &data, [n[0], n[1], 2, 2]).unwrap();
        let nw = [0, 1].map(|d| (n[d] + 2 * pad[d] - win[d]) / stride[d] + 1);
        let (k, wins) = (win[0] * win[1], nw[0] * nw[1]);
        for column in [true, false] {
            let u = x.unwrap(win, stride, pad, column);
            let dims = u.shape().dims();
            assert_eq!(dims, if column { [k, wins, 2, 2] } else { [wins, k, 2, 2] }, "unwrap dims");
            let got = u.to_vec().unwrap();
            for plane in 0..4 {
                for wi in 0..wins {
                    for cell in 0..k {
                        let a0 = ((wi % nw[0]) * stride[0] + cell % win[0]) as i64 - pad[0] as i64;
                        let a1 = ((wi / nw[0]) * stride[1] + cell / win[0]) as i64 - pad[1] as i64;
                        let want = if (0..n[0] as i64).contains(&a0) && (0..n[1] as i64).contains(&a1) { data[plane * n[0] * n[1] + a0 as usize + n[0] * a1 as usize] } else { 0.0 };
                        let at = if column { cell + k * (wi + wins * plane) } else { wi + wins * (cell + k * plane) };
                        assert_eq!(got[at], want, "unwrap {n:?} {win:?} {stride:?} {pad:?} column {column} plane {plane} win {wi} cell {cell}");
                    }
                }
            }
            // wrap sums the windows back: every cell becomes `value * (number of windows covering it)`.
            let wr = u.wrap(n, win, stride, pad, column);
            assert_eq!(wr.shape().dims(), [n[0], n[1], 2, 2]);
            let back = wr.to_vec().unwrap();
            for plane in 0..4 {
                for j in 0..n[1] {
                    for i in 0..n[0] {
                        let mut cover = 0;
                        for w1 in 0..nw[1] {
                            for w0 in 0..nw[0] {
                                let (s0, s1) = (w0 * stride[0], w1 * stride[1]);
                                let (x0, x1) = (i + pad[0], j + pad[1]);
                                if x0 >= s0 && x0 < s0 + win[0] && x1 >= s1 && x1 < s1 + win[1] {
                                    cover += 1;
                                }
                            }
                        }
                        let want = data[plane * n[0] * n[1] + i + n[0] * j] * cover as f32;
                        let g = back[plane * n[0] * n[1] + i + n[0] * j];
                        assert!((g - want).abs() < 1e-4, "wrap {n:?} {win:?} {stride:?} {pad:?}: {g} vs {want}");
                    }
                }
            }
        }
    }
    // A convolution as a matrix product of the unwrapped windows (im2col) agrees with conv2.
    let (h, w) = (12usize, 10usize);
    let data = values(h * w, 9, -1.0, 1.0);
    let kd = values(9, 10, -1.0, 1.0);
    let x = Array::from_slice_on(&ctx, &data, [h, w]).unwrap();
    let u = x.unwrap([3, 3], [1, 1], [1, 1], true).to_vec().unwrap();
    let got: Vec<f32> = (0..h * w).map(|wi| (0..9).map(|c| u[c + 9 * wi] * kd[c]).sum()).collect();
    let kern = Array::from_slice_on(&ctx, &kd, [3, 3]).unwrap();
    let want = x.conv2(&kern, ConvSpec::new()).to_vec().unwrap();
    for (g, w) in got.iter().zip(&want) {
        assert!((g - w).abs() < 1e-4);
    }
    // Shape errors poison.
    assert!(matches!(x.unwrap([20, 3], [1, 1], [0, 0], true).to_vec(), Err(LumaError::Op { .. })));
    assert!(matches!(x.unwrap([3, 3], [0, 1], [0, 0], true).to_vec(), Err(LumaError::Op { .. })));
    assert!(matches!(x.wrap([4, 4], [3, 3], [1, 1], [0, 0], true).to_vec(), Err(LumaError::Op { .. })));
}

// ---------------------------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------------------------

#[test]
fn errors_poison_the_result() {
    let ctx = ctx();
    let x = Array::from_slice_on(&ctx, &values(30, 1, 0.0, 1.0), [5, 6]).unwrap();
    let k3 = Array::from_slice_on(&ctx, &values(9, 2, 0.0, 1.0), [3, 3]).unwrap();
    let k1 = Array::from_slice_on(&ctx, &values(3, 2, 0.0, 1.0), 3).unwrap();
    let k3d = Array::from_slice_on(&ctx, &values(8, 2, 0.0, 1.0), [2, 2, 2]).unwrap();
    for result in [
        x.conv2(&k3, ConvSpec::new().stride([0, 1])).to_vec(),
        x.conv2(&k3, ConvSpec::new().border(Border::Ignore)).to_vec(),
        x.conv2(&k3d, ConvSpec::new()).to_vec(),
        x.conv2(&k3, ConvSpec::new().stride([1, 1, 1, 1])).to_vec(),
        x.conv2(&Array::from_slice_on(&ctx, &values(49, 2, 0.0, 1.0), [7, 7]).unwrap(), ConvSpec::new().padding(Padding::Valid)).to_vec(),
        x.conv2_sep(&k3, &k1, ConvSpec::new()).to_vec(),
        x.medfilt2([3, 3], Border::Ignore).to_vec(),
        x.medfilt2([0, 3], Border::Zero).to_vec(),
        x.sobel_with(Border::Ignore).0.to_vec(),
        x.bilateral(0.0, 1.0).to_vec(),
        x.bilateral(1.0, f32::NAN).to_vec(),
        x.dilate(&k3d).to_vec(),
    ] {
        assert!(matches!(result, Err(LumaError::Op { .. })), "{result:?}");
    }
    // A poisoned operand keeps its own error, and both outputs of a pair carry it.
    let bad = x.conv2(&k3, ConvSpec::new().stride([0, 1]));
    let (a, b) = bad.sobel();
    assert!(a.to_vec().is_err() && b.to_vec().is_err());
    // The error names the operation.
    let message = format!("{}", x.medfilt2([3, 3], Border::Ignore).to_vec().unwrap_err());
    assert!(message.contains("medfilt2"), "{message}");
}

#[test]
fn filters_on_the_empty_batch_edges() {
    // One-cell images and one-cell windows are identities for the rank filters.
    let ctx = ctx();
    let x = Array::from_slice_on(&ctx, &[4.0f32], [1, 1]).unwrap();
    assert_eq!(x.medfilt2([1, 1], Border::Zero).to_vec().unwrap(), [4.0]);
    assert_eq!(x.maxfilt([3, 3], Border::Ignore).to_vec().unwrap(), [4.0]);
    assert_eq!(x.minfilt([3, 3], Border::Zero).to_vec().unwrap(), [0.0]);
    assert_eq!(x.sat().to_vec().unwrap(), [4.0]);
}
