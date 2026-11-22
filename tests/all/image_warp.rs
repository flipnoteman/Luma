//! `resize`, `rotate`, `translate`, `scale`, `skew`, `transform` and `transform_coordinates`
//! against per-pixel `f64` references, known permutations and ArrayFire's documented conventions.

use crate::common::values;
use luma::{transform_coordinates, Array, Interp, LumaError};

const METHODS: [Interp; 6] = [Interp::Nearest, Interp::Lower, Interp::Linear, Interp::Cosine, Interp::Cubic, Interp::CubicSpline];

/// The interpolant along one axis (see the `interp` tests), neighbours clamped.
fn axis(n: usize, x: f64, method: Interp, f: &dyn Fn(usize) -> f64) -> f64 {
    let at = |i: i64| f(i.clamp(0, n as i64 - 1) as usize);
    let i = x.floor() as i64;
    let t = x - x.floor();
    match method {
        Interp::Nearest => at((x + 0.5).floor() as i64),
        Interp::Lower => at(i),
        Interp::Linear => (1.0 - t) * at(i) + t * at(i + 1),
        Interp::Cosine => {
            let c = (1.0 - (t * std::f64::consts::PI).cos()) * 0.5;
            (1.0 - c) * at(i) + c * at(i + 1)
        }
        Interp::Cubic => {
            let (y0, y1, y2, y3) = (at(i - 1), at(i), at(i + 1), at(i + 2));
            let a0 = y3 - y2 - y0 + y1;
            let a1 = y0 - y1 - a0;
            a0 * t * t * t + a1 * t * t + (y2 - y0) * t + y1
        }
        Interp::CubicSpline => {
            let (y0, y1, y2, y3) = (at(i - 1), at(i), at(i + 1), at(i + 2));
            y1 + (-0.5 * y0 + 0.5 * y2) * t + (y0 - 2.5 * y1 + 2.0 * y2 - 0.5 * y3) * t * t + (-0.5 * y0 + 1.5 * y1 - 1.5 * y2 + 0.5 * y3) * t * t * t
        }
    }
}

/// An image `[w, h, c]` (first dimension fastest) sampled at `(x, y)` of channel `ch`.
struct Image<'a> {
    data: &'a [f32],
    w: usize,
    h: usize,
}

impl Image<'_> {
    fn sample(&self, x: f64, y: f64, ch: usize, method: Interp) -> f64 {
        axis(self.h, y, method, &|j| axis(self.w, x, method, &|i| f64::from(self.data[i + self.w * (j + self.h * ch)])))
    }
}

fn dims(a: &Array<f32>) -> [usize; 4] {
    a.shape().dims()
}

/// Whether a source coordinate sits so close to a place where the result changes abruptly (a cell
/// boundary, a rounding tie, the edge of the image) that `f32` and `f64` may disagree.
fn borderline(x: f64, n: usize) -> bool {
    // Exactly on the boundary is computed exactly in `f32` too (integer and dyadic maps).
    let near = |v: f64, target: f64| (v - target).abs() < 2e-3 && v != target;
    near(x, x.round()) || near(x - x.floor(), 0.5) || near(x, 0.0) || near(x, n as f64) || near(x, n as f64 - 0.5)
}

fn compare_pixels(got: &[f32], want: &[Option<f64>], tol: f64, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    let mut checked = 0;
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        if let Some(w) = w {
            assert!((f64::from(*g) - w).abs() <= tol * (1.0 + w.abs()), "{what}: element {i}: got {g}, want {w}");
            checked += 1;
        }
    }
    assert!(checked * 2 > got.len(), "{what}: only {checked} of {} pixels were comparable", got.len());
}

#[test]
fn resize_matches_the_reference() {
    let (w, h, c) = (17, 13, 2);
    let img = values(w * h * c, 1, -3.0, 3.0);
    let a = Array::from_slice(&img, [w, h, c]).unwrap();
    let image = Image { data: &img, w, h };
    for (o0, o1) in [(40, 29), (9, 5), (17, 13), (1, 1), (3, 40), (100, 7)] {
        for method in METHODS {
            let got = a.resize(o0, o1, method);
            assert_eq!(dims(&got), [o0, o1, c, 1]);
            let got = got.to_vec().unwrap();
            let (s0, s1) = (o0 as f32 / w as f32, o1 as f32 / h as f32);
            let mut want = vec![];
            for ch in 0..c {
                for y in 0..o1 {
                    for x in 0..o0 {
                        // The source position is computed in `f32` as `x / scale`.
                        let (px, py) = (x as f32 / s0, y as f32 / s1);
                        want.push(Some(image.sample(f64::from(px), f64::from(py), ch, method)));
                    }
                }
            }
            compare_pixels(&got, &want, 3e-5, &format!("resize {o0}x{o1} {method:?}"));
        }
    }
}

#[test]
fn resize_known_values() {
    let a = Array::from_slice(&[0.0f32, 1.0, 2.0, 3.0], [2, 2]).unwrap();
    // 2x2 -> 4x4, bilinear: positions x * 0.5; the last row and column repeat the last pixel.
    let up = a.resize(4, 4, Interp::Linear).to_vec().unwrap();
    assert_eq!(&up[..4], &[0.0, 0.5, 1.0, 1.0]);
    assert_eq!(&up[4..8], &[1.0, 1.5, 2.0, 2.0]);
    // Nearest 2x -> each pixel repeated, except that .5 rounds up.
    let up = a.resize(4, 2, Interp::Nearest).to_vec().unwrap();
    assert_eq!(up, vec![0.0, 1.0, 1.0, 1.0, 2.0, 3.0, 3.0, 3.0]);
    // Lower floors.
    let low = a.resize(4, 2, Interp::Lower).to_vec().unwrap();
    assert_eq!(low, vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0]);
    // By scale: extents are truncated.
    let by = Array::from_slice(&values(35 * 17, 2, 0.0, 1.0), [35, 17]).unwrap().resize_by(0.33, 1.7, Interp::Linear);
    assert_eq!(dims(&by), [11, 28, 1, 1]);
    // A batch of images keeps its extents.
    let batch = Array::from_slice(&values(4 * 4 * 2 * 3, 3, 0.0, 1.0), [4, 4, 2, 3]).unwrap();
    assert_eq!(dims(&batch.resize(8, 2, Interp::Cubic)), [8, 2, 2, 3]);
}

#[test]
fn resize_errors() {
    let a = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], [2, 2]).unwrap();
    assert!(matches!(a.resize(0, 4, Interp::Linear).to_vec(), Err(LumaError::Op { .. })));
    assert!(a.resize_by(0.0, 1.0, Interp::Linear).to_vec().is_err());
    assert!(a.resize_by(f32::NAN, 1.0, Interp::Linear).to_vec().is_err());
    assert!(a.resize_by(0.1, 1.0, Interp::Linear).to_vec().is_err(), "an extent that truncates to 0");
}

/// The map from output to input pixel coordinates the transform functions apply, in `f64`.
fn inverse_map(m: [f64; 9], x: f64, y: f64) -> (f64, f64) {
    let w = m[6] * x + m[7] * y + m[8];
    ((m[0] * x + m[1] * y + m[2]) / w, (m[3] * x + m[4] * y + m[5]) / w)
}

fn invert(m: [f64; 9]) -> [f64; 9] {
    let det = m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6]) + m[2] * (m[3] * m[7] - m[4] * m[6]);
    let c = |a: usize, b: usize, c: usize, d: usize| m[a] * m[d] - m[b] * m[c];
    [
        c(4, 5, 7, 8) / det,
        -c(1, 2, 7, 8) / det,
        c(1, 2, 4, 5) / det,
        -c(3, 5, 6, 8) / det,
        c(0, 2, 6, 8) / det,
        -c(0, 2, 3, 5) / det,
        c(3, 4, 6, 7) / det,
        -c(0, 1, 6, 7) / det,
        c(0, 1, 3, 4) / det,
    ]
}

fn reference(img: &Image<'_>, c: usize, m: [f64; 9], o: (usize, usize), method: Interp) -> Vec<Option<f64>> {
    let mut want = vec![];
    for ch in 0..c {
        for y in 0..o.1 {
            for x in 0..o.0 {
                let (sx, sy) = inverse_map(m, x as f64, y as f64);
                let reach = if method == Interp::Nearest { 0.5 } else { 0.0 };
                let valid = sx >= 0.0 && sy >= 0.0 && sx + reach < img.w as f64 && sy + reach < img.h as f64;
                let skip = borderline(sx, img.w) || borderline(sy, img.h);
                want.push(if skip && valid {
                    None
                } else if !valid {
                    Some(0.0)
                } else {
                    Some(img.sample(sx, sy, ch, method))
                });
            }
        }
    }
    want
}

fn to9(m: [f32; 6]) -> [f64; 9] {
    [m[0], m[1], m[2], m[3], m[4], m[5], 0.0, 0.0, 1.0].map(f64::from)
}

#[test]
fn transform_matches_the_reference() {
    let (w, h, c) = (19, 11, 2);
    let img = values(w * h * c, 4, 1.0, 5.0);
    let a = Array::from_slice(&img, [w, h, c]).unwrap();
    let image = Image { data: &img, w, h };
    let affine: [[f32; 6]; 4] = [[0.9, -0.3, 4.0, 0.25, 1.1, -2.0], [1.0, 0.0, 2.5, 0.0, 1.0, -1.5], [0.5, 0.0, 0.0, 0.0, 0.5, 0.0], [2.0, 0.2, 1.0, -0.1, 1.5, 3.0]];
    for m in affine {
        for inverse in [true, false] {
            for method in METHODS {
                let got = a.transform(m, 25, 14, method, inverse).to_vec().unwrap();
                let direct = if inverse { to9(m) } else { invert(to9(m)) };
                compare_pixels(&got, &reference(&image, c, direct, (25, 14), method), 1e-4, &format!("transform {m:?} inverse={inverse} {method:?}"));
            }
        }
    }
    // Projective, with the input's size as the default output.
    let per: [f32; 9] = [1.0, 0.1, 0.5, 0.2, 1.0, -0.3, 0.003, 0.002, 1.0];
    for inverse in [true, false] {
        for method in [Interp::Nearest, Interp::Linear, Interp::Cubic] {
            let got = a.transform(per, 0, 0, method, inverse);
            assert_eq!(dims(&got), [w, h, c, 1]);
            let m = per.map(f64::from);
            let direct = if inverse { m } else { invert(m) };
            compare_pixels(&got.to_vec().unwrap(), &reference(&image, c, direct, (w, h), method), 1e-4, &format!("perspective inverse={inverse} {method:?}"));
        }
    }
    // The nested and flat matrix types are the same matrix.
    let flat = a.transform([1.0f32, 0.0, 1.0, 0.0, 1.0, 2.0], 0, 0, Interp::Linear, true).to_vec().unwrap();
    let nested = a.transform([[1.0f32, 0.0, 1.0], [0.0, 1.0, 2.0]], 0, 0, Interp::Linear, true).to_vec().unwrap();
    let full = a.transform([[1.0f32, 0.0, 1.0], [0.0, 1.0, 2.0], [0.0, 0.0, 1.0]], 0, 0, Interp::Linear, true).to_vec().unwrap();
    assert_eq!(flat, nested);
    assert_eq!(flat, full);
}

#[test]
fn transform_known_moves() {
    let a = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], [3, 3]).unwrap();
    // Forward: content moves by (+1, 0): the first column is empty.
    let moved = a.transform([1.0f32, 0.0, 1.0, 0.0, 1.0, 0.0], 3, 3, Interp::Nearest, false);
    assert_eq!(moved.to_vec().unwrap(), vec![0.0, 1.0, 2.0, 0.0, 4.0, 5.0, 0.0, 7.0, 8.0]);
    // Inverse: the matrix is the output -> input map, so the content moves by (-1, 0).
    let back = a.transform([1.0f32, 0.0, 1.0, 0.0, 1.0, 0.0], 3, 3, Interp::Nearest, true);
    assert_eq!(back.to_vec().unwrap(), vec![2.0, 3.0, 0.0, 5.0, 6.0, 0.0, 8.0, 9.0, 0.0]);
    // A bigger output.
    let big = a.transform([1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0], 4, 2, Interp::Nearest, true);
    assert_eq!(big.to_vec().unwrap(), vec![1.0, 2.0, 3.0, 0.0, 4.0, 5.0, 6.0, 0.0]);
    // Nearest: a source of x = 2.4 rounds to the existing pixel 2, 2.5 is outside.
    let edge = a.transform([1.0f32, 0.0, 2.4, 0.0, 1.0, 0.0], 1, 1, Interp::Nearest, true).to_vec().unwrap();
    assert_eq!(edge, vec![3.0]);
    let edge = a.transform([1.0f32, 0.0, 2.5, 0.0, 1.0, 0.0], 1, 1, Interp::Nearest, true).to_vec().unwrap();
    assert_eq!(edge, vec![0.0]);
    // Linear: pixels in [n - 1, n) repeat the last pixel; below 0 is outside.
    let lin = a.transform([1.0f32, 0.0, 2.5, 0.0, 1.0, 0.0], 1, 1, Interp::Linear, true).to_vec().unwrap();
    assert_eq!(lin, vec![3.0]);
    let neg = a.transform([1.0f32, 0.0, -0.25, 0.0, 1.0, 0.0], 1, 1, Interp::Linear, true).to_vec().unwrap();
    assert_eq!(neg, vec![0.0]);
}

#[test]
fn transform_errors() {
    let a = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], [2, 2]).unwrap();
    // Singular matrix, forward.
    assert!(a.transform([1.0f32, 2.0, 0.0, 2.0, 4.0, 0.0], 2, 2, Interp::Linear, false).to_vec().is_err());
    // The same matrix is fine as an output -> input map (everything that lands off the image is 0).
    assert!(a.transform([1.0f32, 2.0, 0.0, 2.0, 4.0, 0.0], 2, 2, Interp::Linear, true).to_vec().is_ok());
    assert!(a.transform([f32::NAN, 0.0, 0.0, 0.0, 1.0, 0.0], 2, 2, Interp::Linear, true).to_vec().is_err());
    assert!(a.transform([1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0], 2, 0, Interp::Linear, true).to_vec().is_err(), "one zero extent");
}

#[test]
fn translate_scale_and_skew_follow_arrayfire() {
    let (w, h) = (9, 7);
    let img = values(w * h, 5, 1.0, 5.0);
    let a = Array::from_slice(&img, [w, h]).unwrap();
    let image = Image { data: &img, w, h };
    // translate(t0, t1): output (x, y) is the input at (x + t0, y + t1).
    let got = a.translate(2.0, -1.0, 0, 0, Interp::Nearest).to_vec().unwrap();
    for y in 0..h {
        for x in 0..w {
            let (sx, sy) = (x as i64 + 2, y as i64 - 1);
            let want = if (0..w as i64).contains(&sx) && (0..h as i64).contains(&sy) { img[sx as usize + w * sy as usize] } else { 0.0 };
            assert_eq!(got[x + w * y], want, "translate ({x}, {y})");
        }
    }
    // scale(s0, s1): output (x, y) is the input at (x / s0, y / s1); the default size is the
    // scaled extents, truncated.
    let scaled = a.scale(0.7, 1.9, 0, 0, Interp::Linear);
    assert_eq!(dims(&scaled), [6, 13, 1, 1]);
    let m = [1.0 / 0.7f32, 0.0, 0.0, 0.0, 1.0 / 1.9, 0.0].map(f64::from);
    let m9 = [m[0], 0.0, 0.0, 0.0, m[4], 0.0, 0.0, 0.0, 1.0];
    compare_pixels(&scaled.to_vec().unwrap(), &reference(&image, 1, m9, (6, 13), Interp::Linear), 1e-4, "scale");
    assert_eq!(dims(&a.scale(2.0, 0.5, 5, 5, Interp::Linear)), [5, 5, 1, 1]);
    assert!(a.scale(0.0, 1.0, 0, 0, Interp::Linear).to_vec().is_err());
    // skew with ta = tan(skew0), tb = tan(skew1): the source is (x + tb y, ta x + y), over
    // (1 - ta tb) when inverse.
    for inverse in [false, true] {
        let (s0, s1) = (0.3f32, -0.2f32);
        let (ta, tb) = (f64::from(s0.tan()), f64::from(s1.tan()));
        let c = if inverse { 1.0 / (1.0 - ta * tb) } else { 1.0 };
        let m9 = [c, c * tb, 0.0, c * ta, c, 0.0, 0.0, 0.0, 1.0];
        let got = a.skew(s0, s1, 0, 0, inverse, Interp::Linear).to_vec().unwrap();
        compare_pixels(&got, &reference(&image, 1, m9, (w, h), Interp::Linear), 1e-4, &format!("skew inverse={inverse}"));
    }
}

#[test]
fn rotate_by_quarter_turns_and_in_general() {
    // A quarter turn of a square image is a permutation: output (x, y) is the input (y, n - 1 - x).
    let n = 6;
    let img = values(n * n, 6, 1.0, 5.0);
    let a = Array::from_slice(&img, [n, n]).unwrap();
    let got = a.rotate(std::f32::consts::FRAC_PI_2, true, Interp::Nearest).to_vec().unwrap();
    for y in 0..n {
        for x in 0..n {
            let want = img[y + n * (n - 1 - x)];
            assert!((got[x + n * y] - want).abs() < 1e-6 || got[x + n * y] == 0.0, "quarter turn ({x}, {y})");
        }
    }
    // No rotation keeps the image.
    let same = a.rotate(0.0, true, Interp::Linear).to_vec().unwrap();
    assert_eq!(same, img);
    // In general: the matrix of ArrayFire's rotate, about the centres of the input and output.
    let (w, h) = (23, 15);
    let img = values(w * h * 2, 7, 1.0, 5.0);
    let a = Array::from_slice(&img, [w, h, 2]).unwrap();
    let image = Image { data: &img, w, h };
    for theta in [0.3f32, -1.1, 2.5] {
        for crop in [true, false] {
            let (c, s) = ((-theta).cos(), (-theta).sin());
            let (o0, o1) = if crop { (w as f32, h as f32) } else { ((w as f32 * c.abs() + h as f32 * s.abs()).trunc(), (h as f32 * c.abs() + w as f32 * s.abs()).trunc()) };
            let got = a.rotate(theta, crop, Interp::Linear);
            assert_eq!(dims(&got), [o0 as usize, o1 as usize, 2, 1], "extent theta={theta} crop={crop}");
            let (nx, ny) = (0.5 * (w as f32 - 1.0), 0.5 * (h as f32 - 1.0));
            let (mx, my) = (0.5 * (o0 - 1.0), 0.5 * (o1 - 1.0));
            let (tx, ty) = (nx - (mx * c - my * s), ny - (mx * s + my * c));
            let m9 = [c, -s, tx, s, c, ty, 0.0, 0.0, 1.0].map(f64::from);
            compare_pixels(&got.to_vec().unwrap(), &reference(&image, 2, m9, (o0 as usize, o1 as usize), Interp::Linear), 1e-4, &format!("rotate {theta} crop={crop}"));
        }
    }
}

#[test]
fn transform_coordinates_maps_the_corners() {
    let corners = transform_coordinates([1.0f32, 0.0, 5.0, 0.0, 1.0, 7.0], 4.0, 3.0).unwrap();
    assert_eq!(corners.shape().dims(), [4, 2, 1, 1]);
    assert_eq!(corners.to_vec().unwrap(), vec![5.0, 5.0, 8.0, 8.0, 7.0, 11.0, 11.0, 7.0]);
    // A projective matrix divides.
    let per: [f32; 9] = [1.0, 0.1, 0.5, 0.2, 1.0, -0.3, 0.02, 0.03, 1.0];
    let got = transform_coordinates(per, 8.0, 6.0).unwrap().to_vec().unwrap();
    let m = per.map(f64::from);
    let pts = [(0.0, 0.0), (0.0, 8.0), (6.0, 8.0), (6.0, 0.0)];
    for (k, (x, y)) in pts.into_iter().enumerate() {
        let (u, v) = inverse_map(m, x, y);
        assert!((f64::from(got[k]) - u).abs() < 1e-5 && (f64::from(got[4 + k]) - v).abs() < 1e-5, "corner {k}");
    }
}

#[test]
fn warps_read_lazy_inputs_and_feed_lazy_expressions() {
    let img = Array::from_slice(&values(12 * 10, 8, 0.0, 1.0), [12, 10]).unwrap();
    let direct = img.resize(24, 20, Interp::Linear).to_vec().unwrap();
    let fused = ((&img * 2.0).resize(24, 20, Interp::Linear) * 0.5).to_vec().unwrap();
    crate::common::assert_close(&fused, &direct, 1e-5, 1e-6);
}
