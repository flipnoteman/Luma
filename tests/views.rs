//! Views (slices and non-circular shifts) and the elementwise functions added with them. Every
//! test compares the result with a host loop over the same index rule.

mod common;

use common::{assert_close, values};
use luma::{concat, concat_slice, Array, Context, ContextOptions, LumaError, PadMode};

/// Element `(x, y)` of a `[w, h]` array held in a `Vec`.
fn at(data: &[f32], w: usize, x: usize, y: usize) -> f32 {
    data[x + w * y]
}

#[test]
fn slice_selects_a_range_along_an_axis() {
    let ctx = Context::new().unwrap();
    let a = values(6 * 5, 1, -1.0, 1.0);
    let x = Array::from_slice_on(&ctx, &a, [6, 5]).unwrap();

    let s = x.slice(0, 1..4);
    assert_eq!(s.shape().dims(), [3, 5, 1, 1]);
    let want: Vec<f32> = (0..5).flat_map(|y| (0..3).map(move |x| (x, y))).map(|(x, y)| at(&a, 6, x + 1, y)).collect();
    assert_eq!(s.to_vec().unwrap(), want);

    let s = x.slice(1, ..2);
    assert_eq!(s.shape().dims(), [6, 2, 1, 1]);
    assert_eq!(s.to_vec().unwrap(), a[..12].to_vec());

    let s = x.slice(1, 3..);
    assert_eq!(s.shape().dims(), [6, 2, 1, 1]);
    assert_eq!(s.to_vec().unwrap(), a[18..].to_vec());

    // Composition: a slice of a slice reads the original through one map.
    let s = x.slice(0, 1..5).slice(1, 2..4);
    assert_eq!(s.shape().dims(), [4, 2, 1, 1]);
    let want: Vec<f32> = (2..4).flat_map(|y| (1..5).map(move |x| (x, y))).map(|(x, y)| at(&a, 6, x, y)).collect();
    assert_eq!(s.to_vec().unwrap(), want);
}

#[test]
fn slice_of_a_lazy_expression_uses_the_sub_program() {
    let ctx = Context::new().unwrap();
    let a = values(8, 2, -1.0, 1.0);
    let x = Array::from_slice_on(&ctx, &a, 8).unwrap();
    let got = (&x * 2.0 + 1.0).slice(0, 2..5).to_vec().unwrap();
    let want: Vec<f32> = a[2..5].iter().map(|v| v * 2.0 + 1.0).collect();
    assert_close(&got, &want, 1e-6, 1e-6);

    // Used twice, at different places, in one expression.
    let e = &x * 3.0;
    let got = (e.slice(0, 1..4) - e.slice(0, 0..3)).to_vec().unwrap();
    let want: Vec<f32> = (0..3).map(|i| a[i + 1] * 3.0 - a[i] * 3.0).collect();
    assert_close(&got, &want, 1e-5, 1e-6);
}

#[test]
#[allow(clippy::reversed_empty_ranges)] // the reversed range is the error under test
fn slice_errors_poison_the_result() {
    let ctx = Context::new().unwrap();
    let x = Array::from_slice_on(&ctx, &values(6, 3, 0.0, 1.0), 6).unwrap();
    for bad in [x.slice(0, 4..4), x.slice(4, ..), x.slice(0, ..7), x.slice(0, 5..2)] {
        let err = bad.to_vec().unwrap_err();
        assert!(matches!(err.root(), LumaError::InvalidShape(_)), "{err}");
    }
}

#[test]
fn edge_shifts_on_a_line() {
    let ctx = Context::new().unwrap();
    let x = Array::from_slice_on(&ctx, &[1.0f32, 2.0, 3.0, 4.0], 4).unwrap();
    assert_eq!(x.shift_fill(1, 9.0).to_vec().unwrap(), [9.0, 1.0, 2.0, 3.0]);
    assert_eq!(x.shift_fill(-1, 9.0).to_vec().unwrap(), [2.0, 3.0, 4.0, 9.0]);
    assert_eq!(x.shift_fill(7, 9.0).to_vec().unwrap(), [9.0; 4]);
    assert_eq!(x.shift_clamp(-1).to_vec().unwrap(), [2.0, 3.0, 4.0, 4.0]);
    assert_eq!(x.shift_clamp(2).to_vec().unwrap(), [1.0, 1.0, 1.0, 2.0]);
    assert_eq!(x.shift_mirror(2).to_vec().unwrap(), [2.0, 1.0, 1.0, 2.0]);
    assert_eq!(x.shift_mirror(-1).to_vec().unwrap(), [2.0, 3.0, 4.0, 4.0]);
    assert_eq!(x.shift_mirror(0).to_vec().unwrap(), [1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn edge_shifts_in_two_dimensions() {
    let ctx = Context::new().unwrap();
    let (w, h) = (5usize, 4usize);
    let a = values(w * h, 4, -1.0, 1.0);
    let x = Array::from_slice_on(&ctx, &a, [w, h]).unwrap();
    let (ox, oy) = (1isize, -1isize);

    let source = |x: usize, y: usize, fix: &dyn Fn(isize, usize) -> Option<usize>| {
        let sx = fix(x as isize - ox, w)?;
        let sy = fix(y as isize - oy, h)?;
        Some(at(&a, w, sx, sy))
    };
    let clamp = |c: isize, n: usize| Some(c.clamp(0, n as isize - 1) as usize);
    let mirror = |c: isize, n: usize| {
        let n = n as isize;
        let folded = if c < 0 { -c - 1 } else if c >= n { 2 * n - 1 - c } else { c };
        Some(folded.clamp(0, n - 1) as usize)
    };
    let fill = |c: isize, n: usize| (0..n as isize).contains(&c).then_some(c as usize);
    let table = |fix: &dyn Fn(isize, usize) -> Option<usize>, fill_value: f32| -> Vec<f32> {
        (0..h).flat_map(|y| (0..w).map(move |x| (x, y))).map(|(x, y)| source(x, y, fix).unwrap_or(fill_value)).collect()
    };

    assert_eq!(x.shift_clamp([ox, oy]).to_vec().unwrap(), table(&clamp, 0.0));
    assert_eq!(x.shift_mirror([ox, oy]).to_vec().unwrap(), table(&mirror, 0.0));
    assert_eq!(x.shift_fill([ox, oy], 7.5).to_vec().unwrap(), table(&fill, 7.5));
}

#[test]
fn a_difference_of_slices_is_one_kernel_result() {
    let ctx = Context::new().unwrap();
    let a = values(9, 5, -1.0, 1.0);
    let x = Array::from_slice_on(&ctx, &a, 9).unwrap();
    let got = (x.slice(0, 1..) - x.slice(0, ..8)).to_vec().unwrap();
    let want: Vec<f32> = (0..8).map(|i| a[i + 1] - a[i]).collect();
    assert_close(&got, &want, 1e-6, 1e-6);
}

#[test]
fn views_of_chunked_arrays() {
    let ctx = Context::with_options(ContextOptions::default().chunk_bytes(1024)).unwrap();
    let a = values(1000, 6, -1.0, 1.0);
    let x = Array::from_slice_on(&ctx, &a, 1000).unwrap();

    assert_eq!(x.slice(0, 100..900).to_vec().unwrap(), a[100..900].to_vec());

    let want: Vec<f32> = (0..1000usize).map(|i| a[(i as isize + 300).clamp(0, 999) as usize]).collect();
    assert_eq!(x.shift_clamp(-300).to_vec().unwrap(), want);

    let want: Vec<f32> = (0..1000usize).map(|i| a[(i as isize - 600).clamp(0, 999) as usize]).collect();
    assert_eq!(x.shift_clamp(600).to_vec().unwrap(), want);

    let want: Vec<f32> =
        (0..1000isize).map(|i| if i - 5 >= 0 { a[(i - 5) as usize] } else { -2.0 }).collect();
    assert_eq!(x.shift_fill(5, -2.0).to_vec().unwrap(), want);

    // A small output reading a split source.
    let tail = x.slice(0, 990..);
    assert_eq!(tail.to_vec().unwrap(), a[990..].to_vec());
    let sum = (x.slice(0, 1..) - x.slice(0, ..999)).to_vec().unwrap();
    let want: Vec<f32> = (0..999).map(|i| a[i + 1] - a[i]).collect();
    assert_close(&sum, &want, 1e-6, 1e-6);
}

#[test]
fn slice_difference_of_a_lazy_expression_on_chunked_arrays() {
    let ctx = Context::with_options(ContextOptions::default().chunk_bytes(1024)).unwrap();
    let a = values(1000, 14, -1.0, 1.0);
    let x = Array::from_slice_on(&ctx, &a, 1000).unwrap();
    let e = (&x * 2.0 + 1.0).abs().sqrt();
    let got = (e.slice(0, 1..) - e.slice(0, ..999)).to_vec().unwrap();
    let host: Vec<f32> = a.iter().map(|v| (v * 2.0 + 1.0).abs().sqrt()).collect();
    let want: Vec<f32> = (0..999).map(|i| host[i + 1] - host[i]).collect();
    assert_close(&got, &want, 1e-5, 1e-6);
}

#[test]
fn elementwise_functions_match_the_host() {
    let ctx = Context::new().unwrap();
    let xs = values(64, 3, -3.0, 3.0);
    let ys = values(64, 4, -3.0, 3.0);
    let x = Array::from_slice_on(&ctx, &xs, 64).unwrap();
    let y = Array::from_slice_on(&ctx, &ys, 64).unwrap();
    let check = |got: Array<f32>, rel: f32, f: &dyn Fn(f32, f32) -> f32| {
        let want: Vec<f32> = xs.iter().zip(&ys).map(|(&a, &b)| f(a, b)).collect();
        assert_close(&got.to_vec().unwrap(), &want, rel, 1e-6);
    };
    check(x.tanh(), 1e-5, &|a, _| a.tanh());
    check(x.sign(), 1e-5, &|a, _| if a > 0.0 { 1.0 } else if a < 0.0 { -1.0 } else { 0.0 });
    check(x.floor(), 1e-5, &|a, _| a.floor());
    check(x.ceil(), 1e-5, &|a, _| a.ceil());
    // The software rasteriser's atan2 is good to about 1e-5 relative only.
    check(x.atan2(&y), 1e-4, &|a, b| a.atan2(b));
    check(x.hypot(&y), 1e-5, &|a, b| a.hypot(b));
    check(x.copysign(&y), 1e-5, &|a, b| a.copysign(b));
    check(x.atan2(0.5), 1e-4, &|a, _| a.atan2(0.5));
}

#[test]
fn integer_remainder() {
    let ctx = Context::new().unwrap();
    let x = Array::from_slice_on(&ctx, &[7u32, 8, 9], 3).unwrap();
    assert_eq!((&x % 4u32).to_vec().unwrap(), [3, 0, 1]);
    let mut y = Array::from_slice_on(&ctx, &[7u32, 8, 9], 3).unwrap();
    y %= 5u32;
    assert_eq!(y.to_vec().unwrap(), [2, 3, 4]);
}

#[test]
fn pad_in_all_modes() {
    let ctx = Context::new().unwrap();
    let (w, h) = (4usize, 3usize);
    let a = values(w * h, 5, -1.0, 1.0);
    let x = Array::from_slice_on(&ctx, &a, [w, h]).unwrap();
    let (bx, ax, by, ay) = (2isize, 1isize, 1isize, 2isize);
    let (pw, ph) = (w as isize + bx + ax, h as isize + by + ay);

    let wrap = |c: isize, n: usize| c.rem_euclid(n as isize) as usize;
    let clamp = |c: isize, n: usize| c.clamp(0, n as isize - 1) as usize;
    let mirror = |c: isize, n: usize| {
        let n = n as isize;
        let c = if c < 0 { -c - 1 } else { c };
        (if c >= n { 2 * n - 1 - c } else { c }).clamp(0, n - 1) as usize
    };
    type Fix<'a> = &'a dyn Fn(isize, usize) -> usize;
    let host = |fix: Fix, zero: bool| -> Vec<f32> {
        let mut out = Vec::new();
        for y in 0..ph {
            for x in 0..pw {
                let (sx, sy) = (x - bx, y - by);
                let inside = (0..w as isize).contains(&sx) && (0..h as isize).contains(&sy);
                out.push(if zero && !inside { 0.0 } else { at(&a, w, fix(sx, w), fix(sy, h)) });
            }
        }
        out
    };
    for (mode, want) in [
        (PadMode::Edge, host(&clamp, false)),
        (PadMode::Mirror, host(&mirror, false)),
        (PadMode::Wrap, host(&wrap, false)),
        (PadMode::Zero, host(&clamp, true)),
    ] {
        let p = x.pad([(2, 1), (1, 2)], mode);
        assert_eq!(p.shape().dims(), [7, 6, 1, 1]);
        assert_eq!(p.to_vec().unwrap(), want, "{mode:?}");
    }
    // Mirror: ghost -1 equals interior 0.
    let line = Array::from_slice_on(&ctx, &[1.0f32, 2.0, 3.0], 3).unwrap();
    assert_eq!(line.pad([(2, 2)], PadMode::Mirror).to_vec().unwrap(), [2.0, 1.0, 1.0, 2.0, 3.0, 3.0, 2.0]);
}

#[test]
fn flip_and_transpose() {
    let ctx = Context::new().unwrap();
    let (w, h) = (5usize, 3usize);
    let a = values(w * h, 6, -1.0, 1.0);
    let x = Array::from_slice_on(&ctx, &a, [w, h]).unwrap();

    let want: Vec<f32> = (0..h).flat_map(|y| (0..w).map(move |x| (x, y))).map(|(x, y)| at(&a, w, w - 1 - x, y)).collect();
    assert_eq!(x.flip(0).to_vec().unwrap(), want);
    let want: Vec<f32> = (0..h).flat_map(|y| (0..w).map(move |x| (x, y))).map(|(x, y)| at(&a, w, x, h - 1 - y)).collect();
    assert_eq!(x.flip(1).to_vec().unwrap(), want);

    let t = x.transpose();
    assert_eq!(t.shape().dims(), [h, w, 1, 1]);
    let want: Vec<f32> = (0..w).flat_map(|y| (0..h).map(move |x| (x, y))).map(|(x, y)| at(&a, w, y, x)).collect();
    assert_eq!(t.to_vec().unwrap(), want);

    // Views compose lazily with expressions.
    let got = (&x * 2.0).transpose().flip(0).to_vec().unwrap();
    let want: Vec<f32> = (0..w).flat_map(|y| (0..h).map(move |x| (x, y))).map(|(x, y)| at(&a, w, y, h - 1 - x) * 2.0).collect();
    assert_close(&got, &want, 1e-6, 1e-6);

    assert!(matches!(x.flip(4).to_vec().unwrap_err().root(), LumaError::InvalidShape(_)));
}

#[test]
fn index_gives_the_coordinate_along_an_axis() {
    let ctx = Context::new().unwrap();
    let x = Array::<f32>::zeros_on(&ctx, [3, 2]).unwrap();
    assert_eq!(x.index(1).to_vec().unwrap(), [0, 0, 0, 1, 1, 1]);
    assert_eq!(x.index(0).to_vec().unwrap(), [0, 1, 2, 0, 1, 2]);
}

#[test]
fn concat_joins_along_an_axis() {
    let ctx = Context::new().unwrap();
    let (a, b, c) = (values(3 * 2, 7, -1.0, 1.0), values(2 * 2, 8, -1.0, 1.0), values(4 * 2, 9, -1.0, 1.0));
    let xa = Array::from_slice_on(&ctx, &a, [3, 2]).unwrap();
    let xb = Array::from_slice_on(&ctx, &b, [2, 2]).unwrap();
    let xc = Array::from_slice_on(&ctx, &c, [4, 2]).unwrap();

    let joined = concat(0, [&xa, &xb]);
    assert_eq!(joined.shape().dims(), [5, 2, 1, 1]);
    let want: Vec<f32> = (0..2).flat_map(|y| a[3 * y..3 * y + 3].iter().chain(&b[2 * y..2 * y + 2]).copied().collect::<Vec<_>>()).collect();
    assert_eq!(joined.to_vec().unwrap(), want);

    // Along axis 1 the parts must agree in width.
    let (p, q, r) = (values(4 * 2, 10, -1.0, 1.0), values(4, 11, -1.0, 1.0), values(4 * 3, 12, -1.0, 1.0));
    let (xp, xq, xr) = (
        Array::from_slice_on(&ctx, &p, [4, 2]).unwrap(),
        Array::from_slice_on(&ctx, &q, [4, 1]).unwrap(),
        Array::from_slice_on(&ctx, &r, [4, 3]).unwrap(),
    );
    let joined = concat(1, [&xp, &xq, &xr]);
    assert_eq!(joined.shape().dims(), [4, 6, 1, 1]);
    let want: Vec<f32> = p.iter().chain(&q).chain(&r).copied().collect();
    assert_eq!(joined.to_vec().unwrap(), want);

    // Mismatching extents poison the result.
    let bad = concat(0, [&xa, &xc.slice(1, ..1)]);
    assert!(matches!(bad.to_vec().unwrap_err().root(), LumaError::ShapeMismatch { .. }));
}

#[test]
fn take_gathers_by_flat_index() {
    let ctx = Context::new().unwrap();
    let a = Array::from_slice_on(&ctx, &[10.0f32, 11.0, 12.0, 13.0], 4).unwrap();
    let idx = Array::from_slice_on(&ctx, &[3u32, 0, 0, 99], 4).unwrap();
    assert_eq!(a.take(&idx).to_vec().unwrap(), [13.0, 10.0, 10.0, 13.0]);
    // The result has the shape of the indices and fuses with the expression using it.
    let idx2 = Array::from_slice_on(&ctx, &[1u32, 2, 3, 0, 1, 2], [3, 2]).unwrap();
    let got = (&a * 2.0).take(&idx2).to_vec().unwrap();
    assert_eq!(got, [22.0, 24.0, 26.0, 20.0, 22.0, 24.0]);
}

#[test]
fn views_are_copies_across_small_chunks() {
    let ctx = Context::with_options(ContextOptions::default().chunk_bytes(1024)).unwrap();
    let a = values(1000, 13, -1.0, 1.0);
    let x = Array::from_slice_on(&ctx, &a, 1000).unwrap();
    let want: Vec<f32> = (0..1000).map(|i: i32| a[(i - 3).clamp(0, 999) as usize]).collect();
    assert_eq!(x.shift_clamp(3).to_vec().unwrap(), want);
    let want: Vec<f32> = (0..1000usize).rev().map(|i| a[i]).collect();
    assert_eq!(x.flip(0).to_vec().unwrap(), want);
    let idx = Array::from_slice_on(&ctx, &(0..1000u32).map(|i| (i * 7) % 1000).collect::<Vec<_>>(), 1000).unwrap();
    let want: Vec<f32> = (0..1000usize).map(|i| a[(i * 7) % 1000]).collect();
    assert_eq!(x.take(&idx).to_vec().unwrap(), want);
}

#[test]
fn concat_slice_joins_parts_along_an_axis() {
    let ctx = Context::new().unwrap();
    let a = Array::from_slice_on(&ctx, &[1.0f32, 2.0], 2).unwrap();
    let b = Array::from_slice_on(&ctx, &[3.0f32, 4.0, 5.0], 3).unwrap();
    let c = Array::from_slice_on(&ctx, &[6.0f32], 1).unwrap();
    let parts = [a.clone(), b.clone(), c];

    assert_eq!(concat_slice(0, &parts).to_vec().unwrap(), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    assert_eq!(concat_slice(0, &parts[..1]).to_vec().unwrap(), [1.0, 2.0]);
    assert_eq!(concat_slice(0, &parts[..2]).to_vec().unwrap(), concat(0, [&a, &b]).to_vec().unwrap());

    // Along the second axis.
    let p = Array::from_slice_on(&ctx, &[1.0f32, 2.0, 3.0, 4.0], [2, 2]).unwrap();
    let q = Array::from_slice_on(&ctx, &[5.0f32, 6.0], [2, 1]).unwrap();
    let joined = concat_slice(1, &[p, q]);
    assert_eq!(joined.shape().dims(), [2, 3, 1, 1]);
    assert_eq!(joined.to_vec().unwrap(), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

    // It fuses with the arithmetic around it, and its parts may be expressions.
    let lazy = [&a * 10.0, &b + 1.0];
    let got = (concat_slice(0, &lazy) * 2.0 + 1.0).to_vec().unwrap();
    assert_eq!(got, [21.0, 41.0, 9.0, 11.0, 13.0]);
}

#[test]
fn concat_slice_of_no_arrays_and_mismatched_parts_fail() {
    let ctx = Context::new().unwrap();
    let empty: [Array<f32>; 0] = [];
    let err = concat_slice(0, &empty).to_vec().unwrap_err();
    assert!(matches!(err.root(), LumaError::InvalidShape(_)), "{err}");

    let p = Array::<f32>::zeros_on(&ctx, [2, 2]).unwrap();
    let q = Array::<f32>::zeros_on(&ctx, [3, 1]).unwrap();
    let err = concat_slice(1, &[p, q]).to_vec().unwrap_err();
    assert!(matches!(err.root(), LumaError::ShapeMismatch { .. }), "{err}");
}

#[test]
fn mutable_views_write_through_and_read_only_ones_do_not() {
    let ctx = Context::new().unwrap();
    let mut w = Array::from_slice_on(&ctx, &[1.0f32, 2.0, 3.0, 4.0], 4).unwrap();
    let mut v = w.slice_mut(0, 1..3);
    v += 10.0;
    assert_eq!(w.to_vec().unwrap(), [1.0, 12.0, 13.0, 4.0]);

    // The read-only slice sees the data and keeps rejecting writes.
    let mut s = w.slice(0, 1..3);
    assert_eq!(s.to_vec().unwrap(), [12.0, 13.0]);
    s += 1.0;
    let err = s.to_vec().unwrap_err();
    assert!(matches!(err.root(), LumaError::InvalidShape(_)), "{err}");
    assert_eq!(w.to_vec().unwrap(), [1.0, 12.0, 13.0, 4.0]);

    // `to_array` of a mutable view is a read-only array of its region.
    let mut region = w.slice_mut(0, 2..).to_array();
    assert_eq!(region.to_vec().unwrap(), [13.0, 4.0]);
    assert!(region.assign(0.0).is_err());
}
