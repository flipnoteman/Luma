//! Writes through views (`assign`) and scatters (`put`): the result must equal a host loop over
//! the old contents, whatever the expression reads.

mod common;

use common::{assert_close, values};
use luma::{Array, Context, LumaError, PadMode};

#[test]
fn an_independently_built_equal_expression_is_not_written_through() {
    let ctx = Context::new().unwrap();
    let xs = values(4, 7, -1.0, 1.0);
    let a = Array::from_slice_on(&ctx, &xs, 4).unwrap();
    let plus: Vec<f32> = xs.iter().map(|v| v + 1.0).collect();

    let t1 = &a + 1.0;
    let t2 = &a + 1.0;
    t1.assign(50.0).unwrap();
    assert_eq!(t1.to_vec().unwrap(), vec![50.0; 4]);
    assert_close(&t2.to_vec().unwrap(), &plus, 1e-6, 1e-6);

    let mut t1 = &a + 1.0;
    let t2 = &a + 1.0;
    t1.write(&[9.0; 4]).unwrap();
    assert_eq!(t1.to_vec().unwrap(), vec![9.0; 4]);
    assert_close(&t2.to_vec().unwrap(), &plus, 1e-6, 1e-6);

    let mut t1 = &a + 1.0;
    let t2 = &a + 1.0;
    t1 += 10.0;
    let want: Vec<f32> = plus.iter().map(|v| v + 10.0).collect();
    assert_close(&t1.to_vec().unwrap(), &want, 1e-6, 1e-6);
    assert_close(&t2.to_vec().unwrap(), &plus, 1e-6, 1e-6);

    let p1 = (&a + 1.0).pad([(1, 1)], luma::PadMode::Edge);
    let p2 = (&a + 1.0).pad([(1, 1)], luma::PadMode::Edge);
    p1.slice(0, ..1).assign(77.0).unwrap();
    assert_eq!(p1.to_vec().unwrap()[0], 77.0);
    assert_close(&p2.to_vec().unwrap()[0..1], &plus[0..1], 1e-6, 1e-6);
}

#[test]
fn a_scalar_fills_a_slice() {
    let ctx = Context::new().unwrap();
    let w = Array::<f32>::zeros_on(&ctx, [6, 4]).unwrap();
    w.slice(1, ..1).assign(7.0).unwrap();
    let got = w.to_vec().unwrap();
    for (i, v) in got.iter().enumerate() {
        assert_eq!(*v, if i < 6 { 7.0 } else { 0.0 }, "{i}");
    }
}

/// The boundary rule of a finite-volume solver: ghost columns from the interior ones.
#[test]
fn the_right_hand_side_may_read_the_target() {
    let ctx = Context::new().unwrap();
    let (ni, nj) = (8usize, 6usize);
    let a = values(ni * nj, 1, -1.0, 1.0);
    let want = {
        let mut want = a.clone();
        for j in 0..nj {
            want[ni * j] = -a[3 + ni * j];
            want[1 + ni * j] = -a[2 + ni * j];
        }
        want
    };

    let w = Array::from_slice_on(&ctx, &a, [ni, nj]).unwrap();
    w.slice(0, ..2).assign(w.slice(0, 2..4).flip(0) * -1.0).unwrap();
    assert_eq!(w.to_vec().unwrap(), want);

    let w = Array::from_slice_on(&ctx, &a, [ni, nj]).unwrap();
    let g = w.slice(0, 2..4).flip(0) * -1.0;
    w.slice(0, ..2).assign(&g).unwrap();
    assert_eq!(w.to_vec().unwrap(), want);
}

#[test]
fn overlap_is_safe() {
    let ctx = Context::new().unwrap();
    let (ni, nj) = (8usize, 3usize);
    let a = values(ni * nj, 2, -1.0, 1.0);
    let mut want = a.clone();
    for j in 0..nj {
        for i in 0..4 {
            want[i + ni * j] = a[i + 1 + ni * j];
        }
    }
    let w = Array::from_slice_on(&ctx, &a, [ni, nj]).unwrap();
    w.slice(0, ..4).assign(w.slice(0, 1..5) * 1.0).unwrap();
    assert_eq!(w.to_vec().unwrap(), want);
}

#[test]
fn lazy_arrays_recorded_before_the_write_keep_the_old_values() {
    let ctx = Context::new().unwrap();
    let a = values(8 * 3, 3, -1.0, 1.0);
    let w = Array::from_slice_on(&ctx, &a, [8, 3]).unwrap();
    let old = &w * 1.0;
    let old_part = w.slice(0, 1..4) * 2.0;
    let view = w.slice(0, ..4);
    let on_view = &view + 1.0;
    view.assign(0.0).unwrap();
    assert_eq!(old.to_vec().unwrap(), a);
    let want: Vec<f32> = (0..3).flat_map(|j| (1..4).map(move |i| (i, j))).map(|(i, j)| a[i + 8 * j] * 2.0).collect();
    assert_close(&old_part.to_vec().unwrap(), &want, 1e-6, 1e-6);
    // What was built on the array assigned to keeps its values; that array itself shows the new data.
    let want: Vec<f32> = (0..3).flat_map(|j| (0..4).map(move |i| (i, j))).map(|(i, j)| a[i + 8 * j] + 1.0).collect();
    assert_close(&on_view.to_vec().unwrap(), &want, 1e-6, 1e-6);
    assert_eq!(view.to_vec().unwrap(), vec![0.0; 12]);
}

#[test]
fn put_scatters() {
    let ctx = Context::new().unwrap();
    let x = Array::<f32>::zeros_on(&ctx, 5).unwrap();
    let idx = Array::from_slice_on(&ctx, &[4u32, 1, 9], 3).unwrap();
    let vals = Array::from_slice_on(&ctx, &[1.0f32, 2.0, 3.0], 3).unwrap();
    x.put(&idx, &vals).unwrap();
    assert_eq!(x.to_vec().unwrap(), [0.0, 2.0, 0.0, 0.0, 1.0]);

    // The values may read the target.
    let y = Array::from_slice_on(&ctx, &[1.0f32, 2.0, 3.0, 4.0, 5.0], 5).unwrap();
    let idx = Array::from_slice_on(&ctx, &[0u32, 1, 2, 3, 4], 5).unwrap();
    y.put(&idx, &y * 2.0).unwrap();
    assert_eq!(y.to_vec().unwrap(), [2.0, 4.0, 6.0, 8.0, 10.0]);
    y.put(&idx, 1.5).unwrap();
    assert_eq!(y.to_vec().unwrap(), [1.5; 5]);

    // Not on a view.
    let err = y.slice(0, 1..3).put(&idx.slice(0, ..2), 1.0).unwrap_err();
    assert!(matches!(err.root(), LumaError::InvalidShape(_)), "{err}");
}

#[test]
fn only_slices_flips_and_transposes_write_through() {
    let ctx = Context::new().unwrap();
    let a_data = values(5, 4, -1.0, 1.0);
    let a = Array::from_slice_on(&ctx, &a_data, 5).unwrap();

    (&a + 1.0).assign(5.0).unwrap();
    assert_eq!(a.to_vec().unwrap(), a_data);
    a.shift_clamp(1).assign(1.0).unwrap();
    assert_eq!(a.to_vec().unwrap(), a_data);

    let b = Array::from_slice_on(&ctx, &[10.0f32, 20.0], 2).unwrap();
    a.slice(0, 1..3).flip(0).assign(&b).unwrap();
    let mut want = a_data.clone();
    want[1] = 20.0;
    want[2] = 10.0;
    assert_eq!(a.to_vec().unwrap(), want);
}

/// A store that reads the buffer through a transposing map is copied first.
#[test]
fn a_transposed_write_of_the_same_array_uses_the_old_contents() {
    let ctx = Context::new().unwrap();
    let n = 6usize;
    let a_data = values(n * n, 5, -1.0, 1.0);
    let a = Array::from_slice_on(&ctx, &a_data, [n, n]).unwrap();
    a.transpose().assign(&a).unwrap();
    let want: Vec<f32> = (0..n).flat_map(|y| (0..n).map(move |x| (x, y))).map(|(x, y)| a_data[y + n * x]).collect();
    assert_eq!(a.to_vec().unwrap(), want);
}

#[test]
fn errors() {
    let ctx = Context::new().unwrap();
    let a = Array::<f32>::zeros_on(&ctx, [4, 2]).unwrap();
    let b = Array::<f32>::zeros_on(&ctx, [3, 2]).unwrap();
    let err = a.assign(&b).unwrap_err();
    assert!(matches!(err.root(), LumaError::ShapeMismatch { .. }), "{err}");
    let poisoned = a.slice(0, ..9);
    let err = a.slice(0, ..2).assign(&poisoned).unwrap_err();
    assert!(matches!(err.root(), LumaError::InvalidShape(_)), "{err}");
}

/// Pending work that reads the target only through something another array still holds is
/// computed before the write too.
#[test]
fn expressions_held_elsewhere_keep_the_old_contents() {
    let ctx = Context::new().unwrap();
    let a = values(20, 6, -1.0, 1.0);
    let w = Array::from_slice_on(&ctx, &a, 20).unwrap();
    let x = &w * 2.0;
    let m = &x + 1.0;
    w.assign(&m).unwrap();
    let want: Vec<f32> = a.iter().map(|v| v * 2.0 + 1.0).collect();
    assert_close(&w.to_vec().unwrap(), &want, 1e-6, 1e-6);
    assert_close(&m.to_vec().unwrap(), &want, 1e-6, 1e-6);
    assert_close(&x.to_vec().unwrap(), &a.iter().map(|v| v * 2.0).collect::<Vec<_>>(), 1e-6, 1e-6);

    // Read through a shift: another element than the one being written.
    let w = Array::from_slice_on(&ctx, &a, 20).unwrap();
    w.assign(&w.shift(1) * 1.0 + &w * 0.5).unwrap();
    let want: Vec<f32> = (0..20).map(|i| a[(i + 19) % 20] + a[i] * 0.5).collect();
    assert_close(&w.to_vec().unwrap(), &want, 1e-6, 1e-6);
}

// ---- assigning into an array that is still an expression ----

/// `a` (`[ni, nj]`, scaled by 2) padded by `(2, 1)` along the first axis and `(1, 2)` along the
/// second with the edge values, on the host: `[ni + 3, nj + 3]`.
fn padded_host(a: &[f32], ni: usize, nj: usize) -> Vec<f32> {
    let (wi, wj) = (ni + 3, nj + 3);
    (0..wi * wj)
        .map(|k| {
            let (x, y) = (k % wi, k / wi);
            let (i, j) = ((x as isize - 2).clamp(0, ni as isize - 1) as usize, (y as isize - 1).clamp(0, nj as isize - 1) as usize);
            a[i + ni * j] * 2.0
        })
        .collect()
}

fn lazy_padded(ctx: &Context, a: &[f32], ni: usize, nj: usize) -> Array<f32> {
    let a = Array::from_slice_on(ctx, a, [ni, nj]).unwrap();
    (&a * 2.0).pad([(2, 1), (1, 2)], PadMode::Edge)
}

#[test]
fn a_ghost_fill_of_a_lazy_padded_array_equals_the_eager_result() {
    let ctx = Context::new().unwrap();
    let (ni, nj) = (5usize, 4usize);
    let a = values(ni * nj, 11, -1.0, 1.0);
    let wi = ni + 3;
    let mut want = padded_host(&a, ni, nj);
    let old = want.clone();
    for y in 0..nj + 3 {
        want[wi * y] = -old[3 + wi * y];
        want[1 + wi * y] = -old[2 + wi * y];
    }

    let w = lazy_padded(&ctx, &a, ni, nj);
    w.slice(0, ..2).assign(w.slice(0, 2..4).flip(0) * -1.0).unwrap();
    assert_close(&w.to_vec().unwrap(), &want, 1e-6, 1e-6);

    // The same statements on an evaluated array (the buffer is written in place).
    let w = lazy_padded(&ctx, &a, ni, nj).cache();
    w.slice(0, ..2).assign(w.slice(0, 2..4).flip(0) * -1.0).unwrap();
    assert_close(&w.to_vec().unwrap(), &want, 1e-6, 1e-6);

    // By reference, with the expression kept alive.
    let w = lazy_padded(&ctx, &a, ni, nj);
    let g = w.slice(0, 2..4).flip(0) * -1.0;
    w.slice(0, ..2).assign(&g).unwrap();
    assert_close(&w.to_vec().unwrap(), &want, 1e-6, 1e-6);
    let want_g: Vec<f32> = (0..nj + 3).flat_map(|y| [-old[3 + wi * y], -old[2 + wi * y]]).collect();
    assert_close(&g.to_vec().unwrap(), &want_g, 1e-6, 1e-6);
}

#[test]
fn successive_assignments_see_what_the_earlier_ones_wrote() {
    let ctx = Context::new().unwrap();
    let (ni, nj) = (5usize, 4usize);
    let a = values(ni * nj, 12, -1.0, 1.0);
    let (wi, wj) = (ni + 3, nj + 3);
    let mut want = padded_host(&a, ni, nj);
    for y in 0..wj {
        want[wi * y] = -want[3 + wi * y];
        want[1 + wi * y] = -want[2 + wi * y];
    }
    // Rows `wj - 2..wj` read rows `3..5` (flipped), the columns the first write filled included.
    let snapshot = want.clone();
    for x in 0..wi {
        want[x + wi * (wj - 2)] = snapshot[x + wi * 4] * 0.5;
        want[x + wi * (wj - 1)] = snapshot[x + wi * 3] * 0.5;
    }

    let w = lazy_padded(&ctx, &a, ni, nj);
    w.slice(0, ..2).assign(w.slice(0, 2..4).flip(0) * -1.0).unwrap();
    w.slice(1, wj - 2..).assign(w.slice(1, 3..5).flip(1) * 0.5).unwrap();
    assert_close(&w.to_vec().unwrap(), &want, 1e-6, 1e-6);
}

#[test]
fn what_was_recorded_before_a_lazy_write_keeps_the_old_values() {
    let ctx = Context::new().unwrap();
    let (ni, nj) = (5usize, 4usize);
    let a = values(ni * nj, 13, -1.0, 1.0);
    let wi = ni + 3;
    let before = padded_host(&a, ni, nj);

    let w = lazy_padded(&ctx, &a, ni, nj);
    let old = &w * 1.0;
    let s = w.slice(1, 1..3);
    let g = w.slice(0, ..1);
    g.assign(7.0).unwrap();

    assert_close(&old.to_vec().unwrap(), &before, 1e-6, 1e-6);
    let want_s: Vec<f32> = (1..3).flat_map(|y| (0..wi).map(move |x| (x, y))).map(|(x, y)| before[x + wi * y]).collect();
    assert_close(&s.to_vec().unwrap(), &want_s, 1e-6, 1e-6);
    // The receiver shows the assigned data, and so does the array.
    assert_eq!(g.to_vec().unwrap(), vec![7.0; nj + 3]);
    let got = w.to_vec().unwrap();
    for (k, v) in got.iter().enumerate() {
        assert_eq!(*v, if k % wi == 0 { 7.0 } else { before[k] }, "{k}");
    }
    // A view taken from the receiver afterwards reads the assigned data as well.
    assert_eq!(g.slice(1, 1..3).to_vec().unwrap(), vec![7.0; 2]);
}

#[test]
fn flips_and_transposes_write_into_a_lazy_array() {
    let ctx = Context::new().unwrap();
    let a = values(5 * 4, 14, -1.0, 1.0);
    let base = Array::from_slice_on(&ctx, &a, [5, 4]).unwrap();

    let w = &base * 2.0;
    let b_data = values(2 * 4, 19, -1.0, 1.0);
    let b = Array::from_slice_on(&ctx, &b_data, [2, 4]).unwrap();
    w.flip(0).slice(0, ..2).assign(&b).unwrap();
    let mut want: Vec<f32> = a.iter().map(|v| v * 2.0).collect();
    for y in 0..4 {
        want[4 + 5 * y] = b_data[2 * y];
        want[3 + 5 * y] = b_data[1 + 2 * y];
    }
    assert_close(&w.to_vec().unwrap(), &want, 1e-6, 1e-6);

    let w = &base * 2.0;
    let c = values(5, 15, -1.0, 1.0);
    let c_arr = Array::from_slice_on(&ctx, &c, [1, 5]).unwrap();
    w.transpose().slice(0, ..1).assign(&c_arr).unwrap();
    let mut want: Vec<f32> = a.iter().map(|v| v * 2.0).collect();
    want[..5].copy_from_slice(&c);
    assert_close(&w.to_vec().unwrap(), &want, 1e-6, 1e-6);
}

#[test]
fn a_slice_of_everything_makes_the_array_the_right_hand_side() {
    let ctx = Context::new().unwrap();
    let a = values(12, 16, -1.0, 1.0);
    let b = values(12, 17, -1.0, 1.0);
    let base = Array::from_slice_on(&ctx, &a, [4, 3]).unwrap();
    let rhs = Array::from_slice_on(&ctx, &b, [4, 3]).unwrap();

    let mut w = &base + 1.0;
    w.slice(0, ..).assign(&rhs).unwrap();
    assert_eq!(w.to_vec().unwrap(), b);
    // The array has a buffer of its own: changing it leaves the right-hand side alone.
    w.write(&[0.0; 12]).unwrap();
    assert_eq!(rhs.to_vec().unwrap(), b);

    let w = &base + 1.0;
    w.slice(0, ..).assign(-1.0).unwrap();
    assert_eq!(w.to_vec().unwrap(), vec![-1.0; 12]);
}

#[test]
fn an_equal_expression_built_independently_keeps_its_ghost_cells() {
    let ctx = Context::new().unwrap();
    let (ni, nj) = (5usize, 4usize);
    let a = values(ni * nj, 18, -1.0, 1.0);
    let before = padded_host(&a, ni, nj);
    let w = lazy_padded(&ctx, &a, ni, nj);
    let w2 = lazy_padded(&ctx, &a, ni, nj);
    w.slice(0, ..2).assign(0.0).unwrap();
    assert_close(&w2.to_vec().unwrap(), &before, 1e-6, 1e-6);
    let got = w.to_vec().unwrap();
    assert_eq!(got[0], 0.0);
    assert_eq!(got[1], 0.0);
}

#[test]
fn errors_of_a_lazy_write_are_those_of_an_evaluated_one() {
    let ctx = Context::new().unwrap();
    let a = Array::<f32>::zeros_on(&ctx, [4, 2]).unwrap();
    let w = &a + 1.0;
    let bad_shape = Array::<f32>::zeros_on(&ctx, [3, 2]).unwrap();
    let err = w.slice(0, ..2).assign(&bad_shape).unwrap_err();
    assert!(matches!(err.root(), LumaError::ShapeMismatch { .. }), "{err}");

    let other = Context::new().unwrap();
    let foreign = Array::<f32>::zeros_on(&other, [2, 2]).unwrap();
    let err = w.slice(0, ..2).assign(&foreign).unwrap_err();
    assert!(matches!(err.root(), LumaError::ContextMismatch), "{err}");
    // Nothing was written.
    assert_eq!(w.to_vec().unwrap(), vec![1.0; 8]);
}
