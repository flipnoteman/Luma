//! `Array<Mask>`: the result of comparisons, its operators, reductions and casts.

use luma::*;

fn f32s(data: &[f32]) -> Array<f32> {
    Array::from_slice(data, data.len()).unwrap()
}

#[test]
fn comparisons_are_masks() {
    let x = f32s(&[1.0, -2.0, 0.0, 3.5, f32::NAN]);
    let m: Array<Mask> = x.gt(0.0);
    assert_eq!(m.to_vec().unwrap(), [Mask::TRUE, Mask::FALSE, Mask::FALSE, Mask::TRUE, Mask::FALSE]);
    assert_eq!(m.to_u32().to_vec().unwrap(), [1, 0, 0, 1, 0]);
    assert_eq!(m.to_f32().to_vec().unwrap(), [1.0, 0.0, 0.0, 1.0, 0.0]);
    assert!(bool::from(Mask::from(true)) && !bool::from(Mask::FALSE));
    let classes: Array<Mask> = x.is_nan();
    assert_eq!(classes.to_u32().to_vec().unwrap(), [0, 0, 0, 0, 1]);
}

#[test]
fn masks_combine_and_count() {
    let xs: Vec<f32> = (0..1000).map(|i| ((i * 37) % 101) as f32 - 50.0).collect();
    let ys: Vec<f32> = (0..1000).map(|i| ((i * 53) % 97) as f32 - 48.0).collect();
    let (x, y) = (f32s(&xs), f32s(&ys));
    let (m, m2) = (x.gt(0.0), y.lt(10.0));
    let want = xs.iter().zip(&ys).filter(|&(&a, &b)| a <= 0.0 && b < 10.0).count() as u32;
    assert_eq!((!&m & &m2).count().item().unwrap(), want);
    let want_or = xs.iter().zip(&ys).filter(|&(&a, &b)| a > 0.0 || b < 10.0).count() as u32;
    assert_eq!((&m | &m2).count().item().unwrap(), want_or);
    let want_xor = xs.iter().zip(&ys).filter(|&(&a, &b)| (a > 0.0) != (b < 10.0)).count() as u32;
    assert_eq!((&m ^ &m2).count().item().unwrap(), want_xor);
    let mut acc = m.clone();
    acc &= &m2;
    assert_eq!(acc.count().item().unwrap(), xs.iter().zip(&ys).filter(|&(&a, &b)| a > 0.0 && b < 10.0).count() as u32);
}

#[test]
fn any_and_all() {
    let x = f32s(&[1.0, 2.0, 3.0]);
    let read = |m: Array<Mask>| bool::from(m.item().unwrap());
    assert!(read(x.gt(2.5).any()));
    assert!(!read(x.gt(2.5).all()));
    assert!(read(x.gt(0.5).all()));
    assert!(!read(x.gt(5.0).any()));
    // Masks stay lazy: the reduction of a pending comparison fuses with it.
    assert!(read((&x * 2.0).gt(5.0).any()));
}

#[test]
fn select_takes_masks() {
    let x = f32s(&[-1.0, 2.0, -3.0, 4.0]);
    assert_eq!(select(&x.gt(0.0), 1.0, 0.0).to_vec().unwrap(), [0.0, 1.0, 0.0, 1.0]);
    assert_eq!(select(&x.gt(0.0), &x, 0.0).to_vec().unwrap(), [0.0, 2.0, 0.0, 4.0]);
    assert_eq!(x.sum_where(&x.lt(0.0)).item().unwrap(), -4.0);
    let from_host = Array::from_slice(&[Mask::TRUE, Mask::FALSE, Mask::TRUE, Mask::FALSE], 4).unwrap();
    assert_eq!(select(&from_host, &x, 9.0).to_vec().unwrap(), [-1.0, 9.0, -3.0, 9.0]);
    assert_eq!(from_host.eq(Mask::TRUE).to_u32().to_vec().unwrap(), [1, 0, 1, 0]);
}

#[test]
fn sums_of_masks_are_rejected() {
    let m = f32s(&[1.0, 2.0]).gt(0.0);
    for poisoned in [m.sum(), m.reduce([Reduction::Sum, Reduction::Max]), m.cumsum(0), m.sum_axis(0)] {
        let error = poisoned.to_vec().unwrap_err();
        assert!(matches!(error.root(), LumaError::Unsupported(text) if text.contains("Mask")), "{error}");
    }
    assert_eq!(m.min().item().unwrap(), Mask::TRUE);
    assert_eq!(m.max_axis(0).to_u32().to_vec().unwrap(), [1]);
}
