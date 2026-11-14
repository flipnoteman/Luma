//! `i32` arrays and the integer/float casts, against the host.

use luma::*;

fn i32s(data: &[i32]) -> Array<i32> {
    Array::from_slice(data, data.len()).unwrap()
}

/// Edge values followed by a pseudo-random spread.
fn words(n: usize, seed: u32) -> Vec<i32> {
    let mut s = seed | 1;
    let mut out = vec![0, 1, -1, i32::MIN, i32::MAX, 123_456_789, -987_654_321];
    while out.len() < n {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        out.push(s as i32);
    }
    out.truncate(n);
    out
}

#[test]
fn float_to_integer_casts_truncate_saturate_and_zero_nan() {
    let x = Array::<f32>::from_slice(&[-1.5, 2.7, f32::NAN, 1e20, -1e20], 5).unwrap();
    assert_eq!(x.to_u32().to_vec().unwrap(), [0, 2, 0, 4_294_967_040, 0]);
    assert_eq!(x.to_i32().to_vec().unwrap(), [-1, 2, 0, 2_147_483_520, i32::MIN]);
    let inf = Array::<f32>::from_slice(&[f32::INFINITY, f32::NEG_INFINITY], 2).unwrap();
    assert_eq!(inf.to_u32().to_vec().unwrap(), [4_294_967_040, 0]);
    assert_eq!(inf.to_i32().to_vec().unwrap(), [2_147_483_520, i32::MIN]);
}

#[test]
fn integer_casts_between_u32_i32_and_f32() {
    let signed = words(1000, 3);
    let bits: Vec<u32> = signed.iter().map(|&v| v as u32).collect();
    let x = i32s(&signed);
    assert_eq!(x.to_u32().to_vec().unwrap(), bits);
    assert_eq!(x.to_u32().to_i32().to_vec().unwrap(), signed);
    let u = Array::from_slice(&bits, bits.len()).unwrap();
    assert_eq!(u.to_i32().to_vec().unwrap(), signed);
    assert_eq!(u.to_f32().to_vec().unwrap(), bits.iter().map(|&v| v as f32).collect::<Vec<_>>());
    assert_eq!(x.to_f32().to_vec().unwrap(), signed.iter().map(|&v| v as f32).collect::<Vec<_>>());
}

#[test]
fn division_remainder_and_abs_edge_cases() {
    let (a, b) = (i32s(&[7, -7, i32::MIN]), i32s(&[0, 2, -1]));
    assert_eq!((&a / &b).to_vec().unwrap(), [7, -3, i32::MIN]);
    assert_eq!((&a % &b).to_vec().unwrap(), [0, -1, 0]);
    assert_eq!(i32s(&[i32::MIN, -5, 5, 0]).abs().to_vec().unwrap(), [i32::MIN, 5, 5, 0]);
    assert_eq!(i32s(&[i32::MIN, -5, 5, 0]).sign().to_vec().unwrap(), [-1, -1, 1, 0]);
}

#[test]
fn arithmetic_wraps_like_the_host() {
    let n = 1000;
    let (xs, ys) = (words(n, 1), words(n, 2));
    let (a, b) = (i32s(&xs), i32s(&ys));
    let zip = |f: fn(i32, i32) -> i32| -> Vec<i32> { xs.iter().zip(&ys).map(|(&x, &y)| f(x, y)).collect() };
    assert_eq!((&a + &b).to_vec().unwrap(), zip(i32::wrapping_add));
    assert_eq!((&a - &b).to_vec().unwrap(), zip(i32::wrapping_sub));
    assert_eq!((&a * &b).to_vec().unwrap(), zip(i32::wrapping_mul));
    assert_eq!((&a + 5).to_vec().unwrap(), xs.iter().map(|x| x.wrapping_add(5)).collect::<Vec<_>>());
    assert_eq!((-3 * &a).to_vec().unwrap(), xs.iter().map(|x| x.wrapping_mul(-3)).collect::<Vec<_>>());
    assert_eq!((-&a).to_vec().unwrap(), xs.iter().map(|x| x.wrapping_neg()).collect::<Vec<_>>());
    assert_eq!(a.minimum(&b).to_vec().unwrap(), zip(i32::min));
    assert_eq!(a.maximum(&b).to_vec().unwrap(), zip(i32::max));
    let nonzero: Vec<i32> = ys.iter().map(|&y| if y == 0 { 1 } else { y }).collect();
    let d = i32s(&nonzero);
    let want = |f: fn(i32, i32) -> i32| -> Vec<i32> { xs.iter().zip(&nonzero).map(|(&x, &y)| f(x, y)).collect() };
    assert_eq!((&a / &d).to_vec().unwrap(), want(i32::wrapping_div));
    assert_eq!((&a % &d).to_vec().unwrap(), want(i32::wrapping_rem));
    assert_eq!((&a % 7).to_vec().unwrap(), xs.iter().map(|x| x.wrapping_rem(7)).collect::<Vec<_>>());
}

#[test]
fn bitwise_and_arithmetic_shifts_match_the_host() {
    let n = 1000;
    let (xs, ys) = (words(n, 5), words(n, 6));
    let (a, b) = (i32s(&xs), i32s(&ys));
    let zip = |f: fn(i32, i32) -> i32| -> Vec<i32> { xs.iter().zip(&ys).map(|(&x, &y)| f(x, y)).collect() };
    assert_eq!((&a & &b).to_vec().unwrap(), zip(|x, y| x & y));
    assert_eq!((&a | &b).to_vec().unwrap(), zip(|x, y| x | y));
    assert_eq!((&a ^ &b).to_vec().unwrap(), zip(|x, y| x ^ y));
    assert_eq!((!&a).to_vec().unwrap(), xs.iter().map(|x| !x).collect::<Vec<_>>());
    assert_eq!((&a << &b).to_vec().unwrap(), zip(|x, y| x.wrapping_shl(y as u32)));
    assert_eq!((&a >> &b).to_vec().unwrap(), zip(|x, y| x.wrapping_shr(y as u32)));
    for k in [0, 1, 5, 31, 32, 33, -1] {
        assert_eq!(a.shl(k).to_vec().unwrap(), xs.iter().map(|x| x.wrapping_shl(k as u32)).collect::<Vec<_>>(), "shl {k}");
        assert_eq!(a.shr(k).to_vec().unwrap(), xs.iter().map(|x| x.wrapping_shr(k as u32)).collect::<Vec<_>>(), "shr {k}");
    }
    assert_eq!(i32s(&[-8, 8, i32::MIN]).shr(2).to_vec().unwrap(), [-2, 2, i32::MIN >> 2]);
}

#[test]
fn comparisons_order_negatives_below_positives() {
    let (xs, ys) = (words(1000, 7), words(1000, 8));
    let (a, b) = (i32s(&xs), i32s(&ys));
    let want = |f: fn(i32, i32) -> bool| -> Vec<u32> { xs.iter().zip(&ys).map(|(&x, &y)| u32::from(f(x, y))).collect() };
    assert_eq!(a.lt(&b).to_vec().unwrap(), want(|x, y| x < y));
    assert_eq!(a.ge(&b).to_vec().unwrap(), want(|x, y| x >= y));
    assert_eq!(a.eq(&b).to_vec().unwrap(), want(|x, y| x == y));
    assert_eq!(a.lt(0).to_vec().unwrap(), xs.iter().map(|&x| u32::from(x < 0)).collect::<Vec<_>>());
    let picked = select(&a.lt(0), &a, 0);
    assert_eq!(picked.to_vec().unwrap(), xs.iter().map(|&x| if x < 0 { x } else { 0 }).collect::<Vec<_>>());
}

#[test]
fn reductions_match_the_host() {
    let xs = words(1 << 20, 11);
    let x = i32s(&xs);
    assert_eq!(x.sum().item().unwrap(), xs.iter().fold(0i32, |s, &v| s.wrapping_add(v)));
    assert_eq!(x.min().item().unwrap(), *xs.iter().min().unwrap());
    assert_eq!(x.max().item().unwrap(), *xs.iter().max().unwrap());
    // A pending expression fuses into the reduction.
    assert_eq!((&x * 3).sum().item().unwrap(), xs.iter().fold(0i32, |s, &v| s.wrapping_add(v.wrapping_mul(3))));
}

#[test]
fn axis_reductions_match_the_host() {
    let (w, h) = (100, 37);
    let xs = words(w * h, 13);
    let x = Array::from_slice(&xs, [w, h]).unwrap();
    let rows = |f: fn(&[i32]) -> i32| -> Vec<i32> { xs.chunks(w).map(f).collect() };
    assert_eq!(x.sum_axis(0).to_vec().unwrap(), rows(|r| r.iter().fold(0, |s, &v| s.wrapping_add(v))));
    assert_eq!(x.min_axis(0).to_vec().unwrap(), rows(|r| *r.iter().min().unwrap()));
    assert_eq!(x.max_axis(0).to_vec().unwrap(), rows(|r| *r.iter().max().unwrap()));
    let cols: Vec<i32> = (0..w).map(|c| (0..h).map(|r| xs[c + r * w]).fold(0, i32::wrapping_add)).collect();
    assert_eq!(x.sum_axis(1).to_vec().unwrap(), cols);
}

#[test]
fn host_side_constructors_and_lazy_scalars() {
    assert_eq!(Array::<i32>::zeros(4).unwrap().to_vec().unwrap(), [0; 4]);
    assert_eq!(Array::<i32>::full(4, -9).unwrap().to_vec().unwrap(), [-9; 4]);
    let s = Scalar::new(-5i32);
    let x = i32s(&[1, 2, 3]);
    assert_eq!((&x + &s).to_vec().unwrap(), [-4, -3, -2]);
    s.set(10);
    assert_eq!((&x * &s).to_vec().unwrap(), [10, 20, 30]);
}
