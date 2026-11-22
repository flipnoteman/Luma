//! The 8-bit and 16-bit integer arrays (`u8`, `i8`, `u16`, `i16`): stored 4 or 2 to a `u32`
//! word, every operation wrapping like Rust's `wrapping_*`, checked against the host: edge
//! values, lengths that are not multiples of a word, views at odd offsets, every cast pair,
//! reductions, counting and sorting, and arrays split over several buffers.

use luma::*;

fn lcg(n: usize, seed: u64) -> Vec<u64> {
    let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            state >> 24
        })
        .collect()
}

fn has_f64(ctx: &Context) -> bool {
    ctx.features().contains(Features::SHADER_F64)
}

/// Lengths around the word boundaries of both layouts, and a few bigger ones.
const SIZES: [usize; 13] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 31, 33, 1000, 4097];

/// A context whose buffers hold 1024 bytes: arrays of 256 or more elements span several buffers.
fn chunked() -> Context {
    Context::with_options(ContextOptions::default().chunk_bytes(1024)).expect("a device is available")
}

/// The values every type is tested on first: both ends of the range, around zero and the sign bit.
fn edge<T: Copy>(from: impl Fn(i64) -> T) -> Vec<T> {
    [0i64, 1, 2, 3, 7, 8, 9, 15, 16, 31, 32, 63, 64, 100, 126, 127, 128, 129, 200, 254, 255, 256, 257, 32766, 32767, 32768, 32769, 65534, 65535]
        .iter()
        .flat_map(|&v| [from(v), from(-v), from(i64::MAX - v)])
        .collect()
}

macro_rules! small_suite {
    ($m:ident, $t:ident, to: $to:ident, others: [$(($other:ident, $o:ident)),*], bits: $bits:expr, wide: $wide:ident) => {
        mod $m {
            use super::*;

            type T = $t;
            const BITS: u32 = $bits;

            fn from(v: i64) -> T {
                v as T
            }

            fn data(n: usize, seed: u64) -> Vec<T> {
                let edges = edge(from);
                lcg(n, seed).iter().enumerate().map(|(i, &v)| if i % 3 == 0 { edges[(v as usize) % edges.len()] } else { v as T }).collect()
            }

            fn array(ctx: &Context, d: &[T]) -> Array<T> {
                Array::from_slice_on(ctx, d, d.len()).unwrap()
            }

            fn contexts() -> Vec<Context> {
                vec![Context::new().unwrap(), chunked()]
            }

            #[test]
            fn upload_and_readback_at_every_length() {
                for ctx in contexts() {
                    for n in SIZES.into_iter().chain([255, 256, 257, 511, 512, 513, 1025]) {
                        let d = data(n, 1);
                        let x = array(&ctx, &d);
                        assert_eq!(x.to_vec().unwrap(), d, "n = {n}");
                        let mut out = vec![0 as T; n];
                        x.read_into(&mut out).unwrap();
                        assert_eq!(out, d, "read_into, n = {n}");
                        // The host slice itself may sit at an odd address.
                        if n > 1 {
                            let y = Array::from_slice_on(&ctx, &d[1..], n - 1).unwrap();
                            assert_eq!(y.to_vec().unwrap(), d[1..], "offset slice, n = {n}");
                        }
                    }
                }
            }

            #[test]
            fn full_zeros_and_write() {
                let ctx = Context::new().unwrap();
                for n in SIZES {
                    let z = Array::<T>::zeros_on(&ctx, n).unwrap();
                    assert_eq!(z.to_vec().unwrap(), vec![0 as T; n]);
                    for v in [0 as T, 1, 5, T::MAX, T::MIN, (T::MAX / 2)] {
                        let f = Array::<T>::full_on(&ctx, n, v).unwrap();
                        assert_eq!(f.to_vec().unwrap(), vec![v; n], "full {v}, n = {n}");
                    }
                    let d = data(n, 2);
                    let mut x = Array::<T>::zeros_on(&ctx, n).unwrap();
                    x.write(&d).unwrap();
                    assert_eq!(x.to_vec().unwrap(), d, "write, n = {n}");
                }
            }

            #[test]
            fn arithmetic_wraps() {
                for ctx in contexts() {
                    for n in SIZES {
                        let (a, b) = (data(n, 3), data(n, 4));
                        let nz: Vec<T> = b.iter().map(|&v| if v == 0 { 1 } else { v }).collect();
                        let (x, y, z) = (array(&ctx, &a), array(&ctx, &b), array(&ctx, &nz));
                        let check = |got: Array<T>, want: Vec<T>, what: &str| assert_eq!(got.to_vec().unwrap(), want, "{what}, n = {n}");
                        check(&x + &y, a.iter().zip(&b).map(|(p, q)| p.wrapping_add(*q)).collect(), "add");
                        check(&x - &y, a.iter().zip(&b).map(|(p, q)| p.wrapping_sub(*q)).collect(), "sub");
                        check(&x * &y, a.iter().zip(&b).map(|(p, q)| p.wrapping_mul(*q)).collect(), "mul");
                        check(&x / &z, a.iter().zip(&nz).map(|(p, q)| p.wrapping_div(*q)).collect(), "div");
                        check(&x % &z, a.iter().zip(&nz).map(|(p, q)| p.wrapping_rem(*q)).collect(), "rem");
                        check(&x & &y, a.iter().zip(&b).map(|(p, q)| p & q).collect(), "and");
                        check(&x | &y, a.iter().zip(&b).map(|(p, q)| p | q).collect(), "or");
                        check(&x ^ &y, a.iter().zip(&b).map(|(p, q)| p ^ q).collect(), "xor");
                        check(!&x, a.iter().map(|p| !p).collect(), "not");
                        check(x.minimum(&y), a.iter().zip(&b).map(|(p, q)| *p.min(q)).collect(), "minimum");
                        check(x.maximum(&y), a.iter().zip(&b).map(|(p, q)| *p.max(q)).collect(), "maximum");
                        check(&x << &y, a.iter().zip(&b).map(|(p, q)| p.wrapping_shl(*q as u32)).collect(), "shl");
                        check(&x >> &y, a.iter().zip(&b).map(|(p, q)| p.wrapping_shr(*q as u32)).collect(), "shr");
                        // With scalars on either side, and a chain that fuses.
                        check(&x + (7 as T), a.iter().map(|p| p.wrapping_add(7)).collect(), "add scalar");
                        check((100 as T) - &x, a.iter().map(|p| (100 as T).wrapping_sub(*p)).collect(), "scalar sub");
                        check(&x * (3 as T), a.iter().map(|p| p.wrapping_mul(3)).collect(), "mul scalar");
                        check(x.shl(3 as T), a.iter().map(|p| p.wrapping_shl(3)).collect(), "shl 3");
                        check(x.shr(BITS as T), a.clone(), "shr by the width");
                        check(
                            ((&x * &y) + (&x - &y)) ^ (&x & (5 as T)),
                            a.iter().zip(&b).map(|(p, q)| p.wrapping_mul(*q).wrapping_add(p.wrapping_sub(*q)) ^ (p & 5)).collect(),
                            "fused chain",
                        );
                    }
                }
            }

            #[test]
            fn in_place_operators() {
                let ctx = Context::new().unwrap();
                for n in SIZES {
                    let (a, b) = (data(n, 5), data(n, 6));
                    let mut x = array(&ctx, &a);
                    let y = array(&ctx, &b);
                    x += &y;
                    x *= 3 as T;
                    x ^= &y;
                    x -= 1 as T;
                    let want: Vec<T> = a.iter().zip(&b).map(|(p, q)| ((p.wrapping_add(*q)).wrapping_mul(3) ^ q).wrapping_sub(1)).collect();
                    assert_eq!(x.to_vec().unwrap(), want, "n = {n}");
                    // A written-through copy sees the new values, an old expression the old ones.
                    let old = &y + (1 as T);
                    let mut y2 = y.clone();
                    y2 += (1 as T);
                    assert_eq!(old.to_vec().unwrap(), y2.to_vec().unwrap());
                }
            }

            #[test]
            fn comparisons_and_select() {
                for ctx in contexts() {
                    for n in SIZES {
                        let (a, b) = (data(n, 7), data(n, 8));
                        let (x, y) = (array(&ctx, &a), array(&ctx, &b));
                        let pairs = a.iter().zip(&b);
                        assert_eq!(x.lt(&y).to_vec().unwrap(), pairs.clone().map(|(p, q)| p < q).collect::<Vec<_>>(), "lt n = {n}");
                        assert_eq!(x.le(&y).to_vec().unwrap(), pairs.clone().map(|(p, q)| p <= q).collect::<Vec<_>>(), "le");
                        assert_eq!(x.gt(&y).to_vec().unwrap(), pairs.clone().map(|(p, q)| p > q).collect::<Vec<_>>(), "gt");
                        assert_eq!(x.ge(&y).to_vec().unwrap(), pairs.clone().map(|(p, q)| p >= q).collect::<Vec<_>>(), "ge");
                        assert_eq!(x.eq(&y).to_vec().unwrap(), pairs.clone().map(|(p, q)| p == q).collect::<Vec<_>>(), "eq");
                        assert_eq!(x.ne(&y).to_vec().unwrap(), pairs.clone().map(|(p, q)| p != q).collect::<Vec<_>>(), "ne");
                        assert_eq!(x.lt(5 as T).to_vec().unwrap(), a.iter().map(|&p| p < 5).collect::<Vec<_>>(), "lt scalar");
                        let picked = select(&x.lt(&y), &x, &y);
                        assert_eq!(picked.to_vec().unwrap(), pairs.clone().map(|(p, q)| if p < q { *p } else { *q }).collect::<Vec<_>>(), "select");
                        let by_scalar = select(&x.ge(&y), 9 as T, &y);
                        assert_eq!(by_scalar.to_vec().unwrap(), pairs.map(|(p, q)| if p >= q { 9 } else { *q }).collect::<Vec<_>>(), "select scalar");
                    }
                }
            }

            #[test]
            fn two_outputs_of_different_widths_in_one_kernel() {
                let ctx = Context::new().unwrap();
                for n in SIZES {
                    let a = data(n, 9);
                    let x = array(&ctx, &a);
                    let small = &x + (1 as T);
                    let wide = x.to_f32() * 2.0;
                    let flag = x.gt(5 as T);
                    eval_all(&[&small]).unwrap();
                    assert_eq!(small.to_vec().unwrap(), a.iter().map(|p| p.wrapping_add(1)).collect::<Vec<_>>());
                    assert_eq!(wide.to_vec().unwrap(), a.iter().map(|&p| p as f32 * 2.0).collect::<Vec<_>>());
                    assert_eq!(flag.to_vec().unwrap(), a.iter().map(|&p| p > 5).collect::<Vec<_>>());
                }
            }

            #[test]
            fn views_at_odd_offsets() {
                for ctx in contexts() {
                    let n = 1013;
                    let a = data(n, 10);
                    let x = array(&ctx, &a);
                    for (start, end) in [(0usize, 1usize), (1, 2), (1, 1013), (3, 9), (5, 1012), (2, 7), (255, 258), (509, 515), (1, 1000)] {
                        let s = x.slice(0, start..end);
                        assert_eq!(s.to_vec().unwrap(), a[start..end], "slice {start}..{end}");
                        let shifted = (s.clone() + (1 as T)).to_vec().unwrap();
                        assert_eq!(shifted, a[start..end].iter().map(|p| p.wrapping_add(1)).collect::<Vec<_>>());
                    }
                    for shift in [0i32, 1, 2, 3, 5, -1, -3, 1012, 2000] {
                        let want: Vec<T> = (0..n as i64).map(|i| a[(i - shift as i64).rem_euclid(n as i64) as usize]).collect();
                        assert_eq!(x.shift([shift as isize]).to_vec().unwrap(), want, "shift {shift}");
                    }
                    assert_eq!(x.flip(0).to_vec().unwrap(), a.iter().rev().copied().collect::<Vec<_>>(), "flip");
                    let want: Vec<T> = (0..n).map(|i| a[(i * 7 + 3) % n]).collect();
                    let idx: Vec<u32> = (0..n).map(|i| ((i * 7 + 3) % n) as u32).collect();
                    let idx = Array::from_slice_on(&ctx, &idx, n).unwrap();
                    assert_eq!(x.take(&idx).to_vec().unwrap(), want, "take");
                }
            }

            #[test]
            fn views_of_matrices() {
                for ctx in contexts() {
                    for (w, h) in [(1usize, 1usize), (3, 5), (5, 3), (7, 9), (33, 17), (64, 64), (130, 31)] {
                        let a = data(w * h, 11);
                        let m = Array::from_slice_on(&ctx, &a, [w, h]).unwrap();
                        let t = m.transpose();
                        let want: Vec<T> = (0..w).flat_map(|c| (0..h).map(move |r| (r, c))).map(|(r, c)| a[r * w + c]).collect();
                        assert_eq!(t.to_vec().unwrap(), want, "transpose {w}x{h}");
                        assert_eq!((t.clone() + (1 as T)).to_vec().unwrap(), want.iter().map(|p| p.wrapping_add(1)).collect::<Vec<_>>());
                        if w > 2 && h > 2 {
                            let sub = m.slice(0, 1..w - 1).slice(1, 1..h - 1);
                            let want: Vec<T> = (1..h - 1).flat_map(|r| (1..w - 1).map(move |c| (r, c))).map(|(r, c)| a[r * w + c]).collect();
                            assert_eq!(sub.to_vec().unwrap(), want, "interior {w}x{h}");
                        }
                        let padded = m.pad([(1, 2), (2, 1)], PadMode::Zero);
                        let (pw, ph) = (w + 3, h + 3);
                        let want: Vec<T> = (0..ph)
                            .flat_map(|r| (0..pw).map(move |c| (r, c)))
                            .map(|(r, c)| if r >= 2 && r < 2 + h && c >= 1 && c < 1 + w { a[(r - 2) * w + (c - 1)] } else { 0 })
                            .collect();
                        assert_eq!(padded.to_vec().unwrap(), want, "pad {w}x{h}");
                        let moved = m.shift_fill([1, 2], 7 as T);
                        let want: Vec<T> = (0..h as i64)
                            .flat_map(|r| (0..w as i64).map(move |c| (r, c)))
                            .map(|(r, c)| if r - 2 < 0 || c - 1 < 0 { 7 } else { a[((r - 2) as usize) * w + (c - 1) as usize] })
                            .collect();
                        assert_eq!(moved.to_vec().unwrap(), want, "shift_fill {w}x{h}");
                        let clamped = m.shift_clamp([-1, 1]);
                        let want: Vec<T> = (0..h as i64)
                            .flat_map(|r| (0..w as i64).map(move |c| (r, c)))
                            .map(|(r, c)| a[((r - 1).clamp(0, h as i64 - 1) as usize) * w + (c + 1).clamp(0, w as i64 - 1) as usize])
                            .collect();
                        assert_eq!(clamped.to_vec().unwrap(), want, "shift_clamp {w}x{h}");
                    }
                }
            }

            #[test]
            fn writes_through_views() {
                let ctx = Context::new().unwrap();
                let n = 37;
                let a = data(n, 12);
                for (start, end) in [(0usize, 5usize), (1, 6), (3, 30), (5, 37), (7, 9)] {
                    let mut x = array(&ctx, &a);
                    x.slice_mut(0, start..end).assign(11 as T).unwrap();
                    let mut want = a.clone();
                    want[start..end].iter_mut().for_each(|v| *v = 11);
                    assert_eq!(x.to_vec().unwrap(), want, "assign {start}..{end}");
                    let src = array(&ctx, &data(end - start, 13));
                    let mut y = array(&ctx, &a);
                    y.slice_mut(0, start..end).assign(&src + (1 as T)).unwrap();
                    let mut want = a.clone();
                    want[start..end].iter_mut().zip(data(end - start, 13)).for_each(|(v, s)| *v = s.wrapping_add(1));
                    assert_eq!(y.to_vec().unwrap(), want, "assign array {start}..{end}");
                }
                // Scatter of values at indices (the elements of one word written by different invocations).
                let mut x = array(&ctx, &a);
                let idx: Vec<u32> = (0..n as u32).rev().collect();
                let idx = Array::from_slice_on(&ctx, &idx, n).unwrap();
                let vals = array(&ctx, &data(n, 14));
                x.put(&idx, &vals).unwrap();
                let want: Vec<T> = data(n, 14).into_iter().rev().collect();
                assert_eq!(x.to_vec().unwrap(), want, "put");
            }

            #[test]
            fn reductions_widen() {
                for (k, ctx) in contexts().into_iter().enumerate() {
                    for n in SIZES {
                        let a = data(n, 15);
                        let x = array(&ctx, &a);
                        let wide: $wide = a.iter().fold(0 as $wide, |s, &v| s.wrapping_add(v as $wide));
                        assert_eq!(x.sum().item().unwrap(), wide, "sum n = {n}");
                        assert_eq!(x.min().item().unwrap(), *a.iter().min().unwrap(), "min");
                        assert_eq!(x.max().item().unwrap(), *a.iter().max().unwrap(), "max");
                        let mean = a.iter().map(|&v| f64::from(v)).sum::<f64>() / n as f64;
                        let got = f64::from(x.mean().item().unwrap());
                        assert!((got - mean).abs() <= 1e-4 * mean.abs().max(1.0), "mean {got} vs {mean}");
                        // (Scans, arg reductions and sorts do not run on arrays of several buffers.)
                        if k == 0 {
                        assert_eq!(x.argmax().item().unwrap() as usize, a.iter().position(|v| v == a.iter().max().unwrap()).unwrap(), "argmax");
                        assert_eq!(x.argmin().item().unwrap() as usize, a.iter().position(|v| v == a.iter().min().unwrap()).unwrap(), "argmin");
                        let running: Vec<$wide> = a.iter().scan(0 as $wide, |s, &v| { *s = s.wrapping_add(v as $wide); Some(*s) }).collect();
                        assert_eq!(x.cumsum(0).to_vec().unwrap(), running, "cumsum");
                        }
                        // A fused expression reduces without being stored.
                        let doubled: $wide = a.iter().fold(0 as $wide, |s, &v| s.wrapping_add(v.wrapping_add(v) as $wide));
                        assert_eq!((&x + &x).sum().item().unwrap(), doubled, "sum of an expression");
                    }
                }
            }

            #[test]
            fn reductions_along_axes() {
                {
                    let ctx = Context::new().unwrap();
                    for (w, h) in [(1usize, 1usize), (3, 5), (5, 3), (17, 33), (64, 40), (131, 9)] {
                        let a = data(w * h, 16);
                        let m = Array::from_slice_on(&ctx, &a, [w, h]).unwrap();
                        let rows: Vec<&[T]> = a.chunks(w).collect();
                        // Axis 0: along a row; axis 1: along a column.
                        let sum0: Vec<$wide> = rows.iter().map(|r| r.iter().fold(0 as $wide, |s, &v| s.wrapping_add(v as $wide))).collect();
                        assert_eq!(m.sum_axis(0).to_vec().unwrap(), sum0, "sum_axis 0 {w}x{h}");
                        let sum1: Vec<$wide> = (0..w).map(|c| rows.iter().fold(0 as $wide, |s, r| s.wrapping_add(r[c] as $wide))).collect();
                        assert_eq!(m.sum_axis(1).to_vec().unwrap(), sum1, "sum_axis 1 {w}x{h}");
                        assert_eq!(m.min_axis(1).to_vec().unwrap(), (0..w).map(|c| rows.iter().map(|r| r[c]).min().unwrap()).collect::<Vec<_>>(), "min_axis");
                        assert_eq!(m.max_axis(0).to_vec().unwrap(), rows.iter().map(|r| *r.iter().max().unwrap()).collect::<Vec<_>>(), "max_axis");
                        let mean: Vec<f64> = (0..w).map(|c| rows.iter().map(|r| f64::from(r[c])).sum::<f64>() / h as f64).collect();
                        let got = m.mean_axis(1).to_vec().unwrap();
                        for (g, w) in got.iter().zip(&mean) {
                            assert!((f64::from(*g) - w).abs() <= 1e-4 * w.abs().max(1.0), "mean_axis {g} vs {w}");
                        }
                        let var: Vec<f64> = (0..w)
                            .map(|c| {
                                let m = mean[c];
                                rows.iter().map(|r| (f64::from(r[c]) - m).powi(2)).sum::<f64>() / h as f64
                            })
                            .collect();
                        let got = m.var_axis(1).to_vec().unwrap();
                        for (g, w) in got.iter().zip(&var) {
                            assert!((f64::from(*g) - w).abs() <= 1e-3 * w.abs().max(1.0), "var_axis {g} vs {w}");
                        }
                    }
                }
            }

            #[test]
            fn sorting() {
                {
                    let ctx = Context::new().unwrap();
                    for n in [1usize, 2, 5, 33, 1000, 4097] {
                        let a = data(n, 18);
                        let x = array(&ctx, &a);
                        let mut up = a.clone();
                        up.sort();
                        assert_eq!(x.sort().to_vec().unwrap(), up, "sort n = {n}");
                        let mut down = a.clone();
                        down.sort_by(|p, q| q.cmp(p));
                        assert_eq!(x.sort_descending().to_vec().unwrap(), down, "sort_descending");
                        let mut idx: Vec<u32> = (0..n as u32).collect();
                        idx.sort_by_key(|&i| a[i as usize]);
                        assert_eq!(x.argsort().to_vec().unwrap(), idx, "argsort");
                    }
                }
                let ctx = Context::new().unwrap();
                let (w, h) = (17usize, 9usize);
                let a = data(w * h, 19);
                let m = Array::from_slice_on(&ctx, &a, [w, h]).unwrap();
                let want: Vec<T> = a.chunks(w).flat_map(|r| { let mut r = r.to_vec(); r.sort(); r }).collect();
                assert_eq!(m.sort_axis(0).to_vec().unwrap(), want, "sort_axis");
            }

            #[test]
            fn casts_from_every_type() {
                let ctx = Context::new().unwrap();
                for n in SIZES {
                    let a = data(n, 20);
                    let x = array(&ctx, &a);
                    assert_eq!(x.to_f32().to_vec().unwrap(), a.iter().map(|&v| v as f32).collect::<Vec<_>>(), "to_f32");
                    assert_eq!(x.to_u32().to_vec().unwrap(), a.iter().map(|&v| v as u32).collect::<Vec<_>>(), "to_u32");
                    assert_eq!(x.to_i32().to_vec().unwrap(), a.iter().map(|&v| v as i32).collect::<Vec<_>>(), "to_i32");
                    assert_eq!(x.to_bool().to_vec().unwrap(), a.iter().map(|&v| v != 0).collect::<Vec<_>>(), "to_bool");
                    $(assert_eq!(x.$other().to_vec().unwrap(), a.iter().map(|&v| v as $o).collect::<Vec<_>>(), stringify!($other));)*
                    assert_eq!(x.to_df32().to_vec().unwrap().iter().map(|d| d.to_f64()).collect::<Vec<_>>(), a.iter().map(|&v| v as f64).collect::<Vec<_>>(), "to_df32");
                    if has_f64(&ctx) {
                        assert_eq!(x.to_f64().to_vec().unwrap(), a.iter().map(|&v| v as f64).collect::<Vec<_>>(), "to_f64");
                    }
                    // A cast reads its operand inside the kernel that uses it, and a chain of casts fuses.
                    let round = x.to_i32().to_f32().to_i32().$to();
                    assert_eq!(round.to_vec().unwrap(), a, "round trip");
                }
            }

            #[test]
            fn casts_to_this_type_from_every_type() {
                let ctx = Context::new().unwrap();
                for n in SIZES {
                    let words: Vec<i64> = lcg(n, 21).iter().map(|&v| (v as i64) >> 8).collect();
                    let u: Vec<u32> = words.iter().map(|&v| v as u32).collect();
                    let i: Vec<i32> = words.iter().map(|&v| v as i32).collect();
                    let f: Vec<f32> = words.iter().enumerate().map(|(k, &v)| match k % 7 {
                        0 => f32::NAN,
                        1 => f32::INFINITY,
                        2 => f32::NEG_INFINITY,
                        3 => 1.0e10,
                        4 => -1.0e10,
                        _ => (v as f32) / 1000.0,
                    }).collect();
                    let flags: Vec<bool> = words.iter().map(|&v| v % 3 == 0).collect();
                    let from_u = Array::from_slice_on(&ctx, &u, n).unwrap().$to();
                    assert_eq!(from_u.to_vec().unwrap(), u.iter().map(|&v| v as T).collect::<Vec<_>>(), "from u32");
                    let from_i = Array::from_slice_on(&ctx, &i, n).unwrap().$to();
                    assert_eq!(from_i.to_vec().unwrap(), i.iter().map(|&v| v as T).collect::<Vec<_>>(), "from i32");
                    let from_f = Array::from_slice_on(&ctx, &f, n).unwrap().$to();
                    assert_eq!(from_f.to_vec().unwrap(), f.iter().map(|&v| v as T).collect::<Vec<_>>(), "from f32 (saturating)");
                    let from_b = Array::from_slice_on(&ctx, &flags, n).unwrap().$to();
                    assert_eq!(from_b.to_vec().unwrap(), flags.iter().map(|&v| v as T).collect::<Vec<_>>(), "from bool");
                    let d = Array::from_slice_on(&ctx, &f, n).unwrap().to_df32().$to();
                    assert_eq!(d.to_vec().unwrap(), f.iter().map(|&v| v as T).collect::<Vec<_>>(), "from Df32");
                    if has_f64(&ctx) {
                        let from_d = Array::from_slice_on(&ctx, &f, n).unwrap().to_f64().$to();
                        assert_eq!(from_d.to_vec().unwrap(), f.iter().map(|&v| v as T).collect::<Vec<_>>(), "from f64");
                    }
                }
            }


            #[test]
            fn filter_concat_stack_range_and_broadcast() {
                let ctx = Context::new().unwrap();
                for n in [1usize, 5, 33, 1000] {
                    let a = data(n, 23);
                    let x = array(&ctx, &a);
                    let flags: Vec<bool> = lcg(n, 24).iter().map(|&v| v % 3 != 0).collect();
                    let mask = Array::from_slice_on(&ctx, &flags, n).unwrap();
                    let want: Vec<T> = a.iter().zip(&flags).filter(|(_, f)| **f).map(|(v, _)| *v).collect();
                    match x.filter(&mask).unwrap() {
                        Some(kept) => assert_eq!(kept.to_vec().unwrap(), want, "filter n = {n}"),
                        None => assert!(want.is_empty()),
                    }
                    let (padded, count) = x.filter_padded(&mask, 9 as T);
                    let mut want_padded = want.clone();
                    want_padded.resize(n, 9);
                    assert_eq!(padded.to_vec().unwrap(), want_padded, "filter_padded n = {n}");
                    assert_eq!(count.item().unwrap() as usize, want.len());

                    let b = data(n, 25);
                    let y = array(&ctx, &b);
                    let joined = concat(0, [&x, &y]);
                    assert_eq!(joined.to_vec().unwrap(), [a.clone(), b.clone()].concat(), "concat n = {n}");
                    let stacked = stack(1, [&x, &y]);
                    assert_eq!(stacked.to_vec().unwrap(), [a.clone(), b.clone()].concat(), "stack n = {n}");
                    let top = stack(0, [&x, &y]);
                    let want: Vec<T> = (0..n).flat_map(|i| [a[i], b[i]]).collect();
                    assert_eq!(top.to_vec().unwrap(), want, "stack axis 0, n = {n}");
                    // Sums of stacked halves go through the widening cast.
                    assert_eq!(((&x + &y) * (2 as T)).reshape([1, n]).to_vec().unwrap(), a.iter().zip(&b).map(|(p, q)| p.wrapping_add(*q).wrapping_mul(2)).collect::<Vec<_>>());

                    let r = Array::<T>::range_on(&ctx, [n, 1], 0).unwrap();
                    assert_eq!(r.to_vec().unwrap(), (0..n).map(|i| i as T).collect::<Vec<_>>(), "range n = {n}");
                    let r = Array::<T>::range_on(&ctx, [3, n.min(7)], 1).unwrap();
                    assert_eq!(r.to_vec().unwrap(), (0..n.min(7)).flat_map(|j| [j as T; 3]).collect::<Vec<_>>(), "range axis 1");
                    assert_eq!(Array::<T>::range_on(&ctx, [n, 1], 0).unwrap().sum().item().unwrap(), (0..n).fold(0 as $wide, |s, i| s.wrapping_add((i as T) as $wide)), "sum of a range");

                    let col = Array::from_slice_on(&ctx, &a[..n.min(5)], [n.min(5), 1]).unwrap();
                    let row = Array::from_slice_on(&ctx, &b[..n.min(7)], [1, n.min(7)]).unwrap();
                    let grid = &col + &row;
                    let want: Vec<T> = (0..n.min(7)).flat_map(|j| (0..n.min(5)).map(move |i| (i, j))).map(|(i, j)| a[i].wrapping_add(b[j])).collect();
                    assert_eq!(grid.to_vec().unwrap(), want, "broadcast n = {n}");
                }
            }

            #[test]
            fn arrays_of_several_buffers_agree_with_one_buffer() {
                let one = Context::new().unwrap();
                let many = chunked();
                for n in [255usize, 256, 257, 1000, 5001] {
                    let a = data(n, 22);
                    let want = (array(&one, &a) * (3 as T) + (array(&one, &a).shift(5) ^ (17 as T))).to_vec().unwrap();
                    let x = array(&many, &a);
                    let got = (x.clone() * (3 as T) + (x.shift(5) ^ (17 as T))).to_vec().unwrap();
                    assert_eq!(got, want, "n = {n}");
                    assert_eq!(array(&many, &a).sum().item().unwrap(), array(&one, &a).sum().item().unwrap(), "sum n = {n}");
                }
            }
        }
    };
}

small_suite!(u8_tests, u8, to: to_u8, others: [(to_i8, i8), (to_u16, u16), (to_i16, i16)], bits: 8, wide: u32);
small_suite!(i8_tests, i8, to: to_i8, others: [(to_u8, u8), (to_u16, u16), (to_i16, i16)], bits: 8, wide: i32);
small_suite!(u16_tests, u16, to: to_u16, others: [(to_u8, u8), (to_i8, i8), (to_i16, i16)], bits: 16, wide: u32);
small_suite!(i16_tests, i16, to: to_i16, others: [(to_u8, u8), (to_i8, i8), (to_u16, u16)], bits: 16, wide: i32);

macro_rules! unsigned_suite {
    ($m:ident, $t:ident, bits: $bits:expr) => {
        mod $m {
            use super::*;

            #[test]
            fn counting() {
                let ctx = Context::new().unwrap();
                for n in SIZES {
                    let a: Vec<$t> = lcg(n, 17).iter().enumerate().map(|(i, &v)| if i % 3 == 0 { (v % 40) as $t } else { v as $t }).collect();
                    let x = Array::from_slice_on(&ctx, &a, n).unwrap();
                    let bins = 40;
                    let mut want = vec![0u32; bins];
                    a.iter().for_each(|&v| if (v as usize) < bins { want[v as usize] += 1 });
                    assert_eq!(x.bincount(bins).to_vec().unwrap(), want, "bincount n = {n}");
                    let all = x.histogram().to_vec().unwrap();
                    assert_eq!(all.len(), if $bits == 8 { 256 } else { 65536 });
                    assert_eq!(all.iter().sum::<u32>() as usize, n);
                    assert_eq!(&all[..bins], &want[..]);
                    // Counting an expression: the elements are not stored.
                    let shifted = (&x + (1 as $t)).bincount(bins).to_vec().unwrap();
                    let mut want = vec![0u32; bins];
                    a.iter().for_each(|&v| if (v.wrapping_add(1) as usize) < bins { want[v.wrapping_add(1) as usize] += 1 });
                    assert_eq!(shifted, want, "bincount of an expression");
                }
            }
        }
    };
}

unsigned_suite!(u8_counting, u8, bits: 8);
unsigned_suite!(u16_counting, u16, bits: 16);
