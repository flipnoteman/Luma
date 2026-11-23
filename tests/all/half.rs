//! The half-precision float arrays (`f16`, `bf16`): stored two to a `u32` word, every operation
//! computed in `f32` and rounded to the type, checked against the `half` crate on the host: the
//! conversions bit for bit (every `f16` pattern, 2^20 random `f32`, the halfway points and the
//! edges), elementwise operations with per-operation rounding, reductions against an `f32`
//! accumulator, casts, matmul, packing at odd lengths and the operations that do not exist.

use luma::*;

use crate::common::values;

fn lcg(n: usize, seed: u64) -> Vec<u64> {
    let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            state >> 24
        })
        .collect()
}

/// Lengths around the word boundary, and a few bigger ones.
const SIZES: [usize; 12] = [1, 2, 3, 4, 5, 7, 8, 9, 31, 33, 1000, 4097];

/// A context whose buffers hold 1024 bytes: arrays of 512 or more elements span several buffers.
fn chunked() -> Context {
    Context::with_options(ContextOptions::default().chunk_bytes(1024)).expect("a device is available")
}

/// The same position on the number line for both signs: halves are equal within `ulps` steps.
fn ordered(bits: u16) -> i32 {
    let magnitude = i32::from(bits & 0x7fff);
    if bits & 0x8000 != 0 { -magnitude } else { magnitude }
}

macro_rules! half_suite {
    ($m:ident, $t:ident, $to:ident, $qnan:expr, $mant:expr) => {
        mod $m {
            use super::*;

            type T = $t;
            const QNAN: u16 = $qnan;
            /// Bits of mantissa: the relative size of one rounding.
            const MANT: i32 = $mant;

            fn t(x: f32) -> T {
                T::from_f32(x)
            }

            /// The bits of `x`, a NaN as the canonical quiet one.
            fn canon(x: T) -> u16 {
                if x.is_nan() { QNAN } else { x.to_bits() }
            }

            fn same(got: &[T], want: &[T], what: &str) {
                assert_eq!(got.len(), want.len(), "{what}: length");
                for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
                    assert_eq!(canon(g), canon(w), "{what}: element {i}: got {g:?} ({:#06x}), want {w:?} ({:#06x})", g.to_bits(), w.to_bits());
                }
            }

            /// Equal within `ulps` steps of the type (NaN equals NaN).
            fn close(got: &[T], want: &[T], ulps: i32, what: &str) {
                assert_eq!(got.len(), want.len(), "{what}: length");
                for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
                    if g.is_nan() || w.is_nan() {
                        assert!(g.is_nan() && w.is_nan(), "{what}: element {i}: got {g:?}, want {w:?}");
                    } else if g.to_f32().abs() < f32::MIN_POSITIVE && w.to_f32().abs() < f32::MIN_POSITIVE {
                        // Both are below the normal range of `f32` (`bf16` has its range): a device may flush it to zero.
                    } else {
                        let d = (ordered(g.to_bits()) - ordered(w.to_bits())).abs();
                        assert!(d <= ulps, "{what}: element {i}: got {g:?}, want {w:?} ({d} steps)");
                    }
                }
            }

            /// Edge values first, then values over a wide range of magnitudes.
            fn data(n: usize, seed: u64) -> Vec<T> {
                let edges = [
                    0.0f32, -0.0, 1.0, -1.0, 0.5, 65504.0, -65504.0, 65520.0, 6.0e-8, -6.0e-8, 1.0e-5, 3.0e38, f32::INFINITY, f32::NEG_INFINITY, f32::NAN, 255.5, 256.5, 2047.0, 2049.0, 0.1, 100.0,
                ];
                lcg(n, seed)
                    .iter()
                    .enumerate()
                    .map(|(i, &v)| {
                        if i % 5 == 0 {
                            t(edges[(v as usize) % edges.len()])
                        } else {
                            let unit = ((v & 0xffff) as f32) / 65536.0 - 0.5;
                            t(unit * 2.0f32.powi((v >> 16) as i32 % 20 - 6))
                        }
                    })
                    .collect()
            }

            /// Finite values of moderate size: no overflow, nothing that is a NaN.
            fn tame(n: usize, seed: u64) -> Vec<T> {
                values(n, seed, -4.0, 4.0).into_iter().map(t).collect()
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
                        same(&x.to_vec().unwrap(), &d, &format!("n = {n}"));
                        let mut out = vec![T::ZERO; n];
                        x.read_into(&mut out).unwrap();
                        same(&out, &d, &format!("read_into, n = {n}"));
                        if n > 1 {
                            let y = Array::from_slice_on(&ctx, &d[1..], n - 1).unwrap();
                            same(&y.to_vec().unwrap(), &d[1..], &format!("offset slice, n = {n}"));
                        }
                    }
                }
            }

            #[test]
            fn full_zeros_and_write() {
                let ctx = Context::new().unwrap();
                for n in SIZES {
                    let z = Array::<T>::zeros_on(&ctx, n).unwrap();
                    same(&z.to_vec().unwrap(), &vec![T::ZERO; n], "zeros");
                    for v in [t(0.0), t(1.5), t(-65504.0), T::INFINITY, T::MIN_POSITIVE, t(7.0)] {
                        let f = Array::<T>::full_on(&ctx, n, v).unwrap();
                        same(&f.to_vec().unwrap(), &vec![v; n], &format!("full {v:?}, n = {n}"));
                    }
                    let d = data(n, 2);
                    let mut x = Array::<T>::zeros_on(&ctx, n).unwrap();
                    x.write(&d).unwrap();
                    same(&x.to_vec().unwrap(), &d, "write");
                }
            }

            #[test]
            fn elementwise_rounds_every_operation() {
                for ctx in contexts() {
                    for n in SIZES {
                        let (a, b) = (data(n, 3), data(n, 4));
                        let (x, y) = (array(&ctx, &a), array(&ctx, &b));
                        let host = |f: &dyn Fn(f32, f32) -> f32| -> Vec<T> { a.iter().zip(&b).map(|(&p, &q)| t(f(p.to_f32(), q.to_f32()))).collect() };
                        same(&(&x + &y).to_vec().unwrap(), &host(&|p, q| p + q), &format!("add, n = {n}"));
                        same(&(&x - &y).to_vec().unwrap(), &host(&|p, q| p - q), "sub");
                        same(&(&x * &y).to_vec().unwrap(), &host(&|p, q| p * q), "mul");
                        close(&(&x / &y).to_vec().unwrap(), &host(&|p, q| p / q), 1, "div");
                        // (Which operand a minimum with a NaN returns is up to the device.)
                        let (p, q): (Vec<T>, Vec<T>) = a.iter().zip(&b).filter(|(p, q)| !p.is_nan() && !q.is_nan()).map(|(&p, &q)| (p, q)).unzip();
                        if !p.is_empty() {
                        close(&array(&ctx, &p).minimum(&array(&ctx, &q)).to_vec().unwrap(), &p.iter().zip(&q).map(|(&p, &q)| if p.to_f32() < q.to_f32() { p } else { q }).collect::<Vec<_>>(), 0, "minimum");
                        }
                        same(&(-&x).to_vec().unwrap(), &a.iter().map(|&p| -p).collect::<Vec<_>>(), "neg");
                        same(&x.abs().to_vec().unwrap(), &a.iter().map(|&p| T::from_bits(p.to_bits() & 0x7fff)).collect::<Vec<_>>(), "abs");
                        // With a scalar on either side.
                        let k = t(3.25);
                        same(&(&x * k).to_vec().unwrap(), &a.iter().map(|&p| t(p.to_f32() * 3.25)).collect::<Vec<_>>(), "mul scalar");
                        same(&(k - &x).to_vec().unwrap(), &a.iter().map(|&p| t(3.25 - p.to_f32())).collect::<Vec<_>>(), "scalar sub");
                        // Comparisons and the tests of a class.
                        assert_eq!(x.lt(&y).to_vec().unwrap(), a.iter().zip(&b).map(|(p, q)| p.to_f32() < q.to_f32()).collect::<Vec<_>>(), "lt");
                        assert_eq!(x.eq(&y).to_vec().unwrap(), a.iter().zip(&b).map(|(p, q)| p.to_f32() == q.to_f32()).collect::<Vec<_>>(), "eq");
                        assert_eq!(x.is_nan().to_vec().unwrap(), a.iter().map(|p| p.is_nan()).collect::<Vec<_>>(), "is_nan");
                        assert_eq!(x.is_inf().to_vec().unwrap(), a.iter().map(|p| p.is_infinite()).collect::<Vec<_>>(), "is_inf");
                        assert_eq!(x.is_finite().to_vec().unwrap(), a.iter().map(|p| p.is_finite()).collect::<Vec<_>>(), "is_finite");
                        let mask = x.lt(&y);
                        same(&select(&mask, &x, &y).to_vec().unwrap(), &a.iter().zip(&b).map(|(&p, &q)| if p.to_f32() < q.to_f32() { p } else { q }).collect::<Vec<_>>(), "select");
                    }
                }
            }

            #[test]
            fn math_functions_round_an_f32_result() {
                let ctx = Context::new().unwrap();
                let a = tame(1000, 5);
                let pos: Vec<T> = a.iter().map(|&p| t(p.to_f32().abs() + 0.01)).collect();
                let (x, p) = (array(&ctx, &a), array(&ctx, &pos));
                let host = |d: &[T], f: &dyn Fn(f32) -> f32| -> Vec<T> { d.iter().map(|&v| t(f(v.to_f32()))).collect() };
                close(&x.exp().to_vec().unwrap(), &host(&a, &|v| v.exp()), 1, "exp");
                close(&p.ln().to_vec().unwrap(), &host(&pos, &|v| v.ln()), 1, "ln");
                close(&p.sqrt().to_vec().unwrap(), &host(&pos, &|v| v.sqrt()), 1, "sqrt");
                close(&x.sin().to_vec().unwrap(), &host(&a, &|v| v.sin()), 1, "sin");
                close(&x.tanh().to_vec().unwrap(), &host(&a, &|v| v.tanh()), 1, "tanh");
                same(&x.floor().to_vec().unwrap(), &host(&a, &|v| v.floor()), "floor");
                same(&x.ceil().to_vec().unwrap(), &host(&a, &|v| v.ceil()), "ceil");
                same(&x.trunc().to_vec().unwrap(), &host(&a, &|v| v.trunc()), "trunc");
                same(&x.sign().to_vec().unwrap(), &host(&a, &|v| if v == 0.0 { 0.0 } else { v.signum() }), "sign");
                close(&x.atan().to_vec().unwrap(), &host(&a, &|v| v.atan()), 1, "atan");
                close(&p.log2().to_vec().unwrap(), &host(&pos, &|v| v.log2()), 1, "log2");
                close(&p.rsqrt().to_vec().unwrap(), &host(&pos, &|v| 1.0 / v.sqrt()), 1, "rsqrt");
                close(&p.pow(t(1.5)).to_vec().unwrap(), &host(&pos, &|v| v.powf(1.5)), 1, "pow");
                close(&x.modulo(t(3.0)).to_vec().unwrap(), &host(&a, &|v| v - 3.0 * (v / 3.0).floor()), 1, "modulo");
                close(&x.sigmoid().to_vec().unwrap(), &host(&a, &|v| 1.0 / (1.0 + (-v).exp())), 1, "sigmoid");
            }

            #[test]
            fn a_fused_chain_equals_the_operations_one_by_one() {
                for ctx in contexts() {
                    let (a, b, c) = (tame(777, 6), tame(777, 7), tame(777, 8));
                    let (x, y, z) = (array(&ctx, &a), array(&ctx, &b), array(&ctx, &c));
                    let fused = (&(&(&x * &y) + &z) * &x).to_vec().unwrap();
                    // One operation at a time: every step is evaluated and stored before the next.
                    let step1 = &x * &y;
                    step1.eval().unwrap();
                    let step2 = &step1 + &z;
                    step2.eval().unwrap();
                    let eager = (&step2 * &x).to_vec().unwrap();
                    same(&fused, &eager, "fused vs eager");
                    let host: Vec<T> = (0..777)
                        .map(|i| {
                            let p = t(a[i].to_f32() * b[i].to_f32());
                            let q = t(p.to_f32() + c[i].to_f32());
                            t(q.to_f32() * a[i].to_f32())
                        })
                        .collect();
                    same(&fused, &host, "fused vs host");
                }
            }

            #[test]
            fn in_place_operators() {
                let ctx = Context::new().unwrap();
                for n in SIZES {
                    let (a, b) = (tame(n, 9), tame(n, 10));
                    let mut x = array(&ctx, &a);
                    let y = array(&ctx, &b);
                    let keep = array(&ctx, &a);
                    x += &y;
                    x *= t(0.5);
                    let want: Vec<T> = a.iter().zip(&b).map(|(&p, &q)| t(t(p.to_f32() + q.to_f32()).to_f32() * 0.5)).collect();
                    same(&x.to_vec().unwrap(), &want, "in place");
                    same(&keep.to_vec().unwrap(), &a, "the original is untouched");
                }
            }

            #[test]
            fn views_at_odd_offsets_and_writes_through_them() {
                let ctx = Context::new().unwrap();
                let d = tame(1001, 11);
                let x = array(&ctx, &d);
                for (from, to) in [(0usize, 1001usize), (1, 1001), (1, 1000), (3, 500), (501, 1001), (7, 8)] {
                    same(&x.slice(0, from..to).to_vec().unwrap(), &d[from..to], &format!("slice {from}..{to}"));
                }
                same(&x.flip(0).to_vec().unwrap(), &d.iter().rev().copied().collect::<Vec<_>>(), "flip");
                let m = Array::from_slice_on(&ctx, &d[..77 * 13], [77, 13]).unwrap();
                let tr = m.transpose().to_vec().unwrap();
                let want: Vec<T> = (0..77 * 13).map(|i| d[(i % 13) * 77 + i / 13]).collect();
                same(&tr, &want, "transpose");
                let mut w = Array::<T>::zeros_on(&ctx, 1001).unwrap();
                w.slice_mut(0, 1..1000).assign(&x.slice(0, 2..1001)).unwrap();
                let mut want = vec![T::ZERO; 1001];
                want[1..1000].copy_from_slice(&d[2..1001]);
                same(&w.to_vec().unwrap(), &want, "write through a view");
                let both = concat(0, [&x, &x]).to_vec().unwrap();
                same(&both, &[d.clone(), d.clone()].concat(), "concat");
            }

            #[test]
            fn gathers_and_filters() {
                let ctx = Context::new().unwrap();
                for n in [1usize, 2, 3, 300, 1001] {
                    let d = tame(n, 21);
                    let x = array(&ctx, &d);
                    let idx: Vec<u32> = lcg(100, 22).iter().map(|&v| (v % n as u64) as u32).collect();
                    let got = x.take(&Array::from_slice_on(&ctx, &idx, 100).unwrap()).to_vec().unwrap();
                    same(&got, &idx.iter().map(|&i| d[i as usize]).collect::<Vec<_>>(), "take");
                    let kept = x.filter(&x.gt(t(0.0))).unwrap().map(|a| a.to_vec().unwrap()).unwrap_or_default();
                    same(&kept, &d.iter().copied().filter(|v| v.to_f32() > 0.0).collect::<Vec<_>>(), "filter");
                }
            }

            #[test]
            fn reductions_accumulate_in_f32_and_round_once() {
                for ctx in contexts() {
                    for n in SIZES {
                        let a = tame(n, 12);
                        let x = array(&ctx, &a);
                        let sum: f64 = a.iter().map(|v| f64::from(v.to_f32())).sum();
                        close(&[x.sum().item().unwrap()], &[t(sum as f32)], 1, &format!("sum n = {n}"));
                        assert_eq!(canon(x.min().item().unwrap()), canon(*a.iter().min_by(|p, q| p.partial_cmp(q).unwrap()).unwrap()), "min");
                        assert_eq!(canon(x.max().item().unwrap()), canon(*a.iter().max_by(|p, q| p.partial_cmp(q).unwrap()).unwrap()), "max");
                        close(&[x.mean().item().unwrap()], &[t((sum / n as f64) as f32)], 1, "mean");
                        let near_one: Vec<T> = values(n.min(20), 13, 0.9, 1.1).into_iter().map(t).collect();
                        let prod: f64 = near_one.iter().map(|v| f64::from(v.to_f32())).product();
                        close(&[array(&ctx, &near_one).prod().item().unwrap()], &[t(prod as f32)], 1, "prod");
                        // The sum of an expression: the operands are not stored first.
                        let doubled: f64 = a.iter().map(|v| f64::from(t(v.to_f32() + v.to_f32()).to_f32())).sum();
                        close(&[(&x + &x).sum().item().unwrap()], &[t(doubled as f32)], 1, "sum of an expression");
                    }
                }
                // Many terms: an f16 accumulator would stall at 2048; the f32 one does not.
                let ctx = Context::new().unwrap();
                let ones = Array::<T>::full_on(&ctx, 100_000, t(1.0)).unwrap();
                assert_eq!(ones.sum().item().unwrap().to_f32(), t(100_000.0).to_f32(), "sum of 1e5 ones");
            }

            #[test]
            fn reductions_along_axes() {
                let ctx = Context::new().unwrap();
                for (w, h) in [(1usize, 1usize), (3, 5), (5, 3), (17, 33), (64, 40), (131, 9)] {
                    let a = tame(w * h, 14);
                    let m = Array::from_slice_on(&ctx, &a, [w, h]).unwrap();
                    let rows: Vec<&[T]> = a.chunks(w).collect();
                    let sum0: Vec<T> = rows.iter().map(|r| t(r.iter().map(|v| f64::from(v.to_f32())).sum::<f64>() as f32)).collect();
                    close(&m.sum_axis(0).to_vec().unwrap(), &sum0, 1, &format!("sum_axis 0 {w}x{h}"));
                    let sum1: Vec<T> = (0..w).map(|c| t(rows.iter().map(|r| f64::from(r[c].to_f32())).sum::<f64>() as f32)).collect();
                    close(&m.sum_axis(1).to_vec().unwrap(), &sum1, 1, "sum_axis 1");
                    let min1: Vec<T> = (0..w).map(|c| *rows.iter().map(|r| &r[c]).min_by(|p, q| p.partial_cmp(q).unwrap()).unwrap()).collect();
                    same(&m.min_axis(1).to_vec().unwrap(), &min1, "min_axis");
                    let max0: Vec<T> = rows.iter().map(|r| *r.iter().max_by(|p, q| p.partial_cmp(q).unwrap()).unwrap()).collect();
                    same(&m.max_axis(0).to_vec().unwrap(), &max0, "max_axis");
                    let mean1: Vec<T> = (0..w).map(|c| t((rows.iter().map(|r| f64::from(r[c].to_f32())).sum::<f64>() / h as f64) as f32)).collect();
                    close(&m.mean_axis(1).to_vec().unwrap(), &mean1, 1, "mean_axis");
                }
            }

            #[test]
            fn casts_in_both_directions() {
                for (k, ctx) in contexts().into_iter().enumerate() {
                    for n in SIZES {
                        let a = data(n, 15);
                        let x = array(&ctx, &a);
                        let wide = x.to_f32().to_vec().unwrap();
                        for (i, (&g, &v)) in wide.iter().zip(&a).enumerate() {
                            assert!(if v.is_nan() { g.is_nan() } else { g.to_bits() == v.to_f32().to_bits() }, "to_f32 element {i}: {g} vs {v:?}");
                        }
                        // f32 to the type, to the type and back, i32 and the other half type.
                        let f: Vec<f32> = wide.iter().map(|&v| if v.is_finite() { v * 1.001 } else { v }).collect();
                        let back = Array::from_slice_on(&ctx, &f, n).unwrap().$to().to_vec().unwrap();
                        same(&back, &f.iter().map(|&v| t(v)).collect::<Vec<_>>(), "f32 -> type");
                        let ints: Vec<i32> = lcg(n, 16).iter().map(|&v| (v as i32) >> (v % 17)).collect();
                        let from_i = Array::from_slice_on(&ctx, &ints, n).unwrap().$to().to_vec().unwrap();
                        same(&from_i, &ints.iter().map(|&v| t(v as f32)).collect::<Vec<_>>(), "i32 -> type");
                        let to_i = x.to_i32().to_vec().unwrap();
                        // Saturating, to the nearest `f32` inside the range; NaN is 0.
                        let want: Vec<i32> = a.iter().map(|v| v.to_f32().clamp(-2147483648.0, 2147483520.0) as i32).collect();
                        assert_eq!(to_i, want, "type -> i32");
                        let to_u = x.to_u32().to_vec().unwrap();
                        assert_eq!(to_u, a.iter().map(|v| v.to_f32().clamp(0.0, 4294967040.0) as u32).collect::<Vec<_>>(), "type -> u32");
                        // (Arrays of different element sizes do not mix over several buffers.)
                        if k == 0 && ctx.features().contains(Features::SHADER_F64) {
                            let d: Vec<f64> = x.to_f64().to_vec().unwrap();
                            for (g, v) in d.iter().zip(&a) {
                                assert!(if v.is_nan() { g.is_nan() } else { *g == v.to_f64() }, "to_f64");
                            }
                            let rounded = Array::from_slice_on(&ctx, &d, n).unwrap().$to().to_vec().unwrap();
                            same(&rounded, &a, "f64 -> type");
                        }
                    }
                }
                let ctx = Context::new().unwrap();
                let a = data(1000, 17);
                let x = array(&ctx, &a);
                // Through the other half type and back: two roundings.
                let other: Vec<f32> = x.to_f32().to_vec().unwrap();
                let via_f16 = x.to_f32().to_f16().to_f32().to_vec().unwrap();
                for (g, v) in via_f16.iter().zip(&other) {
                    assert!(if v.is_nan() { g.is_nan() } else { g.to_bits() == half::f16::from_f32(*v).to_f32().to_bits() }, "f32 -> f16 -> f32");
                }
            }

            #[test]
            fn matmul_against_the_host() {
                let ctx = Context::with_options(ContextOptions::from_env().tf32_matmul(false)).unwrap();
                for (k, m, n) in [(1usize, 1usize, 1usize), (3, 4, 5), (17, 13, 9), (64, 48, 40), (130, 33, 70)] {
                    let (a, b) = (tame(k * m, 18), tame(k * n, 19));
                    let (ma, mb) = (Array::from_slice_on(&ctx, &a, [k, m]).unwrap(), Array::from_slice_on(&ctx, &b, [n, k]).unwrap());
                    let got = ma.matmul(&mb).to_vec().unwrap();
                    assert_eq!(got.len(), n * m);
                    for row in 0..m {
                        for col in 0..n {
                            let sum: f64 = (0..k).map(|i| f64::from(a[i + k * row].to_f32()) * f64::from(b[col + n * i].to_f32())).sum();
                            let g = f64::from(got[col + n * row].to_f32());
                            let tol = sum.abs().max(1.0) * 2.0f64.powi(1 - MANT) + 1e-3 * k as f64 * 2.0f64.powi(-MANT);
                            assert!((g - sum).abs() <= tol, "matmul {k}x{m}x{n} at ({row},{col}): {g} vs {sum}");
                        }
                    }
                    // A lazy operand is fused into the loads.
                    let lazy = (&ma + &ma).matmul(&mb).to_vec().unwrap();
                    let twice = ma.matmul(&mb).to_vec().unwrap();
                    for (l, g) in lazy.iter().zip(&twice) {
                        let (l, g) = (f64::from(l.to_f32()), f64::from(g.to_f32()));
                        assert!((l - 2.0 * g).abs() <= 2.0 * g.abs().max(1.0) * 2.0f64.powi(2 - MANT) + 1e-2, "lazy operand {l} vs {g}");
                    }
                }
            }

            #[test]
            fn operations_that_do_not_exist_are_errors() {
                let ctx = Context::new().unwrap();
                let x = array(&ctx, &tame(100, 20));
                let name = std::any::type_name::<T>().rsplit("::").next().unwrap();
                let check = |what: &str, error: Option<String>| {
                    let message = error.unwrap_or_else(|| panic!("{what} did not fail"));
                    assert!(message.contains(name) && message.contains("not supported"), "{what}: {message}");
                };
                check("sort", x.sort().to_vec().err().map(|e| e.to_string()));
                check("sort_descending", x.sort_descending().to_vec().err().map(|e| e.to_string()));
                check("argsort", x.argsort().to_vec().err().map(|e| e.to_string()));
                check("sort_axis", x.sort_axis(0).to_vec().err().map(|e| e.to_string()));
                check("unique", x.unique().to_vec().err().map(|e| e.to_string()));
                check("cumsum", x.cumsum(0).to_vec().err().map(|e| e.to_string()));
                check("searchsorted", x.searchsorted(&x, luma::Side::Left).to_vec().err().map(|e| e.to_string()));
                check("topk", x.topk(3).0.to_vec().err().map(|e| e.to_string()));
                // The error is on the result, at the call: the array itself is fine.
                assert!(x.error().is_none());
                assert!(x.sort().error().is_some());
            }

            #[test]
            fn overflow_goes_to_infinity() {
                let ctx = Context::new().unwrap();
                let big = array(&ctx, &[t(60000.0), t(-60000.0), t(3.0e38)]);
                let doubled = (&big * t(2.0)).to_vec().unwrap();
                let want = [t(60000.0 * 2.0), t(-60000.0 * 2.0), t(3.0e38 * 2.0)];
                same(&doubled, &want, "x * 2");
                let sum = array(&ctx, &[t(60000.0), t(60000.0)]).sum().item().unwrap();
                same(&[sum], &[t(120000.0)], "the sum rounds at the end");
            }
        }
    };
}

half_suite!(f16_tests, f16, to_f16, 0x7e00, 11);
half_suite!(bf16_tests, bf16, to_bf16, 0x7fc0, 8);

/// A bit-exact comparison of the conversions with the `half` crate, in every kernel that has
/// them: the store kernels (built as Kernel IR) and, through a reduction over lines of one element,
/// the generated WGSL of the composite kernels.
mod conversions {
    use super::*;

    fn canon16(x: half::f16) -> u16 {
        if x.is_nan() { 0x7e00 } else { x.to_bits() }
    }

    fn canonb(x: half::bf16) -> u16 {
        if x.is_nan() { 0x7fc0 } else { x.to_bits() }
    }

    /// `f32` values that are hard to round: both neighbours of every halfway point between
    /// adjacent `f16` and `bf16` values over a range of exponents, the extremes, zeros,
    /// subnormals and NaN payloads, then random bits.
    fn hard_f32s() -> Vec<f32> {
        let mut out = vec![
            0.0f32, -0.0, 1.0, -1.0, f32::INFINITY, f32::NEG_INFINITY, f32::MAX, f32::MIN, f32::MIN_POSITIVE, 65504.0, 65519.99, 65520.0, 65536.0, 6.1035156e-5, 5.9604645e-8, 2.9802322e-8, 2.9802326e-8, 2.9802319e-8,
        ];
        for payload in [1u32, 0x1fff, 0x2000, 0x3fffff, 0x400000, 0x7fffff] {
            out.push(f32::from_bits(0x7f80_0000 | payload));
            out.push(f32::from_bits(0xff80_0000 | payload));
        }
        // Subnormals of f32 and halfway points: the midpoint of every adjacent pair of f16, then
        // one step of f32 on either side of it.
        for h in 0u16..0x7c00 {
            let (a, b) = (half::f16::from_bits(h).to_f32(), half::f16::from_bits(h + 1).to_f32());
            let mid = f32::from_bits((a.to_bits() + b.to_bits()) / 2);
            // (a + b) / 2 in f32 is exact for neighbours; the bits' mean is too when the exponent is equal.
            for m in [(a + b) / 2.0, mid] {
                for step in [-1i64, 0, 1] {
                    let v = f32::from_bits((i64::from(m.to_bits()) + step) as u32);
                    out.push(v);
                    out.push(-v);
                }
            }
        }
        for h in (0u16..0x7f80).step_by(7) {
            let (a, b) = (half::bf16::from_bits(h).to_f32(), half::bf16::from_bits(h + 1).to_f32());
            let m = (a + b) / 2.0;
            for step in [-1i64, 0, 1] {
                out.push(f32::from_bits((i64::from(m.to_bits()) + step) as u32));
            }
        }
        let mut state = 0x9e3779b97f4a7c15u64;
        while out.len() < (1 << 20) + 200_000 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            out.push(f32::from_bits((state >> 16) as u32));
        }
        out
    }

    #[test]
    fn every_f16_pattern_widens_exactly_and_rounds_back() {
        let ctx = Context::new().unwrap();
        let bits: Vec<u16> = (0..=u16::MAX).collect();
        let halves: Vec<half::f16> = bits.iter().map(|&b| half::f16::from_bits(b)).collect();
        let x = Array::from_slice_on(&ctx, &halves, halves.len()).unwrap();
        // The bits survive the buffer.
        assert_eq!(x.to_vec().unwrap().iter().map(|h| h.to_bits()).collect::<Vec<_>>(), bits);
        let wide = x.to_f32().to_vec().unwrap();
        for (&b, &w) in bits.iter().zip(&wide) {
            let h = half::f16::from_bits(b);
            assert!(if h.is_nan() { w.is_nan() } else { w.to_bits() == h.to_f32().to_bits() }, "{b:#06x}: {w}");
        }
        let round_trip = x.to_f32().to_f16().to_vec().unwrap();
        assert_eq!(round_trip.iter().map(|&h| canon16(h)).collect::<Vec<_>>(), halves.iter().map(|&h| canon16(h)).collect::<Vec<_>>());
        // And the same values through the generated source of the composite kernels: a reduction of
        // lines of one element passes each through the conversions of its fused program.
        let lines = Array::from_slice_on(&ctx, &halves, [1, halves.len()]).unwrap();
        let through = lines.to_f32().sum_axis(0).to_vec().unwrap();
        for (&b, &w) in bits.iter().zip(&through) {
            let h = half::f16::from_bits(b);
            assert!(if h.is_nan() { w.is_nan() } else { w == h.to_f32() }, "reduction path {b:#06x}: {w}");
        }
    }

    #[test]
    fn every_bf16_pattern_widens_exactly_and_rounds_back() {
        let ctx = Context::new().unwrap();
        let halves: Vec<half::bf16> = (0..=u16::MAX).map(half::bf16::from_bits).collect();
        let x = Array::from_slice_on(&ctx, &halves, halves.len()).unwrap();
        let wide = x.to_f32().to_vec().unwrap();
        for (h, &w) in halves.iter().zip(&wide) {
            assert!(if h.is_nan() { w.is_nan() } else { w.to_bits() == h.to_f32().to_bits() }, "{:#06x}: {w}", h.to_bits());
        }
        let round_trip = x.to_f32().to_bf16().to_vec().unwrap();
        assert_eq!(round_trip.iter().map(|&h| canonb(h)).collect::<Vec<_>>(), halves.iter().map(|&h| canonb(h)).collect::<Vec<_>>());
    }

    #[test]
    fn f32_rounds_to_nearest_even_like_the_half_crate() {
        let ctx = Context::new().unwrap();
        let values = hard_f32s();
        let x = Array::from_slice_on(&ctx, &values, values.len()).unwrap();
        let got = x.to_f16().to_vec().unwrap();
        for (i, (&v, &g)) in values.iter().zip(&got).enumerate() {
            let want = half::f16::from_f32(v);
            assert_eq!(canon16(g), canon16(want), "f16 of {v:e} ({:#010x}), element {i}", v.to_bits());
        }
        let got = x.to_bf16().to_vec().unwrap();
        for (i, (&v, &g)) in values.iter().zip(&got).enumerate() {
            let want = half::bf16::from_f32(v);
            assert_eq!(canonb(g), canonb(want), "bf16 of {v:e} ({:#010x}), element {i}", v.to_bits());
        }
    }

    #[test]
    fn the_generated_source_rounds_the_same_way() {
        let ctx = Context::new().unwrap();
        let values = hard_f32s();
        // Lines of one element: the sum is the value itself (up to the sign of a zero).
        let lines = Array::from_slice_on(&ctx, &values, [1, values.len()]).unwrap();
        let f16s = lines.to_f16().to_f32().sum_axis(0).to_vec().unwrap();
        let bf16s = lines.to_bf16().to_f32().sum_axis(0).to_vec().unwrap();
        for (i, &v) in values.iter().enumerate() {
            let (w16, wb) = (half::f16::from_f32(v).to_f32(), half::bf16::from_f32(v).to_f32());
            assert!(if w16.is_nan() { f16s[i].is_nan() } else { f16s[i] == w16 }, "f16 of {v:e} ({:#010x}): {} vs {w16}", v.to_bits(), f16s[i]);
            assert!(if wb.is_nan() { bf16s[i].is_nan() } else { bf16s[i] == wb }, "bf16 of {v:e} ({:#010x}): {} vs {wb}", v.to_bits(), bf16s[i]);
        }
    }
}
