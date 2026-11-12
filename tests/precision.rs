//! Extended precision: double-float (`Df32`) arithmetic, compensated reductions and `f64` arrays.

mod common;

use common::values;
use luma::{Array, Context, Df32, LumaError};

fn has_f64(ctx: &Context) -> bool {
    ctx.features().contains(luma::wgpu::Features::SHADER_F64)
}

/// The next float above `x` (positive, finite) for ulp comparisons.
fn ulp(x: f32) -> f32 {
    f32::from_bits(x.abs().to_bits() + 1) - x.abs()
}

#[test]
fn df32_guard_keeps_error_free_transformations() {
    let ctx = Context::new().unwrap();
    let a = Array::<f32>::full_on(&ctx, 64, 1.0e8).unwrap();
    let b = Array::<f32>::full_on(&ctx, 64, 1.0).unwrap();

    // (Plain f32 gives 0.0 for `(a + b) - a` here, or 1.0 where the shader compiler
    // reassociates it: not asserted.) A double-float keeps the 1.0 whatever the compiler does
    // with the algebra.
    let (da, db) = (a.to_df32(), b.to_df32());
    let kept = ((&da + &db) - &da).to_f32().to_vec().unwrap();
    assert!(kept.iter().all(|&v| v == 1.0), "{kept:?}");
}

#[test]
fn df32_sums_match_f64_and_do_not_depend_on_the_tree() {
    let ctx = Context::new().unwrap();
    let data = values(1 << 16, 5, 0.0, 1.0);
    let want: f64 = data.iter().map(|&v| f64::from(v)).sum();
    let x = Array::from_slice_on(&ctx, &data, data.len()).unwrap();

    let got = x.to_df32().sum().item().unwrap().to_f64();
    assert!((got - want).abs() <= 1e-9 * want, "df32 sum {got} vs {want}");

    let compensated = x.sum_compensated().item().unwrap();
    let rounded = want as f32;
    assert!((compensated - rounded).abs() <= ulp(rounded), "compensated {compensated} vs {rounded}");

    // The norm: the squares are accumulated in double-float, too.
    let norm_want = data.iter().map(|&v| f64::from(v) * f64::from(v)).sum::<f64>().sqrt() as f32;
    let norm = x.norm_compensated().item().unwrap();
    assert!((norm - norm_want).abs() <= 2.0 * ulp(norm_want), "norm {norm} vs {norm_want}");

    // Min and max reduce double-floats, too.
    let df = x.to_df32();
    let (lo, hi) = (df.min().item().unwrap().to_f64(), df.max().item().unwrap().to_f64());
    assert_eq!(lo, f64::from(data.iter().copied().fold(f32::INFINITY, f32::min)));
    assert_eq!(hi, f64::from(data.iter().copied().fold(f32::NEG_INFINITY, f32::max)));
}

#[test]
fn df32_arithmetic_is_accurate_to_about_44_bits() {
    let ctx = Context::new().unwrap();
    let wide: Vec<f64> = values(300, 9, 0.5, 2.0).iter().enumerate().map(|(i, &v)| f64::from(v) + i as f64 * 1e-9).collect();
    let data: Vec<Df32> = wide.iter().map(|&v| Df32::from(v)).collect();
    let exact: Vec<f64> = data.iter().map(|d| d.to_f64()).collect();
    let d = Array::from_slice_on(&ctx, &data, data.len()).unwrap();

    let got = (&d * &d / &d).sqrt().to_vec().unwrap();
    for ((g, v), i) in got.iter().zip(&exact).zip(0..) {
        let want = ((v * v) / v).sqrt();
        assert!((g.to_f64() - want).abs() <= 1e-10 * want, "element {i}: {} vs {want}", g.to_f64());
    }

    let sum = (&d + &d.abs()).to_vec().unwrap();
    for (g, v) in sum.iter().zip(&exact) {
        assert!((g.to_f64() - 2.0 * v).abs() <= 1e-12 * v, "{} vs {}", g.to_f64(), 2.0 * v);
    }

    // Comparisons and select go through the library's ordering.
    let big = d.gt(Df32::from(1.0));
    let picked = luma::select(&big, &d, Df32::from(0.0)).to_vec().unwrap();
    for (g, v) in picked.iter().zip(&exact) {
        assert_eq!(g.to_f64(), if *v > 1.0 { *v } else { 0.0 });
    }
    let (lo, hi) = (d.minimum(Df32::from(1.0)).to_vec().unwrap(), d.maximum(Df32::from(1.0)).to_vec().unwrap());
    for ((l, h), v) in lo.iter().zip(&hi).zip(&exact) {
        assert_eq!((l.to_f64(), h.to_f64()), (v.min(1.0), v.max(1.0)));
    }
}

#[test]
fn df32_views_and_assign_round_trip() {
    let ctx = Context::new().unwrap();
    let data: Vec<Df32> = (0..12).map(|i| Df32 { hi: i as f32, lo: i as f32 * 1e-9 }).collect();
    let a = Array::from_slice_on(&ctx, &data, [4, 3]).unwrap();

    let s = a.slice(0, 1..3);
    assert_eq!(s.shape().dims(), [2, 3, 1, 1]);
    let want: Vec<Df32> = (0..3).flat_map(|y| (1..3).map(move |x| (x, y))).map(|(x, y)| data[x + 4 * y]).collect();
    assert_eq!(s.to_vec().unwrap(), want);
    assert_eq!(a.flip(0).to_vec().unwrap()[0], data[3]);
    assert_eq!(a.shift_clamp(-1).to_vec().unwrap()[3], data[3]);

    let w = Array::<Df32>::zeros_on(&ctx, [4, 3]).unwrap();
    let row: Vec<Df32> = data[4..8].to_vec();
    let row = Array::from_slice_on(&ctx, &row, [4, 1]).unwrap();
    w.slice(1, 2..3).assign(&row).unwrap();
    let got = w.to_vec().unwrap();
    assert_eq!(&got[8..], &data[4..8]);
    assert!(got[..8].iter().all(|v| *v == Df32::default()));
}

/// Double-floats with a real low word: `hi` is the `f32` nearest to the value, `lo` the rest.
fn df_values(n: usize, seed: u64) -> Vec<Df32> {
    values(n, seed, -3.0, 3.0).iter().enumerate().map(|(i, &v)| Df32::from(f64::from(v) + (i as f64 + 1.0) * 1.0e-9)).collect()
}

#[test]
fn df32_ordering_kernels_without_arithmetic_bind_no_guard() {
    // These kernels call only `df_abs`, `df_lt`, `df_min`, `df_max`: they never read the guard
    // uniform, so it must not be part of the kernel's bindings.
    let ctx = Context::new().unwrap();
    let (da, db) = (df_values(130, 21), df_values(130, 22));
    let a = Array::from_slice_on(&ctx, &da, da.len()).unwrap();
    let b = Array::from_slice_on(&ctx, &db, db.len()).unwrap();
    let (fa, fb): (Vec<f64>, Vec<f64>) = (da.iter().map(|d| d.to_f64()).collect(), db.iter().map(|d| d.to_f64()).collect());

    let abs = a.abs().to_vec().unwrap();
    let min = a.minimum(&b).to_vec().unwrap();
    let max = a.maximum(&b).to_vec().unwrap();
    let lt = a.lt(&b).to_vec().unwrap();
    let le = a.le(&b).to_vec().unwrap();
    let eq = a.eq(&a).to_vec().unwrap();
    for i in 0..da.len() {
        assert_eq!(abs[i].to_f64(), fa[i].abs(), "abs {i}");
        assert_eq!(min[i].to_f64(), fa[i].min(fb[i]), "min {i}");
        assert_eq!(max[i].to_f64(), fa[i].max(fb[i]), "max {i}");
        assert_eq!(lt[i], u32::from(fa[i] < fb[i]), "lt {i}");
        assert_eq!(le[i], u32::from(fa[i] <= fb[i]), "le {i}");
        assert_eq!(eq[i], 1, "eq {i}");
    }

    // With a scalar operand, and in a reduction (`min`/`max` fuse their ordering into it).
    let capped = a.minimum(Df32::from(0.5)).to_vec().unwrap();
    assert!(capped.iter().zip(&fa).all(|(g, v)| g.to_f64() == v.min(0.5)));
    assert_eq!(a.abs().max().item().unwrap().to_f64(), fa.iter().fold(0.0f64, |m, v| m.max(v.abs())));
    assert_eq!(a.minimum(&b).min().item().unwrap().to_f64(), fa.iter().zip(&fb).fold(f64::INFINITY, |m, (x, y)| m.min(x.min(*y))));
}

#[test]
fn df32_casts_are_exact_under_fusion() {
    let ctx = Context::new().unwrap();
    let data = df_values(200, 23);
    let x = Array::from_slice_on(&ctx, &data, data.len()).unwrap();

    // `x - to_df32(to_f32(x))` is the low word of `x`, however the compiler reassociates.
    let low = (&x - x.to_f32().to_df32()).to_f32().to_vec().unwrap();
    for (g, d) in low.iter().zip(&data) {
        assert_eq!(*g, d.lo, "{d:?}");
    }

    // f32 -> Df32 -> f32 is the identity, also in the middle of cancelling arithmetic.
    let f = values(200, 24, -3.0, 3.0);
    let a = Array::from_slice_on(&ctx, &f, f.len()).unwrap();
    assert_eq!(a.to_df32().to_f32().to_vec().unwrap(), f);
    // `1e6 + g` (g in [1, 2)) fits a double-float exactly; `to_f32` of it is the rounded sum.
    let g = values(200, 25, 1.0, 2.0);
    let ga = Array::from_slice_on(&ctx, &g, g.len()).unwrap();
    let big = Array::<f32>::full_on(&ctx, g.len(), 1.0e6).unwrap();
    let sum = &ga.to_df32() + &big.to_df32();
    let back = (&sum - sum.to_f32().to_df32()).to_f32().to_vec().unwrap();
    let rest = ((&sum - &big.to_df32()).to_f32()).to_vec().unwrap();
    for i in 0..g.len() {
        // What is left of `sum` after the rounded sum is taken away: its rounding error.
        let exact = f64::from(g[i]) + 1.0e6;
        assert_eq!(f64::from(back[i]), exact - f64::from(exact as f32), "{i}");
        assert_eq!(rest[i], g[i], "{i}");
    }

    if has_f64(&ctx) {
        // Df32 -> f64 -> Df32 is the identity; the f64 difference sees the low word.
        let wide = x.to_f64();
        assert_eq!(wide.to_df32().to_vec().unwrap(), data);
        assert_eq!((&x - wide.to_df32()).to_f32().to_vec().unwrap(), vec![0.0; data.len()]);
        let lows = (&wide - x.to_f32().to_f64()).to_f32().to_vec().unwrap();
        for (g, d) in lows.iter().zip(&data) {
            assert_eq!(*g, d.lo, "{d:?}");
        }
        let ones = Array::<f64>::from_slice_on(&ctx, &vec![1.0e8; 8], 8).unwrap();
        let tiny = Array::<f64>::from_slice_on(&ctx, &vec![1.0; 8], 8).unwrap();
        let s = (&ones + &tiny).to_df32();
        let r = (&s - &ones.to_df32()).to_f32().to_vec().unwrap();
        assert_eq!(r, vec![1.0; 8]);
    }
}

#[test]
fn f64_arrays_compute_in_double_precision() {
    let ctx = Context::new().unwrap();
    let data_a: Vec<f64> = (0..200).map(|i| 1.0 + f64::from(i) * 1.0e-3 + 1.0e-12).collect();
    let data_b: Vec<f64> = (0..200).map(|i| 2.0 - f64::from(i) * 1.0e-4 + 3.0e-13).collect();
    let a = Array::from_slice_on(&ctx, &data_a, 200).unwrap();
    let b = Array::from_slice_on(&ctx, &data_b, 200).unwrap();

    if has_f64(&ctx) {
        let got = (&a * &b + 1.0).sqrt().to_vec().unwrap();
        for ((g, x), y) in got.iter().zip(&data_a).zip(&data_b) {
            let want = (x * y + 1.0).sqrt();
            assert!((g - want).abs() <= 1e-14 * want, "{g} vs {want}");
        }

        // Casts: down to f32, round trip through the other types.
        let down = a.to_f32().to_vec().unwrap();
        for (g, x) in down.iter().zip(&data_a) {
            assert_eq!(*g, *x as f32);
        }
        let df = a.to_df32().to_f64().to_vec().unwrap();
        for (g, x) in df.iter().zip(&data_a) {
            assert!((g - x).abs() <= 1e-13 * x, "{g} vs {x}");
        }
        let widened = Array::from_slice_on(&ctx, &values(8, 2, -1.0, 1.0), 8).unwrap().to_f64().to_vec().unwrap();
        assert_eq!(widened.len(), 8);

        let sum = a.sum().item().unwrap();
        let want: f64 = data_a.iter().sum();
        assert!((sum - want).abs() <= 1e-13 * want, "{sum} vs {want}");
        let (lo, hi) = (a.min().item().unwrap(), a.max().item().unwrap());
        assert_eq!((lo, hi), (data_a[0], data_a[199]));

        // Comparisons, select, a scalar operand in full precision, views.
        let (above, zero) = (Array::<f64>::full_on(&ctx, 200, 1.1).unwrap(), Array::<f64>::zeros_on(&ctx, 200).unwrap());
        let masked = luma::select(&a.gt(&above), &a, &zero).to_vec().unwrap();
        for (g, x) in masked.iter().zip(&data_a) {
            assert_eq!(*g, if *x > 1.1 { *x } else { 0.0 });
        }
        let shifted = (&a + 1.0e-13).slice(0, 10..20).to_vec().unwrap();
        for (i, g) in shifted.iter().enumerate() {
            assert_eq!(*g, data_a[10 + i] + 1.0e-13);
        }
        assert_eq!(a.floor().to_vec().unwrap()[199], 1.0);
        let neg_abs = (-&a).abs().to_vec().unwrap();
        assert_eq!(neg_abs, data_a);
        assert!(a.sign().to_vec().unwrap().iter().all(|&v| v == 1.0));
        assert_eq!(a.ceil().to_vec().unwrap()[0], 2.0);
        let (top, bottom) = (Array::<f64>::full_on(&ctx, 200, 1.5).unwrap(), Array::<f64>::full_on(&ctx, 200, 1.2).unwrap());
        let clamped = a.minimum(&top).maximum(&bottom).to_vec().unwrap();
        for (g, x) in clamped.iter().zip(&data_a) {
            assert_eq!(*g, x.clamp(1.2, 1.5));
        }
        let w = Array::<f64>::zeros_on(&ctx, [4, 2]).unwrap();
        w.slice(1, 1..2).assign(Array::from_slice_on(&ctx, &data_a[..4], [4, 1]).unwrap()).unwrap();
        assert_eq!(&w.to_vec().unwrap()[4..], &data_a[..4]);
        let filled = Array::<f64>::full_on(&ctx, 3, 0.1).unwrap().to_vec().unwrap();
        assert_eq!(filled, vec![0.1; 3]);

        // Operations without an f64 form are refused when they are recorded.
        let err = Array::<f64>::range_on(&ctx, 4, 0).expect_err("generators are f32/u32 only");
        assert!(matches!(err.root(), LumaError::Unsupported(_)), "{err}");
    } else {
        // Without 64-bit shader floats every f64 operation is refused.
        let sum = &a + &b;
        let err = sum.to_vec().unwrap_err();
        assert!(matches!(err.root(), LumaError::Unsupported(_)), "{err}");
        let err = a.to_df32().to_vec().unwrap_err();
        assert!(matches!(err.root(), LumaError::Unsupported(_)), "{err}");
        let err = Array::<f32>::zeros_on(&ctx, 4).unwrap().to_f64().to_vec().unwrap_err();
        assert!(matches!(err.root(), LumaError::Unsupported(_)), "{err}");
    }
}

#[test]
fn arrays_split_over_buffers_keep_one_element_size_per_program() {
    use luma::ContextOptions;
    let ctx = Context::with_options(ContextOptions::default().chunk_bytes(1024)).unwrap();
    let data = values(1000, 4, 0.0, 1.0);
    let x = Array::from_slice_on(&ctx, &data, 1000).unwrap();

    // The reduction folds the f32 array in its own layout, whatever the value type.
    let want: f64 = data.iter().map(|&v| f64::from(v)).sum();
    let got = x.to_df32().sum().item().unwrap().to_f64();
    assert!((got - want).abs() <= 1e-9 * want, "{got} vs {want}");
    let compensated = x.sum_compensated().item().unwrap();
    assert!((compensated - want as f32).abs() <= ulp(want as f32));

    // A store from 4 to 8 byte elements would need two segmentations at once.
    let err = x.to_df32().to_vec().unwrap_err();
    assert!(matches!(err.root(), LumaError::Chunked(_)), "{err}");

    // Arrays of 8-byte elements alone are split by their own size.
    let d: Vec<Df32> = data.iter().map(|&v| Df32::from(v)).collect();
    let y = Array::from_slice_on(&ctx, &d, 1000).unwrap();
    let doubled = (&y + &y).to_vec().unwrap();
    assert!(doubled.iter().zip(&data).all(|(g, v)| g.to_f64() == 2.0 * f64::from(*v)));
    assert_eq!(y.slice(0, 100..900).to_vec().unwrap(), d[100..900].to_vec());
    assert_eq!(y.shift_clamp(-300).to_vec().unwrap()[0], d[300]);
}
