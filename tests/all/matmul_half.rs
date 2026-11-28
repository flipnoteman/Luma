//! Products of `f16` and `bf16` arrays on the half-precision matrix units (W3-R7): the unit multiplies exactly and accumulates in
//! `f32`, so every entry is within `4 * 2^-24 * Σ|ab|` of the exact product (the `f32` accumulation) plus the single rounding of the
//! result to the type; a context with `strict()` numerics widens and runs the `f32` kernels instead. Where a device has no unit the
//! tests check the widening path against the same bounds.

use crate::common;

use common::values;
use half::{bf16, f16};
use luma::{Array, Context, ContextOptions, DecisionSite, Numerics};

/// An element type under test.
trait Half: luma::GpuType + Copy + std::fmt::Debug {
    const NAME: &'static str;
    /// The unit's rounding of the output: relative `2^-p` for `p` significant bits.
    const OUT_EPS: f64;
    fn of(x: f32) -> Self;
    fn f64(self) -> f64;
    fn matmul(a: &Array<Self>, b: &Array<Self>, flags: (bool, bool)) -> Array<Self>;
}

macro_rules! half_type {
    ($t:ty, $name:literal, $eps:expr) => {
        impl Half for $t {
            const NAME: &'static str = $name;
            const OUT_EPS: f64 = $eps;
            fn of(x: f32) -> Self {
                <$t>::from_f32(x)
            }
            fn f64(self) -> f64 {
                f64::from(self.to_f32())
            }
            fn matmul(a: &Array<Self>, b: &Array<Self>, flags: (bool, bool)) -> Array<Self> {
                match flags {
                    (false, false) => a.matmul(b),
                    (false, true) => a.matmul_nt(b),
                    (true, false) => a.matmul_tn(b),
                    (true, true) => a.matmul_tt(b),
                }
            }
        }
    };
}
half_type!(f16, "f16", 4.8828125e-4);
half_type!(bf16, "bf16", 3.90625e-3);

struct Case<T: Half> {
    a: Vec<T>,
    b: Vec<T>,
    lhs: Array<T>,
    rhs: Array<T>,
    dims: (usize, usize, usize),
    flags: (bool, bool),
}

/// The operands of an `m x k` by `k x n` product, stored as the flags say.
fn case<T: Half>(ctx: &Context, (m, n, k): (usize, usize, usize), flags: (bool, bool), seed: u64, lo: f32, hi: f32) -> Case<T> {
    let a: Vec<T> = values(m * k, seed, lo, hi).into_iter().map(T::of).collect();
    let b: Vec<T> = values(k * n, seed + 1, lo, hi).into_iter().map(T::of).collect();
    // `a` row-major `m x k` (shape [k, m]), or `k x m` when transposed (shape [m, k]).
    let stored_a: Vec<T> = if flags.0 { (0..k * m).map(|i| a[(i % m) * k + i / m]).collect() } else { a.clone() };
    let stored_b: Vec<T> = if flags.1 { (0..n * k).map(|i| b[(i % k) * n + i / k]).collect() } else { b.clone() };
    let lhs = Array::from_slice_on(ctx, &stored_a, if flags.0 { [m, k] } else { [k, m] }).unwrap();
    let rhs = Array::from_slice_on(ctx, &stored_b, if flags.1 { [k, n] } else { [n, k] }).unwrap();
    Case { a, b, lhs, rhs, dims: (m, n, k), flags }
}

/// Entries of the exact product (`f64`) and of `Σ|ab|` at `(row, col)`.
fn entry<T: Half>(case: &Case<T>, row: usize, col: usize) -> (f64, f64) {
    let (_, n, k) = case.dims;
    let (mut sum, mut abs) = (0.0f64, 0.0f64);
    for q in 0..k {
        let product = case.a[row * k + q].f64() * case.b[q * n + col].f64();
        sum += product;
        abs += product.abs();
    }
    (sum, abs)
}

/// Samples of the output within the unit's bound: `4 * 2^-24 * Σ|ab|` plus one rounding to the type (an `f16` result past 65504 is an
/// infinity).
fn check<T: Half>(case: &Case<T>, got: &[T], samples: usize, what: &str) {
    let (m, n, _) = case.dims;
    assert_eq!(got.len(), m * n, "{what}: length");
    let step = (m * n / samples).max(1);
    for at in (0..m * n).step_by(step).chain([0, m * n - 1]) {
        let (row, col) = (at / n, at % n);
        let (want, abs) = entry(case, row, col);
        let value = got[at].f64();
        if T::NAME == "f16" && want.abs() >= 65520.0 {
            assert!(value.is_infinite() || (value - want).abs() <= 1e-3 * want.abs(), "{what}: ({row}, {col}) overflows: {value} against {want}");
            continue;
        }
        let bound = 4.0 / 16_777_216.0 * abs + T::OUT_EPS * want.abs() * 1.01 + 1e-7;
        assert!((value - want).abs() <= bound, "{what}: ({row}, {col}) of {m}x{n}: got {value}, want {want} (bound {bound:e})");
    }
}

/// Whether the context ran a matrix unit for the last product (its decision is logged).
fn ran_unit(ctx: &Context) -> bool {
    ctx.take_decisions().iter().any(|d| d.site == DecisionSite::Unit && !d.chosen.ends_with("verified"))
}

fn products_within_the_bound<T: Half>() {
    let ctx = Context::new().unwrap();
    ctx.record_decisions(true);
    let mut units = 0;
    // Whole tiles, an edge that is not a multiple of the tile, a row count that is not a multiple of 8 (`Vec4Tails`), a long K
    // (split), and shapes the units do not take (small, odd leading dimensions).
    let shapes = [(128, 128, 128), (256, 256, 256), (1000, 1000, 1000), (1001, 1000, 1024), (512, 64, 4096), (64, 64, 8192), (4096, 64, 512), (16, 16, 16), (100, 99, 98), (130, 130, 96)];
    for dims in shapes {
        for flags in [(false, false), (false, true), (true, false), (true, true)] {
            let case = case::<T>(&ctx, dims, flags, 11 + (dims.0 * 7 + dims.2) as u64, -1.0, 1.0);
            let out = T::matmul(&case.lhs, &case.rhs, case.flags);
            let got = out.to_vec().unwrap();
            check(&case, &got, 400, &format!("{} {dims:?} {flags:?}", T::NAME));
            units += usize::from(ran_unit(&ctx));
            // Reproducible: the second run is bit-equal.
            let again = T::matmul(&case.lhs, &case.rhs, case.flags).to_vec().unwrap();
            assert!(got.iter().zip(&again).all(|(x, y)| x.f64().to_bits() == y.f64().to_bits() || (x.f64().is_nan() && y.f64().is_nan())), "{} {dims:?} {flags:?}: not reproducible", T::NAME);
        }
    }
    println!("{}: {units} of {} products ran a matrix unit", T::NAME, shapes.len() * 4);
}

#[test]
fn f16_products_are_within_the_accumulation_bound() {
    products_within_the_bound::<f16>();
}

#[test]
fn bf16_products_are_within_the_accumulation_bound() {
    products_within_the_bound::<bf16>();
}

#[test]
fn strict_numerics_widen_and_never_run_a_unit() {
    let ctx = Context::with_options(ContextOptions::from_env().numerics(Numerics::strict())).unwrap();
    ctx.record_decisions(true);
    let case = case::<f16>(&ctx, (256, 256, 256), (false, false), 5, -1.0, 1.0);
    let got = f16::matmul(&case.lhs, &case.rhs, case.flags).to_vec().unwrap();
    check(&case, &got, 300, "strict");
    assert!(!ran_unit(&ctx), "strict() runs the f32 kernels");
    // The widening definition: the f32 product rounded once.
    let widened = case.lhs.to_f32().matmul(&case.rhs.to_f32()).to_f16().to_vec().unwrap();
    assert!(got.iter().zip(&widened).all(|(x, y)| x.to_bits() == y.to_bits()));
}

#[test]
fn the_unit_propagates_nan_and_infinity_by_class() {
    let ctx = Context::new().unwrap();
    let (m, n, k) = (64, 64, 64);
    let mut case = case::<f16>(&ctx, (m, n, k), (false, false), 3, 0.5, 1.0);
    case.a[2 * k + 5] = f16::NAN; // row 2
    case.a[3 * k] = f16::INFINITY; // row 3
    case.b[7 * n + 9] = f16::NEG_INFINITY; // column 9 of row 7 of b
    let lhs = Array::from_slice_on(&ctx, &case.a, [k, m]).unwrap();
    let rhs = Array::from_slice_on(&ctx, &case.b, [n, k]).unwrap();
    let c = lhs.matmul(&rhs).to_vec().unwrap();
    assert!(c[2 * n..3 * n].iter().all(|x| x.to_f32().is_nan()), "NaN row");
    assert!(c[3 * n..4 * n].iter().all(|x| x.to_f32().is_infinite() || x.to_f32().is_nan()), "infinite row");
    // Column 9 holds the negative infinity from `b` wherever the row of `a` is positive and finite.
    assert!(c[9] == f16::NEG_INFINITY && c[n + 9] == f16::NEG_INFINITY);
    assert!(c[10].to_f32().is_finite() && c[n + 10].to_f32().is_finite());
}

#[test]
fn an_f16_result_past_its_range_is_an_infinity_and_a_bf16_one_is_not() {
    let ctx = Context::new().unwrap();
    let (m, n, k) = (64, 64, 64);
    let a = vec![60000.0f32; m * k];
    let b = vec![1.0f32; k * n];
    let (a16, b16): (Vec<f16>, Vec<f16>) = (a.iter().map(|&x| f16::from_f32(x)).collect(), b.iter().map(|&x| f16::from_f32(x)).collect());
    let c = Array::from_slice_on(&ctx, &a16, [k, m]).unwrap().matmul(&Array::from_slice_on(&ctx, &b16, [n, k]).unwrap()).to_vec().unwrap();
    assert!(c.iter().all(|&x| x == f16::INFINITY), "64 * 60000 overflows f16");
    let (a16, b16): (Vec<bf16>, Vec<bf16>) = (a.iter().map(|&x| bf16::from_f32(x)).collect(), b.iter().map(|&x| bf16::from_f32(x)).collect());
    let c = Array::from_slice_on(&ctx, &a16, [k, m]).unwrap().matmul(&Array::from_slice_on(&ctx, &b16, [n, k]).unwrap()).to_vec().unwrap();
    assert!(c.iter().all(|x| x.to_f32().is_finite() && (x.to_f32() - 3_840_000.0).abs() < 3_840_000.0 * 4e-3), "bf16 has the range of f32");
}

/// The half-precision conversions of the verification harness are those of the `half` crate (the store's rounding is checked against
/// them in the kernel tests): round to nearest even, canonical NaN, over 2^20 values and the edge set.
#[test]
fn the_harness_conversions_equal_the_half_crate() {
    use luma_core::hal::verify::{bf16_to_f32, f16_to_f32, f32_to_bf16, f32_to_f16};
    let mut values: Vec<f32> = values(1 << 20, 99, -70000.0, 70000.0);
    values.extend(common::values(1 << 18, 5, -1.0, 1.0).into_iter().map(|x| x * 1e-4));
    values.extend([0.0, -0.0, f32::INFINITY, f32::NEG_INFINITY, 65504.0, 65519.9, 65520.0, 6.0e-8, 3.0e-8, 5.96e-8, 1e-40, f32::MAX, f32::MIN_POSITIVE]);
    for x in values {
        assert_eq!(f32_to_f16(x), f16::from_f32(x).to_bits(), "f16 of {x}");
        assert_eq!(f32_to_bf16(x), bf16::from_f32(x).to_bits(), "bf16 of {x}");
    }
    for bits in 0..=u16::MAX {
        let (h, b) = (f16::from_bits(bits), bf16::from_bits(bits));
        assert!(f16_to_f32(bits).to_bits() == h.to_f32().to_bits() || h.is_nan());
        assert!(bf16_to_f32(bits).to_bits() == b.to_f32().to_bits() || b.is_nan());
    }
}

/// The unit's rounding in the store is that of the `half` crate: a product with one non-zero term per entry is exact in `f32` (two
/// 11-bit factors), so its `f16` result is `f16::from_f32` of it, bit for bit, over 2^20 values (whatever the unit and the order of its
/// accumulation).
#[test]
fn the_stores_rounding_is_that_of_the_half_crate() {
    let ctx = Context::new().unwrap();
    let (m, n, k) = (256, 256, 16);
    for seed in 0..16u64 {
        let xs = values(m, 100 + seed, -300.0, 300.0);
        let ys = values(n, 200 + seed, -300.0, 300.0);
        let xs: Vec<f16> = xs.into_iter().map(f16::from_f32).collect();
        let ys: Vec<f16> = ys.into_iter().map(f16::from_f32).collect();
        // `a` is `m x k` with column 0 = xs, `b` is `k x n` with row 0 = ys.
        let mut a = vec![f16::ZERO; m * k];
        let mut b = vec![f16::ZERO; k * n];
        for i in 0..m {
            a[i * k] = xs[i];
        }
        b[..n].copy_from_slice(&ys);
        let c = Array::from_slice_on(&ctx, &a, [k, m]).unwrap().matmul(&Array::from_slice_on(&ctx, &b, [n, k]).unwrap()).to_vec().unwrap();
        for i in 0..m {
            for j in 0..n {
                let want = f16::from_f32(xs[i].to_f32() * ys[j].to_f32());
                assert!(c[i * n + j].to_bits() == want.to_bits() || (c[i * n + j].is_nan() && want.is_nan()), "({i}, {j}): {} * {} = {} want {want}", xs[i], ys[j], c[i * n + j]);
            }
        }
    }
}
