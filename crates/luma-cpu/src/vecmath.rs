//! Four-lane `f32` transcendental functions called once per SIMD packet from compiled kernels.
//!
//! Cranelift hands a `F32X4` call argument over in `xmm0` and takes the result back from `xmm0`
//! (System V), which is exactly the ABI of an `extern "C" fn(__m128) -> __m128`; the entry points
//! below sit in [`crate::abi::MATH_TABLE`]. They are Cephes-style single-precision routines
//! (range reduction, a minimax polynomial, the exponent assembled in the integer unit) written with
//! SSE2 only (the x86-64 baseline), so one call replaces four calls of the scalar `libm` routines
//! plus the lane extracts and inserts around them.
//!
//! Accuracy against a correctly rounded result (see the tests for the sweeps): `expf`, `exp2f`,
//! `logf`, `log2f`, `tanhf` within 2 ulp over the whole `f32` range; `sinf` and `cosf` within 2 ulp
//! of the result or `2^-24` absolute, whichever is larger (relative error near the zeros of `sin`
//! and `cos` is bounded by the absolute error of the reduced argument), for `|x| <= 8192`; a packet
//! with a larger (or infinite or NaN) lane takes the scalar `libm` routine in every lane. NaN,
//! infinities, signed zeros and denormal results follow IEEE / `libm` behaviour.

// The Cephes coefficients are kept at their published precision.
#![allow(clippy::excessive_precision)]

use std::arch::x86_64::*;

type V = __m128;

#[inline]
#[target_feature(enable = "sse2")]
fn c(x: f32) -> V {
    _mm_set1_ps(x)
}
#[inline]
#[target_feature(enable = "sse2")]
fn ci(x: i32) -> __m128i {
    _mm_set1_epi32(x)
}
#[inline]
#[target_feature(enable = "sse2")]
fn sel(mask: V, a: V, b: V) -> V {
    _mm_or_ps(_mm_and_ps(mask, a), _mm_andnot_ps(mask, b))
}
#[inline]
#[target_feature(enable = "sse2")]
fn abs(x: V) -> V {
    _mm_andnot_ps(c(-0.0), x)
}
#[inline]
#[target_feature(enable = "sse2")]
fn mul(a: V, b: V) -> V {
    _mm_mul_ps(a, b)
}
#[inline]
#[target_feature(enable = "sse2")]
fn add(a: V, b: V) -> V {
    _mm_add_ps(a, b)
}
#[inline]
#[target_feature(enable = "sse2")]
fn sub(a: V, b: V) -> V {
    _mm_sub_ps(a, b)
}
/// `a * b + k` (separate multiply and add: SSE2 has no fused form).
#[inline]
#[target_feature(enable = "sse2")]
fn ma(a: V, b: V, k: V) -> V {
    add(mul(a, b), k)
}

/// `e^r * 2^n` for `r` in about `[-ln2/2, ln2/2]` and `n` in `[-150, 128]`; the scale is applied in
/// two halves so a denormal result is rounded once and an overflow lands on infinity.
#[inline]
#[target_feature(enable = "sse2")]
fn exp_scaled(r: V, n: __m128i) -> V {
    let r2 = mul(r, r);
    let mut p = c(1.987_569_15e-4);
    p = ma(p, r, c(1.398_199_950_7e-3));
    p = ma(p, r, c(8.333_451_907_3e-3));
    p = ma(p, r, c(4.166_579_589_4e-2));
    p = ma(p, r, c(1.666_666_545_9e-1));
    p = ma(p, r, c(5.000_000_120_1e-1));
    let p = add(add(mul(p, r2), r), c(1.0));
    let n1 = _mm_srai_epi32(n, 1);
    let n2 = _mm_sub_epi32(n, n1);
    let s1 = _mm_castsi128_ps(_mm_slli_epi32(_mm_add_epi32(n1, ci(127)), 23));
    let s2 = _mm_castsi128_ps(_mm_slli_epi32(_mm_add_epi32(n2, ci(127)), 23));
    mul(mul(p, s1), s2)
}

#[inline]
#[target_feature(enable = "sse2")]
fn nan_to(x: V, r: V) -> V {
    sel(_mm_cmpunord_ps(x, x), x, r)
}

#[inline]
#[target_feature(enable = "sse2")]
fn expf_v(x: V) -> V {
    // Results below `2^-126` are denormal, and a denormal intermediate costs a microcode assist per
    // operation; a masked-out `-inf` (attention scores) must stay cheap. Lanes under `-104` are
    // exactly zero; a packet with a lane in the denormal band takes the scalar routine.
    let low = _mm_cmplt_ps(x, c(-87.33));
    if _mm_movemask_ps(_mm_and_ps(low, _mm_cmpge_ps(x, c(-104.0)))) != 0 {
        return lanewise(x, f32::exp);
    }
    let xc = _mm_min_ps(_mm_max_ps(x, c(-87.33)), c(89.0));
    let n = _mm_cvtps_epi32(mul(xc, c(std::f32::consts::LOG2_E)));
    let nf = _mm_cvtepi32_ps(n);
    let r = sub(
        sub(xc, mul(nf, c(0.693_359_375))),
        mul(nf, c(-2.121_944_4e-4)),
    );
    nan_to(x, _mm_andnot_ps(low, exp_scaled(r, n)))
}

#[inline]
#[target_feature(enable = "sse2")]
fn exp2f_v(x: V) -> V {
    // As `expf_v`: lanes under `-150` are zero, those in `[-150, -126)` denormal.
    let low = _mm_cmplt_ps(x, c(-126.0));
    if _mm_movemask_ps(_mm_and_ps(low, _mm_cmpge_ps(x, c(-150.0)))) != 0 {
        return lanewise(x, f32::exp2);
    }
    let xc = _mm_min_ps(_mm_max_ps(x, c(-126.0)), c(129.0));
    let n = _mm_cvtps_epi32(xc);
    let f = sub(xc, _mm_cvtepi32_ps(n));
    // e^(f ln2) with the two-part ln2 so the product keeps a few bits beyond f32.
    let r = ma(f, c(-1.904_654_3e-9), mul(f, c(std::f32::consts::LN_2)));
    nan_to(x, _mm_andnot_ps(low, exp_scaled(r, n)))
}

/// Splits a positive `x` into `(e, m)` with `x = (1 + m) 2^e`, `m` in `[sqrt(.5) - 1, sqrt(2) - 1)`,
/// and returns `(e as f32, ln(1 + m))`; denormal inputs are scaled into the normal range first.
#[inline]
#[target_feature(enable = "sse2")]
fn log_parts(x: V) -> (V, V) {
    let den = _mm_cmplt_ps(x, c(f32::MIN_POSITIVE));
    let x = sel(den, mul(x, c(8_388_608.0)), x);
    let eadj = _mm_and_si128(_mm_castps_si128(den), ci(-23));
    let bits = _mm_castps_si128(x);
    let mut e = _mm_add_epi32(_mm_sub_epi32(_mm_srli_epi32(bits, 23), ci(126)), eadj);
    let m = _mm_castsi128_ps(_mm_or_si128(
        _mm_and_si128(bits, ci(0x007f_ffff)),
        ci(0x3f00_0000),
    ));
    let lt = _mm_cmplt_ps(m, c(std::f32::consts::FRAC_1_SQRT_2));
    // m - 1 for m >= sqrt(.5); 2m - 1 below, with the exponent one lower.
    let xr = add(sub(m, c(1.0)), _mm_and_ps(lt, m));
    e = _mm_add_epi32(e, _mm_castps_si128(lt)); // lt lanes are -1
    let ef = _mm_cvtepi32_ps(e);
    let z = mul(xr, xr);
    let mut y = c(7.037_683_629_2e-2);
    y = ma(y, xr, c(-1.151_461_031_0e-1));
    y = ma(y, xr, c(1.167_699_874_0e-1));
    y = ma(y, xr, c(-1.242_014_084_6e-1));
    y = ma(y, xr, c(1.424_932_278_7e-1));
    y = ma(y, xr, c(-1.666_805_766_5e-1));
    y = ma(y, xr, c(2.000_071_476_5e-1));
    y = ma(y, xr, c(-2.499_999_399_3e-1));
    y = ma(y, xr, c(3.333_333_117_4e-1));
    y = mul(mul(y, xr), z);
    y = sub(y, mul(c(0.5), z));
    (ef, add(xr, y))
}

/// Lanes that are not finite positive numbers: zero gives `-inf`, negative and NaN give NaN and
/// `+inf` stays.
#[inline]
#[target_feature(enable = "sse2")]
fn log_special(x: V, r: V) -> V {
    let r = sel(_mm_cmpeq_ps(x, c(f32::INFINITY)), x, r);
    let r = sel(_mm_cmpeq_ps(x, c(0.0)), c(f32::NEG_INFINITY), r);
    sel(_mm_cmpnge_ps(x, c(0.0)), c(f32::NAN), r)
}

#[inline]
#[target_feature(enable = "sse2")]
fn logf_v(x: V) -> V {
    let (e, l) = log_parts(x);
    // ln2 as an exactly multipliable high part plus a low correction, like the polynomial's own sum.
    let r = add(add(l, mul(e, c(-2.121_944_4e-4))), mul(e, c(0.693_359_375)));
    log_special(x, r)
}

#[inline]
#[target_feature(enable = "sse2")]
fn log2f_v(x: V) -> V {
    let (e, l) = log_parts(x);
    // log2(e) = 1 + 0.4426950409: the `l` term stays exact for m close to 1.
    let r = add(add(l, mul(l, c(0.442_695_04))), e);
    log_special(x, r)
}

/// Cephes sine / cosine kernels on the reduced argument `r` (`z = r^2`).
#[inline]
#[target_feature(enable = "sse2")]
fn sin_poly(r: V, z: V) -> V {
    let mut y = c(-1.951_529_589_1e-4);
    y = ma(y, z, c(8.332_160_873_6e-3));
    y = ma(y, z, c(-1.666_665_461_1e-1));
    ma(mul(y, z), r, r)
}
#[inline]
#[target_feature(enable = "sse2")]
fn cos_poly(z: V) -> V {
    let mut y = c(2.443_315_711_809_948e-5);
    y = ma(y, z, c(-1.388_731_625_493_765e-3));
    y = ma(y, z, c(4.166_664_568_298_827e-2));
    add(sub(mul(mul(y, z), z), mul(c(0.5), z)), c(1.0))
}

/// Largest `|x|` the single-precision reduction handles accurately.
const TRIG_LIMIT: f32 = 8192.0;

/// Reduces `|x|` by multiples of pi/4 (Cody-Waite, three parts) and returns `(j, r)` with `j`
/// even, `|x| = r + j pi/4`.
#[inline]
#[target_feature(enable = "sse2")]
fn trig_reduce(ax: V) -> (__m128i, V) {
    let j = _mm_cvttps_epi32(mul(ax, c(std::f32::consts::FRAC_2_PI * 2.0)));
    let j = _mm_and_si128(_mm_add_epi32(j, ci(1)), ci(!1));
    let y = _mm_cvtepi32_ps(j);
    let r = sub(
        sub(
            sub(ax, mul(y, c(0.785_156_25))),
            mul(y, c(2.418_756_484_985_351_6e-4)),
        ),
        mul(y, c(3.774_894_977_445_941e-8)),
    );
    (j, r)
}

#[inline]
#[target_feature(enable = "sse2")]
fn trig_in_range(ax: V) -> bool {
    // `<=` is false for NaN and for infinity.
    _mm_movemask_ps(_mm_cmple_ps(ax, c(TRIG_LIMIT))) == 0xf
}

#[inline]
#[target_feature(enable = "sse2")]
fn lanewise(x: V, f: fn(f32) -> f32) -> V {
    let mut a = [0f32; 4];
    // SAFETY: `a` is 16 bytes; unaligned store.
    unsafe { _mm_storeu_ps(a.as_mut_ptr(), x) };
    let o = [f(a[0]), f(a[1]), f(a[2]), f(a[3])];
    // SAFETY: `o` is 16 bytes; unaligned load.
    unsafe { _mm_loadu_ps(o.as_ptr()) }
}

#[inline]
#[target_feature(enable = "sse2")]
fn bit2_to_sign(j: __m128i) -> V {
    // Moves bit 2 (value 4) of `j` to the sign position.
    _mm_castsi128_ps(_mm_slli_epi32(_mm_and_si128(j, ci(4)), 29))
}

#[inline]
#[target_feature(enable = "sse2")]
fn sinf_v(x: V) -> V {
    let ax = abs(x);
    if !trig_in_range(ax) {
        return lanewise(x, f32::sin);
    }
    let (j, r) = trig_reduce(ax);
    let z = mul(r, r);
    let use_cos = _mm_castsi128_ps(_mm_cmpeq_epi32(_mm_and_si128(j, ci(2)), ci(2)));
    let p = sel(use_cos, cos_poly(z), sin_poly(r, z));
    let sign = _mm_xor_ps(_mm_and_ps(x, c(-0.0)), bit2_to_sign(j));
    // `x == 0` keeps its sign through `r` (`-0 - 0 = -0`, polynomial `r + r*(..)`).
    _mm_xor_ps(p, sign)
}

#[inline]
#[target_feature(enable = "sse2")]
fn cosf_v(x: V) -> V {
    let ax = abs(x);
    if !trig_in_range(ax) {
        return lanewise(x, f32::cos);
    }
    let (j, r) = trig_reduce(ax);
    let z = mul(r, r);
    let use_sin = _mm_castsi128_ps(_mm_cmpeq_epi32(_mm_and_si128(j, ci(2)), ci(2)));
    let p = sel(use_sin, sin_poly(r, z), cos_poly(z));
    _mm_xor_ps(p, bit2_to_sign(_mm_add_epi32(j, ci(2))))
}

#[inline]
#[target_feature(enable = "sse2")]
fn tanhf_v(x: V) -> V {
    let z = abs(x);
    let big = _mm_cmpge_ps(z, c(0.625)); // false for NaN, which the small branch propagates
    let x2 = mul(x, x);
    let mut p = c(-5.704_988_727_45e-3);
    p = ma(p, x2, c(2.063_908_879_54e-2));
    p = ma(p, x2, c(-5.373_971_555_31e-2));
    p = ma(p, x2, c(1.333_144_220_36e-1));
    p = ma(p, x2, c(-3.333_328_194_22e-1));
    // The sign of `x` is the sign of the result, which also keeps `-0.0`.
    let small = _mm_or_ps(ma(mul(p, x2), x, x), _mm_and_ps(x, c(-0.0)));
    // 1 - 2 / (e^(2|x|) + 1); e^(2|x|) saturates to infinity, giving exactly 1.
    let e = expf_v(add(z, z));
    let t = sub(c(1.0), _mm_div_ps(c(2.0), add(e, c(1.0))));
    let large = _mm_or_ps(t, _mm_and_ps(x, c(-0.0)));
    sel(big, large, small)
}

macro_rules! entry {
    ($($name:ident => $f:ident),* $(,)?) => {$(
        pub extern "C" fn $name(x: __m128) -> __m128 {
            // SAFETY: SSE2 is part of the x86-64 baseline.
            unsafe { $f(x) }
        }
    )*};
}

entry! {
    sinf4 => sinf_v,
    cosf4 => cosf_v,
    tanhf4 => tanhf_v,
    expf4 => expf_v,
    exp2f4 => exp2f_v,
    logf4 => logf_v,
    log2f4 => log2f_v,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(f: extern "C" fn(__m128) -> __m128, x: [f32; 4]) -> [f32; 4] {
        let mut o = [0f32; 4];
        // SAFETY: 16-byte buffers, unaligned load/store.
        unsafe { _mm_storeu_ps(o.as_mut_ptr(), f(_mm_loadu_ps(x.as_ptr()))) };
        o
    }

    /// Distance in ulps of `got` from the exact `want` (`f64`), with `floor` as the smallest
    /// absolute error that counts (0 for none). Zero and sign are checked separately.
    fn err_ulps(got: f32, want: f64) -> f64 {
        let w32 = want as f32;
        if w32.is_nan() || got.is_nan() {
            return if w32.is_nan() && got.is_nan() {
                0.0
            } else {
                f64::INFINITY
            };
        }
        if w32.is_infinite() || got.is_infinite() {
            // Exact overflow boundary: allow the 1 ulp the rounded `want` hides.
            return if got == w32 || want.abs() > f32::MAX as f64 * 0.999999 {
                0.0
            } else {
                f64::INFINITY
            };
        }
        let ulp = {
            let a = w32.abs().max(f32::MIN_POSITIVE);
            (f32::from_bits(a.to_bits() + 1) - a) as f64
        };
        ((got as f64) - want).abs() / ulp
    }

    struct Sweep {
        worst: f64,
        at: f32,
    }

    fn sweep(
        f: extern "C" fn(__m128) -> __m128,
        reference: fn(f64) -> f64,
        xs: impl Iterator<Item = f32>,
        abs_floor: f64,
    ) -> Sweep {
        let mut s = Sweep {
            worst: 0.0,
            at: 0.0,
        };
        let xs: Vec<f32> = xs.collect();
        for ch in xs.chunks(4) {
            let mut v = [ch[0]; 4];
            v[..ch.len()].copy_from_slice(ch);
            let o = run(f, v);
            for (&x, &g) in v.iter().zip(&o) {
                let want = reference(x as f64);
                let mut e = err_ulps(g, want);
                if abs_floor > 0.0 && ((g as f64) - want).abs() <= abs_floor {
                    e = e.min(0.0);
                }
                if e > s.worst {
                    s = Sweep { worst: e, at: x };
                }
            }
        }
        s
    }

    fn linear(lo: f32, hi: f32, n: usize) -> impl Iterator<Item = f32> {
        (0..n).map(move |i| lo + (hi - lo) * (i as f32 / (n - 1) as f32))
    }

    /// `count` bit patterns evenly spread over the positive values `[lo, hi]` (log-like spacing).
    fn bits(lo: f32, hi: f32, count: usize) -> impl Iterator<Item = f32> {
        let (a, b) = (lo.to_bits() as u64, hi.to_bits() as u64);
        (0..count)
            .map(move |i| f32::from_bits((a + (b - a) * i as u64 / (count as u64 - 1)) as u32))
    }

    /// [`bits`] over `[lo, hi]` and `[-hi, -lo]`.
    fn sym(lo: f32, hi: f32, count: usize) -> impl Iterator<Item = f32> {
        bits(lo, hi, count / 2).flat_map(|x| [x, -x])
    }

    fn report(name: &str, s: &Sweep, bound: f64) {
        eprintln!("{name}: worst {:.3} ulp at {:e}", s.worst, s.at);
        assert!(
            s.worst <= bound,
            "{name}: {} ulp at {:e} (bound {bound})",
            s.worst,
            s.at
        );
    }

    /// Passes through `sel`-style special values that must come out bit-exact or as NaN.
    fn specials(f: extern "C" fn(__m128) -> __m128, reference: fn(f64) -> f64, name: &str) {
        let xs = [
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            0.0,
            -0.0,
            f32::MIN_POSITIVE,
            -f32::MIN_POSITIVE,
            1.0e-45,
            -1.0e-45,
            1.0e-40,
            f32::MAX,
            f32::MIN,
            1.0,
            -1.0,
            0.5,
            -0.5,
        ];
        for ch in xs.chunks(4) {
            let o = run(f, [ch[0], ch[1], ch[2], ch[3]]);
            for (&x, &g) in ch.iter().zip(&o) {
                let want = reference(x as f64);
                let e = err_ulps(g, want);
                assert!(e <= 2.0, "{name}({x:e}) = {g:e}, want {want:e} ({e} ulp)");
                if want == 0.0 || (want as f32) == 0.0 {
                    assert_eq!(
                        g.is_sign_negative(),
                        (want as f32).is_sign_negative(),
                        "{name}({x:e}) sign of zero"
                    );
                }
            }
        }
    }

    #[test]
    fn expf_accuracy() {
        let f = expf4;
        let r = |x: f64| x.exp();
        specials(f, r, "expf");
        report(
            "exp dense",
            &sweep(f, r, linear(-100.0, 90.0, 400_001), 0.0),
            2.0,
        );
        report(
            "exp small",
            &sweep(f, r, linear(-2.0, 2.0, 200_001), 0.0),
            2.0,
        );
        report(
            "exp bits",
            &sweep(f, r, sym(1e-30, 89.0, 400_000), 0.0),
            2.0,
        );
        report("exp tiny", &sweep(f, r, sym(1e-30, 1e-6, 50_000), 0.0), 2.0);
        report(
            "exp huge",
            &sweep(
                f,
                r,
                [1e4f32, -1e4, 88.8, -88.8, 105.0, -105.0, 1e30].into_iter(),
                0.0,
            ),
            2.0,
        );
    }

    #[test]
    fn exp2f_accuracy() {
        let f = exp2f4;
        let r = |x: f64| x.exp2();
        specials(f, r, "exp2f");
        report(
            "exp2 dense",
            &sweep(f, r, linear(-150.0, 130.0, 400_001), 0.0),
            2.0,
        );
        report(
            "exp2 small",
            &sweep(f, r, linear(-2.0, 2.0, 200_001), 0.0),
            2.0,
        );
        report(
            "exp2 ints",
            &sweep(f, r, (-150..=128).map(|i| i as f32), 0.0),
            0.5,
        );
    }

    #[test]
    fn logf_accuracy() {
        let f = logf4;
        let r = |x: f64| x.ln();
        specials(f, r, "logf");
        report(
            "log bits",
            &sweep(f, r, bits(1e-45, f32::MAX, 800_000), 0.0),
            2.0,
        );
        report(
            "log near 1",
            &sweep(f, r, linear(0.5, 2.0, 600_001), 0.0),
            2.0,
        );
        report(
            "log denormal",
            &sweep(f, r, bits(1e-45, 1.1e-38, 100_000), 0.0),
            2.0,
        );
        report(
            "log tiny",
            &sweep(f, r, bits(0.99999, 1.00001, 100_000), 0.0),
            2.0,
        );
        assert!(
            run(f, [-1.0, -2.0, -1e-30, -0.0])[..3]
                .iter()
                .all(|v| v.is_nan())
        );
    }

    #[test]
    fn log2f_accuracy() {
        let f = log2f4;
        let r = |x: f64| x.log2();
        specials(f, r, "log2f");
        report(
            "log2 bits",
            &sweep(f, r, bits(1e-45, f32::MAX, 800_000), 0.0),
            2.0,
        );
        report(
            "log2 near 1",
            &sweep(f, r, linear(0.5, 2.0, 600_001), 0.0),
            2.0,
        );
        report(
            "log2 pow2",
            &sweep(f, r, (-149..=127).map(|i| (i as f64).exp2() as f32), 0.0),
            0.0,
        );
    }

    #[test]
    fn sincos_accuracy() {
        // Absolute floor 2^-24: the reduced argument carries that much absolute error.
        let floor = 6.0e-8;
        for (name, f, r) in [
            (
                "sinf",
                sinf4 as extern "C" fn(__m128) -> __m128,
                (|x: f64| x.sin()) as fn(f64) -> f64,
            ),
            ("cosf", cosf4, |x: f64| x.cos()),
        ] {
            specials(f, r, name);
            report(
                &format!("{name} dense"),
                &sweep(f, r, linear(-100.0, 100.0, 800_001), floor),
                2.0,
            );
            report(
                &format!("{name} wide"),
                &sweep(f, r, linear(-8192.0, 8192.0, 800_001), floor),
                2.0,
            );
            report(
                &format!("{name} small"),
                &sweep(f, r, sym(1e-30, 1e-3, 100_000), 0.0),
                2.0,
            );
            // Beyond the reduction's range the packet takes libm.
            report(
                &format!("{name} huge"),
                &sweep(
                    f,
                    r,
                    [9000.0f32, -1e5, 1e10, 3e38, f32::MAX].into_iter(),
                    floor,
                ),
                2.0,
            );
            let o = run(f, [f32::INFINITY, f32::NEG_INFINITY, f32::NAN, 1.0]);
            assert!(
                o[..3].iter().all(|v| v.is_nan()),
                "{name} of inf/nan: {o:?}"
            );
        }
        assert_eq!(
            run(sinf4, [-0.0, 0.0, 1.0, 1.0])[0].to_bits(),
            (-0.0f32).to_bits()
        );
        assert_eq!(run(cosf4, [-0.0, 0.0, 1.0, 1.0])[0], 1.0);
        // A single large lane sends the packet to libm without disturbing the others.
        let o = run(sinf4, [0.5, 1e9, -2.0, 3.0]);
        assert_eq!(
            o,
            [0.5f32.sin(), 1e9f32.sin(), (-2.0f32).sin(), 3.0f32.sin()]
        );
    }

    #[test]
    fn tanhf_accuracy() {
        let f = tanhf4;
        let r = |x: f64| x.tanh();
        specials(f, r, "tanhf");
        report(
            "tanh dense",
            &sweep(f, r, linear(-12.0, 12.0, 800_001), 0.0),
            2.0,
        );
        report(
            "tanh small",
            &sweep(f, r, sym(1e-30, 1.0, 400_000), 0.0),
            2.0,
        );
        report(
            "tanh tiny",
            &sweep(f, r, sym(1e-30, 1e-3, 100_000), 0.0),
            2.0,
        );
        report(
            "tanh sat",
            &sweep(
                f,
                r,
                [9.0f32, 10.0, 20.0, 100.0, -100.0, 1e30, f32::MAX].into_iter(),
                0.0,
            ),
            2.0,
        );
    }
}
