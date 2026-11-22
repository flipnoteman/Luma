//! The elementwise functions of ArrayFire's `arith.h` that luma added late (`asin` .. `tgamma`,
//! `pow`, `modulo`, `root`, `clamp` with array bounds, `iszero`) against their ArrayFire
//! counterparts, on 16M `f32` elements each; and a fused chain of them.

use arrayfire as af;

use crate::afx;
use crate::bench::Bench;
use crate::cases::{human, L};
use crate::lumax;
use crate::util::*;

const N: usize = 16_000_000;

type LumaOp = fn(&L) -> L;
type AfOp = fn(&af::Array<f32>) -> af::Array<f32>;

fn unary(b: &mut Bench, name: &str, lo: f32, hi: f32, rel: f64, abs: f64, l: LumaOp, a: AfOp) {
    let data = uniform(N, 11, lo, hi);
    let (xl, xa) = (lumax::arr(&b.ctx, &data, N), afx::arr(&data, &[N as u64]));
    let v = compare(&lumax::host(&l(&xl)), &afx::host(&a(&xa)), rel, abs);
    let ctx = b.ctx.clone();
    b.run(
        &format!("7 math {name}"),
        &human(N),
        v,
        "",
        &mut || {
            l(&xl).eval().unwrap();
            ctx.sync().unwrap();
        },
        &mut || {
            a(&xa).eval();
            afx::sync();
        },
    );
}

pub fn math(b: &mut Bench) {
    b.guarded("math", |b| {
        unary(b, "asin", -1.0, 1.0, 1e-5, 1e-6, |x| x.asin(), |x| af::asin(x));
        unary(b, "acos", -1.0, 1.0, 1e-5, 1e-6, |x| x.acos(), |x| af::acos(x));
        unary(b, "atan", -5.0, 5.0, 1e-5, 1e-6, |x| x.atan(), |x| af::atan(x));
        unary(b, "sinh", -5.0, 5.0, 1e-5, 1e-6, |x| x.sinh(), |x| af::sinh(x));
        unary(b, "cosh", -5.0, 5.0, 1e-5, 1e-6, |x| x.cosh(), |x| af::cosh(x));
        unary(b, "asinh", -5.0, 5.0, 1e-5, 1e-6, |x| x.asinh(), |x| af::asinh(x));
        unary(b, "acosh", 1.0, 10.0, 1e-5, 1e-6, |x| x.acosh(), |x| af::acosh(x));
        unary(b, "atanh", -0.99, 0.99, 1e-5, 1e-6, |x| x.atanh(), |x| af::atanh(x));
        unary(b, "erf", -4.0, 4.0, 1e-5, 1e-6, |x| x.erf(), |x| af::erf(x));
        unary(b, "erfc", -4.0, 4.0, 1e-5, 1e-6, |x| x.erfc(), |x| af::erfc(x));
        unary(b, "sigmoid", -8.0, 8.0, 1e-5, 1e-6, |x| x.sigmoid(), |x| af::sigmoid(x));
        unary(b, "log2", 0.01, 100.0, 1e-5, 1e-5, |x| x.log2(), |x| af::log2(x));
        unary(b, "log10", 0.01, 100.0, 1e-5, 1e-5, |x| x.log10(), |x| af::log10(x));
        unary(b, "rsqrt", 0.01, 100.0, 1e-5, 0.0, |x| x.rsqrt(), |x| af::rsqrt(x));
        unary(b, "cbrt", -100.0, 100.0, 1e-5, 0.0, |x| x.cbrt(), |x| af::cbrt(x));
        unary(b, "lgamma", 0.1, 30.0, 1e-5, 1e-5, |x| x.lgamma(), |x| af::lgamma(x));
        unary(b, "tgamma", 0.1, 30.0, 1e-4, 0.0, |x| x.tgamma(), |x| af::tgamma(x));
        unary(b, "pow2", -10.0, 10.0, 1e-5, 0.0, |x| x.pow2(), |x| af::pow2(x));
        unary(b, "fused chain asinh(erf(x)) * 2 + sigmoid(x)", -3.0, 3.0, 1e-4, 1e-5, |x| x.erf().asinh() * 2.0f32 + x.sigmoid(), |x| {
            af::add(&af::mul(&af::asinh(&af::erf(x)), &2.0f32, false), &af::sigmoid(x), false)
        });

        let (ctx, n) = (b.ctx.clone(), N);
        let base = uniform(n, 12, 0.1, 4.0);
        let expo = uniform(n, 13, -2.0, 3.0);
        let (bl, ba) = (lumax::arr(&ctx, &base, n), afx::arr(&base, &[n as u64]));
        let (el, ea) = (lumax::arr(&ctx, &expo, n), afx::arr(&expo, &[n as u64]));
        let v = compare(&lumax::host(&bl.pow(&el)), &afx::host(&af::pow(&ba, &ea, false)), 1e-4, 0.0);
        b.run(
            "7 math pow(array, array)",
            &human(n),
            v,
            "",
            &mut || {
                bl.pow(&el).eval().unwrap();
                ctx.sync().unwrap();
            },
            &mut || {
                af::pow(&ba, &ea, false).eval();
                afx::sync();
            },
        );
        let v = compare(&lumax::host(&bl.pow(2.5f32)), &afx::host(&af::pow(&ba, &2.5f32, false)), 1e-4, 0.0);
        b.run(
            "7 math pow(array, scalar 2.5)",
            &human(n),
            v,
            "",
            &mut || {
                bl.pow(2.5f32).eval().unwrap();
                ctx.sync().unwrap();
            },
            &mut || {
                af::pow(&ba, &2.5f32, false).eval();
                afx::sync();
            },
        );
        let three = af::constant(3.0f32, af::Dim4::new(&[n as u64, 1, 1, 1]));
        let v = compare(&lumax::host(&bl.root(3.0f32)), &afx::host(&af::root(&three, &ba, false)), 1e-4, 0.0);
        b.run(
            "7 math root(3, x)",
            &human(n),
            v,
            "AF takes an array of exponents (constant 3)",
            &mut || {
                bl.root(3.0f32).eval().unwrap();
                ctx.sync().unwrap();
            },
            &mut || {
                af::root(&three, &ba, false).eval();
                afx::sync();
            },
        );
        // Positive operands: both sides agree on `mod` there (sign conventions differ for negatives).
        let divisor = uniform(n, 14, 0.5, 3.0);
        let (dl, da) = (lumax::arr(&ctx, &divisor, n), afx::arr(&divisor, &[n as u64]));
        let v = compare(&lumax::host(&bl.modulo(&dl)), &afx::host(&af::modulo(&ba, &da, false)), 1e-4, 1e-5);
        b.run(
            "7 math mod(array, array)",
            &human(n),
            v,
            "positive operands",
            &mut || {
                bl.modulo(&dl).eval().unwrap();
                ctx.sync().unwrap();
            },
            &mut || {
                af::modulo(&ba, &da, false).eval();
                afx::sync();
            },
        );
        let lo = uniform(n, 15, 0.5, 1.5);
        let hi = uniform(n, 16, 2.0, 3.0);
        let (lol, loa) = (lumax::arr(&ctx, &lo, n), afx::arr(&lo, &[n as u64]));
        let (hil, hia) = (lumax::arr(&ctx, &hi, n), afx::arr(&hi, &[n as u64]));
        let v = exact(&lumax::host(&bl.clamp(&lol, &hil)), &afx::host(&af::clamp(&ba, &loa, &hia, false)));
        b.run(
            "7 math clamp(x, lo array, hi array)",
            &human(n),
            v,
            "",
            &mut || {
                bl.clamp(&lol, &hil).eval().unwrap();
                ctx.sync().unwrap();
            },
            &mut || {
                af::clamp(&ba, &loa, &hia, false).eval();
                afx::sync();
            },
        );
        let zeros: Vec<f32> = base.iter().enumerate().map(|(i, &v)| if i % 3 == 0 { 0.0 } else { v }).collect();
        let (zl, za) = (lumax::arr(&ctx, &zeros, n), afx::arr(&zeros, &[n as u64]));
        let lz: Vec<f32> = lumax::host(&zl.is_zero()).iter().map(|&v| f32::from(u8::from(v))).collect();
        let az: Vec<f32> = afx::host(&af::iszero(&za)).iter().map(|&v| f32::from(v)).collect();
        let v = exact(&lz, &az);
        b.run(
            "7 math iszero",
            &human(n),
            v,
            "",
            &mut || {
                zl.is_zero().eval().unwrap();
                ctx.sync().unwrap();
            },
            &mut || {
                af::iszero(&za).eval();
                afx::sync();
            },
        );
    });
}
