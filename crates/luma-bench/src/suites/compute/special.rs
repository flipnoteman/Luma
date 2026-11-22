//! The elementwise functions of ArrayFire's `arith.h` that were added late: the inverse and
//! hyperbolic functions, `erf`, `lgamma`, `tgamma`, `cbrt`, `log2`, `log10`, `pow` with array
//! operands, `modulo` and `clamp` with array bounds, as memory-bound kernels over 16M elements
//! (and one fused chain), each checked at a sample against `f64` references.

use luma::*;

use super::{check_sampled, host, pick, upload};
use crate::harness::*;

type Op = fn(&Array<f32>) -> Array<f32>;

struct Fun {
    name: &'static str,
    lo: f32,
    hi: f32,
    op: Op,
    host: fn(f64) -> f64,
    rel: f64,
    abs: f64,
}

pub fn run(h: &mut Harness) {
    h.guarded("math", |h| {
        let n: usize = if h.ctx.is_cpu() { pick(h, 1 << 20, 1 << 22) } else { pick(h, 1 << 22, 1 << 24) };
        let funs = [
            Fun { name: "asin", lo: -1.0, hi: 1.0, op: |x| x.asin(), host: f64::asin, rel: 1e-5, abs: 1e-6 },
            Fun { name: "acos", lo: -1.0, hi: 1.0, op: |x| x.acos(), host: f64::acos, rel: 1e-5, abs: 1e-6 },
            Fun { name: "atan", lo: -5.0, hi: 5.0, op: |x| x.atan(), host: f64::atan, rel: 1e-5, abs: 1e-6 },
            Fun { name: "sinh", lo: -5.0, hi: 5.0, op: |x| x.sinh(), host: f64::sinh, rel: 1e-5, abs: 1e-6 },
            Fun { name: "cosh", lo: -5.0, hi: 5.0, op: |x| x.cosh(), host: f64::cosh, rel: 1e-5, abs: 0.0 },
            Fun { name: "asinh", lo: -5.0, hi: 5.0, op: |x| x.asinh(), host: f64::asinh, rel: 1e-5, abs: 1e-6 },
            Fun { name: "acosh", lo: 1.0, hi: 10.0, op: |x| x.acosh(), host: f64::acosh, rel: 1e-5, abs: 1e-6 },
            Fun { name: "atanh", lo: -0.99, hi: 0.99, op: |x| x.atanh(), host: f64::atanh, rel: 1e-5, abs: 1e-6 },
            Fun { name: "erf", lo: -4.0, hi: 4.0, op: |x| x.erf(), host: libm::erf, rel: 1e-5, abs: 1e-6 },
            Fun { name: "erfc", lo: -4.0, hi: 4.0, op: |x| x.erfc(), host: libm::erfc, rel: 1e-5, abs: 0.0 },
            Fun { name: "sigmoid", lo: -8.0, hi: 8.0, op: |x| x.sigmoid(), host: |v| 1.0 / (1.0 + (-v).exp()), rel: 1e-5, abs: 0.0 },
            Fun { name: "log2", lo: 0.01, hi: 100.0, op: |x| x.log2(), host: f64::log2, rel: 1e-5, abs: 1e-5 },
            Fun { name: "log10", lo: 0.01, hi: 100.0, op: |x| x.log10(), host: f64::log10, rel: 1e-5, abs: 1e-5 },
            Fun { name: "rsqrt", lo: 0.01, hi: 100.0, op: |x| x.rsqrt(), host: |v| 1.0 / v.sqrt(), rel: 1e-5, abs: 0.0 },
            Fun { name: "cbrt", lo: -100.0, hi: 100.0, op: |x| x.cbrt(), host: f64::cbrt, rel: 1e-5, abs: 0.0 },
            Fun { name: "lgamma", lo: 0.1, hi: 30.0, op: |x| x.lgamma(), host: libm::lgamma, rel: 1e-5, abs: 1e-5 },
            Fun { name: "tgamma", lo: 0.1, hi: 30.0, op: |x| x.tgamma(), host: libm::tgamma, rel: 1e-4, abs: 0.0 },
            Fun { name: "pow2", lo: -10.0, hi: 10.0, op: |x| x.pow2(), host: f64::exp2, rel: 1e-5, abs: 0.0 },
        ];
        for f in &funs {
            let xs = values(n, 11, f.lo, f.hi);
            let x = upload(h, &xs, n);
            check_sampled(&format!("math/{}", f.name), &host(&(f.op)(&x)), |i| (f.host)(f64::from(xs[i])), 61, f.rel, f.abs);
            h.case(&format!("math/{}", f.name), &format!("{n}"), Work::Bytes(8.0 * n as f64), &mut || (f.op)(&x).eval().unwrap());
        }

        // A chain of them: one kernel.
        let xs = values(n, 12, -3.0, 3.0);
        let x = upload(h, &xs, n);
        let chain = |x: &Array<f32>| x.erf().asinh() * 2.0 + x.sigmoid();
        check_sampled("math/chain", &host(&chain(&x)), |i| 2.0 * libm::erf(f64::from(xs[i])).asinh() + 1.0 / (1.0 + (-f64::from(xs[i])).exp()), 61, 1e-4, 1e-5);
        h.case("math/chain_erf_asinh_sigmoid", &format!("{n}"), Work::Bytes(8.0 * n as f64), &mut || chain(&x).eval().unwrap());

        // Binary operations with array operands.
        let base = values(n, 13, 0.1, 4.0);
        let expo = values(n, 14, -2.0, 3.0);
        let (b, e) = (upload(h, &base, n), upload(h, &expo, n));
        check_sampled("math/pow", &host(&b.pow(&e)), |i| f64::from(base[i]).powf(f64::from(expo[i])), 61, 1e-4, 0.0);
        h.case("math/pow_array", &format!("{n}"), Work::Bytes(12.0 * n as f64), &mut || b.pow(&e).eval().unwrap());
        h.case("math/pow_scalar", &format!("{n}"), Work::Bytes(8.0 * n as f64), &mut || b.pow(2.5).eval().unwrap());
        let divisor = values(n, 15, 0.5, 3.0);
        let d = upload(h, &divisor, n);
        check_sampled("math/modulo", &host(&b.modulo(&d)), |i| f64::from(base[i]) % f64::from(divisor[i]), 61, 1e-4, 1e-5);
        h.case("math/modulo", &format!("{n}"), Work::Bytes(12.0 * n as f64), &mut || b.modulo(&d).eval().unwrap());
        let (lo, hi) = (values(n, 16, 0.5, 1.5), values(n, 17, 2.0, 3.0));
        let (l, u) = (upload(h, &lo, n), upload(h, &hi, n));
        check_sampled("math/clamp", &host(&b.clamp(&l, &u)), |i| f64::from(base[i]).max(f64::from(lo[i])).min(f64::from(hi[i])), 61, 0.0, 0.0);
        h.case("math/clamp_arrays", &format!("{n}"), Work::Bytes(16.0 * n as f64), &mut || b.clamp(&l, &u).eval().unwrap());
    });
}
