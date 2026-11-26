//! The host cost of making matmuls: 10 000 products of 32x32 operands built and dropped unevaluated, so the
//! case times nothing but what a lazy matmul costs to create and to release (`tools/hosttime` gate of W2-LAZY).

use super::common::*;
use crate::harness::*;

const D: usize = 32;
const PRODUCTS: usize = 10_000;

pub fn run(h: &mut Harness) {
    h.guarded("matmul_build", |h| {
        let ctx = h.ctx.clone();
        let a = A::from_slice_on(&ctx, &values(D * D, 7, 0.5, 1.5), [D, D]).unwrap();
        let w = A::from_slice_on(&ctx, &values(D * D, 6, 0.0, 2.0 / D as f32), [D, D]).unwrap();
        h.build_case("matmul_build", &format!("{D}x{D} x{PRODUCTS}"), per_sec(PRODUCTS, "matmul/s"), &mut || {
            let products: Vec<A> = (0..PRODUCTS).map(|_| a.matmul(&w)).collect();
            std::hint::black_box(&products);
        });
    });
}
