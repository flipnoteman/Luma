//! The host cost of making reductions: 10 000 `x.sum()` values built and dropped unevaluated, so the case
//! times nothing but what a lazy reduction costs to create and to release (`tools/hosttime` gate of W1-U1).

use super::common::*;
use crate::harness::*;

const N: usize = 1024;
const SUMS: usize = 10_000;

pub fn run(h: &mut Harness) {
    h.guarded("reduce_build", |h| {
        let ctx = h.ctx.clone();
        let x = A::from_slice_on(&ctx, &values(N, 5, 0.5, 1.5), N).unwrap();
        h.build_case("reduce_build", &format!("n={N} x{SUMS}"), per_sec(SUMS, "sum/s"), &mut || {
            let sums: Vec<A> = (0..SUMS).map(|_| x.sum()).collect();
            std::hint::black_box(&sums);
        });
    });
}
