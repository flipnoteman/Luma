//! The host cost of forcing reductions: 1000 distinct `x.sum()` values built and forced one by one, which is where the strategy
//! registry selects (`tools/hosttime` gate of W2-U2; `reduce_build` covers making them).

use super::common::*;
use crate::harness::*;

const N: usize = 256;
const SUMS: usize = 1000;

pub fn run(h: &mut Harness) {
    h.guarded("reduce_force", |h| {
        let ctx = h.ctx.clone();
        let xh = values(N, 5, 0.5, 1.5);
        let x = A::from_slice_on(&ctx, &xh, N).unwrap();
        // Every sum reads its own scaled copy of the data, so none merges with another by structure.
        let sums = |x: &A| -> Vec<A> { (0..SUMS).map(|i| (x * (1.0 + i as f32 * 1e-3)).sum()).collect() };
        let got = sums(&x)[SUMS - 1].to_vec().unwrap()[0];
        let want = xh.iter().map(|v| f64::from(*v) * (1.0 + (SUMS - 1) as f64 * 1e-3)).sum::<f64>() as f32;
        check("reduce_force", &[got], &[want], 1e-4, 1e-3);
        h.case("reduce_force", &format!("n={N} x{SUMS}"), per_sec(SUMS, "sum/s"), &mut || {
            for sum in &sums(&x) {
                sum.eval().unwrap();
            }
            ctx.sync().unwrap();
        });
    });
}
