//! `half/` cases: half-precision float arrays (`f16`) next to the same work on `f32`, which reads
//! and writes twice the bytes: a fused elementwise chain stored to memory, a sum, and the sum of
//! the chain (never stored). The values are exact in both types (small multiples of 1/8), so the
//! results are checked against the host without a tolerance.

use half::f16;

use crate::harness::*;
use luma::*;

use super::case_trim;

/// 2^26 elements on a GPU in a full run, 2^24 on the CPU backend, 2^22 with `--quick`.
fn elems(h: &Harness) -> usize {
    if h.quick {
        1 << 22
    } else if h.ctx.is_cpu() {
        1 << 24
    } else {
        1 << 26
    }
}

pub(super) fn run(h: &mut Harness) {
    let n = elems(h);
    let sz = n.to_string();
    // Multiples of 1/8 in [-8, 8): every value and every intermediate of the chain below is exact
    // in `f16` (11 bits of mantissa) and the sums are exact in `f32` up to 2^24 of them.
    let host: Vec<f32> = (0..n).map(|i| (((i * 2654435761usize) >> 7) % 128) as f32 / 8.0 - 8.0).collect();
    let half: Vec<f16> = host.iter().map(|&v| f16::from_f32(v)).collect();
    let ctx = h.ctx.clone();
    let (xf, xh) = (Array::from_slice_on(&ctx, &host, n).unwrap(), Array::from_slice_on(&ctx, &half, n).unwrap());
    // `((x + 1) * 0.5 - x) * 2`: four operations; the two fractions are exact.
    let chain_f32 = |x: &Array<f32>| (((x + 1.0f32) * 0.5f32 - x) * 2.0f32).abs();
    let chain_f16 = |x: &Array<f16>| (((x + f16::from_f32(1.0)) * f16::from_f32(0.5) - x) * f16::from_f32(2.0)).abs();
    let want: Vec<f32> = host.iter().map(|&v| (((v + 1.0) * 0.5 - v) * 2.0).abs()).collect();
    assert_eq!(chain_f32(&xf).to_vec().unwrap(), want);
    assert_eq!(chain_f16(&xh).to_vec().unwrap().iter().map(|v| v.to_f32()).collect::<Vec<_>>(), want);
    let total: f64 = want.iter().map(|&v| f64::from(v)).sum();
    let sum_of = |got: f32| assert!((f64::from(got) - total).abs() <= 1e-4 * total.abs().max(1.0), "{got} vs {total}");
    sum_of(chain_f32(&xf).sum().item().unwrap());
    // (The sum of 2^26 of them overflows `f16`: the check reads it as `f32`; the timed sums round it to `f16` and
    // overflow to infinity, which costs the same.)
    sum_of(chain_f16(&xh).to_f32().sum().item().unwrap());

    case_trim(h, "half/chain_f32", &sz, Work::Bytes(8.0 * n as f64), &mut || chain_f32(&xf).eval().unwrap());
    case_trim(h, "half/chain_f16", &sz, Work::Bytes(4.0 * n as f64), &mut || chain_f16(&xh).eval().unwrap());
    case_trim(h, "half/sum_f32", &sz, Work::Bytes(4.0 * n as f64), &mut || {
        xf.sum().item().unwrap();
    });
    case_trim(h, "half/sum_f16", &sz, Work::Bytes(2.0 * n as f64), &mut || {
        xh.sum().item().unwrap();
    });
    case_trim(h, "half/chain_sum_f32", &sz, Work::Bytes(4.0 * n as f64), &mut || {
        chain_f32(&xf).sum().item().unwrap();
    });
    case_trim(h, "half/chain_sum_f16", &sz, Work::Bytes(2.0 * n as f64), &mut || {
        chain_f16(&xh).sum().item().unwrap();
    });
}
