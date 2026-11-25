//! Host <-> device latency: `item()` of a reduction in a loop, `to_vec` of small arrays
//! (evaluated and not), small uploads, and bulk transfers.

use super::common::*;
use crate::harness::*;

pub fn run(h: &mut Harness) {
    h.guarded("readback", |h| {
        let ctx = h.ctx.clone();
        let reads = size(h, 50, 50, 100);

        // item() of a reduction, one read after another: build, reduce, read back.
        let n = 1 << 16;
        let xh = values(n, 13, 0.0, 1.0);
        let x = A::from_slice_on(&ctx, &xh, n).unwrap();
        let sum: f64 = to_f64(&xh).iter().sum();
        check_scalar("readback/item_sum", f64::from((&x * 1.5).sum().item().unwrap()), sum * 1.5, 1e-4);
        let mut c = 1.0f32;
        h.case("readback/item_sum", &format!("n={n} x{reads}"), per_sec(reads, "read/s"), &mut || {
            let mut total = 0.0;
            for _ in 0..reads {
                c += 1e-3;
                total += (&x * c).sum().item().unwrap();
            }
            std::hint::black_box(total);
        });

        // The same with a big input: the reduction dominates, not the latency.
        let nb = size(h, 1 << 20, 1 << 20, 1 << 22);
        let xbh = values(nb, 14, 0.0, 1.0);
        let xb = A::from_slice_on(&ctx, &xbh, nb).unwrap();
        let want = xbh.iter().copied().fold(f32::MIN, f32::max) * 1.0001 + 0.5;
        check_scalar("readback/item_max_big", f64::from((&xb * 1.0001 + 0.5).max().item().unwrap()), f64::from(want), 1e-5);
        let nr = 20;
        h.case("readback/item_max_big", &format!("n={nb} x{nr}"), Work::Bytes(4.0 * nb as f64 * nr as f64), &mut || {
            let mut total = 0.0;
            for _ in 0..nr {
                c += 1e-3;
                total += (&xb * 1.0001 + c).max().item().unwrap();
            }
            std::hint::black_box(total);
        });

        // to_vec of small evaluated arrays (the readback alone) and of a lazy expression.
        for m in [16usize, 4096] {
            let mh = values(m, 15, 0.0, 1.0);
            let y = A::from_slice_on(&ctx, &mh, m).unwrap();
            force(&y);
            check("readback/to_vec", &y.to_vec().unwrap(), &mh, 0.0, 0.0);
            h.transfer_case(&format!("readback/to_vec_evald_{m}"), &format!("n={m} x{reads}"), per_sec(reads, "read/s"), &mut || {
                for _ in 0..reads {
                    std::hint::black_box(y.to_vec().unwrap());
                }
            });
        }
        let m = 4096;
        let mh = values(m, 16, 0.0, 1.0);
        let y = A::from_slice_on(&ctx, &mh, m).unwrap();
        let want: Vec<f32> = mh.iter().map(|v| v * 2.0 + 1.0).collect();
        check("readback/to_vec_expr", &(&y * 2.0 + 1.0).to_vec().unwrap(), &want, 1e-5, 1e-5);
        h.case("readback/to_vec_expr_4096", &format!("n={m} x{reads}"), per_sec(reads, "read/s"), &mut || {
            for _ in 0..reads {
                c += 1e-3;
                std::hint::black_box((&y * 2.0 + c).to_vec().unwrap());
            }
        });

        // Small uploads: a host array becomes a device array, one after another.
        let up = &xh[..256];
        h.transfer_case("readback/upload_256", &format!("n=256 x{reads}"), per_sec(reads, "upload/s"), &mut || {
            for _ in 0..reads {
                force(&A::from_slice_on(&ctx, up, 256).unwrap());
            }
        });

        // Bulk transfers.
        let nt = size(h, 1 << 22, 1 << 22, 1 << 24);
        let th = values(nt, 17, 0.0, 1.0);
        let t = A::from_slice_on(&ctx, &th, nt).unwrap();
        force(&t);
        check("readback/to_vec_big", &t.to_vec().unwrap(), &th, 0.0, 0.0);
        h.transfer_case("readback/to_vec_big", &format!("n={nt}"), Work::Bytes(4.0 * nt as f64), &mut || {
            std::hint::black_box(t.to_vec().unwrap());
        });
        h.transfer_case("readback/upload_big", &format!("n={nt}"), Work::Bytes(4.0 * nt as f64), &mut || {
            force(&A::from_slice_on(&ctx, &th, nt).unwrap());
        });
    });
}
