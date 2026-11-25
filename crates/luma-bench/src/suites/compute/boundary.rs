//! Host-time micro-cases of the boundary ops (sort, scan, scatter, search, compact, merge): 16 to 1024 elements, so what a call costs on
//! the host (selection included) dominates. `tools/hosttime --case compute/boundary/sort_small ...` counts the instructions of
//! one call; W2-U3a gates them against the code before the registry took over the selection.

use luma::*;

use super::{host, upload};
use crate::harness::*;

pub fn run(h: &mut Harness) {
    h.guarded("boundary", |h| {
        let n = 1024usize;
        let data = values(n, 61, 0.0, 1.0);
        let x = upload(h, &data, n);
        let ints_host = ints(n, 62, 6);
        let keys = Array::<u32>::from_slice_on(&h.ctx, &ints_host, n).unwrap();
        keys.eval().unwrap();
        let tag = format!("n={n}");

        let mut want = data.clone();
        want.sort_by(f32::total_cmp);
        assert_eq!(host(&x.sort()), want, "boundary/sort_small");
        h.case("boundary/sort_small", &tag, Work::Bytes(8.0 * n as f64), &mut || x.sort().eval().unwrap());

        let mut run = 0u32;
        let want: Vec<u32> = ints_host
            .iter()
            .map(|&v| {
                run += v;
                run
            })
            .collect();
        assert_eq!(keys.cumsum(0).to_vec().unwrap(), want, "boundary/cumsum_small");
        h.case("boundary/cumsum_small", &tag, Work::Bytes(8.0 * n as f64), &mut || keys.cumsum(0).eval().unwrap());

        let bins = 16usize;
        let slots = Array::<u32>::from_slice_on(&h.ctx, &ints_host.iter().map(|&v| v % bins as u32).collect::<Vec<_>>(), n).unwrap();
        slots.eval().unwrap();
        let mut want = vec![0u32; bins];
        for &v in &ints_host {
            want[(v % bins as u32) as usize] += 1;
        }
        let ctx = h.ctx.clone();
        let mut counts = Array::<u32>::zeros_on(&ctx, bins).unwrap();
        counts.put_add(&slots, 1u32).unwrap();
        assert_eq!(counts.to_vec().unwrap(), want, "boundary/put_add_small");
        h.case("boundary/put_add_small", &tag, Work::Bytes(8.0 * n as f64), &mut || {
            let mut counts = Array::<u32>::zeros_on(&ctx, bins).unwrap();
            counts.put_add(&slots, 1u32).unwrap();
            counts.eval().unwrap();
        });

        let sorted = Array::<u32>::from_slice_on(&h.ctx, &(0..256u32).map(|i| i * 4).collect::<Vec<_>>(), 256usize).unwrap();
        sorted.eval().unwrap();
        let got = sorted.searchsorted(&keys, Side::Left).to_vec().unwrap();
        assert!(got.iter().zip(&ints_host).all(|(&g, &v)| g == v.div_ceil(4).min(256)), "boundary/searchsorted_small");
        h.case("boundary/searchsorted_small", &tag, Work::Bytes(8.0 * n as f64), &mut || sorted.searchsorted(&keys, Side::Left).eval().unwrap());

        let mask = x.gt(0.5f32);
        let kept = x.filter(&mask).unwrap().map(|kept| host(&kept)).unwrap_or_default();
        assert_eq!(kept, data.iter().copied().filter(|&v| v > 0.5).collect::<Vec<_>>(), "boundary/filter_small");
        h.case("boundary/filter_small", &tag, Work::Bytes(8.0 * n as f64), &mut || {
            let _ = x.filter(&mask).unwrap();
        });

        let a = Array::<u32>::from_slice_on(&h.ctx, &(0..512u32).map(|i| i * 2).collect::<Vec<_>>(), 512usize).unwrap();
        let b = Array::<u32>::from_slice_on(&h.ctx, &(0..512u32).map(|i| i * 3).collect::<Vec<_>>(), 512usize).unwrap();
        let mut want: Vec<u32> = (0..512u32).map(|i| i * 2).chain((0..512u32).map(|i| i * 3)).collect();
        want.sort_unstable();
        want.dedup();
        assert_eq!(a.set_union(&b, true).unwrap().to_vec().unwrap(), want, "boundary/merge_small");
        h.case("boundary/merge_small", &tag, Work::Bytes(8.0 * n as f64), &mut || {
            let _ = a.set_union(&b, true).unwrap();
        });
    });
}
