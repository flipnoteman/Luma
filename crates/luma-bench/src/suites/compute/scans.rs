//! Prefix scans and axis reductions over long contiguous lines, many short lines (the batched
//! form of a segmented scan) and strided lines.

use luma::*;

use super::{check64, host, pick, upload};
use crate::harness::*;

/// Running sums along `axis` of an array with `dims` (axis 0 fastest), in `f64`.
fn cumsum_host(data: &[f32], dims: [usize; 3], axis: usize) -> Vec<f64> {
    let mut out: Vec<f64> = data.iter().map(|&v| f64::from(v)).collect();
    let stride = [1, dims[0], dims[0] * dims[1]][axis];
    for i in 0..out.len() {
        let coord = (i / stride) % dims[axis];
        if coord > 0 {
            out[i] += out[i - stride];
        }
    }
    out
}

pub fn run(h: &mut Harness) {
    h.guarded("scan", |h| {
        let cpu = h.ctx.is_cpu();

        // Long contiguous lines.
        let sizes: Vec<usize> = if cpu { vec![1 << 20, pick(h, 1 << 20, 1 << 22)] } else { vec![1 << 20, pick(h, 1 << 22, 1 << 24)] };
        let mut sizes = sizes;
        sizes.dedup();
        for n in sizes {
            let data = values(n, 51, 0.0, 1.0);
            let x = upload(h, &data, n);
            check64("scan/cumsum", &host(&x.cumsum(0)), &cumsum_host(&data, [n, 1, 1], 0), 1e-4, 1e-4);
            h.case("scan/cumsum", &format!("{n}"), Work::Bytes(8.0 * n as f64), &mut || x.cumsum(0).eval().unwrap());
        }
        let n: usize = if cpu { pick(h, 1 << 20, 1 << 22) } else { pick(h, 1 << 22, 1 << 24) };
        {
            let data = ints(n, 52, 4);
            let x = Array::<u32>::from_slice_on(&h.ctx, &data, n).unwrap();
            x.eval().unwrap();
            let mut want = data.clone();
            for i in 1..n {
                want[i] += want[i - 1];
            }
            assert_eq!(x.cumsum(0).to_vec().unwrap(), want, "scan/cumsum_u32");
            h.case("scan/cumsum_u32", &format!("{n}"), Work::Bytes(8.0 * n as f64), &mut || x.cumsum(0).eval().unwrap());
        }
        {
            let n = n.min(1 << 22);
            let data = values(n, 53, -1.0, 1.0);
            let x = upload(h, &data, n);
            let mut want = data.clone();
            for i in 1..n {
                want[i] = want[i].max(want[i - 1]);
            }
            assert_eq!(host(&x.scan(0, |a, b| a.maximum(b))), want, "scan/max_custom");
            h.case("scan/max_custom", &format!("{n}"), Work::Bytes(8.0 * n as f64), &mut || x.scan(0, |a, b| a.maximum(b)).eval().unwrap());
        }

        // Many short lines (axis 0), mid-length lines, and strided lines (axes 1 and 2).
        let lines: [(&str, [usize; 3], usize); 5] = [
            ("scan/cumsum_short_lines", [64, if cpu { pick(h, 1 << 14, 1 << 16) } else { pick(h, 1 << 16, 1 << 18) }, 1], 0),
            ("scan/cumsum_mid_lines", [1024, if cpu { pick(h, 1 << 10, 1 << 12) } else { pick(h, 1 << 12, 1 << 14) }, 1], 0),
            ("scan/cumsum_strided", { let s = if cpu { pick(h, 1024, 2048) } else { pick(h, 2048, 4096) }; [s, s, 1] }, 1),
            ("scan/cumsum_strided_wide", [4, if cpu { pick(h, 1 << 18, 1 << 20) } else { pick(h, 1 << 20, 1 << 22) }, 1], 1),
            ("scan/cumsum_axis2", { let s = if cpu { pick(h, 64, 128) } else { pick(h, 128, 256) }; [s, s, s] }, 2),
        ];
        for (name, dims, axis) in lines {
            let len = dims[0] * dims[1] * dims[2];
            let data = values(len, 54, 0.0, 1.0);
            let x = upload(h, &data, [dims[0], dims[1], dims[2]]);
            check64(name, &host(&x.cumsum(axis)), &cumsum_host(&data, dims, axis), 1e-4, 1e-4);
            h.case(name, &format!("{}x{}x{} axis {axis}", dims[0], dims[1], dims[2]), Work::Bytes(8.0 * len as f64), &mut || x.cumsum(axis).eval().unwrap());
        }

        // Reductions along an axis of a square array: contiguous and strided lines.
        let s: usize = if cpu { pick(h, 1024, 2048) } else { pick(h, 2048, 4096) };
        let data = values(s * s, 55, 0.0, 1.0);
        let x = upload(h, &data, [s, s]);
        let col = |f: &dyn Fn(&mut dyn Iterator<Item = f64>) -> f64, axis: usize| -> Vec<f64> {
            (0..s)
                .map(|o| {
                    let mut it = (0..s).map(|i| f64::from(if axis == 0 { data[i + s * o] } else { data[o + s * i] }));
                    f(&mut it)
                })
                .collect()
        };
        let bytes = Work::Bytes(4.0 * (s * s) as f64);
        let size = format!("{s}x{s}");
        let sum = |it: &mut dyn Iterator<Item = f64>| it.sum::<f64>();
        let max = |it: &mut dyn Iterator<Item = f64>| it.fold(f64::MIN, f64::max);
        check64("scan/sum_axis0", &host(&x.sum_axis(0)), &col(&sum, 0), 1e-4, 1e-4);
        h.case("scan/sum_axis0", &size, bytes, &mut || x.sum_axis(0).eval().unwrap());
        check64("scan/sum_axis1", &host(&x.sum_axis(1)), &col(&sum, 1), 1e-4, 1e-4);
        h.case("scan/sum_axis1", &size, bytes, &mut || x.sum_axis(1).eval().unwrap());
        check64("scan/max_axis1", &host(&x.max_axis(1)), &col(&max, 1), 0.0, 0.0);
        h.case("scan/max_axis1", &size, bytes, &mut || x.max_axis(1).eval().unwrap());
        let var = |it: &mut dyn Iterator<Item = f64>| {
            let v: Vec<f64> = it.collect();
            let mean = v.iter().sum::<f64>() / v.len() as f64;
            v.iter().map(|a| (a - mean) * (a - mean)).sum::<f64>() / v.len() as f64
        };
        check64("scan/var_axis0", &host(&x.var_axis(0)), &col(&var, 0), 1e-3, 1e-5);
        h.case("scan/var_axis0", &size, bytes, &mut || x.var_axis(0).eval().unwrap());
        let both = x.reduce_axis(0, [Reduction::Sum, Reduction::Max]);
        check64("scan/reduce2_axis0 sum", &host(&both[0]), &col(&sum, 0), 1e-4, 1e-4);
        check64("scan/reduce2_axis0 max", &host(&both[1]), &col(&max, 0), 0.0, 0.0);
        h.case("scan/reduce2_axis0", &size, bytes, &mut || {
            let [a, b] = x.reduce_axis(0, [Reduction::Sum, Reduction::Max]);
            eval_all(&[&a, &b]).unwrap();
        });
        let am = host(&x.argmax_axis(1).to_f32());
        let want: Vec<f64> = (0..s)
            .map(|o| {
                let mut best = (f32::MIN, 0usize);
                for i in 0..s {
                    if data[o + s * i] > best.0 {
                        best = (data[o + s * i], i);
                    }
                }
                best.1 as f64
            })
            .collect();
        check64("scan/argmax_axis1", &am, &want, 0.0, 0.0);
        h.case("scan/argmax_axis1", &size, bytes, &mut || x.argmax_axis(1).eval().unwrap());
    });
}
