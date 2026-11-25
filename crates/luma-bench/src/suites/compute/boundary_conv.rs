//! Host-time micro-cases of the spectral, sample, conv, window and sparse-product ops (16 to 1024 elements): what a call costs on
//! the host, selection included, dominates. `tools/hosttime --case compute/boundary/fft_small ...` counts the instructions of one
//! call; W2-U3c gates them against the code before the registry took over the selections.

use luma::*;

use super::{host, upload};
use crate::harness::*;

pub fn run(h: &mut Harness) {
    h.guarded("boundary", |h| {
        let n = 1024usize;
        let data = values(n, 71, 0.0, 1.0);
        let x = upload(h, &data, n);
        let tag = format!("n={n}");

        let z = Array::<C32>::from_slice_on(&h.ctx, &data.iter().map(|&v| C32::new(v, 1.0 - v)).collect::<Vec<_>>(), n).unwrap();
        z.eval().unwrap();
        assert_eq!(z.fft().to_vec().unwrap().len(), n, "boundary/fft_small");
        h.case("boundary/fft_small", &tag, Work::Bytes(16.0 * n as f64), &mut || z.fft().eval().unwrap());

        let pos = upload(h, &values(n, 72, 0.0, (n - 1) as f32), n);
        let at = x.approx1(&pos, Interp::Linear, 0.0).to_vec().unwrap();
        let want = host(&pos).into_iter().map(|p| {
            let (i, f) = (p.floor() as usize, p - p.floor());
            if i + 1 < n { data[i] * (1.0 - f) + data[i + 1] * f } else { data[n - 1] }
        });
        assert!(at.iter().zip(want).all(|(g, w)| (g - w).abs() < 1e-4), "boundary/approx1_small");
        h.case("boundary/approx1_small", &tag, Work::Bytes(12.0 * n as f64), &mut || x.approx1(&pos, Interp::Linear, 0.0).eval().unwrap());

        let taps = upload(h, &values(16, 73, -1.0, 1.0), 16usize);
        let got = x.fir(&taps).to_vec().unwrap();
        let b = host(&taps);
        assert!((got[100] - (0..16).map(|j| b[j] * data[100 - j]).sum::<f32>()).abs() < 1e-4, "boundary/fir_small");
        h.case("boundary/fir_small", &tag, Work::Bytes(8.0 * n as f64), &mut || x.fir(&taps).eval().unwrap());

        let img = upload(h, &data, [32usize, 32]);
        let med = host(&img.medfilt2([3, 3], Border::Edge));
        let mut around: Vec<f32> = (0..3).flat_map(|dy| (0..3).map(move |dx| (dx, dy))).map(|(dx, dy)| data[(15 + dy) * 32 + 15 + dx]).collect();
        around.sort_by(f32::total_cmp);
        assert_eq!(med[16 * 32 + 16], around[4], "boundary/medfilt_small");
        h.case("boundary/medfilt_small", "32x32", Work::Bytes(8.0 * n as f64), &mut || img.medfilt2([3, 3], Border::Edge).eval().unwrap());

        let (rows, cols) = (32usize, 32usize);
        let dense: Vec<f32> = (0..rows * cols).map(|i| if (i * 7 + i / cols) % 8 == 0 { 1.0 + (i % 5) as f32 } else { 0.0 }).collect();
        let a = sparse(&upload(h, &dense, [cols, rows]), SparseFormat::Csr);
        let v = upload(h, &data[..cols], cols);
        let y = host(&a.matmul(&v));
        let want: f32 = (0..cols).map(|c| dense[c] * data[c]).sum();
        assert!((y[0] - want).abs() < 1e-3, "boundary/spmv_small");
        h.case("boundary/spmv_small", "32x32", Work::Bytes(8.0 * n as f64), &mut || a.matmul(&v).eval().unwrap());
    });
}
