//! `nansum`, `nanmean`, `nanmin`, `nanmax` and their axis forms against host computations with
//! NumPy's semantics.

mod common;

use common::values;
use luma::{Array, Context, Df32};

/// `n` values in `-100..100` with every `period`-th one NaN.
fn with_nans(n: usize, seed: u64, period: usize) -> Vec<f32> {
    let mut data = values(n, seed, -100.0, 100.0);
    for i in (0..n).step_by(period) {
        data[i] = f32::NAN;
    }
    data
}

fn host_nansum(data: &[f32]) -> f64 {
    data.iter().filter(|v| !v.is_nan()).map(|&v| f64::from(v)).sum()
}

/// The sum of the magnitudes of the non-NaN elements: what the rounding error of a sum scales with.
fn host_magnitude(data: &[f32]) -> f64 {
    data.iter().filter(|v| !v.is_nan()).map(|&v| f64::from(v).abs()).sum()
}

fn close(got: f32, want: f64, magnitude: f64, what: &str) {
    assert!((f64::from(got) - want).abs() <= 2e-6 * magnitude.max(1.0), "{what}: got {got}, want {want}");
}

#[test]
fn whole_array_reductions_skip_nans() {
    let ctx = Context::new().unwrap();
    for (n, period) in [(1000usize, 7usize), (1 << 20, 3), (5, 2), (1, 1)] {
        let data = with_nans(n, 1, period);
        let x = Array::from_slice_on(&ctx, &data, n).unwrap();
        let valid: Vec<f32> = data.iter().copied().filter(|v| !v.is_nan()).collect();
        let sum = host_nansum(&data);
        close(x.nansum().item().unwrap(), sum, host_magnitude(&data), "nansum");
        if valid.is_empty() {
            assert!(x.nanmin().item().unwrap().is_nan() && x.nanmax().item().unwrap().is_nan() && x.nanmean().item().unwrap().is_nan());
            assert_eq!(x.nansum().item().unwrap(), 0.0);
            continue;
        }
        close(x.nanmean().item().unwrap(), sum / valid.len() as f64, host_magnitude(&data) / valid.len() as f64, "nanmean");
        assert_eq!(x.nanmin().item().unwrap(), valid.iter().copied().fold(f32::INFINITY, f32::min), "n = {n}");
        assert_eq!(x.nanmax().item().unwrap(), valid.iter().copied().fold(f32::NEG_INFINITY, f32::max), "n = {n}");
    }
}

#[test]
fn infinities_are_values_and_all_nan_gives_nan() {
    let ctx = Context::new().unwrap();
    let nan = f32::NAN;
    let inf = f32::INFINITY;
    // An infinity is the extremum even when NaNs surround it; only an all-NaN array gives NaN.
    let x = Array::from_slice_on(&ctx, &[nan, inf, nan], 3).unwrap();
    assert_eq!(x.nanmin().item().unwrap(), inf);
    assert_eq!(x.nanmax().item().unwrap(), inf);
    assert_eq!(x.nansum().item().unwrap(), inf);
    let y = Array::from_slice_on(&ctx, &[nan, -inf, 2.0], 3).unwrap();
    assert_eq!(y.nanmin().item().unwrap(), -inf);
    assert_eq!(y.nanmax().item().unwrap(), 2.0);
    let all = Array::from_slice_on(&ctx, &[nan, nan, nan, nan], 4).unwrap();
    assert!(all.nanmin().item().unwrap().is_nan());
    assert!(all.nanmax().item().unwrap().is_nan());
    assert!(all.nanmean().item().unwrap().is_nan());
    assert_eq!(all.nansum().item().unwrap(), 0.0);
    // Without NaNs they are the plain reductions.
    let plain = Array::from_slice_on(&ctx, &[3.0f32, -1.0, 4.0], 3).unwrap();
    assert_eq!((plain.nansum().item().unwrap(), plain.nanmin().item().unwrap(), plain.nanmax().item().unwrap()), (6.0, -1.0, 4.0));
    assert!((plain.nanmean().item().unwrap() - 2.0).abs() < 1e-6);
}

#[test]
fn reductions_of_pending_expressions_fuse_the_masking() {
    let ctx = Context::new().unwrap();
    let data = with_nans(1 << 16, 3, 5);
    let x = Array::from_slice_on(&ctx, &data, data.len()).unwrap();
    x.eval().unwrap();
    // `x * 2` turns NaNs into NaNs: the mask is computed from the expression, not a stored copy.
    let (got, profile) = ctx.profile(|| (&x * 2.0).nansum().item().unwrap()).unwrap();
    close(got, 2.0 * host_nansum(&data), 2.0 * host_magnitude(&data), "nansum of a product");
    if ctx.profiling_supported() {
        assert!(profile.launches() <= 3, "{profile:?}");
    }
}

#[test]
fn axis_reductions_skip_nans_along_each_axis() {
    let ctx = Context::new().unwrap();
    let (rows, cols) = (37usize, 1000usize);
    let mut data = with_nans(rows * cols, 2, 4);
    // Make one line all NaN along axis 0 (a row of `cols`) and one along axis 1.
    for c in 0..cols {
        data[5 * cols + c] = f32::NAN;
    }
    for r in 0..rows {
        data[r * cols + 9] = f32::NAN;
    }
    // Luma shape [cols, rows]: axis 0 is a row of the host matrix, axis 1 a column.
    let x = Array::from_slice_on(&ctx, &data, [cols, rows]).unwrap();
    let line = |axis: usize, l: usize| -> Vec<f32> {
        if axis == 0 { data[l * cols..(l + 1) * cols].to_vec() } else { (0..rows).map(|r| data[r * cols + l]).collect() }
    };
    for axis in 0..2usize {
        let lines = if axis == 0 { rows } else { cols };
        let sum = x.nansum_axis(axis).to_vec().unwrap();
        let mean = x.nanmean_axis(axis).to_vec().unwrap();
        let min = x.nanmin_axis(axis).to_vec().unwrap();
        let max = x.nanmax_axis(axis).to_vec().unwrap();
        assert_eq!(sum.len(), lines);
        for l in 0..lines {
            let v = line(axis, l);
            let valid: Vec<f32> = v.iter().copied().filter(|e| !e.is_nan()).collect();
            close(sum[l], host_nansum(&v), host_magnitude(&v), &format!("nansum_axis({axis}) line {l}"));
            if valid.is_empty() {
                assert!(mean[l].is_nan() && min[l].is_nan() && max[l].is_nan(), "axis {axis} line {l}: {} {} {}", mean[l], min[l], max[l]);
                assert_eq!(sum[l], 0.0);
            } else {
                close(mean[l], host_nansum(&v) / valid.len() as f64, host_magnitude(&v) / valid.len() as f64, &format!("nanmean_axis({axis}) line {l}"));
                assert_eq!(min[l], valid.iter().copied().fold(f32::INFINITY, f32::min), "axis {axis} line {l}");
                assert_eq!(max[l], valid.iter().copied().fold(f32::NEG_INFINITY, f32::max), "axis {axis} line {l}");
            }
        }
    }
    // The shape keeps extent 1 along the axis.
    assert_eq!(x.nansum_axis(1).shape().dims(), [cols, 1, 1, 1]);
}

#[test]
fn f64_and_df32_have_nansum_nanmin_and_nanmax() {
    let ctx = Context::new().unwrap();
    if !ctx.features().contains(luma::wgpu::Features::SHADER_F64) {
        return;
    }
    let data = [1.5f64, f64::NAN, -2.25, 8.0, f64::NAN, 0.125];
    let x = Array::from_slice_on(&ctx, &data, 6).unwrap();
    assert_eq!(x.nansum().item().unwrap(), 7.375);
    assert_eq!(x.nanmin().item().unwrap(), -2.25);
    assert_eq!(x.nanmax().item().unwrap(), 8.0);
    let all = Array::from_slice_on(&ctx, &[f64::NAN, f64::NAN], 2).unwrap();
    assert!(all.nanmin().item().unwrap().is_nan() && all.nanmax().item().unwrap().is_nan());
    assert_eq!(all.nansum().item().unwrap(), 0.0);

    let df: Vec<Df32> = data.iter().map(|&v| Df32::from(v)).collect();
    let d = Array::from_slice_on(&ctx, &df, 6).unwrap();
    assert_eq!(d.nansum().item().unwrap().to_f64(), 7.375);
    assert_eq!(d.nanmin().item().unwrap().to_f64(), -2.25);
    assert_eq!(d.nanmax().item().unwrap().to_f64(), 8.0);
    let nan = Df32 { hi: f32::NAN, lo: 0.0 };
    let all = Array::from_slice_on(&ctx, &[nan, nan], 2).unwrap();
    assert!(all.nanmin().item().unwrap().to_f64().is_nan() && all.nanmax().item().unwrap().to_f64().is_nan());
    assert_eq!(all.nansum().item().unwrap().to_f64(), 0.0);
}
