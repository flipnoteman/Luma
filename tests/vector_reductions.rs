//! Per-component reductions of `Array<[f32; 2]>` and `Array<[f32; 4]>`.

mod common;

use common::values;
use luma::*;

/// `n` elements of `N` components; component `c` uses its own value range, so a mix-up between
/// components cannot go unnoticed.
fn vectors<const N: usize>(n: usize, seed: u64) -> Vec<[f32; N]> {
    let columns: Vec<Vec<f32>> = (0..N)
        .map(|c| {
            let lo = -3.0 * c as f32 - 1.0;
            let hi = 5.0 * c as f32 + 2.0;
            values(n, seed + c as u64, lo, hi)
        })
        .collect();
    (0..n)
        .map(|i| std::array::from_fn(|c| columns[c][i]))
        .collect()
}

struct Reference<const N: usize> {
    sum: [f64; N],
    min: [f32; N],
    max: [f32; N],
    norm: [f64; N],
}

fn reference<const N: usize>(data: &[[f32; N]]) -> Reference<N> {
    let mut r = Reference {
        sum: [0.0; N],
        min: [f32::INFINITY; N],
        max: [f32::NEG_INFINITY; N],
        norm: [0.0; N],
    };
    for v in data {
        for c in 0..N {
            r.sum[c] += f64::from(v[c]);
            r.norm[c] += f64::from(v[c]) * f64::from(v[c]);
            r.min[c] = r.min[c].min(v[c]);
            r.max[c] = r.max[c].max(v[c]);
        }
    }
    r.norm = r.norm.map(f64::sqrt);
    r
}

/// Relative difference scaled by the magnitude of the sum of absolute values, which is what the
/// accuracy of a float sum is measured against (a sum of mixed signs can be tiny).
fn close(got: f32, want: f64, scale: f64, rel: f64, what: &str) {
    let error = (f64::from(got) - want).abs();
    assert!(
        error <= rel * scale.max(1.0),
        "{what}: got {got}, want {want} (error {error:e})"
    );
}

fn check<const N: usize>(
    sum: [f32; N],
    min: [f32; N],
    max: [f32; N],
    norm: [f32; N],
    data: &[[f32; N]],
) {
    let want = reference(data);
    let abs_sum: Vec<f64> = (0..N)
        .map(|c| data.iter().map(|v| f64::from(v[c]).abs()).sum::<f64>())
        .collect();
    for c in 0..N {
        close(
            sum[c],
            want.sum[c],
            abs_sum[c],
            2e-6,
            &format!("sum[{c}] of {}", data.len()),
        );
        close(
            norm[c],
            want.norm[c],
            want.norm[c],
            2e-6,
            &format!("norm[{c}] of {}", data.len()),
        );
        assert_eq!(min[c], want.min[c], "min[{c}] of {}", data.len());
        assert_eq!(max[c], want.max[c], "max[{c}] of {}", data.len());
    }
}

const SIZES: [usize; 9] = [1, 2, 3, 255, 1024, 1025, 4097, 100_003, 3_000_000];

#[test]
fn vec4_reductions_match_a_f64_reference() {
    for (k, &n) in SIZES.iter().enumerate() {
        let data = vectors::<4>(n, 11 + k as u64);
        let a = Array::from_slice(&data, n).unwrap();
        check(
            a.sum().unwrap(),
            a.min().unwrap(),
            a.max().unwrap(),
            a.norm().unwrap(),
            &data,
        );
    }
}

#[test]
fn vec2_reductions_match_a_f64_reference() {
    for (k, &n) in SIZES.iter().enumerate() {
        let data = vectors::<2>(n, 101 + k as u64);
        let a = Array::from_slice(&data, n).unwrap();
        check(
            a.sum().unwrap(),
            a.min().unwrap(),
            a.max().unwrap(),
            a.norm().unwrap(),
            &data,
        );
    }
}

#[test]
fn vector_sums_stay_accurate_for_ten_million_elements() {
    // Like the scalar test: ten million elements around 1, error of a few 1e-7 relative.
    let n = 10_000_000usize;
    let data: Vec<[f32; 4]> = values(n, 5, 0.9, 1.1)
        .into_iter()
        .map(|v| [v, 2.0 * v, 0.5 * v, 1.0])
        .collect();
    let a = Array::from_slice(&data, n).unwrap();
    let sum = a.sum().unwrap();
    let want = reference(&data).sum;
    for c in 0..4 {
        let rel = (f64::from(sum[c]) - want[c]).abs() / want[c];
        assert!(
            rel < 1e-6,
            "component {c}: got {}, want {}, relative error {rel:e}",
            sum[c],
            want[c]
        );
    }
}

#[test]
fn vector_reductions_of_known_values_and_expression_inputs() {
    let a = Array::from_slice(&[[1.0f32, -2.0], [3.0, 4.0], [-5.0, 0.5]], 3).unwrap();
    assert_eq!(a.sum().unwrap(), [-1.0, 2.5]);
    assert_eq!(a.min().unwrap(), [-5.0, -2.0]);
    assert_eq!(a.max().unwrap(), [3.0, 4.0]);
    let norm = a.norm().unwrap();
    assert!((norm[0] - 35.0f32.sqrt()).abs() < 1e-6 && (norm[1] - 20.25f32.sqrt()).abs() < 1e-6);

    // Reductions see results of kernels recorded before them (same batch) ...
    const DOUBLE: &str = r#"
    @group(1) @binding(0) var<storage, read> src: array<vec4<f32>>;
    @group(1) @binding(1) var<storage, read_write> dst: array<vec4<f32>>;
    @compute @workgroup_size(64)
    fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
        if (gid.x < luma.size.x) { dst[gid.x] = src[gid.x] * 2.0; }
    }"#;
    let src = Array::from_slice(&[[1.0f32, 2.0, 3.0, 4.0]; 1000], 1000).unwrap();
    let mut dst = Array::<[f32; 4]>::zeros(1000).unwrap();
    Kernel::new(DOUBLE)
        .unwrap()
        .launch([1000], args![&src, &mut dst])
        .unwrap();
    assert_eq!(dst.sum().unwrap(), [2000.0, 4000.0, 6000.0, 8000.0]);
    // ... and leave the input intact.
    assert_eq!(src.sum().unwrap(), [1000.0, 2000.0, 3000.0, 4000.0]);
}

#[test]
fn vector_reductions_report_poisoned_inputs() {
    let a = Array::from_slice(&[[1.0f32; 4]; 4], 4).unwrap();
    let bad = a.reshape(5);
    assert!(bad.error().is_some());
    assert!(matches!(
        bad.sum().unwrap_err().root(),
        LumaError::InvalidShape(_)
    ));
    assert!(bad.min().is_err() && bad.max().is_err() && bad.norm().is_err());
}

#[test]
fn vector_reductions_work_next_to_fused_scalar_reductions() {
    let ctx = Context::new().unwrap();
    let n = 10_000;
    let v = Array::<[f32; 4]>::from_slice_on(&ctx, &vec![[1.0, 2.0, 3.0, 4.0]; n], n).unwrap();
    let x = Array::<f32>::full_on(&ctx, n, 2.0).unwrap();
    // A fused scalar reduction and a vector reduction in one batch.
    let lazy_sum = (&x * &x + 1.0).sum().unwrap();
    let sum = v.sum().unwrap();
    assert_eq!(lazy_sum, 5.0 * n as f32);
    assert_eq!(sum, [n as f32, 2.0 * n as f32, 3.0 * n as f32, 4.0 * n as f32]);
}
