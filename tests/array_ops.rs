use luma::*;

/// Arrays on the default context, which all tests share; creating a device per test would be slow
/// and wasteful.
fn array<T: GpuType>(shape: impl Into<Shape>, data: &[T]) -> Array<T> {
    Array::from_slice(data, shape).expect("Could not create Array.")
}

// -- Creation & readback --

#[test]
fn create_u32_array_and_readback() {
    let a = array([3, 1, 1, 1], &[10u32, 20u32, 30u32]);
    assert_eq!(a.to_vec().unwrap(), vec![10u32, 20, 30]);
}

#[test]
fn create_f32_array_and_readback() {
    let a = array([3, 1, 1, 1], &[1.5f32, 2.5f32, 3.5f32]);
    assert_eq!(a.to_vec().unwrap(), vec![1.5f32, 2.5, 3.5]);
}

#[test]
fn shape_returns_correct_value() {
    let a = array([3, 1, 1, 1], &[1u32, 2u32, 3u32]);
    assert_eq!(a.shape(), [3, 1, 1, 1]);
    assert_eq!(a.len(), 3);
}

#[test]
fn zeros_returns_zeroed_array() {
    let a = Array::<f32>::zeros([2, 3]).unwrap();
    assert_eq!(a.shape(), [2, 3, 1, 1]);
    assert_eq!(a.to_vec().unwrap(), vec![0.0f32; 6]);
}

// -- Write & shape validation --

#[test]
fn write_replaces_contents() {
    let mut a = array([3, 1, 1, 1], &[1u32, 2u32, 3u32]);
    a.write(&[7u32, 8u32, 9u32]).unwrap();
    assert_eq!(a.to_vec().unwrap(), vec![7u32, 8, 9]);
}

#[test]
fn write_length_mismatch_returns_error() {
    let mut a = array([3, 1, 1, 1], &[1u32, 2u32, 3u32]);

    let err = a.write(&[1u32, 2u32]).unwrap_err();
    assert!(
        matches!(err, LumaError::LengthMismatch { expected: 3, actual: 2 }),
        "Expected LengthMismatch, got: {err}"
    );
    let err = a.write(&[1u32, 2u32, 3u32, 4u32]).unwrap_err();
    assert!(
        matches!(err, LumaError::LengthMismatch { expected: 3, actual: 4 }),
        "Expected LengthMismatch, got: {err}"
    );

    // A rejected write leaves the array untouched.
    assert_eq!(a.to_vec().unwrap(), vec![1u32, 2, 3]);
}

#[test]
fn from_slice_length_mismatch_returns_error() {
    let err = Array::from_slice(&[1u32, 2u32], 3).unwrap_err();
    assert!(
        matches!(err, LumaError::LengthMismatch { expected: 3, actual: 2 }),
        "Expected LengthMismatch, got: {err}"
    );
}

#[test]
fn invalid_shapes_return_errors() {
    let err = Array::<u32>::zeros([3, 0]).unwrap_err();
    assert!(matches!(err, LumaError::InvalidShape(_)), "got: {err}");

    let err = Array::<u32>::zeros(u32::MAX as usize + 1).unwrap_err();
    assert!(matches!(err, LumaError::DimensionTooLarge { .. }), "got: {err}");

    // Each dimension fits u32, but the element count does not.
    let err = Array::<u32>::zeros([65_536, 65_536]).unwrap_err();
    assert!(matches!(err, LumaError::DimensionTooLarge { .. }), "got: {err}");
}

// -- Scalar multiplication (replaces the old `double`) --

#[test]
fn times_two_u32() {
    let a = array([3, 1, 1, 1], &[1u32, 2u32, 3u32]);
    let result = (&a * 2u32).to_vec().unwrap();
    assert_eq!(result, vec![2u32, 4, 6]);
}

#[test]
fn times_two_f32() {
    let a = array([2, 1, 1, 1], &[1.5f32, 2.5f32]);
    let result = (&a * 2.0f32).to_vec().unwrap();
    assert_eq!(result, vec![3.0f32, 5.0]);
}

// -- Binary operations (u32) --

#[test]
fn add_u32() {
    let a = array([3, 1, 1, 1], &[1u32, 2u32, 3u32]);
    let b = array([3, 1, 1, 1], &[4u32, 5u32, 6u32]);
    let result = (&a + &b).to_vec().unwrap();
    assert_eq!(result, vec![5u32, 7, 9]);
}

#[test]
fn subtract_u32() {
    let a = array([3, 1, 1, 1], &[10u32, 20u32, 30u32]);
    let b = array([3, 1, 1, 1], &[1u32, 2u32, 3u32]);
    let result = (&a - &b).to_vec().unwrap();
    assert_eq!(result, vec![9u32, 18, 27]);
}

#[test]
fn multiply_u32() {
    let a = array([3, 1, 1, 1], &[2u32, 3u32, 4u32]);
    let b = array([3, 1, 1, 1], &[5u32, 6u32, 7u32]);
    let result = (&a * &b).to_vec().unwrap();
    assert_eq!(result, vec![10u32, 18, 28]);
}

#[test]
fn divide_u32() {
    let a = array([3, 1, 1, 1], &[10u32, 20u32, 30u32]);
    let b = array([3, 1, 1, 1], &[2u32, 4u32, 5u32]);
    let result = (&a / &b).to_vec().unwrap();
    assert_eq!(result, vec![5u32, 5, 6]);
}

// -- Binary operations (f32) --

#[test]
fn add_f32() {
    let a = array([3, 1, 1, 1], &[1.0f32, 2.5f32, 3.0f32]);
    let b = array([3, 1, 1, 1], &[4.0f32, 5.5f32, 6.0f32]);
    let result = (&a + &b).to_vec().unwrap();
    assert_eq!(result, vec![5.0f32, 8.0, 9.0]);
}

// -- Large arrays (exceed the 65,535 workgroups-per-dimension dispatch limit) --

const LARGE_LEN: usize = 5_000_003; // > 65_535 * 64, and not a multiple of 64

#[test]
fn add_f32_large_array() {
    let lhs: Vec<f32> = (0..LARGE_LEN).map(|i| i as f32).collect();
    let rhs = vec![0.5f32; LARGE_LEN];
    let a = array([LARGE_LEN, 1, 1, 1], &lhs);
    let b = array([LARGE_LEN, 1, 1, 1], &rhs);
    let result = (&a + &b).to_vec().unwrap();
    assert_eq!(result.len(), LARGE_LEN);
    for i in [0, 1, 63, 64, 65_535 * 64, LARGE_LEN - 1] {
        assert_eq!(result[i], i as f32 + 0.5, "index {i}");
    }
}

#[test]
fn times_two_u32_large_array() {
    let data: Vec<u32> = (0..LARGE_LEN as u32).collect();
    let a = array([LARGE_LEN, 1, 1, 1], &data);
    let result = (&a * 2u32).to_vec().unwrap();
    assert_eq!(result.len(), LARGE_LEN);
    for i in [0, 1, 63, 64, 65_535 * 64, LARGE_LEN - 1] {
        assert_eq!(result[i], 2 * i as u32, "index {i}");
    }
}

// -- Scalar operations (u32) --

#[test]
fn add_scalar_u32() {
    let a = array([3, 1, 1, 1], &[1u32, 2u32, 3u32]);
    let result = (&a + 10u32).to_vec().unwrap();
    assert_eq!(result, vec![11u32, 12, 13]);
}

#[test]
fn subtract_scalar_u32() {
    let a = array([3, 1, 1, 1], &[5u32, 10u32, 15u32]);
    let result = (&a - 1u32).to_vec().unwrap();
    assert_eq!(result, vec![4u32, 9, 14]);
}

#[test]
fn multiply_scalar_u32() {
    let a = array([3, 1, 1, 1], &[2u32, 4u32, 6u32]);
    let result = (&a * 3u32).to_vec().unwrap();
    assert_eq!(result, vec![6u32, 12, 18]);
}

#[test]
fn divide_scalar_u32() {
    let a = array([3, 1, 1, 1], &[10u32, 20u32, 30u32]);
    let result = (&a / 2u32).to_vec().unwrap();
    assert_eq!(result, vec![5u32, 10, 15]);
}

// -- Error case --

#[test]
fn add_dimension_mismatch_returns_error() {
    let a = array([3, 1, 1, 1], &[1u32, 2u32, 3u32]);
    let b = array([2, 1, 1, 1], &[4u32, 5u32]);
    let err = a.try_add(&b).unwrap_err();
    assert!(
        matches!(err, LumaError::ShapeMismatch { lhs: [3, 1, 1, 1], rhs: [2, 1, 1, 1] }),
        "Expected ShapeMismatch, got: {err}"
    );
    let err = err.to_string();
    assert!(err.contains("Dimension mismatch"), "Expected 'Dimension mismatch', got: {err}");

    // The operator form defers the same error to the first synchronization point.
    let err = (&a + &b).to_vec().unwrap_err();
    assert!(
        matches!(err.root(), LumaError::ShapeMismatch { lhs: [3, 1, 1, 1], rhs: [2, 1, 1, 1] }),
        "Expected ShapeMismatch, got: {err}"
    );
    let err = err.to_string();
    assert!(err.contains("Dimension mismatch"), "Expected 'Dimension mismatch', got: {err}");
}

// -- Conv1D (f32 only) --

#[test]
fn conv1d_basic() {
    let input = array([5, 1, 1, 1], &[1.0f32, 2.0, 3.0, 4.0, 5.0]);
    let kernel = array([3, 1, 1, 1], &[1.0f32, 0.0, -1.0]);
    let result = input.conv1d(&kernel, 1, 0).to_vec().unwrap();
    assert_eq!(result, vec![-2.0f32, -2.0, -2.0]);
}

#[test]
fn conv1d_with_padding() {
    let input = array([3, 1, 1, 1], &[1.0f32, 2.0, 3.0]);
    let kernel = array([2, 1, 1, 1], &[1.0f32, 1.0]);
    let result = input.conv1d(&kernel, 1, 1).to_vec().unwrap();
    assert_eq!(result, vec![1.0f32, 3.0, 5.0, 3.0]);
}

#[test]
fn conv1d_with_stride() {
    let input = array([6, 1, 1, 1], &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let kernel = array([2, 1, 1, 1], &[1.0f32, 1.0]);
    let result = input.conv1d(&kernel, 2, 0).to_vec().unwrap();
    assert_eq!(result, vec![3.0f32, 7.0, 11.0]);
}

#[test]
fn conv1d_identity_kernel() {
    let input = array([3, 1, 1, 1], &[3.0f32, 7.0, 11.0]);
    let kernel = array([1, 1, 1, 1], &[1.0f32]);
    let result = input.conv1d(&kernel, 1, 0).to_vec().unwrap();
    assert_eq!(result, vec![3.0f32, 7.0, 11.0]);
}

#[test]
fn conv1d_kernel_too_large_returns_error() {
    let input = array([2, 1, 1, 1], &[1.0f32, 2.0]);
    let kernel = array([5, 1, 1, 1], &[1.0f32, 1.0, 1.0, 1.0, 1.0]);
    let err = input.conv1d(&kernel, 1, 0).to_vec().unwrap_err().to_string();
    assert!(err.contains("larger"), "Expected kernel-too-large error, got: {err}");
}

#[test]
fn conv1d_large_input() {
    // 100-element input, 5-element averaging kernel, stride=1, pad=0 → 96 outputs
    let input_data: Vec<f32> = (1..=100).map(|x| x as f32).collect();
    let kernel_data = vec![0.2f32; 5]; // averaging kernel
    let input = array([100, 1, 1, 1], &input_data);
    let kernel = array([5, 1, 1, 1], &kernel_data);
    let result = input.conv1d(&kernel, 1, 0).to_vec().unwrap();
    assert_eq!(result.len(), 96);
    // First output: mean(1,2,3,4,5) = 3.0
    assert!((result[0] - 3.0).abs() < 1e-5);
    // Last output: mean(96,97,98,99,100) = 98.0
    assert!((result[95] - 98.0).abs() < 1e-5);
}

#[test]
fn conv1d_large_with_stride() {
    // 1000 elements, kernel size 3, stride=4, pad=0 → 250 outputs
    let input_data: Vec<f32> = (0..1000).map(|x| (x as f32).sin()).collect();
    let kernel = array([3, 1, 1, 1], &[0.25f32, 0.5, 0.25]);
    let input = array([1000, 1, 1, 1], &input_data);
    let result = input.conv1d(&kernel, 4, 0).to_vec().unwrap();
    assert_eq!(result.len(), 250);
    // Spot-check first element: 0.25*in[0] + 0.5*in[1] + 0.25*in[2]
    let expected = 0.25 * (0f32).sin() + 0.5 * (1f32).sin() + 0.25 * (2f32).sin();
    assert!((result[0] - expected).abs() < 1e-5);
}

#[test]
fn conv1d_large_with_padding() {
    // 512 elements, kernel size 7, stride=1, pad=3 → 512 outputs (same-size convolution)
    let input_data: Vec<f32> = (0..512).map(|x| (x % 10) as f32).collect();
    let kernel_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0]; // symmetric
    let input = array([512, 1, 1, 1], &input_data);
    let kernel = array([7, 1, 1, 1], &kernel_data);
    let result = input.conv1d(&kernel, 1, 3).to_vec().unwrap();
    // pad=3, kernel=7 → output_len = (512 - 7 + 6)/1 + 1 = 512
    assert_eq!(result.len(), 512);
    // Check a middle element (idx=100) where no padding is involved
    let expected: f32 = (0..7).map(|k| input_data[100 - 3 + k] * kernel_data[k]).sum();
    assert!((result[100] - expected).abs() < 1e-4);
}

#[test]
fn conv1d_zero_stride_returns_error() {
    let input = array([3, 1, 1, 1], &[1.0f32, 2.0, 3.0]);
    let kernel = array([1, 1, 1, 1], &[1.0f32]);
    let err = input.conv1d(&kernel, 0, 0).to_vec().unwrap_err();
    assert!(matches!(err.root(), LumaError::InvalidShape(_)), "got: {err}");
}

// -- FFT Conv1D (f32 only) --

#[test]
fn fft_conv1d_basic() {
    // Full linear convolution: [1,2,3,4,5] * [1,0,-1] = [1, 2, 2, 2, 2, -4, -5]
    let input = array([5, 1, 1, 1], &[1.0f32, 2.0, 3.0, 4.0, 5.0]);
    let kernel = array([3, 1, 1, 1], &[1.0f32, 0.0, -1.0]);
    let result = input.conv1d_fft(&kernel);
    assert_eq!(result.shape(), [7, 1, 1, 1]);
    let data = result.to_vec().unwrap();
    let expected = vec![1.0f32, 2.0, 2.0, 2.0, 2.0, -4.0, -5.0];
    for (i, (a, b)) in data.iter().zip(expected.iter()).enumerate() {
        assert!((a - b).abs() < 1e-3, "Mismatch at index {}: got {}, expected {}", i, a, b);
    }
}

#[test]
fn fft_conv1d_identity_kernel() {
    let input = array([3, 1, 1, 1], &[3.0f32, 7.0, 11.0]);
    let kernel = array([1, 1, 1, 1], &[1.0f32]);
    let result = input.conv1d_fft(&kernel);
    assert_eq!(result.shape(), [3, 1, 1, 1]);
    let data = result.to_vec().unwrap();
    let expected = vec![3.0f32, 7.0, 11.0];
    for (i, (a, b)) in data.iter().zip(expected.iter()).enumerate() {
        assert!((a - b).abs() < 1e-3, "Mismatch at index {}: got {}, expected {}", i, a, b);
    }
}

#[test]
fn fft_conv1d_single_element() {
    let input = array([1, 1, 1, 1], &[5.0f32]);
    let kernel = array([1, 1, 1, 1], &[3.0f32]);
    let result = input.conv1d_fft(&kernel);
    assert_eq!(result.shape(), [1, 1, 1, 1]);
    let data = result.to_vec().unwrap();
    assert!((data[0] - 15.0).abs() < 1e-3, "Expected 15.0, got {}", data[0]);
}

#[test]
fn fft_conv1d_matches_spatial() {
    // Compare FFT conv with spatial conv using a SYMMETRIC kernel.
    // Spatial conv1d computes cross-correlation: sum(input[i+k] * kernel[k])
    // FFT conv1d computes true convolution: sum(input[k] * kernel[n-k]) (kernel flipped)
    // With a symmetric kernel these are equivalent in the valid region.
    let input_data: Vec<f32> = (1..=10).map(|x| x as f32).collect();
    let kernel_data = vec![0.25f32, 0.5, 0.25]; // symmetric

    let input = array([10, 1, 1, 1], &input_data);
    let kernel = array([3, 1, 1, 1], &kernel_data);

    let spatial = input.conv1d(&kernel, 1, 0).to_vec().unwrap();
    let fft = input.conv1d_fft(&kernel).to_vec().unwrap();

    assert_eq!(fft.len(), 12); // 10 + 3 - 1
    assert_eq!(spatial.len(), 8); // 10 - 3 + 1

    // With symmetric kernel, spatial[i] should match fft[i + kernel_len - 1]
    for i in 0..spatial.len() {
        let fft_val = fft[i + 2]; // offset by kernel_len - 1 = 2
        assert!(
            (spatial[i] - fft_val).abs() < 1e-2,
            "Mismatch at spatial[{}]={} vs fft[{}]={}", i, spatial[i], i + 2, fft_val
        );
    }
}

#[test]
fn fft_conv1d_large_kernel() {
    // 512-element input, 128-element kernel
    let input_data: Vec<f32> = (0..512).map(|x| (x as f32 * 0.01).sin()).collect();
    let kernel_data: Vec<f32> = (0..128).map(|x| 1.0 / (1.0 + x as f32)).collect();

    let input = array([512, 1, 1, 1], &input_data);
    let kernel = array([128, 1, 1, 1], &kernel_data);

    let result = input.conv1d_fft(&kernel);
    assert_eq!(result.shape(), [639, 1, 1, 1]); // 512 + 128 - 1
    let data = result.to_vec().unwrap();
    assert_eq!(data.len(), 639);

    // Spot-check first element: input[0] * kernel[0]
    let expected_first = input_data[0] * kernel_data[0];
    assert!(
        (data[0] - expected_first).abs() < 1e-3,
        "First element: got {}, expected {}", data[0], expected_first
    );
}

// -- Chained operations --

#[test]
fn chained_operations_produce_correct_result() {
    let a = array([3, 1, 1, 1], &[1u32, 2u32, 3u32]);
    let b = array([3, 1, 1, 1], &[10u32, 20u32, 30u32]);

    let result = ((((&a + &b) * &a   // [11, 44, 99]
        + 1u32)                      // [12, 45, 100]
        * 2u32                       // [24, 90, 200]
        - &b)                        // [14, 70, 170]
        / 2u32)                      // [7, 35, 85]
        .to_vec().unwrap();

    assert_eq!(result, vec![7u32, 35, 85]);
}

// -- Buffer pool --
//
// These tests use their own Context: the shared one is used by tests running in parallel, which
// could take the pooled buffers before the test reuses them.

#[test]
fn pooled_buffer_reuse_keeps_length_and_ignores_stale_data() {
    let ctx = Context::new().unwrap();

    // 1000 and 600 u32s (4000 and 2400 bytes) share the 4096-byte size class.
    let stale = vec![0xDEAD_BEEFu32; 1000];
    let big = Array::from_slice_on(&ctx, &stale, 1000).unwrap();
    let big_doubled = &big * 2u32;
    assert_eq!(big_doubled.to_vec().unwrap().len(), 1000);
    drop(big);
    drop(big_doubled);

    // Takes a recycled, dirty, larger-than-needed buffer.
    let data: Vec<u32> = (0..600).collect();
    let small = Array::from_slice_on(&ctx, &data, 600).unwrap();
    assert_eq!(small.len(), 600);
    assert_eq!(small.to_vec().unwrap(), data);

    let expected_doubled: Vec<u32> = data.iter().map(|x| x * 2).collect();
    assert_eq!((&small * 2u32).to_vec().unwrap(), expected_doubled);
    assert_eq!((&small + &small).to_vec().unwrap(), expected_doubled);
    let expected_plus_one: Vec<u32> = data.iter().map(|x| x + 1).collect();
    assert_eq!((&small + 1u32).to_vec().unwrap(), expected_plus_one);

    // zeros() must clear whatever the recycled buffer held before.
    drop(small);
    let zeros = Array::<u32>::zeros_on(&ctx, 600).unwrap();
    assert_eq!(zeros.to_vec().unwrap(), vec![0u32; 600]);
}

#[test]
fn pooled_scratch_reuse_does_not_change_conv_results() {
    let ctx = Context::new().unwrap();

    // Full linear convolution on the host.
    let reference = |input: &[f32], kernel: &[f32]| -> Vec<f32> {
        let mut out = vec![0.0f32; input.len() + kernel.len() - 1];
        for (i, x) in input.iter().enumerate() {
            for (k, w) in kernel.iter().enumerate() {
                out[i + k] += x * w;
            }
        }
        out
    };

    // Both convolutions have FFT size 128 (output lengths 119 and 109), so the second one runs
    // on scratch buffers the first one left dirty; its output is shorter than the first one's.
    for (input_len, kernel_len) in [(100usize, 20usize), (70, 40), (100, 20)] {
        let input_data: Vec<f32> = (0..input_len).map(|x| (x as f32 * 0.1).sin()).collect();
        let kernel_data: Vec<f32> = (0..kernel_len).map(|x| 1.0 / (1.0 + x as f32)).collect();
        let input = Array::from_slice_on(&ctx, &input_data, input_len).unwrap();
        let kernel = Array::from_slice_on(&ctx, &kernel_data, kernel_len).unwrap();

        let result = input.conv1d_fft(&kernel);
        assert_eq!(result.shape(), [input_len + kernel_len - 1, 1, 1, 1]);
        let data = result.to_vec().unwrap();
        let expected = reference(&input_data, &kernel_data);
        assert_eq!(data.len(), expected.len());
        for (i, (a, b)) in data.iter().zip(&expected).enumerate() {
            assert!((a - b).abs() < 1e-3, "Mismatch at index {i}: got {a}, expected {b}");
        }

        // Spatial conv on the same (recycled) buffers.
        let spatial = input.conv1d(&kernel, 1, 0).to_vec().unwrap();
        assert_eq!(spatial.len(), input_len - kernel_len + 1);
        for (i, got) in spatial.iter().enumerate() {
            let want: f32 = (0..kernel_len).map(|k| input_data[i + k] * kernel_data[k]).sum();
            assert!((got - want).abs() < 1e-3, "Spatial mismatch at {i}: got {got}, want {want}");
        }
    }
}

// -- Contexts --

#[test]
fn arrays_from_different_contexts_cannot_be_combined() {
    let other = Context::new().unwrap();
    let a = array([3, 1, 1, 1], &[1u32, 2u32, 3u32]);
    let b = Array::from_slice_on(&other, &[4u32, 5u32, 6u32], 3).unwrap();
    let err = a.try_add(&b).unwrap_err();
    assert!(matches!(err, LumaError::ContextMismatch), "got: {err}");
    let err = (&a + &b).to_vec().unwrap_err();
    assert!(matches!(err.root(), LumaError::ContextMismatch), "got: {err}");
}

#[test]
fn conv1d_matches_the_cpu_across_tile_and_chunk_boundaries() {
    // Lengths straddle the outputs one workgroup computes and the taps staged at a time; tap
    // counts that are not a multiple of the outputs per invocation; padding on both sides.
    let cases = [(10_000usize, 300usize, 0usize), (9_001, 517, 7), (5_000, 1, 0), (4_113, 16, 2), (300, 260, 130)];
    for (input_len, kernel_len, padding) in cases {
        let input_data: Vec<f32> = (0..input_len).map(|i| ((i * 37 % 101) as f32 - 50.0) * 0.01).collect();
        let kernel_data: Vec<f32> = (0..kernel_len).map(|k| ((k * 13 % 29) as f32 - 14.0) * 0.05).collect();
        let input = array(input_len, &input_data);
        let kernel = array(kernel_len, &kernel_data);
        let got = input.conv1d(&kernel, 1, padding).to_vec().unwrap();
        let out_len = input_len + 2 * padding - kernel_len + 1;
        assert_eq!(got.len(), out_len);
        for (o, g) in got.iter().enumerate() {
            let want: f32 = (0..kernel_len)
                .filter_map(|k| {
                    let at = (o + k).checked_sub(padding)?;
                    input_data.get(at).map(|x| x * kernel_data[k])
                })
                .sum();
            assert!((g - want).abs() < 1e-3, "{input_len}/{kernel_len}/{padding} at {o}: got {g}, want {want}");
        }
    }
}

#[test]
fn fft_conv1d_matches_the_direct_sum_for_transforms_beyond_one_tile() {
    // Transform sizes 2^12 (one column pass), 2^17 and 2^20 (two column passes).
    for (input_len, kernel_len) in [(3_000usize, 50usize), (70_000, 300), (600_000, 1_000)] {
        let input_data: Vec<f32> = (0..input_len).map(|i| ((i * 37 % 101) as f32 - 50.0) / 50.0).collect();
        let kernel_data: Vec<f32> = (0..kernel_len).map(|k| ((k * 13 % 29) as f32 - 14.0) / 14.0).collect();
        let got = array(input_len, &input_data).conv1d_fft(&array(kernel_len, &kernel_data)).to_vec().unwrap();
        assert_eq!(got.len(), input_len + kernel_len - 1);
        let mut worst = 0.0f64;
        for (o, g) in got.iter().enumerate() {
            let lo = o.saturating_sub(input_len - 1);
            let want: f64 = (lo..kernel_len.min(o + 1))
                .map(|k| f64::from(input_data[o - k]) * f64::from(kernel_data[k]))
                .sum();
            worst = worst.max((f64::from(*g) - want).abs());
        }
        // Values are O(sqrt(kernel_len)); f32 transforms of 2^20 points stay within ~1e-4 of that.
        assert!(worst < 1e-3 * (kernel_len as f64).sqrt(), "{input_len}/{kernel_len}: max error {worst}");
    }
}
