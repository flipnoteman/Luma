use luma::*;

// -- Creation & readback --

#[tokio::test]
async fn create_u32_array_and_readback() {
    let a = array!(&[3, 1, 1, 1], &[10u32, 20u32, 30u32]);
    assert_eq!(a.to_vec().await.unwrap(), vec![10u32, 20, 30]);
}

#[tokio::test]
async fn create_f32_array_and_readback() {
    let a = array!(&[3, 1, 1, 1], &[1.5f32, 2.5f32, 3.5f32]);
    assert_eq!(a.to_vec().await.unwrap(), vec![1.5f32, 2.5, 3.5]);
}

#[tokio::test]
async fn dimensions_returns_correct_value() {
    let a = array!(&[3, 1, 1, 1], &[1u32, 2u32, 3u32]);
    assert_eq!(a.dimensions(), &[3, 1, 1, 1]);
}

// -- Unary: double --

#[tokio::test]
async fn double_u32() {
    let a = array!(&[3, 1, 1, 1], &[1u32, 2u32, 3u32]);
    let result = a.double().await.unwrap().to_vec().await.unwrap();
    assert_eq!(result, vec![2u32, 4, 6]);
}

#[tokio::test]
async fn double_f32() {
    let a = array!(&[2, 1, 1, 1], &[1.5f32, 2.5f32]);
    let result = a.double().await.unwrap().to_vec().await.unwrap();
    assert_eq!(result, vec![3.0f32, 5.0]);
}

// -- Binary operations (u32) --

#[tokio::test]
async fn add_u32() {
    let a = array!(&[3, 1, 1, 1], &[1u32, 2u32, 3u32]);
    let b = array!(&[3, 1, 1, 1], &[4u32, 5u32, 6u32]);
    let result = a.add(&b).await.unwrap().to_vec().await.unwrap();
    assert_eq!(result, vec![5u32, 7, 9]);
}

#[tokio::test]
async fn subtract_u32() {
    let a = array!(&[3, 1, 1, 1], &[10u32, 20u32, 30u32]);
    let b = array!(&[3, 1, 1, 1], &[1u32, 2u32, 3u32]);
    let result = a.subtract(&b).await.unwrap().to_vec().await.unwrap();
    assert_eq!(result, vec![9u32, 18, 27]);
}

#[tokio::test]
async fn multiply_u32() {
    let a = array!(&[3, 1, 1, 1], &[2u32, 3u32, 4u32]);
    let b = array!(&[3, 1, 1, 1], &[5u32, 6u32, 7u32]);
    let result = a.multiply(&b).await.unwrap().to_vec().await.unwrap();
    assert_eq!(result, vec![10u32, 18, 28]);
}

#[tokio::test]
async fn divide_u32() {
    let a = array!(&[3, 1, 1, 1], &[10u32, 20u32, 30u32]);
    let b = array!(&[3, 1, 1, 1], &[2u32, 4u32, 5u32]);
    let result = a.divide(&b).await.unwrap().to_vec().await.unwrap();
    assert_eq!(result, vec![5u32, 5, 6]);
}

// -- Binary operations (f32) --

#[tokio::test]
async fn add_f32() {
    let a = array!(&[3, 1, 1, 1], &[1.0f32, 2.5f32, 3.0f32]);
    let b = array!(&[3, 1, 1, 1], &[4.0f32, 5.5f32, 6.0f32]);
    let result = a.add(&b).await.unwrap().to_vec().await.unwrap();
    assert_eq!(result, vec![5.0f32, 8.0, 9.0]);
}

// -- Scalar operations (u32) --

#[tokio::test]
async fn add_scalar_u32() {
    let a = array!(&[3, 1, 1, 1], &[1u32, 2u32, 3u32]);
    let result = a.add_scalar(10u32).await.unwrap().to_vec().await.unwrap();
    assert_eq!(result, vec![11u32, 12, 13]);
}

#[tokio::test]
async fn subtract_scalar_u32() {
    let a = array!(&[3, 1, 1, 1], &[5u32, 10u32, 15u32]);
    let result = a.subtract_scalar(1u32).await.unwrap().to_vec().await.unwrap();
    assert_eq!(result, vec![4u32, 9, 14]);
}

#[tokio::test]
async fn multiply_scalar_u32() {
    let a = array!(&[3, 1, 1, 1], &[2u32, 4u32, 6u32]);
    let result = a.multiply_scalar(3u32).await.unwrap().to_vec().await.unwrap();
    assert_eq!(result, vec![6u32, 12, 18]);
}

#[tokio::test]
async fn divide_scalar_u32() {
    let a = array!(&[3, 1, 1, 1], &[10u32, 20u32, 30u32]);
    let result = a.divide_scalar(2u32).await.unwrap().to_vec().await.unwrap();
    assert_eq!(result, vec![5u32, 10, 15]);
}

// -- Error case --

#[tokio::test]
async fn add_dimension_mismatch_returns_error() {
    let a = array!(&[3, 1, 1, 1], &[1u32, 2u32, 3u32]);
    let b = array!(&[2, 1, 1, 1], &[4u32, 5u32]);
    let err = a.add(&b).await.unwrap_err();
    assert!(err.contains("Dimension mismatch"), "Expected 'Dimension mismatch', got: {err}");
}

// -- Conv1D (f32 only) --

#[tokio::test]
async fn conv1d_basic() {
    let input = array!(&[5, 1, 1, 1], &[1.0f32, 2.0, 3.0, 4.0, 5.0]);
    let kernel = array!(&[3, 1, 1, 1], &[1.0f32, 0.0, -1.0]);
    let result = input.conv1d(&kernel, 1, 0).await.unwrap().to_vec().await.unwrap();
    assert_eq!(result, vec![-2.0f32, -2.0, -2.0]);
}

#[tokio::test]
async fn conv1d_with_padding() {
    let input = array!(&[3, 1, 1, 1], &[1.0f32, 2.0, 3.0]);
    let kernel = array!(&[2, 1, 1, 1], &[1.0f32, 1.0]);
    let result = input.conv1d(&kernel, 1, 1).await.unwrap().to_vec().await.unwrap();
    assert_eq!(result, vec![1.0f32, 3.0, 5.0, 3.0]);
}

#[tokio::test]
async fn conv1d_with_stride() {
    let input = array!(&[6, 1, 1, 1], &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let kernel = array!(&[2, 1, 1, 1], &[1.0f32, 1.0]);
    let result = input.conv1d(&kernel, 2, 0).await.unwrap().to_vec().await.unwrap();
    assert_eq!(result, vec![3.0f32, 7.0, 11.0]);
}

#[tokio::test]
async fn conv1d_identity_kernel() {
    let input = array!(&[3, 1, 1, 1], &[3.0f32, 7.0, 11.0]);
    let kernel = array!(&[1, 1, 1, 1], &[1.0f32]);
    let result = input.conv1d(&kernel, 1, 0).await.unwrap().to_vec().await.unwrap();
    assert_eq!(result, vec![3.0f32, 7.0, 11.0]);
}

#[tokio::test]
async fn conv1d_kernel_too_large_returns_error() {
    let input = array!(&[2, 1, 1, 1], &[1.0f32, 2.0]);
    let kernel = array!(&[5, 1, 1, 1], &[1.0f32, 1.0, 1.0, 1.0, 1.0]);
    let err = input.conv1d(&kernel, 1, 0).await.unwrap_err();
    assert!(err.contains("larger"), "Expected kernel-too-large error, got: {err}");
}

#[tokio::test]
async fn conv1d_large_input() {
    // 100-element input, 5-element averaging kernel, stride=1, pad=0 → 96 outputs
    let input_data: Vec<f32> = (1..=100).map(|x| x as f32).collect();
    let kernel_data = vec![0.2f32; 5]; // averaging kernel
    let input = array!(&[100, 1, 1, 1], &input_data);
    let kernel = array!(&[5, 1, 1, 1], &kernel_data);
    let result = input.conv1d(&kernel, 1, 0).await.unwrap().to_vec().await.unwrap();
    assert_eq!(result.len(), 96);
    // First output: mean(1,2,3,4,5) = 3.0
    assert!((result[0] - 3.0).abs() < 1e-5);
    // Last output: mean(96,97,98,99,100) = 98.0
    assert!((result[95] - 98.0).abs() < 1e-5);
}

#[tokio::test]
async fn conv1d_large_with_stride() {
    // 1000 elements, kernel size 3, stride=4, pad=0 → 250 outputs
    let input_data: Vec<f32> = (0..1000).map(|x| (x as f32).sin()).collect();
    let kernel = array!(&[3, 1, 1, 1], &[0.25f32, 0.5, 0.25]);
    let input = array!(&[1000, 1, 1, 1], &input_data);
    let result = input.conv1d(&kernel, 4, 0).await.unwrap().to_vec().await.unwrap();
    assert_eq!(result.len(), 250);
    // Spot-check first element: 0.25*in[0] + 0.5*in[1] + 0.25*in[2]
    let expected = 0.25 * (0f32).sin() + 0.5 * (1f32).sin() + 0.25 * (2f32).sin();
    assert!((result[0] - expected).abs() < 1e-5);
}

#[tokio::test]
async fn conv1d_large_with_padding() {
    // 512 elements, kernel size 7, stride=1, pad=3 → 512 outputs (same-size convolution)
    let input_data: Vec<f32> = (0..512).map(|x| (x % 10) as f32).collect();
    let kernel_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0]; // symmetric
    let input = array!(&[512, 1, 1, 1], &input_data);
    let kernel = array!(&[7, 1, 1, 1], &kernel_data);
    let result = input.conv1d(&kernel, 1, 3).await.unwrap().to_vec().await.unwrap();
    // pad=3, kernel=7 → output_len = (512 - 7 + 6)/1 + 1 = 512
    assert_eq!(result.len(), 512);
    // Check a middle element (idx=100) where no padding is involved
    let expected: f32 = (0..7).map(|k| input_data[100 - 3 + k] * kernel_data[k]).sum();
    assert!((result[100] - expected).abs() < 1e-4);
}

// -- FFT Conv1D (f32 only) --

#[tokio::test]
async fn fft_conv1d_basic() {
    // Full linear convolution: [1,2,3,4,5] * [1,0,-1] = [1, 2, 2, 2, 2, -4, -5]
    let input = array!(&[5, 1, 1, 1], &[1.0f32, 2.0, 3.0, 4.0, 5.0]);
    let kernel = array!(&[3, 1, 1, 1], &[1.0f32, 0.0, -1.0]);
    let result = input.conv1d_fft(&kernel).await.unwrap();
    assert_eq!(result.dimensions(), &[7, 1, 1, 1]);
    let data = result.to_vec().await.unwrap();
    let expected = vec![1.0f32, 2.0, 2.0, 2.0, 2.0, -4.0, -5.0];
    for (i, (a, b)) in data.iter().zip(expected.iter()).enumerate() {
        assert!((a - b).abs() < 1e-3, "Mismatch at index {}: got {}, expected {}", i, a, b);
    }
}

#[tokio::test]
async fn fft_conv1d_identity_kernel() {
    let input = array!(&[3, 1, 1, 1], &[3.0f32, 7.0, 11.0]);
    let kernel = array!(&[1, 1, 1, 1], &[1.0f32]);
    let result = input.conv1d_fft(&kernel).await.unwrap();
    assert_eq!(result.dimensions(), &[3, 1, 1, 1]);
    let data = result.to_vec().await.unwrap();
    let expected = vec![3.0f32, 7.0, 11.0];
    for (i, (a, b)) in data.iter().zip(expected.iter()).enumerate() {
        assert!((a - b).abs() < 1e-3, "Mismatch at index {}: got {}, expected {}", i, a, b);
    }
}

#[tokio::test]
async fn fft_conv1d_single_element() {
    let input = array!(&[1, 1, 1, 1], &[5.0f32]);
    let kernel = array!(&[1, 1, 1, 1], &[3.0f32]);
    let result = input.conv1d_fft(&kernel).await.unwrap();
    assert_eq!(result.dimensions(), &[1, 1, 1, 1]);
    let data = result.to_vec().await.unwrap();
    assert!((data[0] - 15.0).abs() < 1e-3, "Expected 15.0, got {}", data[0]);
}

#[tokio::test]
async fn fft_conv1d_matches_spatial() {
    // Compare FFT conv with spatial conv using a SYMMETRIC kernel.
    // Spatial conv1d computes cross-correlation: sum(input[i+k] * kernel[k])
    // FFT conv1d computes true convolution: sum(input[k] * kernel[n-k]) (kernel flipped)
    // With a symmetric kernel these are equivalent in the valid region.
    let input_data: Vec<f32> = (1..=10).map(|x| x as f32).collect();
    let kernel_data = vec![0.25f32, 0.5, 0.25]; // symmetric

    let input = array!(&[10, 1, 1, 1], &input_data);
    let kernel = array!(&[3, 1, 1, 1], &kernel_data);

    let spatial = input.conv1d(&kernel, 1, 0).await.unwrap().to_vec().await.unwrap();
    let fft = input.conv1d_fft(&kernel).await.unwrap().to_vec().await.unwrap();

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

#[tokio::test]
async fn fft_conv1d_large_kernel() {
    // 512-element input, 128-element kernel
    let input_data: Vec<f32> = (0..512).map(|x| (x as f32 * 0.01).sin()).collect();
    let kernel_data: Vec<f32> = (0..128).map(|x| 1.0 / (1.0 + x as f32)).collect();

    let input = array!(&[512, 1, 1, 1], &input_data);
    let kernel = array!(&[128, 1, 1, 1], &kernel_data);

    let result = input.conv1d_fft(&kernel).await.unwrap();
    assert_eq!(result.dimensions(), &[639, 1, 1, 1]); // 512 + 128 - 1
    let data = result.to_vec().await.unwrap();
    assert_eq!(data.len(), 639);

    // Spot-check first element: input[0] * kernel[0]
    let expected_first = input_data[0] * kernel_data[0];
    assert!(
        (data[0] - expected_first).abs() < 1e-3,
        "First element: got {}, expected {}", data[0], expected_first
    );
}

// -- Chained operations --

#[tokio::test]
async fn chained_operations_produce_correct_result() {
    let a = array!(&[3, 1, 1, 1], &[1u32, 2u32, 3u32]);
    let b = array!(&[3, 1, 1, 1], &[10u32, 20u32, 30u32]);

    let result = a.add(&b).await.unwrap()              // [11, 22, 33]
        .multiply(&a).await.unwrap()                   // [11, 44, 99]
        .add_scalar(1u32).await.unwrap()               // [12, 45, 100]
        .double().await.unwrap()                       // [24, 90, 200]
        .subtract(&b).await.unwrap()                   // [14, 70, 170]
        .divide_scalar(2u32).await.unwrap()            // [7, 35, 85]
        .to_vec().await.unwrap();

    assert_eq!(result, vec![7u32, 35, 85]);
}
