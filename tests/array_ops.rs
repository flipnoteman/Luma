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
