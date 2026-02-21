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
