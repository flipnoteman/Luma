use luma::*;

#[tokio::main]
async fn main() {
    // A signal with a spike pattern [1, 3, 1] hidden in noise/flat regions.
    // The kernel matches that exact pattern — convolution output will peak
    // wherever the pattern appears.
    let input_data: Vec<f32> = vec![
        0.0, 0.0, 1.0, 3.0, 1.0, 0.0, 0.0, 0.0, 1.0, 3.0, 1.0, 0.0,
    ];
    let kernel_data = vec![1.0f32, 3.0, 1.0]; // the pattern we're looking for

    println!("Signal: {:?}", input_data);
    println!("Kernel: {:?}", kernel_data);
    println!();

    let input = array!(&[12, 1, 1, 1], &input_data);
    let kernel = array!(&[3, 1, 1, 1], &kernel_data);

    let result = input.conv1d(&kernel, 1, 0).await.unwrap().to_vec().await.unwrap();

    println!("Conv1D output ({} elements):", result.len());
    println!("{:?}", result);
    println!();

    // Find the peak — that's where the pattern matched
    let max_val = result.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    println!("Peak value: {}", max_val);
    println!("Match locations (output index → input index):");
    for (i, &v) in result.iter().enumerate() {
        if v == max_val {
            println!(
                "  output[{}] = {} ← pattern found at input[{}..{}]",
                i, v, i, i + 3
            );
        }
    }
}
