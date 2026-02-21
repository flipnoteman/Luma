use luma::*;
use std::time::Instant;

async fn bench_spatial(input: &Array<f32>, kernel: &Array<f32>, iterations: u32) -> f64 {
    // Warmup
    let _ = input.conv1d(kernel, 1, 0).await.unwrap().to_vec().await.unwrap();

    let start = Instant::now();
    for _ in 0..iterations {
        let result = input.conv1d(kernel, 1, 0).await.unwrap();
        let _ = result.to_vec().await.unwrap();
    }
    let elapsed = start.elapsed();
    elapsed.as_secs_f64() / iterations as f64 * 1000.0 // ms per iteration
}

async fn bench_fft(input: &Array<f32>, kernel: &Array<f32>, iterations: u32) -> f64 {
    // Warmup
    let _ = input.conv1d_fft(kernel).await.unwrap().to_vec().await.unwrap();

    let start = Instant::now();
    for _ in 0..iterations {
        let result = input.conv1d_fft(kernel).await.unwrap();
        let _ = result.to_vec().await.unwrap();
    }
    let elapsed = start.elapsed();
    elapsed.as_secs_f64() / iterations as f64 * 1000.0 // ms per iteration
}

#[tokio::main]
async fn main() {
    let configs: Vec<(usize, usize, u32)> = vec![
        // (input_len, kernel_len, iterations)
        // Small inputs — spatial should dominate
        (256,      8,      100),
        (256,      64,     100),
        (1024,     16,     50),
        (1024,     256,    50),
        // Medium inputs
        (4096,     32,     20),
        (4096,     512,    20),
        (4096,     2048,   20),
        // Large inputs — FFT should start winning with big kernels
        (16384,    64,     10),
        (16384,    1024,   10),
        (16384,    4096,   10),
        (65536,    64,     5),
        (65536,    1024,   5),
        (65536,    8192,   5),
        (65536,    32768,  5),
        // Very large
        (262144,   256,    3),
        (262144,   4096,   3),
        (262144,   65536,  3),
    ];

    println!("Conv1D Benchmark: Spatial vs FFT");
    println!("Note: Spatial is O(n*k), FFT is O(n log n) — FFT wins for large kernels");
    println!();
    println!("{}", "=".repeat(80));
    println!(
        "{:<12} {:<12} {:<14} {:<14} {:<14}",
        "Input Len", "Kernel Len", "Spatial (ms)", "FFT (ms)", "Speedup"
    );
    println!("{}", "-".repeat(80));

    for (input_len, kernel_len, iterations) in &configs {
        let input_data: Vec<f32> = (0..*input_len).map(|x| (x as f32 * 0.01).sin()).collect();
        let kernel_data: Vec<f32> = (0..*kernel_len).map(|x| 1.0 / (1.0 + x as f32)).collect();

        let input = array!(&[*input_len, 1, 1, 1], &input_data);
        let kernel = array!(&[*kernel_len, 1, 1, 1], &kernel_data);

        let spatial_ms = bench_spatial(&input, &kernel, *iterations).await;
        let fft_ms = bench_fft(&input, &kernel, *iterations).await;

        let speedup = if spatial_ms < fft_ms {
            format!("Spatial {:.1}x", fft_ms / spatial_ms)
        } else {
            format!("FFT {:.1}x", spatial_ms / fft_ms)
        };

        println!(
            "{:<12} {:<12} {:<14.4} {:<14.4} {}",
            input_len, kernel_len, spatial_ms, fft_ms, speedup
        );
    }

    println!("{}", "=".repeat(80));
}
