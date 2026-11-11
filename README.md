<!-- TODO: logo / banner -->
<!-- <p align="center"><img src="assets/banner.png" alt="Luma" width="600"></p> -->

# Luma

Async GPU compute library for Rust. Write code with ordinary arithmetic operations and methods, and Luma handles execution 
with whatever hardware you have (Vulkan, Metal or DX12, through [wgpu](https://wgpu.rs)).

```rs
use luma::*;

fn main() -> Result<()> {
    // Arrays stored on GPU. The first one opens the default device.
    let x = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], 4)?;
    let y = Array::from_slice(&[10.0f32, 20.0, 30.0, 40.0], 4)?;

    // Operators and math methods are recorded.
    let z = (&x * 2.0 + &y).sqrt();

    // Execution deferred until host requires result.
    println!("{:?}", z.to_vec()?);          // [3.46.., 4.89.., 6.0, 6.92..]
    println!("{}", (&x * &y).sum()?);       // 300
    Ok(())
}
```

Similar to [ArrayFire](https://arrayfire.com), code is written with the familiarity of the ```Array``` interface. 
Execution of operations is deferred until the host requires the result. Chains of elementwise operations are fused
into a single kernel, and all intermediate work is batched on submission so the GPU doesn't waste time on many small, 
I/O dependant, operations. 

## Features

- Arrays of `f32` and `u32` with up to 4 dimensions
- `vec2`/`vec4` element types
- Built-in arithmetic operators, math functions, comparisons, `select`, `clamp`, circular `shift`
- Reductions: sum, min, max, mean, norm
- Lazy evaluation: elementwise chains compile into a single kernel
- Batched submission
- Multi-step stencils with `iterate`
- Ability to use custom WGSL kernels
- Subgroups, `f16` and cooperative matrices (tensor cores) on GPUs that support them
- Kernel tuning per GPU vendor (Apple and NVIDIA measured, generic defaults elsewhere)
- Useful error handling
- Per-kernel GPU timing with `ctx.profile(..)`

## Getting started

You need a recent stable Rust toolchain and a GPU with Vulkan, Metal or DX12 drivers.

Add Luma to a project:

```toml
[dependencies]
luma = "0.1"
```

Or build it from source:

```sh
git clone https://github.com/flipnoteman/Luma
cd Luma
cargo build --release
cargo test
```

### Examples

```sh
cargo run --example double_test                        # expressions, select, reductions
cargo run --example conv1d                             # 1D convolution
cargo run --example custom_kernel                      # custom WGSL kernel
cargo run --release --example bench_conv1d             # direct vs FFT convolution
cargo run --release --example bench_perf -- 67108864   # throughput on large arrays
cargo run --release --example bench_vs_cpu             # the same workloads on every CPU core
cargo run --release --example bench_batching           # sync after every op vs batched
cargo run --release --example bench_launch             # host cost per kernel launch
cargo run --release --example bench_fusion             # one kernel per op vs fused chains
```

### Benchmarks

Criterion benchmarks of real workloads. Each compares CPU and GPU execution (with Luma), and ensures that the results 
match before timing comparison takes place.

```sh
RUSTFLAGS="-C target-cpu=native" cargo bench --bench real_world    # Black-Scholes, Mandelbrot, n-body, SGEMM, 2D heat, FIR
RUSTFLAGS="-C target-cpu=native" cargo bench --bench ml            # k-means, MLP forward pass, similarity matrix
RUSTFLAGS="-C target-cpu=native" cargo bench --bench monte_carlo   # option pricing, integration, Ising model
RUSTFLAGS="-C target-cpu=native" cargo bench --bench imaging       # blurs, 2D convolution, bilateral filter, histogram
```

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in
Luma by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any
additional terms or conditions.
