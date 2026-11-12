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
    println!("{}", (&x * &y).sum().item()?);  // 300
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
- Built-in arithmetic operators, math functions, comparisons, `select` (branches can be arrays or scalars), `clamp`, circular `shift`
- Reductions: sum, min, max, mean, norm
- Counter-based random numbers (`uniform`, `normal`, `random_bits`), `range` and `linspace`, computed inside the kernels that read them instead of stored
- Lazy evaluation: elementwise chains compile into a single kernel; compound assignment (`x += ..`) fuses too and overwrites the array's buffer in place
- Batched submission
- Non-blocking `eval` / `eval_all`: the GPU runs while the host records the next steps; reads and `sync` wait
- Readback into a new `Vec` (`to_vec`) or into host memory you already own (`read_into`)
- Multi-step stencils with `iterate`; elementwise loops over several arrays with `iterate_all`, compiled into one kernel that keeps the state in registers
- Ability to use custom WGSL kernels
- Subgroups, `f16` and cooperative matrices (tensor cores) on GPUs that support them
- Kernel tuning per GPU vendor (Apple and NVIDIA measured, generic defaults elsewhere)
- Useful error handling
- Per-kernel GPU timing with `ctx.profile(..)`
- CPU fallback through Mesa llvmpipe when no GPU is present
- Arrays larger than one device buffer or than device memory

## Getting started

You need a recent stable Rust toolchain and a GPU with Vulkan, Metal or DX12 drivers.

Add Luma to a project:

```toml
[dependencies]
luma = "0.2"
```

Or build it from source:

```sh
git clone https://github.com/flipnoteman/Luma
cd Luma
cargo build --release
cargo test
```

### No GPU

Without a GPU, Luma runs through Mesa's llvmpipe, a software Vulkan driver. It is not built into Luma: the driver has to be
installed on the machine (`mesa-vulkan-drivers` on most Linux distributions), otherwise opening a context returns an error.
It is picked automatically when no GPU opens. `LUMA_DEVICE=cpu` or `Context::cpu()` force it, and `LUMA_DEVICE=gpu` refuses
the fallback and returns an error instead.

### Large arrays

An array bigger than one device buffer is split over several buffers automatically, and the built-in operations run over them
in pieces. Array data beyond the device memory budget (80% of device memory on Vulkan GPUs) moves to host memory and back
on demand, with a one-time warning. Custom kernels need arrays that fit one buffer.

### Environment variables

| Variable | Effect |
| --- | --- |
| `LUMA_DEVICE` | `gpu` or `cpu`: restrict the device kind (default: GPU, CPU as fallback) |
| `LUMA_MEMORY_BUDGET` | Bytes of array data kept on the device before spilling to host memory |
| `LUMA_CHUNK_BYTES` | Largest buffer an array is split into, in bytes (power of two) |
| `LUMA_TUNING` | `apple`, `nvidia`, `generic` or `cpu`: override the kernel tuning profile |
| `LUMA_SUBGROUPS` | `off` keeps the built-in kernels off subgroup operations |
| `LUMA_FUSION_MAX_NODES` | Most operations fused into one kernel (default 1024, at most 4096) |
| `LUMA_FUSION_MAX_LIVE` | Most values a fused kernel keeps live at once before it is split in two (default depends on the GPU profile, `8..=4096`) |
| `LUMA_FUSION_MAX_INLINED` | Most operations of one kernel after nested views of lazy expressions are unfolded into it; a bigger kernel evaluates the most repeated view first (default 1536 on NVIDIA, 1024 on other GPUs, 192 on CPU; `16..=16384`) |
| `LUMA_STAGING` | `off` keeps expressions read through slices and pads at several offsets (fluxes, stencils of an expression) out of workgroup memory: the kernel recomputes them per read instead of computing each once per cell of a workgroup tile (the cost model picks between them) |
| `LUMA_SPECULATE` | `off` stops readbacks in a loop (`step` then `.max().item()?` every iteration) from computing the next steps ahead |
| `LUMA_FAST_MATH` | `1`, `on` or `true`: allow rewrites that change values (`x + 0`, `x * 0`, `x - x`, `x / x`, constant reassociation, `x / c` as `x * (1 / c)`, `powf` with exponent 1, 2 or 0.5); exact rewrites (`x * 1`, `-(-x)`) always run |
| `LUMA_CHECK_FINITE` | `1`, `on` or `true`: fail any operation whose result holds a NaN or Inf (debug mode, waits for the GPU after every evaluated array) |
| `LUMA_CALIBRATE` | `0`, `off` or `false`: skip measuring the device's throughputs when a context opens (about 100 ms on a GPU, once per device and process) and use the tuning profile's fixed cost model |
| `LUMA_EXIT_WAIT` | `0` skips waiting at exit for NVIDIA's driver threads on Linux (saves about 50 ms; risks the driver's exit hang) |

### Examples

```sh
cargo run --example basics                             # expressions, select, reductions
cargo run --example conv1d                             # 1D convolution
cargo run --example custom_kernel                      # custom WGSL kernel
cargo run --release --example bench_conv1d             # direct vs FFT convolution
cargo run --release --example bench_perf -- 67108864   # throughput on large arrays
cargo run --release --example bench_vs_cpu             # the same workloads on every CPU core
cargo run --release --example bench_batching           # sync after every op vs batched
cargo run --release --example bench_launch             # host cost per kernel launch
cargo run --release --example bench_fusion             # one kernel per op vs fused chains
cargo run --release --example bench_threads            # threads on one context vs a context each
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

[`benchmarks/arrayfire`](benchmarks/arrayfire) compares Luma with ArrayFire (CUDA and OpenCL) on five workloads. It is a
separate Cargo workspace and needs an ArrayFire install:

```sh
cd benchmarks/arrayfire && AF_PATH=/path/to/arrayfire ./run.sh
```

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in
Luma by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any
additional terms or conditions.
