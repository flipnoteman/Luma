# Luma

[**User Guide**](https://flipnoteman.github.io/Luma/) · [**API Reference (docs.rs)**](https://docs.rs/luma) · [**Environment Variables**](https://flipnoteman.github.io/Luma/v0.5.0/environment.html)

Async GPU compute library for Rust. Write code with ordinary arithmetic operations and methods, and Luma handles execution
with whatever hardware you have, through a native CUDA backend, a native Metal backend, [wgpu](https://wgpu.rs) (Vulkan, Metal, DX12), or the native SIMD CPU backend.

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

A table of data goes through the same API. Axis 0 varies fastest, so `[3, 4]` below is four rows of three
features:

```rs
use luma::*;

fn main() -> Result<()> {
    let x = Array::from_slice(&[1.0f32, 2.0, 0.0, 2.0, 1.0, 1.0, 0.0, 0.0, 3.0, 4.0, 2.0, 1.0], [3, 4])?;
    let gram = x.transpose().matmul(&x);              // XᵀX, [3, 3]; the transpose is read inside the kernel
    let variance = x.var_axis(0);                      // per row, in two passes

    let codes = Array::from_slice(&[0u32, 1, 0, 1], 4)?;
    let price = Array::from_slice(&[10.0f32, 20.0, 30.0, 50.0], 4)?;
    println!("{:?}", price.group_mean(&codes, 2).to_vec()?);   // [20.0, 35.0]
    println!("{}", price.median().item()?);                    // 25
    println!("{:?}", price.sort_descending().to_vec()?);       // [50.0, 30.0, 20.0, 10.0]
    println!("{:?} {:?}", gram.shape(), variance.len());
    Ok(())
}
```

Similar to [ArrayFire](https://arrayfire.com), code is written with the familiarity of the ```Array``` interface. 
Execution of operations is deferred until the host requires the result. Chains of elementwise operations are fused
into a single kernel, and all intermediate work is batched on submission so the GPU doesn't waste time on many small, 
I/O dependent, operations. 

The **[user guide](https://flipnoteman.github.io/Luma/)** explains the model (lazy evaluation and fusion, views, loops, precision) with runnable examples; the API reference is on [docs.rs](https://docs.rs/luma).

## Features

- Arrays of `f32`, `u32`, `i32`, `bool` (the result of comparisons, stored as one bit per element), `Df32` (double-float, about 48-bit) and `f64` (where supported) with up to 4 dimensions; casts between `f32`, `u32` and `i32`; bitwise operations and shifts on `u32` (see `examples/keccak.rs`)
- `vec2`/`vec4` element types and `luma::fields!` records of arrays
- Built-in arithmetic operators, math functions, comparisons, `select` (branches can be arrays or scalars), `clamp`, circular `shift`
- Views: `slice`, `pad`, `flip`, `transpose`, `swap_axes`, `permute_axes` (`ViewMut` `_mut` forms for writes), `take`, `put`, `concat`, `stack`, `broadcast_to`
- Reductions: sum, min, max, mean, norm, along axes (`reduce_axis` computes several in one pass, with `mean_axis`, `var_axis`, `std_axis`); `scan` / `cumsum`; `solve_tridiagonal`
- Data-science primitives: `matmul`; atomic scatters `put_add` / `put_min` / `put_max`, `bincount` and `histogram`; radix `sort`, `argsort`, `sort_axis`; `group_sum` / `group_mean` / `group_min` / `group_max` / `group_count`, `factorize` / `unique`; `quantile` and `median`; `searchsorted` / `digitize` / `isin`; NaN-aware reductions (`nansum`, `nanmean`, `nanmin`, `nanmax`); `round`, `trunc`, `log1p`, `expm1`. Pending operands are computed inside their kernels instead of being stored first
- Counter-based random numbers (`uniform`, `normal`, `random_bits`), `range` and `linspace`, computed inside the kernels that read them instead of stored
- Lazy evaluation: elementwise chains compile into a single kernel; compound assignment (`x += ..`) fuses too and overwrites the array's buffer in place
- Batched submission
- Non-blocking `eval` / `eval_all`: the GPU runs while the host records the next steps; reads and `sync` wait
- Readback into a new `Vec` (`to_vec`) or into host memory you already own (`read_into`)
- Multi-step stencils with `iterate`; elementwise loops over several arrays with `iterate_all`, compiled into one kernel that keeps the state in registers; `iterate_until` / `iterate_until_all` stop on a device-side condition (`compile_until_all` traces that loop once for solvers that run it in chunks)
- Ability to use custom WGSL kernels
- Subgroups and `f16` on GPUs that support them; cooperative matrices (tensor cores) on the wgpu backend
- Kernel tuning per GPU vendor (Apple and NVIDIA measured, generic defaults elsewhere)
- Useful error handling
- Per-kernel GPU timing with `ctx.profile(..)`
- Native SIMD CPU backend (Cranelift; `LUMA_CPU_THREADS` sets its thread count), used when no GPU is present
- Arrays larger than one device buffer or than device memory

## Getting started

You need a recent stable Rust toolchain and a GPU with a CUDA, Metal, Vulkan or DX12 driver.

Add Luma to a project:

```toml
[dependencies]
luma = "0.5"
```

The three GPU backends and the native CPU fallback (`cpu`) are enabled by default. To build with one GPU backend only (add `cpu` to keep the CPU fallback):

```sh
cargo build --no-default-features --features metal
cargo build --no-default-features --features cuda
cargo build --no-default-features --features wgpu
```

Or build it from source:

```sh
git clone https://github.com/flipnoteman/Luma
cd Luma
cargo build --release
cargo test
```

### Backends

Luma is a Cargo workspace of six crates: `luma-core` (the kernel IR, `luma_core::kir`, and the backend interface, `luma_core::hal`), `luma-wgpu`, `luma-metal` and `luma-cuda` (each holds one backend and its own code generator for WGSL, MSL or NVVM IR), `luma-cpu` (the native SIMD backend, compiling the kernel IR with Cranelift), and the root `luma` crate with the arrays and the JIT. The root crate's features `wgpu`, `metal`, `cuda` and `cpu` each enable one backend crate.

With `BackendKind::Auto` (the default) a context opens the first backend that works, in the order CUDA, Metal, wgpu, and the native CPU backend when no GPU opens. Choose one with `ContextOptions::backend(BackendKind::..)`, with `Context::cuda()`, `Context::metal()`, `Context::wgpu()` or `Context::cpu()`, or with `LUMA_BACKEND`; `Context::backend()` tells which one a context uses, and `Context::device()` (a `DeviceSummary`) or `Context::device_name()` the device: its backend, name and whether it is the host CPU. The CUDA backend needs only the NVIDIA driver, loaded at run time (kernels are compiled to PTX by the driver's own NVVM library; no CUDA toolkit or NVRTC): without an NVIDIA driver the CUDA backend reports an error (and `Auto` moves on) instead of failing to start the program.

Custom WGSL kernels (`Context::kernel`) run on every backend: native backends translate them through the kernel IR. `enable wgpu_cooperative_matrix;` works on the wgpu backend only. The native backends bind more buffers per kernel than WebGPU's limits allow (Metal through an argument buffer on GPUs with tier-2 argument buffers, CUDA through plain pointer arguments), and on Apple silicon uploads and readbacks use the unified memory the CPU and GPU share. On CUDA, Metal and the CPU backend, fixed iteration counts and blocks of `iterate_until_all` steps that Luma has traced are replayed from a cached command graph; wgpu has none.

### No GPU

Without a GPU, Luma runs on the native `cpu` backend (SIMD code on the host's threads; the `cpu` feature). It is picked automatically when no GPU opens. `LUMA_DEVICE=cpu` or `Context::cpu()` force it, and `LUMA_DEVICE=gpu` refuses it and returns an error instead. Software Vulkan adapters (Mesa llvmpipe, SwiftShader, WARP) are never used.

### Large arrays

An array bigger than one device buffer is split over several buffers automatically, and the built-in operations run over them
in pieces. Array data beyond the device memory budget (80% of the device's memory where the backend reports it; none on the CPU backend) moves to host memory and back
on demand, with a one-time warning. Custom kernels need arrays that fit one buffer.

### Environment variables

Common overrides: `LUMA_BACKEND` (`cuda`, `metal`, `wgpu`, `auto`), `LUMA_DEVICE` (`gpu`, `cpu`), `LUMA_MEMORY_BUDGET`, `LUMA_FAST_MATH=1`, `LUMA_CHECK_FINITE=1`, and `LUMA_CACHE=0`. See the [Environment variables reference](https://flipnoteman.github.io/Luma/v0.5.0/environment.html) in the user guide for the full list of tuning, fusion, solver, and cache variables.

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
cargo run --release --example bench_build              # host cost of building and flattening expressions
cargo run --release --example bench_pde                # views, assign/put, records, loops, scans, tridiagonal, Df32/f64
cargo run --release --example bench_solve              # tridiagonal and block-tridiagonal line solvers against a hand-written cyclic-reduction kernel
cargo run --release --example keccak                   # Keccak-f, SHA3-256 and SHAKE128 in lazy array operations
cargo run --release --example bench_datasci            # matmul, axis reductions, scatter, sort, group-by and searchsorted against hand-written WGSL kernels (--gate fails on a slower ratio)
cargo run --release --example mlkem                    # ML-KEM (FIPS 203) KeyGen / Encaps / Decaps over batches, checked against RustCrypto and NIST vectors
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

The `luma-cuda` crate embeds NVIDIA's libdevice math library
(`crates/luma-cuda/vendor/libdevice.10.bc`), which is not covered by the licences above: it is
redistributed under the NVIDIA CUDA Toolkit EULA
([`NVIDIA-CUDA-EULA.txt`](crates/luma-cuda/vendor/NVIDIA-CUDA-EULA.txt)).
