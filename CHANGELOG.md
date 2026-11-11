# Changelog

All notable changes to Luma are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `Array::submit` and `submit_all`: evaluate arrays without waiting for the GPU.
- `select` accepts scalar branches; the result's shape and context come from the mask.

### Changed

- At the fusion limit, live arrays that share pending work with the biggest operand are evaluated
  together in multi-output kernels, instead of that operand alone.
- `Context::sync` polls briefly before blocking.

### Fixed

- Uniform uploads no longer allocate and free a host-visible memory block on every submission
  (about 0.3 ms per submission on NVIDIA/Vulkan when no small upload buffer was alive).

## [0.2.0] - 2026-11-10

### Added

- CPU fallback: when no GPU opens, Luma runs through Mesa llvmpipe (a software Vulkan driver that
  must be installed). `LUMA_DEVICE=cpu` or `Context::cpu()` force it, `LUMA_DEVICE=gpu` refuses it.
  New `DeviceKind`, `ContextOptions`, `Context::with_options`, `Context::is_cpu` and
  `TuningProfile::Cpu`.
- Arrays larger than one device buffer are split over several buffers automatically; every
  built-in operation (elementwise, fills, reductions, shifts, `iterate`, `conv1d`, `conv1d_fft`)
  runs over them in pieces.
- Array data beyond the device memory budget moves to host memory and back on demand, with a
  one-time warning. `ContextOptions::memory_budget` / `LUMA_MEMORY_BUDGET`,
  `ContextOptions::chunk_bytes` / `LUMA_CHUNK_BYTES`, `Context::spilled_bytes`.
- `Array::read_into`: read back into host memory you already own.
- Compound assignment (`x += ..`) fuses with the surrounding expression and writes in place.
- `luma::wgpu` re-export, so the wgpu types in `LumaError` and `Context::features` can be named.
- `LumaError::NoDevice` and `LumaError::Chunked`.
- CI workflow that builds and tests on the software driver.
- Typed `Array<T>` (`u32` and `f32`) with elementwise add, subtract, multiply and divide, scalar
  operations and operators (`+`, `-`, `*`, `/`), an `array!` macro, and several arrays alive at once.
- `conv1d` with stride and padding, and `conv1d_fft` (FFT-based convolution for any kernel length).
- Explicit `Context` that owns the device, queue, pipeline cache and a buffer pool; `LumaError`
  and `luma::Result` replace string errors, with validation of shapes, strides and cross-context use.
- Custom WGSL kernels: `ctx.kernel(wgsl).build()` reflects bindings and workgroup size with naga,
  and `args!` checks read / read-write / uniform arguments.
- Lazy evaluation: elementwise expressions are fused into one kernel (up to 128 operations,
  several outputs), with in-place reuse of buffers, shifts and `Array::iterate` (time-blocked
  stencils). Errors from lazy operations are reported when the result is read.
- Vector reductions (`sum`, `min`, `max`, `norm`, several at once with `reduce`), using subgroup
  operations where the device has them; `LUMA_SUBGROUPS` overrides.
- `flush`, profiling and `Context::features` (subgroup, f16 and cooperative-matrix support).
- Per-GPU tuning: Apple, NVIDIA and generic profiles chosen from the adapter, clamped to the
  device limits; `LUMA_TUNING` overrides.
- Zero-copy uploads on unified-memory devices; large uploads and readbacks use several threads.
- Criterion benchmarks of application workloads against multi-threaded CPU baselines, and the
  `bench_conv1d` example.
- MIT and Apache-2.0 licenses, a rewritten README and a custom-kernel example.

### Changed

- Builds on stable Rust (1.99); the nightly toolchain pin is gone.
- `LumaError`, `TuningProfile`, `DeviceKind` and `ContextOptions` are `#[non_exhaustive]`:
  matches need a wildcard arm, and `ContextOptions` is built with its chainable methods
  (`ContextOptions::default().chunk_bytes(4096)`).
- GPU devices are pooled per process and reused by later contexts instead of being destroyed.
- On GPUs whose buffer limit is below 4 GiB of contiguous binding, arrays between 2 GiB and 4 GiB
  now span two buffers; custom kernels on such arrays fail with `LumaError::Chunked`.
- The example `double_test` is now `basics`.
- Upgraded to wgpu 30; the adapter's limits are requested and errors are returned instead of
  panicking.
- The global async executor (and the `tokio`, `uuid`, `flume` and `once_cell` dependencies) is
  replaced by a synchronous `Context`; only reading data back waits for the GPU.
- WGSL shaders are embedded in the crate instead of being read from a directory at run time.

### Fixed

- Arrays larger than 65,535 workgroups of elementwise work are dispatched correctly, and the
  dimensions are passed to shaders in the layout they expect.
- Integer division by zero keeps its guard in fused kernels (`u32` `x / 0` gives `x`).
- Stencil probe aliasing, shift root flattening and upload / staging buffer reuse; uploads reclaim
  remapped buffers and retry on out-of-memory.
- `conv1d` handles inputs larger than 65,535 elements.
- `conv1d_fft` returned wrong results on devices that run fewer than 16 shader invocations in
  lockstep (a data race in the FFT's local pass). The fix makes `conv1d_fft` about 12% slower on
  an RTX 3090.

### Known limitations

- Custom WGSL kernels need arrays that fit one buffer.
- Convolution weights must fit one buffer; on chunked inputs `conv1d` kernels are limited to half
  a buffer and `conv1d_fft` kernels to a sixteenth.

## 0.1.3 - 2024-08-08

### Added

- `array!` macro for creating an `Array`.
- A global `Executor` that is created once and shared by all arrays.

### Changed

- Updated the README and dependencies.

## 0.1.0 - 2024-07-23

> **Warning:** 0.1.0, the version published on crates.io, was non-functional and must not be
> used. Use 0.2.0 or later.

### Added

- First crate with a `wgpu` dependency and an `Array` struct holding data and four dimensions.
- A WGSL compute shader and an `Executor` that doubles every value of an array (`double_test`).
- A library layout, with the shader loaded from a hard-coded relative path.

[Unreleased]: https://github.com/flipnoteman/Luma/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/flipnoteman/Luma/releases/tag/v0.2.0
