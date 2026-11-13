# Changelog

All notable changes to Luma are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Breaking

- Reductions (`sum`, `min`, `max`, `mean`, `norm`, `reduce`) are lazy and return a one-element
  `Array`; read it with `.item()?` (or `.to_array()?` for `reduce`). `[f32; 2]` / `[f32; 4]`
  reductions still return host values.
- A one-element array broadcasts against any shape (`&x / &x.sum()` stays on the GPU).

### Added

- Views that fuse into the reading kernel: `slice`, `shift_fill` / `shift_clamp` /
  `shift_mirror`, `pad` (`Edge`, `Mirror`, `Wrap`, `Zero`), `flip`, `transpose`, `index`, `take`,
  `concat`.
- Writes: `assign` through a view (lazy, in place) and `put` (scatter); `LumaError::ReadOnly`.
- Records: `Fields` trait and `#[derive(Fields)]` (crate `luma-derive`) for structs of arrays,
  with arithmetic, `dot`, views and `assign` in one kernel for all fields.
- Bitwise operations on `u32` arrays, fused like the rest: `& | ^ !`, `<< >>` (and `+=`-style
  assignments), `rotate_left`, `rotate_right`; amounts are taken modulo 32. `u32` states work in
  `Fields` records and `iterate_all` (register loop). Example `keccak`: Keccak-f[1600], SHA3-256
  and SHAKE128 over a batch of inputs in lazy array operations only.
- Math: `tanh`, `sign`, `floor`, `ceil`, `atan2`, `hypot`, `copysign`; `%` on `u32`.
- Precision: `Array<f64>` (devices with 64-bit shader floats) and `Df32` double-float (about 48
  bits, every device), casts between them, `sum_compensated` / `norm_compensated`.
- Loops: `iterate_until`, and `iterate_until_all` with a GPU-side stop rule
  (`Until::record` / `stop_when` / `check_every`), per-step history, step index and stop reason.
- Line primitives: `solve_tridiagonal` / `solve_block_tridiagonal` (blocks 1-8, `_from`
  variants), `scan`, `cumsum`, `sum_axis` / `min_axis` / `max_axis`. Few long lines use a
  cyclic-reduction solver on NVIDIA (`LUMA_TRIDIAGONAL=thomas|cyclic`), 3-22x faster.
- `item`, `to_array`, `sum_where`, `luma::Scalar` (uniform that changes without recompiling),
  `Array: Clone`, `cache()`, `fusion_limit`.
- Cost model: `CostModel` / `Context::cost_model`, calibrated per device
  (`ContextOptions::calibrate`, `LUMA_CALIBRATE`).
- Simplification of operations as they are built; value-changing rewrites and a cheaper `Df32`
  add only with `ContextOptions::fast_math` / `LUMA_FAST_MATH=1`.
- On-disk cache of compiled pipelines and calibration in `$XDG_CACHE_HOME/luma`
  (`LUMA_CACHE_DIR`, `LUMA_CACHE=off`); dropped above 32 MB.
- Diagnostics: `Profile::splits` with `SplitReason`, `check_finite` / `LUMA_CHECK_FINITE`
  (`LumaError::NonFinite`), `LumaError::Unsupported`.
- Settings: `LUMA_FUSION_MAX_NODES`, `LUMA_FUSION_MAX_LIVE`, `LUMA_STAGING`, `LUMA_SPECULATE`,
  `LUMA_DF_FMA`, `ContextOptions::df_fma`.
- Benchmarks `bench_pde`, `bench_build`, `bench_threads`; new `luma-bench` workloads.
- Example `mlkem`: ML-KEM (FIPS 203) KeyGen, Encaps and Decaps for ML-KEM-512, -768 and -1024 over
  batches of independent instances in lazy array operations only (no custom kernel), checked byte
  for byte against the RustCrypto `ml-kem` crate (implicit rejection included) and NIST ACVP
  vectors (test `mlkem`). RTX 3090, batches of 65536, keys and ciphertexts on the device: 2.4 / 2.1 /
  1.9 M ops/s (KeyGen / Encaps / Decaps) for ML-KEM-1024, 3.7 / 3.0 / 2.7 for -768 and 6.1 / 4.8 /
  4.2 for -512; all 32 CPU threads with the `ml-kem` crate do 0.15-0.52.

### Changed

- Fusion is decided by the cost model: an expression read at several indices is recomputed,
  staged in workgroup memory, or evaluated first (with its sibling fields), whichever is cheapest.
  The fusion limit is 16384 operations (nesting 512 levels); identical expressions are one node.
- Cuts count the operations that are still pending: after a cut, later operations no longer
  inherit the size of what was evaluated (a 10,000-operation chain made hundreds of kernels).
  An expression whose scalars outgrow one uniform buffer is cut (`SplitReason::Scalars`).
- Shifts and rotations by a constant are compiled with the amount as a literal.
- Region `assign` compiles to `if`/`else`, so ghost expressions cost only their own cells.
- A select on the coordinates is judged by the work it saves per cell that reaches it, not per
  cell of the kernel, so the chain of nested selects that `concat` of many parts makes computes
  every part only where it is taken: joining 168 arrays of 590 000 elements 13.8 to 1.0 ms.
- The pool of idle buffers keeps up to half of the memory budget (was an eighth, at most 1 GB):
  a program that cycles through gigabytes of temporaries every call reuses them instead of
  allocating again, which costs about 20 ms per GB on an RTX 3090 (ML-KEM-768 KeyGen of 65536
  instances 137 to 37 ms).
- `iterate*` trace the step once and replay its kernels; `eval` loops are recognized and batched;
  readbacks in convergence loops can be answered from steps computed ahead.
- Reductions fuse their elementwise input and short epilogues (`norm`: 2 launches); an operation
  on a reduction no longer forces it early.
- `Df32` arithmetic 1.5-2x faster (`fma` products where the device's `fma` is fused, one-step
  division and square root).
- Faster scans, axis reductions and tridiagonal solves; less host time to build and flatten
  expressions.
- Results on RTX 3090 (`bench_pde`): `euler/until_all` 2048² 72 to 17 ms, `muscl` 26-34% faster,
  Euler `iterate_all` 2048² 40.8 to 8.4 ms.

### Fixed

- Wrong results with threads sharing a context, and up to 10x slowdown from compile locks.
- `assign` / view / write ordering bugs (writes lost, stale views, loops corrupting arrays created
  before them), found by the new differential fuzzer.
- `Df32` kernels failing to bind their uniform; casts cancelled by the shader compiler.
- Input-buffer count wrong after `cache()`.
- NVIDIA driver hang or crash in `exit`: luma waits for driver-started threads and ends the
  process with `_exit` if one stays busy (`LUMA_EXIT_WAIT=0` skips the wait).

### Testing

- Differential fuzzer against a CPU oracle, invariant validation in debug builds, CI on lavapipe
  with a nightly fuzz job.

### Known limitations

- NVIDIA Linux driver 615.x can still crash or hang in its own shader-compiler thread at
  process exit; luma's exit wait and `_exit` fallback make it rare, not impossible.
- On NVIDIA, creating a device (`Context::new`) blocks submissions of other devices in the
  process until their in-flight GPU work finishes (once per device creation).
- RADV integrated GPUs give wrong results for kernels that use workgroup memory (scans,
  staging, cyclic tridiagonal solves); the cyclic solver is only chosen automatically on NVIDIA.
- `solve_tridiagonal` / `solve_block_tridiagonal` take `f32` arrays only.

## [0.2.0] - 2026-11-11

### Added

- Generator arrays: `Array::uniform`, `Array::normal`, `Array::random_bits` (Threefry-2x32-20, a
  counter-based generator: element `i` depends only on the seed and `i`), `Array::range` and
  `Array::linspace`, each with an `_on` variant. They are computed inside the kernels that read
  them, so a reduction over them reads no memory. Array-code Monte Carlo (pi, European call) over
  2^24 samples: 2.1x and 2.3x faster than ArrayFire CUDA on an RTX 3080 Laptop.
- `iterate_all`: a loop over several arrays (`[&zx, &zy, &count]`). When the step is elementwise,
  the whole loop runs as one kernel that keeps the state in registers, for any step count;
  anything else (shifts, arrays split into several buffers) runs step by step. Mandelbrot 1024²,
  256 iterations: 0.46 ms on an RTX 3080 Laptop, against 3.6 ms for the unevaluated chain.
- ArrayFire comparison benchmarks in `benchmarks/arrayfire` (separate workspace, not part of the crate).
- `select` accepts scalar branches; the result's shape and context come from the mask.
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

- **Breaking:** `Array::eval` and `eval_all` no longer wait for the GPU, like ArrayFire's `eval`:
  they record and hand the batch over, and `Context::sync`, readbacks and reductions wait. A GPU
  error raised by the batch shows up at the next waiting call. A loop that evaluates every step
  no longer pays a full round trip per step (about 35 µs on NVIDIA/Vulkan on Linux).
- `eval` and `eval_all` postpone the computation while the GPU is busy, until an `eval` finds it
  idle or something waits. Requested arrays dropped in the meantime are never computed, and the
  steps in between fuse. Mandelbrot evaluated every iteration (1024², 256 iterations): 25.6 to
  4.8 ms on an RTX 3080 Laptop, against 25.1 ms for ArrayFire CUDA.
- Work evaluated at the fusion limit is handed to the GPU right away, so a long chain built
  without `eval` runs while the host records the rest (mandelbrot 1024², 256 unevaluated
  iterations: 5.6 to 3.7 ms on an RTX 3080 Laptop).
- At the fusion limit, live arrays that share pending work with the biggest operand are evaluated
  together in multi-output kernels, instead of that operand alone.
- `eval` holds back up to 16 kernels while the GPU is still busy and hands them over with the next
  submitting or waiting call (one submission instead of one per call). Call `Context::flush`
  before long work that does not touch Luma.
- `Context::sync` and readbacks (`to_vec`, `read_into`, reductions) poll before blocking in the
  driver, for 1.5x the longest of the last four waits (at least 500 µs; waits predicted beyond 4 ms
  block as before), so a result that is about to arrive is not picked up late by a sleeping thread.
- Uniform values of a batch are copied by the batch's own command buffer (first 4 KiB) instead of a
  second one, saving about 2 µs per submission.
- Building and flattening lazy expressions is cheaper (about 40% less host time for a 50-operation
  expression).
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

- On Linux with the NVIDIA driver, a process could hang, abort or segfault inside `exit` when the
  driver's own exit handler ran while its background worker threads were still busy (seen after
  long runs; about 1 in 4 full `luma-bench` runs on driver 615.78). Luma now registers an exit
  handler that runs first and waits, at most about 4 s, for the GPU and those workers to go idle.
  It destroys nothing, so the driver's deadlock on destroying a device cannot occur. It adds
  about 60 ms to every exit on such systems; `LUMA_EXIT_WAIT=0` skips it.
- A fused expression whose operands were computed after it was built no longer exceeds the
  device's storage-buffer binding limit; its operands are computed first instead.
- Arrays uploaded with `from_slice` on unified-memory GPUs now count against the memory budget and
  can be spilled; before, only arrays computed on the GPU did. Newly resident data is no longer
  the first eviction candidate.
- Moving a segment to host memory reads it back through a staging buffer of at most 32 MiB instead
  of one as large as the segment, which could fail when the device was full.
- FFT passes no longer read and write past the work buffer when the folded grid launches more
  workgroups than there are tiles (more than 65535 tiles).
- Batches without uniforms no longer copy the 4 KiB uniform prefix.
- Chunked readbacks record one wait for the adaptive poll instead of one per piece, so short tail
  waits no longer shrink the poll budget.
- Freeing bind groups of reclaimed buffers ends the open compute pass first, so a later launch
  cannot reuse a stale binding.
- Uniform uploads no longer allocate and free a host-visible memory block on every submission
  (about 0.3 ms per submission on NVIDIA/Vulkan when no small upload buffer was alive).
- Workgroup grids larger than one row are folded evenly. Before, 65536 workgroups (for example 2^24
  elements) were launched as 65535 x 2 with one nearly empty row, costing about 10% on
  bandwidth-bound kernels of that size.
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
