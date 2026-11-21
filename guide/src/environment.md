# Environment variables

Luma reads these when a context opens. `ContextOptions::from_env()` collects the ones that have
a [`ContextOptions`](https://docs.rs/luma/latest/luma/struct.ContextOptions.html) field, and the
builder methods override them:

| Variable | `ContextOptions` |
| --- | --- |
| `LUMA_BACKEND` | `backend` / `.backend(kind)` |
| `LUMA_DEVICE` | `device` / `.device(kind)` |
| `LUMA_MEMORY_BUDGET` | `memory_budget` / `.memory_budget(bytes)` |
| `LUMA_CHUNK_BYTES` | `chunk_bytes` / `.chunk_bytes(bytes)` |
| `LUMA_CHECK_FINITE` | `check_finite` / `.check_finite(on)` |
| `LUMA_CALIBRATE` | `calibrate` / `.calibrate(on)` |
| `LUMA_FAST_MATH` | `fast_math` / `.fast_math(on)` |
| `LUMA_DF_FMA` | `df_fma` / `.df_fma(fused)` |

The others (`LUMA_TUNING`, `LUMA_SUBGROUPS`, `LUMA_SINGLE_PASS_REDUCE`, `LUMA_LOOKBACK_SCAN`, `LUMA_FUSION_*`,
`LUMA_STAGING`, `LUMA_TRIDIAGONAL`, `LUMA_SPECULATE`, `LUMA_CACHE*`, `LUMA_CUDA_DUMP`, `LUMA_CPU_THREADS`, `LUMA_CPU_SPIN_US`, `LUMA_COMPILE_THREADS`, `LUMA_EXIT_WAIT`) are read from the environment only.

## All variables

| Variable | Effect |
| --- | --- |
| `LUMA_BACKEND` | `cuda`, `metal`, `wgpu`, `cpu` or `auto`: the backend (default `auto`: CUDA, then Metal, then wgpu, then the native `cpu` backend when no GPU opens) |
| `LUMA_DEVICE` | `gpu` or `cpu`: restrict the device kind (default: GPU, CPU as fallback; `cpu` is the native `cpu` backend, software Vulkan adapters are never used); also accepts a backend name (`cuda`, `metal`, `wgpu`) when `LUMA_BACKEND` is not set |
| `LUMA_MEMORY_BUDGET` | Bytes of array data kept on the device before spilling to host memory (idle buffers kept for reuse take up to half of it). Default: 80% of the device's memory where the backend reports it, never on CPU devices, otherwise unlimited |
| `LUMA_CHUNK_BYTES` | Largest buffer an array is split into, in bytes: a power of two from 1024 up to the largest bindable size, otherwise opening the context fails with `InvalidShape`; a value that is not a number is ignored |
| `LUMA_TUNING` | `apple`, `nvidia`, `generic` or `cpu`: override the kernel tuning profile |
| `LUMA_SUBGROUPS` | `off` keeps the built-in kernels off subgroup operations |
| `LUMA_SINGLE_PASS_REDUCE` | `off` makes reductions over several workgroups two launches on CUDA again (by default the last workgroup of the first pass finishes the reduction); `on` does the same single-pass reduction on the native CPU backend (other backends ignore it: WGSL does not order memory across workgroups) |
| `LUMA_LOOKBACK_SCAN` | `off` makes the scans of integers (`u32`, `i32`) and the `min` / `max` scans of long contiguous lines two passes over their input on CUDA again (by default one kernel reads the input once and its workgroups wait for each other's prefix, a decoupled look-back, which gives the same results); `on` does the same single-pass scan on the native CPU backend (other backends ignore it: WGSL gives no forward progress or ordering of memory across workgroups). Sums and other floating-point scans keep two passes either way, so that they are reproducible run to run |
| `LUMA_FUSION_MAX_NODES` | Most operations fused into one kernel (default 16384, at most 16384); an expression nests at most 512 levels (256 in builds with debug assertions) |
| `LUMA_FUSION_MAX_INPUTS` | Lowers the most input buffers one fused kernel binds below what the device allows (the limit minus 3, at most 256), to behave like a device with fewer bindings; it cannot raise it. Arrays beyond it are packed into one buffer ([Performance](performance.md#many-arrays-in-one-kernel)) |
| `LUMA_FUSION_MAX_LIVE` | Most values a fused kernel keeps live at once before it is split in two (default depends on the GPU profile, `8..=4096`) |
| `LUMA_FUSION_MAX_INLINED` | Most operations of one kernel after nested views of lazy expressions are unfolded into it; a bigger kernel evaluates the most repeated view first (default 1536 on the NVIDIA tuning profile, 1024 on every other, the CPU one included; `16..=16384`) |
| `LUMA_STAGING` | `off` turns off workgroup staging of expressions read through slices and pads at several offsets (fluxes, stencils of an expression), where each is computed once per cell of a workgroup tile; the cost model still chooses between recomputing them per read and evaluating them first |
| `LUMA_TRIDIAGONAL` | `thomas`, `cyclic` or `auto`: the algorithm of `solve_tridiagonal` / `solve_block_tridiagonal` (one invocation per line, or one workgroup per line by cyclic reduction); the default (`auto`) picks cyclic reduction for few, long lines on the NVIDIA and Apple tuning profiles only (and always for stacked blocks wider than 5 there) |
| `LUMA_SPECULATE` | `off` stops readbacks in a loop (`step` then `.max().item()?` every iteration) from computing the next steps ahead |
| `LUMA_FAST_MATH` | `1`, `on` or `true`: allow rewrites that change values (`x + 0`, `x * 0`, `x - x`, `x / x`, constant reassociation and `x / c` as `x * (1 / c)` (both `f32` and `f64` only), `powf` with exponent 1, 2 or 0.5) and the cheaper `Df32` addition, whose error grows with the cancellation (below 2^-47 of the larger operand); exact rewrites (`x * 1`, `-(-x)`) always run |
| `LUMA_DF_FMA` | `0`, `off` or `false` makes `Df32` products split their factors (Dekker) instead of taking the error of a product from `fma`; `1`, `on` or `true` trusts that the device's `fma` is fused. Default: a one-time test kernel decides per device (stored with the calibration, see `LUMA_CACHE`); the results are the same either way |
| `LUMA_CHECK_FINITE` | `1`, `on` or `true`: fail any operation whose `f32` or `Df32` result holds a NaN or Inf (`f64` is not checked; debug mode, waits for the GPU after every evaluated array) |
| `LUMA_CALIBRATE` | `0`, `off` or `false`: skip measuring the device's throughputs when a context opens (about 150 ms on a GPU, once per machine: a clean result is kept in the cache directory, see `LUMA_CACHE`; a measurement that looks contended by other processes is dropped, and noted, so that processes use the profile's numbers for a day before measuring again) and use the tuning profile's fixed cost model |
| `LUMA_CACHE` | `0`, `off` or `false`: do not keep anything between processes. By default luma stores the driver's compiled pipelines (wgpu `PipelineCache`, Vulkan), the PTX that the native CUDA backend compiles with the driver's NVVM (keyed by NVVM version and device architecture), the machine code that the native `cpu` backend compiles (keyed by the host's instruction set) and the measured cost model per adapter and driver (the native Metal backend keeps no compiled pipelines between processes), so a second run skips the multi-second compile of large (`Df32`) kernels and the calibration |
| `LUMA_CUDA_DUMP` | a directory: the CUDA backend writes the NVVM IR (`.ll`), the PTX (`.ptx`) and the label (`.label`) of every kernel it builds or loads from the cache there, named like the cache entry. For inspecting generated code |
| `LUMA_CACHE_DIR` | Directory of that cache (default `$XDG_CACHE_HOME/luma` when that is an absolute path, else `%LOCALAPPDATA%\luma` on Windows, else `~/.cache/luma`); files are written atomically, so processes may share it, and a damaged file is ignored. A wgpu pipeline cache that grows past 32 MB is dropped and starts again |
| `LUMA_CPU_THREADS` | Number of worker threads of the native `cpu` backend (default: the machine's available parallelism; a value that is not a positive number is ignored) |
| `LUMA_COMPILE_THREADS` | Worker threads that compile luma's own kernels in the background on the `cuda` and `cpu` backends (default: the machine's available parallelism; `0` compiles on the calling thread). A kernel is waited for at its first launch, which then reports a compile error. Exit waits (up to 20 s) for the compiles in progress |
| `LUMA_CPU_SPIN_US` | Microseconds an idle worker of the native `cpu` backend polls for the next launch before it sleeps (default 1000; `0` sleeps at once, which saves power and helps when the machine is oversubscribed, at the cost of a wake-up per launch) |
| `LUMA_EXIT_WAIT` | `0` skips waiting at exit for the driver's threads (the wgpu backend, on Linux; saves about 60 ms; risks the driver's exit hang or abort). By default exit waits up to 2 s for the GPU and up to 2 s for busy driver compiler threads (NVIDIA's Vulkan driver) and then ends the process with `_exit` |

## Testing variables

These only affect the crate's own test suite: the first three the differential fuzzer (`cargo test fuzz::`), the last the cache directory:

| Variable | Effect |
| --- | --- |
| `LUMA_FUZZ_CASES` | Number of random programs per fuzz test (default 256, 24 in debug builds) |
| `LUMA_FUZZ_SEED` | Fixed seed (default: random; failing cases are also kept in `tests/proptest-regressions/`) |
| `LUMA_FUZZ_VERBOSE` | Set to anything: print a line for each program that ends early on a documented limit |
| `LUMA_TEST_CACHE` | `1`: the process uses a cache directory under the system temporary directory (`luma-test-cache`) instead of the user's, unless `LUMA_CACHE` or `LUMA_CACHE_DIR` is set (unit tests always do) |
