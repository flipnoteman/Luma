# Environment variables

Luma reads these when a context opens. `ContextOptions::from_env()` collects the ones that have
a [`ContextOptions`](https://docs.rs/luma/latest/luma/struct.ContextOptions.html) field, and the
builder methods override them:

| Variable | `ContextOptions` |
| --- | --- |
| `LUMA_DEVICE` | `device` / `.device(kind)` |
| `LUMA_MEMORY_BUDGET` | `memory_budget` / `.memory_budget(bytes)` |
| `LUMA_CHUNK_BYTES` | `chunk_bytes` / `.chunk_bytes(bytes)` |
| `LUMA_CHECK_FINITE` | `check_finite` / `.check_finite(on)` |
| `LUMA_CALIBRATE` | `calibrate` / `.calibrate(on)` |
| `LUMA_FAST_MATH` | `fast_math` / `.fast_math(on)` |
| `LUMA_DF_FMA` | `df_fma` / `.df_fma(fused)` |

The others (`LUMA_TUNING`, `LUMA_SUBGROUPS`, `LUMA_FUSION_*`, `LUMA_STAGING`, `LUMA_TRIDIAGONAL`,
`LUMA_SPECULATE`, `LUMA_CACHE*`, `LUMA_EXIT_WAIT`) are read from the environment only.

## All variables

| Variable | Effect |
| --- | --- |
| `LUMA_DEVICE` | `gpu` or `cpu`: restrict the device kind (default: GPU, CPU as fallback) |
| `LUMA_MEMORY_BUDGET` | Bytes of array data kept on the device before spilling to host memory (idle buffers kept for reuse take up to half of it) |
| `LUMA_CHUNK_BYTES` | Largest buffer an array is split into, in bytes (power of two) |
| `LUMA_TUNING` | `apple`, `nvidia`, `generic` or `cpu`: override the kernel tuning profile |
| `LUMA_SUBGROUPS` | `off` keeps the built-in kernels off subgroup operations |
| `LUMA_FUSION_MAX_NODES` | Most operations fused into one kernel (default 16384, at most 16384); an expression nests at most 512 levels |
| `LUMA_FUSION_MAX_LIVE` | Most values a fused kernel keeps live at once before it is split in two (default depends on the GPU profile, `8..=4096`) |
| `LUMA_FUSION_MAX_INLINED` | Most operations of one kernel after nested views of lazy expressions are unfolded into it; a bigger kernel evaluates the most repeated view first (default 1536 on NVIDIA, 1024 on other GPUs, 192 on CPU; `16..=16384`) |
| `LUMA_STAGING` | `off` keeps expressions read through slices and pads at several offsets (fluxes, stencils of an expression) out of workgroup memory: the kernel recomputes them per read instead of computing each once per cell of a workgroup tile (the cost model picks between them) |
| `LUMA_TRIDIAGONAL` | `thomas` or `cyclic`: force the algorithm of `solve_tridiagonal` / `solve_block_tridiagonal` (one invocation per line, or one workgroup per line by cyclic reduction); the default picks cyclic reduction for few lines |
| `LUMA_SPECULATE` | `off` stops readbacks in a loop (`step` then `.max().item()?` every iteration) from computing the next steps ahead |
| `LUMA_FAST_MATH` | `1`, `on` or `true`: allow rewrites that change values (`x + 0`, `x * 0`, `x - x`, `x / x`, constant reassociation, `x / c` as `x * (1 / c)`, `powf` with exponent 1, 2 or 0.5) and the cheaper `Df32` addition, whose error grows with the cancellation (below 2^-47 of the larger operand); exact rewrites (`x * 1`, `-(-x)`) always run |
| `LUMA_DF_FMA` | `off` makes `Df32` products split their factors (Dekker) instead of taking the error of a product from `fma`; `on` trusts that the device's `fma` is fused. Default: a one-time test kernel decides per device (stored with the calibration, see `LUMA_CACHE`); the results are the same either way |
| `LUMA_CHECK_FINITE` | `1`, `on` or `true`: fail any operation whose result holds a NaN or Inf (debug mode, waits for the GPU after every evaluated array) |
| `LUMA_CALIBRATE` | `0`, `off` or `false`: skip measuring the device's throughputs when a context opens (about 150 ms on a GPU, once per machine: a clean result is kept in the cache directory, see `LUMA_CACHE`; a measurement that looks contended by other processes is dropped) and use the tuning profile's fixed cost model |
| `LUMA_CACHE` | `0`, `off` or `false`: do not keep anything between processes. By default luma stores the driver's compiled pipelines (wgpu `PipelineCache`, Vulkan) and the measured cost model per adapter and driver, so a second run skips the multi-second compile of large (`Df32`) kernels and the calibration |
| `LUMA_CACHE_DIR` | Directory of that cache (default `$XDG_CACHE_HOME/luma`, else `~/.cache/luma`); files are written atomically, so processes may share it, and a damaged file is ignored |
| `LUMA_EXIT_WAIT` | `0` skips waiting at exit for NVIDIA's driver threads on Linux (saves about 50 ms; risks the driver's exit hang or abort). By default exit waits up to 2 s for the GPU and up to 2 s for busy driver compiler threads and then ends the process with `_exit` |

## Testing variables

These only affect the differential fuzzer in the crate's own test suite (`cargo test fuzz::`):

| Variable | Effect |
| --- | --- |
| `LUMA_FUZZ_CASES` | Number of random programs per fuzz test (default 256, 24 in debug builds) |
| `LUMA_FUZZ_SEED` | Fixed seed (default: random; failing cases are also kept in `proptest-regressions/`) |
| `LUMA_FUZZ_VERBOSE` | Set to anything: print every generated program |
