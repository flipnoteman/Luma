# Environment variables

Luma reads these when a context opens. `ContextOptions::from_env()` collects the ones that have
a [`ContextOptions`](https://docs.rs/luma/latest/luma/struct.ContextOptions.html) field, and the
builder methods override them:

| Variable | `ContextOptions` |
| --- | --- |
| `LUMA_BACKEND` | `backend` / `.backend(kind)` |
| `LUMA_DEVICE` | `device` / `.device(kind)` |
| `LUMA_MEMORY_BUDGET` | `memory_budget` / `.memory_budget(bytes)` |
| `LUMA_HOST_BUDGET` | `host_budget` / `.host_budget(Some(bytes))` |
| `LUMA_CHUNK_BYTES` | `chunk_bytes` / `.chunk_bytes(bytes)` |
| `LUMA_CHECK_FINITE` | `check_finite` / `.check_finite(on)` |
| `LUMA_CALIBRATE` | `calibrate` / `.calibrate(on)` |
| `LUMA_NUMERICS` | `numerics` / `.numerics(Numerics)` |

The others (`LUMA_CACHE*`, `LUMA_CUDA_DUMP`, `LUMA_WGPU_DUMP`, `LUMA_CPU_THREADS`, `LUMA_CPU_SPIN_US`, `LUMA_COMPILE_THREADS`,
`LUMA_EXIT_WAIT`, `LUMA_RECORD_DECISIONS`) are read from the environment only. The *Development switches* at the end of the table
(`LUMA_TUNING`, `LUMA_FUSION_*`, `LUMA_LINALG_*`, `LUMA_CHOL_*`, `LUMA_NATIVE_*`, `LUMA_SAMPLER_*` and the rest) are not part of the supported interface: they exist to bisect and
tune the library, may change meaning, and are removed before 0.6; programs should not depend on them. `LUMA_BENCH_*` belong to the
`luma-bench` crate (last section).

## All variables

| Variable | Effect |
| --- | --- |
| `LUMA_BACKEND` | `cuda`, `metal`, `wgpu`, `cpu` or `auto`: the backend (default `auto`: CUDA, then Metal, then wgpu, then the native `cpu` backend when no GPU opens) |
| `LUMA_DEVICE` | `gpu` or `cpu`: restrict the device kind (default: GPU, CPU as fallback; `cpu` is the native `cpu` backend, software Vulkan adapters are never used); also accepts a backend name (`cuda`, `metal`, `wgpu`) when `LUMA_BACKEND` is not set |
| `LUMA_MEMORY_BUDGET` | Bytes of array data kept on the device before spilling to host memory (idle buffers kept for reuse take up to half of it). Default: 80% of the device's memory where the backend reports it, never on CPU devices, otherwise unlimited |
| `LUMA_CHUNK_BYTES` | Largest buffer an array is split into, in bytes: a power of two from 1024 up to the largest bindable size, otherwise opening the context fails with `InvalidShape`; a value that is not a number is ignored |
| `LUMA_HOST_BUDGET` | Bytes of host memory for host copies of device data: the segments moved off the device when `LUMA_MEMORY_BUDGET` is exceeded plus the copies kept of segments brought back (a segment whose bytes the host still holds is evicted without a transfer). Default: half of the available host memory, at most half of the cgroup limit. Over budget, kept copies are dropped first (least recently used first); segments that exist only in host memory stay, with a warning once. `Context::paging_stats()` reports the use |
| `LUMA_RECORD_DECISIONS` | `1`, `on` or `true`: contexts start recording the planner's decisions (`Context::record_decisions(true)`; read them with `Context::take_decisions()`); nothing else about evaluation changes |
| `LUMA_NUMERICS` | `strict`, `default`, `algebraic` or `fast`, optionally followed by modifiers `+name` / `-name` (`tf32-matmul`, `tf32-linalg`, `fast-math`, `reassociate`, `reformulate`, `exact-finite`), e.g. `strict+tf32-linalg`; an unknown value panics in `ContextOptions::from_env`. `default` is TF32 `matmul` and nothing else; `strict` is exact `f32` everywhere (what `LUMA_TF32_MATMUL=0` was); `fast` adds `fast-math` (what `LUMA_FAST_MATH=1` was): allow rewrites that change values (`x + 0`, `x * 0`, `x - x`, `x / x`, constant reassociation and `x / c` as `x * (1 / c)` (both `f32` and `f64` only), `powf` with exponent 1, 2 or 0.5) and the cheaper `Df32` addition, whose error grows with the cancellation (below 2^-47 of the larger operand); exact rewrites (`x * 1`, `-(-x)`) always run |
| `LUMA_FAST_MATH`, `LUMA_TF32_MATMUL`, `LUMA_LINALG_TF32` | deprecated aliases of `LUMA_NUMERICS`, read by `ContextOptions::from_env` until 0.6 with a one-time warning: `LUMA_FAST_MATH=1` adds `fast-math`, `LUMA_TF32_MATMUL=0` removes `tf32-matmul`, `LUMA_LINALG_TF32=1` adds `tf32-linalg`; they apply over `LUMA_NUMERICS` |
| `LUMA_CHECK_FINITE` | `1`, `on` or `true`: fail any operation whose `f32` or `Df32` result holds a NaN or Inf (`f64` is not checked; debug mode, waits for the GPU after every evaluated array) |
| `LUMA_CALIBRATE` | `0`, `off` or `false`: skip measuring the device's throughputs when a context opens (about 150 ms on a GPU, once per machine: a clean result is kept in the cache directory, see `LUMA_CACHE`; a measurement that looks contended by other processes is dropped, and noted, so that processes use the profile's numbers for a day before measuring again) and use the tuning profile's fixed cost model |
| `LUMA_CACHE` | `0`, `off` or `false`: do not keep anything between processes. By default luma stores the driver's compiled pipelines (wgpu `PipelineCache`, Vulkan), the PTX that the native CUDA backend compiles with the driver's NVVM (keyed by NVVM version and device architecture), the machine code that the native `cpu` backend compiles (keyed by the host's instruction set) and the measured cost model per adapter and driver (the native Metal backend keeps no compiled pipelines between processes), so a second run skips the multi-second compile of large (`Df32`) kernels and the calibration |
| `LUMA_CUDA_DUMP` | a directory: the CUDA backend writes the NVVM IR (`.ll`), the PTX (`.ptx`) and the label (`.label`) of every kernel it builds or loads from the cache there, named like the cache entry. For inspecting generated code |
| `LUMA_CACHE_DIR` | Directory of that cache (default `$XDG_CACHE_HOME/luma` when that is an absolute path, else `%LOCALAPPDATA%\luma` on Windows, else `~/.cache/luma`); files are written atomically, so processes may share it, and a damaged file is ignored. A wgpu pipeline cache that grows past 32 MB is dropped and starts again |
| `LUMA_CPU_THREADS` | Number of worker threads of the native `cpu` backend (default: the machine's available parallelism; a value that is not a positive number is ignored) |
| `LUMA_COMPILE_THREADS` | Worker threads that compile luma's own kernels in the background on the `cuda` and `cpu` backends (default: the machine's available parallelism; `0` compiles on the calling thread). A kernel is waited for at its first launch, which then reports a compile error. Exit waits (up to 20 s) for the compiles in progress |
| `LUMA_CPU_SPIN_US` | Microseconds an idle worker of the native `cpu` backend polls for the next launch before it sleeps (default 200; `0` sleeps at once, which saves power and helps when the machine is oversubscribed, at the cost of a wake-up per launch). The worker on the other hardware thread of the core the launching thread runs on never polls (on Linux): it would slow the host work between launches |
| `LUMA_STRATEGY` | `family:strategy[,family:strategy...]` forces the named strategy of an operation family wherever it applies (today `reduce:host`, `merged`, `axis-merged`, `bits`, `single-pass`, `two-pass`, `axis`; `sort:host`, `radix_sweep`, `radix_ballots`, `radix_count_scan`, `lines`, `lines_radix`, `by_key_payload`, `by_key_permutation`, `by_key_axis`; `scan:rows`, `warp`, `lookback`, `chunk_single`, `chunk_three_pass`, `seq`, `fold_walk`, `by_key_single`, `by_key_chunked`; `scatter:host`, `plain`, `atomic_private`, `atomic_global`); an operation it does not apply to runs as it would without, and the profile's decisions (`ReducePlan`, `SortPlan`, `ScanPlan`, `ScatterPlan`) say so. An unknown family or strategy makes the context fail to open with `LumaError::InvalidOption`, and so does a family that only records its decision (`search`, `compact`, `merge`: "family not ported"). For investigations and tests: two strategies of the same bit class give the same bits |
| `LUMA_TELEMETRY` | `off` or a number `n` (default 32): one execution of a strategy in `n` is timed on the device (an event pair on CUDA, timestamp queries on wgpu, a command buffer of its own on Metal, the launch's own clock on the `cpu` backend), and the median of the last timings ranks strategies that give the same bits. `off` makes the context behave as if the feature did not exist. The sampled launch costs a few microseconds of host time |
| `LUMA_VARIANT_BUDGET` | Most kernels of non-baseline strategies one operation family builds per context (default 48): over it a new kernel is refused where a strategy of the same bit class can run instead (the wgpu pipeline cache is dropped whole above 32 MB) |
| `LUMA_EXIT_WAIT` | `0` skips waiting at exit for the driver's threads (the wgpu backend, on Linux; saves about 60 ms; risks the driver's exit hang or abort). By default exit waits up to 2 s for the GPU and up to 2 s for busy driver compiler threads (NVIDIA's Vulkan driver) and then ends the process with `_exit` |
| `LUMA_WGPU_DUMP` | a directory: the wgpu backend writes the WGSL of every kernel it builds from luma's kernel IR there, as `<label>-<hash>.wgsl`. For inspecting generated code (the counterpart of `LUMA_CUDA_DUMP`) |
| (`+tf32-linalg` of `LUMA_NUMERICS`) | the trailing-update products of the factorisations (`lu`, `qr`, `cholesky`, ...) run in TF32 on CUDA with compute capability 8.0 and up; the factors then carry a backward error of about 1e-3 of the updates' entries instead of 1e-7. Default: off (exact `f32` products); other backends ignore it. Replaces `LUMA_LINALG_TF32=1` |

## Development switches

Unsupported: for bisecting and tuning luma itself. Each is removed before 0.6, so do not depend on them in programs.

| Variable | Effect |
| --- | --- |
| `LUMA_TUNING` | `apple`, `nvidia`, `generic` or `cpu`: override the kernel tuning profile |
| `LUMA_SUBGROUPS` | `off` keeps the built-in kernels off subgroup operations |
| `LUMA_SINGLE_PASS_REDUCE` | `off` makes reductions over several workgroups two launches on CUDA again (by default the last workgroup of the first pass finishes the reduction); `on` does the same single-pass reduction on the native CPU backend (other backends ignore it: WGSL does not order memory across workgroups) |
| `LUMA_LOOKBACK_SCAN` | `off` makes the scans of integers (`u32`, `i32`) and the `min` / `max` scans of long contiguous lines two passes over their input on CUDA again (by default one kernel reads the input once and its workgroups wait for each other's prefix, a decoupled look-back, which gives the same results); `on` does the same single-pass scan on the native CPU backend (other backends ignore it: WGSL gives no forward progress or ordering of memory across workgroups). Sums and other floating-point scans keep two passes either way, so that they are reproducible run to run |
| `LUMA_TELEMETRY_MIN` | Samples of a strategy and shape class before their median counts as a measured cost (default 8) |
| `LUMA_EXPLORE` | `m`: one execution in `m` runs a rival strategy of the same bit class instead of the chosen one until it has `LUMA_TELEMETRY_MIN` samples (default off; it is on in benchmark runs that fill the timing cells) |
| `LUMA_FUSION_MAX_NODES` | Most operations fused into one kernel (default 16384, at most 16384); an expression nests at most 512 levels (256 in builds with debug assertions) |
| `LUMA_FUSION_MAX_INPUTS` | Lowers the most input buffers one fused kernel binds below what the device allows (the limit minus 3, at most 256), to behave like a device with fewer bindings; it cannot raise it. Arrays beyond it are packed into one buffer ([Performance](performance.md#many-arrays-in-one-kernel)) |
| `LUMA_FUSION_MAX_LIVE` | Most values a fused kernel keeps live at once before it is split in two (default depends on the GPU profile, `8..=4096`) |
| `LUMA_FUSION_MAX_INLINED` | Most operations of one kernel after nested views of lazy expressions are unfolded into it; a bigger kernel evaluates the most repeated view first (default 1536 on the NVIDIA tuning profile, 1024 on every other, the CPU one included; `16..=16384`) |
| `LUMA_COARSEN` | `off`, `auto` or `1`, `2`, `4`, `8`: how many elements each invocation of a fused elementwise kernel computes (thread coarsening: elements `i`, `i + T`, `i + 2T`, ... for `T` invocations in the launch, the loads of all of them issued before the first store, results bit-identical to one per invocation). `auto` (the default on NVIDIA GPUs, `off` elsewhere) coarsens launches of at least 16 waves of invocations whose kernel waits on long dependent chains: many `f64` operations (8 elements) or gathers (4); a number forces that many wherever the kernel is a plain store (not for `bool` results, staged, segmented or scatter kernels), whatever the size |
| `LUMA_STAGING` | `off` turns off workgroup staging of expressions read through slices and pads at several offsets (fluxes, stencils of an expression), where each is computed once per cell of a workgroup tile; the cost model still chooses between recomputing them per read and evaluating them first |
| `LUMA_TRIDIAGONAL` | `thomas`, `cyclic` or `auto`: the algorithm of `solve_tridiagonal` / `solve_block_tridiagonal` (one invocation per line, or one workgroup per line by cyclic reduction); the default (`auto`) picks cyclic reduction for few, long lines on the NVIDIA and Apple tuning profiles only (and always for stacked blocks wider than 5 there) |
| `LUMA_PLAN_CACHE` | `off`, `0`, `false` or `no` switches off the plan cache of eager evaluation: an `eval` whose pending graph has the structure of an earlier one (the loop `x = step(&x); x.eval()?`) normally runs that evaluation's recorded launches over the new buffers and scalars instead of planning it again; results are bit-identical either way. `trace` prints what the cache records and why it gives up on a structure |
| `LUMA_SPECULATE` | `off` stops readbacks in a loop (`step` then `.max().item()?` every iteration) from computing the next steps ahead |
| `LUMA_DF_FMA` | `0`, `off` or `false` makes `Df32` products split their factors (Dekker) instead of taking the error of a product from `fma`; `1`, `on` or `true` trusts that the device's `fma` is fused. Default: a one-time test kernel decides per device (stored with the calibration, see `LUMA_CACHE`); the results are the same either way |
| `LUMA_MULTI_REDUCE` | `off` (or `1`) keeps every pending reduction in a kernel of its own; a number sets the most accumulators one kernel computes when several reductions of the same length are forced together (clamped to what a kernel can hold; default 16) |
| `LUMA_MULTI_REDUCE_AFTER` | How many times a group of pending reductions is recorded one by one before the kernel that computes the group together is compiled in the background (default 4 on CUDA, 0 elsewhere: the group's kernel is compiled at once) |
| `LUMA_NATIVE_SOLVE` | `off` runs the line solves (`solve_tridiagonal` and relatives) of a backend that has native ones (the CPU backend) as compiled shaders, like the GPU backends |
| `LUMA_NATIVE_GEMM` | `off` runs the matrix products of a backend that has native ones (the CPU backend) as compiled tile kernels, like the GPU backends |
| `LUMA_OWN_GEMM` | `off` keeps the `f32` matrix products on the shader kernels even where the backend has its own (the PTX kernels of the CUDA backend) |
| `LUMA_HOST_FAST` | `off` never runs small operations on the host; `on` is only read by the crate's own tests. Default: operations on small arrays may run on the host on devices with zero-copy memory (never on CPU devices, while recording a loop, with `LUMA_CHECK_FINITE` or while profiling) |
| `LUMA_CHOL_NB` | `32`, `64`, `96` or `128`: rows of the diagonal blocks of the blocked Cholesky and triangular solves. Default: the largest of 128, 64 and 32 whose kernels fit in a workgroup's memory |
| `LUMA_CHOL_WIDE` | Columns of a blocked Cholesky that one outer step brings up to date with one product: a multiple of the block size (`LUMA_CHOL_NB`), at least the block size; default four blocks |
| `LUMA_TRSV` | `0` keeps triangular solves with up to 32 right-hand sides off the single-launch solve kernel (one matrix, on backends that run the look-back scan, see `LUMA_LOOKBACK_SCAN`); default on |
| `LUMA_TRSV_INV` | `1`: `solve_triangular` takes the diagonal blocks as explicit inverses (the solves of `solve` and `solve_spd` always do, from `LUMA_TRSV_INV_MIN` rows on) |
| `LUMA_TRSV_INV_MIN` | Rows from which inverted diagonal blocks are used (default 768) |
| `LUMA_TRSV_KC` | `1`, `2`, `4`, `8`, `16` or `32`: right-hand-side columns one workgroup of the triangular-solve kernel takes (default 8); more are solved by workgroups of their own |
| `LUMA_SVD_PRECOND` | `off` turns off the decomposition of a big matrix through the QR factorisation of its tall form; a number changes the smaller side from which it applies (default 1024) |
| `LUMA_SAMPLER_COARSEN` | `1`, `2` or `4`: elements per invocation of the sampling kernels (`approx`, `approx2`, ...; default 2) |
| `LUMA_SAMPLER_TILE` | `XxY` with `X * Y = 256` (for example `32x8`): the workgroup of the shaped sampling kernel, `X` along dimension 0 (default `32x8`) |
| `LUMA_LINALG_FENCE` | `1` / `0`: put a storage fence before the workgroup barriers of the LU and QR panel kernels (default: on everywhere but the CPU backend; `0` is the opt-out) |
| `LUMA_LINALG_PANEL` | `8`, `16` or `32` (default 32): columns of the panels the LU and QR of a big matrix are factored in; narrower panels measured slower on an RTX 3080 Laptop |
| `LUMA_QR_TSQR` | `0`: factor the panels (up to 32 columns) of a single QR with the global-memory kernel that one workgroup runs through the whole panel, instead of the two level tall-skinny QR (default: the tree where the device and the panel allow it) |

## Testing variables

These only affect the crate's own test suite: the first three the differential fuzzer (`cargo test fuzz::`), `LUMA_TEST_CACHE` the cache directory:

| Variable | Effect |
| --- | --- |
| `LUMA_FUZZ_CASES` | Number of random programs per fuzz test (default 256, 24 in debug builds) |
| `LUMA_FUZZ_SEED` | Fixed seed (default: random; failing cases are also kept in `tests/proptest-regressions/`) |
| `LUMA_FUZZ_VERBOSE` | Set to anything: print a line for each program that ends early on a documented limit |
| `LUMA_TEST_CACHE` | `1`: the process uses a cache directory under the system temporary directory (`luma-test-cache`) instead of the user's, unless `LUMA_CACHE` or `LUMA_CACHE_DIR` is set (unit tests always do) |
| `LUMA_EXIT_ORDER_CHILD` | Internal: marks the child process of the wgpu backend's exit-handler-order test |

## luma-bench variables

The `luma-bench` crate (`crates/luma-bench`) reads these:

| Variable | Effect |
| --- | --- |
| `LUMA_BENCH_REPS` | Timed repetitions per case (default: 3 for a case slower than a second, 5 with `--quick`, 21 for one under 10 ms, otherwise 9) |
| `LUMA_BENCH_KERNELS` | Set to anything: list the kernels of one call under each case (the `sparse` suite lists them under each of its rows) |
| `LUMA_BENCH_REF` | Text recorded as the `ref` of every JSON result line (`tools/benchref.sh` sets it to the commit) |
| `LUMA_BENCH_SHAPES` | Set (not on the CPU): the `matmul` suite runs its per-shape cases; `small` runs a shorter list |
| `LUMA_BENCH_TF32` | Set (not on the CPU): the `matmul` suite compares TF32 with `f32` products; `small` runs a shorter list |
| `LUMA_BENCH_DECISIONS` | A file path: write the planner's decisions of every case as JSONL, cold (plan cache emptied) and then warm (replay), plus the launches per kernel; compare two logs with `tools/decisiondiff OLD NEW`. The harness treats the GPU as idle while it records, so two runs of one binary agree |
