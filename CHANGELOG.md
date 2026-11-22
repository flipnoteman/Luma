# Changelog

All notable changes to Luma are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `Array::matmul_batched` / `matmul_batched()`: a batch of matrix products in one launch, `[K, M, B]` times `[N, K, B]` as `[N, M, B]` (the batch along axes 2 and 3; an operand without the batch, `[N, K]`, is shared by every product, like the weights of a layer). Products up to 128 on a side (`K` up to 1024) run in a kernel in which a workgroup computes the output tile of one product or of several small ones (4x4 outputs per invocation, 16-wide slices of `K` staged in workgroup memory, `src/bmm.rs`); bigger ones are one `matmul` each, as is a single product. On the CPU backend a workgroup is a whole product multiplied by `matrixmultiply`, spread over the cores (a few big products are left to `matmul`'s tiles). On an RTX 3090 4096 products of 32x32x32 take 0.07 ms (CUDA; 3.9 TFLOP/s, at the memory bandwidth), against 1.3 ms for the same batch as a broadcast multiply fused into a reduction (`(&a * &b).sum_axis(0)`, 200 GFLOP/s) and 0.87 ms for 256 of them as a loop of `matmul` (19 GFLOP/s). The `ml/batched` bench cases compare the three.
- luma-bench `cfd` suite: `bc/face_apply` (the `n_apply` pass of a block line solve: 5 x 5 face-matrix products over ghost-extended line ends, 50 coefficient streams and 5 writes per face, i / j / k) and `bc/ghost_region` (a padded copy of the primitives with five region assigns of a shared characteristic far-field expression, one kernel writing five padded fields), both verified against double precision host mirrors, on the bowshock sector grid. Measured with ncu on the RTX 3090 (bowshock `bench3` sector f32 diag): the 85-94 us `Fused[5](Generate(ParamRange ...))` kernels of that step are the `n_apply` passes, not ghost assigns; they read 51-69 MB and run at 830 GB/s (89 % of the DRAM peak), so there is no codegen cost to remove.
- `KernelBuilder::unchecked_bounds` (`unsafe`): builds a custom kernel without per-access bounds checks, for kernels whose author vouches that every access is in range (`luma_core::KernelTrust::UncheckedBounds`; the backend interface takes a `KernelTrust` where it took `trusted: bool`). On wgpu this is the only way to drop WebGPU's own checks.
- `DeviceInfo::compute_units`: the device's multiprocessors (CUDA `MULTIPROCESSOR_COUNT`, Vulkan `VK_NV_shader_sm_builtins` / `VK_AMD_shader_core_properties` on wgpu, worker threads on the CPU backend, unknown on Metal).
- `Array::quantiles(&[q])` and `quantiles_axis(axis, &[q])`: many quantiles of an `f32` array (or of every line along an axis) from one sort. A call of `quantile` / `quantile_axis` per quantile sorts every time, and the sort is nearly all of the cost. On an RTX 3090 (CUDA) seven quantiles of 16M elements take 1.3 ms with `quantiles` against 8.7 ms for seven `quantile` calls, and three quantiles per column of a 1M x 64 table 8.7 ms with `quantiles_axis` against 26.2 ms. Entry `j` along the axis is the quantile `qs[j]`; nothing is cached between calls (a sorted copy would cost as much memory as the array).
- `Array::sum_segments`, `min_segments` and `max_segments` (`u32`, `i32`, `f32`): the reduction of every contiguous segment of an array, the segments given by their offsets (a CSR row pointer, the boundaries of a sorted group-by), with `offsets.len() - 1` results and the identity for an empty segment. `(&val * x.take(&col)).sum_segments(&ptr)` is a CSR matrix-vector product in two launches, the first computing the pending operand inside the kernel (`src/segments.rs`): the entries are cut in tiles of 1024 that a workgroup reduces in workgroup memory with a segmented scan, whatever the lengths of the segments, so a row of a hundred thousand entries among rows of a handful costs no more than regular rows; the parts of the segments that cross a tile are combined by a second small pass in a fixed order, so the result is the same on every run and no atomics are used. RTX 3090: a sum over 10M `f32` rows in 1000 sorted groups takes 0.068 ms with `sum_segments` against 0.45 ms for `group_sum` by code (CUDA; wgpu 0.094 ms against 3.0 ms), 16 groups 0.068 against 0.89 ms (wgpu 0.12 against 7.1 ms), 1M groups 0.12 against 0.20 ms (`datasci/group/sum_sorted_seg_*`). A CSR matrix-vector product of 2M rows with power-law row lengths (23M entries) takes 1.29 ms (CUDA) and 1.69 ms (wgpu) against 1.40 and 1.73 for the best of the other four formulations in `sparse/spmv`, and 1.96 / 2.15 ms for the atomic `group_sum`; with uniform rows (32M entries) 1.76 ms against 1.80 for ELLPACK and 2.15 atomic: the gather and product alone, summed to one number (`spmv/uniform_floor`), take 1.31 ms on CUDA (2.30 on wgpu), so no way of summing by row can reach the 1.5x over ELLPACK that the 8 bytes per entry would suggest; power-law floor 0.94 ms (1.66 wgpu), banded 0.110 ms (0.150 wgpu), where ELLPACK (0.148 ms) still beats `sum_segments` (0.183 ms).

### Changed

- Graph construction of eager steps (`cfd/*_per_step`, new `cfd/euler|muscl/*_build_step` cases that record and drop a step without evaluating it): the hash-consing lookup is skipped while a node operand has no consumer yet (an equal node would be one), the masks of region assigns (`assign_lazy`) are remembered per box so the fields of a record share one mask instead of rebuilding it, `measure` has allocation-free paths for the one- and two-operand and the buffer-only cases, `create` takes the operation by reference and the cons table prunes at 256 entries. Instructions per recorded step drop 5-7 % (callgrind: `euler/3d` 2.02M to 1.88M, `muscl/3d` 2.40M to 2.28M, `euler/2d` 1.06M to 1.00M, `muscl/2d` 1.31M to 1.24M); on an RTX 3080 Laptop (CUDA, ABBA, pooled medians) `per_step` is 1.01-1.12x faster (euler 2d 1.11x, muscl 2d 1.12x, muscl 3d 1.03x, euler 3d 1.01x) and `build_step` up to 1.17x. No change beyond noise in `jit/launch`, `jit/loop`, `jit/trace` and the other `cfd` cases. The rest of the build cost is spread over atomics (Arc/Weak counts, reader locks), the 504-byte node allocation and the id-set bookkeeping, none above 6 %.
- Thread coarsening of fused elementwise kernels (`src/jit/coarsen.rs`, `LUMA_COARSEN=off|auto|<n>`): a plain store kernel can compute 2, 4 or 8 elements per invocation, `i`, `i + T`, `i + 2T`, ... for the `T` invocations of the launch (the lanes of a warp stay on neighbouring elements; contiguous chunks of elements per invocation were measured too and lost: 1.8x and 3.8x slower with 64 outputs at 2 and 4 elements, 1.2x and 2.2x slower with 24 inputs), with the loads of all the elements issued before the first store and the elements past the end reading the last one (their stores are skipped), so that results are bit-identical to one element per invocation. The planner (on by default on the NVIDIA profile, CUDA and wgpu; the other profiles and the CPU backend keep one element per invocation, where it was not measured) coarsens launches of at least 16 waves of invocations (`compute_units * 1536`) whose program waits on a chain: 8 or more `f64` operations (8 elements; `wide/chain16_f64` over 16M elements 3.90 to 0.93 ms on an RTX 3090 with CUDA, 3.81 to 0.93 ms with wgpu, 7.35 to 1.77 ms on an RTX 3080 Laptop: the double-precision pipe is fed by the elements of an invocation, not by one element's dependent chain) or 2 or more gathers (4 elements; `image/resize_up2x` 4096x4096 0.141 to 0.122 ms CUDA, 0.179 to 0.164 ms wgpu on the 3090, 0.296 to 0.27 ms on the laptop), and light programs (at most 40 `f32` operations per element) of 4 to 64 waves (2 elements; 1M elements: `ew/copy` -5 %, `fusion/stencil_shift2d` -11 % CUDA and -18 % wgpu, `fusion/select8` -4 %, `fusion/heavy_f32` -5 %; `ew/saxpy` +2.7 % is the one case measured slower), with at most as many copies as `max_live` allows for the live values of the program. Everything else, streaming kernels at the memory bandwidth included (`ew/*` at 16M to 64M elements: within 0.5 %), and the kernels that stage values in workgroup memory, run per segment, write `bool` bits or scatter, keep one element per invocation. Geometric mean over `jit/fusion`, `compute/{ew,wide,image,mc}` (64 cases, A B B A against the previous commit, RTX 3090): 0.974 CUDA, 0.979 wgpu; `cfd/{euler,muscl,heat}` and `ml/{dense,norm,train}` unchanged within the 5 to 15 % that a second run of the same binary varies by on their small kernels.
- Plan cache for eager evaluation (`LUMA_PLAN_CACHE=off` disables it): an `eval` whose pending graph has the structure of an earlier one (operations, lengths, wiring, views, which operands are the same buffer, which scalars are equal, the flags the planner reads, the cost model) and that is all the lazy work alive runs the store launches that evaluation recorded over the new buffers and scalar values, skipping flattening, cutting and judgement. Structures whose planning looked at other state (in-place buffer takeover, packing, failures, loops, reductions) are remembered as uncacheable and plan as before; the cache holds 128 plans per context and backs off after runs of misses. Results are bit-identical (randomized comparison tests over views, broadcasts, reductions, multi-output and cut programs). RTX 3090 CUDA, ms per call (median, 5 ABBA runs): `cfd/euler/3d_per_step` 6.60 -> 4.33 (1.53x), `cfd/muscl/3d_per_step` 3.49 -> 2.44 (1.43x), `cfd/euler/2d_per_step` 3.06 -> 2.42 (1.26x; wgpu 3.46 -> 2.59), `cfd/muscl/2d_per_step` 3.10 -> 2.77 (1.12x), `cfd/heat/2d_per_step` 1.08x; the graph building that remains (160-190 us of 200-240 us per 3-D step) bounds the gain. Nodes are also written field by field instead of copied from a stack temporary.
- Host cost of building and flattening lazy expressions, which bounds eager per-step loops (`cfd/*_per_step`: 100-370 us of host time per step against 5-20 us of GPU time): operations are passed to `create`, the simplifier and the hash-consing lookup by reference (an `Op` is 168 bytes, so every by-value use was a `memcpy`), a node is built in its allocation instead of on the stack, the fields the walks read (result, id, length, counters, operands) come first in `Node` and the fingerprint and type first in `Op` (a node visit touches fewer cache lines: 26% fewer L1 data read misses per step in a simulation of `cfd/euler/3d_per_step`), the recipe signature of a graph no longer copies every operation (the recipe's operations are compared with the graph's nodes instead), the over-limit cutting of `create` is a separate cold function, and the scans for segmented inputs read the instructions of a program only if an input has several segments. On an RTX 3090 `cfd/euler/3d_per_step` goes from 7.02 to 6.54 ms (20 steps), `euler/2d_per_step` from 3.17 to 2.96, `muscl/3d_per_step` from 3.77 to 3.46 and `muscl/2d_per_step` from 3.20 to 2.94 (best median of two, ABBA; the `cfd` suite's geometric mean is 0.983 of before, `jit` 0.990). The host path stays a few thousand instructions per node (creation and drop are about 70% of an eager step, graph walks 25%, kernel lookup and launch under 3%); a step of the same structure still builds its graph every time, which only `iterate_all` replay avoids.
- CPU backend: loads whose index is computed from lane ids and uniforms but not followed by the code generator (a clamped, wrapped or mirrored view coordinate; never an index read from memory, which rarely is consecutive) check at run time whether the four lanes are consecutive and in bounds and take one 128-bit load, instead of four bounds-checked lane loads; reads of a uniform's fields are one unchecked, shareable load marked uniform across the lanes (they are inside the padded storage the launch allocates), a function ending in its only `return` hands its value back with its lane knowledge instead of masking it, and kernels of up to 16000 IR instructions (was 2500) go through Cranelift's optimiser. `ml/conv/stencil5x5` (25 `shift_clamp` taps in one kernel) 4.91 to 0.82 ms, `ml/conv/depthwise3x3` 4.68 to 0.64 ms, `ml/conv/conv2d_im2col` 1.6-1.9x, `conv1d` 1.3-1.5x (32-thread CPU, best of ABBA).
- CPU backend: loads whose index is computed from lane ids and uniforms but not followed by the code generator (a clamped, wrapped or mirrored view coordinate; never an index read from memory, which rarely is consecutive) check at run time whether the four lanes are consecutive and in bounds and take one 128-bit load, instead of four bounds-checked lane loads; reads of a uniform's fields are one unchecked, shareable load marked uniform across the lanes (they are inside the padded storage the launch allocates), a function ending in its only `return` hands its value back with its lane knowledge instead of masking it, and kernels with such loads of up to 16000 IR instructions (was 2500) go through Cranelift's optimiser. `ml/conv/stencil5x5` (25 `shift_clamp` taps in one kernel) 4.91 to 0.82 ms, `ml/conv/depthwise3x3` 4.68 to 0.64 ms, `ml/conv/conv2d_im2col` 1.6-1.9x, `conv1d` 1.3-1.5x (32-thread CPU, best of ABBA).
- CPU backend: `exp`, `ln`, `exp2`, `log2`, `sin`, `cos` and `tanh` of `f32` run as one call per 4-lane packet (Cephes-style routines in SSE2, `src/vecmath.rs`: range reduction, a minimax polynomial, the exponent assembled in the integer unit; at most 2 ulp over the whole `f32` range, 1.9 measured, with `sin` and `cos` taking the scalar `libm` for a packet with a lane beyond 8192) instead of four indirect calls of the scalar `libm` routines with a lane extract and insert around each; other functions and `f64` are unchanged, and other targets than x86-64 on Unix keep the lanewise calls. On a 32-thread Ryzen 9 7950X `jit/fusion/chain512` (1M elements, 512 unary operations, two sevenths `sin` / `tanh`) goes from 35.4 to 26.2 ms, `chain256` from 17.8 to 13.7, `chain128` from 10.2 to 7.9, `fusion/dag` from 7.8 to 6.4, `compute rng/normal` (4M) from 6.0 to 3.6 ms, `ew/sin16` from 44.7 to 33.2, `ew/exp_ln_tanh` from 11.6 to 8.4, `mc/black_scholes_call` (4M options) from 4.1 to 3.1, and the softmax and cross-entropy cases of the ML suite by 5-13 %.
- `luma-bench` keeps calling a case until it has run for 250 ms before timing it, so GPUs that idled between cases are at working clocks: an NVIDIA GPU drops to 210 MHz after ~50 ms idle, which made wgpu's sub-millisecond `jit/fusion` kernels (12 us) measure 8-15x slower than CUDA's, and an RTX 3080 Laptop starting a case in P8 ran a 1 ms inference step in 12 ms.
- `luma-bench` `jit/trace/fresh_until` builds recipes of 6 steps (10^6 kernel structures; were 3, so 1000): within a run and across runs (compiled kernels are cached on disk, and a new CUDA emitter revision empties that cache) the "fresh" kernels met structures built before and the case timed the state of the disk cache, which read as a 1.5-2x regression of the loop code (43.9 to 65.5 ms on an RTX 3090, 2.45x on an RTX 3080 Laptop, when the byte-form `getelementptr` bumped the revision). With a cold cache the 3-step case is 23-25 ms on every revision from 5b798b8 to 71e80d8; the 6-step case takes 160 ms on both.
- Axis reductions of `f32` (`sum_axis`, `min_axis`, `max_axis`, `reduce_axis`, `mean_axis`, `var_axis`, `std_axis`) on a GPU backend stay pending until one of their results is used, and the ones pending over the same lines are then recorded in one kernel (up to 8 ops; `LUMA_MULTI_REDUCE=off` keeps each on its own; the CPU backend leaves only reductions of an expression of at least 12 f32 operations pending, see Fixed): the three component sums of an all-pairs force, a layer norm's `mean_axis` and `var_axis`, a table's column means, extrema and standard deviations. The kernel's first pass computes the distinct operands together, once what they share, and every op combines its line in the order a reduction alone does, so each result is the one it has alone: the same bits on the CPU backend (tested against a context that records them alone), within an ulp or two on CUDA and wgpu, where a kernel that holds several uses of a product rounds its multiply-adds differently. Reductions made while a step is recorded or the profile is on, `u32`/`i32` ones and `argmax_axis` still run when called.
- The plan constants measured on an RTX 3090 (82 multiprocessors) scale with the device's multiprocessor count and are unchanged at 82: the twisted Thomas solve's line limit (`TWISTED_LINES`) and the line count its workgroup is sized for (4096-line `tri/solve_axis1` on an RTX 3080 Laptop: 3.98 to 2.43 ms), and, upward only, the workgroups of a K-split matmul (`SPLIT_GROUPS`, `SPLIT_TILES`). Backends that cannot report the count (Metal, wgpu on other vendors) and the CPU backend keep the 3090 values.
- Bounds checks of untrusted kernels skip the accesses they can prove in range: indices built from `local_invocation_id`, `local_invocation_index`, literals, wrap-free arithmetic and counted `for` loops (`for (var k = 0u; k < 16u; k += 2u)`) into workgroup and function arrays (a tiled SGEMM's inner loop no longer pays a select per workgroup-memory read).
- `Array<f32>::var_axis` (and `std_axis`) is one pass instead of two: the sum and sum of squares of `self - K` (K the first element of each line, subtracted inside the reduction) come from one `reduce_axis`, and the variance is `(S2 - S1*S1/n)/n` clamped at 0 (NaN kept), accurate for data far from zero. `ml/norm/layernorm` (mean + var + normalise) takes 3 launches instead of 4 (768x8192 on an RTX 3090: 0.173 ms to 0.143 ms, 4096x1024 0.129 to 0.104, 64x65536 0.163 to 0.124).
- CUDA: the NVVM IR luma emits is about a third of its former size (reductions 65 to 28 KB, tiled matrix products 199 to 129 KB, elementwise kernels 10-20 to 5-7 KB), which shortens the first build of every kernel (cold NVVM time 5-25 % lower, kernel runtimes unchanged within measurement noise). Loads and stores are typed `getelementptr`s of the buffer instead of byte offsets with casts, scalar `let`s are SSA values instead of stack slots, constants are folded and unused instructions dropped before NVVM sees the text, thread ids of size-1 dimensions are constants and libdevice is only linked for kernels that call it. Subgroup operations that provably run with every lane of their warp present (whole-warp workgroups, warp-uniform control flow around them; the reductions' two-stage trees qualify) take the full-warp shuffle path without the `activemask` fallback, which makes their PTX about 25 % smaller; others keep the fallback. The PTX cache revision is bumped, so the first run after the upgrade recompiles once.
- CUDA: a captured graph is no longer one chain. Its kernel nodes get dependency edges from the buffers each launch reads and writes (read after write, write after write, write after read, the buffer of an indirect launch's workgroup counts counted as read; buffers told apart by device address, so a pool buffer handed out again stays ordered), with edges implied by others dropped, and launches that share nothing run concurrently. Results are bit-identical. A graph that is mostly one chain (fewer than 1.5 launches per launch of its longest chain) is still captured as a chain; so is any capture that fails with edges. On an RTX 3090 a step of 16 independent small kernels replayed 200 times goes from 3.6 ms to 1.55 ms (`jit/trace/wide16_replay`), `cfd/converge/points_compiled` and `points_chunks_compiled` from 5.97 / 7.45 ms to 5.57 / 6.96 ms; chains and large grids are unchanged.
- `iterate_all`, `iterate_until_all` and `compile_until_all`: a state the step hands back unchanged (itself, or `&s * 1.0` that has not run) takes no launch per step. The stepper binds it to one buffer at every step, copies the initial state there once per run and into the other parity's buffer once at the end, instead of one copy launch per step; a step that returned a state itself used to fall back to tracing every step. `jit/trace/step30_pass6_replay` (30 stages and six unchanged states) goes from 6.87 to 6.67 ms on an RTX 3090.
- CPU backend: the line solves (`solve_tridiagonal*`, `solve_block_tridiagonal*`, on stacked blocks and on parts, blocks of 1 to 8) run as native multithreaded Rust instead of compiled shader code: the block Thomas algorithm (same pivoting and arithmetic order as the shader) with the swept blocks of a line kept in a per-thread buffer, and four neighbouring lines (the lines of a strided axis, or four lines of the contiguous axis gathered) eliminated together in vector lanes. The planner never picks cyclic reduction on its own there, and blocks above 5 no longer take the slow loop form (`bench_solve`, 32 threads: 5x5 blocks of 400x200 cells 13.3 / 7.2 ms along the two axes to 0.5 ms, 3x3 5.1 / 2.1 ms to 0.15 ms, 6x6 318 / 220 ms to 1.1 ms; parts of 5x5 blocks on a 3-D grid 12.7 to 25 ms to 2.4 to 3 ms). Parts that are lazy expressions are computed together first (one extra launch) instead of being fused into the solve. `LUMA_NATIVE_SOLVE=off` restores the shader solves. New backend hooks in `luma-hal-core`: `Backend::native_line_solve`, `Backend::compile_line_solve` and `LineSolveSpec`.
- Pending reductions that are forced together are recorded in one kernel. When a reduction is recorded (its value is read, used on the GPU or evaluated with `eval_all`), the other live, unrecorded `f32`, `u32` and `i32` reductions of the same length that read what it reads (or are small) are taken along, up to 16 accumulators per kernel (`LUMA_MULTI_REDUCE=off|n`): one program computes their operands together, so what they share (the gather behind a dozen masked sums, the columns of a `describe`) is computed and read once, and the monitor of a solver that makes dozens of `sum_where` reductions launches a handful of kernels instead of one each. Every reduction keeps its own combine order, so its value is the bits it has when recorded alone; reductions that wait for another (a variance on a mean) run after it, and reductions made while a step is being recorded, or by other threads, are never taken along.
- CPU backend: matrix products (`matmul`) run as native multithreaded Rust (`matrixmultiply`: packed, blocked GEMM with AVX-512 / AVX2+FMA / NEON microkernels picked at run time) instead of the compiled GPU tile kernels: the pool takes output tiles of 128, 64 or 32 (enough of them for two per core) and long products over few tiles are split over K and summed. Lazy operands are computed first, together; a transposed operand is read through its strides instead of being copied. A 4096-cube product goes from 64 to about 1100 GFLOP/s on a 32-thread Ryzen 9 7950X, the dense layers of the ML suite from 30-60 to 600-1100 GFLOP/s, the MLP training steps (batch 1024, 1024-wide) from 183 ms to 12-17 ms and the logistic regression steps from 250 ms to 8-10 ms. The results agree with the shader's up to the order of the sums. `LUMA_NATIVE_GEMM=off` runs the tile kernels as on the GPU backends. New hooks `Backend::native_matmul` / `compile_matmul` and `MatmulSpec` (`luma-hal-core`); the GPU backends are unchanged.
- CUDA and CPU: the kernels luma builds itself compile on a pool of worker threads (`LUMA_COMPILE_THREADS`, default the available parallelism, `0` compiles on the calling thread as before) instead of one after another on the caller's thread. A kernel is waited for where it is first launched or captured, and a compile error surfaces there with the same message; custom kernels (`Context::kernel(..).build()`) still compile, and fail, in `build`. The kernels of a traced step therefore compile side by side while the step is still being recorded, which shortens the first run after an upgrade or with a cold kernel cache. libNVVM is no longer serialized by a lock. Kernel disk caches are unchanged.
- `factorize`/`unique` of `u32`/`i32` keys whose range is small (at most the key count and 2^24, from 2^18 keys) count into a presence table instead of radix sorting: one min/max readback, a scan and a gather. Same results; wide ranges and recordings keep the sort.
- `matmul`: products with a small output and a long `K` (a Gram matrix `ZᵀZ`, one-hot labels times features, a gradient `Xᵀr`, a covariance, a matrix times a few vectors) run as a new kernel family instead of the 32-wide staged tiles split over `K`: each invocation owns up to 10x8 outputs in registers and loads both operands straight from global memory, with no workgroup staging and no barriers in the loop, and the invocations the small output does not need take other `k` (reduced pairwise through workgroup memory, in a fixed order); the slices are then summed by the existing partial-sum pass. A matrix stored along `K` times up to 8 columns reads along `K` with a few rows per invocation. A pending operand that many tiles would recompute at every load (the standardised matrix of a Gram product) is evaluated first. On an RTX 3090 (CUDA): `onehot(y)ᵀ X` over 1M x 32 takes 0.19 ms instead of 0.51, `Xᵀ r` over 1M x 64 0.33 ms instead of 1.04, the Gram matrix of 16 features over 1M samples 0.17 ms instead of 0.59 and of 64 features 1.7 ms instead of 2.3, a 2048x2048 matrix times a vector 0.03 ms instead of 0.19.
- CPU backend: a launch no longer wakes every worker. The pool learns how long each kernel takes (the time its threads spend in workgroups); a launch too small to pay for a hand-off runs inline on the calling thread, a larger one enlists only as many helpers as it has work for. Idle workers poll for up to 200 microseconds before they sleep (`LUMA_CPU_SPIN_US`, `0` to sleep at once), so back-to-back launches skip the futex round trip, and a launch is claimed with atomics instead of a global mutex and condvar.
- `sort_axis` (and so `quantile_axis`, `median_axis`) on lines of up to 2048 elements (256 on the CPU backend), along any axis, sorts them in workgroup memory in one kernel that reads and writes every element once (a bitonic sort of tiles of whole lines; along an axis other than 0 a tile takes neighbouring lines so that the loads stay coalesced). RTX 3090, CUDA: a 1M x 64 table along axis 1 (1M lines of 64) 17.4 -> 1.4 ms, a 64 x 1M table along axis 0 12.6 -> 1.4 ms, `[16, 262144]` 0.96 -> 0.06 ms, `[1024, 4096]` 0.83 -> 0.21 ms (wgpu alike). Longer lines still use the radix sort of the whole array, but carry the values through the regrouping by line instead of gathering them afterwards and writing them back with a strided scatter; along an axis other than 0 the lines return to their layout with the tiled transpose. The 1M x 64 table along axis 0 (lines of 1M) takes 8.7 ms instead of 9.6 on CUDA and holds one more array of the table's size while it sorts. `sort_axis` returns NaNs as the positive quiet NaN, as `sort` does (it used to keep their bits).
- Host cost of building lazy expressions: a large unrolled expression (24 Keccak rounds written out, ML-KEM) spends its host time in `create`, one node per operation. The id set every node keeps of the expression below it is merged with POPCNT/AVX2 word loops (picked at run time; the generic loops are the fallback), built in one allocation, and its buffer is reused by the next node of that size once the node is dropped (up to 8 MiB per thread); the table that makes equal operations one node and the list of live nodes are kept per thread instead of behind a lock per context (a thread only ever shared or swept the nodes it made); the references a node keeps to its first three readers need no allocation. Generated kernels, kernel keys and results are unchanged.
- wgpu: the scatter-add kernels (`put_add`, group-by sums, histograms, embedding gradients) use the device's native `f32` atomic add (`SHADER_FLOAT32_ATOMIC`, Vulkan `VK_EXT_shader_atomic_float`, Metal 3) when it has one, instead of a compare-and-swap loop on the bits, which collapsed when many invocations add to few cells (60x slower than CUDA on 16 hot bins). Bindings that also take `min`/`max` keep the loop. `LUMA_WGPU_DUMP=<dir>` writes the WGSL of every kernel the backend compiles, and `luma-bench` prints the per-kernel GPU times of each case with `LUMA_BENCH_KERNELS=1`.
- Thomas line solves (`solve_tridiagonal*`, `solve_block_tridiagonal*`, on parts too) of scalars and 4x4 blocks load four rows before eliminating them instead of waiting a memory round trip per row (RTX 3090: scalar solve along a strided axis of 4096x4096 1.96 -> 0.89 ms, 4x4 blocks along axis 1 of 1024x1024 1.12 -> 0.68 ms), and stacked scalar solves on strided axes use the warp-wide plan of the parts solve.
- `iterate_all` and `Array::iterate`: a step of lazy expressions only that is not a register loop (shifts, say) is also traced as the body of up to 16 steps, so that the recorded launches are those the planner makes of the chain (it fuses several steps of shifted expressions into a kernel, as an eager `u = step(&u); u.eval()` loop does), and the loop replays that body `steps / k` times and the remaining steps one by one. It is used when the chain makes fewer launches than the steps one by one, the cost model gives it less time, and tracing it costs less than half of what the loop takes step by step; steps that launch or read back while traced, or whose chain does not fuse, replay as before, and so do all loops on the CPU backend (which recomputes the shifted values of fused steps instead of staging them in workgroup memory, and is slower for a large state). The 1D stencil on 4M `f32` for 100 steps takes 100 launches and 4.2 ms on an RTX 3090 CUDA before, 40 launches and 1.8 ms now. `step` is called up to 16 times more while the body is traced. Rounding may differ from the step-by-step loop in the last bits (fused steps recompute shifted values).
- `iterate_until_all`: a one-shot loop that has run 100 launches without stopping replays the rest as repeated graphs of a small unit of steps (at most 16, about 64 launches) instead of launch by launch, where the backend captures graphs (CUDA): until loops of 200 steps of a 30-launch step run in 10.6 ms instead of 14.3 ms. A compiled loop keeps capturing whole blocks.
- `iterate_until_all` on backends without a single-pass reduction (wgpu): the gate runs in the last pass that finishes the record's reduction instead of a launch of its own, so a step is one launch less (a step of one launch and a record is two, not three).
- Fused expressions that read a transposed operand (`a.t() + b`, `a.t() * s + b`, the transpose of a sum, `select` of a transposed comparison; operands of `f32`, `u32` and `i32`, matrices or batches of them, other operands contiguous, broadcast or sliced) run as a tiled kernel like the pure transpose instead of the generic kernel, which read the transposed operand with a stride: each workgroup stages a padded 32 x 32 tile of every transposed operand (a sub-program such as `a + b` under the transpose is computed while the tile is loaded) in workgroup memory, computes its tile of the result with the fused program and writes it with consecutive invocations on consecutive elements, so every global read and write is coalesced. Programs of at least 65536 elements, up to three staged operands of one geometry; anything else keeps the generic kernel. RTX 3090 `compute/transpose/add` (`a.t() + b`), CUDA: 1024x1024 0.025 to 0.021 ms, 4096x4096 0.395 to 0.256 ms, 8192x8192 1.77 to 1.06 ms (a pure transpose copy moves two thirds of the bytes in 0.72 ms); wgpu: 0.061 to 0.050 ms, 0.692 to 0.281 ms, 5.09 to 1.06 ms. `jit/fusion/transpose` `(a^T+b)*0.5` 2048x2048: CUDA 0.086 to 0.069 ms, wgpu 0.137 to 0.108 ms.
- Opening a context and dropping it no longer costs time proportional to the on-disk pipeline cache (wgpu, Vulkan; up to the 32 MiB it keeps). Three costs went: the files' checksum was byte-serial FNV-1a (1.2 ms per MB: 30 ms for 32 MB, paid when the file was read, when the driver's cache was compared with it, and again when it was written), now a four-lane word-at-a-time checksum (1.7 ms for 32 MB; the cache files' format number is 2, so existing files are read as absent and rewritten once: kernels, PTX and CPU code compile again on first use, and the device is measured again); the stored cache was copied after reading to cut its header off, now handed to the driver as read; and releasing a context read the whole driver cache back (`vkGetPipelineCacheData`) to find it unchanged, at every release, now at most every 5 s after the device opened or the cache was last written (and always at exit, as before). The cache is loaded by a thread started when the device opens, which the first pipeline built (or the exit handler) waits for, so a process start no longer waits for it: `luma-bench` `jit/compile/newctx_empty` (open + 1 op, RTX 3080 Laptop, wgpu, 27 MB cache) 30.0 ms to 0.35 ms (empty cache 2.25 ms to 0.35 ms, `LUMA_CACHE=off` 0.34 ms), `newctx_kernels` 29.6 to 3.1 ms, and a process that opens a context and exits takes 519 ms instead of 659 ms (548 ms with an empty cache, 571 ms with `LUMA_CACHE=off`). Cold compile (`compile/cold_chain12` 33.7 and 30.8 ms before, 33.1 and 31.2 ms after) and the first calls of `ml/conv`, `cfd/bc` cases are unchanged. CUDA reads its PTX per kernel, never the whole cache, and is unchanged (`newctx_empty` 0.19-0.22 ms both).

### Fixed

- wgpu: axis reductions of fewer than 524288 cells (`Tuning::axis_defer_cells`) are recorded when called, not left pending for the sibling fusion; the CPU backend keeps its operation-count rule. `ml/train/sgd_small`, `sgd_medium`, `adam_small` and `adam_medium` had become 9-12 % slower than before the fusion (RTX 3090 `sgd_medium` 0.662 ms against 0.606; the bias gradients of 131072 cells or fewer ran as late as their first reader, behind the work recorded after them, and merged into one kernel without a saving), while at 1048576 cells (`sgd_large`) the fusion is 7 % faster (1.306 ms against 1.405) and stays.
- `matmul`: a skinny product of more than 96 rows and more than 8 columns takes the staged tiles again (`Xᵀ R` of a softmax regression, 128 features, 20 classes, 1M samples, RTX 3090: 0.78 ms tiled, 1.30 skinny; 128 x 32 0.81 against 1.53; the shapes up to 96 rows keep the skinny kernel: 64 x 20 0.69 against 1.15). The training step of a 1M x 128 x 20 softmax regression (`ml/classic/softmax`, new) takes 19.7 ms per 10 steps again instead of 25.1; the 200k x 64 x 10 one keeps its 5.0 ms (8.0 without the skinny kernels).
- CUDA: the compact IR's typed `getelementptr` for dynamically indexed accesses (`float* + index`, 64-bit index arithmetic) made ptxas schedule the operand loads of fused matmul tiles worse; dynamic indices use the byte form again, constant offsets stay typed. RTX 3090: `ml/attention/single` T2048 0.526 -> 0.452 ms (T512 0.093 -> 0.086), `train/sgd_medium` 0.583 -> 0.560 ms; the fused PV matmul kernel 412 -> 338 us (before cudair 336 us); the IR stays a third of its old size.
- `matmul`: a skinny product of an operand that waits for axis reductions (a correlation matrix, `z.matmul(&z.transpose())` with `z = (x - mean) / std`) went back to recomputing `z` in every tile: an axis reduction that is not recorded yet is a leaf with no buffer of its own, so `z` read one buffer and looked cheap enough to recompute (1M x 32 `table/corr` on an RTX 3090: 1.33 ms after the sibling axis reductions were left pending, 1.20 ms before). The pending reductions count as the buffers they become, so the staged tiles run again and the one kernel that records the mean and deviation together keeps its saving: 1.33 -> 0.92 ms.
- CUDA cold start: a group of pending reductions that is recorded in one merged kernel (`perf-multireduce2`: `concat_slice` of a few dozen `sum_where` results, a 3-D solver's wall forces) is recorded one reduction at a time, with the kernel all reductions of that shape share, until the same group structure (the program and tail, not the values of its scalars) has been recorded `Tuning::reduce_multi_after` = 4 times (`LUMA_MULTI_REDUCE_AFTER=n` overrides, 0 = merge at once; 0 on other backends). The merged kernels then compile in the background, side by side (the kernels of the other groups of the call too), and no launch waits for them: a group is recorded in its kernel once it is ready. Merging a group cost 0.6-0.7 s of NVVM and PTX JIT per kernel, one kernel after the other (6 kernels, 4.1 s of bowshock's 5.8 s seed-wedge `final`), which a group recorded a few times never earned back. `examples/bench_coldforces` (42 sums in 3 groups, empty kernel cache, RTX 3080 Laptop): first call 1.76 s -> 0.11 s (27 ms at 922fee6 and c6922e2, 2.3 s at 8787baa, `perf-multireduce2`; 1.2 s with `LUMA_MULTI_REDUCE_AFTER=0` and the compiles side by side), calls 2-12 3.2 ms while the merged kernels compile, 0.5-0.6 ms from then on (the call that builds them takes 27-87 ms, the driver's JIT stalls launches).
- CPU backend: three choices of this release's performance work that made kernels slower (32-thread Ryzen 9 7950X, interleaved ABBA against the commit before it, best of two rounds): kernels of 2500 to 16000 IR instructions went through Cranelift's optimiser, whose code was slower than the unoptimised one for most of them, so the optimiser runs again up to 3500 instructions (the all-pairs force of `compute/nbody` is 1.6x faster with it) and up to 16000 only for kernels with run-time contiguity checks (`ml/conv/stencil5x5` 4.9 ms against 0.83 ms; 2.7 ms without it): `compute/fractal/mandelbrot_unrolled` 16.2 to 9.8 ms (10.7 ms before the release), `cfd/muscl/2d_iterate` 2.78 to 1.97 ms (156x122) and 5.47 to 3.82 ms (256x192), `compute/fft/conv1d_fft` 4096 * 64 0.24 to 0.14 ms and 65536 * 1024 0.36 to 0.26 ms. Axis reductions of light operands (under 12 f32 operations per cell: a table's NaN-aware column statistics) are no longer left pending to be recorded together on the CPU backend, where recording them together ran 1.5x slower than one kernel each (`datasci/table/describe_nan` 2.40 to 1.59 ms, 1.66 ms before the release); heavy ones that share a long chain (all-pairs forces) are still merged (`compute/nbody/forces` 4096 bodies 6.1 ms, 35.7 before the release)). The idle workers of the pool poll for 200 microseconds before they sleep instead of a millisecond, which made a launch followed by host work slower (the host's own memset competed with the pollers): `ml/embed/scatter_add` V8192 D128 T1024 0.25 to 0.19 ms, `datasci/take/sorted_idx` 0.57 to 0.50 ms, `datasci/group/sum_g1k` 1.28 to 1.20 ms, `jit/readback/item_max_big` 6.0 to 4.0 ms, with the loops that need polling (`cfd/heat`, `jit/loop`) unchanged; no polling at all makes them 1.4 to 2x slower.
- An intermittent segmentation fault at process exit with the NVIDIA Linux driver (615.71: wgpu, and CUDA's PTX JIT) while the driver's compiler workers were still busy: the exit handlers that wait for them ran too late. `exit` runs handlers newest first, and the driver registers the destructors of its compiler state when it first compiles (`libnvidia-gpucomp`, when the first pipeline is built, long after the device exists; `libnvidia-ptxjitcompiler`, when the first kernel is loaded), so a handler registered once at start-up ran after them and found the state of the compiler freed under its worker (`SIGSEGV` in `libnvidia-gpucomp` while the handler wrote the pipeline cache). The wgpu backend registers its handler again after every call into the driver, and the compile pool of CUDA and the CPU backend after every job; the newest registration does the work.
- `matmul`: a skinny product whose operand is an expression of several buffers that the kernel would recompute 4 to 7 times per element (a correlation matrix, `z.matmul(&z.transpose())` with `z = (x - mean) / std`) takes the staged tiles again, which compute each element twice: RTX 3090, 1M x 32 `table/corr` 1.70 -> 1.20 ms, as before the skinny kernels. Pending operands read 8 or more times are still evaluated first, and stored operands keep the skinny kernels.
- CUDA and CPU: the compile worker pool starts its threads on demand and a thread that waits for a kernel no worker has begun compiles it itself. The pool started one thread per core and woke them in turn, so a kernel compiled on its own (the first launch of a new context, a cold structure in a loop) ran on a thread that had never used the compiler or the driver, which took the 0.2 ms of opening a context and building one cached kernel to 0.5 ms and added a few milliseconds to each cold kernel; now a lone kernel is built on the calling thread as before and the pool only widens when several kernels are queued at once.
- CPU backend: an `eval` loop of a stencil step (`u = step(&u); u.eval()`) ran every step through the time-blocked stencil kernel with one step per pass, several times slower than the fused elementwise kernel (the CPU finishes each step before the next is recorded, so a pass never got more than one). A one-step plan takes the fused path on the CPU.
- CUDA transfers between host and device: uploads and readbacks of tens of megabytes and more allocated and freed pinned staging memory on nearly every call (`cuMemHostAlloc` costs 0.4 ms per MiB, `cuMemFreeHost` a third of that) whenever the transfer needed more blocks than the pool kept, and read or wrote in pieces of 32 MiB (or in one piece up to 64 MiB) with the host copy not overlapping the DMA. Uploads now stage through 2 MiB pieces whose pinned blocks go back to the pool as soon as the DMA out of them is done (a transfer of any size cycles through a few blocks, and an event per piece frees a block without ending the batch), readbacks go in pieces of 1-8 MiB with four in flight, and the pool serves a request only with a block at most four times larger. On an RTX 3090 (PCIe 4.0 x16) an upload of 256 MiB takes 11-21 ms (23 GB/s on a quiet host) instead of 46 ms, one of 1 GiB 48 ms instead of 377 ms, and LumaPy's host-in workloads run 6-9 times faster (Monte Carlo from NumPy normals 134 -> 23 ms, normal-equations regression from NumPy arrays 299 -> 32 ms; wgpu 17 and 33 ms).
- Every host copy asked the OS for the number of CPUs first (`available_parallelism` reads cgroup files: 28-60 us per call), which was most of what a small `to_vec`, `item` or upload cost on every backend that stages through `copy_parallel`. The copy now runs on a pool of eight threads that stay ready between the pieces of a transfer (they poll for 200 us before they park), starts at 2 MiB instead of 8 MiB, and never asks the OS. On CUDA `item()` of a reduction takes 16 us instead of 56-68 us and `to_vec` of a few host-resident values 4-6 us instead of 37-57 us.

## [0.5.1] - 2026-11-21

### Fixed

- CUDA on Windows: the NVVM compiler (`nvvm64.dll`) is looked for in the directory of the loaded NVIDIA driver (`System32\DriverStore\FileRepository\nv_dispi.inf_*`), where the driver installs it but which is not on the DLL search path. The CUDA backend never opened on Windows without that directory on `PATH`, and `Auto` fell back to wgpu.
- CUDA: `sort`, `argsort`, `sort_axis`, `factorize` and everything built on them (`unique`, `median`, `quantile`, `searchsorted`, group-by) failed with `CUDA_ERROR_ILLEGAL_ADDRESS` on every input since the single-digit scatter bypass of 0.5.0. The scatter kernel read the digit offsets at index `-1` for digit 0 inside a `select` (both operands are evaluated); wgpu clamps out-of-bounds reads, the CUDA backend does not. The index is clamped now.

## [0.5.0] - 2026-11-20

### Added

- Six-crate workspace: `luma-hal-core` (kernel IR and backend interface), `luma-hal-wgpu`, `luma-hal-metal`, `luma-hal-cuda` and `luma-hal-cpu` (each a backend with its own code generator), and the root `luma` crate. Cargo features `wgpu`, `metal` and `cuda` (all default) each enable one GPU backend; the CPU backend is always built, so every build runs on any machine.
- Native Metal and CUDA backends next to wgpu. `ContextOptions::backend`, `Context::{cuda, metal, wgpu, cpu, backend}` and `LUMA_BACKEND` (`cuda`, `metal`, `wgpu`, `cpu`, `auto`) choose one; `Auto` tries CUDA, then Metal, then wgpu, then the native CPU backend. `LUMA_DEVICE` also accepts a backend name. The CUDA backend needs only the NVIDIA driver: kernels are emitted as NVVM IR and compiled to PTX by the driver's own NVVM library (`libnvidia-nvvm`), linking NVIDIA's `libdevice` (embedded) for math; no CUDA toolkit or NVRTC is needed. Custom WGSL kernels run on the native backends through the kernel IR; cooperative matrix stays wgpu-only. CUDA, Metal and the CPU backend replay traced fixed-iteration steps from a cached graph; wgpu does not capture graphs.
- A native CPU backend (`luma-hal-cpu`, always built): the kernel IR compiled with Cranelift to SIMD code that runs on a pool of host threads (`LUMA_CPU_THREADS`, default the available parallelism), with its compiled code cached on disk. It replaces the llvmpipe fallback.
- `compile_until_all` and `CompiledUntil` (`run`, `run_steps`): the loop of `iterate_until_all` traced once and run on any number of states of the same shapes, for solvers that step in chunks. `Series` gains `window_mean(k)` and `prev(k)`.
- Time-blocked stencils of an `eval` loop over up to four `f32` arrays evaluated together (one tile holds all the states).
- `Context::device()` returns a `DeviceSummary { backend, name, software }` (also exported from the crate root) and `Context::device_name()` the name, so an application can tell which device runs; `software` is true for the native CPU backend.
- `Array::permute_axes([usize; 4])` and `swap_axes(a, b)` (lazy views like `transpose`, which is
  `swap_axes(0, 1)`), their writing forms `permute_axes_mut` / `swap_axes_mut`, and the same on
  `ViewMut` and `Fields`. Swapping axes 1 and 2 of a `[ni, nj, nk]` array gives the reduction
  or stencil along `j` unit-stride reads without a copy.

### Changed

- Breaking: software wgpu adapters (Mesa llvmpipe, SwiftShader, WARP) are never used. The wgpu backend refuses them, never requests a fallback adapter and refuses `DeviceKind::Cpu`; a CPU device (`LUMA_DEVICE=cpu`, `Context::cpu()`, or `Auto` without a GPU) is the native `cpu` backend, which every build includes. The `cpu` cargo feature is gone.
- A traced step (`iterate`, `iterate_all`, `iterate_until`) replayed from a recorded plan keeps far fewer buffers: the intermediates that the step allocates and drops (a `cache()` stage, a staging buffer) share one buffer whenever the launches that use them do not overlap, where each used to keep a buffer of its own until the plan was dropped. A chain of 8 stored stages held 9 buffers, now 4 (the state's two and 2 shared by the stages). State, probes, results and everything that existed before the trace keep their buffers, so graph capture and the results are unchanged.
- `Df32` round trips evaluate `hi + lo` consistently, including signed zero: the simplifier no longer treats a conversion from `f32` through `Df32` as an identity, and the low word is guarded against compiler folding.
- Breaking: adapter, device and readback errors in `LumaError` carry backend-neutral strings instead of wgpu types. `Features` and profiling types are re-exported from `luma_core`; `luma::wgpu` is re-exported only with the `wgpu` feature.
- Timestamp profiling drains completed query sets periodically instead of retaining every set until the profile ends. This avoids Metal device loss after 32 live query sets while preserving every launch's timing and per-kernel totals.
- `iterate_until_all` and `compile_until_all` loops replay a block of steps as one graph on backends that capture them (CUDA, Metal, CPU): the stop rule's gate kernel counts the steps itself and passes the dispatch arguments from step to step on the device, so nothing in a block depends on the host, and a block costs one launch instead of two per recorded kernel. A `CompiledUntil` keeps its buffers and captured graphs from run to run. The graph interface, `GraphLaunch`, takes a `Dispatch` (direct or indirect) instead of workgroup counts.
- CUDA: a `CompiledUntil` keeps its captured graphs from run to run (the even-step state is copied out instead of detached, which made every run capture its blocks again). Graphs are captured only where they pay: a compiled loop's spans on first use, a `Stepper`'s steps when about to be replayed eight or more times, a one-shot loop's spans the second time they run.
- CUDA: f32 `/`, `sqrt`, `exp`, `log`, `log2`, `sin`, `cos`, `tan` and `pow` compile to the hardware approximations, within WGSL's precision rules, as the wgpu and Metal backends' compilers do (`div.approx`, `sqrt.approx`, libdevice's `__nv_fast_*`); f64 and the other functions keep libdevice's precision. f32 `/`, `sqrt` and `inverseSqrt` use the forms that flush subnormals, which WGSL allows and the Vulkan driver does: the others carry range fix-ups (the Euler step of `bench_pde` has a sixth fewer instructions and takes 10–16 % less GPU time).
- CUDA: a profiled launch's end event is created before the launch, so a short kernel is not charged for creating it. `LUMA_CUDA_DUMP=<dir>` writes every kernel's NVVM IR, PTX and label.
- CPU: `subgroupMin`/`subgroupMax` on floats skip a NaN operand as `min`/`max` do (they propagated it), so `Array::min`/`max` and their axis forms agree on NaN again whatever order the lanes meet in. Compiled CPU kernels cached on disk carry a codegen revision, so blobs from older builds are compiled again.
- Fused stencils: a view read through a view without edge handling (a slice of a padded array, a slice of a slice) is composed into one view whatever its own edge handling, so reads of one element that a stencil reaches by different paths are computed and loaded once; a view nested in another takes its coordinates from the outer one instead of dividing the flat index again, and so does a range (`range_on`, the region masks of a lazy assignment) computed in a view. The Euler step of `bench_pde` has a quarter of the integer divisions and a third fewer clamps per cell: 14–20 % less GPU time on an RTX 3090 and an RTX 3080 Laptop (#8).
- Equal scalars of one kernel share a uniform slot (`GAMMA` written in four places is one value), so the backend compiler computes what is derived from the same cell and constant once: the Euler step computes each cell's speed of sound once instead of once per face (5 instead of 8 square roots per cell; 5 % less GPU time on CUDA). The kernel key now depends on which scalars are equal, still not on their values: values that happen to coincide build one more kernel. A program replayed from a recorded recipe is rebuilt when scalars that shared a slot there differ, or when scalars that had slots of their own there are now equal.
- Recording a traced step holds the memory its intermediates need at once instead of their sum: an allocation during the recording takes a buffer the recorded commands are done with (an intermediate released, or one only the recording still pins), as the merge at the end of the recording would have shared it. Per-solve scratch is reused the same way. bowshock's 708 k-cell 3D implicit step peaks at 3.0–3.1 GiB of device memory instead of 4.0–4.1 GiB (#9).
- A `Stepper` no longer keeps the probes its step was traced on: placeholders take their names in the recorded commands, and their memory goes with the probe arrays.
- `Context::memory_stats()` (resident, pooled and spilled bytes, and the budget) and `Context::trim()`, which gives the idle buffer pool back to the device, for instance after dropping the runners of another grid shape.
- CUDA: a reduction over more than one workgroup is one launch instead of two (`sum().item()` loops, norms and means, stop rules and force monitors that reduce small and medium arrays, the speculated readbacks of a loop and the check of `iterate_until`). Every workgroup of the first pass leaves its partial result, fences it and takes a ticket from a one-word device counter; the workgroup that takes the last one (the classic last-block reduction of CUDA's `threadFenceReduction` sample) combines the partial results exactly as the second pass did, so results keep their bits, applies the epilogue (`norm`, `mean`, ...), writes the values and resets the counter. The wgpu and Metal backends keep two launches: WGSL gives no ordering of memory across workgroups. `LUMA_SINGLE_PASS_REDUCE=off` restores two launches on CUDA. RTX 3090: `iterate_until_all` with a max-norm record on a 1024 x 1024 heat stencil, 200 steps, 7.63 ms to 5.34 ms of GPU time (801 to 601 launches); the Euler `until_all` of `bench_pde` 204 to 154 launches.
- `iterate_until` stores a step and reduces its residual `max |next - prev|` in one kernel: the check kernel recomputed the whole step expression at every cell although the step stores it in the same iteration. With the single-pass reduction an iteration is two launches (the step with its reduction, and the gate) instead of four. RTX 3090, a heat stencil to convergence (`bench_pde` `converge/iterate_until`, 200 steps): 1024 x 1024 5.82 ms to 3.96 ms, 256 x 256 2.50 ms to 1.81 ms.
- `iterate_until_all` / `compile_until_all` with one record that is a reduction (a norm, a sum, a maximum) trace the record of the state a step reads together with the step, so one launch stores the next state and reduces the record (the program shares what the step and the record compute) instead of a launch for the step and one or two for the record; the gate that evaluates the stop rule then runs in the last workgroup of that launch where a reduction finishes in its first launch (CUDA; the CPU backend with `LUMA_SINGLE_PASS_REDUCE=on`), and `iterate_until` runs its gate there too: a step of the loop is one launch instead of three (two for `iterate_until`). The loop makes one launch more than it has steps (the record of step `j` comes out of launch `j`, the last launch computes a step that nobody sees; the state, steps, reasons and histories are the ones of before, bit for bit). The constants of the stop rule are read from the gate's state, so a new tolerance builds no kernel. Several records, records that are not one reduction, steps that write arrays they captured (the extra launch would write them once more), steps that read their index (for the fold), steps that stage sub-programs in workgroup memory and backends without a single-pass reduction keep the launches of before. `bench_pde`, RTX 3080 Laptop: `euler/until_all` 2048 x 2048 39.5 ms to 22.3 ms, `converge/until_all` 1024 x 1024 9.75 ms to 7.50 ms; RTX 3090: `converge/until_all` 256 x 256 2.31 ms to 1.64 ms.
- A materialized axis permutation (`transpose`, `permute_axes`, `swap_axes` evaluated, cached or copied) of an evaluated array moves 32 x 32 tiles through workgroup memory, so both its reads and its writes are coalesced; the generic kernel read the source with the stride of the swapped axis. RTX 3090, 8192 x 8192 `f32`: 1.63 ms to 0.72 ms (329 GB/s to 746 GB/s).
- A record assigned over a region (the ghost cells of a boundary condition: one `select` per field under the same mask) runs what its fields' right-hand sides share inside one branch: each field's `select` was its own branch, so a load or a square root two fields used was computed at every cell of the array, not only in the region. bowshock's sector boundary kernel on an RTX 3090: 52 us to 29 us.
- CUDA: the scans of integers (`u32`, `i32`) and the `min` / `max` scans of any type along a contiguous line of more than 4096 cells read their input once instead of twice. One kernel replaces the chunk totals, the scan of the totals and the seeded second pass (decoupled look-back, Merrill & Garland 2016): a workgroup takes the next chunk from a device counter (so every chunk it waits for has started, whatever the scheduling), scans it in workgroup memory, publishes its aggregate, walks back over the status of the chunks before it a workgroup-wide window at a time until one has published its inclusive prefix, publishes its own and writes the chunk seeded with the prefix. The counter and the status words are zeroed by a small launch before the scan, so a recorded step or a graph replays both. None of these results depends on how the operands are grouped, so they are the same as before; the sums and other floating-point scans keep the two passes (the look-back would group their additions by the timing of the workgroups), so every scan stays reproducible run to run. The wgpu and Metal backends keep the two passes: WGSL gives no forward progress or memory ordering across workgroups. `LUMA_LOOKBACK_SCAN=off` restores them on CUDA. RTX 3090: a running maximum along the contiguous axis of 8192 x 8192 `f32`, 0.98 ms to 0.68 ms.
- CUDA: a small readback after large transfers no longer allocates and frees page-locked host memory every time. The idle staging pool reused only blocks of exactly the requested size class and was full of 32 MiB blocks after the uploads, so every `item()` paid about 1 ms of `cuMemHostAlloc` + `cuMemFreeHost`; idle blocks are now taken by best fit, and a block returned to a full pool evicts larger idle ones. `x.sum().item()` of 2^26 `f32` on an RTX 3090: 0.98 ms to 0.35 ms, `(&x * &y).sum().item()`: 0.99 ms to 0.65 ms (both at DRAM bandwidth now).
- CUDA: device memory comes from a dedicated stream-ordered pool (`cuMemAllocFromPoolAsync` / `cuMemFreeAsync`, memory kept for reuse until `Context::trim` or memory pressure). `cuMemFree` synchronized the whole device, so dropping any buffer stalled the host until the GPU had drained its queue (bowshock's 3D sector run: 165 frees averaging 13 ms); now a drop never waits. Devices without memory pools keep `cuMemAlloc` / `cuMemFree`.
- Stacked block-tridiagonal solves of blocks larger than 5x5 always run by cyclic reduction: the per-line Thomas kernel of 6x6 and 8x8 blocks is 10x to 25x slower at every line count measured, and the line-count rule picked it for many lines (RTX 3090, 400 lines of 200 rows: 6x6 1744 us to 215 us, 8x8 8386 us to 775 us).
- The Thomas line solve of blocks up to 4x4 on fewer than 8192 lines runs two invocations per line, one sweeping down from the first row and one up from the last, meeting in the middle (a twisted factorization: they exchange their last swept blocks through workgroup memory and solve the 2x2 block system there, then each back-substitutes its half). One invocation per line left the GPU nearly idle on a few thousand lines and every invocation walked a long dependent chain. RTX 3090: stacked 4x4 blocks, 1024 lines of 1024 rows 1516 us to 1100 us (along the other axis 1183 us to 709 us); parts along the strided axis of a 187 x 70 x 24 grid (4488 lines of 70 rows), 2x2 blocks 76.8 us to 55.2 us, 4x4 253.9 us to 186.3 us; scalar lines, 4096 strided lines of 4096 rows 2.2 ms to 2.0 ms. 5x5 blocks (more registers per invocation) and many lines (enough invocations already) keep one invocation per line, where the twisted solve measured 7-10 % slower.
- The cyclic-reduction solve of 5x5 blocks along a strided axis takes one invocation per row (up to 128) when its parts are lazy expressions computed in the kernel: each row's entries are then an expensive expression, and one invocation per 8 rows left it to few threads. bowshock's 3D wedge step (156 strided lines of 140 rows), RTX 3090: f32 290 us to 271 us, Df32 391 us to 374 us, viscous 288 us to 276 us. Evaluated parts keep the narrower workgroup, which is faster for them.
- wgpu: staging buffers for readbacks and uploads are taken from the idle pool by best fit, and a buffer returned to a full pool evicts larger idle ones (as on CUDA, a pool filled by large transfers made every small readback create and drop a buffer); the uniform arena reuses its chunks once the submission that read them has completed instead of creating a new buffer for every batch that overflows 256 KiB.

- A pipeline cache that grew past the 32 MB kept on disk is dropped with an `info` log message
  naming its size, instead of silently: every later process then compiles its kernels again
  (about 9.5 s for bowshock's 3D solver). The cache stays one file per device: one cache per
  pipeline, which would let old pipelines be evicted one at a time, takes about 1.77 MB per
  pipeline on NVIDIA's driver (150 kernels: 266 MB).
- Pending expressions that several pending operations read are stored in fewer, smaller kernels. A
  3D implicit CFD step (block-tridiagonal line solves over 50 lazy face-matrix entries per
  direction, each a large expression of the face states and the dissipation, read by the solves and
  by a flux-difference operator) took 4.86 ms of GPU time in 42 launches and 220 fusion cuts without
  a hand-placed `.cache()` on the entries, against 3.14 ms in 27 launches with it. Three causes: the
  products that scale a stored entry (`lo = entry * s`) weighed as much as the whole expression they
  read, so they were stored too, in full; the entries were evaluated together with every pending
  node of the same operation made at the same place, and a helper such as `E::add` makes all of a
  scheme's matrix entries there, so a group held 146 outputs where 50 were read (the group's kernel
  wrote 96 matrices that nothing else read); and what the group shared with the rest, once the group
  was a kernel of its own, was stored level by level, a kernel each. Now a node read from outside
  the kernel is stored only if it is still worth storing with the nodes below it that are stored as
  buffers (`Shared` cuts); the entries evaluated with a node are only the ones that pending
  operations outside the kernel read; and a node of the same length that the group's kernel computes
  anyway and others read too is one more output of it. The step takes 3.28 ms in 29 launches
  uncached (3.45 ms per iteration, 3.25 with `.cache()`), and in `Df32` 4.09 ms against 5.68
  (`.cache()`: 3.98). RTX 3090.
- The Thomas solve on parts along a strided axis (`solve_block_tridiagonal_parts` of blocks up to
  5x5 and `solve_tridiagonal_parts`, lines at least 16 cells apart) is up to 1.5x faster on an
  RTX 3090. A workgroup is a warp or more (64 invocations for blocks of 3 to 5, 128 for 2x2
  blocks, 32 for scalars; it was 8 to 32, which left each warp mostly idle and each load a lone
  sector), and the scratch of the swept upper blocks is tiled: entry `k` of the 32 cells that a
  warp's lines fall in lie in one contiguous 128 byte run, so every store and load of the scratch
  coalesces. Grid of 187 x 70 x 24, 5x5 blocks, parts of 50 face fields (GPU time of the solve,
  best of 7): j lines (4488 of 70 cells) 272 us to 214 us, k lines (13090 of 24) 238 us to 180
  us; with 80 evaluated parts: j 406 us to 273 us, k 249 us to 222 us; j lines of 4x4 blocks
  213 us to 166 us (evaluated 228 us to 184 us), of 3x3 155 us to 119 us (evaluated), of 2x2
  109 us to 85 us (evaluated). What is left is the memory traffic of the solve (161 MB for the
  5x5 j lines, 86 % of the card's bandwidth in 200 us: a variant without the elimination takes
  200 us). Stacked blocks (`solve_block_tridiagonal`), strides below 16 and blocks above 5 keep
  their kernels: the wide tiled solve measured 1.1x to 1.3x slower on them
  (`bench_solve --only sweep/`).
- On NVIDIA GPUs the cyclic-reduction line solve keeps a whole line in workgroup memory up to
  48 KiB (it was 32 KiB) instead of spilling rows to a scratch buffer. 5x5 blocks, 1680
  contiguous lines of 187 cells on parts: 313 us to 251 us (parts of 50 face fields: 274 us to
  231 us); 300 cells of 4x4 or 3x3 blocks, 480 lines: 343 us to 246 us and 215 us to 131 us;
  stacked 5x5 blocks, 400 strided lines of 200 cells: 370 us to 158 us; 8x8 blocks, 156 lines
  of 122 cells: 157 us to 130 us. Two rows of `bench_solve --only sweep/` got slower, by 12 %
  and 13 % (3x3 blocks, 200 lines of 400 cells: 36 us to 40 us; 8x8 blocks, 256 strided lines
  of 64 cells: 122 us to 138 us), where the half-spilled line left room for more workgroups per
  multiprocessor.
- `bench_solve` has `parts3d/...` rows: the parts solves of a 187 x 70 x 24 grid along each axis,
  evaluated and from 50 face fields, with scalar, 2x2, 4x4 and 5x5 blocks.
- Generated kernels no longer bake the extents of their arrays into the WGSL: the same expression
  on a grid of another shape is the same kernel instead of one more driver compile. Moved into the
  uniform: the dims, strides and shifts of circular shifts (`ShiftKind::Dims`), the output
  divisors and wrap extents of views, the strides and sizes of `Array::range`, the extents of
  the circular-shift tiling of kernels that read a lazy expression through shifts, and the
  workgroup-memory rows of cyclic-reduction line solves (rounded up to a multiple of 32). A divisor
  that is a power of two stays a shift or a mask (its own kernel); any other is read as words (`d`,
  a 32-bit magic number, a shift, an `f32` reciprocal): below 2^23 elements an `f32` product and a
  correction of one, above it the exact high-word division. Views of a kernel that divide by the
  same number share the words so that the shader compiler computes the quotient once. Measured
  on the table of 10 Mach points of a CFD solver (each on its own grid of 16 536 to 21 840 cells,
  luma and driver disk caches off, RTX 3090): kernels built 172 -> 50, the first chunk of each
  point after the first 1.4-1.9 s -> 0.05 s (0.2-0.55 s for three points whose grids fall in
  another class of powers of two), first chunks in all 16.5 s -> 4.3 s, table 53 s -> 31 s.
  `euler/per_step` on 2000 x 2000 (arithmetic-bound, every view divides by a runtime extent) is
  3-5 % slower, power-of-two shapes and the other `bench_pde` / `bench_datasci --quick` rows are
  unchanged within noise. Kernels that stage sub-programs through views in workgroup memory and
  stacks (`Tail::ParamStack`) take their extents as parameters too; still built per shape: private
  scatter-adds and staging keys that cannot be lowered to parameters.
- `log::debug!` lines name every kernel compiled, with its WGSL size and compile time.

### Fixed

- The calibration of the cost model (a context opened with `calibrate` on and no stored measurement for its device) bound whole calibration arrays on the first chunk of an array. With a `chunk_bytes` smaller than a calibration array (64 MiB on a GPU, 4 MiB on the CPU), or the default chunk of a GPU with less than about 640 MiB of memory, its kernels read and wrote far past a buffer: on wgpu a validation error that failed the context's later operations, on CUDA and the CPU backend out-of-bounds accesses (a crash on the CPU). Calibration now measures on whole device buffers.
- `MemoryStats::resident_bytes` (and the memory budget) undercounted device memory that recordings, plans and loops hold: the buffers a traced step releases while it is recorded (counted from then on, not only when the recording ends) and keeps in its plan, the placeholders of a stepper, the work buffers of a compiled `iterate_until_all` loop (history, dispatch arguments), the scratch an operation is using, the reduction counter, and the device copy of a captured CUDA graph's uniforms. The doc of `MemoryStats` lists what stays outside both counts (the device context, kernel code, the backend's staging chunks, the driver's pool reserve).
- `Context::trim` on CUDA did not return the memory of the buffers it released itself to the driver: they were freed after the stream-ordered pool had been trimmed and stayed reserved until the next trim. They are freed first now.
- The cost of a region whose assigned value is a scalar or an evaluated array was estimated at half the array whatever the region's size; it now takes the share the region's view shows (or that of a field assigned under the same mask).
- CUDA: a step whose graph capture failed once never tried again; a capture is now retried, and a step goes without a graph after three failures.
- CPU backend: compiling a very wide fused kernel took minutes and tens of gigabytes (a 5661-term `f64` sum cut by the 64 KiB uniform limit: 105 s and 25 GB, the reason `an_expression_with_more_scalars_than_a_uniform_holds_is_cut` skipped the CPU). The lowering kept one Cranelift `Variable` for every single-assignment local and for the accumulator of every gathered load, and the frontend's per-variable map over all blocks grew with their product; those are plain SSA values and block parameters now, and a read at a fixed address (a scalar of the uniform buffer) is one bounds check and one load instead of four guarded lane loads. The same kernel compiles and runs in 2.2 s and 0.44 GB; kernels that read scalars through a uniform buffer got smaller (`view/laplace_*` of `bench_pde` take about a third of the time on the CPU).

## [0.4.1] - 2026-11-15

### Added

- `Array::filter(&mask)` (the elements where an `Array<bool>` or non-zero `Array<u32>` mask holds, in
  flat order) and `nonzero()` (their flat indices) for every `ScalarType` incl. `bool`, as a
  count / scan / scatter compaction on the GPU: about as fast as a copy (0.15 ms for 2^24 `f32`
  with half kept on an RTX 3090, equal to a hand-written kernel). They return `Result<Option<_>>`:
  `None` when nothing is selected (luma has no empty arrays), and they synchronize once (~100 µs)
  to read the count. `filter_padded(&mask, fill)` and `nonzero_padded(fill)` do not: a full-length
  array (kept elements, then `fill`) and the count as a one-element array.
- `topk(k)` and `bottomk(k)` for `u32`, `i32` and `f32`: the `k` largest (smallest) values and
  their flat indices, by a full radix sort.
- `Array::argmax`, `argmin` (flat index, a one-element `u32` array recorded without waiting, like `sum`) and
  `argmax_axis`, `argmin_axis` (the index along an axis) for
  `f32`, `u32` and `i32`: the first of equal elements wins and an `f32` NaN is the first one found
  by both (NumPy). Reducing passes carrying a (value, index) pair, a pending operand computed
  inside the first pass; `argmax` of `2^26` `f32` and `argmax_axis(1)` of `[64, 2^22]` run at about
  860 GB/s on an RTX 3090 (`bench_datasci --only argmax`).
- `Array::searchsorted(&values, Side)`, `Array::digitize(&edges, right)` and `Array::isin(&set)`
  for `u32`, `i32` and `f32` (NumPy semantics; NaNs sort last): lazy binary searches, one per
  value, fused with the expression that makes the values (a lookup of `2^24` values in `2^16`
  edges takes 0.89 ms, 10% less than a hand-written WGSL search). `luma::Side`.
- NaN-aware reductions: `nansum`, `nanmin`, `nanmax` (`f32`, `f64`, `Df32`) and `nanmean`
  (`f32`), lazy one-element arrays, and `nansum_axis`, `nanmean_axis`, `nanmin_axis`,
  `nanmax_axis` for `f32`, with the masking fused into the reduction's first pass (NumPy's
  results: all-NaN gives 0 for the sum and NaN for the others).
- `round` (halves to even, as NumPy), `trunc`, `log1p` and `expm1` for `f32` (`round` and
  `trunc` also `f64`, `log1p` and `expm1` also `Df32`, accurate to 44 bits): lazy and fused;
  `log1p` and `expm1` keep their accuracy for `x` near 0. The fuzzer covers all four.
- Guide: searching sorted arrays, NaN-aware reductions, the new elementwise functions and a
  section of recipes (`diff`, `cumprod`, `softmax` / `logsumexp`, `one_hot`, shuffles, `cov` /
  `corr`, `clip`) in the data chapter.

### Changed

- Devices that bind few storage buffers to a kernel (llvmpipe and many integrated GPUs: 16, Metal:
  29) no longer cut an expression that reads many arrays into a kernel per operation. The lazy
  Keccak of `examples/keccak.rs` (a state of 50 arrays that every round reads) took 7120 kernels
  for three permutations with 16 bindings; it takes 39, and its host time drops from 1.07 s to
  0.18 s (lavapipe, warm; 24 s to 2.5 s the first time, which compiles every kernel). On
  llvmpipe's own limit of 45 bindings 5933 kernels become 24 (3.2 s to 0.12 s). The cut lost
  nothing but the arrays' separate buffers: arrays that a kernel would read beyond its bindings
  (or that a multi-output kernel writes beyond them) now live in one buffer that they are windows
  of, and kernels read the windows through one binding with an offset in the uniform. A write
  (`write`, `+=`, `assign`, a custom kernel taking `&mut`) gives its array a buffer of its own, so
  arrays behave as before; only the arrays used that way pay a copy. Devices with many bindings
  (an RTX 3090 binds 256) never pack. `LUMA_FUSION_MAX_INPUTS=n` lowers the limit to test this on
  any GPU.
- Choosing the arrays a cut evaluates together (`evaluate_alongside`) counts the nodes a candidate
  adds by merging two sorted lists instead of one binary search per node: building the
  three-permutation Keccak on an RTX 3090 takes 54 ms instead of 139 ms.
- The block and scalar tridiagonal solves on parts (`solve_block_tridiagonal_parts`,
  `solve_tridiagonal_parts` and their `_from` forms) choose between cyclic reduction and Thomas
  with a rule of their own: the rule of the stacked solves chose Thomas for thousands of lines
  along the contiguous axis, where its threads read each part a sector per element. Lines along
  the contiguous axis now always take cyclic reduction (RTX 3090: 5x5 blocks, 1680 lines of 187
  cells, 3.2 ms to 0.33 ms; 4x4 blocks, 8192 lines of 187, 15 ms to 0.9 ms), and strided lines
  take it up to `8192 / (2 max(B, 2))` lines (half that when a line spills out of workgroup
  memory) instead of `8192 / 4B`. Over a sweep of blocks 1-5, 24-512 cells and 256-16384 lines
  along both axes the automatic choice now takes 66 ms in total, where always the faster of the
  two would take 68 ms (clock noise) and the old rule took 332 ms.

### Fixed

- A device whose calibration always looks contended (CI's lavapipe on a shared 4-vCPU runner) was
  measured again, twice, by every process and every device it opened, about a second each, and
  the result thrown away: a test opening many contexts spent most of its time calibrating (100 s
  for one test on CI, 7 s without calibration). A contended calibration is now noted in the cache
  directory, and processes use the profile's numbers for a day before measuring again.
- A kernel that declares a binding its entry point never reads can no longer fail the launch with
  `create_bind_group: Number of bindings ... does not match ...` and, because threads share one
  batch command encoder, turn the work of every other thread in that batch into garbage (the
  third time: a fused `slice` and `trunc` allocated uniform slots for a view the lowering never
  applies, found by `fuzz::threads`). wgpu derives the pipeline layout from what the shader
  reads; a launch of one of luma's own kernels now binds exactly the arguments the built
  pipeline uses (reflected once, when it is built), skipping the declared-but-unread ones, and
  checks the argument count and kinds before it records anything, so a mismatch is an error of
  that array only. The lowering no longer allocates uniform slots for the views its instructions
  name but do not apply.
- A kernel that converts a `bool` array to `Df32` and runs no other arithmetic of a `Df32` failed
  with `create_bind_group: Number of bindings in bind group descriptor (3) does not match ... (2)`
  (found by the fuzzer, mostly on `LUMA_DEVICE=cpu`): the program counted the cast as a reader of
  the double-float guard uniform and bound it, but the cast reads nothing from it, so the shader
  compiler dropped the binding.
- `iterate_all` (and `iterate`, `iterate_until`) whose step is a lazy expression that reads a
  reduction of the state (`s.sum() + &c`, `s / s.sum()`) could compute that reduction once, on
  the placeholder that stands for the state while the step is traced: not the recording, and a
  pooled buffer holding unspecified data (what other threads had left in it, so the result
  depended on what ran at the same time: found by the concurrent fuzzer, `fuzz::threads`).
  Flattening the traced step for a register loop no longer evaluates a leaf that reads the
  placeholders; such a step is recorded and replayed with the reduction in it.
- `iterate_all` and `iterate` over an array of several buffers returned the placeholder of the
  state when the step returned its argument unchanged (`-(-x)`), which shares the state's
  buffers: `state *= result` failed with `Kernel arguments 0 and 1 share one buffer`. The
  result is now an array of its own.
- `iterate_until_all` indexed its host arrays with the step count and stop condition it read
  back from the GPU, so a corrupt value (a driver or device fault) panicked with
  `range end index ... out of range`; it now returns `LumaError::Gpu`.

### Known limitations

- Packing arrays into shared buffers on devices with few storage bindings has run on lavapipe
  (16 and 45 bindings) and on an RTX 3090 with `LUMA_FUSION_MAX_INPUTS`, not on Metal or an
  integrated GPU.
- The kernels new in this release (`filter` / `nonzero`, `argmax` / `argmin`, `searchsorted`)
  have not been run on RADV, whose results for workgroup-memory kernels are known to be wrong
  (see 0.3.0).

## [0.4.0] - 2026-11-14

### Breaking

- Comparisons (`lt`, `le`, `gt`, `ge`, `eq`, `ne`) and `is_finite` / `is_nan` / `is_inf` return
  `Array<bool>` instead of `Array<u32>`; `select`, `sum_where` and `Fields::select` take an
  `Array<bool>` or, as before, an `Array<u32>` (non-zero is true: the `Condition` trait). What
  still compiles unchanged: `select(&x.gt(0.0), a, b)`, `sum_where`, `bool * bool` (logical and),
  `mask * &x` / `&x * mask` / `mask * 2.0` and `+` / `-` with a number (the booleans become the 0
  and 1 of the number's type), and `.count()`. What breaks: code that names `Array<u32>` as the
  type of a comparison (`let m: Array<u32> = x.gt(0.0)`), passes one to a `u32` parameter, adds or
  multiplies two comparisons expecting a `u32` result (`a.gt(0.0) + b.gt(0.0)` has no `Add`;
  write `a.gt(0.0).to_u32() + b.gt(0.0).to_u32()`, or `|` for a logical or), divides, shifts or
  `minimum`s them, uses them as an index (`take`, `put`), or reads them back as `Vec<u32>`
  (`to_vec` gives `Vec<bool>`). `sum`, `cumsum` and `sum_axis` of a `bool` array count and return
  `u32` arrays; `reduce_axis`, `scan`, `mean` and the other numeric reductions are not defined for
  `bool`. An integer literal next to a `bool` array needs a suffix (`mask * 2u32`): `u32` and
  `i32` are both promoted to. See the migration chapter of the guide.
- A boolean array takes one bit per element on the GPU (a comparison written to memory is 32
  times smaller than the `u32` array of 0.3), so a custom kernel that reads or writes one sees
  bit-packed words (see the custom kernels chapter of the guide). Writes that cover whole arrays
  (comparisons, `& | ^ !`, `select`, `full`) are packed by the workgroup, and so is `assign` into
  a slice whose rows are runs of 64 or more consecutive elements (whole words inside a run, atomic
  operations only for the words at the ends of a run, which it shares with elements the write
  leaves alone); `put`, `stack`, transposed regions and short rows set and clear single bits with
  atomic operations, which is slower than the same write of another type. `count()`, `any()`
  and `all()` of an evaluated boolean array reduce its words (`countOneBits`), 32 times fewer
  elements than reading the bits one by one.
- The element-type trait `GpuType` no longer requires `bytemuck::Pod` (`bool` is not `Pod`): it
  has an associated `Repr` (the `Pod` form uploaded to the GPU) and conversions, which the
  library routes every host/GPU copy through. `luma::Numeric` (sealed: `u32`, `i32`, `f32`, `f64`,
  `Df32`) bounds the sums, scans and generic reductions that `bool` does not have.

### Added

- A user guide (`guide/`, an mdBook) published at <https://flipnoteman.github.io/Luma/> with a
  version per release (`/vX.Y.Z/`) and `/main/`: arrays, lazy evaluation and fusion, views and
  writes, reductions and scans, records, loops, precision, integer and bit operations, custom
  kernels, a performance guide and worked examples (heat equation, Euler solver, ML-KEM, Keccak,
  CPA). Its code blocks are tested as doctests.
- `Array<i32>`: `+ - * / %` (wrapping; `x / 0` is `x`, `i32::MIN / -1` is `i32::MIN`, `x % 0` is
  0), unary minus, `abs`, `sign`, `& | ^ !`, arithmetic `<< >>`, `minimum` / `maximum`,
  comparisons, `select`, views, reductions, scans and loops; `i32` scalars and `luma::Scalar<i32>`.
- `Array<Df32>::exp`, `ln`, `tanh` and `powf(Df32)`: lazy, fused like the other `Df32` operations,
  relative error below `1e-13` (at most `4.3e-14` measured) for `exp` on `[-55, 80]`, `ln` from
  `1e-24` to `1e30` and near 1, `tanh` on `[-30, 30]` and `powf` of `x` in `[1, 3000]` or
  `[1e-6, 1e6]` with exponents `-0.5..3`. `f64` arrays still have none of them.
- Casts between `f32`, `u32` and `i32`: `to_f32` (from `u32`, `i32`), `to_u32` (from `f32`,
  `i32`), `to_i32` (from `f32`, `u32`). Float to integer truncates toward zero and saturates
  (`u32` at `4294967040`, `i32` at `-2147483648..=2147483520`); NaN becomes 0. `u32` and `i32`
  convert by reinterpreting the bits.
- `Array<bool>`, the element type of comparisons and classifications (stored on the GPU as one bit
  per element: `n` booleans are `ceil(n / 32)` `u32` words, element `i` is bit `i % 32` of word
  `i / 32` and the unused bits of the last word are 0; a custom kernel sees `array<u32>` of the
  words and reads a bit as `(a[i >> 5u] >> (i & 31u)) & 1u`): `from_slice` / `to_vec` / `item` with
  plain `bool`s, `full(true)`, `& | ^ !` between arrays and with `bool`s, `count()` (a `u32`
  array), `sum()` (the same count, as NumPy), `any()` and `all()` (one-element `bool` arrays),
  `to_u32()`, `to_i32()`, `to_f32()`, `to_f64()`, `to_df32()`, `min` / `max`, `sum_axis`,
  `cumsum`, `min_axis` / `max_axis`, views, `take`, `put`, `select` and loops.
- NumPy-style promotions: `bool_array * &number_array` (either order, with a number too:
  `mask * 2.0`), `+` and `-` for `f32`, `u32`, `i32`, `f64` and `Df32` arrays; a product is
  `select(mask, x, 0)` (a masked-out `NaN` or infinity gives 0; `x *= &mask` likewise), a sum or difference converts the
  mask to the number's type first. `bool * bool` is the logical and. There is no `f64` scalar
  promotion, so that `mask * 2.0` is an `f32` array.
- `select`, `sum_where` and `Fields::select` also accept `Array<u32>` conditions (`Condition`).
- The fuzzer covers `i32`, `bool` and the new casts.
- `matmul` (`a.matmul(&b)`, `luma::matmul`) for `f32`: tiled kernels (128x128 and 32x32 output
  tiles, K split over workgroups when the output has few tiles); pending operands, transposes
  included, are computed inside the kernel's loads instead of being stored first.
- `Array::reduce_axis` (up to four of sum, min, max and sum of squares along an axis in one pass
  over the data) and, for `f32`, `mean_axis`, `var_axis` (population variance, two-pass) and
  `std_axis`.
- `put_add`, `put_min` and `put_max` for `u32`, `i32` and `f32` arrays (`AtomicElem`): scatters
  with atomic accumulation, accumulated per workgroup first for targets of up to 4096 elements;
  `f32` sums are not bitwise reproducible. `bincount` and `Array<f32>::histogram`, whose bucket
  expression fuses into the scatter.
- `sort`, `sort_descending`, `argsort`, `argsort_descending` and `sort_axis` for `u32`, `i32` and
  `f32` arrays (`SortKey`): a stable radix sort on the GPU, about 2 ms for 2^24 keys on an RTX 3090.
  Floats sort in total order, NaNs last.
- Group reductions `group_sum`, `group_min`, `group_max` (`u32`, `i32`, `f32`), `group_count`
  and `group_mean` (`f32`): scatter-based, the values and codes fuse into one kernel; codes past
  `groups` are skipped.
- `factorize` (sorted distinct values and dense codes) and `unique` for `u32`, `i32` and `f32`;
  all NaNs form one value, `-0.0` and `0.0` one value.
- `quantile`, `median`, `quantile_axis` and `median_axis` for `f32` (NumPy's linear method, on
  the sorted array).
- `examples/bench_datasci.rs`: every primitive above against a hand-written WGSL kernel of the
  same algorithm (`--gate` fails on a GPU-time ratio above the case's threshold).
- `solve_block_tridiagonal_parts` and `_from`: the block-tridiagonal solve on parts, the entries
  of the blocks and the components of the right-hand side as arrays of their own (the result is
  `[Array<f32>; B]`). Lazy parts are computed inside the solver kernel, so no stack, no unstack and
  no evaluation of the parts: a step from the expressions to the solution is one launch. Falls back
  to the stacked solve when the expressions need more buffers than a kernel binds, for blocks of 6
  to 8, when the Thomas solve is chosen and its scratch exceeds a buffer, and when a part
  cannot be fused.
- `solve_tridiagonal_parts` and `_from`: up to 64 independent scalar systems on one grid in one
  launch, with lazy parts fused likewise. `solve_tridiagonal` already batched systems along an
  extra axis (`[2, ni, nj]` along axis 1 or 2); this is now documented.
- `examples/bench_solve.rs`: the block line solves against a hand-written cyclic reduction kernel
  (parallel cyclic reduction with a closed-form 4x4 inverse), and the solves on parts against the
  stacked forms.
- `SplitReason::Shared`: profiles list the cuts where a pending expression that other pending
  operations read too is stored instead of recomputed.

### Changed

- Fields that read each other at the element itself (`-a` next to `a`, a function of a field
  next to the field) are evaluated in one multi-output kernel by `Fields::cache`, `eval_all`
  and the other multi-array evaluations, also when their expressions read shifted ones. Only a
  field another reads through a view (slice, shift, gather) is still computed first. A record of
  stage data (several cached fields that depend on each other) took one kernel per dependency
  level before.
- `sum_axis`, `min_axis` and `max_axis` no longer evaluate a pending operand first: its expression
  is computed inside the first pass of the reduction (scans still evaluate it).
- The error of `slice_mut`, `flip_mut` or `transpose_mut` on a read-only view (an array made by
  `slice`, `flip` or `transpose`) says that the array is a view of another one and that the
  mutable view is taken on the array that owns the data, with the combined range.
- The cyclic-reduction line solve (one workgroup per line, chosen for few, long lines) is up to
  1.7x faster on blocks of 2 to 4: rows are held as `vec4`s in workgroup memory, the blocks of
  2 to 4 are inverted from their cofactors instead of by elimination (a zero diagonal needs no
  pivot; the block is scaled by its largest entry first, so any uniform magnitude solves), the `vec4`
  coefficient buffers are read whole, and a workgroup has one invocation per row (rounded up to a
  warp, capped at 256, or at 128 for blocks of 3 and 4 along a strided axis) instead of one per
  eight (blocks of 2: one per two rows). 4x4 blocks, 70 lines of 144 rows on an
  RTX 3090: 21 us to 13 us (a hand-written parallel cyclic reduction kernel takes 15 us). The
  scratch of a line that does not fit workgroup memory holds `vec4` rows.

### Removed

- The ArrayFire comparison benchmarks (`benchmarks/arrayfire`, the `luma-bench` crate).

### Fixed

- Debug builds could segfault in the Vulkan loader (`vkSetDebugUtilsObjectNameEXT`) when one
  thread opened a `Context` while another built kernels: the loader links a new device into a
  list it walks to name objects without holding its lock. Devices are opened without
  `VK_EXT_debug_utils` object names now; validation layers still report through their own output.

### Known limitations

- The kernels new in this release that use workgroup memory (`matmul`, sorts, the per-workgroup
  scatters, packed boolean stores without 32-wide subgroups) have not been run on RADV, whose
  results for workgroup-memory kernels are already known to be wrong (see 0.3.0).
- `bench_datasci --gate` compares wall-clock times when the device has no timestamp queries; that
  path has not been exercised (lavapipe has them).

## [0.3.0] - 2026-11-13

### Breaking

- Reductions (`sum`, `min`, `max`, `mean`, `norm`, `reduce`) are lazy and return a one-element
  `Array`; read it with `.item()?` (or `.to_array()?` for `reduce`). `[f32; 2]` / `[f32; 4]`
  reductions still return host values.
- Arrays broadcast per axis over extents of 1 (`[N, 1] + [1, M]` -> `[N, M]`, `select` branches,
  compound assignment), plus explicit `broadcast_to(shape)`; `&x / &x.sum()` stays on the GPU.

### Added

- Views that fuse into the reading kernel: `slice`, `shift_fill` / `shift_clamp` /
  `shift_mirror`, `pad` (`Edge`, `Mirror`, `Wrap`, `Zero`), `flip`, `transpose`, `index`, `take`,
  `concat` / `concat_slice`, `broadcast_to`, and `Array::copy` (`Fields::copy`).
- Mutable views (`ViewMut` via `slice_mut`, `flip_mut`, `transpose_mut`) with in-place `assign`
  (lazy when the base is pending) and compound assignment (`+=`, `-=`, ...), `put` (scatter),
  `luma::view` for strided custom-kernel reads, `LumaError::ReadOnly`, and `pub use encase`. Read
  views reject mutating operations instead of detaching.
- Records: `Fields` trait and the `luma::fields!` macro for structs of arrays,
  with arithmetic, `dot`, `copy`, `broadcast_to`, views and `ViewMut` (`assign`, `+=`) in one
  kernel for all fields.
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
- `stack` / `stack_slice`: equal-shaped arrays joined along a new axis in one kernel that computes
  one element of every part and writes the stacked layout (16 lazy expressions of 144x70 into
  `[16, 144, 70]`: one launch, the host cost of 16 views, 15 selects and a coordinate array gone).
- `is_finite`, `is_nan`, `is_inf` on `f32`, `f64` and `Df32` arrays (lazy `u32` masks, tested on
  the bits of the exponent through the guard the double-float library uses, so a device's
  fast math cannot fold the test of a computed value).
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

- `reshape` of a lazy view (slice, flip, transpose, pad, shift) stays lazy, a handle on the
  pending view as a clone is: `x.slice(0, k..k + 1).reshape([ni, nj])` reads `x` inside the kernel
  that uses it instead of evaluating the slice. Other lazy arrays are still evaluated by it.
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
- `iterate_all` register loops called again from the same place with the same step graph over
  other arrays reuse the earlier program (no flattening), and from the third call on the GPU
  starts before `step` is traced; the steps of a loop are cut into launches of about 40 ms by
  the cost model instead of a fixed work budget (32 lazy Keccak permutations over 2^20 states:
  one launch, 25.7 to 15.7 ms).
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
- Slicing, flipping or transposing a reshaped read-only view read the wrong elements.
- A deadlock in `item()` readbacks that evict under a memory budget while speculating.
- `hypot` with a NaN operand and a zero operand returned 0 instead of NaN.
- Region assigns into lazy expressions (the ghost cells and solid blocks of a finite-volume
  solver) made up to 30 times as many launches: the work of a shared expression was counted once
  per path through it (doubling with every level), so small ghost expressions were evaluated in
  kernels of their own, and a cut forced by the instruction cap inside another cut evaluated the
  arrays built on top of the expression piece by piece. One SSP-RK3 step with a solid block on a
  156x76 grid, single precision, no `cache()`: 89 launches / 320 us to 14 / 60 us.
- NVIDIA driver hang or crash in `exit`: luma waits for driver-started threads and ends the
  process with `_exit` if one stays busy (`LUMA_EXIT_WAIT=0` skips the wait).
- Cyclic-reduction index wrap above $2^{23}$ cells, pager self-deadlock when spilling during a
  traced `iterate*` step, `ReadOnly` alias checks on `+=`/`write`/mutable kernel args, `NaN`
  residuals falsely converging `iterate_until*`, `luma::fields!` record `+=` mutating shared buffers in
  place, `hypot` overflow and Metal `tanh` saturation above `|x| >= 44`, tiled `conv1d` out-of-window
  `Inf` contamination, and Linux `musl` builds.

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

[Unreleased]: https://github.com/flipnoteman/Luma/compare/v0.5.1...HEAD
[0.5.1]: https://github.com/flipnoteman/Luma/compare/v0.5.0...v0.5.1
[0.5.0]: https://github.com/flipnoteman/Luma/compare/v0.4.1...v0.5.0
[0.4.1]: https://github.com/flipnoteman/Luma/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/flipnoteman/Luma/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/flipnoteman/Luma/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/flipnoteman/Luma/releases/tag/v0.2.0
