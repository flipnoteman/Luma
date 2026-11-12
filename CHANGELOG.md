# Changelog

All notable changes to Luma are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Breaking

- Reductions are lazy: `sum`, `min`, `max`, `mean`, `norm` and `reduce` on scalar arrays return an
  `Array` of one element (`N` for `reduce`) instead of a host value, and no longer wait for the
  GPU. Read the value with `.item()?`, or `.to_array()?` for `reduce`:
  `x.sum()?` becomes `x.sum().item()?`, `let [lo, hi] = x.reduce([Min, Max])?` becomes
  `let [lo, hi] = x.reduce([Min, Max]).to_array()?`. The reductions of `[f32; 2]` / `[f32; 4]`
  arrays still return `Result` host values.
- An array of one element combines with an array of any shape: `&x / &x.sum()` normalizes on the GPU
  without a readback, and `&x.max() * &x` takes the shape of `x`. Shapes that differ, with neither
  a single element, are still a `ShapeMismatch`; a one-element array that is the target of `+=` is
  never broadcast.

### Added

- On-disk cache: the driver's compiled pipelines (wgpu `PipelineCache`, Vulkan) and the calibrated
  cost model are kept in `$XDG_CACHE_HOME/luma` (else `~/.cache/luma`; `LUMA_CACHE_DIR` moves it,
  `LUMA_CACHE=off` disables it), one file per adapter and driver, written atomically; a damaged
  file is ignored. A second process that builds the same kernels skips the driver's compile (the
  multi-second first call of large `Df32` kernels), and the calibration runs once per machine. A
  calibration worse than the profile's numbers by more than 4 times (a GPU shared with other
  processes) is measured once more and then dropped, neither used nor stored.
- `SplitReason::Materialised`: reported by profiling when the cost model evaluates a shared
  expression first because recomputing it at every read costs more than storing it.
- Workgroup staging of views: an expression read through slices, shifted slices or pads of a 1-D or
  2-D array at several offsets (a face flux read by the two cells that share it, a stencil of an
  edge-padded expression) can be computed once per cell of a workgroup tile in workgroup memory
  instead of once per read, including expressions that read other staged expressions. The cost
  model chooses between recomputing, staging and storing; `LUMA_STAGING=off` disables staging.
- Region `assign` into a lazy padded array no longer makes every reader recompute the ghost
  expressions: a `select` whose mask depends on the coordinates only is generated as `if`/`else`
  (the right-hand side is computed in the cells of the region, the old value elsewhere), and the
  cost model weights those operations by the share of cells that take the branch. A padded record
  with ghost expressions of any size read at four offsets in each direction is one launch without
  cuts (bowshock-lazy `luma_issues` case 1: 4 launches and 2 cuts at 100 ghost operations before).
- Workgroup staging follows sub-programs that are not staged: what a flux read once (or a slice of
  a slice) reads of a padded array is staged at the summed offsets instead of the padded array
  being recomputed at every read.
- Staged sub-programs of one level over the same region (the four fields of a flux) are lowered
  and computed together, so what they share (the wave speeds of a Riemann solver) is computed
  once per cell instead of once per field. bowshock-lazy `Staging::Fusion`, one `Df32` residual
  evaluation at 200x100: 308-344 us -> 139 us, f32 17-19 us -> 11 us (kernel time on an RTX 3090).
- The cost model searches for what to evaluate first instead of guessing: of the sub-programs a
  kernel reads at several indices it costs the groups that would be evaluated together (the fields
  of one state), each judged the same way, so the stage state of a time step that reads the stage
  before it through views is evaluated once instead of being recomputed by every cell (staged or
  not) that reads it; a program that fits no plan is ordered by what writing it out would cost
  rather than cut where the most instructions repeat. The fusion limit defaults to 4096
  operations (`LUMA_FUSION_MAX_NODES`), which the search makes safe. A pending node that two
  expressions read is only evaluated first if recomputing it costs more than a launch and a pass
  over its bytes, and then together with its siblings (the other fields of its record) and
  without the nodes below it that only it was reading. bowshock-lazy `Staging::Fusion` (no
  hand-placed `cache()`), 200x100, per SSP-RK3 step as `iterate_all` replays it: f32 76 launches /
  346 us -> 8 launches / 38-47 us GPU (hand-cached `Fluxes`: 12 / 53 us); steady us/iteration
  f32 252 -> 31 (`Fluxes` 46), `Df32` 1123 -> 536 (`Fluxes` 250). `euler/until_all` of `bench_pde` 454 -> 204 launches
  (2048x2048: 72 -> 18 ms), `muscl` 16-29% faster.
- `CostModel` and `Context::cost_model`: throughputs of the device (launch, memory bandwidth, f32,
  `Df32` and `f64` arithmetic), measured when a context opens (once per machine, see the cache;
  `ContextOptions::calibrate`, `LUMA_CALIBRATE=off` to use the profile's fixed numbers instead).
- Algebraic simplification when an operation is created: bit-exact rewrites always run (`x * 1`,
  `x / 1`, `x - 0`, `-(-x)`, `abs(abs(x))`, `floor(floor(x))`, `min(x, x)`, `select(m, a, a)`,
  `x < x`, `f32` through `f64`/`Df32` and back, and the `u32` identities), so `&x * 1.0` is one
  copy. `ContextOptions::fast_math` / `LUMA_FAST_MATH=1` also allows the ones that change values
  (`x + 0`, `x * 0`, `x - x`, `x / x`, constants of chained `*` / `+` combined, `x / c` as
  `x * (1 / c)`, `powf` with exponent 1, 2 or 0.5).
- Line primitives: `solve_tridiagonal` / `solve_block_tridiagonal` (block size 1 to 8, with
  `_from` variants where each line starts at its own cell) solve one system per line along an axis
  with the (block) Thomas algorithm; `Array::scan(axis, op)`, `cumsum`, `sum_axis`, `min_axis` and
  `max_axis` scan or reduce along an axis (the scan operator is an elementwise closure of two
  arguments, traced once and compiled into the scan kernels).
- Cyclic-reduction path of `solve_tridiagonal` / `solve_block_tridiagonal` (and the `_from`
  variants) for few, long lines: one workgroup solves a line in workgroup memory (rows beyond
  `Tuning::line_bytes` spill to a global scratch buffer) instead of one invocation walking it
  serially. Chosen automatically on NVIDIA GPUs when there are few lines (up to 8192 scalar lines
  along the contiguous axis, fewer for strided lines and bigger blocks); the Thomas solve stays
  for many lines and on other devices, where the choice is not measured. Same API;
  `LUMA_TRIDIAGONAL=thomas|cyclic` forces one. RTX 3090 (`bench_pde -- --only tri/few_lines`):
  200 lines of 4096 along axis 0 1.3 ms to 0.07 ms, along axis 1 2.0 to 0.12 ms; 100 lines of 1024
  with 4x4 blocks 1.7 to 0.26 ms (axis 1) and 1.5 to 0.29 ms (axis 2).
- `iterate_until_all(init, Until, step)`: loops a record of arrays until a stop rule holds, with the
  rule evaluated on the GPU between steps. `Until::new(max_steps)` takes `record(name, f)` (a
  one-element array computed from the state after every step, kept as history), `stop_when(f)`
  (a `Cond` built from `History`, `Series`, `Stat`: `last`, `first`, `peak`, `trough`,
  `window_max`/`window_min`, `step`, arithmetic, `abs`, `lt`/`le`/`gt`/`ge`, `and`/`or`/`not`, as in
  `(h["residual"].peak() / h["residual"].last()).ge(1e6)`) and `check_every`. The step also gets
  its 0-based index as a one-element array. `Run` has the final state, `steps`, the `StopReason`
  (`MaxSteps` or `Predicate(i)`) and `history(name)`, one value per step run.
- `Array::item()` (the host value of a one-element array) and `Array::to_array::<N>()`.
- `Array::sum_where(&mask)`: the sum of the elements where `mask` is non-zero, in one pass.
- `luma::Scalar`: a host number shared between clones, usable wherever a number is
  (`&x * &s`, `x += &s`). The value is read when the operation is recorded and is a kernel uniform,
  so changing it with `set` never recompiles.
- Views: `Array::slice(axis, range)`, `shift_fill`, `shift_clamp`, `shift_mirror` (shifts that do
  not wrap), `pad(widths, PadMode)` (`Edge`, `Mirror`, `Wrap`, `Zero`), `flip`, `transpose`,
  `index(axis)`, `take(&indices)` and `concat(axis, [..])`. A view moves no data: it is a lazy read
  through an index map that fuses into the expression using it, so `f.slice(0, 1..) - f.slice(0, ..n)`
  is one kernel.
- `Array::assign(rhs)` writes an expression into the array a slice, flip or transpose was taken
  from, in place, and `Array::put(&indices, values)` scatters. The expression may read the target
  (`w.slice(0, ..2).assign(w.slice(0, 2..4).flip(0) * -1.0)` fills ghost cells); passed by value it
  fuses into the write as one kernel. Both return `Result<()>`; `LumaError::ReadOnly` is new.
- `tanh`, `sign`, `floor`, `ceil`, `atan2`, `hypot`, `copysign` on `Array<f32>`, and `%` on
  `Array<u32>`.
- Records: the `Fields` trait and `#[derive(Fields)]` (new crate `luma-derive`, re-exported as
  `luma::Fields`) for a struct of arrays of one element type (`f32`, `Df32`, `f64`; the derive
  reads it from the fields, `#[fields(elem = T)]` sets it for a record of records), nested
  records, arrays and tuples. A record has `+ - * /` (with records, scalars and arrays), `-`,
  `dot`, `slice`, `pad`, `flip`, `transpose`, `shift*`, `take`, `select`, `assign` (one kernel for
  all fields), `put` and `eval`. `iterate_all` and `iterate_until_all` take states of any element
  type.
- `Array` is `Clone`: a clone is another handle on the same data (a write through one is seen by
  the other, as with `reshape`); `&a * 1.0` is an independent copy.
- `Array::iterate_until(max_steps, tol, step)`: runs `step` until the largest change of a step is
  at most `tol`. The convergence check runs on the GPU, and the host waits once per block of
  steps (16, doubling up to 128 while the loop runs on; `Until::check_every` fixes it for
  `iterate_until_all`).
  On an M-series Mac, 200 heat steps on 1024² take 11.8 ms, against 41 ms for 0.2.0's loop with a
  readback every step.
- `luma::fusion_limit(max_nodes, || ..)` sets the fusion limit of the expressions written inside
  the closure. `Array::cache()` and `Fields::cache()` evaluate an expression now, so that every
  later slice, shift or gather reads its buffer instead of recomputing it.
- `Profile::splits`: where a profiled expression was cut into several kernels, as `FusionSplit`
  entries with a `SplitReason` (operation count, nesting depth, input buffers, live values) and
  the source location of the operation that caused the cut; `Display` lists them.
- `LUMA_FUSION_MAX_LIVE` overrides the most values a fused kernel keeps live at once (see below).
- `LUMA_SPECULATE=off` turns off speculation at readbacks (see below).
- Benchmarks: `per_step` now calls `eval` every step, like the ArrayFire side. Also new: a
  Luma-only `lazy_chain` variant, `_cold` rows that time the first run with compilation, a
  `converge` workload with a readback every step, and a tuned ArrayFire run in `run.sh`
  (`AF_*_MAX_JIT_LEN=256`, `--label`).
- Extended precision. `f64` arrays (`Array<f64>`; Vulkan devices with 64-bit shader floats, else the
  result is poisoned with `LumaError::Unsupported`) have `+ - * /`, `sqrt`, `abs`, `sign`, `floor`,
  `ceil`, comparisons, `minimum`/`maximum`, `select`, views, `assign` and reductions;
  everything else on them (`exp`, trigonometry, generators, ...) is refused with `Unsupported`.
  There is no scalar `Operand<f64>` for comparisons, `minimum`/`maximum` and `select` (a second float
  type there would make `select(&mask, 1.0, 0.0)` infer `f64`); the operators take `f64` scalars.
  `Df32` (`{ hi, lo }`, about 48 bits, computed in `f32` arithmetic that no compiler can
  reassociate, on every device) has `+ - * /`, `sqrt`, `abs`, comparisons, `minimum`/`maximum`,
  `select`, views and reductions. Casts: `to_f64` / `to_df32` on `f32`, `to_f32` / `to_df32` on
  `f64`, `to_f32` / `to_f64` on `Df32`. `Array<f32>::sum_compensated()` and `norm_compensated()`
  accumulate in double-float, so the result does not depend on the reduction tree. Programs
  that mix 4- and 8-byte elements (including the casts) on arrays split over several buffers
  fail with `LumaError::Chunked`; scans and reductions along an axis are not available for them.
- `ContextOptions::check_finite(true)` / `LUMA_CHECK_FINITE=1`: every array an operation computes
  is checked for NaN and Inf right after its kernel, and the operation fails with
  `LumaError::NonFinite`, naming it and its source location (it waits for the GPU once per
  evaluated array: a debug mode). `LumaError::Unsupported` is new.
- `examples/bench_threads`: threads that share one context against threads with a context each, for
  `iterate_until_all`, `iterate_all` and an `eval` loop.

### Changed

- A scalar reduction followed by elementwise operations with scalars on its one-element result
  (`x.norm()`, `x.mean()`, `(x.sum() / n).sqrt()`; up to four operations on `f32`) finishes inside
  the last pass of the reduction instead of launching an elementwise kernel for the result: the
  norm of a large array is 2 launches (first pass, one pass over the partial results), of a
  small one 1 (was 3 and 2). The result stays a lazy one-element array that other expressions
  broadcast. A reduction of an `f32` array that is already evaluated is now lazy too (it records
  when its result is used, like the reduction of an expression; a write into the array before
  that evaluates a copy first). If a reduction and a step on it are both used, the reduction
  runs twice unless it ran first.
- Building a lazy expression costs less host time: a chain of a hundred operations over an old
  operand takes a third fewer instructions (`cargo run --release --example bench_build`), the
  solver's `eval` step about 8% fewer in `jit::create`. The sets of nodes below an expression keep
  only the words in use when ids are spread widely (instead of listing every node), the sharing
  table holds hashes and checks the operands on a hit, a new node's weak references are made once,
  and the simplifier reads an operand's operands only for operations that have rewrites.
- Fusion decisions use the cost model: a kernel that reads an expression through views at several
  indices (slices, pads, flips, `take`) recomputes it at every read, or evaluates it first and
  reads it back, whichever the device's `CostModel` estimates to be cheaper (evaluating first must
  be estimated 25% cheaper, and recomputes what it shares with the rest of the program instead of
  evaluating that too). The old rule always recomputed within the instruction cap, which stays as
  a size bound. Circular shifts accept up
  to eight lazy readers of a cheap expression instead of two, and fewer of an expensive one.
- `iterate` and `iterate_all` trace their step once and replay the recorded kernels for steps that
  do not fuse into one kernel (shifts, `cache()`, `assign`, reductions), instead of calling `step`
  once per step. `step` is now called exactly once, and must compute its results from its argument
  and captured arrays only. A step that reads a value back, or states that span several buffers,
  run step by step as before.
- Fusion scales to large expressions: a kernel now holds up to 1024 operations by default
  (`LUMA_FUSION_MAX_NODES`, `fusion_limit`; at most 4096), where it held 128. Nesting is cut at
  256 levels, which bounds the recursion on long chains, and a program that would keep more values
  live than the GPU profile suits (96, 64 on Apple, 256 on the CPU device) is split in the middle
  where fewest values are live.
- Identical lazy operations on identical operands (`spectral(&w, ..)` called twice) are one node,
  which a fused kernel computes once. The sharing ends when a buffer is written.
- A slice, pad, flip, transpose, gather or `concat` of a lazy expression reads it lazily: nested
  views are unfolded into the reading kernel, each distinct (expression, index) pair computed once,
  instead of evaluating the expression into a buffer. A kernel that would grow beyond
  `LUMA_FUSION_MAX_INLINED` operations evaluates its most repeated view first (a split note
  `Inlined` in profiles). A first-order Euler step written with records, `pad`, a ghost `assign`
  and slices is one kernel per step. RTX 3090, 50 steps at 2048² (`examples/bench_pde.rs`):
  `iterate_all` 40.8 ms (6 kernels per step) to 8.4 ms (1), within 1.2x of a hand-written kernel
  (8.5 ms); with `eval` every step 51.8 ms to 9.8 ms; MUSCL, 20 steps, 16.9 ms to 6.6 ms. A lazy
  Mandelbrot chain of 256 iterations at 4096² evaluated once at the end: 1.99 s to 9.4 ms.
- `assign` through a slice, flip or transpose of an array that is still a lazy expression
  launches nothing: the array becomes `select(region, rhs, old)`, and fuses into what reads it.
  Arrays and slices recorded before the write keep the old values.
- Scans and reductions along an axis are restructured for coalesced memory access (workgroup scans
  on contiguous lines, line-fastest order on strided ones): `cumsum` of 2^26 elements 29 ms to
  1.0 ms, along axis 0 of 8192² 18.9 ms to 1.0 ms, `sum_axis(0)` 2.4 ms to 0.35 ms. Block
  tridiagonal solves are straight-line code per block size up to 5 (4x4 blocks on 1024² lines:
  31.6 ms to 3.1 ms), and the scalar solve along the contiguous axis picks its workgroup size from
  the number of lines (4096²: 9.0 ms to 5.0 ms).
- `iterate_until` and `iterate_until_all` check the stop rule in blocks that grow from 16 to 128
  steps while the loop runs on (64² state: 41.5 to 14.1 us per step).
- Less host time per lazy operation and evaluation: node id sets as bit sets, hash-consing keys
  and kernel keys hashed once and interned, no copies of the live-node list (an Euler step graph:
  127 to 66 us to build).
- The crate is a workspace with `luma-derive`; releases publish both crates.
- Loops of `x = step(&x); x.eval()` (or `eval_all`) over `f32` arrays are recognized. The
  postponed steps then run together, as one register loop, one time-blocked stencil, or step by
  step, instead of chaining into ever longer expressions. Kernels no longer depend on how many
  steps were postponed. On an M-series Mac against 0.2.0: Mandelbrot 1024², 256 iterations with
  `eval_all` each, 5.6 ms to 1.4 ms (`iterate_all`: 0.83 ms); heat 4096², 100 steps with `eval`
  each, 33 ms to 21.5 ms (`iterate`: 20.9 ms).
- A readback in a loop of `next = step(&u); err = (&next - &u).abs().max().item()?; u = next` computes
  the next steps and their residuals ahead, once the step has repeated. The readbacks that follow
  are answered from that result, as long as they reduce the same program over the predicted states
  and nothing they read was written in between. The heat `converge` loop at 256², 200 steps:
  35 ms to 5.1 ms.
- `eval` stops postponing work once resident array data reaches three quarters of the memory
  budget.
- A lazy expression read through small shifts (up to 2 cells, 1D and 2D arrays) is computed once
  per cell of a tile in workgroup memory, and the shifts read it there, instead of being
  recomputed by every shift; a direct read of it at the cell itself takes the same value. Up to
  three stencil steps chained without `eval` become one kernel. 100 steps chained without
  `eval` on an M-series Mac, against 0.2.0: Game of Life 1024² 2.9 ms to 2.1 ms, 4096² 53 ms to
  24 ms, 8192² 222 ms to 96 ms; heat 1024² 2.0 ms to 1.6 ms, 8192² 112 ms to 78 ms.
- For other shifts, how much of a lazy expression a shift recomputes instead of evaluating it
  first is now set per GPU profile: 24 operations on Apple and generic GPUs, 32 on NVIDIA. It
  was 16 everywhere.

### Fixed

- Two independently written equal expressions shared one node, so `assign`, `write` or `+=`
  through one of them changed the other.
- A lazy reduction (`sum`, `min`, `max`, `reduce`) of a pending expression recorded before a write
  into the reduced array, or into an array the expression reads, saw the written values; it keeps
  the values it was recorded against, like every other lazy array.
- A `Df32` kernel with only comparisons, `abs`, `minimum` or `maximum` failed to bind its uniform;
  casts from and to `Df32` could be cancelled by the shader compiler
  (`(&x - x.to_f32().to_df32()).to_f32()` was 0).
- After `cache()`, the buffers below the cached part still counted towards the input limit of a
  kernel, which cut expressions that fit.
- Threads sharing a context (`default_context()`, parallel tests) got wrong results: `eval`,
  `flush`, `sync` and `cache` evaluated the pending arrays of other threads (and fed them into a
  recording another thread had under way), and equal expressions were shared across threads. Each
  thread now only evaluates its own pending arrays.
- Threads sharing a context ran up to ten times slower than threads with a context each (the
  accuracy suite of bowshock-lazy, four threads on `default_context()`: 118 s against 11 s). A
  kernel that one thread was compiling, which takes seconds for a large `Df32` kernel, blocked every
  other thread's kernel lookups, and only one thread at a time could trace a loop of `iterate_all` /
  `iterate_until_all`: the others ran theirs on the host, calling `step` and reading the state back
  at every step. Kernels now compile outside the lock of the cache (threads that need different
  kernels compile at the same time, one that needs a kernel somebody is compiling waits for that
  compile), and every thread records its own loop. The history `eval` loops are recognised from
  is per thread too: the calls of threads that alternate no longer reset each other's, which left
  one kernel per step.
- wgpu's validation of indirect dispatches (an extra dispatch before every step of
  `iterate_until`/`iterate_until_all`) is off: the dispatch arguments are always written by the
  library's own gate kernels.
- `x -= &x` with fast math and `x *= 0` on integers failed with "in-place assignment cannot run in
  place": simplification removed the read of the target, and the compound assignment now runs as
  a plain store into it.
- A kernel whose only scalars are the amounts of circular shifts of a lazy expression
  (`n + n.shift(-4)` with the fusion limit cutting the expression) failed with a bind group larger
  than its layout: the staged kernel bakes the amounts in, so it has no uniform to bind.
- `iterate_all` (and `iterate_until_all`) corrupted arrays created before the loop: a lazy
  expression of such an array that the traced step evaluated (a gather from a `concat`, a scan of
  a `x *= c` result) ran inside the recording and took over its operand's buffer, so every
  replayed step overwrote it again (`x *= -2` was applied once per step). Tracing now evaluates
  the live lazy arrays first.
- `assign` through a view that was evaluated (`v.eval()`, or evaluated because the right-hand side
  read it or an earlier write into the base flushed it) left the receiver showing the data from
  before the write; the receiver now denotes its region of the written base, as documented. This
  also fixes `v.assign(&v.shift(3))`, whose receiver kept the old data.
- A slice write into a lazy array after a flipped (or transposed) write into it was lost: the
  claim an array has on its expression could not be shared again once the arrays holding it had
  all been dropped (a temporary `r.flip(0)` that was assigned through).
- A view taken from an array that was a view evaluated before a write into its base read the
  written base instead of the evaluated array (the old data it was recorded against).
- A write through a view of a lazy expression left the view showing the old data when the view
  had been evaluated before the write (by `eval`, or by `cache` of an array that shares work with
  it); it now shows the written data.
- `iterate`, `iterate_all` and `iterate_until_all` could still corrupt an array created before the
  loop when threads shared a context (wrong values, `inf`). An array that had only been computed
  inside the kernel of another one (the last step of an `eval` loop that a busy GPU postponed,
  then read by the next expression) was left lazy when the loop was traced, and the recording
  evaluated it in place over its operand's buffer, so that every replayed step overwrote it
  again. Tracing now evaluates every array the thread holds first.
- `x += c` (and the other compound assignments) on an evaluated lazy array whose expression had
  been replaced by a write through a view (`x.slice(..).assign(..)`) rebound `x` instead of
  changing the buffer it shares with its clones, so the clones did not show the update.
- `x.flip(0).assign(&m)` where `m` simplified to a copy of `x` (`m = x.minimum(&x)`) could read
  elements of `x` after the write had overwritten them: the store of the assignment was folded
  into the copy and no longer had it evaluated first. A copy is no longer folded into the copy it
  reads.
- An array sharing the buffer of a view that was evaluated before a write through the view (the
  result of `v.cache()`) kept the old data: the write gave the view a replacement expression and
  left the buffer alone. The buffer is now overwritten with the written data.
- A write through a view taken from a lazy array after a lazy slice write into it was lost when
  the array was then updated with `x /= c` (or another compound assignment): the update rebound
  the array to a new expression the view did not write into. An array a view of its replaced
  expression was taken from is now updated in place.

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
