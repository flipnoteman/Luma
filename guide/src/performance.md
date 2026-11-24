# Performance guide

Luma is fast when expressions fuse, the host does not wait on the GPU, and transfers are rare. This
chapter shows how to see what Luma did, how it decides between plans, what cuts a kernel, what is
cached, and a checklist for slow programs.

## See what happened: `Profile`

[`Context::profile`](https://docs.rs/luma/latest/luma/struct.Context.html#method.profile) runs a
closure and returns the GPU time of every kernel launched in it, by label, and the places where an
expression was cut into several kernels. It needs timestamp queries on the adapter
(`ctx.profiling_supported()`).

```rust
use luma::*;

let ctx = Context::new()?;
let x = Array::<f32>::full_on(&ctx, 1 << 20, 2.0)?;

if ctx.profiling_supported() {
    let (values, profile) = ctx.profile(|| ((&x * 3.0 + 1.0).sqrt()).to_vec())?;
    assert_eq!(values?.len(), 1 << 20);
    assert_eq!(profile.launches(), 1);                  // the chain is one fused kernel
    println!("{profile}");                              // kernel, launches, total/mean/min/max us
    let fused = &profile.kernels[0];                    // labelled by the operations it holds
    assert!(fused.label.starts_with("Fused"));
}
# Ok::<(), luma::LumaError>(())
```

A [`Profile`](https://docs.rs/luma/latest/luma/struct.Profile.html) has `kernels` (one
[`KernelTiming`](https://docs.rs/luma/latest/luma/struct.KernelTiming.html) per label, the most
expensive first: label, launches, total, min, max, and a `mean()` method), `decisions`, `cuts()`, `launches()` and `total()`.
Fused kernels are labelled by what they hold (`Fused(Mul, Add, Sqrt) f32`, `FusedReduce(Sum: Mul)
f32`); your own kernels by `.label(..)` or their entry point.

While profiling, every launch gets a compute pass of its own so that it can be timed. Use it to find
where the GPU time goes, not to measure throughput. For the host's time, time the calls themselves.

### Why was it cut: `cuts()` and `decisions`

`Profile::cuts()` gives the cuts of expressions in a profiled region: the operation and
source line where an expression was cut, and the
[`SplitReason`](https://docs.rs/luma/latest/luma/enum.SplitReason.html). Every choice the planner made
while profiling
([`Decision`](https://docs.rs/luma/latest/luma/struct.Decision.html): the operation, its source line,
the [`DecisionSite`](https://docs.rs/luma/latest/luma/enum.DecisionSite.html), the chosen label and the
alternatives that lost) is returned by `Context::take_decisions()` right after
`Context::finish_profiling()` (or `Context::profile`). To record decisions without
profiling (host paths, the plan cache and graph capture stay on), use `Context::record_decisions(true)`
(or `LUMA_RECORD_DECISIONS=1`) and `Context::take_decisions()`.

| Reason | Meaning |
| --- | --- |
| `Nodes` | more operations than the fusion limit allows in one kernel |
| `Depth` | nested deeper than the engine walks in one kernel |
| `Inputs` | more distinct buffers than a kernel can bind |
| `Live` | too many values live at once (register pressure) |
| `Scalars` | the scalar operands do not fit one uniform buffer |
| `Inlined` | views of lazy expressions unfolded into more instructions than the device's cap |
| `Materialised` | the [cost model](#the-cost-model) estimates that storing an expression once is cheaper than recomputing it at each read |
| `Shared` | an expression that other pending operations outside the kernel read too is stored once instead of recomputed in each |

```rust
use luma::*;

let ctx = Context::new()?;
let x = Array::<f32>::full_on(&ctx, 1024, 1.0)?;

if ctx.profiling_supported() {
    let (r, profile) = ctx.profile(|| {
        // `fusion_limit` caps the operations per kernel for the expressions written inside.
        let y = fusion_limit(2, || (((&x + 1.0) * 2.0 - 3.0) / 4.0).sqrt());
        y.to_vec()
    })?;
    r?;
    assert!(profile.launches() > 1);
    for (op, location, reason) in profile.cuts() {
        println!("{op} at {location}: {reason:?}");     // add at file:line:col: Nodes { nodes: 3, limit: 2 }
    }
    assert!(profile.cuts().any(|(_, _, reason)| matches!(reason, SplitReason::Nodes { .. })));
}
# Ok::<(), luma::LumaError>(())
```

Counting `profile.launches()` before and after a change is the quickest check that something
still fuses; the repository's tests do exactly that.

## The cost model

Some choices are not a matter of limits but of cost. A view of a lazy expression read at three
offsets could be:

1. **recomputed** at each read: no extra memory traffic, three times the arithmetic;
2. **staged**: computed once per cell of a workgroup tile into workgroup memory, and read from
   there (available for expressions read through plain translations, such as slices and the
   slices of a padded array: the flux that two neighbouring cells read);
3. **materialised**: written to a buffer by one kernel and read back by another.

Luma picks by estimated time. [`CostModel`](https://docs.rs/luma/latest/luma/struct.CostModel.html)
holds the device's throughputs: the cost of a launch (`launch_us`), memory bandwidth
(`bandwidth_gbs`), simple `f32` operations per nanosecond (`f32_ops_per_ns`), and the cost of a
`Df32` or `f64` add or multiply in units of an `f32` one (`df32_factor`, `f64_factor`). Every
operation has a weight (a divide counts 4, an `exp` 8, a view read 6, a `Df32` multiply
`df32_factor` times the `f32` one). A kernel's estimated time is a launch plus the larger of its
memory time and its arithmetic time. The model chooses the cheapest of the three plans when it
estimates a clear gain (staging must beat recomputing by 10%, materialising by 25%), and the
caps below still bound everything.

```rust
use luma::*;

// Measured when the context opens (about 150 ms on a GPU, once per machine: the result is cached).
let ctx = Context::new()?;
let model = ctx.cost_model();
assert!(model.bandwidth_gbs > 0.0 && model.f32_ops_per_ns > 0.0 && model.launch_us > 0.0);

// Or fixed numbers for the kind of GPU, with no measurement:
let fixed = Context::with_options(ContextOptions::default().calibrate(false))?;
assert_eq!(fixed.cost_model(), CostModel::fallback(fixed.tuning_profile()));
# Ok::<(), luma::LumaError>(())
```

`LUMA_CALIBRATE=0` (or `.calibrate(false)`) skips the measurement. A clean result is kept in the
cache directory, a measurement that looks disturbed by other processes is dropped, and then the fixed
numbers are used. `LUMA_STAGING=off` turns staging off; `Context::cost_model` lets you see what the
decisions are based on. The model is not a tuning knob you can set: if a decision is wrong on your
hardware, `x.cache()` forces a materialisation, and `LUMA_STAGING=off` removes the staged plan (the
model still chooses between recomputing and materialising).

### Reading a source through a texture

The same model decides where a kernel reads a two-dimensional source (`jit::cost::read_path`): from
the buffer (on NVIDIA GPUs through the read-only data cache, which a `read` binding gets on its own)
or through the device's texture units, as exact gathers. A texture is a view of the source: on CUDA the
buffer itself as a pitched texture (no copy when its base is 512-byte aligned and its rows a multiple of
32 bytes apart, kept with the array while its buffer stays where it is; otherwise one copy into a CUDA
array), on wgpu a copy of the image (a read and a write of it and a launch, made per use). The saving
is a share of the kernel's time that the gathers take off (per pattern, measured on an RTX 3090 and an RTX
3080 Laptop: bicubic warps and scattered bicubic positions about 25 %, bilinear warps 5-13 %, none for
`resize`, smooth positions or scattered bilinear positions: those kernels are bound by DRAM), and a
texture is read only where the saving is bigger than what the view costs (nothing for a view of the
buffer itself; for a copy of the image on wgpu, 0.4 ms at 1080p, which no measured kernel recovers).
The decision is made for the NVIDIA tuning profile, where it was measured. The kernels the fusion planner
builds never read through a texture: gathers from a 2-D array at computed or data-dependent coordinates
(single `take`s, lookup tables, four-tap bilinear footprints written as array operations, clamped
windows) ran at 0.84-1.0 of the buffer's speed in texture reads on both GPUs, because they are bound by
DRAM or by index arithmetic, and the planner sees flat indices, not a footprint a gather could take. So
on wgpu, Metal (which has no textures) and the CPU backend the sources are read from buffers, and on CUDA
the kernels above are measurably faster; `Profile` shows no extra launch. The one case where luma reads a
texture without the model is the opt-in `Interp::LinearHardware`, the texture unit's own bilinear filter
([Interpolation](signal.md#reading-a-source-through-the-texture-units)); luma never chooses it.

The model also sizes loops: [`iterate_all`](loops.md#launch-sizing) cuts long register loops into
launches of about 40 ms of GPU time by it, and it decides when a circular `shift` of an expression
is recomputed, staged, or evaluated first.

## What fuses, and what cuts

Fused: elementwise operations, comparisons, `select`, casts, generators, views, compound
assignment, the input and the epilogue of a reduction ([Lazy evaluation](lazy.md#what-fuses)),
and, in the primitives below, the expressions they read.

### Primitives that read an expression

These need a kernel of their own, but a pending operand is not stored first: its expression
becomes the load of that kernel.

| Primitive | Computed inside the kernel | When it runs |
| --- | --- | --- |
| `matmul` | each pending operand (a transpose included) is computed in the tile loads of the shader kernels; the CUDA backend's own `f32` kernels and the CPU backend's native product evaluate a lazy operand first, once, and read transposes of stored arrays in place | eager: the product is recorded when called and returned evaluated, so what reads it is another kernel |
| `sum_axis`, `min_axis`, `max_axis`, `reduce_axis`, `mean_axis`, `var_axis`, `std_axis`, `argmax_axis`, `argmin_axis` | the operand, in the first pass over the data | `f32` on a GPU backend: pending until used, then recorded with the other pending axis reductions over the same lines in one kernel (up to 8 ops; on wgpu an axis reduction of fewer than 524288 cells is recorded when called); otherwise eager: the result is evaluated when the call returns. `var_axis` and `std_axis` are one pass (sum and sum of squares of the line minus its first element) |
| `sum`, `min`, `max`, `prod`, `mean`, `sum_where`, ... (whole-array reductions) | the operand, in the first pass | `f32`, `u32` and `i32`: pending until its value is read or used, then recorded together with the other pending reductions of the same length that read what it reads (or are small), up to 16 accumulators per kernel (`LUMA_MULTI_REDUCE=off` or a number); each keeps its own combine order, so the value is the one it has alone. On CUDA a group is recorded reduction by reduction until its structure has been seen 4 times (`LUMA_MULTI_REDUCE_AFTER`), while the merged kernel compiles in the background |
| `solve_tridiagonal`, `solve_tridiagonal_parts`, `solve_block_tridiagonal_parts` | each pending block/vector entry is computed where a cell's system is loaded | eager: the solution is evaluated when the call returns |
| `put_add`, `put_min`, `put_max`, `bincount`, `histogram`, `group_*` | the value and the index (or code, or bucket) expressions of the scatter | an in-place write, like `put` |
| `sort`, `argsort`, `sort_axis` and what builds on them (`quantile`, `quantiles`, `median`, `factorize`, `unique`) | the key expression, in the first pass of the radix sort (`sort_axis` of lines of up to 2048 elements evaluates its operand first) | eager: the sorted array is evaluated; `factorize` and `unique` also read the number of distinct values back, one host wait |

An operand that reads circular `shift`s of other expressions, or lives in several buffers, is
evaluated first instead. `scan` and `cumsum` always evaluate their operand (they read it twice; on CUDA, and on the native CPU backend with `LUMA_LOOKBACK_SCAN=on`, integer and `min` / `max` scans of long contiguous lines read it once).
Because these results are evaluated at the call, a loop that calls them every step builds one
kernel per call rather than one fused chain: count launches with `ctx.profile(..)`. Usage is in
[Matrices, statistics and grouping](data.md): [`matmul`](data.md#matrix-products), [axis statistics](data.md#statistics-along-an-axis), [scatters](data.md#accumulating-scatters), [sorting](data.md#sorting).

An expression is cut when it passes a limit. They are generous, and a cut costs a launch and a trip
through memory, not correctness:

| Limit | Default | Setting |
| --- | --- | --- |
| operations per kernel | 16384 | `LUMA_FUSION_MAX_NODES`, or [`fusion_limit`](https://docs.rs/luma/latest/luma/fn.fusion_limit.html) for expressions written in a scope |
| nesting depth | 512 (256 with debug assertions) | fixed (the walk is recursive) |
| values live at once | by GPU profile | `LUMA_FUSION_MAX_LIVE` |
| instructions after views of lazy expressions are unfolded | 1536 on the NVIDIA profile, 1024 elsewhere (other GPUs and the CPU backend) | `LUMA_FUSION_MAX_INLINED` |
| input buffers per kernel | the device's storage-buffer limit minus 3, at most 256 | `LUMA_FUSION_MAX_INPUTS` lowers it (to test a device with fewer) |
| scalar operands | one uniform buffer | fixed |

When a limit is hit, the biggest operand is evaluated first (together with the live arrays that
share work with it), and the rest is flattened again.

### Many arrays in one kernel

A kernel that reads many different arrays (the 50 words of a Keccak state) meets the device's
binding limit: 16 on many integrated GPUs via WebGPU, 29 on wgpu Metal, and 256 on native CUDA, on the native CPU backend, on native Metal with tier-2 argument buffers (28 without them) and on discrete GPUs.
Cutting such an expression on the limit alone would evaluate it a few operations at a time, because
every round reads every array. Luma instead puts the arrays into one buffer. When a kernel would bind
more arrays than the device allows, the evaluated arrays of one length and type it reads are copied
into one buffer (a few small kernels) and become windows of it, and a kernel with more outputs than
bindings writes them into one buffer and returns them as windows. A kernel reads all the windows of
one buffer through one binding. The arrays are the same arrays afterwards: a write or a custom kernel
that takes one gives it a buffer of its own (a copy of its range), and expressions recorded before the
write still see the old contents. Three permutations of the lazy Keccak (`examples/keccak.rs`)
take 7120 kernels with 16 bindings without this, 39 with it, and 9 on a device with many bindings.

Packing costs one copy of the arrays it moves, so it only happens when the binding limit would force
a cut, and an array of a window is kept alive by the buffer it shares while any other window of it
lives. Arrays read through a view or a circular `shift` are not read through the shared binding
(they get a buffer of their own when the view is made). `ctx.profile(..)` shows the copies as
`FusedStack(Id, ...)` kernels; `LUMA_FUSION_MAX_INPUTS=13` reproduces a 16-binding device on any GPU.

Never fused, because they need whole-array structure: `scan`, stacked block-tridiagonal solves (`solve_block_tridiagonal`), `conv1d` / `conv1d_fft`, and custom kernels. They read computed arrays.

## Caches

- **Kernels** are cached per context by the *structure* of the expression: operations, types, how
  values are wired. Scalar values are uniforms, so `&x * 2.0` and `&x * 3.0` share one pipeline,
  and an expression with the same shape built again costs the host only the flattening. Identical
  operations on the same operands are one node (hash-consing).
- **Compiled pipelines** are cached on disk (CUDA PTX, the native CPU backend's compiled machine code, and wgpu's Vulkan `PipelineCache`) per adapter and
  driver; the native Metal backend keeps none between processes. Repeated runs skip the multi-second compile of large (`Df32`) kernels. The measured cost model is kept next to it.
  `LUMA_CACHE=off` disables both; `LUMA_CACHE_DIR` moves them. A damaged file is ignored. The CPU blobs and the
  CUDA PTX are also keyed by a hash of the sources that generate them (the code generators, the CPU
  math-table ABI, the packet math routines, the planners), so a build whose code generation differs from the
  one that wrote a cached kernel never loads it; the first run after an upgrade compiles again. The cache
  files of the wgpu backend have format number 2, so files written by earlier versions are read as absent
  and rewritten once.
- **Loops**: an `iterate_all` called again from the same place with the same step graph reuses the
  earlier program, and after two such calls the GPU starts before the step is traced
  ([Loops](loops.md#calling-it-again)).
- **Plans of eager evaluation**: an `eval` whose pending graph has the structure of an earlier one runs
  that evaluation's recorded launches over the new buffers and scalars (128 plans per context;
  `LUMA_PLAN_CACHE=off` disables it, results are bit-identical either way,
  [Loops](loops.md#loops-with-eval)).
- **Buffers**: freed arrays go to a pool of idle buffers (up to half of the memory budget), so a
  program that cycles through gigabytes of temporaries reuses them instead of allocating again.

## Loops

How you write the loop decides how much host time each step costs.

| Loop | Host per step | Notes |
| --- | --- | --- |
| `iterate_all` of an elementwise `f32` or `u32` step | none | one kernel, state in registers, launches of about 40 ms |
| `iterate` of a 1-D or 2-D stencil | none | time-blocked: several steps per pass over memory |
| `iterate_until`, `iterate_until_all` | none | the stop rule runs on the GPU; one wait per block of steps; a block is one graph replay on CUDA, Metal and the CPU backend |
| `eval` every step | building the step | recognised as a loop for `f32` steps (a register loop, or a stencil of up to four arrays evaluated together); never waits; a step of the same structure as an earlier one skips planning (the plan cache, `LUMA_PLAN_CACHE`) |
| readback every step (`max().item()?`) | a GPU round trip | speculation hides it when the step repeats |

Prefer the primitives when the step fits them ([Loops](loops.md)). Otherwise `eval` each step
instead of reading, and `sync` once at the end. A readback in the loop (a convergence test) is the
most expensive way: replace it with `iterate_until_all`, or test every k steps.

### The host cost of an eager step

A loop that records a step and evaluates it every iteration (`u = step(&u); u.eval()?`) spends its host
time in building the graph, one node per operation, and in planning it. Two things bound it. The plan
cache ([Loops](loops.md#loops-with-eval)) skips flattening, cutting and judging when the pending graph
has the structure of an earlier evaluation; on an RTX 3090 (CUDA) the per-step cases of the `cfd`
benchmarks went from 6.60 to 4.33 ms (`euler/3d_per_step`) and from 3.49 to 2.44 ms
(`muscl/3d_per_step`), and the graph building that remains is 160-190 us of the 200-240 us of a 3-D step.
`LUMA_PLAN_CACHE=off` turns the cache off. Building itself costs a few thousand instructions per node:
operations are passed by reference, the table that makes equal operations one node and the list of live
nodes are per thread, the id set every node keeps of the expression below it is merged with POPCNT / AVX2
loops chosen at run time, and the buffer of that set is reused by the next node. An unrolled expression
of about 10 000 operations (24 Keccak rounds written out) takes 3.4 ms of host time to record for 4096
messages on an RTX 3090. A step of thousands of operations still pays for each of them every step, so the
loop primitives, which trace the step once, are the cheaper form where they fit.

### Matrix products

On a CUDA device with tensor cores (compute capability 8.0 and up) `matmul` runs in TF32 by default: the
operands are rounded to 10 bits of mantissa and the products accumulated in `f32`, so results differ from
the exact `f32` product (each entry within 1.5e-3 of the sum of the absolute values of its terms). On an
RTX 3090 a 4096 x 4096 x 4096 product runs at 34 TFLOP/s instead of 20. `ContextOptions::numerics(Numerics::strict())`
or `LUMA_NUMERICS=strict` turns it off for a context, `matmul_f32` is the exact product of one call, and
`matmul_tf32` forces TF32 ([Tensor cores](data.md#tensor-cores-tf32)). `matmul_batched` is always `f32`.
Other backends compute `f32` products as before. A product whose output is small and whose `K` is long (a
Gram matrix, a gradient `Xᵀr`) runs as a kernel family with no workgroup staging, and `matmul_nt`,
`matmul_tn` and `matmul_tt` read a transposed operand in place instead of copying it.

### The CPU backend

The native `cpu` backend runs matrix products as multithreaded Rust (`matrixmultiply`, AVX-512, AVX2 + FMA
or NEON microkernels chosen at run time) instead of compiled tile kernels: a 4096-cube product goes from
64 to about 1100 GFLOP/s on a 32-thread Ryzen 9 7950X. `exp`, `ln`, `exp2`, `log2`, `sin`, `cos` and `tanh`
of `f32` run as one call per 4-lane packet on x86-64 Unix (at most 2 ulp over the whole `f32` range;
other targets keep the lanewise `libm` calls), and the line solves (`solve_tridiagonal*`,
`solve_block_tridiagonal*`) are native Rust. A launch too small to pay for a hand-off runs on the calling
thread, and idle workers poll for 200 microseconds before they sleep (`LUMA_CPU_SPIN_US`, `0` to sleep at
once). Its kernels compile on a pool of threads (`LUMA_COMPILE_THREADS`, shared with the CUDA backend).

## Transfers

The GPU is fast, the way to it is not. Data crosses it at `Array::from_slice` (upload),
`to_vec` / `read_into` / `item` (readback, which also waits), and `write`.

- Keep data on the device: chain operations, and read back the final result and small scalars.
  `item()` of a reduction moves 4 bytes; `to_vec()` of the field moves all of it.
- Use `read_into(&mut buf)` for repeated readbacks into a buffer you own.
- Upload once. A uniform scalar changes without recompiling anything, a new array costs an upload.
- On unified memory, `from_slice` uploads into mappable buffers that are reused (zero-copy).
- Arrays beyond the memory budget move to host memory and back on demand (once-only warning). Set
  `LUMA_MEMORY_BUDGET` to something your machine can keep on the device. Arrays split over several
  buffers are never staged and cannot be passed to custom kernels. A moved array that the host
  already holds a copy of (it came back and was only read since) moves nothing, and a cheap
  elementwise result of arrays that stay on the device (at least 16 KiB, in one buffer) is
  dropped and recomputed at its next use, which costs a kernel instead of a transfer; writing
  to one of its inputs computes it first.
- `ctx.memory_stats()` reports the bytes in use, idle in the pool and spilled to host memory and the
  budget, `ctx.paging_stats()` the evictions by kind (`clean_evictions`, `readback_evictions`,
  `recompute_evictions`); `ctx.trim()` gives the idle pool back to the device, for example after a phase whose
  temporaries the next one cannot reuse.

## A checklist for a slow program

1. **Profile it.** `ctx.profile(|| ..)` with the slow section: how many launches? Which kernel
   dominates? Is the time in kernels at all, or in the host (then it is `eval`, readbacks, or
   array construction)?
2. **Count launches** per step. A step that should be one fused kernel and is ten means a cut:
   print `profile.cuts()` and read the reason.
3. **Look for readbacks** inside loops: `item()`, `to_vec()`, `max().item()?` per iteration. Move
   them out of the loop or use `iterate_until_all`.
4. **Use the loop primitives** where the step fits; check that the step is elementwise (register
   loop) or a stencil (time-blocked).
5. **Check the precision.** `Df32` arithmetic costs 27 to 44 `f32` operations on the GPUs measured; a `Df32` field that
   is only read in a residual can be `f32`.
6. **Check views of expensive expressions.** An expression read through slices at several offsets
   is recomputed, staged or materialised by the cost model; if you know better, `x.cache()`
   materialises, `LUMA_STAGING=off` turns staging off.
7. **Reuse arrays.** Allocate outside the loop; ping-pong with two arrays in custom kernels.
8. **Cold start.** The first run compiles kernels (seconds for big `Df32` ones); the second reads
   the disk cache. Measure the second run.
9. **Device.** `LUMA_DEVICE=cpu` forces the native CPU backend, a fallback that runs on the host's
   threads. On a laptop check which GPU opened ([FAQ](faq.md#choosing-a-device)).
