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
expensive first: label, launches, total, mean, min, max), `splits`, `launches()` and `total()`.
Fused kernels are labelled by what they hold (`Fused(Mul, Add, Sqrt) f32`, `FusedReduce(Sum: Mul)
f32`); your own kernels by `.label(..)` or their entry point.

While profiling, every launch gets a compute pass of its own so that it can be timed. Use it to find
where the GPU time goes, not to measure throughput. For the host's time, time the calls themselves.

### Why was it cut: `splits`

Each [`FusionSplit`](https://docs.rs/luma/latest/luma/struct.FusionSplit.html) names the operation
and source line where an expression was cut, and the
[`SplitReason`](https://docs.rs/luma/latest/luma/enum.SplitReason.html):

| Reason | Meaning |
| --- | --- |
| `Nodes` | more operations than the fusion limit allows in one kernel |
| `Depth` | nested deeper than the engine walks in one kernel |
| `Inputs` | more distinct buffers than a kernel can bind |
| `Live` | too many values live at once (register pressure) |
| `Scalars` | the scalar operands do not fit one uniform buffer |
| `Inlined` | views of lazy expressions unfolded into more instructions than the device's cap |
| `Materialised` | the [cost model](#the-cost-model) estimates that storing an expression once is cheaper than recomputing it at each read |

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
    for split in &profile.splits {
        println!("{split}");                            // split at src/..: 3 operations > 2
    }
    assert!(profile.splits.iter().any(|s| matches!(s.reason, SplitReason::Nodes { .. })));
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
hardware, `x.cache()` forces a materialisation, and `LUMA_STAGING=off` forces recomputation.

The model also sizes loops: [`iterate_all`](loops.md#launch-sizing) cuts long register loops into
launches of about 40 ms of GPU time by it, and it decides when a circular `shift` of an expression
is recomputed, staged, or evaluated first.

## What fuses, and what cuts

Fused: elementwise operations, comparisons, `select`, casts, generators, views, compound
assignment, the input and the epilogue of a reduction ([Lazy evaluation](lazy.md#what-fuses)).

An expression is cut when it passes a limit. They are generous, and a cut costs a launch and a trip
through memory, not correctness:

| Limit | Default | Setting |
| --- | --- | --- |
| operations per kernel | 16384 | `LUMA_FUSION_MAX_NODES`, or [`fusion_limit`](https://docs.rs/luma/latest/luma/fn.fusion_limit.html) for expressions written in a scope |
| nesting depth | 512 | fixed (the walk is recursive) |
| values live at once | by GPU profile | `LUMA_FUSION_MAX_LIVE` |
| instructions after views of lazy expressions are unfolded | 1536 on NVIDIA, 1024 on other GPUs, 192 on CPU | `LUMA_FUSION_MAX_INLINED` |
| input buffers per kernel | what the device can bind | fixed (llvmpipe: 16; Metal: 29) |
| scalar operands | one uniform buffer | fixed |

When a limit is hit, the biggest operand is evaluated first (together with the live arrays that
share work with it), and the rest is flattened again. A kernel that reads many different arrays
(the 50 words of a Keccak state) is the one that meets the input limit; keep such loops in
`iterate_all`, or split them in the source.

Never fused, because they need whole-array structure: `scan`, axis reductions, tridiagonal solves,
FFT convolution, and custom kernels. They read computed arrays.

## Caches

- **Kernels** are cached per context by the *structure* of the expression: operations, types, how
  values are wired. Scalar values are uniforms, so `&x * 2.0` and `&x * 3.0` share one pipeline,
  and an expression with the same shape built again costs the host only the flattening. Identical
  operations on the same operands are one node (hash-consing).
- **Compiled pipelines** are cached on disk (wgpu's `PipelineCache`, Vulkan) per adapter and
  driver, so a second process skips the compile of large kernels, which can take seconds
  (a `Df32` Euler step on NVIDIA). The measured cost model is kept next to it.
  `LUMA_CACHE=off` disables both; `LUMA_CACHE_DIR` moves them. A damaged file is ignored.
- **Loops**: an `iterate_all` called again from the same place with the same step graph reuses the
  earlier program, and after two such calls the GPU starts before the step is traced
  ([Loops](loops.md#calling-it-again)).
- **Buffers**: freed arrays go to a pool of idle buffers (up to half of the memory budget), so a
  program that cycles through gigabytes of temporaries reuses them instead of allocating again.

## Loops

How you write the loop decides how much host time each step costs.

| Loop | Host per step | Notes |
| --- | --- | --- |
| `iterate_all` of an elementwise step | none | one kernel, state in registers, launches of about 40 ms |
| `iterate` of a stencil | none | time-blocked: several steps per pass over memory |
| `iterate_until`, `iterate_until_all` | none | the stop rule runs on the GPU; one wait per block of steps |
| `eval` every step | building the step | recognised as a loop for `f32` steps; never waits |
| readback every step (`max().item()?`) | a GPU round trip | speculation hides it when the step repeats |

Prefer the primitives when the step fits them ([Loops](loops.md)). Otherwise `eval` each step
instead of reading, and `sync` once at the end. A readback in the loop (a convergence test) is the
most expensive way: replace it with `iterate_until_all`, or test every k steps.

## Transfers

The GPU is fast, the way to it is not. Data crosses it at `Array::from_slice` (upload),
`to_vec` / `read_into` / `item` (readback, which also waits), and `write`.

- Keep data on the device: chain operations, and read back the final result and small scalars.
  `item()` of a reduction moves 4 bytes; `to_vec()` of the field moves all of it.
- Use `read_into(&mut buf)` for repeated readbacks into a buffer you own.
- Upload once. A uniform scalar changes without recompiling anything, a new array costs an upload.
- On unified memory (integrated GPUs, Apple silicon) uploads and readbacks are cheaper, and
  the pool reuses mapped buffers.
- Arrays beyond the memory budget move to host memory and back on demand (once-only warning). Set
  `LUMA_MEMORY_BUDGET` to something your machine can keep on the device; chunked arrays do not
  stage views or run custom kernels.

## A checklist for a slow program

1. **Profile it.** `ctx.profile(|| ..)` with the slow section: how many launches? Which kernel
   dominates? Is the time in kernels at all, or in the host (then it is `eval`, readbacks, or
   array construction)?
2. **Count launches** per step. A step that should be one fused kernel and is ten means a cut:
   print `profile.splits` and read the reason.
3. **Look for readbacks** inside loops: `item()`, `to_vec()`, `max().item()?` per iteration. Move
   them out of the loop or use `iterate_until_all`.
4. **Use the loop primitives** where the step fits; check that the step is elementwise (register
   loop) or a stencil (time-blocked).
5. **Check the precision.** `Df32` arithmetic costs tens of `f32` operations; a `Df32` field that
   is only read in a residual can be `f32`.
6. **Check views of expensive expressions.** An expression read through slices at several offsets
   is recomputed, staged or materialised by the cost model; if you know better, `x.cache()`
   materialises, `LUMA_STAGING=off` recomputes.
7. **Reuse arrays.** Allocate outside the loop; ping-pong with two arrays in custom kernels.
8. **Cold start.** The first run compiles kernels (seconds for big `Df32` ones); the second reads
   the disk cache. Measure the second run.
9. **Device.** `LUMA_DEVICE=cpu` is for testing; a software device is far slower. On a laptop
   check which GPU opened ([FAQ](faq.md#choosing-a-device)).
