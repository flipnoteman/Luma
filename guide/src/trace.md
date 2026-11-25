# Compiled callables: `trace`

A function that runs the same array code many times (a training step, a solver stage, a filter
applied to every frame) builds the same graph, plans the same fusion and compiles nothing new on
every call: that host work can be as long as the GPU's. `luma::trace` records the launches the
function makes **once** and gives back a [`Compiled`](https://docs.rs/luma/latest/luma/struct.Compiled.html)
that replays them over new input arrays. A call builds no graph, plans nothing and fuses nothing;
its host cost is the encoding of the recorded launches.

```rust
use luma::*;

// Weights captured by the closure; the input arrives with every call.
let w = Array::from_slice(&[0.5f32; 16], [4, 4])?;
let step = trace(move |inp| {
    let x = inp.get::<f32>(0);
    let y = x.matmul(&w).maximum(0.0);
    (y.sum(), y).into()                       // two outputs, of any element types
});

let x = Array::from_slice(&[1.0f32; 16], [4, 4])?;
let mut out = step.call(&[&x])?;              // the first call traces, then runs
let (loss, y): (Array<f32>, Array<f32>) = (out.take(0)?, out.take(1)?);
assert_eq!(loss.item()?, 32.0);
assert_eq!(y.shape(), x.shape());

// The next call runs the recorded launches over new arrays: no tracing.
let x2 = Array::from_slice(&[2.0f32; 16], [4, 4])?;
let loss2 = step.call(&[&x2])?.take::<f32>(0)?;
assert_eq!(loss2.item()?, 64.0);
assert_eq!(step.stats().traces, 1);
# Ok::<(), luma::LumaError>(())
```

It is not an implicit optimisation of eager loops, and it is not a loop: [`iterate_all` and the
`until` loops](loops.md) keep their jobs, and a `Compiled` is a stage a loop can run between its
chunks.

## What the function may do

The function runs **once per trace**, on probe arrays, never per call. It is kept (it must be
`'static`) because a new signature traces it again.

It may:

- build arrays from its [`Inputs`](https://docs.rs/luma/latest/luma/struct.Inputs.html)
  (`inp.get::<T>(i)`) and from arrays it captured,
- call any operation that does not read data back: elementwise math, views, shifts, reductions,
  `matmul`, `sort`, `cumsum`, `conv1d`, `cache()`, custom `Kernel` launches,
- assign into arrays it creates itself (`slice_mut(..).assign(..)`),
- return any number of arrays of any element types (`Outputs` converts from one array, a tuple of
  up to eight, an array or a `Vec`), including an input unchanged or a captured array.

It may not:

- read data back (`to_vec`, `item`, `sync`, an `eval` it then inspects), upload host data
  (`from_slice`, `write`) or make a zeroed array (`zeros`): the trace ends with
  `TraceError::Aborted` naming what it did. Create constants outside the function and capture them.
  (On the `cpu` backend, where host data does not go through a transfer, an array made from host
  data inside the function becomes a constant of the trace instead; do not rely on it.)
- write into an input (`TraceError::WritesInput`) or into an array it captured
  (`TraceError::WritesCaptured`): express an update by returning the new value.
- call `iterate_all`, `iterate_until_all` or another `Compiled` (they record too:
  `TraceError::Nested`).
- depend on state that changes between calls (a `Cell` counter, a changing
  [`Scalar`](https://docs.rs/luma/latest/luma/struct.Scalar.html)): the trace freezes it.

Which operations record without ending the trace is the question `these_library_ops_trace` in the
test suite answers; an operation whose output size depends on the data (`unique`, `nonzero`) reads
back and cannot be traced.

## Captured arrays

An array the closure captures is bound **by buffer** when the function is traced and read when the
callable runs. A weight updated in place between calls (`w.write(..)`, `w += ..` outside the
function) is seen by the next call; a weight that is *replaced* (`w = new`) is not, the trace read
the buffer it was traced with. The callable keeps the captured arrays resident (pinned) while it
lives; [`Compiled::clear`](https://docs.rs/luma/latest/luma/struct.Compiled.html#method.clear)
and dropping the last clone release them. A callable that is forgotten holds memory:
`Compiled::stats().pinned_bytes` says how much.

Closed-over host values (`&x * 2.0`, a `Scalar` read inside the function) are **constants of the
trace**.

## Scalar parameters are one-element arrays

`trace` takes no host scalar arguments. A learning rate that changes per call is an array of shape
`[1]` (an extent-1 axis broadcasts, so `&g * &lr` is the usual expression):

```rust
use luma::*;

let sgd = trace(|inp| {
    let (w, g, lr) = (inp.get::<f32>(0), inp.get::<f32>(1), inp.get::<f32>(2));
    (&w - &g * &lr).into()
});
let w = Array::from_slice(&[1.0f32, 2.0, 3.0], 3)?;
let g = Array::from_slice(&[1.0f32; 3], 3)?;
for rate in [0.5f32, 0.25] {
    let lr = Array::from_slice(&[rate], 1)?;
    let next = sgd.call1::<f32>(&[&w, &g, &lr])?;
    assert_eq!(next.to_vec()?, [1.0 - rate, 2.0 - rate, 3.0 - rate]);
}
assert_eq!(sgd.stats().traces, 1);   // a new rate is a new array, not a new trace
# Ok::<(), luma::LumaError>(())
```

## Shape variants

A call finds its variant by the element types and **exact shapes** of its arguments (and the
context and its fusion settings): a recorded launch bakes the workgroup counts and the extents as
uniform words. A new signature traces the function again and keeps the old variant;
`TraceOptions::max_variants` (default 8) drops the least recently called one. A loop that
alternates over more shapes than variants retraces at every call (`stats().retraces` shows it).

A retrace is cheap in a particular way: extents are uniform words, not kernel text, so tracing at
another shape of the **same class** (`ShapeClass::of(shape)`: each extent is 1, a power of two, or
other; and whether the array has `2^23` elements or more) compiles no kernel, the kernel cache
answers (`TraceStats::retrace_compiled_kernels` is 0). Across a class boundary only the kernels the
new class needs are built. A framework with a variable batch pads to a fixed set of shapes; that is
its job, not luma's.

- `TraceOptions::strict()`: a signature without a variant is an error
  (`TraceError::Signature`, both signatures rendered), not a retrace. Use it with
  `Compiled::precompile(&args)`, which traces without running.
- `Compiled::variants()` lists what is traced (shapes, classes, launches, calls, bytes).

## Errors and fallbacks

Errors come as `LumaError::Trace(TraceError)`, wrapped like every operation's error with the
caller's source location. `TraceOptions::eager_fallback()` turns `Aborted`, `Nested`, `Segmented`
(an argument or output that spans several buffers) and `Empty` into a call that runs the function
eagerly on the real arguments instead, with a one-time warning through the `log` crate; the
signature then runs eagerly from the next call on.

A `Compiled` called while this thread is recording (inside the step of `iterate_all`, inside another
trace) returns `TraceError::Nested`: call the function itself in the loop (`iterate_all` records
it), or call the `Compiled` outside the loop.

## Verification

Structure can change in ways the signature does not see: a closure that reads a counter, a `Scalar`
set in between. `TraceOptions::verify_every(n)` (or `LUMA_TRACE_VERIFY=<n>`, which also applies to
tests) makes every `n`-th call of a variant trace the function again into a scratch variant and
compare fingerprints: the kernels, workgroups, uniform bytes and buffer roles of every launch. A
difference is `TraceError::StructureChanged`; the stale variant is dropped, so the next call traces
again.

## What a trace is

A `Compiled` is the plan the planner made **during its trace**: the cuts depend on which arrays are
alive then, so a traced function and an eager run of the same code can differ in the last bits where
the backend fuses a multiply and an add (CUDA). Tracing needs one extra full-size buffer per input
until the trace ends (the probes), so the first call of a large model is a memory peak; a trace under
memory pressure aborts (`eviction under memory pressure`) and is simply tried again after
`Context::trim`. Calls from several threads share one `Compiled`; the encoding of one variant is
serialised (its intermediate buffers are shared by every call), the GPU work is not.
