# Lazy evaluation and fusion

## What `a * b + c` does

```rust
use luma::*;

let n = 1 << 20;
let a = Array::<f32>::full(n, 2.0)?;
let b = Array::<f32>::full(n, 3.0)?;
let c = Array::<f32>::full(n, 1.0)?;

let r = &a * &b + &c;        // records two operations, launches nothing
assert_eq!(r.to_vec()?[0], 7.0); // one kernel: out[i] = a[i] * b[i] + c[i]
# Ok::<(), luma::LumaError>(())
```

Each operator creates a node in a graph that remembers its inputs. `r` is the node for the add;
nothing is allocated for the product. When the host needs the values, Luma flattens the graph
below `r` into one program, compiles a single kernel for the active backend (CUDA, Metal, Vulkan/DX12, or native SIMD CPU), and launches it. The product never
exists in memory: the kernel reads `a`, `b` and `c` once and writes `r` once. A chain of 20
elementwise operations is still one launch and one pass over memory.

A lazy array is the value as of when it was written. If you later change an input with `write`,
an in-place `+=`, or a mutable view, expressions recorded before the change still see the old
contents:

```rust
use luma::*;

let mut x = Array::from_slice(&[1.0f32, 2.0], 2)?;
let y = &x * 10.0;   // lazy, reads x
x += 1.0;            // in-place, fused too; y is computed first, so it keeps the old x
assert_eq!(y.to_vec()?, vec![10.0, 20.0]);
assert_eq!(x.to_vec()?, vec![2.0, 3.0]);
# Ok::<(), luma::LumaError>(())
```

## Scalars that change

A number in an expression is a uniform of the kernel, not part of it, so `&x * 2.0` and
`&x * 3.0` share one compiled kernel. A [`Scalar`](https://docs.rs/luma/latest/luma/struct.Scalar.html)
is a number you can change between operations: `Scalar::new(v)`, `set(v)`, `get()`. It holds an
`f32`, `u32` or `i32` and works wherever a plain number does. The value is read when the operation
is **recorded**: operations recorded before a `set` keep the old value, also if they run later.
Clones share the value.

```rust
use luma::*;

let x = Array::from_slice(&[1.0f32, 2.0], 2)?;
let s = Scalar::new(2.0f32);
let a = &x * &s;                 // doubles
s.set(3.0);
let b = &x * &s;                 // triples; `a` still doubles
assert_eq!(a.to_vec()?, vec![2.0, 4.0]);
assert_eq!(b.to_vec()?, vec![3.0, 6.0]);
assert_eq!(s.get(), 3.0);
# Ok::<(), luma::LumaError>(())
```

This is how a loop varies a parameter (a time step, a coefficient) without recompiling.

## When kernels run

A kernel runs when one of these happens:

| Call | Effect |
| --- | --- |
| `to_vec`, `read_into`, `item`, `to_array` | compute what the array needs, submit, wait, copy back |
| `a.eval()` / `eval_all(&[&a, &b])` | compute these arrays and hand the batch to the GPU **without waiting** |
| `ctx.flush()` / `luma::flush()` | compute every lazy array this thread still holds and hand the batch to the GPU without waiting |
| `ctx.sync()` | submit and wait until the GPU is idle |
| a custom [`Kernel`](custom-kernels.md) reads or writes the array | the array is computed first |

`eval` is how you keep the GPU busy: it returns as soon as the work is recorded, so the host
builds step *n + 1* while the GPU runs step *n*. Reads and `sync` wait. If the GPU is still busy
with earlier work, Luma may postpone an `eval` to the next one that finds it idle; the steps in
between fuse, and arrays you drop before then are never computed. A program that calls `eval` and
then does long work without touching Luma should call `flush` so the GPU starts.

```rust
use luma::*;

let ctx = Context::new()?;
let mut u = Array::<f32>::full_on(&ctx, 1 << 16, 1.0)?;
for _ in 0..100 {
    u = (&u * 0.99 + 0.01).sqrt();
    u.eval()?;               // non-blocking
}
ctx.sync()?;                 // wait once, at the end
assert!((u.to_vec()?[0] - 1.0).abs() < 1e-3);
# Ok::<(), luma::LumaError>(())
```

Errors the GPU raises while running a batch show up at the next call that waits.

### The host cost of a step

Building an expression costs host time per operation (one node each), and evaluating it flattens the
graph, cuts it where the cost model says and launches. A loop that evaluates a step of the same
structure every iteration (`u = step(&u); u.eval()?`) does not plan it every time: the plan cache runs the
launches the first evaluation recorded over the new buffers and scalar values, with bit-identical results
(`LUMA_PLAN_CACHE=off` switches it off; [Loops](loops.md#loops-with-eval)). Only the building of the
expression stays, and the [loop primitives](loops.md) avoid that as well by tracing the step once.
[Performance guide](performance.md#the-host-cost-of-an-eager-step) has the numbers.

## What fuses

Fused into the kernel that reads them: all elementwise operations, comparisons, `select`, casts,
generators (`range`, `linspace`, `uniform`), [views](views.md) (`slice`, `pad`, `flip`,
`transpose`, `shift`, `take`, `concat`), compound assignment, and the elementwise prologue and
epilogue of a [reduction](reductions.md). A view of a lazy expression is recomputed at the
reader's index, staged in workgroup memory, or evaluated first, whichever the
[cost model](performance.md#the-cost-model) estimates is cheapest (within the caps below: a kernel
that would exceed `LUMA_FUSION_MAX_INLINED` is cut whatever the model says).

Not fused, because they need whole-array structure: `scan`, stacked `solve_block_tridiagonal`, `conv1d` / `conv1d_fft`,
and your own kernels. They read the arrays they are given, which are computed first.
`matmul`, `solve_tridiagonal` / `solve_tridiagonal_parts` / `solve_block_tridiagonal_parts`, the axis reductions (`sum_axis`, `reduce_axis`, `argmax_axis`, ...), the scatters (`put_add`, ...) and
the sorts are kernels of their own too, but they compute a pending operand *inside* their loads
instead of storing it, and `matmul`, the line solves, the sorts and the axis reductions of types other than `f32` (and of `f32` while a step is recorded or profiled) are eager: the result is
evaluated when the call returns, so what reads it is another kernel
([Performance guide](performance.md#primitives-that-read-an-expression)). Whole-array reductions (`sum`,
`max`, `sum_where`, ...) of `f32`, `u32` and `i32` stay pending until their value is read or used, and
those forced together are recorded in one kernel (up to 16 accumulators; `LUMA_MULTI_REDUCE`).

An expression is cut into several kernels when it would grow past a limit (`LUMA_FUSION_MAX_NODES`,
`LUMA_FUSION_MAX_LIVE`, `LUMA_FUSION_MAX_INLINED`) or when evaluating a shared piece first is
cheaper. [`Profile::cuts`](https://docs.rs/luma/latest/luma/struct.Profile.html#method.cuts) tells you where
and why.

## `cache()`

`x.cache()` evaluates an expression now (it is recorded as a kernel, without waiting for the GPU)
into a buffer, and later readers load it instead of recomputing it. Luma decides this itself for
views, so you need `cache()` only when you know better: an expensive expression read at many
unrelated places.

```rust
use luma::*;

let x = Array::linspace(0.0, 1.0, 1000)?;
let e = (x.sin() * x.cos()).exp().cache();     // computed once
let r = &e + &e.slice(0, 1..1000).pad([(0, 1)], PadMode::Edge);
assert_eq!(r.len(), 1000);
# r.to_vec()?;
# Ok::<(), luma::LumaError>(())
```

## Looking at what ran: `Profile`

`ctx.profile(|| ..)` runs a closure and returns the GPU time of every kernel launched during it,
by label, and the places where expressions were split. It needs timestamp queries on the device
([`Context::profiling_supported`](https://docs.rs/luma/latest/luma/struct.Context.html)).

```rust
use luma::*;

let ctx = Context::new()?;
let x = Array::from_slice_on(&ctx, &[1.0f32; 1 << 20], 1 << 20)?;
if ctx.profiling_supported() {
    let (sum, profile) = ctx.profile(|| (&x * 2.0).sum().item())?;
    assert_eq!(sum?, 2.0 * (1 << 20) as f32);
    println!("{profile}");
    assert!(profile.launches() >= 1);
}
# Ok::<(), luma::LumaError>(())
```

While profiling, every launch gets a compute pass of its own, so use it to find where the GPU time
goes, not to measure throughput. Counting `profile.launches()` before and after a change is a
quick way to check that something still fuses.
