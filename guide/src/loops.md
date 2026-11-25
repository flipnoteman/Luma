# Loops

Time stepping, iterative solvers and per-element recurrences all repeat one step many times.
Written as a Rust loop, each iteration builds a graph and the host talks to the GPU once per step.
Luma has loop primitives that take the step *as a closure*, trace it once, and run all the steps
on the GPU:

| Primitive | Use |
| --- | --- |
| [`Array::iterate`](https://docs.rs/luma/latest/luma/struct.Array.html#method.iterate) | `steps` applications of a step on one `f32` array; 1-D and 2-D stencils run time-blocked |
| [`iterate_all`](https://docs.rs/luma/latest/luma/fn.iterate_all.html) | `steps` applications of a step on several arrays of one element type; elementwise `f32` and `u32` steps keep the state in registers |
| [`Array::iterate_until`](https://docs.rs/luma/latest/luma/struct.Array.html#method.iterate_until) | a step until the largest change is at most a tolerance |
| [`iterate_until_all`](https://docs.rs/luma/latest/luma/fn.iterate_until_all.html) | a step on any [record](records.md) until a rule over recorded scalars holds |
| [`compile_until_all`](https://docs.rs/luma/latest/luma/fn.compile_until_all.html) | the same loop traced once and run many times, for a solver that steps in chunks |

The two `_until` loops evaluate their stop rule on the GPU; `iterate` and `iterate_all` run a
fixed number of steps. None of them needs you to manage the second buffer a step needs: the previous state and the next are swapped for you.

## Stencils: `iterate`

```rust
use luma::*;

// A 1-D heat equation: a spike diffuses.
let mut data = vec![0.0f32; 9];
data[4] = 1.0;
let u0 = Array::from_slice(&data, 9)?;

let alpha = 0.25;
let u = u0.iterate(4, |u| {
    let lap = u.shift(1) + u.shift(-1) - u * 2.0;   // circular shifts
    u + lap * alpha
});

// The same on the host.
let mut want = data.clone();
for _ in 0..4 {
    let old = want.clone();
    for i in 0..9 {
        want[i] = old[i] + alpha * (old[(i + 8) % 9] + old[(i + 1) % 9] - 2.0 * old[i]);
    }
}
for (g, w) in u.to_vec()?.iter().zip(&want) {
    assert!((g - w).abs() < 1e-6);
}
# Ok::<(), luma::LumaError>(())
```

When the step is a stencil (one elementwise expression of the argument, circular shifts of it by
a few elements, scalars and other fixed arrays of the same shape), each launch loads a tile into
workgroup memory and runs several steps on it, so the data passes through memory once per several
steps rather than once per step. Any other step (shifts of a computed expression, `cache()`,
`assign`, reductions) is traced and replayed step by step (a step of lazy expressions that does not fuse
into a stencil may be traced as a body of several steps, see
[Elementwise loops](#elementwise-loops-iterate_all)): each step is still the fused kernels
of the step, and the loop costs no host time per step.

A step that is a stencil of 2-D data looks the same: `u.shift([1, 0]) + u.shift([-1, 0]) +
u.shift([0, 1]) + u.shift([0, -1]) - u * 4.0`. Stencils exist for 1-D and 2-D arrays held in one
buffer; a 3-D array, or one split over several buffers, runs step by step.

## Elementwise loops: `iterate_all`

`iterate_all(init, steps, step)` runs the step on `N` arrays at once; `step` gets `&[Array<T>; N]`
and returns the next `[Array<T>; N]`. When every result is an elementwise expression of the
states, of other arrays of the same length, and of scalars, **the whole loop is a kernel that
keeps the states in registers and writes them once per launch**, whatever the step count. That is the shape
of a Mandelbrot iteration, a Monte Carlo walk, or a round of a hash function.

```rust
use luma::*;

// z -> z^2 + c on the real line, counting steps until z leaves [-2, 2].
let c = Array::from_slice(&[0.1f32, -0.5, 2.0, 1.0], 4)?;
let zero = Array::<f32>::zeros(4)?;
let [z, count] = iterate_all([&zero, &zero], 64, |[z, count]| {
    let inside = z.le(2.0);
    [select(&inside, &(z * z + &c), z), count + select(&inside, 1.0, 0.0)]
});
assert_eq!(count.to_vec()?, vec![64.0, 64.0, 2.0, 3.0]);
# z.eval()?;
# Ok::<(), luma::LumaError>(())
```

The states may be `u32` as well, which makes register loops of integer recurrences, hash rounds
and counters (see [Integer and bit operations](integers.md)); all states of one loop have one type:

```rust
use luma::*;

// A linear congruential generator, 5 steps per element, 32-bit wrap-around arithmetic.
let seed = Array::from_slice(&[1u32, 2, 3], 3)?;
let [x] = iterate_all([&seed], 5, |[x]| [x * 1664525u32 + 1013904223u32]);

let want: Vec<u32> = [1u32, 2, 3]
    .iter()
    .map(|&s| (0..5).fold(s, |x, _| x.wrapping_mul(1664525).wrapping_add(1013904223)))
    .collect();
assert_eq!(x.to_vec()?, want);
# Ok::<(), luma::LumaError>(())
```

A step that is *not* elementwise (shifts, `cache()`, `assign`), or states of another element type
(`i32`, `Df32`, `f64`), is traced once and its recorded kernels are replayed step by step (as one
graph on CUDA, Metal and the CPU backend when the step is replayed often enough to pay for the
capture). The results are the same; only the one-kernel register loop is lost. A step of lazy
expressions only that is not a register loop (shifts, say) is also traced as the body of up to 16
steps, so that the recorded launches are those the planner makes of the chain, as an eager
`u = step(&u); u.eval()` loop gets them; the loop replays that body `steps / k` times and the rest one step
at a time. This is used only when the chain makes fewer launches than the steps one by one, the cost model
gives it less time and tracing it costs less than half of what the loop takes step by step; loops on the
CPU backend replay step by step. A state the step hands back unchanged takes no launch per step.

### Launch sizing

A long register loop is not one kernel launch for ever. The cost model
([Performance](performance.md#the-cost-model)) sizes its launches to about 40 ms of GPU time (at least one step per launch), which keeps them far
below a driver watchdog (2 s on Windows) unless a single step is itself that long. The host only
records the launches.

### Calling it again

A register loop called again from the *same place in the source*, whose step builds the same graph
(the same operations and scalar values) over other arrays, is not flattened again. After two calls
in a row did that, the GPU starts the loop before `step` is even traced, on the guess that it will
build the same graph again (a wrong guess only wastes GPU time). `step` is still called once per
call, and a captured array is read when the loop runs. A function that runs the same loop per
batch (the permutation of a hash, one Mandelbrot tile) therefore skips the flattening from its
second call on.

## The step runs once

`step` is called **once** to trace it, with placeholder arrays (`iterate_until_all` passes the state itself only
when it spans several buffers); the recorded launches are replayed for every step. So:

- compute the next state from the arguments and *captured arrays only*, in the closure;
- do not keep the argument or read values back inside the step (`item`, `to_vec`): an
  `iterate_until_all` step that reads back cannot be replayed and runs as a host loop instead
  (still correct, not fast);
- anything that depends on the iteration number must be an array, not a Rust variable. The step
  of `iterate_until_all` gets the 0-based index of the step as a one-element `f32` array for this
  purpose; `iterate_all` loops over rounds (a Keccak round constant) carry a counter in a state;
- the state arrays are read-only inside a step: `assign` on the argument fails with
  `LumaError::ReadOnly`, and `+=` poisons the array with it. Compute a new array instead.

`init` of `iterate` and `iterate_all` is not modified; the result is a new array.

## Stop conditions: `iterate_until_all`

`iterate_until_all(init, until, step)` runs `step` at most `max_steps` times and stops at the first
step after which a condition on *recorded scalars* holds. The condition is evaluated on the GPU by
a small kernel between steps, so the host waits only once per block of steps, not per step.

```rust
use luma::*;

let u = Array::from_slice(&[4.0f32, 2.0, 1.0, 0.5], 4)?;
let run = iterate_until_all(
    (u,),                                        // the state: a record (here a tuple)
    Until::new(500)                              // at most 500 steps
        .record("max", |s: &(Array<f32>,)| s.0.max())     // a scalar per step
        .stop_when(|h| (h["max"].first() / h["max"].last()).ge(1e6)),
    |s, _step| (&s.0 * 0.5,),
)?;
assert_eq!(run.steps, 21);                       // the ratio after s steps is 2^(s-1): 2^19 < 1e6 <= 2^20
assert_eq!(run.reason, StopReason::Predicate(0));
assert_eq!(run.history("max").unwrap().len(), 21);
assert_eq!(run.state.0.to_vec()?[0], 4.0 / 2f32.powi(21));
# Ok::<(), luma::LumaError>(())
```

The pieces:

- **[`Until`](https://docs.rs/luma/latest/luma/struct.Until.html)** is built with `Until::new(max_steps)`.
  `.record(name, f)` computes a one-element `f32` array from the state after every step (always
  `f32`, even for a `Df32` state: record `s.u.max().to_f32()`). `.stop_when(f)` adds a
  condition; `.check_every(n)` fixes the block between waits (default: the first block is 16
  steps and every block that ends without a stop doubles the next, up to 128; steps, reason,
  histories and state never depend on this).
- **[`History`](https://docs.rs/luma/latest/luma/struct.History.html)** is what a stop rule
  receives. `h["name"]` is a [`Series`](https://docs.rs/luma/latest/luma/struct.Series.html);
  `h.step()` is the number of steps done. The rule is called once, before the loop starts, and
  builds a condition.
- **[`Series`](https://docs.rs/luma/latest/luma/struct.Series.html)** turns the recorded values
  into a [`Stat`](https://docs.rs/luma/latest/luma/struct.Stat.html): `last()`, `first()`,
  `peak()` (largest so far), `trough()`, `window_max(k)`, `window_min(k)`, `window_mean(k)` and
  `prev(k)` (the value `k` steps ago).
- **`Stat`** combines with `+ - * /` (with other stats and `f32`) and `abs()`, and compares with
  `lt`, `le`, `gt`, `ge` into a [`Cond`](https://docs.rs/luma/latest/luma/struct.Cond.html);
  conditions combine with `and`, `or`, `not`. A comparison with a NaN never holds.
- **[`Run`](https://docs.rs/luma/latest/luma/struct.Run.html)** is the result: `state`, `steps`,
  `reason`, and `history(name)` with the value recorded after every step.
- **[`StopReason`](https://docs.rs/luma/latest/luma/enum.StopReason.html)** is `MaxSteps` or
  `Predicate(i)`, the index of the first `stop_when` that held.

Several rules and the step counter:

```rust
use luma::*;
use std::cell::Cell;

let calls = Cell::new(0);
let x = Array::from_slice(&[4.0f32, 3.0], 2)?;
let run = iterate_until_all(
    (x,),
    Until::new(100)
        .record("max", |s: &(Array<f32>,)| s.0.max())
        .stop_when(|h| h["max"].last().le(1e-3))          // Predicate(0)
        .stop_when(|h| h.step().ge(5.0))                  // Predicate(1)
        .check_every(8),
    |s, _| {
        calls.set(calls.get() + 1);
        (&s.0 * 0.5,)
    },
)?;
assert_eq!(run.steps, 5);
assert_eq!(run.reason, StopReason::Predicate(1));
assert_eq!(calls.get(), 1);                                // traced once, replayed
# Ok::<(), luma::LumaError>(())
```

A step that reads results back or uses states larger than one buffer runs on the host, with the
same stop rule evaluated there; both evaluations agree.

### Graphs and fused records

On a backend that captures command graphs (CUDA, Metal and the native CPU backend; wgpu does not),
a block of steps is one graph replay: the stop rule's gate kernel counts the steps itself and
passes the dispatch arguments from step to step on the device, so nothing in a block depends on the
host. The first step of a loop reads the initial state, which no graph does, so it is launched on
its own. A graph is captured only where it pays: the blocks of a compiled loop on first use, those of
a one-shot loop the second time they run; until then the launches are replayed one by one, with the
same results. A one-shot loop that has run 100 launches without stopping replays the rest as repeated
graphs of a small unit of steps (at most 16, about 64 launches) where the backend captures graphs (CUDA).

When there is one record and it is a reduction of one array expression (a norm, a sum, a maximum),
Luma traces the record together with the step, so one launch stores the next state and reduces the
record, and where a reduction finishes in its first launch (CUDA; the CPU backend with
`LUMA_SINGLE_PASS_REDUCE=on`) the last workgroup of that launch also evaluates the stop rule: a step
is one launch. `iterate_until` does the same with its residual `max |next - prev|`. Several records,
or records that are not one reduction, run after the step in launches of their own; values, steps
and reasons are the same. On the wgpu backend, which has no single-pass reduction, the stop rule runs in the
last pass that finishes the record's reduction instead of a launch of its own, so a step is one launch less
than it was.

### Chunked solvers: `compile_until_all`

Running a loop in chunks with `iterate_until_all` (to look at the state between chunks) traces the
step again for every chunk. `compile_until_all(&init, until, step)` traces it once, for the shapes
of `init`, and returns a [`CompiledUntil`](https://docs.rs/luma/latest/luma/struct.CompiledUntil.html);
`run(state)` and `run_steps(state, n)` run it on any state of those shapes. Every run starts at step
0 with an empty history. The buffers and the captured graphs stay between runs. There is no host
fallback: a step that reads back, states that span several buffers or `Until::new(0)` return
`LumaError::Unsupported`.

```rust
use luma::*;

let x = Array::from_slice(&[4.0f32, 3.0], 2)?;
let until = Until::new(100)
    .record("max", |s: &(Array<f32>,)| s.0.max())
    .stop_when(|h| h["max"].last().le(1e-3));
let mut compiled = compile_until_all(&(x.clone(),), until, |s, _| (&s.0 * 0.5,))?;
let first = compiled.run((x,))?;
assert_eq!(first.reason, StopReason::Predicate(0));
let again = compiled.run(first.state)?;                  // new initial state, no new trace
assert!(again.steps >= 1);
# Ok::<(), luma::LumaError>(())
```

### Convergence on one array: `iterate_until`

For the common case of one `f32` array stopped when the largest change `|next - prev|` is at most
`tol`, `u.iterate_until(max_steps, tol, step)` needs no records:

```rust
use luma::*;

let u0 = Array::from_slice(&[0.0f32, 0.0, 1.0, 0.0, 0.0], 5)?;
let u = u0.iterate_until(10_000, 1e-6, |u| {
    (u.shift(1) + u.shift(-1)) * 0.5            // averaging: converges to the mean 0.2
});
for v in u.to_vec()? {
    assert!((v - 0.2).abs() < 1e-3);
}
# Ok::<(), luma::LumaError>(())
```

## Loops with `eval`

A loop that you write yourself, with `eval` each iteration, is also fast:

```rust
use luma::*;

let mut u = Array::<f32>::full(1 << 16, 1.0)?;
for _ in 0..1000 {
    u = (&u * 0.99 + 0.01).sqrt();
    u.eval()?;                  // records and hands over; does not wait
}
assert!((u.to_vec()?[0] - 1.0).abs() < 1e-3);
# Ok::<(), luma::LumaError>(())
```

`eval` does not wait for the GPU and may postpone the computation while the GPU is busy
([Lazy evaluation](lazy.md#when-kernels-run)). A loop of `x = step(&x); x.eval()` over `f32`
arrays is *recognised* while evaluation is postponed: the steps are not chained into an ever
longer expression, but run together when a result is needed, as one register loop or
a time-blocked stencil (up to four `f32` arrays evaluated together with `eval_all` form one
stencil that shares a tile). The graph that is pending stays one step deep however long the loop runs.
A step that reads slices or pads of lazy expressions is evaluated step by step; each step still
fuses.

Compared with the loop primitives, an `eval` loop calls your step every iteration (host work per
step) but may use anything inside it, including readbacks. Use the primitive when the step fits
and the host time matters.

When `eval` computes right away (the GPU is idle, as in a loop whose steps are shorter than the host's
work for them) and the graph it evaluates has the structure of an earlier one (the same operations,
lengths, views and wiring, the same buffers read in the same places, equal scalars where they were
equal, the same planner flags and cost model) and is all the lazy work that is alive, Luma runs the store
launches of that earlier evaluation again over the new buffers and scalar values instead of
flattening, cutting and judging the graph again (the plan cache, 128 plans per context). The part that
builds the expression stays: on an RTX 3090 (CUDA) the per-step cases of the `cfd` benchmark suite
(`euler/3d_per_step` 6.60 to 4.33 ms, `muscl/3d_per_step` 3.49 to 2.44 ms, `euler/2d_per_step` 3.06 to
2.42 ms) got 1.1 to 1.5 times faster. The results are bit-identical to what the planner makes.
Structures whose planning looked at other state (an in-place buffer takeover, packing of arrays into one
buffer, failures, loops, reductions) are remembered as uncacheable and plan as before, and the cache backs
off after a run of misses. `LUMA_PLAN_CACHE=off` switches this off and `LUMA_PLAN_CACHE=trace` prints what
it records ([Environment variables](environment.md)).

### Readbacks in a loop

`step` followed by `.max().item()?` every iteration waits once per step. When the same step has
been seen repeating, Luma computes the next few steps and their residuals during a readback and
answers the following readbacks from memory (*speculation*): the round trip is paid once per
several steps. Values are those of the program as written; a miss only costs the wasted GPU
work. `LUMA_SPECULATE=off` disables it ([Environment variables](environment.md)). If the loop
only needs a convergence test, `iterate_until` and `iterate_until_all` wait once per block of
steps instead of once per step.

## Line solvers: `solve_tridiagonal`

An implicit time step (diffusion, a pressure correction) solves one tridiagonal system per grid
line. `solve_tridiagonal(axis, &lo, &diag, &up, &rhs)` solves
`lo[s] x[s-1] + diag[s] x[s] + up[s] x[s+1] = rhs[s]` along `axis` for every line of four
`f32` arrays of one shape (`lo` at the first and `up` at the last cell of a line are ignored); the
result has the shape of `rhs`. It evaluates the operands and does not wait for the GPU. A zero
pivot gives `Inf` / `NaN` in that line; Luma does not check. A shape or context mismatch poisons
the result.

```rust
use luma::*;

// Two lines of four cells along axis 0 (shape [4, 2]): the matrix tridiag(-1, 2, -1).
let lo = Array::from_slice(&[0.0f32, -1.0, -1.0, -1.0, 0.0, -1.0, -1.0, -1.0], [4, 2])?;
let diag = Array::full([4, 2], 2.0f32)?;
let up = Array::from_slice(&[-1.0f32, -1.0, -1.0, 0.0, -1.0, -1.0, -1.0, 0.0], [4, 2])?;
let rhs = Array::from_slice(&[1.0f32, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 5.0], [4, 2])?;

let x = solve_tridiagonal(0, &lo, &diag, &up, &rhs).to_vec()?;
let want = [1.0f32, 1.0, 1.0, 1.0, 1.0, 2.0, 3.0, 4.0];
assert!(x.iter().zip(want).all(|(a, b)| (a - b).abs() < 1e-5));

// `_from`: line `l` starts at cell `start[l]`; the cells before it are 0.
let start = Array::from_slice(&[0u32, 2], 2)?;
let y = solve_tridiagonal_from(0, &lo, &diag, &up, &rhs, &start).to_vec()?;
assert_eq!(&y[4..6], &[0.0, 0.0]);
# Ok::<(), luma::LumaError>(())
```

`solve_block_tridiagonal(block, axis, &lo, &diag, &up, &rhs)` is the same with `block x block`
blocks (`1 <= block <= 8`), for coupled unknowns per cell. `rhs` has shape `[block, g0, g1, g2]`; `lo`, `diag` and `up` have
`[block * block, g0, g1, g2]`, entry `(row, col)` of a block at `row * block + col`; `axis` is
`1..=3`. `solve_block_tridiagonal_from` takes a `start` array like the scalar form.

```rust
use luma::*;

// Block 2, three cells along axis 1, diagonal blocks 2 * identity, no coupling: x = rhs / 2.
let zero = Array::<f32>::zeros([4, 3])?;
let diag = Array::from_slice(&[2.0f32, 0.0, 0.0, 2.0].repeat(3), [4, 3])?;
let rhs = Array::from_slice(&[2.0f32, 4.0, 6.0, 8.0, 10.0, 12.0], [2, 3])?;
let x = solve_block_tridiagonal(2, 1, &zero, &diag, &zero, &rhs);
assert_eq!(x.to_vec()?, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
# Ok::<(), luma::LumaError>(())
```

### Solving on parts, and several systems at once

The blocks of a cell often come as sixteen (or four, or nine) arrays of their own, the entries
of a Jacobian, each an expression of the state; stacking them into the `[block * block, ..]`
arrays above is a kernel per stack, and reading the solution's components back is a gather.
`solve_block_tridiagonal_parts(axis, lo, diag, up, rhs)` takes the entries as arrays of the grid's
shape instead (`lo`, `diag` and `up` as `[&Array<f32>; block * block]`, entry `(row, col)` at
`row * block + col`, `rhs` as `[&Array<f32>; block]`) and returns the components of the solution
as `[Array<f32>; block]`. `axis` is an axis of the grid, as for `solve_tridiagonal`. A part that
is a lazy expression is not evaluated: the solver kernel computes it where it reads the
coefficients of a cell, so a step from the state to the solution can be one launch. Parts that are
evaluated arrays are bound as buffers. If the expressions need more buffers or scalars than a
kernel can bind, or cannot be read by the kernel, or the blocks are 6 to 8 wide, or the Thomas
solve is chosen (`LUMA_TRIDIAGONAL=thomas`) and its scratch exceeds a buffer, the parts are
stacked and the stacked solve runs; the result is the same.
`solve_block_tridiagonal_parts_from` takes a `start` array.

```rust
use luma::*;

// Block 2 on a [5, 3] grid along axis 0: the diagonal blocks are diag(d, 4) with `d` an expression
// (never evaluated alone), no coupling between cells: x = rhs / diag.
let zero = Array::<f32>::zeros([5, 3])?;
let two = Array::<f32>::full([5, 3], 2.0)?;
let (r0, r1) = (Array::<f32>::full([5, 3], 3.0)?, Array::<f32>::full([5, 3], 8.0)?);
let d0 = &two + 0.0;
let d1 = &two * 2.0;
let [x0, x1] = solve_block_tridiagonal_parts(0, [&zero; 4], [&d0, &zero, &zero, &d1], [&zero; 4], [&r0, &r1]);
assert_eq!(x0.to_vec()?, vec![1.5; 15]);
assert_eq!(x1.to_vec()?, vec![2.0; 15]);
# Ok::<(), luma::LumaError>(())
```

Independent scalar systems on one grid (the two equations of a turbulence model, say) solve in
one launch with `solve_tridiagonal_parts(axis, [(&lo_k, &diag_k, &up_k, &rhs_k), (&lo_w, ..)])`,
which returns one solution per system (`_from` with a `start` array likewise; up to 64 systems):
the lines of all systems fill the device together, and the lazy parts are computed in the kernel as
above. An extra axis of a single array batches systems as well: `solve_tridiagonal` on `[2, ni, nj]`
arrays along axis 1 or 2 solves two systems per line, whichever algorithm runs.

```rust
use luma::*;

// Two systems on a [6, 2] grid along axis 0: x[s] - 0.25 (x[s-1] + x[s+1]) = rhs[s], with
// right-hand sides 1 and 2, so the second solution is twice the first.
let lo = Array::<f32>::full([6, 2], -0.25)?;
let up = Array::<f32>::full([6, 2], -0.25)?;
let diag = Array::<f32>::full([6, 2], 1.0)?;
let rhs = Array::<f32>::full([6, 2], 1.0)?;
let rhs2 = &rhs * 2.0;
let [x, y] = solve_tridiagonal_parts(0, [(&lo, &diag, &up, &rhs), (&lo, &diag, &up, &rhs2)]);
let (x, y) = (x.to_vec()?, y.to_vec()?);
assert!(x.iter().zip(&y).all(|(a, b)| (2.0 * a - b).abs() < 1e-5));

// A batch axis: [2, 6, 2] along axis 1 is the two systems above, stacked.
let stack = |a: &Array<f32>| stack_slice(0, &[a.clone(), a.clone()]);
let batched = solve_tridiagonal(1, &stack(&lo), &stack(&diag), &stack(&up), &stack_slice(0, &[rhs.clone(), rhs2.clone()]));
let z = batched.to_vec()?;
assert!((z[0] - x[0]).abs() < 1e-5 && (z[1] - y[0]).abs() < 1e-5);
# Ok::<(), luma::LumaError>(())
```

Which algorithm runs (one invocation per line, or one workgroup per line by cyclic reduction)
is chosen automatically; `LUMA_TRIDIAGONAL` forces one
([Environment variables](environment.md)). The cyclic one uses workgroup memory
([FAQ](faq.md#radv-integrated-gpus-give-wrong-results)).
