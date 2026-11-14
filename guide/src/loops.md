# Loops

Time stepping, iterative solvers and per-element recurrences all repeat one step many times.
Written as a Rust loop, each iteration builds a graph and the host talks to the GPU once per step.
Luma has loop primitives that take the step *as a closure*, trace it once, and run all the steps
on the GPU:

| Primitive | Use |
| --- | --- |
| [`Array::iterate`](https://docs.rs/luma/latest/luma/struct.Array.html#method.iterate) | `steps` applications of a step on one `f32` array; stencils run time-blocked |
| [`iterate_all`](https://docs.rs/luma/latest/luma/fn.iterate_all.html) | `steps` applications of a step on several arrays of one element type; elementwise `f32` and `u32` steps keep the state in registers |
| [`Array::iterate_until`](https://docs.rs/luma/latest/luma/struct.Array.html#method.iterate_until) | a step until the largest change is below a tolerance |
| [`iterate_until_all`](https://docs.rs/luma/latest/luma/fn.iterate_until_all.html) | a step on any [record](records.md) until a rule over recorded scalars holds |

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
`assign`, reductions) is traced and replayed step by step: each step is still the fused kernels
of the step, and the loop costs no host time per step.

A step that is a stencil of 2-D data looks the same: `u.shift([1, 0]) + u.shift([-1, 0]) +
u.shift([0, 1]) + u.shift([0, -1]) - u * 4.0`.

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
and counters (see [Integer and bit operations](integers.md)):

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
(`Df32`, `f64`), is traced once and its recorded kernels are replayed step by step. The results
are the same; only the one-kernel register loop is lost.

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
  `peak()` (largest so far), `trough()`, `window_max(k)`, `window_min(k)`.
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
a time-blocked stencil. The graph that is pending stays one step deep however long the loop runs.
A step that reads slices or pads of lazy expressions is evaluated step by step; each step still
fuses.

Compared with the loop primitives, an `eval` loop calls your step every iteration (host work per
step) but may use anything inside it, including readbacks. Use the primitive when the step fits
and the host time matters.

### Readbacks in a loop

`step` followed by `.max().item()?` every iteration waits once per step. When the same step has
been seen repeating, Luma computes the next few steps and their residuals during a readback and
answers the following readbacks from memory (*speculation*): the round trip is paid once per
several steps. Values are those of the program as written; a miss only costs the wasted GPU
work. `LUMA_SPECULATE=off` disables it ([Environment variables](environment.md)). If the loop
only needs a convergence test, `iterate_until` and `iterate_until_all` wait once per block of
steps instead of once per step.
