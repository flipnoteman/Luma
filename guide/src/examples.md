# Worked examples

Five tours of programs in the repository. The first two have the full code here, runnable; for the
others the chapter points to the code and says what to look at. Run the repository examples
with `cargo run --release --example <name>` from a checkout of
[the repository](https://github.com/flipnoteman/Luma).

## Heat equation

The diffusion of heat on a periodic grid is the smallest program that uses stencils and loops. One
step is an elementwise expression of the grid and four shifted copies of it:

```rust
use luma::*;

fn heat(u: &Array<f32>) -> Array<f32> {
    let lap = u.shift([1, 0]) + u.shift([-1, 0]) + u.shift([0, 1]) + u.shift([0, -1]) - u * 4.0;
    u + lap * 0.2
}

let n = 16;
let mut data = vec![0.0f32; n * n];
data[n / 2 + n * (n / 2)] = 1.0;                    // a spike in the middle
let u0 = Array::from_slice(&data, [n, n])?;

// A fixed number of steps. `heat` is a stencil, so Luma runs several steps per pass over memory.
let u = u0.iterate(100, heat);
assert!((u.sum().item()? - 1.0).abs() < 1e-4);      // periodic: the total heat is conserved
assert!(u.max().item()? < 0.05);                    // the spike has spread

// Until the field stops changing, with the test on the GPU.
let steady = u0.iterate_until(10_000, 1e-6, heat);
for v in steady.to_vec()? {
    assert!((v - 1.0 / (n * n) as f32).abs() < 1e-4);
}
# Ok::<(), luma::LumaError>(())
```

What to look at:

- `heat` is written once, with ordinary arithmetic, and used by `iterate` and `iterate_until`
  unchanged ([Loops](loops.md)). Nothing in it knows about buffers.
- `examples/bench_pde.rs` has the same step in four convergence loops
  (`converge/readback_loop`, `converge/iterate_until`, `converge/until_all`, `converge/iterate`):
  `cargo run --release --example bench_pde -- --only converge`. The readback loop pays a GPU round
  trip per step; the others do not, and the table shows the difference in cell updates per second.
- `Profile` of a few steps shows a single time-blocked kernel for `iterate`
  ([Performance](performance.md)).

## Euler and MUSCL solvers with records

A finite-volume solver for the Euler equations is the example Luma's views, records and writes were
designed for: a state of several fields per cell, ghost cells from a boundary rule, face states as
slices of the padded state, a flux per face, and a difference of fluxes per cell. A 1-D version, in
about thirty lines:

```rust
use luma::*;

const GAMMA: f32 = 1.4;

luma::fields! {
    /// Density, momentum, total energy per cell.
    struct Cons {
        rho: Array<f32>,
        mu: Array<f32>,
        e: Array<f32>,
    }
}

fn pressure(q: &Cons) -> Array<f32> {
    (GAMMA - 1.0) * (&q.e - 0.5 * &q.mu * &q.mu / &q.rho)
}

/// The physical flux of a state.
fn flux(q: &Cons) -> Cons {
    let u = &q.mu / &q.rho;
    let p = pressure(q);
    Cons { rho: &q.mu * 1.0, mu: &q.mu * &u + &p, e: (&q.e + &p) * &u }
}

/// The Rusanov (local Lax-Friedrichs) flux between a left and a right state.
fn rusanov(l: &Cons, r: &Cons) -> Cons {
    let speed = |q: &Cons| (&q.mu / &q.rho).abs() + (GAMMA * pressure(q) / &q.rho).sqrt();
    let smax = speed(l).maximum(speed(r));
    0.5 * (flux(l) + flux(r)) - 0.5 * &smax * (r - l)
}

/// One first-order step: ghost cells, face fluxes, update.
fn step(q: &Cons, dt_dx: f32) -> Cons {
    let n = q.rho.len();
    let w = q.pad([(1, 1)], PadMode::Edge);                        // one ghost cell per side
    let f = rusanov(&w.slice(0, 0..n + 1), &w.slice(0, 1..n + 2)); // n + 1 faces
    q - dt_dx * (f.slice(0, 1..n + 1) - f.slice(0, 0..n))
}

// Sod's shock tube: high pressure on the left half, low on the right.
let n = 100;
let half = |left: f32, right: f32| -> Vec<f32> { (0..n).map(|i| if i < n / 2 { left } else { right }).collect() };
let (rho, p) = (half(1.0, 0.125), half(1.0, 0.1));
let mut q = Cons {
    e: Array::from_slice(&p.iter().map(|p| p / (GAMMA - 1.0)).collect::<Vec<_>>(), n)?,
    rho: Array::from_slice(&rho, n)?,
    mu: Array::<f32>::zeros(n)?,
};

let mass = q.rho.sum().item()?;
for _ in 0..50 {
    q = step(&q, 0.1);
    q.eval()?;                                    // record the step; do not wait
}
assert!((q.rho.sum().item()? - mass).abs() < 1e-3);          // the waves have not reached the walls
assert!(q.rho.min().item()? > 0.1);                          // and the density stays positive
# Ok::<(), luma::LumaError>(())
```

What to look at:

- A **record** (`Cons`) is three arrays that move together. `q.pad`, `w.slice`, `0.5 * (..) - ..`
  and `q - dt_dx * (..)` each apply to all three fields. The code is the maths of the scheme;
  the fields are never listed ([Records](records.md)).
- The **ghost cells** are a pad. A real boundary condition is an assignment into the pad's ghost
  region (`w.slice_mut(0, ..1).assign(mirror(..))`, [Views](views.md#ghost-cells)); the repository
  versions do that for a reflecting wall.
- The fluxes are read at two different **offsets** by the two cells next to a face: slices of the
  padded state through the flux expression. Luma either computes each flux once per face (staging in
  workgroup memory, or evaluating it first) or recomputes it per read, whichever the
  [cost model](performance.md#the-cost-model) estimates is cheaper. `ctx.profile` shows how many
  kernels that took.
- `q.eval()` hands the step over without waiting; the loop is a per-step `eval` loop
  ([Loops](loops.md#loops-with-eval)).
- The 2-D version has the same structure with `[(1, 1), (1, 1)]` pads and slices along both axes
  (the `muscl_faces` of the benchmark adds minmod-limited slopes and two ghost cells).

In the repository:

- [`tests/euler_residual.rs`](https://github.com/flipnoteman/Luma/blob/main/tests/euler_residual.rs):
  a 2-D first-order residual with a mirror-wall ghost assignment, compared with an `f64` host
  implementation. `cargo test --release --test euler_residual`.
- [`examples/bench_pde.rs`](https://github.com/flipnoteman/Luma/blob/main/examples/bench_pde.rs): the
  Euler (`euler/*`) and MUSCL (`muscl/*`) solvers in a handful of variants: a step per `eval`,
  `cache()` of the padded state, `iterate_all`, `iterate_until_all`, and a hand-written
  custom WGSL kernel for comparison. `cargo run --release --example bench_pde -- --quick --only
  euler` prints cell updates per second per variant, with kernel counts and where the expressions
  were cut.

## ML-KEM

[`examples/mlkem.rs`](https://github.com/flipnoteman/Luma/blob/main/examples/mlkem.rs) is ML-KEM
(FIPS 203) KeyGen, Encaps and Decaps for ML-KEM-512, -768 and -1024 over a batch of independent
instances, written only with array operations: no custom kernel, no WGSL. It is the largest
program in the repository and the best look at the integer side of Luma.

```sh
cargo run --release --example mlkem -- --set 768 --op keygen --n 4096
cargo test --release --test mlkem          # known answers and NIST ACVP vectors
```

What to look at (the header of the file explains each):

- **Keccak** as `iterate_all` of one round, 24 times, with the round constants coming from the
  LFSR carried as a state word; hash inputs are views of byte-string arrays.
- **Data layout**: a batch of byte strings is a `[words, n]` array of `u32`; a batch of polynomials
  is `[256, rows]`. Uploads and readbacks need no reshuffle.
- **NTT** as two kernels of register butterflies, each followed by a join of its outputs
  (`concat`/`stack`).
- **Rejection sampling** with a scan over groups of candidates and one scatter (`put`) per
  matrix; `take` for ByteEncode/ByteDecode and Compress/Decompress; Montgomery reduction for
  modular products in `u32` ([Integers](integers.md)).
- The results are checked **byte for byte** against the RustCrypto `ml-kem` crate (a tampered
  ciphertext per instance included) and against ACVP vectors; the table reports operations per
  second with data on the device and with transfers, GPU time, host time and launches per batch.
  `MLKEM_PROFILE=1` lists the kernels of every row.

## Keccak

[`examples/keccak.rs`](https://github.com/flipnoteman/Luma/blob/main/examples/keccak.rs) is
Keccak-f[1600], SHA3-256 and SHAKE128 over a batch of inputs, lazy operations only. A 64-bit lane is
a pair `(lo, hi)` of `u32` arrays with one element per instance; the permutation is a Rust function
of 24 rounds of `^ & | ! << >>` on those arrays ([the lane rotation](integers.md#64-bit-lanes-as-pairs-of-u32)
is the whole of the machinery).

```sh
cargo run --release --example keccak -- --quick
cargo test --release --test keccak
```

What to look at:

- `keccak_round_with`: theta, rho-pi, chi and iota as array expressions; the constants of rotations
  are host numbers, so the shifts compile with constant amounts.
- The two rows of the table: **one permutation per `eval`** and **`iterate_all` over many
  permutations**. The second traces the permutation once and runs it as a register loop where the
  step is elementwise and fits (the `launches/perm` column and the profile after the table show
  how many kernels it took). The `host ms` column is only the tracing of the step: later calls
  run from the program of the first (the header of the file explains).
- The profile after the table: kernels per permutation and the reasons for the cuts.
- The known-answer tests in `tests/keccak.rs`, and the test that a chain of eight permutations
  (about 80 000 operations) is cut into at most 60 kernels (it is skipped on CPU and Apple
  profiles and without timestamp queries).

## CPA hypothesis matrix

A correlation power analysis of a block cipher guesses a key byte `k` and correlates the measured
power with a model of the intermediate value `HW[SBOX[pt ^ k]]` for every trace's plaintext byte
`pt`. All 256 guesses over thousands of traces are a `[traces, 256]` matrix: a gather from the S-box
table, a gather from the Hamming-weight table, and a Pearson correlation along the trace axis.

The complete, runnable program is in [Integer and bit operations](integers.md#lookup-tables-with-take).
What to look at:

- the key guesses are an array along axis 1 (`Array::<u32>::range([1, 256], 1)`) and the plaintexts
  an array along axis 0, so `&pt ^ &guesses` broadcasts to the whole matrix;
- `take` does both table lookups, and `sum_axis(0)` of centred products gives the covariance
  columns, all broadcasting: the program never writes a loop over guesses or traces;
- the gathers and the elementwise products fuse; `sum_axis` does not fuse its operand, so the
  centred and multiplied `[traces, 256]` arrays are computed before it. Wrap the analysis in
  `ctx.profile(..)` and count `launches()` to see.

Replace the stand-in S-box with your cipher's table and the synthetic leakage with a measured
sample per trace to use it on real data.
