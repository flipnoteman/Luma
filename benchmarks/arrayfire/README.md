# Luma vs ArrayFire

Benchmarks of seven workloads written twice: once with Luma (wgpu, Vulkan on Linux) and once with ArrayFire 3.x
(CUDA and OpenCL backends) through the [`arrayfire`](https://crates.io/crates/arrayfire) crate 3.8. Every case is
checked against a multithreaded CPU reference before it is timed.

This is its own Cargo workspace. It is not a member or dev-dependency of the `luma` package, so `cargo build` and
`cargo test` of Luma never need ArrayFire. `luma-bench` depends on Luma by path (`../../..`).

## Requirements

- The Rust toolchain used for Luma.
- An ArrayFire 3.9 or 3.10 binary install, from <https://arrayfire.com/download> or the GitHub releases of
  <https://github.com/arrayfire/arrayfire>. It can be installed without root:
  `sh ArrayFire-*.sh --prefix=<dir> --exclude-subdir --skip-license`.
- A GPU (and drivers for the backends you run: CUDA and/or OpenCL, plus Vulkan for Luma).
- Python 3 for the report.

## Running

```sh
AF_PATH=<dir> ./run.sh            # full run
AF_PATH=<dir> ./run.sh --quick    # small sizes, for checking the setup
AF_PATH=<dir> AF_BACKENDS=cuda ./run.sh
```

`AF_PATH` is the ArrayFire install prefix (it contains `include/arrayfire.h` and `lib64/` or `lib/`). `run.sh`
builds in release mode with `-C target-cpu=native` (appended to any existing `RUSTFLAGS`), sets the rpath and
`LD_LIBRARY_PATH` to the ArrayFire library directory, then runs, one after another:

1. `luma-bench` into `results/luma.csv`;
2. `af-bench --backend <b>` into `results/af-<b>.csv` for each `b` in `AF_BACKENDS` (default `"cuda opencl"`).
   A backend that fails is reported on stderr and skipped.
   Each backend then runs again with `AF_CUDA_MAX_JIT_LEN=256 AF_OPENCL_MAX_JIT_LEN=256` and `--label <b>-tuned` into
   `results/af-<b>-tuned.csv`, so ArrayFire is also shown with a larger JIT tree limit. A failed tuned run is skipped too.
3. `report.py results/*.csv > results/report.md`, which is then printed.

`report.py` works with whichever CSVs exist. A single binary can be run directly, for one workload:

```sh
export LD_LIBRARY_PATH=$AF_PATH/lib64
target/release/luma-bench --only mandelbrot
target/release/af-bench --backend cuda --only fir --quick
```

## Workloads

Sizes are in `common/src/lib.rs::sizes` (`--quick` uses much smaller ones). Timing (`common::time`): 3 warm-ups, then
the median of timed calls, until 1 s and at least 5 runs have passed or 200 runs are done.

| workload | what | variants |
|---|---|---|
| `heat` | 2D periodic 5-point heat stencil, 100 steps, 1024², 4096², 8192² f32 | `per_step`: the host loop builds one step per iteration from array operations and calls `eval` after every step, in both libraries. `lazy_chain` (Luma only): the same loop without per-step `eval`, so the whole chain stays lazy. `iterate` (Luma only): `Array::iterate`, a loop run by the library. `per_step_cold`: the first call of `per_step` in the process, once, with no warm-up (includes shader/kernel compilation). |
| `life` | Game of Life on a torus, f32 cells, 100 steps, same grid sizes | `per_step`, `lazy_chain` and `iterate`, as for `heat`. |
| `converge` | Heat stencil, 200 steps, 256² and 1024² (quick: 256²), reading the largest per-step change back to the host after every step and stopping if it is below 1e-30 (never, so the step count is fixed) | `readback_every_step`. |
| `blackscholes` | Black-Scholes call prices, 2^24 and 2^26 options | `price`: compute and evaluate the price array. `price_sum`: reduce the prices to one scalar (no materialized output). |
| `mandelbrot` | Fixed 256 iterations of the escape-time recurrence written as array ops (per-pixel freeze once outside, no early exit), 1024² and 4096² | `eval_every_N`: evaluate `z` and the count every N iterations (N = 1, 8). `eval` does not wait for the GPU in either library. `eval_end_only`: one evaluation after the whole chain. `iterate_all` (Luma only): the same step through `iterate_all`, one kernel for the whole loop. `eval_every_1_cold`: the first call of `eval_every_1` in the process, once, no warm-up. `custom_kernel` (Luma only): hand-written WGSL kernel with per-pixel early exit. |
| `fir` | Full linear convolution of 2^24 samples with a filter of 127, 255, 4095 or 65535 taps | `fft`: FFT convolution. `direct`: spatial convolution (up to 4095 taps). |
| `montecarlo` | Array-code Monte Carlo over 2^24 and 2^26 samples: an estimate of pi from two uniform arrays, and a European call from normal draws, both reduced to one mean | `pi`, `european`. Luma generates the random numbers inside the reduction kernel; ArrayFire's `randu`/`randn` write them to memory first. Checked against pi and the Black-Scholes price. |

## Reading the results

- Timings include device completion but not host-to-device uploads or readback.
- `eval` is non-blocking in both libraries, so the same variant names compare the same work. `report.py` prints a
  "Like for like" table per workload for this.
- `_cold` rows are one call with no warm-up, so they include first-use kernel compilation. `report.py` leaves them and `lazy_chain` out of the best-of-variants comparison.
- `iterate` and `iterate_all` are Luma features with no ArrayFire counterpart.
- `custom_kernel` is not comparable (not array code, per-pixel early exit), so the report leaves it out of the
  best-of-variants comparison.
- `eval_end_only` shows how each JIT splits a long unevaluated chain. It is not how ArrayFire code is normally written.
- ArrayFire's spatial `convolve1` rejects filters of about 200 taps and more on GPU backends, so `direct` rows
  are missing there.
- Run-to-run noise of 20% or more has been observed on a desktop GPU. Repeat runs before trusting small differences.
