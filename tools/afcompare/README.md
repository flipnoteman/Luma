# afcompare: luma vs ArrayFire 3.10 (CUDA)

A standalone benchmark that runs the same operations, on the same inputs, on the same GPU in
luma and in ArrayFire, interleaved, and checks that the results agree. It is **not** a member
of the luma workspace (`[workspace]` in its `Cargo.toml`): normal luma builds, `clippy` and
`nextest` never need ArrayFire.

## Build

ArrayFire 3.10 with the CUDA backend must be installed (`/opt/arrayfire`, libraries in
`lib64`, registered with `ldconfig`). The Rust `arrayfire` 3.8.0 crate only binds the C API
and works with the 3.10 libraries (the unified `libaf`, which loads `libafcuda` at run time).

```sh
cd tools/afcompare
AF_PATH=/opt/arrayfire cargo build --release     # target/release/afcompare
```

## Run

All GPU timing of this repository goes through the GPU queue (`tools/gpuq/gpuq` of the
`bowshock` checkout); build first, then queue a job that only runs the binary:

```sh
BIN=$PWD/tools/afcompare/target/release/afcompare
gpuq submit --host local --name afcmp --timeout 1800 -- \
    bash -c "AF_PATH=/opt/arrayfire LUMA_BACKEND=cuda $BIN --out out"
gpuq submit --host local --name afcmp-cold --timeout 3600 -- \
    bash -c "AF_PATH=/opt/arrayfire LUMA_BACKEND=cuda $BIN cold --out out"
```

(For the other machine, `rsync` the binary to the same path there and use `--host remote`; it needs
the same ArrayFire install.)

* `afcompare [--only SUBSTR] [--quick] [--out DIR]`: steady-state table
  (`DIR/results.md`, `DIR/results.jsonl`). `--only` selects case groups by name
  (`ew small reduce matmul bmm fft sort scan hist gather stencil filter data mlp sparse image accum`).
* `afcompare cold [--reps N] [--only CASE] [--out DIR]`: first-call (cold) times, each
  measurement in a fresh process (`DIR/cold.md`).
* `AFCOMPARE_KERNELS=1` prints luma's per-kernel GPU time of every timed luma call (a profiled call).
* `afcompare diag` (luma only, no ArrayFire): GPU time and kernels of every product of the MLP step
  with the operand kinds the step gives it (stored, lazy, transposed), of its reductions and softmax
  pieces. `afcompare sweep`: GPU time of `matmul_f32` / `matmul_tf32` (stored and left-transposed
  operands) over product shapes, for comparing planner settings; run it after a warm-up (it wakes
  the GPU for 400 ms first) and compare only runs of one job.

## Method

* Both libraries get identical host data (deterministic generator). Every case first computes the
  result with both, compares them (tolerance appropriate to the operation, many also against an
  `f64`/host reference) and prints `MISMATCH` loudly if they disagree.
* Steady state: each side is warmed up for at least 250 ms (as `luma-bench` does), then
  timed repetitions alternate ABBA / BAAB (luma, ArrayFire, ArrayFire, luma, ...), each fully
  synchronised (luma `eval` + `Context::sync`; ArrayFire `eval` + `af::sync`); the median is
  reported. Results stay on the device (no host copies inside the timed region), except where an
  operation's result size is data dependent (filter), in which case both libraries need a sync.
* Cold times are never mixed with the steady-state ones: see `cold.rs`.
* Matmul: `matmul_f32` vs cuBLAS SGEMM, and luma's TF32 product vs ArrayFire with the cuBLAS math
  mode set to TF32 through `afcu_cublasSetMathMode` (looked up with `dlsym` in `libafcuda`; the
  program probes that the mode really changes ArrayFire's results and says so; if it does not,
  the TF32 rows are labelled as a different precision).
* FFT (group `fft`): `fft`/`ifft` (1M, 16M, 2^20, 2^24, 1000003 points), `rfft` against `fft_r2c`,
  `fft2`/`ifft2` (2048 x 2048), `fft3` (256^3), `fft_convolve2`/`fft_convolve3` against
  ArrayFire's `fft_convolve2`/`fft_convolve3` (same-size output, odd kernels) and `conv1d_fft` against
  `fft_convolve1`. ArrayFire's `ifft` takes the `1 / N` scale as an argument; luma's `ifft` has it built
  in (`FftNorm::Backward`), so both sides produce the same values. `AFCOMPARE_KERNELS=1` prints luma's
  per-kernel times.
* ArrayFire has no scatter-add; the weighted case uses `sort_by_key` + `sum_by_key`, the unweighted
  count uses `histogram`; both are labelled as different algorithms.

## Sparse group

`--only sparse` compares luma's `Sparse` with ArrayFire's sparse arrays (`src/sparse.rs`): CSR
products with a vector and 16 dense columns (plain and transposed) on banded and power-law
matrices of 1M and 10M entries, conversions, construction from triplets and from a dense matrix,
`to_dense` and elementwise operations with a dense matrix. ArrayFire 3.10.1's sparse entry points
(`matmul`, `sparseConvertTo`, `sparse(dense)`, `dense(sparse)`) **segfault** on the machines this was
measured on (in `cusparseCreateSpMatDescr`, a call into the cuSPARSE the build ships), which
`catch_unwind` cannot survive, so each ArrayFire operation is probed first in a child process
(`afcompare sparse-probe OP`) and an operation that crashes has no ArrayFire side. For the
products and the CSR -> CSC conversion the reference is then **cuSPARSE called directly**
(`src/cusparse.rs`: `cusparseSpMV` / `cusparseSpMM` with the default algorithms, the workspace
allocated once, and `cusparseCsr2cscEx2`), which is what ArrayFire's wrappers call; the row's note
says which one it is. Rows with no reference are luma-only.
