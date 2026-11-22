# luma vs ArrayFire 3.10.1 (CUDA 13.4): RTX 3080 Laptop

Same `afcompare` binary and method as `rtx3090.md` (jobs `afcmp-3080` and `afcmp-3080-cold` on the laptop through the GPU queue; luma = origin/cpu-offload 171232f,
before the MLP-step work of this branch). Every row is an interleaved (ABBA) measurement on this GPU; rows of the two GPUs are not comparable with each other.
After the MLP-step changes see `mlp-step.md`.

### NVIDIA GeForce RTX 3080 Laptop GPU - luma vs ArrayFire

Steady-state per-call time, median of the timed repetitions of an interleaved (ABBA) run; ratio = AF / luma (> 1: luma faster).

| case | size | luma ms | ArrayFire ms | ratio (AF/luma) | results agree? | note |
|---|---|---:|---:|---:|---|---|
| 1 ew chain, 8 fused ops | 1M | 0.0516 | 0.0625 | 1.21 | yes (max Δ 6.7e-7) | luma: 1 launches, 0.031 ms GPU |
| 1 ew chain, 8 fused ops | 16M | 0.4072 | 0.4204 | 1.03 | yes (max Δ 6.7e-7) | luma: 1 launches, 0.370 ms GPU |
| 1 ew chain, 8 fused ops | 64M | 1.539 | 1.557 | 1.01 | yes (max Δ 6.7e-7) | luma: 1 launches, 1.443 ms GPU |
| 1 ew chain, 32 fused ops | 16M | 0.4304 | 0.5968 | 1.39 | yes (max Δ 4.8e-7) | luma: 1 launches, 0.368 ms GPU |
| 2 launch overhead: 64 tiny ops, one eval | 4096 | 0.0950 | 0.3875 | 4.08 | yes (max Δ 2.4e-7) | chain built and evaluated every call; includes host-side graph building; luma: 1 launches, 0.010 ms GPU |
| 2b launch overhead: 64 tiny ops, eval each op | 4096 | 0.9257 | 0.5956 | 0.64 | yes (max Δ 2.4e-7) | extra: one eval per op on both sides; AF launches each, luma postpones and fuses evals while the GPU is busy (documented Array::eval behaviour); luma: 64 launches, 0.358 ms GPU |
| 3 reduce sum | 1M | 0.0288 | 0.0241 | 0.84 | yes (max Δ 0.0e0); luma vs f64 ref max Δ 4.9e-3; AF vs f64 ref max Δ 4.9e-3 | device-side 1-element result, no host copy; luma: 1 launches, 0.020 ms GPU |
| 3 reduce max | 1M | 0.0282 | 0.0235 | 0.83 | yes (max Δ 0.0e0); yes (max Δ 0.0e0) | luma: 1 launches, 0.020 ms GPU |
| 3 reduce sum | 100M | 1.158 | 1.109 | 0.96 | yes (max Δ 4.0e0); luma vs f64 ref max Δ 2.0e0; AF vs f64 ref max Δ 2.0e0 | device-side 1-element result, no host copy; luma: 1 launches, 1.092 ms GPU |
| 3 reduce max | 100M | 1.166 | 1.113 | 0.95 | yes (max Δ 0.0e0); yes (max Δ 0.0e0) | luma: 1 launches, 1.093 ms GPU |
| 3 reduce sum along axis 0 (contiguous lines) | 4096 x 4096 | 0.2055 | 0.1946 | 0.95 | yes (max Δ 4.9e-4); luma vs f64 ref max Δ 3.0e-4 | luma: 1 launches, 0.200 ms GPU |
| 3 reduce sum along axis 1 (strided) | 4096 x 4096 | 0.2340 | 0.2089 | 0.89 | yes (max Δ 4.9e-4); luma vs f64 ref max Δ 3.2e-4 | luma: 2 launches, 0.237 ms GPU |
| 3 mean + var (describe-like) | 1M x 32 (per feature) | 0.4753 | 1.830 | 3.85 | yes (max Δ 1.8e-7); yes (max Δ 9.7e-8) | population variance; both results evaluated together; luma: 5 launches, 0.790 ms GPU |
| 4 matmul f32 [square 1024] | 1024 x 1024 x 1024 | 0.3131 | 0.3056 | 0.98 | yes (max Δ 6.9e-5); luma within 5e-4 of f64 on 24 samples; AF within 5e-4 of f64 on 24 samples | luma matmul_f32 (SIMT f32) vs cuBLAS SGEMM; luma: 1 launches, 0.291 ms GPU; 6.9 TFLOP/s luma, 7.0 AF |
| 4 matmul TF32 [square 1024] | 1024 x 1024 x 1024 | 0.2041 | 0.2086 | 1.02 | yes (max Δ 1.6e-2); luma within 3e-2 of f64 on 24 samples; AF within 3e-2 of f64 on 24 samples | both TF32 (AF cuBLAS math mode 3); TF32 tolerance 3e-3*rms; luma: 1 launches, 0.198 ms GPU; 10.5 TFLOP/s luma, 10.3 AF |
| 4 matmul f32 [square 4096] | 4096 x 4096 x 4096 | 16.33 | 17.63 | 1.08 | yes (max Δ 4.3e-4); luma within 1e-3 of f64 on 24 samples; AF within 1e-3 of f64 on 24 samples | luma matmul_f32 (SIMT f32) vs cuBLAS SGEMM; luma: 1 launches, 16.096 ms GPU; 8.4 TFLOP/s luma, 7.8 AF |
| 4 matmul TF32 [square 4096] | 4096 x 4096 x 4096 | 9.364 | 8.789 | 0.94 | yes (max Δ 3.1e-2); luma within 6e-2 of f64 on 24 samples; AF within 6e-2 of f64 on 24 samples | both TF32 (AF cuBLAS math mode 3); TF32 tolerance 3e-3*rms; luma: 1 launches, 9.515 ms GPU; 14.7 TFLOP/s luma, 15.6 AF |
| 4 matmul f32 [square 8192] | 8192 x 8192 x 8192 | 132.7 | 115.7 | 0.87 | yes (max Δ 0.0e0); luma within 2e-3 of f64 on 24 samples; AF within 2e-3 of f64 on 24 samples | luma matmul_f32 (SIMT f32) vs cuBLAS SGEMM; luma: 1 launches, 135.081 ms GPU; 8.3 TFLOP/s luma, 9.5 AF |
| 4 matmul TF32 [square 8192] | 8192 x 8192 x 8192 | 70.63 | 65.31 | 0.92 | yes (max Δ 4.8e-2); luma within 9e-2 of f64 on 24 samples; AF within 9e-2 of f64 on 24 samples | both TF32 (AF cuBLAS math mode 3); TF32 tolerance 3e-3*rms; luma: 1 launches, 70.603 ms GPU; 15.6 TFLOP/s luma, 16.8 AF |
| 4 matmul f32 [dense] | 256 x 1024 x 1024 | 0.1126 | 0.0926 | 0.82 | yes (max Δ 5.7e-5); luma within 5e-4 of f64 on 24 samples; AF within 5e-4 of f64 on 24 samples | luma matmul_f32 (SIMT f32) vs cuBLAS SGEMM; luma: 1 launches, 0.105 ms GPU; 4.8 TFLOP/s luma, 5.8 AF |
| 4 matmul TF32 [dense] | 256 x 1024 x 1024 | 0.0662 | 0.0502 | 0.76 | yes (max Δ 1.6e-2); luma within 3e-2 of f64 on 24 samples; AF within 3e-2 of f64 on 24 samples | both TF32 (AF cuBLAS math mode 3); TF32 tolerance 3e-3*rms; luma: 1 launches, 0.061 ms GPU; 8.1 TFLOP/s luma, 10.7 AF |
| 4 matmul f32 [tall] | 2048 x 512 x 128 | 0.0580 | 0.0571 | 0.98 | yes (max Δ 3.1e-5); luma within 4e-4 of f64 on 24 samples; AF within 4e-4 of f64 on 24 samples | luma matmul_f32 (SIMT f32) vs cuBLAS SGEMM; luma: 1 launches, 0.054 ms GPU; 4.6 TFLOP/s luma, 4.7 AF |
| 4 matmul TF32 [tall] | 2048 x 512 x 128 | 0.0354 | 0.0428 | 1.21 | yes (max Δ 9.0e-3); luma within 2e-2 of f64 on 24 samples; AF within 2e-2 of f64 on 24 samples | both TF32 (AF cuBLAS math mode 3); TF32 tolerance 3e-3*rms; luma: 1 launches, 0.033 ms GPU; 7.6 TFLOP/s luma, 6.3 AF |
| 4 matmul f32 [attention P.V] | 2048 x 2048 x 64 | 0.1447 | 0.1059 | 0.73 | yes (max Δ 1.2e-4); luma within 8e-4 of f64 on 24 samples; AF within 8e-4 of f64 on 24 samples | luma matmul_f32 (SIMT f32) vs cuBLAS SGEMM; luma: 1 launches, 0.137 ms GPU; 3.7 TFLOP/s luma, 5.1 AF |
| 4 matmul TF32 [attention P.V] | 2048 x 2048 x 64 | 0.0805 | 0.0791 | 0.98 | yes (max Δ 1.8e-2); luma within 5e-2 of f64 on 24 samples; AF within 5e-2 of f64 on 24 samples | both TF32 (AF cuBLAS math mode 3); TF32 tolerance 3e-3*rms; luma: 2 launches, 0.079 ms GPU; 6.7 TFLOP/s luma, 6.8 AF |
| 4 matmul f32 [attention Q.K^T] | 2048 x 64 x 2048 | 0.1052 | 0.1140 | 1.08 | yes (max Δ 0.0e0); luma within 1e-4 of f64 on 24 samples; AF within 1e-4 of f64 on 24 samples | luma matmul_f32 (SIMT f32) vs cuBLAS SGEMM; luma: 1 launches, 0.096 ms GPU; 5.1 TFLOP/s luma, 4.7 AF |
| 4 matmul TF32 [attention Q.K^T] | 2048 x 64 x 2048 | 0.0695 | 0.0787 | 1.13 | yes (max Δ 4.8e-3); luma within 8e-3 of f64 on 24 samples; AF within 8e-3 of f64 on 24 samples | both TF32 (AF cuBLAS math mode 3); TF32 tolerance 3e-3*rms; luma: 1 launches, 0.066 ms GPU; 7.7 TFLOP/s luma, 6.8 AF |
| 5 batched small matmul f32 | 4096 x (32x32 @ 32x32) | 0.1459 | 0.6825 | 4.68 | yes (max Δ 0.0e0); luma within 9e-5 of f64 on 24 samples; AF within 9e-5 of f64 on 24 samples | luma matmul_batched (always f32); AF matmul with a batch dimension (cublasSgemmBatched); luma: 1 launches, 0.142 ms GPU |
| 6 FFT convolution (real) | signal 1M (x) kernel 4096 | 0.0989 | 0.2601 | 2.63 | yes (max Δ 4.0e-4); luma vs f64 ref max Δ 1.1e-4; AF vs f64 ref max Δ 2.2e-5 | luma conv1d_fft vs AF fft_convolve1 (cuFFT); luma has no standalone FFT API; luma: 5 launches, 0.103 ms GPU |
| 6 FFT convolution (real) | signal 16M (x) kernel 4096 | 1.253 | 5.082 | 4.06 | yes (max Δ 4.6e-4); luma vs f64 ref max Δ 1.1e-4; AF vs f64 ref max Δ 1.2e-5 | luma conv1d_fft vs AF fft_convolve1 (cuFFT); luma has no standalone FFT API; luma: 5 launches, 1.404 ms GPU |
| 6 FFT convolution (real) | signal 1M (x) kernel 1M | 0.7616 | 1.071 | 1.41 | yes (max Δ 4.4e-3); luma vs f64 ref max Δ 3.9e-4; AF vs f64 ref max Δ 1.1e-4 | luma conv1d_fft vs AF fft_convolve1 (cuFFT); luma has no standalone FFT API; luma: 8 launches, 0.769 ms GPU |
| 6 FFT 1D complex c2c (ArrayFire only) | 1M | n/a | 0.1909 | n/a | n/a (one side only) | luma has no FFT API: no luma row |
| 6 FFT 1D complex c2c (ArrayFire only) | 16M | n/a | 2.960 | n/a | n/a (one side only) | luma has no FFT API: no luma row |
| 6 FFT 2D complex c2c (ArrayFire only) | 2048 x 2048 | n/a | 0.6134 | n/a | n/a (one side only) | luma has no FFT API (no 2D convolution either): no luma row |
| 7 sort f32 | 10M | 1.814 | 1.383 | 0.76 | yes (max Δ 0.0e0); yes (max Δ 0.0e0) | luma: 16 launches, 1.836 ms GPU |
| 7 argsort f32 | 10M | 2.383 | 2.458 | 1.03 | luma valid permutation; AF valid permutation | luma argsort returns indices; AF sort_index also materialises the sorted values; luma: 16 launches, 2.352 ms GPU |
| 7 sort_by_key f32 keys, f32 values | 10M | 7.300 | 2.578 | 0.35 | yes (max Δ 0.0e0); luma values follow keys; AF values follow keys | luma has no sort_by_key: argsort + two takes (gathers); AF sort_by_key; luma: 17 launches, 7.235 ms GPU |
| 7 sort f32 | 100M | 17.53 | 13.18 | 0.75 | yes (max Δ 0.0e0); sorted | luma: 16 launches, 17.696 ms GPU |
| 7 sort along axis (each row) | 1M rows x 64 | 3.199 | 27.34 | 8.54 | yes (max Δ 0.0e0); yes (max Δ 0.0e0) | luma: 1 launches, 3.213 ms GPU |
| 8 cumsum f32 | 100M | 3.382 | 4.643 | 1.37 | yes (max Δ 0.0e0); yes (max Δ 0.0e0) | +-1 input (exact sums); luma: 5 launches, 3.347 ms GPU |
| 8 cumsum along axis 0 (contiguous lines) | 4096 x 4096 | 0.5763 | 0.4045 | 0.70 | yes (max Δ 0.0e0); yes (max Δ 0.0e0) | +-1 input (exact sums); luma: 3 launches, 0.571 ms GPU |
| 8 cumsum along axis 1 (strided) | 4096 x 4096 | 0.6266 | 0.7555 | 1.21 | yes (max Δ 0.0e0); yes (max Δ 0.0e0) | +-1 input (exact sums); luma: 3 launches, 0.623 ms GPU |
| 9 histogram 256 bins | 10M f32 in [0,1) | 0.1435 | 0.1340 | 0.93 | yes (max Δ 0.0e0); luma vs f64 ref max Δ 0.0e0; AF vs f64 ref max Δ 0.0e0 | edge rounding may move a few values between neighbouring bins; luma: 1 launches, 0.129 ms GPU |
| 10 gather (take / lookup) | 10M random indices into 10M | 2.387 | 3.514 | 1.47 | yes (max Δ 0.0e0); yes (max Δ 0.0e0) | luma: 1 launches, 2.294 ms GPU |
| 10 bincount (count into 64K bins) | 64M u32 keys | 2.142 | 2.090 | 0.98 | luma vs f64 ref max Δ 0.0e0; AF vs f64 ref max Δ 0.0e0 | luma bincount (atomic scatter-add of 1) vs AF histogram over u32 keys, 65536 bins: the only AF equivalent (unit weights); luma: 1 launches, 2.002 ms GPU |
| 10 scatter-add f32 (weighted) into 64K bins | 64M | 2.289 | 19.10 | 8.35 | luma vs f64 ref max Δ 8.8e-4; AF vs f64 ref max Δ 9.6e-5 | AF has no scatter-add: AF side is sort_by_key + sum_by_key (a different algorithm; valid because every bin is occupied); luma: 1 launches, 2.276 ms GPU |
| 11 stencil: 5-pt Jacobi, luma iterate_all | 2048 x 2048, 100 iterations | 7.501 | 16.31 | 2.17 | yes (max Δ 0.0e0) | periodic (shift); luma: recorded once, replayed as a fused loop; AF: shift kernels + fused add, evaluated every step; luma: 40 launches, 6.906 ms GPU |
| 11 stencil: 5-pt Jacobi, luma eval per step | 2048 x 2048, 100 iterations | 8.050 | 15.87 | 1.97 | yes (max Δ 0.0e0) | periodic (shift); one eval per step on both sides; AF launches each step (4 shift copies + 1 fused add), luma postpones and fuses evals while the GPU is busy (documented Array::eval behaviour); luma: 19 launches, 8.322 ms GPU |
| 11 stencil: 5-pt Jacobi, convolve2 variant | 2048 x 2048, 100 iterations | 14.57 | 30.23 | 2.07 | yes (max Δ 2.4e-7) | zero boundary: luma iterate_all (shift_fill) vs AF convolve2 3x3 (spatial) + fused add per step; luma: 100 launches, 14.607 ms GPU |
| 12 select(mask, a, b) | 10M (50% true) | 0.4701 | 0.4704 | 1.00 | yes (max Δ 0.0e0); yes (max Δ 0.0e0) | mask computed from x inside the timed call; luma: 1 launches, 0.444 ms GPU |
| 12 filter (mask compaction) | 10M (50% true) | 0.3340 | 1.366 | 4.09 | yes (max Δ 0.0e0); yes (max Δ 0.0e0) | luma filter vs AF where + lookup (y(mask)); both sizes data-dependent, result stays on device; luma: 7 launches, 0.323 ms GPU |
| 12 where / nonzero (indices) | 10M (50% true) | 0.2209 | 0.6027 | 2.73 | yes (max Δ 0.0e0); yes (max Δ 0.0e0) | luma: 7 launches, 0.219 ms GPU |
| 13 MLP SGD step, f32 | 784-512-512-10, batch 256 | 0.5841 | 0.2794 | 0.48 | yes (max Δ 2.4e-7); yes (max Δ 3.7e-9); all 6 params + loss compared; luma loss 2.3219, AF 2.3219; rms step 1.2e-2 | both f32 (luma matmul_f32, AF sgemm); luma: 21 launches, 0.588 ms GPU |
| 13 MLP SGD step, TF32 | 784-512-512-10, batch 256 | 0.3439 | 0.2140 | 0.62 | yes (max Δ 3.3e-6); yes (max Δ 6.4e-5); all 6 params + loss compared; luma loss 2.3219, AF 2.3219; rms step 1.2e-2; max deviation of the TF32 step from the AF f32 step: luma 1.7e-4, AF 1.7e-4 | both TF32 where eligible (luma: dims that are multiples of 4 and >= 32; the 10-class products stay f32); luma: 24 launches, 0.375 ms GPU |

### First-call (cold) times

Cold = first call in a fresh process, after the library's own initialisation and an unrelated 400 ms warm-up kernel; median of 5 processes per cell, children interleaved. `cold`: all on-disk kernel caches empty or off; `disk-warm`: caches filled by an earlier identical run. AF TF32 includes the cuBLAS math-mode call.

| case | luma cold ms | ArrayFire cold ms | ratio (AF/luma) | luma disk-warm ms | ArrayFire disk-warm ms | ratio (AF/luma) |
|---|---:|---:|---:|---:|---:|---:|
| 1 elementwise chain, 8 fused ops, 16M | 11.91 | 39.64 | 3.33 | 1.166 | 0.7105 | 0.61 |
| 4 matmul f32, 4096 x 4096 x 4096 | 93.11 | 39.09 | 0.42 | 12.18 | 41.84 | 3.43 |
| 4 matmul TF32, 4096 x 4096 x 4096 | 33.92 | 32.46 | 0.96 | 8.750 | 35.29 | 4.03 |
| 11 stencil: 5-pt Jacobi, 2048 x 2048, 100 iterations | 43.81 | 71.93 | 1.64 | 9.339 | 14.66 | 1.57 |
