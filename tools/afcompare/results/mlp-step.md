# MLP SGD step (784-512-512-10, batch 256): luma against ArrayFire 3.10.1

Case 13 of `afcompare`, interleaved (ABBA) in one process per row; "before" is luma 171232f (job `afcmp-3090-final` on the
RTX 3090, `afcmp-3080` on the RTX 3080 Laptop), "after" is this branch (jobs `mlp-final-3090`, `mlp-final-3080`). ArrayFire's
own time moves by up to 25 % between jobs (GPU state), so compare the ratio, not the absolute ArrayFire column, across jobs.

| GPU | precision | luma before ms | ArrayFire ms | AF / luma | luma after ms | ArrayFire ms | AF / luma |
|---|---|---:|---:|---:|---:|---:|---:|
| RTX 3090 | f32 | 0.508 | 0.191 | 0.38 | 0.181 | 0.191 | 1.06 |
| RTX 3090 | TF32 | 0.287 | 0.171 | 0.60 | 0.176 | 0.172 | 0.98 |
| RTX 3080 Laptop | f32 | 0.584 | 0.279 | 0.48 | 0.274 | 0.286 | 1.04 |
| RTX 3080 Laptop | TF32 | 0.344 | 0.214 | 0.62 | 0.215 | 0.212 | 0.99 |

## Where the time went (RTX 3090, f32, luma's profiled call)

Before: 21 launches, 545 us of GPU time (the wall time of the step is its GPU time: 0.51 ms).

| kernels | GPU us | |
|---|---:|---|
| `FusedMatmul` (old shader kernels, 7 launches) | 370 | every product with a lazy or transposed-lazy operand (`relu(z) W`, `relu(z)ᵀ dz`, the masked gradients) or `N`/`K` = 10: 25-85 us each at 2-3 TFLOP/s |
| `Matmul` Gemm 32x32 (the only product that qualified) | 36 | |
| axis reductions (bias gradients, softmax) | 120 | one of 38 us: `db3 = dl.sum_axis(1)`, 10 lines of 256 cells, 10 invocations, with `dl`'s expression recomputed inside |
| the rest (weight updates, loss, activations) | 20 | |

Per product in isolation (`afcompare diag`, f32): `relu(z1) W2` 85 us (fused) -> 3.8 + 21 us (evaluated, then gemm);
`relu(z2) W3` (N = 10) 49 -> 30 (pad 3.9 + relu 3.9 + gemm 19.5, 10 with the K split); `relu(z2)ᵀ dL` 25 -> 20 -> 11;
`relu(z1)ᵀ dz2` 45 -> 19.5; `dz2 W2ᵀ`, `xᵀ dz1` 22 / 19 (already on the gemm kernels).

After: 34 launches, 237 us of GPU time. The launch count went up (operands evaluated first once, the padded head operand,
the split-K sums) but every launch is a few microseconds and the 370 us of old shader kernels are gone.

## What changed (src/matmul.rs, src/scan.rs)

1. The `f32` gemm path takes operands that are lazy expressions (and transposed views of them): they are evaluated first,
   once, as the TF32 path already did (`gemm_operand`). Their other readers then use the buffer.
2. Few-column products (`N` of 4 to 31 that is not a multiple of 4, 128 <= `M` <= 4096): `N` padded with zeros (a lazy pad,
   evaluated first), product on the gemm kernels, result sliced back; `N` of 4 to 31 that is a multiple of 4 is taken
   by the gemm kernels directly (`padded_product`, `gemm_shape`).
3. Split K for products of at most 16 output tiles from K = 512 on (eighth of K between 128 and 512 / 1024 per slice): 256 x 512 x 12
   18 -> 9 us, 2048 x 2048 x 64 f32 95 -> 72 us (attention `P V` against ArrayFire: 0.57 -> 0.79), 64 x 8192 x 64 TF32 34 -> 10 us.
   More tiles keep the old thresholds (splitting 256 x 512 x 512 is slower).
4. A reduction along a short strided axis (`n` of 65 to 256 cells) with fewer than 1024 lines is blocked in 16 cells per
   invocation with a second pass (`reduce_block`): the 38 us bias-gradient reduction takes 8 us.

## What is still slower than ArrayFire

* f32 products with a lhs stored along K (`P V` of 2048 x 2048 x 64: 72 us against cuBLAS 58 us; the PTX kernel loads that operand
  through registers and transposes it into shared memory; with the left operand transposed in memory it takes 47 us).
* 256 x 1024 x 1024 (0.82 / 0.83 of ArrayFire f32 / TF32) and the other small-M products: 32x32 tiles at 13 TFLOP/s peak
  against cuBLAS's tuned small-M kernels.
* The step has 34 launches (ArrayFire about 30 calls); the remaining gap on TF32 is spread over the small kernels.
