### NVIDIA GeForce RTX 3090 - luma vs ArrayFire

Steady-state per-call time, median of the timed repetitions of an interleaved (ABBA) run; ratio = AF / luma (> 1: luma faster).

| case | size | luma ms | ArrayFire ms | ratio (AF/luma) | results agree? | note |
|---|---|---:|---:|---:|---|---|
| approx1 nearest | 1M samples, 1M positions | 0.0209 | 0.0210 | 1.00 | yes (max Δ 0.0e0) | increasing jittered positions; luma: 1 launches, 0.018 ms GPU |
| approx1 linear | 1M samples, 1M positions | 0.0216 | 0.0211 | 0.98 | yes (max Δ 0.0e0) | increasing jittered positions; luma: 1 launches, 0.018 ms GPU |
| approx1 cubic | 1M samples, 1M positions | 0.0216 | 0.0213 | 0.99 | yes (max Δ 5.4e-7) | increasing jittered positions; luma: 1 launches, 0.018 ms GPU |
| approx1 nearest | 4M samples, 4M positions | 0.0651 | 0.0636 | 0.98 | yes (max Δ 0.0e0) | increasing jittered positions; luma: 1 launches, 0.062 ms GPU |
| approx1 linear | 4M samples, 4M positions | 0.0648 | 0.0639 | 0.99 | yes (max Δ 0.0e0) | increasing jittered positions; luma: 1 launches, 0.062 ms GPU |
| approx1 cubic | 4M samples, 4M positions | 0.0654 | 0.0648 | 0.99 | yes (max Δ 5.1e-7) | increasing jittered positions; luma: 1 launches, 0.061 ms GPU |
| approx1 nearest | 16M samples, 16M positions | 0.2365 | 0.2357 | 1.00 | yes (max Δ 0.0e0) | increasing jittered positions; luma: 1 launches, 0.233 ms GPU |
| approx1 linear | 16M samples, 16M positions | 0.2366 | 0.2372 | 1.00 | yes (max Δ 0.0e0) | increasing jittered positions; luma: 1 launches, 0.233 ms GPU |
| approx1 cubic | 16M samples, 16M positions | 0.2396 | 0.2424 | 1.01 | yes (max Δ 5.4e-7) | increasing jittered positions; luma: 1 launches, 0.236 ms GPU |
| approx1 linear | 4M samples, 4M random positions | 0.3552 | 0.3514 | 0.99 | yes (max Δ 0.0e0) | random positions; luma: 1 launches, 0.351 ms GPU |
| approx1 linear, batched | 1024 x 4096 samples (4M positions) | 0.0668 | 0.0671 | 1.00 | yes (max Δ 0.0e0) | luma: 1 launches, 0.063 ms GPU |
| approx2 nearest | 1080p image, 1080p positions | 0.0461 | 0.0471 | 1.02 | yes (max Δ 0.0e0) | smooth warp; luma: 1 launches, 0.043 ms GPU |
| approx2 bilinear | 1080p image, 1080p positions | 0.0482 | 0.0469 | 0.97 | yes (max Δ 0.0e0) | smooth warp; luma: 1 launches, 0.045 ms GPU |
| approx2 bicubic | 1080p image, 1080p positions | 0.0543 | 0.0513 | 0.95 | yes (max Δ 4.8e-7) | smooth warp; luma: 1 launches, 0.051 ms GPU |
| approx2 nearest | 4K image, 4K positions | 0.1611 | 0.1651 | 1.03 | yes (max Δ 0.0e0) | smooth warp; luma: 1 launches, 0.158 ms GPU |
| approx2 bilinear | 4K image, 4K positions | 0.1683 | 0.1653 | 0.98 | yes (max Δ 0.0e0) | smooth warp; luma: 1 launches, 0.165 ms GPU |
| approx2 bicubic | 4K image, 4K positions | 0.1932 | 0.1837 | 0.95 | yes (max Δ 4.8e-7) | smooth warp; luma: 1 launches, 0.190 ms GPU |
