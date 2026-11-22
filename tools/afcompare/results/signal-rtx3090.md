### NVIDIA GeForce RTX 3090 - luma vs ArrayFire

Steady-state per-call time, median of the timed repetitions of an interleaved (ABBA) run; ratio = AF / luma (> 1: luma faster).

| case | size | luma ms | ArrayFire ms | ratio (AF/luma) | results agree? | note |
|---|---|---:|---:|---:|---|---|
| approx1 nearest | 1M samples, 1M positions | 0.0221 | 0.0209 | 0.95 | yes (max Δ 0.0e0) | increasing jittered positions; luma: 1 launches, 0.019 ms GPU |
| approx1 linear | 1M samples, 1M positions | 0.0223 | 0.0209 | 0.94 | yes (max Δ 0.0e0) | increasing jittered positions; luma: 1 launches, 0.020 ms GPU |
| approx1 cubic | 1M samples, 1M positions | 0.0230 | 0.0211 | 0.92 | yes (max Δ 5.4e-7) | increasing jittered positions; luma: 1 launches, 0.018 ms GPU |
| approx1 nearest | 4M samples, 4M positions | 0.0647 | 0.0636 | 0.98 | yes (max Δ 0.0e0) | increasing jittered positions; luma: 1 launches, 0.060 ms GPU |
| approx1 linear | 4M samples, 4M positions | 0.0657 | 0.0638 | 0.97 | yes (max Δ 0.0e0) | increasing jittered positions; luma: 1 launches, 0.061 ms GPU |
| approx1 cubic | 4M samples, 4M positions | 0.0681 | 0.0646 | 0.95 | yes (max Δ 5.1e-7) | increasing jittered positions; luma: 1 launches, 0.065 ms GPU |
| approx1 nearest | 16M samples, 16M positions | 0.2348 | 0.2360 | 1.00 | yes (max Δ 0.0e0) | increasing jittered positions; luma: 1 launches, 0.230 ms GPU |
| approx1 linear | 16M samples, 16M positions | 0.2381 | 0.2375 | 1.00 | yes (max Δ 0.0e0) | increasing jittered positions; luma: 1 launches, 0.233 ms GPU |
| approx1 cubic | 16M samples, 16M positions | 0.2474 | 0.2428 | 0.98 | yes (max Δ 5.4e-7) | increasing jittered positions; luma: 1 launches, 0.243 ms GPU |
| approx1 linear | 4M samples, 4M random positions | 0.3532 | 0.3513 | 0.99 | yes (max Δ 0.0e0) | random positions; luma: 1 launches, 0.349 ms GPU |
| approx1 linear, batched | 1024 x 4096 samples (4M positions) | 0.0684 | 0.0669 | 0.98 | yes (max Δ 0.0e0) | luma: 1 launches, 0.064 ms GPU |
| approx2 nearest | 1080p image, 1080p positions | 0.0485 | 0.0468 | 0.97 | yes (max Δ 0.0e0) | smooth warp; luma: 1 launches, 0.044 ms GPU |
| approx2 bilinear | 1080p image, 1080p positions | 0.0497 | 0.0470 | 0.95 | yes (max Δ 0.0e0) | smooth warp; luma: 1 launches, 0.045 ms GPU |
| approx2 bicubic | 1080p image, 1080p positions | 0.0621 | 0.0516 | 0.83 | yes (max Δ 4.8e-7) | smooth warp; luma: 1 launches, 0.057 ms GPU |
| approx2 nearest | 4K image, 4K positions | 0.1672 | 0.1646 | 0.98 | yes (max Δ 0.0e0) | smooth warp; luma: 1 launches, 0.163 ms GPU |
| approx2 bilinear | 4K image, 4K positions | 0.1714 | 0.1652 | 0.96 | yes (max Δ 0.0e0) | smooth warp; luma: 1 launches, 0.167 ms GPU |
| approx2 bicubic | 4K image, 4K positions | 0.2236 | 0.1841 | 0.82 | yes (max Δ 4.8e-7) | smooth warp; luma: 1 launches, 0.220 ms GPU |
| resize nearest | 1080p -> 4K | 0.0733 | 0.0711 | 0.97 | yes (max Δ 0.0e0) | luma: 1 launches, 0.070 ms GPU |
| resize bilinear | 1080p -> 4K | 0.0959 | 0.0869 | 0.91 | yes (max Δ 1.2e-7) | luma: 1 launches, 0.091 ms GPU |
| resize bicubic | 1080p -> 4K | 0.1756 | 0.2277 | 1.30 | yes (max Δ 3.6e-7) | luma: 1 launches, 0.170 ms GPU |
| resize nearest | 4K -> 1080p | 0.0381 | 0.0364 | 0.96 | yes (max Δ 0.0e0) | luma: 1 launches, 0.034 ms GPU |
| resize bilinear | 4K -> 1080p | 0.0565 | 0.0548 | 0.97 | yes (max Δ 0.0e0) | luma: 1 launches, 0.052 ms GPU |
| resize bicubic | 4K -> 1080p | 0.0569 | 0.0992 | 1.74 | yes (max Δ 0.0e0) | luma: 1 launches, 0.053 ms GPU |
| resize bilinear RGB | 1080p -> 4K | 0.2760 | 0.2517 | 0.91 | yes (max Δ 1.2e-7) | luma: 1 launches, 0.271 ms GPU |
| resize bilinear RGB | 4K -> 1080p | 0.1522 | 0.1505 | 0.99 | yes (max Δ 0.0e0) | luma: 1 launches, 0.148 ms GPU |
| rotate nearest | 1080p, 0.3 rad, crop | 0.0301 | 0.0284 | 0.94 | yes (max Δ 0.0e0) | AF rotate() timed; agreement checked against AF transform() with luma's matrix (AF rotate() rounds cos/sin to 3 decimals); luma: 1 launches, 0.026 ms GPU |
| rotate bilinear | 1080p, 0.3 rad, crop | 0.0321 | 0.0519 | 1.61 | yes (max Δ 0.0e0) | AF rotate() timed; agreement checked against AF transform() with luma's matrix (AF rotate() rounds cos/sin to 3 decimals); luma: 1 launches, 0.028 ms GPU |
| transform affine bilinear | 1080p, inverse map | 0.0312 | 0.0624 | 2.00 | yes (max Δ 0.0e0) | luma: 1 launches, 0.026 ms GPU |
| transform perspective bilinear | 1080p, inverse map | 0.0378 | 0.1250 | 3.31 | yes (max Δ 1.6e-4) | luma: 1 launches, 0.031 ms GPU |
| rotate nearest | 4K, 0.3 rad, crop | 0.1011 | 0.0971 | 0.96 | yes (1 of 8294400 pixels at an edge differ) | AF rotate() timed; agreement checked against AF transform() with luma's matrix (AF rotate() rounds cos/sin to 3 decimals); luma: 1 launches, 0.096 ms GPU |
| rotate bilinear | 4K, 0.3 rad, crop | 0.1080 | 0.1924 | 1.78 | yes (1 of 8294400 pixels at an edge differ) | AF rotate() timed; agreement checked against AF transform() with luma's matrix (AF rotate() rounds cos/sin to 3 decimals); luma: 1 launches, 0.104 ms GPU |
| transform affine bilinear | 4K, inverse map | 0.0958 | 0.2143 | 2.24 | yes (max Δ 0.0e0) | luma: 1 launches, 0.092 ms GPU |
| transform perspective bilinear | 4K, inverse map | 0.1036 | 0.4337 | 4.19 | yes (max Δ 3.2e-4) | luma: 1 launches, 0.100 ms GPU |
| scale bilinear | 1080p, 0.7 x 1.3 | 0.0313 | 0.0604 | 1.93 | yes (max Δ 0.0e0) | luma: 1 launches, 0.027 ms GPU |
| fir | 1M samples, 8 taps | 0.0175 | 0.0230 | 1.31 | yes (max Δ 0.0e0); luma vs f64 ref max Δ 4.8e-7 | luma: 1 launches, 0.014 ms GPU |
| fir | 1M samples, 64 taps | 0.0230 | 0.0394 | 1.71 | yes (max Δ 0.0e0); luma vs f64 ref max Δ 4.8e-6 | luma: 1 launches, 0.020 ms GPU |
| fir | 1M samples, 128 taps | 0.0315 | 0.0657 | 2.08 | yes (max Δ 0.0e0); luma vs f64 ref max Δ 7.6e-6 | luma: 1 launches, 0.028 ms GPU |
| fir | 1M samples, 256 taps | 0.0476 | 0.1172 | 2.46 | yes (max Δ 2.5e-5); luma vs f64 ref max Δ 1.9e-5 | luma: 1 launches, 0.045 ms GPU |
| fir | 1M samples, 512 taps | 0.0367 | 0.1170 | 3.18 | yes (max Δ 1.4e-4); luma vs f64 ref max Δ 1.5e-4 | luma: 3 launches, 0.037 ms GPU |
| fir | 1M samples, 1024 taps | 0.0370 | 0.1168 | 3.15 | yes (max Δ 1.8e-4); luma vs f64 ref max Δ 1.8e-4 | luma: 3 launches, 0.037 ms GPU |
| fir | 1M samples, 8192 taps | 0.0623 | 0.1177 | 1.89 | yes (max Δ 5.7e-4) | luma: 6 launches, 0.065 ms GPU |
| fir | 16M samples, 8 taps | 0.1636 | 0.2238 | 1.37 | yes (max Δ 0.0e0) | luma: 1 launches, 0.160 ms GPU |
| fir | 16M samples, 64 taps | 0.2367 | 0.5005 | 2.11 | yes (max Δ 0.0e0) | luma: 1 launches, 0.233 ms GPU |
| fir | 16M samples, 128 taps | 0.3672 | 0.9260 | 2.52 | yes (max Δ 0.0e0) | luma: 1 launches, 0.362 ms GPU |
| fir | 16M samples, 256 taps | 0.6310 | 2.331 | 3.69 | yes (max Δ 2.9e-5) | luma: 1 launches, 0.625 ms GPU |
| fir | 16M samples, 1024 taps | 0.3853 | 2.330 | 6.05 | yes (max Δ 2.2e-4) | luma: 3 launches, 0.383 ms GPU |
| fir | 1024 x 4096 samples, 32 taps | 0.0514 | 0.0930 | 1.81 | yes (max Δ 0.0e0) | luma: 1 launches, 0.048 ms GPU |
| iir | 1M samples, biquad | 0.1207 | 248.3 | 2056.71 | luma vs f64 ref max Δ 2.2e-7; AF vs f64 ref max Δ 1.8e-7 | luma: 4 launches, 0.047 ms GPU |
| iir | 4M samples, biquad | 0.2332 | 984.5 | 4221.72 | luma vs f64 ref max Δ 2.2e-7; AF vs f64 ref max Δ 1.8e-7 | luma: 4 launches, 0.154 ms GPU |
| iir | 16M samples, biquad | 0.5719 | 3937.2 | 6884.73 | luma vs f64 ref max Δ 2.2e-7; AF vs f64 ref max Δ 2.0e-7 | luma: 4 launches, 0.509 ms GPU |
| iir | 1M samples, order 4 | 0.1016 | 246.5 | 2425.83 | luma vs f64 ref max Δ 5.5e-7; AF vs f64 ref max Δ 4.9e-7 | luma: 4 launches, 0.053 ms GPU |
| iir | 1024 x 4096 samples, biquad | 0.1659 | 3.691 | 22.25 | luma (first signals) vs f64 ref max Δ 1.6e-7; yes (max Δ 2.4e-7) | luma: 4 launches, 0.165 ms GPU |
| iir | 64 x 65536 samples, biquad | 0.1587 | 16.17 | 101.90 | luma (first signals) vs f64 ref max Δ 1.8e-7; yes (max Δ 2.4e-7) | luma: 4 launches, 0.152 ms GPU |
