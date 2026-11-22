### NVIDIA GeForce RTX 3080 Laptop GPU - luma vs ArrayFire

Steady-state per-call time, median of the timed repetitions of an interleaved (ABBA) run; ratio = AF / luma (> 1: luma faster).

| case | size | luma ms | ArrayFire ms | ratio (AF/luma) | results agree? | note |
|---|---|---:|---:|---:|---|---|
| approx1 nearest | 1M samples, 1M positions | 0.0417 | 0.0408 | 0.98 | yes (max Δ 0.0e0) | increasing jittered positions; luma: 1 launches, 0.039 ms GPU |
| approx1 linear | 1M samples, 1M positions | 0.0423 | 0.0410 | 0.97 | yes (max Δ 0.0e0) | increasing jittered positions; luma: 1 launches, 0.039 ms GPU |
| approx1 cubic | 1M samples, 1M positions | 0.0435 | 0.0415 | 0.95 | yes (max Δ 5.4e-7) | increasing jittered positions; luma: 1 launches, 0.038 ms GPU |
| approx1 nearest | 4M samples, 4M positions | 0.1409 | 0.1402 | 0.99 | yes (max Δ 0.0e0) | increasing jittered positions; luma: 1 launches, 0.136 ms GPU |
| approx1 linear | 4M samples, 4M positions | 0.1423 | 0.1404 | 0.99 | yes (max Δ 0.0e0) | increasing jittered positions; luma: 1 launches, 0.138 ms GPU |
| approx1 cubic | 4M samples, 4M positions | 0.1460 | 0.1422 | 0.97 | yes (max Δ 5.1e-7) | increasing jittered positions; luma: 1 launches, 0.145 ms GPU |
| approx1 nearest | 16M samples, 16M positions | 0.5744 | 0.5743 | 1.00 | yes (max Δ 0.0e0) | increasing jittered positions; luma: 1 launches, 0.532 ms GPU |
| approx1 linear | 16M samples, 16M positions | 0.5792 | 0.5757 | 0.99 | yes (max Δ 0.0e0) | increasing jittered positions; luma: 1 launches, 0.538 ms GPU |
| approx1 cubic | 16M samples, 16M positions | 0.5928 | 0.5829 | 0.98 | yes (max Δ 5.4e-7) | increasing jittered positions; luma: 1 launches, 0.550 ms GPU |
| approx1 linear | 4M samples, 4M random positions | 0.8096 | 0.8062 | 1.00 | yes (max Δ 0.0e0) | random positions; luma: 1 launches, 0.768 ms GPU |
| approx1 linear, batched | 1024 x 4096 samples (4M positions) | 0.1492 | 0.1477 | 0.99 | yes (max Δ 0.0e0) | luma: 1 launches, 0.144 ms GPU |
| approx2 nearest | 1080p image, 1080p positions | 0.1016 | 0.0995 | 0.98 | yes (max Δ 0.0e0) | smooth warp; luma: 1 launches, 0.097 ms GPU |
| approx2 bilinear | 1080p image, 1080p positions | 0.1034 | 0.1001 | 0.97 | yes (max Δ 0.0e0) | smooth warp; luma: 1 launches, 0.099 ms GPU |
| approx2 bicubic | 1080p image, 1080p positions | 0.1448 | 0.1194 | 0.82 | yes (max Δ 4.8e-7) | smooth warp; luma: 1 launches, 0.137 ms GPU |
| approx2 nearest | 4K image, 4K positions | 0.3780 | 0.3739 | 0.99 | yes (max Δ 0.0e0) | smooth warp; luma: 1 launches, 0.374 ms GPU |
| approx2 bilinear | 4K image, 4K positions | 0.3864 | 0.3760 | 0.97 | yes (max Δ 0.0e0) | smooth warp; luma: 1 launches, 0.381 ms GPU |
| approx2 bicubic | 4K image, 4K positions | 0.5886 | 0.4512 | 0.77 | yes (max Δ 4.8e-7) | smooth warp; luma: 1 launches, 0.546 ms GPU |
| resize nearest | 1080p -> 4K | 0.1595 | 0.1530 | 0.96 | yes (max Δ 0.0e0) | luma: 1 launches, 0.155 ms GPU |
| resize bilinear | 1080p -> 4K | 0.2196 | 0.1996 | 0.91 | yes (max Δ 1.2e-7) | luma: 1 launches, 0.213 ms GPU |
| resize bicubic | 1080p -> 4K | 0.4064 | 0.5460 | 1.34 | yes (max Δ 3.6e-7) | luma: 1 launches, 0.401 ms GPU |
| resize nearest | 4K -> 1080p | 0.0794 | 0.0775 | 0.98 | yes (max Δ 0.0e0) | luma: 1 launches, 0.076 ms GPU |
| resize bilinear | 4K -> 1080p | 0.1223 | 0.1208 | 0.99 | yes (max Δ 0.0e0) | luma: 1 launches, 0.118 ms GPU |
| resize bicubic | 4K -> 1080p | 0.1255 | 0.1786 | 1.42 | yes (max Δ 0.0e0) | luma: 1 launches, 0.121 ms GPU |
| resize bilinear RGB | 1080p -> 4K | 0.6563 | 0.6017 | 0.92 | yes (max Δ 1.2e-7) | luma: 1 launches, 0.613 ms GPU |
| resize bilinear RGB | 4K -> 1080p | 0.3478 | 0.3459 | 0.99 | yes (max Δ 0.0e0) | luma: 1 launches, 0.343 ms GPU |
| rotate nearest | 1080p, 0.3 rad, crop | 0.0596 | 0.0571 | 0.96 | yes (max Δ 0.0e0) | AF rotate() timed; agreement checked against AF transform() with luma's matrix (AF rotate() rounds cos/sin to 3 decimals); luma: 1 launches, 0.054 ms GPU |
| rotate bilinear | 1080p, 0.3 rad, crop | 0.0606 | 0.1040 | 1.71 | yes (max Δ 0.0e0) | AF rotate() timed; agreement checked against AF transform() with luma's matrix (AF rotate() rounds cos/sin to 3 decimals); luma: 1 launches, 0.058 ms GPU |
| transform affine bilinear | 1080p, inverse map | 0.0586 | 0.1280 | 2.18 | yes (max Δ 0.0e0) | luma: 1 launches, 0.054 ms GPU |
| transform perspective bilinear | 1080p, inverse map | 0.0644 | 0.1888 | 2.93 | yes (max Δ 1.6e-4) | luma: 1 launches, 0.059 ms GPU |
| rotate nearest | 4K, 0.3 rad, crop | 0.2384 | 0.2161 | 0.91 | yes (1 of 8294400 pixels at an edge differ) | AF rotate() timed; agreement checked against AF transform() with luma's matrix (AF rotate() rounds cos/sin to 3 decimals); luma: 1 launches, 0.234 ms GPU |
| rotate bilinear | 4K, 0.3 rad, crop | 0.2442 | 0.4318 | 1.77 | yes (1 of 8294400 pixels at an edge differ) | AF rotate() timed; agreement checked against AF transform() with luma's matrix (AF rotate() rounds cos/sin to 3 decimals); luma: 1 launches, 0.240 ms GPU |
| transform affine bilinear | 4K, inverse map | 0.2206 | 0.5318 | 2.41 | yes (max Δ 0.0e0) | luma: 1 launches, 0.212 ms GPU |
| transform perspective bilinear | 4K, inverse map | 0.2320 | 0.7486 | 3.23 | yes (max Δ 3.2e-4) | luma: 1 launches, 0.228 ms GPU |
| scale bilinear | 1080p, 0.7 x 1.3 | 0.0618 | 0.1221 | 1.97 | yes (max Δ 0.0e0) | luma: 1 launches, 0.057 ms GPU |
| fir | 1M samples, 8 taps | 0.0324 | 0.0400 | 1.23 | yes (max Δ 0.0e0); luma vs f64 ref max Δ 4.8e-7 | luma: 1 launches, 0.028 ms GPU |
| fir | 1M samples, 64 taps | 0.0420 | 0.0778 | 1.85 | yes (max Δ 0.0e0); luma vs f64 ref max Δ 4.8e-6 | luma: 1 launches, 0.040 ms GPU |
| fir | 1M samples, 128 taps | 0.0606 | 0.1344 | 2.22 | yes (max Δ 0.0e0); luma vs f64 ref max Δ 7.6e-6 | luma: 1 launches, 0.057 ms GPU |
| fir | 1M samples, 256 taps | 0.0948 | 0.2585 | 2.73 | yes (max Δ 2.5e-5); luma vs f64 ref max Δ 1.9e-5 | luma: 1 launches, 0.092 ms GPU |
| fir | 1M samples, 512 taps | 0.0714 | 0.2557 | 3.58 | yes (max Δ 1.4e-4); luma vs f64 ref max Δ 1.5e-4 | luma: 3 launches, 0.071 ms GPU |
| fir | 1M samples, 1024 taps | 0.0743 | 0.2559 | 3.44 | yes (max Δ 1.8e-4); luma vs f64 ref max Δ 1.8e-4 | luma: 3 launches, 0.077 ms GPU |
| fir | 1M samples, 8192 taps | 0.1242 | 0.2586 | 2.08 | yes (max Δ 5.7e-4) | luma: 6 launches, 0.130 ms GPU |
| fir | 16M samples, 8 taps | 0.3683 | 0.5402 | 1.47 | yes (max Δ 0.0e0) | luma: 1 launches, 0.364 ms GPU |
| fir | 16M samples, 64 taps | 0.5810 | 1.199 | 2.06 | yes (max Δ 0.0e0) | luma: 1 launches, 0.546 ms GPU |
| fir | 16M samples, 128 taps | 0.8697 | 2.156 | 2.48 | yes (max Δ 0.0e0) | luma: 1 launches, 0.827 ms GPU |
| fir | 16M samples, 256 taps | 1.504 | 5.090 | 3.38 | yes (max Δ 2.9e-5) | luma: 1 launches, 1.626 ms GPU |
| fir | 16M samples, 1024 taps | 0.8440 | 5.084 | 6.02 | yes (max Δ 2.2e-4) | luma: 3 launches, 0.861 ms GPU |
| fir | 1024 x 4096 samples, 32 taps | 0.1105 | 0.2093 | 1.89 | yes (max Δ 0.0e0) | luma: 1 launches, 0.109 ms GPU |
| iir | 1M samples, biquad | 0.0915 | 272.1 | 2974.86 | luma vs f64 ref max Δ 2.2e-7; AF vs f64 ref max Δ 1.8e-7 | luma: 4 launches, 0.079 ms GPU |
| iir | 4M samples, biquad | 0.3041 | 1089.1 | 3581.37 | luma vs f64 ref max Δ 2.2e-7; AF vs f64 ref max Δ 1.8e-7 | luma: 4 launches, 0.280 ms GPU |
| iir | 16M samples, biquad | 1.116 | 4358.1 | 3905.24 | luma vs f64 ref max Δ 2.2e-7; AF vs f64 ref max Δ 2.0e-7 | luma: 4 launches, 1.058 ms GPU |
| iir | 1M samples, order 4 | 0.1084 | 272.4 | 2512.88 | luma vs f64 ref max Δ 5.5e-7; AF vs f64 ref max Δ 4.9e-7 | luma: 4 launches, 0.080 ms GPU |
| iir | 1024 x 4096 samples, biquad | 0.3158 | 6.190 | 19.60 | luma (first signals) vs f64 ref max Δ 1.6e-7; yes (max Δ 2.4e-7) | luma: 4 launches, 0.309 ms GPU |
| iir | 64 x 65536 samples, biquad | 0.3066 | 18.99 | 61.96 | luma (first signals) vs f64 ref max Δ 1.8e-7; yes (max Δ 2.4e-7) | luma: 4 launches, 0.298 ms GPU |
