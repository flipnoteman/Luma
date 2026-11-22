### NVIDIA GeForce RTX 3090 - luma vs ArrayFire

Steady-state per-call time, median of the timed repetitions of an interleaved (ABBA) run; ratio = AF / luma (> 1: luma faster).

| case | size | luma ms | ArrayFire ms | ratio (AF/luma) | results agree? | note |
|---|---|---:|---:|---:|---|---|
| 13 conv2 3x3 | 1920x1080 | 0.0281 | 0.0419 | 1.49 | yes (max Δ 7.2e-7) | luma cross-correlation = AF convolve2 (spatial) with the flipped kernel; zero border, same size; luma: 1 launches, 0.026 ms GPU |
| 13 conv2 7x7 | 1920x1080 | 0.0380 | 0.0743 | 1.96 | yes (max Δ 2.4e-6) | luma cross-correlation = AF convolve2 (spatial) with the flipped kernel; zero border, same size; luma: 1 launches, 0.036 ms GPU |
| 13 conv2 15x15 | 1920x1080 | 0.0995 | 0.2392 | 2.40 | yes (max Δ 1.5e-5) | luma cross-correlation = AF convolve2 (spatial) with the flipped kernel; zero border, same size; luma: 1 launches, 0.096 ms GPU |
| 13 conv2_sep 5 | 1920x1080 | 0.0496 | 0.0617 | 1.24 | yes (max Δ 1.8e-7) | symmetric Gaussian taps along both axes; luma: 2 launches, 0.047 ms GPU |
| 13 conv2_sep 15 | 1920x1080 | 0.0548 | 0.0772 | 1.41 | yes (max Δ 2.4e-7) | symmetric Gaussian taps along both axes; luma: 2 launches, 0.053 ms GPU |
| 13 medfilt2 3x3 | 1920x1080 | 0.0280 | 0.0388 | 1.39 | yes (max Δ 0.0e0) | zero border; luma: 1 launches, 0.025 ms GPU |
| 13 medfilt2 5x5 | 1920x1080 | 0.0590 | 0.0901 | 1.53 | yes (max Δ 0.0e0) | zero border; luma: 1 launches, 0.055 ms GPU |
| 13 medfilt2 7x7 | 1920x1080 | 0.3974 | 7.749 | 19.50 | yes (max Δ 0.0e0) | zero border; luma: 1 launches, 0.394 ms GPU |
| 13 minfilt 3x3 | 1920x1080 | 0.0281 | 0.0398 | 1.42 | yes (max Δ 0.0e0) | cells outside the image ignored (AF's only mode); luma: 1 launches, 0.025 ms GPU |
| 13 maxfilt 3x3 | 1920x1080 | 0.0279 | 0.0398 | 1.42 | yes (max Δ 0.0e0) | cells outside the image ignored (AF's only mode); luma: 1 launches, 0.025 ms GPU |
| 13 minfilt 15x15 | 1920x1080 | 0.0541 | 0.2415 | 4.46 | yes (max Δ 0.0e0) | cells outside the image ignored (AF's only mode); luma: 2 launches, 0.052 ms GPU |
| 13 maxfilt 15x15 | 1920x1080 | 0.0541 | 0.2413 | 4.46 | yes (max Δ 0.0e0) | cells outside the image ignored (AF's only mode); luma: 2 launches, 0.052 ms GPU |
| 13 dilate 3x3 cross | 1920x1080 | 0.0285 | 0.0377 | 1.32 | yes (max Δ 0.0e0) | luma: 1 launches, 0.025 ms GPU |
| 13 erode 3x3 cross | 1920x1080 | 0.0285 | 0.0378 | 1.33 | yes (max Δ 0.0e0) | luma: 1 launches, 0.026 ms GPU |
| 13 sobel (dx, dy) | 1920x1080 | 0.0367 | 0.0410 | 1.12 | yes (max Δ 4.8e-7); yes (max Δ 4.8e-7) | both derivatives; interior compared (the borders differ by convention); luma: 1 launches, 0.033 ms GPU |
| 13 grad (dx, dy) | 1920x1080 | 0.0368 | 0.0382 | 1.04 | yes (max Δ 0.0e0); yes (max Δ 0.0e0) | luma: 1 launches, 0.035 ms GPU |
| 13 bilateral | 1920x1080 | 0.1130 | 0.1321 | 1.17 | yes (max Δ 4.2e-7) | spatial sigma 2, range sigma 0.1; luma: 1 launches, 0.111 ms GPU |
| 13 sat | 1920x1080 | 0.0603 | 0.0937 | 1.55 | yes (max Δ 1.9e-1) | f32 running sums; summation orders differ; luma: 3 launches, 0.060 ms GPU |
| 13 hist_equal | 1920x1080 | 0.0485 | 0.1612 | 3.32 | yes (max Δ 0.0e0) | same histogram on both sides; lookup by the bin of every pixel; luma: 4 launches, 0.035 ms GPU |
| 13 moments (M00 M01 M10 M11) | 1920x1080 | 0.0393 | 0.3662 | 9.33 | yes (max Δ 5.8e2) | both read the four sums back (ArrayFire's reductions end in a host read); luma: 9 launches, 0.107 ms GPU |
| 14 rgb2gray | 1920x1080 | 0.0465 | 0.0482 | 1.04 | yes (max Δ 0.0e0) | luma: 1 launches, 0.044 ms GPU |
| 14 rgb2hsv | 1920x1080 | 0.0662 | 0.0673 | 1.02 | yes (max Δ 1.2e-7) | luma: 1 launches, 0.063 ms GPU |
| 14 hsv2rgb | 1920x1080 | 0.0663 | 0.0672 | 1.01 | yes (max Δ 0.0e0) | luma: 1 launches, 0.063 ms GPU |
| 14 rgb2ycbcr 601 | 1920x1080 | 0.0665 | 0.0718 | 1.08 | yes (max Δ 3.1e-5) | luma: 1 launches, 0.062 ms GPU |
| 14 ycbcr2rgb 601 | 1920x1080 | 0.0666 | 0.0704 | 1.06 | yes (max Δ 3.6e-7) | luma: 1 launches, 0.063 ms GPU |
| 14 unwrap 3x3 | 1920x1080 | 0.1704 | 0.3256 | 1.91 | yes (max Δ 0.0e0) | column layout, pad 1; luma: 1 launches, 0.168 ms GPU |
| 14 wrap 3x3 | 1920x1080 | 0.1058 | 0.1304 | 1.23 | yes (max Δ 0.0e0) | sum of overlapping windows (luma gathers, AF scatters); luma: 1 launches, 0.103 ms GPU |
| 13 conv2 3x3 | 3840x2160 | 0.0878 | 0.1423 | 1.62 | yes (max Δ 7.2e-7) | luma cross-correlation = AF convolve2 (spatial) with the flipped kernel; zero border, same size; luma: 1 launches, 0.085 ms GPU |
| 13 conv2 7x7 | 3840x2160 | 0.1225 | 0.2668 | 2.18 | yes (max Δ 2.4e-6) | luma cross-correlation = AF convolve2 (spatial) with the flipped kernel; zero border, same size; luma: 1 launches, 0.120 ms GPU |
| 13 conv2 15x15 | 3840x2160 | 0.3682 | 0.9241 | 2.51 | yes (max Δ 1.7e-5) | luma cross-correlation = AF convolve2 (spatial) with the flipped kernel; zero border, same size; luma: 1 launches, 0.362 ms GPU |
| 13 conv2_sep 5 | 3840x2160 | 0.1697 | 0.2096 | 1.23 | yes (max Δ 1.8e-7) | symmetric Gaussian taps along both axes; luma: 2 launches, 0.168 ms GPU |
| 13 conv2_sep 15 | 3840x2160 | 0.1828 | 0.2696 | 1.47 | yes (max Δ 2.4e-7) | symmetric Gaussian taps along both axes; luma: 2 launches, 0.181 ms GPU |
| 13 medfilt2 3x3 | 3840x2160 | 0.0877 | 0.1374 | 1.57 | yes (max Δ 0.0e0) | zero border; luma: 1 launches, 0.085 ms GPU |
| 13 medfilt2 5x5 | 3840x2160 | 0.2072 | 0.3386 | 1.63 | yes (max Δ 0.0e0) | zero border; luma: 1 launches, 0.205 ms GPU |
| 13 medfilt2 7x7 | 3840x2160 | 1.549 | 30.37 | 19.61 | yes (max Δ 0.0e0) | zero border; luma: 1 launches, 1.545 ms GPU |
| 13 minfilt 3x3 | 3840x2160 | 0.0877 | 0.1302 | 1.48 | yes (max Δ 0.0e0) | cells outside the image ignored (AF's only mode); luma: 1 launches, 0.085 ms GPU |
| 13 maxfilt 3x3 | 3840x2160 | 0.0878 | 0.1302 | 1.48 | yes (max Δ 0.0e0) | cells outside the image ignored (AF's only mode); luma: 1 launches, 0.085 ms GPU |
| 13 minfilt 15x15 | 3840x2160 | 0.1759 | 0.9226 | 5.25 | yes (max Δ 0.0e0) | cells outside the image ignored (AF's only mode); luma: 2 launches, 0.174 ms GPU |
| 13 maxfilt 15x15 | 3840x2160 | 0.1753 | 0.9152 | 5.22 | yes (max Δ 0.0e0) | cells outside the image ignored (AF's only mode); luma: 2 launches, 0.174 ms GPU |
| 13 dilate 3x3 cross | 3840x2160 | 0.0882 | 0.1280 | 1.45 | yes (max Δ 0.0e0) | luma: 1 launches, 0.085 ms GPU |
| 13 erode 3x3 cross | 3840x2160 | 0.0882 | 0.1281 | 1.45 | yes (max Δ 0.0e0) | luma: 1 launches, 0.085 ms GPU |
| 13 sobel (dx, dy) | 3840x2160 | 0.1274 | 0.1421 | 1.12 | yes (max Δ 4.8e-7); yes (max Δ 4.8e-7) | both derivatives; interior compared (the borders differ by convention); luma: 1 launches, 0.125 ms GPU |
| 13 grad (dx, dy) | 3840x2160 | 0.1274 | 0.1331 | 1.04 | yes (max Δ 0.0e0); yes (max Δ 0.0e0) | luma: 1 launches, 0.123 ms GPU |
| 13 bilateral | 3840x2160 | 0.4313 | 0.5043 | 1.17 | yes (max Δ 4.8e-7) | spatial sigma 2, range sigma 0.1; luma: 1 launches, 0.429 ms GPU |
| 13 sat | 3840x2160 | 0.1921 | 0.2926 | 1.52 | yes (max Δ 1.0e0) | f32 running sums; summation orders differ; luma: 3 launches, 0.193 ms GPU |
| 13 hist_equal | 3840x2160 | 0.1089 | 0.5200 | 4.78 | yes (max Δ 3.1e-5) | same histogram on both sides; lookup by the bin of every pixel; luma: 4 launches, 0.096 ms GPU |
| 13 moments (M00 M01 M10 M11) | 3840x2160 | 0.0698 | 1.280 | 18.33 | yes (max Δ 1.1e7) | both read the four sums back (ArrayFire's reductions end in a host read); luma: 9 launches, 0.201 ms GPU |
| 14 rgb2gray | 3840x2160 | 0.1625 | 0.1644 | 1.01 | yes (max Δ 0.0e0) | luma: 1 launches, 0.160 ms GPU |
| 14 rgb2hsv | 3840x2160 | 0.2467 | 0.2473 | 1.00 | yes (max Δ 1.2e-7) | luma: 1 launches, 0.244 ms GPU |
| 14 hsv2rgb | 3840x2160 | 0.2469 | 0.2471 | 1.00 | yes (max Δ 0.0e0) | luma: 1 launches, 0.244 ms GPU |
| 14 rgb2ycbcr 601 | 3840x2160 | 0.2472 | 0.2515 | 1.02 | yes (max Δ 3.4e-5) | luma: 1 launches, 0.245 ms GPU |
| 14 ycbcr2rgb 601 | 3840x2160 | 0.2470 | 0.2506 | 1.01 | yes (max Δ 3.6e-7) | luma: 1 launches, 0.243 ms GPU |
| 14 unwrap 3x3 | 3840x2160 | 0.6654 | 1.293 | 1.94 | yes (max Δ 0.0e0) | column layout, pad 1; luma: 1 launches, 0.664 ms GPU |
| 14 wrap 3x3 | 3840x2160 | 0.3928 | 0.5119 | 1.30 | yes (max Δ 0.0e0) | sum of overlapping windows (luma gathers, AF scatters); luma: 1 launches, 0.401 ms GPU |
| 13 gaussian_kernel 5x5 |  | 0.0045 | 0.0328 | 7.28 | yes (max Δ 3.7e-9) | default sigma; the result stays on the device; luma: 0 launches, 0.000 ms GPU |
| 13 gaussian_kernel 15x15 |  | 0.0048 | 0.0327 | 6.81 | yes (max Δ 1.9e-9) | default sigma; the result stays on the device; luma: 0 launches, 0.000 ms GPU |
| 13 gaussian_kernel 31x31 |  | 0.0059 | 0.0330 | 5.61 | yes (max Δ 4.7e-10) | default sigma; the result stays on the device; luma: 0 launches, 0.000 ms GPU |
| 13 conv3 3x3x3 | 256^3 | 0.4449 | 0.6908 | 1.55 | yes (max Δ 1.2e-6) | flipped kernel for AF; zero border; luma: 1 launches, 0.442 ms GPU |
| 13 conv3 5x5x5 | 256^3 | 1.340 | 1.713 | 1.28 | yes (max Δ 9.1e-6) | flipped kernel for AF; zero border; luma: 1 launches, 1.337 ms GPU |
| 13 dilate3 3x3x3 (6-neighbourhood) | 256^3 | 0.4082 | 0.4956 | 1.21 | yes (max Δ 0.0e0) | luma: 1 launches, 0.406 ms GPU |
| 13 erode3 3x3x3 (6-neighbourhood) | 256^3 | 0.4056 | 0.4947 | 1.22 | yes (max Δ 0.0e0) | luma: 1 launches, 0.402 ms GPU |
| resize nearest | 1080p -> 4K | 0.0587 | 0.0708 | 1.21 | yes (max Δ 0.0e0) | luma: 1 launches, 0.056 ms GPU |
| resize bilinear | 1080p -> 4K | 0.0675 | 0.0856 | 1.27 | yes (max Δ 1.2e-7) | luma: 1 launches, 0.065 ms GPU |
| resize bicubic | 1080p -> 4K | 0.1184 | 0.2259 | 1.91 | yes (max Δ 3.6e-7) | luma: 1 launches, 0.116 ms GPU |
| resize nearest | 4K -> 1080p | 0.0384 | 0.0356 | 0.93 | yes (max Δ 0.0e0) | luma: 1 launches, 0.036 ms GPU |
| resize bilinear | 4K -> 1080p | 0.0565 | 0.0537 | 0.95 | yes (max Δ 0.0e0) | luma: 1 launches, 0.054 ms GPU |
| resize bicubic | 4K -> 1080p | 0.0649 | 0.0978 | 1.51 | yes (max Δ 0.0e0) | luma: 1 launches, 0.062 ms GPU |
| resize bilinear RGB | 1080p -> 4K | 0.1897 | 0.2494 | 1.31 | yes (max Δ 1.2e-7) | luma: 1 launches, 0.188 ms GPU |
| resize bilinear RGB | 4K -> 1080p | 0.1511 | 0.1494 | 0.99 | yes (max Δ 0.0e0) | luma: 1 launches, 0.148 ms GPU |
| rotate nearest | 1080p, 0.3 rad, crop | 0.0264 | 0.0281 | 1.07 | yes (max Δ 0.0e0) | AF rotate() timed; agreement checked against AF transform() with luma's matrix (AF rotate() rounds cos/sin to 3 decimals); luma: 1 launches, 0.024 ms GPU |
| rotate bilinear | 1080p, 0.3 rad, crop | 0.0306 | 0.0514 | 1.68 | yes (max Δ 0.0e0) | AF rotate() timed; agreement checked against AF transform() with luma's matrix (AF rotate() rounds cos/sin to 3 decimals); luma: 1 launches, 0.028 ms GPU |
| transform affine bilinear | 1080p, inverse map | 0.0280 | 0.0598 | 2.13 | yes (max Δ 0.0e0) | luma: 1 launches, 0.025 ms GPU |
| transform perspective bilinear | 1080p, inverse map | 0.0317 | 0.1150 | 3.62 | yes (max Δ 1.6e-4) | luma: 1 launches, 0.029 ms GPU |
| rotate nearest | 4K, 0.3 rad, crop | 0.0907 | 0.0966 | 1.07 | yes (1 of 8294400 pixels at an edge differ) | AF rotate() timed; agreement checked against AF transform() with luma's matrix (AF rotate() rounds cos/sin to 3 decimals); luma: 1 launches, 0.088 ms GPU |
| rotate bilinear | 4K, 0.3 rad, crop | 0.1049 | 0.1870 | 1.78 | yes (1 of 8294400 pixels at an edge differ) | AF rotate() timed; agreement checked against AF transform() with luma's matrix (AF rotate() rounds cos/sin to 3 decimals); luma: 1 launches, 0.104 ms GPU |
| transform affine bilinear | 4K, inverse map | 0.0936 | 0.2131 | 2.28 | yes (max Δ 0.0e0) | luma: 1 launches, 0.091 ms GPU |
| transform perspective bilinear | 4K, inverse map | 0.1056 | 0.4317 | 4.09 | yes (max Δ 3.2e-4) | luma: 1 launches, 0.102 ms GPU |
| scale bilinear | 1080p, 0.7 x 1.3 | 0.0305 | 0.0596 | 1.95 | yes (max Δ 0.0e0) | luma: 1 launches, 0.028 ms GPU |
