### NVIDIA GeForce RTX 3080 Laptop GPU - luma vs ArrayFire

Steady-state per-call time, median of the timed repetitions of an interleaved (ABBA) run; ratio = AF / luma (> 1: luma faster).

| case | size | luma ms | ArrayFire ms | ratio (AF/luma) | results agree? | note |
|---|---|---:|---:|---:|---|---|
| 16 upload s16 (host to device) | 1024 x 32768 (32M elements) | 6.909 | 5.552 | 0.80 | yes (identical) | allocation, copy and evaluation; the host slice is the same; luma: 0 launches, 0.000 ms GPU |
| 17 sum of s16 (read as 2 bytes, 32-bit sum) | 1024 x 32768 (32M elements) | 0.2039 | 0.2035 | 1.00 | yes (sum -86234637) | luma: Array<i16>::sum (i32); ArrayFire: sum_all of the s16 array; luma: 1 launches, 0.193 ms GPU |
| 18 cast s16 to f32 + sum | 1024 x 32768 (32M elements) | 0.2032 | 0.9457 | 4.65 | yes (sum -86234637) | luma: the cast is fused into the reduction; ArrayFire: cast (a 4-byte array) then sum_all; luma: 1 launches, 0.193 ms GPU |
| 19 per-column mean + var of s16 | 1024 x 32768 (32M elements) | 0.3210 | 2.373 | 7.39 | yes (max Δ 3.1e-5); yes (max Δ 1.2e0); luma mean vs f64 ref max Δ 0.0e0; luma var vs f64 ref max Δ 1.2e0 | population variance over the traces (axis 1); both results evaluated together; luma: 5 launches, 0.484 ms GPU |
| 20 per-column mean + var of f32 (same data) | 1024 x 32768 (32M elements) | 0.4762 | 1.927 | 4.05 | same data as 19 | the 4-byte baseline of 19; luma: 5 launches, 0.841 ms GPU |
| 21 histogram of u8 (256 bins) | 38M bytes | 0.1965 | 0.4430 | 2.25 | yes (exact) | luma: Array<u8>::histogram; ArrayFire: histogram of the u8 array over [0, 256); luma: 1 launches, 0.193 ms GPU |
| 22 histogram of u16 (16384 bins) | 38M values | 1.308 | 1.293 | 0.99 | yes (exact) | luma: Array<u16>::bincount; ArrayFire: histogram of the u16 array over [0, 16384); luma: 1 launches, 1.263 ms GPU |

