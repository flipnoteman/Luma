# Compact integers: `u8`, `i8`, `u16`, `i16`

Sensor samples, image channels, quantized weights, symbols and trace data are 8 or 16 bits wide.
Luma stores them that way: an `Array<u8>` or `Array<i16>` takes one or two bytes per element in
device memory (and on the host: `from_slice(&[u8])` and `to_vec()` copy the bytes, nothing is
widened), and reading it costs one or two bytes of memory traffic per element, inside any kernel.
Everything the 32-bit integers do works on them: operators, comparisons, `select`, `minimum` /
`maximum`, [views](views.md), reductions, `bincount`, sorting, and [fusion](lazy.md).

| Type | Bytes | Sum accumulates in | Mean, variance |
| --- | --- | --- | --- |
| `u8`, `u16` | 1, 2 | `u32` | `f32` |
| `i8`, `i16` | 1, 2 | `i32` | `f32` |

```rust
use luma::*;

let samples: Vec<i16> = vec![-3, 12000, 32767, -32768, 5];
let x = Array::from_slice(&samples, 5)?;

// Casts fuse into the kernel that reads the array: this reads 2 bytes per element.
let scaled = x.to_f32() * 0.5;
assert_eq!(scaled.to_vec()?, vec![-1.5, 6000.0, 16383.5, -16384.0, 2.5]);

// Arithmetic wraps like Rust's wrapping_* operations.
assert_eq!((&x + 1i16).to_vec()?, vec![-2, 12001, -32768, -32767, 6]);

// Sums widen to 32 bits; minima and maxima keep the type; means are f32.
assert_eq!(x.sum().item()?, -3 + 12000 + 32767 - 32768 + 5);
assert_eq!(x.min().item()?, -32768i16);
assert_eq!(x.max().item()?, 32767i16);
# Ok::<(), luma::LumaError>(())
```

## Wraparound

Every operation computes in a 32-bit register and wraps its result to the type again, as Rust's
`wrapping_add`, `wrapping_sub`, `wrapping_mul`, `wrapping_neg`, `wrapping_abs`, `wrapping_div`,
`wrapping_shl` and `wrapping_shr` do: `250u8 + 10` is `4`, `i16::MIN - 1` is `i16::MAX`,
`-i8::MIN` is `i8::MIN`, `i8::MIN / -1` is `i8::MIN`. A shift amount is taken modulo the width
of the type (`1u8 << 9` is `2`); `>>` is logical for the unsigned types and arithmetic for the
signed ones. `!` is the bitwise complement. Division and remainder by zero are as undefined as
for `u32` and `i32`. There are no rotations (the width of a rotation is the type's), and no
operators with a shared host-side [`Scalar`](lazy.md): use plain values (`&x + 3u8`).

## Casts

A cast **from** a compact integer is by value: `-1i8` is `-1i32`, `0xFFFF_FFFFu32` as `u32`,
`-1.0f32`; `to_bool` is `!= 0`. A cast **to** one wraps an integer (it keeps the low bits: `300u32`
is `44u8`, `-1i32` is `255u8` and `65535u16`) and saturates a float (a NaN is 0), as Rust's `as`.
Every dtype has `to_u8`, `to_i8`, `to_u16` and `to_i16` (except itself), and the compact types have
`to_f32`, `to_f64` (where the device has 64-bit floats), `to_df32`, `to_u32`, `to_i32`, `to_bool`
and the other three. A cast is lazy and fuses like any elementwise operation: it is the
widening of `x.sum()` and `x.mean_axis(1)`, which therefore never store a widened copy.

## Reductions, sorting and counting

* `sum()`, `sum_axis()`, `cumsum()`: in `u32` / `i32`, so they wrap beyond 32 bits: sums of `u8`
  are exact for up to 16.8 million elements of 255, `u16` sums for 65 537 of 65 535. Widen first
  (`x.to_f32().sum()`, or `to_f64()`) for longer arrays.
* `min()`, `max()`, `min_axis()`, `max_axis()`: the same type. `argmax`, `argmin` (and `_axis`):
  indices as for any array.
* `mean()`, `mean_axis()`, `var_axis()`, `std_axis()`: `f32`, accumulated in `f32` like those of an
  `f32` array (every 8-bit and 16-bit value is exact in `f32`).
* `sort()`, `sort_descending()`, `argsort()`: a stable radix sort over the bits of the elements (one pass for 8-bit types, two for 16-bit ones, where `u32` takes four), the first pass reading the elements as stored; `sort_axis()` sorts the widened lines.
* `bincount(bins)` of `u8` and `u16`, and `histogram()` (256 or 65536 bins): the counts as a `u32` array,
  computed in one kernel that reads the elements at their own size.

```rust
use luma::*;

// Per-column statistics of a matrix of 16-bit traces: [samples, traces], columns over axis 1.
let (samples, traces) = (4usize, 3usize);
let data: Vec<i16> = (0..(samples * traces) as i16).map(|v| v * 100 - 500).collect();
let m = Array::from_slice(&data, [samples, traces])?;
let mean = m.mean_axis(1);
let var = m.var_axis(1);
assert_eq!(mean.to_vec()?, vec![-100.0, 0.0, 100.0, 200.0]);
for v in var.to_vec()? {
    assert!((v - 106_666.67).abs() < 1.0);
}

// A histogram of bytes, read as bytes.
let bytes = Array::from_slice(&[1u8, 1, 2, 255, 1, 0], 6)?;
let counts = bytes.histogram().to_vec()?;
assert_eq!((counts[0], counts[1], counts[2], counts[255]), (1, 3, 1, 1));
# Ok::<(), luma::LumaError>(())
```

## How they are stored

Elements share `u32` words in device buffers: element `i` of a `u8` array is byte `i % 4` of word
`i / 4`, and of a `u16` or `i16` array half `i % 2` of word `i / 2`, little endian: the bytes of
the array on the host. An array of `n` elements takes `ceil(n / 4)` (or `ceil(n / 2)`) words. A
custom [kernel](custom-kernels.md) sees an `array<u32>` of those words (as for `bool`, which has 32
to a word), and arrays split over several buffers split like `f32` arrays, so mixed
operations (a cast to `f32`) run piecewise.

The kernels read the words and extract the field (`(word >> shift) & mask`, or two shifts to
sign-extend), and write **whole words**: a fused elementwise kernel with a compact output gives each
invocation the 4 (for bytes) or 2 (for halves) neighbouring elements that make a word, so
no two invocations ever write one word and no atomics are needed. Kernels that cannot do that (a
`put` of scattered indices, a region assign, a `stack`) update their element with an atomic
`and` and `or` on its word, which is also race free but slower. The CUDA and CPU backends read the
same words; the layout is byte-identical to a plain array of `u8` or `i16`, so nothing differs
between backends.

Where it pays: memory-bound work. On an RTX 3090 (CUDA), a chain of `u8` arithmetic over 64M
elements takes 0.17 ms where the same chain on `u32` takes 0.65 ms, and the sum of 64M `i16` takes
0.17 ms against 0.32 ms for `f32`.
