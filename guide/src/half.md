# Half-precision floats: `f16` and `bf16`

Neural-network weights and activations, image data and anything that is bound by memory traffic
rather than by arithmetic can use two-byte floats. `Array<f16>` and `Array<bf16>` (the types of
the [`half`](https://docs.rs/half) crate, re-exported as `luma::f16` and `luma::bf16`) take two
bytes per element in device memory and on the host (`from_slice(&[f16])`, `to_vec()` and
`read_into` copy the bytes), and a kernel that reads one loads two bytes per element. Like the
[compact integers](compact.md), elements share `u32` words in buffers (two to a word, little
endian, byte-identical to the host slice).

| Type | Bytes | Mantissa bits | Largest finite | Smallest normal |
| --- | --- | --- | --- | --- |
| `f16` | 2 | 11 | 65504 | 6.1e-5 (subnormals down to 6.0e-8) |
| `bf16` | 2 | 8 | 3.4e38 | 1.2e-38 |

This is **storage plus `f32` compute**: every elementwise operation widens its operands to `f32`,
computes in `f32` and rounds the result to the element type, to nearest even. There is no native
half-precision arithmetic and no half-precision matrix unit in this release (the matrix units that
consume these types are later work).

```rust
use luma::*;

let x = Array::from_slice(&[f16::from_f32(1.5), f16::from_f32(-2.0), f16::from_f32(0.1)], 3)?;

// Every operation rounds its result to f16, fused or not: this is
// round((round(x * x) + 1)) per element, computed in one kernel that reads 2 bytes per element.
let y = &x * &x + f16::from_f32(1.0);
assert_eq!(y.to_vec()?, vec![f16::from_f32(3.25), f16::from_f32(5.0), f16::from_f32(1.01)]);

// Casts fuse into their neighbours.
let wide: Array<f32> = x.to_f32();
assert_eq!(wide.to_vec()?, vec![1.5, -2.0, f16::from_f32(0.1).to_f32()]);
let narrow: Array<bf16> = wide.to_bf16();
assert_eq!(narrow.to_vec()?[0], bf16::from_f32(1.5));
# Ok::<(), luma::LumaError>(())
```

## Rounding

Every conversion from a wider type (`f32`, `f64`, `i32`, `u32`, the other half type) and the
result of every arithmetic operation round to nearest, ties to even. A value too large for the type
becomes infinity (`f16` above 65519.99..., so `60000.0 * 2.0` is `inf`), subnormals are kept, and
every NaN becomes the canonical quiet NaN (`0x7e00` for `f16`, `0x7fc0` for `bf16`); the sign and
payload of a NaN are not preserved. All backends produce the same bits: the conversions are
written with integer arithmetic (`builtin::HALF_WGSL`) and `tests/all/half.rs` compares them with
the `half` crate for every `f16` and `bf16` pattern, for the halfway points between neighbours and
for a million random `f32`.

A fused chain gives the bits of the same operations evaluated one at a time: fusion rounds after
every operation. Operations that cannot produce a value the type does not hold (negation, `abs`,
`minimum`, `maximum`, `select`, `floor`, `ceil`, `trunc`, `round`, `sign`) skip the rounding.

## Operations

* Operators `+ - * /` (and the assigning forms) with arrays and with values of the type, `-x`,
  comparisons, `select`, `minimum`, `maximum`, `modulo`, `pow`, `is_nan`, `is_finite`, `is_inf`.
* `sqrt`, `exp`, `ln`, `abs`, `sin`, `cos`, `tan`, `tanh`, `sign`, `floor`, `ceil`, `trunc`,
  `round`, `log1p`, `expm1`, `asin`, `acos`, `atan`, `sinh`, `cosh`, `asinh`, `acosh`, `atanh`,
  `pow2`, `log2`, `log10`, `rsqrt`, `cbrt`, `erf`, `erfc`, `lgamma`, `tgamma`, `sigmoid`: the `f32`
  function, then rounded.
* [Views](views.md), `concat`, `filter`, `take`, `full`, `zeros`, writes through views, arrays of
  several buffers.
* Casts: `to_f32`, `to_f64` (where the device has 64-bit floats), `to_i32`, `to_u32` and the other
  half type from either; `to_f16` and `to_bf16` from `f32`, `f64`, `i32` and `u32`.

## Reductions accumulate in `f32`

`sum`, `prod`, `mean`, `norm`, `sum_axis`, `prod_axis`, `mean_axis`, `var_axis`, `std_axis`
widen the elements (the cast fuses into the first kernel), accumulate in `f32` and round **once**,
at the end. That is more accurate than rounding after every addition, as a half-precision
accumulator would: the sum of a hundred thousand ones is exactly 100000 (an `f16` accumulator stalls
at 2048, a `bf16` one at 256), but it is then rounded to the type (`100000` is `inf` in `f16`).
`min`, `max`, `min_axis`, `max_axis`, `argmax` and `argmin` are exact. Widen explicitly
(`x.to_f32().sum()`) to keep the sum in `f32`.

## Matrix products

`matmul`, `matmul_nt`, `matmul_tn`, `matmul_tt` and `dot` widen the operands where the kernel loads
them (an operand that is a lazy expression is fused as for `f32`), run the `f32` kernels (so the
context's TF32 setting applies, and the sums are `f32`) and round the result to the element type.
It saves memory and the bandwidth of loading the operands; it is not faster arithmetic.

## Not supported

`sort`, `sort_descending`, `sort_axis`, `argsort`, `argsort_descending`, `unique`, the set
operations, `cumsum` and other scans, `searchsorted`, `topk`, scatter-add and the other atomics,
the by-key operations, and the image and vision kernels are not defined for half-precision arrays.
The ones that have a method on `Array<f16>` / `Array<bf16>` return an array poisoned with
`LumaError::Unsupported` when called; the rest do not compile (the element type does
not implement their trait). Convert with `to_f32()` first: it fuses into the first kernel, so
`x.to_f32().sort()` reads two bytes per element.

## Performance

Memory-bound work moves half the bytes: see the `datasci/half` cases of `luma-bench` (a fused
elementwise chain stored to memory and a sum, in `f16` and in `f32`). Each half-precision load
and store runs the conversion (a few integer operations), and each rounded operation two more, so
a chain of transcendental functions is no faster than in `f32`.
