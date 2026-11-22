# Complex numbers

`Array<C32>` holds complex numbers of two `f32` (`C32 { re, im }`, 8 bytes per element, the layout of
`vec2<f32>`, of CUDA's `cufftComplex` and of C's `float _Complex`). It is the element type the FFTs
return. Like every elementwise type it is lazy and fused: a chain of complex operations is one kernel
that reads the inputs once and writes the result once, and it runs on every device.

```rust
use luma::*;

let z = Array::from_slice(&[C32::new(1.0, 2.0), C32::new(-3.0, 0.5), C32::new(0.0, -1.0)], 3)?;
let w = Array::from_slice(&[C32::new(0.5, 0.0), C32::new(2.0, 2.0), C32::new(1.0, 1.0)], 3)?;

// One kernel: products, a sum with a real number, the conjugate, a quotient and the modulus.
let r = ((&z * &w + &z).conj() / (&w + 1.0f32)).abs();
assert_eq!(r.to_vec()?.len(), 3);

let c = C32::new(2.0, -1.0);
assert_eq!((&z * c).to_vec()?[0], C32::new(1.0, 2.0) * c);
# Ok::<(), luma::LumaError>(())
```

## What exists

| Operation | Result |
| --- | --- |
| `+ - * /`, `-z`, and the compound forms `+= -= *= /=` | `Array<C32>`; the other side is an array, a `C32` or an `f32` (`z * 2.0`, `1.0 / &z`), a `bool` mask multiplies |
| `conj()`, `exp()`, `sqrt()` | `Array<C32>` |
| `real()`, `imag()`, `abs()`, `arg()`, `norm_sqr()` | `Array<f32>` |
| `Array::from_parts(&re, &im)`, `Array::from_polar(&r, &theta)`, `x.to_complex()` | `Array<C32>` from `f32` arrays (`to_complex` has zero imaginary parts) |
| `eq(rhs)`, `ne(rhs)`, `select(&cond, a, b)` | `Array<bool>` / `Array<C32>`; complex numbers are not ordered: `lt`, `minimum` and friends poison with `LumaError::Unsupported` |
| `sum()`, `mean()`, `sum_axis(axis)`, `mean_axis(axis)` | `Array<C32>`, lazy like the `f32` reductions |
| views (`slice`, `flip`, `transpose`, ...), `copy`, `zeros`, `full`, `from_slice`, `to_vec`, `item` | as for the other element types |

There is no sort, scan, `matmul`, `min` or `max` of complex numbers. Take `real()` / `imag()` / `abs()`
first, or treat the parts as two `f32` arrays; `Array::from_parts` puts them back.

The parts are what the reductions work on: `z.sum()` is `from_parts(&z.real().sum(), &z.imag().sum())`,
and `z.abs().sum()` computes the complex expression inside the reduction, so no array of complex
numbers is stored for it.

## Numerics

- **Products** are the plain `f32` formula `(a.re b.re - a.im b.im, a.re b.im + a.im b.re)`, which fuses to
  `fma` on the devices that have it; the error is relative to `|a| |b|`, not to a result that cancels.
- **Quotients** use Smith's algorithm: the components of the denominator scale each other, so nothing
  overflows or underflows where the result is representable (`(1e30 + 1e30 i) / (1e30 + 1e30 i)` is `1`).
  A zero denominator gives `(re / 0, im / 0)`: infinity or `NaN` per component as C99's Annex G has it, with no trap.
  (A device's fast `f32` division returns 0 for a divisor above `2^126`, about `8.5e37`: that is the limit of `|b|` here.)
- **`abs`** scales the components by a power of two when the larger is above `2^62` or below `2^-62` (no division), so `|3e30 + 4e30 i|` is `5e30` and `|3e-30 + 4e-30 i|` is
  `5e-30`; it is infinite when a component is infinite even if the other is `NaN`. `norm_sqr` has no scaling.
- **`arg`** is `atan2(im, re)` in `[-pi, pi]`; `exp` is `e^re (cos im + i sin im)`; `sqrt` is the principal root.
- **`conj`**, `-z`, `real` and `imag` are exact (a `NaN` stays one, `-0` flips its sign).

A real number `s` in an operation (`z + s`, `s - z`, `z * s`, `z / s`) works on the parts (`z * s` is
`(re * s, im * s)`), so `inf * 2.0` does not turn into `NaN` the way a product with `2 + 0i` does.
`s / z` is the complex quotient.

## Building arrays on the host

`C32::new(re, im)`, `C32::from_polar(r, theta)`, `C32::ZERO`, `C32::ONE`, `C32::I` and `C32::from(f32)`
make values; `C32` has `+ - * /`, `conj`, `abs` and `arg` on the host (for checking results), and
`Array::from_slice(&[C32], shape)` uploads them. `to_vec()` returns `Vec<C32>`.
