# Arrays

An `Array<T>` is a block of elements on the device with a shape of up to four dimensions.

## Creating arrays

```rust
use luma::*;

// From host data. The first dimension varies fastest: a [3, 2] array holds two columns of three.
let a = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], [3, 2])?;
assert_eq!(a.shape(), Shape::from([3, 2]));
assert_eq!(a.len(), 6);

// Constants and generators. `zeros` and `full` are buffers filled at once; the generators
// (`linspace`, `range`, `uniform`) are computed inside the kernel that reads them, not stored.
let zeros = Array::<f32>::zeros([3, 2])?;
let twos = Array::<f32>::full([3, 2], 2.0)?;
let ramp = Array::linspace(0.0, 1.0, 5)?;
let rows = Array::<f32>::range([3, 2], 0)?;   // the index along dimension 0
let noise = Array::<f32>::uniform([1000], 42)?; // counter-based, same seed -> same numbers

assert_eq!((&a + &twos).to_vec()?, vec![3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
assert_eq!(ramp.to_vec()?, vec![0.0, 0.25, 0.5, 0.75, 1.0]);
assert_eq!(rows.to_vec()?, vec![0.0, 1.0, 2.0, 0.0, 1.0, 2.0]);
assert_eq!(zeros.len() + noise.len(), 1006);
# Ok::<(), luma::LumaError>(())
```

Every constructor has an `_on(&ctx, ..)` form that takes an explicit [`Context`](faq.md#choosing-a-device)
instead of the lazily opened default one. Arrays of different contexts cannot be mixed.

## Element types

| Type | Use |
| --- | --- |
| `f32` | the default float; every operation |
| `u32` | indices, counters, [bit operations](integers.md) |
| `i32` | signed integers, [casts](integers.md#signed-integers-and-casts) |
| `Mask` | the result of comparisons: a boolean per element ([masks](integers.md#comparisons-masks-and-select)) |
| `Df32` | double-float, about 48 bits of mantissa, runs on every device ([Precision](precision.md)) |
| `f64` | 64-bit float, only on devices with `SHADER_F64` |

`vec2` / `vec4` element types exist for custom kernels, and [records](records.md) group several
arrays behind one struct. The built-in operations exist for `f32`, `u32`, `i32`, `Df32` and `f64`
(the *scalar* types); what each type supports is listed in the chapters named in the table.

### Masks

A `Mask` is a boolean. Comparisons and `is_nan` / `is_finite` / `is_inf` produce `Array<Mask>`;
you can also upload one from `Mask::TRUE` / `Mask::FALSE` (or `Mask::from(bool)`). On the host a
mask reads back as `Mask`, and `bool::from` converts it. `item()` of a one-element mask (the
result of `any()` and `all()`) reads one value. Masks have no arithmetic and no generators
(`range`, `uniform`): they combine with `& | ^ !`, count with `count()`, steer `select`, and convert
with `to_u32()` / `to_f32()` ([Comparisons, masks and `select`](integers.md#comparisons-masks-and-select)).

```rust
use luma::*;

let m = Array::from_slice(&[Mask::TRUE, Mask::FALSE, Mask::from(true)], 3)?;
assert_eq!(m.to_vec()?, vec![Mask::TRUE, Mask::FALSE, Mask::TRUE]);
assert!(bool::from(m.to_vec()?[0]));
assert_eq!(m.count().item()?, 2);                       // a `u32`
assert_eq!(bool::from(m.any().item()?), true);
assert_eq!(bool::from(m.all().item()?), false);
assert_eq!(m.to_f32().to_vec()?, vec![1.0, 0.0, 1.0]);
# Ok::<(), luma::LumaError>(())
```

### Casts

Casts are methods named for the type they produce. They are lazy and fuse like any elementwise
operation.

| From | Methods | Rule |
| --- | --- | --- |
| `f32` | `to_u32`, `to_i32` | truncates toward zero, saturates, NaN becomes 0 ([values](integers.md#signed-integers-and-casts)) |
| `u32` | `to_f32`, `to_i32` | `to_f32` rounds above 2^24; `to_i32` reinterprets the bits |
| `i32` | `to_f32`, `to_u32` | `to_f32` rounds above 2^24; `to_u32` reinterprets the bits |
| `Mask` | `to_u32`, `to_f32` | 1 where true, 0 where false |
| `f32` | `to_f64`, `to_df32` | exact (`to_f64` needs `SHADER_F64`) |
| `f64`, `Df32` | `to_f32`, `to_df32` / `to_f64` | rounds to `f32`; `f64` to `Df32` is exact to 48 bits |

There is no cast to `Mask`: write a comparison (`x.ne(0u32)`). A cast pair not in the table does
not exist, so a type mix-up is a compile error rather than a silent conversion.

```rust
use luma::*;

let x = Array::from_slice(&[1.0f32, 2.0, 3.0], 3)?;
let d = x.to_df32();                  // exact, `lo` is zero
let back = (&d / &d).to_f32();
assert_eq!(back.to_vec()?, vec![1.0, 1.0, 1.0]);

let k = Array::from_slice(&[0u32, 1, 2], 3)?;
assert_eq!((&x * k.to_f32()).to_vec()?, vec![0.0, 2.0, 6.0]);
# Ok::<(), luma::LumaError>(())
```

## Elementwise operations

Operators `+ - * /` (and `%` on integers, unary `-` on floats and `i32`) take references or
values, arrays or scalars, and exist in compound forms (`+=`). Methods cover the rest; they are all
lazy and fuse:

| Type | Methods |
| --- | --- |
| `f32` | `sqrt`, `exp`, `ln`, `abs`, `sin`, `cos`, `tan`, `tanh`, `sign`, `floor`, `ceil`; `powf(exponent)`, `clamp(lo, hi)` (host scalars); `atan2(x)`, `hypot(other)`, `copysign(sign)` (array or scalar) |
| `f64` | `sqrt`, `abs`, `sign`, `floor`, `ceil` |
| `Df32` | `sqrt`, `abs`, `sign`, `floor`, `ceil` ([Precision](precision.md)) |
| `i32` | `abs`, `sign`, `shl`, `shr` ([Signed integers](integers.md#signed-integers-and-casts)) |
| `u32` | bit operations, `shl`, `shr`, `rotate_left`, `rotate_right` ([Integers](integers.md)) |
| every scalar type | `minimum(rhs)`, `maximum(rhs)`, `lt` `le` `gt` `ge` `eq` `ne` (masks), `select(&mask, a, b)` |

`atan2(x)` is `atan2(self, x)` with `self` the `y`. The right-hand side of `minimum`, the comparisons and the
`try_` methods is an `Operand`: an array or reference to one (equal shapes, or extents of 1 that
broadcast), a host value, or a [`Scalar`](lazy.md#scalars-that-change) handle.

```rust
use luma::*;

let y = Array::from_slice(&[1.0f32, -1.0, 0.0], 3)?;
assert_eq!(y.abs().sqrt().to_vec()?, vec![1.0, 1.0, 0.0]);
assert_eq!(y.clamp(-0.5, 0.5).to_vec()?, vec![0.5, -0.5, 0.0]);
assert_eq!(y.maximum(0.25).to_vec()?, vec![1.0, 0.25, 0.25]);
assert_eq!(y.hypot(0.0).to_vec()?, vec![1.0, 1.0, 0.0]);
assert_eq!(Array::from_slice(&[3.0f32], 1)?.copysign(&y.slice(0, 1..2)).to_vec()?, vec![-3.0]);
assert!(y.powf(2.0).to_vec()?[1].is_nan());                  // a negative base has no real power
assert!(y.sin().to_vec()?[0] > 0.84 && (y.atan2(1.0).to_vec()?[0] - std::f32::consts::FRAC_PI_4).abs() < 1e-5);
# Ok::<(), luma::LumaError>(())
```

Operators do not return `Result`s, so a mistake shows up late. `try_add`, `try_sub`, `try_mul` and
`try_div` check at once: they return `Err` for a shape or context mismatch instead of a poisoned
array ([Errors](#errors)).

```rust
use luma::*;

let a = Array::from_slice(&[1.0f32, 2.0, 3.0], 3)?;
let b = Array::from_slice(&[1.0f32, 2.0], 2)?;
assert_eq!(a.try_add(&a)?.to_vec()?, vec![2.0, 4.0, 6.0]);
assert!(a.try_add(&b).is_err());
# Ok::<(), luma::LumaError>(())
```

## Shapes and broadcasting

Operators need equal shapes, with two exceptions: a scalar broadcasts, and an extent of 1
broadcasts per axis.

```rust
use luma::*;

let col = Array::from_slice(&[1.0f32, 2.0, 3.0], [3, 1])?;
let row = Array::from_slice(&[10.0f32, 20.0], [1, 2])?;
let table = &col + &row;                      // [3, 1] + [1, 2] -> [3, 2]
assert_eq!(table.shape(), Shape::from([3, 2]));
assert_eq!(table.to_vec()?, vec![11.0, 12.0, 13.0, 21.0, 22.0, 23.0]);

// A reduction is a one-element array; it broadcasts against the array it came from
// and never leaves the GPU.
let x = Array::from_slice(&[1.0f32, 3.0], 2)?;
let normalised = &x / &x.sum();
assert_eq!(normalised.to_vec()?, vec![0.25, 0.75]);

// Explicit form:
assert_eq!(col.broadcast_to([3, 4]).shape(), Shape::from([3, 4]));
# Ok::<(), luma::LumaError>(())
```

`reshape` changes the shape without moving data (it shares the buffer, like `clone`); use
`copy()` for an independent array.

## Reading back

| Call | What it does |
| --- | --- |
| `to_vec()` | copies into a new `Vec`; submits the pending batch and waits |
| `read_into(&mut buf)` | copies into host memory you own (cheaper when repeated) |
| `item()` | the value of a one-element array |
| `to_array::<N>()` | the `N` values of an `N`-element array |
| `write(&data)` | overwrites the contents; operations recorded earlier still see the old ones |

```rust
use luma::*;

let mut x = Array::from_slice(&[1.0f32, 2.0, 3.0], 3)?;
let before = &x * 10.0;          // recorded against the current contents
x.write(&[7.0, 8.0, 9.0])?;      // does not disturb `before`
let mut out = [0.0f32; 3];
before.read_into(&mut out)?;
assert_eq!(out, [10.0, 20.0, 30.0]);
assert_eq!(x.to_vec()?, vec![7.0, 8.0, 9.0]);
# Ok::<(), luma::LumaError>(())
```

## Convolution

`x.conv1d(&kernel, stride, padding)` is a 1-D cross-correlation of an `f32` array with a kernel
(the kernel is not flipped), with zero padding on both sides; the output has
`(len - kernel_len + 2 * padding) / stride + 1` elements. `x.conv1d_fft(&kernel)` is the full
*linear convolution* (the kernel is flipped), `len + kernel_len - 1` elements, computed with FFTs:
use it for long kernels, where it is faster than the direct loop, and expect `f32` rounding error in
its result. Both evaluate their operands and fail (poisoned array) for a bad geometry such as a
kernel longer than the padded input or a zero stride.

```rust
use luma::*;

let x = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0], 5)?;
let k = Array::from_slice(&[1.0f32, 0.0, -1.0], 3)?;
assert_eq!(x.conv1d(&k, 1, 0).to_vec()?, vec![-2.0, -2.0, -2.0]);       // x[i] - x[i + 2]
assert_eq!(x.conv1d(&k, 2, 0).to_vec()?, vec![-2.0, -2.0]);
assert_eq!(x.conv1d(&k, 1, 1).to_vec()?, vec![-2.0, -2.0, -2.0, -2.0, 4.0]);

let full = x.conv1d_fft(&k).to_vec()?;                                  // k is flipped
let want = [1.0f32, 2.0, 2.0, 2.0, 2.0, -4.0, -5.0];
assert!(full.iter().zip(want).all(|(a, b)| (a - b).abs() < 1e-4));
# Ok::<(), luma::LumaError>(())
```

## Errors

Operators return arrays, not `Result`s, so expressions stay readable. A failing operation (a shape
mismatch, a slice out of range) returns a poisoned array; everything built from it is poisoned
with the same error, and the first call that reads it reports it:

```rust
use luma::*;

let a = Array::from_slice(&[1.0f32, 2.0, 3.0], 3)?;
let b = Array::from_slice(&[1.0f32, 2.0], 2)?;
let c = (&a + &b) * 2.0;                 // no panic here
assert!(c.check().is_err());             // `check` asks without waiting for the GPU
assert!(c.to_vec().is_err());            // the error says `add` failed, with its source line
# Ok::<(), luma::LumaError>(())
```
