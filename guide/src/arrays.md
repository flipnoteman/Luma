# Arrays

An `Array<T>` is a block of elements on the device with a shape of up to four dimensions.

## Creating arrays

```rust
use luma::*;

// From host data. The first dimension varies fastest: a [3, 2] array holds two columns of three.
let a = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], [3, 2])?;
assert_eq!(a.shape(), Shape::from([3, 2]));
assert_eq!(a.len(), 6);

// Constants and generators. Generators are computed inside the kernel that reads them,
// they are not stored.
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
| `u32` | indices, masks, [bit operations](integers.md) (comparisons return `Array<u32>`, `0` or `1`) |
| `Df32` | double-float, about 48 bits of mantissa, runs on every device ([Precision](precision.md)) |
| `f64` | 64-bit float, only on devices with `SHADER_F64` |

`vec2` / `vec4` element types exist for custom kernels, and [records](records.md) group several
arrays behind one struct. Convert between float types with `to_f64`, `to_df32` and `to_f32`:

```rust
use luma::*;

let x = Array::from_slice(&[1.0f32, 2.0, 3.0], 3)?;
let d = x.to_df32();                  // exact, `lo` is zero
let back = (&d / &d).to_f32();
assert_eq!(back.to_vec()?, vec![1.0, 1.0, 1.0]);
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
assert!(c.to_vec().is_err());            // the error names the failed `+` and its source line
# Ok::<(), luma::LumaError>(())
```
