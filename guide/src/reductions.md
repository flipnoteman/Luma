# Reductions and scans

A reduction turns an array into fewer numbers: a sum, a minimum, a norm. In Luma a reduction
returns a one-element array and the expression behind it is fused into the reduction's first
pass. Over an `f32` array, or over any lazy expression, nothing runs until the result is needed;
over an already computed `u32`, `f64` or `Df32` array the passes are recorded at once (still
without waiting). Nothing waits for the GPU until the host asks for the value.

## Lazy one-element results

```rust
use luma::*;

let x = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], 4)?;

let s = x.sum();                    // an Array<f32> of one element, nothing has run
assert_eq!(s.shape(), Shape::from(1));
assert_eq!(s.item()?, 10.0);        // `item` computes and reads it

assert_eq!(x.min().item()?, 1.0);
assert_eq!(x.max().item()?, 4.0);
assert_eq!(x.mean().item()?, 2.5);
assert!((x.norm().item()? - 30.0f32.sqrt()).abs() < 1e-6);

// Several reductions in one pass over the data (up to four).
let [lo, hi] = x.reduce([Reduction::Min, Reduction::Max]).to_array()?;
assert_eq!((lo, hi), (1.0, 4.0));
# Ok::<(), luma::LumaError>(())
```

| Method | Result |
| --- | --- |
| `sum()`, `min()`, `max()` | any element type; `u32` sums wrap around |
| `mean()`, `norm()` | `f32`; the norm is the Euclidean (L2) norm |
| `reduce([Reduction; N])` | `N` of `Sum`, `Min`, `Max`, `SumOfSquares` (one to four) in one pass, as an `N`-element array |
| `sum_where(&mask)` | the sum of the elements where a `u32` mask is non-zero, in one pass |
| `item()` | the value of a one-element array |
| `to_array::<N>()` | the values of an `N`-element array |

Because the result is an array, it can stay on the GPU. Reading it with `item()` is the only
step that waits. In a convergence loop, `step.max().item()?` once per iteration is the readback;
[Loops](loops.md) shows how to avoid one per step.

## Fusion

The expression a reduction reads is not stored. `(&x * &y).sum()` computes the product
inside the reduction's first pass, which reads `x` and `y` once; the product never exists in
memory.

```rust
use luma::*;

let ctx = Context::new()?;
let n = 1 << 20;
let x = Array::<f32>::full_on(&ctx, n, 2.0)?;
let y = Array::<f32>::full_on(&ctx, n, 3.0)?;

if ctx.profiling_supported() {
    let (dot, profile) = ctx.profile(|| (&x * &y).sum().item())?;
    assert_eq!(dot?, 6.0 * n as f32);
    // One fused pass over the data and one combine of its partial sums.
    assert!(profile.launches() <= 2, "{profile}");
}
# Ok::<(), luma::LumaError>(())
```

A large array takes two launches: a first pass in which every workgroup reduces a slice of the
expression to a partial result, and a second that combines the partials. An array small enough
for one workgroup takes one. Float sums combine in trees (see the `Array::sum` documentation), so the
error stays far below that of a sequential sum.

### Epilogues

For `f32` reductions, elementwise operations *with scalars* on the result extend the reduction
instead of creating an expression: the last pass applies them. `norm()` is `SumOfSquares` followed by a
square root in the last pass, and `mean()` is a sum followed by a division. You can chain up to
four such steps yourself without an extra launch:

```rust
use luma::*;

let x = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], 4)?;
let rms = ((&x * &x).sum() / 4.0).sqrt();   // the division and the root are in the last pass
assert!((rms.item()? - 7.5f32.sqrt()).abs() < 1e-6);
# Ok::<(), luma::LumaError>(())
```

### Using a reduction in an expression

A one-element array broadcasts against the array it came from, so normalising by the sum stays
on the GPU:

```rust
use luma::*;

let x = Array::from_slice(&[1.0f32, 1.0, 2.0, 4.0], 4)?;
let p = &x / &x.sum();                       // the sum is computed, then one kernel divides
assert_eq!(p.to_vec()?, vec![0.125, 0.125, 0.25, 0.5]);

// Softmax-style: subtract the maximum, exponentiate, normalise.
let z = (&x - &x.max()).exp();
let softmax = &z / &z.sum();
assert!((softmax.sum().item()? - 1.0).abs() < 1e-6);
# Ok::<(), luma::LumaError>(())
```

A reduction needs every element before any output exists, so the kernel that divides by the sum
runs after the kernel that computes it: the example above is a few launches, not one. The values
are the ones a host loop would give.

### Masks

`sum_where(&mask)` sums where a `u32` mask is non-zero (comparisons return such masks), the
count of a condition is the sum of the mask itself:

```rust
use luma::*;

let x = Array::from_slice(&[1.0f32, -2.0, 3.0, -4.0], 4)?;
let positive = x.gt(0.0);
assert_eq!(x.sum_where(&positive).item()?, 4.0);
assert_eq!(positive.sum().item()?, 2);
# Ok::<(), luma::LumaError>(())
```

### Vector elements

Arrays of `[f32; 2]` and `[f32; 4]` reduce per component, and these reductions return host
values at once (`Result<[f32; N]>`):

```rust
use luma::*;

let v = Array::from_slice(&[[1.0f32, 10.0], [2.0, 20.0], [3.0, 30.0]], 3)?;
assert_eq!(v.sum()?, [6.0, 60.0]);
assert_eq!(v.max()?, [3.0, 30.0]);
# Ok::<(), luma::LumaError>(())
```

## Along an axis

`sum_axis`, `min_axis` and `max_axis` reduce along one axis and leave extent 1 there. The result
broadcasts back against the array:

```rust
use luma::*;

// Two columns of three (the first dimension varies fastest).
let x = Array::from_slice(&[1.0f32, 1.0, 2.0, 2.0, 2.0, 4.0], [3, 2])?;
let totals = x.sum_axis(0);
assert_eq!(totals.shape(), Shape::from([1, 2]));
assert_eq!(totals.to_vec()?, vec![4.0, 8.0]);
assert_eq!((&x / &totals).to_vec()?, vec![0.25, 0.25, 0.5, 0.25, 0.25, 0.5]);
assert_eq!(x.max_axis(1).to_vec()?, vec![2.0, 2.0, 4.0]);
# Ok::<(), luma::LumaError>(())
```

## Scans

`cumsum(axis)` is the running sum along an axis. `scan(axis, op)` is the inclusive scan of any
*associative* elementwise operator:

```rust
use luma::*;

let x = Array::from_slice(&[3.0f32, 1.0, 4.0, 1.0, 5.0], 5)?;
assert_eq!(x.cumsum(0).to_vec()?, vec![3.0, 4.0, 8.0, 9.0, 14.0]);
assert_eq!(x.scan(0, |a, b| a.maximum(b)).to_vec()?, vec![3.0, 3.0, 4.0, 4.0, 5.0]);
# Ok::<(), luma::LumaError>(())
```

`op` is called once with placeholders and compiled into the scan kernels, so it must be an
elementwise expression of its two arguments and scalars (scalars it captures are uniforms:
changing them does not recompile); anything else poisons the result. Non-associative operators
give unspecified results, because the kernels regroup operands. Scans and axis reductions do not
fuse into the expression behind them: the arrays are evaluated first. They are available for
`f32` and `u32` arrays only (`f64` and `Df32` fail with `Unsupported`).

## Precision

Float reductions run in the element type of the array. For more accuracy, reduce in a wider
type:

- `sum_compensated()` and `norm_compensated()` accumulate in `Df32` (about 48 bits of
  mantissa against 24) and round to `f32` at the end. Use them for residual norms and regression tests.
- `to_df32().sum()` returns the `Df32` result itself, `to_f64().sum()` an `f64` on devices that
  have it.

```rust
use luma::*;

let x = Array::<f32>::full(1 << 20, 0.1)?;
let exact = f64::from(0.1f32) * (1 << 20) as f64;
assert!((f64::from(x.sum_compensated().item()?) - exact).abs() < 0.01);
assert!((x.to_df32().sum().item()?.to_f64() - exact).abs() < 1e-6);
# Ok::<(), luma::LumaError>(())
```

The [Precision](precision.md) chapter has the details of `Df32` and `f64`.
