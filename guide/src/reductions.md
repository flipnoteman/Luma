# Reductions and scans

A reduction turns an array into fewer numbers: a sum, a minimum, a norm. In Luma a reduction
returns a one-element array and the expression behind it is fused into the reduction's first
pass. Over an `f32` array, or over any lazy expression, nothing runs until the result is needed;
over an already computed `u32`, `i32`, `f64` or `Df32` array the passes are recorded at once (still
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
| `prod()` / `product()` | the product; NaN if any element is NaN, `u32` / `i32` products wrap around |
| `reduce([Reduction; N])` | `N` of `Sum`, `Min`, `Max`, `SumOfSquares`, `Product` (one to four) in one pass, as an `N`-element array |
| `norm_p(Norm)` | `One`, `Two`, `Inf`, `P(p)`, `Frobenius` and the matrix `MatrixOne` / `MatrixInf` norms, `f32` (`f64` except `P`) |
| `sum_where(&cond)` | the sum of the elements where a `bool` (or non-zero `u32`) condition is true, in one pass |
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

A large array takes at most two launches: a first pass in which every workgroup reduces a slice of the
expression to a partial result, and a second that combines the partials (on CUDA the last workgroup of the first pass finishes the reduction in one launch). An array small enough
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

### Booleans

`sum_where(&cond)` sums where a condition (an `Array<bool>`, such as a comparison, or a `u32` array of
non-zero elements) is true; the `sum()` of a boolean array is the number of true elements, as a `u32`
array (also `count()`, as in NumPy). `any()` and `all()` are one-element `bool` arrays, and `bool`
arrays have `min` / `max` but no `mean`, `norm` or `reduce`. Along an axis, `count_axis`, `all_axis` and
`any_axis` do the same per line:

```rust
use luma::*;

let x = Array::from_slice(&[1.0f32, -2.0, 3.0, -4.0], 4)?;
let positive = x.gt(0.0);
assert_eq!(x.sum_where(&positive).item()?, 4.0);
assert_eq!(positive.count().item()?, 2);
assert_eq!(positive.sum().item()?, 2);
assert!(positive.any().item()?);

let table = Array::from_slice(&[1.0f32, -2.0, 3.0, 4.0, 5.0, -6.0], [2, 3])?;
let flags = table.gt(0.0);
assert_eq!(flags.count_axis(0).to_vec()?, vec![1, 2, 1]);
assert_eq!(flags.all_axis(1).to_vec()?, vec![true, false]);
assert_eq!(flags.any_axis(1).to_vec()?, vec![true, true]);
# Ok::<(), luma::LumaError>(())
```

### Products and norms

`prod()` (alias `product()`, ArrayFire's name) multiplies all elements, fused with the expression behind
it like a sum; `prod_axis(axis)` does it per line. `norm_p` takes a [`Norm`]: the absolute value or power
is computed inside the reduction, so every norm reads the data once.

```rust
use luma::*;

let x = Array::from_slice(&[3.0f32, -4.0, 2.0], 3)?;
assert_eq!(x.prod().item()?, -24.0);
assert_eq!(x.norm_p(Norm::One).item()?, 9.0);
assert_eq!(x.norm_p(Norm::Inf).item()?, 4.0);
assert!((x.norm_p(Norm::Two).item()? - 29.0f32.sqrt()).abs() < 1e-6);

// A matrix: axis 0 is the row index (as in ArrayFire), so MatrixOne is the largest column sum.
let m = Array::from_slice(&[1.0f32, -2.0, 3.0, 4.0], [2, 2])?;
assert_eq!(m.norm_p(Norm::MatrixOne).item()?, 7.0);
assert_eq!(m.norm_p(Norm::MatrixInf).item()?, 6.0);
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
broadcasts back against the array. They are available for `f32`, `u32` and `i32` arrays. A
pending operand is not evaluated first: its expression is computed inside the first pass of the
reduction, so `(&x * &y).sum_axis(1)` never stores the product. `reduce_axis`, `mean_axis`,
`var_axis` and `std_axis` are in [Matrices, statistics and
grouping](data.md#statistics-along-an-axis).

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
give unspecified results, because the kernels regroup operands. Scans do not fuse into the
expression behind them: the array is evaluated first (it is read twice; on CUDA the scans of
integers and `min` / `max` of long contiguous lines read it once, with the same results:
[`LUMA_LOOKBACK_SCAN`](environment.md)). Scans and axis
reductions are available for `f32`, `u32` and `i32` arrays (`f64` and `Df32` fail with
`Unsupported`); `bool` arrays have `sum_axis` and `cumsum` that count (`u32` results) and
`min_axis` / `max_axis`, but no `scan` or `reduce_axis`.

## Sorting, groups, unique values and quantiles

`sort`, `argsort`, `group_sum` and the other group reductions, `factorize` / `unique`, `histogram`
and `quantile` / `median` have their own chapter: [Matrices, statistics and
grouping](data.md).

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
