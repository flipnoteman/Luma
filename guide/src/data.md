# Matrices, statistics and grouping

The operations of this chapter are what table and matrix workloads are made of: products,
statistics along an axis, counting and grouping, sorting, unique values and quantiles. They
compose with everything in the earlier chapters: operands that are lazy expressions are computed
inside the kernels instead of being stored first, wherever the section says so.

Remember the shape convention: **axis 0 varies fastest**. A `[rows, columns]` array in the
NumPy sense is a luma array of shape `[columns, rows]`, and a table with one *column per
feature* is naturally `[samples, features]`, so that each feature is contiguous and reductions
over the samples are `axis` 0.

## Matrix products

`a.matmul(&b)` (or `matmul(&a, &b)`) multiplies two `f32` matrices. NumPy's `a @ b` with `a` of
shape `(M, K)` and `b` of shape `(K, N)` becomes, with the axes reversed:

- `a` has shape `[K, M]`,
- `b` has shape `[N, K]`,
- the result has shape `[N, M]`, element `(n, m)` being `Σ_k a[k, m] * b[n, k]`.

Written out as flat data, each matrix is listed row by row, which is exactly what row-major
NumPy data looks like:

```rust
use luma::*;

// [[1, 2, 3],      [[1, 0],        [[4,  5],
//  [4, 5, 6]]  @    [0, 1],    =    [10, 11]]
//                   [1, 1]]
let a = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], [3, 2])?; // K = 3, M = 2
let b = Array::from_slice(&[1.0f32, 0.0, 0.0, 1.0, 1.0, 1.0], [2, 3])?; // N = 2, K = 3
let c = a.matmul(&b);
assert_eq!(c.shape(), Shape::from([2, 2]));
assert_eq!(c.to_vec()?, vec![4.0, 5.0, 10.0, 11.0]);
# Ok::<(), luma::LumaError>(())
```

One-dimensional operands count as having extent 1 on axis 1; extents beyond axis 1 must be 1.
`transpose()` is a view that the kernel reads in place, so products with transposed operands cost
no copy:

```rust
use luma::*;

let a = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], [3, 2])?;
// `a` is [[1, 2, 3], [4, 5, 6]]; a @ a.T is 2 x 2, a.T @ a is 3 x 3.
let rows = a.matmul(&a.transpose());
assert_eq!(rows.to_vec()?, vec![14.0, 32.0, 32.0, 77.0]);
let cols = a.transpose().matmul(&a);
assert_eq!(cols.shape(), Shape::from([3, 3]));
assert_eq!(cols.to_vec()?, vec![17.0, 22.0, 27.0, 22.0, 29.0, 36.0, 27.0, 36.0, 45.0]);
# Ok::<(), luma::LumaError>(())
```

An operand that is a pending expression is not stored first: its expression is evaluated inside
the product's loads, once per use. A standardised Gram (correlation-style) matrix of a table with
one column per feature (`[samples, features]`) is one launch plus the statistics:

```rust
use luma::*;

let samples = 8;
let x = Array::from_slice(
    &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, // feature 0
      2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, // feature 1 = 2 * feature 0
      8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0], // feature 2 = 9 - feature 0
    [samples, 3],
)?;
let z = (&x - &x.mean_axis(0)) / x.std_axis(0); // pending: standardised columns
// features x samples times samples x features: the Gram matrix of the columns.
let corr = z.matmul(&z.transpose()) / (samples as f32);
let corr = corr.to_vec()?;
let expected = [1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0, -1.0, 1.0];
for (got, want) in corr.iter().zip(expected) {
    assert!((got - want).abs() < 1e-4);
}
# Ok::<(), luma::LumaError>(())
```

The product itself is *eager*: it is recorded when `matmul` is called and the result is an
evaluated array, so whatever consumes the result (here the division) runs as its own kernel.
Operands whose expressions read circular shifts are evaluated first; everything else fuses.

Errors poison the result like everywhere else: `ShapeMismatch` if `a.shape()[0] !=
b.shape()[1]` or an operand has more than two axes above 1, `ContextMismatch` for operands of
different contexts, and `Chunked` for operands or results that need more than one buffer.
Only `f32` has a product.

```rust
use luma::*;

let a = Array::<f32>::zeros([3, 2])?;
let b = Array::<f32>::zeros([3, 2])?; // needs shape [N, 3]
assert!(matches!(a.matmul(&b).to_vec(), Err(LumaError::Op { .. })));
# Ok::<(), luma::LumaError>(())
```

## Statistics along an axis

`reduce_axis(axis, [ops])` computes one to four reductions (`Reduction::Sum`, `Min`, `Max`,
`SumOfSquares`) along an axis in **one pass**: every element is read once for all of them. It
returns an array of results, one per op, each with extent 1 along `axis`. A pending operand is
computed inside the pass and never stored.

```rust
use luma::*;

// Two columns of four; reduce along axis 0.
let x = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 10.0, 20.0, 10.0, 20.0], [4, 2])?;
let [sum, sum_sq, max] = x.reduce_axis(0, [Reduction::Sum, Reduction::SumOfSquares, Reduction::Max]);
assert_eq!(sum.to_vec()?, vec![10.0, 60.0]);
assert_eq!(sum_sq.to_vec()?, vec![30.0, 1000.0]);
assert_eq!(max.to_vec()?, vec![4.0, 20.0]);
# Ok::<(), luma::LumaError>(())
```

The results are evaluated when the call returns. Another number of ops poisons the results.
`sum_axis`, `min_axis` and `max_axis` are the single-op cases. They (and `reduce_axis`) work on
`f32`, `u32` and `i32` (`SumOfSquares` of integers wraps); `bool` arrays have `sum_axis` (a
count, `u32`), `min_axis` and `max_axis` only.

For `f32`, `mean_axis`, `var_axis` and `std_axis` give the mean, the **population** variance
(divisor `n`, NumPy's `ddof = 0`) and its square root. They are lazy expressions on top of the
reductions, so they broadcast back against the array. `var_axis` is two-pass (the mean, then the
mean of squared deviations), which keeps it accurate for data far from zero:

```rust
use luma::*;

let x = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 10.0, 20.0, 10.0, 20.0], [4, 2])?;
assert_eq!(x.mean_axis(0).to_vec()?, vec![2.5, 15.0]);
assert_eq!(x.var_axis(0).to_vec()?, vec![1.25, 25.0]);
assert_eq!(x.std_axis(0).to_vec()?[1], 5.0);

// Far from zero the two-pass variance is still right.
let y = Array::from_slice(&[1.0e4f32 + 1.0, 1.0e4 + 2.0, 1.0e4 + 3.0, 1.0e4 + 4.0], 4)?;
assert!((y.var_axis(0).item()? - 1.25).abs() < 1e-3);
# Ok::<(), luma::LumaError>(())
```

## Accumulating scatters

[`put`](views.md) overwrites: of several equal indices one (unspecified) wins. `put_add`,
`put_min` and `put_max` *combine* instead: every `i` does `self[indices[i]] += values[i]` (or
`min`, `max`), duplicates included. They exist for `u32`, `i32` and `f32` arrays, work in place
on top of the contents the array already has, skip indices of `len` or more, and must be called
on the array itself, not on a view. `values` is an array of the shape of `indices` or one value;
both are computed inside the kernel, not stored first.

```rust
use luma::*;

let mut totals = Array::<u32>::full(4, 100)?;
let indices = Array::from_slice(&[0u32, 2, 2, 7, 2], 5)?; // 7 is out of range: skipped
totals.put_add(&indices, &Array::from_slice(&[1u32, 10, 20, 5, 30], 5)?)?;
assert_eq!(totals.to_vec()?, vec![101, 100, 160, 100]);

let mut lo = Array::<f32>::full(3, f32::INFINITY)?;
lo.put_min(&Array::from_slice(&[1u32, 1, 2], 3)?, &Array::from_slice(&[4.0f32, 2.0, 9.0], 3)?)?;
assert_eq!(lo.to_vec()?, vec![f32::INFINITY, 2.0, 9.0]);

let mut hi = Array::<i32>::full(2, i32::MIN)?;
hi.put_max(&Array::from_slice(&[0u32, 0, 1], 3)?, -3)?;
assert_eq!(hi.to_vec()?, vec![-3, -3]);
# Ok::<(), luma::LumaError>(())
```

Integer sums wrap and are exact. `f32` sums are accumulated atomically in no fixed order, so
their last bits can differ between runs (and from a host loop); `min` and `max` do not depend
on the order. The scatters are atomic, so a million values into a few cells is fine.

## Counting and histograms

`bincount(&indices, bins)` is the number of occurrences of each value in `0..bins`, as a `u32`
array; values of `bins` or more are not counted. `histogram(bins, lo, hi)` counts the elements
of an `f32` array in `bins` equal-width bins over `[lo, hi)`. Values below `lo` and NaN go to
bin 0, values of `hi` or more to the last bin. The bin computation runs inside the scatter
kernel, so a pending array is not stored. `bins == 0` or `hi <= lo` (or NaN bounds) poison the
result with `InvalidShape`.

```rust
use luma::*;

let codes = Array::from_slice(&[2u32, 0, 2, 2, 5, 9], 6)?;
assert_eq!(bincount(&codes, 4).to_vec()?, vec![1, 0, 3, 0]); // 5 and 9 are not counted

let x = Array::from_slice(&[-5.0f32, 0.5, 1.5, 1.6, 3.9, 4.0, 100.0, f32::NAN], 8)?;
assert_eq!(x.histogram(4, 0.0, 4.0).to_vec()?, vec![3, 2, 0, 3]);
assert!(x.histogram(0, 0.0, 4.0).to_vec().is_err());
# Ok::<(), luma::LumaError>(())
```

## Sorting

`sort`, `sort_descending`, `argsort`, `argsort_descending` and `sort_axis(axis)` order `u32`,
`i32` and `f32` arrays with a stable radix sort on the GPU (`f64` and `Df32` fail with
`Unsupported`; arrays over several buffers with `Chunked`). `sort` and `argsort` work on the flat
array and keep its shape. `argsort` returns the flat indices that sort the array, and since the
sort is stable, equal elements keep their input order. The results are recorded like any other
operation, and a pending input is not evaluated first: the keys are computed inside the first
pass of the sort.

```rust
use luma::*;

let x = Array::from_slice(&[3.0f32, -1.0, f32::NAN, 2.0, -1.0, 0.5], [3, 2])?;
assert_eq!(x.argsort().to_vec()?, vec![1, 4, 5, 3, 0, 2]);
// Sorts every line along axis 0 (the first dimension varies fastest).
let lines = x.sort_axis(0).to_vec()?;
assert_eq!(&lines[..2], &[-1.0, 3.0]);
assert!(lines[2].is_nan());
assert_eq!(&lines[3..], &[-1.0, 0.5, 2.0]);

let n = Array::from_slice(&[2u32, 7, 2, 9], 4)?;
assert_eq!(n.sort().to_vec()?, vec![2, 2, 7, 9]);
assert_eq!(n.sort_descending().to_vec()?, vec![9, 7, 2, 2]);
// Descending is the exact reverse by key: the tied 2s still keep their input order.
assert_eq!(n.argsort_descending().to_vec()?, vec![3, 1, 0, 2]);
# Ok::<(), luma::LumaError>(())
```

Floats sort in the total order of their bits: `-inf < .. < -0.0 < +0.0 < .. < +inf < NaN`, so
NaNs come last (first when descending) and `-0.0` sorts before `+0.0` (and `+0.0` before
`-0.0` when descending, by input order for ties). The sign of a NaN is not kept.
`sort_axis(axis)` sorts every line along `axis` independently, stably, and keeps the shape.

## Groups

`group_sum`, `group_min`, `group_max`, `group_mean` and `group_count` reduce the elements that
share a code, a groupby over dense codes `0..groups`: result element `g` combines the elements
`i` with `codes[i] == g`. `codes` is a `u32` array of the shape of the values (else
`ShapeMismatch`), and codes of `groups` or more are skipped. The values (`u32`, `i32` or `f32`;
`group_mean` is `f32` only) and the codes fuse into one scatter kernel; `f32` sums are not
bitwise reproducible (see above).

```rust
use luma::*;

let x = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0], 5)?;
let codes = Array::from_slice(&[0u32, 1, 0, 1, 3], 5)?; // group 2 is empty
assert_eq!(x.group_sum(&codes, 4).to_vec()?, vec![4.0, 6.0, 0.0, 5.0]);
assert_eq!(x.group_count(&codes, 4).to_vec()?, vec![2, 2, 0, 1]);
assert_eq!(x.group_min(&codes, 4).to_vec()?, vec![1.0, 2.0, f32::INFINITY, 5.0]);
assert_eq!(x.group_max(&codes, 4).to_vec()?, vec![3.0, 4.0, f32::NEG_INFINITY, 5.0]);
let mean = x.group_mean(&codes, 4).to_vec()?;
assert_eq!(&mean[..2], &[2.0, 3.0]);
assert!(mean[2].is_nan());
# Ok::<(), luma::LumaError>(())
```

Empty groups give 0 for sums and counts, NaN for means and the identity of the operation for
`group_min` / `group_max`: `+inf` / `-inf` for `f32`, the integer maximum / minimum for `u32`
and `i32` (`u32::MAX` / `0`, `i32::MAX` / `i32::MIN`).

Where the groups are labels rather than `0..groups`, `factorize` (next section) makes the codes.

## Unique values

`factorize()` returns the sorted distinct values of an array (flat, one dimension) and, shaped
like the array, the index of each element's value in them: `uniques.take(&codes)` is the array
again. `unique()` is the first result alone. Both evaluate the array and read back the number of
distinct values, which synchronises once. For floats all NaNs are one value (the last) and `-0.0`
and `+0.0` are one value, returned as `-0.0`.

```rust
use luma::*;

let labels = Array::from_slice(&[50u32, 30, 50, 90, 30], 5)?;
let (uniques, codes) = labels.factorize();
assert_eq!(uniques.to_vec()?, vec![30, 50, 90]);
assert_eq!(codes.to_vec()?, vec![1, 0, 1, 2, 0]);
assert_eq!(uniques.take(&codes).to_vec()?, vec![50, 30, 50, 90, 30]);
// The codes feed the group reductions.
let amount = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0], 5)?;
assert_eq!(amount.group_sum(&codes, uniques.len()).to_vec()?, vec![7.0, 4.0, 4.0]);

let f = Array::from_slice(&[0.0f32, -0.0, f32::NAN, 1.0, f32::NAN], 5)?;
let u = f.unique().to_vec()?;
assert_eq!(u.len(), 3);
assert!(u[0] == 0.0 && u[0].is_sign_negative());
assert_eq!(u[1], 1.0);
assert!(u[2].is_nan());
# Ok::<(), luma::LumaError>(())
```

## Quantiles

`quantile(q)` and `median()` of an `f32` array give a one-element array, interpolating linearly
between the sorted neighbours as NumPy's default does: with `h = q * (len - 1)` the value at
`floor(h)` plus `h - floor(h)` times the step to the value at `ceil(h)`. `quantile_axis(axis, q)`
and `median_axis(axis)` do it for every line along an axis (extent 1 there in the result). `q`
outside `[0, 1]` (NaN included) poisons the result with `InvalidShape`. The sort is evaluated,
the interpolation is lazy.

```rust
use luma::*;

let x = Array::from_slice(&[4.0f32, 1.0, 3.0, 2.0], 4)?;
assert_eq!(x.median().item()?, 2.5);
assert_eq!(x.quantile(0.0).item()?, 1.0);
assert_eq!(x.quantile(0.25).item()?, 1.75);
assert_eq!(x.quantile(1.0).item()?, 4.0);
assert!(x.quantile(1.5).item().is_err());

// Two columns of four: the median of each.
let t = Array::from_slice(&[4.0f32, 1.0, 3.0, 2.0, 10.0, 50.0, 30.0, 20.0], [4, 2])?;
assert_eq!(t.median_axis(0).to_vec()?, vec![2.5, 25.0]);
# Ok::<(), luma::LumaError>(())
```

NaNs sort last, so they only affect the high quantiles; NumPy returns NaN for every quantile of
an array with a NaN, this does not.

## Selecting elements

`x.filter(&mask)` returns the elements where a mask holds, in flat order, as a one-dimensional
array; the mask is an `Array<bool>` (a comparison, for example) or an `Array<u32>` whose non-zero
elements are selected, and has the shape of `x`. `mask.nonzero()` returns the flat indices of the
selected elements (axis 0 varies fastest, so element `[i0, i1]` of a `[d0, d1]` array has index
`i0 + d0 * i1`). Both are a count, a scan and a scatter on the GPU, about as fast as copying the
array.

The length of the result depends on the data, so both read the count back and synchronise once
(about 100 µs on an RTX 3090), and return `Result<Option<Array<_>>>`: `None` means nothing was
selected, because luma has no arrays without elements, and selecting nothing is a result rather
than an error.

```rust
use luma::*;

let x = Array::from_slice(&[5.0f32, -1.0, 7.0, -3.0, 2.0], 5)?;
let positive = x.gt(0.0);
assert_eq!(x.filter(&positive)?.unwrap().to_vec()?, vec![5.0, 7.0, 2.0]);
assert_eq!(positive.nonzero()?.unwrap().to_vec()?, vec![0, 2, 4]);
assert!(x.filter(&x.gt(10.0))?.is_none());
# Ok::<(), luma::LumaError>(())
```

To stay on the device, `filter_padded(&mask, fill)` and `nonzero_padded(fill)` return an array of
the full length (the selected elements first, then `fill`) and the number of selected elements as
a lazy one-element array.

```rust
use luma::*;

let x = Array::from_slice(&[5.0f32, -1.0, 7.0, -3.0, 2.0], 5)?;
let (kept, count) = x.filter_padded(&x.gt(0.0), 0.0);
assert_eq!(kept.to_vec()?, vec![5.0, 7.0, 2.0, 0.0, 0.0]);
assert_eq!(count.item()?, 3);
# Ok::<(), luma::LumaError>(())
```

`topk(k)` and `bottomk(k)` of a `u32`, `i32` or `f32` array return the `k` largest (smallest)
values and their flat indices, largest (smallest) first, ties in input order, NaNs counting as
the largest. They sort the whole array with `argsort`, so the cost does not depend on `k`.

```rust
use luma::*;

let x = Array::from_slice(&[3.0f32, 9.0, 1.0, 7.0], 4)?;
let (values, indices) = x.topk(2);
assert_eq!(values.to_vec()?, vec![9.0, 7.0]);
assert_eq!(indices.to_vec()?, vec![1, 3]);
# Ok::<(), luma::LumaError>(())
```
