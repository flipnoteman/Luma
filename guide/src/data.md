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

`matmul_nt`, `matmul_tn` and `matmul_tt` (and the functions of the same names) are the products
with the right, the left or both operands transposed, written for the row-major matrices of NumPy:
`a.matmul_nt(&b)` is `A Bᵀ`, `a.matmul_tn(&b)` is `Aᵀ B` and `a.matmul_tt(&b)` is `Aᵀ Bᵀ`. The
shapes are those of the stored matrices (`matmul_nt`: `a` is `[K, M]`, `b` is `[K, N]`), the result
is `[N, M]`, and the backend's own gemm kernels read a stored operand in place, with no transposed
copy; an operand that is an expression is evaluated first. The precision is that of `matmul`.
`a.dot(&b)` is the dot product of two arrays of one shape as a lazy one-element array, `(a * b).sum()`
in one fused reduction:

```rust
use luma::*;

let a = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], [3, 2])?; // [[1, 2, 3], [4, 5, 6]]
assert_eq!(a.matmul_nt(&a).to_vec()?, vec![14.0, 32.0, 32.0, 77.0]); // A Aᵀ
assert_eq!(a.matmul_tn(&a).to_vec()?, vec![17.0, 22.0, 27.0, 22.0, 29.0, 36.0, 27.0, 36.0, 45.0]); // Aᵀ A
assert_eq!(a.dot(&a).item()?, 91.0);
# Ok::<(), luma::LumaError>(())
```

`a.matmul_batched(&b)` multiplies a batch of matrices in one launch (`matmul` in a loop is one launch
per product). The batch is axis 2 of both operands: `a` is `[K, M, B]`, `b` is `[N, K, B]` and the
result `[N, M, B]`. An operand without the batch (`[N, K]`) is shared by every product, as the
weights of a layer are:

```rust
use luma::*;

// Two products of [[1, 2], [3, 4]] @ [[1, 0], [0, 1]] and [[5, 6], [7, 8]] @ [[1, 0], [0, 1]].
let a = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], [2, 2, 2])?;
let identity = Array::from_slice(&[1.0f32, 0.0, 0.0, 1.0], [2, 2])?; // shared by both
let c = a.matmul_batched(&identity);
assert_eq!(c.shape(), Shape::from([2, 2, 2]));
assert_eq!(c.to_vec()?, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
# Ok::<(), luma::LumaError>(())
```

### Tensor cores (TF32)

On a CUDA device with tensor cores (compute capability 8.0 and up: Ampere, Ada, Hopper) `matmul`
runs on them by default, in TF32: the operands are rounded to TF32, which keeps the 8-bit exponent
of `f32` but only 10 bits of mantissa, and the products are accumulated in `f32`. Each entry
differs from the `f32` product by up to 1.5e-3 of the sum of the absolute values of its terms
(about 2e-4 of the entries' RMS for operands of random sign), the precision of PyTorch's
`allow_tf32`; infinities and NaNs go through as in `f32`. On an RTX 3090 a 4096 x 4096 x 4096
product runs at 34 TFLOP/s instead of 20 (the card's `f32` peak is 35.6), a 1024 x 1024 x 1024 one
at 21 instead of 13, and the small and skinny products of a network (a 256-row batch through a
1024-wide layer, a 2048 x 512 x 128 weight gradient) 3-4 times faster.

Where the precision matters, ask for `f32`: `a.matmul_f32(&b)` is the full-precision product of
one call (a chain of fused multiply-adds in a fixed order per entry), and
`ContextOptions::numerics(Numerics::strict())` (`LUMA_NUMERICS=strict`) makes every `matmul` of a context
that product. `matmul_tf32` forces TF32 whatever the context says, and `matmul_batched` is always
`f32`. On CUDA the `f32` product has kernels of its own for the shapes above (an RTX 3090: 23
TFLOP/s at 8192 x 8192 x 8192, 18 at 1024 x 1024 x 1024); operands that are lazy expressions
are evaluated once first and go through the same kernels. Products whose `M`, `N` and `K` are not all multiples of 4 and at least 32, and every other
backend (wgpu, Metal, the CPU), compute the `f32` product, so the same program runs everywhere
and differs only by the rounding where tensor cores are used. Transposed operands are read in
place; other lazy operands are evaluated first, where the `f32` kernels fuse them.

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

### Contractions written as subscripts: `einsum`

`luma::einsum(spec, &[&a, &b, ..])` contracts any number of operands from a NumPy-style spec. Subscripts list the axes of an
operand **slowest first**, the way a matrix is written: the `[K, M]` array above is `"mk"` (`M` rows of `K` elements), so the
product of the example is `einsum("mk,kn->mn", &[&a, &b])`, and its transpose is `"mk,kn->nm"`. Without `->` the output is the
labels that appear once, alphabetically. A label that is in no later operand and not in the output is summed; a label repeated
inside one operand takes the diagonal (`"ii->i"`, `"ii"` is `a.trace()`). `...`, a repeated output label and arrays of more than
four axes are errors, and so is a spec with an unknown label or the wrong number of operands (the result is poisoned).

```rust
use luma::*;

let a = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], [3, 2])?;
let b = Array::from_slice(&[1.0f32, 0.0, 0.0, 1.0, 1.0, 1.0], [2, 3])?;
assert_eq!(einsum("mk,kn->mn", &[&a, &b]).to_vec()?, a.matmul(&b).to_vec()?);
assert_eq!(einsum("mk,mk->", &[&a, &a]).to_vec()?, [91.0]); // the squared Frobenius norm
assert_eq!(einsum("mk->m", &[&a]).to_vec()?, [6.0, 15.0]);
assert_eq!(a.trace().shape(), Shape::from([1]));
# Ok::<(), luma::LumaError>(())
```

Operands are contracted pairwise from the left. Each pair runs as the first of three lowerings that fits: `matmul` or
`matmul_batched` (one contracted label, at most one free label per operand and two batch labels), a generic comprehension for
anything else (Hadamard and outer products, several contracted labels, three batch labels), and, if the generic lowering cannot run
the pair, a copy followed by one `matmul_batched`. No valid spec is refused, but only the first tier is as fast as `matmul`. The sum
of a pair runs over its contracted labels in the order they are written, the first label slowest. Under
`Numerics::reassociate` (`algebraic()` and `fast()`) the order of the pairs is searched instead (exhaustive up to four operands, a
subset search up to eight, greedy above) by flops plus bytes at the device's bandwidth; the result is remembered per spec and shapes
and differs from the left-to-right one by rounding only.

## Statistics along an axis

`reduce_axis(axis, [ops])` computes one to four reductions (`Reduction::Sum`, `Min`, `Max`,
`SumOfSquares`, `Product`) along an axis in **one pass**: every element is read once for all of them. It
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

`f32` results stay pending until one is used, then it is recorded together with every other axis reduction over the same lines that is pending, in one kernel (each result is the value it has alone: the same bits on the CPU backend, within an ulp or two on GPU backends); other types are evaluated when the call returns. Another number of ops poisons the results.
`sum_axis`, `min_axis` and `max_axis` are the single-op cases. They (and `reduce_axis`) work on
`f32`, `u32` and `i32` (`SumOfSquares` of integers wraps); `bool` arrays have `sum_axis` (a
count, `u32`), `min_axis` and `max_axis` only.

For `f32`, `mean_axis`, `var_axis` and `std_axis` give the mean, the **population** variance
(divisor `n`, NumPy's `ddof = 0`) and its square root. They are lazy expressions on top of the
reductions, so they broadcast back against the array. `var_axis` is one pass: the sum and the sum of
squares of the elements minus the first element of each line (subtracted inside the reduction) come
from one `reduce_axis`, and the variance is `(S2 - S1 * S1 / n) / n`, clamped at 0 (NaN is kept). The
shift keeps it accurate for data far from zero:

```rust
use luma::*;

let x = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 10.0, 20.0, 10.0, 20.0], [4, 2])?;
assert_eq!(x.mean_axis(0).to_vec()?, vec![2.5, 15.0]);
assert_eq!(x.var_axis(0).to_vec()?, vec![1.25, 25.0]);
assert_eq!(x.std_axis(0).to_vec()?[1], 5.0);

// Far from zero the shifted one-pass variance is still right.
let y = Array::from_slice(&[1.0e4f32 + 1.0, 1.0e4 + 2.0, 1.0e4 + 3.0, 1.0e4 + 4.0], 4)?;
assert!((y.var_axis(0).item()? - 1.25).abs() < 1e-3);
# Ok::<(), luma::LumaError>(())
```

### Variance with a divisor, weights, covariance

`var(bias)`, `std(bias)` and `meanvar(bias)` do the same for the whole array (`f32` and `f64`), and
`var_axis_bias(axis, bias)`, `std_axis_bias` and `meanvar_axis` along an axis. A [`Bias`] is ArrayFire's
`af_var_bias`: `Population` divides by `n`, `Sample` by `n - 1`. The mean and variance of a `meanvar`
come from one pass over the data (the sum and the sum of squares of the data minus its first element).
`mean_weighted(&w)` and `meanvar_weighted(&w, bias)` take weights of the array's shape (or extent 1
along the axes to broadcast), `mean_axis_weighted(axis, &w)` along an axis. `cov(&y, bias)` and
`corrcoef(&y)` pair two arrays of one shape element by element (the sums of the three expressions are
recorded in one kernel); `cov_axis` and `corrcoef_axis` work per line. Axis forms are `f32` only
(there are no axis reductions of `f64`).

```rust
use luma::*;

let x = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], 4)?;
let y = Array::from_slice(&[2.0f32, 4.0, 6.0, 8.0], 4)?;
assert_eq!(x.var(Bias::Population).item()?, 1.25);
assert!((x.var(Bias::Sample).item()? - 5.0 / 3.0).abs() < 1e-6);
let (mean, var) = x.meanvar(Bias::Population);
assert_eq!((mean.item()?, var.item()?), (2.5, 1.25));
let w = Array::from_slice(&[1.0f32, 1.0, 1.0, 5.0], 4)?;
assert_eq!(x.mean_weighted(&w).item()?, 3.25);
assert!((x.corrcoef(&y).item()? - 1.0).abs() < 1e-6);
assert!((x.cov(&y, Bias::Sample).item()? - 10.0 / 3.0).abs() < 1e-5);
# Ok::<(), luma::LumaError>(())
```

## Argmax and argmin

`argmax` and `argmin` give the **flat index** of the largest or smallest element as a
one-element `u32` array, recorded without waiting like `sum` and usable on the GPU at once. A
pending operand is computed inside the first pass, never stored. `argmax_axis(axis)` and
`argmin_axis(axis)` give, for every line along `axis`, the index *along the axis* (extent 1
there in the result). The first of equal elements wins, and an `f32` NaN is returned before every number, by
`argmin` as well as by `argmax` (the first NaN, as in NumPy). They work on `f32`, `u32` and `i32`
(`f64` and `Df32` arrays poison the result with `Unsupported`, like the other reductions along
an axis).

```rust
use luma::*;

let x = Array::from_slice(&[3.0f32, 9.0, 1.0, 9.0, 7.0, 2.0, 8.0, 0.5], [4, 2])?;
assert_eq!(x.argmax().item()?, 1); // the first of the two 9s
assert_eq!(x.argmin().item()?, 7);

// Along axis 0 (a column of four per result): the row of the extreme of each column.
assert_eq!(x.argmax_axis(0).to_vec()?, vec![1, 2]);
assert_eq!(x.argmin_axis(0).to_vec()?, vec![2, 3]);

// Along axis 1: which of the two columns is larger, for each row.
assert_eq!(x.argmax_axis(1).to_vec()?, vec![1, 0, 1, 0]);

// An expression is not stored: the index of the sample nearest to 7.5.
let nearest = (&x - 7.5).abs().argmin();
assert_eq!(nearest.item()?, 4);

// A NaN is found first, whichever extreme is asked for.
let y = Array::from_slice(&[1.0f32, f32::NAN, 5.0, f32::NAN], 4)?;
assert_eq!((y.argmax().item()?, y.argmin().item()?), (1, 1));
# Ok::<(), luma::LumaError>(())
```

A flat index `i` of a `[w, h]` array is the element at `(i % w, i / w)`. The passes are those of
`sum_axis` (a workgroup per chunk of a contiguous line, blocks that an invocation walks for a
strided one) carrying a (value, index) pair, so an `argmax_axis` reads the data once and costs
the time of `max_axis`. Reading the result of `argmax()` back with `item()` is a host
synchronization (about 100 µs); use it on the GPU (`x.take(&x.argmax())`) to avoid it.

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
`sort_axis(axis)` sorts every line along `axis` independently and keeps the shape. Lines of up to
2048 elements (256 on the CPU backend; any axis) are sorted in workgroup memory, in one pass over the data; longer lines
take the radix sort of the whole array followed by a regrouping by line (and, along an axis other
than 0, a tiled transpose back to the layout). `sort_axis` returns the NaNs as the positive quiet NaN,
like `sort`.

`sort_by_key(&values)` sorts the keys (`u32`, `i32` or `f32`, ordered as above) and moves the
elements of `values`, an array of the same shape and any element type, with them: it returns
`(keys, values)`, stably, in flat order. `sort_by_key_descending(&values)` is the exact reverse by
key, and `sort_by_key_axis(axis, &values)` and `sort_by_key_axis_descending(axis, &values)` sort every
line along an axis independently. Values of 4 bytes (`u32`, `i32`, `f32`) travel through the radix
passes with their keys, one pass reading and writing both; other types are gathered by the
permutation of an `argsort`.

```rust
use luma::*;

let keys = Array::from_slice(&[3u32, 1, 3, 2], 4)?;
let names = Array::from_slice(&[10.0f32, 11.0, 12.0, 13.0], 4)?;
let (k, v) = keys.sort_by_key(&names);
assert_eq!(k.to_vec()?, vec![1, 2, 3, 3]);
assert_eq!(v.to_vec()?, vec![11.0, 13.0, 10.0, 12.0]); // the two keys 3 keep their order
# Ok::<(), luma::LumaError>(())
```

On the CUDA backend `sort`, `argsort`, `sort_by_key` and what is built on them run as a single-sweep
radix sort (as CUB's onesweep): one kernel counts the digits of all four passes in one read of the
keys, and each pass is one kernel that finds where its tile's keys go by a decoupled look-back over
the tiles before it, so a sort is 6 launches instead of 16. Where the device does not order memory
across workgroups (wgpu, the CPU backend) the count, scan and scatter passes are unchanged. The
results are the same on every backend.

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

### Segments

When the groups are contiguous runs of the array (the rows of a CSR matrix given by its row
pointer, a sorted group-by given by the offsets of its groups), `sum_segments`, `min_segments` and
`max_segments` reduce segment `i`, the elements `offsets[i]..offsets[i + 1]`, to element `i` of
the result (`offsets.len() - 1` elements, `u32`, non-decreasing, the first 0 and the last the
length of the values). An empty segment gives the identity, as for the group reductions. They
need no codes and no atomics: the entries are split evenly over the workgroups whatever the
lengths of the segments, so a power-law matrix with one row of 100000 entries among rows of a
handful costs no more than a regular one, and the result is the same on every run. A pending
operand is computed inside the kernel, so a sparse matrix-vector product is one expression:

```rust
use luma::*;

// [[1, 0, 2], [0, 0, 0], [0, 3, 4]] in CSR.
let ptr = Array::from_slice(&[0u32, 2, 2, 4], 4)?;
let col = Array::from_slice(&[0u32, 2, 1, 2], 4)?;
let val = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], 4)?;
let x = Array::from_slice(&[1.0f32, 10.0, 100.0], 3)?;
let y = (&val * x.take(&col)).sum_segments(&ptr);
assert_eq!(y.to_vec()?, vec![201.0, 0.0, 430.0]);
assert_eq!(val.max_segments(&ptr).to_vec()?, vec![2.0, f32::NEG_INFINITY, 4.0]);
# Ok::<(), luma::LumaError>(())
```

### Runs of equal keys

`scan_by_key(&keys, axis, op)` is the inclusive scan of an associative operator (as `scan`) that
restarts at every run of equal consecutive keys along `axis`: the keys are `u32`, `i32` or `f32`, of
the shape of the values, and equal keys that are apart are different runs (NaN keys are runs of one
element). `scan_by_key_exclusive(&keys, axis, identity, op)` is the exclusive form (the first element
of every run is `identity`) and `cumsum_by_key(&keys, axis)` the running sum. The results are the
same on every run. `sum_by_key`, `min_by_key`, `max_by_key`, `product_by_key`, `count_by_key`,
`any_true_by_key` and `all_true_by_key` reduce every run of equal consecutive keys and return the
first key of each run with the result, both one-dimensional with one element per run (ArrayFire's
`*ByKey`); the number of runs depends on the data, so they read it back and synchronise once.

```rust
use luma::*;

let keys = Array::from_slice(&[1u32, 1, 2, 2, 2, 1], 6)?;
let x = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], 6)?;
assert_eq!(x.cumsum_by_key(&keys, 0).to_vec()?, vec![1.0, 3.0, 3.0, 7.0, 12.0, 6.0]);
let (first_keys, sums) = x.sum_by_key(&keys)?;
assert_eq!(first_keys.to_vec()?, vec![1, 2, 1]);
assert_eq!(sums.to_vec()?, vec![3.0, 12.0, 6.0]);
# Ok::<(), luma::LumaError>(())
```

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

### Sets

`x.set_unique(is_sorted)`, `a.set_union(&b, is_unique)` and `a.set_intersect(&b, is_unique)`
(ArrayFire's `setUnique`, `setUnion`, `setIntersect`) work on `u32`, `i32` and `f32` arrays taken in
flat order and return sorted, distinct elements as one-dimensional arrays (as for `factorize`, all
NaNs are one value, `-0.0` and `+0.0` one, written as `+0.0`). `is_sorted` and `is_unique` are
ArrayFire's hints: `is_sorted` says the array is already ascending, `is_unique` that both arrays are
ascending and without duplicates, which skips the sorts (a wrong claim gives an unspecified
result). `set_intersect` returns `None` when the arrays have nothing in common. The result size
depends on the data, so each of them synchronises once.

```rust
use luma::*;

let a = Array::from_slice(&[5u32, 1, 3, 3, 9], 5)?;
let b = Array::from_slice(&[3u32, 4, 5, 5], 4)?;
assert_eq!(a.set_unique(false)?.to_vec()?, vec![1, 3, 5, 9]);
assert_eq!(a.set_union(&b, false)?.to_vec()?, vec![1, 3, 4, 5, 9]);
assert_eq!(a.set_intersect(&b, false)?.unwrap().to_vec()?, vec![3, 5]);
# Ok::<(), luma::LumaError>(())
```

Union and intersection are a merge of the two sorted arrays: the unsorted forms sort both first, and
with `is_unique` they are the merge alone. On CUDA the merge (`src/setmerge.rs`) is two launches and
one 4-byte readback (a partition of the merged order into tiles, then one kernel that merges a tile,
counts what it keeps, finds its first output slot by a decoupled look-back and writes coalesced).
Measured for sorted distinct `u32` arrays of 3 181 963 elements each, against ArrayFire 3.10.1
(interleaved): on an RTX 3090 `set_union` takes 0.086 ms (was 0.175; ArrayFire 0.098) and
`set_intersect` 0.076 ms (was 0.149; ArrayFire 0.080), on an RTX 3080 Laptop 0.162 ms (was 0.312;
ArrayFire 0.177) and 0.127 ms (was 0.268; ArrayFire 0.137); the unsorted forms take 0.58 / 1.16 ms
(union) and 0.57 / 1.12 ms (intersection) on the two GPUs. `set_unique` is a sort, a mark of the
first of every run and a compaction. Other backends, and `LUMA_LOOKBACK_SCAN=off`, run the merge as
the three kernels (partition, count, write) and a scan, with the same results.

## Quantiles

`quantile(q)` and `median()` of an `f32` array give a one-element array, interpolating linearly
between the sorted neighbours as NumPy's default does: with `h = q * (len - 1)` the value at
`floor(h)` plus `h - floor(h)` times the step to the value at `ceil(h)`. `quantile_axis(axis, q)`
and `median_axis(axis)` do it for every line along an axis (extent 1 there in the result). `q`
outside `[0, 1]` (NaN included) poisons the result with `InvalidShape`. The sort is evaluated,
the interpolation is lazy.

Every call sorts, and the sort is nearly all of the cost, so ask for several quantiles in one
call: `quantiles(&[q0, q1, ..])` gives a one-dimensional array with one entry per `q`, and
`quantiles_axis(axis, &qs)` the shape of the array with `qs.len()` entries along `axis` (entry
`j` is `quantile_axis(axis, qs[j])`). A describe-style table of 25% / 50% / 75% costs one sort
instead of three. Nothing is cached between calls: a sorted copy kept for the next call would
cost as much memory as the array.

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

// Three quantiles of each column with one sort: [3, 2], the quantiles along axis 0.
let q = t.quantiles_axis(0, &[0.0, 0.5, 1.0]);
assert_eq!(q.to_vec()?, vec![1.0, 2.5, 4.0, 10.0, 25.0, 50.0]);
assert_eq!(x.quantiles(&[0.0, 0.25, 1.0]).to_vec()?, vec![1.0, 1.75, 4.0]);
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
the largest. They sort the whole array with `argsort`, so the cost does not depend on `k`. `k` of
0 or more than the length poisons both results with `InvalidShape`.

```rust
use luma::*;

let x = Array::from_slice(&[3.0f32, 9.0, 1.0, 7.0], 4)?;
let (values, indices) = x.topk(2);
assert_eq!(values.to_vec()?, vec![9.0, 7.0]);
assert_eq!(indices.to_vec()?, vec![1, 3]);
# Ok::<(), luma::LumaError>(())
```

## Searching sorted arrays

`sorted.searchsorted(&values, side)` is NumPy's `searchsorted`: element `i` of the result is the
position where `values[i]` would be inserted into `sorted` to keep it ordered, `Side::Left` before
the elements equal to it (the number of elements below it) and `Side::Right` after them (below or
equal). `sorted` is one-dimensional and ascending, as `sort()` makes it (floats: NaNs last, larger
than everything, equal to each other; `-0.0` equals `+0.0`); it works for `u32`, `i32` and `f32`.
The result has the shape of `values`.

`x.digitize(&edges, right)` is `np.digitize` for increasing edges: the number of the bin of each
element, `0` below the first edge and `edges.len()` above the last, with `right` choosing whether a
value on an edge belongs to the bin below (`edges[i - 1] < x <= edges[i]`) or above it
(`edges[i - 1] <= x < edges[i]`). Together with `bincount` it is a histogram with uneven bins.
`x.isin(&set)` is `np.isin`: a `bool` array of whether each element is in `set` (any shape; NaN is
in no set).

```rust
use luma::*;

let edges = Array::from_slice(&[0.0f32, 1.0, 2.5, 5.0], 4)?;
let x = Array::from_slice(&[-1.0f32, 0.0, 1.0, 2.5, 4.9, 9.0], 6)?;
assert_eq!(edges.searchsorted(&x, Side::Left).to_vec()?, vec![0, 0, 1, 2, 3, 4]);
assert_eq!(edges.searchsorted(&x, Side::Right).to_vec()?, vec![0, 1, 2, 3, 3, 4]);

// Bins: [.., 0), [0, 1), [1, 2.5), [2.5, 5), [5, ..)
let bins = x.digitize(&edges, false);
assert_eq!(bins.to_vec()?, vec![0, 1, 2, 3, 3, 4]);
assert_eq!(bincount(&bins, 5).to_vec()?, vec![1, 1, 1, 2, 1]);
assert_eq!(x.digitize(&edges, true).to_vec()?, vec![0, 0, 1, 2, 3, 4]);

let allowed = Array::from_slice(&[3u32, 5, 8], 3)?;
let ids = Array::from_slice(&[1u32, 3, 5, 7, 8], 5)?;
assert_eq!(ids.isin(&allowed).to_vec()?, vec![false, true, true, false, true]);
# Ok::<(), luma::LumaError>(())
```

The search is *lazy* and elementwise over the values: a lookup is one binary search per value,
`ceil(log2(n + 1))` steps of gathers from `sorted`, and when the values are a pending expression
(`(x * scale + offset)`) they are computed inside the kernel instead of being stored first.
`sorted` is read as a buffer (a pending one is evaluated first). Measured on an RTX 3090,
`2^24` values into `2^16` edges take 0.89 ms, 10% less than a hand-written WGSL binary search
(`cargo run --release --example bench_datasci -- --only searchsorted`). `isin` sorts the set when it
is called (eager, no host synchronisation; duplicates in the set do not matter) and then looks the
elements up like `searchsorted` does and compares the element found. A `sorted` that is not
one-dimensional poisons the result with `InvalidShape`.

## NaN-aware reductions

`nansum`, `nanmean`, `nanmin` and `nanmax` skip the NaNs, as NumPy's functions of the same names. Like
`sum` and `min` they are lazy one-element arrays; `nansum_axis`, `nanmean_axis`, `nanmin_axis` and
`nanmax_axis` reduce along an axis (extent 1 there in the result). An array of NaNs only has
sum `0`, and mean, minimum and maximum NaN. Infinities are values: `nanmax` of `[NaN, inf]` is `inf`.
`nansum`, `nanmin` and `nanmax` exist for `f32`, `f64` and `Df32`, the others (mean and the axis forms,
for which `f64` and `Df32` have no reductions) for `f32`.

```rust
use luma::*;

// Two columns of three: [1, NaN, 3] and [4, NaN, NaN].
let x = Array::from_slice(&[1.0f32, f32::NAN, 3.0, 4.0, f32::NAN, f32::NAN], [3, 2])?;
assert_eq!(x.nansum().item()?, 8.0);
assert_eq!(x.nanmin().item()?, 1.0);
assert_eq!(x.nanmax().item()?, 4.0);
assert_eq!(x.nansum_axis(0).to_vec()?, vec![4.0, 4.0]);
assert_eq!(x.nanmean_axis(0).to_vec()?, vec![2.0, 4.0]);
assert_eq!(x.nanmin_axis(0).to_vec()?, vec![1.0, 4.0]);
assert_eq!(x.nanmax_axis(0).to_vec()?, vec![3.0, 4.0]);

let nothing = Array::from_slice(&[f32::NAN; 3], 3)?;
assert_eq!(nothing.nansum().item()?, 0.0);
assert!(nothing.nanmean().item()?.is_nan() && nothing.nanmin().item()?.is_nan());
# Ok::<(), luma::LumaError>(())
```

The masking (`select(!is_nan(x), x, 0)`) is part of the reduction's first pass, so a pending operand
is not stored. `nansum` reads the elements once; `nanmean` (sum and count) and `nanmin` / `nanmax`
(the extremum, and whether any element is not NaN, which tells an array of NaNs from one whose
minimum is `inf`) read them twice.

## More elementwise functions

`round`, `trunc`, `log1p` and `expm1` are lazy `f32` functions (`round` and `trunc` also for `f64`,
`log1p` and `expm1` also for `Df32`). `round` goes to the nearest integer and *halves to the even
one* (`0.5` is `0`, `1.5` and `2.5` are `2`), what NumPy's `round` and WGSL's `round` do; `trunc`
rounds toward zero. `log1p(x)` is `ln(1 + x)` and `expm1(x)` is `e^x - 1`, accurate where
`(1.0 + x).ln()` and `x.exp() - 1.0` have lost the digits of a small `x`: for `|x|` below 1
(`expm1`) or in `(-0.5, 1)` (`log1p`) they evaluate a series, elsewhere the plain formula (about `1e-6`
relative error; `Df32`: 44 bits).

```rust
use luma::*;

let x = Array::from_slice(&[0.5f32, 1.5, 2.5, -0.5, -2.7], 5)?;
assert_eq!(x.round().to_vec()?, vec![0.0, 2.0, 2.0, 0.0, -3.0]);
assert_eq!(x.trunc().to_vec()?, vec![0.0, 1.0, 2.0, 0.0, -2.0]);

let tiny = Array::from_slice(&[1.0e-9f32, -1.0e-9], 2)?;
// `(1.0 + x).ln()` is 0 for both: 1 + 1e-9 rounds to 1.
for (got, want) in tiny.log1p().to_vec()?.iter().zip([1.0e-9f32, -1.0e-9]) {
    assert!(((got - want) / want).abs() < 1e-6);
}
for (got, want) in tiny.expm1().to_vec()?.iter().zip([1.0e-9f32, -1.0e-9]) {
    assert!(((got - want) / want).abs() < 1e-6);
}
# Ok::<(), luma::LumaError>(())
```

## Recipes

Small statistics and array manipulations that are a line or two of the operations above, and that
fuse like them. Axis 0 varies fastest, as everywhere.

**`diff`** (differences of neighbours along an axis): two slices, one subtraction, no copy:

```rust
use luma::*;

let x = Array::from_slice(&[1.0f32, 4.0, 9.0, 16.0], 4)?;
let n = x.len();
let diff = x.slice(0, 1..) - x.slice(0, ..n - 1);
assert_eq!(diff.to_vec()?, vec![3.0, 5.0, 7.0]);
# Ok::<(), luma::LumaError>(())
```

**`cumprod`**, a scan with `*` (a running sum is `cumsum`, any associative operator can be scanned):

```rust
use luma::*;

let x = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], 4)?;
assert_eq!(x.scan(0, |a, b| a * b).to_vec()?, vec![1.0, 2.0, 6.0, 24.0]);
# Ok::<(), luma::LumaError>(())
```

**`softmax` and `logsumexp`** along an axis: subtract the maximum first, so that `exp` cannot
overflow; the subtraction and `exp` are computed inside the sum:

```rust
use luma::*;

// Two rows of three logits (axis 0 is the row).
let logits = Array::from_slice(&[1.0f32, 2.0, 3.0, 0.0, 0.0, 0.0], [3, 2])?;
let max = logits.max_axis(0);
let e = (&logits - &max).exp();
let total = e.sum_axis(0);
let softmax = (&e / &total).to_vec()?;
let logsumexp = (&max + &total.ln()).to_vec()?;
let third = 1.0f32 / 3.0;
for (got, want) in softmax.iter().zip([0.09003057, 0.24472848, 0.66524094, third, third, third]) {
    assert!((got - want).abs() < 1e-6, "{got} {want}");
}
assert!((logsumexp[0] - 3.4076059).abs() < 1e-5 && (logsumexp[1] - 3.0f32.ln()).abs() < 1e-5);
# Ok::<(), luma::LumaError>(())
```

**`one_hot`** of class labels, one column per label: compare the labels with the class index
(`range`), which broadcasts without a copy:

```rust
use luma::*;

let labels = Array::from_slice(&[2u32, 0, 1, 2], 4)?;
let classes = Array::<u32>::range([3, 4], 0)?; // element (c, i) is c
let one_hot = classes.eq(&labels.reshape([1, 4])).to_f32();
assert_eq!(one_hot.to_vec()?, vec![0., 0., 1., 1., 0., 0., 0., 1., 0., 0., 0., 1.]);
# Ok::<(), luma::LumaError>(())
```

**Shuffles and random permutations**: sort random keys and take the order (`argsort` is stable, so a
tie of two keys keeps the input order and the result is a permutation regardless):

```rust
use luma::*;

let keys = Array::uniform(8, 42)?; // the seed picks the permutation
let order = keys.argsort();
let data = Array::from_slice(&[10u32, 11, 12, 13, 14, 15, 16, 17], 8)?;
let mut shuffled = data.take(&order).to_vec()?;
shuffled.sort();
assert_eq!(shuffled, vec![10, 11, 12, 13, 14, 15, 16, 17]);
# Ok::<(), luma::LumaError>(())
```

**`cov` and `corr`** of a table with one column per feature (`[samples, features]`): the product of
the centred (or standardised) data with its transpose, the centring fused into the product's loads:

```rust
use luma::*;

let samples = 4;
// Two features: the second is twice the first.
let x = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 2.0, 4.0, 6.0, 8.0], [samples, 2])?;
let centred = &x - &x.mean_axis(0);
let cov = (centred.matmul(&centred.transpose()) / (samples as f32 - 1.0)).to_vec()?;
for (got, want) in cov.iter().zip([5.0 / 3.0, 10.0 / 3.0, 10.0 / 3.0, 20.0 / 3.0]) {
    assert!((got - want).abs() < 1e-5, "{got} {want}");
}
let z = &centred / &x.std_axis(0);
let corr = (z.matmul(&z.transpose()) / (samples as f32)).to_vec()?;
assert!(corr.iter().all(|c| (c - 1.0).abs() < 1e-5));
# Ok::<(), luma::LumaError>(())
```

**`clip`** is `clamp`, with the bounds as plain numbers:

```rust
use luma::*;

let x = Array::from_slice(&[-2.0f32, 0.5, 3.0], 3)?;
assert_eq!(x.clamp(0.0, 1.0).to_vec()?, vec![0.0, 0.5, 1.0]);
# Ok::<(), luma::LumaError>(())
```

## Tiles, diagonals, triangles, differences

ArrayFire's structural functions are index maps and generators in luma: none is a kernel of its
own, each fuses into the expression that reads it, and none reads or writes memory until an
evaluation needs it. Shapes list the extents with the first dimension fastest, exactly like
ArrayFire's `dim4`, so the dimension arguments carry over unchanged (`af::tile(a, 2, 3)` is
`a.tile([2, 3])`, `af::diff1(a, 1)` is `a.diff1(1)`). Element `(i, j)` has `i` along dimension 0;
diagonal `k` holds the elements with `j - i = k`.

**`tile`** repeats an array along every dimension, a lazy read like a wrapping pad; **`iota`**
counts the elements of a shape in memory order and tiles the count; **`identity`** is generated
inside the kernel that uses it:

```rust
use luma::*;

let a = Array::from_slice(&[1.0f32, 2.0, 3.0], 3)?;
assert_eq!(a.tile([2, 2]).to_vec()?, vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
assert_eq!(Array::<u32>::iota([2, 2], [1, 2])?.to_vec()?, vec![0, 1, 2, 3, 0, 1, 2, 3]);
let id = Array::<f32>::identity([3, 3])?;
assert_eq!(id.to_vec()?, vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
# Ok::<(), luma::LumaError>(())
```

**`diag(k)`** extracts the `k`-th diagonal of every matrix of dimensions 0 and 1 (dimensions 2 and
3 are batches) as a view, and **`diag_matrix(k)`** builds the matrix with a column on its
diagonal `k`; **`lower(unit)`** and **`upper(unit)`** zero the other triangle (with `unit` the
diagonal is one) and work on non-square matrices and batches. All of them accept every element
type (`f32`, `i32`, `u32`, `bool`, `f64`, `Df32`), and a NaN that is not kept is zeroed, not
propagated:

```rust
use luma::*;

let m = Array::from_slice(&[1.0f32, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 9.0], [3, 3])?;
assert_eq!(m.diag(0).to_vec()?, vec![1.0, 5.0, 9.0]);
assert_eq!(m.diag(1).to_vec()?, vec![2.0, 6.0]);
assert_eq!(m.diag(-1).to_vec()?, vec![4.0, 8.0]);
assert_eq!(m.lower(false).to_vec()?, vec![1.0, 4.0, 7.0, 0.0, 5.0, 8.0, 0.0, 0.0, 9.0]);
assert_eq!(m.upper(true).to_vec()?, vec![1.0, 0.0, 0.0, 2.0, 1.0, 0.0, 3.0, 6.0, 1.0]);
let v = Array::from_slice(&[1.0f32, 2.0], 2)?;
assert_eq!(v.diag_matrix(1).to_vec()?, vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0, 0.0]);
# Ok::<(), luma::LumaError>(())
```

`lower` and `upper` of a stored matrix read only the half they keep: a `select` by a mask of the
coordinates between an input and another value is lowered as a branch when not reading the input
saves enough, and the zeroed half is only written (an RTX 3090: `lower` of a 4096 x 4096 matrix takes
0.132 ms against ArrayFire's 0.131, 0.185 ms before; an RTX 3080 Laptop 0.290 against 0.290, 0.383
before). A kernel that reads through a wrapping view such as `tile` computes 4 elements per
invocation on a long grid (a 4 x 4 tile of 1024 x 1024 takes 0.098 ms on an RTX 3090, 0.117 before,
ArrayFire 0.108; 0.209 ms on an RTX 3080 Laptop, 0.249 before, ArrayFire 0.227). `iota` counts exactly for
`u32`, `i32`, `f64` and `Df32`; an `f32` one rounds above `2^24`.

A diagonal outside the matrix would be empty, and luma has no empty arrays: it poisons the result
with `InvalidShape` (as `diff1` does for an extent below 2).

**`replace(&cond, b)`** is `af::replace` returning a new array (`a` where `cond`, else `b`, an
array or a scalar: `select(&cond, &a, b)`), and **`diff1(axis)`** / **`diff2(axis)`** are the first
and second differences along an axis, one and two shorter, as a fused difference of slices that
reads the array once:

```rust
use luma::*;

let x = Array::from_slice(&[1.0f32, f32::NAN, 4.0, 9.0], 4)?;
assert_eq!(x.replace(&!x.is_nan(), 0.0).to_vec()?, vec![1.0, 0.0, 4.0, 9.0]);
let y = Array::from_slice(&[0.0f32, 1.0, 4.0, 9.0, 16.0], 5)?;
assert_eq!(y.diff1(0).to_vec()?, vec![1.0, 3.0, 5.0, 7.0]);
assert_eq!(y.diff2(0).to_vec()?, vec![2.0, 2.0, 2.0]);
# Ok::<(), luma::LumaError>(())
```
