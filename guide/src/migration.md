# Migration from 0.3

0.4 changes the type of comparisons: they return boolean arrays (`Array<bool>`) instead of
`Array<u32>`. Most 0.3 code compiles as it is, because booleans promote to numbers the way they do
in NumPy (below). Everything else in 0.4 is added ([the list below](#new-in-04-that-you-may-want)).

## Comparisons return `Array<bool>`

`lt`, `le`, `gt`, `ge`, `eq`, `ne` and `is_finite` / `is_nan` / `is_inf` return `Array<bool>`. On the
GPU a `bool` is one bit (`n` booleans are `ceil(n / 32)` `u32` words: a
[custom kernel](custom-kernels.md) that took a comparison as one `u32` per element reads
`(a[i >> 5u] >> (i & 31u)) & 1u` now); on the host `from_slice`, `to_vec` and `item` use plain `bool`s.

What compiles unchanged:

- `select(&cond, a, b)`, `sum_where(&cond)` and `Fields::select` take an `Array<bool>`, and still
  take an `Array<u32>` (non-zero is true), so code that builds its own `u32` conditions is as
  before.
- `bool * bool` is the logical and: `a.gt(0.0) * b.gt(0.0)`.
- A boolean array times a number is the number where the boolean is true and 0 elsewhere, in
  either order and with a scalar: `mask * &x`, `&x * mask`, `mask * 0.5`. `+` and `-` convert the
  booleans to the 0 and 1 of the other operand's type first. These work for `f32`, `u32`, `i32`,
  `f64` and `Df32` arrays, and the product fuses as a `select`, not a cast and a multiply.
- `sum()` of booleans counts the true elements and returns an `Array<u32>` (as in NumPy), also
  `count()`, `sum_axis` and `cumsum`.

What does not:

```text
// 0.3
let m: Array<u32> = x.gt(0.0);              // the type of a comparison was named
let both = &x.gt(0.0) + &y.gt(0.0);         // 0, 1 or 2
let idx = x.gt(0.0).cumsum(0) - 1u32;       // running count, minus one
let hit = x.gt(0.0);  let miss = &hit ^ 1u32;
// 0.4
let m: Array<bool> = x.gt(0.0);
let both = x.gt(0.0).to_u32() + y.gt(0.0).to_u32();    // or `|` / `&` for a logical result
let idx = x.gt(0.0).cumsum(0) - 1u32;                  // cumsum of booleans is a u32 array: unchanged
let miss = !&hit;
```

Also changed: a comparison is no longer an index (`take`, `put`: convert with `to_u32()`), has no
`/`, `%`, shifts, `minimum` or `maximum` (convert first), and reads back as `Vec<bool>`. An
integer literal next to a boolean array needs a suffix (`mask * 2u32`), since `u32` and `i32`
arrays are both valid results.

```rust
use luma::*;

let x = Array::from_slice(&[1.5f32, -2.0, 3.0, -4.0], 4)?;
let above: Array<bool> = x.gt(0.0);
let both = &above & &x.lt(2.0);                    // booleans combine with & | ^ !
assert_eq!(above.count().item()?, 2);
assert_eq!(both.to_vec()?, vec![true, false, false, false]);
assert_eq!(above.to_f32().to_vec()?, vec![1.0, 0.0, 1.0, 0.0]);
assert!(above.any().item()?);
assert!(!above.all().item()?);
assert_eq!(select(&above, 1.0f32, 0.0).to_vec()?, vec![1.0, 0.0, 1.0, 0.0]);

// Promotion, as in NumPy: the number where true, 0 elsewhere.
assert_eq!((&above * &x).to_vec()?, vec![1.5, 0.0, 3.0, 0.0]);
assert_eq!((&above * 0.5).to_vec()?, vec![0.5, 0.0, 0.5, 0.0]);
assert_eq!((&x + &above).to_vec()?, vec![2.5, -2.0, 4.0, -4.0]);
// A `u32` condition still works.
let words = Array::from_slice(&[1u32, 0, 7, 0], 4)?;
assert_eq!(select(&words, &x, 0.0).to_vec()?, vec![1.5, 0.0, 3.0, 0.0]);
# Ok::<(), luma::LumaError>(())
```

The numeric reductions (`reduce_axis`, `scan`, `mean`, `norm`) are not defined for `bool`: convert
with `to_u32()` or `to_f32()`. A masked-out `NaN` or infinity in `mask * &x` gives 0, where a
multiplication would give `NaN`.

## New in 0.4 that you may want

- [`i32` arrays, casts and `bool`](arrays.md#element-types) (`to_u32`, `to_i32`, `to_f32`; `count`,
  `any`, `all`).
- [`matmul`, axis statistics (`reduce_axis`, `mean_axis`, `var_axis`, `std_axis`), accumulating
  scatters (`put_add`, `bincount`, `histogram`), sorting, group-by, `unique` and quantiles](data.md).
  Axis reductions no longer evaluate a pending operand first
  ([What fuses](performance.md#primitives-that-read-an-expression)).
- Guide pages for [`Scalar`](lazy.md#scalars-that-change), [elementwise math](arrays.md#elementwise-operations),
  [convolution](arrays.md#convolution) and the [tridiagonal solvers](loops.md#line-solvers-solve_tridiagonal),
  which earlier releases already had.

# Migration from 0.2

0.3 changes two things your 0.2 code can trip over. Almost everything else in 0.3 is added; the few behaviour changes (a lazy `reshape` of a view, fusion decided by a cost model, a larger buffer pool) are listed under *Changed* in the changelog.

## Reductions are lazy arrays

In 0.2, `sum`, `min`, `max`, `mean`, `norm` and `reduce` returned host values. In 0.3 they return a
one-element `Array` that fuses with what follows, so `&x / &x.sum()` stays on the GPU. Read the
value with `item()?` (and `reduce` results with `to_array()?`):

```text
// 0.2
let total: f32 = x.sum()?;
// 0.3
let total: f32 = x.sum().item()?;
```

```rust
use luma::*;

let x = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], 4)?;
assert_eq!(x.sum().item()?, 10.0);
assert_eq!(x.max().item()?, 4.0);

// Several reductions in one pass over the data come back as an array:
let both = x.reduce([Reduction::Min, Reduction::Max]);
assert_eq!(both.to_array::<2>()?, [1.0, 4.0]);

// Staying on the GPU:
let mean_removed = &x - &x.mean();
assert_eq!(mean_removed.to_vec()?, vec![-1.5, -0.5, 0.5, 1.5]);
# Ok::<(), luma::LumaError>(())
```

The `[f32; 2]` and `[f32; 4]` reductions (`vec2` / `vec4` element arrays) still return host values.

## Broadcasting per axis

An extent of 1 now broadcasts: `[N, 1] + [1, M]` is `[N, M]`, also for the branches of `select` and
in compound assignment. Code that relied on a shape mismatch being an error between such shapes
now gets a result. `broadcast_to(shape)` makes it explicit.

## New in 0.3 that you may want

- [Views](views.md) that fuse into the reading kernel (`slice`, `pad`, `flip`, `transpose`, `take`,
  `concat`, ...) and mutable views with `assign`.
- [Records](records.md) (`luma::fields!`) for solver state.
- [Bit operations](integers.md) on `u32`, [`Df32` and `f64`](precision.md).
- [`iterate_until_all`](loops.md) with a device-side stop rule.
- `Context::cost_model`, `Profile::splits`, `LUMA_CHECK_FINITE`, and an on-disk cache of compiled
  pipelines ([Environment variables](environment.md)).

The complete list, including performance changes and fixes, is in the
[changelog](https://github.com/flipnoteman/Luma/blob/main/CHANGELOG.md).
