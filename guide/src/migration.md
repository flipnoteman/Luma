# Migration from 0.3

0.4 changes the type of comparisons: they return masks (`Array<Mask>`) instead of `Array<u32>`.

## Comparisons return `Array<Mask>`

`lt`, `le`, `gt`, `ge`, `eq`, `ne` and `is_finite` / `is_nan` / `is_inf` return `Array<Mask>`;
`select`, `sum_where` and `Fields::select` take `&Array<Mask>`. A mask is a boolean per element,
stored as a `u32` that is 0 or 1. Code that passes a comparison to `select` or `sum_where` compiles
as it is. Code that used the `0` / `1` as a number converts it:

```text
// 0.3
let above = x.gt(0.0);
let count = above.sum().item()?;          // Array<u32>
let weight = &above * 0.5f32;             // does not exist: u32 * f32
let negated = &above ^ 1u32;
// 0.4
let above = x.gt(0.0);
let count = above.count().item()?;
let weight = above.to_f32() * 0.5;
let negated = !&above;
```

```rust
use luma::*;

let x = Array::from_slice(&[1.5f32, -2.0, 3.0, -4.0], 4)?;
let above: Array<Mask> = x.gt(0.0);
let both = &above & &x.lt(2.0);                    // masks combine with & | ^ !
assert_eq!(above.count().item()?, 2);
assert_eq!(both.to_u32().to_vec()?, vec![1, 0, 0, 0]);
assert_eq!(above.to_f32().to_vec()?, vec![1.0, 0.0, 1.0, 0.0]);
assert!(bool::from(above.any().item()?));
assert!(!bool::from(above.all().item()?));
assert_eq!(select(&above, 1.0f32, 0.0).to_vec()?, vec![1.0, 0.0, 1.0, 0.0]);
# Ok::<(), luma::LumaError>(())
```

Sums of masks (`sum`, `cumsum`, `sum_axis`, `mean`) are not defined, since the result would no
longer be a mask; `count()` is the sum of the true elements, `min` and `max` are `all` and `any`.

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
