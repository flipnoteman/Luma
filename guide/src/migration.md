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
