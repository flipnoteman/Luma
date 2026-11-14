# Views and writes

A *view* is an array that reads another array through an index map: a slice, a padded copy, a
reversed copy, a transpose. Views move no data. They are lazy operations like any other, so they
fuse into the kernel that reads them, and a view of a lazy expression is computed inside that
kernel at the index it is read. This chapter covers the views Luma has, the arrays that write
through to what they view, and scattering with `put`.

## Reading through views

```rust
use luma::*;

// A [4, 3] array; the first dimension varies fastest.
let a = Array::from_slice(&(0..12).map(|i| i as f32).collect::<Vec<_>>(), [4, 3])?;

// `slice(axis, range)`: the part `range` of the extent along `axis`.
assert_eq!(a.slice(0, 1..3).shape(), Shape::from([2, 3]));
assert_eq!(a.slice(1, ..1).to_vec()?, vec![0.0, 1.0, 2.0, 3.0]);

// `flip(axis)` reverses, `transpose()` swaps the first two axes.
assert_eq!(a.slice(1, ..1).flip(0).to_vec()?, vec![3.0, 2.0, 1.0, 0.0]);
assert_eq!(a.transpose().shape(), Shape::from([3, 4]));
assert_eq!(a.transpose().slice(1, ..1).to_vec()?, vec![0.0, 4.0, 8.0]);

// A difference of two slices reads `a` once, in one kernel.
let d = a.slice(0, 1..) - a.slice(0, ..3);
assert_eq!(d.to_vec()?, vec![1.0; 9]);
# Ok::<(), luma::LumaError>(())
```

| View | Result |
| --- | --- |
| [`slice(axis, range)`](https://docs.rs/luma/latest/luma/struct.Array.html#method.slice) | the range along one axis; an empty or out-of-range range poisons the result |
| `flip(axis)` | reversed along the axis |
| `transpose()` | the first two axes swapped |
| `broadcast_to(shape)` | extents of 1 repeated to `shape` (see [Arrays](arrays.md#shapes-and-broadcasting)) |
| `shift(offsets)` | circular shift: element `i` moves to `i + offset`, wrapping |
| `shift_fill(offsets, value)` | the same, the cells nothing moves into read `value` |
| `shift_clamp(offsets)` | the same, the cells read the nearest element |
| `shift_mirror(offsets)` | the same, the cells read the array mirrored at its boundary |
| `pad(widths, mode)` | extra cells on both sides of each axis, filled as the [`PadMode`](https://docs.rs/luma/latest/luma/enum.PadMode.html) says |
| `take(&indices)` | gather: element `i` is `self[indices[i]]` (flat index, clamped to the last element) |
| `index(axis)` | each element's coordinate along `axis`, as `u32` |

Offsets are `isize` or arrays of up to four of them (`shift(1)`, `shift([1, -1])`).

```rust
use luma::*;

let x = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], 4)?;
assert_eq!(x.shift(1).to_vec()?,              vec![4.0, 1.0, 2.0, 3.0]);
assert_eq!(x.shift_fill(1, 9.0).to_vec()?,    vec![9.0, 1.0, 2.0, 3.0]);
assert_eq!(x.shift_clamp(-1).to_vec()?,       vec![2.0, 3.0, 4.0, 4.0]);
assert_eq!(x.shift_mirror(2).to_vec()?,       vec![2.0, 1.0, 1.0, 2.0]);

// `pad([(before, after), ..], mode)` pads the first axes; the others are left alone.
assert_eq!(x.pad([(2, 1)], PadMode::Zero).to_vec()?,   vec![0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 0.0]);
assert_eq!(x.pad([(1, 1)], PadMode::Edge).to_vec()?,   vec![1.0, 1.0, 2.0, 3.0, 4.0, 4.0]);
assert_eq!(x.pad([(1, 1)], PadMode::Mirror).to_vec()?, vec![1.0, 1.0, 2.0, 3.0, 4.0, 4.0]);
assert_eq!(x.pad([(1, 1)], PadMode::Wrap).to_vec()?,   vec![4.0, 1.0, 2.0, 3.0, 4.0, 1.0]);

// `take` gathers by flat index, `index` generates coordinates.
let picks = Array::from_slice(&[3u32, 0, 0, 9], 4)?;
assert_eq!(x.take(&picks).to_vec()?, vec![4.0, 1.0, 1.0, 4.0]); // 9 is clamped to 3
assert_eq!(x.index(0).to_vec()?, vec![0, 1, 2, 3]);
# Ok::<(), luma::LumaError>(())
```

`PadMode::Mirror` mirrors about the *boundary face*, not the boundary cell: `-1` reads `0`, `-2`
reads `1`, `n` reads `n - 1`. That is the ghost cell of a finite-volume wall, and it is why the
`Mirror` row above equals the `Edge` row for a pad of one. Circular `shift` is the one view that
wraps; `shift_fill`, `shift_clamp` and `shift_mirror` are the shifts of a bounded domain.

A view is read-only. Writing to it (`assign`, `write`, `+=`, `put`) is an error that names
`slice_mut` and friends, which are the views that write through ([below](#views-that-write)).

### Joining arrays

`concat(axis, [&a, &b, ..])` and `concat_slice(axis, &[a, b, ..])` join arrays along an existing
axis; `stack(axis, [..])` and `stack_slice` join equal-shaped arrays along a new axis.

```rust
use luma::*;

let a = Array::from_slice(&[1.0f32, 2.0], 2)?;
let b = Array::from_slice(&[3.0f32, 4.0, 5.0], 3)?;
assert_eq!(concat(0, [&a, &b]).to_vec()?, vec![1.0, 2.0, 3.0, 4.0, 5.0]);

// Stack equal shapes: a new axis of extent `parts` at position `axis`.
let c = Array::from_slice(&[10.0f32, 20.0], 2)?;
let s = stack(1, [&a, &c]);             // two columns of two: shape [2, 2]
assert_eq!(s.shape(), Shape::from([2, 2]));
assert_eq!(s.to_vec()?, vec![1.0, 2.0, 10.0, 20.0]);
let rows = stack(0, [&a, &c]);          // [2 parts, 2]: element (k, i) is parts[k][i]
assert_eq!(rows.to_vec()?, vec![1.0, 10.0, 2.0, 20.0]);
# Ok::<(), luma::LumaError>(())
```

They differ in what they cost:

- **`concat` is lazy.** Every part is read through a view and the result picks by coordinate. It
  fuses into whatever reads it; the parts may be lazy expressions (the ghost cells of a state,
  say), and only the part a cell needs is computed at that cell. Use it for a few pieces read
  by something else.
- **`stack` is eager and one kernel.** The parts are fused into a single kernel that writes
  them side by side, so each output element is computed once. The result is an evaluated array.
  Use it to assemble many expressions into an array that is then read as a whole (the entries of
  a block matrix). Parts that live in several buffers (very large arrays) are joined lazily like
  `concat`.

### Reshape

`reshape(shape)` keeps the element count and shares the buffer. The reshape of a *lazy view*
(a slice, flip, transpose, pad or non-circular shift that has not run) stays lazy: it is another
handle on the pending view, and fuses into what reads it. `x.slice(0, k..k + 1).reshape([ni, nj])`
of an `[n, ni, nj]` array reads `x` inside the kernel that uses it. Any *other* pending
expression is evaluated first, because the data it shares must exist. The reshape of a view
does not write through to the array the view came from.

```rust
use luma::*;

let x = Array::from_slice(&(0..12).map(|i| i as f32).collect::<Vec<_>>(), [2, 3, 2])?;
let plane = x.slice(2, 1..2).reshape([2, 3]);   // [2, 3, 1] -> [2, 3], still lazy
assert_eq!(plane.to_vec()?, vec![6.0, 7.0, 8.0, 9.0, 10.0, 11.0]);
# Ok::<(), luma::LumaError>(())
```

## Views that write

`slice`, `flip` and `transpose` have `_mut` forms that borrow the array mutably and return a
[`ViewMut`](https://docs.rs/luma/latest/luma/struct.ViewMut.html): a region that writes through
to the array it was taken from. Only views that keep every element distinct can write through; a
`shift`, `pad` or `take` cannot (several cells would read one element), and writing to one just
fills its own buffer without touching what it was computed from.

```rust
use luma::*;

let mut w = Array::<f32>::zeros([4, 3])?;

// `assign` writes an expression (or a single value) into the region.
w.slice_mut(1, ..1).assign(7.0)?;                  // the first row
w.slice_mut(0, 2..).slice(1, 1..2).assign(5.0)?;   // refine a view: row 1, columns 2 and 3

// Compound assignment works on the view too: `region op= rhs`.
let mut v = w.slice_mut(0, 0..2);
v += 1.0;

// `flip_mut` writes the region in reversed order, `transpose_mut` in transposed order.
let r = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], 4)?;
w.slice_mut(1, 2..3).flip(0).assign(&r)?;          // row 2, last element first

assert_eq!(
    w.to_vec()?,
    vec![8.0, 8.0, 7.0, 7.0,   1.0, 1.0, 5.0, 5.0,   4.0, 3.0, 2.0, 1.0],
);
# Ok::<(), luma::LumaError>(())
```

Writes are lazy like everything else: nothing waits for the GPU, and lazy arrays recorded
*before* the write keep the values they were recorded against (the same rule as `write` and `+=`,
see [Lazy evaluation](lazy.md)). The right-hand side may read the array being written. When it
reads the cells it overwrites, Luma computes what it needs first; the result is always the one a
host loop over the old contents would give.

```rust
use luma::*;

let mut u = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], 6)?;
let before = &u * 1.0;
// Overlapping read and write: the right-hand side sees the old contents.
let rhs = u.slice(0, 2..4).flip(0) * -1.0;
u.slice_mut(0, ..2).assign(rhs)?;
assert_eq!(u.to_vec()?, vec![-4.0, -3.0, 3.0, 4.0, 5.0, 6.0]);
assert_eq!(before.to_vec()?, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
# Ok::<(), luma::LumaError>(())
```

`assign` fails with `LumaError::ReadOnly` on the state passed to an `iterate` step and with
`LumaError::Chunked` on arrays split over several buffers (see the
[FAQ](faq.md#results-and-errors)). Custom kernels take `&mut view` arguments as well
([Custom kernels](custom-kernels.md)).

### Ghost cells

The pattern behind the finite-volume solvers of the [worked examples](examples.md): pad the state
with ghost cells, then assign a boundary condition into the ghost region.

```rust
use luma::*;

let n = 6;
let q = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], n)?;

// Two ghost cells on the left, copied from the edge...
let mut w = q.pad([(2, 2)], PadMode::Edge);
// ...then overwritten with the interior mirrored and negated (a reflecting wall).
let wall = w.slice(0, 2..4).flip(0) * -1.0;
w.slice_mut(0, ..2).assign(wall)?;

assert_eq!(w.to_vec()?, vec![-2.0, -1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 6.0, 6.0]);
# Ok::<(), luma::LumaError>(())
```

Here `w` was a lazy pad when the assignment ran, so nothing was written and nothing launched:
from then on `w` *is* the expression "the pad with this region replaced" and fuses into whatever
reads it (a flux, a stencil). The values are the same as when `w` has a buffer. When the
boundary expression is costly, the ghost cells compute it and the interior cells do not: Luma
writes the replacement as an `if`/`else` on the coordinate.

## Scatter: `put`

`a.put(&indices, values)` writes `a[indices[i]] = values[i]` in place. `values` is an array of
the shape of `indices` or a single value. Indices of `len` or more are skipped; of several equal
indices one (unspecified) wins. Call it on the array itself, not on a view.

```rust
use luma::*;

let mut a = Array::<f32>::zeros(6)?;
let idx = Array::from_slice(&[4u32, 0, 99], 3)?;
a.put(&idx, &Array::from_slice(&[1.0f32, 2.0, 3.0], 3)?)?;   // 99 is skipped
assert_eq!(a.to_vec()?, vec![2.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
a.put(&idx, 9.0)?;
assert_eq!(a.to_vec()?, vec![9.0, 0.0, 0.0, 0.0, 9.0, 0.0]);
# Ok::<(), luma::LumaError>(())
```

`take` and `put` are the gather and scatter pair; the [integer chapter](integers.md#lookup-tables-with-take)
uses `take` for table lookups, and the ML-KEM example uses `put` for rejection sampling.

## What fuses

- Every view of an array or expression is read inside the kernel that uses it. `a.slice(..) -
  a.slice(..)` is one kernel reading `a` once.
- A view of a *lazy* expression is computed at the index it is read. When one expression is read
  at several offsets (the flux of a face read by the two cells next to it), the
  [cost model](performance.md#the-cost-model) picks the cheapest plan: recompute it per read,
  compute it once per cell of a workgroup tile in workgroup memory, or evaluate it first.
  `x.cache()` evaluates on request.
- `concat` fuses; `stack` is its own eager kernel; `reshape` of a view fuses.
- An `assign` into a lazy array fuses; into an array with a buffer it is one kernel that
  writes the region. Expressions that do not read the cells being written fuse into that
  kernel.
- Circular `shift` has its own rule: a small expression is recomputed at a few offsets or
  staged; a larger one is evaluated first.

What does not fuse: `scan` and axis reductions ([Reductions](reductions.md)), and custom kernels,
which see the arrays computed.
