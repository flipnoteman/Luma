# Records

A flow solver has four fields per cell, a Monte Carlo step has a position and a velocity, a hash
has fifty words of state. You can keep them in separate variables, but then every view, padding,
ghost-cell write and loop is repeated per field. A *record* is a group of arrays that you treat as
one value: `q.slice(0, 1..)`, `&q * 2.0`, `q.pad(..)` apply to every array in it.

Records are only a way to write less code. Fusion is the same as for separate arrays: the
arithmetic is on arrays, and the engine sees the same graph.

## What is a record

Anything that implements the [`Fields`](https://docs.rs/luma/latest/luma/trait.Fields.html)
trait:

- an `Array<T>` (a record of one field),
- an array `[F; N]` of records,
- a tuple of up to six records `(A, B, ..)`,
- a struct declared with `luma::fields!`.

All the arrays of a record have the same element type (`f32`, `Df32`, `f64` or `u32`) and the same
shape. `Fields::Elem` is that type.

```rust
use luma::*;

// A tuple and an array are records already.
let pair = (Array::from_slice(&[1.0f32, 2.0, 3.0], 3)?, Array::from_slice(&[4.0f32, 5.0, 6.0], 3)?);
let tail = pair.slice(0, 1..);                     // a slice of every array
assert_eq!(tail.1.to_vec()?, vec![5.0, 6.0]);

let padded = pair.pad([(1, 0)], PadMode::Zero);    // a pad of every array
assert_eq!(padded.0.to_vec()?, vec![0.0, 1.0, 2.0, 3.0]);

let sum = pair.dot(&pair);                         // sum over the fields of products: a.a + b.b
assert_eq!(sum.to_vec()?, vec![17.0, 29.0, 45.0]);
# Ok::<(), luma::LumaError>(())
```

Tuples and arrays get the *methods* of the trait. For named fields and arithmetic operators
declare a struct with `luma::fields!`.

## `luma::fields!`

```rust
use luma::*;

luma::fields! {
    /// The conserved state of a flow: density and energy.
    #[derive(Clone, Debug)]
    pub struct Cons {
        pub rho: Array<f32>,
        pub e: Array<f32>,
    }
}

let q = Cons {
    rho: Array::from_slice(&[1.0, 2.0], 2)?,
    e: Array::from_slice(&[3.0, 4.0], 2)?,
};

// Arithmetic between records, with a value, and with an array (applied to every field).
let r = &q * 2.0 + &q;
assert_eq!(r.rho.to_vec()?, vec![3.0, 6.0]);
assert_eq!(r.e.to_vec()?, vec![9.0, 12.0]);

let weights = Array::from_slice(&[1.0f32, 10.0], 2)?;
assert_eq!((&q * &weights).e.to_vec()?, vec![3.0, 40.0]);
assert_eq!(Cons::COUNT, 2);
# Ok::<(), luma::LumaError>(())
```

The macro takes one struct with named fields and writes the struct, `impl Fields`, and the
operators.

- **Element type.** It is read from the first field when that is written `Array<T>`. For any other
  first field (a nested record, `[Array<T>; N]`, a tuple, `luma::Array<T>`) put `#[elem = T]`
  before the other attributes:

  ```rust
  use luma::*;

  luma::fields! {
      #[elem = f32]
      struct Both {
          q: (Array<f32>, Array<f32>),
          t: [Array<f32>; 2],
      }
  }
  assert_eq!(Both::COUNT, 4);
  ```

- **Attributes and visibility** (`#[derive(..)]`, doc comments, `pub`, `pub(crate)`) are passed
  through; a trailing comma is optional.
- **Field types** are `Array<T>`, arrays and tuples of records, and other records. A mismatched
  element type is a compile error. Nested records count every array (`COUNT` is the total).
- **Not supported:** generic structs, tuple structs, attributes on fields, and an empty record.
- **`u32` records** work and have no unary minus: `luma::fields! { struct Lanes { lo: Array<u32>,
  hi: Array<u32> } }`.

### What a record gets

| | |
| --- | --- |
| operators | `+ - * /` between records, with an element value (`f32` for an `f32` record) and with an `Array<Elem>`, any reference/owned combination, the scalar or array on either side; `+= -= *= /=` with `&Record`, a value and `&Array`; unary `-` on `&Record` and `Record` (not `u32`) |
| views | `slice`, `pad`, `flip`, `transpose`, `shift`, `shift_fill`, `shift_clamp`, `shift_mirror`, `take`, `broadcast_to`, `copy` |
| writes | `slice_mut`, `flip_mut`, `transpose_mut` (a [`ViewMut`](views.md#views-that-write) of the record), `assign`, `put` |
| helpers | `map(f)`, `zip(&other, f)`, `dot(&other)`, `Fields::select(&mask, &a, &b)`, `arrays()`, `eval()`, `cache()` |

Every method applies the array operation of the same name to each field, keeps its laziness, and
poisons the fields with the caller's source location on a failure.

## Writes and ghost cells

A record assigned through a region is written field by field, in one kernel when the fields have
buffers; fields that are still lazy expressions are not written at all (the record denotes the
new values from then on). Here is the wall boundary of a flow solver on a whole state:

```rust
use luma::*;

luma::fields! {
    struct Cons {
        rho: Array<f32>,
        e: Array<f32>,
    }
}

let q = Cons {
    rho: Array::from_slice(&[1.0, 2.0], 2)?,
    e: Array::from_slice(&[3.0, 4.0], 2)?,
};

// One ghost cell on each side, copied from the edge; then the left one gets
// the interior neighbour's density and the opposite energy (a reflecting wall).
let mut w = q.pad([(1, 1)], PadMode::Edge);
let wall = w.slice(0, 1..2);
let ghost = Cons { rho: wall.rho, e: -&wall.e };
w.slice_mut(0, ..1).assign(ghost)?;

assert_eq!(w.rho.to_vec()?, vec![1.0, 1.0, 2.0, 2.0]);
assert_eq!(w.e.to_vec()?, vec![-3.0, 3.0, 4.0, 4.0]);
# Ok::<(), luma::LumaError>(())
```

Passing the right-hand side by value (`assign(ghost)`) lets its expressions fuse into the write;
by reference (`assign(&ghost)`) keeps `ghost` usable afterwards.

## Updating one field

A record's fields are ordinary public arrays. Update one and the others stay as they are:

```rust
use luma::*;

luma::fields! {
    struct Particles {
        x: Array<f32>,
        v: Array<f32>,
    }
}

let mut p = Particles {
    x: Array::from_slice(&[0.0, 1.0], 2)?,
    v: Array::from_slice(&[1.0, -1.0], 2)?,
};
let dt = 0.5;
p.x += &(&p.v * dt);          // only `x` changes (compound assignment on the field)
p.v = &p.v * 0.5;             // or replace the field by a new expression
assert_eq!(p.x.to_vec()?, vec![0.5, 0.5]);
assert_eq!(p.v.to_vec()?, vec![0.5, -0.5]);
# Ok::<(), luma::LumaError>(())
```

## Records in loops

[`iterate_until_all`](loops.md#stop-conditions-iterate_until_all) takes any record as the state
and returns it, so the step reads like the maths:

```rust
use luma::*;

luma::fields! {
    struct Cons {
        rho: Array<f32>,
        e: Array<f32>,
    }
}

let q = Cons {
    rho: Array::from_slice(&[1.0, 2.0], 2)?,
    e: Array::from_slice(&[3.0, 4.0], 2)?,
};
let run = iterate_until_all(
    q,
    Until::new(100)
        .record("max", |s: &Cons| s.rho.max())
        .stop_when(|h| h["max"].last().le(1e-3)),
    |s, _| s * 0.5,                      // the step is one record expression
)?;
assert_eq!(run.steps, 11);               // 2 / 2^11 < 1e-3
assert_eq!(run.state.e.to_vec()?, vec![3.0 / 2048.0, 4.0 / 2048.0]);
# Ok::<(), luma::LumaError>(())
```

[`iterate_all`](loops.md#elementwise-loops-iterate_all) takes its states as `[&Array<T>; N]`;
use `Fields::arrays()` or the fields directly.

## When records help

- **Records shorten code.** A Rusanov flux over a four-field state is `0.5 * (&fl + &fr) -
  0.5 * &smax * (&ur - &ul)`, written once, instead of four near-identical lines per face.
  Views (`w.slice(0, 1..)`), ghost cells and `cache()` become one call.
- **Fusion does not care.** A record of four arrays is four arrays: the same nodes, the same
  kernels, the same cost-model decisions. A record op on four fields is not one big operation;
  it is four operations that the engine fuses and evaluates together. `cache()` and `eval()` on
  a record evaluate all its fields in one batch, so fields that share work compute it once.
- **Use separate arrays** when the fields are not treated alike: different shapes, different
  element types, or operations that touch one field only. Mixed element types cannot be in one
  record (`Df32` and `f32`, say); keep two records.

The Euler solver in the [worked examples](examples.md) is written entirely with records.
