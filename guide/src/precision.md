# Precision

`f32` is the default and what GPUs are built for. When a result needs more digits (a residual
that falls below `1e-7`, a long time integration, a sum of a billion terms), Luma has two wider
types and compensated reductions:

| Type | Bits | Where it runs | Cost |
| --- | --- | --- | --- |
| `f32` | 24 of mantissa | everywhere | 1 |
| `Df32` | about 48 | everywhere (computed with `f32` arithmetic) | about 27 to 44 `f32` operations per add or multiply on the GPUs measured, about 2 on llvmpipe |
| `f64` | 53 | devices with `SHADER_F64` | 11 to 41 `f32` operations on the GPUs measured |

Both wide types take 8 bytes per element. Pick the narrowest that gives the accuracy you need,
and use wide types only where it matters: the cost model weighs `Df32` and `f64` arithmetic by
the factors it measured ([Performance](performance.md#the-cost-model)).

## `f64`

`Array<f64>` exists on every context, but only works on devices that have 64-bit shader floats
(`wgpu::Features::SHADER_F64`, in practice Vulkan GPUs). Without it every `f64` operation is
poisoned with [`LumaError::Unsupported`](https://docs.rs/luma/latest/luma/enum.LumaError.html).
Check first:

```rust
use luma::*;

let ctx = Context::new()?;
let x = Array::from_slice_on(&ctx, &[2.0f32, 3.0], 2)?;

if ctx.features().contains(luma::wgpu::Features::SHADER_F64) {
    let r = x.to_f64().sqrt();
    let got = r.to_vec()?;
    assert!((got[0] - 2.0f64.sqrt()).abs() < 1e-15);
} else {
    // Every f64 operation is refused, with a clear error.
    let err = x.to_f64().to_vec().unwrap_err();
    assert!(matches!(err.root(), LumaError::Unsupported(_)));
}
# Ok::<(), luma::LumaError>(())
```

`f64` arrays have `+ - * /`, `sqrt`, `abs`, `sign`, `floor`, `ceil`, `trunc`, `round`, `minimum`, `maximum`,
comparisons, `select`, views and reductions. Anything else (`exp`, trigonometry, generators such
as `range`) is refused when recorded. Use `Df32` when you need a wide type on every device, or
cast down to `f32` for the function.

## `Df32`

A `Df32` is the unevaluated sum `hi + lo` of two `f32`, with `|lo|` at most half an ulp of `hi`.
Together they carry about 48 bits of mantissa (a relative error near `2^-46`; division and `sqrt`
are accurate to about `2^-45`), and the arithmetic uses `f32` operations only, so it runs on every device.

```rust
use luma::*;

// Plain f32 loses the 1.0 in (1e8 + 1) - 1e8; a double-float keeps it.
let a = Array::<f32>::full(8, 1.0e8)?.to_df32();
let one = Array::<f32>::full(8, 1.0)?.to_df32();
let kept = ((&a + &one) - &a).to_f32();
assert!(kept.to_vec()?.iter().all(|&v| v == 1.0));

// Df32 values are built from f64 on the host; `hi` and `lo` hold the 48 bits.
let x = Array::from_slice(&[Df32::from(1.0 + 1e-12)], 1)?;
let tiny = (&x - Df32::from(1.0f32)).to_f32().item()?;
assert!((f64::from(tiny) - ((1.0 + 1e-12) - 1.0)).abs() < 1e-18);
# Ok::<(), luma::LumaError>(())
```

`Array<Df32>` has `+ - * /`, `sqrt`, `abs`, comparisons, `minimum`, `maximum`, `select`, views,
`assign`, and reductions (`sum`, `min`, `max`; `item()` gives a `Df32`). It also has `exp`, `ln`,
`tanh`, `log1p`, `expm1` and `powf` (below). It has no trigonometry: cast to `f32` and back around such a call.
Scalars may be `Df32` values; a Rust `f32` literal in a `Df32` expression is not converted for you
(write `Df32::from(1.0f32)` or build the array with `to_df32()`).

**Functions.** `exp`, `ln`, `tanh`, `log1p`, `expm1` and `powf(Df32)` are built from `Df32` arithmetic and fuse like
it. Their relative error is below `2^-43` (`1e-13`; at most `4.3e-14` measured against `f64` on the
RTX 3090 and on llvmpipe): `exp` on `[-55, 80]`, `ln` from `1e-24` to `1e30` (and near 1, where the
error is relative to the result), `tanh` on `[-30, 30]` (also near 0), `log1p` and `expm1` (also near 0: a series in the argument
rather than `ln(1 + x)`), and `x^1.5` or `x^0.7` for
`x` in `[1, 3000]`. `powf` splits the exponent of `x` exactly, so its error does not grow with
`y ln x`. Where a `Df32` is below about `1e-25` its low word is under the smallest normal `f32`
and loses bits, which limits every function there: `exp` of an argument under `-55` has an absolute
error of about `1e-38` (a relative error of that over the result), and a result below `2^-126` is `0`. Out of range `exp` is
infinity above `88.72` and `0` below `-87.33`, `tanh` is `+-1` from `|x| > 20`, and `ln` and `powf`
follow `f32` (`ln(0)` is `-inf`, `ln` and `powf` of a negative base are `NaN`, `x^0` is `1`, zero
and infinite bases give the `pow` results). `f64` arrays have none of these functions.

```rust
use luma::*;

let t = Array::from_slice(&[Df32::from(300.0), Df32::from(2000.0)], 2)?;
// Sutherland-type T^1.5 and an SST blending function, kept in Df32.
let power = t.powf(Df32::from(1.5)).to_vec()?;
assert!((power[0].to_f64() / 300f64.powf(1.5) - 1.0).abs() < 1e-13);
let blend = (&t * Df32::from(0.001)).tanh().to_vec()?;
assert!((blend[1].to_f64() / 2f64.tanh() - 1.0).abs() < 1e-13);
let back = (t.ln()).exp().to_vec()?;
assert!((back[1].to_f64() / 2000.0 - 1.0).abs() < 1e-13);
# Ok::<(), luma::LumaError>(())
```

**Cost.** One `Df32` add or multiply costs `df32_factor` `f32` operations: 27 on the RTX 3090
and 44 on the integrated GPU the model was measured on, 2.3 on llvmpipe. The
[`CostModel`](https://docs.rs/luma/latest/luma/struct.CostModel.html) of the device records the
measured factor and the fusion decisions use it. A higher factor makes recomputing a `Df32` expression at several offsets more expensive in the
model's estimate, which weighs it against staging and materialising it
([Performance](performance.md#the-cost-model)).

**Fused multiply-add.** The product of two `Df32` takes the rounding error of the `f32` product from
`fma`. Where the device's `fma` is a real fused multiply-add (checked by a small kernel when a
device first opens) this is cheaper than splitting both factors (Dekker splitting takes about ten times as many
operations for the error of the product). `LUMA_DF_FMA`
and `ContextOptions::df_fma` force either way; the results are the same, only the speed differs.

## Casts

```rust
use luma::*;

let x = Array::from_slice(&[0.1f32, 0.2], 2)?;
let d = x.to_df32();               // exact, `lo` is zero
let back = d.to_f32();             // rounds
assert_eq!(back.to_vec()?, vec![0.1, 0.2]);
# Ok::<(), luma::LumaError>(())
```

| From | To | Methods |
| --- | --- | --- |
| `f32` | `f64`, `Df32` | `to_f64()` (needs `SHADER_F64`), `to_df32()` |
| `f64` | `f32`, `Df32` | `to_f32()`, `to_df32()` |
| `Df32` | `f32`, `f64` | `to_f32()`, `to_f64()` (needs `SHADER_F64`) |

Casts are lazy and fuse; they are exact where the target type holds the value. Constants and
generators are `f32`: `Array::<f32>::linspace(..).to_df32()` has only `f32` precision, so for a
wide initial condition build the host data in `f64` and upload with `Df32::from(f64)`.

## Compensated sums

A plain `f32` sum, even with Luma's tree order, has an error that grows with the length. For
residual norms and regression tests use the compensated forms, which accumulate in `Df32` and round
to `f32` at the end:

```rust
use luma::*;

let x = Array::<f32>::full(1 << 20, 0.1)?;
let exact = f64::from(0.1f32) * (1 << 20) as f64;
let sum = x.sum_compensated().item()?;      // the exact sum rounded to f32
assert!((f64::from(sum) - exact).abs() < 0.01);

let norm = x.norm_compensated().item()?;    // squares accumulated in Df32, too
assert!((f64::from(norm) - (exact * f64::from(0.1f32)).sqrt()).abs() < 1e-3);
# Ok::<(), luma::LumaError>(())
```

They read the array once and cost several times the plain sum's arithmetic. The order of the reduction tree can change the bits below `f32`
precision, so the result is the correctly rounded sum except where the terms cancel almost
completely. To keep the extra bits, reduce a `Df32` array yourself: `x.to_df32().sum()`.

## Non-finite values

`f32`, `f64` and `Df32` arrays have lazy `bool` arrays `is_finite()`, `is_nan()` and `is_inf()`.
They test the bits of the exponent, so they do not depend on how the device compares NaN (a shader
compiler may assume NaN does not exist and fold `x != x`). They fuse like any elementwise
operation:

```rust
use luma::*;

let x = Array::from_slice(&[1.0f32, f32::INFINITY, f32::NAN, -2.0], 4)?;
assert_eq!(x.is_finite().to_u32().to_vec()?, vec![1, 0, 0, 1]);
assert_eq!(x.is_nan().to_u32().to_vec()?, vec![0, 0, 1, 0]);
assert_eq!(x.is_inf().to_u32().to_vec()?, vec![0, 1, 0, 0]);

// Sanitise in one fused kernel.
let clean = select(&x.is_finite(), &x, 0.0);
assert_eq!(clean.to_vec()?, vec![1.0, 0.0, 0.0, -2.0]);
# Ok::<(), luma::LumaError>(())
```

To find where a NaN starts, open the context in debug mode with `check_finite`: every evaluated
`f32` and `Df32` array is checked (`f64` is not) and the first operation whose result holds a NaN
or Inf fails with
`LumaError::NonFinite`, naming its source line. It waits for the GPU after every array; for
debugging only.

```rust
use luma::*;

let ctx = Context::with_options(ContextOptions::default().check_finite(true))?;
let x = Array::from_slice_on(&ctx, &[1.0f32, 2.0], 2)?;
let y = &x / 0.0;
let err = y.to_vec().unwrap_err();
assert!(matches!(err.root(), LumaError::NonFinite));   // the message names this line
# Ok::<(), luma::LumaError>(())
```

## Fast math

By default Luma rewrites expressions only where the result is bit-identical (`x * 1`, `-(-x)`,
`min(x, x)`, `abs(abs(x))`, `select` with two equal branches, `x * 0` on integers, ...). `fast_math` (a
[`ContextOptions`](https://docs.rs/luma/latest/luma/struct.ContextOptions.html) field,
`LUMA_FAST_MATH=1`) also allows rewrites that change values, which can make a kernel shorter:

| Rewrite | Changes |
| --- | --- |
| `x + 0` becomes `x` | `-0.0 + 0.0` is `0.0`, the rewrite keeps `-0.0` |
| `x * 0` becomes `0`, `x - x` becomes `0` | wrong for NaN and infinity |
| `x / x` becomes `1` | wrong for 0, NaN and infinity |
| `x <= x`, `x >= x`, `x == x` become true; `x != x` false | wrong for NaN |
| `(x * a) * b` and `(x + a) + b` combine their constants (`f32` and `f64` only) | different rounding |
| `x / a` for a constant `a` becomes `x * (1 / a)` (`f32` and `f64` only) | different rounding |
| `powf(x, e)` with `e` of 1, 2 or 0.5 becomes `x`, `x * x`, `sqrt(x)` | different rounding |
| `Df32` adds and subtracts use the cheaper algorithm | the relative error grows with the cancellation (below `2^-47` of the larger operand, instead of `2^-46` of the result); reductions are excepted |

```rust
use luma::*;

let ctx = Context::with_options(ContextOptions::default().fast_math(true))?;
let x = Array::from_slice_on(&ctx, &[0.0f32, f32::INFINITY, 3.0], 3)?;
assert_eq!((&x / &x).to_vec()?, vec![1.0, 1.0, 1.0]);   // x / x is 1, NaNs ignored
assert_eq!((&x * 0.0).to_vec()?, vec![0.0, 0.0, 0.0]);
# Ok::<(), luma::LumaError>(())
```

Do not turn it on if you rely on NaN or infinity propagating, or on the exact bits of a
reassociated expression. With it off (the default), a rewrite that would change any bit is not
applied.
