# FAQ and troubleshooting

## Choosing a device

**Which device does Luma use?** `Context::new()` and the default context open CUDA, then Metal, then
the high-performance wgpu adapter (a discrete GPU over an integrated one). If no GPU opens, Luma falls
back to the native `cpu` backend. Software Vulkan drivers (Mesa llvmpipe) are never used.
`LUMA_BACKEND` / `ContextOptions::backend` (or `Context::cuda()`, `metal()`, `wgpu()`, `cpu()`) pick one backend, `LUMA_DEVICE` or `ContextOptions::device` narrow the kind of device, and `Context::backend()` and `Context::device()` say what opened:

| Setting | Effect |
| --- | --- |
| `LUMA_DEVICE=gpu` / `DeviceKind::Gpu` | a hardware GPU only; `LumaError::NoDevice` or `NoAdapter` if none exists |
| `LUMA_DEVICE=cpu` / `DeviceKind::Cpu` / `Context::cpu()` | the native `cpu` backend, whatever GPUs exist |
| unset / `DeviceKind::Auto` | GPU, the `cpu` backend as fallback |

```rust
use luma::*;

match Context::cpu() {
    Ok(cpu) => {
        assert!(cpu.is_cpu());
        let x = Array::from_slice_on(&cpu, &[1.0f32, 2.0], 2)?;
        assert_eq!((&x + &x).to_vec()?, vec![2.0, 4.0]);
    }
    Err(LumaError::NoAdapter(_) | LumaError::NoDevice(_)) => {
        eprintln!("The cpu backend is not built");
    }
    Err(error) => return Err(error),
}
# Ok::<(), luma::LumaError>(())
```

**Opening a context returns `LumaError::NoAdapter` ("Found no adapters").** With `LUMA_DEVICE=gpu`, no hardware GPU driver (CUDA, Metal, Vulkan or DX12) was found. Install your GPU vendor's driver, or leave `LUMA_DEVICE` unset: every build includes the native SIMD CPU backend and falls back to it.

**I have two GPUs and want the other one.** Luma asks for the high-performance adapter and has no
adapter-by-name setting. On Linux the Vulkan loader decides which drivers wgpu sees, so pointing
`VK_ICD_FILENAMES` at one driver's `.json` file (for example
`/usr/share/vulkan/icd.d/radeon_icd.x86_64.json`) selects that driver. Note the RADV issue below.

**Arrays of different contexts.** Every `Array` belongs to one `Context`; mixing arrays of two
contexts is an error. The `_on(&ctx, ..)` constructors take a context explicitly, and the
`default_context()` is shared by every constructor without it.

## NVIDIA

**The process hangs or aborts when it exits.** (wgpu backend, Linux.) NVIDIA's Linux driver can crash or hang in its own
shader-compiler threads during `exit`. Luma waits at exit for the GPU (up to 2 s) and for busy
Vulkan driver threads (up to 2 s), and calls `_exit` only if they are still busy after that. That makes the problem
rare, not impossible (driver 615.x was seen to still do it). `LUMA_EXIT_WAIT=0` skips the wait,
which saves about 60 ms of exit time and risks the hang. A normal `main` return goes through
this path; calling `std::process::exit` yourself does too.

**Opening a second device stalls.** On NVIDIA's Vulkan driver, creating a device (`Context::new`) blocks
submissions of *other* devices in the process until their in-flight GPU work finishes. This
happens once per device creation. Create your contexts up front, before starting long GPU work,
or use one context (the default one is shared by threads).

**Opening a context fails with `RequestDevice("Parent device is lost")`, though no device was lost.**
(wgpu backend on NVIDIA.) A process opens at most 63 or 64 Vulkan devices, however many instances it has:
past that limit `vkCreateDevice` fails with `VK_ERROR_INITIALIZATION_FAILED`, which wgpu words that way.
A context frees its device with its last handle (the arrays, kernels and caches of a context do not keep
it alive, 0.5.2), and a `Context::new` on the wgpu backend that meets the limit waits up to 60 s for another
context to give its device back and takes it. A program or a test suite that holds more than 63 contexts
alive at the same time still fails: drop contexts you no longer use, share one context between threads, or
run the tests with fewer threads per process.

## RADV integrated GPUs give wrong results

Kernels that use workgroup memory are wrong on RADV integrated GPUs: scans (`scan`, `cumsum`,
axis reductions), `matmul`, sorts (and what builds on them), scatters into targets of up to 4096
elements, workgroup staging of views, and the cyclic-reduction tridiagonal solver. The
cyclic solver is chosen automatically only on the NVIDIA and Apple tuning profiles (a GPU detected as such,
or `LUMA_TUNING=nvidia` / `apple`); elsewhere (and with `LUMA_TRIDIAGONAL=thomas`) the solver does not use it. If you see wrong results on such a GPU:

- `LUMA_STAGING=off` keeps expressions read through views out of workgroup memory (the cost model
  then recomputes them or evaluates them first),
- `LUMA_TRIDIAGONAL=thomas` pins the solver that does not use workgroup memory,
- scans, axis reductions, `matmul`, sorts and scatters have no switch: run on another device (`LUMA_DEVICE=cpu`, or a
  different GPU through `VK_ICD_FILENAMES`).

## Results and errors

**An operation "did nothing".** Operators do not panic or return `Result`; they return a poisoned
array and the error comes out at `to_vec`, `item`, `eval` or `check()`. Call `check()` on an
array to find the first failure early. The error carries the source location of the operation.

**I get `NaN` or `Inf` and do not know where it starts.** Set `LUMA_CHECK_FINITE=1` (or
`ContextOptions::check_finite`): every evaluated `f32` and `Df32` array is checked (`f64` is not) and
the first operation whose result holds a non-finite value fails with `LumaError::NonFinite`. It waits for the GPU after every
array, so use it for debugging only. For `f32`, `f64` and `Df32` arrays,
`is_finite()` / `is_nan()` / `is_inf()` give lazy `bool` arrays you can use in expressions.

**`Array<f64>` fails with `Unsupported`.** The device has no 64-bit shader floats. Check
`ctx.features().contains(luma::Features::SHADER_F64)`, or use `Df32`
([Precision](precision.md)), which runs everywhere.

**Out of memory.** Array data beyond the memory budget (80% of the device's memory where the backend reports it: CUDA, Metal and Vulkan GPUs; none on the CPU backend)
moves to host memory and back on demand, with a one-time warning. `LUMA_MEMORY_BUDGET` sets the
budget. Custom kernels need arrays that fit one buffer (`LumaError::Chunked` otherwise).

## Numbers

**`put_add`, `group_sum` or `histogram` of `f32` gives slightly different sums on every run.**
Equal indices are accumulated with atomics in no fixed order, and float addition is not
associative, so the last bits of an `f32` sum differ between runs (and devices). Integer sums
(`u32`, `i32`, `bincount`, `group_count`) are exact, and `put_min` / `put_max` do not depend on the
order. If you need a reproducible float sum, scatter into integers (fixed point) or sort first and
reduce ([Accumulating scatters](data.md#accumulating-scatters)).

**Where do NaNs go in a sort or a quantile?** `sort`, `argsort` and `sort_axis` order floats as
`-inf < .. < -0.0 < +0.0 < .. < +inf < NaN`: NaNs are last (first in the descending forms), and
`-0.0` sorts before `+0.0`. A quantile is read from the sorted array, so a NaN in the data makes the
high quantiles NaN and leaves the low ones alone; NumPy returns NaN for every quantile in that case.
`factorize` and `unique` treat all NaNs as one value (the last) and `-0.0` and `0.0` as one. In
`histogram` a NaN goes to bin 0.

**`x.ne(&x)` does not find the NaNs.** Comparisons with a NaN are the GPU's. `lt`, `le`, `gt`,
`ge` and `eq` give false for a NaN operand, and `ne` gave false as well on NVIDIA and on
llvmpipe (the tests that ran), where IEEE says true. Use `x.is_nan()`, and `select(&x.is_nan(), a, b)`
to replace them. `LUMA_FAST_MATH` additionally folds a comparison of an array with itself.

**What does a float-to-integer cast give out of range?** `to_u32` and `to_i32` truncate toward
zero and saturate at the largest `f32` inside the range, and NaN becomes 0:

| Cast | Value | Result |
| --- | --- | --- |
| `to_u32` | negative, `-0.5`, NaN | 0 |
| `to_u32` | `1e20`, `+inf` | `4294967040` |
| `to_i32` | `-1e20`, `-inf` | `-2147483648` |
| `to_i32` | `1e20`, `+inf` | `2147483520` |

```rust
use luma::*;

let x = Array::from_slice(&[-1.5f32, f32::NAN, 1e20, -1e20, f32::INFINITY], 5)?;
assert_eq!(x.to_u32().to_vec()?, vec![0, 0, 4_294_967_040, 0, 4_294_967_040]);
assert_eq!(x.to_i32().to_vec()?, vec![-1, 0, 2_147_483_520, i32::MIN, 2_147_483_520]);
# Ok::<(), luma::LumaError>(())
```

**Integer division by zero** gives the dividend (`x / 0 == x`) and nothing fails; for `i32`,
`x % 0` is 0 and `i32::MIN / -1` is `i32::MIN` ([Integers](integers.md#signed-integers-and-casts)).

**Which matrix shape is `matmul`'s?** Axis 0 varies fastest, so the shapes are NumPy's
`a @ b` reversed: `lhs` `[K, M]`, `rhs` `[N, K]`, result `[N, M]`
([Matrix products](data.md#matrix-products)).

**`matmul` differs from the exact `f32` product in the last digits (or by 1e-3 relative to the size of
its terms).** On a CUDA device of compute capability 8.0 or higher (Ampere and later) `matmul` runs on tensor cores in
TF32 by default: the operands are rounded to 10 bits of mantissa and the products accumulated in `f32`, so
each entry is within 1.5e-3 of the sum of the absolute values of its terms of the `f32` product (the
precision of PyTorch's `allow_tf32`). `a.matmul_f32(&b)` is the exact product of one call;
`ContextOptions::tf32_matmul(false)` or `LUMA_TF32_MATMUL=0` makes every `matmul` of a context exact, and
`Context::tf32_matmul()` says which a context does. `matmul_batched` and the other backends (wgpu, Metal,
the CPU) compute `f32` products, so results also differ between a CUDA run and one on another backend
([Tensor cores](data.md#tensor-cores-tf32)). The dense linear algebra (`lu`, `qr`, `cholesky`, `svd`, ...)
uses `f32` products only.

## Performance

**The first run is slow, later ones are fast.** Large kernels (especially `Df32`) take seconds
to compile on some drivers. Luma keeps the driver's compiled pipelines (wgpu), the CUDA PTX, the CPU backend's machine code and the measured cost
model in `$XDG_CACHE_HOME/luma` (`~/.cache/luma`), so the second process skips that. Set
`LUMA_CACHE=off` to turn it off or `LUMA_CACHE_DIR` to move it. The compiled CPU code and the CUDA PTX are
keyed by a hash of the sources that generate them, so the first run after an upgrade (or after any change
to the code generators) compiles everything again and then caches it; a stale kernel of an older build is
never loaded. The wgpu pipeline cache files have a format number (2 since 0.5.2): files of an older format
are read as absent and rewritten once.

**Context creation takes about 150 ms.** That is the device calibration (once per machine; the
result is cached). `LUMA_CALIBRATE=0` skips it and uses fixed numbers for the GPU kind.

**My loop is slow.** See the [Performance guide](performance.md) and [Loops](loops.md): use
`iterate`/`iterate_all` where they apply, `eval` each step instead of reading, and look at
`ctx.profile(..)` for kernel counts and split reasons.
