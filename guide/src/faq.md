# FAQ and troubleshooting

## Choosing a device

**Which device does Luma use?** `Context::new()` and the default context ask wgpu for the
high-performance adapter (a discrete GPU over an integrated one). If no GPU opens, Luma falls back
to a software Vulkan driver (Mesa llvmpipe). `LUMA_DEVICE` or `ContextOptions::device` narrow this:

| Setting | Effect |
| --- | --- |
| `LUMA_DEVICE=gpu` / `DeviceKind::Gpu` | a hardware GPU only; `LumaError::NoDevice` if only a software adapter exists |
| `LUMA_DEVICE=cpu` / `DeviceKind::Cpu` / `Context::cpu()` | the software adapter, whatever GPUs exist |
| unset / `DeviceKind::Auto` | GPU, CPU as fallback |

```rust
use luma::*;

let cpu = Context::cpu()?;
assert!(cpu.is_cpu());
let x = Array::from_slice_on(&cpu, &[1.0f32, 2.0], 2)?;
assert_eq!((&x + &x).to_vec()?, vec![2.0, 4.0]);
# Ok::<(), luma::LumaError>(())
```

**Opening a context returns "No suitable adapter".** There is no Vulkan, Metal or DX12 driver
Luma can use. On Linux install `mesa-vulkan-drivers` (lavapipe) for the CPU fallback, or your GPU
vendor's Vulkan driver. CI runners need the former.

**I have two GPUs and want the other one.** Luma asks for the high-performance adapter and has no
adapter-by-name setting. On Linux the Vulkan loader decides which drivers wgpu sees, so pointing
`VK_ICD_FILENAMES` at one driver's `.json` file (for example
`/usr/share/vulkan/icd.d/radeon_icd.x86_64.json`) selects that driver. Note the RADV issue below.

**Arrays of different contexts.** Every `Array` belongs to one `Context`; mixing arrays of two
contexts is an error. The `_on(&ctx, ..)` constructors take a context explicitly, and the
`default_context()` is shared by every constructor without it.

## NVIDIA

**The process hangs or aborts when it exits.** NVIDIA's Linux driver can crash or hang in its own
shader-compiler threads during `exit`. Luma waits at exit for the GPU (up to 2 s) and for busy
driver compiler threads (up to 2 s), then ends the process with `_exit`. That makes the problem
rare, not impossible (driver 615.x was seen to still do it). `LUMA_EXIT_WAIT=0` skips the wait,
which saves about 50 ms of exit time and risks the hang. A normal `main` return goes through
this path; calling `std::process::exit` yourself does too.

**Opening a second device stalls.** On NVIDIA, creating a device (`Context::new`) blocks
submissions of *other* devices in the process until their in-flight GPU work finishes. This
happens once per device creation. Create your contexts up front, before starting long GPU work,
or use one context (the default one is shared by threads).

## RADV integrated GPUs give wrong results

Kernels that use workgroup memory are wrong on RADV integrated GPUs: scans (`scan`, `cumsum`,
axis reductions), workgroup staging of views, and the cyclic-reduction tridiagonal solver. The
cyclic solver is chosen automatically only on NVIDIA; elsewhere (and with
`LUMA_TRIDIAGONAL=thomas`) the solver does not use it. If you see wrong results on such a GPU:

- `LUMA_STAGING=off` keeps expressions out of workgroup memory (the kernel recomputes them),
- `LUMA_TRIDIAGONAL=thomas` pins the solver that does not use workgroup memory,
- scans and axis reductions have no switch: run on another device (`LUMA_DEVICE=cpu`, or a
  different GPU through `VK_ICD_FILENAMES`).

## Results and errors

**An operation "did nothing".** Operators do not panic or return `Result`; they return a poisoned
array and the error comes out at `to_vec`, `item`, `eval` or `check()`. Call `check()` on an
array to find the first failure early. The error carries the source location of the operation.

**I get `NaN` or `Inf` and do not know where it starts.** Set `LUMA_CHECK_FINITE=1` (or
`ContextOptions::check_finite`): every evaluated array is checked and the first operation whose
result holds a non-finite value fails with `LumaError::NonFinite`. It waits for the GPU after every
array, so use it for debugging only. For `f32`, `f64` and `Df32` arrays,
`is_finite()` / `is_nan()` / `is_inf()` give lazy `u32` masks you can use in expressions.

**`Array<f64>` fails with `Unsupported`.** The device has no 64-bit shader floats. Check
`ctx.features().contains(luma::wgpu::Features::SHADER_F64)`, or use `Df32`
([Precision](precision.md)), which runs everywhere.

**Out of memory.** Array data beyond the memory budget (80% of device memory on Vulkan GPUs)
moves to host memory and back on demand, with a one-time warning. `LUMA_MEMORY_BUDGET` sets the
budget. Custom kernels need arrays that fit one buffer (`LumaError::Chunked` otherwise).

## Performance

**The first run is slow, later ones are fast.** Large kernels (especially `Df32`) take seconds
to compile on some drivers. Luma keeps the driver's compiled pipelines and the measured cost
model in `$XDG_CACHE_HOME/luma` (`~/.cache/luma`), so the second process skips that. Set
`LUMA_CACHE=off` to turn it off or `LUMA_CACHE_DIR` to move it.

**Context creation takes about 150 ms.** That is the device calibration (once per machine; the
result is cached). `LUMA_CALIBRATE=0` skips it and uses fixed numbers for the GPU kind.

**My loop is slow.** See the [Performance guide](performance.md) and [Loops](loops.md): use
`iterate`/`iterate_all` where they apply, `eval` each step instead of reading, and look at
`ctx.profile(..)` for kernel counts and split reasons.
