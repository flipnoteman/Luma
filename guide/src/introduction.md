# Introduction

Luma is an array library for Rust that runs on the GPU. You write ordinary arithmetic on `Array`s;
Luma records it, fuses chains of elementwise operations into a single kernel, and runs the result
on whatever hardware you have (Vulkan, Metal or DX12, through [wgpu](https://wgpu.rs)). If no GPU
opens it falls back to Mesa's llvmpipe software driver.

It is meant for numerical code where the interesting part is the maths: stencils, finite-volume
solvers, Monte Carlo, bit-sliced cryptography, small machine-learning kernels. You do not write
shaders, manage buffers or schedule submissions. When you need something Luma does not have, a
[custom WGSL kernel](custom-kernels.md) runs in the same batch as the built-in operations.

This guide explains how to use the library and why it behaves as it does. The
[API reference](https://docs.rs/luma) lists every method; the guide shows how they fit together.

## Install

You need a recent stable Rust toolchain and a GPU with Vulkan, Metal or DX12 drivers. Add Luma
to a project:

```toml
[dependencies]
luma = "0.3"
```

Without a GPU, install a software Vulkan driver (`mesa-vulkan-drivers` on most Linux
distributions). Luma picks it up automatically when no GPU opens, and `LUMA_DEVICE=cpu` forces
it. See the [FAQ](faq.md#choosing-a-device).

## A first program

```rust
use luma::*;

fn main() -> Result<()> {
    // Arrays live on the GPU. The first one opens the default device.
    let x = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], 4)?;
    let y = Array::from_slice(&[10.0f32, 20.0, 30.0, 40.0], 4)?;

    // Operators and math methods are recorded, not run.
    let z = (&x * 2.0 + &y).sqrt();

    // The host asks for a result: the batch is compiled, run, and read back.
    let z = z.to_vec()?;
    assert_eq!(z[2], 6.0);

    // Reductions are lazy one-element arrays; `item` reads the value.
    assert_eq!((&x * &y).sum().item()?, 300.0);
    Ok(())
}
```

Three things to notice:

- `&x * 2.0 + &y` does not compute anything. It builds an expression. The multiply, the add and
  the square root become one kernel when something needs the values
  ([Lazy evaluation and fusion](lazy.md)).
- Operators take references (`&x`) so the array stays usable; scalars (`2.0`) are broadcast.
- Errors do not panic. A failing operation returns a poisoned array, and the error comes out as
  `Err` at the next call that reads or evaluates it (`to_vec`, `item`, `eval`), with the source
  location of the operation that failed.

## Where to go next

- [Arrays](arrays.md): creation, element types, shapes, broadcasting, reading back.
- [Lazy evaluation and fusion](lazy.md): when kernels run, `eval`, `flush`, `sync`, `cache`.
- [Worked examples](examples.md): heat equation, Euler solver, ML-KEM, CPA.
- [Environment variables](environment.md) and the [FAQ](faq.md) when something does not behave.

The source repository, with the runnable programs referred to throughout the guide, is at
[github.com/flipnoteman/Luma](https://github.com/flipnoteman/Luma). Every code block in this guide
is compiled and run by `cargo test --doc`, so what you read here works with the version of Luma
the page belongs to (see the version selector in the menu bar).
