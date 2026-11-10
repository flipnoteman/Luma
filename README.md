# Luma

GPU arrays for Rust that feel like [ArrayFire](https://arrayfire.com): write math with operators
and methods, run it on any GPU (Vulkan, Metal, DX12) through [wgpu](https://wgpu.rs), and drop
down to your own WGSL kernels when you need to. WIP.

```rs
use luma::*;

fn main() -> Result<()> {
    // Arrays live on the GPU. The first one opens the default device.
    let x = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], 4)?;
    let y = Array::from_slice(&[10.0f32, 20.0, 30.0, 40.0], 4)?;

    // Operators and math methods are recorded, not run one by one ...
    let z = (&x * 2.0 + &y).sqrt();

    // ... until the host needs a result.
    println!("{:?}", z.to_vec()?);          // [3.46.., 4.89.., 6.0, 6.92..]
    println!("{}", (&x * &y).sum()?);       // 300
    Ok(())
}
```

## Arrays and shapes

`Array<T>` holds up to four dimensions, the first one varying fastest. Anything shape-like
converts into a `Shape`: `4`, `[2, 3]`, `[2, 3, 4]`, `[2, 3, 4, 5]`.

```rs
let a = Array::from_slice(&data, [640, 480])?;    // from the host
let b = Array::<f32>::zeros([640, 480])?;
let c = Array::full([640, 480], 0.5f32)?;
let v = a.reshape(640 * 480);                      // metadata only: shares the buffer
```

Every constructor has a form for an explicit device, `Array::from_slice_on(&ctx, ..)`,
`zeros_on`, `full_on`; the plain forms use `luma::default_context()`, a lazily opened global
`Context`. A failure to open the GPU is reported as an `Err` by every call, not a panic.

`u32` and `f32` have the built-in math; arrays of `[f32; 2]`, `[f32; 4]`, `[u32; 2]`, `[u32; 4]`
(WGSL `vec2`/`vec4`) can be created, read back and passed to custom kernels, and `[f32; 2]` /
`[f32; 4]` arrays have per-component reductions (see below).

## Math

```rs
// Operators: + - * / on arrays and scalars (scalar on either side), unary -, and += -= *= /=.
let mut u = &a * 2.0 - 1.0 / &b;
u += &c;
u *= 0.5;

// f32 functions
let w = u.sqrt() + u.exp() * u.sin().abs();
let clipped = w.clamp(0.0, 1.0).powf(2.2).minimum(&c);

// Comparisons give 0/1 masks (Array<u32>); select picks by mask: here, relu.
let relu = select(&u.gt(0.0), &u, &Array::zeros(u.shape())?);

// Circular shifts, like af::shift: a three-point stencil with periodic boundaries.
let smooth = (u.shift(1) + &u + u.shift(-1)) / 3.0;

// Whole-array reductions return host values (and synchronize).
let (s, lo, hi) = (u.sum()?, u.min()?, u.max()?);
let (mean, norm) = (u.mean()?, u.norm()?);
// Several at once read the data once.
let [s, lo, hi] = u.reduce([Reduction::Sum, Reduction::Min, Reduction::Max])?;

// Several results of the same inputs, computed in one kernel (inputs read once).
let ratio = (&spot / &strike).ln();
let (up, down) = (&ratio * 2.0, &ratio * -2.0);
eval_all(&[&up, &down])?;

// Time stepping: a stencil step runs several steps per pass over memory.
let heat = u0.iterate(100, |u| u + (u.shift([1, 0]) + u.shift([-1, 0]) + u.shift([0, 1]) + u.shift([0, -1]) - u * 4.0) * 0.2);

// 1D convolutions
let filtered = signal.conv1d(&kernel, 1, 0);
let full = signal.conv1d_fft(&kernel);
```

Operands need the same shape, or be a scalar; there is no general broadcasting yet. Float sums use
a workgroup tree reduction, so summing ten million elements is accurate to a few `1e-7`
relative.

Arrays of `[f32; 2]` and `[f32; 4]` reduce per component, with the same tree reduction (and so the
same accuracy); the result has one value per component:

```rs
let state = Array::<[f32; 4]>::zeros([1000, 500])?;
let total: [f32; 4] = state.sum()?;
let (lo, hi) = (state.min()?, state.max()?);      // [f32; 4] each
let l2: [f32; 4] = state.norm()?;                 // sqrt of the sum of squares, per component
```

## Nothing waits until you ask

Operations are recorded into the context's pending command buffer and submitted together: when
you read a result (`to_vec`, `eval`, a reduction, `Context::sync`) and whenever a few hundred
dispatches piled up. Hundreds of small operations cost one submission, not hundreds
(`cargo run --release --example bench_batching`). `ctx.flush()` (or `luma::flush()` for the default
context) submits what is pending *without waiting*, so the GPU starts while the host keeps
recording: flush after each chunk of a time-stepping loop and read results back only once in a
while. Live lazy arrays (see below) count as recorded work: `flush` evaluates them first, each as
one fused kernel, so the GPU starts on elementwise chains too; arrays already computed inside
another fused kernel (the temporaries of `(&x * 2.0 + &y).sqrt().eval()`) are not computed again.
Only `sync` and the readbacks wait. Host uploads, `write`, and the values of
uniforms are recorded in order with the work around them, so reusing or overwriting an array never
affects operations recorded before it.

## Elementwise chains are fused

Operators, math functions, comparisons, `select`, `powf`, `clamp` and `shift` do not run when you
write them: they build an expression, and the array you get back is *lazy*. When something needs
its data (`to_vec`, `eval`, `write`, a convolution, a custom kernel, `Context::sync`), the
whole expression is compiled into **one** generated kernel that reads every input once and writes
the result once, so `(&x * 2.0 + &y).sqrt() - 1.0` is a single dispatch with no temporaries.
A shift evaluates the array it shifts and becomes a shifted read in the kernel that uses it, so a
stencil step `&u + (u.shift(1) + u.shift(-1) - &u * 2.0) * 0.1` reads `u` once.
Reductions fold the expression into their first pass: `(&a * &b).sum()?` never materializes the
product (the `[f32; 2]`/`[f32; 4]` arrays above are never lazy; they reduce a buffer). Kernels are cached by the structure of the expression, so changing a scalar (`&x * dt`
with a new `dt` every step) reuses the pipeline.

A lazy array is the value its expression had when you wrote it down: overwriting an input
afterwards (`write`, `+=`, a kernel taking `&mut`) does not change it. Errors still arise where
the faulty operation is written, with its source location. Fusion stops at 128 operations per
kernel or at the device's binding limit (the expression is cut and continues in the next kernel),
and a result used by several operations is computed once instead of inside each of them.
`LUMA_FUSION_MAX_NODES=1` in the environment disables fusion (one kernel per operation); the
effect is measured by `cargo run --release --example bench_fusion`.

## Kernel parameters per GPU

Workgroup sizes and tile shapes of the built-in kernels (elementwise, reductions, tiled `conv1d`,
time-blocked `iterate`) are chosen when the context opens its device: Apple GPUs get the values
measured on an M5 Pro, NVIDIA GPUs those measured on an RTX 3090, every other GPU common defaults
that have not been measured yet. All are shrunk to fit the device's limits. `ctx.tuning_profile()`
reports the choice, and `LUMA_TUNING=apple`, `LUMA_TUNING=nvidia` or `LUMA_TUNING=generic` in the
environment forces one.

## Errors do not panic

Operators and math methods return `Array<T>`, not `Result`. If an operation fails (shape
mismatch, arrays from different devices, out of memory, ...) the result is a *poisoned* array that
carries the error and the source location of the failed operation, and every later operation on
it is poisoned too. The error comes out as `Err` at the next sync point, or earlier when you ask:

```rs
let a = Array::from_slice(&[1.0f32, 2.0, 3.0], 3)?;
let b = Array::from_slice(&[1.0f32, 2.0], 2)?;

let c = (&a + &b) * 2.0;   // line 5: shape mismatch, `c` is poisoned
let d = c.sqrt() - 1.0;    // still poisoned, still pointing at line 5

assert!(d.error().is_some());  // early detection (also: d.check())
println!("{}", d.to_vec().unwrap_err());
// `add` failed at src/main.rs:5:14: Dimension mismatch: [3, 1, 1, 1] vs [2, 1, 1, 1]
```

Use `try_add`, `try_sub`, `try_mul`, `try_div` for an immediate `Result`. `err.root()` is the
underlying `LumaError`, `err.location()` the call site. Errors the GPU itself reports while
running a batch (validation, out of memory) are returned the same way from the next sync point.

## Custom kernels

When the built-ins are not enough, write WGSL and run it on your arrays, in the same batch as
everything else. Luma binds a `luma` grid uniform (`luma.size`, the problem size) in `@group(0)`
and provides `luma_idx2(i, j)` / `luma_idx3(i, j, k)` (x fastest). Your bindings go in `@group(1)`,
numbered from 0 in argument order: `&Array` binds `var<storage, read>`, `&mut Array` binds
`var<storage, read_write>` and `uniform(&value)` binds a `var<uniform>` (laid out by
[encase](https://crates.io/crates/encase), so `#[derive(encase::ShaderType)]` structs match
WGSL's padding rules). Arguments are checked against the shader at launch.

```rs
use luma::*;

#[derive(encase::ShaderType)]
struct Params { alpha: f32, n: u32 }

const SAXPY: &str = r#"
struct Params { alpha: f32, n: u32 }
@group(1) @binding(0) var<storage, read> x: array<f32>;
@group(1) @binding(1) var<storage, read_write> y: array<f32>;
@group(1) @binding(2) var<uniform> p: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= p.n) { return; }
    y[gid.x] = p.alpha * x[gid.x] + y[gid.x];
}
"#;

fn main() -> Result<()> {
    let x = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], 4)?;
    let mut y = Array::from_slice(&[10.0f32; 4], 4)?;

    let saxpy = Kernel::new(SAXPY)?;   // or ctx.kernel(SAXPY).entry("main").build()?
    saxpy.launch([4], args![&x, &mut y, uniform(&Params { alpha: 2.0, n: 4 })])?;

    let y = y * 0.5;                   // operators and launches share one ordered batch
    println!("{:?}", y.to_vec()?);      // [6.0, 7.0, 8.0, 9.0]
    Ok(())
}
```

Only the globals an entry point uses (directly or through the functions it calls) are its
arguments, so one WGSL module can hold several entry points with different bindings: build each
with `ctx.kernel(src).entry("name")`. The bindings used by one entry point are numbered
contiguously from 0 in `@group(1)`; other globals may be numbered, grouped or typed freely.

`args![..]` builds a stack array (no allocation), uniforms up to 128 bytes are stored inline, and
launches are cheap: the launches of a batch are recorded into one compute pass, bind groups are
cached, and what is already bound is not bound again. On the development machine (RTX 3090,
Vulkan) a launch of a cached kernel costs 0.6 – 0.9 microseconds of host time, down from 3.6 – 4.0
(`cargo run --release --example bench_launch`).

Arrays that share a buffer (through `reshape`) cannot be bound to one launch if one binding is
writable; that is an error (`KernelArgAlias`), as is an in-place operation on a view of its own
operand.

## GPU timing

Where the adapter supports timestamp queries, `ctx.profile(|| { ... })` returns the GPU time of
every kernel launched inside the closure, by label (a custom kernel's `.label(..)`, default its
entry point; built-in operations by operation and element type; fused kernels by the operations
they contain, e.g. `Fused(Mul, Add, Sqrt) f32`):

```rs
let spin = ctx.kernel(SPIN).label("spin").build()?;
let (c, profile) = ctx.profile(|| {
    for _ in 0..10 {
        spin.launch([n], args![&mut a, uniform(&Params { rounds: 20_000 })]).unwrap();
    }
    &b * 3.0
})?;
println!("{profile}");
// kernel                        launches     total us      mean us       min us       max us
// spin                                10     12137.63      1213.76      1211.94      1214.66
// Fused(Mul) f32                       1         3.74         3.74         3.74         3.74
// (a run of this snippet's kernel on an RTX 3090)
```

While profiling, every launch runs in a pass of its own (so the timings are those of single
launches, and host cost is higher). `ctx.enable_profiling()` / `ctx.finish_profiling()` bracket
manual sections, and `ctx.profiling_supported()` says whether the GPU can do it at all; without
timestamp queries these return `LumaError::Profiling` and everything else keeps working.

## Examples

```sh
cargo run --example double_test                 # expressions, select, reductions
cargo run --example conv1d
cargo run --release --example bench_conv1d      # spatial vs FFT convolution
cargo run --release --example bench_perf -- 67108864   # GPU throughput of large-array workloads
cargo run --release --example bench_vs_cpu     # the same workloads against every CPU core
cargo run --release --example bench_batching    # sync after every op vs batched
cargo run --release --example bench_launch      # host cost per kernel launch, flush overlap
cargo run --release --example bench_fusion      # one kernel per operation vs fused chains
```

## Benchmarks against the CPU

Criterion benchmarks of application workloads, each run with Luma and with a multi-threaded
CPU implementation of the same algorithm (vectorized loops; Accelerate for matrix products on
macOS, `rustfft` for FFT filtering). Every group checks first that both give the same result.

```sh
RUSTFLAGS="-C target-cpu=native" cargo bench --bench real_world    # Black-Scholes, Mandelbrot, n-body, SGEMM, 2-D heat, FIR
RUSTFLAGS="-C target-cpu=native" cargo bench --bench ml            # k-means, MLP forward pass, similarity matrix
RUSTFLAGS="-C target-cpu=native" cargo bench --bench monte_carlo   # option pricing, integration, Ising model
RUSTFLAGS="-C target-cpu=native" cargo bench --bench imaging       # blurs, 2-D convolution, bilateral filter, color pipeline, histogram
```
