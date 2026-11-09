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
(WGSL `vec2`/`vec4`) can be created, read back and passed to custom kernels.

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

// 1D convolutions
let filtered = signal.conv1d(&kernel, 1, 0);
let full = signal.conv1d_fft(&kernel);
```

Operands need the same shape, or be a scalar; there is no general broadcasting yet. Float sums use
a workgroup tree reduction, so summing ten million elements is accurate to a few `1e-7`
relative.

## Nothing waits until you ask

Operations are recorded into the context's pending command buffer and submitted together: when
you read a result (`to_vec`, `eval`, a reduction, `Context::sync`) and whenever a few hundred
dispatches piled up. Hundreds of small operations cost one submission, not hundreds
(`cargo run --release --example bench_batching`). Host uploads, `write`, and the values of
uniforms are recorded in order with the work around them, so reusing or overwriting an array never
affects operations recorded before it.

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

Arrays that share a buffer (through `reshape`) cannot be bound to one launch if one binding is
writable; that is an error (`KernelArgAlias`), as is an in-place operation on a view of its own
operand.

## Examples

```sh
cargo run --example double_test                 # expressions, select, reductions
cargo run --example conv1d
cargo run --release --example bench_conv1d      # spatial vs FFT convolution
cargo run --release --example bench_batching    # sync after every op vs batched
```
