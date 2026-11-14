# Custom kernels

When an algorithm does not fit array operations (a scatter with atomics, a tuned stencil, an
existing shader), write the kernel in WGSL and run it on Luma arrays. A custom kernel joins the same
ordered batch as the built-in operations, reads and writes `Array`s directly and needs no buffer
management.

## A first kernel

```rust
use luma::*;

// The uniform struct: derive `encase::ShaderType` so the layout matches WGSL's rules.
#[derive(luma::encase::ShaderType)]
struct Saxpy {
    alpha: f32,
    n: u32,
}

const SAXPY: &str = r#"
struct Saxpy { alpha: f32, n: u32 }
@group(1) @binding(0) var<storage, read> x: array<f32>;
@group(1) @binding(1) var<storage, read_write> y: array<f32>;
@group(1) @binding(2) var<uniform> p: Saxpy;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= p.n) { return; }
    y[gid.x] = p.alpha * x[gid.x] + y[gid.x];
}
"#;

let x = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], 4)?;
let mut y = Array::from_slice(&[10.0f32; 4], 4)?;

let saxpy = Kernel::new(SAXPY)?;                       // on the default context, entry `main`
saxpy.launch([4], args![&x, &mut y, uniform(&Saxpy { alpha: 2.0, n: 4 })])?;

// Operators and launches share one ordered batch; nothing has run yet.
let y = y * 0.5;
assert_eq!(y.to_vec()?, vec![6.0, 7.0, 8.0, 9.0]);
# Ok::<(), luma::LumaError>(())
```

The rules:

- **Bindings** go in `@group(1)`, numbered from 0 in the order of the arguments to `launch`.
  `&Array` binds `var<storage, read>`, `&mut Array` binds `var<storage, read_write>`, and
  `uniform(&value)` binds `var<uniform>`.
- **`@group(0)` belongs to Luma.** It holds `luma`, a uniform with `size: vec3<u32>` (the problem
  size you passed to `launch`). Luma appends two helpers to your source: `luma_idx2(i, j)` and
  `luma_idx3(i, j, k)` compute a flat index with the first dimension fastest.
- **Problem size.** `launch([x])`, `launch([x, y])` or `launch([x, y, z])` runs at least that many
  invocations: `ceil(size / workgroup_size)` workgroups per dimension. Guard against
  `global_invocation_id >= luma.size`. `launch_raw([wx, wy, wz], args)` runs exactly that many
  workgroups.
- **Uniform values** are copied when you pass them. Any type
  [encase](https://docs.rs/encase) can lay out works: a struct with `#[derive(ShaderType)]`,
  scalars, vectors, matrices, arrays. `luma::encase` is a re-export of the same version Luma uses.
- **Element sizes** are checked: a binding `array<f32>` takes arrays of 4-byte elements,
  `array<vec2<f32>>` arrays of 8-byte ones (`[f32; 2]`, `Df32`, `f64`). Only the size is compared,
  not the type: an `Array<u32>` passes for `array<f32>`, so keep the types straight yourself.

## Several entry points, 2-D launches

One module can hold several entry points; each only sees the globals it uses, and those must be
numbered contiguously from 0 *for that entry point*. Choose one with `.entry(name)`, and name it
in profiles with `.label(name)`.

```rust
use luma::*;

const GRID: &str = r#"
@group(1) @binding(0) var<storage, read_write> dst: array<f32>;
@group(1) @binding(1) var<storage, read> src: array<f32>;

// Each element's flat index, on a 2-D grid.
@compute @workgroup_size(8, 8)
fn fill_index(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= luma.size.x || gid.y >= luma.size.y) { return; }
    let i = luma_idx2(gid.x, gid.y);
    dst[i] = f32(i);
}

// dst (h x w) = transpose of src (w x h); luma.size is the shape of src.
@compute @workgroup_size(8, 8)
fn transpose(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= luma.size.x || gid.y >= luma.size.y) { return; }
    dst[gid.y + gid.x * luma.size.y] = src[luma_idx2(gid.x, gid.y)];
}
"#;

let ctx = default_context()?;
let fill = ctx.kernel(GRID).entry("fill_index").build()?;
let transpose = ctx.kernel(GRID).entry("transpose").label("my transpose").build()?;
assert_eq!(fill.bindings().len(), 1);           // only `dst`
assert_eq!(transpose.bindings().len(), 2);      // `dst` and `src`
assert_eq!(transpose.workgroup(), [8, 8, 1]);

let (w, h) = (4u32, 3u32);
let mut a = Array::<f32>::zeros([w as usize, h as usize])?;
let mut t = Array::<f32>::zeros([h as usize, w as usize])?;
fill.launch([w, h], args![&mut a])?;
transpose.launch([w, h], args![&mut t, &a])?;

assert_eq!(t.shape(), Shape::from([3, 4]));
assert_eq!(t.sum().item()?, 66.0);              // built-in reductions work on the output
assert_eq!(t.to_vec()?[1], 4.0);                // t(1, 0) = a(0, 1)
# Ok::<(), luma::LumaError>(())
```

`Kernel::new(src)` is `default_context()?.kernel(src).build()`. Building the same source twice on
a context reuses the pipeline, and a `Kernel` is cheap to clone. The shader is parsed and validated
when you build it: a mistake comes out as `LumaError::ShaderCompile` with the WGSL error and
line numbers that match your source (Luma appends its prelude, so it never shifts them).

## Mixing with built-in operations

Launches are recorded in order with the lazy operations, in one batch:

- An array that is still a lazy expression is computed before a kernel reads or writes it.
- A kernel that writes an array first lets lazy expressions recorded earlier that read it finish
  with the old contents (the same rule as `write`: [Lazy evaluation](lazy.md)).
- Lazy operations on a kernel's output fuse as usual, and reductions and views work on it.

```rust
use luma::*;

const SQUARE_PLUS: &str = r#"
@group(1) @binding(0) var<storage, read> src: array<f32>;
@group(1) @binding(1) var<storage, read_write> dst: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= luma.size.x) { return; }
    dst[gid.x] = src[gid.x] * src[gid.x] + 1.0;
}
"#;

let k = Kernel::new(SQUARE_PLUS)?;
let x = Array::from_slice(&[1.0f32, 2.0, 3.0], 3)?;
let lazy = &x * 2.0;                           // a lazy expression: computed when the kernel reads it
let mut out = Array::<f32>::zeros(3)?;
k.launch([3], args![&lazy, &mut out])?;        // [2, 4, 6]^2 + 1
let total = (&out + 1.0).sum();                // the kernel's output feeds a lazy reduction
assert_eq!(total.item()?, (4.0 + 1.0) + (16.0 + 1.0) + (36.0 + 1.0) + 3.0);
# Ok::<(), luma::LumaError>(())
```

## Ping-pong buffers

A step that reads one array and writes another is the pattern of every explicit time stepper. Keep
two arrays and swap them with `std::mem::swap`; each launch is recorded and nothing waits until
the end.

```rust
use luma::*;

#[derive(luma::encase::ShaderType)]
struct Params {
    alpha: f32,
    n: u32,
}

const HEAT: &str = r#"
struct Params { alpha: f32, n: u32 }
@group(1) @binding(0) var<storage, read> src: array<f32>;
@group(1) @binding(1) var<storage, read_write> dst: array<f32>;
@group(1) @binding(2) var<uniform> p: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p.n) { return; }
    let l = src[(i + p.n - 1u) % p.n];
    let r = src[(i + 1u) % p.n];
    dst[i] = src[i] + p.alpha * (l + r - 2.0 * src[i]);
}
"#;

let n = 9usize;
let mut data = vec![0.0f32; n];
data[4] = 1.0;
let mut a = Array::from_slice(&data, n)?;
let mut b = Array::<f32>::zeros(n)?;
let kernel = Kernel::new(HEAT)?;
let params = Params { alpha: 0.25, n: n as u32 };

for _ in 0..10 {
    kernel.launch([n as u32], args![&a, &mut b, uniform(&params)])?;
    std::mem::swap(&mut a, &mut b);              // `a` is the newest state again
}

let mut want = data.clone();
for _ in 0..10 {
    let old = want.clone();
    for i in 0..n {
        want[i] = old[i] + 0.25 * (old[(i + n - 1) % n] + old[(i + 1) % n] - 2.0 * old[i]);
    }
}
for (g, w) in a.to_vec()?.iter().zip(&want) {
    assert!((g - w).abs() < 1e-6);
}
# Ok::<(), luma::LumaError>(())
```

(For a stencil in a loop, [`iterate`](loops.md) usually beats a hand-written kernel: it runs several
steps per pass over the data.)

## Strided views

`luma::view(&array)` passes a *read view* (a slice, flip or transpose that has not been evaluated)
without copying it. It takes two consecutive bindings: the buffer and a `LumaView` uniform that
describes the strides. `LumaView` and the indexing functions `luma_view_idx2(v, i, j)`,
`luma_view_idx3`, `luma_view_idx4(v, vec4<u32>)` and `luma_view_idx(v, flat_index)` are added to
your source when it mentions `LumaView`.

```rust
use luma::*;

const VIEW_COPY: &str = r#"
@group(1) @binding(0) var<storage, read> src: array<f32>;
@group(1) @binding(1) var<uniform> src_view: LumaView;
@group(1) @binding(2) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= luma.size.x || gid.y >= luma.size.y) { return; }
    out[luma_idx2(gid.x, gid.y)] = src[luma_view_idx2(src_view, gid.x, gid.y)];
}
"#;

let a = Array::from_slice(&(0..12).map(|i| i as f32).collect::<Vec<_>>(), [4, 3])?;
let view = a.slice(0, 1..3).flip(1);           // a [2, 3] view: read in place, never copied
let mut out = Array::<f32>::zeros([2, 3])?;
Kernel::new(VIEW_COPY)?.launch([2, 3], args![luma::view(&view), &mut out])?;
assert_eq!(out.to_vec()?, vec![9.0, 10.0, 5.0, 6.0, 1.0, 2.0]);
# Ok::<(), luma::LumaError>(())
```

A [`ViewMut`](views.md#views-that-write) passed as `&mut view` binds the same way and the kernel
writes the region in place. Any other array (not a view) is passed as a contiguous array. Views
that are not plain translations and flips are evaluated first.

## Errors

Launches validate every argument against the shader's reflected bindings and return errors, not
panics:

```rust
use luma::*;

const COPY: &str = r#"
@group(1) @binding(0) var<storage, read> src: array<f32>;
@group(1) @binding(1) var<storage, read_write> dst: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= luma.size.x) { return; }
    dst[gid.x] = src[gid.x];
}
"#;

let k = Kernel::new(COPY)?;
let src = Array::<f32>::full(4, 1.0)?;
let mut dst = Array::<f32>::zeros(4)?;

// A read-only array where the shader writes:
let err = k.launch([4], args![&dst, &src]).unwrap_err();
assert!(matches!(err, LumaError::KernelArgAccess { .. }), "{err}");
// The wrong number of arguments:
let err = k.launch([4], args![&src]).unwrap_err();
assert!(matches!(err, LumaError::KernelArgCount { expected: 2, actual: 1 }), "{err}");
// The wrong element size (8-byte elements for an `array<f32>`):
let pairs = Array::from_slice(&[[0.0f32; 2]; 4], 4)?;
let err = k.launch([4], args![&pairs, &mut dst]).unwrap_err();
assert!(matches!(err, LumaError::KernelArgElementSize { .. }), "{err}");
// A shader that does not compile:
assert!(matches!(Kernel::new("@compute fn main("), Err(LumaError::ShaderCompile(_))));
# Ok::<(), luma::LumaError>(())
```

A poisoned array passed as an argument fails the launch with its own error. Other errors:
`KernelArgKind` (an array where a uniform goes, or the reverse), `KernelArgUniformSize`,
`KernelEntryPoint`, `KernelBinding` (bindings outside `@group(1)` or not numbered from 0),
`KernelWorkgroup`, `KernelGridTooLarge`. See [`LumaError`](https://docs.rs/luma/latest/luma/enum.LumaError.html).

## Limits

- Kernel arguments must be arrays that fit one buffer; an array split over several buffers
  fails with `LumaError::Chunked`.
- You cannot pass a read-only view as `&mut`, nor write the state of an `iterate` step.
- The shader features the device was opened with decide what WGSL compiles (`f64` needs
  `SHADER_F64`; check `ctx.features()`).
