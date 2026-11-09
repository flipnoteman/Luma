
# Luma
A cross-platform GPU-accelerated linear algebra library WIP

Current Usage Example:

```rs
use luma::*;

fn main() -> Result<()> {
    let t = std::time::Instant::now();

    // A [Context] is an explicit handle to the GPU. It is cheap to clone and creates [Array]s.
    let ctx = Context::new()?;

    let array1 = ctx.from_slice(&[3, 1, 1, 1], &[1u32, 6, 5])?;
    let array2 = ctx.from_slice(&[3, 1, 1, 1], &[4u32, 12, 10])?;

    // Operations are submitted to the GPU without waiting and return new arrays.
    let sum = array1.add(&array2)?;     // [5, 18, 15]
    let doubled = sum.double()?;        // [10, 36, 30]

    // `read()` is the only call that waits for the GPU and copies data back to the host.
    println!("Result = {:?}; shape = {:?}", doubled.read()?, doubled.shape());

    println!("Program Time: {:?}", t.elapsed());
    Ok(())
}
```

## Custom kernels

Write your own WGSL compute kernels and run them on `Array`s. Luma binds a `luma` grid uniform
(`luma.size`, the problem size) in `@group(0)` and provides `luma_idx2(i, j)` / `luma_idx3(i, j, k)`
(row-major, x fastest). Your own bindings go in `@group(1)`, numbered from 0 in argument order:
`&Array` binds `var<storage, read>`, `&mut Array` binds `var<storage, read_write>` and
`uniform(&value)` binds a `var<uniform>` (laid out by [encase](https://crates.io/crates/encase),
so `#[derive(encase::ShaderType)]` structs match WGSL's padding rules). Arguments are checked
against the shader at dispatch; dispatches are submitted without waiting.

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
    let ctx = Context::new()?;
    let x = ctx.from_slice(&[4, 1, 1, 1], &[1.0f32, 2.0, 3.0, 4.0])?;
    let mut y = ctx.from_slice(&[4, 1, 1, 1], &[10.0f32; 4])?;

    let saxpy = ctx.kernel(SAXPY).build()?;
    saxpy.dispatch([4], args![&x, &mut y, uniform(&Params { alpha: 2.0, n: 4 })])?;

    println!("{:?}", y.read()?); // [12.0, 14.0, 16.0, 18.0]
    Ok(())
}
```

Arrays can also hold `[f32; 2]`, `[f32; 4]`, `[u32; 2]` and `[u32; 4]` elements (WGSL `vec2`/`vec4`);
the built-in operations (`add`, `double`, ...) are only available for `u32` and `f32`.
