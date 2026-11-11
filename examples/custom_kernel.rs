//! Running your own WGSL kernels on Luma arrays.
//!
//! Luma binds a `luma` uniform in `@group(0)` (`luma.size`, the problem size passed to `launch`)
//! and provides `luma_idx2(i, j)` / `luma_idx3(i, j, k)` for indexing (x fastest). Your bindings
//! go in `@group(1)`, numbered from 0 in argument order:
//!
//! - `&Array`        -> `var<storage, read>`
//! - `&mut Array`    -> `var<storage, read_write>`
//! - `uniform(&val)` -> `var<uniform>` (use `#[derive(encase::ShaderType)]` so the layout matches)
//!
//! Arguments are checked against the shader at launch. Launches are recorded in the same batch
//! as built-in operations, so they can be freely mixed.

use luma::*;

// 1. A single-entry kernel with a uniform: y = alpha * x + y.
#[derive(encase::ShaderType)]
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

// 2. One module with several entry points. Each entry point's arguments are only the globals it
//    uses, and those must be numbered contiguously from 0: `fill_index` takes (dst) and
//    `transpose` takes (dst, src).
const GRID: &str = r#"
@group(1) @binding(0) var<storage, read_write> dst: array<f32>;
@group(1) @binding(1) var<storage, read> src: array<f32>;

// Writes each element's flat index, on a 2D grid.
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

fn main() -> Result<()> {
    // --- saxpy -------------------------------------------------------------------------------
    let x = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], 4)?;
    let mut y = Array::from_slice(&[10.0f32; 4], 4)?;

    let saxpy = Kernel::new(SAXPY)?; // shorthand for ctx.kernel(SAXPY).entry("main").build()
    saxpy.launch([4], args![&x, &mut y, uniform(&Saxpy { alpha: 2.0, n: 4 })])?;

    // Operators and launches share one ordered batch; nothing has run on the GPU yet.
    let y = y * 0.5;
    println!("saxpy:     {:?}", y.to_vec()?); // [6.0, 7.0, 8.0, 9.0]

    // --- several entry points, 2D launches ---------------------------------------------------
    let ctx = default_context()?;
    let fill = ctx.kernel(GRID).entry("fill_index").build()?;
    let transpose = ctx.kernel(GRID).entry("transpose").label("transpose").build()?;

    let (w, h) = (4u32, 3u32);
    let mut a = Array::<f32>::zeros([w as usize, h as usize])?;
    let mut t = Array::<f32>::zeros([h as usize, w as usize])?;

    fill.launch([w, h], args![&mut a])?;
    transpose.launch([w, h], args![&mut t, &a])?;

    println!("a ({w}x{h}):  {:?}", a.to_vec()?);
    println!("t ({h}x{w}):  {:?}", t.to_vec()?);

    // Built-in reductions work on the output like on any other array.
    println!("sum of t:  {}", t.sum()?); // 0 + 1 + ... + 11 = 66

    // Mistakes in the arguments are reported as errors, not panics.
    let err = transpose.launch([w, h], args![&a, &mut t]).unwrap_err();
    println!("bad args:  {err}");

    Ok(())
}
