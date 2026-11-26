//! Declared kernel reach (`Reach`, `KernelBuilder::elementwise`, `KernelBuilder::halo`).

use luma::*;

fn ctx() -> Context {
    default_context().expect("Could not create Context.")
}

const SCALE: &str = r#"
@group(1) @binding(0) var<storage, read> x: array<f32>;
@group(1) @binding(1) var<storage, read_write> y: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= luma.size.x) { return; }
    y[gid.x] = 2.0 * x[gid.x];
}
"#;

const STENCIL: &str = r#"
@group(1) @binding(0) var<storage, read> x: array<f32>;
@group(1) @binding(1) var<storage, read_write> y: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = luma.size.x;
    if (gid.x >= n) { return; }
    var lo = 0.0;
    var hi = 0.0;
    if (gid.x > 0u) { lo = x[gid.x - 1u]; }
    if (gid.x + 1u < n) { hi = x[gid.x + 1u]; }
    y[gid.x] = lo + x[gid.x] + hi;
}
"#;

fn array(n: usize) -> Array<f32> {
    Array::from_slice(&(0..n).map(|i| i as f32).collect::<Vec<_>>(), n).unwrap()
}

#[test]
fn reach_defaults_to_whole() {
    let kernel = ctx().kernel(SCALE).build().unwrap();
    assert_eq!(kernel.reach(), Reach::Whole);
    assert_eq!(ctx().kernel(SCALE).elementwise().build().unwrap().reach(), Reach::Elementwise);
    assert_eq!(ctx().kernel(SCALE).halo(2).build().unwrap().reach(), Reach::Halo { radius: 2 });
}

#[test]
fn an_elementwise_kernel_rejects_arrays_of_other_lengths() {
    let kernel = ctx().kernel(SCALE).elementwise().build().unwrap();
    let x = array(100);
    let mut y = Array::<f32>::zeros(99).unwrap();
    let err = kernel.launch([100u32], args![&x, &mut y]).unwrap_err();
    assert!(matches!(err, LumaError::KernelReach { index: 1, .. }), "{err}");
    let mut y = Array::<f32>::zeros(100).unwrap();
    kernel.launch([100u32], args![&x, &mut y]).unwrap();
    assert_eq!(y.to_vec().unwrap()[7], 14.0);
    // The problem size must match too.
    let err = kernel.launch([50u32], args![&x, &mut y]).unwrap_err();
    assert!(matches!(err, LumaError::KernelReach { index: 0, .. }), "{err}");
}

#[test]
fn an_elementwise_kernel_rejects_launch_raw_views_and_textures() {
    let kernel = ctx().kernel(SCALE).elementwise().build().unwrap();
    let x = array(128);
    let mut y = Array::<f32>::zeros(128).unwrap();
    let err = kernel.launch_raw([2, 1, 1], args![&x, &mut y]).unwrap_err();
    assert!(matches!(err, LumaError::KernelReach { .. }), "{err}");

    // A view binds two slots; the kernel above has two bindings, so give it a shape that
    // reaches the reach check first: a one-argument kernel over a view.
    const VIEWED: &str = r#"
    @group(1) @binding(0) var<storage, read> x: array<f32>;
    @group(1) @binding(1) var<uniform> x_view: LumaView;
    @group(1) @binding(2) var<storage, read_write> y: array<f32>;
    @compute @workgroup_size(64)
    fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
        if (gid.x >= luma.size.x) { return; }
        y[gid.x] = x[luma_view_idx(x_view, gid.x)];
    }
    "#;
    let kernel = ctx().kernel(VIEWED).elementwise().build().unwrap();
    let err = kernel.launch([128u32], args![view(&x), &mut y]).unwrap_err();
    assert!(matches!(err, LumaError::KernelReach { index: 0, .. }), "{err}");
}

#[test]
fn a_halo_kernel_runs_a_three_point_stencil_with_its_own_edge_guard() {
    let kernel = ctx().kernel(STENCIL).halo(1).build().unwrap();
    let n = 300;
    let x = array(n);
    let mut y = Array::<f32>::zeros(n).unwrap();
    kernel.launch([n as u32], args![&x, &mut y]).unwrap();
    let got = y.to_vec().unwrap();
    let want: Vec<f32> = (0..n)
        .map(|i| {
            let lo = if i > 0 { (i - 1) as f32 } else { 0.0 };
            let hi = if i + 1 < n { (i + 1) as f32 } else { 0.0 };
            lo + i as f32 + hi
        })
        .collect();
    assert_eq!(got, want);
}

#[test]
fn a_whole_kernel_still_gives_chunked_on_split_arrays() {
    let ctx = Context::with_options(ContextOptions::from_env().chunk_bytes(4096)).unwrap();
    let n = 5000;
    let x = Array::from_slice_on(&ctx, &vec![1.0f32; n], n).unwrap();
    let mut y = Array::<f32>::zeros_on(&ctx, n).unwrap();
    let kernel = ctx.kernel(SCALE).build().unwrap();
    let err = kernel.launch([n as u32], args![&x, &mut y]).unwrap_err();
    assert!(matches!(err.root(), LumaError::Chunked(_)), "{err}");
    // A declared kernel is refused the same way in wave 2.
    let kernel = ctx.kernel(SCALE).elementwise().build().unwrap();
    let err = kernel.launch([n as u32], args![&x, &mut y]).unwrap_err();
    assert!(matches!(err.root(), LumaError::Chunked(_)), "{err}");
}
