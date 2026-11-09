use luma::*;

/// The default device, shared by all tests.
fn ctx() -> Context {
    default_context().expect("Could not create Context.")
}

// -- SAXPY with an encase uniform --

#[derive(encase::ShaderType)]
struct SaxpyParams {
    alpha: f32,
    n: u32,
}

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

#[test]
fn saxpy_non_multiple_of_workgroup() {
    let n = 1000usize; // 15.6 workgroups
    let xs: Vec<f32> = (0..n).map(|i| i as f32 * 0.5).collect();
    let ys: Vec<f32> = (0..n).map(|i| 100.0 - i as f32).collect();
    let alpha = 2.5f32;

    let x = Array::from_slice(&xs, [n, 1, 1, 1]).unwrap();
    let mut y = Array::from_slice(&ys, [n, 1, 1, 1]).unwrap();

    let kernel = ctx().kernel(SAXPY).label("saxpy").build().unwrap();
    assert_eq!(kernel.workgroup(), [64, 1, 1]);
    assert_eq!(kernel.bindings().len(), 3);
    assert_eq!(kernel.bindings()[2].kind, BindingKind::Uniform);

    kernel
        .launch([n as u32], args![&x, &mut y, uniform(&SaxpyParams { alpha, n: n as u32 })])
        .unwrap();

    let expected: Vec<f32> = xs.iter().zip(&ys).map(|(x, y)| alpha * x + y).collect();
    assert_eq!(y.to_vec().unwrap(), expected);
}

#[test]
fn building_twice_and_explicit_workgroup() {
    let a = ctx().kernel(SAXPY).workgroup([64, 1, 1]).build().unwrap();
    let b = ctx().kernel(SAXPY).workgroup([64, 1, 1]).build().unwrap();
    assert_eq!(a.workgroup(), b.workgroup());
    assert!(matches!(
        ctx().kernel(SAXPY).workgroup([32, 1, 1]).build(),
        Err(LumaError::KernelWorkgroup(_))
    ));
}

// -- 2D Jacobi stencil via luma_idx2 --

const JACOBI: &str = r#"
@group(1) @binding(0) var<storage, read> src: array<f32>;
@group(1) @binding(1) var<storage, read_write> dst: array<f32>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let j = gid.y;
    if (i >= luma.size.x || j >= luma.size.y) { return; }
    if (i == 0u || j == 0u || i + 1u == luma.size.x || j + 1u == luma.size.y) {
        dst[luma_idx2(i, j)] = src[luma_idx2(i, j)];
        return;
    }
    dst[luma_idx2(i, j)] = 0.25 * (src[luma_idx2(i - 1u, j)] + src[luma_idx2(i + 1u, j)]
        + src[luma_idx2(i, j - 1u)] + src[luma_idx2(i, j + 1u)]);
}
"#;

#[test]
fn jacobi_stencil_2d() {
    let (nx, ny) = (37usize, 29usize);
    let idx = |i: usize, j: usize| i + j * nx;
    let input: Vec<f32> = (0..nx * ny).map(|k| ((k * 7919) % 101) as f32).collect();

    let mut expected = input.clone();
    for j in 1..ny - 1 {
        for i in 1..nx - 1 {
            expected[idx(i, j)] = 0.25
                * (input[idx(i - 1, j)] + input[idx(i + 1, j)] + input[idx(i, j - 1)]
                    + input[idx(i, j + 1)]);
        }
    }

    let src = Array::from_slice(&input, [nx, ny, 1, 1]).unwrap();
    let mut dst = Array::<f32>::zeros([nx, ny, 1, 1]).unwrap();
    let kernel = ctx().kernel(JACOBI).build().unwrap();
    assert_eq!(kernel.workgroup(), [8, 8, 1]);
    kernel.launch([nx as u32, ny as u32], args![&src, &mut dst]).unwrap();

    assert_eq!(dst.to_vec().unwrap(), expected);
}

#[test]
fn launch_raw_sets_grid_to_workgroups_times_size() {
    const SRC: &str = r#"
    @group(1) @binding(0) var<storage, read_write> out: array<u32>;
    @compute @workgroup_size(4, 2, 1)
    fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
        if (gid.x == 0u && gid.y == 0u && gid.z == 0u) {
            out[0] = luma.size.x; out[1] = luma.size.y; out[2] = luma.size.z;
            out[3] = luma_idx3(1u, 1u, 1u);
        }
    }"#;
    let mut out = Array::<u32>::zeros([4, 1, 1, 1]).unwrap();
    let kernel = ctx().kernel(SRC).build().unwrap();
    kernel.launch_raw([3, 5, 2], args![&mut out]).unwrap();
    // size = (12, 10, 2); idx3(1,1,1) = 1 + 12 * (1 + 10 * 1)
    assert_eq!(out.to_vec().unwrap(), vec![12, 10, 2, 133]);
}

// -- vec4 arrays --

#[test]
fn vec4_array_round_trip() {
    const SRC: &str = r#"
    @group(1) @binding(0) var<storage, read> src: array<vec4<f32>>;
    @group(1) @binding(1) var<storage, read_write> dst: array<vec4<f32>>;
    @compute @workgroup_size(32)
    fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
        if (gid.x >= luma.size.x) { return; }
        let v = src[gid.x];
        dst[gid.x] = vec4<f32>(v.w, v.z, v.y, v.x) * 2.0;
    }"#;
    let n = 50usize;
    let input: Vec<[f32; 4]> = (0..n).map(|i| [i as f32, 1.0, -2.0, 0.5 * i as f32]).collect();
    let src = Array::from_slice(&input, [n, 1, 1, 1]).unwrap();
    let mut dst = Array::<[f32; 4]>::zeros([n, 1, 1, 1]).unwrap();

    ctx().kernel(SRC).build().unwrap().launch([n as u32], args![&src, &mut dst]).unwrap();

    let expected: Vec<[f32; 4]> =
        input.iter().map(|v| [v[3] * 2.0, v[2] * 2.0, v[1] * 2.0, v[0] * 2.0]).collect();
    assert_eq!(dst.to_vec().unwrap(), expected);
    assert_eq!(src.to_vec().unwrap(), input);
}

#[test]
fn other_vector_element_types_round_trip() {
    let a = Array::from_slice(&[[1u32, 2], [3, 4]], [2, 1, 1, 1]).unwrap();
    assert_eq!(a.to_vec().unwrap(), vec![[1u32, 2], [3, 4]]);
    let b = Array::from_slice(&[[1.0f32, 2.0]], [1, 1, 1, 1]).unwrap();
    assert_eq!(b.to_vec().unwrap(), vec![[1.0f32, 2.0]]);
    let c = Array::from_slice(&[[1u32, 2, 3, 4]], [1, 1, 1, 1]).unwrap();
    assert_eq!(c.to_vec().unwrap(), vec![[1u32, 2, 3, 4]]);
}

// -- Uniform layout that naive Pod would get wrong --

/// A 3-component vector for encase (16-byte aligned, 12 bytes wide in WGSL).
#[derive(Clone, Copy)]
struct Vec3([f32; 3]);

impl AsRef<[f32; 3]> for Vec3 {
    fn as_ref(&self) -> &[f32; 3] {
        &self.0
    }
}

impl AsMut<[f32; 3]> for Vec3 {
    fn as_mut(&mut self) -> &mut [f32; 3] {
        &mut self.0
    }
}

impl From<[f32; 3]> for Vec3 {
    fn from(v: [f32; 3]) -> Self {
        Vec3(v)
    }
}

encase::impl_vector!(3, Vec3, f32; using AsRef AsMut From);

/// In WGSL `v` sits at offset 16, `b` at offset 28. A repr(C) struct would put `v` at 4.
#[derive(encase::ShaderType)]
struct Padded {
    a: f32,
    v: Vec3,
    b: f32,
}

#[test]
fn uniform_with_vec3_padding() {
    const SRC: &str = r#"
    struct Padded { a: f32, v: vec3<f32>, b: f32 }
    @group(1) @binding(0) var<uniform> p: Padded;
    @group(1) @binding(1) var<storage, read_write> out: array<f32>;
    @compute @workgroup_size(1)
    fn main() {
        out[0] = p.a; out[1] = p.v.x; out[2] = p.v.y; out[3] = p.v.z; out[4] = p.b;
    }"#;
    let mut out = Array::<f32>::zeros([5, 1, 1, 1]).unwrap();
    let value = Padded { a: 1.0, v: Vec3([2.0, 3.0, 4.0]), b: 5.0 };
    ctx().kernel(SRC).build().unwrap().launch([1], args![uniform(&value), &mut out]).unwrap();
    assert_eq!(out.to_vec().unwrap(), vec![1.0, 2.0, 3.0, 4.0, 5.0]);
}

// -- Errors --

fn saxpy_args<'a>(x: &'a Array<f32>, y: &'a mut Array<f32>) -> Vec<KernelArg<'a>> {
    args![x, y, uniform(&SaxpyParams { alpha: 1.0, n: 4 })]
}

#[test]
fn wrong_arg_count() {
    let x = Array::<f32>::zeros([4, 1, 1, 1]).unwrap();
    let kernel = ctx().kernel(SAXPY).build().unwrap();
    let err = kernel.launch([4], args![&x]).unwrap_err();
    assert!(matches!(err, LumaError::KernelArgCount { expected: 3, actual: 1 }), "{err}");
}

#[test]
fn shared_array_to_read_write_binding() {
    let x = Array::<f32>::zeros([4, 1, 1, 1]).unwrap();
    let y = Array::<f32>::zeros([4, 1, 1, 1]).unwrap();
    let kernel = ctx().kernel(SAXPY).build().unwrap();
    let err = kernel
        .launch([4], args![&x, &y, uniform(&SaxpyParams { alpha: 1.0, n: 4 })])
        .unwrap_err();
    assert!(matches!(err, LumaError::KernelArgAccess { index: 1 }), "{err}");
}

#[test]
fn mutable_array_to_read_binding_is_allowed() {
    let mut x = Array::from_slice(&[1.0f32; 4], 4).unwrap();
    let mut y = Array::from_slice(&[1.0f32; 4], 4).unwrap();
    let kernel = ctx().kernel(SAXPY).build().unwrap();
    kernel
        .launch([4], args![&mut x, &mut y, uniform(&SaxpyParams { alpha: 2.0, n: 4 })])
        .unwrap();
    assert_eq!(y.to_vec().unwrap(), vec![3.0; 4]);
}

#[test]
fn uniform_to_storage_binding_and_vice_versa() {
    let x = Array::<f32>::zeros([4, 1, 1, 1]).unwrap();
    let mut y = Array::<f32>::zeros([4, 1, 1, 1]).unwrap();
    let kernel = ctx().kernel(SAXPY).build().unwrap();
    let p = SaxpyParams { alpha: 1.0, n: 4 };

    let err = kernel.launch([4], args![uniform(&p), &mut y, uniform(&p)]).unwrap_err();
    assert!(
        matches!(err, LumaError::KernelArgKind { index: 0, expected: BindingKind::StorageRead, .. }),
        "{err}"
    );
    let err = kernel.launch([4], args![&x, &mut y, &x]).unwrap_err();
    assert!(
        matches!(err, LumaError::KernelArgKind { index: 2, expected: BindingKind::Uniform, .. }),
        "{err}"
    );
}

#[test]
fn element_size_mismatch() {
    let x = Array::<[f32; 4]>::zeros([4, 1, 1, 1]).unwrap();
    let mut y = Array::<f32>::zeros([4, 1, 1, 1]).unwrap();
    let kernel = ctx().kernel(SAXPY).build().unwrap();
    let err = kernel
        .launch([4], args![&x, &mut y, uniform(&SaxpyParams { alpha: 1.0, n: 4 })])
        .unwrap_err();
    assert!(
        matches!(err, LumaError::KernelArgElementSize { index: 0, expected: 4, actual: 16 }),
        "{err}"
    );
}

#[test]
fn arrays_from_another_context_are_rejected() {
    let other = Context::new().unwrap();
    let x = Array::<f32>::zeros_on(&other, [4, 1, 1, 1]).unwrap();
    let mut y = Array::<f32>::zeros([4, 1, 1, 1]).unwrap();
    let kernel = ctx().kernel(SAXPY).build().unwrap();
    let err = kernel.launch([4], saxpy_args(&x, &mut y)).unwrap_err();
    assert!(matches!(err, LumaError::ContextMismatch), "{err}");
}

#[test]
fn user_binding_in_group_0() {
    const SRC: &str = r#"
    @group(0) @binding(1) var<storage, read_write> out: array<f32>;
    @compute @workgroup_size(1)
    fn main() { out[0] = 1.0; }"#;
    let err = ctx().kernel(SRC).build().unwrap_err();
    assert!(matches!(&err, LumaError::KernelBinding(m) if m.contains("@group(0)")), "{err}");
}

#[test]
fn syntax_error_is_shader_compile() {
    let err = ctx().kernel("@compute @workgroup_size(1) fn main( { }").build().unwrap_err();
    assert!(matches!(err, LumaError::ShaderCompile(_)), "{err}");
}

#[test]
fn validation_error_is_shader_compile() {
    const SRC: &str = r#"
    @compute @workgroup_size(1)
    fn main() { let x: u32 = 1.5; }"#;
    let err = ctx().kernel(SRC).build().unwrap_err();
    assert!(matches!(err, LumaError::ShaderCompile(_)), "{err}");
}

#[test]
fn missing_entry_point() {
    let err = ctx().kernel(SAXPY).entry("nope").build().unwrap_err();
    assert!(matches!(err, LumaError::KernelEntryPoint(_)), "{err}");
}

#[test]
fn grid_too_large() {
    let mut y = Array::<f32>::zeros([4, 1, 1, 1]).unwrap();
    let x = Array::<f32>::zeros([4, 1, 1, 1]).unwrap();
    let kernel = ctx().kernel(SAXPY).build().unwrap();
    let err = kernel.launch([u32::MAX], saxpy_args(&x, &mut y)).unwrap_err();
    assert!(matches!(err, LumaError::KernelGridTooLarge { .. }), "{err}");
}
