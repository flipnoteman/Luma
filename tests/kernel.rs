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

fn saxpy_args<'a>(x: &'a Array<f32>, y: &'a mut Array<f32>) -> [KernelArg<'a>; 3] {
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

// -- Entry-point scoped reflection --

/// One module, several entry points with different binding sets. `x` and `y` are bindings 0 and
/// 1; `scale_params` is reached through a helper function; `junk` is used by no entry point and
/// sits in a group luma does not allow.
const MULTI: &str = r#"
struct Params { factor: f32, n: u32 }
@group(1) @binding(0) var<storage, read_write> x: array<f32>;
@group(1) @binding(1) var<storage, read_write> y: array<f32>;
@group(1) @binding(2) var<uniform> scale_params: Params;
@group(2) @binding(0) var<storage, read> junk: array<f32>;

fn factor() -> f32 { return scale_params.factor; }

// Uses only `x`: one argument.
@compute @workgroup_size(16)
fn increment(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x < luma.size.x) { x[gid.x] = x[gid.x] + 1.0; }
}

// Uses `x` and `y`: two arguments.
@compute @workgroup_size(16)
fn add(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x < luma.size.x) { y[gid.x] = y[gid.x] + x[gid.x]; }
}

// Uses `x`, `y` and, through `factor()`, the uniform: three arguments.
@compute @workgroup_size(16)
fn scale_into(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x < luma.size.x) { y[gid.x] = x[gid.x] * factor(); }
}

// Uses only `y` (@binding(1)): the numbering does not start at 0.
@compute @workgroup_size(16)
fn only_y(@builtin(global_invocation_id) gid: vec3<u32>) {
    y[gid.x] = 0.0;
}

// Uses nothing but the grid.
@compute @workgroup_size(16)
fn nothing(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= luma.size.x) { return; }
}
"#;

#[derive(encase::ShaderType)]
struct ScaleParams {
    factor: f32,
    n: u32,
}

#[test]
fn entry_points_of_one_module_have_their_own_bindings() {
    let ctx = ctx();
    let build = |entry: &str| ctx.kernel(MULTI).entry(entry).build();

    let increment = build("increment").unwrap();
    let add = build("add").unwrap();
    let scale = build("scale_into").unwrap();
    let nothing = build("nothing").unwrap();

    let names = |k: &Kernel| k.bindings().iter().map(|b| b.name.clone()).collect::<Vec<_>>();
    assert_eq!(names(&increment), ["x"]);
    assert_eq!(names(&add), ["x", "y"]);
    assert_eq!(names(&scale), ["x", "y", "scale_params"]);
    assert_eq!(scale.bindings()[2].kind, BindingKind::Uniform);
    assert!(nothing.bindings().is_empty());

    let n = 100u32;
    let mut x = Array::from_slice(&(0..n).map(|i| i as f32).collect::<Vec<_>>(), n as usize).unwrap();
    let mut y = Array::<f32>::zeros(n as usize).unwrap();

    increment.launch([n], args![&mut x]).unwrap(); // x = i + 1
    add.launch([n], args![&mut x, &mut y]).unwrap(); // y = i + 1
    add.launch([n], args![&mut x, &mut y]).unwrap(); // y = 2 (i + 1)
    nothing.launch([n], args![]).unwrap();
    nothing.launch([n], args![]).unwrap();
    assert_eq!(
        y.to_vec().unwrap(),
        (0..n).map(|i| 2.0 * (i + 1) as f32).collect::<Vec<_>>()
    );

    scale.launch([n], args![&mut x, &mut y, uniform(&ScaleParams { factor: 3.0, n })]).unwrap();
    assert_eq!(y.to_vec().unwrap(), (0..n).map(|i| 3.0 * (i + 1) as f32).collect::<Vec<_>>());

    // The argument count is the entry point's, not the module's.
    let err = increment.launch([n], args![&mut x, &mut y]).unwrap_err();
    assert!(matches!(err, LumaError::KernelArgCount { expected: 1, actual: 2 }), "{err}");
    let err = add.launch([n], args![&x]).unwrap_err();
    assert!(matches!(err, LumaError::KernelArgCount { expected: 2, actual: 1 }), "{err}");
}

#[test]
fn entry_point_bindings_must_still_be_contiguous_from_zero() {
    let err = ctx().kernel(MULTI).entry("only_y").build().unwrap_err();
    match err {
        LumaError::KernelBinding(message) => {
            assert!(message.contains("`y`") && message.contains("only_y"), "{message}")
        }
        other => panic!("expected KernelBinding, got {other}"),
    }
}

#[test]
fn unused_globals_in_unsupported_places_are_ignored_but_used_ones_are_not() {
    // `junk` is in @group(2) in MULTI and builds fine because no entry point uses it.
    const USES_JUNK: &str = r#"
    @group(2) @binding(0) var<storage, read> junk: array<f32>;
    @group(1) @binding(0) var<storage, read_write> out: array<f32>;
    @compute @workgroup_size(1)
    fn main() { out[0] = junk[0]; }"#;
    assert!(matches!(
        ctx().kernel(USES_JUNK).build(),
        Err(LumaError::KernelBinding(message)) if message.contains("@group(2)")
    ));

    // Same module, a second entry point that does not touch it: builds.
    const BOTH: &str = r#"
    @group(2) @binding(0) var<storage, read> junk: array<f32>;
    @group(1) @binding(0) var<storage, read_write> out: array<f32>;
    @compute @workgroup_size(1)
    fn uses_junk() { out[0] = junk[0]; }
    @compute @workgroup_size(1)
    fn fine() { out[0] = 7.0; }"#;
    assert!(ctx().kernel(BOTH).entry("uses_junk").build().is_err());
    let fine = ctx().kernel(BOTH).entry("fine").build().unwrap();
    let mut out = Array::<f32>::zeros(1).unwrap();
    fine.launch([1], args![&mut out]).unwrap();
    assert_eq!(out.to_vec().unwrap(), vec![7.0]);
}

#[test]
fn grid_uniform_slots_are_shared_per_size_without_mixing_sizes_up() {
    const SRC: &str = r#"
    @group(1) @binding(0) var<storage, read_write> out: array<u32>;
    @group(1) @binding(1) var<uniform> slot: u32;
    @compute @workgroup_size(1)
    fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
        if (gid.x == 0u && gid.y == 0u && gid.z == 0u) {
            out[slot * 3u] = luma.size.x; out[slot * 3u + 1u] = luma.size.y; out[slot * 3u + 2u] = luma.size.z;
        }
    }"#;
    #[derive(encase::ShaderType)]
    struct Slot {
        slot: u32,
    }
    let kernel = ctx().kernel(SRC).build().unwrap();
    let sizes: [[u32; 3]; 4] = [[3, 1, 1], [5, 2, 1], [7, 3, 2], [3, 1, 1]];
    // Enough launches to cross batch limits and several arena chunks, each size revisited.
    let launches = 3000usize;
    let mut out = Array::<u32>::zeros(launches * 3).unwrap();
    let mut expected = vec![0u32; launches * 3];
    for i in 0..launches {
        let size = sizes[(i * 7 + i / 5) % sizes.len()];
        kernel.launch(size, args![&mut out, uniform(&Slot { slot: i as u32 })]).unwrap();
        expected[i * 3..i * 3 + 3].copy_from_slice(&size);
    }
    assert_eq!(out.to_vec().unwrap(), expected);
}

#[test]
fn many_arguments_and_large_uniforms_take_the_allocating_paths() {
    // 20 storage bindings (more than the inline argument buffer holds) and a uniform larger
    // than the inline uniform buffer (256 bytes: 64 f32 in 16 vec4).
    let mut src = String::from("struct Big { v: array<vec4<f32>, 16> }\n");
    for i in 0..20 {
        src += &format!("@group(1) @binding({i}) var<storage, read_write> b{i}: array<f32>;\n");
    }
    src += "@group(1) @binding(20) var<uniform> big: Big;\n";
    src += "@compute @workgroup_size(1)\nfn main() {\n";
    for i in 0..20 {
        src += &format!("    b{i}[0] = big.v[15].w + {i}.0;\n");
    }
    src += "}\n";

    #[derive(encase::ShaderType)]
    struct Big {
        v: [encase_vec4::V4; 16],
    }
    mod encase_vec4 {
        #[derive(Clone, Copy, encase::ShaderType)]
        pub struct V4 {
            pub x: f32,
            pub y: f32,
            pub z: f32,
            pub w: f32,
        }
    }
    // An array of structs of four scalars has the stride 16 that uniform arrays need.
    let big = Big { v: [encase_vec4::V4 { x: 0.0, y: 0.0, z: 0.0, w: 100.0 }; 16] };

    let kernel = ctx().kernel(&src).build().unwrap();
    assert_eq!(kernel.bindings().len(), 21);
    let mut arrays: Vec<Array<f32>> = (0..20).map(|_| Array::<f32>::zeros(1).unwrap()).collect();
    let mut launch_args: Vec<KernelArg<'_>> = arrays.iter_mut().map(KernelArg::from).collect();
    launch_args.push(uniform(&big));
    kernel.launch([1], launch_args).unwrap();
    for (i, array) in arrays.iter().enumerate() {
        assert_eq!(array.to_vec().unwrap(), vec![100.0 + i as f32]);
    }
}

// -- strided views --

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

const VIEW_FILL: &str = r#"
@group(1) @binding(0) var<storage, read_write> dst: array<f32>;
@group(1) @binding(1) var<uniform> dst_view: LumaView;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= luma.size.x || gid.y >= luma.size.y) { return; }
    dst[luma_view_idx2(dst_view, gid.x, gid.y)] = 100.0 + f32(gid.x) + 10.0 * f32(gid.y);
}
"#;

const VIEW_FILL_LINEAR: &str = r#"
@group(1) @binding(0) var<storage, read_write> dst: array<f32>;
@group(1) @binding(1) var<uniform> dst_view: LumaView;

@compute @workgroup_size(16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= dst_view.len) { return; }
    dst[luma_view_idx(dst_view, gid.x)] = f32(gid.x);
}
"#;

fn grid() -> Vec<f32> {
    (0..12).map(|n| n as f32).collect()
}

#[test]
fn read_view_reads_a_flipped_slice_in_place() {
    let a = Array::from_slice(&grid(), [4, 3, 1, 1]).unwrap();
    let v = a.slice(0, 1..3).flip(1);
    let out = Array::<f32>::zeros([2, 3, 1, 1]).unwrap();
    let mut out = out;
    let kernel = ctx().kernel(VIEW_COPY).build().unwrap();
    assert_eq!(kernel.bindings().len(), 3);
    kernel.launch([2, 3], args![luma::view(&v), &mut out]).unwrap();

    let mut expected = vec![0.0; 6];
    for j in 0..3 {
        for i in 0..2 {
            expected[i + 2 * j] = ((1 + i) + 4 * (2 - j)) as f32;
        }
    }
    assert_eq!(out.to_vec().unwrap(), expected);
}

#[test]
fn read_view_reads_a_transpose() {
    let a = Array::from_slice(&grid(), [4, 3, 1, 1]).unwrap();
    let t = a.transpose();
    let mut out = Array::<f32>::zeros([3, 4, 1, 1]).unwrap();
    let kernel = ctx().kernel(VIEW_COPY).build().unwrap();
    kernel.launch([3, 4], args![luma::view(&t), &mut out]).unwrap();

    let mut expected = vec![0.0; 12];
    for j in 0..4 {
        for i in 0..3 {
            expected[i + 3 * j] = (j + 4 * i) as f32;
        }
    }
    assert_eq!(out.to_vec().unwrap(), expected);
}

#[test]
fn mutable_view_writes_a_subregion_and_flushes_earlier_readers() {
    let mut dst = Array::<f32>::zeros([4, 3, 1, 1]).unwrap();
    let before = &dst + 1.0;

    let kernel = ctx().kernel(VIEW_FILL).build().unwrap();
    {
        let mut v = dst.slice_mut(0, 1..3);
        kernel.launch([2, 3], args![&mut v]).unwrap();
    }

    assert_eq!(before.to_vec().unwrap(), vec![1.0; 12]);
    let mut expected = vec![0.0; 12];
    for j in 0..3 {
        for i in 0..2 {
            expected[(1 + i) + 4 * j] = 100.0 + i as f32 + 10.0 * j as f32;
        }
    }
    assert_eq!(dst.to_vec().unwrap(), expected);
}

#[test]
fn mutable_view_linear_index_follows_view_order() {
    let mut dst = Array::<f32>::zeros([4, 3, 1, 1]).unwrap();
    let kernel = ctx().kernel(VIEW_FILL_LINEAR).build().unwrap();
    {
        let mut v = dst.slice_mut(1, 1..3);
        kernel.launch([8], args![&mut v]).unwrap();
    }
    let mut expected = vec![0.0; 12];
    for (n, slot) in expected[4..12].iter_mut().enumerate() {
        *slot = n as f32;
    }
    assert_eq!(dst.to_vec().unwrap(), expected);
}

#[test]
fn read_only_view_cannot_be_passed_as_mut_array() {
    let a = Array::from_slice(&grid(), [4, 3, 1, 1]).unwrap();
    let mut s = a.slice(0, 1..3);
    let kernel = ctx().kernel(SAXPY).build().unwrap();
    let x = Array::<f32>::zeros([2, 1, 1, 1]).unwrap();
    let result = kernel.launch([2], args![&x, &mut s, uniform(&SaxpyParams { alpha: 1.0, n: 2 })]);
    assert!(matches!(result, Err(LumaError::InvalidShape(_))));
}

#[test]
fn read_view_keeps_snapshot_after_base_is_overwritten() {
    let mut a = Array::from_slice(&grid(), [4, 3, 1, 1]).unwrap();
    let s = a.slice(0, 1..3);
    a.write(&[99.0; 12]).unwrap();
    let s2 = s.slice(0, 1..2);

    let kernel = ctx().kernel(VIEW_COPY).build().unwrap();
    let mut out1 = Array::<f32>::zeros([2, 3, 1, 1]).unwrap();
    kernel.launch([2, 3], args![luma::view(&s), &mut out1]).unwrap();
    assert_eq!(out1.to_vec().unwrap(), vec![1.0, 2.0, 5.0, 6.0, 9.0, 10.0]);

    let mut out2 = Array::<f32>::zeros([1, 3, 1, 1]).unwrap();
    kernel.launch([1, 3], args![luma::view(&s2), &mut out2]).unwrap();
    assert_eq!(out2.to_vec().unwrap(), vec![2.0, 6.0, 10.0]);
}
