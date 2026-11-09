use bytemuck::Pod;

/// Element types that can be stored in an [`Array`](crate::Array) and read by custom kernels.
///
/// Besides the scalars `u32` and `f32`, arrays can hold `[f32; 2]`, `[f32; 4]`, `[u32; 2]` and
/// `[u32; 4]`, whose layout matches WGSL's `vec2`/`vec4` (there is deliberately no `[f32; 3]`:
/// `vec3` is 16-byte aligned in WGSL, which a plain 12-byte array is not).
///
/// The built-in math only exists for [`ScalarType`]s; `[f32; 2]` and `[f32; 4]` arrays also have
/// per-component reductions (`sum`, `min`, `max`, `norm`).
pub trait GpuType: Pod + std::fmt::Debug {}

/// Element types the built-in operators, math functions and reductions are implemented for.
pub trait ScalarType: GpuType {
    /// The WGSL name of the type, used to generate the shaders of the built-in operations.
    fn wgsl_type() -> &'static str;
}

impl GpuType for u32 {}
impl GpuType for f32 {}
impl GpuType for [f32; 2] {}
impl GpuType for [f32; 4] {}
impl GpuType for [u32; 2] {}
impl GpuType for [u32; 4] {}

impl ScalarType for u32 {
    fn wgsl_type() -> &'static str {
        "u32"
    }
}

impl ScalarType for f32 {
    fn wgsl_type() -> &'static str {
        "f32"
    }
}
