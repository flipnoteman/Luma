use bytemuck::Pod;

/// Element types that can be stored in an [`Array`](crate::Array) and read by custom kernels.
///
/// Besides the scalars `u32` and `f32`, arrays can hold `[f32; 2]`, `[f32; 4]`, `[u32; 2]` and
/// `[u32; 4]`, whose layout matches WGSL's `vec2`/`vec4` (there is deliberately no `[f32; 3]`:
/// `vec3` is 16-byte aligned in WGSL, which a plain 12-byte array is not).
///
/// The built-in elementwise operations only exist for [`ScalarType`]s.
pub trait GpuType: Pod + std::fmt::Debug {}

/// Element types the built-in operations (`add`, `double`, ...) are implemented for.
pub trait ScalarType: GpuType {
    /// Suffix of the embedded shader implementing an operation for this type.
    fn type_suffix() -> &'static str;
}

impl GpuType for u32 {}
impl GpuType for f32 {}
impl GpuType for [f32; 2] {}
impl GpuType for [f32; 4] {}
impl GpuType for [u32; 2] {}
impl GpuType for [u32; 4] {}

impl ScalarType for u32 {
    fn type_suffix() -> &'static str { "" }
}

impl ScalarType for f32 {
    fn type_suffix() -> &'static str { "_f32" }
}
