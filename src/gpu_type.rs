use bytemuck::Pod;

/// Trait to map Rust types to shader type suffixes.
pub trait GpuType: Pod + std::fmt::Debug {
    fn type_suffix() -> &'static str;
}

impl GpuType for u32 {
    fn type_suffix() -> &'static str { "" }
}

impl GpuType for f32 {
    fn type_suffix() -> &'static str { "_f32" }
}
