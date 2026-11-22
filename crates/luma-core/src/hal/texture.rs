//! Textures: sampled 2-D images a kernel reads through the device's texture units (its texture
//! cache, hardware clamp-to-edge addressing and, if asked, hardware bilinear filtering).
//!
//! A texture is made from the data of a buffer by [`Backend::create_texture`](super::Backend::create_texture)
//! and passed to a launch in place of a buffer, for bindings of kind
//! [`BindingKind::Texture2d`](super::BindingKind::Texture2d) (a `texture_2d<f32>`) and
//! [`BindingKind::Sampler`](super::BindingKind::Sampler) (a `sampler`, bound from the same
//! texture). A kernel reads it with `textureLoad` (integer texel coordinates, no filtering),
//! `textureGather` (the four texels around a point, unfiltered and exact) and
//! `textureSampleLevel` (level 0, the filter of the texture).
//!
//! Addressing is always clamp-to-edge. Coordinates of `textureGather` and `textureSampleLevel`
//! are normalized, as in WGSL: texel `i` of `n` is centred on `(i + 0.5) / n`.

/// The texel layout of a texture.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TextureFormat {
    /// One `f32` per texel (the other channels of a fetch read 0, 0, 1).
    R32Float,
    /// Four `f32` per texel, interleaved.
    Rgba32Float,
}

impl TextureFormat {
    /// `f32`s per texel.
    pub const fn channels(self) -> u32 {
        match self {
            TextureFormat::R32Float => 1,
            TextureFormat::Rgba32Float => 4,
        }
    }
    /// Bytes per texel.
    pub const fn texel_bytes(self) -> u32 {
        4 * self.channels()
    }
}

/// How `textureSampleLevel` filters between texels.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TextureFilter {
    /// The nearest texel.
    Nearest,
    /// Bilinear, by the texture unit. NVIDIA hardware weights the texels with 8 bits of
    /// fraction (steps of 1/256), other hardware may use more.
    Linear,
}

/// A texture to create.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TextureDesc {
    /// Texels per row.
    pub width: u32,
    /// Rows.
    pub height: u32,
    /// Texel layout.
    pub format: TextureFormat,
    /// Filtering of `textureSampleLevel`.
    pub filter: TextureFilter,
}

/// What a backend's textures can do.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TextureCaps {
    /// The largest width and height.
    pub max_dim: u32,
    /// The source rows of a texture are a multiple of this many bytes apart.
    pub pitch_align: u32,
    /// Whether a texture can alias its source buffer: building it then costs no copy.
    pub aliases: bool,
}
