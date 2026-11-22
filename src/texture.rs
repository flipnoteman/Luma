//! Textures over an array's buffer: the access path a kernel takes for reads that are 2-D in nature
//! (a position that is computed, a window with 2-D locality, a border), through the texture units
//! of a device that has them (see [`luma_core::Backend::create_texture`]).
//!
//! A texture is an implementation detail of how luma reads an `f32` array: nothing is exposed to
//! the user, and the reads are exact (point fetches and gathers, no hardware filtering) unless the
//! caller asked for [filtered sampling](crate::Interp). The decision to read through one is made by
//! [`crate::jit::cost::texture_pays`], which weighs what making the view costs.

use luma_core::{Buffer, TextureDesc, TextureFilter, TextureFormat};

use crate::array::Array;
use crate::context::Context;
use crate::error::Result;
use crate::gpu::pager::Pin;

/// A texture of the first `width * height` elements of an `f32` array, rows of `width` elements.
pub(crate) struct Texture {
    pub(crate) ctx: Context,
    pub(crate) buffer: Buffer,
    /// Keeps the array's buffer where the texture can read it.
    pub(crate) pin: Pin,
}

impl Context {
    /// A texture of `src` as `width` x `height` texels, or `None` when the device has none, the
    /// size does not fit, or a recording is active (the texture would be built at record time and
    /// read at replay).
    pub(crate) fn texture_2d(&self, src: &Array<f32>, width: usize, height: usize, filter: TextureFilter) -> Result<Option<Texture>> {
        let Some(caps) = self.hal().texture_caps() else { return Ok(None) };
        let (Ok(w), Ok(h)) = (u32::try_from(width), u32::try_from(height)) else { return Ok(None) };
        if w == 0 || h == 0 || w > caps.max_dim || h > caps.max_dim || self.recording_here() {
            return Ok(None);
        }
        let pitch = u64::from(w) * 4;
        // The device wants rows this many bytes apart: a texture of another width is not made
        // (building it would copy the image, see `jit::cost::texture_pays`).
        if pitch % u64::from(caps.pitch_align) != 0 {
            return Ok(None);
        }
        let Ok((pin, _)) = src.ready() else { return Ok(None) };
        let desc = TextureDesc { width: w, height: h, format: TextureFormat::R32Float, filter };
        let buffer = self.hal().create_texture(pin.buffer(), 0, pitch, &desc)?;
        Ok(Some(Texture { ctx: self.clone(), buffer, pin }))
    }
}
