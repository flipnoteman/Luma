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

#[derive(Debug)]
/// The texture a segment keeps for its data (see [`Context::texture_2d`]).
pub(crate) struct Cached {
    /// The buffer the data was in (a segment that moved to host memory and back has another one).
    source: u64,
    width: u32,
    height: u32,
    filter: TextureFilter,
    texture: Buffer,
}

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
        if !pitch.is_multiple_of(u64::from(caps.pitch_align)) {
            return Ok(None);
        }
        let Ok((pin, _)) = src.ready() else { return Ok(None) };
        // A texture that reads the buffer itself stays valid while the array changes, and is kept
        // for the array's next reader; one that copied the data is only good for what it copied.
        let keep = caps.aliases;
        let segment = std::sync::Arc::clone(pin.segment());
        let slot = segment.texture_slot();
        if keep
            && let Some(cached) = crate::gpu::lock(slot).as_ref()
            && cached.source == pin.buffer().id()
            && (cached.width, cached.height, cached.filter) == (w, h, filter)
        {
            return Ok(Some(Texture { ctx: self.clone(), buffer: cached.texture.clone(), pin }));
        }
        let desc = TextureDesc { width: w, height: h, format: TextureFormat::R32Float, filter };
        let buffer = self.hal().create_texture(pin.buffer(), 0, pitch, &desc)?;
        if keep {
            *crate::gpu::lock(slot) = Some(Cached { source: pin.buffer().id(), width: w, height: h, filter, texture: buffer.clone() });
        }
        Ok(Some(Texture { ctx: self.clone(), buffer, pin }))
    }
}

#[cfg(test)]
mod tests {
    use luma_core::TextureFilter;

    use crate::array::Array;
    use crate::kernel::{uniform, Kernel, KernelArg};

    const W: usize = 128;
    const H: usize = 5;

    /// Every fetch of the texture kernel at texel (x, y): the load, the four texels of the
    /// gather (WGSL order) at the corner `(x + 1, y + 1)`, and the sample at the texel's centre.
    const SOURCE: &str = "
struct P { w: u32, h: u32 }
@group(1) @binding(0) var src: texture_2d<f32>;
@group(1) @binding(1) var smp: sampler;
@group(1) @binding(2) var<storage, read_write> out: array<f32>;
@group(1) @binding(3) var<uniform> p: P;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p.w * p.h) { return; }
    let x = i % p.w;
    let y = i / p.w;
    let wf = f32(p.w);
    let hf = f32(p.h);
    let l = textureLoad(src, vec2<i32>(i32(x), i32(y)), 0).x;
    let g = textureGather(0, src, smp, vec2<f32>((f32(x) + 1.0) / wf, (f32(y) + 1.0) / hf));
    let s = textureSampleLevel(src, smp, vec2<f32>((f32(x) + 0.5) / wf, (f32(y) + 0.5) / hf), 0.0).x;
    out[i * 6u] = l;
    out[i * 6u + 1u] = g.x;
    out[i * 6u + 2u] = g.y;
    out[i * 6u + 3u] = g.z;
    out[i * 6u + 4u] = g.w;
    out[i * 6u + 5u] = s;
}
";

    #[derive(crate::encase::ShaderType)]
    struct P {
        w: u32,
        h: u32,
    }

    /// The kernel as luma builds it (a naga module on wgpu, KIR on CUDA) and as KIR the backend
    /// lowers itself (what a kernel the JIT builds takes), both read the texture the same way.
    #[test]
    fn texture_fetches_agree_with_the_texels_on_every_lowering() {
        let ctx = crate::context::default_context().unwrap();
        if ctx.hal().texture_caps().is_none() {
            return;
        }
        let img: Vec<f32> = (0..W * H).map(|i| (i * 7 % 101) as f32 - 50.0).collect();
        let a = Array::from_slice(&img, [W, H]).unwrap();
        let Some(texture) = ctx.texture_2d(&a, W, H, TextureFilter::Nearest).unwrap() else { return };
        let at = |x: i64, y: i64| img[x.clamp(0, W as i64 - 1) as usize + W * y.clamp(0, H as i64 - 1) as usize];
        let mut want = Vec::new();
        for y in 0..H as i64 {
            for x in 0..W as i64 {
                // The load, the gather in WGSL order (x, y+1), (x+1, y+1), (x+1, y), (x, y), the
                // sample at the texel's centre.
                want.extend([at(x, y), at(x, y + 1), at(x + 1, y + 1), at(x + 1, y), at(x, y), at(x, y)]);
            }
        }
        let params = P { w: W as u32, h: H as u32 };
        let direct = ctx.kernel(SOURCE).entry("main").build().unwrap();
        let (_, lowered, ..) = luma_core::kir::parse_wgsl(SOURCE, "main", None, ctx.features(), ctx.hal().info().limits.max_workgroup_size, ctx.hal().info().limits.max_workgroup_invocations, ctx.hal().info().limits.max_dynamic_uniforms, true).unwrap();
        let core = crate::kernel::build_module(&ctx, lowered, Some("texture test"), luma_core::KernelTrust::Trusted).unwrap();
        let lowered = Kernel::from_core(&ctx, std::sync::Arc::new(core));
        for (name, kernel) in [("direct", direct), ("lowered", lowered)] {
            let mut out = Array::<f32>::zeros_on(&ctx, W * H * 6).unwrap();
            kernel.launch([(W * H) as u32], [KernelArg::texture(&texture), KernelArg::texture(&texture), KernelArg::from(&mut out), uniform(&params)]).unwrap();
            assert_eq!(out.to_vec().unwrap(), want, "{name}");
        }
    }
}
