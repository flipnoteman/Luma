//! Texture objects (`CUtexObject`) over the data of a buffer.
//!
//! A texture reads the source buffer in place (a pitched linear resource, no copy) when its base
//! address is 512-byte aligned and its rows are a multiple of 32 bytes apart, which an image of
//! 8-aligned width on a fresh buffer is; otherwise the texels are copied once into a CUDA array.
//! Coordinates are normalized and addressing is clamp-to-edge, as WGSL's.

use super::*;

/// Bytes the base address of a pitched texture resource is aligned to.
const BASE_ALIGN: u64 = 512;
/// Bytes the row pitch of a pitched texture resource is a multiple of.
const PITCH_ALIGN: u64 = 32;

/// A texture object and what keeps its data alive.
#[derive(Debug)]
pub(super) struct CudaTexture {
    object: sys::CUtexObject,
    /// The buffer a pitched texture reads; it lives as long as the texture.
    _source: Option<Buffer>,
    /// The `CUarray` a copied texture reads (as an address: raw pointers are not `Send`).
    array: usize,
    key: u64,
    ctx: Arc<CudaContext>,
}

impl NativeBuffer for CudaTexture {}

impl CudaTexture {
    /// Identifies the data the texture reads among everything a launch touches.
    pub(super) fn key(&self) -> u64 {
        self.key
    }
}

impl Drop for CudaTexture {
    fn drop(&mut self) {
        if bind_raw(&self.ctx).is_ok() {
            // SAFETY: created by `make_texture`, destroyed once; a launch holds the texture until
            // it has run.
            unsafe {
                let _ = sys::cuTexObjectDestroy(self.object);
                if self.array != 0 {
                    let _ = sys::cuArrayDestroy(self.array as sys::CUarray);
                }
            }
        }
    }
}

/// The handle a kernel takes for the texture `buffer`.
pub(super) fn object(buffer: &Buffer) -> Result<u64, BackendError> {
    buffer.downcast_ref::<CudaTexture>().map(|t| t.object).ok_or_else(|| BackendError::KernelBinding("a texture binding was given something that is not a CUDA texture".into()))
}

/// What 2-D textures of the device can do.
pub(super) fn caps(attribute: &dyn Fn(sys::CUdevice_attribute) -> Result<i32, BackendError>) -> Option<TextureCaps> {
    use sys::CUdevice_attribute as A;
    let dim = |a: A| attribute(a).ok().filter(|&v| v > 0).map(|v| v as u32);
    let max_dim = [
        A::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH,
        A::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT,
        A::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_WIDTH,
        A::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_HEIGHT,
        A::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_WIDTH,
        A::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_HEIGHT,
    ]
    .into_iter()
    .map(dim)
    .collect::<Option<Vec<u32>>>()?
    .into_iter()
    .min()?;
    Some(TextureCaps { max_dim, pitch_align: PITCH_ALIGN as u32, aliases: true })
}

impl CudaBackend {
    pub(super) fn make_texture(&self, source: &Buffer, offset: u64, pitch: u64, desc: &TextureDesc) -> Result<Buffer, BackendError> {
        let caps = self.shared.textures.ok_or_else(|| BackendError::Unsupported("the device reports no 2-D textures".into()))?;
        let (w, h) = (u64::from(desc.width), u64::from(desc.height));
        let texel = u64::from(desc.format.texel_bytes());
        if w == 0 || h == 0 || desc.width > caps.max_dim || desc.height > caps.max_dim {
            return Err(BackendError::Unsupported(format!("a {w}x{h} texture (up to {0}x{0} are possible)", caps.max_dim)));
        }
        let row = w * texel;
        if pitch < row || pitch % u64::from(caps.pitch_align) != 0 || offset + pitch * (h - 1) + row > source.size() {
            return Err(BackendError::KernelBinding(format!("a {w}x{h} texture does not fit {} bytes at offset {offset} with rows {pitch} bytes apart", source.size())));
        }
        let base = device_ptr(source)?.checked_add(offset).ok_or_else(|| BackendError::KernelBinding("texture offset overflows".into()))?;
        self.bind()?;

        let format = sys::CUarray_format::CU_AD_FORMAT_FLOAT;
        let channels = desc.format.channels();
        let max_pitch = self.shared.ctx.attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_PITCH).map_err(gpu)? as u64;
        let pitched = base % BASE_ALIGN == 0 && pitch % PITCH_ALIGN == 0 && pitch <= max_pitch;
        let (resource, array, key, keep) = if pitched {
            let res = sys::CUDA_RESOURCE_DESC {
                resType: sys::CUresourcetype::CU_RESOURCE_TYPE_PITCH2D,
                res: sys::CUDA_RESOURCE_DESC_st__bindgen_ty_1 {
                    pitch2D: sys::CUDA_RESOURCE_DESC_st__bindgen_ty_1__bindgen_ty_4 { devPtr: base, format, numChannels: channels, width: w as usize, height: h as usize, pitchInBytes: pitch as usize },
                },
                flags: 0,
            };
            (res, 0usize, device_ptr(source)?, Some(source.clone()))
        } else {
            let mut array: sys::CUarray = std::ptr::null_mut();
            let ad = sys::CUDA_ARRAY_DESCRIPTOR { Width: w as usize, Height: h as usize, Format: format, NumChannels: channels };
            // SAFETY: `ad` is a valid descriptor and `array` receives the handle.
            unsafe { sys::cuArrayCreate_v2(&mut array, &ad) }.result().map_err(gpu)?;
            let copy = sys::CUDA_MEMCPY2D {
                srcXInBytes: 0,
                srcY: 0,
                srcMemoryType: sys::CUmemorytype::CU_MEMORYTYPE_DEVICE,
                srcHost: std::ptr::null(),
                srcDevice: base,
                srcArray: std::ptr::null_mut(),
                srcPitch: pitch as usize,
                dstXInBytes: 0,
                dstY: 0,
                dstMemoryType: sys::CUmemorytype::CU_MEMORYTYPE_ARRAY,
                dstHost: std::ptr::null_mut(),
                dstDevice: 0,
                dstArray: array,
                dstPitch: 0,
                WidthInBytes: row as usize,
                Height: h as usize,
            };
            // SAFETY: the source rows lie inside `source` (checked above); the copy is ordered
            // on the stream before every launch that reads the texture, and `source` is held
            // until the batch has run.
            let queued = unsafe { sys::cuMemcpy2DAsync_v2(&copy, self.stream.cu_stream()) }.result().map_err(gpu);
            if let Err(error) = queued {
                // SAFETY: the array was created above and is not in use.
                unsafe { sys::cuArrayDestroy(array) };
                return Err(error);
            }
            let res = sys::CUDA_RESOURCE_DESC {
                resType: sys::CUresourcetype::CU_RESOURCE_TYPE_ARRAY,
                res: sys::CUDA_RESOURCE_DESC_st__bindgen_ty_1 { array: sys::CUDA_RESOURCE_DESC_st__bindgen_ty_1__bindgen_ty_1 { hArray: array } },
                flags: 0,
            };
            (res, array as usize, (array as u64) | 1 << 63, None)
        };
        let clamp = sys::CUaddress_mode::CU_TR_ADDRESS_MODE_CLAMP;
        let filter = match desc.filter {
            TextureFilter::Nearest => sys::CUfilter_mode::CU_TR_FILTER_MODE_POINT,
            TextureFilter::Linear => sys::CUfilter_mode::CU_TR_FILTER_MODE_LINEAR,
        };
        let tex = sys::CUDA_TEXTURE_DESC {
            addressMode: [clamp; 3],
            filterMode: filter,
            flags: sys::CU_TRSF_NORMALIZED_COORDINATES,
            maxAnisotropy: 1,
            mipmapFilterMode: sys::CUfilter_mode::CU_TR_FILTER_MODE_POINT,
            mipmapLevelBias: 0.0,
            minMipmapLevelClamp: 0.0,
            maxMipmapLevelClamp: 0.0,
            borderColor: [0.0; 4],
            reserved: [0; 12],
        };
        let mut object: sys::CUtexObject = 0;
        // SAFETY: valid descriptors; `object` receives the handle.
        let created = unsafe { sys::cuTexObjectCreate(&mut object, &resource, &tex, std::ptr::null()) }.result().map_err(gpu);
        if let Err(error) = created {
            if array != 0 {
                // SAFETY: created above, unused by any object.
                unsafe { sys::cuArrayDestroy(array as sys::CUarray) };
            }
            return Err(error);
        }
        let texture = Buffer::new(row * h, BufferUsage::TEXTURE, CudaTexture { object, _source: keep, array, key, ctx: Arc::clone(&self.shared.ctx) });
        if array != 0 {
            // The copy reads `source` and fills the array on the stream: both stay until it ran.
            let mut st = lock(&self.state);
            st.held.buffers.push(source.clone());
            st.held.buffers.push(texture.clone());
            st.dirty = true;
        }
        Ok(texture)
    }
}
