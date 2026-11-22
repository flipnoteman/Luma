# HANDOFF TexUnits (branch gap-tex, worktree ~/Documents/lp-tex, base 0edb1ed)

Task: texture-unit support (HAL + CUDA tex objects + wgpu textures), exact mode (point/gather fetches, results unchanged)
default, hardware-filtered mode opt-in (documented ~1/256 weight precision), route ops where it pays, afcompare signal rows
before/after on both GPUs. Files I own: src/sampler.rs, src/image/warp.rs, src/interp.rs, HAL texture parts, luma-cuda,
luma-wgpu (texture parts), tools/afcompare/src/signal.rs.

## Measured (baseline, before any change) -- afcompare signal_approx + signal_image, AFCOMPARE_KERNELS=1
Binary: ~/Documents/lp-tex-bin/base (same path on remote 192.168.1.88). Outputs: handoff-data/base-3090-*.md (laptop was unreachable by ssh at handoff time; its baseline is the tail in handoff-data/baseline-afcompare-job-5220.log and full files on the laptop under ~/Documents/lp-tex-bin/out-base-*/results.md),
base-3080L-*.md (result tables per GPU, never mix), logs handoff-data/baseline-afcompare-job-*.log (gpuq jobs
0-20261122-105430-31641 = 3090, -5220 = 3080 Laptop; both finished status 0).
3090 rows to close (AF/luma): approx2 bicubic 1080p 0.83 / 4K 0.82 (luma GPU 0.220 ms vs AF 0.184 at 4K), approx2
linear/nearest 0.95-0.98, approx1 1M-4M 0.92-0.98, resize bilinear 1080p->4K 0.91 (RGB 0.91).
Back-of-envelope: 4K approx2 nearest moves ~133 MB => ~0.14-0.16 ms at 3090 bandwidth; luma 0.163 ms GPU is already near it, so
nearest/linear gaps (1-3 us) are launch/latency, texture unlikely to help. Bicubic (16 taps, runtime `method` branching in a
uniform => one big kernel; sampler.rs `source()`) is compute/L1 bound: candidates: (a) specialise kernel per method/kind via WGSL
consts (not yet tried; owned by sampler.rs), (b) tld4 gather (exact, 4 gathers for 4x4), (c) hw-bilinear opt-in (4 fetches B-spline trick;
for AF `Cubic` weights w1,w2 >= 0 so middle pair can be one bilinear fetch => 3x3=9 fetches; Catmull-Rom same). MEASURE before building routing.

## Design decided (HAL)
* A texture is returned as a `Buffer` whose native is the backend texture (so RawArg/record.rs/cpu/metal need NO changes);
  only bindings of kind `BindingKind::Texture2d` / `BindingKind::Sampler` accept it. The same texture Buffer is passed for BOTH
  the `texture_2d<f32>` binding and its `sampler` binding (sampler state = TextureDesc.filter, clamp-to-edge).
* `Backend::texture_caps() -> Option<TextureCaps{max_dim,pitch_align,aliases}>`, `Backend::create_texture(source:&Buffer, offset, pitch, &TextureDesc) -> Result<Buffer>`
  (defaults: None / Unsupported). CPU/metal: unsupported => buffer path.
* KIR: `Expr::Texture{op: TextureOp::{Load,Gather(comp),Sample}, binding, coords}` -> vec4<f32>; from_wgsl lowers naga ImageLoad
  (textureLoad, vec2<i32|u32> coords) / ImageSample (textureGather, textureSampleLevel level 0) for `texture_2d<f32>` + non-comparison `sampler`.
  Sampler bindings are reflected (BindingDecl elem U32) but have no KIR use.
* Gather order is NOT what the doc comment in kir/ir.rs says: WGSL/Vulkan/CUDA order is x=(i,j+1) y=(i+1,j+1) z=(i+1,j) w=(i,j) -- VERIFY by test
  then fix the TextureOp::Gather doc. Gather at integer texel coord (fx+1, fy+1)/dims is robust (0.5-texel margin) for exact 2x2 footprint fx..fx+1.
* wgpu: needs Features FLOAT32_FILTERABLE (layout Float{filterable:true} + Filtering sampler for all texture kernels); copy_buffer_to_texture needs
  bytes_per_row multiple of 256 (caps.pitch_align=256; root repacks via a luma kernel when not); bind groups with textures: skip the bind-group cache (build per launch).
  Plan was to delegate wgpu to a subagent once HAL+CUDA contract exists.
* CUDA: cuTexObjectCreate/Destroy are NOT bound in cudarc 0.19 sys (structs CUDA_RESOURCE_DESC/CUDA_TEXTURE_DESC exist): dlsym from libcuda via libloading.
  Linear pitch2D resource aliasing the source when pointer 512-aligned and pitch % 32 == 0 (zero copy: 1080p/4K widths qualify), else cuArrayCreate + cuMemcpy2DAsync.
  Texture object flags: CU_TRSF_NORMALIZED_COORDINATES, clamp, filter per desc. Kernel param `i64 %texN` (CudaArgAbi::Texture), sampler binding => no param (CudaArgAbi::Unbound,
  must be skipped in Params::build and in the ptrs loop). Emission: `llvm.nvvm.tex.unified.2d.v4f32.{s32,f32}` and `llvm.nvvm.tld4.unified.{r,g,b,a}.2d.v4f32.f32`
  returning {float x4}; inttoptr the i64 at kernel entry into cx.binds (helper functions take `i8*`), ptrtoint at use. UNVERIFIED that libnvvm accepts these intrinsics and that
  s32 loads ignore normalization -- first thing to test (tiny luma-cuda test under gpuq).

## Done (committed as wip:)
* crates/luma-core/src/hal/texture.rs + hal/mod.rs (BindingKind::{Texture2d,Sampler}, Backend::texture_caps/create_texture), lib.rs exports.
* kir/ir.rs (TextureOp, Expr::Texture + child walks), kir/opt.rs type arm, kir/from_wgsl.rs (reflect_kind(module,var), bindings, used_bindings, ImageLoad/ImageSample lowering, image_binding, uses counting). `cargo check -p luma-hal-core` passes.
* Baseline afcompare numbers (above).

## NOT done / build is currently broken for other crates
Adding BindingKind variants + Expr::Texture breaks exhaustive matches in: luma-cuda (codegen/mod.rs scan/expr_ty/expr/kernel param emission, conv.rs `uniform()`, backend.rs Params::build),
luma-wgpu (backend.rs build() layout entries, codegen.rs expr_ty/expr), luma-metal (codegen.rs:310,1180 + expr), luma-cpu (codegen.rs expr match: reject with Unsupported). Fix those (cpu/metal arms are outside my file list: report to Main).
Then: CudaTexture + create_texture + tex-object dlsym; root crate: src/texture.rs (pub Texture wrapper, KernelArg::from(&Texture) => RawArg::Storage with the texture Buffer, kernel.rs check_storage must accept
Texture2d/Sampler kinds for it), Context caps; sampler.rs texture WGSL variant (load/gather/sample), interp opts for filtered mode (opt-in, documented precision), routing by measurement; tests (exact == buffer path on every backend, filtered within tolerance);
afcompare after-rows on both GPUs; CHANGELOG/guide docs; gates (clippy+nextest CPU, gpuq cuda+wgpu tests on both hosts).

## Next steps (in order)
1. Re-read this file; `cargo check --workspace` and fix the exhaustive matches listed above (stub Unsupported for cpu/metal/wgpu-codegen).
2. CUDA: dlsym tex object fns, CudaTexture, create_texture, codegen for Expr::Texture/params, test via a tiny gpuq job (verify intrinsics, gather order, s32 load clamp, aliasing alignment). Use /usr/local/cuda-13.4/bin/ncu on the 3090 (and /opt/cuda/bin on laptop) now that profilers exist to see whether L1/TEX pipe or instruction issue bounds bicubic.
3. Experiment first (ncu on current sampler bicubic 4K; method-specialised kernel variant) before routing decisions; then wgpu (delegate), root API, routing, tests, afcompare after-run (copy binary to ~/Documents/lp-tex-bin/new + rsync to remote, interleaved ABBA vs base), docs, gates.
No gpuq jobs are queued.
