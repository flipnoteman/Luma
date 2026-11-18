//! Whether a device's `fma` is a fused multiply-add: the `Df32` library takes the exact error of a
//! product from it (see `builtin::df_library`), which is only exact when `fma(a, b, -p)` rounds
//! once. Some compilers and software rasterizers lower `fma` to a multiplication and an
//! addition; there the library must split the factors instead.

use std::mem::MaybeUninit;
use std::sync::Arc;
use std::time::Duration;

use luma_core::kir::{used_bindings, KernelModule, KirType, ModuleBuilder};
use luma_core::{Backend, BufferUsage, CompiledKernel, Dispatch, RawArg};

/// `out[i] = fma(a[i], b[i], -p)` with `p` the rounded product made opaque to the compiler by the
/// bits of `zero[i]` (all zero): a fused `fma` returns the rounding error of the product, a
/// multiplication and an addition return 0.
fn module() -> KernelModule {
    let mut mb = ModuleBuilder::new("luma fma", "main", [1, 1, 1]);
    // The error-free transformation needs the arithmetic exactly as written.
    mb.set_uses_exact_f32(true);
    let a = mb.add_storage_read("a", KirType::F32);
    let b = mb.add_storage_read("b", KirType::F32);
    let zero = mb.add_storage_read("zero", KirType::U32);
    let out = mb.add_storage_read_write("out", KirType::F32);
    let mut f = mb.entry_builder();
    let g = f.builtin(luma_core::kir::BuiltinVar::GlobalInvocationId);
    let i = f.field(g, 0, KirType::U32);
    let i = f.let_val("i", i);
    let (av, bv) = (f.load(a, i), f.load(b, i));
    let (av, bv) = (f.let_val("av", av), f.let_val("bv", bv));
    let prod = f.mul(av, bv);
    let bits = f.bitcast(prod, KirType::U32);
    let masked = f.bit_or(bits, f.load(zero, i));
    let p = f.bitcast(masked, KirType::F32);
    let p = f.let_val("p", p);
    let neg = f.neg(p);
    let r = f.fma(av, bv, neg);
    f.store(out, i, r);
    f.finish();
    mb.finish()
}

/// Factor pairs of 24 significant bits each, whose products need more than 24 (so that the
/// rounding error of the product is not zero): a fixed pseudo-random sequence.
fn pairs() -> Vec<(f32, f32)> {
    let mut state = 0x2545_f491_4f6c_dd1d_u64;
    let mut next = move || {
        state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        // 24 random bits in [1, 2): the mantissa of an f32.
        f32::from_bits(0x3f80_0000 | ((state >> 40) as u32 & 0x007f_ffff))
    };
    (0..64).map(|_| (next(), next())).collect()
}

/// Runs the exactness test on `backend`: `Some(true)` when every `fma(a, b, -p)` returned exactly
/// the rounding error of `a * b`. `None` when the test could not run (buffer, compile, launch or
/// readback failure), so the caller does not keep a transient failure.
pub(crate) fn probe(backend: &dyn Backend) -> Option<bool> {
    let pairs = pairs();
    let expected: Vec<f32> = pairs.iter().map(|&(a, b)| (f64::from(a) * f64::from(b) - f64::from(a * b)) as f32).collect();
    if expected.iter().all(|&e| e == 0.0) {
        return Some(false);
    }
    let bytes = (pairs.len() * 4) as u64;
    let max_buffer_bytes = backend.info().limits.max_buffer_bytes;
    let a: Vec<f32> = pairs.iter().map(|p| p.0).collect();
    let b: Vec<f32> = pairs.iter().map(|p| p.1).collect();
    let zero = vec![0u32; pairs.len()];
    let upload = |data: &[u8]| {
        if backend.info().zero_copy {
            return backend.alloc_filled(data, max_buffer_bytes).ok();
        }
        let buffer = backend.create_buffer(data.len() as u64, BufferUsage::STORAGE | BufferUsage::COPY_DST, false)?;
        backend.record_upload(&buffer, data, max_buffer_bytes).ok()?;
        Some(buffer)
    };
    let (buf_a, buf_b, buf_zero) = (upload(bytemuck::cast_slice(&a))?, upload(bytemuck::cast_slice(&b))?, upload(bytemuck::cast_slice(&zero))?);
    let out = backend.create_buffer(bytes, BufferUsage::STORAGE | BufferUsage::COPY_SRC | BufferUsage::COPY_DST, false)?;

    let module = module();
    let (declared, used) = used_bindings(&module);
    let bindings = module
        .bindings
        .iter()
        .enumerate()
        .filter(|&(at, _)| used.as_ref().is_none_or(|used| used[at]))
        .map(|(_, binding)| binding.info())
        .collect();
    let raw = backend.compile_kernel(&module, used.as_deref(), "luma fma", true).ok()?;
    let kernel = CompiledKernel::new(Arc::from("luma fma"), raw, bindings, declared, used, module.workgroup_size);

    let count = pairs.len() as u32;
    backend
        .record_launch(
            &kernel,
            Dispatch::Direct([count, 1, 1]),
            [count, 1, 1],
            &[RawArg::read(&buf_a, bytes), RawArg::read(&buf_b, bytes), RawArg::read(&buf_zero, bytes), RawArg::write(&out, bytes)],
        )
        .ok()?;
    backend.flush_batch().ok()?;
    let mut got = vec![MaybeUninit::<u8>::uninit(); bytes as usize];
    backend.read_into(&out, &mut got, false, Duration::from_micros(500)).ok()?;
    // SAFETY: `read_into` returned Ok, so every byte of `got` was written.
    let got: Vec<u8> = got.into_iter().map(|byte| unsafe { byte.assume_init() }).collect();
    let got: Vec<f32> = got.as_chunks::<4>().0.iter().map(|&word| f32::from_ne_bytes(word)).collect();
    Some(got.iter().zip(&expected).all(|(g, e)| g.to_bits() == e.to_bits()))
}

#[cfg(test)]
mod tests {
    use crate::{Context, ContextOptions};

    #[test]
    fn the_test_is_repeatable_and_the_context_uses_its_answer() {
        let ctx = Context::with_options(ContextOptions::from_env().calibrate(false)).unwrap();
        let first = super::probe(ctx.hal());
        assert_eq!(first, super::probe(ctx.hal()));
        if std::env::var_os("LUMA_DF_FMA").is_none() {
            // The stored answer of an earlier process may be used instead of a new test, but both are one device.
            assert_eq!(Some(ctx.tuning().df_fma), first);
        }
        let forced = Context::with_options(ContextOptions::from_env().calibrate(false).df_fma(false)).unwrap();
        assert!(!forced.tuning().df_fma);
    }
}
