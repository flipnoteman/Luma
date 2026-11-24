//! Raw-HAL helpers shared by `tests/all/transfer.rs` and `examples/bench_transfer.rs`: opens the
//! backend `LUMA_BACKEND` names and builds the one kernel the transfer-lane checks run.
#![allow(dead_code)]

use std::sync::Arc;

use luma_core::kir::{BuiltinVar, KirType, ModuleBuilder, used_bindings};
use luma_core::{Backend, BackendError, Buffer, BufferUsage, CompiledKernel, DeviceKind, Dispatch, KernelTrust, RawArg};

/// What a buffer of a transfer is for.
pub const USAGE: BufferUsage = BufferUsage::STORAGE.union(BufferUsage::COPY_SRC).union(BufferUsage::COPY_DST);

/// The backend `LUMA_BACKEND` names (`cuda`, `metal`, `wgpu`, `cpu`; unset or `auto`: the first
/// that opens, in that order), `None` when none opens. A named backend that fails to open
/// panics.
pub fn open() -> Option<Box<dyn Backend>> {
    let want = std::env::var("LUMA_BACKEND").ok().map(|v| v.trim().to_ascii_lowercase()).filter(|v| v != "auto");
    let candidates: Vec<&str> = match want.as_deref() {
        Some(name) => vec![name],
        None => vec!["cuda", "metal", "wgpu", "cpu"],
    };
    for name in candidates {
        let opened: Result<Box<dyn Backend>, BackendError> = match name {
            #[cfg(feature = "cuda")]
            "cuda" => luma_cuda::open(),
            #[cfg(feature = "metal")]
            "metal" => luma_metal::open(),
            #[cfg(feature = "wgpu")]
            "wgpu" => luma_wgpu::open(DeviceKind::Gpu),
            "cpu" => luma_cpu::open(),
            other => Err(BackendError::Unsupported(format!("the backend `{other}` is not built"))),
        };
        match opened {
            Ok(backend) => return Some(backend),
            Err(error) if want.is_some() => panic!("LUMA_BACKEND={name}: {error}"),
            Err(_) => {}
        }
    }
    None
}

/// `out[i] = f^iters(inp[i])` with `f(x) = 3x + 1` (wrapping), one invocation per element.
pub fn affine_kernel(backend: &dyn Backend, iters: u32) -> CompiledKernel {
    let mut mb = ModuleBuilder::new("affine", "main", [64, 1, 1]);
    let inp = mb.add_storage_read("inp", KirType::U32);
    let out = mb.add_storage_read_write("out", KirType::U32);
    let mut b = mb.entry_builder();
    let gid = b.builtin(BuiltinVar::GlobalInvocationId);
    let i = b.field(gid, 0, KirType::U32);
    let n = b.grid_size(0);
    let oob = b.ge(i, n);
    b.if_then(oob, |b| b.ret(None));
    let first = b.load(inp, i);
    let x = b.var_decl("x", KirType::U32, Some(first));
    let (zero, end, one) = (b.lit_u32(0), b.lit_u32(iters), b.lit_u32(1));
    b.for_range("k", zero, end, one, |b, _| {
        let (three, one) = (b.lit_u32(3), b.lit_u32(1));
        let scaled = b.mul(b.read_var(x), three);
        let next = b.add(scaled, one);
        b.assign_var(x, next);
    });
    let result = b.read_var(x);
    b.store(out, i, result);
    b.finish();
    let module = mb.finish();
    let (declared, used) = used_bindings(&module);
    let raw = backend.compile_kernel(&module, used.as_deref(), "affine", KernelTrust::Trusted).expect("the kernel compiles");
    let infos = module.bindings.iter().map(|b| b.info()).collect();
    CompiledKernel::new(Arc::from("affine"), raw, infos, declared, used, [64, 1, 1])
}

/// Records the launch of `kernel` over `words` elements, `inp` to `out`.
pub fn launch(backend: &dyn Backend, kernel: &CompiledKernel, inp: &Buffer, out: &Buffer, words: u32) {
    let bytes = u64::from(words) * 4;
    backend
        .record_launch(kernel, Dispatch::Direct([words.div_ceil(64), 1, 1]), [words, 1, 1], &[RawArg::read(inp, bytes), RawArg::write(out, bytes)])
        .expect("the launch records");
}

/// The host value of the kernel for `x`.
pub fn affine_host(mut x: u32, iters: u32) -> u32 {
    for _ in 0..iters {
        x = x.wrapping_mul(3).wrapping_add(1);
    }
    x
}
