//! The epilogue hook of the tensor-core products (W3-U4 part C) at the backend: the identity hook and `relu(acc + bias[col])` against
//! the unhooked product bit for bit, and (with `--ignored`) their cost against the unhooked product alone and followed by the separate
//! elementwise kernel, the numbers the spec's acceptance asks for.
//!
//! `cargo +stable test --release -p luma-hal-cuda --test gemm_hook -- --ignored --nocapture`

use std::mem::MaybeUninit;
use std::sync::Arc;
use std::time::Duration;

use luma_core::kir::{BuiltinVar, IrBinOp, KernelModule, KirType, ModuleBuilder, used_bindings};
use luma_core::{Backend, BindingInfo, BindingKind, BufferUsage, CompiledKernel, Dispatch, GemmLanes, GemmSpec, HookId, KernelTrust, RawArg};
use luma_cuda::CudaBackend;

fn backend() -> Option<CudaBackend> {
    match CudaBackend::new() {
        Ok(backend) if backend.gemm_hooks() => Some(backend),
        Ok(_) => None,
        Err(error) => {
            println!("SKIPPED: no CUDA device ({error})");
            None
        }
    }
}

struct Rng(u64);

impl Rng {
    fn vec(&mut self, n: usize) -> Vec<f32> {
        (0..n)
            .map(|_| {
                self.0 ^= self.0 << 13;
                self.0 ^= self.0 >> 7;
                self.0 ^= self.0 << 17;
                ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
            })
            .collect()
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Identity,
    ReluBias,
}

/// The hook module: function 0 is `luma_epi(acc, row, col, n)`, binding 0 the bias when there is one.
fn hook(kind: Kind) -> KernelModule {
    let mut mb = ModuleBuilder::new("luma_epi", "main", [1, 1, 1]);
    let bias = (kind == Kind::ReluBias).then(|| mb.add_storage_read("e0", KirType::F32));
    let sink = mb.add_storage_read_write("sink", KirType::F32);
    let params = [("acc", KirType::F32), ("row", KirType::U32), ("col", KirType::U32), ("n", KirType::U32)];
    let f = mb.add_function("luma_epi", &params, Some(KirType::F32), |fb| {
        let acc = fb.param(0);
        let value = match bias {
            Some(bias) => {
                let b = fb.load(bias, fb.param(2));
                let sum = fb.add(acc, b);
                fb.binary(IrBinOp::Max, sum, fb.lit_f32(0.0))
            }
            None => acc,
        };
        fb.ret(Some(value));
    });
    let mut fb = mb.entry_builder();
    let (z, zu) = (fb.lit_f32(0.0), fb.lit_u32(0));
    let v = fb.call(f, &[z, zu, zu, zu], KirType::F32);
    fb.store(sink, zu, v);
    fb.finish();
    mb.finish()
}

/// `out[i] = max(c[i] + bias[i % n], 0)`: the separate elementwise pass.
fn elementwise() -> KernelModule {
    let mut mb = ModuleBuilder::new("ew", "main", [256, 1, 1]);
    let out = mb.add_storage_read_write("out", KirType::F32);
    let c = mb.add_storage_read("c", KirType::F32);
    let bias = mb.add_storage_read("bias", KirType::F32);
    let p = mb.add_uniform_u32_struct("Params", &["n", "len"]);
    let p = mb.add_uniform("p", KirType::Struct(p));
    let mut fb = mb.entry_builder();
    let gid = fb.builtin(BuiltinVar::GlobalInvocationId);
    let nwg = fb.builtin(BuiltinVar::NumWorkgroups);
    let (gx, gy, nx) = (fb.field(gid, 0, KirType::U32), fb.field(gid, 1, KirType::U32), fb.field(nwg, 0, KirType::U32));
    let i = gx + gy * nx * fb.lit_u32(256);
    let i = fb.let_val("i", i);
    let (n, len) = (fb.uniform_field(p, 0, KirType::U32), fb.uniform_field(p, 1, KirType::U32));
    let inside = fb.lt(i, len);
    fb.if_then(inside, |fb| {
        let sum = fb.add(fb.load(c, i), fb.load(bias, fb.rem(i, n)));
        let v = fb.binary(IrBinOp::Max, sum, fb.lit_f32(0.0));
        fb.store(out, i, v);
    });
    fb.finish();
    mb.finish()
}

fn wait(raw: &luma_core::RawKernel) {
    let start = std::time::Instant::now();
    while !raw.is_ready() {
        assert!(start.elapsed() < Duration::from_secs(120), "the kernel never finished compiling");
        std::thread::sleep(Duration::from_millis(1));
    }
}

fn binding(name: &str, kind: BindingKind) -> BindingInfo {
    BindingInfo { name: name.into(), kind, element_stride: Some(4), min_size: None }
}

fn gemm(backend: &CudaBackend, spec: &GemmSpec, hook: Option<&Arc<KernelModule>>, operands: usize) -> CompiledKernel {
    let started = std::time::Instant::now();
    let raw = backend.compile_gemm(spec, hook).expect("the backend plans the product");
    wait(&raw);
    if hook.is_some() {
        println!("  compile {:?} + hook: {:.1} ms", (spec.rows, spec.cols), started.elapsed().as_secs_f64() * 1e3);
    }
    let mut bindings = vec![binding("c", BindingKind::StorageReadWrite), binding("a", BindingKind::StorageRead), binding("b", BindingKind::StorageRead)];
    bindings.extend((0..operands).map(|k| binding(&format!("e{k}"), BindingKind::StorageRead)));
    bindings.push(BindingInfo { min_size: Some(32), ..binding("u", BindingKind::Uniform) });
    let declared = bindings.len();
    CompiledKernel::new("gemm".into(), raw, bindings, declared, None, [1, 1, 1])
}

struct Bufs {
    a: luma_core::Buffer,
    b: luma_core::Buffer,
    c: luma_core::Buffer,
    bias: luma_core::Buffer,
    out: luma_core::Buffer,
    host_bias: Vec<f32>,
}

fn bufs(backend: &CudaBackend, (m, n, k): (usize, usize, usize)) -> Bufs {
    let bytes = |v: &[f32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    let host_bias = rng.vec(n);
    Bufs {
        a: backend.alloc_filled(&bytes(&rng.vec(m * k)), u64::MAX).unwrap(),
        b: backend.alloc_filled(&bytes(&rng.vec(k * n)), u64::MAX).unwrap(),
        c: backend.create_buffer((m * n * 4) as u64, BufferUsage::STORAGE, false).unwrap(),
        bias: backend.alloc_filled(&bytes(&host_bias), u64::MAX).unwrap(),
        out: backend.create_buffer((m * n * 4) as u64, BufferUsage::STORAGE, false).unwrap(),
        host_bias,
    }
}

fn launch_gemm(backend: &CudaBackend, kernel: &CompiledKernel, spec: &GemmSpec, (m, n, k): (usize, usize, usize), b: &Bufs, with_bias: bool, tag: Option<u64>) {
    let mut uniform = Vec::new();
    for word in [m, n, k, k, k, n, 0, 0] {
        uniform.extend_from_slice(&(word as u32).to_le_bytes());
    }
    let grid = [n.div_ceil(spec.cols as usize) as u32, m.div_ceil(spec.rows as usize) as u32, 1];
    let mut args = vec![RawArg::write(&b.c, b.c.size()), RawArg::read(&b.a, b.a.size()), RawArg::read(&b.b, b.b.size())];
    if with_bias {
        args.push(RawArg::read(&b.bias, b.bias.size()));
    }
    args.push(RawArg::Uniform(&uniform));
    if let Some(tag) = tag {
        backend.time_next_launch(tag);
    }
    backend.record_launch(kernel, Dispatch::Direct(grid), [grid[0] * spec.cols, grid[1] * spec.rows, 1], &args).expect("the launch is recorded");
}

fn read(backend: &CudaBackend, buffer: &luma_core::Buffer, n: usize) -> Vec<f32> {
    backend.sync(Duration::ZERO).unwrap();
    let mut out = vec![MaybeUninit::<u8>::uninit(); n * 4];
    backend.read_into(buffer, &mut out, false, Duration::ZERO).expect("the result");
    out.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(c.map(|byte| unsafe { byte.assume_init() }))).collect()
}

fn tf32_tiles() -> [(u32, u32); 2] {
    [(128, 128), (64, 64)]
}

#[test]
fn the_hooked_product_is_the_unhooked_one_then_the_elementwise_pass_bit_for_bit() {
    let Some(backend) = backend() else { return };
    for (rows, cols) in tf32_tiles() {
        for lanes in [GemmLanes::Vec4, GemmLanes::Vec4Tails, GemmLanes::Scalar] {
            for (m, n, k) in [(128, 256, 192), (100, 130, 72), (33, 65, 17)] {
                if lanes == GemmLanes::Vec4 && (m % 4 != 0 || n % 4 != 0 || k % 4 != 0) {
                    continue;
                }
                let spec = GemmSpec::new(rows, cols, false, false, false).with_lanes(lanes);
                let b = bufs(&backend, (m, n, k));
                let plain = gemm(&backend, &spec, None, 0);
                launch_gemm(&backend, &plain, &spec, (m, n, k), &b, false, None);
                let expected = read(&backend, &b.c, m * n);
                // Identity hook: the plain bits. Relu with a bias: the host's `max(acc + bias, 0)`.
                for kind in [Kind::Identity, Kind::ReluBias] {
                    let module = Arc::new(hook(kind));
                    let spec = spec.with_hook(HookId(kind as u64 + 1));
                    let operands = usize::from(kind == Kind::ReluBias);
                    let kernel = gemm(&backend, &spec, Some(&module), operands);
                    launch_gemm(&backend, &kernel, &spec, (m, n, k), &b, kind == Kind::ReluBias, None);
                    let got = read(&backend, &b.c, m * n);
                    for (at, (g, e)) in got.iter().zip(&expected).enumerate() {
                        let want = if kind == Kind::Identity { *e } else { (e + b.host_bias[at % n]).max(0.0) };
                        assert_eq!(g.to_bits(), want.to_bits(), "{rows}x{cols} {lanes:?} {:?} element {at}", (m, n, k));
                    }
                }
            }
        }
    }
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

#[test]
#[ignore = "a measurement"]
fn cost_of_the_hook() {
    let Some(backend) = backend() else { return };
    if !backend.can_time_launches() {
        return;
    }
    println!("device: {}", backend.info().name);
    let ew_module = elementwise();
    let (declared, used) = used_bindings(&ew_module);
    let ew_raw = backend.compile_kernel(&ew_module, used.as_deref(), "ew", KernelTrust::Trusted).unwrap();
    let ew_bindings: Vec<BindingInfo> = ew_module.bindings.iter().map(|b| b.info()).collect();
    let ew = CompiledKernel::new("ew".into(), ew_raw, ew_bindings, declared, used, [256, 1, 1]);
    println!("tile      shape            unhooked   +ew      identity  relu+bias  G2(identity/U)  G3(relu/(U+ew))  bound(U/(U+ew))");
    for (m, n, k) in [(256, 512, 512), (256, 784, 512), (2048, 2048, 2048), (4096, 4096, 4096)] {
        for (rows, cols) in tf32_tiles() {
            let spec = GemmSpec::new(rows, cols, false, false, false);
            let b = bufs(&backend, (m, n, k));
            let plain = gemm(&backend, &spec, None, 0);
            let id_spec = spec.with_hook(HookId(1));
            let id = gemm(&backend, &id_spec, Some(&Arc::new(hook(Kind::Identity))), 0);
            let relu_spec = spec.with_hook(HookId(2));
            let relu = gemm(&backend, &relu_spec, Some(&Arc::new(hook(Kind::ReluBias))), 1);
            let len = (m * n) as u32;
            let mut uniform = Vec::new();
            for word in [n as u32, len] {
                uniform.extend_from_slice(&word.to_le_bytes());
            }
            let ew_groups = [len.div_ceil(256), 1, 1];
            let launch_ew = |tag: u64| {
                let args = [RawArg::write(&b.out, b.out.size()), RawArg::read(&b.c, b.c.size()), RawArg::read(&b.bias, b.bias.size()), RawArg::Uniform(&uniform)];
                backend.time_next_launch(tag);
                backend.record_launch(&ew, Dispatch::Direct(ew_groups), [len, 1, 1], &args).unwrap();
            };
            // Tags: 0 unhooked, 1 elementwise, 2 identity, 3 relu. Interleaved samples, the order rotating.
            let rounds = if m * n * k >= 1 << 30 { 40 } else { 200 };
            let mut samples: [Vec<f64>; 4] = Default::default();
            for round in 0..rounds {
                for step in 0..4 {
                    match (step + round) % 4 {
                        0 => launch_gemm(&backend, &plain, &spec, (m, n, k), &b, false, Some(0)),
                        1 => launch_ew(1),
                        2 => launch_gemm(&backend, &id, &id_spec, (m, n, k), &b, false, Some(2)),
                        _ => launch_gemm(&backend, &relu, &relu_spec, (m, n, k), &b, true, Some(3)),
                    }
                    backend.sync(Duration::ZERO).unwrap();
                    let mut out = Vec::new();
                    backend.drain_launch_times(&mut out);
                    for (tag, time) in out {
                        samples[tag as usize].push(time.as_secs_f64() * 1e6);
                    }
                }
            }
            let [u, e, i, r] = samples.each_mut().map(|s| median(s));
            println!(
                "{rows:>3}x{cols:<3} {m:>5}x{n:>4}x{k:<5} {u:>9.1} {e:>8.1} {i:>9.1} {r:>9.1}  {:>10.3}  {:>14.3}  {:>14.3}",
                i / u,
                r / (u + e),
                u / (u + e)
            );
        }
    }
}
