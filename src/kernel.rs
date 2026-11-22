//! Custom compute kernels: user WGSL run against [`Array`]s and uniforms. An `Array<bool>` binds as
//! `array<u32>` of bit-packed words: element `i` is bit `i % 32` of word `i / 32`
//! (`(a[i >> 5u] >> (i & 31u)) & 1u`), and the bits past the end of the array are 0.

use std::sync::Arc;

use luma_core::kir::{self, KernelModule};
use luma_core::naga;
use luma_core::{Backend, BackendError, CompiledKernel, KernelTrust, LineSolveSpec, MatmulSpec, RawKernel, GemmSpec};
pub use luma_core::{BindingInfo, BindingKind};

use crate::array::{Array, Failure};
use crate::context::{default_context, Context, RawArg};
use crate::gpu::pager::Pin;
use crate::error::{LumaError, Result};
use crate::gpu_type::{GpuType, ScalarType};

/// Arguments a launch validates into a stack array; more than this take one heap allocation.
const INLINE_ARGS: usize = 16;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct KernelKey {
    source: String,
    entry: String,
    workgroup: Option<[u32; 3]>,
    /// Part of the key so that a cached kernel is named as the caller asked: the profile lists
    /// the label of the kernel it launched.
    label: Option<String>,
    /// Whether the author vouched for the kernel's bounds (see [`KernelBuilder::unchecked_bounds`]).
    trust: KernelTrust,
}

impl KernelKey {
    pub(crate) fn new(source: String, entry: &str, workgroup: Option<[u32; 3]>) -> Self {
        KernelKey { source, entry: entry.to_owned(), workgroup, label: None, trust: KernelTrust::Checked }
    }

    /// The key of a kernel of luma's own, entry point `main`, named `label`, whose every storage
    /// access is in bounds by construction (see [`KernelTrust::Trusted`]).
    pub(crate) fn trusted(source: String, label: &str) -> Self {
        KernelKey { source, entry: "main".to_owned(), workgroup: None, label: Some(label.to_owned()), trust: KernelTrust::Trusted }
    }
}

/// Everything about a built kernel that does not depend on the caller; cached per context.
pub(crate) type KernelCore = CompiledKernel;

/// Builds a [`Kernel`]; created by [`Context::kernel`].
#[derive(Debug)]
pub struct KernelBuilder<'a> {
    ctx: &'a Context,
    source: String,
    entry: String,
    workgroup: Option<[u32; 3]>,
    label: Option<String>,
    trust: KernelTrust,
}

impl<'a> KernelBuilder<'a> {
    pub(crate) fn new(ctx: &'a Context, source: &str) -> Self {
        KernelBuilder {
            ctx,
            source: source.to_owned(),
            entry: "main".to_owned(),
            workgroup: None,
            label: None,
            trust: KernelTrust::Checked,
        }
    }

    /// Name of the `@compute` entry point. Defaults to `"main"`. Only the bindings this entry
    /// point uses become the kernel's arguments (see [`Kernel`]).
    pub fn entry(mut self, name: &str) -> Self {
        self.entry = name.to_owned();
        self
    }

    /// Asserts the workgroup size. By default it is read from the shader's `@workgroup_size`;
    /// if given, it must equal it.
    pub fn workgroup(mut self, size: [u32; 3]) -> Self {
        self.workgroup = Some(size);
        self
    }

    /// Debug label for the GPU objects. It also names the kernel in a
    /// [`Profile`](crate::Profile); the default is the entry point name.
    pub fn label(mut self, label: &str) -> Self {
        self.label = Some(label.to_owned());
        self
    }

    /// Builds the kernel without bounds checks on its memory accesses.
    ///
    /// By default every access to a storage array, workgroup array or local array is checked, so
    /// that a bug in a kernel cannot read or write outside the memory it was given (a read out
    /// of range yields zero or another element, a write out of range is dropped). The checks
    /// cost a few percent of a tight kernel, which is why the compiler leaves out those it can
    /// prove unnecessary (a tiled loop over workgroup memory, for example). A kernel that is
    /// known to stay in range, benchmarked or tested, can skip the rest. Workgroup memory
    /// still starts zeroed, integer division by zero and loops keep their guards.
    ///
    /// # Safety
    ///
    /// Every access of every invocation must be within the bounds of the array it names
    /// (`arrayLength` for storage arrays, the declared length for workgroup and function
    /// arrays). An access outside them is undefined behavior: it may read or overwrite other
    /// arrays, crash the process or hang the device. Kernels built on the same source with and
    /// without this call are separate kernels.
    pub unsafe fn unchecked_bounds(mut self) -> Self {
        self.trust = KernelTrust::UncheckedBounds;
        self
    }

    /// Parses and validates the shader, reflects its bindings and creates (or fetches from the
    /// context's cache) the compute pipeline.
    pub fn build(self) -> Result<Kernel> {
        let key = KernelKey {
            source: self.source,
            entry: self.entry,
            workgroup: self.workgroup,
            label: self.label,
            trust: self.trust,
        };
        let label = key.label.as_deref();
        let (core, _) = self.ctx.kernel_cache().get_or_build(&key, || build_core(self.ctx, &key, label, key.trust).map(Arc::new))?;
        Ok(Kernel { ctx: self.ctx.clone(), core })
    }
}

/// Compiles `key` into a kernel of the context's backend. `KernelTrust::Trusted` kernels are luma's own,
/// whose every storage access is in bounds by construction; the backend compiles them without
/// per-access bounds checks.
///
/// The WGSL is parsed, validated and lowered once (see [`kir::parse_wgsl`]); the backend that
/// takes naga modules (wgpu) is handed the module it validated, any other the lowered
/// [`KernelModule`].
pub(crate) fn build_core(ctx: &Context, key: &KernelKey, label: Option<&str>, trust: KernelTrust) -> Result<KernelCore> {
    build_core_native(ctx, key, label, trust, None)
}

/// What a backend may run as native code instead of compiling the shader that describes it (the
/// shader is still parsed for the bindings).
#[derive(Clone, Copy)]
pub(crate) enum Native {
    /// A line solve, see [`LineSolveSpec`]; the shader is the fallback.
    LineSolve(LineSolveSpec),
    /// A matrix product, see [`MatmulSpec`]; its shader is not the product of the tile the spec
    /// names, so the backend has to compile it.
    Matmul(MatmulSpec),
    /// A matrix product on tensor cores, see [`GemmSpec`]; like [`Native::Matmul`] the backend
    /// has to compile it.
    Gemm(GemmSpec),
    /// A batched matrix product, see [`Backend::compile_batched_matmul`]; the shader is the
    /// fallback.
    BatchedMatmul,
}

/// [`build_core`] where the backend may run `native` as native code instead of compiling the shader.
pub(crate) fn build_core_native(
    ctx: &Context,
    key: &KernelKey,
    label: Option<&str>,
    trust: KernelTrust,
    native: Option<Native>,
) -> Result<KernelCore> {
    let info = ctx.info();
    let limits = &info.limits;
    if let Some(dir) = std::env::var_os("LUMA_WGPU_DUMP").filter(|v| !v.is_empty()) {
        let dir = std::path::Path::new(&dir);
        let name: String = label.unwrap_or(&key.entry).chars().take(100).map(|c| if c.is_ascii_alphanumeric() || c == '_' { c } else { '_' }).collect();
        let _ = std::fs::create_dir_all(dir);
        let _ = std::fs::write(dir.join(format!("{name}-{:016x}.wgsl", luma_core::hal::cache::fnv1a(key.source.as_bytes()))), &key.source);
    }
    let (module, lowered, bindings, declared, used) = kir::parse_wgsl(
        &key.source,
        &key.entry,
        key.workgroup,
        info.features,
        limits.max_workgroup_size,
        limits.max_workgroup_invocations,
        limits.max_dynamic_uniforms,
        trust == KernelTrust::Trusted,
    )?;
    let workgroup = lowered.workgroup_size;
    let name = label.unwrap_or(&key.entry);
    let backend = ctx.hal();
    let requested = native;
    // A native kernel takes its buffers by position, so every binding of the shader must be bound.
    let all_bound = used.as_deref().is_none_or(|used| used.iter().all(|&bound| bound));
    let native = native.filter(|_| all_bound).and_then(|spec| match spec {
        Native::LineSolve(spec) => backend.compile_line_solve(&spec),
        Native::Matmul(spec) => backend.compile_matmul(&spec),
        Native::Gemm(spec) => backend.compile_gemm(&spec),
        Native::BatchedMatmul => backend.compile_batched_matmul(),
    });
    if native.is_none() && matches!(requested, Some(Native::Matmul(_) | Native::Gemm(_))) {
        return Err(BackendError::Unsupported("the backend cannot compile the native matrix product it planned".into()).into());
    }
    let raw = match native {
        Some(raw) => raw,
        None => match backend.compile_naga(&module, &key.entry, &bindings, workgroup, name, trust) {
            Some(raw) => raw?,
            None => {
                // A cooperative matrix module is lowered to its bindings and no body: only a backend
                // that takes the naga module can run it.
                let cooperative = module.types.iter().any(|(_, t)| matches!(t.inner, naga::TypeInner::CooperativeMatrix { .. }));
                if cooperative && lowered.entry_body.is_empty() {
                    return Err(BackendError::Unsupported("cooperative matrix requires the wgpu backend".into()).into());
                }
                compile(backend, lowered, used.clone(), name, trust)?
            }
        },
    };
    Ok(CompiledKernel::new(Arc::from(name), raw, bindings, declared, used, workgroup))
}

/// Compiles `module` on `backend`. Luma's own (`KernelTrust::Trusted`) kernels compile in the background where
/// the backend can, so the kernels of a traced step compile side by side; the first launch waits
/// for the kernel and reports a compile error. A caller's kernel compiles at once: building it is
/// where its error belongs.
fn compile(backend: &dyn Backend, module: KernelModule, used: Option<Box<[bool]>>, name: &str, trust: KernelTrust) -> std::result::Result<RawKernel, BackendError> {
    if trust == KernelTrust::Trusted {
        backend.compile_kernel_deferred(module, used, name, trust)
    } else {
        backend.compile_kernel(&module, used.as_deref(), name, trust)
    }
}

/// Compiles a kernel luma built as a [`KernelModule`] (see `gpu::lower`), skipping the text.
pub(crate) fn build_module(
    ctx: &Context,
    module: KernelModule,
    label: Option<&str>,
    trust: KernelTrust,
) -> Result<KernelCore> {
    let (declared, used) = kir::used_bindings(&module);
    let bindings = module
        .bindings
        .iter()
        .enumerate()
        .filter(|&(at, _)| used.as_ref().is_none_or(|used| used[at]))
        .map(|(_, binding)| binding.info())
        .collect();
    let name: Arc<str> = Arc::from(label.unwrap_or(&module.entry));
    let workgroup = module.workgroup_size;
    let raw = compile(ctx.hal(), module, used.clone(), &name, trust)?;
    Ok(CompiledKernel::new(name, raw, bindings, declared, used, workgroup))
}

/// A problem size in invocations: `[x]`, `[x, y]` or `[x, y, z]` (missing dimensions are 1).
pub trait ProblemSize: sealed::SizeSealed {
    /// The problem size as `[x, y, z]`, with missing dimensions set to 1.
    #[doc(hidden)]
    fn into_size(self) -> [u32; 3];
}

mod sealed {
    pub trait SizeSealed {}
    pub trait UniformSealed {}
}

impl sealed::SizeSealed for [u32; 1] {}
impl sealed::SizeSealed for [u32; 2] {}
impl sealed::SizeSealed for [u32; 3] {}

impl ProblemSize for [u32; 1] {
    fn into_size(self) -> [u32; 3] {
        [self[0], 1, 1]
    }
}

impl ProblemSize for [u32; 2] {
    fn into_size(self) -> [u32; 3] {
        [self[0], self[1], 1]
    }
}

impl ProblemSize for [u32; 3] {
    fn into_size(self) -> [u32; 3] {
        self
    }
}

enum ArgRepr<'a> {
    Storage {
        ctx: &'a Context,
        /// Keeps the buffer on the device until the launch is recorded.
        pin: Pin,
        bytes: u64,
        element_size: u32,
        writable: bool,
    },
    /// A strided view: the buffer and its `LumaView` descriptor, two consecutive bindings.
    View {
        ctx: &'a Context,
        pin: Pin,
        bytes: u64,
        element_size: u32,
        writable: bool,
        desc: crate::view::RawViewDesc,
    },
    Uniform(std::result::Result<UniformBytes, String>),
    /// The array passed here is poisoned; launching fails with its error.
    Poisoned(Arc<LumaError>),
}

/// Uniform values up to this many bytes are kept inside the [`KernelArg`], so passing one does
/// not allocate.
const INLINE_UNIFORM_BYTES: usize = 128;

/// The encoded bytes of a uniform argument.
#[doc(hidden)]
pub enum UniformBytes {
    Inline { len: usize, bytes: [u8; INLINE_UNIFORM_BYTES] },
    Heap(Vec<u8>),
}

impl UniformBytes {
    fn as_slice(&self) -> &[u8] {
        match self {
            UniformBytes::Inline { len, bytes } => bytes.get(..*len).unwrap_or_default(),
            UniformBytes::Heap(bytes) => bytes,
        }
    }
}

/// One kernel argument, bound to `@group(1)` at the position it has in [`args!`](crate::args).
///
/// Build it from `&Array` (read storage), `&mut Array` (read_write storage) or [`uniform`].
pub struct KernelArg<'a> {
    repr: ArgRepr<'a>,
}

fn storage_arg<'a, T: GpuType>(array: &'a Array<T>, writable: bool) -> KernelArg<'a> {
    if writable && array.is_read_view() {
        if let Some(error) = array.error() {
            return KernelArg { repr: ArgRepr::Poisoned(Arc::new(error.clone())) };
        }
        return KernelArg {
            repr: ArgRepr::Poisoned(Arc::new(LumaError::InvalidShape(
                "cannot pass a read-only view as &mut Array; use slice_mut, flip_mut or transpose_mut".into(),
            ))),
        };
    }
    if writable
        && let Ok(storage) = array.storage()
        && storage.is_alias()
    {
        return KernelArg { repr: ArgRepr::Poisoned(Arc::new(LumaError::ReadOnly("the state passed to an iteration step"))) };
    }
    if writable {
        // Lazy operations recorded earlier still read the contents the kernel is about to change.
        array.flush_readers();
    }
    let repr = match array.ready() {
        Ok((pin, bytes)) => ArgRepr::Storage {
            ctx: array.context(),
            pin,
            bytes,
            element_size: T::LAYOUT.stride() as u32,
            writable,
        },
        Err(Failure::Inherited(error)) => ArgRepr::Poisoned(error),
        Err(Failure::New(error)) => ArgRepr::Poisoned(Arc::new(error)),
    };
    KernelArg { repr }
}

fn view_arg<'a>(
    ctx: &'a Context,
    element_size: u32,
    writable: bool,
    ready: std::result::Result<(Pin, u64, crate::view::RawViewDesc), Failure>,
) -> KernelArg<'a> {
    let repr = match ready {
        Ok((pin, bytes, desc)) => ArgRepr::View { ctx, pin, bytes, element_size, writable, desc },
        Err(Failure::Inherited(error)) => ArgRepr::Poisoned(error),
        Err(Failure::New(error)) => ArgRepr::Poisoned(Arc::new(error)),
    };
    KernelArg { repr }
}

impl<'a, T: GpuType> From<&'a Array<T>> for KernelArg<'a> {
    fn from(array: &'a Array<T>) -> Self {
        storage_arg(array, false)
    }
}

impl<'a, T: GpuType> From<&'a mut Array<T>> for KernelArg<'a> {
    fn from(array: &'a mut Array<T>) -> Self {
        storage_arg(array, true)
    }
}

impl<'a, 'b, T: ScalarType> From<&'a crate::view::ViewMut<'b, Array<T>>> for KernelArg<'a> {
    fn from(view: &'a crate::view::ViewMut<'b, Array<T>>) -> Self {
        view_arg(view.context(), T::LAYOUT.stride() as u32, false, view.ready_view(false))
    }
}

impl<'a, 'b, T: ScalarType> From<&'a mut crate::view::ViewMut<'b, Array<T>>> for KernelArg<'a> {
    fn from(view: &'a mut crate::view::ViewMut<'b, Array<T>>) -> Self {
        view_arg(view.context(), T::LAYOUT.stride() as u32, true, view.ready_view(true))
    }
}

/// Passes a strided read view of `array` (for example `array.slice(0, 1..5).flip(1)`) without
/// materializing a copy. It binds two consecutive `@group(1)` slots: the buffer
/// (`var<storage, read> a: array<T>`) and a `var<uniform> a_view: LumaView`, indexed with
/// `luma_view_idx{,2,3,4}`.
pub fn view<'a, T: GpuType>(array: &'a Array<T>) -> KernelArg<'a> {
    view_arg(array.context(), T::LAYOUT.stride() as u32, false, array.ready_view())
}

/// A value [`uniform`] can pass: any type encase can lay out as a WGSL uniform
/// (`#[derive(encase::ShaderType)]`, scalars, vectors, matrices, arrays). Sealed.
pub trait UniformValue: sealed::UniformSealed {
    #[doc(hidden)]
    fn encode_uniform(&self) -> std::result::Result<UniformBytes, String>;
}

impl<T: encase::ShaderType + encase::internal::WriteInto> sealed::UniformSealed for T {}

impl<T: encase::ShaderType + encase::internal::WriteInto> UniformValue for T {
    fn encode_uniform(&self) -> std::result::Result<UniformBytes, String> {
        let size = self.size().get();
        let encoded = match usize::try_from(size) {
            Ok(len) if len <= INLINE_UNIFORM_BYTES => {
                let mut buffer = encase::UniformBuffer::new([0u8; INLINE_UNIFORM_BYTES]);
                buffer
                    .write(self)
                    .map(|()| UniformBytes::Inline { len, bytes: buffer.into_inner() })
            }
            _ => {
                let mut buffer = encase::UniformBuffer::new(Vec::<u8>::new());
                buffer.write(self).map(|()| UniformBytes::Heap(buffer.into_inner()))
            }
        };
        encoded.map_err(|e| e.to_string())
    }
}

/// Passes `value` as a uniform, laid out by encase according to WGSL's rules (so a `#[derive(
/// encase::ShaderType)]` struct matches the shader's struct even where `repr(C)` would not).
/// The value is copied at call time, and every launch of a batch keeps its own copy.
pub fn uniform<'a, T: UniformValue>(value: &T) -> KernelArg<'a> {
    KernelArg { repr: ArgRepr::Uniform(value.encode_uniform()) }
}

/// Builds the arguments for [`Kernel::launch`]: `args![&x, &mut y, uniform(&params)]`. The
/// result is an array on the stack, so it does not allocate.
#[macro_export]
macro_rules! args {
    () => {
        [] as [$crate::KernelArg<'static>; 0]
    };
    ($($arg:expr),+ $(,)?) => {
        [$($crate::KernelArg::from($arg)),+]
    };
}

/// A compiled custom compute kernel.
///
/// Luma binds a uniform `luma: LumaGrid` (`size: vec3<u32>`, the problem size) at
/// `@group(0) @binding(0)` and provides `luma_idx2(i, j)` / `luma_idx3(i, j, k)` (row-major,
/// x fastest) in every kernel. The kernel's own resources must be in `@group(1)`, numbered from 0
/// in the order of the arguments passed to [`Kernel::launch`].
///
/// Only the globals the chosen entry point uses (directly or through the functions it calls)
/// count: one WGSL module can hold several entry points that use different bindings, each built
/// with [`KernelBuilder::entry`]. The bindings of an entry point must be numbered contiguously
/// from 0 among *its* bindings; globals it does not use may be numbered, grouped or typed
/// arbitrarily.
///
/// Launches join the context's automatic batch: they are recorded in order with the operators
/// and math functions on arrays and run when the batch is submitted (at the next
/// [`Array::to_vec`], [`Array::eval`], [`Context::sync`] or [`Context::flush`], or when the batch
/// grows large).
///
/// Kernels are cheap to clone; building the same source twice on a context reuses the pipeline.
#[derive(Clone, Debug)]
pub struct Kernel {
    ctx: Context,
    core: Arc<KernelCore>,
}

impl Kernel {
    /// Builds a kernel on the [default context](crate::default_context) with the default entry
    /// point `main`. Use [`Context::kernel`] to choose the context or other options.
    pub fn new(wgsl_src: &str) -> Result<Kernel> {
        default_context()?.kernel(wgsl_src).build()
    }

    /// The reflected `@group(1)` bindings, in argument order.
    pub fn bindings(&self) -> &[BindingInfo] {
        &self.core.bindings
    }

    /// The context the kernel was built on.
    pub(crate) fn context(&self) -> &Context {
        &self.ctx
    }

    /// The kernel's workgroup size.
    pub fn workgroup(&self) -> [u32; 3] {
        self.core.workgroup
    }

    /// Runs the kernel with at least `problem_size` invocations (`ceil(size / workgroup)`
    /// workgroups per dimension; `luma.size` is `problem_size`, so guard against
    /// `global_invocation_id >= luma.size`). Records the launch without waiting.
    pub fn launch<'a>(
        &self,
        problem_size: impl ProblemSize,
        args: impl AsRef<[KernelArg<'a>]>,
    ) -> Result<()> {
        let size = problem_size.into_size();
        if size.contains(&0) {
            return Err(LumaError::InvalidShape(format!(
                "Problem size {size:?} has a zero dimension"
            )));
        }
        let wg = self.core.workgroup;
        let [(sx, wx), (sy, wy), (sz, wz)] = [(size[0], wg[0]), (size[1], wg[1]), (size[2], wg[2])];
        let workgroups = [sx.div_ceil(wx), sy.div_ceil(wy), sz.div_ceil(wz)];
        self.run(workgroups, size, args.as_ref())
    }

    /// Runs exactly `workgroups` workgroups; `luma.size` is `workgroups * workgroup`.
    pub fn launch_raw<'a>(
        &self,
        workgroups: [u32; 3],
        args: impl AsRef<[KernelArg<'a>]>,
    ) -> Result<()> {
        if workgroups.contains(&0) {
            return Err(LumaError::InvalidShape(format!(
                "Workgroup count {workgroups:?} has a zero dimension"
            )));
        }
        let wg = self.core.workgroup;
        // Within the device limit (checked in `run`) this cannot overflow; saturate otherwise.
        let [x, y, z] = workgroups;
        let size = [x.saturating_mul(wg[0]), y.saturating_mul(wg[1]), z.saturating_mul(wg[2])];
        self.run(workgroups, size, args.as_ref())
    }

    fn run(&self, workgroups: [u32; 3], size: [u32; 3], args: &[KernelArg<'_>]) -> Result<()> {
        let ctx = &self.ctx;
        let core = &*self.core;

        let max = ctx.max_workgroups_per_dimension();
        if workgroups.iter().any(|&w| w > max) {
            return Err(LumaError::KernelGridTooLarge { workgroups, max });
        }
        let total: usize = args.iter().map(|a| if matches!(a.repr, ArgRepr::View { .. }) { 2 } else { 1 }).sum();
        if total != core.bindings.len() {
            return Err(LumaError::KernelArgCount { expected: core.bindings.len(), actual: total });
        }

        // The validated arguments, on the stack unless there are very many.
        let mut inline = [RawArg::Uniform(&[]); INLINE_ARGS];
        let mut heap;
        let raw: &mut [RawArg<'_>] = match inline.get_mut(..total) {
            Some(raw) => raw,
            None => {
                heap = vec![RawArg::Uniform(&[]); total];
                &mut heap
            }
        };

        let max_uniform = ctx.max_uniform_bytes();
        let mut index = 0;
        for arg in args {
            let info = &core.bindings[index];
            match &arg.repr {
                ArgRepr::Poisoned(error) => return Err(LumaError::clone(error)),
                ArgRepr::View { ctx: arg_ctx, pin, bytes, element_size, writable, desc } => {
                    if !arg_ctx.same_as(ctx) {
                        return Err(LumaError::ContextMismatch);
                    }
                    check_storage(info, index, *element_size, *writable)?;
                    let desc_info = &core.bindings[index + 1];
                    if desc_info.kind != BindingKind::Uniform {
                        return Err(LumaError::KernelArgKind {
                            index: index + 1,
                            expected: desc_info.kind,
                            actual: BindingKind::Uniform,
                        });
                    }
                    let desc_bytes = std::mem::size_of::<crate::view::RawViewDesc>() as u64;
                    if let Some(expected) = desc_info.min_size
                        && desc_bytes < u64::from(expected) {
                            return Err(LumaError::KernelArgUniformSize {
                                index: index + 1,
                                expected: u64::from(expected),
                                actual: desc_bytes,
                            });
                        }
                    if desc_bytes > max_uniform {
                        return Err(LumaError::BufferTooLarge { requested: desc_bytes, max: max_uniform });
                    }
                    raw[index] = RawArg::Storage { buffer: pin.buffer(), bytes: *bytes, writable: *writable };
                    raw[index + 1] = RawArg::Uniform(bytemuck::bytes_of(desc));
                    index += 2;
                    continue;
                }
                ArgRepr::Storage { ctx: arg_ctx, pin, bytes, element_size, writable } => {
                    if !arg_ctx.same_as(ctx) {
                        return Err(LumaError::ContextMismatch);
                    }
                    check_storage(info, index, *element_size, *writable)?;
                    raw[index] = RawArg::Storage { buffer: pin.buffer(), bytes: *bytes, writable: *writable };
                }
                ArgRepr::Uniform(encoded) => {
                    if info.kind != BindingKind::Uniform {
                        return Err(LumaError::KernelArgKind {
                            index,
                            expected: info.kind,
                            actual: BindingKind::Uniform,
                        });
                    }
                    let data = encoded
                        .as_ref()
                        .map_err(|message| LumaError::KernelArgUniform {
                            index,
                            message: message.clone(),
                        })?
                        .as_slice();
                    let bytes = data.len() as u64;
                    if let Some(expected) = info.min_size
                        && bytes < u64::from(expected) {
                            return Err(LumaError::KernelArgUniformSize {
                                index,
                                expected: u64::from(expected),
                                actual: bytes,
                            });
                        }
                    if bytes > max_uniform {
                        return Err(LumaError::BufferTooLarge { requested: bytes, max: max_uniform });
                    }
                    raw[index] = RawArg::Uniform(data);
                }
            }
            index += 1;
        }

        ctx.launch(core, workgroups, size, raw)
    }
}

/// Checks a storage argument against the binding it lands on.
fn check_storage(info: &BindingInfo, index: usize, element_size: u32, writable: bool) -> Result<()> {
    let actual = if writable { BindingKind::StorageReadWrite } else { BindingKind::StorageRead };
    match info.kind {
        BindingKind::Uniform => {
            return Err(LumaError::KernelArgKind { index, expected: info.kind, actual })
        }
        BindingKind::StorageReadWrite if !writable => return Err(LumaError::KernelArgAccess { index }),
        _ => {}
    }
    if let Some(stride) = info.element_stride
        && stride != element_size {
            return Err(LumaError::KernelArgElementSize { index, expected: stride, actual: element_size });
        }
    Ok(())
}
