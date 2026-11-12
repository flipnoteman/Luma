//! Custom compute kernels: user WGSL run against [`Array`]s and uniforms.

use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use crate::array::{Array, Failure};
use crate::context::{default_context, Context, RawArg};
use crate::gpu::pager::Pin;
use crate::error::{LumaError, Result};
use crate::gpu_type::GpuType;

/// WGSL appended to every kernel. Declarations are order independent in WGSL, so appending (not
/// prepending) keeps the line numbers of compile errors identical to the user's source and
/// leaves `enable` directives first.
const PRELUDE: &str = "
struct LumaGrid { size: vec3<u32>, _pad: u32 }
@group(0) @binding(0) var<uniform> luma: LumaGrid;
fn luma_idx2(i: u32, j: u32) -> u32 { return i + j * luma.size.x; }
fn luma_idx3(i: u32, j: u32, k: u32) -> u32 { return i + luma.size.x * (j + luma.size.y * k); }
";

/// Arguments a launch validates into a stack array; more than this take one heap allocation.
const INLINE_ARGS: usize = 16;

/// What a kernel binding in `@group(1)` is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BindingKind {
    /// `var<storage, read>`: bound from `&Array`.
    StorageRead,
    /// `var<storage, read_write>`: bound from `&mut Array`.
    StorageReadWrite,
    /// `var<uniform>`: bound from [`uniform`].
    Uniform,
}

impl fmt::Display for BindingKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            BindingKind::StorageRead => "read-only storage buffer",
            BindingKind::StorageReadWrite => "read_write storage buffer",
            BindingKind::Uniform => "uniform",
        })
    }
}

/// One reflected `@group(1)` binding of a kernel; its index is the argument position.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BindingInfo {
    /// The variable's name in the shader.
    pub name: String,
    /// Whether the binding is a read-only storage buffer, a read-write storage buffer or a uniform.
    pub kind: BindingKind,
    /// For storage bindings of `array<T>`: the byte stride of `T`. `None` when not an array.
    pub element_stride: Option<u32>,
    /// For uniforms: the byte size of the uniform's type. `None` for storage bindings.
    pub min_size: Option<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct KernelKey {
    source: String,
    entry: String,
    workgroup: Option<[u32; 3]>,
}

impl KernelKey {
    pub(crate) fn new(source: String, entry: &str, workgroup: Option<[u32; 3]>) -> Self {
        KernelKey { source, entry: entry.to_owned(), workgroup }
    }
}

/// Everything about a built kernel that does not depend on the caller; cached per context.
#[derive(Debug, Clone)]
pub(crate) struct KernelCore {
    /// Unique per built kernel; identifies it in the bind group cache.
    pub(crate) id: u64,
    /// Names the kernel in a [`Profile`](crate::Profile).
    pub(crate) label: Arc<str>,
    pub(crate) pipeline: wgpu::ComputePipeline,
    pub(crate) args_layout: wgpu::BindGroupLayout,
    bindings: Vec<BindingInfo>,
    pub(crate) workgroup: [u32; 3],
}

static NEXT_KERNEL_ID: AtomicU64 = AtomicU64::new(0);

/// Builds a [`Kernel`]; created by [`Context::kernel`].
#[derive(Debug)]
pub struct KernelBuilder<'a> {
    ctx: &'a Context,
    source: String,
    entry: String,
    workgroup: Option<[u32; 3]>,
    label: Option<String>,
}

impl<'a> KernelBuilder<'a> {
    pub(crate) fn new(ctx: &'a Context, source: &str) -> Self {
        KernelBuilder {
            ctx,
            source: source.to_owned(),
            entry: "main".to_owned(),
            workgroup: None,
            label: None,
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

    /// Parses and validates the shader, reflects its bindings and creates (or fetches from the
    /// context's cache) the compute pipeline.
    pub fn build(self) -> Result<Kernel> {
        let key = KernelKey {
            source: self.source,
            entry: self.entry,
            workgroup: self.workgroup,
        };
        let mut cache = self.ctx.kernel_cache();
        if let Some(core) = cache.get(&key) {
            return Ok(Kernel { ctx: self.ctx.clone(), core: core.clone() });
        }
        let core = Arc::new(build_core(self.ctx, &key, self.label.as_deref(), false)?);
        cache.insert(key, core.clone());
        Ok(Kernel { ctx: self.ctx.clone(), core })
    }
}

fn reflect_kind(space: naga::AddressSpace) -> Option<BindingKind> {
    match space {
        naga::AddressSpace::Uniform => Some(BindingKind::Uniform),
        naga::AddressSpace::Storage { access } if access.contains(naga::StorageAccess::STORE) => {
            Some(BindingKind::StorageReadWrite)
        }
        naga::AddressSpace::Storage { .. } => Some(BindingKind::StorageRead),
        _ => None,
    }
}

/// What naga may accept in a kernel: the shader features the device was opened with.
fn shader_capabilities(features: wgpu::Features) -> naga::valid::Capabilities {
    use naga::valid::Capabilities as Caps;
    let mut caps = Caps::empty();
    if features.contains(wgpu::Features::SUBGROUP) {
        caps |= Caps::SUBGROUP | Caps::SUBGROUP_BARRIER;
    }
    if features.contains(wgpu::Features::SHADER_F16) {
        caps |= Caps::SHADER_FLOAT16;
    }
    if features.contains(wgpu::Features::SHADER_F64) {
        caps |= Caps::FLOAT64;
    }
    if features.contains(wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX) {
        caps |= Caps::COOPERATIVE_MATRIX;
    }
    caps
}

/// Compiles `key` into a pipeline. `trusted` kernels are luma's own, whose every storage access
/// is in bounds by construction; they are compiled without wgpu's per-access bounds checks.
pub(crate) fn build_core(
    ctx: &Context,
    key: &KernelKey,
    label: Option<&str>,
    trusted: bool,
) -> Result<KernelCore> {
    let combined = format!("{}\n{PRELUDE}", key.source);
    let module = naga::front::wgsl::parse_str(&combined)
        .map_err(|e| LumaError::ShaderCompile(e.emit_to_string(&combined)))?;

    let mut validator = naga::valid::Validator::new(naga::valid::ValidationFlags::all(), shader_capabilities(ctx.device().features()));
    let info = validator
        .validate(&module)
        .map_err(|e| LumaError::ShaderCompile(e.emit_to_string(&combined)))?;

    let (entry_index, entry) = module
        .entry_points
        .iter()
        .enumerate()
        .find(|(_, ep)| ep.name == key.entry && ep.stage == naga::ShaderStage::Compute)
        .ok_or_else(|| {
            let available: Vec<&str> = module
                .entry_points
                .iter()
                .filter(|ep| ep.stage == naga::ShaderStage::Compute)
                .map(|ep| ep.name.as_str())
                .collect();
            LumaError::KernelEntryPoint(format!(
                "No @compute entry point `{}`; compute entry points: {available:?}",
                key.entry
            ))
        })?;

    // Bindings: group 0 belongs to luma, user resources must be in group 1 with binding
    // numbers 0..n. Only the globals the entry point uses count (directly or through the
    // functions it calls), so one module can hold several entry points with different bindings.
    let used = info.get_entry_point(entry_index);
    let mut user: Vec<(u32, BindingInfo)> = Vec::new();
    for (handle, var) in module.global_variables.iter() {
        let Some(rb) = var.binding else { continue };
        if used[handle].is_empty() {
            continue;
        }
        let name = var.name.clone().unwrap_or_default();
        if rb.group == 0 {
            if name == "luma" && rb.binding == 0 {
                continue;
            }
            return Err(LumaError::KernelBinding(format!(
                "`{name}` is in @group(0), which luma reserves for the grid uniform; \
                 declare kernel arguments in @group(1)"
            )));
        }
        if rb.group != 1 {
            return Err(LumaError::KernelBinding(format!(
                "`{name}` is in @group({}); kernel arguments must be in @group(1)",
                rb.group
            )));
        }
        let kind = reflect_kind(var.space).ok_or_else(|| {
            LumaError::KernelBinding(format!(
                "`{name}` is neither a storage buffer nor a uniform"
            ))
        })?;
        let inner = &module.types[var.ty].inner;
        let (element_stride, min_size) = match kind {
            BindingKind::Uniform => (None, inner.try_size(module.to_ctx())),
            _ => (
                match inner {
                    naga::TypeInner::Array { stride, .. } => Some(*stride),
                    _ => None,
                },
                None,
            ),
        };
        user.push((rb.binding, BindingInfo { name, kind, element_stride, min_size }));
    }
    user.sort_by_key(|(binding, _)| *binding);
    for (expected, (binding, info)) in user.iter().enumerate() {
        if *binding != expected as u32 {
            return Err(LumaError::KernelBinding(format!(
                "the @group(1) bindings used by entry point `{}` must be numbered 0..n in \
                 argument order, but `{}` is @binding({binding}) where @binding({expected}) \
                 was expected",
                key.entry, info.name
            )));
        }
    }
    let bindings: Vec<BindingInfo> = user.into_iter().map(|(_, info)| info).collect();

    if entry.workgroup_size_overrides.is_some() {
        return Err(LumaError::KernelWorkgroup(
            "workgroup sizes that depend on `override` constants are not supported".into(),
        ));
    }
    let workgroup = entry.workgroup_size;
    if workgroup.contains(&0) {
        return Err(LumaError::KernelWorkgroup(format!(
            "@workgroup_size{workgroup:?} has a zero dimension"
        )));
    }
    if let Some(requested) = key.workgroup
        && requested != workgroup {
            return Err(LumaError::KernelWorkgroup(format!(
                "requested {requested:?} but the shader declares @workgroup_size{workgroup:?}"
            )));
        }
    let limits = ctx.device().limits();
    let per_dim = [
        limits.max_compute_workgroup_size_x,
        limits.max_compute_workgroup_size_y,
        limits.max_compute_workgroup_size_z,
    ];
    let invocations = workgroup.iter().map(|&d| u64::from(d)).product::<u64>();
    if workgroup.iter().zip(per_dim).any(|(&d, max)| d > max)
        || invocations > u64::from(limits.max_compute_invocations_per_workgroup)
    {
        return Err(LumaError::KernelWorkgroup(format!(
            "{workgroup:?} exceeds the device limits (per dimension {per_dim:?}, \
             {} invocations in total)",
            limits.max_compute_invocations_per_workgroup
        )));
    }

    // The grid and every user uniform are bound at dynamic offsets into the uniform arena.
    let uniforms = bindings.iter().filter(|b| b.kind == BindingKind::Uniform).count() as u32;
    if uniforms + 1 > ctx.max_dynamic_uniforms() {
        return Err(LumaError::KernelBinding(format!(
            "{uniforms} uniform arguments exceed the device limit of {} dynamic uniform buffers \
             per kernel (one is used by luma's grid)",
            ctx.max_dynamic_uniforms().saturating_sub(1)
        )));
    }

    let device = ctx.device();
    let arg_entries: Vec<_> = bindings
        .iter()
        .enumerate()
        .map(|(i, info)| wgpu::BindGroupLayoutEntry {
            binding: i as u32,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: match info.kind {
                    BindingKind::Uniform => wgpu::BufferBindingType::Uniform,
                    BindingKind::StorageRead => wgpu::BufferBindingType::Storage { read_only: true },
                    BindingKind::StorageReadWrite => {
                        wgpu::BufferBindingType::Storage { read_only: false }
                    }
                },
                has_dynamic_offset: info.kind == BindingKind::Uniform,
                min_binding_size: None,
            },
            count: None,
        })
        .collect();
    let args_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label,
        entries: &arg_entries,
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label,
        bind_group_layouts: &[Some(ctx.grid_layout()), Some(&args_layout)],
        immediate_size: 0,
    });
    let shader_desc = wgpu::ShaderModuleDescriptor {
        label,
        source: wgpu::ShaderSource::Wgsl(combined.into()),
    };
    let shader = if trusted {
        // Integer division keeps its guards: WGSL defines `x / 0` and `x % 0`, and user data
        // reaches luma's kernels.
        let checks = wgpu::ShaderRuntimeChecks {
            bounds_checks: false,
            force_loop_bounding: false,
            ..wgpu::ShaderRuntimeChecks::checked()
        };
        // SAFETY: luma's built-in and generated kernels index storage buffers only below the
        // lengths they are launched with, which never exceed the bound ranges, and all of their
        // loops are bounded.
        unsafe { device.create_shader_module_trusted(shader_desc, checks) }
    } else {
        device.create_shader_module(shader_desc)
    };
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label,
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some(&key.entry),
        // Trusted kernels write their workgroup memory before reading it.
        compilation_options: wgpu::PipelineCompilationOptions {
            zero_initialize_workgroup_memory: !trusted,
            ..Default::default()
        },
        cache: None,
    });

    Ok(KernelCore {
        id: NEXT_KERNEL_ID.fetch_add(1, Ordering::Relaxed),
        label: Arc::from(label.unwrap_or(&key.entry)),
        pipeline,
        args_layout,
        bindings,
        workgroup,
    })
}

/// A problem size in invocations: `[x]`, `[x, y]` or `[x, y, z]` (missing dimensions are 1).
pub trait ProblemSize {
    /// The problem size as `[x, y, z]`, with missing dimensions set to 1.
    fn into_size(self) -> [u32; 3];
}

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
    Uniform(std::result::Result<UniformBytes, String>),
    /// The array passed here is poisoned; launching fails with its error.
    Poisoned(Arc<LumaError>),
}

/// Uniform values up to this many bytes are kept inside the [`KernelArg`], so passing one does
/// not allocate.
const INLINE_UNIFORM_BYTES: usize = 128;

/// The encoded bytes of a uniform argument.
enum UniformBytes {
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
    if writable {
        // Lazy operations recorded earlier still read the contents the kernel is about to change.
        array.flush_readers();
    }
    let repr = match array.ready() {
        Ok((pin, bytes)) => ArgRepr::Storage {
            ctx: array.context(),
            pin,
            bytes,
            element_size: std::mem::size_of::<T>() as u32,
            writable,
        },
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

/// Passes `value` as a uniform, laid out by encase according to WGSL's rules (so a `#[derive(
/// encase::ShaderType)]` struct matches the shader's struct even where `repr(C)` would not).
/// The value is copied at call time, and every launch of a batch keeps its own copy.
pub fn uniform<'a, T>(value: &T) -> KernelArg<'a>
where
    T: encase::ShaderType + encase::internal::WriteInto,
{
    let size = value.size().get();
    let encoded = match usize::try_from(size) {
        Ok(len) if len <= INLINE_UNIFORM_BYTES => {
            let mut buffer = encase::UniformBuffer::new([0u8; INLINE_UNIFORM_BYTES]);
            buffer
                .write(value)
                .map(|()| UniformBytes::Inline { len, bytes: buffer.into_inner() })
        }
        _ => {
            let mut buffer = encase::UniformBuffer::new(Vec::<u8>::new());
            buffer.write(value).map(|()| UniformBytes::Heap(buffer.into_inner()))
        }
    };
    KernelArg { repr: ArgRepr::Uniform(encoded.map_err(|e| e.to_string())) }
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
        if args.len() != core.bindings.len() {
            return Err(LumaError::KernelArgCount {
                expected: core.bindings.len(),
                actual: args.len(),
            });
        }

        // The validated arguments, on the stack unless there are very many.
        let mut inline = [RawArg::Uniform(&[]); INLINE_ARGS];
        let mut heap;
        let raw: &mut [RawArg<'_>] = match inline.get_mut(..args.len()) {
            Some(raw) => raw,
            None => {
                heap = vec![RawArg::Uniform(&[]); args.len()];
                &mut heap
            }
        };

        let max_uniform = ctx.max_uniform_bytes();
        for (index, ((arg, info), slot)) in
            args.iter().zip(&core.bindings).zip(raw.iter_mut()).enumerate()
        {
            match &arg.repr {
                ArgRepr::Poisoned(error) => return Err(LumaError::clone(error)),
                ArgRepr::Storage { ctx: arg_ctx, pin, bytes, element_size, writable } => {
                    if !arg_ctx.same_as(ctx) {
                        return Err(LumaError::ContextMismatch);
                    }
                    let actual = if *writable {
                        BindingKind::StorageReadWrite
                    } else {
                        BindingKind::StorageRead
                    };
                    match info.kind {
                        BindingKind::Uniform => {
                            return Err(LumaError::KernelArgKind {
                                index,
                                expected: info.kind,
                                actual,
                            })
                        }
                        BindingKind::StorageReadWrite if !writable => {
                            return Err(LumaError::KernelArgAccess { index })
                        }
                        _ => {}
                    }
                    if let Some(stride) = info.element_stride
                        && stride != *element_size {
                            return Err(LumaError::KernelArgElementSize {
                                index,
                                expected: stride,
                                actual: *element_size,
                            });
                        }
                    *slot = RawArg::Storage { buffer: pin.buffer(), bytes: *bytes, writable: *writable };
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
                    *slot = RawArg::Uniform(data);
                }
            }
        }

        ctx.launch(core, workgroups, size, raw)
    }
}
