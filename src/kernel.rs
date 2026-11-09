//! Custom compute kernels: user WGSL run against [`Array`]s and uniforms.

use std::fmt;

use wgpu::{BindGroupEntry, BindingResource};

use crate::array::Array;
use crate::context::{bind, Context, UNIFORM_USAGE};
use crate::error::{LumaError, Result};
use crate::gpu::pipeline::Scratch;
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

/// Size of `LumaGrid` in bytes.
const GRID_BYTES: u64 = 16;

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

/// Everything about a built kernel that does not depend on the caller; cached per context.
#[derive(Debug)]
pub(crate) struct KernelCore {
    pipeline: wgpu::ComputePipeline,
    grid_layout: wgpu::BindGroupLayout,
    args_layout: wgpu::BindGroupLayout,
    bindings: Vec<BindingInfo>,
    workgroup: [u32; 3],
}

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

    /// Name of the `@compute` entry point. Defaults to `"main"`.
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

    /// Debug label for the GPU objects.
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
        let core = std::sync::Arc::new(build_core(self.ctx, &key, self.label.as_deref())?);
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

fn build_core(ctx: &Context, key: &KernelKey, label: Option<&str>) -> Result<KernelCore> {
    let combined = format!("{}\n{PRELUDE}", key.source);
    let module = naga::front::wgsl::parse_str(&combined)
        .map_err(|e| LumaError::ShaderCompile(e.emit_to_string(&combined)))?;

    // Bindings: group 0 belongs to luma, user resources must be in group 1 with binding
    // numbers 0..n.
    let mut user: Vec<(u32, BindingInfo)> = Vec::new();
    for (_, var) in module.global_variables.iter() {
        let Some(rb) = var.binding else { continue };
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
                "@group(1) bindings must be numbered 0..n in argument order, but `{}` is \
                 @binding({binding}) where @binding({expected}) was expected",
                info.name
            )));
        }
    }
    let bindings: Vec<BindingInfo> = user.into_iter().map(|(_, info)| info).collect();

    let mut validator = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::empty(),
    );
    validator
        .validate(&module)
        .map_err(|e| LumaError::ShaderCompile(e.emit_to_string(&combined)))?;

    let entry = module
        .entry_points
        .iter()
        .find(|ep| ep.name == key.entry && ep.stage == naga::ShaderStage::Compute)
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
    if entry.workgroup_size_overrides.is_some() {
        return Err(LumaError::KernelWorkgroup(
            "workgroup sizes that depend on `override` constants are not supported".into(),
        ));
    }
    let workgroup = entry.workgroup_size;
    if let Some(requested) = key.workgroup {
        if requested != workgroup {
            return Err(LumaError::KernelWorkgroup(format!(
                "requested {requested:?} but the shader declares @workgroup_size{workgroup:?}"
            )));
        }
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

    let device = ctx.device();
    let uniform_entry = |binding: u32, read_only: Option<bool>| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: match read_only {
                Some(read_only) => wgpu::BufferBindingType::Storage { read_only },
                None => wgpu::BufferBindingType::Uniform,
            },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    };
    let grid_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label,
        entries: &[uniform_entry(0, None)],
    });
    let arg_entries: Vec<_> = bindings
        .iter()
        .enumerate()
        .map(|(i, info)| {
            uniform_entry(
                i as u32,
                match info.kind {
                    BindingKind::Uniform => None,
                    BindingKind::StorageRead => Some(true),
                    BindingKind::StorageReadWrite => Some(false),
                },
            )
        })
        .collect();
    let args_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label,
        entries: &arg_entries,
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label,
        bind_group_layouts: &[Some(&grid_layout), Some(&args_layout)],
        immediate_size: 0,
    });
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label,
        source: wgpu::ShaderSource::Wgsl(combined.into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label,
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some(&key.entry),
        compilation_options: Default::default(),
        cache: None,
    });

    Ok(KernelCore { pipeline, grid_layout, args_layout, bindings, workgroup })
}

/// A problem size in invocations: `[x]`, `[x, y]` or `[x, y, z]` (missing dimensions are 1).
pub trait ProblemSize {
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
        buffer: &'a wgpu::Buffer,
        bytes: u64,
        element_size: u32,
        writable: bool,
    },
    Uniform(std::result::Result<Vec<u8>, String>),
}

/// One kernel argument, bound to `@group(1)` at the position it has in [`args!`](crate::args).
///
/// Build it from `&Array` (read storage), `&mut Array` (read_write storage) or [`uniform`].
pub struct KernelArg<'a> {
    repr: ArgRepr<'a>,
}

impl<'a, T: GpuType> From<&'a Array<T>> for KernelArg<'a> {
    fn from(array: &'a Array<T>) -> Self {
        KernelArg {
            repr: ArgRepr::Storage {
                ctx: array.context(),
                buffer: array.storage(),
                bytes: array.byte_len(),
                element_size: std::mem::size_of::<T>() as u32,
                writable: false,
            },
        }
    }
}

impl<'a, T: GpuType> From<&'a mut Array<T>> for KernelArg<'a> {
    fn from(array: &'a mut Array<T>) -> Self {
        KernelArg {
            repr: ArgRepr::Storage {
                ctx: array.context(),
                buffer: array.storage(),
                bytes: array.byte_len(),
                element_size: std::mem::size_of::<T>() as u32,
                writable: true,
            },
        }
    }
}

/// Passes `value` as a uniform, laid out by encase according to WGSL's rules (so a `#[derive(
/// encase::ShaderType)]` struct matches the shader's struct even where `repr(C)` would not).
/// The value is copied at call time.
pub fn uniform<'a, T>(value: &T) -> KernelArg<'a>
where
    T: encase::ShaderType + encase::internal::WriteInto,
{
    let mut buffer = encase::UniformBuffer::new(Vec::<u8>::new());
    let bytes = buffer
        .write(value)
        .map(|()| buffer.into_inner())
        .map_err(|e| e.to_string());
    KernelArg { repr: ArgRepr::Uniform(bytes) }
}

/// Builds the `Vec<KernelArg>` for [`Kernel::dispatch`]: `args![&x, &mut y, uniform(&params)]`.
#[macro_export]
macro_rules! args {
    ($($arg:expr),* $(,)?) => {
        ::std::vec![$($crate::KernelArg::from($arg)),*]
    };
}

/// A compiled custom compute kernel.
///
/// Luma binds a uniform `luma: LumaGrid` (`size: vec3<u32>`, the problem size) at
/// `@group(0) @binding(0)` and provides `luma_idx2(i, j)` / `luma_idx3(i, j, k)` (row-major,
/// x fastest) in every kernel. The kernel's own resources must be in `@group(1)`, numbered from 0
/// in the order of the arguments passed to [`Kernel::dispatch`].
///
/// Kernels are cheap to clone; building the same source twice on a context reuses the pipeline.
#[derive(Clone, Debug)]
pub struct Kernel {
    ctx: Context,
    core: std::sync::Arc<KernelCore>,
}

impl Kernel {
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
    /// `global_invocation_id >= luma.size`). Submits without waiting.
    pub fn dispatch(&self, problem_size: impl ProblemSize, args: Vec<KernelArg<'_>>) -> Result<()> {
        let size = problem_size.into_size();
        if size.contains(&0) {
            return Err(LumaError::InvalidShape(format!(
                "Problem size {size:?} has a zero dimension"
            )));
        }
        let wg = self.core.workgroup;
        let workgroups = std::array::from_fn(|i| size[i].div_ceil(wg[i]));
        self.run(workgroups, size, &args)
    }

    /// Runs exactly `workgroups` workgroups; `luma.size` is `workgroups * workgroup`.
    pub fn dispatch_raw(&self, workgroups: [u32; 3], args: Vec<KernelArg<'_>>) -> Result<()> {
        if workgroups.contains(&0) {
            return Err(LumaError::InvalidShape(format!(
                "Workgroup count {workgroups:?} has a zero dimension"
            )));
        }
        let wg = self.core.workgroup;
        // Within the device limit (checked in `run`) this cannot overflow; saturate otherwise.
        let size = std::array::from_fn(|i| workgroups[i].saturating_mul(wg[i]));
        self.run(workgroups, size, &args)
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

        let max_uniform = u64::from(ctx.device().limits().max_uniform_buffer_binding_size);
        let mut scratch = Scratch::new(ctx);
        // (buffer, bytes to bind) per argument.
        let mut bound: Vec<(wgpu::Buffer, u64)> = Vec::with_capacity(args.len());
        for (index, (arg, info)) in args.iter().zip(&core.bindings).enumerate() {
            match &arg.repr {
                ArgRepr::Storage { ctx: arg_ctx, buffer, bytes, element_size, writable } => {
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
                    if let Some(stride) = info.element_stride {
                        if stride != *element_size {
                            return Err(LumaError::KernelArgElementSize {
                                index,
                                expected: stride,
                                actual: *element_size,
                            });
                        }
                    }
                    bound.push(((*buffer).clone(), *bytes));
                }
                ArgRepr::Uniform(encoded) => {
                    if info.kind != BindingKind::Uniform {
                        return Err(LumaError::KernelArgKind {
                            index,
                            expected: info.kind,
                            actual: BindingKind::Uniform,
                        });
                    }
                    let data = encoded.as_ref().map_err(|message| LumaError::KernelArgUniform {
                        index,
                        message: message.clone(),
                    })?;
                    let bytes = data.len() as u64;
                    if let Some(expected) = info.min_size {
                        if bytes < u64::from(expected) {
                            return Err(LumaError::KernelArgUniformSize {
                                index,
                                expected: u64::from(expected),
                                actual: bytes,
                            });
                        }
                    }
                    if bytes > max_uniform {
                        return Err(LumaError::BufferTooLarge { requested: bytes, max: max_uniform });
                    }
                    let buffer = scratch.take(bytes, UNIFORM_USAGE)?;
                    ctx.queue().write_buffer(&buffer, 0, data);
                    bound.push((buffer, bytes));
                }
            }
        }

        let grid = scratch.take(GRID_BYTES, UNIFORM_USAGE)?;
        ctx.queue().write_buffer(
            &grid,
            0,
            bytemuck::cast_slice(&[size[0], size[1], size[2], 0u32]),
        );

        let device = ctx.device();
        let grid_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &core.grid_layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: BindingResource::Buffer(bind(&grid, GRID_BYTES)),
            }],
        });
        let entries: Vec<BindGroupEntry<'_>> = bound
            .iter()
            .enumerate()
            .map(|(i, (buffer, bytes))| BindGroupEntry {
                binding: i as u32,
                resource: BindingResource::Buffer(bind(buffer, *bytes)),
            })
            .collect();
        let args_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &core.args_layout,
            entries: &entries,
        });

        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(&core.pipeline);
            pass.set_bind_group(0, &grid_group, &[]);
            pass.set_bind_group(1, &args_group, &[]);
            pass.dispatch_workgroups(workgroups[0], workgroups[1], workgroups[2]);
        }
        ctx.queue().submit(Some(encoder.finish()));
        Ok(())
    }
}
