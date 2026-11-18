//! WGSL front end: parses and validates user (or luma) WGSL with naga, reflects its bindings the
//! way a kernel launch needs them, and lowers the entry point and everything it calls to a
//! [`KernelModule`].

use std::collections::{HashMap, HashSet};

use naga::{Arena, Expression, Function, Handle, Statement};

use super::ir::*;
use crate::hal::{BackendError, BindingInfo, BindingKind, Features};

/// WGSL appended to every kernel. Declarations are order independent in WGSL, so appending (not
/// prepending) keeps the line numbers of compile errors identical to the user's source and
/// leaves `enable` directives first.
pub const PRELUDE: &str = "
struct LumaGrid { size: vec3<u32>, _pad: u32 }
@group(0) @binding(0) var<uniform> luma: LumaGrid;
fn luma_idx2(i: u32, j: u32) -> u32 { return i + j * luma.size.x; }
fn luma_idx3(i: u32, j: u32, k: u32) -> u32 { return i + luma.size.x * (j + luma.size.y * k); }
";

/// Appended after [`PRELUDE`] to untrusted sources that mention `LumaView`.
pub const VIEW_PRELUDE: &str = "
struct LumaView { offset: u32, len: u32, _pad: vec2<u32>, stride: vec4<i32>, dims: vec4<u32> }
fn luma_view_idx2(v: LumaView, i: u32, j: u32) -> u32 {
    return u32(i32(v.offset) + v.stride.x * i32(i) + v.stride.y * i32(j));
}
fn luma_view_idx3(v: LumaView, i: u32, j: u32, k: u32) -> u32 {
    return u32(i32(v.offset) + v.stride.x * i32(i) + v.stride.y * i32(j) + v.stride.z * i32(k));
}
fn luma_view_idx4(v: LumaView, c: vec4<u32>) -> u32 {
    let d = vec4<i32>(c) * v.stride;
    return u32(i32(v.offset) + d.x + d.y + d.z + d.w);
}
fn luma_view_idx(v: LumaView, i: u32) -> u32 {
    let c0 = i % v.dims.x;
    let r0 = i / v.dims.x;
    let c1 = r0 % v.dims.y;
    let r1 = r0 / v.dims.y;
    let c2 = r1 % v.dims.z;
    let c3 = r1 / v.dims.z;
    return luma_view_idx4(v, vec4<u32>(c0, c1, c2, c3));
}
";

type R<T> = Result<T, BackendError>;

fn unsupported<T>(what: impl Into<String>) -> R<T> {
    Err(BackendError::Unsupported(what.into()))
}

/// The bindings `module`'s entry point reaches (directly or through the functions it calls):
/// `(declared, None)` when it reaches all of them, `(declared, Some(mask))` otherwise.
pub fn used_bindings(module: &KernelModule) -> (usize, Option<Box<[bool]>>) {
    let declared = module.bindings.len();
    let reachable = module.reachable_functions();
    let mut mask = vec![false; declared];
    let mut mark = |b: u32| {
        if let Some(slot) = mask.get_mut(b as usize) {
            *slot = true;
        }
    };
    let mut scan = |block: &[Stmt]| {
        walk_stmts(block, &mut |s| {
            match s {
                Stmt::Store { binding, .. } => mark(*binding),
                Stmt::Assign { dst, .. } => {
                    if let Some(b) = dst.storage_binding() {
                        mark(b);
                    }
                }
                Stmt::AtomicStore { target, .. } | Stmt::AtomicRmw { target, .. } | Stmt::AtomicCompareExchange { target, .. } => {
                    if let AtomicTarget::Storage { binding, .. } = target {
                        mark(*binding);
                    }
                }
                _ => {}
            }
            s.for_each_expr(&mut |e| {
                e.walk(&mut |x| match x {
                    Expr::Load { binding, .. }
                    | Expr::StorageRead { binding, .. }
                    | Expr::UniformRead { binding, .. }
                    | Expr::ArrayLength { binding } => mark(*binding),
                    Expr::AtomicLoad { target: AtomicTarget::Storage { binding, .. }, .. } => mark(*binding),
                    _ => {}
                })
            });
        });
    };
    scan(&module.entry_body);
    for (f, live) in module.functions.iter().zip(&reachable) {
        if *live {
            scan(&f.body);
        }
    }
    let all = mask.iter().all(|&b| b);
    (declared, (!all).then(|| mask.into_boxed_slice()))
}

/// What naga may accept in a kernel: the shader features the device was opened with.
fn shader_capabilities(features: Features) -> naga::valid::Capabilities {
    use naga::valid::Capabilities as Caps;
    let mut caps = Caps::empty();
    if features.contains(Features::SUBGROUP) {
        caps |= Caps::SUBGROUP | Caps::SUBGROUP_BARRIER;
    }
    if features.contains(Features::SHADER_F16) {
        caps |= Caps::SHADER_FLOAT16;
    }
    if features.contains(Features::SHADER_F64) {
        caps |= Caps::FLOAT64;
    }
    if features.contains(Features::EXPERIMENTAL_COOPERATIVE_MATRIX) {
        caps |= Caps::COOPERATIVE_MATRIX;
    }
    caps
}

fn reflect_kind(space: naga::AddressSpace) -> Option<BindingKind> {
    match space {
        naga::AddressSpace::Uniform => Some(BindingKind::Uniform),
        naga::AddressSpace::Storage { access } if access.contains(naga::StorageAccess::STORE) => Some(BindingKind::StorageReadWrite),
        naga::AddressSpace::Storage { .. } => Some(BindingKind::StorageRead),
        _ => None,
    }
}

/// Parses, validates, reflects and lowers `source`'s compute entry point `entry`.
///
/// Returns the validated naga module (for a backend that consumes naga directly), the lowered
/// [`KernelModule`], the reflected bindings the entry point uses (one per launch argument it
/// binds), the number of arguments a launch passes and, for trusted kernels that leave some
/// declared argument unread, the mask of those it reads.
///
/// A module that uses cooperative matrices (only possible when `features` has
/// [`Features::EXPERIMENTAL_COOPERATIVE_MATRIX`]) is not lowered: its [`KernelModule`] holds the
/// bindings and an empty body, for the backend that compiles the naga module itself.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn parse_wgsl(
    source: &str,
    entry: &str,
    requested_workgroup: Option<[u32; 3]>,
    features: Features,
    max_workgroup_size: [u32; 3],
    max_workgroup_invocations: u32,
    max_dynamic_uniforms: u32,
    trusted: bool,
) -> Result<(naga::Module, KernelModule, Vec<BindingInfo>, usize, Option<Box<[bool]>>), BackendError> {
    if !features.contains(Features::EXPERIMENTAL_COOPERATIVE_MATRIX) && source.contains("wgpu_cooperative_matrix") {
        return Err(BackendError::Unsupported("cooperative matrix requires the wgpu backend".into()));
    }
    let combined = if !trusted && source.contains("LumaView") {
        format!("{source}\n{PRELUDE}{VIEW_PRELUDE}")
    } else {
        format!("{source}\n{PRELUDE}")
    };
    let module = naga::front::wgsl::parse_str(&combined).map_err(|e| BackendError::ShaderCompile(e.emit_to_string(&combined)))?;

    let mut validator = naga::valid::Validator::new(naga::valid::ValidationFlags::all(), shader_capabilities(features));
    let info = validator.validate(&module).map_err(|e| BackendError::ShaderCompile(e.emit_to_string(&combined)))?;

    let (entry_index, entry_point) = module
        .entry_points
        .iter()
        .enumerate()
        .find(|(_, ep)| ep.name == entry && ep.stage == naga::ShaderStage::Compute)
        .ok_or_else(|| {
            let available: Vec<&str> =
                module.entry_points.iter().filter(|ep| ep.stage == naga::ShaderStage::Compute).map(|ep| ep.name.as_str()).collect();
            BackendError::KernelEntryPoint(format!("No @compute entry point `{entry}`; compute entry points: {available:?}"))
        })?;

    // Bindings: group 0 belongs to luma, user resources must be in group 1 with binding
    // numbers 0..n. Only the globals the entry point uses count (directly or through the
    // functions it calls), so one module can hold several entry points with different bindings.
    let used = info.get_entry_point(entry_index);
    let mut user: Vec<(u32, BindingInfo, Handle<naga::GlobalVariable>)> = Vec::new();
    // The @group(1) variables the module declares, used or not.
    let mut declared: Vec<(u32, Handle<naga::GlobalVariable>)> = Vec::new();
    for (handle, var) in module.global_variables.iter() {
        let Some(rb) = var.binding else { continue };
        if trusted && rb.group == 1 {
            declared.push((rb.binding, handle));
        }
        if used[handle].is_empty() {
            continue;
        }
        let name = var.name.clone().unwrap_or_default();
        if rb.group == 0 {
            if name == "luma" && rb.binding == 0 {
                continue;
            }
            return Err(BackendError::KernelBinding(format!(
                "`{name}` is in @group(0), which luma reserves for the grid uniform; \
                 declare kernel arguments in @group(1)"
            )));
        }
        if rb.group != 1 {
            return Err(BackendError::KernelBinding(format!("`{name}` is in @group({}); kernel arguments must be in @group(1)", rb.group)));
        }
        let kind = reflect_kind(var.space)
            .ok_or_else(|| BackendError::KernelBinding(format!("`{name}` is neither a storage buffer nor a uniform")))?;
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
        user.push((rb.binding, BindingInfo { name, kind, element_stride, min_size }, handle));
    }
    user.sort_by_key(|(binding, ..)| *binding);
    let (declared_count, used_mask, decl_order) = if trusted {
        // Luma's launches pass one argument per variable the source declares, numbered 0..n; the
        // kernel binds those the entry point reads.
        declared.sort_unstable_by_key(|(binding, _)| *binding);
        for (expected, (binding, _)) in declared.iter().enumerate() {
            if *binding != expected as u32 {
                return Err(BackendError::KernelBinding(format!(
                    "the @group(1) variables of `{entry}` must be numbered 0..n, but @binding({binding}) \
                     is where @binding({expected}) was expected"
                )));
            }
        }
        let mask: Box<[bool]> =
            (0..declared.len() as u32).map(|binding| user.iter().any(|(used, ..)| *used == binding)).collect();
        let all = mask.iter().all(|&bound| bound);
        let order: Vec<_> = declared.iter().map(|(_, h)| *h).collect();
        (declared.len(), (!all).then_some(mask), order)
    } else {
        for (expected, (binding, info, _)) in user.iter().enumerate() {
            if *binding != expected as u32 {
                return Err(BackendError::KernelBinding(format!(
                    "the @group(1) bindings used by entry point `{entry}` must be numbered 0..n in \
                     argument order, but `{}` is @binding({binding}) where @binding({expected}) \
                     was expected",
                    info.name
                )));
            }
        }
        (user.len(), None, user.iter().map(|(_, _, h)| *h).collect())
    };
    let infos: Vec<BindingInfo> = user.into_iter().map(|(_, info, _)| info).collect();

    if entry_point.workgroup_size_overrides.is_some() {
        return Err(BackendError::KernelWorkgroup("workgroup sizes that depend on `override` constants are not supported".into()));
    }
    let workgroup = entry_point.workgroup_size;
    if workgroup.contains(&0) {
        return Err(BackendError::KernelWorkgroup(format!("@workgroup_size{workgroup:?} has a zero dimension")));
    }
    if let Some(requested) = requested_workgroup
        && requested != workgroup
    {
        return Err(BackendError::KernelWorkgroup(format!(
            "requested {requested:?} but the shader declares @workgroup_size{workgroup:?}"
        )));
    }
    let invocations = workgroup.iter().fold(1u64, |total, &d| total.saturating_mul(u64::from(d)));
    if workgroup.iter().zip(max_workgroup_size).any(|(&d, max)| d > max) || invocations > u64::from(max_workgroup_invocations) {
        return Err(BackendError::KernelWorkgroup(format!(
            "{workgroup:?} exceeds the device limits (per dimension {max_workgroup_size:?}, \
             {max_workgroup_invocations} invocations in total)"
        )));
    }

    // The grid and every user uniform are bound at dynamic offsets into the uniform arena.
    let uniforms = infos.iter().filter(|b| b.kind == BindingKind::Uniform).count() as u32;
    if uniforms + 1 > max_dynamic_uniforms {
        return Err(BackendError::KernelBinding(format!(
            "{uniforms} uniform arguments exceed the device limit of {max_dynamic_uniforms_minus_one} dynamic uniform buffers \
             per kernel (one is used by luma's grid)",
            max_dynamic_uniforms_minus_one = max_dynamic_uniforms.saturating_sub(1)
        )));
    }

    let uses_coop = module.types.iter().any(|(_, t)| matches!(t.inner, naga::TypeInner::CooperativeMatrix { .. }));
    let mut kernel = if uses_coop {
        skeleton(entry, workgroup, &infos)
    } else {
        Lowerer::new(&module)?.lower(entry_index, used, &decl_order)?
    };
    kernel.workgroup_size = workgroup;
    kernel.uses_exact_f32 = module.functions.iter().any(|(_, f)| matches!(f.name.as_deref(), Some("luma_o" | "luma_bits")));
    for body in std::iter::once(&kernel.entry_body).chain(kernel.functions.iter().map(|f| &f.body)) {
        walk_stmts(body, &mut |stmt| {
            stmt.for_each_expr(&mut |expr| {
                expr.walk(&mut |node| {
                    if matches!(node, Expr::Cast { kind: CastKind::FloatToIntSat, .. }) {
                        kernel.uses_exact_f32 = true;
                    }
                });
            });
        });
    }
    Ok((module, kernel, infos, declared_count, used_mask))
}

/// A module of just bindings, for sources a backend compiles from their naga module.
fn skeleton(entry: &str, workgroup: [u32; 3], infos: &[BindingInfo]) -> KernelModule {
    KernelModule {
        label: entry.to_owned(),
        entry: entry.to_owned(),
        workgroup_size: workgroup,
        uses_exact_f32: false,
        structs: Vec::new(),
        bindings: infos
            .iter()
            .map(|i| BindingDecl {
                name: i.name.clone(),
                kind: i.kind,
                elem_ty: KirType::U32,
                is_unsized_array: i.element_stride.is_some(),
                element_stride: i.element_stride,
                min_size: i.min_size,
            })
            .collect(),
        shared: Vec::new(),
        privates: Vec::new(),
        functions: Vec::new(),
        entry_builtins: Vec::new(),
        entry_locals: Vec::new(),
        entry_body: Vec::new(),
    }
}

// ---------------------------------------------------------------------------------------------
// Lowering.

#[derive(Clone, Copy)]
enum Slot {
    None,
    Grid,
    Binding(u32),
    Shared(u32),
    Private(u32),
}

#[derive(Clone)]
enum Root {
    Local(LocalId),
    Private(u32),
    Storage(u32),
    Shared(u32),
    Uniform(u32),
    Grid,
}

/// A pointer: where in memory, and the type it points to.
#[derive(Clone)]
struct Place {
    root: Root,
    root_ty: KirType,
    path: Vec<AccessStep>,
    ty: KirType,
}

enum Low {
    Val(Expr, KirType),
    Place(Place),
    /// Result of an atomic compare-exchange: `(old_value, exchanged)`.
    Pair(LocalId, LocalId),
}

enum ArgInfo {
    Value(KirType),
    Builtin(LocalId, KirType),
    /// A struct argument whose members are builtins.
    Members(Vec<LocalId>),
}

/// Marker type index of `LumaGrid`, which has no declaration in the IR.
const GRID_STRUCT: u32 = u32::MAX;

struct FnCtx<'m> {
    arena: &'m Arena<Expression>,
    func: Option<&'m Function>,
    locals: Vec<(String, KirType, bool)>,
    uses: Vec<u32>,
    baked: Vec<Option<(LocalId, KirType)>>,
    var_locals: Vec<LocalId>,
    args: Vec<ArgInfo>,
    pairs: HashMap<usize, (LocalId, LocalId)>,
    out: Vec<Vec<Stmt>>,
}

impl<'m> FnCtx<'m> {
    fn new(arena: &'m Arena<Expression>, func: Option<&'m Function>) -> Self {
        FnCtx {
            arena,
            func,
            locals: Vec::new(),
            uses: vec![0; arena.len()],
            baked: vec![None; arena.len()],
            var_locals: Vec::new(),
            args: Vec::new(),
            pairs: HashMap::new(),
            out: vec![Vec::new()],
        }
    }

    fn local(&mut self, name: &str, ty: KirType, mutable: bool) -> LocalId {
        self.locals.push((name.to_owned(), ty, mutable));
        LocalId(self.locals.len() as u32 - 1)
    }

    fn emit(&mut self, s: Stmt) {
        self.out.last_mut().expect("block").push(s);
    }
}

fn expr_operands(e: &Expression, f: &mut impl FnMut(Handle<Expression>)) {
    match e {
        Expression::Compose { components, .. } => components.iter().for_each(|c| f(*c)),
        Expression::Access { base, index } => {
            f(*base);
            f(*index);
        }
        Expression::AccessIndex { base, .. } => f(*base),
        Expression::Splat { value, .. } => f(*value),
        Expression::Swizzle { vector, .. } => f(*vector),
        Expression::Load { pointer } => f(*pointer),
        Expression::Unary { expr, .. } | Expression::As { expr, .. } | Expression::Derivative { expr, .. } => f(*expr),
        Expression::Binary { left, right, .. } => {
            f(*left);
            f(*right);
        }
        Expression::Select { condition, accept, reject } => {
            f(*condition);
            f(*accept);
            f(*reject);
        }
        Expression::Relational { argument, .. } => f(*argument),
        Expression::Math { arg, arg1, arg2, arg3, .. } => {
            f(*arg);
            [arg1, arg2, arg3].into_iter().flatten().for_each(|a| f(*a));
        }
        Expression::ArrayLength(p) => f(*p),
        _ => {}
    }
}

fn block_operands(block: &naga::Block, f: &mut impl FnMut(Handle<Expression>)) {
    for s in block.iter() {
        match s {
            Statement::Block(b) => block_operands(b, f),
            Statement::If { condition, accept, reject } => {
                f(*condition);
                block_operands(accept, f);
                block_operands(reject, f);
            }
            Statement::Switch { selector, cases } => {
                f(*selector);
                cases.iter().for_each(|c| block_operands(&c.body, f));
            }
            Statement::Loop { body, continuing, break_if } => {
                block_operands(body, f);
                block_operands(continuing, f);
                if let Some(b) = break_if {
                    f(*b);
                }
            }
            Statement::Return { value: Some(v) } => f(*v),
            Statement::Store { pointer, value } => {
                f(*pointer);
                f(*value);
            }
            Statement::Atomic { pointer, fun, value, .. } => {
                f(*pointer);
                f(*value);
                if let naga::AtomicFunction::Exchange { compare: Some(c) } = fun {
                    f(*c);
                }
            }
            Statement::WorkGroupUniformLoad { pointer, .. } => f(*pointer),
            Statement::Call { arguments, .. } => arguments.iter().for_each(|a| f(*a)),
            Statement::SubgroupBallot { predicate: Some(p), .. } => f(*p),
            Statement::SubgroupGather { mode, argument, .. } => {
                f(*argument);
                use naga::GatherMode as G;
                match mode {
                    G::Broadcast(l) | G::Shuffle(l) | G::ShuffleDown(l) | G::ShuffleUp(l) | G::ShuffleXor(l) | G::QuadBroadcast(l) => f(*l),
                    G::BroadcastFirst | G::QuadSwap(_) => {}
                }
            }
            Statement::SubgroupCollectiveOperation { argument, .. } => f(*argument),
            _ => {}
        }
    }
}

fn collect_calls(block: &naga::Block, out: &mut Vec<Handle<Function>>) {
    for s in block.iter() {
        match s {
            Statement::Call { function, .. } => out.push(*function),
            Statement::Block(b) => collect_calls(b, out),
            Statement::If { accept, reject, .. } => {
                collect_calls(accept, out);
                collect_calls(reject, out);
            }
            Statement::Switch { cases, .. } => cases.iter().for_each(|c| collect_calls(&c.body, out)),
            Statement::Loop { body, continuing, .. } => {
                collect_calls(body, out);
                collect_calls(continuing, out);
            }
            _ => {}
        }
    }
}

struct Lowerer<'m> {
    m: &'m naga::Module,
    layouter: naga::proc::Layouter,
    types: HashMap<Handle<naga::Type>, KirType>,
    structs: Vec<StructDecl>,
    slots: Vec<Slot>,
    bindings: Vec<BindingDecl>,
    shared: Vec<SharedDecl>,
    privates: Vec<PrivateDecl>,
    fn_ids: HashMap<Handle<Function>, FnId>,
}

fn is_trivial(e: &Expr) -> bool {
    matches!(
        e,
        Expr::LitBool(_)
            | Expr::LitU32(_)
            | Expr::LitI32(_)
            | Expr::LitF16(_)
            | Expr::LitF32(_)
            | Expr::LitF64(_)
            | Expr::ZeroVal(_)
            | Expr::Local(_)
            | Expr::FnParam(_)
            | Expr::GridSize(_)
    )
}

fn bool_like(t: &KirType) -> KirType {
    match t {
        KirType::Vec { lanes, .. } => KirType::vec(ScalarTy::Bool, *lanes),
        _ => KirType::BOOL,
    }
}

impl<'m> Lowerer<'m> {
    fn new(m: &'m naga::Module) -> R<Self> {
        let mut layouter = naga::proc::Layouter::default();
        layouter.update(m.to_ctx()).map_err(|e| BackendError::ShaderCompile(format!("{e:?}")))?;
        Ok(Lowerer {
            m,
            layouter,
            types: HashMap::new(),
            structs: Vec::new(),
            slots: vec![Slot::None; m.global_variables.len()],
            bindings: Vec::new(),
            shared: Vec::new(),
            privates: Vec::new(),
            fn_ids: HashMap::new(),
        })
    }

    // Types.

    fn scalar(&self, s: naga::Scalar) -> R<ScalarTy> {
        use naga::ScalarKind as K;
        Ok(match (s.kind, s.width) {
            (K::Bool, _) => ScalarTy::Bool,
            (K::Sint | K::AbstractInt, 4 | 8) => ScalarTy::I32,
            (K::Uint, 4) => ScalarTy::U32,
            (K::Float | K::AbstractFloat, 4) => ScalarTy::F32,
            (K::Float, 2) => ScalarTy::F16,
            (K::Float, 8) => ScalarTy::F64,
            (kind, width) => return unsupported(format!("{kind:?} of {width} bytes")),
        })
    }

    fn ty(&mut self, h: Handle<naga::Type>) -> R<KirType> {
        if let Some(t) = self.types.get(&h) {
            return Ok(t.clone());
        }
        use naga::TypeInner as T;
        let m = self.m;
        let ty = match &m.types[h].inner {
            T::Scalar(s) => KirType::Scalar(self.scalar(*s)?),
            T::Vector { size, scalar } => KirType::Vec { elem: self.scalar(*scalar)?, lanes: u8::from(*size) },
            T::Atomic(s) => KirType::Atomic(self.scalar(*s)?),
            T::Array { base, size: naga::ArraySize::Constant(n), stride } => {
                KirType::Array { elem: Box::new(self.ty(*base)?), len: n.get(), stride: *stride }
            }
            T::Struct { members, span } => {
                let mut fields = Vec::with_capacity(members.len());
                for (i, member) in members.iter().enumerate() {
                    let name = member.name.clone().unwrap_or_else(|| format!("f{i}"));
                    fields.push(StructField { name, ty: self.ty(member.ty)?, offset: member.offset });
                }
                let name = m.types[h].name.clone().unwrap_or_else(|| format!("S{}", self.structs.len()));
                let align = self.layouter[h].alignment.round_up(1);
                self.structs.push(StructDecl { name, fields, size: *span, align });
                KirType::Struct(self.structs.len() as u32 - 1)
            }
            // A matrix is its columns, as in memory: `array<vecR<T>, C>` at the matrix's stride.
            T::Matrix { columns, rows, scalar } => {
                let col = KirType::Vec { elem: self.scalar(*scalar)?, lanes: u8::from(*rows) };
                let (size, align) = col.layout(&self.structs);
                KirType::Array { elem: Box::new(col), len: u32::from(*columns), stride: size.next_multiple_of(align) }
            }
            other => return unsupported(format!("type {other:?}")),
        };
        self.types.insert(h, ty.clone());
        Ok(ty)
    }

    // Declarations.

    fn lower(mut self, entry_index: usize, used: &naga::valid::FunctionInfo, decl_order: &[Handle<naga::GlobalVariable>]) -> R<KernelModule> {
        let m = self.m;
        // Bindings, by binding number.
        for (number, &h) in decl_order.iter().enumerate() {
            let var = &m.global_variables[h];
            let name = var.name.clone().unwrap_or_default();
            let kind = reflect_kind(var.space)
                .ok_or_else(|| BackendError::KernelBinding(format!("`{name}` is neither a storage buffer nor a uniform")))?;
            let inner = &m.types[var.ty].inner;
            let decl = match (kind, inner) {
                (BindingKind::Uniform, inner) => BindingDecl {
                    name,
                    kind,
                    elem_ty: self.ty(var.ty)?,
                    is_unsized_array: false,
                    element_stride: None,
                    min_size: inner.try_size(m.to_ctx()),
                },
                (_, naga::TypeInner::Array { base, size: naga::ArraySize::Dynamic, stride }) => BindingDecl {
                    name,
                    kind,
                    elem_ty: self.ty(*base)?,
                    is_unsized_array: true,
                    element_stride: Some(*stride),
                    min_size: None,
                },
                (_, naga::TypeInner::Array { stride, .. }) => BindingDecl {
                    name,
                    kind,
                    elem_ty: self.ty(var.ty)?,
                    is_unsized_array: false,
                    element_stride: Some(*stride),
                    min_size: None,
                },
                _ => BindingDecl { name, kind, elem_ty: self.ty(var.ty)?, is_unsized_array: false, element_stride: None, min_size: None },
            };
            self.bindings.push(decl);
            self.slots[h.index()] = Slot::Binding(number as u32);
        }
        // Grid, workgroup and private variables the entry point uses.
        for (h, var) in m.global_variables.iter() {
            if used[h].is_empty() {
                continue;
            }
            match var.space {
                naga::AddressSpace::Uniform | naga::AddressSpace::Storage { .. } => {
                    if var.binding.is_some_and(|rb| rb.group == 0) {
                        self.slots[h.index()] = Slot::Grid;
                    }
                }
                naga::AddressSpace::WorkGroup => {
                    let name = var.name.clone().unwrap_or_else(|| format!("shared{}", self.shared.len()));
                    let (elem_ty, len) = match &m.types[var.ty].inner {
                        naga::TypeInner::Array { base, size: naga::ArraySize::Constant(n), .. } => (self.ty(*base)?, Some(n.get())),
                        _ => (self.ty(var.ty)?, None),
                    };
                    self.shared.push(SharedDecl { name, elem_ty, len });
                    self.slots[h.index()] = Slot::Shared(self.shared.len() as u32 - 1);
                }
                naga::AddressSpace::Private => {
                    let name = var.name.clone().unwrap_or_else(|| format!("private{}", self.privates.len()));
                    let ty = self.ty(var.ty)?;
                    let init = match var.init {
                        Some(init) => Some(self.global_expr(init)?.0),
                        None => None,
                    };
                    self.privates.push(PrivateDecl { name, ty, init });
                    self.slots[h.index()] = Slot::Private(self.privates.len() as u32 - 1);
                }
                other => return unsupported(format!("global variable `{}` in address space {other:?}", var.name.as_deref().unwrap_or(""))),
            }
        }

        // Functions the entry point reaches, callees first (naga keeps its arena in that order).
        let entry = &m.entry_points[entry_index];
        let mut reached: HashSet<Handle<Function>> = HashSet::new();
        let mut work = Vec::new();
        collect_calls(&entry.function.body, &mut work);
        while let Some(f) = work.pop() {
            if reached.insert(f) {
                collect_calls(&m.functions[f].body, &mut work);
            }
        }
        let order: Vec<Handle<Function>> = m.functions.iter().map(|(h, _)| h).filter(|h| reached.contains(h)).collect();
        for (i, h) in order.iter().enumerate() {
            self.fn_ids.insert(*h, FnId(i as u32));
        }
        let mut functions = Vec::with_capacity(order.len());
        for h in &order {
            functions.push(self.lower_function(&m.functions[*h], false)?.into_function());
        }
        let lowered = self.lower_function(&entry.function, true)?;
        Ok(KernelModule {
            label: entry.name.clone(),
            entry: entry.name.clone(),
            workgroup_size: entry.workgroup_size,
            uses_exact_f32: false,
            structs: self.structs,
            bindings: self.bindings,
            shared: self.shared,
            privates: self.privates,
            functions,
            entry_builtins: lowered.builtins,
            entry_locals: lowered.locals,
            entry_body: lowered.body,
        })
    }

    fn global_expr(&mut self, h: Handle<Expression>) -> R<(Expr, KirType)> {
        let m = self.m;
        let mut ctx = FnCtx::new(&m.global_expressions, None);
        self.val(&mut ctx, h)
    }

    // Functions.

    fn lower_function(&mut self, f: &'m Function, is_entry: bool) -> R<Lowered> {
        let mut ctx = FnCtx::new(&f.expressions, Some(f));
        let mut params = Vec::new();
        let mut builtins = Vec::new();
        for (i, arg) in f.arguments.iter().enumerate() {
            let name = arg.name.clone().unwrap_or_else(|| format!("arg{i}"));
            if !is_entry {
                let ty = self.ty(arg.ty)?;
                params.push((name, ty.clone()));
                ctx.args.push(ArgInfo::Value(ty));
                continue;
            }
            if let Some(naga::Binding::BuiltIn(b)) = &arg.binding {
                let var = self.builtin(*b)?;
                let local = ctx.local(&name, var.ty(), false);
                builtins.push((var, local));
                ctx.args.push(ArgInfo::Builtin(local, var.ty()));
            } else if let naga::TypeInner::Struct { members, .. } = &self.m.types[arg.ty].inner {
                let mut locals = Vec::new();
                for (k, member) in members.iter().enumerate() {
                    let Some(naga::Binding::BuiltIn(b)) = &member.binding else {
                        return unsupported(format!("entry point argument `{name}` member without a builtin binding"));
                    };
                    let var = self.builtin(*b)?;
                    let local = ctx.local(&format!("{name}_{}", member.name.as_deref().unwrap_or(&k.to_string())), var.ty(), false);
                    builtins.push((var, local));
                    locals.push(local);
                }
                ctx.args.push(ArgInfo::Members(locals));
            } else {
                return unsupported(format!("entry point argument `{name}` without a builtin binding"));
            }
        }
        let ret_ty = match (&f.result, is_entry) {
            (Some(r), false) => Some(self.ty(r.ty)?),
            (Some(_), true) => return unsupported("an entry point that returns a value"),
            (None, _) => None,
        };

        // Uses of each expression, to decide which values get a name.
        let mut uses = vec![0u32; f.expressions.len()];
        for (_, e) in f.expressions.iter() {
            expr_operands(e, &mut |h| uses[h.index()] += 1);
        }
        block_operands(&f.body, &mut |h| uses[h.index()] += 1);
        ctx.uses = uses;

        for (_, var) in f.local_variables.iter() {
            let ty = self.ty(var.ty)?;
            let local = ctx.local(var.name.as_deref().unwrap_or("v"), ty, true);
            ctx.var_locals.push(local);
        }
        for ((_, var), local) in f.local_variables.iter().zip(ctx.var_locals.clone()) {
            let init = match var.init {
                Some(h) => Some(self.val(&mut ctx, h)?.0),
                None => None,
            };
            ctx.emit(Stmt::Var { local, init });
        }
        self.stmts(&mut ctx, &f.body)?;
        let body = ctx.out.pop().expect("body");
        Ok(Lowered {
            name: f.name.clone().unwrap_or_default(),
            params,
            ret_ty,
            locals: ctx.locals,
            body,
            builtins,
        })
    }

    fn builtin(&self, b: naga::BuiltIn) -> R<BuiltinVar> {
        use naga::BuiltIn as B;
        Ok(match b {
            B::GlobalInvocationId => BuiltinVar::GlobalInvocationId,
            B::LocalInvocationId => BuiltinVar::LocalInvocationId,
            B::LocalInvocationIndex => BuiltinVar::LocalInvocationIndex,
            B::WorkGroupId => BuiltinVar::WorkgroupId,
            B::NumWorkGroups => BuiltinVar::NumWorkgroups,
            B::SubgroupInvocationId => BuiltinVar::SubgroupInvocationId,
            B::SubgroupSize => BuiltinVar::SubgroupSize,
            other => return unsupported(format!("builtin {other:?}")),
        })
    }

    // Statements.

    fn block(&mut self, ctx: &mut FnCtx<'m>, block: &naga::Block) -> R<Vec<Stmt>> {
        ctx.out.push(Vec::new());
        let r = self.stmts(ctx, block);
        let out = ctx.out.pop().expect("block");
        r.map(|()| out)
    }

    fn stmts(&mut self, ctx: &mut FnCtx<'m>, block: &naga::Block) -> R<()> {
        for s in block.iter() {
            self.stmt(ctx, s)?;
        }
        Ok(())
    }

    fn stmt(&mut self, ctx: &mut FnCtx<'m>, s: &Statement) -> R<()> {
        match s {
            Statement::Emit(range) => {
                for h in range.clone() {
                    self.bake(ctx, h)?;
                }
            }
            Statement::Block(b) => self.stmts(ctx, b)?,
            Statement::If { condition, accept, reject } => {
                let (cond, _) = self.val(ctx, *condition)?;
                let then_body = self.block(ctx, accept)?;
                let else_body = self.block(ctx, reject)?;
                ctx.emit(Stmt::If { cond, then_body, else_body });
            }
            Statement::Switch { selector, cases } => {
                let (selector, _) = self.val(ctx, *selector)?;
                let mut out = Vec::new();
                let mut default = Vec::new();
                for case in cases {
                    let body = self.block(ctx, &case.body)?;
                    match case.value {
                        naga::SwitchValue::I32(v) => out.push((SwitchCase::I32(v), body, case.fall_through)),
                        naga::SwitchValue::U32(v) => out.push((SwitchCase::U32(v), body, case.fall_through)),
                        naga::SwitchValue::Default => default = body,
                    }
                }
                ctx.emit(Stmt::Switch { selector, cases: out, default });
            }
            Statement::Loop { body, continuing, break_if } => {
                let body = self.block(ctx, body)?;
                // The break condition is evaluated after the continuing block, in its scope.
                ctx.out.push(Vec::new());
                let r = self.stmts(ctx, continuing).and_then(|()| match break_if {
                    Some(h) => self.val(ctx, *h).map(|(e, _)| Some(e)),
                    None => Ok(None),
                });
                let continuing = ctx.out.pop().expect("block");
                ctx.emit(Stmt::Loop { body, continuing, break_if: r? });
            }
            Statement::Break => ctx.emit(Stmt::Break),
            Statement::Continue => ctx.emit(Stmt::Continue),
            Statement::Return { value } => {
                let value = match value {
                    Some(h) => Some(self.val(ctx, *h)?.0),
                    None => None,
                };
                ctx.emit(Stmt::Return(value));
            }
            Statement::ControlBarrier(b) | Statement::MemoryBarrier(b) => {
                let storage = b.contains(naga::Barrier::STORAGE);
                let group = b.contains(naga::Barrier::WORK_GROUP);
                let scope = match (storage, group) {
                    (true, true) => BarrierScope::StorageAndWorkgroup,
                    (false, true) => BarrierScope::Workgroup,
                    (true, false) => BarrierScope::Storage,
                    (false, false) => return unsupported(format!("barrier {b:?}")),
                };
                ctx.emit(Stmt::Barrier(scope));
            }
            Statement::Store { pointer, value } => {
                let place = self.place(ctx, *pointer)?;
                let (val, _) = self.val(ctx, *value)?;
                let stmt = self.write_place(place, val)?;
                ctx.emit(stmt);
            }
            Statement::Atomic { pointer, fun, value, result } => self.atomic(ctx, *pointer, fun, *value, *result)?,
            Statement::WorkGroupUniformLoad { pointer, result } => {
                let place = self.place(ctx, *pointer)?;
                let (read, ty) = self.read_place(&place)?;
                ctx.emit(Stmt::Barrier(BarrierScope::Workgroup));
                self.bind(ctx, *result, read, ty);
                ctx.emit(Stmt::Barrier(BarrierScope::Workgroup));
            }
            Statement::Call { function, arguments, result } => {
                let func = *self.fn_ids.get(function).ok_or_else(|| BackendError::Unsupported("call of an unreachable function".into()))?;
                let mut args = Vec::with_capacity(arguments.len());
                for a in arguments {
                    args.push(self.val(ctx, *a)?.0);
                }
                let ret = match &self.m.functions[*function].result {
                    Some(r) => Some(self.ty(r.ty)?),
                    None => None,
                };
                let call = Expr::Call { func, args, ty: ret.clone() };
                match (result, ret) {
                    (Some(r), Some(ty)) => self.bind(ctx, *r, call, ty),
                    _ => ctx.emit(Stmt::Discard(call)),
                }
            }
            Statement::SubgroupBallot { result, predicate } => {
                let arg = match predicate {
                    Some(p) => Some(self.val(ctx, *p)?.0.boxed()),
                    None => None,
                };
                let ty = KirType::vec(ScalarTy::U32, 4);
                let e = Expr::Subgroup { op: IrSubgroupOp::Ballot, arg, lane: None, ty: ty.clone() };
                self.bind(ctx, *result, e, ty);
            }
            Statement::SubgroupGather { mode, argument, result } => {
                let (arg, ty) = self.val(ctx, *argument)?;
                let lane = match mode {
                    naga::GatherMode::BroadcastFirst => None,
                    naga::GatherMode::Broadcast(l) | naga::GatherMode::Shuffle(l) => Some(self.val(ctx, *l)?.0.boxed()),
                    other => return unsupported(format!("subgroup gather {other:?}")),
                };
                let e = Expr::Subgroup { op: IrSubgroupOp::Shuffle, arg: Some(arg.boxed()), lane, ty: ty.clone() };
                self.bind(ctx, *result, e, ty);
            }
            Statement::SubgroupCollectiveOperation { op, collective_op, argument, result } => {
                use naga::{CollectiveOperation as C, SubgroupOperation as O};
                let (arg, ty) = self.val(ctx, *argument)?;
                let sub = |op: IrSubgroupOp, arg: Expr, ty: &KirType| Expr::Subgroup {
                    op,
                    arg: Some(arg.boxed()),
                    lane: None,
                    ty: ty.clone(),
                };
                let e = match (collective_op, op) {
                    (C::Reduce, O::Add) => sub(IrSubgroupOp::Add, arg, &ty),
                    (C::Reduce, O::Min) => sub(IrSubgroupOp::Min, arg, &ty),
                    (C::Reduce, O::Max) => sub(IrSubgroupOp::Max, arg, &ty),
                    (C::ExclusiveScan, O::Add) => sub(IrSubgroupOp::ExclusiveAdd, arg, &ty),
                    (C::InclusiveScan, O::Add) => {
                        let arg = self.stable(ctx, arg, &ty, "scan");
                        let ex = sub(IrSubgroupOp::ExclusiveAdd, arg.clone(), &ty);
                        Expr::Binary { op: IrBinOp::Add, lhs: ex.boxed(), rhs: arg.boxed(), ty: ty.clone() }
                    }
                    (c, o) => return unsupported(format!("subgroup operation {o:?} ({c:?})")),
                };
                self.bind(ctx, *result, e, ty);
            }
            other => return unsupported(format!("statement {other:?}")),
        }
        Ok(())
    }

    /// Names the value of expression `h`: `let t = e;`.
    fn bind(&mut self, ctx: &mut FnCtx<'m>, h: Handle<Expression>, e: Expr, ty: KirType) {
        let name = ctx
            .func
            .and_then(|f| f.named_expressions.get(&h))
            .cloned()
            .unwrap_or_else(|| format!("t{}", h.index()));
        let local = ctx.local(&name, ty.clone(), false);
        ctx.emit(Stmt::Let { local, init: e });
        ctx.baked[h.index()] = Some((local, ty));
    }

    /// At an `Emit`: names the loads (their value depends on when they run) and the values used
    /// more than once.
    fn bake(&mut self, ctx: &mut FnCtx<'m>, h: Handle<Expression>) -> R<()> {
        let arena = ctx.arena;
        let is_load = matches!(arena[h], Expression::Load { .. });
        if !is_load && ctx.uses[h.index()] < 2 {
            return Ok(());
        }
        if let Low::Val(e, ty) = self.expr(ctx, h)? {
            let cheap = matches!(e, Expr::GridSize(_) | Expr::UniformRead { .. } | Expr::LitU32(_) | Expr::LitI32(_) | Expr::LitF32(_) | Expr::LitBool(_));
            if !(is_trivial(&e) && !is_load) && !cheap {
                self.bind(ctx, h, e, ty);
            }
        }
        Ok(())
    }

    /// Emits `let name = e;` for a value that is used more than once inside a lowered expansion;
    /// constant expressions have nowhere to put it and repeat `e`.
    fn stable(&mut self, ctx: &mut FnCtx<'m>, e: Expr, ty: &KirType, name: &str) -> Expr {
        if is_trivial(&e) || ctx.func.is_none() {
            return e;
        }
        let local = ctx.local(name, ty.clone(), false);
        ctx.emit(Stmt::Let { local, init: e });
        Expr::Local(local)
    }

    fn atomic(&mut self, ctx: &mut FnCtx<'m>, pointer: Handle<Expression>, fun: &naga::AtomicFunction, value: Handle<Expression>, result: Option<Handle<Expression>>) -> R<()> {
        use naga::AtomicFunction as A;
        let place = self.place(ctx, pointer)?;
        let KirType::Atomic(elem_ty) = place.ty else {
            return unsupported("atomic operation on a non-atomic pointer");
        };
        let target = self.atomic_target(&place)?;
        let (val, _) = self.val(ctx, value)?;
        let op = match fun {
            A::Add => IrAtomicOp::Add,
            A::Subtract => IrAtomicOp::Sub,
            A::And => IrAtomicOp::And,
            A::InclusiveOr => IrAtomicOp::Or,
            A::ExclusiveOr => IrAtomicOp::Xor,
            A::Min => IrAtomicOp::Min,
            A::Max => IrAtomicOp::Max,
            A::Exchange { compare: None } => IrAtomicOp::Exchange,
            A::Exchange { compare: Some(cmp) } => {
                let (cmp, _) = self.val(ctx, *cmp)?;
                let old = ctx.local("old_value", KirType::Scalar(elem_ty), false);
                let ex = ctx.local("exchanged", KirType::BOOL, false);
                if let Some(r) = result {
                    ctx.pairs.insert(r.index(), (old, ex));
                }
                ctx.emit(Stmt::AtomicCompareExchange { dst_old: old, dst_exchanged: ex, elem_ty, target, cmp, val });
                return Ok(());
            }
        };
        let dst = result.map(|r| {
            let local = ctx.local(&format!("t{}", r.index()), KirType::Scalar(elem_ty), false);
            ctx.baked[r.index()] = Some((local, KirType::Scalar(elem_ty)));
            local
        });
        ctx.emit(Stmt::AtomicRmw { dst, op, elem_ty, target, val });
        Ok(())
    }

    // Places.

    fn step_ty(&self, base: &KirType, step: &AccessStep) -> R<KirType> {
        match (base, step) {
            (KirType::Struct(GRID_STRUCT), AccessStep::Field(0)) => Ok(KirType::vec(ScalarTy::U32, 3)),
            (KirType::Struct(GRID_STRUCT), AccessStep::Field(1)) => Ok(KirType::U32),
            (KirType::Struct(s), AccessStep::Field(i)) => self
                .structs
                .get(*s as usize)
                .and_then(|d| d.fields.get(*i as usize))
                .map(|f| f.ty.clone())
                .ok_or_else(|| BackendError::Unsupported("struct field out of range".into())),
            (KirType::Vec { elem, .. }, _) => Ok(KirType::Scalar(*elem)),
            (KirType::Array { elem, .. }, AccessStep::Index(_)) => Ok((**elem).clone()),
            (base, step) => unsupported(format!("access {step:?} into {base:?}")),
        }
    }

    /// The step of `AccessIndex(base, i)`.
    fn index_step(base: &KirType, i: u32) -> AccessStep {
        match base {
            KirType::Array { .. } => AccessStep::Index(Expr::LitU32(i).boxed()),
            _ => AccessStep::Field(i),
        }
    }

    fn extend_place(&self, mut place: Place, step: AccessStep) -> R<Place> {
        place.ty = self.step_ty(&place.ty, &step)?;
        place.path.push(step);
        Ok(place)
    }

    fn place(&mut self, ctx: &mut FnCtx<'m>, h: Handle<Expression>) -> R<Place> {
        match self.expr(ctx, h)? {
            Low::Place(p) => Ok(p),
            _ => unsupported("a pointer that is not a variable access"),
        }
    }

    fn atomic_target(&self, place: &Place) -> R<AtomicTarget> {
        let index = match place.path.as_slice() {
            [] => None,
            [AccessStep::Index(i)] => Some(i.clone()),
            _ => return unsupported("atomic in a struct field"),
        };
        match place.root {
            Root::Storage(binding) => Ok(AtomicTarget::Storage { binding, index }),
            Root::Shared(shared) => Ok(AtomicTarget::Shared { shared, index }),
            _ => unsupported("atomic outside storage or workgroup memory"),
        }
    }

    fn read_place(&self, place: &Place) -> R<(Expr, KirType)> {
        if let KirType::Atomic(s) = place.ty {
            let target = self.atomic_target(place)?;
            return Ok((Expr::AtomicLoad { target, ty: s }, KirType::Scalar(s)));
        }
        let ty = place.ty.clone();
        let fold = |mut e: Expr| -> R<(Expr, KirType)> {
            let mut cur = place.root_ty.clone();
            for step in &place.path {
                cur = self.step_ty(&cur, step)?;
                e = self.access_val(e, step.clone(), cur.clone());
            }
            Ok((e, cur))
        };
        match place.root {
            Root::Grid => self.read_grid(&place.path),
            Root::Local(l) => fold(Expr::Local(l)),
            Root::Private(p) => fold(Expr::Private(p)),
            Root::Uniform(binding) => Ok((Expr::UniformRead { binding, path: place.path.clone(), ty: ty.clone() }, ty)),
            Root::Storage(binding) => {
                if self.bindings[binding as usize].is_unsized_array
                    && let [AccessStep::Index(i)] = place.path.as_slice()
                {
                    return Ok((Expr::Load { binding, index: i.clone(), ty: ty.clone() }, ty));
                }
                Ok((Expr::StorageRead { binding, path: place.path.clone(), ty: ty.clone() }, ty))
            }
            Root::Shared(shared) => {
                if self.shared[shared as usize].len.is_some()
                    && let [AccessStep::Index(i)] = place.path.as_slice()
                {
                    return Ok((Expr::SharedLoad { shared, index: i.clone(), ty: ty.clone() }, ty));
                }
                Ok((Expr::SharedRead { shared, path: place.path.clone(), ty: ty.clone() }, ty))
            }
        }
    }

    fn read_grid(&self, path: &[AccessStep]) -> R<(Expr, KirType)> {
        let vec3 = KirType::vec(ScalarTy::U32, 3);
        let size = || Expr::Compose { ty: vec3.clone(), comps: (0..3).map(Expr::GridSize).collect() };
        match path {
            [AccessStep::Field(0)] => Ok((size(), vec3.clone())),
            [AccessStep::Field(0), AccessStep::Field(k)] if *k < 3 => Ok((Expr::GridSize(*k as u8), KirType::U32)),
            [AccessStep::Field(0), AccessStep::Index(i)] => match **i {
                Expr::LitU32(k) if k < 3 => Ok((Expr::GridSize(k as u8), KirType::U32)),
                _ => Ok((Expr::Access { base: size().boxed(), step: AccessStep::Index(i.clone()), ty: KirType::U32 }, KirType::U32)),
            },
            [AccessStep::Field(1)] => Ok((Expr::LitU32(0), KirType::U32)),
            _ => unsupported("this use of the luma grid uniform"),
        }
    }

    fn write_place(&self, place: Place, val: Expr) -> R<Stmt> {
        if matches!(place.ty, KirType::Atomic(_)) {
            let target = self.atomic_target(&place)?;
            return Ok(Stmt::AtomicStore { target, val });
        }
        let fold = |mut lv: LValue| {
            for step in place.path.iter().cloned() {
                lv = match step {
                    AccessStep::Field(field) => LValue::Field { base: Box::new(lv), field },
                    AccessStep::Index(index) => LValue::Index { base: Box::new(lv), index },
                };
            }
            lv
        };
        Ok(match place.root {
            Root::Local(l) => Stmt::Assign { dst: fold(LValue::Local(l)), val },
            Root::Private(p) => Stmt::Assign { dst: fold(LValue::Private(p)), val },
            Root::Storage(binding) => {
                if self.bindings[binding as usize].is_unsized_array
                    && let [AccessStep::Index(i)] = place.path.as_slice()
                {
                    Stmt::Store { binding, index: (**i).clone(), val }
                } else {
                    Stmt::Assign { dst: LValue::Storage { binding, path: place.path }, val }
                }
            }
            Root::Shared(shared) => {
                if self.shared[shared as usize].len.is_some()
                    && let [AccessStep::Index(i)] = place.path.as_slice()
                {
                    Stmt::SharedStore { shared, index: (**i).clone(), val }
                } else {
                    Stmt::Assign { dst: LValue::Shared { shared, path: place.path }, val }
                }
            }
            Root::Uniform(_) | Root::Grid => return unsupported("a store to a uniform"),
        })
    }

    fn access_val(&self, base: Expr, step: AccessStep, ty: KirType) -> Expr {
        // A lane of a vector built from scalars is that scalar.
        if let (Expr::Compose { comps, ty: KirType::Vec { lanes, .. } }, AccessStep::Field(k)) = (&base, &step)
            && comps.len() == usize::from(*lanes)
            && let Some(c) = comps.get(*k as usize)
        {
            return c.clone();
        }
        Expr::Access { base: base.boxed(), step, ty }
    }

    // Expressions.

    fn val(&mut self, ctx: &mut FnCtx<'m>, h: Handle<Expression>) -> R<(Expr, KirType)> {
        match self.expr(ctx, h)? {
            Low::Val(e, t) => Ok((e, t)),
            Low::Place(_) => unsupported("a pointer used as a value"),
            Low::Pair(..) => unsupported("the whole result of an atomic compare-exchange"),
        }
    }

    fn literal(&self, l: naga::Literal) -> R<(Expr, KirType)> {
        use naga::Literal as L;
        Ok(match l {
            L::F64(v) => (Expr::LitF64(v), KirType::Scalar(ScalarTy::F64)),
            L::F32(v) => (Expr::LitF32(v), KirType::F32),
            L::F16(v) => (Expr::LitF16(f32::from(v)), KirType::Scalar(ScalarTy::F16)),
            L::U32(v) => (Expr::LitU32(v), KirType::U32),
            L::I32(v) => (Expr::LitI32(v), KirType::I32),
            L::Bool(v) => (Expr::LitBool(v), KirType::BOOL),
            L::AbstractInt(v) => (Expr::LitI32(v as i32), KirType::I32),
            L::AbstractFloat(v) => (Expr::LitF32(v as f32), KirType::F32),
            other => return unsupported(format!("literal {other:?}")),
        })
    }

    fn expr(&mut self, ctx: &mut FnCtx<'m>, h: Handle<Expression>) -> R<Low> {
        if let Some((l, ty)) = &ctx.baked[h.index()] {
            return Ok(Low::Val(Expr::Local(*l), ty.clone()));
        }
        if let Some(&(a, b)) = ctx.pairs.get(&h.index()) {
            return Ok(Low::Pair(a, b));
        }
        let arena = ctx.arena;
        let m = self.m;
        match &arena[h] {
            Expression::Literal(l) => {
                let (e, t) = self.literal(*l)?;
                Ok(Low::Val(e, t))
            }
            Expression::Constant(c) => {
                let (e, t) = self.global_expr(m.constants[*c].init)?;
                Ok(Low::Val(e, t))
            }
            Expression::ZeroValue(t) => {
                let ty = self.ty(*t)?;
                Ok(Low::Val(Expr::ZeroVal(ty.clone()), ty))
            }
            Expression::Compose { ty, components } => {
                let ty = self.ty(*ty)?;
                let mut comps = Vec::with_capacity(components.len());
                for c in components {
                    comps.push(self.val(ctx, *c)?.0);
                }
                Ok(Low::Val(Expr::Compose { ty: ty.clone(), comps }, ty))
            }
            Expression::Access { base, index } => {
                let (idx, _) = self.val(ctx, *index)?;
                let step = AccessStep::Index(idx.boxed());
                match self.expr(ctx, *base)? {
                    Low::Place(p) => Ok(Low::Place(self.extend_place(p, step)?)),
                    Low::Val(e, t) => {
                        let ty = self.step_ty(&t, &step)?;
                        Ok(Low::Val(self.access_val(e, step, ty.clone()), ty))
                    }
                    Low::Pair(..) => unsupported("indexing an atomic compare-exchange result"),
                }
            }
            Expression::AccessIndex { base, index } => {
                if let Expression::FunctionArgument(i) = &arena[*base]
                    && let Some(ArgInfo::Members(locals)) = ctx.args.get(*i as usize)
                {
                    let local = *locals.get(*index as usize).ok_or_else(|| BackendError::Unsupported("argument member out of range".into()))?;
                    let ty = ctx.locals[local.0 as usize].1.clone();
                    return Ok(Low::Val(Expr::Local(local), ty));
                }
                match self.expr(ctx, *base)? {
                    Low::Place(p) => {
                        let step = Self::index_step(&p.ty, *index);
                        Ok(Low::Place(self.extend_place(p, step)?))
                    }
                    Low::Val(e, t) => {
                        let step = Self::index_step(&t, *index);
                        let ty = self.step_ty(&t, &step)?;
                        Ok(Low::Val(self.access_val(e, step, ty.clone()), ty))
                    }
                    Low::Pair(old, exchanged) => {
                        let (local, ty) = if *index == 0 {
                            (old, ctx.locals[old.0 as usize].1.clone())
                        } else {
                            (exchanged, KirType::BOOL)
                        };
                        Ok(Low::Val(Expr::Local(local), ty))
                    }
                }
            }
            Expression::Splat { size, value } => {
                let (v, t) = self.val(ctx, *value)?;
                let elem = t.scalar().ok_or_else(|| BackendError::Unsupported("splat of a non-scalar".into()))?;
                let lanes = u8::from(*size);
                Ok(Low::Val(Expr::Splat { elem, lanes, val: v.boxed() }, KirType::vec(elem, lanes)))
            }
            Expression::Swizzle { size, vector, pattern } => {
                let (v, t) = self.val(ctx, *vector)?;
                let elem = t.elem_scalar().ok_or_else(|| BackendError::Unsupported("swizzle of a non-vector".into()))?;
                let len = u8::from(*size);
                let indices = [pattern[0] as u8, pattern[1] as u8, pattern[2] as u8, pattern[3] as u8];
                let ty = KirType::vec(elem, len);
                Ok(Low::Val(Expr::Swizzle { base: v.boxed(), indices, len, ty: ty.clone() }, ty))
            }
            Expression::FunctionArgument(i) => match ctx.args.get(*i as usize) {
                Some(ArgInfo::Value(ty)) => Ok(Low::Val(Expr::FnParam(*i), ty.clone())),
                Some(ArgInfo::Builtin(l, ty)) => Ok(Low::Val(Expr::Local(*l), ty.clone())),
                _ => unsupported("an entry point argument struct used as a whole"),
            },
            Expression::GlobalVariable(g) => self.global_place(*g).map(Low::Place),
            Expression::LocalVariable(l) => {
                let ty = self.ty(ctx.func.expect("function").local_variables[*l].ty)?;
                let local = ctx.var_locals[l.index()];
                Ok(Low::Place(Place { root: Root::Local(local), root_ty: ty.clone(), path: Vec::new(), ty }))
            }
            Expression::Load { pointer } => {
                let place = self.place(ctx, *pointer)?;
                let (e, t) = self.read_place(&place)?;
                Ok(Low::Val(e, t))
            }
            Expression::Unary { op, expr } => {
                let (e, ty) = self.val(ctx, *expr)?;
                let op = match op {
                    naga::UnaryOperator::Negate => IrUnOp::Neg,
                    naga::UnaryOperator::LogicalNot => IrUnOp::Not,
                    naga::UnaryOperator::BitwiseNot => IrUnOp::BitNot,
                };
                Ok(Low::Val(Expr::Unary { op, arg: e.boxed(), ty: ty.clone() }, ty))
            }
            Expression::Binary { op, left, right } => {
                use naga::BinaryOperator as B;
                let (l, lt) = self.val(ctx, *left)?;
                let (r, rt) = self.val(ctx, *right)?;
                if Self::mat_shape(&lt).is_some() || Self::mat_shape(&rt).is_some() {
                    let (e, t) = self.matrix_binary(ctx, *op, l, lt, r, rt)?;
                    return Ok(Low::Val(e, t));
                }
                let op = match op {
                    B::Add => IrBinOp::Add,
                    B::Subtract => IrBinOp::Sub,
                    B::Multiply => IrBinOp::Mul,
                    B::Divide => IrBinOp::Div,
                    B::Modulo => IrBinOp::Rem,
                    B::Equal => IrBinOp::Eq,
                    B::NotEqual => IrBinOp::Ne,
                    B::Less => IrBinOp::Lt,
                    B::LessEqual => IrBinOp::Le,
                    B::Greater => IrBinOp::Gt,
                    B::GreaterEqual => IrBinOp::Ge,
                    B::And => IrBinOp::BitAnd,
                    B::ExclusiveOr => IrBinOp::BitXor,
                    B::InclusiveOr => IrBinOp::BitOr,
                    B::LogicalAnd => IrBinOp::And,
                    B::LogicalOr => IrBinOp::Or,
                    B::ShiftLeft => IrBinOp::Shl,
                    B::ShiftRight => IrBinOp::Shr,
                };
                let ty = if op.is_comparison() {
                    bool_like(if matches!(lt, KirType::Vec { .. }) { &lt } else { &rt })
                } else if matches!(op, IrBinOp::And | IrBinOp::Or) {
                    KirType::BOOL
                } else if matches!(lt, KirType::Vec { .. }) || !matches!(rt, KirType::Vec { .. }) {
                    lt
                } else {
                    rt
                };
                Ok(Low::Val(Expr::Binary { op, lhs: l.boxed(), rhs: r.boxed(), ty: ty.clone() }, ty))
            }
            Expression::Select { condition, accept, reject } => {
                let (c, _) = self.val(ctx, *condition)?;
                let (a, ty) = self.val(ctx, *accept)?;
                let (r, _) = self.val(ctx, *reject)?;
                Ok(Low::Val(Expr::Ternary { op: IrTernaryOp::Select, a: c.boxed(), b: a.boxed(), c: r.boxed(), ty: ty.clone() }, ty))
            }
            Expression::Relational { fun, argument } => {
                let (e, t) = self.val(ctx, *argument)?;
                let (op, ty) = match fun {
                    naga::RelationalFunction::All => (IrUnOp::All, KirType::BOOL),
                    naga::RelationalFunction::Any => (IrUnOp::Any, KirType::BOOL),
                    naga::RelationalFunction::IsNan => (IrUnOp::IsNan, bool_like(&t)),
                    naga::RelationalFunction::IsInf => (IrUnOp::IsInf, bool_like(&t)),
                };
                Ok(Low::Val(Expr::Unary { op, arg: e.boxed(), ty: ty.clone() }, ty))
            }
            Expression::Math { fun, arg, arg1, arg2, .. } => {
                let mut args = vec![self.val(ctx, *arg)?];
                for a in [arg1, arg2].into_iter().flatten() {
                    args.push(self.val(ctx, *a)?);
                }
                let (e, t) = self.math(ctx, *fun, args)?;
                Ok(Low::Val(e, t))
            }
            Expression::As { expr, kind, convert } => {
                let (e, src) = self.val(ctx, *expr)?;
                if let Some(shape) = Self::mat_shape(&src) {
                    let (e, t) = self.matrix_cast(ctx, e, src, shape, *kind, *convert)?;
                    return Ok(Low::Val(e, t));
                }
                let from = src.elem_scalar().ok_or_else(|| BackendError::Unsupported("cast of an aggregate".into()))?;
                let width = convert.unwrap_or(from.size_bytes() as u8);
                let to = self.scalar(naga::Scalar { kind: *kind, width: if *kind == naga::ScalarKind::Bool { 1 } else { width } })?;
                if to == from {
                    return Ok(Low::Val(e, src));
                }
                let dst = match &src {
                    KirType::Vec { lanes, .. } => KirType::vec(to, *lanes),
                    _ => KirType::Scalar(to),
                };
                let cast = match convert {
                    None => CastKind::Bitcast,
                    Some(_) if from.is_float() && to.is_int() => CastKind::FloatToIntSat,
                    Some(_) => CastKind::Convert,
                };
                Ok(Low::Val(Expr::Cast { kind: cast, expr: e.boxed(), src_ty: src, dst_ty: dst.clone() }, dst))
            }
            Expression::ArrayLength(p) => match self.place(ctx, *p)? {
                Place { root: Root::Storage(binding), path, .. } if path.is_empty() => Ok(Low::Val(Expr::ArrayLength { binding }, KirType::U32)),
                _ => unsupported("arrayLength of anything but a storage binding"),
            },
            Expression::CallResult(_)
            | Expression::AtomicResult { .. }
            | Expression::WorkGroupUniformLoadResult { .. }
            | Expression::SubgroupBallotResult
            | Expression::SubgroupOperationResult { .. } => unsupported("a statement result used before its statement"),
            other => unsupported(format!("expression {other:?}")),
        }
    }

    fn global_place(&mut self, g: Handle<naga::GlobalVariable>) -> R<Place> {
        let m = self.m;
        let var = &m.global_variables[g];
        let place = |root, root_ty: KirType| Place { root, ty: root_ty.clone(), root_ty, path: Vec::new() };
        match self.slots[g.index()] {
            Slot::Grid => Ok(place(Root::Grid, KirType::Struct(GRID_STRUCT))),
            Slot::Binding(b) => {
                let decl = &self.bindings[b as usize];
                let root = if decl.kind == BindingKind::Uniform { Root::Uniform(b) } else { Root::Storage(b) };
                let root_ty = if decl.is_unsized_array {
                    KirType::Array { elem: Box::new(decl.elem_ty.clone()), len: 0, stride: decl.element_stride.unwrap_or(0) }
                } else {
                    decl.elem_ty.clone()
                };
                Ok(place(root, root_ty))
            }
            Slot::Shared(s) => {
                let decl = &self.shared[s as usize];
                let root_ty = match decl.len {
                    Some(len) => {
                        let (size, align) = decl.elem_ty.layout(&self.structs);
                        KirType::Array { elem: Box::new(decl.elem_ty.clone()), len, stride: size.next_multiple_of(align) }
                    }
                    None => decl.elem_ty.clone(),
                };
                Ok(place(Root::Shared(s), root_ty))
            }
            Slot::Private(p) => Ok(place(Root::Private(p), self.privates[p as usize].ty.clone())),
            Slot::None => unsupported(format!("global variable `{}`", var.name.as_deref().unwrap_or(""))),
        }
    }

    // Math.

    /// A literal of the scalar type of `ty` (splatted for vectors).
    fn lit_like(ty: &KirType, v: f64) -> Expr {
        let elem = ty.elem_scalar().unwrap_or(ScalarTy::F32);
        let lit = match elem {
            ScalarTy::F64 => Expr::LitF64(v),
            ScalarTy::F16 => Expr::LitF16(v as f32),
            ScalarTy::I32 => Expr::LitI32(v as i32),
            ScalarTy::U32 => Expr::LitU32(v as u32),
            _ => Expr::LitF32(v as f32),
        };
        match ty {
            KirType::Vec { lanes, .. } => Expr::Splat { elem, lanes: *lanes, val: lit.boxed() },
            _ => lit,
        }
    }

    fn math(&mut self, ctx: &mut FnCtx<'m>, fun: naga::MathFunction, args: Vec<(Expr, KirType)>) -> R<(Expr, KirType)> {
        use naga::MathFunction as M;
        let mut it = args.into_iter();
        let (a, ta) = it.next().expect("math argument");
        let b = it.next();
        let c = it.next();
        let un = |op: IrUnOp, a: Expr, ty: &KirType| (Expr::Unary { op, arg: a.boxed(), ty: ty.clone() }, ty.clone());
        let bin = |op: IrBinOp, l: Expr, r: Expr, ty: &KirType| (Expr::Binary { op, lhs: l.boxed(), rhs: r.boxed(), ty: ty.clone() }, ty.clone());
        let sub = |l: Expr, r: Expr, ty: &KirType| Expr::Binary { op: IrBinOp::Sub, lhs: l.boxed(), rhs: r.boxed(), ty: ty.clone() };
        let mulx = |l: Expr, r: Expr, ty: &KirType| Expr::Binary { op: IrBinOp::Mul, lhs: l.boxed(), rhs: r.boxed(), ty: ty.clone() };
        let addx = |l: Expr, r: Expr, ty: &KirType| Expr::Binary { op: IrBinOp::Add, lhs: l.boxed(), rhs: r.boxed(), ty: ty.clone() };
        match fun {
            M::Abs => Ok(un(IrUnOp::Abs, a, &ta)),
            M::Cos => Ok(un(IrUnOp::Cos, a, &ta)),
            M::Sin => Ok(un(IrUnOp::Sin, a, &ta)),
            M::Tan => Ok(un(IrUnOp::Tan, a, &ta)),
            M::Tanh => Ok(un(IrUnOp::Tanh, a, &ta)),
            M::Asin => Ok(un(IrUnOp::Asin, a, &ta)),
            M::Acos => Ok(un(IrUnOp::Acos, a, &ta)),
            M::Atan => Ok(un(IrUnOp::Atan, a, &ta)),
            M::Exp => Ok(un(IrUnOp::Exp, a, &ta)),
            M::Exp2 => Ok(un(IrUnOp::Exp2, a, &ta)),
            M::Log => Ok(un(IrUnOp::Ln, a, &ta)),
            M::Log2 => Ok(un(IrUnOp::Log2, a, &ta)),
            M::Sqrt => Ok(un(IrUnOp::Sqrt, a, &ta)),
            M::InverseSqrt => Ok(un(IrUnOp::InverseSqrt, a, &ta)),
            M::Floor => Ok(un(IrUnOp::Floor, a, &ta)),
            M::Ceil => Ok(un(IrUnOp::Ceil, a, &ta)),
            M::Trunc => Ok(un(IrUnOp::Trunc, a, &ta)),
            M::Round => Ok(un(IrUnOp::Round, a, &ta)),
            M::Sign => Ok(un(IrUnOp::Sign, a, &ta)),
            M::CountOneBits => Ok(un(IrUnOp::CountOneBits, a, &ta)),
            M::FirstTrailingBit => Ok(un(IrUnOp::FirstTrailingBit, a, &ta)),
            M::FirstLeadingBit => Ok(un(IrUnOp::FirstLeadingBit, a, &ta)),
            M::CountLeadingZeros => Ok(un(IrUnOp::CountLeadingZeros, a, &ta)),
            M::CountTrailingZeros => Ok(un(IrUnOp::CountTrailingZeros, a, &ta)),
            M::Min | M::Max | M::Atan2 | M::Pow => {
                let op = match fun {
                    M::Min => IrBinOp::Min,
                    M::Max => IrBinOp::Max,
                    M::Atan2 => IrBinOp::Atan2,
                    _ => IrBinOp::Pow,
                };
                let (b, _) = b.expect("second argument");
                Ok(bin(op, a, b, &ta))
            }
            M::Clamp | M::Fma => {
                let op = if fun == M::Clamp { IrTernaryOp::Clamp } else { IrTernaryOp::Fma };
                let ((b, _), (c, _)) = (b.expect("second argument"), c.expect("third argument"));
                Ok((Expr::Ternary { op, a: a.boxed(), b: b.boxed(), c: c.boxed(), ty: ta.clone() }, ta))
            }
            M::Saturate => {
                let (lo, hi) = (Self::lit_like(&ta, 0.0), Self::lit_like(&ta, 1.0));
                Ok((Expr::Ternary { op: IrTernaryOp::Clamp, a: a.boxed(), b: lo.boxed(), c: hi.boxed(), ty: ta.clone() }, ta))
            }
            M::Fract => {
                let a = self.stable(ctx, a, &ta, "fract");
                let floor = un(IrUnOp::Floor, a.clone(), &ta).0;
                Ok((sub(a, floor, &ta), ta))
            }
            M::Mix => {
                let ((b, _), (t, tt)) = (b.expect("second argument"), c.expect("third argument"));
                let (a, b, t) = (self.stable(ctx, a, &ta, "mix_a"), self.stable(ctx, b, &ta, "mix_b"), self.stable(ctx, t, &tt, "mix_t"));
                let one_minus = sub(Self::lit_like(&tt, 1.0), t.clone(), &tt);
                Ok((addx(mulx(a, one_minus, &ta), mulx(b, t, &ta), &ta), ta))
            }
            M::Step => {
                // step(edge, x): 1 when x >= edge.
                let (x, tx) = b.expect("second argument");
                let cond = Expr::Binary { op: IrBinOp::Ge, lhs: x.boxed(), rhs: a.boxed(), ty: bool_like(&tx) };
                let one = Self::lit_like(&tx, 1.0);
                let zero = Self::lit_like(&tx, 0.0);
                Ok((Expr::Ternary { op: IrTernaryOp::Select, a: cond.boxed(), b: one.boxed(), c: zero.boxed(), ty: tx.clone() }, tx))
            }
            M::SmoothStep => {
                let ((e1, _), (x, _)) = (b.expect("second argument"), c.expect("third argument"));
                // smoothstep(e0, e1, x); `a` is e0.
                let (e0, x, e1) = (self.stable(ctx, a, &ta, "ss_e0"), self.stable(ctx, x, &ta, "ss_x"), self.stable(ctx, e1, &ta, "ss_e1"));
                let ratio = Expr::Binary {
                    op: IrBinOp::Div,
                    lhs: sub(x, e0.clone(), &ta).boxed(),
                    rhs: sub(e1, e0, &ta).boxed(),
                    ty: ta.clone(),
                };
                let t = Expr::Ternary {
                    op: IrTernaryOp::Clamp,
                    a: ratio.boxed(),
                    b: Self::lit_like(&ta, 0.0).boxed(),
                    c: Self::lit_like(&ta, 1.0).boxed(),
                    ty: ta.clone(),
                };
                let t = self.stable(ctx, t, &ta, "ss_t");
                let poly = sub(Self::lit_like(&ta, 3.0), mulx(Self::lit_like(&ta, 2.0), t.clone(), &ta), &ta);
                Ok((mulx(mulx(t.clone(), t, &ta), poly, &ta), ta))
            }
            M::Transpose => self.matrix_transpose(ctx, a, ta),
            M::Determinant => self.matrix_determinant(ctx, a, ta),
            M::Radians => Ok((mulx(a, Self::lit_like(&ta, std::f64::consts::PI / 180.0), &ta), ta)),
            M::Degrees => Ok((mulx(a, Self::lit_like(&ta, 180.0 / std::f64::consts::PI), &ta), ta)),
            M::Dot => {
                let (b, _) = b.expect("second argument");
                let elem = KirType::Scalar(ta.elem_scalar().unwrap_or(ScalarTy::F32));
                Ok((self.dot(ctx, a, b, &ta, &elem), elem))
            }
            M::Length => {
                let elem = KirType::Scalar(ta.elem_scalar().unwrap_or(ScalarTy::F32));
                if matches!(ta, KirType::Vec { .. }) {
                    let a = self.stable(ctx, a, &ta, "len");
                    let d = self.dot(ctx, a.clone(), a, &ta, &elem);
                    Ok(un(IrUnOp::Sqrt, d, &elem))
                } else {
                    Ok(un(IrUnOp::Abs, a, &elem))
                }
            }
            M::Distance => {
                let (b, _) = b.expect("second argument");
                let elem = KirType::Scalar(ta.elem_scalar().unwrap_or(ScalarTy::F32));
                let d = sub(a, b, &ta);
                if matches!(ta, KirType::Vec { .. }) {
                    let d = self.stable(ctx, d, &ta, "dist");
                    let dd = self.dot(ctx, d.clone(), d, &ta, &elem);
                    Ok(un(IrUnOp::Sqrt, dd, &elem))
                } else {
                    Ok(un(IrUnOp::Abs, d, &elem))
                }
            }
            M::Normalize => {
                let a = self.stable(ctx, a, &ta, "norm");
                let elem = KirType::Scalar(ta.elem_scalar().unwrap_or(ScalarTy::F32));
                let d = self.dot(ctx, a.clone(), a.clone(), &ta, &elem);
                let len = un(IrUnOp::Sqrt, d, &elem).0;
                let KirType::Vec { elem: e, lanes } = ta else { return unsupported("normalize of a scalar") };
                let splat = Expr::Splat { elem: e, lanes, val: len.boxed() };
                Ok((Expr::Binary { op: IrBinOp::Div, lhs: a.boxed(), rhs: splat.boxed(), ty: ta.clone() }, ta))
            }
            M::Cross => {
                let (b, _) = b.expect("second argument");
                let (a, b) = (self.stable(ctx, a, &ta, "cross_a"), self.stable(ctx, b, &ta, "cross_b"));
                let elem = KirType::Scalar(ta.elem_scalar().unwrap_or(ScalarTy::F32));
                let lane = |e: &Expr, k: u32| Expr::Access { base: e.clone().boxed(), step: AccessStep::Field(k), ty: elem.clone() };
                let comp = |i: u32, j: u32| sub(mulx(lane(&a, i), lane(&b, j), &elem), mulx(lane(&a, j), lane(&b, i), &elem), &elem);
                Ok((Expr::Compose { ty: ta.clone(), comps: vec![comp(1, 2), comp(2, 0), comp(0, 1)] }, ta))
            }
            other => unsupported(format!("math function {other:?}")),
        }
    }

    // Matrices. A `matCxR<T>` is an array of C column vectors of R lanes; its operators expand to
    // per-column vector arithmetic, with operands named once (`stable`) so none is re-evaluated.

    /// `(columns, rows, scalar)` of a matrix type.
    fn mat_shape(t: &KirType) -> Option<(u32, u8, ScalarTy)> {
        match t {
            KirType::Array { elem, len, .. } => match **elem {
                KirType::Vec { elem: s, lanes } => Some((*len, lanes, s)),
                _ => None,
            },
            _ => None,
        }
    }

    fn mat_ty(&self, s: ScalarTy, cols: u32, rows: u8) -> KirType {
        let col = KirType::vec(s, rows);
        let (size, align) = col.layout(&self.structs);
        KirType::Array { elem: Box::new(col), len: cols, stride: size.next_multiple_of(align) }
    }

    fn column(m: &Expr, k: u32, col: &KirType) -> Expr {
        Expr::Access { base: m.clone().boxed(), step: AccessStep::Index(Expr::LitU32(k).boxed()), ty: col.clone() }
    }

    fn lane(v: Expr, k: u32, s: ScalarTy) -> Expr {
        Expr::Access { base: v.boxed(), step: AccessStep::Field(k), ty: KirType::Scalar(s) }
    }

    fn binary(op: IrBinOp, l: Expr, r: Expr, ty: &KirType) -> Expr {
        Expr::Binary { op, lhs: l.boxed(), rhs: r.boxed(), ty: ty.clone() }
    }

    /// `m * v`: the sum of the columns scaled by the lanes of `v`.
    fn mat_vec(m: &Expr, cols: u32, col: &KirType, s: ScalarTy, v: &Expr) -> Expr {
        let mut acc: Option<Expr> = None;
        for k in 0..cols {
            let term = Self::binary(IrBinOp::Mul, Self::column(m, k, col), Self::lane(v.clone(), k, s), col);
            acc = Some(match acc {
                None => term,
                Some(a) => Self::binary(IrBinOp::Add, a, term, col),
            });
        }
        acc.expect("matrix with columns")
    }


    fn matrix_binary(&mut self, ctx: &mut FnCtx<'m>, op: naga::BinaryOperator, l: Expr, lt: KirType, r: Expr, rt: KirType) -> R<(Expr, KirType)> {
        use naga::BinaryOperator as B;
        match (op, Self::mat_shape(&lt), Self::mat_shape(&rt)) {
            (B::Add | B::Subtract, Some(shape @ (cols, rows, s)), Some(other)) if other == shape => {
                let ir = if matches!(op, B::Add) { IrBinOp::Add } else { IrBinOp::Sub };
                let col = KirType::vec(s, rows);
                let l = self.stable(ctx, l, &lt, "ma");
                let r = self.stable(ctx, r, &rt, "mb");
                let comps = (0..cols).map(|k| Self::binary(ir, Self::column(&l, k, &col), Self::column(&r, k, &col), &col)).collect();
                Ok((Expr::Compose { ty: lt.clone(), comps }, lt))
            }
            (B::Multiply, Some((cols, rows, s)), None) => {
                let col = KirType::vec(s, rows);
                let m = self.stable(ctx, l, &lt, "ma");
                if rt.scalar() == Some(s) {
                    let k = self.stable(ctx, r, &rt, "ms");
                    let comps = (0..cols).map(|j| Self::binary(IrBinOp::Mul, Self::column(&m, j, &col), k.clone(), &col)).collect();
                    return Ok((Expr::Compose { ty: lt.clone(), comps }, lt));
                }
                match &rt {
                    KirType::Vec { elem, lanes } if *elem == s && u32::from(*lanes) == cols => {
                        let v = self.stable(ctx, r, &rt, "mv");
                        Ok((Self::mat_vec(&m, cols, &col, s, &v), col))
                    }
                    _ => unsupported(format!("matrix product of {lt:?} and {rt:?}")),
                }
            }
            (B::Multiply, None, Some((cols, rows, s))) => {
                let col = KirType::vec(s, rows);
                let a = self.stable(ctx, l, &lt, "mv");
                let m = self.stable(ctx, r, &rt, "ma");
                if lt.scalar() == Some(s) {
                    let comps = (0..cols).map(|j| Self::binary(IrBinOp::Mul, a.clone(), Self::column(&m, j, &col), &col)).collect();
                    return Ok((Expr::Compose { ty: rt.clone(), comps }, rt));
                }
                match &lt {
                    // `v * m`: lane `j` is `dot(v, column j)`.
                    KirType::Vec { elem, lanes } if *elem == s && *lanes == rows => {
                        let mut comps = Vec::with_capacity(cols as usize);
                        for j in 0..cols {
                            comps.push(self.dot(ctx, a.clone(), Self::column(&m, j, &col), &lt, &KirType::Scalar(s)));
                        }
                        let ty = KirType::vec(s, cols as u8);
                        Ok((Expr::Compose { ty: ty.clone(), comps }, ty))
                    }
                    _ => unsupported(format!("matrix product of {lt:?} and {rt:?}")),
                }
            }
            (B::Multiply, Some((c1, r1, s)), Some((c2, r2, s2))) if s == s2 && u32::from(r2) == c1 => {
                let (lcol, rcol) = (KirType::vec(s, r1), KirType::vec(s, r2));
                let m = self.stable(ctx, l, &lt, "ma");
                let n = self.stable(ctx, r, &rt, "mb");
                let mut comps = Vec::with_capacity(c2 as usize);
                for k in 0..c2 {
                    let v = self.stable(ctx, Self::column(&n, k, &rcol), &rcol, "mc");
                    comps.push(Self::mat_vec(&m, c1, &lcol, s, &v));
                }
                let ty = self.mat_ty(s, c2, r1);
                Ok((Expr::Compose { ty: ty.clone(), comps }, ty))
            }
            _ => unsupported(format!("matrix operation {op:?} on {lt:?} and {rt:?}")),
        }
    }

    fn matrix_cast(
        &mut self,
        ctx: &mut FnCtx<'m>,
        e: Expr,
        src: KirType,
        (cols, rows, from): (u32, u8, ScalarTy),
        kind: naga::ScalarKind,
        convert: Option<u8>,
    ) -> R<(Expr, KirType)> {
        let Some(width) = convert else { return unsupported("a bitcast of a matrix") };
        let to = self.scalar(naga::Scalar { kind, width })?;
        if to == from {
            return Ok((e, src));
        }
        let (from_col, to_col) = (KirType::vec(from, rows), KirType::vec(to, rows));
        let m = self.stable(ctx, e, &src, "mc");
        let comps = (0..cols)
            .map(|k| Expr::Cast { kind: CastKind::Convert, expr: Self::column(&m, k, &from_col).boxed(), src_ty: from_col.clone(), dst_ty: to_col.clone() })
            .collect();
        let dst = self.mat_ty(to, cols, rows);
        Ok((Expr::Compose { ty: dst.clone(), comps }, dst))
    }

    fn matrix_transpose(&mut self, ctx: &mut FnCtx<'m>, m: Expr, ty: KirType) -> R<(Expr, KirType)> {
        let Some((cols, rows, s)) = Self::mat_shape(&ty) else { return unsupported("transpose of a non-matrix") };
        let m = self.stable(ctx, m, &ty, "mt");
        let (col, row) = (KirType::vec(s, rows), KirType::vec(s, cols as u8));
        let comps = (0..u32::from(rows))
            .map(|j| Expr::Compose { ty: row.clone(), comps: (0..cols).map(|k| Self::lane(Self::column(&m, k, &col), j, s)).collect() })
            .collect();
        let dst = self.mat_ty(s, u32::from(rows), cols as u8);
        Ok((Expr::Compose { ty: dst.clone(), comps }, dst))
    }

    fn matrix_determinant(&mut self, ctx: &mut FnCtx<'m>, m: Expr, ty: KirType) -> R<(Expr, KirType)> {
        let Some((cols, rows, s)) = Self::mat_shape(&ty) else { return unsupported("determinant of a non-matrix") };
        if cols != u32::from(rows) {
            return unsupported("determinant of a non-square matrix");
        }
        let m = self.stable(ctx, m, &ty, "md");
        let (col, scalar) = (KirType::vec(s, rows), KirType::Scalar(s));
        let entry = |r: u32, c: u32| Self::lane(Self::column(&m, c, &col), r, s);
        let all: Vec<u32> = (0..cols).collect();
        Ok((Self::det(&entry, &all, &all, &scalar), scalar))
    }

    /// Laplace expansion along the first remaining row; `entry(row, column)`.
    fn det(entry: &dyn Fn(u32, u32) -> Expr, rows: &[u32], cols: &[u32], ty: &KirType) -> Expr {
        if let ([r], [c]) = (rows, cols) {
            return entry(*r, *c);
        }
        let mut acc: Option<Expr> = None;
        for (j, &c) in cols.iter().enumerate() {
            let rest: Vec<u32> = cols.iter().copied().filter(|&x| x != c).collect();
            let term = Self::binary(IrBinOp::Mul, entry(rows[0], c), Self::det(entry, &rows[1..], &rest, ty), ty);
            acc = Some(match acc {
                None => term,
                Some(a) => Self::binary(if j % 2 == 1 { IrBinOp::Sub } else { IrBinOp::Add }, a, term, ty),
            });
        }
        acc.expect("square matrix")
    }

    /// `dot(a, b)` as the sum of the lane products.
    fn dot(&mut self, ctx: &mut FnCtx<'m>, a: Expr, b: Expr, vec: &KirType, elem: &KirType) -> Expr {
        let lanes = u32::from(vec.lanes());
        let (a, b) = (self.stable(ctx, a, vec, "dot_a"), self.stable(ctx, b, vec, "dot_b"));
        let lane = |e: &Expr, k: u32| Expr::Access { base: e.clone().boxed(), step: AccessStep::Field(k), ty: elem.clone() };
        let mut sum: Option<Expr> = None;
        for k in 0..lanes {
            let prod = Expr::Binary { op: IrBinOp::Mul, lhs: lane(&a, k).boxed(), rhs: lane(&b, k).boxed(), ty: elem.clone() };
            sum = Some(match sum {
                None => prod,
                Some(s) => Expr::Binary { op: IrBinOp::Add, lhs: s.boxed(), rhs: prod.boxed(), ty: elem.clone() },
            });
        }
        sum.expect("vector with lanes")
    }
}

struct Lowered {
    name: String,
    params: Vec<(String, KirType)>,
    ret_ty: Option<KirType>,
    locals: Vec<(String, KirType, bool)>,
    body: Vec<Stmt>,
    builtins: Vec<(BuiltinVar, LocalId)>,
}

impl Lowered {
    fn into_function(self) -> KirFunction {
        KirFunction { name: self.name, params: self.params, ret_ty: self.ret_ty, locals: self.locals, body: self.body }
    }
}
