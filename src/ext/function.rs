//! Custom elementwise functions: a WGSL scalar function that becomes an operation fusing with its neighbours like `sin` or `max`.

use std::marker::PhantomData;
use std::panic::Location;

use luma_core::naga::{self, Expression, ScalarKind, Statement, TypeInner};

use crate::array::Array;
use crate::context::{default_context, intern, Context};
use crate::error::{LumaError, Result};
use crate::gpu_type::ScalarType;
use crate::jit::{calls, ElemTy, Op, OpKind, Scalar, Src};
use crate::ops::{broadcast_shape, lazy, Operand, Outcome, Rhs};
use crate::shape::Shape;

mod sealed {
    pub trait Sealed {}
}

/// Element types an [`ElementwiseFn`] computes on: `f32`, `u32` and `i32`. Sealed.
pub trait ElemFnType: ScalarType + sealed::Sealed {
    #[doc(hidden)]
    const WGSL: &'static str;
    #[doc(hidden)]
    const KIND: (ScalarKind, u8);
}

macro_rules! elem_fn_type {
    ($t:ty, $wgsl:literal, $kind:expr) => {
        impl sealed::Sealed for $t {}
        impl ElemFnType for $t {
            const WGSL: &'static str = $wgsl;
            const KIND: (ScalarKind, u8) = ($kind, 4);
        }
    };
}
elem_fn_type!(f32, "f32", ScalarKind::Float);
elem_fn_type!(u32, "u32", ScalarKind::Uint);
elem_fn_type!(i32, "i32", ScalarKind::Sint);

/// A WGSL function of one to three scalars, registered with [`Context::elementwise_fn`]: an array operation that fuses with the
/// operations around it into one kernel, like a built-in one. Cheap to clone.
///
/// The function is pure and is called once per element; the result is exactly what the WGSL computes on each backend (the
/// backend compiler may contract `a * b + c` into an FMA, as it does in fused chains of built-ins).
pub struct ElementwiseFn<T> {
    ctx: Context,
    id: u16,
    name: &'static str,
    _type: PhantomData<fn(T) -> T>,
}

impl<T> Clone for ElementwiseFn<T> {
    fn clone(&self) -> Self {
        ElementwiseFn { ctx: self.ctx.clone(), id: self.id, name: self.name, _type: PhantomData }
    }
}

impl<T> std::fmt::Debug for ElementwiseFn<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ElementwiseFn").field("name", &self.name).field("arity", &calls::def(self.id).arity).finish()
    }
}

impl Context {
    /// Registers the WGSL function `name` (defined in `wgsl` together with its helpers) as an elementwise operation on `T`.
    ///
    /// The function takes one to three parameters and returns a value, all of the same scalar type `T` (`f32`, `u32` or
    /// `i32`). It must be pure: no `var<storage>`, `var<uniform>`, `var<workgroup>` or `var<private>`, no arrays, pointers,
    /// atomics, barriers, subgroup, derivative or texture operations; it calls only its own helpers and WGSL built-ins.
    /// Helper functions and module constants are named `name` or `name_*`, so that two functions in one fused kernel cannot
    /// collide. Registering the same name again with the same source gives the same function; with another source, an error.
    /// Violations are [`LumaError::ShaderCompile`].
    pub fn elementwise_fn<T: ElemFnType>(&self, name: &str, wgsl: &str) -> Result<ElementwiseFn<T>> {
        let arity = validate::<T>(name, wgsl)?;
        let id = calls::register(name, wgsl, arity)?;
        let interned = intern(name.to_owned());
        let mut known = self.ext_registry().functions.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        match known.get(name) {
            Some(&earlier) if earlier != id => {
                return Err(LumaError::ShaderCompile(format!("elementwise function `{name}` is already registered with another source")));
            }
            Some(_) => {}
            None => {
                known.insert(name.to_owned(), id);
            }
        }
        Ok(ElementwiseFn { ctx: self.clone(), id, name: interned, _type: PhantomData })
    }
}

/// [`Context::elementwise_fn`] on the [default context](crate::default_context).
pub fn elementwise_fn<T: ElemFnType>(name: &str, wgsl: &str) -> Result<ElementwiseFn<T>> {
    default_context()?.elementwise_fn(name, wgsl)
}

impl<T: ElemFnType> ElementwiseFn<T> {
    /// The cost model's weight of one call, in `f32` operations (default 4, about a mid-sized special function). It decides
    /// how much of an expression the planner fuses before it cuts. The weight belongs to the function, which every context
    /// of the process shares.
    #[must_use]
    pub fn weight(self, weight: f32) -> Self {
        if weight.is_finite() && weight >= 0.0 {
            calls::def(self.id).set_weight(weight);
        }
        self
    }

    /// The name the function was registered under.
    pub fn name(&self) -> &str {
        self.name
    }

    /// `f(a)` of every element; the function takes one parameter.
    #[track_caller]
    pub fn call1(&self, a: impl Operand<T>) -> Array<T> {
        self.call(&[a.rhs()], Location::caller())
    }

    /// `f(a, b)` of every element; the function takes two parameters. Operands broadcast along their extent-1 axes.
    #[track_caller]
    pub fn call2(&self, a: impl Operand<T>, b: impl Operand<T>) -> Array<T> {
        self.call(&[a.rhs(), b.rhs()], Location::caller())
    }

    /// `f(a, b, c)` of every element; the function takes three parameters.
    #[track_caller]
    pub fn call3(&self, a: impl Operand<T>, b: impl Operand<T>, c: impl Operand<T>) -> Array<T> {
        self.call(&[a.rhs(), b.rhs(), c.rhs()], Location::caller())
    }

    fn call(&self, operands: &[Rhs<'_, T>], location: &'static Location<'static>) -> Array<T> {
        let mut shape = None;
        let result = self.build(operands, &mut shape, location);
        Array::resolve(&self.ctx, shape.unwrap_or(Shape::from(1)), result, self.name, location)
    }

    fn build(&self, operands: &[Rhs<'_, T>], shape: &mut Option<Shape>, location: &'static Location<'static>) -> Outcome<Array<T>> {
        let def = calls::def(self.id);
        if operands.len() != def.arity as usize {
            return Err(LumaError::InvalidShape(format!("`{}` takes {} operands, got {}", self.name, def.arity, operands.len())).into());
        }
        for operand in operands {
            if let Rhs::Array(array) = operand {
                if !self.ctx.same_as(array.context()) {
                    return Err(LumaError::ContextMismatch.into());
                }
                let target = match *shape {
                    None => array.shape(),
                    Some(so_far) => broadcast_shape(so_far, array.shape()).ok_or(LumaError::ShapeMismatch { lhs: so_far.dims(), rhs: array.shape().dims() })?,
                };
                *shape = Some(target);
            }
        }
        let Some(target) = *shape else {
            return Err(LumaError::InvalidShape(format!("`{}` needs at least one array operand", self.name)).into());
        };
        let mut srcs = [Src::unused(), Src::unused(), Src::unused()];
        for (slot, operand) in srcs.iter_mut().zip(operands) {
            *slot = match operand {
                Rhs::Array(array) => array.broadcast_operand(target, location)?,
                Rhs::Scalar(value) => Src::Scalar(Scalar::of(*value)?),
            };
        }
        let op = Op::new(OpKind::Call { id: self.id, arity: def.arity }, ElemTy::of::<T>()?);
        lazy(&self.ctx, target, op, srcs, self.name, location)
    }
}

fn refuse(message: impl std::fmt::Display) -> LumaError {
    LumaError::ShaderCompile(message.to_string())
}

/// Checks `wgsl` defines `name` as a pure function of one to three `T` scalars returning `T`; the arity.
fn validate<T: ElemFnType>(name: &str, wgsl: &str) -> Result<u8> {
    if name.is_empty() || !name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') || name.starts_with(|c: char| c.is_ascii_digit()) {
        return Err(refuse(format!("`{name}` is not a function name")));
    }
    if name.starts_with("luma_") || name.starts_with("df_") || name.starts_with("__") {
        return Err(refuse(format!("`{name}`: names starting with `luma_`, `df_` or `__` belong to the library")));
    }
    let module = naga::front::wgsl::parse_str(wgsl).map_err(|e| refuse(e.emit_to_string(wgsl)))?;
    naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::empty())
        .validate(&module)
        .map_err(|e| refuse(e.emit_to_string(wgsl)))?;

    if !module.entry_points.is_empty() {
        return Err(refuse("the source of an elementwise function has no entry points"));
    }
    if let Some((_, global)) = module.global_variables.iter().next() {
        return Err(refuse(format!("an elementwise function is pure: it declares the global variable `{}` ({:?})", global.name.as_deref().unwrap_or("?"), global.space)));
    }
    if !module.overrides.is_empty() {
        return Err(refuse("an elementwise function has no pipeline-overridable constants"));
    }
    let owned = |item: &str| item == name || item.strip_prefix(name).is_some_and(|rest| rest.starts_with('_'));
    for (_, constant) in module.constants.iter() {
        if let Some(item) = &constant.name
            && !owned(item)
        {
            return Err(refuse(format!("the constant `{item}` must be named `{name}` or `{name}_*`")));
        }
    }
    for (_, ty) in module.types.iter() {
        match &ty.inner {
            TypeInner::Scalar(_) | TypeInner::Vector { .. } | TypeInner::Matrix { .. } => {}
            TypeInner::Struct { .. } if ty.name.as_deref().is_some_and(|n| n.starts_with("__") || owned(n)) => {}
            TypeInner::Array { .. } | TypeInner::BindingArray { .. } => return Err(refuse("an elementwise function uses no arrays")),
            TypeInner::Pointer { .. } | TypeInner::ValuePointer { .. } => return Err(refuse("an elementwise function uses no pointers")),
            TypeInner::Atomic(_) => return Err(refuse("an elementwise function uses no atomics")),
            TypeInner::Image { .. } | TypeInner::Sampler { .. } => return Err(refuse("an elementwise function uses no textures")),
            other => return Err(refuse(format!("an elementwise function cannot use the type {other:?}; structs are named `{name}_*`"))),
        }
    }

    let (kind, width) = T::KIND;
    let is_t = |ty: naga::Handle<naga::Type>| matches!(module.types[ty].inner, TypeInner::Scalar(s) if s.kind == kind && s.width == width);
    let mut arity = None;
    for (_, function) in module.functions.iter() {
        let Some(function_name) = function.name.as_deref() else {
            return Err(refuse("an elementwise function has no anonymous functions"));
        };
        if !owned(function_name) {
            return Err(refuse(format!("the function `{function_name}` must be named `{name}` or `{name}_*`")));
        }
        for (_, expression) in function.expressions.iter() {
            match expression {
                Expression::Literal(_)
                | Expression::Constant(_)
                | Expression::ZeroValue(_)
                | Expression::Compose { .. }
                | Expression::Access { .. }
                | Expression::AccessIndex { .. }
                | Expression::Splat { .. }
                | Expression::Swizzle { .. }
                | Expression::FunctionArgument(_)
                | Expression::LocalVariable(_)
                | Expression::Load { .. }
                | Expression::Unary { .. }
                | Expression::Binary { .. }
                | Expression::Select { .. }
                | Expression::Relational { .. }
                | Expression::Math { .. }
                | Expression::As { .. }
                | Expression::CallResult(_) => {}
                Expression::Derivative { .. } => return Err(refuse("an elementwise function uses no derivatives")),
                Expression::ImageSample { .. } | Expression::ImageLoad { .. } | Expression::ImageQuery { .. } => return Err(refuse("an elementwise function uses no texture operations")),
                Expression::AtomicResult { .. } => return Err(refuse("an elementwise function uses no atomics")),
                Expression::SubgroupBallotResult | Expression::SubgroupOperationResult { .. } => return Err(refuse("an elementwise function uses no subgroup operations")),
                Expression::GlobalVariable(_) | Expression::ArrayLength(_) | Expression::WorkGroupUniformLoadResult { .. } => return Err(refuse("an elementwise function reads no global memory")),
                other => return Err(refuse(format!("an elementwise function cannot use the expression {other:?}"))),
            }
        }
        check_block(&function.body)?;
        if function_name == name {
            let count = function.arguments.len();
            if !(1..=3).contains(&count) {
                return Err(refuse(format!("`{name}` takes {count} parameters, an elementwise function takes 1 to 3")));
            }
            if let Some(argument) = function.arguments.iter().find(|argument| !is_t(argument.ty)) {
                return Err(refuse(format!("`{name}`: every parameter and the result must be {} (parameter {:?} is {:?})", T::WGSL, argument.name, module.types[argument.ty].inner)));
            }
            match &function.result {
                Some(result) if is_t(result.ty) => {}
                _ => return Err(refuse(format!("`{name}` must return {}", T::WGSL))),
            }
            arity = Some(count as u8);
        }
    }
    arity.ok_or_else(|| refuse(format!("the source defines no function `{name}`")))
}

fn check_block(block: &naga::Block) -> Result<()> {
    for statement in block {
        match statement {
            Statement::Emit(_) | Statement::Break | Statement::Continue | Statement::Return { .. } | Statement::Store { .. } | Statement::Call { .. } => {}
            Statement::Block(inner) => check_block(inner)?,
            Statement::If { accept, reject, .. } => {
                check_block(accept)?;
                check_block(reject)?;
            }
            Statement::Switch { cases, .. } => {
                for case in cases {
                    check_block(&case.body)?;
                }
            }
            Statement::Loop { body, continuing, .. } => {
                check_block(body)?;
                check_block(continuing)?;
            }
            Statement::ControlBarrier(_) | Statement::MemoryBarrier(_) => return Err(refuse("an elementwise function uses no barriers")),
            Statement::Atomic { .. } | Statement::ImageAtomic { .. } => return Err(refuse("an elementwise function uses no atomics")),
            Statement::ImageStore { .. } => return Err(refuse("an elementwise function uses no texture operations")),
            Statement::SubgroupBallot { .. } | Statement::SubgroupGather { .. } | Statement::SubgroupCollectiveOperation { .. } => {
                return Err(refuse("an elementwise function uses no subgroup operations"));
            }
            other => return Err(refuse(format!("an elementwise function cannot use the statement {other:?}"))),
        }
    }
    Ok(())
}
