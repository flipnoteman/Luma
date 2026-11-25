//! Arrays that forget their element type: [`AnyArray`] is what a function taking arrays of several
//! types (a compiled callable's arguments, a registered operation's operands) is handed, and
//! [`Dyn`] is the owned form that comes back. The element type is kept as an [`ElemKind`] and
//! checked when an array is turned back into an [`Array<T>`] ([`Dyn::take`]).

use std::any::Any;
use std::fmt;
use std::sync::Arc;

use crate::array::{Array, Failure, Storage};
use crate::context::Context;
use crate::error::{LumaError, Result};
use crate::gpu_type::GpuType;
use crate::jit::Node;
use crate::shape::Shape;

/// The element type of an array, as a value.
///
/// The `Display` form is the Rust name: `"f32"`, `"u32"`, `"bf16"`, `"[f32; 4]"`, ...
#[non_exhaustive]
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum ElemKind {
    /// `f32`.
    F32,
    /// `u32`.
    U32,
    /// `i32`.
    I32,
    /// `bool` (one bit per element).
    Bool,
    /// `u8`.
    U8,
    /// `i8`.
    I8,
    /// `u16`.
    U16,
    /// `i16`.
    I16,
    /// `f16`.
    F16,
    /// `bf16`.
    Bf16,
    /// `f64`.
    F64,
    /// [`Df32`](crate::Df32).
    Df32,
    /// [`C32`](crate::C32).
    C32,
    /// `[f32; 2]`.
    F32x2,
    /// `[f32; 4]`.
    F32x4,
    /// `[u32; 2]`.
    U32x2,
    /// `[u32; 4]`.
    U32x4,
}

impl ElemKind {
    /// The kind of `T`.
    pub(crate) const fn of<T: GpuType>() -> ElemKind {
        T::ELEM_KIND
    }

    const fn name(self) -> &'static str {
        match self {
            ElemKind::F32 => "f32",
            ElemKind::U32 => "u32",
            ElemKind::I32 => "i32",
            ElemKind::Bool => "bool",
            ElemKind::U8 => "u8",
            ElemKind::I8 => "i8",
            ElemKind::U16 => "u16",
            ElemKind::I16 => "i16",
            ElemKind::F16 => "f16",
            ElemKind::Bf16 => "bf16",
            ElemKind::F64 => "f64",
            ElemKind::Df32 => "df32",
            ElemKind::C32 => "c32",
            ElemKind::F32x2 => "[f32; 2]",
            ElemKind::F32x4 => "[f32; 4]",
            ElemKind::U32x2 => "[u32; 2]",
            ElemKind::U32x4 => "[u32; 4]",
        }
    }
}

impl fmt::Display for ElemKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// An array's storage, as [`AnyArray::storage_ref`] hands it out (only crate-internal code looks inside).
#[doc(hidden)]
pub struct StorageRef<'a>(pub(crate) &'a Arc<Storage>);

mod sealed {
    pub trait Sealed {}
}

impl<T: GpuType> sealed::Sealed for Array<T> {}
impl sealed::Sealed for Dyn {}

/// An array of any element type: implemented by [`Array<T>`] for every [`GpuType`] and by
/// [`Dyn`]. It cannot be implemented outside luma.
pub trait AnyArray: sealed::Sealed + Send + Sync {
    /// The shape of the array.
    fn shape(&self) -> Shape;

    /// The element type of the array.
    fn elem(&self) -> ElemKind;

    /// The context the array lives on.
    fn context(&self) -> &Context;

    /// Another handle on the array that does not name its element type: forces the storage and
    /// clones the handle.
    #[doc(hidden)]
    fn erased(&self) -> Erased;

    /// The array's storage, evaluating the array first if it is still lazy: no clone of the
    /// handle, for the callers that only look at the data.
    #[doc(hidden)]
    fn storage_ref(&self) -> std::result::Result<StorageRef<'_>, Arc<LumaError>>;
}

impl<T: GpuType> AnyArray for Array<T> {
    fn shape(&self) -> Shape {
        Array::shape(self)
    }

    fn elem(&self) -> ElemKind {
        ElemKind::of::<T>()
    }

    fn context(&self) -> &Context {
        Array::context(self)
    }

    fn erased(&self) -> Erased {
        Erased { handle: Box::new(self.clone()) }
    }

    fn storage_ref(&self) -> std::result::Result<StorageRef<'_>, Arc<LumaError>> {
        Array::storage(self).map(StorageRef).map_err(Failure::into_shared)
    }
}

impl AnyArray for Dyn {
    fn shape(&self) -> Shape {
        self.erased.shape()
    }

    fn elem(&self) -> ElemKind {
        self.erased.elem()
    }

    fn context(&self) -> &Context {
        self.erased.context()
    }

    fn erased(&self) -> Erased {
        self.erased.clone()
    }

    fn storage_ref(&self) -> std::result::Result<StorageRef<'_>, Arc<LumaError>> {
        self.erased.storage_ref()
    }
}

/// What an [`Erased`] array does without knowing its type.
#[allow(dead_code)] // the callables and registered operations that use all of it build on this
pub(crate) trait Handle: Send + Sync {
    fn shape(&self) -> Shape;
    fn elem(&self) -> ElemKind;
    fn context(&self) -> &Context;
    fn error(&self) -> Option<&LumaError>;
    fn storage(&self) -> std::result::Result<&Arc<Storage>, Failure>;
    fn pending(&self) -> Option<&Arc<Node>>;
    fn passes_through(&self, probe: &Arc<Storage>) -> bool;
    /// A new array of the same type and shape with a new buffer of unspecified contents.
    fn fresh_probe(&self) -> std::result::Result<Box<dyn Handle>, Failure>;
    fn clone_handle(&self) -> Box<dyn Handle>;
    fn into_any(self: Box<Self>) -> Box<dyn Any>;
}

impl<T: GpuType> Handle for Array<T> {
    fn shape(&self) -> Shape {
        Array::shape(self)
    }

    fn elem(&self) -> ElemKind {
        ElemKind::of::<T>()
    }

    fn context(&self) -> &Context {
        Array::context(self)
    }

    fn error(&self) -> Option<&LumaError> {
        Array::error(self)
    }

    fn storage(&self) -> std::result::Result<&Arc<Storage>, Failure> {
        Array::storage(self)
    }

    fn pending(&self) -> Option<&Arc<Node>> {
        Array::pending(self)
    }

    fn passes_through(&self, probe: &Arc<Storage>) -> bool {
        Array::passes_through(self, probe)
    }

    fn fresh_probe(&self) -> std::result::Result<Box<dyn Handle>, Failure> {
        Ok(Box::new(Array::fresh_probe(self)?))
    }

    fn clone_handle(&self) -> Box<dyn Handle> {
        Box::new(self.clone())
    }

    fn into_any(self: Box<Self>) -> Box<dyn Any> {
        self
    }
}

/// A handle on an array whose element type is a value ([`ElemKind`]) instead of a type
/// parameter, with the operations the library needs without naming it. Crate-internal: the
/// public form is [`Dyn`].
#[doc(hidden)]
pub struct Erased {
    handle: Box<dyn Handle>,
}

impl Clone for Erased {
    fn clone(&self) -> Self {
        Erased { handle: self.handle.clone_handle() }
    }
}

impl fmt::Debug for Erased {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Erased").field("elem", &self.elem()).field("shape", &self.shape()).finish()
    }
}

#[allow(dead_code)] // see `Handle`
impl Erased {
    pub(crate) fn shape(&self) -> Shape {
        self.handle.shape()
    }

    pub(crate) fn elem(&self) -> ElemKind {
        self.handle.elem()
    }

    pub(crate) fn context(&self) -> &Context {
        self.handle.context()
    }

    /// The error that poisoned the array, if one did.
    pub(crate) fn error(&self) -> Option<&LumaError> {
        self.handle.error()
    }

    /// The array's storage, evaluating the array first if it is still lazy.
    pub(crate) fn storage(&self) -> std::result::Result<&Arc<Storage>, Failure> {
        self.handle.storage()
    }

    /// [`Erased::storage`] for [`AnyArray::storage_ref`].
    pub(crate) fn storage_ref(&self) -> std::result::Result<StorageRef<'_>, Arc<LumaError>> {
        self.handle.storage().map(StorageRef).map_err(Failure::into_shared)
    }

    /// The expression behind the array if it has not run yet.
    pub(crate) fn pending(&self) -> Option<&Arc<Node>> {
        self.handle.pending()
    }

    /// See [`Array::passes_through`].
    pub(crate) fn passes_through(&self, probe: &Arc<Storage>) -> bool {
        self.handle.passes_through(probe)
    }

    /// A new array of the same type and shape with a new buffer of unspecified contents (see
    /// `Array::fresh_probe`).
    pub(crate) fn fresh_probe(&self) -> std::result::Result<Erased, Failure> {
        Ok(Erased { handle: self.handle.fresh_probe()? })
    }

    /// An array of element type `kind` viewing `storage` as `shape`.
    pub(crate) fn from_storage(kind: ElemKind, ctx: &Context, shape: Shape, storage: Arc<Storage>) -> Erased {
        fn make<T: GpuType>(ctx: &Context, shape: Shape, storage: Arc<Storage>) -> Erased {
            Erased { handle: Box::new(Array::<T>::from_storage(ctx, shape, storage)) }
        }
        match kind {
            ElemKind::F32 => make::<f32>(ctx, shape, storage),
            ElemKind::U32 => make::<u32>(ctx, shape, storage),
            ElemKind::I32 => make::<i32>(ctx, shape, storage),
            ElemKind::Bool => make::<bool>(ctx, shape, storage),
            ElemKind::U8 => make::<u8>(ctx, shape, storage),
            ElemKind::I8 => make::<i8>(ctx, shape, storage),
            ElemKind::U16 => make::<u16>(ctx, shape, storage),
            ElemKind::I16 => make::<i16>(ctx, shape, storage),
            ElemKind::F16 => make::<half::f16>(ctx, shape, storage),
            ElemKind::Bf16 => make::<half::bf16>(ctx, shape, storage),
            ElemKind::F64 => make::<f64>(ctx, shape, storage),
            ElemKind::Df32 => make::<crate::gpu_type::Df32>(ctx, shape, storage),
            ElemKind::C32 => make::<crate::complex::C32>(ctx, shape, storage),
            ElemKind::F32x2 => make::<[f32; 2]>(ctx, shape, storage),
            ElemKind::F32x4 => make::<[f32; 4]>(ctx, shape, storage),
            ElemKind::U32x2 => make::<[u32; 2]>(ctx, shape, storage),
            ElemKind::U32x4 => make::<[u32; 4]>(ctx, shape, storage),
        }
    }

    /// The array as an [`Array<T>`], or `None` if the element type is not `T`.
    pub(crate) fn downcast<T: GpuType>(self) -> std::result::Result<Array<T>, Erased> {
        if self.elem() != ElemKind::of::<T>() {
            return Err(self);
        }
        match self.handle.into_any().downcast::<Array<T>>() {
            Ok(array) => Ok(*array),
            Err(_) => unreachable!("an array of kind {} that is not an Array<{}>", ElemKind::of::<T>(), ElemKind::of::<T>()),
        }
    }
}

/// An array that does not name its element type: what a compiled callable returns, and what
/// [`AnyArray`] arguments turn into. [`Dyn::take`] gives the typed [`Array`] back.
pub struct Dyn {
    erased: Erased,
}

#[allow(dead_code)] // see `Handle`
impl Dyn {
    pub(crate) fn new(erased: Erased) -> Dyn {
        Dyn { erased }
    }

    pub(crate) fn as_erased(&self) -> &Erased {
        &self.erased
    }

    pub(crate) fn into_erased(self) -> Erased {
        self.erased
    }
}

impl Dyn {
    /// The shape of the array.
    pub fn shape(&self) -> Shape {
        self.erased.shape()
    }

    /// The element type of the array.
    pub fn elem(&self) -> ElemKind {
        self.erased.elem()
    }

    /// The array as an `Array<T>`. Fails with [`LumaError::InvalidShape`] if it holds another
    /// element type.
    pub fn take<T: GpuType>(self) -> Result<Array<T>> {
        self.erased.downcast::<T>().map_err(|erased| {
            LumaError::InvalidShape(format!("this array holds {}, not {}", erased.elem(), ElemKind::of::<T>()))
        })
    }
}

impl Clone for Dyn {
    /// Another handle on the same data, as [`Array::clone`].
    fn clone(&self) -> Self {
        Dyn { erased: self.erased.clone() }
    }
}

impl fmt::Debug for Dyn {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Dyn").field("elem", &self.elem()).field("shape", &self.shape()).finish()
    }
}

impl<T: GpuType> From<Array<T>> for Dyn {
    fn from(array: Array<T>) -> Dyn {
        Dyn { erased: Erased { handle: Box::new(array) } }
    }
}
