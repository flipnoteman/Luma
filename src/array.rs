use std::marker::PhantomData;
use std::panic::Location;
use std::sync::Arc;

use crate::context::{default_context, Context, STORAGE_USAGE};
use crate::error::{LumaError, Result};
use crate::gpu_type::GpuType;
use crate::shape::Shape;

/// Why an operation on arrays did not produce a result. Internal: operators turn it into a
/// poisoned array, `try_*` methods into a plain error.
pub(crate) enum Failure {
    /// The operation itself failed.
    New(LumaError),
    /// An input was already poisoned; its error (with the location of the operation that
    /// originally failed) is passed on unchanged.
    Inherited(Arc<LumaError>),
}

impl From<LumaError> for Failure {
    fn from(error: LumaError) -> Self {
        Failure::New(error)
    }
}

impl Failure {
    pub(crate) fn into_error(self) -> LumaError {
        match self {
            Failure::New(error) => error,
            Failure::Inherited(error) => LumaError::clone(&error),
        }
    }
}

/// A GPU buffer owned (shared) by arrays; goes back to the context's pool when the last array
/// using it is dropped.
#[derive(Debug)]
pub(crate) struct Storage {
    ctx: Context,
    /// May be larger than the data it holds; only the first `byte_len` bytes are the array.
    buffer: wgpu::Buffer,
}

impl Drop for Storage {
    fn drop(&mut self) {
        // Cloning a wgpu buffer only bumps a reference count; it is still the same buffer.
        self.ctx.recycle(self.buffer.clone());
    }
}

#[derive(Debug)]
enum Repr {
    Valid(Arc<Storage>),
    /// An operation producing this array failed; see [`Array::error`].
    Poisoned(Arc<LumaError>),
}

/// GPU-resident array of up to four dimensions.
///
/// Arrays are created from host data ([`Array::from_slice`], [`Array::zeros`], ...) on the
/// [default context](crate::default_context) or, with the `_on` variants, on an explicit
/// [`Context`]. Math is written like on the host: operators (`&a + &b * 2.0`), methods
/// (`a.sqrt()`) and reductions (`a.sum()?`). Nothing waits for the GPU until a result is needed:
/// [`Array::to_vec`], [`Array::eval`] and the reductions are the synchronization points.
///
/// # Errors do not panic: poisoned arrays
///
/// Operators and math methods return `Array<T>` directly. If one fails (shape mismatch, arrays
/// of different contexts, allocation failure, ...) it returns a *poisoned* array that remembers
/// the error and the source location of the failed operation, and every later operation on it
/// yields a poisoned array carrying the same error. The error comes out as `Err` at the next
/// synchronization point, or earlier through [`Array::error`] / [`Array::check`]. The `try_*`
/// methods return the error immediately instead.
///
/// ```no_run
/// use luma::Array;
/// let a = Array::from_slice(&[1.0f32, 2.0, 3.0], 3)?;
/// let b = Array::from_slice(&[1.0f32, 2.0], 2)?;
/// let c = (&a + &b) * 2.0; // shape mismatch: `c` is poisoned
/// let err = c.to_vec().unwrap_err();
/// println!("{err}"); // `add` failed at src/main.rs:5:14: Dimension mismatch: ...
/// # Ok::<(), luma::LumaError>(())
/// ```
///
/// # Sharing
///
/// [`Array::reshape`] shares the buffer. In-place operations (`+=`), [`Array::write`] and kernels
/// taking the array as `&mut` therefore change every array sharing it.
///
/// Dropping an array returns its buffer to the context's pool for reuse.
#[derive(Debug)]
pub struct Array<T: GpuType> {
    ctx: Context,
    shape: Shape,
    repr: Repr,
    _marker: PhantomData<T>,
}

impl<T: GpuType> Array<T> {
    // -- Construction --

    /// Allocates a valid array of `shape` whose contents are unspecified.
    pub(crate) fn alloc(ctx: &Context, shape: Shape) -> Result<Self> {
        let len = shape.validate()?;
        let bytes = len as u64 * std::mem::size_of::<T>() as u64;
        let buffer = ctx.alloc_buffer(bytes, STORAGE_USAGE)?;
        Ok(Array {
            ctx: ctx.clone(),
            shape,
            repr: Repr::Valid(Arc::new(Storage { ctx: ctx.clone(), buffer })),
            _marker: PhantomData,
        })
    }

    fn poisoned(ctx: &Context, shape: Shape, error: Arc<LumaError>) -> Self {
        Array { ctx: ctx.clone(), shape, repr: Repr::Poisoned(error), _marker: PhantomData }
    }

    /// Turns the outcome of an operation into the array it returns: a failure becomes a poisoned
    /// array of `shape` (a new failure is tagged with the operation and its call site; an
    /// inherited one is passed on untouched).
    pub(crate) fn resolve(
        ctx: &Context,
        shape: Shape,
        result: std::result::Result<Self, Failure>,
        op: &'static str,
        location: &'static Location<'static>,
    ) -> Self {
        match result {
            Ok(array) => array,
            Err(Failure::Inherited(error)) => Self::poisoned(ctx, shape, error),
            Err(Failure::New(error)) => {
                let error = LumaError::Op { op, location, source: Arc::new(error) };
                Self::poisoned(ctx, shape, Arc::new(error))
            }
        }
    }

    /// Creates an array of the given shape on the [default context](crate::default_context) and
    /// uploads `data` (the first dimension varies fastest) into it.
    ///
    /// Every dimension must be non-zero and `data.len()` must equal their product.
    pub fn from_slice(data: &[T], shape: impl Into<Shape>) -> Result<Self> {
        Self::from_slice_on(&default_context()?, data, shape)
    }

    /// [`Array::from_slice`] on an explicit context.
    pub fn from_slice_on(ctx: &Context, data: &[T], shape: impl Into<Shape>) -> Result<Self> {
        let shape = shape.into();
        let len = shape.validate()?;
        if data.len() != len {
            return Err(LumaError::LengthMismatch { expected: len, actual: data.len() });
        }
        let array = Self::alloc(ctx, shape)?;
        let (buffer, _) = array.ready().map_err(Failure::into_error)?;
        ctx.upload(buffer, bytemuck::cast_slice(data))?;
        Ok(array)
    }

    /// An array of the given shape filled with zeros, on the default context.
    pub fn zeros(shape: impl Into<Shape>) -> Result<Self> {
        Self::zeros_on(&default_context()?, shape)
    }

    /// [`Array::zeros`] on an explicit context.
    pub fn zeros_on(ctx: &Context, shape: impl Into<Shape>) -> Result<Self> {
        let array = Self::alloc(ctx, shape.into())?;
        let (buffer, bytes) = array.ready().map_err(Failure::into_error)?;
        // Pooled buffers hold stale data, so zeroing is required.
        ctx.clear(buffer, bytes)?;
        Ok(array)
    }

    // -- Introspection --

    /// The [`Context`] this array lives on.
    pub fn context(&self) -> &Context {
        &self.ctx
    }

    /// The shape of the array. A poisoned array keeps the shape its failed operation was
    /// supposed to give it (the shape of the left operand).
    pub fn shape(&self) -> Shape {
        self.shape
    }

    /// Total number of elements (the product of the shape).
    pub fn len(&self) -> usize {
        self.shape.len()
    }

    /// Always `false`: arrays have at least one element.
    pub fn is_empty(&self) -> bool {
        false
    }

    /// The error this array is poisoned with, if any. It names the operation that failed and
    /// where it was called from.
    pub fn error(&self) -> Option<&LumaError> {
        match &self.repr {
            Repr::Valid(_) => None,
            Repr::Poisoned(error) => Some(error),
        }
    }

    /// `Err` with the error of a poisoned array, `Ok` otherwise. Errors reported by the GPU
    /// itself (as opposed to failed operations) only show up at synchronization points.
    pub fn check(&self) -> Result<()> {
        match &self.repr {
            Repr::Valid(_) => Ok(()),
            Repr::Poisoned(error) => Err(LumaError::clone(error)),
        }
    }

    /// The buffer and the number of bytes of it holding the array, or the reason the array is
    /// not valid.
    pub(crate) fn ready(&self) -> std::result::Result<(&wgpu::Buffer, u64), Failure> {
        match &self.repr {
            Repr::Valid(storage) => Ok((&storage.buffer, self.byte_len())),
            Repr::Poisoned(error) => Err(Failure::Inherited(Arc::clone(error))),
        }
    }

    /// Size of the array's data in bytes (not the size of the possibly larger buffer).
    pub(crate) fn byte_len(&self) -> u64 {
        self.shape.len() as u64 * std::mem::size_of::<T>() as u64
    }

    #[cfg(test)]
    pub(crate) fn buffer(&self) -> Option<&wgpu::Buffer> {
        self.ready().ok().map(|(buffer, _)| buffer)
    }

    // -- Synchronization --

    /// Copies the array back to the host. Submits the pending batch and blocks until the GPU has
    /// finished it, so this is where a poisoned array's error (or an error the GPU reported
    /// while running the batch) comes out.
    pub fn to_vec(&self) -> Result<Vec<T>> {
        let (buffer, bytes) = self.ready().map_err(Failure::into_error)?;
        let raw = self.ctx.read_bytes(buffer, bytes)?;
        Ok(bytemuck::pod_collect_to_vec::<u8, T>(&raw))
    }

    /// Submits the pending batch and waits until the GPU has executed it, without copying
    /// anything back. Returns the array's error if it is poisoned.
    pub fn eval(&self) -> Result<()> {
        self.check()?;
        self.ctx.sync()
    }

    /// Overwrites the array's contents with `data`, which must have exactly [`Array::len`]
    /// elements. The upload is recorded in order with the operations before it and does not wait
    /// for the GPU: operations recorded earlier still see the old contents.
    pub fn write(&mut self, data: &[T]) -> Result<()> {
        let (buffer, _) = self.ready().map_err(Failure::into_error)?;
        if data.len() != self.len() {
            return Err(LumaError::LengthMismatch { expected: self.len(), actual: data.len() });
        }
        self.ctx.upload(buffer, bytemuck::cast_slice(data))
    }

    // -- Views --

    /// The same data with a different shape. Only metadata changes: the result shares the
    /// buffer, no copy is made (see "Sharing" in the type's documentation). The element count
    /// must stay the same; otherwise the result is poisoned.
    #[track_caller]
    pub fn reshape(&self, shape: impl Into<Shape>) -> Array<T> {
        let location = Location::caller();
        let shape = shape.into();
        let result = self.reshaped(shape);
        Self::resolve(&self.ctx, shape, result, "reshape", location)
    }

    fn reshaped(&self, shape: Shape) -> std::result::Result<Array<T>, Failure> {
        let storage = match &self.repr {
            Repr::Valid(storage) => storage,
            Repr::Poisoned(error) => return Err(Failure::Inherited(Arc::clone(error))),
        };
        let len = shape.validate()?;
        if len != self.len() {
            return Err(LumaError::InvalidShape(format!(
                "Cannot reshape {} elements {} into {} elements {shape}",
                self.len(),
                self.shape,
                len
            ))
            .into());
        }
        Ok(Array {
            ctx: self.ctx.clone(),
            shape,
            repr: Repr::Valid(Arc::clone(storage)),
            _marker: PhantomData,
        })
    }

    /// Marks this array poisoned (used by in-place operations that failed).
    pub(crate) fn poison(&mut self, error: Arc<LumaError>) {
        self.repr = Repr::Poisoned(error);
    }
}
