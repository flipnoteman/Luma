use std::marker::PhantomData;
use std::panic::Location;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use crate::context::{default_context, Context, STORAGE_USAGE};
use crate::error::{LumaError, Result};
use crate::gpu_type::GpuType;
use crate::jit::{Node, Readers, Src};
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
    id: u64,
    ctx: Context,
    /// May be larger than the data it holds; only the first `byte_len` bytes are the array.
    buffer: wgpu::Buffer,
    /// Lazy operations that still have to read this buffer; they are evaluated before anything
    /// overwrites it (see [`crate::jit`]).
    pub(crate) readers: Readers,
}

static NEXT_STORAGE_ID: AtomicU64 = AtomicU64::new(0);

impl Storage {
    pub(crate) fn new(ctx: &Context, buffer: wgpu::Buffer) -> Self {
        Storage {
            id: NEXT_STORAGE_ID.fetch_add(1, Ordering::Relaxed),
            ctx: ctx.clone(),
            buffer,
            readers: Readers::default(),
        }
    }

    /// Unique per storage; identifies it as an input of a fused kernel.
    pub(crate) fn id(&self) -> u64 {
        self.id
    }

    pub(crate) fn buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }
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
    /// An elementwise expression that has not run yet; evaluated (once) when its buffer is
    /// needed. See [`crate::jit`].
    Lazy(Arc<Node>),
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
/// # Lazy arrays and fusion
///
/// Elementwise operators and math functions do not run immediately: the result is a *lazy*
/// array holding the expression, and a chain such as `(&x * 2.0 + &y).sqrt()` becomes a single
/// generated kernel when the data is first needed (reading it back, `eval`, `write`, a shift, a
/// convolution, a custom kernel, [`Context::sync`]). Reductions fuse the expression into their
/// first pass. A lazy array behaves as the value its expression had when it was written: later
/// changes of its inputs (`write`, `+=`, kernels taking `&mut`) do not affect it.
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
            repr: Repr::Valid(Arc::new(Storage::new(ctx, buffer))),
            _marker: PhantomData,
        })
    }

    /// An array that is the (not yet evaluated) result of `node`.
    pub(crate) fn from_node(ctx: &Context, shape: Shape, node: Arc<Node>) -> Self {
        Array { ctx: ctx.clone(), shape, repr: Repr::Lazy(node), _marker: PhantomData }
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
        self.failed().map(|error| &**error)
    }

    /// The error of a poisoned array, or of a lazy array whose evaluation failed.
    fn failed(&self) -> Option<&Arc<LumaError>> {
        match &self.repr {
            Repr::Valid(_) => None,
            Repr::Lazy(node) => node.failure(),
            Repr::Poisoned(error) => Some(error),
        }
    }

    /// `Err` with the error of a poisoned array, `Ok` otherwise. Errors reported by the GPU
    /// itself (as opposed to failed operations) only show up at synchronization points.
    pub fn check(&self) -> Result<()> {
        match self.failed() {
            None => Ok(()),
            Some(error) => Err(LumaError::clone(error)),
        }
    }

    /// The array's storage, evaluating the array first if it is still lazy (this records the
    /// work; it does not wait for the GPU). Fails with the reason the array is not valid.
    pub(crate) fn storage(&self) -> std::result::Result<&Arc<Storage>, Failure> {
        match &self.repr {
            Repr::Valid(storage) => Ok(storage),
            Repr::Lazy(node) => {
                node.force().as_ref().map_err(|error| Failure::Inherited(Arc::clone(error)))
            }
            Repr::Poisoned(error) => Err(Failure::Inherited(Arc::clone(error))),
        }
    }

    /// The buffer and the number of bytes of it holding the array, or the reason the array is
    /// not valid. Evaluates a lazy array.
    pub(crate) fn ready(&self) -> std::result::Result<(&wgpu::Buffer, u64), Failure> {
        Ok((self.storage()?.buffer(), self.byte_len()))
    }

    /// This array as an operand of a lazy operation, without evaluating it: its buffer, or the
    /// pending expression that produces it.
    pub(crate) fn src(&self) -> std::result::Result<Src, Failure> {
        match &self.repr {
            Repr::Valid(storage) => Ok(Src::Buffer(Arc::clone(storage))),
            Repr::Lazy(node) => node.as_src(),
            Repr::Poisoned(error) => Err(Failure::Inherited(Arc::clone(error))),
        }
    }

    /// The expression behind this array if it has not been evaluated yet.
    pub(crate) fn pending(&self) -> Option<&Arc<Node>> {
        match &self.repr {
            Repr::Lazy(node) if node.is_pending() => Some(node),
            _ => None,
        }
    }

    /// Evaluates every pending lazy operation that reads this array's buffer. Must run before
    /// the buffer is overwritten, so those operations keep seeing the data they were recorded
    /// against.
    pub(crate) fn flush_readers(&self) {
        if let Ok(storage) = self.storage() {
            storage.readers.flush();
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
        self.ctx.read(buffer, bytes)
    }

    /// Records the array's pending computation, submits the batch and waits until the GPU has
    /// executed it, without copying anything back. Returns the array's error if it is poisoned.
    pub fn eval(&self) -> Result<()> {
        self.ready().map_err(Failure::into_error)?;
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
        // Lazy operations recorded earlier still read the old contents.
        self.flush_readers();
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
        if let Some(error) = self.failed() {
            return Err(Failure::Inherited(Arc::clone(error)));
        }
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
        // A view shares the buffer, so the array has to exist.
        let storage = self.storage()?;
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
