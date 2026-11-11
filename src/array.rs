use std::marker::PhantomData;
use std::panic::Location;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

use crate::context::{default_context, Context};
use crate::gpu::pager::{Pin, Segment};
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

/// The data of an array: one or several segments, each a device buffer for a contiguous element
/// range, shared by the arrays using it. Segment buffers go back to the context's pool when the
/// last array using them is dropped.
///
/// Segment `j` holds elements `j * C .. min(len, (j + 1) * C)` where `C` is
/// [`Storage::segment_len`]; every storage of the same length and element size on a context has
/// the same segmentation.
#[derive(Debug)]
pub(crate) struct Storage {
    id: u64,
    ctx: Context,
    /// Number of elements.
    len: usize,
    /// Bytes per element.
    elem: usize,
    segments: Box<[Arc<Segment>]>,
    /// Kept alive when this storage aliases another one's segments (see [`Array::probe`]).
    owner: Option<Arc<Storage>>,
    /// Set once a segment was handed out for an eager command or readback.
    materialized: AtomicBool,
    /// Lazy operations that still have to read this buffer; they are evaluated before anything
    /// overwrites it (see [`crate::jit`]).
    pub(crate) readers: Readers,
}

static NEXT_STORAGE_ID: AtomicU64 = AtomicU64::new(0);

/// Elements per full segment of arrays of `elem`-byte elements on `ctx`.
fn segment_len(ctx: &Context, elem: usize) -> usize {
    (usize::try_from(ctx.segment_bytes()).unwrap_or(usize::MAX) / elem).max(1)
}

impl Storage {
    /// A storage of `len` elements of `elem` bytes whose segments have no buffers yet; they are
    /// allocated when first pinned, so this cannot fail.
    pub(crate) fn new(ctx: &Context, len: usize, elem: usize) -> Self {
        let c = segment_len(ctx, elem);
        let segments = (0..len.div_ceil(c))
            .map(|j| Segment::empty(ctx, ((len.min((j + 1) * c) - j * c) * elem) as u64))
            .collect();
        Self::with_segments(ctx, len, elem, segments, None)
    }

    /// A storage whose segments are the given buffers, one per segment in order, each holding
    /// its element range already. The segments recycle them when dropped.
    pub(crate) fn adopt(ctx: &Context, len: usize, elem: usize, buffers: Vec<wgpu::Buffer>) -> Self {
        let c = segment_len(ctx, elem);
        let segments = buffers
            .into_iter()
            .enumerate()
            .map(|(j, buffer)| Segment::filled(ctx, ((len.min((j + 1) * c) - j * c) * elem) as u64, buffer))
            .collect();
        Self::with_segments(ctx, len, elem, segments, None)
    }

    /// A distinct storage identity backed by `owner`'s segments without recycling them
    /// separately.
    pub(crate) fn alias(owner: &Arc<Storage>) -> Self {
        Self::with_segments(
            &owner.ctx,
            owner.len,
            owner.elem,
            owner.segments.clone(),
            Some(Arc::clone(owner)),
        )
    }

    fn with_segments(
        ctx: &Context,
        len: usize,
        elem: usize,
        segments: Box<[Arc<Segment>]>,
        owner: Option<Arc<Storage>>,
    ) -> Self {
        Storage {
            id: NEXT_STORAGE_ID.fetch_add(1, Ordering::Relaxed),
            ctx: ctx.clone(),
            len,
            elem,
            segments,
            owner,
            materialized: AtomicBool::new(false),
            readers: Readers::default(),
        }
    }

    /// Unique per storage; identifies it as an input of a fused kernel.
    pub(crate) fn id(&self) -> u64 {
        self.id
    }

    /// Number of segments.
    pub(crate) fn segments(&self) -> usize {
        self.segments.len()
    }

    /// Bytes per element.
    pub(crate) fn elem(&self) -> usize {
        self.elem
    }

    /// Bytes of data in the whole storage.
    pub(crate) fn byte_len(&self) -> u64 {
        (self.len * self.elem) as u64
    }

    /// Elements of a full segment, `C`.
    pub(crate) fn segment_len(&self) -> usize {
        segment_len(&self.ctx, self.elem)
    }

    /// The elements segment `j` holds.
    pub(crate) fn range(&self, j: usize) -> std::ops::Range<usize> {
        let c = self.segment_len();
        j * c..self.len.min((j + 1) * c)
    }

    /// Segment `j` for an eager command or readback; marks the storage as materialized.
    pub(crate) fn segment(&self, j: usize) -> &Arc<Segment> {
        self.materialized.store(true, Ordering::Relaxed);
        &self.segments[j]
    }

    /// Pins segment `j` on the device (allocating its buffer if needed); marks the storage as
    /// materialized.
    pub(crate) fn pin(&self, j: usize) -> Result<Pin> {
        self.segment(j).pin()
    }

    /// Pins the only segment of a storage that fits one buffer; operations that cannot run
    /// piecewise use this and fail with [`LumaError::Chunked`] on larger ones.
    pub(crate) fn pin_single(&self) -> Result<Pin> {
        if self.segments.len() > 1 {
            return Err(LumaError::Chunked("this operation needs an array that fits one buffer"));
        }
        self.pin(0)
    }

    pub(crate) fn was_materialized(&self) -> bool {
        self.materialized.load(Ordering::Relaxed)
    }

    /// Whether this storage borrows another one's buffers.
    pub(crate) fn is_alias(&self) -> bool {
        self.owner.is_some()
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
        // The segments get their buffers when first pinned.
        Ok(Array {
            ctx: ctx.clone(),
            shape,
            repr: Repr::Valid(Arc::new(Storage::new(ctx, len, std::mem::size_of::<T>()))),
            _marker: PhantomData,
        })
    }

    /// An array that is the (not yet evaluated) result of `node`.
    pub(crate) fn from_node(ctx: &Context, shape: Shape, node: Arc<Node>) -> Self {
        Array { ctx: ctx.clone(), shape, repr: Repr::Lazy(node), _marker: PhantomData }
    }

    /// Placeholder sharing this array's buffer under a fresh [`Storage`] id, for [`Array::iterate`].
    pub(crate) fn probe(&self) -> std::result::Result<Self, Failure> {
        let storage = self.storage()?;
        Ok(Array {
            ctx: self.ctx.clone(),
            shape: self.shape,
            repr: Repr::Valid(Arc::new(Storage::alias(storage))),
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
        if ctx.zero_copy() {
            let storage = Storage::new(ctx, len, std::mem::size_of::<T>());
            let mut buffers = Vec::with_capacity(storage.segments());
            for j in 0..storage.segments() {
                let filled = ctx
                    .make_room(storage.range(j).len() as u64 * std::mem::size_of::<T>() as u64)
                    .and_then(|()| ctx.alloc_filled(bytemuck::cast_slice(&data[storage.range(j)])));
                match filled {
                    Ok(buffer) => buffers.push(buffer),
                    Err(err) => {
                        for buffer in buffers {
                            ctx.recycle(buffer);
                        }
                        return Err(err);
                    }
                }
            }
            let storage = Storage::adopt(ctx, len, std::mem::size_of::<T>(), buffers);
            return Ok(Array {
                ctx: ctx.clone(),
                shape,
                repr: Repr::Valid(Arc::new(storage)),
                _marker: PhantomData,
            });
        }
        let array = Self::alloc(ctx, shape)?;
        let storage = array.storage().map_err(Failure::into_error)?;
        for j in 0..storage.segments() {
            let pin = storage.pin(j)?;
            ctx.upload(pin.buffer(), bytemuck::cast_slice(&data[storage.range(j)]))?;
        }
        Ok(array)
    }

    /// An array of the given shape filled with zeros, on the default context.
    pub fn zeros(shape: impl Into<Shape>) -> Result<Self> {
        Self::zeros_on(&default_context()?, shape)
    }

    /// [`Array::zeros`] on an explicit context.
    pub fn zeros_on(ctx: &Context, shape: impl Into<Shape>) -> Result<Self> {
        let array = Self::alloc(ctx, shape.into())?;
        let storage = array.storage().map_err(Failure::into_error)?;
        for j in 0..storage.segments() {
            let pin = storage.pin(j)?;
            // Pooled buffers hold stale data, so zeroing is required.
            ctx.clear(pin.buffer(), (storage.range(j).len() * storage.elem()) as u64)?;
        }
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

    /// A pin on the buffer holding the array and the number of bytes of it holding the array,
    /// or the reason the array is not valid. Evaluates a lazy array. The pin keeps the buffer
    /// on the device until the command using it is recorded; arrays split into several buffers
    /// fail with [`LumaError::Chunked`].
    pub(crate) fn ready(&self) -> std::result::Result<(Pin, u64), Failure> {
        Ok((self.storage()?.pin_single()?, self.byte_len()))
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

    /// Whether this array is the only holder of its own (non-aliased) buffer: nothing else can
    /// observe the buffer changing. An evaluated lazy array holds it through its node.
    pub(crate) fn sole_owner(&self) -> bool {
        let storage = match &self.repr {
            Repr::Valid(storage) => storage,
            Repr::Lazy(node) if Arc::strong_count(node) == 1 => match node.evaluated() {
                Some(storage) => storage,
                None => return false,
            },
            _ => return false,
        };
        Arc::strong_count(storage) == 1 && !storage.is_alias()
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
    pub(crate) fn buffer(&self) -> Option<wgpu::Buffer> {
        self.ready().ok().map(|(pin, _)| pin.buffer().clone())
    }

    // -- Synchronization --

    /// Copies the array back to the host. Submits the pending batch and blocks until the GPU has
    /// finished it, so this is where a poisoned array's error (or an error the GPU reported
    /// while running the batch) comes out.
    pub fn to_vec(&self) -> Result<Vec<T>> {
        let storage = self.storage().map_err(Failure::into_error)?;
        self.ctx.read_storage(storage)
    }

    /// Like [`Array::to_vec`], but copies into `out`, which must have exactly [`Array::len`]
    /// elements. A buffer the host already owns needs no fresh pages from the operating system,
    /// which cost about as much as the transfer itself on large arrays: prefer this when
    /// reading results back repeatedly.
    pub fn read_into(&self, out: &mut [T]) -> Result<()> {
        let storage = self.storage().map_err(Failure::into_error)?;
        if out.len() != self.len() {
            return Err(LumaError::LengthMismatch { expected: self.len(), actual: out.len() });
        }
        let bytes: &mut [u8] = bytemuck::cast_slice_mut(out);
        // SAFETY: initialized bytes viewed as `MaybeUninit<u8>`; only initialized bytes are
        // written through the view, so `out` stays valid (`T: Pod`).
        let dst = unsafe {
            std::slice::from_raw_parts_mut(bytes.as_mut_ptr().cast::<std::mem::MaybeUninit<u8>>(), bytes.len())
        };
        self.ctx.read_storage_into(storage, dst)
    }

    /// Records the array's pending computation, submits the batch and waits until the GPU has
    /// executed it, without copying anything back. Returns the array's error if it is poisoned.
    pub fn eval(&self) -> Result<()> {
        self.storage().map_err(Failure::into_error)?;
        self.ctx.sync()
    }

    /// Records the array's pending computation and hands the batch to the GPU, without waiting
    /// for it: like [`Array::eval`], but the GPU runs while the host keeps recording the next
    /// operations. Lazy arrays other than this one (and what it is computed from) stay lazy,
    /// unlike [`Context::flush`]. Returns the array's error if it is poisoned.
    ///
    /// Nothing waits, so an error the GPU raises while running the batch only shows up at the
    /// next call that waits ([`Context::sync`], [`Array::to_vec`], [`Array::eval`], reductions).
    ///
    /// While the GPU is still busy with earlier work, up to 16 kernels are held back instead of
    /// being submitted one by one (each submission costs the host and leaves a gap on the GPU).
    /// They are handed over by the next Luma call that submits: another `submit`, a
    /// [`Context::flush`], anything that waits, or the batch limit. A program that calls `submit`
    /// and then does long work without touching Luma should call [`Context::flush`] to force the
    /// hand-over.
    pub fn submit(&self) -> Result<()> {
        self.storage().map_err(Failure::into_error)?;
        self.ctx.submit()
    }

    /// Overwrites the array's contents with `data`, which must have exactly [`Array::len`]
    /// elements. The upload is recorded in order with the operations before it and does not wait
    /// for the GPU: operations recorded earlier still see the old contents.
    pub fn write(&mut self, data: &[T]) -> Result<()> {
        let storage = self.storage().map_err(Failure::into_error)?;
        if data.len() != self.len() {
            return Err(LumaError::LengthMismatch { expected: self.len(), actual: data.len() });
        }
        // Lazy operations recorded earlier still read the old contents.
        self.flush_readers();
        for j in 0..storage.segments() {
            let pin = storage.pin(j)?;
            self.ctx.upload(pin.buffer(), bytemuck::cast_slice(&data[storage.range(j)]))?;
        }
        Ok(())
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
