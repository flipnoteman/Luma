use std::marker::PhantomData;
use std::panic::Location;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

use crate::context::{default_context, Context};
use crate::gpu::pager::{Pin, Segment};
use crate::error::{LumaError, Result};
use crate::gpu_type::{ElemLayout, GpuType};
use crate::builtin::UnOp;
use crate::jit::{ElemTy, Held, Node, Op, OpKind, Readers, Src};
use crate::reduce::Reduced;
use crate::shape::Shape;
use crate::view::ViewOf;

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
/// the same segmentation (a bit-packed `bool` counts as a 4-byte element, and `C` is a multiple
/// of 32, so a segment starts on a word).
#[derive(Debug)]
pub(crate) struct Storage {
    id: u64,
    ctx: Context,
    /// Number of elements.
    len: usize,
    /// How the elements sit in the segments' buffers.
    layout: ElemLayout,
    segments: Box<[Arc<Segment>]>,
    /// Kept alive when this storage aliases another one's segments (see [`Array::probe`]).
    owner: Option<Arc<Storage>>,
    /// Whether the one segment was made a window of another storage (see
    /// [`Storage::window_of`]); it stops being one when something pins it.
    windowed: AtomicBool,
    /// Set once a segment was handed out for an eager command or readback.
    materialized: AtomicBool,
    /// Lazy operations that still have to read this buffer; they are evaluated before anything
    /// overwrites it (see [`crate::jit`]).
    pub(crate) readers: Readers,
    /// Id of the lazy node this is the result of, `u64::MAX` if none; finds the state of a loop
    /// whose node was dropped (see `jit::looped`).
    producer: AtomicU64,
}

static NEXT_STORAGE_ID: AtomicU64 = AtomicU64::new(0);

/// Elements per full segment of arrays of `layout` on `ctx`.
fn segment_len(ctx: &Context, layout: ElemLayout) -> usize {
    (usize::try_from(ctx.segment_bytes()).unwrap_or(usize::MAX) / layout.stride()).max(1)
}

impl Storage {
    /// A storage of `len` elements of `layout` whose segments have no buffers yet; they are
    /// allocated when first pinned, so this cannot fail.
    pub(crate) fn new(ctx: &Context, len: usize, layout: ElemLayout) -> Self {
        let c = segment_len(ctx, layout);
        // Packed segments start on a word: the chunk size is a power of two of at least 1024 bytes.
        debug_assert!(!layout.is_packed() || c.is_multiple_of(32));
        let segments = (0..len.div_ceil(c))
            .map(|j| Segment::empty(ctx, layout.bytes(len.min((j + 1) * c) - j * c)))
            .collect();
        Self::with_segments(ctx, len, layout, segments, None)
    }

    /// A storage of `len` elements that shows the elements `offset..offset + len` of `pack` (a
    /// storage of one segment, not bit-packed) without a buffer of its own. Kernels that
    /// build their program through [`crate::jit`] read the range through the pack's one
    /// binding; whatever else needs the buffer (a readback, a write, a custom kernel) gets a
    /// copy of the range first, which is the storage's buffer from then on.
    pub(crate) fn window_of(pack: &Arc<Storage>, offset: usize, len: usize) -> Self {
        debug_assert!(pack.segments.len() == 1 && !pack.layout.is_packed() && offset + len <= pack.len);
        let layout = pack.layout;
        let segment = Segment::windowed(&pack.ctx, layout.bytes(len), Arc::clone(pack), layout.bytes(offset));
        let storage = Self::with_segments(&pack.ctx, len, layout, Box::new([segment]), None);
        storage.windowed.store(true, Ordering::Relaxed);
        storage
    }

    /// Whether [`Storage::adopt`] can make this storage a window: one segment of plain elements
    /// that is a window or has a device buffer no command being recorded holds, and no other
    /// storage's.
    pub(crate) fn can_adopt(&self) -> bool {
        self.segments.len() == 1 && !self.layout.is_packed() && self.owner.is_none() && self.segments[0].can_adopt()
    }

    /// Gives a window a buffer of its own, holding a copy of its range (see
    /// [`Storage::window_of`]); nothing for any other storage.
    pub(crate) fn detach(&self) -> Result<()> {
        if self.is_window() {
            self.pin_quiet(0)?;
        }
        Ok(())
    }

    /// Makes this storage the elements `offset..offset + len` of `pack`, which hold its data
    /// (see [`Storage::window_of`]), and gives its buffer back. `false` if it cannot (see
    /// [`Storage::can_adopt`]).
    pub(crate) fn adopt(&self, pack: &Arc<Storage>, offset: usize) -> bool {
        debug_assert!(pack.segments.len() == 1 && offset + self.len <= pack.len);
        if !self.can_adopt() || !self.segments[0].adopt(Arc::clone(pack), pack.layout.bytes(offset)) {
            return false;
        }
        self.windowed.store(true, Ordering::Relaxed);
        true
    }

    /// The pack and the first element of this storage in it, if it still is a window (see
    /// [`Storage::window_of`]).
    pub(crate) fn window(&self) -> Option<(Arc<Storage>, usize)> {
        if !self.windowed.load(Ordering::Relaxed) {
            return None;
        }
        let (pack, offset) = self.segments.first()?.window()?;
        let at = usize::try_from(offset / pack.layout.stride() as u64).ok()?;
        Some((pack, at))
    }

    /// Whether the storage has not got a buffer of its own: it is a range of another one's.
    pub(crate) fn is_window(&self) -> bool {
        self.windowed.load(Ordering::Relaxed) && self.segments.first().is_some_and(|segment| segment.window().is_some())
    }

    /// The id of the buffer a kernel binds to read this storage: its own, or the pack's for a
    /// window, which the arrays of one pack share.
    pub(crate) fn binding_id(&self) -> u64 {
        match self.window() {
            Some((pack, _)) => pack.id,
            None => self.id,
        }
    }

    /// A distinct storage identity backed by `owner`'s segments without recycling them
    /// separately.
    pub(crate) fn alias(owner: &Arc<Storage>) -> Self {
        Self::with_segments(
            &owner.ctx,
            owner.len,
            owner.layout,
            owner.segments.clone(),
            Some(Arc::clone(owner)),
        )
    }

    fn with_segments(
        ctx: &Context,
        len: usize,
        layout: ElemLayout,
        segments: Box<[Arc<Segment>]>,
        owner: Option<Arc<Storage>>,
    ) -> Self {
        Storage {
            id: NEXT_STORAGE_ID.fetch_add(1, Ordering::Relaxed),
            ctx: ctx.clone(),
            len,
            layout,
            segments,
            owner,
            windowed: AtomicBool::new(false),
            materialized: AtomicBool::new(false),
            readers: Readers::default(),
            producer: AtomicU64::new(u64::MAX),
        }
    }

    /// Unique per storage; identifies it as an input of a fused kernel.
    pub(crate) fn id(&self) -> u64 {
        self.id
    }

    pub(crate) fn producer(&self) -> u64 {
        self.producer.load(Ordering::Relaxed)
    }

    pub(crate) fn set_producer(&self, node: u64) {
        self.producer.store(node, Ordering::Relaxed);
    }

    /// Number of segments.
    pub(crate) fn segments(&self) -> usize {
        self.segments.len()
    }

    /// Number of elements.
    pub(crate) fn len(&self) -> usize {
        self.len
    }

    /// How the elements sit in the buffers.
    pub(crate) fn layout(&self) -> ElemLayout {
        self.layout
    }

    /// Bytes of data in the whole storage.
    pub(crate) fn byte_len(&self) -> u64 {
        self.layout.bytes(self.len)
    }

    /// Bytes of data in segment `j`.
    pub(crate) fn segment_byte_len(&self, j: usize) -> u64 {
        self.layout.bytes(self.range(j).len())
    }

    /// Elements of a full segment, `C`.
    pub(crate) fn segment_len(&self) -> usize {
        segment_len(&self.ctx, self.layout)
    }

    /// The elements segment `j` holds.
    pub(crate) fn range(&self, j: usize) -> std::ops::Range<usize> {
        let c = self.segment_len();
        j * c..self.len.min((j + 1) * c)
    }

    /// The words (`GpuType::Repr`) of a host copy of the whole array that segment `j` holds.
    pub(crate) fn words(&self, j: usize) -> std::ops::Range<usize> {
        self.layout.words(self.range(j))
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

    /// [`Storage::pin`] without marking the storage as materialized: the tracers of `iterate` take
    /// a materialized probe for one that was used eagerly.
    pub(crate) fn pin_quiet(&self, j: usize) -> Result<Pin> {
        self.segments[j].pin()
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

    /// Calls `f` with the bytes of the storage when they are host-readable now (one segment, held
    /// in host memory or in an idle directly accessible buffer).
    pub(crate) fn with_host_bytes<R>(&self, f: impl FnOnce(&[u8]) -> R) -> Option<R> {
        if self.segments.len() != 1 || self.is_window() {
            return None;
        }
        self.segments[0].with_host_bytes(f)
    }

    /// [`Storage::with_host_bytes`] with write access; pending readers are evaluated first and an
    /// alias or window is never written.
    pub(crate) fn with_host_bytes_mut<R>(&self, f: impl FnOnce(&mut [u8]) -> R) -> Option<R> {
        if self.segments.len() != 1 || self.is_window() || self.is_alias() {
            return None;
        }
        self.readers.flush();
        self.segments[0].with_host_bytes_mut(f)
    }

    /// A storage of `len` elements of `layout` in a mappable buffer that `f` fills in place. Only
    /// with [`Context::zero_copy`], which is all [`Context::can_run_on_host`] allows.
    pub(crate) fn alloc_host(ctx: &Context, len: usize, layout: ElemLayout, f: impl FnOnce(&mut [u8])) -> Result<Arc<Storage>> {
        let bytes = layout.bytes(len);
        ctx.make_room(bytes)?;
        let buffer = ctx.alloc_host_buffer(bytes)?;
        let ptr = ctx.host_ptr(&buffer, bytes).expect("a fresh mappable buffer has a host pointer");
        // SAFETY: `buffer` holds at least `bytes` bytes and no GPU command touches it.
        f(unsafe { std::slice::from_raw_parts_mut(ptr, bytes as usize) });
        let segment = Segment::filled(ctx, bytes, buffer);
        Ok(Arc::new(Storage::with_segments(ctx, len, layout, Box::new([segment]), None)))
    }
}

#[derive(Debug)]
enum Repr {
    Valid(Arc<Storage>),
    /// An elementwise expression that has not run yet; evaluated (once) when its buffer is
    /// needed. See [`crate::jit`].
    Lazy(Held),
    /// An operation producing this array failed; see [`Array::error`].
    Poisoned(Arc<LumaError>),
    /// A reduction of a pending expression: recorded into a buffer when the GPU needs it, read
    /// from the expression when the host does. See [`crate::reduce`].
    Reduced(Arc<Reduced>),
}

/// GPU-resident array of up to four dimensions.
///
/// Arrays are created from host data ([`Array::from_slice`], [`Array::zeros`], ...) on the
/// [default context](crate::default_context) or, with the `_on` variants, on an explicit
/// [`Context`]. Math is written like on the host: operators (`&a + &b * 2.0`), methods
/// (`a.sqrt()`) and reductions (`a.sum()`). Nothing waits for the GPU until a result is needed:
/// [`Array::to_vec`] and [`Array::item`] are the synchronization points.
///
/// # Lazy arrays and fusion
///
/// Elementwise operators and math functions do not run immediately: the result is a *lazy*
/// array holding the expression, and a chain such as `(&x * 2.0 + &y).sqrt()` becomes a single
/// generated kernel when the data is first needed (reading it back, `eval`, `write`, a shift, a
/// convolution, a custom kernel, [`Context::sync`]). Reductions fuse the expression into their
/// first pass and are lazy themselves: `x.sum()` is a one-element array, so `&x / &x.sum()`
/// normalizes on the GPU without reading anything back, and [`Array::item`] gives the host value.
/// A lazy array behaves as the value its expression had when it was written: later
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
/// [`Array::reshape`] and [`Clone::clone`] share the buffer (or, for an array that has not run
/// yet, the pending expression: it is evaluated once, whichever clone needs it first). A clone is
/// a cheap handle, not a copy: in-place operations (`+=`), [`Array::write`],
/// [`Array::assign`] and kernels taking the array as `&mut` therefore change every array sharing
/// the data, clones and reshapes alike. Lazy expressions recorded before the write keep the
/// values they were recorded against. Use [`Array::copy`] for an independent copy.
///
/// Dropping an array returns its buffer to the context's pool for reuse.
#[derive(Debug)]
pub struct Array<T: GpuType> {
    ctx: Context,
    shape: Shape,
    repr: Repr,
    /// Set on views that write through to the array they read (see [`Array::slice`]).
    view: Option<Arc<ViewOf>>,
    /// Set on slices, flips and transposes (and reshapes of them) taken through `&self`: they
    /// read the array they came from but reject every write (see [`Array::slice_mut`]).
    read_view: bool,
    _marker: PhantomData<T>,
}

impl<T: GpuType> Clone for Array<T> {
    /// Another handle on the same data: the buffer, or the pending expression (counted as a
    /// second handle, so that nothing treats the expression as private to one array). See
    /// "Sharing" in the type's documentation.
    fn clone(&self) -> Self {
        let repr = match &self.repr {
            Repr::Valid(storage) => Repr::Valid(Arc::clone(storage)),
            Repr::Lazy(node) => Repr::Lazy(node.share()),
            Repr::Poisoned(error) => Repr::Poisoned(Arc::clone(error)),
            Repr::Reduced(reduced) => Repr::Reduced(Arc::clone(reduced)),
        };
        Array { ctx: self.ctx.clone(), shape: self.shape, repr, view: self.view.clone(), read_view: self.read_view, _marker: PhantomData }
    }
}

impl<T: GpuType> Array<T> {
    // -- Construction --

    /// Allocates a valid array of `shape` whose contents are unspecified.
    pub(crate) fn alloc(ctx: &Context, shape: Shape) -> Result<Self> {
        let len = shape.validate()?;
        Ok(Self::new(ctx, shape, Repr::Valid(Arc::new(Storage::new(ctx, len, T::LAYOUT)))))
    }

    fn new(ctx: &Context, shape: Shape, repr: Repr) -> Self {
        Array { ctx: ctx.clone(), shape, repr, view: None, read_view: false, _marker: PhantomData }
    }

    pub(crate) fn from_storage(ctx: &Context, shape: Shape, storage: Arc<Storage>) -> Self {
        Self::new(ctx, shape, Repr::Valid(storage))
    }

    /// An array that is the (not yet evaluated) result of `node`.
    pub(crate) fn from_node(ctx: &Context, shape: Shape, node: Arc<Node>) -> Self {
        Self::new(ctx, shape, Repr::Lazy(Held::new(node)))
    }

    /// The (not yet recorded) reduction `reduced` as an array of `shape`.
    pub(crate) fn from_reduced(ctx: &Context, shape: Shape, reduced: Arc<Reduced>) -> Self {
        Self::new(ctx, shape, Repr::Reduced(reduced))
    }

    /// Placeholder sharing this array's buffer under a fresh [`Storage`] id, for [`Array::iterate`].
    pub(crate) fn probe(&self) -> std::result::Result<Self, Failure> {
        let storage = self.storage()?;
        Ok(Self::new(&self.ctx, self.shape, Repr::Valid(Arc::new(Storage::alias(storage)))))
    }

    /// A placeholder with a new buffer of unspecified contents, for tracing a step that is
    /// recorded and replayed (see `jit::replay`). Fails like [`Array::probe`] on a poisoned array.
    pub(crate) fn fresh_probe(&self) -> std::result::Result<Self, Failure> {
        self.storage()?;
        Ok(Array::alloc(&self.ctx, self.shape)?)
    }

    pub(crate) fn poisoned(ctx: &Context, shape: Shape, error: Arc<LumaError>) -> Self {
        Self::new(ctx, shape, Repr::Poisoned(error))
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
            let layout = T::LAYOUT;
            let words = T::to_repr(data);
            let c = segment_len(ctx, layout);
            // Registered as they are filled, so dropping them on failure recycles them and
            // making room for the next one sees them.
            let mut segments = Vec::with_capacity(len.div_ceil(c));
            for start in (0..len).step_by(c) {
                let end = len.min(start + c);
                let bytes = layout.bytes(end - start);
                ctx.make_room(bytes)?;
                let buffer = ctx.alloc_filled(bytemuck::cast_slice(&words[layout.words(start..end)]))?;
                segments.push(Segment::filled(ctx, bytes, buffer));
            }
            let storage = Storage::with_segments(ctx, len, layout, segments.into_boxed_slice(), None);
            return Ok(Self::new(ctx, shape, Repr::Valid(Arc::new(storage))));
        }
        let array = Self::alloc(ctx, shape)?;
        let storage = array.storage().map_err(Failure::into_error)?;
        let words = T::to_repr(data);
        for j in 0..storage.segments() {
            let pin = storage.pin(j)?;
            ctx.upload(pin.buffer(), bytemuck::cast_slice(&words[storage.words(j)]))?;
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
            ctx.clear(pin.buffer(), storage.segment_byte_len(j))?;
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
            Repr::Lazy(held) => held.latest().failure(),
            Repr::Poisoned(error) => Some(error),
            Repr::Reduced(reduced) => reduced.failure(),
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

    /// The array's storage if it is already valid; never evaluates a lazy array.
    pub(crate) fn storage_if_valid(&self) -> Option<&Arc<Storage>> {
        match &self.repr {
            Repr::Valid(storage) => Some(storage),
            _ => None,
        }
    }

    /// The array's storage, evaluating the array first if it is still lazy (this records the
    /// work; it does not wait for the GPU). Fails with the reason the array is not valid.
    pub(crate) fn storage(&self) -> std::result::Result<&Arc<Storage>, Failure> {
        match &self.repr {
            Repr::Valid(storage) => Ok(storage),
            Repr::Lazy(held) => {
                held.latest().force().as_ref().map_err(|error| Failure::Inherited(Arc::clone(error)))
            }
            Repr::Poisoned(error) => Err(Failure::Inherited(Arc::clone(error))),
            Repr::Reduced(reduced) => {
                reduced.force(&self.ctx).as_ref().map_err(|error| Failure::Inherited(Arc::clone(error)))
            }
        }
    }

    /// [`Array::storage`] for a shift, view or gather that decided to read this array evaluated:
    /// a pending expression is computed together with the live arrays that share work with it
    /// (the other fields of a record, say) in one multi-output kernel instead of alone.
    pub(crate) fn storage_alongside(&self) -> std::result::Result<&Arc<Storage>, Failure> {
        if let Some(node) = self.pending() {
            self.ctx.jit().evaluate_alongside(Arc::clone(node));
        }
        self.storage()
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
            Repr::Lazy(held) => held.latest().as_src(),
            Repr::Poisoned(error) => Err(Failure::Inherited(Arc::clone(error))),
            Repr::Reduced(reduced) => match reduced.result_if_recorded() {
                Some(result) => result.as_ref().map(|storage| Src::Buffer(Arc::clone(storage))).map_err(|error| Failure::Inherited(Arc::clone(error))),
                // Not recorded yet: an operation on it is built without recording it.
                None => Ok(Src::Node(reduced.leaf(&self.ctx, self.len()))),
            },
        }
    }

    /// This array marked as writing through to what `view` reads (see [`ViewMut`]).
    pub(crate) fn with_view(mut self, view: ViewOf) -> Self {
        self.view = Some(Arc::new(view));
        self
    }

    /// Marks this array as a read-only view (rejects every write).
    pub(crate) fn into_read_view(mut self) -> Self {
        self.read_view = true;
        self
    }

    /// Whether this array is a slice, flip or transpose taken through `&self` (or a reshape of
    /// one): it reads the array it came from but rejects every write.
    pub(crate) fn is_read_view(&self) -> bool {
        self.read_view
    }

    /// This array without the link to what it is a view of: an ordinary array holding the same data.
    pub(crate) fn without_view(mut self) -> Self {
        self.view = None;
        self.read_view = false;
        self
    }

    /// What this array is a view of, if it is an injective view of another array's data.
    pub(crate) fn view(&self) -> Option<&ViewOf> {
        self.view.as_deref()
    }

    /// The expression behind this array if it has not been evaluated yet.
    pub(crate) fn pending(&self) -> Option<&Arc<Node>> {
        match &self.repr {
            Repr::Lazy(held) => Some(held.latest()).filter(|node| node.is_pending()),
            _ => None,
        }
    }

    /// Whether this array is the placeholder `probe` of a traced step handed back unchanged: its
    /// buffer, or a copy of it that has not run yet (`&x * 1.0`), which there is no need to run.
    pub(crate) fn passes_through(&self, probe: &Arc<Storage>) -> bool {
        if self.view.is_some() || self.read_view {
            return false;
        }
        match self.pending() {
            Some(node) => node.is_copy_of(probe),
            None => matches!(&self.repr, Repr::Valid(storage) if Arc::ptr_eq(storage, probe)),
        }
    }

    /// The reduction this array is (recorded into a buffer or not), if it is one.
    pub(crate) fn reduced(&self) -> Option<&Arc<Reduced>> {
        match &self.repr {
            Repr::Reduced(reduced) => Some(reduced),
            _ => None,
        }
    }

    /// The claim this array has on its expression, if it is lazy.
    pub(crate) fn held(&self) -> Option<&Held> {
        match &self.repr {
            Repr::Lazy(held) => Some(held),
            _ => None,
        }
    }

    /// Whether another array holds the pending expression this one is (a clone or a view that
    /// reads its data back): rebinding this array to a new expression would not reach it.
    pub(crate) fn expression_shared(&self) -> bool {
        matches!(&self.repr, Repr::Lazy(node) if node.is_shared())
    }

    /// Whether this array is the only holder of its own (non-aliased) buffer: nothing else can
    /// observe the buffer changing. An evaluated lazy array holds it through its node; the
    /// arrays that denote that node through a replacement (see [`crate::jit::supersede`]) are
    /// counted as claims on the node they were created holding, not by the reference count of
    /// the node that took over.
    pub(crate) fn sole_owner(&self) -> bool {
        let storage = match &self.repr {
            Repr::Valid(storage) => storage,
            Repr::Lazy(held) if Arc::strong_count(held.latest()) == 1 && !held.is_shared() => match held.latest().evaluated() {
                Some(storage) => storage,
                None => return false,
            },
            _ => return false,
        };
        Arc::strong_count(storage) == 1 && !storage.is_alias() && !storage.is_window()
    }

    /// Evaluates every pending lazy operation that reads this array's buffer. Must run before
    /// the buffer is overwritten, so those operations keep seeing the data they were recorded
    /// against.
    pub(crate) fn flush_readers(&self) {
        if let Ok(storage) = self.storage() {
            storage.readers.flush();
            self.ctx.jit().note_write();
        }
    }

    /// Size of the array's data in bytes (not the size of the possibly larger buffer).
    pub(crate) fn byte_len(&self) -> u64 {
        T::LAYOUT.bytes(self.shape.len())
    }

    #[cfg(test)]
    pub(crate) fn buffer(&self) -> Option<luma_core::Buffer> {
        self.ready().ok().map(|(pin, _)| pin.buffer().clone())
    }

    // -- Synchronization --

    /// Copies the array back to the host. Submits the pending batch and blocks until the GPU has
    /// finished it, so this is where a poisoned array's error (or an error the GPU reported
    /// while running the batch) comes out.
    pub fn to_vec(&self) -> Result<Vec<T>> {
        if let Repr::Reduced(reduced) = &self.repr {
            if reduced.is_unforced() {
                return reduced.read::<T>(&self.ctx);
            }
            if let Some(values) = reduced.packed_values::<T>(&self.ctx) {
                return values;
            }
        }
        let storage = self.storage().map_err(Failure::into_error)?;
        Ok(T::from_repr(self.ctx.read_storage::<T::Repr>(storage)?, self.len()))
    }

    /// Like [`Array::to_vec`], but copies into `out`, which must have exactly [`Array::len`]
    /// elements. A buffer the host already owns needs no fresh pages from the operating system,
    /// which cost about as much as the transfer itself on large arrays: prefer this when
    /// reading results back repeatedly.
    pub fn read_into(&self, out: &mut [T]) -> Result<()> {
        if let Repr::Reduced(reduced) = &self.repr {
            let values = if reduced.is_unforced() { Some(reduced.read::<T>(&self.ctx)) } else { reduced.packed_values::<T>(&self.ctx) };
            if let Some(values) = values {
                if out.len() != self.len() {
                    return Err(LumaError::LengthMismatch { expected: self.len(), actual: out.len() });
                }
                out.copy_from_slice(&values?);
                return Ok(());
            }
        }
        let storage = self.storage().map_err(Failure::into_error)?;
        if out.len() != self.len() {
            return Err(LumaError::LengthMismatch { expected: self.len(), actual: out.len() });
        }
        let Some(words) = T::repr_mut(out) else {
            // A representation of its own (`bool`): read the words, then convert them.
            out.copy_from_slice(&T::from_repr(self.ctx.read_storage::<T::Repr>(storage)?, out.len()));
            return Ok(());
        };
        let bytes: &mut [u8] = bytemuck::cast_slice_mut(words);
        // SAFETY: initialized bytes viewed as `MaybeUninit<u8>`; only initialized bytes are
        // written through the view, so `words` stays valid (`T::Repr: Pod`).
        let dst = unsafe {
            std::slice::from_raw_parts_mut(bytes.as_mut_ptr().cast::<std::mem::MaybeUninit<u8>>(), bytes.len())
        };
        self.ctx.read_storage_into(storage, dst)
    }

    /// Records the array's pending computation and hands the batch to the GPU without waiting
    /// for it, like ArrayFire's `eval`: the GPU runs while the host records the next steps.
    /// [`Context::sync`], [`Array::to_vec`] and [`Array::item`] wait. Lazy arrays other than this one
    /// (and what it is computed from) stay lazy, unlike [`Context::flush`]. Returns the array's
    /// error if it is poisoned.
    ///
    /// Nothing waits, so an error the GPU raises while running the batch only shows up at the
    /// next call that waits ([`Context::sync`], [`Array::to_vec`], [`Array::item`]).
    ///
    /// While the GPU is still busy with earlier work, the computation is postponed to the next
    /// `eval` that finds it idle, [`Context::sync`] or [`Context::flush`]; reading the array
    /// computes it at once. Arrays the program drops before then are never computed, and the
    /// steps in between fuse, so a loop that evaluates every step costs about what one evaluating
    /// every few steps does. A program that calls `eval` and then does long work without touching
    /// Luma should call [`Context::flush`] to start the GPU. A failure while computing the array
    /// is returned here if the computation was not postponed, and otherwise by the array's next
    /// use.
    pub fn eval(&self) -> Result<()> {
        crate::eval_all(&[self])
    }

    /// Asks for the array to be computed at the next submission, or returns its error.
    pub(crate) fn request(&self) -> Result<()> {
        match self.pending() {
            Some(node) => {
                self.ctx.jit().request(node);
                Ok(())
            }
            None => self.storage().map(|_| ()).map_err(Failure::into_error),
        }
    }

    /// Overwrites the array's contents with `data`, which must have exactly [`Array::len`]
    /// elements. The upload is recorded in order with the operations before it and does not wait
    /// for the GPU: operations recorded earlier still see the old contents.
    pub fn write(&mut self, data: &[T]) -> Result<()> {
        if self.is_read_view() {
            self.check()?;
            return Err(LumaError::InvalidShape(
                "cannot write into a read-only view; use slice_mut, flip_mut or transpose_mut".into(),
            ));
        }
        let storage = self.storage().map_err(Failure::into_error)?;
        if storage.is_alias() {
            return Err(LumaError::ReadOnly("the state passed to an iteration step"));
        }
        if data.len() != self.len() {
            return Err(LumaError::LengthMismatch { expected: self.len(), actual: data.len() });
        }
        // Lazy operations recorded earlier still read the old contents.
        self.flush_readers();
        let words = T::to_repr(data);
        for j in 0..storage.segments() {
            let pin = storage.pin(j)?;
            self.ctx.upload(pin.buffer(), bytemuck::cast_slice(&words[storage.words(j)]))?;
        }
        Ok(())
    }

    /// An independent array holding the current contents: later writes, in-place operations and
    /// kernels taking either array as `&mut` do not reach the other (unlike [`Clone::clone`] and
    /// [`Array::reshape`], which share the data). The copy of a slice, flip or transpose is an
    /// ordinary array. Lazy for `f32`, `u32`, `f64` and [`Df32`](crate::Df32) elements: the copy
    /// fuses into whatever reads it and moves no data of its own; other element types are copied
    /// on the device at once. A poisoned array copies to a poisoned array with the same error.
    #[track_caller]
    pub fn copy(&self) -> Array<T> {
        self.copy_at(Location::caller())
    }

    /// [`Array::copy`] reporting failures at `location`.
    pub(crate) fn copy_at(&self, location: &'static Location<'static>) -> Array<T> {
        let result = self.run_copy(location);
        Self::resolve(&self.ctx, self.shape, result, "copy", location)
    }

    fn run_copy(&self, location: &'static Location<'static>) -> std::result::Result<Array<T>, Failure> {
        let src = self.src()?;
        if let Some(ty) = Self::scalar_ty() {
            let args = [src, Src::unused(), Src::unused()];
            let node = crate::jit::create(&self.ctx, self.len(), Op::new(OpKind::Unary(UnOp::Id), ty), args, "copy", location)?;
            return Ok(Self::from_node(&self.ctx, self.shape, node));
        }
        let from = self.storage()?;
        let out = Self::alloc(&self.ctx, self.shape)?;
        let to = out.storage()?;
        for j in 0..from.segments() {
            let (source, target) = (from.pin(j)?, to.pin(j)?);
            self.ctx.copy(source.buffer(), 0, target.buffer(), 0, from.segment_byte_len(j))?;
        }
        Ok(out)
    }

    /// The element type of the built-in arithmetic, if `T` is one.
    fn scalar_ty() -> Option<ElemTy> {
        ElemTy::try_of::<T>()
    }

    // -- Views --

    /// The same data with a different shape. Only metadata changes: the result shares the
    /// buffer, no copy is made (see "Sharing" in the type's documentation). The element count
    /// must stay the same; otherwise the result is poisoned.
    ///
    /// The reshape of a lazy view (a slice, flip, transpose, pad or non-circular shift that has
    /// not run) stays lazy: it is another handle on the pending view, which fuses into whatever
    /// reads it, so `x.slice(0, k..k + 1).reshape([ni, nj])` of an `[n, ni, nj]` array reads `x`
    /// inside the kernel that uses it. The result does not write through to the array the view
    /// was taken from, and the reshape of a read-only slice, flip or transpose is read-only too.
    /// Any other array that has not run yet is evaluated first (the data it shares must exist).
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
        // A pending view (an index map of what it reads) is shared like by a clone (evaluated
        // once, by whichever handle needs it first): an expression is a sequence of elements,
        // whatever its shape. Other expressions are evaluated first (below): what reads them
        // through several slices would recompute them.
        if let Repr::Lazy(held) = &self.repr
            && held.latest().is_pending_view()
        {
            return Ok(Array {
                ctx: self.ctx.clone(),
                shape,
                repr: Repr::Lazy(held.share()),
                view: None,
                read_view: self.read_view,
                _marker: PhantomData,
            });
        }
        // A reduction that has not been recorded is the same pending reduction in the new shape: recorded
        // when either is used, together with the others that are pending (see `jit::multireduce`).
        if let Repr::Reduced(reduced) = &self.repr
            && reduced.is_unforced()
        {
            return Ok(Array {
                ctx: self.ctx.clone(),
                shape,
                repr: Repr::Reduced(Arc::clone(reduced)),
                view: None,
                read_view: self.read_view,
                _marker: PhantomData,
            });
        }
        // A view shares the buffer, so the array has to exist.
        let storage = self.storage()?;
        Ok(Array {
            ctx: self.ctx.clone(),
            shape,
            repr: Repr::Valid(Arc::clone(storage)),
            view: None,
            read_view: self.read_view,
            _marker: PhantomData,
        })
    }

    /// Marks this array poisoned (used by in-place operations that failed).
    pub(crate) fn poison(&mut self, error: Arc<LumaError>) {
        self.repr = Repr::Poisoned(error);
    }
}
