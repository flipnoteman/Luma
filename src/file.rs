//! File-backed arrays: a segment gets a *home*, a byte range of a file.
//!
//! [`Array::open`] and [`Array::create`] make an array whose data lives in a file (raw bytes
//! with an optional sidecar, or `.npy`). Nothing is read until a kernel needs a segment; the
//! pager reads the file extent into the device (or the CPU backend's buffer), drops clean
//! segments when it needs room and writes dirty ones back (see `gpu::pager`). The data never
//! has to fit in device or host memory.

use std::any::TypeId;
use std::mem::MaybeUninit;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, LazyLock, Mutex, Weak};

use luma_io::{
    Advice, Dtype, Extent, IoFile, IoTicket, NpyHeader, StagePtr, Sidecar, SidecarState, TileIo, parse_npy, parse_sidecar, read_blocking,
    write_blocking, write_npy_header, write_sidecar,
};

use crate::array::{Array, Failure, Storage};
use crate::complex::C32;
use crate::context::{Context, default_context};
use crate::error::{LumaError, Result};
use crate::gpu::hash::FxMap;
use crate::gpu::lock;
use crate::gpu_type::GpuType;
use crate::jit::{self, ElemTy, Op, OpKind, Src};
use crate::shape::Shape;
use crate::builtin::UnOp;
use luma_core::hal::{HostBlock, HostBytes};

/// The largest `.npy` header read (NumPy's own limit is 1 MiB).
const HEAD_BYTES: u64 = (1 << 20) + 64;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Format {
    Raw,
    Npy,
}

/// How a file array is opened or created: the format, where the data starts, whether it may be
/// written. Built with [`FileOptions::raw`] or [`FileOptions::npy`] and the methods taking and
/// returning `Self`.
///
/// ```no_run
/// use luma::{Array, FileOptions};
/// let a = Array::<f32>::open("field.f32", FileOptions::raw().shape([1024, 512]))?;
/// let b = Array::<f32>::open("field.npy", FileOptions::npy().writable())?;
/// # Ok::<(), luma::LumaError>(())
/// ```
#[derive(Clone, Debug)]
pub struct FileOptions {
    format: Format,
    offset: Option<u64>,
    shape: Option<Shape>,
    writable: bool,
    lock: bool,
    sidecar: bool,
    overwrite: bool,
    recover: bool,
    drop_cache: bool,
    engine: Option<Arc<dyn TileIo>>,
}

impl FileOptions {
    fn new(format: Format) -> Self {
        FileOptions {
            format,
            offset: None,
            shape: None,
            writable: false,
            lock: true,
            sidecar: true,
            overwrite: false,
            recover: false,
            drop_cache: false,
            engine: None,
        }
    }

    /// Raw little-endian elements, first dimension fastest, with an optional sidecar
    /// `<file>.luma` that gives the shape and element type.
    pub fn raw() -> Self {
        Self::new(Format::Raw)
    }

    /// A NumPy `.npy` file (versions 1 to 3, little endian, the element types luma has). A C-order
    /// file of shape `(a, b, c)` is an array of shape `[c, b, a]` (no data is reordered); a
    /// Fortran-order file keeps its dimensions.
    pub fn npy() -> Self {
        Self::new(Format::Npy)
    }

    /// Raw only: the byte offset where the data starts (default: the sidecar's, else 0).
    pub fn offset(mut self, bytes: u64) -> Self {
        self.offset = Some(bytes);
        self
    }

    /// The shape of the array: required to open a raw file that has no sidecar; checked against
    /// the sidecar when there is one.
    pub fn shape(mut self, shape: impl Into<Shape>) -> Self {
        self.shape = Some(shape.into());
        self
    }

    /// Opens the file for reading and writing (`create` always does). Without it every write
    /// to the array fails with [`LumaError::ReadOnly`].
    pub fn writable(mut self) -> Self {
        self.writable = true;
        self
    }

    /// Takes an advisory whole-file lock (shared for a read-only array, exclusive for a writable
    /// one) for as long as the array lives. On by default; a file system without locks
    /// proceeds unlocked with a warning.
    pub fn lock(mut self, on: bool) -> Self {
        self.lock = on;
        self
    }

    /// Raw files: `create` writes the sidecar `<file>.luma`, and `open` reads it when it is
    /// there. On by default. (`bf16` has no sidecar element type, so it never gets one.)
    pub fn sidecar(mut self, on: bool) -> Self {
        self.sidecar = on;
        self
    }

    /// `create`: replace an existing file (otherwise an existing file is an `AlreadyExists` error).
    pub fn overwrite(mut self) -> Self {
        self.overwrite = true;
        self
    }

    /// Opens a file that was being written when its writer stopped (its sidecar says `writing`, or
    /// the marker file `<file>.luma-writing` of an `.npy` is still there).
    pub fn recover(mut self) -> Self {
        self.recover = true;
        self
    }

    /// Tells the page cache to forget a segment once it is dropped clean. Off by default: paging
    /// that goes through the data several times profits from the cache.
    pub fn drop_cache(mut self, on: bool) -> Self {
        self.drop_cache = on;
        self
    }

    /// The I/O engine to read and write with (tests inject failures this way). The default is
    /// the process-wide thread pool of `luma-io`.
    #[doc(hidden)]
    pub fn engine(mut self, engine: Arc<dyn TileIo>) -> Self {
        self.engine = Some(engine);
        self
    }
}

/// Identifies a file independently of its path.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
struct FileKey(u64, u64);

fn file_key(file: &IoFile) -> std::io::Result<FileKey> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        let meta = file.file().metadata()?;
        Ok(FileKey(meta.dev(), meta.ino()))
    }
    #[cfg(not(unix))]
    {
        use std::hash::{Hash, Hasher};
        let canonical = std::fs::canonicalize(file.path())?;
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        canonical.hash(&mut hasher);
        Ok(FileKey(hasher.finish(), 0))
    }
}

/// What turns a file complete when it is first flushed.
#[derive(Debug)]
enum Completion {
    /// The marker file next to an `.npy` that is being written.
    Marker(PathBuf),
    /// The sidecar of a raw file, rewritten with `state = complete`.
    Sidecar(PathBuf, Sidecar),
}

/// An open file shared by the segments of the arrays on it.
#[derive(Debug)]
pub(crate) struct FileStore {
    file: Arc<IoFile>,
    path: Arc<Path>,
    writable: bool,
    engine: Arc<dyn TileIo>,
    ctx: Context,
    /// The bytes of the file the array's data covers.
    data: std::ops::Range<u64>,
    dtype: Option<Dtype>,
    shape: Shape,
    drop_cache: bool,
    completion: Mutex<Option<Completion>>,
    /// The storage of the array opened first, so that an identical re-open shares it.
    storage: Mutex<Weak<Storage>>,
    /// The first error of a write-back since the last successful flush.
    failed: Mutex<Option<Arc<LumaError>>>,
}

impl FileStore {
    /// Whether the file may be written.
    pub(crate) fn writable(&self) -> bool {
        self.writable
    }

    pub(crate) fn drop_cache(&self) -> bool {
        self.drop_cache
    }

    /// Tells the page cache the bytes are not needed.
    pub(crate) fn advise_drop(&self, offset: u64, len: u64) {
        self.file.advise(Extent { offset, len }, Advice::DontNeed);
    }

    /// Records a failed write-back for the next `flush`.
    pub(crate) fn failed_with(&self, error: &LumaError) {
        lock(&self.failed).get_or_insert_with(|| Arc::new(error.clone()));
    }

    /// Reads `dst.len()` bytes from `offset`; every byte of `dst` is written on success.
    pub(crate) fn read(&self, offset: u64, dst: &mut [MaybeUninit<u8>]) -> Result<()> {
        if dst.is_empty() {
            return Ok(());
        }
        let extent = Extent { offset, len: dst.len() as u64 };
        read_blocking(&*self.engine, &self.file, extent, dst).map(|_| ()).map_err(|e| LumaError::io("read", &self.path, Some(offset), e))
    }

    /// Starts reading `file_bytes` bytes from `offset` into a new block of `total` bytes (the rest
    /// of the block is zero once [`PendingRead::finish`] is called). Nothing waits.
    pub(crate) fn start_read(self: &Arc<Self>, offset: u64, file_bytes: u64, total: usize) -> PendingRead {
        let mut block = HostBytes::block(total);
        let filled = (file_bytes as usize).min(total);
        // SAFETY: the block lives (on the heap, at a fixed address) until `PendingRead` is dropped, which waits for the ticket first; nothing else touches the first `filled` bytes meanwhile.
        let stage = unsafe { StagePtr::new(block.as_mut_ptr(), filled) };
        let ticket = self.engine.read(&self.file, Extent { offset, len: filled as u64 }, stage);
        PendingRead { store: Arc::clone(self), offset, ticket: Some(ticket), block: Some(block), filled }
    }

    /// Writes `src` at `offset`.
    pub(crate) fn write(&self, offset: u64, src: &[u8]) -> Result<()> {
        if src.is_empty() {
            return Ok(());
        }
        let extent = Extent { offset, len: src.len() as u64 };
        let result = write_blocking(&*self.engine, &self.file, extent, src)
            .map(|_| ())
            .map_err(|e| LumaError::io("write", &self.path, Some(offset), e));
        if let Err(error) = &result {
            self.failed_with(error);
        }
        result
    }

    /// Syncs the file data and, the first time, turns the file complete.
    fn finish_flush(&self) -> Result<()> {
        self.file.sync_data().map_err(|e| LumaError::io("sync", &self.path, None, e))?;
        let mut completion = lock(&self.completion);
        match completion.take() {
            None => {}
            Some(Completion::Marker(marker)) => {
                if let Err(e) = std::fs::remove_file(&marker)
                    && e.kind() != std::io::ErrorKind::NotFound
                {
                    *completion = Some(Completion::Marker(marker));
                    return Err(LumaError::io("remove marker", &self.path, None, e));
                }
            }
            Some(Completion::Sidecar(path, mut sidecar)) => {
                sidecar.state = SidecarState::Complete;
                if let Err(e) = std::fs::write(&path, write_sidecar(&sidecar)) {
                    sidecar.state = SidecarState::Writing;
                    *completion = Some(Completion::Sidecar(path, sidecar));
                    return Err(LumaError::io("write sidecar", &self.path, None, e));
                }
            }
        }
        Ok(())
    }
}

/// The error of a write into a read-only storage.
pub(crate) fn read_only(storage: &Storage) -> LumaError {
    if storage.is_homed() {
        LumaError::ReadOnly("a file array opened without `writable()`")
    } else {
        LumaError::ReadOnly("the state passed to an iteration step")
    }
}

/// The in-process view of which extents of which files are open.
static REGISTRY: LazyLock<Mutex<FxMap<FileKey, Vec<Weak<FileStore>>>>> = LazyLock::new(|| Mutex::new(FxMap::default()));

/// File systems that gave up on locks (warned once each).
static WARNED_LOCKS: AtomicBool = AtomicBool::new(false);

/// The element type of the file for `T`: `Some(dtype)` for the types a sidecar or `.npy` header
/// can name, `None` for `bf16` (raw only).
fn file_dtype<T: GpuType>() -> Result<Option<Dtype>> {
    let id = TypeId::of::<T>();
    let dtype = if id == TypeId::of::<f32>() {
        Dtype::F32
    } else if id == TypeId::of::<u32>() {
        Dtype::U32
    } else if id == TypeId::of::<i32>() {
        Dtype::I32
    } else if id == TypeId::of::<f64>() {
        Dtype::F64
    } else if id == TypeId::of::<u8>() {
        Dtype::U8
    } else if id == TypeId::of::<i8>() {
        Dtype::I8
    } else if id == TypeId::of::<u16>() {
        Dtype::U16
    } else if id == TypeId::of::<i16>() {
        Dtype::I16
    } else if id == TypeId::of::<half::f16>() {
        Dtype::F16
    } else if id == TypeId::of::<C32>() {
        Dtype::C64
    } else if id == TypeId::of::<half::bf16>() {
        return Ok(None);
    } else {
        return Err(LumaError::Unsupported(format!(
            "file arrays of {} are not supported (their layout in memory is not the layout in a file)",
            std::any::type_name::<T>()
        )));
    };
    Ok(Some(dtype))
}

fn elem_bytes<T: GpuType>() -> u64 {
    std::mem::size_of::<T>() as u64
}

fn sidecar_path(path: &Path) -> PathBuf {
    let mut name = path.as_os_str().to_owned();
    name.push(".luma");
    PathBuf::from(name)
}

fn marker_path(path: &Path) -> PathBuf {
    let mut name = path.as_os_str().to_owned();
    name.push(".luma-writing");
    PathBuf::from(name)
}

fn format_error(path: &Arc<Path>, message: impl Into<String>) -> LumaError {
    LumaError::FileFormat { path: Arc::clone(path), message: message.into() }
}

/// Where the data of a file is and what it holds.
struct Located {
    dtype: Option<Dtype>,
    shape: Shape,
    data_offset: u64,
    completion: Option<Completion>,
}

/// Takes the advisory lock of `file`; a file system without locks proceeds without.
fn lock_file(file: &IoFile, writable: bool, path: &Arc<Path>) -> Result<()> {
    let result = if writable { file.file().try_lock() } else { file.file().try_lock_shared() };
    match result {
        Ok(()) => Ok(()),
        Err(std::fs::TryLockError::WouldBlock) => {
            Err(LumaError::io("lock", path, None, std::io::Error::new(std::io::ErrorKind::WouldBlock, "locked by another process")))
        }
        Err(std::fs::TryLockError::Error(error)) => {
            // ENOLCK: Linux 37, macOS 77.
            let unsupported = error.kind() == std::io::ErrorKind::Unsupported || matches!(error.raw_os_error(), Some(37 | 77));
            if !unsupported {
                return Err(LumaError::io("lock", path, None, error));
            }
            if !WARNED_LOCKS.swap(true, Ordering::Relaxed) {
                log::warn!("luma: the file system of {} does not support file locks; opening it unlocked", path.display());
            }
            Ok(())
        }
    }
}

/// Reads the head of a `.npy` file.
fn read_head(engine: &dyn TileIo, file: &Arc<IoFile>, path: &Arc<Path>) -> Result<Vec<u8>> {
    let len = file.len().min(HEAD_BYTES);
    let mut head = vec![0u8; len as usize];
    // SAFETY: initialized bytes viewed as `MaybeUninit<u8>`; the read only writes initialized bytes.
    let dst = unsafe { std::slice::from_raw_parts_mut(head.as_mut_ptr().cast::<MaybeUninit<u8>>(), head.len()) };
    if !dst.is_empty() {
        read_blocking(engine, file, Extent { offset: 0, len }, dst).map_err(|e| LumaError::io("read", path, Some(0), e))?;
    }
    Ok(head)
}

/// The shape of a `.npy` header in luma's dimensions.
fn npy_shape(header: &NpyHeader, path: &Arc<Path>) -> Result<Shape> {
    // `parse_npy` gives luma's dimensions: the file's as they are for Fortran order, reversed for C order.
    let dims = &header.shape;
    if dims.len() > 4 {
        return Err(format_error(path, format!("the array has {} dimensions; luma arrays have at most 4", dims.len())));
    }
    let mut out = [1usize; 4];
    out[..dims.len()].copy_from_slice(dims);
    Ok(Shape::new(out))
}

/// Checks the dtype of a file against the element type `T`.
fn check_dtype<T: GpuType>(path: &Arc<Path>, file: Dtype) -> Result<()> {
    match file_dtype::<T>()? {
        Some(want) if want == file => Ok(()),
        Some(want) => Err(format_error(path, format!("the file holds {} elements ({}), the array is {} ({})", file.name(), file.npy_descr(), want.name(), std::any::type_name::<T>()))),
        None => Err(format_error(path, format!("the file holds {} elements; {} has no file element type", file.name(), std::any::type_name::<T>()))),
    }
}

impl<T: GpuType> Array<T> {
    /// Opens a file as an array on the [default context](crate::default_context). Nothing is
    /// read yet: each part of the array is read from the file when an operation first needs
    /// it, and given up again (a part that was not written costs nothing to drop) when the
    /// device needs the room, so the file may be larger than device or host memory. Every
    /// operation that works on arrays works on a file array; one that has to see the whole array
    /// at once (a sort, a transposition) fails with [`LumaError::Chunked`] if it does not fit
    /// one buffer.
    ///
    /// The element type `T` must be the file's (`f32`, `u32`, `i32`, `f64`, `u8`, `i8`, `u16`,
    /// `i16`, `f16`, `C32`; `bf16` raw only); `bool` and `Df32` files are not supported. A file
    /// of more than 2³² elements is a *wide* array: elementwise operations and the whole-array
    /// `sum`, `prod`, `min`, `max`, `any` and `all` work on it, nothing that needs 32-bit
    /// indices or counts does (`mean`, `var`, `std`, axis reductions, shifts, views, `count`).
    ///
    /// Without [`FileOptions::writable`] the array is read-only: every write fails with
    /// [`LumaError::ReadOnly`]. A file array is never taken over in place by an expression
    /// whose other operands are dropped; [`Array::copy`] makes an anonymous array.
    ///
    /// The file is locked (shared or exclusive) while any array on it lives; a held lock is a
    /// [`LumaError::Io`] of kind `WouldBlock`. Opening an extent that overlaps an open one is
    /// refused (`ResourceBusy`) unless both are read-only; an identical re-open shares the array.
    pub fn open(path: impl AsRef<Path>, opts: FileOptions) -> Result<Array<T>> {
        Self::open_on(&default_context()?, path, opts)
    }

    /// [`Array::open`] on an explicit context.
    pub fn open_on(ctx: &Context, path: impl AsRef<Path>, opts: FileOptions) -> Result<Array<T>> {
        open_array(ctx, path.as_ref(), opts)
    }

    /// Creates a file of `shape` (zeros) and opens it writable on the default context. Fails
    /// with `AlreadyExists` unless [`FileOptions::overwrite`]. The disk space is reserved up
    /// front, so a full disk fails here. A raw file gets its sidecar (`state = writing`), an
    /// `.npy` file its header and a marker file `<file>.luma-writing`; both turn complete at the
    /// first [`Array::flush`].
    pub fn create(path: impl AsRef<Path>, shape: impl Into<Shape>, opts: FileOptions) -> Result<Array<T>> {
        Self::create_on(&default_context()?, path, shape, opts)
    }

    /// [`Array::create`] on an explicit context.
    pub fn create_on(ctx: &Context, path: impl AsRef<Path>, shape: impl Into<Shape>, opts: FileOptions) -> Result<Array<T>> {
        create_array(ctx, path.as_ref(), shape.into(), opts, true)
    }

    /// Writes this array into `target` (a writable array of the same shape): one kernel per
    /// piece of `target`, so neither array has to fit in memory. Lazy: the writes are recorded and
    /// run with everything else; [`Array::flush`] makes them durable.
    #[track_caller]
    pub fn store(&self, target: &mut Array<T>) -> Result<()> {
        let location = std::panic::Location::caller();
        target.clear_props();
        if !self.context().same_as(target.context()) {
            return Err(LumaError::ContextMismatch);
        }
        if self.shape() != target.shape() {
            return Err(LumaError::ShapeMismatch { lhs: self.shape().dims(), rhs: target.shape().dims() });
        }
        if target.is_read_view() {
            target.check()?;
            return Err(LumaError::InvalidShape("cannot store into a read-only view; use slice_mut, flip_mut or transpose_mut".into()));
        }
        let to = Arc::clone(target.storage().map_err(Failure::into_error)?);
        if to.is_read_only() {
            return Err(read_only(&to));
        }
        let ty = ElemTy::try_of::<T>().ok_or_else(|| LumaError::Unsupported(format!("store of {} arrays", std::any::type_name::<T>())))?;
        let src = self.src().map_err(Failure::into_error)?;
        // What reads the target now keeps the data it was recorded against.
        target.flush_readers();
        // A copy node of its own: it becomes evaluated with the target as its result, which the array that is stored must not.
        let node = jit::create(self.context(), self.len(), Op::new(OpKind::Unary(UnOp::Id), ty), [src, Src::unused(), Src::unused()], "store", location)
            .map_err(Failure::into_error)?;
        jit::evaluate_into(&node, &to).map_err(Failure::into_error)
    }

    /// Writes this array to a new file at `path` and returns when it is on disk: the data goes
    /// to a temporary file next to it, which is renamed onto `path` once it is complete and
    /// synced (an `.npy` file appears atomically; a raw file with a sidecar renames the data,
    /// then the sidecar, which is not atomic). On any failure the temporary file is removed and
    /// `path` is untouched. `opts` says the format; [`FileOptions::overwrite`] replaces an existing `path`.
    pub fn write_to(&self, path: impl AsRef<Path>, opts: FileOptions) -> Result<()> {
        write_array(self, path.as_ref(), opts)
    }

    /// Waits for everything recorded, writes every changed part of a file array back to its
    /// file and syncs the file data. The first flush of a created file turns it complete (sidecar
    /// `state = complete`, or the marker file removed). Does nothing for an array that is not
    /// file-backed. Returns the first error; the parts that could not be written stay changed in
    /// memory and the next flush tries again.
    pub fn flush(&self) -> Result<()> {
        let storage = self.storage().map_err(Failure::into_error)?;
        let Some(file) = storage.file() else { return Ok(()) };
        if !file.writable() {
            return Ok(());
        }
        self.context().sync()?;
        let mut first = None;
        for j in 0..storage.segments() {
            if let Err(error) = storage.segment(j).write_back() {
                first.get_or_insert(error);
            }
        }
        if let Some(error) = first {
            file.failed_with(&error);
            return Err(error);
        }
        file.finish_flush()?;
        lock(&file.failed).take();
        Ok(())
    }

    /// [`Array::flush`], then lets go of the array; the error of a write-back that failed
    /// earlier is reported here too. Dropping an array flushes and only logs failures.
    pub fn close(self) -> Result<()> {
        self.flush()?;
        if let Ok(storage) = self.storage()
            && let Some(file) = storage.file()
            && let Some(error) = lock(&file.failed).take()
        {
            return Err(LumaError::clone(&error));
        }
        Ok(())
    }
}

/// Locates and validates what `opts` and the file say about the array, for opening.
fn locate<T: GpuType>(engine: &dyn TileIo, file: &Arc<IoFile>, path: &Arc<Path>, opts: &FileOptions) -> Result<Located> {
    let want = file_dtype::<T>()?;
    match opts.format {
        Format::Npy => {
            if want.is_none() {
                return Err(LumaError::Unsupported(format!("{} has no .npy element type; use a raw file", std::any::type_name::<T>())));
            }
            if opts.offset.is_some() {
                return Err(LumaError::InvalidShape("FileOptions::offset is for raw files".into()));
            }
            let head = read_head(engine, file, path)?;
            let header = parse_npy(&head).map_err(|e| format_error(path, e.to_string()))?;
            check_dtype::<T>(path, header.dtype)?;
            let shape = npy_shape(&header, path)?;
            if let Some(given) = opts.shape
                && given != shape
            {
                return Err(format_error(path, format!("the file's shape is {shape}, FileOptions::shape says {given}")));
            }
            let marker = marker_path(path);
            let writing = marker.exists();
            if writing && !opts.recover {
                return Err(format_error(path, "the file was being written (its marker file is still there); open it with recover() to use it as it is"));
            }
            let completion = writing.then_some(Completion::Marker(marker));
            Ok(Located { dtype: Some(header.dtype), shape, data_offset: header.data_offset, completion })
        }
        Format::Raw => {
            let side_path = sidecar_path(path);
            let sidecar = if opts.sidecar && want.is_some() && side_path.exists() {
                let text = std::fs::read_to_string(&side_path).map_err(|e| LumaError::io("read sidecar", path, None, e))?;
                Some(parse_sidecar(&text).map_err(|e| format_error(path, e.to_string()))?)
            } else {
                None
            };
            let mut completion = None;
            let (shape, sidecar_offset) = match &sidecar {
                Some(side) => {
                    check_dtype::<T>(path, side.dtype)?;
                    let shape = Shape::new(side.shape);
                    if let Some(given) = opts.shape
                        && given != shape
                    {
                        return Err(format_error(path, format!("the sidecar's shape is {shape}, FileOptions::shape says {given}")));
                    }
                    if side.state == SidecarState::Writing {
                        if !opts.recover {
                            return Err(format_error(path, "the file was being written (its sidecar says `state = writing`); open it with recover() to use it as it is"));
                        }
                        completion = Some(Completion::Sidecar(side_path, side.clone()));
                    }
                    (shape, side.offset)
                }
                None => {
                    let Some(shape) = opts.shape else {
                        return Err(format_error(path, "a raw file without a sidecar needs FileOptions::shape"));
                    };
                    (shape, 0)
                }
            };
            Ok(Located { dtype: want.filter(|_| sidecar.is_some()), shape, data_offset: opts.offset.unwrap_or(sidecar_offset), completion })
        }
    }
}

fn busy(path: &Arc<Path>, what: &str) -> LumaError {
    LumaError::io("open", path, None, std::io::Error::new(std::io::ErrorKind::ResourceBusy, what.to_string()))
}

/// Builds the array for `located` over `file`: registers the extent, takes the lock, makes the storage.
#[allow(clippy::too_many_arguments)]
fn attach<T: GpuType>(
    ctx: &Context,
    file: Arc<IoFile>,
    key: FileKey,
    path: &Arc<Path>,
    located: Located,
    opts: &FileOptions,
    take_lock: bool,
    shape_len: usize,
) -> Result<Array<T>> {
    let bytes = shape_len as u64 * elem_bytes::<T>();
    let data = located.data_offset..located.data_offset + bytes;
    let engine = opts.engine.clone().unwrap_or_else(luma_io::default_engine);
    let writable = file.writable();
    let mut registry = lock(&REGISTRY);
    let stores = registry.entry(key).or_default();
    stores.retain(|weak| weak.strong_count() > 0);
    let mut shared_file = None;
    for store in stores.iter().filter_map(Weak::upgrade) {
        let overlaps = store.data.start < data.end && data.start < store.data.end;
        let identical = store.data == data && store.shape == located.shape && store.writable == writable && store.ctx.same_as(ctx) && store.dtype == located.dtype;
        if overlaps
            && identical
            && let Some(storage) = lock(&store.storage).upgrade()
        {
            return Ok(Array::from_storage(ctx, located.shape, storage));
        }
        if overlaps && (writable || store.writable) && !identical {
            return Err(busy(path, "the extent overlaps an open file array that is not read-only"));
        }
        if overlaps && identical && (writable || store.writable) && !store.ctx.same_as(ctx) {
            return Err(busy(path, "the extent is open for writing on another context"));
        }
        if store.writable || !writable {
            shared_file = Some(Arc::clone(&store.file));
        }
    }
    let file = match shared_file {
        // The same open file description holds the one lock of this process.
        Some(shared) => shared,
        None => {
            if take_lock && opts.lock {
                lock_file(&file, writable, path)?;
            }
            file
        }
    };
    let store = Arc::new(FileStore {
        file,
        path: Arc::clone(path),
        writable,
        engine,
        ctx: ctx.clone(),
        data,
        dtype: located.dtype,
        shape: located.shape,
        drop_cache: opts.drop_cache,
        completion: Mutex::new(located.completion),
        storage: Mutex::new(Weak::new()),
        failed: Mutex::new(None),
    });
    let layout = T::LAYOUT;
    let storage = Arc::new(Storage::homed(ctx, shape_len, layout, &store, store.data.start, elem_bytes::<T>()));
    if shape_len > u32::MAX as usize && storage.segments() < 2 {
        return Err(LumaError::DimensionTooLarge { shape: located.shape.dims() });
    }
    *lock(&store.storage) = Arc::downgrade(&storage);
    registry.entry(key).or_default().push(Arc::downgrade(&store));
    Ok(Array::from_storage(ctx, located.shape, storage))
}

fn open_array<T: GpuType>(ctx: &Context, path: &Path, opts: FileOptions) -> Result<Array<T>> {
    file_dtype::<T>()?;
    let path: Arc<Path> = Arc::from(path);
    let file = Arc::new(IoFile::open(&path, opts.writable).map_err(|e| LumaError::io("open", &path, None, e))?);
    let key = file_key(&file).map_err(|e| LumaError::io("stat", &path, None, e))?;
    let engine = opts.engine.clone().unwrap_or_else(luma_io::default_engine);
    let located = locate::<T>(&*engine, &file, &path, &opts)?;
    let shape_len = located.shape.validate_wide()?;
    let end = located.data_offset.checked_add(shape_len as u64 * elem_bytes::<T>()).ok_or_else(|| format_error(&path, "the region ends beyond the largest file offset"))?;
    let file_len = file.len();
    if file_len < end {
        return Err(LumaError::io(
            "open",
            &path,
            Some(file_len),
            std::io::Error::new(std::io::ErrorKind::UnexpectedEof, format!("the file has {file_len} bytes, the array needs {end}")),
        ));
    }
    if opts.format == Format::Npy && file_len > end {
        return Err(format_error(&path, format!("{} bytes follow the array's data", file_len - end)));
    }
    attach::<T>(ctx, file, key, &path, located, &opts, true, shape_len)
}

/// The directory entry of `path` made durable (POSIX).
fn sync_parent(path: &Path) {
    #[cfg(unix)]
    if let Some(dir) = path.parent() {
        let dir = if dir.as_os_str().is_empty() { Path::new(".") } else { dir };
        if let Ok(dir) = std::fs::File::open(dir) {
            let _ = dir.sync_all();
        }
    }
    #[cfg(not(unix))]
    let _ = path;
}

fn create_array<T: GpuType>(ctx: &Context, path: &Path, shape: Shape, opts: FileOptions, markers: bool) -> Result<Array<T>> {
    let want = file_dtype::<T>()?;
    let shape_len = shape.validate_wide()?;
    let path: Arc<Path> = Arc::from(path);
    let bytes = shape_len as u64 * elem_bytes::<T>();
    let (dtype, header, completion) = match opts.format {
        Format::Npy => {
            let Some(dtype) = want else {
                return Err(LumaError::Unsupported(format!("{} has no .npy element type; use a raw file", std::any::type_name::<T>())));
            };
            if opts.offset.is_some() {
                return Err(LumaError::InvalidShape("FileOptions::offset is for raw files".into()));
            }
            let mut dims = shape.dims().to_vec();
            while dims.len() > 1 && dims.last() == Some(&1) {
                dims.pop();
            }
            // Fortran order: luma's dimensions as they are, so that NumPy sees the same indices.
            let header = write_npy_header(dtype, &dims, true);
            let completion = markers.then(|| Completion::Marker(marker_path(&path)));
            (Some(dtype), header, completion)
        }
        Format::Raw => {
            let completion = match want.filter(|_| opts.sidecar) {
                Some(dtype) => Some(Completion::Sidecar(
                    sidecar_path(&path),
                    Sidecar { version: 1, dtype, shape: shape.dims(), offset: opts.offset.unwrap_or(0), state: SidecarState::Writing },
                )),
                None => None,
            };
            (want.filter(|_| opts.sidecar), Vec::new(), completion)
        }
    };
    let data_offset = if opts.format == Format::Raw { opts.offset.unwrap_or(0) } else { header.len() as u64 };
    let total = data_offset + bytes;
    let engine = opts.engine.clone().unwrap_or_else(luma_io::default_engine);

    let create_error = |e| LumaError::io("create", &path, None, e);
    let file = if opts.overwrite && path.exists() {
        let file = Arc::new(IoFile::open(&path, true).map_err(create_error)?);
        // Nothing else may be using the file before it is emptied.
        lock_file(&file, true, &path)?;
        let key = file_key(&file).map_err(|e| LumaError::io("stat", &path, None, e))?;
        if lock(&REGISTRY).get(&key).is_some_and(|stores| stores.iter().any(|store| store.strong_count() > 0)) {
            return Err(busy(&path, "the file is open as an array"));
        }
        file.set_len(0).map_err(create_error)?;
        file.set_len(total).map_err(create_error)?;
        file
    } else {
        let file = Arc::new(IoFile::create_new(&path, total).map_err(create_error)?);
        if opts.lock {
            lock_file(&file, true, &path)?;
        }
        file
    };
    let built = (|| -> Result<Array<T>> {
        file.preallocate(total).map_err(|e| LumaError::io("allocate", &path, None, e))?;
        if !header.is_empty() {
            write_blocking(&*engine, &file, Extent { offset: 0, len: header.len() as u64 }, &header).map_err(|e| LumaError::io("write", &path, Some(0), e))?;
        }
        match &completion {
            Some(Completion::Sidecar(side_path, sidecar)) => {
                std::fs::write(side_path, write_sidecar(sidecar)).map_err(|e| LumaError::io("write sidecar", &path, None, e))?;
            }
            Some(Completion::Marker(marker)) => {
                std::fs::write(marker, b"").map_err(|e| LumaError::io("write marker", &path, None, e))?;
            }
            None => {}
        }
        let key = file_key(&file).map_err(|e| LumaError::io("stat", &path, None, e))?;
        let located = Located { dtype, shape, data_offset, completion };
        // The lock was taken above, on this very file.
        attach::<T>(ctx, Arc::clone(&file), key, &path, located, &opts, false, shape_len)
    })();
    if built.is_err() {
        // Nothing of a failed create is left behind.
        drop(file);
        let _ = std::fs::remove_file(&*path);
        let _ = std::fs::remove_file(sidecar_path(&path));
        let _ = std::fs::remove_file(marker_path(&path));
    }
    built
}

fn write_array<T: GpuType>(array: &Array<T>, path: &Path, opts: FileOptions) -> Result<()> {
    array.check()?;
    let ctx = array.context().clone();
    let temp = {
        let mut name = path.as_os_str().to_owned();
        name.push(format!(".{}.tmp", std::process::id()));
        PathBuf::from(name)
    };
    if !opts.overwrite && path.exists() {
        return Err(LumaError::io("create", &Arc::from(path), None, std::io::Error::new(std::io::ErrorKind::AlreadyExists, "the file exists")));
    }
    let wants_sidecar = opts.format == Format::Raw && opts.sidecar && file_dtype::<T>()?.is_some();
    let result = (|| -> Result<()> {
        let temp_opts = FileOptions { overwrite: true, ..opts.clone() };
        let mut target = create_array::<T>(&ctx, &temp, array.shape(), temp_opts, false)?;
        array.store(&mut target)?;
        target.flush()?;
        target.close()?;
        std::fs::rename(&temp, path).map_err(|e| LumaError::io("rename", &Arc::from(path), None, e))?;
        if wants_sidecar {
            std::fs::rename(sidecar_path(&temp), sidecar_path(path)).map_err(|e| LumaError::io("rename", &Arc::from(path), None, e))?;
        }
        sync_parent(path);
        Ok(())
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(&temp);
        let _ = std::fs::remove_file(sidecar_path(&temp));
    }
    result
}

/// A read of a file extent into host memory that is under way (see [`FileStore::start_read`]).
#[derive(Debug)]
pub(crate) struct PendingRead {
    store: Arc<FileStore>,
    offset: u64,
    ticket: Option<IoTicket>,
    block: Option<HostBlock>,
    filled: usize,
}

impl PendingRead {
    /// Waits for the read; the bytes past the file's are zero.
    pub(crate) fn finish(mut self) -> Result<HostBytes> {
        let ticket = self.ticket.take().expect("a read is finished once");
        let result = if self.filled == 0 { Ok(0) } else { self.store.engine.wait(&ticket) };
        result.map_err(|e| LumaError::io("read", &self.store.path, Some(self.offset), e))?;
        let block = self.block.take().expect("a read is finished once");
        // SAFETY: the read of the first `filled` bytes completed successfully, and nothing writes the block any more.
        Ok(unsafe { block.finish(self.filled) })
    }
}

impl Drop for PendingRead {
    fn drop(&mut self) {
        // The engine writes into the block until the ticket completes.
        if let Some(ticket) = self.ticket.take() {
            let _ = self.store.engine.wait(&ticket);
        }
    }
}

#[cfg(test)]
mod readahead_tests {
    use super::*;
    use crate::context::ContextOptions;
    use luma_io::{FailPlan, FailingIo, Fault, PoolIo};

    struct Temp(PathBuf);

    impl Drop for Temp {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    fn temp(name: &str, bytes: &[u8]) -> Temp {
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let n = NEXT.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!("luma-readahead-{}-{n}-{name}", std::process::id()));
        std::fs::write(&path, bytes).unwrap();
        Temp(path)
    }

    /// 4 KiB segments, a 16 KiB device budget and 256 KiB of host budget: the staging budget is 32 KiB (8 segments' worth, capped at the depth).
    fn ctx(depth: usize) -> Context {
        let ctx = Context::with_options(ContextOptions::from_env().calibrate(false).chunk_bytes(1 << 12).memory_budget(1 << 14).host_budget(Some(1 << 18))).unwrap();
        ctx.pager().set_readahead_depth(depth);
        ctx
    }

    fn values(len: usize) -> Vec<f32> {
        (0..len).map(|i| ((i * 7 + 3) % 1021) as f32).collect()
    }

    fn bytes(data: &[f32]) -> Vec<u8> {
        data.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    #[test]
    fn results_are_bit_identical_with_and_without_read_ahead() {
        let len = 1 << 16;
        let data = values(len);
        let file = temp("identical", &bytes(&data));
        let mut sums = Vec::new();
        for depth in [0, 4] {
            let ctx = ctx(depth);
            let x = Array::<f32>::open_on(&ctx, &file.0, FileOptions::raw().shape(len)).unwrap();
            let sum = x.sum().item().unwrap();
            let chain = ((&x * 2.0f32 + 1.0f32).abs() - 0.5f32).sum().item().unwrap();
            let out = temp("identical-out", &[]);
            std::fs::remove_file(&out.0).unwrap();
            let mut y = Array::<f32>::create_on(&ctx, &out.0, len, FileOptions::raw().sidecar(false)).unwrap();
            (&x * 2.0f32 + 1.0f32).store(&mut y).unwrap();
            y.flush().unwrap();
            drop(y);
            let stored = std::fs::read(&out.0).unwrap();
            sums.push((sum.to_bits(), chain.to_bits(), stored));
            if depth > 0 && !ctx.is_cpu() {
                assert!(ctx.paging_stats().readahead_peak_bytes > 0, "the walk read ahead");
            }
        }
        assert_eq!(sums[0], sums[1]);
    }

    #[test]
    fn read_ahead_stays_within_the_staging_budget() {
        let len = 1 << 16;
        let file = temp("budget", &bytes(&values(len)));
        let ctx = ctx(8);
        let x = Array::<f32>::open_on(&ctx, &file.0, FileOptions::raw().shape(len)).unwrap();
        x.sum().item().unwrap();
        assert!(ctx.paging_stats().readahead_peak_bytes <= (1 << 18) / 8, "{:?}", ctx.paging_stats());
    }

    #[test]
    fn a_short_last_segment_reads_with_a_zero_tail() {
        // 4099 bytes: the second segment of a u8 array holds 3 bytes of the file in a word.
        let len = 4099;
        let data: Vec<u8> = (0..len).map(|i| (i % 251) as u8 + 1).collect();
        let file = temp("tail", &data);
        let ctx = ctx(4);
        let x = Array::<u8>::open_on(&ctx, &file.0, FileOptions::raw().shape(len)).unwrap();
        assert_eq!(x.to_vec().unwrap(), data);
        let x = Array::<u8>::open_on(&ctx, &file.0, FileOptions::raw().shape(len)).unwrap();
        assert_eq!((&x + 1u8).to_vec().unwrap(), data.iter().map(|v| v + 1).collect::<Vec<u8>>());
    }

    #[test]
    fn a_failed_read_ahead_surfaces_at_the_pin_that_needs_it() {
        let len = 1 << 14;
        let data = values(len);
        let file = temp("failing", &bytes(&data));
        let ctx = ctx(4);
        // The 4th read (segment 3 when reads go in order) fails.
        let engine: Arc<dyn TileIo> = Arc::new(FailingIo::new(PoolIo::new(2), FailPlan { read: Some((3, Fault::Fail(std::io::ErrorKind::Other))), ..FailPlan::default() }));
        let x = Array::<f32>::open_on(&ctx, &file.0, FileOptions::raw().shape(len).engine(engine)).unwrap();
        let error = (&x * 2.0f32).sum().item().unwrap_err();
        assert!(matches!(error.root(), LumaError::Io { op: "read", .. }), "{error}");
    }
}
