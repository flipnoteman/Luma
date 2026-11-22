//! Sparse matrices: [`Sparse`] in CSR, CSC and COO storage, with conversions, products with dense
//! matrices and vectors, and elementwise operations with dense arrays and scalars.
//!
//! A matrix of `rows x cols` is stored as arrays of `u32` indices and `f32` values (one per
//! stored entry) on the device. Dense matrices follow the rest of luma: a dense `rows x cols`
//! matrix is an [`Array`] of shape `[cols, rows]` (axis 0, the fastest, is the column index), so
//! the dense form of a sparse matrix, [`Sparse::to_dense`], has that shape and a dense right-hand
//! side of `c` columns has shape `[c, cols]`.
//!
//! * **CSR** (compressed sparse row): `row_ptr` of `rows + 1` offsets (entry `k` of row `r`
//!   lies in `row_ptr[r]..row_ptr[r + 1]`), the column and the value of every entry in row
//!   order.
//! * **CSC** (compressed sparse column): `col_ptr` of `cols + 1` offsets, the row and the value
//!   of every entry in column order.
//! * **COO** (coordinate): the row, the column and the value of every entry, in any order.
//!
//! Duplicate entries (the same row and column stored twice, which COO allows and conversions
//! keep) add up: they count as one entry holding their sum in [`Sparse::to_dense`], in products
//! and in elementwise operations with dense operands, while [`Sparse::nnz`] counts them
//! separately. Indices must be in range (column below `cols`, row below `rows`); out-of-range
//! indices give unspecified results, never reads or writes outside the arrays.
//!
//! luma has no arrays without elements, so a matrix without stored entries has no index or
//! value arrays: [`Sparse::values`] and the other accessors return `None` for it
//! ([`Sparse::empty`]; [`Sparse::from_dense`] of an all-zero matrix gives one).
//!
//! # Products
//!
//! [`Sparse::matmul`] computes `A x` and [`Sparse::matmul_t`] `A^T x` for a vector or a dense
//! matrix `x`. CSR `A x` and CSC `A^T x` are a gather of `x` at the stored indices, a product with
//! the values and a segmented sum over the row (column) pointer ([`Array::sum_segments`]): two
//! launches, no atomics, entries split evenly over the workgroups whatever the row lengths. The
//! other combinations scatter the products with an atomic add by output index
//! ([`Array::group_sum`]), whose rounding depends on the order in which the atomics land.
//!
//! # Errors
//!
//! As everywhere in luma, a failure poisons the result: a bad constructor argument gives a
//! [`Sparse`] whose [`Sparse::error`] is set and which poisons everything computed from it; a
//! dense result is a poisoned array.

use std::fmt;
use std::ops::{Div, Mul, Neg};
use std::panic::Location;
use std::sync::{Arc, OnceLock};

use crate::array::Array;
use crate::context::{default_context, Context};
use crate::error::{LumaError, Result};
use crate::scatter::{bincount, AtomicElem};
use crate::search::Side;
use crate::shape::Shape;
use crate::view::concat;

/// How the entries of a [`Sparse`] matrix are stored.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SparseFormat {
    /// Compressed sparse row: a row pointer, and the column and value of every entry in row order.
    Csr,
    /// Compressed sparse column: a column pointer, and the row and value of every entry in column
    /// order.
    Csc,
    /// Coordinate list: the row, column and value of every entry, in any order.
    Coo,
}

/// Element types of [`Sparse`]: `f32`.
pub trait SparseElem: AtomicElem {}

impl SparseElem for f32 {}

/// A sparse `rows x cols` matrix of `f32` in CSR, CSC or COO storage. See the [module
/// documentation](self) for the layouts, the conventions and the errors.
///
/// ```no_run
/// use luma::{Array, Sparse, SparseFormat};
///
/// // [[1, 0, 2], [0, 0, 3]] as CSR, times [1, 10, 100].
/// let a = Sparse::from_csr(
///     2,
///     3,
///     &Array::from_slice(&[0u32, 2, 3], 3)?,
///     &Array::from_slice(&[0u32, 2, 2], 3)?,
///     &Array::from_slice(&[1.0f32, 2.0, 3.0], 3)?,
/// );
/// let x = Array::from_slice(&[1.0f32, 10.0, 100.0], 3)?;
/// assert_eq!(a.matmul(&x).to_vec()?, [201.0, 300.0]);
/// assert_eq!(a.to_format(SparseFormat::Coo).nnz(), 3);
/// # Ok::<(), luma::LumaError>(())
/// ```
pub struct Sparse<T: SparseElem = f32> {
    ctx: Context,
    rows: usize,
    cols: usize,
    format: SparseFormat,
    nnz: usize,
    /// CSR row pointer or CSC column pointer.
    ptr: Option<Array<u32>>,
    /// The row of every entry (CSC and COO).
    row: Option<Array<u32>>,
    /// The column of every entry (CSR and COO).
    col: Option<Array<u32>>,
    val: Option<Array<T>>,
    /// The expanded pointer: the row of every entry of a CSR matrix, the column of every entry
    /// of a CSC matrix.
    expanded: OnceLock<Array<u32>>,
    error: Option<Arc<LumaError>>,
}

/// The arrays of a [`Sparse`] matrix as [`Sparse::info`] returns them (ArrayFire's
/// `sparseGetInfo`).
#[derive(Clone, Debug)]
pub struct SparseInfo<T: SparseElem = f32> {
    /// The storage format.
    pub format: SparseFormat,
    /// Number of rows.
    pub rows: usize,
    /// Number of columns.
    pub cols: usize,
    /// Number of stored entries.
    pub nnz: usize,
    /// The values of the entries; `None` if there are none.
    pub values: Option<Array<T>>,
    /// CSR: the row pointer (`rows + 1` offsets); CSC and COO: the row of every entry (`None`
    /// without entries).
    pub row_idx: Option<Array<u32>>,
    /// CSC: the column pointer (`cols + 1` offsets); CSR and COO: the column of every entry
    /// (`None` without entries).
    pub col_idx: Option<Array<u32>>,
}

impl<T: SparseElem> Clone for Sparse<T> {
    fn clone(&self) -> Self {
        let expanded = OnceLock::new();
        if let Some(e) = self.expanded.get() {
            let _ = expanded.set(e.clone());
        }
        Sparse {
            ctx: self.ctx.clone(),
            rows: self.rows,
            cols: self.cols,
            format: self.format,
            nnz: self.nnz,
            ptr: self.ptr.clone(),
            row: self.row.clone(),
            col: self.col.clone(),
            val: self.val.clone(),
            expanded,
            error: self.error.clone(),
        }
    }
}

impl<T: SparseElem> fmt::Debug for Sparse<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Sparse")
            .field("rows", &self.rows)
            .field("cols", &self.cols)
            .field("format", &self.format)
            .field("nnz", &self.nnz)
            .field("error", &self.error)
            .finish()
    }
}

/// The failure `error` of `op` called at `location`, as the error a result is poisoned with.
fn failure(op: &'static str, location: &'static Location<'static>, error: LumaError) -> Arc<LumaError> {
    Arc::new(LumaError::Op { op, location, source: Arc::new(error) })
}

/// `Some(error)` unless `rows` and `cols` are non-zero and fit in `u32` with everything the
/// kernels index (`rows * cols` need not fit: only dense forms need it).
fn check_dims(rows: usize, cols: usize) -> Option<LumaError> {
    if rows == 0 || cols == 0 {
        return Some(LumaError::InvalidShape(format!("a sparse matrix of {rows} x {cols}: dimensions must be non-zero")));
    }
    if u32::try_from(rows).is_err() || u32::try_from(cols).is_err() || u32::try_from(rows + 1).is_err() || u32::try_from(cols + 1).is_err() {
        return Some(LumaError::DimensionTooLarge { shape: [cols, rows, 1, 1] });
    }
    None
}

/// `array` as one-dimensional, evaluated or recorded for good: what the sparse arrays are stored as.
fn flat<T: crate::gpu_type::ScalarType>(array: &Array<T>) -> Array<T> {
    let n = array.len();
    if array.shape().dims() == [n, 1, 1, 1] { array.cache() } else { array.reshape(n).cache() }
}

/// The offsets of `n` groups from the group of every entry (`ids`, not necessarily sorted):
/// `n + 1` offsets, the first 0.
fn pointer(ids: &Array<u32>, n: usize) -> Array<u32> {
    let ends = bincount(ids, n).cumsum(0);
    match Array::<u32>::zeros_on(ids.context(), 1) {
        Ok(zero) => concat(0, [&zero, &ends]).cache(),
        Err(error) => Array::poisoned(ids.context(), Shape::from(n + 1), Arc::new(error)),
    }
}

/// `(pointer, other, values)` of the entries regrouped by `key` (a stable sort, so the entries of a
/// group keep their order), for `n` groups.
fn regroup(key: &Array<u32>, other: &Array<u32>, val: &Array<f32>, n: usize) -> (Array<u32>, Array<u32>, Array<f32>) {
    let perm = key.argsort();
    (pointer(key, n), other.take(&perm).cache(), val.take(&perm).cache())
}

impl Sparse<f32> {
    // -- Construction --

    fn assemble(
        ctx: &Context,
        (rows, cols): (usize, usize),
        format: SparseFormat,
        ptr: Option<Array<u32>>,
        row: Option<Array<u32>>,
        col: Option<Array<u32>>,
        val: Array<f32>,
    ) -> Sparse {
        Sparse { ctx: ctx.clone(), rows, cols, format, nnz: val.len(), ptr, row, col, val: Some(val), expanded: OnceLock::new(), error: None }
    }

    fn poisoned(ctx: &Context, rows: usize, cols: usize, format: SparseFormat, error: Arc<LumaError>) -> Sparse {
        Sparse { ctx: ctx.clone(), rows, cols, format, nnz: 0, ptr: None, row: None, col: None, val: None, expanded: OnceLock::new(), error: Some(error) }
    }

    /// A matrix of `rows x cols` without stored entries, in `format`, on the default context.
    ///
    /// # Errors
    ///
    /// A zero dimension, or one that does not fit in `u32`, is an error.
    pub fn empty(rows: usize, cols: usize, format: SparseFormat) -> Result<Sparse> {
        if let Some(e) = check_dims(rows, cols) {
            return Err(e);
        }
        Ok(Self::empty_on(&default_context()?, rows, cols, format))
    }

    /// [`Sparse::empty`] on an explicit context. An invalid size gives a poisoned matrix
    /// ([`Sparse::error`]).
    #[track_caller]
    pub fn empty_on(ctx: &Context, rows: usize, cols: usize, format: SparseFormat) -> Sparse {
        let loc = Location::caller();
        if let Some(e) = check_dims(rows, cols) {
            return Self::poisoned(ctx, rows, cols, format, failure("Sparse::empty", loc, e));
        }
        let n = match format {
            SparseFormat::Csr => rows,
            SparseFormat::Csc => cols,
            SparseFormat::Coo => 0,
        };
        let ptr = if format == SparseFormat::Coo {
            None
        } else {
            match Array::<u32>::zeros_on(ctx, n + 1) {
                Ok(p) => Some(p),
                Err(e) => return Self::poisoned(ctx, rows, cols, format, failure("Sparse::empty", loc, e)),
            }
        };
        Sparse { ctx: ctx.clone(), rows, cols, format, nnz: 0, ptr, row: None, col: None, val: None, expanded: OnceLock::new(), error: None }
    }

    /// The non-zero entries of a dense matrix of shape `[cols, rows]` (`rows x cols`, one
    /// dimension of extent 1 for a vector; more than two axes are an error) in `format`. Entries
    /// equal to zero (`-0.0` too) are dropped, NaN is kept. CSR and COO list the entries in row
    /// order, CSC in column order, both ascending inside a row (column).
    ///
    /// The number of entries depends on the data, so this reads a count back and
    /// **synchronizes** once, as [`Array::filter`] does. Errors, including a poisoned `dense`,
    /// poison the matrix.
    #[track_caller]
    pub fn from_dense(dense: &Array<f32>, format: SparseFormat) -> Sparse {
        let loc = Location::caller();
        let ctx = dense.context();
        let d = dense.shape().dims();
        let (cols, rows) = (d[0], d[1]);
        if d[2] != 1 || d[3] != 1 {
            let e = LumaError::InvalidShape(format!("sparse of an array of shape {}: it must be a matrix", dense.shape()));
            return Self::poisoned(ctx, rows, cols, format, failure("sparse", loc, e));
        }
        match Self::dense_entries(dense, format, rows, cols) {
            Ok(s) => s,
            Err(e) => Self::poisoned(ctx, rows, cols, format, failure("sparse", loc, e)),
        }
    }

    fn dense_entries(dense: &Array<f32>, format: SparseFormat, rows: usize, cols: usize) -> Result<Sparse> {
        let ctx = dense.context();
        if let Some(e) = check_dims(rows, cols) {
            return Err(e);
        }
        // Compressed order: the major axis is what the pointer indexes, the minor one what is
        // stored per entry. The dense matrix is read along that order (a transpose for CSC).
        let (major, minor, source) = match format {
            SparseFormat::Csc => (cols, rows, dense.transpose()),
            _ => (rows, cols, dense.clone()),
        };
        // NaN is an entry (a NaN comparison is unspecified on some backends, so it is tested apart).
        let mask = source.ne(0.0f32) | source.is_nan();
        let Some(idx) = mask.nonzero()? else {
            return Ok(Self::empty_on(ctx, rows, cols, format));
        };
        let idx = idx.cache();
        let val = source.take(&idx).cache();
        let (id_major, id_minor) = ((&idx / minor as u32).cache(), (&idx % minor as u32).cache());
        let ptr = || -> Array<u32> {
            let counts = mask.sum_axis(0).reshape(major);
            match Array::<u32>::zeros_on(ctx, 1) {
                Ok(zero) => concat(0, [&zero, &counts.cumsum(0)]).cache(),
                Err(e) => Array::poisoned(ctx, Shape::from(major + 1), Arc::new(e)),
            }
        };
        Ok(match format {
            SparseFormat::Csr => Self::assemble(ctx, (rows, cols), format, Some(ptr()), None, Some(id_minor), val),
            SparseFormat::Csc => Self::assemble(ctx, (rows, cols), format, Some(ptr()), Some(id_minor), None, val),
            SparseFormat::Coo => Self::assemble(ctx, (rows, cols), format, None, Some(id_major), Some(id_minor), val),
        })
    }

    /// A matrix from `(row, column, value)` triplets in `format`: entry `k` is `values[k]` at
    /// `(row_idx[k], col_idx[k])`. The three arrays must have the same length (else the result is
    /// poisoned with [`LumaError::LengthMismatch`]); the order is free and duplicates are kept
    /// (they add up, see the [module documentation](self)). Into CSR (CSC) the entries are
    /// sorted by row (column) with a stable sort: entries of one row (column) keep their input
    /// order, which is not sorted by column (row) unless the input was.
    #[track_caller]
    pub fn from_triplets(rows: usize, cols: usize, row_idx: &Array<u32>, col_idx: &Array<u32>, values: &Array<f32>, format: SparseFormat) -> Sparse {
        let loc = Location::caller();
        let ctx = values.context();
        let n = values.len();
        let bad = if let Some(e) = check_dims(rows, cols) {
            Some(e)
        } else if row_idx.len() != n {
            Some(LumaError::LengthMismatch { expected: n, actual: row_idx.len() })
        } else if col_idx.len() != n {
            Some(LumaError::LengthMismatch { expected: n, actual: col_idx.len() })
        } else if !ctx.same_as(row_idx.context()) || !ctx.same_as(col_idx.context()) {
            Some(LumaError::ContextMismatch)
        } else {
            None
        };
        if let Some(e) = bad {
            return Self::poisoned(ctx, rows, cols, format, failure("Sparse::from_triplets", loc, e));
        }
        let (r, c, v) = (flat(row_idx), flat(col_idx), flat(values));
        match format {
            SparseFormat::Coo => Self::assemble(ctx, (rows, cols), format, None, Some(r), Some(c), v),
            SparseFormat::Csr => {
                let (ptr, col, val) = regroup(&r, &c, &v, rows);
                Self::assemble(ctx, (rows, cols), format, Some(ptr), None, Some(col), val)
            }
            SparseFormat::Csc => {
                let (ptr, row, val) = regroup(&c, &r, &v, cols);
                Self::assemble(ctx, (rows, cols), format, Some(ptr), Some(row), None, val)
            }
        }
    }

    /// A COO matrix from the arrays of its entries: [`Sparse::from_triplets`] with
    /// [`SparseFormat::Coo`].
    #[track_caller]
    pub fn from_coo(rows: usize, cols: usize, row_idx: &Array<u32>, col_idx: &Array<u32>, values: &Array<f32>) -> Sparse {
        Self::from_triplets(rows, cols, row_idx, col_idx, values, SparseFormat::Coo)
    }

    /// A CSR matrix from its arrays: `row_ptr` has `rows + 1` offsets (non-decreasing, the first 0
    /// and the last the number of entries, which this cannot check without reading them back),
    /// `col_idx` and `values` one element per entry. Wrong lengths poison the result with
    /// [`LumaError::LengthMismatch`]. The arrays are used as they are (no copy of evaluated ones).
    #[track_caller]
    pub fn from_csr(rows: usize, cols: usize, row_ptr: &Array<u32>, col_idx: &Array<u32>, values: &Array<f32>) -> Sparse {
        let loc = Location::caller();
        match Self::check_compressed(rows, cols, row_ptr, rows, col_idx, values) {
            Some(e) => Self::poisoned(values.context(), rows, cols, SparseFormat::Csr, failure("Sparse::from_csr", loc, e)),
            None => Self::assemble(values.context(), (rows, cols), SparseFormat::Csr, Some(flat(row_ptr)), None, Some(flat(col_idx)), flat(values)),
        }
    }

    /// A CSC matrix from its arrays: `col_ptr` has `cols + 1` offsets, `row_idx` and `values` one
    /// element per entry; otherwise as [`Sparse::from_csr`].
    #[track_caller]
    pub fn from_csc(rows: usize, cols: usize, col_ptr: &Array<u32>, row_idx: &Array<u32>, values: &Array<f32>) -> Sparse {
        let loc = Location::caller();
        match Self::check_compressed(rows, cols, col_ptr, cols, row_idx, values) {
            Some(e) => Self::poisoned(values.context(), rows, cols, SparseFormat::Csc, failure("Sparse::from_csc", loc, e)),
            None => Self::assemble(values.context(), (rows, cols), SparseFormat::Csc, Some(flat(col_ptr)), Some(flat(row_idx)), None, flat(values)),
        }
    }

    fn check_compressed(rows: usize, cols: usize, ptr: &Array<u32>, major: usize, idx: &Array<u32>, values: &Array<f32>) -> Option<LumaError> {
        if let Some(e) = check_dims(rows, cols) {
            return Some(e);
        }
        if ptr.len() != major + 1 {
            return Some(LumaError::LengthMismatch { expected: major + 1, actual: ptr.len() });
        }
        if idx.len() != values.len() {
            return Some(LumaError::LengthMismatch { expected: values.len(), actual: idx.len() });
        }
        let ctx = values.context();
        (!ctx.same_as(ptr.context()) || !ctx.same_as(idx.context())).then_some(LumaError::ContextMismatch)
    }

    // -- Introspection --

    /// Number of rows.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Number of columns.
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// The number of stored entries (duplicates count separately).
    pub fn nnz(&self) -> usize {
        self.nnz
    }

    /// The storage format.
    pub fn format(&self) -> SparseFormat {
        self.format
    }

    /// The [`Context`] the arrays live on.
    pub fn context(&self) -> &Context {
        &self.ctx
    }

    /// The error this matrix is poisoned with, if any (it names the operation that failed and
    /// where it was called from).
    pub fn error(&self) -> Option<&LumaError> {
        self.error.as_deref()
    }

    /// The values of the stored entries (`nnz` elements, in storage order); `None` if there are
    /// none. Sharing the matrix's buffer: no copy.
    pub fn values(&self) -> Option<Array<f32>> {
        self.val.clone()
    }

    /// ArrayFire's row array: CSR, the row pointer of `rows + 1` offsets; CSC and COO, the row
    /// of every entry (`None` without entries). See [`SparseInfo::row_idx`].
    pub fn row_idx(&self) -> Option<Array<u32>> {
        match self.format {
            SparseFormat::Csr => self.ptr.clone(),
            _ => self.row.clone(),
        }
    }

    /// ArrayFire's column array: CSC, the column pointer of `cols + 1` offsets; CSR and COO, the
    /// column of every entry (`None` without entries). See [`SparseInfo::col_idx`].
    pub fn col_idx(&self) -> Option<Array<u32>> {
        match self.format {
            SparseFormat::Csc => self.ptr.clone(),
            _ => self.col.clone(),
        }
    }

    /// The format, size and arrays of the matrix in one value (ArrayFire's `sparseGetInfo`).
    pub fn info(&self) -> SparseInfo<f32> {
        SparseInfo {
            format: self.format,
            rows: self.rows,
            cols: self.cols,
            nnz: self.nnz,
            values: self.values(),
            row_idx: self.row_idx(),
            col_idx: self.col_idx(),
        }
    }

    /// The shape of the dense form, `[cols, rows]`.
    fn dense_shape(&self) -> Shape {
        Shape::new([self.cols, self.rows, 1, 1])
    }

    /// A poisoned dense array of `shape` if this matrix is poisoned.
    fn dense_poison(&self, shape: Shape) -> Option<Array<f32>> {
        self.error.as_ref().map(|e| Array::poisoned(&self.ctx, shape, Arc::clone(e)))
    }

    /// The row of every entry of a CSR matrix, the column of every entry of a CSC matrix: the
    /// pointer expanded by a binary search per entry, computed once and kept. Only for a matrix
    /// with entries.
    fn expanded(&self) -> &Array<u32> {
        self.expanded.get_or_init(|| {
            let (ptr, nnz) = (self.ptr.as_ref().expect("compressed matrix"), self.nnz);
            let major = ptr.len() - 1;
            let ids = match Array::<u32>::range_on(&self.ctx, nnz, 0) {
                Ok(ids) => ids,
                Err(e) => return Array::poisoned(&self.ctx, Shape::from(nnz), Arc::new(e)),
            };
            ptr.slice(0, 1..=major).searchsorted(&ids, Side::Right).cache()
        })
    }

    /// The row and the column of every entry. Only for a matrix with entries.
    fn coords(&self) -> (Array<u32>, Array<u32>) {
        match self.format {
            SparseFormat::Csr => (self.expanded().clone(), self.col.clone().expect("entries")),
            SparseFormat::Csc => (self.row.clone().expect("entries"), self.expanded().clone()),
            SparseFormat::Coo => (self.row.clone().expect("entries"), self.col.clone().expect("entries")),
        }
    }

    /// The flat index of every entry in the dense form (`row * cols + col`).
    fn linear(&self) -> Array<u32> {
        let (r, c) = self.coords();
        &r * self.cols as u32 + &c
    }

    // -- Conversions --

    /// The same matrix in `format`. The same format is a clone (no work, buffers shared); other
    /// conversions are a stable sort of the entries by their new major index (CSR to CSC and back
    /// keep the entries of a row, a column, in order; COO to a compressed format as in
    /// [`Sparse::from_triplets`]) and take the time of one sort of `u32` keys with a gather of
    /// the indices and values. Conversions to COO move no data beyond expanding the pointer. A
    /// poisoned matrix converts to a poisoned matrix.
    pub fn to_format(&self, format: SparseFormat) -> Sparse {
        if self.error.is_some() || format == self.format {
            return self.clone();
        }
        if self.nnz == 0 {
            return Self::empty_on(&self.ctx, self.rows, self.cols, format);
        }
        let (ctx, rows, cols) = (&self.ctx, self.rows, self.cols);
        let val = self.val.as_ref().expect("entries");
        let (r, c) = self.coords();
        match format {
            SparseFormat::Coo => Self::assemble(ctx, (rows, cols), format, None, Some(r), Some(c), val.clone()),
            SparseFormat::Csr => {
                let (ptr, col, val) = regroup(&r, &c, val, rows);
                Self::assemble(ctx, (rows, cols), format, Some(ptr), None, Some(col), val)
            }
            SparseFormat::Csc => {
                let (ptr, row, val) = regroup(&c, &r, val, cols);
                Self::assemble(ctx, (rows, cols), format, Some(ptr), Some(row), None, val)
            }
        }
    }

    /// The dense matrix, of shape `[cols, rows]`: zeros with the entries added in (duplicates
    /// add up). Needs `rows * cols` to fit in `u32` like any array; a matrix that does not
    /// gives a poisoned array.
    #[track_caller]
    pub fn to_dense(&self) -> Array<f32> {
        let loc = Location::caller();
        let shape = self.dense_shape();
        if let Some(p) = self.dense_poison(shape) {
            return p;
        }
        let mut out = match Array::<f32>::zeros_on(&self.ctx, shape) {
            Ok(out) => out,
            Err(e) => return Array::poisoned(&self.ctx, shape, failure("Sparse::to_dense", loc, e)),
        };
        if self.nnz > 0 {
            let val = self.val.as_ref().expect("entries");
            if let Err(e) = out.put_add(&self.linear(), val) {
                out.poison(Arc::new(e));
            }
        }
        out
    }

    /// The transpose, a `cols x rows` matrix. Free for the compressed formats (a CSR matrix is
    /// the CSC form of its transpose: the arrays are shared and only the format changes) and for
    /// COO (rows and columns swap).
    pub fn transpose(&self) -> Sparse {
        let expanded = OnceLock::new();
        if let Some(e) = self.expanded.get() {
            let _ = expanded.set(e.clone());
        }
        let (format, row, col) = match self.format {
            SparseFormat::Csr => (SparseFormat::Csc, self.col.clone(), None),
            SparseFormat::Csc => (SparseFormat::Csr, None, self.row.clone()),
            SparseFormat::Coo => (SparseFormat::Coo, self.col.clone(), self.row.clone()),
        };
        Sparse {
            ctx: self.ctx.clone(),
            rows: self.cols,
            cols: self.rows,
            format,
            nnz: self.nnz,
            ptr: self.ptr.clone(),
            row,
            col,
            val: self.val.clone(),
            expanded,
            error: self.error.clone(),
        }
    }

    // -- Products --

    /// The matrix product `A x` of this `rows x cols` matrix with a dense `x`: a vector (shape
    /// `[cols]`, result `[rows]`) or a matrix of `c` columns (shape `[c, cols]`, result
    /// `[c, rows]`); see [`Sparse::matmul_t`] for `A^T x`. A shape that is neither poisons the
    /// result with [`LumaError::ShapeMismatch`]. `x` may be a pending expression (it is gathered
    /// from inside the kernel). See the [module documentation](self) for how each format runs.
    #[track_caller]
    pub fn matmul(&self, x: &Array<f32>) -> Array<f32> {
        self.product(x, false, "Sparse::matmul", Location::caller())
    }

    /// The product `A^T x` with the transpose of this matrix (no transposed copy is made):
    /// `x` is a vector of `rows` elements or a matrix of shape `[c, rows]`, the result has `cols`
    /// rows. Otherwise as [`Sparse::matmul`].
    #[track_caller]
    pub fn matmul_t(&self, x: &Array<f32>) -> Array<f32> {
        self.product(x, true, "Sparse::matmul_t", Location::caller())
    }

    fn product(&self, x: &Array<f32>, transposed: bool, op: &'static str, loc: &'static Location<'static>) -> Array<f32> {
        let (inner, outer) = if transposed { (self.rows, self.cols) } else { (self.cols, self.rows) };
        let xd = x.shape().dims();
        // A vector of `inner` elements, or `[c, inner]` columns.
        let (c, vector) = if xd[1..] == [1, 1, 1] && xd[0] == inner {
            (1, true)
        } else if xd[2..] == [1, 1] && xd[1] == inner {
            (xd[0], false)
        } else {
            let lhs = self.dense_shape().dims();
            let e = LumaError::ShapeMismatch { lhs, rhs: xd };
            return Array::poisoned(&self.ctx, Shape::from(outer), failure(op, loc, e));
        };
        let out_shape = if vector { Shape::from(outer) } else { Shape::new([c, outer, 1, 1]) };
        if let Some(p) = self.dense_poison(out_shape) {
            return p;
        }
        if !self.ctx.same_as(x.context()) {
            return Array::poisoned(&self.ctx, out_shape, failure(op, loc, LumaError::ContextMismatch));
        }
        if self.nnz == 0 {
            return Array::zeros_on(&self.ctx, out_shape).unwrap_or_else(|e| Array::poisoned(&self.ctx, out_shape, failure(op, loc, e)));
        }
        let val = self.val.as_ref().expect("entries");
        // The sum over a compressed pointer needs no atomics: CSR `A x`, CSC `A^T x`.
        let segmented = matches!((self.format, transposed), (SparseFormat::Csr, false) | (SparseFormat::Csc, true));
        if c == 1 {
            let y = if segmented {
                let minor = if self.format == SparseFormat::Csr { self.col.as_ref() } else { self.row.as_ref() };
                (val * x.reshape(inner).take(minor.expect("entries"))).sum_segments(self.ptr.as_ref().expect("compressed"))
            } else {
                let (r, cc) = self.coords();
                let (gather, scatter) = if transposed { (r, cc) } else { (cc, r) };
                (val * x.reshape(inner).take(&gather)).group_sum(&scatter, outer)
            };
            return y.reshape(out_shape);
        }
        // `c` columns: the `c` values of an entry are adjacent in `x` and in the result, so an
        // entry's products are one `[c]` run of the product array, scattered with atomics.
        let nnz = self.nnz;
        let (r, cc) = self.coords();
        let (gather, scatter) = if transposed { (r, cc) } else { (cc, r) };
        let j = match Array::<u32>::range_on(&self.ctx, [c, nnz], 0) {
            Ok(j) => j,
            Err(e) => return Array::poisoned(&self.ctx, out_shape, failure(op, loc, e)),
        };
        let wide = |ids: &Array<u32>| ids.reshape([1, nnz]).broadcast_to([c, nnz]) * c as u32 + &j;
        let prod = val.reshape([1, nnz]).broadcast_to([c, nnz]) * x.take(&wide(&gather));
        prod.group_sum(&wide(&scatter), outer * c).reshape(out_shape)
    }

    // -- Elementwise --

    /// The error of an operation with `dense` unless this matrix is valid and `dense` has the
    /// shape `[cols, rows]` of the dense form.
    fn dense_operand_error(&self, dense: &Array<f32>, op: &'static str, loc: &'static Location<'static>) -> Option<Arc<LumaError>> {
        if let Some(e) = &self.error {
            return Some(Arc::clone(e));
        }
        let shape = self.dense_shape();
        if dense.shape() != shape {
            let e = LumaError::ShapeMismatch { lhs: shape.dims(), rhs: dense.shape().dims() };
            return Some(failure(op, loc, e));
        }
        (!self.ctx.same_as(dense.context())).then(|| failure(op, loc, LumaError::ContextMismatch))
    }

    /// This matrix with `f` applied to its values: same pattern, new values array.
    fn map_values(&self, f: impl FnOnce(&Array<f32>) -> Array<f32>) -> Sparse {
        let mut s = self.clone();
        if let Some(v) = &self.val {
            s.val = Some(f(v).cache());
        }
        s
    }

    /// The sparse matrix with the entries multiplied by the dense matrix at the same positions,
    /// `A .* D` (`dense` has shape `[cols, rows]`): the pattern is that of this matrix (a zero
    /// in `dense` gives a stored zero; an infinite or NaN `dense` where this matrix has no
    /// entry does not matter, as the implied `0 * inf` is not computed). Duplicate entries
    /// multiply separately, which adds up to the product of their sum.
    #[track_caller]
    pub fn mul_dense(&self, dense: &Array<f32>) -> Sparse {
        self.elementwise_dense(dense, "Sparse::mul_dense", Location::caller(), |v, d| v * d)
    }

    /// The sparse matrix with the entries divided by the dense matrix at the same positions,
    /// `A ./ D`; like [`Sparse::mul_dense`].
    #[track_caller]
    pub fn div_dense(&self, dense: &Array<f32>) -> Sparse {
        self.elementwise_dense(dense, "Sparse::div_dense", Location::caller(), |v, d| v / d)
    }

    fn elementwise_dense(&self, dense: &Array<f32>, op: &'static str, loc: &'static Location<'static>, f: impl FnOnce(&Array<f32>, Array<f32>) -> Array<f32>) -> Sparse {
        if let Some(e) = self.dense_operand_error(dense, op, loc) {
            return Self::poisoned(&self.ctx, self.rows, self.cols, self.format, e);
        }
        if self.nnz == 0 {
            return self.clone();
        }
        let at = dense.take(&self.linear());
        self.map_values(|v| f(v, at))
    }

    /// `A * s`: the values multiplied by a scalar, same pattern.
    pub fn mul_scalar(&self, s: f32) -> Sparse {
        self.map_values(|v| v * s)
    }

    /// `A / s`: the values divided by a scalar, same pattern.
    pub fn div_scalar(&self, s: f32) -> Sparse {
        self.map_values(|v| v / s)
    }

    /// `-A`.
    pub fn neg(&self) -> Sparse {
        self.map_values(|v| -v)
    }

    /// `A + D` as a dense matrix (`dense` has shape `[cols, rows]`): a copy of `dense` with the
    /// entries added in (duplicates add up). `f32` sums of several entries into one position
    /// (duplicates) are accumulated atomically in no fixed order.
    #[track_caller]
    pub fn add_dense(&self, dense: &Array<f32>) -> Array<f32> {
        self.accumulate(dense, false, "Sparse::add_dense", Location::caller())
    }

    /// `A - D` as a dense matrix; see [`Sparse::add_dense`].
    #[track_caller]
    pub fn sub_dense(&self, dense: &Array<f32>) -> Array<f32> {
        self.accumulate(dense, true, "Sparse::sub_dense", Location::caller())
    }

    fn accumulate(&self, dense: &Array<f32>, negate_dense: bool, op: &'static str, loc: &'static Location<'static>) -> Array<f32> {
        if let Some(e) = self.dense_operand_error(dense, op, loc) {
            return Array::poisoned(&self.ctx, self.dense_shape(), e);
        }
        let mut out = if negate_dense { (-dense).cache() } else { dense.copy().cache() };
        if self.nnz > 0 {
            let val = self.val.as_ref().expect("entries");
            if let Err(e) = out.put_add(&self.linear(), val) {
                out.poison(Arc::new(e));
            }
        }
        out
    }

    /// `D - A` as a dense matrix; see [`Sparse::add_dense`].
    #[track_caller]
    pub fn rsub_dense(&self, dense: &Array<f32>) -> Array<f32> {
        let (op, loc) = ("Sparse::rsub_dense", Location::caller());
        if let Some(e) = self.dense_operand_error(dense, op, loc) {
            return Array::poisoned(&self.ctx, self.dense_shape(), e);
        }
        let mut out = dense.copy().cache();
        if self.nnz > 0 {
            let val = self.val.as_ref().expect("entries");
            if let Err(e) = out.put_add(&self.linear(), &-val) {
                out.poison(Arc::new(e));
            }
        }
        out
    }

    /// `A + s` as a dense matrix: every element, stored or not, plus the scalar.
    #[track_caller]
    pub fn add_scalar(&self, s: f32) -> Array<f32> {
        self.to_dense() + s
    }

    /// `A - s` as a dense matrix; see [`Sparse::add_scalar`].
    #[track_caller]
    pub fn sub_scalar(&self, s: f32) -> Array<f32> {
        self.to_dense() - s
    }
}

impl Mul<f32> for &Sparse<f32> {
    type Output = Sparse<f32>;

    /// [`Sparse::mul_scalar`].
    fn mul(self, s: f32) -> Sparse<f32> {
        self.mul_scalar(s)
    }
}

impl Div<f32> for &Sparse<f32> {
    type Output = Sparse<f32>;

    /// [`Sparse::div_scalar`].
    fn div(self, s: f32) -> Sparse<f32> {
        self.div_scalar(s)
    }
}

impl Neg for &Sparse<f32> {
    type Output = Sparse<f32>;

    /// `-A`, as [`Sparse::neg`].
    fn neg(self) -> Sparse<f32> {
        Sparse::neg(self)
    }
}

/// The sparse matrix of a dense one in `format`: [`Sparse::from_dense`] (ArrayFire's `sparse`).
#[track_caller]
pub fn sparse(dense: &Array<f32>, format: SparseFormat) -> Sparse<f32> {
    Sparse::from_dense(dense, format)
}
