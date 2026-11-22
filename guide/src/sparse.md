# Sparse matrices

`Sparse` is a sparse `rows x cols` matrix of `f32` in CSR, CSC or COO storage, with the
operations of ArrayFire's `sparse.h` and a sparse-dense `matmul`. It is built from luma arrays
(a few `u32` index arrays and the `f32` values), so everything runs on the device and is recorded
like any other operation; errors poison the result instead of panicking.

Dense matrices keep the convention of [the previous chapter](data.md): a dense `rows x cols`
matrix is an array of shape `[cols, rows]`, and a dense right-hand side with `c` columns has shape
`[c, cols]`.

## Building a matrix

```rust
use luma::*;

// [[1, 0, 2],
//  [0, 0, 3]]
let dense = Array::from_slice(&[1.0f32, 0.0, 2.0, 0.0, 0.0, 3.0], [3, 2])?;

// From a dense matrix (reads the entry count back: one synchronisation).
let a = sparse(&dense, SparseFormat::Csr);
assert_eq!((a.rows(), a.cols(), a.nnz()), (2, 3, 3));

// From (row, column, value) triplets, in any order; duplicates add up.
let rows = Array::from_slice(&[1u32, 0, 0], 3)?;
let cols = Array::from_slice(&[2u32, 2, 0], 3)?;
let vals = Array::from_slice(&[3.0f32, 2.0, 1.0], 3)?;
let b = Sparse::from_triplets(2, 3, &rows, &cols, &vals, SparseFormat::Csc);
assert_eq!(b.to_dense().to_vec()?, dense.to_vec()?);

// From the arrays of a CSR matrix.
let ptr = Array::from_slice(&[0u32, 2, 3], 3)?;
let col = Array::from_slice(&[0u32, 2, 2], 3)?;
let csr_vals = Array::from_slice(&[1.0f32, 2.0, 3.0], 3)?;
let c = Sparse::from_csr(2, 3, &ptr, &col, &csr_vals);
assert_eq!(c.to_dense().to_vec()?, dense.to_vec()?);
# Ok::<(), luma::LumaError>(())
```

`values()`, `row_idx()`, `col_idx()` and `info()` give the arrays back with ArrayFire's meaning:
the row array of CSR is the row pointer, the column array of CSC is the column pointer. luma has
no empty arrays, so they are `None` for a matrix without entries (`Sparse::empty`, or the sparse
form of an all-zero matrix).

## Conversions and the transpose

`to_format` converts between the three formats (a stable sort by the new major index, one `u32`
sort and a gather); `to_dense` scatters the entries into zeros. `transpose` is free: the transpose
of a CSR matrix is the CSC matrix with the same arrays, and COO swaps its two index arrays.

## Products

`matmul` multiplies by a vector (shape `[cols]`) or a dense matrix (`[c, cols]`); `matmul_t` uses
the transpose without building it.

```rust
use luma::*;

let dense = Array::from_slice(&[1.0f32, 0.0, 2.0, 0.0, 0.0, 3.0], [3, 2])?;
let a = sparse(&dense, SparseFormat::Csr);
let x = Array::from_slice(&[1.0f32, 10.0, 100.0], 3)?;
assert_eq!(a.matmul(&x).to_vec()?, vec![201.0, 300.0]);

let y = Array::from_slice(&[1.0f32, 2.0], 2)?;
assert_eq!(a.matmul_t(&y).to_vec()?, vec![1.0, 0.0, 8.0]);
# Ok::<(), luma::LumaError>(())
```

CSR `A x` and CSC `A^T x` (the product whose output runs along the compressed axis) run on kernels
written for the format: for a vector, a thread (or a few lanes) per row when the rows are short, a
gather and a segmented sum over the pointer (`sum_segments`) when some are long; for a dense matrix
one invocation per output element walks its row, reading `c` adjacent elements of `x` together,
and the few very long rows of a power-law graph are cut in pieces summed by a workgroup each. No
atomics are used, so the results are the same on every run. The first product of any other
combination (CSC `A x`, CSR `A^T x`, COO) scatters with atomic adds, so its `f32` sums can differ in
the last bits between runs; the second converts the matrix once to the form that suits the product
and keeps the copy (one more copy of the entries for the life of the matrix), and results are then
reproducible. The first product of a matrix reads its row lengths back (one synchronization). A
pending expression as `x` is computed inside the kernel only for the segmented sum; the row kernels
evaluate it first.

## Elementwise operations

`mul_dense` and `div_dense` keep the pattern of the sparse matrix and give a sparse result;
`mul_scalar`, `div_scalar`, `neg` (and the operators `*`, `/`, `-` with `f32`) do too. Sums and
differences with a dense matrix (`add_dense`, `sub_dense`, `rsub_dense`) and with a scalar
(`add_scalar`) fill the zeros, so their results are dense arrays.
