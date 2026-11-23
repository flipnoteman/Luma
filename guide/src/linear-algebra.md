# Dense linear algebra

Factorisations and solvers for dense `f32` matrices, on every backend: Cholesky, triangular and
symmetric positive definite solves, LU with partial pivoting, general solves, inverse and
determinant, Householder QR and least squares. They are ordinary lazy operations: recorded, not
waited for; the factors are arrays that stay on the device.

## Layout

A matrix is `[cols, rows]` like the operands of `matmul` (row `i` is contiguous, element `(i, j)`
at `j + cols * i`), and a batch of matrices is a third axis, `[cols, rows, batch]`. A right-hand
side of `k` columns for an `n`-row system is `[k, n]`; a plain `[n]` array is one vector and the
solution has the shape of the right-hand side.

```rust
use luma::*;

// [[4, 2], [2, 5]] x = [2, 3]  =>  x = [0.25, 0.5]
let a = Array::from_slice(&[4.0f32, 2.0, 2.0, 5.0], [2, 2])?;
let b = Array::from_slice(&[2.0f32, 3.0], 2)?;

let l = a.cholesky();                      // A = L Lᵀ, L lower triangular
assert_eq!(l.to_vec()?, vec![2.0, 0.0, 1.0, 2.0]);
let x = a.solve_spd(&b);                   // Cholesky + two triangular solves
let y = a.solve(&b);                       // LU with partial pivoting
for (u, v) in x.to_vec()?.iter().zip(y.to_vec()?) {
    assert!((u - v).abs() < 1e-5);
}
assert!((a.det().item()? - 16.0).abs() < 1e-4);

let f = a.lu();                            // P A = L U, packed, with the permutation
let back = f.solve(&b);                    // reuse the factors for more right-hand sides
assert!((back.to_vec()?[0] - 0.25).abs() < 1e-5);
# Ok::<(), luma::LumaError>(())
```

| Method | Result |
| --- | --- |
| `cholesky()`, `cholesky_upper()` | the factor `L` (`A = L Lᵀ`) or `U` (`A = Uᵀ U`) of an SPD matrix; reads one triangle |
| `solve_spd(&b)`, `cholesky_solve(&b)` | `A X = B` for SPD `A` (from `A`, or from its factor) |
| `solve_triangular(&b, Triangle, transpose, unit_diag)` | `op(A) X = B` for a triangular `A` |
| `lu()` | `Lu { lu, perm }`: `P A = L U`, with `lower()`, `upper()`, `solve`, `inverse`, `det` |
| `solve(&b)` | square: LU; tall: least squares; wide: minimum norm |
| `inverse()`, `det()`, `slogdet()` | inverse; determinant; sign and log of the absolute determinant (no overflow) |
| `qr()` | `Qr { qr, tau, t }` with `r()`, `q_thin()`, `q()`, `apply_q`, `apply_qt`, `lstsq` |
| `lstsq(&b)` | least squares by QR |

## What runs where

* Small matrices and every matrix of a batch are factored by one workgroup each, in one launch:
  a batch of 4096 matrices of 16 x 16 costs one launch, where a library without batched LAPACK
  loops over calls.
* A big matrix (more than 64 rows) is blocked. LU and QR recurse on the columns: the leaves are
  panels of up to 32 columns and every other flop is a full-`f32` `matmul_f32` of the halves
  (never TF32: a factorisation's accuracy depends on it).
  * LU factors a panel by a tournament (CALU) instead of one workgroup streaming the whole panel
    through one multiprocessor: workgroups pick pivot rows in chunks of up to 512 rows, one
    workgroup plays the winners off against each other, and a last kernel gives every row its
    multipliers. A panel of up to 512 rows is one chunk and pivots exactly as partial pivoting
    does. A panel of more rows has multipliers that can exceed 1 (mildly; the tests bound them by
    8 above 64 rows), so the pivoting is not the classical one there, though the result is the
    same on every backend. Triangular solves inside the recursion use register-resident
    substitution leaves of 64 rows.
  * QR factors a panel by a two level tall-skinny QR (TSQR): blocks of rows are factored by one
    workgroup each, the stack of their `R`s is factored the same way, and the compact WY form is
    rebuilt from the explicit `Q`. A panel of up to 512 rows is one launch. Panels too tall for the
    tree (over 8192 rows with 32 columns), devices with workgroups of fewer than 64 invocations
    and, for the tree, less than 30 KB of workgroup memory keep the older kernel in which one
    workgroup streams the panel; `LUMA_QR_TSQR=0` selects it everywhere.
  * Cholesky is left-looking in blocks of 128 columns (64 or fewer where workgroup memory is
    small), all writing into one output: one product of the finished columns per 512 columns plus a
    small correction per block, one workgroup that factors the diagonal block and one launch of
    warps that solve the rows below, about 100 launches at 4096 rows. `LUMA_CHOL_NB` and
    `LUMA_CHOL_WIDE` pick the block sizes.
  * The update products of Cholesky and of the blocked triangular solves are exact `f32` too;
    `LUMA_LINALG_TF32=1` runs them on tensor cores (CUDA, compute capability 8.0 and up), with a
    backward error of about 1e-3 of the entries of the updates instead of 1e-7. It is off by
    default and is independent of `ContextOptions::tf32_matmul`, which governs `matmul` only.
* A triangular solve with up to 32 right-hand sides (`solve_triangular`, and the two solves inside
  `solve` and `solve_spd`) of a triangle of more than 64 rows is one launch on CUDA devices: one
  workgroup per block of 64 rows takes its turn from an atomic ticket, waits for the solved blocks
  before it (the words they publish are the complement of the bits, 0 meaning "not there yet": no flags
  or fences), and solves its own block. The solves inside `solve` and `solve_spd` take the diagonal
  blocks as explicit inverses from 768 rows (computed while the workgroup waits; the blocks are
  well-conditioned triangles of a factorisation, a Cholesky factor or a unit `L` with bounded
  multipliers); `solve_triangular` of an arbitrary triangle substitutes. More than 8 right-hand
  sides are split over workgroups of their own. More than 32 right-hand sides, and every backend
  that does not order memory across workgroups (wgpu, the CPU), solve in blocks of 128 rows, one
  product plus one launch each. `LUMA_TRSV=0` turns the one-launch solve off.
* The kernels of the factorisations are cached by the context that built them, not process-wide, so
  a dropped context releases them (and its device).
* Numerical failure never poisons or raises: a matrix that is not positive definite leaves `NaN`
  on the diagonal of its Cholesky factor, a singular matrix a zero on the diagonal of `U` (the
  solve then gives `Inf`/`NaN`, the determinant 0). Shape errors poison the result as usual.

## SVD, pseudo-inverse, rank and the symmetric eigenproblem

```rust
use luma::*;

let a = Array::from_slice(&[3.0f32, 0.0, 0.0, -2.0], [2, 2])?;
let svd = a.svd();                         // A = U diag(s) Vᵀ, s descending
let s = svd.s.to_vec()?;
assert!((s[0] - 3.0).abs() < 1e-5 && (s[1] - 2.0).abs() < 1e-5);
assert_eq!(a.rank().to_vec()?, vec![2]);   // singular values above max(m, n) eps s_max

let sym = Array::from_slice(&[2.0f32, 1.0, 1.0, 2.0], [2, 2])?;
let e = sym.eigh();                        // reads the lower triangle; values ascending
let l = e.values.to_vec()?;
assert!((l[0] - 1.0).abs() < 1e-5 && (l[1] - 3.0).abs() < 1e-5);
# Ok::<(), luma::LumaError>(())
```

| Method | Result |
| --- | --- |
| `svd()` | `Svd { u, s, vt }`, thin: `u` `[k, m]`, `s` `[k, 1]`, `vt` `[n, k]`, `k = min(m, n)`, any shape |
| `singular_values()` | the values alone (no `V`, no vectors written) |
| `pinv()`, `pinv_tol(tol)` | Moore-Penrose pseudo-inverse `[m, n]`; values `<= tol s_max` count as zero (default `max(m, n) eps`) |
| `rank()`, `rank_tol(tol)` | numerical rank as a `u32` `[1, 1]`: the singular values above `tol s_max` (default `max(m, n) eps`) |
| `cond()` | `s_max / s_min` |
| `eigh()`, `eigvalsh()` | `Eigh { values, vectors }` of a symmetric matrix (lower triangle read), ascending; the values alone |

The method is one-sided Jacobi: columns of `A` are rotated in pairs (the pairs of a round-robin
round are disjoint, so a round is parallel) until they are orthogonal; the singular values are
their norms. A matrix with its `V` that fits workgroup memory (up to 64 x 64 on devices with 32 KiB
of it, 32 x 32 with 16 KiB) and every matrix of a batch is decomposed by one workgroup, in one
launch, stopping inside the kernel when no rotation is left. A bigger matrix lives in global
memory. From 128 columns on it is decomposed by block Jacobi: the columns are cut in blocks of 16
and a round pairs the blocks, in three launches (the Gram matrix of every block pair, a small
two-sided Jacobi eigenproblem per pair whose rotation is made orthogonal with one Newton-Schulz
step, and the two thin products that apply it to the columns and to `V`); below 128 columns one
launch per round-robin round rotates `n / 2` pairs at once. From 1024 columns the decomposition
goes through `A = Q R` and the iteration works on `Rᵀ`, which converges in fewer sweeps
(`LUMA_SVD_PRECOND=off` turns it off, a number replaces the 1024). The sweeps continue until one
rotates nothing (the host looks every few sweeps, so a big `svd()` waits for the device, unlike
the other operations). `eigh` is the SVD of `A + ||A||_F I` read from the lower triangle:
positive semi-definite, so `U = V` holds the eigenvectors and `lambda = sigma - ||A||_F`.

Accuracy in `f32`: `||A - U diag(s) Vᵀ||_F / ||A||_F` and the singular values are within a few
`eps` of `s_max` (absolute), the orthogonality of `U` and `V` degrades as about `n sqrt(n) eps`;
the eigenvalues of `eigh` carry an absolute error of a few `eps ||A||_F`. A zero singular value
gets a left vector completed to an orthonormal set. `NaN` input gives `NaN` values and never
hangs (the sweeps are bounded). For a 4096 x 4096 matrix the backward error is 8e-6 and the loss of
orthogonality 1.8e-5 (RTX 3090).

## Rank of a big matrix

`rank` of a matrix that fits workgroup memory is decided by its singular values, lazily, on the
device. A bigger matrix makes `rank` wait for the device, and the answer is always the one of the
singular values (the number above `tol s_max`), reached in one of two ways:

* When the cut is well above the rounding errors of a QR factorisation (a matrix of full rank that
  is not nearly singular, or a looser tolerance given with `rank_tol`), two unpivoted QR
  factorisations bound the number of singular values above the cut from both sides (the rows of
  `R` that can be cut from the bottom, and the smallest singular value of the rows kept, from the
  norm of the inverse of their triangle) and give the rank without any singular value. `rank_tol(4e-3)`
  of a 4096 x 4096 matrix of rank 2048 takes 185 ms on an RTX 3090 (a QR factorisation plus the certificate, the
  factorisation being most of it), which is 0.20 in ArrayFire to luma time ratio: ArrayFire's `rank`
  is faster there.
* Otherwise the singular values decide: at the default tolerance `max(m, n) eps` the rounding
  errors of an unpivoted QR factorisation of a rank-deficient matrix are as big as the cut, and a
  matrix whose rank is set by singular values close to the cut cannot be certified either. Only
  columns above `tol s_max / (2 sqrt(n))` are rotated any more, so a low-rank matrix needs the
  sweeps of its rank. The values-only decomposition takes 1.43 s at 4096 columns and 68 ms at
  1024 (RTX 3090; ArrayFire's QR based `rank` with an absolute tolerance takes 37 ms and 4.6 ms).

## Measured against ArrayFire

RTX 3090, ArrayFire 3.10.1 (cuSOLVER), interleaved, from the 0.5.2 changelog. A batch of small
matrices (Cholesky, LU, solve, inverse, determinant, QR and SVD, 2048 matrices of 8 x 8 up to 1024
of 32 x 32) is one launch against a loop of ArrayFire calls: 340 to 1900 times faster. A single big
matrix is a closer race, in ms, luma (ArrayFire):

* `cholesky`: 0.12 at 256 rows, 0.66 at 1024, 4.6 at 4096 (3.2).
* `lu`: 17.4 at 4096 (66 before the tournament panels); the ArrayFire to luma time ratio is 0.85
  at 256, 0.80 at 1024 and 0.83 at 4096 (below 1: cuSOLVER is faster).
* `qr`: 0.45 (0.71) at 256, 3.6 (3.9) at 1024, 31.3 (26.0) at 4096; `lstsq` of 2048 x 1024 takes
  6.3 (6.0).
* `svd`: 54 (90) at 1024 x 1024, 1570 (1860) at 4096 x 4096; `singular_values` 42 at 1024 and
  1120 at 4096.
* `solve_triangular`, 4096 rows and 16 right-hand sides: 0.630 (0.981); a single right-hand side
  with 4096 rows 0.137 (ArrayFire's `trsm` about 0.2).

What is left of the gap to cuSOLVER is in the Cholesky and LU diagonal and panel kernels, which are
latency bound.
